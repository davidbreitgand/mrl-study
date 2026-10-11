#!/usr/bin/env python3
"""Turn a low-concurrency service-time probe into calibration.json.

Reads a probe run's journal, computes the mean per-request service
time μ under near-zero queueing (target_concurrent=1), then derives:

    study_lambda = target_rho · c_effective / mu_service

so subsequent open-loop cells drive the fleet at ρ ≈ target_rho by
construction. This is the honest calibration form of the ρ = λ/(cμ)
identity: measure μ, know c, pick λ.

WHY THIS SHAPE (not "measure completions/sec at some candidate λ"):
A completions-per-second measurement equals arrivals-per-second at
any steady state (conservation of flow), so it can't distinguish
"fleet at its ceiling" from "fleet under-driven by the driver." μ
measured under target_concurrent=1 IS the per-request service time,
independent of arrival rate.

Reads:
  Journal files under --results, matching pattern journal/**/*.jsonl.

Writes:
  {
    "probe_rows_completed":    <int>,
    "warmup_dropped":          <int>,
    "mean_service_seconds":    <float>,   # μ
    "median_service_seconds":  <float>,
    "p95_service_seconds":     <float>,
    "p99_service_seconds":     <float>,
    "c_effective":             <int>,     # sum of max_num_seqs across spokes
    "target_rho":              <float>,
    "study_lambda":            <float>,   # target_rho · c / μ, arrivals/sec
    "calibrated_at":           "<UTC ISO>"
  }

Usage:
  scripts/compute_lambda.py \\
      --results /tmp/mrl-calibration-results/ \\
      --c-effective 16 \\
      --target-rho 0.98 \\
      --warmup 10 \\
      --output local_results/calibration.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import statistics
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=pathlib.Path, required=True,
                    help="probe results directory (contains journal/**/*.jsonl).")
    ap.add_argument("--c-effective", type=int, required=True,
                    help="fleet's effective concurrency, sum of max_num_seqs "
                         "across running spokes. For this study: 2 spokes × 8 = 16.")
    ap.add_argument("--target-rho", type=float, default=0.98,
                    help="target offered load (0.98 matches the sim).")
    ap.add_argument("--warmup", type=int, default=10,
                    help="early rows to drop from the mean before computing μ. "
                         "10 covers vLLM's prefill-cache-warm-up on the first "
                         "few requests.")
    ap.add_argument("--output", type=pathlib.Path, required=True,
                    help="path to write calibration.json.")
    args = ap.parse_args()

    # Collect elapsed_seconds from every journal row.
    journals = sorted(args.results.glob("**/*.jsonl"))
    if not journals:
        print(f"ERROR: no *.jsonl files under {args.results}", file=sys.stderr)
        return 1

    elapsed_seconds: list[float] = []
    for jpath in journals:
        # progress.jsonl carries per-arrival dispatch records, not
        # completions; skip it.
        if jpath.name == "progress.jsonl":
            continue
        with jpath.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                extra = row.get("extra_fields", {}) or {}
                e = extra.get("elapsed_seconds")
                if e is None:
                    continue
                elapsed_seconds.append(float(e))

    if len(elapsed_seconds) <= args.warmup + 10:
        print(f"ERROR: only {len(elapsed_seconds)} completions found with "
              f"elapsed_seconds; need at least {args.warmup + 10} for a "
              f"reliable mean.", file=sys.stderr)
        return 1

    elapsed_seconds.sort()  # sorting IS lossy for warmup semantics — see below.

    # WARMUP HANDLING: we want to drop the FIRST N rows to skip vLLM's
    # cold-start latency. `elapsed_seconds` values themselves have no
    # ordering — a fast late request looks identical to a fast early
    # request. To drop by arrival order we'd need arrival_ts_seconds.
    # For this probe (closed-loop target_concurrent=1) all early rows
    # tend to be slower (prefill cache cold), so sorting ascending and
    # dropping the SMALLEST N is wrong. Sorting descending and dropping
    # the LARGEST N is also wrong (would drop legitimate tail).
    # Right thing: re-read with arrival_ts_seconds and drop by arrival
    # order. Redoing that pass:
    elapsed_by_arrival: list[tuple[float, float]] = []
    for jpath in journals:
        if jpath.name == "progress.jsonl":
            continue
        with jpath.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                extra = row.get("extra_fields", {}) or {}
                arrival = extra.get("arrival_ts_seconds")
                e = extra.get("elapsed_seconds")
                if arrival is None or e is None:
                    continue
                elapsed_by_arrival.append((float(arrival), float(e)))

    elapsed_by_arrival.sort(key=lambda t: t[0])
    steady = [e for _, e in elapsed_by_arrival[args.warmup:]]
    if not steady:
        print(f"ERROR: {len(elapsed_by_arrival)} rows found but all "
              f"dropped by warmup={args.warmup}.", file=sys.stderr)
        return 1

    mu = statistics.fmean(steady)
    steady_sorted = sorted(steady)
    n = len(steady_sorted)

    def q(p: float) -> float:
        if n == 1:
            return steady_sorted[0]
        rank = p * (n - 1)
        lo = int(rank)
        hi = min(lo + 1, n - 1)
        frac = rank - lo
        return steady_sorted[lo] * (1 - frac) + steady_sorted[hi] * frac

    study_lambda = args.target_rho * args.c_effective / mu

    out = {
        "probe_rows_completed":   len(elapsed_by_arrival),
        "warmup_dropped":         args.warmup,
        "mean_service_seconds":   mu,
        "median_service_seconds": q(0.50),
        "p95_service_seconds":    q(0.95),
        "p99_service_seconds":    q(0.99),
        "c_effective":            args.c_effective,
        "target_rho":             args.target_rho,
        "study_lambda":           study_lambda,
        "calibrated_at":          datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"[compute_lambda] wrote {args.output}", file=sys.stderr)
    print(json.dumps(out, indent=2))

    # Sanity warnings, but not fatal.
    if mu <= 0:
        print("\nERROR: mean service time is non-positive. The journal is "
              "malformed.", file=sys.stderr)
        return 1
    p99_over_mean = q(0.99) / mu if mu > 0 else float("inf")
    if p99_over_mean > 10:
        print(f"\nWARN: p99/mean = {p99_over_mean:.1f}. Service time is "
              f"very heavy-tailed. ρ = λ/(cμ) uses the MEAN; that's still "
              f"the right calibration for offered load. But the fleet will "
              f"see p99 waits far larger than mu at ρ = {args.target_rho}. "
              f"If p99/mean is unexpectedly high, verify the probe wasn't "
              f"contaminated by cold-start rows — try --warmup 20 or 30.",
              file=sys.stderr)
    if n < 50:
        print(f"\nWARN: only {n} steady-state rows in the mean; μ may be "
              f"noisy. Consider a larger --rows on the probe.",
              file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
