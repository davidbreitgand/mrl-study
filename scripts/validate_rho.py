#!/usr/bin/env python3
"""Validate that the offered load ρ landed near target_rho in a cell.

Post-cell diagnostic: reads a completed cell's journal + the study's
calibration.json, and reports the observed ρ against the target.

Two independent estimates of ρ, each derived directly from the journal:

  ρ_completion = arrivals_per_sec / (c_effective / mean_elapsed_after_filter)
                 = λ_observed / (c · μ̂)
  ρ_littles    = median(in_flight_at_arrival) / c_effective
                 [Little's Law approximation: E[N] = λ · E[T] = ρ · c]

Both should land near the calibration's target_rho (0.98). If they
disagree by more than a few percent, the trace-derived μ_TP2 was off
and the study's λ needs adjustment before the next cell.

Also reports:
  - the observed arrival rate (arrivals per second in steady state)
  - the observed spoke split (fraction of rows on each spoke)

Usage:
  scripts/validate_rho.py \\
      --cell local_results/poisson/ \\
      --calibration local_results/calibration.json \\
      [--warmup 30]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from collections import Counter


def iter_journal_rows(cell_dir: pathlib.Path):
    """Yield each row from every *.jsonl under cell_dir, skipping
    progress.jsonl and smoke/."""
    for jpath in sorted(cell_dir.glob("**/*.jsonl")):
        if jpath.name == "progress.jsonl":
            continue
        if "smoke/" in str(jpath):
            continue
        with jpath.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", type=pathlib.Path, required=True,
                    help="cell directory (e.g. local_results/poisson/).")
    ap.add_argument("--calibration", type=pathlib.Path, required=True,
                    help="calibration.json from compute_lambda_from_trace.py.")
    ap.add_argument("--warmup", type=int, default=30,
                    help="early rows to drop before computing steady-state stats.")
    args = ap.parse_args()

    if not args.cell.exists():
        print(f"ERROR: cell {args.cell} not found", file=sys.stderr)
        return 1
    if not args.calibration.exists():
        print(f"ERROR: calibration {args.calibration} not found", file=sys.stderr)
        return 1

    with args.calibration.open() as f:
        calib = json.load(f)
    target_rho = float(calib["target_rho"])
    study_lambda = float(calib["study_lambda"])
    c_effective = int(calib["c_effective"])
    mu_calib_TP2 = float(calib["mu_TP2_seconds"])

    # Collect rows.
    rows: list[tuple[float, float, int, str]] = []
    for r in iter_journal_rows(args.cell):
        extra = r.get("extra_fields", {}) or {}
        arrival = extra.get("arrival_ts_seconds")
        elapsed = extra.get("elapsed_seconds")
        in_flight = extra.get("in_flight_at_arrival")
        served = extra.get("served_endpoint", "unknown")
        if arrival is None or elapsed is None:
            continue
        rows.append((float(arrival),
                     float(elapsed),
                     int(in_flight) if in_flight is not None else 0,
                     str(served)))

    if len(rows) < args.warmup + 100:
        print(f"ERROR: only {len(rows)} rows; need at least {args.warmup + 100}.",
              file=sys.stderr)
        return 1

    rows.sort(key=lambda t: t[0])
    steady = rows[args.warmup:]

    # 1. Observed arrival rate = arrivals per second at steady state.
    #    Since arrivals are open-loop-staged, this SHOULD equal
    #    study_lambda in expectation.
    steady_arrivals = [a for a, _, _, _ in steady]
    duration = steady_arrivals[-1] - steady_arrivals[0]
    lambda_observed = (len(steady_arrivals) - 1) / duration if duration > 0 else 0.0

    # 2. Observed μ (any spoke) — the asymmetric fleet has two μ's, so
    #    this is a fleet-weighted average. Compare to expected
    #    fleet-weighted μ from the calibration.
    #    Under λ = target_rho · (c/μ_TP2 + c/μ_TP4):
    #      spoke-a served fraction = (c/μ_TP2) / total_cap  (slower ⇒ fewer)
    #      spoke-b served fraction = (c/μ_TP4) / total_cap  (faster ⇒ more)
    #    Fleet-weighted mean elapsed under ρ→1 approaches
    #      (frac_a · μ_TP2 + frac_b · μ_TP4)
    #    but under moderate ρ (0.98) is close to that if queueing
    #    contributions cancel across spokes.
    steady_elapsed = [e for _, e, _, _ in steady]
    mean_elapsed = statistics.fmean(steady_elapsed)

    # 3. ρ estimate via Little's Law: E[N] = ρ · c (approximately, in
    #    an M/M/c-like regime). Use median in_flight_at_arrival as a
    #    robust E[N] estimate.
    steady_inflight = [inf for _, _, inf, _ in steady]
    steady_inflight_sorted = sorted(steady_inflight)
    median_inflight = steady_inflight_sorted[len(steady_inflight_sorted)//2]
    rho_littles = median_inflight / c_effective if c_effective else 0.0

    # 4. ρ estimate via completion-rate: what fraction of the fleet's
    #    theoretical throughput did we actually consume? This uses the
    #    trace-derived μ from calibration.json, so it's really asking
    #    "was our μ estimate right?"
    expected_lambda_at_rho_1 = c_effective / mu_calib_TP2  # WRONG for asymmetric fleet
    # Better: use the calibration's total_cap directly.
    total_cap_new = float(calib.get("total_cap_new_per_sec",
                                    c_effective / mu_calib_TP2))
    rho_completion = lambda_observed / total_cap_new if total_cap_new > 0 else 0.0

    # 5. Spoke split.
    spoke_counts = Counter(s for _, _, _, s in steady)
    total_steady = sum(spoke_counts.values())
    spoke_share = {s: c / total_steady for s, c in spoke_counts.items()}

    out = {
        "cell":                       str(args.cell),
        "warmup_dropped":             args.warmup,
        "rows_used":                  len(steady),
        "duration_seconds":           duration,
        "target_rho":                 target_rho,
        "study_lambda":               study_lambda,
        "lambda_observed_per_sec":    lambda_observed,
        "lambda_observed_over_target": lambda_observed / study_lambda if study_lambda else 0.0,
        "mean_elapsed_seconds":       mean_elapsed,
        "median_inflight_at_arrival": median_inflight,
        "p95_inflight_at_arrival":    steady_inflight_sorted[int(0.95 * len(steady_inflight_sorted))],
        "max_inflight_at_arrival":    steady_inflight_sorted[-1],
        "c_effective":                c_effective,
        "rho_littles":                rho_littles,
        "rho_completion":             rho_completion,
        "spoke_share":                spoke_share,
    }
    print(json.dumps(out, indent=2))

    # Verdict.
    lambda_ratio = lambda_observed / study_lambda if study_lambda else 0.0
    lambda_delta_pct = abs(lambda_ratio - 1) * 100
    print(f"\n=== Verdict ===", file=sys.stderr)
    print(f"target_rho = {target_rho:.3f}", file=sys.stderr)
    print(f"lambda: observed={lambda_observed:.4f}/sec, "
          f"target={study_lambda:.4f}/sec, ratio={lambda_ratio:.3f} "
          f"({'OK' if lambda_delta_pct < 5 else 'DRIFT'})", file=sys.stderr)
    print(f"rho estimates: littles={rho_littles:.3f}, "
          f"completion={rho_completion:.3f}", file=sys.stderr)

    # A very rough judgment: if BOTH ρ estimates are in [0.90, 1.05],
    # we're at operationally-target load. Outside that band, worth a
    # closer look.
    verdict = "OK"
    reasons: list[str] = []
    if not (0.90 <= rho_littles <= 1.05):
        verdict = "DRIFT"
        reasons.append(f"rho_littles={rho_littles:.3f} outside [0.90, 1.05]")
    if not (0.90 <= rho_completion <= 1.05):
        verdict = "DRIFT"
        reasons.append(f"rho_completion={rho_completion:.3f} outside [0.90, 1.05]")
    if lambda_delta_pct > 5:
        verdict = "DRIFT"
        reasons.append(f"observed λ off by {lambda_delta_pct:.1f}%")

    print(f"\nverdict: {verdict}", file=sys.stderr)
    if reasons:
        for r in reasons:
            print(f"  - {r}", file=sys.stderr)
        print("\nIf DRIFT: μ_TP2 in the trace was off; re-derive with a "
              "different --max-inflight-for-mu, or run the closed-loop "
              "calibrate-lambda.sh probe for an honest measurement.",
              file=sys.stderr)
    return 0 if verdict == "OK" else 2


if __name__ == "__main__":
    sys.exit(main())
