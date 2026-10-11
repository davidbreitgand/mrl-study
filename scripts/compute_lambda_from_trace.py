#!/usr/bin/env python3
"""Compute study_lambda from a pre-existing (no-queueing) journal trace.

Alternative to `calibrate-lambda.sh` that saves cluster GPU-hours by
reusing an existing trace. Works when:

  - The trace was captured under CLOSED-LOOP `target_concurrent` well
    below the fleet's c_effective, so queueing was negligible and
    elapsed_seconds is a good estimate of μ_service.
  - The trace's spokes were TP=2 (matching the new fleet's spoke-a).
  - The new fleet's TP=4 spoke can be modeled by scaling μ_TP2 by a
    known TP-speedup factor.

Analytic derivation of λ_new for the mrl_study's asymmetric fleet
(spoke-a TP=2, spoke-b TP=4):

    total_cap_new = c_per_spoke / μ_TP2  +  c_per_spoke / μ_TP4
                  = c_per_spoke / μ_TP2  +  c_per_spoke · TP_speedup / μ_TP2
                  = c_per_spoke · (1 + TP_speedup) / μ_TP2

    study_lambda  = target_rho · total_cap_new

μ_TP2 is measured from the trace. TP_speedup and c_per_spoke are
inputs.

Provenance is preserved in calibration.json: the trace path, the row
count, the warmup, and every derived value.

Usage:
  scripts/compute_lambda_from_trace.py \\
      --trace /path/to/consolidated/totrandom-shuffled_4834-gemma-4-26b-a4b.json \\
      --c-per-spoke 8 \\
      --tp-speedup 1.6 \\
      --target-rho 0.98 \\
      --warmup 30 \\
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
    ap.add_argument("--trace", type=pathlib.Path, required=True,
                    help="path to a consolidated/*.json trace file.")
    ap.add_argument("--c-per-spoke", type=int, default=8,
                    help="vLLM max_num_seqs per spoke (default 8, matches this study).")
    ap.add_argument("--tp-speedup", type=float, default=1.6,
                    help="μ_TP2 / μ_TP4 speedup ratio (default 1.6).")
    ap.add_argument("--target-rho", type=float, default=0.98,
                    help="target offered load (0.98 matches the sim).")
    ap.add_argument("--warmup", type=int, default=30,
                    help="early rows (by arrival order) to drop before "
                         "computing μ. 30 covers vLLM's cold-start.")
    ap.add_argument("--max-inflight-for-mu", type=int, default=None,
                    help="filter rows to only those dispatched when "
                         "in_flight_at_arrival <= this value. Defaults "
                         "to c_per_spoke (=8), which keeps only rows "
                         "where at least one spoke had a free running "
                         "slot at dispatch — i.e., NO router-side "
                         "queueing contributed to elapsed_seconds. This "
                         "removes the μ inflation caused by transient "
                         "queueing in the source trace.")
    ap.add_argument("--output", type=pathlib.Path, required=True,
                    help="path to write calibration.json.")
    args = ap.parse_args()

    if not args.trace.exists():
        print(f"ERROR: trace {args.trace} not found", file=sys.stderr)
        return 1

    with args.trace.open() as f:
        trace = json.load(f)

    records = trace.get("records")
    if not isinstance(records, list) or not records:
        print(f"ERROR: {args.trace} has no `records` list", file=sys.stderr)
        return 1

    # Filter threshold for what counts as "dispatched without router-
    # side queueing." Default: c_per_spoke — if at least one spoke had
    # a free running slot at dispatch, the request wasn't waiting on
    # the router queue.
    inflight_ceiling = (args.max_inflight_for_mu
                        if args.max_inflight_for_mu is not None
                        else args.c_per_spoke)

    # Extract (arrival_ts, elapsed, in_flight, served) tuples. Warmup
    # drops by arrival order; the queueing filter drops by
    # in_flight_at_arrival.
    rows: list[tuple[float, float, int, str]] = []
    dropped_no_data = 0
    for r in records:
        extra = r.get("extra_fields", {}) or {}
        arrival = extra.get("arrival_ts_seconds")
        elapsed = extra.get("elapsed_seconds")
        served = extra.get("served_endpoint", "unknown")
        in_flight = extra.get("in_flight_at_arrival")
        if arrival is None or elapsed is None or in_flight is None:
            dropped_no_data += 1
            continue
        rows.append((float(arrival), float(elapsed), int(in_flight), str(served)))

    if len(rows) <= args.warmup + 50:
        print(f"ERROR: only {len(rows)} rows with arrival_ts, "
              f"elapsed_seconds, and in_flight_at_arrival; need at "
              f"least {args.warmup + 50}.", file=sys.stderr)
        return 1

    rows.sort(key=lambda t: t[0])
    steady = rows[args.warmup:]

    # Filter to rows dispatched without router-side queueing.
    no_queue_rows = [r for r in steady if r[2] <= inflight_ceiling]
    if len(no_queue_rows) < 100:
        print(f"ERROR: only {len(no_queue_rows)} rows in the trace were "
              f"dispatched with in_flight_at_arrival <= "
              f"{inflight_ceiling}. That's too few for a reliable μ. "
              f"Either raise --max-inflight-for-mu, or run "
              f"calibrate-lambda.sh against a live fleet.", file=sys.stderr)
        return 1

    elapsed_steady = [e for _, e, _, _ in no_queue_rows]
    mu_TP2 = statistics.fmean(elapsed_steady)
    elapsed_sorted = sorted(elapsed_steady)
    n = len(elapsed_sorted)

    def q(p: float) -> float:
        if n == 1:
            return elapsed_sorted[0]
        rank = p * (n - 1)
        lo = int(rank)
        hi = min(lo + 1, n - 1)
        frac = rank - lo
        return elapsed_sorted[lo] * (1 - frac) + elapsed_sorted[hi] * frac

    # ρ = λ / (c · μ) solved for λ, using per-spoke terms because the
    # new fleet is asymmetric:
    #   total_cap_new = c/μ_TP2 + c/μ_TP4
    #                 = c · (1 + tp_speedup) / μ_TP2
    mu_TP4 = mu_TP2 / args.tp_speedup
    total_cap_a = args.c_per_spoke / mu_TP2   # spoke-a (TP=2)
    total_cap_b = args.c_per_spoke / mu_TP4   # spoke-b (TP=4)
    total_cap_new = total_cap_a + total_cap_b
    study_lambda = args.target_rho * total_cap_new

    # Diagnostic: what would ρ have been under the trace's own
    # (target_concurrent, c_effective) if we had that info? We don't
    # have target_concurrent in the trace itself, but we can note the
    # observed in_flight_at_arrival distribution.
    inflight = [r.get("extra_fields", {}).get("in_flight_at_arrival")
                for r in records
                if r.get("extra_fields", {}).get("in_flight_at_arrival") is not None]
    inflight_sorted = sorted(int(x) for x in inflight)
    inflight_stats = {
        "n": len(inflight_sorted),
        "min": inflight_sorted[0] if inflight_sorted else None,
        "median": inflight_sorted[len(inflight_sorted)//2] if inflight_sorted else None,
        "p95": inflight_sorted[int(0.95*len(inflight_sorted))] if inflight_sorted else None,
        "max": inflight_sorted[-1] if inflight_sorted else None,
    }

    out = {
        "source":                        "trace",
        "trace_path":                    str(args.trace),
        "trace_rows_total":              len(rows),
        "warmup_dropped":                args.warmup,
        "max_inflight_for_mu":           inflight_ceiling,
        "trace_rows_used_for_mu":        len(elapsed_steady),
        "trace_rows_dropped_queueing":   len(steady) - len(no_queue_rows),
        "mu_TP2_seconds":                mu_TP2,
        "mu_TP2_median_seconds":    q(0.50),
        "mu_TP2_p95_seconds":       q(0.95),
        "mu_TP2_p99_seconds":       q(0.99),
        "tp_speedup_assumed":       args.tp_speedup,
        "mu_TP4_seconds":           mu_TP4,
        "c_per_spoke":              args.c_per_spoke,
        "c_effective":              2 * args.c_per_spoke,
        "target_rho":               args.target_rho,
        "total_cap_new_per_sec":    total_cap_new,
        "study_lambda":             study_lambda,
        "trace_inflight_stats":     inflight_stats,
        "calibrated_at":            datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "notes": (
            "study_lambda derived analytically from the trace's μ_TP2 via "
            "λ = target_rho · c_per_spoke · (1 + tp_speedup) / μ_TP2. "
            "Assumes the new fleet's spoke-b (TP=4) has service time "
            "μ_TP2 / tp_speedup. Validate empirically by inspecting the "
            "first cell's `in_flight_at_arrival` distribution — median "
            "should hover near the analytical E[N] at ρ=target_rho."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"[compute_lambda_from_trace] wrote {args.output}", file=sys.stderr)
    print(json.dumps(out, indent=2))

    # Sanity warnings.
    p99_over_mean = q(0.99) / mu_TP2 if mu_TP2 > 0 else float("inf")
    if p99_over_mean > 10:
        print(f"\nWARN: p99/mean = {p99_over_mean:.1f}. Service time is "
              f"very heavy-tailed. ρ = λ/(cμ) uses the MEAN; that's "
              f"still the right calibration for offered load. But the "
              f"fleet will see p99 waits far larger than μ at ρ = "
              f"{args.target_rho}.", file=sys.stderr)

    if inflight_stats["max"] is not None:
        max_inflight = inflight_stats["max"]
        c_effective = 2 * args.c_per_spoke
        if max_inflight > c_effective:
            print(f"\nWARN: trace's max in_flight_at_arrival = "
                  f"{max_inflight} > c_effective = {c_effective}. "
                  f"This means the trace HAD transient queueing on "
                  f"the router side. μ_TP2 is slightly inflated over "
                  f"true isolated-request service time; the derived "
                  f"study_lambda is a conservative (slightly-under) "
                  f"estimate. Not a hard problem — validate ρ in the "
                  f"first cell.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
