#!/usr/bin/env python3
"""Analyze one stage of the staged λ-validation probe.

Reads a probe stage's journal (a completed harness run at some
candidate λ), computes empirical ρ two ways via Little's Law, and
prints the next-λ suggestion for the bisection loop.

The two ρ estimators are derived explicitly in the module docstring
below; the invocation contract is minimal so the outer bash driver
(`validate_lambda_probe.sh`) can shell out and parse the JSON.

Estimator ρ̂_honest — Little's Law + PASTA:

  Little's Law:              E[N] = λ · E[T]
  Decompose:                 E[N] = E[N_running] + E[N_queue]
  Utilization definition:    ρ = E[N_running] / c
  PASTA (Poisson Arrivals See Time Averages):
      time-average E[N] can be estimated from arrival-observed values.
  E[N_running] estimator:    per row, min(c, in_flight_at_arrival) is
      the number that were RUNNING when the row's arrival landed
      (anything above c was in a waiting FIFO).
  So:  ρ̂_honest = mean(min(c, in_flight_at_arrival)) / c

Estimator ρ̂_lambda — direct from λ · s / c:

  ρ̂_lambda = λ_observed · mean(elapsed_seconds) / c

  Requires that the probe actually reached steady state so
  mean(elapsed_seconds) reflects sojourn time under the target load.
  Under transient, this estimator overstates ρ.

Both estimators should agree within a few percent at steady state.
Disagreement flags a probe that didn't run long enough, or a fleet
that wasn't stable during the probe.

Usage:
  probe_stage_analyze.py \\
      --results /tmp/mrl-probe-stage-N/ \\
      --lambda-driven 0.319 \\
      --c-effective 16 \\
      --target-rho 0.98 \\
      --target-band 0.95 1.02 \\
      --warmup 30

Exits 0 on success; JSON dumped to stdout, human summary to stderr.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys


def iter_rows(results_dir: pathlib.Path):
    """Yield rows from every *.jsonl under results_dir, skipping
    progress.jsonl."""
    for jpath in sorted(results_dir.glob("**/*.jsonl")):
        if jpath.name == "progress.jsonl":
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
    ap.add_argument("--results", type=pathlib.Path, required=True)
    ap.add_argument("--lambda-driven", dest="lam_driven", type=float, required=True,
                    help="the λ the harness was driven at during this stage.")
    ap.add_argument("--c-effective", type=int, required=True)
    ap.add_argument("--target-rho", type=float, default=0.98)
    ap.add_argument("--target-band", type=float, nargs=2, default=[0.95, 1.02],
                    metavar=("LO", "HI"))
    ap.add_argument("--warmup", type=int, default=30)
    args = ap.parse_args()

    rows: list[tuple[float, float, int, str]] = []
    for r in iter_rows(args.results):
        extra = r.get("extra_fields", {}) or {}
        arrival = extra.get("arrival_ts_seconds")
        elapsed = extra.get("elapsed_seconds")
        in_flight = extra.get("in_flight_at_arrival")
        served = extra.get("served_endpoint", "unknown")
        if arrival is None or elapsed is None or in_flight is None:
            continue
        rows.append((float(arrival), float(elapsed),
                     int(in_flight), str(served)))

    if len(rows) <= args.warmup + 50:
        print(f"ERROR: only {len(rows)} usable rows; need at least "
              f"{args.warmup + 50}. Was the probe wide enough?",
              file=sys.stderr)
        return 1

    rows.sort(key=lambda t: t[0])
    steady = rows[args.warmup:]

    # ρ̂_honest via PASTA + min-with-c
    running_at_arrival = [min(args.c_effective, inflight)
                          for _, _, inflight, _ in steady]
    mean_running = statistics.fmean(running_at_arrival)
    rho_honest = mean_running / args.c_effective

    # ρ̂_lambda: measure observed λ, mean sojourn, apply ρ = λ · s / c
    arrivals = [a for a, _, _, _ in steady]
    duration = arrivals[-1] - arrivals[0]
    lam_observed = (len(arrivals) - 1) / duration if duration > 0 else 0.0
    elapsed_seconds = [e for _, e, _, _ in steady]
    mean_elapsed = statistics.fmean(elapsed_seconds)
    rho_lambda = lam_observed * mean_elapsed / args.c_effective

    # Queue-formation diagnostic: E[N_queue] = E[N] - E[N_running]
    all_in_flight = [inflight for _, _, inflight, _ in steady]
    mean_in_flight = statistics.fmean(all_in_flight)
    mean_queue = mean_in_flight - mean_running
    inflight_sorted = sorted(all_in_flight)
    median_in_flight = inflight_sorted[len(inflight_sorted)//2]
    max_in_flight = inflight_sorted[-1]

    # Consistency check
    disagreement_pct = (abs(rho_honest - rho_lambda) / max(rho_honest, rho_lambda) * 100
                        if max(rho_honest, rho_lambda) > 0 else 0.0)

    # Verdict: is ρ_honest in the target band?
    lo, hi = args.target_band
    in_band = (lo <= rho_honest <= hi)
    # Guardrails on the analytic step:
    # - floor rho_honest at 0.01 to prevent division blow-up when a
    #   very under-driven probe measured no in-flight overlap.
    # - cap lambda_next to 3x lambda_driven to prevent a runaway
    #   overshoot into semaphore saturation if a fluke gave a very
    #   low rho.
    # - when rho_honest saturates at 1.0, its magnitude is lost — the
    #   sensor cannot distinguish "just at capacity" from "8x
    #   overdriven" because it clips at c. Fall back to rho_lambda
    #   (which reflects the true magnitude via mean_elapsed's inclusion
    #   of queue wait) so a saturated stage takes a step proportional
    #   to the overload rather than a ~1% correction.
    rho_saturated = rho_honest >= 0.99
    if rho_saturated and rho_lambda > rho_honest:
        rho_for_step = rho_lambda
    else:
        rho_for_step = max(rho_honest, 0.01)
    lambda_analytic = args.lam_driven * args.target_rho / rho_for_step
    if in_band:
        verdict = "IN_BAND"
        lambda_next = args.lam_driven          # keep this λ
    elif rho_honest < lo:
        verdict = "UNDER_DRIVEN"
        # Aim for target_rho with a single analytic step, then bisect
        # toward it (midpoint of current and analytic estimate).
        lambda_next = (args.lam_driven + lambda_analytic) / 2
        lambda_next = min(lambda_next, 3.0 * args.lam_driven)
    else:
        verdict = "OVER_DRIVEN"
        lambda_next = (args.lam_driven + lambda_analytic) / 2
        # Symmetric cap on the downside: don't fall below 1/3 in one step.
        lambda_next = max(lambda_next, args.lam_driven / 3.0)

    out = {
        "lambda_driven":         args.lam_driven,
        "lambda_observed":       lam_observed,
        "c_effective":           args.c_effective,
        "n_steady_rows":         len(steady),
        "warmup_dropped":        args.warmup,
        "mean_elapsed_seconds":  mean_elapsed,
        "mean_running_at_arrival": mean_running,
        "mean_queue_at_arrival": mean_queue,
        "mean_in_flight_at_arrival": mean_in_flight,
        "median_in_flight":      median_in_flight,
        "max_in_flight":         max_in_flight,
        "rho_honest":            rho_honest,
        "rho_lambda":            rho_lambda,
        "disagreement_pct":      disagreement_pct,
        "target_rho":            args.target_rho,
        "target_band":           list(args.target_band),
        "verdict":               verdict,
        "queues_observed":       mean_queue > 0.5,   # any measurable queueing
        "lambda_next":           lambda_next,
    }
    print(json.dumps(out, indent=2))

    print(f"\n=== Stage summary ===", file=sys.stderr)
    print(f"  λ driven      = {args.lam_driven:.4f}/sec", file=sys.stderr)
    print(f"  λ observed    = {lam_observed:.4f}/sec", file=sys.stderr)
    print(f"  ρ̂_honest      = {rho_honest:.3f}  (E[N_running] / c)", file=sys.stderr)
    print(f"  ρ̂_lambda      = {rho_lambda:.3f}  (λ · E[T] / c)", file=sys.stderr)
    print(f"  disagreement  = {disagreement_pct:.2f}% "
          f"({'OK' if disagreement_pct < 5 else 'HIGH — probe may not be at steady state'})",
          file=sys.stderr)
    print(f"  mean queued   = {mean_queue:.2f}  "
          f"({'QUEUES FORMED' if mean_queue > 0.5 else 'no queueing'})",
          file=sys.stderr)
    print(f"  verdict       = {verdict}", file=sys.stderr)
    if not in_band:
        print(f"  next λ        = {lambda_next:.4f}/sec  (bisection step)", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
