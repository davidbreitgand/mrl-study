#!/usr/bin/env python3
"""Smoke-test validator for a single mrl-study cell.

Reads a smoke run's outputs (journal + progress) and asserts the run
was healthy enough to justify launching the 2000-row full run.

Exits 0 on pass, non-zero with a diagnostic on any failure.

Checks (each enabled by a flag):
  --forbid-5xx                     no journal row has a 5xx finish_reason.
  --require-served-endpoint        every journal row has a non-empty
                                   extra_fields.served_endpoint.
  --forbid-semaphore-saturation    progress.jsonl shows no long steady
                                   stretch of `in_flight ==
                                   max_concurrent_requests`.
  --require-mrl-digests-warm       (MRL runs only) the observer's
                                   Prometheus gauges show both digests
                                   past minSamples=60 on both spokes
                                   at end of smoke. Currently a stub
                                   that checks whether the caller wired
                                   the gauge dump into the results dir
                                   under `metrics/` — until the
                                   observer's Prometheus exposure is
                                   wired end-to-end, this check emits
                                   a warning rather than failing hard.

Usage:
  scripts/validate_smoke.py <smoke-dir> --require-served-endpoint \\
      --forbid-5xx --forbid-semaphore-saturation
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def iter_journal_rows(smoke_dir: pathlib.Path):
    """Yield each row from every *.jsonl under smoke_dir/journal/**."""
    for jpath in sorted(smoke_dir.glob("**/*.jsonl")):
        # Skip progress.jsonl — it's the arrival/dispatch log, not
        # per-completion records.
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


def check_forbid_5xx(smoke_dir: pathlib.Path) -> list[str]:
    """Return a list of failure messages; empty means pass."""
    bad: list[str] = []
    total = 0
    for row in iter_journal_rows(smoke_dir):
        total += 1
        extra = row.get("extra_fields", {}) or {}
        finish = str(extra.get("finish_reason", "")).lower()
        if finish.startswith("5") or "server_error" in finish or "5xx" in finish:
            bad.append(f"row {row.get('index')}: finish_reason={finish!r}")
    if bad:
        return [f"5xx failures in {len(bad)}/{total} rows:"] + bad[:5]
    return []


def check_require_served_endpoint(smoke_dir: pathlib.Path) -> list[str]:
    missing: list[int] = []
    total = 0
    for row in iter_journal_rows(smoke_dir):
        total += 1
        extra = row.get("extra_fields", {}) or {}
        ep = extra.get("served_endpoint")
        if not ep:
            missing.append(row.get("index"))
    if missing:
        return [
            f"missing served_endpoint on {len(missing)}/{total} rows "
            f"(first few indices: {missing[:5]})"
        ]
    return []


def check_forbid_semaphore_saturation(smoke_dir: pathlib.Path) -> list[str]:
    """Look for stretches where in_flight_at_arrival stays at the cap."""
    progress = list(smoke_dir.glob("**/progress.jsonl"))
    if not progress:
        return [
            "no progress.jsonl found — cannot check semaphore saturation "
            "(pass-through, not a hard failure)"
        ]
    # The cap value is not directly recorded in progress.jsonl; infer
    # it as the max in_flight seen across the run and warn if that max
    # holds for a long consecutive stretch of arrivals.
    max_in_flight = 0
    values: list[int] = []
    for p in progress:
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                v = row.get("in_flight_at_arrival")
                if v is not None:
                    values.append(int(v))
                    max_in_flight = max(max_in_flight, int(v))
    if not values:
        return []
    # Semaphore default is 128. Under a well-calibrated cell the max
    # is much lower. Flag when >20% of arrivals are at max_in_flight.
    saturated = sum(1 for v in values if v == max_in_flight)
    if max_in_flight >= 100 and saturated > 0.2 * len(values):
        return [
            f"in_flight_at_arrival saturated at {max_in_flight} on "
            f"{saturated}/{len(values)} arrivals — lambda too high for "
            f"the fleet or the sanity-cap semaphore is binding at "
            f"steady state; recalibrate before the full run."
        ]
    return []


def check_require_mrl_digests_warm(smoke_dir: pathlib.Path) -> list[str]:
    """Look for a metrics/mrl-digest-counts.json emitted by the Hub's
    Prometheus scrape. Absent-file is currently a warning; when the
    Hub's metrics wiring is complete, upgrade this to a hard failure.
    """
    metrics = smoke_dir / "metrics" / "mrl-digest-counts.json"
    if not metrics.exists():
        return [
            f"WARN: {metrics} not present — the Hub's Prometheus "
            f"exposure for the mrl scorer's digest counts is not "
            f"wired to the results PVC yet. Falling back to a "
            f"plumbing-only smoke; this WILL BE a hard failure once "
            f"the metrics path is complete."
        ]
    with metrics.open() as f:
        data = json.load(f)
    bad: list[str] = []
    for endpoint, counts in (data.get("endpoints") or {}).items():
        ttft = counts.get("ttft_count", 0)
        decode = counts.get("decode_count", 0)
        if ttft < 60 or decode < 60:
            bad.append(f"{endpoint}: ttft={ttft} decode={decode} (need >= 60 each)")
    if bad:
        return ["mrl digest warm-up incomplete at end of smoke:"] + bad
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("smoke_dir", type=pathlib.Path)
    ap.add_argument("--forbid-5xx", action="store_true")
    ap.add_argument("--require-served-endpoint", action="store_true")
    ap.add_argument("--forbid-semaphore-saturation", action="store_true")
    ap.add_argument("--require-mrl-digests-warm", action="store_true")
    args = ap.parse_args()

    if not args.smoke_dir.exists():
        print(f"ERROR: smoke_dir {args.smoke_dir} not found", file=sys.stderr)
        return 1

    problems: list[str] = []
    warnings: list[str] = []

    if args.forbid_5xx:
        problems += check_forbid_5xx(args.smoke_dir)
    if args.require_served_endpoint:
        problems += check_require_served_endpoint(args.smoke_dir)
    if args.forbid_semaphore_saturation:
        problems += check_forbid_semaphore_saturation(args.smoke_dir)
    if args.require_mrl_digests_warm:
        # The digests-warm check currently emits a warning when the
        # metrics file is absent (not a hard fail). Sort those out.
        for line in check_require_mrl_digests_warm(args.smoke_dir):
            if line.startswith("WARN:"):
                warnings.append(line)
            else:
                problems.append(line)

    for w in warnings:
        print(w, file=sys.stderr)
    if problems:
        print("smoke validation FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    print("[validate_smoke] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
