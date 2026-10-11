#!/usr/bin/env python3
"""Aggregate the Poisson-replicate study for one dataset.

Walks local_results/<dataset>/poisson/<scorer>-poisson-<dataset>-rep<N>/
for N = 0, 1, 2, 3 (or however many replicates exist), computes per-cell
elapsed_seconds percentiles + rho_honest, and reports per-scorer
mean/std across the replicates.

The report exposes three quantities per (scorer):

  1. Point estimates aggregated across replicates:
       mean(p99), std(p99), min/max — how tight is the tail estimate?
       mean(mean_elapsed), mean(median), mean(p95) — central tendency.
  2. Delivered load audit:
       mean(rho_honest), std(rho_honest) — did the scorer see the same
       operating point each replicate?
  3. Per-replicate rows for transparency.

Δ% vs QD is computed on the mean-p99 across replicates, so the
head-to-head comparison uses the tighter estimator.

This is deliberately a small, single-regime tool; the full replicate
generalization of build_headtohead_table.py is a separate task.

Usage:
  python scripts/build_poisson_replicate_table.py \\
      --dataset totrandom-tail472 \\
      --out local_results/totrandom-tail472/head-to-head-poisson-replicated/
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import sys


SCORERS = ("QD", "AR", "LAT", "MRL")
C_EFFECTIVE = 16
WARMUP = 30


def iter_journal_rows(cell_dir: pathlib.Path):
    for jpath in sorted(cell_dir.glob("**/*.jsonl")):
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


def percentiles(values):
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0}
    vs = sorted(values)
    n = len(vs)

    def q(p):
        if n == 1:
            return vs[0]
        rank = p * (n - 1)
        lo = int(rank)
        hi = min(lo + 1, n - 1)
        frac = rank - lo
        return vs[lo] * (1 - frac) + vs[hi] * frac

    return {
        "n":      n,
        "mean":   statistics.fmean(vs),
        "median": q(0.50),
        "p95":    q(0.95),
        "p99":    q(0.99),
    }


def summarise_cell(cell_dir: pathlib.Path) -> dict:
    """Read one replicate's journal; return elapsed percentiles + rho_honest."""
    elapsed = []
    inflight = []
    arm_counts: dict[str, int] = {}
    for row in iter_journal_rows(cell_dir):
        e = row.get("extra_fields", {}) or {}
        el = e.get("elapsed_seconds")
        if el is not None:
            elapsed.append(float(el))
        ifi = e.get("in_flight_at_arrival")
        if ifi is not None:
            inflight.append(int(ifi))
        served = e.get("served_endpoint") or "unknown"
        arm_counts[served] = arm_counts.get(served, 0) + 1

    # Warmup drop matches the calibration convention.
    inflight_ss = inflight[WARMUP:]
    elapsed_ss  = elapsed[WARMUP:]

    pct = percentiles(elapsed_ss)
    if inflight_ss:
        running = [min(C_EFFECTIVE, v) for v in inflight_ss]
        rho_honest = statistics.fmean(running) / C_EFFECTIVE
    else:
        rho_honest = None

    total = sum(arm_counts.values())
    arm_share = {k: v / total if total else 0.0 for k, v in arm_counts.items()}
    return {
        "percentiles": pct,
        "rho_honest":  rho_honest,
        "arm_share":   arm_share,
        "n_rows":      len(elapsed),
    }


def discover_replicates(dataset_root: pathlib.Path, scorer: str, dataset: str):
    """Return sorted list of (rep_index, cell_dir) for the scorer's Poisson replicates."""
    slug = scorer.lower()
    poisson_dir = dataset_root / "poisson"
    if not poisson_dir.exists():
        return []
    pat = re.compile(rf"^{slug}-poisson-{re.escape(dataset)}-rep(\d+)$")
    reps: list[tuple[int, pathlib.Path]] = []
    for d in poisson_dir.iterdir():
        if not d.is_dir():
            continue
        m = pat.match(d.name)
        if m:
            reps.append((int(m.group(1)), d))
    reps.sort(key=lambda x: x[0])
    return reps


def aggregate_replicates(cell_summaries: list[dict]) -> dict:
    """Per-scorer stats across N replicates."""
    if not cell_summaries:
        return {}
    ps = [c["percentiles"] for c in cell_summaries]
    rhos = [c["rho_honest"] for c in cell_summaries if c["rho_honest"] is not None]

    def stats(vs):
        if not vs:
            return {"mean": None, "std": None, "min": None, "max": None}
        return {
            "mean": statistics.fmean(vs),
            "std":  statistics.stdev(vs) if len(vs) > 1 else 0.0,
            "min":  min(vs),
            "max":  max(vs),
        }

    return {
        "n_replicates": len(cell_summaries),
        "p99":    stats([p["p99"]    for p in ps]),
        "p95":    stats([p["p95"]    for p in ps]),
        "median": stats([p["median"] for p in ps]),
        "mean":   stats([p["mean"]   for p in ps]),
        "rho_honest": stats(rhos),
    }


def delta_pct(base: float, alt: float) -> float:
    if base is None or alt is None or abs(base) < 1e-12:
        return float("nan")
    return 100.0 * (base - alt) / base


def emit_report(scorer_summaries: dict, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- JSON dump ---
    with (out_dir / "poisson-replicated.json").open("w") as f:
        json.dump(scorer_summaries, f, indent=2)
    print(f"[build-poisson-replicate] wrote {out_dir / 'poisson-replicated.json'}", file=sys.stderr)

    # --- Text log ---
    lines: list[str] = []
    lines.append("=== Poisson replicate summary (N replicates per scorer, per-cell warmup=30) ===")
    lines.append("")
    lines.append("Table 1 — Aggregated central-tendency stats (mean ± SD across replicates)")
    lines.append("")
    header = (f"{'Scorer':<6}  {'reps':>4}  {'ρ_hon±':>13}  "
              f"{'mean±':>13}  {'median±':>13}  {'p95±':>13}  {'p99±':>13}")
    lines.append(header)
    lines.append("-" * len(header))
    qd_agg = scorer_summaries.get("QD", {}).get("aggregate") or {}
    qd_stat = {k: qd_agg.get(k, {}).get("mean") for k in ("mean", "median", "p95", "p99")}
    for scorer in SCORERS:
        agg = scorer_summaries.get(scorer, {}).get("aggregate")
        if not agg:
            lines.append(f"{scorer:<6}  MISSING (no replicates found)")
            continue
        rho   = agg["rho_honest"]
        rho_s = f"{rho['mean']:.3f}±{rho['std']:.3f}" if rho['mean'] is not None else "n/a"
        def fmt(s):
            return f"{s['mean']:>7.2f}±{s['std']:>5.2f}"
        lines.append(
            f"{scorer:<6}  {agg['n_replicates']:>4}  {rho_s:>13}  "
            f"{fmt(agg['mean']):>13}  {fmt(agg['median']):>13}  "
            f"{fmt(agg['p95']):>13}  {fmt(agg['p99']):>13}"
        )
    lines.append("")

    lines.append("Table 2 — Δ% vs QD (positive = tail reduction; computed on mean-across-replicates)")
    lines.append("")
    header2 = (f"{'Scorer':<6}  {'Δmean%':>10}  {'Δmedian%':>10}  {'Δp95%':>10}  {'Δp99%':>10}")
    lines.append(header2)
    lines.append("-" * len(header2))
    for scorer in SCORERS:
        agg = scorer_summaries.get(scorer, {}).get("aggregate")
        if not agg:
            continue
        if scorer == "QD":
            lines.append(f"{scorer:<6}  {'—':>10}  {'—':>10}  {'—':>10}  {'—':>10}")
            continue
        row = f"{scorer:<6}"
        for key in ("mean", "median", "p95", "p99"):
            d = delta_pct(qd_stat[key], agg[key]["mean"])
            row += f"  {d:>+9.1f}%"
        lines.append(row)
    lines.append("")

    lines.append("Table 3 — Range across replicates (min/max)")
    lines.append("")
    header3 = (f"{'Scorer':<6}  "
               f"{'p99_min':>9}  {'p99_max':>9}  "
               f"{'p95_min':>9}  {'p95_max':>9}  "
               f"{'median_min':>10}  {'median_max':>10}")
    lines.append(header3)
    lines.append("-" * len(header3))
    for scorer in SCORERS:
        agg = scorer_summaries.get(scorer, {}).get("aggregate")
        if not agg:
            continue
        p99  = agg["p99"]; p95  = agg["p95"]; med  = agg["median"]
        lines.append(
            f"{scorer:<6}  "
            f"{p99['min']:>9.2f}  {p99['max']:>9.2f}  "
            f"{p95['min']:>9.2f}  {p95['max']:>9.2f}  "
            f"{med['min']:>10.2f}  {med['max']:>10.2f}"
        )
    lines.append("")

    lines.append("=== Per-replicate detail (elapsed_seconds) ===")
    lines.append("")
    hdr = f"{'Scorer':<6}  {'rep':>4}  {'n':>4}  {'ρ_honest':>8}  {'mean':>8}  {'median':>8}  {'p95':>8}  {'p99':>8}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for scorer in SCORERS:
        entry = scorer_summaries.get(scorer, {})
        for rep in entry.get("replicates", []):
            p = rep["cell"]["percentiles"]
            rho = rep["cell"]["rho_honest"]
            rho_s = f"{rho:.3f}" if rho is not None else "  n/a"
            lines.append(
                f"{scorer:<6}  {rep['rep']:>4}  {p['n']:>4}  {rho_s:>8}  "
                f"{p['mean']:>8.2f}  {p['median']:>8.2f}  "
                f"{p['p95']:>8.2f}  {p['p99']:>8.2f}"
            )
        lines.append("")

    log_path = out_dir / "poisson-replicated-log.txt"
    with log_path.open("w") as f:
        f.write("\n".join(lines))
    print(f"[build-poisson-replicate] wrote {log_path}", file=sys.stderr)
    print("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True,
                    help="dataset name; the aggregator walks "
                         "local_results/<dataset>/poisson/")
    ap.add_argument("--results-root", type=pathlib.Path, default=None,
                    help="local_results root. Default: <script>/../local_results.")
    ap.add_argument("--out", type=pathlib.Path, required=True,
                    help="output directory for the summary artefacts.")
    args = ap.parse_args()

    results_root = args.results_root or (
        pathlib.Path(__file__).resolve().parents[1] / "local_results"
    )
    dataset_root = results_root / args.dataset
    if not dataset_root.exists():
        print(f"ERROR: dataset root {dataset_root} not found", file=sys.stderr)
        return 1

    scorer_summaries: dict = {}
    for scorer in SCORERS:
        reps = discover_replicates(dataset_root, scorer, args.dataset)
        if not reps:
            scorer_summaries[scorer] = {
                "replicates": [],
                "aggregate":  None,
                "error":      f"no replicates found under {dataset_root}/poisson/",
            }
            continue
        rep_entries = []
        cell_summaries = []
        for rep_idx, cell_dir in reps:
            s = summarise_cell(cell_dir)
            rep_entries.append({"rep": rep_idx, "cell_dir": str(cell_dir), "cell": s})
            cell_summaries.append(s)
        scorer_summaries[scorer] = {
            "replicates": rep_entries,
            "aggregate":  aggregate_replicates(cell_summaries),
        }

    emit_report(scorer_summaries, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
