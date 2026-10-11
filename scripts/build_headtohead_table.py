#!/usr/bin/env python3
"""4-scorer × 3-regime head-to-head builder for one dataset.

Reads local_results/<dataset>/{poisson,weibull-0.5,weibull-0.3}/ for
each of the four scorers (QD, AR, LAT, MRL), computes elapsed-time
percentiles per cell, and writes:

  <out>/head-to-head.json         one dict per (scorer, regime) cell.
  <out>/head-to-head.png          3 stacked sub-plots (one per regime),
                                  grouped bar chart of mean/median/p95/p99.
  <out>/arm-attribution.png       12 stacked bars — spoke-a vs spoke-b share.
  <out>/head-to-head-log.txt      one paragraph per regime.

Δ% is computed within each arrival regime against the QD row:
  Δ% = 100 * (QD_value - scorer_value) / QD_value
so positive Δ% means the scorer reduced the statistic vs QD.

Each cell dir is expected to contain a scorer-namespaced subdir
(e.g. mrl-poisson-<dataset>/) with journal + consolidated outputs, as
run-config.sh writes them.

Usage:
  python scripts/build_headtohead_table.py \\
      --dataset totrandom-tail472 \\
      --out local_results/totrandom-tail472/head-to-head/
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from typing import Iterable


REGIMES = ("poisson", "weibull-0.5", "weibull-0.3")
SCORERS = ("QD", "AR", "LAT", "MRL")   # QD first; used as Δ% denominator


def iter_journal_rows(cell_dir: pathlib.Path) -> Iterable[dict]:
    """Yield each row from every *.jsonl under cell_dir/journal/**.

    Skips progress.jsonl. Skips smoke/ (audit-only)."""
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


def percentiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0}
    values_sorted = sorted(values)
    n = len(values_sorted)

    def q(p: float) -> float:
        # Linear interpolation matching numpy's default. Sim uses pandas
        # .quantile; this replicates it without a numpy/pandas dependency.
        if n == 1:
            return values_sorted[0]
        rank = p * (n - 1)
        lo = int(rank)
        hi = min(lo + 1, n - 1)
        frac = rank - lo
        return values_sorted[lo] * (1 - frac) + values_sorted[hi] * frac

    return {
        "n": n,
        "mean":   statistics.fmean(values_sorted),
        "median": q(0.50),
        "p95":    q(0.95),
        "p99":    q(0.99),
    }


def summarise_cell(cell_dir: pathlib.Path) -> dict:
    """Read one cell dir; return per-arm share + short-class percentiles."""
    elapsed: list[float] = []
    arm_counts: dict[str, int] = {}
    for row in iter_journal_rows(cell_dir):
        extra = row.get("extra_fields", {}) or {}
        e = extra.get("elapsed_seconds")
        if e is None:
            continue
        elapsed.append(float(e))
        served = extra.get("served_endpoint") or "unknown"
        arm_counts[served] = arm_counts.get(served, 0) + 1

    percentile_stats = percentiles(elapsed)
    total = sum(arm_counts.values())
    arm_share = {k: (v / total if total else 0.0) for k, v in arm_counts.items()}

    return {
        "percentiles": percentile_stats,
        "arm_share":  arm_share,
        "arm_counts": arm_counts,
        "total":      total,
    }


def build_cells(root_dirs: dict[str, pathlib.Path],
                dataset: str = "") -> dict[str, dict[str, dict]]:
    """Return {scorer: {regime: cell_summary}}.

    All scorers share one dataset root under the new layout; the
    per-scorer subdir inside each regime dir is
    <scorer_lower>-<regime>-<dataset>/. summarise_cell then only reads
    that specific subdir so QD and AR journals don't mix.
    """
    cells: dict[str, dict[str, dict]] = {}
    for scorer, root in root_dirs.items():
        cells[scorer] = {}
        for regime in REGIMES:
            regime_dir = root / regime
            if not regime_dir.exists():
                cells[scorer][regime] = {"error": f"missing {regime_dir}"}
                continue
            scorer_slug = scorer.lower()
            subdir = regime_dir / f"{scorer_slug}-{regime}-{dataset}" if dataset \
                     else regime_dir / f"{scorer_slug}-{regime}"
            if not subdir.exists():
                cells[scorer][regime] = {"error": f"missing {subdir}"}
                continue
            cells[scorer][regime] = summarise_cell(subdir)
    return cells


def delta_pct(base: float, alt: float) -> float:
    if abs(base) < 1e-12:
        return float("nan")
    return 100.0 * (base - alt) / base


def emit_json(cells: dict, out: pathlib.Path) -> None:
    payload = {}
    for scorer, regimes in cells.items():
        payload[scorer] = {}
        for regime, summary in regimes.items():
            if "error" in summary:
                payload[scorer][regime] = {"error": summary["error"]}
                continue
            entry = {
                "n":         summary["percentiles"]["n"],
                "mean":      summary["percentiles"]["mean"],
                "median":    summary["percentiles"]["median"],
                "p95":       summary["percentiles"]["p95"],
                "p99":       summary["percentiles"]["p99"],
                "arm_share": summary["arm_share"],
            }
            if scorer != "QD":
                qd = cells.get("QD", {}).get(regime)
                if qd and "percentiles" in qd:
                    for stat in ("mean", "median", "p95", "p99"):
                        entry[f"delta_pct_vs_qd_{stat}"] = delta_pct(
                            qd["percentiles"][stat], entry[stat]
                        )
            payload[scorer][regime] = entry
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"[build_headtohead] wrote {out}", file=sys.stderr)


def emit_log(cells: dict, out: pathlib.Path) -> None:
    lines: list[str] = []
    for regime in REGIMES:
        lines.append(f"=== Arrival: {regime} ===")
        header = f"{'Scorer':<8}  {'n':>5}  {'mean':>8}  {'median':>8}  {'p95':>8}  {'p99':>8}  {'Δp99%':>8}"
        lines.append(header)
        lines.append("-" * len(header))
        qd_p99 = cells.get("QD", {}).get(regime, {}).get("percentiles", {}).get("p99")
        for scorer in SCORERS:
            summary = cells.get(scorer, {}).get(regime, {})
            if "error" in summary or "percentiles" not in summary:
                lines.append(f"{scorer:<8}  MISSING ({summary.get('error', 'no data')})")
                continue
            p = summary["percentiles"]
            dp99 = "  —"
            if scorer != "QD" and qd_p99 is not None:
                dp99 = f"{delta_pct(qd_p99, p['p99']):+7.1f}%"
            lines.append(f"{scorer:<8}  {p['n']:>5}  {p['mean']:>8.2f}  {p['median']:>8.2f}  "
                         f"{p['p95']:>8.2f}  {p['p99']:>8.2f}  {dp99:>8}")
        lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("\n".join(lines))
    print(f"[build_headtohead] wrote {out}", file=sys.stderr)
    print("\n".join(lines))


def emit_png(cells: dict, out: pathlib.Path, kind: str) -> None:
    """Write head-to-head.png (percentile bars) or arm-attribution.png.

    matplotlib is optional; if it's not installed the PNGs are skipped
    and the aggregator still emits JSON + text log.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[build_headtohead] matplotlib not installed; skipping {out.name}", file=sys.stderr)
        return

    if kind == "percentile-bars":
        fig, axes = plt.subplots(len(REGIMES), 1, figsize=(10, 10), sharey=False)
        percentile_names = ["mean", "median", "p95", "p99"]
        for ax, regime in zip(axes, REGIMES):
            x = list(range(len(percentile_names)))
            width = 0.2
            for i, scorer in enumerate(SCORERS):
                summary = cells.get(scorer, {}).get(regime, {})
                p = summary.get("percentiles") or {}
                heights = [p.get(name, 0.0) for name in percentile_names]
                ax.bar([xi + (i - 1.5) * width for xi in x], heights, width=width, label=scorer)
            ax.set_xticks(x, percentile_names)
            ax.set_title(f"arrival = {regime}")
            ax.set_ylabel("elapsed_seconds")
            ax.legend(loc="upper left")
        fig.suptitle("mrl head-to-head — 4 scorers × 3 regimes")
        fig.tight_layout()
    elif kind == "arm-attribution":
        fig, ax = plt.subplots(figsize=(12, 5))
        labels: list[str] = []
        share_a: list[float] = []
        share_b: list[float] = []
        for scorer in SCORERS:
            for regime in REGIMES:
                summary = cells.get(scorer, {}).get(regime, {})
                arm_share = summary.get("arm_share") or {}
                # "spoke-a" / "spoke-b" if the served_endpoint header uses those names;
                # otherwise fall back to fraction of "unknown".
                a = sum(v for k, v in arm_share.items() if "spoke-gemma-a" in k or "spoke-a" in k)
                b = sum(v for k, v in arm_share.items() if "spoke-gemma-b" in k or "spoke-b" in k)
                labels.append(f"{scorer}\n{regime}")
                share_a.append(a)
                share_b.append(b)
        x = list(range(len(labels)))
        ax.bar(x, share_a, label="spoke-a (TP=2)")
        ax.bar(x, share_b, bottom=share_a, label="spoke-b (TP=4)")
        ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("fraction served")
        ax.set_title("arm attribution — spoke-a (TP=2) vs spoke-b (TP=4) per cell")
        ax.legend()
        fig.tight_layout()
    else:
        raise ValueError(f"unknown kind: {kind}")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[build_headtohead] wrote {out}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=str, required=True,
                    help="dataset name (e.g. totrandom-tail472). Cells are "
                         "read from local_results/<dataset>/<regime>/ for "
                         "each of the four scorers.")
    ap.add_argument("--results-root", type=pathlib.Path, default=None,
                    help="mrl_study/local_results root. Default: "
                         "<script>/../local_results.")
    ap.add_argument("--out", type=pathlib.Path, required=True,
                    help="output directory for the head-to-head artefacts.")
    args = ap.parse_args()

    results_root = args.results_root or (
        pathlib.Path(__file__).resolve().parents[1] / "local_results"
    )
    dataset_root = results_root / args.dataset
    if not dataset_root.exists():
        print(f"ERROR: dataset root {dataset_root} not found", file=sys.stderr)
        return 1

    # All four scorers share one dataset root; scorer namespacing lives
    # in the RUN_TAG subdir inside each regime dir (e.g. mrl-poisson-<ds>/).
    root_dirs = {scorer: dataset_root for scorer in SCORERS}
    cells = build_cells(root_dirs, dataset=args.dataset)

    args.out.mkdir(parents=True, exist_ok=True)
    emit_json(cells, args.out / "head-to-head.json")
    emit_log( cells, args.out / "head-to-head-log.txt")
    emit_png( cells, args.out / "head-to-head.png",     kind="percentile-bars")
    emit_png( cells, args.out / "arm-attribution.png",  kind="arm-attribution")
    return 0


if __name__ == "__main__":
    sys.exit(main())
