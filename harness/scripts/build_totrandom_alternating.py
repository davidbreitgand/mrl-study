"""Build the totrandom-alternating dataset.

Partitions totrandom.json records by empirical service-time bucket
(long-source vs short-source), shuffles each bucket independently,
and emits an alternating [long, short, long, short, ...] sequence
of length --total.

Long bucket:  source_dataset in {arc-agi-v1, gpqa, aime}.
Short bucket: source_dataset in {mmlupro, livecodebench, livemathbench}.

The alternating ordering is adversarial to a queue-length-only picker:
under strict long/short alternation, QD's tie-break-random-shuffle
picker cannot phase-lock its decisions with the request-length
pattern, exposing (or refuting) the claim that queue-depth signals
mis-route service-time-variable workloads.
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
from pathlib import Path

LONGS = {"arc-agi-v1", "gpqa", "aime"}
SHORTS = {"mmlupro", "livecodebench", "livemathbench"}

DEFAULT_SRC = Path(__file__).resolve().parents[1] / "src" / "pricing_reversal_harness" / "data" / "totrandom.json"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "src" / "pricing_reversal_harness" / "data" / "totrandom-alternating.json"


def build(src_path: Path, out_path: Path, total: int, seed: int) -> None:
    with src_path.open("r", encoding="utf-8") as f:
        doc = json.load(f)

    longs = [r for r in doc["records"] if r.get("source_dataset") in LONGS]
    shorts = [r for r in doc["records"] if r.get("source_dataset") in SHORTS]

    rng = random.Random(seed)
    rng.shuffle(longs)
    rng.shuffle(shorts)

    if total % 2 != 0:
        raise SystemExit(f"total must be even (got {total})")
    n_long = total // 2
    n_short = total // 2
    if n_long > len(longs):
        raise SystemExit(f"asked for {n_long} longs but pool has {len(longs)}")
    if n_short > len(shorts):
        raise SystemExit(f"asked for {n_short} shorts but pool has {len(shorts)}")

    sliced = []
    for i in range(n_long):
        sliced.append(longs[i])
        sliced.append(shorts[i])

    for i, rec in enumerate(sliced, start=1):
        rec["index"] = i

    out = {
        "dataset_name": "totrandom-alternating",
        "split": f"alternating_{total}",
        "sources": doc.get("sources", []),
        "total": total,
        "seed": seed,
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "derived_from": "totrandom, alternating long-source and short-source records",
        "long_sources": sorted(LONGS),
        "short_sources": sorted(SHORTS),
        "records": sliced,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Verify
    got_longs = sum(1 for r in sliced if r["source_dataset"] in LONGS)
    got_shorts = sum(1 for r in sliced if r["source_dataset"] in SHORTS)
    origin_queries = {r["origin_query"] for r in sliced}
    odd_all_long = all(sliced[i]["source_dataset"] in LONGS for i in range(0, total, 2))
    even_all_short = all(sliced[i]["source_dataset"] in SHORTS for i in range(1, total, 2))
    print(f"wrote {out_path}")
    print(f"  total={total}  longs={got_longs}  shorts={got_shorts}")
    print(f"  unique origin_queries={len(origin_queries)} (expected {total})")
    print(f"  odd-index all-long: {odd_all_long}")
    print(f"  even-index all-short: {even_all_short}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--total", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build(args.src, args.out, args.total, args.seed)


if __name__ == "__main__":
    main()
