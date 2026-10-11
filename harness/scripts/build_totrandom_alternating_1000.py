"""Build the totrandom-alternating-1000 dataset.

1000-row extension of totrandom-alternating: 500 long-source records
alternating with 500 short-source records. The long pool has 287
unique records (arc-agi-v1, gpqa, aime), so long records are reused
with wrap-around (~1.7x per long record on average). The short pool
has 4547 records, so no short reuse is required.

Adversarial ordering for a queue-length-only picker; see
build_totrandom_alternating.py for rationale and the QD
adversarial-workload sanity-check plan.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import random
from pathlib import Path

LONGS = {"arc-agi-v1", "gpqa", "aime"}
SHORTS = {"mmlupro", "livecodebench", "livemathbench"}

DEFAULT_SRC = Path(__file__).resolve().parents[1] / "src" / "pricing_reversal_harness" / "data" / "totrandom.json"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "src" / "pricing_reversal_harness" / "data" / "totrandom-alternating-1000.json"


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
    if n_short > len(shorts):
        raise SystemExit(f"asked for {n_short} shorts but pool has {len(shorts)}")

    # Long records use wrap-around; shorts must be unique.
    long_seq = [copy.deepcopy(longs[i % len(longs)]) for i in range(n_long)]
    short_seq = [copy.deepcopy(shorts[i]) for i in range(n_short)]

    sliced = []
    for i in range(n_long):
        sliced.append(long_seq[i])
        sliced.append(short_seq[i])

    for i, rec in enumerate(sliced, start=1):
        rec["index"] = i

    out = {
        "dataset_name": "totrandom-alternating-1000",
        "split": f"alternating_{total}",
        "sources": doc.get("sources", []),
        "total": total,
        "seed": seed,
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "derived_from": "totrandom, alternating long-source and short-source; long records reused with wrap",
        "long_sources": sorted(LONGS),
        "short_sources": sorted(SHORTS),
        "long_pool_unique": len(longs),
        "short_pool_unique": len(shorts),
        "records": sliced,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    got_longs = sum(1 for r in sliced if r["source_dataset"] in LONGS)
    got_shorts = sum(1 for r in sliced if r["source_dataset"] in SHORTS)
    odd_all_long = all(sliced[i]["source_dataset"] in LONGS for i in range(0, total, 2))
    even_all_short = all(sliced[i]["source_dataset"] in SHORTS for i in range(1, total, 2))
    long_origin_queries = [r["origin_query"] for r in sliced if r["source_dataset"] in LONGS]
    short_origin_queries = [r["origin_query"] for r in sliced if r["source_dataset"] in SHORTS]
    print(f"wrote {out_path}")
    print(f"  total={total}  longs={got_longs}  shorts={got_shorts}")
    print(f"  unique long origin_queries: {len(set(long_origin_queries))} of {len(long_origin_queries)}")
    print(f"  unique short origin_queries: {len(set(short_origin_queries))} of {len(short_origin_queries)}")
    print(f"  odd-index all-long: {odd_all_long}")
    print(f"  even-index all-short: {even_all_short}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--total", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=137)
    args = ap.parse_args()
    build(args.src, args.out, args.total, args.seed)


if __name__ == "__main__":
    main()
