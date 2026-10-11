"""Build the totrandom-wide-alternating-1000 dataset.

Filters the QD-alone n=2000 baseline journal by empirical elapsed
time to select the tail of the long-source distribution and the
head of the short-source distribution — widening the effective
service-time bimodality from ~3.8x to ~10x. The filtered records
are then interleaved 500 long + 500 short in strict alternation.

Purpose: rerun the QD-alone open-loop staged sanity check under a
workload where longs are actually long enough for the classical
queue-length-vs-work trap to compound.

Long filter:  source_dataset in {arc-agi-v1, gpqa, aime}
              AND baseline elapsed_seconds >= 150.
Short filter: source_dataset in {mmlupro, livecodebench, livemathbench}
              AND baseline elapsed_seconds <= 30.
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

DEFAULT_JRNL = Path("/Users/davidbr/git/clean/pricing-reversal/costguard_study/local_results/config-cgqd-alpha02-N8-totrandom2k/QD/journal/gemma-4-26b-a4b/totrandom.jsonl")
DEFAULT_SRC = Path(__file__).resolve().parents[1] / "src" / "pricing_reversal_harness" / "data" / "totrandom.json"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "src" / "pricing_reversal_harness" / "data" / "totrandom-wide-alternating-1000.json"


def build(jrnl_path: Path, src_path: Path, out_path: Path,
          total: int, long_min: float, short_max: float,
          seed: int) -> None:
    with src_path.open("r", encoding="utf-8") as f:
        src_doc = json.load(f)
    src_records = src_doc["records"]

    # Read baseline journal, filter, cross-reference back to source
    # totrandom by index to recover the full record schema.
    long_pool = []
    short_pool = []
    with jrnl_path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            idx = r["index"]
            src_ds = src_records[idx - 1]["source_dataset"]
            elapsed = r["extra_fields"].get("elapsed_seconds")
            if not isinstance(elapsed, (int, float)):
                continue
            if src_ds in LONGS and elapsed >= long_min:
                long_pool.append(src_records[idx - 1])
            elif src_ds in SHORTS and elapsed <= short_max:
                short_pool.append(src_records[idx - 1])

    print(f"filtered long_pool size:  {len(long_pool)} (from {LONGS}, elapsed >= {long_min}s)")
    print(f"filtered short_pool size: {len(short_pool)} (from {SHORTS}, elapsed <= {short_max}s)")

    if total % 2 != 0:
        raise SystemExit(f"total must be even (got {total})")
    n_long = total // 2
    n_short = total // 2
    if len(long_pool) == 0 or len(short_pool) == 0:
        raise SystemExit("empty pool after filter")
    if n_short > len(short_pool):
        raise SystemExit(
            f"asked for {n_short} shorts but pool has {len(short_pool)}"
        )

    rng = random.Random(seed)
    rng.shuffle(long_pool)
    rng.shuffle(short_pool)

    # Long pool typically < n_long, so wrap with modulo. Deepcopy each
    # record so index reassignment doesn't cross-contaminate.
    long_seq = [copy.deepcopy(long_pool[i % len(long_pool)]) for i in range(n_long)]
    short_seq = [copy.deepcopy(short_pool[i]) for i in range(n_short)]

    sliced = []
    for i in range(n_long):
        sliced.append(long_seq[i])
        sliced.append(short_seq[i])

    for i, rec in enumerate(sliced, start=1):
        rec["index"] = i

    out = {
        "dataset_name": "totrandom-wide-alternating-1000",
        "split": f"wide_alternating_{total}",
        "sources": src_doc.get("sources", []),
        "total": total,
        "seed": seed,
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "derived_from": (
            f"filtered QD-baseline journal: LONG=({sorted(LONGS)} & elapsed>={long_min}s), "
            f"SHORT=({sorted(SHORTS)} & elapsed<={short_max}s), strict alternation, long wrap"
        ),
        "long_sources": sorted(LONGS),
        "short_sources": sorted(SHORTS),
        "long_pool_unique": len(long_pool),
        "short_pool_unique": len(short_pool),
        "long_min_elapsed_s": long_min,
        "short_max_elapsed_s": short_max,
        "records": sliced,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    got_longs = sum(1 for r in sliced if r["source_dataset"] in LONGS)
    got_shorts = sum(1 for r in sliced if r["source_dataset"] in SHORTS)
    odd_all_long = all(sliced[i]["source_dataset"] in LONGS for i in range(0, total, 2))
    even_all_short = all(sliced[i]["source_dataset"] in SHORTS for i in range(1, total, 2))
    long_oq = [r["origin_query"] for r in sliced if r["source_dataset"] in LONGS]
    short_oq = [r["origin_query"] for r in sliced if r["source_dataset"] in SHORTS]
    print(f"\nwrote {out_path}")
    print(f"  total={total}  longs={got_longs}  shorts={got_shorts}")
    print(f"  unique long origin_queries: {len(set(long_oq))} of {len(long_oq)} (~{len(long_oq)/len(set(long_oq)):.1f}x wrap)")
    print(f"  unique short origin_queries: {len(set(short_oq))} of {len(short_oq)}")
    print(f"  odd-index all-long: {odd_all_long}")
    print(f"  even-index all-short: {even_all_short}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", type=Path, default=DEFAULT_JRNL)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--total", type=int, default=1000)
    ap.add_argument("--long-min", type=float, default=150.0,
                    help="Minimum elapsed_seconds for long-bucket records.")
    ap.add_argument("--short-max", type=float, default=30.0,
                    help="Maximum elapsed_seconds for short-bucket records.")
    ap.add_argument("--seed", type=int, default=271)
    args = ap.parse_args()
    build(args.journal, args.src, args.out, args.total,
          args.long_min, args.short_max, args.seed)


if __name__ == "__main__":
    main()
