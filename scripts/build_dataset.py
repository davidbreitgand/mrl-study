#!/usr/bin/env python3
"""Build a `totrandom` mixed-benchmark dataset from six public sources.

Produces a JSON file compatible with the harness's `data/*.json` schema
by pulling records from six HuggingFace benchmarks, formatting each into
a full evaluation prompt, shuffling with a deterministic seed, and
writing the result plus an optional tail slice.

The six source benchmarks and their target record counts (matching the
composition of the study's canonical totrandom.json):

    Source            Rows    HuggingFace ID
    -----------------------------------------------------------------
    mmlupro           3000    TIGER-Lab/MMLU-Pro
    livecodebench     1055    livecodebench/code_generation_lite
    arc-agi-v1        400     dataartsciensocietyIT/arc-agi-v1
    gpqa              198     Idavidrein/gpqa (subset: gpqa_main)
    livemathbench     121     opencompass/LiveMathBench
    aime              60      Maxwell-Jia/AIME_2024
                    ------
    Total             4834

The builder writes:

    <out-dir>/totrandom.json           full 4834-row shuffle (seed=42)
    <out-dir>/totrandom-tail472.json   the "sustained-shock" 472-row tail
                                       (records[1529..2000] of the shuffle)

The tail slice covers the same 472 records the study's replicated
Poisson head-to-head measured against. Its `derived_from` metadata
records the provenance.

Reproducibility note: the canonical `totrandom.json` used by the study
was built on 2026-09-08 from the versions of those benchmarks
published at that time. HuggingFace datasets are versioned, and the
per-benchmark record ordering the study's builder used may not match
the ordering this script produces from today's dataset revisions.
Byte-for-byte reproducibility of the original prompt strings is NOT
guaranteed. Statistical reproducibility (same distribution of prompt
lengths, same benchmark mix) IS.

If exact-prompt reproducibility matters (e.g., to match published p99
figures within their sample-variance band), pin each `datasets.load_dataset`
call to a specific `revision=` and record the SHA in the output metadata.

Attribution: each source benchmark has its own license. Users of this
builder are responsible for compliance with the source-license terms.
The generated totrandom.json is a derivative work of six independent
benchmarks; redistributing it requires meeting each source's
redistribution requirements. Providing this builder — instead of
redistributing prebuilt prompts — is the intentional way to hand
license responsibility to the operator.

Usage:
    scripts/build_dataset.py \\
        --out-dir harness/src/pricing_reversal_harness/data/

    scripts/build_dataset.py --seed 42 --tail-slice 1529 2000 \\
        --out-dir /tmp/dataset-cache/

Dependencies:
    pip install datasets

Exit codes:
    0   totrandom.json + totrandom-tail472.json written.
    1   argument error.
    2   dependency missing (datasets library not installed).
    3   one or more source benchmarks failed to load.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import random
import sys
from typing import Callable


SOURCES = [
    # (short_name, target_rows, hf_id, subset_or_None)
    ("mmlupro",       3000, "TIGER-Lab/MMLU-Pro",                None),
    ("livecodebench", 1055, "livecodebench/code_generation_lite", None),
    ("arc-agi-v1",     400, "dataartsciensocietyIT/arc-agi-v1",   None),
    ("gpqa",           198, "Idavidrein/gpqa",                    "gpqa_main"),
    ("livemathbench",  121, "opencompass/LiveMathBench",          None),
    ("aime",            60, "Maxwell-Jia/AIME_2024",              None),
]


def format_mmlupro(row: dict) -> tuple[str, str, str]:
    """Return (origin_query, prompt, ground_truth) for one MMLU-Pro row."""
    question = row.get("question", "")
    options = row.get("options") or []
    answer = row.get("answer", "")
    letters = "ABCDEFGHIJ"[:len(options)]
    opts_block = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))
    subject = row.get("category") or row.get("subject") or "general"
    prompt = (
        f"Answer the following {subject} question. The last line of your "
        f"response should be of the following format: 'Answer: $LETTER' "
        f"(without quotes) where LETTER is one of {letters}.\n\n"
        f"{question}\n\n{opts_block}\n\nLet's think step by step."
    )
    return question, prompt, answer


def format_livecodebench(row: dict) -> tuple[str, str, str]:
    q = row.get("question_content") or row.get("question") or ""
    starter = row.get("starter_code") or ""
    prompt = (
        f"Solve the following programming problem. Provide a complete, "
        f"working solution.\n\n{q}\n\n"
        f"Starter code:\n```\n{starter}\n```"
    )
    gt = row.get("public_test_cases") or ""
    return q[:200], prompt, str(gt)[:200]


def format_arc_agi_v1(row: dict) -> tuple[str, str, str]:
    train_pairs = row.get("train") or []
    test_input = row.get("test") or []
    prompt = (
        "You are solving an ARC-AGI reasoning task. Study the "
        "input/output examples below and predict the output for the "
        "test input.\n\n"
    )
    for i, pair in enumerate(train_pairs):
        prompt += f"Example {i+1}:\nInput: {pair.get('input')}\nOutput: {pair.get('output')}\n\n"
    if test_input:
        prompt += f"Test input: {test_input[0].get('input') if test_input else ''}\nTest output:"
    gt = json.dumps(test_input[0].get("output")) if test_input else ""
    return f"arc-agi-{row.get('task_id', 'unknown')}", prompt, gt


def format_gpqa(row: dict) -> tuple[str, str, str]:
    question = row.get("Question", "")
    choices = [
        row.get("Correct Answer", ""),
        row.get("Incorrect Answer 1", ""),
        row.get("Incorrect Answer 2", ""),
        row.get("Incorrect Answer 3", ""),
    ]
    letters = "ABCD"
    opts_block = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    prompt = (
        f"Answer the following graduate-level physics/chemistry/biology "
        f"question. The last line of your response should be of the "
        f"format: 'Answer: $LETTER' (without quotes) where LETTER is "
        f"one of A, B, C, D.\n\n{question}\n\n{opts_block}\n\n"
        f"Let's think step by step."
    )
    return question, prompt, "A"


def format_livemathbench(row: dict) -> tuple[str, str, str]:
    q = row.get("question") or row.get("problem") or ""
    answer = row.get("answer") or ""
    prompt = (
        f"Solve the following math problem. Show your work and give the "
        f"final numerical answer.\n\n{q}"
    )
    return q[:200], prompt, str(answer)


def format_aime(row: dict) -> tuple[str, str, str]:
    q = row.get("Problem") or row.get("problem") or ""
    answer = row.get("Answer") or row.get("answer") or ""
    prompt = (
        f"Solve the following AIME problem. The answer is an integer "
        f"between 0 and 999. Give your final answer on the last line as "
        f"'Answer: $NUMBER'.\n\n{q}"
    )
    return q[:200], prompt, str(answer)


FORMATTERS: dict[str, Callable[[dict], tuple[str, str, str]]] = {
    "mmlupro":       format_mmlupro,
    "livecodebench": format_livecodebench,
    "arc-agi-v1":    format_arc_agi_v1,
    "gpqa":          format_gpqa,
    "livemathbench": format_livemathbench,
    "aime":          format_aime,
}


def load_source(short_name: str, target_rows: int, hf_id: str, subset: str | None) -> list[dict]:
    """Load `target_rows` records from a source benchmark."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("ERROR: `datasets` library not installed. Run: pip install datasets",
              file=sys.stderr)
        sys.exit(2)

    print(f"[build_dataset] loading {short_name} ({hf_id}{'/' + subset if subset else ''})",
          file=sys.stderr)
    try:
        if subset:
            ds = load_dataset(hf_id, subset, split="test")
        else:
            ds = load_dataset(hf_id, split="test")
    except Exception as e:
        print(f"[build_dataset] WARN: {short_name} 'test' split failed ({e}); trying 'train'",
              file=sys.stderr)
        try:
            if subset:
                ds = load_dataset(hf_id, subset, split="train")
            else:
                ds = load_dataset(hf_id, split="train")
        except Exception as e2:
            print(f"[build_dataset] ERROR: {short_name} failed to load: {e2}",
                  file=sys.stderr)
            return []

    formatter = FORMATTERS[short_name]
    rows: list[dict] = []
    for i, raw in enumerate(ds):
        if len(rows) >= target_rows:
            break
        try:
            origin_query, prompt, ground_truth = formatter(raw)
        except Exception as e:
            print(f"[build_dataset] WARN: {short_name}[{i}] failed to format: {e}",
                  file=sys.stderr)
            continue
        rows.append({
            "origin_query":   origin_query,
            "prompt":         prompt,
            "ground_truth":   ground_truth,
            "source_index":   i,
            "source_dataset": short_name,
        })

    if len(rows) < target_rows:
        print(f"[build_dataset] WARN: {short_name} produced only {len(rows)} rows "
              f"(wanted {target_rows}); benchmark may have fewer records than expected",
              file=sys.stderr)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True,
                    help="directory to write totrandom.json + totrandom-tail472.json")
    ap.add_argument("--seed", type=int, default=42,
                    help="shuffle seed (default 42, matches the study's canonical build)")
    ap.add_argument("--tail-slice", type=int, nargs=2, default=[1529, 2000],
                    metavar=("START", "END"),
                    help="slice [start:end) of the shuffled records to write as "
                         "totrandom-tail472.json (default 1529 2000)")
    ap.add_argument("--skip-tail", action="store_true",
                    help="skip writing the tail slice; only write totrandom.json")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    failed_sources: list[str] = []
    per_source_counts: dict[str, int] = {}
    for short_name, target_rows, hf_id, subset in SOURCES:
        rows = load_source(short_name, target_rows, hf_id, subset)
        if not rows:
            failed_sources.append(short_name)
            continue
        all_records.extend(rows)
        per_source_counts[short_name] = len(rows)

    if failed_sources:
        print(f"[build_dataset] ERROR: failed to load: {failed_sources}", file=sys.stderr)
        return 3

    print(f"[build_dataset] loaded {len(all_records)} records total; shuffling seed={args.seed}",
          file=sys.stderr)
    rng = random.Random(args.seed)
    rng.shuffle(all_records)

    for i, r in enumerate(all_records):
        r["index"] = i

    now_utc = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    totrandom = {
        "dataset_name": "totrandom",
        "split":        "shuffled",
        "sources":      per_source_counts,
        "total":        len(all_records),
        "seed":         args.seed,
        "built_at":     now_utc,
        "records":      all_records,
    }
    totrandom_path = args.out_dir / "totrandom.json"
    with totrandom_path.open("w") as f:
        json.dump(totrandom, f, indent=2)
    print(f"[build_dataset] wrote {totrandom_path} ({len(all_records)} records)", file=sys.stderr)

    if args.skip_tail:
        return 0

    start, end = args.tail_slice
    if start < 0 or end > len(all_records) or start >= end:
        print(f"[build_dataset] ERROR: tail slice [{start}, {end}) invalid for "
              f"{len(all_records)}-row shuffle", file=sys.stderr)
        return 1
    tail_records = []
    for i, r in enumerate(all_records[start:end]):
        r2 = dict(r)
        r2["totrandom_index"] = start + i
        r2["index"] = i
        tail_records.append(r2)

    tail = {
        "dataset_name": "totrandom-tail",
        "split":        "shuffled",
        "sources":      per_source_counts,
        "total":        len(tail_records),
        "seed":         args.seed,
        "built_at":     now_utc,
        "derived_from": f"records[{start}:{end}) of the seed={args.seed} shuffle of totrandom",
        "records":      tail_records,
    }
    tail_name = f"totrandom-tail{len(tail_records)}.json"
    tail_path = args.out_dir / tail_name
    with tail_path.open("w") as f:
        json.dump(tail, f, indent=2)
    print(f"[build_dataset] wrote {tail_path} ({len(tail_records)} records)", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
