"""GPQA dataset loader.

Emits (index, origin_query, prompt, ground_truth) rows. The GPQA test rows
contain a correct answer plus three distractors — the loader randomizes
the choice ordering with a fixed per-query seed so each model sees the
same option letter assignments, then records the shuffled letter as
ground_truth.
"""

from __future__ import annotations

import random
from typing import Iterable

from ..config import DatasetSpec


_LETTERS = ["A", "B", "C", "D"]


def load_gpqa(spec: DatasetSpec, cache_dir: str) -> Iterable[dict]:
    from datasets import load_dataset  # lazy

    if spec.hf_config:
        ds = load_dataset(spec.hf_source, spec.hf_config, split=spec.hf_split, cache_dir=cache_dir)
    else:
        ds = load_dataset(spec.hf_source, split=spec.hf_split, cache_dir=cache_dir)

    for i, row in enumerate(ds, start=1):
        question = row.get("Question") or row.get("question") or ""
        correct = row.get("Correct Answer") or row.get("correct_answer")
        incorrect = [
            row.get("Incorrect Answer 1") or row.get("incorrect_answer_1"),
            row.get("Incorrect Answer 2") or row.get("incorrect_answer_2"),
            row.get("Incorrect Answer 3") or row.get("incorrect_answer_3"),
        ]
        incorrect = [x for x in incorrect if x is not None]
        if correct is None or len(incorrect) < 3:
            continue

        choices = [correct, *incorrect[:3]]
        rng = random.Random(i)  # deterministic per-index seed
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [choices[k] for k in order]
        gt_letter = _LETTERS[order.index(0)]

        prompt = spec.prompt_template.format(
            question=question,
            choice_a=shuffled[0],
            choice_b=shuffled[1],
            choice_c=shuffled[2],
            choice_d=shuffled[3],
        )
        yield {
            "index": i,
            "origin_query": question,
            "prompt": prompt,
            "ground_truth": gt_letter,
        }
