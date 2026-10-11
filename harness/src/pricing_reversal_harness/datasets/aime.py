"""AIME dataset loader.

Emits (index, origin_query, prompt, ground_truth) rows using the same prompt
template the paper uses (data/consolidated/aime-hybrid-*.json).

Source: HuggingFace `AI-MO/aimo-validation-aime` train split (60 problems).
"""

from __future__ import annotations

from typing import Iterable

from ..config import DatasetSpec


def load_aime(spec: DatasetSpec, cache_dir: str) -> Iterable[dict]:
    from datasets import load_dataset  # lazy; only inside the harness image

    ds = load_dataset(spec.hf_source, split=spec.hf_split, cache_dir=cache_dir)
    q_field = spec.question_field or "problem"
    a_field = spec.answer_field or "answer"
    for i, row in enumerate(ds, start=1):
        origin = row[q_field]
        try:
            prompt = spec.prompt_template.format(question=origin)
        except (IndexError, KeyError) as exc:
            raise ValueError(
                f"aime prompt_template is malformed (literal braces must be "
                f"doubled): {exc}"
            ) from exc
        yield {
            "index": i,
            "origin_query": origin,
            "prompt": prompt,
            "ground_truth": str(row[a_field]),
        }
