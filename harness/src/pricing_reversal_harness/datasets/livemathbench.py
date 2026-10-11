"""LiveMathBench dataset loader.

Emits uniform rows for the paper's "livemathbench" dataset. Scoring is
`boxed_integer` (final answer wrapped in `\\boxed{...}` — same shape as
AIME). The HF dataset uses `question` for the prompt and `answer` for
the ground truth.
"""

from __future__ import annotations

from typing import Iterable

from ..config import DatasetSpec


def load_livemathbench(spec: DatasetSpec, cache_dir: str) -> Iterable[dict]:
    from datasets import load_dataset  # lazy; only inside the harness image

    if spec.hf_config:
        ds = load_dataset(spec.hf_source, spec.hf_config,
                          split=spec.hf_split, cache_dir=cache_dir)
    else:
        ds = load_dataset(spec.hf_source, split=spec.hf_split, cache_dir=cache_dir)

    q_field = spec.question_field or "question"
    a_field = spec.answer_field or "answer"

    for i, row in enumerate(ds, start=1):
        origin = row.get(q_field) or row.get("problem")
        answer = row.get(a_field)
        if origin is None or answer is None:
            # Skip malformed rows — surface a WARNING via the caller's log,
            # never abort the whole dataset load.
            continue
        try:
            prompt = spec.prompt_template.format(question=origin)
        except (IndexError, KeyError) as exc:
            raise ValueError(
                f"livemathbench prompt_template is malformed "
                f"(literal braces must be doubled): {exc}"
            ) from exc
        yield {
            "index": i,
            "origin_query": origin,
            "prompt": prompt,
            "ground_truth": str(answer),
        }
