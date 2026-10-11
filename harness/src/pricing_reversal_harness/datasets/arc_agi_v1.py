"""ARC-AGI-v1 dataset loader.

Emits uniform rows for the paper's "arc-agi-v1" dataset (400 problems).
Scoring is `none` — grid-exact match is a follow-up scorer, not
currently implemented; the harness collects prompts, responses, and
token counts for offline scoring.

Source: the ARC-AGI-v1 evaluation set on the `lordspline/arc-agi`
HuggingFace mirror. Rows have `train` (few-shot input/output grid
pairs) and `test` (query grids). The mirror stores the ground-truth
grid as the `output` field of the first `test` entry — there is no
separate `solution` column on this mirror.

Loading strategy: pulls the parquet shard directly via `huggingface_hub`
and reads it with `pyarrow`, deliberately bypassing the `datasets`
library. This keeps arc-agi-v1 independent of the `datasets<4.0` pin
that livecodebench requires — `lordspline/arc-agi`'s descriptor uses
the `List` feature type introduced in datasets 4.x, which the 3.x pin
rejects with `ValueError: Feature type 'List' not found`. Same
solution mmlupro adopted.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from ..config import DatasetSpec


# lordspline/arc-agi publishes one shard per split under this layout.
# If the mirror ever re-shards or renames splits, the loader raises
# rather than picking a wrong file — see _split_parquet_path.
_SPLIT_PATHS = {
    "evaluation": "data/evaluation-00000-of-00001.parquet",
    "training":   "data/training-00000-of-00001.parquet",
    "trial":      "data/trial-00000-of-00001.parquet",
}


def _split_parquet_path(split: str) -> str:
    try:
        return _SPLIT_PATHS[split]
    except KeyError as exc:
        raise ValueError(
            f"arc-agi-v1 loader has no parquet path for split={split!r}; "
            f"known splits: {sorted(_SPLIT_PATHS)}"
        ) from exc


def _grid_to_str(g: Any) -> str:
    """Serialize a grid as a JSON-encoded 2-D array — the canonical
    ARC-AGI representation."""
    return json.dumps(g)


def load_arc_agi_v1(spec: DatasetSpec, cache_dir: str) -> Iterable[dict]:
    from huggingface_hub import hf_hub_download  # lazy
    # Use pyarrow.parquet.ParquetFile directly rather than the higher-level
    # read_table — the latter goes through pyarrow.dataset which imports
    # pandas, which triggers an unrelated NumPy 1/2 ABI check on some
    # machines. ParquetFile is a thin binding and needs neither.
    from pyarrow.parquet import ParquetFile  # lazy

    filename = _split_parquet_path(spec.hf_split)
    local_path = hf_hub_download(
        repo_id=spec.hf_source,
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )

    pf = ParquetFile(local_path)
    table = pf.read()
    columns = {name: table.column(name).to_pylist() for name in table.column_names}
    n_rows = table.num_rows

    for i in range(n_rows):
        # Row shape on lordspline/arc-agi:
        #   train: list of {"input": grid, "output": grid}
        #   test:  list of {"input": grid, "output": grid}
        # Test's `output` IS the ground-truth grid; no separate `solution`
        # column on this mirror. Some other ARC-AGI mirrors carry a top-
        # level `solution` field — we prefer that if it ever appears here,
        # else fall back to test[0].output.
        train_pairs = columns.get("train", [None] * n_rows)[i] or []
        test_pairs = columns.get("test", [None] * n_rows)[i] or []
        solution = columns.get("solution", [None] * n_rows)[i] \
            if "solution" in columns else None

        if not test_pairs:
            continue

        first_test = test_pairs[0]
        test_input = first_test.get("input") if isinstance(first_test, dict) else first_test

        if solution is not None:
            gt_grid = solution
        elif isinstance(first_test, dict):
            gt_grid = first_test.get("output")
        else:
            gt_grid = None

        # Render the few-shot examples.
        train_block_parts = []
        for j, pair in enumerate(train_pairs, start=1):
            inp = pair.get("input") if isinstance(pair, dict) else None
            out = pair.get("output") if isinstance(pair, dict) else None
            if inp is None or out is None:
                continue
            train_block_parts.append(
                f"Example {j} input:\n{_grid_to_str(inp)}\n"
                f"Example {j} output:\n{_grid_to_str(out)}"
            )
        train_block = "\n\n".join(train_block_parts)
        test_block = _grid_to_str(test_input)

        try:
            prompt = spec.prompt_template.format(
                train_examples=train_block, test_input=test_block,
            )
        except (IndexError, KeyError) as exc:
            raise ValueError(
                f"arc-agi-v1 prompt_template is malformed "
                f"(literal braces must be doubled): {exc}"
            ) from exc

        yield {
            "index": i + 1,
            "origin_query": test_block,
            "prompt": prompt,
            # Ground truth stays as the raw grid JSON so a future scorer
            # can parse + compare against the model's output.
            "ground_truth": _grid_to_str(gt_grid),
        }
