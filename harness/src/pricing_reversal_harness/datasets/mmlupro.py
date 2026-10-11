"""MMLU-Pro dataset loader.

Emits uniform rows for the paper's "mmlupro" dataset. Scoring is
`mcq_letter` (extended to A-J to cover MMLU-Pro's 10-option format).

Source: `TIGER-Lab/MMLU-Pro`. Each row has a `question`, an `options`
list of up to 10 strings, and an `answer` letter (A-J). Some rows carry
`answer_index` as an integer 0-9 — used as a fallback when `answer` is
absent.

The loader consumes the FIRST `expected_count` rows in row order for
determinism across models. The paper's `mmlupro-test_3000` slice
corresponds to rows 0..2999 in the test split.

Loading strategy: pulls the parquet shard directly via `huggingface_hub`
and reads it with `pyarrow`, deliberately bypassing the `datasets`
library. This keeps mmlupro independent of the `datasets<4.0` pin that
livecodebench requires — the two libraries disagreed on the `List`
feature type before this refactor (mmlupro's descriptor uses `List`,
which datasets 3.x rejects).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..config import DatasetSpec


_LETTERS = list("ABCDEFGHIJ")

# TIGER-Lab/MMLU-Pro publishes one shard per split under this layout.
# If TIGER-Lab ever re-shards, the loader raises rather than picking a
# wrong subset — see _split_parquet_path.
_SPLIT_PATHS = {
    "test": "data/test-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
}


def _split_parquet_path(split: str) -> str:
    try:
        return _SPLIT_PATHS[split]
    except KeyError as exc:
        raise ValueError(
            f"mmlupro loader has no parquet path for split={split!r}; "
            f"known splits: {sorted(_SPLIT_PATHS)}"
        ) from exc


def load_mmlupro(spec: DatasetSpec, cache_dir: str) -> Iterable[dict]:
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
    table = pf.read()  # single-shard split — the whole file is what we want
    columns = {name: table.column(name).to_pylist() for name in table.column_names}
    n_rows = table.num_rows

    limit = spec.expected_count
    emitted = 0

    for i in range(n_rows):
        if limit is not None and emitted >= limit:
            break

        question = columns.get("question", [None] * n_rows)[i]
        options = columns.get("options", [None] * n_rows)[i] or []
        gt_letter = columns.get("answer", [None] * n_rows)[i]
        if not gt_letter:
            idx = columns.get("answer_index", [None] * n_rows)[i]
            if isinstance(idx, int) and 0 <= idx < len(_LETTERS):
                gt_letter = _LETTERS[idx]

        if question is None or not options or gt_letter is None:
            continue

        options_block = "\n".join(
            f"{_LETTERS[k]}) {opt}" for k, opt in enumerate(options[: len(_LETTERS)])
        )

        try:
            prompt = spec.prompt_template.format(
                question=question, options=options_block,
            )
        except (IndexError, KeyError) as exc:
            raise ValueError(
                f"mmlupro prompt_template is malformed "
                f"(literal braces must be doubled): {exc}"
            ) from exc

        emitted += 1
        yield {
            "index": emitted,
            "origin_query": question,
            "prompt": prompt,
            "ground_truth": str(gt_letter).upper(),
        }
