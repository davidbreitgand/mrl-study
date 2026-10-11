"""totrandom dataset loader.

Reads the pre-built, shuffled prompt set at
``pricing_reversal_harness/data/totrandom.json`` (bundled as package data
by pyproject.toml, so the image gets a copy at install time). Emits one
row per record, in the file's stored order.

The file is produced by ``openshift_study/scripts/build-totrandom.py``
from the paper's traces under ``pricing-reversal/data/consolidated/``.
Each record already contains the fully-formed prompt (template applied
at build time by the source dataset's own loader), so THIS loader is
deliberately trivial: no HF download, no prompt_template.format(), no
per-item scoring dispatch. Prompt bodies pass through byte-identical.

Row shape emitted:

    {"index": int,             # 1..N, matches record["index"] as built
     "origin_query": str,      # underlying question text, template-free
     "prompt": str,            # fully-formed prompt sent to the model
     "ground_truth": Any,      # copied from the source dataset
     "source_dataset": str}    # e.g. "aime" — used by the dispatch
                               # scorer to route to the right scorer

The `source_dataset` field is emitted alongside the harness's standard
row shape; DatasetRunner ignores fields it doesn't know, but
scorers/dispatch.py reads it to route per-item scoring.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Iterable

from ..config import DatasetSpec


_PACKAGE_DATA = "pricing_reversal_harness.data"
_TOTRANDOM_FILENAME = "totrandom.json"


def load_totrandom(spec: DatasetSpec, cache_dir: str) -> Iterable[dict]:
    # `cache_dir` is unused — the dataset is bundled in the image. We
    # accept the parameter to match the loader interface every other
    # loader in this package exposes.
    del cache_dir

    # importlib.resources.files() -> Traversable pointing at the
    # package data root. Works whether the package is installed as a
    # wheel, editable-installed, or run from a source checkout.
    src = resources.files(_PACKAGE_DATA).joinpath(_TOTRANDOM_FILENAME)
    with src.open("r", encoding="utf-8") as fp:
        doc = json.load(fp)

    limit = spec.expected_count  # None = take all
    emitted = 0
    for rec in doc.get("records") or ():
        if limit is not None and emitted >= limit:
            break
        prompt = rec.get("prompt")
        origin_query = rec.get("origin_query")
        source_dataset = rec.get("source_dataset")
        if not isinstance(prompt, str) or not prompt:
            # Malformed record — skip. build-totrandom.py should never
            # emit these, but if the file is edited by hand we don't
            # want to crash the whole study on one bad row.
            continue

        emitted += 1
        yield {
            "index": emitted,
            "origin_query": origin_query if isinstance(origin_query, str)
                            else "",
            "prompt": prompt,
            "ground_truth": rec.get("ground_truth"),
            "source_dataset": source_dataset if isinstance(source_dataset, str)
                              else "",
        }
