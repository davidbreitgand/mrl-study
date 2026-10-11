"""totrandom-alternating-1000 dataset loader.

Reads ``pricing_reversal_harness/data/totrandom-alternating-1000.json``
— a 1000-row bundled slice that alternates long-source (arc-agi-v1,
gpqa, aime) and short-source (mmlupro, livecodebench, livemathbench)
records. 500 unique longs + 500 unique shorts in strict alternation.

Row shape matches totrandom's loader contract:

    {"index": int,             # 1..1000
     "origin_query": str,
     "prompt": str,
     "ground_truth": Any,
     "source_dataset": str}
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Iterable

from ..config import DatasetSpec


_PACKAGE_DATA = "pricing_reversal_harness.data"
_FILENAME = "totrandom-alternating-1000.json"


def load_totrandom_alternating_1000(spec: DatasetSpec, cache_dir: str) -> Iterable[dict]:
    del cache_dir

    src = resources.files(_PACKAGE_DATA).joinpath(_FILENAME)
    with src.open("r", encoding="utf-8") as fp:
        doc = json.load(fp)

    limit = spec.expected_count
    emitted = 0
    for rec in doc.get("records") or ():
        if limit is not None and emitted >= limit:
            break
        prompt = rec.get("prompt")
        origin_query = rec.get("origin_query")
        source_dataset = rec.get("source_dataset")
        if not isinstance(prompt, str) or not prompt:
            continue

        emitted += 1
        yield {
            "index": emitted,
            "origin_query": origin_query if isinstance(origin_query, str) else "",
            "prompt": prompt,
            "ground_truth": rec.get("ground_truth"),
            "source_dataset": source_dataset if isinstance(source_dataset, str) else "",
        }
