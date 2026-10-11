"""LiveCodeBench dataset loader.

Emits uniform rows for the paper's "livecodebench" dataset. Scoring is
`none` — code execution against hidden test cases requires a sandbox
the harness doesn't ship. Traces (prompts, responses, token counts,
costs) are collected; scoring is a separate offline pass.

Source: ``livecodebench/code_generation_lite``. Each row has:
  * ``question_content`` — the problem statement
  * ``starter_code`` — an optional function signature the answer must
    complete (some problems have this, others don't)
  * ground truth is a set of hidden tests we do not run

Loading strategy: pulls the release's JSONL shards directly via
``huggingface_hub``, deliberately bypassing the ``datasets`` library.
The upstream repo ships a script-backed loader
(``code_generation_lite.py``) that assembles ``release_v1``..
``release_v6`` from separate JSONL shards (``test.jsonl`` for v1, plus
``test2.jsonl``..``test6.jsonl`` for the additional versions). datasets
3.x rejects that script with a `trust_remote_code` deprecation warning
that turns fatal on some paths, and datasets 4.x removes script support
entirely — this loader sidesteps both by reading the shards directly.

With mmlupro (commit 1b42c94) and arc-agi-v1 (commit bda11e2) already
on the same pattern, livecodebench is the last loader that used to
depend on the `datasets<4.0` pin. Once this refactor lands, the pin
itself can be dropped from pyproject.toml.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..config import DatasetSpec


# Maps ``hf_config`` values ("release_v1"..."release_v6") to the ordered
# list of JSONL shards that make up that release. Verbatim from the
# upstream ``code_generation_lite.py``'s ``ALLOWED_FILES`` table —
# copied here so we don't need to fetch or execute the loader script.
# If LiveCodeBench ever adds ``release_v7``, add the entry here.
_RELEASE_SHARDS: dict[str, list[str]] = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl",
                   "test4.jsonl", "test5.jsonl"],
    "release_v6": ["test.jsonl", "test2.jsonl", "test3.jsonl",
                   "test4.jsonl", "test5.jsonl", "test6.jsonl"],
    "release_latest": ["test.jsonl", "test2.jsonl", "test3.jsonl",
                       "test4.jsonl", "test5.jsonl", "test6.jsonl"],
}


def _shards_for_release(hf_config: str | None) -> list[str]:
    """Resolve the release tag to its shard list. Missing/unknown tag
    raises loudly rather than silently picking a wrong slice."""
    if hf_config is None:
        raise ValueError(
            "livecodebench requires an explicit hf_config release tag "
            f"(one of {sorted(_RELEASE_SHARDS)}); got None"
        )
    try:
        return _RELEASE_SHARDS[hf_config]
    except KeyError as exc:
        raise ValueError(
            f"livecodebench has no shard mapping for hf_config={hf_config!r}; "
            f"known: {sorted(_RELEASE_SHARDS)}"
        ) from exc


def _iter_jsonl(path: Path) -> Iterable[dict]:
    """Yield one parsed JSON object per non-empty line."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_livecodebench(spec: DatasetSpec, cache_dir: str) -> Iterable[dict]:
    from huggingface_hub import hf_hub_download  # lazy

    shard_names = _shards_for_release(spec.hf_config)

    # Fetch each shard into cache_dir. hf_hub_download caches by digest
    # so re-fetches after the first are cheap. On a cold cache the six
    # shards total ~few tens of MB.
    local_paths: list[Path] = []
    for name in shard_names:
        p = hf_hub_download(
            repo_id=spec.hf_source,
            filename=name,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        local_paths.append(Path(p))

    limit = spec.expected_count
    emitted = 0

    for shard_path in local_paths:
        if limit is not None and emitted >= limit:
            break
        for row in _iter_jsonl(shard_path):
            if limit is not None and emitted >= limit:
                break
            question = row.get("question_content") or row.get("question")
            starter = row.get("starter_code") or ""
            # Ground truth is the hidden test suite — we surface a
            # stable identifier (problem_id / question_id / title) so
            # a downstream scoring pass can look up the tests.
            gt_id = (
                row.get("question_id")
                or row.get("problem_id")
                or row.get("question_title")
                or ""
            )
            if question is None:
                continue

            try:
                prompt = spec.prompt_template.format(
                    question=question,
                    starter_code=starter,
                )
            except (IndexError, KeyError) as exc:
                raise ValueError(
                    f"livecodebench prompt_template is malformed "
                    f"(literal braces must be doubled): {exc}"
                ) from exc

            emitted += 1
            yield {
                "index": emitted,
                "origin_query": question,
                "prompt": prompt,
                "ground_truth": str(gt_id),
            }
