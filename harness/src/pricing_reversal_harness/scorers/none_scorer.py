"""No-op scorer for datasets whose scoring is deferred.

Returns `(None, None)` for every (output, ground_truth) pair. Records
land in the journal with `score: null` and `prediction: null`, matching
the paper's `data/repeated_trial/*.json` shape.

Used for datasets like ARC-AGI-v1 and LiveCodeBench where automated
scoring requires infrastructure the harness doesn't ship (grid-exact
match, sandboxed code execution). Traces (prompts, responses, token
counts, costs) are still collected — scoring can be added in a
separate offline pass.
"""

from __future__ import annotations


def score_none(model_output: str, ground_truth: str) -> tuple[float | None, str | None]:
    return None, None
