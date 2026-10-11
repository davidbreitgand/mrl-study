"""Deterministic scorers + a no-op scorer for datasets whose scoring
is deferred (ARC-AGI-v1, LiveCodeBench)."""

from typing import Any, Callable

from .boxed import score_boxed_integer
from .mcq import score_mcq_letter
from .none_scorer import score_none

Scorer = Callable[[str, str], tuple[float | None, str | None]]

_REGISTRY: dict[str, Scorer] = {
    "boxed_integer": score_boxed_integer,
    "mcq_letter": score_mcq_letter,
    "none": score_none,
}


# The totrandom dataset pools prompts from six source datasets that each
# want a different scorer. We route by the row's `source_dataset` field
# (populated by datasets/totrandom.py) at score time. Reading directly
# from the dataset id -> scorer name mapping in `datasets.yaml` would
# be cleaner in theory, but that config lives one layer above scorers/
# and would create an import cycle; a small hard-coded table here
# tracks the six source datasets we build totrandom from and matches
# their `scorer:` values in datasets.yaml exactly.
_TOTRANDOM_ROUTING = {
    "aime":           "boxed_integer",
    "gpqa":           "mcq_letter",
    "livemathbench":  "boxed_integer",
    "mmlupro":        "mcq_letter",
    "arc-agi-v1":     "none",
    "livecodebench":  "none",
}


def get_scorer(name: str) -> Scorer:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"no scorer registered under {name!r}") from exc


def get_totrandom_scorer_for(row: dict[str, Any]) -> Scorer:
    """Pick the right underlying scorer for a totrandom row based on
    its `source_dataset` field. Falls back to `score_none` if the
    field is missing or points at an unknown source — safer than
    guessing, matches the harness's "record traces, defer scoring"
    behaviour for arc-agi-v1 / livecodebench.

    Called only when the dataset's `scorer` is set to
    ``"dispatch_by_source_dataset"``. Every other dataset keeps
    binding its scorer once at run start, as before.
    """
    src = row.get("source_dataset")
    name = _TOTRANDOM_ROUTING.get(src) if isinstance(src, str) else None
    if name is None:
        return score_none
    return get_scorer(name)


def registered_scorer_names() -> set[str]:
    """Used by the config-lint preflight to catch scorer typos.

    Includes the meta scorer name ``dispatch_by_source_dataset``
    because config-lint validates dataset.scorer values against this
    set — totrandom's config entry uses that meta name."""
    return set(_REGISTRY.keys()) | {"dispatch_by_source_dataset"}
