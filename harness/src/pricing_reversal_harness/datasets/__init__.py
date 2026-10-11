"""Dataset loaders yielding uniform (index, origin_query, prompt, ground_truth) rows."""

from __future__ import annotations

from typing import Iterable

from ..config import DatasetSpec
from .aime import load_aime
from .arc_agi_v1 import load_arc_agi_v1
from .gpqa import load_gpqa
from .livecodebench import load_livecodebench
from .livemathbench import load_livemathbench
from .mmlupro import load_mmlupro
from .totrandom import load_totrandom
from .totrandom_tail472 import load_totrandom_tail472
from .totrandom_alternating import load_totrandom_alternating
from .totrandom_alternating_1000 import load_totrandom_alternating_1000
from .totrandom_wide_alternating_1000 import load_totrandom_wide_alternating_1000


_LOADERS = {
    "aime": load_aime,
    "gpqa": load_gpqa,
    "arc-agi-v1": load_arc_agi_v1,
    "livemathbench": load_livemathbench,
    "mmlupro": load_mmlupro,
    "livecodebench": load_livecodebench,
    # Bundled shuffle of the six datasets above — for the costguard
    # study. Ships in the image; no HF download.
    "totrandom": load_totrandom,
    # 472-row slice of totrandom (rows 1529..2000 of the CG+QD 1.0/1.0
    # QD-alone journal, sustained-shock regime). Bundled in the image.
    "totrandom-tail472": load_totrandom_tail472,
    # 500-row alternating-long/short slice of totrandom for QD's
    # adversarial-workload sanity check. Bundled in the image.
    "totrandom-alternating": load_totrandom_alternating,
    # 1000-row extension for MID-only statistical-power follow-up.
    "totrandom-alternating-1000": load_totrandom_alternating_1000,
    # Wide-bimodality (~10x) variant filtered from the QD-baseline
    # journal; used to test whether the queue-length-vs-work trap
    # manifests when longs are truly long enough to phase-lock.
    "totrandom-wide-alternating-1000": load_totrandom_wide_alternating_1000,
}


def load_dataset(spec: DatasetSpec, cache_dir: str) -> Iterable[dict]:
    try:
        loader = _LOADERS[spec.id]
    except KeyError as exc:
        raise KeyError(f"no loader for dataset id {spec.id!r}") from exc
    return loader(spec, cache_dir)


def registered_dataset_ids() -> set[str]:
    """Used by the config-lint preflight to catch dataset-id typos."""
    return set(_LOADERS.keys())
