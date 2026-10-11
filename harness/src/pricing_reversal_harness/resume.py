"""Cross-study resume — reuse a prior study's completed datasets.

Called by the orchestrator before each `_run_consolidated` and
`_run_repeated_trial`. If a valid seed file for the (model, dataset[,
run_index]) tuple is already present in this study's `results_dir`
(having been copied there by the uber's stage-to-PVC step or by a
prior partial run), the orchestrator skips the actual run and emits a
`*_skipped_resume` phase.

A "valid seed" satisfies THREE checks:

  1. File exists and parses as JSON with a `records` list.
  2. `len(records)` matches `expected_count` from the dataset spec —
     truncated / partial files are rejected so downstream aggregates
     don't silently under-count.
  3. `data_fingerprint` (SHA-256 of the sorted (index, prompt) pairs)
     matches what a freshly-loaded dataset would produce — catches
     loader-refactor drift, prompt_template changes, and any other
     input-side edit between studies.

Any check failing => the seed file is removed (invalid seed is worse
than no seed) and the caller runs the dataset from scratch.

Seed files live at their normal per-study paths — same location the
current study will WRITE its own output — so a successful resume-from
run produces one merged `consolidated/` tree that `make results`
retrieves without any special handling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .journal import data_fingerprint


def _load_seed(path: Path) -> dict[str, Any] | None:
    """Return the parsed JSON, or None if the file is missing / invalid."""
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as fp:
            doc = json.load(fp)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(doc, dict) or not isinstance(doc.get("records"), list):
        return None
    return doc


def _remove_invalid(path: Path) -> None:
    """Delete a seed file that failed validation. Best-effort: filesystem
    errors are ignored so the caller can proceed with a fresh run."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def check_consolidated_seed(
    seed_path: Path,
    *,
    fresh_rows: list[dict[str, Any]],
    expected_count: int | None,
) -> tuple[bool, str]:
    """Decide whether a consolidated seed at `seed_path` can be trusted.

    Returns ``(ok, reason)``. On ``ok=True`` the caller skips its
    consolidated run entirely. On ``ok=False`` the seed file is
    removed and ``reason`` names the failure mode for stderr logging.

    `fresh_rows` is the freshly-loaded dataset rows for THIS study —
    used to compute the fingerprint we compare against the seed's
    stored `data_fingerprint`.
    """
    doc = _load_seed(seed_path)
    if doc is None:
        return False, "missing or malformed"

    # (2) counts check. `counts` is the paper's aggregate; fall back to
    # len(records) if the field is absent on an older seed.
    records = doc["records"]
    stored_counts = doc.get("counts")
    actual_counts = int(stored_counts) if isinstance(stored_counts, int) else len(records)
    if expected_count is not None and actual_counts != expected_count:
        _remove_invalid(seed_path)
        return False, f"counts={actual_counts} != expected {expected_count}"
    if actual_counts != len(records):
        _remove_invalid(seed_path)
        return False, (f"stated counts={actual_counts} != len(records)="
                       f"{len(records)} (truncated file)")

    # (3) fingerprint check.
    stored_fp = doc.get("data_fingerprint")
    if not isinstance(stored_fp, str) or not stored_fp:
        _remove_invalid(seed_path)
        return False, "missing data_fingerprint"
    fresh_fp = data_fingerprint(fresh_rows)
    if stored_fp != fresh_fp:
        _remove_invalid(seed_path)
        return False, (f"data_fingerprint mismatch: stored={stored_fp[:12]}..., "
                       f"fresh={fresh_fp[:12]}... (loader/template drift)")

    return True, "match"


def check_repeated_trial_seed(
    seed_path: Path,
    *,
    fresh_rows: list[dict[str, Any]],
    expected_count: int | None,
) -> tuple[bool, str]:
    """Same rules as consolidated seeds. Repeated-trial files carry the
    same fields (records[], counts, data_fingerprint) so the same
    validation covers both. Named separately in case per-run rules
    diverge in the future."""
    return check_consolidated_seed(
        seed_path,
        fresh_rows=fresh_rows,
        expected_count=expected_count,
    )
