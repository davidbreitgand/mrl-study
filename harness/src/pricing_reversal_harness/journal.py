"""Crash-safe per-query journaling.

One JSONL file per (model, dataset[, run_index]) under
<results_dir>/journal/<model>/ where <results_dir> is the caller's
directory (worker: /results/<study-id>/workers/<model>/). Every
completed query is appended and fsynced.

On restart, `load_completed_indices` reads the journal and returns the set
of query indices already recorded, so the runner resumes from the next one.

`finalize_consolidated` and `finalize_repeated_trial` fold a completed
journal into the paper-shaped output JSON at atomic move time.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Iterable


class Journal:
    """Append-only JSONL journal for one (model, dataset[, run_index])."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        """Append and fsync a single JSON record."""

        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fp:
                fp.write(line)
                fp.flush()
                os.fsync(fp.fileno())

    def load_records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    # A partial line from a crash; skip it. Full-line + fsync
                    # writes make this rare, but we defend against it anyway.
                    continue
        return out

    def load_completed_indices(self) -> set[int]:
        return {int(r["index"]) for r in self.load_records() if "index" in r}


def journal_path(results_dir: Path, model_id: str, dataset_id: str,
                 run_index: int | None = None) -> Path:
    base = results_dir / "journal" / model_id
    if run_index is None:
        return base / f"{dataset_id}.jsonl"
    return base / f"{dataset_id}-run{run_index}.jsonl"


def _atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        json.dump(obj, fp, ensure_ascii=False, indent=2)
        fp.flush()
        os.fsync(fp.fileno())
    tmp.replace(path)


def data_fingerprint(records: Iterable[dict[str, Any]]) -> str:
    """SHA-256 of the sorted list of (index, prompt) pairs — mirrors the
    paper's `data_fingerprint` field.

    Publicly exported so cross-study resume logic can compute the same
    fingerprint from a freshly-loaded dataset and compare against the
    one stamped into a prior study's consolidated JSON.
    """

    payload = [
        {"index": int(r["index"]), "prompt": r.get("prompt", "")}
        for r in sorted(records, key=lambda r: int(r["index"]))
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


# Back-compat alias; existing call sites in this module still reference
# the leading-underscore name.
_data_fingerprint = data_fingerprint


def finalize_consolidated(
    journal: Journal,
    out_path: Path,
    *,
    model_name: str,
    dataset_name: str,
    split: str | None,
    elapsed_seconds: float,
    extra_metrics: dict[str, Any] | None = None,
) -> Path:
    """Aggregate journal into a paper-shaped consolidated JSON file."""

    records = sorted(journal.load_records(), key=lambda r: int(r["index"]))
    prompt_tokens = sum(int(r.get("prompt_tokens") or 0) for r in records)
    completion_tokens = sum(int(r.get("completion_tokens") or 0) for r in records)
    cost = sum(float(r.get("cost") or 0.0) for r in records)
    scores = [r["score"] for r in records if r.get("score") is not None]
    performance = (sum(scores) / len(scores)) if scores else 0.0

    extras = dict(extra_metrics or {})
    extras["gpu_hour_cost_total"] = sum(
        float((r.get("extra_fields") or {}).get("gpu_hour_cost") or 0.0)
        for r in records
    )

    payload = {
        "performance": performance,
        "time_taken": elapsed_seconds,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost": cost,
        "counts": len(records),
        "model_name": model_name,
        "dataset_name": dataset_name,
        "split": split,
        "demo": False,
        "extra_metrics": extras,
        "data_fingerprint": _data_fingerprint(records),
        "records": records,
    }
    _atomic_write_json(out_path, payload)
    return out_path


def finalize_repeated_trial(
    journal: Journal,
    out_path: Path,
    *,
    model_name: str,
    dataset_name: str,
    split: str | None,
    run_index: int,
) -> Path:
    """Aggregate journal into a paper-shaped repeated-trial JSON file."""

    records = sorted(journal.load_records(), key=lambda r: int(r["index"]))
    payload = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "split": split,
        "experiment": "repeated_trial",
        "run_index": run_index,
        "records": records,
    }
    _atomic_write_json(out_path, payload)
    return out_path


def write_manifest(consolidated_dir: Path) -> Path | None:
    """Write _manifest.json with SHA-256 of every *.json under
    consolidated_dir. Used by `make results` to verify rsync integrity.

    Returns the manifest path on success, or None when the directory
    holds no eligible files. Writing `{"files": []}` for a worker that
    was skipped-not-ready would make `make results` treat the empty run
    as successful-but-empty and promote nothing; the absence of a
    manifest is a clearer "this worker produced no consolidated
    outputs" signal. Callers can key off the None return to emit a
    distinct progress event.
    """

    consolidated_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for p in sorted(consolidated_dir.glob("*.json")):
        if p.name == "_manifest.json":
            continue
        h = hashlib.sha256()
        with p.open("rb") as fp:
            for chunk in iter(lambda: fp.read(65536), b""):
                h.update(chunk)
        entries.append({"path": p.name, "sha256": h.hexdigest(), "bytes": p.stat().st_size})
    if not entries:
        return None
    manifest_path = consolidated_dir / "_manifest.json"
    _atomic_write_json(manifest_path, {"files": entries})
    return manifest_path
