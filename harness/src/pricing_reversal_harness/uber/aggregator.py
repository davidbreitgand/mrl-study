"""Aggregator: reads per-worker heartbeat + state files under
/results/<study-id>/workers/<id>/ and produces
/results/<study-id>/uber/aggregated.json.

Also performs the "promote finalized outputs to study-top-level" step:
whenever a worker's consolidated/*.json or repeated_trial/**/*.json
appears, it is hardlinked (or copied) into
/results/<study-id>/consolidated/ and /results/<study-id>/repeated_trial/
so the study's top-level tree contains ALL models' outputs by the end
of the study.

The Aggregator itself is study-agnostic — it takes a `results_dir`
argument (the caller passes /results/<study-id>) and composes every
path underneath it. This class does not construct study ids or touch
paths outside the one directory it was handed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


log = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except (json.JSONDecodeError, OSError):
        # Mid-write from a worker; skip this tick — we'll pick it up next.
        return None


def read_worker_snapshot(worker_dir: Path) -> dict[str, Any]:
    """Compose a per-worker snapshot from heartbeat.txt + state.json."""
    hb = _safe_read_json(worker_dir / "heartbeat.txt")
    st = _safe_read_json(worker_dir / "state.json")

    snap: dict[str, Any] = {}
    if hb is not None:
        det = hb.get("detail") or {}
        snap.update({
            "heartbeat_ts": hb.get("ts"),
            "phase": (hb.get("state") or {}).get("phase"),
            "current_model": (hb.get("state") or {}).get("current_model"),
            "current_dataset": det.get("dataset") or (hb.get("state") or {}).get("current_dataset"),
            "run_index": det.get("run_index"),
            "done": det.get("done"),
            "total": det.get("total"),
            "last_index": det.get("last_index"),
        })
    elif st is not None:
        snap.update({
            "phase": st.get("phase"),
            "current_model": st.get("current_model"),
            "current_dataset": st.get("current_dataset"),
        })
    return snap


def _hardlink_or_copy(src: Path, dst: Path) -> str:
    """Prefer hardlink to save space + I/O; fall back to copy on cross-filesystem
    or unsupported filesystems. Returns 'link' or 'copy' for logging."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        # Already promoted — trust the existing file (worker writes atomically).
        return "skipped"
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        # Different device, or hardlinks unsupported (rare on Spectrum Scale).
        shutil.copy2(src, dst)
        return "copy"


def promote_worker_outputs(
    worker_dir: Path,
    top_consolidated: Path,
    top_repeated_trial: Path,
) -> list[dict[str, Any]]:
    """Promote a worker's finalized JSONs into the top-level dirs.

    Returns a list of {src, dst, action} records for logging.
    """
    actions: list[dict[str, Any]] = []
    ws_cons = worker_dir / "consolidated"
    if ws_cons.is_dir():
        for f in sorted(ws_cons.glob("*.json")):
            if f.name == "_manifest.json":
                continue
            dst = top_consolidated / f.name
            action = _hardlink_or_copy(f, dst)
            actions.append({"src": str(f), "dst": str(dst), "action": action})

    ws_rep = worker_dir / "repeated_trial"
    if ws_rep.is_dir():
        for f in sorted(ws_rep.rglob("*.json")):
            rel = f.relative_to(ws_rep)
            dst = top_repeated_trial / rel
            action = _hardlink_or_copy(f, dst)
            actions.append({"src": str(f), "dst": str(dst), "action": action})
    return actions


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_top_manifest(root: Path) -> Path | None:
    """Write /results/<study-id>/_manifest.json covering the promoted consolidated
    and repeated_trial trees. Returns the manifest path, or None when
    zero eligible files were promoted — see `journal.write_manifest`
    for the same reasoning."""
    files: list[dict[str, Any]] = []
    for sub in ("consolidated", "repeated_trial"):
        d = root / sub
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*.json")):
            if f.name == "_manifest.json":
                continue
            files.append({
                "path": str(f.relative_to(root)),
                "sha256": sha256_of(f),
                "bytes": f.stat().st_size,
            })
    if not files:
        return None
    manifest = {"ts": _utcnow_iso(), "files": files}
    out = root / "_manifest.json"
    tmp = out.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)
        fp.flush()
        os.fsync(fp.fileno())
    tmp.replace(out)
    return out
    return out


class Aggregator:
    """Composes per-worker snapshots into <study-dir>/uber/aggregated.json."""

    def __init__(self, results_dir: Path) -> None:
        self.results_dir = results_dir
        self.workers_dir = results_dir / "workers"
        self.uber_dir = results_dir / "uber"
        self.top_consolidated = results_dir / "consolidated"
        self.top_repeated_trial = results_dir / "repeated_trial"
        for d in (self.uber_dir, self.top_consolidated, self.top_repeated_trial):
            d.mkdir(parents=True, exist_ok=True)

    def refresh(self, workers: dict[str, dict[str, Any]]) -> Path:
        """Given a dict of {model_id: job_status_info} from the K8s watch,
        merge with each worker's on-disk snapshot and atomically replace
        <study-dir>/uber/aggregated.json.

        `workers[model_id]` is a dict with at least a `job_status` key
        (e.g. "Active", "Complete", "Failed", "Pending", "NotCreated") and
        optionally `pod_pending_reason`, `pod_phase`, etc.
        """
        merged: dict[str, dict[str, Any]] = {}
        for model_id, jinfo in workers.items():
            snap = read_worker_snapshot(self.workers_dir / model_id)
            merged[model_id] = {**jinfo, **snap}

        payload = {"ts": _utcnow_iso(), "workers": merged}
        out = self.uber_dir / "aggregated.json"
        tmp = out.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
            fp.flush()
            os.fsync(fp.fileno())
        tmp.replace(out)
        return out

    def promote(self, model_id: str) -> list[dict[str, Any]]:
        """Called when a worker becomes terminal — promote its outputs."""
        return promote_worker_outputs(
            self.workers_dir / model_id,
            self.top_consolidated,
            self.top_repeated_trial,
        )

    def finalize_manifest(self) -> Path | None:
        """Called at study_complete. Returns None if there were no
        eligible files to include — the uber controller uses the None
        signal to distinguish "the study completed but produced
        nothing" from "the study produced N files"."""
        return write_top_manifest(self.results_dir)
