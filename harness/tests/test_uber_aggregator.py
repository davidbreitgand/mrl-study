"""Aggregator tests — pure Python, no K8s dependency."""

from __future__ import annotations

import json
from pathlib import Path

from pricing_reversal_harness.uber.aggregator import (
    Aggregator, read_worker_snapshot, promote_worker_outputs,
    write_top_manifest,
)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_read_worker_snapshot_empty(tmp_path: Path):
    snap = read_worker_snapshot(tmp_path / "workers" / "qwen3-8b")
    assert snap == {}


def test_read_worker_snapshot_from_heartbeat(tmp_path: Path):
    wd = tmp_path / "workers" / "qwen3-8b"
    _write_json(wd / "heartbeat.txt", {
        "ts": "2026-08-25T12:00:00Z",
        "state": {"phase": "consolidated", "current_model": "qwen3-8b"},
        "detail": {"model": "qwen3-8b", "dataset": "gpqa",
                   "done": 154, "total": 198, "last_index": 160},
    })
    snap = read_worker_snapshot(wd)
    assert snap["phase"] == "consolidated"
    assert snap["current_dataset"] == "gpqa"
    assert snap["done"] == 154
    assert snap["total"] == 198
    assert snap["last_index"] == 160


def test_read_worker_snapshot_state_only_fallback(tmp_path: Path):
    """If heartbeat is missing but state.json exists, fall back to state."""
    wd = tmp_path / "workers" / "granite"
    _write_json(wd / "state.json", {
        "phase": "model_up",
        "current_model": "granite",
        "current_dataset": None,
    })
    snap = read_worker_snapshot(wd)
    assert snap["phase"] == "model_up"
    assert snap["current_model"] == "granite"


def test_aggregator_refresh_merges_worker_snapshots(tmp_path: Path):
    agg = Aggregator(tmp_path)
    _write_json(tmp_path / "workers/qwen3-8b/heartbeat.txt", {
        "ts": "t1", "state": {"phase": "consolidated"},
        "detail": {"model": "qwen3-8b", "dataset": "aime", "done": 30, "total": 60},
    })
    _write_json(tmp_path / "workers/granite/heartbeat.txt", {
        "ts": "t2", "state": {"phase": "model_up"},
        "detail": {"model": "granite"},
    })
    out = agg.refresh({
        "qwen3-8b": {"job_status": "Active"},
        "granite": {"job_status": "Pending"},
        "deepseek": {"job_status": "NotCreated"},
    })
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload["workers"].keys()) == {"qwen3-8b", "granite", "deepseek"}
    q = payload["workers"]["qwen3-8b"]
    assert q["job_status"] == "Active"
    assert q["done"] == 30 and q["total"] == 60
    assert q["current_dataset"] == "aime"
    # NotCreated worker with no on-disk state stays minimal.
    ds = payload["workers"]["deepseek"]
    assert ds["job_status"] == "NotCreated"


def test_promote_worker_outputs_hardlinks(tmp_path: Path):
    top_cons = tmp_path / "consolidated"
    top_rep = tmp_path / "repeated_trial"
    worker = tmp_path / "workers/qwen3-8b"
    # Fake finalized worker outputs:
    _write_json(worker / "consolidated" / "aime-hybrid-qwen3-8b.json", {"performance": 0.7})
    _write_json(worker / "consolidated" / "gpqa-test-qwen3-8b.json", {"performance": 0.5})
    _write_json(worker / "consolidated" / "_manifest.json", {"files": []})  # skipped
    _write_json(worker / "repeated_trial/aime/qwen3-8b/run0.json", {"run_index": 0})

    actions = promote_worker_outputs(worker, top_cons, top_rep)
    names_promoted = {Path(a["src"]).name for a in actions}
    assert "aime-hybrid-qwen3-8b.json" in names_promoted
    assert "gpqa-test-qwen3-8b.json" in names_promoted
    assert "_manifest.json" not in names_promoted  # excluded by design
    assert (top_cons / "aime-hybrid-qwen3-8b.json").exists()
    assert (top_rep / "aime" / "qwen3-8b" / "run0.json").exists()


def test_promote_is_idempotent(tmp_path: Path):
    top_cons = tmp_path / "consolidated"
    top_rep = tmp_path / "repeated_trial"
    worker = tmp_path / "workers/qwen3-8b"
    _write_json(worker / "consolidated" / "aime-hybrid-qwen3-8b.json", {"performance": 0.7})

    a1 = promote_worker_outputs(worker, top_cons, top_rep)
    a2 = promote_worker_outputs(worker, top_cons, top_rep)
    # First round did the link, second round is a no-op ("skipped").
    assert any(a["action"] == "link" for a in a1)
    assert all(a["action"] == "skipped" for a in a2)


def test_write_top_manifest_sha256s(tmp_path: Path):
    (tmp_path / "consolidated").mkdir()
    (tmp_path / "consolidated/aime-hybrid-qwen3-8b.json").write_text('{"p":1}', encoding="utf-8")
    (tmp_path / "consolidated/gpqa-test-qwen3-8b.json").write_text('{"p":0.5}', encoding="utf-8")
    manifest = write_top_manifest(tmp_path)
    m = json.loads(manifest.read_text(encoding="utf-8"))
    paths = {e["path"] for e in m["files"]}
    assert paths == {"consolidated/aime-hybrid-qwen3-8b.json",
                     "consolidated/gpqa-test-qwen3-8b.json"}
    for e in m["files"]:
        assert len(e["sha256"]) == 64


def test_write_top_manifest_excludes_nested_manifests(tmp_path: Path):
    (tmp_path / "consolidated").mkdir()
    (tmp_path / "consolidated/aime-hybrid-x.json").write_text('{"p":1}', encoding="utf-8")
    (tmp_path / "consolidated/_manifest.json").write_text('{"files":[]}', encoding="utf-8")
    m = json.loads(write_top_manifest(tmp_path).read_text(encoding="utf-8"))
    paths = {e["path"] for e in m["files"]}
    assert "consolidated/_manifest.json" not in paths
    assert "consolidated/aime-hybrid-x.json" in paths
