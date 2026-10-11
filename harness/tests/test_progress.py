import json
from pathlib import Path

from pricing_reversal_harness.progress import Progress


def test_progress_emit_writes_and_fsyncs(tmp_path: Path):
    p = Progress(tmp_path)
    p.emit("phase_a", model="m1", detail="hello")
    p.emit("phase_b", model="m1", dataset="aime")

    lines = (tmp_path / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    a = json.loads(lines[0])
    b = json.loads(lines[1])
    assert a["event"] == "progress"
    assert a["phase"] == "phase_a"
    assert b["phase"] == "phase_b"
    assert b["dataset"] == "aime"


def test_state_update_atomic(tmp_path: Path):
    p = Progress(tmp_path)
    p.update_state(current_model="m1", phase="model_up")
    p.update_state(phase="consolidated")
    payload = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert payload["current_model"] == "m1"
    assert payload["phase"] == "consolidated"


def test_state_loaded_by_new_instance(tmp_path: Path):
    Progress(tmp_path).update_state(phase="one", tag="x")
    p2 = Progress(tmp_path)
    assert p2.state == {"phase": "one", "tag": "x"}
