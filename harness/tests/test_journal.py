import json
from pathlib import Path

from pricing_reversal_harness.journal import (
    Journal, finalize_consolidated, finalize_repeated_trial, journal_path,
    write_manifest,
)


def _mk_record(i: int, score: float = 1.0, prompt_tokens: int = 100,
               completion_tokens: int = 200, thinking_tokens: int = 150,
               cost: float = 0.001, gpu_cost: float = 0.005) -> dict:
    return {
        "index": i,
        "origin_query": f"q{i}",
        "prompt": f"prompt-{i}",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "thinking_tokens": thinking_tokens,
        "cost": cost,
        "score": score,
        "prediction": "42",
        "ground_truth": "42",
        "raw_output": "\\boxed{42}",
        "extra_fields": {"gpu_hour_cost": gpu_cost, "elapsed_seconds": 1.0},
    }


def test_journal_roundtrip(tmp_path: Path):
    p = journal_path(tmp_path, "m1", "aime")
    j = Journal(p)
    j.append(_mk_record(1))
    j.append(_mk_record(2, score=0.0))
    assert j.load_completed_indices() == {1, 2}
    records = j.load_records()
    assert len(records) == 2
    assert records[0]["index"] == 1


def test_journal_skips_partial_line(tmp_path: Path):
    p = journal_path(tmp_path, "m1", "aime")
    j = Journal(p)
    j.append(_mk_record(1))
    # simulate a torn write from an interrupted process
    with p.open("a", encoding="utf-8") as fp:
        fp.write("{\"incomplete\": ")  # no newline, no closing brace
    records = j.load_records()
    assert len(records) == 1
    assert records[0]["index"] == 1


def test_finalize_consolidated_shape(tmp_path: Path):
    p = journal_path(tmp_path, "m1", "aime")
    j = Journal(p)
    for i in range(1, 4):
        j.append(_mk_record(i, score=1.0 if i < 3 else 0.0))

    out = tmp_path / "consolidated" / "aime-hybrid-m1.json"
    finalize_consolidated(
        j, out,
        model_name="m1", dataset_name="aime", split="hybrid",
        elapsed_seconds=12.34,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    # top-level keys mirror data/consolidated/*.json
    for key in (
        "performance", "time_taken", "prompt_tokens", "completion_tokens",
        "cost", "counts", "model_name", "dataset_name", "split", "demo",
        "extra_metrics", "data_fingerprint", "records",
    ):
        assert key in payload
    assert payload["counts"] == 3
    assert payload["performance"] == 2 / 3
    assert payload["prompt_tokens"] == 300
    assert payload["completion_tokens"] == 600
    assert abs(payload["cost"] - 0.003) < 1e-9
    assert payload["extra_metrics"]["gpu_hour_cost_total"] == 0.015


def test_finalize_repeated_trial_shape(tmp_path: Path):
    p = journal_path(tmp_path, "m1", "aime", run_index=2)
    j = Journal(p)
    j.append(_mk_record(1))

    out = tmp_path / "repeated_trial" / "aime" / "m1" / "run2.json"
    finalize_repeated_trial(
        j, out,
        model_name="m1", dataset_name="aime", split="hybrid", run_index=2,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["experiment"] == "repeated_trial"
    assert payload["run_index"] == 2
    assert len(payload["records"]) == 1


def test_manifest_covers_files(tmp_path: Path):
    d = tmp_path / "consolidated"
    d.mkdir()
    (d / "aime-hybrid-m1.json").write_text("{\"performance\": 1}", encoding="utf-8")
    (d / "gpqa-test-m1.json").write_text("{\"performance\": 0.5}", encoding="utf-8")
    path = write_manifest(d)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    names = {e["path"] for e in manifest["files"]}
    assert names == {"aime-hybrid-m1.json", "gpqa-test-m1.json"}
    for e in manifest["files"]:
        assert len(e["sha256"]) == 64


def test_manifest_skipped_when_no_eligible_files(tmp_path: Path):
    """A worker that gets skip-not-ready produces zero consolidated
    files. Writing `{"files": []}` here would make `make results`
    treat the empty run as successful-but-empty; the absence of a
    manifest is the clearer signal. The caller (orchestrator) keys off
    the None return to emit a `study_no_outputs` phase and exit
    non-zero so the Kubernetes Job goes Failed."""
    d = tmp_path / "consolidated"
    d.mkdir()
    result = write_manifest(d)
    assert result is None
    assert not (d / "_manifest.json").exists()
