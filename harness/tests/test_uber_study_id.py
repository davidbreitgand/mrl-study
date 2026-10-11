"""Tests for the study-id feature on UberController.

The uber generates a per-study id at construction time (default:
UTC timestamp) and threads it through:
  * every path it writes to on the aggregation PVC
    (`self.study_dir` = results_dir / study_id, uber_dir under it)
  * every worker Job manifest render, via the `study_id` Jinja var
    the real worker template uses in `--results-subdir=<id>/workers/...`.

These tests verify both surfaces.
"""

from __future__ import annotations

from pathlib import Path

from pricing_reversal_harness.uber.controller import UberController


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = str(REPO_ROOT / "config")
REAL_WORKER_TMPL = REPO_ROOT / "stack" / "uber" / "worker-job.yaml.tmpl"


def _make_uber(tmp_path: Path, *, study_id: str | None = None) -> UberController:
    return UberController(
        config_dir=CONFIG_DIR,
        results_dir=str(tmp_path),
        namespace="test-ns",
        hf_token_secret="x",
        worker_image="ghcr.io/example/harness:latest",
        worker_template_path=str(REAL_WORKER_TMPL),
        router_values_template_path="/config/router-values.yaml.tmpl",
        aggregator_interval_seconds=0.01,
        dry_run=True,
        study_id=study_id,
    )


def test_default_study_id_is_utc_timestamp(tmp_path: Path):
    uber = _make_uber(tmp_path)
    # YYYY-MM-DDTHH-MM-SSZ — 20 chars with a Z suffix.
    assert len(uber.study_id) == 20
    assert uber.study_id.endswith("Z")
    assert uber.study_id[4] == "-" and uber.study_id[7] == "-"
    assert uber.study_id[10] == "T"


def test_study_dir_lives_under_results_dir(tmp_path: Path):
    uber = _make_uber(tmp_path, study_id="2026-08-31T12-00-00Z")
    assert uber.study_dir == tmp_path / "2026-08-31T12-00-00Z"
    assert uber.uber_dir == tmp_path / "2026-08-31T12-00-00Z" / "uber"
    # Uber's progress dir must exist (Progress created it in __init__).
    assert uber.uber_dir.is_dir()


def test_aggregator_rooted_under_study_dir(tmp_path: Path):
    uber = _make_uber(tmp_path, study_id="2026-08-31T12-00-00Z")
    study = tmp_path / "2026-08-31T12-00-00Z"
    assert uber.aggregator.results_dir == study
    assert uber.aggregator.workers_dir == study / "workers"
    assert uber.aggregator.uber_dir == study / "uber"
    assert uber.aggregator.top_consolidated == study / "consolidated"
    assert uber.aggregator.top_repeated_trial == study / "repeated_trial"


def test_dataset_cache_stays_shared_across_studies(tmp_path: Path):
    """The uber's own paths are all under study_dir, but the top-level
    /results/ (results_dir) stays outside — that's where workers'
    top_results_dir points, and dataset_cache/ / resume-from/ live
    there. Verify by checking that study_dir is strictly deeper than
    results_dir."""
    uber = _make_uber(tmp_path, study_id="2026-08-31T12-00-00Z")
    assert uber.results_dir == tmp_path
    assert uber.study_dir.parent == tmp_path


def test_worker_manifest_carries_study_id_in_subdir(tmp_path: Path):
    """Rendering a worker Job manifest must produce a
    `--results-subdir=<study_id>/workers/<model>` line so the worker's
    orchestrator writes into the same per-study subtree as the uber."""
    uber = _make_uber(tmp_path, study_id="2026-08-31T12-00-00Z")
    model = uber.cfg.models.models[0]
    manifest = uber._render_worker_manifest(model)
    expected = f"--results-subdir=2026-08-31T12-00-00Z/workers/{model.id}"
    assert expected in manifest, (
        f"expected {expected!r} in rendered worker manifest, got:\n"
        f"{manifest[:800]}"
    )
    # Uber still passes --results-dir=/results — the study id lives in
    # the subdir, not the root, so top_results_dir stays shared.
    assert "--results-dir=/results" in manifest


def test_study_id_can_be_injected_for_determinism(tmp_path: Path):
    fixed = "2020-01-01T00-00-00Z"
    uber = _make_uber(tmp_path, study_id=fixed)
    assert uber.study_id == fixed
