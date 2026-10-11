"""Tests for the Orchestrator's fail-fast-on-all-errored short-circuit.

We patch `DatasetRunner` at import time in the orchestrator module so
`_run_consolidated` / `_run_repeated_trial` use a fake runner that
returns a scripted outcome list. Then we call `_run_one_model` with
dry_run=True to skip Kubernetes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from pricing_reversal_harness.orchestrator import (
    Orchestrator, RunAllErroredError, _fraction_errored, _is_error_outcome,
)
from pricing_reversal_harness.runner import QueryOutcome


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = str(REPO_ROOT / "config")
VLLM_TEMPLATE = str(REPO_ROOT / "stack" / "modelserver" / "vllm-deployment.yaml.tmpl")


def _err_outcome(i: int) -> QueryOutcome:
    return QueryOutcome(i, {
        "index": i, "cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
        "extra_fields": {"error": "HTTP 503: no backend"},
    })


def _ok_outcome(i: int) -> QueryOutcome:
    return QueryOutcome(i, {
        "index": i, "cost": 0.001, "prompt_tokens": 100, "completion_tokens": 200,
        "extra_fields": {"elapsed_seconds": 1.0, "gpu_hour_cost": 0.001},
    })


def test_is_error_outcome_detects_error_field():
    assert _is_error_outcome(_err_outcome(1))
    assert not _is_error_outcome(_ok_outcome(1))


def test_fraction_errored_math():
    outcomes = [_ok_outcome(1), _ok_outcome(2), _err_outcome(3)]
    assert _fraction_errored(outcomes) == pytest.approx(1 / 3)
    all_err = [_err_outcome(i) for i in range(10)]
    assert _fraction_errored(all_err) == 1.0
    assert _fraction_errored([]) == 0.0


def test_run_all_errored_error_carries_context():
    err = RunAllErroredError(
        model_id="qwen3-8b", dataset_id="gpqa", run_index=None,
        error_rate=1.0, total=198,
    )
    assert err.model_id == "qwen3-8b"
    assert err.dataset_id == "gpqa"
    assert err.error_rate == 1.0
    assert err.total == 198
    assert "gpqa" in str(err)


def _make_orch(tmp_path: Path):
    return Orchestrator(
        config_dir=CONFIG_DIR,
        results_dir=str(tmp_path),
        namespace="test-ns",
        hf_token_secret="x",
        vllm_template_path=VLLM_TEMPLATE,
        router_service="pricing-reversal-router-epp",
        router_port=80,
        dry_run=True,
        only_model="qwen3-8b",
    )


class _FakeRunner:
    """DatasetRunner stand-in. Returns a scripted outcome list."""
    def __init__(self, outcomes_by_call: list, **kwargs):
        self.outcomes_by_call = list(outcomes_by_call)
        self.calls = 0

    async def run(self, rows):
        self.calls += 1
        return self.outcomes_by_call.pop(0)


async def test_all_errored_consolidated_short_circuits_remaining_datasets(tmp_path: Path):
    """First consolidated dataset all-errors → orchestrator raises
    RunAllErroredError → _run_one_model catches it, skips the second
    dataset AND the 5 repeated-trial runs → total FakeRunner calls = 1."""
    orch = _make_orch(tmp_path)
    # Two consolidated datasets configured (aime, gpqa) + 5 repeated trials.
    # If short-circuit works, only 1 runner.run() should happen.
    scripts = [
        [_err_outcome(i) for i in range(60)],  # first consolidated: aime — all errored
    ]
    fake_runner = _FakeRunner(scripts)

    with patch("pricing_reversal_harness.orchestrator.DatasetRunner",
               return_value=fake_runner), \
         patch("pricing_reversal_harness.orchestrator.load_dataset",
               side_effect=lambda spec, cd: [{
                   "index": i + 1, "origin_query": f"q{i}", "prompt": f"p{i}",
                   "ground_truth": "42",
               } for i in range(60)]):
        model = next(m for m in orch.cfg.models.models if m.id == "qwen3-8b")
        await orch._run_one_model(model)

    assert fake_runner.calls == 1, \
        f"expected only 1 runner call (aime); got {fake_runner.calls}"

    # Progress emitted the skip event.
    progress_text = (tmp_path / "progress.jsonl").read_text(encoding="utf-8")
    assert "model_skip_all_errored" in progress_text


async def test_healthy_run_completes_all_datasets_and_repeated_trials(tmp_path: Path):
    """No errors → orchestrator processes every consolidated dataset AND
    every repeated-trial run.

    The exact count is derived from the shipped experiment.yaml, not
    hardcoded, so the test tracks the config as it grows or shrinks.
    """
    orch = _make_orch(tmp_path)
    n_consolidated = len(orch.cfg.experiment.consolidated_datasets)
    n_repeated = orch.cfg.experiment.repeated_trials
    expected_calls = n_consolidated + n_repeated

    scripts = [
        [_ok_outcome(i + 1) for i in range(3)]  # each run returns 3 ok outcomes
        for _ in range(expected_calls)
    ]
    fake_runner = _FakeRunner(scripts)

    with patch("pricing_reversal_harness.orchestrator.DatasetRunner",
               return_value=fake_runner), \
         patch("pricing_reversal_harness.orchestrator.load_dataset",
               side_effect=lambda spec, cd: [{
                   "index": i + 1, "origin_query": f"q{i}", "prompt": f"p{i}",
                   "ground_truth": "42",
               } for i in range(3)]):
        model = next(m for m in orch.cfg.models.models if m.id == "qwen3-8b")
        await orch._run_one_model(model)

    assert fake_runner.calls == expected_calls, (
        f"expected {expected_calls} runner calls "
        f"({n_consolidated} consolidated + {n_repeated} repeated), "
        f"got {fake_runner.calls}"
    )

    progress_text = (tmp_path / "progress.jsonl").read_text(encoding="utf-8")
    assert "model_skip_all_errored" not in progress_text
    assert "model_complete" in progress_text
