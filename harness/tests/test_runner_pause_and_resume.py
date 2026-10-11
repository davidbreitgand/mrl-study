"""Tests for the pause-on-no-endpoints + shared re-probe behaviour in
DatasetRunner.

We fake the VLLMClient entirely — no HTTP, no async gymnastics — so the
tests exercise just the pause/resume logic.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from pricing_reversal_harness.config import (
    DatasetSpec, ExperimentConfig, ModelSpec, PricingConfig, Sampling,
    HostedProxyEntry,
)
from pricing_reversal_harness.progress import Progress
from pricing_reversal_harness.runner import DatasetRunner
from pricing_reversal_harness.vllm_client import NoBackendError


def _model(hf_repo: str = "test/repo") -> ModelSpec:
    return ModelSpec(
        id="test-model",
        hf_repo=hf_repo,
        served_model_name=hf_repo,
        tensor_parallel_size=1,
        reasoning_parser="qwen3",
        max_model_len=32768,
        gpu_memory_utilization=0.9,
        extra_vllm_args=[],
        chat_template_kwargs={},
        sampling=Sampling(),
        hf_cache_size_gi=100,
        ready_timeout_seconds=1800,
    )


def _dataset() -> DatasetSpec:
    return DatasetSpec(
        id="aime",
        split="hybrid",
        file_prefix="aime-hybrid",
        hf_source="test",
        category="math",
        scorer="boxed_integer",
        prompt_template="{question}",
        question_field="q",
        answer_field="a",
        expected_count=3,
    )


def _pricing() -> PricingConfig:
    return PricingConfig(
        gpu_hourly_rate_usd={"h100_80gb": 2.5},
        hosted_proxy={"test-model": HostedProxyEntry(
            input_price_per_MTok=0.5, output_price_per_MTok=2.0,
        )},
    )


def _experiment(recovery: int = 5) -> ExperimentConfig:
    return ExperimentConfig(
        no_endpoints_recovery_seconds=recovery,
    )


def _rows(n: int = 3) -> list[dict[str, Any]]:
    return [{"index": i, "origin_query": f"q{i}", "prompt": f"prompt-{i}",
             "ground_truth": "42"} for i in range(1, n + 1)]


class FakeClient:
    """Programmable stand-in for VLLMClient.

    `chat_script` is a list of per-call outcomes: either a dict (body to
    return, wrapped in a 200-OK envelope) or an Exception to raise.
    `wait_ready_result` controls what the re-probe returns.
    """

    def __init__(self, chat_script, wait_ready_result: bool = True) -> None:
        self._script = list(chat_script)
        self.wait_ready_calls = 0
        self.wait_ready_result = wait_ready_result
        # Give tests a way to sequence probe → chat outcomes
        self._probe_gate: asyncio.Event | None = None

    async def chat(self, **kwargs):
        if not self._script:
            raise RuntimeError("chat_script exhausted")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        # Wrap the payload as a minimal chat completion.
        body = {
            "choices": [{"message": {"content": item["content"]},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": len(item["content"])},
        }
        return body, 0.001

    async def wait_ready(self, **kwargs) -> bool:
        self.wait_ready_calls += 1
        return self.wait_ready_result


async def test_runner_pauses_and_resumes_on_no_backend(tmp_path: Path):
    """First query hits NoBackendError → probe returns True → retry succeeds.
    All queries end up journaled successfully (not as errors)."""
    progress = Progress(tmp_path)
    # Script: 1 error, 3 successes (initial fail + retry + 2 fresh).
    client = FakeClient(
        chat_script=[
            NoBackendError(503, "failed to find endpoint candidates"),
            {"content": "\\boxed{42}"},   # retry of #1 after probe
            {"content": "\\boxed{42}"},   # #2
            {"content": "\\boxed{42}"},   # #3
        ],
        wait_ready_result=True,
    )
    runner = DatasetRunner(
        client=client,
        model=_model(),
        dataset=_dataset(),
        pricing=_pricing(),
        experiment=_experiment(recovery=2),
        results_dir=tmp_path,
        progress=progress,
        run_index=None,
        preserve_scoring=True,
    )
    outcomes = await runner.run(_rows(3))

    # None errored.
    assert all("error" not in (o.record.get("extra_fields") or {}) for o in outcomes)
    # Re-probe was called exactly once.
    assert client.wait_ready_calls == 1
    # Progress emitted the recovery events.
    progress_text = (tmp_path / "progress.jsonl").read_text(encoding="utf-8")
    assert "backend_no_endpoints" in progress_text
    assert "backend_recovered" in progress_text


async def test_runner_gives_up_when_reprobe_deadline_exhausts(tmp_path: Path):
    """First query hits NoBackendError → probe returns False → runner
    marks the backend as gave up; all queued queries surface NoBackendError
    without hitting the backend again."""
    progress = Progress(tmp_path)
    client = FakeClient(
        chat_script=[
            NoBackendError(503, "failed to find endpoint candidates"),
            # No more chat calls expected — everyone should short-circuit
            # via the _backend_gave_up flag.
        ],
        wait_ready_result=False,
    )
    runner = DatasetRunner(
        client=client,
        model=_model(),
        dataset=_dataset(),
        pricing=_pricing(),
        experiment=_experiment(recovery=1),
        results_dir=tmp_path,
        progress=progress,
        run_index=None,
        preserve_scoring=True,
    )
    outcomes = await runner.run(_rows(3))
    # All errored.
    assert all("error" in (o.record.get("extra_fields") or {}) for o in outcomes)
    # Re-probe attempted once (not per-query).
    assert client.wait_ready_calls == 1
    progress_text = (tmp_path / "progress.jsonl").read_text(encoding="utf-8")
    assert "backend_no_endpoints" in progress_text
    assert "backend_gave_up" in progress_text


async def test_runner_concurrent_errors_share_recovery(tmp_path: Path):
    """N concurrent in-flight queries seeing NoBackendError should share
    a single re-probe rather than each dispatching its own. The runner
    invariant: never more than one probe running at a time.

    The FakeClient here models "the router had one outage; the first
    query into it hits NoBackendError, everyone else gets in behind the
    gate, one probe recovers, all queries then succeed on their
    (single) chat call." That's the realistic preemption scenario.
    """
    progress = Progress(tmp_path)
    # 6 script items: 1 initial NoBackendError, then 5 successes.
    # Row 1: chat → error → probe → chat → success (2 script slots consumed)
    # Rows 2..5: await gate, then chat → success (1 slot each)
    # Total = 2 + 4 = 6. This matches the real "single outage" flow.
    script: list = [NoBackendError(503, "failed to find endpoint candidates")]
    script += [{"content": "\\boxed{42}"} for _ in range(5)]
    client = FakeClient(chat_script=script, wait_ready_result=True)
    runner = DatasetRunner(
        client=client,
        model=_model(),
        dataset=_dataset(),
        pricing=_pricing(),
        experiment=_experiment(recovery=2),
        results_dir=tmp_path,
        progress=progress,
        run_index=None,
        preserve_scoring=True,
    )
    outcomes = await runner.run(_rows(5))
    # All 5 succeed after the shared recovery.
    assert all("error" not in (o.record.get("extra_fields") or {}) for o in outcomes)
    # Exactly one probe was dispatched — the shared-gate design.
    assert client.wait_ready_calls == 1


async def test_runner_journal_error_records_include_error_field(tmp_path: Path):
    """When probe gives up, records still get journaled with an `error`
    field so `_fraction_errored` in the orchestrator can detect the
    all-errored condition."""
    progress = Progress(tmp_path)
    client = FakeClient(
        chat_script=[NoBackendError(503, "failed to find endpoint candidates")],
        wait_ready_result=False,
    )
    runner = DatasetRunner(
        client=client,
        model=_model(),
        dataset=_dataset(),
        pricing=_pricing(),
        experiment=_experiment(recovery=1),
        results_dir=tmp_path,
        progress=progress,
        run_index=None,
        preserve_scoring=True,
    )
    outcomes = await runner.run(_rows(2))
    for o in outcomes:
        assert "error" in o.record["extra_fields"]
        # Cost and tokens are zero for errored records — the aggregator
        # relies on this shape.
        assert o.record["cost"] == 0.0
        assert o.record["prompt_tokens"] == 0
        assert o.record["completion_tokens"] == 0
