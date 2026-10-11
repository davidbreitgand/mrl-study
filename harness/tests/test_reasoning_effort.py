"""`reasoning_effort` propagation through the harness.

Three layers are covered:

  * ExperimentConfig accepts a valid value + rejects an invalid one
    (the pydantic field_validator catches typos before any Deployment
    is applied).
  * VLLMClient.chat injects the field into the OpenAI-compatible
    request body when the argument is non-None, and omits it otherwise.
  * DatasetRunner passes the ExperimentConfig value straight through
    to every chat() call — both the primary call and the post-recovery
    retry after a NoBackendError.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from pricing_reversal_harness.config import (
    DatasetSpec, ExperimentConfig, ModelSpec, PricingConfig, Sampling,
    HostedProxyEntry,
)
from pricing_reversal_harness.progress import Progress
from pricing_reversal_harness.runner import DatasetRunner
from pricing_reversal_harness.vllm_client import NoBackendError, VLLMClient


# ---- ExperimentConfig validation --------------------------------------


def test_experiment_config_defaults_to_high() -> None:
    """The design intent is "high on every request" — verify the default
    matches so an operator dropping in a fresh config gets it without
    thinking about it."""
    cfg = ExperimentConfig()
    assert cfg.reasoning_effort == "high"


@pytest.mark.parametrize("value", ["low", "medium", "high", "none", None])
def test_experiment_config_accepts_valid_reasoning_effort(value) -> None:
    cfg = ExperimentConfig(reasoning_effort=value)
    assert cfg.reasoning_effort == value


def test_experiment_config_rejects_bogus_reasoning_effort() -> None:
    """A typo like 'HIGH' or 'max' must fail loudly at load time — a
    bad value silently accepted by vLLM would poison the whole study."""
    with pytest.raises(Exception):
        ExperimentConfig(reasoning_effort="HIGH")
    with pytest.raises(Exception):
        ExperimentConfig(reasoning_effort="max")


# ---- VLLMClient payload shape -----------------------------------------


class _RecordingTransport(httpx.AsyncBaseTransport):
    """Intercepts one POST and records its JSON body; returns a canned
    chat completion so the caller can inspect what was sent."""

    def __init__(self) -> None:
        self.last_body: dict | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        import json
        self.last_body = json.loads(request.content.decode("utf-8"))
        response_body = {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        return httpx.Response(200, json=response_body)


def _client_with_transport(transport: _RecordingTransport) -> VLLMClient:
    """Build a VLLMClient whose underlying httpx client uses our
    recording transport instead of hitting the network."""
    c = VLLMClient.__new__(VLLMClient)
    c.base_url = "http://router"
    c.timeout_seconds = 10.0
    c.retry_max_attempts = 1
    c.retry_backoff_seconds = 0.0
    c._sem = asyncio.Semaphore(1)
    c._client = httpx.AsyncClient(transport=transport, timeout=10.0)
    return c


async def test_chat_includes_reasoning_effort_when_set() -> None:
    tr = _RecordingTransport()
    c = _client_with_transport(tr)
    try:
        await c.chat(
            model="google/gemma-4-26B-A4B-it",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10, temperature=0.0, top_p=1.0,
            reasoning_effort="high",
        )
    finally:
        await c.aclose()
    assert tr.last_body is not None
    assert tr.last_body.get("reasoning_effort") == "high"


async def test_chat_omits_reasoning_effort_when_none() -> None:
    """Passing None must not smuggle the key in — some vLLM versions
    may reject unknown/null fields, and the current behaviour of
    omitting it entirely is what preserves backwards-compat with the
    live cluster."""
    tr = _RecordingTransport()
    c = _client_with_transport(tr)
    try:
        await c.chat(
            model="google/gemma-4-26B-A4B-it",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10, temperature=0.0, top_p=1.0,
            reasoning_effort=None,
        )
    finally:
        await c.aclose()
    assert tr.last_body is not None
    assert "reasoning_effort" not in tr.last_body


# ---- Runner integration -----------------------------------------------


def _model_spec(hf_repo: str = "test/repo") -> ModelSpec:
    return ModelSpec(
        id="test-model",
        hf_repo=hf_repo,
        served_model_name=hf_repo,
        tensor_parallel_size=1,
        reasoning_parser="gemma4",
        max_model_len=32768,
        gpu_memory_utilization=0.9,
        extra_vllm_args=[],
        chat_template_kwargs={},
        sampling=Sampling(),
        hf_cache_size_gi=100,
        ready_timeout_seconds=1800,
    )


def _dataset_spec() -> DatasetSpec:
    return DatasetSpec(
        id="aime", split="hybrid", file_prefix="aime-hybrid",
        hf_source="test", category="math", scorer="boxed_integer",
        prompt_template="{question}", question_field="q", answer_field="a",
        expected_count=1,
    )


def _pricing_cfg() -> PricingConfig:
    return PricingConfig(
        gpu_hourly_rate_usd={"h100_80gb": 2.5},
        hosted_proxy={"test-model": HostedProxyEntry(
            input_price_per_MTok=0.5, output_price_per_MTok=2.0,
        )},
    )


class _RecordingClient:
    """Fake client that records every chat() kwargs it sees."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        body = {
            "choices": [{"message": {"content": "\\boxed{42}"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        return body, 0.001


async def test_runner_forwards_reasoning_effort_to_client(tmp_path: Path) -> None:
    """DatasetRunner must pass experiment.reasoning_effort into every
    chat() call so vLLM's auto-injection kicks in server-side."""
    client = _RecordingClient()
    runner = DatasetRunner(
        client=client,
        model=_model_spec(),
        dataset=_dataset_spec(),
        pricing=_pricing_cfg(),
        experiment=ExperimentConfig(reasoning_effort="high"),
        results_dir=tmp_path,
        progress=Progress(tmp_path),
        run_index=None,
    )
    rows = [{"index": 1, "origin_query": "q", "prompt": "p", "ground_truth": "42"}]
    await runner.run(rows)
    assert len(client.calls) == 1
    assert client.calls[0]["reasoning_effort"] == "high"


async def test_runner_passes_none_when_config_disables(tmp_path: Path) -> None:
    """An operator can opt out by setting reasoning_effort=null in
    experiment.yaml — the runner must faithfully pass None down."""
    client = _RecordingClient()
    runner = DatasetRunner(
        client=client,
        model=_model_spec(),
        dataset=_dataset_spec(),
        pricing=_pricing_cfg(),
        experiment=ExperimentConfig(reasoning_effort=None),
        results_dir=tmp_path,
        progress=Progress(tmp_path),
        run_index=None,
    )
    rows = [{"index": 1, "origin_query": "q", "prompt": "p", "ground_truth": "42"}]
    await runner.run(rows)
    assert client.calls[0]["reasoning_effort"] is None
