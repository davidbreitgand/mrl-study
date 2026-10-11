"""Dynamic-max_tokens ceiling in DatasetRunner.

DatasetRunner._max_tokens_for(prompt) must never let
`prompt_tokens + max_tokens` exceed `max_model_len - headroom`.
The tokenizer is stubbed out via reasoning_parsers.count_prompt_tokens
so these tests don't require network / HF cache access.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from pricing_reversal_harness.config import (
    DatasetSpec, ExperimentConfig, ModelSpec, PricingConfig, Sampling,
    HostedProxyEntry,
)
from pricing_reversal_harness.progress import Progress
from pricing_reversal_harness.runner import DatasetRunner


def _model(*, max_model_len: int, max_tokens: int) -> ModelSpec:
    return ModelSpec(
        id="test-model",
        hf_repo="test/repo",
        served_model_name="test/repo",
        tensor_parallel_size=1,
        reasoning_parser="qwen3",
        max_model_len=max_model_len,
        gpu_memory_utilization=0.9,
        extra_vllm_args=[],
        chat_template_kwargs={},
        sampling=Sampling(temperature=0.7, top_p=0.95, max_tokens=max_tokens),
        hf_cache_size_gi=100,
        ready_timeout_seconds=1800,
    )


def _dataset() -> DatasetSpec:
    return DatasetSpec(
        id="aime", split="hybrid", file_prefix="aime-hybrid",
        hf_source="test", category="math", scorer="boxed_integer",
        prompt_template="{question}", question_field="q", answer_field="a",
        expected_count=3,
    )


def _pricing() -> PricingConfig:
    return PricingConfig(
        gpu_hourly_rate_usd={"h100_80gb": 2.5},
        hosted_proxy={"test-model": HostedProxyEntry(
            input_price_per_MTok=0.5, output_price_per_MTok=2.0,
        )},
    )


def _runner(tmp_path: Path, model: ModelSpec, headroom: int = 256) -> DatasetRunner:
    return DatasetRunner(
        client=None,  # unused by _max_tokens_for
        model=model,
        dataset=_dataset(),
        pricing=_pricing(),
        experiment=ExperimentConfig(max_tokens_headroom=headroom),
        results_dir=tmp_path,
        progress=Progress(tmp_path),
        run_index=None,
    )


def test_short_prompt_returns_sampling_ceiling(tmp_path: Path) -> None:
    """When plenty of context is available the ceiling is the model's own
    sampling.max_tokens — the fix must not silently under-serve."""
    model = _model(max_model_len=32768, max_tokens=8192)
    runner = _runner(tmp_path, model)
    with patch("pricing_reversal_harness.runner.count_prompt_tokens",
               return_value=100):
        assert runner._max_tokens_for("short") == 8192


def test_long_prompt_shrinks_max_tokens_to_fit(tmp_path: Path) -> None:
    """The arc-agi-v1 pattern that broke on gemma: prompt_tokens is big
    enough that sampling.max_tokens would overflow max_model_len. Returned
    budget must be exactly max_model_len - prompt_tokens - headroom."""
    model = _model(max_model_len=69632, max_tokens=65536)
    runner = _runner(tmp_path, model, headroom=256)
    with patch("pricing_reversal_harness.runner.count_prompt_tokens",
               return_value=6000):
        # 69632 - 6000 - 256 = 63376, below the 65536 sampling ceiling
        assert runner._max_tokens_for("6k prompt") == 63376


def test_never_returns_below_one(tmp_path: Path) -> None:
    """If a prompt would already consume the whole context window plus
    headroom, we still return >=1 so callers get a clean vLLM 400 rather
    than passing 0 or a negative number."""
    model = _model(max_model_len=1024, max_tokens=512)
    runner = _runner(tmp_path, model, headroom=256)
    with patch("pricing_reversal_harness.runner.count_prompt_tokens",
               return_value=2000):
        assert runner._max_tokens_for("huge") == 1


def test_uses_headroom_setting(tmp_path: Path) -> None:
    """Headroom is subtracted from the ceiling."""
    model = _model(max_model_len=10000, max_tokens=9999)
    runner = _runner(tmp_path, model, headroom=500)
    with patch("pricing_reversal_harness.runner.count_prompt_tokens",
               return_value=1000):
        # 10000 - 1000 - 500 = 8500, below sampling ceiling of 9999
        assert runner._max_tokens_for("1k") == 8500
