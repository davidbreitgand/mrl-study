from pricing_reversal_harness.config import (
    HostedProxyEntry, ModelSpec, PricingConfig, Sampling,
)
from pricing_reversal_harness.cost import gpu_hour_cost, hosted_proxy_cost


def _model(tp: int = 1) -> ModelSpec:
    return ModelSpec(
        id="test-model",
        hf_repo="test/repo",
        served_model_name="test/repo",
        tensor_parallel_size=tp,
        reasoning_parser="qwen3",
        max_model_len=32768,
        gpu_memory_utilization=0.9,
        extra_vllm_args=[],
        chat_template_kwargs={},
        sampling=Sampling(),
        hf_cache_size_gi=100,
        ready_timeout_seconds=1800,
    )


def _pricing() -> PricingConfig:
    return PricingConfig(
        gpu_hourly_rate_usd={"h100_80gb": 2.5},
        hosted_proxy={
            "test-model": HostedProxyEntry(
                input_price_per_MTok=0.5, output_price_per_MTok=2.0
            ),
        },
    )


def test_hosted_proxy_cost_matches_paper_formula():
    m = _model()
    p = _pricing()
    # 1000 prompt tokens @ $0.5/MTok + 2000 completion tokens @ $2.0/MTok
    #   = 0.0005 + 0.004 = 0.0045 USD
    got = hosted_proxy_cost(p, m, prompt_tokens=1000, completion_tokens=2000)
    assert abs(got - 0.0045) < 1e-12


def test_hosted_proxy_zero_when_model_missing():
    m = _model()
    p = PricingConfig(gpu_hourly_rate_usd={}, hosted_proxy={})
    assert hosted_proxy_cost(p, m, 1000, 1000) == 0.0


def test_gpu_hour_cost_attributes_from_clusters_table():
    p = _pricing()
    # 3600s wall clock, tp=2, rate=$2.5/hr, /8 node-share
    #   = 3600 * 2 * 2.5 / 3600 / 8 = 0.625 USD
    table = {"piggy-gemma": (2, 2.5)}
    got = gpu_hour_cost(p, 3600.0, "piggy-gemma", table)
    assert abs(got - 0.625) < 1e-12


def test_gpu_hour_cost_zero_when_served_endpoint_missing():
    p = _pricing()
    table = {"piggy-gemma": (2, 2.5)}
    assert gpu_hour_cost(p, 100.0, None, table) == 0.0


def test_gpu_hour_cost_zero_when_endpoint_not_in_table():
    p = _pricing()
    table = {"piggy-gemma": (2, 2.5)}
    assert gpu_hour_cost(p, 100.0, "unknown-arm", table) == 0.0


def test_gpu_hour_cost_zero_when_clusters_table_none():
    p = _pricing()
    assert gpu_hour_cost(p, 100.0, "piggy-gemma", None) == 0.0
