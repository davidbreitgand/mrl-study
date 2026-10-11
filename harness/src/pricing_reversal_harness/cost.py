"""Per-query cost accounting.

Two independent numbers are computed:
  * hosted_proxy_cost: input_$/MTok * prompt_tokens/1e6 + output_$/MTok * completion_tokens/1e6
    (populated in the top-level `cost` field of records + files, matching the
    paper's schema and letting analysis/ notebooks work unchanged)
  * gpu_hour_cost:   elapsed_s * (tp_size * $/gpu-hr) / 3600 / 8
    (stored under record["extra_fields"]["gpu_hour_cost"])

Cross-cluster CostGuard variant: gpu_hour_cost is ATTRIBUTED to the arm
that served the request. The Hub EPP writes response header
`x-conformance-test-served-endpoint` on every response; the runner reads
it, joins it against a `clusters.yaml`-derived table of
`served_endpoint -> (tp, hourly_rate)`, and computes per-record cost
from THAT tuple. The pricing.yaml + model.tp_size fallback used by
per-cluster studies is not applicable here because the two arms can
have different tp and different hardware rates.
"""

from __future__ import annotations

from .config import ModelSpec, PricingConfig


def hosted_proxy_cost(
    pricing: PricingConfig, model: ModelSpec, prompt_tokens: int, completion_tokens: int
) -> float:
    entry = pricing.hosted_proxy.get(model.id)
    if entry is None:
        return 0.0
    # PricingConfig.resolve_tiers() populates both fields from tier
    # references at load time; if a config file is used without going
    # through load_config() (e.g. some tests), fall back to 0 rather
    # than crashing on None.
    ip = entry.input_price_per_MTok or 0.0
    op = entry.output_price_per_MTok or 0.0
    return (
        prompt_tokens * ip / 1_000_000.0
        + completion_tokens * op / 1_000_000.0
    )


def gpu_hour_cost(
    pricing: PricingConfig,
    elapsed_seconds: float,
    served_endpoint: str | None,
    clusters_table: dict[str, tuple[int, float]] | None,
) -> float:
    """Retroactive per-request cost attribution.

    served_endpoint is the value of the response header
    `x-conformance-test-served-endpoint`. clusters_table is the
    Hub-side view of `clusters.yaml`, keyed by both endpoint `name` and
    `address:port` (so it matches whatever granularity the header
    carries).

    Returns 0.0 when attribution is missing so a mis-configured Hub
    doesn't crash the run — the analyze-run.py pass flags missing
    attribution separately.
    """
    if served_endpoint is None or clusters_table is None:
        return 0.0
    entry = clusters_table.get(served_endpoint)
    if entry is None:
        return 0.0
    tp, rate = entry
    return elapsed_seconds * tp * rate / 3600.0 / 8.0
