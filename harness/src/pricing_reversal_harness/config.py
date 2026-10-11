"""Typed configuration objects loaded from the mounted /config ConfigMap.

The four YAML files (models, datasets, pricing, experiment) are validated on
load; a bad config aborts the harness before any Deployment is applied.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# DNS-1035: lower-case alphanumeric + `-`, must start with a letter and end
# with alphanumeric. Model ids become parts of Helm release names, k8s
# object names, and PVC names, so anything else (dots in particular) makes
# the chart fail at install time with an opaque validation error.
_DNS1035 = re.compile(r"^[a-z]([-a-z0-9]*[a-z0-9])?$")


class Sampling(BaseModel):
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 16384


class ModelSpec(BaseModel):
    id: str
    hf_repo: str

    @field_validator("id")
    @classmethod
    def _id_must_be_dns1035(cls, v: str) -> str:
        if not _DNS1035.fullmatch(v):
            raise ValueError(
                f"model id {v!r} is not a valid DNS-1035 label. "
                f"Allowed: lowercase alphanumerics + '-', starting with a "
                f"letter and ending alphanumeric. No dots. This id becomes "
                f"part of the router Helm release name, so a dot causes an "
                f"opaque `Service invalid` failure at helm install."
            )
        return v
    served_model_name: str
    tensor_parallel_size: int
    reasoning_parser: str
    max_model_len: int
    gpu_memory_utilization: float = 0.9
    extra_vllm_args: list[str] = Field(default_factory=list)
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    sampling: Sampling = Field(default_factory=Sampling)
    hf_cache_size_gi: int = 100
    # Two-tier readiness budget — see K8s.wait_deployment_available.
    # `pending_budget_seconds` bounds "waiting for the scheduler to
    # place a Pod on a GPU node"; `startup_budget_seconds` bounds
    # "vLLM weights loading + JIT after the container starts". If
    # `pending_budget_seconds` is unset, the loader falls back to the
    # legacy `ready_timeout_seconds` treating that whole budget as
    # startup — preserves behaviour for old configs but is Pending-blind.
    pending_budget_seconds: int | None = None
    startup_budget_seconds: int | None = None
    ready_timeout_seconds: int = 1800  # legacy; migrated into startup at load

    @model_validator(mode="after")
    def _fill_readiness_budgets(self) -> "ModelSpec":
        """Back-compat migration: a config that only sets the legacy
        `ready_timeout_seconds` field gets that value used as the
        startup budget, with pending_budget defaulting to a generous
        3600 s so the caller sees a two-tier structure everywhere.
        A config that explicitly sets both new fields is used verbatim."""
        if self.startup_budget_seconds is None:
            object.__setattr__(self, "startup_budget_seconds",
                               self.ready_timeout_seconds)
        if self.pending_budget_seconds is None:
            object.__setattr__(self, "pending_budget_seconds", 3600)
        return self


class ModelsConfig(BaseModel):
    models: list[ModelSpec]


class DatasetSpec(BaseModel):
    id: str
    split: str | None = None
    file_prefix: str
    # hf_source / prompt_template are Optional because bundled-data
    # loaders (currently: totrandom) read a JSON blob from the image
    # rather than fetching from HF, and the record's prompt is already
    # fully formed — no template to apply. Every existing HF loader
    # still requires both fields at call time; making them optional at
    # the schema layer just lets bundled loaders omit them cleanly.
    hf_source: str | None = None
    hf_config: str | None = None
    hf_split: str = "train"
    category: str
    scorer: str
    prompt_template: str | None = None
    question_field: str | None = None
    answer_field: str | None = None
    expected_count: int | None = None


class DatasetsConfig(BaseModel):
    datasets: list[DatasetSpec]


class HostedProxyEntry(BaseModel):
    """A per-model hosted-provider price row.

    Two shapes are accepted:
      * Explicit prices: `input_price_per_MTok` and `output_price_per_MTok`
        set directly on the row.
      * Tier reference: `tier: <name>` pointing at a `tiers` entry in the
        same PricingConfig. Resolved at load time.

    Explicit prices win over tier lookup when both are present, so a
    specific model with a public quote can override its tier.
    """

    input_price_per_MTok: float | None = None
    output_price_per_MTok: float | None = None
    tier: str | None = None
    # underscore-prefixed metadata is preserved but not enforced
    class Config:
        extra = "allow"


class TierEntry(BaseModel):
    input_price_per_MTok: float
    output_price_per_MTok: float

    class Config:
        extra = "allow"


class PricingConfig(BaseModel):
    gpu_hourly_rate_usd: dict[str, float]
    hosted_proxy: dict[str, HostedProxyEntry]
    # Optional named tiers. If a hosted_proxy entry references a tier,
    # the loader resolves it into concrete prices at load time.
    tiers: dict[str, TierEntry] = Field(default_factory=dict)

    class Config:
        extra = "allow"

    def resolve_tiers(self) -> None:
        """Fill in `input_price_per_MTok` / `output_price_per_MTok` on
        every hosted_proxy entry that only references a tier. Idempotent.
        Explicit per-model prices are preserved.

        Raises `ValueError` if an entry references a tier that isn't in
        `tiers`, or if an entry has neither explicit prices nor a tier.
        """
        for model_id, entry in self.hosted_proxy.items():
            if entry.input_price_per_MTok is not None and \
                    entry.output_price_per_MTok is not None:
                continue  # already has concrete prices
            if entry.tier is None:
                raise ValueError(
                    f"hosted_proxy[{model_id!r}] has neither explicit prices "
                    f"nor a tier reference"
                )
            tier = self.tiers.get(entry.tier)
            if tier is None:
                raise ValueError(
                    f"hosted_proxy[{model_id!r}] references unknown "
                    f"tier {entry.tier!r} (known tiers: "
                    f"{sorted(self.tiers.keys())})"
                )
            entry.input_price_per_MTok = tier.input_price_per_MTok
            entry.output_price_per_MTok = tier.output_price_per_MTok


class ArrivalStage(BaseModel):
    # A fixed-rate segment of the open_loop_staged driver.
    # `lambda_` is arrivals per second (mean rate); `rows` is how many
    # spawn iterations use this rate before advancing to the next stage.
    #
    # `weibull_shape`, when set, switches this stage's inter-arrival gaps
    # from exponential (Poisson arrivals) to Weibull with the given shape
    # parameter. The scale is chosen so E[gap] = 1/lambda_, so offered
    # load stays fixed and only the burstiness (CV of gaps) changes.
    # Shape < 1 => bursty (CV > 1); shape = 1 => exponential (Poisson);
    # shape > 1 => sub-Poisson. Matches the simulation study's Weibull
    # arrival regimes at sim/queue_sim.py:161.
    lambda_: float = Field(alias="lambda", gt=0.0)
    rows: int = Field(gt=0)
    weibull_shape: float | None = Field(default=None, gt=0.0)

    model_config = {"populate_by_name": True}


class ExperimentConfig(BaseModel):
    repeated_trials: int = 5
    repeated_trial_dataset: str = "aime"
    consolidated_datasets: list[str] = Field(default_factory=lambda: ["aime", "gpqa"])
    # Upper bound on the number of in-flight requests. Under the closed
    # loop Poisson driver this is a runaway guard, not the load-shaping
    # knob; `target_concurrent` sets E[N]. Under the open_loop_staged
    # driver, this same semaphore IS the divergence guard: when
    # arrivals persistently outpace service, fleet in-flight rises
    # until it hits this ceiling, at which point further dispatches
    # block in the driver instead of piling into vLLM.
    max_concurrent_requests: int = 128
    # Target expected number of in-flight requests. In closed_loop
    # mode: the runner sets λ = target_concurrent / s_hat by Little's
    # law, and dispatches the first `target_concurrent` rows
    # back-to-back to prime the estimator. In open_loop_staged mode:
    # this field is advisory (kept for backward compatibility with
    # existing configs); λ is set explicitly per stage.
    target_concurrent: int = 12
    # Smoothing on the service-time EWMA (closed_loop only):
    # s_hat_new = alpha * elapsed + (1 - alpha) * s_hat_old.
    # Small alpha (0.05) means slow drift, resistant to per-request
    # noise; helpful because CostGuard shifts the arm distribution over
    # a run, and the mean service time drifts with it.
    service_time_ewma_alpha: float = 0.05
    # Seed for the arrival-process RNG. None = nondeterministic. Set
    # for reproducible arrival traces across runs (useful when
    # comparing arm-A-only vs arm-B-only baselines at the same load).
    arrival_seed: int | None = None
    # Arrival-process regime. "closed_loop" preserves the historical
    # behavior (λ = target_concurrent / s_hat with EWMA feedback).
    # "open_loop_staged" runs the driver at a fixed λ per stage,
    # independent of observed service time — arrivals can outpace
    # service and pile into vLLM's num_requests_waiting (bounded by
    # max_concurrent_requests as the divergence guard).
    arrival_mode: str = "closed_loop"
    # Stage schedule for open_loop_staged. Required when arrival_mode
    # is "open_loop_staged"; ignored otherwise. Each stage has a fixed
    # λ (arrivals/s) and a row count; the driver advances to the next
    # stage after that many spawn iterations. Sum of `rows` must equal
    # the running dataset's `expected_count`.
    arrival_stages: list[ArrivalStage] | None = None
    request_timeout_seconds: int = 600
    retry_max_attempts: int = 3
    retry_backoff_seconds: float = 5.0
    router_service: str = "pricing-reversal-router-epp"
    router_port: int = 80
    heartbeat_interval_seconds: float = 5.0
    ready_probe_prompt: str = "Reply with the single word: READY"
    ready_probe_max_tokens: int = 8
    ready_probe_deadline_seconds: int = 60
    # When the router returns HTTP 503 with "failed to find endpoint
    # candidates" (typically because our vLLM Pod was preempted and is
    # mid-reschedule), the runner pauses and dispatches a shared re-probe.
    # This is the max wall-clock the probe waits before declaring the
    # backend permanently gone. With Phase-A priority + RWX PVC in place,
    # 300 s is comfortably longer than a realistic reschedule.
    no_endpoints_recovery_seconds: int = 300
    # If ≥ this fraction of a run's queries error out, treat the run as
    # environmentally lost and skip the model's remaining datasets +
    # repeated trials. 0.98 leaves headroom for genuinely-hard queries
    # in a healthy run.
    all_errored_threshold: float = 0.98
    # Reserved tokens on top of the counted prompt when computing a per-
    # query max_tokens ceiling. Accounts for chat-template overhead
    # (system framing, role markers) that AutoTokenizer.encode of the
    # raw user prompt does not include, plus a safety margin for
    # tokenizer disagreements with vLLM's server-side count.
    max_tokens_headroom: int = 256
    # Startup budget for the router/EPP Deployment. Router pods don't
    # need a GPU so their pending phase is negligible; we skip the
    # pending budget for them entirely and use this whole budget as
    # startup. Legacy hardcoded value was 600s.
    router_startup_budget_seconds: int = 600
    # OpenAI-compatible `reasoning_effort` sent on every chat-completion
    # request. vLLM maps this to a chat_template_kwargs injection
    # (`enable_thinking: true` for low/medium/high, false for "none").
    # Templates that don't declare `enable_thinking` silently ignore
    # it, so it's safe to set unconditionally. Set to null to omit
    # the field entirely.
    reasoning_effort: str | None = "high"
    # Cross-study resume: absolute path INSIDE the container where the
    # uber has staged a prior study's `consolidated/` and
    # `repeated_trial/` subtrees. When set, each worker's orchestrator
    # copies matching per-(model, dataset[, run_index]) seed files from
    # that path into its own `results_dir` on startup, and the
    # per-dataset resume check in `resume.py` skips runs whose seed is
    # already valid (content + data_fingerprint match). Set by the
    # uber before spawning workers via a ConfigMap patch — the source
    # of truth is the operator's `--resume-from` CLI arg. Null means
    # "no resume; start every dataset from scratch."
    resume_from: str | None = None

    @field_validator("reasoning_effort")
    @classmethod
    def _reasoning_effort_is_recognized(cls, v: str | None) -> str | None:
        if v is None:
            return v
        allowed = {"low", "medium", "high", "none"}
        if v not in allowed:
            raise ValueError(
                f"experiment.reasoning_effort={v!r} is not one of {sorted(allowed)}"
            )
        return v

    @field_validator("arrival_mode")
    @classmethod
    def _arrival_mode_is_recognized(cls, v: str) -> str:
        allowed = {"closed_loop", "open_loop_staged"}
        if v not in allowed:
            raise ValueError(
                f"experiment.arrival_mode={v!r} is not one of {sorted(allowed)}"
            )
        return v

    @model_validator(mode="after")
    def _arrival_stages_match_mode(self) -> "ExperimentConfig":
        if self.arrival_mode == "open_loop_staged":
            if not self.arrival_stages:
                raise ValueError(
                    "experiment.arrival_mode='open_loop_staged' requires "
                    "arrival_stages to be a non-empty list of "
                    "{lambda, rows} entries"
                )
        # In closed_loop mode arrival_stages is ignored; do not error
        # if it is present so a config can carry both fields.
        return self


class HarnessConfig(BaseModel):
    models: ModelsConfig
    datasets: DatasetsConfig
    pricing: PricingConfig
    experiment: ExperimentConfig


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def load_config(config_dir: str | Path) -> HarnessConfig:
    """Load and validate all four config files from a ConfigMap-mounted dir.

    Also resolves pricing.yaml tier references so downstream code
    (cost.py) only ever sees concrete `input_price_per_MTok` /
    `output_price_per_MTok` values on every hosted_proxy entry.
    """

    root = Path(config_dir)
    pricing = PricingConfig.model_validate(_load_yaml(root / "pricing.yaml"))
    pricing.resolve_tiers()
    return HarnessConfig(
        models=ModelsConfig.model_validate(_load_yaml(root / "models.yaml")),
        datasets=DatasetsConfig.model_validate(_load_yaml(root / "datasets.yaml")),
        pricing=pricing,
        experiment=ExperimentConfig.model_validate(
            _load_yaml(root / "experiment.yaml").get("experiment", {})
        ),
    )
