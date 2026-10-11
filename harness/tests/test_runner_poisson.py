"""Tests for the Poisson-arrival dispatch driver in DatasetRunner.

The driver replaces the old asyncio.gather(*coroutines) fan-out with an
open-loop scheduler. We use a fake VLLMClient — no HTTP — and asyncio's
real event loop to exercise:
  * cold-start burst (first target_concurrent rows dispatch back-to-back)
  * steady-state pacing (λ = target_concurrent / s_hat)
  * end-of-dataset drain (no rows dropped, no double-journal)
  * seeded reproducibility (identical arrival gaps under a fixed seed)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from pricing_reversal_harness.config import (
    DatasetSpec, ExperimentConfig, HostedProxyEntry, ModelSpec, PricingConfig,
    Sampling,
)
from pricing_reversal_harness.progress import Progress
from pricing_reversal_harness.runner import DatasetRunner


def _model() -> ModelSpec:
    return ModelSpec(
        id="test-model",
        hf_repo="test/repo",
        served_model_name="test/repo",
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


def _experiment(**overrides: Any) -> ExperimentConfig:
    return ExperimentConfig(**overrides)


def _rows(n: int) -> list[dict[str, Any]]:
    return [{"index": i, "origin_query": f"q{i}", "prompt": f"prompt-{i}",
             "ground_truth": "42"} for i in range(1, n + 1)]


class FakeClient:
    """Minimal VLLMClient stand-in.

    Each chat() call awaits an asyncio.sleep of `service_time` (default
    0) and returns a canned successful body. `spawn_times_by_index`
    records the wall-clock (loop.time()) at which each row's chat() was
    first invoked — the driver's arrival timing shows up here.
    """

    def __init__(self, service_time: float = 0.0) -> None:
        self.service_time = service_time
        self.spawn_times: dict[int, float] = {}
        self.in_flight_now = 0
        self.max_in_flight = 0
        self._n = 0

    async def chat(self, **kwargs) -> tuple[dict[str, Any], float]:
        self._n += 1
        idx = self._n
        loop = asyncio.get_running_loop()
        self.spawn_times[idx] = loop.time()
        self.in_flight_now += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight_now)
        try:
            if self.service_time > 0:
                await asyncio.sleep(self.service_time)
        finally:
            self.in_flight_now -= 1
        body = {
            "choices": [{"message": {"content": "\\boxed{42}"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 10},
        }
        return body, self.service_time or 0.001

    async def wait_ready(self, **kwargs) -> bool:
        return True


async def test_cold_start_bursts_target_concurrent_rows(tmp_path: Path):
    """First `target_concurrent` rows dispatch back-to-back (no
    inter-arrival wait), so max_in_flight reaches at least
    target_concurrent immediately."""
    target = 8
    client = FakeClient(service_time=0.5)  # slow enough that burst overlaps
    runner = DatasetRunner(
        client=client,
        model=_model(),
        dataset=_dataset(),
        pricing=_pricing(),
        experiment=_experiment(target_concurrent=target, arrival_seed=1),
        results_dir=tmp_path,
        progress=Progress(tmp_path),
    )
    await runner.run(_rows(target))
    assert client.max_in_flight >= target, (
        f"cold-start should burst at least {target} rows into flight, "
        f"observed max={client.max_in_flight}"
    )


async def test_drain_awaits_all_in_flight_no_row_lost(tmp_path: Path):
    """Even with a slow client, run() returns only after every
    dispatched row has completed. No rows dropped, no double-journal."""
    client = FakeClient(service_time=0.1)
    runner = DatasetRunner(
        client=client,
        model=_model(),
        dataset=_dataset(),
        pricing=_pricing(),
        experiment=_experiment(target_concurrent=4, arrival_seed=1),
        results_dir=tmp_path,
        progress=Progress(tmp_path),
    )
    outcomes = await runner.run(_rows(10))
    assert len(outcomes) == 10
    # Journal file has exactly 10 lines.
    journal = tmp_path / "journal" / "test-model" / "aime.jsonl"
    n_lines = sum(1 for _ in journal.open("r", encoding="utf-8"))
    assert n_lines == 10
    # in-flight counter drained to zero.
    assert client.in_flight_now == 0


async def test_arrival_telemetry_in_journal_records(tmp_path: Path):
    """Every success record carries arrival_ts_seconds and
    in_flight_at_arrival in extra_fields. Values are monotone in
    arrival_ts_seconds because rows dispatch in source order."""
    client = FakeClient(service_time=0.0)
    runner = DatasetRunner(
        client=client,
        model=_model(),
        dataset=_dataset(),
        pricing=_pricing(),
        experiment=_experiment(target_concurrent=3, arrival_seed=42),
        results_dir=tmp_path,
        progress=Progress(tmp_path),
    )
    outcomes = await runner.run(_rows(6))
    for o in outcomes:
        ef = o.record.get("extra_fields") or {}
        assert "arrival_ts_seconds" in ef
        assert "in_flight_at_arrival" in ef
        assert isinstance(ef["arrival_ts_seconds"], float)
        assert isinstance(ef["in_flight_at_arrival"], int)
        assert ef["arrival_ts_seconds"] >= 0
        assert ef["in_flight_at_arrival"] >= 0
    # Arrival timestamps are monotone by dispatch order.
    by_index = {o.record["index"]: o.record["extra_fields"]["arrival_ts_seconds"]
                for o in outcomes}
    ordered = [by_index[i] for i in sorted(by_index)]
    assert ordered == sorted(ordered), "arrivals should fire in source order"


async def test_no_runaway_before_first_completion(tmp_path: Path):
    """Regression: without waiting for the first completion after the
    cold-start burst, the driver would loop through all remaining rows
    with s_hat still None, spawning them all without any inter-arrival
    delay — a runaway. This test asserts the driver stops spawning at
    `target` rows until at least one completion has fired.

    With a slow client (service_time=2s) and a dataset larger than
    `target`, after spawning `target` rows the driver must block. The
    max_in_flight observed while the first completion is still pending
    must equal `target` — not `len(pending)`.
    """
    target = 4
    dataset_size = 20  # far more than target
    client = FakeClient(service_time=2.0)
    runner = DatasetRunner(
        client=client,
        model=_model(),
        dataset=_dataset(),
        pricing=_pricing(),
        experiment=_experiment(target_concurrent=target, arrival_seed=1),
        results_dir=tmp_path,
        progress=Progress(tmp_path),
    )
    await runner.run(_rows(dataset_size))
    # Every row was journaled.
    journal = tmp_path / "journal" / "test-model" / "aime.jsonl"
    n_lines = sum(1 for _ in journal.open("r", encoding="utf-8"))
    assert n_lines == dataset_size
    # The peak in-flight (observed by the fake client) must have been
    # around `target`, not the dataset size. Some slack because in
    # steady state Poisson variance may push us slightly above target.
    assert client.max_in_flight <= target + 4, (
        f"driver ran away: max_in_flight={client.max_in_flight}, expected ~{target}"
    )


async def test_in_flight_at_arrival_climbs_during_cold_start(tmp_path: Path):
    """Regression: cold-start arrivals must observe monotonically-
    increasing in_flight because the driver increments the counter at
    spawn time, not inside the coroutine body. Without the sync
    increment every burst arrival reads 0 (bug hit live: all first-25
    rows showed in_flight_at_arrival=0 despite target_concurrent=12).

    The service_time is large enough that no cold-start task can
    complete before the next arrival spawns; so the sequence of
    observed in_flight values for the first `target_concurrent` rows
    must be exactly 0, 1, 2, ..., target-1.
    """
    target = 8
    client = FakeClient(service_time=5.0)  # much longer than the burst
    runner = DatasetRunner(
        client=client,
        model=_model(),
        dataset=_dataset(),
        pricing=_pricing(),
        experiment=_experiment(target_concurrent=target, arrival_seed=1),
        results_dir=tmp_path,
        progress=Progress(tmp_path),
    )
    outcomes = await runner.run(_rows(target))
    by_index = {o.record["index"]:
                o.record["extra_fields"]["in_flight_at_arrival"]
                for o in outcomes}
    observed = [by_index[i] for i in sorted(by_index)]
    assert observed == list(range(target)), (
        f"cold-start in_flight should ramp 0..{target-1}, got {observed}"
    )


async def test_seeded_reproducibility_matches_across_runs(tmp_path: Path):
    """Two runs with the same arrival_seed produce identical
    inter-arrival gaps. We check the sequence of arrival_ts_seconds
    from the extra_fields (past the cold-start burst)."""
    target = 4

    async def _run(subdir: str) -> list[float]:
        d = tmp_path / subdir
        d.mkdir()
        client = FakeClient(service_time=0.01)
        runner = DatasetRunner(
            client=client,
            model=_model(),
            dataset=_dataset(),
            pricing=_pricing(),
            experiment=_experiment(
                target_concurrent=target,
                arrival_seed=1234,
                service_time_ewma_alpha=1.0,  # snap s_hat to last elapsed for stability
            ),
            results_dir=d,
            progress=Progress(d),
        )
        outs = await runner.run(_rows(target * 3))
        # Post-burst gaps: differences between consecutive arrival_ts.
        stamps = sorted(
            (o.record["extra_fields"]["arrival_ts_seconds"] for o in outs),
        )
        return [b - a for a, b in zip(stamps, stamps[1:])]

    gaps_a = await _run("a")
    gaps_b = await _run("b")
    # Cold-start rows all arrive at ~t=0 (differences tiny + noisy in
    # real time). Compare only post-cold-start gaps.
    tail = target  # skip the initial burst
    assert len(gaps_a) == len(gaps_b)
    assert gaps_a[tail:] == pytest.approx(gaps_b[tail:], abs=0.05), (
        "same arrival_seed must produce identical inter-arrival draws"
    )


async def test_service_time_ewma_primes_from_first_completion(tmp_path: Path):
    """After the cold-start burst, s_hat should be non-None. Verified
    indirectly: the driver must not raise ZeroDivisionError even when a
    completion returns very fast, and all rows must journal."""
    client = FakeClient(service_time=0.02)
    runner = DatasetRunner(
        client=client,
        model=_model(),
        dataset=_dataset(),
        pricing=_pricing(),
        experiment=_experiment(target_concurrent=3, arrival_seed=1),
        results_dir=tmp_path,
        progress=Progress(tmp_path),
    )
    outcomes = await runner.run(_rows(12))
    assert len(outcomes) == 12
    # After the burst there was at least one completion, so s_hat is set.
    assert runner._s_hat is not None
    assert runner._s_hat > 0


# ---- open_loop_staged tests ------------------------------------------------


async def test_open_loop_staged_rate_honored(tmp_path: Path):
    """open_loop_staged: with a single stage of lambda=20 arrivals/s,
    sample mean inter-arrival gap over N=200 spawns should be within
    20% of 1/lambda=0.05s. Uses zero service time so the driver's
    sleep is the only pacing signal."""
    n = 200
    lam = 20.0
    client = FakeClient(service_time=0.0)
    runner = DatasetRunner(
        client=client,
        model=_model(),
        dataset=_dataset(),
        pricing=_pricing(),
        experiment=_experiment(
            arrival_mode="open_loop_staged",
            arrival_stages=[{"lambda": lam, "rows": n}],
            arrival_seed=42,
        ),
        results_dir=tmp_path,
        progress=Progress(tmp_path),
    )
    outcomes = await runner.run(_rows(n))
    assert len(outcomes) == n
    ts = sorted(o.record["extra_fields"]["arrival_ts_seconds"]
                for o in outcomes)
    gaps = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    mean_gap = sum(gaps) / len(gaps)
    expected = 1.0 / lam
    # 20% band is generous but tight enough to catch a broken rate.
    assert 0.8 * expected < mean_gap < 1.2 * expected, (
        f"mean gap {mean_gap:.4f}s vs expected {expected:.4f}s "
        f"(1/lambda where lambda={lam})"
    )


async def test_open_loop_staged_multi_stage_transitions(tmp_path: Path):
    """Two stages at very different lambdas: rate should visibly
    change at the boundary. First 60 gaps ~ 1/10=0.1s, next 60 ~
    1/100=0.01s."""
    client = FakeClient(service_time=0.0)
    runner = DatasetRunner(
        client=client,
        model=_model(),
        dataset=_dataset(),
        pricing=_pricing(),
        experiment=_experiment(
            arrival_mode="open_loop_staged",
            arrival_stages=[
                {"lambda": 10.0, "rows": 60},
                {"lambda": 100.0, "rows": 60},
            ],
            arrival_seed=7,
        ),
        results_dir=tmp_path,
        progress=Progress(tmp_path),
    )
    outcomes = await runner.run(_rows(120))
    assert len(outcomes) == 120
    ts = sorted(o.record["extra_fields"]["arrival_ts_seconds"]
                for o in outcomes)
    gaps = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    # Skip the last gap of the first stage (transition may straddle).
    first_stage = gaps[:55]
    # Skip the first gap of the second stage similarly.
    second_stage = gaps[65:]
    mean_first = sum(first_stage) / len(first_stage)
    mean_second = sum(second_stage) / len(second_stage)
    assert 0.7 * 0.1 < mean_first < 1.3 * 0.1, (
        f"stage-1 mean gap {mean_first:.4f}s should be ~0.1s"
    )
    assert 0.5 * 0.01 < mean_second < 1.5 * 0.01, (
        f"stage-2 mean gap {mean_second:.4f}s should be ~0.01s"
    )
    # Sanity: the two means are strongly separated.
    assert mean_second < mean_first / 3, (
        f"stage-2 mean {mean_second:.4f}s should be well below "
        f"stage-1 mean {mean_first:.4f}s"
    )


async def test_open_loop_staged_open_loop_is_truly_open(tmp_path: Path):
    """open_loop_staged does NOT throttle on service time. With
    lambda=50 arrivals/s and service_time=0.5s (so per-arm effective
    rate is 2/s), fleet in-flight rises above target_concurrent (=12)
    — bounded only by the VLLMClient semaphore. Closed-loop would
    pin fleet in-flight near target_concurrent regardless."""
    lam = 50.0
    n = 100
    client = FakeClient(service_time=0.5)
    runner = DatasetRunner(
        client=client,
        model=_model(),
        dataset=_dataset(),
        pricing=_pricing(),
        experiment=_experiment(
            arrival_mode="open_loop_staged",
            arrival_stages=[{"lambda": lam, "rows": n}],
            arrival_seed=1,
            target_concurrent=12,  # advisory in open-loop mode
            max_concurrent_requests=128,
        ),
        results_dir=tmp_path,
        progress=Progress(tmp_path),
    )
    await runner.run(_rows(n))
    # In-flight should climb far above target_concurrent because
    # service (2/s fleet) cannot keep up with arrivals (50/s). The
    # semaphore caps at 128 so we do not require an unbounded rise;
    # we just require the driver to demonstrably outpace target.
    assert client.max_in_flight > 12 * 2, (
        f"open-loop should let in-flight rise well above target=12, "
        f"observed max={client.max_in_flight}"
    )


async def test_open_loop_staged_config_validation():
    """arrival_mode='open_loop_staged' without arrival_stages must
    fail validation up front."""
    with pytest.raises(Exception):
        ExperimentConfig(
            arrival_mode="open_loop_staged",
            arrival_stages=None,
        )
    with pytest.raises(Exception):
        ExperimentConfig(
            arrival_mode="open_loop_staged",
            arrival_stages=[],
        )
    # Unrecognized arrival_mode also fails.
    with pytest.raises(Exception):
        ExperimentConfig(arrival_mode="rubbish")


async def test_closed_loop_default_still_works(tmp_path: Path):
    """Regression: default arrival_mode remains closed_loop; existing
    tests keep passing without any config change."""
    ec = _experiment()
    assert ec.arrival_mode == "closed_loop"
    assert ec.arrival_stages is None
    # And a small run through the closed-loop path still journals.
    client = FakeClient(service_time=0.05)
    runner = DatasetRunner(
        client=client,
        model=_model(),
        dataset=_dataset(),
        pricing=_pricing(),
        experiment=_experiment(target_concurrent=3, arrival_seed=1),
        results_dir=tmp_path,
        progress=Progress(tmp_path),
    )
    outcomes = await runner.run(_rows(9))
    assert len(outcomes) == 9
