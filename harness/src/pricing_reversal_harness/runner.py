"""Bounded-concurrency runner for one (model, dataset[, run_index]) pass.

Resumes from an existing journal on startup. Every completed record is
appended + fsynced before the next task starts. Records mirror the paper's
schema; a repeated-trial pass sets `score`/`prediction` explicitly to null
if the caller asks (matches data/repeated_trial/*.json shape).

Preemption resilience: when the router returns HTTP 503 with the
"no endpoint candidates" body (see NoBackendError), a runner-scoped
gate is cleared and one shared re-probe task is dispatched. In-flight
queries `await` the gate before their next attempt; once the probe
succeeds, everyone resumes. If the probe deadline
(`no_endpoints_recovery_seconds`) expires, the runner gives up cleanly
and every awaiting query raises NoBackendError — the orchestrator's
run_all_errored logic then short-circuits the model's remaining passes
rather than filling 5 more datasets with error records.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable

from .config import DatasetSpec, ExperimentConfig, ModelSpec, PricingConfig
from .cost import gpu_hour_cost, hosted_proxy_cost
from .journal import Journal, journal_path
from .progress import Progress
from .reasoning_parsers import count_prompt_tokens, count_thinking_tokens
from .scorers import get_scorer, get_totrandom_scorer_for
from .vllm_client import NoBackendError, VLLMClient

log = logging.getLogger(__name__)


class QueryOutcome:
    __slots__ = ("index", "record")

    def __init__(self, index: int, record: dict[str, Any]) -> None:
        self.index = index
        self.record = record


class DatasetRunner:
    def __init__(
        self,
        *,
        client: VLLMClient,
        model: ModelSpec,
        dataset: DatasetSpec,
        pricing: PricingConfig,
        experiment: ExperimentConfig,
        results_dir: Path,
        progress: Progress,
        run_index: int | None = None,
        preserve_scoring: bool = True,
        clusters_table: dict[str, tuple[int, float]] | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.dataset = dataset
        self.pricing = pricing
        self.experiment = experiment
        self.results_dir = results_dir
        self.progress = progress
        self.run_index = run_index
        # Cross-cluster CostGuard cost attribution: map served-endpoint
        # identifier (either endpoint `name` or `address:port` per how
        # the Hub emits it) to (tp_size, gpu_hourly_rate_usd). Empty /
        # None means "no attribution" — records get gpu_hour_cost=0.0
        # and analyze-run.py flags the miss.
        self.clusters_table = clusters_table
        # For repeated-trial passes we still score deterministically — it is
        # cheap — but the paper's repeated_trial files carry null score fields.
        # If `preserve_scoring=False`, we null those out before journaling.
        self.preserve_scoring = preserve_scoring

        self.journal = Journal(
            journal_path(results_dir, model.id, dataset.id, run_index)
        )

        # ---- preemption-resilience state ---------------------------------
        # `backend_up` is Set when the router has eligible endpoints and
        # Cleared when a NoBackendError has been observed. In-flight queries
        # await this gate before their next chat attempt.
        self._backend_up = asyncio.Event()
        self._backend_up.set()
        # Serializes access to the single-shot re-probe dispatch. Once the
        # gate is cleared, only the *first* caller launches the probe task;
        # subsequent callers just await the gate.
        self._probe_lock = asyncio.Lock()
        self._probe_task: asyncio.Task | None = None
        # `_backend_gave_up` is set when the re-probe hit its deadline
        # without ever seeing a healthy response — meaning the run is
        # effectively lost. All awaiting queries then unblock and raise
        # NoBackendError up to `_process_row`.
        self._backend_gave_up = False

        # ---- Poisson arrival driver state --------------------------------
        # `s_hat` is the EWMA estimate of mean service time (seconds).
        # None until the first completion; while None, the driver is in
        # cold-start (dispatch back-to-back to prime the estimator).
        self._s_hat: float | None = None
        # Wall-clock at run start. `_arrival_ts` on each row is
        # `perf_counter() - _run_started_at`.
        self._run_started_at: float | None = None
        # `_in_flight` counts spawned but not-yet-completed row tasks.
        # Read by the driver at spawn time and stamped onto the record
        # so analyze-run can plot queue-depth vs time.
        self._in_flight: int = 0
        # RNG for inter-arrival draws. Seeded from
        # `experiment.arrival_seed` (None = nondeterministic).
        self._arrival_rng = random.Random(experiment.arrival_seed)

    async def run(self, rows: Iterable[dict[str, Any]]) -> list[QueryOutcome]:
        rows = list(rows)
        completed = self.journal.load_completed_indices()
        pending = [r for r in rows if int(r["index"]) not in completed]
        total = len(rows)
        done_prev = total - len(pending)

        self.progress.emit(
            "run_start",
            model=self.model.id,
            dataset=self.dataset.id,
            run_index=self.run_index,
            total=total,
            already_done=done_prev,
        )

        outcomes: list[QueryOutcome] = []
        counter = {"done": done_prev}

        # Poisson-arrival driver state: reset per run() invocation so a
        # reused runner (rare, but supported) starts cold-start fresh.
        self._s_hat = None
        self._run_started_at = time.perf_counter()
        self._in_flight = 0

        async def _handle(row: dict[str, Any], arrival_ts: float,
                          in_flight_at_arrival: int) -> None:
            # NOTE: self._in_flight is incremented by the DRIVER at spawn
            # time (line ~192), not here. That's because asyncio.create_task
            # only schedules the coroutine; the coroutine body doesn't run
            # until the next scheduler tick. During the cold-start burst
            # the driver spawns `target_concurrent` tasks without yielding,
            # so if we incremented here every arrival would observe
            # in_flight = 0. Decrement stays here — the coroutine owns the
            # end of the lifetime.
            try:
                outcome = await self._process_row(
                    row,
                    arrival_ts=arrival_ts,
                    in_flight_at_arrival=in_flight_at_arrival,
                )
            finally:
                self._in_flight -= 1
            # Feed the service-time estimator. Late-arriving errors carry
            # elapsed too; using it keeps the estimator responsive even if
            # a burst of failures happens.
            elapsed = (outcome.record.get("extra_fields") or {}).get(
                "elapsed_seconds"
            )
            if isinstance(elapsed, (int, float)) and elapsed > 0:
                if self._s_hat is None:
                    self._s_hat = float(elapsed)
                else:
                    a = self.experiment.service_time_ewma_alpha
                    self._s_hat = a * float(elapsed) + (1.0 - a) * self._s_hat
            outcomes.append(outcome)
            counter["done"] += 1
            self.progress.set_heartbeat_detail(
                model=self.model.id,
                dataset=self.dataset.id,
                run_index=self.run_index,
                last_index=outcome.index,
                done=counter["done"],
                total=total,
            )

        # Poisson-arrival driver — branches on experiment.arrival_mode.
        #
        # closed_loop (historical default):
        #   Cold start: dispatch the first `target_concurrent` rows
        #   back-to-back so `_s_hat` primes. After that, wait Exp(λ)
        #   between spawns with λ = target_concurrent / s_hat.
        #   By Little's law E[N] = λ · E[S] = target_concurrent.
        #
        # open_loop_staged:
        #   Iterate through experiment.arrival_stages; each stage
        #   fixes λ (arrivals/s) and a row count. Draw gap = Exp(λ_stage)
        #   per iteration when weibull_shape is unset, or
        #   gap = Weibull(shape, scale) with scale = 1/(λ·Γ(1+1/shape))
        #   when weibull_shape is set (mean gap stays 1/λ, so offered
        #   load matches Poisson at the same λ). s_hat is ignored.
        #   Fleet in-flight is no longer pinned — it is bounded by the
        #   VLLMClient semaphore max_concurrent_requests (default 128).
        #
        # The VLLMClient semaphore stays wired as a runaway guard in
        # closed_loop and as the divergence guard in open_loop_staged.
        target = max(1, self.experiment.target_concurrent)
        arrival_mode = self.experiment.arrival_mode
        stages = list(self.experiment.arrival_stages or [])
        if arrival_mode == "open_loop_staged" and not stages:
            raise ValueError(
                "experiment.arrival_mode='open_loop_staged' requires "
                "non-empty arrival_stages"
            )
        # Open-loop stage cursor: index of current stage and count of
        # rows dispatched in the current stage. Emitted alongside the
        # progress markers whenever we cross a boundary.
        stage_idx = 0
        stage_row_count = 0
        if arrival_mode == "open_loop_staged":
            first = stages[0]
            self.progress.emit(
                "arrival_stage",
                model=self.model.id,
                dataset=self.dataset.id,
                run_index=self.run_index,
                stage_index=0,
                lam=first.lambda_,
                rows=first.rows,
                boundary_row=1,
            )

        tasks: list[asyncio.Task[None]] = []
        for i, row in enumerate(pending):
            arrival_ts = time.perf_counter() - self._run_started_at
            in_flight_now = self._in_flight
            # Increment BEFORE create_task. asyncio.create_task only
            # schedules the coroutine; the coroutine body runs at the
            # next scheduler tick. Without this, cold-start arrivals
            # spawn in a single tick and every one reads in_flight=0.
            # The corresponding decrement lives in _handle's finally.
            self._in_flight += 1
            tasks.append(asyncio.create_task(
                _handle(row, arrival_ts, in_flight_now)
            ))
            if i + 1 >= len(pending):
                break  # last spawn — go straight to drain

            if arrival_mode == "open_loop_staged":
                # Advance the stage cursor. This spawn consumed one
                # row of the current stage; if we have hit the stage's
                # quota, advance to the next stage before drawing the
                # next gap.
                stage_row_count += 1
                if stage_row_count >= stages[stage_idx].rows and \
                        stage_idx + 1 < len(stages):
                    stage_idx += 1
                    stage_row_count = 0
                    nxt = stages[stage_idx]
                    self.progress.emit(
                        "arrival_stage",
                        model=self.model.id,
                        dataset=self.dataset.id,
                        run_index=self.run_index,
                        stage_index=stage_idx,
                        lam=nxt.lambda_,
                        rows=nxt.rows,
                        boundary_row=i + 2,  # 1-indexed next row
                    )
                stage = stages[stage_idx]
                lam = stage.lambda_
                if stage.weibull_shape is not None:
                    # Weibull-shape inter-arrivals. Scale chosen so
                    # E[gap] = 1/lam, holding offered load fixed while
                    # varying only the burstiness. Matches the simulation
                    # study's arrival driver at sim/queue_sim.py:161.
                    shape = stage.weibull_shape
                    scale = 1.0 / (lam * math.gamma(1.0 + 1.0 / shape))
                    gap = self._arrival_rng.weibullvariate(scale, shape)
                else:
                    gap = self._arrival_rng.expovariate(lam)
                await asyncio.sleep(gap)
                continue

            # closed_loop branch (historical behavior).
            if i + 1 < target:
                # Cold-start burst: no inter-arrival wait for the first
                # `target` rows. This primes the concurrency so a real
                # service-time sample can land.
                continue

            # Past the cold-start burst. If s_hat has not been primed yet
            # (no completion has fired), block on the FIRST completion
            # before proceeding — otherwise the driver would spin the
            # spawn loop and pile arrivals faster than the server can
            # answer, driving in-flight far above `target`. Once a
            # completion lands, `_handle` sets `_s_hat` and this
            # `asyncio.wait` returns.
            while self._s_hat is None or self._s_hat <= 0:
                if not tasks:
                    break
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            lam = target / self._s_hat
            gap = self._arrival_rng.expovariate(lam)
            await asyncio.sleep(gap)

        # Drain: dataset iterator exhausted, wait for every in-flight
        # task to complete. No hard wall-clock cutoff.
        if tasks:
            await asyncio.gather(*tasks)

        self.progress.emit(
            "run_end",
            model=self.model.id,
            dataset=self.dataset.id,
            run_index=self.run_index,
            total=total,
            done=counter["done"],
        )
        return outcomes

    def _max_tokens_for(self, prompt: str) -> int:
        """Return the largest safe `max_tokens` for this prompt on this
        model: never above the configured sampling ceiling, and never so
        large that prompt_tokens + max_tokens exceeds max_model_len minus
        the configured headroom. Guarantees at least 1 output token even
        when the prompt approaches the ceiling — vLLM will still reject
        that request server-side, but the harness records a clear
        query_error rather than silently under-serving."""
        prompt_tokens = count_prompt_tokens(self.model.hf_repo, prompt)
        budget = (
            self.model.max_model_len
            - prompt_tokens
            - self.experiment.max_tokens_headroom
        )
        return max(1, min(self.model.sampling.max_tokens, budget))

    async def _process_row(
        self,
        row: dict[str, Any],
        *,
        arrival_ts: float | None = None,
        in_flight_at_arrival: int | None = None,
    ) -> QueryOutcome:
        # `dispatch_by_source_dataset` is the meta scorer used by the
        # totrandom dataset — the actual scorer is chosen per-row based
        # on `row["source_dataset"]` because totrandom pools items from
        # six datasets that each want different scoring rules. Every
        # other dataset resolves its scorer once from `dataset.scorer`
        # and never varies per row.
        if self.dataset.scorer == "dispatch_by_source_dataset":
            scorer = get_totrandom_scorer_for(row)
        else:
            scorer = get_scorer(self.dataset.scorer)
        messages = [{"role": "user", "content": row["prompt"]}]
        max_tokens = self._max_tokens_for(row["prompt"])
        started = time.perf_counter()
        # Wait for the backend to be up before attempting. If it went down
        # mid-run and the re-probe recovered it, we resume here; if the
        # probe gave up, `_backend_up` is set again with `_backend_gave_up`
        # true, and we surface NoBackendError immediately without hitting
        # the router again.
        await self._backend_up.wait()
        if self._backend_gave_up:
            exc = NoBackendError(503, "backend recovery deadline expired")
            elapsed = time.perf_counter() - started
            self._journal_query_error(
                row, elapsed, exc,
                arrival_ts=arrival_ts,
                in_flight_at_arrival=in_flight_at_arrival,
            )
            return QueryOutcome(int(row["index"]),
                                self._error_record(
                                    row, elapsed, exc,
                                    arrival_ts=arrival_ts,
                                    in_flight_at_arrival=in_flight_at_arrival,
                                ))
        try:
            body, elapsed = await self.client.chat(
                model=self.model.served_model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.model.sampling.temperature,
                top_p=self.model.sampling.top_p,
                chat_template_kwargs=self.model.chat_template_kwargs or None,
                reasoning_effort=self.experiment.reasoning_effort,
            )
        except NoBackendError as exc:
            # Trip the shared gate and dispatch a re-probe (once). Then
            # wait for either recovery or give-up, and retry ONCE.
            await self._on_no_backend()
            await self._backend_up.wait()
            if self._backend_gave_up:
                elapsed = time.perf_counter() - started
                self._journal_query_error(
                    row, elapsed, exc,
                    arrival_ts=arrival_ts,
                    in_flight_at_arrival=in_flight_at_arrival,
                )
                return QueryOutcome(int(row["index"]),
                                    self._error_record(
                                        row, elapsed, exc,
                                        arrival_ts=arrival_ts,
                                        in_flight_at_arrival=in_flight_at_arrival,
                                    ))
            try:
                body, elapsed = await self.client.chat(
                    model=self.model.served_model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=self.model.sampling.temperature,
                    top_p=self.model.sampling.top_p,
                    chat_template_kwargs=self.model.chat_template_kwargs or None,
                    reasoning_effort=self.experiment.reasoning_effort,
                )
            except Exception as exc2:  # noqa: BLE001
                # Post-recovery second attempt still failed — journal it.
                elapsed = time.perf_counter() - started
                self._journal_query_error(
                    row, elapsed, exc2,
                    arrival_ts=arrival_ts,
                    in_flight_at_arrival=in_flight_at_arrival,
                )
                return QueryOutcome(int(row["index"]),
                                    self._error_record(
                                        row, elapsed, exc2,
                                        arrival_ts=arrival_ts,
                                        in_flight_at_arrival=in_flight_at_arrival,
                                    ))
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - started
            self._journal_query_error(
                row, elapsed, exc,
                arrival_ts=arrival_ts,
                in_flight_at_arrival=in_flight_at_arrival,
            )
            return QueryOutcome(int(row["index"]),
                                self._error_record(
                                    row, elapsed, exc,
                                    arrival_ts=arrival_ts,
                                    in_flight_at_arrival=in_flight_at_arrival,
                                ))

        message = ((body.get("choices") or [{}])[0]).get("message") or {}
        content = message.get("content") or ""
        usage = body.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)

        thinking_tokens = count_thinking_tokens(
            self.model.hf_repo, message, content
        )

        score, prediction = scorer(content, str(row["ground_truth"]))
        if not self.preserve_scoring:
            score_out = None
            prediction_out = None
        else:
            score_out = score
            prediction_out = prediction

        served_endpoint = body.get("_served_endpoint")
        extra_fields: dict[str, Any] = {
            "elapsed_seconds": elapsed,
            "gpu_hour_cost": gpu_hour_cost(
                self.pricing, elapsed, served_endpoint, self.clusters_table
            ),
            "served_endpoint": served_endpoint,
            "finish_reason": ((body.get("choices") or [{}])[0]).get("finish_reason"),
        }
        if arrival_ts is not None:
            extra_fields["arrival_ts_seconds"] = arrival_ts
        if in_flight_at_arrival is not None:
            extra_fields["in_flight_at_arrival"] = in_flight_at_arrival
        record = {
            "index": int(row["index"]),
            "origin_query": row["origin_query"],
            "prompt": row["prompt"],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "thinking_tokens": thinking_tokens,
            "cost": hosted_proxy_cost(self.pricing, self.model, prompt_tokens, completion_tokens),
            "score": score_out,
            "prediction": prediction_out,
            "ground_truth": row["ground_truth"],
            "raw_output": content,
            "extra_fields": extra_fields,
        }
        try:
            self.journal.append(record)
        except OSError as exc:
            # Journal write failure — typically "Disk quota exceeded"
            # when the shared aggregation PVC is full, or NFS/Spectrum
            # Scale I/O errors under storage pressure. Emit a distinct
            # phase so the aggregator + progress dashboard see a real
            # signal (rather than the process silently dying and the
            # `run()`-level except catching a CancelledError from the
            # async task cascade). Re-raise so `_run_one_model`'s
            # try-except still unwinds cleanly — the orchestrator's
            # top-level `run()` catch then emits `study_error` with the
            # OSError message.
            self.progress.emit(
                "journal_write_error",
                model=self.model.id,
                dataset=self.dataset.id,
                run_index=self.run_index,
                index=int(row["index"]),
                error=str(exc)[:400],
                errno=getattr(exc, "errno", None),
            )
            raise
        return QueryOutcome(record["index"], record)

    # ---- shared no-backend gate machinery -------------------------------
    async def _on_no_backend(self) -> None:
        """Clear the gate and dispatch a one-shot re-probe task. Safe to
        call from many coroutines concurrently — only the first caller
        actually launches the probe."""
        async with self._probe_lock:
            if self._probe_task is not None and not self._probe_task.done():
                # Someone else already dispatched; just await the gate.
                return
            if self._backend_up.is_set():
                self._backend_up.clear()
                self.progress.emit(
                    "backend_no_endpoints",
                    model=self.model.id,
                    dataset=self.dataset.id,
                    run_index=self.run_index,
                )
                self._probe_task = asyncio.create_task(self._reprobe())

    async def _reprobe(self) -> None:
        """Poll the router with a tiny probe until it succeeds or the
        configured recovery deadline expires. Runs at most once per
        outage; other in-flight queries await `_backend_up`."""
        deadline = self.experiment.no_endpoints_recovery_seconds
        try:
            ok = await self.client.wait_ready(
                model=self.model.served_model_name,
                deadline_seconds=deadline,
                probe_prompt=self.experiment.ready_probe_prompt,
                probe_max_tokens=self.experiment.ready_probe_max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            ok = False
            self.progress.emit(
                "backend_reprobe_error",
                model=self.model.id, dataset=self.dataset.id,
                run_index=self.run_index, error=str(exc)[:400],
            )
        if ok:
            self.progress.emit(
                "backend_recovered",
                model=self.model.id,
                dataset=self.dataset.id,
                run_index=self.run_index,
            )
        else:
            self._backend_gave_up = True
            self.progress.emit(
                "backend_gave_up",
                model=self.model.id,
                dataset=self.dataset.id,
                run_index=self.run_index,
                deadline_seconds=deadline,
            )
        # Whether recovered or gave-up, unblock everyone waiting.
        self._backend_up.set()

    def _journal_query_error(
        self,
        row: dict[str, Any],
        elapsed: float,
        exc: Exception,
        *,
        arrival_ts: float | None = None,
        in_flight_at_arrival: int | None = None,
    ) -> None:
        """Emit a query_error progress event and append an error record
        to the journal. Consolidates what used to be duplicated in three
        catch clauses.

        If the journal write itself fails (typically OSError "Disk
        quota exceeded"), we emit a `journal_write_error` phase and
        re-raise so orchestrator-level cleanup sees it — same treatment
        as the success-path journal write. Otherwise a full-disk in the
        middle of writing an error record would silently kill the
        process."""
        self.progress.emit(
            "query_error",
            model=self.model.id,
            dataset=self.dataset.id,
            run_index=self.run_index,
            index=int(row["index"]),
            error=str(exc)[:400],
        )
        try:
            self.journal.append(self._error_record(
                row, elapsed, exc,
                arrival_ts=arrival_ts,
                in_flight_at_arrival=in_flight_at_arrival,
            ))
        except OSError as journal_exc:
            self.progress.emit(
                "journal_write_error",
                model=self.model.id,
                dataset=self.dataset.id,
                run_index=self.run_index,
                index=int(row["index"]),
                error=str(journal_exc)[:400],
                errno=getattr(journal_exc, "errno", None),
            )
            raise

    def _error_record(
        self,
        row: dict[str, Any],
        elapsed: float,
        exc: Exception,
        *,
        arrival_ts: float | None = None,
        in_flight_at_arrival: int | None = None,
    ) -> dict[str, Any]:
        extra_fields: dict[str, Any] = {
            "elapsed_seconds": elapsed,
            "gpu_hour_cost": 0.0,
            "error": str(exc)[:400],
        }
        if arrival_ts is not None:
            extra_fields["arrival_ts_seconds"] = arrival_ts
        if in_flight_at_arrival is not None:
            extra_fields["in_flight_at_arrival"] = in_flight_at_arrival
        return {
            "index": int(row["index"]),
            "origin_query": row["origin_query"],
            "prompt": row["prompt"],
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "thinking_tokens": 0,
            "cost": 0.0,
            "score": None,
            "prediction": None,
            "ground_truth": row["ground_truth"],
            "raw_output": "",
            "extra_fields": extra_fields,
        }
