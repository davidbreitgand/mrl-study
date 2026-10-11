"""Study orchestrator.

For each model in models.yaml:
  1. Render its vLLM Deployment from the shipped template.
  2. Apply it via the in-cluster API.
  3. Wait for Deployment availability, then probe the router for readiness.
  4. Run the consolidated experiment (one pass per dataset in
     experiment.consolidated_datasets).
  5. Run the repeated-trial experiment (5x AIME by default).
  6. Delete the vLLM Deployment (retains HF-cache PVC).

Path layout on the aggregation PVC (study-scoped so multiple studies
accumulate without wiping):
  --results-dir     = /results (shared root; holds dataset_cache/ and
                      resume-from/ across all studies)
  --results-subdir  = <study-id>/workers/<model-id> for parallel workers
  → self.top_results_dir = /results             (dataset_cache root)
  → self.results_dir     = /results/<study-id>/workers/<model-id>

Every phase transition is journaled to progress.jsonl under
`self.results_dir`.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import jinja2
import yaml

from .config import HarnessConfig, ModelSpec, DatasetSpec, load_config


def _load_clusters_table(path: str) -> dict[str, tuple[int, float]]:
    """Load clusters.yaml into a lookup table for cost attribution.

    Indexes each entry by BOTH `name` and `address:port` so the runner
    can resolve regardless of which form the response header carries
    (see analysis/multi-cluster-epp.md §4 — the served-endpoint header
    reports the address:port that the Hub's Envoy actually dialed).
    """
    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    table: dict[str, tuple[int, float]] = {}
    for e in doc.get("endpoints", []) or []:
        name = e["name"]
        addr_port = f'{e["address"]}:{e["port"]}'
        labels = e.get("labels") or {}
        tp = int(labels["llm-d.ai/tp-size"])
        rate = float(labels["llm-d.ai/gpu-hourly-rate-usd"])
        table[name] = (tp, rate)
        table[addr_port] = (tp, rate)
    return table


# The llm-d-router chart's `llm-d-router.name` helper truncates the Helm
# release name to 40 chars before appending `-epp`, so its Deployment /
# Service objects don't necessarily match `<release>-epp` — they match
# `<release[:40]>-epp`. Long release names get silently shortened. We
# reproduce the rule so the orchestrator's k8s waits + base_url point
# at the object that actually exists.
_CHART_RELEASE_TRUNC = 40


def chart_epp_object_name(release_name: str) -> str:
    """Reproduce the chart's llm-d-router.name helper on the Python side."""
    return f"{release_name[:_CHART_RELEASE_TRUNC]}-epp"
from .datasets import load_dataset
from .journal import (
    Journal,
    finalize_consolidated,
    finalize_repeated_trial,
    journal_path,
    write_manifest,
)
from .progress import Progress
from .resume import check_consolidated_seed, check_repeated_trial_seed
from .runner import DatasetRunner, QueryOutcome
from .vllm_client import VLLMClient


log = logging.getLogger(__name__)


class RunAllErroredError(RuntimeError):
    """Raised by the orchestrator when a `runner.run(...)` returns an
    outcome list where ≥ `all_errored_threshold` records were errored.
    Signals that the environment (router / backend) is broken for this
    model, so the remaining datasets + repeated-trial runs should be
    skipped rather than fill the journal with more error records.

    Distinct from a plain exception: `_run_one_model` catches this
    explicitly and lets the finally-block run its cleanup normally.
    """

    def __init__(self, model_id: str, dataset_id: str, run_index: int | None,
                 error_rate: float, total: int) -> None:
        self.model_id = model_id
        self.dataset_id = dataset_id
        self.run_index = run_index
        self.error_rate = error_rate
        self.total = total
        super().__init__(
            f"model={model_id} dataset={dataset_id} run_index={run_index}: "
            f"{error_rate:.0%} of {total} records errored — skipping remaining passes"
        )


def _is_error_outcome(outcome: QueryOutcome) -> bool:
    return "error" in (outcome.record.get("extra_fields") or {})


def _fraction_errored(outcomes: list[QueryOutcome]) -> float:
    if not outcomes:
        return 0.0
    errs = sum(1 for o in outcomes if _is_error_outcome(o))
    return errs / len(outcomes)


class Orchestrator:
    def __init__(
        self,
        *,
        config_dir: str,
        results_dir: str,
        namespace: str,
        hf_token_secret: str,
        vllm_template_path: str,
        router_service: str,
        router_port: int,
        resume: bool = True,
        dry_run: bool = False,
        only_model: str | None = None,
        results_subdir: str | None = None,
        install_router: bool = False,
        router_values_template_path: str | None = None,
        router_chart: str = "oci://ghcr.io/llm-d/charts/llm-d-router-standalone",
        router_chart_version: str = "v0.9.0",
        clusters_yaml_path: str | None = None,
        skip_model_deploy: bool = False,
    ) -> None:
        self.config_dir = config_dir
        # `results_subdir` scopes ALL output paths under a per-worker prefix.
        # The dataset_cache lives at the top level so workers running in
        # parallel share the same HF cache instead of re-downloading.
        _top_results_dir = Path(results_dir)
        self.top_results_dir = _top_results_dir
        self.results_dir = (
            _top_results_dir / results_subdir if results_subdir else _top_results_dir
        )
        self.namespace = namespace
        self.hf_token_secret = hf_token_secret
        self.vllm_template_path = vllm_template_path
        self.router_service = router_service
        self.router_port = router_port
        self.resume = resume
        self.dry_run = dry_run
        self.only_model = only_model
        self.install_router = install_router
        self.router_values_template_path = router_values_template_path
        self.router_chart = router_chart
        self.router_chart_version = router_chart_version
        # When true, the orchestrator assumes the vLLM Deployment(s) that
        # serve the model already exist elsewhere and skips the local
        # per-model deploy + wait + teardown. Used by cluster-scoped
        # (multicluster) studies where request traffic is fanned out to
        # spoke gateways whose vLLM pods live in their own namespaces,
        # and any Deployment the harness would apply in its own
        # namespace is redundant (and often GPU-starved). The router
        # readiness probe still runs — if the router can't route yet,
        # the study aborts as before.
        self.skip_model_deploy = skip_model_deploy
        # Cross-cluster CostGuard attribution: load the same clusters.yaml
        # the Hub EPP mounts, index by both `name` and `address:port`, and
        # hand it to every DatasetRunner so per-request cost is computed
        # from the arm that ACTUALLY served the request (per response
        # header `x-conformance-test-served-endpoint`).
        if clusters_yaml_path:
            self.clusters_table = _load_clusters_table(clusters_yaml_path)
        else:
            self.clusters_table = {}

        self.cfg: HarnessConfig = load_config(config_dir)
        self.progress = Progress(self.results_dir)
        (self.results_dir / "consolidated").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "repeated_trial").mkdir(parents=True, exist_ok=True)
        # dataset_cache is shared across workers (top-level).
        (self.top_results_dir / "dataset_cache").mkdir(parents=True, exist_ok=True)

        # Import kubernetes lazily so dry-run mode + unit tests don't need
        # the module installed. Production Pods have it via pyproject.toml.
        if dry_run:
            self.k8s = None
        else:
            from .k8s import K8sClient
            self.k8s = K8sClient(namespace)
        self._vllm_template = jinja2.Template(
            Path(vllm_template_path).read_text(encoding="utf-8"),
            undefined=jinja2.StrictUndefined,
        )
        # Router values template is optional; only loaded when install_router
        # is on. Kept lazy so a missing file doesn't break sequential mode.
        self._router_values_template: jinja2.Template | None = None
        if self.install_router and self.router_values_template_path:
            self._router_values_template = jinja2.Template(
                Path(self.router_values_template_path).read_text(encoding="utf-8"),
                undefined=jinja2.StrictUndefined,
            )

    # ---- entry ----------------------------------------------------------
    def _select_models(self) -> list[ModelSpec]:
        """Return the model list to run — either all of them (sequential
        mode) or one (worker mode filtered by --only-model)."""
        all_models = self.cfg.models.models
        if not self.only_model:
            return all_models
        found = next((m for m in all_models if m.id == self.only_model), None)
        if found is None:
            raise KeyError(
                f"--only-model={self.only_model!r} not in models.yaml; "
                f"known ids: {[m.id for m in all_models]}"
            )
        return [found]

    async def run(self) -> int:
        self.progress.start_heartbeat(self.cfg.experiment.heartbeat_interval_seconds)
        try:
            selected = self._select_models()
            self.progress.emit(
                "study_start",
                namespace=self.namespace,
                models=[m.id for m in selected],
                only_model=self.only_model,
                datasets=self.cfg.experiment.consolidated_datasets,
                repeated_trials=self.cfg.experiment.repeated_trials,
                dry_run=self.dry_run,
                resume=self.resume,
                install_router=self.install_router,
            )
            # Reset any stale `error` field from a previous Job invocation
            # (state.json survives Pod restarts; a resume shouldn't inherit
            # the previous run's crash).
            self.progress.update_state(
                phase="starting",
                models=[m.id for m in selected],
                error=None,
            )

            # Cross-study resume: if the uber has set experiment.resume_from,
            # copy matching seed files from that path into this worker's
            # own results_dir BEFORE running any model. The per-dataset
            # resume check in _run_consolidated / _run_repeated_trial then
            # sees them at the normal output path and skips accordingly.
            self._stage_resume_seeds(selected)

            for model in selected:
                await self._run_one_model(model)

            manifest_path = write_manifest(self.results_dir / "consolidated")
            if manifest_path is None:
                # No consolidated outputs produced (skip-not-ready,
                # skip-probe-failed, skip-all-errored, or a
                # zero-dataset config). Emit a distinct phase so the
                # aggregator + `make results` can tell empty from
                # successful, and exit non-zero so the Kubernetes Job
                # goes Failed rather than Complete — otherwise a run
                # with no data at all is reported as success by every
                # downstream tool.
                self.progress.emit(
                    "study_no_outputs",
                    models=[m.id for m in selected],
                )
                self.progress.update_state(phase="no_outputs")
                return 3
            self.progress.emit("study_complete")
            self.progress.update_state(phase="complete")
            return 0
        except Exception as exc:  # noqa: BLE001
            self.progress.emit("study_error", error=str(exc)[:400])
            self.progress.update_state(phase="error", error=str(exc)[:400])
            raise
        finally:
            self.progress.stop_heartbeat()

    # ---- per-model ------------------------------------------------------
    def _stage_resume_seeds(self, selected: list[ModelSpec]) -> None:
        """Copy seed files from experiment.resume_from into this study's
        own results_dir for the selected models.

        The uber has already staged a prior study's `consolidated/` and
        `repeated_trial/` subtrees at ``resume_from`` on the shared
        PVC. Here we copy the subset that belongs to the models this
        worker is running (typically one model per worker) into the
        target output path. Per-dataset validation still happens later
        in `_run_consolidated` / `_run_repeated_trial` via resume.py —
        a copied-but-invalid seed will be re-run and overwritten.

        Read-only: source files at resume_from are copied, not moved.
        Idempotent: overwrites existing target files (a partial local
        seed from a prior failed run gets replaced by the resume-from
        copy, which is the more-trusted source).
        """
        resume_from = self.cfg.experiment.resume_from
        if not resume_from:
            return
        import shutil
        source_dir = Path(resume_from)
        if not source_dir.is_dir():
            # Uber patched the ConfigMap but the staged files aren't
            # present. Warn and continue — every worker will do a
            # fresh run because resume.py will find no seed files.
            self.progress.emit(
                "resume_stage_missing",
                resume_from=str(source_dir),
            )
            return

        selected_ids = {m.id for m in selected}
        n_consolidated = 0
        n_repeated = 0

        # consolidated/<file_prefix>-<model_id>.json — one per (model, dataset).
        src_cons = source_dir / "consolidated"
        if src_cons.is_dir():
            dst_cons = self.results_dir / "consolidated"
            dst_cons.mkdir(parents=True, exist_ok=True)
            for jf in src_cons.glob("*.json"):
                # Filenames match the file_prefix-<model_id>.json pattern.
                # Match on suffix "-<model_id>.json" so we only copy files
                # belonging to models this worker is running.
                stem = jf.stem  # <prefix>-<model_id>
                matched_model = None
                for mid in selected_ids:
                    if stem.endswith(f"-{mid}"):
                        matched_model = mid
                        break
                if matched_model is None:
                    continue
                shutil.copy2(jf, dst_cons / jf.name)
                n_consolidated += 1

        # repeated_trial/<dataset>/<model>/run<N>.json — per-run granular.
        src_rep = source_dir / "repeated_trial"
        if src_rep.is_dir():
            for ds_dir in src_rep.iterdir():
                if not ds_dir.is_dir():
                    continue
                for m_dir in ds_dir.iterdir():
                    if not m_dir.is_dir() or m_dir.name not in selected_ids:
                        continue
                    dst_m_dir = (self.results_dir / "repeated_trial"
                                 / ds_dir.name / m_dir.name)
                    dst_m_dir.mkdir(parents=True, exist_ok=True)
                    for rf in m_dir.glob("run*.json"):
                        shutil.copy2(rf, dst_m_dir / rf.name)
                        n_repeated += 1

        self.progress.emit(
            "resume_seeds_staged",
            resume_from=str(source_dir),
            consolidated=n_consolidated,
            repeated_trial=n_repeated,
            models=sorted(selected_ids),
        )

    async def _run_one_model(self, model: ModelSpec) -> None:
        self.progress.emit("model_start", model=model.id)
        self.progress.update_state(current_model=model.id, phase="model_up")

        client: VLLMClient | None = None
        deployment_applied = False
        router_installed = False
        try:
            # Optional: this worker owns a per-model router Helm release.
            # Sequential mode leaves install_router=False and the router
            # is assumed to be installed externally.
            if self.install_router and not self.dry_run:
                self._helm_install_router(model)
                router_installed = True
                # Wait for the router EPP to be Ready before doing anything
                # that talks to its Service. Router pods don't need a GPU,
                # so we skip the pending budget and treat the whole wait as
                # startup.
                outcome = self.k8s.wait_deployment_available(
                    self.router_service,
                    pending_budget_seconds=None,
                    startup_budget_seconds=self.cfg.experiment.router_startup_budget_seconds,
                )
                if not outcome.ok:
                    self.progress.emit(
                        "router_not_ready", model=model.id,
                        router=self.router_service,
                        reason=outcome.reason,
                        elapsed_pending=outcome.elapsed_pending,
                        elapsed_startup=outcome.elapsed_startup,
                    )
                    return

            if not self.dry_run and not self.skip_model_deploy:
                self._deploy_model(model)
                deployment_applied = True
                outcome = self.k8s.wait_deployment_available(
                    f"pricing-reversal-vllm-{model.id}",
                    pending_budget_seconds=model.pending_budget_seconds,
                    startup_budget_seconds=model.startup_budget_seconds,
                )
                if not outcome.ok:
                    # Two distinct failure modes get two distinct phases so
                    # an operator can tell "cluster is oversubscribed"
                    # (pending_timeout) from "vLLM startup is genuinely
                    # broken" (startup_timeout). The legacy
                    # `model_skip_not_ready` name is preserved for the
                    # startup case for callers that grep progress files.
                    phase = ("model_skip_pending_timeout"
                             if outcome.reason == "pending_timeout"
                             else "model_skip_not_ready")
                    self.progress.emit(
                        phase, model=model.id,
                        reason=outcome.reason,
                        elapsed_pending=outcome.elapsed_pending,
                        elapsed_startup=outcome.elapsed_startup,
                    )
                    return

            base_url = f"http://{self.router_service}:{self.router_port}"
            client = VLLMClient(
                base_url=base_url,
                timeout_seconds=self.cfg.experiment.request_timeout_seconds,
                max_concurrent=self.cfg.experiment.max_concurrent_requests,
                retry_max_attempts=self.cfg.experiment.retry_max_attempts,
                retry_backoff_seconds=self.cfg.experiment.retry_backoff_seconds,
            )

            if not self.dry_run:
                probe_ok = await client.wait_ready(
                    model=model.served_model_name,
                    deadline_seconds=self.cfg.experiment.ready_probe_deadline_seconds,
                    probe_prompt=self.cfg.experiment.ready_probe_prompt,
                    probe_max_tokens=self.cfg.experiment.ready_probe_max_tokens,
                )
                if not probe_ok:
                    self.progress.emit("model_skip_probe_failed", model=model.id)
                    return

            try:
                for ds_id in self.cfg.experiment.consolidated_datasets:
                    ds = self._get_dataset(ds_id)
                    await self._run_consolidated(client, model, ds)

                repeated_ds = self._get_dataset(self.cfg.experiment.repeated_trial_dataset)
                for run_index in range(self.cfg.experiment.repeated_trials):
                    await self._run_repeated_trial(client, model, repeated_ds, run_index)
            except RunAllErroredError as exc:
                # Skip the model's remaining datasets + repeated trials.
                # The finally-block below still runs and cleans up the
                # Deployment + router as usual.
                self.progress.emit(
                    "model_skip_all_errored",
                    model=model.id, dataset=exc.dataset_id,
                    run_index=exc.run_index,
                    error_rate=exc.error_rate, total=exc.total,
                )
        finally:
            if client is not None:
                await client.aclose()

            if not self.dry_run and deployment_applied:
                self.progress.emit("model_down", model=model.id)
                self.progress.update_state(phase="model_down")
                try:
                    self.k8s.delete_deployment(f"pricing-reversal-vllm-{model.id}")
                    self.k8s.delete_service_account(f"pricing-reversal-vllm-{model.id}")
                except Exception as exc:  # noqa: BLE001
                    # Cleanup best-effort — a raise here would mask a more
                    # interesting exception from the try body above.
                    self.progress.emit(
                        "model_down_cleanup_error",
                        model=model.id, error=str(exc)[:400],
                    )
                # HF-cache PVC retained on purpose (survives to next study).

            if self.install_router and router_installed and not self.dry_run:
                self.progress.emit("router_uninstall", model=model.id)
                try:
                    self._helm_uninstall_router(model)
                except Exception as exc:  # noqa: BLE001
                    self.progress.emit(
                        "router_uninstall_error",
                        model=model.id, error=str(exc)[:400],
                    )

        self.progress.emit("model_complete", model=model.id)

    # ---- helm --------------------------------------------------------
    def _release_name(self, model: ModelSpec) -> str:
        return f"pricing-reversal-router-{model.id}"

    def _helm_install_router(self, model: ModelSpec) -> None:
        """helm upgrade --install per-worker router. Idempotent."""
        if self._router_values_template is None:
            raise RuntimeError(
                "install_router=True but router_values_template_path was not provided"
            )
        values = self._router_values_template.render(
            model_id=model.id,
            namespace=self.namespace,
        )
        release = self._release_name(model)
        cmd = [
            "helm", "upgrade", "--install", release,
            self.router_chart,
            "--version", self.router_chart_version,
            "-n", self.namespace,
            "-f", "-",  # values on stdin
            "--wait", "--timeout", "10m",
        ]
        log.info("helm install %s", release)
        r = subprocess.run(
            cmd, input=values, text=True,
            capture_output=True, timeout=900,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"helm install {release} failed rc={r.returncode}: "
                f"{(r.stderr or r.stdout)[:600]}"
            )
        self.progress.emit("router_installed", model=model.id, release=release)

    def _helm_uninstall_router(self, model: ModelSpec) -> None:
        """helm uninstall the per-worker router. Idempotent — 'not found'
        exit codes are swallowed."""
        release = self._release_name(model)
        log.info("helm uninstall %s", release)
        r = subprocess.run(
            ["helm", "uninstall", release, "-n", self.namespace,
             "--ignore-not-found", "--wait", "--timeout", "5m"],
            capture_output=True, text=True, timeout=360,
        )
        if r.returncode != 0:
            # Log but don't raise — cleanup is best-effort.
            log.warning("helm uninstall %s rc=%d: %s", release, r.returncode,
                        (r.stderr or r.stdout)[:400])

    # ---- deploy helper --------------------------------------------------
    def _deploy_model(self, model: ModelSpec) -> None:
        tp = model.tensor_parallel_size
        rendered = self._vllm_template.render(
            namespace=self.namespace,
            hf_token_secret=self.hf_token_secret,
            model_id=model.id,
            hf_repo=model.hf_repo,
            served_model_name=model.served_model_name,
            tensor_parallel_size=tp,
            reasoning_parser=model.reasoning_parser,
            max_model_len=model.max_model_len,
            gpu_memory_utilization=model.gpu_memory_utilization,
            extra_vllm_args=model.extra_vllm_args,
            hf_cache_size_gi=model.hf_cache_size_gi,
            cpu_request=8 * tp,
            cpu_limit=16 * tp,
            memory_request_gi=64 * tp,
            memory_limit_gi=128 * tp,
        )
        self.k8s.apply_manifest(rendered)
        self.progress.emit("model_deployment_applied", model=model.id)

    # ---- consolidated pass ---------------------------------------------
    async def _run_consolidated(
        self, client: VLLMClient, model: ModelSpec, ds: DatasetSpec
    ) -> None:
        self.progress.emit("consolidated_start", model=model.id, dataset=ds.id)
        self.progress.update_state(phase="consolidated", current_dataset=ds.id)

        rows = list(load_dataset(ds, str(self.top_results_dir / "dataset_cache")))

        # Cross-study resume check: if a valid seed file for this
        # (model, dataset) pair already lives at the target output path
        # (staged there by `make resume-from`), skip the actual run.
        # The seed is validated on counts + data_fingerprint; a
        # malformed / drift-affected seed is removed and we fall
        # through to a fresh run.
        seed_out = self.results_dir / "consolidated" / f"{ds.file_prefix}-{model.id}.json"
        ok, reason = check_consolidated_seed(
            seed_out,
            fresh_rows=rows,
            expected_count=ds.expected_count,
        )
        if ok:
            self.progress.emit(
                "consolidated_skipped_resume",
                model=model.id, dataset=ds.id,
                output=str(seed_out), reason=reason,
            )
            return
        if seed_out.exists() or reason != "missing or malformed":
            # `check_consolidated_seed` deletes an invalid seed but a
            # "missing or malformed" reason on a truly-missing file is
            # the common case (no seed present) and shouldn't be
            # logged as a resume rejection — only real drift is.
            self.progress.emit(
                "consolidated_resume_rejected",
                model=model.id, dataset=ds.id, reason=reason,
            )

        runner = DatasetRunner(
            client=client,
            model=model,
            dataset=ds,
            pricing=self.cfg.pricing,
            experiment=self.cfg.experiment,
            results_dir=self.results_dir,
            progress=self.progress,
            run_index=None,
            preserve_scoring=True,
            clusters_table=self.clusters_table,
        )
        started = time.perf_counter()
        outcomes = await runner.run(rows)
        elapsed = time.perf_counter() - started

        out = self.results_dir / "consolidated" / f"{ds.file_prefix}-{model.id}.json"
        journal = Journal(journal_path(self.results_dir, model.id, ds.id, None))
        finalize_consolidated(
            journal,
            out,
            model_name=model.id,
            dataset_name=ds.id,
            split=ds.split,
            elapsed_seconds=elapsed,
        )
        self.progress.emit("consolidated_done", model=model.id, dataset=ds.id, output=str(out))

        err_rate = _fraction_errored(outcomes)
        if err_rate >= self.cfg.experiment.all_errored_threshold and outcomes:
            raise RunAllErroredError(
                model_id=model.id, dataset_id=ds.id, run_index=None,
                error_rate=err_rate, total=len(outcomes),
            )

    # ---- repeated-trial pass -------------------------------------------
    async def _run_repeated_trial(
        self, client: VLLMClient, model: ModelSpec, ds: DatasetSpec, run_index: int
    ) -> None:
        self.progress.emit(
            "repeated_trial_start",
            model=model.id, dataset=ds.id, run_index=run_index,
        )
        self.progress.update_state(
            phase="repeated_trial", current_dataset=ds.id, run_index=run_index
        )

        rows = list(load_dataset(ds, str(self.top_results_dir / "dataset_cache")))

        # Cross-study resume check for this specific run_index. Per-run
        # granularity: if run_index=2 has a valid seed but run_index=3
        # doesn't, we skip only run_index=2 and run 3 fresh.
        seed_out = (self.results_dir / "repeated_trial" / ds.id / model.id
                    / f"run{run_index}.json")
        ok, reason = check_repeated_trial_seed(
            seed_out,
            fresh_rows=rows,
            expected_count=ds.expected_count,
        )
        if ok:
            self.progress.emit(
                "repeated_trial_skipped_resume",
                model=model.id, dataset=ds.id, run_index=run_index,
                output=str(seed_out), reason=reason,
            )
            return
        if seed_out.exists() or reason != "missing or malformed":
            self.progress.emit(
                "repeated_trial_resume_rejected",
                model=model.id, dataset=ds.id, run_index=run_index,
                reason=reason,
            )

        runner = DatasetRunner(
            client=client,
            model=model,
            dataset=ds,
            pricing=self.cfg.pricing,
            experiment=self.cfg.experiment,
            results_dir=self.results_dir,
            progress=self.progress,
            run_index=run_index,
            preserve_scoring=True,
            clusters_table=self.clusters_table,
        )
        outcomes = await runner.run(rows)

        out_dir = self.results_dir / "repeated_trial" / ds.id / model.id
        out = out_dir / f"run{run_index}.json"
        journal = Journal(journal_path(self.results_dir, model.id, ds.id, run_index))
        finalize_repeated_trial(
            journal,
            out,
            model_name=model.id,
            dataset_name=ds.id,
            split=ds.split,
            run_index=run_index,
        )
        self.progress.emit(
            "repeated_trial_done",
            model=model.id, dataset=ds.id, run_index=run_index, output=str(out),
        )

        err_rate = _fraction_errored(outcomes)
        if err_rate >= self.cfg.experiment.all_errored_threshold and outcomes:
            raise RunAllErroredError(
                model_id=model.id, dataset_id=ds.id, run_index=run_index,
                error_rate=err_rate, total=len(outcomes),
            )

    # ---- helpers --------------------------------------------------------
    def _get_dataset(self, ds_id: str) -> DatasetSpec:
        for d in self.cfg.datasets.datasets:
            if d.id == ds_id:
                return d
        raise KeyError(f"dataset {ds_id!r} not in datasets.yaml")
