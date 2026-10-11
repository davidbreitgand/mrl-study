"""Uber controller: reconcile, fan-out, watch, sweep, verify, finalize.

Design invariants:
  * Every study is scoped by a `study_id` (YYYY-MM-DDTHH-MM-SSZ) generated
    once at UberController construction, so multiple studies accumulate
    side-by-side on the same PVC without wiping between runs.
  * Uber writes only under /results/<study-id>/uber/ and
    /results/<study-id>/{consolidated,repeated_trial}/.
  * Workers write only under /results/<study-id>/workers/<model-id>/.
  * dataset_cache/ and resume-from/ live at /results/ top level, shared
    across studies (the harness's `top_results_dir` points at the raw
    --results-dir, unaffected by the per-study prefix in --results-subdir).
  * All mutations of Jobs/Deployments/Helm releases go through the K8s API
    or `helm` subprocess; nothing is done imperatively without an emitted
    progress event first.

Failure semantics:
  * Worker Job crash → Job restartPolicy: OnFailure handles it.
  * Uber Pod crash → uber Job restartPolicy: OnFailure handles it; on restart
    the reconcile pass attaches to existing worker Jobs rather than re-creating.
  * A worker's vLLM Deployment leaked after its Pod is dead → uber's
    per-terminal sweep + final `verify_all_gpus_freed()` pick it up.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jinja2
from kubernetes.client.exceptions import ApiException

from ..config import HarnessConfig, ModelSpec, load_config
from ..progress import Progress
from .aggregator import Aggregator


log = logging.getLogger(__name__)

LABEL_GUIDE = "llm-d.ai/guide=pricing-reversal"
LABEL_COMPONENT_WORKER = "pricing-reversal.study/component=worker"


def worker_job_name(model_id: str) -> str:
    return f"pricing-reversal-worker-{model_id}"


def vllm_deployment_name(model_id: str) -> str:
    return f"pricing-reversal-vllm-{model_id}"


def router_release_name(model_id: str) -> str:
    return f"pricing-reversal-router-{model_id}"


def _is_job_complete(job: Any) -> bool:
    """True if the Job's status has a Complete condition with status True."""
    st = getattr(job, "status", None)
    if st is None:
        return False
    for c in (st.conditions or []):
        if c.type == "Complete" and c.status == "True":
            return True
    return bool(st.succeeded and st.succeeded >= 1)


def _is_job_failed(job: Any) -> bool:
    st = getattr(job, "status", None)
    if st is None:
        return False
    for c in (st.conditions or []):
        if c.type == "Failed" and c.status == "True":
            return True
    return False


def _job_summary(job: Any) -> dict[str, Any]:
    """Build a compact summary for the aggregated dashboard."""
    st = getattr(job, "status", None) or type("S", (), {})()
    if _is_job_complete(job):
        js = "Complete"
    elif _is_job_failed(job):
        js = "Failed"
    elif getattr(st, "active", 0) or 0 > 0:
        js = "Active"
    else:
        js = "Pending"
    return {
        "job_status": js,
        "active": int(getattr(st, "active", 0) or 0),
        "succeeded": int(getattr(st, "succeeded", 0) or 0),
        "failed": int(getattr(st, "failed", 0) or 0),
    }


class UberController:
    def __init__(
        self,
        *,
        config_dir: str,
        results_dir: str,
        namespace: str,
        hf_token_secret: str,
        worker_image: str,
        worker_template_path: str,
        router_values_template_path: str,
        aggregator_interval_seconds: float = 5.0,
        dry_run: bool = False,
        image_pull_policy: str = "Always",
        resume_from: str | None = None,
        study_id: str | None = None,
    ) -> None:
        self.config_dir = config_dir
        self.results_dir = Path(results_dir)
        self.namespace = namespace
        self.hf_token_secret = hf_token_secret
        self.worker_image = worker_image
        self.worker_template_path = worker_template_path
        self.router_values_template_path = router_values_template_path
        self.aggregator_interval_seconds = aggregator_interval_seconds
        self.dry_run = dry_run
        # Propagated into every worker Job manifest. "Always" is the
        # safe default because a fresh `docker push` to the same tag
        # still needs to reach every worker; hash-tagged builds can
        # override to IfNotPresent to save pull traffic.
        self.image_pull_policy = image_pull_policy
        # Cross-study resume: path INSIDE the container (mounted on the
        # aggregation PVC) where a prior study's `consolidated/` and
        # `repeated_trial/` subtrees have been staged. None = no resume.
        # See `run()` for the ConfigMap-patch that propagates this to
        # workers before they spawn.
        self.resume_from = resume_from

        # Study id — generated once at construction so multiple studies
        # can accumulate on the same PVC. Every uber path and every
        # worker Job template render prefixes its subdir with this id.
        # Tests can inject a fixed value; production always defaults
        # to the current UTC timestamp.
        self.study_id = study_id or datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H-%M-%SZ"
        )

        # Study-scoped root: /results/<study-id>/. Everything the uber
        # writes and every worker's subdir hangs off this.
        self.study_dir = self.results_dir / self.study_id

        self.cfg: HarnessConfig = load_config(config_dir)

        # Uber's own progress + state live under /results/<study-id>/uber/.
        self.uber_dir = self.study_dir / "uber"
        self.uber_dir.mkdir(parents=True, exist_ok=True)
        self.progress = Progress(self.uber_dir)

        # Aggregator merges worker snapshots + promotes finalized outputs
        # under /results/<study-id>/.
        self.aggregator = Aggregator(self.study_dir)

        # K8s client, lazy in dry-run mode.
        if dry_run:
            self.k8s = None
        else:
            from ..k8s import K8sClient
            self.k8s = K8sClient(namespace)

        # Worker Job template.
        self._worker_template = jinja2.Template(
            Path(worker_template_path).read_text(encoding="utf-8"),
            undefined=jinja2.StrictUndefined,
        )

        # Track which workers we've already promoted / swept so those
        # actions don't repeat on every watch tick.
        self._promoted: set[str] = set()
        self._swept: set[str] = set()

    # ---- reconcile ------------------------------------------------------
    def reconcile(self) -> dict[str, Any]:
        """Discover existing worker Jobs (by label) and figure out what
        to create. Returns a dict {model_id: existing_Job_or_None}.
        Idempotent — safe to call from a fresh Pod restart."""
        known = {m.id: None for m in self.cfg.models.models}
        if self.dry_run:
            return known

        for job in self.k8s.list_jobs(label_selector=LABEL_COMPONENT_WORKER):
            labels = (job.metadata.labels or {})
            model_id = labels.get("pricing-reversal.study/model")
            if model_id in known:
                known[model_id] = job
            else:
                self.progress.emit(
                    "orphan_job_ignored",
                    job=job.metadata.name,
                    model=model_id or "?",
                )

        for model_id, job in known.items():
            if job is None:
                self.progress.emit("worker_create_pending", model=model_id)
            elif _is_job_complete(job):
                self.progress.emit("worker_attached_complete", model=model_id)
            elif _is_job_failed(job):
                self.progress.emit("worker_attached_failed", model=model_id)
            else:
                self.progress.emit("worker_attached_active", model=model_id)

        return known

    # ---- fan-out --------------------------------------------------------
    def create_worker_job(self, model: ModelSpec) -> None:
        rendered = self._render_worker_manifest(model)
        if self.dry_run:
            self.progress.emit("worker_dry_run", model=model.id, manifest_bytes=len(rendered))
            return
        self.k8s.apply_manifest(rendered)
        self.progress.emit("worker_created", model=model.id,
                           job=worker_job_name(model.id))

    def _render_worker_manifest(self, model: ModelSpec) -> str:
        return self._worker_template.render(
            namespace=self.namespace,
            model_id=model.id,
            job_name=worker_job_name(model.id),
            worker_image=self.worker_image,
            image_pull_policy=self.image_pull_policy,
            hf_token_secret=self.hf_token_secret,
            router_values_template=self.router_values_template_path,
            router_chart_version="v0.9.0",
            study_id=self.study_id,
        )

    # ---- terminal sweep -------------------------------------------------
    def cleanup_after_worker(self, model_id: str) -> None:
        """Idempotent per-model cleanup. Safe to run multiple times."""
        if model_id in self._swept or self.dry_run:
            return
        self._swept.add(model_id)

        # Deployment + ServiceAccount left by the worker's finally-block failure.
        for kind, name in [
            ("Deployment", vllm_deployment_name(model_id)),
            ("ServiceAccount", vllm_deployment_name(model_id)),
        ]:
            try:
                if kind == "Deployment":
                    existed = self.k8s.deployment_exists(name)
                    self.k8s.delete_deployment(name)
                else:
                    existed = True  # cheap to attempt; delete is idempotent
                    self.k8s.delete_service_account(name)
                if existed:
                    self.progress.emit(
                        "orphan_deleted",
                        model=model_id, kind=kind, name=name,
                    )
            except Exception as exc:  # noqa: BLE001
                self.progress.emit(
                    "orphan_delete_error",
                    model=model_id, kind=kind, error=str(exc)[:400],
                )

        # Helm uninstall — best-effort, idempotent.
        release = router_release_name(model_id)
        rc = subprocess.run(
            ["helm", "uninstall", release, "-n", self.namespace,
             "--ignore-not-found", "--wait", "--timeout", "5m"],
            capture_output=True, text=True, timeout=360,
        )
        if rc.returncode != 0:
            log.warning("helm uninstall %s rc=%d: %s", release, rc.returncode,
                        (rc.stderr or rc.stdout)[:400])

    # ---- verification ---------------------------------------------------
    def verify_all_gpus_freed(self) -> dict[str, Any]:
        """Final sweep before study_complete. Returns a report dict; the
        caller decides whether to accept it or attempt one retry."""
        if self.dry_run:
            return {"ok": True, "residual_gpus": 0, "unclean": [], "dry_run": True}

        unclean: list[dict[str, str]] = []
        # 1. No study-owned vLLM Deployments.
        for m in self.cfg.models.models:
            if self.k8s.deployment_exists(vllm_deployment_name(m.id)):
                unclean.append({"kind": "Deployment",
                                "name": vllm_deployment_name(m.id),
                                "model": m.id})
        # 2. No study-owned decode Pods still Running/Pending/ContainerCreating.
        for p in self.k8s.list_pods(
            label_selector="llm-d.ai/guide=pricing-reversal,llm-d.ai/role=decode"
        ):
            phase = getattr(p.status, "phase", None)
            if phase in {"Running", "Pending", "ContainerCreating"}:
                unclean.append({"kind": "Pod",
                                "name": p.metadata.name,
                                "phase": phase or "?"})
        # 3. No Helm releases still installed for our models.
        for m in self.cfg.models.models:
            release = router_release_name(m.id)
            r = subprocess.run(
                ["helm", "status", release, "-n", self.namespace, "-o", "json"],
                capture_output=True, timeout=30,
            )
            if r.returncode == 0:
                unclean.append({"kind": "HelmRelease", "name": release, "model": m.id})

        # 4. Residual GPUs across every guide-labeled pod.
        residual_gpus = 0
        for p in self.k8s.list_pods(label_selector="llm-d.ai/guide=pricing-reversal"):
            if getattr(p.status, "phase", None) in {"Running", "Pending"}:
                for c in (p.spec.containers or []):
                    req = (getattr(c.resources, "requests", None) or {}).get("nvidia.com/gpu", "0")
                    try:
                        residual_gpus += int(req)
                    except (TypeError, ValueError):
                        pass

        report = {
            "ok": (residual_gpus == 0 and not unclean),
            "residual_gpus": residual_gpus,
            "unclean": unclean,
        }
        self.progress.emit("gpu_verification", **report)
        return report

    # ---- final delete pass ---------------------------------------------
    def final_delete_pass(self) -> None:
        """Best-effort second-chance cleanup if verify found residuals."""
        for m in self.cfg.models.models:
            try:
                self.k8s.delete_deployment(vllm_deployment_name(m.id))
                self.k8s.delete_service_account(vllm_deployment_name(m.id))
            except Exception as exc:  # noqa: BLE001
                log.warning("final delete Deployment %s: %s", m.id, exc)
            release = router_release_name(m.id)
            r = subprocess.run(
                ["helm", "uninstall", release, "-n", self.namespace,
                 "--ignore-not-found", "--wait", "--timeout", "5m"],
                capture_output=True, text=True, timeout=360,
            )
            if r.returncode != 0:
                log.warning("final helm uninstall %s: %s", release,
                            (r.stderr or r.stdout)[:200])

    # ---- watch loop -----------------------------------------------------
    def _model_ids(self) -> list[str]:
        return [m.id for m in self.cfg.models.models]

    def _all_terminal(self, jobs: dict[str, Any]) -> bool:
        for j in jobs.values():
            if j is None:
                return False
            if not (_is_job_complete(j) or _is_job_failed(j)):
                return False
        return True

    def _fetch_jobs(self) -> dict[str, Any]:
        found: dict[str, Any] = {m.id: None for m in self.cfg.models.models}
        for job in self.k8s.list_jobs(label_selector=LABEL_COMPONENT_WORKER):
            labels = (job.metadata.labels or {})
            mid = labels.get("pricing-reversal.study/model")
            if mid in found:
                found[mid] = job
        return found

    def _watch_loop(self) -> dict[str, str]:
        """Poll Jobs every `aggregator_interval_seconds` seconds, refresh
        the aggregated dashboard, promote finalized worker outputs, and
        run the per-terminal sweep. Exits when all workers are terminal.
        Returns a dict {model_id: outcome} where outcome ∈ {Complete, Failed}."""
        outcomes: dict[str, str] = {}
        while True:
            jobs = self._fetch_jobs()

            # Build the aggregator view.
            view: dict[str, dict[str, Any]] = {}
            for mid, j in jobs.items():
                if j is None:
                    view[mid] = {"job_status": "NotCreated"}
                else:
                    view[mid] = _job_summary(j)
            self.aggregator.refresh(view)

            # Promote + sweep on terminal transitions.
            for mid, j in jobs.items():
                if j is None:
                    continue
                if _is_job_complete(j):
                    if mid not in self._promoted:
                        actions = self.aggregator.promote(mid)
                        self.progress.emit(
                            "worker_completed", model=mid,
                            promoted=len(actions),
                        )
                        self._promoted.add(mid)
                        outcomes[mid] = "Complete"
                    self.cleanup_after_worker(mid)
                elif _is_job_failed(j):
                    if mid not in self._promoted:
                        # Even a failed worker may have finalized *some*
                        # datasets before crashing — promote whatever exists.
                        actions = self.aggregator.promote(mid)
                        self.progress.emit(
                            "worker_failed", model=mid,
                            promoted=len(actions),
                        )
                        self._promoted.add(mid)
                        outcomes[mid] = "Failed"
                    self.cleanup_after_worker(mid)

            if self._all_terminal(jobs):
                return outcomes

            time.sleep(self.aggregator_interval_seconds)

    # ---- cross-study resume --------------------------------------------
    def _patch_configmap_resume_from(self, resume_from: str) -> None:
        """Patch `pricing-reversal-parallel-config`'s `experiment.yaml`
        entry to add ``experiment.resume_from: <path>``.

        Workers mount this ConfigMap read-only under /config, so they
        see the patched value at study_start. Kubelet refreshes the
        mounted file within ~60s of the ConfigMap update — for freshly
        spawned worker Pods this is fine because their Pod-startup
        (image pull + Python init) takes longer than the refresh.

        Uses read-modify-write on `.data["experiment.yaml"]`. Preserves
        every other field in experiment.yaml verbatim; only the
        `resume_from` key gets added under `experiment:` (or updated
        if it was already there, e.g. from a prior aborted run of the
        same uber Pod).
        """
        import yaml as pyyaml
        cm_name = "pricing-reversal-parallel-config"
        try:
            cm = self.core.read_namespaced_config_map(cm_name, self.namespace)
        except ApiException as exc:
            self.progress.emit(
                "resume_patch_failed",
                resume_from=resume_from,
                error=f"read ConfigMap {cm_name}: {exc.status} {exc.reason}",
            )
            raise

        data = dict(cm.data or {})
        exp_yaml = data.get("experiment.yaml", "")
        doc = pyyaml.safe_load(exp_yaml) or {}
        exp_block = doc.get("experiment") or {}
        prev = exp_block.get("resume_from")
        exp_block["resume_from"] = resume_from
        doc["experiment"] = exp_block
        data["experiment.yaml"] = pyyaml.dump(doc, sort_keys=False,
                                              default_flow_style=False)

        cm.data = data
        self.core.replace_namespaced_config_map(cm_name, self.namespace, cm)

        self.progress.emit(
            "resume_configmap_patched",
            configmap=cm_name,
            resume_from=resume_from,
            previous=prev,
        )

    # ---- entry ----------------------------------------------------------
    def run(self) -> int:
        """Blocking entry point. Returns 0 on clean completion, non-zero on
        study_complete_with_residuals."""
        self.progress.start_heartbeat(self.aggregator_interval_seconds)
        try:
            self.progress.emit(
                "study_start",
                namespace=self.namespace,
                models=self._model_ids(),
                dry_run=self.dry_run,
                study_id=self.study_id,
            )
            self.progress.update_state(
                phase="starting",
                models=self._model_ids(),
                error=None,
                study_id=self.study_id,
            )

            # Cross-study resume: if --resume-from was passed, patch the
            # experiment.yaml ConfigMap key BEFORE any worker Job is
            # created. Each worker's Pod mounts the same ConfigMap and
            # reads experiment.resume_from at study_start; without the
            # patch, workers would come up with a null resume_from and
            # do fresh runs.
            if self.resume_from and not self.dry_run:
                self._patch_configmap_resume_from(self.resume_from)

            # Reconcile — attach to existing worker Jobs; create the missing ones.
            known = self.reconcile()
            for model in self.cfg.models.models:
                if known.get(model.id) is None:
                    self.create_worker_job(model)

            if self.dry_run:
                self.progress.emit("study_complete", dry_run=True)
                return 0

            # Watch + aggregate until every worker is terminal.
            outcomes = self._watch_loop()

            # Verification pass. One retry if residuals appear.
            report = self.verify_all_gpus_freed()
            if not report["ok"]:
                self.progress.emit("verification_retry", reason="residuals found")
                self.final_delete_pass()
                report = self.verify_all_gpus_freed()

            # Finalize the top-level manifest.
            self.aggregator.finalize_manifest()

            if report["ok"]:
                self.progress.emit("study_complete", outcomes=outcomes)
                self.progress.update_state(phase="complete")
                return 0
            else:
                self.progress.emit(
                    "study_complete_with_residuals",
                    outcomes=outcomes,
                    residuals=report,
                )
                self.progress.update_state(phase="complete_with_residuals")
                return 2
        except Exception as exc:  # noqa: BLE001
            self.progress.emit("study_error", error=str(exc)[:400])
            self.progress.update_state(phase="error", error=str(exc)[:400])
            raise
        finally:
            self.progress.stop_heartbeat()
