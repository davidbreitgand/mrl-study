"""Thin wrapper over the in-cluster Kubernetes API for the orchestrator.

Uses `kubernetes` client with in-cluster ServiceAccount credentials mounted
by Kubernetes. All operations are namespace-scoped; the harness's Role does
not grant cluster-scoped verbs.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import yaml
from kubernetes import client as k8s_client
from kubernetes import config as k8s_config
from kubernetes.client.exceptions import ApiException


log = logging.getLogger(__name__)


class K8sClient:
    def __init__(self, namespace: str) -> None:
        try:
            k8s_config.load_incluster_config()
        except k8s_config.ConfigException:
            k8s_config.load_kube_config()  # dev-mode fallback
        self.namespace = namespace
        self.core = k8s_client.CoreV1Api()
        self.apps = k8s_client.AppsV1Api()
        self.batch = k8s_client.BatchV1Api()

    # ---- apply ----------------------------------------------------------
    def apply_manifest(self, yaml_text: str) -> None:
        """Server-side apply each document in a multi-doc YAML stream.

        Simple replay: try create; on 409 conflict, patch with strategic-merge.
        Adequate for our small set of Deployment/ServiceAccount/PVC objects.
        """

        for doc in yaml.safe_load_all(yaml_text):
            if not doc:
                continue
            self._apply_one(doc)

    def _apply_one(self, doc: dict[str, Any]) -> None:
        kind = doc.get("kind")
        name = (doc.get("metadata") or {}).get("name")
        ns = (doc.get("metadata") or {}).get("namespace") or self.namespace
        if not kind or not name:
            raise ValueError(f"manifest doc missing kind/name: {doc!r}")

        try:
            if kind == "Deployment":
                try:
                    self.apps.create_namespaced_deployment(ns, doc)
                except ApiException as e:
                    if e.status == 409:
                        self.apps.patch_namespaced_deployment(name, ns, doc)
                    else:
                        raise
            elif kind == "Service":
                try:
                    self.core.create_namespaced_service(ns, doc)
                except ApiException as e:
                    if e.status == 409:
                        self.core.patch_namespaced_service(name, ns, doc)
                    else:
                        raise
            elif kind == "PersistentVolumeClaim":
                try:
                    self.core.create_namespaced_persistent_volume_claim(ns, doc)
                except ApiException as e:
                    if e.status == 409:
                        log.info("pvc %s already exists; leaving as-is", name)
                    else:
                        raise
            elif kind == "ServiceAccount":
                try:
                    self.core.create_namespaced_service_account(ns, doc)
                except ApiException as e:
                    if e.status == 409:
                        log.info("serviceaccount %s already exists", name)
                    else:
                        raise
            elif kind == "ConfigMap":
                try:
                    self.core.create_namespaced_config_map(ns, doc)
                except ApiException as e:
                    if e.status == 409:
                        self.core.patch_namespaced_config_map(name, ns, doc)
                    else:
                        raise
            elif kind == "Job":
                try:
                    self.batch.create_namespaced_job(ns, doc)
                except ApiException as e:
                    if e.status == 409:
                        # Jobs are immutable on most fields; keep the
                        # existing Job and let the caller notice via
                        # its own reconcile logic.
                        log.info("job %s already exists; leaving as-is", name)
                    else:
                        raise
            else:
                raise ValueError(f"unsupported kind for apply: {kind}")
        except ApiException as e:
            log.error("apply %s/%s failed: %s", kind, name, e)
            raise

    # ---- delete ---------------------------------------------------------
    def delete_deployment(self, name: str) -> None:
        try:
            self.apps.delete_namespaced_deployment(name, self.namespace)
        except ApiException as e:
            if e.status != 404:
                raise

    def delete_service_account(self, name: str) -> None:
        try:
            self.core.delete_namespaced_service_account(name, self.namespace)
        except ApiException as e:
            if e.status != 404:
                raise

    def delete_pvc(self, name: str) -> None:
        try:
            self.core.delete_namespaced_persistent_volume_claim(name, self.namespace)
        except ApiException as e:
            if e.status != 404:
                raise

    # ---- readiness ------------------------------------------------------
    class WaitOutcome:
        """Return of `wait_deployment_available`. `.ok` mirrors the old bool
        return; `.reason` distinguishes the two failure modes for the
        caller to emit distinct progress phases."""
        __slots__ = ("ok", "reason", "elapsed_pending", "elapsed_startup")

        def __init__(self, ok: bool, reason: str,
                     elapsed_pending: float, elapsed_startup: float) -> None:
            self.ok = ok
            self.reason = reason  # "ready" | "pending_timeout" | "startup_timeout"
            self.elapsed_pending = elapsed_pending
            self.elapsed_startup = elapsed_startup

        def __bool__(self) -> bool:
            return self.ok

    def wait_deployment_available(
        self,
        name: str,
        *,
        pending_budget_seconds: int | None,
        startup_budget_seconds: int,
    ) -> "K8sClient.WaitOutcome":
        """Wait for a Deployment to reach ready replicas, with SEPARATE
        wall-clock budgets for Pod-Pending time and container-startup
        time. On this shared GPU cluster the former is often minutes to
        hours (waiting for a GPU node to free up) and the latter is
        ~5-10 min (vLLM weight loading + JIT). A single fixed budget
        conflates the two, so long Pending waits eat into what should be
        startup budget and the harness gives up on a model that would
        otherwise come up cleanly.

        Two independent absolute-elapsed budgets are enforced:

          * `pending_budget_seconds`: max wall-clock spent while ANY
            managed Pod is in phase Pending (or the ReplicaSet hasn't
            created a Pod yet). Reflects "the cluster is oversubscribed;
            give up rather than blocking downstream work forever".
            Pass `None` to skip this check entirely — appropriate for
            router/EPP deployments that don't need a GPU and schedule
            immediately.
          * `startup_budget_seconds`: max wall-clock spent while EVERY
            managed Pod is past Pending (Running/etc.) but the
            Deployment doesn't yet report enough Ready replicas. This
            is the "vLLM startup is genuinely slow" knob.

        Both budgets are accumulated across the entire wait — a Pod
        that flips Pending → Running → Pending (e.g. preempted mid-
        warmup) consumes both budgets in parallel; either expiring
        aborts the wait. Total wall-clock worst case: `pending +
        startup`.

        Returns a `WaitOutcome`. `.ok=True` iff the Deployment reached
        its replica count. `.reason` is one of `ready`,
        `pending_timeout`, `startup_timeout` for the caller to route
        into distinct progress phases.
        """
        pending_deadline_disabled = pending_budget_seconds is None
        elapsed_pending = 0.0
        elapsed_startup = 0.0
        # Discover the Deployment's pod selector once — refreshing every
        # poll would double the API load for no gain (a Deployment's
        # selector is immutable).
        selector: str | None = None
        last_tick = time.time()
        while True:
            try:
                dep = self.apps.read_namespaced_deployment_status(name, self.namespace)
            except ApiException as e:
                if e.status == 404:
                    now = time.time()
                    dt = now - last_tick
                    last_tick = now
                    # Missing Deployment counts as "still Pending" — the
                    # ReplicaSet may not have been observed yet.
                    elapsed_pending += dt
                    if (not pending_deadline_disabled
                            and elapsed_pending >= pending_budget_seconds):
                        return K8sClient.WaitOutcome(
                            False, "pending_timeout",
                            elapsed_pending, elapsed_startup,
                        )
                    time.sleep(2.0)
                    continue
                raise

            status = dep.status
            desired = dep.spec.replicas or 1
            ready = int(status.ready_replicas or 0)
            available = int(status.available_replicas or 0)
            if ready >= desired and available >= desired:
                return K8sClient.WaitOutcome(
                    True, "ready", elapsed_pending, elapsed_startup,
                )

            if selector is None:
                sel_match = (dep.spec.selector.match_labels or {}) if dep.spec.selector else {}
                selector = ",".join(f"{k}={v}" for k, v in sel_match.items())

            # Any Pod still in Pending → this tick counts against
            # pending_budget. Otherwise it counts against startup_budget.
            # An empty pod list is treated as "still pending" — the
            # ReplicaSet exists but hasn't materialised a Pod yet.
            pods = self.list_pods(label_selector=selector) if selector else []
            any_pending = (not pods) or any(
                (getattr(p.status, "phase", None) or "Pending") == "Pending"
                for p in pods
            )

            now = time.time()
            dt = now - last_tick
            last_tick = now
            if any_pending:
                elapsed_pending += dt
            else:
                elapsed_startup += dt

            if (not pending_deadline_disabled
                    and elapsed_pending >= pending_budget_seconds):
                return K8sClient.WaitOutcome(
                    False, "pending_timeout",
                    elapsed_pending, elapsed_startup,
                )
            if elapsed_startup >= startup_budget_seconds:
                return K8sClient.WaitOutcome(
                    False, "startup_timeout",
                    elapsed_pending, elapsed_startup,
                )
            time.sleep(5.0)

    # ---- deployment queries ---------------------------------------------
    def deployment_exists(self, name: str) -> bool:
        try:
            self.apps.read_namespaced_deployment(name, self.namespace)
            return True
        except ApiException as e:
            if e.status == 404:
                return False
            raise

    # ---- job queries + management ---------------------------------------
    def list_jobs(self, label_selector: str = "") -> list[Any]:
        resp = self.batch.list_namespaced_job(
            self.namespace, label_selector=label_selector,
        )
        return list(resp.items)

    def read_job(self, name: str) -> Any:
        return self.batch.read_namespaced_job(name, self.namespace)

    def delete_job(self, name: str, propagation: str = "Foreground") -> None:
        try:
            self.batch.delete_namespaced_job(
                name, self.namespace,
                propagation_policy=propagation,
            )
        except ApiException as e:
            if e.status != 404:
                raise

    # ---- pod queries ----------------------------------------------------
    def list_pods(self, label_selector: str = "") -> list[Any]:
        resp = self.core.list_namespaced_pod(
            self.namespace, label_selector=label_selector,
        )
        return list(resp.items)
