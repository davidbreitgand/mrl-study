"""Two-tier readiness-budget behaviour in K8s.wait_deployment_available.

We drive the poll loop with a Fake apps/core client whose responses are
programmable per call so we can assert:
  * Ready is returned as soon as replica counts satisfy .spec.replicas
  * Pending-only ticks consume pending_budget; ready_replicas=0 while
    Pods are past Pending consume startup_budget
  * Either budget expiring returns a WaitOutcome with the matching
    reason so the caller can emit distinct progress phases
  * pending_budget_seconds=None disables the Pending check entirely
    (router pods that don't need a GPU)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from pricing_reversal_harness.k8s import K8sClient


class FakeAppsV1:
    """Stand-in for AppsV1Api.read_namespaced_deployment_status. `script`
    is a list of Deployment status objects returned in order; the last
    entry sticks after the list is exhausted."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def read_namespaced_deployment_status(self, name, ns):
        self.calls += 1
        idx = min(self.calls - 1, len(self.script) - 1)
        return self.script[idx]


class FakeCoreV1:
    """Stand-in for CoreV1Api.list_namespaced_pod. `phase_script` is a
    list of pod-phase lists (one list per poll iteration)."""

    def __init__(self, phase_script):
        self.phase_script = list(phase_script)
        self.calls = 0

    def list_namespaced_pod(self, ns, label_selector=""):
        self.calls += 1
        idx = min(self.calls - 1, len(self.phase_script) - 1)
        phases = self.phase_script[idx]
        items = [SimpleNamespace(status=SimpleNamespace(phase=p)) for p in phases]
        return SimpleNamespace(items=items)


def _dep_status(*, replicas=1, ready=0, available=0):
    return SimpleNamespace(
        spec=SimpleNamespace(
            replicas=replicas,
            selector=SimpleNamespace(match_labels={"app": "vllm"}),
        ),
        status=SimpleNamespace(ready_replicas=ready, available_replicas=available),
    )


def _make_k8s(apps, core):
    """Build a K8s that uses the fakes without touching kube config."""
    k = K8sClient.__new__(K8sClient)
    k.namespace = "test-ns"
    k.apps = apps
    k.core = core
    k.batch = None
    return k


def test_returns_ok_when_deployment_reports_ready() -> None:
    apps = FakeAppsV1([_dep_status(ready=1, available=1)])
    core = FakeCoreV1([["Running"]])
    k = _make_k8s(apps, core)
    with patch("time.sleep"):
        out = k.wait_deployment_available(
            "vllm", pending_budget_seconds=60, startup_budget_seconds=60,
        )
    assert out.ok is True
    assert out.reason == "ready"


def test_pending_timeout_when_pod_stays_pending() -> None:
    """time.time() advances by 30 s per call so 5 poll iterations
    exhaust a 60 s pending budget."""
    apps = FakeAppsV1([_dep_status(ready=0)])
    core = FakeCoreV1([["Pending"]])
    k = _make_k8s(apps, core)
    fake_time = iter(range(0, 3600, 30))

    def _t():
        return next(fake_time)
    with patch("time.sleep"), patch("time.time", _t):
        out = k.wait_deployment_available(
            "vllm", pending_budget_seconds=60, startup_budget_seconds=3600,
        )
    assert out.ok is False
    assert out.reason == "pending_timeout"
    assert out.elapsed_startup == 0


def test_startup_timeout_when_running_but_not_ready() -> None:
    """Pod is Running (past Pending) but Deployment never reaches
    ready_replicas — budget attribution is startup, not pending."""
    apps = FakeAppsV1([_dep_status(ready=0)])
    core = FakeCoreV1([["Running"]])
    k = _make_k8s(apps, core)
    fake_time = iter(range(0, 3600, 30))

    def _t():
        return next(fake_time)
    with patch("time.sleep"), patch("time.time", _t):
        out = k.wait_deployment_available(
            "vllm", pending_budget_seconds=3600, startup_budget_seconds=60,
        )
    assert out.ok is False
    assert out.reason == "startup_timeout"
    assert out.elapsed_pending == 0


def test_pending_budget_none_disables_pending_check() -> None:
    """Router path: no GPU needed, so we don't want a pending timeout.
    Verify that with pending_budget=None, staying Pending past what
    WOULD be a pending timeout does NOT abort the wait; only the
    startup budget can enforce a ceiling. To make the test terminate
    deterministically we transition the Pod to Running after a few
    ticks so startup_budget then enforces the eventual timeout."""
    apps = FakeAppsV1([
        _dep_status(ready=0),
        _dep_status(ready=0),
        _dep_status(ready=0),   # still not ready
        _dep_status(ready=0),
        _dep_status(ready=0),
    ])
    # First 3 polls: Pending. From tick 4 onward: Running-but-not-Ready.
    # With pending_budget=60 s and 30 s/tick, a normal run would have
    # aborted at tick 2 with pending_timeout. Passing None skips that.
    core = FakeCoreV1([["Pending"], ["Pending"], ["Pending"],
                       ["Running"], ["Running"]])
    k = _make_k8s(apps, core)
    fake_time = iter([0, 30, 60, 90, 120, 150, 180])

    def _t():
        try:
            return next(fake_time)
        except StopIteration:
            return 999999  # keep loop bounded; will trip startup_timeout
    with patch("time.sleep"), patch("time.time", _t):
        out = k.wait_deployment_available(
            "vllm", pending_budget_seconds=None, startup_budget_seconds=60,
        )
    assert out.ok is False
    # Never pending_timeout when pending_budget is None.
    assert out.reason == "startup_timeout"
    # And we lasted past the 60 s that pending_budget=60 would have
    # tripped at — proves the disable took effect.
    assert out.elapsed_pending >= 60


def test_ready_after_transient_pending() -> None:
    """Realistic warmup: Pending for a while, then Running, then Ready."""
    apps = FakeAppsV1([
        _dep_status(ready=0),  # first two polls: not ready
        _dep_status(ready=0),
        _dep_status(ready=1, available=1),  # third poll: ready
    ])
    core = FakeCoreV1([["Pending"], ["Running"], ["Running"]])
    k = _make_k8s(apps, core)
    with patch("time.sleep"):
        out = k.wait_deployment_available(
            "vllm", pending_budget_seconds=3600, startup_budget_seconds=3600,
        )
    assert out.ok is True
    assert out.reason == "ready"
