"""UberController tests — reconcile, terminal detection, verification.

We stub the K8sClient (dry-run) and pass the controller a fake instead
of the real one; then swap it out at test time to poke behaviors that
would otherwise require a live cluster.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest

from pricing_reversal_harness.uber.controller import (
    UberController, _is_job_complete, _is_job_failed, _job_summary,
    worker_job_name, vllm_deployment_name, router_release_name,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = str(REPO_ROOT / "config")
WORKER_TMPL = REPO_ROOT / "stack" / "harness" / "harness-job.yaml.tmpl"
# We just need SOME valid Jinja template for tests; the sequential one works.


def _mk_job(model_id: str, *, complete=False, failed=False, active=0,
            succeeded=0, failed_count=0) -> Any:
    conditions = []
    if complete:
        conditions.append(types.SimpleNamespace(type="Complete", status="True"))
    if failed:
        conditions.append(types.SimpleNamespace(type="Failed", status="True"))
    return types.SimpleNamespace(
        metadata=types.SimpleNamespace(
            name=worker_job_name(model_id),
            namespace="test-ns",
            labels={
                "llm-d.ai/guide": "pricing-reversal",
                "pricing-reversal.study/component": "worker",
                "pricing-reversal.study/model": model_id,
            },
        ),
        status=types.SimpleNamespace(
            active=active, succeeded=succeeded, failed=failed_count,
            conditions=conditions,
        ),
    )


def test_job_status_helpers():
    assert _is_job_complete(_mk_job("qwen3-8b", complete=True))
    assert not _is_job_failed(_mk_job("qwen3-8b", complete=True))
    assert _is_job_failed(_mk_job("qwen3-8b", failed=True))
    assert not _is_job_complete(_mk_job("qwen3-8b", failed=True))
    # Succeeded field as a fallback (no explicit Complete condition).
    j = types.SimpleNamespace(
        metadata=types.SimpleNamespace(name="j", labels={}),
        status=types.SimpleNamespace(active=0, succeeded=1, failed=0, conditions=[]),
    )
    assert _is_job_complete(j)


def test_job_summary_shapes():
    s = _job_summary(_mk_job("qwen3-8b", complete=True, succeeded=1))
    assert s["job_status"] == "Complete"
    assert s["succeeded"] == 1
    s = _job_summary(_mk_job("qwen3-8b", failed=True, failed_count=6))
    assert s["job_status"] == "Failed"
    assert s["failed"] == 6


def test_naming_conventions():
    assert worker_job_name("qwen3-8b") == "pricing-reversal-worker-qwen3-8b"
    assert vllm_deployment_name("qwen3-8b") == "pricing-reversal-vllm-qwen3-8b"
    assert router_release_name("qwen3-8b") == "pricing-reversal-router-qwen3-8b"


def _make_uber(tmp_path: Path):
    # Minimal Jinja template covering every var the controller passes in.
    # In production this is `stack/uber/worker-job.yaml.tmpl`.
    tpl = tmp_path / "worker.tmpl"
    tpl.write_text("""apiVersion: batch/v1
kind: Job
metadata:
  name: {{ job_name }}
  namespace: {{ namespace }}
  labels:
    pricing-reversal.study/model: {{ model_id }}
spec:
  template:
    spec:
      containers:
        - name: worker
          image: {{ worker_image }}
          env:
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef: {name: {{ hf_token_secret }}, key: HF_TOKEN}
          args:
            - --only-model={{ model_id }}
            - --router-values-template={{ router_values_template }}
            - --router-chart-version={{ router_chart_version }}
""", encoding="utf-8")
    return UberController(
        config_dir=CONFIG_DIR,
        results_dir=str(tmp_path),
        namespace="test-ns",
        hf_token_secret="x",
        worker_image="ghcr.io/example/harness:latest",
        worker_template_path=str(tpl),
        router_values_template_path="/config/router-values.yaml.tmpl",
        aggregator_interval_seconds=0.01,  # fast for unit tests
        dry_run=True,
    )


def test_reconcile_dry_run_returns_all_pending(tmp_path: Path):
    uber = _make_uber(tmp_path)
    known = uber.reconcile()
    ids_from_config = {m.id for m in uber.cfg.models.models}
    assert set(known.keys()) == ids_from_config
    assert all(v is None for v in known.values())


class FakeK8sClient:
    """Minimal stand-in that satisfies the calls the controller makes.

    Test authors mutate `.jobs`, `.deployments_present`, `.pods` between
    controller invocations to script cluster state.
    """

    def __init__(self, namespace: str, initial_jobs: list | None = None,
                 deployments_present: set[str] | None = None,
                 pods: list | None = None) -> None:
        self.namespace = namespace
        self.jobs = list(initial_jobs or [])
        self.deployments_present = set(deployments_present or ())
        self.pods = list(pods or [])
        self.deletes: list[tuple[str, str]] = []  # (kind, name)
        self.applied: list[str] = []              # yaml texts

    def list_jobs(self, label_selector: str = "") -> list:
        return self.jobs

    def apply_manifest(self, yaml_text: str) -> None:
        self.applied.append(yaml_text)

    def delete_deployment(self, name: str) -> None:
        self.deletes.append(("Deployment", name))
        self.deployments_present.discard(name)

    def delete_service_account(self, name: str) -> None:
        self.deletes.append(("ServiceAccount", name))

    def deployment_exists(self, name: str) -> bool:
        return name in self.deployments_present

    def list_pods(self, label_selector: str = "") -> list:
        return self.pods


def test_reconcile_labels_existing_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    uber = _make_uber(tmp_path)
    fake = FakeK8sClient("test-ns", initial_jobs=[
        _mk_job("qwen3-8b", complete=True, succeeded=1),
        _mk_job("granite-3-3-8b-thinking", active=1),
        # deepseek not present — should be flagged for creation.
    ])
    uber.k8s = fake
    uber.dry_run = False  # exercise the real code path

    known = uber.reconcile()
    assert _is_job_complete(known["qwen3-8b"])
    assert known["granite-3-3-8b-thinking"] is not None
    assert not _is_job_complete(known["granite-3-3-8b-thinking"])
    assert known["deepseek-r1-distill-14b"] is None


def test_reconcile_flags_orphan_jobs(tmp_path: Path):
    uber = _make_uber(tmp_path)
    # A stale worker job for a model no longer in models.yaml.
    orphan = _mk_job("older-model-not-in-config")
    fake = FakeK8sClient("test-ns", initial_jobs=[orphan])
    uber.k8s = fake
    uber.dry_run = False

    uber.reconcile()

    # Progress emitted an orphan event. Read from uber.uber_dir so this
    # test doesn't care whether the study id lives in the path — the
    # controller's own API is the contract.
    events = [json.loads(l) for l in
              (uber.uber_dir / "progress.jsonl").read_text("utf-8").splitlines()]
    assert any(e["phase"] == "orphan_job_ignored" for e in events)


def test_verify_all_gpus_freed_green(tmp_path: Path):
    uber = _make_uber(tmp_path)
    fake = FakeK8sClient("test-ns", deployments_present=set(), pods=[])
    uber.k8s = fake
    uber.dry_run = False

    # Monkeypatch subprocess.run so helm status always says "not found" (rc != 0).
    import pricing_reversal_harness.uber.controller as ctl_mod
    class _Rc:
        returncode = 1
    ctl_mod.subprocess.run = lambda *a, **kw: _Rc()

    report = uber.verify_all_gpus_freed()
    assert report["ok"] is True
    assert report["residual_gpus"] == 0
    assert report["unclean"] == []


def test_verify_all_gpus_freed_detects_leaked_deployment(tmp_path: Path):
    uber = _make_uber(tmp_path)
    fake = FakeK8sClient(
        "test-ns",
        deployments_present={"pricing-reversal-vllm-qwen3-8b"},
        pods=[],
    )
    uber.k8s = fake
    uber.dry_run = False

    import pricing_reversal_harness.uber.controller as ctl_mod
    class _Rc:
        returncode = 1
    ctl_mod.subprocess.run = lambda *a, **kw: _Rc()

    report = uber.verify_all_gpus_freed()
    assert report["ok"] is False
    kinds = {u["kind"] for u in report["unclean"]}
    assert "Deployment" in kinds


def test_verify_all_gpus_freed_counts_residual_gpus(tmp_path: Path):
    uber = _make_uber(tmp_path)
    # A pod still Running, holding 2 GPUs.
    pod = types.SimpleNamespace(
        metadata=types.SimpleNamespace(name="leaked-vllm", namespace="test-ns"),
        status=types.SimpleNamespace(phase="Running"),
        spec=types.SimpleNamespace(containers=[
            types.SimpleNamespace(resources=types.SimpleNamespace(
                requests={"nvidia.com/gpu": "2"},
            )),
        ]),
    )
    fake = FakeK8sClient("test-ns", pods=[pod])
    uber.k8s = fake
    uber.dry_run = False

    import pricing_reversal_harness.uber.controller as ctl_mod
    class _Rc:
        returncode = 1
    ctl_mod.subprocess.run = lambda *a, **kw: _Rc()

    report = uber.verify_all_gpus_freed()
    assert report["ok"] is False
    assert report["residual_gpus"] == 2


def test_all_terminal_helper(tmp_path: Path):
    uber = _make_uber(tmp_path)
    j_ok = _mk_job("qwen3-8b", complete=True)
    j_bad = _mk_job("granite", failed=True)
    j_active = _mk_job("deepseek", active=1)
    assert uber._all_terminal({"a": j_ok, "b": j_bad}) is True
    assert uber._all_terminal({"a": j_ok, "b": j_active}) is False
    assert uber._all_terminal({"a": j_ok, "b": None}) is False


def test_dry_run_run_emits_study_start_and_complete(tmp_path: Path):
    uber = _make_uber(tmp_path)
    rc = uber.run()
    assert rc == 0
    events = [json.loads(l) for l in
              (uber.uber_dir / "progress.jsonl").read_text("utf-8").splitlines()]
    phases = [e["phase"] for e in events]
    assert "study_start" in phases
    assert phases[-1] == "study_complete"
