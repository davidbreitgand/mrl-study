"""Tests for the worker-mode additions to Orchestrator.

Verified without a Kubernetes API or a live vLLM: we exercise the
model-selection filter and the results-subdir scoping, both of which
happen at __init__ time (no async, no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pricing_reversal_harness.orchestrator import Orchestrator


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = str(REPO_ROOT / "config")
VLLM_TEMPLATE = str(REPO_ROOT / "stack" / "modelserver" / "vllm-deployment.yaml.tmpl")


def _make_orch(**overrides):
    kwargs = dict(
        config_dir=CONFIG_DIR,
        results_dir=str(overrides.pop("results_dir")),
        namespace="test-ns",
        hf_token_secret="x",
        vllm_template_path=VLLM_TEMPLATE,
        router_service="pricing-reversal-router-epp",
        router_port=80,
        dry_run=True,      # skips K8sClient construction, safe for unit tests
    )
    kwargs.update(overrides)
    return Orchestrator(**kwargs)


def test_select_models_default_returns_all(tmp_path: Path):
    orch = _make_orch(results_dir=tmp_path)
    ids = [m.id for m in orch._select_models()]
    assert len(ids) >= 1
    assert set(ids) == {m.id for m in orch.cfg.models.models}


def test_select_models_only_model_filters_to_one(tmp_path: Path):
    orch = _make_orch(results_dir=tmp_path, only_model="qwen3-8b")
    selected = orch._select_models()
    assert [m.id for m in selected] == ["qwen3-8b"]


def test_select_models_only_model_unknown_raises(tmp_path: Path):
    orch = _make_orch(results_dir=tmp_path, only_model="does-not-exist")
    with pytest.raises(KeyError) as ei:
        orch._select_models()
    assert "does-not-exist" in str(ei.value)


def test_results_subdir_scopes_output_paths(tmp_path: Path):
    top = tmp_path
    orch = _make_orch(
        results_dir=top,
        only_model="qwen3-8b",
        results_subdir="workers/qwen3-8b",
    )
    # Scoped tree is created under the subdir.
    assert (top / "workers" / "qwen3-8b" / "consolidated").is_dir()
    assert (top / "workers" / "qwen3-8b" / "repeated_trial").is_dir()
    # Dataset cache stays at the top level so parallel workers share it.
    assert (top / "dataset_cache").is_dir()
    # Sanity: without a subdir the paths land at the top.
    assert (top / "consolidated").is_dir() is False  # subdir mode: no top-level consolidated
    # But progress.jsonl for THIS worker lives under the subdir:
    assert orch.results_dir == top / "workers" / "qwen3-8b"


def test_results_subdir_absent_uses_top_level(tmp_path: Path):
    orch = _make_orch(results_dir=tmp_path)
    assert orch.results_dir == tmp_path
    assert (tmp_path / "consolidated").is_dir()
    assert (tmp_path / "repeated_trial").is_dir()
    assert (tmp_path / "dataset_cache").is_dir()


def test_install_router_requires_values_template(tmp_path: Path):
    # Present template — orchestrator constructs the Template object.
    tpl = tmp_path / "router-values.yaml.tmpl"
    tpl.write_text("router:\n  modelServers:\n    matchLabels:\n"
                   "      pricing-reversal.study/model: {{ model_id }}\n",
                   encoding="utf-8")
    orch = _make_orch(
        results_dir=tmp_path,
        only_model="qwen3-8b",
        install_router=True,
        router_values_template_path=str(tpl),
    )
    # Template loaded, ready to render.
    rendered = orch._router_values_template.render(
        model_id="qwen3-8b", namespace="test-ns"
    )
    assert "qwen3-8b" in rendered


def test_release_name_matches_convention(tmp_path: Path):
    orch = _make_orch(results_dir=tmp_path)
    m = orch.cfg.models.models[0]
    assert orch._release_name(m) == f"pricing-reversal-router-{m.id}"


def test_chart_epp_object_name_truncates_at_40():
    """The chart's llm-d-router.name helper truncates the Helm release name
    to 40 chars before appending `-epp`, so long release names produce
    Deployment/Service objects the orchestrator would miss otherwise.
    """
    from pricing_reversal_harness.orchestrator import chart_epp_object_name

    # Short releases pass through unchanged.
    assert chart_epp_object_name("pricing-reversal-router-qwen3-8b") == \
        "pricing-reversal-router-qwen3-8b-epp"

    # Long releases get truncated at 40 chars before `-epp` is appended.
    # `pricing-reversal-router-deepseek-r1-distill-14b` is 47 chars.
    got = chart_epp_object_name("pricing-reversal-router-deepseek-r1-distill-14b")
    assert got == "pricing-reversal-router-deepseek-r1-dist-epp"
    assert len(got) == 44

    # 40-char release: still gets truncated (well, no-op) + `-epp`.
    exactly40 = "a" * 40
    assert chart_epp_object_name(exactly40) == exactly40 + "-epp"
