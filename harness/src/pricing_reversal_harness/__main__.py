"""Entry point for the in-cluster orchestrator.

Sequential mode (all models, one Pod):
    python -m pricing_reversal_harness \
        --config-dir=/config --results-dir=/results \
        --namespace=<your-namespace> --hf-token-secret=llm-d-hf-token \
        --vllm-template=/config/vllm-deployment.yaml.tmpl \
        --router-service=pricing-reversal-router-epp --router-port=80 \
        --resume

Worker mode (one model, one Pod — invoked by the uber-harness):
    python -m pricing_reversal_harness \
        --config-dir=/config --results-dir=/results \
        --namespace=<your-namespace> --hf-token-secret=llm-d-hf-token \
        --vllm-template=/config/vllm-deployment.yaml.tmpl \
        --only-model=qwen3-8b \
        --results-subdir=workers/qwen3-8b \
        --install-router --router-values-template=/config/router-values.yaml.tmpl \
        --resume
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys

from .orchestrator import Orchestrator
from .progress import Progress


def _configure_logging() -> None:
    # Route stdlib `logging` to STDERR so it doesn't interleave with the
    # structured JSON progress events on STDOUT. `oc logs` still shows both
    # (kubelet merges them), but our tail-progress.sh filter can rely on
    # every stdout line being a JSON object.
    level = os.environ.get("HARNESS_LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="pricing-reversal-harness")
    ap.add_argument("--config-dir", default="/config")
    ap.add_argument("--results-dir", default="/results")
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--hf-token-secret", default="llm-d-hf-token")
    ap.add_argument("--vllm-template", default="/config/vllm-deployment.yaml.tmpl")
    ap.add_argument("--router-service", default="pricing-reversal-router-epp")
    ap.add_argument("--router-port", type=int, default=80)
    ap.add_argument("--resume", action="store_true",
                    help="Resume from existing journals and state.json (default off).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan without applying any Kubernetes objects.")
    # Worker mode: restrict the orchestrator to a single model + a scoped
    # subdirectory of the results tree. Both defaults are None so
    # sequential-mode invocations are backwards-compatible.
    ap.add_argument("--only-model", default=None,
                    help="Restrict the orchestrator to a single model id from models.yaml.")
    ap.add_argument("--results-subdir", default=None,
                    help="Write all outputs under results-dir/<subdir> (e.g. 'workers/qwen3-8b').")
    # Per-worker router install: the worker owns its own Helm release so
    # every model has an isolated router endpoint.
    ap.add_argument("--install-router", action="store_true",
                    help="helm install/uninstall a per-worker router around this study.")
    ap.add_argument("--router-values-template", default="/config/router-values.yaml.tmpl",
                    help="Jinja2 template for per-worker router Helm values.")
    ap.add_argument("--router-chart",
                    default="oci://ghcr.io/llm-d/charts/llm-d-router-standalone",
                    help="Router Helm chart OCI reference.")
    ap.add_argument("--router-chart-version", default="v0.9.0",
                    help="Router Helm chart version.")
    # Cross-cluster CostGuard attribution: mounted from the same
    # ConfigMap the Hub EPP reads. Both keys and address:port entries
    # get indexed so lookups work regardless of what the response
    # header carries. See analysis/multi-cluster-epp.md §4.
    ap.add_argument("--clusters-yaml", default="/etc/harness/clusters.yaml",
                    help="Path to clusters.yaml for cost attribution.")
    ap.add_argument("--skip-model-deploy", action="store_true",
                    help="Skip the harness's local vLLM Deployment + wait + "
                         "teardown. Use when the vLLM pods that serve the "
                         "model already exist elsewhere (e.g. spoke "
                         "namespaces in a cluster-scoped multicluster "
                         "study). The router readiness probe still runs.")
    return ap.parse_args(argv)


def _install_abort_handler(progress: Progress) -> None:
    """SIGTERM/SIGINT handler: emit a terminal `study_aborted` progress
    line so PVC-side tooling (study-progress.sh, aggregator) can tell
    "Pod was killed before completing" from "Pod died mid-write with no
    trace". Kubernetes sends SIGTERM ~30 s before SIGKILL on eviction,
    preemption, and JobController deletion, which is plenty of time to
    flush one JSON line and exit.

    The handler is one-shot: after emitting, it re-raises the default
    action so a second signal (or the follow-up SIGKILL) tears the
    process down as usual. Progress.emit is fsync'd + stdout-flushed so
    the line is durable even if the kernel kills us next tick.
    """

    def _handle(signum: int, _frame: object) -> None:
        try:
            progress.emit(
                "study_aborted",
                signal=signal.Signals(signum).name,
            )
            progress.update_state(phase="aborted", signal=signal.Signals(signum).name)
        except Exception:  # noqa: BLE001
            # A failed emit must never mask the shutdown — we're
            # already dying. Log to stderr and continue.
            logging.getLogger(__name__).exception(
                "study_aborted emit failed for signal %s", signum,
            )
        # Re-raise as default action so subsequent signals kill promptly.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = _parse_args(argv)

    # If a worker is installing its own router, the router-service arg needs
    # to point at THIS worker's Helm-installed EPP Service (per-model name).
    # The chart truncates the Helm release name to 40 chars before appending
    # `-epp`, so we mirror the same rule here (see chart_epp_object_name).
    if args.install_router and args.only_model:
        from .orchestrator import chart_epp_object_name
        release_name = f"pricing-reversal-router-{args.only_model}"
        default_router_service = chart_epp_object_name(release_name)
        if args.router_service == "pricing-reversal-router-epp":
            # User didn't override; use the per-worker default.
            args.router_service = default_router_service

    orch = Orchestrator(
        config_dir=args.config_dir,
        results_dir=args.results_dir,
        namespace=args.namespace,
        hf_token_secret=args.hf_token_secret,
        vllm_template_path=args.vllm_template,
        router_service=args.router_service,
        router_port=args.router_port,
        resume=args.resume,
        dry_run=args.dry_run,
        only_model=args.only_model,
        results_subdir=args.results_subdir,
        install_router=args.install_router,
        router_values_template_path=args.router_values_template,
        router_chart=args.router_chart,
        router_chart_version=args.router_chart_version,
        clusters_yaml_path=args.clusters_yaml,
        skip_model_deploy=args.skip_model_deploy,
    )
    _install_abort_handler(orch.progress)
    return asyncio.run(orch.run())


if __name__ == "__main__":
    sys.exit(main())
