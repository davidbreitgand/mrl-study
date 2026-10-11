"""Entry point for the uber-harness controller.

Applied by `stack/uber/uber-job.yaml.tmpl`; runs one Pod that spawns N
worker Jobs (one per model in models.yaml), watches them, aggregates
their heartbeats into a dashboard, and verifies GPU cleanup on
completion.

Typical invocation:
    python -m pricing_reversal_harness.uber \
        --config-dir=/config \
        --results-dir=/results \
        --namespace=<your-namespace> \
        --hf-token-secret=llm-d-hf-token \
        --worker-image=<your-registry>/pricing-reversal-harness:latest \
        --worker-template=/config/worker-job.yaml.tmpl
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys

from ..progress import Progress
from .controller import UberController


def _install_abort_handler(progress: Progress) -> None:
    """SIGTERM/SIGINT handler: emit `study_aborted` for the uber
    controller before dying. Kubernetes sends SIGTERM ~30 s before
    SIGKILL on Job deletion / preemption / eviction. See the twin
    handler in `pricing_reversal_harness.__main__` for details."""

    def _handle(signum: int, _frame: object) -> None:
        try:
            progress.emit(
                "study_aborted",
                signal=signal.Signals(signum).name,
            )
            progress.update_state(phase="aborted", signal=signal.Signals(signum).name)
        except Exception:  # noqa: BLE001
            logging.getLogger(__name__).exception(
                "study_aborted emit failed for signal %s", signum,
            )
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)


def _configure_logging() -> None:
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
    ap = argparse.ArgumentParser(prog="pricing-reversal-uber")
    ap.add_argument("--config-dir", default="/config")
    ap.add_argument("--results-dir", default="/results")
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--hf-token-secret", default="llm-d-hf-token")
    ap.add_argument("--worker-image", required=True,
                    help="Container image the worker Jobs use "
                         "(same image as the uber-harness).")
    ap.add_argument("--worker-template", default="/config/worker-job.yaml.tmpl",
                    help="Jinja2 template rendered to a worker Job manifest per model.")
    ap.add_argument("--router-values-template",
                    default="/config/router-values.yaml.tmpl",
                    help="Path passed through to each worker for its Helm values.")
    ap.add_argument("--aggregator-interval-seconds", type=float, default=5.0,
                    help="How often to refresh /results/uber/aggregated.json.")
    ap.add_argument("--image-pull-policy", default="Always",
                    choices=("Always", "IfNotPresent", "Never"),
                    help="imagePullPolicy stamped into every rendered "
                         "worker Job. Default Always is belt-and-suspenders "
                         "with hash-tagged builds; IfNotPresent saves pull "
                         "traffic when the tag is provably immutable.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan without applying any Kubernetes objects.")
    ap.add_argument("--resume-from", default=None,
                    help="Cross-study resume: absolute path INSIDE the "
                         "container where a prior study's `consolidated/` "
                         "and `repeated_trial/` subtrees have been staged "
                         "(via `make resume-from FROM=<local-folder>`). "
                         "Uber patches the ConfigMap's experiment.yaml "
                         "to expose this path BEFORE spawning workers, "
                         "so each worker sees it at study_start and "
                         "copies matching seeds into its own results_dir. "
                         "Per-dataset validation (counts + fingerprint) "
                         "happens per worker in resume.py; only valid "
                         "seeds cause the corresponding dataset run to "
                         "be skipped. Null = no resume (fresh study).")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = _parse_args(argv)

    ctl = UberController(
        config_dir=args.config_dir,
        results_dir=args.results_dir,
        namespace=args.namespace,
        hf_token_secret=args.hf_token_secret,
        worker_image=args.worker_image,
        worker_template_path=args.worker_template,
        router_values_template_path=args.router_values_template,
        aggregator_interval_seconds=args.aggregator_interval_seconds,
        dry_run=args.dry_run,
        image_pull_policy=args.image_pull_policy,
        resume_from=args.resume_from,
    )
    _install_abort_handler(ctl.progress)
    return ctl.run()


if __name__ == "__main__":
    sys.exit(main())
