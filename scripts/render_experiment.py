#!/usr/bin/env python3
"""Render a per-cell experiment.yaml for the mrl head-to-head studies.

Given an arrival regime name (poisson / weibull-0.5 / weibull-0.3), a
lambda value (from calibrate-lambda.sh's calibration.json), and a row
count, produces a valid experiment.yaml the harness Job can consume.

Reads:
  - {STUDY_DIR}/config/experiment.yaml       — base experiment settings
  - {STUDY_DIR}/config/arrival-regimes.yaml  — arrival_stages templates

Writes:
  - Path given by --output (default: /tmp/mrl-experiment.yaml)

Usage:
  scripts/render_experiment.py \\
      --regime poisson \\
      --lambda 15.0 \\
      --rows 2000 \\
      --output /tmp/mrl-experiment.yaml
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import yaml


REGIME_NAMES = ("poisson", "weibull-0.5", "weibull-0.3")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regime", required=True, choices=REGIME_NAMES,
                    help="which arrival regime to render (poisson, "
                         "weibull-0.5, weibull-0.3).")
    ap.add_argument("--lambda", dest="lam", type=float, required=True,
                    help="arrivals per second for this cell; take the "
                         "study_lambda field from calibration.json "
                         "(computed as target_rho * c_effective / mu_service).")
    ap.add_argument("--rows", type=int, required=True,
                    help="row count for this run (2000 for full, "
                         "50/150 for smoke).")
    ap.add_argument("--dataset", type=str, default=None,
                    help="Override consolidated_datasets. Default: use the "
                         "list from experiment.yaml (typically 'totrandom', "
                         "2000 rows). The calibration probe passes a smaller "
                         "pre-truncated slice (e.g. 'totrandom-tail472') so "
                         "the arrival stage's `rows` cap actually terminates "
                         "the run — the harness drives until the dataset is "
                         "exhausted, not until the last stage's `rows` quota.")
    ap.add_argument("--output", type=pathlib.Path,
                    default=pathlib.Path("/tmp/mrl-experiment.yaml"),
                    help="path to write the rendered experiment.yaml.")
    ap.add_argument("--study-dir", type=pathlib.Path,
                    default=None,
                    help="mrl_study/ root. Default: parent of this script's dir.")
    args = ap.parse_args()

    study_dir = args.study_dir or pathlib.Path(__file__).resolve().parents[1]
    experiment_yaml = study_dir / "config" / "experiment.yaml"
    regimes_yaml = study_dir / "config" / "arrival-regimes.yaml"

    if not experiment_yaml.exists():
        print(f"ERROR: {experiment_yaml} not found", file=sys.stderr)
        return 1
    if not regimes_yaml.exists():
        print(f"ERROR: {regimes_yaml} not found", file=sys.stderr)
        return 1

    with experiment_yaml.open() as f:
        exp = yaml.safe_load(f) or {}
    with regimes_yaml.open() as f:
        regs = yaml.safe_load(f) or {}

    # The base experiment.yaml is the costguard defaults; override the
    # arrival mechanics for this study cell.
    root = exp.get("experiment") or exp  # tolerate both shapes
    root["arrival_mode"] = "open_loop_staged"

    if args.dataset:
        root["consolidated_datasets"] = [args.dataset]

    # Build one ArrivalStage from the regime template. Templates carry
    # `${STUDY_LAMBDA}` and `rows: 2000` placeholders; substitute here.
    regime = regs["regimes"][args.regime]
    stages = []
    for stage_tmpl in regime["arrival_stages"]:
        stage = {"lambda": float(args.lam), "rows": int(args.rows)}
        if "weibull_shape" in stage_tmpl:
            stage["weibull_shape"] = float(stage_tmpl["weibull_shape"])
        stages.append(stage)
    root["arrival_stages"] = stages

    # target_concurrent is meaningless in open-loop mode; leave it in
    # the file (harness config schema requires it) but ensure it's a
    # sane placeholder.
    root.setdefault("target_concurrent", 12)

    # experiment.yaml can be either flat (root == exp) or under
    # `experiment:` — preserve whichever shape the base file used.
    if "experiment" in exp:
        out = {"experiment": root}
    else:
        out = root

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print(f"[render_experiment] wrote {args.output}", file=sys.stderr)
    print(f"[render_experiment]   regime={args.regime} lambda={args.lam} rows={args.rows}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
