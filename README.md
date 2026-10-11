# mrl head-to-head study

Kubernetes-native evaluation harness for the two-term MRL scorer
(`mrl-scorer-hub`) against three production Distribution-axis scorers
in the llm-d-router EPP:

- `multicluster-queue-scorer` (production baseline, referred to as **QD**)
- `active-request-scorer` (**AR**)
- `latency-observation-scorer-hub` (**LAT**)
- `mrl-scorer-hub` (**MRL**, the new plugin)

Four scorer configurations × three arrival regimes (Poisson,
Weibull-0.5, Weibull-0.3) = 12 sequential cells against the same two
asymmetric spokes:

- spoke-gemma-a — TP=2 (slower per request)
- spoke-gemma-b — TP=4 (faster per request)

## Running the study

The full runbook is [docs/mrl-headtohead.md](docs/mrl-headtohead.md).
For convenience, a top-level `Makefile` wraps every step:

```bash
export KUBECONFIG=/path/to/your/kubeconfig
make prep                                                     # GPU preflight + spokes + PVC + RBAC
make images                                                   # build the EPP + harness images
make calibrate DATASET=<name> TARGET_RHO=<rho>                # find study_lambda for TARGET_RHO
make study     DATASET=<name> SCORER=<qd|ar|lat|mrl>          # 3 regimes for one scorer
make headtohead DATASET=<name>                                # aggregate + plots
make clean                                                    # tear down every K8s resource
```

See `make help` for the parameter surface.

## Reproducibility — quick facts

- **Runtime**: ~12-16 hours end-to-end for one cell 10 replication of a 4-scorer × 3-regime study
  at n=472 per cell (add ~10 hours if you want the p99-replicated
  Poisson variant covered by `scripts/build_poisson_replicate_table.py`).
- **Hardware**: 6 free NVIDIA H200 GPUs (spoke-gemma-a wants TP=2 = 2
  GPUs, spoke-gemma-b wants TP=4 = 4 GPUs) on a single Kubernetes
  cluster. See `docs/mrl-headtohead.md § Reproducibility` for the
  exact software prerequisites and step-by-step reviewer flow.
- **EPP image**: build from the llm-d-router `feature/mrl-aware`
  branch at the specific commit that ships with the accompanying
  pull request. See `scripts/build-image.sh` for the exact commands;
  the PR's tracking Issue records the reproducibility SHA.
- **Dataset**: `scripts/build_dataset.py` regenerates the `totrandom`
  and `totrandom-tail472` corpora from six public HuggingFace
  benchmarks. See § Datasets below and the script's header for the
  attribution table.

## Datasets

Any JSON file under
`harness/src/pricing_reversal_harness/data/<name>.json` whose top
level is `{"dataset_name": ..., "records": [...]}` is a valid
dataset. The operator drops in their own; the harness serves it
verbatim.

One example dataset is `totrandom` — a random-uniform shuffle
(seed=42) of prompts drawn from six public benchmarks:

| Source benchmark | Rows in totrandom (of 4834) | Category |
|---|---:|---|
| mmlupro | 3000 | Multiple-choice reasoning (A–J letter answers) |
| livecodebench | 1055 | Code generation / competitive programming |
| arc-agi-v1 | 400 | Abstraction and reasoning corpus (grid pattern tasks) |
| gpqa | 198 | Graduate-level physics/chemistry/biology QA |
| livemathbench | 121 | Math reasoning |
| aime | 60 | American Invitational Mathematics Examination problems |

To regenerate `totrandom.json` (and the `totrandom-tail472.json` tail
slice used by the head-to-head) from source:

```bash
pip install datasets
python scripts/build_dataset.py --out-dir harness/src/pricing_reversal_harness/data/
```

The builder script's header documents each source benchmark's
HuggingFace ID and the per-benchmark row count. Each source has its
own license; the builder deliberately does not redistribute prompt
content — the operator regenerates locally from the primary sources
so license responsibility stays with them.

The canonical `totrandom` shipped with the head-to-head measurements
uses seed=42 and was built on 2026-09-08. Byte-for-byte reproducibility
of the exact prompt strings is not guaranteed (HuggingFace dataset
revisions may drift); statistical reproducibility (same distribution
of prompt lengths, same benchmark mix) is.

## Calibration

`make calibrate DATASET=<name> TARGET_RHO=<rho>` probes the actual
fleet to find the `study_lambda` that lands ρ_honest at `TARGET_RHO`
on the given dataset. Two modes:

- **Full** (default): staged bisection. Each stage runs a 200-row
  Poisson probe against the QD Hub, measures ρ via Little's Law +
  PASTA, and bisects toward the target band. Converges in 2-5 stages
  typically.
- **Fast**: pass `LAMBDA=<value>` and the calibrator runs a single
  500-row confirmation probe, reports the observed ρ_honest, and
  warns if outside `TARGET_RHO ± 0.03`. Trust-the-operator; the
  cells then run with that λ regardless.

The result is written to `local_results/<DATASET>/calibration.json`,
which `run-config.sh` reads on every subsequent cell. Passing
`LAMBDA=<value>` to `make cell` or `make study` bypasses the file
entirely.

The operating-point choice (which `TARGET_RHO` to use) is left to
the operator; a companion simulation study can inform it if
applicable to your workload.

An example calibration file for the `totrandom-tail472` dataset at
`TARGET_RHO=0.90` ships at
`local_results/totrandom-tail472/calibration.json`. Its
`revision_history` records the full derivation trail from the
original tail472 head-to-head. External users on different fleets
should regenerate their own via `make calibrate`.

## Directory shape

**Hub** is the llm-d-router EPP that fronts the two spokes; each
scorer has its own Helm values file under `stack/router/`. See
[docs/mrl-headtohead.md § Stack layout](docs/mrl-headtohead.md) for
the full stack breakdown.

```
mrl_study/
  LICENSE                              # Apache 2.0
  Makefile                             # top-level driver
  README.md                            # this file
  docs/mrl-headtohead.md               # full runbook
  config/                              # experiment.yaml, arrival-regimes.yaml, ...
  contrib/                             # provider-specific extras (optional)
    coreweave/                           # CoreWeave-specific cluster picker
  harness/                             # canonical harness (Weibull-inter-arrival added)
    src/pricing_reversal_harness/data/ # drop dataset .json files here
  scripts/                             # everything the Makefile calls
  stack/
    router/                            # hub-values-{qd,ar,lat,mrl}.yaml
    modelserver/                       # vllm-deployment.yaml.tmpl
    harness/                           # job.yaml.tmpl, pvc.yaml, rbac.yaml
  local_results/<dataset>/             # per-dataset outputs; created on first run
    calibration.json                     # written by `make calibrate`
    <arrival-regime>/                    # e.g. poisson, weibull-0.5, weibull-0.3
      <scorer>-<arrival>-<dataset>-rep<N>/   # per-cell run outputs (rep0..repN)
    head-to-head-<arrival>-replicated/   # aggregator output
```

Canonical result paths for a completed study on dataset `<D>`:

- Aggregated results: `local_results/<D>/head-to-head-<arrival>-replicated/`
- Raw per-cell journals: `local_results/<D>/<arrival>/<scorer>-<arrival>-<D>-rep<N>/journal/<model>/*.jsonl`
- Calibration: `local_results/<D>/calibration.json`

## License

Apache License 2.0. See `LICENSE` in this directory.

The `totrandom` dataset generated by `scripts/build_dataset.py` is a
derivative work of six independent public benchmarks; users
redistributing generated datasets are responsible for meeting each
source's redistribution requirements.
