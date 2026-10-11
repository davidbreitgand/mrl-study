# mrl head-to-head study — runbook

Drives the 4-scorer × 3-regime head-to-head on any dataset, at any
target ρ. Each cell writes to
`local_results/<dataset>/<arrival-regime>/<scorer>-<arrival>-<dataset>-rep<N>/`;
the aggregator produces
`local_results/<dataset>/head-to-head-<arrival>-replicated/` with the
JSON, plots, and per-regime text summary.

Every step is wrapped in the top-level `Makefile`. This document
explains what each target does and lists the raw scripts underneath
in case the operator wants to run them by hand.

## Reproducibility

To reproduce the head-to-head numbers in the pull request's tracking
Issue, follow this exact sequence.

### 1. Prerequisites

Software:

- Kubernetes cluster (this study was measured on CoreWeave; any
  cluster with the hardware below works). `KUBECONFIG` must point at
  it.
- `kubectl`, `helm` v3.x, `python3` ≥ 3.11 on the operator's laptop.
- `docker` or `podman` with push access to a container registry the
  cluster can pull from.
- `pip install datasets` if you plan to regenerate the totrandom
  corpus locally.

Hardware:

- 6 free NVIDIA H200 GPUs in one cluster, dedicated to this study
  for the duration of the run:
  - spoke-gemma-a: TP=2 → 2 GPUs
  - spoke-gemma-b: TP=4 → 4 GPUs
- The spokes serve `google/gemma-4-26B-A4B-it` with vLLM
  `max_num_seqs=8` per spoke (fleet `c_effective = 16`).
- If your fleet differs, you must recalibrate `study_lambda` for
  your `TARGET_RHO` via `make calibrate`; the shipped
  `totrandom-tail472/calibration.json` is specific to this fleet.

### 2. Wall-clock expectation

At `study_lambda = 0.272 req/s` (the ρ=0.90 calibration point) and
`n=472` rows per cell, one Poisson cell completes in ~35-60 min
including drain. Full-study cost:

- **Basic head-to-head (3 arrivals × 4 scorers × n=472, single-shot)**:
  ~12-14 hours end-to-end on a dedicated fleet.
- **Poisson-replicate variant (Poisson × 4 scorers × 4 replicates)**:
  ~12-13 hours on top of the basic study.
- **Full replicate variant (3 arrivals × 4 scorers × 4 replicates)**:
  ~50-60 hours on top.

Add ~10 min per Hub swap between scorer transitions and ~10 min for
the `make calibrate` probe if you re-derive `study_lambda`.

### 3. EPP image from the PR

The scorer implementation under test lives in the llm-d-router
`feature/mrl-aware` branch. Reviewers verifying the head-to-head
should build from the exact commit that ships with the PR:

```bash
cd path/to/your/llm-d-router-checkout
git fetch origin feature/mrl-aware
git checkout <TODO: PR commit SHA — updated after PR opens; check the tracking Issue for the current value>
cd path/to/mrl_study/
export OWNER=your-registry-namespace           # your ghcr / dockerhub user
ROUTER_DIR=path/to/your/llm-d-router-checkout scripts/build-image.sh
```

The script refuses to build off a branch other than `feature/mrl-aware`
unless `ALLOW_ANY_BRANCH=1` is set, so a stray branch checkout fails
loud rather than silently building the wrong image. On success it
writes `.epp-image` next to this file with the fully-qualified image
reference; `scripts/deploy-hub.sh` reads that file and passes the
registry, repository, and tag to Helm via `--set-string`.

### 4. Dataset

Regenerate the `totrandom` corpus locally:

```bash
pip install datasets
python scripts/build_dataset.py \
    --out-dir harness/src/pricing_reversal_harness/data/
```

This writes both `totrandom.json` (4834 rows) and
`totrandom-tail472.json` (the tail slice used by the head-to-head).
Each source benchmark ships under its own license; the builder
delegates license compliance to the operator by not redistributing
prompt content.

### 5. Full end-to-end sequence

```bash
export KUBECONFIG=/path/to/your/kubeconfig
export OWNER=your-registry-namespace

# Build images and prepare cluster
scripts/build-image.sh                                    # requires the PR commit checkout
scripts/build-harness-image.sh
make prep                                                  # spokes + PVC + RBAC

# Calibrate lambda for your fleet
make calibrate DATASET=totrandom-tail472 TARGET_RHO=0.90

# Run four scorers, three regimes each, sequentially
for scorer in qd ar lat mrl; do
    make study DATASET=totrandom-tail472 SCORER=$scorer
done

# Aggregate
make headtohead DATASET=totrandom-tail472
```

Results land under
`local_results/totrandom-tail472/head-to-head-<arrival>-replicated/`
per the [canonical path convention](#canonical-result-paths).

## Parameter surface

| Parameter | Set by | Meaning |
|---|---|---|
| `DATASET` | Makefile variable | Basename of a JSON file in `harness/src/pricing_reversal_harness/data/<name>.json`. Also the leaf directory under `local_results/`. |
| `TARGET_RHO` | Makefile variable | Utilization the calibrator drives the fleet at. Cluster head-to-heads typically target 0.85-0.95. |
| `LAMBDA` | Makefile variable (optional) | Operator-supplied arrivals/sec. Overrides `calibration.json` for cell runs; switches `make calibrate` to fast mode. |
| `SCORER` | Makefile variable | One of `qd`, `ar`, `lat`, `mrl`. Selects which Hub is deployed and which `hub-values-*.yaml` is used. |
| `ARRIVAL` | Makefile variable | One of `poisson`, `weibull-0.5`, `weibull-0.3`. |
| `ROWS` | Makefile variable (optional) | Row count for the cell. Defaults to the dataset's `total` field. |
| `NAMESPACE` | Env / Makefile variable | K8s namespace; default `mrl-headtohead`. |

Dataset-scoped namespacing works two ways:

- **On the PVC**: `RUN_TAG = <scorer>-<arrival>-<dataset>` (e.g.
  `mrl-poisson-totrandom-tail472`) so cells from different datasets
  never collide on `/results/`, and the harness's `--resume` never
  short-circuits a new dataset's cell.
- **On local disk**: `local_results/<dataset>/<regime>/<RUN_TAG>/` so
  successive `make study DATASET=...` invocations against different
  datasets don't overwrite each other after `results-rsync.sh` pulls
  them down.

The PVC itself and the K8s namespace stay unchanged across datasets.

## Stack layout

The Kubernetes manifests and Helm values consumed by the scripts live
under `stack/`, separated by the piece of the fleet they configure:

| Dir | Configures | Consumed by |
|---|---|---|
| `stack/router/` | Hub EPP + spoke gateway | `scripts/deploy-hub.sh`, `scripts/deploy-spoke.sh` |
| `stack/modelserver/` | vLLM Deployment on each spoke | `scripts/deploy-spoke.sh` |
| `stack/harness/` | Harness Job + PVC + RBAC | `scripts/apply-harness-job.sh`, `make prep` |

**Terminology used throughout this runbook:**

- **Hub** — the llm-d-router Endpoint Picker (EPP) that fronts the
  fleet. All harness traffic hits the Hub's Service; the Hub scores
  candidate spokes and forwards each request to the winner. `make
  cell` / `make study` deploy the Hub with `hub-values-<scorer>.yaml`,
  so which scorer is measured is determined by which Hub is up.
- **Spoke** — a vLLM Deployment serving the model. Two spokes,
  intentionally asymmetric: `spoke-gemma-a` at TP=2, `spoke-gemma-b`
  at TP=4. Both serve the same model.

### `stack/router/` files

- **`hub-values-qd.yaml`** — Helm values for the QD-scorer Hub.
  Sets the EPP image, wires the `multicluster-queue-scorer` plugin,
  injects `destination-endpoint-served` as the
  `ResponseHeaderProcessor` (so `hub-smoke.sh` can attribute each
  response to a spoke), mounts the `clusters.yaml` ConfigMap.
- **`hub-values-ar.yaml`** — same shape; plugin block wires
  `active-request-scorer` (AR).
- **`hub-values-latency.yaml`** — wires
  `latency-observation-scorer-hub` and its paired
  `latency-observer-producer-hub`.
- **`hub-values-mrl.yaml`** — wires `mrl-scorer-hub` and its paired
  `sojourn-time-observer-hub` producer.
- **`base.values.yaml`** — common Helm values shared by all four
  Hubs (chart repo pins, image pull policy defaults, service
  definitions). Referenced from each `hub-values-<scorer>.yaml`.
- **`spoke-values.yaml`** — Helm values for the router side of a
  spoke deployment (gateway, service).
- **`spoke-gateway.yaml.tmpl`** — Kubernetes template rendered
  per-spoke by `scripts/deploy-spoke.sh`; defines the Envoy-based
  gateway that talks to vLLM on that spoke.

### Why four separate `hub-values-*.yaml`, not one parameterized file

Each scorer's EPP `pluginsConfigFile` block has a different plugin
graph — different `Produces` / `Consumes` DataKey wiring, different
paired producers, sometimes different `pluginsCustomConfig`
parameters. The Helm chart's `pluginsCustomConfig` field takes the
raw YAML string, so fully parameterizing the four scorers inside one
values file would need per-scorer conditionals in Go templates. Four
short files diff more cleanly. **`make cell SCORER=<x>` resolves to
`HUB_VALUES=stack/router/hub-values-<x>.yaml`** — the Makefile just
picks the right file by scorer name.

## Prep — one-time, before any study

Bootstraps the cluster: hard-checks that the required GPUs are free,
deploys the two asymmetric spokes, and applies the harness PVC + RBAC.

```bash
export KUBECONFIG=/path/to/your/kubeconfig    # point at your cluster
make prep
make images
```

Raw scripts:

- `scripts/check-gpu-availability.sh` — hard preflight; exits 42
  with a diagnostic pod list on a conflict.
- `scripts/deploy-spoke.sh {a,b}` — helm install of the two spokes
  into `<NAMESPACE>-cw-{a,b}`.
- `scripts/build-image.sh` / `scripts/build-harness-image.sh` — build
  and push the EPP and harness images; write `.epp-image` /
  `.harness-image` tag files the deploy scripts read.

CoreWeave-specific helper: `contrib/coreweave/select-cluster.sh`
iterates through a hard-coded list of CoreWeave cluster kubeconfigs
and picks the first one with enough free GPUs. Use it via
`eval "$(contrib/coreweave/select-cluster.sh)"` before `make prep`.
Non-CoreWeave users skip this entirely and export `KUBECONFIG`
directly.

## Calibration — one-time per (dataset, target-ρ)

`make calibrate DATASET=<name> TARGET_RHO=<rho>` probes the fleet
against the actual dataset to find the `study_lambda` that lands at
`TARGET_RHO`.

### Full mode (default)

Runs a staged bisection. Each stage:

1. Deploys the QD Hub (calibration reference — no per-request routing math).
2. Renders a Poisson probe experiment at the current λ candidate.
3. Applies the probe Job, waits, pulls results.
4. Runs `scripts/probe_stage_analyze.py` to measure ρ two ways:

   - **ρ_honest** = mean(min(c, in_flight_at_arrival)) / c — PASTA-derived, clips at 1.0.
   - **ρ_λ** = λ_observed · mean(elapsed_seconds) / c — Little's Law form, inflates past 1.0 with queue wait.

5. Verdict: `IN_BAND` accepts and writes `calibration.json`;
   `UNDER_DRIVEN` raises λ; `OVER_DRIVEN` lowers λ. Halts after
   `--max-stages` (default 5).

### Fast mode

Pass `LAMBDA=<value>` and calibration switches to a single 500-row
confirmation probe. The observed `ρ_honest` is reported and warned on
if outside the target band, but no iteration happens — the operator
is responsible for the value. `calibration.json` is written
regardless. Useful when the operator already has a well-founded
estimate and just wants to verify.

```bash
make calibrate DATASET=totrandom-tail472 TARGET_RHO=0.90
make calibrate DATASET=totrandom-tail472 TARGET_RHO=0.90 LAMBDA=0.272   # fast mode
```

Raw script: `scripts/validate_lambda_probe.sh --mode {full,fast}
--dataset <name> --target-rho <rho> [--lambda <value>] [--rows <n>]
[--max-stages <n>] [--band-half-width <hw>]`.

Bypass entirely: pass `LAMBDA=<value>` to every subsequent
`make cell` / `make study` invocation; that skips
`calibration.json` on the read side.

## Study — one scorer × three regimes

```bash
make study DATASET=<name> SCORER=<qd|ar|lat|mrl>
```

Deploys the scorer's Hub once, runs `hub-smoke.sh` for a 3-curl
sanity check, then iterates over `poisson`, `weibull-0.5`,
`weibull-0.3` sequentially. `ROWS` defaults to the dataset's total;
override for shorter probes.

Repeat with each of the four `SCORER=` values to fill the 12-cell
grid. Order does not matter for correctness (per-cell RUN_TAG on the
PVC + per-dataset partitioning on disk).

## Cell — one scorer × one regime

```bash
make cell DATASET=<name> SCORER=<qd|ar|lat|mrl> ARRIVAL=<regime>
```

Same as `make study` but only one regime. Useful for reruns after a
single failure.

## Head-to-head aggregator

Once the 12 cells for a dataset are complete:

```bash
make headtohead DATASET=<name>
```

Reads `local_results/<name>/<arrival>/<scorer>-<arrival>-<name>-rep<N>/`
for each of the four scorers (all arrival regimes), computes
elapsed-time mean/median/p95/p99 per cell, and writes to
`local_results/<name>/head-to-head-<arrival>-replicated/`:

- `head-to-head.json` — 4-scorer × 3-regime tables with Δ% vs QD.
- `head-to-head.png` — grouped bar chart, one sub-plot per regime.
- `arm-attribution.png` — per-cell spoke-a/spoke-b traffic split.
- `head-to-head-log.txt` — one paragraph per regime.

Raw script: `scripts/build_headtohead_table.py --dataset <name>
--out <out-dir>`.

For the Poisson-replicated variant (mean ± SD across replicates of
p99 and central-tendency stats), run
`scripts/build_poisson_replicate_table.py --dataset <name>
--out <out-dir>` after all replicates land under
`local_results/<name>/poisson/`.

## Canonical result paths

Every study invocation lands its outputs in a deterministic path
under `local_results/`, so multiple datasets, arrivals, scorers, and
replicates coexist without collision.

| Path | Contents | Written by |
|---|---|---|
| `local_results/<D>/calibration.json` | `study_lambda`, `target_rho`, validation history | `make calibrate` |
| `local_results/<D>/<A>/<S>-<A>-<D>-rep<N>/journal/<model>/*.jsonl` | Per-request records | `make cell` / `make study` |
| `local_results/<D>/<A>/<S>-<A>-<D>-rep<N>/consolidated/*.json` | Roll-up JSON for that cell | (same) |
| `local_results/<D>/<A>/<S>-<A>-<D>-rep<N>/progress.jsonl` | Harness telemetry | (same) |
| `local_results/<D>/head-to-head-<A>-replicated/` | Aggregated tables + plots | `make headtohead` / `build_poisson_replicate_table.py` |

Where:

- `<D>` — dataset name (e.g. `totrandom-tail472`)
- `<A>` — arrival regime (`poisson`, `weibull-0.5`, `weibull-0.3`)
- `<S>` — scorer (`qd`, `ar`, `lat`, `mrl`)
- `<N>` — replicate index (0 = base cell, 1..k = additional replicates
  via `run-config.sh --replicate <N>`)

The RUN_TAG on the PVC side is `<S>-<A>-<D>-rep<N>` for replicated
cells and `<S>-<A>-<D>` for the base cell. The PVC is single across
datasets; local-disk namespacing by `<D>` prevents cross-dataset
contamination when `results-rsync.sh` pulls to disk.

## Teardown

```bash
make clean
```

Removes every K8s resource this study creates: uninstalls the Hub
release, deletes both spokes, drops the harness PVC + RBAC +
ConfigMaps + all harness Jobs, and finally deletes all three
namespaces (`<NAMESPACE>`, `<NAMESPACE>-cw-a`, `<NAMESPACE>-cw-b`).
`local_results/` on disk is left intact.

## Attaching a dataset

Drop the dataset's JSON file into
`harness/src/pricing_reversal_harness/data/<name>.json`, then
`make images` to rebuild the harness image with it bundled. Schema:

```json
{
  "dataset_name": "...",
  "total": 4834,
  "sources": {"benchmark-a": 3000, "benchmark-b": 1834},
  "seed": 42,
  "built_at": "2026-09-08T14:39:13Z",
  "records": [
    { "prompt": "...", "ground_truth": "...", "source_dataset": "...", "source_index": 0, "index": 0 },
    ...
  ]
}
```

### Building the `totrandom` example dataset

The study's canonical example dataset is `totrandom` — a random-shuffle
mix of six public HuggingFace benchmarks. To regenerate it locally:

```bash
pip install datasets
python scripts/build_dataset.py \
    --out-dir harness/src/pricing_reversal_harness/data/
```

That produces both `totrandom.json` (4834 rows across mmlupro,
livecodebench, arc-agi-v1, gpqa, livemathbench, aime) and
`totrandom-tail472.json` (the tail slice used by the shipped
head-to-head). See `scripts/build_dataset.py`'s header for per-benchmark
row counts, HuggingFace IDs, and license considerations.

The builder does NOT redistribute prompt content; each source benchmark
has its own license and the operator regenerates from the primary
sources so license responsibility stays local.

## Exit gate

A cell is done when:

- `local_results/<dataset>/<regime>/<scorer>-<regime>-<dataset>/`
  contains a journal (`journal/<model>/*.jsonl`), a consolidated
  roll-up JSON, and the harness telemetry files
  (`progress.jsonl`, `state.json`, `heartbeat.txt`).
- `scripts/run-config.sh` exited zero.
- `scripts/validate_smoke.py --require-served-endpoint --forbid-5xx
  --forbid-semaphore-saturation` passed (invoked from inside
  `run-config.sh` after each rsync).

A study is done when all 12 cells for the dataset pass their gates
and `make headtohead DATASET=<name>` writes non-empty artefacts.

## Troubleshooting

- **`check-gpu-availability.sh` exits 42.** Some team's workload is
  holding the GPUs. Inspect the printed pod list with
  `kubectl describe pod -n <ns> <name>`. Coordinate or move to a
  different cluster with `MRL_MIN_FREE_GPUS=6 scripts/select-cluster.sh`.

- **`hub-smoke.sh` fails.** The Hub is not routing traffic to any
  spoke, or the Hub's EPP is not attaching the
  `x-conformance-test-served-endpoint` header. Common causes: the
  Hub's `pluginsCustomConfig` is missing the
  `destination-endpoint-served` ResponseHeaderProcessor, or the
  `clusters.yaml` ConfigMap didn't mount at `/etc/epp/clusters.yaml`.
  Inspect the Hub EPP pod's logs with
  `kubectl -n <ns> logs -l app.kubernetes.io/instance=<ns>-hub -c epp --tail=100`.

- **`run-config.sh` halts on `validate_smoke.py`.** One of
  `--require-served-endpoint`, `--forbid-5xx`, or
  `--forbid-semaphore-saturation` failed. Inspect the cell's
  `progress.jsonl` for `semaphore_saturated_seconds`, and grep the
  journal for non-200 `status_code`.

- **A cell's Job runs long past its expected duration.** At
  `study_lambda = λ` req/s and `rows = R`, the arrival-driven floor
  is `R / λ` seconds; add a modest drain time. If a cell exceeds
  `FULL_TIMEOUT_SECONDS` (6 h default), the Job fails. Rerun with a
  larger timeout only after confirming progress:
  `kubectl -n <ns> logs -f job/harness-<scorer>-<arrival>-<dataset>`.

- **`kubectl wait` detaches without emitting `cell done`.** The
  driver's `kubectl wait --for=condition=complete` may age out if
  the kubeconfig token times out mid-wait, even though the Job
  itself completes. Symptom: pod shows `Completed`, Job shows
  `Complete`, but `run-config.sh` is still running. Kill the driver
  and manually finish:

  ```bash
  scripts/results-rsync.sh <NAMESPACE> \
      local_results/<DATASET>/<ARRIVAL>/ \
      <SCORER>-<ARRIVAL>-<DATASET>
  scripts/validate_smoke.py local_results/<DATASET>/<ARRIVAL>/ \
      --require-served-endpoint --forbid-5xx \
      --forbid-semaphore-saturation
  ```

- **Calibration fails to converge in full mode.** Rerun with a
  widened `--band-half-width` (e.g. `0.05` instead of `0.03`), or
  pick a manual λ and run `make calibrate ... LAMBDA=<value>` in
  fast mode to accept it with one confirmation probe.
