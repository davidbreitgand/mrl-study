#!/usr/bin/env bash
#
# run-config.sh — one-shot driver for a single (scorer, arrival regime,
# dataset) cell.
#
# Executes:
#
#   1. Resolve study_lambda:
#        - if --lambda passed, use that (and record its provenance).
#        - else, read from local_results/${DATASET}/calibration.json,
#          which `make calibrate` produces via the staged bisection
#          probe.
#   2. Render experiment.yaml for the target arrival regime + lambda,
#      pointed at the named dataset.
#   3. Apply + wait for the harness Job. RUN_TAG is
#      ${SCORER}-${ARRIVAL}-${DATASET} so multiple datasets can coexist
#      on the same PVC without the harness's --resume flag
#      short-circuiting a new dataset's cell.
#   4. Rsync outputs to local_results/${DATASET}/${ARRIVAL}/, so
#      results from different datasets never contaminate each other on
#      local disk.
#   5. Validate plumbing (served-endpoint header present, no 5xx,
#      no semaphore saturation) against the run's journal.
#
# Scorer configuration (which Hub is up) is NOT this script's concern.
# The caller runs `scripts/deploy-hub.sh` with the right HUB_VALUES,
# then `scripts/hub-smoke.sh` for a 3-curl sanity check, before invoking
# run-config.sh.
#
# Usage:
#   scripts/run-config.sh \
#       --scorer   mrl \
#       --arrival  poisson \
#       --rows     472 \
#       --dataset  totrandom-tail472
#
#   scripts/run-config.sh \
#       --scorer   qd \
#       --arrival  weibull-0.5 \
#       --rows     2000 \
#       --dataset  totrandom2000 \
#       --lambda   0.31
#
#   # Replicated Poisson cell (rep 1 of 3):
#   scripts/run-config.sh \
#       --scorer   mrl \
#       --arrival  poisson \
#       --rows     472 \
#       --dataset  totrandom-tail472 \
#       --replicate 1
#
# --replicate: optional non-negative integer (default 0). 0 means
# "the base cell" and elides the -rep suffix from RUN_TAG so behavior
# matches the pre-replication invocation. Positive values namespace both
# the PVC path (/results/<tag>-rep<N>) and the local subdir
# (local_results/<dataset>/<arrival>/<tag>-rep<N>) so multiple replicates
# of the same (scorer, arrival, dataset) coexist without collision.
#
# Env vars honored:
#   NAMESPACE              — default: mrl-headtohead
#   FULL_TIMEOUT_SECONDS   — default: 21600 (6 h); ceiling for the cell Job.

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
STUDY_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

ARRIVAL=""
ROWS=""
DATASET=""
SCORER=""
LAMBDA_OVERRIDE=""
REPLICATE="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arrival)   ARRIVAL="$2"; shift 2 ;;
    --rows)      ROWS="$2";    shift 2 ;;
    --dataset)   DATASET="$2"; shift 2 ;;
    --scorer)    SCORER="$2";  shift 2 ;;
    --lambda)    LAMBDA_OVERRIDE="$2"; shift 2 ;;
    --replicate) REPLICATE="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

[[ -n "${ARRIVAL}" ]] || { echo "--arrival required (poisson|weibull-0.5|weibull-0.3)" >&2; exit 1; }
[[ -n "${ROWS}"    ]] || { echo "--rows required"    >&2; exit 1; }
[[ -n "${DATASET}" ]] || { echo "--dataset required (basename of a JSON in harness/src/pricing_reversal_harness/data/)" >&2; exit 1; }
[[ -n "${SCORER}"  ]] || { echo "--scorer required (qd|ar|lat|mrl)" >&2; exit 1; }

case "${ARRIVAL}" in
  poisson|weibull-0.5|weibull-0.3) ;;
  *) echo "unknown arrival regime: ${ARRIVAL}" >&2; exit 1 ;;
esac

case "${SCORER}" in
  qd|ar|lat|mrl) ;;
  *) echo "unknown scorer: ${SCORER} (expected qd|ar|lat|mrl)" >&2; exit 1 ;;
esac

# REPLICATE must be a non-negative integer. Default 0 means "the base
# (unreplicated) cell" — RUN_TAG omits the -rep suffix so callers that
# do not opt into replication see the same behavior as before.
if ! [[ "${REPLICATE}" =~ ^[0-9]+$ ]]; then
  echo "--replicate must be a non-negative integer (got: ${REPLICATE})" >&2; exit 1
fi

log()  { printf '[run-config][%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
fail() { printf '[run-config][FAIL] %s\n' "$*" >&2; exit 1; }

NAMESPACE="${NAMESPACE:-mrl-headtohead}"
FULL_TIMEOUT_SECONDS="${FULL_TIMEOUT_SECONDS:-21600}"

# ---- Resolve study_lambda ------------------------------------------------
# Two paths: explicit override, or calibration.json under the dataset's
# local_results subtree. The two mechanisms are mutually exclusive at
# invocation time; if both are wanted (probe first, then override on a
# later cell) the operator writes the desired value into
# local_results/${DATASET}/calibration.json manually.
if [[ -n "${LAMBDA_OVERRIDE}" ]]; then
  STUDY_LAMBDA="${LAMBDA_OVERRIDE}"
  log "study_lambda = ${STUDY_LAMBDA} (from --lambda; operator-supplied)"
else
  CALIB="${STUDY_DIR}/local_results/${DATASET}/calibration.json"
  [[ -r "${CALIB}" ]] || fail "no --lambda provided and calibration.json missing at ${CALIB}; run 'make calibrate DATASET=${DATASET} TARGET_RHO=<target>' first, or pass --lambda"
  STUDY_LAMBDA=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['study_lambda'])" "${CALIB}")
  log "study_lambda = ${STUDY_LAMBDA} (from ${CALIB})"
fi

# ---- Resolve output directory --------------------------------------------
# Results partition by DATASET on local disk so consecutive runs against
# different datasets never write into the same local_results subtree.
OUT="${STUDY_DIR}/local_results/${DATASET}/${ARRIVAL}"
mkdir -p "${OUT}"

# ---- Render experiment ---------------------------------------------------
# RUN_TAG namespaces by scorer, dataset, and replicate so cells from
# different datasets, scorers, and replicates each write to a distinct
# PVC path (/results/${SCORER}-${ARRIVAL}-${DATASET}[-rep${N}]) and a
# distinct local subdir. Without dataset in the tag, a second dataset's
# cell on the same PVC would resume against the prior dataset's journal;
# without replicate in the tag, a second replicate of the same dataset
# would do likewise. REPLICATE=0 elides the -rep suffix so the base
# invocation matches the pre-replication layout.
if [[ "${REPLICATE}" -eq 0 ]]; then
  RUN_TAG="${SCORER}-${ARRIVAL}-${DATASET}"
else
  RUN_TAG="${SCORER}-${ARRIVAL}-${DATASET}-rep${REPLICATE}"
fi
FULL_YAML="/tmp/mrl-${RUN_TAG}-experiment.yaml"
log "rendering experiment (${ROWS} rows, ${ARRIVAL}, dataset=${DATASET}, replicate=${REPLICATE})"
"${SCRIPT_DIR}/render_experiment.py" \
  --regime "${ARRIVAL}" \
  --lambda "${STUDY_LAMBDA}" \
  --rows "${ROWS}" \
  --dataset "${DATASET}" \
  --output "${FULL_YAML}"

# ---- Apply + wait --------------------------------------------------------
log "applying Job (harness-${RUN_TAG})"
RUN_TAG="${RUN_TAG}" \
  HARNESS_EXPERIMENT_YAML="${FULL_YAML}" \
  "${SCRIPT_DIR}/apply-harness-job.sh"

log "waiting up to ${FULL_TIMEOUT_SECONDS}s for Job to complete"
kubectl wait --for=condition=complete "job/harness-${RUN_TAG}" \
  --namespace "${NAMESPACE}" --timeout="${FULL_TIMEOUT_SECONDS}s" || \
  fail "Job did not complete in ${FULL_TIMEOUT_SECONDS}s"

# ---- Rsync + validate ----------------------------------------------------
log "pulling results to ${OUT}"
if ! "${SCRIPT_DIR}/results-rsync.sh" "${NAMESPACE}" "${OUT}" "${RUN_TAG}"; then
  fail "results-rsync.sh returned non-zero; PVC state at /results/${RUN_TAG} may be intact, rerun the rsync step manually to recover"
fi

# Verify the rsync actually produced the expected journal on local disk;
# results-rsync.sh's exit code is not always reliable across pipe/tee
# wrappers so we cross-check the concrete artifact here. Fails loud if
# the journal is missing so downstream aggregators do not silently see
# an incomplete replicate set.
EXPECTED_JOURNAL="${OUT}/${RUN_TAG}/journal"
if [[ ! -d "${EXPECTED_JOURNAL}" ]]; then
  fail "rsync completed but ${EXPECTED_JOURNAL} is missing; local disk is inconsistent with PVC"
fi
JOURNAL_ROWS=$(find "${EXPECTED_JOURNAL}" -name "*.jsonl" -exec wc -l {} \; 2>/dev/null | awk '{s+=$1} END {print s+0}')
if [[ "${JOURNAL_ROWS}" -lt "${ROWS}" ]]; then
  fail "rsync completed but journal has ${JOURNAL_ROWS} rows (expected ${ROWS}); rerun rsync manually"
fi
log "journal integrity check: ${JOURNAL_ROWS} rows at ${EXPECTED_JOURNAL}"

log "validating plumbing (served-endpoint header, no 5xx, no semaphore saturation)"
"${SCRIPT_DIR}/validate_smoke.py" "${OUT}" \
  --require-served-endpoint --forbid-5xx --forbid-semaphore-saturation \
  || fail "validation failed; inspect ${OUT} and rerun once resolved"

log "cell done: scorer=${SCORER} arrival=${ARRIVAL} dataset=${DATASET} replicate=${REPLICATE} rows=${ROWS} out=${OUT}"
