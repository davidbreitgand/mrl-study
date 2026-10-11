#!/usr/bin/env bash
#
# validate_lambda_probe.sh — probe the actual fleet to find a
# study_lambda that lands at target_rho on the given dataset.
#
# Two modes:
#
# --mode full       (default): staged bisection. Runs up to --max-stages
#                   Poisson probes, each measuring empirical ρ from the
#                   journal via Little's Law. Verdicts IN_BAND /
#                   UNDER_DRIVEN / OVER_DRIVEN steer the bisection.
#                   Writes local_results/${DATASET}/calibration.json on
#                   convergence.
#
# --mode fast       Single-stage: drives the fleet with the operator's
#                   --lambda for --rows requests, reports the observed
#                   ρ_honest, warns if outside target ± --band-half-width
#                   (default 0.03), and always writes calibration.json.
#                   Does NOT iterate; operator is responsible for the
#                   value.
#
# Both modes require the QD Hub to be deployed (QD is the calibration
# reference because it does no per-request routing math).
#
# Usage:
#   scripts/validate_lambda_probe.sh --dataset totrandom-tail472 \
#       --target-rho 0.90 --mode full
#
#   scripts/validate_lambda_probe.sh --dataset totrandom-tail472 \
#       --lambda 0.272 --target-rho 0.90 --mode fast

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
STUDY_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

MODE="full"
DATASET=""
TARGET_RHO=""
LAMBDA_START=""
PROBE_ROWS=""
MAX_STAGES="5"
STAGE_TIMEOUT_SECONDS="5400"
BAND_HALF_WIDTH="0.03"
WARMUP="30"
C_EFFECTIVE="16"
NAMESPACE="${NAMESPACE:-mrl-headtohead}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)            MODE="$2"; shift 2 ;;
    --dataset)         DATASET="$2"; shift 2 ;;
    --target-rho)      TARGET_RHO="$2"; shift 2 ;;
    --lambda)          LAMBDA_START="$2"; shift 2 ;;
    --rows)            PROBE_ROWS="$2"; shift 2 ;;
    --max-stages)      MAX_STAGES="$2"; shift 2 ;;
    --stage-timeout)   STAGE_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --band-half-width) BAND_HALF_WIDTH="$2"; shift 2 ;;
    --warmup)          WARMUP="$2"; shift 2 ;;
    --c-effective)     C_EFFECTIVE="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

log()  { printf '[validate-lambda-probe][%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
fail() { printf '[validate-lambda-probe][FAIL] %s\n' "$*" >&2; exit 1; }

command -v kubectl >/dev/null 2>&1 || fail "kubectl not on PATH"
command -v python3 >/dev/null 2>&1 || fail "python3 not on PATH"

case "${MODE}" in
  full|fast) ;;
  *) fail "--mode must be one of: full, fast" ;;
esac
[[ -n "${DATASET}"    ]] || fail "--dataset required"
[[ -n "${TARGET_RHO}" ]] || fail "--target-rho required"

# --rows default depends on mode: full-bisection probes short (200 rows,
# many stages), fast-probe drives longer (default 500 rows) since it is
# operator-committed to that value.
if [[ -z "${PROBE_ROWS}" ]]; then
  if [[ "${MODE}" == "full" ]]; then
    PROBE_ROWS="200"
  else
    PROBE_ROWS="500"
  fi
fi

# Target band is [target_rho - band_half_width, target_rho + band_half_width].
TARGET_BAND_LO=$(python3 -c "print(round(${TARGET_RHO} - ${BAND_HALF_WIDTH}, 4))")
TARGET_BAND_HI=$(python3 -c "print(round(${TARGET_RHO} + ${BAND_HALF_WIDTH}, 4))")

# Per-dataset calibration.json.
CALIB_DIR="${STUDY_DIR}/local_results/${DATASET}"
CALIB="${CALIB_DIR}/calibration.json"
mkdir -p "${CALIB_DIR}"

# ---- Resolve initial λ candidate -----------------------------------------
# fast mode: --lambda is required.
# full mode: --lambda optional; if omitted, use analytic prior of
#            target_rho * c_effective / mu_seed, with mu_seed=60s (a
#            reasonable order-of-magnitude for gemma-4-26b-a4b on the
#            asymmetric fleet). Operator can pre-seed calibration.json
#            with a smarter analytic estimate to skip this default.
if [[ "${MODE}" == "fast" ]]; then
  [[ -n "${LAMBDA_START}" ]] || fail "--lambda required in fast mode"
  LAMBDA_CURRENT="${LAMBDA_START}"
elif [[ -n "${LAMBDA_START}" ]]; then
  LAMBDA_CURRENT="${LAMBDA_START}"
elif [[ -r "${CALIB}" ]]; then
  LAMBDA_CURRENT=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['study_lambda'])" "${CALIB}")
  log "reusing existing calibration.json λ as starting point: ${LAMBDA_CURRENT}"
else
  MU_SEED="60"
  LAMBDA_CURRENT=$(python3 -c "print(${TARGET_RHO} * ${C_EFFECTIVE} / ${MU_SEED})")
  log "no --lambda given; using analytic seed target_rho·c/μ_seed=${LAMBDA_CURRENT} (μ_seed=${MU_SEED}s)"
fi

log "mode:                   ${MODE}"
log "dataset:                ${DATASET}"
log "target ρ:               ${TARGET_RHO}"
log "target band:            [${TARGET_BAND_LO}, ${TARGET_BAND_HI}]"
log "c_effective:            ${C_EFFECTIVE}"
log "probe rows/stage:       ${PROBE_ROWS}"
log "initial λ:              ${LAMBDA_CURRENT}"
log ""

STAGE_HISTORY_JSON="[]"
VALIDATION_TMP="$(mktemp -d)"
trap 'rm -rf "${VALIDATION_TMP}"' EXIT

# In fast mode we run exactly one stage regardless of the verdict.
if [[ "${MODE}" == "fast" ]]; then
  STAGES_TO_RUN=1
else
  STAGES_TO_RUN="${MAX_STAGES}"
fi

converged="0"
for STAGE in $(seq 0 $((STAGES_TO_RUN - 1))); do
  STAGE_DIR="${VALIDATION_TMP}/stage-${STAGE}"
  mkdir -p "${STAGE_DIR}"

  log "=== Stage ${STAGE}: driving λ = ${LAMBDA_CURRENT} on dataset ${DATASET} ==="

  RENDERED_YAML="/tmp/mrl-probe-${DATASET}-stage-${STAGE}-experiment.yaml"
  "${SCRIPT_DIR}/render_experiment.py" \
    --regime poisson \
    --lambda "${LAMBDA_CURRENT}" \
    --rows "${PROBE_ROWS}" \
    --dataset "${DATASET}" \
    --output "${RENDERED_YAML}"

  PROBE_TAG="probe-${DATASET}-${STAGE}"
  log "  applying probe Job (harness-${PROBE_TAG})"
  RUN_TAG="${PROBE_TAG}" \
    HARNESS_EXPERIMENT_YAML="${RENDERED_YAML}" \
    "${SCRIPT_DIR}/apply-harness-job.sh" || fail "stage ${STAGE} Job failed to start"

  log "  waiting up to ${STAGE_TIMEOUT_SECONDS}s for stage Job to complete"
  kubectl wait --for=condition=complete "job/harness-${PROBE_TAG}" \
    --namespace "${NAMESPACE}" --timeout="${STAGE_TIMEOUT_SECONDS}s" || \
    fail "stage ${STAGE} Job did not complete in ${STAGE_TIMEOUT_SECONDS}s"

  log "  pulling stage results"
  "${SCRIPT_DIR}/results-rsync.sh" "${NAMESPACE}" "${STAGE_DIR}" "${PROBE_TAG}"

  STAGE_JSON="${VALIDATION_TMP}/stage-${STAGE}.json"
  "${SCRIPT_DIR}/probe_stage_analyze.py" \
    --results "${STAGE_DIR}" \
    --lambda-driven "${LAMBDA_CURRENT}" \
    --c-effective "${C_EFFECTIVE}" \
    --target-rho "${TARGET_RHO}" \
    --target-band "${TARGET_BAND_LO}" "${TARGET_BAND_HI}" \
    --warmup "${WARMUP}" \
    > "${STAGE_JSON}"

  STAGE_HISTORY_JSON=$(python3 -c "
import json, sys
history = json.loads(sys.argv[1])
stage = json.load(open(sys.argv[2]))
stage['stage_index'] = int(sys.argv[3])
history.append(stage)
print(json.dumps(history))
" "${STAGE_HISTORY_JSON}" "${STAGE_JSON}" "${STAGE}")

  VERDICT=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['verdict'])" "${STAGE_JSON}")
  RHO_OBS=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['rho_honest'])" "${STAGE_JSON}")

  if [[ "${MODE}" == "fast" ]]; then
    log "  fast-probe result: λ=${LAMBDA_CURRENT} landed ρ_honest=${RHO_OBS} (verdict=${VERDICT})"
    if [[ "${VERDICT}" != "IN_BAND" ]]; then
      log "  WARNING: observed ρ=${RHO_OBS} is outside target band [${TARGET_BAND_LO}, ${TARGET_BAND_HI}]"
      log "  WARNING: proceeding anyway on operator's judgment. Rerun with --mode full to bisect."
    fi
    converged="1"
    break
  fi

  # full mode
  LAMBDA_NEXT=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['lambda_next'])" "${STAGE_JSON}")
  if [[ "${VERDICT}" == "IN_BAND" ]]; then
    log "  verdict: IN_BAND — accept λ = ${LAMBDA_CURRENT} (ρ_honest=${RHO_OBS})"
    converged="1"
    break
  fi

  log "  verdict: ${VERDICT} (ρ_honest=${RHO_OBS}) — bisect toward λ = ${LAMBDA_NEXT}"
  LAMBDA_CURRENT="${LAMBDA_NEXT}"
done

if [[ "${MODE}" == "full" && "${converged}" != "1" ]]; then
  log ""
  log "FAILED to converge after ${MAX_STAGES} stages."
  log "Stage history:"
  python3 -c "
import json, sys
h = json.loads(sys.argv[1])
for s in h:
    print(f\"  stage {s['stage_index']}: λ_driven={s['lambda_driven']:.4f}, \"
          f\"ρ_honest={s['rho_honest']:.3f}, verdict={s['verdict']}\")
" "${STAGE_HISTORY_JSON}"
  log ""
  log "Options:"
  log "  1. Rerun with --mode fast --lambda <manual-pick> to accept a value and validate it once."
  log "  2. Rerun with a widened --band-half-width."
  exit 2
fi

# Write calibration.json.
log ""
log "writing λ = ${LAMBDA_CURRENT} to ${CALIB}"
python3 <<PYEOF
import json, datetime, os
calib = {
    "dataset":           "${DATASET}",
    "target_rho":        float("${TARGET_RHO}"),
    "target_band":       [float("${TARGET_BAND_LO}"), float("${TARGET_BAND_HI}")],
    "c_effective":       int("${C_EFFECTIVE}"),
    "study_lambda":      float("${LAMBDA_CURRENT}"),
    "study_lambda_source": "${MODE}_probe",
    "validation_stage_history": json.loads('''${STAGE_HISTORY_JSON}'''),
    "validated_at":      datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
os.makedirs(os.path.dirname("${CALIB}"), exist_ok=True)
with open("${CALIB}", "w") as f:
    json.dump(calib, f, indent=2)
PYEOF

log "calibration.json:"
cat "${CALIB}"
