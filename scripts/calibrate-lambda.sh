#!/usr/bin/env bash
#
# calibrate-lambda.sh — one-time lambda calibration for the mrl studies.
#
# The correct calibration is the ρ = λ / (c · μ) identity solved for λ:
#
#     study_lambda = target_rho · c_effective / mu_service
#
# where mu_service is the mean per-request service time measured under
# near-zero queueing. Running a closed-loop probe at target_concurrent=1
# gives that measurement directly: only one request in flight at any
# moment, so `elapsed_seconds` in each journal row IS mu_service
# (no queueing contribution).
#
# Why NOT "drive at a candidate lambda and measure completions/sec":
# under any steady state, completions/sec == arrivals/sec (conservation
# of flow). That measurement can't distinguish "fleet at its ceiling"
# from "fleet under-driven by the driver." mu_service under
# target_concurrent=1 is a direct measurement, no fixed-point iteration
# required.
#
# Prereqs:
#   - Spokes are deployed (scripts/deploy-spoke.sh a; scripts/deploy-spoke.sh b).
#   - The QD Hub is deployed. QD is the calibration reference because
#     it does no per-request work of its own; the measured mu reflects
#     the spokes, not the scorer.
#     Deploy with:
#       HUB_VALUES=queue_depth_study/stack/router/hub-values-qd.yaml \\
#         scripts/deploy-hub.sh
#
# Usage:
#   scripts/calibrate-lambda.sh                           # defaults
#   scripts/calibrate-lambda.sh --rows 200                # more samples
#   scripts/calibrate-lambda.sh --c-effective 24          # 3 spokes × 8 seqs
#   scripts/calibrate-lambda.sh --target-rho 0.95         # different rho
#
# Overridable env:
#   NAMESPACE           default mrl-headtohead
#
# Output:
#   local_results/calibration.json  — study_lambda ready for run-config.sh

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
STUDY_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

# The fleet's effective concurrency: sum of vLLM max_num_seqs across
# running spokes. This study runs 2 spokes with max_num_seqs=8 (see
# config/models.yaml), so c_effective = 16.
C_EFFECTIVE="16"
PROBE_ROWS="100"
TARGET_RHO="0.98"
WARMUP="10"
NAMESPACE="${NAMESPACE:-mrl-headtohead}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --c-effective) C_EFFECTIVE="$2"; shift 2 ;;
    --rows)        PROBE_ROWS="$2";  shift 2 ;;
    --target-rho)  TARGET_RHO="$2";  shift 2 ;;
    --warmup)      WARMUP="$2";      shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

log()  { printf '[calibrate-lambda][%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
fail() { printf '[calibrate-lambda][FAIL] %s\n' "$*" >&2; exit 1; }

command -v kubectl >/dev/null 2>&1 || fail "kubectl not on PATH"
command -v python3 >/dev/null 2>&1 || fail "python3 not on PATH"

CALIB_OUT="${STUDY_DIR}/local_results/calibration.json"
LOCAL_RESULTS_TMP="$(mktemp -d)"
trap 'rm -rf "${LOCAL_RESULTS_TMP}"' EXIT

log "c_effective:       ${C_EFFECTIVE}"
log "probe rows:        ${PROBE_ROWS}"
log "target rho:        ${TARGET_RHO}"
log "warmup drop:       ${WARMUP}"
log "namespace:         ${NAMESPACE}"

# 1. Write the closed-loop, target_concurrent=1 experiment.yaml.
#    Distinct shape from run-config.sh's arrival regimes: no
#    arrival_stages, no arrival lambda; the driver keeps exactly one
#    request in flight at all times.
RENDERED_YAML="/tmp/mrl-calibration-experiment.yaml"
log "writing calibration probe experiment.yaml -> ${RENDERED_YAML}"
cat > "${RENDERED_YAML}" <<YAMLEOF
# Calibration probe: measure μ_service under near-zero queueing.
# target_concurrent=1 => at most one request in flight at any moment,
# so elapsed_seconds is service time, not sojourn time.
experiment:
  arrival_mode: closed_loop
  target_concurrent: 1
  max_concurrent_requests: 4          # runaway guard only; probe never hits it
  service_time_ewma_alpha: 0.05       # unused at target_concurrent=1
  arrival_seed: 42
  repeated_trials: 0
  repeated_trial_dataset: totrandom
  consolidated_datasets: [totrandom]
  no_endpoints_recovery_seconds: 300
  all_errored_threshold: 0.98
  ready_probe_deadline_seconds: 300
YAMLEOF

# 2. Apply the probe as a ConfigMap + Job via apply-harness-job.sh.
log "applying probe Job (RUN_TAG=calibration)"
RUN_TAG="calibration" \
  HARNESS_EXPERIMENT_YAML="${RENDERED_YAML}" \
  "${SCRIPT_DIR}/apply-harness-job.sh" || fail "probe Job failed to start"

# 3. Wait for completion. A 100-row probe at target_concurrent=1 with
#    typical service time ~5-15 s takes 10-25 minutes wall-clock.
#    Deadline is generous.
log "waiting up to 60 min for probe Job to complete"
kubectl wait --for=condition=complete "job/harness-calibration" \
  --namespace "${NAMESPACE}" --timeout=3600s || \
  fail "probe Job did not complete in 60 min"

# 4. Rsync the probe results.
log "pulling probe results"
"${SCRIPT_DIR}/results-rsync.sh" "${NAMESPACE}" "${LOCAL_RESULTS_TMP}"

# 5. Compute μ_service and derive study_lambda.
mkdir -p "${STUDY_DIR}/local_results"
"${SCRIPT_DIR}/compute_lambda.py" \
  --results "${LOCAL_RESULTS_TMP}" \
  --c-effective "${C_EFFECTIVE}" \
  --target-rho "${TARGET_RHO}" \
  --warmup "${WARMUP}" \
  --output "${CALIB_OUT}"

log "calibration written to ${CALIB_OUT}"
cat "${CALIB_OUT}"
