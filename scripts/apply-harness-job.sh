#!/usr/bin/env bash
#
# apply-harness-job.sh — render + apply a single harness Job for one cell.
#
# Consumed by both calibrate-lambda.sh (probe Job) and run-config.sh
# (smoke + full-run Jobs). Factored out so the Job-apply mechanics live
# in one place.
#
# Required env vars:
#   RUN_TAG                    — becomes the Job name suffix
#                                (e.g. "poisson", "poisson-smoke", "calibration")
#   HARNESS_EXPERIMENT_YAML    — path to the rendered experiment.yaml
#
# Optional env vars:
#   NAMESPACE                  — default: mrl-headtohead
#   HARNESS_IMAGE              — default: ghcr.io/<user>/mrl-harness:<sha>
#                                (falls back to the tag written to .harness-image)
#   ROUTER_SERVICE             — bare Service name (default:
#                                mrl-headtohead-hub-epp). The harness
#                                concatenates ROUTER_SERVICE:ROUTER_PORT into
#                                http://<svc>:<port> internally, so this MUST
#                                NOT include a scheme.
#   ROUTER_PORT                — Hub Service port (default: 80)
#   JOB_DEADLINE_SECONDS       — default: 21600 (6 h)

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
STUDY_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

: "${RUN_TAG:?RUN_TAG must be set (e.g. RUN_TAG=poisson)}"
: "${HARNESS_EXPERIMENT_YAML:?HARNESS_EXPERIMENT_YAML must point at a rendered experiment.yaml}"

NAMESPACE="${NAMESPACE:-mrl-headtohead}"
HARNESS_IMAGE="${HARNESS_IMAGE:-$(cat "${STUDY_DIR}/.harness-image" 2>/dev/null || echo "ghcr.io/example/mrl-harness:latest")}"
ROUTER_SERVICE="${ROUTER_SERVICE:-mrl-headtohead-hub-epp}"
ROUTER_PORT="${ROUTER_PORT:-80}"
JOB_DEADLINE_SECONDS="${JOB_DEADLINE_SECONDS:-21600}"

log()  { printf '[apply-harness-job][%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
fail() { printf '[apply-harness-job][FAIL] %s\n' "$*" >&2; exit 1; }

command -v kubectl >/dev/null 2>&1 || fail "kubectl not on PATH"
command -v envsubst >/dev/null 2>&1 || fail "envsubst not on PATH"

JOB_TMPL="${STUDY_DIR}/stack/harness/job.yaml.tmpl"
[[ -r "${JOB_TMPL}" ]] || fail "stack/harness/job.yaml.tmpl missing"
[[ -r "${HARNESS_EXPERIMENT_YAML}" ]] || fail "HARNESS_EXPERIMENT_YAML ${HARNESS_EXPERIMENT_YAML} missing"

# 1. Ensure the namespace + harness PVC + RBAC are up.
log "ensuring namespace ${NAMESPACE}"
kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

if [[ -r "${STUDY_DIR}/stack/harness/pvc.yaml" ]]; then
  kubectl apply -n "${NAMESPACE}" -f "${STUDY_DIR}/stack/harness/pvc.yaml"
fi
if [[ -r "${STUDY_DIR}/stack/harness/rbac.yaml" ]]; then
  kubectl apply -n "${NAMESPACE}" -f "${STUDY_DIR}/stack/harness/rbac.yaml"
fi

# 2. Refresh the experiment.yaml ConfigMap.
CM_NAME="harness-experiment-${RUN_TAG}"
log "applying ConfigMap ${CM_NAME}"
kubectl create configmap "${CM_NAME}" \
  --namespace "${NAMESPACE}" \
  --from-file=experiment.yaml="${HARNESS_EXPERIMENT_YAML}" \
  --from-file="${STUDY_DIR}/config/models.yaml" \
  --from-file="${STUDY_DIR}/config/datasets.yaml" \
  --from-file="${STUDY_DIR}/config/pricing.yaml" \
  --from-file="${STUDY_DIR}/stack/modelserver/vllm-deployment.yaml.tmpl" \
  --dry-run=client -o yaml | kubectl apply -f -

# 3. Delete any previous Job with the same tag (a Job is immutable).
JOB_NAME="harness-${RUN_TAG}"
kubectl delete job "${JOB_NAME}" -n "${NAMESPACE}" --ignore-not-found --wait=false >/dev/null 2>&1 || true

# 4. Render + apply the Job.
log "applying Job ${JOB_NAME}"
export RUN_TAG NAMESPACE HARNESS_IMAGE ROUTER_SERVICE ROUTER_PORT JOB_DEADLINE_SECONDS CM_NAME
envsubst < "${JOB_TMPL}" | kubectl apply -n "${NAMESPACE}" -f -

log "Job ${JOB_NAME} applied. Watch progress with:"
echo "  kubectl -n ${NAMESPACE} logs -f job/${JOB_NAME}"
