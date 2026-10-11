#!/usr/bin/env bash
#
# deploy-hub.sh — one-shot bring-up of the mrl head-to-head Hub EPP.
#
# Deploys the multicluster-aware EPP + Envoy pair to the target cluster.
# The Hub reads clusters.yaml via ConfigMap mount and picks between the
# two spokes (spoke-gemma-a TP=2, spoke-gemma-b TP=4) per request using
# whichever Distribution-axis scorer the caller's HUB_VALUES selects.
#
# The scorer choice IS the HUB_VALUES file — this same deploy script
# drives all four studies (mrl, QD, AR, LAT) via env var:
#
#     HUB_VALUES=stack/router/hub-values-mrl.yaml     scripts/deploy-hub.sh
#     HUB_VALUES=queue_depth_study/stack/router/hub-values-qd.yaml scripts/deploy-hub.sh
#     HUB_VALUES=ar_study/stack/router/hub-values-ar.yaml          scripts/deploy-hub.sh
#     HUB_VALUES=latency_observer_study/stack/router/hub-values-latency.yaml scripts/deploy-hub.sh
#
# Prereqs the caller must satisfy:
#   * Two vLLM Deployments (gemma-4-26b-a4b at TP=2 and TP=4) already
#     running in namespaces NS_SPOKE_A / NS_SPOKE_B, with istio Gateway
#     Services reachable.
#   * KUBECONFIG points at the Hub's cluster.
#   * `helm` and `kubectl` on PATH.

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
STUDY_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

NAMESPACE="${NAMESPACE:-mrl-headtohead}"
RELEASE="${RELEASE:-mrl-headtohead-hub}"
ROUTER_CHART="${ROUTER_CHART:-oci://ghcr.io/llm-d/charts/llm-d-router-standalone}"
ROUTER_VERSION="${ROUTER_VERSION:-v0.9.0}"

log()  { printf '[deploy-hub][%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
fail() { printf '[deploy-hub][FAIL] %s\n' "$*" >&2; exit 1; }

command -v helm    >/dev/null 2>&1 || fail "helm not on PATH"
command -v kubectl >/dev/null 2>&1 || fail "kubectl not on PATH"

# The HUB_VALUES path may be absolute, or relative to mrl_study/ (this
# script's parent). Sibling studies live at mrl_study/{queue_depth,ar,latency_observer}_study/,
# so a relative path like queue_depth_study/stack/router/hub-values-qd.yaml
# resolves cleanly from STUDY_DIR.
HUB_VALUES="${HUB_VALUES:-stack/router/hub-values-mrl.yaml}"
if [[ "${HUB_VALUES}" != /* ]]; then
  HUB_VALUES="${STUDY_DIR}/${HUB_VALUES}"
fi
[[ -r "${STUDY_DIR}/config/clusters.yaml"          ]] || fail "config/clusters.yaml missing"
[[ -r "${HUB_VALUES}"                              ]] || fail "HUB_VALUES ${HUB_VALUES} missing"
[[ -r "${STUDY_DIR}/stack/router/base.values.yaml" ]] || fail "stack/router/base.values.yaml missing"

log "namespace:   ${NAMESPACE}"
log "release:     ${RELEASE}"
log "HUB_VALUES:  ${HUB_VALUES}"

# Resolve per-spoke istio Gateway Service ClusterIPs and substitute
# them into clusters.yaml at ConfigMap-apply time. The Hub's Envoy
# uses ORIGINAL_DST with `use_http_header: x-gateway-destination-endpoint`,
# which requires an ip:port literal — a hostname value is rejected.
NS_SPOKE_A="${NS_SPOKE_A:-mrl-headtohead-cw-a}"
NS_SPOKE_B="${NS_SPOKE_B:-mrl-headtohead-cw-b}"
SPOKE_A_RELEASE="${SPOKE_A_RELEASE:-mrl-headtohead-spoke-a}"
SPOKE_B_RELEASE="${SPOKE_B_RELEASE:-mrl-headtohead-spoke-b}"
SPOKE_A_GATEWAY_SVC="${SPOKE_A_GATEWAY_SVC:-${SPOKE_A_RELEASE}-istio}"
SPOKE_B_GATEWAY_SVC="${SPOKE_B_GATEWAY_SVC:-${SPOKE_B_RELEASE}-istio}"
SPOKE_PORT="${SPOKE_PORT:-80}"

log "resolving spoke Gateway Service ClusterIPs"
SPOKE_A_IP="$(kubectl -n "${NS_SPOKE_A}" get svc "${SPOKE_A_GATEWAY_SVC}" -o jsonpath='{.spec.clusterIP}' 2>/dev/null || true)"
SPOKE_B_IP="$(kubectl -n "${NS_SPOKE_B}" get svc "${SPOKE_B_GATEWAY_SVC}" -o jsonpath='{.spec.clusterIP}' 2>/dev/null || true)"
[[ -n "${SPOKE_A_IP}" ]] || fail "spoke A istio Service ${SPOKE_A_GATEWAY_SVC}.${NS_SPOKE_A} not found; run 'scripts/deploy-spoke.sh a' first"
[[ -n "${SPOKE_B_IP}" ]] || fail "spoke B istio Service ${SPOKE_B_GATEWAY_SVC}.${NS_SPOKE_B} not found; run 'scripts/deploy-spoke.sh b' first"
log "  spoke A (${NS_SPOKE_A}): ${SPOKE_A_IP}"
log "  spoke B (${NS_SPOKE_B}): ${SPOKE_B_IP}"

# Render clusters.yaml with the discovered IPs. The template uses
# ${SPOKE_A_IP}/${SPOKE_B_IP} / ${SPOKE_A_PORT}/${SPOKE_B_PORT} envsubst
# placeholders — see config/clusters.yaml.
RESOLVED_YAML="$(mktemp)"
trap 'rm -f "${RESOLVED_YAML}"' EXIT
SPOKE_A_PORT="${SPOKE_PORT}" SPOKE_B_PORT="${SPOKE_PORT}" \
  SPOKE_A_IP="${SPOKE_A_IP}" SPOKE_B_IP="${SPOKE_B_IP}" \
  envsubst < "${STUDY_DIR}/config/clusters.yaml" > "${RESOLVED_YAML}"

log "applying clusters ConfigMap"
kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
kubectl create configmap mrl-headtohead-clusters \
  --namespace "${NAMESPACE}" \
  --from-file=clusters.yaml="${RESOLVED_YAML}" \
  --dry-run=client -o yaml | kubectl apply -f -

if [[ "${SKIP_HELM:-0}" == "1" ]]; then
  log "SKIP_HELM=1 — only refreshed the clusters ConfigMap."
  log "  multicluster-file-discovery watchFile picks the change up automatically."
  exit 0
fi

# If build-image.sh ran and wrote .epp-image with a specific tag,
# pass it on the helm command line via --set-string. This overrides
# the `image.tag` field in hub-values-*.yaml (which is a placeholder
# like "latest"). Without this override the helm install would try to
# pull mrl-aware-epp:latest, which doesn't exist on the registry
# (build-image.sh publishes to a sha-derived tag).
# EPP image: registry, repository, tag can all come from three sources
# (in decreasing precedence):
#   1. EPP_REGISTRY / EPP_REPOSITORY env vars set by the caller.
#   2. The <registry>/<repository>:<tag> reference in .epp-image, written by
#      scripts/build-image.sh at build time.
#   3. The values in ${HUB_VALUES} (all four files intentionally ship with
#      a placeholder registry, so precedence 1 or 2 MUST supply the real one).
#
# Rationale: the four hub-values-*.yaml files must not carry the operator's
# personal registry namespace, so they use a placeholder that will fail loud
# at pull time if no override reaches this script. External users publish
# EPP images to their own registry (via scripts/build-image.sh) and either
# export EPP_REGISTRY or rely on .epp-image being present.
EPP_IMAGE_FILE="${STUDY_DIR}/.epp-image"
HELM_IMAGE_ARGS=()
EPP_REGISTRY_RESOLVED="${EPP_REGISTRY:-}"
EPP_REPOSITORY_RESOLVED="${EPP_REPOSITORY:-}"
EPP_TAG_RESOLVED=""
if [[ -r "${EPP_IMAGE_FILE}" ]]; then
  EPP_IMAGE_REF="$(cat "${EPP_IMAGE_FILE}")"
  # Parse "<registry>/<repository>:<tag>" — <registry> may contain "/", but
  # <repository> is the last path component before ":<tag>".
  EPP_TAG_FROM_FILE="${EPP_IMAGE_REF##*:}"
  EPP_REF_WITHOUT_TAG="${EPP_IMAGE_REF%:*}"
  EPP_REPOSITORY_FROM_FILE="${EPP_REF_WITHOUT_TAG##*/}"
  EPP_REGISTRY_FROM_FILE="${EPP_REF_WITHOUT_TAG%/*}"
  if [[ -z "${EPP_TAG_FROM_FILE}" ]] || [[ "${EPP_TAG_FROM_FILE}" == "${EPP_IMAGE_REF}" ]]; then
    fail "cannot parse tag from .epp-image content: ${EPP_IMAGE_REF}"
  fi
  EPP_TAG_RESOLVED="${EPP_TAG_FROM_FILE}"
  # Env vars win over the file (allows a caller to build under one registry
  # and deploy from a re-published copy under another).
  EPP_REGISTRY_RESOLVED="${EPP_REGISTRY_RESOLVED:-${EPP_REGISTRY_FROM_FILE}}"
  EPP_REPOSITORY_RESOLVED="${EPP_REPOSITORY_RESOLVED:-${EPP_REPOSITORY_FROM_FILE}}"
  log "EPP image resolved from .epp-image: ${EPP_REGISTRY_RESOLVED}/${EPP_REPOSITORY_RESOLVED}:${EPP_TAG_RESOLVED}"
fi

if [[ -z "${EPP_REGISTRY_RESOLVED}" ]]; then
  fail "EPP image registry not resolved; either export EPP_REGISTRY, or run 'scripts/build-image.sh' first to produce .epp-image"
fi
HELM_IMAGE_ARGS+=(--set-string "router.epp.image.registry=${EPP_REGISTRY_RESOLVED}")
if [[ -n "${EPP_REPOSITORY_RESOLVED}" ]]; then
  HELM_IMAGE_ARGS+=(--set-string "router.epp.image.repository=${EPP_REPOSITORY_RESOLVED}")
fi
if [[ -n "${EPP_TAG_RESOLVED}" ]]; then
  HELM_IMAGE_ARGS+=(--set-string "router.epp.image.tag=${EPP_TAG_RESOLVED}")
fi

log "helm upgrade --install ${RELEASE}"
helm upgrade --install "${RELEASE}" "${ROUTER_CHART}" \
  --version "${ROUTER_VERSION}" \
  --namespace "${NAMESPACE}" \
  -f "${STUDY_DIR}/stack/router/base.values.yaml" \
  -f "${HUB_VALUES}" \
  "${HELM_IMAGE_ARGS[@]}"

log "waiting up to 5 min for Hub EPP rollout"
kubectl rollout status --namespace "${NAMESPACE}" \
  deployment/"${RELEASE}-epp" --timeout=300s || true

log "Hub state:"
kubectl -n "${NAMESPACE}" get deploy,svc -l "app.kubernetes.io/instance=${RELEASE}"

log "done. Verify with:"
echo "  kubectl -n ${NAMESPACE} get pods -l app.kubernetes.io/instance=${RELEASE}"
echo "  kubectl -n ${NAMESPACE} logs -l app.kubernetes.io/instance=${RELEASE} -c epp --tail=100"
