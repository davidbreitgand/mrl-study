#!/usr/bin/env bash
#
# deploy-spoke.sh — one-shot bring-up of a spoke EPP + vLLM Deployment.
#
# One invocation deploys ONE spoke: `deploy-spoke.sh a` deploys
# spoke-gemma-a (TP=2) in namespace ${NS_SPOKE_A}; `deploy-spoke.sh b`
# deploys spoke-gemma-b (TP=4) in ${NS_SPOKE_B}. Runbook calls this
# twice, once per letter.
#
# Preflight: refuses to deploy if the target cluster has fewer than 6
# NVIDIA-H200 GPUs free across all namespaces. The check itself lives in
# scripts/check-gpu-availability.sh; the study is halted with exit 42 on
# a conflict so the operator can inspect the claiming pods.
#
# TP asymmetry: spoke-a runs at models.yaml's tensor_parallel_size (=2),
# spoke-b overrides to TP=4 via TP_OVERRIDE. The vLLM Deployment template
# consumes this override at render time.

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
STUDY_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

usage() {
  cat <<'USAGE' >&2
Usage:
  scripts/deploy-spoke.sh a   # spoke-gemma-a, TP=2, namespace NS_SPOKE_A
  scripts/deploy-spoke.sh b   # spoke-gemma-b, TP=4, namespace NS_SPOKE_B

Environment (all overridable):
  NS_SPOKE_A            (default: mrl-headtohead-cw-a)
  NS_SPOKE_B            (default: mrl-headtohead-cw-b)
  SPOKE_A_RELEASE       (default: mrl-headtohead-spoke-a)
  SPOKE_B_RELEASE       (default: mrl-headtohead-spoke-b)
  ROUTER_CHART          (default: oci://ghcr.io/llm-d/charts/llm-d-router-standalone)
  ROUTER_VERSION        (default: v0.9.0)
  SKIP_GPU_PREFLIGHT=1  (skip check-gpu-availability.sh; NOT RECOMMENDED)
USAGE
}

[[ $# -eq 1 ]] || { usage; exit 1; }

SPOKE_LETTER="$1"
case "${SPOKE_LETTER}" in
  a|A)
    NAMESPACE="${NS_SPOKE_A:-mrl-headtohead-cw-a}"
    RELEASE="${SPOKE_A_RELEASE:-mrl-headtohead-spoke-a}"
    TP_OVERRIDE="${TP_OVERRIDE:-2}"
    ;;
  b|B)
    NAMESPACE="${NS_SPOKE_B:-mrl-headtohead-cw-b}"
    RELEASE="${SPOKE_B_RELEASE:-mrl-headtohead-spoke-b}"
    TP_OVERRIDE="${TP_OVERRIDE:-4}"
    ;;
  *) usage; exit 1 ;;
esac

ROUTER_CHART="${ROUTER_CHART:-oci://ghcr.io/llm-d/charts/llm-d-router-standalone}"
ROUTER_VERSION="${ROUTER_VERSION:-v0.9.0}"

log()  { printf '[deploy-spoke][%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
fail() { printf '[deploy-spoke][FAIL] %s\n' "$*" >&2; exit 1; }

command -v helm    >/dev/null 2>&1 || fail "helm not on PATH"
command -v kubectl >/dev/null 2>&1 || fail "kubectl not on PATH"
[[ -r "${STUDY_DIR}/stack/router/spoke-values.yaml" ]] || fail "stack/router/spoke-values.yaml missing"
[[ -r "${STUDY_DIR}/stack/router/base.values.yaml"  ]] || fail "stack/router/base.values.yaml missing"
[[ -r "${STUDY_DIR}/stack/modelserver/vllm-deployment.yaml.tmpl" ]] || fail "vllm-deployment template missing"

# 0. GPU-availability preflight. Refuse to deploy when the cluster
#    doesn't have enough free H200s. This runs whether the operator
#    called check-gpu-availability.sh explicitly or not.
if [[ "${SKIP_GPU_PREFLIGHT:-0}" != "1" ]]; then
  log "preflight: checking H200 availability"
  "${SCRIPT_DIR}/check-gpu-availability.sh" --required 6
fi

log "spoke:       ${SPOKE_LETTER}"
log "namespace:   ${NAMESPACE}"
log "release:     ${RELEASE}"
log "TP override: ${TP_OVERRIDE}"

# 1. Create namespace.
kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

# 2. Render + apply the vLLM Deployment for this spoke.
#    vllm-deployment.yaml.tmpl uses Jinja2 ({% for %} loops + {{ var }}
#    substitutions), so we use Jinja2 to render it (matching the
#    costguard study's stack/scripts/render_manifest.py approach).
log "rendering vLLM Deployment (TP=${TP_OVERRIDE})"
RESOLVED_VLLM="$(mktemp)"
trap 'rm -f "${RESOLVED_VLLM}"' EXIT
python3 - <<PYEOF > "${RESOLVED_VLLM}"
import os, sys
try:
    import jinja2
except ImportError:
    sys.stderr.write("ERROR: jinja2 required; install with 'pip install jinja2'\n")
    sys.exit(1)
tmpl_str = open("${STUDY_DIR}/stack/modelserver/vllm-deployment.yaml.tmpl").read()
subs = {
    "namespace":              "${NAMESPACE}",
    "model_id":               "gemma-4-26b-a4b",
    "hf_repo":                "google/gemma-4-26B-A4B-it",
    "served_model_name":      "google/gemma-4-26B-A4B-it",
    "tensor_parallel_size":   ${TP_OVERRIDE},
    "max_model_len":          69632,
    "gpu_memory_utilization": 0.9,
    "reasoning_parser":       "gemma4",
    # Jinja loop expands each element into a separate --arg line.
    "extra_vllm_args":        ["--enable-chunked-prefill", "--max-num-seqs=8"],
    "hf_cache_size_gi":       200,
    "hf_token_secret":        os.environ.get("HF_TOKEN_SECRET", "hf-token"),
    "cpu_limit":              "32",
    "cpu_request":            "8",
    "memory_limit_gi":        128,
    "memory_request_gi":      32,
}
env = jinja2.Environment(
    undefined=jinja2.StrictUndefined,
    keep_trailing_newline=True,
)
print(env.from_string(tmpl_str).render(**subs))
PYEOF
kubectl apply -n "${NAMESPACE}" -f "${RESOLVED_VLLM}"

# 3. Helm install / upgrade the spoke EPP.
log "helm upgrade --install ${RELEASE}"
helm upgrade --install "${RELEASE}" "${ROUTER_CHART}" \
  --version "${ROUTER_VERSION}" \
  --namespace "${NAMESPACE}" \
  -f "${STUDY_DIR}/stack/router/base.values.yaml" \
  -f "${STUDY_DIR}/stack/router/spoke-values.yaml"

log "waiting up to 90 min for vLLM ready (model download + GPU warmup)"
kubectl rollout status --namespace "${NAMESPACE}" \
  deployment/pricing-reversal-vllm-gemma-4-26b-a4b --timeout=5400s || fail "vLLM Deployment did not become ready in 90 min"

log "waiting up to 5 min for spoke EPP rollout"
kubectl rollout status --namespace "${NAMESPACE}" \
  deployment/"${RELEASE}-epp" --timeout=300s || true

# 4. Render + apply the spoke Gateway/HTTPRoute for Hub reachability.
GATEWAY_TMPL="${STUDY_DIR}/stack/router/spoke-gateway.yaml.tmpl"
if [[ -r "${GATEWAY_TMPL}" ]]; then
  log "rendering + applying spoke Gateway/HTTPRoute"
  python3 - "${GATEWAY_TMPL}" "${NAMESPACE}" "${RELEASE}" <<'PYEOF' | kubectl apply -n "${NAMESPACE}" -f -
import sys
tmpl_path, namespace, release = sys.argv[1:]
with open(tmpl_path) as f:
    text = f.read()
text = text.replace("{{ namespace }}", namespace)
text = text.replace("{{ gateway_name }}", release)
text = text.replace("{{ pool_name }}", release)
sys.stdout.write(text)
PYEOF
  log "waiting for spoke gateway Programmed=True"
  kubectl -n "${NAMESPACE}" wait --for=condition=Programmed --timeout=120s \
    "gateway.gateway.networking.k8s.io/${RELEASE}" || true
else
  log "spoke-gateway.yaml.tmpl not found; skipping Gateway apply"
fi

log "spoke state:"
kubectl -n "${NAMESPACE}" get deploy,svc,gateway,httproute -l "llm-d.ai/guide=mrl-headtohead" 2>/dev/null || \
kubectl -n "${NAMESPACE}" get deploy,svc

log "done. Spoke gateway ClusterIP address for the Hub's clusters.yaml:"
echo "  $(kubectl -n "${NAMESPACE}" get svc "${RELEASE}-istio" -o jsonpath='{.spec.clusterIP}' 2>/dev/null):80"
