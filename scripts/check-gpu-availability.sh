#!/usr/bin/env bash
# check-gpu-availability.sh — H200 allocation preflight for the mrl studies.
#
# Refuses to proceed when the cluster does not have enough free H200 GPUs to
# host both spokes (TP=2 spoke-gemma-a + TP=4 spoke-gemma-b = 6 GPUs total).
#
# Called explicitly at the start of the one-time prep sequence (see
# docs/mrl-headtohead.md), and internally by scripts/deploy-spoke.sh as a
# preflight so the check runs even when the operator forgets to invoke it.
#
# Exit codes:
#   0   — enough GPUs free, safe to deploy.
#   42  — GPUs busy (specific code so deploy-spoke.sh can distinguish this
#         from other failures; the operator inspects and decides whether to
#         wait, evict, or move to a different cluster).
#   1   — script failure (kubectl unavailable, cluster unreachable, jq
#         missing, no matching nodes visible, etc.).
#
# No automatic remediation. This script never evicts, deletes, or drains
# anything: shared clusters may host other teams' running workloads, and a
# false conflict is far cheaper than a wrong eviction.
set -euo pipefail

REQUIRED="${REQUIRED:-6}"
RESOURCE="${RESOURCE:-nvidia.com/gpu}"
GPU_PRODUCT="${GPU_PRODUCT:-NVIDIA-H200}"

usage() {
  cat <<USAGE >&2
Usage: $0 [--required N] [--resource NAME] [--gpu-product PRODUCT]
  --required     Number of GPUs the study needs free (default: ${REQUIRED}).
  --resource     Kubernetes resource key (default: ${RESOURCE}).
  --gpu-product  nvidia.com/gpu.product label (default: ${GPU_PRODUCT}).
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --required)    REQUIRED="$2";    shift 2 ;;
    --resource)    RESOURCE="$2";    shift 2 ;;
    --gpu-product) GPU_PRODUCT="$2"; shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    *)             echo "unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

command -v kubectl >/dev/null 2>&1 || { echo "ERROR: kubectl not on PATH" >&2; exit 1; }
command -v jq      >/dev/null 2>&1 || { echo "ERROR: jq not on PATH"      >&2; exit 1; }

# The GPU resource name embeds a dot ("nvidia.com/gpu"), which jsonpath treats
# as a segment separator unless escaped. jq handles it cleanly, so use jq for
# the node walk too rather than mixing paths.
total_allocatable=$(kubectl get nodes \
    -l "nvidia.com/gpu.product=${GPU_PRODUCT}" \
    -o json \
  | jq --arg r "$RESOURCE" '
      [ .items[]
        | .status.allocatable[$r]?
        | tonumber? // 0
      ] | add // 0
    ')

if [[ "$total_allocatable" -eq 0 ]]; then
  echo "ERROR: no ${GPU_PRODUCT} nodes visible in this kube context." >&2
  echo "       Check that KUBECONFIG points at the right cluster and" >&2
  echo "       that the target cluster actually has ${GPU_PRODUCT} nodes." >&2
  exit 1
fi

# Enumerate Pods claiming this resource across all namespaces. Only Running
# and Pending Pods hold GPU claims; Succeeded/Failed have released theirs.
claiming_json=$(kubectl get pods -A -o json)

claimed_count=$(jq --arg r "$RESOURCE" '
    [ .items[]
      | select(.status.phase == "Running" or .status.phase == "Pending")
      | .spec.containers[]?.resources.requests[$r]?
      | tonumber? // 0
    ] | add // 0
  ' <<<"$claiming_json")

free=$((total_allocatable - claimed_count))

echo "GPU inventory on cluster:"
echo "  ${GPU_PRODUCT} allocatable across nodes: ${total_allocatable}"
echo "  ${GPU_PRODUCT} currently claimed by Pods: ${claimed_count}"
echo "  ${GPU_PRODUCT} free right now:            ${free}"
echo "  required for the two spokes:              ${REQUIRED}"

if [[ "$free" -lt "$REQUIRED" ]]; then
  echo "" >&2
  echo "ERROR: only ${free} ${GPU_PRODUCT} GPUs free; the two spokes need ${REQUIRED}." >&2
  echo "       (spoke-gemma-a TP=2 wants 2 GPUs, spoke-gemma-b TP=4 wants 4.)" >&2
  echo "" >&2
  echo "Pods currently claiming ${RESOURCE}:" >&2
  jq -r --arg r "$RESOURCE" '
      .items[]
      | select(.status.phase == "Running" or .status.phase == "Pending")
      | . as $pod
      | .spec.containers[]?.resources.requests[$r]? as $req
      | select($req != null and ($req | tonumber? // 0) > 0)
      | "  \($pod.metadata.namespace)/\($pod.metadata.name) requests=\($req)"
    ' <<<"$claiming_json" >&2
  echo "" >&2
  echo "Halting before spoke deploy. Inspect the claiming pods with:" >&2
  echo "  kubectl describe pod -n <namespace> <name>" >&2
  echo "and decide whether to wait, coordinate with the owning team, or" >&2
  echo "point KUBECONFIG at a different cluster." >&2
  exit 42
fi

echo "OK: ${free} ${GPU_PRODUCT} GPUs free, deployment can proceed."
