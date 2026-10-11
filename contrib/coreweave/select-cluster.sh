#!/usr/bin/env bash
#
# select-cluster.sh — pick the first CoreWeave cluster with enough
# capacity to host the mrl_study. Cascade order:
#
#   piggy → fozzie → kermit
#
# For each candidate:
#   * Kubeconfig must be readable at the expected path.
#   * The cluster must respond to `kubectl get nodes`.
#   * shared-vast StorageClass must exist.
#   * InferencePool CRD (inferencepools.inference.networking.k8s.io)
#     must exist.
#   * Free NVIDIA-H200 GPUs (allocatable − requested by Running+Pending
#     pods across all namespaces) must be ≥ the required count for the
#     study (default: 6 = TP=2 for spoke-a + TP=4 for spoke-b).
#
# On success prints two lines to STDOUT:
#     KUBECONFIG=<path>
#     CLUSTER=<name>
# suitable for `eval "$(scripts/select-cluster.sh)"` or `. <(scripts/...)`.
#
# On failure of ALL candidates, exits 1 and prints a per-candidate
# rejection reason to STDERR.
#
# Overridable via env:
#   KUBECONFIG_PIGGY  — default $HOME/Downloads/CWKubeconfig_piggy_US-EAST-01A
#   KUBECONFIG_FOZZIE — default $HOME/Downloads/CWKubeconfig_fozzie_US-EAST-01A
#   KUBECONFIG_KERMIT — default $HOME/Downloads/CWKubeconfig_kermit_US-EAST-01A
#   MRL_MIN_FREE_GPUS — default 6

set -o errexit
set -o nounset
set -o pipefail

KUBECONFIG_PIGGY="${KUBECONFIG_PIGGY:-$HOME/Downloads/CWKubeconfig_piggy_US-EAST-01A}"
KUBECONFIG_FOZZIE="${KUBECONFIG_FOZZIE:-$HOME/Downloads/CWKubeconfig_fozzie_US-EAST-01A}"
KUBECONFIG_KERMIT="${KUBECONFIG_KERMIT:-$HOME/Downloads/CWKubeconfig_kermit_US-EAST-01A}"
MRL_MIN_FREE_GPUS="${MRL_MIN_FREE_GPUS:-6}"

log()  { printf '[select-cluster] %s\n' "$*" >&2; }
warn() { printf '[select-cluster][skip] %s\n' "$*" >&2; }
fail() { printf '[select-cluster][FAIL] %s\n' "$*" >&2; exit 1; }

# Returns 0 (this cluster is chosen) if the candidate passes all checks;
# otherwise returns non-zero and prints the skip reason to STDERR.
try_cluster() {
  local name="$1"
  local kc="$2"

  log "trying ${name} (kubeconfig=${kc})"

  [[ -r "${kc}" ]] \
    || { warn "${name}: kubeconfig not readable at ${kc}"; return 1; }

  # Reachability probe: `kubectl get --raw /healthz` is cheap.
  if ! KUBECONFIG="${kc}" kubectl get --raw /healthz >/dev/null 2>&1; then
    warn "${name}: cluster unreachable via /healthz"
    return 1
  fi

  # shared-vast StorageClass (used by every PVC in stack/).
  if ! KUBECONFIG="${kc}" kubectl get storageclass shared-vast >/dev/null 2>&1; then
    warn "${name}: shared-vast StorageClass missing"
    return 1
  fi

  # InferencePool CRD (required by the router chart).
  if ! KUBECONFIG="${kc}" kubectl get crd inferencepools.inference.networking.k8s.io >/dev/null 2>&1; then
    warn "${name}: InferencePool CRD missing"
    return 1
  fi

  # GPU capacity — allocatable minus what's currently reserved by
  # Running+Pending pods (Succeeded/Failed pods release their slots).
  local alloc req free
  alloc="$(KUBECONFIG="${kc}" kubectl get nodes -l nvidia.com/gpu.product=NVIDIA-H200 \
    -o jsonpath='{range .items[*]}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' 2>/dev/null \
    | awk 'BEGIN{s=0}/^[0-9]+$/{s+=$1}END{print s}')"
  req="$( { KUBECONFIG="${kc}" kubectl get pods --all-namespaces --field-selector=status.phase=Running \
              -o jsonpath='{range .items[*]}{range .spec.containers[*]}{.resources.requests.nvidia\.com/gpu}{"\n"}{end}{end}' 2>/dev/null;
            KUBECONFIG="${kc}" kubectl get pods --all-namespaces --field-selector=status.phase=Pending \
              -o jsonpath='{range .items[*]}{range .spec.containers[*]}{.resources.requests.nvidia\.com/gpu}{"\n"}{end}{end}' 2>/dev/null;
          } | awk 'BEGIN{s=0}/^[0-9]+$/{s+=$1}END{print s}')"
  free=$(( alloc - req ))

  if (( free < MRL_MIN_FREE_GPUS )); then
    warn "${name}: insufficient GPUs (allocatable=${alloc} requested=${req} free=${free} < required=${MRL_MIN_FREE_GPUS})"
    return 1
  fi

  log "${name}: OK (allocatable=${alloc} requested=${req} free=${free} required=${MRL_MIN_FREE_GPUS})"
  echo "KUBECONFIG=${kc}"
  echo "CLUSTER=${name}"
  return 0
}

# Cascade in priority order. First hit prints, exits 0.
try_cluster piggy  "${KUBECONFIG_PIGGY}"  && exit 0 || true
try_cluster fozzie "${KUBECONFIG_FOZZIE}" && exit 0 || true
try_cluster kermit "${KUBECONFIG_KERMIT}" && exit 0 || true

fail "no CoreWeave cluster met the capacity requirements. Rerun once one clears, or override KUBECONFIG_* / MRL_MIN_FREE_GPUS."
