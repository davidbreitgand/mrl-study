#!/usr/bin/env bash
#
# hub-smoke.sh — 3-curl sanity check that the Hub is reachable and its
# scorer emits the served-endpoint header.
#
# Runs once per Hub redeploy, before the (arrival regime) cells for that
# scorer. Ports the Hub Service through kubectl port-forward locally,
# issues three POST /v1/chat/completions, and verifies:
#
#   1. HTTP 200 on all three.
#   2. `x-conformance-test-served-endpoint` response header set on each,
#      carrying `spoke-gemma-a` or `spoke-gemma-b`.
#
# Exits non-zero with a diagnostic on any failure; the caller should
# refuse to launch cell runs when this smoke fails.
#
# Usage:
#   scripts/hub-smoke.sh                    # defaults
#   NAMESPACE=foo SERVICE=bar scripts/hub-smoke.sh

set -o errexit
set -o nounset
set -o pipefail

NAMESPACE="${NAMESPACE:-mrl-headtohead}"
SERVICE="${SERVICE:-mrl-headtohead-hub-epp}"
LOCAL_PORT="${LOCAL_PORT:-18080}"
REMOTE_PORT="${REMOTE_PORT:-80}"
N_REQUESTS="${N_REQUESTS:-3}"
MODEL="${MODEL:-google/gemma-4-26B-A4B-it}"
PROMPT="${PROMPT:-hi}"
MAX_TOKENS="${MAX_TOKENS:-3}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-30}"

log()  { printf '[hub-smoke][%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
fail() { printf '[hub-smoke][FAIL] %s\n' "$*" >&2; exit 1; }

command -v kubectl >/dev/null 2>&1 || fail "kubectl not on PATH"
command -v curl    >/dev/null 2>&1 || fail "curl not on PATH"

# Free the local port defensively (a prior port-forward may have leaked).
lsof -i ":${LOCAL_PORT}" -sTCP:LISTEN -t 2>/dev/null | xargs -r kill -9 2>/dev/null || true

log "port-forwarding ${SERVICE}.${NAMESPACE}:${REMOTE_PORT} to localhost:${LOCAL_PORT}"
PF_LOG="$(mktemp)"
kubectl -n "${NAMESPACE}" port-forward "svc/${SERVICE}" "${LOCAL_PORT}:${REMOTE_PORT}" >"${PF_LOG}" 2>&1 &
PF_PID=$!
trap 'kill "${PF_PID}" 2>/dev/null; wait "${PF_PID}" 2>/dev/null; rm -f "${PF_LOG}"' EXIT
sleep 3

if ! kill -0 "${PF_PID}" 2>/dev/null; then
  cat "${PF_LOG}" >&2
  fail "port-forward exited early"
fi

log "issuing ${N_REQUESTS} smoke requests"
declare -a ENDPOINTS_SEEN=()
for i in $(seq 1 "${N_REQUESTS}"); do
  HEADERS_TMP="$(mktemp)"
  body=$(curl -sS --max-time "${REQUEST_TIMEOUT_SECONDS}" \
    -D "${HEADERS_TMP}" \
    -X POST "http://127.0.0.1:${LOCAL_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"max_tokens\":${MAX_TOKENS}}" \
    2>&1) || { rm -f "${HEADERS_TMP}"; fail "request ${i} transport error: ${body}"; }

  status=$(head -n 1 "${HEADERS_TMP}" | awk '{print $2}')
  served=$(grep -i '^x-conformance-test-served-endpoint:' "${HEADERS_TMP}" | tr -d '\r' | sed 's/^[^:]*: *//')
  rm -f "${HEADERS_TMP}"

  [[ "${status}" == "200" ]] || fail "request ${i} status=${status:-<none>} (expected 200) body=${body:0:200}"
  [[ -n "${served}" ]] || fail "request ${i} missing x-conformance-test-served-endpoint header"

  log "  request ${i}: HTTP 200, served-endpoint=${served}"
  ENDPOINTS_SEEN+=("${served}")
done

log "smoke passed: ${N_REQUESTS} × HTTP 200, served-endpoint header on each"
log "  endpoints observed: ${ENDPOINTS_SEEN[*]}"
