#!/usr/bin/env bash
#
# build-harness-image.sh — build + push the mrl_study harness image.
#
# Produces ghcr.io/${OWNER}/mrl-harness:<sha> from mrl_study/harness/'s
# Dockerfile (the canonical Weibull-supporting harness fork). On success
# writes .harness-image at the mrl_study root; apply-harness-job.sh
# reads that file to know which image the Job manifest should reference.
#
# The <sha> is derived from the pricing-reversal repo's HEAD (since the
# harness fork lives inside pricing-reversal, its own state is what
# determines whether the built image is meaningful). If the working tree
# is dirty, the tag gets a `-dirty` suffix.
#
# Overridable env:
#   OWNER               ghcr.io/<OWNER>/mrl-harness (default: $USER)
#   IMAGE_REGISTRY      default ghcr.io/${OWNER}
#   HARNESS_IMAGE_NAME  default mrl-harness
#   PUSH                default "1" (set to "0" to skip `docker push`)
#   CONTAINER_RUNTIME   default docker

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
STUDY_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

OWNER="${OWNER:-${USER}}"
IMAGE_REGISTRY="${IMAGE_REGISTRY:-ghcr.io/${OWNER}}"
HARNESS_IMAGE_NAME="${HARNESS_IMAGE_NAME:-mrl-harness}"
PUSH="${PUSH:-1}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-docker}"

log()  { printf '[build-harness-image][%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
fail() { printf '[build-harness-image][FAIL] %s\n' "$*" >&2; exit 1; }

command -v git                 >/dev/null 2>&1 || fail "git not on PATH"
command -v "${CONTAINER_RUNTIME}" >/dev/null 2>&1 || fail "${CONTAINER_RUNTIME} not on PATH"

HARNESS_DIR="${STUDY_DIR}/harness"
[[ -f "${HARNESS_DIR}/Dockerfile" ]] || fail "harness/Dockerfile missing at ${HARNESS_DIR}"

# Derive a tag from the enclosing git repo's HEAD when possible; when
# the harness is unpacked outside a git checkout (e.g. from a release
# tarball), fall back to a UTC timestamp tag so the build still succeeds.
if PR_ROOT="$(git -C "${STUDY_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
  SHA="$(git -C "${PR_ROOT}" rev-parse --short=12 HEAD)"
  if ! git -C "${PR_ROOT}" diff --quiet HEAD -- "${HARNESS_DIR}"; then
    SHA="${SHA}-dirty"
    log "WARN: harness/ has uncommitted changes; tagging as ${SHA}"
  fi
else
  SHA="notgit-$(date -u +%Y%m%d%H%M%S)"
  log "WARN: not inside a git checkout; tagging as ${SHA}"
fi

HARNESS_IMAGE="${IMAGE_REGISTRY}/${HARNESS_IMAGE_NAME}:${SHA}"
log "building ${HARNESS_IMAGE}"
log "  source:   ${HARNESS_DIR} @ pricing-reversal ${SHA}"

(
  cd "${HARNESS_DIR}"
  "${CONTAINER_RUNTIME}" build -t "${HARNESS_IMAGE}" .
)

if [[ "${PUSH}" == "1" ]]; then
  log "pushing ${HARNESS_IMAGE}"
  "${CONTAINER_RUNTIME}" push "${HARNESS_IMAGE}"
fi

echo "${HARNESS_IMAGE}" > "${STUDY_DIR}/.harness-image"
log "wrote ${STUDY_DIR}/.harness-image = ${HARNESS_IMAGE}"

cat <<EOF

Next step: no manifest edit required. apply-harness-job.sh reads
${STUDY_DIR}/.harness-image at Job-render time, so subsequent
calibrate-lambda.sh / run-config.sh invocations pick this tag up
automatically.
EOF
