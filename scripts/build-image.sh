#!/usr/bin/env bash
#
# build-image.sh — build + push the EPP image from llm-d-router feature/mrl-aware.
#
# For reproducibility of the head-to-head results, build from the exact
# commit that landed with the accompanying pull request:
#
#     git -C "${ROUTER_DIR}" fetch origin
#     git -C "${ROUTER_DIR}" checkout <TODO: PR commit SHA — updated after PR opens; check the tracking Issue for the current value>
#     scripts/build-image.sh
#
# Reviewers verifying the study should check out that specific commit
# rather than the tip of the branch, so their build sees the same
# scorer implementation the head-to-head measured. Once the PR merges
# to main, the recorded SHA becomes the merge commit's parent.
#
# Produces <IMAGE_REGISTRY>/<EPP_IMAGE_NAME>:<git-sha> and, on success,
# writes the fully-qualified image reference to .epp-image at the
# mrl_study root. scripts/deploy-hub.sh reads that file and passes it
# to helm via --set-string, so no manual hub-values edits are required.
#
# Requires:
#   - llm-d-router checkout at ${ROUTER_DIR} (default: ../../llm-d-router,
#     or override with ROUTER_DIR env).
#   - The checkout is on feature/mrl-aware; this script REFUSES to build
#     an unrelated branch (avoid accidentally publishing the wrong scorer).
#     Set ALLOW_ANY_BRANCH=1 to override.
#   - docker or podman on PATH.
#   - Write access to <IMAGE_REGISTRY> for the push step (set PUSH=0 to
#     build without pushing, useful for local testing).
#
# Overridable env:
#   OWNER               ghcr.io/<OWNER>/mrl-aware-epp (default: $USER)
#   ROUTER_DIR          path to llm-d-router checkout
#   IMAGE_REGISTRY      default ghcr.io/${OWNER}
#   EPP_IMAGE_NAME      default mrl-aware-epp
#   PUSH                default "1" (set to "0" to skip `docker push`)

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
STUDY_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

ROUTER_DIR="${ROUTER_DIR:-${STUDY_DIR}/../../llm-d-router}"
OWNER="${OWNER:-${USER}}"
IMAGE_REGISTRY="${IMAGE_REGISTRY:-ghcr.io/${OWNER}}"
EPP_IMAGE_NAME="${EPP_IMAGE_NAME:-mrl-aware-epp}"
PUSH="${PUSH:-1}"

log()  { printf '[build-image][%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
fail() { printf '[build-image][FAIL] %s\n' "$*" >&2; exit 1; }

command -v git >/dev/null 2>&1 || fail "git not on PATH"

[[ -d "${ROUTER_DIR}/.git" ]] || fail "ROUTER_DIR ${ROUTER_DIR} is not a git checkout"

# Refuse to build a branch other than feature/mrl-aware. The one exception
# is when an explicit ALLOW_ANY_BRANCH=1 is passed (for CI or a
# maintainer testing a variant); this prints a loud warning either way.
BRANCH="$(git -C "${ROUTER_DIR}" rev-parse --abbrev-ref HEAD)"
if [[ "${BRANCH}" != "feature/mrl-aware" ]]; then
  if [[ "${ALLOW_ANY_BRANCH:-0}" != "1" ]]; then
    fail "expected llm-d-router on feature/mrl-aware, got ${BRANCH}. \
Rerun with ALLOW_ANY_BRANCH=1 to override."
  fi
  log "WARN: building from ${BRANCH}, not feature/mrl-aware — override active"
fi

SHA="$(git -C "${ROUTER_DIR}" rev-parse --short=12 HEAD)"
if ! git -C "${ROUTER_DIR}" diff --quiet HEAD; then
  SHA="${SHA}-dirty"
  log "WARN: working tree has uncommitted changes; tagging as ${SHA}"
fi

EPP_IMAGE="${IMAGE_REGISTRY}/${EPP_IMAGE_NAME}:${SHA}"
log "building ${EPP_IMAGE}"
log "  source:   ${ROUTER_DIR} @ ${BRANCH} ${SHA}"

# The router's Makefile drives image-build-epp inside a builder container
# (Dockerfile.builder). This handles cross-platform builds and produces the
# right binary layout.
(
  cd "${ROUTER_DIR}"
  EPP_IMAGE="${EPP_IMAGE}" make image-build-epp
)

if [[ "${PUSH}" == "1" ]]; then
  log "pushing ${EPP_IMAGE}"
  (
    cd "${ROUTER_DIR}"
    EPP_IMAGE="${EPP_IMAGE}" make image-push-epp 2>/dev/null || \
      docker push "${EPP_IMAGE}"
  )
fi

# Record the image ref for downstream deploy scripts.
echo "${EPP_IMAGE}" > "${STUDY_DIR}/.epp-image"
log "wrote ${STUDY_DIR}/.epp-image = ${EPP_IMAGE}"

# scripts/deploy-hub.sh reads .epp-image and passes the registry,
# repository, and tag to helm via --set-string, so no manual edits to
# hub-values-*.yaml are required. Next step is simply:
#
#   HUB_VALUES=stack/router/hub-values-<scorer>.yaml scripts/deploy-hub.sh
