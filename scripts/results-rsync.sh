#!/usr/bin/env bash
# results-rsync.sh — pull one study's outputs from the harness's PVC to
# a per-study folder under ./local_results/.
#
# Layout produced:
#   local_results/
#     <YYYY-MM-DDTHH-MM-SSZ>/       ← one folder per study
#       uber/…
#       workers/<model>/…
#       consolidated/…
#       repeated_trial/…
#       _manifest.json
#     latest -> <newest-experiment>/   ← always the newest study pulled
#
# The PVC layout is /results/<study-id>/… (see UberController — the id
# is stamped at study_start and every path underneath is scoped to it).
# `dataset_cache/` and `resume-from/` live at /results/ top level and
# are NOT part of the retrieval.
#
# By default this script pulls the newest study on the PVC. Pass an
# explicit study id as the third arg to pull a specific one — this is
# how you retrieve a study that isn't the latest anymore.
#
# The retrieval is idempotent: if the target folder already exists
# (partial pull, or a second pull of the same study after more runs)
# the extracted snapshot is merged into it.
#
# Prefers a throwaway helper Pod that mounts the PVC and snapshots
# inside itself with a tolerant `tar`, then streams the frozen tarball
# out via chunked `kubectl exec cat` with per-chunk sha256 verification and
# retry. Racing the live uber Pod's own writes drops transfers.
#
# Usage: results-rsync.sh [<namespace>] [<dest>] [<study-id>]
set -euo pipefail

NAMESPACE="${1:-mrl-headtohead}"
DEST="${2:-local_results}"
STUDY_ID="${3:-}"

log() { echo "[results-rsync] $*" >&2; }

# Turn "2026-08-25T14:59:23.548Z" or "2026-08-25T14:59:23Z" into
# "2026-08-25T14-59-23Z" (colons and fractional seconds stripped).
# Prints the derived string on stdout, or nothing if the input is not
# recognizable ISO-8601.
derive_folder_from_ts() {
  local ts="$1"
  # Python is guaranteed on the laptop side (used by verify_manifest).
  python3 - <<PY 2>/dev/null || true
import re, sys
ts = ${ts@Q}
m = re.match(r'^(\d{4}-\d{2}-\d{2}T\d{2}):(\d{2}):(\d{2})(?:\.\d+)?Z$', ts)
if not m:
    sys.exit(0)
print(f"{m.group(1)}-{m.group(2)}-{m.group(3)}Z")
PY
}

# Read the first non-empty JSON line of the uber's progress.jsonl in
# DIR (where DIR is the extracted study tarball root, so its uber log
# lives at DIR/uber/progress.jsonl) and return the derived folder name
# — or empty if not available. Kept as a fallback for the rare case
# where the caller didn't specify a study id and the PVC-side listing
# also didn't yield one (pre-migration tarballs where the root itself
# was /results).
derive_from_progress() {
  local dir="$1"
  local prog="${dir}/uber/progress.jsonl"
  [[ -f "${prog}" ]] || prog="${dir}/progress.jsonl"
  [[ -f "${prog}" ]] || return 0
  local first_ts
  first_ts="$(python3 - <<PY 2>/dev/null || true
import json, sys, pathlib
p = pathlib.Path(${prog@Q})
try:
    with p.open() as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = d.get("ts")
            if ts:
                print(ts); break
except FileNotFoundError:
    pass
PY
  )"
  [[ -z "${first_ts}" ]] && return 0
  derive_folder_from_ts "${first_ts}"
}

# Fallback timestamp = the moment we're retrieving now (UTC).
retrieval_timestamp() {
  date -u '+%Y-%m-%dT%H-%M-%SZ'
}

pick_pvc_for_pod() {
  # Which PVC does this pod mount at /results? Read from the pod's spec.
  kubectl get pod "$1" -n "${NAMESPACE}" \
    -o jsonpath='{.spec.volumes[?(@.name=="results")].persistentVolumeClaim.claimName}' 2>/dev/null
}

pod_phase() {
  kubectl get pod "$1" -n "${NAMESPACE}" -o jsonpath='{.status.phase}' 2>/dev/null
}

# ---- 1. Locate the results PVC ------------------------------------------
# We ALWAYS spawn a fresh helper Pod, even when the uber Pod is Running,
# because rsync'ing from the uber Pod races against its own writes:
# `tar: file changed as we read it → exit 1`, dropping the whole transfer.
# The helper mounts the same PVC (RWX), snapshots inside itself with a
# tolerant `tar`, and streams the frozen tarball out via `kubectl cp`.

PVC=""
for LABEL in "app.kubernetes.io/component=uber" \
             "app.kubernetes.io/component=harness"; do
  CAND="$(kubectl get pods -n "${NAMESPACE}" -l "${LABEL}" \
          -o jsonpath='{.items[-1:].metadata.name}' 2>/dev/null || true)"
  [[ -z "${CAND}" ]] && continue
  PVC="$(pick_pvc_for_pod "${CAND}")"
  [[ -n "${PVC}" ]] && break
done
if [[ -z "${PVC}" ]]; then
  for CAND_PVC in mrl-headtohead-results pricing-reversal-agg pricing-reversal-results; do
    if kubectl get pvc "${CAND_PVC}" -n "${NAMESPACE}" >/dev/null 2>&1; then
      PVC="${CAND_PVC}"; break
    fi
  done
fi
if [[ -z "${PVC}" ]]; then
  log "no results PVC found (mrl-headtohead-results, pricing-reversal-agg, pricing-reversal-results)"
  exit 1
fi

# ---- 2. Spawn a helper Pod on that PVC ----------------------------------
# The helper mounts the PVC read-only, so we also use it to resolve the
# study id (if the caller didn't pass one) before the snapshot step.
HELPER_POD="results-rsync-helper-$$"
cleanup_helper() {
  kubectl delete pod "${HELPER_POD}" -n "${NAMESPACE}" \
    --ignore-not-found --wait=false >/dev/null 2>&1 || true
}
trap cleanup_helper EXIT

log "spawning helper pod/${HELPER_POD} on PVC ${PVC}"
# Full ubi9 image (not ubi-minimal) — ships `tar`. The helper reads
# /results from the PVC (RWX-mounted alongside the uber Pod) and writes
# its snapshot tarball to an emptyDir at /snapshot so it doesn't
# consume PVC space.
cat <<EOF | kubectl apply -n "${NAMESPACE}" -f - >/dev/null
apiVersion: v1
kind: Pod
metadata:
  name: ${HELPER_POD}
  labels: {llm-d.ai/guide: mrl-headtohead, app.kubernetes.io/component: study-utility}
spec:
  restartPolicy: Never
  containers:
    - name: helper
      image: registry.access.redhat.com/ubi9/ubi:latest
      command: ["/bin/sh", "-c", "sleep 3600"]
      volumeMounts:
        - {name: results, mountPath: /results, readOnly: true}
        - {name: snapshot, mountPath: /snapshot}
  volumes:
    - name: results
      persistentVolumeClaim: {claimName: ${PVC}}
    - name: snapshot
      emptyDir: {sizeLimit: 5Gi}
EOF

# Wait for Running (short deadline; PVC binding + image pull only).
for _ in $(seq 1 60); do
  if [[ "$(pod_phase "${HELPER_POD}")" == "Running" ]]; then break; fi
  sleep 2
done
if [[ "$(pod_phase "${HELPER_POD}")" != "Running" ]]; then
  log "helper pod failed to start"
  kubectl describe pod "${HELPER_POD}" -n "${NAMESPACE}" 2>&1 | awk '/^Events:/,0' | tail -20 >&2
  exit 1
fi

# ---- 3. Resolve which study to pull -------------------------------------
# If the caller passed a specific STUDY_ID, use it. Otherwise ask the
# helper to list /results/ and pick the newest YYYY-MM-DD…Z entry,
# ignoring the two top-level shared dirs (dataset_cache, resume-from)
# and any lingering `.tmp*` scratch. That's what makes multi-study
# accumulation retrievable — the operator doesn't need to remember an
# id to pull the latest.
if [[ -z "${STUDY_ID}" ]]; then
  STUDY_ID="$(kubectl exec -n "${NAMESPACE}" "${HELPER_POD}" -- sh -c "
    set -eu
    ls -1 /results 2>/dev/null \
      | grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}-[0-9]{2}-[0-9]{2}Z$' \
      | sort \
      | tail -n 1
  " 2>/dev/null || true)"
  if [[ -z "${STUDY_ID}" ]]; then
    log "no study directories found under /results/ on PVC ${PVC}"
    log "(expected YYYY-MM-DDTHH-MM-SSZ subdirs — is the study actually running?)"
    exit 1
  fi
  log "picked latest study on PVC: ${STUDY_ID}"
else
  # Verify the caller-provided id actually exists on the PVC.
  if ! kubectl exec -n "${NAMESPACE}" "${HELPER_POD}" -- \
         test -d "/results/${STUDY_ID}" 2>/dev/null; then
    log "study ${STUDY_ID} not found under /results/ on PVC ${PVC}"
    exit 1
  fi
  log "pulling study: ${STUDY_ID}"
fi

# ---- 4. Snapshot /results/<study-id> inside the helper ------------------
# `--warning=no-file-changed` + `--ignore-failed-read` turn the mid-write
# and mid-delete races into warnings (on stderr, still visible if you
# look) instead of a non-zero exit code. Result: a stable
# /snapshot/results.tar that we can copy out.
#
# tar's -C is the study dir itself, so the tarball root maps 1:1 onto
# local_results/<study-id>/ — no in-tarball path munging needed.
#
# Run tar as its own command (not a pipe) so bash's `if !` reads tar's
# own exit code. An earlier attempt piped tar to a grep filter — that
# made bash observe the grep exit code, which is 1 when grep filters out
# ALL input lines, incorrectly failing the check.
log "snapshotting /results/${STUDY_ID} -> /snapshot/results.tar inside helper (tolerates live writes)"
if ! kubectl exec "${HELPER_POD}" -n "${NAMESPACE}" -- \
       tar -C "/results/${STUDY_ID}" \
           --warning=no-file-changed --ignore-failed-read \
           -cf /snapshot/results.tar .; then
  log "snapshot tar failed"
  exit 1
fi
SIZE="$(kubectl exec "${HELPER_POD}" -n "${NAMESPACE}" -- \
        stat -c '%s' /snapshot/results.tar 2>/dev/null || echo 0)"
log "snapshot ready (${SIZE} bytes)"

# ---- 5. Copy snapshot tarball out (chunked, verified, retried) ----------
# Earlier revisions streamed the entire tarball through a single
# `kubectl exec cat > local.tar` pipe. On this cluster's API server that
# SPDY stream gets truncated once the tarball is large enough to keep
# the connection open longer than the ingress idle timeout — the
# client returns exit 0 (remote `cat` completed fine) but the local
# file is short by tens or hundreds of KiB. The `SIZE == LOCAL_SIZE`
# check catches it but the whole retrieval then fails.
#
# Fix: split the snapshot into fixed-size chunks inside the helper,
# sha256 each one, then pull chunks one at a time with per-chunk
# digest verification + retry. A single failing chunk retries in
# isolation instead of restarting the whole transfer.
mkdir -p "${DEST}"
SCRATCH="$(mktemp -d "${DEST}/.rsync-scratch.XXXXXX")"
scratch_cleanup() {
  [[ -d "${SCRATCH}" ]] && rm -rf "${SCRATCH}" || true
}

# Empirically determined against this cluster's API server proxy:
# `kubectl exec cat` reliably transfers ≤4 MiB per stream; 8 MiB truncates
# occasionally; 16-32 MiB truncates a lot. 4 MiB with a small retry
# budget survives every observed transient. 200 MiB → ~50 chunks —
# per-chunk `kubectl exec` overhead is ~0.5 s, so total transfer stays
# well under a minute on a warm connection.
CHUNK_BYTES=$((512 * 1024))          # 512 KiB per chunk
# Note: this cluster's API-server SPDY tunnel truncates single `kubectl exec cat`
# streams at ~3 MiB — a 4 MiB chunk returns 3.2 MiB and no error, and the
# sha256 verify fails every retry the same way. It also drops bursts of
# smaller transfers under sustained load; 1 MiB chunks still lost enough
# individual pulls that 5 retries weren't sufficient for a 450+ chunk
# transfer. 512 KiB stays well clear of the truncation ceiling AND is
# small enough that each retry is fast; combined with the 15-retry
# budget below, transient network dropouts survive without human
# intervention.
CHUNK_RETRIES=15                     # attempts per chunk before giving up
# Wall-clock cap on a single `kubectl exec cat <chunk>` invocation. Some
# API-server-side truncations don't return exit-with-short-data; they
# stall the SPDY stream indefinitely with no EOF, no error, no bytes.
# Without a timeout the script wedges on that first attempt forever
# and the retry loop below never fires. 60 s is generous — a healthy
# 4 MiB transfer completes in under 5 s on this cluster.
CHUNK_TIMEOUT_SECONDS=60

log "splitting helper:/snapshot/results.tar into ${CHUNK_BYTES}-byte chunks + hashing"
# split -d = numeric suffixes (chunk-000, chunk-001, ...); -a 4 = 4-digit
# suffix so we handle up to 10000 chunks (~312 GiB) without overflow.
if ! kubectl exec -n "${NAMESPACE}" "${HELPER_POD}" -- sh -c "
    set -eu
    cd /snapshot
    rm -f chunk-* chunks.manifest
    split -d -a 4 -b ${CHUNK_BYTES} results.tar chunk-
    for f in chunk-*; do
      sz=\$(stat -c '%s' \"\$f\")
      dg=\$(sha256sum \"\$f\" | awk '{print \$1}')
      printf '%s\t%s\t%s\n' \"\$f\" \"\$sz\" \"\$dg\" >> chunks.manifest
    done
    cat chunks.manifest
  " > "${SCRATCH}/chunks.manifest"; then
  log "helper-side chunk + hash failed"
  exit 1
fi
NUM_CHUNKS="$(wc -l < "${SCRATCH}/chunks.manifest" | tr -d ' ')"
log "helper prepared ${NUM_CHUNKS} chunks; pulling with per-chunk verify"

# Portable sha256 for the local file: prefer `shasum -a 256` (macOS +
# most Linux); fall back to `sha256sum` if only that's available.
if command -v shasum >/dev/null 2>&1; then
  SHA256_LOCAL() { shasum -a 256 "$1" | awk '{print $1}'; }
elif command -v sha256sum >/dev/null 2>&1; then
  SHA256_LOCAL() { sha256sum "$1" | awk '{print $1}'; }
else
  log "neither shasum nor sha256sum found on the laptop; cannot verify chunks"
  exit 1
fi

fetch_chunk() {
  # Pull a single chunk and verify its digest. Prints nothing on
  # success; returns non-zero (silently) so the caller can retry.
  #
  # The `kubectl exec cat` is run in the background so we can enforce
  # CHUNK_TIMEOUT_SECONDS ourselves — `timeout(1)` is not portable
  # (macOS ships without it by default) and a stalled SPDY stream
  # returns neither EOF nor error, so relying on the child's own exit
  # would wedge the loop forever.
  local name="$1" want_size="$2" want_digest="$3"
  kubectl exec -n "${NAMESPACE}" "${HELPER_POD}" -- \
       cat "/snapshot/${name}" > "${SCRATCH}/${name}" 2>/dev/null &
  local pid=$!
  local waited=0
  while kill -0 "${pid}" 2>/dev/null; do
    if [[ "${waited}" -ge "${CHUNK_TIMEOUT_SECONDS}" ]]; then
      # SIGTERM then, if it doesn't die, SIGKILL — some `kubectl exec`
      # processes ignore SIGTERM when their SPDY stream is wedged.
      kill "${pid}" 2>/dev/null || true
      sleep 1
      kill -9 "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
      return 1
    fi
    sleep 1
    waited=$((waited + 1))
  done
  wait "${pid}"
  local rc=$?
  if [[ "${rc}" -ne 0 ]]; then
    return 1
  fi
  local got_size
  got_size="$(stat -f '%z' "${SCRATCH}/${name}" 2>/dev/null \
              || stat -c '%s' "${SCRATCH}/${name}" 2>/dev/null || echo 0)"
  if [[ "${got_size}" != "${want_size}" ]]; then
    return 1
  fi
  local got_digest
  got_digest="$(SHA256_LOCAL "${SCRATCH}/${name}")"
  if [[ "${got_digest}" != "${want_digest}" ]]; then
    return 1
  fi
  return 0
}

while IFS=$'\t' read -r cname csize cdigest; do
  [[ -z "${cname}" ]] && continue
  ok=0
  for attempt in $(seq 1 "${CHUNK_RETRIES}"); do
    if fetch_chunk "${cname}" "${csize}" "${cdigest}"; then
      ok=1; break
    fi
    log "  chunk ${cname} attempt ${attempt}/${CHUNK_RETRIES} failed; retrying"
    rm -f "${SCRATCH}/${cname}"
    sleep 2
  done
  if [[ "${ok}" != "1" ]]; then
    log "chunk ${cname} failed all ${CHUNK_RETRIES} attempts; giving up"
    log "scratch left at ${SCRATCH} for post-mortem"
    trap - EXIT
    exit 1
  fi
done < "${SCRATCH}/chunks.manifest"

log "all ${NUM_CHUNKS} chunks verified; reassembling"
# Chunks are guaranteed to sort lexicographically in order because
# `split -d -a 4` emits fixed-width numeric suffixes.
cat "${SCRATCH}"/chunk-* > "${SCRATCH}/results.tar"
LOCAL_SIZE="$(stat -f '%z' "${SCRATCH}/results.tar" 2>/dev/null \
              || stat -c '%s' "${SCRATCH}/results.tar" 2>/dev/null || echo 0)"
if [[ "${LOCAL_SIZE}" != "${SIZE}" ]]; then
  log "reassembled size mismatch: helper had ${SIZE} bytes, got ${LOCAL_SIZE}"
  log "this indicates a chunk-manifest / split disagreement, not a transport"
  log "issue — inspect ${SCRATCH} to diagnose"
  exit 1
fi
# Chunks served their purpose; drop them before extraction so the
# scratch directory doesn't briefly hold ~2× the tarball's bytes.
rm -f "${SCRATCH}"/chunk-* "${SCRATCH}/chunks.manifest"
log "reassembled ${LOCAL_SIZE} bytes intact"

log "extracting tarball locally"
tar -xf "${SCRATCH}/results.tar" -C "${SCRATCH}/" && rm "${SCRATCH}/results.tar"

# ---- 6. Derive the per-experiment folder name ---------------------------
# The study id we already resolved on the PVC is authoritative — the tar
# root IS /results/<STUDY_ID>/, so its own uber/progress.jsonl and this
# id are the same value. Keep the derive-from-progress and retrieval
# fallbacks so the script still degrades gracefully if a pre-migration
# snapshot ever gets fed in.
FOLDER="${STUDY_ID}"
if [[ -z "${FOLDER}" ]]; then
  FOLDER="$(derive_from_progress "${SCRATCH}")"
fi
if [[ -z "${FOLDER}" ]]; then
  FOLDER="$(retrieval_timestamp)"
  log "no study id resolved; falling back to retrieval time ${FOLDER}"
else
  log "using folder ${FOLDER}"
fi
TARGET="${DEST}/${FOLDER}"

# ---- 7. Move scratch → target (or merge if target exists) ---------------
if [[ -d "${TARGET}" ]]; then
  # A previous partial retrieval of the same study already created this
  # folder; merge into it. Since we extracted from a stable tarball, a
  # plain rsync-in-place on the LOCAL filesystem is safe (no in-flight
  # writes) — use cp -a to keep it dependency-free.
  log "target ${TARGET} exists — merging scratch into it"
  # Copy each top-level entry into the target (cp -a preserves attrs +
  # follows into subdirs). If cp complains, keep going: we prefer
  # partial retrieval over a hard failure.
  (cd "${SCRATCH}" && for f in .[!.]* * ; do
     [[ "$f" = ".[!.]*" || "$f" = "*" ]] && continue
     cp -a --update "$f" "${TARGET}/" 2>&1 || true
   done)
  scratch_cleanup
else
  # Atomic when scratch + DEST are on the same filesystem (they are,
  # since scratch was mkdtemp'd under DEST). Otherwise mv degrades to
  # copy-then-remove, still correct.
  log "moving ${SCRATCH} -> ${TARGET}"
  mv "${SCRATCH}" "${TARGET}"
fi

# ---- 8. Update the `latest` symlink -------------------------------------
LATEST="${DEST}/latest"
# `-n` prevents ln from dereferencing an existing symlink and creating
# `latest/<folder>` inside the old target — we want to REPLACE the link.
ln -sfn "${FOLDER}" "${LATEST}"
log "latest -> ${FOLDER}"

# ---- 9. Verify manifest --------------------------------------------------
if [[ -f "${TARGET}/consolidated/_manifest.json" ]]; then
  python3 "$(dirname "$0")/verify_manifest.py" "${TARGET}/consolidated" || true
fi

log "results at ${TARGET}"
