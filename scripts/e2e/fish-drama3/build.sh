#!/usr/bin/env bash
# Release build of izwi-server for one backend, with a private
# CARGO_TARGET_DIR (never shared with another checkout or session),
# IZWI_BUILD_GIT_SHA pinned to HEAD, CARGO_INCREMENTAL=0, a free-disk check
# before building, and the binary copied out in the same command, then
# checked for size (a full disk once left a 4 KB binary), for the variant id
# and for the sha.
#
# Usage: build.sh <cpu|metal|cuda> [--synthetic] [--dry-run]
#   --synthetic  a debug build with izwi-core's `test-support` feature, which
#                serves the synthetic checkpoint (run-e2e.sh --synthetic) and
#                nothing a release build would not; into <out>-synthetic/
#   --dry-run    print the build command and its environment; build nothing
# Env: FD3_SERVER_FEATURES (metal: metal; cuda: cuda), CUDA_COMPUTE_CAP
#      (default: nvidia-smi, else 80), FD3_CARGO_TARGET_DIR,
#      FD3_BUILD_MIN_FREE_GB (25), FD3_OUT_ROOT (default <repo>/target/fish-drama3).
# Writes <out>/bin/izwi-server, <out>/build-info.json and <out>/build.log.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BACKEND="${1:-}"
if [[ "$BACKEND" == "-h" || "$BACKEND" == "--help" ]]; then
    sed -n '2,18p' "$0"
    exit 0
fi
fd3_check_backend "$BACKEND"
shift
SYNTHETIC=0
DRY_RUN=0
while (($#)); do
    case "$1" in
        --synthetic) SYNTHETIC=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h | --help) sed -n '2,18p' "$0"; exit 0 ;;
        *) fd3_die "unknown argument $1" ;;
    esac
    shift
done

OUT="$(fd3_backend_out "$BACKEND" "$SYNTHETIC")"
TARGET_DIR="$(fd3_target_dir "$BACKEND")"
SHA="$(fd3_git_sha)"
DIRTY="$(fd3_git_dirty)"
FEATURES="$(fd3_server_features "$BACKEND")"
PROFILE_ARGS=(--release)
PROFILE_DIR=release
if ((SYNTHETIC)); then
    FEATURES="${FEATURES:+$FEATURES,}izwi-core/test-support"
    PROFILE_ARGS=()
    PROFILE_DIR=debug
fi
FEATURE_ARGS=()
if [[ -n "$FEATURES" ]]; then FEATURE_ARGS=(--features "$FEATURES"); fi
COMPUTE_CAP=""
if [[ "$BACKEND" == "cuda" ]]; then COMPUTE_CAP="$(fd3_cuda_compute_cap)"; fi
BUILD=(cargo build ${PROFILE_ARGS[@]+"${PROFILE_ARGS[@]}"} --locked -p izwi-server --bin izwi-server
    ${FEATURE_ARGS[@]+"${FEATURE_ARGS[@]}"})

if ((DRY_RUN)); then
    fd3_log "dry run: would build in $REPO_ROOT with"
    fd3_log "  CARGO_TARGET_DIR=$TARGET_DIR CARGO_INCREMENTAL=0 IZWI_BUILD_GIT_SHA=$SHA${COMPUTE_CAP:+ CUDA_COMPUTE_CAP=$COMPUTE_CAP}"
    fd3_log "  $(fd3_show "${BUILD[@]}")"
    fd3_log "  then copy $TARGET_DIR/$PROFILE_DIR/izwi-server to $OUT/bin/izwi-server (dirty: $DIRTY)"
    fd3_log "  $(fd3_free_gb "$REPO_ROOT") GB free; needs ${FD3_BUILD_MIN_FREE_GB:-25} GB"
    exit 0
fi

case "$BACKEND" in
    metal) [[ "$(uname -s)" == "Darwin" ]] || fd3_die "metal builds need macOS" ;;
    cuda) command -v nvcc >/dev/null 2>&1 || fd3_die "cuda builds need nvcc on PATH" ;;
esac
mkdir -p "$OUT/bin" "$TARGET_DIR"
fd3_require_free_gb "$TARGET_DIR" "${FD3_BUILD_MIN_FREE_GB:-25}"
if [[ "$DIRTY" == "true" ]]; then
    fd3_log "warning: the checkout has uncommitted changes; the binary reports $SHA anyway"
fi
if [[ -n "$COMPUTE_CAP" ]]; then
    export CUDA_COMPUTE_CAP="$COMPUTE_CAP"
    fd3_log "CUDA_COMPUTE_CAP=$CUDA_COMPUTE_CAP"
fi

export CARGO_TARGET_DIR="$TARGET_DIR"
export CARGO_INCREMENTAL=0
export IZWI_BUILD_GIT_SHA="$SHA"

fd3_log "building izwi-server ($BACKEND, $PROFILE_DIR, features: ${FEATURES:-none}) at $SHA into $TARGET_DIR"
started=$(date +%s)
set +e
(cd "$REPO_ROOT" && "${BUILD[@]}" 2>&1) | tee "$OUT/build.log"
status=${PIPESTATUS[0]}
set -e
((status == 0)) || fd3_die "cargo build failed ($status); see $OUT/build.log"
seconds=$(($(date +%s) - started))

BUILT="$TARGET_DIR/$PROFILE_DIR/izwi-server"
[[ -x "$BUILT" ]] || fd3_die "no binary at $BUILT; see $OUT/build.log"
cp "$BUILT" "$OUT/bin/izwi-server.tmp" && mv "$OUT/bin/izwi-server.tmp" "$OUT/bin/izwi-server"
BIN="$OUT/bin/izwi-server"
BYTES="$(fd3_file_size "$BIN")"
if ((BYTES < 10 * 1024 * 1024)); then
    fd3_die "$BIN is only $BYTES bytes (truncated link? check free disk)"
fi
if ! grep -a -q "$FD3_VARIANT" "$BIN"; then
    fd3_die "$BIN does not contain $FD3_VARIANT: a stale or wrong build"
fi
if ! grep -a -q "$SHA" "$BIN"; then
    fd3_die "$BIN does not carry IZWI_BUILD_GIT_SHA=$SHA"
fi
BIN_SHA="$(fd3_sha256 "$BIN")"

"$(fd3_python)" - "$OUT/build-info.json" <<PY
import json, sys
info = {
    "backend": "$BACKEND",
    "synthetic": $([[ "$SYNTHETIC" == 1 ]] && echo True || echo False),
    "profile": "$PROFILE_DIR",
    "git_sha": "$SHA",
    "dirty": $([[ "$DIRTY" == true ]] && echo True || echo False),
    "features": "$FEATURES",
    "cuda_compute_cap": "$COMPUTE_CAP" or None,
    "binary": "bin/izwi-server",
    "binary_bytes": $BYTES,
    "binary_sha256": "$BIN_SHA",
    "build_seconds": $seconds,
    "cargo_target_dir": "$TARGET_DIR",
    "rustc": """$(rustc -V 2>/dev/null)""",
    "cargo": """$(cargo -V 2>/dev/null)""",
    "host": """$(uname -a)""",
}
open(sys.argv[1], "w").write(json.dumps(info, indent=1) + "\n")
PY

fd3_log "binary $BIN"
fd3_log "  bytes  $BYTES"
fd3_log "  sha256 $BIN_SHA"
fd3_log "  git    $SHA (dirty: $DIRTY)"
fd3_log "build-info: $OUT/build-info.json"
