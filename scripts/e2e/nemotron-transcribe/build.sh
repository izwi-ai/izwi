#!/usr/bin/env bash
# Release build of izwi-server for one backend, with the lessons applied:
# a private CARGO_TARGET_DIR (never shared with another checkout or session),
# IZWI_BUILD_GIT_SHA pinned to HEAD, CARGO_INCREMENTAL=0, a free-disk check
# before building, and the binary copied out in the same command, then checked
# for size (a full disk once left a 4 KB binary) and for the variant id.
#
# Usage: build.sh <cpu|metal|cuda>
# Env: NT_SERVER_FEATURES (metal: metal; cuda: cuda), CUDA_COMPUTE_CAP (default:
#      nvidia-smi, else 80), NT_CARGO_TARGET_DIR, NT_BUILD_MIN_FREE_GB (25),
#      NT_OUT_ROOT (default <repo>/target/nemotron-transcribe).
# Writes <out>/bin/izwi-server, <out>/build-info.json and <out>/build.log.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BACKEND="${1:-}"
nt_check_backend "$BACKEND"
OUT="$(nt_backend_out "$BACKEND")"
TARGET_DIR="$(nt_target_dir "$BACKEND")"
mkdir -p "$OUT/bin" "$TARGET_DIR"

case "$BACKEND" in
    metal) [[ "$(uname -s)" == "Darwin" ]] || nt_die "metal builds need macOS" ;;
    cuda) command -v nvcc >/dev/null 2>&1 || nt_die "cuda builds need nvcc on PATH" ;;
esac

nt_require_free_gb "$TARGET_DIR" "${NT_BUILD_MIN_FREE_GB:-25}"

SHA="$(nt_git_sha)"
DIRTY="$(nt_git_dirty)"
if [[ "$DIRTY" == "true" ]]; then
    nt_log "warning: the checkout has uncommitted changes; the binary reports $SHA anyway"
fi
FEATURES="$(nt_server_features "$BACKEND")"
FEATURE_ARGS=()
if [[ -n "$FEATURES" ]]; then FEATURE_ARGS=(--features "$FEATURES"); fi
COMPUTE_CAP=""
if [[ "$BACKEND" == "cuda" ]]; then
    COMPUTE_CAP="$(nt_cuda_compute_cap)"
    export CUDA_COMPUTE_CAP="$COMPUTE_CAP"
    nt_log "CUDA_COMPUTE_CAP=$CUDA_COMPUTE_CAP"
fi

export CARGO_TARGET_DIR="$TARGET_DIR"
export CARGO_INCREMENTAL=0
export IZWI_BUILD_GIT_SHA="$SHA"

nt_log "building izwi-server ($BACKEND, features: ${FEATURES:-none}) at $SHA into $TARGET_DIR"
started=$(date +%s)
(
    cd "$REPO_ROOT"
    cargo build --release --locked -p izwi-server --bin izwi-server \
        ${FEATURE_ARGS[@]+"${FEATURE_ARGS[@]}"} 2>&1
) | tee "$OUT/build.log"
seconds=$(($(date +%s) - started))

BUILT="$TARGET_DIR/release/izwi-server"
[[ -x "$BUILT" ]] || nt_die "no binary at $BUILT; see $OUT/build.log"
cp "$BUILT" "$OUT/bin/izwi-server.tmp" && mv "$OUT/bin/izwi-server.tmp" "$OUT/bin/izwi-server"
BIN="$OUT/bin/izwi-server"
BYTES="$(nt_file_size "$BIN")"
if ((BYTES < 10 * 1024 * 1024)); then
    nt_die "$BIN is only $BYTES bytes (truncated link? check free disk)"
fi
if ! grep -a -q "$NT_VARIANT" "$BIN"; then
    nt_die "$BIN does not contain $NT_VARIANT: a stale or wrong build"
fi
if ! grep -a -q "$SHA" "$BIN"; then
    nt_die "$BIN does not carry IZWI_BUILD_GIT_SHA=$SHA"
fi
BIN_SHA="$(nt_sha256 "$BIN")"

"$(nt_python)" - "$OUT/build-info.json" <<PY
import json, sys
info = {
    "backend": "$BACKEND",
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

nt_log "binary $BIN"
nt_log "  bytes  $BYTES"
nt_log "  sha256 $BIN_SHA"
nt_log "  git    $SHA (dirty: $DIRTY)"
nt_log "build-info: $OUT/build-info.json"
