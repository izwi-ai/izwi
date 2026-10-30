#!/usr/bin/env bash
# End-to-end matrix (plan §10.4) against the HTTP API of the binary build.sh
# made. Starts izwi-server with RUST_LOG=info, stops unless its logged
# build_git_sha is the built one, loads the model and runs (each case passes,
# fails, errors or is skipped into <out>/report.json):
#   json, verbose_json, srt, vtt, SSE (== the json transcript); language and
#   an unsupported language; prompt override; keywords; translation en->de and
#   de->en, refused pairs and translation + word timestamps refused; native
#   word timestamps; speaker-attributed transcription (jobs API); the 60-minute
#   file; cancellation mid-prefill, mid-decode and by client disconnect, with
#   cancel->idle latency; a soak with untracked device memory flat; unload and
#   reload (identical transcript); then a second server with an undersized
#   memory budget that must refuse the load cleanly and quickly.
#
# Usage: run-e2e.sh <cpu|metal|cuda> [nt_harness.py run options...]
#   e.g. --skip-long, --de-clip PATH, --only keywords,stream_sse,
#        --skip long_60min, --soak-requests 48, --undersized-budget-gib 8
# Env: IZWI_NEMOTRON_TRANSCRIBE_FP8_MODE (w8a16 | vllm-fp8) is passed to the
#      server; NT_MODELS_DIR / IZWI_MODELS_DIR; IZWI_SERVER_BIN overrides the
#      binary (then pass --expect-sha or --no-sha-check).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BACKEND="${1:-}"
nt_check_backend "$BACKEND"
shift
OUT="$(nt_backend_out "$BACKEND")"
BIN="${IZWI_SERVER_BIN:-$OUT/bin/izwi-server}"
[[ -x "$BIN" ]] || nt_die "no server binary at $BIN; run build.sh $BACKEND"
command -v ffmpeg >/dev/null 2>&1 || nt_die "ffmpeg is needed to prepare the audio"
bash "$NT_DIR/fetch.sh" --dest "$NT_MODELS_DIR/$NT_VARIANT" --verify-only

nt_log "E2E on $BACKEND with $BIN; models in $NT_MODELS_DIR; report $OUT/report.json"
exec "$(nt_python)" "$NT_DIR/nt_harness.py" run --backend "$BACKEND" --binary "$BIN" \
    --models-dir "$NT_MODELS_DIR" --out "$OUT" --data-dir "$REPO_ROOT/data" "$@"
