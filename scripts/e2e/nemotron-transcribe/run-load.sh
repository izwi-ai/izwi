#!/usr/bin/env bash
# Concurrency and parity under load: scripts/bench/run-audio-load.py
# --require-parity at c = 1, 2, 4 (cpu, metal) or 1, 2, 4, 8, 16, 32 (cuda),
# with mixed clip lengths (2-95 s cut from four real recordings) and Poisson
# (staggered) arrivals; every request's text must equal its solo text. With
# NT_CHAT_MODEL (or --chat-model), chat inter-token latency is measured alone
# and beside a running one-hour transcription.
#
# Usage: run-load.sh <cpu|metal|cuda> [nt_harness.py load options...]
#   e.g. --concurrency 1,2 --request-rate 0.25 --max-secs 60
#        --chat-model Qwen3-0.6B
# Writes <out>/load-c<N>.json and the `load` section of <out>/report.json.
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

case "$BACKEND" in
    cuda) LEVELS="${NT_LOAD_CONCURRENCY:-1,2,4,8,16,32}" ;;
    *) LEVELS="${NT_LOAD_CONCURRENCY:-1,2,4}" ;;
esac
CHAT=()
if [[ -n "${NT_CHAT_MODEL:-}" ]]; then CHAT=(--chat-model "$NT_CHAT_MODEL"); fi

nt_log "load sweep on $BACKEND at c=$LEVELS"
exec "$(nt_python)" "$NT_DIR/nt_harness.py" load --backend "$BACKEND" --binary "$BIN" \
    --models-dir "$NT_MODELS_DIR" --out "$OUT" --data-dir "$REPO_ROOT/data" \
    --concurrency "$LEVELS" ${CHAT[@]+"${CHAT[@]}"} "$@"
