#!/usr/bin/env bash
# Concurrency and parity under load: scripts/bench/run-audio-load.py --task
# tts --require-parity at c = 1, 2 (cpu, metal) or 1, 2, 4, 8, 16, 32
# (cuda), seeded, with mixed text lengths (4-60 words, so some inputs are
# chunk sessions) and Poisson (staggered) arrivals; every take under load
# must equal its solo take sample for sample (--tts-tolerance 0). At each
# width a streaming probe measures time to first audio, RTF and generated
# frames/s as the card's serving table does (reported beside it). With
# FD3_CHAT_MODEL (or --chat-model), chat inter-token latency is measured
# alone and beside a long Drama 3 take.
#
# Usage: run-load.sh <cpu|metal|cuda> [--synthetic] [--dry-run] [fd3_harness.py load options...]
#   e.g. --concurrency 1,2 --request-rate 0.25 --voice clone --seed 7
#        --chat-model Qwen3.5-0.8B --probe-rounds 4
# Env: FD3_LOAD_CONCURRENCY overrides the widths; IZWI_FISH_DRAMA3_NATIVE_BATCHING
#      (Metal: two rows at once) reaches the server.
# Writes <out>/load-c<N>.json and the `load` section of <out>/report.json.
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
PASS=()
while (($#)); do
    case "$1" in
        --synthetic) SYNTHETIC=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h | --help) sed -n '2,18p' "$0"; exit 0 ;;
        *) PASS+=("$1") ;;
    esac
    shift
done
MODE=()
MODELS_DIR="$FD3_MODELS_DIR"
if ((SYNTHETIC)); then
    MODE=(--synthetic)
    MODELS_DIR="$FD3_SYNTHETIC_MODELS_DIR"
fi
OUT="$(fd3_backend_out "$BACKEND" "$SYNTHETIC")"
BIN="${IZWI_SERVER_BIN:-$OUT/bin/izwi-server}"
case "$BACKEND" in
    cuda) LEVELS="${FD3_LOAD_CONCURRENCY:-1,2,4,8,16,32}" ;;
    *) LEVELS="${FD3_LOAD_CONCURRENCY:-1,2}" ;;
esac
CHAT=()
if [[ -n "${FD3_CHAT_MODEL:-}" ]]; then CHAT=(--chat-model "$FD3_CHAT_MODEL"); fi
LOAD=("$(fd3_python)" "$FD3_DIR/fd3_harness.py" load --backend "$BACKEND" --binary "$BIN"
    --models-dir "$MODELS_DIR" --out "$OUT" --data-dir "$REPO_ROOT/data"
    --concurrency "$LEVELS" ${MODE[@]+"${MODE[@]}"} ${CHAT[@]+"${CHAT[@]}"}
    ${PASS[@]+"${PASS[@]}"})

if ((DRY_RUN)); then
    fd3_log "dry run: load sweep on $BACKEND at c=$LEVELS (a later --concurrency wins)"
    fd3_log "  $(fd3_show "${LOAD[@]}")"
    fd3_log "  each width: run-audio-load.py --task tts --require-parity --tts-tolerance 0, then a streaming probe"
    exit 0
fi

[[ -x "$BIN" ]] || fd3_die "no server binary at $BIN; run build.sh $BACKEND${MODE[*]:+ --synthetic}"
if ((SYNTHETIC)); then
    [[ -f "$MODELS_DIR/$FD3_VARIANT/config.json" ]] ||
        fd3_die "no synthetic checkpoint at $MODELS_DIR; run run-e2e.sh $BACKEND --synthetic first"
else
    bash "$FD3_DIR/fetch.sh" --dest "$MODELS_DIR/$FD3_VARIANT" --verify-only
fi

fd3_log "load sweep on $BACKEND at c=$LEVELS"
exec "${LOAD[@]}"
