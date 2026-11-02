#!/usr/bin/env bash
# The end-to-end matrix against the HTTP API of the binary build.sh made.
# Starts izwi-server with RUST_LOG=info, stops unless its logged
# build_git_sha is the built one, loads the model and runs (each case
# passes, fails, errors, is inconclusive or is skipped into
# <out>/report.json, and every take is kept under <out>/audio/):
#   a voice the seed sets (twice: the same take); data/fox.wav cloned inline
#   and as a saved voice; tags, two leading their lines; a two-speaker script
#   with a reference each, then each line alone; long text as a chunk session
#   with its history and samples where the chunks meet; SSE time to first audio
#   and streamed PCM == the WAV for one seed; the formats; a take edit through
#   previous_take_ids; cancellation mid-preparation, mid-prefill, mid-decode
#   and after the first audio; a soak with untracked device memory flat; the
#   backend's fast-path counters; optionally every take through an ASR model;
#   unload and reload; then a second server with an undersized memory budget
#   that must refuse the load cleanly and quickly.
#
# Usage: run-e2e.sh <cpu|metal|cuda> [--synthetic] [--dry-run] [fd3_harness.py run options...]
#   --synthetic  serve the synthetic checkpoint (random tiny weights, a stub
#                codec) with the build.sh --synthetic binary: checks the kit
#                and the routes without the weights, not the model
#   --dry-run    print the cases and what would run; start nothing
#   e.g. --only plain,stream_sse  --skip soak_untracked_memory
#        --soak-minutes 30  --asr-model Whisper-Large-v3-Turbo
#        --undersized-budget-mib 8192  --seed 1
# Env: FD3_MODELS_DIR / IZWI_MODELS_DIR; IZWI_SERVER_BIN overrides the binary
#      (then pass --expect-sha or --no-sha-check); IZWI_FISH_DRAMA3_* engine
#      knobs reach the server and are recorded in the report.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BACKEND="${1:-}"
if [[ "$BACKEND" == "-h" || "$BACKEND" == "--help" ]]; then
    sed -n '2,28p' "$0"
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
        -h | --help) sed -n '2,28p' "$0"; exit 0 ;;
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
PY="$(fd3_python)"
# The synthetic checkpoint comes from an izwi-server test, built with the
# server binary's features in the same target directory.
FEATURES="$(fd3_server_features "$BACKEND")"
WRITER=(cargo test --locked -p izwi-server --lib)
if [[ -n "$FEATURES" ]]; then WRITER+=(--features "$FEATURES"); fi
WRITER+=(-- --ignored --exact api::fish_drama3_speech_tests::write_the_e2e_kit_synthetic_checkpoint)

if ((DRY_RUN)); then
    fd3_log "dry run: $BIN against $MODELS_DIR; report $OUT/report.json"
    if ((SYNTHETIC)); then
        fd3_log "  first: CARGO_TARGET_DIR=$(fd3_target_dir "$BACKEND") $(fd3_show "${WRITER[@]}")"
    fi
    exec "$PY" "$FD3_DIR/fd3_harness.py" plan --backend "$BACKEND" ${MODE[@]+"${MODE[@]}"} \
        ${PASS[@]+"${PASS[@]}"}
fi

[[ -x "$BIN" ]] || fd3_die "no server binary at $BIN; run build.sh $BACKEND${MODE[*]:+ --synthetic}"
if ((SYNTHETIC)); then
    fd3_log "writing the synthetic checkpoint into $MODELS_DIR"
    export CARGO_TARGET_DIR="$(fd3_target_dir "$BACKEND")"
    export CARGO_INCREMENTAL=0
    mkdir -p "$OUT"
    (cd "$REPO_ROOT" && "${WRITER[@]}") >"$OUT/synthetic-writer.log" 2>&1 ||
        fd3_die "writing the synthetic checkpoint failed; see $OUT/synthetic-writer.log"
    [[ -f "$MODELS_DIR/$FD3_VARIANT/config.json" ]] ||
        fd3_die "no synthetic checkpoint at $MODELS_DIR/$FD3_VARIANT"
else
    bash "$FD3_DIR/fetch.sh" --dest "$MODELS_DIR/$FD3_VARIANT" --verify-only
fi

fd3_log "E2E on $BACKEND with $BIN; models in $MODELS_DIR; report $OUT/report.json"
exec "$PY" "$FD3_DIR/fd3_harness.py" run --backend "$BACKEND" --binary "$BIN" \
    --models-dir "$MODELS_DIR" --out "$OUT" --data-dir "$REPO_ROOT/data" \
    ${MODE[@]+"${MODE[@]}"} ${PASS[@]+"${PASS[@]}"}
