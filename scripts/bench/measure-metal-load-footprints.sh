#!/usr/bin/env bash
# Measure what each model's Metal load really allocates beside what load
# admission prices for it. Each model loads in its own process (Candle's Metal
# pool outlives a model), through the lifecycle's registry dispatch and without
# admission, so a model the server refuses can still be measured.
#
# See crates/izwi-core/src/runtime/lifecycle/load/footprint_tests.rs.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/bench/measure-metal-load-footprints.sh [options] [MODEL...]

MODEL is a catalog name (for example Qwen3.5-4B). Without one, every
downloaded catalog model in the default list is measured.

Options:
  --output PATH      JSON Lines output (default: target/metal-load-footprints.jsonl,
                     replaced on each run)
  --models-dir DIR   Models directory (default: ~/Library/Application Support/izwi/models)
  -h, --help         Show this help
EOF
}

output="target/metal-load-footprints.jsonl"
models_dir="${IZWI_MODELS_DIR:-$HOME/Library/Application Support/izwi/models}"
models=()
while (($#)); do
    case "$1" in
        --output) output="$2"; shift 2 ;;
        --models-dir) models_dir="$2"; shift 2 ;;
        -h | --help) usage; exit 0 ;;
        -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
        *) models+=("$1"); shift ;;
    esac
done

if ((${#models[@]} == 0)); then
    # One or more per family; Fish S2 Pro (11 GB) is left out on purpose,
    # name it explicitly to measure it.
    models=(
        Qwen3.5-0.8B Qwen3.5-2B Qwen3.5-4B
        LFM2.5-1.2B-Instruct-GGUF
        Qwen3-ASR-0.6B-GGUF Qwen3-ASR-1.7B-GGUF
        Granite-Speech-4.1-2B-Plus
        Whisper-Large-v3-Turbo
        Parakeet-TDT-0.6B-v3
        Nemotron-3.5-ASR-Streaming-0.6B
        LFM2.5-Audio-1.5B-GGUF
        Qwen3-ForcedAligner-0.6B
        diar_streaming_sortformer_4spk-v2.1 Nemotron-3-Diarization
        Qwen3-TTS-12Hz-0.6B-Base Qwen3-TTS-12Hz-0.6B-CustomVoice
        Qwen3-TTS-12Hz-1.7B-Base Qwen3-TTS-12Hz-1.7B-CustomVoice
        Qwen3-TTS-12Hz-1.7B-VoiceDesign
        Kokoro-82M
        VibeVoice-1.5B
    )
fi

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

echo "Building the izwi-core Metal test binary (release)..." >&2
binary="$(
    cargo test -p izwi-core --release --features metal --lib --no-run \
        --message-format=json 2>/dev/null |
        python3 -I -c '
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("reason") == "compiler-artifact" and msg.get("executable") \
            and msg["target"]["name"] == "izwi_core" and msg["profile"]["test"]:
        print(msg["executable"])
' | tail -n 1
)"
if [[ -z "$binary" ]]; then
    echo "could not find the izwi-core test binary; run the cargo command above to see the error" >&2
    exit 1
fi

mkdir -p "$(dirname "$output")"
: >"$output"
failed=()
for model in "${models[@]}"; do
    echo "== $model" >&2
    if ! IZWI_MEASURE_MODEL="$model" IZWI_MEASURE_OUT="$output" IZWI_MODELS_DIR="$models_dir" \
        "$binary" measure_metal_load_footprint --ignored --nocapture --test-threads=1 \
        2>&1 | grep -E "FOOTPRINT |panicked|failed to load|not downloaded|skipped"; then
        failed+=("$model")
    fi
done

python3 -I - "$output" <<'EOF'
import json, sys

rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
gb = lambda value: f"{value / 1e9:6.2f}"
print(f"\n{'model':38} {'stored':>6} {'admit':>6} {'plan_pk':>7} {'plan_res':>8} "
      f"{'mtl_pk':>6} {'mtl_res':>7} {'fp_pk':>6} {'fp_res':>6} {'ms':>7}")
for r in rows:
    print(f"{r['model'][:38]:38} {gb(r['stored_bytes'])} {gb(r['admission_bytes'])} "
          f"{gb(r['plan_load_peak_bytes']):>7} {gb(r['plan_resident_bytes']):>8} "
          f"{gb(r['metal_peak_bytes'])} {gb(r['metal_settled_bytes']):>7} "
          f"{gb(r['footprint_peak_bytes'])} {gb(r['footprint_settled_bytes'])} {r['load_ms']:>7}")
print("\nGB = 1e9 bytes. admit: what the Metal pre-load check compares. mtl: Metal "
      "currentAllocatedSize growth; fp: process phys_footprint growth; res: after the "
      "load fence and a pool flush.")
EOF

if ((${#failed[@]})); then
    echo "failed: ${failed[*]}" >&2
    exit 1
fi
