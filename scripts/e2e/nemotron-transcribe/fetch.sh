#!/usr/bin/env bash
# Resumable download of nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8 at
# the pinned revision into izwi's models directory, checked against the exact
# sizes at that revision (plan §2.1). Bash 3.2 compatible.
#
# Usage: fetch.sh [--dest DIR] [--jobs 1|2] [--verify-only]
#
# Each file is a shell loop around `curl -C -` that appends in place to the
# file itself (no part files to join) until it has its exact size; stalled
# connections are dropped after NT_FETCH_STALL_SECS below 10 KB/s and resumed.
# At most two files download at once (lessons: more connections did not add
# bandwidth from Hugging Face's CDN). Optional env: HF_TOKEN (the repo is not
# gated), NT_FETCH_MAX_ATTEMPTS (200), NT_FETCH_STALL_SECS (120).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# name size: the files izwi's downloader requires, then the backbone config
# the L2 generator may read. Small files first.
FILES="
config.json 433812
generation_config.json 92
hf_quant_config.json 2275
tokenizer_config.json 491
chat_template.jinja 12482
llm_backbone/config.json 2403
model.safetensors.index.json 1975374
tokenizer.json 17078144
model-ctc-bf16.safetensors 201897998
perception.safetensors 838256800
model-mtp-fp8.safetensors 1335778712
model-00004-of-00004.safetensors 2296485296
model-00001-of-00004.safetensors 9997859176
model-00002-of-00004.safetensors 10000407944
model-00003-of-00004.safetensors 9998941520
"

fetch_one() {
    local dest="$1" name="$2" size="$3"
    local path="$dest/$name"
    local url="https://huggingface.co/$NT_REPO_ID/resolve/$NT_REVISION/$name"
    local attempt=0 have
    mkdir -p "$(dirname "$path")"
    while true; do
        have=0
        if [[ -f "$path" ]]; then have="$(nt_file_size "$path")"; fi
        if ((have == size)); then
            nt_log "ok   $name ($size bytes)"
            return 0
        fi
        if ((have > size)); then
            nt_log "$name has $have bytes, more than the pinned $size; starting it again"
            rm -f "$path"
            have=0
        fi
        attempt=$((attempt + 1))
        if ((attempt > ${NT_FETCH_MAX_ATTEMPTS:-200})); then
            nt_die "$name: giving up after $((attempt - 1)) attempts at $have/$size bytes"
        fi
        nt_log "get  $name from byte $have of $size (attempt $attempt)"
        local auth=()
        if [[ -n "${HF_TOKEN:-}" ]]; then auth=(-H "Authorization: Bearer $HF_TOKEN"); fi
        if ! curl -L --fail --silent --show-error -C - --connect-timeout 30 \
            --speed-time "${NT_FETCH_STALL_SECS:-120}" --speed-limit 10240 \
            ${auth[@]+"${auth[@]}"} -o "$path" "$url"; then
            sleep $((attempt < 30 ? attempt : 30))
        fi
    done
}

if [[ "${1:-}" == "--one" ]]; then
    # Internal: one file, called through xargs.
    fetch_one "$2" "$3" "$4"
    exit 0
fi

DEST="$NT_MODEL_DIR"
JOBS="${NT_FETCH_JOBS:-2}"
VERIFY_ONLY=0
while (($#)); do
    case "$1" in
        --dest) DEST="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --verify-only) VERIFY_ONLY=1; shift ;;
        -h | --help) sed -n '2,15p' "$0"; exit 0 ;;
        *) nt_die "unknown argument $1" ;;
    esac
done
if ((JOBS < 1 || JOBS > 2)); then nt_die "--jobs must be 1 or 2"; fi
mkdir -p "$DEST"

remaining=0
bad=0
while read -r name size; do
    [[ -z "$name" ]] && continue
    have=0
    if [[ -f "$DEST/$name" ]]; then have="$(nt_file_size "$DEST/$name")"; fi
    if ((have != size)); then
        bad=$((bad + 1))
        if ((have < size)); then remaining=$((remaining + size - have)); else remaining=$((remaining + size)); fi
    fi
done <<<"$FILES"

if ((VERIFY_ONLY)); then
    if ((bad)); then nt_die "$bad files are missing or not at their pinned size in $DEST"; fi
    nt_log "all files at their pinned sizes in $DEST (revision $NT_REVISION)"
    exit 0
fi

need_gb=$(((remaining >> 30) + 2))
nt_log "downloading $NT_REPO_ID@$NT_REVISION into $DEST: $((remaining >> 20)) MiB to go"
nt_require_free_gb "$DEST" "$need_gb"

# Small files one at a time, then the large ones JOBS at a time.
echo "$FILES" | awk 'NF == 2 && $2 < 100000000' | while read -r name size; do
    fetch_one "$DEST" "$name" "$size"
done
echo "$FILES" | awk 'NF == 2 && $2 >= 100000000 { print $1, $2 }' |
    xargs -n 2 -P "$JOBS" bash "$0" --one "$DEST"

# Final check, every file at its exact size.
bash "$0" --dest "$DEST" --verify-only
nt_log "izwi reads it from IZWI_MODELS_DIR=$(dirname "$DEST") as $NT_VARIANT"
