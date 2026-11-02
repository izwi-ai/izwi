#!/usr/bin/env bash
# Resumable download of fishaudio/drama-3 at the pinned revision into izwi's
# models directory, checked against the exact sizes and LFS SHA-256 of that
# revision (the manifest in fish_drama3/artifacts.rs). Bash 3.2 compatible.
#
# Usage: fetch.sh [--dest DIR] [--jobs 1|2] [--verify-only] [--dry-run]
#   --verify-only  check that every file is at its pinned size and that the
#                  bundle was hashed (izwi-artifact.json); no network, no hashing
#   --dry-run      print what would be downloaded and the free disk; no network
#
# Each file is a shell loop around `curl -C -` that appends in place to the
# file itself (no part files to join) until it has its exact size; stalled
# connections are dropped after FD3_FETCH_STALL_SECS below 10 KB/s and
# resumed. At most two files download at once (more connections did not add
# bandwidth from Hugging Face's CDN). Then every file is hashed once against
# its pin (a file already hashed at its size and mtime is skipped), and
# izwi-artifact.json is written as izwi's own downloader writes it after the
# same check, so izwi reads the bundle as downloaded and never hashes the
# 41.5 GB again. Optional env: HF_TOKEN, FD3_FETCH_MAX_ATTEMPTS (200),
# FD3_FETCH_STALL_SECS (120).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# name size sha256 at 78e5ab79, small files first. 24 files, 41,502,569,692
# bytes; `.gitattributes` is not part of the bundle.
FILES="
config.json 5534 aea53f0f2b9a15023fc0f8eb0f42360663671c99814037bada6867d508010111
chat_template.jinja 7756 a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715
README.md 7996 6d147e079b87e0b9bb4e1c899f75da3e658754933457a3ca753ca9912d1f5e10
LICENSE 19685 733a32442087acbda12ecafe8e71eb8ee5890c89b50ce7ac4ea3777faa3c823e
special_tokens_map.json 102779 08aed9e321dc05275ac68cf883bfa96006b3fc09ff5cccadf262b6bce0873adb
tokenizer_config.json 103580 00e800279d8436c324b924e97872c7eacd2b19406aa0fe71eeb557b4cd53aa3e
model.safetensors.index.json 6880196 a8e9aca4e9fddca51f03fad63f52c82c470e98e28b6b0b1b1562dbe6456b50bb
tokenizer.json 20791070 0e9b9db76b2f0389b04b6ef057a2ac3b89100b7e96f2dd7034d19580a09654d6
codec.pth 1871099728 74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f
model-00001-of-00015.safetensors 2862881576 a1b7af15210d2eda2d75a2cf976295f80e345855b745b109aa3ce49d26a44497
model-00002-of-00015.safetensors 2628610248 a95e113c6f32f8fcb23ecb618cd4384fad6358f3c0ac31cfbf352daf5370db94
model-00003-of-00015.safetensors 2628610224 a34d0952d6369f5c73e6ae06138ef96bc6a3af9bb259c4e41425a33706cf5b68
model-00004-of-00015.safetensors 2628611792 13ac368d87441a18752e424e485b711a115ea1084e3a18ab13276e73eedef8b4
model-00005-of-00015.safetensors 2641525264 c432fa0dac97471f0d625bc67eb94fe084ca2e5ecd8865f457b751479b50e363
model-00006-of-00015.safetensors 2628614896 661a31433747fa9d029632a61350ea9eb105bf7276d013ff0901b6a939fb2dcb
model-00007-of-00015.safetensors 2628614880 4a7574b5c238f424b37aa945a736da1994a26bd38b86bf3ec4abf524c0969cc8
model-00008-of-00015.safetensors 2628614888 7fac6d1e899939eb03c9cf28a33b0bfd9c7a599298662f26d9a59d64c96d007f
model-00009-of-00015.safetensors 2641525264 fcf1313744ea5978e363b2bb49c3b66748d886d79f3ec102f4346a462385de4b
model-00010-of-00015.safetensors 2628614896 6db3f7cd9d49806ae4f17b6f6d17a0c8dd9686aee57728da1b50ec0c6139a629
model-00011-of-00015.safetensors 2628614880 a9027ef4db30bb1875bd2bc93b80bae785c5943807efe63bf09834b66e49fc16
model-00012-of-00015.safetensors 2628614888 c16fea955e973010064ff7bc907a6fa67821956e0de23f74807c31fb48339f3a
model-00013-of-00015.safetensors 2641525264 64861c8e1ee13a0cbb5ec9431f402cee33edcbcbc744ff63091fe3a997f0da19
model-00014-of-00015.safetensors 3749111392 ba5319475a6258984143b69af8380f166df8179af90378f2f4e02694cad87baf
model-00015-of-00015.safetensors 1409461016 1e2ddf01c1be73fb19f3022c28e90d24e730c1bc759e5de78083afb39aab3fb7
"
# What izwi's downloader writes once the bundle hashed to its pins
# (ARTIFACT_MANIFEST_FILE in crates/izwi-core/src/artifacts/downloader.rs).
MANIFEST="izwi-artifact.json"
# name size mtime of every file already hashed to its pin.
HASHED=".fd3-fetch-hashed"

fd3_mtime() {
    stat -c %Y "$1" 2>/dev/null || stat -f %m "$1"
}

fetch_one() {
    local dest="$1" name="$2" size="$3"
    local path="$dest/$name"
    local url="https://huggingface.co/$FD3_REPO_ID/resolve/$FD3_REVISION/$name"
    local attempt=0 have
    mkdir -p "$(dirname "$path")"
    while true; do
        have=0
        if [[ -f "$path" ]]; then have="$(fd3_file_size "$path")"; fi
        if ((have == size)); then
            fd3_log "ok   $name ($size bytes)"
            return 0
        fi
        if ((have > size)); then
            fd3_log "$name has $have bytes, more than the pinned $size; starting it again"
            rm -f "$path"
            have=0
        fi
        attempt=$((attempt + 1))
        if ((attempt > ${FD3_FETCH_MAX_ATTEMPTS:-200})); then
            fd3_die "$name: giving up after $((attempt - 1)) attempts at $have/$size bytes"
        fi
        fd3_log "get  $name from byte $have of $size (attempt $attempt)"
        local auth=()
        if [[ -n "${HF_TOKEN:-}" ]]; then auth=(-H "Authorization: Bearer $HF_TOKEN"); fi
        if ! curl -L --fail --silent --show-error -C - --connect-timeout 30 \
            --speed-time "${FD3_FETCH_STALL_SECS:-120}" --speed-limit 10240 \
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

DEST="$FD3_MODEL_DIR"
JOBS="${FD3_FETCH_JOBS:-2}"
VERIFY_ONLY=0
DRY_RUN=0
while (($#)); do
    case "$1" in
        --dest) DEST="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --verify-only) VERIFY_ONLY=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h | --help) sed -n '2,20p' "$0"; exit 0 ;;
        *) fd3_die "unknown argument $1" ;;
    esac
done
if ((JOBS < 1 || JOBS > 2)); then fd3_die "--jobs must be 1 or 2"; fi

remaining=0
bad=0
total=0
while read -r name size _sha; do
    [[ -z "$name" ]] && continue
    total=$((total + size))
    have=0
    if [[ -f "$DEST/$name" ]]; then have="$(fd3_file_size "$DEST/$name")"; fi
    if ((have != size)); then
        bad=$((bad + 1))
        if ((have < size)); then remaining=$((remaining + size - have)); else remaining=$((remaining + size)); fi
        if ((DRY_RUN)); then fd3_log "need $name: $have of $size bytes"; fi
    fi
done <<<"$FILES"

if ((VERIFY_ONLY)); then
    if ((bad)); then fd3_die "$bad files are missing or not at their pinned size in $DEST; run fetch.sh"; fi
    [[ -f "$DEST/$MANIFEST" ]] ||
        fd3_die "$DEST has every file but was never hashed against its pins; run fetch.sh"
    fd3_log "all 24 files at their pinned sizes and hashed in $DEST (revision ${FD3_REVISION:0:8})"
    exit 0
fi

need_gb=$(((remaining >> 30) + 2))
if ((DRY_RUN)); then
    fd3_log "dry run: $FD3_REPO_ID@$FD3_REVISION into $DEST"
    fd3_log "  $bad of 24 files to fetch, $((remaining >> 20)) MiB of $((total >> 20)) MiB to go"
    fd3_log "  needs ${need_gb} GB free; $(fd3_free_gb "$DEST") GB free at $DEST"
    fd3_log "  then hashes every file once and writes $DEST/$MANIFEST"
    exit 0
fi

mkdir -p "$DEST"
if ((bad)); then
    fd3_log "downloading $FD3_REPO_ID@$FD3_REVISION into $DEST: $((remaining >> 20)) MiB to go"
    fd3_require_free_gb "$DEST" "$need_gb"
    # Small files one at a time, then the large ones JOBS at a time.
    echo "$FILES" | awk 'NF == 3 && $2 < 100000000' | while read -r name size _sha; do
        fetch_one "$DEST" "$name" "$size"
    done
    echo "$FILES" | awk 'NF == 3 && $2 >= 100000000 { print $1, $2 }' |
        xargs -n 2 -P "$JOBS" bash "$0" --one "$DEST"
fi

# Every file at its exact size, then its SHA-256 once.
touch "$DEST/$HASHED"
mismatched=0
while read -r name size sha; do
    [[ -z "$name" ]] && continue
    path="$DEST/$name"
    [[ -f "$path" && "$(fd3_file_size "$path")" == "$size" ]] ||
        fd3_die "$name is not at its pinned size $size after the download; run fetch.sh again"
    stamp="$name $size $(fd3_mtime "$path")"
    if grep -qxF "$stamp" "$DEST/$HASHED"; then
        continue
    fi
    fd3_log "hash $name"
    digest="$(fd3_sha256 "$path")"
    if [[ "$digest" != "$sha" ]]; then
        fd3_log "$name hashes to $digest; the pin has $sha. Deleted it: run fetch.sh again"
        rm -f "$path"
        mismatched=$((mismatched + 1))
        continue
    fi
    echo "$stamp" >>"$DEST/$HASHED"
done <<<"$FILES"
if ((mismatched)); then
    rm -f "$DEST/$MANIFEST"
    fd3_die "$mismatched files did not match their pins"
fi

"$(fd3_python)" - "$DEST/$MANIFEST" "$FD3_VARIANT" "$FD3_REPO_ID" "$FD3_REVISION" <<PY
import json, sys
path, variant, repo, revision = sys.argv[1:5]
files = """$FILES""".split()[0::3]
manifest = {"schema_version": 1, "variant": variant, "repo_id": repo, "revision": revision,
            "files": files}
with open(path + ".part", "w") as handle:
    handle.write(json.dumps(manifest, indent=2))
import os
os.replace(path + ".part", path)
PY
bash "$0" --dest "$DEST" --verify-only
fd3_log "izwi reads it from IZWI_MODELS_DIR=$(dirname "$DEST") as $FD3_VARIANT"
