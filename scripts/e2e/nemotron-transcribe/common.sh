#!/usr/bin/env bash
# Shared settings for the Nemotron 3.5 Transcribe hand-off kit. Sourced by
# fetch.sh, build.sh, run-parity.sh, run-e2e.sh, run-load.sh and collect.sh.
# Bash 3.2 compatible (macOS /bin/bash): no associative arrays, no `wait -n`.

NT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$NT_DIR/../../.." && pwd)"

NT_VARIANT="Nemotron-3.5-Transcribe-30B-A4B-FP8"
NT_REPO_ID="nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8"
NT_REVISION="006d73cd8385237ed5603fc6c7e76200e50abbd2"

# izwi's default models directory (`dirs::data_local_dir()/izwi/models`).
nt_default_models_dir() {
    if [[ -n "${IZWI_MODELS_DIR:-}" ]]; then
        echo "$IZWI_MODELS_DIR"
    elif [[ "$(uname -s)" == "Darwin" ]]; then
        echo "$HOME/Library/Application Support/izwi/models"
    else
        echo "${XDG_DATA_HOME:-$HOME/.local/share}/izwi/models"
    fi
}

NT_MODELS_DIR="${NT_MODELS_DIR:-$(nt_default_models_dir)}"
NT_MODEL_DIR="${NT_MODEL_DIR:-$NT_MODELS_DIR/$NT_VARIANT}"
# Everything the kit writes (binaries, cargo target, reports) lives here.
NT_OUT_ROOT="${NT_OUT_ROOT:-$REPO_ROOT/target/nemotron-transcribe}"
NT_FIXTURE_DIR="$REPO_ROOT/crates/izwi-core/tests/fixtures/nemotron_transcribe"

nt_log() {
    printf '[nemotron-transcribe] %s\n' "$*" >&2
}

nt_die() {
    nt_log "error: $*"
    exit 1
}

nt_check_backend() {
    case "${1:-}" in
        cpu | metal | cuda) ;;
        *)
            echo "backend must be one of: cpu, metal, cuda" >&2
            exit 2
            ;;
    esac
}

nt_backend_out() {
    echo "$NT_OUT_ROOT/$1"
}

# A private target directory per checkout and backend (lessons: never share
# one CARGO_TARGET_DIR between checkouts or with another session).
nt_target_dir() {
    echo "${NT_CARGO_TARGET_DIR:-$NT_OUT_ROOT/cargo-target-$1}"
}

nt_file_size() {
    stat -c %s "$1" 2>/dev/null || stat -f %z "$1"
}

nt_free_gb() {
    local path="$1"
    while [[ ! -e "$path" ]]; do path="$(dirname "$path")"; done
    df -Pk "$path" | awk 'NR == 2 { printf "%d", $4 / 1048576 }'
}

nt_require_free_gb() {
    local path="$1" need="$2" have
    have="$(nt_free_gb "$path")"
    if ((have < need)); then
        nt_die "only ${have} GB free at ${path}; need ${need} GB (df lesson: check before builds and downloads)"
    fi
    nt_log "${have} GB free at ${path}"
}

nt_git_sha() {
    git -C "$REPO_ROOT" rev-parse HEAD
}

nt_git_dirty() {
    if [[ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no)" ]]; then
        echo true
    else
        echo false
    fi
}

nt_sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{ print $1 }'
    else
        shasum -a 256 "$1" | awk '{ print $1 }'
    fi
}

# izwi-core features for a backend (cargo test of the family).
nt_core_features() {
    case "$1" in
        cpu) echo "" ;;
        metal) echo "${NT_CORE_FEATURES:-metal}" ;;
        cuda) echo "${NT_CORE_FEATURES:-cuda}" ;;
    esac
}

# izwi-server features for a backend (the E2E binary).
nt_server_features() {
    case "$1" in
        cpu) echo "" ;;
        metal) echo "${NT_SERVER_FEATURES:-metal}" ;;
        cuda) echo "${NT_SERVER_FEATURES:-cuda}" ;;
    esac
}

nt_cuda_compute_cap() {
    if [[ -n "${CUDA_COMPUTE_CAP:-}" ]]; then
        echo "$CUDA_COMPUTE_CAP"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' .'
    else
        echo "80"
    fi
}

nt_python() {
    if [[ -n "${NT_PYTHON:-}" ]]; then
        echo "$NT_PYTHON"
    else
        command -v python3
    fi
}
