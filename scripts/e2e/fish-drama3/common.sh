#!/usr/bin/env bash
# Shared settings for the Fish Audio Drama 3 hand-off kit. Sourced by
# fetch.sh, build.sh, run-parity.sh, run-e2e.sh, run-load.sh and collect.sh.
# Bash 3.2 compatible (macOS /bin/bash): no associative arrays, no `wait -n`.

FD3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$FD3_DIR/../../.." && pwd)"

FD3_VARIANT="FishAudio-Drama-3"
FD3_REPO_ID="fishaudio/drama-3"
FD3_REVISION="78e5ab79e49aa9a25abe8799eff2c4d73a1c3216"

# izwi's default models directory (`dirs::data_local_dir()/izwi/models`).
fd3_default_models_dir() {
    if [[ -n "${IZWI_MODELS_DIR:-}" ]]; then
        echo "$IZWI_MODELS_DIR"
    elif [[ "$(uname -s)" == "Darwin" ]]; then
        echo "$HOME/Library/Application Support/izwi/models"
    else
        echo "${XDG_DATA_HOME:-$HOME/.local/share}/izwi/models"
    fi
}

FD3_MODELS_DIR="${FD3_MODELS_DIR:-$(fd3_default_models_dir)}"
FD3_MODEL_DIR="${FD3_MODEL_DIR:-$FD3_MODELS_DIR/$FD3_VARIANT}"
# Everything the kit writes (binaries, cargo target, reports) lives here.
FD3_OUT_ROOT="${FD3_OUT_ROOT:-$REPO_ROOT/target/fish-drama3}"
FD3_FIXTURE_DIR="$REPO_ROOT/crates/izwi-core/tests/fixtures/fish_drama3"
# Where the izwi-server test `write_the_e2e_kit_synthetic_checkpoint` writes
# the synthetic checkpoint (a models directory; see run-e2e.sh --synthetic).
FD3_SYNTHETIC_MODELS_DIR="$REPO_ROOT/target/fish-drama3/synthetic-models"

fd3_log() {
    printf '[fish-drama3] %s\n' "$*" >&2
}

fd3_die() {
    fd3_log "error: $*"
    exit 1
}

fd3_check_backend() {
    case "${1:-}" in
        cpu | metal | cuda) ;;
        *)
            echo "backend must be one of: cpu, metal, cuda" >&2
            exit 2
            ;;
    esac
}

# The output directory of a backend; `synthetic` runs keep their own.
fd3_backend_out() {
    if [[ "${2:-0}" == "1" ]]; then
        echo "$FD3_OUT_ROOT/$1-synthetic"
    else
        echo "$FD3_OUT_ROOT/$1"
    fi
}

# A private target directory per checkout and backend (lessons: never share
# one CARGO_TARGET_DIR between checkouts or with another session).
fd3_target_dir() {
    echo "${FD3_CARGO_TARGET_DIR:-$FD3_OUT_ROOT/cargo-target-$1}"
}

fd3_file_size() {
    stat -c %s "$1" 2>/dev/null || stat -f %z "$1"
}

fd3_free_gb() {
    local path="$1"
    while [[ ! -e "$path" ]]; do path="$(dirname "$path")"; done
    df -Pk "$path" | awk 'NR == 2 { printf "%d", $4 / 1048576 }'
}

fd3_require_free_gb() {
    local path="$1" need="$2" have
    have="$(fd3_free_gb "$path")"
    if ((have < need)); then
        fd3_die "only ${have} GB free at ${path}; need ${need} GB (check df before builds and downloads)"
    fi
    fd3_log "${have} GB free at ${path}"
}

fd3_git_sha() {
    git -C "$REPO_ROOT" rev-parse HEAD
}

fd3_git_dirty() {
    if [[ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no)" ]]; then
        echo true
    else
        echo false
    fi
}

fd3_sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{ print $1 }'
    else
        shasum -a 256 "$1" | awk '{ print $1 }'
    fi
}

# izwi-core features for a backend (cargo test of the family).
fd3_core_features() {
    case "$1" in
        cpu) echo "${FD3_CORE_FEATURES:-}" ;;
        metal) echo "${FD3_CORE_FEATURES:-metal}" ;;
        cuda) echo "${FD3_CORE_FEATURES:-cuda}" ;;
    esac
}

# izwi-server features for a backend (the E2E binary).
fd3_server_features() {
    case "$1" in
        cpu) echo "${FD3_SERVER_FEATURES:-}" ;;
        metal) echo "${FD3_SERVER_FEATURES:-metal}" ;;
        cuda) echo "${FD3_SERVER_FEATURES:-cuda}" ;;
    esac
}

fd3_cuda_compute_cap() {
    if [[ -n "${CUDA_COMPUTE_CAP:-}" ]]; then
        echo "$CUDA_COMPUTE_CAP"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' .'
    else
        echo "80"
    fi
}

fd3_python() {
    if [[ -n "${FD3_PYTHON:-}" ]]; then
        echo "$FD3_PYTHON"
    else
        command -v python3
    fi
}

# Prints a command as a shell would read it (for dry runs and logs).
fd3_show() {
    local word out=""
    for word in "$@"; do
        out+="$(printf '%q' "$word") "
    done
    printf '%s\n' "${out% }"
}
