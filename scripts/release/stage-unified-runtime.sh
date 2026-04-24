#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/release/stage-unified-runtime.sh [options]

Stages private CUDA runtime binaries under the release directory while keeping
the public Izwi binary names unchanged.

Options:
  --release-dir <dir>       Public release directory (default: target/release)
  --cuda-target-dir <dir>   Cargo target dir used for CUDA build (default: target/cuda-release)
  --copy-cuda-libs          Copy redistributable CUDA runtime libraries into runtime/cuda
  --cuda-lib-dir <dir>      CUDA library directory to copy from
  -h, --help                Show this help
EOF
}

release_dir="target/release"
cuda_target_dir="target/cuda-release"
copy_cuda_libs=0
cuda_lib_dir=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --release-dir)
            release_dir="${2:?missing value for --release-dir}"
            shift
            ;;
        --cuda-target-dir)
            cuda_target_dir="${2:?missing value for --cuda-target-dir}"
            shift
            ;;
        --copy-cuda-libs)
            copy_cuda_libs=1
            ;;
        --cuda-lib-dir)
            cuda_lib_dir="${2:?missing value for --cuda-lib-dir}"
            shift
            ;;
        -h|--help|help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

cuda_release_dir="${cuda_target_dir}/release"
runtime_dir="${release_dir}/runtime/cuda"
required_bins=(izwi izwi-server)

for bin in "${required_bins[@]}"; do
    if [[ ! -x "${release_dir}/${bin}" ]]; then
        echo "Missing public release binary: ${release_dir}/${bin}" >&2
        exit 1
    fi
    if [[ ! -x "${cuda_release_dir}/${bin}" ]]; then
        echo "Missing CUDA runtime binary: ${cuda_release_dir}/${bin}" >&2
        exit 1
    fi
done

mkdir -p "${runtime_dir}"

for bin in "${required_bins[@]}"; do
    install -m 0755 "${cuda_release_dir}/${bin}" "${runtime_dir}/${bin}"
done

resolve_cuda_lib_dir() {
    local candidates=()

    if [[ -n "${cuda_lib_dir}" ]]; then
        candidates+=("${cuda_lib_dir}")
    fi
    if [[ -n "${CUDA_PATH:-}" ]]; then
        candidates+=("${CUDA_PATH}/lib64")
    fi
    if [[ -n "${CUDA_HOME:-}" ]]; then
        candidates+=("${CUDA_HOME}/lib64")
    fi
    candidates+=("/usr/local/cuda/lib64")

    for candidate in "${candidates[@]}"; do
        if [[ -d "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done

    return 1
}

if [[ "${copy_cuda_libs}" -eq 1 ]]; then
    source_lib_dir="$(resolve_cuda_lib_dir)" || {
        echo "Could not find CUDA library directory. Pass --cuda-lib-dir or set CUDA_PATH." >&2
        exit 1
    }

    shopt -s nullglob
    cuda_patterns=(
        "libcudart.so*"
        "libcublas.so*"
        "libcublasLt.so*"
        "libcurand.so*"
        "libnvrtc.so*"
        "libnvrtc-builtins.so*"
    )

    copied=0
    for pattern in "${cuda_patterns[@]}"; do
        matches=("${source_lib_dir}"/${pattern})
        for match in "${matches[@]}"; do
            cp -a "${match}" "${runtime_dir}/"
            copied=1
        done
    done

    if [[ "${copied}" -eq 0 ]]; then
        echo "No CUDA runtime libraries matched in ${source_lib_dir}" >&2
        exit 1
    fi
fi

echo "Staged unified runtime assets in ${runtime_dir}"
