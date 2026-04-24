#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/release/verify-unified-runtime.sh [options]

Verifies the unified CPU+CUDA release layout for Linux release assets while
preserving stable public binary names.

Options:
  --release-dir <dir>             Public release directory (default: target/release)
  --skip-config-check             Skip Tauri release config validation
  --skip-cuda-compiled-check      Skip private CLI "Compiled Backends" CUDA check
  --skip-cuda-library-check       Skip required packaged CUDA library checks
  -h, --help                      Show this help
EOF
}

release_dir="target/release"
skip_config_check=0
skip_cuda_compiled_check=0
skip_cuda_library_check=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --release-dir)
            release_dir="${2:?missing value for --release-dir}"
            shift
            ;;
        --skip-config-check)
            skip_config_check=1
            ;;
        --skip-cuda-compiled-check)
            skip_cuda_compiled_check=1
            ;;
        --skip-cuda-library-check)
            skip_cuda_library_check=1
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

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
audit_script="${script_dir}/audit-runtime-deps.sh"

if [[ "${release_dir}" != /* ]]; then
    release_dir="${PWD}/${release_dir}"
fi

runtime_dir="${release_dir}/runtime/cuda"
public_bins=(izwi izwi-server)
forbidden_bins=(izwi-cuda izwi-server-cuda)
cuda_pattern='(libcuda|libcudart|libcublas|libcublasLt|libcurand|libnvrtc|libcudnn|nvcuda|cudart|cublas|curand|nvrtc|cudnn)'

fail() {
    echo "error: $*" >&2
    exit 1
}

have_command() {
    command -v "$1" >/dev/null 2>&1
}

require_executable() {
    local path="$1"
    [[ -f "${path}" ]] || fail "missing file: ${path}"
    [[ -x "${path}" ]] || fail "file is not executable: ${path}"
}

assert_no_forbidden_public_names() {
    for bin in "${forbidden_bins[@]}"; do
        [[ ! -e "${release_dir}/${bin}" ]] || fail "public CUDA-suffixed binary is not allowed: ${release_dir}/${bin}"
        [[ ! -e "${runtime_dir}/${bin}" ]] || fail "private runtime must keep the public basename, found: ${runtime_dir}/${bin}"
    done
}

compiled_backend_lines() {
    local binary="$1"
    "${binary}" version --full | awk '
        /^Compiled Backends:/ { in_section = 1; next }
        in_section && NF == 0 { exit }
        in_section { print }
    '
}

compiled_section_has_cuda() {
    grep -Eiq '(^|[^[:alpha:]])CUDA([^[:alpha:]]|$)'
}

assert_cli_backend_contract() {
    local public_cli="${release_dir}/izwi"
    local private_cli="${runtime_dir}/izwi"
    local public_backends

    public_backends="$(compiled_backend_lines "${public_cli}")"
    echo "Public CLI compiled backends:"
    printf '%s\n' "${public_backends}" | sed 's/^/  /'

    if printf '%s\n' "${public_backends}" | compiled_section_has_cuda; then
        fail "public izwi binary is CUDA-compiled; public entrypoints must remain CPU-safe"
    fi

    if [[ "${skip_cuda_compiled_check}" -eq 1 ]]; then
        echo "Skipping private CLI CUDA compiled-backend check."
        return
    fi

    local private_backends
    private_backends="$(compiled_backend_lines "${private_cli}")"
    echo "Private CUDA CLI compiled backends:"
    printf '%s\n' "${private_backends}" | sed 's/^/  /'

    if ! printf '%s\n' "${private_backends}" | compiled_section_has_cuda; then
        fail "private runtime izwi binary does not report CUDA in compiled backends"
    fi
}

assert_public_not_cuda_linked() {
    local binary="$1"
    local deps=""

    case "$(uname -s)" in
        Linux)
            if have_command readelf; then
                deps="$(readelf -d "${binary}" 2>&1 || true)"
            elif have_command ldd; then
                deps="$(ldd "${binary}" 2>&1 || true)"
            else
                echo "No readelf or ldd available; skipping direct CUDA dependency check for ${binary}."
                return
            fi
            ;;
        Darwin)
            if have_command otool; then
                deps="$(otool -L "${binary}" 2>&1 || true)"
            else
                echo "otool is not available; skipping direct CUDA dependency check for ${binary}."
                return
            fi
            ;;
        *)
            echo "No direct CUDA dependency check for $(uname -s)."
            return
            ;;
    esac

    if printf '%s\n' "${deps}" | grep -Eiq "${cuda_pattern}"; then
        printf '%s\n' "${deps}" | grep -Ei "${cuda_pattern}" | sed 's/^/  /'
        fail "public binary has loader-visible CUDA dependencies: ${binary}"
    fi
}

assert_required_cuda_libraries() {
    if [[ "${skip_cuda_library_check}" -eq 1 ]]; then
        echo "Skipping packaged CUDA library check."
        return
    fi

    if [[ "$(uname -s)" != "Linux" ]]; then
        echo "Packaged CUDA library check is Linux-only in this script."
        return
    fi

    local patterns=(
        "libcudart.so*"
        "libcublas.so*"
        "libcublasLt.so*"
        "libcurand.so*"
        "libnvrtc.so*"
        "libnvrtc-builtins.so*"
    )

    shopt -s nullglob
    for pattern in "${patterns[@]}"; do
        local matches=("${runtime_dir}"/${pattern})
        if [[ ${#matches[@]} -eq 0 ]]; then
            fail "missing packaged CUDA runtime library matching ${pattern} in ${runtime_dir}"
        fi
    done
}

validate_tauri_linux_config() {
    if [[ "${skip_config_check}" -eq 1 ]]; then
        echo "Skipping Tauri release config check."
        return
    fi

    local config="${repo_root}/crates/izwi-desktop/tauri.release.linux.conf.json"
    [[ -f "${config}" ]] || fail "missing Linux Tauri release config: ${config}"

    if ! have_command node; then
        echo "node is not available; skipping structured Tauri config validation."
        return
    fi

    node - "${config}" <<'NODE'
const fs = require('node:fs');
const configPath = process.argv[2];
const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
const resources = config.bundle?.resources ?? {};
const debFiles = config.bundle?.linux?.deb?.files ?? {};

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

assertEqual(resources['../../target/release/izwi'], 'bin/izwi', 'Linux resource izwi');
assertEqual(resources['../../target/release/izwi-server'], 'bin/izwi-server', 'Linux resource izwi-server');
assertEqual(resources['../../target/release/runtime'], 'bin/runtime', 'Linux resource runtime');
assertEqual(debFiles['/usr/bin/izwi'], '../../target/release/izwi', 'deb izwi');
assertEqual(debFiles['/usr/bin/izwi-server'], '../../target/release/izwi-server', 'deb izwi-server');
assertEqual(debFiles['/usr/lib/izwi/runtime'], '../../target/release/runtime', 'deb runtime');
NODE
}

[[ -d "${release_dir}" ]] || fail "release directory does not exist: ${release_dir}"
[[ -d "${runtime_dir}" ]] || fail "private CUDA runtime directory does not exist: ${runtime_dir}"
[[ -x "${audit_script}" ]] || fail "runtime dependency audit script is not executable: ${audit_script}"

for bin in "${public_bins[@]}"; do
    require_executable "${release_dir}/${bin}"
    require_executable "${runtime_dir}/${bin}"
done

assert_no_forbidden_public_names
assert_cli_backend_contract

for bin in "${public_bins[@]}"; do
    assert_public_not_cuda_linked "${release_dir}/${bin}"
done

assert_required_cuda_libraries
validate_tauri_linux_config

echo "Auditing public CPU-safe binaries:"
"${audit_script}" "${release_dir}/izwi" "${release_dir}/izwi-server"

echo "Auditing private CUDA runtime binaries:"
if [[ "$(uname -s)" == "Linux" ]]; then
    LD_LIBRARY_PATH="${runtime_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
        "${audit_script}" --allow-missing --skip-startup-probe "${runtime_dir}/izwi" "${runtime_dir}/izwi-server"
else
    "${audit_script}" --allow-missing --skip-startup-probe "${runtime_dir}/izwi" "${runtime_dir}/izwi-server"
fi

echo "Unified runtime verification passed for ${release_dir}"
