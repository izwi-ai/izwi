#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/release/verify-packaged-artifacts.sh [options]

Verifies final Linux release artifacts contain the unified CPU+CUDA runtime
layout after packaging, not only in target/release.

Options:
  --terminal-archive <path>  Verify the terminal tar.gz archive.
  --deb <path>               Verify the Tauri .deb payload.
  --appimage <path>          Extract and verify the Tauri AppImage payload.
  -h, --help                 Show this help.
EOF
}

terminal_archive=""
deb_path=""
appimage_path=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --terminal-archive)
            terminal_archive="${2:?missing value for --terminal-archive}"
            shift
            ;;
        --deb)
            deb_path="${2:?missing value for --deb}"
            shift
            ;;
        --appimage)
            appimage_path="${2:?missing value for --appimage}"
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

if [[ -z "${terminal_archive}" && -z "${deb_path}" && -z "${appimage_path}" ]]; then
    usage >&2
    exit 1
fi

fail() {
    echo "error: $*" >&2
    exit 1
}

require_file() {
    local path="$1"
    [[ -f "${path}" ]] || fail "missing artifact: ${path}"
}

require_pattern() {
    local contents_file="$1"
    local pattern="$2"
    local label="$3"

    if ! grep -Eq "${pattern}" "${contents_file}"; then
        echo "Artifact contents:" >&2
        sed 's/^/  /' "${contents_file}" >&2
        fail "missing ${label}"
    fi
}

require_unified_runtime_patterns() {
    local contents_file="$1"
    local root_pattern="$2"

    require_pattern "${contents_file}" "${root_pattern}runtime/cuda/izwi$" "private CUDA CLI"
    require_pattern "${contents_file}" "${root_pattern}runtime/cuda/izwi-server$" "private CUDA server"
    require_pattern "${contents_file}" "${root_pattern}runtime/cuda/libcudart\\.so" "CUDA runtime library"
    require_pattern "${contents_file}" "${root_pattern}runtime/cuda/libcublas\\.so" "cuBLAS runtime library"
    require_pattern "${contents_file}" "${root_pattern}runtime/cuda/libcurand\\.so" "cuRAND runtime library"
    require_pattern "${contents_file}" "${root_pattern}runtime/cuda/libnvrtc\\.so" "NVRTC runtime library"
}

verify_terminal_archive() {
    local archive="$1"
    local contents_file

    require_file "${archive}"
    contents_file="$(mktemp)"
    tar -tzf "${archive}" | sed 's#^\./##' > "${contents_file}"

    require_pattern "${contents_file}" '^izwi$' "public CLI"
    require_pattern "${contents_file}" '^izwi-server$' "public server"
    require_pattern "${contents_file}" '^izwi-desktop$' "desktop binary"
    require_unified_runtime_patterns "${contents_file}" '^'

    rm -f "${contents_file}"
    echo "Verified terminal archive: ${archive}"
}

verify_deb() {
    local deb="$1"
    local contents_file

    require_file "${deb}"
    command -v dpkg-deb >/dev/null 2>&1 || fail "dpkg-deb is required to inspect ${deb}"

    contents_file="$(mktemp)"
    dpkg-deb --contents "${deb}" | awk '{print $NF}' | sed 's#^\./##' > "${contents_file}"

    require_pattern "${contents_file}" '^usr/bin/izwi$' "deb public CLI"
    require_pattern "${contents_file}" '^usr/bin/izwi-server$' "deb public server"
    require_unified_runtime_patterns "${contents_file}" '^usr/lib/izwi/'

    rm -f "${contents_file}"
    echo "Verified deb payload: ${deb}"
}

verify_appimage() {
    local appimage="$1"
    local appimage_abs
    local extract_dir
    local contents_file

    require_file "${appimage}"
    chmod +x "${appimage}"
    appimage_abs="$(cd "$(dirname "${appimage}")" && pwd)/$(basename "${appimage}")"
    extract_dir="$(mktemp -d)"
    contents_file="$(mktemp)"

    (
        cd "${extract_dir}"
        APPIMAGE_EXTRACT_AND_RUN=1 "${appimage_abs}" --appimage-extract >/dev/null
    )

    find "${extract_dir}/squashfs-root" -type f | sed "s#^${extract_dir}/squashfs-root/##" > "${contents_file}"

    require_pattern "${contents_file}" '(^|/)bin/izwi$' "AppImage public CLI resource"
    require_pattern "${contents_file}" '(^|/)bin/izwi-server$' "AppImage public server resource"
    require_unified_runtime_patterns "${contents_file}" '(^|/)bin/'

    rm -rf "${extract_dir}"
    rm -f "${contents_file}"
    echo "Verified AppImage payload: ${appimage}"
}

if [[ -n "${terminal_archive}" ]]; then
    verify_terminal_archive "${terminal_archive}"
fi

if [[ -n "${deb_path}" ]]; then
    verify_deb "${deb_path}"
fi

if [[ -n "${appimage_path}" ]]; then
    verify_appimage "${appimage_path}"
fi
