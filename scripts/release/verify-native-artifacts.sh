#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/release/verify-native-artifacts.sh [options]

Verifies native release artifacts are CPU-only and stay within a safe size
limit before upload/publish.

Options:
  --artifact-dir <dir>       Verify every file directly under a release dir.
  --terminal-archive <path>  Verify the CLI tar.gz archive.
  --zip <path>               Verify a CLI zip archive.
  --deb <path>               Verify the Tauri .deb payload.
  --appimage <path>          Extract and verify the Tauri AppImage payload.
  --max-bytes <bytes>        Maximum allowed file size (default: 1073741824).
  -h, --help                 Show this help.
EOF
}

artifact_dir=""
terminal_archive=""
zip_path=""
deb_path=""
appimage_path=""
max_bytes="${IZWI_NATIVE_ARTIFACT_MAX_BYTES:-1073741824}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --artifact-dir)
            artifact_dir="${2:?missing value for --artifact-dir}"
            shift
            ;;
        --terminal-archive)
            terminal_archive="${2:?missing value for --terminal-archive}"
            shift
            ;;
        --zip)
            zip_path="${2:?missing value for --zip}"
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
        --max-bytes)
            max_bytes="${2:?missing value for --max-bytes}"
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

if [[ -z "${artifact_dir}" && -z "${terminal_archive}" && -z "${zip_path}" && -z "${deb_path}" && -z "${appimage_path}" ]]; then
    usage >&2
    exit 1
fi

forbidden_pattern='runtime/cuda|libcuda\.so|libcudart\.so|libcublas(Lt)?\.so|libcurand\.so|libnvrtc(-builtins)?\.so|nvcuda\.dll|cudart64_|cublas(Lt)?64_|curand64_|nvrtc(-builtins)?64_'

fail() {
    echo "error: $*" >&2
    exit 1
}

require_file() {
    local path="$1"
    [[ -f "${path}" ]] || fail "missing artifact: ${path}"
}

file_size() {
    local path="$1"
    if stat -c %s "${path}" >/dev/null 2>&1; then
        stat -c %s "${path}"
    else
        stat -f %z "${path}"
    fi
}

assert_size_ok() {
    local path="$1"
    local size

    size="$(file_size "${path}")"
    if [[ "${size}" -gt "${max_bytes}" ]]; then
        fail "${path} is ${size} bytes; max allowed is ${max_bytes}"
    fi
}

assert_no_forbidden_entries() {
    local contents_file="$1"
    local label="$2"

    if grep -Eiq "${forbidden_pattern}" "${contents_file}"; then
        echo "Forbidden CUDA/native-runtime entries in ${label}:" >&2
        grep -Ein "${forbidden_pattern}" "${contents_file}" | sed 's/^/  /' >&2
        fail "${label} contains CUDA runtime payload"
    fi
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

verify_tar_archive() {
    local archive="$1"
    local contents_file

    require_file "${archive}"
    assert_size_ok "${archive}"
    contents_file="$(mktemp)"
    tar -tzf "${archive}" | sed 's#^\./##' > "${contents_file}"
    assert_no_forbidden_entries "${contents_file}" "${archive}"

    case "$(basename "${archive}")" in
        izwi-cli-*.tar.gz)
            require_pattern "${contents_file}" '^izwi$' "public CLI"
            require_pattern "${contents_file}" '^izwi-server$' "public server"
            require_pattern "${contents_file}" '^izwi-desktop$' "desktop binary"
            ;;
    esac

    rm -f "${contents_file}"
    echo "Verified tar archive: ${archive}"
}

verify_zip() {
    local archive="$1"
    local contents_file

    require_file "${archive}"
    assert_size_ok "${archive}"
    command -v unzip >/dev/null 2>&1 || fail "unzip is required to inspect ${archive}"

    contents_file="$(mktemp)"
    unzip -Z1 "${archive}" | sed 's#^\./##' > "${contents_file}"
    assert_no_forbidden_entries "${contents_file}" "${archive}"

    case "$(basename "${archive}")" in
        izwi-cli-*.zip)
            require_pattern "${contents_file}" '(^|/)izwi\.exe$' "public CLI"
            require_pattern "${contents_file}" '(^|/)izwi-server\.exe$' "public server"
            require_pattern "${contents_file}" '(^|/)izwi-desktop\.exe$' "desktop binary"
            ;;
    esac

    rm -f "${contents_file}"
    echo "Verified zip archive: ${archive}"
}

verify_deb() {
    local deb="$1"
    local contents_file

    require_file "${deb}"
    assert_size_ok "${deb}"
    command -v dpkg-deb >/dev/null 2>&1 || fail "dpkg-deb is required to inspect ${deb}"

    contents_file="$(mktemp)"
    dpkg-deb --contents "${deb}" | awk '{print $NF}' | sed 's#^\./##' > "${contents_file}"
    assert_no_forbidden_entries "${contents_file}" "${deb}"
    require_pattern "${contents_file}" '^usr/bin/izwi$' "deb public CLI"
    require_pattern "${contents_file}" '^usr/bin/izwi-server$' "deb public server"

    rm -f "${contents_file}"
    echo "Verified deb payload: ${deb}"
}

verify_appimage() {
    local appimage="$1"
    local appimage_abs
    local extract_dir
    local contents_file

    require_file "${appimage}"
    assert_size_ok "${appimage}"
    chmod +x "${appimage}"
    appimage_abs="$(cd "$(dirname "${appimage}")" && pwd)/$(basename "${appimage}")"
    extract_dir="$(mktemp -d)"
    contents_file="$(mktemp)"

    (
        cd "${extract_dir}"
        APPIMAGE_EXTRACT_AND_RUN=1 "${appimage_abs}" --appimage-extract >/dev/null
    )

    find "${extract_dir}/squashfs-root" -type f | sed "s#^${extract_dir}/squashfs-root/##" > "${contents_file}"
    assert_no_forbidden_entries "${contents_file}" "${appimage}"
    require_pattern "${contents_file}" '(^|/)bin/izwi$' "AppImage public CLI resource"
    require_pattern "${contents_file}" '(^|/)bin/izwi-server$' "AppImage public server resource"

    rm -rf "${extract_dir}"
    rm -f "${contents_file}"
    echo "Verified AppImage payload: ${appimage}"
}

verify_file() {
    local path="$1"
    local name

    require_file "${path}"
    assert_size_ok "${path}"
    name="$(basename "${path}")"

    case "${name}" in
        *.AppImage)
            verify_appimage "${path}"
            ;;
        *.deb)
            verify_deb "${path}"
            ;;
        *.zip)
            verify_zip "${path}"
            ;;
        *.tar.gz)
            verify_tar_archive "${path}"
            ;;
        *)
            echo "Verified size-only artifact: ${path}"
            ;;
    esac
}

if [[ -n "${artifact_dir}" ]]; then
    [[ -d "${artifact_dir}" ]] || fail "missing artifact directory: ${artifact_dir}"
    while IFS= read -r -d '' path; do
        verify_file "${path}"
    done < <(find "${artifact_dir}" -maxdepth 1 -type f -print0)
fi

if [[ -n "${terminal_archive}" ]]; then
    verify_tar_archive "${terminal_archive}"
fi

if [[ -n "${zip_path}" ]]; then
    verify_zip "${zip_path}"
fi

if [[ -n "${deb_path}" ]]; then
    verify_deb "${deb_path}"
fi

if [[ -n "${appimage_path}" ]]; then
    verify_appimage "${appimage_path}"
fi
