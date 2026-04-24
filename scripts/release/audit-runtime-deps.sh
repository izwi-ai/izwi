#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/release/audit-runtime-deps.sh [--allow-missing] <binary>...

Audits release binaries for loader-visible runtime dependencies and verifies
that each binary can at least enter its version/help path. This is intended for
CPU-only and CUDA-host smoke checks after building release artifacts.

Options:
  --allow-missing   Report missing shared libraries without failing the audit.
  -h, --help        Show this help.
EOF
}

allow_missing=0
binaries=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --allow-missing)
            allow_missing=1
            ;;
        -h|--help|help)
            usage
            exit 0
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do
                binaries+=("$1")
                shift
            done
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            binaries+=("$1")
            ;;
    esac
    shift
done

if [[ ${#binaries[@]} -eq 0 ]]; then
    usage >&2
    exit 1
fi

cuda_pattern='(libcuda|libcudart|libcublas|libcublasLt|libcurand|libnvrtc|libcudnn|nvcuda|cudart|cublas|curand|nvrtc|cudnn)'
failed=0

have_command() {
    command -v "$1" >/dev/null 2>&1
}

probe_startup() {
    local binary="$1"

    echo "Startup probe:"
    if IZWI_BACKEND=cpu "$binary" --version >/tmp/izwi-runtime-audit-version.out 2>/tmp/izwi-runtime-audit-version.err; then
        sed 's/^/  /' /tmp/izwi-runtime-audit-version.out
        return 0
    fi

    if IZWI_BACKEND=cpu "$binary" --help >/tmp/izwi-runtime-audit-help.out 2>/tmp/izwi-runtime-audit-help.err; then
        sed -n '1,8p' /tmp/izwi-runtime-audit-help.out | sed 's/^/  /'
        return 0
    fi

    echo "  failed to execute --version or --help"
    echo "  --version stderr:"
    sed 's/^/    /' /tmp/izwi-runtime-audit-version.err || true
    echo "  --help stderr:"
    sed 's/^/    /' /tmp/izwi-runtime-audit-help.err || true
    return 1
}

audit_linux() {
    local binary="$1"
    local ldd_output

    if have_command readelf; then
        echo "ELF dynamic entries:"
        readelf -d "$binary" | grep -E 'NEEDED|RPATH|RUNPATH' | sed 's/^/  /' || true
    fi

    if ! have_command ldd; then
        echo "ldd is not available; skipping Linux loader check"
        return 0
    fi

    echo "ldd:"
    ldd_output="$(ldd "$binary" 2>&1 || true)"
    printf '%s\n' "$ldd_output" | sed 's/^/  /'

    echo "CUDA-linked libraries detected by loader:"
    printf '%s\n' "$ldd_output" | grep -Ei "$cuda_pattern" | sed 's/^/  /' || echo "  (none)"

    if printf '%s\n' "$ldd_output" | grep -q 'not found'; then
        echo "Missing shared libraries detected."
        if [[ "$allow_missing" -eq 0 ]]; then
            return 1
        fi
    fi
}

audit_macos() {
    local binary="$1"

    if ! have_command otool; then
        echo "otool is not available; skipping macOS loader check"
        return 0
    fi

    echo "otool -L:"
    otool -L "$binary" | sed 's/^/  /'
}

for binary in "${binaries[@]}"; do
    echo "==> $binary"

    if [[ ! -f "$binary" ]]; then
        echo "Binary does not exist: $binary" >&2
        failed=1
        continue
    fi

    if [[ ! -x "$binary" ]]; then
        echo "Binary is not executable: $binary" >&2
        failed=1
        continue
    fi

    if have_command file; then
        file "$binary" | sed 's/^/  /'
    fi

    case "$(uname -s)" in
        Linux)
            audit_linux "$binary" || failed=1
            ;;
        Darwin)
            audit_macos "$binary" || failed=1
            ;;
        *)
            echo "No platform-specific loader audit for $(uname -s)"
            ;;
    esac

    probe_startup "$binary" || failed=1
    echo
done

exit "$failed"
