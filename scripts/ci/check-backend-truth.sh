#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/ci/check-backend-truth.sh <command>

Commands:
  cargo-cpu     Run CPU-focused cargo checks for the CLI and server
  cargo-cuda    Run CUDA-focused cargo checks for the CLI and server
  docker-cpu    Validate the default Docker Compose config, build, and smoke the CPU image
  docker-cuda   Validate the CUDA Docker Compose profile, build, and audit the CUDA image
EOF
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}

resolve_cuda_compute_cap() {
    if [[ -n "${CUDA_COMPUTE_CAP:-}" ]]; then
        echo "${CUDA_COMPUTE_CAP}"
    else
        echo "80"
    fi
}

smoke_docker_server() {
    local image="$1"

    echo "Smoke-checking ${image}"
    docker run --rm \
        --entrypoint /usr/local/bin/izwi-server \
        "${image}" \
        --help >/dev/null

    assert_docker_espeak "${image}"
}

assert_docker_espeak() {
    local image="$1"

    echo "Checking Kokoro phonemizer dependency in ${image}"
    docker run --rm \
        --entrypoint /bin/sh \
        "${image}" \
        -c 'command -v espeak-ng >/dev/null'
}

audit_cuda_docker_server() {
    local image="$1"

    assert_docker_espeak "${image}"

    echo "Auditing CUDA dependencies in ${image}"
    docker run --rm \
        --entrypoint /bin/sh \
        "${image}" \
        -c '
            set -eu

            test -x /usr/local/bin/izwi-server

            ldd_output="$(ldd /usr/local/bin/izwi-server || true)"
            printf "%s\n" "${ldd_output}"

            if ! printf "%s\n" "${ldd_output}" | grep -Eq "lib(cuda|cudart|cublas|curand|nvrtc).*\.so"; then
                echo "Expected izwi-server to link against CUDA shared libraries." >&2
                exit 1
            fi

            missing="$(printf "%s\n" "${ldd_output}" | awk "/not found/ { print \$1 }")"
            unexpected_missing="$(printf "%s\n" "${missing}" | grep -Ev "^(libcuda\.so\.1)?$" || true)"
            if [ -n "${unexpected_missing}" ]; then
                echo "Unexpected missing shared libraries:" >&2
                printf "%s\n" "${unexpected_missing}" >&2
                exit 1
            fi

            if printf "%s\n" "${missing}" | grep -qx "libcuda.so.1"; then
                echo "Host driver library libcuda.so.1 is intentionally supplied by the NVIDIA container runtime."
            fi
        '
}

run_cargo_cpu() {
    require_command cargo

    cargo check --locked -p izwi-cli
    cargo check --locked -p izwi-server
}

run_cargo_cuda() {
    require_command cargo
    require_command nvcc

    local cuda_compute_cap
    cuda_compute_cap="$(resolve_cuda_compute_cap)"
    export CUDA_COMPUTE_CAP="${cuda_compute_cap}"
    echo "Using CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP}"

    cargo check --locked -p izwi-cli --features cuda
    cargo check --locked -p izwi-server --features cuda
}

run_docker_cpu() {
    require_command docker

    docker compose config >/dev/null
    docker build --target production -t izwi-ci:production .
    smoke_docker_server izwi-ci:production
}

run_docker_cuda() {
    require_command docker

    local cuda_compute_cap
    cuda_compute_cap="$(resolve_cuda_compute_cap)"

    docker compose --profile cuda config >/dev/null
    docker build \
        --build-arg CUDA_COMPUTE_CAP="${cuda_compute_cap}" \
        --target production-cuda \
        -t izwi-ci:production-cuda \
        .
    audit_cuda_docker_server izwi-ci:production-cuda
}

main() {
    if [[ $# -ne 1 ]]; then
        usage
        exit 1
    fi

    case "$1" in
        cargo-cpu)
            run_cargo_cpu
            ;;
        cargo-cuda)
            run_cargo_cuda
            ;;
        docker-cpu)
            run_docker_cpu
            ;;
        docker-cuda)
            run_docker_cuda
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            echo "Unknown command: $1" >&2
            usage
            exit 1
            ;;
    esac
}

main "$@"
