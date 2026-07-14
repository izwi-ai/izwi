#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/ci/check-backend-truth.sh <command>

Commands:
  cargo-cpu     Run CPU-focused cargo checks and core scheduler regressions
  cargo-metal   Run Metal-focused cargo checks and core scheduler regressions on macOS
  cargo-cuda    Run CUDA-focused checks, portable regressions, and a device smoke when available
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

resolve_cuda_features() {
    echo "${IZWI_CUDA_FEATURES:-cuda,cudnn}"
}

run_core_scheduler_regressions() {
    local features="${1:-}"
    local cargo_args=(--locked -p izwi-core)
    local suites=(
        engine::scheduler
        engine::resources
        engine::core
        runtime::coordinator
        runtime::service
        runtime::rollout
        engine::executor
        engine::execution
        engine::output
    )

    if [[ -n "${features}" ]]; then
        cargo_args+=(--features "${features}")
    fi

    for suite in "${suites[@]}"; do
        echo "Running ${suite} regressions"
        cargo test "${cargo_args[@]}" "${suite}" --lib
    done
}

run_server_scheduler_regressions() {
    local features="${1:-}"
    local cargo_args=(--locked -p izwi-server --lib)
    local suites=(
        saturated_chat_stream
        saturated_stream_emits_explicit_terminal_error
        terminal_events_wait_for_capacity_and_preserve_order
        http_shutdown_
    )

    if [[ -n "${features}" ]]; then
        cargo_args+=(--features "${features}")
    fi

    for suite in "${suites[@]}"; do
        echo "Running izwi-server ${suite} regressions"
        cargo test "${cargo_args[@]}" "${suite}"
    done
}

smoke_cuda_device_if_available() {
    local cuda_features="$1"

    if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
        echo "No usable NVIDIA device exposed; portable CUDA regressions completed."
        return
    fi

    require_command curl

    local port="${IZWI_CUDA_SMOKE_PORT:-18081}"
    local health_url="http://127.0.0.1:${port}/internal/health"
    local log_path="${RUNNER_TEMP:-/tmp}/izwi-cuda-device-smoke.log"
    local server_binary="${CARGO_TARGET_DIR:-target}/debug/izwi-server"
    local server_pid
    local health_payload=""

    echo "Smoke-checking the CUDA device through izwi-server"
    cargo build --locked -p izwi-server --features "${cuda_features}"
    IZWI_MODELS_DIR="${RUNNER_TEMP:-/tmp}/izwi-cuda-smoke-models" \
    IZWI_PRELOAD_MODELS= \
    IZWI_WARMUP_PRELOADED_MODELS=0 \
    "${server_binary}" \
        --host 127.0.0.1 \
        --port "${port}" \
        --backend cuda >"${log_path}" 2>&1 &
    server_pid=$!
    trap 'kill '"${server_pid}"' >/dev/null 2>&1 || true; wait '"${server_pid}"' >/dev/null 2>&1 || true' EXIT

    for _ in {1..60}; do
        if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
            echo "CUDA smoke server exited before becoming healthy:" >&2
            sed -n '1,240p' "${log_path}" >&2
            return 1
        fi
        if health_payload="$(curl -fsS "${health_url}" 2>/dev/null)"; then
            break
        fi
        sleep 1
    done

    if [[ -z "${health_payload}" ]]; then
        echo "CUDA smoke server did not become healthy:" >&2
        sed -n '1,240p' "${log_path}" >&2
        return 1
    fi

    for expected in \
        '"requested_backend":"cuda"' \
        '"requested_backend_available":true' \
        '"selected_backend":"cuda"' \
        '"cuda":true' \
        '"driver_available":true' \
        '"device_usable":true'; do
        if ! grep -Fq "${expected}" <<<"${health_payload}"; then
            echo "CUDA health response is missing ${expected}:" >&2
            printf '%s\n' "${health_payload}" >&2
            return 1
        fi
    done

    kill "${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" >/dev/null 2>&1 || true
    trap - EXIT
}

smoke_docker_server() {
    local image="$1"

    echo "Smoke-checking ${image}"
    docker run --rm \
        --entrypoint /usr/local/bin/izwi-server \
        "${image}" \
        --help >/dev/null

    assert_docker_runtime_commands "${image}"
}

assert_docker_runtime_commands() {
    local image="$1"

    echo "Checking runtime command dependencies in ${image}"
    docker run --rm \
        --entrypoint /bin/sh \
        "${image}" \
        -c '
            set -eu

            for cmd in espeak-ng tar unzip zip which; do
                command -v "${cmd}" >/dev/null
            done
        '
}

cuda_features_include() {
    local features="$1"
    local feature="$2"
    case ",${features}," in
        *",${feature},"*) return 0 ;;
        *) return 1 ;;
    esac
}

assert_cuda_docker_builder_dependencies() {
    local dockerfile="${1:-Dockerfile}"

    if ! awk '
        /^FROM .* AS rust-builder-cuda$/ { in_cuda_builder = 1; next }
        /^FROM / && in_cuda_builder { in_cuda_builder = 0 }
        in_cuda_builder && /^[[:space:]]*git([[:space:]]*\\)?[[:space:]]*$/ { found_git = 1 }
        END { exit found_git ? 0 : 1 }
    ' "${dockerfile}"; then
        echo "Docker CUDA builder must install git for Candle flash-attn CUTLASS checkout." >&2
        exit 1
    fi
}

audit_cuda_docker_server() {
    local image="$1"
    local cuda_features="$2"

    assert_docker_runtime_commands "${image}"

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
        '

    if cuda_features_include "${cuda_features}" "cudnn"; then
        docker run --rm \
            --entrypoint /bin/sh \
            "${image}" \
            -c '
                set -eu
                ldd_output="$(ldd /usr/local/bin/izwi-server || true)"
                if ! printf "%s\n" "${ldd_output}" | grep -Eq "libcudnn.*\.so"; then
                    echo "Expected izwi-server to link against cuDNN shared libraries." >&2
                    exit 1
                fi
            '
    fi

    docker run --rm \
        --entrypoint /bin/sh \
        "${image}" \
        -c '
            set -eu
            ldd_output="$(ldd /usr/local/bin/izwi-server || true)"

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
    run_core_scheduler_regressions
    run_server_scheduler_regressions
}

run_cargo_metal() {
    require_command cargo

    if [[ "$(uname -s)" != "Darwin" ]]; then
        echo "Metal checks require macOS." >&2
        exit 1
    fi

    cargo check --locked -p izwi-core --features metal
    cargo check --locked -p izwi-cli --features metal
    cargo check --locked -p izwi-server
    run_core_scheduler_regressions metal
    run_server_scheduler_regressions
}

run_cargo_cuda() {
    require_command cargo
    require_command nvcc

    local cuda_compute_cap
    cuda_compute_cap="$(resolve_cuda_compute_cap)"
    local cuda_features
    cuda_features="$(resolve_cuda_features)"

    export CUDA_COMPUTE_CAP="${cuda_compute_cap}"
    echo "Using CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP}"
    echo "Using IZWI_CUDA_FEATURES=${cuda_features}"

    cargo check --locked -p izwi-cli --features "${cuda_features}"
    cargo check --locked -p izwi-server --features "${cuda_features}"
    run_core_scheduler_regressions "${cuda_features}"
    run_server_scheduler_regressions "${cuda_features}"
    smoke_cuda_device_if_available "${cuda_features}"
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
    local cuda_features
    cuda_features="$(resolve_cuda_features)"

    docker compose --profile cuda config >/dev/null
    if cuda_features_include "${cuda_features}" "flash-attn"; then
        assert_cuda_docker_builder_dependencies Dockerfile
    fi
    docker build \
        --build-arg CUDA_COMPUTE_CAP="${cuda_compute_cap}" \
        --build-arg IZWI_CUDA_FEATURES="${cuda_features}" \
        --target production-cuda \
        -t izwi-ci:production-cuda \
        .
    audit_cuda_docker_server izwi-ci:production-cuda "${cuda_features}"
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
        cargo-metal)
            run_cargo_metal
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
