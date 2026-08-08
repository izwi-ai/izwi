#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/ci/check-backend-truth.sh <command>

Commands:
  hygiene       Run repository format, Clippy, all-target, diff, and shell gates
  cargo-cpu     Run CPU-focused cargo checks and core scheduler regressions
  cargo-metal   Run Metal-focused cargo checks and core scheduler regressions on macOS
  cargo-cuda-compile
                Compile/link CUDA harnesses without executing CUDA-linked binaries
  cargo-cuda    Compatibility alias for cargo-cuda-compile
  cargo-cuda-device
                Require NVIDIA hardware and run native/FA2 numerical certification
  docker-cpu    Validate the default Docker Compose config, build, and smoke the CPU image
  docker-cuda   Validate the CUDA Docker Compose profile, build, and audit the CUDA image
  test-cuda-feature-mapping
                Validate wrapper-to-core CUDA feature translation without running Cargo
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

resolve_cuda_wrapper_features() {
    echo "${IZWI_CUDA_FEATURES:-cuda,cudnn}"
}

resolve_cuda_core_features() {
    local wrapper_features="$1"
    local require_cuda=0
    local require_cudnn=0
    local require_flash_attn=0
    local feature
    local features=()

    IFS=',' read -r -a features <<<"${wrapper_features}"
    for feature in "${features[@]}"; do
        feature="${feature//[[:space:]]/}"
        case "${feature}" in
            cuda-base)
                require_cuda=1
                ;;
            cudnn-base)
                require_cuda=1
                require_cudnn=1
                ;;
            cuda|flash-attn)
                require_cuda=1
                require_flash_attn=1
                ;;
            cudnn)
                require_cuda=1
                require_cudnn=1
                require_flash_attn=1
                ;;
            "")
                ;;
            *)
                echo "Unsupported izwi-cli/izwi-server CUDA feature: ${feature}" >&2
                return 1
                ;;
        esac
    done

    if [[ "${require_cuda}" -ne 1 ]]; then
        echo "CUDA validation requires a cuda or cuda-base wrapper feature" >&2
        return 1
    fi

    local core_features="cuda"
    if [[ "${require_cudnn}" -eq 1 ]]; then
        core_features+=",cudnn"
    fi
    if [[ "${require_flash_attn}" -eq 1 ]]; then
        core_features+=",flash-attn"
    fi
    echo "${core_features}"
}

assert_cuda_feature_mapping() {
    local wrapper_features="$1"
    local expected_core_features="$2"
    local actual_core_features
    actual_core_features="$(resolve_cuda_core_features "${wrapper_features}")"
    if [[ "${actual_core_features}" != "${expected_core_features}" ]]; then
        echo "CUDA feature mapping mismatch for ${wrapper_features}: expected ${expected_core_features}, got ${actual_core_features}" >&2
        return 1
    fi
}

test_cuda_feature_mapping() {
    assert_cuda_feature_mapping "cuda-base,cudnn-base" "cuda,cudnn"
    assert_cuda_feature_mapping "cuda,cudnn" "cuda,cudnn,flash-attn"
    assert_cuda_feature_mapping "flash-attn" "cuda,flash-attn"

    if resolve_cuda_core_features "cuda,unknown-feature" >/dev/null 2>&1; then
        echo "Unknown CUDA wrapper features must fail closed" >&2
        return 1
    fi

    echo "CUDA wrapper/core feature mapping is valid."
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

cuda_device_available() {
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

compile_cuda_test_harnesses() {
    local wrapper_features="$1"
    local core_features="$2"

    echo "Compiling CUDA-linked test harnesses without executing them"
    cargo test --locked -p izwi-core --features "${core_features}" --lib --no-run
    cargo test --locked -p izwi-server --features "${wrapper_features}" --lib --no-run
}

smoke_cuda_device_if_available() {
    local wrapper_features="$1"

    if ! cuda_device_available; then
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
    cargo build --locked -p izwi-server --features "${wrapper_features}"
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

    if cuda_features_include "${cuda_features}" "cudnn" \
        || cuda_features_include "${cuda_features}" "cudnn-base"; then
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
    cargo test --locked -p izwi-core --lib --tests
    cargo test --locked -p izwi-server --lib
    scripts/bench/run_kv_cache_matrix.sh --lane default --iterations 1 --warmup 0
    scripts/ci/run-kv-lifecycle-soak.sh --profile pr
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
    cargo test --locked -p izwi-core --features metal --lib --tests
    cargo test --locked -p izwi-server --lib
    scripts/bench/run_kv_cache_matrix.sh --lane metal --iterations 1 --warmup 0 --require-device
}

run_hygiene() {
    require_command cargo

    git diff --check
    # The repository still has a documented, pre-existing workspace rustfmt
    # backlog. Gate the production KV surface touched by this rollout without
    # rewriting or silently grandfathering new drift in those files.
    rustfmt --edition 2021 --check --config skip_children=true \
        crates/izwi-core/src/backends/kv/mod.rs \
        crates/izwi-core/src/backends/kv/cpu.rs \
        crates/izwi-core/src/backends/kv/accelerator.rs \
        crates/izwi-core/src/kernels/cuda.rs \
        crates/izwi-core/src/kernels/metal.rs \
        crates/izwi-core/src/runtime/rollout.rs \
        crates/izwi-core/src/engine/cache/managed_stress.rs \
        crates/izwi-core/examples/kv-cache-bench.rs \
        crates/izwi-core/tests/kv_public_compatibility.rs
    cargo clippy --locked --workspace --all-targets -- -D warnings
    cargo check --locked --workspace --all-targets
    bash -n scripts/ci/*.sh scripts/bench/*.sh
    scripts/bench/test-run-kv-cache-matrix.sh
    scripts/bench/test-run-cuda-model-evidence.sh
}

run_cargo_cuda_compile() {
    require_command cargo
    require_command nvcc
    test_cuda_feature_mapping

    local cuda_compute_cap
    cuda_compute_cap="$(resolve_cuda_compute_cap)"
    local wrapper_features
    wrapper_features="$(resolve_cuda_wrapper_features)"
    local core_features
    core_features="$(resolve_cuda_core_features "${wrapper_features}")"

    export CUDA_COMPUTE_CAP="${cuda_compute_cap}"
    echo "Using CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP}"
    echo "Using wrapper CUDA features=${wrapper_features}"
    echo "Using izwi-core CUDA features=${core_features}"

    cargo check --locked -p izwi-cli --features "${wrapper_features}"
    cargo check --locked -p izwi-server --features "${wrapper_features}"
    # CUDA devel images provide linker stubs but ordinary hosted runners do not
    # mount the driver-owned `libcuda.so.1`. Compile/link CUDA harnesses without
    # running them, then execute only backend-neutral regressions.
    compile_cuda_test_harnesses "${wrapper_features}" "${core_features}"
    run_core_scheduler_regressions
    run_server_scheduler_regressions

    echo "CUDA evidence level: cuda_compiled (runtime not observed)"
}

run_cargo_cuda() {
    echo "cargo-cuda is a compatibility alias for compile-only CUDA evidence." >&2
    echo "Use cargo-cuda-device for required real-device execution." >&2
    run_cargo_cuda_compile
}

run_cargo_cuda_device_profile() {
    local wrapper_features
    wrapper_features="$(resolve_cuda_wrapper_features)"
    local core_features
    core_features="$(resolve_cuda_core_features "${wrapper_features}")"

    cargo check --locked -p izwi-cli --features "${wrapper_features}"
    cargo check --locked -p izwi-server --features "${wrapper_features}"
    run_core_scheduler_regressions "${core_features}"
    run_server_scheduler_regressions "${wrapper_features}"
    smoke_cuda_device_if_available "${wrapper_features}"
}

run_cargo_cuda_device() {
    require_command cargo
    require_command nvcc
    if ! cuda_device_available; then
        echo "CUDA hardware certification requires a device visible to nvidia-smi." >&2
        exit 1
    fi

    test_cuda_feature_mapping
    export CUDA_COMPUTE_CAP="$(resolve_cuda_compute_cap)"
    echo "Using CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP}"

    IZWI_CUDA_FEATURES="cuda-base,cudnn-base" run_cargo_cuda_device_profile
    IZWI_CUDA_FEATURES="cuda,cudnn" run_cargo_cuda_device_profile
    scripts/bench/run_kv_cache_matrix.sh \
        --lane cuda \
        --iterations "${IZWI_KV_CERT_ITERATIONS:-30}" \
        --warmup "${IZWI_KV_CERT_WARMUP:-5}" \
        --require-device \
        --output "${IZWI_KV_CERT_OUTPUT:-target/kv-cache-cuda-certification.jsonl}"
}

run_docker_cpu() {
    require_command docker

    docker compose config >/dev/null
    docker build --target production -t izwi-ci:production .
    smoke_docker_server izwi-ci:production
}

run_docker_cuda() {
    require_command docker
    test_cuda_feature_mapping

    local cuda_compute_cap
    cuda_compute_cap="$(resolve_cuda_compute_cap)"
    local wrapper_features
    wrapper_features="$(resolve_cuda_wrapper_features)"
    local core_features
    core_features="$(resolve_cuda_core_features "${wrapper_features}")"

    docker compose --profile cuda config >/dev/null
    if cuda_features_include "${core_features}" "flash-attn"; then
        assert_cuda_docker_builder_dependencies Dockerfile
    fi
    docker build \
        --build-arg CUDA_COMPUTE_CAP="${cuda_compute_cap}" \
        --build-arg IZWI_CUDA_FEATURES="${wrapper_features}" \
        --target production-cuda \
        -t izwi-ci:production-cuda \
        .
    audit_cuda_docker_server izwi-ci:production-cuda "${wrapper_features}"
}

main() {
    if [[ $# -ne 1 ]]; then
        usage
        exit 1
    fi

    case "$1" in
        hygiene)
            run_hygiene
            ;;
        cargo-cpu)
            run_cargo_cpu
            ;;
        cargo-metal)
            run_cargo_metal
            ;;
        cargo-cuda-compile)
            run_cargo_cuda_compile
            ;;
        cargo-cuda)
            run_cargo_cuda
            ;;
        cargo-cuda-device)
            run_cargo_cuda_device
            ;;
        docker-cpu)
            run_docker_cpu
            ;;
        docker-cuda)
            run_docker_cuda
            ;;
        test-cuda-feature-mapping)
            test_cuda_feature_mapping
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
