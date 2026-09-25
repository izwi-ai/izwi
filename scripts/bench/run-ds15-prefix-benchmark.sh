#!/usr/bin/env bash
#
# DS1.5 prefix-caching benchmark: runs the shared/cold gateway chat workloads
# against one local worker + one local gateway on the requested backend lane,
# captures the worker's managed-KV prometheus counters around each run, and
# writes one manifest per lane/workload under benchmarks/manifests/.
#
# The model is the tiny synthetic qwen38 hybrid fixture: this is serving-path
# evidence (publishes/attaches/reuse counters and TTFT shape), not a speed
# claim. Attaches>0 on the shared workload and attaches==0 on the cold
# workload are hard gates; a violating run fails the script.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
public_model="Qwen3.8-27B-FP8"
lane="cpu"
workloads="shared,cold"
requests=40
concurrency=4
max_tokens=16
prefix_tokens=64
suffix_tokens=16
models_dir="${repo_root}/target/ds15-bench/models"
output_dir="${repo_root}/benchmarks/manifests"
worker_bin="${repo_root}/target/debug/izwi-serving-worker"
gateway_bin="${repo_root}/target/debug/izwi-server"
skip_build=0
dry_run=0

usage() {
    cat <<'EOF'
Usage: scripts/bench/run-ds15-prefix-benchmark.sh [options]

Options:
  --lane cpu|metal       Backend lane (default: cpu)
  --workloads LIST       Comma-separated subset of shared,cold (default: shared,cold)
  --requests N           Requests per workload (default: 40)
  --concurrency N        Client concurrency (default: 4)
  --max-tokens N         Output tokens per request (default: 16)
  --prefix-tokens N      Approximate shared prefix length in words (default: 64)
  --suffix-tokens N      Approximate per-request suffix length in words (default: 16)
  --models-dir PATH      Fixture models root (default: target/ds15-bench/models)
  --output-dir PATH      Manifest output directory (default: benchmarks/manifests)
  --worker-bin PATH      Worker binary (default: target/debug/izwi-serving-worker)
  --gateway-bin PATH     Gateway binary (default: target/debug/izwi-server)
  --skip-build           Reuse existing binaries; no cargo build
  --dry-run              Print the resolved plan (env, commands) and exit
  -h, --help             Show this help

The script starts the worker first, waits for its authenticated descriptor,
then starts the gateway and waits for /readyz (the gateway fails closed if it
wins the race). Warm-up requests precede each timed run. Worker prometheus
counters are captured before and after every workload. This is an opt-in
benchmark over a synthetic fixture; it makes no speed claim and certifies no
hardware. CUDA is not a lane here; it has its own evidence runners.
EOF
}

log() { echo "[ds15] $*"; }

die() { echo "[ds15] error: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lane) lane="${2:-}"; shift 2 ;;
        --workloads) workloads="${2:-}"; shift 2 ;;
        --requests) requests="${2:-}"; shift 2 ;;
        --concurrency) concurrency="${2:-}"; shift 2 ;;
        --max-tokens) max_tokens="${2:-}"; shift 2 ;;
        --prefix-tokens) prefix_tokens="${2:-}"; shift 2 ;;
        --suffix-tokens) suffix_tokens="${2:-}"; shift 2 ;;
        --models-dir) models_dir="${2:-}"; shift 2 ;;
        --output-dir) output_dir="${2:-}"; shift 2 ;;
        --worker-bin) worker_bin="${2:-}"; shift 2 ;;
        --gateway-bin) gateway_bin="${2:-}"; shift 2 ;;
        --skip-build) skip_build=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done

[[ "$lane" == "cpu" || "$lane" == "metal" ]] || die "--lane must be cpu or metal"
[[ "$requests" =~ ^[0-9]+$ && "$requests" -gt 0 ]] || die "--requests must be a positive integer"
[[ "$concurrency" =~ ^[0-9]+$ && "$concurrency" -ge 1 && "$concurrency" -le 64 ]] || die "--concurrency must be between 1 and 64"
[[ "$max_tokens" =~ ^[0-9]+$ && "$max_tokens" -gt 0 && "$max_tokens" -le 32 ]] || die "--max-tokens must be between 1 and 32 (worker advertises 32)"
[[ "$prefix_tokens" =~ ^[0-9]+$ && "$prefix_tokens" -gt 0 ]] || die "--prefix-tokens must be a positive integer"
[[ "$suffix_tokens" =~ ^[0-9]+$ && "$suffix_tokens" -gt 0 ]] || die "--suffix-tokens must be a positive integer"
command -v jq >/dev/null || die "jq is required"
command -v python3 >/dev/null || die "python3 is required"

workload_list=()
IFS=',' read -r -a workload_list <<<"$workloads"
for workload in "${workload_list[@]}"; do
    [[ "$workload" == "shared" || "$workload" == "cold" ]] || die "workload '$workload' is not one of shared,cold"
done

# The chunked-prefill threshold must cover the whole prompt so the publishing
# session commits its entire prefix in the first prefix-eligible chunk. The
# prompt token count runs ~1.6x the word count (the engine prepends the Xhigh
# reasoning-instruction system block, ~43 tokens, to every request), so size
# the threshold with headroom. The KV page must stay LARGER than that
# structural block (kv_page_size=64 > ~43 tokens): with 16-token pages every
# request — cold included — shares the block's 2 pages by construction and the
# cold workload could never measure zero reuse.
prompt_words=$((prefix_tokens + suffix_tokens))
chunk_threshold=$((prompt_words * 2 + 64))
kv_page_size=64
deployment_id="ds15-${lane}-bench-v1"
credential_id="ds15-bench-credential"
bearer_token=$(openssl rand -hex 24 2>/dev/null || echo "ds15-local-bearer-0000000000000000")
api_key=$(openssl rand -hex 24 2>/dev/null || echo "ds15-local-apikey-000000000000000")
artifact_revision="017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
fixture_model_dir="${models_dir}/${public_model}"
work_dir="${repo_root}/target/ds15-bench/run-$(date +%Y%m%d-%H%M%S)"

worker_env=(
    "IZWI_WORKER_BIND=127.0.0.1:<worker-port>"
    "IZWI_MODELS_DIR=${models_dir}"
    "IZWI_WORKER_MODEL=${public_model}"
    "IZWI_WORKER_DEPLOYMENT_ID=${deployment_id}"
    "IZWI_WORKER_ARTIFACT_REVISION=${artifact_revision}"
    "IZWI_WORKER_CREDENTIAL_ID=${credential_id}"
    "IZWI_WORKER_BEARER_TOKEN=${bearer_token}"
    "IZWI_WORKER_MAX_ACTIVE=${concurrency}"
    "IZWI_ALLOW_SYNTHETIC_QWEN38_GEOMETRY=1"
    "IZWI_ENABLE_PREFIX_CACHING=1"
    "IZWI_MANAGED_PREFIX_CACHE_SALT=ds15-bench-salt"
    "IZWI_ENABLE_CHUNKED_PREFILL=1"
    "IZWI_CHUNKED_PREFILL_THRESHOLD=${chunk_threshold}"
    "IZWI_KV_PAGE_SIZE=64"
    "IZWI_CUDA_MTP=off"
    "IZWI_MAX_PREFIX_CACHE_PAGES=256"
    "IZWI_MAX_SEQUENCE_LENGTH=4096"
    "RUST_LOG=warn"
)
if [[ "$lane" == "cpu" ]]; then
    worker_env+=(
        "IZWI_BACKEND=cpu"
        "IZWI_WORKER_CPU_THREADS=${DS15_WORKER_CPU_THREADS:-2}"
        "IZWI_CPU_MEMORY_BUDGET_BYTES=2147483648"
        "IZWI_WORKER_HOST_MEMORY_LIMIT_BYTES=2147483648"
    )
else
    worker_env+=(
        "IZWI_BACKEND=metal"
        "IZWI_METAL_DEVICE_ORDINAL=0"
        "IZWI_WORKER_EXPECTED_DEVICE_ID=<host metal:<registryID>>"
    )
fi

if [[ "$dry_run" == 1 ]]; then
    echo "ds15 dry-run plan"
    echo "lane=$lane workloads=${workload_list[*]} requests=$requests concurrency=$concurrency"
    echo "max_tokens=$max_tokens prefix_tokens=$prefix_tokens suffix_tokens=$suffix_tokens"
    echo "chunked_prefill_threshold=$chunk_threshold deployment_id=$deployment_id"
    echo "fixture_model_dir=$fixture_model_dir"
    echo "output_dir=$output_dir worker_bin=$worker_bin gateway_bin=$gateway_bin"
    echo "worker_env:"
    printf '  %s\n' "${worker_env[@]}"
    echo "gateway_cmd: $gateway_bin --role gateway --host 127.0.0.1 --port <gateway-port> --public-model $public_model --gateway-worker-approval http://127.0.0.1:<worker-port>|chat|$public_model|$deployment_id|1"
    echo "gateway_env: IZWI_GATEWAY_API_KEY IZWI_GATEWAY_WORKER_CREDENTIAL_ID IZWI_GATEWAY_WORKER_BEARER_TOKEN IZWI_GATEWAY_TENANT_MAX_CONCURRENT=<concurrency> IZWI_GATEWAY_TENANT_REQUESTS_PER_MINUTE=3000 IZWI_GATEWAY_TENANT_BURST_REQUESTS=256"
    echo "gateway_max_in_flight=$concurrency (bounded by the worker's active capacity)"
    echo "harness_vocab: a,b,c (the fixture tokenizer only knows a/b/c; other words collapse to unk)"
    echo "harness: run-gateway-chat-benchmark.py --workload {shared,cold} --metadata {deployed_sha,topology,backend,model,artifact_revision,...}"
    exit 0
fi

[[ -x "$worker_bin" ]] || die "worker binary missing at $worker_bin (build first or drop --skip-build)"
[[ -x "$gateway_bin" ]] || die "gateway binary missing at $gateway_bin (build first or drop --skip-build)"

free_port() {
    python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'
}
worker_port=$(free_port)
gateway_port=$(free_port)

if [[ "$lane" == "metal" ]]; then
    metal_device_id=$(swift -e 'import Metal; if let d = MTLCreateSystemDefaultDevice() { print("metal:" + String(d.registryID)) }' 2>/dev/null | tail -1 || true)
    [[ "$metal_device_id" == metal:* ]] || die "metal lane requires an Apple GPU; could not resolve metal:<registryID>"
fi

mkdir -p "$work_dir" "$output_dir"

worker_pid=""
gateway_pid=""
cleanup() {
    [[ -n "$gateway_pid" ]] && kill "$gateway_pid" 2>/dev/null || true
    [[ -n "$worker_pid" ]] && kill "$worker_pid" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

if [[ "$skip_build" == 0 ]]; then
    # Host quirks (see tasks/lessons.md): CommandLineTools avoids the Xcode
    # license prompt; CXXFLAGS repairs the CLT libc++ include path.
    export DEVELOPER_DIR="${DEVELOPER_DIR:-/Library/Developer/CommandLineTools}"
    if [[ -z "${CXXFLAGS:-}" ]]; then
        sdk=$(ls -d /Library/Developer/CommandLineTools/SDKs/MacOSX*.sdk 2>/dev/null | sort -V | tail -1 || true)
        [[ -n "$sdk" ]] && export CXXFLAGS="-I${sdk}/usr/include/c++/v1"
    fi
    if [[ "$lane" == "metal" ]]; then
        log "building worker (metal feature)"
        cargo build --locked --quiet -p izwi-serving-worker --features metal
    else
        log "building worker (cpu)"
        cargo build --locked --quiet -p izwi-serving-worker
    fi
    log "building gateway"
    cargo build --locked --quiet -p izwi-server
fi

if [[ ! -f "$fixture_model_dir/izwi-artifact.json" ]]; then
    log "generating tiny qwen38 hybrid fixture into $models_dir"
    IZWI_BENCH_FIXTURE_DIR="$models_dir" cargo test --locked --quiet \
        -p izwi-serving-worker --test backend_parity \
        generate_qwen38_benchmark_fixture -- --ignored --nocapture
fi
[[ -f "$fixture_model_dir/izwi-artifact.json" ]] || die "fixture generation did not produce $fixture_model_dir"

log "starting worker on 127.0.0.1:$worker_port (lane=$lane)"

worker_full_env=()
for entry in "${worker_env[@]}"; do
    entry="${entry/<worker-port>/$worker_port}"
    if [[ "$entry" == "IZWI_WORKER_EXPECTED_DEVICE_ID=<host metal:<registryID>>" ]]; then
        entry="IZWI_WORKER_EXPECTED_DEVICE_ID=${metal_device_id}"
    fi
    worker_full_env+=("$entry")
done

env "${worker_full_env[@]}" "$worker_bin" >"$work_dir/worker.log" 2>&1 &
worker_pid=$!

worker_ready=0
for _ in $(seq 1 240); do
    code=$(curl -s -o /dev/null -w '%{http_code}' \
        -H "Authorization: Bearer ${bearer_token}" \
        -H "x-izwi-service-credential-id: ${credential_id}" \
        "http://127.0.0.1:${worker_port}/internal/v1/worker" || true)
    if [[ "$code" == "200" ]]; then worker_ready=1; break; fi
    if ! kill -0 "$worker_pid" 2>/dev/null; then
        echo "--- worker.log tail ---" >&2; tail -40 "$work_dir/worker.log" >&2
        die "worker exited before its descriptor became ready"
    fi
    sleep 0.5
done
[[ "$worker_ready" == 1 ]] || { tail -40 "$work_dir/worker.log" >&2; die "worker descriptor not ready within 120s"; }
log "worker descriptor ready"

log "starting gateway on 127.0.0.1:$gateway_port"
env \
    "IZWI_GATEWAY_API_KEY=${api_key}" \
    "IZWI_GATEWAY_WORKER_CREDENTIAL_ID=${credential_id}" \
    "IZWI_GATEWAY_WORKER_BEARER_TOKEN=${bearer_token}" \
    "IZWI_GATEWAY_TENANT_MAX_CONCURRENT=${concurrency}" \
    "IZWI_GATEWAY_TENANT_REQUESTS_PER_MINUTE=3000" \
    "IZWI_GATEWAY_TENANT_BURST_REQUESTS=256" \
    "IZWI_GATEWAY_WORKER_STATUS_POLL_MS=200" \
    "IZWI_GATEWAY_WORKER_STATUS_TTL_MS=5000" \
    "RUST_LOG=warn" \
    "$gateway_bin" --role gateway --host 127.0.0.1 --port "$gateway_port" \
    --public-model "$public_model" \
    --gateway-max-in-flight "$concurrency" \
    --gateway-worker-approval "http://127.0.0.1:${worker_port}|chat|${public_model}|${deployment_id}|1" \
    >"$work_dir/gateway.log" 2>&1 &
gateway_pid=$!

gateway_ready=0
for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${gateway_port}/readyz" >/dev/null 2>&1; then gateway_ready=1; break; fi
    if ! kill -0 "$gateway_pid" 2>/dev/null; then
        echo "--- gateway.log tail ---" >&2; tail -40 "$work_dir/gateway.log" >&2
        die "gateway exited before ready"
    fi
    sleep 0.5
done
[[ "$gateway_ready" == 1 ]] || { tail -40 "$work_dir/gateway.log" >&2; die "gateway not ready within 60s"; }
log "gateway ready"

gateway_url="http://127.0.0.1:${gateway_port}"
worker_metrics_url="http://127.0.0.1:${worker_port}/internal/v1/metrics/prometheus"
fetch_counters() {
    curl -sf \
        -H "Authorization: Bearer ${bearer_token}" \
        -H "x-izwi-service-credential-id: ${credential_id}" \
        "$worker_metrics_url"
}

warm_up() {
    curl -sf -X POST "$gateway_url/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${api_key}" \
        -H "x-request-id: ds15-warmup-$1" \
        -d "{\"model\":\"${public_model}\",\"messages\":[{\"role\":\"user\",\"content\":\"warm up request $1\"}],\"max_tokens\":4,\"stream\":false}" \
        >/dev/null
}
log "warm-up"
warm_up 1
warm_up 2

deployed_sha=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || echo "unknown")

fail_gate=""
for workload in "${workload_list[@]}"; do
    log "running workload=$workload (requests=$requests concurrency=$concurrency)"
    fetch_counters >"$work_dir/counters-${workload}-before.txt"

    metadata=$(jq -n \
        --arg sha "$deployed_sha" \
        --arg lane "$lane" \
        --arg model "$public_model" \
        --arg revision "$artifact_revision" \
        --arg deployment "$deployment_id" \
        --arg workload "$workload" \
        --argjson prefix "$prefix_tokens" \
        --argjson suffix "$suffix_tokens" \
        --argjson threshold "$chunk_threshold" \
        --argjson kv_page_size "$kv_page_size" \
        --argjson requests "$requests" \
        --argjson concurrency "$concurrency" \
        --arg vocab "a,b,c" \
        '{deployed_sha: $sha, topology: "single-node-loopback", backend: $lane,
          model: $model, artifact_revision: $revision, deployment_id: $deployment,
          fixture: "tiny synthetic qwen38 hybrid; no speed claim", workload: $workload,
          prefix_tokens: $prefix, suffix_tokens: $suffix,
          chunked_prefill_threshold: $threshold, kv_page_size: $kv_page_size,
          requests: $requests, concurrency: $concurrency, vocab: $vocab}')
    python3 "$repo_root/scripts/bench/run-gateway-chat-benchmark.py" \
        --gateway "$gateway_url" \
        --api-key "$api_key" \
        --model "$public_model" \
        --requests "$requests" \
        --concurrency "$concurrency" \
        --max-tokens "$max_tokens" \
        --stream \
        --workload "$workload" \
        --vocab "a,b,c" \
        --prefix-tokens "$prefix_tokens" \
        --suffix-tokens "$suffix_tokens" \
        --max-retries "${DS15_MAX_RETRIES:-40}" \
        --metadata "$metadata" \
        --output "$work_dir/ds15-${lane}-${workload}.json"

    fetch_counters >"$work_dir/counters-${workload}-after.txt"

    jq -n \
        --slurpfile benchmark "$work_dir/ds15-${lane}-${workload}.json" \
        --rawfile before "$work_dir/counters-${workload}-before.txt" \
        --rawfile after "$work_dir/counters-${workload}-after.txt" \
        '
        def counters(text):
            [text | scan("(izwi_engine_[a-z_]+) ([0-9]+)") | {(.[0]): (.[1] | tonumber)}] | add // {};
        {
            schema: "izwi.ds15-prefix-benchmark.v1",
            lane: $benchmark[0].metadata.backend,
            workload: $benchmark[0].workload,
            benchmark: $benchmark[0],
            counters_before: counters($before),
            counters_after: counters($after)
        } * (counters($after) | with_entries(.value -= (counters($before)[.key] // 0))
             | {counters_delta: .})' \
        >"$output_dir/ds15-${lane}-${workload}.json"

    attaches=$(jq -r '.counters_delta["izwi_engine_tensor_snapshot_attaches_total"] // 0' "$output_dir/ds15-${lane}-${workload}.json")
    publishes=$(jq -r '.counters_delta["izwi_engine_tensor_snapshot_publishes_total"] // 0' "$output_dir/ds15-${lane}-${workload}.json")
    completed=$(jq -r '.benchmark.completed' "$output_dir/ds15-${lane}-${workload}.json")
    log "workload=$workload completed=$completed publishes=$publishes attaches=$attaches"

    [[ "$completed" -eq "$requests" ]] || fail_gate="${fail_gate} ${workload}:completed=$completed"
    [[ "$publishes" -ge 1 ]] || fail_gate="${fail_gate} ${workload}:publishes=$publishes"
    if [[ "$workload" == "shared" && "$attaches" -lt 1 ]]; then
        fail_gate="${fail_gate} shared:attaches=$attaches"
    fi
    if [[ "$workload" == "cold" && "$attaches" -ne 0 ]]; then
        fail_gate="${fail_gate} cold:attaches=$attaches"
    fi
done

if [[ -f "$output_dir/ds15-${lane}-shared.json" && -f "$output_dir/ds15-${lane}-cold.json" ]]; then
    jq -n -S \
        --slurpfile shared "$output_dir/ds15-${lane}-shared.json" \
        --slurpfile cold "$output_dir/ds15-${lane}-cold.json" \
        '{
            schema: "izwi.ds15-prefix-summary.v1",
            lane: $shared[0].lane,
            ttft_ms: {
                shared_p50: $shared[0].benchmark.ttft_ms.p50,
                shared_p95: $shared[0].benchmark.ttft_ms.p95,
                cold_p50: $cold[0].benchmark.ttft_ms.p50,
                cold_p95: $cold[0].benchmark.ttft_ms.p95,
                p50_delta_ms: (($shared[0].benchmark.ttft_ms.p50 // 0) - ($cold[0].benchmark.ttft_ms.p50 // 0))
            },
            shared_counter_deltas: $shared[0].counters_delta,
            cold_counter_deltas: $cold[0].counters_delta
        }' >"$output_dir/ds15-${lane}-summary.json"
    log "summary: $output_dir/ds15-${lane}-summary.json"
fi

if [[ -n "$fail_gate" ]]; then
    die "benchmark gate failed:$fail_gate"
fi
log "done; manifests in $output_dir; per-run artifacts in $work_dir"
