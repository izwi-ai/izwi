#!/usr/bin/env bash
#
# DS4 hierarchical KV offload benchmark: runs the shared gateway chat workload
# at high concurrency against ONE local worker + one local gateway on the
# requested backend lane, once with the DS4 host pool disabled and once with
# it enabled, and writes per-leg manifests with the worker's managed-KV
# counter deltas (demotions, promotions, host pages) under
# benchmarks/manifests/.
#
# The model is the tiny synthetic qwen38 hybrid fixture: this is offload-path
# evidence (a committed shared prefix must demote into the host pool under
# pressure and promote back for later requests), not a speed claim. Hard
# gates: every leg completes all requests, and the offload-on leg shows at
# least one demotion and one promotion with host usage inside the pool
# budget. A violating run fails the script.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
public_model="Qwen3.8-27B-FP8"
lane="cpu"
requests=16
concurrency=8
max_tokens=8
prefix_tokens=64
suffix_tokens=16
# The fixture arena resolves to 512 tokens (8 pages at kv_page_size=64), so
# the watermarks are calibrated to fixture scale: one retained shared page
# (1/8 of the arena) already crosses the high watermark. Production geometry
# keeps the defaults (0.85/0.70).
high_watermark=0.10
low_watermark=0.05
pool_budget_bytes=8388608
# Fixture page bytes at kv_page_size=64: one full-attention layer, kv_heads 1,
# head_dim 2, K+V, bf16 -> 64*1*2*2*2 = 512.
page_bytes=512
models_dir="${repo_root}/target/ds4-bench/models"
output_dir="${repo_root}/benchmarks/manifests"
worker_bin="${repo_root}/target/debug/izwi-serving-worker"
gateway_bin="${repo_root}/target/debug/izwi-server"
skip_build=0
dry_run=0

usage() {
    cat <<'EOF'
Usage: scripts/bench/run-ds4-offload-benchmark.sh [options]

Options:
  --lane cpu|metal       Backend lane (default: cpu)
  --requests N           Shared-workload requests per leg (default: 16)
  --concurrency N        Client concurrency (default: 8)
  --max-tokens N         Output tokens per request (default: 8)
  --prefix-tokens N      Approximate shared system prefix length in words (default: 64)
  --suffix-tokens N      Approximate per-request user suffix length in words (default: 16)
  --pool-budget BYTES   DS4 host pool budget for the on leg (default: 8388608)
  --models-dir PATH      Fixture models root (default: target/ds4-bench/models)
  --output-dir PATH      Manifest output directory (default: benchmarks/manifests)
  --worker-bin PATH      Worker binary (default: target/debug/izwi-serving-worker)
  --gateway-bin PATH     Gateway binary (default: target/debug/izwi-server)
  --skip-build           Reuse existing binaries; no cargo build
  --dry-run              Print the resolved plan (env, commands) and exit
  -h, --help             Show this help

Each leg (host pool off, then on) starts one fresh worker and one gateway, so
per-leg counter deltas never see the other leg's cache state. Workers start
first; the gateway fails closed if it wins that race. This is an opt-in
benchmark over a synthetic fixture; it makes no speed claim and certifies no
hardware. CUDA is not a lane here.
EOF
}

log() { echo "[ds4] $*"; }

die() { echo "[ds4] error: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lane) lane="${2:-}"; shift 2 ;;
        --requests) requests="${2:-}"; shift 2 ;;
        --concurrency) concurrency="${2:-}"; shift 2 ;;
        --max-tokens) max_tokens="${2:-}"; shift 2 ;;
        --prefix-tokens) prefix_tokens="${2:-}"; shift 2 ;;
        --suffix-tokens) suffix_tokens="${2:-}"; shift 2 ;;
        --pool-budget) pool_budget_bytes="${2:-}"; shift 2 ;;
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
[[ "$pool_budget_bytes" =~ ^[0-9]+$ && "$pool_budget_bytes" -gt 0 ]] || die "--pool-budget must be a positive integer"
command -v jq >/dev/null || die "jq is required"
command -v python3 >/dev/null || die "python3 is required"

# Same sizing rationale as the DS2 rig: the chunked-prefill threshold must
# cover the whole prompt (word count runs ~1.6x tokens including the ~43-token
# Xhigh block), and the KV page must stay larger than that structural block so
# unrelated requests share nothing by construction (kv_page_size=64).
prompt_words=$((prefix_tokens + suffix_tokens))
chunk_threshold=$((prompt_words * 2 + 64))
kv_page_size=64
deployment_id="ds4-${lane}-bench-v1"
credential_id="ds4-bench-credential"
bearer_token=$(openssl rand -hex 24 2>/dev/null || echo "ds4-local-bearer-0000000000000000")
api_key=$(openssl rand -hex 24 2>/dev/null || echo "ds4-local-apikey-000000000000000")
artifact_revision="017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
fixture_model_dir="${models_dir}/${public_model}"
work_dir="${repo_root}/target/ds4-bench/run-$(date +%Y%m%d-%H%M%S)"

worker_env_base=(
    "IZWI_MODELS_DIR=${models_dir}"
    "IZWI_WORKER_MODEL=${public_model}"
    "IZWI_WORKER_DEPLOYMENT_ID=${deployment_id}"
    "IZWI_WORKER_ARTIFACT_REVISION=${artifact_revision}"
    "IZWI_WORKER_CREDENTIAL_ID=${credential_id}"
    "IZWI_WORKER_BEARER_TOKEN=${bearer_token}"
    "IZWI_WORKER_MAX_ACTIVE=${concurrency}"
    "IZWI_ALLOW_SYNTHETIC_QWEN38_GEOMETRY=1"
    "IZWI_ENABLE_PREFIX_CACHING=1"
    "IZWI_MANAGED_PREFIX_CACHE_SALT=ds4-bench-salt"
    "IZWI_ENABLE_CHUNKED_PREFILL=1"
    "IZWI_CHUNKED_PREFILL_THRESHOLD=${chunk_threshold}"
    "IZWI_KV_PAGE_SIZE=${kv_page_size}"
    "IZWI_CUDA_MTP=off"
    "IZWI_MAX_PREFIX_CACHE_PAGES=256"
    "IZWI_MAX_SEQUENCE_LENGTH=4096"
    "IZWI_KV_OFFLOAD_HIGH_WATERMARK=${high_watermark}"
    "IZWI_KV_OFFLOAD_LOW_WATERMARK=${low_watermark}"
    "RUST_LOG=warn"
)
worker_cpu_env=(
    "IZWI_BACKEND=cpu"
    "IZWI_WORKER_CPU_THREADS=${DS4_WORKER_CPU_THREADS:-2}"
    "IZWI_CPU_MEMORY_BUDGET_BYTES=2147483648"
    "IZWI_WORKER_HOST_MEMORY_LIMIT_BYTES=2147483648"
)
worker_metal_env=(
    "IZWI_BACKEND=metal"
    "IZWI_METAL_DEVICE_ORDINAL=0"
    "IZWI_WORKER_EXPECTED_DEVICE_ID=<host metal:<registryID>>"
)

if [[ "$dry_run" == 1 ]]; then
    echo "ds4 dry-run plan"
    echo "lane=$lane requests=$requests concurrency=$concurrency max_tokens=$max_tokens"
    echo "prefix_tokens=$prefix_tokens suffix_tokens=$suffix_tokens pool_budget_bytes=$pool_budget_bytes"
    echo "high_watermark=$high_watermark low_watermark=$low_watermark"
    echo "chunked_prefill_threshold=$chunk_threshold deployment_id=$deployment_id"
    echo "fixture_model_dir=$fixture_model_dir"
    echo "output_dir=$output_dir worker_bin=$worker_bin gateway_bin=$gateway_bin"
    echo "per-leg worker env delta:"
    echo "  off: (no IZWI_KV_HOST_POOL_BUDGET_BYTES; pool dormant)"
    echo "  on:  IZWI_KV_HOST_POOL_BUDGET_BYTES=${pool_budget_bytes}"
    echo "gateway_cmd: $gateway_bin --role gateway ... --gateway-worker-approval http://127.0.0.1:<port>|chat|$public_model|$deployment_id|1"
    echo "harness_vocab: a,b,c (the fixture tokenizer only knows a/b/c; other words collapse to unk)"
    exit 0
fi

[[ -x "$worker_bin" ]] || die "worker binary missing at $worker_bin (build first or drop --skip-build)"
[[ -x "$gateway_bin" ]] || die "gateway binary missing at $gateway_bin (build first or drop --skip-build)"

free_port() {
    python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'
}

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

if [[ ! -f "$fixture_model_dir/izwi-artifact.json" ]]; then
    log "generating tiny qwen38 hybrid fixture into $models_dir"
    IZWI_BENCH_FIXTURE_DIR="$models_dir" cargo test --locked --quiet \
        -p izwi-serving-worker --test backend_parity \
        generate_qwen38_benchmark_fixture -- --ignored --nocapture
fi
[[ -f "$fixture_model_dir/izwi-artifact.json" ]] || die "fixture generation did not produce $fixture_model_dir"

deployed_sha=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || echo "unknown")

worker_ready_wait() {
    local worker_pid=$1 worker_port=$2
    for _ in $(seq 1 240); do
        code=$(curl -s -o /dev/null -w '%{http_code}' \
            -H "Authorization: Bearer ${bearer_token}" \
            -H "x-izwi-service-credential-id: ${credential_id}" \
            "http://127.0.0.1:${worker_port}/internal/v1/worker" || true)
        if [[ "$code" == "200" ]]; then return 0; fi
        if ! kill -0 "$worker_pid" 2>/dev/null; then
            echo "--- worker.log tail ---" >&2; tail -40 "$work_dir/worker-$3.log" >&2
            die "worker ($3) exited before its descriptor became ready"
        fi
        sleep 0.5
    done
    tail -40 "$work_dir/worker-$3.log" >&2
    die "worker ($3) descriptor not ready within 120s"
}

gateway_ready_wait() {
    local gateway_pid=$1 gateway_port=$2
    for _ in $(seq 1 120); do
        if curl -sf "http://127.0.0.1:${gateway_port}/readyz" >/dev/null 2>&1; then return 0; fi
        if ! kill -0 "$gateway_pid" 2>/dev/null; then
            echo "--- gateway.log tail ---" >&2; tail -40 "$work_dir/gateway-$3.log" >&2
            die "gateway ($3) exited before ready"
        fi
        sleep 0.5
    done
    tail -40 "$work_dir/gateway-$3.log" >&2
    die "gateway ($3) not ready within 60s"
}

fail_gate=""
for leg in off on; do
    log "=== leg offload=$leg ==="
    worker_port=$(free_port)
    gateway_port=$(free_port)

    log "starting one worker (port $worker_port, lane=$lane, pool=$leg)"
    local_envs=()
    for entry in "${worker_env_base[@]}"; do local_envs+=("$entry"); done
    local_envs+=("IZWI_WORKER_BIND=127.0.0.1:${worker_port}")
    local_envs+=("IZWI_WORKER_ID=worker-ds4")
    if [[ "$lane" == "cpu" ]]; then
        for entry in "${worker_cpu_env[@]}"; do local_envs+=("$entry"); done
    else
        for entry in "${worker_metal_env[@]}"; do
            entry="${entry/<host metal:<registryID>>/${metal_device_id}}"
            local_envs+=("$entry")
        done
    fi
    if [[ "$leg" == "on" ]]; then
        local_envs+=("IZWI_KV_HOST_POOL_BUDGET_BYTES=${pool_budget_bytes}")
    fi
    env "${local_envs[@]}" "$worker_bin" >"$work_dir/worker-$leg.log" 2>&1 &
    worker_pid=$!
    worker_ready_wait "$worker_pid" "$worker_port" "$leg"
    log "worker ready"

    log "starting gateway on 127.0.0.1:$gateway_port (offload=$leg)"
    env \
        "IZWI_GATEWAY_API_KEY=${api_key}" \
        "IZWI_GATEWAY_WORKER_CREDENTIAL_ID=${credential_id}" \
        "IZWI_GATEWAY_WORKER_BEARER_TOKEN=${bearer_token}" \
        "IZWI_GATEWAY_TENANT_MAX_CONCURRENT=${concurrency}" \
        "IZWI_GATEWAY_TENANT_REQUESTS_PER_MINUTE=3000" \
        "IZWI_GATEWAY_TENANT_BURST_REQUESTS=256" \
        "IZWI_GATEWAY_WORKER_STATUS_POLL_MS=200" \
        "IZWI_GATEWAY_WORKER_STATUS_TTL_MS=5000" \
        "$gateway_bin" --role gateway --host 127.0.0.1 --port "$gateway_port" \
        --public-model "$public_model" \
        --gateway-max-in-flight "$concurrency" \
        --gateway-worker-approval "http://127.0.0.1:${worker_port}|chat|${public_model}|${deployment_id}|1" \
        >"$work_dir/gateway-$leg.log" 2>&1 &
    gateway_pid=$!
    gateway_ready_wait "$gateway_pid" "$gateway_port" "$leg"
    log "gateway ready"

    gateway_url="http://127.0.0.1:${gateway_port}"
    fetch_counters() {
        curl -sf \
            -H "Authorization: Bearer ${bearer_token}" \
            -H "x-izwi-service-credential-id: ${credential_id}" \
            "http://127.0.0.1:${worker_port}/internal/v1/metrics/prometheus"
    }

    warm_up() {
        local id=$1
        curl -sf -X POST "$gateway_url/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer ${api_key}" \
            -H "x-request-id: ds4-warmup-$leg-$id" \
            -d "{\"model\":\"${public_model}\",\"messages\":[{\"role\":\"user\",\"content\":\"warm up request $id\"}],\"max_tokens\":4,\"stream\":false}" \
            >/dev/null
    }
    log "warm-up"
    warm_up 1
    warm_up 2

    fetch_counters >"$work_dir/counters-$leg-before.txt"

    metadata=$(jq -n \
        --arg sha "$deployed_sha" \
        --arg lane "$lane" \
        --arg model "$public_model" \
        --arg revision "$artifact_revision" \
        --arg deployment "$deployment_id" \
        --arg offload "$leg" \
        --argjson requests "$requests" \
        --argjson prefix "$prefix_tokens" \
        --argjson suffix "$suffix_tokens" \
        --argjson threshold "$chunk_threshold" \
        --argjson kv_page_size "$kv_page_size" \
        --argjson concurrency "$concurrency" \
        --argjson pool_budget_bytes "$pool_budget_bytes" \
        --argjson high_watermark "$high_watermark" \
        --argjson low_watermark "$low_watermark" \
        --arg vocab "a,b,c" \
        '{deployed_sha: $sha, topology: "single-node-one-worker-loopback", backend: $lane,
          model: $model, artifact_revision: $revision, deployment_id: $deployment,
          fixture: "tiny synthetic qwen38 hybrid; no speed claim", offload: $offload,
          requests: $requests, prefix_tokens: $prefix, suffix_tokens: $suffix,
          chunked_prefill_threshold: $threshold, kv_page_size: $kv_page_size,
          concurrency: $concurrency, pool_budget_bytes: $pool_budget_bytes,
          high_watermark: $high_watermark, low_watermark: $low_watermark,
          vocab: $vocab}')
    log "running shared workload (requests=$requests concurrency=$concurrency)"
    python3 "$repo_root/scripts/bench/run-gateway-chat-benchmark.py" \
        --gateway "$gateway_url" \
        --api-key "$api_key" \
        --model "$public_model" \
        --requests "$requests" \
        --concurrency "$concurrency" \
        --max-tokens "$max_tokens" \
        --stream \
        --workload shared \
        --vocab "a,b,c" \
        --prefix-tokens "$prefix_tokens" \
        --suffix-tokens "$suffix_tokens" \
        --max-retries "${DS4_MAX_RETRIES:-40}" \
        --metadata "$metadata" \
        --output "$work_dir/ds4-${lane}-offload-${leg}.json"

    # Sequential trailer: while the workload runs at concurrency, some client
    # always holds a table reference on the shared chain, so no prepare ever
    # sees it unreferenced over the watermark. Two sequential shared-prefix
    # requests after the workload give the manager the safe point it needs:
    # the first trailer's tick demotes the released chain and its lookup
    # continues into the host tier, promoting the pages back at its commit.
    log "running sequential trailer (drives demote+promote on the released chain)"
    trailer_metadata=$(jq -n         --arg sha "$deployed_sha"         --arg lane "$lane"         --arg offload "$leg"         '{deployed_sha: $sha, backend: $lane, offload: $offload, phase: "trailer"}')
    python3 "$repo_root/scripts/bench/run-gateway-chat-benchmark.py" \
        --gateway "$gateway_url" \
        --api-key "$api_key" \
        --model "$public_model" \
        --requests 2 \
        --concurrency 1 \
        --max-tokens "$max_tokens" \
        --stream \
        --workload shared \
        --vocab "a,b,c" \
        --prefix-tokens "$prefix_tokens" \
        --suffix-tokens "$suffix_tokens" \
        --max-retries "${DS4_MAX_RETRIES:-40}" \
        --metadata "$trailer_metadata" \
        --output "$work_dir/ds4-${lane}-trailer-${leg}.json"

    fetch_counters >"$work_dir/counters-$leg-after.txt"

    jq -n \
        --slurpfile benchmark "$work_dir/ds4-${lane}-offload-${leg}.json" \
        --rawfile before "$work_dir/counters-$leg-before.txt" \
        --rawfile after "$work_dir/counters-$leg-after.txt" \
        '
        def counters(text):
            [text | scan("(izwi_engine_[a-z_]+) ([0-9]+)") | {(.[0]): (.[1] | tonumber)}] | add // {};
        def deltas(after_map; before_map):
            after_map | with_entries(.value -= (before_map[.key] // 0));
        {
            schema: "izwi.ds4-offload-benchmark.v1",
            lane: $benchmark[0].metadata.backend,
            offload: $benchmark[0].metadata.offload,
            benchmark: $benchmark[0],
            worker: {
                counters_before: counters($before),
                counters_after: counters($after),
                counters_delta: deltas(counters($after); counters($before))
            }
        }' >"$output_dir/ds4-${lane}-offload-${leg}.json"

    completed=$(jq -r '.benchmark.completed' "$output_dir/ds4-${lane}-offload-${leg}.json")
    expected=$requests
    demotions=$(jq -r '.worker.counters_delta["izwi_engine_kv_cache_demotions_total"] // 0' "$output_dir/ds4-${lane}-offload-${leg}.json")
    promotions=$(jq -r '.worker.counters_delta["izwi_engine_kv_cache_promotions_total"] // 0' "$output_dir/ds4-${lane}-offload-${leg}.json")
    host_pages=$(jq -r '.worker.counters_after["izwi_engine_kv_cache_host_pages"] // 0' "$output_dir/ds4-${lane}-offload-${leg}.json")
    budget_pages=$((pool_budget_bytes / page_bytes))
    reused=$(jq -r '.benchmark.reused_tokens.total // 0' "$output_dir/ds4-${lane}-offload-${leg}.json" 2>/dev/null || echo 0)
    ttft_p50=$(jq -r '.benchmark.ttft_ms.p50 // 0' "$output_dir/ds4-${lane}-offload-${leg}.json")
    log "leg=$leg completed=$completed/$expected demotions=$demotions promotions=$promotions host_pages=$host_pages/$budget_pages ttft_p50=$ttft_p50"

    [[ "$completed" -eq "$expected" ]] || fail_gate="${fail_gate} ${leg}:completed=$completed/$expected"
    if [[ "$leg" == "on" ]]; then
        [[ "$demotions" -ge 1 ]] || fail_gate="${fail_gate} on:demotions=$demotions"
        [[ "$promotions" -ge 1 ]] || fail_gate="${fail_gate} on:promotions=$promotions"
        [[ "$host_pages" -le "$budget_pages" ]] || fail_gate="${fail_gate} on:host_pages=$host_pages>$budget_pages"
    fi

    # Stop both processes so the next leg starts from a fresh cache state.
    kill "$gateway_pid" "$worker_pid" 2>/dev/null || true
    wait "$gateway_pid" "$worker_pid" 2>/dev/null || true
    gateway_pid=""; worker_pid=""
    sleep 1
done

if [[ -f "$output_dir/ds4-${lane}-offload-off.json" && -f "$output_dir/ds4-${lane}-offload-on.json" ]]; then
    jq -n -S \
        --slurpfile off "$output_dir/ds4-${lane}-offload-off.json" \
        --slurpfile on "$output_dir/ds4-${lane}-offload-on.json" \
        '{
            schema: "izwi.ds4-offload-summary.v1",
            lane: $off[0].lane,
            ttft_ms: {
                off_p50: $off[0].benchmark.ttft_ms.p50,
                off_p95: $off[0].benchmark.ttft_ms.p95,
                on_p50: $on[0].benchmark.ttft_ms.p50,
                on_p95: $on[0].benchmark.ttft_ms.p95,
                p50_delta_ms: (($on[0].benchmark.ttft_ms.p50 // 0) - ($off[0].benchmark.ttft_ms.p50 // 0))
            },
            offload_counter_deltas: {
                off: {
                    demotions: ($off[0].worker.counters_delta["izwi_engine_kv_cache_demotions_total"] // 0),
                    promotions: ($off[0].worker.counters_delta["izwi_engine_kv_cache_promotions_total"] // 0),
                    host_pages_end: ($off[0].worker.counters_after["izwi_engine_kv_cache_host_pages"] // 0)
                },
                on: {
                    demotions: ($on[0].worker.counters_delta["izwi_engine_kv_cache_demotions_total"] // 0),
                    promotions: ($on[0].worker.counters_delta["izwi_engine_kv_cache_promotions_total"] // 0),
                    host_pages_end: ($on[0].worker.counters_after["izwi_engine_kv_cache_host_pages"] // 0)
                }
            },
            off_worker_counter_deltas: $off[0].worker,
            on_worker_counter_deltas: $on[0].worker
        }' >"$output_dir/ds4-${lane}-summary.json"
    log "summary: $output_dir/ds4-${lane}-summary.json"
fi

if [[ -n "$fail_gate" ]]; then
    die "benchmark gate failed:$fail_gate"
fi
log "done; manifests in $output_dir; per-run artifacts in $work_dir"
