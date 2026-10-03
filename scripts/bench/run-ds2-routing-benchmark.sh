#!/usr/bin/env bash
#
# DS2 routing benchmark: runs the multi_turn gateway chat workload across TWO
# local workers + one local gateway on the requested backend lane, once with
# routing (cache affinity + conversation pinning) off and once on, and writes
# per-leg manifests with per-worker managed-KV counter deltas under
# benchmarks/manifests/.
#
# The model is the tiny synthetic qwen38 hybrid fixture: this is routing-path
# evidence (per-worker attach/attach distribution and TTFT shape), not a speed
# claim. Hard gates: every leg completes all conversations and the routing-on
# leg shows strictly more committed-prefix attaches than the routing-off leg
# (a conversation's turns must stick to a warm worker to reuse its prefix).
# A violating run fails the script.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
public_model="Qwen3.8-27B-FP8"
lane="cpu"
conversations=8
turns=4
concurrency=4
max_tokens=8
prefix_tokens=64
suffix_tokens=16
models_dir="${repo_root}/target/ds2-bench/models"
output_dir="${repo_root}/benchmarks/manifests"
worker_bin="${repo_root}/target/debug/izwi-serving-worker"
gateway_bin="${repo_root}/target/debug/izwi-server"
skip_build=0
dry_run=0

usage() {
    cat <<'EOF'
Usage: scripts/bench/run-ds2-routing-benchmark.sh [options]

Options:
  --lane cpu|metal       Backend lane (default: cpu)
  --conversations N      Multi-turn conversations per leg (default: 8)
  --turns N              Turns per conversation (default: 4)
  --concurrency N        Client concurrency (conversations in flight; default: 4)
  --max-tokens N         Output tokens per turn (default: 8)
  --prefix-tokens N      Approximate per-conversation system prefix length in words (default: 64)
  --suffix-tokens N      Approximate per-turn user message length in words (default: 16)
  --models-dir PATH      Fixture models root (default: target/ds2-bench/models)
  --output-dir PATH      Manifest output directory (default: benchmarks/manifests)
  --worker-bin PATH      Worker binary (default: target/debug/izwi-serving-worker)
  --gateway-bin PATH     Gateway binary (default: target/debug/izwi-server)
  --skip-build           Reuse existing binaries; no cargo build
  --dry-run              Print the resolved plan (env, commands) and exit
  -h, --help             Show this help

Each leg (routing off, then routing on) starts two fresh workers and one
gateway, so per-leg counter deltas never see the other leg's cache state.
Workers start first; the gateway fails closed if it wins that race. This is an
opt-in benchmark over a synthetic fixture; it makes no speed claim and
certifies no hardware. CUDA is not a lane here.
EOF
}

log() { echo "[ds2] $*"; }

die() { echo "[ds2] error: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lane) lane="${2:-}"; shift 2 ;;
        --conversations) conversations="${2:-}"; shift 2 ;;
        --turns) turns="${2:-}"; shift 2 ;;
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
[[ "$conversations" =~ ^[0-9]+$ && "$conversations" -gt 0 ]] || die "--conversations must be a positive integer"
[[ "$turns" =~ ^[0-9]+$ && "$turns" -ge 2 && "$turns" -le 16 ]] || die "--turns must be between 2 and 16"
[[ "$concurrency" =~ ^[0-9]+$ && "$concurrency" -ge 1 && "$concurrency" -le 64 ]] || die "--concurrency must be between 1 and 64"
[[ "$max_tokens" =~ ^[0-9]+$ && "$max_tokens" -gt 0 && "$max_tokens" -le 32 ]] || die "--max-tokens must be between 1 and 32 (worker advertises 32)"
[[ "$prefix_tokens" =~ ^[0-9]+$ && "$prefix_tokens" -gt 0 ]] || die "--prefix-tokens must be a positive integer"
[[ "$suffix_tokens" =~ ^[0-9]+$ && "$suffix_tokens" -gt 0 ]] || die "--suffix-tokens must be a positive integer"
command -v jq >/dev/null || die "jq is required"
command -v python3 >/dev/null || die "python3 is required"

# Same sizing rationale as the DS1.5 rig: the chunked-prefill threshold must
# cover the whole prompt (word count runs ~1.6x tokens including the ~43-token
# Xhigh block), and the KV page must stay larger than that structural block so
# unrelated conversations share nothing by construction (kv_page_size=64).
prompt_words=$((prefix_tokens + turns * suffix_tokens))
chunk_threshold=$((prompt_words * 2 + 64))
kv_page_size=64
deployment_id="ds2-${lane}-bench-v1"
credential_id="ds2-bench-credential"
bearer_token=$(openssl rand -hex 24 2>/dev/null || echo "ds2-local-bearer-0000000000000000")
api_key=$(openssl rand -hex 24 2>/dev/null || echo "ds2-local-apikey-000000000000000")
artifact_revision="017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
fixture_model_dir="${models_dir}/${public_model}"
work_dir="${repo_root}/target/ds2-bench/run-$(date +%Y%m%d-%H%M%S)"

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
    "IZWI_MANAGED_PREFIX_CACHE_SALT=ds2-bench-salt"
    "IZWI_ENABLE_CHUNKED_PREFILL=1"
    "IZWI_CHUNKED_PREFILL_THRESHOLD=${chunk_threshold}"
    "IZWI_KV_PAGE_SIZE=64"
    "IZWI_CUDA_MTP=off"
    "IZWI_MAX_PREFIX_CACHE_PAGES=256"
    "IZWI_MAX_SEQUENCE_LENGTH=4096"
    "RUST_LOG=warn"
)
worker_cpu_env=(
    "IZWI_BACKEND=cpu"
    "IZWI_WORKER_CPU_THREADS=${DS2_WORKER_CPU_THREADS:-2}"
    "IZWI_CPU_MEMORY_BUDGET_BYTES=2147483648"
    "IZWI_WORKER_HOST_MEMORY_LIMIT_BYTES=2147483648"
)
worker_metal_env=(
    "IZWI_BACKEND=metal"
    "IZWI_METAL_DEVICE_ORDINAL=0"
    "IZWI_WORKER_EXPECTED_DEVICE_ID=<host metal:<registryID>>"
)

if [[ "$dry_run" == 1 ]]; then
    echo "ds2 dry-run plan"
    echo "lane=$lane conversations=$conversations turns=$turns concurrency=$concurrency"
    echo "max_tokens=$max_tokens prefix_tokens=$prefix_tokens suffix_tokens=$suffix_tokens"
    echo "chunked_prefill_threshold=$chunk_threshold deployment_id=$deployment_id"
    echo "fixture_model_dir=$fixture_model_dir"
    echo "output_dir=$output_dir worker_bin=$worker_bin gateway_bin=$gateway_bin"
    echo "per-leg gateway env:"
    echo "  off: (defaults; IZWI_GATEWAY_ROUTER_CACHE_AFFINITY unset, IZWI_GATEWAY_SESSION_PIN unset)"
    echo "  on:  IZWI_GATEWAY_ROUTER_CACHE_AFFINITY=on IZWI_GATEWAY_SESSION_PIN=on"
    echo "gateway_cmd: $gateway_bin --role gateway ... --gateway-worker-approval http://127.0.0.1:<portA>|chat|$public_model|$deployment_id|1 --gateway-worker-approval http://127.0.0.1:<portB>|chat|$public_model|$deployment_id|1"
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

gateway_pid=""
worker_a_pid=""
worker_b_pid=""
cleanup() {
    [[ -n "$gateway_pid" ]] && kill "$gateway_pid" 2>/dev/null || true
    [[ -n "$worker_a_pid" ]] && kill "$worker_a_pid" 2>/dev/null || true
    [[ -n "$worker_b_pid" ]] && kill "$worker_b_pid" 2>/dev/null || true
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

worker_ready_wait() {
    local worker_pid=$1 worker_port=$2 name=$3
    for _ in $(seq 1 240); do
        code=$(curl -s -o /dev/null -w '%{http_code}' \
            -H "Authorization: Bearer ${bearer_token}" \
            -H "x-izwi-service-credential-id: ${credential_id}" \
            "http://127.0.0.1:${worker_port}/internal/v1/worker" || true)
        if [[ "$code" == "200" ]]; then return 0; fi
        if ! kill -0 "$worker_pid" 2>/dev/null; then
            echo "--- $name.log tail ---" >&2; tail -40 "$work_dir/$name.log" >&2
            die "$name exited before its descriptor became ready"
        fi
        sleep 0.5
    done
    tail -40 "$work_dir/$name.log" >&2
    die "$name descriptor not ready within 120s"
}

fail_gate=""
for leg in off on; do
    log "=== leg routing=$leg ==="
    worker_a_port=$(free_port)
    worker_b_port=$(free_port)
    gateway_port=$(free_port)

    start_worker() {
        local port=$1 worker=$2 name=$3
        local -a envs=()
        for entry in "${worker_env_base[@]}"; do envs+=("$entry"); done
        envs+=("IZWI_WORKER_BIND=127.0.0.1:${port}")
        envs+=("IZWI_WORKER_ID=${worker}")
        if [[ "$lane" == "cpu" ]]; then
            for entry in "${worker_cpu_env[@]}"; do envs+=("$entry"); done
        else
            for entry in "${worker_metal_env[@]}"; do
                entry="${entry/<host metal:<registryID>>/${metal_device_id}}"
                envs+=("$entry")
            done
        fi
        env "${envs[@]}" "$worker_bin" >"$work_dir/$name.log" 2>&1 &
        echo $!
    }

    log "starting two workers (ports $worker_a_port / $worker_b_port, lane=$lane)"
    worker_a_pid=$(start_worker "$worker_a_port" "worker-a" "worker-a-$leg")
    worker_b_pid=$(start_worker "$worker_b_port" "worker-b" "worker-b-$leg")
    worker_ready_wait "$worker_a_pid" "$worker_a_port" "worker-a-$leg"
    worker_ready_wait "$worker_b_pid" "$worker_b_port" "worker-b-$leg"
    log "workers ready"

    routing_envs=()
    if [[ "$leg" == "on" ]]; then
        routing_envs+=(
            "IZWI_GATEWAY_ROUTER_CACHE_AFFINITY=on"
            "IZWI_GATEWAY_SESSION_PIN=on"
        )
    fi

    log "starting gateway on 127.0.0.1:$gateway_port (routing=$leg)"
    env \
        "IZWI_GATEWAY_API_KEY=${api_key}" \
        "IZWI_GATEWAY_WORKER_CREDENTIAL_ID=${credential_id}" \
        "IZWI_GATEWAY_WORKER_BEARER_TOKEN=${bearer_token}" \
        "IZWI_GATEWAY_TENANT_MAX_CONCURRENT=${concurrency}" \
        "IZWI_GATEWAY_TENANT_REQUESTS_PER_MINUTE=3000" \
        "IZWI_GATEWAY_TENANT_BURST_REQUESTS=256" \
        "IZWI_GATEWAY_WORKER_STATUS_POLL_MS=200" \
        "IZWI_GATEWAY_WORKER_STATUS_TTL_MS=5000" \
        ${routing_envs[@]+"${routing_envs[@]}"} \
        "$gateway_bin" --role gateway --host 127.0.0.1 --port "$gateway_port" \
        --public-model "$public_model" \
        --gateway-max-in-flight "$concurrency" \
        --gateway-worker-approval "http://127.0.0.1:${worker_a_port}|chat|${public_model}|${deployment_id}|1" \
        --gateway-worker-approval "http://127.0.0.1:${worker_b_port}|chat|${public_model}|${deployment_id}|1" \
        >"$work_dir/gateway-$leg.log" 2>&1 &
    gateway_pid=$!
    gateway_ready_wait "$gateway_pid" "$gateway_port" "$leg"
    log "gateway ready"

    gateway_url="http://127.0.0.1:${gateway_port}"
    fetch_counters() {
        local port=$1
        curl -sf \
            -H "Authorization: Bearer ${bearer_token}" \
            -H "x-izwi-service-credential-id: ${credential_id}" \
            "http://127.0.0.1:${port}/internal/v1/metrics/prometheus"
    }

    warm_up() {
        local id=$1
        curl -sf -X POST "$gateway_url/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer ${api_key}" \
            -H "x-request-id: ds2-warmup-$leg-$id" \
            -d "{\"model\":\"${public_model}\",\"messages\":[{\"role\":\"user\",\"content\":\"warm up request $id\"}],\"max_tokens\":4,\"stream\":false}" \
            >/dev/null
    }
    log "warm-up"
    warm_up 1
    warm_up 2

    fetch_counters "$worker_a_port" >"$work_dir/counters-$leg-a-before.txt"
    fetch_counters "$worker_b_port" >"$work_dir/counters-$leg-b-before.txt"

    metadata=$(jq -n \
        --arg sha "$deployed_sha" \
        --arg lane "$lane" \
        --arg model "$public_model" \
        --arg revision "$artifact_revision" \
        --arg deployment "$deployment_id" \
        --arg routing "$leg" \
        --argjson conversations "$conversations" \
        --argjson turns "$turns" \
        --argjson prefix "$prefix_tokens" \
        --argjson suffix "$suffix_tokens" \
        --argjson threshold "$chunk_threshold" \
        --argjson kv_page_size "$kv_page_size" \
        --argjson concurrency "$concurrency" \
        --arg vocab "a,b,c" \
        '{deployed_sha: $sha, topology: "single-node-two-workers-loopback", backend: $lane,
          model: $model, artifact_revision: $revision, deployment_id: $deployment,
          fixture: "tiny synthetic qwen38 hybrid; no speed claim", routing: $routing,
          conversations: $conversations, turns: $turns,
          prefix_tokens: $prefix, suffix_tokens: $suffix,
          chunked_prefill_threshold: $threshold, kv_page_size: $kv_page_size,
          concurrency: $concurrency, vocab: $vocab}')
    log "running multi_turn workload (conversations=$conversations turns=$turns concurrency=$concurrency)"
    python3 "$repo_root/scripts/bench/run-gateway-chat-benchmark.py" \
        --gateway "$gateway_url" \
        --api-key "$api_key" \
        --model "$public_model" \
        --requests "$conversations" \
        --concurrency "$concurrency" \
        --max-tokens "$max_tokens" \
        --stream \
        --workload multi_turn \
        --turns "$turns" \
        --vocab "a,b,c" \
        --prefix-tokens "$prefix_tokens" \
        --suffix-tokens "$suffix_tokens" \
        --max-retries "${DS2_MAX_RETRIES:-40}" \
        --metadata "$metadata" \
        --output "$work_dir/ds2-${lane}-routing-${leg}.json"

    fetch_counters "$worker_a_port" >"$work_dir/counters-$leg-a-after.txt"
    fetch_counters "$worker_b_port" >"$work_dir/counters-$leg-b-after.txt"

    jq -n \
        --slurpfile benchmark "$work_dir/ds2-${lane}-routing-${leg}.json" \
        --rawfile a_before "$work_dir/counters-$leg-a-before.txt" \
        --rawfile a_after "$work_dir/counters-$leg-a-after.txt" \
        --rawfile b_before "$work_dir/counters-$leg-b-before.txt" \
        --rawfile b_after "$work_dir/counters-$leg-b-after.txt" \
        '
        def counters(text):
            [text | scan("(izwi_engine_[a-z_]+) ([0-9]+)") | {(.[0]): (.[1] | tonumber)}] | add // {};
        def deltas(after_map; before_map):
            after_map | with_entries(.value -= (before_map[.key] // 0));
        {
            schema: "izwi.ds2-routing-benchmark.v1",
            lane: $benchmark[0].metadata.backend,
            routing: $benchmark[0].metadata.routing,
            benchmark: $benchmark[0],
            workers: {
                "worker-a": {
                    counters_before: counters($a_before),
                    counters_after: counters($a_after),
                    counters_delta: deltas(counters($a_after); counters($a_before))
                },
                "worker-b": {
                    counters_before: counters($b_before),
                    counters_after: counters($b_after),
                    counters_delta: deltas(counters($b_after); counters($b_before))
                }
            }
        }' >"$output_dir/ds2-${lane}-routing-${leg}.json"

    completed=$(jq -r '.benchmark.completed' "$output_dir/ds2-${lane}-routing-${leg}.json")
    expected=$conversations
    attaches_a=$(jq -r '.workers["worker-a"].counters_delta["izwi_engine_tensor_snapshot_attaches_total"] // 0' "$output_dir/ds2-${lane}-routing-${leg}.json")
    attaches_b=$(jq -r '.workers["worker-b"].counters_delta["izwi_engine_tensor_snapshot_attaches_total"] // 0' "$output_dir/ds2-${lane}-routing-${leg}.json")
    attaches=$((attaches_a + attaches_b))
    ttft_p50=$(jq -r '.benchmark.ttft_ms.p50 // 0' "$output_dir/ds2-${lane}-routing-${leg}.json")
    log "leg=$leg completed=$completed/$expected turns attaches=$attaches (a=$attaches_a b=$attaches_b) ttft_p50=$ttft_p50"

    [[ "$completed" -eq "$expected" ]] || fail_gate="${fail_gate} ${leg}:completed=$completed/$expected"
    [[ "$attaches" -ge 1 ]] || fail_gate="${fail_gate} ${leg}:attaches=$attaches"

    # Stop both processes so the next leg starts from a fresh cache state.
    kill "$gateway_pid" "$worker_a_pid" "$worker_b_pid" 2>/dev/null || true
    wait "$gateway_pid" "$worker_a_pid" "$worker_b_pid" 2>/dev/null || true
    gateway_pid=""; worker_a_pid=""; worker_b_pid=""
    sleep 1
done

attaches_off=$(jq -r '([.workers[].counters_delta["izwi_engine_tensor_snapshot_attaches_total"] // 0] | add) ' "$output_dir/ds2-${lane}-routing-off.json")
attaches_on=$(jq -r '([.workers[].counters_delta["izwi_engine_tensor_snapshot_attaches_total"] // 0] | add) ' "$output_dir/ds2-${lane}-routing-on.json")
if [[ "$attaches_on" -le "$attaches_off" ]]; then
    fail_gate="${fail_gate} routing-on attaches ($attaches_on) did not exceed routing-off attaches ($attaches_off)"
fi

if [[ -f "$output_dir/ds2-${lane}-routing-off.json" && -f "$output_dir/ds2-${lane}-routing-on.json" ]]; then
    jq -n -S \
        --slurpfile off "$output_dir/ds2-${lane}-routing-off.json" \
        --slurpfile on "$output_dir/ds2-${lane}-routing-on.json" \
        --argjson attaches_off "$attaches_off" \
        --argjson attaches_on "$attaches_on" \
        '{
            schema: "izwi.ds2-routing-summary.v1",
            lane: $off[0].lane,
            ttft_ms: {
                off_p50: $off[0].benchmark.ttft_ms.p50,
                off_p95: $off[0].benchmark.ttft_ms.p95,
                on_p50: $on[0].benchmark.ttft_ms.p50,
                on_p95: $on[0].benchmark.ttft_ms.p95,
                p50_delta_ms: (($on[0].benchmark.ttft_ms.p50 // 0) - ($off[0].benchmark.ttft_ms.p50 // 0))
            },
            attaches_total: {off: $attaches_off, on: $attaches_on},
            off_worker_counter_deltas: $off[0].workers,
            on_worker_counter_deltas: $on[0].workers
        }' >"$output_dir/ds2-${lane}-summary.json"
    log "summary: $output_dir/ds2-${lane}-summary.json"
fi

if [[ -n "$fail_gate" ]]; then
    die "benchmark gate failed:$fail_gate"
fi
log "done; manifests in $output_dir; per-run artifacts in $work_dir"
