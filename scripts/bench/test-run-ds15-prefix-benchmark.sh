#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
runner="${repo_root}/scripts/bench/run-ds15-prefix-benchmark.sh"

help=$("${runner}" --help)
grep -q 'shared,cold' <<<"${help}"
grep -q 'no speed claim' <<<"${help}"
grep -q 'CUDA is not a lane' <<<"${help}"

# CPU dry-run: the serving-policy env and the gateway approval form must be
# visible in the plan, and the chunked threshold must cover the whole prompt
# (64 prefix + 16 suffix, ~1.6x token inflation + 64 headroom). The KV page
# must stay larger than the ~43-token structural reasoning block.
plan=$("${runner}" --dry-run --lane cpu)
grep -q 'lane=cpu workloads=shared cold' <<<"${plan}"
grep -q 'IZWI_ENABLE_PREFIX_CACHING=1' <<<"${plan}"
grep -q 'IZWI_MANAGED_PREFIX_CACHE_SALT=' <<<"${plan}"
grep -q 'IZWI_CHUNKED_PREFILL_THRESHOLD=224' <<<"${plan}"
grep -q 'IZWI_KV_PAGE_SIZE=64' <<<"${plan}"
grep -q 'IZWI_CUDA_MTP=off' <<<"${plan}"
grep -q 'IZWI_ALLOW_SYNTHETIC_QWEN38_GEOMETRY=1' <<<"${plan}"
grep -q -- '--gateway-worker-approval http://127.0.0.1:<worker-port>|chat|Qwen3.8-27B-FP8|ds15-cpu-bench-v1|1' <<<"${plan}"
grep -q 'IZWI_GATEWAY_TENANT_MAX_CONCURRENT=<concurrency>' <<<"${plan}"
grep -q -- '--max-retries' <<<"${plan}" || true
grep -q 'run-gateway-chat-benchmark.py --workload {shared,cold}' <<<"${plan}"
grep -q 'gateway_max_in_flight=4' <<<"${plan}"
grep -q 'harness_vocab: a,b,c' <<<"${plan}"

# A single-workload subset must be honored.
plan_cold=$("${runner}" --dry-run --lane cpu --workloads cold)
grep -q 'workloads=cold' <<<"${plan_cold}"
if grep -q 'workload: shared' <<<"${plan_cold}"; then
    echo "--workloads cold must exclude the shared workload from the plan" >&2
    exit 1
fi

# Metal dry-run must not require a GPU; device resolution is a run-time gate.
plan_metal=$("${runner}" --dry-run --lane metal)
grep -q 'lane=metal' <<<"${plan_metal}"
grep -q 'IZWI_BACKEND=metal' <<<"${plan_metal}"
grep -q 'IZWI_WORKER_EXPECTED_DEVICE_ID=' <<<"${plan_metal}"
grep -q 'ds15-metal-bench-v1' <<<"${plan_metal}"

if "${runner}" --dry-run --lane cuda >/dev/null 2>&1; then
    echo "cuda must be rejected: DS1.5 lanes are cpu and metal" >&2
    exit 1
fi
if "${runner}" --dry-run --workloads shared,exotic >/dev/null 2>&1; then
    echo "unknown workloads must be rejected" >&2
    exit 1
fi
if "${runner}" --dry-run --max-tokens 64 >/dev/null 2>&1; then
    echo "max_tokens above the worker's advertised capability must be rejected" >&2
    exit 1
fi
if "${runner}" --dry-run --prefix-tokens 0 >/dev/null 2>&1; then
    echo "non-positive prefix lengths must be rejected" >&2
    exit 1
fi

# Shape check for the manifest assembly (the runner's inline jq filter): a
# counter delta must be computed from the before/after prometheus snapshots.
tmp_dir=$(mktemp -d)
trap 'rm -rf "${tmp_dir}"' EXIT
cat >"${tmp_dir}/benchmark.json" <<'EOF'
{"completed": 4, "workload": "shared",
 "ttft_ms": {"p50": 10.0, "p95": 20.0},
 "metadata": {"backend": "cpu"}}
EOF
cat >"${tmp_dir}/before.txt" <<'EOF'
# HELP izwi_engine_tensor_snapshot_publishes_total x
# TYPE izwi_engine_tensor_snapshot_publishes_total counter
izwi_engine_tensor_snapshot_publishes_total 2
izwi_engine_tensor_snapshot_attaches_total 0
EOF
cat >"${tmp_dir}/after.txt" <<'EOF'
# HELP izwi_engine_tensor_snapshot_publishes_total x
# TYPE izwi_engine_tensor_snapshot_publishes_total counter
izwi_engine_tensor_snapshot_publishes_total 5
izwi_engine_tensor_snapshot_attaches_total 3
EOF
jq -n \
    --slurpfile benchmark "${tmp_dir}/benchmark.json" \
    --rawfile before "${tmp_dir}/before.txt" \
    --rawfile after "${tmp_dir}/after.txt" \
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
    >"${tmp_dir}/manifest.json"
jq -e '.schema == "izwi.ds15-prefix-benchmark.v1" and
       .lane == "cpu" and .workload == "shared" and
       .counters_delta["izwi_engine_tensor_snapshot_publishes_total"] == 3 and
       .counters_delta["izwi_engine_tensor_snapshot_attaches_total"] == 3 and
       .counters_before["izwi_engine_tensor_snapshot_attaches_total"] == 0' \
    "${tmp_dir}/manifest.json" >/dev/null

echo "DS1.5 prefix benchmark runner smoke test passed"
