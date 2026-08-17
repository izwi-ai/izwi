#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cell_runner="${repo_root}/scripts/bench/run-qwen38-cuda-evidence.sh"
certifier="${repo_root}/scripts/bench/certify-qwen38-mtp-evidence.sh"
tmp_dir=$(mktemp -d)
trap 'rm -rf "${tmp_dir}"' EXIT

help=$(${certifier} --help)
grep -q 'implemented_unvalidated' <<<"${help}"
grep -q 'runtime_validated' <<<"${help}"
grep -q 'performance_certified' <<<"${help}"

if ${cell_runner} --output "${tmp_dir}/missing-depth" --dry-run >/dev/null 2>&1; then
    echo "Qwen3.8 evidence must require an explicit MTP depth" >&2
    exit 1
fi

for depth in 0 1 2 3; do
    ${cell_runner} --mtp-depth "${depth}" --output "${tmp_dir}/dry-${depth}" \
        --dry-run >/dev/null
done

${certifier} --baseline "${tmp_dir}/dry-0" --depth-1 "${tmp_dir}/dry-1" \
    --depth-2 "${tmp_dir}/dry-2" --depth-3 "${tmp_dir}/dry-3" \
    --output "${tmp_dir}/paired-dry" --dry-run >/dev/null
jq -e '.status == "unsupported" and
       .reason == "paired_manifests_validated_without_runtime" and
       .evidence_level == "implemented_unvalidated" and
       .promotion_eligible == false and
       [.cells[].mtp.draft_tokens] == [null,1,2,3]' \
    "${tmp_dir}/paired-dry/certificate.json" >/dev/null

if ${certifier} --baseline "${tmp_dir}/dry-0" --depth-1 "${tmp_dir}/dry-1" \
    --depth-2 "${tmp_dir}/dry-2" --depth-3 "${tmp_dir}/dry-2" \
    --output "${tmp_dir}/reject-dry-depth" --dry-run >/dev/null 2>&1; then
    echo "paired MTP manifests must reject a missing depth" >&2
    exit 1
fi
jq -e '.status == "failed" and .promotion_eligible == false' \
    "${tmp_dir}/reject-dry-depth/certificate.json" >/dev/null

make_runtime_cell() {
    local source="$1"
    local destination="$2"
    local depth="$3"
    local completion_tps="$4"
    local ttft_p95="$5"
    local peak_memory="$6"
    local enabled=false
    local draft_tokens=null
    local rounds=0
    local drafted_tokens=0
    local verified_tokens=0
    if [[ "${depth}" -gt 0 ]]; then
        enabled=true
        draft_tokens="${depth}"
        rounds=10
        drafted_tokens=$((depth * rounds))
        verified_tokens=$((drafted_tokens + rounds))
    fi
    jq --argjson enabled "${enabled}" --argjson draft_tokens "${draft_tokens}" \
       --argjson rounds "${rounds}" --argjson drafted_tokens "${drafted_tokens}" \
       --argjson verified_tokens "${verified_tokens}" \
       --argjson completion_tps "${completion_tps}" --argjson ttft_p95 "${ttft_p95}" \
       --argjson peak_memory "${peak_memory}" '
      .status = "passed" |
      .reason = "measured_qwen38_cuda_profile_evidence_passed" |
      .evidence_level = "runtime_validated" |
      .promotion_eligible = false |
      .device = {uuid:"GPU-test-uuid",name:"NVIDIA L40S",compute_capability:"8.9",
        total_memory_bytes:48000000000,driver_version:"580.1",ordinal:0} |
      .runtime = {hardware_provider:"nvidia-cuda",actual_device_kind:"cuda",
        actual_compute_dtype:"bf16",kv_storage_provider:"cuda-bf16-candidate",
        mtp:{enabled:$enabled,draft_tokens:$draft_tokens,
          source:"loaded_model_diagnostics",runtime_validated:true}} |
      .memory = {source:"nvidia-smi-memory.used",sample_interval_seconds:1,
        sample_count:8,peak_device_memory_used_bytes:$peak_memory,
        minimum_device_memory_free_bytes:1000000000} |
      .measurements = [{name:"decode-short",target_prompt_words:32,
        observed_prompt_tokens_avg:40,completion_tokens_avg:128,concurrent:1,
        ttft_ms:{count:3,avg:($ttft_p95 * 0.8),p50:($ttft_p95 * 0.75),p95:$ttft_p95},
        end_to_end_completion_tps:{count:3,avg:$completion_tps,
          p50:$completion_tps,p95:($completion_tps * 1.02)},
        server_generation_ms:{count:3,avg:1000,p50:1000,p95:1050},
        decode_samples:[
          {index:0,completion_tokens:128,generation_ms:1000,tokens_per_second:128},
          {index:1,completion_tokens:128,generation_ms:1000,tokens_per_second:128},
          {index:2,completion_tokens:128,generation_ms:1000,tokens_per_second:128}
        ],
        mtp:{rounds:$rounds,drafted_tokens:$drafted_tokens,accepted_tokens:$drafted_tokens,
          rejected_rounds:0,bonus_tokens:$rounds,target_verified_tokens:$verified_tokens,
          target_replay_tokens:0},
        end_to_end_ms:{count:3,avg:1100,p50:1100,p95:1150}}]
    ' "${source}" >"${destination}"
}

make_runtime_cell "${tmp_dir}/dry-0/certificate.json" "${tmp_dir}/runtime-0.json" 0 100 10 1000000000
make_runtime_cell "${tmp_dir}/dry-1/certificate.json" "${tmp_dir}/runtime-1.json" 1 108 10 1050000000
make_runtime_cell "${tmp_dir}/dry-2/certificate.json" "${tmp_dir}/runtime-2.json" 2 115 10 1070000000
make_runtime_cell "${tmp_dir}/dry-3/certificate.json" "${tmp_dir}/runtime-3.json" 3 125 10 1100000000

${certifier} --baseline "${tmp_dir}/runtime-0.json" --depth-1 "${tmp_dir}/runtime-1.json" \
    --depth-2 "${tmp_dir}/runtime-2.json" --depth-3 "${tmp_dir}/runtime-3.json" \
    --output "${tmp_dir}/paired-runtime" >/dev/null
jq -e '.status == "passed" and .evidence_level == "runtime_validated" and
       .reason == "paired_mtp_runtime_validated_thresholds_undeclared" and
       .promotion_eligible == false and .recommended_depth == null and
       (.comparisons | length) == 3' \
    "${tmp_dir}/paired-runtime/certificate.json" >/dev/null

thresholds='{"mtp":{"minimum_completion_tps_p50_speedup_ratio":1.05,"maximum_ttft_p95_regression_ratio":1.05,"maximum_peak_device_memory_ratio":1.15}}'
for depth in 0 1 2 3; do
    jq --argjson thresholds "${thresholds}" \
        '.acceptance.performance_thresholds.values = $thresholds' \
        "${tmp_dir}/runtime-${depth}.json" >"${tmp_dir}/performance-${depth}.json"
done
${certifier} --baseline "${tmp_dir}/performance-0.json" \
    --depth-1 "${tmp_dir}/performance-1.json" --depth-2 "${tmp_dir}/performance-2.json" \
    --depth-3 "${tmp_dir}/performance-3.json" --output "${tmp_dir}/paired-performance" >/dev/null
jq -e '.status == "passed" and .evidence_level == "performance_certified" and
       .reason == "paired_mtp_performance_thresholds_passed" and
       .promotion_eligible == true and .certified_depths == [1,2,3] and
       .recommended_depth == 3' "${tmp_dir}/paired-performance/certificate.json" >/dev/null

expect_rejection() {
    local name="$1"
    local candidate="$2"
    if ${certifier} --baseline "${tmp_dir}/runtime-0.json" \
        --depth-1 "${tmp_dir}/runtime-1.json" --depth-2 "${candidate}" \
        --depth-3 "${tmp_dir}/runtime-3.json" --output "${tmp_dir}/reject-${name}" \
        >/dev/null 2>&1; then
        echo "paired MTP evidence accepted rejected case: ${name}" >&2
        exit 1
    fi
    jq -e '.status == "failed" and .promotion_eligible == false' \
        "${tmp_dir}/reject-${name}/certificate.json" >/dev/null
}

jq '.run.git_sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"' \
    "${tmp_dir}/runtime-2.json" >"${tmp_dir}/bad-git.json"
expect_rejection git-sha "${tmp_dir}/bad-git.json"

jq '.run.checkpoint_revision = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"' \
    "${tmp_dir}/runtime-2.json" >"${tmp_dir}/bad-revision.json"
expect_rejection checkpoint-revision "${tmp_dir}/bad-revision.json"

jq '.device.uuid = "GPU-different-device"' \
    "${tmp_dir}/runtime-2.json" >"${tmp_dir}/bad-device.json"
expect_rejection hardware-identity "${tmp_dir}/bad-device.json"

jq '.runtime.kv_storage_provider = "different-provider"' \
    "${tmp_dir}/runtime-2.json" >"${tmp_dir}/bad-provider.json"
expect_rejection provider "${tmp_dir}/bad-provider.json"

jq 'del(.memory)' "${tmp_dir}/runtime-2.json" >"${tmp_dir}/missing-memory.json"
expect_rejection memory "${tmp_dir}/missing-memory.json"

jq 'del(.measurements[0].ttft_ms.p95)' \
    "${tmp_dir}/runtime-2.json" >"${tmp_dir}/incomplete-case.json"
expect_rejection performance-case "${tmp_dir}/incomplete-case.json"

echo "Qwen3.8 MTP paired evidence certifier smoke test passed"
