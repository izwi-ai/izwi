#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
runner="${repo_root}/scripts/bench/run-qwen38-cuda-evidence.sh"
base_workload="${repo_root}/benchmarks/manifests/qwen38-l40s-evidence.json"
tmp_dir=$(mktemp -d)
trap 'rm -rf "${tmp_dir}"' EXIT

help=$(${runner} --help)
grep -q 'hardware profile' <<<"${help}"
grep -q 'does not certify every CUDA-capable GPU' <<<"${help}"

# A non-L40S profile proves that the reusable runner does not bake the initial
# device name or compute capability into its workflow validation.
jq '
  .hardware_profile.id = "portable-sm80-sm86" |
  .hardware_profile.device_name_regex = "A(100|10)|RTX A[0-9]+" |
  .hardware_profile.compute_capability_regex = "^8[.][06]$" |
  .hardware_profile.minimum_total_memory_bytes = 32000000000 |
  .acceptance.performance_thresholds.scope = "portable-sm80-sm86"
' "${base_workload}" >"${tmp_dir}/portable-profile.json"

${runner} --workload "${tmp_dir}/portable-profile.json" \
    --mtp-depth 3 --output "${tmp_dir}/dry" --dry-run >/dev/null
jq -e '.schema == "izwi.qwen38-cuda-evidence.v1" and
       .hardware_profile.id == "portable-sm80-sm86" and
       .hardware_profile.compute_capability_regex == "^8[.][06]$" and
       .hardware_profile.minimum_total_memory_bytes == 32000000000 and
       .acceptance.performance_thresholds.scope == "portable-sm80-sm86" and
       .configuration.mtp == {"enabled":true,"draft_tokens":3} and
       .evidence_level == "implemented_unvalidated" and
       .promotion_eligible == false and .status == "unsupported"' \
    "${tmp_dir}/dry/certificate.json" >/dev/null

jq 'del(.hardware_profile.compute_capability_regex)' \
    "${tmp_dir}/portable-profile.json" >"${tmp_dir}/invalid-profile.json"
if ${runner} --workload "${tmp_dir}/invalid-profile.json" \
    --mtp-depth 3 --output "${tmp_dir}/invalid" --dry-run >/dev/null 2>&1; then
    echo "Qwen3.8 CUDA evidence must reject incomplete hardware profiles" >&2
    exit 1
fi

if IZWI_QWEN38_EVIDENCE_NVIDIA_SMI=/usr/bin/false \
    ${runner} --workload "${tmp_dir}/portable-profile.json" \
    --mtp-depth 3 --output "${tmp_dir}/required" >/dev/null 2>&1; then
    echo "required Qwen3.8 CUDA evidence must fail without an NVIDIA device" >&2
    exit 1
fi
jq -e '.status == "failed" and .reason == "nvidia_device_not_observed" and
       .hardware_profile.id == "portable-sm80-sm86" and .measurements == null' \
    "${tmp_dir}/required/certificate.json" >/dev/null

echo "Qwen3.8 portable CUDA evidence runner smoke test passed"
