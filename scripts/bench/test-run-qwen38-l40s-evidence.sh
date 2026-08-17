#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
runner="${repo_root}/scripts/bench/run-qwen38-l40s-evidence.sh"
workload="${repo_root}/benchmarks/manifests/qwen38-l40s-evidence.json"
tmp_dir=$(mktemp -d)
trap 'rm -rf "${tmp_dir}"' EXIT

help=$(${runner} --help)
grep -q 'never estimates or synthesizes performance' <<<"${help}"
grep -q -- '--allow-unsupported' <<<"${help}"

${runner} --workload "${workload}" --mtp-depth 3 \
    --output "${tmp_dir}/dry" --dry-run >/dev/null
jq -e '.schema == "izwi.qwen38-cuda-evidence.v1" and
       .hardware_profile.id == "nvidia-l40s-48gb" and
       .hardware_profile.promotion_scope == "profile_only" and
       .configuration.mtp == {"enabled":true,"draft_tokens":3} and
       .evidence_level == "implemented_unvalidated" and
       .promotion_eligible == false and
       .status == "unsupported" and .reason == "dry_run"' \
    "${tmp_dir}/dry/certificate.json" >/dev/null
[[ $(grep -c '^\[\[benchmarks\]\]$' "${tmp_dir}/dry/imported-manifest.toml") -eq 9 ]]
[[ $(grep -c '^model = "Qwen3.8-27B-FP8"$' "${tmp_dir}/dry/imported-manifest.toml") -eq 9 ]]
grep -q '^warmup = true$' "${tmp_dir}/dry/imported-manifest.toml"
grep -q '^max_tokens = 2048$' "${tmp_dir}/dry/imported-manifest.toml"

jq '.cases[0].iterations = 2' "${workload}" >"${tmp_dir}/invalid.json"
if ${runner} --workload "${tmp_dir}/invalid.json" --mtp-depth 3 \
    --output "${tmp_dir}/invalid" --dry-run >/dev/null 2>&1; then
    echo "invalid Qwen3.8 workload must be rejected" >&2
    exit 1
fi

if IZWI_QWEN38_EVIDENCE_NVIDIA_SMI=/usr/bin/false \
    ${runner} --workload "${workload}" --mtp-depth 3 \
    --output "${tmp_dir}/required" >/dev/null 2>&1; then
    echo "required Qwen3.8 L40S evidence must fail without an NVIDIA device" >&2
    exit 1
fi
jq -e '.status == "failed" and .reason == "nvidia_device_not_observed" and .measurements == null' \
    "${tmp_dir}/required/certificate.json" >/dev/null

IZWI_QWEN38_EVIDENCE_NVIDIA_SMI=/usr/bin/false \
    ${runner} --workload "${workload}" --output "${tmp_dir}/unsupported" \
    --mtp-depth 3 --allow-unsupported >/dev/null
jq -e '.status == "unsupported" and .reason == "nvidia_device_not_observed" and .measurements == null' \
    "${tmp_dir}/unsupported/certificate.json" >/dev/null

echo "Qwen3.8 L40S evidence runner smoke test passed"
