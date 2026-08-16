#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
runner="${repo_root}/scripts/bench/run-cuda-model-evidence.sh"
manifest="${repo_root}/benchmarks/manifests/vibevoice-cuda.toml"
tmp_dir=$(mktemp -d)
trap 'rm -rf "${tmp_dir}"' EXIT

help=$(${runner} --help)
grep -q -- '--allow-unsupported' <<<"${help}"
grep -q -- '--require-optimized-kernel-evidence' <<<"${help}"
grep -q -- '--require-continuous-batch-evidence' <<<"${help}"
grep -q 'actual_device_kind=cuda' <<<"${help}"

dry_output=$(${runner} --manifest "${manifest}" --output "${tmp_dir}/dry" --dry-run \
    --require-continuous-batch-evidence)
grep -q 'IZWI_BENCH_QUALITY_MODE=strict' <<<"${dry_output}"
grep -q -- '--artifact-dir' <<<"${dry_output}"
jq -e '.schema == "izwi.cuda-model-evidence.v1" and .status == "unsupported" and
       .reason == "dry_run" and .requirements.continuous_batch == true and
       (.run.worktree_clean | type == "boolean")' \
    "${tmp_dir}/dry/certificate.json" >/dev/null
[[ ! -e "${tmp_dir}/dry/benchmark" ]]

if IZWI_CUDA_EVIDENCE_NVIDIA_SMI=/usr/bin/false \
    ${runner} --manifest "${manifest}" --output "${tmp_dir}/required" >/dev/null 2>&1; then
    echo "required CUDA evidence must fail without an NVIDIA device" >&2
    exit 1
fi
jq -e '.status == "failed" and .reason == "CUDA model certification requires a device visible to nvidia-smi"' \
    "${tmp_dir}/required/certificate.json" >/dev/null

IZWI_CUDA_EVIDENCE_NVIDIA_SMI=/usr/bin/false \
    ${runner} --manifest "${manifest}" --output "${tmp_dir}/optional" \
    --allow-unsupported >/dev/null
jq -e '.status == "unsupported" and .reason == "nvidia_device_not_observed"' \
    "${tmp_dir}/optional/certificate.json" >/dev/null

echo "CUDA model evidence runner smoke test passed"
