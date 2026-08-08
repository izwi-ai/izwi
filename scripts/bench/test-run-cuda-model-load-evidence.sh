#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
runner="${repo_root}/scripts/bench/run-cuda-model-load-evidence.sh"
manifest="${repo_root}/benchmarks/manifests/cuda-family-load.txt"
tmp_dir=$(mktemp -d)
trap 'rm -rf "${tmp_dir}"' EXIT

help=$(${runner} --help)
grep -q -- '--dry-run' <<<"${help}"
grep -q 'actual_device_kind=cuda' <<<"${help}"

${runner} --manifest "${manifest}" --output "${tmp_dir}/dry" --dry-run >/dev/null
jq -e '
    .schema == "izwi.cuda-model-load-evidence.v1" and
    .status == "unsupported" and
    .reason == "dry_run" and
    (.models | length) == 20 and
    ([.models[].model] | length == (unique | length)) and
    ([.models[].model] | index("Qwen3-ForcedAligner-0.6B") != null) and
    ([.models[].model] | index("diar_streaming_sortformer_4spk-v2.1") != null) and
    ([.models[].model] | index("Qwen3-TTS-Tokenizer-12Hz") != null)
' "${tmp_dir}/dry/certificate.json" >/dev/null

if IZWI_CUDA_EVIDENCE_NVIDIA_SMI=/usr/bin/false \
    ${runner} --manifest "${manifest}" --output "${tmp_dir}/required" >/dev/null 2>&1; then
    echo "required CUDA model load evidence must fail without an NVIDIA device" >&2
    exit 1
fi

echo "CUDA model load evidence runner smoke test passed"
