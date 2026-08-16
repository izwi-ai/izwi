#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
workload="${repo_root}/benchmarks/manifests/qwen38-l40s-evidence.json"
server="http://127.0.0.1:8080"
output_dir="${repo_root}/target/qwen38-cuda-evidence"
izwi_bin="${IZWI_QWEN38_EVIDENCE_IZWI:-${repo_root}/target/release/izwi}"
nvidia_smi="${IZWI_QWEN38_EVIDENCE_NVIDIA_SMI:-nvidia-smi}"
cuda_runner="${IZWI_QWEN38_EVIDENCE_CUDA_RUNNER:-${repo_root}/scripts/bench/run-cuda-model-evidence.sh}"
allow_remote=0
allow_unsupported=0
dry_run=0
started_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

usage() {
    cat <<'EOF'
Usage: scripts/bench/run-qwen38-cuda-evidence.sh [options]

Options:
  --workload PATH      Qwen3.8 workload JSON
  --server URL         Izwi server URL (default: http://127.0.0.1:8080)
  --output DIR         Evidence bundle directory
  --izwi-bin PATH      Izwi CLI binary
  --allow-remote       Permit a non-loopback server URL
  --allow-unsupported  Record unsupported when no NVIDIA device is visible
  --dry-run            Validate and materialize the imported TOML manifest only
  -h, --help           Show this help

This runner never estimates or synthesizes performance. A passing certificate
requires the NVIDIA device selected by the server to match the workload's
hardware profile, an exact-SHA CUDA server, and measured TTFT and completion
throughput for every imported workload case. Evidence applies only to that
hardware profile; it does not certify every CUDA-capable GPU.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workload) workload="${2:-}"; shift 2 ;;
        --server) server="${2:-}"; shift 2 ;;
        --output) output_dir="${2:-}"; shift 2 ;;
        --izwi-bin) izwi_bin="${2:-}"; shift 2 ;;
        --allow-remote) allow_remote=1; shift ;;
        --allow-unsupported) allow_unsupported=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! -f "${workload}" ]]; then
    echo "Qwen3.8 workload does not exist: ${workload}" >&2
    exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
    echo "Missing required command: jq" >&2
    exit 1
fi
if [[ ! -x "${cuda_runner}" ]]; then
    echo "CUDA evidence runner is not executable: ${cuda_runner}" >&2
    exit 1
fi
if ! jq -e '
    .schema == "izwi.qwen38-cuda-workload.v1" and
    .model == "Qwen3.8-27B-FP8" and
    .checkpoint.repository == "Qwen/Qwen3.8-27B-FP8" and
    (.checkpoint.revision | test("^[0-9a-f]{40}$")) and
    (.hardware_profile.id | test("^[a-z0-9][a-z0-9._-]*$")) and
    (.hardware_profile.device_name_regex | type == "string" and length > 0) and
    (.hardware_profile.compute_capability_regex | type == "string" and length > 0) and
    (.hardware_profile.minimum_total_memory_bytes | type == "number" and . > 0 and floor == .) and
    (.hardware_profile.driver_version_regex | type == "string" and length > 0) and
    (.hardware_profile.promotion_scope == "profile_only") and
    (.acceptance.performance_thresholds | type == "object") and
    (.acceptance.performance_thresholds.scope == .hardware_profile.id) and
    (.acceptance.performance_thresholds.policy == "declare-before-promotion-runs") and
    ((.acceptance.performance_thresholds.values == null) or
     (.acceptance.performance_thresholds.values | type == "object")) and
    (.cases | type == "array" and length > 0) and
    ([.cases[].name] | length == (unique | length)) and
    ([.cases[] |
        (.name | test("^[a-z0-9-]+$")) and
        (.prompt_words | type == "number" and . >= 1 and floor == .) and
        (.max_tokens | type == "number" and . >= 1 and floor == .) and
        (.iterations | type == "number" and . >= 3 and floor == .) and
        (.concurrent | type == "number" and . >= 1 and floor == .)
    ] | all(. == true)) and
    ([.cases[].prompt_words] | contains([32, 512, 2048, 8192, 32768])) and
    ([.cases[].concurrent] | contains([1, 2, 4, 8])) and
    ([.cases[].max_tokens] | any(. >= 2048))
' "${workload}" >/dev/null; then
    echo "Invalid or incomplete Qwen3.8 CUDA workload/profile: ${workload}" >&2
    exit 2
fi

mkdir -p "${output_dir}"
manifest_path="${output_dir}/imported-manifest.toml"
certificate_path="${output_dir}/certificate.json"
cuda_output="${output_dir}/cuda-evidence"
git_sha=$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || printf unknown)
if command -v sha256sum >/dev/null 2>&1; then
    workload_hash=$(sha256sum "${workload}" | awk '{print $1}')
else
    workload_hash=$(shasum -a 256 "${workload}" | awk '{print $1}')
fi

write_certificate() {
    local status="$1"
    local reason="$2"
    jq -n --arg status "${status}" --arg reason "${reason}" \
        --arg git_sha "${git_sha}" --arg workload "${workload}" \
        --arg workload_sha256 "${workload_hash}" --arg started_at "${started_at}" \
        --arg ended_at "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
        --slurpfile plan "${workload}" \
        '{schema:"izwi.qwen38-cuda-evidence.v1",status:$status,reason:$reason,
          run:{git_sha:$git_sha,workload:$workload,workload_sha256:$workload_sha256,
               checkpoint_revision:$plan[0].checkpoint.revision,
               started_at:$started_at,ended_at:$ended_at},
          hardware_profile:$plan[0].hardware_profile,
          acceptance:$plan[0].acceptance,
          promotion_eligible:false,device:null,measurements:null,
          artifacts:{imported_manifest:"imported-manifest.toml"}}' >"${certificate_path}"
}

model=$(jq -r '.model' "${workload}")
{
    printf '# Generated from %s\n# SHA-256: %s\n\n' "${workload}" "${workload_hash}"
    while IFS=$'\t' read -r name prompt_words max_tokens iterations concurrent; do
        prompt=$(awk -v count="${prompt_words}" 'BEGIN { for (i=1;i<=count;i++) printf "%s%s",(i==1?"":" "),"evidence" }')
        printf '[[benchmarks]]\nname = "%s"\ncommand = "chat"\nmodel = "%s"\n' "${name}" "${model}"
        printf 'iterations = %s\nconcurrent = %s\nwarmup = true\nmax_tokens = %s\n' "${iterations}" "${concurrent}" "${max_tokens}"
        printf 'prompt = "%s"\n\n' "${prompt}"
    done < <(jq -r '.cases[] | [.name,.prompt_words,.max_tokens,.iterations,.concurrent] | @tsv' "${workload}")
} >"${manifest_path}"

cuda_args=(--manifest "${manifest_path}" --server "${server}" --output "${cuda_output}" --izwi-bin "${izwi_bin}")
if [[ "${allow_remote}" -eq 1 ]]; then cuda_args+=(--allow-remote); fi
if [[ "${allow_unsupported}" -eq 1 ]]; then cuda_args+=(--allow-unsupported); fi
if [[ "${dry_run}" -eq 1 ]]; then cuda_args+=(--dry-run); fi

if [[ "${dry_run}" -eq 1 ]]; then
    IZWI_CUDA_EVIDENCE_NVIDIA_SMI="${nvidia_smi}" "${cuda_runner}" "${cuda_args[@]}"
    write_certificate unsupported dry_run
    echo "Qwen3.8 CUDA evidence dry run: ${certificate_path}"
    exit 0
fi
if ! command -v "${nvidia_smi}" >/dev/null 2>&1 || ! "${nvidia_smi}" -L >/dev/null 2>&1; then
    if [[ "${allow_unsupported}" -eq 1 ]]; then
        IZWI_CUDA_EVIDENCE_NVIDIA_SMI="${nvidia_smi}" "${cuda_runner}" "${cuda_args[@]}"
        write_certificate unsupported nvidia_device_not_observed
        exit 0
    fi
    write_certificate failed nvidia_device_not_observed
    echo "Qwen3.8 CUDA evidence requires a device visible to nvidia-smi" >&2
    exit 1
fi

device_name_regex=$(jq -r '.hardware_profile.device_name_regex' "${workload}")
device_names=$("${nvidia_smi}" --query-gpu=name --format=csv,noheader 2>/dev/null || true)
if ! grep -Eq "${device_name_regex}" <<<"${device_names}"; then
    write_certificate failed nvidia_hardware_profile_not_observed
    echo "Qwen3.8 CUDA evidence requires a device matching profile regex: ${device_name_regex}" >&2
    exit 1
fi
"${nvidia_smi}" -q >"${output_dir}/nvidia-smi-q.txt"
"${nvidia_smi}" --query-gpu=timestamp,index,uuid,name,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory,memory.total,memory.used,memory.free --format=csv >"${output_dir}/nvidia-smi.csv"
if command -v nvcc >/dev/null 2>&1; then nvcc --version >"${output_dir}/nvcc-version.txt"; fi
uname -a >"${output_dir}/uname.txt"

IZWI_CUDA_EVIDENCE_NVIDIA_SMI="${nvidia_smi}" "${cuda_runner}" "${cuda_args[@]}"
report_path="${cuda_output}/benchmark/report.json"
cuda_certificate_path="${cuda_output}/certificate.json"
selected_ordinal=$(jq -r '.device.ordinal' "${cuda_certificate_path}")
driver_version=$("${nvidia_smi}" --query-gpu=index,driver_version --format=csv,noheader,nounits 2>/dev/null | \
    awk -F',' -v wanted="${selected_ordinal}" '$1 + 0 == wanted + 0 { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2; exit }')
compute_capability_regex=$(jq -r '.hardware_profile.compute_capability_regex' "${workload}")
minimum_total_memory_bytes=$(jq -r '.hardware_profile.minimum_total_memory_bytes' "${workload}")
driver_version_regex=$(jq -r '.hardware_profile.driver_version_regex' "${workload}")
if ! jq -e --arg device_name_regex "${device_name_regex}" \
    --arg compute_capability_regex "${compute_capability_regex}" \
    --argjson minimum_total_memory_bytes "${minimum_total_memory_bytes}" \
    '.status == "passed" and
     (.device.name | test($device_name_regex)) and
     (.device.compute_capability | test($compute_capability_regex)) and
     (.device.total_memory_bytes >= $minimum_total_memory_bytes)' \
    "${cuda_certificate_path}" >/dev/null ||
   [[ -z "${driver_version}" ]] ||
   ! grep -Eq "${driver_version_regex}" <<<"${driver_version}"; then
    write_certificate failed cuda_hardware_profile_mismatch
    echo "Selected CUDA device does not match the workload hardware profile" >&2
    exit 1
fi
if ! jq -e --arg model "${model}" --slurpfile plan "${workload}" \
    --slurpfile cuda_certificate "${cuda_certificate_path}" '
    .schema_version == 1 and
    ($cuda_certificate[0].status == "passed") and
    (.reports | length) == ($plan[0].cases | length) and
    ([.reports[].name] | sort) == ([$plan[0].cases[].name] | sort) and
    ([.reports[] | .report.config.model == $model and .report.config.warmup == true and
      (.report.samples | length) == .report.config.iterations and
      (.report.summary.ttft_ms.count // 0) > 0 and
      (.report.summary.completion_tps.count // 0) > 0 and
      (.report.summary.server_generation_ms.count // 0) > 0 and
      ([.report.samples[] | (.prompt_tokens // 0) > 0 and (.completion_tokens // 0) > 0 and
        (.ttft_ms // 0) > 0 and (.completion_tps // 0) > 0 and
        (.server_generation_ms // 0) > 0] | all(. == true))] | all(. == true))
' "${report_path}" >/dev/null; then
    write_certificate failed missing_qwen38_performance_measurements
    echo "Qwen3.8 report is missing required measured TTFT or completion throughput" >&2
    exit 1
fi

jq -n --arg git_sha "${git_sha}" --arg workload "${workload}" \
    --arg workload_sha256 "${workload_hash}" --arg started_at "${started_at}" \
    --arg ended_at "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
    --arg driver_version "${driver_version}" \
    --slurpfile cuda_certificate "${cuda_certificate_path}" \
    --slurpfile report "${report_path}" --slurpfile plan "${workload}" \
    '{schema:"izwi.qwen38-cuda-evidence.v1",status:"passed",reason:"measured_qwen38_cuda_profile_evidence_passed",
      run:{git_sha:$git_sha,workload:$workload,workload_sha256:$workload_sha256,
           checkpoint_revision:$plan[0].checkpoint.revision,started_at:$started_at,ended_at:$ended_at},
      hardware_profile:$plan[0].hardware_profile,
      acceptance:$plan[0].acceptance,
      promotion_eligible:false,
      device:($cuda_certificate[0].device + {driver_version:$driver_version}),
      measurements:[$report[0].reports[] as $result |
        ($plan[0].cases[]|select(.name==$result.name)) as $case |
        {name:$result.name,target_prompt_words:$case.prompt_words,
         observed_prompt_tokens_avg:$result.report.summary.prompt_tokens_avg,
         completion_tokens_avg:$result.report.summary.completion_tokens_avg,
         concurrent:$result.report.config.concurrent,ttft_ms:$result.report.summary.ttft_ms,
         end_to_end_completion_tps:$result.report.summary.completion_tps,
         server_generation_ms:$result.report.summary.server_generation_ms,
         decode_samples:[$result.report.samples[] |
           {index:.index,completion_tokens:.completion_tokens,generation_ms:.server_generation_ms,
            tokens_per_second:(.completion_tokens * 1000 / .server_generation_ms)}],
         end_to_end_ms:$result.report.summary.end_to_end_ms}],
      artifacts:{imported_manifest:"imported-manifest.toml",cuda_certificate:"cuda-evidence/certificate.json",
        report:"cuda-evidence/benchmark/report.json",nvidia_smi_query:"nvidia-smi.csv",
        nvidia_smi_detail:"nvidia-smi-q.txt",uname:"uname.txt"}}' >"${certificate_path}"

echo "Qwen3.8 CUDA profile evidence passed: ${certificate_path}"
