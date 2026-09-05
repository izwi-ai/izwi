#!/usr/bin/env bash

set -euo pipefail

baseline=""
depth_1=""
depth_2=""
depth_3=""
output_dir="target/qwen38-mtp-evidence"
dry_run=0

usage() {
    cat <<'EOF'
Usage: scripts/bench/certify-qwen38-mtp-evidence.sh [options]

Required:
  --baseline PATH  MTP-disabled Qwen3.8 certificate (or bundle directory)
  --depth-1 PATH   MTP depth-1 Qwen3.8 certificate (or bundle directory)
  --depth-2 PATH   MTP depth-2 Qwen3.8 certificate (or bundle directory)
  --depth-3 PATH   MTP depth-3 Qwen3.8 certificate (or bundle directory)

Options:
  --output DIR     Paired certificate directory (default: target/qwen38-mtp-evidence)
  --dry-run        Validate four dry-run manifests; emit implemented_unvalidated
  -h, --help       Show this help

Real evidence fails closed unless all four cells are runtime_validated and
share one exact checkpoint revision, Git SHA, workload, hardware profile,
physical device identity, CUDA provider, compute dtype, and KV provider. Every
cell must contain sampled device memory and the same complete performance cases.
Declared thresholds may advance the pair from runtime_validated to
performance_certified; a dry run can only be implemented_unvalidated.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline) baseline="${2:-}"; shift 2 ;;
        --depth-1) depth_1="${2:-}"; shift 2 ;;
        --depth-2) depth_2="${2:-}"; shift 2 ;;
        --depth-3) depth_3="${2:-}"; shift 2 ;;
        --output) output_dir="${2:-}"; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if ! command -v jq >/dev/null 2>&1; then
    echo "Missing required command: jq" >&2
    exit 1
fi

certificate_path() {
    if [[ -d "$1" ]]; then
        printf '%s/certificate.json\n' "${1%/}"
    else
        printf '%s\n' "$1"
    fi
}

for argument in baseline depth_1 depth_2 depth_3; do
    value="${!argument}"
    if [[ -z "${value}" ]]; then
        echo "--${argument//_/-} is required" >&2
        exit 2
    fi
done

baseline=$(certificate_path "${baseline}")
depth_1=$(certificate_path "${depth_1}")
depth_2=$(certificate_path "${depth_2}")
depth_3=$(certificate_path "${depth_3}")
mkdir -p "${output_dir}"
output_certificate="${output_dir}/certificate.json"

write_failure() {
    local reason="$1"
    jq -n --arg reason "${reason}" \
        --arg baseline "${baseline}" --arg depth_1 "${depth_1}" \
        --arg depth_2 "${depth_2}" --arg depth_3 "${depth_3}" \
        '{schema:"izwi.qwen38-mtp-paired-evidence.v1",status:"failed",reason:$reason,
          evidence_level:"implemented_unvalidated",promotion_eligible:false,
          sources:{baseline:$baseline,depth_1:$depth_1,depth_2:$depth_2,depth_3:$depth_3}}' \
        >"${output_certificate}"
}

for certificate in "${baseline}" "${depth_1}" "${depth_2}" "${depth_3}"; do
    if [[ ! -s "${certificate}" ]] || ! jq -e . "${certificate}" >/dev/null 2>&1; then
        write_failure invalid_or_missing_source_certificate
        echo "Missing or invalid Qwen3.8 source certificate: ${certificate}" >&2
        exit 1
    fi
done

common_filter=' 
  def sha40: type == "string" and test("^[0-9a-f]{40}$");
  def sha256: type == "string" and test("^[0-9a-f]{64}$");
  def mtp($enabled; $depth):
    . as $cell |
    ($cell.configuration.mtp.enabled == $enabled) and
    ($cell.configuration.mtp.draft_tokens == $depth);
  [$baseline[0], $depth_1[0], $depth_2[0], $depth_3[0]] as $cells |
  ([$cells[].schema] | all(. == "izwi.qwen38-cuda-evidence.v1")) and
  ($cells[0] | mtp(false; null)) and
  ($cells[1] | mtp(true; 1)) and
  ($cells[2] | mtp(true; 2)) and
  ($cells[3] | mtp(true; 3)) and
  ([$cells[].run.git_sha] | all(sha40) and (unique | length == 1)) and
  ([$cells[].run.checkpoint_revision] | all(sha40) and (unique | length == 1)) and
  ([$cells[].run.workload_sha256] | all(sha256) and (unique | length == 1)) and
  ([$cells[].hardware_profile] | unique | length == 1) and
  ([$cells[].acceptance] | unique | length == 1) and
  ($cells[0].hardware_profile.promotion_scope == "profile_only")
'

if ! jq -n -e --slurpfile baseline "${baseline}" --slurpfile depth_1 "${depth_1}" \
    --slurpfile depth_2 "${depth_2}" --slurpfile depth_3 "${depth_3}" \
    "${common_filter}" >/dev/null; then
    write_failure unpaired_or_invalid_source_contract
    echo "Qwen3.8 MTP sources do not share the required manifest and exact-SHA contract" >&2
    exit 1
fi

if [[ "${dry_run}" -eq 1 ]]; then
    if ! jq -n -e --slurpfile baseline "${baseline}" --slurpfile depth_1 "${depth_1}" \
        --slurpfile depth_2 "${depth_2}" --slurpfile depth_3 "${depth_3}" '
        [$baseline[0], $depth_1[0], $depth_2[0], $depth_3[0]] |
        all(.status == "unsupported" and .reason == "dry_run" and
            .evidence_level == "implemented_unvalidated" and
            .promotion_eligible == false and .device == null and .runtime == null and
            .memory == null and .measurements == null)
    ' >/dev/null; then
        write_failure dry_run_requires_four_unvalidated_manifests
        echo "--dry-run accepts only four Qwen3.8 dry-run certificates" >&2
        exit 1
    fi
    jq -n --slurpfile baseline "${baseline}" --slurpfile depth_1 "${depth_1}" \
        --slurpfile depth_2 "${depth_2}" --slurpfile depth_3 "${depth_3}" \
        --arg baseline_path "${baseline}" --arg depth_1_path "${depth_1}" \
        --arg depth_2_path "${depth_2}" --arg depth_3_path "${depth_3}" '
        {schema:"izwi.qwen38-mtp-paired-evidence.v1",status:"unsupported",
         reason:"paired_manifests_validated_without_runtime",
         evidence_level:"implemented_unvalidated",promotion_eligible:false,
         binding:{git_sha:$baseline[0].run.git_sha,
           checkpoint_revision:$baseline[0].run.checkpoint_revision,
           workload_sha256:$baseline[0].run.workload_sha256,
           hardware_profile:$baseline[0].hardware_profile},
         cells:[
           {role:"baseline",mtp:$baseline[0].configuration.mtp,certificate:$baseline_path},
           {role:"candidate",mtp:$depth_1[0].configuration.mtp,certificate:$depth_1_path},
           {role:"candidate",mtp:$depth_2[0].configuration.mtp,certificate:$depth_2_path},
           {role:"candidate",mtp:$depth_3[0].configuration.mtp,certificate:$depth_3_path}
         ]}
    ' >"${output_certificate}"
    echo "Qwen3.8 MTP paired manifests validated without runtime evidence: ${output_certificate}"
    exit 0
fi

if ! jq -n -e --slurpfile baseline "${baseline}" --slurpfile depth_1 "${depth_1}" \
    --slurpfile depth_2 "${depth_2}" --slurpfile depth_3 "${depth_3}" '
    def positive_number: type == "number" and . > 0;
    def integer_number: type == "number" and . >= 1 and floor == .;
    def complete_stats:
      (.count | integer_number) and (.avg | positive_number) and
      (.p50 | positive_number) and (.p95 | positive_number);
    def complete_case:
      . as $case |
      ($case.name | type == "string" and length > 0) and
      ($case.target_prompt_words | integer_number) and
      ($case.observed_prompt_tokens_avg | positive_number) and
      ($case.completion_tokens_avg | positive_number) and
      ($case.concurrent | integer_number) and
      ($case.ttft_ms | complete_stats) and
      ($case.end_to_end_completion_tps | complete_stats) and
      ($case.server_generation_ms | complete_stats) and
      ($case.end_to_end_ms | complete_stats) and
      ($case.server_request_samples | type == "array" and length > 0 and
        length == $case.end_to_end_completion_tps.count and
        all((.completion_tokens | integer_number) and
            (.generation_ms | positive_number) and
            (.tokens_per_second | positive_number) and
            ((.tokens_per_second - .completion_tokens * 1000 / .generation_ms) | fabs < 0.000001) and
            (.finish_reason == "stop" or .finish_reason == "length"))) and
      ($case.decode_wall_samples | type == "array" and
        length == ($case.server_request_samples | length) and
        all((.committed_tokens | integer_number) and (.wall_ms | positive_number) and
            (.tokens_per_second | positive_number) and
            ((.tokens_per_second - .committed_tokens * 1000 / .wall_ms) | fabs < 0.000001) and
            (.physical_service_ms | type == "number" and . >= 0))) and
      ([$case.decode_wall_samples[] as $decode |
        [$case.server_request_samples[] | select(.index == $decode.index)] |
        length == 1 and .[0].completion_tokens >= $decode.committed_tokens and
        .[0].generation_ms >= $decode.wall_ms] | all) and
      ([$case.server_request_samples[].index] | length == (unique | length)) and
      ([$case.decode_wall_samples[].index] | length == (unique | length)) and
      ($case.sampling == {"temperature":0,"seed":0,"reasoning_policy":"model_default"}) and
      ($case.prompt | type == "string" and length > 0) and
      ($case.max_tokens | integer_number) and
      ($case.prefix_cache == "cold" or $case.prefix_cache == "warm") and
      (if $case.prefix_cache == "cold" then $case.prefix_hits_delta == 0 else true end);
    def runtime_cell($enabled; $depth):
      .status == "passed" and .evidence_level == "runtime_validated" and
      .promotion_eligible == false and
      .runtime.hardware_provider == "nvidia-cuda" and
      .runtime.actual_device_kind == "cuda" and
      (.runtime.actual_compute_dtype | type == "string" and length > 0) and
      (.runtime.kv_storage_provider | type == "string" and length > 0) and
      .runtime.mtp.enabled == $enabled and .runtime.mtp.draft_tokens == $depth and
      .runtime.mtp.source == "loaded_model_diagnostics" and
      .runtime.mtp.runtime_validated == true and
      (.device.uuid | type == "string" and length > 0) and
      (.device.name | type == "string" and length > 0) and
      (.device.compute_capability | type == "string" and length > 0) and
      (.device.total_memory_bytes | positive_number) and
      (.device.driver_version | type == "string" and length > 0) and
      .memory.source == "nvidia-smi-memory.used" and
      (.memory.sample_count | integer_number) and
      (.memory.peak_device_memory_used_bytes | positive_number) and
      (.memory.minimum_device_memory_free_bytes | type == "number" and . >= 0) and
      (.measurements | type == "array" and length > 0 and
        ([.[].name] | length == (unique | length)) and all(complete_case));
    def mtp_case($enabled; $depth):
      if $enabled then
        (if .concurrent == 1 then
           (.mtp.rounds | integer_number) and (.mtp.drafted_tokens | integer_number) and
           (.mtp.target_verified_tokens | integer_number)
         else [.mtp.rounds,.mtp.drafted_tokens,.mtp.target_verified_tokens] | all(type == "number" and . >= 0 and floor == .) end) and
        .mtp.drafted_tokens <= (.mtp.rounds * $depth)
      else
        .mtp.rounds == 0 and .mtp.drafted_tokens == 0
      end;
    [$baseline[0], $depth_1[0], $depth_2[0], $depth_3[0]] as $cells |
    ($cells[0] | runtime_cell(false; null)) and
    ($cells[1] | runtime_cell(true; 1)) and
    ($cells[2] | runtime_cell(true; 2)) and
    ($cells[3] | runtime_cell(true; 3)) and
    ([$cells[].device | {uuid,name,compute_capability,total_memory_bytes,driver_version}] |
      unique | length == 1) and
    ([$cells[].runtime | {hardware_provider,actual_device_kind,actual_compute_dtype,kv_storage_provider}] |
      unique | length == 1) and
    ([$cells[] | [.measurements[].name] | sort] | unique | length == 1) and
    ([$cells[0].measurements[] | mtp_case(false; null)] | all) and
    ([$cells[1].measurements[] | mtp_case(true; 1)] | all) and
    ([$cells[2].measurements[] | mtp_case(true; 2)] | all) and
    ([$cells[3].measurements[] | mtp_case(true; 3)] | all) and
    ([$cells[1], $cells[2], $cells[3]] | all(. as $candidate |
      [$candidate.measurements[] as $case |
       $cells[0].measurements[] |
       select(.name == $case.name) |
       .target_prompt_words == $case.target_prompt_words and .concurrent == $case.concurrent and
       .prompt == $case.prompt and .system == $case.system and .max_tokens == $case.max_tokens and
       .prefix_cache == $case.prefix_cache and .sampling == $case.sampling and
       (.server_request_samples | length) == ($case.server_request_samples | length)]
      | all))
  ' >/dev/null; then
    write_failure incomplete_or_unpaired_runtime_evidence
    echo "Qwen3.8 MTP evidence is missing an observed policy/provider, measured memory, or complete paired cases" >&2
    exit 1
fi

thresholds=$(jq -c '.acceptance.performance_thresholds.values' "${baseline}")
if [[ "${thresholds}" != "null" ]] && ! jq -e '
    type == "object" and (.mtp | type == "object") and
    (.mtp.minimum_completion_tps_p50_speedup_ratio | type == "number" and . > 1) and
    (.mtp.maximum_ttft_p95_regression_ratio | type == "number" and . >= 1) and
    (.mtp.maximum_peak_device_memory_ratio | type == "number" and . >= 1) and
    (.single_sequence.minimum_user_tps_p50 | type == "number" and . > 0) and
    (.single_sequence.minimum_decode_tps_p50 | type == "number" and . > 0) and
    (.single_sequence.minimum_runs | type == "number" and . >= 10 and floor == .) and
    (.single_sequence.user_cases | type == "array" and length > 0 and all(type == "string")) and
    (.single_sequence.sustained_cases | type == "object" and length > 0 and all(.[]; type == "number" and . > 0 and floor == .))
  ' <<<"${thresholds}" >/dev/null; then
    write_failure invalid_declared_mtp_performance_thresholds
    echo "Declared MTP thresholds are incomplete or cannot certify an improvement" >&2
    exit 1
fi

jq -n --slurpfile baseline "${baseline}" --slurpfile depth_1 "${depth_1}" \
    --slurpfile depth_2 "${depth_2}" --slurpfile depth_3 "${depth_3}" \
    --arg baseline_path "${baseline}" --arg depth_1_path "${depth_1}" \
    --arg depth_2_path "${depth_2}" --arg depth_3_path "${depth_3}" '
    def quantile($p): sort | .[(($p * (length - 1)) | floor)];
    def single_sequence($candidate; $thresholds):
      $thresholds.single_sequence as $gate |
      if $gate == null then false else
        ([$gate.user_cases[] as $name |
          [$candidate.measurements[] | select(.name == $name)] |
          length == 1 and (.[0] | .concurrent == 1 and .prefix_cache == "cold" and .prefix_hits_delta == 0 and
            (.server_request_samples | length) >= $gate.minimum_runs and
            ([.server_request_samples[].tokens_per_second] | quantile(0.5)) >= $gate.minimum_user_tps_p50)] | all) and
        ([$gate.sustained_cases | to_entries[] as $case |
          [$candidate.measurements[] | select(.name == $case.key)] |
          length == 1 and (.[0] | .concurrent == 1 and .prefix_cache == "cold" and .prefix_hits_delta == 0 and
            (.decode_wall_samples | length) >= $gate.minimum_runs and
            ([.decode_wall_samples[].committed_tokens] | all(. >= $case.value)) and
            ([.decode_wall_samples[].tokens_per_second] | quantile(0.5)) >= $gate.minimum_decode_tps_p50)] | all)
      end;
    def comparison($candidate; $depth; $baseline; $thresholds):
      [$candidate.measurements[] as $case |
       ($baseline.measurements[] | select(.name == $case.name)) as $base |
       {name:$case.name,
        user_tps_p10:([$case.server_request_samples[].tokens_per_second] | quantile(0.1)),
        user_tps_p50:([$case.server_request_samples[].tokens_per_second] | quantile(0.5)),
        decode_tps_p10:([$case.decode_wall_samples[].tokens_per_second] | quantile(0.1)),
        decode_tps_p50:([$case.decode_wall_samples[].tokens_per_second] | quantile(0.5)),
        completion_tps_p50_ratio:($case.end_to_end_completion_tps.p50 / $base.end_to_end_completion_tps.p50),
        ttft_p95_ratio:($case.ttft_ms.p95 / $base.ttft_ms.p95)} |
       . + {meets_declared_thresholds:
         (if $thresholds == null then null else
            .completion_tps_p50_ratio >= $thresholds.mtp.minimum_completion_tps_p50_speedup_ratio and
            .ttft_p95_ratio <= $thresholds.mtp.maximum_ttft_p95_regression_ratio
          end)}] as $cases |
      ($candidate.memory.peak_device_memory_used_bytes /
       $baseline.memory.peak_device_memory_used_bytes) as $memory_ratio |
      {depth:$depth,peak_device_memory_ratio:$memory_ratio,cases:$cases,
       meets_single_sequence_thresholds:single_sequence($candidate; $thresholds),
       mean_completion_tps_p50_ratio:([$cases[].completion_tps_p50_ratio] | add / length),
       meets_declared_thresholds:
         (if $thresholds == null then null else
            single_sequence($candidate; $thresholds) and
            $memory_ratio <= $thresholds.mtp.maximum_peak_device_memory_ratio and
            ([$cases[].meets_declared_thresholds] | all)
          end)};
    $baseline[0] as $base |
    $base.acceptance.performance_thresholds.values as $thresholds |
    [comparison($depth_1[0];1;$base;$thresholds),
     comparison($depth_2[0];2;$base;$thresholds),
     comparison($depth_3[0];3;$base;$thresholds)] as $comparisons |
    ([$comparisons[] | select(.meets_declared_thresholds == true)] |
     sort_by(.mean_completion_tps_p50_ratio)) as $certified |
    ($thresholds != null and ($certified | length) > 0) as $performance_certified |
    {schema:"izwi.qwen38-mtp-paired-evidence.v1",status:"passed",
     reason:(if $performance_certified then "paired_mtp_performance_thresholds_passed"
             elif $thresholds == null then "paired_mtp_runtime_validated_thresholds_undeclared"
             else "paired_mtp_runtime_validated_no_candidate_met_thresholds" end),
     evidence_level:(if $performance_certified then "performance_certified" else "runtime_validated" end),
     promotion_eligible:$performance_certified,
     binding:{git_sha:$base.run.git_sha,checkpoint_revision:$base.run.checkpoint_revision,
       workload_sha256:$base.run.workload_sha256,hardware_profile:$base.hardware_profile,
       device:($base.device | {uuid,name,compute_capability,total_memory_bytes,driver_version}),
       runtime_provider:($base.runtime | {hardware_provider,actual_device_kind,actual_compute_dtype,kv_storage_provider})},
     declared_thresholds:$thresholds,
     certified_depths:[$certified[].depth],
     recommended_depth:(if $performance_certified then $certified[-1].depth else null end),
     comparisons:$comparisons,
     cells:[
       {role:"baseline",mtp:$base.configuration.mtp,memory:$base.memory,certificate:$baseline_path},
       {role:"candidate",mtp:$depth_1[0].configuration.mtp,memory:$depth_1[0].memory,certificate:$depth_1_path},
       {role:"candidate",mtp:$depth_2[0].configuration.mtp,memory:$depth_2[0].memory,certificate:$depth_2_path},
       {role:"candidate",mtp:$depth_3[0].configuration.mtp,memory:$depth_3[0].memory,certificate:$depth_3_path}
     ]}
  ' >"${output_certificate}"

echo "Qwen3.8 MTP paired evidence validated: ${output_certificate}"
