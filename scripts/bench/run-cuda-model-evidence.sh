#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
manifest=""
server="http://127.0.0.1:8080"
output_dir="${repo_root}/target/cuda-model-evidence"
izwi_bin="${IZWI_CUDA_EVIDENCE_IZWI:-${repo_root}/target/release/izwi}"
nvidia_smi="${IZWI_CUDA_EVIDENCE_NVIDIA_SMI:-nvidia-smi}"
allow_remote=0
allow_unsupported=0
dry_run=0
require_optimized_kernel_evidence=0
require_continuous_batch_evidence=0
require_resumable_prefill_evidence=0
certificate_written=0
started_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

usage() {
    cat <<'EOF'
Usage: scripts/bench/run-cuda-model-evidence.sh --manifest PATH [options]

Options:
  --server URL          Izwi server URL (default: http://127.0.0.1:8080)
  --output DIR          Evidence bundle directory
  --izwi-bin PATH       Izwi CLI binary
  --allow-remote        Permit a non-loopback server URL
  --allow-unsupported   Missing NVIDIA hardware emits unsupported instead of failing
  --require-optimized-kernel-evidence
                        Require every case to observe a fused/paged/RoPE CUDA path
  --require-continuous-batch-evidence
                        Require every case to prove run-local multi-row continuous batching
  --require-resumable-prefill-evidence
                        Require every case to prove a prompt ran as multiple resumable spans
  --dry-run             Print the benchmark command without probing hardware/server
  -h, --help            Show this help

Real certification is fail-closed by default. It requires observed CUDA health,
strict benchmark quality, per-case telemetry, and actual_device_kind=cuda.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest)
            manifest="${2:-}"
            shift 2
            ;;
        --server)
            server="${2:-}"
            shift 2
            ;;
        --output)
            output_dir="${2:-}"
            shift 2
            ;;
        --izwi-bin)
            izwi_bin="${2:-}"
            shift 2
            ;;
        --allow-remote)
            allow_remote=1
            shift
            ;;
        --allow-unsupported)
            allow_unsupported=1
            shift
            ;;
        --require-optimized-kernel-evidence)
            require_optimized_kernel_evidence=1
            shift
            ;;
        --require-continuous-batch-evidence)
            require_continuous_batch_evidence=1
            shift
            ;;
        --require-resumable-prefill-evidence)
            require_resumable_prefill_evidence=1
            shift
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${manifest}" ]]; then
    echo "--manifest is required" >&2
    exit 2
fi
if [[ ! -f "${manifest}" ]]; then
    echo "Benchmark manifest does not exist: ${manifest}" >&2
    exit 2
fi

mkdir -p "${output_dir}"
certificate_path="${output_dir}/certificate.json"
runner_log="${output_dir}/runner.log"
health_path="${output_dir}/health.json"
benchmark_dir="${output_dir}/benchmark"
: >"${runner_log}"
git_sha=$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || printf 'unknown')
if [[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=normal 2>/dev/null)" ]]; then
    worktree_clean=true
else
    worktree_clean=false
fi
if command -v sha256sum >/dev/null 2>&1; then
    manifest_hash=$(sha256sum "${manifest}" | awk '{print $1}')
elif command -v shasum >/dev/null 2>&1; then
    manifest_hash=$(shasum -a 256 "${manifest}" | awk '{print $1}')
else
    echo "Missing required SHA-256 command (sha256sum or shasum)" >&2
    exit 1
fi

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        return 1
    fi
}

write_status_certificate() {
    local status="$1"
    local reason="$2"
    local ended_at
    ended_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    jq -n \
        --arg status "${status}" \
        --arg reason "${reason}" \
        --arg git_sha "${git_sha}" \
        --arg manifest "${manifest}" \
        --arg manifest_sha256 "${manifest_hash}" \
        --arg started_at "${started_at}" \
        --arg ended_at "${ended_at}" \
        --argjson worktree_clean "${worktree_clean}" \
        --argjson require_optimized "${require_optimized_kernel_evidence}" \
        --argjson require_continuous "${require_continuous_batch_evidence}" \
        --argjson require_resumable_prefill "${require_resumable_prefill_evidence}" \
        '{
            schema: "izwi.cuda-model-evidence.v1",
            status: $status,
            reason: $reason,
            run: {
                git_sha: $git_sha,
                manifest: $manifest,
                manifest_sha256: $manifest_sha256,
                worktree_clean: $worktree_clean,
                started_at: $started_at,
                ended_at: $ended_at
            },
            requirements: {
                optimized_kernel: ($require_optimized == 1),
                continuous_batch: ($require_continuous == 1),
                resumable_prefill: ($require_resumable_prefill == 1)
            },
            device: null,
            cases: [],
            artifacts: {runner_log: "runner.log"}
        }' >"${certificate_path}"
    certificate_written=1
}

fail() {
    local reason="$1"
    echo "${reason}" >&2
    write_status_certificate "failed" "${reason}"
    exit 1
}

on_error() {
    local code=$?
    trap - ERR
    if [[ "${certificate_written}" -eq 0 ]] && command -v jq >/dev/null 2>&1; then
        write_status_certificate "failed" "unexpected runner failure"
    fi
    exit "${code}"
}
trap on_error ERR

require_command jq

if [[ "${dry_run}" -eq 1 ]]; then
    printf 'DRY-RUN IZWI_BENCH_QUALITY_MODE=strict %q --server %q --output-format json bench run %q --artifact-dir %q\n' \
        "${izwi_bin}" "${server}" "${manifest}" "${benchmark_dir}"
    write_status_certificate "unsupported" "dry_run"
    exit 0
fi

if [[ "${allow_remote}" -ne 1 ]]; then
    case "${server}" in
        http://127.0.0.1:*|http://localhost:*|http://\[::1\]:*) ;;
        *) fail "CUDA evidence requires a loopback server unless --allow-remote is set" ;;
    esac
fi

if ! command -v "${nvidia_smi}" >/dev/null 2>&1 || ! "${nvidia_smi}" -L >/dev/null 2>&1; then
    if [[ "${allow_unsupported}" -eq 1 ]]; then
        write_status_certificate "unsupported" "nvidia_device_not_observed"
        exit 0
    fi
    fail "CUDA model certification requires a device visible to nvidia-smi"
fi
if [[ "${worktree_clean}" != true ]]; then
    fail "CUDA model certification requires a clean exact-SHA worktree"
fi

require_command curl
if [[ ! -x "${izwi_bin}" ]]; then
    fail "Izwi CLI binary is not executable: ${izwi_bin}"
fi

if ! curl -fsS "${server%/}/v1/health" -o "${health_path}"; then
    fail "Unable to read Izwi health from ${server}"
fi
if ! jq -e '
    (.runtime.build_git_sha | type == "string" and length > 0) and
    .runtime.requested_backend == "cuda" and
    .runtime.requested_backend_available == true and
    .runtime.selected_backend == "cuda" and
    .runtime.compiled_backends.cuda == true and
    .runtime.cuda_runtime.driver_available == true and
    .runtime.cuda_runtime.device_usable == true
' "${health_path}" >/dev/null; then
    fail "Izwi health does not prove a usable selected CUDA runtime"
fi
if ! jq -e --arg git_sha "${git_sha}" '.runtime.build_git_sha == $git_sha' \
    "${health_path}" >/dev/null; then
    fail "Izwi health is not bound to the checked-out git SHA"
fi

if ! IZWI_BENCH_QUALITY_MODE=strict \
    "${izwi_bin}" \
        --server "${server}" \
        --output-format json \
        bench run "${manifest}" \
        --artifact-dir "${benchmark_dir}" 2>&1 | tee "${runner_log}"; then
    fail "Strict CUDA model benchmark failed"
fi

for artifact in report.json metadata.json observability.json manifest.toml; do
    if [[ ! -s "${benchmark_dir}/${artifact}" ]]; then
        fail "CUDA benchmark artifact is missing: benchmark/${artifact}"
    fi
done

report_path="${benchmark_dir}/report.json"
metadata_path="${benchmark_dir}/metadata.json"
observability_path="${benchmark_dir}/observability.json"

if ! jq -e \
    --argjson require_optimized "${require_optimized_kernel_evidence}" \
    --argjson require_continuous "${require_continuous_batch_evidence}" \
    --argjson require_resumable_prefill "${require_resumable_prefill_evidence}" '
    .schema_version == 1 and
    (.reports | length) > 0 and
    ([.reports[].name] | all(type == "string") and length == (unique | length)) and
    ([.reports[].report.samples | length] | all(. > 0)) and
    ([.reports[].report.summary.quality_gates.failed] | all(. == 0)) and
    ([.reports[].report.telemetry.delta_available] | all(. == true)) and
    ([.reports[] |
        . as $case |
        [(.report.telemetry.after.models // [])[] |
            select(
                .variant_id == $case.report.config.model and
                .backend_kind == "cuda" and
                .actual_device_kind == "cuda" and
                (.actual_compute_dtype == "f16" or
                 .actual_compute_dtype == "bf16" or
                 .actual_compute_dtype == "f32")
            )] | length > 0
    ] | all(. == true)) and
    ([.reports[] |
        .report.telemetry as $telemetry |
        $telemetry.after.requests_failed == $telemetry.before.requests_failed and
        $telemetry.after.worker_panics == $telemetry.before.worker_panics and
        $telemetry.after.worker_restarts == $telemetry.before.worker_restarts
    ] | all(. == true)) and
    (($require_continuous == 0) or ([.reports[] |
        .report as $report |
        $report.telemetry as $telemetry |
        (($telemetry.after.engine.tensor_continuous_batches_total -
          $telemetry.before.engine.tensor_continuous_batches_total) as $continuous |
         ($telemetry.after.engine.tensor_batches_total -
          $telemetry.before.engine.tensor_batches_total) as $tensor_batches |
         ($telemetry.after.engine.tensor_continuous_multirow_batches_total -
          $telemetry.before.engine.tensor_continuous_multirow_batches_total) as $continuous_multirow |
         ($telemetry.after.engine.tensor_batch_rows_total -
          $telemetry.before.engine.tensor_batch_rows_total) as $rows |
         ($telemetry.after.engine.tensor_batch_capacity_rows_total -
          $telemetry.before.engine.tensor_batch_capacity_rows_total) as $capacity_rows |
         ($telemetry.after.engine.tensor_batch_useful_elements_total -
          $telemetry.before.engine.tensor_batch_useful_elements_total) as $useful_elements |
         ($telemetry.after.engine.tensor_batch_materialized_elements_total -
          $telemetry.before.engine.tensor_batch_materialized_elements_total) as $materialized_elements |
         $report.command == "chat" and
         $report.config.concurrent >= 2 and
         $continuous > 0 and
         $continuous_multirow > 0 and
         $tensor_batches >= $continuous and
         $rows > $continuous and
         $capacity_rows >= $rows and
         $useful_elements > 0 and
         $materialized_elements >= $useful_elements and
         $telemetry.after.engine.tensor_batch_max_width >= 2 and
         $telemetry.after.engine.physical_batch_rejections_total ==
           $telemetry.before.engine.physical_batch_rejections_total)
    ] | all(. == true))) and
    (($require_resumable_prefill == 0) or ([.reports[] |
        .report as $report |
        $report.telemetry as $telemetry |
        (($telemetry.after.kernel_path.prefill_sequence_spans_total // 0) -
         ($telemetry.before.kernel_path.prefill_sequence_spans_total // 0)) as $spans |
        (($telemetry.after.kernel_path.prefill_sequence_tokens_total // 0) -
         ($telemetry.before.kernel_path.prefill_sequence_tokens_total // 0)) as $tokens |
        $report.command == "chat" and
        $spans > ($report.samples | length) and
        $tokens > $spans
    ] | all(. == true))) and
    (($require_optimized == 0) or ([.reports[] |
        .report.telemetry as $telemetry |
        (((($telemetry.after.kernel_path.fused_attention_success_total // 0) -
           ($telemetry.before.kernel_path.fused_attention_success_total // 0)) > 0) or
         ((($telemetry.after.kernel_path.decode_attention_paged_total // 0) -
           ($telemetry.before.kernel_path.decode_attention_paged_total // 0)) > 0) or
         ((($telemetry.after.kernel_path.rope_kernel_total // 0) -
           ($telemetry.before.kernel_path.rope_kernel_total // 0)) > 0))
    ] | all(. == true)))
' "${report_path}" >/dev/null; then
    fail "CUDA benchmark report failed case, sample, quality, runtime, batching, prefill, or optimized-kernel validation"
fi
if ! jq -e --arg git_sha "${git_sha}" '.schema_version == 1 and .git_sha == $git_sha' \
    "${metadata_path}" >/dev/null; then
    fail "CUDA benchmark metadata is not bound to the checked-out git SHA"
fi
if ! jq -e '
    [.after.metrics | .. | objects |
        select(.backend_kind? == "cuda" and .actual_device_kind? == "cuda")]
    | length > 0
' "${observability_path}" >/dev/null; then
    fail "Benchmark telemetry does not contain an observed CUDA model"
fi

ended_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
jq -n \
    --arg git_sha "${git_sha}" \
    --arg manifest "${manifest}" \
    --arg manifest_sha256 "${manifest_hash}" \
    --arg started_at "${started_at}" \
    --arg ended_at "${ended_at}" \
    --argjson require_optimized "${require_optimized_kernel_evidence}" \
    --argjson require_continuous "${require_continuous_batch_evidence}" \
    --argjson require_resumable_prefill "${require_resumable_prefill_evidence}" \
    --slurpfile health "${health_path}" \
    --slurpfile report "${report_path}" \
    '{
        schema: "izwi.cuda-model-evidence.v1",
        status: "passed",
        reason: "strict_cuda_model_evidence_passed",
        run: {
            git_sha: $git_sha,
            manifest: $manifest,
            manifest_sha256: $manifest_sha256,
            worktree_clean: true,
            started_at: $started_at,
            ended_at: $ended_at
        },
        requirements: {
            optimized_kernel: ($require_optimized == 1),
            continuous_batch: ($require_continuous == 1),
            resumable_prefill: ($require_resumable_prefill == 1)
        },
        device: {
            build_git_sha: $health[0].runtime.build_git_sha,
            requested_backend: $health[0].runtime.requested_backend,
            selected_backend: $health[0].runtime.selected_backend,
            cuda_compiled: $health[0].runtime.compiled_backends.cuda,
            driver_available: $health[0].runtime.cuda_runtime.driver_available,
            device_usable: $health[0].runtime.cuda_runtime.device_usable,
            ordinal: $health[0].runtime.detected_device.cuda_device_ordinal,
            name: $health[0].runtime.detected_device.cuda_device_name,
            compute_capability: $health[0].runtime.detected_device.cuda_compute_capability,
            free_memory_bytes: $health[0].runtime.detected_device.available_memory_bytes,
            total_memory_bytes: $health[0].runtime.detected_device.cuda_total_memory_bytes,
            selected_dtype: $health[0].runtime.dtype_policy.selected_dtype
        },
        cases: [
            $report[0].reports[] |
            . as $case |
            ([($case.report.telemetry.after.models // [])[] |
                select(
                    .variant_id == $case.report.config.model and
                    .backend_kind == "cuda" and
                    .actual_device_kind == "cuda"
                )] | first) as $runtime |
            $case.report.telemetry as $telemetry |
            (($telemetry.after.engine.tensor_batches_total // 0) -
             ($telemetry.before.engine.tensor_batches_total // 0)) as $tensor_batches |
            (($telemetry.after.engine.tensor_static_batches_total // 0) -
             ($telemetry.before.engine.tensor_static_batches_total // 0)) as $static_batches |
            (($telemetry.after.engine.tensor_continuous_batches_total // 0) -
             ($telemetry.before.engine.tensor_continuous_batches_total // 0)) as $continuous_batches |
            (($telemetry.after.engine.tensor_continuous_multirow_batches_total // 0) -
             ($telemetry.before.engine.tensor_continuous_multirow_batches_total // 0)) as $continuous_multirow_batches |
            (($telemetry.after.engine.request_parallel_batches_total // 0) -
             ($telemetry.before.engine.request_parallel_batches_total // 0)) as $request_parallel_batches |
            (($telemetry.after.engine.physical_batch_rejections_total // 0) -
             ($telemetry.before.engine.physical_batch_rejections_total // 0)) as $batch_rejections |
            (($telemetry.after.engine.tensor_batch_rows_total // 0) -
             ($telemetry.before.engine.tensor_batch_rows_total // 0)) as $rows |
            (($telemetry.after.engine.tensor_batch_capacity_rows_total // 0) -
             ($telemetry.before.engine.tensor_batch_capacity_rows_total // 0)) as $capacity_rows |
            (($telemetry.after.engine.tensor_batch_useful_elements_total // 0) -
             ($telemetry.before.engine.tensor_batch_useful_elements_total // 0)) as $useful_elements |
            (($telemetry.after.engine.tensor_batch_materialized_elements_total // 0) -
             ($telemetry.before.engine.tensor_batch_materialized_elements_total // 0)) as $materialized_elements |
            (($telemetry.after.engine.batch_workspace_bytes_total // 0) -
             ($telemetry.before.engine.batch_workspace_bytes_total // 0)) as $workspace_bytes |
            {
                name: .name,
                command: .report.command,
                model: .report.config.model,
                samples: (.report.samples | length),
                quality_failed: .report.summary.quality_gates.failed,
                telemetry_delta_available: .report.telemetry.delta_available,
                backend_kind: $runtime.backend_kind,
                actual_device_kind: $runtime.actual_device_kind,
                actual_compute_dtype: $runtime.actual_compute_dtype,
                batch_delta: {
                    tensor_batches: $tensor_batches,
                    static_batches: $static_batches,
                    continuous_batches: $continuous_batches,
                    continuous_multirow_batches: $continuous_multirow_batches,
                    request_parallel_batches: $request_parallel_batches,
                    physical_batch_rejections: $batch_rejections,
                    max_width_after: ($telemetry.after.engine.tensor_batch_max_width // 0),
                    rows: $rows,
                    capacity_rows: $capacity_rows,
                    fill_ratio: (if $capacity_rows > 0 then $rows / $capacity_rows else null end),
                    useful_elements: $useful_elements,
                    materialized_elements: $materialized_elements,
                    padding_ratio: (if $materialized_elements > 0 then
                        1 - ($useful_elements / $materialized_elements)
                      else null end),
                    workspace_bytes: $workspace_bytes
                },
                kernel_delta: {
                    host_read_ops: ((.report.telemetry.after.kernel_path.host_read_ops_total // 0) - (.report.telemetry.before.kernel_path.host_read_ops_total // 0)),
                    host_read_bytes: ((.report.telemetry.after.kernel_path.host_read_bytes_total // 0) - (.report.telemetry.before.kernel_path.host_read_bytes_total // 0)),
                    dtype_cast_ops: ((.report.telemetry.after.kernel_path.dtype_cast_ops_total // 0) - (.report.telemetry.before.kernel_path.dtype_cast_ops_total // 0)),
                    layout_copy_ops: ((.report.telemetry.after.kernel_path.layout_copy_ops_total // 0) - (.report.telemetry.before.kernel_path.layout_copy_ops_total // 0)),
                    prefill_sequence_spans: ((.report.telemetry.after.kernel_path.prefill_sequence_spans_total // 0) - (.report.telemetry.before.kernel_path.prefill_sequence_spans_total // 0)),
                    prefill_sequence_tokens: ((.report.telemetry.after.kernel_path.prefill_sequence_tokens_total // 0) - (.report.telemetry.before.kernel_path.prefill_sequence_tokens_total // 0)),
                    rope_kernel: ((.report.telemetry.after.kernel_path.rope_kernel_total // 0) - (.report.telemetry.before.kernel_path.rope_kernel_total // 0)),
                    rope_manual: ((.report.telemetry.after.kernel_path.rope_manual_total // 0) - (.report.telemetry.before.kernel_path.rope_manual_total // 0)),
                    fused_attention_attempts: ((.report.telemetry.after.kernel_path.fused_attention_attempts_total // 0) - (.report.telemetry.before.kernel_path.fused_attention_attempts_total // 0)),
                    fused_attention_success: ((.report.telemetry.after.kernel_path.fused_attention_success_total // 0) - (.report.telemetry.before.kernel_path.fused_attention_success_total // 0)),
                    fused_attention_fallback: ((.report.telemetry.after.kernel_path.fused_attention_fallback_total // 0) - (.report.telemetry.before.kernel_path.fused_attention_fallback_total // 0)),
                    decode_attention_paged: ((.report.telemetry.after.kernel_path.decode_attention_paged_total // 0) - (.report.telemetry.before.kernel_path.decode_attention_paged_total // 0))
                }
            }
        ],
        artifacts: {
            report: "benchmark/report.json",
            metadata: "benchmark/metadata.json",
            observability: "benchmark/observability.json",
            manifest: "benchmark/manifest.toml",
            health: "health.json",
            runner_log: "runner.log"
        }
    }' >"${certificate_path}"
certificate_written=1

echo "CUDA model evidence passed: ${certificate_path}"
