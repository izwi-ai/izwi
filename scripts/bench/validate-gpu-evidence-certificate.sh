#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Validate a retained exact-SHA backend evidence certificate.

Usage: scripts/bench/validate-gpu-evidence-certificate.sh OPTIONS

Options:
  --certificate PATH                  Certificate JSON to validate
  --backend cpu|metal|cuda            Required runtime backend
  --expected-git-sha SHA              Required 40-character source SHA
  --require-continuous-batch-evidence Require certified multi-row continuous batching
  --require-resumable-prefill-evidence Require certified multi-span resumable prefill
  -h, --help                          Show this help

The validator accepts izwi.gpu-kv-evidence.v1 for Metal/CUDA KV matrices,
legacy izwi.cuda-model-evidence.v1, and backend-neutral izwi.model-evidence.v2.
Unsupported, skipped, dirty-worktree, wrong-provider, missing-cell, and
SHA-mismatched certificates are rejected.
EOF
}

certificate=
backend=
expected_git_sha=
require_continuous=0
require_resumable_prefill=0

while (($#)); do
  case "$1" in
    --certificate)
      certificate=${2:-}
      shift 2
      ;;
    --backend)
      backend=${2:-}
      shift 2
      ;;
    --expected-git-sha)
      expected_git_sha=${2:-}
      shift 2
      ;;
    --require-continuous-batch-evidence)
      require_continuous=1
      shift
      ;;
    --require-resumable-prefill-evidence)
      require_resumable_prefill=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$certificate" || ! -s "$certificate" ]]; then
  echo "error: --certificate must name a non-empty JSON file" >&2
  exit 2
fi
case "$backend" in
  cpu|metal|cuda) ;;
  *) echo "error: --backend must be cpu, metal, or cuda" >&2; exit 2 ;;
esac
if [[ ! "$expected_git_sha" =~ ^[0-9a-f]{40}$ ]]; then
  echo "error: --expected-git-sha must be a lowercase 40-character Git SHA" >&2
  exit 2
fi
command -v jq >/dev/null 2>&1 || { echo "error: jq is required" >&2; exit 1; }

schema=$(jq -r '.schema // empty' "$certificate")
case "$schema" in
  izwi.gpu-kv-evidence.v1)
    if [[ "$backend" == cpu ]]; then
      echo "error: GPU KV evidence cannot certify a CPU runtime" >&2
      exit 1
    fi
    if ((require_continuous || require_resumable_prefill)); then
      echo "error: KV-only evidence cannot certify model batching or resumable prefill" >&2
      exit 1
    fi
    if ! jq -e \
      --arg backend "$backend" \
      --arg git_sha "$expected_git_sha" '
        .schema == "izwi.gpu-kv-evidence.v1" and
        .status == "passed" and
        .reason == "required_gpu_kv_matrix_passed" and
        .run.git_sha == $git_sha and
        .run.worktree_clean == true and
        .backend == $backend and
        .required_device == true and
        .required_cells > 0 and
        .measured_records == (.required_cells * 5) and
        .coverage.unsupported_records == 0 and
        .coverage.failed_records == 0 and
        .coverage.gqa_observed == true and
        (.coverage.page_tokens | index(16) != null and index(32) != null and index(64) != null) and
        (.coverage.head_dims | index(64) != null and index(128) != null and index(256) != null) and
        (.coverage.batch_sizes | index(1) != null and index(8) != null and index(32) != null) and
        (.coverage.partition_boundary_contexts == [2047, 2048, 2049]) and
        (if $backend == "cuda" then
          (.coverage.observed_providers |
            index("cuda_native") != null and index("cuda_flash_attention") != null)
         else
          (.coverage.observed_providers | index("metal_native") != null)
         end)
      ' "$certificate" >/dev/null; then
      echo "error: GPU KV certificate failed exact-SHA, provider, or required-cell validation" >&2
      exit 1
    fi

    matrix_path=$(jq -r '.artifacts.matrix // empty' "$certificate")
    matrix_sha256=$(jq -r '.artifacts.matrix_sha256 // empty' "$certificate")
    if [[ -z "$matrix_path" || ! -f "$matrix_path" ]]; then
      certificate_dir=$(cd "$(dirname "$certificate")" && pwd)
      if [[ -n "$matrix_path" && -f "$certificate_dir/$matrix_path" ]]; then
        matrix_path="$certificate_dir/$matrix_path"
      else
        echo "error: retained KV matrix referenced by the certificate is missing" >&2
        exit 1
      fi
    fi
    if command -v sha256sum >/dev/null 2>&1; then
      actual_sha256=$(sha256sum "$matrix_path" | awk '{print $1}')
    elif command -v shasum >/dev/null 2>&1; then
      actual_sha256=$(shasum -a 256 "$matrix_path" | awk '{print $1}')
    else
      echo "error: sha256sum or shasum is required" >&2
      exit 1
    fi
    if [[ "$actual_sha256" != "$matrix_sha256" ]]; then
      echo "error: retained KV matrix SHA-256 does not match the certificate" >&2
      exit 1
    fi
    required_cells=$(jq -r '.required_cells' "$certificate")
    if ! jq -s -e \
      --arg backend "$backend" \
      --arg git_sha "$expected_git_sha" \
      --argjson required_cells "$required_cells" '
        length == ($required_cells * 5) and
        all(.[];
          .schema == "izwi.kv-cache-matrix.v2" and
          .status == "measured" and
          .backend == $backend and
          .git_sha == $git_sha and
          .dispatches > 0 and
          .max_abs_error <= .tolerance) and
        ([.[].operation] | group_by(.) |
          length == 5 and all(.[]; length == $required_cells)) and
        ([.[].page_tokens] | index(16) != null and index(32) != null and index(64) != null) and
        ([.[].head_dim] | index(64) != null and index(128) != null and index(256) != null) and
        ([.[].batch_size] | index(1) != null and index(8) != null and index(32) != null) and
        ([.[].requested_context_len] |
          index(2047) != null and index(2048) != null and index(2049) != null) and
        any(.[];
          (.operation == "paged_prefill" or .operation == "paged_decode") and
          .query_heads > .kv_heads and (.query_heads % .kv_heads) == 0) and
        (if $backend == "cuda" then
          ([.[] | select(.operation == "paged_prefill" or .operation == "paged_decode") |
            .observed_provider] |
            index("cuda_native") != null and index("cuda_flash_attention") != null)
         else
          ([.[] | select(.operation == "paged_prefill" or .operation == "paged_decode") |
            .observed_provider] | index("metal_native") != null)
         end)
      ' "$matrix_path" >/dev/null; then
      echo "error: retained KV matrix failed independent required-cell reconstruction" >&2
      exit 1
    fi
    ;;
  izwi.cuda-model-evidence.v1|izwi.model-evidence.v2)
    if [[ "$schema" == izwi.cuda-model-evidence.v1 && "$backend" != cuda ]]; then
      echo "error: legacy CUDA model evidence cannot certify another runtime" >&2
      exit 1
    fi
    if ! jq -e \
      --arg schema "$schema" \
      --arg backend "$backend" \
      --arg git_sha "$expected_git_sha" \
      --argjson require_continuous "$require_continuous" \
      --argjson require_resumable_prefill "$require_resumable_prefill" '
        .schema == $schema and
        (if $schema == "izwi.model-evidence.v2" then .backend == $backend else $backend == "cuda" end) and
        .status == "passed" and
        .run.git_sha == $git_sha and
        .run.worktree_clean == true and
        .device.build_git_sha == $git_sha and
        .device.requested_backend == $backend and
        .device.selected_backend == $backend and
        (if $schema == "izwi.model-evidence.v2" then
           .device.backend_compiled == true
         else .device.cuda_compiled == true end) and
        .device.driver_available == true and
        .device.device_usable == true and
        (.cases | length) > 0 and
        all(.cases[];
          .quality_failed == 0 and
          .telemetry_delta_available == true and
          .backend_kind == $backend and
          .actual_device_kind == $backend) and
        (($require_continuous == 0) or
          (.requirements.continuous_batch == true and
           all(.cases[];
             .command == "chat" and
             .batch_delta.continuous_batches > 0 and
             .batch_delta.continuous_multirow_batches > 0 and
             .batch_delta.tensor_batches >= .batch_delta.continuous_batches and
             .batch_delta.rows > .batch_delta.continuous_batches and
             .batch_delta.capacity_rows >= .batch_delta.rows and
             .batch_delta.max_width_after >= 2 and
             .batch_delta.useful_elements > 0 and
             .batch_delta.materialized_elements >= .batch_delta.useful_elements and
             .batch_delta.model_tensor_batches > 0 and
             .batch_delta.model_multirow_calls > 0 and
             .batch_delta.model_rows > .batch_delta.model_multirow_calls and
             .batch_delta.model_max_width_after >= 2 and
             .batch_delta.continuous_scalar_fallbacks == 0 and
             .batch_delta.physical_batch_rejections == 0)))
        and
        (($require_resumable_prefill == 0) or
          (.requirements.resumable_prefill == true and
           all(.cases[];
             .command == "chat" and
             .prefill_delta.multispan_requests >= .samples and
             .prefill_delta.committed_quanta > .prefill_delta.multispan_requests and
             .prefill_delta.committed_tokens > .prefill_delta.committed_quanta)))
      ' "$certificate" >/dev/null; then
      echo "error: model certificate failed exact-SHA, runtime, batching, or prefill validation" >&2
      exit 1
    fi
    ;;
  *)
    echo "error: unsupported GPU evidence certificate schema: ${schema:-missing}" >&2
    exit 1
    ;;
esac

echo "GPU evidence certificate validated: $certificate"
