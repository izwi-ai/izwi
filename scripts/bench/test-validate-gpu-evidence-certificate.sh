#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
validator="$repo_root/scripts/bench/validate-gpu-evidence-certificate.sh"
tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT
git_sha=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

help=$($validator --help)
grep -q 'Unsupported, skipped' <<<"$help"
grep -q -- '--require-continuous-batch-evidence' <<<"$help"
grep -q -- '--require-resumable-prefill-evidence' <<<"$help"
grep -q -- '--require-audio-streaming-evidence' <<<"$help"

jq -cn --arg git_sha "$git_sha" '
  [
    {page: 16, dim: 64, batch: 1, context: 2047, provider: "cuda_native"},
    {page: 32, dim: 128, batch: 8, context: 2048, provider: "cuda_native"},
    {page: 64, dim: 256, batch: 32, context: 2049, provider: "cuda_flash_attention"}
  ][] as $cell |
  ["slot_write", "paged_prefill", "paged_decode", "page_zero", "page_copy"][] as $operation |
  {
    schema: "izwi.kv-cache-matrix.v2",
    status: "measured",
    git_sha: $git_sha,
    backend: "cuda",
    dtype: "f16",
    page_tokens: $cell.page,
    profile: "ragged",
    contexts: ($cell.context | tostring),
    requested_context_len: $cell.context,
    requested_provider: (if $cell.provider == "cuda_flash_attention" then "optimized" else "portable" end),
    observed_provider: (if $operation == "paged_prefill" or $operation == "paged_decode" then $cell.provider else null end),
    first_page_offset: 0,
    window_tokens: null,
    softcap: null,
    query_heads: 8,
    kv_heads: 2,
    head_dim: $cell.dim,
    batch_size: $cell.batch,
    iterations: 2,
    warmup: 1,
    operation: $operation,
    dispatches: 2,
    max_abs_error: 0.001,
    tolerance: 0.01
  }
' >"$tmp_dir/matrix.jsonl"
if command -v sha256sum >/dev/null 2>&1; then
  matrix_sha256=$(sha256sum "$tmp_dir/matrix.jsonl" | awk '{print $1}')
else
  matrix_sha256=$(shasum -a 256 "$tmp_dir/matrix.jsonl" | awk '{print $1}')
fi
jq -n \
  --arg git_sha "$git_sha" \
  --arg matrix "$tmp_dir/matrix.jsonl" \
  --arg matrix_sha256 "$matrix_sha256" '
  {
    schema: "izwi.gpu-kv-evidence.v1",
    status: "passed",
    reason: "required_gpu_kv_matrix_passed",
    run: {git_sha: $git_sha, worktree_clean: true},
    backend: "cuda",
    required_device: true,
    required_cells: 3,
    measured_records: 15,
    coverage: {
      unsupported_records: 0,
      failed_records: 0,
      gqa_observed: true,
      page_tokens: [16, 32, 64],
      head_dims: [64, 128, 256],
      batch_sizes: [1, 8, 32],
      partition_boundary_contexts: [2047, 2048, 2049],
      observed_providers: ["cuda_flash_attention", "cuda_native"]
    },
    artifacts: {matrix: $matrix, matrix_sha256: $matrix_sha256}
  }' >"$tmp_dir/kv-certificate.json"
$validator --certificate "$tmp_dir/kv-certificate.json" --backend cuda \
  --expected-git-sha "$git_sha" >/dev/null

jq '.coverage.observed_providers = ["cuda_native"]' "$tmp_dir/kv-certificate.json" \
  >"$tmp_dir/kv-missing-provider.json"
if $validator --certificate "$tmp_dir/kv-missing-provider.json" --backend cuda \
  --expected-git-sha "$git_sha" >/dev/null 2>&1; then
  echo 'validator accepted a KV certificate with a missing required provider' >&2
  exit 1
fi

jq -n --arg git_sha "$git_sha" '
  {
    schema: "izwi.model-evidence.v2",
    backend: "cuda",
    status: "passed",
    run: {git_sha: $git_sha, worktree_clean: true},
    requirements: {continuous_batch: true, resumable_prefill: true},
    device: {
      build_git_sha: $git_sha,
      requested_backend: "cuda",
      selected_backend: "cuda",
      backend_compiled: true,
      driver_available: true,
      device_usable: true
    },
    cases: [{
      name: "qwen3-continuous",
      command: "chat",
      samples: 8,
      quality_failed: 0,
      telemetry_delta_available: true,
      backend_kind: "cuda",
      actual_device_kind: "cuda",
      batch_delta: {
        tensor_batches: 4,
        continuous_batches: 4,
        continuous_multirow_batches: 3,
        physical_batch_rejections: 0,
        max_width_after: 8,
        rows: 24,
        capacity_rows: 32,
        useful_elements: 3072,
        materialized_elements: 4096,
        model_tensor_batches: 4,
        model_rows: 24,
        model_multirow_calls: 3,
        model_max_width_after: 8,
        continuous_scalar_fallbacks: 0
      },
      prefill_delta: {committed_quanta: 16, committed_tokens: 2048, multispan_requests: 8},
      kernel_delta: {prefill_sequence_spans: 16, prefill_sequence_tokens: 2048}
    }]
  }' >"$tmp_dir/model-certificate.json"
$validator --certificate "$tmp_dir/model-certificate.json" --backend cuda \
  --expected-git-sha "$git_sha" --require-continuous-batch-evidence \
  --require-resumable-prefill-evidence >/dev/null

jq -n '{reports: [{report: {command: "asr", config: {model: "audio-model"}}}]}' \
  >"$tmp_dir/audio-report.json"
jq -n --arg git_sha "$git_sha" '{
  schema: "izwi.audio-runtime-evidence.v1", status: "passed",
  git_sha: $git_sha, backend: "cuda",
  device: {selected_backend: "cuda", identity: "test-gpu", runtime_id: "run-1"},
  models: [{
    model: "audio-model", task: "asr", concurrency: [1, 2, 4, 8],
    correctness: {samples_compared: 8, mismatches: 0},
    fairness: {requests: 8, completed: 8, starved: 0, max_queue_wait_ms: 10, queue_wait_limit_ms: 20},
    cancellation: {cancelled_requests: 1, post_cancel_outputs: 0, live_peers_completed: 1, retained_sessions_after: 0},
    cache_pressure: {pressure_events: 1, rejections: 1, recovered_requests: 1, retained_bytes_after: 0},
    unload_drain: {attempts: 1, completed: 1, active_requests_after: 0, retained_sessions_after: 0},
    fallback: {c1_requests: 1, c1_completed: 1, unexpected_backend_fallbacks: 0},
    memory: {samples: 3, plateau: true, growth_bytes: 0, tolerance_bytes: 1024}
  }]
}' >"$tmp_dir/audio-runtime.json"
if command -v sha256sum >/dev/null 2>&1; then
  audio_sha256=$(sha256sum "$tmp_dir/audio-runtime.json" | awk '{print $1}')
else
  audio_sha256=$(shasum -a 256 "$tmp_dir/audio-runtime.json" | awk '{print $1}')
fi
jq --arg audio_sha256 "$audio_sha256" \
  --arg audio "$tmp_dir/audio-runtime.json" --arg report "$tmp_dir/audio-report.json" '
  .requirements.audio_streaming = true |
  .audio_runtime_evidence_sha256 = $audio_sha256 |
  .artifacts = {audio_runtime_evidence: $audio, report: $report} |
  .cases = ([1, 2, 4, 8] | map({
    name: ("audio-c" + tostring), command: "asr", model: "audio-model",
    concurrent: ., samples: 1, quality_failed: 0,
    telemetry_delta_available: true, backend_kind: "cuda",
    actual_device_kind: "cuda",
    streaming_latency: {
      first_audio_ms: null, inter_frame_ms: null,
      first_transcript_ms: {count: 1}, inter_transcript_ms: {count: 1}
    }
  }))
' "$tmp_dir/model-certificate.json" >"$tmp_dir/audio-certificate.json"
$validator --certificate "$tmp_dir/audio-certificate.json" --backend cuda \
  --expected-git-sha "$git_sha" --require-audio-streaming-evidence >/dev/null

jq '.models[0].memory.growth_bytes = 1' "$tmp_dir/audio-runtime.json" \
  >"$tmp_dir/audio-runtime-mutated.json"
jq --arg audio "$tmp_dir/audio-runtime-mutated.json" \
  '.artifacts.audio_runtime_evidence = $audio' "$tmp_dir/audio-certificate.json" \
  >"$tmp_dir/audio-certificate-bad-digest.json"
if $validator --certificate "$tmp_dir/audio-certificate-bad-digest.json" --backend cuda \
  --expected-git-sha "$git_sha" --require-audio-streaming-evidence >/dev/null 2>&1; then
  echo 'validator accepted mutated audio runtime evidence with a stale digest' >&2
  exit 1
fi

for portable_backend in cpu metal; do
  jq --arg backend "$portable_backend" '
    .backend = $backend |
    .device.requested_backend = $backend |
    .device.selected_backend = $backend |
    .cases[0].backend_kind = $backend |
    .cases[0].actual_device_kind = $backend
  ' "$tmp_dir/model-certificate.json" >"$tmp_dir/model-${portable_backend}.json"
  $validator --certificate "$tmp_dir/model-${portable_backend}.json" \
    --backend "$portable_backend" --expected-git-sha "$git_sha" \
    --require-continuous-batch-evidence --require-resumable-prefill-evidence >/dev/null
done

jq '.cases[0].prefill_delta.multispan_requests = .cases[0].samples - 1' \
  "$tmp_dir/model-certificate.json" >"$tmp_dir/model-no-resume.json"
if $validator --certificate "$tmp_dir/model-no-resume.json" --backend cuda \
  --expected-git-sha "$git_sha" --require-resumable-prefill-evidence >/dev/null 2>&1; then
  echo 'validator accepted a certificate without multiple prefill spans per sample' >&2
  exit 1
fi

jq '.cases[0].batch_delta.continuous_scalar_fallbacks = 1' \
  "$tmp_dir/model-certificate.json" >"$tmp_dir/model-scalar-fallback.json"
if $validator --certificate "$tmp_dir/model-scalar-fallback.json" --backend cuda \
  --expected-git-sha "$git_sha" --require-continuous-batch-evidence >/dev/null 2>&1; then
  echo 'validator accepted a scalar model fallback inside a continuous envelope' >&2
  exit 1
fi

jq '.cases[0].batch_delta.continuous_batches = 0' "$tmp_dir/model-certificate.json" \
  >"$tmp_dir/model-no-continuous.json"
if $validator --certificate "$tmp_dir/model-no-continuous.json" --backend cuda \
  --expected-git-sha "$git_sha" --require-continuous-batch-evidence >/dev/null 2>&1; then
  echo 'validator accepted a certificate without continuous batches' >&2
  exit 1
fi
jq '.cases[0].batch_delta.continuous_multirow_batches = 0' \
  "$tmp_dir/model-certificate.json" >"$tmp_dir/model-no-multirow.json"
if $validator --certificate "$tmp_dir/model-no-multirow.json" --backend cuda \
  --expected-git-sha "$git_sha" --require-continuous-batch-evidence >/dev/null 2>&1; then
  echo 'validator accepted continuous batches without a multi-row dispatch' >&2
  exit 1
fi
if $validator --certificate "$tmp_dir/model-certificate.json" --backend cuda \
  --expected-git-sha bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb \
  --require-continuous-batch-evidence >/dev/null 2>&1; then
  echo 'validator accepted a certificate from the wrong Git SHA' >&2
  exit 1
fi

echo 'GPU evidence certificate validator smoke test passed'
