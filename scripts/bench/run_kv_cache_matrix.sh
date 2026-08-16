#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the managed-KV certification microbenchmark matrix.

Usage: scripts/bench/run_kv_cache_matrix.sh [OPTIONS]

Options:
  --lane all|default|metal|cuda  Feature lane to run (default: all)
  --iterations N                Measured iterations per case (default: 10)
  --warmup N                    Warmup iterations per case (default: 2)
  --long-context N              Maximum context for long cases (default: 1024)
  --output PATH                 Also write JSON Lines to PATH
  --certificate PATH            Write a strict GPU certificate (Metal/CUDA only)
  --expected-git-sha SHA        Exact 40-character source SHA for certification
  --require-device             Fail instead of skipping a requested hardware lane
  --dry-run                     Print commands/capability decisions only
  -h, --help                    Show this help

The default lane benchmarks CPU with izwi-core's default feature set. Metal is
attempted only on macOS. CUDA is attempted only when `nvidia-smi -L` reports a
usable device. Skipped lanes emit explicit JSON records; they are never
reported as measurements. Timings come from the public KV arena ABI and include
an explicit synchronization after every operation.

Test-only capability overrides used by the companion smoke test:
  IZWI_KV_MATRIX_OS, IZWI_KV_MATRIX_METAL_DEVICE, IZWI_KV_MATRIX_CUDA_DEVICE
EOF
}

lane=all
iterations=10
warmup=2
long_context=1024
output=
certificate=
expected_git_sha=
dry_run=0
require_device=0
started_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

while (($#)); do
  case "$1" in
    --lane)
      lane=${2:-}
      shift 2
      ;;
    --iterations)
      iterations=${2:-}
      shift 2
      ;;
    --warmup)
      warmup=${2:-}
      shift 2
      ;;
    --long-context)
      long_context=${2:-}
      shift 2
      ;;
    --output)
      output=${2:-}
      shift 2
      ;;
    --certificate)
      certificate=${2:-}
      shift 2
      ;;
    --expected-git-sha)
      expected_git_sha=${2:-}
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --require-device)
      require_device=1
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

case "$lane" in
  all|default|metal|cuda) ;;
  *) echo "error: --lane must be all, default, metal, or cuda" >&2; exit 2 ;;
esac
case "$iterations" in ''|*[!0-9]*|0) echo "error: --iterations must be positive" >&2; exit 2 ;; esac
case "$warmup" in ''|*[!0-9]*) echo "error: --warmup must be non-negative" >&2; exit 2 ;; esac
case "$long_context" in ''|*[!0-9]*|0) echo "error: --long-context must be positive" >&2; exit 2 ;; esac

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
git_sha=$(git rev-parse HEAD 2>/dev/null || true)

if [[ -n "$expected_git_sha" ]]; then
  if [[ ! "$expected_git_sha" =~ ^[0-9a-f]{40}$ ]]; then
    echo "error: --expected-git-sha must be a lowercase 40-character Git SHA" >&2
    exit 2
  fi
  if [[ "$git_sha" != "$expected_git_sha" ]]; then
    echo "error: checked-out Git SHA $git_sha does not match expected SHA $expected_git_sha" >&2
    exit 1
  fi
fi

if [[ -n "$certificate" ]]; then
  if ((dry_run)); then
    echo "error: --certificate cannot be combined with --dry-run" >&2
    exit 2
  fi
  if [[ "$lane" != metal && "$lane" != cuda ]]; then
    echo "error: --certificate requires --lane metal or --lane cuda" >&2
    exit 2
  fi
  if ((require_device == 0)); then
    echo "error: --certificate requires --require-device" >&2
    exit 2
  fi
  if [[ -z "$output" ]]; then
    echo "error: --certificate requires --output" >&2
    exit 2
  fi
  if [[ -z "$expected_git_sha" ]]; then
    echo "error: --certificate requires --expected-git-sha" >&2
    exit 2
  fi
  if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
    echo "error: GPU certification requires a clean exact-SHA worktree" >&2
    exit 1
  fi
  command -v jq >/dev/null 2>&1 || {
    echo "error: GPU certification requires jq" >&2
    exit 1
  }
fi

if [[ -n "$output" ]]; then
  mkdir -p "$(dirname "$output")"
  : >"$output"
fi
if ((dry_run == 0)) && ! command -v jq >/dev/null 2>&1; then
  echo "error: managed-KV evidence validation requires jq" >&2
  exit 1
fi

write_record() {
  local record=$1
  printf '%s\n' "$record"
  if [[ -n "$output" ]]; then
    printf '%s\n' "$record" >>"$output"
  fi
}

emit_skip() {
  local backend=$1
  local reason=$2
  local page profile
  for page in 16 32 64; do
    for profile in ragged long; do
      write_record "{\"schema\":\"izwi.kv-cache-matrix.v2\",\"status\":\"unsupported\",\"backend\":\"$backend\",\"page_tokens\":$page,\"profile\":\"$profile\",\"reason\":\"$reason\"}"
    done
  done
  ((require_device == 0))
}

run_case() {
  local backend=$1
  local feature=$2
  local page=$3
  local profile=$4
  local dtype=$5
  local provider=$6
  shift 6
  local -a command=(cargo run --locked --quiet -p izwi-core)
  if [[ -n "$feature" ]]; then
    command+=(--features "$feature")
  fi
  command+=(--example kv-cache-bench --
    --backend "$backend"
    --page-tokens "$page"
    --profile "$profile"
    --dtype "$dtype"
    --provider "$provider"
    --iterations "$iterations"
    --warmup "$warmup"
    --long-context "$long_context" "$@")

  if ((dry_run)); then
    printf 'DRY-RUN'
    printf ' %q' "${command[@]}"
    printf '\n'
    return
  fi

  local result
  if ! result=$("${command[@]}"); then
    write_record "{\"schema\":\"izwi.kv-cache-matrix.v2\",\"status\":\"failed\",\"git_sha\":\"$git_sha\",\"backend\":\"$backend\",\"dtype\":\"$dtype\",\"page_tokens\":$page,\"profile\":\"$profile\",\"requested_provider\":\"$provider\",\"reason\":\"benchmark command failed\"}"
    return 1
  fi
  if ((require_device)) && grep -q '"status":"unsupported"' <<<"$result"; then
    printf '%s\n' "$result"
    echo "error: required $backend device case was reported unsupported" >&2
    return 1
  fi
  local expected_provider
  case "$backend:$provider" in
    cpu:*) expected_provider=portable ;;
    metal:*) expected_provider=metal_native ;;
    cuda:optimized) expected_provider=cuda_flash_attention ;;
    cuda:*) expected_provider=cuda_native ;;
    *)
      echo "error: no provider contract for $backend/$provider" >&2
      return 1
      ;;
  esac
  if ! jq -s -e \
    --arg backend "$backend" \
    --arg dtype "$dtype" \
    --arg profile "$profile" \
    --arg provider "$provider" \
    --arg expected_provider "$expected_provider" \
    --argjson page "$page" '
      length == 5 and
      all(.[];
        .schema == "izwi.kv-cache-matrix.v2" and
        .status == "measured" and
        .backend == $backend and
        .dtype == $dtype and
        .profile == $profile and
        .page_tokens == $page and
        .requested_provider == $provider and
        (.dispatches | type == "number" and . > 0) and
        (.max_abs_error | type == "number") and
        (.tolerance | type == "number") and
        .max_abs_error <= .tolerance) and
      ([.[].operation] | sort) ==
        ["page_copy", "page_zero", "paged_decode", "paged_prefill", "slot_write"] and
      all(.[] | select(.operation == "paged_prefill" or .operation == "paged_decode");
        .observed_provider == $expected_provider)
    ' <<<"$result" >/dev/null; then
    echo "error: $backend/$dtype/page-$page/$profile/$provider did not emit the exact required operation/provider cell" >&2
    write_record "{\"schema\":\"izwi.kv-cache-matrix.v2\",\"status\":\"failed\",\"git_sha\":\"$git_sha\",\"backend\":\"$backend\",\"dtype\":\"$dtype\",\"page_tokens\":$page,\"profile\":\"$profile\",\"requested_provider\":\"$provider\",\"reason\":\"required operation or provider evidence missing\"}"
    return 1
  fi
  while IFS= read -r record; do
    if [[ -n "$record" ]]; then
      record=$(jq -c --arg git_sha "$git_sha" '. + {git_sha: $git_sha}' <<<"$record")
      write_record "$record"
    fi
  done <<<"$result"
}

run_lane() {
  local backend=$1
  local feature=$2
  local dtypes=$3
  local providers=$4
  local page profile dtype provider
  local primary_dtype=${dtypes%% *}
  local primary_provider=${providers%% *}
  local lane_failed=0
  for dtype in $dtypes; do
    for provider in $providers; do
      for page in 16 32 64; do
        # The optimized CUDA provider requires a page size divisible by 32.
        # Keep page-16 in the portable corpus instead of accepting fallback
        # while claiming an optimized certification cell.
        if [[ "$provider" == optimized && "$page" == 16 ]]; then
          continue
        fi
        for profile in ragged long; do
          if [[ "$provider" == optimized ]]; then
            # Flash paged attention requires zero first-page offsets. Make the
            # requested provider cell eligible instead of accepting native
            # fallback under an "optimized" label.
            run_case "$backend" "$feature" "$page" "$profile" "$dtype" "$provider" \
              --first-page-offset 0 || lane_failed=1
          else
            run_case "$backend" "$feature" "$page" "$profile" "$dtype" "$provider" \
              || lane_failed=1
          fi
        done
      done
    done
  done
  # One deliberately non-default semantic/shape cell prevents the broad matrix
  # from accidentally certifying only zero-offset, full-context GQA.
  run_case "$backend" "$feature" 64 long "$primary_dtype" "$primary_provider" \
    --first-page-offset 1 --window-tokens 128 --softcap 30 \
    --query-heads 8 --kv-heads 1 --head-dim 80 --batch-size 3 --context-len 257 \
    || lane_failed=1
  # Required realistic geometry cells cover MQA/GQA, the production head
  # dimensions, and tensor widths used by the candidate CUDA tiers.
  run_case "$backend" "$feature" 16 ragged "$primary_dtype" "$primary_provider" \
    --first-page-offset 0 --query-heads 8 --kv-heads 1 --head-dim 64 \
    --batch-size 1 --context-len 257 || lane_failed=1
  run_case "$backend" "$feature" 32 ragged "$primary_dtype" "$primary_provider" \
    --first-page-offset 0 --query-heads 32 --kv-heads 8 --head-dim 128 \
    --batch-size 8 --context-len 513 || lane_failed=1
  run_case "$backend" "$feature" 64 ragged "$primary_dtype" "$primary_provider" \
    --first-page-offset 0 --query-heads 16 --kv-heads 4 --head-dim 256 \
    --batch-size 32 --context-len 257 || lane_failed=1
  # CUDA switches to partitioned native decode at 2,048 tokens. Keep the
  # exact below/at/above cells in every lane so provider parity is comparable.
  local boundary_context
  for boundary_context in 2047 2048 2049; do
    run_case "$backend" "$feature" 64 long "$primary_dtype" "$primary_provider" \
      --first-page-offset 0 --query-heads 8 --kv-heads 2 --head-dim 128 \
      --batch-size 8 --context-len "$boundary_context" || lane_failed=1
  done
  return "$lane_failed"
}

host_os=${IZWI_KV_MATRIX_OS:-$(uname -s)}
metal_device=${IZWI_KV_MATRIX_METAL_DEVICE:-auto}
cuda_device=${IZWI_KV_MATRIX_CUDA_DEVICE:-auto}
failed=0

if [[ "$lane" == all || "$lane" == default ]]; then
  run_lane cpu "" "f32 f16 bf16" "portable" || failed=1
fi

if [[ "$lane" == all || "$lane" == metal ]]; then
  if [[ "$metal_device" == 0 || ("$metal_device" == auto && "$host_os" != Darwin) ]]; then
    emit_skip metal "Metal lane requires a macOS Metal device" || failed=1
  else
    run_lane metal metal "f32 f16" "portable" || failed=1
  fi
fi

if [[ "$lane" == all || "$lane" == cuda ]]; then
  if [[ "$cuda_device" == auto ]]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
      cuda_device=1
    else
      cuda_device=0
    fi
  fi
  if [[ "$cuda_device" == 0 ]]; then
    emit_skip cuda "CUDA lane requires a device visible to nvidia-smi" || failed=1
  else
    run_lane cuda flash-attn "f16 bf16" "portable optimized" || failed=1
  fi
fi

if ((failed)); then
  exit 1
fi

if [[ -n "$certificate" ]]; then
  expected_case_count=19
  if [[ "$lane" == cuda ]]; then
    expected_case_count=27
  fi
  expected_record_count=$((expected_case_count * 5))
  if ! jq -s -e \
    --arg backend "$lane" \
    --arg git_sha "$git_sha" \
    --argjson expected_records "$expected_record_count" '
      length == $expected_records and
      all(.[];
        .schema == "izwi.kv-cache-matrix.v2" and
        .status == "measured" and
        .backend == $backend and
        .git_sha == $git_sha and
        .max_abs_error <= .tolerance) and
      ([.[].page_tokens] | index(16) != null and index(32) != null and index(64) != null) and
      ([.[].head_dim] | index(64) != null and index(128) != null and index(256) != null) and
      ([.[].batch_size] | index(1) != null and index(8) != null and index(32) != null) and
      ([.[].requested_context_len] |
        index(2047) != null and index(2048) != null and index(2049) != null) and
      ([.[] | select(
        (.operation == "paged_prefill" or .operation == "paged_decode") and
        .query_heads > .kv_heads and
        (.query_heads % .kv_heads) == 0)] | length > 0) and
      (if $backend == "cuda" then
        ([.[] | select(.operation == "paged_prefill" or .operation == "paged_decode") |
          .observed_provider] |
          index("cuda_native") != null and index("cuda_flash_attention") != null)
       else
        ([.[] | select(.operation == "paged_prefill" or .operation == "paged_decode") |
          .observed_provider] | index("metal_native") != null)
       end)
    ' "$output" >/dev/null; then
    echo "error: retained GPU matrix is missing a required geometry, boundary, or provider cell" >&2
    exit 1
  fi

  if command -v sha256sum >/dev/null 2>&1; then
    matrix_sha256=$(sha256sum "$output" | awk '{print $1}')
  elif command -v shasum >/dev/null 2>&1; then
    matrix_sha256=$(shasum -a 256 "$output" | awk '{print $1}')
  else
    echo "error: GPU certification requires sha256sum or shasum" >&2
    exit 1
  fi
  ended_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  mkdir -p "$(dirname "$certificate")"
  jq -s \
    --arg git_sha "$git_sha" \
    --arg backend "$lane" \
    --arg started_at "$started_at" \
    --arg ended_at "$ended_at" \
    --arg matrix "$output" \
    --arg matrix_sha256 "$matrix_sha256" \
    --arg host_os "$host_os" \
    --argjson expected_cells "$expected_case_count" '
      {
        schema: "izwi.gpu-kv-evidence.v1",
        status: "passed",
        reason: "required_gpu_kv_matrix_passed",
        run: {
          git_sha: $git_sha,
          worktree_clean: true,
          started_at: $started_at,
          ended_at: $ended_at,
          host_os: $host_os
        },
        backend: $backend,
        required_device: true,
        required_cells: $expected_cells,
        measured_records: length,
        coverage: {
          page_tokens: ([.[].page_tokens] | unique | sort),
          dtypes: ([.[].dtype] | unique | sort),
          requested_providers: ([.[].requested_provider] | unique | sort),
          observed_providers: ([.[].observed_provider | select(. != null)] | unique | sort),
          head_dims: ([.[].head_dim] | unique | sort),
          batch_sizes: ([.[].batch_size] | unique | sort),
          partition_boundary_contexts: ([.[].requested_context_len |
            select(. == 2047 or . == 2048 or . == 2049)] | unique | sort),
          gqa_observed: any(.[];
            (.operation == "paged_prefill" or .operation == "paged_decode") and
            .query_heads > .kv_heads and (.query_heads % .kv_heads) == 0),
          unsupported_records: ([.[] | select(.status == "unsupported")] | length),
          failed_records: ([.[] | select(.status == "failed")] | length)
        },
        artifacts: {
          matrix: $matrix,
          matrix_sha256: $matrix_sha256
        }
      }
    ' "$output" >"$certificate"
  echo "GPU KV evidence passed: $certificate" >&2
fi

exit 0
