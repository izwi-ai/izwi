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
dry_run=0

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
    --dry-run)
      dry_run=1
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

if [[ -n "$output" ]]; then
  mkdir -p "$(dirname "$output")"
  : >"$output"
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
  for page in 16 32; do
    for profile in ragged long; do
      write_record "{\"schema\":\"izwi.kv-cache-matrix.v1\",\"status\":\"unsupported\",\"backend\":\"$backend\",\"page_tokens\":$page,\"profile\":\"$profile\",\"reason\":\"$reason\"}"
    done
  done
}

run_case() {
  local backend=$1
  local feature=$2
  local page=$3
  local profile=$4
  local -a command=(cargo run --locked --quiet -p izwi-core)
  if [[ -n "$feature" ]]; then
    command+=(--features "$feature")
  fi
  command+=(--example kv-cache-bench --
    --backend "$backend"
    --page-tokens "$page"
    --profile "$profile"
    --iterations "$iterations"
    --warmup "$warmup"
    --long-context "$long_context")

  if ((dry_run)); then
    printf 'DRY-RUN'
    printf ' %q' "${command[@]}"
    printf '\n'
    return
  fi

  local result
  if ! result=$("${command[@]}"); then
    write_record "{\"schema\":\"izwi.kv-cache-matrix.v1\",\"status\":\"failed\",\"backend\":\"$backend\",\"page_tokens\":$page,\"profile\":\"$profile\",\"reason\":\"benchmark command failed\"}"
    return 1
  fi
  while IFS= read -r record; do
    [[ -n "$record" ]] && write_record "$record"
  done <<<"$result"
}

run_lane() {
  local backend=$1
  local feature=$2
  local page profile
  local lane_failed=0
  for page in 16 32; do
    for profile in ragged long; do
      run_case "$backend" "$feature" "$page" "$profile" || lane_failed=1
    done
  done
  return "$lane_failed"
}

host_os=${IZWI_KV_MATRIX_OS:-$(uname -s)}
metal_device=${IZWI_KV_MATRIX_METAL_DEVICE:-auto}
cuda_device=${IZWI_KV_MATRIX_CUDA_DEVICE:-auto}
failed=0

if [[ "$lane" == all || "$lane" == default ]]; then
  run_lane cpu "" || failed=1
fi

if [[ "$lane" == all || "$lane" == metal ]]; then
  if [[ "$metal_device" == 0 || ("$metal_device" == auto && "$host_os" != Darwin) ]]; then
    emit_skip metal "Metal lane requires a macOS Metal device"
  else
    run_lane metal metal || failed=1
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
    emit_skip cuda "CUDA lane requires a device visible to nvidia-smi"
  else
    run_lane cuda flash-attn || failed=1
  fi
fi

exit "$failed"
