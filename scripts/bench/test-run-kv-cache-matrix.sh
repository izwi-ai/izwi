#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
runner="$repo_root/scripts/bench/run_kv_cache_matrix.sh"

help=$($runner --help)
grep -q -- '--lane all|default|metal|cuda' <<<"$help"
grep -q 'explicit JSON records' <<<"$help"
grep -q -- '--require-device' <<<"$help"

default_dry=$($runner --lane default --dry-run --iterations 1 --warmup 0)
[[ $(grep -c '^DRY-RUN' <<<"$default_dry") -eq 19 ]]
grep -q -- '--backend cpu' <<<"$default_dry"
grep -q -- '--page-tokens 16' <<<"$default_dry"
grep -q -- '--page-tokens 32' <<<"$default_dry"
grep -q -- '--page-tokens 64' <<<"$default_dry"
grep -q -- '--dtype bf16' <<<"$default_dry"
grep -q -- '--first-page-offset 1' <<<"$default_dry"
grep -q -- '--window-tokens 128' <<<"$default_dry"
grep -q -- '--softcap 30' <<<"$default_dry"

metal_skip=$(IZWI_KV_MATRIX_OS=Linux IZWI_KV_MATRIX_METAL_DEVICE=0 $runner --lane metal --dry-run)
[[ $(grep -c '"status":"unsupported"' <<<"$metal_skip") -eq 6 ]]
! grep -q '^DRY-RUN' <<<"$metal_skip"
if IZWI_KV_MATRIX_OS=Linux IZWI_KV_MATRIX_METAL_DEVICE=0 \
  $runner --lane metal --dry-run --require-device >/dev/null; then
  echo 'required Metal lane must fail when no device is available' >&2
  exit 1
fi

cuda_dry=$(IZWI_KV_MATRIX_CUDA_DEVICE=1 $runner --lane cuda --dry-run)
[[ $(grep -c '^DRY-RUN' <<<"$cuda_dry") -eq 21 ]]
grep -q -- '--features flash-attn' <<<"$cuda_dry"
grep -q -- '--provider portable' <<<"$cuda_dry"
grep -q -- '--provider optimized' <<<"$cuda_dry"

echo 'KV cache matrix runner smoke test passed'
