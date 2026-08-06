#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
runner="$repo_root/scripts/bench/run_kv_cache_matrix.sh"

help=$($runner --help)
grep -q -- '--lane all|default|metal|cuda' <<<"$help"
grep -q 'explicit JSON records' <<<"$help"

default_dry=$($runner --lane default --dry-run --iterations 1 --warmup 0)
[[ $(grep -c '^DRY-RUN' <<<"$default_dry") -eq 4 ]]
grep -q -- '--backend cpu' <<<"$default_dry"
grep -q -- '--page-tokens 16' <<<"$default_dry"
grep -q -- '--page-tokens 32' <<<"$default_dry"

metal_skip=$(IZWI_KV_MATRIX_OS=Linux IZWI_KV_MATRIX_METAL_DEVICE=0 $runner --lane metal --dry-run)
[[ $(grep -c '"status":"unsupported"' <<<"$metal_skip") -eq 4 ]]
! grep -q '^DRY-RUN' <<<"$metal_skip"

cuda_dry=$(IZWI_KV_MATRIX_CUDA_DEVICE=1 $runner --lane cuda --dry-run)
[[ $(grep -c '^DRY-RUN' <<<"$cuda_dry") -eq 4 ]]
grep -q -- '--features flash-attn' <<<"$cuda_dry"

echo 'KV cache matrix runner smoke test passed'
