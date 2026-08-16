#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

# Strict convenience profile for the original L40S deployment. The generic
# runner performs all validation; operator arguments come last so workflow
# smoke tests can still override --workload and --output.
exec "${repo_root}/scripts/bench/run-qwen38-cuda-evidence.sh" \
    --workload "${repo_root}/benchmarks/manifests/qwen38-l40s-evidence.json" \
    --output "${repo_root}/target/qwen38-l40s-evidence" \
    "$@"
