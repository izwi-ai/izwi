#!/usr/bin/env bash
# Prove the serving control plane stays accelerator-free.
#
# The gateway, protocol, transport client, and supervisor must never depend
# on inference backends: a gateway process must not construct a runtime,
# probe a device, or link accelerator code, on CPU, Metal, or CUDA alike.
# Only izwi-serving-worker (which hosts RuntimeService) may depend on
# izwi-core. This gate fails closed if any boundary crate gains a backend
# dependency.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

boundary_crates=(
    izwi-serving-protocol
    izwi-serving-client
    izwi-serving-supervisor
)
forbidden_pattern='(izwi-core|candle-core|candle-nn|candle-transformers|metal |/metal-|cuda|cudnn|accelerate)'

failures=0
for crate in "${boundary_crates[@]}"; do
    tree="$(cargo tree --offline -p "${crate}" --prefix none 2>/dev/null || true)"
    if [ -z "${tree}" ]; then
        echo "error: could not resolve dependency tree for ${crate}" >&2
        failures=$((failures + 1))
        continue
    fi
    hit="$(printf '%s\n' "${tree}" | grep -Ei "${forbidden_pattern}" || true)"
    if [ -n "${hit}" ]; then
        echo "error: boundary crate ${crate} depends on accelerator code:" >&2
        printf '%s\n' "${hit}" >&2
        failures=$((failures + 1))
    else
        echo "ok: ${crate} is accelerator-free"
    fi
done

worker_tree="$(cargo tree --offline -p izwi-serving-worker --prefix none 2>/dev/null || true)"
if ! printf '%s\n' "${worker_tree}" | grep -Eq 'izwi-core v'; then
    echo "error: izwi-serving-worker must own its runtime via izwi-core" >&2
    failures=$((failures + 1))
else
    echo "ok: izwi-serving-worker owns its runtime via izwi-core"
fi

if [ "${failures}" -ne 0 ]; then
    echo "error: serving boundary violated (${failures} failures)" >&2
    exit 1
fi
echo "Serving boundary contract passed."
