#!/usr/bin/env bash
# Prove the serving control plane stays accelerator-free.
#
# The gateway, protocol, transport client, and supervisor must never depend
# on inference backends: a gateway process must not construct a runtime,
# probe a device, or link accelerator code, on CPU, Metal, or CUDA alike.
# Only izwi-serving-worker (which hosts RuntimeService) may depend on
# izwi-core. The gateway role is still served by izwi-server, which links
# izwi-core for its chat path; the gateway components already in
# izwi-gateway must stay free of it. This gate fails closed if any boundary
# crate gains a backend dependency. Only linked (normal and build) dependencies count: test-only
# dev-dependencies never reach a shipped binary.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

boundary_crates=(
    izwi-env
    izwi-gateway
    izwi-serving-protocol
    izwi-serving-client
    izwi-serving-fence
    izwi-serving-supervisor
)
tree_of() {
    cargo tree --locked -p "$1" --edges normal,build --prefix none 2>/dev/null || true
}
forbidden_pattern='(izwi-core|candle-core|candle-nn|candle-transformers|metal |/metal-|cuda|cudnn|accelerate)'

failures=0
for crate in "${boundary_crates[@]}"; do
    tree="$(tree_of "${crate}")"
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

worker_tree="$(tree_of izwi-serving-worker)"
if ! printf '%s\n' "${worker_tree}" | grep -Eq 'izwi-core v'; then
    echo "error: izwi-serving-worker must own its runtime via izwi-core" >&2
    failures=$((failures + 1))
else
    echo "ok: izwi-serving-worker owns its runtime via izwi-core"
fi
# The worker shares only the launch contract (izwi-serving-fence) with its
# supervisor; linking the supervisor would drag its control plane in.
if printf '%s\n' "${worker_tree}" | grep -Eq 'izwi-serving-supervisor v'; then
    echo "error: izwi-serving-worker must not link izwi-serving-supervisor" >&2
    failures=$((failures + 1))
else
    echo "ok: izwi-serving-worker does not link izwi-serving-supervisor"
fi

# A worker is the server's engine in another process: each backend feature
# must enable exactly the izwi-core features the server's does (Accelerate on
# macOS, FlashAttention with `cuda`), whichever binary a release builds.
core_features_of() {
    local tree
    tree="$(cargo tree --locked -p "$1" --target "$2" ${3:+--features "$3"} \
        --edges features,normal -i izwi-core --prefix none)" || return 1
    printf '%s\n' "${tree}" | sed -n 's/^izwi-core feature "\([^"]*\)".*/\1/p' \
        | sort -u | tr '\n' ' '
}
feature_mismatches=0
for target in aarch64-apple-darwin x86_64-unknown-linux-gnu; do
    for features in "" metal cuda-base cudnn-base cuda cudnn flash-attn; do
        if ! server="$(core_features_of izwi-server "${target}" "${features}")" \
            || ! worker="$(core_features_of izwi-serving-worker "${target}" "${features}")"; then
            echo "error: could not resolve izwi-core features for --features '${features}' on ${target}" >&2
            feature_mismatches=$((feature_mismatches + 1))
        elif [ "${server}" != "${worker}" ]; then
            echo "error: --features '${features}' on ${target} enables izwi-core [${server}] for izwi-server but [${worker}] for izwi-serving-worker" >&2
            feature_mismatches=$((feature_mismatches + 1))
        fi
    done
done
if [ "${feature_mismatches}" -ne 0 ]; then
    failures=$((failures + feature_mismatches))
else
    echo "ok: izwi-serving-worker enables the same izwi-core features as izwi-server"
fi

if [ "${failures}" -ne 0 ]; then
    echo "error: serving boundary violated (${failures} failures)" >&2
    exit 1
fi
echo "Serving boundary contract passed."
