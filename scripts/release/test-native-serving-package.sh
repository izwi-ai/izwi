#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
verifier="${repo_root}/scripts/release/verify-native-artifacts.sh"
example_config="${repo_root}/config/serving/izwi-serving-node.example.toml"
fixture_root="$(mktemp -d)"
trap 'rm -rf "${fixture_root}"' EXIT

stage="${fixture_root}/stage"
archive="${fixture_root}/izwi-cli-vtest-native.tar.gz"
windows_stage="${fixture_root}/windows-stage"
windows_archive="${fixture_root}/izwi-cli-vtest-windows-x86_64.zip"
mkdir -p "${stage}"
touch \
    "${stage}/izwi" \
    "${stage}/izwi-server" \
    "${stage}/izwi-serving-supervisor" \
    "${stage}/izwi-serving-worker" \
    "${stage}/izwi-desktop"
cp "${example_config}" "${stage}/izwi-serving-node.example.toml"

tar -czf "${archive}" -C "${stage}" \
    izwi \
    izwi-server \
    izwi-serving-supervisor \
    izwi-serving-worker \
    izwi-serving-node.example.toml \
    izwi-desktop
"${verifier}" --terminal-archive "${archive}"

mkdir -p "${windows_stage}"
touch \
    "${windows_stage}/izwi.exe" \
    "${windows_stage}/izwi-server.exe" \
    "${windows_stage}/izwi-serving-supervisor.exe" \
    "${windows_stage}/izwi-serving-worker.exe" \
    "${windows_stage}/izwi-desktop.exe"
cp "${example_config}" "${windows_stage}/izwi-serving-node.example.toml"
(
    cd "${windows_stage}"
    zip -q "${windows_archive}" \
        izwi.exe \
        izwi-server.exe \
        izwi-serving-supervisor.exe \
        izwi-serving-worker.exe \
        izwi-serving-node.example.toml \
        izwi-desktop.exe
)
"${verifier}" --zip "${windows_archive}"

rm "${stage}/izwi-serving-worker"
tar -czf "${archive}" -C "${stage}" \
    izwi \
    izwi-server \
    izwi-serving-supervisor \
    izwi-serving-node.example.toml \
    izwi-desktop
if "${verifier}" --terminal-archive "${archive}" >/dev/null 2>&1; then
    echo "error: verifier accepted a serving archive without its worker" >&2
    exit 1
fi

rm "${stage}/izwi-serving-node.example.toml"
touch "${stage}/izwi-serving-worker"
tar -czf "${archive}" -C "${stage}" \
    izwi \
    izwi-server \
    izwi-serving-supervisor \
    izwi-serving-worker \
    izwi-desktop
if "${verifier}" --terminal-archive "${archive}" >/dev/null 2>&1; then
    echo "error: verifier accepted a serving archive without its node configuration example" >&2
    exit 1
fi

cp "${example_config}" "${stage}/izwi-serving-node.example.toml"
rm "${stage}/izwi-serving-supervisor"
tar -czf "${archive}" -C "${stage}" \
    izwi \
    izwi-server \
    izwi-serving-worker \
    izwi-serving-node.example.toml \
    izwi-desktop
if "${verifier}" --terminal-archive "${archive}" >/dev/null 2>&1; then
    echo "error: verifier accepted a serving archive without its supervisor" >&2
    exit 1
fi

cp "${example_config}" "${stage}/izwi-serving-node.example.toml"
python3 - "${stage}/izwi-serving-node.example.toml" << 'PYEOF'
import sys
import tomllib

path = sys.argv[1]
with open(path, "rb") as handle:
    config = tomllib.load(handle)
assert config.get("schema_version") == 2, "node example must pin schema version 2"
workers = config.get("workers", [])
assert len(workers) >= 1, "node example must declare at least one worker"
for worker in workers:
    assert worker.get("worker_id"), "every example worker needs an identity"
    assert worker.get("bearer_token_env"), "example secrets must stay in the environment"
    assert "bearer_token" not in worker, "example must never inline a secret"
    deployment = worker.get("deployment", {})
    assert deployment.get("model_generation", 0) != 0, "example generation must be non-zero"
print("node example topology contract passed.")
PYEOF

echo "Native serving package contract passed."
