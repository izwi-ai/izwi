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
    "${stage}/izwi-serving-worker" \
    "${stage}/izwi-desktop"
cp "${example_config}" "${stage}/izwi-serving-node.example.toml"

tar -czf "${archive}" -C "${stage}" \
    izwi \
    izwi-server \
    izwi-serving-worker \
    izwi-serving-node.example.toml \
    izwi-desktop
"${verifier}" --terminal-archive "${archive}"

mkdir -p "${windows_stage}"
touch \
    "${windows_stage}/izwi.exe" \
    "${windows_stage}/izwi-server.exe" \
    "${windows_stage}/izwi-serving-worker.exe" \
    "${windows_stage}/izwi-desktop.exe"
cp "${example_config}" "${windows_stage}/izwi-serving-node.example.toml"
(
    cd "${windows_stage}"
    zip -q "${windows_archive}" \
        izwi.exe \
        izwi-server.exe \
        izwi-serving-worker.exe \
        izwi-serving-node.example.toml \
        izwi-desktop.exe
)
"${verifier}" --zip "${windows_archive}"

rm "${stage}/izwi-serving-worker"
tar -czf "${archive}" -C "${stage}" \
    izwi \
    izwi-server \
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
    izwi-serving-worker \
    izwi-desktop
if "${verifier}" --terminal-archive "${archive}" >/dev/null 2>&1; then
    echo "error: verifier accepted a serving archive without its node configuration example" >&2
    exit 1
fi

echo "Native serving package contract passed."
