#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=scripts/install-cli.sh
source "$repo_root/scripts/install-cli.sh"

[[ "$(resolve_build_backend_for auto darwin arm64 12)" == cpu ]]
[[ "$(resolve_build_backend_for auto darwin arm64 14)" == cpu ]]
[[ "$(resolve_build_backend_for auto darwin arm64 15)" == metal ]]
[[ "$(resolve_build_backend_for auto darwin arm64 16)" == metal ]]
[[ "$(resolve_build_backend_for auto darwin x86_64 15)" == cpu ]]
[[ "$(resolve_build_backend_for auto linux x86_64 0)" == cpu ]]
[[ "$(resolve_build_backend_for metal darwin arm64 14 2>/dev/null)" == cpu ]]
[[ "$(resolve_build_backend_for metal darwin arm64 15)" == metal ]]
[[ "$(resolve_build_backend_for cuda linux x86_64 0)" == cuda ]]

echo "Install backend selection policy passed"
