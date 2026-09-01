#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the deterministic managed-KV lifecycle stress test.

Usage: scripts/ci/run-kv-lifecycle-soak.sh [OPTIONS]

Options:
  --profile smoke|pr|nightly|release  Workload preset (default: smoke)
  --iterations N                     Override the preset minimum iterations
  --duration-seconds N               Override the preset wall-clock duration
  --dry-run                          Print the resolved test command only
  -h, --help                         Show this help

Profiles:
  smoke    2 iterations, no minimum duration (developer/runner smoke)
  pr       24 iterations and at least 5 minutes (required CPU CI gate)
  nightly  128 iterations and at least 2 hours
  release  512 iterations and at least 8 hours (external release runner)

The test validates internal physical-arena accounting, ownership, transaction,
prefix, generation, and unload invariants. It does not claim to measure process
RSS or device VRAM.
EOF
}

profile=smoke
iterations=
duration_seconds=
dry_run=0

while (($#)); do
  case "$1" in
    --profile)
      profile=${2:-}
      shift 2
      ;;
    --iterations)
      iterations=${2:-}
      shift 2
      ;;
    --duration-seconds)
      duration_seconds=${2:-}
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

case "$profile" in
  smoke)
    : "${iterations:=2}"
    : "${duration_seconds:=0}"
    ;;
  pr)
    : "${iterations:=24}"
    : "${duration_seconds:=300}"
    ;;
  nightly)
    : "${iterations:=128}"
    : "${duration_seconds:=7200}"
    ;;
  release)
    : "${iterations:=512}"
    : "${duration_seconds:=28800}"
    ;;
  *)
    echo "error: --profile must be smoke, pr, nightly, or release" >&2
    exit 2
    ;;
esac

case "$iterations" in
  ''|*[!0-9]*|0) echo "error: --iterations must be positive" >&2; exit 2 ;;
esac
case "$duration_seconds" in
  ''|*[!0-9]*) echo "error: --duration-seconds must be non-negative" >&2; exit 2 ;;
esac

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

command=(cargo test --locked -p izwi-core --lib
  engine::cache::managed::stress_tests::managed_kv_lifecycle_soak --
  --ignored --exact --nocapture)

echo "managed KV lifecycle soak: profile=${profile} iterations=${iterations} duration_seconds=${duration_seconds}"
if ((dry_run)); then
  printf 'IZWI_KV_SOAK_ITERATIONS=%q IZWI_KV_SOAK_DURATION_SECONDS=%q' \
    "$iterations" "$duration_seconds"
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi

IZWI_KV_SOAK_ITERATIONS="$iterations" \
IZWI_KV_SOAK_DURATION_SECONDS="$duration_seconds" \
  "${command[@]}"
