#!/usr/bin/env bash
# Pack one backend's results into nemotron-transcribe-report-<backend>-<sha>.tar.gz
# for sending back: report.json (build, parity, e2e, load, wer sections),
# build-info.json, parity and load outputs, wer JSON, the admission and
# memory-ledger snapshots (/v1/metrics coordinator, /v1/health,
# /v1/admin/models/<id>) and the tail of every server log.
#
# Usage: collect.sh <cpu|metal|cuda> [--base-url URL]
#   --base-url  also snapshot a server that is still running
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BACKEND="${1:-}"
nt_check_backend "$BACKEND"
shift
BASE_URL=""
while (($#)); do
    case "$1" in
        --base-url) BASE_URL="$2"; shift 2 ;;
        *) nt_die "unknown argument $1" ;;
    esac
done
OUT="$(nt_backend_out "$BACKEND")"
[[ -d "$OUT" ]] || nt_die "nothing to collect: $OUT does not exist"
PY="$(nt_python)"

if [[ -n "$BASE_URL" ]]; then
    "$PY" "$NT_DIR/nt_harness.py" snapshot --base-url "$BASE_URL" --out "$OUT" --label collect
fi
if [[ -f "$OUT/build-info.json" ]]; then
    "$PY" "$NT_DIR/nt_harness.py" merge --report "$OUT/report.json" --section build \
        --file "$OUT/build-info.json"
fi

SHA="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("git_sha","unknown"))' \
    "$OUT/build-info.json" 2>/dev/null || echo unknown)"
SHORT="${SHA:0:12}"
STAGE="$(mktemp -d)"
NAME="nemotron-transcribe-report-$BACKEND-$SHORT"
mkdir -p "$STAGE/$NAME"
trap 'rm -rf "$STAGE"' EXIT

for file in report.json build-info.json parity.json parity-results.txt; do
    [[ -f "$OUT/$file" ]] && cp "$OUT/$file" "$STAGE/$NAME/"
done
for file in "$OUT"/load-*.json "$OUT"/wer*.json "$OUT"/parity-*.log; do
    [[ -f "$file" ]] && cp "$file" "$STAGE/$NAME/"
done
[[ -d "$OUT/snapshots" ]] && cp -R "$OUT/snapshots" "$STAGE/$NAME/"
mkdir -p "$STAGE/$NAME/logs"
for log in "$OUT"/server-*.log "$OUT"/build.log; do
    [[ -f "$log" ]] || continue
    tail -n "${NT_COLLECT_LOG_LINES:-20000}" "$log" >"$STAGE/$NAME/logs/$(basename "$log")"
done
{
    echo "collected_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "backend: $BACKEND"
    echo "git_sha: $SHA"
    echo "host: $(uname -a)"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
    fi
    if [[ "$(uname -s)" == "Darwin" ]]; then
        sysctl -n machdep.cpu.brand_string hw.memsize 2>/dev/null || true
    else
        grep -m1 'model name' /proc/cpuinfo 2>/dev/null || true
        grep MemTotal /proc/meminfo 2>/dev/null || true
    fi
} >"$STAGE/$NAME/host.txt"

TARBALL="$OUT/$NAME.tar.gz"
tar -czf "$TARBALL" -C "$STAGE" "$NAME"
nt_log "wrote $TARBALL ($(nt_file_size "$TARBALL") bytes)"
"$PY" - "$OUT/report.json" <<'PY' || true
import json, sys
report = json.load(open(sys.argv[1]))
for name, section in sorted((report.get("sections") or {}).items()):
    if not isinstance(section, dict):
        continue
    summary = section.get("summary") or section.get("status") or ""
    print(f"  {name:12} {json.dumps(summary)[:120]}")
PY
