#!/usr/bin/env bash
# Pack one backend's results into fish-drama3-report-<backend>-<sha>.tar.gz
# for sending back: report.json (build, parity, e2e, load, eval sections),
# build-info.json, the parity, load and eval outputs, the admission and
# memory-ledger snapshots (/v1/metrics with the coordinator and the model's
# path counters, /v1/health, /v1/metrics/batch-runtime,
# /v1/admin/models/FishAudio-Drama-3), the tail of every server log and the
# WAV samples (every take of the E2E run; eval WAVs only with --eval-wavs).
#
# Usage: collect.sh <cpu|metal|cuda> [--synthetic] [--base-url URL] [--eval-wavs] [--dry-run]
#   --base-url   also snapshot a server that is still running
#   --eval-wavs  include eval-mini's generated WAVs (several hundred MB)
#   --dry-run    list what would be packed
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BACKEND="${1:-}"
if [[ "$BACKEND" == "-h" || "$BACKEND" == "--help" ]]; then
    sed -n '2,13p' "$0"
    exit 0
fi
fd3_check_backend "$BACKEND"
shift
SYNTHETIC=0
BASE_URL=""
EVAL_WAVS=0
DRY_RUN=0
while (($#)); do
    case "$1" in
        --synthetic) SYNTHETIC=1 ;;
        --base-url) BASE_URL="$2"; shift ;;
        --eval-wavs) EVAL_WAVS=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h | --help) sed -n '2,13p' "$0"; exit 0 ;;
        *) fd3_die "unknown argument $1" ;;
    esac
    shift
done
OUT="$(fd3_backend_out "$BACKEND" "$SYNTHETIC")"
[[ -d "$OUT" ]] || fd3_die "nothing to collect: $OUT does not exist"
PY="$(fd3_python)"

# What goes in, relative to $OUT: files, then directories.
FILES=(report.json build-info.json parity.json parity-results.txt)
for file in "$OUT"/load-*.json "$OUT"/eval-*.json "$OUT"/parity-*.log; do
    [[ -f "$file" ]] && FILES+=("$(basename "$file")")
done
DIRS=(snapshots audio)
if ((EVAL_WAVS)) && [[ -d "$OUT/eval-wavs" ]]; then DIRS+=(eval-wavs); fi

if ((DRY_RUN)); then
    fd3_log "dry run: would pack from $OUT"
    for file in "${FILES[@]}"; do [[ -f "$OUT/$file" ]] && fd3_log "  $file"; done
    for dir in "${DIRS[@]}"; do [[ -d "$OUT/$dir" ]] && fd3_log "  $dir/ ($(du -sh "$OUT/$dir" | cut -f1))"; done
    for log in "$OUT"/server-*.log "$OUT"/build.log "$OUT"/synthetic-writer.log; do
        [[ -f "$log" ]] && fd3_log "  logs/$(basename "$log") (last ${FD3_COLLECT_LOG_LINES:-20000} lines)"
    done
    exit 0
fi

if [[ -n "$BASE_URL" ]]; then
    "$PY" "$FD3_DIR/fd3_harness.py" snapshot --base-url "$BASE_URL" --out "$OUT" --label collect
fi
if [[ -f "$OUT/build-info.json" ]]; then
    "$PY" "$FD3_DIR/fd3_harness.py" merge --report "$OUT/report.json" --section build \
        --file "$OUT/build-info.json"
fi

SHA="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("git_sha","unknown"))' \
    "$OUT/build-info.json" 2>/dev/null || echo unknown)"
SUFFIX=""
if ((SYNTHETIC)); then SUFFIX="-synthetic"; fi
NAME="fish-drama3-report-$BACKEND$SUFFIX-${SHA:0:12}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/$NAME/logs"

for file in "${FILES[@]}"; do
    if [[ -f "$OUT/$file" ]]; then cp "$OUT/$file" "$STAGE/$NAME/"; fi
done
for dir in "${DIRS[@]}"; do
    if [[ -d "$OUT/$dir" ]]; then cp -R "$OUT/$dir" "$STAGE/$NAME/"; fi
done
for log in "$OUT"/server-*.log "$OUT"/build.log "$OUT"/synthetic-writer.log; do
    [[ -f "$log" ]] || continue
    tail -n "${FD3_COLLECT_LOG_LINES:-20000}" "$log" >"$STAGE/$NAME/logs/$(basename "$log")"
done
{
    echo "collected_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "backend: $BACKEND"
    echo "synthetic: $SYNTHETIC"
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
fd3_log "wrote $TARBALL ($(fd3_file_size "$TARBALL") bytes)"
"$PY" - "$OUT/report.json" <<'PY' || true
import json, sys
report = json.load(open(sys.argv[1]))
for name, section in sorted((report.get("sections") or {}).items()):
    if not isinstance(section, dict):
        continue
    summary = section.get("summary") or section.get("status") or ""
    print(f"  {name:12} {json.dumps(summary)[:120]}")
PY
