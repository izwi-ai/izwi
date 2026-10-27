#!/usr/bin/env bash
# Live TTS matrix: one isolated izwi-server per model, fixed text cases, a
# process footprint guard, and a scored JSON summary.
#
# Each model gets a fresh server with its own database, media and spool
# directories. The harness refuses a busy port and checks that its own server
# owns it (a stale run on the same port once cross-contaminated results), waits
# for free memory before each model on macOS, and kills the server above
# --guard-mb so an unbounded request cannot swap the host (the 2026-10-12 TTS
# audit saw end-of-request vocoder spikes past 14 GB on a 16 GB Mac).

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/bench/run-tts-matrix.sh --backend <cpu|metal|cuda> --models <Model:family,...> [options]

Families: kokoro, lfm, qwen_cv, qwen_base, qwen_greedy, vibevoice, voxtral, fish, chatterbox
(see scripts/bench/score-tts-matrix.py for each family's cases).

Options:
  --only <cases>      Comma-separated case ids to run (default: every case)
  --guard-mb <n>      Kill the server above this footprint (default: 12000)
  --min-free-pct <n>  macOS: wait up to 5 min for this much free memory (default: 50)
  --port <n>          Server port (default: 18190)
  --server <path>     izwi-server binary (default: target/release/izwi-server)
  --models-dir <dir>  Model directory (default: IZWI_MODELS_DIR or the platform default)
  --asr-base <url>    Score intelligibility through this server's ASR (optional)
  --asr-model <name>  ASR model for the round trip (default: Nemotron-3.5-ASR-Streaming-0.6B)
  --out <dir>         Output directory (default: target/tts-matrix/<backend>-<timestamp>)
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND=""
MODELS=""
ONLY=""
GUARD_MB=12000
MIN_FREE_PCT=50
PORT=18190
SERVER="${ROOT}/target/release/izwi-server"
MODELS_DIR="${IZWI_MODELS_DIR:-}"
ASR_BASE=""
ASR_MODEL="Nemotron-3.5-ASR-Streaming-0.6B"
OUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) BACKEND="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        --guard-mb) GUARD_MB="$2"; shift 2 ;;
        --min-free-pct) MIN_FREE_PCT="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --server) SERVER="$2"; shift 2 ;;
        --models-dir) MODELS_DIR="$2"; shift 2 ;;
        --asr-base) ASR_BASE="$2"; shift 2 ;;
        --asr-model) ASR_MODEL="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "${BACKEND}" || -z "${MODELS}" ]]; then
    usage >&2
    exit 2
fi
if [[ ! -x "${SERVER}" ]]; then
    echo "izwi-server binary not found at ${SERVER}; build it or pass --server" >&2
    exit 1
fi
if [[ -z "${MODELS_DIR}" && "$(uname -s)" == "Darwin" ]]; then
    MODELS_DIR="${HOME}/Library/Application Support/izwi/models"
fi
OUT="${OUT:-${ROOT}/target/tts-matrix/${BACKEND}-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${OUT}"
BASE="http://127.0.0.1:${PORT}"
CLIENT="${ROOT}/scripts/bench/score-tts-matrix.py"

# Process footprint in MiB: macOS `footprint` counts unified GPU memory;
# elsewhere fall back to resident set size.
footprint_mib() {
    local pid="$1"
    if command -v footprint >/dev/null 2>&1; then
        footprint "${pid}" 2>/dev/null \
            | sed -n 's/.*Footprint: \([0-9.]*\) \([KMG]B\).*/\1 \2/p' | head -1 \
            | awk '{v=$1; if($2=="KB")v/=1024; if($2=="GB")v*=1024; printf "%d\n", v}'
    else
        ps -o rss= -p "${pid}" 2>/dev/null | awk '{printf "%d\n", $1/1024}'
    fi
}

swap_mib() {
    if [[ "$(uname -s)" == "Darwin" ]]; then
        sysctl -n vm.swapusage | sed -n 's/.*used = \([0-9.]*\)M.*/\1/p' | awk '{printf "%d\n",$1}'
    else
        awk '/SwapTotal/ {t=$2} /SwapFree/ {f=$2} END {printf "%d\n", (t-f)/1024}' /proc/meminfo
    fi
}

# Samples the footprint once a second and kills the server above the guard.
# Readings above 1 TiB are sampler glitches and are recorded but ignored.
guard_server() {
    local pid="$1" log="$2" mib
    while kill -0 "${pid}" 2>/dev/null; do
        mib="$(footprint_mib "${pid}" || true)"
        if [[ -n "${mib}" ]]; then
            if (( mib > 1048576 )); then
                echo "$(date +%s) bogus ${mib}" >> "${log}"
            else
                echo "$(date +%s) ${mib} swap=$(swap_mib)" >> "${log}"
                if (( mib > GUARD_MB )); then
                    echo "$(date +%s) guard ${mib}" >> "${log}"
                    kill -9 "${pid}" 2>/dev/null || true
                    return
                fi
            fi
        fi
        sleep 1
    done
}

port_owner() {
    lsof -nP -t -iTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null | head -1
}

now() { python3 -c 'import time; print(time.time())'; }

IFS=',' read -ra MODEL_LIST <<< "${MODELS}"
for item in "${MODEL_LIST[@]}"; do
    model="${item%%:*}"
    family="${item##*:}"
    dir="${OUT}/${model}"
    state="${dir}/state"
    mkdir -p "${state}/media" "${state}/spool"
    memory_log="${dir}/footprint.log"
    : > "${memory_log}"

    free_pct=""
    if command -v memory_pressure >/dev/null 2>&1; then
        for _ in $(seq 1 60); do
            free_pct="$(memory_pressure | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p')"
            (( ${free_pct:-0} >= MIN_FREE_PCT )) && break
            sleep 5
        done
    fi
    if [[ -n "$(port_owner)" ]]; then
        echo "port ${PORT} already in use; refusing to run ${model}" >&2
        exit 3
    fi

    IZWI_BACKEND="${BACKEND}" IZWI_HOST=127.0.0.1 IZWI_PORT="${PORT}" \
        IZWI_MODELS_DIR="${MODELS_DIR}" IZWI_DB_PATH="${state}/izwi.sqlite3" \
        IZWI_MEDIA_DIR="${state}/media" IZWI_SPEECH_SPOOL_DIR="${state}/spool" \
        RUST_LOG="${RUST_LOG:-info}" "${SERVER}" > "${dir}/server.log" 2>&1 &
    server_pid=$!
    for _ in $(seq 1 120); do
        curl -sf "${BASE}/v1/health" >/dev/null 2>&1 && break
        sleep 1
    done
    owner="$(port_owner)"
    if [[ "${owner}" != "${server_pid}" ]]; then
        echo "port ${PORT} owned by pid ${owner:-none}, not our server ${server_pid}; aborting" >&2
        kill "${server_pid}" 2>/dev/null || true
        exit 3
    fi
    guard_server "${server_pid}" "${memory_log}" &
    guard_pid=$!
    echo "== ${model} (${family}) backend=${BACKEND} free=${free_pct:-?}% swap=$(swap_mib)M"

    started="$(now)"
    load_status="$(curl -s -o "${dir}/load.json" -w '%{http_code}' -X POST --max-time 1200 \
        "${BASE}/v1/admin/models/${model}/load" || echo 000)"
    load_secs="$(python3 -c "print(round($(now) - ${started}, 2))")"
    sleep 2
    after_load="$(footprint_mib "${server_pid}" || echo 0)"
    printf '{"model":"%s","family":"%s","backend":"%s","load_http":"%s","load_s":%s,"footprint_after_load_mib":%s}\n' \
        "${model}" "${family}" "${BACKEND}" "${load_status}" "${load_secs}" "${after_load:-0}" \
        | tee "${dir}/load_summary.json"
    curl -s --max-time 10 "${BASE}/v1/metrics" > "${dir}/metrics_after_load.json" 2>/dev/null || true
    if [[ "${load_status}" == "200" ]]; then
        PYTHONDONTWRITEBYTECODE=1 python3 "${CLIENT}" request --base "${BASE}" --model "${model}" \
            --family "${family}" --out "${dir}/cases" --only "${ONLY}" 2>&1 | tee "${dir}/client.log"
    fi
    curl -s --max-time 10 "${BASE}/v1/metrics" > "${dir}/metrics_after_cases.json" 2>/dev/null || true
    curl -s -o /dev/null -X POST --max-time 300 "${BASE}/v1/admin/models/${model}/unload" || true

    kill "${server_pid}" 2>/dev/null || true
    sleep 3
    kill -9 "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    wait "${guard_pid}" 2>/dev/null || true
    peak="$(awk '$2 ~ /^[0-9]+$/ {if ($2 > p) p = $2} END {print p + 0}' "${memory_log}")"
    guarded="$(grep -c ' guard ' "${memory_log}" || true)"
    printf '{"model":"%s","peak_mib":%s,"guard_kills":%s}\n' "${model}" "${peak}" "${guarded}" \
        | tee "${dir}/peak.json"
    sleep 5
done

score_args=(score --runs "${OUT}" --out "${OUT}/summary.json")
if [[ -n "${ASR_BASE}" ]]; then
    score_args+=(--asr-base "${ASR_BASE}" --asr-model "${ASR_MODEL}")
fi
PYTHONDONTWRITEBYTECODE=1 python3 "${CLIENT}" "${score_args[@]}" | tee "${OUT}/summary.txt"
echo "results: ${OUT}"
