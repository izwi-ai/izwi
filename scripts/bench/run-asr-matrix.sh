#!/usr/bin/env bash
# Live ASR matrix: one isolated izwi-server per model, a fixed fixture set,
# a process footprint guard, and a scored JSON summary.
#
# Each model gets a fresh server with its own database, media and spool
# directories, so a failed or memory-heavy model cannot contaminate the next
# one. The guard kills the server when its footprint passes --guard-mb, which
# keeps an unbounded request from swapping the host (see the 2026-10-12 ASR
# audit: a 600 s Parakeet request reached a 24 GB footprint on a 16 GB Mac).

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/bench/run-asr-matrix.sh --backend <cpu|metal|cuda> --models <A,B,...> [options]

Options:
  --fixtures <list>   Comma-separated fixture names (default: fox,test,silence,noise)
                      fox      data/fox.wav, 3.6 s, reference data/fox.md
                      test     data/test.wav, 29.7 s, reference data/test.md
                      diar116  data/diarization-2.mp3, 116 s stereo 44.1 kHz
                      diar60   first 60 s of data/diarization-2.mp3 (needs ffmpeg)
                      long600  first 600 s of data/ceo.mp3 (needs ffmpeg)
                      silence  10 s of digital silence, empty transcript expected (needs sox)
                      noise    10 s of pink noise, empty transcript expected (needs sox)
  --form <k=v>        Extra multipart field for every request (repeatable),
                      for example --form stream=true or --form max_tokens=256
  --language <code>   Language field (default: en; "none" omits it)
  --guard-mb <n>      Kill the server above this footprint (default: 12000)
  --port <n>          Server port (default: 18090)
  --server <path>     izwi-server binary (default: target/release/izwi-server)
  --models-dir <dir>  Model directory (default: IZWI_MODELS_DIR or the platform default)
  --out <dir>         Output directory (default: target/asr-matrix/<backend>-<timestamp>)
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND=""
MODELS=""
FIXTURES="fox,test,silence,noise"
LANGUAGE="en"
GUARD_MB=12000
PORT=18090
SERVER="${ROOT}/target/release/izwi-server"
MODELS_DIR="${IZWI_MODELS_DIR:-}"
OUT=""
FORM=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) BACKEND="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        --fixtures) FIXTURES="$2"; shift 2 ;;
        --form) FORM+=(-F "$2"); shift 2 ;;
        --language) LANGUAGE="$2"; shift 2 ;;
        --guard-mb) GUARD_MB="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --server) SERVER="$2"; shift 2 ;;
        --models-dir) MODELS_DIR="$2"; shift 2 ;;
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
OUT="${OUT:-${ROOT}/target/asr-matrix/${BACKEND}-$(date +%Y%m%d-%H%M%S)}"
FIXTURE_DIR="${OUT}/fixtures"
mkdir -p "${FIXTURE_DIR}"
BASE="http://127.0.0.1:${PORT}"

# Resolve one fixture name to a file, generating derived fixtures on demand.
fixture_path() {
    local name="$1"
    case "${name}" in
        fox) echo "${ROOT}/data/fox.wav" ;;
        test) echo "${ROOT}/data/test.wav" ;;
        diar116) echo "${ROOT}/data/diarization-2.mp3" ;;
        diar60)
            [[ -f "${FIXTURE_DIR}/diar60.wav" ]] || ffmpeg -nostdin -loglevel error -y \
                -i "${ROOT}/data/diarization-2.mp3" -t 60 "${FIXTURE_DIR}/diar60.wav"
            echo "${FIXTURE_DIR}/diar60.wav" ;;
        long600)
            [[ -f "${FIXTURE_DIR}/long600.mp3" ]] || ffmpeg -nostdin -loglevel error -y \
                -i "${ROOT}/data/ceo.mp3" -t 600 -c copy "${FIXTURE_DIR}/long600.mp3"
            echo "${FIXTURE_DIR}/long600.mp3" ;;
        silence)
            [[ -f "${FIXTURE_DIR}/silence.wav" ]] || sox -n -r 16000 -c 1 -b 16 \
                "${FIXTURE_DIR}/silence.wav" trim 0 10
            echo "${FIXTURE_DIR}/silence.wav" ;;
        noise)
            [[ -f "${FIXTURE_DIR}/noise.wav" ]] || sox -n -r 16000 -c 1 -b 16 \
                "${FIXTURE_DIR}/noise.wav" synth 10 pinknoise vol 0.05
            echo "${FIXTURE_DIR}/noise.wav" ;;
        *) echo "Unknown fixture: ${name}" >&2; return 1 ;;
    esac
}

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
                echo "$(date +%s) ${mib}" >> "${log}"
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

now() { python3 -c 'import time; print(time.time())'; }

IFS=',' read -ra MODEL_LIST <<< "${MODELS}"
IFS=',' read -ra FIXTURE_LIST <<< "${FIXTURES}"
RESULTS="${OUT}/results.jsonl"
: > "${RESULTS}"

for model in "${MODEL_LIST[@]}"; do
    state="${OUT}/${model}/state"
    mkdir -p "${state}/media" "${state}/spool"
    server_log="${OUT}/${model}/server.log"
    memory_log="${OUT}/${model}/footprint.log"
    : > "${memory_log}"

    IZWI_BACKEND="${BACKEND}" IZWI_HOST=127.0.0.1 IZWI_PORT="${PORT}" \
        IZWI_MODELS_DIR="${MODELS_DIR}" IZWI_DB_PATH="${state}/izwi.sqlite3" \
        IZWI_MEDIA_DIR="${state}/media" IZWI_SPEECH_SPOOL_DIR="${state}/spool" \
        RUST_LOG="${RUST_LOG:-info}" "${SERVER}" > "${server_log}" 2>&1 &
    server_pid=$!
    for _ in $(seq 1 120); do
        curl -sf "${BASE}/health" >/dev/null 2>&1 && break
        sleep 1
    done
    guard_server "${server_pid}" "${memory_log}" &
    guard_pid=$!

    started="$(now)"
    load_status="$(curl -s -o "${OUT}/${model}/load.json" -w '%{http_code}' -X POST \
        --max-time 900 "${BASE}/v1/admin/models/${model}/load" || echo 000)"
    load_secs="$(python3 -c "print(round($(now) - ${started}, 3))")"
    printf '{"kind":"load","model":"%s","http":"%s","secs":%s,"body":"%s"}\n' \
        "${model}" "${load_status}" "${load_secs}" "${OUT}/${model}/load.json" >> "${RESULTS}"

    for fixture in "${FIXTURE_LIST[@]}"; do
        file="$(fixture_path "${fixture}")"
        response="${OUT}/${model}/${fixture}.json"
        language_field=()
        if [[ "${LANGUAGE}" != "none" && "${fixture}" != "silence" && "${fixture}" != "noise" ]]; then
            language_field=(-F "language=${LANGUAGE}")
        fi
        started="$(now)"
        status="$(curl -s -o "${response}" -w '%{http_code}' --max-time 3600 \
            "${BASE}/v1/audio/transcriptions" -F "file=@${file}" -F "model=${model}" \
            -F response_format=verbose_json "${language_field[@]+"${language_field[@]}"}" \
            "${FORM[@]+"${FORM[@]}"}" || echo 000)"
        secs="$(python3 -c "print(round($(now) - ${started}, 3))")"
        printf '{"kind":"transcription","model":"%s","fixture":"%s","http":"%s","secs":%s,"body":"%s"}\n' \
            "${model}" "${fixture}" "${status}" "${secs}" "${response}" >> "${RESULTS}"
        echo "${model} ${fixture} http=${status} secs=${secs}"
    done

    curl -s -o /dev/null -X POST --max-time 300 "${BASE}/v1/admin/models/${model}/unload" || true
    kill "${server_pid}" 2>/dev/null || true
    sleep 3
    kill -9 "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    wait "${guard_pid}" 2>/dev/null || true
    peak="$(awk '$2 ~ /^[0-9]+$/ {if ($2 > p) p = $2} END {print p + 0}' "${memory_log}")"
    guarded="$(grep -c ' guard ' "${memory_log}" || true)"
    printf '{"kind":"footprint","model":"%s","peak_mib":%s,"guard_kills":%s}\n' \
        "${model}" "${peak}" "${guarded}" >> "${RESULTS}"
done

python3 "${ROOT}/scripts/bench/score-asr-matrix.py" --root "${ROOT}" "${RESULTS}" \
    | tee "${OUT}/summary.txt"
echo "results: ${OUT}"
