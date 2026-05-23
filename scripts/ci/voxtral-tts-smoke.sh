#!/usr/bin/env bash
set -euo pipefail

# Voxtral TTS server smoke verification against a running izwi server.
#
# Expected setup:
#   1. Start izwi-server with the desired backend.
#   2. Ensure mistralai/Voxtral-4B-TTS-2603 is downloaded.
#   3. Run this script.
#
# Environment:
#   IZWI_VOXTRAL_TTS_SMOKE_URL       Default: http://127.0.0.1:3000
#   IZWI_VOXTRAL_TTS_SMOKE_MODEL     Default: Voxtral-4B-TTS-2603
#   IZWI_VOXTRAL_TTS_SMOKE_VOICE     Default: casual_male
#   IZWI_VOXTRAL_TTS_SMOKE_TEXT      Default: short deterministic phrase
#   IZWI_VOXTRAL_TTS_SMOKE_OUT_DIR   Default: target/voxtral-tts-smoke

BASE_URL="${IZWI_VOXTRAL_TTS_SMOKE_URL:-http://127.0.0.1:3000}"
MODEL="${IZWI_VOXTRAL_TTS_SMOKE_MODEL:-Voxtral-4B-TTS-2603}"
VOICE="${IZWI_VOXTRAL_TTS_SMOKE_VOICE:-casual_male}"
TEXT="${IZWI_VOXTRAL_TTS_SMOKE_TEXT:-Voxtral TTS smoke test.}"
OUT_DIR="${IZWI_VOXTRAL_TTS_SMOKE_OUT_DIR:-target/voxtral-tts-smoke}"
OUT_FILE="${OUT_DIR}/voxtral-tts.wav"

mkdir -p "${OUT_DIR}"

echo "Loading ${MODEL} via ${BASE_URL}/v1/admin/models/${MODEL}/load"
curl -fsS -X POST "${BASE_URL}/v1/admin/models/${MODEL}/load" >/dev/null

echo "Generating Voxtral TTS sample with voice ${VOICE}"
curl -fsS \
  -H "content-type: application/json" \
  -X POST "${BASE_URL}/v1/audio/speech" \
  -d "{\"model\":\"${MODEL}\",\"input\":\"${TEXT}\",\"voice\":\"${VOICE}\",\"response_format\":\"wav\",\"stream\":false}" \
  -o "${OUT_FILE}"

if [[ ! -s "${OUT_FILE}" ]]; then
  echo "Voxtral TTS smoke failed: output file is empty: ${OUT_FILE}" >&2
  exit 1
fi

riff_header="$(head -c 4 "${OUT_FILE}" | od -An -tx1 | tr -d ' \n')"
if [[ "${riff_header}" != "52494646" ]]; then
  echo "Voxtral TTS smoke failed: expected WAV RIFF header in ${OUT_FILE}" >&2
  exit 1
fi

echo "Voxtral TTS smoke verification completed: ${OUT_FILE}"
