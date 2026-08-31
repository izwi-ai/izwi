#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
server_url="${IZWI_PARITY_SERVER_URL:-http://127.0.0.1:8080}"
asr_model="${IZWI_PARITY_ASR_MODEL:-Qwen3-ASR-0.6B-GGUF}"
tts_model="${IZWI_PARITY_TTS_MODEL:-Kokoro-82M}"
output_root="${IZWI_PARITY_OUTPUT_DIR:-${repo_root}/target/audio-api-parity}"
fixture_audio="${repo_root}/data/fox.wav"
fixture_text="${repo_root}/data/fox.md"
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
output_dir="${output_root}/${run_id}"
summary_file="${output_dir}/summary.json"
current_stage="initialization"
current_model=""

mkdir -p "${output_dir}"

finish() {
  exit_code=$?
  trap - EXIT INT TERM
  if [[ -n "${current_model}" ]]; then
    curl -fsS -X POST "${server_url}/v1/admin/models/${current_model}/unload" >/dev/null 2>&1 || true
  fi
  jq -n \
    --arg status "$([[ ${exit_code} -eq 0 ]] && printf passed || printf failed)" \
    --arg stage "${current_stage}" \
    --arg server_url "${server_url}" \
    --arg asr_model "${asr_model}" \
    --arg tts_model "${tts_model}" \
    --arg fixture_audio "${fixture_audio}" \
    --arg fixture_text "${fixture_text}" \
    --argjson exit_code "${exit_code}" \
    '{schema_version:1,status:$status,stage:$stage,exit_code:$exit_code,server_url:$server_url,asr_model:$asr_model,tts_model:$tts_model,fixture_audio:$fixture_audio,fixture_text:$fixture_text}' \
    >"${summary_file}"
  printf 'Audio API parity artifact: %s\n' "${summary_file}"
  exit "${exit_code}"
}
trap finish EXIT INT TERM

for dependency in curl jq base64 od; do
  command -v "${dependency}" >/dev/null
done
test -s "${fixture_audio}"
test -s "${fixture_text}"

normalize_text() {
  tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:][:space:]' | tr -s '[:space:]' ' ' | sed 's/^ //; s/ $//'
}

assert_wav() {
  wav_path=$1
  test "$(wc -c <"${wav_path}" | tr -d ' ')" -gt 44
  test "$(od -An -N4 -c "${wav_path}" | tr -d '[:space:]')" = "RIFF"
}

poll_record() {
  record_url=$1
  output_path=$2
  attempt=0
  while (( attempt < 900 )); do
    curl -fsS "${record_url}" >"${output_path}"
    status="$(jq -r '.processing_status // .status // empty' "${output_path}")"
    case "${status}" in
      completed) return 0 ;;
      failed|cancelled|expired)
        jq . "${output_path}" >&2
        return 1
        ;;
    esac
    attempt=$((attempt + 1))
    sleep 1
  done
  printf 'Timed out polling %s\n' "${record_url}" >&2
  return 1
}

current_stage="server readiness"
curl -fsS "${server_url}/v1/health" >"${output_dir}/health.json"

current_stage="ASR model load"
curl -fsS -X POST "${server_url}/v1/admin/models/${asr_model}/load" >"${output_dir}/asr-load.json"
current_model="${asr_model}"

current_stage="OpenAI ASR"
curl -fsS \
  -F "file=@${fixture_audio};type=audio/wav" \
  -F "model=${asr_model}" \
  -F "response_format=json" \
  "${server_url}/v1/audio/transcriptions" >"${output_dir}/asr-openai.json"

current_stage="product ASR"
audio_base64="$(base64 <"${fixture_audio}" | tr -d '\r\n')"
jq -n \
  --arg audio_base64 "${audio_base64}" \
  --arg model_id "${asr_model}" \
  '{audio_base64:$audio_base64,model_id:$model_id,language:"English",stream:false,include_timestamps:false,generate_summary:false}' \
  >"${output_dir}/asr-product-request.json"
curl -fsS \
  -H 'content-type: application/json' \
  --data-binary "@${output_dir}/asr-product-request.json" \
  "${server_url}/v1/speech-to-text/jobs?job_kind=transcription" \
  >"${output_dir}/asr-product-created.json"
asr_record_id="$(jq -er '.id' "${output_dir}/asr-product-created.json")"
poll_record \
  "${server_url}/v1/speech-to-text/jobs/${asr_record_id}?job_kind=transcription" \
  "${output_dir}/asr-product.json"

current_stage="ASR result comparison"
expected_text="$(normalize_text <"${fixture_text}")"
openai_text="$(jq -r '.text // .transcription // empty' "${output_dir}/asr-openai.json" | normalize_text)"
product_text="$(jq -r '.transcription // .text // empty' "${output_dir}/asr-product.json" | normalize_text)"
test -n "${openai_text}"
test "${openai_text}" = "${product_text}"
test "${openai_text}" = "${expected_text}"

current_stage="ASR model unload"
curl -fsS -X POST "${server_url}/v1/admin/models/${asr_model}/unload" >"${output_dir}/asr-unload.json"
current_model=""

current_stage="TTS model load"
curl -fsS -X POST "${server_url}/v1/admin/models/${tts_model}/load" >"${output_dir}/tts-load.json"
current_model="${tts_model}"
tts_text="$(tr '\n' ' ' <"${fixture_text}" | sed 's/[[:space:]]*$//')"

current_stage="OpenAI TTS"
jq -n \
  --arg model "${tts_model}" \
  --arg input "${tts_text}" \
  '{model:$model,input:$input,response_format:"wav",max_output_tokens:256,stream:false}' \
  >"${output_dir}/tts-openai-request.json"
curl -fsS \
  -H 'content-type: application/json' \
  --data-binary "@${output_dir}/tts-openai-request.json" \
  "${server_url}/v1/audio/speech" >"${output_dir}/tts-openai.wav"
assert_wav "${output_dir}/tts-openai.wav"

current_stage="product TTS"
jq -n \
  --arg model_id "${tts_model}" \
  --arg text "${tts_text}" \
  '{model_id:$model_id,text:$text,max_output_tokens:256,stream:false}' \
  >"${output_dir}/tts-product-request.json"
curl -fsS \
  -H 'content-type: application/json' \
  --data-binary "@${output_dir}/tts-product-request.json" \
  "${server_url}/v1/text-to-speech" >"${output_dir}/tts-product-created.json"
tts_record_id="$(jq -er '.id' "${output_dir}/tts-product-created.json")"
poll_record "${server_url}/v1/text-to-speech/${tts_record_id}" "${output_dir}/tts-product.json"
curl -fsS "${server_url}/v1/text-to-speech/${tts_record_id}/audio" >"${output_dir}/tts-product.wav"
assert_wav "${output_dir}/tts-product.wav"

current_stage="TTS model unload"
curl -fsS -X POST "${server_url}/v1/admin/models/${tts_model}/unload" >"${output_dir}/tts-unload.json"
current_model=""

current_stage="complete"
