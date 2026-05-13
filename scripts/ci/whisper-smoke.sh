#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Whisper CPU/Metal smoke verification against a running izwi server.

Required environment:
  IZWI_WHISPER_SMOKE_BACKEND       Label for this run: cpu or metal
  IZWI_WHISPER_SMOKE_SHORT_AUDIO   Short speech fixture, under 30 seconds
  IZWI_WHISPER_SMOKE_LONG_AUDIO    Long real speech fixture, over 30 seconds
  IZWI_WHISPER_SMOKE_SILENCE_AUDIO Mostly silent or silent fixture
  IZWI_WHISPER_SMOKE_NON_EN_AUDIO  Non-English speech fixture

Optional environment:
  IZWI_WHISPER_SMOKE_NON_EN_LANGUAGE Language hint for the non-English fixture
  IZWI_WHISPER_SMOKE_SERVER          Default: http://127.0.0.1:8080
  IZWI_WHISPER_SMOKE_MODEL           Default: Whisper-Large-v3-Turbo
  IZWI_WHISPER_SMOKE_OUT_DIR         Default: target/whisper-smoke/<backend>
  IZWI_WHISPER_SMOKE_TIMEOUT_SECS    Default: 900

Start the server separately with the backend being verified, for example:
  IZWI_BACKEND=cpu izwi serve --backend cpu
  IZWI_BACKEND=metal izwi serve --backend metal
USAGE
}

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: ${name}" >&2
    usage >&2
    exit 2
  fi
}

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -r "$path" ]]; then
    echo "${label} fixture is not readable: ${path}" >&2
    exit 2
  fi
}

request_transcription() {
  local case_name="$1"
  local audio_path="$2"
  local response_format="$3"
  local language="$4"
  local output_path="$5"
  local stream="$6"

  python3 - "$SERVER" "$MODEL" "$case_name" "$audio_path" "$response_format" "$language" "$output_path" "$stream" "$BACKEND" <<'PY'
import base64
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

server, model, case_name, audio_path, response_format, language, output_path, stream, backend = sys.argv[1:]
timeout = int(os.environ.get("IZWI_WHISPER_SMOKE_TIMEOUT_SECS", "900"))
payload = {
    "model": model,
    "audio_base64": base64.b64encode(Path(audio_path).read_bytes()).decode("ascii"),
    "response_format": response_format,
}
if language:
    payload["language"] = language
if response_format == "verbose_json":
    payload["timestamp_granularities"] = ["word", "segment"]
if stream == "true":
    payload["stream"] = True

request = urllib.request.Request(
    f"{server.rstrip('/')}/v1/audio/transcriptions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
except urllib.error.HTTPError as exc:
    body = exc.read()
    sys.stderr.write(body.decode("utf-8", errors="replace") + "\n")
    raise


def collect_whisper_diagnostics(diagnostics):
    found = []
    if not isinstance(diagnostics, dict):
        return found
    if diagnostics.get("model_family") == "whisper_asr":
        found.append(diagnostics)
    chunking = diagnostics.get("chunking")
    if isinstance(chunking, dict):
        for chunk in chunking.get("chunk_transcriptions", []):
            if not isinstance(chunk, dict):
                continue
            model_diagnostics = chunk.get("model_diagnostics")
            if (
                isinstance(model_diagnostics, dict)
                and model_diagnostics.get("model_family") == "whisper_asr"
            ):
                found.append(model_diagnostics)
    return found


def require_non_negative_number(case_name, label, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise SystemExit(f"{case_name}: expected non-negative numeric {label}, got {value!r}")


def validate_whisper_timings(case_name, idx, node):
    timings = node.get("timings_ms")
    if not isinstance(timings, dict):
        raise SystemExit(f"{case_name}: Whisper diagnostics {idx} missing timings_ms object")
    for key in ("model_total", "decode", "encoder_forward", "mel_prepare"):
        require_non_negative_number(
            case_name,
            f"Whisper diagnostics {idx} timings_ms.{key}",
            timings.get(key),
        )


def validate_whisper_logit_filters(case_name, idx, node, backend):
    filters = node.get("logit_filters")
    if not isinstance(filters, dict):
        raise SystemExit(f"{case_name}: Whisper diagnostics {idx} missing logit_filters object")
    device_greedy_decode = filters.get("device_greedy_decode")
    device_greedy_active = filters.get("device_greedy_active")
    if not isinstance(device_greedy_decode, bool):
        raise SystemExit(
            f"{case_name}: expected boolean logit_filters.device_greedy_decode, got {device_greedy_decode!r}"
        )
    if not isinstance(device_greedy_active, bool):
        raise SystemExit(
            f"{case_name}: expected boolean logit_filters.device_greedy_active, got {device_greedy_active!r}"
        )
    expected_active = backend == "metal" and device_greedy_decode
    if device_greedy_active is not expected_active:
        raise SystemExit(
            f"{case_name}: expected device_greedy_active={expected_active} for {backend}, got {device_greedy_active}"
        )


def validate_long_chunking(case_name, diagnostics):
    if case_name != "long_verbose":
        return
    if not isinstance(diagnostics, dict):
        raise SystemExit(f"{case_name}: missing diagnostics object")
    chunking = diagnostics.get("chunking")
    if not isinstance(chunking, dict):
        raise SystemExit(f"{case_name}: expected long-form chunking diagnostics")
    chunk_count = chunking.get("chunk_count")
    if not isinstance(chunk_count, int) or chunk_count <= 1:
        raise SystemExit(f"{case_name}: expected chunk_count > 1, got {chunk_count!r}")
    chunk_transcriptions = chunking.get("chunk_transcriptions")
    if not isinstance(chunk_transcriptions, list):
        raise SystemExit(f"{case_name}: missing chunk_transcriptions list")
    if len(chunk_transcriptions) != chunk_count:
        raise SystemExit(
            f"{case_name}: expected {chunk_count} chunk_transcriptions, got {len(chunk_transcriptions)}"
        )
    for idx, chunk in enumerate(chunk_transcriptions):
        if not isinstance(chunk, dict):
            raise SystemExit(f"{case_name}: chunk_transcriptions[{idx}] is not an object")
        model_diagnostics = chunk.get("model_diagnostics")
        if not isinstance(model_diagnostics, dict):
            raise SystemExit(f"{case_name}: chunk_transcriptions[{idx}] missing model_diagnostics")
        if model_diagnostics.get("model_family") != "whisper_asr":
            raise SystemExit(
                f"{case_name}: chunk_transcriptions[{idx}] is not Whisper diagnostics"
            )


def validate_cpu_metal_whisper_diagnostics(case_name, diagnostics, backend, response_text):
    validate_long_chunking(case_name, diagnostics)
    nodes = collect_whisper_diagnostics(diagnostics)
    if not nodes and "silence" in case_name and not response_text.strip():
        return
    if not nodes:
        raise SystemExit(f"{case_name}: no Whisper diagnostics found to validate")

    expected_kind = {"cpu": "Cpu", "metal": "Metal"}[backend]
    for idx, node in enumerate(nodes):
        device = node.get("device")
        if not isinstance(device, dict):
            raise SystemExit(f"{case_name}: Whisper diagnostics {idx} missing device object")
        if device.get("kind") != expected_kind:
            raise SystemExit(
                f"{case_name}: expected Whisper device kind {expected_kind}, got {device.get('kind')}"
            )
        if device.get("model_dtype") != "F32":
            raise SystemExit(
                f"{case_name}: expected Whisper model dtype F32, got {device.get('model_dtype')}"
            )
        if device.get("cuda_dtype_shim") is not False:
            raise SystemExit(f"{case_name}: CPU/Metal run reported CUDA dtype shim")
        if device.get("whisper_impl") != "local_whisper":
            raise SystemExit(
                f"{case_name}: expected local_whisper Whisper impl, got {device.get('whisper_impl')}"
            )
        validate_whisper_timings(case_name, idx, node)
        validate_whisper_logit_filters(case_name, idx, node, backend)


Path(output_path).write_bytes(body)
text = body.decode("utf-8", errors="replace")

if stream == "true":
    if "transcript.text.done" not in text:
        raise SystemExit(f"{case_name}: streaming response did not include transcript.text.done")
elif response_format == "verbose_json":
    parsed = json.loads(text)
    if "text" not in parsed:
        raise SystemExit(f"{case_name}: verbose_json response is missing text")
    if "izwi_asr_diagnostics" not in parsed:
        raise SystemExit(f"{case_name}: verbose_json response is missing izwi_asr_diagnostics")
    validate_cpu_metal_whisper_diagnostics(
        case_name,
        parsed.get("izwi_asr_diagnostics"),
        backend,
        str(parsed.get("text") or ""),
    )
else:
    if not text.strip():
        raise SystemExit(f"{case_name}: {response_format} response was empty")
PY
}

require_var IZWI_WHISPER_SMOKE_BACKEND
require_var IZWI_WHISPER_SMOKE_SHORT_AUDIO
require_var IZWI_WHISPER_SMOKE_LONG_AUDIO
require_var IZWI_WHISPER_SMOKE_SILENCE_AUDIO
require_var IZWI_WHISPER_SMOKE_NON_EN_AUDIO

BACKEND="$IZWI_WHISPER_SMOKE_BACKEND"
SERVER="${IZWI_WHISPER_SMOKE_SERVER:-http://127.0.0.1:8080}"
MODEL="${IZWI_WHISPER_SMOKE_MODEL:-Whisper-Large-v3-Turbo}"
OUT_DIR="${IZWI_WHISPER_SMOKE_OUT_DIR:-target/whisper-smoke/${BACKEND}}"
NON_EN_LANGUAGE="${IZWI_WHISPER_SMOKE_NON_EN_LANGUAGE:-}"

case "$BACKEND" in
  cpu|metal) ;;
  *)
    echo "IZWI_WHISPER_SMOKE_BACKEND must be cpu or metal, got: ${BACKEND}" >&2
    exit 2
    ;;
esac

require_file short "$IZWI_WHISPER_SMOKE_SHORT_AUDIO"
require_file long "$IZWI_WHISPER_SMOKE_LONG_AUDIO"
require_file silence "$IZWI_WHISPER_SMOKE_SILENCE_AUDIO"
require_file non_en "$IZWI_WHISPER_SMOKE_NON_EN_AUDIO"

mkdir -p "$OUT_DIR"

SUMMARY="$OUT_DIR/summary.md"
{
  echo "# Whisper Smoke: ${BACKEND}"
  echo
  echo "- server: ${SERVER}"
  echo "- model: ${MODEL}"
  echo "- git: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "- short: ${IZWI_WHISPER_SMOKE_SHORT_AUDIO}"
  echo "- long: ${IZWI_WHISPER_SMOKE_LONG_AUDIO}"
  echo "- silence: ${IZWI_WHISPER_SMOKE_SILENCE_AUDIO}"
  echo "- non_en: ${IZWI_WHISPER_SMOKE_NON_EN_AUDIO}"
  echo "- non_en_language: ${NON_EN_LANGUAGE:-unset}"
  echo
  echo "| Case | Output |"
  echo "| --- | --- |"
} > "$SUMMARY"

run_case() {
  local case_name="$1"
  local audio_path="$2"
  local response_format="$3"
  local language="$4"
  local stream="$5"
  local output_path="$OUT_DIR/${case_name}.${stream}.${response_format}"
  if [[ "$response_format" == "verbose_json" ]]; then
    output_path="${output_path}.json"
  elif [[ "$stream" == "true" ]]; then
    output_path="${output_path}.sse"
  else
    output_path="${output_path}.txt"
  fi

  echo "running ${case_name} (${response_format}, stream=${stream})"
  request_transcription "$case_name" "$audio_path" "$response_format" "$language" "$output_path" "$stream"
  echo "| ${case_name} | ${output_path} |" >> "$SUMMARY"
}

run_case short_verbose "$IZWI_WHISPER_SMOKE_SHORT_AUDIO" verbose_json "" false
run_case long_verbose "$IZWI_WHISPER_SMOKE_LONG_AUDIO" verbose_json "" false
run_case silence_verbose "$IZWI_WHISPER_SMOKE_SILENCE_AUDIO" verbose_json "" false
run_case non_en_verbose "$IZWI_WHISPER_SMOKE_NON_EN_AUDIO" verbose_json "$NON_EN_LANGUAGE" false
run_case long_stream "$IZWI_WHISPER_SMOKE_LONG_AUDIO" json "" true
run_case short_srt "$IZWI_WHISPER_SMOKE_SHORT_AUDIO" srt "" false
run_case short_vtt "$IZWI_WHISPER_SMOKE_SHORT_AUDIO" vtt "" false

echo
echo "Whisper smoke verification completed for ${BACKEND}."
echo "Summary: ${SUMMARY}"
