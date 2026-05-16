#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Qwen-ASR smoke verification against a running izwi server.

Required environment:
  IZWI_QWEN_ASR_SMOKE_BACKEND      Label for this run: cpu, metal, or cuda
  IZWI_QWEN_ASR_SMOKE_SHORT_AUDIO  Short speech fixture, under 30 seconds
  IZWI_QWEN_ASR_SMOKE_LONG_AUDIO   Long real speech fixture

Optional environment:
  IZWI_QWEN_ASR_SMOKE_SERVER       Default: http://127.0.0.1:8080
  IZWI_QWEN_ASR_SMOKE_MODEL        Default: Qwen3-ASR-0.6B-GGUF
  IZWI_QWEN_ASR_SMOKE_OUT_DIR      Default: target/qwen-asr-smoke/<backend>
  IZWI_QWEN_ASR_SMOKE_TIMEOUT_SECS Default: 900
  IZWI_QWEN_ASR_SMOKE_SHORT_EXPECT Expected substring in the short transcript
  IZWI_QWEN_ASR_SMOKE_LONG_EXPECTED_PREFIX
                                      Expected prefix for the long transcript

Start the server separately with the backend being verified, for example:
  IZWI_BACKEND=cuda izwi serve --backend cuda
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
  local output_path="$4"
  local stream="$5"

  python3 - "$SERVER" "$MODEL" "$case_name" "$audio_path" "$response_format" "$output_path" "$stream" "$BACKEND" <<'PY'
import base64
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

server, model, case_name, audio_path, response_format, output_path, stream, backend = sys.argv[1:]
timeout = int(os.environ.get("IZWI_QWEN_ASR_SMOKE_TIMEOUT_SECS", "900"))
payload = {
    "model": model,
    "audio_base64": base64.b64encode(Path(audio_path).read_bytes()).decode("ascii"),
    "response_format": response_format,
}
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

Path(output_path).write_bytes(body)
text = body.decode("utf-8", errors="replace")


def require_non_negative_number(label, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise SystemExit(f"{case_name}: expected non-negative numeric {label}, got {value!r}")


def validate_qwen_diagnostics(diagnostics):
    if not isinstance(diagnostics, dict):
        raise SystemExit(f"{case_name}: missing izwi_asr_diagnostics object")
    if diagnostics.get("model_family") != "qwen3_asr":
        raise SystemExit(
            f"{case_name}: expected qwen3_asr diagnostics, got {diagnostics.get('model_family')!r}"
        )
    for section in ("audio", "prompt", "decode", "timings_ms"):
        if not isinstance(diagnostics.get(section), dict):
            raise SystemExit(f"{case_name}: diagnostics missing {section} object")
    timings = diagnostics["timings_ms"]
    for key in ("resample", "mel", "audio_encode", "prefill", "decode", "model_total"):
        require_non_negative_number(f"timings_ms.{key}", timings.get(key))
    for section, keys in (
        ("audio", ("audio_tokens", "mel_frames")),
        ("prompt", ("prompt_tokens",)),
        ("decode", ("generated_tokens", "max_new_tokens")),
    ):
        node = diagnostics[section]
        for key in keys:
            value = node.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SystemExit(
                    f"{case_name}: expected non-negative integer {section}.{key}, got {value!r}"
                )
    if diagnostics["decode"]["generated_tokens"] == 0 and case_name != "short_verbose":
        raise SystemExit(f"{case_name}: generated_tokens was zero")


def validate_text(response_text):
    stripped = response_text.strip()
    if not stripped:
        raise SystemExit(f"{case_name}: transcript text was empty")
    if case_name.startswith("short"):
        expected = os.environ.get("IZWI_QWEN_ASR_SMOKE_SHORT_EXPECT", "")
        if expected and expected.lower() not in stripped.lower():
            raise SystemExit(
                f"{case_name}: transcript did not contain expected text {expected!r}: {stripped[:120]!r}"
            )
    if case_name.startswith("long"):
        expected_prefix = os.environ.get("IZWI_QWEN_ASR_SMOKE_LONG_EXPECTED_PREFIX", "")
        if expected_prefix and not stripped.lower().startswith(expected_prefix.lower()):
            raise SystemExit(
                f"{case_name}: transcript did not start with {expected_prefix!r}: {stripped[:120]!r}"
            )
        prefix_window = stripped[:8]
        if any(ord(ch) > 127 for ch in prefix_window):
            raise SystemExit(
                f"{case_name}: transcript starts with non-ASCII characters: {prefix_window!r}"
            )


if stream == "true":
    if "transcript.text.done" not in text:
        raise SystemExit(f"{case_name}: streaming response did not include transcript.text.done")
    prefix = os.environ.get("IZWI_QWEN_ASR_SMOKE_LONG_EXPECTED_PREFIX", "")
    if prefix and prefix.lower() not in text.lower():
        raise SystemExit(f"{case_name}: streaming response did not include expected prefix {prefix!r}")
else:
    parsed = json.loads(text)
    validate_text(str(parsed.get("text") or ""))
    diagnostics = parsed.get("izwi_asr_diagnostics")
    if isinstance(diagnostics, dict) and isinstance(diagnostics.get("model_diagnostics"), dict):
        diagnostics = diagnostics["model_diagnostics"]
    validate_qwen_diagnostics(diagnostics)
PY
}

require_var IZWI_QWEN_ASR_SMOKE_BACKEND
require_var IZWI_QWEN_ASR_SMOKE_SHORT_AUDIO
require_var IZWI_QWEN_ASR_SMOKE_LONG_AUDIO

BACKEND="$IZWI_QWEN_ASR_SMOKE_BACKEND"
SERVER="${IZWI_QWEN_ASR_SMOKE_SERVER:-http://127.0.0.1:8080}"
MODEL="${IZWI_QWEN_ASR_SMOKE_MODEL:-Qwen3-ASR-0.6B-GGUF}"
OUT_DIR="${IZWI_QWEN_ASR_SMOKE_OUT_DIR:-target/qwen-asr-smoke/${BACKEND}}"

case "$BACKEND" in
  cpu|metal|cuda) ;;
  *)
    echo "IZWI_QWEN_ASR_SMOKE_BACKEND must be cpu, metal, or cuda, got: ${BACKEND}" >&2
    exit 2
    ;;
esac

require_file short "$IZWI_QWEN_ASR_SMOKE_SHORT_AUDIO"
require_file long "$IZWI_QWEN_ASR_SMOKE_LONG_AUDIO"

mkdir -p "$OUT_DIR"

SUMMARY="$OUT_DIR/summary.md"
{
  echo "# Qwen-ASR Smoke: ${BACKEND}"
  echo
  echo "- server: ${SERVER}"
  echo "- model: ${MODEL}"
  echo "- git: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "- short: ${IZWI_QWEN_ASR_SMOKE_SHORT_AUDIO}"
  echo "- long: ${IZWI_QWEN_ASR_SMOKE_LONG_AUDIO}"
  echo "- short_expect: ${IZWI_QWEN_ASR_SMOKE_SHORT_EXPECT:-unset}"
  echo "- long_expected_prefix: ${IZWI_QWEN_ASR_SMOKE_LONG_EXPECTED_PREFIX:-unset}"
  echo
  echo "| Case | Output |"
  echo "| --- | --- |"
} > "$SUMMARY"

run_case() {
  local case_name="$1"
  local audio_path="$2"
  local response_format="$3"
  local stream="$4"
  local output_path="$OUT_DIR/${case_name}.${stream}.${response_format}"
  if [[ "$stream" == "true" ]]; then
    output_path="${output_path}.sse"
  else
    output_path="${output_path}.json"
  fi

  echo "running ${case_name} (${response_format}, stream=${stream})"
  request_transcription "$case_name" "$audio_path" "$response_format" "$output_path" "$stream"
  echo "| ${case_name} | ${output_path} |" >> "$SUMMARY"
}

run_case short_verbose "$IZWI_QWEN_ASR_SMOKE_SHORT_AUDIO" verbose_json false
run_case long_verbose "$IZWI_QWEN_ASR_SMOKE_LONG_AUDIO" verbose_json false
run_case long_stream "$IZWI_QWEN_ASR_SMOKE_LONG_AUDIO" json true

echo
echo "Qwen-ASR smoke verification completed for ${BACKEND}."
echo "Summary: ${SUMMARY}"
