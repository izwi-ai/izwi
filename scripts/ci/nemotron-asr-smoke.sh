#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Nemotron-ASR scaffold smoke verification against a running izwi server.

Optional environment:
  IZWI_NEMOTRON_ASR_SMOKE_SERVER       Default: http://127.0.0.1:8080
  IZWI_NEMOTRON_ASR_SMOKE_MODEL        Default: Nemotron-3.5-ASR-Streaming-0.6B
  IZWI_NEMOTRON_ASR_SMOKE_AUDIO        WAV/FLAC/MP3 fixture. If unset, a short silent WAV is generated.
  IZWI_NEMOTRON_ASR_SMOKE_OUT_DIR      Default: target/nemotron-asr-smoke
  IZWI_NEMOTRON_ASR_SMOKE_TIMEOUT_SECS Default: 900

This script validates the current native Rust scaffold: model artifacts should
load, and transcription should fail with the expected pending FastConformer-RNNT
forward-pass diagnostic until the tensor mapping is completed.

Start the server separately, for example:
  IZWI_BACKEND=cpu izwi serve --backend cpu
USAGE
}

SERVER="${IZWI_NEMOTRON_ASR_SMOKE_SERVER:-http://127.0.0.1:8080}"
MODEL="${IZWI_NEMOTRON_ASR_SMOKE_MODEL:-Nemotron-3.5-ASR-Streaming-0.6B}"
OUT_DIR="${IZWI_NEMOTRON_ASR_SMOKE_OUT_DIR:-target/nemotron-asr-smoke}"
TIMEOUT="${IZWI_NEMOTRON_ASR_SMOKE_TIMEOUT_SECS:-900}"
mkdir -p "$OUT_DIR"

AUDIO_PATH="${IZWI_NEMOTRON_ASR_SMOKE_AUDIO:-$OUT_DIR/silence.wav}"
if [[ -z "${IZWI_NEMOTRON_ASR_SMOKE_AUDIO:-}" ]]; then
  python3 - "$AUDIO_PATH" <<'PY'
import struct
import sys
import wave

path = sys.argv[1]
sample_rate = 16000
samples = [0] * (sample_rate // 2)
with wave.open(path, "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(sample_rate)
    wav.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))
PY
fi

if [[ ! -r "$AUDIO_PATH" ]]; then
  echo "audio fixture is not readable: $AUDIO_PATH" >&2
  usage >&2
  exit 2
fi

python3 - "$SERVER" "$MODEL" "$AUDIO_PATH" "$OUT_DIR" "$TIMEOUT" <<'PY'
import base64
import json
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request

server, model, audio_path, out_dir, timeout_s = sys.argv[1:]
timeout = int(timeout_s)
server = server.rstrip("/")
encoded_model = urllib.parse.quote(model, safe="")

def post_json(path, payload):
    request = urllib.request.Request(
        f"{server}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")

status, body = post_json(f"/v1/admin/models/{encoded_model}/load", {})
Path(out_dir, "load.json").write_text(body)
if status >= 400:
    raise SystemExit(f"load failed with HTTP {status}: {body[:500]}")

payload = {
    "model": model,
    "audio_base64": base64.b64encode(Path(audio_path).read_bytes()).decode("ascii"),
    "response_format": "json",
    "language": "auto",
}
status, body = post_json("/v1/audio/transcriptions", payload)
Path(out_dir, "transcription.json").write_text(body)

expected = "native FastConformer-RNNT forward pass is not enabled yet"
if status < 400:
    raise SystemExit(
        "Nemotron transcription unexpectedly succeeded before native forward mapping was completed"
    )
if expected not in body:
    raise SystemExit(
        f"Nemotron scaffold error did not contain expected diagnostic {expected!r}: {body[:500]}"
    )

print("Nemotron ASR scaffold smoke passed.")
PY
