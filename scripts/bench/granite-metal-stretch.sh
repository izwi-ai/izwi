#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Granite Speech Metal stretch benchmark gate.

This script benchmarks Granite-Speech-4.1-2B-Plus against one running izwi
server and validates the JSON artifact for Metal execution, non-empty ASR text,
and stretch latency targets.

Required by default:
  A single local izwi-server / `izwi serve` process must already be running.

Optional environment:
  IZWI_GRANITE_METAL_BENCH_SERVER       Default: http://127.0.0.1:8080
  IZWI_GRANITE_METAL_BENCH_MODEL        Default: Granite-Speech-4.1-2B-Plus
  IZWI_GRANITE_METAL_BENCH_AUDIO        Default: data/diarization.wav
  IZWI_GRANITE_METAL_BENCH_OUT_DIR      Default: target/granite-metal-stretch
  IZWI_GRANITE_METAL_BENCH_ITERATIONS   Default: 10
  IZWI_GRANITE_METAL_BENCH_CONCURRENT   Default: 1
  IZWI_GRANITE_METAL_BENCH_MAX_TOKENS   Default: 76
  IZWI_GRANITE_METAL_BENCH_WARMUP       Default: 1
  IZWI_GRANITE_METAL_BENCH_EXPECT       Expected transcript substring
  IZWI_GRANITE_METAL_BENCH_REQUIRE_LOCAL_SERVER
                                           Default: 1; set 0 for remote servers
  IZWI_GRANITE_METAL_BENCH_AVG_MS       Default: 3800
  IZWI_GRANITE_METAL_BENCH_P95_MS       Default: 4000
  IZWI_GRANITE_METAL_BENCH_DECODE_MS    Default: 3700
  IZWI_GRANITE_METAL_BENCH_RTF          Default: 0.140
  IZWI_GRANITE_METAL_BENCH_IZWI         Path to izwi CLI binary

Example:
  IZWI_BACKEND=metal izwi serve --backend metal
  scripts/bench/granite-metal-stretch.sh
USAGE
}

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

SERVER="${IZWI_GRANITE_METAL_BENCH_SERVER:-http://127.0.0.1:8080}"
MODEL="${IZWI_GRANITE_METAL_BENCH_MODEL:-Granite-Speech-4.1-2B-Plus}"
AUDIO="${IZWI_GRANITE_METAL_BENCH_AUDIO:-data/diarization.wav}"
OUT_DIR="${IZWI_GRANITE_METAL_BENCH_OUT_DIR:-target/granite-metal-stretch}"
ITERATIONS="${IZWI_GRANITE_METAL_BENCH_ITERATIONS:-10}"
CONCURRENT="${IZWI_GRANITE_METAL_BENCH_CONCURRENT:-1}"
MAX_TOKENS="${IZWI_GRANITE_METAL_BENCH_MAX_TOKENS:-76}"
WARMUP="${IZWI_GRANITE_METAL_BENCH_WARMUP:-1}"
REQUIRE_LOCAL_SERVER="${IZWI_GRANITE_METAL_BENCH_REQUIRE_LOCAL_SERVER:-1}"
AVG_TARGET="${IZWI_GRANITE_METAL_BENCH_AVG_MS:-3800}"
P95_TARGET="${IZWI_GRANITE_METAL_BENCH_P95_MS:-4000}"
DECODE_TARGET="${IZWI_GRANITE_METAL_BENCH_DECODE_MS:-3700}"
RTF_TARGET="${IZWI_GRANITE_METAL_BENCH_RTF:-0.140}"
EXPECT_TEXT="${IZWI_GRANITE_METAL_BENCH_EXPECT:-}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -r "$AUDIO" ]]; then
  echo "audio fixture is not readable: $AUDIO" >&2
  usage >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

python3 - "$REQUIRE_LOCAL_SERVER" <<'PY'
import os
import re
import subprocess
import sys

require_local = sys.argv[1].lower() not in {"0", "false", "no", "off"}
current = {os.getpid(), os.getppid()}
raw = subprocess.check_output(["ps", "-axo", "pid=,command="], text=True)
matches = []
for line in raw.splitlines():
    line = line.strip()
    if not line:
        continue
    pid_text, _, command = line.partition(" ")
    try:
        pid = int(pid_text)
    except ValueError:
        continue
    if pid in current:
        continue
    if "izwi-server" in command or re.search(r"\bizwi\s+serve\b", command):
        if "granite-metal-stretch.sh" not in command:
            matches.append((pid, command))

if len(matches) > 1:
    print("expected at most one local izwi server process, found:", file=sys.stderr)
    for pid, command in matches:
        print(f"  {pid} {command}", file=sys.stderr)
    sys.exit(2)
if require_local and len(matches) != 1:
    print("expected exactly one local izwi server process, found none", file=sys.stderr)
    print("start one Metal server first, for example:", file=sys.stderr)
    print("  IZWI_BACKEND=metal izwi serve --backend metal", file=sys.stderr)
    sys.exit(2)
if matches:
    pid, command = matches[0]
    print(f"using local server process: {pid} {command}")
else:
    print("no local server process detected; assuming remote server")
PY

bench_json="$OUT_DIR/bench.json"
bench_stderr="$OUT_DIR/bench.stderr"

if [[ -n "${IZWI_GRANITE_METAL_BENCH_IZWI:-}" ]]; then
  cli=("${IZWI_GRANITE_METAL_BENCH_IZWI}")
elif [[ -x target/release/izwi ]]; then
  cli=("target/release/izwi")
elif [[ -x target/debug/izwi ]]; then
  cli=("target/debug/izwi")
else
  cli=("cargo" "run" "-p" "izwi-cli" "--quiet" "--")
fi

cmd=(
  "${cli[@]}"
  "--server" "$SERVER"
  "--output-format" "json"
  "bench" "asr"
  "--model" "$MODEL"
  "--iterations" "$ITERATIONS"
  "--file" "$AUDIO"
  "--concurrent" "$CONCURRENT"
  "--max-tokens" "$MAX_TOKENS"
)

if truthy "$WARMUP"; then
  cmd+=("--warmup")
fi

echo "running Granite Metal benchmark..."
echo "  server: $SERVER"
echo "  model:  $MODEL"
echo "  audio:  $AUDIO"
echo "  json:   $bench_json"

IZWI_GRANITE_DECODE_PROFILE="${IZWI_GRANITE_DECODE_PROFILE:-1}" \
  "${cmd[@]}" >"$bench_json" 2>"$bench_stderr"

python3 - "$bench_json" "$AVG_TARGET" "$P95_TARGET" "$DECODE_TARGET" "$RTF_TARGET" "$EXPECT_TEXT" <<'PY'
import json
import sys
from pathlib import Path

path, avg_target, p95_target, decode_target, rtf_target, expected = sys.argv[1:]
avg_target = float(avg_target)
p95_target = float(p95_target)
decode_target = float(decode_target)
rtf_target = float(rtf_target)
report = json.loads(Path(path).read_text())


def fail(message):
    raise SystemExit(message)


def number_at(node, *path):
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        return None
    return float(node)


summary = report.get("summary")
if not isinstance(summary, dict):
    fail("benchmark report missing summary")

avg = number_at(summary, "latency_ms", "avg")
p95 = number_at(summary, "latency_ms", "p95")
rtf = number_at(summary, "rtf", "avg")
if avg is None or p95 is None or rtf is None:
    fail("benchmark summary missing latency_ms.avg, latency_ms.p95, or rtf.avg")
if avg > avg_target:
    fail(f"average latency {avg:.2f}ms exceeds target {avg_target:.2f}ms")
if p95 > p95_target:
    fail(f"p95 latency {p95:.2f}ms exceeds target {p95_target:.2f}ms")
if rtf > rtf_target:
    fail(f"RTF {rtf:.3f} exceeds target {rtf_target:.3f}")

samples = report.get("samples")
if not isinstance(samples, list) or not samples:
    fail("benchmark report missing samples")

decode_values = []
for index, sample in enumerate(samples, start=1):
    if not isinstance(sample, dict):
        fail(f"sample {index} is not an object")
    text = sample.get("asr_text")
    if not isinstance(text, str) or not text.strip():
        fail(f"sample {index} has empty ASR text")
    if expected and expected.lower() not in text.lower():
        fail(f"sample {index} transcript does not contain expected text {expected!r}")

    diagnostics = sample.get("asr_diagnostics")
    if not isinstance(diagnostics, dict):
        fail(f"sample {index} missing asr_diagnostics")
    family = diagnostics.get("family") or diagnostics.get("model_family")
    if family != "granite_speech_asr":
        fail(f"sample {index} expected granite_speech_asr diagnostics, got {family!r}")
    if diagnostics.get("device_kind") != "Metal":
        fail(f"sample {index} expected Metal device_kind, got {diagnostics.get('device_kind')!r}")

    execution = diagnostics.get("execution")
    if not isinstance(execution, dict):
        fail(f"sample {index} missing execution diagnostics")
    for key in (
        "dense_head_decode_enabled",
        "dense_decode_preallocated",
        "audio_embedding_cache_hit",
        "qkv_projection_fused",
        "gate_up_projection_fused",
        "residual_branches_prescaled",
    ):
        if execution.get(key) is not True:
            fail(f"sample {index} expected execution.{key}=true")
    if (
        execution.get("chunked_stop_check") is not True
        and execution.get("deferred_stop_check") is not True
    ):
        fail(f"sample {index} expected chunked or deferred stop checks")

    decode_ms = number_at(diagnostics, "timings_ms", "decode")
    if decode_ms is None:
        fail(f"sample {index} missing timings_ms.decode")
    decode_values.append(decode_ms)

decode_avg = sum(decode_values) / len(decode_values)
if decode_avg > decode_target:
    fail(f"decode average {decode_avg:.2f}ms exceeds target {decode_target:.2f}ms")

print("Granite Metal stretch gate passed:")
print(f"  latency avg: {avg:.2f}ms")
print(f"  latency p95: {p95:.2f}ms")
print(f"  decode avg:  {decode_avg:.2f}ms")
print(f"  rtf avg:     {rtf:.3f}")
PY

echo "stderr log: $bench_stderr"
