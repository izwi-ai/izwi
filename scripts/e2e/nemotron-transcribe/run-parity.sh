#!/usr/bin/env bash
# The real-weight parity tests (L2) and the family's unit/L1 tests on one
# backend:
#   cargo test -p izwi-core --release --features <metal|cuda|> --lib \
#       -- nemotron_transcribe::reference_tests --ignored
# with IZWI_NEMOTRON_TRANSCRIBE_MODEL_DIR, IZWI_NEMOTRON_TRANSCRIBE_AUDIO_DIR
# and IZWI_NEMOTRON_TRANSCRIBE_BACKEND set.
#
# Usage: run-parity.sh <cpu|metal|cuda> [--both-modes] [--skip-unit]
#   --both-modes  run the ignored tests under IZWI_NEMOTRON_TRANSCRIBE_FP8_MODE
#                 w8a16 and vllm-fp8 (compare vllm-fp8 with an SM89+ Leg B)
#   --skip-unit   skip the non-ignored `nemotron_transcribe::` tests (L0/L1)
# Env: IZWI_NEMOTRON_TRANSCRIBE_MODEL_DIR (default: the fetch.sh location),
#      IZWI_NEMOTRON_TRANSCRIBE_AUDIO_DIR (default: the generator's
#      tests/fixtures/nemotron_transcribe/reference-audio),
#      IZWI_NEMOTRON_TRANSCRIBE_FP8_MODE (default w8a16), CUDA_COMPUTE_CAP.
# Writes <out>/parity-*.log and the `parity` section of <out>/report.json.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BACKEND="${1:-}"
nt_check_backend "$BACKEND"
shift
BOTH=0
UNIT=1
while (($#)); do
    case "$1" in
        --both-modes) BOTH=1 ;;
        --skip-unit) UNIT=0 ;;
        *) nt_die "unknown argument $1" ;;
    esac
    shift
done

OUT="$(nt_backend_out "$BACKEND")"
mkdir -p "$OUT"
export IZWI_NEMOTRON_TRANSCRIBE_MODEL_DIR="${IZWI_NEMOTRON_TRANSCRIBE_MODEL_DIR:-$NT_MODEL_DIR}"
export IZWI_NEMOTRON_TRANSCRIBE_AUDIO_DIR="${IZWI_NEMOTRON_TRANSCRIBE_AUDIO_DIR:-$NT_FIXTURE_DIR/reference-audio}"
export IZWI_NEMOTRON_TRANSCRIBE_BACKEND="$BACKEND"
REFERENCE="$NT_FIXTURE_DIR/reference.json"

bash "$NT_DIR/fetch.sh" --dest "$IZWI_NEMOTRON_TRANSCRIBE_MODEL_DIR" --verify-only
[[ -f "$REFERENCE" ]] || nt_die "no $REFERENCE: run scripts/fixtures/generate_nemotron_transcribe_reference.py on the GPU box first"
[[ -d "$IZWI_NEMOTRON_TRANSCRIBE_AUDIO_DIR" ]] || nt_die "no audio directory $IZWI_NEMOTRON_TRANSCRIBE_AUDIO_DIR"

# The clips must be the very samples the reference was made from.
"$(nt_python)" - "$REFERENCE" "$IZWI_NEMOTRON_TRANSCRIBE_AUDIO_DIR" <<'PY'
import hashlib, json, sys
from pathlib import Path
reference, audio = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
bad = []
for clip in reference.get("clips", []):
    for key, digest in (("audio_file", "sha256_wav"), ("raw_file", "sha256_f32")):
        path = audio / clip[key]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != clip[digest]:
            bad.append(str(path))
if bad:
    sys.exit("audio differs from reference.json (regenerate or copy the generator's "
             "reference-audio): " + ", ".join(bad))
print(f"{len(reference.get('clips', []))} clips match reference.json")
PY

FEATURES="$(nt_core_features "$BACKEND")"
FEATURE_ARGS=()
if [[ -n "$FEATURES" ]]; then FEATURE_ARGS=(--features "$FEATURES"); fi
if [[ "$BACKEND" == "cuda" ]]; then export CUDA_COMPUTE_CAP="$(nt_cuda_compute_cap)"; fi
export CARGO_TARGET_DIR="$(nt_target_dir "$BACKEND")"
export CARGO_INCREMENTAL=0
nt_require_free_gb "$CARGO_TARGET_DIR" "${NT_BUILD_MIN_FREE_GB:-25}"

RESULTS="$OUT/parity-results.txt"
: >"$RESULTS"

run_tests() {
    local label="$1" filter="$2" ignored="$3" mode="$4"
    local log="$OUT/parity-$label.log" started status
    local extra=()
    if [[ "$ignored" == "1" ]]; then extra=(--ignored); fi
    nt_log "cargo test ($label, IZWI_NEMOTRON_TRANSCRIBE_FP8_MODE=$mode) -> $log"
    started=$(date +%s)
    set +e
    (
        cd "$REPO_ROOT"
        IZWI_NEMOTRON_TRANSCRIBE_FP8_MODE="$mode" cargo test -p izwi-core --release --locked \
            ${FEATURE_ARGS[@]+"${FEATURE_ARGS[@]}"} --lib -- "$filter" \
            ${extra[@]+"${extra[@]}"} --nocapture --test-threads=1 2>&1
    ) | tee "$log"
    status=${PIPESTATUS[0]}
    set -e
    printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$mode" "$status" "$(($(date +%s) - started))" \
        "$log" >>"$RESULTS"
}

if ((UNIT)); then
    run_tests "unit" "nemotron_transcribe::" 0 "${IZWI_NEMOTRON_TRANSCRIBE_FP8_MODE:-w8a16}"
fi
if ((BOTH)); then
    run_tests "reference-w8a16" "nemotron_transcribe::reference_tests" 1 "w8a16"
    run_tests "reference-vllm-fp8" "nemotron_transcribe::reference_tests" 1 "vllm-fp8"
else
    mode="${IZWI_NEMOTRON_TRANSCRIBE_FP8_MODE:-w8a16}"
    run_tests "reference-$mode" "nemotron_transcribe::reference_tests" 1 "$mode"
fi

"$(nt_python)" - "$RESULTS" "$OUT/parity.json" "$BACKEND" <<'PY'
import json, re, sys
from pathlib import Path
runs = []
for line in Path(sys.argv[1]).read_text().splitlines():
    label, mode, status, seconds, log = line.split("\t")
    text = Path(log).read_text(errors="replace")
    counts = {}
    for match in re.finditer(r"test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored", text):
        counts = {"result": match.group(1), "passed": int(match.group(2)),
                  "failed": int(match.group(3)), "ignored": int(match.group(4))}
    failed = re.findall(r"^test (\S+) \.\.\. FAILED", text, re.M)
    runs.append({"label": label, "fp8_mode": mode, "exit_code": int(status),
                 "seconds": int(seconds), "log": Path(log).name, **counts,
                 "failed_tests": failed,
                 "ran_any": counts.get("passed", 0) + counts.get("failed", 0) > 0})
section = {"backend": sys.argv[3], "runs": runs,
           "status": "pass" if all(r["exit_code"] == 0 and r["ran_any"] for r in runs) else "fail"}
Path(sys.argv[2]).write_text(json.dumps(section, indent=1) + "\n")
print(json.dumps(section, indent=1))
PY
"$(nt_python)" "$NT_DIR/nt_harness.py" merge --report "$OUT/report.json" --section parity \
    --file "$OUT/parity.json"
if grep -q '"status": "fail"' "$OUT/parity.json"; then
    nt_log "parity FAILED (a run failed or ran no test); see $OUT/parity-*.log"
    exit 1
fi
nt_log "parity passed on $BACKEND"
