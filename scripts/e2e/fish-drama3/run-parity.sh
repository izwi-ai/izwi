#!/usr/bin/env bash
# The family's parity suites on one backend, in one release test build of
# izwi-core (`--features metal|cuda` on those backends), leg by leg:
#   unit       the non-ignored family suites (`fish_drama3`, the codec
#              equivalence; on CUDA also the device kernels the family uses),
#              with a missing device a failure, not a skip
#   devices    the ignored codec device tests (`metal_*` / `cuda_*`)
#   tokenizer  the published tokenizer's golden ids and prompts, and the
#              upstream prompt and chunk goldens when generated
#              (IZWI_FISH_DRAMA3_TOKENIZER_PATH)
#   codec      the published codec.pth bit-identical to Fish S2's, and its
#              memory price (IZWI_FISH_DRAMA3_CODEC_PATH)
#   tiny       L1: `reference_tests::tiny_*` (tests/fixtures/fish_drama3/tiny/)
#   slice      L1.5: `reference_tests::slice_*` (IZWI_FISH_DRAMA3_SLICE_DIR)
#   reference  L2: `reference_tests::reference_*` (IZWI_FISH_DRAMA3_MODEL_DIR,
#              IZWI_FISH_DRAMA3_REFERENCE_DIR, IZWI_FISH_DRAMA3_BACKEND)
# A leg whose fixture is missing is skipped and reported; with
# --require-all a skipped leg fails the run.
#
# Usage: run-parity.sh <cpu|metal|cuda> [--only a,b] [--skip a,b]
#                      [--l2-cpu-form cpu|cpu-q8_0] [--require-all] [--dry-run]
#   --l2-cpu-form  the CPU's L2 form: cpu-q8_0 (default; the production Q8_0
#                  weights, about 40 GiB, divergence reported only) or cpu
#                  (F32 weights, about 140 GiB of RAM; must reach 64 frames)
# Env: IZWI_FISH_DRAMA3_MODEL_DIR (default: the fetch.sh location),
#      IZWI_FISH_DRAMA3_SLICE_DIR (default: <repo>/target/fish-drama3-slices),
#      IZWI_FISH_DRAMA3_REFERENCE_DIR (default: <repo>/target/fish-drama3-reference),
#      IZWI_FISH_DRAMA3_CODEBOOK_FIXTURE (CUDA codec streaming; skipped unset),
#      CUDA_COMPUTE_CAP, FD3_CARGO_TARGET_DIR, FD3_BUILD_MIN_FREE_GB (25).
# Writes <out>/parity-<leg>.log, <out>/parity.json and the `parity` section
# of <out>/report.json.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BACKEND="${1:-}"
if [[ "$BACKEND" == "-h" || "$BACKEND" == "--help" ]]; then
    sed -n '2,33p' "$0"
    exit 0
fi
fd3_check_backend "$BACKEND"
shift
ONLY=""
SKIP=""
L2_CPU_FORM="cpu-q8_0"
REQUIRE_ALL=0
DRY_RUN=0
while (($#)); do
    case "$1" in
        --only) ONLY="$2"; shift ;;
        --skip) SKIP="$2"; shift ;;
        --l2-cpu-form) L2_CPU_FORM="$2"; shift ;;
        --require-all) REQUIRE_ALL=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h | --help) sed -n '2,33p' "$0"; exit 0 ;;
        *) fd3_die "unknown argument $1" ;;
    esac
    shift
done
case "$L2_CPU_FORM" in cpu | cpu-q8_0) ;; *) fd3_die "--l2-cpu-form must be cpu or cpu-q8_0" ;; esac

OUT="$(fd3_backend_out "$BACKEND")"
MODEL_DIR="${IZWI_FISH_DRAMA3_MODEL_DIR:-$FD3_MODEL_DIR}"
SLICE_DIR="${IZWI_FISH_DRAMA3_SLICE_DIR:-$REPO_ROOT/target/fish-drama3-slices}"
REFERENCE_DIR="${IZWI_FISH_DRAMA3_REFERENCE_DIR:-$REPO_ROOT/target/fish-drama3-reference}"
L2_BACKEND="$BACKEND"
if [[ "$BACKEND" == "cpu" ]]; then L2_BACKEND="$L2_CPU_FORM"; fi

FEATURES="$(fd3_core_features "$BACKEND")"
CARGO=(cargo test -p izwi-core --release --locked)
if [[ -n "$FEATURES" ]]; then CARGO+=(--features "$FEATURES"); fi
CARGO+=(--lib --)
# A missing device fails instead of skipping.
DEVICE_ENV=()
case "$BACKEND" in
    metal) DEVICE_ENV=(IZWI_REQUIRE_METAL_PARITY=1) ;;
    cuda) DEVICE_ENV=(IZWI_REQUIRE_CUDA_TEST_DEVICE=1) ;;
esac

wanted() {
    local leg="$1"
    if [[ -n "$ONLY" && ",$ONLY," != *",$leg,"* ]]; then return 1; fi
    if [[ ",$SKIP," == *",$leg,"* ]]; then return 1; fi
    return 0
}

model_present() {
    bash "$FD3_DIR/fetch.sh" --dest "$MODEL_DIR" --verify-only >/dev/null 2>&1
}

# Sets LEG_SKIP (a reason, or empty), LEG_ENV and LEG_ARGS (the test
# filters and flags) for one leg. Arrays, since the models directory holds
# a space on macOS.
leg_plan() {
    LEG_SKIP=""
    LEG_ENV=(${DEVICE_ENV[@]+"${DEVICE_ENV[@]}"})
    LEG_ARGS=()
    case "$1" in
        unit)
            LEG_ARGS=(fish_drama3 fish_codec_equivalence)
            if [[ "$BACKEND" == "cuda" ]]; then
                LEG_ARGS+=(kernels::cuda::row_gemv kernels::cuda::split_decode)
            fi
            ;;
        devices)
            if [[ "$BACKEND" == "cpu" ]]; then
                LEG_SKIP="no device tests on the CPU"
                return
            fi
            LEG_ARGS=(fish_drama3::codec:: "fish_codec_equivalence::${BACKEND}_synthetic" --ignored)
            # Needs the published codec; the codec leg runs it.
            if [[ "$BACKEND" == "cuda" ]]; then
                LEG_ARGS+=(--skip cuda_real_codec_streaming_matches_frozen_codebooks)
            fi
            ;;
        tokenizer)
            if [[ ! -f "$MODEL_DIR/tokenizer.json" ]]; then
                LEG_SKIP="no $MODEL_DIR/tokenizer.json (run fetch.sh)"
                return
            fi
            LEG_ENV+=("IZWI_FISH_DRAMA3_TOKENIZER_PATH=$MODEL_DIR/tokenizer.json")
            LEG_ARGS=(the_published_tokenizer_)
            if [[ -f "$FD3_FIXTURE_DIR/prompt_goldens.json" ]]; then
                LEG_ARGS+=(upstream_prompt_goldens)
            fi
            if [[ -f "$FD3_FIXTURE_DIR/chunk_goldens.json" ]]; then
                LEG_ARGS+=(upstream_chunk_goldens upstream_history_selection_goldens)
            fi
            LEG_ARGS+=(--ignored)
            ;;
        codec)
            if [[ ! -f "$MODEL_DIR/codec.pth" ]]; then
                LEG_SKIP="no $MODEL_DIR/codec.pth (run fetch.sh)"
                return
            fi
            LEG_ENV+=("IZWI_FISH_DRAMA3_CODEC_PATH=$MODEL_DIR/codec.pth")
            LEG_ARGS=(real_codecs_are_bit_identical_on_cpu
                the_pinned_codec_memory_is_the_published_codec_s)
            case "$BACKEND" in
                metal) LEG_ARGS+=(metal_real_codecs_are_bit_identical) ;;
                cuda)
                    LEG_ARGS+=(cuda_real_codecs_are_bit_identical)
                    if [[ -n "${IZWI_FISH_DRAMA3_CODEBOOK_FIXTURE:-}" ]]; then
                        LEG_ARGS+=(cuda_real_codec_streaming_matches_frozen_codebooks)
                    fi
                    ;;
            esac
            LEG_ARGS+=(--ignored)
            ;;
        tiny)
            if [[ ! -f "$FD3_FIXTURE_DIR/tiny/expected.json" ]]; then
                LEG_SKIP="no $FD3_FIXTURE_DIR/tiny/ (run scripts/fixtures/generate_fish_drama3_tiny.py)"
                return
            fi
            LEG_ARGS=(fish_drama3::reference_tests::tiny --ignored)
            ;;
        slice)
            if [[ ! -f "$SLICE_DIR/slice_expected.json" ]]; then
                LEG_SKIP="no $SLICE_DIR/slice_expected.json (run fetch_fish_drama3_slices.py, then generate_fish_drama3_slices.py)"
                return
            fi
            LEG_ENV+=("IZWI_FISH_DRAMA3_SLICE_DIR=$SLICE_DIR")
            LEG_ARGS=(fish_drama3::reference_tests::slice --ignored)
            ;;
        reference)
            if [[ ! -f "$REFERENCE_DIR/reference.json" ]]; then
                LEG_SKIP="no $REFERENCE_DIR/reference.json (run scripts/fixtures/generate_fish_drama3_reference.py)"
                return
            fi
            if ! model_present; then
                LEG_SKIP="$MODEL_DIR is not the pinned bundle (run fetch.sh)"
                return
            fi
            LEG_ENV+=("IZWI_FISH_DRAMA3_MODEL_DIR=$MODEL_DIR"
                "IZWI_FISH_DRAMA3_REFERENCE_DIR=$REFERENCE_DIR"
                "IZWI_FISH_DRAMA3_BACKEND=$L2_BACKEND")
            LEG_ARGS=(fish_drama3::reference_tests::reference_ --ignored)
            ;;
    esac
    # One real-weight load in memory at a time; device tests share a device.
    LEG_ARGS+=(--test-threads=1)
}

LEGS="unit devices tokenizer codec tiny slice reference"
TARGET_DIR="$(fd3_target_dir "$BACKEND")"
if ((DRY_RUN)); then
    fd3_log "dry run on $BACKEND (CARGO_TARGET_DIR=$TARGET_DIR CARGO_INCREMENTAL=0):"
    for leg in $LEGS; do
        wanted "$leg" || continue
        leg_plan "$leg"
        if [[ -n "$LEG_SKIP" ]]; then
            fd3_log "  $leg: skipped, $LEG_SKIP"
        else
            fd3_log "  $leg: $(fd3_show ${LEG_ENV[@]+"${LEG_ENV[@]}"} "${CARGO[@]}" "${LEG_ARGS[@]}")"
        fi
    done
    exit 0
fi

mkdir -p "$OUT"
if [[ "$BACKEND" == "cuda" ]]; then export CUDA_COMPUTE_CAP="$(fd3_cuda_compute_cap)"; fi
export CARGO_TARGET_DIR="$TARGET_DIR"
export CARGO_INCREMENTAL=0
mkdir -p "$CARGO_TARGET_DIR"
fd3_require_free_gb "$CARGO_TARGET_DIR" "${FD3_BUILD_MIN_FREE_GB:-25}"

RESULTS="$OUT/parity-results.txt"
: >"$RESULTS"
for leg in $LEGS; do
    wanted "$leg" || continue
    leg_plan "$leg"
    if [[ -n "$LEG_SKIP" ]]; then
        fd3_log "skip $leg: $LEG_SKIP"
        printf '%s\tskip\t0\t%s\n' "$leg" "$LEG_SKIP" >>"$RESULTS"
        continue
    fi
    log="$OUT/parity-$leg.log"
    fd3_log "$leg: $(fd3_show ${LEG_ENV[@]+"${LEG_ENV[@]}"} "${CARGO[@]}" "${LEG_ARGS[@]}") -> $log"
    started=$(date +%s)
    set +e
    (cd "$REPO_ROOT" && env ${LEG_ENV[@]+"${LEG_ENV[@]}"} "${CARGO[@]}" "${LEG_ARGS[@]}" --nocapture 2>&1) |
        tee "$log"
    status=${PIPESTATUS[0]}
    set -e
    printf '%s\t%s\t%s\t%s\n' "$leg" "$status" "$(($(date +%s) - started))" "$log" >>"$RESULTS"
done

"$(fd3_python)" - "$RESULTS" "$OUT/parity.json" "$BACKEND" "$L2_BACKEND" "$REQUIRE_ALL" <<'PY'
import json, re, sys
from pathlib import Path
results, output, backend, l2_backend, require_all = sys.argv[1:6]
legs = []
for line in Path(results).read_text().splitlines():
    leg, status, seconds, detail = line.split("\t")
    if status == "skip":
        legs.append({"leg": leg, "status": "skipped", "reason": detail})
        continue
    text = Path(detail).read_text(errors="replace")
    counts = {"passed": 0, "failed": 0, "ignored": 0}
    for match in re.finditer(r"test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored", text):
        for key, value in zip(("passed", "failed", "ignored"), match.groups()):
            counts[key] += int(value)
    failed = re.findall(r"^test (\S+) \.\.\. FAILED", text, re.M)
    skipped = sorted(set(re.findall(r"^(\S+): skipped", text, re.M)))
    ran = counts["passed"] + counts["failed"] > 0
    legs.append({"leg": leg, "exit_code": int(status), "seconds": int(seconds),
                 "log": Path(detail).name, **counts, "failed_tests": failed,
                 "device_skips": skipped, "ran_any": ran,
                 "status": "pass" if int(status) == 0 and ran and not skipped else "fail"})
ran = [leg for leg in legs if leg["status"] != "skipped"]
status = "pass" if ran and all(leg["status"] == "pass" for leg in ran) else "fail"
if status == "pass" and len(ran) < len(legs):
    status = "fail" if require_all == "1" else "incomplete"
section = {"backend": backend, "l2_backend": l2_backend, "status": status, "legs": legs}
Path(output).write_text(json.dumps(section, indent=1) + "\n")
for leg in legs:
    print(f"  {leg['leg']:10} {leg['status']:8} "
          + (leg.get("reason") or f"{leg.get('passed')} passed, {leg.get('failed')} failed"))
print(f"parity: {status}")
PY
"$(fd3_python)" "$FD3_DIR/fd3_harness.py" merge --report "$OUT/report.json" --section parity \
    --file "$OUT/parity.json"
case "$("$(fd3_python)" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$OUT/parity.json")" in
    pass) fd3_log "parity passed on $BACKEND" ;;
    incomplete) fd3_log "parity passed on $BACKEND for the legs that ran; some were skipped (see above)" ;;
    *) fd3_die "parity FAILED on $BACKEND (a leg failed, skipped a device, or ran no test); see $OUT/parity-*.log" ;;
esac
