#!/usr/bin/env bash

set -euo pipefail

evidence=""
report=""
backend=""
git_sha=""

usage() {
    cat <<'EOF'
Usage: scripts/bench/validate-audio-runtime-evidence.sh \
  --evidence PATH --report PATH --backend cpu|metal|cuda --expected-git-sha SHA

Validates the externally produced destructive/stress portion of ASR/TTS
certification. The artifact must cover exactly the models in the benchmark
report and must prove fairness, cancellation, cache-pressure recovery,
unload/drain, c1 fallback, memory plateau, and exact-device execution.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --evidence) evidence="${2:-}"; shift 2 ;;
        --report) report="${2:-}"; shift 2 ;;
        --backend) backend="${2:-}"; shift 2 ;;
        --expected-git-sha) git_sha="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "${backend}" in cpu|metal|cuda) ;; *) echo "--backend must be cpu, metal, or cuda" >&2; exit 2 ;; esac
for value in evidence report git_sha; do
    if [[ -z "${!value}" ]]; then
        echo "--${value//_/-} is required" >&2
        exit 2
    fi
done
for path in "${evidence}" "${report}"; do
    if [[ ! -s "${path}" ]]; then
        echo "Required evidence input is missing or empty: ${path}" >&2
        exit 1
    fi
done
command -v jq >/dev/null 2>&1 || { echo "Missing required command: jq" >&2; exit 1; }

expected_cases=$(jq -c '
    [.reports[].report | {model: .config.model, task: .command}]
    | sort_by(.task, .model) | unique
' "${report}")

if ! jq -e \
    --arg backend "${backend}" \
    --arg git_sha "${git_sha}" \
    --argjson expected_cases "${expected_cases}" '
    def finite_nonnegative:
        # jq saturates overflowing JSON numbers at the largest finite double,
        # so retain an intentionally generous sanity ceiling as well.
        type == "number" and . >= 0 and . < 1e100 and . == .;
    .schema == "izwi.audio-runtime-evidence.v1" and
    .status == "passed" and
    .git_sha == $git_sha and
    .backend == $backend and
    .device.selected_backend == $backend and
    (.device.identity | type == "string" and length > 0) and
    (.device.runtime_id | type == "string" and length > 0) and
    ([.models[] | {model, task}] | sort_by(.task, .model) == $expected_cases) and
    ([.models[] | [.task, .model]] | length == (unique | length)) and
    ([.models[] |
        (.task == "asr" or .task == "tts") and
        .concurrency == [1, 2, 4, 8] and
        .correctness.samples_compared > 0 and
        .correctness.mismatches == 0 and
        .fairness.requests > 0 and
        .fairness.completed == .fairness.requests and
        .fairness.starved == 0 and
        (.fairness.max_queue_wait_ms | finite_nonnegative) and
        (.fairness.queue_wait_limit_ms | finite_nonnegative) and
        .fairness.max_queue_wait_ms <= .fairness.queue_wait_limit_ms and
        .cancellation.cancelled_requests > 0 and
        .cancellation.post_cancel_outputs == 0 and
        .cancellation.live_peers_completed > 0 and
        .cancellation.retained_sessions_after == 0 and
        .cache_pressure.pressure_events > 0 and
        .cache_pressure.rejections > 0 and
        .cache_pressure.recovered_requests > 0 and
        .cache_pressure.retained_bytes_after == 0 and
        .unload_drain.attempts > 0 and
        .unload_drain.completed == .unload_drain.attempts and
        .unload_drain.active_requests_after == 0 and
        .unload_drain.retained_sessions_after == 0 and
        .fallback.c1_requests > 0 and
        .fallback.c1_completed == .fallback.c1_requests and
        .fallback.unexpected_backend_fallbacks == 0 and
        .memory.samples >= 3 and
        .memory.plateau == true and
        (.memory.growth_bytes | finite_nonnegative) and
        (.memory.tolerance_bytes | finite_nonnegative) and
        .memory.growth_bytes <= .memory.tolerance_bytes
    ] | all(. == true))
' "${evidence}" >/dev/null; then
    echo "Audio runtime evidence failed exact-SHA/device, coverage, correctness, fairness, cancellation, cache-pressure, unload/drain, fallback, or memory validation" >&2
    exit 1
fi

echo "Audio runtime evidence passed: ${evidence}"
