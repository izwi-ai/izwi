#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
validator="${repo_root}/scripts/bench/validate-audio-runtime-evidence.sh"
tmp_dir=$(mktemp -d)
trap 'rm -rf "${tmp_dir}"' EXIT

jq -n '{reports: [
  {report: {command: "asr", config: {model: "asr-model"}}},
  {report: {command: "tts", config: {model: "tts-model"}}}
]}' >"${tmp_dir}/report.json"

jq -n '
  def row($model; $task): {
    model: $model, task: $task, concurrency: [1, 2, 4, 8],
    correctness: {samples_compared: 8, mismatches: 0},
    fairness: {requests: 8, completed: 8, starved: 0, max_queue_wait_ms: 12.0},
    cancellation: {cancelled_requests: 1, post_cancel_outputs: 0, live_peers_completed: 1, retained_sessions_after: 0},
    cache_pressure: {pressure_events: 1, rejections: 1, recovered_requests: 1, retained_bytes_after: 0},
    unload_drain: {attempts: 1, completed: 1, active_requests_after: 0, retained_sessions_after: 0},
    fallback: {c1_requests: 1, c1_completed: 1, unexpected_backend_fallbacks: 0},
    memory: {samples: 4, plateau: true, growth_bytes: 1024, tolerance_bytes: 2048}
  };
  {
    schema: "izwi.audio-runtime-evidence.v1", status: "passed",
    git_sha: "deadbeef", backend: "cpu",
    device: {selected_backend: "cpu", identity: "host:test", runtime_id: "run-1"},
    models: [row("asr-model"; "asr"), row("tts-model"; "tts")]
  }
' >"${tmp_dir}/valid.json"

"${validator}" --evidence "${tmp_dir}/valid.json" --report "${tmp_dir}/report.json" \
    --backend cpu --expected-git-sha deadbeef >/dev/null

for mutation in cancellation memory fallback coverage duplicate concurrency task nonfinite device sha; do
    case "${mutation}" in
        cancellation) filter='.models[0].cancellation.post_cancel_outputs = 1' ;;
        memory) filter='.models[0].memory.plateau = false' ;;
        fallback) filter='.models[0].fallback.unexpected_backend_fallbacks = 1' ;;
        coverage) filter='.models = [.models[0]]' ;;
        duplicate) filter='.models += [.models[0]]' ;;
        concurrency) filter='.models[0].concurrency = [1, 4, 2, 8]' ;;
        task) filter='.models[0].task = "tts"' ;;
        nonfinite) filter='.models[0].fairness.max_queue_wait_ms = 1.7976931348623157e+308' ;;
        device) filter='.device.selected_backend = "metal"' ;;
        sha) filter='.git_sha = "other"' ;;
    esac
    jq "${filter}" "${tmp_dir}/valid.json" >"${tmp_dir}/${mutation}.json"
    if "${validator}" --evidence "${tmp_dir}/${mutation}.json" \
        --report "${tmp_dir}/report.json" --backend cpu \
        --expected-git-sha deadbeef >/dev/null 2>&1; then
        echo "${mutation} mutation must fail closed" >&2
        exit 1
    fi
done

echo "Audio runtime evidence validator tests passed"
