#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
manifest="${repo_root}/benchmarks/manifests/cuda-family-load.txt"
server="http://127.0.0.1:8080"
output="${repo_root}/target/cuda-model-load-evidence"
allow_remote=0
dry_run=0
iterations=1
cache_state=unknown
cache_provenance=""
prepare_run=""
server_log=""
first_request_json=""
nvidia_smi="${IZWI_CUDA_EVIDENCE_NVIDIA_SMI:-nvidia-smi}"

usage() {
    cat <<'EOF'
Usage: scripts/bench/run-cuda-model-load-evidence.sh [options]

Options:
  --manifest PATH   Newline-delimited model ids (default: CUDA family load manifest)
  --server URL      Local CUDA server URL (default: http://127.0.0.1:8080)
  --output PATH     Evidence directory
  --allow-remote    Permit a non-loopback server explicitly
  --iterations N    Load/unload runs per model (default: 1; use >=3 for evidence)
  --cache-state S   unknown|source-cold|source-warm|derived-cold|derived-warm|reload
  --cache-provenance PATH  Operator JSON recording OS/cache preparation and process identity
  --prepare-run PATH  Executable called with model and run index before each run;
                      use to prepare caches/restart the exact-SHA server externally
  --first-request PATH  Chat JSON body; model is replaced for each selected model
                        Qwen3.8 defaults to the reported prompt with 8 output tokens
  --server-log PATH  Retain lifecycle Ready phase log alongside health diagnostics
  --dry-run         Validate and print the model plan without HTTP or CUDA access
  --help            Show this help

The required path fails closed unless health proves selected CUDA, a usable
device, and a build Git SHA equal to the checked-out repository SHA. Every
model must load, report actual_device_kind=cuda, and unload successfully.
Timing uses a monotonic clock. Ready and first-request completion are separate.
Cache state is operator-declared, never inferred from process restart. This
runner does not flush the OS cache and does not certify a loading speedup.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) manifest="$2"; shift 2 ;;
        --server) server="$2"; shift 2 ;;
        --output) output="$2"; shift 2 ;;
        --allow-remote) allow_remote=1; shift ;;
        --iterations) iterations="$2"; shift 2 ;;
        --cache-state) cache_state="$2"; shift 2 ;;
        --cache-provenance) cache_provenance="$2"; shift 2 ;;
        --prepare-run) prepare_run="$2"; shift 2 ;;
        --first-request) first_request_json="$2"; shift 2 ;;
        --server-log) server_log="$2"; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "${iterations}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid iterations" >&2; exit 2; }
case "${cache_state}" in unknown|source-cold|source-warm|derived-cold|derived-warm|reload) ;; *) echo "Invalid cache state" >&2; exit 2 ;; esac
if [[ "${cache_state}" != unknown && ! -s "${cache_provenance}" ]]; then
    echo "Declared cache states require --cache-provenance JSON; restart alone does not prove OS cache state" >&2
    exit 2
fi
[[ -z "${prepare_run}" || -x "${prepare_run}" ]] || { echo "--prepare-run must be executable" >&2; exit 2; }
command -v python3 >/dev/null || { echo "Missing required command: python3" >&2; exit 1; }
monotonic_ns() { python3 -c 'import time; print(time.monotonic_ns())'; }
command -v jq >/dev/null || { echo "Missing required command: jq" >&2; exit 1; }
[[ -f "${manifest}" ]] || { echo "Model load manifest not found: ${manifest}" >&2; exit 1; }

models=()
while IFS= read -r model; do
    models+=("${model}")
done < <(sed -e 's/[[:space:]]*#.*$//' -e '/^[[:space:]]*$/d' "${manifest}")
if [[ "${#models[@]}" -eq 0 ]]; then
    echo "Model load manifest is empty" >&2
    exit 1
fi
if [[ "$(printf '%s\n' "${models[@]}" | sort -u | wc -l | tr -d ' ')" -ne "${#models[@]}" ]]; then
    echo "Model load manifest contains duplicate model ids" >&2
    exit 1
fi

mkdir -p "${output}"
certificate="${output}/certificate.json"
results="${output}/results.jsonl"
: >"${results}"
if [[ -n "${cache_provenance}" ]]; then
    jq -e 'type == "object"' "${cache_provenance}" >/dev/null
    cp "${cache_provenance}" "${output}/cache-provenance.json"
fi

if [[ "${dry_run}" -eq 1 ]]; then
    printf '%s\n' "${models[@]}" | jq -R . | jq -s \
        --arg manifest "${manifest}" \
        '{schema:"izwi.cuda-model-load-evidence.v1",status:"unsupported",reason:"dry_run",manifest:$manifest,models:map({model:.,status:"planned"})}' \
        >"${certificate}"
    jq -c . "${certificate}"
    exit 0
fi

case "${server}" in
    http://127.0.0.1:*|http://localhost:*|http://[::1]:*) ;;
    *)
        if [[ "${allow_remote}" -ne 1 ]]; then
            echo "CUDA load evidence requires a loopback server unless --allow-remote is explicit" >&2
            exit 1
        fi
        ;;
esac

if ! command -v "${nvidia_smi}" >/dev/null 2>&1 || ! "${nvidia_smi}" -L >/dev/null 2>&1; then
    echo "CUDA model load evidence requires a device visible to nvidia-smi" >&2
    exit 1
fi
command -v curl >/dev/null || { echo "Missing required command: curl" >&2; exit 1; }

git_sha=$(git -C "${repo_root}" rev-parse --verify HEAD)
health="${output}/health.json"
curl -fsS "${server%/}/v1/health" -o "${health}"
jq -e --arg sha "${git_sha}" '
    .runtime.build_git_sha == $sha and
    .runtime.requested_backend == "cuda" and
    .runtime.requested_backend_available == true and
    .runtime.selected_backend == "cuda" and
    .runtime.compiled_backends.cuda == true and
    .runtime.cuda_runtime.driver_available == true and
    .runtime.cuda_runtime.device_usable == true
' "${health}" >/dev/null || {
    echo "Server health does not prove the checked-out usable CUDA build" >&2
    exit 1
}

overall="passed"
for index in "${!models[@]}"; do
  model="${models[$index]}"
  encoded=$(jq -rn --arg value "${model}" '$value|@uri')
  for ((run=1; run<=iterations; run++)); do
    if [[ -n "${prepare_run}" ]]; then "${prepare_run}" "${model}" "${run}"; fi
    curl -fsS "${server%/}/v1/health" -o "${health}"
    # Never hide an already resident model in a load timing or unload another user model.
    jq -e --arg sha "${git_sha}" '.runtime.build_git_sha == $sha and
      .runtime.selected_backend == "cuda" and (.runtime.loaded_models | length) == 0' "${health}" >/dev/null || {
        echo "Each measured load requires an empty exact-SHA CUDA server" >&2; exit 1;
    }
    stem="$(printf '%02d-%02d' "$((index + 1))" "${run}")"
    load_body="${output}/${stem}-load.json"
    unload_body="${output}/${stem}-unload.json"
    ready_health="${output}/${stem}-ready-health.json"
    request_body="${output}/${stem}-first-request.json"
    response_body="${output}/${stem}-first-response.json"
    started_ns=$(monotonic_ns)
    load_code=$(curl -sS --max-time 1800 -o "${load_body}" -w '%{http_code}' -X POST \
        "${server%/}/v1/admin/models/${encoded}/load" || true)
    status="failed"
    reason="load_http_${load_code}"
    observed_device=""
    ready_ns=null
    first_started_ns=null
    first_done_ns=null
    load_diagnostics=null

    if [[ "${load_code}" =~ ^2[0-9][0-9]$ ]]; then
        curl -fsS "${server%/}/v1/health" -o "${ready_health}"
        observed_device=$(jq -r --arg model "${model}" \
            '[.runtime.loaded_models[] | select(.variant_id == $model)][0].actual_device_kind // empty' \
            "${ready_health}")
        if [[ "${observed_device}" == "cuda" ]]; then
            ready_ns=$(monotonic_ns)
            status="passed"
            reason=""
            load_diagnostics=$(jq -c --arg model "${model}" \
                '[.runtime.loaded_models[] | select(.variant_id == $model)][0].family_diagnostics.load_timing // null' "${ready_health}")
            if [[ -n "${first_request_json}" ]]; then
                jq --arg model "${model}" '.model=$model | .stream=false' "${first_request_json}" >"${request_body}"
            elif [[ "${model}" == Qwen3.8-27B-FP8 ]]; then
                jq -n --arg model "${model}" '{model:$model,messages:[{role:"user",content:"Explain llm inference to me"}],max_completion_tokens:8,temperature:0,seed:0,stream:false}' >"${request_body}"
            fi
            if [[ -s "${request_body}" ]]; then
                first_started_ns=$(monotonic_ns)
                first_code=$(curl -sS --max-time 600 -o "${response_body}" -w '%{http_code}' \
                    -H 'Content-Type: application/json' -d @"${request_body}" "${server%/}/v1/chat/completions" || true)
                first_done_ns=$(monotonic_ns)
                if [[ ! "${first_code}" =~ ^2[0-9][0-9]$ ]] || ! jq -e '(.usage.completion_tokens // 0) > 0 and (.choices | length) > 0' "${response_body}" >/dev/null; then
                    status="failed"
                    reason="first_request_failed"
                fi
            fi
        else
            reason="loaded_model_did_not_report_actual_cuda"
        fi
        unload_code=$(curl -sS --max-time 1800 -o "${unload_body}" -w '%{http_code}' -X POST \
            "${server%/}/v1/admin/models/${encoded}/unload" || true)
        if [[ ! "${unload_code}" =~ ^2[0-9][0-9]$ ]]; then
            status="failed"
            reason="unload_http_${unload_code}"
        fi
    fi

    if [[ "${status}" != "passed" ]]; then overall="failed"; fi
    jq -cn \
        --arg model "${model}" --argjson run "${run}" \
        --arg status "${status}" --arg reason "${reason}" \
        --arg actual_device_kind "${observed_device}" \
        --arg load_response "$(basename "${load_body}")" \
        --arg unload_response "$(basename "${unload_body}")" \
        --arg cache_state "${cache_state}" --argjson phases "${load_diagnostics}" \
        --argjson started "${started_ns}" --argjson ready "${ready_ns}" \
        --argjson first_started "${first_started_ns}" --argjson first_done "${first_done_ns}" \
        '{model:$model,run:$run,status:$status,reason:$reason,actual_device_kind:$actual_device_kind,
          load_response:$load_response,unload_response:$unload_response,
          cache_state:{declared:$cache_state,source:"operator_declared",os_cache_flushed_by_runner:false,
            conversion_observation:(if $phases == null then "unobserved"
              elif ($phases.cache_hits // 0)>0 and ($phases.cache_misses // 0)>0 then "partial_hit"
              elif ($phases.cache_hits // 0)>0 then "hit"
              elif ($phases.cache_misses // 0)>0 then "miss" else "unobserved" end)},
          timing:{clock:"monotonic",boundary:"before_load_http_to_ready_health_observed",
            load_ready_ms:(if $ready == null then null else ($ready-$started)/1000000 end),
            first_request_ms:(if $first_done == null then null else ($first_done-$first_started)/1000000 end),
            first_request_ready_ms:(if $first_done == null then null else ($first_done-$started)/1000000 end)},
          loader_diagnostics:$phases}' >>"${results}"
    # A failed unload/load can retain unknown state. Stop rather than contaminate later runs.
    if [[ "${status}" != passed ]]; then break; fi
  done
  if [[ "${overall}" != passed ]]; then break; fi
done
if [[ -n "${server_log}" ]]; then cp "${server_log}" "${output}/server-load-phases.log"; fi

jq -s \
    --arg status "${overall}" \
    --arg git_sha "${git_sha}" \
    --arg manifest "${manifest}" \
    '{schema:"izwi.cuda-model-load-evidence.v1",status:$status,git_sha:$git_sha,manifest:$manifest,performance_certified:false,models:.}' \
    "${results}" >"${certificate}"

if [[ "${overall}" != "passed" ]]; then
    jq -c '.models[] | select(.status != "passed")' "${certificate}" >&2
    exit 1
fi

jq -c '{schema,status,git_sha,models:(.models|length)}' "${certificate}"
