#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
manifest="${repo_root}/benchmarks/manifests/cuda-family-load.txt"
server="http://127.0.0.1:8080"
output="${repo_root}/target/cuda-model-load-evidence"
allow_remote=0
dry_run=0
nvidia_smi="${IZWI_CUDA_EVIDENCE_NVIDIA_SMI:-nvidia-smi}"

usage() {
    cat <<'EOF'
Usage: scripts/bench/run-cuda-model-load-evidence.sh [options]

Options:
  --manifest PATH   Newline-delimited model ids (default: CUDA family load manifest)
  --server URL      Local CUDA server URL (default: http://127.0.0.1:8080)
  --output PATH     Evidence directory
  --allow-remote    Permit a non-loopback server explicitly
  --dry-run         Validate and print the model plan without HTTP or CUDA access
  --help            Show this help

The required path fails closed unless health proves selected CUDA, a usable
device, and a build Git SHA equal to the checked-out repository SHA. Every
model must load, report actual_device_kind=cuda, and unload successfully.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) manifest="$2"; shift 2 ;;
        --server) server="$2"; shift 2 ;;
        --output) output="$2"; shift 2 ;;
        --allow-remote) allow_remote=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

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
    load_body="${output}/$(printf '%02d' "$((index + 1))")-load.json"
    unload_body="${output}/$(printf '%02d' "$((index + 1))")-unload.json"
    load_code=$(curl -sS -o "${load_body}" -w '%{http_code}' -X POST \
        "${server%/}/v1/admin/models/${encoded}/load" || true)
    status="failed"
    reason="load_http_${load_code}"
    observed_device=""

    if [[ "${load_code}" =~ ^2[0-9][0-9]$ ]]; then
        curl -fsS "${server%/}/v1/health" -o "${health}"
        observed_device=$(jq -r --arg model "${model}" \
            '[.runtime.loaded_models[] | select(.variant_id == $model)][0].actual_device_kind // empty' \
            "${health}")
        if [[ "${observed_device}" == "cuda" ]]; then
            status="passed"
            reason=""
        else
            reason="loaded_model_did_not_report_actual_cuda"
        fi
        unload_code=$(curl -sS -o "${unload_body}" -w '%{http_code}' -X POST \
            "${server%/}/v1/admin/models/${encoded}/unload" || true)
        if [[ ! "${unload_code}" =~ ^2[0-9][0-9]$ ]]; then
            status="failed"
            reason="unload_http_${unload_code}"
        fi
    fi

    if [[ "${status}" != "passed" ]]; then
        overall="failed"
    fi
    jq -cn \
        --arg model "${model}" \
        --arg status "${status}" \
        --arg reason "${reason}" \
        --arg actual_device_kind "${observed_device}" \
        --arg load_response "$(basename "${load_body}")" \
        --arg unload_response "$(basename "${unload_body}")" \
        '{model:$model,status:$status,reason:$reason,actual_device_kind:$actual_device_kind,load_response:$load_response,unload_response:$unload_response}' \
        >>"${results}"
done

jq -s \
    --arg status "${overall}" \
    --arg git_sha "${git_sha}" \
    --arg manifest "${manifest}" \
    '{schema:"izwi.cuda-model-load-evidence.v1",status:$status,git_sha:$git_sha,manifest:$manifest,models:.}' \
    "${results}" >"${certificate}"

if [[ "${overall}" != "passed" ]]; then
    jq -c '.models[] | select(.status != "passed")' "${certificate}" >&2
    exit 1
fi

jq -c '{schema,status,git_sha,models:(.models|length)}' "${certificate}"
