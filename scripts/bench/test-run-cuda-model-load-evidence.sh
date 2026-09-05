#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
runner="${repo_root}/scripts/bench/run-cuda-model-load-evidence.sh"
manifest="${repo_root}/benchmarks/manifests/cuda-family-load.txt"
tmp_dir=$(mktemp -d)
trap 'rm -rf "${tmp_dir}"' EXIT

help=$(${runner} --help)
grep -q -- '--dry-run' <<<"${help}"
grep -q 'actual_device_kind=cuda' <<<"${help}"

${runner} --manifest "${manifest}" --output "${tmp_dir}/dry" --dry-run >/dev/null
jq -e '
    .schema == "izwi.cuda-model-load-evidence.v1" and
    .status == "unsupported" and
    .reason == "dry_run" and
    (.models | length) == 21 and
    ([.models[].model] | length == (unique | length)) and
    ([.models[].model] | index("Qwen3.8-27B-FP8") != null) and
    ([.models[].model] | index("Qwen3-ForcedAligner-0.6B") != null) and
    ([.models[].model] | index("diar_streaming_sortformer_4spk-v2.1") != null) and
    ([.models[].model] | index("Qwen3-TTS-Tokenizer-12Hz") != null)
' "${tmp_dir}/dry/certificate.json" >/dev/null

if IZWI_CUDA_EVIDENCE_NVIDIA_SMI=/usr/bin/false \
    ${runner} --manifest "${manifest}" --output "${tmp_dir}/required" >/dev/null 2>&1; then
    echo "required CUDA model load evidence must fail without an NVIDIA device" >&2
    exit 1
fi

echo "CUDA model load evidence runner smoke test passed"

# Exercise real runner timing/health/first-request parsing against local tool
# fixtures. The emitted fixture certificate is never CUDA device evidence.
mkdir -p "${tmp_dir}/bin"
cat >"${tmp_dir}/bin/nvidia-smi" <<'MOCK'
#!/usr/bin/env bash
printf 'GPU 0: fixture device\n'
MOCK
cat >"${tmp_dir}/bin/curl" <<'MOCK'
#!/usr/bin/env python3
import json, os, pathlib, sys
args=sys.argv[1:]
url=next(a for a in reversed(args) if a.startswith('http'))
output=pathlib.Path(args[args.index('-o')+1])
state=pathlib.Path(os.environ['LOAD_FIXTURE_STATE'])
loaded=state.exists()
model='Qwen3.8-27B-FP8'
if url.endswith('/health'):
    data={'runtime':{'build_git_sha':os.environ['LOAD_FIXTURE_SHA'],
        'requested_backend':'cuda','requested_backend_available':True,
        'selected_backend':'cuda','compiled_backends':{'cuda':True},
        'cuda_runtime':{'driver_available':True,'device_usable':True},
        'loaded_models':([{'variant_id':model,'actual_device_kind':'cuda',
          'family_diagnostics':{'load_timing':{'discovery_ms':1.25,'conversion_ms':0,
          'upload_ms':2.5,'validation_ms':0.5,'cache_hits':3,'cache_misses':0,
          'converted_bytes':0,'uploaded_bytes':4096}}}] if loaded else [])}}
elif url.endswith('/load'):
    state.write_text('loaded'); data={}
elif url.endswith('/unload'):
    state.unlink(); data={}
elif url.endswith('/chat/completions'):
    body=json.loads(pathlib.Path(args[args.index('-d')+1][1:]).read_text())
    assert body['messages'][0]['content']=='Explain llm inference to me'
    assert body['model']==model and body['temperature']==0 and body['stream']==False
    data={'usage':{'completion_tokens':0 if os.getenv('LOAD_FIXTURE_BAD_RESPONSE') else 8},
          'choices':[{'message':{'content':'fixture output'},'finish_reason':'length'}]}
else:
    raise AssertionError(url)
output.write_text(json.dumps(data))
if '-w' in args: print('200',end='')
MOCK
chmod +x "${tmp_dir}/bin/curl" "${tmp_dir}/bin/nvidia-smi"
printf 'Qwen3.8-27B-FP8\n' >"${tmp_dir}/qwen-only.txt"
export LOAD_FIXTURE_STATE="${tmp_dir}/loaded-state"
export LOAD_FIXTURE_SHA
LOAD_FIXTURE_SHA=$(git -C "${repo_root}" rev-parse HEAD)
PATH="${tmp_dir}/bin:${PATH}" ${runner} --manifest "${tmp_dir}/qwen-only.txt" \
    --iterations 3 --output "${tmp_dir}/timed" >/dev/null
jq -e '.status == "passed" and .performance_certified == false and
  (.models | length) == 3 and
  ([.models[] | .timing.load_ready_ms > 0 and .timing.first_request_ms > 0 and
    .timing.first_request_ready_ms > (.timing.load_ready_ms + .timing.first_request_ms) and
    .cache_state.declared == "unknown" and .cache_state.conversion_observation == "hit" and
    .cache_state.os_cache_flushed_by_runner == false and
    .loader_diagnostics.uploaded_bytes == 4096] | all)' "${tmp_dir}/timed/certificate.json" >/dev/null
[[ ! -e "${LOAD_FIXTURE_STATE}" ]]
if PATH="${tmp_dir}/bin:${PATH}" LOAD_FIXTURE_BAD_RESPONSE=1 ${runner} \
    --manifest "${tmp_dir}/qwen-only.txt" --output "${tmp_dir}/bad-response" >/dev/null 2>&1; then
    echo 'Load runner must reject a failed first request' >&2; exit 1
fi
jq -e '.status == "failed" and .models[0].reason == "first_request_failed"' "${tmp_dir}/bad-response/certificate.json" >/dev/null
[[ ! -e "${LOAD_FIXTURE_STATE}" ]]
if ${runner} --cache-state source-cold --dry-run --output "${tmp_dir}/false-cold" >/dev/null 2>&1; then
    echo 'Cold OS-cache claims require provenance' >&2; exit 1
fi
printf '{"os_cache":"operator prepared cold source","process":"fresh externally started"}\n' >"${tmp_dir}/provenance.json"
PATH="${tmp_dir}/bin:${PATH}" ${runner} --manifest "${tmp_dir}/qwen-only.txt" \
    --cache-state source-cold --cache-provenance "${tmp_dir}/provenance.json" \
    --output "${tmp_dir}/declared" >/dev/null
jq -e '.models[0].cache_state.declared == "source-cold" and
    .models[0].cache_state.source == "operator_declared" and
    .models[0].cache_state.conversion_observation == "hit"' "${tmp_dir}/declared/certificate.json" >/dev/null
echo 'Timed load runner fixtures passed; cache declarations remain distinct from observed conversion counters'
