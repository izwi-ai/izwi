# KV cache certification benchmark

`run_kv_cache_matrix.sh` exercises the public managed-KV arena ABI without
loading a model. Its default matrix covers:

- default features on CPU, plus Metal/CUDA feature lanes when a compatible
  host device is detected;
- 16-, 32-, and 64-token physical pages;
- F32/F16/BF16 CPU reference cells and backend-supported accelerator dtypes;
- ragged short contexts and configurable long contexts;
- paged prefill and decode compared numerically with the CPU provider;
- readback-validated slot writes, page copy cycles, and page zeroing;
- offset/window/softcap and MQA shape coverage in each backend lane.

Run the supported matrix and save its JSON Lines output:

```bash
scripts/bench/run_kv_cache_matrix.sh --output target/kv-cache-matrix.jsonl
```

Run a single lane or reduce the iteration count during development:

```bash
scripts/bench/run_kv_cache_matrix.sh --lane default --iterations 3 --warmup 1
```

The harness certifies correctness before reporting a timing. It synchronizes
after every measured operation. Reported latency is
therefore dispatch-to-completion latency at the arena boundary, not an
end-to-end model latency or throughput result. JSONL records include the
observed attention provider, correctness error/tolerance, dispatches, resident
plan cache/upload counters, backing allocations, and host synchronizations.
Unavailable workspace, RSS, and VRAM measurements are encoded as JSON `null`;
they are never inferred from the selected feature or page size.

Unsupported accelerator lanes emit `status: "unsupported"` records and are not
compiled or presented as measurements. A visible accelerator can still reject
a case at runtime; that produces `status: "failed"` and a non-zero matrix exit.
The runner deliberately does not infer CUDA availability from feature
compilation alone.

Designated hardware jobs must pass `--require-device`. In that mode a missing
device or any benchmark-level `unsupported` record fails the lane:

```bash
scripts/bench/run_kv_cache_matrix.sh --lane cuda --require-device \
  --iterations 30 --warmup 5 --output target/kv-cache-cuda-certification.jsonl
```

Retained Metal/CUDA evidence must bind a clean worktree to an explicit SHA:

```bash
git_sha=$(git rev-parse HEAD)
scripts/bench/run_kv_cache_matrix.sh --lane metal --require-device \
  --expected-git-sha "$git_sha" --iterations 30 --warmup 5 \
  --output target/metal-kv-evidence/matrix.jsonl \
  --certificate target/metal-kv-evidence/certificate.json
scripts/bench/validate-gpu-evidence-certificate.sh \
  --certificate target/metal-kv-evidence/certificate.json \
  --backend metal --expected-git-sha "$git_sha"

scripts/bench/run_kv_cache_matrix.sh --lane cuda --require-device \
  --expected-git-sha "$git_sha" --iterations 30 --warmup 5 \
  --output target/cuda-kv-evidence/matrix.jsonl \
  --certificate target/cuda-kv-evidence/certificate.json
scripts/bench/validate-gpu-evidence-certificate.sh \
  --certificate target/cuda-kv-evidence/certificate.json \
  --backend cuda --expected-git-sha "$git_sha"
```

The CUDA lane builds with `--features flash-attn` (which implies `cuda`) and
exercises both the portable kill switch and enabled optimized rollout with F16
and BF16. Provider attribution comes from the arena after execution, so native
fallback is visible instead of being guessed from CLI arguments. Treat this as
a long-running production certification lane: the first CUDA/FA2 build can be
substantial and requires a compatible CUDA toolchain as well as a device. This
script does not claim or substitute CUDA measurements on hosts lacking either
prerequisite.

Validate argument routing and capability classification without compiling:

```bash
scripts/bench/test-run-kv-cache-matrix.sh
```

## Serving benchmark

`run-serving-benchmark.py` drives chat completions on the local role or the
gateway with Poisson arrivals, request-rate sweeps or a concurrency cap. It
reports TTFT, pooled per-delta ITL, TPOT, throughput, goodput under SLOs and
server-side counted metrics, and has a report-only regression profile per
backend. See [docs/dev/SERVING_BENCHMARK.md](../../docs/dev/SERVING_BENCHMARK.md).

```bash
python3 scripts/bench/run-serving-benchmark.py --model LFM2.5-1.2B-Instruct-GGUF \
  --request-rate 1,2,4,inf --num-prompts 200 --output serving.json
```

## Backend model evidence

`run-model-evidence.sh` wraps an existing strict benchmark manifest in a
versioned CPU, Metal, or CUDA certification bundle. It requires the requested
backend to be selected by the local Izwi server, zero failed quality gates,
telemetry for every case, and matching `actual_device_kind`. CUDA additionally
requires an observed NVIDIA device. Certification requires the running
server's compile-time Git SHA to match the checked-out CLI/repository SHA.

```bash
scripts/bench/run-model-evidence.sh --backend cuda \
  --manifest benchmarks/manifests/cuda-family-api.toml \
  --server http://127.0.0.1:8080 \
  --output target/cuda-model-evidence
```

Missing hardware fails by default. `--allow-unsupported` is only for local
exploration and emits an explicit unsupported certificate; hardware CI must not
use it. The runner never downloads models and rejects remote servers unless
`--allow-remote` is explicit. The family manifest covers the 17 implementations
reachable through the current chat/TTS/ASR benchmark API. Forced alignment,
diarization, and the standalone speech tokenizer require dedicated benchmark
producers before they can issue equivalent retained runtime certificates.

Use `--require-optimized-kernel-evidence` for a manifest whose every case is
expected to exercise fused attention, paged attention, or a fused RoPE path.
The protected workflow runs this stricter check separately from broad family
coverage so generic Candle CUDA execution is not mislabeled as an optimized
custom kernel.

Audio concurrency evidence uses the `cpu-audio-concurrency.toml`,
`metal-audio-concurrency.toml`, and `cuda-audio-concurrency.toml` manifests.
Each manifest covers every benchmarkable ASR and TTS family at c1/c2/c4/c8
through the SSE endpoints. The CLI records first-audio and inter-audio-chunk
latency for TTS, plus first-transcript and inter-transcript-delta latency for
ASR. Require the complete matrix with the fail-closed gate:

```bash
scripts/bench/run-model-evidence.sh --backend metal \
  --manifest benchmarks/manifests/metal-audio-concurrency.toml \
  --server http://127.0.0.1:8080 \
  --output target/metal-audio-evidence \
  --audio-runtime-evidence target/metal-audio-runtime-stress.json \
  --require-audio-streaming-evidence
```

The gate rejects a missing concurrency cell, a missing per-request first or
inter-output sample, any sample-level quality failure, or an incomplete run.
Successful benchmark traffic cannot prove destructive lifecycle properties,
so the gate also requires an externally produced `izwi.audio-runtime-evidence.v1`
stress artifact. Validate one independently with:

```bash
scripts/bench/validate-audio-runtime-evidence.sh \
  --evidence target/metal-audio-runtime-stress.json \
  --report target/metal-audio-evidence/benchmark/report.json \
  --backend metal --expected-git-sha "$(git rev-parse HEAD)"

scripts/bench/validate-gpu-evidence-certificate.sh \
  --certificate target/metal-audio-evidence/certificate.json \
  --backend metal --expected-git-sha "$(git rev-parse HEAD)" \
  --require-audio-streaming-evidence
```

The stress artifact must cover exactly the report's ASR/TTS models and prove,
per model, c1/c2/c4/c8 execution, scalar/width-one completion without an
unexpected backend fallback, output parity, bounded non-starving service,
mixed cancellation with no post-cancel publication, cache-pressure rejection
and recovery, unload/drain to zero active retained sessions, and a measured
memory plateau of at least three samples. It is intentionally external: the
runner does not synthesize cancellation, unload, pressure, or memory claims
from successful requests. The validator binds it to the clean Git SHA, selected
backend, and non-empty device/runtime identities, then retains its SHA-256 in
the model certificate.
The complete machine-readable field contract is exercised by
`scripts/bench/test-validate-audio-runtime-evidence.sh`; producers should use
that passing fixture as the minimal schema example.

A locally passing CPU report and stress artifact do not certify Metal or CUDA.
Accelerator certificates remain bound to their own exact selected device,
runtime backend, clean Git SHA, and captured telemetry. Until those external
runs exist, their result is unavailable, not passing or failing.

Continuous batching uses CPU, Metal, and CUDA concurrent manifests covering
Qwen3, Qwen3.5, Qwen3.8, LFM2, and Gemma. The
certificate rejects missing run-local multi-row continuous batches, zero work,
width below two, or physical-batch rejections:

```bash
git_sha=$(git rev-parse HEAD)
backend=cuda # use cpu or metal with the matching manifest
scripts/bench/run-model-evidence.sh --backend "$backend" \
  --manifest "benchmarks/manifests/${backend}-continuous-batching.toml" \
  --require-continuous-batch-evidence \
  --output "target/${backend}-continuous-batching-evidence"
scripts/bench/validate-gpu-evidence-certificate.sh \
  --certificate "target/${backend}-continuous-batching-evidence/certificate.json" \
  --backend "$backend" --expected-git-sha "$git_sha" \
  --require-continuous-batch-evidence
```

Chunked-prefill certification is separate because concurrency alone does not
prove that one prompt crossed a resumable safe point. Start the exact-SHA
server with chunked prefill enabled, then require every model case to commit at
least two scheduler-visible prefill quanta:

```bash
scripts/bench/run-model-evidence.sh --backend "$backend" \
  --manifest "benchmarks/manifests/${backend}-resumable-prefill.toml" \
  --require-resumable-prefill-evidence \
  --output "target/${backend}-resumable-prefill-evidence"
scripts/bench/validate-gpu-evidence-certificate.sh \
  --certificate "target/${backend}-resumable-prefill-evidence/certificate.json" \
  --backend "$backend" --expected-git-sha "$git_sha" \
  --require-resumable-prefill-evidence
```

These certificates establish runtime behavior, not universal performance.
Retain CPU, Apple Silicon, and NVIDIA before/after runs with the same model
revision, prompt matrix, sampling policy, and concurrency. Promotion requires
no quality regression and reviewed TTFT, inter-token latency, throughput,
memory, host-read, metadata-upload, batch-width, and padding deltas for the
exact hardware cell. Qwen3.8 CUDA improvements use capability-based Auto defaults with explicit
opt-outs; hardware validation remains profile-scoped; source compatibility with Qwen3.5
does not qualify them for cross-family promotion.

The ignored native CUDA GQA oracle fails if explicitly run without CUDA or if
the observed provider is not `cuda_native`:

```bash
cargo test -p izwi-core --features cuda \
  backends::kv::accelerator::tests::cuda_paged_decode_matches_cpu_for_offsets_and_gqa \
  -- --ignored --exact
```

The same workflow first runs `run-cuda-model-load-evidence.sh` against one
representative from every registered implementation family. This closes the
load-only coverage gap for forced alignment, diarization, and the standalone
speech tokenizer while keeping their evidence distinct from inference and
kernel certification.

## Required NVIDIA CUDA/KV matrix

Before promoting a CUDA provider, retain both the KV JSONL and model evidence bundle for the
exact Git SHA. At minimum cover:

- `cuda-base`, product `cuda`/FlashAttention, and `cudnn` builds;
- SM 8.0 and newer for graph/partition policy;
- F16 and BF16; page sizes 16, 32, and 64; MQA/GQA; equal 64/128/256 head
  dimensions; ragged batches; non-zero first-page offsets; windows and softcap;
- contexts immediately below, at, and above the 2,048-token partition boundary,
  then the loaded model maximum and an admission-overflow rejection;
- first eager call, graph warm/capture/replay, cancellation, arena growth, graph
  generation invalidation, and eager recovery after an injected capture error;
- dense logits/output quality, peak VRAM, host reads, dtype/layout
  copies, p50/p95 prefill and decode latency, and continuous-batch throughput.

FP8 promotion is a separate blocked project: scaled page storage, scale-aware
mutation/accounting, and numerical evidence must exist before any FP8 lane can
become selectable.

Every model case must report `actual_device_kind=cuda`, strict quality success,
no worker panic/restart/request-failure delta, and the expected observed
provider. Compile-only CI and an `unsupported` record cannot promote a runtime
cell.

## Qwen3.8 CUDA hardware-profile evidence

The shipping policy is default-on supported CUDA paths with typed opt-outs.
`IZWI_CUDA_MODE=off` and `IZWI_LOADING_MODE=off` select subsystem baselines;
empty settings exercise shipping Auto policy. The L40S is a measurement profile,
not a production device-name gate. See
[the full protocol](../../docs/dev/QWEN38_L40S_VALIDATION.md) for exact policy,
per-feature opt-outs, cross-SM scopes and lifecycle/quality requirements.
Auto uses Q8 pending measured device crossover evidence. Explicit `native_fp8`
is the software-decode W8A16 provider; true W8A8 is not implemented. The 40 t/s
hardware gate remains unmeasured.

```bash
scripts/bench/run-qwen38-l40s-evidence.sh --mtp-depth 1 \
  --server http://127.0.0.1:8080 --izwi-bin target/release/izwi \
  --output target/qwen38-l40s-default
```

Set `IZWI_ENABLE_PREFIX_CACHING=false` on the server for primary evidence. The
runner checks no measured prefix hits/reused tokens. The versioned L40S manifest
predeclares median >=40 t/s for the existing server/UI generation-time metric on
`Explain llm inference to me`, and >=40 committed decode wall t/s for sustained
single-sequence 512/2048-budget cases. Each primary case has ten measured runs;
sustained cases require actual >=384/1536 decode tokens per sample. Natural EOS
and budget termination are recorded separately. Temperature and seed are zero.

CLI `chat_timing` preserves queue, prefill, physical service, monotonic decode
wall time and committed counts. `server_request_tps` and `decode_wall_tps`
summaries include p10/p50; latencies retain p95. Certificates use
`server_request_samples` for the preserved generation denominator and
`decode_wall_samples` for actual decode timing. Old misleading `decode_samples`
are rejected. Neither speculative drafts nor SSE chunk counts are token counts.
Strict OpenAI mode omits these extensions; use relaxed mode for evidence.
The UI stream now explicitly requests true usage; its displayed formula is
unchanged. The previous direct API path could estimate tokens as text length/4
if usage was absent. Compare actual committed counts on both baseline and
candidate builds: correcting the estimate is not an inference speedup. The
reported original 16–18 t/s remains unverified until its build/counts are known.

The reusable `run-qwen38-cuda-evidence.sh --workload PATH` runner accepts other
versioned device profiles. Keep the exact-SHA CLI/server, pinned checkpoint,
selected GPU UUID/SM/driver/memory, provider and configuration checks intact.
Declare device-specific thresholds before running; L40S absolute targets do not
apply universally. `--allow-unsupported` and `--dry-run` never certify CUDA.

Collect separate MTP-disabled and depths 1/2/3 bundles with other settings fixed:

```bash
scripts/bench/certify-qwen38-mtp-evidence.sh \
  --baseline target/qwen38-mtp-disabled --depth-1 target/qwen38-mtp-depth-1 \
  --depth-2 target/qwen38-mtp-depth-2 --depth-3 target/qwen38-mtp-depth-3 \
  --output target/qwen38-mtp-paired
```

`--mtp-depth` asserts the loaded model state; it does not change server config.
The paired certifier recomputes both rates, binds matching workloads and hardware,
and requires independent absolute single-sequence gates plus relative completion,
TTFT and memory gates. `runtime_validated` remains distinct from
`performance_certified`. Portable fixture tests exercise pass/fail logic only:

```bash
bash scripts/bench/test-run-qwen38-cuda-evidence.sh
bash scripts/bench/test-run-qwen38-l40s-evidence.sh
bash scripts/bench/test-certify-qwen38-mtp-evidence.sh
```

### Timed CUDA loading

```bash
scripts/bench/run-cuda-model-load-evidence.sh \
  --manifest benchmarks/manifests/qwen38-cuda-load.txt --iterations 3 \
  --cache-state reload --cache-provenance /path/to/provenance.json \
  --server-log /path/to/server.log --output target/qwen38-load-reload
bash scripts/bench/test-run-cuda-model-load-evidence.sh
```

The family manifest now includes Qwen3.8. The timed runner checks an empty
exact-SHA CUDA server before loading, records Ready and first real request
completion with a monotonic clock, captures optional
`family_diagnostics.load_timing` phases/counters, and unloads. Conversion/upload
phase counters currently cover Q8 tiles only, not dense/raw materialization. First-request
failure or unload failure stops the cell. A supplied `--first-request` JSON body
supports a custom chat workload; Qwen3.8 has an actual short first-request default.

Source-cold/source-warm/derived-cold/derived-warm/reload are **operator-declared**
cache states requiring provenance JSON. `unknown` is the safe default. Observed
conversion-cache counters are separate from OS-cache declarations. The runner
never flushes OS caches and restarting a process does not prove a cold OS cache.
Current cached-tile lookups rehash source on each lookup for external-source
integrity; derived-cache hits must not be described as zero-I/O warm loading.
Use `--prepare-run` to externally prepare/restart each fresh-process cell and
record process identities; otherwise iterations are same-process reloads.
Retain >=3 fresh-process runs per cache cell and separately compare reloads.
The collector always reports `performance_certified:false`; no loading speedup
has been measured on the development host.

Server Ready logs retain lifecycle phases. Scheduler grant diagnostics can be
collected using `RUST_LOG=info,izwi_core::engine::scheduler=debug`; aggregate
`Decode quantum granted` events by request_id and quantum_reason to show actual
sustained grants, soft-deadline fallbacks and fairness behavior.

### Qwen3.8 KV cache precision

The FP8 checkpoint describes weight storage, not KV cache storage. Loaded-model
diagnostics must currently report `cuda_kv_storage.quantized=false` and
`physical_format=dense`: Metal uses F16 and CPU uses F32.
CUDA defaults to BF16 KV on observed compute capability 8.0+;
`IZWI_QWEN38_CUDA_BF16_KV=0` selects the F16 comparison. Unknown/older CUDA
capabilities retain F16. Runtime validation remains a separate evidence gate. FP8 KV stays
unselectable until the implementation has calibrated per-layer K/V scales,
scale-aware page mutation/accounting, fused paged prefill and decode kernels,
and retained numerical, quality, memory, and latency evidence. A standalone
dequantization pass is not an acceptable promotion path because it can erase
the bandwidth benefit.

### Automatic-length CUDA chat concurrency

`run-cuda-chat-concurrency.py` exercises the user-facing concurrency admission
contract through both `/v1/chat/completions` and independent first-party
`/v1/chat/threads/{id}/messages` conversations. Every generation omits both output
limit fields. Run against an idle, dedicated server with the model already loaded,
using a clean checkout of the **exact server binary SHA**. Qwen3.8 incremental CUDA
admission is enabled by default. Ensure `IZWI_CUDA_INCREMENTAL_CHAT` is unset or `1`;
use `0` for a conservative baseline. The resolved policy is visible in `/v1/health` at
`runtime.chat_concurrency_policy` and `/v1/metrics` at
`engine.chat_concurrency_policy`. It reports the requested flag, whether CUDA makes
it effective, the eligible `qwen38_chat` replay family, and effective scheduler
chunked prefill. These values come from startup configuration, not a new environment
read during the health request. Adapters without resumable prefill keep their existing behavior;
policy visibility alone does not certify device execution.


```bash
python3 scripts/bench/run-cuda-chat-concurrency.py \
  --model Qwen3.8-27B-FP8 --output target/cuda-chat-concurrency
python3 scripts/bench/test-cuda-chat-concurrency.py
```

The default matrix is c1, c2 and c3 on each route, followed by c3 with a late
arrival after the first two streams produce output. `--extended` additionally
requires c4 and c8, where the configured resource envelope supports them. The
late-arrival case disconnects its first client and requires surviving streams to
produce additional output. Normal cases disconnect all clients after
`--events 32` nonempty stream deltas per request; `--timeout 120` bounds each
observation and cleanup phase. These are client cancellation bounds, not hidden
`max_tokens` settings. Temporary conversations are deleted after cleanup.

A pass requires a shared interval of observed text generation and at least two
**actual model forwards** at the requested width, measured as a per-case delta
of `engine.model_tensor_batch_width_counts` (exact widths 1–64; key 0
is an overflow bucket and cannot certify any width). HTTP connection overlap, lifetime
maximum batch width, or scheduler envelope counters alone cannot pass. Stream
deltas may contain multiple tokens, so the report deliberately does not invent
token counts or tokens-per-second. Stream errors fail acceptance, and cancellation
must drain running and queued requests and restore active cache claims, page-table
ownership, reservations and execution/transfer pins before the next case begins.
Reusable prefix-cache retention is permitted.

The JSON report preserves request bodies, text-delta timestamps, loaded model
representation/diagnostics, exact SHA and runtime health, before/after metrics,
and periodic full metrics and NVIDIA memory samples. It contains no generated
text. `--allow-remote` explicitly permits another host; in that case `--nvidia-smi`
must name a wrapper that queries **the server host**, not a local unrelated GPU.
Results describe only the device/configuration observed. Run the same matrix
with candidate rollout settings and with MTP disabled and enabled as appropriate;
retain separate output directories and record server launch configuration.

This is an admission/overlap/cancellation gate, not complete numerical or memory
pressure certification. Forced replay boundaries, loss/duplication of streamed
output, long-prompt fairness, slow clients, and resource-exhaustion recovery still
require the engine tests and targeted CUDA stress runs in the implementation plan.
No production inference is run by the deterministic Python fixture tests.

## Live ASR matrix

`run-asr-matrix.sh` transcribes a fixed fixture set with every listed ASR model
through the OpenAI `/v1/audio/transcriptions` route. Each model gets its own
`izwi-server` with an isolated database, media and spool directory, so a failed
or memory-heavy model cannot contaminate the next. A footprint guard samples the
server once a second and kills it above `--guard-mb` (macOS `footprint`, which
includes unified GPU memory; resident set size elsewhere).

```bash
cargo build --release -p izwi-server --features izwi-server/metal
scripts/bench/run-asr-matrix.sh --backend metal \
    --models Parakeet-TDT-0.6B-v3,Whisper-Large-v3-Turbo \
    --fixtures fox,test,diar116,long600,silence,noise
```

Fixtures are `fox` and `test` (scored against `data/fox.md` and
`data/test.md`), `diar116` and `diar60` (two-speaker MP3 at 44.1 kHz), `long600`
(the first 10 minutes of `data/ceo.mp3`), and generated `silence` and `noise`,
which must come back empty. Derived fixtures need `ffmpeg` or `sox`. Use
`--form stream=true` to score streamed responses; the scorer then also checks
that the concatenated deltas equal the final transcript.

`score-asr-matrix.py` writes `summary.json` beside `results.jsonl` with WER,
word counts, returned duration and rtf, segment counts, the number of long-form
chunks that stopped on their token budget, and the peak footprint per model.
Run it with the same matrix before and after a change and compare the summaries;
the deterministic scorer checks live in `test-score-asr-matrix.py`.

## Live TTS matrix

`run-tts-matrix.sh` runs a fixed set of text cases for every listed TTS model
through the OpenAI `/v1/audio/speech` route, non-stream and SSE. As with the ASR
matrix, each model gets its own `izwi-server` with an isolated database, media
and spool directory and a footprint guard (`--guard-mb`). The harness also
refuses a busy port and checks that its own server owns it, and on macOS waits
for `--min-free-pct` free memory before each model.

```bash
cargo build --release -p izwi-server --features izwi-server/metal
scripts/bench/run-tts-matrix.sh --backend metal \
    --models Kokoro-82M:kokoro,Qwen3-TTS-12Hz-0.6B-CustomVoice:qwen_cv \
    --asr-base http://127.0.0.1:18090
```

Each model names its case family (`kokoro`, `lfm`, `qwen_cv`, `qwen_base`,
`qwen_greedy`, `vibevoice`, `voxtral`, `fish`). Texts are a 9-word sentence, a
45-word paragraph, a numbers sentence, a 250-word passage and a leading-dash
probe; clone families use `data/fox.wav` with `data/fox.md` as the reference.
Cases are written as WAV plus JSON under `<out>/<model>/cases/`.

`score-tts-matrix.py score` writes `summary.json` with audio-health statistics
(duration, level, peak, leading and trailing silence), the stop reason
(`X-Stop-Reason` or the SSE `audio.done` event), `decode_steps` and
`decode_tokens` (equal means the budget, not end of speech, stopped the model),
the terminal SSE event, and the peak footprint per model. With `--asr-base`,
every WAV is transcribed by a separately started ASR server (Nemotron on CPU
by default) and scored by WER. The deterministic checks live in
`test-score-tts-matrix.py`.

## Realtime voice and transcription load

`run-voice-realtime-load.py` opens N concurrent realtime sessions against a
running `izwi-server` and reports latency percentiles across every turn of
every session. In `voice` mode each session speaks a WAV utterance in real
time (20 ms frames plus trailing silence) to `/v1/voice/realtime/ws` for
`--turns` turns and records, from the server's `user_speech_end`, the time to
the first assistant audio frame, to the final transcript and to `turn_done`.
In `transcription` mode it streams the WAV to `/v1/speech-to-text/realtime/ws`
and records the first partial and the final after `session_stop`.

```bash
cargo build --release -p izwi-server --features izwi-server/metal
scripts/bench/run-voice-realtime-load.py --base-url http://127.0.0.1:8080 \
    --sessions 2 --turns 4 --audio data/fox.wav \
    --asr-model Parakeet-TDT-0.6B-v3 --text-model LFM2.5-1.2B-Instruct-GGUF \
    --tts-model Kokoro-82M \
    --metrics-url http://127.0.0.1:8080/v1/metrics/prometheus --json-out target/voice-load.json
```

With `--metrics-url` the report also carries the server's voice latency
histograms (`izwi_voice_turn_first_audio_seconds` and the ASR, reply and TTS
stages) as p50/p95 of their bucket deltas over the run. Run one warm-up
session first so model loads are not measured. It is standard-library Python
(a minimal RFC 6455 client); its tests are `test-voice-realtime-load.py`, run
in the hygiene lane.

## Metal load footprints

`measure-metal-load-footprints.sh` loads each model on Metal in its own process
and reports what the load actually allocated next to what load admission
prices for it. It loads through the lifecycle's registry dispatch, without
admission, so it can measure a model the server would refuse. It records:

- stored checkpoint bytes;
- the bytes the Metal pre-load check compares;
- the plan's peak and resident figures;
- Metal `currentAllocatedSize` growth (peak, at the load fence, and after a pool
  flush);
- process `phys_footprint` growth.

```bash
scripts/bench/measure-metal-load-footprints.sh --output target/metal-load-footprints.jsonl
scripts/bench/measure-metal-load-footprints.sh Qwen3.5-4B Granite-Speech-4.1-2B-Plus
```

Models are named by catalog name or model directory. Without names it measures
one or more models per family, leaving out Fish S2 Pro. The harness is the
ignored test `measure_metal_load_footprint` in
`crates/izwi-core/src/runtime/lifecycle/load/footprint_tests.rs`. It measures
weight loading only; per-request state is admitted and allocated later.

Set `IZWI_MEASURE_DIAG_DIR=<dir>` to also write `footprint` and `vmmap --summary`
reports from before the load and after it settles. Add `MallocStackLogging=1`
to get the live heap by allocation stack (`malloc_history -allBySize`) as well.

Process footprint on macOS includes freed large blocks that libmalloc keeps
cached. Run with `MallocLargeCache=0` to measure only live host memory.
