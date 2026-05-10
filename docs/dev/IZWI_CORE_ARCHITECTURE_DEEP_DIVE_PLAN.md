# izwi-core Architecture Deep Dive And Improvement Plan

Date: 2026-05-10

Status: Planning only

## Scope

This document reviews `crates/izwi-core` after the current
`architecture-improvements-2` branch. It focuses on organizational layout,
architectural boundaries, runtime functionality, extension points, and a phased
plan for making the crate easier to maintain and expand without regressing TTS,
ASR, chat, speech-to-speech, realtime voice, diarization, model lifecycle, or
metrics behavior.

No runtime code changes are part of this phase.

## External Engine Lessons

### vLLM

Useful source material:

- [Architecture overview](https://docs.vllm.ai/en/latest/design/arch_overview/)
- [V1 guide](https://docs.vllm.ai/en/latest/usage/v1_guide/)
- [Model registration](https://docs.vllm.ai/en/latest/contributing/model/registration/)
- [Metrics design](https://docs.vllm.ai/en/stable/design/metrics/)

Relevant lessons:

- Keep transport/input processing separate from engine core. vLLM splits API
  server work from engine core scheduling/KV management and GPU worker execution.
- Keep the engine core narrow. Scheduler, KV cache, model runner, and worker
  execution are explicit boundaries, not mixed into API routes.
- Prefer one cohesive scheduler/cache architecture over capability-specific
  scheduling paths. vLLM V1's direction is explicitly toward a simpler,
  modular, easy-to-hack core.
- Make model extensibility a registry problem. Built-in models live in a known
  model executor directory, and external models can register through plugin
  hooks instead of changing unrelated serving code.
- Treat metrics as a production contract. vLLM separates server-level metrics
  from request-level SLO metrics such as TTFT, inter-token latency, queue state,
  prefill, decode, and end-to-end latency.

### SGLang

Useful source material:

- [Model Gateway / Router](https://docs.sglang.io/advanced_features/router.html)
- [How to support new models](https://docs.sglang.io/supported_models/support_new_models.html)
- [Production metrics](https://docs.sglang.io/docs/references/production_metrics)
- [Production request tracing](https://sgl-project.github.io/references/production_request_trace.html)

Relevant lessons:

- Separate local runtime from gateway mode. SGLang has router/gateway concepts
  for worker lifecycle, health, routing, and load balancing; those concerns do
  not need to pollute the local engine API.
- Make health and circuit-breaker state first-class when gateway mode exists.
  Worker readiness should be capability-aware and model-aware.
- Model support should have a small contributor workflow. For standard models,
  SGLang's model path is intentionally close to "add one model file, register
  it, test it."
- Production observability should include request traces and replay/debug
  affordances, not just counters.
- Prefix/cache-aware routing and advanced cache features are runtime concerns,
  not route concerns.

### llama.cpp

Useful source material:

- [llama.cpp README](https://github.com/ggml-org/llama.cpp)
- [Build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)

Relevant lessons:

- Keep a small stable library surface. llama.cpp's main product is the core
  `llama` library; tools and servers sit on top.
- Keep hardware backends isolated behind a compute/backend layer. CPU, Metal,
  CUDA, Vulkan, and other backends are selected without changing higher-level
  serving interfaces.
- Make local/simple deployment excellent. Server/router features are layered on
  top of a portable inference library rather than required for basic use.
- Keep model files, context, batch/decode, KV memory, and backend graph
  execution conceptually separate.

## Current izwi-core Shape

Current top-level layout:

```text
crates/izwi-core/src/
  artifacts/          compatibility re-export for model artifact types
  audio/              audio encoding/decoding/preprocessing helpers
  backends/           device probing, backend selection, CUDA/private runtime
  catalog/            model identity, metadata, capability flags, parsing
  codecs/             placeholder codec namespace
  config.rs           runtime/engine configuration
  engine/             scheduler, KV cache, executor, request, output
  error.rs            core error type
  kernels/            low-level CPU/Metal/CUDA kernel helpers
  model/              legacy model metadata/artifact/manager surface
  models/             concrete model implementations and loaded registry
  runtime/            RuntimeService facade and capability orchestration
  runtime_models/     public alias over internal models namespace
  serve_runtime.rs    server runtime configuration
  tokenizer.rs        tokenizer wrapper
```

Largest local pressure points by approximate line count:

| File | Role | Concern |
| --- | --- | --- |
| `models/architectures/sortformer/diarization/mod.rs` | Sortformer implementation | Large family implementation with pipeline-specific behavior. |
| `runtime/diarization.rs` | Diarization transcript orchestration | Pipeline/DAG logic lives inside one runtime extension file. |
| `models/architectures/qwen3/asr/mod.rs` | Qwen ASR/aligner | Large capability-specific model body. |
| `models/architectures/qwen3/tts/mod.rs` | Qwen TTS | Large model body plus generation behavior. |
| `models/architectures/qwen35/text.rs` and `chat.rs` | Qwen3.5 chat | Large model-specific runtime logic. |
| `engine/scheduler.rs` | Scheduler policy | Many policy ideas in one module. |
| `engine/kv_cache.rs` | KV cache manager | Cache allocation, prefix reuse, residency, streaming types in one file. |
| `model/download.rs` | Artifact download | Download, progress, cache, HTTP, cancellation in one file. |
| `models/registry.rs` | Loaded native model registry | Loader registration, runtime enums, generation methods, lookup/unload all in one module. |
| `runtime/service.rs` | Runtime facade | Facade plus telemetry, worker loop, broker observation, lifecycle access. |

## Current Architectural Boundaries

### What Improved On This Branch

- `runtime/request.rs` now gives task-specific wrappers before broad engine
  requests are built.
- `runtime/adapters.rs` has capability metadata and execution target metadata.
- `runtime/broker.rs` can observe and validate engine requests in shadow/on
  modes.
- `model/residency.rs` and `model/manager.rs` now enforce active model leases.
- `engine/request.rs` carries stream policy into engine execution.
- `engine/metrics.rs` exposes scheduler/KV/stream metric names and counters.
- `runtime/pipeline.rs` defines voice and diarization graph shapes, and the live
  voice/diarization paths now record graph telemetry.

### Remaining Boundary Problems

1. `RuntimeService` is still too broad.

   It owns public facade methods, model lifecycle access, telemetry storage,
   worker loop management, broker observation, direct registry access, audio
   codec state, loaded TTS state, and request execution. This makes it hard to
   see which component owns a bug.

2. `EngineCoreRequest` is still an optional-field union.

   The new typed wrappers validate inputs before engine conversion, but the core
   engine still receives one broad struct with many optional fields. That makes
   invalid states representable and pushes task interpretation into executor
   handlers.

3. Capability adapters are metadata-only.

   The registry can say a model supports `tts`, `asr`, `chat`, etc., but it does
   not yet own validation, loading requirements, request normalization,
   execution, stream behavior, output typing, or conformance fixtures.

4. Direct model access remains scattered.

   `runtime/asr.rs`, `runtime/chat.rs`, `runtime/tts.rs`,
   `runtime/kokoro.rs`, `runtime/diarization.rs`, `runtime/service.rs`, and
   `engine/executor/**` all touch the model registry or concrete models. This
   weakens scheduling, cancellation, residency, and telemetry consistency.

5. The model registry mixes several jobs.

   `models/registry.rs` currently handles loader registration, loaded handle
   storage, native capability enums, per-family generation/transcription helper
   methods, and unload paths. This is the biggest contributor-friction point for
   adding models.

6. Pipeline graphs are observable but not executable.

   Voice and diarization graph contracts are now recorded, but graph stages do
   not yet drive execution, cancellation, stage metrics, or retry/fallback
   behavior.

7. Engine organization is close, but submodule ownership is muddy.

   Scheduler, KV cache, stream sinks, output processing, executor dispatch, and
   stateful decode live under `engine/`, but the boundaries are still file-level
   rather than concept-level. `engine/kv_cache.rs` and `engine/scheduler.rs` are
   large enough that future policy work will become risky.

8. Tests are mostly in-module.

   In-module tests are useful, but there is no `crates/izwi-core/tests/`
   conformance suite that verifies capability contracts end-to-end with fake or
   lightweight adapters.

## Target Design Principles

1. `RuntimeService` should become a compatibility facade, not the runtime
   implementation.
2. Every inference request should be represented by a typed capability payload,
   not optional fields.
3. The broker should route capability requests to execution targets, not only
   observe engine requests.
4. Adapters should be executable contracts: validate, load, execute, stream,
   label metrics, and declare fixtures.
5. Model loading and model execution should use separate registries.
6. Pipelines should be reusable graphs with stage-level typed inputs, outputs,
   cancellation, and telemetry.
7. Engine core should stay capability-light: schedule work, manage memory,
   execute runners, and publish outputs.
8. Hardware/backend logic should stay below model/capability code.
9. Public compatibility surfaces should remain thin adapters over runtime
   commands.
10. Every reorg phase should have a compile and conformance gate before behavior
    changes.

## Proposed Organizational Layout

This is the target folder shape inside `crates/izwi-core/src`. It should be
migrated in behavior-preserving phases, not in one large move.

```text
crates/izwi-core/src/
  lib.rs
  error.rs
  config/
    mod.rs
    runtime.rs
    engine.rs

  catalog/
    mod.rs
    variant.rs
    metadata.rs
    capabilities.rs
    aliases.rs

  artifacts/
    mod.rs
    repository.rs
    downloader.rs
    manifest.rs
    progress.rs

  residency/
    mod.rs
    controller.rs
    lease.rs
    policy.rs
    snapshot.rs

  runtime/
    mod.rs
    service.rs            # thin compatibility facade
    broker/
      mod.rs
      request.rs          # InferenceRequest enum + envelope
      response.rs         # InferenceOutput enum + stream event enum
      admission.rs
      router.rs
      cancellation.rs
      deployment.rs
    capabilities/
      mod.rs
      adapter.rs          # executable adapter trait
      registry.rs
      conformance.rs
      tts.rs
      asr.rs
      chat.rs
      audio_chat.rs
      speech_to_speech.rs
      diarization.rs
      forced_alignment.rs
      vad.rs
      endpointing.rs
      tokenizer.rs
    pipelines/
      mod.rs
      graph.rs
      executor.rs
      stages.rs
      voice.rs
      diarization.rs
    sessions/
      mod.rs
      voice.rs
      realtime_asr.rs
    telemetry/
      mod.rs
      metrics.rs
      tracing.rs
      replay.rs
    streams/
      mod.rs
      sink.rs
      policy.rs
      events.rs

  engine/
    mod.rs
    core/
      mod.rs
      loop.rs
      state.rs
      timing.rs
    request/
      mod.rs
      task.rs             # EngineTask enum, no broad optional union
      processor.rs
    scheduler/
      mod.rs
      queue.rs
      policy.rs
      budget.rs
      preemption.rs
      stats.rs
    cache/
      mod.rs
      kv.rs
      prefix.rs
      residency.rs
      metal.rs
      stats.rs
    executor/
      mod.rs
      runner.rs
      dispatch.rs
      handlers/
        tts.rs
        asr.rs
        chat.rs
        audio_chat.rs
      decode_state.rs
    output/
      mod.rs
      processor.rs
      streaming.rs
    metrics.rs
    signal_frontend.rs

  models/
    mod.rs
    registry/
      mod.rs
      loader.rs
      handles.rs
      factories.rs
    shared/
      attention/
      audio/
      chat.rs
      config.rs
      memory/
      telemetry.rs
      weights/
    families/
      qwen3/
        mod.rs
        chat.rs
        asr.rs
        forced_alignment.rs
        tts/
      qwen35/
      gemma3/
      lfm2/
      lfm25_audio/
      kokoro/
      parakeet/
      whisper/
      sortformer/
      voxtral/

  backends/
    mod.rs
    device.rs
    router.rs
    policy.rs
    model_io.rs
    kernels.rs
    cuda_runtime.rs

  audio/
  kernels/
  tokenizer.rs
```

### Why This Shape

- `catalog/`, `artifacts/`, and `residency/` become three distinct concerns:
  identity, disk state, and loaded-memory/device state.
- `runtime/broker/` becomes the single runtime entrypoint for typed work.
- `runtime/capabilities/` becomes the contributor-facing extension surface.
- `runtime/pipelines/` becomes the reusable DAG layer for voice and diarization.
- `runtime/telemetry/` isolates metrics/tracing/replay from orchestration.
- `engine/` becomes scheduler/cache/executor infrastructure rather than a
  product-aware inference router.
- `models/families/` names what the folder contains better than
  `models/architectures/` now that each family includes tokenizers,
  preprocessors, loaders, configs, and runtime helpers.

## Target Runtime Flow

```text
Transport/API/CLI/UI
  -> RuntimeService facade
    -> broker::InferenceRequest
      -> admission + deadline + cancellation
      -> capability registry
        -> adapter.validate()
        -> residency controller acquire/ensure_ready()
        -> execution target
          -> EngineCore for token/streaming scheduled work
          -> BatchRunner for ASR/diarization-style jobs
          -> PipelineExecutor for multi-stage DAGs
          -> DirectRunner only for explicitly non-schedulable local helpers
      -> typed InferenceOutput or RuntimeStreamEvent
```

### Target Capability Adapter Contract

The current `ModelCapabilityAdapter` should evolve from metadata-only into an
executable contract. Conceptually:

```rust
trait CapabilityAdapter {
    fn metadata(&self) -> CapabilityMetadata;
    fn artifact_requirements(&self, model: ModelVariant) -> ArtifactRequirements;
    fn backend_requirements(&self, model: ModelVariant) -> BackendRequirements;
    fn validate(&self, request: &InferenceRequest) -> Result<()>;
    async fn prepare(&self, ctx: &RuntimeContext, model: ModelVariant) -> Result<ModelLease>;
    async fn execute(&self, ctx: &RuntimeContext, request: InferenceRequest)
        -> Result<InferenceOutput>;
    async fn stream(&self, ctx: &RuntimeContext, request: InferenceRequest, sink: StreamSink)
        -> Result<InferenceOutput>;
    fn conformance_cases(&self) -> &'static [ConformanceCase];
}
```

Implementation should stay concrete Rust rather than dynamic plugin loading for
now. Out-of-tree registration can come later if contributor needs justify it.

## Functional Improvements By Capability

### TTS

Current state:

- Qwen TTS flows through typed runtime request into engine.
- Kokoro and LFM2.5 audio TTS still use direct runtime/model paths.

Target:

- `runtime/capabilities/tts.rs` owns all TTS validation and execution choices.
- Qwen TTS uses `EngineCore`.
- Kokoro can use a `BatchRunner` or `DirectRunner`, but through the same adapter
  and residency/telemetry path.
- LFM2.5 audio TTS should be modeled as a speech/audio-chat adapter variant,
  not a special case inside `runtime/tts.rs`.
- Voice/reference inputs should contribute to request/cache identity.

### ASR

Current state:

- Native ASR uses typed wrappers and engine request paths.
- Audio-chat ASR can bypass the engine via direct registry access.

Target:

- ASR adapter owns byte/base64/path inputs, language hints, output detail level,
  and chunking strategy.
- Batch ASR, incremental ASR, and realtime ASR become distinct execution modes.
- Long-form ASR should use a job/batch runner with cancellation and progress
  events, not route-specific loops.

### Chat

Current state:

- Runtime builds chat prompt tokens through the registry, then uses engine
  requests.
- Tool/media normalization still lives mostly above core runtime.

Target:

- Chat adapter owns prompt rendering, chat config, tokenization, sampling
  settings, streaming deltas, and request identity.
- Multimodal media placeholders should be represented explicitly in request
  identity for future cache correctness.
- Product OpenAI/Responses formatting remains in `izwi-server`.

### Speech-to-Speech / Audio Chat

Current state:

- Unified voice calls LFM2.5 audio through runtime helpers and engine streaming.

Target:

- Audio-chat and speech-to-speech adapters own audio input normalization,
  history/messages, text/audio output multiplexing, and stream event typing.
- Voice mode should call a pipeline stage, not a websocket-specific helper.

### Diarization

Current state:

- `runtime/diarization.rs` is a large pipeline function that decodes audio,
  diarizes, runs ASR, optionally aligns, attributes speakers, and optionally
  refines with an LLM.

Target:

- `runtime/pipelines/diarization.rs` defines the graph.
- Stage adapters own `DecodeAudio`, `Diarize`, `Transcribe`, `Align`,
  `AttributeSpeakers`, and `RefineWithLlm`.
- The pipeline executor provides cancellation, stage-level metrics, partial
  outputs, and deterministic fallback rules.

### Realtime Voice

Current state:

- The websocket handler orchestrates modular and unified voice turns.
- Pipeline graphs are recorded but not executable.

Target:

- `runtime/sessions/voice.rs` owns turn lifecycle, VAD, endpointing,
  cancellation, barge-in, transcript events, and audio output events.
- `runtime/pipelines/voice.rs` owns modular and unified graph execution.
- WebSocket, WebRTC, desktop, and future telephony become protocol adapters over
  the same session runtime.

## Phased Plan

### Phase A: Boundary Inventory And Tests

Goal: add guardrails before moving code.

Deliverables:

- Create `crates/izwi-core/tests/` conformance skeleton with fake adapters.
- Add compile-only tests for all current public reexports.
- Add capability fixture manifests for TTS, ASR, chat, speech-to-speech,
  diarization, forced alignment, and voice pipelines.
- Add architectural assertions that product routes do not import internal
  `models::architectures::*`.

Verification:

- `cargo test -p izwi-core runtime::request --lib`
- `cargo test -p izwi-core runtime::broker --lib`
- `cargo test -p izwi-core runtime::pipeline --lib`
- New conformance skeleton tests.

### Phase B: Split Model Lifecycle Namespaces

Goal: separate catalog, artifact repository, and residency controller without
changing behavior.

Deliverables:

- Move `model/download.rs` responsibilities into `artifacts/`.
- Move lease/residency controller into `residency/`.
- Keep `model/` as compatibility reexports.
- Introduce `LoadedModelRegistry` naming for loaded handles.

Verification:

- Model lifecycle snapshot tests.
- Load/unload/delete behavior tests.
- UI/server compile checks.

### Phase C: Replace Optional Engine Request Union Internally

Goal: make invalid engine task states unrepresentable.

Deliverables:

- Add `EngineTask` enum:
  - `Tts(TtsEngineInput)`
  - `Asr(AsrEngineInput)`
  - `Chat(ChatEngineInput)`
  - `SpeechToSpeech(AudioChatEngineInput)`
- Keep `EngineCoreRequest` public constructors, but internally store a typed
  task payload.
- Move task-specific validation from handlers into task constructors.

Verification:

- Existing engine executor tests.
- Focused tests for impossible-state prevention.
- `cargo check -p izwi-server`.

### Phase D: Make Capability Adapters Executable

Goal: make adapters the extension surface for every model capability.

Deliverables:

- Add `runtime/capabilities/adapter.rs`.
- Add adapter registry methods for `validate`, `prepare`, `execute`, and
  `stream`.
- Move TTS/ASR/chat/speech-to-speech model selection logic behind adapters.
- Keep `RuntimeService` methods as compatibility shims.

Verification:

- Per-capability fake adapter tests.
- Existing route contract tests.
- Broker shadow/on tests.

### Phase E: Shrink RuntimeService Into A Facade

Goal: make ownership readable.

Deliverables:

- Extract:
  - `RuntimeTelemetryCollector` -> `runtime/telemetry/metrics.rs`
  - request execution loop -> `runtime/broker/router.rs` or
    `runtime/execution/local.rs`
  - cancellation/waiter map -> `runtime/broker/cancellation.rs`
  - stream policy -> `runtime/streams/`
- Leave `RuntimeService` as constructor plus stable public methods.

Verification:

- No public API behavior change.
- `cargo check -p izwi-core`
- `cargo check -p izwi-server`

### Phase F: Introduce PipelineExecutor

Goal: make voice and diarization graphs executable.

Deliverables:

- Add `PipelineExecutor`.
- Add typed stage input/output envelope.
- Move diarization transcript orchestration stage-by-stage.
- Move realtime voice turn orchestration out of websocket handler into
  `runtime/sessions/voice.rs` and `runtime/pipelines/voice.rs`.
- Add stage-level metrics and cancellation.

Verification:

- Diarization transcript parity tests.
- Realtime voice modular/unified event-shape tests.
- Pipeline cancellation tests.

### Phase G: Split Engine Subsystems

Goal: reduce engine change blast radius.

Deliverables:

- Split `engine/scheduler.rs` into `scheduler/{queue,policy,budget,preemption,stats}.rs`.
- Split `engine/kv_cache.rs` into `cache/{kv,prefix,residency,stats}.rs`.
- Move stream helpers into `engine/output/streaming.rs` or
  `runtime/streams/` depending on ownership.
- Keep public reexports stable.

Verification:

- Scheduler tests.
- KV cache tests.
- Streaming backpressure tests.
- `cargo test -p izwi-core engine::executor --lib`.

### Phase H: Model Family Contributor Path

Goal: make adding a model a predictable bounded change.

Deliverables:

- Rename or alias `models/architectures/` as `models/families/`.
- For each family, define a local `FamilyRegistration`:
  - supported variants
  - loader function
  - capability adapters
  - default backend/dtype policy
  - conformance fixture IDs
- Replace monolithic registry match arms with family registration tables.

Verification:

- One-family migration proof, likely Kokoro or Gemma first.
- Registry coverage test over `ModelVariant::all()`.
- Capability inventory parity test.

### Phase I: Production Telemetry And Replay

Goal: make debugging and capacity planning practical.

Deliverables:

- Unify runtime, engine, voice, pipeline, model lifecycle, and backend metrics.
- Add trace spans for broker admission, model load, scheduler wait, prefill,
  decode, first chunk, stage execution, cancellation, and unload.
- Add optional request dump/replay for sanitized local debugging.
- Add metric deprecation rules.

Verification:

- Prometheus contract tests.
- Trace attribute tests with no request-content logging by default.
- Replay fixture for fake adapter requests.

## Recommended First Implementation Sequence

The safest next implementation work is not a folder move. It is capability
contract hardening:

1. Add `crates/izwi-core/tests/capability_conformance.rs` with fake adapters.
2. Add typed `InferenceRequest` and `InferenceOutput` enums under
   `runtime/broker/` while keeping existing wrappers.
3. Make adapter execution real for one low-risk capability, preferably Kokoro
   TTS or chat fake adapter, before migrating all paths.
4. Extract runtime telemetry into `runtime/telemetry/metrics.rs`.
5. Extract diarization transcript into a pipeline executor with behavior parity
   tests.

This sequence creates the new architecture's skeleton and tests before moving
large files. It also keeps every public route working while internals migrate.

## Non-Goals For The Next Phase

- Do not split `izwi-core` into multiple crates yet.
- Do not require gateway/worker mode for local desktop/server use.
- Do not add new hardware backend support while reorganizing runtime ownership.
- Do not rewrite model-family implementations during boundary extraction.
- Do not move product persistence from `izwi-server` into `izwi-core`.

## Open Questions

- Should long-form ASR and diarization share a generic `JobRunner` distinct from
  the interactive `EngineCore` scheduler?
- Should model-family registration be static Rust tables only, or should it
  support out-of-tree registration later?
- Should audio preprocessing identity become part of request/cache identity
  before any prefix/cache optimization is attempted?
- Should `runtime_models` become the canonical module and `models` become fully
  private, or should `models/families` remain visible to advanced contributors?
- Which capability should be the first executable-adapter migration proof:
  Kokoro TTS, chat, or ASR?

## Review Checklist Before Implementation

- Public reexports remain stable or have explicit deprecation aliases.
- No product route behavior changes without parity tests.
- Every moved module has a behavior-preserving commit.
- Every phase has focused tests and `cargo check -p izwi-server`.
- `tasks/*` remains local-only and unstaged.
