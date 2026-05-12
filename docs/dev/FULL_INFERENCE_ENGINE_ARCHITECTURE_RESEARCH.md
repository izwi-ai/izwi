# Full Inference Engine Architecture Research

Date: 2026-05-09

## Scope

This is a research, analysis, and planning document. It does not prescribe an
immediate production-code patch. The goal is to move Izwi from a set of
capability-specific runtime paths toward a production-ready inference
architecture that covers TTS, ASR, chat, speech-to-speech, realtime voice,
diarization, and future multimodal models.

The recent voice work is useful precedent, but the target architecture should
not be limited to the `/voice` UI route or to downstream voice handlers.

## Recent Commit Review

The last architecture-oriented commits added useful runtime primitives, but they
remain partial and unevenly applied across capabilities.

| Commit | Area | What changed | Architecture implication |
| --- | --- | --- | --- |
| `4f924cc1` | Model lifecycle | Added artifact and residency states plus runtime lifecycle snapshots. | Separates "artifact exists" from "model resident", but snapshots are derived from `ModelInfo` and are not yet a control-plane contract. |
| `4e98dcae` | Engine streaming | Introduced `StreamSink` and `StreamBackpressurePolicy::FailOnFull`. | Good first abstraction for stream pressure, but only one policy exists and route-level streams still use mixed channel strategies. |
| `48a8054d` | Voice session | Added `VoiceSession` state machine and wired it into realtime voice connection state. | Useful state-machine precedent, but it is connection-local and not yet a shared pipeline runtime. |
| `e075e010` | Voice metrics | Added a voice metric contract and benchmark manifest. | Defines the metric surface before implementation, but most values are still contract-only. |
| `c82e9e96` | Gateway mode | Documented voice gateway/worker mode. | Correctly treats gateway mode as deployment architecture, not the local runtime core. |
| `568c1382` | Voice session close | Keeps realtime sessions open while active turns are running. | Improves lifecycle behavior, but completion handling is still opportunistic. |
| `ebf65f55` | Voice interruption | Interrupts only active turns. | Better turn semantics, but normal turn completion is not a first-class state transition yet. |
| `b623c067` | Runtime telemetry | Exposed voice telemetry snapshots and Prometheus counters. | Establishes a runtime telemetry pattern, but coverage is voice-specific. |

Bottom line: the recent work introduced good primitives in runtime, lifecycle,
streaming, telemetry, and voice state. The next step is to generalize these
primitives into a capability-neutral inference core.

## External Engine Notes

### vLLM

Primary sources:

- [vLLM architecture overview](https://docs.vllm.ai/en/stable/design/arch_overview/)
- [vLLM automatic prefix caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [vLLM multimodal processing](https://docs.vllm.ai/en/stable/design/mm_processing/)

Useful design patterns:

- Transport and engine are separate. The OpenAI-compatible API server handles
  HTTP, input processing, media loading, and response streaming, while engine
  core owns scheduling and KV-cache management.
- Process topology is explicit: API server processes, engine core processes,
  GPU workers, and optional data-parallel coordination.
- Cache identity lives in the KV manager. Prefix cache block identity includes
  parent block hash, block tokens, extra hashes for LoRA or multimodal payloads,
  and tenant cache salt.
- Multimodal input processing preserves the correspondence between prompt
  placeholder tokens and raw/preprocessed modality payloads. That matters for
  chunked prefill, caching, and correctness.
- New model support follows a repeatable shape: model class, input processing,
  dummy input construction, and registration. Contributors do not patch API
  routes for every new model.

Lessons for Izwi:

- Keep OpenAI/first-party routes as product adapters; put scheduling,
  residency, cache, cancellation, and stream pressure in the inference core.
- Treat audio features, voice references, speaker embeddings, modality
  placeholders, and tool-context transforms as part of request/cache identity.
- Introduce uniform model capability adapters so new models do not require
  scattered route/runtime/registry edits.

### SGLang

Primary sources:

- [SGLang HiCache design](https://docs.sglang.io/docs/advanced_features/hicache_design)
- [SGLang PD disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)
- [SGLang model gateway](https://docs.sglang.io/docs/advanced_features/sgl_model_gateway)
- [SGLang production metrics](https://docs.sglang.io/docs/references/production_metrics)

Useful design patterns:

- Serving exposes production knobs for queue size, running requests, chunked
  prefill, scheduling policy, priority, memory fraction, watchdogs, logging,
  tracing, and Prometheus.
- RadixAttention and HiCache make prefix KV reuse a runtime/cache concern, not
  an API concern. HiCache extends GPU cache reuse into host memory and optional
  distributed storage tiers.
- Prefill/decode disaggregation is a deployment mode for high-scale serving, not
  the minimum viable local architecture.
- The gateway/router is responsible for worker discovery, health-aware routing,
  load/cache-aware routing, circuit breaking, rate limiting, and API proxying.

Lessons for Izwi:

- Build a strong local inference core first; gateway/worker mode should remain
  optional deployment architecture.
- Make scheduler behavior observable before making it clever: queue depth,
  active requests, TTFT, inter-token latency, prefill/decode time, cache hits,
  eviction, preemption, stream backpressure, and memory pressure.
- Split long-term scale concerns into clear layers: local runtime, local worker,
  gateway/router, and distributed cache.

### llama.cpp

Primary sources:

- [llama.cpp README](https://github.com/ggml-org/llama.cpp)
- [llama-server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [llama-server developer notes](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README-dev.md)
- [llama.cpp multi-GPU guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md)

Useful design patterns:

- The core is simple and portable: model, context, batch/decode loop, and backend
  kernels. Server concerns are layered on top.
- The server supports OpenAI-compatible routes, monitoring endpoints, parallel
  decoding, continuous batching, and multi-user slots.
- Inference mode and router mode are distinct. A router can manage multiple
  model-serving instances without making distributed routing part of the basic
  engine abstraction.
- Multi-GPU support is pragmatic and policy-driven: layer split as the stable
  default, tensor split as experimental, and clear constraints around KV cache
  placement.

Lessons for Izwi:

- Keep the single-node/local runtime easy to reason about.
- Use explicit request slots/leases for active work so residency, cancellation,
  and metrics stay correct.
- Let backend/device policy be visible and constrained per model, instead of
  hiding it behind a global setting.

## Current Izwi Architecture Map

### Layers Today

| Layer | Current modules | Current role |
| --- | --- | --- |
| Product/API adapters | `crates/izwi-server/src/api/**`, `crates/izwi-server/src/app/**` | OpenAI-compatible routes, first-party persisted routes, realtime websocket routes, request/response formatting, persistence, route-level admission. |
| Runtime service | `crates/izwi-core/src/runtime/service.rs`, `runtime/{tts,asr,chat,speech_to_speech,diarization}.rs` | Main orchestrator for model loading and capability calls. Some flows go through `EngineCore`; others call models directly. |
| Runtime request/adapters | `runtime/request.rs`, `runtime/adapters.rs` | TTS has a typed request wrapper and metadata adapter check. Other capabilities do not yet have equivalent wrappers/adapters. |
| Engine core | `engine/core.rs`, `engine/request.rs`, `engine/scheduler.rs`, `engine/executor/**` | Schedules `EngineCoreRequest`, dispatches by `TaskType`, tracks active model variants, handles stream sink policy for some paths. |
| Models and registry | `models/registry.rs`, `models/architectures/**` | Holds loaded model handles and family-specific enums/match arms for ASR, audio-chat, chat, TTS, diarization, and forced alignment. |
| Model lifecycle | `model/**`, `runtime/lifecycle/**`, `catalog/**` | Resolves artifacts, downloads, instantiates, publishes state, unloads, and evicts idle models. |
| Backend/device policy | `backends/**` | Resolves global backend context and diagnostics; per-model backend hints exist but all current variants effectively use native execution paths. |
| Telemetry | `engine/metrics.rs`, `runtime/voice_metrics.rs`, `api/internal/metrics.rs` | Runtime queue/completion/kernel/worker counters plus new voice counters. |

### TTS Flow

Current flow:

1. `/v1/audio/speech` parses voice, model, format, speed, and streaming mode.
2. Route may resolve saved voice metadata.
3. Route calls `runtime.load_model(variant)` before calling generation.
4. `RuntimeService::generate` or `generate_streaming` resolves the TTS variant
   again and calls the TTS runtime.
5. TTS runtime uses the adapter registry only for TTS metadata validation.
6. Kokoro and LFM audio paths are special-cased. Other TTS paths build a
   `TtsRuntimeRequest` and run `EngineCoreRequest` through the core engine.
7. Routes encode audio, SSE, or chunked streaming responses.

Pain points:

- Route-level model loading can duplicate runtime loading.
- TTS has typed request and adapter scaffolding, but other capabilities do not.
- Model-family special cases live in runtime instead of adapters.
- Admission, timeout, response encoding, and stream policy are split across
  route/runtime/engine.

### ASR Flow

Current flow:

1. `/v1/audio/transcriptions` accepts JSON or multipart audio.
2. Route validates model name and acquires a route-level permit.
3. Runtime resolves ASR or audio-chat variants and loads the model.
4. Native ASR builds `EngineCoreRequest::asr` or `asr_bytes`.
5. Audio-chat transcription bypasses the core engine and calls registry model
   methods directly.
6. Route formats JSON, text, SRT, VTT, or SSE-style deltas.

Pain points:

- No typed ASR runtime request or capability adapter.
- Audio-chat ASR bypasses the shared scheduler and active-model tracking.
- Realtime transcription uses repeated sliding-window batch ASR rather than a
  true incremental ASR session.
- Cancellation and long-job budgets are inconsistent across synchronous and
  background transcription flows.

### Chat/Responses Flow

Current flow:

1. `/v1/chat/completions`, `/v1/responses`, and first-party chat routes parse
   messages, tools, media, and model parameters.
2. Server app helpers normalize into `ChatExecutionRequest`.
3. Route/app layer acquires semaphore permits and starts streaming tasks.
4. Runtime loads the model, pre-tokenizes prompt/messages through the registry,
   builds `EngineCoreRequest::chat`, and runs core engine or streaming engine.
5. Product routes shape OpenAI-compatible or first-party responses.

Pain points:

- Product adapters currently own too much inference policy: admission, tool/media
  transforms, streaming task shape, and some error semantics.
- Chat request construction is better centralized than route code, but it is
  still not a capability-neutral broker API.
- Multimodal/chat cache identity is not formalized.

### Speech-to-Speech and Realtime Voice Flow

Current flow:

1. Realtime voice websocket supports modular mode and unified mode.
2. Modular mode orchestrates VAD, ASR, agent/chat, and TTS in the websocket
   handler.
3. Unified mode routes audio to an audio-chat model through
   `speech_to_speech_generate_streaming_bytes_with_variant`.
4. Voice session state and voice metrics are connection-local runtime helpers.
5. Server routes persist sessions, turns, transcript events, and generated audio.

Pain points:

- The websocket handler is effectively a product-specific pipeline engine.
- Normal turn completion, cancellation, barge-in, and downstream cleanup should
  be broker/pipeline concerns, not connection-local details.
- Unified audio-chat and modular ASR/chat/TTS flows should share admission,
  model leasing, metrics, and stream pressure.

### Diarization Flow

Current flow:

1. OpenAI-style and first-party routes accept audio and diarization options.
2. Runtime decodes/resamples audio, loads Sortformer diarization, and calls the
   diarization model directly.
3. Runtime may run ASR, forced alignment, speaker attribution, and optional LLM
   refinement.
4. First-party routes persist jobs and generate summaries.

Pain points:

- Diarization is a pipeline/DAG, but it is encoded as a runtime function with
  direct registry access and optional nested inference calls.
- Direct model calls can avoid core scheduling and active residency tracking.
- Long-running audio jobs need a separate execution class from interactive
  voice or token streaming.

### Model Lifecycle and Backend Flow

Current flow:

1. Admin/UI/CLI routes list, download, cancel, load, unload, and delete models.
2. `RuntimeService::load_model` resolves backend diagnostics, artifacts, and
   instantiate/publish phases.
3. Model handles are registered in family-specific registry maps.
4. Unload aborts active core requests for a variant and unregisters handles.
5. LRU eviction consults active variants known to the core engine.

Pain points:

- Active work outside the core engine can be invisible to eviction and unload
  policy.
- Model family registration is split across catalog metadata, lifecycle
  instantiate/unload logic, registry enums, and runtime special cases.
- Backend selection is global-first; capability-specific backend constraints are
  mostly diagnostics rather than scheduling inputs.

## Target Architecture

The recommended architecture is a layered inference runtime with a broker at the
center:

```text
API and Product Adapters
  -> Inference Broker
  -> Capability Runtime Graphs
  -> Model Capability Adapters
  -> Residency Controller and Model Repository
  -> Execution Substrate
  -> Backend and Kernel Layer
  -> Observability and Control Plane
```

### 1. API and Product Adapters

Responsibilities:

- Parse transport-specific payloads.
- Authenticate/authorize when needed.
- Convert OpenAI, first-party, CLI, and websocket payloads into typed inference
  requests.
- Format responses, SSE events, websocket messages, and persisted records.

Non-responsibilities:

- Model loading.
- Admission control.
- Scheduling.
- Stream backpressure policy.
- Active model residency.
- Capability-specific execution.

### 2. Inference Broker

The broker becomes the main runtime entry point behind `RuntimeService`.

Responsibilities:

- Resolve model aliases/defaults.
- Validate request against model capability metadata.
- Assign request IDs, correlation IDs, deadlines, priorities, and cancellation.
- Acquire admission permits by workload class.
- Acquire model residency leases.
- Route requests to the correct capability graph and execution substrate.
- Own stream sink policy.
- Emit consistent metrics for every flow.

Core shape:

```rust
pub struct InferenceBroker {
    adapters: CapabilityRegistry,
    residency: ModelResidencyController,
    execution: ExecutionSubstrate,
    telemetry: InferenceTelemetry,
}

pub struct RequestEnvelope {
    pub request_id: RequestId,
    pub correlation_id: Option<String>,
    pub session_id: Option<String>,
    pub capability: CapabilityKind,
    pub model: ResolvedModel,
    pub priority: RequestPriority,
    pub deadline: Option<Instant>,
    pub cancellation: CancellationToken,
    pub stream_policy: StreamBackpressurePolicy,
}

pub enum InferenceRequest {
    Tts(TtsRequest),
    Asr(AsrRequest),
    Chat(ChatRequest),
    AudioChat(AudioChatRequest),
    Diarization(DiarizationRequest),
    Vad(VadRequest),
    Alignment(AlignmentRequest),
}
```

Migration note: keep `RuntimeService` as the public facade at first, but make it
delegate to the broker internally. That keeps server route churn low.

### 3. Capability Runtime Graphs

Some capabilities are single-stage requests. Others are pipelines. The broker
should treat both as runtime graphs.

Initial graphs:

- `TtsGraph`: text/voice/reference audio -> acoustic/audio generation -> codec.
- `AsrGraph`: audio decode/resample -> ASR -> optional alignment/formatting.
- `ChatGraph`: message normalization -> prompt processing -> token decode.
- `AudioChatGraph`: audio/text prompt -> audio-language model -> transcript or
  generated audio/text.
- `VoiceTurnGraph`: VAD/endpointing -> ASR or audio-chat -> agent/chat -> TTS
  or unified audio response.
- `DiarizationGraph`: decode -> diarization -> ASR -> alignment -> attribution
  -> optional LLM refinement.

This makes realtime voice and diarization reusable runtime behavior instead of
route-specific orchestration.

### 4. Model Capability Adapters

`runtime/adapters.rs` should evolve from TTS metadata validation into the main
extension point for all model families.

Recommended traits:

```rust
pub trait CapabilityAdapter: Send + Sync {
    fn metadata(&self) -> CapabilityMetadata;
    fn validate(&self, request: &InferenceRequest) -> Result<PreparedRequest>;
    fn execution_target(&self, request: &PreparedRequest) -> ExecutionTarget;
}

pub enum ExecutionTarget {
    TokenEngine(TokenEnginePlan),
    BatchRunner(BatchPlan),
    RealtimeRunner(RealtimePlan),
    Pipeline(PipelinePlan),
    DirectModel(DirectModelPlan),
}
```

Capability metadata should include:

- Supported capabilities and aliases.
- Accepted input modalities and output modalities.
- Streaming support.
- Batchability and incremental-session support.
- Required artifacts and tokenizer/codec assets.
- Backend/device constraints.
- Max context, max audio duration, sample-rate expectations, and format limits.
- Cache identity fields.
- Conformance test fixtures.

The goal is for a contributor adding a model to add a manifest/adapter/loader and
tests, not route patches.

### 5. Residency Controller and Model Repository

Current artifact and residency state should be expanded into a lease-based
controller.

Responsibilities:

- Track artifact state: missing, downloading, available, corrupt, deleting.
- Track residency state: not resident, loading, warming, ready, busy, idle,
  unloading, failed.
- Issue leases for active requests and pipeline stages.
- Prevent unload/eviction while leases exist.
- Track memory budgets per backend/device.
- Expose readiness, warmup, load latency, unload latency, eviction reason, and
  active lease counts.

Every model access path, including direct registry calls and pipeline internals,
should require a residency lease.

### 6. Execution Substrate

Instead of pushing every capability through a single optional-field
`EngineCoreRequest`, split execution into substrate classes:

| Substrate | Workloads | Notes |
| --- | --- | --- |
| `TokenEngine` | Chat, responses, audio-chat decoders, token-based TTS | Owns token scheduler, KV cache, prefix cache, chunked prefill, streaming token output. |
| `BatchRunner` | ASR, diarization, alignment, embedding-style jobs | Owns bounded queues, per-model concurrency, CPU/GPU work placement, long-job fairness. |
| `RealtimeRunner` | Realtime ASR, VAD, endpointing, streaming audio-chat | Owns session state, incremental buffers, partial results, low-latency deadlines. |
| `PipelineRunner` | Voice turns, diarization+ASR+refinement, summaries | Owns DAG execution, cancellation propagation, retries, and stage metrics. |

`EngineCoreRequest` can remain internally during migration, but new public
runtime contracts should be typed and capability-specific.

### 7. Scheduler and Cache Policy

Short-term scheduler policy:

- Separate workload classes: interactive voice, streaming chat, batch ASR,
  long-form TTS, diarization, and background summaries.
- Admission by workload class and model residency budget.
- Central deadline and cancellation handling.
- Consistent stream pressure behavior for SSE, websocket, and chunked audio.
- Metrics for queue wait, execution time, TTFT, inter-token/audio-chunk latency,
  cancellation, timeout, backpressure, and dropped/coalesced events.

Longer-term token scheduler work:

- Block-based KV cache with refcounts and free-list.
- Prefix cache identity including model, tokenizer, prompt tokens, multimodal
  hashes, voice/reference audio hashes, LoRA/adapters if added, and tenant salt.
- Optional chunked prefill for chat/audio-chat.
- Optional prefill/decode split as a deployment mode, not the local runtime core.

### 8. Stream Policy

Generalize the current `StreamSink` work.

Recommended policies:

- `FailOnFull`: current behavior for strict correctness.
- `BlockWithDeadline`: useful when losing data is unacceptable but latency has a
  bound.
- `DropOldest`: useful for realtime partials where the newest update matters.
- `Coalesce`: useful for ASR partials, queue-depth metrics, and UI progress.
- `Sample`: useful for high-frequency diagnostics.

Every policy should emit counters and latency histograms.

### 9. Backend and Device Policy

Backend selection should become a typed execution decision:

- Model manifest declares allowed backends and preferred backend order.
- Adapter prepares a backend plan based on model, capability, input size, and
  device availability.
- Runtime can reject unsupported model/backend combinations early.
- Health/readiness exposes backend diagnostics per model.
- Gateway mode can route by model residency and backend health.

This preserves the current simple local mode while making production deployment
explicit.

### 10. Observability and Control Plane

Minimum metrics by phase:

- `inference_requests_total{capability,model,status}`
- `inference_queue_wait_seconds{capability,model,priority}`
- `inference_execute_seconds{capability,model,substrate}`
- `inference_stream_backpressure_total{capability,model,policy}`
- `inference_cancellations_total{capability,model,reason}`
- `model_artifact_state{model,state}`
- `model_residency_state{model,state,backend,device}`
- `model_residency_leases{model}`
- `model_load_seconds{model,backend}`
- `kv_cache_hit_total{model,cache_level}`
- `kv_cache_evict_total{model,reason}`
- `voice_turns_total{mode,status}` as a voice-specific rollup, not the only
  observability surface.

Control-plane APIs should expose:

- Model artifact state.
- Model residency state and active leases.
- Warmup/readiness state.
- Backend diagnostics.
- Queue/admission state by workload class.
- Recent inference errors and cancellation reasons.

## Proposed Module Layout

The new architecture can be introduced alongside existing runtime modules:

```text
crates/izwi-core/src/inference/
  mod.rs
  broker.rs
  envelope.rs
  request.rs
  response.rs
  capabilities/
    mod.rs
    tts.rs
    asr.rs
    chat.rs
    audio_chat.rs
    diarization.rs
    vad.rs
    alignment.rs
  adapters/
    mod.rs
    registry.rs
    metadata.rs
    manifest.rs
  residency/
    mod.rs
    controller.rs
    lease.rs
    budget.rs
  execution/
    mod.rs
    token_engine.rs
    batch_runner.rs
    realtime_runner.rs
    pipeline_runner.rs
  stream/
    mod.rs
    sink.rs
    policy.rs
  telemetry.rs
  conformance.rs
```

Compatibility strategy:

- `runtime/*` remains the public facade during migration.
- Existing routes continue to call `RuntimeService`.
- `RuntimeService` gradually delegates to `inference::InferenceBroker`.
- Old `EngineCoreRequest` remains internal until typed requests cover all flows.
- Existing model registry maps remain until adapters own execution routing.

## Pre-Implementation Regression Review

This architecture must be introduced as an internal refactor first. Public
behavior should remain indistinguishable from the current application unless a
phase explicitly opts into a measured performance improvement.

### Compatibility Invariants

The following invariants must hold through every phase:

- Public HTTP, websocket, CLI, and UI contracts remain stable.
- OpenAI-compatible request and response shapes remain stable, including error
  status codes, streaming event order, terminal events, response formats, and
  usage metadata where currently emitted.
- First-party persistence behavior remains stable for speech history,
  transcription jobs, diarization jobs, chat threads, responses, voice sessions,
  and voice turns.
- Model aliases, default model selection, voice defaults, sample-rate defaults,
  generation defaults, and diarization defaults do not change as a side effect
  of architectural work.
- `RuntimeService` remains the compatibility facade until every capability has
  parity tests and the old path can be removed deliberately.
- Every migration step has a fallback to the existing path until that capability
  is proven equivalent.
- New validation from capability adapters starts in audit/shadow mode before it
  is allowed to reject requests in the default path.
- No phase may make gateway/worker infrastructure mandatory for local desktop or
  single-node server usage.

### Required Compatibility Matrix

Before implementation, create or identify tests for this matrix and keep them
green after every phase:

| Surface | Paths to preserve |
| --- | --- |
| TTS | `/v1/audio/speech` non-streaming, SSE/chunked streaming, saved voice resolution, long-form first-party jobs, CLI `tts`, UI text-to-speech route. |
| ASR | `/v1/audio/transcriptions` JSON and multipart, text/SRT/VTT/JSON formats, first-party transcription jobs, realtime transcription websocket, CLI `transcribe`, UI speech-text route. |
| Chat | `/v1/chat/completions` streaming and non-streaming, `/v1/responses`, tool/media normalization, first-party chat threads, CLI `chat`, UI chat playground. |
| Voice | Realtime modular voice, unified audio-chat voice, barge-in, interruption, turn persistence, transcript/audio event ordering, voice metrics. |
| Diarization | First-party speech-to-text jobs with `job_kind=diarization`, ASR attribution, forced alignment, optional LLM refinement, export formats, UI diarization route. |
| Model ops | List, download, cancel download, load, unload, delete, lifecycle snapshots, UI/CLI/admin compatibility. |
| Metrics | Existing `/v1/metrics`, `/v1/metrics/prometheus`, `/internal/*` aliases, runtime counters, voice counters, benchmark snapshots. |

### Feature-Flag and Fallback Rules

Implementation should use explicit rollout controls:

- Add a top-level broker mode before behavior changes:
  `IZWI_INFERENCE_BROKER=off|shadow|on`.
- Default to `off` or `shadow` until parity and performance gates pass.
- Allow per-capability rollout controls for TTS, ASR, chat, audio-chat,
  diarization, and realtime voice if the migration reaches execution behavior.
- In `shadow` mode, build envelopes, validate adapters, and emit audit metrics,
  but execute through the current path.
- In `on` mode, broker execution may be used only for capabilities whose
  compatibility matrix is passing.
- Keep the current path callable until the replacement path has equivalent tests,
  metrics, and cancellation behavior.

### Performance Gates

Architectural changes should improve performance, not merely rearrange code. For
each implementation phase that touches runtime behavior, capture a baseline on
the current path and compare after the change:

- TTS: cold load, warm generation latency, first audio chunk latency, total audio
  generation time, stream chunk cadence, peak memory.
- ASR: short audio latency, long audio throughput, realtime partial cadence,
  word timestamp/format cost, peak memory.
- Chat/responses: prompt processing time, time to first token, tokens per
  second, inter-token latency, stream terminal latency, peak memory.
- Voice: endpoint-to-ASR-final latency, ASR-final-to-first-audio latency,
  interruption latency, turn completion cleanup time, dropped/coalesced events.
- Diarization: decode/resample time, diarization time, ASR attribution time,
  optional refinement time, end-to-end job time, cancellation responsiveness.
- Model ops: cold load time, warm residency lookup time, unload time, eviction
  decisions, active lease count accuracy.

Default acceptance gate:

- No unapproved p50/p95 latency or throughput regression greater than 5 percent
  for the same model, backend, prompt/audio fixture, and machine.
- No increase in memory that changes whether a currently supported model fits on
  the same device.
- No added audio copies, full-buffer waits, or token buffering in streaming paths
  unless a test demonstrates lower latency or better correctness.
- Any intentional tradeoff must be documented with the before/after numbers and
  the user-visible reason.

### Safe Migration Order

The safest implementation order is:

1. Add typed request/envelope structures and tests with no runtime routing
   changes.
2. Add capability registry coverage in audit mode, with no new rejections.
3. Add broker in `shadow` mode behind `RuntimeService`, preserving current
   execution paths.
4. Move common metrics and request identity into the broker without changing
   outputs.
5. Move admission/deadline/stream policy one capability at a time, with fallback.
6. Add residency leases around existing direct model calls before changing
   eviction or unload behavior.
7. Move execution routing behind adapters only after old/new output parity is
   proven.
8. Move voice and diarization into pipeline graphs after cancellation, metrics,
   and stream event order are covered.
9. Add cache/scheduler optimizations only after baseline metrics exist.

### Rollback Requirements

Every behavioral phase must be reversible:

- A single environment toggle must restore the old execution path for a
  capability.
- New state must be additive or derivable from existing state during rollout.
- No database or persisted response format migration should be required for the
  broker phases.
- Metrics names may be added, but existing metrics names must remain available
  until dashboards and benchmark tooling are updated.
- If broker execution fails in a capability still under rollout, fallback should
  either call the old path safely or fail with the same error shape the old path
  would have produced.

## Migration Plan

### Phase 0: Contract and Inventory

- Keep this document and existing voice architecture docs as planning artifacts.
- Add a capability inventory table for every existing model variant in
  [INFERENCE_CAPABILITY_INVENTORY.md](INFERENCE_CAPABILITY_INVENTORY.md).
- Decide canonical capability names and output modalities.
- Define conformance fixtures for TTS, ASR, chat, audio-chat, diarization, VAD,
  and alignment.

Exit criteria:

- No production code behavior changes.
- Each current model has an explicit capability inventory row.

### Phase 1: Typed Requests for Every Capability

- Add typed request wrappers for ASR, chat, audio-chat, diarization, VAD, and
  alignment, mirroring the existing TTS wrapper pattern.
- Introduce `RequestEnvelope` with correlation, priority, deadline, cancellation,
  model identity, and stream policy.
- Keep conversion into `EngineCoreRequest` where necessary.

Exit criteria:

- All runtime task entry points can produce typed requests.
- Existing route behavior is unchanged.
- Unit tests prove model identity and required inputs per capability.

### Phase 2: Capability Registry for All Models

- Expand `RuntimeAdapterRegistry` into a capability registry for all model
  families.
- Register current TTS, ASR, chat, audio-chat, diarization, forced alignment,
  VAD, and endpointing capabilities.
- Move model-family special cases behind adapter metadata where possible.

Exit criteria:

- Runtime validates every capability through the registry.
- No route handler contains model-family execution decisions.

### Phase 3: Inference Broker Behind RuntimeService

- Add `InferenceBroker` and make `RuntimeService` delegate to it.
- Move model alias resolution, admission, deadlines, load/lease acquisition,
  stream policy, and common metrics into broker code.
- Remove route-level direct `load_model` calls where broker ownership is clear.

Exit criteria:

- OpenAI speech, transcription, chat, responses, diarization, and realtime voice
  still pass existing tests.
- Every request emits common broker metrics.

### Phase 4: Lease-Based Residency

- Introduce residency leases for active work.
- Require direct model registry access to carry a lease.
- Update unload/eviction policy to respect leases from both core-engine and
  direct/pipeline execution.
- Expose lifecycle snapshots through admin/health APIs.

Exit criteria:

- Unload and eviction cannot race active ASR/audio-chat/diarization work.
- Metrics expose active leases and residency transitions.

### Phase 5: Execution Target Abstraction

- Make adapters return `ExecutionTarget`.
- Move Kokoro, LFM audio, audio-chat ASR, and diarization direct paths behind
  adapter execution plans.
- Split token, batch, realtime, and pipeline execution classes.

Exit criteria:

- New models can be added by adapter/loader/manifest tests without route edits.
- `EngineCoreRequest` is no longer the public runtime shape.

### Phase 6: Unified Streaming

- Generalize `StreamSink` and `StreamBackpressurePolicy`.
- Apply stream policy to SSE, websocket, chunked audio, ASR partials, and chat
  tokens.
- Add stream backpressure metrics and tests.

Exit criteria:

- All streaming flows have explicit bounded behavior.
- Dropped/coalesced/fail/block events are observable.

### Phase 7: Pipeline Runner for Voice and Diarization

- Move realtime voice turn orchestration into `VoiceTurnGraph`.
- Move diarization+ASR+alignment+refinement into `DiarizationGraph`.
- Propagate cancellation through graph stages.
- Tie voice session phase changes to actual stage completion/failure.

Exit criteria:

- Websocket handlers become transport adapters.
- Diarization and voice share broker admission, leases, metrics, and stream
  policy.

### Phase 8: Scheduler and Cache Improvements

- Add token-engine cache metrics first.
- Introduce block-based prefix cache for token models.
- Include multimodal/audio identity in cache keys.
- Evaluate chunked prefill and cache-aware routing only after baseline metrics
  are stable.

Exit criteria:

- Cache hit/eviction metrics are reliable.
- Chat/audio-chat latency can be measured before and after cache changes.

### Phase 9: Optional Gateway/Worker Mode

- Keep local runtime as the default.
- Add gateway mode for deployment: worker discovery, health checks, routing by
  model residency/backend health, rate limiting, and circuit breaking.
- Consider prefill/decode split only as an advanced deployment mode.

Exit criteria:

- Single-node desktop/server mode does not require gateway infrastructure.
- Multi-worker deployments have explicit health and routing contracts.

## Verification Plan

For each implementation phase after this planning phase:

- Run `cargo check -p izwi-core -p izwi-server -p izwi-cli`.
- Run focused unit tests for typed requests, adapters, residency leases, stream
  policy, and broker routing.
- Run OpenAI-compatible contract tests for speech, transcription, chat,
  responses, and diarization.
- Run realtime smoke tests for voice websocket and transcription websocket.
- Run long-form ASR and diarization jobs to verify cancellation and persistence.
- Compare metrics before/after for queue wait, latency, backpressure, and model
  residency.
- Confirm `tasks/*` remains local-only and unstaged.

## Staff-Engineer Review Questions

- Can a new model be added without editing API route handlers?
- Can a route handler accidentally load/unload a model outside residency policy?
- Does every active request have a cancellation path?
- Does every stream have bounded backpressure behavior?
- Are all direct model calls visible to metrics and eviction policy?
- Can local runtime stay simple while gateway mode scales out?
- Are capability boundaries clear enough for contributors to work independently?

## Recommended Decision

Adopt a broker-centered inference architecture. Keep `RuntimeService` as the
compatibility facade, but introduce typed capability requests, capability
adapters, lease-based residency, and execution substrates behind it. Migrate
capability by capability, starting with request contracts and registry coverage,
then centralize admission/residency/streaming/metrics, and only then move voice
and diarization into reusable pipeline graphs.

This path keeps behavior stable while moving Izwi toward an engine that is
production-ready, observable, and easier to extend.
