# Voice AI Runtime Architecture

Status: Accepted

Date: 2026-05-09

## Context

Izwi is moving from a collection of audio inference capabilities toward a
production-grade Voice AI Runtime. The runtime must keep existing HTTP, CLI,
desktop, and model defaults compatible while making the internals easier to
extend.

The current implementation has the right foundations: a Rust core, task-aware
runtime service, engine scheduler, KV cache, native model registry, first-party
APIs, realtime WebSockets, probes, metrics, and release contracts. The main
architecture gap is not missing capability; it is boundary clarity. Product
routes, voice-session state, model residency, model dispatch, and streaming
pressure currently sit close enough together that new capabilities can require
changes across several unrelated domains.

This document defines the target internal boundaries and invariants that future
runtime changes should preserve.

## Goals

- Keep public API and CLI behavior compatible by default.
- Make every inference request carry explicit model identity after defaults are
  resolved.
- Keep protocol/product code thin: translate external requests into runtime
  commands, then delegate.
- Make realtime voice orchestration reusable outside WebSocket handlers.
- Separate artifact state from loaded model residency state.
- Add stable adapter seams for model capabilities.
- Make streaming pressure an explicit runtime policy.
- Make production voice metrics part of the runtime contract.

## Non-Goals

- Do not require a multi-process or gateway deployment for local use.
- Do not clone any one external inference engine architecture.
- Do not expand backend support unless the backend is covered by the support
  matrix and verification contract.
- Do not move durable product persistence into the inference engine.

## Target Layers

```text
Transport and Product APIs
  -> Voice Session Runtime
  -> Inference Broker
  -> Model Capability Adapters
  -> Model Repository and Residency
  -> Backend and Execution Substrate
  -> Ops, Metrics, Benchmarks, and Control Plane
```

### Transport and Product APIs

Examples: OpenAI-compatible HTTP, first-party REST APIs, CLI, desktop, WebSocket,
future WebRTC, and telephony adapters.

Responsibilities:

- Parse external protocol shapes.
- Apply auth, request context, CORS, and API compatibility rules.
- Resolve public defaults into explicit runtime inputs.
- Persist product state when the product feature owns that state.
- Return protocol-specific responses.

Invariant: transport code must not own model scheduling, model residency, or
voice turn orchestration.

### Voice Session Runtime

`VoiceSession` is the reusable duplex conversation boundary.

Input events:

- `AudioFrame`
- `UserText`
- `ControlEvent`
- `ToolResult`
- `SessionConfig`

Output events:

- `TranscriptDelta`
- `TranscriptFinal`
- `TurnStarted`
- `AssistantTextDelta`
- `AssistantTextFinal`
- `AudioChunk`
- `BargeIn`
- `RuntimeMetric`
- `Error`
- `Closed`

Responsibilities:

- Session lifecycle and turn state.
- Audio ingress and egress timing.
- VAD and endpointing decisions.
- ASR partial and final transcript handling.
- Agent, chat, tool, and TTS orchestration.
- Cancellation, interruption, and barge-in.
- Per-session correlation IDs, metrics, and trace attributes.

Invariant: WebSocket handlers are protocol adapters over `VoiceSession`, not the
long-term home of reusable voice behavior.

### Inference Broker

The broker is the runtime entrypoint for typed inference work.

Responsibilities:

- Accept typed inference requests from sessions, APIs, CLI, and desktop.
- Enforce per-request model identity.
- Coordinate admission control, deadlines, cancellation, and priority.
- Coordinate model residency with the scheduler/executor path.
- Attach request/session correlation to telemetry.
- Hide whether work is executed by the token scheduler, a batch-only model, a
  streaming adapter, or a future worker process.

Invariant: after request defaults are resolved, every broker request must include
the resolved model ID and capability.

### Model Capability Adapters

Adapters are stable contracts around runtime capabilities rather than around
routes.

Initial capability families:

- ASR
- realtime ASR
- TTS
- streaming TTS
- chat
- audio chat
- diarization
- VAD
- endpointing

Adapter responsibilities:

- Declare supported modalities and streaming modes.
- Declare artifact requirements.
- Declare backend and device constraints.
- Validate typed requests.
- Produce capability-specific outputs.
- Report metrics labels and conformance-test requirements.

Invariant: adding a model family should not require edits in unrelated product
routes once the relevant adapter contract exists.

### Model Repository and Residency

The model lifecycle has two distinct states:

- Artifact state: what is present, validated, downloading, failed, or missing on
  disk.
- Residency state: what is loaded, warming, ready, pinned, active, evictable, or
  unloading in process memory/device memory.

Target concepts:

- `ModelRepository`: artifact inventory, validation, download state, and local
  manifest handling.
- `ModelCatalog`: stable IDs, aliases, capabilities, and support policy.
- `ModelResidencyController`: loaded handles, memory budgets, pinning, warmup,
  unload, and LRU eviction.
- `ModelRegistry`: low-level runtime handle lookup behind residency APIs.

Invariant: runtime readiness must not confuse artifact availability with loaded
model readiness.

### Backend and Execution Substrate

The backend layer remains a curated support contract rather than a maximal
hardware matrix.

Responsibilities:

- Device discovery and reporting.
- Backend selection and policy.
- Kernel path metrics.
- Backend conformance checks.
- Executor/model runner integration.

Invariant: CPU and Metal behavior must remain protected when adding CUDA or
future backend-specific paths.

### Ops, Metrics, Benchmarks, and Control Plane

Production runtime behavior must be observable at request, stream, model, and
session levels.

Voice metric names should use these stable prefixes:

- `voice.session.*`
- `voice.audio.ingress.*`
- `voice.audio.egress.*`
- `voice.vad.*`
- `voice.endpointing.*`
- `voice.asr.*`
- `voice.llm.*`
- `voice.tts.*`
- `voice.barge_in.*`
- `voice.stream.*`
- `voice.model.*`

Required metric concepts:

- Session start, close, duration, and close reason.
- Audio ingress jitter and dropped frames.
- VAD start/end and endpoint latency.
- ASR first partial and final latency.
- LLM first token latency.
- TTS first audio latency.
- Audio underruns and client backpressure.
- Barge-in detection and cancellation latency.
- Model load, warm, ready, unload, and eviction events.

Invariant: new voice capabilities must add metrics and deterministic session
tests alongside behavior changes.

## Compatibility Rules

- Public route paths remain stable unless an API compatibility document changes.
- Existing model IDs, aliases, and defaults remain stable unless deprecated
  through docs and compatibility tests.
- Existing non-stream and stream response formats remain stable unless a phase
  explicitly records a behavior change.
- Internal refactors must include tests or compile checks at the smallest useful
  scope before they are committed.
- `tasks/*` files are local planning notes and are not part of production
  commits.

## Migration Order

1. Add this architecture spec and invariants.
2. Carry explicit per-request TTS model identity through runtime calls.
3. Introduce typed runtime request wrappers around the existing broad engine
   request.
4. Add model capability adapter traits and a registry seam.
5. Clarify artifact state versus loaded residency state.
6. Introduce explicit stream sink/backpressure policy.
7. Move realtime voice orchestration behind a reusable session boundary.
8. Add production voice observability and benchmark contracts.
9. Add optional gateway/worker-mode documentation after local seams are clean.

The optional gateway/worker-mode shape is documented in
[Voice Gateway And Worker Mode](./VOICE_GATEWAY_WORKER_MODE.md).
