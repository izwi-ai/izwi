# Izwi Inference Engine

> **Comprehensive reference** — architecture, component deep-dive, configuration, and extension guide.

---

## Table of Contents

1. [Overview & Design Goals](#1-overview--design-goals)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Module Layout](#3-module-layout)
4. [Layer-by-Layer Breakdown](#4-layer-by-layer-breakdown)
   - 4.1 [Runtime Orchestration (`runtime/`)](#41-runtime-orchestration-runtime)
   - 4.2 [Model Catalog (`catalog/`)](#42-model-catalog-catalog)
   - 4.3 [Backend Router (`backends/`)](#43-backend-router-backends)
   - 4.4 [Model Families (`families/`)](#44-model-families-families)
   - 4.5 [Shared Model Infrastructure (`models/shared/`)](#45-shared-model-infrastructure-modelsshared)
   - 4.6 [Model Architectures (`models/architectures/`)](#46-model-architectures-modelsarchitectures)
   - 4.7 [Codec Namespace (`codecs/`)](#47-codec-namespace-codecs)
5. [Engine Core (`engine/`)](#5-engine-core-engine)
   - 5.1 [Entry Points — `Engine`](#51-entry-points--engine)
   - 5.2 [Central Orchestrator — `EngineCore`](#52-central-orchestrator--enginecore)
   - 5.3 [Request Processor](#53-request-processor)
   - 5.4 [Scheduler](#54-scheduler)
   - 5.5 [Executor — `UnifiedExecutor` / `NativeExecutor`](#55-executor--unifiedexecutor--nativeexecutor)
   - 5.6 [KV Cache Manager](#56-kv-cache-manager)
   - 5.7 [Output Processor](#57-output-processor)
   - 5.8 [Signal Frontend](#58-signal-frontend)
6. [Request Lifecycle](#6-request-lifecycle)
   - 6.1 [Prefill Phase](#61-prefill-phase)
   - 6.2 [Decode Phase](#62-decode-phase)
   - 6.3 [Chunked Prefill](#63-chunked-prefill)
7. [Attention Mechanisms](#7-attention-mechanisms)
8. [Metal / Apple Silicon Optimisations](#8-metal--apple-silicon-optimisations)
9. [Configuration Reference](#9-configuration-reference)
10. [API Surface](#10-api-surface)
    - 10.1 [OpenAI-Compatible Endpoints](#101-openai-compatible-endpoints)
    - 10.2 [Admin Endpoints](#102-admin-endpoints)
11. [Unimplemented / Planned Features](#11-unimplemented--planned-features)
12. [Extension Points](#12-extension-points)
13. [Optimisation Opportunities & Recommendations](#13-optimisation-opportunities--recommendations)

---

## 1. Overview & Design Goals

Izwi is a **multi-modal audio inference server** built in Rust. Its inference engine is inspired by [vLLM](https://github.com/vllm-project/vllm) and targets **Apple Silicon (Metal/MPS)** as the primary compute substrate, while remaining backend-agnostic through a pluggable executor model.

| Goal | Mechanism |
|---|---|
| High throughput | Continuous batching, chunked prefill |
| Low latency | Paged KV-cache, streaming output |
| Memory efficiency | Block-level KV-cache with reference counting |
| Hardware flexibility | `BackendRouter` selects CPU / Metal / CUDA at runtime |
| OpenAI compatibility | Drop-in replacement for `/v1/audio/*`, `/v1/chat/*` |
| Extensibility | Trait-based model executor, pluggable scheduler policies |

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        izwi-server                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Axum HTTP Layer                       │   │
│  │   /v1/audio/*   /v1/chat/*   /v1/admin/*   /v1/models   │   │
│  └──────────────────────┬───────────────────────────────────┘   │
└─────────────────────────│───────────────────────────────────────┘
                          │ Arc<RuntimeService>
┌─────────────────────────▼───────────────────────────────────────┐
│                       izwi-core                                 │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │               Runtime Layer (runtime/)                  │    │
│  │   RuntimeService → Engine → EngineCore                  │    │
│  └──────────────────────┬──────────────────────────────────┘    │
│                         │                                       │
│  ┌──────────────────────▼──────────────────────────────────┐    │
│  │              Engine Core (engine/)                      │    │
│  │                                                         │    │
│  │  RequestProcessor → Scheduler → UnifiedExecutor         │    │
│  │                              ↘ KVCacheManager           │    │
│  │                    OutputProcessor ←────────────────────│    │
│  └──────────────────────┬──────────────────────────────────┘    │
│                         │                                       │
│  ┌──────────────────────▼──────────────────────────────────┐    │
│  │            Backend Router (backends/)                   │    │
│  │   CandleNative │ CandleMetal │ CandleCuda               │    │
│  └──────────────────────┬──────────────────────────────────┘    │
│                         │                                       │
│  ┌──────────────────────▼──────────────────────────────────┐    │
│  │          Model Catalog + Families (catalog/, families/) │    │
│  │   ModelVariant → ModelFamily → ModelTask                │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Module Layout

```
crates/izwi-core/src/
├── engine/
│   ├── mod.rs               # Engine public API
│   ├── core.rs              # EngineCore — central inference loop
│   ├── config.rs            # EngineCoreConfig
│   ├── scheduler.rs         # Scheduler, SchedulingPolicy
│   ├── executor.rs          # UnifiedExecutor, NativeExecutor, ModelExecutor trait
│   ├── kv_cache.rs          # KVCacheManager, KVBlock, paged attention
│   ├── metal_kv_cache.rs    # MetalKVCacheManager, memory pressure handling
│   ├── request_processor.rs # Tokenisation, prompt validation
│   ├── output_processor.rs  # Token sampling, streaming output assembly
│   └── signal_frontend.rs   # Audio pre-processing, VAD
├── runtime/
│   ├── mod.rs               # Runtime module exports
│   ├── service.rs           # RuntimeService — top-level orchestrator
│   ├── diarization.rs       # Speaker diarization runtime handler
│   └── lfm2.rs              # LFM-2 TTS runtime handler
├── catalog/
│   └── variant.rs           # ModelVariant, ModelFamily, ModelTask, InferenceBackendHint
├── backends/
│   └── mod.rs               # ExecutionBackend, BackendRouter
├── families/
│   └── …                    # Per-family model loading helpers
├── models/
│   ├── shared/
│   │   ├── attention/
│   │   │   ├── batched.rs   # Batched attention kernel
│   │   │   └── paged.rs     # Paged attention kernel
│   │   └── …
│   └── architectures/       # Concrete model implementations
└── codecs/                  # Audio codec wrappers (Encodec, DAC, …)
```

---

## 4. Layer-by-Layer Breakdown

### 4.1 Runtime Orchestration (`runtime/`)

The runtime layer is the **top-level owner of all engine state**. `RuntimeService` holds an `Arc<Engine>`, a model manager, and a telemetry snapshot (`RuntimeTelemetrySnapshot`). It exposes a unified interface consumed by the HTTP layer.

Sub-modules handle task-specific orchestration:

| Module | Responsibility |
|---|---|
| `service.rs` | Lifecycle management, telemetry, model hot-swap |
| `diarization.rs` | Speaker diarization pipeline orchestration |
| `lfm2.rs` | LFM-2 TTS pipeline orchestration |

### 4.2 Model Catalog (`catalog/`)

`variant.rs` is the single source of truth for **model identity and capability mapping**.

```rust
pub enum ModelFamily { Whisper, Qwen, Lfm2, Dia, … }
pub enum ModelTask   { Asr, Tts, Chat, Diarization, … }
pub enum InferenceBackendHint { CandleNative }
```

`ModelVariant` implements:
- `family()` — maps variant string to `ModelFamily`
- `primary_task()` — maps variant to `ModelTask`
- `backend_hint()` — advises `BackendRouter` on preferred backend

The `parse_model_variant` function and task-specific resolvers (`resolve_tts_variant`, `resolve_asr_variant`, etc.) handle string-to-variant parsing from API requests and config files.

### 4.3 Backend Router (`backends/`)

`BackendRouter` selects the concrete `ExecutionBackend` at model-load time:

```rust
pub enum ExecutionBackend {
    CandleNative,   // CPU via candle
    CandleMetal,    // Apple GPU via candle + Metal
    CandleCuda,     // NVIDIA GPU via candle + CUDA
}
```

Selection logic: `ModelVariant` → `InferenceBackendHint` → device availability check → `ExecutionBackend`.

### 4.4 Model Families (`families/`)

Per-family modules contain weight-loading helpers, tokeniser wrappers, and family-specific configuration parsing. They sit between the catalog (identity) and the architecture implementations (compute).

### 4.5 Shared Model Infrastructure (`models/shared/`)

Reusable building blocks shared across model architectures:

- **`attention/batched.rs`** — standard batched multi-head attention
- **`attention/paged.rs`** — paged attention over KV-cache blocks
- Positional encodings, layer normalisations, feed-forward helpers

### 4.6 Model Architectures (`models/architectures/`)

Concrete model graph implementations (Whisper encoder-decoder, Qwen transformer, LFM-2, Dia, etc.). Each architecture implements the `ModelExecutor` trait consumed by `NativeExecutor`.

### 4.7 Codec Namespace (`codecs/`)

Audio codec wrappers (Encodec, DAC, and others) used by TTS and speech-to-speech pipelines to convert between discrete audio tokens and waveforms.

---

## 5. Engine Core (`engine/`)

### 5.1 Entry Points — `Engine`

`Engine` (`engine/mod.rs`) is the **public API** for inference. It owns an `Arc<EngineCore>`, a `RequestProcessor`, and an `OutputProcessor`.

```
Engine
 ├── generate(request)          → blocking, returns complete output
 ├── generate_streaming(request) → returns async Stream of chunks
 └── run()                      → drives the EngineCore event loop
```

### 5.2 Central Orchestrator — `EngineCore`

`EngineCore` (`engine/core.rs`) coordinates all sub-systems in a tight **step loop**:

```
EngineCore::step()
  1. Scheduler::schedule()          → produces ScheduledBatch
  2. UnifiedExecutor::execute()     → runs model forward pass
  3. KVCacheManager::commit()       → updates block table
  4. OutputProcessor::process()     → samples tokens, emits chunks
```

`EngineCore` also owns the `KvCacheBackend` enum, which dynamically selects between the standard and Metal KV-cache managers:

```rust
enum KvCacheBackend {
    Standard(KVCacheManager),
    Metal(MetalKVCacheManager),
}
```

**Metal execution note:** On MPS devices the step loop runs decode and prefill **sequentially** (not in parallel) to avoid Metal command-buffer contention.

### 5.3 Request Processor

Handles the ingestion side:
- Tokenises text prompts / encodes audio inputs
- Validates sequence lengths against `max_model_len`
- Constructs `EngineCoreRequest` objects placed into the scheduler queue

### 5.4 Scheduler

`Scheduler` (`engine/scheduler.rs`) implements a **continuous-batching** scheduler with two policies:

```rust
pub enum SchedulingPolicy {
    Fcfs,      // First-come, first-served (default)
    Priority,  // Priority queue with preemption
}
```

Key `SchedulerConfig` parameters (all tunable via `EngineCoreConfig`):

| Parameter | Default | Description |
|---|---|---|
| `max_num_seqs` | 256 | Maximum concurrent sequences |
| `max_num_batched_tokens` | 8192 | Token budget per step |
| `max_model_len` | model-dependent | Maximum sequence length |
| `enable_chunked_prefill` | true | Split long prefills across steps |
| `preemption_mode` | Recompute | How to handle KV eviction |

**Preemption:** When KV blocks are exhausted the scheduler can preempt lower-priority sequences via `PreemptionReason`. VAD-triggered preemption (`VadPreemptionEvent`) allows the signal frontend to interrupt sequences when silence is detected.

### 5.5 Executor — `UnifiedExecutor` / `NativeExecutor`

```
UnifiedExecutor
  └── Arc<RwLock<Box<dyn ModelExecutor>>>
        └── NativeExecutor   (current concrete implementation)
```

`UnifiedExecutor` provides an async-safe wrapper around any `ModelExecutor` implementation. `NativeExecutor` is the current concrete backend and manages per-task decode state:

| Decode State Struct | Task |
|---|---|
| `ActiveChatDecode` | Chat / LLM |
| `ActiveAsrDecode` | Automatic Speech Recognition |
| `ActiveQwenTtsDecode` | Qwen TTS |
| `ActiveLfm2TtsDecode` | LFM-2 TTS |
| `ActiveSpeechToSpeechDecode` | Speech-to-speech |

**Parallel execution (CPU):** `NativeExecutor::execute_requests_parallel` uses `thread::scope` to fan out requests across CPU threads. This path is disabled on MPS (`can_parallelize_requests` returns `false`), keeping Metal execution serial.

### 5.6 KV Cache Manager

The KV cache uses **paged attention** — memory is allocated in fixed-size blocks rather than contiguously per sequence.

#### Standard KV Cache (`kv_cache.rs`)

```rust
pub struct KVCacheConfig {
    pub block_size: usize,          // tokens per block (default 16)
    pub num_cpu_blocks: usize,
    pub num_gpu_blocks: usize,
    pub dtype: DType,
    pub enable_quantization: bool,  // Int8 KV quantization
}

pub struct KVBlock {
    pub block_id: u32,
    pub ref_count: AtomicU32,       // reference-counted sharing
    pub residency: CacheResidency,  // Gpu | Cpu | Pinned
}
```

`PinnedBlockHandle` enables zero-copy CPU↔GPU transfers for block swapping during preemption.

#### Metal KV Cache (`metal_kv_cache.rs`)

`MetalKVCacheManager` extends the standard manager with Apple Silicon-specific features:

```rust
pub struct MetalKVCacheConfig {
    pub base: KVCacheConfig,
    pub unified_memory_fraction: f32,  // fraction of unified memory to use
    pub optimal_block_size: usize,     // tuned for Metal page size
}

pub enum MemoryPressure { Normal, Warning, Critical }
```

Automatic responses to memory pressure:
- **Warning** — reduce batch size, increase eviction aggressiveness
- **Critical** — suspend new prefills, aggressively evict cold blocks

### 5.7 Output Processor

Converts raw logits from the executor into user-facing output:
- Token sampling (greedy / top-p / top-k)
- Stop-sequence detection
- Streaming chunk assembly and back-pressure management
- Audio token → waveform decoding (via `codecs/`)

### 5.8 Signal Frontend

`signal_frontend.rs` handles audio pre-processing before tokens reach the scheduler:
- Resampling, normalisation, mel-spectrogram extraction
- **Voice Activity Detection (VAD):** current implementation uses a simple energy-based detector. Silero VAD integration is planned (marked `TODO` in source).

---

## 6. Request Lifecycle

```
HTTP Request
    │
    ▼
RequestProcessor
    │  tokenise / encode audio
    ▼
Scheduler queue
    │
    ├─── Prefill phase ──────────────────────────────────────────┐
    │    • Allocate KV blocks for prompt tokens                  │
    │    • Run full forward pass over prompt                     │
    │    • Emit first token                                      │
    │                                                            │
    └─── Decode phase ───────────────────────────────────────────┤
         • Allocate one new KV block per step (if needed)        │
         • Run single-token forward pass                         │
         • Sample → emit token chunk → check stop condition      │
         • Loop until EOS or max_tokens                          │
                                                                 │
OutputProcessor ◄────────────────────────────────────────────────┘
    │
    ▼
HTTP Response (streaming or complete)
```

### 6.1 Prefill Phase

All prompt tokens are processed in a **single forward pass** (or across multiple chunked steps — see §6.3). KV vectors for every prompt position are written into allocated blocks. The first output token is produced at the end of prefill.

### 6.2 Decode Phase

Each decode step:
1. Reads KV vectors from the block table (paged attention)
2. Runs a single-token forward pass
3. Samples the next token
4. Appends the new KV vector to the current block (or allocates a new block when the current one is full)
5. Checks stop conditions (EOS token, max length, stop strings)

### 6.3 Chunked Prefill

When `enable_chunked_prefill = true`, long prompts are split into chunks of at most `max_num_batched_tokens` tokens. This allows the scheduler to interleave prefill chunks with decode steps, reducing time-to-first-token for concurrent requests.

---

## 7. Attention Mechanisms

Two attention kernels are available under `models/shared/attention/`:

| Kernel | File | Use case |
|---|---|---|
| Batched attention | `batched.rs` | Standard multi-head attention for prefill |
| Paged attention | `paged.rs` | Block-table-based attention for decode |

**Paged attention** is the key enabler of the KV-cache design: instead of a contiguous KV tensor per sequence, the kernel reads from a block table that maps logical positions to physical block slots. This allows:
- Non-contiguous memory allocation
- Block sharing between sequences (prefix caching, beam search)
- Fine-grained eviction and swapping

**Flash Attention** integration is listed as an optimisation opportunity (see §13).

---

## 8. Metal / Apple Silicon Optimisations

Izwi is designed with Apple Silicon as the primary target. Several subsystems have Metal-specific code paths:

### Unified Memory Awareness

`MetalKVCacheManager` is aware that CPU and GPU share the same physical memory on Apple Silicon. `unified_memory_fraction` controls how much of the total unified memory budget is reserved for KV blocks, avoiding over-commitment that would trigger OS memory pressure.

### Memory Pressure Handling

The engine subscribes to macOS memory pressure notifications. Responses are tiered:

| Pressure Level | Engine Response |
|---|---|
| `Normal` | Standard operation |
| `Warning` | Reduce active batch size; increase block eviction rate |
| `Critical` | Halt new prefills; aggressively evict cold KV blocks |

### Optimal Block Sizing

`MetalKVCacheConfig::optimal_block_size` is tuned to align KV blocks with Metal's internal page size, reducing fragmentation and improving GPU cache utilisation.

### Serial Execution on MPS

Because Metal command buffers are not thread-safe, `NativeExecutor::can_parallelize_requests` returns `false` for MPS devices. The `EngineCore::step()` loop runs decode and prefill sequentially on Metal, avoiding command-buffer races at the cost of reduced CPU parallelism.

### `KvCacheBackend` Selection

At startup, `EngineCore` inspects the configured device and selects:
- `KvCacheBackend::Metal(MetalKVCacheManager)` — when Metal is enabled
- `KvCacheBackend::Standard(KVCacheManager)` — for CPU / other backends

---

## 9. Configuration Reference

All engine parameters are centralised in `EngineCoreConfig` (`engine/config.rs`).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_type` | `ModelVariant` | — | Model to load |
| `model_path` | `PathBuf` | — | Path to weights directory |
| `max_num_seqs` | `usize` | 256 | Max concurrent sequences |
| `max_num_batched_tokens` | `usize` | 8192 | Token budget per step |
| `max_model_len` | `usize` | model-dep. | Max sequence length |
| `block_size` | `usize` | 16 | KV block size in tokens |
| `num_cpu_blocks` | `usize` | auto | CPU KV block count |
| `num_gpu_blocks` | `usize` | auto | GPU KV block count |
| `enable_chunked_prefill` | `bool` | `true` | Chunked prefill |
| `backend` | `enum` | `auto` | Backend preference (`auto`, `cpu`, `metal`, `cuda`) |
| `unified_memory_fraction` | `f32` | 0.85 | Metal memory budget |
| `scheduling_policy` | `SchedulingPolicy` | `Fcfs` | Scheduler policy |
| `enable_kv_quantization` | `bool` | `false` | Int8 KV quantization |
| `streaming_chunk_tokens` | `usize` | 1 | Tokens per stream chunk |

`WorkerConfig` is derived from `EngineCoreConfig` and passed to `NativeExecutor` at construction time.

---

## 10. API Surface

The HTTP layer is implemented in `crates/izwi-server/src/api/`. The main router (`router.rs`) nests a mixed first-party and compatibility surface under `/v1`:

```
/v1
 ├── (internal)
 ├── first-party persisted resources and realtime APIs
 ├── openai-compatible endpoints
 └── admin
```

Static UI assets are served from the same router.

### 10.1 First-Party Persisted Resource Endpoints

These routes back the desktop UI's saved history and reusable assets. Canonical routes follow plural resource naming; legacy `/records` aliases are kept temporarily for compatibility.

| Method | Path | Description |
|---|---|---|
| `GET, POST` | `/v1/transcriptions` | List or create saved transcription records |
| `GET, DELETE` | `/v1/transcriptions/:id` | Fetch or delete a saved transcription record |
| `GET` | `/v1/transcriptions/:id/audio` | Fetch stored transcription source audio |
| `GET, POST` | `/v1/diarizations` | List or create saved diarization records |
| `GET, PATCH, PUT, DELETE` | `/v1/diarizations/:id` | Fetch, update, or delete a saved diarization record |
| `GET` | `/v1/diarizations/:id/audio` | Fetch stored diarization source audio |
| `POST` | `/v1/diarizations/:id/reruns` | Re-run diarization from a saved record's source audio |
| `GET, POST` | `/v1/text-to-speech-generations` | List or create saved TTS generations |
| `GET, DELETE` | `/v1/text-to-speech-generations/:id` | Fetch or delete a saved TTS generation |
| `GET` | `/v1/text-to-speech-generations/:id/audio` | Fetch generated TTS audio |
| `GET, POST` | `/v1/voice-design-generations` | List or create saved voice design generations |
| `GET, DELETE` | `/v1/voice-design-generations/:id` | Fetch or delete a saved voice design generation |
| `GET` | `/v1/voice-design-generations/:id/audio` | Fetch generated voice design audio |
| `GET, POST` | `/v1/voice-clone-generations` | List or create saved voice clone generations |
| `GET, DELETE` | `/v1/voice-clone-generations/:id` | Fetch or delete a saved voice clone generation |
| `GET` | `/v1/voice-clone-generations/:id/audio` | Fetch generated voice clone audio |
| `GET, POST` | `/v1/voices` | List or create reusable saved voices |
| `GET, DELETE` | `/v1/voices/:voice_id` | Fetch or delete a saved voice |
| `GET` | `/v1/voices/:voice_id/audio` | Fetch saved voice reference audio |

### 10.2 OpenAI-Compatible Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/audio/speech` | Text-to-speech synthesis |
| `POST` | `/v1/audio/transcriptions` | Speech-to-text (Whisper) |
| `POST` | `/v1/audio/translations` | Speech translation |
| `POST` | `/v1/chat/completions` | Chat / LLM completions |
| `GET` | `/v1/models` | List available models |
| `POST` | `/v1/responses` | Structured response generation |

Sub-routers: `audio`, `chat`, `models`, `responses` (defined in `api/openai/mod.rs`).

### 10.3 Admin Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/admin/models` | List all known model variants |
| `POST` | `/v1/admin/models/:variant/download` | Download model weights |
| `POST` | `/v1/admin/models/:variant/load` | Load model into engine |
| `POST` | `/v1/admin/models/:variant/unload` | Unload model from engine |
| `GET` | `/v1/admin/models/:variant` | Get model info |
| `DELETE` | `/v1/admin/models/:variant` | Delete model weights |

---

## 11. Unimplemented / Planned Features

The following features are scaffolded or partially implemented but not yet active:

| Feature | Status | Notes |
|---|---|---|
| **Speculative Decoding** | Stub only | Draft model infrastructure not wired |
| **Silero VAD** | TODO | Energy-based VAD is active; Silero integration pending |
| **KV Cache Quantization (Int8)** | Config flag exists | Quantization kernel not yet applied |
| **Flash Attention** | Planned | Standard attention used; Flash Attention would reduce memory bandwidth |
| **Prefix Caching** | Planned | Block sharing infrastructure exists; hash-based prefix lookup not implemented |
| **Beam Search** | Planned | Sampling infrastructure supports it; beam expansion logic pending |
| **CUDA / ROCm Backend** | Planned | `ExecutionBackend` enum has slots; only CPU and Metal are active |

---

## 12. Extension Points

### Adding a New Model Family

1. Add a variant to `ModelFamily` and `ModelTask` in `catalog/variant.rs`.
2. Implement `family()`, `primary_task()`, and `backend_hint()` arms for the new variant.
3. Create a loader in `families/<new_family>/`.
4. Implement `ModelExecutor` for the new architecture in `models/architectures/`.
5. Add an `Active*Decode` state struct in `executor.rs` if the model has incremental decode state.
6. Wire a runtime handler in `runtime/` if task-specific orchestration is needed.

### Adding a New Scheduler Policy

1. Add a variant to `SchedulingPolicy` in `scheduler.rs`.
2. Implement the scheduling logic in `Scheduler::schedule()`.
3. Expose the new policy via `EngineCoreConfig`.

### Adding a New Execution Backend

1. Add a variant to `ExecutionBackend` in `backends/mod.rs`.
2. Implement `BackendRouter` selection logic for the new backend.
3. Implement `ModelExecutor` for the backend in a new module.
4. Add device initialisation in the engine startup path.

---

## 13. Optimisation Opportunities & Recommendations

### Near-Term (High Impact)

| Opportunity | Expected Gain |
|---|---|
| **Flash Attention** | 2–4× memory bandwidth reduction during decode; lower latency |
| **Prefix Caching** | Eliminate redundant prefill for shared prompt prefixes (e.g., system prompts) |
| **KV Int8 Quantization** | ~50% KV memory reduction; enable larger batches |
| **Silero VAD** | More accurate silence detection; reduce wasted decode steps |

### Medium-Term

| Opportunity | Notes |
|---|---|
| **Speculative Decoding** | Draft model must be same family; requires beam-compatible sampler |
| **Continuous Batching Tuning** | Profile `max_num_batched_tokens` vs. latency on target hardware |
| **Metal Kernel Fusion** | Fuse attention + softmax + projection into a single Metal kernel |

### Architecture Recommendations

- **Separate prefill and decode workers** — vLLM's "disaggregated prefill" pattern can further reduce head-of-line blocking for long prompts.
- **Block-level prefix hashing** — implement a `HashMap<BlockHash, BlockId>` in `KVCacheManager` to enable automatic prefix block reuse.
- **Async KV swap** — use Metal's async blit encoder to overlap CPU↔GPU block swaps with the next decode step.
- **Metrics exposure** — expose `RuntimeTelemetrySnapshot` via a Prometheus-compatible `/metrics` endpoint for production observability.
