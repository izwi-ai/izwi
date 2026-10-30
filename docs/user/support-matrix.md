---
title: "Runtime Support Matrix"
description: "Supported operating systems, hardware backends, deployment paths, and API maturity for Izwi."
sidebarTitle: "Support Matrix"
icon: "clipboard-check"
---
This page is the public support contract for Izwi's current runtime surfaces.

It answers four questions:

1. Which OS and hardware combinations are supported?
2. Which shipped artifact types expose which backends?
3. Which deployment targets are considered supported?
4. Which API surfaces are stable vs preview?

If another page says something different, this page should win.

---

## Backend Matrix

| Surface | OS / Hardware | Backend status | Support level | Notes |
|---------|----------------|----------------|---------------|-------|
| **Desktop app from GitHub Releases** | Apple Silicon, macOS 15+ | `metal` | Stable | The release selects Metal automatically when the runtime is eligible. |
| **Desktop app from GitHub Releases** | macOS 13.3-14 | `cpu` | Stable | The same release remains launchable and deliberately avoids Candle's Metal constructor. The desktop app needs macOS 13.3+ (WebKit 16.4) for its UI. |
| **Desktop app from GitHub Releases** | Linux x86_64 | `cpu` | Stable | Native Linux installers are CPU-only and do not bundle CUDA runtime libraries. |
| **Desktop app from GitHub Releases** | Windows x86_64 | `cpu` | Stable | Native Windows installers are CPU-only and do not bundle CUDA runtime DLLs. |
| **Terminal bundle from GitHub Releases** | Linux x86_64 | `cpu` | Stable | Linux terminal tarballs contain the public CPU-only CLI, server, and desktop shell binaries. |
| **Terminal bundle from GitHub Releases** | Apple Silicon, macOS 15+ | `metal` | Stable | Metal is compiled into the macOS bundle and runtime-gated. |
| **Terminal bundle from GitHub Releases** | macOS 12-14 | `cpu` | Stable | Metal requests fall back to CPU before device construction. |
| **Terminal bundle from GitHub Releases** | Windows x86_64 | `cpu` | Stable | Windows terminal zips contain the public CPU-only CLI, server, and desktop shell binaries. |
| **Source build** | Apple Silicon, macOS 15+ with `--features metal` | `metal` | Stable | Recommended GPU path on supported macOS runtimes. |
| **Source build** | Linux x86_64 with `--features cuda` and CUDA toolkit installed | `cuda` | Supported | Useful for development, custom builds, and debugging outside Docker. Requires a compatible NVIDIA driver/toolkit environment. |
| **Source build** | Windows with `--features cuda` and CUDA toolkit installed | `cuda` | Preview | Useful for development and custom validation. Native Windows release artifacts remain CPU-only. |
| **Docker `production` target** | Linux x86_64 | `cpu` | Stable | CPU-only container image. |
| **Docker `production-cuda` target / `docker compose --profile cuda`** | Linux x86_64 + NVIDIA GPU | `cuda` | Preview | Shipped CUDA binary path. The final image is based on `nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04` and needs NVIDIA driver 580 or newer. It targets compute capability 8.0 (Ampere) and newer; set `CUDA_COMPUTE_CAP` at build time for another architecture. |

---

## Deployment Matrix

| Deployment target | Status | Notes |
|-------------------|--------|-------|
| **Single-user macOS desktop evaluation** | Stable | Best-supported path for local evaluation. |
| **Single-host Linux server on CPU** | Stable | Supported via GitHub Release packages, source builds, and the Docker CPU image. |
| **Single-host Linux server on NVIDIA GPU** | Supported / Preview by artifact | Use the Docker CUDA image/profile, or build from source with `--features cuda`. Native Linux release artifacts are CPU-only. |
| **Windows desktop evaluation** | Stable CPU | Native Windows release artifacts are CPU-only. CUDA on Windows is source-build preview only. |
| **Docker Compose on CPU** | Stable | Use the default `izwi` service. |
| **Docker Compose on NVIDIA GPU** | Preview | Use `docker compose --profile cuda up -d izwi-cuda`; name the `izwi-cuda` service, because the profile otherwise also starts the CPU `izwi` service on the same port. The default setup is for development (published on `127.0.0.1`, no API key); the production overlay requires a key. See [Docker](/installation/docker). The image targets compute capability 8.0 and newer; set `CUDA_COMPUTE_CAP` at build time for another architecture. |
| **Kubernetes / Helm / multi-node production orchestration** | Not yet supported | Not published in OSS today. |

---

## API Surface Maturity

The runtime exposes both compatibility APIs and first-party local workflow APIs under `/v1`.
When the server is running, open `/docs` for the local Scalar API reference or
`/openapi.json` for the raw OpenAPI document. The generated OpenAPI document
covers the stable OpenAI-compatible contract, `/v1/responses` preview routes,
readiness probes, and Scalar sidebar entries for preview first-party, operator,
and realtime route families. Detailed preview behavior is documented in the
[API Reference](/api).

| Surface | Status | Notes |
|---------|--------|-------|
| **`POST /v1/audio/speech`** | Stable | Core OpenAI-compatible TTS surface. Native OSS output formats are WAV and raw PCM; recognized compressed names require explicit WAV fallback opt-in until bundled encoders are added. |
| **`POST /v1/audio/transcriptions`** | Stable | Core OpenAI-compatible transcription surface. |
| **`POST /v1/audio/align`** | Stable | Izwi extension for word-level forced alignment of reference text to audio. |
| **`POST /v1/chat/completions`** | Stable | Core OpenAI-compatible chat surface. |
| **`GET /v1/models`** | Stable | Live model catalog / availability surface. |
| **Operational probes (`/livez`, `/readyz`, `/v1/live`, `/v1/ready`)** | Stable | Use `/livez` for cheap liveness and `/readyz` for readiness or deployment healthchecks. `/v1/health` remains the richer status payload. |
| **Local OpenAPI reference (`/docs`, `/openapi.json`)** | Stable | Served by the same `izwi-server` process for the OpenAI-compatible contract, probes, and Scalar navigation for preview route families. |
| **Markdown API reference (`/docs/api` on the website, `docs/user/api.md` in the repo)** | Stable | Provides detailed behavior for the broader preview first-party, operator, and realtime route surface. |
| **Local CLI workflows (`izwi serve`, `izwi pull`, `izwi tts`, `izwi transcribe`)** | Stable | Primary user-facing local runtime workflows. |
| **`POST /v1/responses` and response-object lifecycle routes** | Preview | Response objects are stored in bounded process memory for compatibility convenience. `store:false` skips retention; retained records can be evicted and are lost on server restart. |
| **`/v1/admin/models*` model-management APIs** | Preview | Operator-oriented local model lifecycle and capability APIs; auth and long-term contract are not finalized. |
| **Persisted speech and voice workflow APIs (`/v1/speech-to-text/jobs*`, `/v1/diarizations*`, `/v1/text-to-speech*`, `/v1/voice-designs*`, `/v1/voice-clones*`, `/v1/voices*`, `/v1/studio/*`)** | Preview | Powerful local product APIs, but the public compatibility/support contract is still evolving. Both speech-to-text diarization jobs and direct diarization records are supported first-party surfaces. |
| **Local chat, agent, and voice state APIs (`/v1/chat/threads*`, `/v1/agent/sessions*`, `/v1/voice/profile`, `/v1/voice/observations`, `/v1/voice/sessions*`)** | Preview | Agent session metadata is process-local and bounded today. Linked chat threads, voice sessions, voice turns, and voice observations are the durable SQLite-backed local stores. Voice sessions now include REST create/update/end/delete/turn-list/export controls for external apps. |
| **Local media lifecycle (`/v1/media*`)** | Preview | OSS local media can be listed, uploaded from base64 payloads, downloaded by catch-all relative path, and deleted. Provider-backed object storage can wrap the same route family; listing may be unavailable unless the provider exposes a local media root. |
| **Realtime WebSocket APIs (`/v1/speech-to-text/realtime/ws`, `/v1/voice/realtime/ws`)** | Preview | Low-latency browser-facing protocols for streaming transcription and voice AI conversations. |

---

## CUDA Caveats

- Linux and Windows GitHub Releases keep public binary names unchanged: `izwi` and `izwi-server` on Linux, `izwi.exe` and `izwi-server.exe` on Windows.
- Linux and Windows GitHub Release artifacts are CPU-only and must not contain CUDA runtime libraries or private CUDA binaries.
- Release installers do not replace the host NVIDIA driver. CUDA acceleration requires a compatible NVIDIA driver and CUDA-capable GPU.
- Source builds still require the CUDA toolkit and remain useful for development or fallback validation.
- The Docker CUDA image/profile is the CUDA distribution path for NVIDIA Linux hosts. It targets compute capability 8.0 (Ampere) and newer unless built with another `CUDA_COMPUTE_CAP`.
- On macOS 15+, the recommended GPU path is Metal, not CUDA. macOS 12-14 are CPU-only.
- `Chatterbox-500M` and `Chatterbox-Multilingual-500M` build for CUDA but are compile-checked only: they have not yet run on an NVIDIA GPU. Their T3 speech-token models run in BF16 on CUDA devices with BF16, F16 on Metal and F32 on CPU, behind a load-time self-check that falls back to F32; their other networks run in F32. `Chatterbox-500M` is verified end to end on CPU and Metal, and `Chatterbox-Multilingual-500M` on Metal.
- `Maya1-3B` builds for CUDA but is compile-checked only: it has not yet run on an NVIDIA GPU. Its decoder runs in BF16 on CUDA devices with BF16 and F32 on CPU, each behind a load-time range probe that falls back to F32, and in F16 on Metal, where a failed probe refuses the load (F32 would need 13.2 GB of weights); SNAC runs in F32 everywhere. `Maya1-3B-GGUF` keeps its decoder's weights in Q8_0 and runs it in F32 on every backend, with no probe. Both are verified on Metal (16 GB M1 Pro): the decoder tracks upstream's BF16 logits, the release server passes its end-to-end checks (voices, tags, streaming, cancellation, reload, Whisper transcripts), and requests at four and eight in flight sound exactly as they do alone; CPU runs the unit and layer-truncated parity tests only.
- `Maya2-Native-2.5B` has a CUDA path but has not yet run on an NVIDIA GPU. Its Qwen3 backbone runs in BF16 on CUDA devices with BF16, behind a load-time range probe that falls back to F32; in F16 on Metal, where a failed probe refuses the load (F32 would need 10 GB of weights); and in F32 on CPU. Its depth decoder runs in F16 on Metal and CUDA and in F32 on CPU or after an F32 fallback; Mimi runs in F32 everywhere, and the ECAPA speaker encoder for clones runs on the CPU in F32.
- `LFM3-Audio-2B-GGUF` and `LFM3-Audio-2B-Q8_0-GGUF` build for CUDA but are compile-checked only: they have not yet run on an NVIDIA GPU. On CUDA the quantized language model, speech encoder, MaskGIT depth decoder and LiquidCodec decoder use Candle CUDA kernels, while mel features, sampling and the inverse FFT run on the host. LFM3 Audio is verified end to end on real weights on CPU and Metal.
- `Nemotron-3.5-Transcribe-30B-A4B-FP8` builds for CUDA but is compile-checked only: it has not run on an NVIDIA GPU. It needs compute capability 8.0 or newer. See [Nemotron 3.5 Transcribe](#nemotron-35-transcribe).

### Nemotron 3.5 Transcribe

`Nemotron-3.5-Transcribe-30B-A4B-FP8` runs on CPU, Metal and CUDA.

| Backend | Resident weights | Activations |
|---|---|---|
| CPU | About 33 GiB: FP8 linears re-quantized to Q8_0 at load | F32 |
| Metal | About 31 GiB: raw FP8 | F32 |
| CUDA (compute capability 8.0+) | About 31 GiB: raw FP8; the routed experts stay unpadded, the other FP8 linears are padded for the FP8 kernels | BF16 |

Each request adds about 48 MiB of decoder state plus its attention KV. A load
that the backend's memory budget cannot hold is refused before any weights are
read. `IZWI_NEMOTRON_TRANSCRIBE_FP8_MODE=vllm-fp8` rounds activations to FP8
as vLLM's W8A8 path does; it is a parity-testing mode, not a speed-up.

| Capability | Status |
|---|---|
| Offline transcription, translation, context terms | Supported |
| Speaker labels and speaker-attributed ASR jobs | Supported |
| Word timestamps from its own CTC head | Supported |
| [Single-model diarization](/features/diarization#single-model-diarization) (a diarization job that names it as `model`) | Supported |
| ASR stage of a diarization pipeline | Not supported |
| Live transcription, voice-mode ASR, audio chat | Not supported (offline model) |

### Qwen3.8 CUDA weight residency

`Qwen3.8-27B-FP8` keeps its official block-scaled FP8 checkpoint as the resident
format on CUDA by default. Projections stay as the checkpoint's E4M3 bytes plus one
F32 scale per 128x128 block (about 24.7 GB), and the compact FP8 kernels apply
`weight_scale_inv` inside every matrix multiplication with 16-bit activations
(software FP8 decode, not FP8 Tensor Core arithmetic). Row-compatible projections
are packed into one matrix. Runtime diagnostics report
`resident_representation: native_block_fp8_with_q8_fallback_dense_bf16` and
`fp8_execution_mode: native_block_fp8_auto_self_checked`.

The runtime enables this only after checks on the device, before load admission:
every projection shape must fit the kernel contract, the kernels must match a
reference at the checkpoint's real inner dimensions, and a probe on real weights
read from the checkpoint must match a host reference built from the same bytes.
The load then checks each kind of projection, packs included, against the original
checkpoint bytes. If any check fails, the model loads with the compressed Q8_0
fallback instead (`weight_scale_inv` applied, then requantized to resident Candle
Q8_0, about 26.2 GB), reported as `q8_0_requantized_projections_with_dense_bf16`
and `fp8_execution_mode: q8_0_fallback:<reason>`. A check that fails during
loading fails that load, and the next load uses Q8_0 with matching memory
admission. Set `IZWI_CUDA_PROJECTION_BACKEND=q8` or
`IZWI_QWEN38_FP8_PROJECTIONS=legacy` to choose Q8_0, and
`IZWI_QWEN38_FP8_GEMV=legacy` to keep the reference decode kernel.

With native block FP8, the LM head (2.54 GB in BF16) is also quantized at load to
the same block-FP8 format (1.27 GB), which the target logits and every MTP draft
read. This is a deliberate precision reduction: a host quality gate compares FP8
and BF16 logits on probe rows across the vocabulary before the head is used, and
`fast_paths.fp8_lm_head` reports the observed error. Logits stay BF16. Set
`IZWI_QWEN38_FP8_LM_HEAD=legacy` to keep the BF16 head.

Both representations are designed to fit 40/48 GB-class CUDA devices with a
resource-fitted context, subject to actual free VRAM and allocator headroom. Native
block FP8 is not CUDA-certified until the corresponding NVIDIA evidence is
retained.

### Qwen3.8 Metal weight residency

On Apple Silicon, `Qwen3.8-27B-FP8` also keeps its projections as the checkpoint's
block-FP8 bytes (about 28.9 GiB resident including a 1 GiB prefill scratch, against
about 51 GiB for the F16 expansion), so 48 GB and 64 GB Macs can load it. Decode
rows (up to 8) run an FP8 matrix-vector kernel that decodes E4M3 in software with
F32 accumulation; longer prefill spans expand each projection to a transient F16
matrix for one matrix multiplication. The LM head and the other dense tensors stay
F16. Every kind of projection is checked against the original checkpoint bytes at
load; a projection whose matrix-vector kernel disagrees runs the expansion instead,
and a check that still fails fails that load, after which the next load expands to
F16. Diagnostics report `resident_representation: native_block_fp8_with_dense_f16`.
Set `IZWI_METAL_PROJECTION_BACKEND=expanded` (or `[runtime.performance.metal]
projection_backend = "expanded"`) to expand to F16 at load.

### Qwen3.8 CPU weight residency

On the CPU, `Qwen3.8-27B-FP8` keeps its block-FP8 projections and its LM head as
Q8_0 (the block scales applied, then requantized to 8-bit blocks of 32 values),
about 30.7 GiB resident instead of about 102 GiB for the F32 expansion; the
embedding and other dense tensors stay F32. Decode runs Candle's Q8_0 matrix
multiplication, which also quantizes activations to 8 bits; prompt spans longer
than 8 tokens expand row tiles of the weights to F32 for one F32 matrix
multiplication. This changes CPU numerics: on the test fixtures, logits sit 2-4%
(relative L2) from the F32 expansion, and greedy tokens can flip at near-ties.
Diagnostics report `resident_representation: q8_0_projections_with_dense_f32`. Set
`IZWI_CPU_PROJECTION_BACKEND=expanded` (or `[runtime.performance.cpu]
projection_backend = "expanded"`) to keep the F32 expansion, and
`IZWI_QWEN38_CPU_LM_HEAD=dense` to keep only the head in F32.

### Qwen3.8 multi-token prediction defaults

Native Qwen3.8 MTP is enabled by default with one draft token on CPU and
Metal. On CUDA the default (`cuda.mtp = "auto"`) leaves the MTP head out
whenever CUDA graph decode is planned (see below): until MTP verification
also runs in CUDA graphs, one graph launch per token is expected to beat
MTP's eager draft and verify rounds. The head still loads on CUDA when graph
decode is not planned (`IZWI_QWEN38_CUDA_GRAPHS=legacy`,
`IZWI_CUDA_DECODE_GRAPHS=off`, `IZWI_CUDA_FUSED_DECODE=off`, or the Q8_0
projection fallback). Set `IZWI_QWEN38_MTP=1` (or `IZWI_CUDA_MTP=1`) to load
the head on CUDA anyway, and `IZWI_QWEN38_MTP=0` to disable MTP everywhere;
restart or reload the model after changing either. The runtime diagnostics
report the decision as `optimization_evidence.mtp.decision`. This CUDA
default awaits a fixed-prompt throughput comparison on the H100; until then,
`IZWI_QWEN38_MTP=1` restores the previous behavior. The
`IZWI_QWEN38_MTP_DRAFT_TOKENS` setting controls proposal depth rather than
enablement; its default is `1`, and supported explicit values are `1` through
`3`.

### Qwen3.8 CUDA graph decode

On CUDA, single-row decode without an MTP head runs through captured CUDA
graphs when the projections are native block FP8 and the fused decode kernels
pass their load-time checks:

- The whole step (the token embedding, all 64 layers, the final norm and the
  block-FP8 LM head) is one graph, driven by a device step buffer that holds
  the token, position, KV slot and block table. Where it does not apply (an
  image position, or `IZWI_QWEN38_STEP_GRAPH=legacy`), 17 graph segments run
  with attention eager between them.
- Each graph is warmed eagerly, with its logits checked against the standard
  decode on a copy of the row state, then captured and verified twice against
  an eager run before it replays. A KV arena that grows forces a recapture.
  Any failure turns graph decode off for the process with the reason in the
  diagnostics, and decode continues on the eager fused path.
- A decode quantum of up to 8 tokens per engine step
  (`IZWI_QWEN38_DECODE_QUANTUM`, 1 to 64) keeps its width past a request's soft
  scheduling deadline while the row decodes alone. Once the whole-step graph
  replays, the quantum queues step k+1 before it reads token k, choosing tokens
  with the device sampler (greedy or seeded sampling with penalties). Requests
  that ask for logprobs decode one step at a time.
- Load admission reserves graph decode's working memory, three recurrent-state
  slabs plus 64 MiB (about 513 MiB for the 27B checkpoint), whenever graph
  decode is planned.

`cuda_graphs` in the runtime diagnostics reports the resolution, the phase
(`warm`, `capture`, `verify`, `replay` or `disabled`), the counters, and the
whole-step graph and decode quantum. Switches: `IZWI_QWEN38_CUDA_GRAPHS=legacy`
(Qwen3.8 only), `IZWI_CUDA_DECODE_GRAPHS=off` (every model family),
`IZWI_QWEN38_STEP_GRAPH=legacy` (segments only),
`IZWI_QWEN38_DECODE_PIPELINE=legacy` (one step at a time) and
`IZWI_QWEN38_FUSED_AB_GEMV=legacy` (cuBLAS for the DeltaNet alpha/beta
projection). CUDA graph decode is not CUDA-certified until the corresponding
NVIDIA evidence is retained.

---

## Managed Inference-State Support

This matrix describes ownership and provider classification in the current
source tree. It is narrower than general model availability: a model can be
available while a particular cache provider or hardware cell remains
uncertified.

| Backend | ABI-v2 physical state | Paged provider class | Current boundary |
|---|---|---|---|
| **CPU** | Supported | Portable | Dense F32/F16/BF16 KV policy; direct paged write, prefill, decode, zero, and copy. |
| **Metal** | Supported when built with `metal` | Portable | Direct native Metal operations are intentionally classified Portable until a separately measured optimized cell is promoted. CPU and Metal share one unified-memory authority. |
| **CUDA (`cuda-base`)** | Supported when built with CUDA | Portable | Native direct-page provider. CUDA compilation and device execution are separate release gates. |
| **CUDA (`cuda` / `flash-attn`)** | Supported when built with CUDA and FlashAttention | Portable, or Optimized-eligible per resolved cell | Optimized eligibility requires F16/BF16, full attention, page size divisible by 32, zero first-page offsets, and equal K/V head dimensions divisible by 8 and no larger than 512. Runtime and performance certification require retained NVIDIA evidence. Ineligible cells remain Portable. |

The runtime fails model loading when a required ABI-v2 operation set or exact
model/capability route is incomplete. Source/build eligibility never upgrades
a route to runtime-validated or performance-certified evidence. The runtime
does not silently switch back to a model-owned cache.

### State topology by model family

The load path currently publishes ABI-v2 state as follows:

| Model family / route | ABI-v2 topology | Provider-promotion status |
|---|---|---|
| **Qwen3.5 chat** | Composite retained paged/ring state | Portable route validated; eligible CUDA paged cells remain unverified until NVIDIA evidence is retained |
| **Qwen3.8 chat (`Qwen38Chat`)** | Composite retained paged/tensor state under an independent family contract | Separate conformance and backend evidence required; Qwen3.5 results do not certify this family |
| **Qwen3 ASR** | Retained paged state plus bounded invocation workspace | Portable route validated; eligible CUDA cells remain unverified until NVIDIA evidence is retained |
| **Qwen3 TTS** | Retained paged predictor state plus bounded invocation state/workspace | Portable route validated; eligible CUDA cells remain unverified until NVIDIA evidence is retained |
| **LFM3 chat** | The LFM2.5 chat state: retained paged attention (global and 256-token sliding-window layers share one page table) plus the transactional ShortConv ring | Real-checkpoint parity on CPU and Metal; CUDA compile-checked, unverified on hardware |
| **LFM2.5 chat** | Retained paged attention plus transactional ShortConv ring state | Portable route validated; eligible accelerator cells remain unverified until hardware evidence is retained |
| **LFM2.5 Audio** | Bounded invocation-scoped paged/ring/composite state | ABI-v2 physical ownership; no Optimized attestation |
| **LFM3 Audio** | Speech, transcription and voice-chat replies run as retained rows that batch: each request keeps the language model's attention pages and ShortConv ring in the managed arena between steps, and concurrent requests decode together (two rows at a time on CPU and Metal, the portable width, unless `IZWI_MAX_BATCH_SIZE` sets one), each producing exactly what it produces alone. Audio over 30 s transcribes in clips of at most 30 s, one engine dispatch each. A voice-chat turn's transcript runs as its own transcription row beside the reply. Beyond English, a real-weight check speaks one sentence in French, German, Spanish and Japanese: Whisper Large v3 Turbo transcribes LFM3 Audio's speech exactly in all four (error rate 0.00 on CPU and Metal), LFM3 Audio transcribes its own speech exactly, and it transcribes German, Spanish and Japanese spoken by Qwen3-TTS at 0.00–0.04. French is weaker: where Qwen3-TTS mispronounces a word, LFM3 Audio drifts into English (word error rate 0.50 on Metal). Realtime transcription (LiveASR) streams run as engine realtime sessions: each 0.96 s chunk prefills alone and its text decodes in batches across streams, and a stream continues in 29 s segments that hear the previous one's last 2.88 s again | Verified end to end on real weights on CPU and Metal; CUDA compile-checked only, unverified on hardware |
| **Whisper, Parakeet, VibeVoice ASR, Granite Speech, Voxtral ASR** | Bounded invocation-scoped physical state; Granite Speech 4.1 publishes both ASR routes; Granite Speech 5.0 leases its 16k-token decoder pages per invocation | ABI-v2 physical ownership; no Optimized attestation |
| **Nemotron ASR** | Offline bounded invocation state and retained realtime tensor state | ABI-v2 physical ownership; no Optimized attestation |
| **Nemotron 3.5 Transcribe** | Each request is a retained row: paged KV for the 6 attention layers plus the Mamba2 recurrent and convolution state (about 48 MiB a row) in the managed arena. Prompts prefill in resumable spans (512 tokens on CPU and Metal, 2,048 on CUDA) whether or not chunked prefill is on, so a one-hour recording never holds the device for its whole prefill; decode runs in batches across rows. A single-model diarization job runs as one such row. No prefix reuse | ABI-v2 physical ownership; no Optimized attestation |
| **VibeVoice, Fish S2, Voxtral TTS** | Bounded invocation-scoped physical state | ABI-v2 physical ownership; no Optimized attestation |
| **Chatterbox** | Retained paged T3 state (two guidance branches per request, decoded in batches across requests) plus bounded acoustic-stage scratch, run one step at a time so other requests interleave | Portable route validated on CPU and Metal; CUDA is compile-checked only until NVIDIA evidence is retained |
| **Maya1** | Retained paged decoder state (decoded in batches across requests) plus bounded SNAC-window scratch, one window per step so other requests interleave | Portable route validated on Metal for both releases, with exact solo parity under load; CUDA is compile-checked only |
| **Maya 2 Native** | Retained paged backbone state (decoded in batches across requests) plus bounded depth-decoder and Mimi-window scratch, one window per step so other requests interleave | Portable route validated on Metal and CPU, with exact solo parity under load; CUDA has not run on a GPU |
| **Sortformer diarization** | Bounded invocation-scoped physical state | ABI-v2 physical ownership; no Optimized attestation |
| **Stateless/catalog-only routes** | No retained mutable state, or no loaded state publication | No cache provider applies |

“ABI-v2 physical ownership” is not a claim of complete model-quality or
hardware performance certification. The exact model revision, capability,
backend, dtype, page geometry, attention semantics, and build feature cell must
pass its release lane.

### CUDA context policy

CUDA uses the native context reported by the loaded chat checkpoint. CPU and
Metal allocate the chat KV arena in full when the model loads, so without an
explicit `max_sequence_length` they fit the checkpoint's context to the memory
budget and cap it at 32,768 tokens; an idle model therefore no longer holds its
whole advertised context in unified memory. Set `max_sequence_length` (or
`IZWI_MAX_SEQUENCE_LENGTH`) to choose a longer context that the memory budget
can hold. Chat requests claim KV as they generate rather than reserving their
whole output budget, so concurrent requests share one arena. The current
native CUDA ceilings are:

| Model | CUDA ceiling | Source |
|---|---:|---|
| Qwen3.5 chat | 262,144 tokens | [Official Qwen3.5 config](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json) |
| Qwen3.8 chat | 262,144 logical tokens; effective context is resource-fitted | [Pinned official Qwen3.8 config](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/blob/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a/config.json) |
| LFM2.5 chat | 32,768 tokens (Izwi cap; the checkpoint advertises 128,000) | [Official LFM2.5 config](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/main/config.json) |
| LFM3 chat | 32,768 tokens by default on CPU and Metal; the checkpoint's 262,144 on CUDA, capped by the memory its arena fits at load; `IZWI_LFM3_CONTEXT_TOKENS` sets it on any backend | [Pinned LFM3-1.7B config](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd/config.json) |

Optional YaRN extensions are not included: the local adapters do not implement
their scaling parameters. Qwen3.8 fits its effective context to remaining
resources without changing the 262,144-token logical maximum. Other CUDA chat
routes reserve their selected paged KV capacity before publishing a model as
ready. Explicit context requests still fail when the requested state cannot
fit; the runtime does not overcommit VRAM.

For audio generation, automatic output budgets remain bounded. Explicit CUDA
requests can use the 32,768-token LFM2.5 Audio and Fish S2 contexts; Voxtral TTS
accepts the upstream deployment ceiling of 2,048 output frames. CPU and Metal
retain their existing limits. VibeVoice ASR uses its documented 60-minute
single-pass window on CUDA by default; `IZWI_VIBEVOICE_ASR_CUDA_MAX_AUDIO_SECS`
can lower that window but cannot raise it above 60 minutes. See the
[LFM2.5 Audio model card](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B),
[Fish S2 config](https://huggingface.co/fishaudio/s2-pro/blob/main/config.json),
[Voxtral deployment](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/deploy/voxtral_tts.yaml),
and [VibeVoice ASR model card](https://huggingface.co/microsoft/VibeVoice-ASR).

Qwen3 ASR accepts up to 1,200 seconds per CUDA transcription chunk; CPU and
Metal retain the checkpoint preprocessor window, and the CUDA route does not
silently truncate audio at that limit. The Qwen3 forced aligner takes one
window of up to 180 seconds on every backend, so its timings do not depend on
the backend; longer audio is aligned in sub-windows cut at quiet points.

Voxtral Realtime CUDA offline decoding uses the loaded model position range
(131,072 tokens in the current checkpoint) instead of the portable 1,024-frame
service limit. Its physical paged cache rotates the 8,192-token attention
window with one spare page; inputs beyond the loaded position range fail
explicitly instead of silently dropping the tail.

Architectural processing windows are not expanded. Whisper still processes
30-second encoder windows, Kokoro still uses at most 510 phonemes per model
chunk, and streaming ASR/diarization families retain their bounded working
buffers while the runtime orchestrates longer inputs.

### Cache policy support

| Policy | Status | Notes |
|---|---|---|
| Dense `float16`, `bfloat16`, `float32` KV | Supported by configuration boundary | Backend/model negotiation can still reject an exact incompatible cell. |
| `int8` / `q4` KV | Unsupported | Parsed for migration diagnostics, then rejected before model readiness. |
| CUDA FP8 E4M3 KV | Experimental, blocked | Kernel experiments exist, but scaled storage, scale-aware mutation/accounting, and NVIDIA evidence are incomplete. Arena and direct dispatch both fail closed to dense KV. |
| Prefix reuse | Catalog default | Engaged per model and backend where the catalog cell is supported (Qwen3.8 on CPU, Metal and CUDA, without its MTP head; Qwen3.6 on CPU and CUDA; LFM2.5 and LFM3 chat on CPU and Metal, with resumable prefill on), with a per-process namespace and bounded page and snapshot budgets. `IZWI_ENABLE_PREFIX_CACHING=0` turns it off; `IZWI_QWEN38_PREFIX_REUSE=0` for Qwen3.8 alone. |
| Optimized-provider demotion | Supported | Set `IZWI_KV_DISABLE_OPTIMIZED_PROVIDER=1`; this can only demote to the validated Portable route. |
| Tiered/offloaded or distributed KV | Not supported | No production ownership or eviction contract is published for these modes. |

Eligible CUDA routes automatically use resident block-table metadata,
admission-grown physical backing, device/shape-keyed attention selection,
bounded sampling readback, VRAM-tiered continuous batches, and stable one-pass
decode graph buckets. Graph keys include exact K/V and metadata tensor
identities, geometry, dtype, scale, and softcap. Partitioned decode, FP8
storage, unobserved/pre-Ampere devices, and any capture/replay failure retain
eager native decode. CPU and Metal do not enter this policy.

For configuration, counters, benchmarks, and rollback, see
[KV Cache Operations](/kv-cache-operations).

---

## Verification Guidance

Use the following expectations when validating a host:

- **Apple Silicon, macOS 15+:** build or install a Metal-capable binary and run with `--backend metal` or `IZWI_BACKEND=metal`.
- **macOS 12-14:** run with `--backend cpu`; `auto` and explicit `metal` requests fall back to CPU.
- **Linux/Windows GitHub Release:** run `izwi serve --backend cpu`, then `izwi status --detailed`.
- **Docker CUDA on NVIDIA Linux hosts:** run `docker compose --profile cuda up -d izwi-cuda`, then confirm the container selects CUDA through `/v1/health` or `izwi status --detailed`. With the production overlay, send `Authorization: Bearer <key>` (or export the same `IZWI_API_KEY` for the CLI). Eligible Candle FlashAttention and Qwen3/Qwen3.5 RoPE CUDA routes activate automatically; their environment variables remain explicit rollback switches. Qwen3.8 is an independently gated family and must report its own execution path and evidence.
- **Linux/Windows source build for CUDA:** build with `cargo build --release -p izwi-cli -p izwi-server --features cuda`, then run with `--backend cuda` or `IZWI_BACKEND=cuda`. The `cuda` wrapper includes Candle FlashAttention, while `cudnn` additionally enables matching Candle/cuDNN convolution paths.

---

## See Also

- [Installation](/installation)
- [Getting Started](/getting-started)
- [CLI Reference](/cli)
