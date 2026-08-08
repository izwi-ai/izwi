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
| **Desktop app from GitHub Releases** | macOS on Apple Silicon | `metal` | Stable | Desktop and terminal binaries bundled in the macOS release can use Metal acceleration. |
| **Desktop app from GitHub Releases** | Linux x86_64 | `cpu` | Stable | Native Linux installers are CPU-only and do not bundle CUDA runtime libraries. |
| **Desktop app from GitHub Releases** | Windows x86_64 | `cpu` | Stable | Native Windows installers are CPU-only and do not bundle CUDA runtime DLLs. |
| **Terminal bundle from GitHub Releases** | Linux x86_64 | `cpu` | Stable | Linux terminal tarballs contain the public CPU-only CLI, server, and desktop shell binaries. |
| **Terminal bundle from GitHub Releases** | macOS Apple Silicon | `metal` | Stable | Metal is compiled into the macOS build path. |
| **Terminal bundle from GitHub Releases** | Windows x86_64 | `cpu` | Stable | Windows terminal zips contain the public CPU-only CLI, server, and desktop shell binaries. |
| **Source build** | macOS Apple Silicon with `--features metal` | `metal` | Stable | Recommended GPU path on macOS. |
| **Source build** | Linux x86_64 with `--features cuda` and CUDA toolkit installed | `cuda` | Supported | Useful for development, custom builds, and debugging outside Docker. Requires a compatible NVIDIA driver/toolkit environment. |
| **Source build** | Windows with `--features cuda` and CUDA toolkit installed | `cuda` | Preview | Useful for development and custom validation. Native Windows release artifacts remain CPU-only. |
| **Docker `production` target** | Linux x86_64 | `cpu` | Stable | CPU-only container image. |
| **Docker `production-cuda` target / `docker compose --profile cuda`** | Linux x86_64 + NVIDIA GPU | `cuda` | Preview | Shipped CUDA binary path. The final image is based on `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`. When building on a machine without `nvidia-smi`, set `CUDA_COMPUTE_CAP` for the target GPU architecture. |

---

## Deployment Matrix

| Deployment target | Status | Notes |
|-------------------|--------|-------|
| **Single-user macOS desktop evaluation** | Stable | Best-supported path for local evaluation. |
| **Single-host Linux server on CPU** | Stable | Supported via GitHub Release packages, source builds, and the Docker CPU image. |
| **Single-host Linux server on NVIDIA GPU** | Supported / Preview by artifact | Use the Docker CUDA image/profile, or build from source with `--features cuda`. Native Linux release artifacts are CPU-only. |
| **Windows desktop evaluation** | Stable CPU | Native Windows release artifacts are CPU-only. CUDA on Windows is source-build preview only. |
| **Docker Compose on CPU** | Stable | Use the default `izwi` service. |
| **Docker Compose on NVIDIA GPU** | Preview | Use `docker compose --profile cuda up`; the profile runs the `izwi-cuda` service and may require `CUDA_COMPUTE_CAP` when built on a non-GPU machine. |
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
- The Docker CUDA image/profile is the CUDA distribution path for NVIDIA Linux hosts and may require `CUDA_COMPUTE_CAP` when built on a machine without `nvidia-smi`.
- On macOS, the recommended GPU path is Metal, not CUDA.

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
| **CUDA (`cuda` / `flash-attn`)** | Supported when built with CUDA and FlashAttention | Portable or Optimized per resolved cell | Optimized requires F16/BF16, full attention, page size divisible by 32, zero first-page offsets, and equal K/V head dimensions divisible by 8 and no larger than 512. Ineligible cells remain Portable. |

The runtime fails model loading when a required ABI-v2 operation set or exact
model/capability route is not certified. It does not silently switch back to a
model-owned cache.

### State topology by model family

The load path currently publishes ABI-v2 state as follows:

| Model family / route | ABI-v2 topology | Provider-promotion status |
|---|---|---|
| **Qwen3 chat** | Retained paged KV | Portable certified; eligible CUDA cells may be Optimized |
| **Qwen3.5 chat** | Composite retained paged/ring state | Portable certified; eligible CUDA paged cells may be Optimized |
| **Gemma 3 chat** | Retained paged KV with model-declared attention semantics | Portable certified; eligible CUDA cells may be Optimized |
| **Qwen3 ASR** | Retained paged state plus bounded invocation workspace | Portable certified; eligible CUDA cells may be Optimized |
| **Qwen3 TTS** | Retained paged predictor state plus bounded invocation state/workspace | Portable certified; eligible CUDA cells may be Optimized |
| **LFM2 chat; LFM2.5 Audio** | Bounded invocation-scoped paged/ring/composite state | ABI-v2 physical ownership; no Optimized attestation |
| **Whisper, Parakeet, VibeVoice ASR, Granite Speech, Voxtral ASR** | Bounded invocation-scoped physical state; Granite publishes both ASR routes | ABI-v2 physical ownership; no Optimized attestation |
| **Nemotron ASR** | Offline bounded invocation state and retained realtime tensor state | ABI-v2 physical ownership; no Optimized attestation |
| **VibeVoice, Fish S2, Voxtral TTS** | Bounded invocation-scoped physical state | ABI-v2 physical ownership; no Optimized attestation |
| **Sortformer diarization** | Bounded invocation-scoped physical state | ABI-v2 physical ownership; no Optimized attestation |
| **Stateless/catalog-only routes** | No retained mutable state, or no loaded state publication | No cache provider applies |

“ABI-v2 physical ownership” is not a claim of complete model-quality or
hardware performance certification. The exact model revision, capability,
backend, dtype, page geometry, attention semantics, and build feature cell must
pass its release lane.

### CUDA context policy

CUDA uses the native context reported by the loaded chat checkpoint. CPU and
Metal retain the configured sequence limit (4,096 tokens by default). The
current native CUDA ceilings are:

| Model | CUDA ceiling | Source |
|---|---:|---|
| Qwen3 chat | 40,960 tokens | [Official Qwen3 config](https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json) |
| Qwen3.5 chat | 262,144 tokens | [Official Qwen3.5 config](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json) |
| LFM2.5 chat | 128,000 tokens | [Official LFM2.5 config](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/main/config.json) |
| Gemma 3 1B / 4B | 32,768 / 131,072 tokens | [Google Gemma 3 model card](https://ai.google.dev/gemma/docs/core/model_card_3) |

Optional YaRN extensions are not included: the local adapters do not implement
their scaling parameters. CUDA reserves enough paged KV capacity for the
largest native context before publishing a model as ready. This can require
substantial VRAM and may make model loading fail on a smaller GPU rather than
silently reducing the context.

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
| Prefix reuse | Opt-in | Disabled by default; requires a non-empty isolation namespace and an independently bounded page budget. |
| Optimized-provider demotion | Supported | Set `IZWI_KV_DISABLE_OPTIMIZED_PROVIDER=1`; this can only demote to a certified Portable provider. |
| Tiered/offloaded or distributed KV | Not supported | No production ownership or eviction contract is published for these modes. |

For configuration, counters, benchmarks, and rollback, see
[KV Cache Operations](/kv-cache-operations).

---

## Verification Guidance

Use the following expectations when validating a host:

- **macOS Apple Silicon:** build or install a Metal-capable binary and run with `--backend metal` or `IZWI_BACKEND=metal`.
- **Linux/Windows GitHub Release:** run `izwi serve --backend cpu`, then `izwi status --detailed`.
- **Docker CUDA on NVIDIA Linux hosts:** run `docker compose --profile cuda up`, then confirm the container selects CUDA through `/v1/health` or `izwi status --detailed` from a matching client environment. Eligible Candle FlashAttention, Qwen3/Qwen3.5 RoPE, and Gemma RMSNorm CUDA routes activate automatically; their environment variables remain explicit rollback switches.
- **Linux/Windows source build for CUDA:** build with `cargo build --release --features cuda`, then run with `--backend cuda` or `IZWI_BACKEND=cuda`. The `cuda` wrapper includes Candle FlashAttention, while `cudnn` additionally enables matching Candle/cuDNN convolution paths.

---

## See Also

- [Installation](/installation)
- [Getting Started](/getting-started)
- [CLI Reference](/cli)
