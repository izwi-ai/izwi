# Runtime Support Matrix

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
| **Docker `production-cuda` target / `docker compose --profile cuda`** | Linux x86_64 + NVIDIA GPU | `cuda` | Preview | Shipped CUDA binary path. The final image is based on `nvidia/cuda:12.4.1-runtime-ubuntu22.04`. When building on a machine without `nvidia-smi`, set `CUDA_COMPUTE_CAP` for the target GPU architecture. |

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
[API Reference](./api.md).

| Surface | Status | Notes |
|---------|--------|-------|
| **`POST /v1/audio/speech`** | Stable | Core OpenAI-compatible TTS surface. |
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
| **Local media lifecycle (`/v1/media*`)** | Preview | OSS local media can be listed, uploaded from base64 payloads, downloaded by catch-all relative path, and deleted. Enterprise object storage can wrap the same route family. |
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

## Verification Guidance

Use the following expectations when validating a host:

- **macOS Apple Silicon:** build or install a Metal-capable binary and run with `--backend metal` or `IZWI_BACKEND=metal`.
- **Linux/Windows GitHub Release:** run `izwi serve --backend cpu`, then `izwi status --detailed`.
- **Docker CUDA on NVIDIA Linux hosts:** run `docker compose --profile cuda up`, then confirm the container selects CUDA through `/v1/health` or `izwi status --detailed` from a matching client environment.
- **Linux/Windows source build for CUDA:** build with `cargo build --release --features cuda`, then run with `--backend cuda` or `IZWI_BACKEND=cuda`.

---

## See Also

- [Installation](./installation/index.md)
- [Getting Started](./getting-started.md)
- [CLI Reference](./cli/index.md)
