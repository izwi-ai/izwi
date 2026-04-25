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
| **Desktop app from GitHub Releases** | Linux x86_64 | `cpu`, `cuda` preview | Stable CPU / Preview CUDA | Public `izwi` and `izwi-server` names stay CPU-safe; the installer includes a private CUDA runtime for NVIDIA hosts. |
| **Desktop app from GitHub Releases** | Windows x86_64 | `cpu`, `cuda` preview | Stable CPU / Preview CUDA | Public `izwi.exe` and `izwi-server.exe` names stay CPU-safe; the installer includes a private CUDA runtime for NVIDIA hosts. |
| **Terminal bundle from GitHub Releases** | Linux x86_64 | `cpu`, `cuda` preview | Stable CPU / Preview CUDA | Linux terminal tarballs use the same private CUDA runtime layout as desktop bundles. |
| **Terminal bundle from GitHub Releases** | macOS Apple Silicon | `metal` | Stable | Metal is compiled into the macOS build path. |
| **Terminal bundle from GitHub Releases** | Windows x86_64 | `cpu`, `cuda` preview | Stable CPU / Preview CUDA | Windows terminal zips use the same private CUDA runtime layout as desktop bundles. |
| **Source build** | macOS Apple Silicon with `--features metal` | `metal` | Stable | Recommended GPU path on macOS. |
| **Source build** | Linux x86_64 with `--features cuda` and CUDA toolkit installed | `cuda` | Supported | Useful for development, custom builds, or validating ahead of release packaging. Requires a compatible NVIDIA driver/toolkit environment. |
| **Source build** | Windows with `--features cuda` and CUDA toolkit installed | `cuda` | Preview | Useful for development and fallback validation while Windows CUDA release packaging remains preview. |
| **Docker `production` target** | Linux x86_64 | `cpu` | Stable | CPU-only container image. |
| **Docker `production-cuda` target / `docker compose --profile cuda`** | Linux x86_64 + NVIDIA GPU | `cuda` | Preview | Intended for NVIDIA hosts. When building on a machine without `nvidia-smi`, set `CUDA_COMPUTE_CAP` for the target GPU architecture. |

---

## Deployment Matrix

| Deployment target | Status | Notes |
|-------------------|--------|-------|
| **Single-user macOS desktop evaluation** | Stable | Best-supported path for local evaluation. |
| **Single-host Linux server on CPU** | Stable | Supported via GitHub Release packages, source builds, and the Docker CPU image. |
| **Single-host Linux server on NVIDIA GPU** | Supported / Preview by artifact | Source builds are supported. Release installers and terminal bundles include CUDA packaging as preview until GPU-host release smoke coverage is automated. Docker CUDA remains preview. |
| **Windows desktop evaluation** | Stable CPU / Preview CUDA | The installer remains CPU-safe and can use the packaged CUDA runtime on NVIDIA hosts when driver/device checks pass. |
| **Docker Compose on CPU** | Stable | Use the default `izwi` service. |
| **Docker Compose on NVIDIA GPU** | Preview | Use `docker compose --profile cuda up`; the profile runs the `izwi-cuda` service and may require `CUDA_COMPUTE_CAP` when built on a non-GPU machine. |
| **Kubernetes / Helm / multi-node production orchestration** | Not yet supported | Not published in OSS today. |

---

## API Surface Maturity

The runtime exposes both compatibility APIs and first-party local workflow APIs under `/v1`.
When the server is running, open `/docs` for the local Scalar API reference or
`/openapi.json` for the raw OpenAPI document. Preview endpoints are tagged as
preview in the API reference.

| Surface | Status | Notes |
|---------|--------|-------|
| **`POST /v1/audio/speech`** | Stable | Core OpenAI-compatible TTS surface. |
| **`POST /v1/audio/transcriptions`** | Stable | Core OpenAI-compatible transcription surface. |
| **`POST /v1/chat/completions`** | Stable | Core OpenAI-compatible chat surface. |
| **`GET /v1/models`** | Stable | Live model catalog / availability surface. |
| **Operational probes (`/livez`, `/readyz`, `/v1/live`, `/v1/ready`)** | Stable | Use `/livez` for cheap liveness and `/readyz` for readiness or deployment healthchecks. `/v1/health` remains the richer status payload. |
| **Local API reference (`/docs`, `/openapi.json`)** | Stable | Served by the same `izwi-server` process. Endpoint-level preview status is shown in the API reference. |
| **Local CLI workflows (`izwi serve`, `izwi pull`, `izwi tts`, `izwi transcribe`)** | Stable | Primary user-facing local runtime workflows. |
| **`POST /v1/responses` and response-object lifecycle routes** | Preview | Response objects are stored in bounded process memory for compatibility convenience. `store:false` skips retention; retained records can be evicted and are lost on server restart. |
| **`/v1/admin/*` model-management APIs** | Preview | Operator-oriented local admin APIs; auth and long-term contract are not finalized. |
| **Persisted first-party workflow APIs (`/v1/transcriptions/jobs`, `/v1/diarizations`, `/v1/text-to-speech-generations`, `/v1/studio/*`, `/v1/voices*`)** | Preview | Powerful local product APIs, but the public compatibility/support contract is still evolving. |
| **Local agent/session features** | Preview | Agent session metadata is process-local and bounded today. Linked chat threads, voice sessions, voice turns, and voice observations are the durable SQLite-backed local stores. |

---

## CUDA Caveats

- Linux and Windows GitHub Releases keep public binary names unchanged: `izwi` and `izwi-server` on Linux, `izwi.exe` and `izwi-server.exe` on Windows.
- The public release entrypoints are CPU-safe. CUDA-capable runtime binaries are private package resources under `runtime/cuda`.
- Release installers do not replace the host NVIDIA driver. CUDA acceleration requires a compatible NVIDIA driver and CUDA-capable GPU.
- CUDA release packaging is still preview until a real NVIDIA-host smoke test is automated in CI. Hosted release CI verifies layout, packaged runtime libraries, and CPU-safe startup.
- Source builds still require the CUDA toolkit and remain useful for development or fallback validation.
- The Docker CUDA image/profile is intended for NVIDIA Linux hosts and may require `CUDA_COMPUTE_CAP` when built on a machine without `nvidia-smi`.
- On macOS, the recommended GPU path is Metal, not CUDA.

---

## Verification Guidance

Use the following expectations when validating a host:

- **macOS Apple Silicon:** build or install a Metal-capable binary and run with `--backend metal` or `IZWI_BACKEND=metal`.
- **Linux/Windows GitHub Release on CPU-only hosts:** run `izwi serve --backend cpu`, then `izwi status --detailed`.
- **Linux/Windows GitHub Release on NVIDIA hosts:** install a compatible NVIDIA driver, run `izwi serve --backend cuda`, then confirm `Selected:  cuda` and the CUDA runtime diagnostics in `izwi status --detailed`.
- **Linux/Windows source build for CUDA:** build with `cargo build --release --features cuda`, then run with `--backend cuda` or `IZWI_BACKEND=cuda`.

---

## See Also

- [Installation](./installation/index.md)
- [Getting Started](./getting-started.md)
- [CLI Reference](./cli/index.md)
