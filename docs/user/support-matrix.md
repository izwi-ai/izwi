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
| **Desktop app from GitHub Releases** | Linux x86_64 | `cpu` | Stable | Current Linux release artifacts do not ship CUDA-enabled binaries. |
| **Desktop app from GitHub Releases** | Windows x86_64 | `cpu` | Stable | Current Windows release artifacts do not ship CUDA-enabled binaries. |
| **Terminal bundle from GitHub Releases** | Linux x86_64 | `cpu` | Stable | Current release workflow builds CPU binaries only. |
| **Terminal bundle from GitHub Releases** | macOS Apple Silicon | `metal` | Stable | Metal is compiled into the macOS build path. |
| **Terminal bundle from GitHub Releases** | Windows x86_64 | `cpu` | Stable | Current release workflow builds CPU binaries only. |
| **Source build** | macOS Apple Silicon with `--features metal` | `metal` | Stable | Recommended GPU path on macOS. |
| **Source build** | Linux x86_64 with `--features cuda` and CUDA toolkit installed | `cuda` | Supported | This is the primary NVIDIA path today. Requires a compatible NVIDIA driver/toolkit environment. |
| **Source build** | Windows with `--features cuda` and CUDA toolkit installed | `cuda` | Preview | Source-level CUDA hooks exist, but current release and verification coverage are Linux-first. |
| **Docker `production` target** | Linux x86_64 | `cpu` | Stable | CPU-only container image. |
| **Docker `production-cuda` target / `docker compose --profile cuda`** | Linux x86_64 + NVIDIA GPU | `cuda` | Preview | Intended for NVIDIA hosts. When building on a machine without `nvidia-smi`, set `CUDA_COMPUTE_CAP` for the target GPU architecture. |

---

## Deployment Matrix

| Deployment target | Status | Notes |
|-------------------|--------|-------|
| **Single-user macOS desktop evaluation** | Stable | Best-supported path for local evaluation. |
| **Single-host Linux server on CPU** | Stable | Supported via source build and Docker CPU image. |
| **Single-host Linux server on NVIDIA GPU** | Supported | Supported via Linux source builds today. Docker CUDA is available as a preview path and may require an explicit `CUDA_COMPUTE_CAP` when built on a non-GPU machine. |
| **Windows desktop evaluation** | Stable | CPU-focused today. |
| **Docker Compose on CPU** | Stable | Use the default `izwi` service. |
| **Docker Compose on NVIDIA GPU** | Preview | Intended path is `izwi-cuda`, but see current CUDA packaging caveat above. |
| **Kubernetes / Helm / multi-node production orchestration** | Not yet supported | Not published in OSS today. |

---

## API Surface Maturity

The runtime exposes both compatibility APIs and first-party local workflow APIs under `/v1`.

| Surface | Status | Notes |
|---------|--------|-------|
| **`POST /v1/audio/speech`** | Stable | Core OpenAI-compatible TTS surface. |
| **`POST /v1/audio/transcriptions`** | Stable | Core OpenAI-compatible transcription surface. |
| **`POST /v1/chat/completions`** | Stable | Core OpenAI-compatible chat surface. |
| **`GET /v1/models`** | Stable | Live model catalog / availability surface. |
| **Local CLI workflows (`izwi serve`, `izwi pull`, `izwi tts`, `izwi transcribe`)** | Stable | Primary user-facing local runtime workflows. |
| **`POST /v1/responses`** | Preview | Available today, but persistence/runtime contract is still being clarified. |
| **`/v1/admin/*` model-management APIs** | Preview | Operator-oriented local admin APIs; auth and long-term contract are not finalized. |
| **Persisted first-party workflow APIs (`/v1/transcriptions/jobs`, `/v1/diarizations`, `/v1/text-to-speech-generations`, `/v1/studio/*`, `/v1/voices*`)** | Preview | Powerful local product APIs, but the public compatibility/support contract is still evolving. |
| **Local agent/session features** | Preview | Available for local use, but not yet positioned as a stable deployment contract. |

---

## Current CUDA Caveats

- GitHub Releases do **not** currently ship CUDA-enabled Linux or Windows binaries.
- CUDA on NVIDIA should currently be treated as a **source-build-first** capability.
- The Docker CUDA image/profile is intended for NVIDIA Linux hosts and may require `CUDA_COMPUTE_CAP` when built on a machine without `nvidia-smi`.
- On macOS, the recommended GPU path is Metal, not CUDA.

---

## Verification Guidance

Use the following expectations when validating a host:

- **macOS Apple Silicon:** build or install a Metal-capable binary and run with `--backend metal` or `IZWI_BACKEND=metal`.
- **Linux NVIDIA source build:** build with `cargo build --release --features cuda`, then run with `--backend cuda` or `IZWI_BACKEND=cuda`.
- **GitHub Release binaries on Linux/Windows:** assume CPU unless a release asset explicitly says otherwise.

---

## See Also

- [Installation](./installation/index.md)
- [Getting Started](./getting-started.md)
- [CLI Reference](./cli/index.md)
