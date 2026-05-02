<p align="center">
  <img src="images/app-icon.png" alt="Izwi icon" width="140" />
</p>

<h1 align="center">Izwi</h1>

<p align="center"><strong>Local-first audio inference engine for TTS, ASR, and voice AI workflows.</strong></p>

<p align="center">
  <a href="https://izwiai.com">Website</a> •
  <a href="https://izwiai.com/docs">Documentation</a> •
  <a href="https://github.com/izwi-ai/izwi/releases">Releases</a> •
  <a href="https://izwiai.com/docs/getting-started">Getting Started</a>
</p>

<p align="center">
  <img src="images/screenshot.png" alt="Izwi Screenshot" width="800" />
</p>

---

## Overview

Izwi is a privacy-focused audio AI platform that runs entirely on your machine. No cloud services, no API keys, no data leaving your device.

**Core capabilities:**

- **Voice Mode** — Real-time voice conversations with AI
- **Text-to-Speech** — Generate natural speech from text
- **Studio** — Build long-form TTS projects and exports
- **Speech Recognition** — Convert audio to text with high accuracy
- **Speaker Diarization** — Identify and separate multiple speakers
- **Voice Cloning** — Clone any voice from a short audio sample
- **Voice Design** — Create custom voices from text descriptions
- **Forced Alignment** — Word-level audio-text alignment
- **Chat** — Text-based AI conversations

The server exposes OpenAI-compatible API routes under `/v1`. When the server is
running, the local Scalar API reference is available at
`http://localhost:8080/docs`, and the raw OpenAPI document is available at
`http://localhost:8080/openapi.json`.

## Runtime Support Matrix

Backend support depends on both the host and the artifact you install.

- macOS on Apple Silicon: Metal is the recommended and stable GPU path.
- Linux and Windows GitHub Release artifacts: public commands remain `izwi` / `izwi-server` and their Windows `.exe` counterparts, and are intentionally CPU-only.
- Source builds: CUDA is supported when you build with `--features cuda` on a compatible NVIDIA host.
- Docker CUDA profile: the CUDA distribution path for NVIDIA Linux hosts; when building on a machine without `nvidia-smi`, set `CUDA_COMPUTE_CAP` for the target GPU architecture.

See the full [Runtime Support Matrix](https://izwiai.com/docs/support-matrix).

---

## Quick Install

### macOS

Download the latest `.dmg` from [GitHub Releases](https://github.com/izwi-ai/izwi/releases):

1. Open the `.dmg` file
2. Drag **Izwi.app** to Applications
3. Launch Izwi

### Linux

```bash
wget https://github.com/izwi-ai/izwi/releases/latest/download/izwi_amd64.deb
sudo dpkg -i izwi_amd64.deb
```

### Windows

Download and run the installer from [GitHub Releases](https://github.com/izwi-ai/izwi/releases).

> **Full installation guides:** [macOS](https://izwiai.com/docs/installation/macos) • [Linux](https://izwiai.com/docs/installation/linux) • [Windows](https://izwiai.com/docs/installation/windows) • [From Source](https://izwiai.com/docs/installation/from-source)

---

## Quick Start

### 1. Start the server

```bash
izwi serve
```

Open `http://localhost:8080` in your browser.

API users can also open `http://localhost:8080/docs` for the local Scalar API
reference.

### 2. Download a model

```bash
izwi pull Qwen3-TTS-12Hz-0.6B-Base
```

### 3. Generate speech

```bash
izwi tts "Hello from Izwi!" --output hello.wav
```

### 4. Transcribe audio

```bash
izwi pull Parakeet-TDT-0.6B-v3
izwi transcribe audio.wav
```

Long-form ASR is handled automatically: Izwi now chunks long recordings,
stitches overlapping transcripts, and returns a full transcript instead of
only the first model window.

Optional tuning knobs:

```bash
IZWI_ASR_CHUNK_TARGET_SECS=24
IZWI_ASR_CHUNK_MAX_SECS=30
IZWI_ASR_CHUNK_OVERLAP_SECS=3
# Optional: preload models at server startup to reduce first-request cold latency.
# Comma-separated model IDs (for example Whisper-Large-v3-Turbo,Qwen3.5-4B)
IZWI_PRELOAD_MODELS=Whisper-Large-v3-Turbo
# Optional: run a short synthetic ASR warmup after preloading (enabled by default).
IZWI_WARMUP_PRELOADED_MODELS=1
IZWI_ASR_WARMUP_DURATION_MS=800
# Optional: tune text streaming queue depth when using per-character ASR streaming.
IZWI_STREAM_TEXT_QUEUE_CAPACITY=4096
```

---

## Anonymous Analytics (Desktop)

Izwi desktop supports optional, opt-in anonymous usage analytics powered by Aptabase.

- Disabled by default until users explicitly opt in.
- Can be enabled during onboarding or later in **Settings**.
- Users can opt out at any time.
- No prompts, transcripts, audio payloads, local paths, or personal identifiers are sent.

To enable analytics transport in the desktop shell, set the app key in the runtime environment:

```bash
APTABASE_APP_KEY=A-US-XXXXXXXXXXXXXXX
```

Use the exact key from Aptabase (for example `A-US-...` or `A-EU-...`).

Without this variable, analytics calls are treated as no-op events.

---

## Supported Models

| Category | Models |
|----------|--------|
| **TTS** | Qwen3-TTS 12Hz (0.6B Base/CustomVoice, 1.7B Base/CustomVoice/VoiceDesign), Kokoro-82M |
| **ASR** | Qwen3-ASR GGUF (0.6B, 1.7B), Parakeet-TDT-0.6B-v3, Whisper-Large-v3-Turbo |
| **Diarization** | Sortformer 4-speaker |
| **Chat** | Qwen3 GGUF (0.6B, 1.7B, 4B, 8B), Qwen3.5 GGUF (0.8B, 2B, 4B, 9B), LFM2.5 (1.2B Instruct/Thinking GGUF), Gemma 3 (1B) |
| **Audio** | LFM2.5-Audio-1.5B-GGUF |
| **Alignment** | Qwen3-ForcedAligner-0.6B (full, 4-bit) |

Run `izwi list` to see all available models.

> **Full model documentation:** [Models Guide](https://izwiai.com/docs/models)

---

## Documentation

| Resource | Link |
|----------|------|
| **Getting Started** | [izwiai.com/docs/getting-started](https://izwiai.com/docs/getting-started) |
| **Installation** | [izwiai.com/docs/installation](https://izwiai.com/docs/installation) |
| **Features** | [izwiai.com/docs/features](https://izwiai.com/docs/features) |
| **CLI Reference** | [izwiai.com/docs/cli](https://izwiai.com/docs/cli) |
| **Models** | [izwiai.com/docs/models](https://izwiai.com/docs/models) |
| **Local API Reference** | `http://localhost:8080/docs` when `izwi serve` is running |
| **Troubleshooting** | [izwiai.com/docs/troubleshooting](https://izwiai.com/docs/troubleshooting) |

---

## License

Apache 2.0

## Acknowledgments

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba
- [Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) by NVIDIA
- [Gemma](https://ai.google.dev/gemma) by Google
- [HuggingFace Hub](https://huggingface.co/) for model hosting
