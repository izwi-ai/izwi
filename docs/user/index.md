---
title: "Izwi Documentation"
description: "User documentation for Izwi, a local-first audio inference engine for speech, voice, and chat workflows."
sidebarTitle: "Overview"
icon: "house"
---
# Izwi Documentation

Welcome to the official documentation for **Izwi** — a local-first audio inference engine for text-to-speech, speech recognition, and voice AI workflows.

![Izwi icon](/images/app-icon.png)

---

## What is Izwi?

Izwi is a powerful, privacy-focused audio AI platform that runs entirely on your machine. No cloud services, no API keys, no data leaving your device.

**Key capabilities:**

- **Voice Mode** — Real-time voice conversations with AI
- **Text-to-Speech** — Generate natural speech from text
- **Voice Cloning** — Clone any voice from a short audio sample
- **Voice Design** — Create custom voices from text descriptions
- **Transcription** — Convert audio to text with high accuracy
- **Diarization** — Identify and separate multiple speakers
- **Chat** — Text-based AI conversations

---

## Quick Links

| Section | Description |
|---------|-------------|
| [Getting Started](/getting-started) | Install Izwi and run your first command |
| [Installation](/installation) | Platform-specific installation guides |
| [Runtime Support Matrix](/support-matrix) | Supported OS, hardware, artifact, and API surfaces |
| [API Reference](/api) | Stable, preview, first-party, operator, and realtime HTTP/WebSocket APIs |
| [Features](/features) | Learn about each feature in detail |
| [Models](/models) | Understand and manage AI models |
| [CLI Reference](/cli) | Complete command-line reference |
| [Troubleshooting](/troubleshooting) | Common issues and solutions |

---

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **macOS** | 12.0+ (Monterey) | 14.0+ (Sonoma) |
| **Linux** | Ubuntu 20.04+ | Ubuntu 22.04+ |
| **Windows** | Windows 10 | Windows 11 |
| **RAM** | 8 GB | 16 GB+ |
| **Storage** | 10 GB free | 50 GB+ free |
| **GPU** | — | Apple Silicon / NVIDIA GPU (see support matrix) |

> **Note:** Izwi is optimized for Apple Silicon Macs with Metal acceleration. NVIDIA CUDA support exists in the runtime, but artifact-level support varies by source build, Docker image, and release package. See the [Runtime Support Matrix](/support-matrix).

---

## Getting Help

- **GitHub Issues** — [Report bugs or request features](https://github.com/izwi-ai/izwi/issues)
- **Discussions** — [Ask questions and share ideas](https://github.com/izwi-ai/izwi/discussions)

---

## License

Izwi is open source software licensed under [Apache 2.0](https://github.com/izwi-ai/izwi/blob/main/LICENSE).
