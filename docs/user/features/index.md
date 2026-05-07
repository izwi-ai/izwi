# Features

Izwi provides a comprehensive suite of audio AI capabilities. Each feature is accessible via the web UI, desktop app, and command line.

---

## Core Features

| Feature | Description | Guide |
|---------|-------------|-------|
| **Voice** | Real-time voice conversations with AI | [Voice Guide](./voice.md) |
| **Chat** | Text-based AI conversations | [Chat Guide](./chat.md) |
| **Text-to-Speech** | Generate natural speech from text | [TTS Guide](./text-to-speech.md) |
| **Studio** | Manage long-form TTS projects and exports | [Studio Guide](./studio.md) |
| **Transcription** | Convert audio to text | [Transcription Guide](./transcription.md) |
| **Diarization** | Identify multiple speakers | [Diarization Guide](./diarization.md) |
| **Voice Cloning** | Clone voices from audio samples | [Voice Cloning Guide](./voice-cloning.md) |
| **Voice Design** | Create voices from descriptions | [Voice Design Guide](./voice-design.md) |

---

## Feature Comparison

| Feature | Web UI | Desktop | CLI | API |
|---------|--------|---------|-----|-----|
| Voice | ✓ | ✓ | — | ✓ |
| Chat | ✓ | ✓ | ✓ | ✓ |
| Text-to-Speech | ✓ | ✓ | ✓ | ✓ |
| Studio | ✓ | ✓ | — | ✓ |
| Transcription | ✓ | ✓ | ✓ | ✓ |
| Diarization | ✓ | ✓ | — | ✓ |
| Voice Cloning | ✓ | ✓ | ✓ | ✓ |
| Voice Design | ✓ | ✓ | ✓ | ✓ |

---

## Getting Started

1. **Start the server:**
   ```bash
   izwi serve
   ```

2. **Open the web UI:**
   ```
   http://localhost:8080
   ```

3. **Download required models:**
   ```bash
   izwi pull Qwen3-TTS-12Hz-0.6B-Base
   izwi pull Qwen3-ASR-0.6B-GGUF
   izwi pull Qwen3-8B-GGUF
   ```

---

## Model Requirements

Different features require different models:

| Feature | Required Models |
|---------|-----------------|
| Voice | TTS + ASR + Chat model (or unified `LFM2.5-Audio-1.5B-GGUF`) |
| Chat | Chat model (Qwen3, Qwen3.5, LFM2.5, or Gemma) |
| Text-to-Speech | TTS model |
| Studio | TTS model |
| Transcription | ASR model (`Parakeet-TDT-0.6B-v3` default; Qwen3/Whisper/LFM2.5 also supported) |
| Diarization | `diar_streaming_sortformer_4spk-v2.1` (+ optional ASR and aligner models) |
| Forced Alignment | `Qwen3-ForcedAligner-0.6B` (or `-4bit`) |
| Voice Cloning | Qwen3 TTS Base model (`Qwen3-TTS-12Hz-*-Base*`) |
| Voice Design | Qwen3 TTS VoiceDesign model (`Qwen3-TTS-12Hz-1.7B-VoiceDesign*`) |

---

## Next Steps

Choose a feature to learn more:

- [Voice Mode](./voice.md) — Real-time conversations
- [Text-to-Speech](./text-to-speech.md) — Generate speech
- [Studio](./studio.md) — Build long-form TTS projects
- [Transcription](./transcription.md) — Convert audio to text
- [API Reference](../api.md) — Integrate with HTTP, SSE, and WebSocket APIs
