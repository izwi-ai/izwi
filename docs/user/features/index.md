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
   izwi pull qwen3-tts-0.6b-base
   izwi pull qwen3-asr-0.6b
   ```

---

## Model Requirements

Different features require different models:

| Feature | Required Models |
|---------|-----------------|
| Voice | TTS + ASR + Chat model |
| Chat | Chat model (Qwen3 or Gemma) |
| Text-to-Speech | TTS model |
| Studio | TTS model |
| Transcription | ASR model (Qwen3 or Parakeet) |
| Diarization | Diarization model (Sortformer) |
| Forced Alignment | Forced aligner model |
| Voice Cloning | TTS CustomVoice model |
| Voice Design | TTS VoiceDesign model |

---

## Next Steps

Choose a feature to learn more:

- [Voice Mode](./voice.md) — Real-time conversations
- [Text-to-Speech](./text-to-speech.md) — Generate speech
- [Studio](./studio.md) — Build long-form TTS projects
- [Transcription](./transcription.md) — Convert audio to text
