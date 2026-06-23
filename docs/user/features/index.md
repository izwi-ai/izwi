---
title: "Features"
description: "Explore Izwi capabilities across voice mode, chat, transcription, voices, TTS, Studio, diarization, and voice creation."
icon: "sparkles"
---
## Core Features

| Feature | Description | Guide |
|---------|-------------|-------|
| **Voice** | Real-time voice conversations with AI | [Voice Guide](/features/voice) |
| **Chat** | Text-based AI conversations | [Chat Guide](/features/chat) |
| **Transcription** | Convert audio to text | [Transcription Guide](/features/transcription) |
| **Speaker Attributed ASR** | Generate Granite speaker-turn transcripts | [SAA Guide](/features/speaker-attributed-asr) |
| **Voices** | Create, clone, design, preview, manage, and reuse voice assets | [Voice Studio Guide](/features/voices) |
| **Text-to-Speech** | Generate natural speech from text | [TTS Guide](/features/text-to-speech) |
| **Studio** | Manage long-form TTS projects and exports | [Studio Guide](/features/studio) |
| **Settings** | Configure appearance, updates, privacy, onboarding, and desktop behavior | [Settings Guide](/features/settings) |
| **Diarization** | Identify multiple speakers | [Diarization Guide](/features/diarization) |
| **Voice Cloning** | Clone voices from audio samples inside Voice Studio | [Voice Cloning Guide](/features/voice-cloning) |
| **Voice Design** | Create voices from descriptions inside Voice Studio | [Voice Design Guide](/features/voice-design) |

---

## Feature Comparison

| Feature | Web UI | Desktop | CLI | API |
|---------|--------|---------|-----|-----|
| Voice | ✓ | ✓ | — | ✓ |
| Chat | ✓ | ✓ | ✓ | ✓ |
| Transcription | ✓ | ✓ | ✓ | ✓ |
| Speaker Attributed ASR | ✓ | ✓ | — | ✓ |
| Voices | ✓ | ✓ | ✓ | ✓ |
| Text-to-Speech | ✓ | ✓ | ✓ | ✓ |
| Studio | ✓ | ✓ | — | ✓ |
| Settings | ✓ | ✓ | — | ✓ |
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
| Speaker Attributed ASR | `Granite-Speech-4.1-2B-Plus` |
| Voices | Built-in voice model for presets; Base or VibeVoice model for cloning; VoiceDesign model for design |
| Text-to-Speech | TTS model |
| Studio | TTS model |
| Settings | No model required |
| Transcription | ASR model (`Parakeet-TDT-0.6B-v3` default; Qwen3/Whisper/Granite Speech/LFM2.5 also supported) |
| Diarization | `diar_streaming_sortformer_4spk-v2.1` (+ optional ASR and aligner models) |
| Forced Alignment | `Qwen3-ForcedAligner-0.6B` (or `-4bit`) |
| Voice Cloning | Qwen3 TTS Base model (`Qwen3-TTS-12Hz-*-Base*`) |
| Voice Design | Qwen3 TTS VoiceDesign model (`Qwen3-TTS-12Hz-1.7B-VoiceDesign*`) |

---

## Next Steps

Choose a feature to learn more:

- [Voice Mode](/features/voice) — Real-time conversations
- [Voices](/features/voices) — Manage and create reusable voices
- [Text-to-Speech](/features/text-to-speech) — Generate speech
- [Studio](/features/studio) — Build long-form TTS projects
- [Settings](/features/settings) — Configure app preferences
- [Transcription](/features/transcription) — Convert audio to text
- [Speaker Attributed ASR](/features/speaker-attributed-asr) — Granite speaker-turn transcripts
- [API Reference](/api) — Integrate with HTTP, SSE, and WebSocket APIs
