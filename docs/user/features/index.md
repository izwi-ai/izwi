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
| **Speaker Attributed ASR** | Generate speaker-turn transcripts | [SAA Guide](/features/speaker-attributed-asr) |
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
| Diarization | ✓ | ✓ | ✓ | ✓ |
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
   izwi pull Kokoro-82M
   izwi pull Qwen3-ASR-0.6B-GGUF
   izwi pull Qwen3.5-4B
   ```

---

## Model Requirements

Different features require different models:

| Feature | Required Models |
|---------|-----------------|
| Voice | TTS + ASR + Chat model (or one unified model: `LFM3-Audio-2B-GGUF` by default, or `LFM2.5-Audio-1.5B-GGUF`) |
| Chat | Chat model (Qwen3.5, Qwen3.6, Qwen3.8, LFM2.5, or LFM3) |
| Speaker Attributed ASR | `Granite-Speech-4.1-2B-Plus` (default) or `Nemotron-3.5-Transcribe-30B-A4B-FP8` |
| Voices | Built-in voice model for presets; Base, VibeVoice, `FishAudio-S2-Pro`, Chatterbox or Maya 2 Native model for cloning; VoiceDesign model for design |
| Text-to-Speech | TTS model |
| Studio | TTS model |
| Settings | No model required |
| Transcription | ASR model (`Parakeet-TDT-0.6B-v3` default; Qwen3/Whisper/Granite Speech/Nemotron 3.5 Transcribe/LFM2.5/LFM3 Audio also supported) |
| Diarization | `diar_streaming_sortformer_4spk-v2.1` or `Nemotron-3-Diarization` (+ optional ASR and aligner models), or `Nemotron-3.5-Transcribe-30B-A4B-FP8` alone |
| Forced Alignment | `Qwen3-ForcedAligner-0.6B` |
| Voice Cloning | `Qwen3-TTS-12Hz-0.6B-Base`, `Qwen3-TTS-12Hz-1.7B-Base`, `VibeVoice-1.5B`, `FishAudio-S2-Pro`, `Chatterbox-500M`, `Chatterbox-Multilingual-500M` or `Maya2-Native-2.5B` |
| Voice Design | Qwen3 TTS VoiceDesign model (`Qwen3-TTS-12Hz-1.7B-VoiceDesign*`) or `Maya1-3B` |

---

## Next Steps

Choose a feature to learn more:

- [Voice Mode](/features/voice) — Real-time conversations
- [Voices](/features/voices) — Manage and create reusable voices
- [Text-to-Speech](/features/text-to-speech) — Generate speech
- [Studio](/features/studio) — Build long-form TTS projects
- [Settings](/features/settings) — Configure app preferences
- [Transcription](/features/transcription) — Convert audio to text
- [Speaker Attributed ASR](/features/speaker-attributed-asr) — Speaker-turn transcripts
- [API Reference](/api) — Integrate with HTTP, SSE, and WebSocket APIs
