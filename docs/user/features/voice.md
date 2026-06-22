---
title: "Voice Mode"
description: "Use Izwi for real-time voice conversations with speech recognition, chat, and text-to-speech."
sidebarTitle: "Voice Mode"
icon: "audio-lines"
---
# Voice Mode

Voice mode enables real-time spoken conversations with AI. Speak naturally and receive spoken responses — all processed locally on your device.

---

## Overview

Voice mode combines:
- **Speech recognition** — Converts your voice to text
- **AI chat** — Processes your message and generates a response
- **Text-to-speech** — Speaks the response back to you

Everything runs locally with no cloud services.

---

## Getting Started

### Required Models

Download the necessary models:

```bash
# Text-to-speech
izwi pull Qwen3-TTS-12Hz-0.6B-Base

# Speech recognition
izwi pull Qwen3-ASR-0.6B-GGUF

# Chat
izwi pull Qwen3-8B-GGUF

# Optional unified speech model
izwi pull LFM2.5-Audio-1.5B-GGUF
```

### Start Voice Mode

1. Start the server:
   ```bash
   izwi serve
   ```

2. Open the web UI:
   ```
   http://localhost:8080/voice
   ```

3. Click the microphone button to start speaking

---

## Using Voice Mode

### Web UI

1. Navigate to **Voice** in the sidebar
2. Click the **microphone button** to start recording
3. Speak your message
4. Click again to stop recording (or wait for auto-detection)
5. Listen to the AI response

### Controls

| Control | Action |
|---------|--------|
| **Microphone** | Start/stop recording |
| **Speaker** | Mute/unmute assistant responses |
| **Settings** | Configure models, playback speed, and speech detection |

---

## Configuration

### Select Voice

Choose from available voices in the settings panel. Different TTS models offer different voice options.

### Select Models

Configure which models to use:
- **ASR Model** — For speech recognition
- **TTS Model** — For speech synthesis
- **Chat Model** — For response generation

### Voice Agent Prompt

Use the **Voice Agent Prompt** section in settings to customize the assistant's speaking style and behavior. Prompt changes are saved locally and apply the next time you start a voice session.

### Observational Memory

Use **Observational Memory** in settings to review, enable, disable, or delete the stable user memories captured from modular voice conversations. You can forget individual memories or clear them all at any time.

### API Routes

Voice mode uses preview APIs for the voice profile, observations, saved
sessions, and low-latency WebSocket transport. See the
[API Reference](/api#voice-profile-memory-and-sessions) for persisted
voice state and [Voice Realtime](/api#voice-realtime) for the WebSocket
protocol.

### Audio Settings

- **Auto-detect silence** — Automatically stop recording when you stop speaking
- **Playback speed** — Adjust response playback speed from `0.75x` to `1.75x`
- **Mute output** — Silence assistant playback without ending the session

---

## Tips for Best Results

1. **Use a good microphone** — Built-in laptop mics work but external mics are better
2. **Minimize background noise** — Find a quiet environment
3. **Speak clearly** — Natural pace, clear pronunciation
4. **Wait for responses** — Let the AI finish before speaking again

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Space` | Start or stop the current voice session |
| `Escape` | Stop the current voice session |
| `M` | Mute/unmute |

> Shortcuts are ignored while focus is inside a text field or other editable control.

---

## Troubleshooting

### No audio input detected

1. Check your microphone permissions in system settings
2. Ensure the correct input device is selected
3. Test your microphone in another application

### Responses are slow

1. Use smaller models for faster responses
2. Ensure models are loaded (not loading on-demand)
3. Check system resources (RAM, CPU usage)

### Poor transcription accuracy

1. Speak more clearly and slowly
2. Reduce background noise
3. Try a larger ASR model (`Qwen3-ASR-1.7B-GGUF`)

---

## See Also

- [Chat](/features/chat) — Text-based conversations
- [Transcription](/features/transcription) — Batch audio transcription
- [Text-to-Speech](/features/text-to-speech) — Generate speech from text
