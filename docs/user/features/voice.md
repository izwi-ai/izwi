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
izwi pull qwen3-tts-0.6b-base

# Speech recognition
izwi pull qwen3-asr-0.6b

# Chat (optional, for smarter responses)
izwi pull qwen3-0.6b
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
3. Try a larger ASR model (`qwen3-asr-1.7b`)

---

## See Also

- [Chat](./chat.md) — Text-based conversations
- [Transcription](./transcription.md) — Batch audio transcription
- [Text-to-Speech](./text-to-speech.md) — Generate speech from text
