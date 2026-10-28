---
title: "Voice Mode"
description: "Use Izwi for real-time voice conversations with speech recognition, chat, and text-to-speech."
sidebarTitle: "Voice Mode"
icon: "audio-lines"
---
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

Open **Voice** in the sidebar. The first time, Izwi lists the models voice
needs and downloads them with one click. Models load when you start a
conversation, so there is no separate load step.

Voice has two engines, chosen in **Voice settings › Engine**:

| Engine | Models | Notes |
|--------|--------|-------|
| **Standard** (default) | `Parakeet-TDT-0.6B-v3` (speech recognition), `Qwen3.5-4B` (assistant), `Kokoro-82M` (voice) | Best answers and the full voice library |
| **Compact** | `LFM2.5-Audio-1.5B-GGUF` (default) or `LFM3-Audio-2B-GGUF` | One model listens and speaks; smaller download, fewer voices |

To download them from the command line instead:

```bash
# Standard
izwi pull Parakeet-TDT-0.6B-v3
izwi pull Qwen3.5-4B
izwi pull Kokoro-82M

# Compact
izwi pull LFM2.5-Audio-1.5B-GGUF
```

---

## Using Voice Mode

1. Navigate to **Voice** in the sidebar
2. Click **Start talking** (or press `Space`) and allow microphone access if asked
3. Speak naturally; Izwi answers when you pause
4. Start talking while Izwi speaks to interrupt it
5. Click **End** (or press `Space`) when you are done

### Controls

| Control | Action |
|---------|--------|
| **Microphone** | Mute or unmute your microphone |
| **End** | End the conversation |
| **Transcript** | Show what you and Izwi said |
| **Voice settings** (top right) | Voice, instructions, memory and engine |

---

## Voice Settings

- **Voice** — Choose the assistant's voice; press play to hear a preview
- **Instructions** — How the assistant should answer. Saved when you leave the field and used from your next conversation; **Reset to default** restores the original
- **Memory** — When on, Izwi remembers lasting preferences and facts you mention. **Manage memories** lists them so you can forget one or all
- **Engine** — Standard or Compact, the status of its models, and a button to load or unload them

Turn detection is automatic: the server ends your turn after a short silence.
Steady background noise does not start a turn, and a cough or noise burst
while Izwi talks lowers its volume instead of cutting it off.

### API Routes

Voice mode uses preview APIs for the voice profile, observations, saved
sessions, and low-latency WebSocket transport. See the
[API Reference](/api#voice-profile-memory-and-sessions) for persisted
voice state and [Voice Realtime](/api#voice-realtime) for the WebSocket
protocol.

`LFM3-Audio-2B-GGUF` also answers spoken turns, with interleaved text and
speech, through the same speech-to-speech routes: name it as `s2s_model_id` in
the WebSocket's `unified` mode. The **Compact** engine uses
`LFM2.5-Audio-1.5B-GGUF` by default: choose an LFM3 Audio model in
**Voice settings › Engine**, and with only LFM3 Audio downloaded the Compact
engine uses it.

With LFM3 Audio a spoken turn can call tools: the server's built-in tools (the
local date and time) by default, and function tools a WebSocket client declares
in `input_stream_start` and answers with `function_call_output`. The reply calls
a tool, hears its result, then speaks the answer. See
[Voice Realtime](/api#voice-realtime) for the fields and events.

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
| `Space` | Start or end the conversation |
| `Escape` | End the conversation |
| `M` | Mute or unmute your microphone |

> Shortcuts are ignored while focus is inside a text field or on a button.

---

## Troubleshooting

### No audio input detected

1. Check your microphone permissions in system settings
2. Ensure the correct input device is selected
3. Test your microphone in another application

### "Couldn't load the voice models"

The models do not fit in the memory Izwi can use alongside the other models
it has loaded. Unload models you are not using on the **Models** page, or
switch to the **Compact** engine in **Voice settings › Engine**.

### Responses are slow

1. The first reply after starting includes loading the models; later replies are faster
2. Try the **Compact** engine
3. Check system resources (RAM, CPU usage)

### Poor transcription accuracy

1. Speak more clearly and slowly
2. Reduce background noise
3. Use an external microphone

---

## See Also

- [Chat](/features/chat) — Text-based conversations
- [Transcription](/features/transcription) — Batch audio transcription
- [Text-to-Speech](/features/text-to-speech) — Generate speech from text
