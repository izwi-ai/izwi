---
title: "Text-to-Speech"
description: "Generate natural speech from text with Izwi models, voices, streaming, and audio output formats."
sidebarTitle: "Text-to-Speech"
icon: "volume-2"
---
Generate natural, human-like speech from text using state-of-the-art TTS models.

---

## Overview

Izwi's text-to-speech converts written text into spoken audio. Features include:

- **Natural voices** — High-quality, expressive speech
- **Local audio output** — WAV for files and raw PCM for low-level API clients
- **Speed control** — Adjust playback speed
- **Streaming** — Real-time audio generation
- **Local processing** — No cloud, complete privacy

---

## Getting Started

### Download a TTS Model

```bash
izwi pull Kokoro-82M
```

### Kokoro-82M Prerequisite (`espeak-ng`)

`Kokoro-82M` is the default TTS model. Install `espeak-ng` on your system first.
Izwi uses it for Kokoro phonemization and will return an error if it is missing.

- macOS: see [macOS Installation](/installation/macos#optional-install-espeak-ng-for-kokoro-82m)
- Linux: see [Linux Installation](/installation/linux#optional-install-espeak-ng-for-kokoro-82m)
- Windows: see [Windows Installation](/installation/windows#optional-install-espeak-ng-for-kokoro-82m)

### Generate Speech

**Command line:**

```bash
izwi tts "Hello, welcome to Izwi!" --output hello.wav
```

**Play after generation:**

```bash
izwi tts "Hello, welcome to Izwi!" --play
```

Izwi saves the file, then plays it with the system player.

---

## Using the CLI

### Basic Usage

```bash
izwi tts "Your text here" --output output.wav
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model`, `-m` | TTS model to use | `Kokoro-82M` |
| `--output`, `-o` | Output file path | `izwi_output_<timestamp>.wav` in the current directory |
| `--format`, `-f` | Audio format: `wav` or `pcm` (raw 16-bit little-endian mono, no header) | `wav` |
| `--speed`, `-r` | Speech speed multiplier (models with speed control) | Model default |
| `--speaker`, `-s` | Voice/speaker ID | Model default (Kokoro: `af_heart`) |
| `--saved-voice-id` | Reuse a saved reference voice | — |
| `--reference-audio` | Reference audio file for cloning | — |
| `--reference-text` | Transcript for the reference audio (needs `--reference-audio`; optional for Chatterbox) | — |
| `--reference-text-file` | File containing the reference transcript | — |
| `--instructions` | Voice-design direction prompt | — |
| `--temperature`, `-t` | Sampling temperature | Model default |
| `--language` | Language of the text (`Chatterbox-Multilingual-500M`: a code such as `fr` or a name; a hint for Qwen3-TTS) | Model default |
| `--exaggeration` | Emotion intensity, 0 to 2 (Chatterbox only) | `0.5` |
| `--cfg-weight` | Classifier-free guidance weight, 0 to 1; `0` turns guidance off (Chatterbox only) | `0.5` |
| `--min-p` | Min-p sampling threshold, 0 to 1 (Chatterbox only) | `0.05` |
| `--play`, `-p` | Play the saved file with the system player (WAV only) | — |
| `--stream` | Stream the audio as it is generated into a WAV file and report time to first audio | — |

### Examples

**WAV output:**

```bash
izwi tts "Hello world" --format wav --output hello.wav
```

**Adjust speed:**

```bash
# Slower (0.5x - 1.0x)
izwi tts "Speaking slowly" --speed 0.75 --output slow.wav

# Faster (1.0x - 2.0x)
izwi tts "Speaking quickly" --speed 1.5 --output fast.wav
```

**Read from stdin:**

```bash
echo "Text from pipe" | izwi tts - --output piped.wav
cat article.txt | izwi tts - --output article.wav
```

**Streaming output:**

```bash
izwi tts "Long text for streaming" --stream --play
```

**Clone from reference audio:**

```bash
izwi tts "This line uses the reference voice." \
  --model Qwen3-TTS-12Hz-0.6B-Base \
  --reference-audio samples/reference.wav \
  --reference-text "This is the text spoken in the reference sample." \
  --output cloned.wav
```

**Clone with Chatterbox (no transcript), more expressive:**

```bash
izwi tts "This line uses the reference voice with more emotion." \
  --model Chatterbox-500M \
  --reference-audio samples/reference.wav \
  --exaggeration 0.7 \
  --cfg-weight 0.3 \
  --output expressive.wav
```

For expressive or fast-talking reference voices, Resemble AI suggests a lower
`--cfg-weight` (about 0.3) with `--exaggeration` around 0.7 or higher.

**Reuse a saved voice:**

```bash
izwi tts "This line uses a saved voice." \
  --model Qwen3-TTS-12Hz-0.6B-Base \
  --saved-voice-id voice_abc123 \
  --output saved-voice.wav
```

**Design a prompted voice:**

```bash
izwi tts "This line uses a designed voice." \
  --model Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --instructions "A friendly audiobook narrator with warm pacing" \
  --output designed.wav
```

---

## Using the Web UI

1. Navigate to **Text to Speech** in the sidebar
2. Enter your text in the input field
3. Select a voice (if available)
4. Click **Generate**
5. Play or download the audio

With Chatterbox selected, the dialog shows sliders for exaggeration, CFG weight
and min-p. With Chatterbox Multilingual it also asks which language the text is
in.

### Features

- **Live preview** — Hear audio as it generates
- **Download** — Save audio files locally
- **History** — Access recent generations

---

## Using the API

### Endpoint

```
POST /v1/audio/speech
```

### Request

```json
{
  "model": "Kokoro-82M",
  "input": "Hello, this is a test.",
  "voice": "af_heart",
  "speed": 1.0,
  "response_format": "wav"
}
```

### Response

Binary audio data with appropriate `Content-Type` header.

Set `stream` to `true` or `stream_format` to `sse` to receive server-sent
audio events instead of one binary response. See the
[API Reference](/api#audio-speech) for streaming event shapes,
voice-cloning fields, saved voices, and model-specific controls.

### Example (curl)

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "Kokoro-82M", "input": "Hello world", "voice": "af_heart"}' \
  --output speech.wav
```

---

## Available Models

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| `Kokoro-82M` | ~0.4 GB | Good | Fast |
| `Qwen3-TTS-12Hz-0.6B-Base` | ~2.3 GB | Good | Fast |
| `Qwen3-TTS-12Hz-1.7B-Base` | ~4.2 GB | Better | Medium |
| `Qwen3-TTS-12Hz-0.6B-CustomVoice` | ~2.3 GB | Good | Fast |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | ~4.2 GB | Better | Medium |
| `VibeVoice-1.5B` | ~5.0 GB | Better long-form/reference voice | Medium |
| `Voxtral-4B-TTS-2603` | ~8.1 GB | Better | Medium |
| `Chatterbox-500M` | ~3.2 GB | English reference voice, emotion control | Slow |
| `Chatterbox-Multilingual-500M` | ~3.3 GB | 22 languages, reference voice, emotion control | Slow |

For reference-audio cloning, use **Base** variants, `VibeVoice-1.5B`, `FishAudio-S2-Pro`, `Chatterbox-500M` or `Chatterbox-Multilingual-500M`.  
For built-in voice presets, use `Kokoro-82M`, **CustomVoice** variants, `Voxtral-4B-TTS-2603` or `LFM2.5-Audio-1.5B-GGUF`.  
For prompt-based voice design, use **VoiceDesign** variants.
`Kokoro-82M` requires `espeak-ng` to be installed separately.
`Voxtral-4B-TTS-2603` supports 20 preset voices and emits 24 kHz audio. Its
model and bundled voice assets inherit a CC BY-NC 4.0 license.
`Chatterbox-500M` (English, 24 kHz, MIT) speaks with its built-in `default`
voice or clones from reference audio alone. It streams (first audio in about
2 s on an Apple M1 Pro with Metal), has no speed control or voice design, and
runs slower than real time (a real-time factor of about 2.2 for one request on
that Mac, where its T3 model runs in F16 and the model takes about 2.6 GB;
set `IZWI_METAL_DTYPE=f32` to keep it in F32), so streamed audio can pause
between windows. Long text is split between
sentences and spoken in full. Every utterance carries Resemble AI's imperceptible PerTh watermark;
operators can turn it off with `IZWI_CHATTERBOX_WATERMARK=false`.
`Chatterbox-Multilingual-500M` does the same in Arabic, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Italian, Korean, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swahili, Swedish and Turkish, named per
request with `language` (English when unset). Japanese is not supported yet.
For built-in speaker IDs, see [Voice Presets](/models/voice-presets).

---

## Audio Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| WAV | `.wav` | Uncompressed, highest quality |
| PCM | `.pcm` | Raw PCM for low-level playback pipelines |
| MP3, OPUS, OGG, FLAC, AAC | Matching extension | Recognized request names. The OSS server does not bundle compressed encoders yet, so API clients must set `allow_format_fallback: true` if they intentionally want WAV bytes returned for these names. |

---

## Tips

1. **Punctuation matters** — Use proper punctuation for natural pauses
2. **Break long text** — Split very long text into paragraphs
3. **Test different speeds** — Find the right pace for your use case
4. **Use appropriate models** — Larger models = better quality but slower

---

## See Also

- [Voice Cloning](/features/voice-cloning) — Clone custom voices
- [Voice Design](/features/voice-design) — Create voices from descriptions
- [Voices](/features/voices) — Manage saved and built-in voices
- [CLI Reference](/cli) — Full command documentation
