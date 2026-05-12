# Text-to-Speech

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
izwi pull Qwen3-TTS-12Hz-0.6B-Base
```

### Kokoro-82M Prerequisite (`espeak-ng`)

If you plan to use `Kokoro-82M`, install `espeak-ng` on your system first.
Izwi uses it for Kokoro phonemization and will return an error if it is missing.

- macOS: see [macOS Installation](../installation/macos.md#optional-install-espeak-ng-for-kokoro-82m)
- Linux: see [Linux Installation](../installation/linux.md#optional-install-espeak-ng-for-kokoro-82m)
- Windows: see [Windows Installation](../installation/windows.md#optional-install-espeak-ng-for-kokoro-82m)

### Generate Speech

**Command line:**

```bash
izwi tts "Hello, welcome to Izwi!" --output hello.wav
```

**With playback:**

```bash
izwi tts "Hello, welcome to Izwi!" --play
```

---

## Using the CLI

### Basic Usage

```bash
izwi tts "Your text here" --output output.wav
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model`, `-m` | TTS model to use | `qwen3-tts-0.6b-base` |
| `--output`, `-o` | Output file path | stdout |
| `--format`, `-f` | Audio format | `wav` |
| `--speed`, `-r` | Speech speed (0.5-2.0) | `1.0` |
| `--speaker`, `-s` | Voice/speaker ID | `default` |
| `--temperature`, `-t` | Sampling temperature | `0.7` |
| `--play`, `-p` | Play audio after generation | — |
| `--stream` | Stream output in real-time | — |

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

---

## Using the Web UI

1. Navigate to **Text to Speech** in the sidebar
2. Enter your text in the input field
3. Select a voice (if available)
4. Click **Generate**
5. Play or download the audio

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
  "model": "Qwen3-TTS-12Hz-0.6B-Base",
  "input": "Hello, this is a test.",
  "voice": "default",
  "speed": 1.0,
  "response_format": "wav"
}
```

### Response

Binary audio data with appropriate `Content-Type` header.

Set `stream` to `true` or `stream_format` to `sse` to receive server-sent
audio events instead of one binary response. See the
[API Reference](../api.md#audio-speech) for streaming event shapes,
voice-cloning fields, saved voices, and model-specific controls.

### Example (curl)

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-TTS-12Hz-0.6B-Base", "input": "Hello world"}' \
  --output speech.wav
```

---

## Available Models

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| `Kokoro-82M` | ~0.4 GB | Good | Fast |
| `Qwen3-TTS-12Hz-0.6B-Base` | ~2.3 GB | Good | Fast |
| `Qwen3-TTS-12Hz-1.7B-Base` | ~4.2 GB | Better | Medium |
| `Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit` | ~1.6 GB | Good | Fast |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit` | ~2.2 GB | Better | Medium |

For reference-audio cloning, use **Base** variants.  
For built-in voice presets, use **CustomVoice** variants.  
For prompt-based voice design, use **VoiceDesign** variants.
`Kokoro-82M` requires `espeak-ng` to be installed separately.

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

- [Voice Cloning](./voice-cloning.md) — Clone custom voices
- [Voice Design](./voice-design.md) — Create voices from descriptions
- [CLI Reference](../cli/index.md) — Full command documentation
