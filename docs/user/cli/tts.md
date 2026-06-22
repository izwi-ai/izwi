---
title: "izwi tts"
description: "Generate text-to-speech audio from the command line with model, voice, and output options."
icon: "volume-2"
---
# izwi tts

Generate speech from text.

---

## Synopsis

```bash
izwi tts <TEXT> [OPTIONS]
```

---

## Description

Converts text to speech using a TTS model. The OSS server emits WAV audio for the CLI path, with voice selection and real-time streaming.

### Kokoro-82M Prerequisite (`espeak-ng`)

`Kokoro-82M` requires `espeak-ng` to be installed on the host system (used for phonemization).

- Install instructions:
  - [macOS](/installation/macos#optional-install-espeak-ng-for-kokoro-82m)
  - [Linux](/installation/linux#optional-install-espeak-ng-for-kokoro-82m)
  - [Windows](/installation/windows#optional-install-espeak-ng-for-kokoro-82m)

---

## Arguments

| Argument | Description |
|----------|-------------|
| `<TEXT>` | Text to synthesize (use `-` to read from stdin) |

---

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model <MODEL>` | TTS model to use | `qwen3-tts-0.6b-base` |
| `-s, --speaker <VOICE>` | Voice/speaker (name or audio path) | `default` |
| `-o, --output <PATH>` | Output file path | stdout |
| `-f, --format <FORMAT>` | Audio format requested from the server. OSS native output is `wav`; compressed choices require server-side encoder support. | `wav` |
| `-r, --speed <SPEED>` | Speech speed (0.5-2.0) | `1.0` |
| `-t, --temperature <TEMP>` | Sampling temperature | `0.7` |
| `--stream` | Stream output in real-time | — |
| `--allow-format-fallback` | Allow WAV output when a requested compressed format is unavailable | — |
| `-p, --play` | Play audio after generation | — |

---

## Examples

### Basic usage

```bash
izwi tts "Hello, world!" --output hello.wav
```

### Kokoro-82M

```bash
izwi tts "Hello my name is Bella" \
  --model Kokoro-82M \
  --speaker af_bella \
  --output kokoro.wav
```

### Play immediately

```bash
izwi tts "Hello, world!" --play
```

### WAV output

```bash
izwi tts "Hello, world!" --format wav --output hello.wav
```

### Adjust speed

```bash
# Slower
izwi tts "Speaking slowly" --speed 0.75 --output slow.wav

# Faster
izwi tts "Speaking quickly" --speed 1.5 --output fast.wav
```

### Read from stdin

```bash
echo "Text from pipe" | izwi tts - --output piped.wav
cat article.txt | izwi tts - --output article.wav
```

### CustomVoice speaker presets

```bash
izwi tts "Hello from a built-in CustomVoice speaker" \
  --model Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --speaker Aiden \
  --output cloned.wav
```

### Streaming with playback

```bash
izwi tts "Long text for streaming" --stream --play
```

> CLI `izwi tts` currently exposes speaker/voice selection only.
> Reference-audio cloning and prompt-based voice design are available via Web UI/API workflows.

---

## Audio Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| `wav` | `.wav` | Native OSS output format. |
| `mp3`, `ogg`, `flac`, `aac` | Matching extension when supported | Recognized by the CLI enum, but the OSS server does not bundle compressed encoders yet. Add `--allow-format-fallback` when you intentionally want the CLI to accept WAV bytes for a compressed request; default filenames then use the actual returned extension. |

---

## Models

| Model | Type | Description |
|-------|------|-------------|
| `Kokoro-82M` | Standard | Lightweight TTS (requires `espeak-ng`) |
| `qwen3-tts-0.6b-base` | Base | General-purpose TTS + reference-voice workflows |
| `qwen3-tts-0.6b-customvoice` | CustomVoice | Built-in speaker presets |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Design | Voice design-capable model family |
| `Voxtral-4B-TTS-2603` | Preset voices | 20 built-in voices, 24 kHz output, CC BY-NC 4.0 |
| `qwen3-tts-1.7b-*` | Larger | Higher quality variants |

---

## See Also

- [Text-to-Speech Guide](/features/text-to-speech)
- [Voice Cloning Guide](/features/voice-cloning)
- [Voice Design Guide](/features/voice-design)
