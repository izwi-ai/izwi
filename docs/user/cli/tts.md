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
| `-s, --speaker <VOICE>` | Built-in voice/speaker ID, or VibeVoice speaker label | model default |
| `--saved-voice-id <ID>` | Reuse a saved reference voice from `/v1/voices` | — |
| `--reference-audio <PATH>` | Reference audio file for voice cloning | — |
| `--reference-text <TEXT>` | Transcript for the reference audio | — |
| `--reference-text-file <PATH>` | File containing the reference transcript | — |
| `--instructions <TEXT>` | Voice direction prompt for voice-design models | — |
| `-o, --output <PATH>` | Output file path | stdout |
| `-f, --format <FORMAT>` | Audio format requested from the server. OSS native output is `wav`; compressed choices require server-side encoder support. | `wav` |
| `-r, --speed <SPEED>` | Speech speed (0.5-2.0) | `1.0` |
| `-t, --temperature <TEMP>` | Sampling temperature | `0.7` |
| `--stream` | Stream output in real-time | — |
| `--allow-format-fallback` | Allow WAV output when a requested compressed format is unavailable | — |
| `-p, --play` | Request playback after generation. The current CLI saves audio and reports playback as not implemented. | — |

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

### Request playback

```bash
izwi tts "Hello, world!" --play
```

The current CLI still saves the generated audio, then reports that playback is
not implemented in this version.

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

### Reference-audio voice cloning

Base and VibeVoice models can clone from a reference audio sample. Provide the
audio and matching transcript together:

```bash
izwi tts "This sentence will use the reference voice." \
  --model Qwen3-TTS-12Hz-0.6B-Base \
  --reference-audio samples/reference.wav \
  --reference-text "This is the exact text spoken in the reference sample." \
  --output cloned.wav
```

For longer transcripts, read the reference text from a file:

```bash
izwi tts "Generate with the saved reference transcript." \
  --model Qwen3-TTS-12Hz-1.7B-Base \
  --reference-audio samples/reference.wav \
  --reference-text-file samples/reference.txt \
  --output cloned-long.wav
```

### Saved voice reuse

Saved voices created through Voice Studio or `/v1/voices` can be reused without
resending reference audio:

```bash
izwi tts "Use my saved narration voice." \
  --model Qwen3-TTS-12Hz-0.6B-Base \
  --saved-voice-id voice_abc123 \
  --output saved-voice.wav
```

Do not combine `--saved-voice-id` with `--reference-audio` or
`--reference-text`; choose one voice source per request.

### Prompt-based voice design

VoiceDesign models accept a natural-language direction prompt:

```bash
izwi tts "Welcome to the evening edition." \
  --model Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --instructions "A calm, warm newsreader voice with measured pacing" \
  --output designed.wav
```

### Streaming with playback request

```bash
izwi tts "Long text for streaming" --stream --play
```

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
| `VibeVoice-1.5B` | Long-form/reference voice | Reference-voice TTS with long-form positioning |
| `Voxtral-4B-TTS-2603` | Preset voices | 20 built-in voices, 24 kHz output, CC BY-NC 4.0 |
| `qwen3-tts-1.7b-*` | Larger | Higher quality variants |

---

## See Also

- [Text-to-Speech Guide](/features/text-to-speech)
- [Voice Presets](/models/voice-presets)
- [Voice Cloning Guide](/features/voice-cloning)
- [Voice Design Guide](/features/voice-design)
