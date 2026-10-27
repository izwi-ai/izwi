---
title: "izwi tts"
description: "Generate text-to-speech audio from the command line with model, voice, and output options."
icon: "volume-2"
---
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
| `-m, --model <MODEL>` | TTS model to use | `Kokoro-82M` |
| `-s, --speaker <VOICE>` | Built-in voice/speaker ID, or VibeVoice speaker label | model default |
| `--saved-voice-id <ID>` | Reuse a saved reference voice from `/v1/voices` | — |
| `--reference-audio <PATH>` | Reference audio file for voice cloning | — |
| `--reference-text <TEXT>` | Transcript for the reference audio; needs `--reference-audio` (optional for Chatterbox) | — |
| `--reference-text-file <PATH>` | File containing the reference transcript | — |
| `--instructions <TEXT>`, `--voice-description <TEXT>` | Voice direction for voice-design models; for `Maya1-3B`, the description of the voice to speak with (replaces `--speaker`) | — |
| `-o, --output <PATH>` | Output file path | `izwi_output_<timestamp>.wav` in the current directory |
| `-f, --format <FORMAT>` | `wav` or `pcm` (raw 16-bit little-endian mono, no header) | `wav` |
| `-r, --speed <SPEED>` | Speech speed multiplier (models with speed control) | model default |
| `-t, --temperature <TEMP>` | Sampling temperature | model default |
| `--language <LANG>` | Language of the text, as a code or name: required for non-English text with `Chatterbox-Multilingual-500M` (e.g. `fr`); a hint for Qwen3-TTS (e.g. `English`); for `Maya2-Native-2.5B`, a code (e.g. `hi`) or `auto` to read it from the text's script | model default |
| `--reference-language <CODE>` | Language the reference audio is spoken in (`Maya2-Native-2.5B` only, which can clone a voice into another language); unset, it is read from the transcript's script | — |
| `--exaggeration <F>` | Emotion intensity, 0 to 2 (Chatterbox only) | `0.5` |
| `--cfg-weight <F>` | Classifier-free guidance weight, 0 to 1; `0` turns guidance off (Chatterbox only) | `0.5` |
| `--min-p <F>` | Min-p sampling threshold, 0 to 1 (Chatterbox only) | `0.05` |
| `--stream` | Stream audio as it is generated into a WAV file and report time to first audio (WAV only) | — |
| `-p, --play` | Play the saved WAV with the system player (afplay on macOS; paplay, aplay, ffplay or play on Linux; PowerShell on Windows). Not with `--format pcm`. | — |

`--exaggeration`, `--cfg-weight` and `--min-p` are sent only when set, and the
server refuses one the chosen model does not support.

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

### Play after generation

```bash
izwi tts "Hello, world!" --play
```

Izwi saves the generated audio, then plays it with the system player.

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

Base, VibeVoice and FishAudio-S2-Pro models can clone from a reference audio sample. Provide the
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

### Chatterbox

`Chatterbox-500M` speaks with its built-in `default` voice, or clones from
`--reference-audio` alone (a transcript is optional and ignored):

```bash
izwi tts "This sentence will use the reference voice." \
  --model Chatterbox-500M \
  --reference-audio samples/reference.wav \
  --exaggeration 0.7 \
  --cfg-weight 0.3 \
  --output chatterbox.wav
```

For expressive or fast-talking reference voices, Resemble AI suggests a lower
`--cfg-weight` (about 0.3) with `--exaggeration` around 0.7 or higher.

`Chatterbox-Multilingual-500M` speaks the language `--language` names (English
without one), with the same voices and controls:

```bash
izwi tts "Habari za asubuhi, karibu sana." \
  --model Chatterbox-Multilingual-500M \
  --language sw \
  --output swahili.wav
```

`Chatterbox-500M` returns `400` for any language but English.
With `--stream`, Chatterbox's first audio arrives after about 2 s on an Apple
M1 Pro (Metal); the rest follows in windows. It refuses `--speed` and
`--instructions`.

### Maya1

`Maya1-3B` speaks with one of nine described voices (`--speaker ava`, the
default, or `leo`, `nora`, `james`, …), or with a voice you describe. Tags in
the text, such as `<laugh>`, `<sigh>` or `<whisper>`, are performed where they
stand:

```bash
izwi tts "I really did not expect that. <laugh> Let's do it again!" \
  --model Maya1-3B \
  --voice-description "Realistic female voice in the 40s age with a british accent. Low pitch, warm timbre, slow pacing, excited tone delivery at med intensity." \
  --output maya1.wav
```

The tags are `<angry>`, `<appalled>`, `<chuckle>`, `<cry>`, `<curious>`,
`<disappointed>`, `<excited>`, `<exhale>`, `<gasp>`, `<giggle>`, `<gulp>`,
`<laugh>`, `<laugh_harder>`, `<mischievous>`, `<sarcastic>`, `<scream>`,
`<sigh>`, `<sing>`, `<snort>` and `<whisper>`; the server returns `400` for
any other `<word>` in Maya1's text. Long text is spoken in sentences of up to
about 11 seconds, each with the same voice description.

### Maya 2 Native

`Maya2-Native-2.5B` speaks English and 11 Indic languages with one of 20
voices (`--speaker Ahana`, the default, or `Vaidehi`, `Anirban`, …; see
[Voice Presets](/models/voice-presets#maya-2-native)). It reads the language
from the text's script, or takes a code with `--language`. Devanagari text
reads as Hindi and Bengali script as Bengali, so name Marathi (`mr`) and
Assamese (`as`):

```bash
izwi tts "नमस्ते, आपका ऑर्डर कल पहुँच जाएगा।" \
  --model Maya2-Native-2.5B \
  --speaker Vaidehi \
  --output maya2.wav
```

It also clones a voice from reference audio and its exact transcript, and the
clone can speak another of its languages. `--reference-language` names the
language of the reference; without it, Izwi reads it from the transcript:

```bash
izwi tts "आपका ऑर्डर कल शाम तक पहुँच जाएगा।" \
  --model Maya2-Native-2.5B \
  --reference-audio samples/reference.wav \
  --reference-text "This is the exact text spoken in the reference sample." \
  --reference-language en \
  --output maya2-clone.wav
```

Tags in square brackets, such as `[laughs]`, `[sighs]` or `[whispers]`, are
performed on a best-effort basis. Maya Research names `[laughs]`, `[sighs]`,
`[whispers]`, `[sad]`, `[excited]`, `[curious]`, `[angry]` and `[surprised]`;
other lowercase `[tags]` are passed to the model too. Numbers, dates, currency
and similar are spoken as words in the text's language. Text longer than 280
characters is spoken in whole sentences of up to 280 characters (about 20
seconds each). It refuses `--speed` and `--instructions`.

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

### Streaming, then play

```bash
izwi tts "Long text for streaming" --stream --play
```

Izwi saves the streamed audio, then plays it.

---

## Audio Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| `wav` | `.wav` | 16-bit PCM WAV (default) |
| `pcm` | `.pcm` | Raw 16-bit little-endian mono PCM, no header (no `--play` or `--stream`) |

---

## Models

| Model | Type | Description |
|-------|------|-------------|
| `Kokoro-82M` | Standard | Default; lightweight TTS with built-in voices (requires `espeak-ng`) |
| `qwen3-tts-0.6b-base` | Base | Voice cloning only: needs `--reference-audio` + `--reference-text` or `--saved-voice-id` |
| `qwen3-tts-0.6b-customvoice` | CustomVoice | Built-in speaker presets |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Design | Voice design-capable model family; needs `--instructions` |
| `VibeVoice-1.5B` | Long-form/reference voice | Reference-voice TTS with long-form positioning |
| `FishAudio-S2-Pro` | Reference voice | Voice cloning only: needs `--reference-audio` + `--reference-text` or `--saved-voice-id` |
| `Voxtral-4B-TTS-2603` | Preset voices | 20 built-in voices, 24 kHz output, CC BY-NC 4.0 |
| `Chatterbox-500M` | Built-in/reference voice | English; built-in `default` voice or cloning from `--reference-audio` alone; watermarked, 24 kHz output, MIT |
| `Chatterbox-Multilingual-500M` | Built-in/reference voice | Chatterbox in 22 languages chosen with `--language`; watermarked, 24 kHz output, MIT |
| `qwen3-tts-1.7b-*` | Larger | Higher quality variants |

---

## See Also

- [Text-to-Speech Guide](/features/text-to-speech)
- [Voice Presets](/models/voice-presets)
- [Voice Cloning Guide](/features/voice-cloning)
- [Voice Design Guide](/features/voice-design)
