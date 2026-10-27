---
title: "Voice Cloning"
description: "Create custom voices from reference audio and use them for local text-to-speech generation."
sidebarTitle: "Voice Cloning"
icon: "copy"
---
Clone any voice from a short audio sample and use it for text-to-speech generation.

---

## Overview

Voice cloning creates a custom voice from a reference audio sample. Use it to:

- **Personalize TTS** — Generate speech in a specific voice
- **Create characters** — Unique voices for games or media
- **Accessibility** — Preserve a person's voice
- **Localization** — Maintain voice consistency across languages

---

## Getting Started

### Download a Voice Cloning Model

```bash
izwi pull Qwen3-TTS-12Hz-0.6B-Base
```

### Clone a Voice

1. Prepare a reference audio file (5-30 seconds of clear speech)
2. Use the voice for TTS generation

---

## Using the Web UI

Voice cloning now lives inside the unified **Voices** workspace.

### Step 1: Upload Reference Audio

1. Navigate to **Voices** in the sidebar and choose the clone flow
2. Upload a reference audio file
3. The audio should be:
   - 5-30 seconds long
   - Clear speech, minimal background noise
   - Single speaker

The reference transcript is marked optional: Chatterbox clones from the audio
alone, while Qwen3-TTS Base, VibeVoice, Fish Audio S2 Pro and Maya 2 Native
need it. With Maya 2 Native, the dialog also asks for the language of the
recording, or detects it from the transcript; the voice can then speak any of
the model's languages.

### Step 2: Generate Speech

1. Enter the text you want to speak
2. Click **Generate**
3. Listen to the output in the cloned voice

### Step 3: Save and Reuse

- Download generated audio
- Save the voice profile for future use

---

## Using the CLI

Use `izwi tts` with a Base, VibeVoice, `FishAudio-S2-Pro` or `Maya2-Native-2.5B` TTS model and provide both the
reference audio and its transcript:

```bash
izwi tts "This sentence will use the cloned voice." \
  --model Qwen3-TTS-12Hz-0.6B-Base \
  --reference-audio samples/reference.wav \
  --reference-text "This is the exact text spoken in the reference sample." \
  --output cloned.wav
```

For longer reference transcripts, store the transcript in a file:

```bash
izwi tts "This sentence uses a transcript file." \
  --model Qwen3-TTS-12Hz-1.7B-Base \
  --reference-audio samples/reference.wav \
  --reference-text-file samples/reference.txt \
  --output cloned-from-file.wav
```

`Chatterbox-500M` and `Chatterbox-Multilingual-500M` clone from the reference
audio alone. They do not condition on the reference's words, so a transcript is
optional and ignored. With the multilingual model, a reference in a language
other than the request's tends to lend the output its accent; Resemble AI
suggests a matching reference, or `cfg_weight` 0 to soften the accent:

```bash
izwi tts "This sentence will use the cloned voice." \
  --model Chatterbox-500M \
  --reference-audio samples/reference.wav \
  --output cloned-chatterbox.wav
```

`Maya2-Native-2.5B` clones from reference audio and its exact transcript, and
the cloned voice can speak any of the model's 12 languages, not only the
reference's. Name the language of the reference with `--reference-language`
(a code such as `en`), or leave it to be read from the transcript's script.
The reference must last 0.5 to 30 seconds; Maya Research recommends 3 to 10.
Here an English reference speaks Hindi, which Izwi reads from the text's
Devanagari script (`--language hi` says the same):

```bash
izwi tts "आपका ऑर्डर कल शाम तक पहुँच जाएगा।" \
  --model Maya2-Native-2.5B \
  --reference-audio samples/reference.wav \
  --reference-text "This is the exact text spoken in the reference sample." \
  --reference-language en \
  --output cloned-hindi.wav
```

Saved voices created in Voice Studio or through `/v1/voices` can be reused from
the CLI:

```bash
izwi tts "This sentence uses my saved voice." \
  --model Qwen3-TTS-12Hz-0.6B-Base \
  --saved-voice-id voice_abc123 \
  --output saved-voice.wav
```

`--saved-voice-id` is mutually exclusive with direct `--reference-audio` and
`--reference-text` input, and `--reference-text` needs `--reference-audio`.

---

## Using the API

### Endpoint

```
POST /v1/audio/speech
```

### Request (JSON)

| Field | Type | Description |
|-------|------|-------------|
| `model` | String | Base model ID (for example `Qwen3-TTS-12Hz-0.6B-Base`) |
| `input` | String | Text to synthesize |
| `reference_audio` | String | Base64-encoded reference audio |
| `reference_text` | String | Transcript of reference audio (optional and ignored for Chatterbox) |
| `reference_language` | String | Language the reference audio is spoken in, one of Maya 2 Native's 12 codes (`Maya2-Native-2.5B` only; read from the transcript when unset) |
| `saved_voice_id` | String | Optional saved voice reference to reuse instead of resending audio |

### Example (curl)

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-TTS-12Hz-0.6B-Base",
    "input": "Hello, this is my cloned voice",
    "reference_audio": "<base64-audio>",
    "reference_text": "Hello, this is my cloned voice sample."
  }' \
  --output cloned.wav
```

Saved voices can also be managed through `/v1/voices` and reused from
`/v1/audio/speech`. See the [API Reference](/api#saved-voices) for the
saved voice routes and exact fields. A voice saved with a `reference_language`
passes it to `Maya2-Native-2.5B` requests that use it, unless the request sets
its own.

---

## Reference Audio Guidelines

### Ideal Reference Audio

| Aspect | Recommendation |
|--------|----------------|
| **Duration** | 5-30 seconds |
| **Quality** | High quality, clear audio |
| **Content** | Natural speech, varied intonation |
| **Background** | Minimal noise |
| **Speaker** | Single speaker only |

### Good Examples

- Podcast clips
- Interview segments
- Voice memos
- Audiobook excerpts

### Poor Examples

- Music with vocals
- Multiple speakers
- Heavy background noise
- Very short clips (under 3 seconds)
- Whispered or distorted speech

---

## Tips for Best Results

1. **Quality over quantity** — A clear 10-second clip beats a noisy 30-second one
2. **Natural speech** — Avoid monotone or exaggerated delivery
3. **Match content** — Reference emotion should match desired output
4. **Consistent volume** — Avoid clips with volume changes
5. **No music** — Background music interferes with cloning

---

## Available Models

| Model | Size | Quality |
|-------|------|---------|
| `Qwen3-TTS-12Hz-0.6B-Base` | ~2.3 GB | Good |
| `Qwen3-TTS-12Hz-1.7B-Base` | ~4.2 GB | Better |
| `VibeVoice-1.5B` | ~5.0 GB | Better long-form/reference voice |
| `FishAudio-S2-Pro` | ~11.5 GB | Reference voice or saved voice only |
| `Chatterbox-500M` | ~3.2 GB | English; clones from audio alone, no transcript needed |
| `Chatterbox-Multilingual-500M` | ~3.3 GB | 22 languages (set `language`); clones from audio alone |
| `Maya2-Native-2.5B` | ~5.5 GB | English and 11 Indic languages; needs a transcript; clones across languages |

Larger models produce more accurate voice clones.

---

## Ethical Considerations

Voice cloning is a powerful technology. Please use it responsibly:

- **Get consent** — Only clone voices with permission
- **Don't impersonate** — Never use cloned voices to deceive
- **Respect privacy** — Don't clone voices without authorization
- **Legal compliance** — Follow applicable laws and regulations

Both Chatterbox models mark every utterance with Resemble AI's imperceptible PerTh
watermark, which Resemble AI's open-source `perth` package can detect.

---

## Limitations

- **Accent accuracy** — May not perfectly capture all accents
- **Emotional range** — Cloned voices may have limited expressiveness
- **Unique characteristics** — Some voice qualities are hard to replicate
- **Language** — Best results in the model's primary language

---

## See Also

- [Voices](/features/voices) — Manage and reuse saved voices
- [Voice Design](/features/voice-design) — Create voices from descriptions
- [Text-to-Speech](/features/text-to-speech) — Standard TTS
- [Models](/models) — Download models
