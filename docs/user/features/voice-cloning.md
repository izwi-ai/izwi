# Voice Cloning

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

### Step 1: Upload Reference Audio

1. Navigate to **Voice Cloning** in the sidebar
2. Upload a reference audio file
3. The audio should be:
   - 5-30 seconds long
   - Clear speech, minimal background noise
   - Single speaker

### Step 2: Generate Speech

1. Enter the text you want to speak
2. Click **Generate**
3. Listen to the output in the cloned voice

### Step 3: Save and Reuse

- Download generated audio
- Save the voice profile for future use

---

## Using the CLI

Reference-audio cloning workflows are currently exposed in the Web UI and API routes.
CLI `izwi tts` supports voice/speaker selection, but does not currently expose direct `reference_audio` + `reference_text` parameters.

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
| `reference_text` | String | Transcript of reference audio |
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
`/v1/audio/speech`. See the [API Reference](../api.md#saved-voices) for the
saved voice routes and exact fields.

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
- Very short clips (<3 seconds)
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

Larger models produce more accurate voice clones.

---

## Ethical Considerations

Voice cloning is a powerful technology. Please use it responsibly:

- **Get consent** — Only clone voices with permission
- **Don't impersonate** — Never use cloned voices to deceive
- **Respect privacy** — Don't clone voices without authorization
- **Legal compliance** — Follow applicable laws and regulations

---

## Limitations

- **Accent accuracy** — May not perfectly capture all accents
- **Emotional range** — Cloned voices may have limited expressiveness
- **Unique characteristics** — Some voice qualities are hard to replicate
- **Language** — Best results in the model's primary language

---

## See Also

- [Voice Design](./voice-design.md) — Create voices from descriptions
- [Text-to-Speech](./text-to-speech.md) — Standard TTS
- [Models](../models/index.md) — Download models
