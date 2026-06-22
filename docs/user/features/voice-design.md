---
title: "Voice Design"
description: "Design synthetic voices from text descriptions and use them in Izwi speech workflows."
sidebarTitle: "Voice Design"
icon: "wand-sparkles"
---
# Voice Design

Create custom voices from text descriptions — no audio samples required.

---

## Overview

Voice design generates unique voices based on natural language descriptions. Describe the voice you want, and Izwi creates it:

- **No samples needed** — Create voices from scratch
- **Infinite variety** — Design any voice you can describe
- **Quick iteration** — Rapidly test different voice concepts
- **Creative freedom** — Perfect for characters and personas

---

## Getting Started

### Download a Voice Design Model

```bash
izwi pull Qwen3-TTS-12Hz-1.7B-VoiceDesign
```

### Design a Voice

Describe the voice you want:

```
A warm, friendly female voice with a slight British accent. 
Middle-aged, professional but approachable.
```

---

## Using the Web UI

### Step 1: Describe Your Voice

1. Navigate to **Voice Design** in the sidebar
2. Enter a description of your desired voice
3. Be specific about characteristics you want

### Step 2: Generate Sample

1. Enter sample text to hear the voice
2. Click **Generate**
3. Listen to the result

### Step 3: Iterate

- Adjust your description
- Generate again
- Repeat until satisfied

---

## Voice Description Tips

### Effective Descriptions

Include details about:

| Aspect | Examples |
|--------|----------|
| **Gender** | Male, female, androgynous |
| **Age** | Young, middle-aged, elderly |
| **Tone** | Warm, authoritative, playful |
| **Accent** | British, Southern US, neutral |
| **Pace** | Fast, measured, deliberate |
| **Energy** | Energetic, calm, subdued |
| **Character** | Professional, friendly, mysterious |

### Example Descriptions

**News anchor:**
```
A professional male voice, mid-30s, with a clear American accent. 
Authoritative and trustworthy, with measured pacing.
```

**Children's narrator:**
```
A warm, enthusiastic female voice. Friendly and expressive, 
perfect for storytelling. Slightly higher pitch with playful energy.
```

**AI assistant:**
```
A calm, neutral voice with no strong accent. Clear and helpful, 
not robotic but not overly emotional. Professional and efficient.
```

**Audiobook narrator:**
```
A rich, deep male voice with a slight British accent. 
Mature and sophisticated, with excellent diction and 
a storytelling quality.
```

---

## Using the CLI

Prompt-driven voice design is available in Web UI/API workflows.
CLI `izwi tts` currently does not expose a dedicated `voice_description`/`instructions` flag.

---

## Using the API

### Endpoint

```
POST /v1/audio/speech
```

### Request

```json
{
  "model": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
  "input": "Hello, this is my designed voice.",
  "instructions": "A warm, friendly female voice with a British accent"
}
```

### Example (curl)

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "input": "Hello, this is my designed voice.",
    "instructions": "A warm, friendly female voice"
  }' \
  --output designed.wav
```

Voice design history is also available through
`/v1/voice-designs`. See the
[API Reference](/api#speech-history) for the persisted route
family and `/v1/audio/speech` streaming details.

---

## Available Models

| Model | Size | Quality |
|-------|------|---------|
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | ~4.2 GB | Better |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit` | ~2.2 GB | Good + lower memory |

Larger models better interpret complex descriptions.

---

## Best Practices

### Be Specific

❌ "A nice voice"

✅ "A warm, professional female voice in her 40s with a calm, reassuring tone"

### Use Comparisons

"Similar to a podcast host — conversational but polished"

### Describe the Context

"A voice suitable for meditation apps — slow, soothing, and peaceful"

### Iterate

Start broad, then refine:
1. "A male voice"
2. "A young male voice with energy"
3. "A young male voice with energy, like a sports commentator"

---

## Limitations

- **Consistency** — Same description may produce slightly different voices
- **Extreme requests** — Very unusual voices may not generate well
- **Accents** — Some accents are better supported than others
- **Singing** — Designed for speech, not singing

---

## Voice Design vs Voice Cloning

| Aspect | Voice Design | Voice Cloning |
|--------|--------------|---------------|
| **Input** | Text description | Audio sample |
| **Use case** | Create new voices | Replicate existing voices |
| **Consistency** | May vary slightly | More consistent |
| **Flexibility** | Unlimited creativity | Limited to source |

---

## See Also

- [Voice Cloning](/features/voice-cloning) — Clone from audio samples
- [Text-to-Speech](/features/text-to-speech) — Standard TTS
- [Models](/models) — Download models
