---
title: "Voice Studio"
description: "Create, clone, design, preview, manage, and reuse voices from the unified Izwi Voices workspace."
sidebarTitle: "Voices"
icon: "library"
---
Voice Studio is the unified workspace for reusable voices. In the current app,
open **Voices** from the sidebar to manage saved voices, browse built-in voices,
and create new cloned or designed voices.

The legacy `/voice-cloning` and `/voice-design` routes redirect into this
workspace when Voice Studio is enabled.

---

## What You Can Do

| Workflow | What it does |
|----------|--------------|
| Saved voices | List, preview, reuse, and delete voices saved from cloning or design workflows. |
| Built-in voices | Browse model-provided speaker presets and generate preview samples. |
| New Voice | Start a cloned voice from reference audio or a designed voice from a text prompt. |
| Models | Download, load, unload, or switch the models needed for built-in voices and voice design. |
| Use in TTS | Send a saved voice or built-in speaker directly to the Text to Speech page. |

---

## Using the Web UI

1. Open `http://localhost:8080`.
2. Select **Voices** in the sidebar.
3. Use the **Saved**, **Built-in**, or **All** views to find a voice.
4. Select **New Voice** to create a cloned or designed voice.
5. Select **Use in TTS** to continue generation in Text to Speech.

The **Models** action opens the same route-aware model manager used elsewhere
in the app. Built-in voice previews require a compatible built-in voice model to
be loaded.

---

## Saved Voices

Saved voices are reusable voice assets stored by the local server. A saved voice
can come from:

- A reference-audio voice clone
- A prompt-designed voice
- A generated voice source saved through the API

Saved voices can be previewed, deleted, and reused in Text to Speech. The API
surface is `/v1/voices`, and `izwi tts` can reuse a saved voice with
`--saved-voice-id`.

```bash
izwi tts "Use my saved narration voice." \
  --model Qwen3-TTS-12Hz-0.6B-Base \
  --saved-voice-id voice_abc123 \
  --output narration.wav
```

---

## Built-In Voices

Built-in voices are model-provided speaker presets. Voice Studio shows the
available speakers for the selected model, can generate short preview samples,
and can send a speaker into Text to Speech.

Use built-in voices when you want quick, repeatable presets without reference
audio or a design prompt. Use saved voices when you want a custom reusable voice
asset.

See [Voice Presets](/models/voice-presets) for the current speaker IDs.

---

## Creating New Voices

Select **New Voice** from Voice Studio to choose a creation flow:

| Flow | Input | Best for |
|------|-------|----------|
| Clone | Reference audio plus reference transcript | Reusing an existing permitted voice |
| Design | Natural-language voice description | Creating a new synthetic character or style |

See the dedicated guides for details:

- [Voice Cloning](/features/voice-cloning)
- [Voice Design](/features/voice-design)

---

## API Routes

| Route family | Purpose |
|--------------|---------|
| `/v1/voices` | Saved voice list/create/delete/audio routes |
| `/v1/voice-clones` | Persisted voice-cloning generation history |
| `/v1/voice-designs` | Persisted voice-design generation history |
| `/v1/audio/speech` | Immediate generation with built-in speakers, saved voices, reference audio, or design instructions |

See the [API Reference](/api#saved-voices) for exact saved voice fields and
the [Audio Speech API](/api#audio-speech) for generation inputs.

---

## See Also

- [Text-to-Speech](/features/text-to-speech)
- [Voice Cloning](/features/voice-cloning)
- [Voice Design](/features/voice-design)
- [Models](/models)
