---
title: "Voice Presets"
description: "Reference built-in voice and speaker IDs for Qwen3 CustomVoice, Kokoro, Voxtral TTS, LFM2.5 Audio, and Chatterbox."
sidebarTitle: "Voice Presets"
icon: "audio-lines"
---
Use these IDs with `izwi tts --speaker`, the Text to Speech page, or
the `voice` field on `/v1/audio/speech` (or `speaker` on `/v1/text-to-speech`).

Saved and reference voices use `--saved-voice-id`, `reference_audio`, or the
Voice Studio saved voice library instead of these built-in speaker IDs.

---

## Model Families

| Family | Voice source |
|--------|--------------|
| Qwen3 CustomVoice | Nine named built-in speakers |
| Kokoro-82M | 41 voice IDs (English, Spanish, French, Hindi, Italian, Portuguese); requires `espeak-ng` |
| Voxtral TTS | 20 bundled preset voices |
| LFM2.5 Audio | Four English regional presets |
| VibeVoice-1.5B | Reference voice or saved voice only; no built-in presets |
| FishAudio-S2-Pro | Reference voice or saved voice only |
| Chatterbox-500M, Chatterbox-Multilingual-500M | One built-in voice (`default`), or a reference or saved voice (no transcript needed) |
| Qwen3 Base | Reference voice or saved voice only |
| Qwen3 VoiceDesign | Prompt instructions, then optional saved voice reuse |

---

## Qwen3 CustomVoice

| ID | Display name | Language | Description |
|----|--------------|----------|-------------|
| `Vivian` | Vivian | Chinese | Warm and expressive female voice |
| `Serena` | Serena | English | Clear and professional female voice |
| `Ryan` | Ryan | English | Confident and friendly male voice |
| `Aiden` | Aiden | English | Young and energetic male voice |
| `Dylan` | Dylan | English | Deep and authoritative male voice |
| `Eric` | Eric | English | Calm and measured male voice |
| `Sohee` | Sohee | Korean | Gentle and melodic female voice |
| `Ono_anna` | Anna | Japanese | Soft and pleasant female voice |
| `Uncle_fu` | Uncle Fu | Chinese | Mature and wise male voice |

Example:

```bash
izwi tts "Hello from Serena." \
  --model Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --speaker Serena \
  --output serena.wav
```

---

## Kokoro-82M

Kokoro voice IDs use a two-letter prefix:

| Prefix | Language/region | Gender marker |
|--------|-----------------|---------------|
| `af`, `am` | American English | female / male |
| `bf`, `bm` | British English | female / male |
| `ef`, `em` | Spanish | female / male |
| `ff` | French | female |
| `hf`, `hm` | Hindi | female / male |
| `if`, `im` | Italian | female / male |
| `pf`, `pm` | Brazilian Portuguese | female / male |

Available IDs:

```text
af_alloy, af_aoede, af_bella, af_heart, af_jessica, af_kore,
af_nicole, af_nova, af_river, af_sarah, af_sky, am_adam,
am_echo, am_eric, am_fenrir, am_liam, am_michael, am_onyx,
am_puck, am_santa, bf_alice, bf_emma, bf_isabella, bf_lily,
bm_daniel, bm_fable, bm_george, bm_lewis, ef_dora, em_alex,
em_santa, ff_siwis, hf_alpha, hf_beta, hm_omega, hm_psi,
if_sara, im_nicola, pf_dora, pm_alex, pm_santa
```

Example:

```bash
izwi tts "Hello from Bella." \
  --model Kokoro-82M \
  --speaker af_bella \
  --output kokoro-bella.wav
```

---

## Voxtral TTS

| ID | Voice |
|----|-------|
| `casual_female` | Casual English female |
| `casual_male` | Casual English male |
| `cheerful_female` | Cheerful English female |
| `neutral_female` | Neutral English female |
| `neutral_male` | Neutral English male |
| `pt_male`, `pt_female` | Portuguese male/female |
| `nl_male`, `nl_female` | Dutch male/female |
| `it_male`, `it_female` | Italian male/female |
| `fr_male`, `fr_female` | French male/female |
| `es_male`, `es_female` | Spanish male/female |
| `de_male`, `de_female` | German male/female |
| `ar_male` | Arabic male |
| `hi_male`, `hi_female` | Hindi male/female |

---

## LFM2.5 Audio

| ID | Voice |
|----|-------|
| `US Female` | US female preset |
| `US Male` | US male preset |
| `UK Female` | UK female preset |
| `UK Male` | UK male preset |

---

## Chatterbox

| ID | Voice |
|----|-------|
| `default` | The English voice bundled with the model (`conds.pt`); Chatterbox Multilingual speaks other languages with it too |

---

## See Also

- [Voices](/features/voices)
- [Text-to-Speech](/features/text-to-speech)
- [Models](/models)
