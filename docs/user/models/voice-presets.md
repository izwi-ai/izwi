---
title: "Voice Presets"
description: "Reference built-in voice and speaker IDs for Qwen3 CustomVoice, Kokoro, Voxtral TTS, LFM2.5 Audio, Chatterbox, Maya1, and Maya 2 Native."
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
| Maya1-3B | Nine described voices, or a description of your own in `instructions` |
| Maya2-Native-2.5B | 20 named voices, or a reference or saved voice (transcript needed) |
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

## Maya1

`Maya1-3B` has no recorded voices: each built-in voice is a description in
the style the model was trained on, and the model invents a voice to match
it. A request's own description (`instructions`, or `--voice-description` in
the CLI) replaces the voice. The same description and seed give the same
audio for the same text; across different texts, the voice is similar rather
than identical. OpenAI's voice names map to the nearest preset (`alloy` and
`nova` to `ava`, `ash` and `echo` to `leo`, `ballad` and `fable` to `james`,
`coral` and `shimmer` to `zoe`, `onyx` and `cedar` to `omar`, `sage` to
`nora`, `verse` to `arjun`, `marin` to `priya`).

| Voice | Description |
|-------|-------------|
| `ava` (default) | Realistic female voice in the 30s age with a american accent. Normal pitch, warm timbre, conversational pacing, neutral tone delivery at med intensity, podcast domain, podcast_host role, neutral delivery. |
| `leo` | Realistic male voice in the 30s age with a american accent, otherwise as `ava`. |
| `nora` | Realistic female voice in the 30s age with a british accent. Normal pitch, smooth timbre, conversational pacing, corporate domain, explainer_video_voice role, formal delivery. |
| `james` | Realistic male voice in the 40s age with a british accent. Low pitch, deep timbre, slow pacing, entertainment domain, storyteller role. |
| `priya` | Realistic female voice in the 30s age with a indian accent. Normal pitch, smooth timbre, conversational pacing, support domain, customer_support_agent role. |
| `arjun` | Realistic male voice in the 20s age with a indian accent. Normal pitch, warm timbre, energetic tone, education domain, elearning_instructor role, casual delivery. |
| `zoe` | Realistic female voice in the 20s age with a american accent. High pitch, smooth timbre, brisk pacing, energetic tone, social_content domain, youtube_vlogger role. |
| `omar` | Realistic male voice in the 30s age with a middle_eastern accent. Low pitch, deep timbre, corporate domain, event_host role, formal delivery. |
| `robot` | Creative, ai_machine_voice character. Male voice in their 20s with a american accent. Normal pitch, robotic timbre, conversational pacing. |

Descriptions work best when they follow this template (age, gender, accent,
pitch, timbre, pacing, emotion and intensity, and for realistic voices a
domain and role) and end with a full stop: the model reads a short or
unfinished description aloud more often. Izwi adds the final full stop when
it is missing.

## Maya 2 Native

`Maya2-Native-2.5B` has 20 named voices. Any voice can speak any of the
model's 12 languages: the last column lists the languages with at least an
hour of that voice's training audio, and the voice speaks the others
cross-lingually. No built-in voice has Assamese training audio. Voice names
are not case-sensitive, and a request without one uses `Ahana`. OpenAI's
voice names map to voices trained on ten or more languages (`alloy` and `sage`
to `Vaidehi`, `ash` to `Siddhant`, `ballad` and `onyx` to `Vasant`, `cedar`
and `fable` to `Anirban`, `coral` to `Jahnavi`, `echo` to `Nakul`, `marin` to
`Malini`, `nova` to `Ahana`, `shimmer` to `Charvi`, `verse` to `Satvik`).

| Voice | Description | Trained languages |
|-------|-------------|-------------------|
| `Ahana` (default) | Young female voice, bright, youthful | All but Assamese |
| `Vaidehi` | Adult female voice, calm, gentle | All but Assamese |
| `Malini` | Mature female voice, steady narrator | All but Assamese |
| `Charvi` | Young female voice, energetic, upbeat | All but Assamese |
| `Jahnavi` | Adult female voice, warm, friendly | All but Assamese and Punjabi |
| `Anirban` | Adult male voice, steady narrator | All but Assamese |
| `Nakul` | Young male voice, bright, youthful | All but Assamese |
| `Vasant` | Mature male voice, steady narrator | All but Assamese and Punjabi |
| `Siddhant` | Adult male voice, clear, professional | All but Assamese |
| `Satvik` | Young male voice, warm, friendly | All but Assamese |
| `Abhay` | Adult male voice, natural conversational style | English, Hindi |
| `Nivedita` | Adult female voice, natural conversational style | English, Hindi |
| `Ishwari` | Adult female voice, natural conversational style | Hindi |
| `Ritvik` | Adult male voice, natural conversational style | Hindi |
| `Vrinda` | Young female voice, natural conversational style | Hindi |
| `Sreekar` | Adult male voice, natural conversational style | English, Hindi, Telugu |
| `Kayal` | Adult female voice, natural conversational style | English, Tamil |
| `Shaunak` | Young male voice, natural conversational style | English, Hindi |
| `Akshat` | Adult male voice, natural conversational style | English, Hindi |
| `Tanushree` | Adult female voice, natural conversational style | English, Hindi |

Example:

```bash
izwi tts "வணக்கம், எப்படி இருக்கிறீர்கள்?" \
  --model Maya2-Native-2.5B \
  --speaker Kayal \
  --output maya2-kayal.wav
```

To speak in a voice of your own instead, clone one from reference audio and
its transcript (see [Voice Cloning](/features/voice-cloning)).
