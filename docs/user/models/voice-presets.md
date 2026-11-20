---
title: "Voice Presets"
description: "Reference built-in voice and speaker IDs for Qwen3 CustomVoice, Kokoro, Voxtral TTS, LFM2.5 Audio, LFM3 Audio, Chatterbox, Maya1, Maya 2 Native, and Ming 3.0 TTS."
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
| LFM3 Audio | Two voices (`female`, `male`) in 19 locales |
| VibeVoice-1.5B | Reference voice or saved voice only; no built-in presets |
| FishAudio-S2-Pro | Reference voice or saved voice only |
| FishAudio-Drama-3 | No presets: a voice set by the sampling `seed`, or a reference or saved voice (transcript needed) |
| Audio8-TTS-Realistic-Air | No presets: a voice of the model's own, repeatable with `seed`, or a reference or saved voice (transcript needed) |
| Chatterbox-500M, Chatterbox-Multilingual-500M | One built-in voice (`default`), or a reference or saved voice (no transcript needed) |
| Maya1-3B | Nine described voices, or a description of your own in `instructions` |
| Maya2-Native-2.5B | 20 named voices, or a reference or saved voice (transcript needed) |
| Ming-3.0-tts-0.5B | 85 named voices (many with emotion variants), a reference or saved voice (transcript optional), or a description in `instructions` |
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

## LFM3 Audio

`LFM3-Audio-2B-GGUF` and `LFM3-Audio-2B-Q8_0-GGUF` have two voices. Voice
names are not case-sensitive, and a request without one uses `female`.

| ID | Voice |
|----|-------|
| `female` | Female voice (default) |
| `male` | Male voice |

Either voice speaks any of the 19 locales that `language` (`--language` in the
CLI) names: `en-US` (American English, the default), `en-GB` (British
English), `ar`, `da`, `de`, `el`, `fi`, `fr`, `he`, `it`, `ja`, `ko`, `nl`,
`pl`, `pt`, `es`, `sv`, `tr` and `zh`. Any other language returns `400`.

Example:

```bash
izwi tts "Guten Morgen, Ihre Bestellung ist unterwegs." \
  --model LFM3-Audio-2B-GGUF \
  --speaker male \
  --language de \
  --output lfm3-audio-de.wav
```

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

## Ming 3.0 TTS

`Ming-3.0-tts-0.5B` has 85 built-in voices: three English ones and 82 Chinese
ones. Any voice can be asked to speak any of the model's languages; each was
recorded in the one its table names. A request without a voice uses `US_Emma`
for English text and `灵小柔` for Chinese text. OpenAI's voice names map to
the English voices (`alloy`, `marin`, `nova` and `shimmer` to `US_Emma`;
`coral`, `fable` and `sage` to `UK_Charlotte`; the others to `UK_George`).

Many Chinese voices were recorded only in a set of emotions. Ask for one with
`emotion` (`--emotion` in the CLI); without it the voice speaks its neutral
delivery, or its first one when it has no neutral. A voice with one delivery
takes no `emotion`. A cloned voice takes any of the seven emotions.

### English voices

| Voice | Gender | Accent and use |
|-------|--------|----------------|
| `US_Emma` (default for English) | female | American; assistants, narration, conversational product voice |
| `UK_George` | male | British; audiobooks, formal announcements |
| `UK_Charlotte` | female | British; audiobooks, measured narration |

### Chinese voices

| Voice | Gender | Description | Emotions |
|-------|--------|-------------|----------|
| `灵小安` | female | Sweet, crisp and energetic, with an approachable, cute anime-style delivery. | — |
| `灵语汐` | female | Clear, bright and youthful, with a sunny, energetic delivery. | — |
| `灵小萌` | female | Bright, crisp and brisk, with an energetic yet dependable delivery. | — |
| `灵小柔` | female | Clear, sweet and articulate, with polite, patient and professional intonation. | — |
| `灵小雅` | female | Sweet, bright and youthful, with an energetic delivery. | — |
| `花小呗` | female | Crisp and sweet, with a distinctly childlike, lively delivery. | — |
| `灵韵` | female | Bright and crisp, with brisk, dependable delivery. | — |
| `妩媚妲己` | female | Sweet and crisp, with light, rising intonation and an alluring delivery. | — |
| `灵婉` | female | Clear and sweet, with a youthful, intimate and soothing delivery. | — |
| `灵小月` | female | Sweet, clear and energetic, with cheerful, welcoming intonation. | — |
| `灵朵儿` | female | Warm, thoughtful and approachable, with an unhurried, soothing storytelling style. | — |
| `灵小锋` | male | Warm and resonant, with natural confidence and a balance of friendliness and maturity. | — |
| `灵小川` | male | Warm, mellow and resonant, with sensitive, expressive delivery. | — |
| `灵小甄` | female | Bright, crisp and lively, with a friendly, persuasive recommendation style. | — |
| `灵小知` | female | Bright and confident, with strong, articulate and persuasive delivery. | — |
| `灵小惠` | female | Very fast and penetrating, conveying urgency, enthusiasm and professional confidence. | — |
| `灵小晴` | female | Very fast yet clear, with a powerful, bright delivery that conveys urgency. | — |
| `灵小琪` | male | Sunny, friendly and natural, with an enthusiastic recommendation style. | — |
| `灵梦` | female | Very sweet and youthful, with playful, affectionate delivery. | — |
| `灵恬` | female | Honey-sweet, with a gently playful yet assured romantic delivery. | — |
| `灵小汐` | female | Clear and sweet, with relaxed, natural and endearingly innocent delivery. | — |
| `灵小棠` | female | Clear and warm, with calm pacing and a gentle, experienced storytelling style. | — |
| `灵小梨` | female | Sweet and clear, combining youthful playfulness with gentle, expressive narration. | — |
| `灵小芸` | female | Warm and thoughtful, with sincere, calm and sensitive delivery. | — |
| `灵空` | male | Low and resonant, with close, breathy delivery that conveys affection and reassurance. | — |
| `灵宇` | male | Youthful and resonant, with a confident smile and gentle teasing. | — |
| `灵朗` | male | Low and resonant, combining confident warmth with affectionate reassurance. | — |
| `灵楷` | male | Warm and resonant, with slow, soothing late-night-radio narration. | — |
| `灵旭` | male | Clear, youthful and scholarly, with calm, gentle storytelling. | — |
| `灵江` | male | Warm and resonant, with relaxed, heartfelt reflections on youth and enthusiasm. | — |
| `悲伤女声` | female | Gentle yet determined, with a thoughtful, intimate storytelling style. | — |
| `皮六` | male | Bright, youthful and confident, with strong contrast and dramatic emphasis. | — |
| `悟空` | male | Highly expressive and theatrical, with vivid but measured exaggeration. | — |
| `灵晓声` | male | Youthful and resonant, with calm confidence and an assertive, charismatic delivery. | neutral, happy, sad, disgusted, surprised |
| `灵风信` | female | Sweet and clear, with warm, concerned and reassuring delivery. | neutral, happy, sad, angry, disgusted |
| `灵惜梦` | female | Bright and crisp, combining approachable warmth with independence and decisiveness. | neutral, happy, sad, angry, fearful, disgusted, surprised |
| `灵逸风` | male | Fresh and youthful, with relaxed, casual and slightly evasive delivery. | neutral, happy, sad, fearful, surprised |
| `灵悦澜` | female | Sweet and crisp, with a smiling, approachable and lively delivery. | neutral, happy, sad, angry, fearful, disgusted |
| `灵御心` | female | Bright, crisp and direct, with efficient, youthful energy. | neutral, happy, angry |
| `灵语恬` | female | Clear and gentle, with attentive, earnest delivery. | neutral, happy, sad, angry, fearful, surprised |
| `灵心恬` | female | Crisp and composed, with objective delivery and vivid storytelling. | neutral, happy, sad, angry, fearful, disgusted, surprised |
| `灵亦尘` | male | Warm and resonant, with natural, gentle concern and reassuring delivery. | neutral, fearful |
| `灵语尘` | male | Low, husky and world-weary, conveying helplessness and resignation. | neutral, happy, sad, fearful, disgusted, surprised |
| `灵悦兮` | female | Bright and crisp, with lively, confident and action-oriented delivery. | neutral, happy, sad, angry, fearful, disgusted, surprised |
| `灵若虚` | female | Clean and efficient, with thoughtful, objective yet warm delivery. | neutral, happy, sad, angry, disgusted, surprised |
| `灵未央` | female | Sweet and crisp, with playful, rising intonation and confident explanations. | neutral, happy, sad, angry, fearful, surprised |
| `灵梦瑶` | female | Soft and sweet, with a friendly, earnest and soothing explanatory style. | neutral, happy, sad, angry, surprised |
| `灵妙声` | male | Sweet and crisp, with standard pronunciation and a fresh, approachable, professional delivery. | neutral, sad |
| `灵言澈` | male | Warm, resonant and reassuring, with gentle, affectionate concern. | neutral, happy |
| `灵嘉肴` | male | Bright and energetic, with precise articulation and rational, approachable confidence. | neutral, fearful |
| `灵孤影` | male | Warm and resonant, with composed, sincere appreciation and natural satisfaction. | neutral |
| `灵雨汐` | female | Soft and sweet, with natural breathiness and an affectionate, relaxed youthful delivery. | neutral, happy, sad, fearful, surprised |
| `灵信远` | male | Clear and warm, with a slight smile and audible breath, balancing youthful freshness and mature affection. | happy, sad, angry, fearful, surprised |
| `灵思齐` | male | Sunny, bright and engaging, with sincere praise and approachable warmth. | happy, sad, fearful, disgusted, surprised |
| `灵叙柔` | female | Composed and dignified, with clear articulation and a balance of authority and gentle narration. | happy, surprised |
| `灵绮木` | female | Professional yet approachable, with gentle, assured delivery. | happy, angry, disgusted |
| `灵渊霆` | male | Youthful, natural and relaxed, like a familiar friend making a casual invitation. | happy, sad, angry, fearful, disgusted, surprised |
| `灵素心` | female | Sweet, crisp and energetic, with lively, curious intonation. | sad, disgusted |
| `灵采薇` | female | Warm and thoughtful, with clear articulation and calm, trustworthy delivery. | happy, angry, disgusted |
| `灵卓然` | male | Full and resonant, with precise pronunciation and calm, objective delivery. | sad, fearful |
| `灵浅忧` | male | Bright and crisp, with precise articulation and professional, objective confidence. | happy, sad, fearful, disgusted |
| `灵昭野` | male | Clear and youthful, with sincere admiration and a warm, captivated delivery. | happy, sad, disgusted |
| `灵心澜` | female | Mature and composed, with assured, discerning and dignified delivery. | happy, sad |
| `灵犀言` | male | Deep and full, with mature composure, firm articulation and an authoritative delivery. | sad, angry |
| `灵语冰` | female | Thoughtful and efficient, with precise articulation and objective, trustworthy delivery. | happy, sad, surprised |
| `灵惊鸿` | male | Brisk and direct, with natural, easygoing business confidence. | happy, sad, angry, disgusted, surprised |
| `灵惜蕊` | female | Clear and pleasant, with standard pronunciation and patient, professional warmth. | happy, sad, angry, fearful, disgusted, surprised |
| `灵语萱` | female | Clear and sweet, with gentle yet firm affection, trust and reassurance. | happy, sad, angry, fearful, disgusted |
| `灵语烟` | female | Bright, crisp and lively, expressing surprise and everyday banter. | happy, sad, angry, fearful, disgusted |
| `灵律雅` | female | Efficient and grounded in urban life, with restrained, authentic frustration. | happy, angry, fearful |
| `灵惊涛` | male | Young male delivery with a distinctive Nanning accent, conveying humorous frustration at an overly cold internet cafe. | fearful |
| `灵寒陌` | male | Clear and youthful, with urgent, innocent attempts to clear up a misunderstanding. | happy, disgusted |
| `灵决冥` | male | Resonant and precise, with calm, objective and engaging documentary-style delivery. | angry, fearful, disgusted |
| `灵慈萱` | female | Crisp, youthful and natural, with a familiar, slightly urgent delivery. | sad, angry, fearful |
| `灵妙音` | female | Resonant and forceful, with efficient delivery and an unquestionable sense of authority. | happy, sad, fearful, disgusted, surprised |
| `灵断尘` | male | Youthful and conversational, with light questions and decisive refusals. | happy, sad, fearful, disgusted, surprised |
| `灵熙婉` | female | Sweet and clear, with a smiling, approachable and lively storytelling style. | happy, sad, fearful |
| `灵旭扬` | male | Warm and resonant, with affectionate delivery and a hint of illness-related weakness and desire for attention. | happy, sad, angry, fearful, disgusted, surprised |
| `灵阅川` | male | Deep and resonant, with experienced, composed and dignified storytelling. | happy, sad, angry, fearful, disgusted |
| `灵妩月` | female | Gentle and thoughtful, combining caring, rational advice with quiet strength. | happy, sad, angry, fearful, disgusted, surprised |
| `灵汐月` | female | Sweet and soothing, with relaxed, heartfelt delivery conveying intimacy and trust. | happy, sad |
| `灵黛嗔` | female | Warm, thoughtful and engaging, with sincere, confident recommendations. | happy, disgusted |

The model also knows eight voices named after film, television and cartoon
characters. Izwi does not list them; a request can still name any voice the
model knows.

Example:

```bash
izwi tts "所有测试都顺利通过了，我们终于可以把这个好消息告诉大家！" \
  --model Ming-3.0-tts-0.5B \
  --speaker 灵风信 \
  --emotion happy \
  --output ming-happy.wav
```

To speak in a voice of your own instead, clone one from a reference clip (see
[Voice Cloning](/features/voice-cloning)), or describe one with
`--instructions`.
