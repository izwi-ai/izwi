---
title: "Text-to-Speech"
description: "Generate natural speech from text with Izwi models, voices, streaming, and audio output formats."
sidebarTitle: "Text-to-Speech"
icon: "volume-2"
---
Generate natural, human-like speech from text using state-of-the-art TTS models.

---

## Overview

Izwi's text-to-speech converts written text into spoken audio. Features include:

- **Natural voices** — High-quality, expressive speech
- **Local audio output** — WAV for files and raw PCM for low-level API clients
- **Speed control** — Adjust playback speed
- **Streaming** — Real-time audio generation
- **Local processing** — No cloud, complete privacy

---

## Getting Started

### Download a TTS Model

```bash
izwi pull Kokoro-82M
```

### Kokoro-82M Prerequisite (`espeak-ng`)

`Kokoro-82M` is the default TTS model. Install `espeak-ng` on your system first.
Izwi uses it for Kokoro phonemization and will return an error if it is missing.

- macOS: see [macOS Installation](/installation/macos#optional-install-espeak-ng-for-kokoro-82m)
- Linux: see [Linux Installation](/installation/linux#optional-install-espeak-ng-for-kokoro-82m)
- Windows: see [Windows Installation](/installation/windows#optional-install-espeak-ng-for-kokoro-82m)

### Generate Speech

**Command line:**

```bash
izwi tts "Hello, welcome to Izwi!" --output hello.wav
```

**Play after generation:**

```bash
izwi tts "Hello, welcome to Izwi!" --play
```

Izwi saves the file, then plays it with the system player.

---

## Using the CLI

### Basic Usage

```bash
izwi tts "Your text here" --output output.wav
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model`, `-m` | TTS model to use | `Kokoro-82M` |
| `--output`, `-o` | Output file path | `izwi_output_<timestamp>.wav` in the current directory |
| `--format`, `-f` | Audio format: `wav` or `pcm` (raw 16-bit little-endian mono, no header) | `wav` |
| `--speed`, `-r` | Speech speed multiplier (models with speed control) | Model default |
| `--speaker`, `-s` | Voice/speaker ID | Model default (Kokoro: `af_heart`) |
| `--saved-voice-id` | Reuse a saved reference voice | — |
| `--reference-audio` | Reference audio file for cloning | — |
| `--reference-text` | Transcript for the reference audio (needs `--reference-audio`; optional for Chatterbox) | — |
| `--reference-text-file` | File containing the reference transcript | — |
| `--instructions`, `--voice-description` | Voice-design direction prompt; for `Maya1-3B`, the voice to speak with | — |
| `--temperature`, `-t` | Sampling temperature | Model default |
| `--language` | Language of the text (`Chatterbox-Multilingual-500M`: a code such as `fr` or a name; `Maya2-Native-2.5B`: a code such as `hi`, or `auto`; `LFM3-Audio-2B-GGUF`: one of its 19 locales such as `en-GB` or `ja`; a hint for Qwen3-TTS) | Model default |
| `--reference-language` | Language of the reference audio (`Maya2-Native-2.5B` only; read from the transcript when unset) | — |
| `--exaggeration` | Emotion intensity, 0 to 2 (Chatterbox only) | `0.5` |
| `--cfg-weight` | Classifier-free guidance weight, 0 to 1; `0` turns guidance off (Chatterbox only) | `0.5` |
| `--min-p` | Min-p sampling threshold, 0 to 1 (Chatterbox only) | `0.05` |
| `--play`, `-p` | Play the saved file with the system player (WAV only) | — |
| `--stream` | Stream the audio as it is generated into a WAV file and report time to first audio | — |

### Examples

**WAV output:**

```bash
izwi tts "Hello world" --format wav --output hello.wav
```

**Adjust speed:**

```bash
# Slower (0.5x - 1.0x)
izwi tts "Speaking slowly" --speed 0.75 --output slow.wav

# Faster (1.0x - 2.0x)
izwi tts "Speaking quickly" --speed 1.5 --output fast.wav
```

**Read from stdin:**

```bash
echo "Text from pipe" | izwi tts - --output piped.wav
cat article.txt | izwi tts - --output article.wav
```

**Streaming output:**

```bash
izwi tts "Long text for streaming" --stream --play
```

**Clone from reference audio:**

```bash
izwi tts "This line uses the reference voice." \
  --model Qwen3-TTS-12Hz-0.6B-Base \
  --reference-audio samples/reference.wav \
  --reference-text "This is the text spoken in the reference sample." \
  --output cloned.wav
```

**Clone with Chatterbox (no transcript), more expressive:**

```bash
izwi tts "This line uses the reference voice with more emotion." \
  --model Chatterbox-500M \
  --reference-audio samples/reference.wav \
  --exaggeration 0.7 \
  --cfg-weight 0.3 \
  --output expressive.wav
```

For expressive or fast-talking reference voices, Resemble AI suggests a lower
`--cfg-weight` (about 0.3) with `--exaggeration` around 0.7 or higher.

**Reuse a saved voice:**

```bash
izwi tts "This line uses a saved voice." \
  --model Qwen3-TTS-12Hz-0.6B-Base \
  --saved-voice-id voice_abc123 \
  --output saved-voice.wav
```

**Design a prompted voice:**

```bash
izwi tts "This line uses a designed voice." \
  --model Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --instructions "A friendly audiobook narrator with warm pacing" \
  --output designed.wav
```

---

## Using the Web UI

1. Navigate to **Text to Speech** in the sidebar
2. Enter your text in the input field
3. Select a voice (if available)
4. Click **Generate**
5. Play or download the audio

With Chatterbox selected, the dialog shows sliders for exaggeration, CFG weight
and min-p. With Chatterbox Multilingual it also asks which language the text is
in. With Maya 2 Native it asks the same, with **Auto-detect** (the default) to
read the language from the text's script, and offers its inline tags.

With Fish Audio Drama 3 the saved voice is optional (**None (seeded voice)**)
and the dialog has a **Seed**, filled with a new number each time it opens;
keep it to get the same voice again. **Dialogue**, beside **Single voice**,
edits a scene:

- **Speakers**: each speaks in a saved voice that has a transcript, or in a
  voice the seed sets. Up to 16 speakers and 8 saved voices.
- **Lines**: each has its speaker, its text and **Insert a tag** for
  performance tags. **Preview** speaks one line without saving it.
- **Generate scene** speaks every line as one take.

A Drama 3 take's page has **Edit this take**: the dialog opens with the take's
text (its lines, for a dialogue), its voices and its seed. Change the line and
generate: the new take continues from the old one, so the model says the new
text in the context of what it already said. The new take's page links back
to it under **Edited from**. A dialogue take's **⋯** menu has **Open in
dialogue editor**, which brings its lines back as a new scene with the same
voices and seed.

Saved voices and the seed come back. A voice given as a reference clip does
not, because a take keeps no clip audio: its speaker starts in a seeded voice.
Takes made before take editing kept no voices or seed and cannot be edited;
their lines still open in the dialogue editor, every speaker in a seeded
voice.

### Features

- **Live preview** — Hear audio as it generates
- **Download** — Save audio files locally
- **History** — Access recent generations

---

## Using the API

### Endpoint

```
POST /v1/audio/speech
```

### Request

```json
{
  "model": "Kokoro-82M",
  "input": "Hello, this is a test.",
  "voice": "af_heart",
  "speed": 1.0,
  "response_format": "wav"
}
```

### Response

Binary audio data with appropriate `Content-Type` header.

Set `stream` to `true` or `stream_format` to `sse` to receive server-sent
audio events instead of one binary response. See the
[API Reference](/api#audio-speech) for streaming event shapes,
voice-cloning fields, saved voices, and model-specific controls.

### Example (curl)

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "Kokoro-82M", "input": "Hello world", "voice": "af_heart"}' \
  --output speech.wav
```

---

## Available Models

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| `Kokoro-82M` | ~0.4 GB | Good | Fast |
| `Qwen3-TTS-12Hz-0.6B-Base` | ~2.3 GB | Good | Fast |
| `Qwen3-TTS-12Hz-1.7B-Base` | ~4.2 GB | Better | Medium |
| `Qwen3-TTS-12Hz-0.6B-CustomVoice` | ~2.3 GB | Good | Fast |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | ~4.2 GB | Better | Medium |
| `VibeVoice-1.5B` | ~5.0 GB | Better long-form/reference voice | Medium |
| `Voxtral-4B-TTS-2603` | ~8.1 GB | Better | Medium |
| `Chatterbox-500M` | ~3.2 GB | English reference voice, emotion control | Slow |
| `Chatterbox-Multilingual-500M` | ~3.3 GB | 22 languages, reference voice, emotion control | Slow |
| `FishAudio-Drama-3` | ~41.5 GB | Seeded or reference voice, dialogue of up to 16 speakers, performance tags, 44.1 kHz | Slow; needs about 40 GB of memory |

For reference-audio cloning, use **Base** variants, `VibeVoice-1.5B`, `FishAudio-S2-Pro`, `FishAudio-Drama-3`, `Chatterbox-500M`, `Chatterbox-Multilingual-500M` or `Maya2-Native-2.5B`.  
For built-in voice presets, use `Kokoro-82M`, **CustomVoice** variants, `Voxtral-4B-TTS-2603`, `LFM2.5-Audio-1.5B-GGUF`, `LFM3-Audio-2B-GGUF` or `Maya2-Native-2.5B`.  
For prompt-based voice design, use **VoiceDesign** variants.
`Kokoro-82M` requires `espeak-ng` to be installed separately.
`Voxtral-4B-TTS-2603` supports 20 preset voices and emits 24 kHz audio. Its
model and bundled voice assets inherit a CC BY-NC 4.0 license.
`Chatterbox-500M` (English, 24 kHz, MIT) speaks with its built-in `default`
voice or clones from reference audio alone. It streams (first audio in about
2 s on an Apple M1 Pro with Metal), has no speed control or voice design, and
runs slower than real time (a real-time factor of about 2.2 for one request on
that Mac, where its T3 model runs in F16 and the model takes about 2.6 GB;
set `IZWI_METAL_DTYPE=f32` to keep it in F32), so streamed audio can pause
between windows. Long text is split between
sentences and spoken in full. Every utterance carries Resemble AI's imperceptible PerTh watermark;
operators can turn it off with `IZWI_CHATTERBOX_WATERMARK=false`.
`Chatterbox-Multilingual-500M` does the same in Arabic, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Italian, Korean, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swahili, Swedish and Turkish, named per
request with `language` (English when unset). Japanese is not supported yet.
`Maya1-3B` (English, 24 kHz, Apache-2.0) speaks with a built-in voice or one
described in `instructions`, and performs tags such as `<laugh>` placed in the
text. On an Apple M1 Pro with Metal it streams first audio in about 2 s and
runs at a real-time factor of about 4.5 for one request (about 3 with
`Maya1-3B-GGUF`, which also needs half the memory). A request's audio is the
same alone, streamed or beside other requests for the same `seed`.
`LFM3-Audio-2B-GGUF` (24 kHz, LFM Open License v1.0) speaks with a `female`
or `male` voice in one of 19 locales named with `language` (American English
when unset). It streams, splits long text into sentences automatically, and
has no voice cloning, voice design or speed control. The same model also
transcribes speech and answers in voice chat.
`FishAudio-Drama-3` (44.1 kHz, Fish Audio Community License v1.0) speaks
without any voice input, or clones a reference voice; see
[Fish Audio Drama 3](#fish-audio-drama-3) below.
For built-in speaker IDs, see [Voice Presets](/models/voice-presets).

---

## Fish Audio Drama 3

`FishAudio-Drama-3` needs no voice input. Without a reference voice, the
sampling `seed` sets the voice: the same `seed` gives the same voice, and a
request without one gets a different voice each time. With reference audio and
its transcript, or a saved voice that has one, it speaks in that voice (see
[Voice Cloning](/features/voice-cloning)). It reads the language from the text;
a `[language: xx]` tag in the text steers it.

Tags in the text direct the performance where they stand: `[sighs]`,
`[laughs]`, `[pause]`, `[gasps]`, `[clears throat]`, `[hesitates]`, free-text
tags such as `[with quiet excitement]`, and `<whisper>…</whisper>`. They reach
the model as text.

```bash
izwi tts "[sighs] It has been a long day. [pause] But we made it, and <whisper>nobody noticed</whisper>." \
  --model FishAudio-Drama-3 \
  --output drama3.wav
```

Long text is spoken as a chunk session. The model's own chunker cuts the text
into chunks of about 300 bytes, at sentence ends where it can, and each chunk
is its own generation. Each chunk's prompt holds the references and the first
and latest chunks with the audio already generated for them, so the voices
carry through the whole text. Chunks play one after another with no added
pause. With `--stream` (or `stream: true` on the API), audio arrives as it is
generated: first after 4 frames, then every 16 frames.

### Dialogue

One request can voice up to 16 speakers. Mark each turn with a speaker tag,
`<|speaker:0|>` to `<|speaker:15|>`, or send a script of `{speaker, text}`
lines. Give each speaker a reference voice (a clip with its transcript, or a
saved voice), up to 8 references in all; a speaker without one speaks in a
voice the seed sets. The chunker keeps each turn whole with its speaker.

```bash
izwi tts --script scene.jsonl \
  --model FishAudio-Drama-3 \
  --speaker-reference 0=voices/anna.wav:voices/anna.txt \
  --speaker-voice 1=voice_ben \
  --seed 7 \
  --output scene.wav
```

On the API, the same request is `references` and `script` on
`/v1/audio/speech` or the text-to-speech records; see
[Drama 3 dialogue](/api#drama-3-dialogue).

### Take editing

A take can continue from earlier takes: to say a line again, or the next
line, in the context of what was already said. Its prompt holds the earlier
takes as finished turns, their text and the audio generated for it, before the
new text; that is how the model was trained to edit takes. Each Drama 3 take
keeps the audio codes of its first and last chunks, its voices (saved voices
by id) and its seed, so a later take can name it.

On the text-to-speech records, send `previous_take_ids` with up to 2 earlier
takes, oldest first, beside the new text, voices and seed. Voices and seed are
not inherited: send the ones you want, for example those the earlier take's
record reports. Takes of another model, unfinished or failed takes, and takes
made before take editing are refused. See
[Drama 3 take editing](/api#drama-3-take-editing). In the Web UI, **Edit this
take** does this for you.

Its license allows commercial use only below USD 500,000 in annual revenue,
and offering the model to third parties as a hosted speech service needs a
license from Fish Audio; see [Models](/models).

---

## Audio Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| WAV | `.wav` | Uncompressed, highest quality |
| PCM | `.pcm` | Raw PCM for low-level playback pipelines |
| MP3, OPUS, OGG, FLAC, AAC | Matching extension | Recognized request names. The OSS server does not bundle compressed encoders yet, so API clients must set `allow_format_fallback: true` if they intentionally want WAV bytes returned for these names. |

---

## Tips

1. **Punctuation matters** — Use proper punctuation for natural pauses
2. **Break long text** — Split very long text into paragraphs
3. **Test different speeds** — Find the right pace for your use case
4. **Use appropriate models** — Larger models = better quality but slower

---

## See Also

- [Voice Cloning](/features/voice-cloning) — Clone custom voices
- [Voice Design](/features/voice-design) — Create voices from descriptions
- [Voices](/features/voices) — Manage saved and built-in voices
- [CLI Reference](/cli) — Full command documentation
