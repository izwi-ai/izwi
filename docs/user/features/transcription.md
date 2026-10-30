---
title: "Transcription"
description: "Convert audio to text with Izwi through the CLI, web UI, and local API."
icon: "mic"
---
Convert audio to text with high accuracy using automatic speech recognition (ASR).

---

## Overview

Izwi's transcription feature converts spoken audio into written text. Capabilities include:

- **High accuracy** — State-of-the-art speech recognition
- **Multiple formats** — Support for WAV, MP3, M4A, FLAC, and more
- **Language detection** — Automatic language identification
- **Timestamps** — Optional word-level timing through a forced aligner, or from the model itself (Nemotron 3.5 Transcribe)
- **Summaries** — Optional AI summaries for persisted speech-text jobs
- **Local processing** — Complete privacy, no cloud

---

## Getting Started

### Download an ASR Model

```bash
izwi pull Parakeet-TDT-0.6B-v3
```

### Transcribe Audio

```bash
izwi transcribe audio.wav
```

---

## Using the CLI

### Basic Usage

```bash
izwi transcribe <audio-file>
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model`, `-m` | ASR model to use | `parakeet-tdt-0.6b-v3` |
| `--language`, `-l` | Language hint | auto-detect |
| `--translate-to` | Translate into this language (Granite Speech 5.0, Nemotron 3.5 Transcribe) | — |
| `--keyword` | Bias recognition toward a term; repeatable (Granite Speech 5.0, Nemotron 3.5 Transcribe) | — |
| `--format`, `-f` | Output format | `text` |
| `--output`, `-o` | Output file | stdout |
| `--word-timestamps` | Include word timing | — |
| `--speaker-labels` | Tag speakers and return the speaker turns (Nemotron 3.5 Transcribe); implies `verbose_json` | — |

The CLI uploads the file as multipart form data. The server accepts up to 25 MiB per file by default; set `IZWI_OPENAI_AUDIO_UPLOAD_LIMIT_MB` on the server to raise it.

### Examples

**Basic transcription:**

```bash
izwi transcribe meeting.wav
```

**Save to file:**

```bash
izwi transcribe meeting.wav --output transcript.txt
```

**JSON output with metadata:**

```bash
izwi transcribe meeting.wav --format json --output transcript.json
```

**With word timestamps:**

```bash
izwi transcribe meeting.wav --format verbose-json --word-timestamps
```

**Specify language:**

```bash
izwi transcribe audio.wav --language en
izwi transcribe audio.wav --language es
```

---

## Using the Web UI

1. Navigate to **Transcription** in the sidebar
2. Choose a mode
3. Upload an audio file or record directly
4. Select the required model
5. Click **Transcribe** or submit the selected workflow
6. View, copy, summarize, or download the result

The Transcription workspace has three modes:

| Mode | Best for | Notes |
|------|----------|-------|
| **Transcription** | Single-speaker or general ASR | Streaming is enabled by default. Word timestamps disable streaming and require a ready `Qwen3-ForcedAligner-0.6B` aligner, except with `Nemotron-3.5-Transcribe-30B-A4B-FP8`, which times words itself. With `Granite-Speech-5.0-2B`, its GGUF release or `Nemotron-3.5-Transcribe-30B-A4B-FP8`, **Translate to** and **Keywords** appear; a translation has no timestamps. |
| **Speaker Attributed ASR** | Speaker-turn transcripts written by the ASR model | Requires `Granite-Speech-4.1-2B-Plus` (the default) or `Nemotron-3.5-Transcribe-30B-A4B-FP8`, accepts speaker expectation hints, and disables streaming/timestamps. |
| **Diarization** | Speaker-separated timelines with start/end times | Uses the diarization pipeline and optional ASR/aligner models, or `Nemotron-3.5-Transcribe-30B-A4B-FP8` alone (**Single model**). |

### Features

- **Drag and drop** — Upload files easily
- **Record** — Transcribe directly from microphone
- **Mode switch** — Choose Transcription, Speaker Attributed ASR, or Diarization
- **Summaries** — Generate or regenerate AI summaries for saved records
- **Copy** — One-click copy to clipboard
- **Download** — Save as text or JSON

---

## Using the API

### Endpoint

```
POST /v1/audio/transcriptions
```

### Request (multipart/form-data)

| Field | Type | Description |
|-------|------|-------------|
| `file` | File | Audio file to transcribe |
| `model` | String | Model name |
| `language` | String | Language code (optional) |
| `response_format` | String | `text`, `json`, `verbose_json`, `srt`, or `vtt` |
| `stream` | Boolean/String | Enable SSE transcript events (`true`, `1`, `yes`, or `on`) |

### Example (curl)

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=Parakeet-TDT-0.6B-v3" \
  -F "response_format=json"
```

### Response (JSON)

```json
{
  "text": "Hello, this is a transcription test."
}
```

### Response (verbose_json)

```json
{
  "text": "Hello, this is a transcription test.",
  "language": "en",
  "duration": 3.5,
  "processing_time_ms": 812.4,
  "rtf": 0.23,
  "izwi_asr_diagnostics": null
}
```

Streaming responses emit SSE payloads with `type` values such as
`transcript.text.delta`, `transcript.text.done`, and `error`.

See the [API Reference](/api#audio-transcriptions) for JSON input,
streaming events, upload limits, and exact response shapes.

### Persisted Speech-Text Jobs

The web UI uses `/v1/speech-to-text/jobs` for saved transcription records,
Speaker Attributed ASR records, and diarization jobs:

```bash
curl -X POST "http://localhost:8080/v1/speech-to-text/jobs?job_kind=transcription" \
  -F "file=@meeting.wav" \
  -F "model_id=Parakeet-TDT-0.6B-v3" \
  -F "generate_summary=true"
```

Use `job_kind=speaker_attributed_asr` or `job_kind=saa` for speaker-turn
transcripts (Granite Speech 4.1 Plus or Nemotron 3.5 Transcribe), and
`job_kind=diarization` for speaker timelines.

---

## Nemotron 3.5 Transcribe

`Nemotron-3.5-Transcribe-30B-A4B-FP8` is NVIDIA's 30B speech language model.
It transcribes a recording of up to 60 minutes in one pass, translates,
takes context terms, labels speakers and times words itself.

It needs about 31 GiB of memory on Metal and CUDA and about 33 GiB on CPU;
a machine that cannot hold it is refused when the model loads.

```bash
izwi pull Nemotron-3.5-Transcribe-30B-A4B-FP8
izwi transcribe meeting.mp3 --model Nemotron-3.5-Transcribe-30B-A4B-FP8
```

### Languages and wording

It transcribes 40 languages: Arabic, Bengali, Bulgarian, Chinese, Croatian,
Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Greek,
Hebrew, Hindi, Hungarian, Indonesian, Italian, Japanese, Korean, Latvian,
Lithuanian, Maltese, Marathi, Polish, Portuguese, Romanian, Russian, Slovak,
Slovenian, Spanish, Swedish, Tagalog, Tamil, Telugu, Thai, Turkish, Ukrainian
and Vietnamese. Name the language with a code (`de`, `pt-BR`) or an English
name, or leave it out (or send `auto`) to let the model detect it. Any other
language, Norwegian included, returns `400`.

By default the transcript is the model card's "readable text": punctuation and
capitalization, with fillers and accidental repetitions left out. A `prompt`
(`--prompt`) replaces that instruction entirely, after tags such as
`<spk:0>` are removed (at most 4,000 characters). Because it replaces the
whole instruction, a prompt cannot be combined with translation, keywords
or speaker labels; such a request returns `400`.

### Translation

`target_language` on the API, `--translate-to` in the CLI, or **Translate
to** in the UI translates the speech into another language. Translation runs
to or from English only. Every language above translates except Bengali,
Indonesian, Marathi, Tagalog, Tamil, Telugu, Turkish and Vietnamese. A
translation has no word timestamps, and it cannot be combined with keywords or
speaker labels (the request returns `400`).

### Context terms

`keywords` on the API, `--keyword` in the CLI, or **Keywords** in the UI lists
names and vocabulary the speech contains. The model receives them as the
card's "Context terms" instruction. Up to 64 terms of at most 80 characters
each; tags in a term are removed.

### Speaker labels

The model tags speakers itself. On `/v1/audio/transcriptions`, set
`speaker_labels=true` with `response_format=verbose_json` (CLI:
`--speaker-labels`). The `text` field stays plain; the speaker turns come back
in `speaker_turns`:

```json
{
  "text": "Good morning. Morning, shall we start?",
  "speaker_turns": [
    { "speaker": 0, "text": "Good morning.", "start": 0.32, "end": 1.1 },
    { "speaker": 1, "text": "Morning, shall we start?", "start": 1.6, "end": 3.04 }
  ]
}
```

`speaker` is the model's tag number, or `null` for text before the first tag.
`start` and `end` are in seconds and appear only when the model timed the
words. Speaker labels work in Arabic, Chinese, English, French, German, Hindi,
Italian, Japanese, Korean, Portuguese, Russian, Spanish and Thai; naming
another language returns `400`.

In the web UI, choose **Speaker Attributed ASR** and select this model. Its
turns are labelled "Speaker 1", "Speaker 2" and so on, and timed when its word
timestamps align. The speaker expectation does not steer the model: a count
outside it only adds a warning.

For "who spoke when" segments, the model can also diarize on its own: name it
as the `model` of a diarization job (see
[Single-Model Diarization](/features/diarization#single-model-diarization)).
It cannot be the ASR model of a Sortformer or Nemotron-3-Diarization pipeline.

### Word timestamps

The model times words with its own CTC head, so no forced aligner is needed:
`timestamp_granularities[]=word` on the API, `--word-timestamps` in the CLI,
or **Include timestamps** in the UI (no aligner model is asked for). If the
model cannot align a transcription job's words, the server times them with the
forced aligner instead. Thai is split into words at spaces only, so Thai is
timed in larger pieces than NVIDIA's own tools, which segment Thai into words.

### Long recordings

The whole recording runs in one pass, up to 60 minutes; Izwi does not split it.
Uploads keep their usual limits: 64 MiB for transcription jobs (the web UI),
and 25 MiB by default on `/v1/audio/transcriptions` and in the CLI
(`IZWI_OPENAI_AUDIO_UPLOAD_LIMIT_MB` raises it). A one-hour WAV at 16 kHz,
mono, 16-bit is about 115 MB and is refused with `413`. Send long recordings
compressed:

- An hour of mono MP3 at 48 kbit/s is about 22 MB and fits both limits. At
  64 kbit/s (about 29 MB) it fits the job limit but not the default 25 MiB.
- FLAC is lossless but compresses speech much less; a full hour may still be
  close to 64 MiB.

The CLI waits at most 5 minutes for the transcript. For long recordings on a
slower machine, create a transcription job in the web UI or with
`/v1/speech-to-text/jobs`.

`izwi_asr_diagnostics.finish_reason` says why decoding stopped: `stop` (the
model finished), `length` (the token budget ran out) or `repetition` (the
model repeated itself and was stopped).

### Not supported

- Live transcription and voice mode: the model is offline only.
- The ASR stage of a diarization pipeline. It diarizes on its own instead
  (see [Speaker labels](#speaker-labels)).

---

## Supported Audio Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| WAV | `.wav` | Best quality, recommended |
| MP3 | `.mp3` | Widely compatible |
| M4A | `.m4a` | Apple format |
| FLAC | `.flac` | Lossless |
| OGG | `.ogg` | Open format |
| WebM | `.webm` | Web recordings |

---

## Available Models

| Model | Size | Accuracy | Speed |
|-------|------|----------|-------|
| `Parakeet-TDT-0.6B-v3` | 2.5 GB | Strong baseline | Medium |
| `Whisper-Large-v3-Turbo` | 1.5 GB | Strong multilingual baseline | Medium |
| `Qwen3-ASR-0.6B-GGUF` | 1.0 GB | Good | Fast |
| `Qwen3-ASR-1.7B-GGUF` | 2.5 GB | Better | Medium |
| `VibeVoice-ASR` | 16.2 GB | Long-form Microsoft ASR checkpoint | Medium |
| `Nemotron-3.5-ASR-Streaming-0.6B` | 2.37 GB | 40-locale NVIDIA FastConformer-RNNT; native offline transcription with prompt-conditioned language control | Medium |
| `Granite-Speech-4.1-2B` | 4.6 GB | IBM ASR with punctuated, capitalized transcripts | Medium |
| `Granite-Speech-4.1-2B-Plus` | 4.2 GB | IBM rich transcription model with speaker-attributed output and native word timestamps; unpunctuated | Medium |
| `Granite-Speech-5.0-2B` | 4.6 GB | IBM ASR with punctuated transcripts, speech translation and keyword biasing | Medium |
| `Granite-Speech-5.0-2B-GGUF` | 3.1 GB | Granite Speech 5.0 with a Q8_0 language model; the faster choice, above all on CPU | Fast |
| `Nemotron-3.5-Transcribe-30B-A4B-FP8` | 34.7 GB | NVIDIA's 30B speech language model: up to 60 minutes in one pass, 40 languages, translation, context terms, speaker labels and its own word timestamps; needs about 32 GiB of memory | Not yet measured |
| `LFM2.5-Audio-1.5B-GGUF` | 1.2 GB | Good integrated speech model | Medium |
| `LFM3-Audio-2B-GGUF` | 1.46 GB | Integrated speech model in 18 languages; no word timestamps of its own | Not yet measured |
| `Voxtral-Mini-4B-Realtime-2602` | 8 GB | Rust/Candle offline transcription; realtime planned | Medium |

Use larger models for:
- Noisy audio
- Accented speech
- Technical vocabulary

---

## Output Formats

### Text

Plain text transcript:

```
Hello, this is a transcription test.
```

### JSON

```json
{
  "text": "Hello, this is a transcription test."
}
```

### Verbose JSON

Includes language, duration, processing-time, realtime-factor, and optional
runtime diagnostics. Add `timestamp_granularities[]=word` (CLI
`--word-timestamps`) to get a `words` array.

```json
{
  "text": "Hello, this is a transcription test.",
  "language": "en",
  "duration": 3.5,
  "processing_time_ms": 812.4,
  "rtf": 0.23,
  "izwi_asr_diagnostics": null
}
```

---

## Tips for Best Results

1. **Use quality audio** — Clear recordings transcribe better
2. **Minimize noise** — Background noise reduces accuracy
3. **Proper format** — WAV files work best
4. **Right model size** — Larger models for difficult audio
5. **Language hints** — Specify language if known

---

## See Also

- [Diarization](/features/diarization) — Identify multiple speakers
- [Speaker Attributed ASR](/features/speaker-attributed-asr) — Speaker-turn transcripts
- [Voice Mode](/features/voice) — Real-time transcription
- [CLI Reference](/cli) — Full command documentation
