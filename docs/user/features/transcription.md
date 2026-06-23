---
title: "Transcription"
description: "Convert audio to text with Izwi through the CLI, web UI, and local API."
icon: "mic"
---
# Transcription

Convert audio to text with high accuracy using automatic speech recognition (ASR).

---

## Overview

Izwi's transcription feature converts spoken audio into written text. Capabilities include:

- **High accuracy** — State-of-the-art speech recognition
- **Multiple formats** — Support for WAV, MP3, M4A, FLAC, and more
- **Language detection** — Automatic language identification
- **Timestamps** — Optional word-level timing through a forced aligner
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
| `--format`, `-f` | Output format | `text` |
| `--output`, `-o` | Output file | stdout |
| `--word-timestamps` | Include word timing | — |

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
izwi transcribe meeting.wav --format verbose_json --word-timestamps
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
| **Transcription** | Single-speaker or general ASR | Streaming is enabled by default. Word timestamps require a ready `Qwen3-ForcedAligner-0.6B` aligner and disable streaming. |
| **Speaker Attributed ASR** | Granite Speech speaker-turn transcripts | Requires `Granite-Speech-4.1-2B-Plus`, accepts speaker expectation hints, and disables streaming/timestamps. |
| **Diarization** | Speaker-separated timelines with start/end times | Uses the diarization pipeline and optional ASR/aligner models. |

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

Use `job_kind=speaker_attributed_asr` or `job_kind=saa` for Granite speaker-turn
transcripts, and `job_kind=diarization` for speaker timelines.

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
| `Parakeet-TDT-0.6B-v3` | 9.4 GB | Strong baseline | Medium |
| `Whisper-Large-v3-Turbo` | 1.5 GB | Strong multilingual baseline | Medium |
| `Qwen3-ASR-0.6B-GGUF` | 1.0 GB | Good | Fast |
| `Qwen3-ASR-1.7B-GGUF` | 2.5 GB | Better | Medium |
| `VibeVoice-ASR` | 16.2 GB | Long-form Microsoft ASR checkpoint | Medium |
| `Nemotron-3.5-ASR-Streaming-0.6B` | 2.37 GB | 40-locale NVIDIA FastConformer-RNNT; native offline transcription with prompt-conditioned language control | Medium |
| `Granite-Speech-4.1-2B-Plus` | 4.2 GB | IBM rich transcription model with prompt guidance, speaker-attributed output, and word timestamps | Medium |
| `LFM2.5-Audio-1.5B-GGUF` | 1.2 GB | Good integrated speech model | Medium |
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
runtime diagnostics. Word-level timestamps are not currently returned by this
endpoint.

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
- [Speaker Attributed ASR](/features/speaker-attributed-asr) — Granite speaker-turn transcripts
- [Voice Mode](/features/voice) — Real-time transcription
- [CLI Reference](/cli) — Full command documentation
