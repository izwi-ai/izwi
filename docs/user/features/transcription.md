# Transcription

Convert audio to text with high accuracy using automatic speech recognition (ASR).

---

## Overview

Izwi's transcription feature converts spoken audio into written text. Capabilities include:

- **High accuracy** — State-of-the-art speech recognition
- **Multiple formats** — Support for WAV, MP3, M4A, FLAC, and more
- **Language detection** — Automatic language identification
- **Timestamps** — Optional word-level timing
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
2. Upload an audio file or record directly
3. Select the ASR model
4. Click **Transcribe**
5. View, copy, or download the transcript

> The Transcription workspace now also hosts diarization workflows. Use the mode switch in `/transcription` to open speaker-separated runs.

### Features

- **Drag and drop** — Upload files easily
- **Record** — Transcribe directly from microphone
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
| `response_format` | String | `text`, `json`, or `verbose_json` |

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
  "text": "Hello, this is a transcription test.",
  "language": "en",
  "duration": 3.5
}
```

### Response (verbose_json)

```json
{
  "text": "Hello, this is a transcription test.",
  "language": "en",
  "duration": 3.5,
  "words": [
    {"word": "Hello", "start": 0.0, "end": 0.5},
    {"word": "this", "start": 0.6, "end": 0.8},
    ...
  ]
}
```

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
| `LFM2.5-Audio-1.5B-GGUF` | 1.2 GB | Good integrated speech model | Medium |

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

Includes word-level timestamps and metadata:

```json
{
  "text": "Hello, this is a transcription test.",
  "language": "en",
  "duration": 3.5,
  "words": [
    {"word": "Hello", "start": 0.0, "end": 0.5},
    {"word": "this", "start": 0.6, "end": 0.8}
  ]
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

- [Diarization](./diarization.md) — Identify multiple speakers
- [Voice Mode](./voice.md) — Real-time transcription
- [CLI Reference](../cli/index.md) — Full command documentation
