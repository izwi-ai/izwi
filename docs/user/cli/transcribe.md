---
title: "izwi transcribe"
description: "Transcribe audio files from the command line with output formats and ASR model options."
icon: "mic"
---
Convert audio to text.

---

## Synopsis

```bash
izwi transcribe <FILE> [OPTIONS]
```

---

## Description

Transcribes audio files to text using automatic speech recognition (ASR). Supports multiple audio formats and output options.

---

## Arguments

| Argument | Description |
|----------|-------------|
| `<FILE>` | Audio file to transcribe |

---

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model <MODEL>` | ASR model to use | `parakeet-tdt-0.6b-v3` |
| `-l, --language <LANG>` | Language hint (e.g., `en`, `es`) | Auto-detect |
| `--prompt <TEXT>` | Initial ASR prompt or keyword guidance | — |
| `--translate-to <LANGUAGE>` | Translate the speech into this language instead of transcribing it (models with speech translation, such as `Granite-Speech-5.0-2B`) | — |
| `--keyword <TERM>` | A term to bias recognition toward; repeat for more (models with keyword biasing, such as `Granite-Speech-5.0-2B`) | — |
| `--max-tokens <N>` | Maximum number of ASR decoder tokens to generate | Model default |
| `-f, --format <FORMAT>` | Output format: `text`, `json`, `verbose-json` | `text` |
| `-o, --output <PATH>` | Output file (default: stdout) | — |
| `--word-timestamps` | Request word-level timestamps; automatically uses `verbose_json` response format | — |

The CLI uploads the audio file as multipart form data. The server accepts up to 25 MiB per file by default; raise it with `IZWI_OPENAI_AUDIO_UPLOAD_LIMIT_MB` on the server.

---

## Examples

### Basic transcription

```bash
izwi transcribe audio.wav
```

### Save to file

```bash
izwi transcribe audio.wav --output transcript.txt
```

### JSON output

```bash
izwi transcribe audio.wav --format json
```

### Prompt-guided Granite Speech transcription

```bash
izwi transcribe meeting.wav \
  --model Granite-Speech-4.1-2B-Plus \
  --prompt "keywords: Izwi, Granite Speech" \
  --max-tokens 256 \
  --format verbose-json
```

### Granite Speech 5.0 translation and keyword biasing

```bash
# Translate German speech into English text
izwi transcribe interview.wav \
  --model Granite-Speech-5.0-2B \
  --language de \
  --translate-to en

# Bias recognition toward names and product terms
izwi transcribe standup.wav \
  --model Granite-Speech-5.0-2B \
  --keyword Izwi --keyword "Granite Speech" --keyword Kubernetes
```

`--translate-to` and `--keyword` need a model with speech translation or
keyword biasing (`Granite-Speech-5.0-2B` and its GGUF release); the server
rejects them for other models.

### LFM3 Audio transcription

```bash
izwi transcribe interview.wav \
  --model LFM3-Audio-2B-GGUF \
  --language fr
```

`LFM3-Audio-2B-GGUF` (and `LFM3-Audio-2B-Q8_0-GGUF`) transcribes 18
languages: `en`, `ar`, `da`, `de`, `el`, `es`, `fi`, `fr`, `he`, `it`, `ja`,
`ko`, `nl`, `pl`, `pt`, `sv`, `tr` and `zh`. Without `--language`, or with
`auto`, the model chooses; the server refuses any other code with `400`. It
ignores `--prompt`, has no word timestamps of its own, and transcribes long
audio in chunks of at most 30 seconds.

### Word timestamps

```bash
izwi transcribe audio.wav \
  --model Granite-Speech-4.1-2B-Plus \
  --word-timestamps
```

### With timestamps

```bash
izwi transcribe audio.wav --format verbose-json --word-timestamps
```

### Specify language

```bash
izwi transcribe audio.wav --language en
izwi transcribe audio.wav --language es
```

### Use larger model

```bash
izwi transcribe audio.wav --model Qwen3-ASR-1.7B-GGUF
```

### Use Voxtral

```bash
izwi transcribe audio.wav --model Voxtral-Mini-4B-Realtime-2602 --format verbose-json
```

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

`--word-timestamps` requests `verbose_json` output with a `words` array when the
selected model or forced aligner can provide word timing metadata.

---

## Supported Audio Formats

- WAV (`.wav`)
- MP3 (`.mp3`)
- M4A (`.m4a`)
- FLAC (`.flac`)
- OGG (`.ogg`)
- WebM (`.webm`)

---

## Models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `Parakeet-TDT-0.6B-v3` | 2.5 GB | Medium | Strong baseline (default) |
| `Whisper-Large-v3-Turbo` | 1.5 GB | Medium | Strong multilingual baseline |
| `Qwen3-ASR-0.6B-GGUF` | 1.0 GB | Fast | Good |
| `Qwen3-ASR-1.7B-GGUF` | 2.5 GB | Medium | Better |
| `VibeVoice-ASR` | 16.2 GB | Medium | Microsoft long-form ASR checkpoint |
| `Nemotron-3.5-ASR-Streaming-0.6B` | 2.37 GB | Medium | 40-locale NVIDIA FastConformer-RNNT; native offline transcription with prompt-conditioned language control |
| `Granite-Speech-4.1-2B` | 4.6 GB | Medium | IBM ASR with punctuated, capitalized transcripts |
| `Granite-Speech-4.1-2B-Plus` | 4.2 GB | Medium | IBM rich transcription model with speaker turns and native word timestamps; unpunctuated |
| `Granite-Speech-5.0-2B` | 4.6 GB | Medium | IBM ASR with punctuated transcripts, speech translation (`--translate-to`) and keyword biasing (`--keyword`) |
| `Granite-Speech-5.0-2B-GGUF` | 3.1 GB | Fast | Granite Speech 5.0 with a Q8_0 language model; the faster choice, above all on CPU |
| `LFM2.5-Audio-1.5B-GGUF` | 1.2 GB | Medium | Unified audio model with transcription support |
| `LFM3-Audio-2B-GGUF` | 1.46 GB | Not yet measured | Unified audio model with transcription in 18 languages; `LFM3-Audio-2B-Q8_0-GGUF` (2.20 GB) has a Q8_0 language model |
| `Voxtral-Mini-4B-Realtime-2602` | 8 GB | Medium | Rust/Candle offline transcription; realtime planned |

---

## See Also

- [Transcription Guide](/features/transcription)
- [Diarization Guide](/features/diarization)
