# izwi transcribe

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
| `-f, --format <FORMAT>` | Output format: `text`, `json`, `verbose_json` | `text` |
| `-o, --output <PATH>` | Output file (default: stdout) | — |
| `--word-timestamps` | Future-compatible flag; current server warns and ignores it | — |

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

### With timestamps

```bash
izwi transcribe audio.wav --format verbose_json --word-timestamps
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
izwi transcribe audio.wav --model Voxtral-Mini-4B-Realtime-2602 --format verbose_json
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

`--word-timestamps` is accepted by the CLI for future compatibility, but the
current server warns that word-level timestamps are not yet supported and ignores
the option.

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
| `Parakeet-TDT-0.6B-v3` | 9.4 GB | Medium | Strong baseline (default) |
| `Whisper-Large-v3-Turbo` | 1.5 GB | Medium | Strong multilingual baseline |
| `Qwen3-ASR-0.6B-GGUF` | 1.0 GB | Fast | Good |
| `Qwen3-ASR-1.7B-GGUF` | 2.5 GB | Medium | Better |
| `Nemotron-3.5-ASR-Streaming-0.6B` | 2.37 GB | Medium | 40-locale NVIDIA FastConformer-RNNT; native forward path pending final weight-map work |
| `Voxtral-Mini-4B-Realtime-2602` | 8 GB | Medium | Rust/Candle offline transcription; realtime planned |

---

## See Also

- [Transcription Guide](../features/transcription.md)
- [Diarization Guide](../features/diarization.md)
