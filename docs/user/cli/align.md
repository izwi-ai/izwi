---
title: "izwi align"
description: "Align reference text to audio and export timestamped word or segment results."
icon: "git-compare-arrows"
---
# izwi align

Forced alignment — align text to audio at word level.

---

## Synopsis

```bash
izwi align <FILE> <TEXT> [OPTIONS]
```

---

## Description

Aligns reference text to audio, producing word-level timestamps. Useful for:

- Subtitle generation
- Karaoke timing
- Audio editing
- Pronunciation analysis

---

## Arguments

| Argument | Description |
|----------|-------------|
| `<FILE>` | Audio file to align |
| `<TEXT>` | Reference text to align |

---

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model <MODEL>` | Alignment model | `qwen3-forcedaligner-0.6b` |
| `-f, --format <FORMAT>` | Output format: `text`, `json`, `verbose_json` | `json` |
| `-o, --output <PATH>` | Output file (default: stdout) | — |

---

## Examples

### Basic alignment

```bash
izwi align audio.wav "Hello world, this is a test." --model Qwen3-ForcedAligner-0.6B
```

### Save to file

```bash
izwi align audio.wav "Hello world" --output alignment.json
```

### Text output

```bash
izwi align audio.wav "Hello world" --format text
```

---

## Output Formats

### JSON (default)

```json
{
  "alignments": [
    {"word": "Hello", "start": 0.0, "end": 0.45},
    {"word": "world", "start": 0.50, "end": 0.95},
    {"word": "this", "start": 1.10, "end": 1.30},
    {"word": "is", "start": 1.35, "end": 1.45},
    {"word": "a", "start": 1.50, "end": 1.55},
    {"word": "test", "start": 1.60, "end": 2.00}
  ],
  "duration": 2.0
}
```

### Text

```
Hello     0.00 - 0.45
world     0.50 - 0.95
this      1.10 - 1.30
is        1.35 - 1.45
a         1.50 - 1.55
test      1.60 - 2.00
```

---

## Use Cases

### Subtitle Generation

Generate precise timestamps for subtitles:

```bash
izwi align video_audio.wav "$(cat script.txt)" --output subtitles.json
```

### Audio Editing

Find exact word boundaries for editing:

```bash
izwi align podcast.wav "um actually" --format json
```

### Pronunciation Analysis

Analyze timing of spoken words:

```bash
izwi align recording.wav "The quick brown fox" --format verbose_json
```

---

## Available Models

| Model | Description |
|-------|-------------|
| `qwen3-forcedaligner-0.6b` | CLI default alias |
| `Qwen3-ForcedAligner-0.6B` | Canonical forced aligner model ID |
| `Qwen3-ForcedAligner-0.6B-4bit` | Lower-memory variant |

---

## See Also

- [`izwi transcribe`](./transcribe.md) — Speech-to-text
- [`izwi diarize`](./diarize.md) — Speaker diarization
