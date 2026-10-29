---
title: "izwi diarize"
description: "Run speaker diarization from the command line and export text or JSON results."
icon: "users"
---
Speaker diarization — identify and separate multiple speakers in audio.

---

## Synopsis

```bash
izwi diarize <FILE> [OPTIONS]
```

---

## Description

Analyzes audio to identify different speakers and when they spoke. The output includes a transcript with speaker labels.

---

## Arguments

| Argument | Description |
|----------|-------------|
| `<FILE>` | Audio file to analyze |

---

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model <MODEL>` | Diarization model, or `Nemotron-3.5-Transcribe-30B-A4B-FP8` to diarize with that model alone | `diar_streaming_sortformer_4spk-v2.1` |
| `-n, --num-speakers <N>` | Expected number of speakers | Auto-detect |
| `-f, --format <FORMAT>` | Output format: `text`, `json`, `verbose-json` | `text` |
| `-o, --output <PATH>` | Output file (default: stdout) | — |
| `--transcribe` | Compatibility flag (transcript output is included by default) | — |
| `--asr-model <MODEL>` | ASR model of the pipeline (optional; omit it with a single-model `--model`) | The server's default, `parakeet-tdt-0.6b-v3` |

---

## Examples

### Basic diarization

```bash
izwi diarize meeting.wav
```

### With known speaker count

```bash
izwi diarize meeting.wav --num-speakers 3
```

### Transcript output (default behavior)

```bash
izwi diarize meeting.wav --num-speakers 2
```

### JSON output

```bash
izwi diarize meeting.wav --format json --output diarization.json
```

### Single-model diarization

```bash
izwi diarize call.wav --model Nemotron-3.5-Transcribe-30B-A4B-FP8
```

`Nemotron-3.5-Transcribe-30B-A4B-FP8` transcribes, labels speakers and times
words itself, so no diarizer, ASR or aligner model is loaded. Do not pass
`--asr-model` with it: the server returns `400`. `--num-speakers` can be at
most 8, and only adds a warning when the model counts a different number. The
model is experimental and needs about 32 GiB of memory; see
[Single-Model Diarization](/features/diarization#single-model-diarization).

### Full pipeline with custom models

```bash
izwi diarize interview.wav \
  --asr-model Qwen3-ASR-1.7B-GGUF \
  --format verbose-json \
  --output interview_transcript.json
```

---

## Output Formats

### Text

```
SPEAKER_00 [0.00s - 5.20s]: Welcome to the meeting.
SPEAKER_01 [5.50s - 12.10s]: Thanks for having me.
SPEAKER_00 [12.30s - 20.00s]: Let's start with the agenda.
```

### JSON

```json
{
  "segments": [
    {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.2},
    {"speaker": "SPEAKER_01", "start": 5.5, "end": 12.1}
  ],
  "transcript": "SPEAKER_00 [0.00s - 5.20s]: Welcome to the meeting.\nSPEAKER_01 [5.50s - 12.10s]: Thanks for having me."
}
```

### Verbose JSON (with transcription)

```json
{
  "segments": [
    {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.2},
    {"speaker": "SPEAKER_01", "start": 5.5, "end": 12.1}
  ],
  "words": [
    {"word": "Welcome", "speaker": "SPEAKER_00", "start": 0.0, "end": 0.6, "overlaps_segment": true}
  ],
  "utterances": [
    {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.2, "text": "Welcome to the meeting.", "word_start": 0, "word_end": 3},
    {"speaker": "SPEAKER_01", "start": 5.5, "end": 12.1, "text": "Thanks for having me.", "word_start": 4, "word_end": 7}
  ],
  "asr_text": "Welcome to the meeting. Thanks for having me.",
  "raw_transcript": "SPEAKER_00 [0.00s - 5.20s]: Welcome to the meeting.\nSPEAKER_01 [5.50s - 12.10s]: Thanks for having me.",
  "transcript": "SPEAKER_00 [0.00s - 5.20s]: Welcome to the meeting.\nSPEAKER_01 [5.50s - 12.10s]: Thanks for having me.",
  "alignment_coverage": 1.0,
  "unattributed_words": 0,
  "speaker_count": 2,
  "duration": 120.5,
  "processing_time_ms": 4210.0,
  "rtf": 0.035
}
```

`words` is shortened here; the real output lists every word.

---

## Available Models

| Model | Description |
|-------|-------------|
| `sortformer-4spk` | Alias for `diar_streaming_sortformer_4spk-v2.1` |
| `diar_streaming_sortformer_4spk-v2.1` | Canonical Sortformer model ID (default), up to 4 speakers |
| `Nemotron-3-Diarization` | NVIDIA Nemotron 3 diarization, up to 8 speakers |
| `Nemotron-3.5-Transcribe-30B-A4B-FP8` | Diarizes on its own, with no pipeline ASR or aligner; up to 8 speakers; experimental |

---

## See Also

- [Diarization Guide](/features/diarization)
- [`izwi transcribe`](/cli/transcribe) — Single-speaker transcription
