---
title: "Diarization"
description: "Identify multiple speakers in audio and generate speaker-attributed transcripts with Izwi."
icon: "users"
---
Identify and separate multiple speakers in audio recordings with speaker diarization.

---

## Overview

Speaker diarization answers the question "who spoke when?" It segments audio by speaker, making it invaluable for:

- **Meeting transcripts** — Attribute statements to participants
- **Interviews** — Separate interviewer and interviewee
- **Podcasts** — Identify hosts and guests
- **Call recordings** — Distinguish callers

> Speaker-turn transcripts written by the ASR model itself are documented
> separately as [Speaker Attributed ASR](/features/speaker-attributed-asr).
> Use diarization when you need timestamped speaker segments.

---

## Getting Started

### Download Diarization Pipeline Models

For best results, use a diarization + ASR + aligner pipeline:

```bash
izwi pull diar_streaming_sortformer_4spk-v2.1
izwi pull Parakeet-TDT-0.6B-v3
izwi pull Qwen3-ForcedAligner-0.6B
```

For meetings with more than four speakers, `Nemotron-3-Diarization`
resolves up to 8 speaker channels with the same streaming pipeline:

```bash
izwi pull Nemotron-3-Diarization
```

`Nemotron-3.5-Transcribe-30B-A4B-FP8` can instead diarize on its own, with no
separate diarizer, ASR or aligner model; see
[Single-Model Diarization](#single-model-diarization).

### Start the Server

```bash
izwi serve
```

---

## Using the Web UI

1. Navigate to **Transcription** in the sidebar
2. Switch to **Diarization** mode
3. Upload an audio file with multiple speakers
4. Click **Analyze**
5. View the speaker-segmented transcript

### Output

The diarization view shows:
- **Speaker labels** — Speaker 1, Speaker 2, etc.
- **Timestamps** — When each speaker talks
- **Transcript** — What each speaker said

Example output:

```
[00:00 - 00:05] Speaker 1: Welcome to the meeting.
[00:05 - 00:12] Speaker 2: Thanks for having me.
[00:12 - 00:20] Speaker 1: Let's start with the agenda.
```

---

## Using the API

### Endpoint

```
POST /v1/speech-to-text/jobs?job_kind=diarization
```

### Request (multipart/form-data)

| Field | Type | Description |
|-------|------|-------------|
| `file` | File | Audio file to analyze |
| `model` | String | Diarization model (`diar_streaming_sortformer_4spk-v2.1` or `Nemotron-3-Diarization`), or `Nemotron-3.5-Transcribe-30B-A4B-FP8` for [single-model diarization](#single-model-diarization). The selected model bounds `min_speakers` and `max_speakers`: 4 for Sortformer v2.1, 8 for Nemotron-3 and Nemotron 3.5 Transcribe; a higher value returns 400 |
| `asr_model` | String | Optional ASR model override for the pipeline (an ASR or Voxtral model; other ids return 400). `Nemotron-3.5-Transcribe-30B-A4B-FP8` cannot be the pipeline's ASR model (400); name it as `model` instead. Sending `asr_model` with a single-model `model` returns 400 |
| `aligner_model` | String | Optional forced aligner model override (non-aligner ids return 400). Sending it with a single-model `model` returns 400 |
| `llm_model` | String | Optional chat model for the summary (default `Qwen3.5-4B`; non-chat ids return 400) |
| `min_speakers` | Integer | Optional minimum expected speakers |
| `max_speakers` | Integer | Optional maximum expected speakers |
| `min_speech_duration_ms` | Number | Optional VAD speech-duration tuning (not used in single-model diarization) |
| `min_silence_duration_ms` | Number | Optional VAD silence-duration tuning (not used in single-model diarization) |
| `generate_summary` | Boolean/String | Summarize the transcript with `llm_model` once it is ready (default `false`) |
| `language` | String | Optional ASR language hint (default: detected per chunk) |
| `vad_gate` | Boolean/String | Clip diarizer segments to VAD speech regions (default off; helps on noisy input; not used in single-model diarization) |

### Example (curl)

```bash
curl -X POST "http://localhost:8080/v1/speech-to-text/jobs?job_kind=diarization" \
  -F "file=@meeting.wav" \
  -F "model=diar_streaming_sortformer_4spk-v2.1" \
  -F "asr_model=Parakeet-TDT-0.6B-v3" \
  -F "aligner_model=Qwen3-ForcedAligner-0.6B" \
  -F "min_speakers=2" \
  -F "max_speakers=2"
```

### Response

The create response is a persisted diarization job. Poll the returned `id` with
`GET /v1/speech-to-text/jobs/{record_id}?job_kind=diarization` until
`processing_status` is `ready`.

```json
{
  "id": "diarization_...",
  "kind": "diarization",
  "processing_status": "pending"
}
```

Ready job records include `segments`, `words`, `utterances`, `speaker_count`,
`duration_secs`, `alignment_coverage`, processing metrics, any pipeline `warnings`,
and the formatted `transcript`.

See the [API Reference](/api#speech-text-jobs) for JSON input and exact response shapes.

---

## Single-Model Diarization

`Nemotron-3.5-Transcribe-30B-A4B-FP8` transcribes, labels speakers and times
words itself: it carries the Nemotron-3-Diarization speaker model inside it.
Name it as the diarization `model` and Izwi runs no pipeline. It loads no
diarizer, ASR or aligner model and runs one transcription with speaker tags
and word timestamps, up to 60 minutes in one pass.

The model needs about 32 GiB of memory (see
[Transcription](/features/transcription#nemotron-35-transcribe)). Sortformer
stays the default diarization model.

```bash
curl -X POST "http://localhost:8080/v1/speech-to-text/jobs?job_kind=diarization" \
  -F "file=@meeting.mp3" \
  -F "model=Nemotron-3.5-Transcribe-30B-A4B-FP8" \
  -F "max_speakers=4"
```

The ready record has the same fields as a pipeline record:

- `segments` are the embedded speaker model's segments, labelled
  `SPEAKER_00`, `SPEAKER_01`, … by speaker channel.
- `words` take the speaker channel their speaker tag was matched to. A word
  whose tag matched no channel is labelled `UNKNOWN` and counted in
  `unattributed_words`.
- `utterances` and `transcript` are built from the words as in the pipeline.
  `asr_text` is the model's own transcript.

Rules for this mode:

- `asr_model` and `aligner_model` return 400: the model diarizes on its own.
- `min_speakers` and `max_speakers` can be at most 8. The model counts the
  speakers itself; a count outside the bounds only adds a warning.
- `min_speech_duration_ms`, `min_silence_duration_ms` and `vad_gate` are not
  used.
- `language` is passed to the model, as in
  [Transcription](/features/transcription#languages-and-wording).
- If the transcript cannot be aligned (for example, it holds a character the
  model's timing vocabulary lacks), the job still returns the transcript and
  the speaker segments, with a warning and no words.

In the web UI, choose the model under **Single model** in the diarization
model picker. The ASR, aligner and timing fields are then hidden and not sent.
In the CLI:

```bash
izwi diarize call.wav --model Nemotron-3.5-Transcribe-30B-A4B-FP8
```

The model still cannot be the ASR stage of a pipeline run with Sortformer or
Nemotron-3-Diarization.

---

## Configuration

### Number of Speakers

If you know how many speakers are in the audio, specify it for better accuracy:

```bash
# Via API
curl -X POST "http://localhost:8080/v1/speech-to-text/jobs?job_kind=diarization" \
  -F "file=@meeting.wav" \
  -F "min_speakers=3" \
  -F "max_speakers=3"
```

### Speaker Labels

By default, speakers are labeled "Speaker 1", "Speaker 2", etc. You can rename them in the UI after processing.

---

## Tips for Best Results

1. **Quality audio** — Clear recordings with minimal background noise
2. **Distinct voices** — Works best when speakers have different voice characteristics
3. **Minimal overlap** — Speakers talking over each other reduces accuracy
4. **Specify speaker count** — If known, helps the algorithm
5. **Longer segments** — Short utterances are harder to attribute

---

## Limitations

- **Similar voices** — May confuse speakers with very similar voices
- **Overlapping speech** — Simultaneous talking is challenging
- **Background noise** — Reduces speaker detection accuracy
- **Very short clips** — Need enough audio to identify speaker patterns

---

## Use Cases

### Meeting Minutes

Upload a meeting recording to get a transcript with speaker attribution:

1. Record your meeting
2. Upload to Diarization
3. Export the speaker-labeled transcript
4. Edit speaker names as needed

### Interview Transcription

Perfect for journalist interviews or research:

1. Record the interview
2. Process with diarization
3. Get clean Q&A format output

### Podcast Production

Identify speakers for editing and show notes:

1. Upload raw podcast audio
2. See who spoke when
3. Use timestamps for editing

---

## See Also

- [Transcription](/features/transcription) — Single-speaker transcription
- [Speaker Attributed ASR](/features/speaker-attributed-asr) — Speaker-turn transcripts
- [Voice Mode](/features/voice) — Real-time conversations
- [CLI Reference](/cli) — Command documentation
