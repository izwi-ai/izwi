# Diarization

Identify and separate multiple speakers in audio recordings with speaker diarization.

---

## Overview

Speaker diarization answers the question "who spoke when?" It segments audio by speaker, making it invaluable for:

- **Meeting transcripts** — Attribute statements to participants
- **Interviews** — Separate interviewer and interviewee
- **Podcasts** — Identify hosts and guests
- **Call recordings** — Distinguish callers

---

## Getting Started

### Download Diarization Pipeline Models

For best results, use a diarization + ASR + aligner pipeline:

```bash
izwi pull diar_streaming_sortformer_4spk-v2.1
izwi pull Parakeet-TDT-0.6B-v3
izwi pull Qwen3-ForcedAligner-0.6B
```

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
POST /v1/audio/diarizations
```

Legacy alias still accepted for older clients:

```
POST /v1/audio/diarize
```

### Request (multipart/form-data)

| Field | Type | Description |
|-------|------|-------------|
| `file` | File | Audio file to analyze |
| `model` | String | Diarization model (for example `diar_streaming_sortformer_4spk-v2.1`) |
| `asr_model` | String | Optional ASR model override |
| `aligner_model` | String | Optional forced aligner model override |
| `llm_model` | String | Optional transcript refinement model |
| `min_speakers` | Integer | Optional minimum expected speakers |
| `max_speakers` | Integer | Optional maximum expected speakers |
| `num_speakers` | Integer | Legacy shortcut; maps to both min/max when provided |
| `min_speech_duration_ms` | Number | Optional VAD speech-duration tuning |
| `min_silence_duration_ms` | Number | Optional VAD silence-duration tuning |
| `enable_llm_refinement` | Boolean/String | Enable optional transcript refinement |
| `response_format` | String | `json`, `verbose_json`, or `text` |

### Example (curl)

```bash
curl -X POST http://localhost:8080/v1/audio/diarizations \
  -F "file=@meeting.wav" \
  -F "model=diar_streaming_sortformer_4spk-v2.1" \
  -F "asr_model=Parakeet-TDT-0.6B-v3" \
  -F "aligner_model=Qwen3-ForcedAligner-0.6B" \
  -F "min_speakers=2" \
  -F "max_speakers=2" \
  -F "response_format=verbose_json"
```

### Response

The default `json` response contains speaker segments and a formatted transcript:

```json
{
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.0,
      "end": 5.2
    },
    {
      "speaker": "SPEAKER_01",
      "start": 5.5,
      "end": 12.1
    }
  ],
  "transcript": "SPEAKER_00 [0.00s - 5.20s]: Welcome to the meeting.\nSPEAKER_01 [5.50s - 12.10s]: Thanks for having me."
}
```

Use `response_format=verbose_json` for words, utterances, speaker count,
duration, alignment coverage, LLM refinement status, and processing metrics.
Streaming diarization is not currently supported on this route.

See the [API Reference](../api.md#audio-diarizations) for JSON input,
legacy alias behavior, and exact response shapes.

---

## Configuration

### Number of Speakers

If you know how many speakers are in the audio, specify it for better accuracy:

```bash
# Via API
curl -X POST http://localhost:8080/v1/audio/diarizations \
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

- [Transcription](./transcription.md) — Single-speaker transcription
- [Voice Mode](./voice.md) — Real-time conversations
- [CLI Reference](../cli/index.md) — Command documentation
