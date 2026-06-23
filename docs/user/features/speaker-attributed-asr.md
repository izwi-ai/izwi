---
title: "Speaker Attributed ASR"
description: "Generate Granite Speech speaker-turn transcripts through the Transcription workspace and speech-text jobs API."
sidebarTitle: "Speaker Attributed ASR"
icon: "users-round"
---
Speaker Attributed ASR (SAA) is a rich transcription mode that asks Granite
Speech to produce a transcript with speaker-turn labels. Use it when you want a
speaker-attributed text transcript, not acoustic diarization segments.

For "who spoke when" timelines with timestamps, use [Diarization](/features/diarization).

---

## When to Use SAA

| Use SAA when... | Use diarization when... |
|-----------------|-------------------------|
| You want a readable `[Speaker N]:` style transcript. | You need speaker segments with start/end times. |
| Granite Speech's language model can infer speaker turns from the audio. | You need acoustic speaker separation from the Sortformer pipeline. |
| You do not need streaming or timestamp alignment. | You need reruns, alignment metrics, and diarization-specific quality controls. |

---

## Model Requirement

SAA currently requires:

```bash
izwi pull Granite-Speech-4.1-2B-Plus
```

If the API request omits `model_id`, the persisted SAA workflow defaults to
`Granite-Speech-4.1-2B-Plus`. Supplying a different model returns a validation
error.

---

## Using the Web UI

1. Open **Transcription** in the sidebar.
2. Choose **Speaker Attributed ASR** from the mode switch.
3. Upload or record audio.
4. Select a ready Granite Speech model.
5. Choose a speaker expectation: **Auto**, **2+**, **3+**, or **4+**.
6. Optionally enable summary generation.
7. Submit the job and review the speaker-turn transcript.

SAA disables streaming and timestamp alignment. Those controls belong to normal
transcription or diarization workflows.

---

## Using the API

Create a persisted SAA job with `job_kind=speaker_attributed_asr`:

```bash
curl -X POST "http://localhost:8080/v1/speech-to-text/jobs?job_kind=speaker_attributed_asr" \
  -F "file=@meeting.wav" \
  -F "model_id=Granite-Speech-4.1-2B-Plus" \
  -F "language=English" \
  -F "generate_summary=true" \
  -F "min_speakers=2"
```

The short alias `job_kind=saa` is also accepted.

Poll the returned record until `processing_status` is `ready`:

```bash
curl "http://localhost:8080/v1/speech-to-text/jobs/saa_123?job_kind=speaker_attributed_asr"
```

Fetch stored audio or regenerate summaries with the same job kind:

```bash
curl "http://localhost:8080/v1/speech-to-text/jobs/saa_123/audio?job_kind=speaker_attributed_asr" \
  --output source.wav

curl -X POST \
  "http://localhost:8080/v1/speech-to-text/jobs/saa_123/summary/regenerate?job_kind=speaker_attributed_asr"
```

---

## Request Fields

JSON and multipart create requests accept:

| Field | Description |
|-------|-------------|
| `file` / `audio_base64` | Source audio upload. |
| `model_id` / `model` | Optional model override. Must be `Granite-Speech-4.1-2B-Plus` when present. |
| `language` | Optional language hint, such as `English`. |
| `generate_summary` | Generate an AI summary after the transcript completes. Defaults to `false`. |
| `min_speakers`, `max_speakers` | Optional speaker expectation bounds. `min_speakers` is what the current UI sends. |

Unsupported for SAA:

- `stream`
- `include_timestamps`
- `word_timestamps`
- `aligner_model_id`

---

## See Also

- [Transcription](/features/transcription)
- [Diarization](/features/diarization)
- [API Reference](/api#speech-text-jobs)
