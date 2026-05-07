# API Reference

Izwi serves a local HTTP API from the same process that powers the web UI and desktop app. By default the base URL is:

```text
http://localhost:8080
```

Most API routes are under `/v1`. The exceptions are `/docs`, `/openapi.json`, `/livez`, and `/readyz`.

When `izwi serve` is running:

- `http://localhost:8080/docs` opens the local Scalar reference.
- `http://localhost:8080/openapi.json` returns the generated OpenAPI document. It includes Scalar sidebar entries for preview first-party, operator, and realtime route families with lightweight summaries.
- This page is the detailed contract guide for the broader first-party, preview, operator, and realtime route surface.

## Surface Maturity

| Surface | Status | Notes |
|---------|--------|-------|
| `/v1/models`, `/v1/models/{model}` | Stable | OpenAI-compatible model catalog. |
| `/v1/chat/completions` | Stable | OpenAI-compatible chat completions. |
| `/v1/audio/speech` | Stable | OpenAI-compatible text-to-speech endpoint with Izwi voice extensions. |
| `/v1/audio/transcriptions` | Stable | OpenAI-compatible transcription endpoint with local streaming support. |
| `/livez`, `/readyz`, `/v1/live`, `/v1/ready`, `/v1/health` | Stable | Operational health and readiness endpoints. |
| `/v1/responses*` | Preview | OpenAI-compatible Responses API shape with process-local response retention. |
| `/v1/audio/diarizations`, `/v1/audio/diarize` | Preview | Izwi diarization API; OpenAI-style transport, not an upstream OpenAI endpoint. |
| First-party workflow routes | Preview | Persisted local product APIs used by the web UI and desktop app. |
| `/v1/admin/*` | Preview | Local model-management APIs. Bind carefully on shared hosts. |
| WebSocket realtime routes | Preview | Browser-facing low-latency protocols that may evolve. |
| `/internal/*` aliases | Internal | Compatibility aliases for tooling. Prefer `/v1/*` or root probes where available. |

## Common Conventions

### Request IDs

Clients may send `x-request-id`. If absent, the server generates one. Responses include the same header and structured logs use it as the correlation ID.

```bash
curl -H "x-request-id: demo-123" http://localhost:8080/readyz
```

### Errors

JSON errors use this envelope:

```json
{
  "error": {
    "message": "Unsupported transcription model: Example",
    "type": "invalid_request_error",
    "param": null,
    "code": "400"
  }
}
```

Common status codes:

| Status | Meaning |
|--------|---------|
| `400` | Invalid request shape, model id, unsupported field, or invalid option. |
| `404` | Resource, model, media object, or process-local record was not found. |
| `413` | Body exceeded an upload limit before the handler could parse it. |
| `415` | Endpoint expected JSON or multipart content but received another content type. |
| `500` | Runtime, model, storage, or server failure. |
| `503` | Readiness endpoint reports the server is alive but not ready. |

Enterprise builds can inject authentication and policy hooks. Community builds use local anonymous defaults. If an enterprise hook rejects a request, the response can be `401`, `403`, or `500` before the route handler runs.

### Security And CORS

Community builds do not require API keys by default. Treat the server as a local trusted process unless you deliberately expose it.

- `izwi serve` defaults to port `8080`.
- `--host 0.0.0.0` binds beyond loopback. Use it only on trusted networks or behind your own access controls.
- `--cors` enables wildcard browser CORS responses.
- Desktop origins such as `tauri://localhost` are allowed for the native app.

### Pagination

Preview list APIs use cursor pagination where the response includes a `pagination` object.

Query parameters:

| Parameter | Description |
|-----------|-------------|
| `limit` | Page size. Values are clamped by each store, usually up to `500`. |
| `cursor` | Opaque cursor returned from the previous page. |

Response shape:

```json
{
  "records": [],
  "pagination": {
    "next_cursor": null,
    "has_more": false,
    "limit": 50
  }
}
```

Some older list routes return arrays or route-specific wrapper names. The route sections below note those families.

### Limits And Runtime Controls

| Control | Default | Notes |
|---------|---------|-------|
| `--max-concurrent` | `100` | Maximum concurrent runtime requests. |
| `--timeout` | `300` seconds | Request timeout for regular HTTP routes. Long ASR streaming avoids a hard wall-clock cutoff while active. |
| `IZWI_OPENAI_AUDIO_UPLOAD_LIMIT_MB` | `25` strict, `64` relaxed | Upload limit for OpenAI audio routes. |
| First-party audio upload limit | `64 MiB` | Applies to persisted transcription, diarization, TTS, voice design, voice clone, and saved voice creation routes. |
| `IZWI_AUDIO_STREAM_EVENT_QUEUE_CAPACITY` | `32` | Buffered SSE audio events for `/v1/audio/speech`. |
| `IZWI_MAX_RESPONSE_STORE_ENTRIES` | `512` | Process-local `/v1/responses` retention cap. |
| `IZWI_MAX_AGENT_SESSION_STORE_ENTRIES` | `512` | Process-local agent session metadata cap. |

### Streaming

HTTP streaming routes use server-sent events (SSE):

- Response content type is `text/event-stream`.
- Each payload is sent as a `data:` frame.
- OpenAI-compatible chat and Responses streams end with `data: [DONE]`.
- Some preview first-party streams emit JSON objects with an `event` field and close after the terminal event.
- Client disconnects cancel delivery; some model work may finish internally before cleanup.

## OpenAI-Compatible APIs

### Models

`GET /v1/models`

Returns enabled model variants in OpenAI list format.

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen3-8B-GGUF",
      "object": "model",
      "created": 1760000000,
      "owned_by": "agentem"
    }
  ]
}
```

`GET /v1/models/{model}`

Returns one enabled model in the same object shape. Unknown or disabled variants return `404`.

Use `/v1/admin/models` when you need download, load, unload, local path, status, or speech-capability details.

### Chat Completions

`POST /v1/chat/completions`

Basic request:

```json
{
  "model": "Qwen3-8B-GGUF",
  "messages": [
    { "role": "system", "content": "You are concise." },
    { "role": "user", "content": "Say hello." }
  ],
  "max_tokens": 128,
  "stream": false
}
```

Supported request fields:

| Field | Notes |
|-------|-------|
| `model` | Required model variant. Must resolve to a chat-capable model. |
| `messages` | Required array. Roles: `system`, `user`, `assistant`, `tool`. |
| `max_tokens`, `max_completion_tokens` | Optional output budgets. |
| `stream` | `true` returns SSE chat chunks. |
| `stream_options.include_usage` | Adds usage to the terminal stream chunk. |
| `temperature`, `top_p`, `presence_penalty` | Passed to runtime where supported. |
| `frequency_penalty`, `stop` | Rejected in strict OpenAI compatibility mode when non-default. |
| `n` | Only `1` is supported. |
| `tools`, `tool_choice` | Accepted for tool-call prompting. Strict mode only allows `tool_choice` as `auto`, `none`, or `null`. |
| `enable_thinking` | Izwi extension for thinking-capable local models. |
| `user` | Accepted for compatibility; not used for local auth. |

Compatibility profile:

```bash
IZWI_OPENAI_COMPAT_PROFILE=strict   # default
IZWI_OPENAI_COMPAT_PROFILE=relaxed
```

Streaming sequence:

```text
data: {"object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant"}}]}
data: {"object":"chat.completion.chunk","choices":[{"delta":{"content":"Hel"}}]}
data: {"object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

Tool behavior:

- Assistant messages may include `tool_calls`.
- Tool responses can be sent with role `tool`.
- Model-emitted tool calls are returned with `finish_reason: "tool_calls"` when detected.

Multimodal content parts:

```json
{
  "role": "user",
  "content": [
    { "type": "input_text", "text": "Describe this." },
    { "type": "input_image", "image_url": { "url": "https://example.com/image.png" } }
  ]
}
```

Image and video inputs are validated against the selected model. Text-only chat models reject media parts.

### Audio Speech

`POST /v1/audio/speech`

Generates audio bytes. JSON request:

```json
{
  "model": "Qwen3-TTS-12Hz-0.6B-Base",
  "input": "Hello from Izwi.",
  "voice": "default",
  "response_format": "wav"
}
```

Request fields:

| Field | Notes |
|-------|-------|
| `model` | Required TTS model variant. |
| `input` | Required text to synthesize. |
| `voice` | Built-in voice/speaker name where the model supports presets. |
| `response_format` | `wav`, `pcm`, `pcm16`, `pcm_i16`, `raw_i16`, `raw_f32`, `pcm_f32`, `mp3`, `opus`, `aac`, `flac`. OpenAI compressed names currently fall back to WAV bytes. |
| `speed` | Optional model-dependent speed control. |
| `language` | Optional language hint such as `English`, `Chinese`, or `Auto`. |
| `temperature`, `top_k` | Optional sampling controls. |
| `max_tokens`, `max_output_tokens` | Optional output token budget aliases. |
| `instructions` | Voice-design prompt for voice-design models. |
| `reference_audio` | Base64 audio for voice cloning. |
| `reference_text` | Transcript of the reference audio. |
| `saved_voice_id` | Server-side saved voice reference to reuse. |
| `stream`, `stream_format` | `stream: true` or `stream_format: "sse"` enables SSE audio chunks. |

Non-stream response:

- Body is binary audio.
- Content type follows the generated/fallback format.
- `mp3`, `opus`, `aac`, and `flac` request names are accepted for OpenAI compatibility but currently return WAV bytes with a transparent fallback.

SSE events:

| Event | Fields |
|-------|--------|
| `audio.started` | `request_id`, `sample_rate`, `audio_format`, optional fallback note in `error`. |
| `audio.chunk` | `request_id`, `sequence`, `audio_base64`, `sample_count`, `is_final`. |
| `audio.done` | `request_id`, `tokens_generated`, `generation_time_ms`, `audio_duration_secs`, `rtf`. |
| `audio.failed` | `request_id`, `error`. |

Example SSE request:

```bash
curl -N http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"Kokoro-82M","input":"Stream this.","stream_format":"sse"}'
```

### Audio Transcriptions

`POST /v1/audio/transcriptions`

Accepts JSON or multipart input.

JSON request:

```json
{
  "audio_base64": "<base64-audio>",
  "model": "Parakeet-TDT-0.6B-v3",
  "language": "English",
  "response_format": "verbose_json",
  "stream": false
}
```

Multipart fields:

| Field | Notes |
|-------|-------|
| `file` or `audio` | Uploaded audio file. |
| `audio_base64` | Base64 audio alternative. |
| `model` | Optional ASR, Voxtral, or audio-chat model variant. |
| `language` | Optional language hint. |
| `response_format` | `json`, `verbose_json`, `text`, `srt`, or `vtt`. Default `json`. |
| `stream` | `true`, `1`, `yes`, or `on` enables SSE. |
| `prompt`, `temperature`, `timestamp_granularities[]` | Accepted for compatibility; currently not used by the runtime path. |

`json` response:

```json
{
  "text": "Hello, this is a transcription test."
}
```

`verbose_json` response:

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

SSE events:

| Type | Fields |
|------|--------|
| `transcript.text.delta` | `delta` |
| `transcript.text.done` | `text`, `language`, `audio_duration_secs` |
| `error` | `error.message` |

Example multipart request:

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@meeting.wav" \
  -F "model=Parakeet-TDT-0.6B-v3" \
  -F "response_format=verbose_json"
```

### Audio Diarizations

`POST /v1/audio/diarizations`

Legacy alias:

```text
POST /v1/audio/diarize
```

This route is Izwi-specific preview API. It accepts JSON or multipart input in the same style as transcription.

Request fields:

| Field | Notes |
|-------|-------|
| `audio_base64`, `file`, `audio` | Audio input. |
| `model` or `diarization_model` | Diarization model variant. |
| `asr_model` | Optional ASR override. |
| `aligner_model` | Optional forced aligner override. |
| `llm_model` | Optional transcript refinement model. |
| `response_format` | `json`, `verbose_json`, or `text`. |
| `num_speakers` | Legacy shortcut that sets min and max to the same value. |
| `min_speakers`, `max_speakers` | Speaker bounds. |
| `min_speech_duration_ms`, `min_silence_duration_ms` | VAD tuning. |
| `enable_llm_refinement` | Enables LLM transcript refinement. |
| `stream` | Accepted, but `true` returns `400`; streaming diarization is not supported. |

`json` response:

```json
{
  "segments": [
    { "speaker": "SPEAKER_00", "start": 0.0, "end": 5.2, "confidence": 0.91 }
  ],
  "transcript": "SPEAKER_00 [0.00s - 5.20s]: Welcome."
}
```

`verbose_json` adds `words`, `utterances`, `asr_text`, `raw_transcript`, `llm_refined`, `alignment_coverage`, `unattributed_words`, `speaker_count`, `duration`, `processing_time_ms`, and `rtf`.

### Responses

`POST /v1/responses`

Preview OpenAI-compatible Responses API shape.

```json
{
  "model": "Qwen3-8B-GGUF",
  "instructions": "Be concise.",
  "input": "Write one sentence.",
  "max_output_tokens": 128,
  "store": true
}
```

Request fields:

| Field | Notes |
|-------|-------|
| `model` | Required chat-capable model variant. |
| `input` | Text, one input item, or an array of input items. |
| `instructions` | Optional system instruction. Required if `input` is empty. |
| `max_output_tokens` | Optional output limit. |
| `stream` | `true` returns SSE events. |
| `metadata`, `user` | Stored or accepted for compatibility. |
| `temperature`, `top_p` | Optional runtime controls. |
| `store` | `false` skips process-local retention. Default retains completed records. |
| `tools`, `tool_choice`, `enable_thinking` | Same behavior as chat completions. |

Stored records are process-local:

- They are lost on server restart.
- They can be evicted after `IZWI_MAX_RESPONSE_STORE_ENTRIES`.
- Streaming records are stored only after a terminal completion or failure.
- `cancel` does not provide durable active-response cancellation semantics.

Lifecycle routes:

| Method | Path | Notes |
|--------|------|-------|
| `GET` | `/v1/responses/{response_id}` | Fetch retained process-local response. |
| `DELETE` | `/v1/responses/{response_id}` | Delete retained process-local response. |
| `POST` | `/v1/responses/{response_id}/cancel` | Mark retained process-local response canceled. |
| `GET` | `/v1/responses/{response_id}/input_items` | Return normalized retained input items. |

Streaming events:

```text
response.created
response.in_progress
response.output_item.added
response.content_part.added
response.output_text.delta
response.output_text.done
response.content_part.done
response.output_item.done
response.completed or response.failed
[DONE]
```

## First-Party Workflow APIs

These routes are preview APIs used by the web UI and desktop app. They are local, SQLite-backed stores unless otherwise noted.

### Speech-Text Jobs

Canonical saved transcription and diarization job routes:

| Method | Path | Notes |
|--------|------|-------|
| `GET` | `/v1/transcriptions/jobs` | List jobs. Supports `limit`, `cursor`, and `job_kind=transcription|diarization|all`. |
| `POST` | `/v1/transcriptions/jobs` | Create transcription or diarization job. Multipart uploads allowed. |
| `GET` | `/v1/transcriptions/jobs/{record_id}` | Fetch one job. `job_kind` can disambiguate. |
| `PATCH`, `PUT` | `/v1/transcriptions/jobs/{record_id}` | Update editable metadata such as title, transcript fields, speaker labels, or summary state depending on job kind. |
| `DELETE` | `/v1/transcriptions/jobs/{record_id}` | Delete job and associated stored media. |
| `GET` | `/v1/transcriptions/jobs/{record_id}/audio` | Fetch stored source audio. |
| `POST` | `/v1/transcriptions/jobs/{record_id}/reruns` | Re-run diarization from stored source audio. |
| `POST` | `/v1/transcriptions/jobs/{record_id}/cancel` | Cancel an in-flight diarization job. |
| `POST` | `/v1/transcriptions/jobs/{record_id}/summary/regenerate` | Regenerate transcription or diarization summary. |

The `job_kind` query parameter is important for shared IDs and for clients that want a specific record family.

Legacy transcription-only routes:

| Method | Path |
|--------|------|
| `GET`, `POST` | `/v1/transcriptions` |
| `GET`, `DELETE` | `/v1/transcriptions/{record_id}` |
| `GET` | `/v1/transcriptions/{record_id}/audio` |
| `POST` | `/v1/transcriptions/{record_id}/summary/regenerate` |

### Diarization Records

Persisted diarization routes:

| Method | Path | Notes |
|--------|------|-------|
| `GET`, `POST` | `/v1/diarizations` | List or create saved diarization records. |
| `GET`, `PATCH`, `PUT`, `DELETE` | `/v1/diarizations/{record_id}` | Fetch, update, or delete a saved diarization record. |
| `GET` | `/v1/diarizations/{record_id}/audio` | Fetch source audio. |
| `POST` | `/v1/diarizations/{record_id}/reruns` | Re-run diarization. |
| `POST` | `/v1/diarizations/{record_id}/cancel` | Cancel in-flight diarization. |
| `POST` | `/v1/diarizations/{record_id}/summary/regenerate` | Regenerate the LLM summary. |

### Speech Generation History

All three speech history families share list/create, member, audio, pagination, and deletion behavior. Create routes can generate audio and persist the resulting record.

| Route family | Purpose |
|--------------|---------|
| `/v1/text-to-speech-generations` | Plain TTS history. |
| `/v1/voice-design-generations` | Voice-design prompt generations. |
| `/v1/voice-clone-generations` | Reference-audio voice clone generations. |

Routes:

| Method | Path pattern |
|--------|--------------|
| `GET`, `POST` | `/v1/{family}` |
| `GET`, `DELETE` | `/v1/{family}/{record_id}` |
| `GET` | `/v1/{family}/{record_id}/audio` |

Streaming create responses emit JSON SSE events with an `event` field:

| Event | Notes |
|-------|-------|
| `created` | Includes the persisted record shell. |
| `start` | Includes `request_id`, `sample_rate`, and `audio_format`. |
| `chunk` | Includes `request_id`, `sequence`, `audio_base64`, and `sample_count`. |
| `final` | Includes generation stats and the completed record. |
| `error` | Includes an error string. |
| `done` | Terminal stream marker. |

### Saved Voices

Reusable voice clone references:

| Method | Path | Notes |
|--------|------|-------|
| `GET` | `/v1/voices` | List saved voices with cursor pagination. |
| `POST` | `/v1/voices` | Create a saved voice from reference audio/text or a generated voice source. |
| `GET` | `/v1/voices/{voice_id}` | Fetch saved voice metadata. |
| `DELETE` | `/v1/voices/{voice_id}` | Delete saved voice and audio. |
| `GET` | `/v1/voices/{voice_id}/audio` | Fetch saved reference audio. |

Use `saved_voice_id` on `/v1/audio/speech` or first-party generation routes to reuse a saved voice without resending reference audio.

### Studio

Studio is the long-form TTS project API.

Project and folder routes:

| Method | Path |
|--------|------|
| `GET`, `POST` | `/v1/studio/folders` |
| `GET`, `POST` | `/v1/studio/projects` |
| `GET`, `PATCH`, `DELETE` | `/v1/studio/projects/{project_id}` |
| `GET` | `/v1/studio/projects/{project_id}/audio` |
| `GET`, `PATCH` | `/v1/studio/projects/{project_id}/meta` |

Audio export query parameters:

| Parameter | Notes |
|-----------|-------|
| `download=true` | Prefer attachment-style download headers. |
| `format=wav|mp3|flac|raw_i16` | Requested export format. |
| `segment_ids=a,b,c` | Export selected segments in order. |

Pronunciations and snapshots:

| Method | Path |
|--------|------|
| `GET`, `POST` | `/v1/studio/projects/{project_id}/pronunciations` |
| `DELETE` | `/v1/studio/projects/{project_id}/pronunciations/{pronunciation_id}` |
| `GET`, `POST` | `/v1/studio/projects/{project_id}/snapshots` |
| `POST` | `/v1/studio/projects/{project_id}/snapshots/{snapshot_id}/restore` |

Render jobs:

| Method | Path |
|--------|------|
| `GET`, `POST` | `/v1/studio/projects/{project_id}/render-jobs` |
| `PATCH` | `/v1/studio/projects/{project_id}/render-jobs/{job_id}` |

Segment editing:

| Method | Path |
|--------|------|
| `POST` | `/v1/studio/projects/{project_id}/segments` |
| `GET`, `PATCH`, `DELETE` | `/v1/studio/projects/{project_id}/segments/{segment_id}` |
| `POST` | `/v1/studio/projects/{project_id}/segments/{segment_id}/split` |
| `POST` | `/v1/studio/projects/{project_id}/segments/{segment_id}/merge-next` |
| `PATCH` | `/v1/studio/projects/{project_id}/segments/reorder` |
| `POST` | `/v1/studio/projects/{project_id}/segments/bulk-delete` |
| `POST` | `/v1/studio/projects/{project_id}/segments/{segment_id}/render` |

Render-job statuses are route-specific preview values such as queued, running, completed, failed, cancelled, or stale. Clients should preserve unknown statuses.

### Chat Threads

Durable local chat history:

| Method | Path | Notes |
|--------|------|-------|
| `GET`, `POST` | `/v1/chat/threads` | List or create threads. |
| `GET`, `PATCH`, `DELETE` | `/v1/chat/threads/{thread_id}` | Fetch, rename, or delete a thread. |
| `GET`, `POST` | `/v1/chat/threads/{thread_id}/messages` | List messages or send a new user message. |

Send-message request fields:

| Field | Notes |
|-------|-------|
| `model` | Optional chat model. |
| `content` | User text. |
| `content_parts` | Multimodal content parts in OpenAI-like shape. |
| `max_tokens` | Optional output limit. |
| `stream` | `true` emits SSE events. |
| `system_prompt` | Optional per-request system prompt. |
| `enable_thinking` | Izwi extension for thinking-capable models. |

Streaming thread events:

| Event | Notes |
|-------|-------|
| `start` | Includes `thread_id`, `model_id`, and persisted user message. |
| `delta` | Text delta. |
| `done` | Includes persisted assistant message and generation stats. |
| `error` | Error string. |

### Agent Sessions

Agent session metadata is process-local preview state. The linked chat thread is durable.

| Method | Path | Notes |
|--------|------|-------|
| `POST` | `/v1/agent/sessions` | Create an agent session and linked thread. |
| `GET` | `/v1/agent/sessions/{session_id}` | Fetch retained process-local session metadata. |
| `POST` | `/v1/agent/sessions/{session_id}/turns` | Run one agent turn. |

Create fields include `agent_id`, `model_id`, `system_prompt`, `planning_mode` (`off`, `auto`, `on`), and `title`.

Turn responses include assistant text, optional plan steps, tool calls, and ordered events such as `turn_started`, `plan_created`, `tool_call_started`, `tool_call_completed`, `assistant_message`, and `turn_completed`.

### Voice Profile, Memory, And Sessions

Voice-mode persisted state:

| Method | Path | Notes |
|--------|------|-------|
| `GET`, `PATCH` | `/v1/voice/profile` | Fetch or update name, system prompt, and observational memory setting. |
| `GET`, `DELETE` | `/v1/voice/observations` | List or clear remembered observations. `limit` controls list size. |
| `DELETE` | `/v1/voice/observations/{observation_id}` | Forget one observation. |
| `GET` | `/v1/voice/sessions` | List voice sessions. |
| `GET` | `/v1/voice/sessions/{session_id}` | Fetch a session with turns. |

Observational memory is applied to modular voice conversations. Updates are stored locally and can be cleared by the user.

### Media

`GET /v1/media/{path}`

Serves persisted media objects used by chat attachments and local workflows.

Rules:

- The path is relative to the media root.
- Absolute paths and `..` traversal are rejected.
- Unknown media returns `404`.
- Treat media URLs as local API resources, not stable public object-store URLs.

### Onboarding And Preferences

Small first-party UI state APIs:

| Method | Path | Response |
|--------|------|----------|
| `GET` | `/v1/onboarding` | `{ completed, completed_at, analytics_opt_in }` |
| `POST` | `/v1/onboarding/complete` | Marks onboarding complete and returns state. |
| `GET` | `/v1/preferences` | `{ analytics_opt_in }` |
| `PUT` | `/v1/preferences/analytics` | Body `{ "opt_in": true }`; returns preferences. |

## Operator APIs

### Health And Readiness

| Method | Path | Notes |
|--------|------|-------|
| `GET` | `/livez` | Cheap liveness probe. |
| `GET` | `/readyz` | Deployment readiness probe. Returns `503` when alive but not ready. |
| `GET` | `/v1/live` | `/v1` alias for liveness. |
| `GET` | `/v1/ready` | `/v1` alias for readiness. |
| `GET` | `/v1/health` | Rich runtime/backend status used by `izwi status`. |
| `GET` | `/internal/live`, `/internal/ready`, `/internal/health` | Internal compatibility aliases. |

`/v1/health` includes requested and selected backend, compiled backend support, detected device capabilities, dtype policy, CUDA runtime diagnostics, and fused-attention status.

### Metrics

| Method | Path | Notes |
|--------|------|-------|
| `GET` | `/v1/metrics` | JSON runtime telemetry snapshot. |
| `GET` | `/v1/metrics/prometheus` | Prometheus text format. |
| `GET` | `/internal/metrics` | Internal alias. |
| `GET` | `/internal/metrics/prometheus` | Internal alias. |

### Admin Model Management

Preview local admin routes:

| Method | Path | Notes |
|--------|------|-------|
| `GET` | `/v1/admin/models` | List known enabled variants and local status. |
| `GET` | `/v1/admin/models/{variant}` | Fetch one model status. |
| `POST` | `/v1/admin/models/{variant}/download` | Start background download. |
| `GET` | `/v1/admin/models/{variant}/download/progress` | SSE download progress. |
| `POST` | `/v1/admin/models/{variant}/download/cancel` | Cancel active download. |
| `POST` | `/v1/admin/models/{variant}/load` | Load model into runtime memory. |
| `POST` | `/v1/admin/models/{variant}/unload` | Unload model from runtime memory. |
| `DELETE` | `/v1/admin/models/{variant}` | Unload and delete local model files. |

Model status values:

```text
not_downloaded
downloading
downloaded
loading
ready
error
```

Speech model capabilities, when present:

```json
{
  "supports_builtin_voices": true,
  "built_in_voice_count": 54,
  "supports_reference_voice": false,
  "supports_voice_description": false,
  "supports_streaming": true,
  "supports_speed_control": true,
  "supports_auto_long_form": true
}
```

Download progress SSE payload:

```json
{
  "variant": "Qwen3-8B-GGUF",
  "downloaded_bytes": 1048576,
  "total_bytes": 2147483648,
  "current_file": "model.gguf",
  "current_file_downloaded": 1048576,
  "current_file_total": 2147483648,
  "files_completed": 0,
  "files_total": 1,
  "percent": 0.05,
  "status": "downloading"
}
```

Progress `status` can be `downloading`, `completed`, `error`, or `cancelled`.

## Realtime WebSocket APIs

Realtime routes are preview browser protocols. They use JSON text messages for control events and binary PCM16 frames for audio.

### Transcription Realtime

`GET /v1/transcription/realtime/ws`

Server starts with:

```json
{ "type": "session_ready", "protocol": "transcription_realtime_v2" }
```

Client JSON messages:

| Type | Fields |
|------|--------|
| `session_start` | Optional `model_id`, `language`. |
| `session_stop` | No fields. |
| `ping` | Optional `timestamp_ms`. |

Client binary frame:

| Bytes | Value |
|-------|-------|
| `0..4` | ASCII magic `ITRW`. |
| `4` | Version `1`. |
| `5` | Kind `1` for client PCM16. |
| `6..8` | Reserved. |
| `8..12` | Little-endian `sample_rate` (`u32`). |
| `12..16` | Little-endian `frame_seq` (`u32`). |
| `16..` | Mono PCM16 little-endian audio bytes. |

Server events:

| Type | Notes |
|------|-------|
| `session_started` | Session accepted. |
| `transcript_partial` | Includes `sequence`, `text`, optional `language`, and audio duration. |
| `session_done` | Session stopped. |
| `pong` | Ping response. |
| `error` | Error message. |

Constraints:

- Binary frames larger than 512 KiB are rejected.
- Sample rate must remain stable during a session.
- Sample rates outside the accepted runtime range return an error.

### Voice Realtime

`GET /v1/voice/realtime/ws`

Server starts with:

```json
{ "type": "connected", "protocol": "voice_realtime_v1" }
```

Client JSON messages:

| Type | Fields |
|------|--------|
| `session_start` | Optional `system_prompt`. |
| `input_stream_start` | Optional `mode`, model ids, speaker, ASR language, max tokens, VAD settings, and input sample rate. |
| `input_stream_stop` | Stops listening and closes the current session. |
| `interrupt` | Optional `reason`; interrupts active assistant turn. |
| `ping` | Optional `timestamp_ms`. |

`mode` values:

| Mode | Notes |
|------|-------|
| `modular` | ASR -> chat/agent -> TTS. Requires ASR, text chat, and TTS models. |
| `unified` | Uses a supported audio-chat model for speech-to-speech style turns. |

`input_stream_start` fields:

| Field | Notes |
|-------|-------|
| `asr_model_id`, `text_model_id`, `tts_model_id` | Modular model overrides. |
| `s2s_model_id` | Unified audio-chat model override. |
| `speaker` | TTS speaker/voice. |
| `asr_language` | Language hint. |
| `max_output_tokens` | Text output budget. |
| `vad_threshold` | Speech detection threshold. |
| `min_speech_ms` | Minimum speech duration. |
| `silence_duration_ms` | Silence before utterance end. |
| `max_utterance_ms` | Hard utterance duration cap. |
| `pre_roll_ms` | Audio retained before speech start. |
| `input_sample_rate` | Expected input sample rate. |

Client binary frame:

| Bytes | Value |
|-------|-------|
| `0..4` | ASCII magic `IVWS`. |
| `4` | Version `1`. |
| `5` | Kind `1` for client PCM16. |
| `6..8` | Reserved. |
| `8..12` | Little-endian `sample_rate` (`u32`). |
| `12..16` | Little-endian `frame_seq` (`u32`). |
| `16..` | Mono PCM16 little-endian audio bytes. |

Assistant audio binary frame:

| Bytes | Value |
|-------|-------|
| `0..4` | ASCII magic `IVWS`. |
| `4` | Version `1`. |
| `5` | Kind `2` for assistant PCM16. |
| `6..8` | Flags; bit `0` marks final chunk. |
| `8..16` | Little-endian `utterance_seq` (`u64`). |
| `16..20` | Little-endian chunk sequence (`u32`). |
| `20..24` | Little-endian sample rate (`u32`). |
| `24..` | Mono PCM16 little-endian audio bytes. |

Server events:

| Type | Notes |
|------|-------|
| `session_ready` | Voice session initialized. |
| `input_stream_ready` | Includes resolved VAD settings. |
| `input_stream_stopped` | Input stream stopped. |
| `listening` | Ready for an utterance. |
| `user_speech_start`, `user_speech_end` | VAD utterance boundaries. |
| `turn_processing` | Turn started. |
| `user_transcript_start`, `user_transcript_delta`, `user_transcript_final` | User transcript events. |
| `assistant_text_start`, `assistant_text_delta`, `assistant_text_final` | Assistant text events. |
| `assistant_audio_start`, `assistant_audio_done` | Assistant audio envelope around binary chunks. |
| `turn_done` | Terminal turn status: `ok`, `error`, `timeout`, `interrupted`, or `no_input`. |
| `pong` | Ping response. |
| `error` | Error with optional utterance identifiers. |

Voice realtime persists voice sessions and turns in the local store. Modular turns can also update observational memory when that feature is enabled.
