---
title: "API Reference"
description: "HTTP, SSE, and WebSocket API contracts for the local Izwi server and first-party workflows."
icon: "braces"
mode: "wide"
---
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
| First-party workflow routes | Preview | Persisted local product APIs used by the web UI and desktop app. |
| `/v1/jobs/*` | Preview | Durable batch runtime job trace, artifact, cancel, and retry APIs. |
| `/v1/media*` | Preview | Local media lifecycle APIs with runtime asset registration for uploads. |
| `/v1/metrics/batch-runtime` | Preview | Durable batch runtime queue, job, stage, and worker telemetry. |
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

JSON errors use OpenAI's envelope:

```json
{
  "error": {
    "message": "Model not found: Example",
    "type": "not_found_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

`code` names what went wrong; it is never the HTTP status. `param` names the request field the error is about, when there is one. The local server and the gateway answer the same failure with the same status, type and code.

| Status | `type` | `code` | Meaning |
|--------|--------|--------|---------|
| `400` | `invalid_request_error` | `invalid_value` | Invalid request shape, field, option or model input (for example audio over the duration limit). A body that is not valid JSON, or JSON of the wrong shape, is this error too. |
| `400` | `invalid_request_error` | `unsupported_parameter` | A valid parameter this deployment cannot honour. |
| `400` | `invalid_request_error` | `context_length_exceeded` | The prompt, or the prompt plus the requested output, does not fit the model's context. |
| `401` | `authentication_error` | `invalid_api_key` | The server is bound to a network address (or is the gateway) and the request lacks a valid API key. |
| `403` | `permission_denied_error` | `permission_denied` | The request is not permitted by policy, or a browser request came from a disallowed `Origin` or `Host`. |
| `404` | `not_found_error` | `model_not_found` | The requested model is not served here (`param` is `model`). |
| `404` | `not_found_error` | `not_found` | Another resource, media object or record was not found. |
| `408` | `server_error` | `timeout` | The request's deadline passed. |
| `409` | `invalid_request_error` | `conflict` | The resource is in use, for example unloading a model that is serving requests. |
| `413` | `invalid_request_error` | `payload_too_large` | The body or an input exceeded a size limit. |
| `415` | `invalid_request_error` | `unsupported_media_type` | The endpoint expected JSON or multipart content but received another content type (writes with `text/plain` or form-encoded bodies are always rejected). |
| `429` | `rate_limit_error` | `rate_limit_exceeded` | A rate, quota or admission limit was reached. |
| `499` | `server_error` | `cancelled` | The request was cancelled. |
| `500` | `server_error` | `server_error` | Runtime, model, storage or server failure. |
| `502` | `server_error` | `bad_gateway` | Gateway only: a worker's answer could not be used. |
| `503` | `service_unavailable_error` | `service_unavailable` | Temporarily out of capacity or not ready: the requested model cannot be loaded while other resident models are busy, the runtime is overloaded, a worker is draining, or a runtime dependency is missing. These responses carry a `Retry-After` header. Also returned by the readiness endpoint when the server is alive but not ready. |

The message of a `500` or `502` can name file paths, devices and internal state, so the server logs it with the request ID and, unless it is a loopback-bound local server, returns only a generic message that includes the request ID (also in the `x-request-id` response header). A loopback-bound local server (the default, and the desktop app) returns the full message, since its only client is this machine.

Enterprise builds can inject authentication and policy hooks. Community builds use local anonymous defaults. If an enterprise hook rejects a request, the response can be `401`, `403`, or `500` before the route handler runs.

### Security And CORS

The server binds `127.0.0.1:8080` by default, so only this machine can reach it and no API key is needed.

To reach it from other devices, bind a network address and set an API key:

```bash
export IZWI_API_KEY="$(openssl rand -hex 32)"
izwi serve --host 0.0.0.0
```

- With a non-loopback host (`0.0.0.0`, `::`, a LAN IP or hostname) the server refuses to start unless `IZWI_API_KEY` is set (16-4096 printable characters, no spaces). The key is read from the environment only, never from the command line.
- Clients send the key as `Authorization: Bearer <key>` or `x-api-key: <key>`. `izwi` CLI commands send `IZWI_API_KEY` automatically, and the web UI asks for the key once and keeps it in a same-site cookie.
- Every route under `/v1` and `/internal` needs the key. The probes (`/livez`, `/readyz`, `/v1/live`, `/v1/ready`, `/internal/live`, `/internal/ready`), the docs and the web UI's static files do not.
- `--allow-unauthenticated-network` (or `IZWI_ALLOW_UNAUTHENTICATED_NETWORK=1`) starts a network bind without a key. Anyone who can reach the port can then download, load and delete models and read or delete history. Use it only behind your own access controls.
- Browser protections apply to every bind: a loopback-bound server only answers `localhost`, `*.localhost`, `127.0.0.1` and `[::1]` `Host` names (this blocks DNS rebinding), and writes and WebSocket upgrades from another site's page are rejected by their `Origin`. Allowed origins are the server's own origin, loopback pages, the desktop app (`tauri://localhost`, `http://tauri.localhost`) and any configured CORS origins.
- `--cors` enables wildcard browser CORS responses, which also lets any web page send requests. Prefer listing origins in `server.cors_origins` (or `IZWI_CORS_ORIGINS`).
- A reverse proxy in front of a loopback-bound server must forward a loopback `Host` header (nginx does by default with `proxy_pass http://127.0.0.1:8080`).

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

Returns the models this server can run now (enabled, and downloaded or
loaded) in OpenAI list format. `created` is when the model's files arrived on
this host (`0` when unknown).

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen3.5-4B",
      "object": "model",
      "created": 1760000000,
      "owned_by": "izwi"
    }
  ]
}
```

`GET /v1/models/{model}`

Returns one of those models in the same object shape. An unknown, disabled or
not-downloaded model is a `404` with `code: model_not_found` and `param:
model`.

The gateway serves both routes for every chat model its worker approvals
name (with `created: 0`), behind the same API key as chat completions. A chat
request names its model in `model` and is routed to that model's workers; a
model the gateway does not serve is a `404` with `code: model_not_found`.
Models added to a shared approvals file after the gateway started are served
after a restart.

The gateway serves chat completions only: `/v1/audio/*` and `/v1/responses`
answer `404` there, because the worker protocol has no unary speech or
Responses invocation. Use a local-role server for those routes (realtime
speech goes through the gateway's realtime socket).

The gateway is the standalone `izwi-gateway` binary, which runs no model, or
`izwi-server --role gateway`; both serve the same API. Its `/openapi.json`
lists only the routes above, plus the probes and `/docs`.

Use `/v1/admin/models` when you need download, load, unload, local path, status, or speech-capability details.

### Chat Completions

`POST /v1/chat/completions`

Basic request:

```json
{
  "model": "Qwen3.5-4B",
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
| `max_tokens`, `max_completion_tokens` | Optional output budgets (`max_completion_tokens` wins). Unset, the reply may use the rest of the context. An explicit budget that does not fit beside the prompt is a `400` (as in vLLM) unless `truncation` is `auto`. |
| `truncation` | Izwi extension, as in the Responses API. `disabled` (default): a prompt that does not fit the context is a `400` with `code: context_length_exceeded`, `param: messages` and the token counts. `auto`: the oldest complete turns are dropped (the system message and newest user message are kept) and an explicit budget is shortened to the remaining context. Through the gateway, workers on worker protocol minor 4 apply this exactly as the local server; older workers still truncate automatically. |
| `stream` | `true` returns SSE chat chunks. The request is admitted before the stream starts, so a request that cannot run (for example `context_length_exceeded`) is a plain HTTP error, not a stream. |
| `stream_options.include_usage` | Sends the usage in one more chunk after the finish chunk, with `choices: []`; every earlier chunk then carries `usage: null`, as in OpenAI. |
| `logprobs`, `top_logprobs` | Per-token logprobs (`top_logprobs` 0–20 alternatives). Returned on `choices[].logprobs.content` in the completion and in each stream chunk. |
| `temperature`, `top_p`, `top_k`, `repetition_penalty`, `presence_penalty` | Optional runtime sampling controls. Explicit values override model profiles. Qwen3.5, Qwen3.6 and Qwen3.8 default to Qwen's published profile for the active thinking mode. LFM2.5 1.2B defaults to Liquid's model-card profile: temperature 0.1 (Instruct) or 0.05 (Thinking), top-k 50, repetition penalty 1.05. On Qwen3.5 the presence penalty counts generated tokens only, and the repetition penalty counts the prompt and generated tokens. |
| `seed` | Optional signed 64-bit sampling seed. The same seed and parameters sample the same tokens on the same backend; without one, each request samples independently. |
| `stop` | A string or an array of up to 4 non-empty strings (each at most 1024 bytes). Generation ends before the first match, the stop text is not returned, and `finish_reason` is `stop`. Streams hold back text that could still begin a stop string until it is settled. Requests with `stop` decode one token per scheduler step. |
| `frequency_penalty` | Rejected in strict OpenAI compatibility mode when non-default. |
| `n` | Only `1` is supported. |
| Ranges | `temperature` 0–2, `top_p` greater than 0 and at most 1, `top_k` -1 or 0 (disabled) or positive, `presence_penalty` and `frequency_penalty` -2–2, `repetition_penalty` 1–2, `max_tokens` and `max_completion_tokens` at least 1. Anything else is a `400` whose `error.param` names the field and whose `error.code` is `invalid_value`. |
| `tools` | OpenAI function tools (`{"type": "function", "function": {"name": …}}`), rendered with the model's tool-calling template. Qwen3.5, Qwen3.6, Qwen3.8 and LFM2.5 have one; tools for another chat model are a `400` (`unsupported_parameter`) rather than silently ignored. |
| `tool_choice` | `auto` (default) offers the tools and parses calls; `none` neither offers them nor parses calls. `required` and named functions need constrained decoding and are a `400` (`unsupported_parameter`); other values are `invalid_value`. |
| `enable_thinking` | Izwi extension for thinking-capable local models. Qwen3.8 defaults to `true`. |
| `reasoning_effort` | Qwen3.8 reasoning level: `xhigh` (default), `medium`, or `low`. Unknown values are rejected. |
| `preserve_thinking` | Preserve prior assistant reasoning in Qwen3.8 history; defaults to `true`. Qwen3.5 history behavior is unchanged. |
| `chat_template_kwargs` | Compatibility object accepting `enable_thinking`, `reasoning_effort`, and `preserve_thinking`. A conflicting direct field is rejected. |
| `reasoning_format` | Izwi extension. `none` (default) leaves `<think>…</think>` reasoning in `content`, including the `<think>` a Qwen prompt opens when thinking is on; `parsed` returns it in `reasoning_content` (message and stream deltas) and keeps only the answer in `content`. |
| `user` | Accepted for compatibility; not used for local auth. |

Through the gateway, a chat request runs on a worker exactly as on the local
server: the same sampling profiles, `seed`, `stop`, `top_k`, penalties,
`logprobs`/`top_logprobs`, `response_format: json_object`, output budget and
`truncation`. Each of these controls needs a worker on worker protocol minor
4 that advertises it; the gateway routes the request only to such workers, and
when none is ready it returns a `400` with `code: unsupported_parameter` and
`param` naming the control, never dropping it. `tools` need a worker that
advertises tool support; the gateway parses the calls in the worker's text
exactly as the local server does. The reasoning controls are not yet
supported through the gateway (400).

Gateway limits and headers:

- Every gateway chat response carries OpenAI's rate-limit headers for the
  caller's tenant: `x-ratelimit-limit-requests`,
  `x-ratelimit-remaining-requests` and `x-ratelimit-reset-requests` (for
  example `30s`), plus the `-tokens` set when the operator configured a
  tokens-per-minute limit. A `429` or `503` carries `Retry-After` in seconds.
- With a token limit, a request is admitted against an estimate (its output
  budget plus about one token per four prompt bytes) and charged what the
  worker reports it used. A request whose estimate exceeds the remaining
  budget is a `429` with message `Gateway tenant token rate limit exceeded`.
- A request body that does not arrive within the gateway's body-read deadline
  (10 seconds by default) is a `408` with `code: timeout`.
- During a gateway shutdown, a stream that has not finished by the end of the
  drain receives a `503` error event (`code: service_unavailable`) and
  `data: [DONE]`. Retry it; nothing was replayed.

Compatibility profile:

```bash
IZWI_OPENAI_COMPAT_PROFILE=strict   # default
IZWI_OPENAI_COMPAT_PROFILE=relaxed
```

Completion ids are `chatcmpl-<hex>`, and every chunk of a stream shares one
id. Streaming sequence (with `stream_options.include_usage`):

```text
data: {"id":"chatcmpl-…","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}],"usage":null}
data: {"id":"chatcmpl-…","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}],"usage":null}
data: {"id":"chatcmpl-…","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":null}
data: {"id":"chatcmpl-…","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":9,"completion_tokens":2,"total_tokens":11,"prompt_tokens_details":{"cached_tokens":0}}}
data: [DONE]
```

If generation fails after the stream started, the last event before
`[DONE]` is the OpenAI error object, `{"error": {"message", "type", "param",
"code"}}`, with the type the same failure has as an HTTP error (the OpenAI
SDKs raise it as `APIError`). There is no finish chunk then.

Tool behavior:

- Assistant messages may include `tool_calls`; a `tool` message's
  `tool_call_id`, when given, must match a call of the preceding assistant
  message (`400` otherwise). The templates read tool results in order.
- Calls are parsed only when the request offered tools. They are returned as
  `message.tool_calls` (ids `call_<hex>`) with `finish_reason: "tool_calls"`,
  and the call markup never appears in `content`.
- Streamed, calls arrive as `delta.tool_calls`: the first fragment of each
  carries its `index`, `id`, `type` and `function.name`, and the
  `function.arguments` fragments concatenate to the JSON arguments. A Qwen call
  is sent as soon as its name is known and its arguments one parameter at a
  time; an LFM2.5 call list is sent when it closes. Text that might open a
  call is held back until the next delta settles it, and a block that does
  not parse is sent as text.
- LFM2.5 models use their own format: tools are listed in the system turn,
  `tool` messages become native tool turns, and calls the model writes as
  `<|tool_call_start|>[name(arg="v")]<|tool_call_end|>` are returned as
  `tool_calls`. The Qwen3.5 family uses Qwen's XML tool format.
- LFM2.5 1.2B supports 32,768 tokens of context, the length on Liquid's model
  cards, on every backend.

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
  "model": "Kokoro-82M",
  "input": "Hello from Izwi.",
  "voice": "af_heart",
  "response_format": "wav"
}
```

Request fields:

| Field | Notes |
|-------|-------|
| `model` | Required TTS model variant. |
| `input` | Required text to synthesize. |
| `voice` | Built-in voice/speaker name where the model supports presets. |
| `response_format` | Native OSS formats: `wav` (default), `pcm`, `pcm16`, `pcm_i16`, `raw_i16`, `raw_f32`, `pcm_f32`. Recognized compressed names `mp3`, `opus`, `ogg`, `aac`, and `flac` require explicit fallback opt-in because compressed encoders are not bundled. |
| `allow_format_fallback` | Optional boolean. When `true`, recognized compressed `response_format` values return WAV bytes with `X-Actual-Response-Format: wav`, `X-Response-Format-Fallback`, and `Warning` headers. When omitted or `false`, compressed formats return `400`. |
| `speed` | Optional model-dependent speed control. |
| `language` | Optional language hint such as `English`, `Chinese`, or `Auto`. `Chatterbox-Multilingual-500M` speaks the language it names, as a code or English name (`fr`, `French`), and returns `400` for one it does not support; `Chatterbox-500M` returns `400` for any language but English. `Maya2-Native-2.5B` takes one of its 12 language codes (`hi`, `ta`, …), or none or `auto` to read the language from the text's script, and returns `400` for anything else. `GET /v1/admin/models` lists a Chatterbox or Maya 2 Native model's languages in `speech_capabilities.languages`. |
| `temperature`, `top_k` | Optional sampling controls. |
| `exaggeration` | Chatterbox only. Emotion intensity, 0 to 2 (default `0.5`). |
| `cfg_weight` | Chatterbox only. Classifier-free guidance weight, 0 to 1 (default `0.5`); `0` turns guidance off. For expressive or fast-talking reference voices, try about `0.3` with `exaggeration` around `0.7` or higher. |
| `min_p` | Chatterbox only. Min-p sampling threshold, 0 to 1 (default `0.05`). |
| `max_tokens`, `max_output_tokens` | Optional output token budget aliases. |
| `instructions` | Voice-design prompt for voice-design models; for `Maya1-3B`, the description of the voice to speak with, replacing `voice`. |
| `reference_audio` | Base64 audio for voice cloning. |
| `reference_text` | Transcript of the reference audio. Optional for Chatterbox, which ignores it. |
| `reference_language` | `Maya2-Native-2.5B` only. The language the reference audio is spoken in, one of its 12 codes; unset, it is read from the transcript's script. Any other model returns `400` if it is set. The speech history create routes accept it too. |
| `saved_voice_id` | Server-side saved voice reference to reuse. |
| `stream`, `stream_format` | `stream: true` or `stream_format: "sse"` enables SSE audio chunks. Text one request cannot speak streams as consecutive long-form segments, as a final-only request splits it. |

`exaggeration`, `cfg_weight` and `min_p` are also accepted by the speech
history create routes. Any other model returns `400` if one is set, and a value
out of range returns `400`.

`Chatterbox-500M` specifics:

- It speaks with its built-in `default` voice, or clones from `reference_audio`
  (or a saved voice) alone.
- Its other sampling defaults are `temperature` 0.8, repetition penalty 1.2 and
  top-p 1.0.
- With `stream: true` it streams audio as T3 samples speech tokens: the first
  second arrives once 25 tokens are sampled, then windows of 2 and 4 s (about
  2 s to first audio for one request on an Apple M1 Pro with Metal). It runs
  slower than real time there, so chunks can arrive seconds apart; buffer
  before playing. A streamed request's audio matches the final-only response
  for the same `seed` in length and words, not sample for sample.
- `speed` and `instructions` (voice descriptions) each return `400`.
- One request speaks at most 1,000 speech tokens (about 40 s). Without
  `max_tokens`, Izwi sets a budget from the text (50 tokens, plus 2 per
  character, plus 16 per digit) and splits longer text between whole sentences
  into long-form segments joined by a short faded pause, so long passages are
  spoken in full. An explicit `max_tokens` is used as given, without splitting.
- Every utterance carries Resemble AI's imperceptible PerTh watermark. Set
  `IZWI_CHATTERBOX_WATERMARK=false` before the model loads to turn it off.

`Maya1-3B` specifics:

- Its voice comes from a description: `instructions`, else the built-in voice
  `voice` names (`ava`, the default, `leo`, `nora`, `james`, `priya`, `arjun`,
  `zoe`, `omar`, `robot`; OpenAI's names map to the nearest of these). A
  reference voice returns `400`.
- Tags in `input` (`<laugh>`, `<sigh>`, `<whisper>` and 17 more, listed in
  `speech_capabilities.inline_tags` of `GET /v1/admin/models`) are performed
  where they stand; any other `<word>` returns `400`, as the model would read
  it aloud.
- Sampling defaults follow the model card: `temperature` 0.4, top-p 0.9,
  top-k 50, repetition penalty 1.1. `seed` makes a request reproducible.
- With `stream: true` it streams audio in windows as the decoder writes it;
  streamed and final-only audio are the same samples for the same `seed`, and
  so is a request served beside others. It runs slower than real time on
  Apple Silicon (first audio in about 2 s on an M1 Pro, then windows seconds
  apart), so buffer before playing.
- One request speaks at most 2,048 decoder tokens (about 25 s). Without
  `max_tokens`, Izwi sets a budget from the text and splits text longer than
  about 160 characters into sentences of up to about 11 s, each with the same
  description.

`Maya2-Native-2.5B` specifics:

- It speaks Assamese, Bengali, English, Gujarati, Hindi, Kannada, Malayalam,
  Marathi, Odia, Punjabi, Tamil and Telugu (codes `as`, `bn`, `en`, `gu`,
  `hi`, `kn`, `ml`, `mr`, `or`, `pa`, `ta`, `te`). Without `language`, or with
  `auto`, it reads the language from the text's main script: Latin letters as
  English, Devanagari as Hindi and Bengali script as Bengali, so name `mr` or
  `as` for Marathi or Assamese text. `GET /v1/admin/models` marks this with
  `speech_capabilities.language_auto_detect`, and cloning across languages
  with `speech_capabilities.cross_lingual_reference`.
- Its voice is one of 20 built-in voices that `voice` names (`Ahana`, the
  default, `Vaidehi`, `Anirban`, …; OpenAI's names map to one of them, see
  [Voice Presets](/models/voice-presets#maya-2-native)), or a clone of
  `reference_audio` (or a saved voice) with its exact `reference_text`, which
  is required. The reference must last 0.5 to 30 s. A clone can speak any of
  the 12 languages: `reference_language` names the language of the reference
  and `language` the language of the output. A saved voice's
  `reference_language` applies unless the request sets its own.
  `instructions` and `speed` each return `400`.
- Tags in `input` are lowercase words in square brackets, such as `[laughs]`,
  `[sighs]` or `[whispers]`. The model performs them on a best-effort basis.
  `speech_capabilities.inline_tags` of `GET /v1/admin/models` lists the eight
  that Maya Research names; any other lowercase `[tag]` is passed to the model
  rather than refused.
- Numbers, currency, dates, times, phone numbers, URLs and e-mail addresses are
  spoken as words in the request's language by default. Set
  `normalize_text: false` to send the text as written.
- Sampling defaults follow Maya Research's release: `temperature` 0.9 (0.1 to
  1.5), top-k 64, top-p 1.0, and a presence penalty of 0.8 (0 to 5) on
  recently sampled codes, which `repetition_penalty` sets. `seed` makes a
  request reproducible.
- With `stream: true` it streams audio as the Mimi codec decodes it: the first
  80 ms frame, then windows of 320 ms. Streamed and final-only audio are the
  same samples for the same `seed`, and a request's audio is the same whether
  it runs alone or beside other requests.
- One request speaks at most 1,500 frames (120 s). Without `max_tokens`, Izwi
  sets a budget from the text and splits text longer than 280 characters into
  whole sentences of up to 280 characters (about 20 s each). Sentences also end
  at a danda (`।` or `॥`), and a tag is never sent as a segment on its own.

`Chatterbox-Multilingual-500M` takes the same fields and speaks the `language`
the request names: Arabic, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Italian, Korean, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swahili, Swedish and Turkish (codes `ar`, `zh`, `da`, `nl`, `en`, `fi`,
`fr`, `de`, `el`, `he`, `hi`, `it`, `ko`, `ms`, `no`, `pl`, `pt`, `ru`, `es`,
`sw`, `sv`, `tr`). Without `language`, or with `Auto`, it speaks English.
Japanese returns `400` until Izwi has its text processing.

Non-stream response:

- Body is binary audio.
- Content type follows the actual generated format.
- `X-Requested-Response-Format` and `X-Actual-Response-Format` are exposed. Explicit fallbacks also include `X-Response-Format-Fallback` and an HTTP `Warning` header.

SSE events:

| Event | Fields |
|-------|--------|
| `audio.started` | `request_id`, `sample_rate`, `audio_format`, optional explicit fallback note in `error`. |
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
| `model` | Optional ASR, Granite Speech, Voxtral offline transcription, or audio-chat model variant. Voxtral realtime is planned separately. |
| `language` | Optional language hint. Granite Speech accepts IBM's codes and names (4.1 base and 5.0: en, fr, de, es, pt, ja; 4.1 plus drops ja), names it in the instruction, and rejects other languages with `400`. |
| `target_language` | Izwi extension. Translate the speech into this language instead of transcribing it. Only models with speech translation accept it (`Granite-Speech-5.0-2B` and `Granite-Speech-5.0-2B-GGUF`: into English from fr, de, es, pt, ja, and from English into fr, de, es, pt, ja, it, zh); others return `400`. When `language` is also set, the pair must be one of these. The response `language` is the target. Word timestamps are not available for a translation (`timestamp_granularities[]` with it returns `400`). |
| `keywords` | Izwi extension. Terms to bias recognition toward: repeat the field, or separate terms with commas or newlines (JSON: an array). Only models with keyword biasing accept it (both Granite Speech 5.0 releases, up to 2,000 terms of at most 128 characters); others return `400`. A long list shortens the audio one invocation can hold. |
| `response_format` | `json`, `verbose_json`, `text`, `srt`, or `vtt`. Default `json`. |
| `stream` | `true`, `1`, `yes`, or `on` enables SSE. |
| `timestamp_granularities[]` | Optional `word`, `segment`, or both. Requires `response_format=verbose_json`; model-provided timestamps are used before forced alignment fallback (Parakeet reports word times from its TDT decoder; `Granite-Speech-4.1-2B-Plus` runs its native timestamp mode; base Granite text is timed by the aligner). Without it, `verbose_json` returns `segments` only, as OpenAI does. |
| `aligner_model` | Optional forced-aligner model for timestamp generation. Defaults to `Qwen3-ForcedAligner-0.6B`. |
| `prompt` | Optional ASR prompt/context. Granite Speech uses it only when it is a Granite instruction (it starts with `<|audio|>`, `transcribe the speech`, `can you transcribe`, `translate the speech`, `Speaker attribution:` or `Timestamps:`; 5.0 also takes `The speech is in`) or `keywords: a, b` for keyword biasing; other text is ignored and reported as `prompt_ignored` in the diagnostics. |
| `max_tokens` | Optional ASR decoder token budget. |
| `temperature` | Accepted for compatibility; currently ignored by native ASR models. |

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
  "words": [
    { "word": "Hello", "start": 0.0, "end": 0.45 }
  ],
  "segments": [
    { "id": 0, "start": 0.0, "end": 3.5, "text": "Hello, this is a transcription test." }
  ],
  "processing_time_ms": 812.4,
  "rtf": 0.23,
  "izwi_asr_diagnostics": {
    "model_family": "voxtral",
    "execution": {
      "device_kind": "Metal",
      "dtype": "F32",
      "cache": {
        "page_size": 64,
        "dense_decode_enabled": false,
        "kv_quantization": "none"
      }
    }
  }
}
```

Audio without speech returns an empty transcript without running the model, for the models that otherwise hallucinate a short phrase on silence or noise (Whisper, Qwen3-ASR, Granite Speech, VibeVoice, Voxtral, LFM2.5 Audio). An input counts as speech when some second of it holds 200 ms of frames the VAD scores as speech that are also 6 dB above the input's own noise floor. Silence and steady noise (hum, fan, pink or white noise) at any level are skipped this way; tonal music and crowd babble are not, and reach the model. `verbose_json` reports a skipped input as `izwi_asr_diagnostics.no_speech_gate`, with the speech evidence found (`evidence_ms`) and needed (`required_evidence_ms`). `IZWI_VAD_SNR_MARGIN_DB` sets the margin (default `6`); `0` turns the noise-floor check off.

`verbose_json` always includes `segments`. Without word or segment timestamps from the model or the aligner, each long-form chunk becomes one segment covering its time range, and a short file gets one segment spanning the whole audio. `words` appears only when `timestamp_granularities[]` includes `word`. If forced alignment fails, the transcript is still returned with segments, without `words`, and the failure is reported as `izwi_asr_diagnostics.alignment_error`.

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

### Audio Alignment

`POST /v1/audio/align`

Forced alignment accepts JSON or multipart input and aligns reference text to audio at word level.

JSON request:

```json
{
  "audio_base64": "<base64-audio>",
  "text": "Hello world, this is a test.",
  "model": "Qwen3-ForcedAligner-0.6B",
  "language": "English",
  "response_format": "json"
}
```

Multipart fields:

| Field | Notes |
|-------|-------|
| `file` or `audio` | Uploaded audio file. |
| `audio_base64` | Base64 audio alternative. |
| `text` or `reference_text` | Required reference text to align. |
| `model` | Optional forced-aligner model variant. Defaults to `Qwen3-ForcedAligner-0.6B`. |
| `language` | Optional language hint. |
| `response_format` | `json`, `verbose_json`, or `text`. Default `json`. |

`json` response:

```json
{
  "alignments": [
    { "word": "Hello", "start": 0.0, "end": 0.45 },
    { "word": "world", "start": 0.5, "end": 0.95 }
  ],
  "duration": 0.95
}
```

`verbose_json` adds `model`, `language`, `word_count`, and `processing_time_ms`.

Example multipart request:

```bash
curl -X POST http://localhost:8080/v1/audio/align \
  -F "file=@speech.wav" \
  -F "text=Hello world" \
  -F "model=Qwen3-ForcedAligner-0.6B" \
  -F "response_format=verbose_json"
```

### Responses

`POST /v1/responses`

Preview OpenAI-compatible Responses API shape.

```json
{
  "model": "Qwen3.5-4B",
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
| `truncation` | `disabled` (default) or `auto`; same behavior as chat completions. |
| `stream` | `true` returns SSE events. |
| `metadata`, `user` | Stored or accepted for compatibility. |
| `temperature`, `top_p`, `top_k`, `repetition_penalty`, `presence_penalty` | Optional runtime sampling controls. |
| `store` | `false` skips process-local retention. Default retains completed records. |
| `tools`, `tool_choice` | Function tools in the Responses form (`{"type": "function", "name", "parameters"}`) or the chat completions form. A tool call in the answer is a `function_call` output item (`id` `fc_<hex>`, `call_id` `call_<hex>`, `name`, JSON `arguments`) after the message, which is left out when the answer only calls tools; streaming sends `response.output_item.added`, `response.function_call_arguments.delta`/`.done` and `response.output_item.done` for it. Send the result back as a `{"type": "function_call_output", "call_id", "output"}` input item, after the `function_call` item itself. As in chat completions, `tool_choice` `required` and named functions are refused. |
| `enable_thinking`, `reasoning_effort`, `preserve_thinking`, `chat_template_kwargs` | Same behavior as chat completions. Response ids are `resp_<hex>` and message ids `msg_<hex>`. |

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

Non-streaming first-party speech-to-text and text-to-speech create routes are
backed by durable runtime jobs. Route-specific records remain the compatibility
contract for web and desktop clients; the generic `/v1/jobs/*` APIs expose
runtime traces, artifacts, cancellation, and retry controls once a runtime job
ID is known.

### Route Rename Migration

The following preview route names were replaced by canonical names. The old
runtime routes were removed.

| Removed route family | Current route family |
|----------------------|----------------------|
| `/v1/text-to-speech-generations` | `/v1/text-to-speech` |
| `/v1/voice-design-generations` | `/v1/voice-designs` |
| `/v1/voice-clone-generations` | `/v1/voice-clones` |
| `/v1/transcriptions` | `/v1/speech-to-text/jobs?job_kind=transcription` |
| `/v1/transcriptions/{record_id}` | `/v1/speech-to-text/jobs/{record_id}?job_kind=transcription` |
| `/v1/transcriptions/{record_id}/audio` | `/v1/speech-to-text/jobs/{record_id}/audio?job_kind=transcription` |
| `/v1/transcriptions/{record_id}/summary/regenerate` | `/v1/speech-to-text/jobs/{record_id}/summary/regenerate?job_kind=transcription` |
| `/v1/transcriptions/jobs` | `/v1/speech-to-text/jobs` |
| `/v1/transcription/realtime/ws` | `/v1/speech-to-text/realtime/ws` |
| `/v1/audio/diarize` | `/v1/speech-to-text/jobs?job_kind=diarization` |
| `/v1/audio/diarizations` | `/v1/speech-to-text/jobs?job_kind=diarization` |

The speech history and speech-to-text renames keep response payloads, record IDs,
pagination, audio download behavior, and SSE event names unchanged. The removed
direct saved transcription routes now use `job_kind=transcription` on the
persisted speech-to-text job flow. The removed direct audio diarization routes
use the same job flow with `job_kind=diarization`: create a job, poll the
returned record until `processing_status` is `ready`, and then read the
diarization fields from that job record.

Direct `/v1/diarizations*` routes remain supported first-party APIs. Use
`/v1/speech-to-text/jobs?job_kind=diarization` when an app wants a unified
speech-text list across transcription and diarization. Use `/v1/diarizations*`
when an app wants diarization-specific resource names and does not need to mix
transcription records into the same collection.

### Speech-Text Jobs

Canonical saved transcription, speaker-attributed ASR, and diarization job
routes:

| Method | Path | Notes |
|--------|------|-------|
| `GET` | `/v1/speech-to-text/jobs` | List jobs. Supports `limit`, `cursor`, and `job_kind=transcription|speaker_attributed_asr|saa|diarization|all`. |
| `POST` | `/v1/speech-to-text/jobs` | Create transcription, speaker-attributed ASR, or diarization job. Multipart uploads allowed. |
| `GET` | `/v1/speech-to-text/jobs/{record_id}` | Fetch one job. `job_kind` can disambiguate. |
| `PATCH`, `PUT` | `/v1/speech-to-text/jobs/{record_id}` | Update editable metadata such as title, transcript fields, speaker labels, or summary state depending on job kind. |
| `DELETE` | `/v1/speech-to-text/jobs/{record_id}` | Delete job and associated stored media. |
| `GET` | `/v1/speech-to-text/jobs/{record_id}/audio` | Fetch stored source audio. |
| `POST` | `/v1/speech-to-text/jobs/{record_id}/reruns` | Re-run diarization from stored source audio. |
| `POST` | `/v1/speech-to-text/jobs/{record_id}/cancel` | Cancel an in-flight diarization job. |
| `POST` | `/v1/speech-to-text/jobs/{record_id}/summary/regenerate` | Regenerate transcription, speaker-attributed ASR, or diarization summary. |

The `job_kind` query parameter is important for shared IDs and for clients that
want a specific record family. `speaker_attributed_asr` and `saa` select the
Granite Speech speaker-turn transcript mode.

For transcription job creation, JSON and multipart requests accept
`generate_summary`. It defaults to `false`; set it to `true` to generate an AI
summary automatically after the transcript finishes. Records created without an
automatic summary can still use
`POST /v1/speech-to-text/jobs/{record_id}/summary/regenerate?job_kind=transcription`
later.

Transcription jobs also accept `target_language` and `keywords`, with the
same meaning and limits as on `/v1/audio/transcriptions`: multipart repeats
`keywords` or separates terms with commas or newlines; JSON takes an array.
Models whose `route_capabilities` report `speech_translation` or
`keyword_biasing` accept them (both Granite Speech 5.0 releases); others return `400`,
as does a translation with `include_timestamps` or in speaker-attributed mode.
The record stores both as `target_language` and `keywords`, keeps the spoken
`language` the job named, and a retry asks for the same translation.

For speaker-attributed ASR, use:

```http
POST /v1/speech-to-text/jobs?job_kind=speaker_attributed_asr
```

SAA requests use the transcription store but require
`Granite-Speech-4.1-2B-Plus`. JSON and multipart requests accept
`model_id`/`model`, `language`, `generate_summary`, `min_speakers`, and
`max_speakers`. SAA does not support streaming, timestamp alignment, or
`include_timestamps`; the server clears aligner/timestamp fields for this mode.

Non-streaming transcription and SAA creates return `202` with the route record
as before, then run through the durable batch worker. The runtime records the
source media asset, input artifact, execution stage, transcript text asset, and
output transcript artifact. Streaming ASR remains on the SSE path and does not
use the batch runtime in this phase.

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

### Speech History

All three speech history families share list/create, member, audio, pagination, and deletion behavior. Create routes can generate audio and persist the resulting record.

| Route family | Purpose |
|--------------|---------|
| `/v1/text-to-speech` | Plain TTS history. |
| `/v1/voice-designs` | Voice-design prompt records. |
| `/v1/voice-clones` | Reference-audio voice clone records. |

Routes:

| Method | Path pattern |
|--------|--------------|
| `GET`, `POST` | `/v1/{family}` |
| `GET`, `DELETE` | `/v1/{family}/{record_id}` |
| `GET` | `/v1/{family}/{record_id}/audio` |

Non-streaming `/v1/text-to-speech` creates return `202` with a pending history
record, then run through the durable batch worker. The runtime records the
input text asset and generated WAV media asset/artifact before projecting the
finished state back into the speech history record. Streaming TTS, voice
design, and voice cloning stay on their route-specific execution paths.

Streaming create responses emit JSON SSE events with an `event` field:

| Event | Notes |
|-------|-------|
| `created` | Includes the persisted record shell. |
| `start` | Includes `request_id`, `sample_rate`, and `audio_format`. |
| `chunk` | Includes `request_id`, `sequence`, `audio_base64`, and `sample_count`. |
| `final` | Includes generation stats and the completed record. |
| `error` | Includes an error string. |
| `done` | Terminal stream marker. |

### Durable Runtime Jobs

Preview runtime job routes expose the durable batch execution graph for
first-party ASR and non-streaming text-to-speech jobs:

| Method | Path | Notes |
|--------|------|-------|
| `GET` | `/v1/jobs/{job_id}` | Fetch the runtime job, ordered stages, and artifacts. |
| `GET` | `/v1/jobs/{job_id}/artifacts` | Fetch only artifact lineage for the job. |
| `POST` | `/v1/jobs/{job_id}/cancel` | Cancel a non-terminal job and update the linked route record when one exists. |
| `POST` | `/v1/jobs/{job_id}/retry` | Requeue a failed or cancelled job that has retryable stages. |

Runtime jobs and their stages use the statuses `queued`, `running`, `retrying`,
`completed`, `failed`, and `cancelled`. `completed`, `failed`, and `cancelled`
are terminal for workers. A manual retry requeues the job while the job's own
retry budget (`max_attempts`) is not used up and every failed or cancelled
stage still has attempts left; a stage that used all of its attempts is not
revived, so submit a new job instead. Stages and artifacts include IDs,
capability/model metadata, attempt counts, timing/error fields, media/text
asset references, and storage keys when available.

### Saved Voices

Reusable voice clone references:

| Method | Path | Notes |
|--------|------|-------|
| `GET` | `/v1/voices` | List saved voices with cursor pagination. |
| `POST` | `/v1/voices` | Create a saved voice from reference audio (transcript optional) or a generated voice source. |
| `GET` | `/v1/voices/{voice_id}` | Fetch saved voice metadata. |
| `DELETE` | `/v1/voices/{voice_id}` | Delete saved voice and audio. |
| `GET` | `/v1/voices/{voice_id}/audio` | Fetch saved reference audio. |

Use `saved_voice_id` on `/v1/audio/speech` or first-party generation routes to reuse a saved voice without resending reference audio.

A voice saved without a transcript works with `Chatterbox-500M` and `Chatterbox-Multilingual-500M`, which clone from audio alone. Models that need a transcript (Qwen3-TTS Base, VibeVoice, Fish Audio S2 Pro and Maya 2 Native) return `400` for it, asking for a voice saved with one.

`POST /v1/voices` also takes an optional `reference_language`: the language the
reference is spoken in, one of Maya 2 Native's 12 codes (any other value
returns `400`). A `Maya2-Native-2.5B` request that uses the voice inherits it
unless the request sets its own `reference_language`.

Saved voice records include a local permission baseline:
`permission_scope`, `consent_status`, `allowed_uses`,
`permission_provenance`, and `permission_revoked_at`. Newly created local
voices default to `permission_scope: "local_owner"`,
`consent_status: "granted"`, and `allowed_uses` containing `local_tts`.
Migrated local voices use `permission_scope: "legacy_local"`. Reuse through
`saved_voice_id` returns `403` when consent is not granted, the voice is
revoked, or `local_tts` is not allowed.

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
| `format=wav|raw_i16|raw_f32` | Requested export format. |
| `segment_ids=a,b,c` | Export selected segments in order. |
| `pause_ms=500` | Silence between segments, up to 5000 ms. |

Export refuses segments that aren't rendered or changed since they were
rendered, and names them by paragraph number. Takes at different sample
rates are resampled to the highest one.

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

The server renders jobs in the background, one segment at a time, whether or
not a client is connected, and resumes them after a restart. `POST` takes
`{"segment_ids": [...]}`, or an empty body to queue every segment whose audio
is missing, stale or failed. `PATCH` accepts only `{"status": "cancelled"}`:
queued segments leave the queue and a segment already rendering finishes.
A project's `active_render_job` is the job in progress.

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
| `GET` | `/v1/studio/projects/{project_id}/segments/{segment_id}/audio` |

Each segment has a `render_state`: `not_rendered`, `rendered`, `stale` (the
audio was rendered from other text, voice, speed or pronunciations), `queued`,
`rendering`, or `failed` (with `render_error`). Edits keep a segment's audio
and make it stale. `PATCH .../segments/{segment_id}` with
`"use_project_voice": true` drops the segment's own voice. Pronunciation rules
match whole words in any letter case.

Render-job statuses are queued, running, completed, failed and cancelled. Clients should preserve unknown statuses.

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
| `reasoning_effort`, `preserve_thinking`, `chat_template_kwargs` | Qwen3.8 reasoning controls; compatible kwargs must not conflict with direct fields. |
| `top_k`, `repetition_penalty`, `presence_penalty` | Optional sampling controls. |
| `truncation` | `disabled` (default) or `auto`. The chat UI sends `auto`, so a long thread drops its oldest turns instead of failing. |

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
| `POST` | `/v1/voice/sessions` | Create a persisted session shell. Defaults to the default profile, `modular` mode, and the profile system prompt. |
| `GET` | `/v1/voice/sessions/{session_id}` | Fetch a session with turns. |
| `PATCH` | `/v1/voice/sessions/{session_id}` | Update `system_prompt` and/or set `ended: true`. |
| `DELETE` | `/v1/voice/sessions/{session_id}` | Delete a session and its stored turns. |
| `GET` | `/v1/voice/sessions/{session_id}/turns` | List only the stored turns for a session. |
| `POST` | `/v1/voice/sessions/{session_id}/end` | Mark a session ended. |
| `GET` | `/v1/voice/sessions/{session_id}/export?format=json\|text` | Export session metadata, turn metadata, and a transcript view. |

Observational memory is applied to modular voice conversations. Updates are stored locally and can be cleared by the user.

### Media

Media lifecycle routes:

| Method | Path | Notes |
|--------|------|-------|
| `GET` | `/v1/media?limit=100` | List media objects when the server is using local media storage; provider-backed storage can return `501` if listing is unavailable. |
| `POST` | `/v1/media` | Upload a base64 media object. |
| `GET` | `/v1/media/{path}` | Download a persisted media object. |
| `DELETE` | `/v1/media/{path}` | Delete a persisted media object. |

Serves persisted media objects used by chat attachments and local workflows.
The server route is a catch-all, so `{path}` can contain nested segments such as
`images/example.png` or `chat/thread-1/attachment.wav`.

Upload request:

```json
{
  "data_base64": "UklGRiQAAABXQVZF...",
  "content_type": "audio/wav",
  "filename": "utterance.wav",
  "namespace": "voice/session-1"
}
```

`audio_base64` is accepted as an alias for `data_base64`, and data URLs such as
`data:audio/wav;base64,...` are accepted. Upload responses include `path`,
`url`, `content_type`, `filename`, `size_bytes`, `media_asset_id`,
`canonical_media_asset_id`, `canonical_path`, and `canonical_url`.

For audio uploads, the server preserves the original object at `path`/`url`,
registers it as a `media_assets` row, and writes a canonical 16 kHz mono WAV
derivative when decoding succeeds. The canonical derivative has its own media
asset ID and media path/URL. Non-audio uploads keep the existing object
lifecycle behavior and leave canonical audio fields empty.

Rules:

- The path is relative to the media root.
- Nested paths are allowed.
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
| `GET` | `/v1/metrics/batch-runtime` | JSON durable batch runtime queue, job, stage, and worker telemetry. |
| `GET` | `/v1/metrics/prometheus` | Prometheus text format. |
| `GET` | `/internal/metrics` | Internal alias. |
| `GET` | `/internal/metrics/batch-runtime` | Internal alias for batch runtime telemetry. |
| `GET` | `/internal/metrics/prometheus` | Internal alias. |

Batch runtime metrics include `queued_stages`, `jobs_by_status`,
`stages_by_status`, and a local worker snapshot. Prometheus output appends
`izwi_batch_runtime_queued_stages`, `izwi_batch_runtime_jobs`,
`izwi_batch_runtime_stages`, and `izwi_batch_runtime_worker_running`.

Latencies are Prometheus histograms (cumulative `_bucket{le=...}`, `_sum`,
`_count`) in seconds, so they aggregate across replicas and `histogram_quantile`
works on them. The JSON snapshot keeps its windowed averages and percentiles
for the UI. Labels are bounded: model names come from the model catalog,
families keep at most 64 series (the rest share `other`), and no request or
tenant IDs are used.

| Metric | Labels | Meaning |
|---|---|---|
| `izwi_request_duration_seconds` | `model`, `backend`, `task`, `outcome` | Server admission to the terminal output. |
| `izwi_request_ttft_seconds` | `model`, `backend`, `task` | Server admission to the first user-visible output. |
| `izwi_request_queue_wait_seconds` | `model`, `backend`, `task` | Server admission until the engine first schedules the request, including the runtime coordinator, input preparation and model residency. |
| `izwi_request_prefill_seconds` | `model`, `backend`, `task` | Engine prefill time of one request. |
| `izwi_request_prompt_tokens_total`, `izwi_request_generation_tokens_total` | `model`, `task` | Counters for completed requests (generation counts audio frames for speech models). |
| `izwi_engine_inter_token_latency_seconds` | `model`, `task` | Gap between successive committed tokens of one request; a commit of k tokens after a gap g counts k gaps of g/k. |
| `izwi_engine_step_host_seconds` | `phase` (`prepare`, `commit`, `deliver`) | Host time of engine step phases under the step gate. |
| `izwi_engine_batch_rows`, `izwi_engine_batch_tokens` | none | Rows and work units (tokens, or frames for speech) per dispatched physical batch. |
| `izwi_engine_model_calls_per_step` | none | Model calls one engine step dispatched: one per tensor batch or serial row, one per row of a request-parallel batch. |
| `izwi_engine_mixed_rows`, `izwi_engine_mixed_tokens` | `phase` (`decode`, `prefill`) on rows | Rows of each phase and tokens per committed mixed prefill+decode batch (models certified for mixed steps). |
| `izwi_engine_device_idle_seconds` | `device` (`cpu`, `metal`, `cuda:0`) | Time the device waited between the end of one model step and the start of the next while the engine had work: the host gap (commit, delivery, prepare, dispatch) between steps. |
| `izwi_engine_device_queue_wait_seconds` | `device`, `class` (workload class) | Time a model dispatch waited on its device's queue before it started. Control work (KV page operations, loads, frees) runs first; dispatches of different models share the device by workload-class weight, between jobs. |
| `izwi_engine_device_ops_off_thread_total` | none | Counter of device operations (model calls, KV page operations, state allocation and frees) that ran off their device's thread. Zero in a healthy process. |
| `izwi_runtime_stage_duration_seconds` | `stage`, `workload_class` | Duration of runtime pipeline stages. |
| `izwi_voice_turn_first_audio_seconds` | none | Realtime voice: endpoint decision to the turn's first assistant audio. |
| `izwi_voice_endpointing_latency_seconds`, `izwi_voice_asr_first_partial_seconds`, `izwi_voice_asr_final_seconds`, `izwi_voice_llm_first_token_seconds`, `izwi_voice_tts_first_audio_seconds`, `izwi_voice_barge_in_latency_seconds`, `izwi_voice_session_duration_seconds` | none | Realtime voice stage latencies; `izwi_voice_metric_contract_info` lists each with its description. |
| `izwi_voice_audio_egress_underruns_total`, `izwi_voice_audio_ingress_dropped_frames_total` | none | Assistant audio sent after playback had drained; input frames missing from the client's sequence. |

The windowed `quantile="avg"|"p50"|"p95"` gauges (`izwi_latency_*`,
`izwi_runtime_stage_duration_ms`, `izwi_runtime_workload_*`) were removed in
favour of these histograms. The latency breakdown a request returns also
reports `pre_engine_ms`, the time between runtime admission and engine
insertion.

Requests may carry a W3C `traceparent`; the server continues that trace and
logs `trace_id`, `span_id` and `parent_span_id` on the request. See the
production serving runbook for JSON logs and the optional OTLP span export
(`IZWI_OTLP_TRACES_ENDPOINT`).

### Admin Model Management

Preview local admin routes. Use these routes as the OSS model lifecycle and discovery surface for voice apps: each model record includes local status, broad modalities, speech-generation capabilities when present, and route-level capability booleans.

When a model is loaded, `runtime_diagnostics.family` preserves its runtime
family identity. Qwen3.8 reports `qwen38_chat`; clients must not collapse that
value into the distinct `qwen35_chat` family.

For a CUDA-loaded Qwen3.8 model,
`runtime_diagnostics.family_diagnostics.resident_representation` is
`native_block_fp8_with_q8_fallback_dense_bf16` and `fp8_execution_mode` is
`native_block_fp8_auto_self_checked` (or `native_block_fp8_explicit_self_checked`
when `projection_backend = "native_fp8"` was set): projections stay resident as the
checkpoint's block-FP8 bytes and run on the compact FP8 kernels after the load's
device gates and checks against the checkpoint bytes. When those gates fail or a
fallback is requested, the representation is
`q8_0_requantized_projections_with_dense_bf16` and `fp8_execution_mode` is
`q8_0_fallback:<reason>`; `fallback_reason` explains the choice either way. The
per-path outcomes appear under `fast_paths.fp8_projections`, `fast_paths.fp8_gemv`,
`fast_paths.fp8_packing` and `fast_paths.fp8_lm_head` (the block-FP8 LM head, a
deliberate precision reduction, with its observed gate error). On Metal the
representation is `native_block_fp8_with_dense_f16` (or `expanded_f16` with
`IZWI_METAL_PROJECTION_BACKEND=expanded`), with the per-kind GEMV checks under
`fast_paths.metal_fp8_gemv`; the CPU reports `q8_0_projections_with_dense_f32`
(or `expanded_f32` with `IZWI_CPU_PROJECTION_BACKEND=expanded`).

| Method | Path | Notes |
|--------|------|-------|
| `GET` | `/v1/admin/models` | List known enabled variants, local status, modalities, and route capabilities. |
| `GET` | `/v1/admin/models/{variant}` | Fetch one model status and capability contract. |
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
  "built_in_voice_count": 41,
  "supports_reference_voice": false,
  "supports_voice_description": false,
  "supports_streaming": true,
  "supports_speed_control": true,
  "supports_auto_long_form": true
}
```

Model records also expose route and batch capability flags so clients can discover which models can drive OpenAI-compatible speech, persisted speech-to-text jobs, diarization records, Studio projects, realtime voice sessions, saved voices, forced alignment, tokenizer workflows, and durable ASR/TTS jobs:

```json
{
  "variant": "Kokoro-82M",
  "enabled": true,
  "status": "ready",
  "deployment_state": "ready",
  "modalities": ["text_input", "audio_output"],
  "route_capabilities": {
    "openai_chat_completions": false,
    "openai_responses": false,
    "openai_audio_speech": true,
    "openai_audio_transcriptions": false,
    "speech_to_text_jobs": false,
    "speech_to_text_realtime": false,
    "diarization_records": false,
    "text_to_speech_records": true,
    "voice_design_records": false,
    "voice_clone_records": false,
    "saved_voice_reuse": false,
    "studio_projects": true,
    "voice_realtime_text_model": false,
    "voice_realtime_modular_asr": false,
    "voice_realtime_modular_tts": true,
    "voice_realtime_unified": false,
    "forced_alignment": false,
    "tokenizer": false
  },
  "batch_capabilities": {
    "batch_asr": false,
    "batch_tts": true,
    "durable_jobs": true,
    "artifacts": true,
    "cancellation": true,
    "retry": true,
    "requires_voice_permission": false
  }
}
```

Download progress SSE payload:

```json
{
  "variant": "Qwen3.5-4B",
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

`GET /v1/speech-to-text/realtime/ws`

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
| `error` | Error message. `fatal: true` ends the session. |

Constraints:

- Binary frames larger than 512 KiB are rejected.
- Natively streaming models serve at most `IZWI_ASR_REALTIME_MAX_SESSIONS` streams at once. A session over that cap gets one non-fatal `error` with code `realtime_stream_capacity` (typed protocol: `recoverable_error` with code `backpressure`) and keeps going by re-transcribing a rolling window, which adds latency.
- Frames that arrive while the model is busy are handed to it together, up to five seconds of audio at a time.
- Sample rate must remain stable during a session.
- Sample rates outside the accepted runtime range return an error.

Noise handling:

- Models that stream natively (Voxtral Realtime, Nemotron) transcribe as audio arrives. Voxtral would otherwise turn background noise into made-up words, so its input passes a noise gate: audio is held for 500 ms, and anything that is not confirmed speech (300 ms of voiced speech, 6 dB above the stream's own noise floor) reaches the model as silence. Its transcripts therefore arrive about 500 ms later. Steady noise such as a fan, hum or pink noise produces no text; music and crowd babble can still pass. `IZWI_ASR_REALTIME_NOISE_GATE=0` turns the gate off.
- Other models re-transcribe a rolling window of recent audio. A window that only grew by non-speech since the last pass is not transcribed again.

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
| `conversation_item_truncate` | `utterance_seq` and `audio_end_ms`: the client played only that much of the reply's audio (also accepted as `conversation.item.truncate`, after OpenAI Realtime). The conversation remembers the reply only up to there. |
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
| `max_output_tokens` | Reply token budget: default `384` in modular mode (a few spoken sentences), `1536` in unified mode. |
| `output_audio_format` | `pcm_i16` (default: PCM16 at the model's rate), `g711_ulaw` or `g711_alaw` (8 kHz, one byte a sample). |
| `vad_threshold` | Earshot speech score threshold, default `0.5`. Values are clamped to the valid score range. |
| `vad_snr_margin_db` | How far, in dB, a frame must rise above the stream's own noise floor to start an utterance; half of it keeps one going. Default `6` (or `IZWI_VAD_SNR_MARGIN_DB`), clamped to `0..40`. Earshot scores steady noise such as a fan or hum as speech; this margin is what ignores it. `0` turns the check off. |
| `min_speech_ms` | Minimum speech duration. |
| `silence_duration_ms` | Silence before utterance end. |
| `max_utterance_ms` | Hard utterance duration cap. |
| `pre_roll_ms` | Audio retained before speech start. |
| `barge_in_min_speech_ms` | Voiced speech, default `300`, that interrupts a speaking assistant. A shorter sound (a cough, a door, speaker echo) does not interrupt it. `0` interrupts on the first speech frame. |
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
| `5` | Kind: `2` assistant PCM16, `3` G.711 μ-law, `4` G.711 A-law (the session's `output_audio_format`). |
| `6..8` | Flags; bit `0` marks final chunk. |
| `8..16` | Little-endian `utterance_seq` (`u64`). |
| `16..20` | Little-endian chunk sequence (`u32`). |
| `20..24` | Little-endian sample rate (`u32`): the model's rate for PCM16, `8000` for G.711. |
| `24..` | Mono audio: PCM16 little-endian, or one G.711 byte a sample. |

Server events:

| Type | Notes |
|------|-------|
| `connected` | Socket accepted and protocol version announced. |
| `session_ready` | Voice session initialized. |
| `input_stream_ready` | Includes resolved VAD settings (backend, threshold, `snr_margin_db`, score sample rate and frame duration, speech and silence durations) and the `output_audio_format`. The typed v2 protocol's `session_started` does not carry them. |
| `models_ready` | Sent once the stream's models have loaded in the background after `input_stream_start`: each model's `stage`, `model_id`, `ready`, and `error` when it could not load. |
| `input_stream_stopped` | Input stream stopped. |
| `user_speech_start`, `user_speech_end` | VAD utterance boundaries. A client should duck, not stop, assistant playback on `user_speech_start`: the server cancels the turn only after `barge_in_min_speech_ms` of speech (`turn_interrupted`), and a `user_speech_end` starts a new turn that replaces it. |
| `user_speech_rejected` | A speech start was rejected as noise (too little voiced speech, or no second with 200 ms of speech evidence) and the input stream remains ready. Every `user_speech_start` ends in either `user_speech_end` or this event. Ducked playback can resume. The typed v2 protocol reports it as `turn_completed` with status `no_input`. |
| `turn_processing` | Turn started. |
| `user_transcript_start`, `user_transcript_delta`, `user_transcript_final` | User transcript events. |
| `assistant_text_start`, `assistant_text_snapshot`, `assistant_text_final` | Assistant text events. A snapshot carries the whole reply so far; the final event also has `raw_text`. |
| `assistant_audio_start`, `assistant_audio_done` | Assistant audio envelope around binary chunks; the start event names the `sample_rate` and `audio_format`. |
| `conversation_item_truncated` | Acknowledges a truncate with the `text` the conversation now remembers for that reply. |
| `turn_done` | Terminal turn status: `ok`, `error`, `timeout`, `interrupted`, or `no_input`. |
| `pong` | Ping response. |
| `error` | Error with optional utterance identifiers. |

How a modular turn runs:

- The models load at `input_stream_start` (`models_ready`), so the first turn does not pay the load.
- The reply streams from the chat model with thinking disabled (any reasoning text is dropped). Complete sentences are spoken as they arrive: the first segment is the fewest sentences reaching about 40 characters, later ones pack the sentences that have arrived (up to 240 characters), and the next segment is synthesized while the current one plays. Segments are joined with a short pause and fades.
- Each stage (ASR, the reply, each TTS segment) takes its own request admission slot, so a turn holds none while it waits.
- Assistant audio is paced by the client: when more than about 5 s (or 4 MiB) of audio is queued, synthesis waits for the client to take it rather than closing the session. A client that stops reading for 5 s ends the session.
- The prompt carries the most recent history (24 messages, about 3000 tokens). A reply is remembered without reasoning, and only as far as it was heard: up to `conversation_item_truncate`'s `audio_end_ms`, or up to the client's estimated playback position when the user interrupted it.

Voice realtime persists voice sessions and turns in the local store. Modular turns can also update observational memory when that feature is enabled.
