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
| `/v1/audio/speech` | Stable | OpenAI-compatible text-to-speech endpoint (mp3, opus, flac, wav, pcm; buffered, chunked `audio` or SSE) with Izwi voice extensions. |
| `/v1/audio/translations`, `/v1/audio/voices*`, `/v1/audio/voice_consents*` | Stable | OpenAI-compatible translation and custom-voice routes. |
| `/v1/audio/speech/batch`, `/v1/audio/speech/{id}`, `/v1/audio/speech/stream` | Preview | Batch speech, outcome lookup (process-local) and the text-in speech WebSocket. |
| `/v1/audio/transcriptions` | Stable | OpenAI-compatible transcription endpoint with local streaming support. |
| `/v1/files*` | Stable | OpenAI-compatible Files API for batch input, audio and output. |
| `/v1/batches*` | Stable | OpenAI-compatible Batch API over the audio transcription, translation and speech routes. |
| `/livez`, `/readyz`, `/v1/live`, `/v1/ready`, `/v1/health` | Stable | Operational health and readiness endpoints. |
| `/v1/responses*` | Preview | OpenAI-compatible Responses API shape with process-local response retention. |
| First-party workflow routes | Preview | Persisted local product APIs used by the web UI and desktop app. |
| `/v1/jobs/*` | Preview | Durable batch runtime job trace, artifact, cancel, and retry APIs. |
| `/v1/media*` | Preview | Local media lifecycle APIs with runtime asset registration for uploads. |
| `/v1/metrics/batch-runtime` | Preview | Durable batch runtime queue, job, stage, and worker telemetry. |
| `/v1/admin/*` | Preview | Local model-management APIs. Bind carefully on shared hosts. |
| WebSocket realtime routes | Preview | Browser-facing low-latency protocols that may evolve. |
| `/v1/realtime`, `/v1/realtime/client_secrets` | Preview | OpenAI Realtime API (GA) over WebSocket, served by local models. |
| `/v1/realtime/media-stream` | Preview | A phone call as a telephony media stream, run as a `/v1/realtime` session. |
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
| `429` | `rate_limit_error` | `rate_limit_exceeded` | A rate, quota or admission limit was reached, or the server stayed at capacity for the whole admission queue timeout (`IZWI_ADMISSION_QUEUE_TIMEOUT_SECS`). These responses carry a `Retry-After` header. |
| `429` | `rate_limit_error` | `resource_unavailable` | A `flex` (batch-class) request found no free capacity within the admission queue timeout. The message starts with `Resource Unavailable`, as OpenAI's flex answer does. Retry later, or send the request at the default tier. |
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
| `IZWI_ADMISSION_QUEUE_TIMEOUT_SECS` | `30` seconds | How long a chat, agent or `/v1/audio/*` request waits for an admission slot when the server is at capacity, before it is refused with `429`. `0` waits without limit. Realtime sessions, the first-party record and job routes and background work always wait. |
| `IZWI_OPENAI_AUDIO_UPLOAD_LIMIT_MB` | `25` strict, `64` relaxed | Upload limit for OpenAI audio routes. |
| First-party audio upload limit | `64 MiB` | Applies to persisted transcription, diarization, TTS, voice design, voice clone, and saved voice creation routes. |
| `IZWI_FILES_MAX_BYTES` | `209715200` (200 MB) | Largest file `/v1/files` accepts, at most 1 GiB. A larger upload is refused with `413`. |
| `IZWI_BATCH_MAX_REQUESTS` | `50000` | Requests one `/v1/batches` input file may hold. A larger file fails its batch. |
| `IZWI_BATCH_WORKER_CONCURRENCY` | The runtime's capacity | `/v1/batches` lines in flight at once, and durable jobs the batch worker runs at once. |
| `IZWI_BATCH_STAGE_TIMEOUT_SECS` | `21600` (six hours) | Wall-clock limit of one durable job stage, from its start (30 to 86400). `0` disables it. |
| `IZWI_BATCH_WEBHOOK_SECRET` | Unset (webhooks off) | Turns on `/v1/batches` completion webhooks and signs each delivery. See [Batch API](#batch-api). |
| `IZWI_BATCH_WEBHOOK_ALLOW_HTTP`, `IZWI_BATCH_WEBHOOK_ALLOW_PRIVATE` | `false` | Allow `http://` webhook URLs, and webhook hosts on loopback, private or link-local addresses. |
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
speech goes through the gateway's realtime socket, and OpenAI Realtime
sessions through its `/v1/realtime` pass-through to izwi-servers).

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
| Audio input | No chat model hears audio. A message carrying an audio content part (`input_audio`, or an `audio_url` / `audio` part) is a `400`; for spoken conversations use the [voice realtime API](#voice-realtime). LFM3-Audio is not a chat model. |
| `modalities`, `audio` | Chat completions answer in text. `modalities` other than `["text"]`, and any `audio` output object, are a `400` naming the field; spoken replies use the [voice realtime API](#voice-realtime). |
| `max_tokens`, `max_completion_tokens` | Optional output budgets (`max_completion_tokens` wins). Unset, the reply may use the rest of the context. An explicit budget that does not fit beside the prompt is a `400` (as in vLLM) unless `truncation` is `auto`. |
| `truncation` | Izwi extension, as in the Responses API. `disabled` (default): a prompt that does not fit the context is a `400` with `code: context_length_exceeded`, `param: messages` and the token counts. `auto`: the oldest complete turns are dropped (the system message and newest user message are kept) and an explicit budget is shortened to the remaining context. Through the gateway, workers on worker protocol minor 4 apply this exactly as the local server; older workers still truncate automatically. |
| `stream` | `true` returns SSE chat chunks. The request is admitted before the stream starts, so a request that cannot run (for example `context_length_exceeded`) is a plain HTTP error, not a stream. |
| `stream_options.include_usage` | Sends the usage in one more chunk after the finish chunk, with `choices: []`; every earlier chunk then carries `usage: null`, as in OpenAI. |
| `logprobs`, `top_logprobs` | Per-token logprobs (`top_logprobs` 0–20 alternatives). Returned on `choices[].logprobs.content` in the completion and in each stream chunk. |
| `temperature`, `top_p`, `top_k`, `repetition_penalty`, `presence_penalty` | Optional runtime sampling controls. Explicit values override model profiles. Qwen3.5, Qwen3.6 and Qwen3.8 default to Qwen's published profile for the active thinking mode. LFM2.5 1.2B defaults to Liquid's model-card profile: temperature 0.1 (Instruct) or 0.05 (Thinking), top-k 50, repetition penalty 1.05. LFM3 1.7B defaults to its card and `generation_config.json`: temperature 0.1, top-k 50, repetition penalty 1.1. On Qwen3.5 the presence penalty counts generated tokens only, and the repetition penalty counts the prompt and generated tokens. |
| `seed` | Optional signed 64-bit sampling seed. The same seed and parameters sample the same tokens on the same backend; without one, each request samples independently. |
| `stop` | A string or an array of up to 4 non-empty strings (each at most 1024 bytes). Generation ends before the first match, the stop text is not returned, and `finish_reason` is `stop`. Streams hold back text that could still begin a stop string until it is settled. Requests with `stop` decode one token per scheduler step. |
| `frequency_penalty` | Rejected in strict OpenAI compatibility mode when non-default. |
| `n` | Only `1` is supported. |
| Ranges | `temperature` 0–2, `top_p` greater than 0 and at most 1, `top_k` -1 or 0 (disabled) or positive, `presence_penalty` and `frequency_penalty` -2–2, `repetition_penalty` 1–2, `max_tokens` and `max_completion_tokens` at least 1. Anything else is a `400` whose `error.param` names the field and whose `error.code` is `invalid_value`. |
| `tools` | OpenAI function tools (`{"type": "function", "function": {"name": …}}`), written into the prompt exactly as the model's own chat template writes them, in the key order you send. Qwen3, Qwen3.5, Qwen3.6, Qwen3.8, LFM2.5 and LFM3 call tools; Gemma 3 has no tool template, so tools for it are a `400` (`unsupported_parameter`) rather than silently ignored. |
| `tool_choice` | `auto` (default) offers the tools and parses calls; `none` neither offers them nor parses calls. `required` makes the answer call a tool, and `{"type": "function", "function": {"name": …}}` makes it call that tool: the prompt opens the call for the model, with thinking off for that turn, and the sampler allows only declared tool names and arguments that fit the tool's `parameters` schema. On Qwen3.8 a forced call is opened by the prompt alone (its arguments are not checked). A named tool that is not in `tools`, or a forced choice without `tools`, is a `400`. |
| `parallel_tool_calls` | `false` keeps the answer to one tool call: generation ends when the call closes. Default `true`. |
| `strict` (on a tool's `function`) | `true` guarantees that calls name a declared tool and that their arguments fit the `parameters` schema, also with `tool_choice: auto`: the sampler is unconstrained until the model opens a call, then follows the schema until the call closes. When any tool is `strict`, every tool's calls are checked. Parameters are written in schema order, required ones always. A schema the grammar cannot express is a `400` naming `tools`; Qwen3.8 cannot enforce `strict` and refuses it (`unsupported_parameter`). Without `strict` and without a forced choice the model is free and its output is parsed leniently. |
| Deferred tools (LFM3) | A tool with `"defer_loading": true` (on the tool or its `function`) stays out of the prompt's tool list until a tool result names it, as LFM3's template defines. A `tool` message's content can name tools with `{"type": "tool_reference", "name": …}` items, list definitions in `{"result": {"tools": [...]}}` items, or carry them in an `additional_tools` key (definitions, or `true` when the content holds them); the model then sees them in `<additional_tools>`. Send every tool, deferred ones included, on every request: a reference to an undefined tool is a `400`. Consecutive tool messages with the same `tool_call_id` are one result. Other models refuse `tool_reference` items and `additional_tools` with a `400`. Tool definitions keep the key order you send. |
| `enable_thinking` | Izwi extension for thinking-capable local models. Qwen3.8 and LFM3 default to `true`; LFM3 with `false` answers directly. |
| `reasoning_effort` | Qwen3.8 reasoning level: `xhigh` (default), `medium`, or `low`. Unknown values are rejected. |
| `preserve_thinking` | Preserve prior assistant reasoning in Qwen3.8 history; defaults to `true`. LFM3 defaults to `false` (its template keeps reasoning only for turns after the last user message). Qwen3.5 history behavior is unchanged. |
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
exactly as the local server does. A forced `tool_choice` and `strict` tools
need a worker that advertises them too. The reasoning controls are not yet
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
  message (`400` otherwise). The templates read tool results in order. An
  assistant message may also carry the `reasoning_content` a previous answer
  returned, so the model keeps its reasoning across a tool round.
- Calls are parsed only when the request offered tools. They are returned as
  `message.tool_calls` (ids `call_<hex>`) with `finish_reason: "tool_calls"`,
  and the call markup never appears in `content`. An answer the output budget
  cut off ends with `finish_reason: "length"`, also when it made calls: a
  call cut off there keeps only its complete arguments.
- Arguments keep the order the model wrote them in. In the XML formats a
  `string` parameter keeps its text even when it reads as a number, `true`
  or `null`; other parameters are read by their declared type.
- A call to a tool the request did not declare is returned as text, not as a
  call. A tool call ends a reasoning block the model left open, so with
  `reasoning_format: "parsed"` the call is not lost in `reasoning_content`.
- Streamed, calls arrive as `delta.tool_calls`: the first fragment of each
  carries its `index`, `id`, `type` and `function.name`, and the
  `function.arguments` fragments concatenate to the JSON arguments. A Qwen
  XML call is sent as soon as its name is known and its arguments one
  parameter at a time; a parameter declared as a `string` is sent as its
  text arrives, so a long value (a file body) streams instead of arriving
  at once. A Qwen3 JSON call and an LFM2.5 call list are sent when they
  close. Text that might open a call is held back until the next
  delta settles it, and a block that does not parse is sent as text.
- Qwen3 uses the Hermes format its template defines: tools are listed in the
  system turn, the model writes
  `<tool_call>{"name": …, "arguments": {…}}</tool_call>`, and results are
  `<tool_response>` blocks in a user turn.
- Qwen3.5, Qwen3.6 and Qwen3.8 use Qwen's XML tool format (`<tool_call>`,
  `<function=…>`, `<parameter=…>`); consecutive tool results share one user
  turn, as their templates write them.
- LFM2.5 models use their own format: tools are listed in the system turn,
  `tool` messages become native tool turns, and calls the model writes as
  `<|tool_call_start|>[name(arg="v")]<|tool_call_end|>` are returned as
  `tool_calls`.
- LFM2.5 1.2B supports 32,768 tokens of context, the length on Liquid's model
  cards, on every backend.
- LFM3 uses the Qwen XML call format: tools are listed in the system turn
  inside `<tools>`, one JSON object per line, `tool` messages become native
  tool turns wrapped in `<tool_response>`, and calls are returned as
  `tool_calls`. Its context is 32,768 tokens by default on CPU and Metal and
  the checkpoint's 262,144 on CUDA (capped by the memory the arena fits); the
  server's `IZWI_LFM3_CONTEXT_TOKENS` sets it on any backend.

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

### Audio Service Tier

`POST /v1/audio/speech`, `/v1/audio/speech/batch`, `/v1/audio/transcriptions`
and `/v1/audio/translations` take OpenAI's `service_tier`. OpenAI defines the
field for Chat Completions and Responses; on the audio routes it is an Izwi
extension with the same values:

| `service_tier` | Served as | Effect |
|----------------|-----------|--------|
| `flex` | `flex` | The request runs as batch work: it is admitted and scheduled behind interactive and streaming requests. Use it for recordings and bulk synthesis that need not answer at once. `stream` and `stream_format` still work. |
| `default`, `auto`, or absent | `default` | The route's own class: interactive for a whole-file answer, streaming for SSE and chunked audio. On `/v1/audio/speech/batch` an absent tier is `flex`. |
| `priority` | `default` | Accepted for compatibility and served as `default`. |

Any other value returns `400` with `param: "service_tier"`.

When the server is at capacity a request waits for an admission slot for up
to `IZWI_ADMISSION_QUEUE_TIMEOUT_SECS` (30 seconds) and is then refused with
`429` and a `Retry-After` header, before any stream starts. A `default`
request gets code `rate_limit_exceeded`; a `flex` request gets code
`resource_unavailable` and a message starting `Resource Unavailable`, as
OpenAI's flex tier does, so a client can retry later or fall back to
`default`. In a batch each refused item carries the error in its result.

Send it as the `service_tier` field of a JSON body or multipart form. A
client that cannot add a form field sends the `X-Izwi-Service-Tier` request
header instead; when both are present the body field wins.

The gateway's speech pass-through relays the field and the header to the
izwi-server unchanged.

[Real-time and batch workloads](/realtime-and-batch) covers which route to
use for which workload, how the server keeps capacity for live traffic
beside batch work, and the settings and metrics involved.

Every response names the tier that served it in the `X-Izwi-Service-Tier`
response header, binary audio, text, SRT and VTT included. When the request
named a tier, JSON bodies and the final SSE event (`transcript.text.done`,
`audio.done`) also carry `"service_tier": "flex"` or `"default"`; a request
that names none gets the bodies it always has.

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@recording.wav" \
  -F "model=Parakeet-TDT-0.6B-v3" \
  -F "service_tier=flex"
```

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
| `model` | Required TTS model variant, or an OpenAI model name an operator mapped with `IZWI_OPENAI_MODEL_ALIASES` (`tts-1=Kokoro-82M,…`). Unmapped OpenAI names return `400`. |
| `input` | Required text to synthesize, unless `script` gives it. |
| `voice` | A built-in voice/speaker name where the model has presets; OpenAI's custom-voice object `{"id": "<voice id>"}` for a saved voice (`/v1/audio/voices`, `/v1/voices`); or, for a model that clones and has no built-in voices, an uploaded voice's name. |
| `response_format` | `mp3`, `opus` (Ogg Opus), `flac`, `wav` (the default; `IZWI_SPEECH_DEFAULT_FORMAT` changes it), `pcm` (OpenAI's: 16-bit mono at 24 kHz, resampled when the model runs at another rate), `pcm_i16`/`pcm16`/`raw_i16` and `pcm_f32`/`raw_f32` (headerless, at the model's rate), `g711_ulaw`, `g711_alaw` (8 kHz). `aac` returns `400`: AAC encoders need a patent licence. |
| `sample_rate` | Optional rate to encode at, resampled when the model's differs. Opus takes 8000, 12000, 16000, 24000 or 48000; MP3 its MPEG rates (8000 to 48000); other formats 8000 to 192000. Unset, the model's rate is kept where the format carries it (Opus: 44.1 kHz becomes 48 kHz, 22.05 kHz becomes 24 kHz). |
| `bitrate` | Optional bit rate for `mp3` (CBR; 8 to 160 kbit/s at 24 kHz, 32 to 320 at 44.1/48 kHz; default 64 or 128) and `opus` (6 to 510 kbit/s; default 32 kbit/s at 24 kHz). |
| `allow_format_fallback` | Optional boolean. When `true`, a recognized format this build cannot encode (`aac`, or `mp3`/`opus` in a build without them) returns WAV bytes with `X-Actual-Response-Format: wav`, `X-Response-Format-Fallback` and `Warning` headers instead of `400`. |
| `speed` | Optional model-dependent speed control. |
| `language` | Optional language hint such as `English`, `Chinese`, or `Auto`. `Chatterbox-Multilingual-500M` speaks the language it names, as a code or English name (`fr`, `French`), and returns `400` for one it does not support; `Chatterbox-500M` returns `400` for any language but English. `Maya2-Native-2.5B` takes one of its 12 language codes (`hi`, `ta`, …), or none or `auto` to read the language from the text's script, and returns `400` for anything else. LFM3 Audio takes one of its 19 locales (`en-US` when unset, `en-GB`, `fr`, `ja`, …) and returns `400` for anything else. `GET /v1/admin/models` lists a Chatterbox, Maya 2 Native or LFM3 Audio model's languages in `speech_capabilities.languages`. |
| `temperature`, `top_k` | Optional sampling controls. |
| `exaggeration` | Chatterbox only. Emotion intensity, 0 to 2 (default `0.5`). |
| `cfg_weight` | Chatterbox only. Classifier-free guidance weight, 0 to 1 (default `0.5`); `0` turns guidance off. For expressive or fast-talking reference voices, try about `0.3` with `exaggeration` around `0.7` or higher. |
| `min_p` | Chatterbox only. Min-p sampling threshold, 0 to 1 (default `0.05`). |
| `max_tokens`, `max_output_tokens` | Optional output token budget aliases. |
| `instructions` | Voice-design prompt for voice-design models; for `Maya1-3B`, the description of the voice to speak with, replacing `voice`. At most 4,096 characters. |
| `reference_audio` (`ref_audio`) | Audio for voice cloning: base64, a data URL, or an `http(s)` URL on a host `IZWI_ALLOWED_MEDIA_DOMAINS` allows (unset, URLs return `400`). URLs must resolve to public addresses only, and at most 10 MiB is fetched in 30 s. |
| `reference_text` (`ref_text`) | Transcript of the reference audio. Optional for Chatterbox, which ignores it. |
| `reference_language` | `Maya2-Native-2.5B` only. The language the reference audio is spoken in, one of its 12 codes; unset, it is read from the transcript's script. Any other model returns `400` if it is set. The speech history create routes accept it too. |
| `saved_voice_id` | Server-side saved voice reference to reuse. |
| `references` | Izwi extension, for models that voice several speakers (`speech_capabilities.supports_speaker_dialogue`: `FishAudio-Drama-3`). Reference voices per speaker, each `{speaker, audio, text}` or `{speaker, saved_voice_id}`; it replaces `reference_audio`, `reference_text` and `saved_voice_id`. See the Drama 3 dialogue notes below. |
| `script` | Izwi extension, for the same models. A dialogue as `[{speaker, text}]` lines, instead of `input`. |
| `service_tier` | Optional `flex`, `default`, `auto` or `priority`; see [Audio Service Tier](#audio-service-tier). |
| `stream`, `stream_format` | `stream_format: "audio"` sends the encoded file as a chunked body while it is generated (OpenAI's streaming). `stream_format: "sse"` or `stream: true` sends server-sent events whose deltas are base64 slices of the same bytes. Without either, the whole file arrives after generation, with `Content-Length` and statistics headers. Text one request cannot speak streams as consecutive long-form segments, as a final-only request splits it. |

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

`LFM3-Audio-2B-GGUF` and `LFM3-Audio-2B-Q8_0-GGUF` specifics:

- Its voice is `female` (the default) or `male` (see
  [Voice Presets](/models/voice-presets#lfm3-audio)). `language` names one of
  19 locales: `en-US` (the default), `en-GB`, `ar`, `da`, `de`, `el`, `fi`,
  `fr`, `he`, `it`, `ja`, `ko`, `nl`, `pl`, `pt`, `es`, `sv`, `tr` or `zh`.
- A reference voice (`reference_audio` or a saved voice), `instructions` and
  `speed` each return `400`.
- Sampling defaults follow Liquid AI's model card: `temperature` 1.0, top-k
  128, and a repetition penalty of 1.1 over the last 4,096 frames. `seed`
  makes a request reproducible.
- `stream: true` streams 24 kHz audio as it is generated. Long text is split
  into sentences automatically.
- The same model transcribes speech (see
  [Audio Transcriptions](#audio-transcriptions)) and answers spoken turns with
  interleaved text and speech in the `unified` mode of
  [Voice Realtime](#voice-realtime), where it can call tools. It also streams
  transcription in its native LiveASR mode through
  [Transcription Realtime](#transcription-realtime). It is not a chat model:
  chat completions and Responses refuse it. Prefix reuse is not supported yet.

`FishAudio-Drama-3` specifics:

- It takes the same fields as Fish S2 Pro: `input`, `reference_audio` with
  `reference_text`, `saved_voice_id`, `stream` / `stream_format`,
  `response_format`, `temperature`, `top_p`, `top_k`, `max_tokens` and
  `seed`. `voice` is ignored.
- A reference voice is optional. Without one, the sampling `seed` sets the
  voice: the same `seed` gives the same voice, and without a `seed` each
  request picks one. With one, its transcript is required, inline
  (`reference_text`) or from a saved voice that has one. An inline clip may
  last up to 60 s, and saved voices hold up to 30 s. `reference_audio` and
  `reference_text` (or `saved_voice_id`) give one reference, for speaker 0;
  `references` gives up to 8 for a dialogue (below).
- Audio is generated at 44.1 kHz; `response_format` and `sample_rate`
  apply as for any model (Opus goes out at 48 kHz).
- `language` is accepted as a hint and not passed to the model, which reads
  the language from the text; a `[language: xx]` tag in `input` steers it.
- Tags in `input` reach the model as text: `[sighs]`, `[laughs]`, `[pause]`,
  `[gasps]`, `[clears throat]`, `[hesitates]`, free-text tags such as
  `[with quiet excitement]`, `<whisper>…</whisper>` and `[language: xx]`.
  `speech_capabilities.inline_tags` of `GET /v1/admin/models` lists the
  trained point tags. Speaker tags `<|speaker:0|>` to `<|speaker:15|>` are
  accepted; any other `<|speaker…` form returns `400`.
- Control tokens return `400` naming the token, in `input` and in
  `reference_text`: `<|im_start|>`, `<|im_end|>`, `<|voice|>`, `<|text|>`,
  `<|interleave|>` and `<|semantic:N|>`. `FishAudio-S2-Pro` refuses them the
  same way; it used to pass them into its prompt.
- Sampling defaults: `temperature` 1.0 and top-p 0.9. Without `top_k` the main
  token has no cap and the residual codebooks keep 30; `top_k: 0` also means
  no cap. `seed` seeds the sampler and, for a request without a reference,
  fixes the voice.
- Long text is a chunk session. The model's own chunker cuts `input` into
  chunks of about 300 UTF-8 bytes: sentence ends first, then clause marks,
  then spaces; decimals, `<|…|>` tokens, `<tag>…</tag>` and `[…]` stay whole;
  a speaker's lines keep their tag; a chunk holds at most three speakers. Each
  chunk is its own generation, and chunk `i` samples from `seed + i`. Its
  prompt holds the reference, the first chunk and the latest chunk with the
  audio codes generated for them, so the voice carries through the whole
  text; when a prompt would be too long, the latest chunk is left out first,
  then the first chunk. A chunk whose prompt is too long for the model before
  any audio is cut in two and retried. Chunks play one after another with no
  added pause. The server setting `IZWI_FISH_DRAMA3_LONG_FORM_ENABLED=false`
  refuses text that needs more than one chunk with `400` (Fish S2 Pro keeps
  `IZWI_FISH_LONG_FORM_ENABLED`).
- `max_tokens` is a frame budget per chunk (21.5 frames a second), 1 to
  4,096; larger values are clamped to 4,096. Unset, each chunk gets about 2.5
  times its speaking estimate, at most 1,024 frames (about 47 s). A chunk that
  runs out of budget before the model ends its speech fails the request with
  a `frame_limit` error (`500`) instead of returning a cut-off take.
- With `stream: true`, each chunk streams as it is generated: first audio
  after 4 frames, then every 16 frames. The final `audio.done` event also
  reports `planned_segment_count` (the chunks) and
  `token_unit: "semantic_frames"`.
- First-party text-to-speech records journal their audio per chunk, so an
  unfinished record can be played from its start
  (`GET /v1/text-to-speech/{record_id}/events`), and a durable job that stops
  between chunks resumes with the same voice.
- A speech record keeps what a later take needs to continue it: its audio
  codes, its voices and its seed. See
  [Drama 3 take editing](#drama-3-take-editing). `/v1/audio/speech` keeps no
  records and does not take `previous_take_ids`.
- The weights' license (Fish Audio Community License v1.0) allows commercial
  use only below USD 500,000 in annual revenue, and offering the model to
  third parties as a hosted speech service, such as this API, needs a license
  from Fish Audio. Nothing is refused because of it.

#### Drama 3 dialogue

`FishAudio-Drama-3` voices up to 16 speakers in one request. Its card reports
`supports_speaker_dialogue: true`, `max_speakers: 16` and `max_references: 8`
under `speech_capabilities` in `GET /v1/admin/models`.

- **Who says what.** Tag the text: `<|speaker:0|>` to `<|speaker:15|>` before
  each turn in `input`, or send `script` instead of `input`, a list of
  `{"speaker": N, "text": "..."}` lines. Izwi writes a script as one
  `<|speaker:N|> text` line per entry, joined by newlines; that rendered text
  is what a speech record stores as its `input_text`. Text before the first
  tag is speaker 0. A script line's text may not carry its own speaker tag;
  performance tags such as `[laughs]` are fine.
- **Voices.** `references` is a list of up to 8 reference voices. Each is
  `{"speaker": N, "audio": "<base64>", "text": "<transcript>"}` or
  `{"speaker": N, "saved_voice_id": "<id>"}` (the saved voice must have a
  transcript). `speaker` defaults to 0, or to the speaker of a
  `<|speaker:N|>` tag that starts `text`. Several references may voice one
  speaker. Every chunk of a long dialogue holds every reference.
- **Seeded speakers.** A speaker with no reference speaks in a voice the
  `seed` sets, so a request may mix referenced and seeded speakers, or give no
  references at all.
- **One reference form.** `references` replaces `reference_audio`,
  `reference_text` and `saved_voice_id`; those mean one reference for speaker
  0.
- **Size.** Inline clips (`audio`) may total 16 MiB of base64 per request,
  below the route's body limit; save larger voices with `POST /v1/voices` and
  pass their `saved_voice_id`. Each inline clip lasts at most 60 s.

```json
{
  "model": "FishAudio-Drama-3",
  "script": [
    { "speaker": 0, "text": "Ready when you are." },
    { "speaker": 1, "text": "Always. [laughs] Let's go." }
  ],
  "references": [
    { "speaker": 0, "saved_voice_id": "voice_anna" },
    { "speaker": 1, "audio": "<base64 WAV>", "text": "Thanks for calling, how can I help?" }
  ],
  "seed": 7
}
```

The same dialogue with tags in `input`:

```json
{
  "model": "FishAudio-Drama-3",
  "input": "<|speaker:0|> Ready when you are.\n<|speaker:1|> Always. [laughs] Let's go.",
  "references": [
    { "speaker": 0, "saved_voice_id": "voice_anna" },
    { "speaker": 1, "saved_voice_id": "voice_ben" }
  ]
}
```

The speech record routes (`POST /v1/text-to-speech`, `POST /v1/voice-clones`)
take the same `references` and `script`, with `script` replacing `text`.

These return `400` before any model loads, naming the field:

- `references` or `script` for a model without `supports_speaker_dialogue`;
- `references` together with `reference_audio`, `reference_text` or
  `saved_voice_id`, or `script` together with `input` (`text` on the record
  routes);
- more than 8 references, or a `speaker` outside 0 to 15 in a reference or a
  script line;
- a reference that is not exactly one of `audio` (with its `text`) and
  `saved_voice_id`, a saved-voice reference given its own `text`, or a saved
  voice without a transcript;
- a reference `text` that starts with another speaker's tag than its
  `speaker`;
- control tokens in a reference `text` or a script line;
- an empty script, an empty script line, or a script line with a speaker tag;
- inline clips over 16 MiB in total.

Two limits are checked when the loaded model prepares the request, and also
return `400`: an inline clip over 60 s, and references that fill the prompt
(8,192 rows; audio takes about 21.5 rows a second) with no room left for the
text. Use fewer or shorter references then. Text that is too long beside the
references is cut into shorter chunks instead.

Non-stream response:

- Body is binary audio with exact headers (WAV sizes, FLAC totals and MD5, the MP3 LAME tag).
- `Content-Type` is the format's: `audio/mpeg`, `audio/ogg` (Opus), `audio/flac`, `audio/wav`, `audio/pcm` (`pcm`), `application/octet-stream` (`pcm_i16`, `pcm_f32`).
- `X-Audio-Sample-Rate` is the rate the audio plays at; `X-Izwi-Speech-Id` names the request for `GET /v1/audio/speech/{speech_id}`.
- `X-Requested-Response-Format` and `X-Actual-Response-Format` are exposed. Explicit fallbacks also include `X-Response-Format-Fallback` and an HTTP `Warning` header.
- `X-Izwi-Service-Tier` is the tier that served the request (see [Audio Service Tier](#audio-service-tier)), in every delivery.

`stream_format: "audio"` response: the same bytes as a chunked body. Headers
(`Content-Type`, `X-Audio-Sample-Rate`, `X-Izwi-Speech-Id` and its alias
`X-SGLang-Omni-Speech-Id`) go out with the first audio; a failure before it is
an ordinary error response, and a failure after it ends the body early. The
lookup then reports `status: "failed"`. A streamed WAV's RIFF sizes are
`0xFFFFFFFF` ("unknown"), FLAC's STREAMINFO has no totals, and an MP3 stream
has no LAME tag.

SSE events (each is also an OpenAI event, by its `type`):

| Event | `type` | Fields |
|-------|--------|--------|
| `audio.started` | | `request_id`, `sample_rate` (output), `audio_format`, optional explicit fallback note in `error`. |
| `audio.chunk` | `speech.audio.delta` | `audio` and `audio_base64` (base64 of the next encoded bytes: concatenated, the deltas are one file), `request_id`, `sequence`, `sample_count`, `is_final`, `sample_rate`. |
| `audio.done` | `speech.audio.done` | `usage` (`input_tokens`, which Izwi reports as 0; `output_tokens`; `total_tokens`), `tokens_generated`, `generation_time_ms`, `audio_duration_secs`, `rtf`, `stop_reason`, `timing`, and `service_tier` when the request named one. |
| `audio.failed` | `error` | `request_id`, `error`. |

Example SSE request:

```bash
curl -N http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"Kokoro-82M","input":"Stream this.","stream_format":"sse"}'
```

### Audio Speech Lookup

`GET /v1/audio/speech/{speech_id}` returns the outcome of a `/v1/audio/speech`
request this server ran, in any delivery (and of batch items and speech
socket utterances), by its `X-Izwi-Speech-Id`:

```json
{"id": "speech_…", "object": "audio.speech", "status": "completed", "model": "Kokoro-82M",
 "mode": "audio_stream", "response_format": "mp3", "sample_rate": 24000,
 "audio_duration_secs": 3.4, "finish_reason": "stop",
 "usage": {"input_tokens": 0, "output_tokens": 120, "total_tokens": 120, "prompt_tokens": 0, "completion_tokens": 120},
 "created_at": 1762430000, "completed_at": 1762430001, "request_id": "speech_…"}
```

`status` is `in_progress`, `completed`, `failed` or `cancelled` (the client
went away); `finish_reason` is `stop`, `length` (the output budget cut it),
`cancelled` or `error`. Outcomes are kept in memory: at most
`IZWI_SPEECH_OUTCOME_CAPACITY` (1024), for `IZWI_SPEECH_OUTCOME_TTL_SECS`
(an hour), and not across restarts. A missing id is `404`. No audio is kept.

### Audio Speech Batch

`POST /v1/audio/speech/batch` speaks up to `IZWI_TTS_BATCH_MAX_ITEMS` (32)
requests, in the envelope SGLang-Omni and vLLM-Omni share. Top-level speech
fields are defaults each item overrides:

```json
{"model": "Kokoro-82M", "voice": "af_heart", "response_format": "mp3",
 "items": [{"input": "One."}, {"input": "Two.", "speed": 1.1}]}
```

The response keeps the items' order. A failed item carries an OpenAI error
object, with `param` prefixed `items.<i>.`:

```json
{"id": "speech_batch_…", "object": "audio.speech.batch",
 "results": [
   {"index": 0, "status": "success", "id": "speech_…", "audio_data": "<base64>", "format": "mp3",
    "media_type": "audio/mpeg", "sample_rate": 24000, "audio_duration_secs": 0.8,
    "finish_reason": "stop", "usage": {"input_tokens": 0, "output_tokens": 40, "total_tokens": 40}},
   {"index": 1, "status": "error", "error": {"message": "…", "type": "invalid_request_error", "param": "items.1.speed", "code": null}}],
 "total": 2, "succeeded": 1, "failed": 1}
```

Items run `IZWI_TTS_BATCH_CONCURRENCY` (8) at a time, each admitted on its
own. An empty `items`, too many items, or `stream`/`stream_format` anywhere
fails the whole request with `400`. `service_tier` (see
[Audio Service Tier](#audio-service-tier)) is a top-level field that applies
to every item; inside an item it returns `400`. A batch runs as batch work
(`flex`) unless it says otherwise: send `"service_tier": "default"` to run
its items in the interactive class, as before the tier existed. The response
carries `service_tier` when the request named one. Items whose audio would pass
`IZWI_TTS_BATCH_MAX_OUTPUT_MB` (256) fail with `413`-class errors.

### Audio Speech Stream

`GET /v1/audio/speech/stream` is a WebSocket that takes text in and sends audio
out. Send text as you have it (an LLM's tokens, say). Izwi cuts it into
segments and synthesizes the next while the last plays: the first sentence
goes alone, then complete sentences up to 240 characters, and abbreviations
and decimals do not end a sentence.

Client messages (JSON text frames):

| `type` | Fields | Effect |
|--------|--------|--------|
| `session.config` | The speech fields of `/v1/audio/speech` except `input`, either nested under `session` or flat; plus `stream_audio` (default `true`), `split_granularity` (`sentence`, the default, then `clause` or `none`), `audio_transport` (`binary`, the default, or `base64`), and `close_on_done`. | Must be first, within 10 s. It may be sent again between utterances. `response_format` defaults to `pcm`. |
| `input.text` | `text` (at most 128 KiB) | Adds text. Segments that are complete start synthesizing. |
| `input.commit` | | Speaks the buffered text now, even without a sentence end; the utterance continues. |
| `input.done` | | Speaks the rest and ends the utterance (`session.done`). |
| `input.cancel` | | Drops buffered text and stops the segments in flight. |
| `session.close` | | Finishes the utterance, then closes. |
| `ping` | | Returns `pong`. |

Server events: `session.configured`, then for each segment `audio.start`
(`id`, `utterance_index`, `sentence_index`, `sentence_text`, `format`,
`sample_rate`, `channels`). Its audio follows as binary frames, or
`audio.chunk` events with `audio_b64` when `audio_transport` is `base64`.
Each segment ends with `audio.done` (`total_bytes`, `error`, `cancelled`).
After an utterance comes `input.committed` (after a commit) and
`session.done` (`total_sentences`, `speech_id`, `cancelled`). Errors are
`{"type": "error", "message", "error_type", "code", "fatal"}`. A fatal error
closes the socket; any other leaves it usable.

Each segment's audio is one complete file of the format, so segments can be
played one by one. Concatenated, they still decode: Ogg streams chain, and
MP3 frames join.

`?dialect=sglang-omni` closes the socket after `session.done`, as
SGLang-Omni's `input.done` does. `izwi` (the default) and `vllm-omni` keep it
open for the next utterance.

Limits:

- A session takes a realtime session slot (`IZWI_MAX_REALTIME_SESSIONS`).
- At most two segments are in flight.
- A reader slower than the audio pauses synthesis, and one stalled past
  `IZWI_AUDIO_STREAM_STALL_TIMEOUT_SECS` is closed.
- Unspoken text is capped at `IZWI_SPEECH_STREAM_MAX_PENDING_CHARS` (65,536;
  past it, `input_buffer_full`), and a session's text at 1 MiB.
- An idle session closes after `IZWI_SPEECH_STREAM_IDLE_TIMEOUT_SECS` (30).

### Audio Voices and Consents

OpenAI's custom-voice routes, which also serve SGLang-Omni and vLLM-Omni voice
clients. They sit over the saved-voice store, so a voice made here is in
`/v1/voices` too, and `/v1/realtime` sessions can speak it.

| Method | Path | Notes |
|--------|------|-------|
| `POST` | `/v1/audio/voice_consents` | Multipart `name`, `recording` (at most 10 MiB) and `language` (BCP 47). Returns `{"object": "audio.voice_consent", "id": "cons_…", "name", "language", "created_at", "status"}`. |
| `GET` | `/v1/audio/voice_consents` | `after`, `limit` (1 to 100). Returns `{"object": "list", "data", "first_id", "last_id", "has_more"}`. |
| `GET`, `POST`, `DELETE` | `/v1/audio/voice_consents/{consent_id}` | Fetch it, rename it (JSON `{"name"}`), or delete it. Deleting a consent revokes every voice made with it. |
| `GET` | `/v1/audio/consent_phrases` | The phrases a verified recording says, per language. |
| `POST` | `/v1/audio/voices` | Multipart `name`, `audio_sample` (at most 10 MiB) and `consent`, plus optional `ref_text` and `reference_language`. Returns `{"object": "audio.voice", "id", "name", "type": "audio_sample", "created_at", "consent", "usable", …}`. `type=prompt` returns `400`; use `/v1/voice-designs` instead. OpenAI's voice names (`alloy`, `marin`, …) and `default` are reserved. |
| `GET` | `/v1/audio/voices` | OpenAI-style `data`, plus SGLang-Omni's `voices` and `uploaded_voices` (`?names_only=true` gives `uploaded_voice_names`). |
| `GET`, `DELETE` | `/v1/audio/voices/{voice_id}` | By id or by name. |

`IZWI_VOICE_CONSENT_MODE` sets what `consent` must be:

- `attestation`: any non-empty `consent`, kept as the voice's provenance.
- `recording`: the id of a consent recording.
- `verified`: the id of a consent recording whose transcript matched a
  published phrase.

Unset, a server protected by an API key uses `recording` and an open local
server `attestation`. Speak a voice with `voice: {"id": "<id>"}` on
`/v1/audio/speech`, or with `session.audio.output.voice` on `/v1/realtime`.

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
| `language` | Optional language hint. Granite Speech accepts IBM's codes and names (4.1 base and 5.0: en, fr, de, es, pt, ja; 4.1 plus drops ja), names it in the instruction, and rejects other languages with `400`. LFM3 Audio takes one of its 18 ISO 639-1 codes (en, ar, da, de, el, es, fi, fr, he, it, ja, ko, nl, pl, pt, sv, tr, zh), or none or `auto` to let the model choose, and rejects any other code with `400`. `Nemotron-3.5-Transcribe-30B-A4B-FP8` takes a code or English name of one of its 40 languages, or none or `auto` to let the model detect it, and rejects any other language (Norwegian included) with `400`. |
| `target_language` | Izwi extension. Translate the speech into this language instead of transcribing it. Only models with speech translation accept it (`Granite-Speech-5.0-2B` and `Granite-Speech-5.0-2B-GGUF`: into English from fr, de, es, pt, ja, and from English into fr, de, es, pt, ja, it, zh; `Nemotron-3.5-Transcribe-30B-A4B-FP8`: to or from English for 31 of its languages, all but bn, id, mr, tl, ta, te, tr and vi); others return `400`. When `language` is also set, the pair must be one of these. The response `language` is the target. Word timestamps are not available for a translation (`timestamp_granularities[]` with it returns `400`). `Nemotron-3.5-Transcribe-30B-A4B-FP8` translates a recording longer than 30 seconds in segments cut at pauses and joins them in order, since on a long recording in one pass it returns a transcript in the spoken language; a stream then receives one delta per segment. |
| `keywords` | Izwi extension. Terms to bias recognition toward: repeat the field, or separate terms with commas or newlines (JSON: an array). Only models with keyword biasing accept it (both Granite Speech 5.0 releases, up to 2,000 terms of at most 128 characters; `Nemotron-3.5-Transcribe-30B-A4B-FP8`, up to 64 terms of at most 80 characters, given to the model as "Context terms" with tags removed; not with `target_language`, which returns `400`); others return `400`. A long Granite list shortens the audio one invocation can hold. |
| `speaker_labels` | Izwi extension. `true` asks a model that tags speakers (`Nemotron-3.5-Transcribe-30B-A4B-FP8`) to label them; others return `400`. Multipart takes `true`/`false` (or `1`/`0`, `yes`/`no`, `on`/`off`), JSON a boolean. The speaker turns come back in `verbose_json` as `speaker_turns` (below); `text` stays untagged. It cannot be combined with `target_language`, and naming a language the model does not label speakers in returns `400`. |
| `response_format` | `json`, `verbose_json`, `text`, `srt`, or `vtt`. Default `json`. |
| `stream` | `true`, `1`, `yes`, or `on` enables SSE. |
| `service_tier` | Optional `flex`, `default`, `auto` or `priority`; see [Audio Service Tier](#audio-service-tier). `json` and `verbose_json` answers and the `transcript.text.done` event carry the tier served when it is set. |
| `timestamp_granularities[]` | Optional `word`, `segment`, or both. Requires `response_format=verbose_json`; model-provided timestamps are used before forced alignment fallback (Parakeet reports word times from its TDT decoder; `Granite-Speech-4.1-2B-Plus` runs its native timestamp mode; `Nemotron-3.5-Transcribe-30B-A4B-FP8` times words with its own CTC head; base Granite text is timed by the aligner). Without it, `verbose_json` returns `segments` only, as OpenAI does. |
| `aligner_model` | Optional forced-aligner model for timestamp generation. Defaults to `Qwen3-ForcedAligner-0.6B`. |
| `prompt` | Optional ASR prompt/context. Granite Speech uses it only when it is a Granite instruction (it starts with `<|audio|>`, `transcribe the speech`, `can you transcribe`, `translate the speech`, `Speaker attribution:` or `Timestamps:`; 5.0 also takes `The speech is in`) or `keywords: a, b` for keyword biasing; other text is ignored and reported as `prompt_ignored` in the diagnostics. LFM3 Audio ignores it. For `Nemotron-3.5-Transcribe-30B-A4B-FP8` it replaces the model's instruction entirely (tags removed, at most 4,000 characters), so combining it with `target_language`, `keywords` or `speaker_labels` returns `400`. |
| `max_tokens` | Optional ASR decoder token budget. |
| `temperature` | `0` to `1`; a value outside that range returns `400`. It does not change the transcript: no ASR model samples its output. Whisper Large v3 Turbo decodes greedily and retries a window at higher temperatures only when its own quality checks fail, which is what OpenAI documents for `0`; Parakeet, Qwen3-ASR, Granite Speech, Nemotron 3.5 Transcribe, VibeVoice-ASR, Voxtral and the LFM audio models always take the most likely token. |
| `chunking_strategy` | `"auto"` or omitted: the server splits long audio itself, at pauses near `IZWI_ASR_CHUNK_TARGET_SECS` (see the notes below for models that read a recording in one pass). The `{"type": "server_vad", …}` object returns `400` (`unsupported_parameter`): chunking is set per server, not per request. Multipart sends `auto` or the object as JSON text. |
| `include[]` | `logprobs` returns `400` (`unsupported_parameter`): no model returns token log-probabilities. Any other value returns `400`. JSON takes `include` as an array. |

`LFM3-Audio-2B-GGUF` and `LFM3-Audio-2B-Q8_0-GGUF` produce no word timestamps
of their own, so `timestamp_granularities[]` falls back to the forced aligner.
They transcribe long audio in chunks of at most 30 seconds.

`Nemotron-3.5-Transcribe-30B-A4B-FP8` transcribes up to 60 minutes in one pass,
without chunks. Its default instruction is the model card's readable-text form
(punctuation, fillers left out). It times words with its own CTC head, so
`timestamp_granularities[]=word` needs no aligner; a translation has no word
timestamps. `izwi_asr_diagnostics.finish_reason` is `stop`, `length` or
`repetition` (stopped after repeating itself). It is offline only: the realtime
WebSocket APIs refuse it. Its memory needs and backends are in the
[support matrix](/support-matrix#nemotron-35-transcribe).

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

Audio without speech returns an empty transcript without running the model, for the models that otherwise hallucinate a short phrase on silence or noise (Whisper, Qwen3-ASR, Granite Speech, Nemotron 3.5 Transcribe, VibeVoice, Voxtral, LFM2.5 Audio). An input counts as speech when some second of it holds 200 ms of frames the VAD scores as speech that are also 6 dB above the input's own noise floor. Silence and steady noise (hum, fan, pink or white noise) at any level are skipped this way; tonal music and crowd babble are not, and reach the model. `verbose_json` reports a skipped input as `izwi_asr_diagnostics.no_speech_gate`, with the speech evidence found (`evidence_ms`) and needed (`required_evidence_ms`). `IZWI_VAD_SNR_MARGIN_DB` sets the margin (default `6`); `0` turns the noise-floor check off.

`verbose_json` always includes `segments`. Without word or segment timestamps from the model or the aligner, each long-form chunk becomes one segment covering its time range, and a short file gets one segment spanning the whole audio. `words` appears only when `timestamp_granularities[]` includes `word`. If forced alignment fails, the transcript is still returned with segments, without `words`, and the failure is reported as `izwi_asr_diagnostics.alignment_error`.

With `speaker_labels=true`, `verbose_json` also has `speaker_turns`, one entry
per run of one speaker's text:

```json
"speaker_turns": [
  { "speaker": 0, "text": "Good morning.", "start": 0.32, "end": 1.1 },
  { "speaker": 1, "text": "Morning, shall we start?", "start": 1.6, "end": 3.04 }
]
```

`speaker` is the model's speaker tag number, or `null` for text before the
first tag. `start` and `end` (seconds) are present only when the model timed
the turn's words. Other response formats carry no speaker turns.

SSE events:

| Type | Fields |
|------|--------|
| `transcript.text.delta` | `delta` |
| `transcript.text.done` | `text`, `language`, `audio_duration_secs`, and `service_tier` when the request named one |
| `error` | `error.message` |

Example multipart request:

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@meeting.wav" \
  -F "model=Parakeet-TDT-0.6B-v3" \
  -F "response_format=verbose_json"
```

### Audio Translations

`POST /v1/audio/translations` is OpenAI's translation route, served by the
catalog's speech translation models through the transcription pipeline:
`Granite-Speech-5.0-2B` (and its GGUF) and
`Nemotron-3.5-Transcribe-30B-A4B-FP8`. Any other model returns `400`, naming
the ones that translate. Whisper Large v3 Turbo was not trained to translate.

Multipart (or JSON with `audio_base64`):

- `file` and `model`, both required. `model` may be an
  `IZWI_OPENAI_MODEL_ALIASES` name such as `whisper-1`.
- `prompt` and `temperature` (`0` to `1`, without effect, as for
  transcriptions).
- `response_format`: `json`, `text`, `srt`, `verbose_json` or `vtt`.
- `language`: a hint for the source language.
- `to_language` or `target_language`: the target. English unless set; a
  model that translates out of English takes others.
- `stream`: `transcript.text.delta` and `transcript.text.done` events.
- `service_tier`: as on transcriptions (see
  [Audio Service Tier](#audio-service-tier)).

`verbose_json` carries `"task": "translate"`. Translations have no word
timestamps.

`/v1/audio/transcriptions` also takes `languages[]` (OpenAI `gpt-transcribe`'s
hint list; the first becomes `language`), and `to_language` as a name for
`target_language`.

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

### Files

OpenAI's Files API, so the stock SDK's `client.files` calls work unchanged.
Files hold what the Batch API reads and writes.

| Method | Path | Notes |
|--------|------|-------|
| `POST` | `/v1/files` | Multipart `file` and `purpose`. Returns `{"id": "file-…", "object": "file", "bytes", "created_at", "filename", "purpose"}`. |
| `GET` | `/v1/files` | `purpose`, `limit` (1 to 10000), `after` and `order` (`asc` or `desc`, default `desc`). Returns `{"object": "list", "data", "has_more", "first_id", "last_id"}`. |
| `GET` | `/v1/files/{file_id}` | The file's metadata. |
| `GET` | `/v1/files/{file_id}/content` | The file's bytes. |
| `DELETE` | `/v1/files/{file_id}` | Returns `{"id", "object": "file", "deleted": true}`. |

| Purpose | Written by | Holds |
|---------|-----------|-------|
| `batch` | Client | A batch's JSONL input. Lines are checked when the batch is created, not at upload. |
| `user_data` | Client | Audio a batch line names by `file_id`. |
| `batch_output` | Server | A batch's output, error and audio files. They carry `expires_at` and are deleted 30 days after creation. |

Any other purpose is refused with `400`. An upload may be at most
`IZWI_FILES_MAX_BYTES` (200 MB by default); a larger one is refused with
`413`. Uploads are kept until deleted, and survive a restart. A file is
visible only to the principal that created it; anyone else gets `404`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
uploaded = client.files.create(file=open("requests.jsonl", "rb"), purpose="batch")
print(client.files.retrieve(uploaded.id).bytes)
print(client.files.content(uploaded.id).text)
client.files.delete(uploaded.id)
```

### Batch API

OpenAI's Batch API, so the stock SDK's `client.batches` calls work
unchanged. A batch runs the requests of an uploaded JSONL file against one
endpoint as batch work, which yields to interactive and real-time requests,
and writes the results to an output file within 24 hours.

OpenAI batches chat, embeddings and image requests. Izwi batches the audio
routes instead, in the same shape (an Izwi extension):

- `/v1/audio/transcriptions`
- `/v1/audio/translations`
- `/v1/audio/speech`

Any other `endpoint` is refused with `400`. `/v1/audio/speech/batch` stays
for small sets answered in one response.
[Real-time and batch workloads](/realtime-and-batch) compares the bulk
routes and explains how batch work shares a server with live traffic.

| Method | Path | Notes |
|--------|------|-------|
| `POST` | `/v1/batches` | JSON `input_file_id` (a file of purpose `batch`), `endpoint`, `completion_window` (`"24h"` only) and optional `metadata` (up to 16 string pairs). Returns the batch, `validating`. |
| `GET` | `/v1/batches` | `after`, `limit` (1 to 100, default 20), newest first. Returns `{"object": "list", "data", "first_id", "last_id", "has_more"}`. |
| `GET` | `/v1/batches/{batch_id}` | The batch. |
| `POST` | `/v1/batches/{batch_id}/cancel` | Cancels a batch that is `validating` or `in_progress`. Lines in flight stop, and the batch ends `cancelled` with the results of the lines that finished. A batch that is `finalizing` or has ended is a `409`. |

The batch object is OpenAI's: `id` (`batch_…`), `object`, `endpoint`,
`errors`, `input_file_id`, `completion_window`, `status`, `output_file_id`,
`error_file_id`, `created_at`, `in_progress_at`, `expires_at`,
`finalizing_at`, `completed_at`, `failed_at`, `expired_at`, `cancelling_at`,
`cancelled_at`, `request_counts` (`total`, `completed`, `failed`) and
`metadata`. Statuses are `validating`, `failed`, `in_progress`, `finalizing`,
`completed`, `expired`, `cancelling` and `cancelled`.

**Input.** One request per line, at most `IZWI_BATCH_MAX_REQUESTS` (50,000)
lines in a file of at most `IZWI_FILES_MAX_BYTES` (200 MB):

```json
{"custom_id": "call-1", "method": "POST", "url": "/v1/audio/transcriptions", "body": {"model": "Parakeet-TDT-0.6B-v3", "file_id": "file-abc123"}}
```

`custom_id` is required and unique in the file, `method` is `POST`, and
`url` is the batch's endpoint. `body` is the JSON body the synchronous route
takes, with these differences:

- A transcription or translation line names its audio by `file_id`, a file
  uploaded with purpose `user_data`, or carries it as `audio_base64`. Use
  `file_id` for anything but short clips: a line holds at most one upload's
  worth of audio (`IZWI_OPENAI_AUDIO_UPLOAD_LIMIT_MB`).
- `stream` and `stream_format` are refused.
- `service_tier` is ignored. Every line runs as batch work.

A file that breaks these rules fails the batch: its status becomes `failed`
and `errors.data` lists up to 100 entries of `{"code", "message", "param",
"line"}` (`invalid_json_line`, `missing_custom_id`, `duplicate_custom_id`,
`invalid_method`, `invalid_url`, `invalid_body`, `streaming_not_supported`,
`line_too_long`, `too_many_requests`, `empty_file`,
`input_file_not_found`). What a body asks for is checked when its line runs,
by the route's own code, and a line the route refuses fails alone.

**Output.** When the batch ends, `output_file_id` names a JSONL file with one
line per request that was served, in input order:

```json
{"id": "batch_req_…", "custom_id": "call-1", "response": {"status_code": 200, "request_id": "batch_req_…", "body": {"text": "…"}}, "error": null}
```

- For a transcription or translation line, `response.body` is the JSON the
  synchronous route returns for the line's `response_format` (`json` or
  `verbose_json`). A text format (`text`, `srt`, `vtt`) is wrapped as
  `{"text": "…"}`.
- For a speech line, the audio is written to its own file, named
  `<custom_id>.<format>`, and `response.body` is `{"audio_file_id":
  "file-…", "format": "mp3", "bytes": 12345}` (an Izwi extension). Download
  it with `GET /v1/files/{audio_file_id}/content`.

`error_file_id` names a file of the same shape for the requests that failed:
`response` carries the route's error status and error body, and `error` its
`{"code", "message"}`. Either id is null when the file would be empty. Output,
error and audio files have purpose `batch_output` and are deleted 30 days
after they are written.

**Lifecycle.** A batch survives a restart: lines that finished are not run
again, and the rest resume. A line refused for capacity (`429`, `503`) is
run again rather than failed. At `expires_at`, 24 hours after creation,
lines still running are stopped, the results so far are written, and the
batch becomes `expired`. Requests that were cancelled or ran out of time
appear in neither file; `request_counts` tells how many there were. A
server runs four batches at a time and the rest wait their turn; the lines
in flight across them are bounded by `IZWI_BATCH_WORKER_CONCURRENCY`. A
batch, like its files, is visible only to the principal that created it.

Lines do not run in input order. Transcription and translation lines are
read up to 1,024 at a time and run longest audio first within that window,
judged by the size of the file a line names or of the audio it carries, so
that recordings of similar length are batched together on the model. Speech
lines run in input order. The output file is in input order either way.

**Completion webhook.** A batch created with `metadata.webhook_url` has an
event posted to that URL when it ends (an Izwi extension; the event and its
signature follow OpenAI's webhooks):

```json
{"id": "evt_…", "object": "event", "type": "batch.completed", "created_at": 1762790400, "data": {"id": "batch_…"}}
```

`type` is `batch.completed`, `batch.failed`, `batch.cancelled` or
`batch.expired`. Fetch the batch for its counts and files. Each delivery
carries the Standard Webhooks headers `webhook-id` (the event id),
`webhook-timestamp` (epoch seconds) and `webhook-signature`:
`v1,` followed by the base64 HMAC-SHA256 of `{webhook-id}.{webhook-timestamp}.{body}`
under the server's `IZWI_BATCH_WEBHOOK_SECRET`. A secret that starts with
`whsec_` is base64 of the key, as the OpenAI SDK's
`client.webhooks.unwrap(body, headers, secret=…)` and the Standard Webhooks
libraries read it; any other secret is used as the key itself.

Webhooks are off until the operator sets `IZWI_BATCH_WEBHOOK_SECRET`, and a
batch that names `webhook_url` is refused with `400` while they are off. The
URL must be `https://` (`IZWI_BATCH_WEBHOOK_ALLOW_HTTP=1` also allows
`http://`), carry no credentials, and its host must resolve only to public
addresses (`IZWI_BATCH_WEBHOOK_ALLOW_PRIVATE=1` also allows loopback,
private and link-local ones). A host given as an address is checked when
the batch is created; a host name is resolved and checked at each delivery,
and the connection goes to the addresses that were checked. Redirects are
not followed and no proxy is used. A delivery that gets no `2xx` within 10
seconds is tried again after 2 and then 8 seconds, with the same
`webhook-id`, and dropped after the third attempt. Delivery is best effort:
it never changes the batch, and an event can be lost if the server stops
between the batch ending and the delivery, so keep a poll as the fallback.

**Through the gateway.** The gateway relays `/v1/files` and `/v1/batches`
to its audio upstreams only with `IZWI_GATEWAY_AUDIO_SHARED_STATE=true`,
which says every upstream shares one records database and media store (a
single upstream does). The upstream sees the gateway as one caller, so the
gateway tells it which tenant each request is for (the key's `tenant_id`,
or the key itself when it has none). Start every upstream with
`IZWI_TRUST_GATEWAY_TENANT_HEADER=true` and each tenant sees only its own
files and batches: another tenant's id answers `404`. Only the gateway may
reach such an upstream, because anyone holding the upstream's API key could
name any tenant. Without the setting, files and batches made through the
gateway are shared by every gateway principal. Uploads stream through and
are bounded by the upstream's `IZWI_FILES_MAX_BYTES`.

Transcribe recordings:

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

lines = []
for name in ["call-1.wav", "call-2.wav"]:
    audio = client.files.create(file=open(name, "rb"), purpose="user_data")
    lines.append({
        "custom_id": name,
        "method": "POST",
        "url": "/v1/audio/transcriptions",
        "body": {"model": "Parakeet-TDT-0.6B-v3", "file_id": audio.id},
    })
requests = "\n".join(json.dumps(line) for line in lines).encode()
source = client.files.create(file=("requests.jsonl", requests), purpose="batch")

batch = client.batches.create(
    input_file_id=source.id,
    endpoint="/v1/audio/transcriptions",
    completion_window="24h",
)
# Poll client.batches.retrieve(batch.id) until status is "completed".
batch = client.batches.retrieve(batch.id)
for line in client.files.content(batch.output_file_id).text.splitlines():
    result = json.loads(line)
    print(result["custom_id"], result["response"]["body"]["text"])
```

Speak a list of texts:

```python
lines = [
    {
        "custom_id": f"prompt-{index}",
        "method": "POST",
        "url": "/v1/audio/speech",
        "body": {"model": "Kokoro-82M", "voice": "af_heart", "input": text, "response_format": "mp3"},
    }
    for index, text in enumerate(["Welcome back.", "Your order has shipped."])
]
requests = "\n".join(json.dumps(line) for line in lines).encode()
source = client.files.create(file=("speech.jsonl", requests), purpose="batch")
batch = client.batches.create(
    input_file_id=source.id, endpoint="/v1/audio/speech", completion_window="24h"
)
# Once the batch is "completed":
batch = client.batches.retrieve(batch.id)
for line in client.files.content(batch.output_file_id).text.splitlines():
    result = json.loads(line)
    body = result["response"]["body"]
    audio = client.files.content(body["audio_file_id"]).content
    open(f"{result['custom_id']}.{body['format']}", "wb").write(audio)
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
| `input` | Text, one input item, or an array of input items. Message content parts are as in chat completions; audio parts are a `400`. |
| `instructions` | Optional system instruction. Required if `input` is empty. |
| `max_output_tokens` | Optional output limit. |
| `truncation` | `disabled` (default) or `auto`; same behavior as chat completions. |
| `stream` | `true` returns SSE events. |
| `metadata`, `user` | Stored or accepted for compatibility. |
| `temperature`, `top_p`, `top_k`, `repetition_penalty`, `presence_penalty` | Optional runtime sampling controls. |
| `store` | `false` skips process-local retention. Default retains completed records. |
| `tools`, `tool_choice` | Function tools in the Responses form (`{"type": "function", "name", "parameters"}`) or the chat completions form. A tool call in the answer is a `function_call` output item (`id` `fc_<hex>`, `call_id` `call_<hex>`, `name`, JSON `arguments`) after the message, which is left out when the answer only calls tools; streaming sends `response.output_item.added`, `response.function_call_arguments.delta`/`.done` and `response.output_item.done` for it. Send the result back as a `{"type": "function_call_output", "call_id", "output"}` input item, after the `function_call` item itself; for LFM3, `output` may be a list with deferred-tool items and the item may carry `additional_tools`, as in chat completions. `tool_choice` (`required`, or `{"type": "function", "name": …}`), `parallel_tool_calls` and `strict` work as in chat completions. |
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
speaker-turn transcript mode.

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
`keyword_biasing` accept them (both Granite Speech 5.0 releases and
`Nemotron-3.5-Transcribe-30B-A4B-FP8`); others return `400`,
as does a translation with `include_timestamps` or in speaker-attributed mode.
The record stores both as `target_language` and `keywords`, keeps the spoken
`language` the job named, and a retry asks for the same translation.

With `include_timestamps` (or `word_timestamps`), a model whose
`route_capabilities` report `native_word_timestamps`
(`Nemotron-3.5-Transcribe-30B-A4B-FP8`) times the words itself: no
`aligner_model_id` is needed, and the record keeps none. When the model's
alignment yields no words, the server times the transcript with the forced
aligner instead.

For speaker-attributed ASR, use:

```http
POST /v1/speech-to-text/jobs?job_kind=speaker_attributed_asr
```

SAA requests use the transcription store and take a model whose
`route_capabilities` report `speaker_attributed_asr`:
`Granite-Speech-4.1-2B-Plus` (the default) or
`Nemotron-3.5-Transcribe-30B-A4B-FP8`. The Nemotron model tags speakers in one
transcription pass; its turns are labelled `Speaker 1`, `Speaker 2`, … and timed
when its own word timestamps align, and `min_speakers`/`max_speakers` only add warnings.
JSON and multipart requests accept
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

Diarization creates on both routes (`/v1/diarizations` and
`/v1/speech-to-text/jobs?job_kind=diarization`) take `model` (alias
`model_id`, `diarization_model`), `asr_model`, `aligner_model`, `llm_model`,
`language`, `min_speakers`, `max_speakers`, `min_speech_duration_ms`,
`min_silence_duration_ms`, `vad_gate` and `generate_summary`. The rules:

- `model` is a diarization model (`diar_streaming_sortformer_4spk-v2.1`, the
  default, or `Nemotron-3-Diarization`) or a model whose `route_capabilities`
  report `one_model_diarization` (`Nemotron-3.5-Transcribe-30B-A4B-FP8`).
  Other ids return `400`.
- `min_speakers` and `max_speakers` may not exceed the model's speaker
  channels (4 for Sortformer v2.1, 8 for the other two), and `min_speakers`
  may not exceed `max_speakers`; otherwise `400`.
- `asr_model` must be a model whose `route_capabilities` report
  `diarization_pipeline_asr`; `Nemotron-3.5-Transcribe-30B-A4B-FP8` returns
  `400` there.

A one-model diarization `model` runs no pipeline: no diarizer, ASR or aligner
is loaded. The server runs one transcription with speaker tags and word
timestamps and builds the record from it:

- `segments` are the embedded diarizer's segments (its 10 ms predictions
  above 0.5), labelled `SPEAKER_NN` by diarizer channel.
- `words` take the channel their `<spk:N>` tag was assigned to, or `UNKNOWN`
  when the tag has no channel (counted in `unattributed_words`).
- `utterances` and `transcript` are built as in the pipeline; `asr_text` is
  the model's transcript.
- `asr_model` or `aligner_model` with such a model returns `400` ("diarizes
  on its own").
- The model counts speakers itself: `min_speakers` and `max_speakers` (at most
  8) only add a warning when the count falls outside them.
- `min_speech_duration_ms`, `min_silence_duration_ms` and `vad_gate` are not
  used.
- If the transcript cannot be aligned (for example, a character the CTC
  vocabulary lacks), the record still has the transcript and the diarizer's
  segments, with a warning and no words.

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

#### Drama 3 take editing

A `FishAudio-Drama-3` take can continue earlier takes: to say a line again, or
the next one, in the context of what was already said.

- **Request.** `previous_take_ids` on `POST /v1/text-to-speech` and
  `POST /v1/voice-clones`: up to 2 record ids of the same route, oldest first.
  The prompt holds those takes as finished turns (their text and the audio
  codes generated for it) before the new text, the layout the model was
  trained on for take editing. They join the prompt the way a long text's
  earlier chunks do: the first take's first chunk stays, and the latest chunk
  follows; when the prompt would pass 8,192 rows, the latest is left out
  first, then the first. A new text longer than one chunk is a chunk session
  after them.
- **Voices and seed are not inherited.** Send the `references` (or
  `saved_voice_id`) and `seed` you want, for example the ones the earlier
  take's record reports.
- **Record fields.** A Drama 3 record reports:
  - `editable: true` once it has finished and kept its codes, so
    `previous_take_ids` can name it. The codes themselves are never returned.
  - `references`: `[{"speaker": N, "saved_voice_id": "<id>"}]`, one per
    reference in request order. An inline clip shows only its speaker, as its
    audio is not kept; `[]` means every speaker spoke in a seeded voice.
  - `seed`, when the request gave one.
  - `parent_take_id`: the last of the request's `previous_take_ids`. The
    parent may have been deleted since.

  The fields are absent on other models' records and on Drama 3 takes made
  before take editing, which cannot be continued.
- **Who can name a take.** Any record the route's `GET` returns; records
  belong to no caller.

```json
{
  "model_id": "FishAudio-Drama-3",
  "script": [{ "speaker": 0, "text": "Ready when you are. [laughs]" }],
  "references": [{ "speaker": 0, "saved_voice_id": "voice_anna" }],
  "seed": 8,
  "previous_take_ids": ["5d0b8e3a-6f0e-4a57-9a3c-2f1e9b7c4d21"]
}
```

These are refused before any model loads:

- `404` for an id that is not a record of the request's route;
- `400` for a take of another model;
- `400` for a take that kept no codes: still pending or processing, failed, or
  made before take editing;
- `400` for an empty or repeated id, or more than 2 ids;
- `400` for `previous_take_ids` with a model other than Drama 3.

A durable record reads its earlier takes again at every stage attempt, and its
recovery checkpoint and `Idempotency-Key` digest include their codes' hashes;
an earlier take deleted while the record runs fails it with the `404`
message.

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

A stage is one whole transcription or synthesis job. One that runs longer
than `IZWI_BATCH_STAGE_TIMEOUT_SECS` (six hours by default, counted from the
stage's start; `0` disables the limit) is cancelled and retried or failed
like any other stage failure. Raise it for recordings or scripts that take
longer than that on your device.

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

A voice saved without a transcript works with `Chatterbox-500M` and `Chatterbox-Multilingual-500M`, which clone from audio alone. Models that need a transcript (Qwen3-TTS Base, VibeVoice, Fish Audio S2 Pro, Fish Audio Drama 3 and Maya 2 Native) return `400` for it, asking for a voice saved with one.

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
| `content_parts` | Multimodal content parts in OpenAI-like shape: images for Qwen3.5. A thread can switch models: earlier turns keep their media for a model that hears it and otherwise reach the model as their text, or as `[Image attachment]` when a turn has none. A new message carrying audio is a `400`; audio clips in threads saved while LFM3-Audio answered chat still play back and reach the model as their text, or as `[Audio attachment]`. |
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

### Agent Sessions (deprecated)

These routes are deprecated and will be removed in a later release. Use `/v1/responses` for text turns with tools and stored state, and `/v1/realtime` for voice agents. Every response from them carries a `Deprecation: true` header and a `Link` header naming `/v1/responses` as the successor.

| Method | Path | Notes |
|--------|------|-------|
| `POST` | `/v1/agent/sessions` | Create an agent session and linked thread. |
| `GET` | `/v1/agent/sessions/{session_id}` | Fetch retained process-local session metadata. |
| `POST` | `/v1/agent/sessions/{session_id}/turns` | Run one agent turn. |

Create fields include `agent_id`, `model_id`, `system_prompt` and `title`. `planning_mode` is accepted and ignored.

A turn lets the model call the server's built-in tools (`time`). Turn responses include the assistant text, the tool calls (`call_id`, `name`, `arguments`, `status`, `output`) and ordered events: `turn_started`, `tool_call_started`, `tool_call_completed`, `assistant_message` and `turn_completed`.

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
`stages_by_status`, a local worker snapshot, and `api_batches` (the
`/v1/batches` batches by status, with the requests they have finished).
Prometheus output appends `izwi_batch_runtime_queued_stages`,
`izwi_batch_runtime_jobs`, `izwi_batch_runtime_stages`,
`izwi_batch_runtime_worker_running`, `izwi_batch_runtime_api_batches{status}`
and `izwi_batch_runtime_api_batch_requests{outcome}` (`completed`, `failed`).

Latencies are Prometheus histograms (cumulative `_bucket{le=...}`, `_sum`,
`_count`) in seconds, so they aggregate across replicas and `histogram_quantile`
works on them. The JSON snapshot keeps its windowed averages and percentiles
for the UI. Labels are bounded: model names come from the model catalog,
families keep at most 64 series (the rest share `other`), and no request or
tenant IDs are used.

| Metric | Labels | Meaning |
|---|---|---|
| `izwi_request_duration_seconds` | `model`, `backend`, `task`, `workload_class`, `outcome` | Server admission to the terminal output. |
| `izwi_request_ttft_seconds` | `model`, `backend`, `task`, `workload_class` | Server admission to the first user-visible output: the first text delta of a chat or transcription request (`task="asr"` covers streamed `/v1/audio/transcriptions` and each `/v1/realtime` transcription turn), the first generated frame or audio of a speech request. A model that returns its transcript whole reports the transcript's time. |
| `izwi_request_queue_wait_seconds` | `model`, `backend`, `task`, `workload_class` | Server admission until the engine first schedules the request, including the runtime coordinator, input preparation and model residency. |
| `izwi_tts_first_audio_seconds` | `model`, `backend`, `workload_class` | Server admission until a streamed speech request hands its first audio to the transport. Recorded on every streaming speech route (SSE and chunked `/v1/audio/speech`, `/v1/audio/speech/stream`, `/v1/realtime` and realtime voice), once per segment of a long text. |
| `izwi_tts_output_audio_seconds_total` | `model`, `backend`, `workload_class` | Counter of the seconds of audio completed speech requests produced, streamed or buffered. Its rate is the audio seconds the server speaks per second. |
| `izwi_request_prefill_seconds` | `model`, `backend`, `task` | Engine prefill time of one request. |
| `izwi_request_decode_seconds` | `model`, `backend`, `task` | Engine decode time of one request, after its prefill. |
| `izwi_request_input_audio_seconds` | `model`, `backend`, `task` | Input audio length of completed transcription and audio-chat requests. `rate(_sum)` is the audio seconds the server transcribes per second (its RTFx under load). |
| `izwi_request_prompt_tokens_total`, `izwi_request_generation_tokens_total` | `model`, `task` | Counters for completed requests (generation counts audio frames for speech models). |
| `izwi_engine_inter_token_latency_seconds` | `model`, `task` | Gap between successive committed tokens of one request; a commit of k tokens after a gap g counts k gaps of g/k. |
| `izwi_engine_step_host_seconds` | `phase` (`prepare`, `commit`, `deliver`) | Host time of engine step phases under the step gate. |
| `izwi_engine_batch_rows`, `izwi_engine_batch_tokens` | none | Rows and work units (tokens, or frames for speech) per dispatched physical batch. |
| `izwi_engine_model_calls_per_step` | none | Model calls one engine step dispatched: one per tensor batch or serial row, one per row of a request-parallel batch. |
| `izwi_asr_preparation_batch_rows` | `model` | Requests one cross-request ASR encoder batch ran in one device call (Parakeet, Nemotron 3.5 ASR Streaming). Concurrent Parakeet clips of up to 82 s and Nemotron clips of up to 30 s batch together; longer files encode alone. |
| `izwi_engine_mixed_rows`, `izwi_engine_mixed_tokens` | `phase` (`decode`, `prefill`) on rows | Rows of each phase and tokens per committed mixed prefill+decode batch (models certified for mixed steps). |
| `izwi_engine_device_idle_seconds` | `device` (`cpu`, `metal`, `cuda:0`) | Time the device waited between the end of one model step and the start of the next while the engine had work: the host gap (commit, delivery, prepare, dispatch) between steps. |
| `izwi_engine_device_queue_wait_seconds` | `device`, `class` (workload class) | Time a model dispatch waited on its device's queue before it started. Control work (KV page operations, loads, frees) runs first; dispatches of different models share the device by workload-class weight, between jobs. |
| `izwi_engine_scheduler_reserve_deferrals_total` | none | Counter of Batch and Background requests whose start waited for the real-time reserve: while a model has a real-time or streaming request (or had one in the last 30 seconds), batch work starts only up to the model's rows less `IZWI_REALTIME_ROW_RESERVE`. A model that retains more sequences than its rows keeps the reserve free among the extra sequences at all times, and requests held for that count too. Each request counts once. |
| `izwi_engine_asr_encoder_live_yields_total` | none | Counter of device jobs a Batch or Background Parakeet long-audio encode ended after a window group because its model had a real-time or streaming request (or had one in the last 30 seconds); the encode continues in a new job, so the live request waits for one window group at most. |
| `izwi_engine_scheduler_stream_pacing_transitions_total` | none | Counter of the times a speech stream ran more than `IZWI_STREAM_LEAD_SECS` (default 3) seconds of audio ahead of real time and was scheduled from the batch class's share of service until its lead fell to half of that. Nothing is throttled: with no competing work the stream still runs at full speed. The return to its own class is not counted. |
| `izwi_inference_coordinator_queue_depth`, `izwi_inference_coordinator_queue_capacity` | `pool` (`general`, `realtime`) | Gauges of the requests the runtime queue holds and can hold. Real-time requests and open real-time sessions use the `realtime` pool, every other class the `general` pool (`IZWI_MAX_QUEUED_REQUESTS`); a request that finds its pool full is refused with 503. |
| `izwi_engine_device_ops_off_thread_total` | none | Counter of device operations (model calls, KV page operations, state allocation and frees) that ran off their device's thread. Zero in a healthy process. |
| `izwi_engine_device_frees_rerouted_total` | none | Counter of device buffers whose last drop began off their CUDA device's thread (request-owned ASR encoder outputs) and was moved onto it. |
| `izwi_runtime_stage_duration_seconds` | `stage`, `workload_class` | Duration of runtime pipeline stages. |
| `izwi_voice_turn_first_audio_seconds` | none | Realtime voice: endpoint decision to the turn's first assistant audio. |
| `izwi_voice_endpointing_latency_seconds`, `izwi_voice_asr_first_partial_seconds`, `izwi_voice_asr_final_seconds`, `izwi_voice_asr_live_first_partial_seconds`, `izwi_voice_llm_first_token_seconds`, `izwi_voice_tts_first_audio_seconds`, `izwi_voice_barge_in_latency_seconds`, `izwi_voice_session_duration_seconds` | none | Realtime voice stage latencies; `izwi_voice_metric_contract_info` lists each with its description. |
| `izwi_voice_audio_egress_underruns_total`, `izwi_voice_audio_ingress_dropped_frames_total` | none | Assistant audio sent after playback had drained; input frames missing from the client's sequence. |

`workload_class` is the class the request was admitted under: `realtime`,
`interactive`, `streaming`, `online`, `batch` or `background`.

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

A model whose text-to-speech records journal their audio as it is generated
(Fish S2 Pro and Fish Audio Drama 3) also reports
`"supports_in_progress_replay": true`: an unfinished record of it can be played
from its start.

A model that speaks without any voice input (Fish Audio Drama 3) also reports
`"supports_seeded_voice": true`: a request with no reference gets a voice its
sampling `seed` sets, so a reference is optional. The field is present only
when it is `true`. `reference_transcript_required` says whether a reference
voice needs its transcript.

A model that voices several speakers in one request (Fish Audio Drama 3) also
reports `"supports_speaker_dialogue": true`, with `max_speakers` (16: speakers
0 to 15) and `max_references` (8), the limits of its `references` and
`script` fields (see [Drama 3 dialogue](#drama-3-dialogue)). The three fields
are present only for such a model.

Model records carry the weights' license label in `license` and, when the
label alone does not say what it permits, a `license_notice`. Fish Audio
Drama 3's is "Free for research and non-commercial use; commercial use only
below USD 500k annual revenue; offering it as a hosted service requires a
licence from Fish Audio". The notice is informational: nothing is refused
because of it.

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
    "diarization_pipeline_asr": false,
    "speech_to_text_realtime": false,
    "speech_translation": false,
    "keyword_biasing": false,
    "speaker_attributed_asr": false,
    "native_word_timestamps": false,
    "diarization_records": false,
    "one_model_diarization": false,
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

Speech-to-text route flags:

| Flag | Meaning |
|------|---------|
| `speech_to_text_jobs` | The model runs persisted speech-to-text jobs. |
| `diarization_pipeline_asr` | The diarization pipeline may use the model as its ASR stage. `false` for `Nemotron-3.5-Transcribe-30B-A4B-FP8`, which labels speakers itself. |
| `speech_to_text_realtime` | The model serves realtime transcription. `false` for `Nemotron-3.5-Transcribe-30B-A4B-FP8`, which is offline only. |
| `speech_translation` | Transcription requests may ask for a translation (`target_language`). |
| `keyword_biasing` | Transcription requests may pass `keywords`. |
| `speaker_attributed_asr` | Speaker-attributed ASR jobs accept the model: it writes speaker turns itself. |
| `native_word_timestamps` | Word timestamps come from the model itself, so no forced aligner is needed. |
| `diarization_records` | The model is a diarizer that can drive a diarization pipeline. |
| `one_model_diarization` | A diarization job may name the model alone as its `model`: it transcribes, tags speakers and times words itself, with no pipeline ASR or aligner. `true` only for `Nemotron-3.5-Transcribe-30B-A4B-FP8`. |
| `voice_realtime_modular_asr` | Voice sessions may use the model as their ASR stage. `false` for `Nemotron-3.5-Transcribe-30B-A4B-FP8`. |

`Nemotron-3.5-Transcribe-30B-A4B-FP8` reports `speech_translation`,
`keyword_biasing`, `speaker_attributed_asr`, `native_word_timestamps` and
`one_model_diarization` as `true`, and `diarization_pipeline_asr`,
`diarization_records`, `speech_to_text_realtime` and
`voice_realtime_modular_asr` as `false`.

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

- Models that stream natively (Voxtral Realtime, Nemotron, LFM3-Audio) transcribe as audio arrives. Voxtral and LFM3-Audio would otherwise turn background noise into made-up words, so their input passes a noise gate: audio is held for 500 ms, and anything that is not confirmed speech (300 ms of voiced speech, 6 dB above the stream's own noise floor) reaches the model as silence. Their transcripts therefore arrive about 500 ms later. Steady noise such as a fan, hum or pink noise produces no text; music and crowd babble can still pass. `IZWI_ASR_REALTIME_NOISE_GATE=0` turns the gate off.
- LFM3-Audio transcribes English only and takes no prompt. It writes the text of each 0.96 s of audio once that audio has arrived. Every 29 s (sooner at a pause after about 23 s) a stream continues in a fresh context that hears the last 2.88 s again; the transcript keeps the repeated words once.
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
| `function_call_output` | `call_id` and `output` (text, or any JSON): the result of a call the server sent as `function_call` to the client. A call that is unknown, already answered or timed out is an `error`. |
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
| `s2s_model_id` | Unified audio-chat model override (`LFM2.5-Audio-1.5B-GGUF` or `LFM3-Audio-2B-GGUF`). |
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
| `tools` | Modular mode with a chat model that calls tools (Qwen3, Qwen3.5, LFM2.5, LFM3), or unified mode on LFM3-Audio: OpenAI function tools (`{"type": "function", "function": {"name", "description", "parameters"}}`, or the Responses form with the fields on the tool) the reply may call. The client runs them: it gets `function_call` and answers with `function_call_output`. A name must be letters, digits, `_` or `-`, unique, and not a built-in tool's. At most 64. A tool with `"background": true` does not hold the reply: the call is reported with status `started`, the reply tells the user it is under way, and the output the client sends within 120 s is told in the next turn. |
| `builtin_tools` | With a model that calls tools: whether the reply may also call the server's built-in tools (today `time`, the local date and time). Default `true`. |
| `function_call_timeout_ms` | How long a turn waits for a call's result: default `10000`, clamped to `1000..60000`. |
| `mcp_servers` | MCP servers whose tools the reply may call, run by the Izwi server: `[{"label", "url", "headers", "allowed_tools"}]`, at most 8, each with at most 64 tools (name them in `allowed_tools` to narrow a larger server). Streamable HTTP servers only. The server connects only to hosts listed in `IZWI_MCP_ALLOWED_HOSTS`; with that unset, `mcp_servers` is refused. A tool's name must not repeat another tool's. Calls are reported as `function_call` with `executor: "mcp"`. |
| `tool_wait_phrase` | Modular mode: what the reply says when a tool has kept it waiting 1.5 s and it has said nothing yet. Default `"One moment."` when `asr_language` is unset or English, nothing otherwise; `""` turns it off. |

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
| `function_call` | The reply calls a tool: `call_id`, `name`, `arguments` (a JSON object as text) and `executor`, `server` for a built-in tool (the server runs it) or `client` (answer with `function_call_output`); `null` for a tool the session does not offer. The typed v2 protocol sends it as `function_call`. |
| `function_call_completed` | A call's `call_id`, `name`, `status` (`ok`, `error` or `timeout`) and the `output` the reply hears. The typed v2 protocol sends it as `function_call_completed`. |
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

How a turn with tools runs:

- The reply is told the tools in its prompt. A modular reply may say something before it calls ("Let me check"), which is spoken; an LFM3-Audio reply that calls tools speaks nothing in that round. In both, the calls are sent as `function_call`, run (built-in tools on the server, the others by the client), reported as `function_call_completed`, and the reply's next round reads their results and speaks. A turn runs at most 3 rounds and makes at most 8 calls; past either it ends with an `error`.
- Each round takes its own request admission slot, so no slot is held while a tool runs. An interruption ends a call the client has not answered.
- Text events and the remembered reply carry what the reply says, never its call markup. On LFM3-Audio, audio chunk numbers count on across rounds, and only the last round's last chunk is marked final.
- Tool calls and their results last for the turn: the stored conversation keeps what the user said and what the reply said.

Voice realtime persists voice sessions and turns in the local store. Modular turns can also update observational memory when that feature is enabled.

## OpenAI Realtime API

`GET /v1/realtime` speaks the OpenAI Realtime API (GA) over WebSocket, so an unmodified OpenAI Realtime client holds a speech-to-speech session with local models: the OpenAI SDKs, LiveKit Agents' OpenAI realtime plugin and Pipecat's OpenAI realtime service. Point the client's base URL at Izwi:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://127.0.0.1:8080/v1", api_key="your-izwi-key")
async with client.realtime.connect(model="izwi-voice") as connection:
    ...
```

LiveKit: `openai.realtime.RealtimeModel(base_url="http://127.0.0.1:8080/v1", ...)`. Pipecat: `OpenAIRealtimeLLMService(base_url="ws://127.0.0.1:8080/v1/realtime", ...)`.

Events are the GA protocol's: `session.update`, `input_audio_buffer.*`, `conversation.item.*` (create, retrieve, truncate, delete), `response.create` and `response.cancel` from the client; `session.*`, `conversation.*`, `input_audio_buffer.*`, input transcription, `response.*` (output items, content parts, audio, transcript and text deltas, function-call arguments) and `error` from the server. The beta interface (`OpenAI-Beta: realtime=v1`, flat session fields) is not served: OpenAI retired it.

### Choosing models with `model`

`?model=` names a realtime profile:

| `model` | Session |
| --- | --- |
| `izwi-voice` (the default) | Modular: Parakeet-TDT-0.6B-v3 → Qwen3.5-4B → Kokoro-82M |
| `izwi-voice-compact` | Speech-to-speech: LFM3-Audio-2B-GGUF |
| an audio-chat model id (`LFM3-Audio-2B-GGUF`, `LFM2.5-Audio-1.5B-GGUF`) | Speech-to-speech with that model |
| a chat model id (`Qwen3.5-9B`) | Modular, with that chat model and the default ASR and TTS |
| an OpenAI model name (`gpt-realtime`, `gpt-realtime-2.1`, `gpt-4o-realtime-preview`), or none | The default profile (`IZWI_REALTIME_DEFAULT_PROFILE`) |

`IZWI_REALTIME_PROFILES` adds profiles (a JSON array of `{id, asr_model_id, text_model_id, tts_model_id}` or `{id, s2s_model_id}`). An unknown `model` refuses the upgrade with `400`. The resolved models are in the upgrade response's `x-izwi-realtime-profile` header, and `GET /v1/realtime/capabilities` lists the profiles, the transcription models and how each transcribes, voices, audio formats and limits.

### Audio, voices and turns

- Audio is base64 in JSON: `audio/pcm` at 24 kHz (16-bit mono), or G.711 `audio/pcmu` / `audio/pcma` at 8 kHz, in both directions. Speech from a model at another rate is resampled to 24 kHz; output deltas carry at most 100 ms each.
- `audio.output.voice` takes OpenAI's names (`alloy`, `ash`, `ballad`, `coral`, `echo`, `sage`, `shimmer`, `verse`, `marin`, `cedar`), mapped onto each TTS model's own voices; any other name is the model's own voice (`af_heart` for Kokoro). `{"id": "<saved voice id>"}` speaks in a saved voice when the profile's TTS model clones voices. The voice is fixed once the session has spoken. `speed` is accepted and ignored.
- `turn_detection` is `server_vad` (the default; threshold, `prefix_padding_ms`, `silence_duration_ms`, `idle_timeout_ms`, `create_response`, `interrupt_response`), `semantic_vad` (its `eagerness` picks the trailing-silence window: `high` 350 ms, `medium`/`auto` 800 ms, `low` 1500 ms), or `null` for manual turns (`input_audio_buffer.commit`, then `response.create`). Voice activity is Izwi's VAD; `input_audio_buffer.speech_started` is sent once speech is confirmed (about 300 ms of voiced speech) and dated back to where it began, and with `interrupt_response` it cancels the response in progress.
- `audio.input.noise_reduction` is accepted and ignored: the session echoes the value and applies no noise reduction.
- `audio.input.transcription` turns on `conversation.item.input_audio_transcription.*` events. OpenAI transcription model names (`whisper-1`, `gpt-4o-mini-transcribe`, …) run on the profile's ASR; an Izwi ASR model id picks that model (any live ASR model, Voxtral Realtime or LFM3-Audio).
- A model with a native streaming session (Nemotron streaming ASR, Voxtral Realtime, LFM3-Audio LiveASR) transcribes a turn live: `conversation.item.input_audio_transcription.delta` events start while the user is still speaking, once `speech_started` has been sent (or, with manual turns, from the first audio appended), and carry the id of the item the turn becomes. `...completed` follows the commit with the full transcript, which is the streaming session's final text; the reply reads the same text. Every other model transcribes the turn after it is committed, as do conversations on a speech-to-speech profile. A live turn that cannot get a streaming session (`IZWI_ASR_REALTIME_MAX_SESSIONS` are all in use, or the session fails) is transcribed after its commit instead, with no `error` event. `GET /v1/realtime/capabilities` reports the mode as `transcription_mode` (`"live"` or `"turn"`): for each model `audio.input.transcription.model` may name under `transcription_models`, and for each profile's default.

### Responses, tools and history

- Responses run on the conversation's items. Clients may add items with `conversation.item.create` (`previous_item_id`, `"root"`), including `function_call` and `function_call_output` items; `response.create` takes `instructions` (replacing the session's for that response), `tools`, `tool_choice`, `max_output_tokens`, `output_modalities` (`["audio"]` or `["text"]`), `metadata` (echoed on `response.created` and `response.done`), and `conversation: "none"` with `input` (items or `item_reference`s) for an out-of-band response kept out of the conversation.
- A response that calls functions ends there with its `function_call` items. The client answers with `function_call_output` items and a new `response.create`. Function tools need a chat model that calls tools (Qwen3, Qwen3.5, LFM2.5, LFM3) or LFM3-Audio; tools for another chat model fail the response instead of being dropped. With a chat model, `tool_choice: "required"` and a named tool open the reply with that call, as in chat completions; LFM3-Audio is asked in the prompt, which it may not obey.
- MCP servers: a `tools` entry `{"type": "mcp", "server_label", "server_url", "require_approval": "never"}` (optional `authorization` bearer token or `headers`, and `allowed_tools`) lets a chat-model profile call that server's tools. Izwi lists them when the session is updated (`mcp_list_tools.in_progress`, an `mcp_list_tools` item in `conversation.item.done`, then `mcp_list_tools.completed` or `.failed`) and runs the calls itself: a response that calls one carries an `mcp_call` output item (`response.mcp_call_arguments.delta`/`.done`, `response.mcp_call.in_progress`, then `.completed` or `.failed`) whose `output` or `error` is filled in before `response.done`. As with OpenAI, the response ends there and the client sends `response.create` for the answer. Differences from OpenAI: the server connects only to hosts in `IZWI_MCP_ALLOWED_HOSTS` (unset, MCP entries fail the response); approvals are not implemented, so `require_approval` must be `"never"`; connectors (`connector_id`) and the label-only reuse form are refused; the session echo leaves out `headers`; speech-to-speech profiles do not take MCP tools.
- One response at a time writes to the conversation, and one out-of-band response may run beside it. `conversation.item.truncate` cuts an assistant item's transcript to what the client played.
- `response.done` reports usage: the chat model's text tokens, and audio tokens estimated at OpenAI's rates (1 per 100 ms of user audio, 1 per 50 ms of assistant audio).

### Sessions, admission and errors

- Each socket takes a realtime session slot (`IZWI_MAX_REALTIME_SESSIONS`); a full server answers with an `error` and close `1013`. Each stage of a turn (transcription, the reply, each speech segment) takes its own realtime admission permit.
- Errors are `error` events with `type`, `code`, `message`, `param` and the client's `event_id`; the session stays open. Requests that conflict harmlessly (a different `model` in `session.update`, a voice change after the session has spoken, a truncate past the end) are accepted and ignored.
- Sessions are not written to the voice history unless `IZWI_REALTIME_PERSIST=1`.
- `?intent=transcription` opens a transcription session: input audio, VAD or manual commits and transcription events, no responses.
- WebRTC (`/v1/realtime/calls`), SIP and image input are not served. [Phone calls and call centres](/features/telephony) covers how a call reaches a session.

### Browser clients and client secrets

A browser cannot set `Authorization` on a WebSocket. `POST /v1/realtime/client_secrets` (authenticated with the API key) mints an `ek_` key bound to a session config, valid for `expires_after.seconds` (10 to 7200, default 600). The browser offers it as the `openai-insecure-api-key.<key>` subprotocol beside `realtime`; the key admits `/v1/realtime` and `/v1/realtime/media-stream` only, and its session config applies to each session it opens. Browser pages on other origins still need the server's CORS and origin settings.

```bash
curl -s http://127.0.0.1:8080/v1/realtime/client_secrets \
  -H "Authorization: Bearer $IZWI_API_KEY" -H "Content-Type: application/json" \
  -d '{"session": {"type": "realtime", "model": "izwi-voice", "instructions": "Be brief."}}'
```

### Telephony media streams

`GET /v1/realtime/media-stream` takes a phone call in the WebSocket framing call-centre platforms and carrier APIs stream call audio in, and runs it as a `/v1/realtime` session: G.711 μ-law at 8 kHz in both directions, turns cut by `server_vad`. One socket carries one call and takes one realtime session slot. [Phone calls and call centres](/features/telephony) has worked examples and the other ways to connect a call.

The call is configured in the query string, or by the same names in the `start` frame's `customParameters` (which win; some platforms drop the query string):

| Parameter | Meaning |
| --- | --- |
| `model` | The realtime profile, as on `/v1/realtime`. |
| `intent` | `transcription` opens a transcription call: transcripts, no replies. |
| `voice`, `instructions` | The session's `audio.output.voice` and `instructions`. |
| `transcription_model`, `language` | The session's `audio.input.transcription` model and language. A conversation sends transcripts of the caller only when one of them is set, or when a client secret's session config sets `audio.input.transcription`. |
| `transcripts` | `false` keeps `transcript` frames off the stream. |

Authentication is the same as `/v1/realtime`: the API key as a bearer token, or a client secret as a bearer token or the `openai-insecure-api-key.<key>` subprotocol. A client secret's session config applies under the call's own parameters; its audio formats are replaced by G.711 μ-law, and a `turn_detection` of `null` by `server_vad`. A bad `model` or `intent` in the query string refuses the upgrade with `400`.

Frames are JSON text, one event each. From the carrier:

| `event` | Handling |
| --- | --- |
| `connected` | Ignored. |
| `start` | Starts the call. `streamSid` (top level or in `start`) names the stream and is echoed on every frame Izwi sends. `start.mediaFormat`, when present, must be `audio/x-mulaw`, 8000 Hz, 1 channel. `start.customParameters` configure the call. A `start` must arrive within 10 seconds. |
| `media` | `media.payload` is base64 μ-law, any frame length (20 ms is usual). Frames whose `media.track` is not `inbound` are ignored. |
| `mark`, `dtmf` | Ignored. |
| `stop` | Ends the call; Izwi closes the socket. Closing the socket ends the call too. |

Any other event, and any frame that is not JSON, is ignored. From Izwi:

| `event` | Meaning |
| --- | --- |
| `media` | Speech of a reply: `media.payload` is base64 μ-law at 8 kHz, up to 100 ms a frame, sent as it is generated. |
| `mark` | The end of a reply's speech; `mark.name` is the reply's item id. A carrier echoes it when playback reaches it. |
| `clear` | The caller started speaking while a reply was still playing: drop the audio not yet played. Izwi cancels the reply and keeps, as what the agent said, the share of it that had time to play. |
| `transcript` | Izwi extension. `item_id`, `type` (`delta` or `completed`) and `text`: the caller's words. A model that transcribes live sends `delta` frames while the caller speaks; every model sends `completed` with the whole utterance. |
| `error` | Izwi extension. `error` is the session's error object (`type`, `code`, `message`): a failed reply or transcription, or a bad frame. The call stays open. |

A call that cannot start gets an `error` frame and a close: code `1008` for a `start` that is late, has no `streamSid`, names another media format or carries bad parameters, and `1013` when every realtime session slot is in use.

```json
{"event": "start", "streamSid": "MZ123", "start": {"streamSid": "MZ123", "callSid": "CA456", "tracks": ["inbound"],
  "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
  "customParameters": {"model": "izwi-voice", "voice": "cedar"}}}
{"event": "media", "streamSid": "MZ123", "media": {"track": "inbound", "chunk": "1", "timestamp": "0", "payload": "//////..."}}
```

Function tools, manual turns and the rest of the Realtime protocol are not reachable through this framing; use `/v1/realtime` for them.

### Through the gateway

With `IZWI_GATEWAY_OPENAI_REALTIME_UPSTREAMS` (izwi-server base URLs) the gateway serves `/v1/realtime` too: it authenticates and admits each session, then relays it to an upstream izwi-server (round robin, `IZWI_GATEWAY_OPENAI_REALTIME_UPSTREAM_API_KEY` as the upstream's key, capped by `IZWI_GATEWAY_REALTIME_MAX_SESSIONS`). The session itself runs on the izwi-server. `/v1/realtime/media-stream` is relayed the same way, under the same session cap.

The gateway relays `POST /v1/realtime/client_secrets` and `GET /v1/realtime/capabilities` to the same upstreams, after its own authentication, admission and tenant lease. An izwi-server keeps the `ek_` keys it mints in memory, so the gateway remembers which upstream minted each key and for which principal. A browser presents the key on the gateway's `/v1/realtime` (or a carrier on its `/v1/realtime/media-stream`) as it would to an izwi-server (the `openai-insecure-api-key.<key>` subprotocol, no gateway API key): the gateway treats the session as the minting principal's, sends it to the upstream that minted the key, and that upstream checks the key and applies its session config. The key opens no other gateway route. The gateway's memory of keys is per process and is lost on restart, so with several gateway replicas a key works only through the replica that minted it: run one replica for this, or route a browser's requests to the replica its backend used. `capabilities` is answered by one upstream, so configure the upstreams' profiles alike.
