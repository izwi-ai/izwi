---
title: "Chat"
description: "Run local text and multimodal chat conversations through the Izwi CLI, web UI, and API."
icon: "message-square"
---
Have local conversations with chat models running on your own machine.

---

## Overview

Izwi chat provides:

- **Local inference** — Model execution stays on-device
- **Multiple model families** — Qwen3.5, Qwen3.6, Qwen3.8, and LFM2.5
- **System prompts** — Shape assistant behavior
- **Streaming output** — Incremental response tokens
- **Multimodal support (Qwen3.5 only)** — Image inputs in chat API requests

---

## Getting Started

### Download a Chat Model

```bash
izwi pull Qwen3.5-4B
```

### Start Chatting

```bash
izwi chat --model Qwen3.5-4B
```

Web UI:

```
http://localhost:8080/chat
```

---

## Using the CLI

| Option | Description | Default |
|--------|-------------|---------|
| `--model`, `-m` | Chat model to use | The chat model the server has loaded, else `Qwen3.5-0.8B` |
| `--system`, `-s` | System prompt | — |
| `--thinking` | Let the model reason before it answers (models that can think); the reasoning is shown dimmed and kept out of the history | The model's own setting |
| `--no-thinking` | Answer without reasoning first | The model's own setting |

Without `--model`, `izwi chat` uses the chat model the server has loaded, or `Qwen3.5-0.8B` when none is loaded (pull it first with `izwi pull Qwen3.5-0.8B`).
Pick any enabled model from `izwi list`, such as `Qwen3.5-4B` or `Qwen3.5-9B`.

Examples:

```bash
izwi chat --system "You are a helpful coding assistant."
izwi chat --model Qwen3.5-4B
izwi chat --model Qwen3.5-9B
izwi chat --model LFM2.5-1.2B-Instruct-GGUF
```

---

## Using the Web UI

1. Open **Chat** in the sidebar
2. Enter a prompt
3. Send and review streamed output
4. Switch loaded models from the model selector

---

## Using the API

### Text Chat Endpoint

```
POST /v1/chat/completions
```

### Text Request Example

```json
{
  "model": "Qwen3.5-4B",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize this project in three bullets."}
  ],
  "stream": true
}
```

### cURL Example

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-4B",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Multimodal (Image) Example

Image inputs are supported only on Qwen3.5 GGUF chat variants:

```json
{
  "model": "Qwen3.5-4B",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "input_text", "text": "What is in this image?"},
        {"type": "input_image", "image_url": {"url": "https://example.com/cat.png"}}
      ]
    }
  ]
}
```

The API also supports SSE streaming, `stream_options.include_usage`, tool-call
payloads, and strict/relaxed OpenAI compatibility profiles. See the
[API Reference](/api#chat-completions) for the full request contract and
streaming sequence.

---

## Supported Chat Models

| Family | Models |
|--------|--------|
| Qwen3.5 | `Qwen3.5-0.8B`, `Qwen3.5-2B`, `Qwen3.5-4B`, `Qwen3.5-9B` |
| Qwen3.6 FP8 | `Qwen3.6-35B-A3B-FP8` (text only) |
| Qwen3.8 FP8 | `Qwen3.8-27B-FP8` (text only) |
| LFM2.5 | `LFM2.5-1.2B-Instruct-GGUF`, `LFM2.5-1.2B-Thinking-GGUF` |

Qwen3.8 is registered as its own `Qwen38Chat` runtime family. It does not
inherit Qwen3.5 backend certification, state-conformance evidence, or media
capabilities. On CUDA its projections run as
[native block FP8](/support-matrix#qwen38-cuda-weight-residency) by default, with a
scale-aware Q8_0 fallback.

---

## Multimodal Limits

- Multimodal media chat is currently limited to **Qwen3.5 GGUF** models.
- **Video inputs are not yet implemented**.
- Non-Qwen3.5 chat variants, including `Qwen3.8-27B-FP8`, currently support text-only requests.

---

## Tips

1. Use `izwi list` to pick a currently enabled model ID.
2. Use stronger models (`Qwen3.5-9B`, `Qwen3.8-27B-FP8`) for harder tasks when the host has sufficient memory; check the [Qwen3.8 CUDA residency requirements](/support-matrix#qwen38-cuda-weight-residency) before deploying the 27B checkpoint.
3. Use smaller models (`Qwen3.5-0.8B`, `LFM2.5-1.2B-*`) for low-latency usage.

---

## See Also

- [Voice Mode](/features/voice)
- [Models](/models)
- [CLI Reference](/cli)

## CUDA concurrent chat

Qwen3.8 batches independent CUDA chat requests through one loaded model by default.
The incremental admission policy grows cache reservations with active sequences
and can suspend and replay a request when the shared cache is under pressure. It preserves automatic
output limits and already streamed text. Capacity is derived from the device's
memory and model state, rather than a GPU model name.

Use your normal CUDA server command; no environment flag is needed. The policy
also enables scheduler-visible chunked prefill for resumable adapters. To opt out
and return to conservative admission, set this before restarting the server:

```bash
export IZWI_CUDA_INCREMENTAL_CHAT=0
```

Set it to `1` or remove the variable to restore the default.
Other model families keep their existing admission policy until they implement the
published-sequence replay contract.

Hybrid models (Qwen3.8 and Qwen3.6) keep a fixed block of recurrent state for every
conversation they hold at once (about 157 MB per conversation for Qwen3.8 and 66 MB for
Qwen3.6). When `IZWI_MAX_RETAINED_SEQUENCES` is unset, the server sizes this
automatically at load: recurrent state may use up to a quarter of the memory left after
the weights, and the rest holds the KV cache. It never keeps fewer conversations than the
decode width (8 on CUDA). The chat-history snapshots below come out of the same quarter.
On an 80 GB H100 that is about 68 conversations plus 8 history snapshots for Qwen3.8,
and about 160 plus 8 for Qwen3.6. Further requests wait for a slot and start as soon as
one finishes.
The load log line `Fitted CUDA resident state rows and context` and the
`retained_sequence_capacity` field under `kv_cache.models[]` in `/v1/metrics` show the
chosen size. To give recurrent state a different share, set a value between 0 and 1:

```bash
export IZWI_CUDA_STATE_BUDGET_FRACTION=0.4
```

Setting `IZWI_MAX_RETAINED_SEQUENCES` explicitly turns the automatic sizing off. The
server then reserves exactly that many conversations, plus `IZWI_MAX_STAGED_TRANSACTIONS`
in-flight copies (by default the decode width), and refuses the load with the byte
figures if they do not fit.

### Chat history reuse

Each chat turn re-renders the whole conversation, but Qwen3.8 on CUDA and Qwen3.6 on
CPU and CUDA do not recompute the history they already processed (Qwen3.8 on CPU and
Metal too, once its MTP draft head is off with `IZWI_QWEN38_MTP=0`). A
turn's prompt is split before its final assistant header (earlier answers are
re-rendered, so only the history up to there repeats in the next turn); the server
keeps the KV pages of that history and a snapshot of the recurrent state at its last
full 64-token page. The next turn of the conversation attaches there and prefills only
the previous answer and the new message, which shortens time to first token on long
conversations. A history shorter than one page has no snapshot, so after a very short
first prompt the third turn is the first to reuse.

Snapshots are bounded: `IZWI_MAX_PREFIX_SNAPSHOTS` (default 8) snapshots per model,
about 157 MB each for Qwen3.8, reserved at load. The least recently used snapshot is
dropped first, so up to 8 conversations continue with their history reused at once;
others prefill in full. History pages share the KV cache and are released when a
request needs the room. A conversation's whole history must fit the prefix page
budget (`IZWI_MAX_PREFIX_CACHE_PAGES`); longer conversations prefill in full. With
thinking on, answers often run to several thousand tokens, so on CUDA the budget
defaults to a quarter of the model's KV arena, which is allocated at load either way:
about 2,400 pages of 64 tokens (over 150,000 tokens of history) for Qwen3.8 on an
80 GB H100. On CPU and Metal the default is 128 pages (8,192 tokens). The model's
`kv_cache.models[].arenas[].prefix_capacity_pages` in `/v1/metrics` shows the
effective budget. A value you set applies as set, for example:

```bash
export IZWI_MAX_PREFIX_CACHE_PAGES=4096
```

To see it working, compare `usage.prompt_tokens_details.cached_tokens` on a second
turn (the reused tokens) with `prompt_tokens`, or read `prefix_reuse` in the model's
runtime diagnostics (`attached_prefills`, `reused_prompt_tokens`) and
`kv_cache.counters.tensor_snapshot_attaches` in `/v1/metrics`. To turn reuse off
for Qwen3.8 only:

```bash
export IZWI_QWEN38_PREFIX_REUSE=0
```

`IZWI_ENABLE_PREFIX_CACHING=0` turns it off for every model. A Qwen3.8 load with its MTP
draft head (`IZWI_QWEN38_MTP=1` on CUDA; the default on CPU and Metal) keeps its
history private and prefills every turn in full: the draft head's attention state has
no snapshot to attach.

Use separate conversations, or independent `/v1/chat/completions` requests, for
concurrent answers. Sends to the same conversation remain ordered. The playground
allows one active stream per mounted view; separate tabs can use separate conversations.

Inspect `/v1/health` for the effective `runtime.chat_concurrency_policy`, and
`/v1/metrics` for actual model batch widths, cache claims, suspensions and replay
work. Open HTTP streams alone do not establish concurrent model execution. Heavy
cache pressure can still queue or suspend requests; simultaneous maximum-length
histories must fit the available state budget.

Temporary batch workspace pressure is also handled before model execution: the
scheduler reduces batch width or resumable prefill size and retries the same
request without repeating streamed output. Under persistent pressure it can
suspend eligible Qwen3.8 requests. Progress from competing work keeps waiting
requests eligible; a request that is refused for 5 seconds with no progress
anywhere returns an explicit capacity error. Requests that cannot fit alone
cannot be guaranteed to finish. Capacity diagnostics group pending reservations
by owner class to distinguish model, request, and workspace promises from
materialized memory.

When it sizes the KV cache at load, the CUDA server keeps part of the GPU free
for memory that CUDA allocates later without a reservation: kernels and library
workspaces loaded on first use, CUDA graphs, and per-step scratch. This reserve
is the larger of 1 GiB and 3% of GPU memory (about 2.55 GB on an 80 GB H100). The
load log line `Fitted CUDA resident state rows and context` reports it as
`context_reserve_bytes`. `coordinator.untracked_device_memory_bytes` in
`/v1/metrics` shows how much memory that reserve is covering. Compare it right
after loading with its value after a few requests: if it grows by more than the
reserve, set a larger reserve in bytes before restarting the server:

```bash
export IZWI_CUDA_CONTEXT_RESERVE_BYTES=4294967296
```

A larger reserve leaves less memory for the KV cache.

For a controlled hardware acceptance run, use
`scripts/bench/run-cuda-chat-concurrency.py` as documented in
`scripts/bench/README.md`. It checks uncapped requests through both API routes,
staggered arrivals, cancellation, overlapping output, actual multirow forwards,
and resource release. CPU tests do not certify CUDA throughput or numerical behavior.
