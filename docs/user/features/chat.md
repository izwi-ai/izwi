---
title: "Chat"
description: "Run local text and multimodal chat conversations through the Izwi CLI, web UI, and API."
icon: "message-square"
---
# Chat

Have local conversations with chat models running on your own machine.

---

## Overview

Izwi chat provides:

- **Local inference** — Model execution stays on-device
- **Multiple model families** — Qwen3, Qwen3.5, LFM2.5, and Gemma
- **System prompts** — Shape assistant behavior
- **Streaming output** — Incremental response tokens
- **Multimodal support (Qwen3.5 only)** — Image inputs in chat API requests

---

## Getting Started

### Download a Chat Model

```bash
izwi pull Qwen3-8B-GGUF
```

### Start Chatting

```bash
izwi chat --model Qwen3-8B-GGUF
```

Web UI:

```
http://localhost:8080/chat
```

---

## Using the CLI

| Option | Description | Default |
|--------|-------------|---------|
| `--model`, `-m` | Chat model to use | `qwen3-0.6b-4bit` |
| `--system`, `-s` | System prompt | — |
| `--voice`, `-v` | Voice for spoken responses | — |

`qwen3-0.6b-4bit` remains the CLI default for backward compatibility.
For new setups, prefer an enabled model from `izwi list`, such as `Qwen3-8B-GGUF` or `Qwen3.5-4B`.

Examples:

```bash
izwi chat --system "You are a helpful coding assistant."
izwi chat --model Qwen3-8B-GGUF
izwi chat --model Qwen3.5-4B
izwi chat --model LFM2.5-1.2B-Instruct-GGUF
izwi chat --model Gemma-3-1b-it
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
  "model": "Qwen3-8B-GGUF",
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
    "model": "Qwen3-8B-GGUF",
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
[API Reference](../api.md#chat-completions) for the full request contract and
streaming sequence.

---

## Supported Chat Models

| Family | Models |
|--------|--------|
| Qwen3 | `Qwen3-0.6B-GGUF`, `Qwen3-1.7B-GGUF`, `Qwen3-4B-GGUF`, `Qwen3-8B-GGUF` |
| Qwen3.5 | `Qwen3.5-0.8B`, `Qwen3.5-2B`, `Qwen3.5-4B`, `Qwen3.5-9B` |
| LFM2.5 | `LFM2.5-1.2B-Instruct-GGUF`, `LFM2.5-1.2B-Thinking-GGUF` |
| Gemma | `Gemma-3-1b-it` |

---

## Multimodal Limits

- Multimodal media chat is currently limited to **Qwen3.5 GGUF** models.
- **Video inputs are not yet implemented**.
- Non-Qwen3.5 chat variants currently support text-only requests.

---

## Tips

1. Use `izwi list` to pick a currently enabled model ID.
2. Use stronger models (`Qwen3-8B-GGUF`, `Qwen3.5-9B`) for harder tasks.
3. Use smaller models (`Qwen3.5-0.8B`, `LFM2.5-1.2B-*`) for low-latency usage.

---

## See Also

- [Voice Mode](./voice.md)
- [Models](../models/index.md)
- [CLI Reference](../cli/index.md)
