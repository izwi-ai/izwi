---
title: "izwi chat"
description: "Start an interactive local chat session from the command line."
icon: "message-square"
---
# izwi chat

Interactive chat with AI models.

---

## Synopsis

```bash
izwi chat [OPTIONS]
```

---

## Description

Starts an interactive chat session with a loaded chat model. Type messages and receive AI responses in real-time.

---

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model <MODEL>` | Chat model to use | `qwen3-0.6b-4bit` |
| `-s, --system <PROMPT>` | System prompt | — |
| `-v, --voice <VOICE>` | Voice for spoken responses | — |

`qwen3-0.6b-4bit` remains the CLI default for backward compatibility.
For new setups, use an enabled catalog ID from `izwi list` (for example `Qwen3-8B-GGUF` or `Qwen3.5-4B`).

---

## Examples

### Start chat

```bash
izwi chat
```

### With system prompt

```bash
izwi chat --system "You are a helpful coding assistant"
```

### With specific model

```bash
izwi chat --model Qwen3-8B-GGUF
izwi chat --model Qwen3.5-4B
izwi chat --model LFM2.5-1.2B-Instruct-GGUF
izwi chat --model Gemma-3-1b-it
```

### With voice responses

```bash
izwi chat --voice default
```

---

## Interactive Commands

During a chat session:

| Command | Action |
|---------|--------|
| Type message + Enter | Send message |
| `exit` or `quit` | End session |
| `clear` | Clear conversation |
| `Ctrl+C` | Exit immediately |

---

## See Also

- [Chat Guide](../features/chat.md)
- [Voice Mode](../features/voice.md)
- [Models Guide](../models/index.md)
