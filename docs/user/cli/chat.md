---
title: "izwi chat"
description: "Start an interactive local chat session from the command line."
icon: "message-square"
---
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
| `-m, --model <MODEL>` | Chat model to use | The chat model the server has loaded, else `Qwen3.5-0.8B` |
| `-s, --system <PROMPT>` | System prompt | — |
| `--thinking` | Let the model reason before it answers (models that can think); the reasoning is shown dimmed and kept out of the history | The model's own setting |
| `--no-thinking` | Answer without reasoning first | — |

When no chat model is loaded, the CLI uses `Qwen3.5-0.8B`; pull it first with `izwi pull Qwen3.5-0.8B`. There is no default system prompt; set one with `--system`.
Use an enabled catalog ID from `izwi list` (for example `Qwen3.5-4B`, `Qwen3.5-9B`, or `Qwen3.8-27B-FP8`). Model IDs are accepted as free-form strings, so no CLI upgrade-specific enum is required when a new server catalog model is added.
The CLI forwards `Qwen3.8-27B-FP8` unchanged; the server routes it through the
separate `Qwen38Chat` family rather than Qwen3.5.

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
izwi chat --model Qwen3.5-4B
izwi chat --model Qwen3.5-9B
izwi chat --model Qwen3.8-27B-FP8
izwi chat --model LFM2.5-1.2B-Instruct-GGUF
```

### Without reasoning

```bash
izwi chat --model Qwen3.5-4B --no-thinking
```

---

## Interactive Commands

During a chat session:

| Command | Action |
|---------|--------|
| Type message + Enter | Send message |
| `/quit` (or `/exit`, `/q`) | End session |
| `/clear` | Clear conversation |
| `/help` | Show the commands |
| `Ctrl+D` | End session |
| `Ctrl+C` | Exit immediately |

---

## See Also

- [Chat Guide](/features/chat)
- [Voice Mode](/features/voice)
- [Models Guide](/models)
