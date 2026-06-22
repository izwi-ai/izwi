---
title: "izwi pull"
description: "Download an Izwi model to the local model cache."
icon: "download-cloud"
---
# izwi pull

Download a model from Hugging Face.

---

## Synopsis

```bash
izwi pull <MODEL> [OPTIONS]
```

---

## Description

Downloads a model from the Hugging Face Hub and caches it locally. Supports resuming interrupted downloads.

---

## Arguments

| Argument | Description |
|----------|-------------|
| `<MODEL>` | Model variant to download |

---

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `-f, --force` | Force re-download even if exists | — |
| `-y, --yes` | Download without confirmation | — |

---

## Examples

### Download a model

```bash
izwi pull Qwen3-TTS-12Hz-0.6B-Base
```

### Skip confirmation

```bash
izwi pull Qwen3-TTS-12Hz-0.6B-Base --yes
```

### Force re-download

```bash
izwi pull Qwen3-TTS-12Hz-0.6B-Base --force
```

---

## Available Models

Run `izwi list` to see all available models.

Common models:

| Model | Type | Size |
|-------|------|------|
| `Qwen3-TTS-12Hz-0.6B-Base` | TTS (base) | ~2.3 GB |
| `Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit` | TTS (built-in voices) | ~1.6 GB |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit` | TTS (voice design) | ~2.2 GB |
| `Voxtral-4B-TTS-2603` | TTS (20 preset voices, CC BY-NC 4.0) | ~8.1 GB |
| `Parakeet-TDT-0.6B-v3` | ASR (default) | ~9.4 GB |
| `Qwen3-ASR-0.6B-GGUF` | ASR (compact) | ~1.0 GB |
| `Nemotron-3.5-ASR-Streaming-0.6B` | ASR (NVIDIA multilingual `.nemo`) | ~2.37 GB |
| `Granite-Speech-4.1-2B-Plus` | ASR (IBM rich transcription) | ~4.2 GB |
| `Voxtral-Mini-4B-Realtime-2602` | ASR (offline transcription; realtime planned) | ~8 GB |
| `Qwen3-8B-GGUF` | Chat | ~5.2 GB |
| `Qwen3.5-4B` | Chat | ~3.4 GB |
| `LFM2.5-1.2B-Instruct-GGUF` | Chat | ~0.7 GB |
| `Gemma-3-1b-it` | Chat | ~2.2 GB |

---

## Resume Downloads

If a download is interrupted, run the same command again. The download will resume from where it left off.

---

## See Also

- [`izwi list`](/cli/list) — List models
- [`izwi rm`](/cli/rm) — Remove models
- [Models Guide](/models) — Model documentation
