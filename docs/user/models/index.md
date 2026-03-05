# Models

Izwi uses AI models for text-to-speech, speech recognition, and chat. This guide explains how to find, download, and manage models.

---

## Available Models

Izwi supports several model families optimized for different tasks:

### Text-to-Speech (TTS)

| Model | Size | Description |
|-------|------|-------------|
| `Kokoro-82M` | ~0.4 GB | Lightweight TTS (requires `espeak-ng`) |
| `Qwen3-TTS-12Hz-0.6B-Base` | ~2.3 GB | Fast, general-purpose TTS |
| `Qwen3-TTS-12Hz-0.6B-CustomVoice` | ~2.3 GB | TTS with voice cloning support |
| `Qwen3-TTS-12Hz-1.7B-Base` | ~4.2 GB | Higher quality TTS |
| `Qwen3-TTS-12Hz-1.7B-CustomVoice` | ~4.2 GB | Higher quality with voice cloning |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | ~4.2 GB | Voice design from descriptions |
| `LFM2.5-Audio-1.5B` | ~3.0 GB | Liquid AI audio model (v2.5) |

> `Kokoro-82M` uses `espeak-ng` for phonemization. Install it separately before using Kokoro:
> [macOS](../installation/macos.md#optional-install-espeak-ng-for-kokoro-82m),
> [Linux](../installation/linux.md#optional-install-espeak-ng-for-kokoro-82m),
> [Windows](../installation/windows.md#optional-install-espeak-ng-for-kokoro-82m)

### Speech Recognition (ASR)

| Model | Size | Description |
|-------|------|-------------|
| `Qwen3-ASR-0.6B` | ~1.8 GB | Fast speech-to-text |
| `Qwen3-ASR-1.7B` | ~4.4 GB | Higher accuracy transcription |
| `Parakeet-TDT-0.6B-v2` | ~4.6 GB | NVIDIA Parakeet ASR |
| `Parakeet-TDT-0.6B-v3` | ~9.4 GB | NVIDIA Parakeet ASR (latest) |

### Speaker Diarization

| Model | Size | Description |
|-------|------|-------------|
| `diar_streaming_sortformer_4spk-v2.1` | ~0.5 GB | NVIDIA Sortformer, up to 4 speakers |

### Chat

| Model | Size | Description |
|-------|------|-------------|
| `Qwen3-0.6B` | ~1.4 GB | Compact Qwen3 chat model (full precision) |
| `Qwen3-0.6B-4bit` | ~0.8 GB | Compact Qwen3 chat model |
| `Qwen3-1.7B` | ~3.8 GB | Larger Qwen3 chat model |
| `Gemma-3-1b-it` | ~2.0 GB | Google Gemma 3 1B Instruct |
| `Gemma-3-4b-it` | ~8.0 GB | Google Gemma 3 4B Instruct |

### Specialized

| Model | Size | Description |
|-------|------|-------------|
| `Qwen3-ForcedAligner-0.6B` | ~1.7 GB | Word-level audio alignment |
| `Voxtral-Mini-4B-Realtime-2602` | ~7.5 GB | Mistral realtime audio (coming soon) |

---

## Downloading Models

### Via CLI

```bash
# List all available models
izwi list

# Download a model
izwi pull qwen3-tts-0.6b-base

# Download with progress
izwi pull qwen3-asr-0.6b
```

### Via Web UI

1. Open `http://localhost:8080`
2. Go to **Models** in the sidebar
3. Click **Download** on any model

---

## Managing Models

### View Downloaded Models

```bash
izwi list --local
```

### Get Model Information

```bash
izwi models info qwen3-tts-0.6b-base
```

### Load a Model into Memory

```bash
izwi models load qwen3-tts-0.6b-base
```

### Unload a Model

```bash
izwi models unload qwen3-tts-0.6b-base
```

### Delete a Model

```bash
izwi rm qwen3-tts-0.6b-base
```

---

## Model Storage

Models are stored in your system's application data directory:

| Platform | Location |
|----------|----------|
| **macOS** | `~/Library/Application Support/izwi/models/` |
| **Linux** | `~/.local/share/izwi/models/` |
| **Windows** | `%APPDATA%\izwi\models\` |

### Custom Model Directory

Set a custom location:

```bash
# Via CLI flag
izwi serve --models-dir /path/to/models

# Via environment variable
export IZWI_MODELS_DIR=/path/to/models
izwi serve
```

---

## Manual Downloads

Some models require manual download from Hugging Face due to licensing:

- [Manual Download: Gemma 3 1B](./manual-gemma-3-1b-download.md)
- [Manual Download Guide](./manual-download.md)

---

## Model Status

Models can be in several states:

| Status | Description |
|--------|-------------|
| **not_downloaded** | Model available but not on disk |
| **downloading** | Currently downloading |
| **downloaded** | On disk but not loaded |
| **loading** | Being loaded into memory |
| **ready** | Loaded and ready for inference |

Check status:

```bash
izwi status --detailed
```

---

## Quantized Models

Some models offer quantized variants for reduced memory usage:

- **4-bit** — Smallest size, some quality loss
- **8-bit** — Balanced size and quality
- **Full** — Original quality, largest size

Quantized models have suffixes like `-4bit` or `-q4`.

---

## Next Steps

- [Manual Model Downloads](./manual-download.md)
- [CLI Reference](../cli/index.md)
- [Troubleshooting](../troubleshooting.md)
