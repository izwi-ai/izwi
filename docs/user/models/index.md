# Models

Izwi uses local models for text-to-speech, speech recognition, diarization, alignment, and chat.

---

## Current Model Catalog

Use `izwi list` (or `GET /v1/models`) to see the live, currently enabled catalog.
Those endpoints only show variants that are enabled for download/use.

> Izwi accepts many legacy aliases (for example lowercase IDs), but the canonical IDs below match `izwi list` output.

### Text-to-Speech (TTS)

| Family | Canonical IDs |
|--------|---------------|
| Qwen3 Base (reference-voice cloning) | `Qwen3-TTS-12Hz-0.6B-Base`, `Qwen3-TTS-12Hz-0.6B-Base-4bit`, `Qwen3-TTS-12Hz-1.7B-Base`, `Qwen3-TTS-12Hz-1.7B-Base-4bit` |
| Qwen3 CustomVoice (built-in speakers) | `Qwen3-TTS-12Hz-0.6B-CustomVoice`, `Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit`, `Qwen3-TTS-12Hz-1.7B-CustomVoice`, `Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit` |
| Qwen3 VoiceDesign | `Qwen3-TTS-12Hz-1.7B-VoiceDesign`, `Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit` |
| Kokoro | `Kokoro-82M` |

> `Kokoro-82M` requires `espeak-ng`:
> [macOS](../installation/macos.md#optional-install-espeak-ng-for-kokoro-82m),
> [Linux](../installation/linux.md#optional-install-espeak-ng-for-kokoro-82m),
> [Windows](../installation/windows.md#optional-install-espeak-ng-for-kokoro-82m)

### Speech Recognition (ASR)

| Model | Notes |
|-------|-------|
| `Parakeet-TDT-0.6B-v3` | CLI default for transcription/diarization ASR |
| `Whisper-Large-v3-Turbo` | Whisper ASR option |
| `Qwen3-ASR-0.6B-GGUF` | Smaller Qwen3 ASR |
| `Qwen3-ASR-1.7B-GGUF` | Higher-accuracy Qwen3 ASR |
| `LFM2.5-Audio-1.5B-GGUF` | Unified audio model (ASR + speech generation) |
| `Voxtral-Mini-4B-Realtime-2602` | Mistral Voxtral offline transcription; realtime support planned |

### Diarization and Alignment

| Task | Model |
|------|-------|
| Speaker diarization | `diar_streaming_sortformer_4spk-v2.1` |
| Forced alignment | `Qwen3-ForcedAligner-0.6B`, `Qwen3-ForcedAligner-0.6B-4bit` |

### Chat

| Family | Canonical IDs |
|--------|---------------|
| Qwen3 GGUF | `Qwen3-0.6B-GGUF`, `Qwen3-1.7B-GGUF`, `Qwen3-4B-GGUF`, `Qwen3-8B-GGUF` |
| Qwen3.5 GGUF | `Qwen3.5-0.8B`, `Qwen3.5-2B`, `Qwen3.5-4B`, `Qwen3.5-9B` |
| LFM2.5 text | `LFM2.5-1.2B-Instruct-GGUF`, `LFM2.5-1.2B-Thinking-GGUF` |
| Gemma | `Gemma-3-1b-it` |

### Currently Disabled (Not Listed by `izwi list`)

These variants exist in the catalog but are not currently enabled for standard listing/download:

- Legacy Qwen3 chat IDs: `Qwen3-0.6B`, `Qwen3-0.6B-4bit`, `Qwen3-1.7B`, `Qwen3-1.7B-4bit`
- `Qwen3-14B-GGUF`
- `Gemma-3-4b-it`

---

## Downloading Models

### Via CLI

```bash
# List enabled catalog models
izwi list

# Download a model
izwi pull Qwen3-TTS-12Hz-0.6B-Base

# Download an ASR model
izwi pull Qwen3-ASR-0.6B-GGUF
```

### Via Web UI

1. Open `http://localhost:8080`
2. Go to **Models** in the sidebar
3. Click **Download** on a model

---

## Managing Models

### View Downloaded Models

```bash
izwi list --local
```

### Get Model Information

```bash
izwi models info Qwen3-TTS-12Hz-0.6B-Base
```

### Load a Model into Memory

```bash
izwi models load Qwen3-TTS-12Hz-0.6B-Base
```

### Unload a Model

```bash
izwi models unload Qwen3-TTS-12Hz-0.6B-Base
```

### Delete a Model

```bash
izwi rm Qwen3-TTS-12Hz-0.6B-Base
```

---

## Model Storage

| Platform | Location |
|----------|----------|
| **macOS** | `~/Library/Application Support/izwi/models/` |
| **Linux** | `~/.local/share/izwi/models/` |
| **Windows** | `%APPDATA%\izwi\models\` |

### Custom Model Directory

```bash
# CLI flag
izwi serve --models-dir /path/to/models

# Environment variable
export IZWI_MODELS_DIR=/path/to/models
izwi serve
```

---

## Manual Downloads

Some models (for example Gemma) may require manual Hugging Face access setup:

- [Manual Download: Gemma 3 1B](./manual-gemma-3-1b-download.md)
- [Manual Download Guide](./manual-download.md)

---

## Model Status

| Status | Description |
|--------|-------------|
| **not_downloaded** | Available but not on disk |
| **downloading** | Currently downloading |
| **downloaded** | On disk but not loaded |
| **loading** | Being loaded into memory |
| **ready** | Loaded and ready for inference |

Check status:

```bash
izwi status --detailed
```

---

## Quantization Notes

- `-4bit` / `-8bit` / `-bf16` are reduced-precision variants.
- `-GGUF` variants are quantized GGUF artifacts.
- Smaller/quantized variants reduce memory and disk use at some quality/accuracy tradeoff.

---

## Next Steps

- [Manual Model Downloads](./manual-download.md)
- [CLI Reference](../cli/index.md)
- [Troubleshooting](../troubleshooting.md)
