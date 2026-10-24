---
title: "Models"
description: "Understand the Izwi model catalog, download models, manage local storage, and review quantization notes."
icon: "database"
---
## Current Model Catalog

Use `izwi list` (or `GET /v1/models`) to see the live, currently enabled catalog.
Those endpoints only show variants that are enabled for download/use.

> Izwi accepts many legacy aliases (for example lowercase IDs), but the canonical IDs below match `izwi list` output.

### Text-to-Speech (TTS)

| Family | Canonical IDs |
|--------|---------------|
| Qwen3 Base (reference-voice cloning) | `Qwen3-TTS-12Hz-0.6B-Base`, `Qwen3-TTS-12Hz-1.7B-Base` |
| Qwen3 CustomVoice (built-in speakers) | `Qwen3-TTS-12Hz-0.6B-CustomVoice`, `Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| Qwen3 VoiceDesign | `Qwen3-TTS-12Hz-1.7B-VoiceDesign` |
| Voxtral TTS | `Voxtral-4B-TTS-2603` |
| VibeVoice TTS | `VibeVoice-1.5B` |
| Fish Audio S2 Pro (reference voice only) | `FishAudio-S2-Pro` |
| Kokoro | `Kokoro-82M` |
| LFM2.5 Audio (built-in speakers) | `LFM2.5-Audio-1.5B-GGUF` |

> `Kokoro-82M` requires `espeak-ng`:
> [macOS](/installation/macos#optional-install-espeak-ng-for-kokoro-82m),
> [Linux](/installation/linux#optional-install-espeak-ng-for-kokoro-82m),
> [Windows](/installation/windows#optional-install-espeak-ng-for-kokoro-82m)

> `Voxtral-4B-TTS-2603` includes bundled voice assets licensed under CC BY-NC
> 4.0 and supports 20 preset voices with 24 kHz output.

> `VibeVoice-1.5B` is a Microsoft long-form TTS model with reference-voice
> cloning. It uses saved or direct reference voices rather than built-in speaker
> presets.

For built-in speaker IDs, see [Voice Presets](/models/voice-presets).

### Speech Recognition (ASR)

| Model | Notes |
|-------|-------|
| `Parakeet-TDT-0.6B-v3` | CLI default for transcription/diarization ASR |
| `Whisper-Large-v3-Turbo` | Whisper ASR option |
| `Qwen3-ASR-0.6B-GGUF` | Smaller Qwen3 ASR |
| `Qwen3-ASR-1.7B-GGUF` | Higher-accuracy Qwen3 ASR |
| `VibeVoice-ASR` | Microsoft long-form ASR checkpoint |
| `Nemotron-3.5-ASR-Streaming-0.6B` | NVIDIA multilingual FastConformer-RNNT `.nemo`; native artifact/config/tokenizer and streaming-state support |
| `Granite-Speech-4.1-2B` | IBM Granite Speech ASR with punctuated, capitalized transcripts (en, fr, de, es, pt, ja); the default Granite Speech model |
| `Granite-Speech-4.1-2B-Plus` | IBM Granite Speech rich transcription model with speaker-attributed output and native word timestamps; unpunctuated by design (en, fr, de, es, pt) |
| `Granite-Speech-5.0-2B` | IBM Granite Speech 5.0: punctuated transcripts (en, fr, de, es, pt, ja), speech translation to and from English (plus English to Italian and Mandarin), and keyword lists of up to 2,000 terms; the recommended Granite Speech model |
| `Granite-Speech-5.0-2B-GGUF` | IBM's GGUF release of Granite Speech 5.0: the same transcription, translation and keywords with a Q8_0 language model, in less memory and faster (several times faster on CPU) |
| `LFM2.5-Audio-1.5B-GGUF` | Unified audio model (ASR + speech generation) |
| `Voxtral-Mini-4B-Realtime-2602` | Mistral Voxtral offline transcription; realtime support planned |

### Diarization and Alignment

| Task | Model |
|------|-------|
| Speaker diarization | `diar_streaming_sortformer_4spk-v2.1`, `Nemotron-3-Diarization` (8 speakers) |
| Forced alignment | `Qwen3-ForcedAligner-0.6B` |

### Chat

| Family | Canonical IDs |
|--------|---------------|
| Qwen3.5 GGUF | `Qwen3.5-0.8B`, `Qwen3.5-2B`, `Qwen3.5-4B`, `Qwen3.5-9B` |
| Qwen3.6 FP8 | `Qwen3.6-35B-A3B-FP8` (text chat) |
| Qwen3.8 FP8 | `Qwen3.8-27B-FP8` (text chat; 28.74 GiB checkpoint payload) |
| LFM2.5 text | `LFM2.5-1.2B-Instruct-GGUF`, `LFM2.5-1.2B-Thinking-GGUF` |

`Qwen3.8-27B-FP8` uses the separate `Qwen38Chat` architecture/model family.
Qwen3.5 compatibility or backend evidence is not applied to it implicitly.
The listed 28.74 GiB is checkpoint payload size, not an expanded runtime-memory
estimate. On CUDA, projections stay resident as
[native block FP8](/support-matrix#qwen38-cuda-weight-residency) by default, with a
Q8_0 fallback.

### Currently Disabled (Not Listed by `izwi list`)

These variants exist in the catalog but are not currently enabled for standard listing/download:

- Legacy Qwen3 chat IDs: `Qwen3-0.6B`, `Qwen3-0.6B-4bit`, `Qwen3-1.7B`, `Qwen3-1.7B-4bit`
- `Qwen3-14B-GGUF`
- `Gemma-3-4b-it`
- TTS 8-bit and BF16 metadata variants such as `Qwen3-TTS-12Hz-0.6B-Base-8bit`
  and `Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16`.

### Removed variants

The five mlx 4-bit Qwen3-TTS conversions (`Qwen3-TTS-12Hz-0.6B-Base-4bit`,
`Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit`, `Qwen3-TTS-12Hz-1.7B-Base-4bit`,
`Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit`, `Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit`)
are no longer supported: they never reach the end of speech and produce
unintelligible audio. Use the dense variant of the same name without `-4bit`.

The mlx 4-bit forced aligner (`Qwen3-ForcedAligner-0.6B-4bit`) is no longer
supported: it was dequantized at load, so it saved download size but no
memory. Use `Qwen3-ForcedAligner-0.6B`.

- Requests and `izwi pull` naming a removed ID return HTTP 400 naming its
  replacement. The replacement is not substituted automatically: the Qwen3-TTS
  replacements need more memory, and the aligner's is a larger download.
- Studio projects, queued jobs, transcription and diarization reruns, and the
  remembered model selection that name a removed ID use its replacement.
- A removed variant already downloaded stays on disk until you delete it. The
  admin model list reports it under `removed_models` with its size, and
  `DELETE /v1/admin/models/<id>` removes it.

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

# Download NVIDIA Nemotron 3.5 ASR
izwi pull Nemotron-3.5-ASR-Streaming-0.6B

# Download IBM Granite Speech (punctuated ASR, and rich ASR with speaker turns)
izwi pull Granite-Speech-4.1-2B
izwi pull Granite-Speech-4.1-2B-Plus
# Granite Speech 5.0 (transcription, translation and keyword biasing)
izwi pull Granite-Speech-5.0-2B
# or its GGUF release, with a Q8_0 language model
izwi pull Granite-Speech-5.0-2B-GGUF

# Download Microsoft VibeVoice models
izwi pull VibeVoice-1.5B
izwi pull VibeVoice-ASR
```

### Via Web UI

1. Open `http://localhost:8080`
2. Go to **Models** in the sidebar
3. Click **Download** on a model

---

## Managing Models

For the complete UI, CLI, and API workflow, see
[Model Management](/models/management).

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
| **Windows** | `%LOCALAPPDATA%\izwi\models\` |

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

Some models may require manual Hugging Face access setup (for example gated
repositories that need license acceptance):

- [Manual Download Guide](/models/manual-download)

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
- `izwi list` shows enabled variants only. Some catalog metadata exists for
  experimental 8-bit/BF16 TTS artifacts; they are not downloadable.

---

## Next Steps

- [Manual Model Downloads](/models/manual-download)
- [Model Management](/models/management)
- [Voice Presets](/models/voice-presets)
- [CLI Reference](/cli)
- [Troubleshooting](/troubleshooting)
