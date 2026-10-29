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
| Chatterbox (English; reference-voice cloning) | `Chatterbox-500M` |
| Chatterbox Multilingual (22 languages; reference-voice cloning) | `Chatterbox-Multilingual-500M` |
| Maya1 (voice design from a description; emotion tags) | `Maya1-3B` |
| Maya1 GGUF (the same, Q8_0 decoder) | `Maya1-3B-GGUF` |
| Maya 2 Native (English and 11 Indic languages; 20 voices; reference-voice cloning) | `Maya2-Native-2.5B` |
| Kokoro | `Kokoro-82M` |
| LFM2.5 Audio (built-in speakers) | `LFM2.5-Audio-1.5B-GGUF` |
| LFM3 Audio (two built-in voices; 19 locales; also ASR and voice chat) | `LFM3-Audio-2B-GGUF` (Q4_K_M, default), `LFM3-Audio-2B-Q8_0-GGUF` |

> `Kokoro-82M` requires `espeak-ng`:
> [macOS](/installation/macos#optional-install-espeak-ng-for-kokoro-82m),
> [Linux](/installation/linux#optional-install-espeak-ng-for-kokoro-82m),
> [Windows](/installation/windows#optional-install-espeak-ng-for-kokoro-82m)

> `Voxtral-4B-TTS-2603` includes bundled voice assets licensed under CC BY-NC
> 4.0 and supports 20 preset voices with 24 kHz output.

> `VibeVoice-1.5B` is a Microsoft long-form TTS model with reference-voice
> cloning. It uses saved or direct reference voices rather than built-in speaker
> presets.

> `Chatterbox-500M` is Resemble AI's English TTS model (MIT code and weights):
> a 0.5B Llama speech-token model with classifier-free guidance, then S3Gen flow
> matching and a HiFT vocoder, with 24 kHz output. It has one built-in voice
> (`default`) and clones from reference audio alone, without a transcript. Every
> utterance carries Resemble AI's imperceptible PerTh watermark (the catalog
> labels it **Watermarked**), whose checkpoint is downloaded from GitHub rather
> than Hugging Face (see [Manual Downloads](/models/manual-download#chatterbox)).

> `Maya1-3B` is Maya Research's English voice-design model (Apache-2.0): a
> Llama 3.2-3B decoder that writes SNAC audio codes, 24 kHz output. Its voice
> comes from a written description (`instructions`), such as "Realistic female
> voice in the 30s age with a british accent. Low pitch, warm timbre, slow
> pacing." Nine described voices are built in (see
> [Voice Presets](/models/voice-presets#maya1)). Twenty tags placed in the text
> (`<laugh>`, `<sigh>`, `<whisper>`, …) make it laugh, sigh or whisper there.
> At 3B parameters it is slower than real time on Apple Silicon: on a 16 GB
> M1 Pro with Metal one request takes about 4.5 times as long as the speech it
> generates (3 times with `Maya1-3B-GGUF`). It has not yet been measured on
> NVIDIA GPUs. The download (about 6.7 GB) includes the SNAC
> 24 kHz decoder from its own repository. `Maya1-3B-GGUF` is the same model with
> its decoder quantized to Q8_0 (llama.cpp's conversion, published as
> `mradermacher/maya1-GGUF`): about 3.6 GB to download and half the weight
> memory, with the same voices and tags.

> `Maya2-Native-2.5B` is Maya Research's Maya 2 Native (Apache-2.0), a TTS
> model for Assamese, Bengali, English, Gujarati, Hindi, Kannada, Malayalam,
> Marathi, Odia, Punjabi, Tamil and Telugu. A Qwen3-1.7B backbone predicts the
> first codebook of each 80 ms audio frame, a depth decoder fills in the other
> 31, and Kyutai's Mimi codec turns the frames into 24 kHz audio. It speaks
> with one of 20 built-in voices (see
> [Voice Presets](/models/voice-presets#maya-2-native)), or clones a voice from
> reference audio and its exact transcript with SpeechBrain's ECAPA speaker
> encoder; the clone can speak any of the 12 languages, not only the
> reference's. A request names its language in `language` (a code such as
> `hi`) or leaves it to be read from the text's script. Tags in square
> brackets (`[laughs]`, `[sighs]`, `[whispers]`, …) are performed on a
> best-effort basis. Numbers, dates, currency and similar are spoken as words
> in the request's language. The download (about 5.5 GB) includes the Mimi
> codec (CC BY 4.0) and the ECAPA speaker encoder from their own repositories.
> Izwi estimates about 7 GB of memory on Metal and CUDA; on CPU it runs in F32,
> and its weights alone take about 10 GB. On a 16 GB M1 Pro with Metal one
> request takes about as long as the speech it generates (0.85 to 1 times),
> and four at once produce speech about 1.3 times faster than real time
> together; on that machine's CPU a request takes 8 to 9 times as long as its
> speech. It has not yet been measured on NVIDIA GPUs.

> `Chatterbox-Multilingual-500M` is the same design trained on 23 languages,
> with its own speech-token model (`t3_mtl23ls_v2`) and grapheme tokenizer and
> the English model's S3Gen, voice encoder, default voice and watermark. Izwi
> speaks 22 of its languages: Arabic, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Italian, Korean, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swahili, Swedish and Turkish. Chinese goes through upstream's
> word segmenter (spacy-pkuseg) and Cangjie table, downloaded with the model.
> Japanese needs a kanji reader Izwi has not ported and returns `400`. A request names its
> language in `language` (a code such as `fr` or a name such as `French`);
> without one it speaks English. It uses upstream's default (V2) speech-token
> model. In Izwi's spot check (two sentences per language, transcribed by
> Whisper) it was less intelligible in Hebrew, Norwegian and Malay than in the
> other languages. Hebrew is spoken without vowel points and Russian without
> stress marks, as upstream does without its optional add-ons.

> `LFM3-Audio-2B-GGUF` is Liquid AI's LFM3-Audio-2B (LFM Open License v1.0,
> free below US$10 million in annual revenue), one model that speaks,
> transcribes and holds voice conversations. It pairs a Q4_K_M language model
> with a Q8_0 mmproj file that holds the speech encoder, the MaskGIT depth
> decoder and the LiquidCodec decoder; `izwi pull LFM3-Audio-2B-GGUF`
> downloads both (about 1.46 GB), and Izwi estimates about 3 GB of memory.
> `LFM3-Audio-2B-Q8_0-GGUF` keeps the language model in Q8_0: about 2.20 GB
> to download and about 4 GB of memory. For speech it uses a `female` or
> `male` voice in 19 locales (American English by default; see
> [Voice Presets](/models/voice-presets#lfm3-audio)), streams 24 kHz audio,
> and splits long text into sentences; it has no voice cloning, voice
> description or speed control. It transcribes 18 languages, and in voice
> chat answers a spoken turn with interleaved text and speech, and can call
> tools, through the same speech-to-speech routes as LFM2.5 Audio. It is the
> default **Compact** voice model; LFM2.5 Audio stays selectable in **Voice
> settings › Engine**. It also streams realtime transcription in its native
> LiveASR mode. Prefix reuse is not supported yet. It runs on CPU, Metal and
> CUDA; CUDA is compile-checked only and has not yet run on an NVIDIA GPU.

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
| `Nemotron-3.5-Transcribe-30B-A4B-FP8` | NVIDIA's 30B speech language model in FP8: one pass of up to 60 minutes in 40 languages, translation to or from English, context terms, speaker labels and its own word timestamps; offline only; needs about 32 GiB of memory; experimental |
| `Granite-Speech-4.1-2B` | IBM Granite Speech ASR with punctuated, capitalized transcripts (en, fr, de, es, pt, ja); the default Granite Speech model |
| `Granite-Speech-4.1-2B-Plus` | IBM Granite Speech rich transcription model with speaker-attributed output and native word timestamps; unpunctuated by design (en, fr, de, es, pt) |
| `Granite-Speech-5.0-2B` | IBM Granite Speech 5.0: punctuated transcripts (en, fr, de, es, pt, ja), speech translation to and from English (plus English to Italian and Mandarin), and keyword lists of up to 2,000 terms; the recommended Granite Speech model |
| `Granite-Speech-5.0-2B-GGUF` | IBM's GGUF release of Granite Speech 5.0: the same transcription, translation and keywords with a Q8_0 language model, in less memory and faster (several times faster on CPU) |
| `LFM2.5-Audio-1.5B-GGUF` | Unified audio model (ASR + speech generation) |
| `LFM3-Audio-2B-GGUF` | Unified audio model (ASR in 18 languages, speech generation and voice chat); `LFM3-Audio-2B-Q8_0-GGUF` has a Q8_0 language model |
| `Voxtral-Mini-4B-Realtime-2602` | Mistral Voxtral offline transcription; realtime support planned |

> `Nemotron-3.5-Transcribe-30B-A4B-FP8` is NVIDIA's Nemotron 3.5 Transcribe
> (OpenMDW-1.1): a speech encoder and the Nemotron-3-Diarization speaker
> encoder feeding a 30B mixture-of-experts language model, with a CTC head
> that times words. It transcribes a recording of up to 60 minutes in one
> pass, in 40 languages (or detects the language), translates to or from
> English, takes context terms, labels speakers and times words without a
> forced aligner (see [Transcription](/features/transcription#nemotron-35-transcribe)).
> The download is 34.7 GB. It needs about 31.6 GiB of memory on Metal and
> CUDA and about 33.6 GiB on CPU, plus about 48 MiB per request in flight;
> machines that cannot hold it are refused when it loads, before any weights
> are read. CUDA needs compute capability 8.0 or newer. It is offline only:
> live transcription and voice mode do not use it. It can diarize on its own
> (see [Diarization](/features/diarization#single-model-diarization)), but it
> cannot be the ASR model of a diarization pipeline.
> **Experimental**: it has run end to end only on a tiny test checkpoint, on
> CPU and Metal. It has not yet run with its real weights on any backend, and
> CUDA is compile-checked only.

### Diarization and Alignment

| Task | Model |
|------|-------|
| Speaker diarization | `diar_streaming_sortformer_4spk-v2.1`, `Nemotron-3-Diarization` (8 speakers); `Nemotron-3.5-Transcribe-30B-A4B-FP8` diarizes on its own, with no pipeline (8 speakers; experimental) |
| Forced alignment | `Qwen3-ForcedAligner-0.6B` |

### Chat

| Family | Canonical IDs |
|--------|---------------|
| Qwen3.5 GGUF | `Qwen3.5-0.8B`, `Qwen3.5-2B`, `Qwen3.5-4B`, `Qwen3.5-9B` |
| Qwen3.6 FP8 | `Qwen3.6-35B-A3B-FP8` (text chat) |
| Qwen3.8 FP8 | `Qwen3.8-27B-FP8` (text chat; 28.74 GiB checkpoint payload) |
| LFM2.5 text | `LFM2.5-1.2B-Instruct-GGUF`, `LFM2.5-1.2B-Thinking-GGUF` |
| LFM3 text | `LFM3-1.7B-GGUF` (QAD Q4_0, default), `LFM3-1.7B-Q8_0-GGUF` |

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

# Download NVIDIA Nemotron 3.5 Transcribe (34.7 GB; needs about 32 GiB of memory)
izwi pull Nemotron-3.5-Transcribe-30B-A4B-FP8

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

# Download Resemble AI Chatterbox (about 3.2 GB each)
izwi pull Chatterbox-500M
izwi pull Chatterbox-Multilingual-500M

# Download Maya Research Maya1 (about 6.7 GB, or 3.6 GB as Q8_0 GGUF)
izwi pull Maya1-3B
izwi pull Maya1-3B-GGUF

# Download Maya Research Maya 2 Native (about 5.5 GB)
izwi pull Maya2-Native-2.5B

# Download Liquid AI LFM3 Audio (about 1.46 GB, or 2.20 GB with a Q8_0 language model)
izwi pull LFM3-Audio-2B-GGUF
izwi pull LFM3-Audio-2B-Q8_0-GGUF
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
