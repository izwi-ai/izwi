<p align="center">
  <img src="images/app-icon.png" alt="Izwi icon" width="140" />
</p>

<h1 align="center">Izwi</h1>

<p align="center"><strong>Local-first voice AI for speech, chat, and audio workflows.</strong></p>

<p align="center">
  <a href="https://izwiai.com">Website</a> -
  <a href="https://izwiai.com/docs">Documentation</a> -
  <a href="https://github.com/izwi-ai/izwi/releases">Releases</a> -
  <a href="https://github.com/izwi-ai/izwi/issues">Issues</a>
</p>

<p align="center">
  <img src="images/screenshot.png" alt="Izwi app screenshot" width="800" />
</p>

Izwi is a desktop app, web UI, CLI, and local inference server for voice AI.
It runs on your machine and exposes both product workflows and OpenAI-compatible
API routes without requiring cloud services or API keys.

## What It Does

- Real-time voice conversations with local ASR, chat, and TTS models.
- Text-to-speech, long-form Studio projects, voice cloning, voice design, and
  saved voices.
- Transcription, speaker diarization, forced alignment, and realtime speech-to-text.
- Local chat, model download/load/unload/delete, history, exports, and settings.
- OpenAI-compatible `/v1` APIs for models, chat completions, audio speech,
  audio transcriptions, and preview Responses support.

Inference data stays local. Optional anonymous desktop analytics are disabled
unless a user explicitly opts in, and they do not send prompts, transcripts,
audio payloads, local paths, or personal identifiers.

## Install

Download the latest build from
[GitHub Releases](https://github.com/izwi-ai/izwi/releases).

- **macOS:** install the `.dmg`, drag `Izwi.app` to Applications, then launch
  it.
- **Linux:** install the `.deb` package with `sudo dpkg -i izwi_*.deb`.
- **Windows:** run the `.exe` installer.

Runtime support depends on the artifact:

- macOS Apple Silicon release builds use Metal.
- Linux and Windows release builds are CPU-only.
- CUDA is supported through the Docker CUDA profile or source builds on
  compatible NVIDIA hosts.

See the [Runtime Support Matrix](https://izwiai.com/docs/support-matrix) for
the full contract.

## Quick Start

Start the local server and web UI:

```bash
izwi serve --mode web
```

Server-only and desktop modes are also available:

```bash
izwi serve
izwi serve --mode desktop
```

Download a model and generate speech:

```bash
izwi pull Qwen3-TTS-12Hz-0.6B-Base
izwi tts "Hello from Izwi." --output hello.wav
```

Transcribe audio:

```bash
izwi pull Parakeet-TDT-0.6B-v3
izwi transcribe audio.wav --model Parakeet-TDT-0.6B-v3
```

Open the app at `http://localhost:8080`. The local API reference is available
at `http://localhost:8080/docs`, and the raw OpenAPI document is available at
`http://localhost:8080/openapi.json`.

## Model Families

Run `izwi list` to see the enabled catalog. Current families include:

- **TTS:** Qwen3-TTS, Kokoro-82M, Voxtral TTS, and VibeVoice.
- **ASR:** Parakeet, Whisper, Qwen3-ASR, Nemotron 3.5 ASR, VibeVoice ASR,
  LFM2.5 Audio, and Voxtral Mini.
- **Diarization and alignment:** Sortformer diarization and Qwen3 ForcedAligner.
- **Chat:** Qwen3, Qwen3.5, LFM2.5, and Gemma.

Some model weights and bundled assets have their own licenses or access terms.
Check the [Models Guide](https://izwiai.com/docs/models) before redistribution
or commercial use of downloaded model artifacts.

## Build From Source

For CLI/server installs, use the project install script:

```bash
./scripts/install-cli.sh
```

For manual builds, scope Cargo to the binaries you need:

```bash
cargo build --release -p izwi-cli
cargo build --release -p izwi-server
```

CUDA source builds require the matching CUDA toolkit and Cargo features for the
target host. See [From Source](https://izwiai.com/docs/installation/from-source)
for platform-specific setup.

## Documentation

- [Getting Started](https://izwiai.com/docs/getting-started)
- [Installation](https://izwiai.com/docs/installation)
- [Features](https://izwiai.com/docs/features)
- [CLI Reference](https://izwiai.com/docs/cli)
- [API Reference](https://izwiai.com/docs/api)
- [Models](https://izwiai.com/docs/models)
- [Troubleshooting](https://izwiai.com/docs/troubleshooting)

## License

Izwi is licensed under the [MIT License](LICENSE).
