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

- **macOS (Apple Silicon):** open `Izwi_<version>_aarch64.dmg`, drag
  `Izwi.app` to Applications, then launch it.
- **Linux:** install the `.deb` package with
  `sudo apt install ./Izwi_*_amd64.deb`, or run `Izwi_<version>_amd64.AppImage`.
- **Windows:** run `Izwi_<version>_x64-setup.exe`.

To use `izwi` in a terminal, open **Settings > System > Command line tool** in
the app and select **Install** (the `.deb` installs it for you).

Runtime support depends on the artifact:

- macOS Apple Silicon release builds use Metal on macOS 15+ and fall back to
  CPU on macOS 12-14.
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

In a second terminal, download the default speech model and generate speech
(Kokoro-82M needs `espeak-ng`: `brew install espeak-ng` or
`sudo apt install espeak-ng`):

```bash
izwi pull Kokoro-82M
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

- **TTS:** Qwen3-TTS, Kokoro-82M, Voxtral TTS, VibeVoice, Fish Audio S2 Pro,
  and LFM2.5 Audio.
- **ASR:** Parakeet, Whisper, Qwen3-ASR, Nemotron 3.5 ASR, VibeVoice ASR,
  Granite Speech 4.1, LFM2.5 Audio, and Voxtral Mini.
- **Diarization and alignment:** Sortformer and Nemotron 3 diarization, and
  Qwen3 ForcedAligner.
- **Chat:** Qwen3.5, Qwen3.6, Qwen3.8, and LFM2.5.

Some model weights and bundled assets have their own licenses or access terms.
Check the [Models Guide](https://izwiai.com/docs/models) before redistribution
or commercial use of downloaded model artifacts.

## Build From Source

For CLI/server installs, use the project install script:

```bash
./scripts/install-cli.sh
```

It needs Node.js 18+ (it installs Rust if missing), and on Linux the Tauri
system packages (see
[From Source](https://izwiai.com/docs/installation/from-source)).

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

Izwi is licensed under the [Apache License 2.0](LICENSE).
