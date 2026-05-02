# Getting Started

Get Izwi running in under 5 minutes.

---

## Step 1: Install Izwi

Before you pick an install path, check the [Runtime Support Matrix](./support-matrix.md) if backend support matters for your deployment. Linux and Windows release installers are CPU-only; use Docker CUDA on NVIDIA Linux hosts or source builds for custom CUDA validation.

### macOS

Download the latest `.dmg` from [GitHub Releases](https://github.com/izwi-ai/izwi/releases), then:

1. Open the downloaded `.dmg` file
2. Drag **Izwi.app** to your Applications folder
3. Launch Izwi from Applications

On first launch, Izwi will set up the `izwi` command-line tool automatically.

### Linux

Download the `.deb` package from [GitHub Releases](https://github.com/izwi-ai/izwi/releases):

```bash
sudo dpkg -i izwi_*.deb
```

> The Linux release package is CPU-only. Use Docker CUDA or a source build for NVIDIA acceleration.

### Windows

Download and run the `.exe` installer from [GitHub Releases](https://github.com/izwi-ai/izwi/releases).

> The Windows release package is CPU-only. CUDA on Windows is currently a source-build preview path.

> See [Installation](./installation/index.md) for detailed platform-specific instructions.

---

## Step 2: Start the Server

Open a terminal and run:

```bash
izwi serve
```

You should see:

```
Izwi server running at http://localhost:8080
```

**Desktop mode** (opens the native app):

```bash
izwi serve --mode desktop
```

**Web mode** (opens in your browser):

```bash
izwi serve --mode web
```

---

## Step 3: Download a Model

Izwi needs AI models to work. Download your first model:

```bash
izwi pull Qwen3-TTS-12Hz-0.6B-Base
```

This downloads a compact, general-purpose text-to-speech model.

View all available models:

```bash
izwi list
```

View your downloaded models:

```bash
izwi list --local
```

---

## Step 4: Try It Out

### Generate Speech

```bash
izwi tts "Hello! Welcome to Izwi." --output hello.wav
```

Play the generated audio:

```bash
izwi tts "Hello! Welcome to Izwi." --play
```

### Transcribe Audio

First, download an ASR model:

```bash
izwi pull Qwen3-ASR-0.6B-GGUF
```

Then transcribe:

```bash
izwi transcribe your-audio.wav --model Qwen3-ASR-0.6B-GGUF
```

---

## Step 5: Open the Web UI

With the server running, open your browser to:

```
http://localhost:8080
```

The web interface provides:

- **Voice** — Real-time voice conversations
- **Chat** — Text-based AI chat
- **Transcription** — Upload and transcribe audio files
- **Text to Speech** — Generate speech from text
- **Voice Cloning** — Clone voices from samples
- **Models** — Manage your downloaded models

---

## Next Steps

- [Learn about all features](./features/index.md)
- [Explore the CLI commands](./cli/index.md)
- [Download more models](./models/index.md)
- [Troubleshoot common issues](./troubleshooting.md)
