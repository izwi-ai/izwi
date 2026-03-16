# Getting Started

Get Izwi running in under 5 minutes.

---

## Step 1: Install Izwi

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

### Windows

Download and run the `.exe` installer from [GitHub Releases](https://github.com/izwi-ai/izwi/releases).

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
izwi pull qwen3-tts-0.6b-base
```

This downloads a compact text-to-speech model (~1.2 GB).

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
izwi pull qwen3-asr-0.6b
```

Then transcribe:

```bash
izwi transcribe your-audio.wav
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
