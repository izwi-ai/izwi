# Izwi CLI

Command-line interface for running and testing Izwi TTS/ASR locally.

## Why use the CLI

The CLI is the fastest way to:
- start the Izwi server,
- manage local models,
- run TTS/ASR from your terminal,
- smoke test the whole stack end-to-end.

## Install

### From this repo (recommended for development)

```bash
# from repository root
cargo install --path crates/izwi-cli --force --offline
```

### Build without installing

```bash
cargo build -p izwi-cli
./target/debug/izwi --help
```

### Install script (macOS/Linux)

```bash
./scripts/install-cli.sh
```

## Quick start

1) Start server:

```bash
izwi serve
# or server + desktop shell:
izwi serve --mode desktop
```

`--mode desktop` requires the `izwi-desktop` binary to be present (installed by `./scripts/install-cli.sh`).

2) In a second terminal, check health and models:

```bash
izwi status
izwi list --local
```

3) Run ASR:

```bash
izwi transcribe data/test.wav --model qwen3-asr-0.6b --format text
```

4) Run TTS:

```bash
izwi tts "hello from izwi cli" \
  --model qwen3-tts-0.6b-base \
  --format wav \
  --output /tmp/hello.wav
```

## Common commands

### Server

```bash
izwi serve
izwi serve --mode desktop
izwi serve --host 0.0.0.0 --port 8080 --backend metal
izwi serve --dev
```

### Models

```bash
izwi list
izwi list --local
izwi list --local --output-format json

izwi pull qwen3-tts-0.6b-base
izwi rm qwen3-tts-0.6b-base

izwi models list --local --detailed
izwi models info qwen3-asr-0.6b
izwi models load qwen3-tts-0.6b-base --wait
izwi models unload qwen3-tts-0.6b-base
izwi models progress qwen3-asr-0.6b
```

### TTS

```bash
izwi tts "Hello world"
izwi tts "Hello world" --speaker default --speed 1.0 --temperature 0.7
echo "Hello from stdin" | izwi tts -
izwi tts "Stream me" --stream --output /tmp/stream.wav
```

### ASR

```bash
izwi transcribe audio.wav
izwi transcribe audio.wav --format json
izwi transcribe audio.wav --format verbose-json
izwi transcribe audio.wav --language English --output transcript.txt
```

Long recordings are transcribed end-to-end using automatic chunking and overlap
stitching across ASR backends.

Optional chunking controls:

```bash
IZWI_ASR_CHUNK_TARGET_SECS=24
IZWI_ASR_CHUNK_MAX_SECS=30
IZWI_ASR_CHUNK_OVERLAP_SECS=3
# Optional: preload/warmup ASR models at server startup for lower first-request latency.
IZWI_PRELOAD_MODELS=Whisper-Large-v3-Turbo
IZWI_WARMUP_PRELOADED_MODELS=1
IZWI_ASR_WARMUP_DURATION_MS=800
# Optional: tune text streaming queue depth for per-character ASR streaming.
IZWI_STREAM_TEXT_QUEUE_CAPACITY=4096
```

Note: `--word-timestamps` is accepted by the CLI but currently ignored by the server.

### Chat

```bash
izwi chat --model qwen3-0.6b
izwi chat --model qwen3-0.6b-4bit
izwi chat --model qwen3-0.6b-4bit --system "You are concise."
```

### Benchmarks

```bash
izwi bench chat --model Qwen3.5-4B --iterations 5 --max-tokens 128 --warmup
izwi bench tts --model qwen3-tts-0.6b-base --iterations 5 --concurrent 2
izwi bench asr --model parakeet-tdt-0.6b-v3 --file data/test.wav --iterations 3 --concurrent 2
izwi bench throughput --duration 10 --concurrent 2
izwi --output-format json bench chat --iterations 5
izwi --output-format json bench run benchmarks/local.toml
izwi bench compare current.json baseline.json --tolerance-percent 5
```

### Config

```bash
izwi config show
izwi config set server.host 127.0.0.1
izwi config get server.host
izwi --config /tmp/izwi.toml config set server.port 8080
izwi config path
```

### Completions

```bash
izwi completions bash > ~/.izwi-completion.bash
izwi completions zsh > ~/.zsh/completions/_izwi
izwi completions fish > ~/.config/fish/completions/izwi.fish
```

## Global options

```text
--config <PATH>             Configuration file path
--server <URL>              API server base URL (default: http://localhost:8080)
--output-format <FORMAT>    table | json | plain | yaml
--quiet                     Suppress non-result output
--verbose                   Enable verbose logs
--no-color                  Disable colored output
```

## Environment variables

```bash
# Server defaults
IZWI_HOST=0.0.0.0
IZWI_PORT=8080
IZWI_SERVE_MODE=server
IZWI_MODELS_DIR=/path/to/models
IZWI_BACKEND=metal
IZWI_NUM_THREADS=8
IZWI_MAX_BATCH_SIZE=8
IZWI_MAX_CONCURRENT=100
IZWI_TIMEOUT=300

# Logging
RUST_LOG=info

# Color control
NO_COLOR=1
```

## Troubleshooting

### Cannot connect to server
- Run `izwi serve` first.
- Or pass a server URL explicitly: `izwi --server http://127.0.0.1:8080 status`.

### Model not found
- Check available models: `izwi list`.
- Download model: `izwi pull <model>`.

### No colors in output
- Unset `NO_COLOR` if you want colors.
