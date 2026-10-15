---
title: "Troubleshooting"
description: "Common installation, model, audio, GPU, web UI, and API issues with practical fixes."
icon: "wrench"
---
Solutions to common issues with Izwi.

---

## Installation Issues

### macOS: "Izwi can't be opened because it is from an unidentified developer"

The app isn't code-signed yet:

1. Go to **System Settings → Privacy & Security**
2. Scroll down to find the Izwi message
3. Click **Open Anyway**

### macOS: Command not found: izwi

The CLI tools aren't in your PATH:

```bash
# Add to ~/.zshrc
export PATH="$HOME/.local/bin:$PATH"

# Reload
source ~/.zshrc
```

### Linux: Permission denied installing .deb

Use sudo:

```bash
sudo dpkg -i izwi_*.deb
```

### Windows: SmartScreen blocks installation

1. Click **More info**
2. Click **Run anyway**

---

## Server Issues

### Server won't start

**Check if port is in use:**

```bash
lsof -i :8080
```

Use a different port:

```bash
izwi serve --port 9000
```

**Check for existing Izwi processes:**

```bash
pkill -f izwi
izwi serve
```

**"refusing to bind ... without an API key":** a non-loopback host such as
`0.0.0.0` requires `IZWI_API_KEY`. Set one, or bind `127.0.0.1` (the default):

```bash
export IZWI_API_KEY="$(openssl rand -hex 32)"
izwi serve --host 0.0.0.0
```

### Can't connect to server

1. Verify server is running:
   ```bash
   izwi status
   ```

2. Check the operational probes:
   ```bash
   curl -f http://localhost:8080/livez
   curl -f http://localhost:8080/readyz
   ```

3. Check the correct URL:
   ```
   http://localhost:8080
   ```

4. From another device: the server listens on `127.0.0.1` by default. Start
   it with `--host 0.0.0.0` and an `IZWI_API_KEY` (see
   [Security And CORS](/api#security-and-cors)).

5. Check firewall settings

6. A `403` saying the `Host` header is not allowed means the server is bound
   to loopback and was reached through another name (a reverse proxy that
   forwards the original `Host`, or DNS rebinding). Make the proxy send
   `Host: 127.0.0.1:8080`, or bind a network address with an API key.

### Server crashes on startup

**Check logs:**

```bash
izwi serve --log-level debug
```

**Common causes:**
- Insufficient memory
- Corrupted model files
- Missing dependencies

---

## Model Issues

### Model download fails

**Network issues:**
- Check internet connection
- Try again (downloads resume automatically)
- Use a VPN if region-blocked

**Disk space:**
```bash
df -h
```

Ensure you have enough free space (models can be 1-10+ GB).

**Corrupted download:**
```bash
izwi rm <model-name>
izwi pull <model-name>
```

### Model won't load

**Insufficient memory:**

Check available RAM:
```bash
# macOS/Linux
free -h

# Or check Activity Monitor / Task Manager
```

Try a smaller model or close other applications.

For `Qwen3.8-27B-FP8` on CUDA, inspect the loaded entry returned by
`/v1/health`. A successful compressed load reports
`family_diagnostics.resident_representation` as
`q8_0_requantized_projections_with_dense_bf16` and
`fp8_execution_mode` as `q8_0_compressed_fallback`. This is intended for
40/48 GB-class devices with a resource-fitted context, but admission can still
fail when free VRAM or allocator headroom is insufficient. It is not a native
FP8 mode; see the [support matrix](/support-matrix#qwen38-cuda-weight-residency).

For Qwen3.8 responses that stop with `No finite Qwen3.8 sampling distribution`,
check `family_diagnostics.optimization_evidence.cuda_kv_storage` in the loaded
model diagnostics. BF16 model activations must retain their exponent range:
F16 KV conversion can turn a finite value into infinity and corrupt subsequent
attention. Supported CUDA devices (observed compute capability 8.0+) now default
to `storage_dtype: bf16` and `selected_provider: cuda_bf16`, with the same KV
memory footprint. Rebuild/restart and reload the model to apply the policy.
Remove an explicit `IZWI_QWEN38_CUDA_BF16_KV=0` override to use the default;
that override remains available for controlled F16 comparisons. Sampling
failures include the target/draft/bonus stage and bounded numerical diagnostics;
retain those details if a failure recurs under BF16 KV.

If the diagnostic reports `phase=draft` and `finite=0`, the optional MTP head
has produced an unusable proposal. Izwi now discards that entire speculative
round, restores its cache position and draft RNG, and uses target-only sampling
for the rest of that request. This recovery is automatic with MTP enabled;
other requests retain their own MTP policy. The warning records the failing
position and draft depth, and
`optimization_evidence.counters.mtp_nonfinite_draft_fallbacks_total` counts
requests switched to target-only sampling. MTP cache maintenance continues to
preserve the loaded adapter's state contract, so recovery does not remove all
MTP computation.
It prevents an invalid optional draft from aborting a healthy target stream;
it does not establish or repair the underlying source of the MTP NaNs.
Target/bonus numerical failures and backend execution failures remain errors.

For long-context requests, inspect `runtime_metrics.kv_cache.models` in the
health/admin diagnostics. `single_sequence_token_capacity` is the largest
sequence the fitted pools can retain, while `full_context_sequence_capacity`
is how many such sequences fit concurrently. Each arena also reports
`token_capacity`, full-request page claims, and workspace budget/high-water
bytes. Izwi reserves the exact prompt plus requested maximum output logically
before dispatch; reduce `max_tokens` or concurrency when that complete demand
does not fit. It does not evict arbitrary tokens from an active full-attention
sequence.

`CUDA_ERROR_OUT_OF_MEMORY` should not be returned for ordinary managed-capacity
pressure. If it appears after this version, capture the exact Git SHA, loaded
model diagnostics, the managed-KV snapshots before/after the request, and the
CUDA driver/device profile; treat it as an allocator/runtime defect rather than
raising the advertised context limit.

**Corrupted model:**
```bash
izwi rm <model-name>
izwi pull <model-name>
```

### Model not detected after manual download

1. Verify correct directory:
   - macOS: `~/Library/Application Support/izwi/models/`
   - Linux: `~/.local/share/izwi/models/`
   - Windows: `%APPDATA%\izwi\models\`

2. Check folder name matches expected variant name

3. Restart the server:
   ```bash
   izwi serve
   ```

---

## Performance Issues

### Inference is slow

**Use GPU acceleration:**

macOS (Metal):
```bash
izwi serve --backend metal
```

Linux (Docker CUDA):
```bash
IZWI_API_KEY="$(openssl rand -hex 32)" CUDA_COMPUTE_CAP=80 docker compose --profile cuda up
```

Linux/Windows source build (CUDA):
```bash
# Linux source install
IZWI_BUILD_BACKEND=cuda ./scripts/install-cli.sh

# Windows package-scoped builds
cargo build --release -p izwi-cli --features cuda
cargo build --release -p izwi-server --features cuda
```

For Whisper CUDA performance experiments, add Candle-backed feature flags only
on hosts with the matching NVIDIA libraries, for example
`--features cuda,flash-attn` or `--features cuda,cudnn`.

**Use smaller models:**
- `Qwen3-TTS-12Hz-0.6B-Base` instead of `Qwen3-TTS-12Hz-1.7B-Base`
- Quantized variants (`-4bit`)

**Close other applications** to free memory

### High memory usage

**Unload unused models:**
```bash
izwi models unload <model-name>
```

**Use quantized models** for lower memory footprint

### Audio generation stutters

- Ensure models are fully loaded before use
- Use streaming mode for long text
- Check system resources

---

## Audio Issues

### No audio output

**Check system audio:**
- Verify speakers/headphones are connected
- Check system volume
- Test with another application

**Check audio file:**
```bash
# Play with system player
afplay output.wav  # macOS
aplay output.wav   # Linux
```

### Poor transcription quality

**Improve audio quality:**
- Use a better microphone
- Reduce background noise
- Speak clearly

**Use a larger model:**
```bash
izwi pull Qwen3-ASR-1.7B-GGUF
izwi transcribe audio.wav --model Qwen3-ASR-1.7B-GGUF
```

**Specify language:**
```bash
izwi transcribe audio.wav --language en
```

### Microphone not detected (Web UI)

1. Check browser permissions for microphone access
2. Ensure correct input device is selected in system settings
3. Try a different browser

---

## GPU Issues

### Metal not working (macOS)

**Verify Apple Silicon:**
```bash
uname -m  # Should show "arm64"
```

**Check macOS version:**
```bash
sw_vers  # Should be 15.0+ for Metal
```

**Enable Metal:**
```bash
izwi serve --backend metal
```

Metal requires macOS 15 or later. On macOS 12-14, Izwi keeps running on CPU
and an explicit Metal request reports a CPU fallback.

### CUDA not detected (Linux/Windows)

**Check NVIDIA drivers:**
```bash
nvidia-smi
```

**For Docker CUDA on Linux, verify the NVIDIA driver and container runtime. For source builds, verify CUDA Toolkit installation:**
```bash
nvcc --version
```

**If you installed from source, rebuild with CUDA support:**
```bash
# Linux source install
IZWI_BUILD_BACKEND=cuda ./scripts/install-cli.sh

# Windows package-scoped builds
cargo build --release -p izwi-cli --features cuda
cargo build --release -p izwi-server --features cuda
```

**Verify runtime backend state:**
```bash
izwi serve --backend cuda
izwi status --detailed
```

---

## Web UI Issues

### UI won't load

1. Verify server is running:
   ```bash
   izwi status
   ```

2. Check the URL: `http://localhost:8080`

3. Clear browser cache

4. Try incognito/private mode

### UI shows "No models loaded"

1. Download a model:
   ```bash
   izwi pull Qwen3-TTS-12Hz-0.6B-Base
   ```

2. Load the model:
   ```bash
   izwi models load Qwen3-TTS-12Hz-0.6B-Base
   ```

3. Refresh the page

### Features not working

Ensure required models are loaded:

| Feature | Required Model Type |
|---------|---------------------|
| TTS | `*-tts-*` |
| Transcription | `Parakeet-*`, `Whisper-*`, `Qwen3-ASR-*`, `Granite-Speech-*`, or `LFM2.5-Audio-*` |
| Chat | `Qwen3-*`, `Qwen3.5-*`, `Qwen3.8-*`, or `LFM2.5-1.2B-*` |
| Voice Cloning | `Qwen3-TTS-12Hz-*-Base*` |
| Voice Design | `Qwen3-TTS-12Hz-1.7B-VoiceDesign*` |

---

## API Issues

### 401 Unauthorized

A server bound to a network address requires its `IZWI_API_KEY`:
- Send `Authorization: Bearer <key>` or `x-api-key: <key>`.
- For `izwi` CLI commands, export `IZWI_API_KEY` in the shell that runs them.
- In the web UI, reload the page and enter the key when asked.

A loopback-bound server (the default) needs no key; if you still get `401`,
check you're connecting to the right server and that no proxy is interfering.

### 403 Cross-origin request rejected

The server rejects writes from web pages on other sites. If your own web app
calls the API, add its origin to `cors_origins` (or `IZWI_CORS_ORIGINS`).

### 404 Not Found

Check the endpoint URL:
- TTS: `POST /v1/audio/speech`
- Transcription: `POST /v1/audio/transcriptions`
- Chat: `POST /v1/chat/completions`

For the full route list, including preview first-party APIs, admin routes, and
realtime WebSockets, see the [API Reference](/api).

### 500 Internal Server Error

Check server logs:
```bash
izwi serve --log-level debug
```

Common causes:
- Model not loaded
- Invalid request format
- Insufficient memory

---

## Getting More Help

### Collect diagnostic information

```bash
izwi version --full
izwi status --detailed
```

### Check logs

```bash
izwi serve --log-level debug
```

### Report issues

Open an issue on GitHub with:
1. Izwi version (`izwi version --full`)
2. Operating system and version
3. Steps to reproduce
4. Error messages and logs

[GitHub Issues](https://github.com/izwi-ai/izwi/issues)

---

## See Also

- [Installation](/installation)
- [Getting Started](/getting-started)
- [CLI Reference](/cli)
