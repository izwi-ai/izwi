# Troubleshooting

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

4. Check firewall settings

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
CUDA_COMPUTE_CAP=80 docker compose --profile cuda up
```

Linux/Windows source build (CUDA):
```bash
# Linux source install
IZWI_BUILD_BACKEND=cuda ./scripts/install-cli.sh

# Windows package-scoped builds
cargo build --release -p izwi-cli --features cuda
cargo build --release -p izwi-server --features cuda
```

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
sw_vers  # Should be 12.0+
```

**Enable Metal:**
```bash
izwi serve --backend metal
```

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
| Transcription | `Parakeet-*`, `Whisper-*`, `Qwen3-ASR-*`, or `LFM2.5-Audio-*` |
| Chat | `Qwen3-*`, `Qwen3.5-*`, `LFM2.5-1.2B-*`, or `Gemma-3-1b-it` |
| Voice Cloning | `Qwen3-TTS-12Hz-*-Base*` |
| Voice Design | `Qwen3-TTS-12Hz-1.7B-VoiceDesign*` |

---

## API Issues

### 401 Unauthorized

Izwi doesn't require authentication by default. If you're getting this error:
- Check you're connecting to the right server
- Verify no proxy is interfering

### 404 Not Found

Check the endpoint URL:
- TTS: `POST /v1/audio/speech`
- Transcription: `POST /v1/audio/transcriptions`
- Chat: `POST /v1/chat/completions`

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

- [Installation](./installation/index.md)
- [Getting Started](./getting-started.md)
- [CLI Reference](./cli/index.md)
