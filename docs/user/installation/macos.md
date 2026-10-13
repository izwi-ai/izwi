---
title: "macOS Installation"
description: "Install Izwi on macOS with the DMG, optional CLI tools, and Apple Silicon acceleration."
sidebarTitle: "macOS"
icon: "apple"
---
Izwi supports macOS 12 and later. Apple Silicon Macs use Metal acceleration on
macOS 15+; macOS 12-14 remain supported with the CPU backend.

---

## Requirements

- **macOS 12.0** (Monterey) or later
- **8 GB RAM** minimum (16 GB recommended)
- **10 GB** free disk space (more for models)

---

## Install from DMG (Recommended)

### Step 1: Download

Download the latest `Izwi-*.dmg` from [GitHub Releases](https://github.com/izwi-ai/izwi/releases).

### Step 2: Install

1. Open the downloaded `.dmg` file
2. Drag **Izwi.app** into your **Applications** folder
3. Eject the disk image

### Step 3: First Launch

1. Open **Izwi** from Applications
2. If prompted about an unidentified developer:
   - Go to **System Settings → Privacy & Security**
   - Click **Open Anyway** next to the Izwi message
3. On first launch, Izwi will:
   - Create the `~/.local/bin` directory if needed
   - Link `izwi` and `izwi-server` to your PATH
   - May prompt for administrator access

### Step 4: Verify

Open Terminal and run:

```bash
izwi version --full
```

---

## Optional: Install `espeak-ng` for Kokoro-82M

`Kokoro-82M` uses `espeak-ng` for phonemization. Install it before using Kokoro voices.

### Install with Homebrew

```bash
brew install espeak-ng
```

### Verify

```bash
espeak-ng --version
```

---

## Install via Homebrew (Coming Soon)

```bash
brew install --cask izwi
```

---

## Command-Line Only Installation

If you only need the CLI tools without the desktop app:

```bash
# Clone the repository
git clone https://github.com/izwi-ai/izwi.git
cd izwi

# Build and install (Metal on macOS 15+, CPU fallback on macOS 12-14)
IZWI_BUILD_BACKEND=metal ./scripts/install-cli.sh
```

This installs `izwi`, `izwi-server`, and `izwi-desktop` to `~/.local/bin`.

---

## Apple Silicon Optimization

Izwi automatically selects CPU on macOS 12-14 and Metal on Apple Silicon with
macOS 15+. On macOS 15+, you can explicitly request Metal:

```bash
izwi serve --backend metal
```

Or set the environment variable:

```bash
export IZWI_BACKEND=metal
izwi serve
```

### Metal precision

Each model family picks its own precision on Metal. `IZWI_METAL_DTYPE`
overrides that choice for every family:

| Value | Effect |
|-------|--------|
| `auto` (default) | Each family's Metal policy. |
| `f32` | F32 weights and activations for every family that uses the shared dtype policy, and F32 encoder weights for Parakeet and Nemotron. Use this if a model produces wrong or empty output on Metal. |
| `f16`, `bf16` | Lower-precision default for families that use the shared dtype policy. Experimental. BF16 is slower than F16 on M1 and M2. |

A per-family variable such as `IZWI_WHISPER_DTYPE` is more specific and takes
precedence over `IZWI_METAL_DTYPE`. On Metal, a per-family `f16` or `bf16`
setting is now applied as written instead of being replaced with F32.

On Metal, Parakeet and Nemotron store their encoder's linear and
pointwise-convolution weights in F16 and keep activations, normalization,
convolutions and the decoder in F32, as whisper.cpp does. This roughly halves
their weight memory (about 2.5 GB to 1.3 GB each). A numeric self-check runs
at load; if it fails, the model loads in F32 and the reason appears in its
diagnostics under `weight_precision`. Set `IZWI_PARAKEET_ASR_DTYPE=f32` or
`IZWI_NEMOTRON_ASR_DTYPE=f32` (or `IZWI_METAL_DTYPE=f32`) to keep every weight
in F32.

Granite Speech applies the same split to its conformer audio encoder (about
0.8 GiB less weight memory); its self-check result appears under
`execution.encoder_weight_precision`, and `IZWI_GRANITE_SPEECH_DTYPE=f32` (or
`IZWI_METAL_DTYPE=f32`) keeps the encoder in F32.

Whisper runs its F16 checkpoint in F16 on every Apple GPU, as whisper.cpp,
mlx-whisper and OpenAI Whisper do. The log-mel input, attention softmax,
LayerNorm statistics and logits stay in F32. This halves Whisper's weight
memory (about 3.0 GiB to 1.5 GiB). A numeric self-check runs one encoder
window and one decoder step at load; if it fails, Whisper loads in F32 and the
reason appears in its diagnostics under `device.precision`. Set
`IZWI_WHISPER_DTYPE=f32` (or `IZWI_METAL_DTYPE=f32`) to run Whisper in F32.

---

## Data Locations

| Data | Location |
|------|----------|
| **Models** | `~/Library/Application Support/izwi/models/` |
| **Config** | `~/Library/Application Support/izwi/config.toml` |
| **Logs** | `~/Library/Application Support/izwi/logs/` |
| **Cache** | `~/Library/Caches/com.agentem.izwi.desktop/` |

---

## Uninstall

### Quick Uninstall

From a cloned repository:

```bash
./scripts/uninstall-macos.sh
```

Or download the uninstall script from releases:

```bash
chmod +x uninstall-izwi-macos.sh
./uninstall-izwi-macos.sh
```

### Manual Uninstall

```bash
# Stop any running processes
pkill -f "izwi-server|izwi serve|izwi-desktop" 2>/dev/null || true

# Remove the app
sudo rm -rf "/Applications/Izwi.app"

# Remove CLI tools
sudo rm -f /opt/homebrew/bin/izwi /opt/homebrew/bin/izwi-server
sudo rm -f /usr/local/bin/izwi /usr/local/bin/izwi-server

# Remove data (optional - this deletes your models!)
rm -rf "$HOME/Library/Application Support/izwi"
rm -rf "$HOME/Library/Application Support/com.agentem.izwi.desktop"
rm -rf "$HOME/Library/Caches/com.agentem.izwi.desktop"
rm -rf "$HOME/Library/Saved Application State/com.agentem.izwi.desktop.savedState"
rm -rf "$HOME/Library/WebKit/com.agentem.izwi.desktop"
```

### Verify Removal

```bash
command -v izwi || echo "izwi removed"
pgrep -af "izwi" || echo "no izwi processes running"
```

---

## Troubleshooting

### "Izwi can't be opened because it is from an unidentified developer"

1. Go to **System Settings → Privacy & Security**
2. Scroll down to find the Izwi message
3. Click **Open Anyway**

### Command not found: izwi

The CLI tools may not be in your PATH. Add this to your `~/.zshrc`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Then reload:

```bash
source ~/.zshrc
```

### Kokoro error: `espeak-ng not found`

Install `espeak-ng`:

```bash
brew install espeak-ng
espeak-ng --version
```

### Metal acceleration not working

Ensure you're on Apple Silicon and running macOS 15.0+:

```bash
# Check your chip
uname -m  # Should show "arm64"

# Check macOS version
sw_vers
```

---

## Next Steps

- [Getting Started](/getting-started)
- [Download Models](/models)
