# macOS Installation

Izwi is optimized for macOS with native Apple Silicon (M1/M2/M3/M4) acceleration via Metal.

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
izwi --version
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

# Build with Metal support
cargo build --release --features metal

# Install CLI tools
./scripts/install-cli.sh
```

This installs `izwi`, `izwi-server`, and `izwi-desktop` to `~/.local/bin`.

---

## Apple Silicon Optimization

Izwi automatically selects the best backend on Apple Silicon Macs. To explicitly force Metal:

```bash
izwi serve --backend metal
```

Or set the environment variable:

```bash
export IZWI_BACKEND=metal
izwi serve
```

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

Ensure you're on Apple Silicon and running macOS 12.0+:

```bash
# Check your chip
uname -m  # Should show "arm64"

# Check macOS version
sw_vers
```

---

## Next Steps

- [Getting Started](../getting-started.md)
- [Download Models](../models/index.md)
