# Linux Installation

Izwi runs on most modern Linux distributions with optional CUDA support for NVIDIA GPUs.

---

## Requirements

- **Ubuntu 20.04+**, Debian 11+, Fedora 36+, or similar
- **8 GB RAM** minimum (16 GB recommended)
- **10 GB** free disk space (more for models)
- **NVIDIA GPU** (optional, for CUDA acceleration)

---

## Install from .deb Package (Debian/Ubuntu)

### Step 1: Download

```bash
wget https://github.com/izwi-ai/izwi/releases/latest/download/izwi_amd64.deb
```

### Step 2: Install

```bash
sudo dpkg -i izwi_amd64.deb
```

If you encounter dependency errors:

```bash
sudo apt-get install -f
```

### Step 3: Verify

```bash
izwi --version
izwi status
```

---

## Optional: Install `espeak-ng` for Kokoro-82M

`Kokoro-82M` uses `espeak-ng` for phonemization. Install it before using Kokoro voices.

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install -y espeak-ng
```

### Fedora

```bash
sudo dnf install -y espeak-ng
```

### Arch Linux

```bash
sudo pacman -S espeak-ng
```

### Verify

```bash
espeak-ng --version
```

---

## Install from Source

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential curl git pkg-config libssl-dev

# Fedora
sudo dnf install -y gcc gcc-c++ make curl git openssl-devel

# Arch Linux
sudo pacman -S base-devel curl git openssl
```

### Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustup update stable
```

### Build Izwi

```bash
# Clone the repository
git clone https://github.com/izwi-ai/izwi.git
cd izwi

# Build release binaries
cargo build --release

# Install CLI tools
./scripts/install-cli.sh
```

---

## CUDA Support (NVIDIA GPUs)

For NVIDIA GPU acceleration:

### Step 1: Install CUDA Toolkit

```bash
# Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-4
```

### Step 2: Build with CUDA

```bash
cargo build --release --features cuda
```

### Step 3: Verify CUDA

```bash
izwi status --detailed
```

Look for "CUDA: available" in the output.

---

## Data Locations

| Data | Location |
|------|----------|
| **Models** | `~/.local/share/izwi/models/` |
| **Config** | `~/.config/izwi/config.toml` |
| **Logs** | `~/.local/share/izwi/logs/` |
| **Binaries** | `/usr/bin/` (deb) or `~/.local/bin/` (source) |

---

## Running as a Service

Create a systemd service for automatic startup:

```bash
sudo tee /etc/systemd/system/izwi.service << 'EOF'
[Unit]
Description=Izwi Audio Inference Server
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
ExecStart=/usr/bin/izwi serve --host 0.0.0.0 --port 8080
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
```

Replace `YOUR_USERNAME` with your actual username, then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable izwi
sudo systemctl start izwi
```

Check status:

```bash
sudo systemctl status izwi
```

---

## Uninstall

### From .deb Package

```bash
sudo dpkg -r izwi
```

### From Source

```bash
rm -f ~/.local/bin/izwi ~/.local/bin/izwi-server ~/.local/bin/izwi-desktop
```

### Remove Data (Optional)

```bash
rm -rf ~/.local/share/izwi
rm -rf ~/.config/izwi
```

---

## Troubleshooting

### Permission denied when binding to port 8080

Either use a higher port or run with elevated privileges:

```bash
# Use a different port
izwi serve --port 8888

# Or allow binding to privileged ports (not recommended)
sudo setcap 'cap_net_bind_service=+ep' $(which izwi-server)
```

### CUDA not detected

1. Verify NVIDIA drivers are installed:
   ```bash
   nvidia-smi
   ```

2. Ensure CUDA toolkit is in your PATH:
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

3. Rebuild with CUDA feature:
   ```bash
   cargo build --release --features cuda
   ```

### Audio playback not working

Install audio libraries:

```bash
# Ubuntu/Debian
sudo apt install -y libasound2-dev

# Fedora
sudo dnf install -y alsa-lib-devel
```

### Kokoro error: `espeak-ng not found`

Install `espeak-ng` with your distro package manager, then verify:

```bash
espeak-ng --version
```

---

## Next Steps

- [Getting Started](../getting-started.md)
- [Download Models](../models/index.md)
