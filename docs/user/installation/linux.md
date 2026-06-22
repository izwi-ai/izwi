---
title: "Linux Installation"
description: "Install Izwi on Linux with release packages, source builds, services, and CUDA guidance."
sidebarTitle: "Linux"
icon: "monitor"
---
# Linux Installation

Izwi runs on most modern Linux distributions. Linux GitHub Release assets keep the public `izwi` and `izwi-server` names and are intentionally CPU-only.

See the [Runtime Support Matrix](/support-matrix) for the current artifact contract.

---

## Requirements

- **Ubuntu 20.04+**, Debian 11+, Fedora 36+, or similar
- **8 GB RAM** minimum (16 GB recommended)
- **10 GB** free disk space (more for models)
- **NVIDIA GPU** (optional, for CUDA acceleration)

---

## Install from GitHub Releases

Linux release surfaces are CPU-only native artifacts:

- `.deb` package for Debian/Ubuntu installs
- AppImage desktop bundle and updater artifact
- terminal tarball

The public command names stay `izwi` and `izwi-server`. CUDA runtime libraries and CUDA-linked binaries are not bundled in native Linux release artifacts.

## Install from .deb Package (Debian/Ubuntu)

> The `.deb` package is CPU-only. Use Docker CUDA or a source build when you need NVIDIA acceleration.

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
izwi version --full
```

For terminal tarballs, unpack the release archive and keep the bundled binaries together.

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

# Install CLI tools (defaults to CPU on Linux)
./scripts/install-cli.sh
```

---

## CUDA Support (NVIDIA GPUs)

Native Linux GitHub Release artifacts are CPU-only. For CUDA-capable binaries on NVIDIA Linux hosts, use the Docker CUDA profile:

```bash
nvidia-smi
git clone https://github.com/izwi-ai/izwi.git
cd izwi
CUDA_COMPUTE_CAP=80 docker compose --profile cuda up
```

Adjust `CUDA_COMPUTE_CAP` for the target GPU architecture if you build on a machine without `nvidia-smi`.

### Source Build CUDA Path

Source builds still require a CUDA toolkit.

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
IZWI_BUILD_BACKEND=cuda ./scripts/install-cli.sh
```

### Step 3: Verify CUDA

Start the server in one terminal:

```bash
izwi serve --backend cuda
```

In a second terminal:

```bash
izwi version --full
izwi status --detailed
```

For source builds, `izwi version --full` should include `CUDA` under **Compiled Backends**. Runtime selection still comes from `izwi status --detailed`.

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

2. If you installed from source, ensure CUDA toolkit is in your PATH:
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

3. If you installed from source, rebuild with CUDA:
   ```bash
   IZWI_BUILD_BACKEND=cuda ./scripts/install-cli.sh
   ```

4. Verify the runtime:
   ```bash
   izwi serve --backend cuda
   izwi status --detailed
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

- [Getting Started](/getting-started)
- [Download Models](/models)
