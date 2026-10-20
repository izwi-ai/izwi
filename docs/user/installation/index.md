---
title: "Installation"
description: "Choose the right Izwi installation path for macOS, Linux, Windows, or source builds."
icon: "download"
---
Before choosing an install path, check the [Runtime Support Matrix](/support-matrix) for the current backend contract. In particular:

- GitHub Release artifacts are not the same thing as CUDA source builds.
- Linux and Windows GitHub Release artifacts are CPU-only and do not bundle CUDA runtime libraries.
- CUDA acceleration requires a compatible NVIDIA driver and GPU; source builds still require the CUDA toolkit.
- The Docker CUDA profile is the CUDA distribution path for NVIDIA Linux hosts and may require `CUDA_COMPUTE_CAP` when built on a machine without `nvidia-smi`.

---

## Platforms

| Platform | Guide |
|----------|-------|
| **macOS** | [macOS Installation](/installation/macos) |
| **Linux** | [Linux Installation](/installation/linux) |
| **Windows** | [Windows Installation](/installation/windows) |
| **From Source** | [Build from Source](/installation/from-source) |

---

## Quick Install

### macOS

```bash
# Download and install Izwi_<version>_aarch64.dmg (Apple Silicon) from GitHub Releases
open https://github.com/izwi-ai/izwi/releases
```

### Linux (Debian/Ubuntu)

Download `Izwi_<version>_amd64.deb` from the [Releases page](https://github.com/izwi-ai/izwi/releases), then:

```bash
sudo apt install ./Izwi_*_amd64.deb
```

### Windows

Download and run `Izwi_<version>_x64-setup.exe` from [GitHub Releases](https://github.com/izwi-ai/izwi/releases).

---

## Verify Installation

After installation, install the `izwi` command line tool from **Settings > System > Command line tool > Install** in the desktop app (the `.deb` installs it for you), then verify everything is working in a new terminal:

```bash
izwi version --full
```

You should see the compiled backend list for the installed CLI. After you start the server with `izwi serve`, use `izwi status --detailed` to verify runtime backend selection.

If you specifically need NVIDIA CUDA support, see:

- [Runtime Support Matrix](/support-matrix)
- [Linux Installation](/installation/linux)
- [Build from Source](/installation/from-source)

### Optional: `espeak-ng` for Kokoro-82M

If you want to use the `Kokoro-82M` TTS model, you also need `espeak-ng` installed separately.
See your platform guide for commands:

- [macOS](/installation/macos#optional-install-espeak-ng-for-kokoro-82m)
- [Linux](/installation/linux#optional-install-espeak-ng-for-kokoro-82m)
- [Windows](/installation/windows#optional-install-espeak-ng-for-kokoro-82m)

---

## Next Steps

- [Getting Started](/getting-started) — Run your first commands
- [Download Models](/models) — Get AI models for inference
