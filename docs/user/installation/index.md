# Installation

Choose your platform for detailed installation instructions.

Before choosing an install path, check the [Runtime Support Matrix](../support-matrix.md) for the current backend contract. In particular:

- GitHub Release artifacts are not the same thing as CUDA source builds.
- NVIDIA support should currently be treated as a Linux source-build-first capability.
- The Docker CUDA profile targets NVIDIA Linux hosts and may require `CUDA_COMPUTE_CAP` when built on a machine without `nvidia-smi`.

---

## Platforms

| Platform | Guide |
|----------|-------|
| **macOS** | [macOS Installation](./macos.md) |
| **Linux** | [Linux Installation](./linux.md) |
| **Windows** | [Windows Installation](./windows.md) |
| **From Source** | [Build from Source](./from-source.md) |

---

## Quick Install

### macOS

```bash
# Download and install the .dmg from GitHub Releases
open https://github.com/izwi-ai/izwi/releases
```

### Linux (Debian/Ubuntu)

```bash
# Download the .deb package
wget https://github.com/izwi-ai/izwi/releases/latest/download/izwi_amd64.deb
sudo dpkg -i izwi_amd64.deb
```

### Windows

Download and run the installer from [GitHub Releases](https://github.com/izwi-ai/izwi/releases).

---

## Verify Installation

After installation, verify everything is working:

```bash
izwi version --full
```

You should see the compiled backend list for the installed CLI. After you start the server with `izwi serve`, use `izwi status --detailed` to verify runtime backend selection.

If you specifically need NVIDIA CUDA support, see:

- [Runtime Support Matrix](../support-matrix.md)
- [Linux Installation](./linux.md)
- [Build from Source](./from-source.md)

### Optional: `espeak-ng` for Kokoro-82M

If you want to use the `Kokoro-82M` TTS model, you also need `espeak-ng` installed separately.
See your platform guide for commands:

- [macOS](./macos.md#optional-install-espeak-ng-for-kokoro-82m)
- [Linux](./linux.md#optional-install-espeak-ng-for-kokoro-82m)
- [Windows](./windows.md#optional-install-espeak-ng-for-kokoro-82m)

---

## Next Steps

- [Getting Started](../getting-started.md) — Run your first commands
- [Download Models](../models/index.md) — Get AI models for inference
