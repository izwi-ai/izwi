# CPU + CUDA Release Runtime Contract

This contract defines how Linux and Windows release installers should support both CPU and CUDA without forking or vendoring Izwi core source files.

## Decision

Use a **single installer with shared-source CPU and CUDA binary variants**:

- Default binaries remain CPU-safe and must start on machines with no NVIDIA driver or CUDA toolkit.
- CUDA-capable sidecar binaries are built from the same crates with `--features cuda`.
- Launch and packaging logic may select a CUDA sidecar only after CUDA runtime dependencies are loadable.
- If CUDA dependencies are missing or unusable, the default CPU binary path remains valid.

Do not replace the CPU-safe Linux/Windows release binaries with CUDA-linked binaries unless Phase 1 loader checks prove they can start without CUDA shared libraries. Current research indicates they cannot.

## Why Not One CUDA-Linked Default Binary?

Izwi can already select CPU or CUDA after process startup, but Candle's current CUDA feature path links CUDA shared libraries at loader time. A CUDA-linked `izwi` or `izwi-server` can therefore fail before `main()` on hosts without CUDA libraries, which prevents Izwi's runtime fallback from running.

The sidecar model keeps the user-facing installer unified while avoiding that startup failure mode.

## Binary Layout

The release package should contain:

- `izwi` / `izwi.exe`: CPU-safe default CLI.
- `izwi-server` / `izwi-server.exe`: CPU-safe default server.
- `izwi-cuda` / `izwi-cuda.exe`: CUDA-capable CLI sidecar, when direct CUDA CLI support is needed.
- `izwi-server-cuda` / `izwi-server-cuda.exe`: CUDA-capable server sidecar.
- A private CUDA runtime library directory, when redistribution is enabled:
  - Linux: `lib/izwi/cuda/`
  - Windows: colocated with CUDA sidecar `.exe` files or under `bin/cuda/`

All binaries are built from the same source tree. The only difference is Cargo feature selection and output name/location.

## Runtime Selection

Default behavior:

1. Start the CPU-safe default binary.
2. If backend preference is `cpu`, stay on CPU.
3. If backend preference is `auto`, a launcher may select CUDA only if:
   - CUDA sidecar binary exists.
   - Required CUDA runtime libraries are loadable.
   - NVIDIA driver probing indicates a usable CUDA device.
4. If backend preference is `cuda`, fail with a clear diagnostic when the CUDA sidecar or runtime dependencies are unavailable.

The runtime health surface should distinguish:

- CUDA sidecar packaged.
- CUDA runtime libraries loadable.
- NVIDIA driver present.
- CUDA device usable.
- selected backend.

## CUDA Library Policy

The required library set must be derived from loader audit output, not guessed. For the current Candle CUDA path, expect these families:

Linux:

- `libcuda.so.1` or compatible driver library from the host NVIDIA driver.
- `libcudart.so*`
- `libcublas.so*`
- `libcublasLt.so*`
- `libcurand.so*`
- `libnvrtc.so*`
- `libnvrtc-builtins.so*`

Windows:

- `nvcuda.dll` from the host NVIDIA driver.
- `cudart64_*.dll`
- `cublas64_*.dll`
- `cublasLt64_*.dll`
- `curand64_*.dll`
- `nvrtc64_*.dll`
- `nvrtc-builtins64_*.dll`

The NVIDIA CUDA EULA lists CUDA Toolkit redistributable files in Attachment A. As of the January 26, 2026 EULA, `cudart`, `cublas`, `cublasLt`, `curand`, `nvrtc`, and `nvrtc-builtins` are listed as redistributable families. Driver components such as `nvcuda.dll` / `libcuda.so.1` should be treated as host driver dependencies unless legal review approves any other approach.

References:

- NVIDIA CUDA EULA: `https://docs.nvidia.com/cuda/eula/index.html`
- CUDA compatibility documentation: `https://docs.nvidia.com/deploy/cuda-compatibility/`

This file is an engineering packaging contract, not legal advice. Verify redistribution terms before publishing CUDA runtime libraries.

## Verification Requirements

Each release candidate must prove:

- CPU-only host with no CUDA libraries: default `izwi` and `izwi-server` start and report CPU fallback.
- Host with CUDA libraries but no usable GPU: default binaries start and explain why CUDA was not selected.
- NVIDIA GPU host: CUDA sidecar can start and `auto`/`cuda` select CUDA as expected.
- Linux `.deb`, Linux AppImage/updater, Linux terminal tarball, Windows NSIS/updater, and Windows zip all expose the same CPU/CUDA packaging contract.

## Non-Forking Rule

Do not copy or fork Izwi core runtime files to create backend-specific source variants. Backend variants must be produced through Cargo features, build profiles, packaging layout, and small launch/runtime diagnostics only.
