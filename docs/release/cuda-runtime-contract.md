# Native CPU + Docker CUDA Release Contract

This contract defines the supported split between public native release
artifacts and CUDA-capable Docker/source builds.

## Decision

Public native artifacts stay CPU-safe:

- Linux and Windows GitHub Release installers and terminal bundles are CPU-only.
- Native Linux and Windows artifacts must not bundle CUDA runtime libraries, CUDA DLLs, or private CUDA-linked binaries.
- Public binary names remain `izwi` / `izwi-server` on Linux and `izwi.exe` / `izwi-server.exe` on Windows.
- macOS release artifacts remain the native accelerated path through Metal on Apple Silicon.
- CUDA-capable binaries are built in CI only for the Docker `production-cuda` target, whose final stage is based on `nvidia/cuda:12.4.1-runtime-ubuntu22.04`.
- Source builds may still enable CUDA with `--features cuda` on compatible NVIDIA hosts.

Do not reintroduce native CUDA runtime bundling unless the release strategy is explicitly reopened and artifact-size, redistribution, and loader-startup risks are resolved first.

## Why Not Native CUDA Bundles?

Izwi can choose CPU or CUDA after process startup, but Candle's CUDA feature path links CUDA shared libraries at loader time. A CUDA-linked native `izwi` or `izwi-server` can fail before `main()` on hosts without CUDA libraries, which prevents the runtime CPU fallback from running.

Bundling CUDA runtime libraries inside native installers also balloons artifact size. The beta 14 native CUDA attempt pushed Linux/Windows assets hundreds of MB larger and caused the `.deb` release asset to exceed GitHub's 2 GiB upload limit.

Docker solves both problems for the CUDA distribution path:

- CUDA libraries come from the NVIDIA runtime image.
- Host driver integration is handled by the NVIDIA container runtime.
- Native release assets return to CPU-only beta 13 scale.

## Artifact Layout

Native GitHub Release artifacts:

- `izwi` / `izwi.exe`: public CPU-only CLI entrypoint.
- `izwi-server` / `izwi-server.exe`: public CPU-only server entrypoint.
- `izwi-desktop` / `izwi-desktop.exe`: public CPU-only desktop shell binary on Linux/Windows.
- Linux `.deb`, Linux AppImage/updater, Linux terminal tarball, Windows NSIS/updater, and Windows terminal zip must not include `runtime/cuda`, CUDA shared libraries, or CUDA DLLs.

Docker CUDA artifact:

- The `rust-builder-cuda` stage builds `izwi-server` with `--features cuda`.
- The `production-cuda` stage copies only that CUDA-capable server binary into the NVIDIA CUDA runtime image.
- The image sets `IZWI_BACKEND=cuda`, `NVIDIA_VISIBLE_DEVICES=all`, and `NVIDIA_DRIVER_CAPABILITIES=compute,utility`.

## Runtime Selection

Native releases:

1. Start the CPU-only public binary.
2. `auto` and `cpu` run on CPU unless a platform-native accelerator such as macOS Metal is available.
3. A CUDA request on a CPU-only native artifact should fail clearly instead of looking for packaged CUDA binaries.

Docker CUDA:

1. Build the `production-cuda` target with `CUDA_COMPUTE_CAP` set for the target GPU architecture when automatic detection is unavailable.
2. Run the CUDA image on an NVIDIA Linux host through Docker Compose profile `cuda` or an equivalent `docker run --gpus` deployment.
3. Health/status surfaces should report CUDA availability and selection from inside the container.

Source CUDA:

1. Install a compatible NVIDIA driver and CUDA toolkit.
2. Build with `cargo build --release --features cuda` or `IZWI_BUILD_BACKEND=cuda ./scripts/install-cli.sh`.
3. Run with `--backend cuda` or `IZWI_BACKEND=cuda`.

## Guardrails

Release workflow guardrails:

- Native release jobs build without CUDA features.
- Native terminal bundles contain only the public binaries.
- Native artifact verification rejects `runtime/cuda`, CUDA shared libraries, and CUDA DLLs.
- Native artifact verification enforces a conservative per-file size cap before upload and again before GitHub release publication.

Docker workflow guardrails:

- CPU and CUDA Docker builds run as separate `Backend Truth` jobs.
- The CUDA Docker job builds `production-cuda`, not a native release package.
- The CPU Docker image is smoke-run with `/usr/local/bin/izwi-server --help`.
- The CUDA Docker image is audited with `ldd`; CUDA runtime libraries must resolve from the image, while `libcuda.so.1` is allowed to remain unresolved because it is supplied by the NVIDIA container runtime on GPU hosts.

## Verification Requirements

Each release candidate must prove:

- Linux and Windows native artifacts install/start on CPU-only hosts.
- Linux and Windows native artifacts do not contain CUDA runtime payloads.
- Native artifact sizes stay comfortably below GitHub's release asset limit.
- The Docker CPU image builds and starts the server binary.
- The Docker CUDA image builds from the NVIDIA CUDA runtime image and exposes a CUDA-linked server binary whose non-driver CUDA libraries resolve inside the image.
- CUDA source builds still compile in the CUDA CI container.

## Non-Forking Rule

Do not copy or fork Izwi core runtime files to create backend-specific source variants. Backend variants must be produced through Cargo features, build profiles, Docker stages, packaging layout, and small launch/runtime diagnostics only.
