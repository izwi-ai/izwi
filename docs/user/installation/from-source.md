# Build from Source

Build Izwi from source for development, backend-specific installs, or to customize your setup.

See the [Runtime Support Matrix](../support-matrix.md) before choosing a build target. In particular:

- GitHub Release artifacts and source builds do not expose the same backend set.
- Metal is the primary accelerated source-build path on Apple Silicon.
- CUDA source builds are useful for development, custom validation, and fallback debugging. Linux and Windows GitHub Release artifacts are CPU-only; the CUDA distribution path is the Docker CUDA image/profile on NVIDIA Linux hosts.

---

## Prerequisites

### All Platforms

- **Git** — Version control
- **Rust** — 1.83 or later (stable)
- **Node.js** — 18+ (for UI development)
- **`espeak-ng`** (optional, required only for `Kokoro-82M` TTS)

### macOS

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y build-essential curl git pkg-config libssl-dev

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

### Windows

1. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Select "Desktop development with C++"
2. Install [Rust](https://rustup.rs/)
3. Install [Git](https://git-scm.com/download/win)

---

## Clone the Repository

```bash
git clone https://github.com/izwi-ai/izwi.git
cd izwi
```

If you plan to use `Kokoro-82M`, install `espeak-ng` using your platform guide before running TTS:

- [macOS `espeak-ng` install](./macos.md#optional-install-espeak-ng-for-kokoro-82m)
- [Linux `espeak-ng` install](./linux.md#optional-install-espeak-ng-for-kokoro-82m)
- [Windows `espeak-ng` install](./windows.md#optional-install-espeak-ng-for-kokoro-82m)

---

## Build

### Recommended: Use the Install Script

The install script builds the CLI, server, and desktop binaries together and makes the backend choice explicit:

```bash
./scripts/install-cli.sh
```

Backend-specific examples:

```bash
# CPU-focused build
IZWI_BUILD_BACKEND=cpu ./scripts/install-cli.sh

# Apple Silicon / Metal build
IZWI_BUILD_BACKEND=metal ./scripts/install-cli.sh

# NVIDIA CUDA build
IZWI_BUILD_BACKEND=cuda ./scripts/install-cli.sh
```

On Linux, the script defaults to `cpu`. On Apple Silicon macOS, it defaults to `metal`.

### Manual Cargo Builds

If you only want specific binaries, use package-scoped commands:

```bash
# CPU-focused CLI + server
cargo build --release -p izwi-cli
cargo build --release -p izwi-server

# Metal-capable CLI (macOS)
cargo build --release -p izwi-cli --features metal

# CUDA-capable CLI + server
cargo build --release -p izwi-cli --features cuda
cargo build --release -p izwi-server --features cuda
```

For Whisper CUDA experiments, source builds can add Candle-backed CUDA features
such as `flash-attn` or `cudnn`, for example
`cargo build --release -p izwi-server --features cuda,flash-attn`. Only enable
`cudnn` when the matching cuDNN development and runtime libraries are installed.

---

## Install UI Dependencies

The web UI requires Node.js:

```bash
cd ui
npm install
cd ..
```

### Build the UI

**Required for desktop app builds.** The UI must be built before compiling `izwi-desktop`:

```bash
cd ui
npm run build
cd ..
```

---

## Install CLI Tools

### Using the Install Script

```bash
./scripts/install-cli.sh
```

This installs to `~/.local/bin`:
- `izwi` — Main CLI
- `izwi-server` — API server
- `izwi-desktop` — Desktop application

Verify the resulting backend support with:

```bash
izwi version --full
```

After you start the server with `izwi serve`, run `izwi status --detailed` to confirm which backend the runtime actually selected.

### Manual Installation

```bash
# Create directory
mkdir -p ~/.local/bin

# Copy binaries
cp target/release/izwi ~/.local/bin/
cp target/release/izwi-server ~/.local/bin/
cp target/release/izwi-desktop ~/.local/bin/

# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.local/bin:$PATH"
```

---

## Development Mode

### Run Server in Dev Mode

```bash
cargo run --bin izwi-server
```

### Run UI in Dev Mode

In a separate terminal:

```bash
cd ui
npm run dev
```

The dev UI runs at `http://localhost:5173` and proxies API requests to the server.

### Run with Hot Reload

```bash
# Install cargo-watch
cargo install cargo-watch

# Run with auto-reload
cargo watch -x "run --bin izwi-server"
```

---

## Project Structure

```
izwi/
├── crates/
│   ├── izwi-cli/      # CLI application
│   ├── izwi-core/     # Core inference engine
│   ├── izwi-server/   # HTTP API server
│   └── izwi-desktop/  # Tauri desktop app
├── ui/                # React web interface
├── docs/              # Documentation
├── scripts/           # Build and install scripts
└── data/              # Sample data files
```

---

## Running Tests

```bash
# Run all tests
cargo test

# Run specific crate tests
cargo test -p izwi-core

# Run with output
cargo test -- --nocapture
```

---

## Building Release Packages

### macOS DMG

```bash
cd crates/izwi-desktop
cargo tauri build
```

Output: `target/release/bundle/dmg/Izwi_*.dmg`

### Linux DEB

```bash
cd crates/izwi-desktop
cargo tauri build
```

Output: `target/release/bundle/deb/izwi_*.deb`

### Windows Installer

```powershell
cd crates/izwi-desktop
cargo tauri build
```

Output: `target/release/bundle/nsis/Izwi_*-setup.exe`

---

## Troubleshooting

### Rust version too old

```bash
rustup update stable
rustup default stable
```

### Missing OpenSSL (Linux)

```bash
sudo apt install -y libssl-dev pkg-config
```

### Metal not available (macOS)

Ensure you're on Apple Silicon and macOS 12.0+:

```bash
uname -m  # Should show "arm64"
```

### CUDA build fails

Ensure CUDA toolkit is installed and `nvcc` is in PATH:

```bash
nvcc --version
```

### frontendDist path doesn't exist

If you see this error when building:

```
error: proc macro panicked
  --> crates/izwi-desktop/src/main.rs
   |
   |         .build(tauri::generate_context!())
   |                ^^^^^^^^^^^^^^^^^^^^^^^^^^
   |
   = help: message: The `frontendDist` configuration is set to `"../../ui/dist"` but this path doesn't exist
```

Build the UI first:

```bash
cd ui
npm install
npm run build
cd ..
```

Then retry the build.

---

## Next Steps

- [Getting Started](../getting-started.md)
- [CLI Reference](../cli/index.md)
