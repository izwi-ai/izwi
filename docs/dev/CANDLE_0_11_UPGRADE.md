# Candle 0.11 Upgrade Notes

Izwi is pinned to Candle `0.11.0` for `candle-core`, `candle-nn`,
`candle-transformers`, `candle-metal-kernels`, and `candle-flash-attn`.

## Candle Dependency Source

Izwi uses Candle directly from crates.io. The root `Cargo.toml` intentionally
does not carry a `[patch.crates-io]` override, and the repository does not
vendor Candle source.

This keeps Docker, CI, local development, and release packaging on the same
dependency source. It also avoids requiring every reduced build context to copy
a local Candle checkout before running `cargo build --locked`.

Apple Silicon stable Rust caveat:

- On the Apple Silicon stable Rust toolchain used during the upgrade, upstream
  `candle-core 0.11.0` can hit unstable `stdarch_neon_f16` code in Candle's
  NEON fp16 specialization.
- The workspace keeps using crates.io Candle and handles this with
  `.cargo/config.toml`, disabling the CPU `fp16` target feature only for
  `aarch64-apple-darwin`. This routes Candle through its stable fallback while
  leaving Metal acceleration available.
- Disabling `fp16` exposes a `gemm-f16 0.19.0` debug-codegen bug where its
  unoptimized inline assembly is emitted outside the required `fullfp16`
  target-feature context. The root `Cargo.toml` therefore uses `opt-level = 1`
  for that exact package in the dev profile. Cargo's test profile inherits the
  override; release builds are already optimized.
- Do not add a local vendored Candle patch for this. Treat any future removal of
  the Cargo config or profile override as an upstream Candle/toolchain
  compatibility decision, and verify Linux Docker/CI paths against the
  crates.io dependency.

## Metal API Migration

Candle `0.11.0` changed Metal command encoding around `CommandsGuard`. Izwi's
custom Metal launch sites now bind buffers through a shared helper that forwards
to the underlying encoder and classifies every buffer as input or output for
Candle hazard tracking.

The migrated launch sites are in:

- `crates/izwi-core/src/kernels/metal.rs`
- `crates/izwi-core/src/models/architectures/kokoro/decoder.rs`
- `crates/izwi-core/src/models/architectures/kokoro/prosody.rs`
- `crates/izwi-core/src/models/architectures/nemotron/asr/metal_kernels.rs`
- `crates/izwi-core/src/models/architectures/parakeet/asr/metal_kernels.rs`

## Verification

Local CPU and Metal verification used serialized jobs to reduce memory pressure:

```sh
cargo check --locked -j 1 -p izwi-core --no-default-features
cargo check --locked -j 1 -p izwi-cli --no-default-features
cargo check --locked -j 1 -p izwi-agent --no-default-features
cargo check --locked -j 1 -p izwi-server
cargo build --release --locked -j 1 -p izwi-cli --features metal
cargo test --locked -j 1 -p izwi-core --no-default-features -- --test-threads=1
cargo test --locked -j 1 -p izwi-core --features metal,accelerate metal -- --test-threads=1
scripts/ci/check-backend-truth.sh cargo-cpu
```

Runtime smoke booted exactly one local server with a temporary DB and probed
`/v1/health`, `/livez`, `/readyz`, `/v1/models`, and
`/v1/metrics/prometheus`.

CUDA and flash-attn compile gates are CI/manual gates on hosts with `nvcc`:

```sh
scripts/ci/check-backend-truth.sh cargo-cuda
IZWI_CUDA_FEATURES=cuda,cudnn,flash-attn scripts/ci/check-backend-truth.sh cargo-cuda
scripts/ci/check-backend-truth.sh docker-cuda
```

On the local macOS machine used for the upgrade, those CUDA compile gates stop at
`Missing required command: nvcc`; dependency resolution still confirms Candle's
CUDA path uses `cudarc v0.19.8` and `candle-flash-attn v0.11.0`.
