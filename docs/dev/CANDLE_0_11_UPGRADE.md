# Candle 0.11 Upgrade Notes

Izwi is pinned to Candle `0.11.1` for `candle-core`, `candle-nn`,
`candle-transformers`, `candle-kernels` (exactly), `candle-metal-kernels`, and
`candle-flash-attn`.

## 0.11.1

`0.11.1` moved Candle's `zip` to 9, clearing the `zip` 8 advisories
(RUSTSEC-2026-0367, -0368, -0369 and -0429) that `.cargo/audit.toml` had
ignored; `tauri-plugin-updater 2.14.1` did the same for its `zip` 4. Despite the
patch version it changes Candle's API:

- `InplaceOp2` and `InplaceOp3` now run through `InplaceOpN<1>` and
  `InplaceOpN<2>`. A source that aliases the destination is refused at run time
  unless the op implements `InplaceOpN` with an access pattern.
- The `ug` kernel compiler and `Device::compile` are gone.
- A tensor's shape must match its buffer exactly: `Tensor::from_vec` with more
  values than a fixed shape holds is now an error (0.11.0 kept the extra).
- With Metal, storage is guarded by `parking_lot`: hold `candle_core::StorageRef`
  rather than naming `std::sync::RwLockReadGuard`.
- CPU SIMD, quantized (a weight repack path), CUDA and Metal backend code,
  layer norm and MoE changed, so outputs can move within each model's
  tolerances.

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
  leaving Metal acceleration available on macOS 15+.
- Disabling `fp16` exposes a `gemm-f16 0.19.0` debug-codegen bug where its
  unoptimized inline assembly is emitted outside the required `fullfp16`
  target-feature context. The root `Cargo.toml` therefore uses `opt-level = 1`
  for that exact package in the dev profile. Cargo's test profile inherits the
  override; release builds are already optimized.
- Do not add a local vendored Candle patch for this. Treat any future removal of
  the Cargo config or profile override as an upstream Candle/toolchain
  compatibility decision, and verify Linux Docker/CI paths against the
  crates.io dependency.
- Published `candle-metal-kernels 0.11.0` constructs a macOS-15-only residency
  descriptor. Izwi keeps the official crates.io package and gates every Metal
  device probe with a runtime availability check: macOS 12-14 select CPU before
  entering Candle, while macOS 15+ retains Metal acceleration.

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
CUDA path uses `cudarc v0.19.10` and `candle-flash-attn v0.11.1`.
