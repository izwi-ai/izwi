# Candle 0.11 Upgrade Notes

Izwi is pinned to Candle `0.11.0` for `candle-core`, `candle-nn`,
`candle-transformers`, `candle-metal-kernels`, and `candle-flash-attn`.

## Local Candle Core Patch

The workspace patches `candle-core` to `vendor/candle-core-0.11.0` from
`[patch.crates-io]` in the root `Cargo.toml`.

The patch is intentionally narrow:

- Apple Silicon stable Rust exposes the `fp16` target feature, but Candle
  `0.11.0`'s NEON fp16 specialization reaches unstable `stdarch_neon_f16`.
- Global `RUSTFLAGS='-C target-feature=-fp16'` lets Candle compile, but breaks
  other dependencies such as `gemm-f16`.
- The local patch keeps Candle at version `0.11.0` and routes macOS ARM through
  Candle's existing f32-widening NEON fallback while leaving other platforms on
  the upstream native fp16 path.
- The patch also removes one unused Metal backend import so local warning-free
  checks stay clean.

Remove the patch when an upstream Candle release or `0.11.x` patch compiles on
Apple Silicon stable Rust without global target-feature overrides and without
breaking `gemm-f16`.

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
cargo check --locked -p izwi-core --no-default-features
cargo check --locked -p izwi-cli --no-default-features
cargo check --locked -p izwi-agent --no-default-features
cargo check --locked -p izwi-server
cargo check --locked -p izwi-cli --features metal
cargo test --locked -p izwi-core --no-default-features --jobs 1 -- --test-threads=1
cargo test --locked -p izwi-core --features metal,accelerate --jobs 1 metal -- --test-threads=1
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
