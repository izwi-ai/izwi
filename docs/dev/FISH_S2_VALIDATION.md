# Fish S2 Pro validation

The native Candle implementation targets `fishaudio/s2-pro` revision
`1de9996b6be38b745688de084d87a5633f714e4e`, compared with Fish Speech revision
`befe4001745417f8c42131739d862b8a6fdbd15a`. New downloads use the pinned revision.
Existing local bundles remain usable; compare their configuration and tensor
inventory before treating them as equivalent to the reviewed checkpoint.

| Backend | Default transformer dtype | Codec dtype |
| --- | --- | --- |
| CPU | F32 | F32 |
| Metal | F16 | F32 |
| CUDA with observed SM 8.0+ BF16 support | checkpoint BF16 | F32 |
| CUDA without BF16 support | F32 | F32 |

`IZWI_FISH_S2_DTYPE` accepts `f32`, `f16`, or `bf16` only when the backend
supports that representation. CPU half and Metal BF16 are rejected before
neural weights are allocated. An explicit supported CUDA F16 override trades
BF16's exponent range for a different representation; compare audio before
using it in deployment.

## Portable regression checks

```sh
cargo test --locked -p izwi-core --features accelerate --lib fish_s2
cargo test --locked -p izwi-cli -p izwi-server --features izwi-core/accelerate fish_s2
```

Use the platform's supported BLAS feature where Accelerate is unavailable.
These checks cover small synthetic tensors, reference mathematical fixtures,
resource arithmetic and serving contracts. They do not measure speech quality
or certify whole-model accelerator performance.

The metadata-only gate can check an installed bundle without allocating its neural
weights:

```sh
IZWI_FISH_S2_MODEL_DIR='/path/to/models/FishAudio-S2-Pro' \
cargo test --locked -p izwi-core --features accelerate --lib \
  fish_s2_real_artifact_metadata_contracts -- --ignored --nocapture
```

Explicit small device probes are also available under the `metal`/`cuda`
features: `metal_rotation_matches_frozen_upstream_oracles`,
`metal_compact_head_matches_full_projection`,
`metal_sampling_matches_cpu_policy_and_rejects_raw_nan`,
`metal_codec_attention_matches_dense_oracle`, and their `cuda_` counterparts.
Run them with `--ignored --nocapture`; device creation must succeed. A missing
device is an error, never an implicit CPU pass.

## Real model through the runtime

The ignored smoke test uses `RuntimeService`, including reference preparation,
retained slow/fast state, scheduled generation and waveform finalization. It
requires enough memory for weights, caches, workspace and the operating system.
Run one loaded model at a time on memory-constrained hosts.

```sh
IZWI_FISH_S2_MODEL_DIR='/path/to/models/FishAudio-S2-Pro' \
IZWI_FISH_S2_BACKEND=cuda \
IZWI_FISH_S2_SMOKE_OUTPUT_WAV=/tmp/fish-s2-cuda.wav \
cargo test --locked -p izwi-core --features cuda,cudnn,flash-attn \
  --lib fish_s2_real_model_smoke_generates_finite_audio -- --ignored --nocapture
```

For Metal, select `metal` for both the backend and build feature; for CPU select
`cpu` and the appropriate BLAS build feature. CUDA product builds using the CLI
or server `cuda` feature already include Candle FlashAttention. A core-only
`cuda` build needs the explicit `flash-attn` feature to include that provider.

The default conditioning pair is `data/fox.wav` and its exact transcript in
`data/fox.md`; the initial target text is also `data/fox.md`. Then repeat with
`IZWI_FISH_S2_TARGET_TEXT` set to a held-out sentence. Custom reference audio
requires `IZWI_FISH_S2_REFERENCE_WAV` and its matching
`IZWI_FISH_S2_REFERENCE_TEXT`. The test defaults to 512 output frames and an
explicit 4096-token context; override `IZWI_FISH_S2_SMOKE_MAX_FRAMES` and
`IZWI_FISH_S2_SMOKE_CONTEXT` for longer probes. These test bounds do not change
production's native-context policy.

Retain the output WAV, exact source commit, model revision, build features,
dtype, device/driver/CUDA/cuDNN versions and logged timings. Inspect intelligibility,
reference voice similarity, duration, clipping, repeated speech, EOS behavior
and non-finite values. Use listening plus transcription against the target;
finite and non-silent output alone is insufficient.

## Backend performance evidence

Use `benchmarks/manifests/cuda-family-api.toml` for family coverage and the
`{cpu,metal,cuda}-audio-concurrency.toml` manifests for concurrency 1, 2, 4 and 8.
Every Fish case provides the same matching reference voice/transcript. Fish
currently returns a final waveform; a streaming-compatible response does not
establish incremental codec streaming or native tensor batching.

Measure warmed and cold preparation, prefill, semantic/codebook decode and codec
finalization separately. Record time to first audio, real-time factor, peak
host/device memory and cancellation latency for short and long reference/target
pairs, including unspaced multilingual text. Verify the actual selected Candle
attention provider on the deployed GPU. Compare numerics and audio before
accepting a faster dtype, convolution algorithm or attention provider.

The codec currently keeps F32 compute on all backends. CUDA convolution uses
Candle's cuDNN `ImplicitGemm` algorithm to keep scratch bounded; faster algorithm
selection requires measured workspace limits and an audio-parity comparison on
the target GPU. This is a memory-safe baseline, not a claim of the fastest
cuDNN algorithm. The slow decoder retains the existing Candle paged-attention
routing, including FlashAttention when the build/device/dtype are supported.
Reference input rates are bounded at 384 kHz and prepared audio at 4096 codec
frames; out-of-contract inputs fail before neural codec allocation.

## Verification recorded on 2026-09-05

On the local macOS arm64 host:

- CPU/Accelerate Fish suite: 84 passed; three real-artifact tests excluded from
  the ordinary run. The metadata-only artifact test was also run and passed.
- Full core suite: 2,404 passed, eight ignored, three pre-existing failures:
  `global_fused_kernel_availability_tracks_known_backends`,
  `predictor_batch_matches_scalar_codebooks_and_isolates_receipts`, and
  `repeated_mrope_positions_match_standard_rope_when_axes_equal`. These match
  the baseline audit; they are outside the Fish changes.
- CLI: all 63 tests passed. Server: all 390 tests passed.
- CPU and Metal feature builds compiled. Actual Metal device probes passed for
  rotary fixtures, compact-vs-full projection, sampling with dense and sparse
  masks, raw-NaN rejection, and blocked codec attention against the dense oracle.
  GPU probes need device access; the restricted sandbox hides Metal devices.

No whole-model speech generation, CUDA build, CUDA device execution or end-to-end
CUDA performance measurement was performed. This 16 GiB host cannot comfortably
hold the F32 model plus its inference state. Use the protocol above on an
appropriately sized device before certifying audible quality, peak memory or
throughput. No custom kernel or dependency upgrade was introduced.
