# KV cache certification benchmark

`run_kv_cache_matrix.sh` exercises the public managed-KV arena ABI without
loading a model. Its default matrix covers:

- default features on CPU, plus Metal/CUDA feature lanes when a compatible
  host device is detected;
- 16-, 32-, and 64-token physical pages;
- F32/F16/BF16 CPU reference cells and backend-supported accelerator dtypes;
- ragged short contexts and configurable long contexts;
- paged prefill and decode compared numerically with the CPU provider;
- readback-validated slot writes, page copy cycles, and page zeroing;
- offset/window/softcap and MQA shape coverage in each backend lane.

Run the supported matrix and save its JSON Lines output:

```bash
scripts/bench/run_kv_cache_matrix.sh --output target/kv-cache-matrix.jsonl
```

Run a single lane or reduce the iteration count during development:

```bash
scripts/bench/run_kv_cache_matrix.sh --lane default --iterations 3 --warmup 1
```

The harness certifies correctness before reporting a timing. It synchronizes
after every measured operation. Reported latency is
therefore dispatch-to-completion latency at the arena boundary, not an
end-to-end model latency or throughput result. JSONL records include the
observed attention provider, correctness error/tolerance, dispatches, resident
plan cache/upload counters, backing allocations, and host synchronizations.
Unavailable workspace, RSS, and VRAM measurements are encoded as JSON `null`;
they are never inferred from the selected feature or page size.

Unsupported accelerator lanes emit `status: "unsupported"` records and are not
compiled or presented as measurements. A visible accelerator can still reject
a case at runtime; that produces `status: "failed"` and a non-zero matrix exit.
The runner deliberately does not infer CUDA availability from feature
compilation alone.

Designated hardware jobs must pass `--require-device`. In that mode a missing
device or any benchmark-level `unsupported` record fails the lane:

```bash
scripts/bench/run_kv_cache_matrix.sh --lane cuda --require-device \
  --iterations 30 --warmup 5 --output target/kv-cache-cuda-certification.jsonl
```

Retained Metal/CUDA evidence must bind a clean worktree to an explicit SHA:

```bash
git_sha=$(git rev-parse HEAD)
scripts/bench/run_kv_cache_matrix.sh --lane metal --require-device \
  --expected-git-sha "$git_sha" --iterations 30 --warmup 5 \
  --output target/metal-kv-evidence/matrix.jsonl \
  --certificate target/metal-kv-evidence/certificate.json
scripts/bench/validate-gpu-evidence-certificate.sh \
  --certificate target/metal-kv-evidence/certificate.json \
  --backend metal --expected-git-sha "$git_sha"

scripts/bench/run_kv_cache_matrix.sh --lane cuda --require-device \
  --expected-git-sha "$git_sha" --iterations 30 --warmup 5 \
  --output target/cuda-kv-evidence/matrix.jsonl \
  --certificate target/cuda-kv-evidence/certificate.json
scripts/bench/validate-gpu-evidence-certificate.sh \
  --certificate target/cuda-kv-evidence/certificate.json \
  --backend cuda --expected-git-sha "$git_sha"
```

The CUDA lane builds with `--features flash-attn` (which implies `cuda`) and
exercises both the portable kill switch and enabled optimized rollout with F16
and BF16. Provider attribution comes from the arena after execution, so native
fallback is visible instead of being guessed from CLI arguments. Treat this as
a long-running production certification lane: the first CUDA/FA2 build can be
substantial and requires a compatible CUDA toolchain as well as a device. This
script does not claim or substitute CUDA measurements on hosts lacking either
prerequisite.

Validate argument routing and capability classification without compiling:

```bash
scripts/bench/test-run-kv-cache-matrix.sh
```

## Backend model evidence

`run-model-evidence.sh` wraps an existing strict benchmark manifest in a
versioned CPU, Metal, or CUDA certification bundle. It requires the requested
backend to be selected by the local Izwi server, zero failed quality gates,
telemetry for every case, and matching `actual_device_kind`. CUDA additionally
requires an observed NVIDIA device. Certification requires the running
server's compile-time Git SHA to match the checked-out CLI/repository SHA.

```bash
scripts/bench/run-model-evidence.sh --backend cuda \
  --manifest benchmarks/manifests/cuda-family-api.toml \
  --server http://127.0.0.1:8080 \
  --output target/cuda-model-evidence
```

Missing hardware fails by default. `--allow-unsupported` is only for local
exploration and emits an explicit unsupported certificate; hardware CI must not
use it. The runner never downloads models and rejects remote servers unless
`--allow-remote` is explicit. The family manifest covers the 17 implementations
reachable through the current chat/TTS/ASR benchmark API. Forced alignment,
diarization, and the standalone speech tokenizer require dedicated benchmark
producers before they can issue equivalent retained runtime certificates.

Use `--require-optimized-kernel-evidence` for a manifest whose every case is
expected to exercise fused attention, paged attention, or a fused RoPE path.
The protected workflow runs this stricter check separately from broad family
coverage so generic Candle CUDA execution is not mislabeled as an optimized
custom kernel.

Continuous batching uses CPU, Metal, and CUDA concurrent manifests covering
Qwen3, Qwen3.5, Qwen3.8, LFM2, and Gemma. The
certificate rejects missing run-local multi-row continuous batches, zero work,
width below two, or physical-batch rejections:

```bash
git_sha=$(git rev-parse HEAD)
backend=cuda # use cpu or metal with the matching manifest
scripts/bench/run-model-evidence.sh --backend "$backend" \
  --manifest "benchmarks/manifests/${backend}-continuous-batching.toml" \
  --require-continuous-batch-evidence \
  --output "target/${backend}-continuous-batching-evidence"
scripts/bench/validate-gpu-evidence-certificate.sh \
  --certificate "target/${backend}-continuous-batching-evidence/certificate.json" \
  --backend "$backend" --expected-git-sha "$git_sha" \
  --require-continuous-batch-evidence
```

Chunked-prefill certification is separate because concurrency alone does not
prove that one prompt crossed a resumable safe point. Start the exact-SHA
server with chunked prefill enabled, then require every model case to commit at
least two scheduler-visible prefill quanta:

```bash
scripts/bench/run-model-evidence.sh --backend "$backend" \
  --manifest "benchmarks/manifests/${backend}-resumable-prefill.toml" \
  --require-resumable-prefill-evidence \
  --output "target/${backend}-resumable-prefill-evidence"
scripts/bench/validate-gpu-evidence-certificate.sh \
  --certificate "target/${backend}-resumable-prefill-evidence/certificate.json" \
  --backend "$backend" --expected-git-sha "$git_sha" \
  --require-resumable-prefill-evidence
```

These certificates establish runtime behavior, not universal performance.
Retain CPU, Apple Silicon, and NVIDIA before/after runs with the same model
revision, prompt matrix, sampling policy, and concurrency. Promotion requires
no quality regression and reviewed TTFT, inter-token latency, throughput,
memory, host-read, metadata-upload, batch-width, and padding deltas for the
exact hardware cell. CUDA-only custom Qwen3.8 kernels remain default-off until
their own NVIDIA profile evidence passes; source compatibility with Qwen3.5
does not qualify them for cross-family promotion.

The ignored native CUDA GQA oracle fails if explicitly run without CUDA or if
the observed provider is not `cuda_native`:

```bash
cargo test -p izwi-core --features cuda \
  backends::kv::accelerator::tests::cuda_paged_decode_matches_cpu_for_offsets_and_gqa \
  -- --ignored --exact
```

The same workflow first runs `run-cuda-model-load-evidence.sh` against one
representative from every registered implementation family. This closes the
load-only coverage gap for forced alignment, diarization, and the standalone
speech tokenizer while keeping their evidence distinct from inference and
kernel certification.

## Required NVIDIA CUDA/KV matrix

Before promoting a CUDA provider, retain both the KV JSONL and model evidence bundle for the
exact Git SHA. At minimum cover:

- `cuda-base`, product `cuda`/FlashAttention, and `cudnn` builds;
- SM 8.0 and newer for graph/partition policy;
- F16 and BF16; page sizes 16, 32, and 64; MQA/GQA; equal 64/128/256 head
  dimensions; ragged batches; non-zero first-page offsets; windows and softcap;
- contexts immediately below, at, and above the 2,048-token partition boundary,
  then the loaded model maximum and an admission-overflow rejection;
- first eager call, graph warm/capture/replay, cancellation, arena growth, graph
  generation invalidation, and eager recovery after an injected capture error;
- dense logits/output quality, peak VRAM, host reads, dtype/layout
  copies, p50/p95 prefill and decode latency, and continuous-batch throughput.

FP8 promotion is a separate blocked project: scaled page storage, scale-aware
mutation/accounting, and numerical evidence must exist before any FP8 lane can
become selectable.

Every model case must report `actual_device_kind=cuda`, strict quality success,
no worker panic/restart/request-failure delta, and the expected observed
provider. Compile-only CI and an `unsupported` record cannot promote a runtime
cell.

## Qwen3.8 CUDA hardware-profile evidence

`run-qwen38-cuda-evidence.sh` imports a versioned Qwen3.8 workload and hardware
profile into the strict CUDA model evidence runner. The included
`benchmarks/manifests/qwen38-l40s-evidence.json` profile covers warmed
single-user decode, longer prompts, a
sustained 2,048-token completion, and concurrency 1/2/4/8. `prompt_words`
controls deterministic input construction; it is not reported as a tokenizer
token count. The certificate retains the actual prompt-token count returned by
the server. The complete operator protocol, candidate matrix, required
artifacts, promotion gates, and deferred phases are in
[`docs/dev/QWEN38_L40S_VALIDATION.md`](../../docs/dev/QWEN38_L40S_VALIDATION.md).

Run the strict L40S convenience profile manually on the exact deployment and
retain the output directory:

```bash
scripts/bench/run-qwen38-l40s-evidence.sh \
  --mtp-depth 1 \
  --server http://127.0.0.1:8080 \
  --izwi-bin target/release/izwi \
  --output target/qwen38-l40s-evidence
```

A pass requires the exact model ID, a selected device matching the manifest's
name, compute capability, minimum VRAM, and driver constraints, a server built
from the checked-out Git SHA, strict quality success, and measured TTFT,
completion throughput, and sampled device-memory use for every case. It records
detailed `nvidia-smi`, OS, imported-manifest, standard CUDA certificate, and raw
benchmark artifacts. The
Qwen3.8 certificate keeps both end-to-end completion throughput and per-sample
decode throughput calculated from the server-reported generation interval; the
two metrics must not be conflated. The workload pins the expected Hugging Face
checkpoint revision; provisioning that
revision remains an operator responsibility because the runner never downloads
or mutates models.

The L40S script is a small strict wrapper around the reusable CUDA runner. For
another NVIDIA GPU, create a separate versioned workload with a distinct
`hardware_profile.id` and run:

```bash
scripts/bench/run-qwen38-cuda-evidence.sh \
  --workload benchmarks/manifests/qwen38-<profile>-evidence.json \
  --mtp-depth 1 \
  --output target/qwen38-<profile>-evidence
```

Each profile owns its performance thresholds. An L40S result is useful for the
`nvidia-l40s-48gb` deployment profile only; it cannot promote a global CUDA
default or set an expected throughput for another GPU. Global promotion needs
representative retained evidence across supported CUDA compute capabilities
and a runtime capability gate that fails closed on unvalidated devices.

The runner measures an existing warmed server; it does not enable optimization
candidates. Set candidate variables on the server process and restart between
cells. `IZWI_QWEN38_PACKED_PROJECTIONS`, `IZWI_QWEN38_CUDA_BF16_KV`,
`IZWI_QWEN38_CAUSAL_CONV_DECODE`, `IZWI_QWEN38_DELTANET_DECODE`, and
`IZWI_QWEN38_DECODE_EPILOGUES` are all default-off and CUDA-unvalidated. Run a
default-off baseline and then one candidate per retained output directory
before testing combinations.

### Qwen3.8 paired MTP evidence

`--mtp-depth` is required for every Qwen3.8 evidence cell. `0` means an
explicitly disabled baseline; `1`, `2`, and `3` mean enabled MTP with that
draft depth. The option does not configure the server. Restart the server for
each cell with matching explicit settings so the runner can compare the
requested policy with loaded-model diagnostics:

Depth 1 is the production default and the first latency candidate to compare.
Deeper cells remain mandatory for a complete profile certificate, but must not
be promoted merely because their accepted length is higher: proposal work,
recurrent-state replay, TTFT, completion TPS, and peak memory all participate
in certification. At concurrency greater than one, scalar fallback is expected
and aggregate throughput—not speculative acceptance—is the deciding metric.

| Cell | Server settings | Runner setting |
|---|---|---|
| Baseline | `IZWI_QWEN38_MTP=0` | `--mtp-depth 0` |
| Depth 1 | `IZWI_QWEN38_MTP=1 IZWI_QWEN38_MTP_DRAFT_TOKENS=1` | `--mtp-depth 1` |
| Depth 2 | `IZWI_QWEN38_MTP=1 IZWI_QWEN38_MTP_DRAFT_TOKENS=2` | `--mtp-depth 2` |
| Depth 3 | `IZWI_QWEN38_MTP=1 IZWI_QWEN38_MTP_DRAFT_TOKENS=3` | `--mtp-depth 3` |

Retain the four directories, then validate the exact pair:

```bash
scripts/bench/certify-qwen38-mtp-evidence.sh \
  --baseline target/qwen38-mtp-disabled \
  --depth-1 target/qwen38-mtp-depth-1 \
  --depth-2 target/qwen38-mtp-depth-2 \
  --depth-3 target/qwen38-mtp-depth-3 \
  --output target/qwen38-mtp-paired
```

The certifier rejects a missing cell or any checkpoint revision, Git SHA,
workload hash, hardware profile, physical device identity, CUDA/compute/KV
provider, sampled-memory, or performance-case mismatch. Its evidence levels
are intentionally monotonic:

- `implemented_unvalidated`: manifests were validated with `--dry-run`; there
  is no device/runtime claim and promotion remains false.
- `runtime_validated`: all four measured cells passed the runtime and pairing
  contract, but profile thresholds are absent or no depth met them.
- `performance_certified`: at least one depth meets every declared per-case
  completion-TPS and TTFT threshold plus the peak-memory threshold. Only those
  depths appear in `certified_depths`. Eligibility remains bound to the exact
  hardware profile/device, checkpoint, and Git SHA in the certificate; it is
  not a global CUDA claim.

To permit the last state, declare these values under
`acceptance.performance_thresholds.values.mtp` before collecting all four
runs: `minimum_completion_tps_p50_speedup_ratio` (strictly greater than `1`),
`maximum_ttft_p95_regression_ratio`, and
`maximum_peak_device_memory_ratio`. These compare candidate completion-TPS p50,
TTFT p95, and sampled peak memory to the disabled baseline. Null thresholds,
including the checked-in L40S manifest today, can establish runtime validation
but cannot synthesize a performance claim.

On a host without an NVIDIA device, the default is a non-zero failure. Local
workflow validation can explicitly record unsupported without inventing data:

```bash
scripts/bench/run-qwen38-l40s-evidence.sh --mtp-depth 1 --allow-unsupported
scripts/bench/test-run-qwen38-cuda-evidence.sh
scripts/bench/test-run-qwen38-l40s-evidence.sh
scripts/bench/test-certify-qwen38-mtp-evidence.sh
```

### Qwen3.8 KV cache precision

The FP8 checkpoint describes weight storage, not KV cache storage. Loaded-model
diagnostics must currently report `cuda_kv_storage.quantized=false` and
`physical_format=dense`: CUDA and Metal use F16 by default, CPU uses F32, and
CUDA has a separately opt-in, unvalidated BF16 candidate. FP8 KV stays
unselectable until the implementation has calibrated per-layer K/V scales,
scale-aware page mutation/accounting, fused paged prefill and decode kernels,
and retained numerical, quality, memory, and latency evidence. A standalone
dequantization pass is not an acceptable promotion path because it can erase
the bandwidth benefit.
