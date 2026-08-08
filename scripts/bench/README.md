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

## CUDA model evidence

`run-cuda-model-evidence.sh` wraps an existing strict benchmark manifest in a
versioned certification bundle. Unlike compile-only CUDA CI, it requires an
observed NVIDIA device, a CUDA-selected local Izwi server, zero failed quality
gates, telemetry for every case, and loaded-model telemetry reporting
`actual_device_kind=cuda`.

```bash
scripts/bench/run-cuda-model-evidence.sh \
  --manifest benchmarks/manifests/vibevoice-cuda.toml \
  --server http://127.0.0.1:8080 \
  --output target/cuda-model-evidence
```

Missing hardware fails by default. `--allow-unsupported` is only for local
exploration and emits an explicit unsupported certificate; hardware CI must not
use it. The runner never downloads models and rejects remote servers unless
`--allow-remote` is explicit.

## Required NVIDIA CUDA/KV matrix

Before promoting a CUDA provider or adding an FP8 cell to the source-reviewed
certification table, retain both the KV JSONL and model evidence bundle for the
exact Git SHA. At minimum cover:

- `cuda-base`, product `cuda`/FlashAttention, and `cudnn` builds;
- SM 8.0 and newer for graph/partition policy, plus SM 9.0 or newer for FP8;
- F16 and BF16; page sizes 16, 32, and 64; MQA/GQA; equal 64/128/256 head
  dimensions; ragged batches; non-zero first-page offsets; windows and softcap;
- contexts immediately below, at, and above the 2,048-token partition boundary,
  then the loaded model maximum and an admission-overflow rejection;
- first eager call, graph warm/capture/replay, cancellation, arena growth, graph
  generation invalidation, and eager recovery after an injected capture error;
- dense-versus-FP8 logits/output quality, peak VRAM, host reads, dtype/layout
  copies, p50/p95 prefill and decode latency, and continuous-batch throughput.

Every model case must report `actual_device_kind=cuda`, strict quality success,
no worker panic/restart/request-failure delta, and the expected observed
provider. Compile-only CI and an `unsupported` record cannot promote a runtime
cell.
