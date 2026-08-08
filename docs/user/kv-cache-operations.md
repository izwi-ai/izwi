---
title: "KV Cache Operations"
description: "Configure, observe, benchmark, and safely roll back Izwi's managed inference-state cache."
sidebarTitle: "KV Cache"
icon: "memory-stick"
---

Izwi uses inference-state ABI v2 for physical KV pages and other retained or
invocation-scoped model state. This page covers the operational contract in
`0.1.0-beta-18`; it does not imply that every model/backend combination has
completed production hardware certification.

## Configuration migration

Existing configuration files that omit the new cache fields continue to load.
They resolve to a 64-token page, dense `float16` KV, and prefix reuse disabled:

```json
{
  "kv_cache_dtype": "float16"
}
```

An explicit current configuration looks like:

```json
{
  "kv_cache_dtype": "float16",
  "kv_page_size": 64,
  "enable_prefix_caching": false,
  "managed_prefix_cache_salt": null,
  "max_prefix_cache_pages": 128
}
```

The public `EngineCoreConfig` uses the equivalent field name `block_size`.
Both public configuration surfaces default to 64. Environment-based backend
selection is resolved separately and does not change serialized defaults.

`float16`, `bfloat16`, and `float32` are accepted dense KV requests. `int8` and
`q4` still deserialize so an old file produces a useful startup error, but they
are rejected before model readiness. They are not silently stored as dense KV.

### Rust API migration

`EngineConfig` remains available from the crate root for the beta-17
compatibility window. New integrations should import cache policy types from
their owning module and resolve the policy before announcing readiness:

```rust
use izwi_core::config::{EngineConfig, PrefixCachePolicy};

let config = EngineConfig::default();
let policy = config
    .resolved_kv_cache_policy(1024)
    .expect("valid KV cache policy");
assert!(matches!(policy.effective.prefix, PrefixCachePolicy::Disabled));
```

Use `ManagedKvRuntimeSnapshot` for downstream cache observability. ABI-v2
plans, arenas, leases, transactions, and operation registries are internal
ownership types and are not replacements for removed proof-of-concept manager
handles. The additive `runtime.kv_cache_policy` health object is the supported
HTTP migration surface.

## Requested and effective policy

Inspect `runtime.kv_cache_policy` in `GET /v1/health`. The response separates
what was requested from what the runtime enforces:

```json
{
  "requested": {
    "page_size": 64,
    "dtype": "float16",
    "prefix": { "mode": "namespaced", "max_pages": 128 }
  },
  "effective": {
    "page_size": 64,
    "dtype": "float16",
    "prefix": { "mode": "namespaced", "max_pages": 64 }
  },
  "fallback_reason": "prefix page budget clamped from 128 to 64 to reserve 64 request pages"
}
```

The raw namespace is deliberately omitted from health output. A non-null
`fallback_reason` is an expected, explicit capacity adjustment—not an implicit
provider fallback. If policy resolution cannot preserve at least one
maximum-length request, startup fails.

## Prefix isolation and capacity

Prefix reuse starts disabled. To enable it, set all three fields:

```json
{
  "enable_prefix_caching": true,
  "managed_prefix_cache_salt": "production/tenant-42",
  "max_prefix_cache_pages": 128
}
```

Use a stable, non-secret namespace that changes whenever tenants or deployments
must not share cache entries. Never reuse one namespace across mutually
untrusted tenants. Rotate it after tokenizer, adapter, prompt-template, position,
or multimodal preprocessing changes if the model generation identity does not
already change.

Capacity is measured in physical pages. The effective prefix bound is the
smaller of `max_prefix_cache_pages` and the pages left after reserving one
`max_sequence_length` request. Watch these runtime fields:

- `counters.prefix_hits`, `prefix_misses`, `prefix_evictions`,
  `prefix_copy_on_write_pages`, `prefix_rejections`, and
  `prefix_retained_pages`;
- `totals.coordinator.allocated_pages`, `free_pages`, `prefix_refs`,
  `execution_pins`, and `active_transactions`;
- `totals.physical_bytes` and `memory_accounting`, which should read
  `physical_arena_backing`.

High rejection or eviction rates with low reuse mean the prefix budget or
workload grouping should be reconsidered. Monotonic retained pages after
requests drain indicate a lifecycle defect; disable prefix reuse and investigate
before expanding capacity.

## CUDA build and provider controls

The feature names differ slightly by crate:

| Build target | Feature | Meaning |
|---|---|---|
| `izwi-core` | `cuda` | Candle CUDA plus Izwi native paged operations |
| `izwi-core` | `flash-attn` | `cuda` plus Candle FlashAttention |
| `izwi-core` | `cudnn` | CUDA plus cuDNN support |
| `izwi-server` / `izwi-cli` | `cuda-base` | Native CUDA without the product FlashAttention bundle |
| `izwi-server` / `izwi-cli` | `cuda` | Product CUDA build, including FlashAttention |
| `izwi-server` / `izwi-cli` | `cudnn` | Product CUDA plus cuDNN |

Examples:

```bash
cargo build --release -p izwi-server --features cuda,cudnn
docker compose --profile cuda up
```

CUDA compilation does not guarantee that an optimized provider will be used.
Eligibility is resolved for the exact dtype, page size, attention pattern, head
geometry, offset, build, device, and model-capability route. Inspect health and
KV operation telemetry rather than inferring provider choice from the binary.

CUDA-native optimizations require no promotion flag. On a supported observed
device the runtime automatically uses admission-grown KV slabs, resident decode
metadata, shape/device-keyed Flash or native attention, batched page mutations,
bounded device-side sampling, VRAM-tiered continuous batching, and stable
one-pass decode graph buckets. A graph bucket is bound to the exact K/V and
metadata tensor owners and the arena backing generation; arena growth creates a
new generation. Capture or replay failure enters a bounded negative-cache
backoff before capture is retried, while the request uses the same eager native
kernel. Warmup, capture, replay, fallback, and backoff counts are exported with
managed KV operation telemetry. Request cancellation is safe because replay
mutates only graph-owned query/output scratch, never KV state. CPU and Metal
behavior is unchanged.

The source also contains authoritative FP8 E4M3 CUDA KV pages and mixed
F16/BF16-query FP8-KV prefill/decode kernels. That path has no environment
promotion switch. Its reviewed hardware/shape certification table is empty
until NVIDIA numerical, quality, VRAM, and latency evidence is accepted, so all
shipping routes continue to allocate dense KV.

To force the Portable provider for an incident or comparison run:

```bash
IZWI_KV_DISABLE_OPTIMIZED_PROVIDER=1 izwi serve --backend cuda
```

Boolean values are strict: `1/true/yes/on` disables Optimized, while
`0/false/no/off` enables normal promotion. Invalid values fail startup.

## Benchmark methodology

The checked-in microbenchmark uses the public physical arena ABI and emits
JSON Lines. It measures synchronized operation latency; it is not end-to-end
model throughput.

```bash
scripts/bench/run_kv_cache_matrix.sh \
  --lane default \
  --warmup 5 \
  --iterations 30 \
  --long-context 8192 \
  --output artifacts/kv/candidate.jsonl
```

For release decisions, run five independent baselines and five candidate runs
on dedicated CPU, Apple Silicon, and NVIDIA machines. Record the exact Git SHA,
Cargo.lock checksum, OS, CPU/GPU, driver/toolkit, power mode, model revisions,
and command line. Retain raw JSONL, time series, and logs. Compare numerical
output, provider identity, dispatch/upload/allocation/synchronization counts,
RSS/VRAM, p50, and p95—not latency alone.

The benchmark executable accepts page sizes 16/32/64, F32/F16/BF16,
provider expectations, offsets, windows, softcaps, head geometry, batch size,
and context length. It compares prefill/decode output with the CPU oracle and
reads back write/copy/zero mutations. The matrix runner currently schedules
only page sizes 16/32 and ragged/long profiles, and emits `unsupported` records
for unavailable hardware. Run the additional executable axes explicitly until
the matrix runner covers them. Model-route parity and release soaks remain
separate gates; do not substitute this microbenchmark for them.

No repository benchmark artifact currently establishes a universal hard
latency target. Provider promotion must remain tied to reviewed results for the
exact hardware and shape cell.

## Rollback runbook

Rollback is policy-only; ABI v1 and model-owned KV are not available.

1. Disable prefix reuse with `enable_prefix_caching: false`. Restart and confirm
   health reports `prefix.mode: disabled`.
2. On CUDA, set `IZWI_KV_DISABLE_OPTIMIZED_PROVIDER=1`. Restart and confirm the
   exact route selects Portable rather than Optimized.
3. If the CUDA backend itself is suspect, explicitly select CPU or Metal and
   confirm `requested_backend_available` and `selected_backend` in health.
4. Drain or restart the process before changing page size or dtype; never mix
   old physical allocations with a new policy.
5. Preserve the failing request, health response, runtime KV snapshot, metrics,
   logs, candidate SHA, model revision, and hardware manifest.

If the Portable provider is not certified for the exact route, model loading
fails. Do not bypass that failure with a model-local cache.

## Known unsupported or uncertified areas

- Configurable `int8` and `q4` physical KV storage and attention kernels.
- FP8 E4M3 CUDA KV promotion; the implementation is source-complete but no
  hardware/shape cell is certified yet, so dense KV remains authoritative.
- Cross-namespace or cross-tenant prefix sharing.
- Host-offloaded/tiered KV storage and distributed/multi-node cache ownership.
- Treating native release binaries for Linux or Windows as CUDA builds; those
  published artifacts remain CPU-only.
- Claiming CUDA device correctness from compile/link CI alone.
- Claiming a provider Optimized without numerical and measured performance
  certification for the exact route.

See the [Runtime Support Matrix](/support-matrix) for deployment support and the
[ABI v2 ADR](https://github.com/izwi-ai/izwi/blob/main/docs/dev/adr/0001-inference-state-abi-v2.md)
for the ownership decision.
