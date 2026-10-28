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
  "max_prefix_cache_pages": null
}
```

A `null` (or omitted) `max_prefix_cache_pages` is automatic; see
[Prefix isolation and capacity](#prefix-isolation-and-capacity).

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

The policy's `dtype` is the configured default. Each model's state contract
picks the storage its kernels support, so a loaded model can store its KV in
another dtype. For example, LFM2.5 chat stores F16 KV on Metal and CUDA and
F32 on CPU. The storage a model actually uses is reported per loaded model in
`runtime.loaded_models[].kv_storage`:

```json
"kv_storage": [{ "domain": 1, "page_tokens": 64, "dtype": "f16" }]
```

## Prefix isolation and capacity

The server engages prefix reuse in catalog-auto mode by default: a model
shares prefixes on the backends where its catalog cell is supported (see the
[support matrix](/support-matrix)), under a per-process namespace unless you
set one. `IZWI_ENABLE_PREFIX_CACHING=0` (or `IZWI_PREFIX_REUSE_AUTO=0`) turns it
off for every model. To enable it explicitly for every model instead, set the
first two fields; the page budget is optional:

```json
{
  "enable_prefix_caching": true,
  "managed_prefix_cache_salt": "production/tenant-42",
  "max_prefix_cache_pages": 4096
}
```

Server deployments can set the same policy without touching engine internals:
config file `[runtime]` keys (`enable_prefix_caching`,
`managed_prefix_cache_salt`, `max_prefix_cache_pages`) or environment variables
(`IZWI_ENABLE_PREFIX_CACHING`, `IZWI_MANAGED_PREFIX_CACHE_SALT`,
`IZWI_MAX_PREFIX_CACHE_PAGES`). Note that a model must also opt in through its
cache contract: dense families (Qwen3, Gemma3, Voxtral, VibeVoice) participate,
and so do the hybrid chat families Qwen3.8, Qwen3.6, LFM2.5 and LFM3, which
snapshot their recurrent and convolution state at each chat turn's history
boundary (see [Hybrid prefix snapshots](#hybrid-prefix-snapshots)). Qwen3.5,
LFM2.5 Audio and LFM3 Audio keep their hybrid state private. `IZWI_QWEN38_PREFIX_REUSE=0` turns reuse off for
Qwen3.8 alone, and a Qwen3.8 load with its MTP head (`IZWI_QWEN38_MTP=1` on
CUDA, the default elsewhere) keeps every domain private.

Use a stable, non-secret namespace that changes whenever tenants or deployments
must not share cache entries. Never reuse one namespace across mutually
untrusted tenants. Rotate it after tokenizer, adapter, prompt-template, position,
or multimodal preprocessing changes if the model generation identity does not
already change.

Capacity is measured in physical pages. The effective prefix bound is the
smaller of `max_prefix_cache_pages` and the pages left after reserving one
`max_sequence_length` request. A chat turn can only attach history whose
whole page chain is still indexed, so a conversation longer than the page
budget stops reusing its history.

When `max_prefix_cache_pages` is not set, the budget is automatic: 128 pages
of 64 tokens (8,192 tokens) on CPU and Metal, and per model on CUDA. A model
whose CUDA KV arena is allocated in full at load (Qwen3.8 and Qwen3.6) gets a
quarter of that arena, never less than 128 pages: about 2,400 pages (over
150,000 tokens of history) for Qwen3.8 on an 80 GB H100. Those pages occupy
memory that is already allocated, and a request that needs pages held only
by the prefix index evicts them, least recently used first, so the larger
budget never holds a request back. Other CUDA models keep 128 pages.

An explicit value, from the config file or `IZWI_MAX_PREFIX_CACHE_PAGES`,
applies to every model as set (startup rejects `0` while prefix reuse is on;
turn reuse off with `IZWI_ENABLE_PREFIX_CACHING=0` instead). CUDA keeps it
unclamped, for example `IZWI_MAX_PREFIX_CACHE_PAGES=4096` (one 262,144-token
conversation). CPU and Metal clamp it to the 1,024-page baseline minus one
maximum-length request. Config files that `izwi config edit` created before
the automatic budget existed contain `max_prefix_cache_pages = 128`; delete
that line to use the automatic budget.

`runtime.kv_cache_policy` reports the configured budget, or the 128-page
floor when it is automatic; each model's effective budget is
`models[].arenas[].prefix_capacity_pages` in the runtime KV snapshot. Watch
these runtime fields:

- `models[].arenas[].prefix_capacity_pages`, the effective prefix page
  budget of each arena;
- `counters.prefix_hits`, `prefix_misses`, `prefix_evictions`,
  `prefix_copy_on_write_pages`, `prefix_rejections`, and
  `prefix_retained_pages`. Hits and misses count one lookup per request
  preparation (admission probes are not counted), across the device and host
  tiers;
- `counters.prefix_query_tokens` and `prefix_hit_tokens`, the prompt tokens
  offered to prefix lookups and the tokens served from the cache. Their ratio
  is the token hit rate;
- `totals.coordinator.allocated_pages`, `free_pages`, `prefix_refs`,
  `execution_pins`, and `active_transactions`;
- `totals.physical_bytes` and `memory_accounting`, which should read
  `physical_arena_backing`.

The same signals are on `/metrics`:

- `izwi_engine_kv_cache_prefix_query_tokens_total` and
  `izwi_engine_kv_cache_prefix_hit_tokens_total`.
- Per arena, with `model`, `backend`, `domain` and `storage_format` labels:
  `izwi_engine_kv_cache_model_capacity_pages`, `model_allocated_pages`,
  `model_prefix_retained_pages`, `model_admission_claimed_pages` and
  `model_admission_available_pages`. Allocated pages include prefix-only
  pages that admission can evict. Admission is bound by available pages, not
  free pages.
- `izwi_engine_scheduler_preemptions_total{kind}`: `replay` (a published
  request suspended and replayed later), `recompute` (an unpublished request
  restarted) and `abort` (the youngest request failed because every claim
  holder was waiting to grow).
- `izwi_engine_scheduler_kv_blocked_requests`, the requests the last engine
  step deferred for KV capacity.

High rejection or eviction rates with low reuse mean the prefix budget or
workload grouping should be reconsidered. Monotonic retained pages after
requests drain indicate a lifecycle defect; disable prefix reuse and investigate
before expanding capacity.

### Hybrid prefix snapshots

Hybrid models (recurrent or convolution state beside attention, such as
Qwen3.6, Qwen3.8, LFM2.5 and LFM3) can attach a shared prefix only where a
committed snapshot of that state exists. Each snapshot is one row of the
model's recurrent state (156,893,184 bytes for Qwen3.8-27B; the LFM ShortConv
rings are small, 745,472 bytes for LFM3-1.7B and 245,760 bytes for
LFM2.5-1.2B) and outlives the request that published it, so snapshots have
their own budget:

- `IZWI_MAX_PREFIX_SNAPSHOTS` (engine field `max_prefix_snapshots`, default
  8) bounds the snapshots one model may hold. `0` keeps hybrid state private,
  so hybrid prefixes never attach.
- The budget is fitted and reserved at load beside the state rows: on CUDA it
  comes out of the automatic rows' state share (`IZWI_CUDA_STATE_BUDGET_FRACTION`),
  so automatic retained rows shrink by the snapshot count and KV is
  unchanged; explicit rows keep their count and the KV reach pays instead.
- A held snapshot is recorded as materialized on the model lease (each
  allocation once, also when a live session shares it); eviction returns its
  bytes to the pending claim before they are freed.

Watch `models[].prefix_snapshot_capacity`, `prefix_snapshot_budget_bytes`,
`prefix_snapshots`, `prefix_snapshot_bytes` and `materialized_tensor_bytes`,
plus the `tensor_snapshot_publishes`, `tensor_snapshot_attaches`,
`tensor_snapshot_truncations` and `tensor_snapshot_evictions` counters.
`prefix_snapshot_bytes` never exceeds the budget.

On CPU and Metal a prompt attaches a prefix only through resumable prefill,
which is off by default there: set `IZWI_ENABLE_CHUNKED_PREFILL=1` for chat
history reuse on those backends. CUDA turns it on with its incremental chat
policy.

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
docker compose --profile cuda up -d izwi-cuda
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

The source contains experimental FP8 E4M3 CUDA KV kernels, but they are not a
complete storage format: scale storage, scale-aware mutation/accounting, and
NVIDIA numerical evidence are still required. Both arena selection and direct
prefill/decode dispatch therefore fail closed; all shipping routes allocate
dense KV and no environment switch can enable the incomplete path.

To force the Portable provider for an incident or comparison run:

```bash
IZWI_KV_DISABLE_OPTIMIZED_PROVIDER=1 izwi serve --backend cuda
```

Boolean values are strict: `1/true/yes/on` disables Optimized, while
`0/false/no/off` enables normal promotion. Invalid values fail startup.

On Metal, KV writes, page zeroes, and page copies are queued without a host
wait. The first wait on any of their fences, normally the decode step's batch
seal, drains the device once for every mutation queued before it. Managed KV
telemetry then reports about one `host_synchronizations` per decode step
instead of one per attention-layer write. To restore one device
synchronization per mutation for an incident or comparison run:

```bash
IZWI_KV_METAL_EAGER_FENCES=1 izwi serve --backend metal
```

It takes the same strict boolean values.

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

1. Disable prefix reuse with `enable_prefix_caching: false` (environment:
   `IZWI_ENABLE_PREFIX_CACHING=0`; for Qwen3.8 alone,
   `IZWI_QWEN38_PREFIX_REUSE=0`). Restart and confirm health reports
   `prefix.mode: disabled`, or the model's `prefix_reuse.engaged: false`.
2. On CUDA, set `IZWI_KV_DISABLE_OPTIMIZED_PROVIDER=1`. Restart and confirm the
   exact route selects Portable rather than Optimized. On Metal, set
   `IZWI_KV_METAL_EAGER_FENCES=1` to synchronize after every KV mutation again.
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
