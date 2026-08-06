# ADR 0001: Inference-state ABI v2 owns all physical state

- Status: Accepted
- Date: 2026-08-06
- Applies from: `0.1.0-beta-18`

## Context

The proof-of-concept cache paths mixed semantic model requirements, scheduler
bookkeeping, model-local tensors, and backend kernel details. That made capacity
claims difficult to reconcile with physical storage and allowed execution paths
to disagree about page geometry or provider support.

Stateful model implementations now publish an inference-state ABI v2 contract.
The contract describes domains, groups, shapes, dtypes, attention semantics,
prefix/checkpoint policy, and bounded workspace requirements. It contains no
device pointers, tensors, scheduler tables, or allocation ownership.

## Decision

ABI v2 is the sole physical-state ownership boundary for production runtime
routes. Ownership flows in one direction:

1. The exact loaded model publishes semantic state requirements.
2. The selected backend resolves one immutable physical plan and complete
   operation set for the exact backend, dtype, page size, and semantics.
3. The engine allocates and accounts for retained arenas and invocation
   workspaces before the model becomes ready.
4. The scheduler acquires generation-tagged leases and stages transactional
   updates. Commit publishes state; abort releases it.
5. Backend providers execute validated write, prefill, decode, zero, and copy
   operations over those allocations.
6. Runtime snapshots expose physical bytes, pages, references, operations, and
   lifecycle counters from that same authority.

No model may create a second KV manager or silently fall back to model-owned KV
after managed-state negotiation fails. Stateless execution remains valid only
when the adapter declares that no mutable state survives the invocation.

CPU and Metal share the host/unified-memory resource authority because Apple
unified memory is one physical pool. CUDA keeps a device-scoped authority.
Logical page capacity, physical bytes, invocation workspace, and model weights
must not be charged twice.

## Provider classification

Provider status is attached to the fully resolved plan, not to a backend name:

- `Portable` means the complete operation set is correctness-certified for the
  resolved cell.
- `Optimized` means the same cell has an independently certified faster
  implementation.
- Unsupported or incomplete cells fail model loading.

CPU and Metal paged providers are currently classified Portable. CUDA is
Portable unless the FlashAttention-compatible cell is selected. A CUDA cell is
eligible for Optimized only when the FlashAttention feature/runtime is present,
the dtype is F16/BF16, the page size is a multiple of 32, offsets and attention
semantics are supported (currently zero first-page offsets and full attention),
and equal non-zero K/V head dimensions are divisible by 8 and no larger than
512. Runtime validation remains authoritative.

`IZWI_KV_DISABLE_OPTIMIZED_PROVIDER=1` demotes eligible CUDA cells to Portable.
It never enables an unsupported route.

## Prefix ownership

Prefix reuse is disabled by default. Enabling it requires a non-empty
deployment/tenant namespace. Reuse identity includes the model generation,
adapter and tokenizer semantics, positions, and relevant multimodal inputs.
Committed prefix pages are reference-counted and copy-on-write; active requests
retain their own capacity reserve. Prefix capacity is independently bounded and
may be clamped before startup.

## Compatibility consequences

The stable downstream projections are `EngineConfig`,
`ResolvedKvCachePolicy`, `ManagedKvRuntimeSnapshot`, and the published engine KV
metric names. ABI v2 plans, leases, arenas, transactions, and backend operation
registries are intentionally crate-private.

Old serialized `EngineConfig` documents remain readable because new fields use
safe defaults. Requests for `int8` or `q4` KV remain parseable but fail policy
resolution with an actionable error. A legacy prefix boolean without an
explicit namespace likewise fails resolution. There is no compatibility shim
that fabricates the removed model-owned manager or handle types.

## Operational consequences

- `/v1/health` reports requested and effective cache policy plus any capacity
  clamp reason.
- Runtime telemetry reports physical backing, not a theoretical logical cache.
- Rollback changes provider or prefix policy; it does not restore ABI v1.
- A release candidate is not production-certified merely because a provider is
  compiled. Hardware numerical, model-route, lifecycle, and soak gates remain
  separate requirements.
