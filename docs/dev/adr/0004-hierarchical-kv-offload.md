# ADR 0004: Hierarchical KV offload is node-local, prefix-scoped, and explicitly opted into

Status: Accepted
Date: 2026-09-26

## Context

DS4 of the distributed-serving plan adds a host tier beneath the device KV
arenas: when device arena pressure rises, committed prefix pages are demoted
to a bounded host pool instead of being dropped, and prefix matches that land
on host-resident pages promote them back before attach. The repository already
reserves the vocabulary for this — `KvStorageTier`/`KvResidencyState`
(`crates/izwi-core/src/kv/residency.rs`) and
`PlacementPolicy::BackendLocalWithHostOffload` — but nothing instantiated
them. The serving plan's standing exclusion of cross-process/cross-node KV
transfer still applies, and the budget invariant DINV-05 requires offloaded
pages to be charged against the same node memory ledger as device pages.

## Decision

1. **Node-local only.** Offload moves page bytes between the device arena and
   a host pool inside one worker process. There is no cross-process or
   cross-node KV transfer, and none will be added behind this policy.
2. **Prefix-index pages only.** Demotion candidates are committed pages held
   by the prefix index and referenced by nothing else (no table refs,
   reservations, or execution pins). Pages belonging to active requests are
   never demoted; capacity pressure against active work continues to resolve
   through the existing preemption behavior. Offload is inert unless the
   deployment uses `PrefixPolicy::CommittedPages`.
3. **Demotion mirrors eviction.** The demotion victim is the LRU index
   entry's subtree — the same unit `evict_lru` removes — so chains stay
   contiguous: a device-resident head followed by a host-resident tail.
4. **Two-phase transfers.** A demotion pins its pages (`transfer_pins`),
   marks them `Offloading` in the residency state machine, copies bytes under
   a bounded in-flight budget, and applies the completion (index removal,
   prefix-ref release, host-chain insertion) at a later manager safe point.
   A failed or stale transfer aborts back to source authority. Pages are
   never labeled host-resident before the destination bytes exist.
5. **The host tier is a separate chain index.** Host-resident prefixes live
   in a host chain index keyed by the same digest chains as the device index.
   The device index's semantics and tests are unchanged. Promotion writes
   host bytes into freshly reserved device pages and re-publishes them
   through the ordinary prefix publication path, so a promoted prefix is
   indistinguishable from a never-demoted one.
6. **Per-backend semantics (DINV-05).** On CUDA, host spill adds real
   capacity across PCIe and is the primary concurrency lever. On Metal
   (unified memory) and CPU there is no additional physical pool: the host
   pool is charged to the shared host/unified ledger, and offload means
   working-set trimming — the expected win is retention of multi-turn
   prefixes and admission headroom, not raw capacity.
7. **Explicit opt-in.** Offload engages only when the operator sets a
   per-assignment host KV pool budget in the supervisor node config, which is
   validated against the assignment's own memory limit and therefore folded
   into the existing aggregate host-memory check. Without a budget (budget 0
   or absent) the feature is fully off. There is no catalog-auto default-on.
8. **Resolved placement stays `BackendLocal`.**
   `BackendLocalWithHostOffload` continues to resolve to backend-local
   placement; the host tier is a runtime layer over it, not a distinct
   physical plan. The `ResolvedPlacement::Host` arm remains reserved.

## Consequences

- Host usage is bounded by an explicit budget and visible through new
  counters (`kv_host_pages`, `demotions_total`, `promotions_total`,
  `promotion_latency`), so operators can size the pool from evidence.
- Promotion happens on the admission path and is strictly cheaper than the
  prefill it avoids; when the device arena cannot take a promoted page, the
  match truncates to its device-resident prefix and the remainder prefills
  cold — reuse degrades, correctness does not.
- CUDA evidence remains hardware-gated (`not run` in the support matrix)
  even though the implementation path is shared with Metal.
