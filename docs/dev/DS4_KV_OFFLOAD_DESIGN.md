# DS4: hierarchical KV offload — design note

Status: Accepted (2026-09-26) · Decisions pinned in
[ADR 0004](adr/0004-hierarchical-kv-offload.md) · Plan items DS4.1–DS4.5
(`PRODUCTION_DISTRIBUTED_SERVING_PLAN.md`).

This note specifies the mechanics of page-granular demotion/promotion between
the device KV arenas and a bounded host pool. It instantiates the reserved
residency scaffolding (`crates/izwi-core/src/kv/residency.rs`) and the
previously inert `PlacementPolicy::BackendLocalWithHostOffload` contract arm.

## 1. What moves, and what never moves

The only demotion candidates are **committed prefix pages**: pages retained by
the per-arena `CoordinatedPrefixIndex` after a request committed, whose only
reference is the index's `prefix_ref`. Concretely, a page is eligible when its
`BlockSlot` has `prefix_refs > 0` and `table_refs == reservations ==
execution_pins == transfer_pins == 0`.

Everything else is out of scope by construction:

- Pages of active requests (table refs, reservations, execution pins) are
  never demoted. Device-arena pressure against active work keeps resolving
  through the existing ladder: prefix LRU eviction → `Backpressure` →
  capacity suspension → preemption of recompute-safe unpublished sessions.
- Pages of non-paged domains, tensor snapshots, and deployments without
  `PrefixPolicy::CommittedPages` never enter the offload path; the feature is
  inert there.
- No page ever leaves the worker process (ADR 0004, decision 1).

## 2. Components

```
ManagedKvCacheManager (owns state, the only &mut writer)
├── KvCacheCoordinator        page metadata, pins, refcounts   (existing)
├── prefix_indexes: per-arena CommittedPrefixIndex             (existing, untouched)
├── host_offload:  HostOffloadState                            (new)
│   ├── KvHostPool            page-sized host slabs, byte budget
│   ├── HostChainIndex        digest-chained host-resident prefixes
│   ├── in-flight: Vec<OffloadTransfer>   pinned, copy pending/done
│   └── copy worker thread + Semaphore(in_flight_page_budget)
└── ManagedKvTelemetry        counters (extended)
```

**KvHostPool** — per arena. Lazily allocated page slabs shaped exactly like
the arena's page geometry (per layer: `[page_tokens, kv_heads, head_dim]` K
and V). Slot allocation is byte-budgeted: the pool refuses a demotion whose
pages would exceed the configured budget, and the manager falls back to plain
eviction for the remainder. Bytes are charged through the engine
`ResourceAuthority` — the shared host/unified authority on Apple Silicon, the
host domain on CUDA — so DINV-05 holds by accounting, not convention.

**HostChainIndex** — keyed exactly like the device index: chained SHA-256
page digests under the same `KvPrefixNamespace` (cache salt included), so
tenant isolation and chain-walk semantics are identical. An entry holds the
host slots for one page plus its digest and access tick. The device index is
**not** modified: `CommittedPrefixIndex` keeps its semantics and its tests.
The two indexes partition a prefix chain: a device-resident head followed by
a host-resident tail.

## 3. Demotion (DS4.2)

**Trigger.** After commits and releases (manager safe points), the manager
computes the device pressure ratio `allocated_pages / capacity_pages` per
arena. Crossing the high watermark starts demotion steps; stepping continues
until the ratio falls to the low watermark, victims run out, or the host
budget is exhausted.

**Victim selection.** Same unit as eviction: the LRU index entry's subtree
(`remove_chain_from` semantics). The whole subtree must be eligible (§1); if
any page is referenced the victim is skipped and the next-LRU entry is tried,
bounded per step. Selected pages get their access ticks refreshed so
concurrent eviction cannot race the transfer midpoint.

**Two-phase transfer.**

1. *Pin and mark* (synchronous, under the manager lock):
   `coordinator.pin_transfer(&blocks)` + residency
   `Resident{Device}.begin_loading(op, Host)` per page; host slots reserved
   and budget-charged; the transfer (id, blocks, slots) enters the in-flight
   set and its copy is queued to the worker thread. The in-flight page count
   is bounded by a semaphore (copy budget). While in flight the pages stay
   matchable and attachable — attaching requests take table refs, and the
   demotion copy is read-only at the source, so attach races are safe.
2. *Copy* (worker thread): read the page rows from the arena into the host
   slabs (`KvArena::capture_pages`, new trait method; portable CPU
   implementation, device-gather + readback on Metal/CUDA behind the existing
   `candle_accelerator_kv_support` mutation-support gate).
3. *Apply* (next manager safe point): for each finished transfer, verify the
   pages are still index-owned and pinned; remove the subtree entries from
   the device index (dropping `prefix_refs`), `unpin_transfer` (freeing
   pages to the arena free list when nothing else holds them), insert the
   host chain entries, acknowledge residency (`Resident{Host}`), and bump
   counters. A stale or failed transfer instead aborts residency back to
   `Resident{Device}` and returns the host slots to the pool.

A page is host-resident only after step 3 — the residency machine's rule that
a destination never becomes authoritative before an acknowledged transfer is
what makes the mid-transfer states unobservable.

**Fallback.** When the host budget cannot take a victim, the manager leaves
that victim to the ordinary eviction path (pages drop). Device eviction
(`evict_lru`/`evict_lru_excluding`) is extended to purge the matching
subtree from the host chain index, so host entries never outlive their
device ancestry: lookup always starts at the chain root, so an orphaned host
tail would be unreachable.

## 4. Promotion (DS4.3)

`lookup_longest` walks the token pages through the device index; on the first
digest miss it consults the host chain index and continues the walk there.
The match result gains a host-resident tail (still contiguous: device head +
host tail).

During `prepare_inner`, before the reservation is built, the host tail is
**promoted**: a mini-transaction reserves `Fresh` pages, `KvArena::restore_pages`
writes the host bytes into them (host→device; the inverse of capture), and
`complete_write_prefix` publishes them — the ordinary DS1 publication path,
so the promoted pages are ordinary index entries afterwards. The host chain
entries are removed as the promotion commits.

Bounds and fallback:

- Promotion is synchronous on the admission path but strictly cheaper than
  the prefill it replaces (a page copy versus recompute).
- A per-prepare promotion page ceiling (configurable) caps pathological
  tails; a tail longer than the ceiling truncates the match at the ceiling.
- If the device arena cannot fit the promoted pages (reserve `Capacity`
  survives the existing evict-retry loop), the promotion transaction aborts
  and the match truncates to its device-resident prefix — the specified
  "miss → cold path": reuse degrades, correctness does not.

## 5. Budget, config, and rollout

| Surface | Key | Default |
| --- | --- | --- |
| Supervisor node config (per assignment) | `host_kv_pool_budget_bytes` | absent = off |
| Worker/engine env | `IZWI_KV_HOST_POOL_BUDGET_BYTES` | 0 = off |
| Kill switch | `IZWI_KV_HOST_OFFLOAD=0` | force-off |
| Watermarks | `IZWI_KV_OFFLOAD_HIGH_WATERMARK` / `..._LOW_WATERMARK` | 0.85 / 0.70 |
| Copy budget | `IZWI_KV_OFFLOAD_MAX_IN_FLIGHT_PAGES` | 8 |
| Promotion ceiling | `IZWI_KV_OFFLOAD_MAX_PROMOTION_PAGES` | 64 |

Validation: the supervisor rejects a pool budget exceeding the assignment's
own `host_memory_limit_bytes` (CPU/CUDA) or `shared_memory_limit_bytes`
(Metal), so the aggregate `HostMemoryOvercommit` check remains truthful
without summing a new term. The worker cross-checks the env value against its
assignment like the other budget twins. Engagement is explicit opt-in only
(ADR 0004, decision 7): no budget → the feature is compiled in but fully
dormant, and the engine records offload as disabled in its snapshot.

**Per-backend semantics.** CUDA: the host pool is additional capacity across
PCIe (unpinned host buffers for v1; pinned pages are a recorded future
optimization) and the primary concurrency lever. Metal: the pool is charged
to the shared host/unified ledger — offload is working-set trimming, and the
win is prefix retention plus admission headroom, never raw capacity. CPU:
identical mechanics to Metal with the same accounting caveat; the arena's
fixed page slots are the contended resource, so demotion still converts cold
prefix pages into free slots for active work.

## 6. Counters and observability

`ManagedKvTelemetry` gains `kv_host_pages` (gauge, projected by the manager
from pool occupancy), `demotions_total` (pages), `promotions_total` (pages),
and `promotion_latency` (total nanoseconds + count, rendered as an average).
They flow the DS2.1 route: engine snapshot → Prometheus
(`izwi_engine_kv_cache_host_pages`, `izwi_engine_kv_cache_demotions_total`,
`izwi_engine_kv_cache_promotions_total`,
`izwi_engine_kv_cache_promotion_latency_avg_seconds`; the standard
`engine.` → `izwi_engine_` dot-to-underscore rendering) → additive `Option`
fields on `LoadedDeployment` → mock-worker routing-signal knobs for contract
tests. The worker's own Prometheus endpoint renders the host-pages gauge and
the demotion/promotion counters beside the other managed-KV counters.
Absence of the fields on older workers means "offload not compiled/not
enabled", matching the additive-field contract.

## 9. As-built deviations (recorded at DS4.5)

- **Demotion is synchronous, not async.** Steps run inside the manager's
  own critical section at safe points (prepare ticks), bounded per tick by
  the in-flight page budget. The two-phase transfer records and the
  residency machine keep the structure an async copy worker needs if a later
  backend makes deferred application worthwhile; the plan's "async" wording
  was an implementation choice, not a contract.
- **Host continuation covers snapshot-sharing arenas.** The matched digest
  chain stays complete across the tier boundary (host digests extend
  `page_digests`), so DS1.2b snapshot reconciliation walks matches back
  through host-resident pages unchanged; prepare restores the full tail
  before execution, so an attach cursor inside the tail is sound.
- **The pool budget is part of the model's load-time resource
  authorization** (same charge shape as the materialization: unified ledger
  on Metal, host domain on CPU/CUDA). Charging only the materialized usage
  without extending the authorization fails the lease check on any real
  engagement.
- **Promotion truncation degrades through the cursor-lost re-plan**: a page
  that fails to restore truncates the promotion there and purges its
  unreachable host descendants; with an admission cursor the shortened match
  raises the managed-prefix cursor-lost signal and the scheduler re-plans.
- The promotion ceiling applies inside the shared lookup helper, so
  admission probes and prepare truncate identically (a probe-only ceiling
  would cursor-lost-loop against a prepare that truncates further).

## 7. Test plan (DS4.4)

Unit (coordinator/pool/index): pool budget enforcement and slot recycling;
subtree eligibility (a referenced page blocks its subtree); residency
transitions driven by the real demotion path; host-index purge on device
eviction.

Managed layer: demote-then-promote round-trip produces byte-identical page
content; demote-while-referenced race (attach between pin and completion
keeps the attached table valid and the transfer aborts cleanly); watermark
stepping; host-budget exhaustion degrades to eviction.

Engine level (DS4.4 acceptance): concurrent shared-prefix requests with a
device arena too small to hold every committed prefix — outputs equal the
cold-run outputs within the fixture tolerance, host usage stays inside the
budget, and counters show demotions and promotions. No-regression run on the
standard baseline suites.

## 8. Evidence (DS4.5)

`scripts/bench/run-ds4-offload-benchmark.sh` clones the DS2 rig shape: one
worker + gateway, `shared` workload at high concurrency, offload off then on,
per-leg worker counter deltas, plus a sequential trailer after the workload
(at concurrency some client always holds a table reference on the shared
chain, so the trailer's prepare is the safe point where the released chain
demotes and the next lookup promotes it back). Manifests
`benchmarks/manifests/ds4-{cpu,metal}-offload-{off,on}.json` plus per-lane
summaries; CUDA recorded `not run` until hardware. On Metal, success means
better retention/admission headroom under the shared budget — not more total
memory.

Recorded (2026-09-26, commit range 39c62ca9..DS4.5, fixture-scale watermarks
0.10/0.05 on the tiny synthetic arena):

- CPU on-leg: demotions=10, promotions=8, host_pages=2 inside the 8 MiB
  budget; Metal on-leg: demotions=7, promotions=3, host_pages=4.
- Reused tokens drop on the on-legs (CPU 2496 off vs 1536 on; Metal 1408 vs
  576): the fixture-scale watermarks deliberately churn chains that the
  production defaults (0.85/0.70) would keep resident, trading reuse for
  headroom — the documented degradation, not a regression.
- Engine-level acceptance (`ds4_host_offload.rs`): concurrent shared-prefix
  sessions on an undersized arena, greedy replay byte-identical to the cold
  run.
