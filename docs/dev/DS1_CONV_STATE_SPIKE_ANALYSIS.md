# DS1.1 spike: committed-state fork for hybrid-attention prefix reuse

Date: 2026-09-23. Scope: the serving plan's deliberate block on
`PrefixPolicy::CommittedPages` for hybrid linear-attention/conv chat models
(qwen38) — recorded in `tasks/multi-user-serving-optimizations-plan-2026-08-21.md`
as "unsound until the recurrent/conv checkpoint-boundary spike answers how
shared spans rebuild linear-attention and conv state."

## DS1.2 enablement status (updated 2026-09-24)

1. **DONE (DS1.2a)** — qwen38 declares `CommittedSnapshots` on its recurrent +
   conv tensor domains and `CommittedPages` on the paged-attention domain,
   behind `enable_prefix_caching`, with MTP kept private.
2. **DONE (DS1.2b)** — the managed cache publishes committed tensor snapshots
   and attaches them by fork. `engine/cache/tensor_snapshots.rs` holds the
   strict sharing gate (`tensor_snapshot_prefix_policy`: exactly one paged
   domain with `CommittedPages`, every tensor domain `CommittedSnapshots` with
   one shared interval, one consistency group) and the bounded LRU
   `TensorStateSnapshotIndex`. Snapshots are keyed by the digest of the paged
   page-chain page whose end equals the snapshot cursor, so tenant salt +
   model generation binding (DINV-02) are inherited from the paged namespace
   instead of re-derived, and tensor reuse can never exceed or diverge from
   the paged reuse the same lookup authenticated. Publication happens in
   `finalize` only when the legacy commit cursor is interval-aligned and pages
   were accepted through that boundary; attach walks the matched paged chain
   back to the nearest snapshot-backed boundary (falling back to zero reuse)
   and seeds the fresh sequence with the DS1.1 prototype recipe
   (read → begin → stage_replace → commit). The scheduler aligns the
   first prefill chunk of a sharing contract to the declared interval via
   `ExecutionProfile::managed_snapshot_prefill_interval`, so a published
   boundary is reachable by longer-prompt attachers (system prompts, growing
   histories). Identical-length sibling sessions would need zero-ingest
   prefill authority over the unchanged-prefix fence — deliberately deferred.
3. **DONE (DS1.2a/DS1.2b)** — MTP stays private and any contract the strict
   gate rejects keeps its declarations admission-inert.
4. **PARTIAL** — the correctness suite covers aligned publish + fork attach +
   value identity under source continuation, off-boundary non-publication,
   truncation, salt-less rejection, abort, and inert contracts. Backend
   parity through the DS0.8 harness still needs a prefix-enabled hybrid
   fixture executing end-to-end (the tiny LFM fixture keeps declarations
   disabled); it gates default-on, which remains off.
5. **DS1.5 process findings (2026-09-24)** — a disk-loadable tiny hybrid
   qwen38 fixture (`IZWI_ALLOW_SYNTHETIC_QWEN38_GEOMETRY`, worker fixture
   builder in `izwi-serving-worker/tests/common`) executes end-to-end through
   the real worker: aligned-chunk snapshot publication works and is
   observable on the worker's `/internal/v1/metrics/prometheus` (managed-KV
   counters). Two soundness/integration defects were found and fixed or
   fenced:
   - **FIXED** — the managed-side first-chunk clamp could reduce the commit
     target below the scheduled input span (when the scheduler-level clamp
     did not apply to the same chunk), so the model appended pages past its
     block table ("physical paged append ends at N, beyond capacity M").
     The clamp now only engages when the scheduled span already ends at or
     below the boundary; alignment is the scheduler's responsibility.
   - **OPEN (blocks DS1.5 attach evidence)** — after a session attaches a
     published snapshot at cursor C, the scheduler still issues the prefill
     span from 0 while the model must append from C, so the paged append
     overruns the block table and the stream closes without a final marker.
     The scheduler/executor must clip (or replay) the scheduled span to the
     attach cursor. Reproduce with
     `cargo test -p izwi-serving-worker --test prefix_attach_repro -- --ignored`.
     Until this lands, DS1.5 records publish-side evidence only and
     `enable_prefix_caching` stays opt-in (default off).

# Original spike analysis (2026-09-23)

## Question

When request B attaches to a token-identical shared prefix of length N that
request A has already ingested, can B's per-layer **recurrent (SSM)** and
**convolution (shortconv)** state be reconstructed soundly at cursor N —
without recomputing the shared span — so cross-request prefix reuse is sound
for hybrid models?

## Findings (all verified in code at `b9df5e07`..`f3bb1802`)

1. **State geometry** (`models/architectures/qwen38/cache.rs:33-103`): the
   composite contract declares three retained domains per model — paged
   attention (full-attention layers), recurrent SSM state, and conv rolling
   state (non-full layers). Each non-attention layer's recurrent/conv state
   is a fixed-size tensor component keyed only by the decoder cursor.
2. **No positional reconstruction problem exists.** Unlike attention KV
   (which is per-token and paged), the recurrent and conv state at cursor N
   is a *single checkpointable tensor*. The contract already stamps tensor
   domains `CheckpointPolicy::Transactional`
   (`models/architectures/qwen38/cache.rs:110-117`).
3. **The fork mechanism exists and is transactional.** The tensor state
   arena supports begin/stage/commit/abort with cursor-aligned snapshots and
   staged restore (`backends/state/tensor.rs`; e.g. staged-snapshot restore
   at `:507-887`, atomic absence handling `:1298-1323`).
4. **Prototype proof**: `committed_state_forks_to_a_new_sequence_and_
   advances_identically` (`backends/state/tensor.rs`, this commit) seeds a
   second sequence from a committed snapshot and proves cursor-aligned,
   value-identical continuation. B's state at cursor N+k equals A's state at
   N+k for identical updates.
5. **Prefix-sharing policy plumbing already anticipates this**: `PrefixPolicy::
   CommittedSnapshots { interval_steps }` exists in the contract enum
   (`kv/v2/contract.rs:932-952`) and the engine's managed-cache shareability
   predicate already accepts `CommittedSnapshots` for tensor domains
   (`engine/cache/managed.rs:3291-3304`).

## Verdict: GO

The old "unsound" verdict assumed shared spans must *rebuild* recurrent/conv
state by recompute (which would indeed be unsound to half-do). The sound
mechanism is **attach-by-fork**: B forks A's committed tensor-state snapshot
at cursor N (transactional, zero recompute) and ingests only tokens [N, len).
Snapshot granularity bounds reuse granularity — set the commit interval to
the page size (or 64-128 tokens) so long shared prefixes (system prompts,
multi-turn histories) hit while short ones do not.

## Enablement requirements for DS1.2 (ordered)

1. Declare `PrefixPolicy::CommittedSnapshots { interval_steps }` on qwen38's
   recurrent + conv tensor domains (and `CommittedPages` on its paged-attention
   domain) behind the existing `enable_prefix_caching` config, default-off.
2. Wire the managed-cache publication path to fork committed tensor snapshots
   into attaching invocations (prototype API: read → begin → stage → commit at
   the fork cursor), enforcing tenant salt + generation binding on the
   snapshot key (DINV-02).
3. **MTP interaction**: the MTP draft domain is transactional but draft-state
   sharing is unproven — enable prefix sharing only for non-MTP rows, or
   snapshot the MTP domain identically, behind a capability flag. This is why
   "solo rows keep MTP" held in the batching work.
4. Correctness suite before default-on: shared-prefix concurrency, eviction
   during fork, salt isolation, and CPU/Metal parity via the DS0.8 harness.

## Risks

- Fork cost is O(state bytes) per attach (state is small: recurrent+conv
  tensors only) — cheap relative to prefill recompute.
- Snapshot-interval alignment reduces hit rate vs exact-page reuse; measure
  in DS1.5's benchmark manifest.
- MTP interaction is the only genuinely unresolved correctness item; it is
  scoped out of first enablement rather than blocking.
