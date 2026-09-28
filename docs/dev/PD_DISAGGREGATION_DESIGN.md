# Prefill/Decode Disaggregation and Page-Transfer Framing Design (DS10 Groundwork)

**Status:** groundwork-tier design (2026-09-28). Covers DS10 entries *prefill/decode
disaggregation* (B1), *page-transfer framing* (B2/D1), and the *hybrid-state serialization
gap* (D3). Recorded decisions stand: no cross-process/cross-node KV path exists in the engine
(serving-plan exclusion), and nothing here builds one. This document specifies what WILL be
built if the register's entry criteria are met, so activation is an engineering task rather
than a research project, and so the local scaffolding (`tests/kv_transfer_rig.rs`) can prove
the contract today. ADR 0008 fixes the two-tier posture.

## 1. PD architecture mapped onto izwi

The literature is consistent about what PD is for: it **never raises throughput** — it
decouples TTFT from ITL (DistServe's goodput), and wins only on ITL tail at high batch with
long prompts over a fast interconnect. The DS10 entry criteria (multi-node RDMA-class fabric
validated by the DS5 rig AND measured ITL-SLO violations colocated scheduling cannot fix)
encode exactly that.

The key structural fact from the codebase: **prefill and decode are already fully separated
phases end-to-end** — `ScheduleResult{decode_requests, prefill_requests}`
(`engine/scheduler/mod.rs`), `form_physical_batches(ExecutionPhase::…)` with a hard error on
mixing phases (`engine/executor.rs`), separate `execute_prefill`/`execute_decode`. And **a
prefill worker already publishes its output in a form a decode worker can attach to**: DS1's
committed pages (digest chains, salted `KvPrefixNamespace`, `probe_managed_prefix`) is an
"attach to foreign-produced KV" interface, currently in-process. XpYd on izwi is therefore:

```
        ┌──────────── prefill pool (role-tagged workers) ────────────┐
chat ──►│ prefill worker: runs PrefillMode::Full, publishes KV       │
        └──────────────┬─────────────────────────────────────────────┘
                       │ page-transfer framing (§2) over transport
        ┌──────────────▼──────────────────────────────────────────────┐
        │ decode pool: adopts sequence state, runs continuous decode  │──► client
        └──────────────────────────────────────────────────────────────┘
```

Missing pieces, in dependency order:

1. **Transport** (§2 framing): process-boundary byte transport for pages. Local prototype =
   loopback TCP / shared memory; production = the validated fabric only.
2. **Scheduler handoff state machine** (§3): suspend-on-A / adopt-on-B.
3. **Role-aware placement** (§5 activation recipe): pool roles, two-leg dispatch.

## 2. Page-transfer framing (B2/D1)

A wire format over the **existing** DS4 page codec — `capture_block` / `decoded_page` /
`restore_block` (`backends/kv/page_transfer.rs`), already proven bit-preserving across
F32/F16/BF16 on CPU and Metal (D2 property tests). Per page:

```
KvPageFrame {
  header:
    magic:        "IZKV1"                     (5 bytes, version gate)
    page_bytes:   u32                         (payload length; == arena_page_bytes(config))
    dtype:        u8   (F32=0, F16=1, BF16=2) (arena storage dtype)
    layout:       u8   (PageTokenHeadDim=0, PageHeadTokenDim=1)
    page_tokens:  u32
    layers:       u16                         (per layer: key block then value block, native order)
    geometry:     [ {num_kv_heads: u16, key_head_dim: u16, value_head_dim: u16}; layers ]
    position_base: u64                        (absolute token position of this page)
    namespace:    [u8; 32]                    (KvPrefixNamespace fingerprint — tenant/salt scope, DINV-02)
    chain:        [u8; 32]                    (digest chain: SHA-256 over position_base ‖ namespace
                                               ‖ previous-page chain ‖ payload — the receiver can
                                               verify and re-derive the DS1 committed-page digest)
    payload:      [u8; page_bytes]            (capture_block output, layer-ordered)
}
```

Receiver-side rules (D1 — identity and re-keying):

- **Compatibility gate:** the receiver's resolved plan must fingerprint-match the frame
  geometry (dtype, page size, per-layer kv geometry) — the same checks
  `KvPlanFingerprint`/`PageSizeConstraint` enforce in-process.
- **Re-keying:** frames carry no arena identity. The receiver imports payload into its own
  `PhysicalArenaId`/generation via `restore_block`-equivalent writes and registers the span
  in its `CoordinatedPrefixIndex` with the **externally supplied** namespace + digest chain,
  so DS1's reuse semantics (tenant isolation, eviction, cursor probing) apply unchanged.
- **Integrity:** chain verification before attach; a mismatch rejects the whole span.
- **Conventions borrowed:** layer-wise push overlapped with prefill compute is the
  activation default (LMCache/vLLM practice); the NIXL shape (register buffers once, exchange
  endpoint metadata, cache-to-cache transfer) is the production target; RDMA/GPUDirect are
  pluggable transports that do not exist in this codebase.

`KvResidencyState`/`KvTransferId` (`kv/residency.rs`) already model an acknowledged transfer
lifecycle (DS4); activation extends Host↔Device transitions with Process↔Process rather than
inventing a state machine.

## 3. Scheduler handoff state machine (design only)

Today the only "cross-worker sequence movement" is recompute (`restart_request_for_recompute`).
PD activation needs an adopt path:

```
PREFILL_ASSIGNED → PREFILL_COMMITTING (all pages framed+acked)
  → HANDOFF_READY (cursor, KV receipts, sampler state exported)
  → DECODE_ADOPTED (decode worker re-keys span, rebuilds ChatDecodeState:
     cache = adopted reservation, pos = cursor, unconsumed_output = final prefill logits,
     sampler + generated_ids + assembled carried in HANDOFF_READY)
  → DECODE_RUNNING (normal continuous decode; ownership fenced by expected_worker_incarnation)
Failure at any step → HANDOFF_FAILED → recompute fallback (existing path).
```

Invariants carried over from DS3's ownership machinery: exactly one live incarnation per
sequence; capacity released only on confirmed teardown; the public API never sees the
handoff. Streaming/logprob continuity: prefill's final logits are consumed by the decode
worker's first sample, so the client's first token is produced post-handoff.

**Hybrid-model gap (D3):** conv/recurrent domains (`TensorStateArena`, Tensor/Append/Ring
kinds — qwen3.8 hybrid, LFM2) have no page codec; their reusable state travels as committed
*tensor snapshots* (DS1.3 `CommittedSnapshots`). A transferred hybrid sequence needs the
snapshot serialization added to this framing; until then transferred KV is valid for
dense-attention models only (this limitation is a design constant, not an implementation
shortcut).

## 4. Local scaffolding (built now, test-only)

`crates/izwi-serving-worker/tests/kv_transfer_rig.rs` (B3): a loopback-TCP producer/consumer
pair. The producer writes real K/V data into a CPU arena, captures pages with the real codec,
frames them per §2, and sends them; the consumer parses frames, verifies the digest chain,
re-keys into a second arena, and the rig asserts bit-identical materialized pages on both
sides. The rig is **contract evidence** (the DINV-09 sibling for PD), not a serving path: no
engine code, no scheduler states, no protocol types ship.

## 5. Activation recipe (documented, applied only at activation)

- **Protocol minor 4:** worker role field (`Option<PoolRole>` prefill/decode/both) +
  `WorkerFeature::{PrefillOnly, DecodeOnly}` + KV-handle identity types
  (`crates/izwi-serving-protocol/src/identity.rs` bounded-id pattern), all additive with
  absent/present decode tests.
- **Gateway:** pool key `(task, public_model, role)`; two-leg dispatcher variant of
  `RemoteChatDispatcher::start` (two `AttemptIdentity`s, retry rules spanning two workers);
  approvals-format version bump.
- **Supervisor:** role-tagged `[[workers]]`, per-pool autoscaling policies.
- **Engine:** the §3 handoff states; hybrid snapshot framing if hybrid models are in scope.
- **Measurement:** the pre-registered `long_prompt` bench workload (2048–8192 unique-token
  prompts, raised `--max-tokens`), collocated vs. disaggregated A/B on the validated fabric;
  record ITL p99 vs SLO per lane in `benchmarks/manifests/` with the standard
  mock-≠-evidence rules.

## 6. Non-goals (groundwork tier)

No production handoff path, no RDMA/NIXL work, no scheduler handoff states in the engine, no
protocol fields until activation, no hybrid snapshot serialization.
