# DS10 Best-Effort Groundwork Plan

**Status:** planning deliverable (2026-09-28). No implementation has started; this document is
the plan. It was produced from four surveys: the model-architecture/catalog layer, the serving
runtime (scheduler/KV/sampling/device), the distributed layer (protocol/gateway/supervisor),
and industry practice (vLLM/SGLang/llama.cpp, DistServe/Splitwise/Mooncake/NIXL, candle
multi-GPU state). Sources are listed at the end.

**Purpose:** DS10 in `PRODUCTION_DISTRIBUTED_SERVING_PLAN.md` is a deferred register —
"maintained as decisions, not tasks", with entry criteria gating any build. This plan proposes a
**two-tier amendment**: each register entry is split into *groundwork* (buildable now, best
effort, without satisfying the entry criteria) and *activation* (still gated exactly as the
register requires). The goal is that when the criteria are met — a MoE model in the catalog,
validated fabric, an SLO case — the activation work is small, because the scaffolding already
exists and has been proven by fixtures and tests.

Nothing in this plan contradicts a recorded rejection. The two rejected entries (in-engine
tensor parallelism, cross-node KV transfer) receive analysis, specifications, and in-process
test scaffolding only — no production serving path is proposed for either.

---

## 0. Proposed register amendment

Current DS10 text: "Deferred register (no build without entry criteria). Maintained as
decisions, not tasks." Amend to:

| Register entry | Groundwork tier (this plan) | Activation tier (unchanged gate) |
|---|---|---|
| MoE architectures + expert parallelism | **Build now**: MoE runtime core, synthetic fixtures, registration, admission fix, EP design + dispatch seam | A MoE chat model added to the supported catalog AND DS5 fleet profile validated on real hardware |
| Prefill/decode disaggregation | **Design + test scaffolding**: PD design note, page-transfer framing spec, mock-transport rig, long-prompt workload | Multi-node RDMA-class fabric validated by the DS5 rig AND measured ITL-SLO violations colocated scheduling cannot fix |
| Tensor parallelism in-engine | **No build.** Flip runbook documented; per-tensor shard addressability confirmed | Permanently rejected unless the DS8 decision gate flips with new evidence (ADR required) |
| Cross-node/cross-process KV transfer | **No production path.** Framing/identity spec + in-process codec round-trip tests | Rejected (serving-plan exclusion); flip requires a plan-level ADR |

Recommendation: record the amendment as **ADR 0008 — "DS10 groundwork posture"** (groundwork
now, activation criteria unchanged), so the register's "no build without entry criteria"
sentence is formally qualified rather than silently overridden.

---

## 1. Ground rules

1. **INV-08 stands.** Everything buildable now must pass on CPU and Metal on this host. CUDA is
   code-complete + unit-tested locally; execution evidence stays `not run` (hardware-gated).
2. **Fixtures over downloads.** The repo already synthesizes real tiny models (GGUF via
   `candle gguf_file::write`, safetensors via `serialize_to_file`, WordLevel vocab `a,b,c`,
   `izwi-artifact.json` manifests — `crates/izwi-serving-worker/tests/common/mod.rs:17-517`).
   A tiny synthetic 2-expert MoE fixture validates the entire MoE path without a downloaded
   model. A real MoE download is an activation-tier activity.
3. **Additive-only protocol changes.** Minor bump + `Option<T>` with
   `#[serde(default, skip_serializing_if = "Option::is_none")]` + absent/present decode tests
   (the DS2.1/DS9.1/DS9.3 recipe — `crates/izwi-serving-protocol/src/types.rs:174-195,
   524-525, 562-563`). Enum variants fail closed, so new capability goes through `Option`
   fields or new `WorkerFeature` variants.
4. **No rejected-path production code.** Entry D gets no cross-process endpoint; Entry C gets
   no collectives. Scaffolding proves contracts with mocks and in-process tests.
5. **Evidence conventions unchanged.** Benchmark manifests with environment metadata, lane
   driver scripts, "mock ≠ evidence / not-run ≠ passed" (plan §"Test and release gates").

---

## 2. What the surveys found (condensed, with anchors)

**Model layer.** Zero MoE code today; all chat families are dense and hand-written in-repo
(gemma3 ≈ 2.5k lines is the minimal family; qwen38 ≈ 15k). Family dispatch is an exhaustive
chain: `ModelVariant` → `ModelFamily` (`catalog/variant.rs:8,88-128`) → loader registry
(`models/registry.rs:473-499`) → `NativeChatModel`/`NativeChatDecodeState` enums
(`registry.rs:3641,4294`) → engine match arms (`engine/executor/handler_chat.rs:320,556-589`).
Adding a family touches ~19 known sites (Appendix A). `candle-transformers` 0.11.0 ships
reference MoE implementations (`qwen3_moe.rs`, `quantized_qwen3_moe.rs`, `mixtral.rs`,
`deepseek2.rs`, `granitemoehybrid.rs`, `fused_moe`) — izwi's house style is to re-implement
the graph in-repo using them as blueprints. Weight loading is name-keyed over all tensors
(`models/shared/weights/gguf.rs:366-410`), so expert tensors load generically; the work is the
model struct, config keys, and catalog metadata.

**Serving runtime.** Prefill and decode are already first-class, fully separated phases
end-to-end: `ScheduleResult{decode_requests, prefill_requests}`
(`engine/scheduler/mod.rs:200-210`), `form_physical_batches(ExecutionPhase::…)`
(`engine/core.rs:4338-4450`), separate `execute_prefill`/`execute_decode`
(`engine/executor.rs:1653-1665`), and a hard error if one physical batch mixes phases
(`executor.rs:1581-1590`). The KV page format is device-agnostic at the byte level: per layer,
dense K/V tensors `[capacity_pages, page_tokens, kv_heads, head_dim]`; DS4's host offload is
already a page-granular codec (`capture_block` / `decoded_page` / `restore_block`,
`backends/kv/page_transfer.rs:31-152`) with `KvHostPool` slots that are exactly page-sized byte
buffers — i.e. **a transfer staging format already exists**. DS1's committed-pages machinery
(digest chains, salted `KvPrefixNamespace`, `probe_managed_prefix`) is precisely an
"attach to foreign-produced KV" interface. Blockers: page identity is arena-local
(`PhysicalArenaId{model_instance, plan, backend, device_ordinal, generation}`,
`kv/v2/batch.rs:14-36`); no bulk/sequence export API; hybrid (conv/recurrent) domains have no
page codec; admission's load-peak formula assumes one dominant weight tensor
(`runtime/lifecycle/load.rs:419-437`) — a real MoE admission hazard.

**Distributed layer.** Protocol minor is 3; three minors have landed without pain (additive
recipe, `izwi-serving-protocol/src/lib.rs:23-33`). Workers advertise `DeviceAssignment`
(single device per worker) + `WorkerFeature` set + `LoadedDeployment` optional signals;
selection filters on all three (`izwi-server/src/worker_registry.rs:1228-1294`). Two workers
on two GPUs of one node work today (`config/serving/examples/multi-device-cuda.toml`);
one worker spanning two GPUs does not (no multi-device substrate anywhere — one `DeviceProfile`
per process, device-pinned arenas). Deployment pools key on `(task, public_model)`
(`gateway_deployments.rs:175-178`); autoscaling policies are per-deployment
(`izwi-serving-supervisor/src/config.rs:752-822`); the DS5 fleet store and DS6 approvals give
coordination substrate. The DS8 vLLM lane has zero code and no ADR — only plan reservations.
Boundary gate: `scripts/ci/check-serving-boundary.sh` forbids izwi-core/candle in the protocol,
client, and supervisor trees; the worker *must* link izwi-core.

**Industry practice.** Minimal viable MoE = router softmax/top-k → per-expert MLPs → weighted
sum, single device, zero collectives — exactly what llama.cpp and single-GPU vLLM do; EP is a
strictly separable later layer (vLLM's `MoELayer`/`FusedMoE` split). Production MoE serving
adds: fused grouped-GEMM kernels, shared experts, load-balancing telemetry (EPLB), and
CUDA-graph-safe padding of per-step token budgets. Memory ∝ *total* params; compute ∝ *active*
params. The on-prem MoE sweet spot is Qwen3-30B-A3B (30.5B/3B active), gpt-oss-20b/120b
(MXFP4), LFM2-8B-A1B, Granite 4.0 — all with GGUF, all in llama.cpp; llama.cpp's
`--n-cpu-moe` (experts in host RAM, attention on GPU) is the proven expert-placement lever.
PD disaggregation never raises throughput — it decouples TTFT from ITL (goodput: DistServe
7.4× more in-SLO requests); it wins on ITL tail at high batch with long prompts, and needs a
fast interconnect (RDMA/GPUDirect in production; TCP is dev-grade). KV handoff conventions:
fixed page size, per-layer contiguous K/V per page, metadata handshake (token ids, dtype,
layout, block table), layer-wise push overlapped with prefill compute — vLLM KVConnector V1 +
NIXL are the de-facto contracts. Candle has no distributed abstraction; its only multi-GPU
artifact is the `llama_multiprocess` Megatron-style TP example (one process per rank, NCCL
all-reduce via cudarc). mistral.rs proves both NCCL TP and a pure-Rust ring backend, and uses
pipeline/layer-split device mapping as the tractable multi-GPU form.

---

## 3. Entry A — MoE architectures + expert parallelism

### 3.1 Posture

Groundwork tier: build full single-device MoE support, validated on synthetic fixtures, so
that adding a real MoE model to the catalog is a data + capability-cell task, not an engine
task. Expert parallelism is **design-only**: the single-device implementation is written behind
a dispatch trait so EP becomes an alternative implementation of one interface, never a rewrite.

### 3.2 Design

**The MoE seam.** One trait owns the expert dimension:

```
trait SparseExpertDispatch: gate → top-k select → run experts → weighted combine
  ├── SingleDeviceDispatch   (build now: router softmax/top-k, loop or grouped GEMM
  │                           over selected experts, shared-expert add)
  └── (future) ExpertParallelDispatch / FusedDispatch — same interface
```

This mirrors vLLM's `MoELayer`/`FusedMoE` split. The family code calls the trait; nothing else
in the engine knows experts exist. Attention, KV geometry, cache contract, sampler, and
scheduler are untouched by MoE (KV math depends only on attention geometry — confirmed:
`kv/v2/resolved.rs:437-477` builds pages purely from attention specs).

**First family: a GGUF MoE chat family** (working name `qwen3-moe`, reference target the
Qwen3-30B-A3B class). Rationale: izwi already has a proven quantized-GGUF lane
(`GgufLoader` + `QMatMul`/`quantized_nn`, qwen35/lfm2 pattern); llama.cpp GGUF MoE is the
best-served quant ecosystem; candle's `quantized_qwen3_moe` is the in-framework reference.
The loader must handle both tensor naming schemes: llama.cpp fused-expert names
(`blk.N.ffn_gate_exps/up_exps/down_exps`) and HF-style per-expert names
(`mlp.experts.{i}.gate_proj/…`, router `mlp.gate`). Config keys: `expert_count`,
`expert_used_count`, shared-expert count/size where present.

**Host-side details that make it "just work" later:**
- Shared `ChatSampler` → grammar FSM (DS9.2) and logprobs (DS9.3) come free.
- Per-expert activation histograms: one counter array per layer per step (num_experts slots,
  incremented on selection). This is the input every production balancer (EPLB) needs; it
  costs almost nothing and is the observability half of EP groundwork.
- Fixed per-step token budget already exists (decode quanta); keep MoE selection inside the
  existing per-row decode path so any future graph capture survives dynamic expert counts
  (the DeepEP "pad to cap" convention).
- MTP stays out of scope for the MoE family (handler already hard-errors MTP to non-Qwen38 —
  `handler_chat.rs:320`).

### 3.3 Work items

- [ ] **A1 — Admission scratch fix (do first, small, benefits today).**
  `estimate_from_tensor_inventory` sets `load_peak = resident + next_pow2(largest_tensor)`
  (`runtime/lifecycle/load.rs:419-437`). MoE checkpoints have many same-sized expert tensors,
  so `largest_tensor` collapses and load scratch is under-reserved → INV-10 under-admission,
  physical OOM at load. Replace with an inventory-aware scratch term (e.g.
  `max(largest_tensor, k × p90 tensor)` plus a dequantization-scratch term for layer-by-layer
  loaders), re-validate DS0.7/DS1.5 baselines are unaffected for dense families.
- [ ] **A2 — Synthetic MoE fixtures.** Tiny 2-expert model in both flavors: GGUF (fused-expert
  names + `expert_count`/`expert_used_count` metadata, via `gguf_file::write`, pattern
  `write_tiny_lfm_fixture`) and safetensors (HF-style per-expert names, via
  `serialize_to_file`, pattern `write_tiny_qwen38_hybrid_fixture`). WordLevel vocab, bundle
  metadata, `izwi-artifact.json`. Env-gated synthetic geometry where the loader validates
  strictly (qwen38 precedent).
- [ ] **A3 — MoE runtime core.** New `models/architectures/<family>/` module: config parse
  (expert keys), `SparseExpertDispatch` trait + `SingleDeviceDispatch` impl (router
  softmax/top-k → per-expert gate/up/down → weighted sum; optional shared expert), family
  core/chat with the shared ChatSampler, cache contract (single paged-attention domain,
  gemma3-style), GGUF + safetensors loading with both naming schemes. Correctness: expert
  output equals the manually-computed routed sum on fixture weights; top-k routing matches a
  reference implementation (candle `qwen3_moe`) on the same tensors.
- [ ] **A4 — Registration sweep.** Work Appendix A end-to-end (the ~19 sites: variant/family
  enums, metadata, loader registry, `NativeChatModel`/`NativeChatDecodeState`, adapters
  family policy, load memory estimate, rollout list, conformance cases, engine arms,
  downloader, admin API, worker checks). The registration tests in
  `models/families/mod.rs:288-413` force most of this; treat them as the checklist.
- [ ] **A5 — Capability cells + catalog.** `prefix_reuse_support` cell (evidence-gated;
  NotRun/Disabled initially + inventory test), `cuda_operator_capabilities` cell, catalog
  metadata entries (`estimated_size`/`memory_required_gb` must include **total** expert bytes,
  not active), downloader manifest entries. Batched-bench decode behavior verified on CPU and
  Metal lanes with the fixture.
- [ ] **A6 — Expert telemetry.** Per-layer expert-activation histograms as engine counters
  (service snapshot surface), routed through the existing counters → `LoadedDeployment`
  optional-signal channel (`izwi-serving-worker/src/lib.rs:1095-1126`). No routing behavior
  change; additive status fields land with the next protocol minor or stay engine-side until
  EP activation (decide at implementation time).
- [ ] **A7 — EP design section (no build).** Written into this document's successor or the
  family design note: why process-boundary EP is latency-infeasible here (per-layer all-to-all
  per decode step vs a bounded NDJSON transport with sync-per-copy device reads —
  `accelerator.rs:1818-1893`); EP's realistic forms are in-process multi-device (no substrate
  today — Entry C territory) or the DS8 vLLM lane for large MoE. Activation recipe: protocol
  minor with `expert_shards` descriptor fields + `WorkerFeature`, expert-affinity selection
  predicate beside `cache_affinity_eligible`, sharding concept above `DeviceAssignment` in the
  supervisor.

### 3.4 Test strategy

Fixture-family tests (routing correctness, weighted combine, shared expert), full registration
inventory tests (already enforced), backend parity harness leg (DS0.8 pattern) for the MoE
fixture on CPU+Metal, continuous-batch fixture test (two concurrent MoE rows — the
shared-arena/quantum machinery is family-agnostic and must stay green), grammar + logprobs
public-contract legs through the mock worker with the MoE fixture.

### 3.5 Non-goals (groundwork tier)

Real model download/validation; fused grouped-GEMM kernel tuning (start with the
straightforward per-expert loop — candle's `fused_moe` is an optimization to adopt later, and
its Metal story is unproven); expert offload/placement (llama.cpp `--n-cpu-moe` analog —
register-worthy as its own future entry, natural on Apple unified memory); EP; MoE+MTP.

---

## 4. Entry B — Prefill/decode disaggregation

### 4.1 Posture

Groundwork tier: design + test scaffolding + pre-registered measurement, so the entry criteria
(fabric + measured ITL-SLO violations) can be *evaluated* cheaply and activation is a build,
not a research project. The honest framing from the literature: PD never raises throughput —
it decouples TTFT from ITL, and wins only on ITL tail at high batch with long prompts over a
fast interconnect. On this host, on-node IPC (shared memory / unified-memory copies) is the
only acceptable transport, and on Metal the "two pools" share one physical memory pool
(DINV-05) — so the local scaffolding proves *contract*, not *win*.

### 4.2 Architecture mapping onto izwi

The key structural fact: **a prefill worker already publishes its output in a form a decode
worker can attach to** — DS1 committed pages + digest chains + salted namespaces +
`probe_managed_prefix` is exactly an "attach to foreign-produced KV" interface, currently
in-process. XpYd on izwi = two role-tagged worker pools (prefill/decode) with the existing
pools/approvals machinery extended by a role dimension; the missing pieces are (1) transport
across the process boundary, (2) a scheduler-level handoff state machine (suspend on A, adopt
on B: cursor + KV receipts + stream/logprob continuity — today only the recompute path exists,
`scheduler/mod.rs:2461`), (3) role-aware selection and two-leg dispatch in the gateway.

### 4.3 Work items

- [ ] **B1 — PD design note** (`docs/dev/PD_DISAGGREGATION_DESIGN.md`): the mapping above,
  the handoff state machine (states, fencing via `expected_worker_incarnation` + attempt
  identity, failure = recompute fallback), hybrid-model gap (conv/recurrent domains need
  snapshot transfer, not just pages), DINV compliance (DINV-02 tenant scope on KV handles,
  DINV-03 degrade, DINV-04 identity), and the activation recipe below.
- [ ] **B2 — Page-transfer framing spec** (shared with Entry D, appendix of B1): a header over
  the *existing* DS4 codec — per page: arena dtype/layout/geometry (`KvPhysicalLayout`,
  `arena_page_bytes`), `KvPrefixNamespace` fingerprint, prompt-token digest chain, position
  semantics, source `KvPlanFingerprint` for receiver compatibility gating; sequence framing =
  ordered pages + per-page digest so the receiver can re-key into its own arena/generation and
  insert via the `CoordinatedPrefixIndex` path. Layer-wise push convention (compute layer *l*
  while sending *l+1*) documented as the activation-time default (LMCache/vLLM convention).
- [ ] **B3 — Mock-transport rig (test-only).** Two mock workers exchange framed pages over
  loopback TCP: producer captures fixture-model pages with the real codec, consumer re-keys and
  attaches them and continues generation; outputs match the collocated run. This proves the
  B2 contract end-to-end **without any engine cross-process path** — the engine stays
  node-local; the rig is contract evidence, not a serving path (mirrors the DS8 mock-vLLM
  pattern, and would be DINV-09's sibling for PD).
- [ ] **B4 — Long-prompt benchmark workload + entry-criteria procedure.** New `--workload`
  choice in `run-gateway-chat-benchmark.py` (multi-thousand-token prompts, the regime where PD
  can win) + a lane-driver script pattern for a future collocated-vs-disaggregated A/B. The
  entry-criteria measurement is then: run the long-prompt workload, record ITL p99 vs SLO
  under saturation, on the validated fabric. Pre-registering this makes the DS10 gate cheap to
  evaluate honestly.
- [ ] **B5 — Activation recipe (documented, applied only at activation):** protocol minor with
  worker role field + `WorkerFeature::PrefillOnly/DecodeOnly`; pool-key extension
  `(task, public_model, role)` + approvals-format version bump; two-leg dispatcher variant of
  `RemoteChatDispatcher::start` (two `AttemptIdentity`s, retry rules spanning two workers);
  per-pool autoscaling policies; KV-shipping endpoint beside the attempt table (realtime-WS
  precedent for a second transport sharing attempt identity).

### 4.4 Non-goals

No production prefill→decode handoff; no RDMA/NIXL work; no scheduler handoff state machine in
the engine; no pool-role protocol fields until activation.

---

## 5. Entry C — Tensor parallelism in-engine (permanently rejected)

### 5.1 Posture

No build. The register's standing answer for big models is the DS8 vLLM lane (which brings its
own TP), and the survey confirms why: in-process TP needs a collective runtime inside the model
loop (cudarc NCCL is CUDA-only — no Metal equivalent, breaking INV-08), candle has no
distributed abstraction, and the whole engine assumes one device per process (one
`DeviceProfile`, device-pinned arenas, device-keyed kernel caches). "Best-effort support" for
this entry means: **keep the flip cheap, and keep the sanctioned path clearly marked.**

### 5.2 Work items

- [ ] **C1 — Flip runbook (documentation in this doc's successor):** trigger conditions
  (what new evidence would reopen it), the required ADR, and the build sketch: multi-device
  `DeviceAssignment` variant + supervisor collective-launch topology + candle Megatron-style
  TP per the `llama_multiprocess` precedent (cudarc NCCL, one process per rank, weight shards
  via `candle_nn::var_builder::Shard`). Confirmations recorded now: GGUF/safetensors loading
  is name-keyed per tensor, so per-layer weight **shard addressability already holds**; the
  protocol is layer-agnostic; process-per-device is already the unit of placement
  (two-GPU-node = two workers today).
- [ ] **C2 — Pipeline-parallelism register proposal (no build):** if in-engine multi-GPU is
  ever needed natively, layer-split pipeline parallelism (point-to-point at layer boundaries —
  the mistral.rs device-mapping model) is the tractable form for candle and maps onto the
  process model; propose it as a *new* DS10 register entry with entry criteria, rather than
  silently extending this rejected one.

### 5.3 Non-goals

Everything else. No collectives, no multi-device assignment, no NCCL dependency.

---

## 6. Entry D — Cross-node/cross-process KV transfer (rejected)

### 6.1 Posture

No production path. The serving-plan exclusion stands (DS4 stays node-local). The survey's
good news: the expensive half already exists — the DS4 page codec is device-agnostic and
symmetric (`capture_page`/`restore_page` on CPU and accelerator arenas; accelerator pages round
-trip through host bytes), `KvResidencyState`/`KvTransferId` already model an acknowledged
transfer lifecycle, and DS1's prefix machinery is the semantic attach interface. Groundwork =
specification + in-process proof, so a future flip (which would ride Entry B's activation)
starts from a proven format.

### 6.2 Work items

- [ ] **D1 — Identity/re-keying + provenance spec** (appendix of the B2 framing spec): how a
  receiving process re-keys foreign pages into its own `PhysicalArenaId`/generation; receiver
  compatibility gate = `KvPlanFingerprint` equality (dtype, page size, layer geometry);
  provenance = namespace fingerprint + digest chain + positions semantics; tenant isolation
  inherits DINV-02 (salt-scoped).
- [ ] **D2 — In-process codec round-trip tests.** Property tests over the existing
  `capture_block`/`restore_block` codec across pages, dtypes (F32/F16/BF16), and layouts on
  CPU and Metal: capture → bytes → restore → bitwise-equal attention output. These tests are
  legal today (no cross-process anything) and pin the byte format the B2 spec documents.
- [ ] **D3 — Hybrid-state gap documented:** conv/recurrent domains (`TensorStateArena`,
  Tensor/Append/Ring kinds) have no page codec — transferred state for hybrid models
  (qwen3.8, LFM2) requires the committed-snapshot path; recorded as an explicit limitation of
  any future transfer design (dense-attention models only, initially).

### 6.3 Non-goals

No transfer endpoint, no wire protocol code, no worker↔worker transport, no ledger changes.
Production KV remains node-local.

---

## 7. Sequencing and effort

| Order | Item | Size | Why this order |
|---|---|---|---|
| 1 | A1 admission scratch fix | S | Independent, correctness-relevant today, unblocks honest MoE admission |
| 2 | A2 synthetic MoE fixtures | S | Everything else validates against them |
| 3 | A3 MoE runtime core | M–L | The heart of "MoE just works" |
| 4 | A4 registration sweep | M | Mechanical once A3 lands; tests force it |
| 5 | A5 capability cells + catalog | S | Gates the family per backend |
| 6 | A6 expert telemetry | S | EP groundwork, no behavior change |
| 7 | D2 codec round-trip tests | S | Independent; pins the KV byte format |
| 8 | B1+B2 design notes, D1/D3 sections | M | Writing task; benefits from A/D lessons |
| 9 | B3 mock-transport rig | M | Contract evidence for B2/D1 |
| 10 | B4 long-prompt workload | S | Pre-registers the DS10 gate measurement |
| 11 | C1/C2, A7, B5 doc sections | S | Analysis captures |

Each item = independently verified commits per the repo's task conventions. S ≈ a session
slice, M ≈ a few, L ≈ the largest prior family-sized effort (but A3 is bounded by using the
shared sampler and the gemma3-style minimal family shape).

## 8. Risks and honest limitations

- **Groundwork ≠ entry criteria.** Shipping A1–A6 does not satisfy DS10's activation gate;
  a real MoE model must still be cataloged and validated, per backend, with evidence.
- **candle fused-MoE maturity on Metal** is unproven — the groundwork deliberately does not
  depend on it (per-expert loop first; fused kernel is a later optimization).
- **GGUF MoE quant diversity** (Q4_K-class expert tensors, unsloth dynamic quants) means the
  loader must be proven against more than one quant layout before activation; fixtures cover
  the mechanics, not the zoo.
- **A1 changes admission numbers** — dense-family baselines must be re-checked so the fix is
  provably a scratch-reservation correction, not a capacity regression.
- **PD scaffolding proves contract, not value.** On this host (loopback, Metal unified memory)
  no PD win is measurable; the entry-criteria measurement (B4) requires the validated fabric
  the register demands.
- **EP honesty:** per-token all-to-all across processes on this transport would dominate decode
  latency; EP activation realistically requires in-process multi-device (Entry C territory) or
  the DS8 lane. The dispatch trait guarantees a swap, not a schedule.

## 9. Sources

vLLM MoE layer design ([#31578](https://github.com/vllm-project/vllm/issues/31578));
[DeepEP](https://github.com/deepseek-ai/DeepEP); [EPLB](https://github.com/deepseek-ai/EPLB);
[lmsys DeepSeek-V3 serving](https://lmsys.org/blog/2025-05-05-deepseekv3-azure/);
[unsloth Qwen3-30B-A3B GGUF](https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF);
[llama.cpp gpt-oss guide](https://github.com/ggml-org/llama.cpp/discussions/15396);
[unsloth Dynamic GGUFs](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs);
[DistServe, arXiv:2401.09670](https://arxiv.org/abs/2401.09670);
[Splitwise, arXiv:2311.18677](https://arxiv.org/abs/2311.18677);
[Mooncake, FAST'25](https://www.usenix.org/system/files/fast25-qin.pdf);
[vLLM disaggregated prefill](https://docs.vllm.ai/en/latest/features/disagg_prefill.html);
[llm-d disaggregation](https://llm-d.ai/docs/dev/architecture/advanced/disaggregation);
[Revisiting disaggregated LLM serving, arXiv:2601.08833](https://arxiv.org/html/2601.08833v1);
[LMCache/NIXL in vLLM V1](https://blog.lmcache.ai/en/2025/04/11/shaping-nixl-based-pd-disaggregation-in-vllm-v1);
[LMCache, arXiv:2510.09665](https://arxiv.org/pdf/2510.09665);
[candle llama_multiprocess](https://github.com/huggingface/candle/tree/main/candle-examples/examples/llama_multiprocess);
[candle #2007](https://github.com/huggingface/candle/issues/2007);
[cudarc](https://lib.rs/crates/cudarc);
[mistral.rs](https://ericlbuehler.github.io/mistral.rs/);
[vllm.rs](https://github.com/guoqingbao/vllm.rs).

## Appendix A — Places to touch when adding a new model family

(extracted from the model-layer survey; A4 executes this list for the MoE family)

1. `crates/izwi-core/src/catalog/variant.rs` — `ModelFamily` enum (:8), variant→family match
   (:88-128), parse aliases (:163+).
2. `crates/izwi-core/src/catalog/metadata.rs` — `ModelVariant` enum (:27), `repo_id` (:317),
   `display_name` (:387), `dir_name` (:444), `estimated_size` (:501, include total expert
   bytes), `memory_required_gb` (:558), quantization predicates (:665-986), `is_enabled`
   (:991), `chat_capabilities` (:810), `all()` (:1033).
3. `crates/izwi-core/src/catalog/prefix_reuse.rs` — capability cell (:124) + inventory test
   (:228).
4. `crates/izwi-core/src/catalog/cuda_support.rs` — capabilities arm (:290) + provider class
   (:347).
5. `crates/izwi-core/src/models/architectures/<family>/` — new module, wired in
   `architectures/mod.rs`.
6. `crates/izwi-core/src/models/families/mod.rs` — registration entry (:120-268); tests
   (:288-413) enforce completeness.
7. `crates/izwi-core/src/models/registry.rs` — loader fn (:282-365 pattern),
   `CHAT_LOADER_REGISTRY` (:473-499), `NativeChatModel` (:3641) and
   `NativeChatDecodeState` (:4294) arms with all matches (e.g. `drain_pending_logprobs`
   :4304).
8. `crates/izwi-core/src/runtime/adapters.rs` — `family_inference_state_policy` (:144),
   `chat_sequence_execution` (:433).
9. `crates/izwi-core/src/runtime/adapters/loaded.rs` — loaded-adapter family gates (:862-941).
10. `crates/izwi-core/src/runtime/lifecycle/load.rs` — memory estimate (:287 / :241 pattern;
    :439), state publication (:1521-1584, generic via `InferenceStateContractProvider`).
11. `crates/izwi-core/src/runtime/rollout.rs` — staged-rollout variant list (:188-260).
12. `crates/izwi-core/src/runtime/conformance.rs` — conformance cases (:133, :182).
13. `crates/izwi-core/src/engine/executor/handler_chat.rs` — managed-cache/state match arms
    (:320, :556-589, :1012-1029); check `engine/executor.rs:3371`, `engine/core.rs:1243`.
14. `crates/izwi-core/src/artifacts/downloader.rs` — per-variant manifests (:43-86, :256-292).
15. `crates/izwi-server/src/api/admin/models.rs` — catalog listing (:864-887).
16. `crates/izwi-server/src/app/chat.rs` (:59), `app/chat_content.rs` (:291),
    `api/openai/chat/completions.rs` (response-format gating).
17. `crates/izwi-serving-worker/src/main.rs` — variant checks (:571).
18. `crates/izwi-serving-worker/tests/common/mod.rs` — fixture builder (:17, :201 patterns).
19. Family-local tests (patterns: `qwen38/chat/recovery_tests.rs`,
    `gemma3/core.rs:894` tiny-weights).
