# Qwen3.5-35B-A3B-FP8 support — research, gap analysis, and implementation plan

Status: **implementation in progress** (2026-09-29). Phases 0–2 are implemented and
committed (catalog registration `4935c9af`, native FP8 ingestion `520e624a`, architecture
+ chat + registry wiring in the Phase 2 commit). The model will still be
downloaded later on a server; local development is best-effort against synthetic
fixtures, per the established completion boundary in `tasks/lessons.md`
("portable source, synthetic/reference, failure-injection, compile, and
evidence-runner checks are the implementation completion boundary; full-model
runtime/performance certification is an exact-SHA hardware handoff").

### Routing-mode pin (G4/D4 resolved, 2026-09-29)

Pinned from the transformers `qwen3_5_moe` modeling source (checked 2026-09-29):
routing is **F32 softmax → top-k → in-top-k renormalization**. No sigmoid, no
`e_score_correction_bias`, and the config class exposes no `norm_topk_prob` /
`scoring_func` knobs. The existing `SparseMoeDispatcher` with
`norm_topk_prob: true` matches exactly, so the dispatcher is reused unchanged.
The shared expert is `down(silu(gate(x))·up(x))`, scaled by
`sigmoid(shared_expert_gate(x))` when the gate tensor is present, added to the
routed output unconditionally.

### Tensor-plan corrections found by the Phase 2 forward evidence (2026-09-29)

The Phase 2 native-trunk forward test forced the plan against the real
architecture semantics and surfaced two Phase 1 shape errors, fixed before
hardware handoff could have:

1. `self_attn.q_proj.weight` is **[num_heads × head_dim × 2, hidden]** — the
   gated full attention fuses the per-head sigmoid output gate into q_proj
   (transformers chunks the q_proj output into query/gate halves; llama.cpp
   encodes the same fusion, which the shared trunk consumes). Pinned-35B
   shape: [8192, 2048], not [4096, 2048].
2. `linear_attn.norm.weight` is **[linear_value_head_dim]** (per-value-head
   gated RMS norm), not `[ssm_v_width]`. Pinned-35B shape: [128], not [4096].

Also confirmed: the published contract pins linear key width = hidden and
value width = 2 × hidden (validated in the config parser), and the DeltaNet
`A_log` must be materialized as `a = -exp(A_log)` before the trunk recurrence
consumes it.

---

## 1. What the model is

`Qwen/Qwen3.5-35B-A3B-FP8` (Apache-2.0), the official FP8 build of
Qwen3.5-35B-A3B. Facts below are from the HF model card and `config.json`
(verified 2026-09-29):

| Property | Value |
|---|---|
| Architecture | `Qwen3_5MoeForConditionalGeneration`, `model_type: qwen3_5_moe` |
| Shape | 40 layers, `full_attention_interval: 4` → 10 × [3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE)] |
| Hidden / vocab | 2048 / 248,320 |
| Gated DeltaNet (30 layers) | 32 V heads, 16 QK heads, head_dim 128; conv + gated-RMS; `mamba_ssm_dtype: float32` |
| Gated full attention (10 layers) | 16 Q heads, 2 KV heads, head_dim 256; partial RoPE (dim 64, factor 0.25), theta 1e7, interleaved mrope sections [11, 11, 10] |
| MoE (all 40 FFNs) | 256 routed experts, 8 activated per token **+ 1 shared expert** (intermediate 512 both); `router_aux_loss_coef 0.001` |
| Quantization | FP8 E4M3, dynamic activation scheme, `weight_block_size [128,128]`; `modules_to_not_convert`: lm_head, embeddings, per-layer `linear_attn` conv1d/in_proj, MoE gates, vision blocks, merger/patch_embed, **MTP modules** |
| Checkpoint | Indexed safetensors only (~37.5 GB): ~34.45B params in F8_E4M3 + ~1.5B in BF16 (embeddings, scales-adjacent dense weights) + F32 norms |
| MTP | 1 hidden MTP layer, no dedicated embeddings — present in the checkpoint (BF16) |
| Context | 262,144 native; YaRN-extensible to ~1M (out of scope here) |
| Vision | Vision tower present (depth 27, hidden 1152); text-only serving is officially supported (`--language-model-only` in vLLM) |
| Thinking | ON by default (generation prompt inserts `<think>\n`); `enable_thinking=False` inserts an empty think block; Qwen3 `/think` `//nothink` soft switches are NOT supported |
| Sampling (card) | temp 1.0, top_p 0.95, top_k 20, presence_penalty 1.5; max output 32,768 |

**Routing hypothesis to pin at implementation time:** the config does not emit
`norm_topk_prob` or `scoring_func`. Qwen3-MoE lineage suggests softmax top-k with
in-top-k renormalization, but the transformers `qwen3_5_moe` config-class defaults
and `modeling_qwen3_5_moe.py` must be checked before any numerical-parity claim.
If it turns out to be sigmoid routing (DeepSeek-style), the dispatcher needs a
second scoring mode (see Gap G4).

## 2. What the branch already has (the three pillars)

Exploration (2026-09-29, three code sweeps) found that this checkpoint is close
to the *intersection* of three existing families:

1. **`Qwen35Chat`** (`crates/izwi-core/src/models/architectures/qwen35/`) — the
   dense Qwen3.5 GGUF family (0.8/2/4/9B). It already implements:
   - the Gated DeltaNet mixer with config-driven head geometry
     (`text.rs:1295-1299`); the 35B's 32V/16QK/dim128 maps directly, and the
     V-head-repeat path + Metal sequence op handle it (`text.rs:1624-1639`);
   - gated full attention with per-head sigmoid output gate and q/k norm
     (`text.rs:913-1012`), head_dim 256 OK;
   - interleaved 3-section mrope via `build_mrope` (`text.rs:1891-1930`) —
     `[11,11,10]` works as-is; partial rotary pass-through (`text.rs:1141-1186`);
   - the interval-4 layer pattern via `is_full_attention_layer`
     (`text.rs:1786-1788`) — exactly the 35B layout;
   - the 3-domain cache contract `qwen35_composite_cache_contract`
     (`cache.rs:24-164`): paged full-attn KV + F32 recurrent (DeltaNet) state +
     conv state, `prefix_shareable: false`;
   - ChatML prompt rendering with thinking handling that already matches the
     35B's contract: `<think>\n` when thinking enabled, empty think block when
     disabled (`chat.rs:1370-1380`), assistant-history reasoning split
     (`chat.rs:1399-1412`).
   - Constraint: the loader requires GGUF arch string exactly `"qwen35"` and is
     dense-only (`chat.rs:539-547`).

2. **`Qwen3MoeChat`** (`models/architectures/qwen3/core.rs` +
   `models/shared/moe.rs`) — the DS10 sparse-expert groundwork:
   - `SparseMoeConfig {num_experts, num_experts_per_tok, norm_topk_prob}`
     (`moe.rs:27-48`), fail-closed resolution (`core.rs:175-199`);
   - `SparseMoeDispatcher::dispatch` (`moe.rs:143-218`): F32 softmax → argsort →
     top-k → optional renorm → host gather → per-expert `ExpertSet::apply_expert`
     → weighted `index_add`; `ExpertActivationCounters` histograms (`moe.rs:77-108`);
   - expert loading from GGUF fused tensors via byte-split
     (`shared/weights/gguf.rs:483-517`, quantized residency preserved) **and**
     from HF per-expert safetensors naming (`core.rs:2395-2437`);
   - per-expert `QMatMul` execution with the F32-activation-cast discipline on
     CUDA/Metal (`core.rs:1035-1105`).

3. **`Qwen38Chat` / `Qwen38_27B_FP8`** (`models/architectures/qwen38/native.rs`
   and siblings) — Qwen3.8-27B-FP8 is *itself* a hybrid GDN + gated-attention
   model ingested from native FP8 safetensors. This is the single most important
   precedent:
   - indexed-shard mmap loader reading `safetensors::Dtype::F8_E4M3` weights with
     BF16 `*.weight_scale_inv` companions, block shape ceil(rows/128) ×
     ceil(cols/128) (`native.rs:1528,1641-1740`, `native/loading.rs:323-393`);
   - dequant-on-load with per-backend materialization policy
     (`qwen38/chat.rs:217-233`): CPU → expanded F32, Metal → expanded F16,
     CUDA → BF16/F16 dense or requantized packed **Q8_0** `QMatMul`
     (`native.rs:899-980`); a native CUDA FP8 kernel exists
     (`kernels/cuda/fp8.rs`) but is deliberately not the default and not
     runtime-certified;
   - strict `config.json` validation including `quantization_config`
     `{quant_method: fp8, fmt: e4m3, activation_scheme: dynamic,
     weight_block_size: [128,128]}`, `layer_types`, mrope parameters
     (`native.rs:380-530`) — currently pinned to the 27B geometry;
   - downloader index-closure bundle contract with pinned revision manifest
     (`artifacts/downloader.rs:1719-1768`, `266-303`, `684-740`);
   - dedicated memory inventory pricing (`qwen38_memory.rs:64-199`) and a
     custom load-authorization block-math estimate
     (`runtime/lifecycle/load.rs:234-286`).

Also relevant: Candle 0.11 has **no FP8 GGML type**, so FP8 can only enter
through safetensors (as qwen38 does) — an FP8 GGUF is impossible today; and
`estimate_from_tensor_inventory` (load.rs:432-478, the DS10-A1 admission scratch
fix) already accounts for MoE-scale per-tensor slack (32 KiB × tensor count),
which matters here (~37k expert tensors ≈ 1.1–1.2 GiB of scratch).

## 3. Gap analysis

- **G1 — No Qwen3.5-MoE family.** `Qwen35Chat` is dense-only; `Qwen3MoeChat`
  rides the plain-qwen3 attention stack. No code path composes "qwen35 hybrid
  backbone + sparse MoE FFN". Per `tasks/lessons.md` (2026-09: "Do not collapse
  a new checkpoint into an existing product/runtime family…"), this gets a
  **dedicated family**, not an extension of an existing one.
- **G2 — No shared expert anywhere** (`grep shared_expert` → zero hits).
  `SparseMoeConfig` has no field for it; the dispatcher returns only routed
  output.
- **G3 — The qwen38 native-FP8 loader is hard-pinned** to the Qwen3.8-27B
  geometry and `model.language_model.` tensor-scope prefix
  (`native.rs:248-560`). The 35B needs its own config parser
  (`Qwen3_5MoeForConditionalGeneration`, nested `text_config`, MoE fields,
  shared-expert size, `mamba_ssm_dtype`, vision-tower tolerance) and scope
  resolution (35B language tensors at `model.` — verify actual shard key prefixes
  at implementation).
- **G4 — Routing config.** Dispatcher is softmax-only; `scoring_func` /
  `norm_topk_prob` defaults for `qwen3_5_moe` must be pinned from transformers
  before parity claims.
- **G5 — Registration sweep.** A new variant touches ~19 sites (DS10 Appendix A
  list): variant/family enums (`catalog/variant.rs:8-132`), parse/heuristics
  (`variant.rs:166-461`), `is_chat`, `chat_capabilities()` (`catalog/metadata.rs:828-852`),
  `is_enabled`/`is_quantized` (`metadata.rs:1018-1060`), size/memory hints
  (`metadata.rs:514-628`), CUDA support tables (`catalog/cuda_support.rs:338-560`),
  family registration (`models/families/mod.rs:56-276`, test-enforced),
  loader registry (`models/registry.rs:489-520`), `model_family_name`
  (`registry.rs:5561-5586`), `NativeChatModel` enum + state-contract dispatch
  (`registry.rs:3662-3696`), adapters gates (`runtime/adapters.rs:144-220,435-446`,
  `adapters/loaded.rs:874-888`), downloader (`artifacts/downloader.rs`),
  conformance frozen counts (`conformance.rs:11-12`: **52 variants / 76
  capability bindings → 53 / 77+**), admin API, worker chat allowlist
  (`izwi-serving-worker/src/main.rs:577-590`), UI `/models` metadata catalog
  (lessons: backend registration without `MODEL_DETAILS` is invisible).
- **G6 — Memory/admission for FP8-MoE at 35B scale.** Generic admission prices
  resident = source bytes, which is only true if quantized residency is kept;
  the qwen38-style representation math must be reproduced with MoE element
  counts. Envelope math in §7.
- **G7 — Metal has no BF16 custom kernels** (`kernels/metal.rs` — zero BF16
  mentions; fused silu_mul accepts F32|F16 only, `metal.rs:4479-4482`), and F16
  V-overflow is a documented NaN hazard (lessons 2026-09-06). The materialization
  policy must respect this.
- **G8 — Feature-table arms.** DS1 prefix-reuse table (`catalog/prefix_reuse.rs:121-207`)
  is an exhaustive match — the new family needs a cell (expected: `NotEnabled`,
  hybrid reuse unproven, same as `Qwen35Chat`); json_object handler allowlist
  (`api/completions.rs:894-910`) currently admits only Qwen3Chat|Gemma3Chat|Lfm2Chat;
  `chat_capabilities` has a `_ => None` catch-all (silent-thinking-UI hazard);
  `is_enabled` has a `_ => !is_quantized()` catch-all that would misclassify a
  quantized variant as enabled.
- **G9 — MTP head.** The checkpoint carries a 1-layer MTP module (BF16), but no
  qwen35/qwen3.5 MTP forward exists (only `qwen38/mtp.rs`). The DS9 speculative
  machinery (`engine/execution.rs:1245`, `mtp_in_continuous_enabled()`) is
  model-generic, so this is an add-the-head phase, not new scheduler work.

## 4. Implementation plan

Seven independently reviewable vertical slices, each committed separately with
its own evidence (lessons: verify and commit each slice before the next). The
variant lands **catalog-disabled** per ADR 0008 posture (a MoE variant is
catalog-disabled until real-checkpoint activation evidence exists — exactly the
`Qwen3Moe30bA3bGguf` precedent).

Naming (mirrors `Qwen3827BFp8`): family `Qwen35MoeChat`, variant
`Qwen35Moe35BA3BFp8`, public id "Qwen3.5-35B-A3B-FP8", architecture module
`models/architectures/qwen35moe/`.

### Phase 0 — Family identity + catalog registration sweep (disabled variant)
- New `ModelVariant::Qwen35Moe35BA3BFp8` + `ModelFamily::Qwen35MoeChat` across
  every G5 site; `chat_capabilities` arm: thinking default-on +
  `preserve_thinking`, no reasoning-effort ladder (that contract is Qwen38-only);
  `is_enabled` → explicit `false` (do **not** rely on the catch-all);
  `is_quantized` → true; `estimated_size`/`memory_required_gb` from the FP8 block
  math (§7); repo/revision pin slot like qwen38 (`metadata.rs:276-277`).
- Conformance counts 52→53 / 76→77+; worker chat allowlist gets the variant;
  UI `MODEL_DETAILS` entry; CUDA support table arms (fail-closed until evidence).
- **Evidence:** workspace build + conformance test + catalog unit tests. Nothing
  else may claim activation.

### Phase 1 — Native FP8 safetensors ingestion
- New `models/architectures/qwen35moe/native.rs` (+ loading submodule) lifting
  the family-agnostic primitives from qwen38: shard mmap via
  `model.safetensors.index.json`, E4M3 + BF16 `weight_scale_inv` block-128
  decode (`decode_e4m3fn`, `load_block_fp8_f32`, `materialize_block_fp8_raw`),
  `materialize_q8_projection(_group)`, scale finiteness validation, and the
  `ProjectionMaterialization` policy seam. Do **not** reuse the qwen38 module
  itself (dedicated-family lesson); lift code into a shared weights module if
  that is cleaner.
- Config parser for `Qwen3_5MoeForConditionalGeneration`: nested `text_config`,
  `full_attention_interval`, `layer_types`, linear-attention head fields,
  mrope/rope_parameters, MoE fields (256/8/512 + `shared_expert_intermediate_size`),
  `mamba_ssm_dtype`, `quantization_config` validation, `modules_to_not_convert`
  awareness; tensor-scope resolution (`model.` vs `model.language_model.` —
  confirm against real shard keys at hardware handoff); skip vision-tower and
  MTP tensors (text-only scope; MTP tensor names recorded for Phase 6).
- Downloader: index-closure file plan (config.json, generation_config.json,
  chat_template.jinja, tokenizer.json, tokenizer_config.json, vocab.json,
  merges.txt, safetensors index + all shards), `require_exact_bundle`, pinned
  `ArtifactManifest`; tolerate extra repo files (preprocessor configs) rather
  than rejecting the official bundle; no `mmproj` requirement (text-only).
- A synthetic-geometry escape hatch like `IZWI_ALLOW_SYNTHETIC_QWEN38_GEOMETRY`
  for fixture checkpoints.
- **Evidence:** unit tests on synthetic E4M3 tensors with known-value dequant
  (exact vectors), config-parser test against the real `config.json` committed
  as a fixture, loader test on a tiny synthetic FP8 checkpoint, failure-injection
  (missing scale tensor, wrong block size, non-finite decoded value).

### Phase 2 — Architecture module: hybrid backbone + sparse MoE
- `models/architectures/qwen35moe/`: compose the qwen35 GDN mixer / gated full
  attention / mrope components (parameterized; visibility adjustments as needed)
  with a new `Qwen35SparseMlp` MoE block: FP8-excluded router (BF16 dense) →
  `SparseMoeDispatcher` (ported to support shared-expert addition; scoring mode
  pinned per G4) → per-expert materialized projections via `ExpertSet` →
  weighted sum **+ shared-expert dense MLP output added unconditionally**.
- MoE-ness decided from checkpoint config (`num_experts` present), mirroring
  `Qwen3Layer::load_gguf` (`core.rs:2632-2640`).
- GGUF fixture path: extend the GGUF config keys (`qwen35moe` arch string or
  `qwen35.*` + `expert_count`/`expert_used_count`/`expert_feed_forward_length`)
  and the fused-expert byte-split for tiny synthetic fixtures — this is how CI
  exercises MoE mechanics without the 35B download (DS10 fixture pattern).
- Chat wrapper `Qwen35MoeChatModel` with its own `ChatDecodeState`; state
  contract = `qwen35_composite_cache_contract` (3 domains; recurrent/conv F32
  per `mamba_ssm_dtype`); context from metadata (262,144).
- **Evidence:** fixture forward parity CPU vs Metal on the tiny GGUF fixture;
  end-to-end generation on the fixture; full existing suite green (registry-wide
  integration contracts per lessons).

### Phase 3 — Runtime gates and execution routes
- Add the family to `chat_sequence_execution` (`runtime/adapters.rs:435-446`),
  `is_continuous_physical_chat` (`adapters/loaded.rs:874-888`),
  `family_inference_state_policy` (`adapters.rs:144-220`); continuous-batch
  adapter leg; prefix cursor plumbing (cached_tokens stays 0 until DS1 engages —
  honest by construction).
- **Evidence:** continuous-batch fixture test; sequence-execution dispatch test
  exercised through the **public** start/push/finish entry points (lessons:
  resolver-only tests miss legacy fallbacks).

### Phase 4 — Chat surface and API
- Thinking: render_prompt thinking path already exists in qwen35 chat rendering —
  port/reuse; `enable_thinking=False` → empty think block; assistant-history
  `<think>` re-render with `preserve_thinking`; EOS set = `<im_end>` +
  `<|endoftext|>`; per-step logprobs via the shared sampler (automatic once the
  decode loop uses `ChatSampler::sample_with_logprobs`).
- json_object: wire grammar at sampler construction (tokenizer + per-family EOS
  set, pattern `qwen3/chat.rs:461`) **and** add the family to
  `ensure_response_format_supported` (`api/completions.rs:894-910`) in the same
  slice — the two must land together or the API 400s a grammar-capable sampler
  (the exact inconsistency `Qwen3MoeChat` is in today; note it, fix only the new
  family here).
- **Evidence:** API-level tests: thinking render both enable/disable paths,
  json_object e2e on the fixture, logprobs shape validation.

### Phase 5 — Memory/admission + distributed feature posture
- Dedicated resource plan (`qwen38_memory.rs:64-199` pattern generalized to
  `qwen35moe_memory.rs`): per-backend persistent-representation math with MoE
  element counts, per-tensor slack at ~37k tensors, load-peak accounting, host
  staging sizing; wire into `model_resource_plan` selection
  (`runtime/lifecycle/load.rs:1249-1284`). Catalog byte pin matches.
- DS1 prefix-reuse table cell: `Qwen35MoeChat → NotEnabled` on all backends
  (hybrid recurrent/conv reuse unproven — same rationale as `Qwen35Chat`,
  `prefix_reuse.rs:179-181`); compile-visible, documented.
- DS4 KV host-offload: dormant by construction (host tier builds only when
  committed prefix pages exist and the contract declares `prefix_shareable:
  false`); document. DS10 PD page-transfer: out of scope for hybrid families
  (no page codec for recurrent/conv domains) — documented, like `Qwen35Chat`.
- DS5/6/7: deployment manifest entry, rollout plan, and `[autoscaling]` block
  examples with **truthful** `host_memory_limit_bytes` from the Phase 5 math
  (the ledger trusts declared numbers).
- **Evidence:** admission unit tests with the real geometry (geometry-only
  regressions per lessons 2026-09-06); kill-switch tests.

### Phase 6 — MTP speculative decoding (optional, after core lands)
- The checkpoint ships a 1-layer BF16 MTP module. Port the `qwen38/mtp.rs`
  pattern (`forward_steps_batch`) to the qwen35moe hidden state; wire into the
  existing DS9 per-row-depth machinery (`mtp_in_continuous_enabled()`,
  bootstrap publish / can-train observe gating). Keep opt-in with kill switch;
  no default-on without hardware evidence (lessons: "do not equate a loaded
  speculative head with an effective speedup").
- **Evidence:** portable contract tests on fixture MTP weights; speedup claims
  deferred to hardware handoff.

### Phase 7 — Portable evidence bundle + hardware handoff doc
- Full portable gate: CPU test matrix, CUDA compile (`--no-run` driverless CI
  tier), failure-injection suites, synthetic FP8 checkpoint evidence runner.
- `docs/dev/` handoff note: exact download command + revision pinning, SHA
  verification, activation checklist per backend (numerical parity vs HF/llama.cpp
  reference on fixed prompts, memory footprint, latency slopes across tokens and
  turns), the ADR 0008 activation gate, and the order of post-activation flips
  (catalog enable → per-backend certification matrix → DS1 cell re-evaluation →
  cuda_support table promotion).

## 5. Feature posture matrix (DS0–DS10) for the new family

| Workstream | Posture for Qwen35MoeChat | Action |
|---|---|---|
| DS0/DS1 prefix reuse (catalog-auto) | `NotEnabled` cell, all backends; hybrid reuse unproven | Phase 5 table arm; re-evaluate post-activation |
| DS2 cache affinity / pinning | Free — keys on deployment identity, not family | None |
| DS3 realtime voice | Text-only LLM: never a realtime stage (relay closes `TaskKind::Chat` with ProtocolViolation; voice pipelines strip `<think>` before TTS) | Add variant to worker chat allowlist only |
| DS4 hierarchical KV offload | Dormant (no committed pages; `prefix_shareable: false`) | Document; no code |
| DS5 fleet authority | Model-id agnostic (manifest strings) | Deployment entries only |
| DS6 blue-green | Model-id agnostic | Config only |
| DS7 autoscaling | Ledger trusts declared budgets | Truthful `host_memory_limit_bytes` from Phase 5 |
| DS9 cached_tokens | 0 until DS1 engages — honest | None beyond DS1 posture |
| DS9 logprobs | Via shared sampler | Decode loop uses `sample_with_logprobs` |
| DS9 json_object | Grammar-capable; allowlist + sampler wiring must land together | Phase 4 |
| DS9 MTP speculative | Checkpoint has MTP weights; machinery generic | Phase 6 |
| DS10 MoE serving | Core of this plan (256/8+1 shared) | Phases 1–2 |
| DS10 PD KV page-transfer | Out of scope for hybrid (no page codec) | Document |

## 6. Silent-failure watchlist (from the touchpoint sweep)

Things that would misbehave without a deliberate arm — all covered by phases
above, listed so review can check each:

1. `chat_capabilities` `_ => None` → thinking UI never appears
   (`catalog/metadata.rs:828-852`).
2. Missing `chat_sequence_execution` / `is_continuous_physical_chat` arms → chat
   loads but runs without sequence execution / continuous batching
   (`runtime/adapters.rs:435`, `adapters/loaded.rs:874`).
3. `is_enabled` catch-all `_ => !is_quantized()` → would silently enable a
   quantized variant (`metadata.rs:1018-1060`).
4. json_object allowlist omission → 400 despite a grammar-capable sampler
   (`api/completions.rs:894-910`).
5. Worker boot chat allowlist (`izwi-serving-worker/src/main.rs:577-590`) → loud
   boot failure in distributed setups if missed.
6. Memory authorization via generic `memory_required_gb` → mis-priced load peak
   for FP8-MoE (`runtime/lifecycle/load.rs:186-292`).
7. Warm-up deadline: a 35B load + warm-up must fit the worker boot deadline
   (`serving-worker/runtime.rs`, 30 s warm-up) — verify at handoff.

## 7. Memory envelopes (representation math, from checkpoint facts)

**Updated 2026-09-29 (Phase 5):** the numbers below are the *implemented*
admission math (`runtime/lifecycle/qwen35moe_memory.rs`), which derives its
element counts from the loader's own pinned tensor plan — FP8 projections
32,862,371,840 elements, dense tensors 1,798,238,848 elements, 62,243 tensors
counting scale companions. The materialization policy is the Phase 2 loader's
(`projection_residency_policy`): CPU packs projections as Q8_0 and keeps dense
tensors F32 (decision D1 resolved), Metal expands F16, CUDA expands BF16. This
supersedes the original projections in this section (CUDA Q8_0 was the qwen38
default; the qwen35moe CUDA route expands BF16 like its Metal sibling — the
native CUDA FP8 kernel remains non-default and uncertified).

| Backend | Resident representation | Weights resident | Load peak |
|---|---|---|---|
| CPU | Q8_0 projections + F32 dense | 42,109,225,472 B ≈ 39.2 GiB | ≈ 42.1 GiB (+1 GiB conversion scratch + 1.9 GiB per-tensor slack) |
| Metal | F16 expanded | 69,321,221,376 B ≈ 64.6 GiB | ≈ 67.5 GiB (+ same scratch terms) |
| CUDA | BF16 expanded | 69,321,221,376 B ≈ 64.6 GiB | device ≈ 66.7 GiB + 8 GiB host staging |

Source checkpoint bytes: 36,458,849,536 (FP8 1 B + dense BF16 2 B), within 3%
of the catalog `estimated_size` pin (37,470,000,000). The catalog
`memory_required_gb` (140.0) stays the deliberately conservative worst-case
CPU-F32-expansion hint; backend-specific admission replaces it. Scale
companions are consumed during dequantization and never become resident.
Per-tensor instantiation slack (32 KiB × 62,243 ≈ 1.9 GiB) is material at MoE
scale and included in every load peak.

**DS5/6/7 deployment budget examples (truthful `host_memory_limit_bytes`).**
The ledger trusts declared numbers; derive them from the table above:

- CPU worker: `host_memory_limit_bytes ≥ 46 GiB` (42.12 GiB load peak + KV and
  scheduler headroom), one resident replica.
- Metal worker: `host_memory_limit_bytes ≥ 70 GiB` unified (64.6 GiB resident +
  load scratch + KV), 96–128 GB-class machine.
- CUDA worker (48 GB card tier is **not** sufficient under the current BF16
  expansion): `host_memory_limit_bytes ≥ 76 GiB` (66.7 GiB device peak + 8 GiB
  host staging + KV), 80 GB-class card. Promotion of the native CUDA FP8
  kernel (a separate, separately-evidenced change) would revisit this.

KV at 262K context is priced separately by the context fitter: 10 full-attn
layers × 2 KV heads × 256 dim ≈ 5.2 GB F16/F32-class + 30 DeltaNet recurrent
states ≈ 62 MB F32 fixed + conv state. Publish fitted context, never the
262K theoretical ceiling as resident.

Load-peak scratch (historical projections kept for review context): the
inventory-based estimate prices `max(largest-tensor dequant bound, 32 KiB ×
tensor count)` per the DS10-A1 fix.

## 8. Open decisions (recommendations made, user can override)

- **D1 — CPU residency policy**: **RESOLVED (Phase 2/5, 2026-09-29)** — CPU
  packs projections as packed Q8_0 with F32 dense state (`PackedQ8_0`
  residency), ≈ 39.2 GiB resident. F32 expansion (~140 GB) stays only as the
  catalog's conservative `memory_required_gb` hint.
- **D4 — Routing mode**: **RESOLVED (Phase 2, 2026-09-29)** — pinned from the
  transformers `qwen3_5_moe` source: F32 softmax → top-k → in-top-k renorm; no
  sigmoid, no correction bias. See the routing-mode pin at the top of this doc.
- **D2 — GGUF ingestion scope**: GGUF support recommended **only** as the tiny
  fixture path for CI (synthetic checkpoints). The official FP8 checkpoint is
  native safetensors; no production GGUF path, no conversion script.
- **D3 — Vision tower**: out of scope (text-only), loader skips vision tensors
  but must accept the official bundle layout. Multimodal serving would be a
  separate future plan.
- **D5 — YaRN / >262K contexts**: out of scope; native 262,144 is the metadata
  ceiling, actual context is fitter-published.

## 9. Honest limits (stated up front)

- No runtime, quality, or performance claims until the server-side hardware
  handoff produces exact-SHA evidence (ADR 0008; lessons 2026-09). The variant
  ships catalog-disabled.
- Text-only: no image/video input despite the vision tower in the checkpoint.
- No PD page-transfer, no prefix-reuse reuse claims for the hybrid state
  domains until separately proven.
- Metal 96 GB-class machines and 48 GB CUDA cards are the *projected* minimums
  from representation math — they are estimates until measured.

## 10. Sources

- HF model card: https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8 (architecture,
  quantization, thinking/serving sections)
- HF `config.json` (fetched 2026-09-29): `Qwen3_5MoeForConditionalGeneration`,
  MoE 256/8 + shared 512, MTP 1 layer, mrope [11,11,10], FP8 block 128,
  `modules_to_not_convert` list
- Code sweep of branch `production-serving` at 65b5dabc (three exploration
  reports, 2026-09-29): qwen35/qwen3/qwen38 architecture + catalog + runtime
  touchpoints, with file:line citations preserved above
