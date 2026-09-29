# Qwen3.5-35B-A3B-FP8 hardware handoff

This is the activation checklist for the `Qwen35MoeChat` family
(`Qwen3_5MoeForConditionalGeneration`, `Qwen/Qwen3.5-35B-A3B-FP8`). The
implementation on `production-serving` is complete to the portable boundary:
every commit is backed by synthetic-fixture, reference-math, failure-injection,
and real-process evidence, and **no runtime, quality, or performance claim is
earned until the checks in this document run against the exact-SHA checkpoint
on target hardware** (ADR 0008 posture; the variant ships catalog-disabled).

Planning doc: `docs/dev/QWEN35_35B_A3B_FP8_SUPPORT_PLAN.md`.

## 1. What has landed (portable evidence bundle)

| Phase | Commit | Evidence |
|---|---|---|
| Catalog registration (disabled) | `4935c9af` | ~19-site variant sweep, conformance counts 53/77+, worker chat allowlist, UI metadata, explicit `is_enabled = false` |
| Native block-FP8 ingestion | `520e624a` | Known-value E4M3 dequant vectors, real `config.json` fixture test, tiny synthetic FP8 checkpoint load, failure injection (missing scale, wrong block size, non-finite values) |
| Hybrid trunk + sparse MoE + registry | `40342690` | Fixture CPU/Metal greedy parity, e2e fixture generation, native-FP8 synthetic trunk forward (finite logits, per-layer expert histograms), full `izwi-core` lib suite green |
| Runtime gates | `bb455280` | Sequence execution, continuous-chat adapter, managed-KV provider route opened for the family; real worker process over HTTP (load, warm-up, worker lane, managed-KV status, public gateway streaming + non-streaming) with a synthetic native FP8 bundle |
| Chat surface | `24fec63f` | Thinking render contract pinned on the MoE wrapper (default on / off / history split), `json_object` grammar wired into the hybrid decode states + API allowlist in the same slice, DS9.3 logprobs per non-stop step, gateway `json_object` request reaches the model |
| Memory/admission | `999a146c` | Representation plan derived from the loader's own pinned tensor plan (exact element counts pinned by test), per-backend residency and load peaks, CUDA host staging, fixture-mode inventory pricing |

Known pre-existing flake (not this family): `fish_s2` `dac batched_transpose`
fails ~1/3 parallel full runs and passes solo.

## 2. Download and verification

```bash
# Models root: IZWI_MODELS_DIR of the serving host.
izwi pull Qwen3.5-35B-A3B-FP8
```

- Bundle contract: `config.json`, `generation_config.json`,
  `chat_template.jinja`, `tokenizer.json`, `tokenizer_config.json`,
  `vocab.json`, `merges.txt`, `preprocessor_config.json`,
  `video_preprocessor_config.json`, `model.safetensors.index.json` and every
  shard the index closes. The downloader enforces the exact index closure
  (`qwen35_moe_bundle_is_complete`) and the pinned revision.
- Pinned revision: `9d1823d2dee688a6b25e77009dc727688c44936e`
  (`ModelVariant::QWEN35_MOE_35B_A3B_FP8_ARTIFACT_REVISION`, HF repo state
  2026-04-24). Re-verify `config.json` against the plan-doc pins before
  activation: `Qwen3_5MoeForConditionalGeneration`, 40 layers,
  `full_attention_interval: 4`, hidden 2048, vocab 248,320, MoE 256/8 + shared
  512, mrope `[11,11,10]`, FP8 block `[128,128]`, `modules_to_not_convert`.
- **Tensor-scope confirmation at handoff**: the loader canonicalizes language
  tensors under `model.language_model.*` / `model.layers.*` / `model.` and
  skips `model.visual.*` and `mtp.*`. Confirm the real shard key prefixes
  match (the plan flagged `model.` vs `model.language_model.` for
  verification against actual shard keys; both layouts are canonicalized, but
  verify, don't assume).
- **Never set `IZWI_ALLOW_SYNTHETIC_QWEN35_MOE_GEOMETRY` on a serving host.**
  Production loads fail closed on anything but the published geometry; the
  escape hatch exists for CI fixtures only.

## 3. Per-backend activation checklist

Admission numbers below come from the implemented representation plan
(`runtime/lifecycle/qwen35moe_memory.rs`); measure the actual footprint and
treat deviations from the plan as defects to fix before activation, not
tolerance to absorb.

### 3.1 CPU (Q8_0 projections + F32 dense)

- Resident weights ≈ 39.2 GiB; load peak ≈ 42.1 GiB (+ KV + scheduler
  headroom). `host_memory_limit_bytes ≥ 46 GiB` for one resident replica.
- Greedy parity: fixed prompts must match the synthetic-fixture parity
  discipline used in Phase 2 (seeded, deterministic, byte-identical across
  reruns) and the HF/llama.cpp reference outputs within tokenizer tolerance.
- Confirm Q8_0 requant parity is acceptable on real weights (the fixture
  cannot prove projection-quantization quality).

### 3.2 Metal (F16 expanded)

- Resident weights ≈ 64.6 GiB unified; `host_memory_limit_bytes ≥ 70 GiB`;
  96–128 GB-class machine.
- Watch the F16 V-overflow NaN hazard (G7; lessons 2026-09-06): validate raw
  logits are finite across long generations before trusting outputs.
- Fused-silu/attention kernels accept F16 — confirm no rank/layout Metal
  rejections on the real 256-expert shapes (fixture scale cannot surface
  large-tensor layout faults).

### 3.3 CUDA (BF16 expanded)

- Resident weights ≈ 64.6 GiB → **a 48 GB card is not sufficient**; the
  activation tier is an 80 GB-class card. Device load peak ≈ 66.7 GiB plus
  8 GiB host staging: `host_memory_limit_bytes ≥ 76 GiB`.
- The native CUDA FP8 kernel (`kernels/cuda/fp8.rs`) stays non-default and
  uncertified. Promoting it is a separate change with its own
  source-review/portable/compile/runtime evidence tiers; a Q8_0-requant CUDA
  residency (qwen38-style, ~36 GiB class) is the candidate to evaluate there.
- Record physical VRAM peak, swap/pageouts, per-token latency, and allocation
  counts across tokens and turns before calling the route stable.

### 3.4 Certification matrix (every backend)

1. Load via normal server admission (not synthetic mode); confirm published
   context is fitter-fitted (never the 262,144 ceiling) and KV domains price
   the composite contract (paged full-attn + F32 recurrent + conv).
2. Numerical parity vs reference on fixed prompts (HF transformers
   `qwen3_5_moe` and/or llama.cpp), including routing histograms: the top-8
   expert selection distribution must be non-degenerate (no single-expert
   collapse).
3. Multi-turn memory and latency slopes across enough tokens/turns; warm-up
   must fit the worker boot deadline (the 30 s warm-up watchlist item —
   a ~35 GB load + warm-up on the target disk/CPU is the risk; measure before
   fleet rollout).
4. json_object constrained decoding on real weights; logprobs shape; thinking
   render on/off through the public API.
5. 40/48 GB-tier claims stay forbidden until measured (see 3.3).

## 4. ADR 0008 activation gate and flip order

The variant is catalog-disabled until real-checkpoint activation evidence
exists. Post-activation flips, in order:

1. **Catalog enable** — flip `is_enabled` for `Qwen35Moe35BA3BFp8` with the
   evidence bundle (exact SHA, device UUIDs, parity artifacts).
2. **Per-backend certification matrix** — CPU/Metal/CUDA rows from §3.4.
3. **DS1 prefix-reuse re-evaluation** — the family's DS1 cell is
   `NotEnabled` on all backends (hybrid recurrent/conv reuse unproven).
   Re-evaluate only with its own evidence; cached_tokens stays honestly 0
   until then.
4. **CUDA support table promotion / cuda_support arms** — from fail-closed to
   the certified tiers.
5. UI visibility and any default posture changes land last, per cell, with
   kill switches retained.

Dormant-by-construction postures (no code change needed): DS4 host offload
(host tier builds only for committed prefix pages; the contract declares
`prefix_shareable: false`), DS10 PD page-transfer (no page codec for
recurrent/conv domains). DS2/DS5/6/7 are model-id agnostic; deployment
manifests and `[autoscaling]` blocks take the truthful budgets from §3.

## 5. MTP (Phase 6) — deliberately deferred, scoped

The checkpoint ships a 1-layer BF16 MTP module. The current loader skips and
records `mtp.*` tensors without validating their manifest, because the real
tensor contract (dense vs MoE FFN inside the MTP layer, attention type, exact
names/counts) could not be pinned without the checkpoint. The qwen35moe
config parser intentionally does not parse MTP fields yet.

Unblock order when the download is available:

1. Pin the real MTP tensor manifest from the downloaded shards and add it to
   the loader's validated plan (mirror `qwen38::native::mtp_tensor_specs`).
2. Port the `qwen38/mtp.rs` pattern (`Qwen38MtpHead`, `forward_steps_batch`,
   `AdaptiveMtp`) onto the qwen35 hybrid hidden state.
3. Wire the executor managed-cache MTP domain and the DS9 per-row-depth
   machinery (`mtp_in_continuous_enabled`, bootstrap publish / can-train
   observe gating), opt-in with kill switches.
4. Portable contract tests on fixture MTP weights; speedup claims only from
   the hardware handoff. Never default-on without measured latency guidance.

## 6. Silent-failure watchlist — resolution status

1. `chat_capabilities` arm — explicit thinking default-on + preserve flags
   (Phase 0). 2. Sequence-execution / continuous-chat / managed-KV gates —
   opened and process-proven (Phase 3). 3. `is_enabled` explicit false, not
   the quantized catch-all (Phase 0). 4. json_object allowlist + sampler
   grammar landed together (Phase 4). 5. Worker boot chat allowlist contains
   the variant with its deployment id (Phase 0). 6. Memory authorization via
   the dedicated representation plan (Phase 5). 7. Warm-up deadline —
   **open**, measure at handoff (§3.4 step 3).

## 7. Honest limits

- No runtime, quality, latency, or memory claim in this document is measured;
  all numbers are representation math validated by unit tests against the
  loader's plan, not by a loaded checkpoint.
- Text-only serving: the vision tower in the checkpoint is skipped. No image
  or video input. Multimodal serving is a separate future plan.
- No prefix-reuse, PD page-transfer, or speculative-decoding claims for the
  hybrid state domains until separately proven (§5).
- YaRN / >262K contexts are out of scope; published context is always
  fitter-fitted.
