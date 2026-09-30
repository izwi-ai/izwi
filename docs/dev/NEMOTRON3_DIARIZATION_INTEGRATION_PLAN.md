# Nemotron-3-Diarization — integration research & plan (2026-09-30, planning only)

Research session — no code written. Deliverable is this plan; implementation
sessions should rotate the checklist into `tasks/todo.md` when approved.

## 0. Verdict

**Yes — it is not merely "like" Sortformer, it *is* Sortformer-family.**
NVIDIA's card loads it through the NeMo class `SortformerEncLabelModel`, it
inherits Streaming Sortformer's Arrival-Order Speaker Cache (AOSC) + FIFO
streaming machinery, and it resolves speaker permutation the same way
("ordering its output channels according to each speaker's first arrival").
NVIDIA's own benchmark table compares it against
`diar_streaming_sortformer_4spk-v2.1` — **the exact checkpoint Izwi already
serves end-to-end**. Integration is therefore "second checkpoint of the
existing diarization stack", not a new paradigm.

There are exactly two real architecture deltas versus the integrated v2.1:

1. **Encoder**: plain 31-layer Transformer + full RoPE (d=512, 8 heads,
   GELU, FFN 2048) instead of v2.1's Conformer encoder + multiscale fusion +
   sortformer transformer stack. Izwi's loader is hard-wired to
   `SortformerConformerEncoder` and its workspace-topology gate pins one
   frozen production topology — both must become checkpoint-driven.
2. **Speaker count**: 8 output channels vs the compile-time
   `MAX_SUPPORTED_SPEAKERS = 4` const woven through the current module.

Everything downstream of the model (product surface, streaming state
machinery, post-processing, pipeline) already exists and is config-driven.

## 1. Model facts (verified against the HF card + config.json)

- Repo `nvidia/Nemotron-3-Diarization`, released 2026-09-23.
  License **OpenMDW-1.1** — same license family as the already-integrated
  `Nemotron-3.5-ASR-Streaming-0.6B`.
- ~99.2M params, F32, single `model.safetensors`. **A `.nemo` file ships in
  the repo** (`Nemotron-3-Diarization.nemo`, ~0.4 GB est.) plus a q8_0 GGUF
  (~107 MB) and `processor_config.json`. No tokenizer files (frame-level
  classification, no text tokens).
- config.json: `model_type nemotron3_diarization`,
  arch `Nemotron3DiarizationForAudioFrameClassification`; audio_config:
  31 layers, hidden 512, 8 heads, 8 KV heads, GELU, intermediate 2048,
  `num_mel_bins: 128`, `subsampling_factor: 8` (10 ms mel → 80 ms frames),
  full RoPE (θ=10000, partial factor 1.0), `max_position_embeddings: 5000`;
  head_config: hidden 192, `num_speakers: 8`, conv upsampler back to 10 ms.
- Streaming config (all in 80 ms frame units, same knobs Izwi already
  parses): chunk 340, right context 40, fifo 40 (top level) / 264
  (streaming), speaker cache 264, update period 222 (300 top-level),
  pred_score_threshold 0.25, scores_boost_latest 0.05, strong_boost_rate
  0.75, weak_boost_rate 1.5, min_positive_scores_rate 0.5, 1 silence frame
  per speaker.
- Latency profiles: offline 30.4 s, low 1.04 s, very-low 0.64 s,
  ultra-low 0.32 s (0.32 s supported floor; 80 ms input buffer possible).
- Output: `[T, 8]` per-speaker activity probabilities at 10 ms stride,
  post-processed into segments (`["speaker1", 0.51, 12.62]`) — not RTTM
  natively, but Izwi's response types are already RTTM-shaped.
- Usage: `SortformerEncLabelModel.from_pretrained(...)` /
  `.diarize(audio=[...])`; Transformers-native
  `AutoModelForAudioFrameClassification` with `set_streaming_mode()` and a
  `speaker_cache` passed between chunks; `NeMo-Speech.cpp` C++ runtime.
- Perf vs baseline v2.1: DIHARD III full 12.73 vs 19.09 DER; NOTSOFAR1 MHM
  full 6.77 vs 21.77. RTFx 1340 eager / 4385 compiled (batch 1, BF16,
  Blackwell RTX PRO 5000).
- Constraints: max 8 speakers, 16 kHz mono, up to 8×8-A100-trained; card
  says optimized for NVIDIA GPUs but the model is small enough for
  CPU/Metal (Izwi's existing rule — CUDA native, CPU otherwise — applies).

## 2. What Izwi already has (reuse map)

- **Full diarization product surface, shipped**: `/v1/diarizations*` +
  `/v1/speech-to-text/jobs?job_kind=diarization` APIs
  (`crates/izwi-server/src/api/diarization/{mod,handlers}.rs`,
  `speech_text_upload.rs`, `transcription/unified_{read,write}.rs`),
  persistence (`diarization_store.rs`, db schema/migrator), CLI `diarize`,
  UI screens (`ui/src/features/diarization/`, DiarizationPlayground,
  DiarizationQualityPanel, export dialog), hooks, batch runtime.
- **Pipeline** (`crates/izwi-core/src/runtime/diarization.rs`): ASR route +
  forced alignment (`Qwen3-ForcedAligner-0.6B`) + speaker attribution +
  utterance building + optional LLM refinement. Model-agnostic — it
  consumes any `NativeDiarizationModel`.
- **Response types** (`runtime/types.rs:98-162`): `DiarizationSegment`,
  `DiarizationWord`, `DiarizationUtterance`, `DiarizationTranscriptResult`,
  `DiarizationConfig` (min/max speakers, VAD tuning). Nothing new needed.
- **Streaming machinery** (`models/architectures/sortformer/diarization/mod.rs`,
  ~3900 lines): `SortformerModulesConfig` already parses *every* knob the
  new checkpoint uses (spkcache/fifo/chunk/update-period/left-right
  context/boost rates/thresholds); `SortformerStreamingState`,
  `SortformerStreamingProfile {Model, LowLatency, HighLatency}`, physical
  ABI-v2 state (physical.rs), workspace observer. The mel featurizer is
  config-driven (`cfg.features.unwrap_or(128)` — 128 is already the
  default) and loads from the checkpoint.
- **.nemo loading**: `sortformer/diarization/nemo.rs` and
  `nemotron/asr/nemo.rs` both use `open_nemo_archive`; downloader gate is
  "the .nemo file exists" (`artifacts/downloader.rs:881` sortformer, :877
  nemotron-asr).
- **Family machinery**: `ModelFamily::SortformerDiarization` registration
  (`models/families/mod.rs:201`, capability `Diarization`, fixture
  `diarization.short_multispeaker`); `ModelLoadable::Sortformer`,
  `NativeDiarizationModel::Sortformer`, `ModelRegistry::load_diarization*`.
- **Conformance**: `ConformanceCapability::Diarization` +
  `diarization.short_multispeaker` (`runtime/conformance.rs:101-180`).

## 3. Gap analysis (what must change)

| # | Gap | Where today |
|---|-----|-------------|
| 1 | Encoder is hard-wired `SortformerConformerEncoder`; new model needs a plain pre/post-norm Transformer + full RoPE (31/512/8H/FFN 2048) | mod.rs:1109, 2074; `SortformerEncoderConfig` only reads `xscaling` (mod.rs:147) |
| 2 | `MAX_SUPPORTED_SPEAKERS = 4` const woven through streaming config validation, prediction buffers, physical state layout | mod.rs:33, 199, 342, 400-402, 502, 566, 605-614, 890-897; physical.rs:14, 65, 96; loader fails closed at mod.rs:693-696 on any other count |
| 3 | Workspace topology gate pins one frozen production topology (v2.1 conformer+transformer shape) and rejects anything else | mod.rs:219-259 (`SortformerWorkspaceTopology::production()`, `validate_production`) |
| 4 | Head: 8-channel head (hidden 192) + Conv1D upsampler ×8 → 10 ms output stride; post-processing must handle the finer stride for segment timestamps | head code + `RawSegment` post-processing in mod.rs |
| 5 | Latency profiles: card's offline/low/very-low/ultra-low tables for the new checkpoint; existing enum has {Model, LowLatency, HighLatency} with v2.1 values | mod.rs:172-176 |
| 6 | RoPE `max_position_embeddings 5000` (= 400 s @ 80 ms): offline path must chunk longer audio (card: chunked inference supported) | offline `infer_speaker_probabilities_offline` |
| 7 | Activation sweep: new `ModelVariant` + all catalog arms + resolver + downloader + admission + admin caps + UI filters + frozen counts | sites listed in §5 P4 |
| 8 | Selection: `resolve_diarization_model_variant` defaults to v2.1; new variant must be selectable by name | catalog/variant.rs:259 |
| 9 | UI `isDiarizationVariant` string-matches the `diar_streaming_sortformer` prefix | ui/src/features/speech-text/modelFilters.ts:42 |
| 10 | UI new-job modal clamps the speaker draft to [1, 4]; must become model-aware (1–8 when Nemotron-3 is selected, 1–4 for v2.1) | ui/src/features/diarization/components/NewDiarizationModal.tsx:252 (default "4" at :119) |

## 4. Decisions (recommendations, for sign-off)

- **D1 Family placement** — add `ModelVariant::Nemotron3Diarization` inside
  the existing `ModelFamily::SortformerDiarization` (recommended): NeMo
  itself uses one model class for both; streaming machinery, response types
  and pipeline are shared, so this collapses the sweep to a variant + a
  loader arm + per-checkpoint config dispatch. Alternative: a new
  `ModelFamily::NemotronDiarization` — cleaner branding but triggers the
  full family checklist (variant.rs, cuda_support, families table,
  downloader, admin caps, UI filters) for zero behavioral gain; DS10's
  new-family precedent doesn't apply because there the *driver semantics*
  differed, here they are identical.
- **D2 Checkpoint source** — the repo's single `.nemo` file (same loader
  path and gate as both existing audio models; bundles featurizer + NeMo
  YAML config). Skip the GGUF (would be a brand-new audio-loader surface
  for no benefit).
- **D3 Catalog posture** — land **catalog-disabled**, enable in a follow-up
  feat commit after CPU evidence + DER sanity (ADR-0008 / Qwen3.5-35B
  precedent: implement → evidence → `84f287b7`-style enable).
- **D4 Default model** — keep `DiarStreamingSortformer4SpkV21` as the default in
  `resolve_diarization_model_variant`; new model selectable via the job
  `model` field. Revisit default flip after DER evidence.
- **D5 Device** — keep the existing diarization device rule (CUDA native,
  CPU otherwise); at 99M params CPU is cheap and Metal is a fast-follow,
  not a launch requirement.
- **D6 Scope** — streaming/realtime diarization stays **out of scope**:
  `izwi-realtime-v1` has no diarization stage (SpeechToText/TextToSpeech
  only, `serving-protocol/src/realtime.rs:5-7`) and inventing one is net-new
  protocol work. The model's streaming capability remains latent (as it
  already is for v2.1 today).

## 5. Implementation phases (checkable)

- [ ] **P1 Parameterize the Sortformer module for per-checkpoint shapes**
  - Replace the `MAX_SUPPORTED_SPEAKERS` const with a config-derived
    speaker count (fail-closed on unsupported values; keep 4 for v2.1,
    8 for Nemotron-3), threading it through streaming validation, buffers
    and the physical ABI-v2 spec (encode speaker count in the state spec or
    version the ABI).
  - Replace the single frozen `SortformerWorkspaceTopology::production()`
    gate with per-checkpoint pinned topologies (v2.1 unchanged;
    nemotron3: 31L/512/8H, 128 mel bins, head 192) so unknown shapes still
    fail closed.
- [ ] **P2 RoPE transformer encoder**
  - Implement the plain Transformer encoder (GELU, FFN 2048, MHA 8 heads,
    full RoPE θ=10000) in the sortformer diarization module; loader
    discriminates encoder type from the checkpoint's NeMo config; reuse the
    existing candle RoPE implementations from the LLM families where
    applicable; unit tests for load + one forward pass.
- [ ] **P3 Head, upsampler, post-processing, profiles**
  - 512→192 head + Conv1D ×8 upsampler to 10 ms output stride; parameterize
    post-processing stride so segment timestamps land on 10 ms.
  - Derive the new checkpoint's `SortformerStreamingConfig` Model profile
    from its YAML (340/40/40, cache 264, update 222, boosts/thresholds from
    config); add low/very-low/ultra-low latency plans if NVIDIA publishes
    reference values, else ship Model-profile only and say so in docs.
  - Verify offline chunked inference beyond 5000 encoder frames (RoPE cap).
- [ ] **P4 Activation sweep (DS10-style mechanical sweep)**
  - `catalog/metadata.rs`: `ModelVariant::Nemotron3Diarization` + all arms
    (repo_id, display_name, dir_name, estimated_size ~0.4 GB,
    memory_required_gb ~2, license OpenMDW-1.1, is_enabled=false at first,
    `is_diarization()`); aliases (`nemotron-3-diarization`, `nemotron3-diarization`).
  - `catalog/variant.rs`: family() arm (stays `SortformerDiarization`), task
    mapping, parse/heuristic resolver arms.
  - `artifacts/downloader.rs`: file list = the `.nemo`, `is_downloaded`
    gate, size estimates.
  - `runtime/lifecycle/load.rs`: admission estimate (single value; no
    per-backend representation override needed at ~0.4 GB F32).
  - `models/families/mod.rs`: variant slice for the existing family entry
    (+ frozen-count test update, 52/76 → 53/77 style).
  - `catalog/cuda_support.rs`: cells for the variant.
  - `api/admin/models.rs`: `AdminModelRouteCapabilities::from_variant()`.
  - UI: `modelMetadata.ts`, `routeModelCatalog.ts`
    (`DIARIZATION_PREFERRED_MODELS`), `modelFilters.ts:42` prefix predicate,
    `NewDiarizationModal.tsx:252` speaker clamp → model-aware upper bound,
    UI pin test (chat-route-pin precedent 7b281f64).
- [ ] **P5 Evidence + docs + enable**
  - CPU real-artifact diarize run of the new checkpoint through the existing
    conformance fixture `diarization.short_multispeaker` and a
    `real_cpu_*`-style test; speaker-count-8 exercise (a ≥5-speaker clip or
    synthetic mixture since the fixture is 2-4 speakers).
  - DER sanity on one public eval clip vs the card's numbers (order-of-
    magnitude agreement, not a benchmark claim).
  - Docs: `docs/user/models/index.md` diarization row,
    `docs/user/support-matrix.md`, `docs/user/features/diarization.md`,
    benchmark manifests (`benchmarks/manifests/*audio*`).
  - Flip `is_enabled` → true with catalog/UI test updates (Qwen3.5-35B
    enable precedent 84f287b7).
- [ ] **P6 Deferred (explicitly out of scope for launch)**
  - Realtime diarization stage in `izwi-realtime-v1` (+ gateway/worker
    stage runners); Metal backend for the module; default-model flip;
    q8_0 GGUF audio loader.

## 6. Risks / gotchas

- **7215003a lesson**: any future attempt to route realtime diarization
  through the Engine path needs a paged managed-state runtime or an
  explicit Direct-path decision, or it repeats the silent-chunked-fallback
  regression. Launch scope (jobs-only) avoids this entirely.
- Physical state layout currently bakes in 4 speakers — the ABI change is
  the trickiest P1 item; keep v2.1's layout byte-identical (existing
  physical-state tests must stay green unchanged).
- `.nemo` YAML schema: same NeMo class, but verify the encoder/preprocessor
  YAML keys on first download before writing the loader (existing
  `SortformerModulesConfig` keys should map 1:1 to `streaming_config`).
- v2.1's spkcache minimum `(1 + sil_frames_per_spk) * speakers` must use the
  per-model speaker count (cache 264 ≫ (1+1)*8 = 16 — fine).
- Streaming state sizes scale ×2 with 8 speakers and ×1.6 with cache 264 vs
  v2.1's — recheck the physical-state memory accounting / workspace budget.
- GPU-native claim in the card notwithstanding, the model is tiny; CPU
  evidence is the realistic gate for this repo (same posture as every other
  audio model).

## 7. Open questions (need user sign-off before implementation)

1. D1: variant inside `SortformerDiarization` family (recommended) vs new
   `NemotronDiarization` family?
2. D3/D4: catalog-disabled at first with default unchanged (recommended),
   or enable-at-launch as the default diarization model?
3. D6: confirm realtime/streaming diarization stays out of scope.
4. Is there a preferred multi-speaker (≥5 spk) eval clip in-house for the
   speaker-count-8 evidence, or should evidence use a synthetic mixture?
