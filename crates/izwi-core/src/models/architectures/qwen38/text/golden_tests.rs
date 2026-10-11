//! End-to-end value parity of the native Qwen3.8 trunk against the reference
//! implementation.
//!
//! The fixture (`scripts/fixtures/generate_qwen38_golden.py`) is a tiny
//! random-init HF `Qwen3_5ForCausalLM` written in the published
//! Qwen3.8-27B-FP8 layout: block-FP8 projections with BF16 scales, BF16 dense
//! tensors, an untied BF16 LM head and the 22-tensor MTP manifest. It has
//! random non-zero zero-centered norm gains, 2 key / 6 value DeltaNet heads,
//! 12 query / 2 KV attention heads, interleaved partial M-RoPE at theta 1e7,
//! and two full-attention layers among eight. `golden.safetensors` holds HF's
//! logits for every position of a 32-token sequence; the generator checked
//! that HF's cached decode reproduces them before writing the fixture.
//!
//! Exercised through the production loader (`Qwen38NativeCheckpoint` +
//! `Qwen38TextModel::load_native_with_performance`): every native value
//! convention (zero-centered `1 + w`, grouped value-head order, `A_log`, conv
//! orientation, FP8 block scales), interleaved partial M-RoPE, the gated full
//! attention with GQA, the DeltaNet sequence and decode recurrences, chunked
//! prefill continuation, scalar decode, and the ragged batched decode.

use super::*;
use crate::backends::kv::{CpuKvArena, KvArena, KvArenaConfig, KvLayerConfig};
use crate::backends::BackendKind;
use crate::engine::ModelInstanceId;
use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
use crate::models::architectures::qwen38::native::Qwen38NativeCheckpoint;

const PAGE_TOKENS: u32 = 8;
const PAGES_PER_SESSION: u32 = 4;
const SESSIONS: u32 = 4;
/// Prefill spans for the chunked-continuation leg.
const PROMPT_CHUNKS: [usize; 2] = [5, 7];

fn fixture_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen38_golden")
}

/// Reference token ids, prompt length, and `[tokens, vocab]` logits recorded
/// from HF transformers' `Qwen3_5ForCausalLM`.
fn golden_reference() -> (Vec<u32>, usize, Vec<Vec<f32>>) {
    let raw = std::fs::read(fixture_dir().join("golden.safetensors")).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&raw).unwrap();
    let i64s = |name: &str| -> Vec<i64> {
        tensors
            .tensor(name)
            .unwrap()
            .data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| i64::from_le_bytes(*b))
            .collect()
    };
    let token_ids = i64s("token_ids").into_iter().map(|v| v as u32).collect();
    let prompt = i64s("prompt_tokens")[0] as usize;
    let logits = tensors.tensor("logits").unwrap();
    let vocab = logits.shape()[1];
    let values: Vec<f32> = logits
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    (
        token_ids,
        prompt,
        values.chunks(vocab).map(<[f32]>::to_vec).collect(),
    )
}

fn load_golden_model(
    checkpoint: &Qwen38NativeCheckpoint,
    device: &Device,
    target: ProjectionMaterialization,
) -> Qwen38TextModel {
    Qwen38TextModel::load_native_with_performance(
        &checkpoint.tensors,
        &checkpoint.config,
        device,
        target,
        &crate::performance::PerformanceConfig::default().cuda,
    )
    .unwrap()
}

/// One paged KV arena shared by every session of a leg, so the batched decode
/// rows can share it the way continuous batching does.
struct GoldenArena {
    arena: Arc<dyn KvArena>,
    id: KvArenaId,
    group: KvGroupId,
    bindings: Vec<KvLayerBinding>,
}

impl GoldenArena {
    fn new(config: &Qwen38NativeConfig, device: &Device, dtype: DType) -> Self {
        let bindings: Vec<KvLayerBinding> = config
            .layer_types
            .iter()
            .enumerate()
            .filter(|(_, kind)| **kind == Qwen38LayerType::FullAttention)
            .enumerate()
            .map(|(physical, (model_layer, _))| KvLayerBinding {
                model_layer: model_layer as u32,
                physical_layer: physical as u32,
            })
            .collect();
        let backend = if device.is_metal() {
            BackendKind::Metal
        } else {
            BackendKind::Cpu
        };
        let id = KvArenaId {
            model_instance: ModelInstanceId::new(3811),
            backend,
            device_ordinal: device.is_metal().then_some(0),
            generation: 1,
        };
        let group = KvGroupId::new(1);
        let arena_config = KvArenaConfig {
            id,
            group,
            page_tokens: PAGE_TOKENS,
            capacity_pages: PAGES_PER_SESSION * SESSIONS,
            growth: None,
            dtype,
            layers: bindings
                .iter()
                .map(|binding| KvLayerConfig {
                    binding: *binding,
                    num_kv_heads: config.text.attention_head_count_kv as u32,
                    key_head_dim: config.text.attention_key_length as u32,
                    value_head_dim: config.text.attention_value_length as u32,
                })
                .collect(),
        };
        let arena: Arc<dyn KvArena> = match backend {
            #[cfg(feature = "metal")]
            BackendKind::Metal => Arc::new(
                crate::backends::kv::CandleAcceleratorKvArena::new_mutation_only(
                    arena_config,
                    device.clone(),
                )
                .unwrap(),
            ),
            _ => Arc::new(CpuKvArena::new(arena_config).unwrap()),
        };
        Self {
            arena,
            id,
            group,
            bindings,
        }
    }

    fn session(&self, index: u32) -> PhysicalPagedKvCache {
        assert!(index < SESSIONS);
        let blocks = (index * PAGES_PER_SESSION..(index + 1) * PAGES_PER_SESSION)
            .map(|block| CacheBlockRef {
                arena: self.id,
                group: self.group,
                index: block,
                slot_generation: 1,
            })
            .collect();
        PhysicalPagedKvCache::new(self.arena.clone(), self.bindings.clone(), blocks, 0).unwrap()
    }
}

/// One trunk logits row, tagged with the execution path and the sequence
/// position whose HF reference row it must match.
struct GoldenRow {
    path: &'static str,
    position: usize,
    logits: Vec<f32>,
}

fn logits_rows(logits: &Tensor) -> Vec<Vec<f32>> {
    let vocab = logits.dim(D::Minus1).unwrap();
    logits
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .chunks(vocab)
        .map(<[f32]>::to_vec)
        .collect()
}

/// Run every production execution path over the golden sequence:
/// - chunked prefill of the prompt, then scalar decode of the rest;
/// - the whole sequence as one prefill span (every row);
/// - a ragged two-row batched decode (rows prefilled to different lengths).
fn golden_trunk_rows(
    model: &Qwen38TextModel,
    arena: &GoldenArena,
    token_ids: &[u32],
    prompt: usize,
) -> Vec<GoldenRow> {
    assert_eq!(PROMPT_CHUNKS.iter().sum::<usize>(), prompt);
    let positions: Vec<[usize; 3]> = (0..token_ids.len()).map(|p| [p, p, p]).collect();
    let mut rows = Vec::new();

    let mut state = model.new_state();
    let mut cache = arena.session(0);
    let mut start = 0;
    for span in PROMPT_CHUNKS {
        let output = model
            .prefill_token_ids_with_hidden_physical(
                &token_ids[start..start + span],
                &positions[start..start + span],
                &mut state,
                &mut cache,
                false,
            )
            .unwrap()
            .expect("prefill output");
        let logits = model
            .project_target_hidden_span(&output.hidden_states)
            .unwrap();
        for (offset, logits) in logits_rows(&logits).into_iter().enumerate() {
            rows.push(GoldenRow {
                path: "chunked prefill",
                position: start + offset,
                logits,
            });
        }
        start += span;
    }
    for position in prompt..token_ids.len() {
        let logits = model
            .forward_token_id_at_physical(
                token_ids[position],
                positions[position],
                &mut state,
                &mut cache,
            )
            .unwrap();
        rows.push(GoldenRow {
            path: "scalar decode",
            position,
            logits: logits_rows(&logits).remove(0),
        });
    }

    let mut state = model.new_state();
    let mut cache = arena.session(1);
    let output = model
        .prefill_token_ids_with_hidden_physical(token_ids, &positions, &mut state, &mut cache, true)
        .unwrap()
        .expect("prefill output");
    let all = model
        .project_target_hidden_span(&output.hidden_states)
        .unwrap();
    for (position, logits) in logits_rows(&all).into_iter().enumerate() {
        rows.push(GoldenRow {
            path: "full prefill",
            position,
            logits,
        });
    }
    // The production prefill head (last row only) must agree with the span head.
    rows.push(GoldenRow {
        path: "full prefill head",
        position: token_ids.len() - 1,
        logits: logits_rows(&output.logits.expect("prefill logits")).remove(0),
    });

    // Ragged batch: row 0 resumes after the prompt, row 1 four tokens later.
    let starts = [prompt, prompt + 4];
    let mut states = [model.new_state(), model.new_state()];
    let mut caches = [arena.session(2), arena.session(3)];
    for row in 0..2 {
        model
            .prefill_token_ids_physical(
                &token_ids[..starts[row]],
                &positions[..starts[row]],
                &mut states[row],
                &mut caches[row],
                false,
            )
            .unwrap();
    }
    for step in 0..token_ids.len() - starts[1] {
        let step_positions = [starts[0] + step, starts[1] + step];
        let [state_a, state_b] = &mut states;
        let [cache_a, cache_b] = &mut caches;
        let output = model
            .forward_token_ids_batch_at_physical(
                &step_positions.map(|position| token_ids[position]),
                &step_positions.map(|position| positions[position]),
                &mut [state_a, state_b],
                &mut [cache_a, cache_b],
            )
            .unwrap();
        for (row, logits) in logits_rows(&output.logits).into_iter().enumerate() {
            rows.push(GoldenRow {
                path: "batched decode",
                position: step_positions[row],
                logits,
            });
        }
    }
    rows
}

/// Largest absolute difference; any non-finite difference saturates to
/// infinity so a NaN can never read as agreement.
fn max_abs_diff(lhs: &[f32], rhs: &[f32]) -> f32 {
    assert_eq!(lhs.len(), rhs.len());
    lhs.iter().zip(rhs).fold(0.0f32, |worst, (a, b)| {
        let diff = (a - b).abs();
        if diff.is_finite() {
            worst.max(diff)
        } else {
            f32::INFINITY
        }
    })
}

/// `|values - reference|_2 / |reference|_2`, infinite on any non-finite value.
fn relative_l2(values: &[f32], reference: &[f32]) -> f32 {
    assert_eq!(values.len(), reference.len());
    let error: f32 = values
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    let norm: f32 = reference.iter().map(|b| b * b).sum();
    let relative = (error / norm).sqrt();
    if relative.is_finite() {
        relative
    } else {
        f32::INFINITY
    }
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .unwrap()
}

#[test]
fn golden_tolerance_helpers_never_pass_non_finite_values() {
    let reference = [1.0f32, -2.0, 3.0];
    assert_eq!(max_abs_diff(&reference, &reference), 0.0);
    for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for index in 0..reference.len() {
            let mut values = reference;
            values[index] = poison;
            assert_eq!(max_abs_diff(&values, &reference), f32::INFINITY);
            assert_eq!(relative_l2(&values, &reference), f32::INFINITY);
        }
    }
}

/// The production CPU plan (F32 materialization, expanded F32 projections,
/// F32 KV) must match HF to F32 tolerance on every row of every path.
#[test]
fn native_trunk_matches_the_hf_reference_logits() {
    let checkpoint = Qwen38NativeCheckpoint::open_synthetic(&fixture_dir()).unwrap();
    let (token_ids, prompt, reference) = golden_reference();
    assert_eq!(reference.len(), token_ids.len());
    assert_eq!(reference[0].len(), checkpoint.config.vocab_size);
    let device = Device::Cpu;
    let model = load_golden_model(&checkpoint, &device, ProjectionMaterialization::F32);
    assert_eq!(
        model.projection_representation(),
        Qwen38ProjectionRepresentation::ExpandedF32
    );
    let arena = GoldenArena::new(&checkpoint.config, &device, DType::F32);

    let rows = golden_trunk_rows(&model, &arena, &token_ids, prompt);
    // Chunked prefill + scalar decode, full prefill + its head, batched decode.
    let expected_rows = 2 * token_ids.len() + 1 + 2 * (token_ids.len() - prompt - 4);
    assert_eq!(rows.len(), expected_rows);
    let mut worst = 0.0f32;
    for row in &rows {
        let diff = max_abs_diff(&row.logits, &reference[row.position]);
        assert!(
            diff < 2e-3,
            "{} logits at position {} diverge from HF by {diff}",
            row.path,
            row.position
        );
        worst = worst.max(diff);
    }
    eprintln!(
        "qwen38 golden CPU: {} rows, worst max |diff| {worst:e}",
        rows.len()
    );
}

/// The production Metal plan (F16 materialization, expanded F16 projections,
/// F16 KV, F32 DeltaNet state) on the GPU. Half precision is the error floor:
/// HF itself run in F16 deviates from its F32 logits by up to 2.6% relative L2
/// on this fixture, and this plan lands at the same level. The bound only
/// catches gross errors: a loader convention error is 50-100%.
#[cfg(feature = "metal")]
#[test]
fn native_trunk_tracks_the_hf_reference_logits_on_metal() {
    let Some(device) = crate::backends::metal_device_if_available(0) else {
        eprintln!("metal device unavailable; qwen38 golden metal leg not run");
        return;
    };
    let checkpoint = Qwen38NativeCheckpoint::open_synthetic(&fixture_dir()).unwrap();
    let (token_ids, prompt, reference) = golden_reference();
    let model = load_golden_model(&checkpoint, &device, ProjectionMaterialization::F16);
    let arena = GoldenArena::new(&checkpoint.config, &device, DType::F16);

    let rows = golden_trunk_rows(&model, &arena, &token_ids, prompt);
    let mut agree = 0usize;
    let mut worst = 0.0f32;
    for row in &rows {
        let relative = relative_l2(&row.logits, &reference[row.position]);
        assert!(
            relative < 0.05,
            "metal {} logits at position {} diverge from HF by {relative} (relative L2)",
            row.path,
            row.position
        );
        worst = worst.max(relative);
        agree += usize::from(argmax(&row.logits) == argmax(&reference[row.position]));
    }
    assert!(
        agree * 10 >= rows.len() * 9,
        "metal greedy tokens agree with HF on only {agree}/{} rows",
        rows.len()
    );
    eprintln!(
        "qwen38 golden metal: {} rows, worst relative L2 {worst:e}, argmax agreement {agree}/{}",
        rows.len(),
        rows.len()
    );
}
