//! Mixed prefill+decode steps against the split calls they replace: decode
//! rows through `forward_managed_decode_batch`, each prefill span through
//! `forward_managed`, on the tiny fixture and on the Qwen3-ASR-0.6B text
//! trunk (ignored; needs the checkpoint).
use super::*;
use crate::models::shared::attention::mixed::MixedRowSpan;
use crate::models::shared::attention::physical::test_physical_caches_with_dtype_on;

/// Decoders join with prompts of these lengths; prefill rows then arrive
/// two at a time, fed in spans of `SPAN` beside every live decoder.
const DECODER_PROMPTS: [usize; 3] = [4, 7, 10];
const PREFILL_PROMPTS: [usize; 5] = [9, 6, 13, 9, 5];
const SPAN: usize = 4;
const PREFILL_ROWS_PER_STEP: usize = 2;
/// The first decoder leaves after this many steps.
const FIRST_DECODER_STEPS: usize = 3;
/// Logit gaps below this make a step a near tie whose argmax may flip.
const NEAR_TIE: f32 = 1e-3;

/// What a teacher-forced run of mixed steps found against the split calls.
#[derive(Debug)]
struct MixedParity {
    /// Largest difference of any sampled row's logits.
    worst: f32,
    /// Steps whose split argmax was decisive; all agreed.
    decisive: usize,
    /// Rows run by mixed steps, and how many of them were prefill spans.
    rows: usize,
    prefill_rows: usize,
    /// Steps in which no row sampled (only non-final spans).
    unsampled_steps: usize,
    /// Largest difference of any K/V value written by the two worlds,
    /// relative to the largest K/V magnitude.
    page_diff: f32,
}

fn token_ids(count: usize, seed: u64, vocab: (u32, u32)) -> Vec<u32> {
    crate::kernels::numeric_check::xorshift_values(count, seed, 1.0)
        .into_iter()
        .map(|unit| {
            let span = (vocab.1 - vocab.0) as f32;
            vocab.0 + (((unit + 1.0) * 0.5 * span) as u32).min(vocab.1 - vocab.0 - 1)
        })
        .collect()
}

fn values(logits: &Tensor) -> Vec<f32> {
    logits
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// Index of the largest value and its margin over the runner-up.
fn argmax(values: &[f32]) -> (u32, f32) {
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_by(|lhs, rhs| values[*rhs].total_cmp(&values[*lhs]));
    (order[0] as u32, values[order[0]] - values[order[1]])
}

/// Largest absolute difference; a non-finite difference counts as infinite.
fn max_abs_diff(lhs: &[f32], rhs: &[f32]) -> f32 {
    assert_eq!(lhs.len(), rhs.len());
    lhs.iter()
        .zip(rhs)
        .map(|(lhs, rhs)| {
            let diff = (lhs - rhs).abs();
            if diff.is_finite() {
                diff
            } else {
                f32::INFINITY
            }
        })
        .fold(0.0, |worst, diff| if diff > worst { diff } else { worst })
}

/// Every value of every page of `arena`, as F32.
fn page_values(arena: &Arc<dyn KvArena>) -> Vec<f32> {
    let config = arena.config();
    (0..config.capacity_pages)
        .flat_map(|index| {
            let mut bytes = vec![0u8; crate::backends::kv::arena_page_bytes(config) as usize];
            arena
                .capture_page(
                    CacheBlockRef {
                        arena: config.id,
                        group: config.group,
                        index,
                        slot_generation: 1,
                    },
                    &mut bytes,
                )
                .unwrap();
            match config.dtype {
                DType::F32 => bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|value| f32::from_le_bytes(*value))
                    .collect::<Vec<_>>(),
                DType::F16 => bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|value| half::f16::from_le_bytes(*value).to_f32())
                    .collect(),
                DType::BF16 => bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|value| half::bf16::from_le_bytes(*value).to_f32())
                    .collect(),
                other => panic!("unexpected KV dtype {other:?}"),
            }
        })
        .collect()
}

/// One sequence of the run: its prompt, how much of it is cached, and the
/// token it feeds next once decoding.
struct Row {
    prompt: Vec<u32>,
    cursor: usize,
    pending: Option<u32>,
}

/// Teacher-forced mixed steps on `model` against the split calls, with rows
/// joining (each prefill row becomes a decoder) and leaving. Both worlds feed
/// the split world's greedy tokens; every decisive step must agree.
fn mixed_steps_match_split(model: &Qwen3Model, device: &Device, vocab: (u32, u32)) -> MixedParity {
    let geometry = model.cfg.attention_geometry().unwrap();
    let capacity = 64;
    let sequences = DECODER_PROMPTS.len() + PREFILL_PROMPTS.len();
    let dtype = model.embed_tokens.embeddings().dtype();
    let world = || {
        test_physical_caches_with_dtype_on(
            device,
            dtype,
            model.cfg.num_hidden_layers,
            model.cfg.num_key_value_heads,
            geometry.key_head_dim(),
            capacity,
            sequences,
        )
    };
    let (mut split_caches, mut mixed_caches) = (world(), world());
    let mut rows = DECODER_PROMPTS
        .iter()
        .chain(&PREFILL_PROMPTS)
        .enumerate()
        .map(|(index, &len)| Row {
            prompt: token_ids(len, 101 + index as u64, vocab),
            cursor: 0,
            pending: None,
        })
        .collect::<Vec<_>>();
    // Decoders prefill alike in both worlds.
    for (index, row) in rows.iter_mut().enumerate().take(DECODER_PROMPTS.len()) {
        let ids = Tensor::from_slice(&row.prompt, (1, row.prompt.len()), device).unwrap();
        let logits = model
            .forward_managed(&ids, 0, &mut split_caches[index])
            .unwrap();
        model
            .forward_managed(&ids, 0, &mut mixed_caches[index])
            .unwrap();
        let last = logits.i((0, row.prompt.len() - 1)).unwrap();
        row.pending = Some(argmax(&values(&last)).0);
        row.cursor = row.prompt.len();
    }

    let mut parity = MixedParity {
        worst: 0.0,
        decisive: 0,
        rows: 0,
        prefill_rows: 0,
        unsampled_steps: 0,
        page_diff: 0.0,
    };
    for step in 0.. {
        // Step 0 runs prefill spans alone (no row samples); decoders join at
        // step 1, the first one leaves after a few steps, and a finished
        // prompt decodes from the next step on.
        let decoders = (0..rows.len())
            .filter(|index| rows[*index].pending.is_some() && step > 0)
            .filter(|index| *index != 0 || step <= FIRST_DECODER_STEPS)
            .collect::<Vec<_>>();
        let prefill = (DECODER_PROMPTS.len()..rows.len())
            .filter(|index| rows[*index].cursor < rows[*index].prompt.len())
            .take(PREFILL_ROWS_PER_STEP)
            .collect::<Vec<_>>();
        if prefill.is_empty() {
            break;
        }
        let spans = prefill
            .iter()
            .map(|index| {
                let row = &rows[*index];
                (row.cursor, (row.cursor + SPAN).min(row.prompt.len()))
            })
            .collect::<Vec<_>>();
        let finals = prefill
            .iter()
            .zip(&spans)
            .filter(|(index, (_, end))| *end == rows[**index].prompt.len())
            .map(|(index, _)| *index)
            .collect::<Vec<_>>();

        // The split calls.
        let mut split_logits = Vec::new();
        if !decoders.is_empty() {
            let ids = decoders
                .iter()
                .map(|index| rows[*index].pending.unwrap())
                .collect::<Vec<_>>();
            let positions = decoders
                .iter()
                .map(|index| rows[*index].cursor)
                .collect::<Vec<_>>();
            let mut caches = split_caches
                .iter_mut()
                .enumerate()
                .filter(|(index, _)| decoders.contains(index))
                .map(|(_, cache)| cache)
                .collect::<Vec<_>>();
            let logits = model
                .forward_managed_decode_batch(
                    &Tensor::from_vec(ids, (decoders.len(), 1), device).unwrap(),
                    &positions,
                    &mut caches,
                )
                .unwrap();
            for row in 0..decoders.len() {
                split_logits.push(values(&logits.i((row, 0)).unwrap()));
            }
        }
        for (index, (start, end)) in prefill.iter().zip(&spans) {
            let span = &rows[*index].prompt[*start..*end];
            let ids = Tensor::from_slice(span, (1, span.len()), device).unwrap();
            let logits = model
                .forward_managed(&ids, *start, &mut split_caches[*index])
                .unwrap();
            if finals.contains(index) {
                split_logits.push(values(&logits.i((0, span.len() - 1)).unwrap()));
            }
        }

        // The same rows as one mixed step.
        let mut layout = decoders
            .iter()
            .map(|index| MixedRowSpan {
                start_pos: rows[*index].cursor,
                q_len: 1,
            })
            .collect::<Vec<_>>();
        let mut ids = decoders
            .iter()
            .map(|index| rows[*index].pending.unwrap())
            .collect::<Vec<_>>();
        for (index, (start, end)) in prefill.iter().zip(&spans) {
            layout.push(MixedRowSpan {
                start_pos: *start,
                q_len: end - start,
            });
            ids.extend_from_slice(&rows[*index].prompt[*start..*end]);
        }
        let sample_rows = (0..decoders.len())
            .chain(
                prefill
                    .iter()
                    .enumerate()
                    .filter(|(_, index)| finals.contains(index))
                    .map(|(slot, _)| decoders.len() + slot),
            )
            .collect::<Vec<_>>();
        let layout = MixedRowLayout::new(layout, decoders.len()).unwrap();
        let mut by_index = mixed_caches.iter_mut().map(Some).collect::<Vec<_>>();
        let mut caches = decoders
            .iter()
            .chain(&prefill)
            .map(|index| by_index[*index].take().unwrap())
            .collect::<Vec<_>>();
        let mixed = model
            .forward_managed_mixed(&ids, layout, &mut caches, &sample_rows)
            .unwrap();
        assert_eq!(mixed.is_some(), !sample_rows.is_empty());
        parity.rows += decoders.len() + prefill.len();
        parity.prefill_rows += prefill.len();
        parity.unsampled_steps += usize::from(sample_rows.is_empty());

        // Teacher forcing: both worlds feed the split world's tokens.
        let sampled = decoders.iter().chain(&finals).collect::<Vec<_>>();
        for (slot, (index, split)) in sampled.iter().zip(&split_logits).enumerate() {
            let mixed = values(&mixed.as_ref().unwrap().i(slot).unwrap());
            parity.worst = parity.worst.max(max_abs_diff(&mixed, split));
            let (token, margin) = argmax(split);
            if margin >= NEAR_TIE {
                parity.decisive += 1;
                assert_eq!(
                    argmax(&mixed).0,
                    token,
                    "step {step} row {index} left the split token"
                );
            }
            rows[**index].pending = Some(token);
        }
        for index in &decoders {
            rows[*index].cursor += 1;
        }
        for (index, (_, end)) in prefill.iter().zip(&spans) {
            rows[*index].cursor = *end;
        }
    }
    let arena = |caches: &[PhysicalPagedKvCache]| caches[0].arena().clone();
    let split_pages = page_values(&arena(&split_caches));
    let scale = split_pages
        .iter()
        .fold(0f32, |max, value| max.max(value.abs()));
    parity.page_diff =
        max_abs_diff(&split_pages, &page_values(&arena(&mixed_caches))) / scale.max(1.0);
    parity
}

fn assert_tiny_parity(device: &Device, tolerance: f32) {
    let model = tiny_qwen3_model_for_test(device);
    let parity = mixed_steps_match_split(&model, device, (0, 8));
    eprintln!("tiny Qwen3 mixed vs split on {device:?}: {parity:?}");
    assert!(
        parity.prefill_rows >= PREFILL_PROMPTS.len() * 2,
        "{parity:?}"
    );
    assert!(parity.rows > 2 * parity.prefill_rows, "{parity:?}");
    assert_eq!(parity.unsampled_steps, 1, "{parity:?}");
    assert!(parity.decisive > 0, "{parity:?}");
    assert!(parity.worst <= tolerance, "{parity:?}");
    assert!(parity.page_diff <= tolerance, "{parity:?}");
}

/// The tiny fixture's dense F32 projections run a matmul of every token of
/// the step instead of one per split call, which moves the last bits
/// (measured 1.8e-7 on aarch64), so the CPU leg is a tolerance too.
#[test]
fn tiny_mixed_steps_match_split_calls_on_cpu() {
    assert_tiny_parity(&Device::Cpu, 1e-5);
}

#[cfg(feature = "metal")]
#[test]
fn tiny_mixed_steps_match_split_calls_on_metal() {
    let Some((device, _serial)) =
        crate::backends::metal_test_device_or_skip("tiny_mixed_steps_match_split_calls_on_metal")
    else {
        return;
    };
    assert_tiny_parity(&device, 1e-4);
}

/// The Qwen3-ASR-0.6B Q8 text trunk (the same `Qwen3Model`), when the
/// checkpoint is present.
fn asr_text_trunk(profile: crate::backends::DeviceProfile) -> Option<Qwen3Model> {
    let dir = dirs::data_local_dir()?
        .join("izwi")
        .join("models")
        .join("Qwen3-ASR-0.6B-GGUF");
    if !dir.exists() {
        use std::io::Write;
        let _ = writeln!(std::io::stderr(), "skipping: {} is missing", dir.display());
        return None;
    }
    let model = crate::models::architectures::qwen3::asr::Qwen3AsrModel::load(
        &dir,
        crate::model::ModelVariant::Qwen3Asr06BGguf,
        profile,
    )
    .unwrap();
    Some(model.into_text_model())
}

/// `logits` bounds the sampled rows' logits; `pages` the written K/V
/// relative to their magnitude.
fn assert_real_parity(profile: crate::backends::DeviceProfile, logits: f32, pages: f32) {
    let device = profile.device.clone();
    let Some(model) = asr_text_trunk(profile) else {
        return;
    };
    let parity = mixed_steps_match_split(&model, &device, (1_000, 30_000));
    eprintln!("Qwen3-ASR-0.6B trunk mixed vs split on {device:?}: {parity:?}");
    assert!(parity.decisive > 0, "{parity:?}");
    assert!(parity.worst <= logits, "{parity:?}");
    assert!(parity.page_diff <= pages, "{parity:?}");
}

#[test]
#[ignore = "needs the Qwen3-ASR-0.6B GGUF"]
fn asr_trunk_mixed_steps_match_split_calls_on_cpu() {
    // Q8 projections run row by row on the CPU: exact.
    assert_real_parity(crate::backends::DeviceProfile::cpu(), 0.0, 0.0);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the Qwen3-ASR-0.6B GGUF"]
fn asr_trunk_mixed_steps_match_split_calls_on_metal() {
    use crate::backends::{BackendPreference, DeviceSelector};
    let Some((_, _serial)) = crate::backends::metal_test_device_or_skip(
        "asr_trunk_mixed_steps_match_split_calls_on_metal",
    ) else {
        return;
    };
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    // F16 activations: the step's T-row matmuls take Candle's half-precision
    // mat-mat tiles where the split decode rows took mat-vec.
    assert_real_parity(profile, 0.15, 0.02);
}
