//! Mixed prefill+decode steps against the split calls they replace: decode
//! rows through the retained decode batch, each prefill span through the
//! retained chunk forward, on the tiny fixture and on LFM2.5-1.2B-Instruct
//! (ignored; needs the checkpoint).
use super::*;
use crate::backends::kv::{arena_page_bytes, CpuKvArena, KvArena, KvArenaConfig, KvLayerConfig};
use crate::backends::BackendKind;
use crate::engine::ModelInstanceId;
use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
use crate::models::shared::attention::mixed::MixedRowSpan;

/// Decoders join with prompts of these lengths; prefill rows then arrive
/// two at a time, fed in spans of `SPAN` beside every live decoder.
const DECODER_PROMPTS: [usize; 3] = [4, 7, 10];
const PREFILL_PROMPTS: [usize; 5] = [9, 6, 13, 9, 5];
const SPAN: usize = 4;
const PREFILL_ROWS_PER_STEP: usize = 2;
/// The first decoder leaves after this many steps.
const FIRST_DECODER_STEPS: usize = 3;
/// Logit gaps below this make a step a near tie whose argmax may flip: tiny
/// for the tiny fixture, and the reference tests' 0.5 on real weights.
const TINY_NEAR_TIE: f32 = 1e-3;
const NEAR_TIE: f32 = 0.5;
const PAGE_TOKENS: u32 = 16;
const PAGES_PER_ROW: u32 = 4;

/// What a teacher-forced run of mixed steps found against the split calls.
#[derive(Debug)]
struct MixedParity {
    /// Largest difference of any sampled row's logits at the split row's
    /// top eight tokens (as the reference tests measure).
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
    /// Largest difference of any row's ShortConv state after the run,
    /// relative to the largest state magnitude.
    state_diff: f32,
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

fn values(tensor: &Tensor) -> Vec<f32> {
    tensor
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

/// The largest difference of `rhs` from `lhs`, relative to `lhs`'s largest
/// magnitude (at least one).
fn relative_diff(lhs: &[f32], rhs: &[f32]) -> f32 {
    let scale = lhs.iter().fold(0f32, |max, value| max.max(value.abs()));
    max_abs_diff(lhs, rhs) / scale.max(1.0)
}

/// `rows` caches of `pages_per_row` pages over one arena of `backbone`'s
/// attention layers on `device`, storing K/V as `dtype`.
fn caches(
    backbone: &QuantizedLfm2Backbone,
    device: &Device,
    dtype: DType,
    rows: usize,
    pages_per_row: u32,
) -> Vec<PhysicalPagedKvCache> {
    let bindings = backbone
        .state_layout
        .attention_model_layers
        .iter()
        .enumerate()
        .map(|(physical, model_layer)| KvLayerBinding {
            model_layer: *model_layer,
            physical_layer: physical as u32,
        })
        .collect::<Vec<_>>();
    let backend = if device.is_metal() {
        BackendKind::Metal
    } else {
        BackendKind::Cpu
    };
    let id = KvArenaId {
        model_instance: ModelInstanceId::new(2_513),
        backend,
        device_ordinal: device.is_metal().then_some(0),
        generation: 1,
    };
    let group = KvGroupId::new(1);
    let head_dim = backbone.cfg.embedding_length / backbone.cfg.attention_head_count;
    let config = KvArenaConfig {
        id,
        group,
        page_tokens: PAGE_TOKENS,
        capacity_pages: pages_per_row * rows as u32,
        growth: None,
        dtype,
        layers: bindings
            .iter()
            .map(|binding| KvLayerConfig {
                binding: *binding,
                num_kv_heads: backbone.cfg.attention_head_count_kv[binding.model_layer as usize]
                    as u32,
                key_head_dim: head_dim as u32,
                value_head_dim: head_dim as u32,
            })
            .collect(),
    };
    let arena: Arc<dyn KvArena> = if device.is_metal() {
        #[cfg(feature = "metal")]
        {
            Arc::new(
                crate::backends::kv::CandleAcceleratorKvArena::new_mutation_only(
                    config,
                    device.clone(),
                )
                .unwrap(),
            )
        }
        #[cfg(not(feature = "metal"))]
        unreachable!("a Metal device without the metal feature")
    } else {
        Arc::new(CpuKvArena::new(config).unwrap())
    };
    (0..rows as u32)
        .map(|row| {
            let blocks = (row * pages_per_row..(row + 1) * pages_per_row)
                .map(|index| CacheBlockRef {
                    arena: id,
                    group,
                    index,
                    slot_generation: 1,
                })
                .collect();
            PhysicalPagedKvCache::new(arena.clone(), bindings.clone(), blocks, 0).unwrap()
        })
        .collect()
}

/// Every value of every page of `arena`, as F32.
fn page_values(arena: &Arc<dyn KvArena>) -> Vec<f32> {
    let config = arena.config();
    (0..config.capacity_pages)
        .flat_map(|index| {
            let mut bytes = vec![0u8; arena_page_bytes(config) as usize];
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
                other => panic!("unexpected KV dtype {other:?}"),
            }
        })
        .collect()
}

/// Every ShortConv ring value of every row.
fn state_values(states: &[Lfm2ShortConvRuntimeState]) -> Vec<f32> {
    states
        .iter()
        .flat_map(|state| &state.components)
        .flat_map(|(_, ring)| values(ring.as_ref().expect("a ring after prefill")))
        .collect()
}

/// One sequence of the run: its prompt, how much of it is cached, and the
/// token it feeds next once decoding.
struct Row {
    prompt: Vec<u32>,
    cursor: usize,
    pending: Option<u32>,
}

/// One world's per-row KV caches and ShortConv states.
struct World {
    caches: Vec<PhysicalPagedKvCache>,
    states: Vec<Lfm2ShortConvRuntimeState>,
}

impl World {
    fn new(backbone: &QuantizedLfm2Backbone, device: &Device, dtype: DType, rows: usize) -> Self {
        Self {
            caches: caches(backbone, device, dtype, rows, PAGES_PER_ROW),
            states: (0..rows).map(|_| backbone.new_shortconv_state()).collect(),
        }
    }

    /// The prompt span through the retained chunk forward: its last
    /// token's logits.
    fn prefill(
        &mut self,
        backbone: &QuantizedLfm2Backbone,
        row: usize,
        embeds: &Tensor,
        start: usize,
    ) -> Vec<f32> {
        let hidden = backbone
            .forward_embeds_retained(embeds, start, &mut self.caches[row], &mut self.states[row])
            .unwrap();
        values(&backbone.project_last_hidden(&hidden).unwrap())
    }

    /// Mutable references to `rows`' caches and states, in that order.
    fn rows(
        &mut self,
        rows: &[usize],
    ) -> (
        Vec<&mut PhysicalPagedKvCache>,
        Vec<&mut Lfm2ShortConvRuntimeState>,
    ) {
        let mut caches = self.caches.iter_mut().map(Some).collect::<Vec<_>>();
        let mut states = self.states.iter_mut().map(Some).collect::<Vec<_>>();
        rows.iter()
            .map(|row| (caches[*row].take().unwrap(), states[*row].take().unwrap()))
            .unzip()
    }
}

/// Teacher-forced mixed steps on `backbone` against the split calls, with
/// rows joining (each prefill row becomes a decoder) and leaving. `embed`
/// maps token ids to `[1, tokens, hidden]` embeddings; both worlds feed the
/// split world's greedy tokens and every step whose split argmax leads by
/// at least `near_tie` must agree.
fn mixed_steps_match_split(
    backbone: &QuantizedLfm2Backbone,
    device: &Device,
    kv_dtype: DType,
    embed: &dyn Fn(&[u32]) -> Tensor,
    vocab: (u32, u32),
    near_tie: f32,
) -> MixedParity {
    let sequences = DECODER_PROMPTS.len() + PREFILL_PROMPTS.len();
    let mut split = World::new(backbone, device, kv_dtype, sequences);
    let mut mixed = World::new(backbone, device, kv_dtype, sequences);
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
        let embeds = embed(&row.prompt);
        let logits = split.prefill(backbone, index, &embeds, 0);
        mixed.prefill(backbone, index, &embeds, 0);
        row.pending = Some(argmax(&logits).0);
        row.cursor = row.prompt.len();
    }

    let mut parity = MixedParity {
        worst: 0.0,
        decisive: 0,
        rows: 0,
        prefill_rows: 0,
        unsampled_steps: 0,
        page_diff: 0.0,
        state_diff: 0.0,
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
        let pending = decoders
            .iter()
            .map(|index| rows[*index].pending.unwrap())
            .collect::<Vec<_>>();
        let positions = decoders
            .iter()
            .map(|index| rows[*index].cursor)
            .collect::<Vec<_>>();

        // The split calls.
        let mut split_logits = Vec::new();
        if !decoders.is_empty() {
            let embeds = embed(&pending).reshape((decoders.len(), 1, ())).unwrap();
            let (mut caches, mut states) = split.rows(&decoders);
            let hidden = backbone
                .forward_embeds_retained_step_batch(&embeds, &positions, &mut states, &mut caches)
                .unwrap();
            let logits = backbone.project_last_hidden(&hidden).unwrap();
            for row in 0..decoders.len() {
                split_logits.push(values(&logits.i(row).unwrap()));
            }
        }
        for (index, (start, end)) in prefill.iter().zip(&spans) {
            let embeds = embed(&rows[*index].prompt[*start..*end]);
            let logits = split.prefill(backbone, *index, &embeds, *start);
            if finals.contains(index) {
                split_logits.push(logits);
            }
        }

        // The same rows as one mixed step.
        let mut layout = positions
            .iter()
            .map(|&start_pos| MixedRowSpan {
                start_pos,
                q_len: 1,
            })
            .collect::<Vec<_>>();
        let mut ids = pending;
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
        let members = decoders.iter().chain(&prefill).copied().collect::<Vec<_>>();
        let (mut caches, mut states) = mixed.rows(&members);
        let (logits, commit) = {
            let cache_refs = caches.iter().map(|cache| &**cache).collect::<Vec<_>>();
            let state_refs = states.iter().map(|state| &**state).collect::<Vec<_>>();
            backbone
                .forward_embeds_mixed_deferred(
                    &embed(&ids),
                    layout,
                    &state_refs,
                    &cache_refs,
                    &sample_rows,
                )
                .unwrap()
        };
        let mixed_logits = logits.as_ref().map(values);
        commit.commit(&mut states, &mut caches).unwrap();
        assert_eq!(mixed_logits.is_some(), !sample_rows.is_empty());
        parity.rows += members.len();
        parity.prefill_rows += prefill.len();
        parity.unsampled_steps += usize::from(sample_rows.is_empty());

        // Teacher forcing: both worlds feed the split world's tokens.
        let sampled = decoders.iter().chain(&finals).collect::<Vec<_>>();
        let width = vocab.1 as usize;
        for (slot, (index, split)) in sampled.iter().zip(&split_logits).enumerate() {
            let mixed = &mixed_logits.as_ref().unwrap()[slot * width..(slot + 1) * width];
            let mut top = (0..split.len()).collect::<Vec<_>>();
            top.sort_by(|lhs, rhs| split[*rhs].total_cmp(&split[*lhs]));
            let top = top.into_iter().take(8).collect::<Vec<_>>();
            parity.worst = parity.worst.max(max_abs_diff(
                &top.iter().map(|id| mixed[*id]).collect::<Vec<_>>(),
                &top.iter().map(|id| split[*id]).collect::<Vec<_>>(),
            ));
            let (token, margin) = argmax(split);
            if margin >= near_tie {
                parity.decisive += 1;
                assert_eq!(
                    argmax(mixed).0,
                    token,
                    "step {step} row {index} left the split token (margin {margin}, logits \
                     differ by {})",
                    max_abs_diff(mixed, split)
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
    let arena = |world: &World| world.caches[0].arena().clone();
    parity.page_diff = relative_diff(&page_values(&arena(&split)), &page_values(&arena(&mixed)));
    parity.state_diff = relative_diff(&state_values(&split.states), &state_values(&mixed.states));
    for (split, mixed) in split.states.iter().zip(&mixed.states) {
        assert_eq!(split.cursor, mixed.cursor);
    }
    parity
}

/// A fixed random embedding table for the embeds-only tiny backbone.
fn tiny_embedder(device: &Device, vocab: usize, hidden: usize) -> impl Fn(&[u32]) -> Tensor {
    let table = Tensor::from_vec(
        crate::kernels::numeric_check::xorshift_values(vocab * hidden, 77, 1.0),
        (vocab, hidden),
        device,
    )
    .unwrap();
    let device = device.clone();
    move |ids: &[u32]| {
        let ids = Tensor::from_slice(ids, ids.len(), &device).unwrap();
        table.index_select(&ids, 0).unwrap().unsqueeze(0).unwrap()
    }
}

fn assert_tiny_parity(device: &Device, tolerance: f32) {
    // Five tokens (the head's outputs), with and without a sliding window
    // shorter than the longest context.
    for window in [None, Some(6)] {
        let backbone = tests::tiny_windowed_backbone_on(window, 5, device);
        let embed = tiny_embedder(device, 5, backbone.cfg.embedding_length);
        let parity =
            mixed_steps_match_split(&backbone, device, DType::F32, &embed, (0, 5), TINY_NEAR_TIE);
        eprintln!("tiny LFM2 (window {window:?}) mixed vs split on {device:?}: {parity:?}");
        assert!(
            parity.prefill_rows >= PREFILL_PROMPTS.len() * 2,
            "{parity:?}"
        );
        assert!(parity.rows > 2 * parity.prefill_rows, "{parity:?}");
        assert_eq!(parity.unsampled_steps, 1, "{parity:?}");
        assert!(parity.decisive > 0, "{parity:?}");
        assert!(parity.worst <= tolerance, "{parity:?}");
        assert!(parity.page_diff <= tolerance, "{parity:?}");
        assert!(parity.state_diff <= tolerance, "{parity:?}");
    }
}

/// The tiny fixture's F32 projections run a matmul of every token of the
/// step instead of one per split call, which can move the last bits.
#[test]
fn tiny_mixed_steps_match_split_calls_on_cpu() {
    assert_tiny_parity(&Device::Cpu, 1e-5);
}

#[cfg(feature = "metal")]
#[test]
fn tiny_mixed_steps_match_split_calls_on_metal() {
    let Some((device, _serial)) = crate::backends::metal_test_device_or_skip(
        "lfm2 tiny_mixed_steps_match_split_calls_on_metal",
    ) else {
        return;
    };
    assert_tiny_parity(&device, 1e-4);
}

/// LFM2.5-1.2B-Instruct's backbone on `profile`, when the checkpoint is
/// present.
fn lfm25_backbone(profile: &crate::backends::DeviceProfile) -> Option<QuantizedLfm2Backbone> {
    let path = std::env::var_os("IZWI_LFM25_MODELS_DIR")
        .map(std::path::PathBuf::from)
        .or_else(|| dirs::data_local_dir().map(|dir| dir.join("izwi").join("models")))?
        .join("LFM2.5-1.2B-Instruct-GGUF")
        .join("LFM2.5-1.2B-Instruct-Q4_K_M.gguf");
    if !path.exists() {
        use std::io::Write;
        let _ = writeln!(std::io::stderr(), "skipping: {} is missing", path.display());
        return None;
    }
    let loader = crate::models::shared::weights::gguf::GgufLoader::from_path_with_backend(
        &path,
        BackendKind::from(profile.kind),
    )
    .unwrap();
    let config = super::super::config::parse_lfm2_backbone_config(&loader).unwrap();
    Some(QuantizedLfm2Backbone::load(&loader, config, &profile.device).unwrap())
}

/// `logits` bounds the sampled rows' logits; `state` the written K/V and
/// the ShortConv states relative to their magnitude.
fn assert_real_parity(profile: crate::backends::DeviceProfile, logits: f32, state: f32) {
    let Some(backbone) = lfm25_backbone(&profile) else {
        return;
    };
    let device = profile.device.clone();
    // The KV storage the chat model declares in production.
    let kv_dtype = match super::super::chat::lfm2_chat_kv_dtype(BackendKind::from(profile.kind)) {
        crate::kv::v2::StateDType::F16 => DType::F16,
        _ => DType::F32,
    };
    let embed = |ids: &[u32]| {
        backbone
            .embed_tokens(&Tensor::from_slice(ids, (1, ids.len()), &device).unwrap())
            .unwrap()
    };
    let vocab = (1_000, backbone.vocab_size() as u32);
    let parity = mixed_steps_match_split(&backbone, &device, kv_dtype, &embed, vocab, NEAR_TIE);
    eprintln!("LFM2.5-1.2B mixed vs split on {device:?}: {parity:?}");
    assert!(parity.decisive > 0, "{parity:?}");
    assert!(parity.worst <= logits, "{parity:?}");
    assert!(parity.page_diff <= state, "{parity:?}");
    assert!(parity.state_diff <= state, "{parity:?}");
}

#[test]
#[ignore = "needs the LFM2.5-1.2B GGUF (IZWI_LFM25_MODELS_DIR)"]
fn lfm25_mixed_steps_match_split_calls_on_cpu() {
    assert_real_parity(crate::backends::DeviceProfile::cpu(), 0.0, 0.0);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the LFM2.5-1.2B GGUF (IZWI_LFM25_MODELS_DIR)"]
fn lfm25_mixed_steps_match_split_calls_on_metal() {
    use crate::backends::{BackendPreference, DeviceSelector};
    let Some((_, _serial)) =
        crate::backends::metal_test_device_or_skip("lfm25_mixed_steps_match_split_calls_on_metal")
    else {
        return;
    };
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    // Decode rows take Candle's half-precision mat-mat tiles in a mixed step
    // where the split decode step takes mat-vec: measured top-8 0.33 (the
    // Metal-vs-F32 reference error is also 0.33), K/V 1.8%, state 0.8%.
    assert_real_parity(profile, 0.6, 0.05);
}

/// Milliseconds since `started`, after the device has drained.
fn synced_ms(device: &Device, started: std::time::Instant) -> f64 {
    device.synchronize().unwrap();
    started.elapsed().as_secs_f64() * 1000.0
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// One timing world: `decoders` rows decoding at a long context and
/// prefill rows continuing long prompts span by span.
struct BenchWorld {
    caches: Vec<PhysicalPagedKvCache>,
    states: Vec<Lfm2ShortConvRuntimeState>,
    positions: Vec<usize>,
}

const BENCH_CONTEXT: usize = 512;
const BENCH_PROMPT: usize = 2_048;

impl BenchWorld {
    fn new(
        backbone: &QuantizedLfm2Backbone,
        device: &Device,
        dtype: DType,
        decoders: usize,
    ) -> Self {
        let rows = decoders + 2;
        let pages_per_row = (BENCH_PROMPT as u32).div_ceil(PAGE_TOKENS) + 8;
        let mut world = Self {
            caches: caches(backbone, device, dtype, rows, pages_per_row),
            states: (0..rows).map(|_| backbone.new_shortconv_state()).collect(),
            positions: vec![0; rows],
        };
        for row in 0..decoders {
            let prompt = token_ids(BENCH_CONTEXT, 900 + row as u64, (1_000, 30_000));
            let ids = Tensor::from_slice(&prompt, (1, prompt.len()), device).unwrap();
            backbone
                .forward_tokens_retained(
                    &ids,
                    0,
                    &mut world.caches[row],
                    &mut world.states[row],
                    false,
                )
                .unwrap();
            world.positions[row] = BENCH_CONTEXT;
        }
        world
    }

    /// Restart prefill row `row` on a fresh prompt when `span` more tokens
    /// would overrun it.
    fn make_room(&mut self, backbone: &QuantizedLfm2Backbone, row: usize, span: usize) {
        if self.positions[row] + span <= BENCH_PROMPT {
            return;
        }
        let cache = &self.caches[row];
        self.caches[row] = PhysicalPagedKvCache::new(
            cache.arena().clone(),
            (0..cache.arena().config().layers.len())
                .map(|layer| cache.layer_binding(layer).unwrap())
                .collect(),
            cache.blocks.clone(),
            0,
        )
        .unwrap();
        self.states[row] = backbone.new_shortconv_state();
        self.positions[row] = 0;
    }
}

/// One step's tokens for `decoders` decode rows and `spans` prefill spans.
fn bench_tokens(decoders: usize, spans: &[usize], step: usize) -> (Vec<u32>, Vec<Vec<u32>>) {
    let decode = vec![1_234u32; decoders];
    let prefill = spans
        .iter()
        .enumerate()
        .map(|(row, span)| {
            token_ids(
                *span,
                5_000 + 31 * step as u64 + row as u64,
                (1_000, 30_000),
            )
        })
        .collect();
    (decode, prefill)
}

/// The split step: one decode-batch call (deferred commit after the
/// readback, as served) and one chunk call per prefill span. Returns the
/// decode and the prefill milliseconds.
fn split_step(
    backbone: &QuantizedLfm2Backbone,
    device: &Device,
    world: &mut BenchWorld,
    decoders: usize,
    spans: &[usize],
    step: usize,
) -> (f64, f64) {
    let (decode, prefill) = bench_tokens(decoders, spans, step);
    let started = std::time::Instant::now();
    let (logits, pending) = {
        let states = world.states[..decoders].iter().collect::<Vec<_>>();
        let caches = world.caches[..decoders].iter().collect::<Vec<_>>();
        backbone
            .forward_token_ids_retained_batch_deferred(
                &decode,
                &world.positions[..decoders],
                &states,
                &caches,
            )
            .unwrap()
    };
    let _ = logits.argmax(D::Minus1).unwrap().to_vec1::<u32>().unwrap();
    {
        let (states, caches) = (&mut world.states, &mut world.caches);
        let mut states = states[..decoders].iter_mut().collect::<Vec<_>>();
        let mut caches = caches[..decoders].iter_mut().collect::<Vec<_>>();
        pending.commit(&mut states, &mut caches).unwrap();
    }
    let decode_ms = synced_ms(device, started);
    let started = std::time::Instant::now();
    for (slot, ids) in prefill.iter().enumerate() {
        let row = decoders + slot;
        let input = Tensor::from_slice(ids, (1, ids.len()), device).unwrap();
        backbone
            .forward_tokens_retained(
                &input,
                world.positions[row],
                &mut world.caches[row],
                &mut world.states[row],
                false,
            )
            .unwrap();
    }
    let prefill_ms = synced_ms(device, started);
    for position in &mut world.positions[..decoders] {
        *position += 1;
    }
    for (slot, span) in spans.iter().enumerate() {
        world.positions[decoders + slot] += span;
    }
    (decode_ms, prefill_ms)
}

/// The same rows as one mixed step.
fn mixed_step(
    backbone: &QuantizedLfm2Backbone,
    device: &Device,
    world: &mut BenchWorld,
    decoders: usize,
    spans: &[usize],
    step: usize,
) -> f64 {
    let (decode, prefill) = bench_tokens(decoders, spans, step);
    let members = decoders + spans.len();
    let mut layout = world.positions[..decoders]
        .iter()
        .map(|&start_pos| MixedRowSpan {
            start_pos,
            q_len: 1,
        })
        .collect::<Vec<_>>();
    let mut ids = decode;
    for (slot, span) in prefill.iter().enumerate() {
        layout.push(MixedRowSpan {
            start_pos: world.positions[decoders + slot],
            q_len: span.len(),
        });
        ids.extend_from_slice(span);
    }
    let layout = MixedRowLayout::new(layout, decoders).unwrap();
    let sample_rows = (0..decoders).collect::<Vec<_>>();
    let started = std::time::Instant::now();
    let (logits, pending) = {
        let states = world.states[..members].iter().collect::<Vec<_>>();
        let caches = world.caches[..members].iter().collect::<Vec<_>>();
        backbone
            .forward_token_ids_mixed_deferred(&ids, layout, &states, &caches, &sample_rows)
            .unwrap()
    };
    let _ = logits
        .unwrap()
        .argmax(D::Minus1)
        .unwrap()
        .to_vec1::<u32>()
        .unwrap();
    {
        let (states, caches) = (&mut world.states, &mut world.caches);
        let mut states = states[..members].iter_mut().collect::<Vec<_>>();
        let mut caches = caches[..members].iter_mut().collect::<Vec<_>>();
        pending.commit(&mut states, &mut caches).unwrap();
    }
    let ms = synced_ms(device, started);
    for position in &mut world.positions[..decoders] {
        *position += 1;
    }
    for (slot, span) in spans.iter().enumerate() {
        world.positions[decoders + slot] += span;
    }
    ms
}

/// Interleaved in-process A/B of a mixed step against the split calls it
/// replaces, on LFM2.5-1.2B: decode rows at a 512-token context beside
/// prefill spans of several sizes. Both worlds run the same rows step by
/// step in alternating order, so machine noise lands on both.
fn mixed_vs_split_timing(profile: crate::backends::DeviceProfile) {
    let Some(backbone) = lfm25_backbone(&profile) else {
        return;
    };
    let device = profile.device.clone();
    let kv_dtype = match super::super::chat::lfm2_chat_kv_dtype(BackendKind::from(profile.kind)) {
        crate::kv::v2::StateDType::F16 => DType::F16,
        _ => DType::F32,
    };
    let steps = std::env::var("IZWI_BENCH_STEPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(16usize);
    let warmup = 3;
    for decoders in [4usize, 8] {
        let mut split = BenchWorld::new(&backbone, &device, kv_dtype, decoders);
        let mut mixed = BenchWorld::new(&backbone, &device, kv_dtype, decoders);
        for spans in [&[64usize][..], &[192], &[376], &[192, 184]] {
            let (mut split_ms, mut decode_ms, mut prefill_ms, mut mixed_ms) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for step in 0..warmup + steps {
                for (slot, span) in spans.iter().enumerate() {
                    split.make_room(&backbone, decoders + slot, *span);
                    mixed.make_room(&backbone, decoders + slot, *span);
                }
                let mut run_split =
                    || split_step(&backbone, &device, &mut split, decoders, spans, step);
                let (split_times, mixed_time) = if step % 2 == 0 {
                    let split_times = run_split();
                    (
                        split_times,
                        mixed_step(&backbone, &device, &mut mixed, decoders, spans, step),
                    )
                } else {
                    let mixed_time =
                        mixed_step(&backbone, &device, &mut mixed, decoders, spans, step);
                    (run_split(), mixed_time)
                };
                if step >= warmup {
                    decode_ms.push(split_times.0);
                    prefill_ms.push(split_times.1);
                    split_ms.push(split_times.0 + split_times.1);
                    mixed_ms.push(mixed_time);
                }
            }
            let (split_ms, mixed_ms) = (median(&mut split_ms), median(&mut mixed_ms));
            eprintln!(
                "{:?} LFM2.5 {decoders} decode rows + spans {spans:?}: split {split_ms:.1} ms \
                 (decode {:.1} + prefill {:.1}, 2 calls), mixed {mixed_ms:.1} ms (1 call), \
                 mixed/split {:.2}",
                profile.kind,
                median(&mut decode_ms),
                median(&mut prefill_ms),
                mixed_ms / split_ms
            );
        }
    }
}

#[test]
#[ignore = "timing; needs the LFM2.5-1.2B GGUF (run with --release)"]
fn lfm25_mixed_vs_split_step_timing_cpu() {
    mixed_vs_split_timing(crate::backends::DeviceProfile::cpu());
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "timing; needs the LFM2.5-1.2B GGUF (run with --release)"]
fn lfm25_mixed_vs_split_step_timing_metal() {
    use crate::backends::{BackendPreference, DeviceSelector};
    let Some((_, _serial)) =
        crate::backends::metal_test_device_or_skip("lfm25_mixed_vs_split_step_timing_metal")
    else {
        return;
    };
    mixed_vs_split_timing(DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap());
}
