//! Opt-in decode trace over a real Qwen3.5 checkpoint.
//!
//! ```text
//! IZWI_QWEN35_TRACE_OUT=/path/trace.json \
//!   cargo test -p izwi-core --features metal --lib qwen35_local_decode_trace -- --ignored
//! ```
//!
//! The test loads a local GGUF checkpoint (`IZWI_QWEN35_TRACE_MODEL`, default
//! `Qwen3.5-0.8B`) on Metal when available, otherwise on the CPU. It prefills
//! a fixed prompt in one span, decodes greedily, and writes each step's argmax
//! and top logits as JSON. Comparing two traces (before and after a change,
//! or a fast path against its `legacy` switch) exposes numerics drift. It also
//! decodes two copies of the prompt as one batch and checks both rows against
//! the solo run, which exercises the per-row position plumbing.

use std::sync::Arc;

use super::*;
#[cfg(any(feature = "cuda", feature = "metal"))]
use crate::backends::kv::CandleAcceleratorKvArena;
use crate::backends::kv::{CpuKvArena, KvArena, KvArenaConfig, KvLayerConfig};
use crate::backends::{BackendPreference, DeviceProfile, DeviceSelector};
use crate::engine::ModelInstanceId;
use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};

const TRACE_STEPS: usize = 32;
const TRACE_TOP: usize = 16;
const PAGE_TOKENS: u32 = 16;
const PAGES_PER_SEQUENCE: u32 = 16;

fn trace_variant(name: &str) -> ModelVariant {
    match name {
        "Qwen3.5-0.8B" => ModelVariant::Qwen3508BGguf,
        "Qwen3.5-2B" => ModelVariant::Qwen352BGguf,
        "Qwen3.5-4B" => ModelVariant::Qwen354BGguf,
        "Qwen3.5-9B" => ModelVariant::Qwen359BGguf,
        other => panic!("unknown Qwen3.5 trace model {other}"),
    }
}

/// One arena with room for `sequences` disjoint sequences, plus a physical
/// cache per sequence. Batched decode requires the rows to share an arena.
pub(super) fn trace_caches(
    config: &Qwen35TextConfig,
    device: &DeviceProfile,
    sequences: u32,
) -> Vec<PhysicalPagedKvCache> {
    let bindings = (0..config.block_count)
        .filter(|layer| {
            config.full_attention_interval > 0
                && (layer + 1).is_multiple_of(config.full_attention_interval)
        })
        .enumerate()
        .map(|(physical, model_layer)| KvLayerBinding {
            model_layer: model_layer as u32,
            physical_layer: physical as u32,
        })
        .collect::<Vec<_>>();
    let backend = BackendKind::from(device.kind);
    let id = KvArenaId {
        model_instance: ModelInstanceId::new(4351),
        backend,
        device_ordinal: (backend != BackendKind::Cpu).then_some(0),
        generation: 1,
    };
    let group = KvGroupId::new(1);
    let arena_config = KvArenaConfig {
        id,
        group,
        page_tokens: PAGE_TOKENS,
        capacity_pages: PAGES_PER_SEQUENCE * sequences,
        growth: None,
        dtype: DType::F32,
        layers: bindings
            .iter()
            .map(|binding| KvLayerConfig {
                binding: *binding,
                num_kv_heads: config.attention_head_count_kv as u32,
                key_head_dim: config.attention_key_length as u32,
                value_head_dim: config.attention_value_length as u32,
            })
            .collect(),
    };
    let arena: Arc<dyn KvArena> = if backend == BackendKind::Cpu {
        Arc::new(CpuKvArena::new(arena_config).expect("cpu arena"))
    } else {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        {
            Arc::new(
                CandleAcceleratorKvArena::new_mutation_only(arena_config, device.device.clone())
                    .expect("accelerator arena"),
            )
        }
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        {
            let _ = arena_config;
            unreachable!("accelerator traces need the cuda or metal feature")
        }
    };
    (0..sequences)
        .map(|sequence| {
            let blocks = (sequence * PAGES_PER_SEQUENCE..(sequence + 1) * PAGES_PER_SEQUENCE)
                .map(|index| CacheBlockRef {
                    arena: id,
                    group,
                    index,
                    slot_generation: 1,
                })
                .collect();
            PhysicalPagedKvCache::new(arena.clone(), bindings.clone(), blocks, 0)
                .expect("physical cache")
        })
        .collect()
}

fn logits_row(logits: &Tensor) -> Vec<f32> {
    logits
        .flatten_all()
        .and_then(|row| row.to_dtype(DType::F32))
        .and_then(|row| row.to_vec1::<f32>())
        .expect("logits row")
}

fn top_logits(row: &[f32], vocab: usize) -> Vec<(u32, f32)> {
    let mut indexed = row[..vocab.min(row.len())]
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| (index as u32, value))
        .collect::<Vec<_>>();
    indexed.sort_by(|left, right| right.1.total_cmp(&left.1));
    indexed.truncate(TRACE_TOP);
    indexed
}

#[test]
#[ignore = "needs a local Qwen3.5 checkpoint and IZWI_QWEN35_TRACE_OUT"]
fn qwen35_local_decode_trace() {
    let Ok(out) = std::env::var("IZWI_QWEN35_TRACE_OUT") else {
        eprintln!("IZWI_QWEN35_TRACE_OUT is unset; trace not run");
        return;
    };
    let name =
        std::env::var("IZWI_QWEN35_TRACE_MODEL").unwrap_or_else(|_| "Qwen3.5-0.8B".to_string());
    let home = std::env::var("HOME").expect("HOME");
    let model_dir = std::path::PathBuf::from(home)
        .join("Library/Application Support/izwi/models")
        .join(&name);
    #[cfg(feature = "metal")]
    let _lease = crate::backends::metal_test_device_or_skip("qwen35_local_decode_trace");
    let device = DeviceSelector::detect_for_preference(BackendPreference::Metal)
        .ok()
        .filter(|profile| profile.kind != crate::backends::DeviceKind::Cpu)
        .unwrap_or_else(DeviceProfile::cpu);
    let mut model = Qwen35ChatModel::load(&model_dir, trace_variant(&name), device.clone())
        .expect("load checkpoint");
    // IZWI_QWEN35_TRACE_EAGER_GRAPH=1 decodes through the CUDA graph's step
    // body, run eagerly with the arena's attention (checks its composition
    // off CUDA).
    if std::env::var("IZWI_QWEN35_TRACE_EAGER_GRAPH").is_ok_and(|value| value == "1") {
        model.exec.text_model.resolve_decode_graph(None, true);
    }
    let text = &model.exec.text_model;
    let vocab = model.exec.tokenizer.vocab_size;
    let messages = vec![ChatMessage {
        role: ChatRole::User,
        content: "Explain in two sentences why the sky looks blue at noon and red at sunset."
            .to_string(),
    }];
    let prompt = model.prompt_token_ids(&messages).expect("prompt ids");
    let positions = build_text_positions(prompt.len());

    // Solo: one-span prefill, then greedy decode.
    let mut caches = trace_caches(model.text_config(), &device, 3);
    let mut solo_cache = caches.remove(0);
    let mut state = text.new_state();
    let mut logits = text
        .prefill_token_ids_physical(&prompt, &positions, &mut state, &mut solo_cache, true)
        .expect("prefill")
        .expect("prefill logits");
    let mut steps = Vec::with_capacity(TRACE_STEPS);
    let mut tokens = Vec::with_capacity(TRACE_STEPS);
    let mut decode_ms = Vec::with_capacity(TRACE_STEPS);
    for step in 0..TRACE_STEPS {
        let row = logits_row(&logits);
        let top = top_logits(&row, vocab);
        let token = top[0].0;
        steps.push(serde_json::json!({
            "token": token,
            "top": top,
            "sum": row[..vocab].iter().map(|value| f64::from(*value)).sum::<f64>(),
        }));
        tokens.push(token);
        let started = std::time::Instant::now();
        logits = text
            .forward_token_id_at_physical(
                token,
                [prompt.len() + step; 3],
                &mut state,
                &mut solo_cache,
            )
            .expect("decode");
        // One scalar readback waits for the device, so the interval covers
        // the whole forward, not just its enqueue.
        logits
            .argmax(D::Minus1)
            .and_then(|index| index.to_dtype(DType::U32))
            .and_then(|index| index.to_scalar::<u32>())
            .expect("decode sync");
        decode_ms.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    decode_ms.sort_by(f64::total_cmp);
    let median_decode_ms = decode_ms[decode_ms.len() / 2];

    // Batch: two copies of the prompt decoded as one two-row batch.
    let mut batch_states = [text.new_state(), text.new_state()];
    let mut batch_rows = Vec::new();
    for (state, cache) in batch_states.iter_mut().zip(caches.iter_mut()) {
        let logits = text
            .prefill_token_ids_physical(&prompt, &positions, state, cache, true)
            .expect("batch prefill")
            .expect("batch prefill logits");
        batch_rows.push(top_logits(&logits_row(&logits), vocab)[0].0);
    }
    let mut batch_max_diff = 0f32;
    for (step, token) in tokens.iter().enumerate().take(TRACE_STEPS - 1) {
        assert_eq!(
            batch_rows,
            vec![*token; 2],
            "batch rows diverged at step {step}"
        );
        let mut state_refs = batch_states.iter_mut().collect::<Vec<_>>();
        let mut cache_refs = caches.iter_mut().collect::<Vec<_>>();
        let logits = text
            .forward_token_ids_batch_at_physical(
                &[*token, *token],
                &[[prompt.len() + step; 3]; 2],
                &mut state_refs,
                &mut cache_refs,
            )
            .expect("batch decode");
        let solo_top = steps[step + 1]["top"].as_array().expect("top").clone();
        batch_rows.clear();
        for row in 0..2 {
            let values = logits_row(&logits.i((row, 0)).expect("batch row"));
            let top = top_logits(&values, vocab);
            for ((_, value), solo) in top.iter().zip(&solo_top) {
                let solo = solo[1].as_f64().expect("solo logit") as f32;
                let diff = (value - solo).abs();
                // `f32::max` would drop a NaN difference.
                batch_max_diff = if diff.is_nan() {
                    f32::INFINITY
                } else {
                    batch_max_diff.max(diff)
                };
            }
            batch_rows.push(top[0].0);
        }
    }

    let trace = serde_json::json!({
        "model": name,
        "device": format!("{:?}", device.kind),
        "prompt_tokens": prompt.len(),
        "tokens": tokens,
        "steps": steps,
        "batch_vs_solo_max_top_logit_diff": batch_max_diff,
        "median_decode_ms": median_decode_ms,
        "telemetry": text.telemetry().snapshot(),
        "gdn_decode": text.gdn_decode_summary(),
        "decode_graph": text.decode_graph_summary(),
    });
    std::fs::write(&out, serde_json::to_string_pretty(&trace).expect("json")).expect("write");
    eprintln!(
        "trace written to {out}: tokens {tokens:?}, batch diff {batch_max_diff}, \
         median decode {median_decode_ms:.2} ms"
    );
}

const WIDTH_ROWS: usize = 8;
const WIDTH_STEPS: usize = 32;
const WIDTH_TIMED_STEPS: usize = 48;
/// Solo steps whose top-2 margin is below this are near ties that a
/// different op shape may legitimately flip.
const WIDTH_NEAR_TIE: f32 = 0.5;

/// Decode parity and throughput across tensor batch widths over a real
/// Qwen3.5 checkpoint (`IZWI_QWEN35_TRACE_MODEL`, default `Qwen3.5-0.8B`),
/// for the certified widths in `catalog::batch_width`.
///
/// Parity: eight prompts decode greedily one row at a time (width 1), then
/// together at widths 2, 4 and 8, teacher-forced with the solo tokens. Every
/// decisive step must pick the solo token, and the solo top-16 logits may
/// move by at most `max_logit_delta`. Throughput: widths 1, 2, 4 and 8
/// decode greedily side by side, one step of each width in turn (an
/// interleaved A/B, so machine load hits every width alike), each step ending
/// in the one argmax readback serving makes.
fn width_leg(device: DeviceProfile, max_logit_delta: f32) {
    let name =
        std::env::var("IZWI_QWEN35_TRACE_MODEL").unwrap_or_else(|_| "Qwen3.5-0.8B".to_string());
    let home = std::env::var("HOME").expect("HOME");
    let model_dir = std::path::PathBuf::from(home)
        .join("Library/Application Support/izwi/models")
        .join(&name);
    if !model_dir.exists() {
        eprintln!("skipping: {} is missing", model_dir.display());
        return;
    }
    let model = Qwen35ChatModel::load(&model_dir, trace_variant(&name), device.clone())
        .expect("load checkpoint");
    let text = &model.exec.text_model;
    let vocab = model.exec.tokenizer.vocab_size;
    let backend = format!("{:?}", device.kind);
    let topics = [
        "why the sky is blue",
        "how bread rises",
        "what a prime number is",
        "how tides work",
        "why leaves change colour",
        "how a bicycle stays upright",
        "what makes ice float",
        "how bees make honey",
    ];
    let prompts: Vec<Vec<u32>> = topics
        .iter()
        .map(|topic| {
            model
                .prompt_token_ids(&[ChatMessage {
                    role: ChatRole::User,
                    content: format!("Explain {topic} in three sentences."),
                }])
                .expect("prompt ids")
        })
        .collect();
    let argmax = |values: &[f32]| top_logits(values, vocab)[0].0;
    // Prefill `rows` (indices into `prompts`) into fresh caches; returns the
    // states, caches and each row's first token.
    let start = |rows: &[usize]| {
        let mut caches = trace_caches(model.text_config(), &device, rows.len() as u32);
        let mut states = Vec::with_capacity(rows.len());
        let mut first = Vec::with_capacity(rows.len());
        for (cache, &index) in caches.iter_mut().zip(rows) {
            let prompt = &prompts[index];
            let mut state = text.new_state();
            let logits = text
                .prefill_token_ids_physical(
                    prompt,
                    &build_text_positions(prompt.len()),
                    &mut state,
                    cache,
                    true,
                )
                .expect("prefill")
                .expect("prefill logits");
            first.push(argmax(&logits_row(&logits)));
            states.push(state);
        }
        (states, caches, first)
    };
    let step = |rows: &[usize],
                states: &mut [Qwen35TextRuntimeState],
                caches: &mut [PhysicalPagedKvCache],
                tokens: &[u32],
                position: &[usize]| {
        let positions: Vec<[usize; 3]> = rows
            .iter()
            .zip(position)
            .map(|(&index, offset)| [prompts[index].len() + offset; 3])
            .collect();
        let mut state_refs = states.iter_mut().collect::<Vec<_>>();
        let mut cache_refs = caches.iter_mut().collect::<Vec<_>>();
        text.forward_token_ids_batch_at_physical(
            tokens,
            &positions,
            &mut state_refs,
            &mut cache_refs,
        )
        .expect("batch decode")
    };

    // Width 1: each row alone, greedy.
    let mut solo: Vec<Vec<(u32, f32, Vec<(u32, f32)>)>> = Vec::with_capacity(WIDTH_ROWS);
    for index in 0..WIDTH_ROWS {
        let (mut states, mut caches, first) = start(&[index]);
        let mut token = first[0];
        let mut steps = Vec::with_capacity(WIDTH_STEPS);
        for offset in 0..WIDTH_STEPS {
            let logits = step(&[index], &mut states, &mut caches, &[token], &[offset]);
            let top = top_logits(&logits_row(&logits.i((0, 0)).expect("row")), vocab);
            token = top[0].0;
            steps.push((token, top[0].1 - top[1].1, top));
        }
        solo.push(steps);
    }
    for width in [2usize, 4, 8] {
        let (mut decisive, mut worst) = (0usize, 0f32);
        for group in (0..WIDTH_ROWS).collect::<Vec<_>>().chunks(width) {
            let (mut states, mut caches, mut tokens) = start(group);
            for offset in 0..WIDTH_STEPS {
                let logits = step(
                    group,
                    &mut states,
                    &mut caches,
                    &tokens,
                    &vec![offset; width],
                );
                for (slot, &index) in group.iter().enumerate() {
                    let values = logits_row(&logits.i((slot, 0)).expect("row"));
                    let (next, margin, top) = &solo[index][offset];
                    for (id, expected) in top {
                        let diff = (values[*id as usize] - expected).abs();
                        worst = if diff.is_nan() {
                            f32::INFINITY
                        } else {
                            worst.max(diff)
                        };
                    }
                    if *margin >= WIDTH_NEAR_TIE {
                        decisive += 1;
                        assert_eq!(
                            argmax(&values),
                            *next,
                            "{backend} width {width}: row {index} step {offset} left the solo \
                             token (margin {margin:.3})"
                        );
                    }
                    tokens[slot] = *next;
                }
            }
        }
        eprintln!(
            "{name} {backend} width {width}: {decisive} decisive steps agree with width 1, \
             worst top-16 logit delta {worst:.4}"
        );
        assert!(
            worst <= max_logit_delta,
            "{backend} width {width}: top-16 logit delta {worst} exceeds {max_logit_delta}"
        );
    }

    // Throughput, interleaved across widths.
    let widths = [1usize, 2, 4, 8];
    let mut groups: Vec<_> = widths
        .iter()
        .map(|&width| {
            let rows: Vec<usize> = (0..width).collect();
            let (states, caches, tokens) = start(&rows);
            (rows, states, caches, tokens)
        })
        .collect();
    let mut elapsed = [std::time::Duration::ZERO; 4];
    for offset in 0..WIDTH_TIMED_STEPS + 8 {
        for ((rows, states, caches, tokens), time) in groups.iter_mut().zip(&mut elapsed) {
            let started = std::time::Instant::now();
            let logits = step(rows, states, caches, tokens, &vec![offset; rows.len()]);
            let next = logits
                .squeeze(1)
                .and_then(|rows| rows.narrow(1, 0, vocab))
                .and_then(|rows| rows.argmax(D::Minus1))
                .and_then(|ids| ids.to_dtype(DType::U32))
                .and_then(|ids| ids.to_vec1::<u32>())
                .expect("argmax readback");
            if offset >= 8 {
                *time += started.elapsed();
            }
            tokens.copy_from_slice(&next);
        }
    }
    for (width, time) in widths.iter().zip(elapsed) {
        let ms = time.as_secs_f64() * 1000.0 / WIDTH_TIMED_STEPS as f64;
        eprintln!(
            "{name} {backend} width {width}: {ms:.2} ms/step, {:.1} tok/s",
            *width as f64 * 1000.0 / ms
        );
    }
}

#[test]
#[ignore = "needs a local Qwen3.5 checkpoint"]
fn qwen35_decode_width_parity_and_throughput_cpu() {
    // Measured on aarch64 (0.8B): widths 2, 4 and 8 equal width 1 exactly.
    width_leg(DeviceProfile::cpu(), 0.05);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "needs a local Qwen3.5 checkpoint"]
fn qwen35_decode_width_parity_and_throughput_metal() {
    let _lease = crate::backends::metal_test_device_or_skip(
        "qwen35_decode_width_parity_and_throughput_metal",
    );
    let device =
        DeviceSelector::detect_for_preference(BackendPreference::Metal).expect("metal device");
    if device.kind == crate::backends::DeviceKind::Cpu {
        eprintln!("skipping: no Metal device");
        return;
    }
    // Measured on an M1 Pro (0.8B and 2B): widths 2, 4 and 8 equal width 1
    // exactly.
    width_leg(device, 0.05);
}
