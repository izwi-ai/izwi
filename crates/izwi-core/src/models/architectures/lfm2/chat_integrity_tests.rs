//! Decode integrity regressions with a tiny loaded hybrid LFM and CPU physical cache.
use super::*;
use crate::backends::kv::{CpuKvArena, KvArenaConfig, KvLayerConfig};
use crate::backends::DeviceProfile;
use crate::engine::ModelInstanceId;
use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};
use std::sync::Arc;

fn tiny_lfm_model() -> Lfm2ChatModel {
    tiny_lfm_model_with(|shape| Tensor::ones(shape, DType::F32, &Device::Cpu).unwrap())
}

/// The tiny model with xorshift weights from `seed`: unlike the all-ones
/// model it does not end every reply at once.
pub(super) fn random_tiny_lfm_model(seed: u64) -> Lfm2ChatModel {
    let mut seed = seed;
    tiny_lfm_model_with(move |shape| {
        seed += 1;
        let values =
            crate::kernels::numeric_check::xorshift_values(shape.iter().product(), seed, 1.0);
        Tensor::from_vec(values, shape, &Device::Cpu).unwrap()
    })
}

fn tiny_lfm_model_with(mut weight: impl FnMut(&[usize]) -> Tensor) -> Lfm2ChatModel {
    let variant = ModelVariant::Lfm2512BInstructGguf;
    let directory =
        std::env::temp_dir().join(format!("izwi-lfm-contract-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&directory).unwrap();
    let tokenizer = tokenizers::Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                [
                    ("<|pad|>".to_string(), 0),
                    ("a".to_string(), 3),
                    ("b".to_string(), 4),
                    ("Ã".to_string(), 5),
                    ("©".to_string(), 6),
                    ("<|im_start|>".to_string(), 1),
                    ("<|im_end|>".to_string(), 2),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("<|pad|>".to_string())
            .build()
            .unwrap(),
    );
    tokenizer
        .save(directory.join("tokenizer.json"), false)
        .unwrap();
    std::fs::write(
        directory.join("tokenizer_config.json"),
        r#"{"added_tokens_decoder":{"0":{"content":"<|pad|>","special":true},"1":{"content":"<|im_start|>"},"2":{"content":"<|im_end|>"}}}"#,
    )
    .unwrap();
    use gguf_file::Value;
    let metadata = [
        ("general.architecture", Value::String("lfm2".into())),
        ("lfm2.block_count", Value::U32(2)),
        ("lfm2.context_length", Value::U32(32)),
        ("lfm2.embedding_length", Value::U32(4)),
        ("lfm2.attention.head_count", Value::U32(1)),
        (
            "lfm2.attention.head_count_kv",
            Value::Array(vec![Value::U32(1), Value::U32(0)]),
        ),
        ("lfm2.attention.layer_norm_rms_epsilon", Value::F32(1e-5)),
        ("lfm2.shortconv.l_cache", Value::U32(3)),
    ];
    let mut weights = vec![];
    let mut add = |name: String, shape: &[usize]| {
        let tensor = weight(shape);
        weights.push((name, QTensor::quantize(&tensor, GgmlDType::F32).unwrap()));
    };
    add("token_embd.weight".into(), &[7, 4]);
    add("output_norm.weight".into(), &[4]);
    for layer in 0..2 {
        for name in ["attn_norm", "ffn_norm"] {
            add(format!("blk.{layer}.{name}.weight"), &[4]);
        }
        for name in ["ffn_gate", "ffn_up", "ffn_down"] {
            add(format!("blk.{layer}.{name}.weight"), &[4, 4]);
        }
    }
    for name in ["attn_q_norm", "attn_k_norm"] {
        add(format!("blk.0.{name}.weight"), &[4]);
    }
    for name in ["attn_q", "attn_k", "attn_v", "attn_output"] {
        add(format!("blk.0.{name}.weight"), &[4, 4]);
    }
    add("blk.1.shortconv.in_proj.weight".into(), &[12, 4]);
    add("blk.1.shortconv.out_proj.weight".into(), &[4, 4]);
    add("blk.1.shortconv.conv.weight".into(), &[4, 3]);
    let filename = match variant {
        ModelVariant::Lfm2512BInstructGguf => "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
        ModelVariant::Lfm2512BThinkingGguf => "LFM2.5-1.2B-Thinking-Q4_K_M.gguf",
        _ => unreachable!(),
    };
    let mut file = std::fs::File::create(directory.join(filename)).unwrap();
    gguf_file::write(
        &mut file,
        &metadata
            .iter()
            .map(|(name, value)| (*name, value))
            .collect::<Vec<_>>(),
        &weights
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    drop(file);
    let mut model = Lfm2ChatModel::load(&directory, variant, DeviceProfile::cpu()).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
    let tokens = ["<|pad|>", "<|im_start|>", "<|im_end|>", "a", "b", "Ã", "©"].map(str::to_owned);
    model.tokenizer.inner = Tokenizer::from_gguf_bpe(&tokens, &[], Some("qwen35"), false).unwrap();
    model
}

fn bindings() -> Vec<KvLayerBinding> {
    vec![KvLayerBinding {
        model_layer: 0,
        physical_layer: 0,
    }]
}

fn cache() -> PhysicalPagedKvCache {
    let id = KvArenaId {
        model_instance: ModelInstanceId::new(101),
        backend: BackendKind::Cpu,
        device_ordinal: None,
        generation: 1,
    };
    let group = KvGroupId::new(1);
    let arena = Arc::new(
        CpuKvArena::new(KvArenaConfig {
            id,
            group,
            page_tokens: 4,
            capacity_pages: 8,
            growth: None,
            dtype: DType::F32,
            layers: vec![KvLayerConfig {
                binding: bindings()[0],
                num_kv_heads: 1,
                key_head_dim: 4,
                value_head_dim: 4,
            }],
        })
        .unwrap(),
    );
    let blocks = (0..8)
        .map(|index| CacheBlockRef {
            arena: id,
            group,
            index,
            slot_generation: 1,
        })
        .collect();
    PhysicalPagedKvCache::new(arena, bindings(), blocks, 0).unwrap()
}

/// One arena whose blocks several caches address, so a publisher and an
/// attacher can share prefix pages and keep their own suffix pages.
fn shared_arena() -> (Arc<CpuKvArena>, KvArenaId, KvGroupId) {
    let id = KvArenaId {
        model_instance: ModelInstanceId::new(102),
        backend: BackendKind::Cpu,
        device_ordinal: None,
        generation: 1,
    };
    let group = KvGroupId::new(1);
    let arena = Arc::new(
        CpuKvArena::new(KvArenaConfig {
            id,
            group,
            page_tokens: 4,
            capacity_pages: 24,
            growth: None,
            dtype: DType::F32,
            layers: vec![KvLayerConfig {
                binding: bindings()[0],
                num_kv_heads: 1,
                key_head_dim: 4,
                value_head_dim: 4,
            }],
        })
        .unwrap(),
    );
    (arena, id, group)
}

fn cache_on(
    (arena, id, group): &(Arc<CpuKvArena>, KvArenaId, KvGroupId),
    blocks: impl IntoIterator<Item = u32>,
    context_len: usize,
) -> PhysicalPagedKvCache {
    let blocks = blocks
        .into_iter()
        .map(|index| CacheBlockRef {
            arena: *id,
            group: *group,
            index,
            slot_generation: 1,
        })
        .collect();
    PhysicalPagedKvCache::new(arena.clone(), bindings(), blocks, context_len).unwrap()
}

fn reservation(state: &ChatDecodeState) -> PhysicalPagedKvCache {
    PhysicalPagedKvCache::new(
        state.physical_kv.arena().clone(),
        bindings(),
        state.physical_kv.blocks.clone(),
        state.physical_kv.context_len(),
    )
    .unwrap()
}

fn new_state(model: &Lfm2ChatModel, config: ChatGenerationConfig) -> ChatDecodeState {
    model
        .begin_resumable_prefill_state_managed(&[3], 64, &config, cache())
        .unwrap()
}

fn inject(
    model: &Lfm2ChatModel,
    state: &mut ChatDecodeState,
    values: &[f32; 7],
) -> Result<ChatDecodeStep> {
    // Isolate selection/decoding while retaining the real request-owned state.
    state.pending_token = None;
    state.unconsumed_output = Some(Tensor::new(&[*values], &Device::Cpu).unwrap());
    model.decode_step(state)
}

fn select(
    model: &Lfm2ChatModel,
    state: &mut ChatDecodeState,
    token: usize,
) -> Result<ChatDecodeStep> {
    let mut values = [-1000f32; 7];
    values[token] = 1000.;
    inject(model, state, &values)
}

#[test]
fn lfm2_decode_rejects_nonfinite_logits_and_never_selects_pad() {
    let model = tiny_lfm_model();
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for index in [0, 3, 6] {
            let mut state = new_state(&model, ChatGenerationConfig::default());
            let mut values = [1f32; 7];
            values[index] = value;
            let error = inject(&model, &mut state, &values)
                .expect_err("reject nonfinite logits")
                .to_string();
            assert!(error.contains("non-finite"), "{error}");
            assert!(state.generated_ids.is_empty());
        }
    }
    // A finite pad peak is masked out, never selected.
    let mut state = new_state(&model, ChatGenerationConfig::default());
    let mut values = [-1000f32; 7];
    values[0] = 1000.;
    values[3] = 0.;
    assert_eq!(inject(&model, &mut state, &values).unwrap().delta, "a");
    assert!(!state.generated_ids.contains(&0));
}

#[test]
fn lfm2_decode_rejects_empty_eos_and_honors_configured_stop() {
    let model = tiny_lfm_model();
    let mut empty = new_state(&model, ChatGenerationConfig::default());
    let error = select(&model, &mut empty, 2)
        .expect_err("empty EOS is a failure")
        .to_string();
    assert!(error.contains("stop_reason=eos"), "{error}");
    assert!(
        model.decode_step(&mut empty).is_err(),
        "failed terminal remains a failure"
    );
    let mut state = new_state(
        &model,
        ChatGenerationConfig {
            stop_token_ids: vec![4],
            ..Default::default()
        },
    );
    assert_eq!(select(&model, &mut state, 3).unwrap().delta, "a");
    let end = select(&model, &mut state, 4).unwrap();
    assert!(end.finished);
    assert_eq!(end.text, "a");
    assert_eq!(end.tokens_generated, 1);
    assert_eq!(state.stop_reason(), Some("configured_stop"));
}

#[test]
fn lfm2_never_selects_a_control_token_greedy_or_sampled() {
    let model = tiny_lfm_model();
    // <|pad|> (0) and <|im_start|> (1) are control tokens; the best
    // generatable id is 4 ("b").
    let logits = [9., 8., -1000., 1., 2., -1000., -1000.];
    let sampled = ChatGenerationConfig {
        temperature: 1.0,
        top_k: 3,
        seed: 7,
        ..Default::default()
    };
    for config in [ChatGenerationConfig::default(), sampled] {
        for _ in 0..8 {
            let mut state = new_state(&model, config.clone());
            let step = inject(&model, &mut state, &logits).expect("control ids are masked");
            assert!(matches!(step.delta.as_str(), "a" | "b"), "{}", step.delta);
        }
    }
    let mut greedy = new_state(&model, ChatGenerationConfig::default());
    assert_eq!(inject(&model, &mut greedy, &logits).unwrap().delta, "b");
}

#[test]
fn lfm2_json_constrained_rows_never_take_the_batched_argmax() {
    let model = tiny_lfm_model();
    let mut plain = new_state(&model, ChatGenerationConfig::default());
    let mut json = new_state(
        &model,
        ChatGenerationConfig {
            constrain_json_object: true,
            ..Default::default()
        },
    );
    // "b" (4) is the argmax, but no JSON value starts with it: the grammar
    // leaves the JSON row only its end-of-turn token, <|im_end|> (2).
    let row = [-1000f32, -1000., 1., 1., 2., -1000., -1000.];
    let logits = Tensor::new(&[row, row], &Device::Cpu).unwrap();
    let picked = model
        .select_batch_tokens(&mut [&mut plain, &mut json], &logits)
        .unwrap();
    assert_eq!((picked[0].0, picked[1].0), (4, 2));
}

#[test]
fn lfm2_batch_selection_reads_back_once_for_mixed_rows() {
    let model = tiny_lfm_model();
    let mut greedy = new_state(&model, ChatGenerationConfig::default());
    let mut sampled = new_state(
        &model,
        ChatGenerationConfig {
            temperature: 1.0,
            top_k: 3,
            seed: 7,
            ..Default::default()
        },
    );
    let mut logprobs = new_state(
        &model,
        ChatGenerationConfig {
            logprobs: true,
            top_logprobs: 2,
            ..Default::default()
        },
    );
    let row = [-1000f32, -1000., 1., 1., 2., 0.5, -1000.];
    let logits = Tensor::new(&[row, row, row], &Device::Cpu).unwrap();
    let before = crate::models::shared::telemetry::thread_host_reads();
    let picked = model
        .select_batch_tokens(&mut [&mut greedy, &mut sampled, &mut logprobs], &logits)
        .unwrap();
    assert_eq!(
        crate::models::shared::telemetry::thread_host_reads() - before,
        1
    );
    assert_eq!(picked[0], (4, None));
    assert!(picked[2].1.is_some());
}

#[test]
fn lfm2_decode_applies_request_history_penalties() {
    let model = tiny_lfm_model();
    let logits = [-1000., -1000., -1000., 2., 1., -1000., -1000.];
    let mut plain = new_state(&model, ChatGenerationConfig::default());
    assert_eq!(inject(&model, &mut plain, &logits).unwrap().delta, "a");
    let mut penalized = new_state(
        &model,
        ChatGenerationConfig {
            repetition_penalty: 4.,
            ..Default::default()
        },
    );
    assert_eq!(inject(&model, &mut penalized, &logits).unwrap().delta, "b");
}

#[test]
fn lfm2_quantum_rollback_restores_seeded_sampler_and_utf8_decoder() {
    let model = tiny_lfm_model();
    let mut state = new_state(
        &model,
        ChatGenerationConfig {
            temperature: 1.,
            seed: 97,
            ..Default::default()
        },
    );
    let checkpoint = state.begin_managed_quantum(reservation(&state)).unwrap();
    let logits = [-1000., -1000., -1000., 0., 0., -1000., -1000.];
    let mut first = String::new();
    for _ in 0..20 {
        first.push_str(&inject(&model, &mut state, &logits).unwrap().delta);
    }
    state.rollback_managed_quantum(checkpoint);
    let mut replay = String::new();
    for _ in 0..20 {
        replay.push_str(&inject(&model, &mut state, &logits).unwrap().delta);
    }
    assert_eq!(replay, first, "RNG state must rewind exactly");
    assert!(
        first.contains('a') && first.contains('b'),
        "seeded sampling must not be greedy: {first}"
    );

    let mut state = new_state(&model, ChatGenerationConfig::default());
    assert_eq!(select(&model, &mut state, 5).unwrap().delta, "");
    let checkpoint = state.begin_managed_quantum(reservation(&state)).unwrap();
    assert_eq!(select(&model, &mut state, 6).unwrap().delta, "é");
    state.rollback_managed_quantum(checkpoint);
    assert_eq!(select(&model, &mut state, 6).unwrap().delta, "é");
    assert_eq!(select(&model, &mut state, 2).unwrap().text, "é");
}

#[test]
fn lfm2_prefill_quantum_rollback_restores_progress_and_can_retry() {
    let model = tiny_lfm_model();
    let mut state = new_state(&model, ChatGenerationConfig::default());
    let checkpoint = state.begin_managed_quantum(reservation(&state)).unwrap();
    model
        .continue_resumable_prefill_managed(&mut state, &[3, 4], 0, 1)
        .unwrap();
    assert_eq!(state.prefill_progress(), 1);
    state.rollback_managed_quantum(checkpoint);
    assert_eq!(state.prefill_progress(), 0);
    model
        .continue_resumable_prefill_managed(&mut state, &[3, 4], 0, 1)
        .unwrap();
    assert_eq!(state.prefill_progress(), 1);
}

#[test]
fn lfm2_repetition_loop_finishes_with_its_text_instead_of_failing() {
    let model = tiny_lfm_model();
    let mut state = model
        .begin_resumable_prefill_state_managed(
            &[3],
            1024,
            &ChatGenerationConfig::default(),
            cache(),
        )
        .unwrap();
    let mut last = None;
    for step in 0..1024 {
        let result = select(&model, &mut state, 3 + step % 2).expect("a loop is not an error");
        if result.finished {
            last = Some(result);
            break;
        }
    }
    let last = last.expect("the loop guard stops the reply");
    assert_eq!(state.stop_reason(), Some("repetition"));
    assert!(last.tokens_generated >= 256 && last.tokens_generated < 1024);
    assert!(last.text.starts_with("abab"), "{}", last.text);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an NVIDIA GPU; run explicitly with --ignored"]
fn cuda_lfm2_decode_rejects_nonfinite_logits_before_argmax() {
    let device = Device::new_cuda(0).expect("CUDA device required");
    let model = tiny_lfm_model();
    for (index, bad) in [
        (0, f32::NAN),
        (32768, f32::INFINITY),
        (65535, f32::NEG_INFINITY),
    ] {
        let mut state = new_state(&model, ChatGenerationConfig::default());
        let mut logits = vec![1f32; 65536];
        logits[index] = bad;
        state.unconsumed_output = Some(Tensor::from_vec(logits, (1, 65536), &device).unwrap());
        let error = model
            .decode_step(&mut state)
            .expect_err("reject GPU nonfinite logits")
            .to_string();
        assert!(error.contains("non-finite"), "{error}");
        assert!(state.generated_ids.is_empty());
    }
}

/// A mixed step whose op shapes equal a split call's (only decode rows, or
/// one prefill row) is that call: same tokens and state.
#[test]
fn lfm2_mixed_steps_with_split_shapes_are_the_split_calls() {
    let model = random_tiny_lfm_model(3);
    let config = ChatGenerationConfig::default();
    let summary = |state: &ChatDecodeState| {
        (
            state.position,
            state.prefill_progress,
            state.pending_token,
            state.generated_ids.clone(),
            state.physical_kv.context_len(),
            state.shortconv.cursor(),
        )
    };
    let prompt = [1, 3, 4, 5];
    let prefilled = || {
        let mut state = model
            .begin_resumable_prefill_state_managed(&prompt, 16, &config, cache())
            .unwrap();
        model
            .continue_resumable_prefill_managed(&mut state, &prompt, 0, prompt.len())
            .unwrap();
        model.decode_step(&mut state).unwrap();
        state
    };
    let (mut split, mut mixed) = (prefilled(), prefilled());
    for _ in 0..3 {
        let expected = model.decode_step_batch(&mut [&mut split]).unwrap();
        let (actual, prefill) = model.step_mixed(&mut [&mut mixed], &mut []).unwrap();
        assert!(prefill.is_empty());
        assert_eq!(expected[0].delta, actual[0].delta);
        assert_eq!(summary(&split), summary(&mixed));
    }

    let prompt = [1, 4, 3, 6, 5];
    let fresh = || {
        model
            .begin_resumable_prefill_state_managed(&prompt, 16, &config, cache())
            .unwrap()
    };
    let (mut split, mut mixed) = (fresh(), fresh());
    for (start, end) in [(0, 3), (3, prompt.len())] {
        let complete = model
            .continue_resumable_prefill_managed(&mut split, &prompt, start, end)
            .unwrap();
        let expected = complete.then(|| model.decode_step(&mut split).unwrap());
        let (decode, mut prefill) = model
            .step_mixed(
                &mut [],
                &mut [MixedPrefillRow {
                    state: &mut mixed,
                    prompt: &prompt,
                    span_start: start,
                    span_end: end,
                }],
            )
            .unwrap();
        assert!(decode.is_empty());
        assert_eq!(
            expected.map(|step| step.delta),
            prefill.remove(0).map(|step| step.delta)
        );
        assert_eq!(summary(&split), summary(&mixed));
    }
}

/// Run the pending token's forward on the retained path and leave its logits
/// for `decode_step` to sample, returning them.
fn pending_logits(model: &Lfm2ChatModel, state: &mut ChatDecodeState) -> Vec<f32> {
    let pending = state.pending_token.take().expect("pending token");
    let input = Tensor::from_slice(&[pending], (1, 1), &Device::Cpu).unwrap();
    let logits = model
        .text_model
        .forward_tokens_retained(
            &input,
            state.position,
            &mut state.physical_kv,
            &mut state.shortconv,
            true,
        )
        .unwrap()
        .expect("logits");
    state.position += 1;
    state.unconsumed_output = Some(logits.clone());
    logits.flatten_all().unwrap().to_vec1().unwrap()
}

/// D3: a hybrid LFM2 session released for capacity mid-generation and
/// rebuilt in a fresh cache and ShortConv state by replaying its prompt and
/// generated tokens over the original span boundaries resumes exactly:
/// bit-identical next-step logits (attention KV and ShortConv state both
/// rebuilt) and the same greedy and seeded streams. A one-span replay runs
/// other op shapes and is held to a tolerance.
#[test]
fn lfm2_replayed_session_resumes_exactly() {
    const MAX_NEW_TOKENS: usize = 20;
    const SUSPEND_AT: usize = 6;
    let model = random_tiny_lfm_model(3);
    let prompt = [1, 3, 4, 5];
    let sampled = ChatGenerationConfig {
        temperature: 1.,
        seed: 97,
        ..Default::default()
    };
    for config in [ChatGenerationConfig::default(), sampled] {
        for one_span in [false, true] {
            let mut caches = crate::models::shared::attention::physical::test_physical_caches_on(
                &Device::Cpu,
                1,
                1,
                4,
                prompt.len() + MAX_NEW_TOKENS + 1,
                3,
            );
            let fresh = caches.pop().unwrap();
            let mut rows = caches
                .into_iter()
                .map(|cache| {
                    let mut state = model
                        .begin_resumable_prefill_state_managed(
                            &prompt,
                            MAX_NEW_TOKENS,
                            &config,
                            cache,
                        )
                        .unwrap();
                    model
                        .continue_resumable_prefill_managed(&mut state, &prompt, 0, prompt.len())
                        .unwrap();
                    state
                })
                .collect::<Vec<_>>();
            let mut suspended = rows.pop().unwrap();
            let mut baseline = rows.pop().unwrap();
            for _ in 0..SUSPEND_AT {
                let step = model.decode_step(&mut baseline).unwrap();
                assert_eq!(model.decode_step(&mut suspended).unwrap().delta, step.delta);
            }
            assert!(
                !baseline.finished,
                "the fixture ended before the suspension"
            );
            let checkpoint = suspended.replay_checkpoint().unwrap();
            assert_eq!(checkpoint.replay_tokens(), prompt.len() + SUSPEND_AT - 1);
            drop(suspended);
            let mut replayed = model.begin_replay_managed(&checkpoint, fresh).unwrap();
            assert!(model.decode_step(&mut replayed).is_err());
            let spans = if one_span {
                vec![(0, checkpoint.replay_tokens())]
            } else {
                std::iter::once((0, prompt.len()))
                    .chain(
                        (prompt.len()..checkpoint.replay_tokens()).map(|start| (start, start + 1)),
                    )
                    .collect()
            };
            for (index, (start, end)) in spans.iter().copied().enumerate() {
                let complete = model
                    .continue_replay(&mut replayed, &prompt, start, end)
                    .unwrap();
                assert_eq!(complete, index + 1 == spans.len());
            }
            assert_eq!(replayed.shortconv.cursor(), baseline.shortconv.cursor());
            let expected = pending_logits(&model, &mut baseline);
            let actual = pending_logits(&model, &mut replayed);
            if one_span {
                let error = expected
                    .iter()
                    .zip(&actual)
                    .map(|(left, right)| abs_err(*left, *right))
                    .fold(0.0, f32::max);
                assert!(error < 1e-4, "one-span replay logits differ by {error}");
                continue;
            }
            assert_eq!(actual, expected, "rebuilt next-step logits");
            while !baseline.finished {
                let step = model.decode_step(&mut baseline).unwrap();
                let resumed = model.decode_step(&mut replayed).unwrap();
                assert_eq!(resumed.delta, step.delta);
                assert_eq!(resumed.text, step.text);
            }
            assert!(replayed.finished);
            assert_eq!(replayed.generated_ids, baseline.generated_ids);
        }
    }
}

/// Non-finite differences saturate to infinity, so they can never pass.
fn abs_err(lhs: f32, rhs: f32) -> f32 {
    let error = (lhs - rhs).abs();
    if error.is_finite() {
        error
    } else {
        f32::INFINITY
    }
}

/// A request that attaches a published prefix (the publisher's pages below a
/// page-aligned boundary and its ShortConv ring at that cursor) prefills and
/// decodes exactly as a fresh request split at the same boundary.
#[test]
fn lfm2_attached_prefix_continues_exactly_as_a_fresh_run() {
    let mut model = random_tiny_lfm_model(21);
    model.set_prefix_caching(true);
    let shared = shared_arena();
    let prompt = [3u32, 4, 5, 6, 3, 4, 5, 6, 3, 4];
    let boundary = 8;
    let config = ChatGenerationConfig::default();

    let begin = |cache| {
        model
            .begin_resumable_prefill_state_managed(&prompt, 16, &config, cache)
            .unwrap()
    };
    let mut fresh = begin(cache_on(&shared, 0..6, 0));
    model
        .continue_resumable_prefill_managed(&mut fresh, &prompt, 0, boundary)
        .unwrap();
    model
        .continue_resumable_prefill_managed(&mut fresh, &prompt, boundary, prompt.len())
        .unwrap();

    let mut publisher = begin(cache_on(&shared, 6..12, 0));
    model
        .continue_resumable_prefill_managed(&mut publisher, &prompt, 0, boundary)
        .unwrap();
    let ring = publisher.shortconv.clone();

    // Two shared prefix pages, then the attacher's own pages.
    let mut attached = begin(cache_on(&shared, [6, 7, 12, 13, 14, 15], boundary));
    assert_eq!(attached.prefill_progress(), boundary);
    attached.shortconv = ring;
    model
        .continue_resumable_prefill_managed(&mut attached, &prompt, boundary, prompt.len())
        .unwrap();

    let logits = |state: &mut ChatDecodeState| {
        state
            .unconsumed_output
            .clone()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    assert_eq!(logits(&mut attached), logits(&mut fresh));
    for _ in 0..4 {
        let expected = model.decode_step(&mut fresh).unwrap();
        let actual = model.decode_step(&mut attached).unwrap();
        assert_eq!(actual.delta, expected.delta);
        if fresh.finished {
            break;
        }
    }

    // Without prefix caching the model refuses a cache that starts above
    // zero, rather than continuing from a reset ring.
    model.set_prefix_caching(false);
    assert!(model
        .begin_resumable_prefill_state_managed(
            &prompt,
            16,
            &config,
            cache_on(&shared, [6, 7, 16, 17], boundary),
        )
        .is_err());
}
