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
        let tensor = Tensor::ones(shape, DType::F32, &Device::Cpu).unwrap();
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
fn lfm2_decode_rejects_nonfinite_logits_before_selecting_pad() {
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
    let mut state = new_state(&model, ChatGenerationConfig::default());
    let error = select(&model, &mut state, 0)
        .expect_err("reject finite pad")
        .to_string();
    assert!(error.contains("non-generatable"), "{error}");
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
