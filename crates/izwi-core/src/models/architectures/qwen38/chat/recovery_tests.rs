//! Exercise numerical draft recovery through real tiny target/MTP forwards.
use super::*;
use crate::backends::kv::{CpuKvArena, KvArenaConfig, KvLayerConfig};
use crate::engine::ModelInstanceId;
use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
use crate::models::architectures::qwen38::mtp::tests::{
    tiny_config, write_tiny_checkpoint, TestDir,
};
use crate::models::architectures::qwen38::native::IndexedSafetensors;
use candle_core::Device;
use safetensors::{tensor::TensorView, Dtype, SafeTensors};
use std::collections::BTreeMap;
use std::sync::Arc;

fn model() -> Qwen38ChatModel {
    model_fixture(false)
}

pub(crate) fn model_fixture(hybrid: bool) -> Qwen38ChatModel {
    let dir = TestDir::new("chat-recovery");
    let mut native = tiny_config();
    if hybrid {
        native.text.block_count = 2;
        native.text.ssm_conv_kernel = 3;
        native.text.full_attention_interval = 2;
        native.layer_types.insert(
            0,
            crate::models::architectures::qwen38::native::Qwen38LayerType::LinearAttention,
        );
    }
    write_tiny_checkpoint(dir.path(), &native);
    // Reuse the zero-projection MTP transformer as the tiny target. Distinct
    // embeddings and an identity-like output head give non-uniform logits.
    let bytes = fs::read(dir.path().join("mtp.safetensors")).unwrap();
    let mtp = SafeTensors::deserialize(&bytes).unwrap();
    let mut views = BTreeMap::new();
    for name in mtp.names() {
        if name.starts_with("mtp.layers.0.") {
            views.insert(
                name.replacen(
                    "mtp.layers.0.",
                    if hybrid {
                        "model.language_model.layers.1."
                    } else {
                        "model.language_model.layers.0."
                    },
                    1,
                ),
                mtp.tensor(name).unwrap(),
            );
        }
    }
    // A nonzero DeltaNet layer makes restoration depend on both convolution
    // history and recurrent state, rather than only the token append cursor.
    let mut hybrid_tensors = Vec::new();
    if hybrid {
        for name in mtp.names() {
            if name.contains(".mlp.")
                || name.ends_with(".input_layernorm.weight")
                || name.ends_with(".post_attention_layernorm.weight")
            {
                views.insert(
                    name.replacen("mtp.layers.0.", "model.language_model.layers.0.", 1),
                    mtp.tensor(name).unwrap(),
                );
            }
        }
        for (name, shape, value) in [
            ("dt_bias", vec![1], 0.1f32),
            ("A_log", vec![1], -1.0),
            ("conv1d.weight", vec![3, 1, 3], 0.25),
            ("norm.weight", vec![1], 1.0),
            ("in_proj_qkv.weight", vec![3, 4], 0.125),
            ("in_proj_z.weight", vec![1, 4], 0.25),
            ("in_proj_a.weight", vec![1, 4], 0.125),
            ("in_proj_b.weight", vec![1, 4], 0.125),
            ("out_proj.weight", vec![4, 1], 0.125),
        ] {
            let bytes = (0..shape.iter().product::<usize>())
                .flat_map(|_| half::bf16::from_f32(value).to_bits().to_le_bytes())
                .collect::<Vec<_>>();
            hybrid_tensors.push((
                format!("model.language_model.layers.0.linear_attn.{name}"),
                shape,
                bytes,
            ));
        }
        for (name, shape, bytes) in &hybrid_tensors {
            views.insert(
                name.clone(),
                TensorView::new(Dtype::BF16, shape.clone(), bytes).unwrap(),
            );
        }
    }
    let matrix: Vec<u8> = (0..32)
        .flat_map(|i| {
            let value = if i % 4 == (i / 4) % 4 { 1.0 } else { 0.125 };
            half::bf16::from_f32(value).to_bits().to_le_bytes()
        })
        .collect();
    let norm = vec![0u8; 8];
    for name in ["model.language_model.embed_tokens.weight", "lm_head.weight"] {
        views.insert(
            name.into(),
            TensorView::new(Dtype::BF16, vec![8, 4], &matrix).unwrap(),
        );
    }
    views.insert(
        "model.language_model.norm.weight".into(),
        TensorView::new(Dtype::BF16, vec![4], &norm).unwrap(),
    );
    safetensors::serialize_to_file(&views, &None, &dir.path().join("target.safetensors")).unwrap();
    let index_path = dir.path().join("model.safetensors.index.json");
    let mut index: serde_json::Value =
        serde_json::from_slice(&fs::read(&index_path).unwrap()).unwrap();
    for name in views.keys() {
        index["weight_map"][name] = serde_json::json!("target.safetensors");
    }
    fs::write(index_path, serde_json::to_vec(&index).unwrap()).unwrap();
    let tensors = IndexedSafetensors::open(dir.path()).unwrap();
    let mut performance = crate::performance::PerformanceConfig::default();
    performance.cuda.mtp_draft_tokens = 2;
    let text_model = Qwen38TextModel::load_native_with_performance(
        &tensors,
        &native,
        &Device::Cpu,
        ProjectionMaterialization::F32,
        &performance.cuda,
    )
    .unwrap();
    let inventory = tensors.validate_mtp_tensor_manifest(&native).unwrap();
    let mtp_head = Qwen38MtpHead::load_native_with_performance(
        &tensors,
        &native,
        &inventory,
        &Device::Cpu,
        ProjectionMaterialization::F32,
        &performance.cuda,
    )
    .unwrap();
    let inner = Tokenizer::from_hf_json_bytes(br#"{
        "version":"1.0","truncation":null,"padding":null,"added_tokens":[],
        "normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,
        "model":{"type":"WordLevel","vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7},"unk_token":"a"}
    }"#).unwrap();
    Qwen38ChatModel {
        device_kind: BackendKind::Cpu,
        performance,
        load_timing: serde_json::json!({}),
        prefill_chunk_size: 4,
        cuda_compute_capability: None,
        kv_storage_provider: Qwen38KvStorageProvider::CpuF32,
        variant: ModelVariant::Qwen3827BFp8,
        tokenizer: Qwen38Tokenizer {
            inner,
            vocab_size: 8,
            specials: SpecialTokenIds {
                im_end: 100,
                eos: 101,
                eos_alt: None,
            },
            literal_special_tokens: Vec::new(),
            chat_template: String::new(),
            default_enable_thinking: false,
        },
        text_config: native.text,
        text_model,
        mtp_policy: Qwen38MtpPolicy::Enabled { draft_tokens: 2 },
        mtp_head: Some(mtp_head),
        prefix_caching: false,
    }
}

fn cache(model_layer: u32) -> PhysicalPagedKvCache {
    let id = KvArenaId {
        model_instance: ModelInstanceId::new(91),
        backend: BackendKind::Cpu,
        device_ordinal: None,
        generation: 1,
    };
    let group = KvGroupId::new(model_layer);
    let binding = KvLayerBinding {
        model_layer,
        physical_layer: 0,
    };
    let arena = Arc::new(
        CpuKvArena::new(KvArenaConfig {
            id,
            group,
            page_tokens: 4,
            capacity_pages: 8,
            growth: None,
            dtype: DType::F32,
            layers: vec![KvLayerConfig {
                binding,
                num_kv_heads: 1,
                key_head_dim: 2,
                value_head_dim: 2,
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
    PhysicalPagedKvCache::new(arena, vec![binding], blocks, 0).unwrap()
}

fn start(model: &Qwen38ChatModel, temperature: f32) -> ChatDecodeState {
    let config = ChatGenerationConfig {
        temperature,
        top_p: 0.9,
        top_k: 6,
        repetition_penalty: 1.1,
        presence_penalty: 0.1,
        seed: 42,
        ..Default::default()
    };
    let prepared = Qwen38PreparedPrompt {
        prompt_ids: vec![1, 2],
        prompt_positions: vec![[0; 3], [1; 3]],
        next_text_position: 2,
    };
    model
        .start_decode_state_physical(&[], 12, &config, Some(&prepared), cache(0), Some(cache(1)))
        .unwrap()
}

fn reservation(cache: &PhysicalPagedKvCache) -> PhysicalPagedKvCache {
    PhysicalPagedKvCache::new(
        cache.arena().clone(),
        vec![cache.layer_binding(0).unwrap()],
        cache.blocks.clone(),
        cache.context_len(),
    )
    .unwrap()
}

fn poison_mtp_cache(state: &ChatDecodeState) {
    use crate::backends::kv::KvWriteArgs;
    use crate::kv::KvSlotRef;
    let cache = state.mtp_physical_kv.as_ref().unwrap();
    let slots = cache
        .arena()
        .lower_slots(&[KvSlotRef {
            block: cache.blocks[0],
            offset: 0,
        }])
        .unwrap();
    let keys = Tensor::zeros((1, 1, 2), DType::F32, &Device::Cpu).unwrap();
    let values = Tensor::full(f32::NAN, (1, 1, 2), &Device::Cpu).unwrap();
    cache
        .arena()
        .write_slots(
            cache.layer_binding(0).unwrap(),
            KvWriteArgs {
                keys: &keys,
                values: &values,
                slots: slots.as_ref(),
            },
        )
        .unwrap()
        .wait()
        .unwrap();
}

#[test]
fn nonfinite_mtp_draft_recovers_to_exact_scalar_sequence() {
    let model = model();
    for (temperature, fail_second_draft) in [(0.0, false), (0.8, false), (0.0, true), (0.8, true)] {
        let mut actual = start(&model, temperature);
        let mut reference = start(&model, temperature);
        reference.adaptive_mtp.disable_after_nonfinite_draft();
        // Emit bootstrap and a healthy scalar step before the fault, matching
        // a stream that fails only after it has already produced valid output.
        for _ in 0..2 {
            model.decode_quantum(&mut actual, 1).unwrap();
            model.decode_quantum(&mut reference, 1).unwrap();
        }
        if fail_second_draft {
            // Proposal one uses a valid anchor and consumes a draft RNG draw.
            // The recurrent forward then reads poisoned V, so proposal two
            // fails after one provisional KV append.
            poison_mtp_cache(&actual);
        } else {
            actual.mtp_anchor_hidden =
                Some(Tensor::full(f32::NAN, (1, 1, 4), &Device::Cpu).unwrap());
        }
        let draft_rng = actual.draft_rng.state;
        let checkpoint = actual
            .begin_shared_step_quantum(
                reservation(&actual.physical_kv),
                actual.mtp_physical_kv.as_ref().map(reservation),
            )
            .unwrap();
        let reference_checkpoint = reference
            .begin_shared_step_quantum(
                reservation(&reference.physical_kv),
                reference.mtp_physical_kv.as_ref().map(reservation),
            )
            .unwrap();
        let recovered = model.decode_quantum(&mut actual, 4).unwrap();
        let expected = model.decode_quantum(&mut reference, 4).unwrap();
        // Check the immediate recovery output before cancellation can erase
        // evidence of an incorrect token, RNG draw or canonical history edit.
        assert_eq!(recovered.delta, expected.delta);
        assert_eq!(recovered.text, expected.text);
        assert_eq!(
            recovered.input_tokens_committed,
            expected.input_tokens_committed
        );
        assert_eq!(actual.history_ids, reference.history_ids);
        assert_eq!(actual.generated_ids, reference.generated_ids);
        assert_eq!(actual.rng.state, reference.rng.state);
        assert_eq!(actual.pending_token, reference.pending_token);
        assert_eq!(actual.next_text_position, reference.next_text_position);
        assert_eq!(
            actual.physical_kv.context_len(),
            reference.physical_kv.context_len()
        );
        assert_eq!(
            actual.mtp_physical_kv.as_ref().unwrap().context_len(),
            actual.physical_kv.context_len()
        );
        assert!(actual.adaptive_mtp.speculation_disabled());
        assert_eq!(actual.draft_rng.state, draft_rng);
        // Cancellation rewinds output and RNG but must not erase the numerical
        // latch, even when the checkpoint's anchor was healthy.
        actual.rollback_shared_step_quantum(checkpoint);
        reference.rollback_shared_step_quantum(reference_checkpoint);
        assert_eq!(actual.generated_ids, reference.generated_ids);
        assert!(actual.adaptive_mtp.speculation_disabled());
        while !actual.finished {
            let step = model.decode_quantum(&mut actual, 4).unwrap();
            let expected = model.decode_quantum(&mut reference, 4).unwrap();
            assert_eq!(step.delta, expected.delta);
            assert_eq!(step.input_tokens_committed, expected.input_tokens_committed);
            assert_eq!(actual.history_ids, reference.history_ids);
            assert_eq!(actual.rng.state, reference.rng.state);
            assert_eq!(actual.draft_rng.state, draft_rng);
            assert_eq!(actual.pending_token, reference.pending_token);
            assert_eq!(actual.next_text_position, reference.next_text_position);
            assert_eq!(
                actual.physical_kv.context_len(),
                reference.physical_kv.context_len()
            );
            assert_eq!(
                actual.mtp_physical_kv.as_ref().unwrap().context_len(),
                actual.physical_kv.context_len()
            );
            assert!(actual.adaptive_mtp.speculation_disabled());
            assert_eq!(actual.finished, reference.finished);
        }
        assert_eq!(actual.tokens_generated, 12);
        assert_eq!(actual.assembled, reference.assembled);
        // The latch belongs to the failed request, not the loaded model.
        assert!(!start(&model, temperature)
            .adaptive_mtp
            .speculation_disabled());
    }
}

#[test]
fn cpu_replay_preserves_published_boundary_and_sampling_with_and_without_mtp() {
    for mtp in [false, true] {
        let mut model = model_fixture(true);
        if !mtp {
            model.mtp_head = None;
        }
        for temperature in [0.0, 0.8] {
            let config = ChatGenerationConfig {
                temperature,
                seed: 42,
                ..Default::default()
            };
            let prepared = Qwen38PreparedPrompt {
                prompt_ids: vec![1, 2],
                prompt_positions: vec![[0; 3], [1; 3]],
                next_text_position: 2,
            };
            let mut uninterrupted = model
                .start_decode_state_physical(
                    &[],
                    12,
                    &config,
                    Some(&prepared),
                    cache(model.text_config.block_count as u32 - 1),
                    mtp.then(|| cache(model.text_config.block_count as u32)),
                )
                .unwrap();
            // Covers unsampled scalar logits and sampled but unpublished MTP bootstrap.
            let mut resumed = model
                .restore_decode_state_physical(
                    &uninterrupted.replay_checkpoint().unwrap(),
                    cache(model.text_config.block_count as u32 - 1),
                    mtp.then(|| cache(model.text_config.block_count as u32)),
                )
                .unwrap();
            while !uninterrupted.finished {
                let expected = model.decode_quantum(&mut uninterrupted, 4).unwrap();
                let actual = model.decode_quantum(&mut resumed, 4).unwrap();
                assert_eq!(actual.delta, expected.delta);
                assert_eq!(actual.tokens_generated, expected.tokens_generated);
                assert_eq!(resumed.generated_ids, uninterrupted.generated_ids);
                assert_eq!(resumed.rng.state, uninterrupted.rng.state);
                assert_eq!(resumed.draft_rng.state, uninterrupted.draft_rng.state);
                assert_eq!(resumed.pending_token, uninterrupted.pending_token);
                assert_eq!(
                    resumed.physical_kv.context_len(),
                    uninterrupted.physical_kv.context_len()
                );
                assert!(
                    resumed.history_ids.is_empty(),
                    "journal must not require penalties"
                );
                if !resumed.finished {
                    let checkpoint = resumed.replay_checkpoint().unwrap();
                    drop(resumed); // No device state from the old session survives restoration.
                    resumed = model
                        .begin_replay_state_physical(
                            &checkpoint,
                            cache(model.text_config.block_count as u32 - 1),
                            mtp.then(|| cache(model.text_config.block_count as u32)),
                        )
                        .unwrap();
                    assert!(model.decode_quantum(&mut resumed, 1).is_err());
                    model.continue_replay_physical(&mut resumed, 0, 1).unwrap();
                    // A second suspension during replay must retain the full
                    // original journal, not just the rebuilt prefix.
                    let again = resumed.replay_checkpoint().unwrap();
                    assert_eq!(again.replay_tokens(), checkpoint.replay_tokens());
                    drop(resumed);
                    resumed = model
                        .begin_replay_state_physical(
                            &again,
                            cache(model.text_config.block_count as u32 - 1),
                            mtp.then(|| cache(model.text_config.block_count as u32)),
                        )
                        .unwrap();
                    for cursor in 0..again.replay_tokens() {
                        let complete = model
                            .continue_replay_physical(&mut resumed, cursor, cursor + 1)
                            .unwrap();
                        assert_eq!(complete, cursor + 1 == again.replay_tokens());
                    }
                    assert!(resumed.replay_tokens().is_none());
                }
            }
            assert_eq!(resumed.assembled, uninterrupted.assembled);
        }
    }
}

#[test]
fn replay_of_partial_prefill_preserves_known_mtp_successor() {
    for mtp in [false, true] {
        let mut model = model_fixture(true);
        if !mtp {
            model.mtp_head = None;
        }
        let config = ChatGenerationConfig {
            seed: 73,
            ..Default::default()
        };
        let prepared = Qwen38PreparedPrompt {
            prompt_ids: vec![1, 2, 3, 4],
            prompt_positions: (0..4).map(|position| [position; 3]).collect(),
            next_text_position: 4,
        };
        let mut original = model
            .begin_chunked_prefill_state_physical(
                &[],
                12,
                &config,
                Some(&prepared),
                cache(model.text_config.block_count as u32 - 1),
                mtp.then(|| cache(model.text_config.block_count as u32)),
            )
            .unwrap();
        model
            .continue_chunked_prefill_physical(
                &mut original,
                &[],
                &config,
                Some(&prepared),
                0,
                2,
                4,
            )
            .unwrap();
        let saved = original.replay_checkpoint().unwrap();
        let mut resumed = model
            .restore_decode_state_physical(
                &saved,
                cache(model.text_config.block_count as u32 - 1),
                mtp.then(|| cache(model.text_config.block_count as u32)),
            )
            .unwrap();
        assert_eq!(resumed.prefill_progress, 2);
        if mtp {
            assert_eq!(resumed.mtp_physical_kv.as_ref().unwrap().context_len(), 2);
        }
        for state in [&mut original, &mut resumed] {
            model
                .continue_chunked_prefill_physical(state, &[], &config, Some(&prepared), 2, 4, 4)
                .unwrap();
        }
        while !original.finished {
            let expected = model.decode_quantum(&mut original, 4).unwrap();
            let actual = model.decode_quantum(&mut resumed, 4).unwrap();
            assert_eq!(actual.delta, expected.delta);
            assert_eq!(resumed.rng.state, original.rng.state);
            assert_eq!(resumed.generated_ids, original.generated_ids);
        }
    }
}

/// Exact small prompt for executor integration fixtures.
pub(crate) fn prepared_prompt() -> Qwen38PreparedPrompt {
    Qwen38PreparedPrompt {
        prompt_ids: vec![1, 2],
        prompt_positions: vec![[0; 3], [1; 3]],
        next_text_position: 2,
    }
}
