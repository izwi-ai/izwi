//! Exercise the real loaded-model capability and the adapter stage join without
//! downloading a checkpoint. The tiny GGUF contains both LFM operator kinds.
use super::*;
use crate::backends::DeviceProfile;
use crate::engine::{ExecutionAdapterBinding, ExecutionGroupId, InputRange, ModelInstanceId};
use crate::models::architectures::lfm2::chat::Lfm2ChatModel;
use crate::models::registry::{ChatModelLease, NativeChatModel};
use crate::runtime::LoadedModelBundleDraft;
use crate::runtime::{CapabilityKind, RuntimeAdapterRegistry};
use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};

fn tiny_lfm_model(variant: ModelVariant) -> NativeChatModel {
    let directory =
        std::env::temp_dir().join(format!("izwi-lfm-contract-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&directory).unwrap();
    let tokenizer = tokenizers::Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                [
                    ("[UNK]".to_string(), 0),
                    ("<|im_start|>".to_string(), 1),
                    ("<|im_end|>".to_string(), 2),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap(),
    );
    tokenizer
        .save(directory.join("tokenizer.json"), false)
        .unwrap();
    std::fs::write(
        directory.join("tokenizer_config.json"),
        r#"{"added_tokens_decoder":{"1":{"content":"<|im_start|>"},"2":{"content":"<|im_end|>"}}}"#,
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
    add("token_embd.weight".into(), &[3, 4]);
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
    let model = Lfm2ChatModel::load(&directory, variant, DeviceProfile::cpu()).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
    NativeChatModel::Lfm2(model)
}

#[test]
fn loaded_lfm_chat_profile_selects_published_sequence_stages() {
    for variant in [
        ModelVariant::Lfm2512BInstructGguf,
        ModelVariant::Lfm2512BThinkingGguf,
    ] {
        let model = ChatModelLease::for_test(tiny_lfm_model(variant));
        let draft = LoadedModelBundleDraft::build(
            &RuntimeAdapterRegistry::built_in(),
            ExecutionGroupId::new(1),
            ModelInstanceId::new(1),
            variant,
            BackendKind::Cpu,
        )
        .unwrap();
        draft.seal_chat_workspace(8192).unwrap();
        for contract in draft.execution_contracts(CapabilityKind::Chat).unwrap() {
            let binding = ExecutionAdapterBinding {
                execution_group_id: contract.execution_group_id,
                model_instance_id: contract.model_instance_id,
                adapter_instance_id: contract.adapter_instance_id,
                adapter_abi_revision: contract.adapter_abi_revision,
                model_variant: variant,
                capability_id: "chat".into(),
                stages: contract.stages,
            };
            for streaming in [false, true] {
                let mut request =
                    EngineCoreRequest::chat(vec![crate::models::shared::chat::ChatMessage {
                        role: crate::models::shared::chat::ChatRole::User,
                        content: "Hello".into(),
                    }]);
                request.model_variant = Some(variant);
                request.streaming = streaming;
                request
                    .install_chat_execution_preparation_with_model(
                        variant,
                        vec![1, 0],
                        None,
                        model.clone(),
                        32,
                    )
                    .unwrap();
                let executor = NativeExecutor::new(WorkerConfig {
                    dtype: "bf16".into(),
                    kv_cache_dtype: "bf16".into(),
                    ..Default::default()
                });
                let profile = executor.execution_profile(&request).unwrap();
                assert_eq!(profile.compute_dtype, "f32");
                assert_eq!(profile.kv_dtype, "f32");
                assert!(profile
                    .cache_namespace
                    .as_ref()
                    .unwrap()
                    .ends_with(":f32:f32"));
                assert!(profile.resolved_from_loaded_model);
                assert!(profile.incremental_decode);
                assert_eq!(profile.mode, contract.execution_profile.mode);
                assert_eq!(profile.mode, ExecutionMode::Sequence);
                for phase in [SequencePhase::Prefill, SequencePhase::Decode] {
                    let work = WorkUnit::SequenceStep {
                        phase,
                        input: InputRange { start: 0, end: 1 },
                        max_output_steps: 1,
                        auxiliary_state: None,
                    };
                    binding
                        .stage_for_work(&work)
                        .expect("executor work must have a loaded adapter stage");
                }
                assert!(binding
                    .stage_for_work(&WorkUnit::AtomicJob {
                        kind: "chat".into()
                    })
                    .is_err());
            }
        }
    }
}

#[test]
fn realtime_profiles_check_out_against_the_lfm3_audio_live_asr_graph() {
    let variant = ModelVariant::Lfm3Audio2BGguf;
    let draft = LoadedModelBundleDraft::build(
        &RuntimeAdapterRegistry::built_in(),
        ExecutionGroupId::new(1),
        ModelInstanceId::new(1),
        variant,
        BackendKind::Cpu,
    )
    .unwrap();
    for contract in draft
        .execution_contracts(CapabilityKind::RealtimeAsr)
        .unwrap()
    {
        let mut request = EngineCoreRequest::asr_bytes(Vec::new()).with_model_variant(variant);
        request
            .bind_execution_adapter(ExecutionAdapterBinding {
                execution_group_id: contract.execution_group_id,
                model_instance_id: contract.model_instance_id,
                adapter_instance_id: contract.adapter_instance_id,
                adapter_abi_revision: contract.adapter_abi_revision,
                model_variant: variant,
                capability_id: "realtime_asr".into(),
                stages: contract.stages,
            })
            .unwrap();
        let mut profile = contract.execution_profile;
        let checked = crate::engine::core::EngineCore::apply_adapter_execution_contract(
            &request,
            profile.clone(),
        )
        .unwrap();
        assert_eq!(checked.mode, crate::engine::ExecutionMode::Realtime);
        assert_eq!(checked.prefill_batch, NativeBatchMode::Static);
        assert_eq!(checked.decode_batch, NativeBatchMode::Continuous);
        // A sequence profile names steps the realtime graph has no stage for.
        profile.mode = crate::engine::ExecutionMode::Sequence;
        assert!(
            crate::engine::core::EngineCore::apply_adapter_execution_contract(&request, profile)
                .is_err()
        );
    }
}

#[test]
fn lfm_audio_profiles_report_actual_f32_despite_global_bf16_preference() {
    let executor = NativeExecutor::new(WorkerConfig {
        backend: BackendKind::Cuda,
        dtype: "bf16".into(),
        kv_cache_dtype: "bf16".into(),
        ..Default::default()
    });
    for mut request in [
        EngineCoreRequest::tts("Hello"),
        EngineCoreRequest::asr_bytes(vec![1]),
        EngineCoreRequest::speech_to_speech("audio.wav"),
    ] {
        request.model_variant = Some(ModelVariant::Lfm25Audio15BGguf);
        let profile = executor.execution_profile(&request).unwrap();
        assert_eq!(profile.compute_dtype, "f32");
        assert_eq!(profile.kv_dtype, "f32");
        assert!(profile.cache_namespace.unwrap().ends_with(":f32:f32"));
    }
}

#[test]
fn lfm_chat_prompt_carries_request_tools_in_its_system_turn() {
    use crate::models::shared::chat::{ChatGenerationConfig, ChatMessage, ChatRole};
    let model = tiny_lfm_model(ModelVariant::Lfm2512BInstructGguf);
    let messages = [ChatMessage {
        role: ChatRole::User,
        content: "Hello".into(),
    }];
    let plain = model
        .prompt_token_ids_with_config(&messages, &ChatGenerationConfig::default())
        .unwrap();
    let mut config = ChatGenerationConfig::default();
    config.request.tools = vec![serde_json::json!({"type": "function", "function": {"name": "f"}})];
    let with_tools = model
        .prompt_token_ids_with_config(&messages, &config)
        .unwrap();
    // One more turn (the system turn holding the tool list), so one more
    // <|im_start|> (id 1) than the plain prompt.
    let starts = |ids: &[u32]| ids.iter().filter(|id| **id == 1).count();
    assert_eq!(starts(&with_tools), starts(&plain) + 1);
}

#[test]
fn sealed_lfm_chat_prefill_is_bounded_and_priced_per_token() {
    let variant = ModelVariant::Lfm2512BInstructGguf;
    let draft = LoadedModelBundleDraft::build(
        &RuntimeAdapterRegistry::built_in(),
        ExecutionGroupId::new(1),
        ModelInstanceId::new(1),
        variant,
        BackendKind::Cpu,
    )
    .unwrap();
    draft.seal_chat_workspace(8192).unwrap();
    let prefill_of = |draft: &LoadedModelBundleDraft| {
        draft
            .execution_contracts(CapabilityKind::Chat)
            .unwrap()
            .into_iter()
            .flat_map(|contract| contract.stages.to_vec())
            .find(|stage| stage.selector == crate::engine::StageWorkSelector::SequencePrefill)
            .expect("chat prefill stage")
    };
    // Unsealed: the adapter's prefill stays unbounded and unpriced.
    let unsealed = prefill_of(&draft);
    assert_eq!(unsealed.max_work_units, u64::MAX);
    assert_eq!(unsealed.workspace_per_work_unit_bytes, 0);

    draft.seal_chat_prefill_workspace(147_456, 1_024).unwrap();
    draft.seal_chat_prefill_workspace(147_456, 1_024).unwrap();
    assert!(draft.seal_chat_prefill_workspace(147_456, 512).is_err());
    let sealed = prefill_of(&draft);
    assert_eq!(sealed.max_work_units, 1_024);
    assert_eq!(sealed.workspace_per_work_unit_bytes, 147_456);
    assert_eq!(sealed.max_workspace_bytes, 147_456 * 1_024);
}

#[test]
fn sealed_decode_quantum_fits_the_continuous_decode_work_budget() {
    let variant = ModelVariant::Lfm2512BInstructGguf;
    let draft = LoadedModelBundleDraft::build(
        &RuntimeAdapterRegistry::built_in(),
        ExecutionGroupId::new(1),
        ModelInstanceId::new(1),
        variant,
        BackendKind::Metal,
    )
    .unwrap();
    draft.seal_chat_workspace(8192).unwrap();
    draft.seal_chat_decode_quantum(8).unwrap();
    assert!(draft.seal_chat_decode_quantum(4).is_err());
    let decode = draft
        .execution_contracts(CapabilityKind::Chat)
        .unwrap()
        .into_iter()
        .flat_map(|contract| contract.stages.to_vec())
        .find(|stage| stage.selector == crate::engine::StageWorkSelector::SequenceDecode)
        .expect("chat decode stage");
    // An 8-token isolated quantum must pass the stage's work-unit check on
    // every backend, not only where the batch width is already 8.
    assert!(decode.max_work_units >= 8, "{}", decode.max_work_units);
}

#[test]
fn sustained_decode_quanta_are_honoured_on_every_backend() {
    for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
        let mut profile = crate::engine::ExecutionProfile::fail_closed(
            backend,
            None,
            crate::engine::ExecutionMode::Sequence,
        );
        profile.resolved_from_loaded_model = true;
        profile.incremental_decode = true;
        profile.preferred_decode_tokens = 8;
        profile.sustained_decode_quantum = true;
        assert!(profile.effective_sustained_decode_quantum(), "{backend:?}");
        profile.sustained_decode_quantum = false;
        assert!(!profile.effective_sustained_decode_quantum(), "{backend:?}");
    }
}

#[test]
fn sealed_mixed_step_replaces_the_split_chat_stages() {
    let variant = ModelVariant::Lfm2512BInstructGguf;
    let draft = LoadedModelBundleDraft::build(
        &RuntimeAdapterRegistry::built_in(),
        ExecutionGroupId::new(1),
        ModelInstanceId::new(1),
        variant,
        BackendKind::Metal,
    )
    .unwrap();
    draft.seal_chat_workspace(8192).unwrap();
    draft.seal_chat_prefill_workspace(1_000, 256).unwrap();
    // Unsealed, which is every family today: the split stages.
    for split in draft.execution_contracts(CapabilityKind::Chat).unwrap() {
        assert!(!split.execution_profile.mixed_step);
        assert_eq!(split.stages.len(), 2);
        assert!(split
            .stages
            .iter()
            .all(|stage| stage.batch_mode != crate::engine::NativeBatchMode::Mixed));
    }

    draft.seal_chat_mixed_step(384).unwrap();
    draft.seal_chat_mixed_step(384).unwrap();
    assert!(draft.seal_chat_mixed_step(512).is_err());
    for mixed in draft.execution_contracts(CapabilityKind::Chat).unwrap() {
        assert!(mixed.execution_profile.mixed_step);
        let [stage] = &*mixed.stages else {
            panic!("a mixed contract publishes exactly one stage");
        };
        assert_eq!(stage.name, "chat.step.mixed");
        assert_eq!(
            stage.selector,
            crate::engine::StageWorkSelector::SequenceStep
        );
        assert_eq!(stage.batch_mode, crate::engine::NativeBatchMode::Mixed);
        assert_eq!(stage.shape_policy, crate::engine::StageShapePolicy::Ragged);
        assert_eq!(stage.max_work_units, 384);
        assert_eq!(stage.workspace_per_work_unit_bytes, 1_000);
        let per_row = stage.workspace_per_row_bytes;
        assert!(per_row > 0);
        assert_eq!(
            stage.max_workspace_bytes,
            1_000 * 384 + per_row * stage.max_batch_size as u64
        );
        let binding = ExecutionAdapterBinding {
            execution_group_id: mixed.execution_group_id,
            model_instance_id: mixed.model_instance_id,
            adapter_instance_id: mixed.adapter_instance_id,
            adapter_abi_revision: mixed.adapter_abi_revision,
            model_variant: variant,
            capability_id: "chat".into(),
            stages: mixed.stages.clone(),
        };
        for phase in [SequencePhase::Prefill, SequencePhase::Decode] {
            let work = WorkUnit::SequenceStep {
                phase,
                input: InputRange { start: 0, end: 1 },
                max_output_steps: 1,
                auxiliary_state: None,
            };
            assert_eq!(binding.stage_for_work(&work).unwrap().id, stage.id);
        }
    }
}

/// A request bound to a sealed mixed stage resolves to resumable prefill and
/// continuous decode rows, the profile the mixed stage requires, so the core
/// accepts the contract and records the mixed step.
#[test]
fn mixed_stage_requests_resolve_to_a_mixed_profile() {
    let variant = ModelVariant::Lfm2512BInstructGguf;
    let model = ChatModelLease::for_test(tiny_lfm_model(variant));
    let draft = LoadedModelBundleDraft::build(
        &RuntimeAdapterRegistry::built_in(),
        ExecutionGroupId::new(1),
        ModelInstanceId::new(1),
        variant,
        BackendKind::Cpu,
    )
    .unwrap();
    draft.seal_chat_workspace(8192).unwrap();
    draft.seal_chat_prefill_workspace(1_000, 256).unwrap();
    draft.seal_chat_mixed_step(384).unwrap();
    for contract in draft.execution_contracts(CapabilityKind::Chat).unwrap() {
        let mut request = EngineCoreRequest::chat(vec![crate::models::shared::chat::ChatMessage {
            role: crate::models::shared::chat::ChatRole::User,
            content: "Hello".into(),
        }]);
        request.model_variant = Some(variant);
        request
            .install_chat_execution_preparation_with_model(
                variant,
                vec![1, 0],
                None,
                model.clone(),
                32,
            )
            .unwrap();
        request
            .bind_execution_adapter(ExecutionAdapterBinding {
                execution_group_id: contract.execution_group_id,
                model_instance_id: contract.model_instance_id,
                adapter_instance_id: contract.adapter_instance_id,
                adapter_abi_revision: contract.adapter_abi_revision,
                model_variant: variant,
                capability_id: "chat".into(),
                stages: contract.stages,
            })
            .unwrap();
        // Chunked prefill stays off: the mixed stage alone makes prefill
        // resumable.
        let executor = NativeExecutor::new(WorkerConfig::default());
        let profile = executor.execution_profile(&request).unwrap();
        assert_eq!(profile.prefill, crate::engine::PrefillMode::Incremental);
        assert_eq!(profile.decode_batch, NativeBatchMode::Continuous);
        let profile =
            crate::engine::core::EngineCore::apply_adapter_execution_contract(&request, profile)
                .unwrap();
        assert!(profile.mixed_step);
    }
}
