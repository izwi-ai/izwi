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
