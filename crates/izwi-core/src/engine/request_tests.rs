/// Special tokens are skipped unless a request turns it off; a request
/// that omits the field keeps the default.
#[test]
fn chat_requests_skip_special_tokens_unless_they_opt_out() {
    let params: super::GenerationParams = serde_json::from_str("{}").unwrap();
    assert!(params.skip_special_tokens);
    let mut request = super::EngineCoreRequest::chat(Vec::new());
    assert!(request.chat_generation_config().skip_special_tokens);
    request.params.skip_special_tokens = false;
    assert!(!request.chat_generation_config().skip_special_tokens);
}

/// KOK-20: a speed outside the engine range is clamped before
/// preparation, so engine admission leaves it unchanged and the prepared
/// artifact still matches the request.
#[test]
fn tts_speed_is_clamped_before_preparation_and_unchanged_by_admission() {
    for (requested, expected) in [(0.25, 0.5), (4.0, 2.0), (1.25, 1.25)] {
        let mut request = super::EngineCoreRequest::tts("Speed probe.");
        request.params.speed = requested;
        request.normalize_tts_params_before_preparation().unwrap();
        assert_eq!(request.params.speed, expected);
        let prepared_speed = request.params.speed;
        let admitted = super::RequestProcessor::new(super::EngineCoreConfig::default())
            .process(request)
            .expect("TTS request admits");
        assert_eq!(admitted.params.speed.to_bits(), prepared_speed.to_bits());
    }
}

#[test]
fn audio_output_credits_survive_commit_transfer_and_abort_returns_capacity() {
    let request = super::EngineCoreRequest::tts("credit test");
    let credit = request
        .try_reserve_audio_output(3 * 1024 * 1024)
        .unwrap()
        .unwrap();
    assert!(request
        .try_reserve_audio_output(2 * 1024 * 1024)
        .unwrap()
        .is_none());
    let committed = request.take_audio_output_credits();
    drop(credit);
    request.begin_stream_staging().unwrap();
    assert!(request
        .try_reserve_audio_output(2 * 1024 * 1024)
        .unwrap()
        .is_none());
    drop(committed);
    let aborted = request
        .try_reserve_audio_output(3 * 1024 * 1024)
        .unwrap()
        .unwrap();
    drop(aborted);
    request.begin_stream_staging().unwrap();
    assert!(request
        .try_reserve_audio_output(3 * 1024 * 1024)
        .unwrap()
        .is_some());
}

#[test]
fn whole_audio_retention_is_explicitly_disabled_only_for_streaming_consumers() {
    let mut request = super::EngineCoreRequest::tts("sample");
    assert!(request.collect_audio_samples);
    request.mark_audio_streaming_only();
    assert!(!request.collect_audio_samples);
}

use super::*;
use crate::backends::BackendKind;
use crate::model::ModelVariant;
use crate::models::shared::chat::ChatRole;

#[test]
fn prepared_asr_audio_is_arc_retained_exactly_and_rejects_source_drift() {
    let variant = ModelVariant::Qwen3Asr06BGguf;
    let mut request = EngineCoreRequest::asr_bytes(vec![1, 2, 3]).with_model_variant(variant);
    request.asr_prompt = Some("original".to_string());
    request
        .install_prepared_asr_audio(variant, vec![0.25, -0.5, 0.75, 0.0], 16_000)
        .unwrap();
    assert_eq!(
        request.prepared_asr_audio_retained_bytes().unwrap(),
        4 * std::mem::size_of::<f32>()
    );
    let (first, sample_rate) = request
        .prepared_asr_audio_for_executor()
        .unwrap()
        .expect("prepared audio");
    assert_eq!(sample_rate, 16_000);
    let cloned = request.clone();
    let (second, _) = cloned
        .prepared_asr_audio_for_executor()
        .unwrap()
        .expect("cloned prepared audio");
    assert!(Arc::ptr_eq(&first, &second));

    request.asr_prompt = Some("changed".to_string());
    assert!(request
        .prepared_asr_audio_for_executor()
        .unwrap_err()
        .to_string()
        .contains("changed after decoded-audio preparation"));
}

#[test]
fn prepared_asr_accessors_reuse_the_preparation_digest_of_the_upload() {
    let full_hashes = || super::ASR_SOURCE_AUDIO_FULL_HASHES.with(|count| count.get());
    let variant = ModelVariant::Qwen3Asr06BGguf;
    let upload = (0..4 * 1024 * 1024).map(|i| i as u8).collect::<Vec<_>>();
    let mut request = EngineCoreRequest::asr_bytes(upload.clone()).with_model_variant(variant);
    request
        .install_prepared_asr_audio(variant, vec![0.25, -0.5], 16_000)
        .unwrap();
    let after_install = full_hashes();
    for _ in 0..50 {
        request.prepared_asr_audio_for_executor().unwrap().unwrap();
    }
    assert_eq!(
        full_hashes(),
        after_install,
        "accessors rehashed the upload"
    );

    // A clone owns another buffer, so its accessor hashes in full once.
    let cloned = request.clone();
    cloned.prepared_asr_audio_for_executor().unwrap().unwrap();
    assert_eq!(full_hashes(), after_install + 1);

    // Replacing the upload with different bytes of the same length is
    // drift, detected by a full hash of the new buffer.
    let mut changed = upload;
    changed[0] ^= 1;
    request.audio_bytes = Some(changed);
    assert!(request
        .prepared_asr_audio_for_executor()
        .unwrap_err()
        .to_string()
        .contains("changed after decoded-audio preparation"));
}

#[test]
fn prepared_asr_input_duration_is_sample_count_over_rate() {
    let variant = ModelVariant::ParakeetTdt06BV3;
    let mut request = EngineCoreRequest::asr_bytes(vec![1, 2, 3]).with_model_variant(variant);
    assert_eq!(request.prepared_asr_input_duration_secs(), None);

    request
        .install_prepared_asr_audio(variant, vec![0.0; 36_000], 24_000)
        .unwrap();
    assert_eq!(request.prepared_asr_input_duration_secs(), Some(1.5));
}

#[test]
fn prepared_qwen_asr_routes_are_mutually_exclusive() {
    let variant = ModelVariant::Qwen3Asr06BGguf;
    let mut normal = EngineCoreRequest::asr_bytes(vec![1, 2, 3]).with_model_variant(variant);
    normal
        .install_prepared_asr_audio(variant, vec![0.0; 16], 16_000)
        .unwrap();
    normal
        .install_prepared_sequence_input_tokens(32, 4096)
        .unwrap();
    assert!(normal.uses_asr_retained_sequence());
    assert!(normal
        .install_prepared_asr_long_form_atomic(
            crate::engine::PlannedAsrChunks::unexecuted_for_tests()
        )
        .is_err());

    let mut long = EngineCoreRequest::asr_bytes(vec![1, 2, 3]).with_model_variant(variant);
    long.install_prepared_asr_audio(variant, vec![0.0; 16], 16_000)
        .unwrap();
    long.install_prepared_asr_long_form_atomic(
        crate::engine::PlannedAsrChunks::unexecuted_for_tests(),
    )
    .unwrap();
    assert!(long.uses_asr_long_form_atomic());
    assert!(long
        .install_prepared_sequence_input_tokens(32, 4096)
        .is_err());
}

#[test]
fn vibevoice_request_projects_only_acoustic_prefill_into_audio_samples() {
    let variant = ModelVariant::VibeVoiceAsr;
    let mut request = EngineCoreRequest::asr_bytes(vec![1, 2, 3]).with_model_variant(variant);
    request
        .install_prepared_asr_audio(variant, vec![0.0; 9_600], 24_000)
        .unwrap();
    request
        .install_prepared_sequence_input_tokens(12, 4096)
        .unwrap();
    request
        .install_prepared_vibevoice_artifact(
            variant,
            Arc::new(VibeVoiceAsrPreparedArtifact::for_test(12, 4..7, 9_600, 8).unwrap()),
        )
        .unwrap();

    let profile = super::super::ExecutionProfile::fail_closed(
        BackendKind::Cpu,
        Some(variant),
        super::super::ExecutionMode::Sequence,
    );
    let mut prefill = StageDescriptor::from_execution_profile(
        StageId::new(1),
        crate::models::architectures::vibevoice::VIBEVOICE_ASR_PREFILL_STAGE,
        &profile,
        super::super::NativeBatchMode::None,
    );
    prefill.retained_state_selections = Some(vec![super::super::ClockedStateSelection::new(
        crate::models::architectures::vibevoice::VIBEVOICE_ASR_TOKENIZER_GROUP,
        crate::kv::v2::StateClock::AudioSamples,
    )
    .unwrap()]);

    let before = request
        .project_auxiliary_state_spans(&prefill, InputRange::new(0, 4).unwrap())
        .unwrap()
        .unwrap();
    assert!(before.is_empty());
    let overlap = request
        .project_auxiliary_state_spans(&prefill, InputRange::new(3, 6).unwrap())
        .unwrap()
        .unwrap();
    assert_eq!(overlap.len(), 1);
    assert_eq!(overlap[0].input(), InputRange::new(0, 6_400).unwrap());

    let mut unauthorized = prefill.clone();
    unauthorized.retained_state_selections = Some(vec![]);
    assert!(request
        .project_auxiliary_state_spans(&unauthorized, InputRange::new(4, 5).unwrap())
        .is_err());
    let mut legacy = prefill.clone();
    legacy.retained_state_selections = None;
    assert!(request
        .project_auxiliary_state_spans(&legacy, InputRange::new(4, 5).unwrap())
        .unwrap()
        .is_none());

    let mut decode = prefill;
    decode.name = crate::models::architectures::vibevoice::VIBEVOICE_ASR_DECODE_STAGE.into();
    decode.retained_state_selections = Some(vec![]);
    assert!(request
        .project_auxiliary_state_spans(&decode, InputRange::new(12, 13).unwrap())
        .unwrap()
        .unwrap()
        .is_empty());
}

#[test]
fn maximal_stream_backpressure_timeout_fails_without_panicking() {
    let error = checked_stream_backpressure_deadline(u64::MAX)
        .expect_err("oversized public timeout must fail closed");
    assert!(matches!(error, Error::InvalidInput(message) if message.contains("60000ms limit")));
    assert!(checked_stream_backpressure_deadline(STREAM_BACKPRESSURE_MAX_TIMEOUT_MS).is_ok());

    let mut request = EngineCoreRequest::tts("bounded policy").with_stream_policy(
        EngineStreamPolicy::BlockWithDeadline {
            timeout_ms: u64::MAX,
        },
    );
    request.streaming = true;
    assert!(matches!(
        RequestProcessor::new(EngineCoreConfig::default()).process(request),
        Err(Error::InvalidInput(message)) if message.contains("60000ms limit")
    ));
}

#[test]
fn model_instance_binding_is_idempotent_and_fenced() {
    let mut request = EngineCoreRequest::tts("hello");
    let first = super::super::ModelInstanceId::new(7);
    let second = super::super::ModelInstanceId::new(8);

    assert_eq!(request.model_instance_id(), None);
    request.bind_model_instance(first).expect("initial binding");
    request
        .bind_model_instance(first)
        .expect("idempotent binding");
    assert_eq!(request.model_instance_id(), Some(first));
    assert!(request.bind_model_instance(second).is_err());
    assert_eq!(request.model_instance_id(), Some(first));
}

#[test]
fn execution_adapter_binding_is_exact_and_idempotent() {
    let variant = ModelVariant::Kokoro82M;
    let instance = super::super::ModelInstanceId::new(7);
    let profile = super::super::ExecutionProfile::fail_closed(
        crate::backends::BackendKind::Cpu,
        Some(variant),
        super::super::ExecutionMode::Atomic,
    );
    let stage = super::super::StageDescriptor::from_execution_profile(
        super::super::StageId::new(0),
        "tts.compatibility",
        &profile,
        super::super::NativeBatchMode::None,
    );
    let binding = super::super::ExecutionAdapterBinding {
        execution_group_id: super::super::ExecutionGroupId::new(1),
        model_instance_id: instance,
        adapter_instance_id: super::super::AdapterInstanceId::new(2),
        adapter_abi_revision: super::super::AdapterAbiRevision::new(1),
        model_variant: variant,
        capability_id: "tts".to_string(),
        stages: std::sync::Arc::from([stage]),
    };
    let mut request = EngineCoreRequest::tts("hello").with_model_variant(variant);

    request
        .bind_execution_adapter(binding.clone())
        .expect("initial binding");
    request
        .bind_execution_adapter(binding.clone())
        .expect("idempotent binding");

    assert_eq!(request.model_instance_id(), Some(instance));
    assert_eq!(request.execution_adapter_binding(), Some(&binding));
    let mut mismatched = binding;
    mismatched.adapter_instance_id = super::super::AdapterInstanceId::new(3);
    assert!(request.bind_execution_adapter(mismatched).is_err());
}

#[test]
fn lfm_asr_prefill_and_decode_costs_remain_separate_across_rebinding() {
    let variant = ModelVariant::Lfm25Audio15BGguf;
    // The production ASR adapter needs an encoder seal; use its sequence
    // selectors here to exercise request cost lifetime independently of
    // checkpoint loading.
    let profile = super::super::ExecutionProfile::fail_closed(
        BackendKind::Cuda,
        Some(variant),
        super::super::ExecutionMode::Sequence,
    );
    let mut stages = Vec::new();
    for (id, selector, mode) in [
        (
            1,
            super::super::StageWorkSelector::SequencePrefill,
            super::super::NativeBatchMode::Static,
        ),
        (
            2,
            super::super::StageWorkSelector::SequenceDecode,
            super::super::NativeBatchMode::Continuous,
        ),
    ] {
        let mut stage = super::super::StageDescriptor::from_execution_profile(
            StageId::new(id),
            "asr.resource-test",
            &profile,
            mode,
        );
        stage.selector = selector;
        stage.max_workspace_bytes = 16_384;
        stages.push(stage);
    }
    let binding = super::super::ExecutionAdapterBinding {
        execution_group_id: super::super::ExecutionGroupId::new(1),
        model_instance_id: super::super::ModelInstanceId::new(2),
        adapter_instance_id: super::super::AdapterInstanceId::new(3),
        adapter_abi_revision: super::super::AdapterAbiRevision::new(1),
        model_variant: variant,
        capability_id: "asr".into(),
        stages: Arc::from(stages),
    };
    let mut request = EngineCoreRequest::asr_bytes(vec![1]).with_model_variant(variant);
    request.bind_execution_adapter(binding.clone()).unwrap();
    let mut requested = Vec::new();
    let envelope_for = |selector| {
        let (work_units, materialized_tensor_elements, workspace_bytes) =
            if selector == super::super::StageWorkSelector::SequencePrefill {
                (32, 128, 8192)
            } else {
                (1, 4, 1024)
            };
        Ok(crate::models::architectures::lfm25_audio::model::Lfm25AudioAsrStepResourceEnvelope {
            backend: BackendKind::Cuda,
            work_units, materialized_tensor_elements,
            host_workspace_bytes: 0,
            device_workspace_bytes: workspace_bytes,
            unified_workspace_bytes: 0,
            workspace_bytes,
        })
    };
    let costs = EngineCoreRequest::lfm25_audio_asr_stage_costs_from_envelopes(
        Some(&binding),
        |selector| {
            requested.push(selector);
            envelope_for(selector)
        },
    )
    .unwrap();
    assert_eq!(
        requested,
        vec![
            super::super::StageWorkSelector::SequencePrefill,
            super::super::StageWorkSelector::SequenceDecode,
        ]
    );
    assert_eq!(costs.len(), 2);
    let prefill = costs[0].1;
    let decode = costs[1].1;
    assert_eq!(prefill.logical_units, 32);
    assert_eq!(prefill.workspace.workspace_bytes().unwrap(), 8192);
    assert_eq!(decode.logical_units, 1);
    assert_eq!(decode.workspace.workspace_bytes().unwrap(), 1024);
    let mut undersized = binding.clone();
    let mut undersized_stages = undersized.stages.to_vec();
    undersized_stages[0].max_workspace_bytes = 8191;
    undersized.stages = Arc::from(undersized_stages);
    assert!(
        EngineCoreRequest::lfm25_audio_asr_stage_costs_from_envelopes(
            Some(&undersized),
            envelope_for,
        )
        .is_err()
    );
    assert!(request.prepared_stage_costs.is_empty());
    for _ in 0..2 {
        request
            .install_prepared_stage_cost(StageId::new(1), prefill)
            .unwrap();
        request
            .install_prepared_stage_cost(StageId::new(2), decode)
            .unwrap();
        request.bind_execution_adapter(binding.clone()).unwrap();
    }
    assert_eq!(request.prepared_stage_costs.len(), 2);
    assert_eq!(request.prepared_stage_cost(StageId::new(1)), Some(prefill));
    assert_eq!(request.prepared_stage_cost(StageId::new(2)), Some(decode));
}

#[test]
fn qwen_scalar_prefill_is_not_a_native_prepared_cost_stage() {
    let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
    let profile = super::super::ExecutionProfile::fail_closed(
        crate::backends::BackendKind::Metal,
        Some(variant),
        super::super::ExecutionMode::Sequence,
    );
    let mut stage = super::super::StageDescriptor::from_execution_profile(
        super::super::StageId::new(1),
        "tts.prefill.physical",
        &profile,
        super::super::NativeBatchMode::None,
    );
    stage.selector = super::super::StageWorkSelector::SequencePrefill;
    let binding = super::super::ExecutionAdapterBinding {
        execution_group_id: super::super::ExecutionGroupId::new(1),
        model_instance_id: super::super::ModelInstanceId::new(2),
        adapter_instance_id: super::super::AdapterInstanceId::new(3),
        adapter_abi_revision: super::super::AdapterAbiRevision::new(1),
        model_variant: variant,
        capability_id: "tts".to_string(),
        stages: std::sync::Arc::from([stage]),
    };

    assert!(EngineCoreRequest::native_prepared_stage(
        Some(&binding),
        super::super::StageWorkSelector::SequencePrefill,
    )
    .is_none());
}

#[test]
fn qwen_tts_output_budget_fits_the_loaded_context_before_admission() {
    let fitted = EngineCoreRequest::fit_qwen_tts_layout_to_loaded_context(
        crate::models::architectures::qwen3::tts::Qwen3TtsPhysicalSessionLayout {
            prefill_tokens: 9,
            max_frames: ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES,
        },
        1_024,
        "request-1",
    )
    .expect("output budget should fit");

    assert_eq!(fitted.prefill_tokens, 9);
    assert_eq!(fitted.max_frames, 1_015);
    assert_eq!(fitted.prefill_tokens + fitted.max_frames, 1_024);
    assert!(EngineCoreRequest::fit_qwen_tts_layout_to_loaded_context(
        crate::models::architectures::qwen3::tts::Qwen3TtsPhysicalSessionLayout {
            prefill_tokens: 1_024,
            max_frames: 1,
        },
        1_024,
        "request-2",
    )
    .is_err());
}

#[test]
fn test_tts_request() {
    let request = EngineCoreRequest::tts("Hello, world!");
    assert_eq!(request.task_type, TaskType::TTS);
    assert_eq!(request.workload_class, WorkloadClass::Online);
    assert_eq!(request.text.as_deref(), Some("Hello, world!"));
    match &request.task {
        EngineTask::Tts(input) => assert!(input.text.is_empty()),
        other => panic!("unexpected task payload: {other:?}"),
    }

    let processed = RequestProcessor::new(EngineCoreConfig::default())
        .process(request)
        .expect("TTS request should canonicalize");
    match &processed.task {
        EngineTask::Tts(input) => assert_eq!(input.text, "Hello, world!"),
        other => panic!("unexpected task payload: {other:?}"),
    }
}

#[test]
fn direct_request_constructors_move_large_inputs_without_cloning() {
    let mut audio = Vec::with_capacity(64);
    audio.extend_from_slice(&[1, 2, 3]);
    let audio_ptr = audio.as_ptr();
    let audio_capacity = audio.capacity();
    let request = EngineCoreRequest::asr_bytes(audio);
    let retained_audio = request.audio_bytes.as_ref().expect("compatibility audio");
    assert_eq!(retained_audio.as_ptr(), audio_ptr);
    assert_eq!(retained_audio.capacity(), audio_capacity);
    assert!(matches!(
        &request.task,
        EngineTask::Asr(AsrEngineInput {
            audio: EngineAudioInput::Bytes(bytes),
            ..
        }) if bytes.capacity() == 0
    ));

    let mut text = String::with_capacity(64);
    text.push_str("bounded later");
    let text_ptr = text.as_ptr();
    let request = EngineCoreRequest::tts(text);
    assert_eq!(
        request.text.as_ref().expect("compatibility text").as_ptr(),
        text_ptr
    );
    assert!(matches!(
        &request.task,
        EngineTask::Tts(TtsEngineInput { text, .. }) if text.capacity() == 0
    ));

    let mut reference = String::with_capacity(64);
    reference.push_str("AQID");
    let reference_ptr = reference.as_ptr();
    let request = EngineCoreRequest::tts("hello").with_reference(reference, "transcript");
    assert_eq!(
        request
            .reference_audio
            .as_ref()
            .expect("compatibility reference")
            .as_ptr(),
        reference_ptr
    );
    assert!(matches!(
        &request.task,
        EngineTask::Tts(TtsEngineInput {
            reference_audio: None,
            ..
        })
    ));
}

#[test]
fn direct_payload_preflight_rejects_before_moving_or_cloning() {
    let limits = DirectPayloadLimits {
        audio_source_bytes: 3,
        reference_source_bytes: 3,
        reference_text_chars: 64,
        metadata_bytes: 64,
        collection_items: 64,
    };
    let mut asr = EngineCoreRequest::asr_bytes(vec![1, 2, 3, 4]);
    let asr_ptr = asr.audio_bytes.as_ref().unwrap().as_ptr();
    let error = asr
        .canonicalize_direct_payloads_with_limits(limits)
        .expect_err("oversized bytes must fail before canonicalization");
    assert!(error.to_string().contains("source limit"));
    assert_eq!(asr.audio_bytes.as_ref().unwrap().as_ptr(), asr_ptr);
    assert!(matches!(
        &asr.task,
        EngineTask::Asr(AsrEngineInput {
            audio: EngineAudioInput::Bytes(bytes),
            ..
        }) if bytes.is_empty()
    ));

    let mut qwen_reference = EngineCoreRequest::tts("hello")
        .with_model_variant(ModelVariant::Qwen3Tts12Hz06BBase)
        .with_reference("AAAAAAAA", "reference transcript");
    let reference_ptr = qwen_reference.reference_audio.as_ref().unwrap().as_ptr();
    let error = qwen_reference
        .canonicalize_direct_payloads_with_limits(limits)
        .expect_err("oversized Qwen reference must fail before canonicalization");
    assert!(error.to_string().contains("3-byte limit"));
    assert_eq!(
        qwen_reference.reference_audio.as_ref().unwrap().as_ptr(),
        reference_ptr
    );
    assert!(matches!(
        &qwen_reference.task,
        EngineTask::Tts(TtsEngineInput {
            reference_audio: None,
            ..
        })
    ));
}

#[test]
fn direct_payload_preflight_rejects_whitespace_amplification_for_audio_and_reference() {
    let limits = DirectPayloadLimits {
        audio_source_bytes: 3,
        reference_source_bytes: 3,
        reference_text_chars: 64,
        metadata_bytes: 64,
        collection_items: 64,
    };
    // Three decoded bytes permit four base64 bytes plus the explicit
    // 1-KiB metadata/whitespace allowance, never unbounded whitespace.
    let amplified = " ".repeat(4 + 1024 + 1);
    let mut asr = EngineCoreRequest::asr(amplified.clone());
    let error = asr
        .canonicalize_direct_payloads_with_limits(limits)
        .expect_err("inference audio whitespace amplification must fail");
    assert!(error.to_string().contains("encoded input limit"));

    let mut qwen_reference = EngineCoreRequest::tts("hello")
        .with_model_variant(ModelVariant::Qwen3Tts12Hz06BBase)
        .with_reference(amplified, "reference transcript");
    let error = qwen_reference
        .canonicalize_direct_payloads_with_limits(limits)
        .expect_err("reference audio whitespace amplification must fail");
    assert!(error.to_string().contains("encoded input limit"));
}

#[test]
fn direct_payload_canonicalization_preserves_legacy_precedence_and_single_ownership() {
    let limits = DirectPayloadLimits {
        audio_source_bytes: 64,
        reference_source_bytes: 64,
        reference_text_chars: 64,
        metadata_bytes: 128,
        collection_items: 64,
    };
    let mut legacy_wins = EngineCoreRequest::asr("AQID");
    legacy_wins.language = Some("legacy-language".to_string());
    legacy_wins.asr_prompt = Some("legacy prompt".to_string());
    legacy_wins.task = EngineTask::Asr(AsrEngineInput {
        audio: EngineAudioInput::Bytes(vec![9, 9, 9]),
        language: Some("typed-language".to_string()),
        prompt: Some("typed prompt".to_string()),
    });
    legacy_wins
        .canonicalize_direct_payloads_with_limits(limits)
        .expect("legacy base64 should canonicalize");
    assert!(legacy_wins.audio_input.is_none());
    assert!(legacy_wins.audio_bytes.is_none());
    assert_eq!(legacy_wins.audio_base64_for_execution(), Some("AQID"));
    assert!(matches!(
        &legacy_wins.task,
        EngineTask::Asr(AsrEngineInput {
            audio: EngineAudioInput::Base64(value),
            language: Some(language),
            prompt: Some(prompt),
            ..
        }) if value == "AQID"
            && language == "legacy-language"
            && prompt == "legacy prompt"
    ));

    let mut bytes_win = EngineCoreRequest::asr("AQID");
    bytes_win.audio_bytes = Some(vec![1, 2, 3]);
    bytes_win
        .canonicalize_direct_payloads_with_limits(limits)
        .expect("legacy bytes should have historical precedence");
    assert!(bytes_win.audio_input.is_none());
    assert!(bytes_win.audio_bytes.is_none());
    assert_eq!(bytes_win.audio_bytes_for_execution(), Some(&[1, 2, 3][..]));
}

#[test]
fn direct_payload_canonicalization_preserves_typed_only_asr_and_speech_metadata() {
    let limits = DirectPayloadLimits {
        audio_source_bytes: 64,
        reference_source_bytes: 64,
        reference_text_chars: 64,
        metadata_bytes: 256,
        collection_items: 64,
    };
    let mut asr = EngineCoreRequest::asr_bytes(vec![1]);
    asr.audio_bytes = None;
    asr.task = EngineTask::Asr(AsrEngineInput {
        audio: EngineAudioInput::Bytes(vec![1, 2, 3]),
        language: Some("typed-language".to_string()),
        prompt: Some("typed prompt".to_string()),
    });
    asr.canonicalize_direct_payloads_with_limits(limits)
        .expect("typed-only ASR request should canonicalize");
    assert_eq!(asr.asr_language_for_execution(), Some("typed-language"));
    assert_eq!(asr.asr_prompt_for_execution(), Some("typed prompt"));
    assert_eq!(asr.audio_bytes_for_execution(), Some(&[1, 2, 3][..]));

    let typed_messages = vec![ChatMessage {
        role: ChatRole::User,
        content: "typed history".to_string(),
    }];
    let mut speech = EngineCoreRequest::speech_to_speech_bytes(vec![1]);
    speech.audio_bytes = None;
    speech.task = EngineTask::SpeechToSpeech(AudioChatEngineInput {
        audio: EngineAudioInput::Bytes(vec![4, 5, 6]),
        messages: typed_messages.clone(),
        system_prompt: Some("typed system".to_string()),
    });
    speech
        .canonicalize_direct_payloads_with_limits(limits)
        .expect("typed-only speech request should canonicalize");
    assert!(EngineCoreRequest::chat_messages_match(
        speech.speech_messages_for_execution(),
        &typed_messages
    ));
    assert_eq!(
        speech.speech_system_prompt_for_execution(),
        Some("typed system")
    );

    speech.chat_messages = Some(vec![ChatMessage {
        role: ChatRole::Assistant,
        content: "legacy history".to_string(),
    }]);
    speech.system_prompt = Some("legacy system".to_string());
    speech
        .canonicalize_direct_payloads_with_limits(limits)
        .expect("legacy speech metadata should retain precedence");
    assert_eq!(
        speech.speech_messages_for_execution()[0].content,
        "legacy history"
    );
    assert_eq!(
        speech.speech_system_prompt_for_execution(),
        Some("legacy system")
    );
}

#[test]
fn qwen_reference_canonicalization_synchronizes_typed_execution_view() {
    let limits = DirectPayloadLimits {
        audio_source_bytes: 64,
        reference_source_bytes: 64,
        reference_text_chars: 64,
        metadata_bytes: 128,
        collection_items: 64,
    };
    let mut request = EngineCoreRequest::tts("hello")
        .with_model_variant(ModelVariant::Qwen3Tts12Hz06BBase)
        .with_reference("AQID", "reference transcript");
    request
        .canonicalize_direct_payloads_with_limits(limits)
        .expect("Qwen reference should canonicalize");
    assert!(request.reference_audio.is_none());
    assert!(request.reference_text.is_none());
    assert_eq!(request.tts_reference_audio_for_execution(), Some("AQID"));
    assert_eq!(
        request.tts_reference_text_for_execution(),
        Some("reference transcript")
    );
    match &request.task {
        EngineTask::Tts(input) => {
            assert_eq!(input.text, "hello");
            assert_eq!(input.reference_audio.as_deref(), Some("AQID"));
            assert_eq!(
                input.reference_text.as_deref(),
                Some("reference transcript")
            );
        }
        other => panic!("unexpected task payload: {other:?}"),
    }
}

#[test]
fn direct_non_chat_metadata_has_one_checked_context_derived_bound() {
    let limits = DirectPayloadLimits {
        audio_source_bytes: 64,
        reference_source_bytes: 64,
        reference_text_chars: 8,
        metadata_bytes: 8,
        collection_items: 1,
    };

    let mut tts = EngineCoreRequest::tts("123456789");
    assert!(tts
        .canonicalize_direct_payloads_with_limits(limits)
        .expect_err("oversized TTS metadata must fail")
        .to_string()
        .contains("TTS metadata"));

    let mut asr = EngineCoreRequest::asr_bytes(vec![1])
        .with_language("12345")
        .with_asr_prompt("6789");
    assert!(asr
        .canonicalize_direct_payloads_with_limits(limits)
        .expect_err("aggregate ASR metadata must fail")
        .to_string()
        .contains("ASR metadata"));

    let mut speech = EngineCoreRequest::speech_to_speech_bytes(vec![1]);
    speech.chat_messages = Some(vec![ChatMessage {
        role: ChatRole::User,
        content: "123456789".to_string(),
    }]);
    assert!(speech
        .canonicalize_direct_payloads_with_limits(limits)
        .expect_err("aggregate speech metadata must fail")
        .to_string()
        .contains("speech-to-speech metadata"));
}

#[test]
fn direct_dynamic_allocations_have_aggregate_and_count_bounds() {
    let limits = DirectPayloadLimits {
        audio_source_bytes: 64,
        reference_source_bytes: 64,
        reference_text_chars: 64,
        metadata_bytes: 64,
        collection_items: 2,
    };

    let mut oversized_voice = EngineCoreRequest::tts("ok");
    oversized_voice.params.voice = Some("v".repeat(65));
    assert!(oversized_voice
        .canonicalize_direct_payloads_with_limits(limits)
        .expect_err("voice allocation must share the metadata bound")
        .to_string()
        .contains("request metadata"));

    let mut excessive_stops = EngineCoreRequest::tts("ok");
    excessive_stops.params.stop_sequences = vec![String::new(); 3];
    assert!(excessive_stops
        .canonicalize_direct_payloads_with_limits(limits)
        .expect_err("stop-sequence count must be bounded")
        .to_string()
        .contains("stop sequences"));

    let mut excessive_prompt_tokens = EngineCoreRequest::asr_bytes(vec![1]);
    excessive_prompt_tokens.prompt_tokens = vec![1, 2, 3];
    assert!(excessive_prompt_tokens
        .canonicalize_direct_payloads_with_limits(limits)
        .expect_err("non-chat prompt-token count must be bounded")
        .to_string()
        .contains("prompt tokens"));
}

#[test]
fn direct_preflight_rejects_small_payloads_with_oversized_retained_capacity() {
    let limits = DirectPayloadLimits {
        audio_source_bytes: 64,
        reference_source_bytes: 64,
        reference_text_chars: 64,
        metadata_bytes: 64,
        collection_items: 64,
    };

    let mut text = String::with_capacity(1024);
    text.push_str("ok");
    let mut request = EngineCoreRequest::tts(text);
    assert!(request
        .canonicalize_direct_payloads_with_limits(limits)
        .expect_err("small text with oversized retained capacity must fail")
        .to_string()
        .contains("TTS metadata"));

    let mut audio = Vec::with_capacity(1024);
    audio.push(1);
    let mut request = EngineCoreRequest::asr_bytes(audio);
    assert!(request
        .canonicalize_direct_payloads_with_limits(limits)
        .expect_err("small audio with oversized retained capacity must fail")
        .to_string()
        .contains("retains"));
}

#[test]
fn streaming_request_defaults_to_streaming_workload_class() {
    let request = EngineCoreRequest::tts("Hello").with_streaming(true);
    assert_eq!(request.workload_class, WorkloadClass::Streaming);

    let request = RequestBuilder::tts("Hello")
        .workload_class(WorkloadClass::Realtime)
        .streaming()
        .build();
    assert_eq!(request.workload_class, WorkloadClass::Realtime);
}

#[test]
fn test_request_builder() {
    let request = RequestBuilder::tts("Hello")
        .voice("us_female")
        .temperature(0.8)
        .max_tokens(1024)
        .streaming()
        .build();

    assert!(request.streaming);
    assert_eq!(request.params.temperature, 0.8);
    assert_eq!(request.params.max_tokens, 1024);
}

#[test]
fn test_request_processor() {
    let config = EngineCoreConfig::default();
    let processor = RequestProcessor::new(config);

    let request = EngineCoreRequest::tts("Test");
    let processed = processor.process(request);
    assert!(processed.is_ok());
}

#[test]
fn test_request_processor_preserves_tts_auto_max_tokens() {
    let config = EngineCoreConfig::default();
    let processor = RequestProcessor::new(config);

    let mut request = EngineCoreRequest::tts("Test");
    request.params.max_tokens = 0;

    let processed = processor.process(request).expect("request should process");
    assert_eq!(processed.params.max_tokens, 0);
}

#[test]
fn test_request_processor_clamps_tts_to_model_native_limit() {
    let config = EngineCoreConfig::default();
    let processor = RequestProcessor::new(config);

    let mut request = EngineCoreRequest::tts("Test");
    request.model_variant = Some(ModelVariant::Qwen3Tts12Hz17BVoiceDesign);
    request.params.max_tokens = 20_000;

    let processed = processor.process(request).expect("request should process");
    assert_eq!(
        processed.params.max_tokens,
        ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES
    );
}

#[test]
fn test_request_processor_keeps_tts_above_engine_seq_len_when_model_allows() {
    let config = EngineCoreConfig::default();
    let processor = RequestProcessor::new(config);

    let mut request = EngineCoreRequest::tts("Test");
    request.model_variant = Some(ModelVariant::Qwen3Tts12Hz06BBase);
    request.params.max_tokens = 5000;

    let processed = processor.process(request).expect("request should process");
    assert_eq!(processed.params.max_tokens, 5000);
}

#[test]
fn request_processor_unlocks_explicit_cuda_audio_contexts() {
    for (variant, expected) in [
        (ModelVariant::Lfm25Audio15BGguf, 5000),
        (ModelVariant::FishAudioS2Pro, 5000),
        (ModelVariant::Voxtral4BTts2603, 2048),
    ] {
        let processor = RequestProcessor::new(EngineCoreConfig {
            backend: BackendKind::Cuda,
            ..EngineCoreConfig::default()
        });
        let mut request = EngineCoreRequest::tts("Test");
        request.model_variant = Some(variant);
        request.params.max_tokens = 5000;

        let processed = processor.process(request).expect("CUDA TTS request");
        assert_eq!(processed.params.max_tokens, expected, "{variant}");
    }

    for backend in [BackendKind::Cpu, BackendKind::Metal] {
        let processor = RequestProcessor::new(EngineCoreConfig {
            backend,
            ..EngineCoreConfig::default()
        });
        let mut request = EngineCoreRequest::tts("Test");
        request.model_variant = Some(ModelVariant::FishAudioS2Pro);
        request.params.max_tokens = 5000;

        let processed = processor.process(request).expect("portable TTS request");
        assert_eq!(processed.params.max_tokens, 4096, "{backend:?}");
    }
}

#[test]
fn test_request_processor_defaults_chat_max_tokens() {
    let config = EngineCoreConfig::default();
    let expected_default = 2048usize.min(config.max_seq_len);
    let processor = RequestProcessor::new(config);

    let mut request = EngineCoreRequest::chat(vec![ChatMessage {
        role: ChatRole::User,
        content: "Hello".to_string(),
    }]);
    request.params.max_tokens = 0;

    let processed = processor.process(request).expect("request should process");
    assert_eq!(processed.params.max_tokens, expected_default);
}

#[test]
fn cuda_chat_automatic_output_budget_adapts_to_backend_context_capacity() {
    for context_capacity in [16_384, 65_536, 262_144] {
        let processor = RequestProcessor::new(EngineCoreConfig {
            backend: BackendKind::Cuda,
            max_seq_len: context_capacity,
            ..EngineCoreConfig::default()
        });
        let mut request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "Hello".to_string(),
        }]);
        request.params.max_tokens = usize::MAX;

        let processed = processor.process(request).expect("CUDA chat request");
        assert_eq!(processed.params.max_tokens, context_capacity);
    }
}

#[test]
fn direct_chat_preparation_rejects_oversized_and_deep_inputs_before_tokenization() {
    let oversized = EngineCoreRequest::chat(vec![ChatMessage {
        role: ChatRole::User,
        content: "x".repeat(DIRECT_CHAT_MIN_INPUT_BYTES + 1),
    }]);
    assert!(matches!(
        oversized.validate_direct_chat_preparation_input(8),
        Err(Error::InvalidInput(message)) if message.contains("preparation input exceeds")
    ));

    let mut nested = serde_json::Value::Null;
    for _ in 0..=DIRECT_CHAT_MAX_TOOL_JSON_DEPTH {
        nested = serde_json::Value::Array(vec![nested]);
    }
    let deep = EngineCoreRequest::chat(vec![ChatMessage {
        role: ChatRole::User,
        content: "tool call".to_string(),
    }])
    .with_chat_config(ChatRequestConfig {
        tools: vec![nested],
        ..ChatRequestConfig::default()
    });
    assert!(matches!(
        deep.validate_direct_chat_preparation_input(4096),
        Err(Error::InvalidInput(message)) if message.contains("nesting limit")
    ));
}

#[test]
fn exact_chat_context_rejects_full_prompts_and_clamps_output_to_remaining_tokens() {
    let mut request = EngineCoreRequest::chat(vec![ChatMessage {
        role: ChatRole::User,
        content: "Hello".to_string(),
    }])
    .with_model_variant(ModelVariant::Qwen306B);
    request.params.max_tokens = 100;
    request
        .install_chat_execution_preparation(ModelVariant::Qwen306B, vec![1, 2, 3], None, 5)
        .unwrap();
    request
        .enforce_chat_context_window(5)
        .expect("two output tokens remain");
    assert_eq!(request.params.max_tokens, 2);

    let mut automatic = EngineCoreRequest::chat(vec![ChatMessage {
        role: ChatRole::User,
        content: "Hello".to_string(),
    }])
    .with_model_variant(ModelVariant::Qwen306B);
    automatic.params.max_tokens = usize::MAX;
    automatic
        .install_chat_execution_preparation(ModelVariant::Qwen306B, vec![1, 2, 3], None, 8_192)
        .unwrap();
    automatic
        .enforce_chat_context_window(8_192)
        .expect("automatic output budget should use the exact remaining context");
    assert_eq!(automatic.params.max_tokens, 8_189);

    let mut full = EngineCoreRequest::chat(vec![ChatMessage {
        role: ChatRole::User,
        content: "Hello".to_string(),
    }])
    .with_model_variant(ModelVariant::Qwen306B);
    full.install_chat_execution_preparation(
        ModelVariant::Qwen306B,
        vec![1, 2, 3, 4, 5],
        None,
        5,
    )
    .unwrap();
    assert!(matches!(
        full.enforce_chat_context_window(5),
        Err(Error::InvalidInput(message)) if message.contains("leaves no output capacity")
    ));
}

#[test]
fn test_request_processor_syncs_but_does_not_authorize_public_prompt_tokens() {
    let config = EngineCoreConfig::default();
    let processor = RequestProcessor::new(config);

    let mut request = EngineCoreRequest::chat(vec![ChatMessage {
        role: ChatRole::User,
        content: "Hello".to_string(),
    }]);
    request.prompt_tokens = vec![41, 42, 43];

    let processed = processor.process(request).expect("request should process");
    assert_eq!(processed.prompt_tokens, vec![41, 42, 43]);
    match &processed.task {
        EngineTask::Chat(input) => assert_eq!(input.prompt_tokens, vec![41, 42, 43]),
        other => panic!("unexpected task payload: {other:?}"),
    }
    assert!(processed
        .validate_chat_execution_preparation()
        .expect_err("public tokens must not create the private marker")
        .to_string()
        .contains("missing exact model prompt preparation"));
}

#[test]
fn exact_chat_preparation_overwrites_forged_token_views() {
    let mut request = EngineCoreRequest::chat(vec![ChatMessage {
        role: ChatRole::User,
        content: "Hello".to_string(),
    }])
    .with_model_variant(ModelVariant::Qwen306B);
    request.prompt_tokens = vec![999];
    request.task = EngineTask::Chat(ChatEngineInput {
        messages: vec![ChatMessage {
            role: ChatRole::Assistant,
            content: "forged".to_string(),
        }],
        chat_config: ChatRequestConfig::default(),
        prompt_tokens: vec![888],
    });

    request
        .install_chat_execution_preparation(
            ModelVariant::Qwen306B,
            vec![41, 42, 43],
            None,
            4096,
        )
        .expect("exact preparation should install");
    request
        .validate_chat_execution_preparation()
        .expect("installed preparation should validate");
    assert_eq!(request.prompt_tokens, vec![41, 42, 43]);
    match &request.task {
        EngineTask::Chat(input) => {
            assert_eq!(input.prompt_tokens, vec![41, 42, 43]);
            assert_eq!(input.messages[0].content, "Hello");
        }
        other => panic!("unexpected task payload: {other:?}"),
    }
}

#[test]
fn exact_chat_preparation_detects_field_and_typed_payload_mutation() {
    let prepared_request = || {
        let mut request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "Hello".to_string(),
        }])
        .with_model_variant(ModelVariant::Qwen306B);
        request
            .install_chat_execution_preparation(
                ModelVariant::Qwen306B,
                vec![41, 42, 43],
                None,
                4096,
            )
            .unwrap();
        request
    };

    let mut field_mutation = prepared_request();
    field_mutation.prompt_tokens[0] = 99;
    assert!(field_mutation
        .validate_chat_execution_preparation()
        .expect_err("field mutation must invalidate preparation")
        .to_string()
        .contains("changed after exact prompt preparation"));

    let mut typed_mutation = prepared_request();
    let EngineTask::Chat(input) = &mut typed_mutation.task else {
        panic!("expected chat task");
    };
    input.prompt_tokens[0] = 99;
    assert!(typed_mutation
        .validate_chat_execution_preparation()
        .expect_err("typed task mutation must invalidate preparation")
        .to_string()
        .contains("typed payload does not match"));
}

#[test]
fn test_request_processor_accepts_audio_bytes_for_asr() {
    let config = EngineCoreConfig::default();
    let processor = RequestProcessor::new(config);

    let request = EngineCoreRequest::asr_bytes(vec![1, 2, 3]);
    let processed = processor.process(request);
    assert!(processed.is_ok());
    let processed = processed.expect("processed");
    assert!(processed.audio_input.is_none());
    assert!(processed.audio_bytes.is_none());
    match processed.task {
        EngineTask::Asr(input) => match input.audio {
            EngineAudioInput::Bytes(bytes) => assert_eq!(bytes, vec![1, 2, 3]),
            other => panic!("unexpected audio input: {other:?}"),
        },
        other => panic!("unexpected task payload: {other:?}"),
    }
}

#[test]
fn prepared_multimodal_shape_is_the_scheduler_prompt_span() {
    let mut request = EngineCoreRequest::asr_bytes(vec![1, 2, 3]);
    request.params.max_tokens = 32;
    request
        .install_prepared_sequence_input_tokens(40, 64)
        .expect("exact ASR shape");

    assert_eq!(request.num_prompt_tokens(), 40);
    assert_eq!(request.params.max_tokens, 24);
    assert!(request
        .install_prepared_sequence_input_tokens(41, 64)
        .is_err());
    assert!(EngineCoreRequest::asr_bytes(vec![1])
        .install_prepared_sequence_input_tokens(64, 64)
        .is_err());
}

#[test]
fn test_request_processor_carries_asr_prompt() {
    let config = EngineCoreConfig::default();
    let processor = RequestProcessor::new(config);

    let request =
        EngineCoreRequest::asr_bytes(vec![1, 2, 3]).with_asr_prompt("spell Izwi correctly");
    let processed = processor.process(request).expect("request should process");

    assert_eq!(
        processed.asr_prompt.as_deref(),
        Some("spell Izwi correctly")
    );
    assert!(processed.num_prompt_tokens() >= 1);
    match processed.task {
        EngineTask::Asr(input) => {
            assert_eq!(input.prompt.as_deref(), Some("spell Izwi correctly"));
        }
        other => panic!("unexpected task payload: {other:?}"),
    }
}

#[test]
fn test_request_processor_defaults_asr_max_tokens() {
    let config = EngineCoreConfig::default();
    let expected_default = 2048usize.min(config.max_seq_len);
    let processor = RequestProcessor::new(config);

    let mut request = EngineCoreRequest::asr("UklGRg==");
    request.params.max_tokens = 0;

    let processed = processor.process(request).expect("request should process");
    assert_eq!(processed.params.max_tokens, expected_default);
}

#[test]
fn test_request_processor_keeps_prepared_parakeet_decode_budget() {
    let config = EngineCoreConfig::default();
    let max_seq_len = config.max_seq_len;
    let budget = max_seq_len * 4;
    let processor = RequestProcessor::new(config);
    let parakeet = |prepared: bool| {
        let mut request = EngineCoreRequest::asr("UklGRg==")
            .with_model_variant(ModelVariant::ParakeetTdt06BV3);
        request.params.max_tokens = budget;
        if prepared {
            request
                .install_prepared_sequence_input_tokens(1, budget + 1)
                .expect("Parakeet budget installs");
        }
        processor
            .process(request)
            .expect("request should process")
            .params
            .max_tokens
    };

    assert_eq!(parakeet(true), budget);
    assert_eq!(parakeet(false), max_seq_len);
}

#[test]
fn test_request_processor_defaults_speech_to_speech_max_tokens() {
    let config = EngineCoreConfig::default();
    let expected_default = 2048usize.min(config.max_seq_len);
    let processor = RequestProcessor::new(config);

    let mut request = EngineCoreRequest::speech_to_speech("UklGRg==");
    request.params.max_tokens = 0;

    let processed = processor.process(request).expect("request should process");
    assert_eq!(processed.params.max_tokens, expected_default);
}

#[test]
fn realtime_operation_duplicate_rejection_preserves_authoritative_payload() {
    let mut request = EngineCoreRequest::asr_bytes(Vec::new());
    request
        .enable_realtime_asr_ingress()
        .expect("ASR session ingress");
    let operation_id = RealtimeOperationId::new(7);
    let _waiter = request
        .install_realtime_asr_operation(
            operation_id,
            RealtimeAsrOperationPayload::Push {
                samples: Arc::from([1.0_f32, 2.0]),
                sample_rate: 16_000,
            },
        )
        .expect("first payload installation");

    assert!(request
        .install_realtime_asr_operation(operation_id, RealtimeAsrOperationPayload::Finish)
        .is_err());
    match request
        .realtime_asr_operation(operation_id)
        .expect("original payload remains authoritative")
    {
        RealtimeAsrOperationPayload::Push {
            samples,
            sample_rate,
        } => {
            assert_eq!(&*samples, &[1.0, 2.0]);
            assert_eq!(sample_rate, 16_000);
        }
        RealtimeAsrOperationPayload::Finish => panic!("duplicate replaced original payload"),
    }
}

#[test]
fn realtime_operation_completion_is_exactly_once_and_releases_payload() {
    let mut request = EngineCoreRequest::asr_bytes(Vec::new());
    request
        .enable_realtime_asr_ingress()
        .expect("ASR session ingress");
    let operation_id = RealtimeOperationId::new(11);
    let samples: Arc<[f32]> = Arc::from([3.0_f32, 4.0, 5.0]);
    let samples_weak = Arc::downgrade(&samples);
    let mut waiter = request
        .install_realtime_asr_operation(
            operation_id,
            RealtimeAsrOperationPayload::Push {
                samples: Arc::clone(&samples),
                sample_rate: 16_000,
            },
        )
        .expect("payload installation");
    drop(samples);

    request
        .complete_realtime_asr_operation(operation_id)
        .expect("first completion");
    assert_eq!(
        waiter.try_recv().expect("completion acknowledgement"),
        Ok(RealtimeAsrOperationAck {
            operation_id,
            kind: RealtimeAsrOperationKind::Push,
            accepted_samples: 3,
        })
    );
    assert!(samples_weak.upgrade().is_none());
    assert!(request
        .complete_realtime_asr_operation(operation_id)
        .is_err());
    assert!(request.realtime_asr_operation(operation_id).is_err());
}

#[test]
fn realtime_operation_ids_cannot_be_reinstalled_or_move_backwards() {
    let mut request = EngineCoreRequest::asr_bytes(Vec::new());
    request
        .enable_realtime_asr_ingress()
        .expect("ASR session ingress");
    let operation_id = RealtimeOperationId::new(13);
    let _waiter = request
        .install_realtime_asr_operation(operation_id, RealtimeAsrOperationPayload::Finish)
        .expect("payload installation");
    request
        .complete_realtime_asr_operation(operation_id)
        .expect("completion");

    assert!(request
        .install_realtime_asr_operation(
            RealtimeOperationId::new(12),
            RealtimeAsrOperationPayload::Finish,
        )
        .is_err());
    assert!(request
        .install_realtime_asr_operation(operation_id, RealtimeAsrOperationPayload::Finish)
        .is_err());
    let _next_waiter = request
        .install_realtime_asr_operation(
            RealtimeOperationId::new(14),
            RealtimeAsrOperationPayload::Finish,
        )
        .expect("strictly newer identity remains valid");
}

#[test]
fn realtime_session_terminal_drains_waiters_payloads_and_fences_ingress() {
    let mut request = EngineCoreRequest::asr_bytes(Vec::new());
    request
        .enable_realtime_asr_ingress()
        .expect("ASR session ingress");
    let samples: Arc<[f32]> = Arc::from([8.0_f32, 9.0]);
    let samples_weak = Arc::downgrade(&samples);
    let mut push_waiter = request
        .install_realtime_asr_operation(
            RealtimeOperationId::new(20),
            RealtimeAsrOperationPayload::Push {
                samples: Arc::clone(&samples),
                sample_rate: 48_000,
            },
        )
        .expect("push installation");
    let mut finish_waiter = request
        .install_realtime_asr_operation(
            RealtimeOperationId::new(21),
            RealtimeAsrOperationPayload::Finish,
        )
        .expect("finish installation");
    drop(samples);

    let terminal = RealtimeAsrTerminalOutcome::Failed(Arc::from("model unloaded"));
    assert_eq!(
        request
            .terminate_realtime_asr_ingress(terminal.clone())
            .expect("terminal drain"),
        2
    );
    assert_eq!(
        push_waiter.try_recv().expect("push terminal"),
        Err(terminal.clone())
    );
    assert_eq!(
        finish_waiter.try_recv().expect("finish terminal"),
        Err(terminal)
    );
    assert!(samples_weak.upgrade().is_none());
    assert_eq!(
        request
            .terminate_realtime_asr_ingress(RealtimeAsrTerminalOutcome::Cancelled)
            .expect("idempotent terminal drain"),
        0
    );
    assert!(request
        .install_realtime_asr_operation(
            RealtimeOperationId::new(22),
            RealtimeAsrOperationPayload::Finish,
        )
        .is_err());
}

/// LFT-2: an LFM TTS request samples audio codes at the reference
/// 0.8 / top-k 64 with a request-derived seed, and honours overrides.
#[test]
fn lfm25_tts_generation_config_uses_reference_audio_sampling() {
    let mut request = EngineCoreRequest::tts("hello");
    let config = request.lfm25_audio_tts_generation_config();
    assert_eq!((config.audio.temperature, config.audio.top_k), (0.8, 64));
    assert_eq!(config.text.temperature, 0.0);
    assert_eq!(
        config.seed,
        EngineCoreRequest::chat_request_seed(&request.id)
    );
    request.params.audio_temperature = Some(0.5);
    request.params.audio_top_k = Some(10);
    let config = request.lfm25_audio_tts_generation_config();
    assert_eq!((config.audio.temperature, config.audio.top_k), (0.5, 10));
    // The speech request fields reach LFM too; a seed overrides the
    // request-derived one.
    let mut request = EngineCoreRequest::tts("hello");
    request.params.speech.temperature = Some(0.3);
    request.params.speech.top_k = Some(8);
    request.params.speech.seed = Some(42);
    let config = request.lfm25_audio_tts_generation_config();
    assert_eq!((config.audio.temperature, config.audio.top_k), (0.3, 8));
    assert_eq!(config.seed, 42);
}

/// TPIPE-24: the shared chat defaults (temperature 0.7, top-p 0.9,
/// repetition penalty 1.1) used to reach Qwen3-TTS sampling. Unset
/// fields now keep the published generation_config, and request fields
/// override it, sub-talker included.
#[test]
fn qwen_tts_unset_sampling_keeps_generation_config_defaults() {
    let request = EngineCoreRequest::tts("hello");
    assert_eq!(request.params.temperature, 0.7, "chat default is present");
    let params = request.qwen_tts_generation_params();
    assert_eq!(
        (params.temperature, params.top_p, params.top_k),
        (0.9, 1.0, 50)
    );
    assert_eq!(params.repetition_penalty, 1.05);
    assert!(params.subtalker.do_sample);
    assert_eq!(
        (
            params.subtalker.temperature,
            params.subtalker.top_k,
            params.subtalker.top_p
        ),
        (0.9, 50, 1.0)
    );
    assert_eq!(params.seed, 0);

    let mut request = EngineCoreRequest::tts("hello");
    request.params.speech.temperature = Some(0.6);
    request.params.speech.top_p = Some(0.8);
    request.params.speech.subtalker_top_k = Some(20);
    request.params.speech.subtalker_do_sample = Some(false);
    request.params.speech.seed = Some(7);
    let params = request.qwen_tts_generation_params();
    assert_eq!((params.temperature, params.top_p), (0.6, 0.8));
    assert_eq!(params.subtalker.top_k, 20);
    assert!(!params.subtalker.do_sample);
    assert_eq!(params.seed, 7);

    // A greedy request stays greedy for every codebook.
    let mut request = EngineCoreRequest::tts("hello");
    request.params.speech.temperature = Some(0.0);
    assert!(!request.qwen_tts_generation_params().subtalker.do_sample);
}

/// Out-of-range speech sampling is an input error before preparation.
#[test]
fn invalid_speech_sampling_is_rejected_before_preparation() {
    for mutate in [
        (|s: &mut super::super::SpeechSamplingOptions| s.top_p = Some(0.0))
            as fn(&mut super::super::SpeechSamplingOptions),
        |s| s.temperature = Some(f32::NAN),
        |s| s.cfg_scale = Some(-1.0),
        |s| s.diffusion_steps = Some(0),
        |s| s.repetition_penalty = Some(0.5),
    ] {
        let mut request = EngineCoreRequest::tts("hello");
        mutate(&mut request.params.speech);
        assert!(matches!(
            request.normalize_tts_params_before_preparation(),
            Err(Error::InvalidInput(_))
        ));
    }
}
