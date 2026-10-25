use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::{AsrChunkTranscription, AsrRouteCapabilities, NativeExecutor, PlannedAsrChunks};
use crate::catalog::ModelFamily;
use crate::engine::request::EngineCoreRequest;
use crate::engine::scheduler::ScheduledRequest;
use crate::engine::{
    ExecutionDisposition, InputRange, ManagedSessionGeneration, SequencePhase,
    SequenceRestartReason, WorkUnit,
};
use crate::model::ModelVariant;
use crate::models::architectures::whisper::asr::WhisperTerminalTransition;
use crate::models::registry::NativeAsrGenerationOptions;

#[test]
fn retained_lfm_asr_prefill_telemetry_distinguishes_b1_and_native_cohorts() {
    assert!(
        super::retained_asr_batch_model_call(crate::engine::NativeBatchMode::Static, 0)
            .is_none()
    );
    assert!(matches!(
        super::retained_asr_batch_model_call(crate::engine::NativeBatchMode::Static, 1),
        Some(crate::engine::metrics::EngineModelCall::ScalarRows {
            envelope: crate::engine::NativeBatchMode::Static,
            rows: 1
        })
    ));
    assert!(matches!(
        super::retained_asr_batch_model_call(crate::engine::NativeBatchMode::Static, 3),
        Some(crate::engine::metrics::EngineModelCall::NativeTensor {
            mode: crate::engine::NativeBatchMode::Static,
            rows: 3
        })
    ));
    assert!(matches!(
        super::retained_asr_batch_model_call(crate::engine::NativeBatchMode::Continuous, 1),
        Some(crate::engine::metrics::EngineModelCall::ScalarRows {
            envelope: crate::engine::NativeBatchMode::Continuous,
            rows: 1
        })
    ));
    assert!(matches!(
        super::retained_asr_batch_model_call(crate::engine::NativeBatchMode::Continuous, 2),
        Some(crate::engine::metrics::EngineModelCall::NativeTensor {
            mode: crate::engine::NativeBatchMode::Continuous,
            rows: 2
        })
    ));
}

#[test]
fn retained_lfm_asr_prefill_mixed_cancellation_selects_only_armed_rows() {
    assert_eq!(
        super::late_cancelled_lfm_asr_prefill_rows(
            &[true, false, true, true],
            &[true, true, false, true]
        ),
        vec![0, 3]
    );
}

#[test]
fn parakeet_mixed_cancellation_selects_only_armed_rows() {
    assert_eq!(
        super::late_cancelled_parakeet_rows(
            &[false, true, true, false, true],
            &[true, true, false, true, true],
        ),
        vec![1, 4]
    );
}

#[test]
fn parakeet_host_rollback_restores_cancelled_then_aborts_remaining_cohort() {
    let mut states = [10_u32, 20, 30];
    let mut checkpoints = [Some(1_u32), Some(2), Some(3)];
    let cancelled = [false, true, false];
    let armed = checkpoints.map(|checkpoint| checkpoint.is_some());

    for row in super::late_cancelled_parakeet_rows(&cancelled, &armed) {
        assert!(super::rollback_parakeet_host_state(
            &mut states[row],
            &mut checkpoints[row],
        ));
    }
    assert_eq!(states, [10, 2, 30]);

    for (state, checkpoint) in states.iter_mut().zip(&mut checkpoints) {
        super::rollback_parakeet_host_state(state, checkpoint);
    }
    assert_eq!(states, [1, 2, 3]);
    assert_eq!(checkpoints, [None, None, None]);
}

#[test]
fn parakeet_committed_host_state_is_not_rolled_back_during_cleanup() {
    let mut state = 9_u32;
    let mut checkpoint = Some(4_u32);
    checkpoint.take();

    assert!(!super::rollback_parakeet_host_state(
        &mut state,
        &mut checkpoint,
    ));
    assert_eq!(state, 9);
}

#[test]
fn atomic_asr_cancellation_discards_output_and_keeps_real_errors() {
    let signal = Arc::new(AtomicBool::new(false));
    let mut request = EngineCoreRequest::asr_bytes(vec![1, 2, 3]);
    request.set_cancellation_signal(signal.clone());

    assert_eq!(
        super::settle_atomic_asr_cancellation(&request, Ok(5_u32)).unwrap(),
        Some(5)
    );
    assert!(matches!(
        super::settle_atomic_asr_cancellation::<u32>(
            &request,
            Err(crate::error::Error::InferenceError("boom".into())),
        ),
        Err(crate::error::Error::InferenceError(_))
    ));
    // The chunk executor's `Cancelled` settles as a cancelled row.
    assert_eq!(
        super::settle_atomic_asr_cancellation::<u32>(
            &request,
            Err(crate::error::Error::Cancelled(request.id.clone())),
        )
        .unwrap(),
        None
    );
    // A single-window call that finished after the signal flipped
    // publishes nothing.
    signal.store(true, Ordering::Release);
    assert_eq!(
        super::settle_atomic_asr_cancellation(&request, Ok(5_u32)).unwrap(),
        None
    );
}

#[test]
fn asr_model_call_observes_cancellation_that_arrives_during_forward() {
    let signal = Arc::new(AtomicBool::new(false));
    let mut request = EngineCoreRequest::asr_bytes(vec![1, 2, 3]);
    request.set_cancellation_signal(signal.clone());
    let physical_call_completed = Cell::new(false);

    let output = super::run_asr_model_call(&request, || {
        physical_call_completed.set(true);
        signal.store(true, Ordering::Release);
        Ok(17_u32)
    })
    .unwrap();

    assert!(physical_call_completed.get());
    assert_eq!(output, None);
    assert!(request.is_cancelled());
}

#[test]
fn whisper_final_prefill_is_a_zero_output_decode_boundary() {
    let step = super::whisper_prefill_boundary_step(11);

    assert_eq!(step.tokens_generated, 11);
    assert!(step.delta.is_empty());
    assert!(step.text.is_empty());
    assert!(!step.finished);
}

#[test]
fn vibevoice_final_prefill_drains_exactly_the_staged_first_sample() {
    let staged = crate::models::registry::NativeAsrDecodeStep {
        delta: "first".into(),
        text: "first token".into(),
        tokens_generated: 1,
        finished: false,
    };

    let step = super::vibevoice_prefill_boundary_step(true, Some(staged.clone()), 0).unwrap();

    assert_eq!(step.delta, staged.delta);
    assert_eq!(step.text, staged.text);
    assert_eq!(step.tokens_generated, 1);
    assert!(!step.finished);
    assert!(super::vibevoice_prefill_boundary_step(true, None, 0).is_err());
}

#[test]
fn vibevoice_nonfinal_prefill_has_no_output_and_rejects_premature_sample() {
    let step = super::vibevoice_prefill_boundary_step(false, None, 7).unwrap();
    assert!(step.delta.is_empty());
    assert!(step.text.is_empty());
    assert_eq!(step.tokens_generated, 7);
    assert!(!step.finished);

    let premature = crate::models::registry::NativeAsrDecodeStep {
        delta: "premature".into(),
        text: "premature".into(),
        tokens_generated: 8,
        finished: false,
    };
    assert!(super::vibevoice_prefill_boundary_step(false, Some(premature), 7).is_err());
}

#[test]
fn vibevoice_late_batch_cancellation_selects_only_still_armed_rows() {
    let rows = [(true, true), (true, false), (false, true), (false, false)];
    let selected = rows
        .into_iter()
        .enumerate()
        .filter_map(|(row, (cancelled, armed))| {
            super::late_cancelled_batch_row(cancelled, armed).then_some(row)
        })
        .collect::<Vec<_>>();

    assert_eq!(selected, [0]);
}

#[test]
fn whisper_terminal_accept_and_skip_publish_only_terminal_policy_output() {
    let mut accepted = crate::models::registry::NativeAsrDecodeStep {
        delta: "provisional".into(),
        text: "provisional".into(),
        tokens_generated: 4,
        finished: true,
    };
    let accept = super::apply_whisper_terminal_transition(
        &mut accepted,
        WhisperTerminalTransition::Accept {
            text: "accepted transcript".into(),
            selected_temperature: 0.0,
        },
        ManagedSessionGeneration::INITIAL,
    )
    .unwrap();
    assert_eq!(accept, super::WhisperTerminalAction::Publish);
    assert_eq!(accepted.delta, "accepted transcript");
    assert_eq!(accepted.text, "accepted transcript");

    let mut skipped = accepted;
    let skip = super::apply_whisper_terminal_transition(
        &mut skipped,
        WhisperTerminalTransition::SkipNoSpeech {
            no_speech_probability: Some(0.99),
        },
        ManagedSessionGeneration::INITIAL,
    )
    .unwrap();
    assert_eq!(skip, super::WhisperTerminalAction::Publish);
    assert!(skipped.delta.is_empty());
    assert!(skipped.text.is_empty());
}

#[test]
fn whisper_retry_is_authenticated_and_restart_result_publishes_nothing() {
    let mut step = crate::models::registry::NativeAsrDecodeStep {
        delta: "rejected attempt".into(),
        text: "rejected attempt".into(),
        tokens_generated: 8,
        finished: true,
    };
    let action = super::apply_whisper_terminal_transition(
        &mut step,
        WhisperTerminalTransition::RetryRequired {
            next_temperature: 0.2,
            reasons: vec!["compression_ratio"],
            expected_generation: 1,
            new_generation: 2,
        },
        ManagedSessionGeneration::INITIAL,
    )
    .unwrap();
    assert_eq!(action, super::WhisperTerminalAction::Restart);
    assert!(step.delta.is_empty());
    assert!(step.text.is_empty());

    let result = super::ModelSessionResult::restart_sequence(
        "whisper-retry".into(),
        SequenceRestartReason::ModelFallback,
    );
    assert_eq!(
        result.disposition,
        ExecutionDisposition::RestartSequence(SequenceRestartReason::ModelFallback)
    );
    assert_eq!(result.output.tokens_processed, 0);
    assert_eq!(result.output.tokens_generated, 0);
    assert!(result.output.audio.is_none());
    assert!(result.output.text.is_none());
    assert!(result.staged_stream_outputs.is_empty());
    assert!(result.managed_cache_completions.is_empty());

    let stale = super::apply_whisper_terminal_transition(
        &mut step,
        WhisperTerminalTransition::RetryRequired {
            next_temperature: 0.4,
            reasons: vec!["log_probability"],
            expected_generation: 1,
            new_generation: 2,
        },
        ManagedSessionGeneration::INITIAL.next().unwrap(),
    );
    assert!(stale.is_err());
}

#[test]
fn whisper_generation_two_restarts_only_at_context_zero_prefill() {
    let scheduled = |is_prefill, num_computed_tokens| ScheduledRequest {
        plan_id: 1,
        request_id: "whisper-generation".into(),
        sequence_id: 1,
        num_tokens: 4,
        is_prefill,
        num_computed_tokens,
        work: WorkUnit::SequenceStep {
            phase: if is_prefill {
                SequencePhase::Prefill
            } else {
                SequencePhase::Decode
            },
            input: InputRange {
                start: num_computed_tokens,
                end: num_computed_tokens + 4,
            },
            max_output_steps: 4,
            auxiliary_state: None,
        },
    };
    let generation_two = ManagedSessionGeneration::INITIAL.next().unwrap();

    assert!(super::begins_whisper_managed_generation(
        &scheduled(true, 0),
        generation_two,
    ));
    assert!(!super::begins_whisper_managed_generation(
        &scheduled(true, 1),
        generation_two,
    ));
    assert!(!super::begins_whisper_managed_generation(
        &scheduled(false, 0),
        generation_two,
    ));
    assert!(!super::begins_whisper_managed_generation(
        &scheduled(true, 0),
        ManagedSessionGeneration::INITIAL,
    ));
}

#[test]
fn whisper_cancellation_after_terminal_policy_prevents_publication() {
    let signal = Arc::new(AtomicBool::new(false));
    let mut request = EngineCoreRequest::asr_bytes(vec![1, 2, 3]);
    request.set_cancellation_signal(signal.clone());
    let mut step = crate::models::registry::NativeAsrDecodeStep {
        delta: "provisional".into(),
        text: "provisional".into(),
        tokens_generated: 3,
        finished: true,
    };

    let action = super::resolve_whisper_terminal_action(
        &request,
        &mut step,
        ManagedSessionGeneration::INITIAL,
        || {
            signal.store(true, Ordering::Release);
            Ok(WhisperTerminalTransition::Accept {
                text: "must not publish".into(),
                selected_temperature: 0.0,
            })
        },
    )
    .unwrap();

    assert_eq!(action, None);
    assert!(request.is_cancelled());
    assert_eq!(step.delta, "provisional");
    assert_eq!(step.text, "provisional");
}

#[test]
fn qwen_asr_execution_reuses_prepared_audio_without_decoding_again() {
    let variant = ModelVariant::Qwen3Asr06BGguf;
    let mut request = EngineCoreRequest::asr_bytes(vec![1, 2, 3]).with_model_variant(variant);
    request
        .install_prepared_asr_audio(variant, vec![0.25, -0.5, 0.75], 16_000)
        .unwrap();
    let decode_calls = Cell::new(0_u32);

    let (audio, sample_rate, decode_ms) =
        super::resolve_asr_execution_audio(&request, ModelFamily::Qwen3Asr, || {
            decode_calls.set(decode_calls.get() + 1);
            Ok((vec![1.0], 8_000))
        })
        .unwrap();

    assert_eq!(decode_calls.get(), 0);
    assert_eq!(audio.samples(), &[0.25, -0.5, 0.75]);
    assert_eq!(sample_rate, 16_000);
    assert_eq!(decode_ms, 0.0);
}

#[test]
fn lfm_long_form_asr_routes_to_its_own_executor_not_the_generic_path() {
    let lfm = ModelVariant::Lfm25Audio15BGguf;
    let mut long_form = EngineCoreRequest::asr_bytes(vec![1, 2, 3]).with_model_variant(lfm);
    long_form
        .install_prepared_asr_long_form_atomic(
            crate::engine::PlannedAsrChunks::unexecuted_for_tests(),
        )
        .unwrap();
    assert_eq!(
        super::atomic_asr_route(&long_form, lfm).unwrap(),
        super::AtomicAsrRoute::Lfm25AudioLongForm
    );

    // Native ASR families keep the generic path, long-form or not.
    for variant in [
        ModelVariant::Qwen3Asr06BGguf,
        ModelVariant::VibeVoiceAsr,
        ModelVariant::ParakeetTdt06BV3,
    ] {
        let mut request =
            EngineCoreRequest::asr_bytes(vec![1, 2, 3]).with_model_variant(variant);
        assert_eq!(
            super::atomic_asr_route(&request, variant).unwrap(),
            super::AtomicAsrRoute::Generic
        );
        request
            .install_prepared_asr_long_form_atomic(
                crate::engine::PlannedAsrChunks::unexecuted_for_tests(),
            )
            .unwrap();
        assert_eq!(
            super::atomic_asr_route(&request, variant).unwrap(),
            super::AtomicAsrRoute::Generic
        );
    }
}

/// The generic path executes the plan preparation stored, as stored: a
/// fake chunk transcriber sees exactly the stored chunk bounds, which no
/// planner would produce for this audio.
#[test]
fn executor_runs_the_stored_route_plan_without_replanning() {
    let variant = ModelVariant::Qwen3Asr06BGguf;
    let sample_rate = 16_000_u32;
    // Each sample holds its own index, so a chunk's first sample names
    // its start.
    let samples = (0..60 * sample_rate as usize)
        .map(|index| index as f32)
        .collect::<Vec<_>>();
    let stored_bounds = [(0, 100_000), (90_000, 500_000), (480_000, samples.len())];
    let mut stored = PlannedAsrChunks::unexecuted_for_tests();
    stored.input_samples = samples.len();
    stored.sample_rate = sample_rate;
    stored.chunks = stored_bounds
        .iter()
        .map(|&(start_sample, end_sample)| izwi_asr_toolkit::AudioChunk {
            start_sample,
            end_sample,
        })
        .collect();
    let mut request = EngineCoreRequest::asr_bytes(vec![1, 2, 3]).with_model_variant(variant);
    request
        .install_prepared_asr_long_form_atomic(stored)
        .unwrap();

    let plan = super::asr_execution_route_plan(&request, &samples, sample_rate, || {
        panic!("a stored plan must not be re-planned")
    })
    .unwrap();
    let mut seen = Vec::new();
    let mut sequence = 0;
    NativeExecutor::transcribe_with_chunk_plan_with_streaming_details_and_options(
        &request.id,
        &|| false,
        None,
        request.stream_policy,
        &mut sequence,
        &samples,
        sample_rate,
        &plan.chunks,
        &plan.config,
        Default::default(),
        |chunk_audio, _sr, _prefix_text, _partial| {
            let start = chunk_audio[0] as usize;
            seen.push((start, start + chunk_audio.len()));
            Ok(AsrChunkTranscription {
                text: format!("chunk{}", seen.len()),
                diagnostics: None,
            })
        },
    )
    .unwrap();
    assert_eq!(seen, stored_bounds);

    // A plan made for other audio, or a long-form row without one, fails.
    assert!(super::asr_execution_route_plan(
        &request,
        &samples[1..],
        sample_rate,
        || unreachable!()
    )
    .is_err());
    let mut lost = EngineCoreRequest::asr_bytes(vec![1, 2, 3]).with_model_variant(variant);
    lost.install_prepared_asr_long_form_atomic(PlannedAsrChunks::unexecuted_for_tests())
        .unwrap();
    lost.prepared_asr_route_plan = None;
    assert!(
        super::asr_execution_route_plan(&lost, &samples, sample_rate, || unreachable!())
            .is_err()
    );

    // A family without a preparation stage is planned here, by the rule.
    let nemotron = EngineCoreRequest::asr_bytes(vec![1, 2, 3])
        .with_model_variant(ModelVariant::Nemotron35AsrStreaming06B);
    let planned = super::asr_execution_route_plan(&nemotron, &samples, sample_rate, || {
        Ok(AsrRouteCapabilities::for_family(
            ModelFamily::NemotronAsr,
            Some(30.0),
        ))
    })
    .unwrap();
    assert_eq!(planned.chunks.len(), 3);
}

#[test]
fn streaming_qwen_asr_runs_the_same_route_as_non_streaming() {
    // 60 s of Qwen3 audio once fitted the streaming plan's single pass
    // while preparation chunked it; both modes now share one plan.
    let sample_rate = 16_000;
    let samples = vec![0.0_f32; 60 * sample_rate as usize];
    let capabilities = AsrRouteCapabilities::for_family(ModelFamily::Qwen3Asr, Some(120.0));
    let offline = NativeExecutor::asr_route_plan(&samples, sample_rate, &capabilities, false);
    let streaming = NativeExecutor::asr_route_plan(&samples, sample_rate, &capabilities, true);
    assert!(offline.requires_chunk_path());
    assert_eq!(offline.chunks, streaming.chunks);

    assert_eq!(
        super::physical_asr_route(ModelFamily::Qwen3Asr).unwrap(),
        super::PhysicalAsrRoute::Qwen3
    );
    // Every family loaded as a `NativeAsrModel` has a physical route, so
    // no ASR request can reach a workspace-less model call.
    for variant in ModelVariant::all()
        .iter()
        .filter(|variant| variant.is_asr())
    {
        assert!(
            super::physical_asr_route(variant.family()).is_ok(),
            "{variant}"
        );
    }
    for family in [
        ModelFamily::Voxtral,
        ModelFamily::Lfm25Audio,
        ModelFamily::Qwen3ForcedAligner,
        ModelFamily::Qwen3Chat,
    ] {
        assert!(
            matches!(
                super::physical_asr_route(family),
                Err(crate::error::Error::InvalidInput(_))
            ),
            "{family:?}"
        );
    }
}

#[test]
fn audio_chat_asr_without_a_long_form_route_fails_closed_before_registry_lookup() {
    // An empty registry makes any `NativeAsrModel` lookup report the model
    // as not loaded, which is the failure this guard replaces.
    let executor = NativeExecutor::new(crate::engine::WorkerConfig {
        model_registry: Some(Arc::new(crate::models::ModelRegistry::new(
            std::path::PathBuf::new(),
            crate::backends::DeviceProfile::cpu(),
        ))),
        ..Default::default()
    });
    let variant = ModelVariant::Lfm25Audio15BGguf;
    for streaming in [true, false] {
        let mut request =
            EngineCoreRequest::asr_bytes(vec![1, 2, 3]).with_model_variant(variant);
        request.streaming = streaming;
        let scheduled = ScheduledRequest {
            plan_id: 1,
            request_id: request.id.clone(),
            sequence_id: 1,
            num_tokens: 1,
            is_prefill: true,
            num_computed_tokens: 0,
            work: WorkUnit::AtomicJob { kind: "asr".into() },
        };

        let error = executor
            .transcribe_request_with_managed_cache(&request, &scheduled, None)
            .expect_err("audio-chat ASR without a long-form route must fail closed");

        assert!(
            matches!(&error, crate::error::Error::InvalidInput(message)
                if message.contains("has no retained or long-form route")),
            "streaming={streaming}: {error}"
        );
    }
}

/// Mirrors the coordinator's failed-invocation fence test for the
/// executor's own lease helpers: every lease of a failed operation is
/// fenced before the helper returns, and the operation's error wins.
#[test]
fn failed_invocation_operation_fences_every_lease_before_returning() {
    use crate::engine::{
        AdapterAbiRevision, AdapterInstanceId, ExecutionAdapterBinding, ExecutionGroupId,
        ExecutionMode, ExecutionProfile, ModelInstanceId, NativeBatchMode, StageDescriptor,
        StageId, StageWorkSelector,
    };
    use crate::error::{Error, Result};

    let variant = ModelVariant::VibeVoiceAsr;
    let profile = ExecutionProfile::fail_closed(
        crate::backends::BackendKind::Cpu,
        Some(variant),
        ExecutionMode::Atomic,
    );
    let mut stage = StageDescriptor::from_execution_profile(
        StageId::new(0),
        "asr.atomic",
        &profile,
        NativeBatchMode::None,
    );
    stage.selector = StageWorkSelector::Atomic;
    let execution = ExecutionAdapterBinding {
        execution_group_id: ExecutionGroupId::new(1),
        model_instance_id: ModelInstanceId::new(9),
        adapter_instance_id: AdapterInstanceId::new(2),
        adapter_abi_revision: AdapterAbiRevision::new(1),
        model_variant: variant,
        capability_id: "asr".to_string(),
        stages: Arc::from([stage]),
    };
    let scheduled = |request: &EngineCoreRequest| ScheduledRequest {
        plan_id: 1,
        request_id: request.id.clone(),
        sequence_id: 1,
        num_tokens: 1,
        is_prefill: true,
        num_computed_tokens: 0,
        work: WorkUnit::AtomicJob { kind: "asr".into() },
    };

    for (fail_operation, fail_fence) in [(false, false), (true, false), (true, true)] {
        // VibeVoice's helper expects exactly three state domains.
        let (state, probe) = crate::kv::v2::fence_probe::fence_probe_state_runtime(&execution, 3);
        probe.fail_fence.store(fail_fence, Ordering::Release);
        let mut request = EngineCoreRequest::asr_bytes(vec![1, 2, 3]).with_model_variant(variant);
        request.bind_execution_adapter(execution.clone()).unwrap();
        request
            .bind_v2_state_runtime(
                state.clone(),
                state.state_fingerprint,
                crate::backends::BackendKind::Cpu,
            )
            .unwrap();

        let result = super::vibevoice::with_vibevoice_invocation_state(
            &request,
            &scheduled(&request),
            |_leases| -> Result<()> {
                if fail_operation {
                    return Err(Error::InvalidInput("enqueued work then failed".into()));
                }
                Ok(())
            },
        );

        if fail_operation {
            let error = result.expect_err("the operation error must propagate");
            assert!(
                matches!(&error, Error::InvalidInput(message)
                    if message == "enqueued work then failed"),
                "the original operation error must win over the fence: {error:?}"
            );
            assert_eq!(probe.fences.load(Ordering::Acquire), 3);
            assert_eq!(probe.aborts.load(Ordering::Acquire), 3);
        } else {
            result.expect("a successful operation releases its leases");
            assert_eq!(probe.fences.load(Ordering::Acquire), 0);
            assert_eq!(probe.aborts.load(Ordering::Acquire), 0);
        }
    }
}

#[test]
fn lfm_long_form_asr_passes_only_a_caller_set_max_tokens() {
    let mut request = EngineCoreRequest::asr_bytes(vec![1, 2, 3])
        .with_model_variant(ModelVariant::Lfm25Audio15BGguf);
    request.asr_auto_max_tokens = true;
    assert_eq!(super::lfm25_audio_asr_long_form_max_tokens(&request), None);

    request.asr_auto_max_tokens = false;
    request.params.max_tokens = 96;
    assert_eq!(
        super::lfm25_audio_asr_long_form_max_tokens(&request),
        Some(96)
    );
}

#[test]
fn vibevoice_legacy_execution_reuses_prepared_audio_and_fails_closed_without_it() {
    let variant = ModelVariant::VibeVoiceAsr;
    let mut request = EngineCoreRequest::asr_bytes(vec![1, 2, 3]).with_model_variant(variant);
    request
        .install_prepared_asr_audio(variant, vec![0.25, -0.5, 0.75], 24_000)
        .unwrap();
    request
        .install_prepared_asr_long_form_atomic(
            crate::engine::PlannedAsrChunks::unexecuted_for_tests(),
        )
        .unwrap();
    let decode_calls = Cell::new(0_u32);

    let (audio, sample_rate, decode_ms) =
        super::resolve_asr_execution_audio(&request, ModelFamily::VibeVoiceAsr, || {
            decode_calls.set(decode_calls.get() + 1);
            Ok((vec![1.0], 8_000))
        })
        .unwrap();

    assert_eq!(decode_calls.get(), 0);
    assert_eq!(audio.samples(), &[0.25, -0.5, 0.75]);
    assert_eq!(sample_rate, 24_000);
    assert_eq!(decode_ms, 0.0);

    let missing = EngineCoreRequest::asr_bytes(vec![1, 2, 3]).with_model_variant(variant);
    let error = super::resolve_asr_execution_audio(&missing, ModelFamily::VibeVoiceAsr, || {
        panic!("VibeVoice promoted execution must not decode a second audio buffer")
    })
    .err()
    .expect("missing promoted VibeVoice audio must fail closed");
    assert!(error
        .to_string()
        .contains("lost its prepared decoded-audio"));
}

#[test]
fn audio_decode_timing_preserves_whisper_model_diagnostics() {
    let diagnostics = serde_json::json!({
        "model_family": "whisper_asr",
        "timings_ms": {
            "model_total": 12.5
        }
    });

    let updated = NativeExecutor::with_audio_decode_timing(Some(diagnostics), 3.25)
        .expect("diagnostics payload");

    assert_eq!(updated["model_family"], "whisper_asr");
    assert_eq!(updated["timings_ms"]["model_total"], 12.5);
    assert_eq!(updated["timings_ms"]["audio_decode"], 3.25);
}

#[test]
fn audio_decode_timing_creates_diagnostics_when_missing() {
    let updated =
        NativeExecutor::with_audio_decode_timing(None, 4.0).expect("diagnostics payload");

    assert_eq!(updated["timings_ms"]["audio_decode"], 4.0);
}

#[test]
fn audio_decode_timing_wraps_non_object_diagnostics() {
    let updated = NativeExecutor::with_audio_decode_timing(Some(serde_json::json!("old")), 5.0)
        .expect("diagnostics payload");

    assert_eq!(updated["model_diagnostics"], "old");
    assert_eq!(updated["timings_ms"]["audio_decode"], 5.0);
}

#[test]
fn granite_auto_chunk_generation_options_use_chunk_duration() {
    let mut request = EngineCoreRequest::asr("UklGRg==");
    request.asr_auto_max_tokens = true;
    request.params.max_tokens = 2048;
    let base = NativeAsrGenerationOptions {
        max_new_tokens: 2048,
        stop_token_ids: Vec::new(),
        stop_sequences: Vec::new(),
        repetition_penalty: None,
    };

    let options = NativeExecutor::asr_chunk_generation_options(
        &request,
        ModelFamily::GraniteSpeechAsr,
        16_000 * 30,
        16_000,
        &base,
        None,
    );

    assert_eq!(options.max_new_tokens, 200);
}

#[test]
fn granite_auto_single_segment_options_use_decoded_audio_duration() {
    let mut request = EngineCoreRequest::asr("UklGRg==");
    request.asr_auto_max_tokens = true;
    request.params.max_tokens = 2048;
    let base = NativeAsrGenerationOptions {
        max_new_tokens: 2048,
        stop_token_ids: Vec::new(),
        stop_sequences: Vec::new(),
        repetition_penalty: None,
    };

    let options = NativeExecutor::asr_chunk_generation_options(
        &request,
        ModelFamily::GraniteSpeechAsr,
        16_000 * 30,
        16_000,
        &base,
        None,
    );

    assert_eq!(options.max_new_tokens, 200);
    assert!(options.max_new_tokens < request.params.max_tokens);
}

#[test]
fn granite_explicit_chunk_generation_options_preserve_user_budget() {
    let mut request = EngineCoreRequest::asr("UklGRg==");
    request.asr_auto_max_tokens = false;
    request.params.max_tokens = 2048;
    let base = NativeAsrGenerationOptions {
        max_new_tokens: 2048,
        stop_token_ids: Vec::new(),
        stop_sequences: Vec::new(),
        repetition_penalty: None,
    };

    let options = NativeExecutor::asr_chunk_generation_options(
        &request,
        ModelFamily::GraniteSpeechAsr,
        16_000 * 30,
        16_000,
        &base,
        None,
    );

    assert_eq!(options.max_new_tokens, 2048);
}

#[test]
fn granite_auto_chunk_budget_is_clamped_to_the_context_room() {
    let mut request = EngineCoreRequest::asr("UklGRg==");
    request.asr_auto_max_tokens = true;
    request.params.max_tokens = 2048;
    let base = NativeAsrGenerationOptions {
        max_new_tokens: 2048,
        stop_token_ids: Vec::new(),
        stop_sequences: Vec::new(),
        repetition_penalty: None,
    };
    let options = |context: Option<(usize, f32)>| {
        NativeExecutor::asr_chunk_generation_options(
            &request,
            ModelFamily::GraniteSpeechAsr,
            16_000 * 60,
            16_000,
            &base,
            context,
        )
        .max_new_tokens
    };

    assert_eq!(options(None), 400);
    assert_eq!(options(Some((4_096, 10.0))), 400);
    // 600 - 128 text reserve - 600 audio tokens leaves nothing: one token.
    assert_eq!(options(Some((600, 10.0))), 1);
    // 1000 - 128 - 600 leaves 272 decode tokens.
    assert_eq!(options(Some((1_000, 10.0))), 272);
}

/// Plus's native timestamp task decodes `word [T:N]` for every word, so
/// its chunks get the timestamp budget; base keeps the ASR budget.
#[test]
fn granite_timestamp_requests_budget_tagged_output_on_plus_only() {
    let base = NativeAsrGenerationOptions {
        max_new_tokens: 2048,
        stop_token_ids: Vec::new(),
        stop_sequences: Vec::new(),
        repetition_penalty: None,
    };
    let budget = |variant| {
        let mut request = EngineCoreRequest::asr("UklGRg==")
            .with_model_variant(variant)
            .with_asr_native_word_timestamps(true);
        request.asr_auto_max_tokens = true;
        request.params.max_tokens = 2048;
        NativeExecutor::asr_chunk_generation_options(
            &request,
            ModelFamily::GraniteSpeechAsr,
            16_000 * 30,
            16_000,
            &base,
            Some((4_096, 10.0)),
        )
        .max_new_tokens
    };

    assert_eq!(budget(ModelVariant::GraniteSpeech412BPlus), 844);
    assert_eq!(budget(ModelVariant::GraniteSpeech412B), 200);
}

#[test]
fn vibevoice_auto_options_use_the_duration_budget_under_the_ceiling() {
    let mut request = EngineCoreRequest::asr("UklGRg==");
    request.asr_auto_max_tokens = true;
    request.params.max_tokens = 32_768;
    let base = NativeExecutor::asr_generation_options(&request);
    let options = |samples: usize, context: Option<(usize, f32)>| {
        NativeExecutor::asr_chunk_generation_options(
            &request,
            ModelFamily::VibeVoiceAsr,
            samples,
            16_000,
            &base,
            context,
        )
        .max_new_tokens
    };

    assert_eq!(options(16_000 * 60, None), 976);
    assert_eq!(options(16_000 * 600, None), 7_456);
    assert_eq!(options(16_000 * 600, Some((131_072, 7.5))), 7_456);
    // 8192 - 128 text reserve - 4500 audio tokens leaves 3564.
    assert_eq!(options(16_000 * 600, Some((8_192, 7.5))), 3_564);
    // The executor solver plans long audio with the same budget.
    assert_eq!(
        super::asr_planned_decode_tokens(&request, ModelFamily::VibeVoiceAsr, 600.0),
        7_456
    );
}

#[test]
fn vibevoice_explicit_options_preserve_the_caller_budget() {
    let mut request = EngineCoreRequest::asr("UklGRg==");
    request.asr_auto_max_tokens = false;
    request.params.max_tokens = 200;
    let base = NativeExecutor::asr_generation_options(&request);

    let options = NativeExecutor::asr_chunk_generation_options(
        &request,
        ModelFamily::VibeVoiceAsr,
        16_000 * 600,
        16_000,
        &base,
        None,
    );

    assert_eq!(options.max_new_tokens, 200);
    assert_eq!(
        super::asr_planned_decode_tokens(&request, ModelFamily::VibeVoiceAsr, 600.0),
        200
    );
}

#[test]
fn granite_invocation_solver_plans_with_the_granite_budget() {
    let mut request = EngineCoreRequest::asr("UklGRg==");
    request.asr_auto_max_tokens = true;
    request.params.max_tokens = 2048;
    assert_eq!(
        super::asr_planned_decode_tokens(&request, ModelFamily::GraniteSpeechAsr, 300.0),
        2_000
    );
    assert_eq!(
        super::asr_planned_decode_tokens(&request, ModelFamily::Qwen3Asr, 300.0),
        super::MAX_ASR_NEW_TOKENS
    );

    // A single streaming segment must leave room for the budget it will
    // be decoded with: 300 s would need 128 + 3,000 + 2,000 tokens.
    let limit = crate::runtime::compose_invocation_audio_limit(
        Some(540.0),
        Some(10.0),
        Some(4_096),
        |secs| super::asr_planned_decode_tokens(&request, ModelFamily::GraniteSpeechAsr, secs),
    )
    .unwrap()
    .unwrap();
    assert!((237.0..=239.0).contains(&limit), "unexpected limit {limit}");
}

#[test]
fn granite_truncated_chunk_retries_once_at_double_budget_within_context() {
    let max_tokens = serde_json::json!({ "decode": { "stop_reason": "max_tokens" } });
    let stopped = serde_json::json!({ "decode": { "stop_reason": "stop_token" } });
    assert!(super::granite_chunk_truncated(true, Some(&max_tokens)));
    assert!(!super::granite_chunk_truncated(false, Some(&max_tokens)));
    assert!(!super::granite_chunk_truncated(true, Some(&stopped)));
    assert!(!super::granite_chunk_truncated(true, None));

    assert_eq!(
        super::granite_truncation_retry_max_tokens(200, 2048, Some(3_668)),
        Some(400)
    );
    assert_eq!(
        super::granite_truncation_retry_max_tokens(200, 2048, None),
        Some(400)
    );
    assert_eq!(
        super::granite_truncation_retry_max_tokens(200, 2048, Some(250)),
        Some(250)
    );
    assert_eq!(
        super::granite_truncation_retry_max_tokens(200, 300, None),
        Some(300)
    );
    assert_eq!(
        super::granite_truncation_retry_max_tokens(200, 2048, Some(200)),
        None
    );
    assert_eq!(
        super::granite_truncation_retry_max_tokens(2048, 2048, None),
        None
    );
}

#[test]
fn granite_truncation_retry_diagnostics_record_the_decision() {
    let diagnostics = super::with_granite_truncation_retry_diagnostics(
        Some(serde_json::json!({ "decode": { "stop_reason": "stop_token" } })),
        200,
        Some(400),
        Some("stop_token"),
        true,
        "retry_selected",
    )
    .unwrap();

    let retry = &diagnostics["chunk_truncation_retry"];
    assert_eq!(retry["decision"], "retry_selected");
    assert_eq!(retry["selected_retry"], true);
    assert_eq!(retry["original_max_new_tokens"], 200);
    assert_eq!(retry["retry_max_new_tokens"], 400);
    assert_eq!(retry["retry_stop_reason"], "stop_token");
    assert_eq!(diagnostics["decode"]["stop_reason"], "stop_token");
}

#[test]
fn granite_prefix_replay_text_is_disabled_by_default() {
    let prefix = (0..140)
        .map(|idx| format!("word{idx}"))
        .collect::<Vec<_>>()
        .join(" ");

    let replay = NativeExecutor::granite_asr_prefix_replay_text(&prefix);

    assert!(replay.is_empty());
}

#[test]
fn recent_word_suffix_keeps_recent_words() {
    let prefix = (0..140)
        .map(|idx| format!("word{idx}"))
        .collect::<Vec<_>>()
        .join(" ");

    let replay = super::recent_word_suffix(&prefix, 96);
    let words = replay.split_whitespace().collect::<Vec<_>>();

    assert_eq!(words.len(), 96);
    assert_eq!(words.first(), Some(&"word44"));
    assert_eq!(words.last(), Some(&"word139"));
}

#[test]
fn repeated_phrase_loop_detects_consecutive_phrase_repeats() {
    let text = "intro words if you speak in trust their nervous system contracts \
            if you speak in trust their nervous system contracts \
            if you speak in trust their nervous system contracts tail";

    let loop_signal = super::repeated_phrase_loop(text).expect("loop signal");

    assert!(loop_signal.phrase.contains("if you speak"));
    assert!(loop_signal.repeat_count >= 3);
}

#[test]
fn trim_repeated_phrase_tail_keeps_first_occurrence() {
    let text = "better relationships with your friends and your family and your coworkers \
            and your boss and your boss and your boss and your boss";
    let loop_signal = super::repeated_phrase_loop(text).expect("loop signal");

    let trimmed =
        super::trim_repeated_phrase_tail(text, &loop_signal).expect("trimmed repeated tail");

    assert_eq!(
        trimmed,
        "better relationships with your friends and your family and your coworkers and your boss"
    );
}

#[test]
fn trim_repeated_phrase_tail_ignores_middle_repeats() {
    let text = "alpha beta gamma alpha beta gamma alpha beta gamma then \
            a normal ending with additional words that should remain";
    let loop_signal = super::repeated_phrase_loop(text).expect("loop signal");

    assert!(super::trim_repeated_phrase_tail(text, &loop_signal).is_none());
}

#[test]
fn granite_chunk_loop_signal_requires_max_token_stop() {
    let text = "alpha beta gamma alpha beta gamma alpha beta gamma";
    let stopped = serde_json::json!({
        "decode": {
            "stop_reason": "stop_token"
        }
    });
    let max_tokens = serde_json::json!({
        "decode": {
            "stop_reason": "max_tokens"
        }
    });

    assert!(super::granite_chunk_loop_signal(text, Some(&stopped)).is_none());
    assert!(super::granite_chunk_loop_signal(text, Some(&max_tokens)).is_some());
}

#[test]
fn granite_loop_recovery_budget_is_shorter_than_original() {
    let retry = super::granite_loop_recovery_max_tokens(16_000 * 30, 16_000, 84);

    assert!(retry < 84);
    assert!(retry >= 24);
}
