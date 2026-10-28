#[test]
fn streamed_completion_survives_empty_audio_merging() {
    let audio = super::EngineCore::merge_audio_output(
        Some(super::AudioOutput::empty(24000)),
        Some(super::AudioOutput::streamed(48000, 24000)),
    )
    .unwrap();
    assert!(audio.samples.is_empty());
    assert_eq!(audio.streamed_samples, Some(48000));
    assert_eq!(audio.duration_secs, 2.0);
    let unchanged = super::EngineCore::merge_audio_output(
        Some(audio),
        Some(super::AudioOutput::empty(24000)),
    )
    .unwrap();
    assert_eq!(unchanged.streamed_samples, Some(48000));
    assert_eq!(unchanged.duration_secs, 2.0);
}

use super::super::executor::{
    ExecutorOutput, ExecutorPhaseTiming, ExecutorStepResult, ModelExecutor, ModelSessionResult,
};
use super::super::scheduler::ScheduledRequest;
use super::super::types::{AudioOutput, TaskType};
use super::*;
use crate::engine::{ClockedStateSpan, InputRange, SessionKey};
use crate::kv::v2::{StateClock, StateGroupId};
use crate::model::ModelVariant;
use crate::models::shared::chat::{ChatMessage, ChatRole};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[test]
fn capacity_replay_cleanup_preserves_stream_delivery_cursor() {
    let mut core = EngineCore::new(EngineCoreConfig::default()).unwrap();
    let session = super::super::SessionKey::new("stream-replay".into(), 1);
    core.stream_sequence_cursors.insert(session.clone(), 7);
    core.incremental_stream_sessions.insert(session.clone());
    core.execution_retry_attempts.insert(session.clone(), 2);
    core.clear_execution_state_for_capacity_replay(&session);
    assert_eq!(core.stream_sequence_cursors.get(&session), Some(&7));
    assert!(core.incremental_stream_sessions.contains(&session));
    assert!(!core.execution_retry_attempts.contains_key(&session));
    core.clear_exact_execution_state(&session);
    assert!(!core.stream_sequence_cursors.contains_key(&session));
    assert!(!core.incremental_stream_sessions.contains(&session));
}

#[test]
fn cuda_pool_ceiling_separates_sequence_context_from_aggregate_capacity() {
    assert_eq!(
        EngineCore::cuda_pool_upper_bound(4096, 16, true).unwrap(),
        65536
    );
    assert_eq!(
        EngineCore::cuda_pool_upper_bound(4096, 16, false).unwrap(),
        4096
    );
    assert!(EngineCore::cuda_pool_upper_bound(u64::MAX, 16, true).is_err());
}

#[test]
fn single_candidate_admission_alternates_under_sustained_decode_and_prefill() {
    let mut preference = SequencePhase::Decode;
    let mut admitted = Vec::new();

    for _ in 0..6 {
        let (decode, prefill) = EngineCore::candidate_phase_quotas(1, 32, 32, preference);
        let phase = match (decode, prefill) {
            (1, 0) => SequencePhase::Decode,
            (0, 1) => SequencePhase::Prefill,
            other => panic!("single candidate produced invalid phase quotas {other:?}"),
        };
        admitted.push(phase);
        preference = EngineCore::next_candidate_preference(
            preference,
            true,
            phase == SequencePhase::Decode,
            phase == SequencePhase::Prefill,
        );
    }

    assert_eq!(
        admitted,
        vec![
            SequencePhase::Decode,
            SequencePhase::Prefill,
            SequencePhase::Decode,
            SequencePhase::Prefill,
            SequencePhase::Decode,
            SequencePhase::Prefill,
        ]
    );
}

#[test]
fn multi_candidate_admission_reserves_both_phases_without_wasting_width() {
    for (decode_work, prefill_work, expected) in
        [(32, 32, (3, 1)), (1, 32, (1, 3)), (32, 1, (3, 1))]
    {
        let quotas = EngineCore::candidate_phase_quotas(
            4,
            decode_work,
            prefill_work,
            SequencePhase::Decode,
        );
        assert_eq!(quotas, expected);
        assert!(quotas.0 > 0, "decode lost reserved service");
        assert!(quotas.1 > 0, "prefill lost reserved service");
        assert_eq!(quotas.0 + quotas.1, 4, "candidate width was wasted");
    }
}

#[test]
fn blocked_single_slot_preference_does_not_advance_or_idle_its_peer() {
    let preference = SequencePhase::Decode;
    assert_eq!(
        EngineCore::next_candidate_preference(preference, true, false, true),
        SequencePhase::Decode,
        "a blocked preferred phase must retain its next service opportunity"
    );
    assert_eq!(
        EngineCore::candidate_phase_quotas(1, 8, 8, SequencePhase::Prefill),
        (0, 1),
        "the peer phase must remain eligible for the current physical slot"
    );
}

#[test]
fn managed_token_reach_honors_portable_context_and_preserves_cuda_ceiling() {
    for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
        assert_eq!(
            resolve_managed_token_reach(backend, 4_096, Some(40_960)).unwrap(),
            Some(4_096)
        );
        assert_eq!(
            resolve_managed_token_reach(backend, 262_144, Some(40_960)).unwrap(),
            Some(40_960)
        );
    }
}

#[test]
fn managed_token_reach_rejects_zero_and_cuda_requires_model_context() {
    for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
        assert!(resolve_managed_token_reach(backend, 4_096, Some(0)).is_err());
    }
    assert_eq!(
        resolve_managed_token_reach(BackendKind::Cpu, 4_096, None).unwrap(),
        None
    );
    assert!(resolve_managed_token_reach(BackendKind::Cuda, 4_096, None).is_err());
}

fn scheduled_prefill(request_id: &str, sequence_id: u64) -> ScheduledRequest {
    ScheduledRequest {
        plan_id: sequence_id + 1,
        request_id: request_id.to_string(),
        sequence_id,
        num_tokens: 1,
        is_prefill: true,
        num_computed_tokens: 0,
        work: crate::engine::WorkUnit::SequenceStep {
            phase: crate::engine::SequencePhase::Prefill,
            input: crate::engine::InputRange { start: 0, end: 1 },
            max_output_steps: 1,
            auxiliary_state: None,
        },
    }
}

fn scheduled_decode(request_id: &str, sequence_id: u64) -> ScheduledRequest {
    ScheduledRequest {
        plan_id: sequence_id + 101,
        request_id: request_id.to_string(),
        sequence_id,
        num_tokens: 1,
        is_prefill: false,
        num_computed_tokens: 1,
        work: crate::engine::WorkUnit::SequenceStep {
            phase: crate::engine::SequencePhase::Decode,
            input: crate::engine::InputRange { start: 1, end: 2 },
            max_output_steps: 1,
            auxiliary_state: None,
        },
    }
}

fn text_progress(request_id: &str, sequence: usize) -> super::super::StreamingOutput {
    super::super::StreamingOutput {
        request_id: request_id.to_string(),
        sequence,
        samples: Vec::new(),
        sample_rate: 0,
        is_final: false,
        text: Some(format!("delta-{sequence}")),
        stats: None,
        asr_progress: None,
        logprobs: Vec::new(),
    }
}

fn wrap_outputs(
    scheduled: &[ScheduledRequest],
    outputs: Vec<ExecutorOutput>,
) -> Vec<ExecutorStepResult> {
    assert_eq!(scheduled.len(), outputs.len());
    scheduled
        .iter()
        .zip(outputs)
        .map(|(scheduled, output)| ExecutorStepResult::new(scheduled, output))
        .collect()
}

struct MockExecutor {
    initialized: bool,
    cleanup_calls: Arc<Mutex<Vec<String>>>,
    prefill: PrefillMode,
}

impl MockExecutor {
    fn new(cleanup_calls: Arc<Mutex<Vec<String>>>) -> Self {
        Self {
            initialized: false,
            cleanup_calls,
            prefill: PrefillMode::Full,
        }
    }

    fn build_outputs(scheduled: &[ScheduledRequest]) -> Vec<ExecutorOutput> {
        scheduled
            .iter()
            .map(|entry| ExecutorOutput {
                request_id: entry.request_id.clone(),
                audio: None,
                text: None,
                input_transcription: None,
                tokens_processed: entry.num_tokens.max(1),
                tokens_generated: usize::from(!entry.is_prefill),
                finished: false,
                phase_timing_override: None,
                asr_diagnostics: None,
                error: None,
                logprobs: Vec::new(),
            })
            .collect()
    }
}

impl ModelExecutor for MockExecutor {
    fn execution_profile(&self, request: &EngineCoreRequest) -> Option<ExecutionProfile> {
        let mut profile = ExecutionProfile::fail_closed(
            BackendKind::Cpu,
            request.model_variant,
            ExecutionMode::Sequence,
        );
        profile.prefill = self.prefill;
        profile.recompute_safe = true;
        profile.cache_release_safe = true;
        Some(profile)
    }

    fn execute_prefill(
        &self,
        _requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        Ok(wrap_outputs(scheduled, Self::build_outputs(scheduled)))
    }

    fn execute_decode(
        &self,
        _requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        Ok(wrap_outputs(scheduled, Self::build_outputs(scheduled)))
    }

    fn is_ready(&self) -> bool {
        self.initialized
    }

    fn initialize(&mut self) -> Result<()> {
        self.initialized = true;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        self.initialized = false;
        Ok(())
    }

    fn cleanup_request(&self, request_id: &str) -> super::super::executor::CacheReleaseReport {
        if let Ok(mut calls) = self.cleanup_calls.lock() {
            calls.push(request_id.to_string());
        }
        super::super::executor::CacheReleaseReport::confirmed(1)
    }
}

struct ManagedReceiptExecutor;

impl ModelExecutor for ManagedReceiptExecutor {
    fn execution_profile(&self, request: &EngineCoreRequest) -> Option<ExecutionProfile> {
        let mut profile = ExecutionProfile::fail_closed(
            BackendKind::Cpu,
            request.model_variant,
            ExecutionMode::Sequence,
        );
        profile.prefill = PrefillMode::Full;
        profile.cache_mode = CacheMode::ExternalPaged;
        profile.cache_release_safe = true;
        profile.recompute_safe = true;
        Some(profile)
    }

    fn cleanup_session(
        &self,
        _session: &super::super::SessionKey,
    ) -> super::super::executor::CacheReleaseReport {
        super::super::executor::CacheReleaseReport::confirmed(1)
    }

    fn execute_physical_batch(
        &self,
        execution: super::super::executor::PhysicalBatchExecution<'_>,
    ) -> super::super::executor::PhysicalDispatchResult {
        let dispatch = execution.expected_dispatch();
        Ok(execution
            .scheduled
            .iter()
            .zip(&execution.batch.rows)
            .map(|(scheduled, row)| {
                let mut result = ExecutorStepResult::new(
                    scheduled,
                    ExecutorOutput {
                        request_id: scheduled.request_id.clone(),
                        audio: None,
                        text: None,
                        input_transcription: None,
                        tokens_processed: scheduled.num_tokens,
                        tokens_generated: 0,
                        finished: false,
                        phase_timing_override: None,
                        asr_diagnostics: None,
                        error: None,
                        logprobs: Vec::new(),
                    },
                );
                result.dispatch = dispatch;
                if let Some(reservation) = &row.managed_cache {
                    result.managed_cache = Some(reservation.completed_write_receipt_for_test());
                }
                result
            })
            .collect())
    }

    fn execute_prefill(
        &self,
        _requests: &[&EngineCoreRequest],
        _scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        unreachable!("physical dispatch is overridden")
    }

    fn execute_decode(
        &self,
        _requests: &[&EngineCoreRequest],
        _scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        unreachable!("physical dispatch is overridden")
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn initialize(&mut self) -> Result<()> {
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }

    fn cleanup_request(&self, _request_id: &str) -> CacheReleaseReport {
        CacheReleaseReport::confirmed(0)
    }
}

struct RestartSequenceExecutor {
    cancel_during_dispatch: bool,
    cleanup_calls: Arc<AtomicUsize>,
}

impl ModelExecutor for RestartSequenceExecutor {
    fn execution_profile(&self, request: &EngineCoreRequest) -> Option<ExecutionProfile> {
        let mut profile = ExecutionProfile::fail_closed(
            BackendKind::Cpu,
            request.model_variant,
            ExecutionMode::Sequence,
        );
        profile.prefill = PrefillMode::Full;
        profile.cache_mode = CacheMode::ExternalPaged;
        profile.cache_release_safe = true;
        profile.recompute_safe = true;
        Some(profile)
    }

    fn cleanup_session(
        &self,
        _session: &super::super::SessionKey,
    ) -> super::super::executor::CacheReleaseReport {
        self.cleanup_calls.fetch_add(1, Ordering::AcqRel);
        super::super::executor::CacheReleaseReport::confirmed(1)
    }

    fn execute_physical_batch(
        &self,
        execution: super::super::executor::PhysicalBatchExecution<'_>,
    ) -> super::super::executor::PhysicalDispatchResult {
        if self.cancel_during_dispatch {
            for request in execution.requests {
                if let Some(cancellation) = request.cancellation.as_ref() {
                    cancellation.store(true, Ordering::Release);
                }
            }
        }
        let dispatch = execution.expected_dispatch();
        Ok(execution
            .scheduled
            .iter()
            .map(|scheduled| {
                ExecutorStepResult::from_session(
                    scheduled,
                    ModelSessionResult::restart_sequence(
                        scheduled.request_id.clone(),
                        super::super::SequenceRestartReason::ModelFallback,
                    ),
                )
                .with_dispatch(dispatch)
            })
            .collect())
    }

    fn execute_prefill(
        &self,
        _requests: &[&EngineCoreRequest],
        _scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        unreachable!("physical dispatch is overridden")
    }

    fn execute_decode(
        &self,
        _requests: &[&EngineCoreRequest],
        _scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        unreachable!("physical dispatch is overridden")
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn initialize(&mut self) -> Result<()> {
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}

struct DeadlineCleanupExecutor {
    events: Arc<Mutex<Vec<String>>>,
    confirm_cleanup: bool,
}

impl DeadlineCleanupExecutor {
    fn new(events: Arc<Mutex<Vec<String>>>, confirm_cleanup: bool) -> Self {
        Self {
            events,
            confirm_cleanup,
        }
    }

    fn execute_phase(
        &self,
        phase: &str,
        scheduled: &[ScheduledRequest],
    ) -> Vec<ExecutorStepResult> {
        let mut outputs = Vec::with_capacity(scheduled.len());
        for entry in scheduled {
            self.events
                .lock()
                .unwrap()
                .push(format!("execute-{phase}:{}", entry.request_id));
            outputs.push(ExecutorStepResult::new(
                entry,
                ExecutorOutput {
                    request_id: entry.request_id.clone(),
                    audio: None,
                    text: None,
                    input_transcription: None,
                    tokens_processed: entry.num_tokens,
                    tokens_generated: 0,
                    finished: false,
                    phase_timing_override: None,
                    asr_diagnostics: None,
                    error: None,
                    logprobs: Vec::new(),
                },
            ));
        }
        outputs
    }
}

impl ModelExecutor for DeadlineCleanupExecutor {
    fn execution_profile(&self, request: &EngineCoreRequest) -> Option<ExecutionProfile> {
        let mut profile = ExecutionProfile::fail_closed(
            BackendKind::Cpu,
            request.model_variant,
            ExecutionMode::Sequence,
        );
        profile.prefill = PrefillMode::Full;
        // This synthetic executor stands in for an external physical
        // runtime whose exact-session cleanup must be confirmed.
        profile.cache_mode = super::super::execution::CacheMode::ExternalPaged;
        profile.cache_release_safe = true;
        profile.prefix_reuse_safe = true;
        profile.resolved_from_loaded_model = true;
        Some(profile)
    }

    fn execute_prefill(
        &self,
        _requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        Ok(self.execute_phase("prefill", scheduled))
    }

    fn execute_decode(
        &self,
        _requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        Ok(self.execute_phase("decode", scheduled))
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn initialize(&mut self) -> Result<()> {
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }

    fn cleanup_session(
        &self,
        session: &super::super::SessionKey,
    ) -> super::super::executor::CacheReleaseReport {
        self.events
            .lock()
            .unwrap()
            .push(format!("cleanup:{}:{}", session.request_id, session.epoch));
        if self.confirm_cleanup {
            super::super::executor::CacheReleaseReport::confirmed(1)
        } else {
            super::super::executor::CacheReleaseReport::unconfirmed()
        }
    }
}

struct ImmediateFinishExecutor;

impl ModelExecutor for ImmediateFinishExecutor {
    fn execute_prefill(
        &self,
        _requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        let outputs = scheduled
            .iter()
            .map(|entry| ExecutorOutput {
                request_id: entry.request_id.clone(),
                audio: None,
                text: Some(format!("done-{}", entry.request_id)),
                input_transcription: None,
                tokens_processed: entry.num_tokens.max(1),
                tokens_generated: 1,
                finished: true,
                phase_timing_override: None,
                asr_diagnostics: None,
                error: None,
                logprobs: Vec::new(),
            })
            .collect();
        Ok(wrap_outputs(scheduled, outputs))
    }

    fn execute_decode(
        &self,
        _requests: &[&EngineCoreRequest],
        _scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        Ok(Vec::new())
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn initialize(&mut self) -> Result<()> {
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}

struct CancelledExecutor;

impl ModelExecutor for CancelledExecutor {
    fn execution_profile(&self, request: &EngineCoreRequest) -> Option<ExecutionProfile> {
        let mut profile = ExecutionProfile::fail_closed(
            BackendKind::Cpu,
            request.model_variant,
            ExecutionMode::Sequence,
        );
        profile.prefill = PrefillMode::Full;
        Some(profile)
    }

    fn execute_prefill(
        &self,
        _requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        Ok(scheduled
            .iter()
            .map(|entry| {
                ExecutorStepResult::from_session(
                    entry,
                    super::super::executor::ModelSessionResult::cancelled(
                        ExecutorOutput::cancelled(entry.request_id.clone()),
                    ),
                )
            })
            .collect())
    }

    fn execute_decode(
        &self,
        _requests: &[&EngineCoreRequest],
        _scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        Ok(Vec::new())
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn initialize(&mut self) -> Result<()> {
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}

struct SelectiveFailureExecutor;

impl ModelExecutor for SelectiveFailureExecutor {
    fn execute_prefill(
        &self,
        _requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        if scheduled.iter().any(|entry| entry.request_id == "bad") {
            return Err(Error::InferenceError("isolated failure".to_string()));
        }
        let outputs = scheduled
            .iter()
            .map(|entry| ExecutorOutput {
                request_id: entry.request_id.clone(),
                audio: None,
                text: Some("ok".to_string()),
                input_transcription: None,
                tokens_processed: entry.num_tokens,
                tokens_generated: 1,
                finished: true,
                phase_timing_override: None,
                asr_diagnostics: None,
                error: None,
                logprobs: Vec::new(),
            })
            .collect();
        Ok(wrap_outputs(scheduled, outputs))
    }

    fn execute_decode(
        &self,
        _requests: &[&EngineCoreRequest],
        _scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        Ok(Vec::new())
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn initialize(&mut self) -> Result<()> {
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}

struct PhaseTimingExecutor;

impl ModelExecutor for PhaseTimingExecutor {
    fn execute_prefill(
        &self,
        _requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        let outputs = scheduled
            .iter()
            .map(|entry| ExecutorOutput {
                request_id: entry.request_id.clone(),
                audio: None,
                text: Some("done".to_string()),
                input_transcription: None,
                tokens_processed: entry.num_tokens.max(1),
                tokens_generated: 1,
                finished: true,
                phase_timing_override: Some(ExecutorPhaseTiming {
                    media_decode_ms: Some(12.5),
                    sampling_ms: Some(1.25),
                    ..ExecutorPhaseTiming::default()
                }),
                asr_diagnostics: None,
                error: None,
                logprobs: Vec::new(),
            })
            .collect();
        Ok(wrap_outputs(scheduled, outputs))
    }

    fn execute_decode(
        &self,
        _requests: &[&EngineCoreRequest],
        _scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        Ok(Vec::new())
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn initialize(&mut self) -> Result<()> {
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}

fn bind_managed_test_state(
    core: &mut EngineCore,
    request: &mut EngineCoreRequest,
    model_instance: ModelInstanceId,
) {
    let state_contract = crate::kv::test_contract();
    let capability = crate::kv::InferenceStateCapability::Managed(state_contract.clone());
    let physical = core
        .load_managed_model_cache(model_instance, &capability, None)
        .expect("load managed state")
        .expect("physical managed state");
    bind_managed_test_runtime(
        request,
        model_instance,
        state_contract,
        physical,
        PrefillMode::Full,
    );
}

fn bind_managed_test_runtime(
    request: &mut EngineCoreRequest,
    model_instance: ModelInstanceId,
    state_contract: crate::kv::v2::InferenceStateContract,
    physical: Arc<super::super::ManagedKvModelRuntime>,
    prefill: PrefillMode,
) {
    bind_managed_test_runtime_with_batch(
        request,
        model_instance,
        state_contract,
        physical,
        prefill,
        NativeBatchMode::None,
        1,
    );
}

fn bind_managed_test_runtime_with_batch(
    request: &mut EngineCoreRequest,
    model_instance: ModelInstanceId,
    state_contract: crate::kv::v2::InferenceStateContract,
    physical: Arc<super::super::ManagedKvModelRuntime>,
    prefill: PrefillMode,
    batch_mode: NativeBatchMode,
    max_batch_size: usize,
) {
    let variant = request.model_variant.expect("managed test model");
    let mut profile =
        ExecutionProfile::fail_closed(BackendKind::Cpu, Some(variant), ExecutionMode::Sequence);
    profile.prefill = prefill;
    profile.incremental_decode = true;
    profile.cache_mode = CacheMode::ExternalPaged;
    profile.cache_namespace = Some("core-test-state-v2".to_string());
    profile.kv_dtype = "state_v2_resolved".to_string();
    profile.resolved_from_loaded_model = true;
    profile.max_batch_size = max_batch_size;
    let stage = super::super::StageDescriptor::from_execution_profile(
        super::super::StageId::new(1),
        "chat.test.physical",
        &profile,
        batch_mode,
    );
    let execution = super::super::ExecutionAdapterBinding {
        execution_group_id: super::super::ExecutionGroupId::new(1),
        model_instance_id: model_instance,
        adapter_instance_id: super::super::AdapterInstanceId::new(model_instance.get()),
        adapter_abi_revision: super::super::AdapterAbiRevision::new(1),
        model_variant: variant,
        capability_id: "chat".to_string(),
        stages: Arc::from([stage]),
    };
    let descriptor = crate::kv::v2::CapabilityStateDescriptorV2::managed_for_stages_test(
        state_contract,
        &execution.stages,
    );
    let runtime = Arc::new(crate::kv::v2::CapabilityStateRuntimeV2::managed(
        crate::kv::v2::ManagedCapabilityRuntimeV2::seal(
            BackendKind::Cpu,
            &execution,
            descriptor,
            physical,
            crate::kv::v2::RetainedStateUseV2::ExternalPaged,
        )
        .expect("seal managed test runtime"),
    ));
    request
        .bind_execution_adapter(execution)
        .expect("bind test execution");
    request
        .bind_v2_state_runtime(runtime.clone(), runtime.state_fingerprint, BackendKind::Cpu)
        .expect("bind managed test state");
}

fn managed_test_request(
    core: &mut EngineCore,
    model_instance: ModelInstanceId,
    id: &str,
    tokens: Vec<u32>,
) -> EngineCoreRequest {
    let mut request = EngineCoreRequest::chat(vec![ChatMessage {
        role: ChatRole::User,
        content: id.to_string(),
    }])
    .with_model_variant(ModelVariant::Qwen306B);
    request.id = id.to_string();
    request.params.max_tokens = 1;
    request
        .install_chat_execution_preparation(ModelVariant::Qwen306B, tokens, None, 4096)
        .expect("prepare prompt");
    request
        .bind_model_instance(model_instance)
        .expect("model instance");
    bind_managed_test_state(core, &mut request, model_instance);
    request
}

#[test]
fn chat_rows_that_spend_their_budget_report_max_tokens() {
    use super::super::types::FinishReason;
    let stop = Some(FinishReason::StopToken);
    assert_eq!(
        chat_budget_finish_reason(stop, true, 128, 128),
        Some(FinishReason::MaxTokens)
    );
    assert_eq!(chat_budget_finish_reason(stop, true, 128, 127), stop);
    // Only chat budgets count tokens; other tasks and terminal kinds keep
    // their reason.
    assert_eq!(chat_budget_finish_reason(stop, false, 8, 8), stop);
    for reason in [
        Some(FinishReason::Aborted),
        Some(FinishReason::Error),
        None,
    ] {
        assert_eq!(
            chat_budget_finish_reason(reason, true, 8, 8),
            reason
        );
    }
    assert_eq!(chat_budget_finish_reason(stop, true, 0, 0), stop);

    // An LFM3-Audio reply answers chat on a chat turn only.
    let mut request = EngineCoreRequest::lfm3_audio_chat(Vec::new());
    assert!(request.answers_chat());
    request.speech_reply = crate::engine::SpeechReplyTurn::Voice;
    assert!(!request.answers_chat());
    assert!(EngineCoreRequest::chat(Vec::new()).answers_chat());
    assert!(!EngineCoreRequest::tts("hi").answers_chat());
}

#[test]
fn test_engine_core_creation() {
    let config = EngineCoreConfig::default();
    let core = EngineCore::new(config);
    assert!(core.is_ok());
}

#[tokio::test]
async fn test_add_request() {
    let config = EngineCoreConfig::default();
    let mut core = EngineCore::new(config).unwrap();

    let request = EngineCoreRequest::tts("Hello, world!");
    let result = core.add_request(request);
    assert!(result.is_ok());
    assert_eq!(core.pending_request_count(), 1);
}

#[tokio::test]
async fn realtime_session_admission_rejects_an_unbound_voxtral_request() {
    let mut core = EngineCore::new(EngineCoreConfig::default()).unwrap();
    let mut request = EngineCoreRequest::asr_bytes(Vec::new())
        .with_model_variant(ModelVariant::VoxtralMini4BRealtime2602);
    request
        .enable_realtime_asr_ingress()
        .expect("realtime ingress");
    let error = core
        .add_realtime_asr_session(request)
        .expect_err("unbound realtime requests must fail before scheduler insertion");
    assert!(error.to_string().contains("loaded model instance"));
    assert_eq!(core.pending_request_count(), 0);
}

#[tokio::test]
async fn realtime_row_deferred_by_serial_prefill_is_scheduled_again() {
    // Resumable prefill may share a schedule with decode-phase rows.
    let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor {
        prefill: PrefillMode::Incremental,
        ..MockExecutor::new(Arc::new(Mutex::new(Vec::new())))
    }));
    let mut core =
        EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor).unwrap();
    assert_eq!(
        core.config
            .resolved_physical_execution_capacity()
            .candidate_dispatch_limit
            .get(),
        1
    );
    // Admission authenticates a loaded Voxtral binding; this test installs
    // only the scheduler session and request it would register.
    let mut realtime = EngineCoreRequest::asr_bytes(Vec::new())
        .with_model_variant(ModelVariant::VoxtralMini4BRealtime2602);
    realtime.id = "live-transcript".to_string();
    realtime
        .enable_realtime_asr_ingress()
        .expect("realtime ingress");
    assert!(core.scheduler.add_realtime_session(&realtime));
    let session = core
        .get_session_key(&realtime.id)
        .expect("realtime session");
    core.requests
        .insert(realtime.id.clone(), Arc::new(realtime));
    core.enqueue_realtime_asr_push(&session, Arc::from(vec![0.0; 160]), 16_000, 1, 1)
        .await
        .expect("realtime push");
    let mut waiting = EngineCoreRequest::tts("waiting prefill");
    waiting.id = "waiting-prefill".to_string();
    core.add_request(waiting).unwrap();

    // The previous step served decode, so the single slot prefers prefill
    // and defers the realtime decode-phase row.
    core.next_single_candidate_phase
        .insert(None, SequencePhase::Prefill);
    core.step().await.expect("prefill step");
    assert!(core.active_plans.is_empty());

    let prepared = core
        .prepare_step()
        .await
        .unwrap()
        .expect("decode-preferred step");
    assert!(
        core.active_plans
            .values()
            .any(|plan| plan.session == session
                && matches!(plan.work, WorkUnit::RealtimePreparation { .. })),
        "the rolled-back realtime operation must be scheduled again"
    );
    assert_eq!(core.rollback_all_in_flight_dispatches(), 1);
    drop(prepared);
}

#[test]
fn v2_state_publication_fails_closed_until_runtime_is_resolved() {
    let mut core = EngineCore::new(EngineCoreConfig::default()).unwrap();
    let mut request = EngineCoreRequest::tts("v2");
    request
        .bind_model_instance(ModelInstanceId::new(101))
        .unwrap();
    request
        .bind_v2_state_descriptor(
            crate::kv::v2::CapabilityStateDescriptorV2::stateless_for_test(),
            [7; 32],
        )
        .unwrap();

    let error = core
        .add_request(request)
        .expect_err("v2 must never fall back to model-owned execution");
    assert!(error.to_string().contains("resolved v2 runtime"));
    assert_eq!(core.pending_request_count(), 0);
}

#[test]
fn load_sealed_stateless_v2_runtime_enters_the_engine() {
    let mut core = EngineCore::new(EngineCoreConfig {
        backend: BackendKind::Cpu,
        ..EngineCoreConfig::default()
    })
    .unwrap();
    let variant = ModelVariant::Kokoro82M;
    let instance = ModelInstanceId::new(102);
    let profile =
        ExecutionProfile::fail_closed(BackendKind::Cpu, Some(variant), ExecutionMode::Atomic);
    let stage = super::super::StageDescriptor::from_execution_profile(
        super::super::StageId::new(0),
        "tts.compatibility",
        &profile,
        NativeBatchMode::None,
    );
    let execution = super::super::ExecutionAdapterBinding {
        execution_group_id: super::super::ExecutionGroupId::new(1),
        model_instance_id: instance,
        adapter_instance_id: super::super::AdapterInstanceId::new(3),
        adapter_abi_revision: super::super::AdapterAbiRevision::new(1),
        model_variant: variant,
        capability_id: "tts".to_string(),
        stages: Arc::from([stage]),
    };
    let descriptor = crate::kv::v2::CapabilityStateDescriptorV2::stateless_for_stages_test(
        &execution.stages,
    );
    let runtime = Arc::new(crate::kv::v2::CapabilityStateRuntimeV2::stateless(
        crate::kv::v2::StatelessCapabilityRuntimeV2::seal(
            BackendKind::Cpu,
            &execution,
            descriptor,
        )
        .unwrap(),
    ));
    let mut request = EngineCoreRequest::tts("v2 stateless").with_model_variant(variant);
    request.bind_execution_adapter(execution.clone()).unwrap();
    request
        .bind_v2_state_runtime(runtime.clone(), runtime.state_fingerprint, BackendKind::Cpu)
        .unwrap();

    core.add_request(request).unwrap();
    assert_eq!(core.pending_request_count(), 1);

    let cuda_descriptor = crate::kv::v2::CapabilityStateDescriptorV2::stateless_for_stages_test(
        &execution.stages,
    );
    let cuda_runtime = Arc::new(crate::kv::v2::CapabilityStateRuntimeV2::stateless(
        crate::kv::v2::StatelessCapabilityRuntimeV2::seal(
            BackendKind::Cuda,
            &execution,
            cuda_descriptor,
        )
        .unwrap(),
    ));
    let mut mismatched = EngineCoreRequest::tts("wrong backend").with_model_variant(variant);
    mismatched.bind_execution_adapter(execution).unwrap();
    mismatched
        .bind_v2_state_runtime(
            cuda_runtime.clone(),
            cuda_runtime.state_fingerprint,
            BackendKind::Cuda,
        )
        .unwrap();
    assert!(core.add_request(mismatched).is_err());
    assert_eq!(core.pending_request_count(), 1);
}

#[tokio::test]
async fn test_abort_requests_for_variant_only_aborts_matching_requests() {
    let cleanup_calls = Arc::new(Mutex::new(Vec::new()));
    let executor =
        UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(cleanup_calls.clone())));
    let config = EngineCoreConfig::default();
    let mut core = EngineCore::new_with_unified_executor(config, executor).unwrap();

    let mut req_a = EngineCoreRequest::tts("variant-a");
    req_a.id = "req-a".to_string();
    req_a.model_variant = Some(ModelVariant::Qwen354BGguf);

    let mut req_b = EngineCoreRequest::tts("variant-b");
    req_b.id = "req-b".to_string();
    req_b.model_variant = Some(ModelVariant::Qwen359BGguf);

    core.add_request(req_a).unwrap();
    core.add_request(req_b).unwrap();

    let aborted = core
        .abort_requests_for_variant(ModelVariant::Qwen354BGguf)
        .await;
    assert_eq!(aborted, vec!["req-a".to_string()]);
    assert!(!core.has_request(&"req-a".to_string()));
    assert!(core.has_request(&"req-b".to_string()));

    let calls = cleanup_calls.lock().unwrap().clone();
    assert!(calls.iter().any(|id| id == "req-a"));
    assert!(!calls.iter().any(|id| id == "req-b"));
}

#[tokio::test]
async fn stream_backpressure_deadline_preserves_delivery_provenance() {
    let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
        Mutex::new(Vec::new()),
    ))));
    let mut core =
        EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor).unwrap();
    let mut request = EngineCoreRequest::tts("stream deadline");
    request.id = "stream-deadline".to_string();
    core.add_request(request).unwrap();
    let session = core
        .get_session_key(&"stream-deadline".to_string())
        .expect("exact session");

    assert!(
        core.handle_stream_delivery_failure(StreamDeliveryFailure {
            session: session.clone(),
            kind: StreamDeliveryFailureKind::Deadline,
        })
        .await
    );

    let terminal = core
        .pending_terminal_outputs
        .back()
        .expect("stream terminal output");
    assert_eq!(terminal.session, session);
    assert_eq!(
        terminal.disposition,
        ExecutionDisposition::Finished(ExecutionFinishReason::TimedOut)
    );
    assert_eq!(
        terminal.provenance,
        OutcomeProvenance::deadline(
            DeadlinePhase::StreamDelivery,
            DispatchState::ProducedOutput,
        )
    );
}

#[tokio::test]
async fn terminal_stream_delivery_failure_replaces_success_without_duplicate_terminal() {
    let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
        Mutex::new(Vec::new()),
    ))));
    let mut core =
        EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor).unwrap();
    let session = super::super::SessionKey::new("terminal-stream".to_string(), 17);
    let mut outputs = vec![core.output_processor.process_execution(
        ExecutorOutput::terminal(session.request_id.clone()),
        &ExecutionDisposition::Finished(ExecutionFinishReason::Completed),
        session.epoch,
        Duration::ZERO,
    )];

    core.reconcile_stream_delivery_failures(
        &mut outputs,
        vec![StreamDeliveryFailure {
            session,
            kind: StreamDeliveryFailureKind::Delivery,
        }],
    )
    .await;

    assert!(outputs[0].is_finished);
    assert_eq!(outputs[0].finish_reason, Some(OutputFinishReason::Error));
    assert_eq!(
        outputs[0].error.as_deref(),
        Some("committed stream delivery failed")
    );
    assert_eq!(
        outputs[0].provenance,
        OutcomeProvenance::failure(
            FailureOrigin::StreamDelivery,
            DispatchState::ProducedOutput,
        )
    );
    assert!(
        core.pending_terminal_outputs.is_empty(),
        "a completed scheduler session must not receive a second terminal event"
    );
}

/// TPIPE-15: a plan that fails to begin is retried, but only for
/// [`MAX_PLAN_BEGIN_ATTEMPTS`] consecutive steps per session, and a
/// successful begin or another session does not share the count.
#[test]
fn plan_begin_retries_are_bounded_per_session() {
    let mut attempts = HashMap::new();
    let session = super::super::SessionKey::new("stuck".into(), 1);
    let other = super::super::SessionKey::new("other".into(), 1);
    for _ in 1..MAX_PLAN_BEGIN_ATTEMPTS {
        assert!(!plan_begin_attempts_exhausted(&mut attempts, &session));
    }
    assert!(!plan_begin_attempts_exhausted(&mut attempts, &other));
    assert!(plan_begin_attempts_exhausted(&mut attempts, &session));
    attempts.remove(&session);
    assert!(!plan_begin_attempts_exhausted(&mut attempts, &session));
}

#[tokio::test]
async fn adapter_without_a_stage_for_scheduled_work_fails_the_request_once() {
    let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
        Mutex::new(Vec::new()),
    ))));
    let mut core =
        EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor).unwrap();
    let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
    let request_id = "no-stage".to_string();
    let mut request = EngineCoreRequest::tts("no stage").with_model_variant(variant);
    request.id = request_id.clone();
    request.prompt_tokens = vec![1];
    // The executor schedules sequence work, but the adapter only has an
    // atomic stage, so no stage matches before or after the scalar fallback.
    let profile =
        ExecutionProfile::fail_closed(BackendKind::Cpu, Some(variant), ExecutionMode::Atomic);
    let mut stage = super::super::StageDescriptor::from_execution_profile(
        StageId::new(0),
        "tts.atomic",
        &profile,
        NativeBatchMode::None,
    );
    stage.selector = super::super::StageWorkSelector::Atomic;
    request
        .bind_execution_adapter(super::super::ExecutionAdapterBinding {
            execution_group_id: ExecutionGroupId::new(1),
            model_instance_id: ModelInstanceId::new(2),
            adapter_instance_id: AdapterInstanceId::new(3),
            adapter_abi_revision: AdapterAbiRevision::new(1),
            model_variant: variant,
            capability_id: "tts".to_string(),
            stages: Arc::from([stage]),
        })
        .unwrap();
    core.add_request(request).unwrap();

    let mut outputs = Vec::new();
    for _ in 0..3 {
        outputs.extend(
            core.step()
                .await
                .expect("a stage-selection mismatch must not fail the engine step"),
        );
        if !core.has_pending_work() {
            break;
        }
    }

    assert_eq!(outputs.len(), 1);
    assert!(outputs[0].is_finished);
    assert!(
        outputs[0]
            .error
            .as_deref()
            .is_some_and(|error| error.contains("no stage for scheduled work")),
        "{:?}",
        outputs[0].error
    );
    assert_eq!(
        outputs[0].provenance,
        OutcomeProvenance::failure(FailureOrigin::AdapterPlanning, DispatchState::NotStarted)
    );
    assert!(!core.has_request(&request_id));
    assert!(
        !core.has_pending_work(),
        "a failed request must not leave work for the run loop to retry"
    );
}

#[tokio::test]
async fn cancelled_execution_surfaces_aborted_terminal_output() {
    let executor = UnifiedExecutor::new_for_test(Box::new(CancelledExecutor));
    let mut core =
        EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor).unwrap();
    let mut request = EngineCoreRequest::tts("cancelled");
    request.id = "cancelled".to_string();
    request.prompt_tokens = vec![1];
    core.add_request(request).unwrap();

    let outputs = core.step().await.unwrap();
    assert_eq!(outputs.len(), 1);
    assert!(outputs[0].is_finished);
    assert_eq!(
        outputs[0].finish_reason,
        Some(super::super::types::FinishReason::Aborted)
    );
    assert_eq!(outputs[0].error.as_deref(), Some("request cancelled"));
    assert!(!core.has_request(&"cancelled".to_string()));
}

#[tokio::test]
async fn waiting_and_running_aborts_emit_one_cancelled_output_each() {
    let cleanup_calls = Arc::new(Mutex::new(Vec::new()));
    let executor =
        UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(cleanup_calls.clone())));
    let mut core =
        EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor).unwrap();

    let mut waiting = EngineCoreRequest::tts("waiting");
    waiting.id = "abort-waiting".to_string();
    waiting.prompt_tokens = vec![1];
    core.add_request(waiting).unwrap();
    let waiting_session = core.get_session_key(&"abort-waiting".to_string()).unwrap();
    assert!(core.abort_request_session(&waiting_session).await);
    let waiting_outputs = core.step().await.unwrap();
    assert_eq!(waiting_outputs.len(), 1);
    assert_eq!(waiting_outputs[0].sequence_id, waiting_session.epoch);
    assert_eq!(
        waiting_outputs[0].finish_reason,
        Some(super::super::types::FinishReason::Aborted)
    );

    let mut running = EngineCoreRequest::tts("running");
    running.id = "abort-running".to_string();
    running.prompt_tokens = vec![1];
    core.add_request(running).unwrap();
    let progress = core.step().await.unwrap();
    assert_eq!(progress.len(), 1);
    assert!(!progress[0].is_finished);
    let running_session = core.get_session_key(&"abort-running".to_string()).unwrap();
    assert!(core.abort_request_session(&running_session).await);
    let running_outputs = core.step().await.unwrap();
    assert_eq!(running_outputs.len(), 1);
    assert_eq!(running_outputs[0].sequence_id, running_session.epoch);
    assert_eq!(
        running_outputs[0].finish_reason,
        Some(super::super::types::FinishReason::Aborted)
    );
    assert!(core.step().await.unwrap().is_empty());

    let calls = cleanup_calls.lock().unwrap();
    assert_eq!(
        calls
            .iter()
            .filter(|id| id.as_str() == "abort-waiting")
            .count(),
        1
    );
    assert_eq!(
        calls
            .iter()
            .filter(|id| id.as_str() == "abort-running")
            .count(),
        1
    );
}

#[tokio::test]
async fn unconfirmed_cancel_cleanup_is_retried_and_keeps_the_id_fenced() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let executor = UnifiedExecutor::new_for_test(Box::new(DeadlineCleanupExecutor::new(
        events.clone(),
        false,
    )));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            max_batch_size: 1,
            max_tokens_per_step: 1,
            block_size: 1,
            max_blocks: 1,
            enable_adaptive_batching: false,
            ..Default::default()
        },
        executor,
    )
    .unwrap();
    core.retry_policy.cleanup_backoff_base = Duration::ZERO;
    core.retry_policy.cleanup_backoff_max = Duration::ZERO;

    let mut request = EngineCoreRequest::tts("retry cleanup");
    request.id = "cleanup-fence".to_string();
    request.prompt_tokens = vec![1];
    core.add_request(request.clone()).unwrap();
    core.step().await.unwrap();
    let session = core.get_session_key(&request.id).unwrap();
    assert!(core.abort_request_session(&session).await);
    assert!(core.add_request(request.clone()).is_err());

    let outputs = core.step().await.unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(
        outputs[0].finish_reason,
        Some(super::super::types::FinishReason::Aborted)
    );
    assert!(core.step().await.unwrap().is_empty());
    let cleanup_attempts = events
        .lock()
        .unwrap()
        .iter()
        .filter(|event| event.starts_with("cleanup:cleanup-fence:"))
        .count();
    assert!(cleanup_attempts >= 2, "cleanup was not retried");
    assert!(core.add_request(request).is_err());
}

#[tokio::test]
async fn cleanup_confirmation_does_not_resolve_on_cancellation_acceptance() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let executor =
        UnifiedExecutor::new_for_test(Box::new(DeadlineCleanupExecutor::new(events, false)));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            max_batch_size: 1,
            max_tokens_per_step: 1,
            block_size: 1,
            max_blocks: 1,
            enable_adaptive_batching: false,
            ..Default::default()
        },
        executor,
    )
    .unwrap();

    let mut request = EngineCoreRequest::tts("cleanup receipt");
    request.id = "cleanup-receipt-pending".to_string();
    request.prompt_tokens = vec![1];
    core.add_request(request).unwrap();
    core.step().await.unwrap();
    let session = core
        .get_session_key(&"cleanup-receipt-pending".to_string())
        .unwrap();
    let mut confirmation = core.begin_confirmed_session_cleanup(&session).await;

    assert!(matches!(
        confirmation.try_recv(),
        Err(tokio::sync::oneshot::error::TryRecvError::Empty)
    ));
}

#[tokio::test]
async fn cleanup_confirmation_resolves_for_an_already_terminal_session() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let executor =
        UnifiedExecutor::new_for_test(Box::new(DeadlineCleanupExecutor::new(events, true)));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            max_batch_size: 1,
            max_tokens_per_step: 1,
            block_size: 1,
            max_blocks: 1,
            enable_adaptive_batching: false,
            ..Default::default()
        },
        executor,
    )
    .unwrap();

    let mut request = EngineCoreRequest::tts("finished cleanup receipt");
    request.id = "cleanup-receipt-terminal".to_string();
    request.prompt_tokens = vec![1];
    core.add_request(request).unwrap();
    core.step().await.unwrap();
    let session = core
        .get_session_key(&"cleanup-receipt-terminal".to_string())
        .unwrap();
    assert!(core.abort_request_session(&session).await);

    let confirmation = core.begin_confirmed_session_cleanup(&session).await;
    confirmation
        .await
        .expect("already-terminal cleanup sender must remain armed")
        .expect("already-terminal cleanup must be confirmed");
}

#[tokio::test]
async fn orderly_shutdown_settles_a_pending_cleanup_receipt() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let executor =
        UnifiedExecutor::new_for_test(Box::new(DeadlineCleanupExecutor::new(events, false)));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            max_batch_size: 1,
            max_tokens_per_step: 1,
            block_size: 1,
            max_blocks: 1,
            enable_adaptive_batching: false,
            ..Default::default()
        },
        executor,
    )
    .unwrap();

    let mut request = EngineCoreRequest::tts("shutdown cleanup receipt");
    request.id = "cleanup-receipt-shutdown".to_string();
    request.prompt_tokens = vec![1];
    core.add_request(request).unwrap();
    core.step().await.unwrap();
    let session = core
        .get_session_key(&"cleanup-receipt-shutdown".to_string())
        .unwrap();
    let confirmation = core.begin_confirmed_session_cleanup(&session).await;

    core.shutdown().await.expect("orderly shutdown");
    confirmation
        .await
        .expect("shutdown must explicitly settle the receipt")
        .expect("successful shutdown is a final cleanup fence");
    assert!(core.cleanup_confirmation_waiters.is_empty());
    assert!(core.pending_managed_releases.is_empty());
}

#[tokio::test]
async fn aborted_terminal_output_survives_plan_failure_until_successful_delivery() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let executor =
        UnifiedExecutor::new_for_test(Box::new(DeadlineCleanupExecutor::new(events, true)));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            max_batch_size: 1,
            max_tokens_per_step: 1,
            min_tokens_per_step: 1,
            block_size: 1,
            max_blocks: 4,
            enable_chunked_prefill: false,
            enable_adaptive_batching: false,
            backend: BackendKind::Cpu,
            ..Default::default()
        },
        executor,
    )
    .unwrap();

    let mut aborted = EngineCoreRequest::tts("durable terminal outbox");
    aborted.id = "durable-abort".to_string();
    aborted.prompt_tokens = vec![1];
    core.add_request(aborted.clone()).unwrap();
    core.step().await.unwrap();

    let aborted_session = core.get_session_key(&aborted.id).unwrap();
    assert!(core.abort_request_session(&aborted_session).await);
    assert!(core.has_pending_terminal_output(&aborted.id));
    assert!(core.add_request(aborted.clone()).is_err());

    let mut plan_fault = EngineCoreRequest::tts("force plan validation failure");
    plan_fault.id = "plan-fault".to_string();
    plan_fault.prompt_tokens = vec![2];
    core.add_request(plan_fault.clone()).unwrap();
    let plan_fault_session = core.get_session_key(&plan_fault.id).unwrap();
    core.execution_trackers.insert(
        plan_fault.id.clone(),
        ExecutionTracker::new(super::super::SessionKey::new(
            plan_fault.id.clone(),
            plan_fault_session.epoch + 1,
        )),
    );

    let error = core.step().await.expect_err("plan preparation must fail");
    assert!(error
        .to_string()
        .contains("already has a different active session"));
    assert!(core.has_pending_terminal_output(&aborted.id));
    assert_eq!(
        core.scheduler
            .pending_release_confirmation_required(&aborted_session),
        Some(true)
    );
    assert!(core.add_request(aborted.clone()).is_err());

    core.execution_trackers.remove(&plan_fault.id);
    assert!(core
        .active_plans
        .values()
        .all(|plan| plan.session != plan_fault_session));
    let retry_outputs = core.step().await.unwrap();
    let aborted_outputs: Vec<_> = retry_outputs
        .iter()
        .filter(|output| output.request_id == aborted.id)
        .collect();
    assert_eq!(aborted_outputs.len(), 1);
    assert_eq!(aborted_outputs[0].sequence_id, aborted_session.epoch);
    assert_eq!(
        aborted_outputs[0].finish_reason,
        Some(super::super::types::FinishReason::Aborted)
    );
    assert!(!core.has_pending_terminal_output(&aborted.id));
    assert_eq!(
        core.scheduler
            .pending_release_confirmation_required(&aborted_session),
        Some(true),
        "the delivered batch remains fenced until its outer consumer routes it"
    );
    assert!(core.add_request(aborted.clone()).is_err());
    assert!(core.acknowledge_terminal_output(&aborted_session));
    assert_eq!(
        core.scheduler
            .pending_release_confirmation_required(&aborted_session),
        None
    );

    let later_outputs = core.step().await.unwrap();
    assert!(later_outputs
        .iter()
        .all(|output| output.request_id != aborted.id));
    core.add_request(aborted).unwrap();
    assert_ne!(
        core.get_session_key(&"durable-abort".to_string())
            .unwrap()
            .epoch,
        aborted_session.epoch
    );
}

#[tokio::test]
async fn stale_session_cannot_abort_reused_request_id() {
    let cleanup_calls = Arc::new(Mutex::new(Vec::new()));
    let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(cleanup_calls)));
    let mut core =
        EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor).unwrap();
    let mut first = EngineCoreRequest::tts("first");
    first.id = "reused".to_string();
    core.add_request(first).unwrap();
    let first_session = core.get_session_key(&"reused".to_string()).unwrap();
    assert!(core.abort_request_session(&first_session).await);

    let mut second = EngineCoreRequest::tts("second");
    second.id = "reused".to_string();
    assert!(
        core.add_request(second.clone()).is_err(),
        "public ID must remain fenced until cancellation is delivered"
    );
    let outputs = core.step().await.unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].sequence_id, first_session.epoch);
    assert_eq!(
        outputs[0].finish_reason,
        Some(super::super::types::FinishReason::Aborted)
    );
    assert_eq!(outputs[0].error.as_deref(), Some("request cancelled"));

    assert!(
        core.add_request(second.clone()).is_err(),
        "returning a terminal output must not acknowledge delivery inside the core"
    );
    assert!(core.acknowledge_terminal_output(&first_session));
    core.add_request(second).unwrap();
    let second_session = core.get_session_key(&"reused".to_string()).unwrap();
    assert_ne!(first_session.epoch, second_session.epoch);
    assert!(!core.abort_request_session(&first_session).await);
    assert!(core.has_request(&"reused".to_string()));
}

#[test]
fn test_merge_executor_output_replaces_cumulative_audio_snapshots() {
    let first = ExecutorOutput {
        request_id: "req-a".to_string(),
        audio: Some(AudioOutput::new(vec![0.1, 0.2], 24_000)),
        text: Some("hello".to_string()),
        input_transcription: None,
        tokens_processed: 1,
        tokens_generated: 1,
        finished: false,
        phase_timing_override: None,
        asr_diagnostics: None,
        error: None,
        logprobs: Vec::new(),
    };
    let second = ExecutorOutput {
        request_id: "req-a".to_string(),
        audio: Some(AudioOutput::new(vec![0.1, 0.2, 0.3], 24_000)),
        text: Some("hello world".to_string()),
        input_transcription: Some("hello there".to_string()),
        tokens_processed: 1,
        tokens_generated: 1,
        finished: true,
        phase_timing_override: None,
        asr_diagnostics: None,
        error: None,
        logprobs: Vec::new(),
    };

    let merged = EngineCore::merge_executor_output(Some(first), second);
    let audio = merged.audio.expect("merged audio");
    assert_eq!(audio.samples, vec![0.1, 0.2, 0.3]);
    assert_eq!(merged.input_transcription.as_deref(), Some("hello there"));
    assert_eq!(merged.text.as_deref(), Some("hello world"));
    assert_eq!(merged.tokens_processed, 2);
    assert_eq!(merged.tokens_generated, 2);
    assert!(merged.finished);
}

#[test]
fn loaded_stage_contract_preserves_independent_request_parallelism() {
    let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
    let model_instance = super::super::ModelInstanceId::new(2);
    let mut request = EngineCoreRequest::tts("contract").with_model_variant(variant);
    request.bind_model_instance(model_instance).unwrap();
    let mut declared =
        ExecutionProfile::fail_closed(BackendKind::Cpu, Some(variant), ExecutionMode::Atomic);
    declared.prefill_batch = NativeBatchMode::Static;
    declared.max_batch_size = 8;
    declared.concurrency = ConcurrencyClass::Batchable;
    let compatibility_stage = super::super::StageDescriptor::from_execution_profile(
        super::super::StageId::new(0),
        "tts.compatibility",
        &declared,
        NativeBatchMode::None,
    );
    request
        .bind_execution_adapter(super::super::ExecutionAdapterBinding {
            execution_group_id: super::super::ExecutionGroupId::new(1),
            model_instance_id: model_instance,
            adapter_instance_id: super::super::AdapterInstanceId::new(3),
            adapter_abi_revision: super::super::AdapterAbiRevision::new(1),
            model_variant: variant,
            capability_id: "tts".to_string(),
            stages: Arc::from([compatibility_stage]),
        })
        .unwrap();

    let effective = EngineCore::apply_adapter_execution_contract(&request, declared).unwrap();
    assert_eq!(effective.prefill_batch, NativeBatchMode::None);
    assert_eq!(effective.decode_batch, NativeBatchMode::None);
    assert_eq!(effective.max_batch_size, 8);
    assert_eq!(effective.concurrency, ConcurrencyClass::Batchable);
}

/// A chat request bound to one stage of `batch_mode` selected by
/// `selector`, for adapter-contract tests.
fn chat_request_with_one_stage(
    selector: super::super::StageWorkSelector,
    batch_mode: NativeBatchMode,
) -> (EngineCoreRequest, ExecutionProfile) {
    let variant = ModelVariant::Qwen306B;
    let model_instance = super::super::ModelInstanceId::new(2);
    let mut request = EngineCoreRequest::chat(vec![ChatMessage {
        role: ChatRole::User,
        content: "contract".into(),
    }])
    .with_model_variant(variant);
    request.bind_model_instance(model_instance).unwrap();
    let mut declared =
        ExecutionProfile::fail_closed(BackendKind::Cpu, Some(variant), ExecutionMode::Sequence);
    declared.prefill = PrefillMode::Incremental;
    declared.incremental_decode = true;
    declared.decode_batch = NativeBatchMode::Continuous;
    declared.max_batch_size = 8;
    declared.concurrency = ConcurrencyClass::Batchable;
    let mut stage = super::super::StageDescriptor::from_execution_profile(
        super::super::StageId::new(3),
        "chat.step.mixed",
        &declared,
        batch_mode,
    );
    stage.selector = selector;
    request
        .bind_execution_adapter(super::super::ExecutionAdapterBinding {
            execution_group_id: super::super::ExecutionGroupId::new(1),
            model_instance_id: model_instance,
            adapter_instance_id: super::super::AdapterInstanceId::new(3),
            adapter_abi_revision: super::super::AdapterAbiRevision::new(1),
            model_variant: variant,
            capability_id: "chat".to_string(),
            stages: Arc::from([stage]),
        })
        .unwrap();
    (request, declared)
}

#[test]
fn one_mixed_stage_for_both_phases_marks_the_profile_mixed() {
    let (request, declared) = chat_request_with_one_stage(
        super::super::StageWorkSelector::SequenceStep,
        NativeBatchMode::Mixed,
    );
    let effective =
        EngineCore::apply_adapter_execution_contract(&request, declared.clone()).unwrap();
    assert!(effective.mixed_step);
    assert!(effective.capabilities().mixed_phase_batch);
    // The executor's declared phases are kept: decode rows stay one-token
    // continuous rows and prefill stays resumable.
    assert_eq!(effective.prefill, PrefillMode::Incremental);
    assert_eq!(effective.decode_batch, NativeBatchMode::Continuous);
    assert_eq!(effective.max_batch_size, 8);

    // A mixed stage needs resumable prefill from the executor.
    let mut full = declared;
    full.prefill = PrefillMode::Full;
    assert!(EngineCore::apply_adapter_execution_contract(&request, full).is_err());

    // A split contract never marks the profile mixed.
    let (request, declared) = chat_request_with_one_stage(
        super::super::StageWorkSelector::Any,
        NativeBatchMode::None,
    );
    let effective = EngineCore::apply_adapter_execution_contract(&request, declared).unwrap();
    assert!(!effective.mixed_step);
}

#[test]
// Explicitly dropping the closure releases its captured request Arc before
// the test asserts unique ownership through Arc::get_mut.
#[allow(clippy::drop_non_drop)]
fn incremental_progress_requires_exact_batch_plan_session_lane_and_sequence() {
    let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
        Mutex::new(Vec::new()),
    ))));
    let mut core = EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor)
        .expect("core");
    let request_id = "fenced-progress".to_string();
    let mut request = EngineCoreRequest::tts("fenced progress");
    request.id = request_id.clone();
    request.prompt_tokens = vec![1];
    request.streaming = true;
    let cancellation = Arc::new(AtomicBool::new(false));
    request.cancellation = Some(cancellation.clone());
    let (delivery_tx, _delivery_rx) = mpsc::channel(8);
    request.streaming_tx = Some(delivery_tx);
    core.add_request(request).unwrap();
    let session = core.get_session_key(&request_id).unwrap();

    let profile = ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Atomic);
    let mut stage = super::super::StageDescriptor::from_execution_profile(
        StageId::new(51),
        "test.atomic.incremental",
        &profile,
        NativeBatchMode::None,
    );
    stage.selector = super::super::StageWorkSelector::Atomic;
    stage.output_visibility = OutputVisibility::IncrementalCommitted;
    let plan_id = 7001;
    let plan = ExecutionPlan {
        plan_id,
        session: session.clone(),
        work: WorkUnit::AtomicJob {
            kind: "tts".to_string(),
        },
        batch_key: BatchKey {
            backend: BackendKind::Cpu,
            model_variant: None,
            task_type: TaskType::TTS,
            work_kind: "tts".to_string(),
            compute_dtype: "f32".to_string(),
            kv_dtype: "none".to_string(),
            cache_namespace: "none".to_string(),
            adapter: None,
        },
        batch_mode: NativeBatchMode::None,
        max_batch_size: 1,
        estimate: ResourceVector::zero(),
        stage: Some(stage),
    };
    let mut tracker = ExecutionTracker::new(session.clone());
    tracker.transition(ExecutionState::Admitted).unwrap();
    tracker.transition(ExecutionState::AtomicRunning).unwrap();
    tracker.begin_plan(&plan).unwrap();
    core.execution_trackers.insert(request_id.clone(), tracker);
    core.active_plans.insert(plan_id, plan.clone());

    let lane = EngineCore::batch_lane(&plan, WorkCost::new(1, 1, 0));
    let batch_id = BatchId::new(81);
    let request = core.requests.get(&request_id).unwrap().clone();
    let scheduled = ScheduledRequest {
        plan_id,
        request_id: request_id.clone(),
        sequence_id: session.epoch,
        num_tokens: 1,
        is_prefill: true,
        num_computed_tokens: 0,
        work: plan.work.clone(),
    };
    let dispatch = PreparedPhysicalDispatch::new(
        ExecutionPhase::Prefill,
        PhysicalBatch {
            batch_id,
            lane: lane.clone(),
            mode: NativeBatchMode::None,
            budget: BatchBudget::width_one(),
            rows: vec![ReadyQuantum {
                plan_id,
                session: session.clone(),
                lane: lane.clone(),
                work: plan.work.clone(),
                cost: WorkCost::new(1, 1, 0),
                managed_cache: None,
            }],
            materialized_tensor_elements: 1,
            workspace: ResourceVector::zero(),
        },
        vec![request.clone()],
        vec![scheduled],
        OutputVisibility::IncrementalCommitted,
        Vec::new(),
        PhysicalLaunchPolicy::ExecutionGroupExclusive,
    )
    .unwrap();
    core.register_prepared_physical_dispatch(&dispatch).unwrap();
    let staging = request.stream_staging_buffer();
    let (progress_tx, mut progress_rx) = mpsc::channel(8);
    let binding = request
        .bind_stream_quantum(
            batch_id,
            lane.clone(),
            plan_id,
            session.clone(),
            OutputVisibility::IncrementalCommitted,
            progress_tx,
            StreamProgressBudget::new(STREAM_PROGRESS_MAX_BUFFERED_BYTES),
        )
        .unwrap();
    let mut next_progress = |sequence| {
        staging
            .push_with_policy(text_progress(&request_id, sequence), request.stream_policy)
            .unwrap();
        progress_rx.try_recv().expect("fenced progress")
    };

    let mut wrong_batch = next_progress(0);
    wrong_batch.batch_id = BatchId::new(82);
    assert!(core
        .commit_incremental_stream_progress(wrong_batch)
        .unwrap_err()
        .to_string()
        .contains("inactive physical batch"));

    let mut wrong_plan = next_progress(0);
    wrong_plan.plan_id += 1;
    assert!(core
        .commit_incremental_stream_progress(wrong_plan)
        .unwrap_err()
        .to_string()
        .contains("inactive plan"));

    let mut wrong_session = next_progress(0);
    wrong_session.session.epoch += 1;
    assert!(core
        .commit_incremental_stream_progress(wrong_session)
        .unwrap_err()
        .to_string()
        .contains("session does not match"));

    let mut wrong_lane = next_progress(0);
    wrong_lane.lane.shape_bucket.push_str(".stale");
    assert!(core
        .commit_incremental_stream_progress(wrong_lane)
        .unwrap_err()
        .to_string()
        .contains("physical batch fence"));

    let wrong_sequence = next_progress(4);
    assert!(core
        .commit_incremental_stream_progress(wrong_sequence)
        .unwrap_err()
        .to_string()
        .contains("did not match expected 0"));

    let delivery = core
        .commit_incremental_stream_progress(next_progress(0))
        .expect("exact progress must commit");
    drop(delivery);
    assert_eq!(core.stream_sequence_cursors.get(&session), Some(&1));
    assert!(core.incremental_stream_sessions.contains(&session));

    assert!(core
        .commit_incremental_stream_progress(next_progress(0))
        .unwrap_err()
        .to_string()
        .contains("did not match expected 1"));

    cancellation.store(true, Ordering::Release);
    assert_eq!(
        core.commit_incremental_stream_progress(next_progress(1))
            .unwrap_err()
            .kind,
        StreamDeliveryFailureKind::Cancelled
    );
    cancellation.store(false, Ordering::Release);

    drop(next_progress);
    drop(binding);
    drop(dispatch);
    drop(request);
    Arc::get_mut(core.requests.get_mut(&request_id).expect("active request"))
        .expect("core owns the last request reference")
        .deadline = Some(Instant::now());
    let request = core.requests.get(&request_id).unwrap().clone();
    let staging = request.stream_staging_buffer();
    let (progress_tx, mut progress_rx) = mpsc::channel(8);
    let _binding = request
        .bind_stream_quantum(
            batch_id,
            lane,
            plan_id,
            session,
            OutputVisibility::IncrementalCommitted,
            progress_tx,
            StreamProgressBudget::new(STREAM_PROGRESS_MAX_BUFFERED_BYTES),
        )
        .unwrap();
    staging
        .push_with_policy(text_progress(&request_id, 1), request.stream_policy)
        .unwrap();
    assert_eq!(
        core.commit_incremental_stream_progress(
            progress_rx.try_recv().expect("deadline-fenced progress")
        )
        .unwrap_err()
        .kind,
        StreamDeliveryFailureKind::RequestDeadline
    );
}

#[test]
fn physical_batch_formation_uses_exact_loaded_lane_identity() {
    let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
        Mutex::new(Vec::new()),
    ))));
    let mut core = EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor)
        .expect("core");
    for id in ["physical-a", "physical-b"] {
        let mut request = EngineCoreRequest::tts("physical batch");
        request.id = id.to_string();
        core.add_request(request).unwrap();
    }
    let scheduled = ["physical-a", "physical-b"]
        .into_iter()
        .map(|id| {
            let epoch = core.get_session_key(&id.to_string()).unwrap().epoch;
            scheduled_prefill(id, epoch)
        })
        .collect::<Vec<_>>();
    let mut profile =
        ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Sequence);
    profile.max_batch_size = 2;
    let mut stage = super::super::StageDescriptor::from_execution_profile(
        super::super::StageId::new(4),
        "tts.generate",
        &profile,
        NativeBatchMode::Static,
    );
    stage.workspace_base_bytes = 4;
    stage.workspace_per_row_bytes = 2;
    stage.workspace_per_work_unit_bytes = 1;
    stage.max_workspace_bytes = 10;
    let adapter_key = super::super::AdapterBindingKey {
        execution_group_id: super::super::ExecutionGroupId::new(1),
        model_instance_id: super::super::ModelInstanceId::new(2),
        adapter_instance_id: super::super::AdapterInstanceId::new(3),
        adapter_abi_revision: super::super::AdapterAbiRevision::new(1),
        capability_id: "tts".to_string(),
        stage_id: stage.id,
    };
    for item in &scheduled {
        core.active_plans.insert(
            item.plan_id,
            ExecutionPlan {
                plan_id: item.plan_id,
                session: item.session_key(),
                work: item.work.clone(),
                batch_key: BatchKey {
                    backend: BackendKind::Cpu,
                    model_variant: None,
                    task_type: TaskType::TTS,
                    work_kind: "prefill".to_string(),
                    compute_dtype: "f32".to_string(),
                    kv_dtype: "none".to_string(),
                    cache_namespace: "none".to_string(),
                    adapter: Some(adapter_key.clone()),
                },
                batch_mode: NativeBatchMode::Static,
                max_batch_size: 2,
                estimate: ResourceVector::zero(),
                stage: Some(stage.clone()),
            },
        );
    }
    let requests = scheduled
        .iter()
        .map(|item| core.requests.get(&item.request_id).unwrap().clone())
        .collect::<Vec<_>>();

    let active_plans = core.active_plans.clone();
    let joined = core
        .form_physical_batches(
            ExecutionPhase::Prefill,
            &requests,
            &scheduled,
            usize::MAX,
            &mut Vec::new(),
            &mut Vec::new(),
            &mut Vec::new(),
        )
        .unwrap();
    assert_eq!(joined.len(), 1);
    assert_eq!(joined[0].physical_batch().rows.len(), 2);
    assert_eq!(
        joined[0].physical_batch().workspace,
        ResourceVector {
            host_bytes: ResourceAmount::Known(10),
            ..ResourceVector::zero()
        }
    );
    assert_eq!(
        joined[0].physical_batch().lane.model_instance,
        super::super::ModelInstanceId::new(2)
    );
    assert_eq!(core.rollback_all_in_flight_dispatches(), 1);
    core.active_plans.extend(active_plans.by_plan);

    core.active_plans
        .get_mut(&scheduled[1].plan_id)
        .unwrap()
        .batch_key
        .adapter
        .as_mut()
        .unwrap()
        .model_instance_id = super::super::ModelInstanceId::new(9);
    let mut deferred = Vec::new();
    let split = core
        .form_physical_batches(
            ExecutionPhase::Prefill,
            &requests,
            &scheduled,
            usize::MAX,
            &mut Vec::new(),
            &mut deferred,
            &mut Vec::new(),
        )
        .unwrap();
    assert_eq!(split.len(), 1);
    assert_eq!(split[0].physical_batch().rows.len(), 1);
    assert_eq!(deferred.len(), 1);
    assert_eq!(deferred[0].plan_id, scheduled[1].plan_id);
}

#[test]
fn cuda_batch_workspace_preserves_host_and_device_domains() {
    let lane = BatchLaneKey {
        execution_group: super::super::ExecutionGroupId::new(1),
        model_instance: super::super::ModelInstanceId::new(2),
        adapter_instance: super::super::AdapterInstanceId::new(3),
        adapter_abi: super::super::AdapterAbiRevision::new(1),
        capability_id: "chat".to_string(),
        stage_id: super::super::StageId::new(4),
        backend: BackendKind::Cuda,
        device_ordinal: None,
        compute_dtype: "f16".to_string(),
        state_dtype: "f16".to_string(),
        tensor_layout: "ragged".to_string(),
        quantization: "none".to_string(),
        state_schema: "test".to_string(),
        kernel_mode: "continuous".to_string(),
        semantic_mode: "decode".to_string(),
        shape_bucket: "ragged".to_string(),
    };
    let row = ReadyQuantum {
        plan_id: 1,
        session: super::super::SessionKey::new("workspace".to_string(), 1),
        lane,
        work: WorkUnit::SequenceStep {
            phase: super::super::SequencePhase::Decode,
            input: super::super::InputRange { start: 1, end: 2 },
            max_output_steps: 1,
            auxiliary_state: None,
        },
        cost: WorkCost::with_workspace(
            1,
            1,
            ResourceVector {
                host_bytes: ResourceAmount::Known(32),
                temporary_bytes: ResourceAmount::Known(8_192),
                ..ResourceVector::zero()
            },
        ),
        managed_cache: None,
    };

    assert_eq!(
        PhysicalBatchAssembly::workspace_resources(
            BackendKind::Cuda,
            StageShapePolicy::Ragged,
            16,
            std::slice::from_ref(&row),
        ),
        Some(ResourceVector {
            host_bytes: ResourceAmount::Known(32),
            device_bytes: ResourceAmount::Known(8_208),
            ..ResourceVector::zero()
        })
    );
}

#[test]
fn padded_batch_charges_every_row_at_cohort_maximum() {
    let lane = BatchLaneKey {
        execution_group: super::super::ExecutionGroupId::new(1),
        model_instance: super::super::ModelInstanceId::new(2),
        adapter_instance: super::super::AdapterInstanceId::new(3),
        adapter_abi: super::super::AdapterAbiRevision::new(1),
        capability_id: "asr".to_string(),
        stage_id: super::super::StageId::new(4),
        backend: BackendKind::Cuda,
        device_ordinal: None,
        compute_dtype: "f16".to_string(),
        state_dtype: "f16".to_string(),
        tensor_layout: "padded".to_string(),
        quantization: "none".to_string(),
        state_schema: "test".to_string(),
        kernel_mode: "static".to_string(),
        semantic_mode: "preparation".to_string(),
        shape_bucket: "padded".to_string(),
    };
    let row = |id, elements, workspace| ReadyQuantum {
        plan_id: id,
        session: super::super::SessionKey::new(format!("padded-{id}"), 1),
        lane: lane.clone(),
        work: WorkUnit::AtomicJob {
            kind: "padded".into(),
        },
        cost: WorkCost::new(1, elements, workspace),
        managed_cache: None,
    };
    let rows = [row(1, 100, 1_000), row(2, 400, 4_000)];

    assert_eq!(
        PhysicalBatchAssembly::materialized_tensor_elements(StageShapePolicy::Padded, &rows),
        Some(800)
    );
    assert_eq!(
        PhysicalBatchAssembly::workspace_resources(
            BackendKind::Cuda,
            StageShapePolicy::Padded,
            16,
            &rows,
        ),
        Some(ResourceVector {
            device_bytes: ResourceAmount::Known(8_016),
            ..ResourceVector::zero()
        })
    );
}

#[test]
fn certified_independent_rows_form_bounded_scalar_candidates() {
    let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
        Mutex::new(Vec::new()),
    ))));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            physical_execution_mode: crate::config::PhysicalExecutionMode::Concurrent,
            max_physical_in_flight: crate::config::PhysicalInFlightLimit::new(2).unwrap(),
            ..EngineCoreConfig::default()
        },
        executor,
    )
    .expect("core");
    let ids = [
        "parallel-a",
        "parallel-b",
        "parallel-c",
        "parallel-d",
        "parallel-e",
    ];
    for id in ids {
        let mut request = EngineCoreRequest::tts(format!("independent {id}"));
        request.id = id.to_string();
        core.add_request(request).unwrap();
    }
    let scheduled = ids
        .into_iter()
        .map(|id| {
            let epoch = core.get_session_key(&id.to_string()).unwrap().epoch;
            scheduled_prefill(id, epoch)
        })
        .collect::<Vec<_>>();
    let mut profile =
        ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Sequence);
    profile.concurrency = ConcurrencyClass::Batchable;
    profile.physical_launch_policy = PhysicalLaunchPolicy::concurrent(2).unwrap();
    profile.max_batch_size = 2;
    let stage = super::super::StageDescriptor::from_execution_profile(
        super::super::StageId::new(6),
        "tts.compatibility",
        &profile,
        NativeBatchMode::None,
    );
    let adapter_key = super::super::AdapterBindingKey {
        execution_group_id: super::super::ExecutionGroupId::new(1),
        model_instance_id: super::super::ModelInstanceId::new(2),
        adapter_instance_id: super::super::AdapterInstanceId::new(3),
        adapter_abi_revision: super::super::AdapterAbiRevision::new(1),
        capability_id: "tts".to_string(),
        stage_id: stage.id,
    };
    for item in &scheduled {
        core.active_plans.insert(
            item.plan_id,
            ExecutionPlan {
                plan_id: item.plan_id,
                session: item.session_key(),
                work: item.work.clone(),
                batch_key: BatchKey {
                    backend: BackendKind::Cpu,
                    model_variant: None,
                    task_type: TaskType::TTS,
                    work_kind: "prefill".to_string(),
                    compute_dtype: "f32".to_string(),
                    kv_dtype: "none".to_string(),
                    cache_namespace: "none".to_string(),
                    adapter: Some(adapter_key.clone()),
                },
                batch_mode: NativeBatchMode::None,
                max_batch_size: 2,
                estimate: ResourceVector::zero(),
                stage: Some(stage.clone()),
            },
        );
    }
    let requests = scheduled
        .iter()
        .map(|item| core.requests.get(&item.request_id).unwrap().clone())
        .collect::<Vec<_>>();

    let mut deferred = Vec::new();
    let batches = core
        .form_physical_batches(
            ExecutionPhase::Prefill,
            &requests,
            &scheduled,
            usize::MAX,
            &mut Vec::new(),
            &mut deferred,
            &mut Vec::new(),
        )
        .unwrap();
    let widths = batches
        .iter()
        .map(|batch| batch.physical_batch().rows.len())
        .collect::<Vec<_>>();
    assert_eq!(widths, vec![1, 1]);
    assert_eq!(deferred.len(), 3);
    assert!(batches
        .iter()
        .all(|batch| batch.physical_batch().mode == NativeBatchMode::None));
    assert!(batches
        .iter()
        .all(|batch| { batch.physical_batch().lane.shape_bucket == "independent" }));
}

#[test]
fn padded_sequence_lanes_separate_auxiliary_and_primary_only_rows() {
    let mut profile = ExecutionProfile::fail_closed(
        BackendKind::Cpu,
        Some(ModelVariant::VibeVoiceAsr),
        ExecutionMode::Sequence,
    );
    profile.prefill_batch = NativeBatchMode::Static;
    profile.max_batch_size = 2;
    let stage = super::super::StageDescriptor::from_execution_profile(
        StageId::new(19),
        "asr.prefill.tensor_static",
        &profile,
        NativeBatchMode::Static,
    );
    let base = ExecutionPlan {
        plan_id: 1,
        session: SessionKey::new("padded-lane".into(), 1),
        work: WorkUnit::SequenceStep {
            phase: SequencePhase::Prefill,
            input: InputRange::new(0, 1).unwrap(),
            max_output_steps: 1,
            auxiliary_state: Some(Arc::from([])),
        },
        batch_key: BatchKey {
            backend: BackendKind::Cpu,
            model_variant: Some(ModelVariant::VibeVoiceAsr),
            task_type: TaskType::ASR,
            work_kind: "prefill".into(),
            compute_dtype: "f32".into(),
            kv_dtype: "state_v2".into(),
            cache_namespace: "test".into(),
            adapter: None,
        },
        batch_mode: NativeBatchMode::Static,
        max_batch_size: 2,
        estimate: ResourceVector::zero(),
        stage: Some(stage),
    };
    let mut auxiliary = base.clone();
    auxiliary.work = WorkUnit::SequenceStep {
        phase: SequencePhase::Prefill,
        input: InputRange::new(0, 1).unwrap(),
        max_output_steps: 1,
        auxiliary_state: Some(Arc::from([ClockedStateSpan::new(
            StateGroupId::new(2),
            StateClock::AudioSamples,
            InputRange::new(0, 3_200).unwrap(),
        )
        .unwrap()])),
    };

    let primary_lane = EngineCore::batch_lane(&base, WorkCost::new(1, 1, 0));
    let auxiliary_lane = EngineCore::batch_lane(&auxiliary, WorkCost::new(1, 1, 0));

    assert_eq!(primary_lane.shape_bucket, "padded.primary");
    assert_eq!(auxiliary_lane.shape_bucket, "padded.auxiliary");
    assert_ne!(primary_lane, auxiliary_lane);
}

#[test]
fn prepared_exact_shapes_split_static_physical_batches() {
    let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
        Mutex::new(Vec::new()),
    ))));
    let mut core = EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor)
        .expect("core");
    let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
    let mut profile =
        ExecutionProfile::fail_closed(BackendKind::Cpu, Some(variant), ExecutionMode::Atomic);
    profile.prefill_batch = NativeBatchMode::Static;
    profile.max_batch_size = 2;
    let mut stage = super::super::StageDescriptor::from_execution_profile(
        super::super::StageId::new(7),
        "tts.generate.tensor_static",
        &profile,
        NativeBatchMode::Static,
    );
    stage.selector = super::super::StageWorkSelector::Atomic;
    stage.shape_policy = StageShapePolicy::Exact;
    stage.max_padding_basis_points = 0;
    stage.max_work_units = 2;
    stage.max_workspace_bytes = 64;
    let binding = super::super::ExecutionAdapterBinding {
        execution_group_id: super::super::ExecutionGroupId::new(1),
        model_instance_id: super::super::ModelInstanceId::new(2),
        adapter_instance_id: super::super::AdapterInstanceId::new(3),
        adapter_abi_revision: super::super::AdapterAbiRevision::new(1),
        model_variant: variant,
        capability_id: "tts".to_string(),
        stages: Arc::from([stage.clone()]),
    };
    for (id, tensor_elements) in [("shape-a", 8), ("shape-b", 9)] {
        let mut request =
            EngineCoreRequest::tts("exact static shape").with_model_variant(variant);
        request.id = id.to_string();
        request.bind_execution_adapter(binding.clone()).unwrap();
        request
            .install_prepared_stage_cost(
                stage.id,
                WorkCost::new(1, tensor_elements, tensor_elements),
            )
            .unwrap();
        core.add_request(request).unwrap();
    }
    let scheduled = ["shape-a", "shape-b"]
        .into_iter()
        .map(|id| {
            let epoch = core.get_session_key(&id.to_string()).unwrap().epoch;
            scheduled_prefill(id, epoch)
        })
        .collect::<Vec<_>>();
    let adapter_key = binding.key_for_stage(stage.id).unwrap();
    for item in &scheduled {
        core.active_plans.insert(
            item.plan_id,
            ExecutionPlan {
                plan_id: item.plan_id,
                session: item.session_key(),
                work: WorkUnit::AtomicJob {
                    kind: "tts".to_string(),
                },
                batch_key: BatchKey {
                    backend: BackendKind::Cpu,
                    model_variant: Some(variant),
                    task_type: TaskType::TTS,
                    work_kind: "tts".to_string(),
                    compute_dtype: "f32".to_string(),
                    kv_dtype: "none".to_string(),
                    cache_namespace: "none".to_string(),
                    adapter: Some(adapter_key.clone()),
                },
                batch_mode: NativeBatchMode::Static,
                max_batch_size: 2,
                estimate: ResourceVector::zero(),
                stage: Some(stage.clone()),
            },
        );
    }
    let requests = scheduled
        .iter()
        .map(|item| core.requests.get(&item.request_id).unwrap().clone())
        .collect::<Vec<_>>();

    let mut deferred = Vec::new();
    let batches = core
        .form_physical_batches(
            ExecutionPhase::Prefill,
            &requests,
            &scheduled,
            usize::MAX,
            &mut Vec::new(),
            &mut deferred,
            &mut Vec::new(),
        )
        .unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(deferred.len(), 1);
    assert!(batches
        .iter()
        .all(|batch| batch.physical_batch().rows.len() == 1));
    assert_eq!(
        batches[0]
            .physical_batch()
            .workspace
            .workspace_bytes()
            .unwrap(),
        8
    );
    assert_eq!(deferred[0].plan_id, scheduled[1].plan_id);
}

#[test]
fn one_token_continuous_budget_admits_multiple_rows() {
    for under_pressure in [false, true] {
        let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
            Mutex::new(Vec::new()),
        ))));
        let mut core =
            EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor)
                .expect("core");
        for id in ["continuous-a", "continuous-b"] {
            let mut request = EngineCoreRequest::tts("continuous batch fixture");
            request.id = id.to_string();
            core.add_request(request).unwrap();
        }
        let scheduled = ["continuous-a", "continuous-b"]
            .into_iter()
            .map(|id| {
                let epoch = core.get_session_key(&id.to_string()).unwrap().epoch;
                scheduled_decode(id, epoch)
            })
            .collect::<Vec<_>>();
        let mut profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Sequence);
        profile.max_batch_size = 2;
        let mut stage = super::super::StageDescriptor::from_execution_profile(
            super::super::StageId::new(5),
            "chat.decode.tensor_continuous",
            &profile,
            NativeBatchMode::Continuous,
        );
        stage.max_work_units = 2;
        let adapter_key = super::super::AdapterBindingKey {
            execution_group_id: super::super::ExecutionGroupId::new(1),
            model_instance_id: super::super::ModelInstanceId::new(2),
            adapter_instance_id: super::super::AdapterInstanceId::new(3),
            adapter_abi_revision: super::super::AdapterAbiRevision::new(1),
            capability_id: "chat".to_string(),
            stage_id: stage.id,
        };
        for item in &scheduled {
            core.active_plans.insert(
                item.plan_id,
                ExecutionPlan {
                    plan_id: item.plan_id,
                    session: item.session_key(),
                    work: item.work.clone(),
                    batch_key: BatchKey {
                        backend: BackendKind::Cpu,
                        model_variant: None,
                        task_type: TaskType::Chat,
                        work_kind: "decode".to_string(),
                        compute_dtype: "f32".to_string(),
                        kv_dtype: "f32".to_string(),
                        cache_namespace: "continuous".to_string(),
                        adapter: Some(adapter_key.clone()),
                    },
                    batch_mode: NativeBatchMode::Continuous,
                    max_batch_size: 2,
                    estimate: ResourceVector::zero(),
                    stage: Some(stage.clone()),
                },
            );
        }
        let requests = scheduled
            .iter()
            .map(|item| core.requests.get(&item.request_id).unwrap().clone())
            .collect::<Vec<_>>();

        if under_pressure {
            for row in &scheduled {
                assert!(core.adapt_workspace_batch(row, 2));
            }
        }
        let batches = core
            .form_physical_batches(
                ExecutionPhase::Decode,
                &requests,
                &scheduled,
                usize::MAX,
                &mut Vec::new(),
                &mut Vec::new(),
                &mut Vec::new(),
            )
            .unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(
            batches[0].physical_batch().mode,
            NativeBatchMode::Continuous
        );
        assert_eq!(
            batches[0].physical_batch().rows.len(),
            if under_pressure { 1 } else { 2 }
        );
        assert_eq!(batches[0].physical_batch().budget.max_logical_units, 2);
    }
}

#[test]
fn prepared_continuous_cost_scales_an_isolated_multi_token_quantum() {
    let variant = ModelVariant::Qwen3508BGguf;
    let mut request = EngineCoreRequest::chat(vec![ChatMessage {
        role: ChatRole::User,
        content: "continuous cost fixture".to_string(),
    }])
    .with_model_variant(variant);
    let profile =
        ExecutionProfile::fail_closed(BackendKind::Cpu, Some(variant), ExecutionMode::Sequence);
    let mut stage = super::super::StageDescriptor::from_execution_profile(
        super::super::StageId::new(17),
        "chat.decode.tensor_continuous",
        &profile,
        NativeBatchMode::Continuous,
    );
    stage.max_workspace_bytes = 32;
    request
        .bind_execution_adapter(super::super::ExecutionAdapterBinding {
            execution_group_id: ExecutionGroupId::new(1),
            model_instance_id: ModelInstanceId::new(2),
            adapter_instance_id: AdapterInstanceId::new(3),
            adapter_abi_revision: AdapterAbiRevision::new(1),
            model_variant: variant,
            capability_id: "chat".to_string(),
            stages: Arc::from([stage.clone()]),
        })
        .unwrap();
    request
        .install_prepared_stage_cost(stage.id, WorkCost::new(1, 128, 32))
        .unwrap();
    let work = WorkUnit::SequenceStep {
        phase: SequencePhase::Decode,
        input: super::super::InputRange { start: 9, end: 13 },
        max_output_steps: 4,
        auxiliary_state: None,
    };

    let cost = EngineCore::work_cost(&request, &work, Some(&stage), BackendKind::Cpu).unwrap();
    assert_eq!(cost.logical_units, 4);
    assert_eq!(cost.tensor_elements, 512);
    assert_eq!(cost.workspace.workspace_bytes().unwrap(), 32);
}

#[test]
fn realtime_work_cost_accounts_for_bounded_input_and_output() {
    let request = EngineCoreRequest::asr("");
    let push = EngineCore::work_cost(
        &request,
        &WorkUnit::RealtimePush {
            operation_id: crate::engine::RealtimeOperationId::new(1),
            input: super::super::InputRange::new(40, 200).unwrap(),
            max_output_steps: 4,
            max_cache_append: 8,
        },
        None,
        BackendKind::Cpu,
    )
    .unwrap();
    assert_eq!(push, WorkCost::new(160, 160, 0));

    let finish = EngineCore::work_cost(
        &request,
        &WorkUnit::RealtimeFinish {
            operation_id: crate::engine::RealtimeOperationId::new(2),
            max_output_steps: 7,
            max_cache_append: 8,
        },
        None,
        BackendKind::Cpu,
    )
    .unwrap();
    assert_eq!(finish, WorkCost::new(8, 8, 0));
}

#[test]
fn realtime_preparation_uses_authenticated_operation_cost() {
    let operation_id = crate::engine::RealtimeOperationId::new(7);
    let mut request = EngineCoreRequest::asr("");
    request.enable_realtime_asr_ingress().unwrap();
    let exact = WorkCost::new(160, 704, 4096);
    let _waiter = request
        .install_realtime_asr_operation_with_cost(
            operation_id,
            RealtimeAsrOperationPayload::Push {
                samples: Arc::from(vec![0.0; 160]),
                sample_rate: 16_000,
            },
            exact,
        )
        .unwrap();

    let cost = EngineCore::work_cost(
        &request,
        &WorkUnit::RealtimePreparation {
            operation_id,
            mode: super::super::RealtimePreparationMode::Push,
            input: super::super::InputRange::new(0, 160).unwrap(),
            max_output_steps: 8,
            max_cache_append: 32,
            retained_state_input: super::super::InputRange::new(0, 1).unwrap(),
            auxiliary_state: None,
        },
        None,
        BackendKind::Cpu,
    )
    .unwrap();

    assert_eq!(cost, exact);
}

#[test]
fn multi_token_continuous_quantum_cannot_join_another_row() {
    let lane = BatchLaneKey {
        execution_group: ExecutionGroupId::new(1),
        model_instance: ModelInstanceId::new(2),
        adapter_instance: AdapterInstanceId::new(3),
        adapter_abi: AdapterAbiRevision::new(1),
        capability_id: "chat".to_string(),
        stage_id: StageId::new(4),
        backend: BackendKind::Cpu,
        device_ordinal: None,
        compute_dtype: "f32".to_string(),
        state_dtype: "f32".to_string(),
        tensor_layout: "ragged".to_string(),
        quantization: "none".to_string(),
        state_schema: "test.v1".to_string(),
        kernel_mode: "test".to_string(),
        semantic_mode: "greedy".to_string(),
        shape_bucket: "ragged".to_string(),
    };
    let first_scheduled = scheduled_decode("solo-quantum", 1);
    let first_row = ReadyQuantum {
        plan_id: first_scheduled.plan_id,
        session: first_scheduled.session_key(),
        lane: lane.clone(),
        work: WorkUnit::SequenceStep {
            phase: SequencePhase::Decode,
            input: super::super::InputRange { start: 1, end: 5 },
            max_output_steps: 4,
            auxiliary_state: None,
        },
        cost: WorkCost::new(4, 4, 0),
        managed_cache: None,
    };
    let first_request = Arc::new(EngineCoreRequest::tts("solo quantum"));
    let mut assembly = PhysicalBatchAssembly::new(
        PhysicalBatch {
            batch_id: BatchId::new(1),
            lane: lane.clone(),
            mode: NativeBatchMode::Continuous,
            budget: BatchBudget {
                max_rows: 2,
                max_logical_units: 5,
                max_tensor_elements: 5,
                max_workspace_bytes: 0,
                max_padding_basis_points: 0,
                max_formation_delay: Duration::ZERO,
            },
            rows: vec![first_row],
            materialized_tensor_elements: 4,
            workspace: ResourceVector::zero(),
        },
        first_request,
        first_scheduled,
        OutputVisibility::AfterQuantumCommit,
        StageShapePolicy::Ragged,
        0,
        PhysicalLaunchPolicy::ExecutionGroupExclusive,
    )
    .expect("one-row batch");
    let second_scheduled = scheduled_decode("shared-row", 2);
    let second_row = ReadyQuantum {
        plan_id: second_scheduled.plan_id,
        session: second_scheduled.session_key(),
        lane,
        work: second_scheduled.work.clone(),
        cost: WorkCost::new(1, 1, 0),
        managed_cache: None,
    };

    assert!(assembly
        .try_push(
            Arc::new(EngineCoreRequest::tts("shared row")),
            second_scheduled,
            second_row,
        )
        .is_err());
    assert_eq!(assembly.physical_batch.rows.len(), 1);
}

#[tokio::test]
async fn executor_progress_is_validated_before_scheduler_mutation() {
    let cleanup_calls = Arc::new(Mutex::new(Vec::new()));
    let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(cleanup_calls)));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            max_batch_size: 1,
            max_tokens_per_step: 8,
            ..Default::default()
        },
        executor,
    )
    .unwrap();
    let mut request = EngineCoreRequest::tts("transaction");
    request.id = "transaction".to_string();
    request.prompt_tokens = vec![1];
    core.add_request(request).unwrap();
    let scheduled = core.scheduler.schedule().prefill_requests.remove(0);
    core.begin_execution_plan(&scheduled).await.unwrap();

    let mut invalid = ExecutorStepResult::new(
        &scheduled,
        ExecutorOutput {
            request_id: scheduled.request_id.clone(),
            audio: None,
            text: None,
            input_transcription: None,
            tokens_processed: scheduled.num_tokens + 1,
            tokens_generated: 0,
            finished: false,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
            logprobs: Vec::new(),
        },
    );
    invalid.disposition = ExecutionDisposition::Progress;
    let output = core
        .commit_executor_result(invalid, 1.0)
        .await
        .expect("invalid result must emit a terminal output")
        .output;

    assert!(output.finished);
    assert!(output
        .error
        .as_deref()
        .is_some_and(|message| message.contains("beyond the scheduled quantum")));
    assert_eq!(
        core.scheduler.get_running_info(&scheduled.request_id),
        Some((0, 0)),
        "invalid progress must not reach scheduler accounting"
    );
}

fn retryable_step_result(
    scheduled: &ScheduledRequest,
    retry: RetryDisposition,
) -> ExecutorStepResult {
    ExecutorStepResult::from_session(
        scheduled,
        super::super::executor::ModelSessionResult {
            output: ExecutorOutput {
                request_id: scheduled.request_id.clone(),
                audio: None,
                text: None,
                input_transcription: None,
                tokens_processed: 0,
                tokens_generated: 0,
                finished: false,
                phase_timing_override: None,
                asr_diagnostics: None,
                error: Some("transient backend failure".to_string()),
                logprobs: Vec::new(),
            },
            disposition: ExecutionDisposition::Failed(ExecutionFailure {
                kind: super::super::execution::FailureKind::Backend,
                scope: super::super::execution::FailureScope::Row,
                retry,
                health: super::super::execution::HealthImpact::Degraded,
                message: "transient backend failure".to_string(),
            }),
            safe_point: true,
            provenance: OutcomeProvenance::failure(
                FailureOrigin::Model,
                DispatchState::Started,
            ),
            staged_stream_outputs: Vec::new(),
            managed_cache_completions: Vec::new(),
            managed_cache_append: None,
            realtime_stage_outcome: None,
            pending_quantum_required: false,
            clocked_state_completion: None,
        },
    )
}

fn workspace_rejection(scheduled: &ScheduledRequest, width: usize) -> ExecutorStepResult {
    let mut result = retryable_step_result(scheduled, RetryDisposition::RetrySameSession);
    if let ExecutionDisposition::Failed(failure) = &mut result.disposition {
        failure.kind = super::super::execution::FailureKind::ResourceExhausted;
        failure.scope = super::super::execution::FailureScope::PhysicalBatch;
        failure.health = super::super::execution::HealthImpact::None;
    }
    result.provenance = OutcomeProvenance::failure(
        FailureOrigin::WorkspaceAdmission,
        DispatchState::NotStarted,
    );
    result.dispatch = BatchDispatch::not_dispatched(width);
    result
}

#[tokio::test]
async fn workspace_backpressure_preserves_published_stream_and_shrinks_before_retry() {
    let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
        Mutex::new(Vec::new()),
    ))));
    let mut core =
        EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor).unwrap();
    core.retry_policy.execution_backoff_base = Duration::ZERO;
    core.retry_policy.execution_backoff_max = Duration::ZERO;
    let mut request = EngineCoreRequest::tts("workspace pressure");
    request.id = "workspace-published".into();
    request.prompt_tokens = vec![1];
    core.add_request(request).unwrap();
    core.refresh_scheduler_execution_profiles().await;
    let mut scheduled = core.scheduler.schedule().prefill_requests.remove(0);
    let session = scheduled.session_key();
    core.incremental_stream_sessions.insert(session.clone());
    core.stream_sequence_cursors.insert(session.clone(), 7);
    for (width, expected_cap) in [(4, 2), (2, 1)] {
        core.begin_execution_plan(&scheduled).await.unwrap();
        let result = workspace_rejection(&scheduled, 1);
        assert!(EngineCore::is_workspace_backpressure(&result));
        assert!(core.adapt_workspace_batch(&scheduled, width));
        let committed = core.commit_executor_result(result, 0.0).await;
        assert!(
            committed.is_none(),
            "unexpected terminal: {:?}",
            committed.map(|value| value.output.error)
        );
        assert_eq!(core.workspace_row_limits[&session], expected_cap);
        assert_eq!(core.stream_sequence_cursors[&session], 7);
        assert!(core.incremental_stream_sessions.contains(&session));
        assert_eq!(
            core.scheduler.get_running_info(&session.request_id),
            Some((0, 0))
        );
        scheduled = core.scheduler.schedule().prefill_requests.remove(0);
        assert_eq!(scheduled.session_key(), session);
    }
    // Once memory fits, the unchanged session can commit its next quantum.
    core.begin_execution_plan(&scheduled).await.unwrap();
    let mut output = workspace_rejection(&scheduled, 1).output;
    output.error = None;
    output.tokens_processed = 1;
    let result = ExecutorStepResult::new(&scheduled, output);
    let committed = core.commit_executor_result(result, 0.0).await.unwrap();
    assert!(committed.output.error.is_none());
    assert_eq!(core.stream_sequence_cursors[&session], 7);
}

#[tokio::test]
async fn workspace_rejection_honors_cancellation_before_retry() {
    let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
        Mutex::new(Vec::new()),
    ))));
    let mut core =
        EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor).unwrap();
    let cancellation = Arc::new(AtomicBool::new(false));
    let mut request = EngineCoreRequest::tts("cancel rejected workspace");
    request.id = "workspace-cancel".into();
    request.prompt_tokens = vec![1];
    request.cancellation = Some(cancellation.clone());
    core.add_request(request).unwrap();
    core.refresh_scheduler_execution_profiles().await;
    let scheduled = core.scheduler.schedule().prefill_requests.remove(0);
    core.begin_execution_plan(&scheduled).await.unwrap();
    core.incremental_stream_sessions
        .insert(scheduled.session_key());
    cancellation.store(true, Ordering::Release);
    let committed = core
        .commit_executor_result(workspace_rejection(&scheduled, 1), 0.0)
        .await
        .unwrap();
    assert!(
        matches!(
            committed.disposition,
            ExecutionDisposition::Finished(ExecutionFinishReason::Cancelled)
        ),
        "{:?}: {:?}",
        committed.disposition,
        committed.output.error
    );
    assert!(committed.output.error.is_none());
    assert!(!core
        .execution_retry_attempts
        .contains_key(&scheduled.session_key()));
}

#[test]
fn workspace_retry_budget_counts_only_failures_without_peer_progress() {
    let mut core = EngineCore::new(EngineCoreConfig::default()).unwrap();
    let session = super::super::SessionKey::new("waiting".into(), 1);
    core.observe_workspace_retry(&session);
    core.execution_retry_attempts.insert(session.clone(), 3);
    core.observe_workspace_retry(&session);
    assert_eq!(core.execution_retry_attempts[&session], 3);
    core.workspace_progress_epoch += 1;
    core.observe_workspace_retry(&session);
    assert!(!core.execution_retry_attempts.contains_key(&session));
    core.execution_retry_attempts.insert(session.clone(), 1);
    core.observe_workspace_retry(&session);
    assert_eq!(core.execution_retry_attempts[&session], 1);
    // Peer progress restarts the wall-clock wait too.
    core.workspace_waits
        .insert(session.clone(), WorkspaceWait::start());
    core.workspace_progress_epoch += 1;
    core.observe_workspace_retry(&session);
    assert!(!core.workspace_waits.contains_key(&session));
    core.workspace_waits
        .insert(session.clone(), WorkspaceWait::start());
    core.clear_exact_execution_state(&session);
    assert!(!core.workspace_retry_progress.contains_key(&session));
    assert!(!core.workspace_waits.contains_key(&session));
}

#[test]
fn workspace_retry_requires_unchanged_pre_device_state() {
    let scheduled = ScheduledRequest {
        plan_id: 1,
        request_id: "fence".into(),
        sequence_id: 1,
        num_tokens: 1,
        is_prefill: true,
        num_computed_tokens: 0,
        work: WorkUnit::SequenceStep {
            phase: SequencePhase::Prefill,
            input: InputRange { start: 0, end: 1 },
            max_output_steps: 1,
            auxiliary_state: None,
        },
    };
    let mut result = workspace_rejection(&scheduled, 4);
    assert!(EngineCore::is_workspace_backpressure(&result));
    result.provenance.dispatch_state = DispatchState::Started;
    assert!(!EngineCore::is_workspace_backpressure(&result));
    result = workspace_rejection(&scheduled, 4);
    result.output.tokens_generated = 1;
    assert!(!EngineCore::is_workspace_backpressure(&result));
    result = workspace_rejection(&scheduled, 4);
    result.pending_quantum_required = true;
    assert!(!EngineCore::is_workspace_backpressure(&result));
    result = workspace_rejection(&scheduled, 4);
    result.output.text = Some("unexpected output".into());
    assert!(!EngineCore::is_workspace_backpressure(&result));
}

#[tokio::test]
async fn same_session_retry_releases_quantum_without_committing_progress() {
    let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
        Mutex::new(Vec::new()),
    ))));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            max_batch_size: 1,
            max_tokens_per_step: 8,
            ..Default::default()
        },
        executor,
    )
    .unwrap();
    core.retry_policy.execution_backoff_base = Duration::ZERO;
    core.retry_policy.execution_backoff_max = Duration::ZERO;
    let mut request = EngineCoreRequest::tts("retry");
    request.id = "same-session-retry".to_string();
    request.prompt_tokens = vec![1];
    core.add_request(request).unwrap();
    core.refresh_scheduler_execution_profiles().await;
    let scheduled = core.scheduler.schedule().prefill_requests.remove(0);
    let session = scheduled.session_key();
    core.begin_execution_plan(&scheduled).await.unwrap();

    let emitted = core
        .commit_executor_result(
            retryable_step_result(&scheduled, RetryDisposition::RetrySameSession),
            1.0,
        )
        .await;
    assert!(
        emitted.is_none(),
        "a retry must not terminalize the request"
    );
    assert_eq!(
        core.scheduler.get_running_info(&scheduled.request_id),
        Some((0, 0))
    );

    let mut retry_schedule = core.scheduler.schedule();
    assert_eq!(
        retry_schedule.prefill_requests.len(),
        1,
        "same-session retry was not rescheduled: {retry_schedule:?}"
    );
    let retry = retry_schedule.prefill_requests.remove(0);
    assert_eq!(retry.session_key(), session);
    assert_ne!(retry.plan_id, scheduled.plan_id);
}

#[tokio::test]
async fn committed_stream_progress_disables_all_executor_retries() {
    for (suffix, retry) in [
        ("same-session", RetryDisposition::RetrySameSession),
        ("recompute", RetryDisposition::Recompute),
    ] {
        let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
            Mutex::new(Vec::new()),
        ))));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_batch_size: 1,
                max_tokens_per_step: 8,
                ..Default::default()
            },
            executor,
        )
        .unwrap();
        core.retry_policy.execution_backoff_base = Duration::ZERO;
        core.retry_policy.execution_backoff_max = Duration::ZERO;
        let mut request = EngineCoreRequest::tts("visible retry");
        request.id = format!("visible-retry-{suffix}");
        request.prompt_tokens = vec![1];
        core.add_request(request).unwrap();
        core.refresh_scheduler_execution_profiles().await;
        let scheduled = core.scheduler.schedule().prefill_requests.remove(0);
        let session = scheduled.session_key();
        core.begin_execution_plan(&scheduled).await.unwrap();
        core.incremental_stream_sessions.insert(session.clone());

        let committed = core
            .commit_executor_result(retryable_step_result(&scheduled, retry), 1.0)
            .await
            .expect("visible progress must turn a retry into a terminal failure");
        assert_eq!(committed.session, session);
        assert!(committed
            .output
            .error
            .as_deref()
            .is_some_and(|message| message.contains("retry is unsafe")));
        assert!(matches!(
            committed.disposition,
            ExecutionDisposition::Failed(ExecutionFailure {
                retry: RetryDisposition::Never,
                ..
            })
        ));
        let retry_schedule = core.scheduler.schedule();
        assert!(retry_schedule.prefill_requests.is_empty());
        assert!(retry_schedule.decode_requests.is_empty());
    }
}

#[tokio::test]
async fn staged_pcm_commit_fences_recompute_and_restart_without_replaying_audio() {
    for restart in [false, true] {
        let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
            Mutex::new(Vec::new()),
        ))));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_batch_size: 1,
                max_tokens_per_step: 8,
                ..Default::default()
            },
            executor,
        )
        .unwrap();
        let mut request = EngineCoreRequest::tts("committed PCM");
        request.id = format!("staged-pcm-retry-{restart}");
        request.prompt_tokens = vec![1];
        request.streaming = true;
        let (tx, _rx) = mpsc::channel(8);
        request.streaming_tx = Some(tx);
        core.add_request(request).unwrap();
        core.refresh_scheduler_execution_profiles().await;
        let prefill = core.scheduler.schedule().prefill_requests.remove(0);
        let session = prefill.session_key();
        core.begin_execution_plan(&prefill).await.unwrap();
        let mut result = ExecutorStepResult::new(
            &prefill,
            MockExecutor::build_outputs(std::slice::from_ref(&prefill)).remove(0),
        );
        result
            .staged_stream_outputs
            .push(super::super::StreamingOutput {
                request_id: session.request_id.clone(),
                sequence: 0,
                samples: vec![0.25, 0.5],
                sample_rate: 44_100,
                is_final: false,
                text: None,
                stats: None,
                asr_progress: None,
                logprobs: Vec::new(),
            });
        let committed = core.commit_executor_result(result, 1.0).await.unwrap();
        assert_eq!(committed.staged_stream_outputs.len(), 1);
        assert!(core.incremental_stream_sessions.contains(&session));
        assert_eq!(core.stream_sequence_cursors.get(&session), Some(&1));

        let decode = core.scheduler.schedule().decode_requests.remove(0);
        core.begin_execution_plan(&decode).await.unwrap();
        let result = if restart {
            ExecutorStepResult::from_session(
                &decode,
                super::super::executor::ModelSessionResult::restart_sequence(
                    session.request_id.clone(),
                    super::super::SequenceRestartReason::ModelFallback,
                ),
            )
        } else {
            retryable_step_result(&decode, RetryDisposition::Recompute)
        };
        let rejected = core.commit_executor_result(result, 1.0).await.unwrap();
        assert!(matches!(
            rejected.disposition,
            ExecutionDisposition::Failed(ExecutionFailure {
                retry: RetryDisposition::Never,
                ..
            })
        ));
        assert!(rejected.staged_stream_outputs.is_empty());
        assert!(rejected
            .output
            .error
            .as_deref()
            .is_some_and(|message| message.contains("unsafe")));
        // Result commitment authenticates the failure; the step's output
        // processing then begins terminal cleanup before another schedule.
        let cause = EngineCore::terminal_release_cause(&rejected.disposition).unwrap();
        core.begin_terminal_release(&session, cause).await;
        let schedule = core.scheduler.schedule();
        assert!(schedule.prefill_requests.is_empty());
        assert!(schedule.decode_requests.is_empty());
    }
}

async fn retry_budget_test_core() -> (EngineCore, ScheduledRequest) {
    let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
        Mutex::new(Vec::new()),
    ))));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            max_batch_size: 1,
            max_tokens_per_step: 8,
            ..Default::default()
        },
        executor,
    )
    .unwrap();
    core.retry_policy.max_execution_retries = 2;
    core.retry_policy.execution_backoff_base = Duration::ZERO;
    core.retry_policy.execution_backoff_max = Duration::ZERO;
    let mut request = EngineCoreRequest::tts("retry budget");
    request.id = "retry-budget".to_string();
    request.prompt_tokens = vec![1];
    core.add_request(request).unwrap();
    core.refresh_scheduler_execution_profiles().await;
    let scheduled = core.scheduler.schedule().prefill_requests.remove(0);
    (core, scheduled)
}

fn assert_retry_budget_terminal(committed: Option<CommittedExecutorOutput>, expected: &str) {
    let committed = committed.expect("retry budget must terminalize");
    assert!(
        committed
            .output
            .error
            .as_deref()
            .is_some_and(|message| message.contains(expected)),
        "{:?}",
        committed.output.error
    );
    assert!(matches!(
        committed.disposition,
        ExecutionDisposition::Failed(ExecutionFailure {
            retry: RetryDisposition::Never,
            ..
        })
    ));
}

#[tokio::test]
async fn retry_budget_exhaustion_terminalizes_the_exact_session() {
    let (mut core, mut scheduled) = retry_budget_test_core().await;
    let session = scheduled.session_key();
    for attempt in 1..=3 {
        core.begin_execution_plan(&scheduled).await.unwrap();
        let committed = core
            .commit_executor_result(
                retryable_step_result(&scheduled, RetryDisposition::RetrySameSession),
                1.0,
            )
            .await;
        if attempt <= 2 {
            assert!(committed.is_none());
            scheduled = core.scheduler.schedule().prefill_requests.remove(0);
            assert_eq!(scheduled.session_key(), session);
        } else {
            assert_eq!(committed.as_ref().map(|c| &c.session), Some(&session));
            assert_retry_budget_terminal(committed, "retry budget exhausted after 2 attempts");
        }
    }
}

/// The H100 field failure (b1d51b89): a running stream's decode workspace
/// was refused 25 MB short, retried 5, 10 and 20 ms later against the same
/// cached capacity sample, and failed after three attempts in ~35 ms. A
/// refused workspace now waits on a wall-clock budget instead.
#[tokio::test]
async fn refused_workspace_waits_on_a_wall_clock_budget_not_the_attempt_budget() {
    let (mut core, mut scheduled) = retry_budget_test_core().await;
    let session = scheduled.session_key();
    core.incremental_stream_sessions.insert(session.clone());
    let before = crate::engine::engine_physical_execution_metrics_snapshot().workspace_waits;
    // Well past max_execution_retries (2): every refusal defers.
    for attempt in 1..=6 {
        core.begin_execution_plan(&scheduled).await.unwrap();
        let committed = core
            .commit_executor_result(workspace_rejection(&scheduled, 1), 1.0)
            .await;
        assert!(
            committed.is_none(),
            "refusal {attempt} failed the stream: {:?}",
            committed.map(|c| c.output.error)
        );
        assert_eq!(core.workspace_waits[&session].attempts, attempt);
        assert!(!core.execution_retry_attempts.contains_key(&session));
        scheduled = core.scheduler.schedule().prefill_requests.remove(0);
        assert_eq!(scheduled.session_key(), session);
    }
    // Once the wait has lasted its budget, the next refusal fails it.
    core.workspace_waits.get_mut(&session).unwrap().started =
        Instant::now() - core.retry_policy.workspace_wait_budget;
    core.begin_execution_plan(&scheduled).await.unwrap();
    let committed = core
        .commit_executor_result(workspace_rejection(&scheduled, 1), 1.0)
        .await;
    assert_retry_budget_terminal(
        committed,
        "retry budget exhausted after 7 attempts over 5.0s of workspace backpressure",
    );
    assert!(!core.workspace_waits.contains_key(&session));
    let after = crate::engine::engine_physical_execution_metrics_snapshot().workspace_waits;
    assert!(after.retried >= before.retried + 6);
    assert!(after.exhausted > before.exhausted);
}

#[tokio::test]
async fn admitted_workspace_ends_the_wait_and_keeps_the_attempt_budget() {
    let (mut core, mut scheduled) = retry_budget_test_core().await;
    let session = scheduled.session_key();
    let before = crate::engine::engine_physical_execution_metrics_snapshot().workspace_waits;
    for _ in 0..4 {
        core.begin_execution_plan(&scheduled).await.unwrap();
        assert!(core
            .commit_executor_result(workspace_rejection(&scheduled, 1), 1.0)
            .await
            .is_none());
        scheduled = core.scheduler.schedule().prefill_requests.remove(0);
    }
    core.begin_execution_plan(&scheduled).await.unwrap();
    let mut output = workspace_rejection(&scheduled, 1).output;
    output.error = None;
    output.tokens_processed = 1;
    let committed = core
        .commit_executor_result(ExecutorStepResult::new(&scheduled, output), 1.0)
        .await;
    assert!(committed.is_none_or(|c| c.output.error.is_none()));
    assert!(!core.workspace_waits.contains_key(&session));
    let after = crate::engine::engine_physical_execution_metrics_snapshot().workspace_waits;
    assert!(after.recovered > before.recovered);
    // The refusals did not spend the attempt budget of other failures.
    scheduled = core
        .scheduler
        .schedule()
        .prefill_requests
        .pop()
        .or_else(|| core.scheduler.schedule().decode_requests.pop())
        .expect("the session continues");
    core.begin_execution_plan(&scheduled).await.unwrap();
    assert!(core
        .commit_executor_result(
            retryable_step_result(&scheduled, RetryDisposition::RetrySameSession),
            1.0,
        )
        .await
        .is_none());
}

#[tokio::test]
async fn recompute_retry_releases_capability_authoritative_session_state() {
    let cleanup_calls = Arc::new(Mutex::new(Vec::new()));
    let executor =
        UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(cleanup_calls.clone())));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            max_batch_size: 1,
            max_tokens_per_step: 8,
            ..Default::default()
        },
        executor,
    )
    .unwrap();
    core.retry_policy.execution_backoff_base = Duration::ZERO;
    core.retry_policy.execution_backoff_max = Duration::ZERO;
    let mut request = EngineCoreRequest::tts("recompute");
    request.id = "recompute-retry".to_string();
    request.prompt_tokens = vec![1];
    core.add_request(request).unwrap();
    core.refresh_scheduler_execution_profiles().await;

    let prefill = core.scheduler.schedule().prefill_requests.remove(0);
    let session = prefill.session_key();
    core.begin_execution_plan(&prefill).await.unwrap();
    core.commit_executor_result(
        ExecutorStepResult::new(
            &prefill,
            MockExecutor::build_outputs(std::slice::from_ref(&prefill)).remove(0),
        ),
        1.0,
    )
    .await
    .expect("prefill progress must be emitted");

    let decode = core.scheduler.schedule().decode_requests.remove(0);
    core.begin_execution_plan(&decode).await.unwrap();
    let emitted = core
        .commit_executor_result(
            retryable_step_result(&decode, RetryDisposition::Recompute),
            1.0,
        )
        .await;
    assert!(
        emitted.is_none(),
        "recompute must not terminalize the request"
    );
    assert!(cleanup_calls
        .lock()
        .unwrap()
        .iter()
        .any(|request_id| request_id == "recompute-retry"));
    assert_eq!(
        core.scheduler.get_running_info(&decode.request_id),
        Some((0, 0))
    );
    assert!(!core.execution_trackers.contains_key(&decode.request_id));

    let retry = core.scheduler.schedule().prefill_requests.remove(0);
    assert_eq!(retry.session_key(), session);
    assert_ne!(retry.plan_id, decode.plan_id);
}

#[tokio::test]
async fn full_prefill_plan_covers_the_model_owned_prompt_operation() {
    let cleanup_calls = Arc::new(Mutex::new(Vec::new()));
    let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(cleanup_calls)));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            max_batch_size: 1,
            max_tokens_per_step: 4,
            enable_chunked_prefill: true,
            ..Default::default()
        },
        executor,
    )
    .unwrap();
    let mut request = EngineCoreRequest::tts("full prefill");
    request.id = "full-prefill".to_string();
    request.prompt_tokens = (0..16).collect();
    core.add_request(request).unwrap();
    core.refresh_scheduler_execution_profiles().await;
    let scheduled = core.scheduler.schedule().prefill_requests.remove(0);
    assert_eq!(scheduled.num_computed_tokens, 0);
    assert_eq!(scheduled.num_tokens, 16);
    let mut partial = scheduled.clone();
    partial.num_tokens = 4;
    assert!(core.begin_execution_plan(&partial).await.is_err());
    core.begin_execution_plan(&scheduled).await.unwrap();
    let plan = core.active_plans.get(&scheduled.plan_id).unwrap();
    assert!(matches!(
        plan.work,
        WorkUnit::SequenceStep {
            input: super::super::InputRange { start: 0, end: 16 },
            max_output_steps: 1,
            ..
        }
    ));

    let result = ExecutorStepResult::new(
        &scheduled,
        ExecutorOutput {
            request_id: scheduled.request_id.clone(),
            audio: None,
            text: None,
            input_transcription: None,
            tokens_processed: 16,
            tokens_generated: 1,
            finished: false,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
            logprobs: Vec::new(),
        },
    );
    let output = core
        .commit_executor_result(result, 1.0)
        .await
        .expect("valid progress must emit an output")
        .output;
    assert!(output.error.is_none());
    assert_eq!(
        core.scheduler.get_running_info(&scheduled.request_id),
        Some((16, 1))
    );
}

#[tokio::test]
async fn cache_plan_estimate_and_executor_observation_remain_distinct() {
    let executor = UnifiedExecutor::new_for_test(Box::new(DeadlineCleanupExecutor::new(
        Arc::new(Mutex::new(Vec::new())),
        true,
    )));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            max_batch_size: 1,
            block_size: 1,
            max_blocks: 8,
            ..Default::default()
        },
        executor,
    )
    .unwrap();
    let mut request = EngineCoreRequest::tts("cache contract");
    request.id = "cache-contract".to_string();
    request.prompt_tokens = vec![1, 2];
    core.add_request(request).unwrap();
    core.refresh_scheduler_execution_profiles().await;
    let scheduled = core.scheduler.schedule().prefill_requests.remove(0);
    core.begin_execution_plan(&scheduled).await.unwrap();
    assert_eq!(
        core.active_plans[&scheduled.plan_id].estimate.kv_bytes,
        ResourceAmount::Known(0),
        "opaque executor caches must not inherit fabricated scheduler KV geometry"
    );

    let observed = ResourceVector {
        kv_bytes: ResourceAmount::Known(123),
        ..ResourceVector::zero()
    };
    let result = ExecutorStepResult::new(
        &scheduled,
        ExecutorOutput {
            request_id: scheduled.request_id.clone(),
            audio: None,
            text: None,
            input_transcription: None,
            tokens_processed: scheduled.num_tokens,
            tokens_generated: 0,
            finished: false,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
            logprobs: Vec::new(),
        },
    )
    .with_observed_resources(observed);
    let report = EngineCore::report_from_result(&result);
    assert_eq!(report.observed_resources, observed);
}

#[tokio::test]
async fn valid_executor_progress_commits_exactly_once() {
    let cleanup_calls = Arc::new(Mutex::new(Vec::new()));
    let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(cleanup_calls)));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            max_batch_size: 1,
            max_tokens_per_step: 8,
            ..Default::default()
        },
        executor,
    )
    .unwrap();
    let mut request = EngineCoreRequest::tts("transaction");
    request.id = "valid-transaction".to_string();
    request.prompt_tokens = vec![1];
    core.add_request(request).unwrap();
    let scheduled = core.scheduler.schedule().prefill_requests.remove(0);
    core.begin_execution_plan(&scheduled).await.unwrap();
    let result = ExecutorStepResult::new(
        &scheduled,
        MockExecutor::build_outputs(std::slice::from_ref(&scheduled)).remove(0),
    );

    let output = core
        .commit_executor_result(result.clone(), 1.0)
        .await
        .expect("valid progress must emit an output")
        .output;
    assert!(output.error.is_none());
    assert_eq!(
        core.scheduler.get_running_info(&scheduled.request_id),
        Some((scheduled.num_tokens, 0))
    );

    assert!(core.commit_executor_result(result, 1.0).await.is_none());
    assert_eq!(
        core.scheduler.get_running_info(&scheduled.request_id),
        Some((scheduled.num_tokens, 0)),
        "duplicate results must not advance progress twice"
    );
}

#[tokio::test]
async fn unresolved_executor_is_planned_as_atomic_and_must_finish() {
    let executor = UnifiedExecutor::new_for_test(Box::new(ImmediateFinishExecutor));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            max_batch_size: 1,
            ..Default::default()
        },
        executor,
    )
    .unwrap();
    let mut request = EngineCoreRequest::tts("atomic");
    request.id = "atomic".to_string();
    request.prompt_tokens = vec![1];
    core.add_request(request).unwrap();
    let scheduled = core.scheduler.schedule().prefill_requests.remove(0);
    core.begin_execution_plan(&scheduled).await.unwrap();
    assert!(matches!(
        core.active_plans
            .get(&scheduled.plan_id)
            .map(|plan| &plan.work),
        Some(WorkUnit::AtomicJob { .. })
    ));

    let result = ExecutorStepResult::new(
        &scheduled,
        ExecutorOutput {
            request_id: scheduled.request_id.clone(),
            audio: None,
            text: None,
            input_transcription: None,
            tokens_processed: 1,
            tokens_generated: 0,
            finished: false,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
            logprobs: Vec::new(),
        },
    );
    let output = core
        .commit_executor_result(result, 1.0)
        .await
        .expect("invalid atomic result must emit an error")
        .output;
    assert!(output
        .error
        .as_deref()
        .is_some_and(|message| message.contains("atomic execution must finish")));
    assert_eq!(
        core.scheduler.get_running_info(&scheduled.request_id),
        Some((0, 0))
    );
}

#[tokio::test]
async fn executor_subbatch_failure_is_isolated_and_reconciled() {
    let executor = UnifiedExecutor::new_for_test(Box::new(SelectiveFailureExecutor));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            max_batch_size: 2,
            ..Default::default()
        },
        executor,
    )
    .unwrap();

    let mut bad = EngineCoreRequest::tts("bad");
    bad.id = "bad".to_string();
    bad.prompt_tokens = vec![1];
    bad.model_variant = Some(ModelVariant::Qwen354BGguf);
    let mut good = EngineCoreRequest::tts("good");
    good.id = "good".to_string();
    good.prompt_tokens = vec![1];
    good.model_variant = Some(ModelVariant::Qwen359BGguf);
    core.add_request(bad).unwrap();
    core.add_request(good).unwrap();

    let mut outputs = core.step().await.unwrap();
    assert_eq!(outputs.len(), 1);
    outputs.extend(core.step().await.unwrap());

    assert_eq!(outputs.len(), 2);
    assert!(outputs
        .iter()
        .any(|output| output.request_id == "bad" && output.error.is_some()));
    assert!(outputs
        .iter()
        .any(|output| output.request_id == "good" && output.error.is_none()));
    assert!(!core.has_request(&"bad".to_string()));
    assert!(!core.has_request(&"good".to_string()));
}

#[tokio::test]
async fn hard_deadline_returns_one_terminal_output_with_original_sequence() {
    let metrics_before = crate::engine::engine_batch_metrics_snapshot();
    let cleanup_calls = Arc::new(Mutex::new(Vec::new()));
    let executor =
        UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(cleanup_calls.clone())));
    let mut core =
        EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor).unwrap();
    let mut request = EngineCoreRequest::tts("expired")
        .with_deadline(Some(Instant::now() - std::time::Duration::from_millis(1)));
    request.id = "expired".to_string();
    core.add_request(request).unwrap();

    let outputs = core.step().await.unwrap();

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].request_id, "expired");
    assert_eq!(outputs[0].sequence_id, 0);
    assert!(outputs[0].is_finished);
    assert_eq!(outputs[0].num_tokens, 0);
    assert_eq!(outputs[0].error.as_deref(), Some(REQUEST_DEADLINE_EXCEEDED));
    assert_eq!(
        outputs[0].provenance,
        OutcomeProvenance::deadline(DeadlinePhase::SchedulerQueue, DispatchState::NotStarted,)
    );
    let metrics_after = crate::engine::engine_batch_metrics_snapshot();
    assert!(
        metrics_after.dispatch_states.not_started > metrics_before.dispatch_states.not_started
    );
    assert!(
        metrics_after.deadline_phases.scheduler_queue
            > metrics_before.deadline_phases.scheduler_queue
    );
    assert!(!core.has_request(&"expired".to_string()));
    assert_eq!(
        cleanup_calls
            .lock()
            .unwrap()
            .iter()
            .filter(|id| id.as_str() == "expired")
            .count(),
        1,
        "deadline cleanup must occur exactly once"
    );
}

#[tokio::test]
async fn deadline_cleanup_precedes_newly_scheduled_execution() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let executor = UnifiedExecutor::new_for_test(Box::new(DeadlineCleanupExecutor::new(
        events.clone(),
        true,
    )));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            max_batch_size: 1,
            max_tokens_per_step: 1,
            min_tokens_per_step: 1,
            block_size: 1,
            max_blocks: 2,
            enable_chunked_prefill: false,
            enable_adaptive_batching: false,
            backend: BackendKind::Cpu,
            ..Default::default()
        },
        executor,
    )
    .unwrap();

    let mut expired = EngineCoreRequest::tts("expires after prefill");
    expired.id = "expired-running".to_string();
    expired.prompt_tokens = vec![1];
    core.add_request(expired).unwrap();
    core.step().await.unwrap();
    assert!(core.scheduler.set_hard_deadline_for_test(
        &"expired-running".to_string(),
        Instant::now() - std::time::Duration::from_millis(1),
    ));

    let mut next = EngineCoreRequest::tts("new work");
    next.id = "new-work".to_string();
    next.prompt_tokens = vec![2];
    core.add_request(next).unwrap();
    core.step().await.unwrap();

    let events = events.lock().unwrap();
    let cleanup_index = events
        .iter()
        .position(|event| event.starts_with("cleanup:expired-running:"))
        .expect("expired session cleanup was not requested");
    let execution_index = events
        .iter()
        .position(|event| event == "execute-prefill:new-work")
        .expect("new work was not executed");
    assert!(
        cleanup_index < execution_index,
        "new work executed before expired cache cleanup: {events:?}"
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| event.starts_with("cleanup:expired-running:"))
            .count(),
        1,
        "expired cleanup must not be repeated during terminal processing"
    );
}

#[tokio::test]
async fn unconfirmed_deadline_cleanup_fences_id_without_blocking_unrelated_work() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let executor = UnifiedExecutor::new_for_test(Box::new(DeadlineCleanupExecutor::new(
        events.clone(),
        false,
    )));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            max_batch_size: 1,
            max_tokens_per_step: 1,
            min_tokens_per_step: 1,
            block_size: 1,
            max_blocks: 1,
            enable_chunked_prefill: false,
            enable_prefix_caching: false,
            enable_adaptive_batching: false,
            backend: BackendKind::Cpu,
            ..Default::default()
        },
        executor,
    )
    .unwrap();

    let mut expired = EngineCoreRequest::tts("expires with model-owned state");
    expired.id = "expired-running".to_string();
    expired.prompt_tokens = vec![1];
    core.add_request(expired).unwrap();
    core.step().await.unwrap();
    assert!(core.scheduler.set_hard_deadline_for_test(
        &"expired-running".to_string(),
        Instant::now() - std::time::Duration::from_millis(1),
    ));

    let mut waiting = EngineCoreRequest::tts("independent work");
    waiting.id = "waiting".to_string();
    waiting.prompt_tokens = vec![1];
    core.add_request(waiting).unwrap();
    let outputs = core.step().await.unwrap();

    assert!(outputs.iter().any(|output| {
        output.request_id == "expired-running"
            && output.error.as_deref() == Some(REQUEST_DEADLINE_EXCEEDED)
    }));
    assert!(events
        .lock()
        .unwrap()
        .iter()
        .any(|event| event == "execute-prefill:waiting"));

    let mut reused = EngineCoreRequest::tts("must remain fenced");
    reused.id = "expired-running".to_string();
    reused.prompt_tokens = vec![3];
    assert!(core.add_request(reused).is_err());
}

#[tokio::test]
async fn test_step_records_exact_prompt_tokens_and_ttft() {
    let executor = UnifiedExecutor::new_for_test(Box::new(ImmediateFinishExecutor));
    let config = EngineCoreConfig {
        enable_chunked_prefill: false,
        block_size: 1,
        max_blocks: 8,
        ..Default::default()
    };
    let mut core = EngineCore::new_with_unified_executor(config, executor).unwrap();

    let mut request = EngineCoreRequest::chat(vec![ChatMessage {
        role: ChatRole::User,
        content: "Hello".to_string(),
    }]);
    request.id = "chat-req".to_string();
    request.model_variant = Some(ModelVariant::Qwen306B);
    request
        .install_chat_execution_preparation(
            ModelVariant::Qwen306B,
            vec![11, 22, 33, 44],
            None,
            4096,
        )
        .unwrap();

    core.add_request(request).unwrap();
    let outputs = core.step().await.unwrap();

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].token_stats.prompt_tokens, 4);
    let latency = outputs[0]
        .latency_breakdown
        .as_ref()
        .expect("latency breakdown");
    assert!(latency.ttft_ms.is_some());
    assert!(latency.ttft_ms.unwrap() >= 0.0);
}

#[tokio::test]
async fn managed_kv_transaction_follows_live_prepare_commit_and_cancel_lifecycle() {
    let model_instance = ModelInstanceId::new(91);
    let executor = UnifiedExecutor::new_for_test(Box::new(ManagedReceiptExecutor));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            enable_chunked_prefill: false,
            block_size: 16,
            max_blocks: 4,
            backend: BackendKind::Cpu,
            ..Default::default()
        },
        executor,
    )
    .expect("core");
    let mut request = EngineCoreRequest::chat(vec![ChatMessage {
        role: ChatRole::User,
        content: "managed".to_string(),
    }])
    .with_model_variant(ModelVariant::Qwen306B);
    request.id = "managed-core".to_string();
    request.params.max_tokens = 1;
    request
        .install_chat_execution_preparation(
            ModelVariant::Qwen306B,
            vec![11, 12, 13, 14],
            None,
            4096,
        )
        .expect("prepare prompt");
    request
        .bind_model_instance(model_instance)
        .expect("model instance");
    bind_managed_test_state(&mut core, &mut request, model_instance);
    core.add_request(request).expect("add request");
    let session = core
        .get_session_key(&"managed-core".to_string())
        .expect("session");

    core.step().await.expect("managed step");
    let snapshot = core
        .managed_kv_cache
        .snapshot(model_instance, &session, crate::kv::CacheDomainId::new(1))
        .expect("committed table");
    assert_eq!(snapshot.committed_tokens, 4);
    assert_eq!(snapshot.version, 1);

    assert!(core.abort_request_session(&session).await);
    assert!(core
        .managed_kv_cache
        .snapshot(model_instance, &session, crate::kv::CacheDomainId::new(1))
        .is_none());
    let released = core.managed_kv_cache.runtime_snapshot();
    assert_eq!(released.totals.registered_sessions, 0);
    assert_eq!(released.totals.coordinator.admission_claimed_pages, 0);
    assert_eq!(released.totals.coordinator.active_transactions, 0);
}

fn restart_test_core(cancel_during_dispatch: bool) -> (EngineCore, Arc<AtomicUsize>) {
    let cleanup_calls = Arc::new(AtomicUsize::new(0));
    let executor = UnifiedExecutor::new_for_test(Box::new(RestartSequenceExecutor {
        cancel_during_dispatch,
        cleanup_calls: cleanup_calls.clone(),
    }));
    let core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            enable_chunked_prefill: false,
            block_size: 16,
            max_blocks: 4,
            max_batch_size: 1,
            backend: BackendKind::Cpu,
            ..Default::default()
        },
        executor,
    )
    .expect("restart test core");
    (core, cleanup_calls)
}

#[tokio::test]
async fn managed_sequence_restart_preserves_epoch_and_requeues_context_zero() {
    let model = ModelInstanceId::new(915);
    let (mut core, cleanup_calls) = restart_test_core(false);
    let request = managed_test_request(&mut core, model, "managed-restart", vec![1, 2, 3, 4]);
    let deadline = Instant::now() + Duration::from_secs(30);
    let request = request.with_deadline(Some(deadline));
    core.add_request(request).unwrap();
    let session = core
        .get_session_key(&"managed-restart".to_string())
        .unwrap();
    core.execution_retry_attempts.insert(session.clone(), 3);
    let first = core.prepare_step().await.unwrap().expect("initial prefill");
    let claim_before = core
        .managed_kv_cache
        .runtime_snapshot()
        .totals
        .coordinator
        .admission_claimed_pages;
    assert!(claim_before > 0);
    let executed = core.execute_prepared_with_progress(first).await.unwrap();
    let committed = core.commit_step(executed).await.unwrap();
    assert!(committed.outputs.is_empty());
    assert_eq!(cleanup_calls.load(Ordering::Acquire), 0);
    assert!(!core.execution_retry_attempts.contains_key(&session));
    assert_eq!(
        core.get_session_key(&"managed-restart".to_string()),
        Some(session.clone())
    );
    assert_eq!(core.requests["managed-restart"].deadline, Some(deadline));
    assert_eq!(
        core.execution_trackers["managed-restart"].state(),
        ExecutionState::Queued
    );
    assert_eq!(
        core.scheduler.get_running_info(&session.request_id),
        Some((0, 0))
    );
    let reset = core
        .managed_kv_cache
        .snapshot(model, &session, crate::kv::CacheDomainId::new(1))
        .unwrap();
    assert_eq!(reset.committed_tokens, 0);
    assert!(reset.groups.is_empty());
    assert_eq!(
        core.managed_kv_cache
            .runtime_snapshot()
            .totals
            .coordinator
            .admission_claimed_pages,
        claim_before
    );

    let prepared = core
        .prepare_step()
        .await
        .unwrap()
        .expect("restarted prefill");
    assert_eq!(
        core.get_session_key(&session.request_id),
        Some(session.clone()),
        "semantic restart must preserve the scheduler epoch"
    );
    let plan = core.active_plans.values().next().unwrap();
    let WorkUnit::SequenceStep { input, .. } = &plan.work else {
        panic!("restart must schedule sequence prefill")
    };
    assert_eq!(input.start, 0);
    let reservation = core.active_managed_cache.values().next().unwrap();
    assert_eq!(reservation.session_generation.get(), 2);
    assert!(reservation
        .domains
        .iter()
        .all(|domain| domain.execution_start_tokens == 0));
    assert_eq!(core.rollback_all_in_flight_dispatches(), 1);
    drop(prepared);
}

#[tokio::test]
async fn cancellation_during_restart_wins_and_releases_managed_session() {
    let model = ModelInstanceId::new(916);
    let (mut core, cleanup_calls) = restart_test_core(false);
    let mut request =
        managed_test_request(&mut core, model, "managed-restart-cancel", vec![1, 2]);
    let cancellation = Arc::new(AtomicBool::new(false));
    request.cancellation = Some(cancellation.clone());
    core.restart_after_reset_hook = Some(Arc::new(move || {
        cancellation.store(true, Ordering::Release);
    }));
    core.add_request(request).unwrap();
    let session = core
        .get_session_key(&"managed-restart-cancel".to_string())
        .unwrap();

    let outputs = core.step().await.unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(
        outputs[0].finish_reason,
        Some(super::super::types::FinishReason::Aborted)
    );
    assert_eq!(cleanup_calls.load(Ordering::Acquire), 1);
    assert!(core
        .managed_kv_cache
        .snapshot(model, &session, crate::kv::CacheDomainId::new(1))
        .is_none());
    assert_eq!(
        core.managed_kv_cache
            .runtime_snapshot()
            .totals
            .coordinator
            .admission_claimed_pages,
        0
    );
}

#[tokio::test]
async fn stale_restart_generation_fails_terminally_without_requeue() {
    let model = ModelInstanceId::new(917);
    let (mut core, _cleanup_calls) = restart_test_core(false);
    let request = managed_test_request(&mut core, model, "managed-restart-stale", vec![1, 2]);
    core.add_request(request).unwrap();
    let session = core
        .get_session_key(&"managed-restart-stale".to_string())
        .unwrap();
    let prepared = core
        .prepare_step()
        .await
        .unwrap()
        .expect("prepared restart");
    core.active_managed_cache
        .values_mut()
        .next()
        .unwrap()
        .session_generation = super::super::ManagedSessionGeneration::for_test(2);

    let executed = core.execute_prepared_with_progress(prepared).await.unwrap();
    let committed = core.commit_step(executed).await.unwrap();
    assert_eq!(committed.outputs.len(), 1);
    assert!(committed.outputs[0]
        .error
        .as_deref()
        .is_some_and(|error| error.contains("managed sequence reset failed")));
    assert!(core
        .managed_kv_cache
        .snapshot(model, &session, crate::kv::CacheDomainId::new(1))
        .is_none());
    assert!(!core.has_request(&session.request_id));
}

#[tokio::test]
async fn scheduler_restart_rejection_fails_terminally_after_cache_reset() {
    let model = ModelInstanceId::new(918);
    let (mut core, _cleanup_calls) = restart_test_core(false);
    let request =
        managed_test_request(&mut core, model, "managed-restart-rejected", vec![1, 2]);
    core.add_request(request).unwrap();
    let session = core
        .get_session_key(&"managed-restart-rejected".to_string())
        .unwrap();
    let prepared = core
        .prepare_step()
        .await
        .unwrap()
        .expect("prepared restart");
    let executed = core.execute_prepared_with_progress(prepared).await.unwrap();
    core.scheduler.finish_request(&session.request_id);

    let committed = core.commit_step(executed).await.unwrap();
    assert_eq!(committed.outputs.len(), 1);
    assert!(committed.outputs[0]
        .error
        .as_deref()
        .is_some_and(|error| error.contains("scheduler rejected a managed sequence restart")));
    assert!(core
        .managed_kv_cache
        .snapshot(model, &session, crate::kv::CacheDomainId::new(1))
        .is_none());
    assert!(!core.has_request(&session.request_id));
}

#[tokio::test]
async fn managed_kv_capacity_defers_without_failing_or_sticking_prefill() {
    let model_instance = ModelInstanceId::new(92);
    let executor = UnifiedExecutor::new_for_test(Box::new(ManagedReceiptExecutor));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            enable_chunked_prefill: false,
            block_size: 16,
            max_blocks: 1,
            max_batch_size: 1,
            backend: BackendKind::Cpu,
            ..Default::default()
        },
        executor,
    )
    .expect("core");
    core.retry_policy.execution_backoff_base = Duration::ZERO;
    core.retry_policy.execution_backoff_max = Duration::ZERO;
    let owner = managed_test_request(
        &mut core,
        model_instance,
        "capacity-owner",
        vec![1, 2, 3, 4],
    );
    core.add_request(owner).expect("add owner");
    core.step().await.expect("commit owner prefill");
    let owner_session = core
        .get_session_key(&"capacity-owner".to_string())
        .expect("owner session");
    core.scheduler.finish_request(&"capacity-owner".to_string());

    let waiter = managed_test_request(
        &mut core,
        model_instance,
        "capacity-waiter",
        vec![5, 6, 7, 8],
    );
    core.add_request(waiter).expect("add waiter");
    assert!(core
        .prepare_step()
        .await
        .expect("capacity is scheduler backpressure")
        .is_none());
    assert!(core.active_plans.is_empty());
    assert!(core.active_managed_cache.is_empty());

    core.managed_kv_cache
        .release_session(&owner_session)
        .expect("release owner capacity");
    core.step()
        .await
        .expect("waiter retries after capacity release");
    let waiter_session = core
        .get_session_key(&"capacity-waiter".to_string())
        .expect("waiter session");
    let snapshot = core
        .managed_kv_cache
        .snapshot(
            model_instance,
            &waiter_session,
            crate::kv::CacheDomainId::new(1),
        )
        .expect("waiter committed table");
    assert_eq!(snapshot.committed_tokens, 4);
}

fn chunked_claim_core(quantum_tokens: usize) -> EngineCore {
    let executor = UnifiedExecutor::new_for_test(Box::new(ManagedReceiptExecutor));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            enable_chunked_prefill: false,
            block_size: 16,
            max_blocks: 4,
            max_batch_size: 2,
            managed_chat_claim_quantum_tokens: quantum_tokens,
            backend: BackendKind::Cpu,
            ..Default::default()
        },
        executor,
    )
    .expect("core");
    core.retry_policy.execution_backoff_base = Duration::ZERO;
    core.retry_policy.execution_backoff_max = Duration::ZERO;
    core
}

fn committed_tokens(core: &EngineCore, model: ModelInstanceId, id: &str) -> Option<u32> {
    let session = core.get_session_key(&id.to_string())?;
    core.managed_kv_cache
        .snapshot(model, &session, crate::kv::CacheDomainId::new(1))
        .map(|snapshot| snapshot.committed_tokens)
}

/// A1: an output budget that covers most of the arena (an omitted OpenAI
/// `max_tokens` becomes "rest of context") must not reserve the whole
/// arena up front. With chunked claims both requests prefill together;
/// full up-front claims serialize the second behind the first.
#[tokio::test]
async fn chunked_chat_claims_admit_concurrent_requests_that_full_claims_serialize() {
    for (quantum, both_admitted) in [(16, true), (0, false)] {
        let model = ModelInstanceId::new(95);
        let mut core = chunked_claim_core(quantum);
        for id in ["first", "second"] {
            let mut request = managed_test_request(&mut core, model, id, (1..=8).collect());
            // 8 prompt + 40 output tokens = three 16-token pages each, but
            // the arena holds four pages in total.
            request.params.max_tokens = 40;
            core.add_request(request).expect("add");
        }
        // A full-prefill row cannot share a step with decode rows; the
        // scheduler's starvation bound serves it within eight steps.
        for _ in 0..12 {
            core.step().await.expect("step");
        }
        assert!(committed_tokens(&core, model, "first").is_some_and(|tokens| tokens >= 8));
        assert_eq!(
            committed_tokens(&core, model, "second").is_some_and(|tokens| tokens >= 8),
            both_admitted,
            "quantum {quantum}"
        );
    }
}

/// A1: chunked claims grow during decode. When every claim holder needs to
/// grow at once and nothing can be suspended or preempted, the youngest
/// request fails instead of every request waiting forever.
#[tokio::test]
async fn chunked_claim_deadlock_fails_only_the_youngest_request() {
    let model = ModelInstanceId::new(96);
    let mut core = chunked_claim_core(16);
    for id in ["older", "younger"] {
        let mut request = managed_test_request(&mut core, model, id, (1..=8).collect());
        request.params.max_tokens = 40;
        core.add_request(request).expect("add");
    }
    let mut outputs = Vec::new();
    let mut older_peak = 0;
    for _ in 0..80 {
        outputs.extend(
            core.step()
                .await
                .expect("a claim deadlock must not fail the engine step"),
        );
        older_peak = older_peak.max(committed_tokens(&core, model, "older").unwrap_or(0));
        if outputs
            .iter()
            .any(|output| output.request_id == "younger" && output.is_finished)
        {
            break;
        }
    }
    let failed = outputs
        .iter()
        .find(|output| output.request_id == "younger" && output.is_finished)
        .expect("the youngest blocked request is failed");
    assert!(
        failed
            .error
            .as_deref()
            .is_some_and(|error| error.contains("exhausted by concurrent requests")),
        "{:?}",
        failed.error
    );
    assert!(core.has_request(&"older".to_string()));
    for _ in 0..4 {
        outputs.extend(core.step().await.expect("survivor step"));
        older_peak = older_peak.max(committed_tokens(&core, model, "older").unwrap_or(0));
    }
    // Both held two pages (32 tokens) when they deadlocked; the survivor
    // grows past that once the younger claim is released.
    assert!(older_peak > 32, "older peaked at {older_peak}");
}

/// D4: a burst of equal-priority chat requests over an arena too small
/// for them: the first two claim holders end up waiting to grow at once
/// (the claim deadlock of
/// `chunked_claim_deadlock_fails_only_the_youngest_request`) while the
/// third waits for admission. With capacity replay (the default) the
/// younger holder is suspended and replayed after the protected request
/// finishes, so every request completes with exactly its token budget and
/// prompt usage. With replay off the claim-deadlock breaker fails one.
#[tokio::test]
async fn kv_pressure_burst_completes_every_request_with_capacity_replay() {
    for replay in [true, false] {
        let model = ModelInstanceId::new(97);
        let executor = UnifiedExecutor::new_for_test(Box::new(ManagedDecodeExecutor {
            token_replay: true,
            ..Default::default()
        }));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                enable_chunked_prefill: false,
                block_size: 16,
                // Each request needs three pages for 8 prompt + 40
                // output tokens; the arena holds four.
                max_blocks: 4,
                max_batch_size: 2,
                managed_chat_claim_quantum_tokens: 16,
                enable_capacity_replay: replay,
                backend: BackendKind::Cpu,
                ..Default::default()
            },
            executor,
        )
        .expect("core");
        core.retry_policy.execution_backoff_base = Duration::ZERO;
        core.retry_policy.execution_backoff_max = Duration::ZERO;
        let ids = ["oldest", "middle", "youngest"];
        for id in ids {
            let mut request = managed_test_request(&mut core, model, id, (1..=8).collect());
            request.params.max_tokens = 40;
            core.add_request(request).expect("add");
        }
        let before = super::super::metrics::engine_batch_metrics_snapshot();
        let mut finished = HashMap::new();
        for _ in 0..600 {
            for output in core
                .step()
                .await
                .expect("capacity pressure must not fail the engine step")
            {
                if output.is_finished {
                    finished.insert(output.request_id.clone(), output);
                }
            }
            if finished.len() == ids.len() {
                break;
            }
        }
        let after = super::super::metrics::engine_batch_metrics_snapshot();
        if !replay {
            assert!(
                finished
                    .values()
                    .any(|output| output.error.as_deref().is_some_and(
                        |error| error.contains("exhausted by concurrent requests")
                    )),
                "without replay the deadlock breaker fails a request"
            );
            continue;
        }
        assert_eq!(finished.len(), ids.len(), "every request completes");
        for id in ids {
            let output = &finished[id];
            assert!(output.error.is_none(), "{id}: {:?}", output.error);
            assert_eq!(output.token_stats.generated_tokens, 40, "{id}");
            assert_eq!(output.token_stats.prompt_tokens, 8, "{id}");
        }
        assert!(after.capacity_suspensions_total > before.capacity_suspensions_total);
    }
}

/// Stage the hybrid test model's recurrent state (domain 2) over a row's
/// paged span, as a hybrid adapter does before its rows commit.
fn stage_test_tensor_state(
    request: &EngineCoreRequest,
    reservation: &super::super::ManagedCacheReservation,
) {
    let arena = request
        .managed_cache_runtime()
        .and_then(|runtime| runtime.tensor_state())
        .expect("hybrid test row has a tensor-state arena");
    let paged = &reservation.domains[0];
    arena
        .stage_replace(
            crate::backends::state::PhysicalStateTransactionId::new(reservation.txn_id)
                .expect("tensor transaction"),
            crate::kv::v2::StateDomainId::new(2),
            u64::from(paged.expected_committed_tokens),
            u64::from(paged.target_committed_tokens),
            vec![crate::backends::state::StateComponentValue {
                component: crate::kv::v2::StateComponentId::new(1),
                tensor: Some(
                    candle_core::Tensor::zeros(
                        4,
                        candle_core::DType::F32,
                        &candle_core::Device::Cpu,
                    )
                    .expect("tensor"),
                ),
            }],
        )
        .expect("stage tensor state");
}

/// Managed rows whose decode quanta emit tokens, so requests finish at
/// their output budget. Prefill is full unless `incremental_prefill`.
/// Records each executed row's plan id, request and phase.
#[derive(Default)]
struct ManagedDecodeExecutor {
    incremental_prefill: bool,
    /// Declare token replay and suspend sessions for capacity, keeping
    /// each session's committed token count as its replay length.
    token_replay: bool,
    committed: Arc<Mutex<HashMap<super::super::SessionKey, usize>>>,
    /// Declare resumable prefill beside continuous decode rows of this
    /// width, which a bound mixed stage turns into mixed steps.
    mixed_width: Option<usize>,
    executed: Arc<Mutex<Vec<(super::super::PlanId, RequestId, bool)>>>,
    /// Each executed batch's rows: plan id and whether it prefilled.
    batches: Arc<Mutex<Vec<Vec<(super::super::PlanId, bool)>>>>,
}

impl ModelExecutor for ManagedDecodeExecutor {
    fn execution_profile(&self, request: &EngineCoreRequest) -> Option<ExecutionProfile> {
        let mut profile = ManagedReceiptExecutor.execution_profile(request)?;
        if self.incremental_prefill {
            profile.prefill = PrefillMode::Incremental;
        }
        if self.token_replay {
            profile.replay = super::super::ReplayCapability::TokenReplay;
        }
        if let Some(width) = self.mixed_width {
            profile.prefill = PrefillMode::Incremental;
            profile.incremental_decode = true;
            profile.decode_batch = NativeBatchMode::Continuous;
            profile.concurrency = ConcurrencyClass::Batchable;
            profile.max_batch_size = width;
        }
        Some(profile)
    }

    fn cleanup_session(
        &self,
        session: &super::super::SessionKey,
    ) -> super::super::executor::CacheReleaseReport {
        ManagedReceiptExecutor.cleanup_session(session)
    }

    fn suspend_session_for_capacity(
        &self,
        session: &super::super::SessionKey,
    ) -> Result<Option<usize>> {
        Ok(self
            .token_replay
            .then(|| self.committed.lock().unwrap().remove(session))
            .flatten())
    }

    fn execute_physical_batch(
        &self,
        execution: super::super::executor::PhysicalBatchExecution<'_>,
    ) -> super::super::executor::PhysicalDispatchResult {
        let dispatch = execution.expected_dispatch();
        self.committed
            .lock()
            .unwrap()
            .extend(execution.scheduled.iter().map(|scheduled| {
                (
                    scheduled.session_key(),
                    scheduled.num_computed_tokens + scheduled.num_tokens,
                )
            }));
        self.executed
            .lock()
            .unwrap()
            .extend(execution.scheduled.iter().map(|scheduled| {
                (
                    scheduled.plan_id,
                    scheduled.request_id.clone(),
                    scheduled.is_prefill,
                )
            }));
        self.batches.lock().unwrap().push(
            execution
                .scheduled
                .iter()
                .map(|scheduled| (scheduled.plan_id, scheduled.is_prefill))
                .collect(),
        );
        Ok(execution
            .scheduled
            .iter()
            .zip(execution.requests)
            .zip(&execution.batch.rows)
            .map(|((scheduled, request), row)| {
                let decoded = (!scheduled.is_prefill).then_some(scheduled.num_tokens);
                let generated = (scheduled.num_computed_tokens + scheduled.num_tokens)
                    .saturating_sub(request.num_prompt_tokens());
                let mut result = ExecutorStepResult::new(
                    scheduled,
                    ExecutorOutput {
                        request_id: scheduled.request_id.clone(),
                        audio: None,
                        text: None,
                        input_transcription: None,
                        tokens_processed: scheduled.num_tokens,
                        tokens_generated: decoded.unwrap_or(0),
                        finished: decoded.is_some() && generated >= request.params.max_tokens,
                        phase_timing_override: None,
                        asr_diagnostics: None,
                        error: None,
                        logprobs: Vec::new(),
                    },
                );
                result.dispatch = dispatch;
                if let Some(reservation) = &row.managed_cache {
                    if reservation.clocked_state.is_some() {
                        stage_test_tensor_state(request, reservation);
                    }
                    result.managed_cache = Some(reservation.completed_write_receipt_for_test());
                }
                result
            })
            .collect())
    }

    fn execute_prefill(
        &self,
        _requests: &[&EngineCoreRequest],
        _scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        unreachable!("physical dispatch is overridden")
    }

    fn execute_decode(
        &self,
        _requests: &[&EngineCoreRequest],
        _scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        unreachable!("physical dispatch is overridden")
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn initialize(&mut self) -> Result<()> {
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }

    fn cleanup_request(&self, _request_id: &str) -> CacheReleaseReport {
        CacheReleaseReport::confirmed(0)
    }
}

/// The Metal burst stall: the starvation bound forces a full prefill after
/// eight decode-only commits, but the prefill needs managed capacity that
/// only the decoding rows free, by finishing. On Metal that was a tensor
/// state sequence slot; here it is the pages two full-request claims
/// hold. The bound used to keep forcing the withheld prefill with no
/// decode budget, so no row ever committed again.
#[tokio::test]
async fn capacity_blocked_forced_prefill_returns_service_to_decode() {
    let model = ModelInstanceId::new(98);
    let prompt: Vec<u32> = (1..=8).collect();
    let executor = UnifiedExecutor::new_for_test(Box::new(ManagedDecodeExecutor::default()));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            enable_chunked_prefill: false,
            block_size: 16,
            // 8 prompt + 24 output tokens claim two pages: two holders
            // fill the arena.
            max_blocks: 4,
            max_batch_size: 8,
            managed_chat_claim_quantum_tokens: 0,
            backend: BackendKind::Cpu,
            ..Default::default()
        },
        executor,
    )
    .expect("core");
    core.retry_policy.execution_backoff_base = Duration::ZERO;
    core.retry_policy.execution_backoff_max = Duration::ZERO;
    let ids = ["r0", "r1", "r2", "r3"];
    for id in ids {
        let mut request = managed_test_request(&mut core, model, id, prompt.clone());
        request.params.max_tokens = 24;
        core.add_request(request).expect("add");
    }

    let mut finished = HashSet::new();
    for _ in 0..400 {
        for output in core.step().await.expect("step") {
            if output.is_finished {
                assert!(output.error.is_none(), "{:?}", output.error);
                finished.insert(output.request_id);
            }
        }
        if finished.len() == ids.len() {
            break;
        }
    }
    assert_eq!(finished.len(), ids.len(), "finished {finished:?}");
}

/// SCH-F2: the scheduler selects no more decode rows than one physical
/// dispatch holds (one unbatched row here, with eight scheduler rows), so
/// every plan it begins executes, and it rotates through the running
/// streams instead of serving the oldest until it finishes.
#[tokio::test]
async fn clamped_decode_executes_every_plan_and_rotates_all_streams() {
    let model = ModelInstanceId::new(99);
    let prompt: Vec<u32> = (1..=8).collect();
    let executed = Arc::new(Mutex::new(Vec::new()));
    let executor = UnifiedExecutor::new_for_test(Box::new(ManagedDecodeExecutor {
        executed: executed.clone(),
        ..Default::default()
    }));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            enable_chunked_prefill: false,
            block_size: 16,
            max_blocks: 64,
            max_batch_size: 8,
            backend: BackendKind::Cpu,
            ..Default::default()
        },
        executor,
    )
    .expect("core");
    core.retry_policy.execution_backoff_base = Duration::ZERO;
    core.retry_policy.execution_backoff_max = Duration::ZERO;
    let ids = ["s0", "s1", "s2", "s3"];
    for id in ids {
        let mut request = managed_test_request(&mut core, model, id, prompt.clone());
        request.params.max_tokens = 200;
        core.add_request(request).expect("add");
    }
    let prefilled = |executed: &Mutex<Vec<(super::super::PlanId, RequestId, bool)>>| {
        executed.lock().unwrap().iter().filter(|row| row.2).count()
    };
    for _ in 0..100 {
        if prefilled(&executed) == ids.len() {
            break;
        }
        core.step().await.expect("prefill step");
    }
    assert_eq!(prefilled(&executed), ids.len());

    let start = executed.lock().unwrap().len();
    for _ in 0..12 {
        core.step().await.expect("decode step");
    }
    let decoded = executed.lock().unwrap()[start..].to_vec();
    assert_eq!(decoded.len(), 12, "one decode row per step: {decoded:?}");
    // Plan ids are allocated per scheduled row, so consecutive ids mean
    // no plan was begun and then rolled back.
    for pair in decoded.windows(2) {
        assert_eq!(pair[1].0, pair[0].0 + 1, "{decoded:?}");
    }
    for window in decoded.chunks(ids.len()) {
        let served = window
            .iter()
            .map(|row| row.1.as_str())
            .collect::<HashSet<_>>();
        assert_eq!(served.len(), ids.len(), "{decoded:?}");
    }
}

/// Runs `prompts` (with their output budgets) through a hybrid paged +
/// recurrent-tensor model with chunked prefill on, and asserts every
/// request finishes without error and no managed transaction is left.
async fn run_chunked_hybrid_requests(
    config: EngineCoreConfig,
    retained_rows: usize,
    prompts: &[(usize, usize)],
) {
    run_hybrid_requests(config, retained_rows, prompts, None).await;
}

/// What one hybrid run executed: each batch's rows (plan id, prefill)
/// and each step's model calls.
struct HybridRun {
    batches: Vec<Vec<(super::super::PlanId, bool)>>,
    step_model_calls: Vec<usize>,
}

/// [`run_chunked_hybrid_requests`], or with `mixed_width` a model bound
/// to one mixed stage of that width with the engine's chunked prefill
/// off (CPU and Metal), so prefill spans run in mixed steps.
async fn run_hybrid_requests(
    config: EngineCoreConfig,
    retained_rows: usize,
    prompts: &[(usize, usize)],
    mixed_width: Option<usize>,
) -> HybridRun {
    use crate::kv::v2::{
        BoundedShape, CheckpointPolicy, PlacementPolicy, PrefixPolicy, ShapeAxis,
        ShapeDimension, ShapeExtent, StateComponentId, StateDType, StateDomainHeader,
        StateDomainId, StateDomainSpec, StateScope, TensorComponentSpec, TensorRole,
        TensorStateDomainSpec,
    };
    let model = ModelInstanceId::new(100);
    let batches = Arc::new(Mutex::new(Vec::new()));
    let executor = UnifiedExecutor::new_for_test(Box::new(ManagedDecodeExecutor {
        incremental_prefill: true,
        mixed_width,
        batches: batches.clone(),
        ..Default::default()
    }));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            enable_chunked_prefill: mixed_width.is_none(),
            backend: BackendKind::Cpu,
            ..config
        },
        executor,
    )
    .expect("core");
    core.retry_policy.execution_backoff_base = Duration::ZERO;
    core.retry_policy.execution_backoff_max = Duration::ZERO;

    // Paged attention plus one recurrent tensor domain, like LFM2, whose
    // tensor-state sequence slots are the model's retained rows.
    let mut contract = crate::kv::test_contract();
    let tensor = StateDomainId::new(2);
    contract
        .domains
        .push(StateDomainSpec::Tensor(TensorStateDomainSpec {
            header: StateDomainHeader {
                id: tensor,
                scope: StateScope::Retained,
                clock: StateClock::DecoderTokens,
                placement: PlacementPolicy::BackendLocalWithHostOffload,
                prefix: PrefixPolicy::Disabled,
                checkpoint: CheckpointPolicy::Transactional,
            },
            components: vec![TensorComponentSpec {
                id: StateComponentId::new(1),
                role: TensorRole::RecurrentHidden,
                shape: BoundedShape {
                    dimensions: vec![ShapeDimension {
                        axis: ShapeAxis::Hidden,
                        extent: ShapeExtent::Fixed { value: 4 },
                    }],
                },
                accepted_dtypes: vec![StateDType::F32],
            }],
        }));
    contract.groups[0].domains.push(tensor);
    contract.groups[0].prefix_shareable = false;
    let physical = core
        .load_managed_model_state_with_row_limits(
            model,
            &contract,
            None,
            retained_rows,
            retained_rows,
        )
        .expect("load hybrid managed state");

    let mut ids = HashSet::new();
    for (index, &(prompt_tokens, max_tokens)) in prompts.iter().enumerate() {
        let id = format!("hybrid-{index}");
        let mut request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: id.clone(),
        }])
        .with_model_variant(ModelVariant::Qwen306B);
        request.id = id.clone();
        request.params.max_tokens = max_tokens;
        let prompt = (1..=prompt_tokens as u32).collect();
        request
            .install_chat_execution_preparation(ModelVariant::Qwen306B, prompt, None, 4096)
            .expect("prepare prompt");
        request.bind_model_instance(model).expect("model instance");
        match mixed_width {
            Some(width) => bind_managed_test_runtime_with_batch(
                &mut request,
                model,
                contract.clone(),
                physical.clone(),
                PrefillMode::Incremental,
                NativeBatchMode::Mixed,
                width,
            ),
            None => bind_managed_test_runtime(
                &mut request,
                model,
                contract.clone(),
                physical.clone(),
                PrefillMode::Incremental,
            ),
        }
        core.add_request(request).expect("add");
        ids.insert(id);
    }

    let mut finished = HashSet::new();
    for _ in 0..2_000 {
        for output in core.step().await.expect("step") {
            if output.is_finished {
                assert!(
                    output.error.is_none(),
                    "{}: {:?}",
                    output.request_id,
                    output.error
                );
                finished.insert(output.request_id);
            }
        }
        if finished == ids {
            break;
        }
    }
    assert_eq!(finished, ids);
    let snapshot = core.managed_kv_cache.runtime_snapshot();
    assert_eq!(snapshot.totals.coordinator.active_transactions, 0);
    let batches = std::mem::take(&mut *batches.lock().unwrap());
    HybridRun {
        batches,
        step_model_calls: std::mem::take(&mut core.step_model_calls),
    }
}

/// Whether some batch of the run held both phases.
fn mixed_batches(run: &HybridRun) -> usize {
    run.batches
        .iter()
        .filter(|rows| {
            rows.iter().any(|(_, prefill)| *prefill)
                && rows.iter().any(|(_, prefill)| !*prefill)
        })
        .count()
}

/// The mixed-step invariants of one run: every batch is decode rows then
/// prefill rows in plan-allocation order (so managed KV was reserved in
/// the order the scheduler allocated the plans), and every step made one
/// model call.
fn assert_mixed_run(run: &HybridRun) {
    for rows in &run.batches {
        assert!(
            rows.windows(2).all(|pair| pair[0].0 < pair[1].0),
            "rows left plan-allocation order: {rows:?}"
        );
        assert!(
            rows.windows(2).all(|pair| !pair[0].1 || pair[1].1),
            "a prefill row preceded a decode row: {rows:?}"
        );
    }
    assert!(
        run.step_model_calls.iter().all(|calls| *calls == 1),
        "{:?}",
        run.step_model_calls
    );
}

/// A mixed step withholds a prefill row whose tensor-state slot is held
/// while its decode rows run. Decode rows are reserved first, so the
/// withheld row's aborted transaction is the step's highest plan id (the
/// order behind the DuplicateTransaction fix, 9d9bae87).
#[tokio::test]
async fn mixed_step_withholds_a_prefill_row_and_keeps_its_decode_rows() {
    let run = run_hybrid_requests(
        EngineCoreConfig {
            chunked_prefill_threshold: 4,
            block_size: 16,
            max_blocks: 64,
            max_batch_size: 4,
            max_retained_sequences: 2,
            max_staged_transactions: 2,
            ..Default::default()
        },
        2,
        &[(8, 6), (8, 6), (8, 6)],
        Some(4),
    )
    .await;
    assert_mixed_run(&run);
    // The third prompt's span was planned beside the two decode rows and
    // withheld: its plan ids never executed.
    let executed = run.batches.iter().map(Vec::len).sum::<usize>();
    let planned = run
        .batches
        .iter()
        .flatten()
        .map(|(plan, _)| *plan)
        .max()
        .unwrap() as usize;
    assert!(planned > executed, "{:?}", run.batches);
}

/// The hybrid burst in mixed steps: more running requests than
/// tensor-state slots, mixed prompt lengths, contended pages, one and
/// two candidate dispatches per step.
#[tokio::test]
async fn mixed_step_hybrid_burst_finishes_every_request() {
    let prompts = (0..16)
        .map(|index| (5 + (index * 7) % 37, 3 + index % 9))
        .collect::<Vec<_>>();
    for mode in [
        crate::config::PhysicalExecutionMode::Serial,
        crate::config::PhysicalExecutionMode::Shadow,
    ] {
        let run = run_hybrid_requests(
            EngineCoreConfig {
                chunked_prefill_threshold: 8,
                block_size: 16,
                max_blocks: 24,
                max_batch_size: 8,
                max_retained_sequences: 4,
                max_staged_transactions: 4,
                physical_execution_mode: mode,
                max_physical_in_flight: crate::config::PhysicalInFlightLimit::new(2).unwrap(),
                ..Default::default()
            },
            4,
            &prompts,
            Some(8),
        )
        .await;
        assert_mixed_run(&run);
        assert!(mixed_batches(&run) > 0, "{mode:?}: {:?}", run.batches);
    }
}

/// Chunked prefill mixes decode and prefill rows in one step. The
/// scheduler numbers decode plans first, but a step can prepare the
/// prefill row first. When that prefill reserved its pages and was then
/// withheld because every tensor-state sequence slot was held, its aborted
/// transaction used to raise the arena's terminal high-water mark above
/// the decode plans prepared next, and each decode prepare failed as a
/// "duplicate" transaction until its request was failed.
#[tokio::test]
async fn chunked_prefill_withheld_for_tensor_state_keeps_decode_rows_alive() {
    run_chunked_hybrid_requests(
        EngineCoreConfig {
            chunked_prefill_threshold: 4,
            block_size: 16,
            max_blocks: 64,
            max_batch_size: 4,
            max_retained_sequences: 2,
            max_staged_transactions: 2,
            ..Default::default()
        },
        2,
        &[(8, 6), (8, 6), (8, 6)],
    )
    .await;
}

/// CPU and Metal serve with chunked prefill off by default, so this burst
/// keeps CUDA's chunked path under test on CPU: more running requests
/// than tensor-state slots, mixed prompt lengths, contended pages, and
/// both one and two candidate dispatches per step.
#[tokio::test]
async fn chunked_prefill_hybrid_burst_finishes_every_request() {
    let prompts = (0..16)
        .map(|index| (5 + (index * 7) % 37, 3 + index % 9))
        .collect::<Vec<_>>();
    for mode in [
        crate::config::PhysicalExecutionMode::Serial,
        crate::config::PhysicalExecutionMode::Shadow,
    ] {
        run_chunked_hybrid_requests(
            EngineCoreConfig {
                chunked_prefill_threshold: 8,
                block_size: 16,
                max_blocks: 24,
                max_batch_size: 8,
                max_retained_sequences: 4,
                max_staged_transactions: 4,
                physical_execution_mode: mode,
                max_physical_in_flight: crate::config::PhysicalInFlightLimit::new(2).unwrap(),
                ..Default::default()
            },
            4,
            &prompts,
        )
        .await;
    }
}

/// SCH-F7: a burst sharing a 64-token system prompt is admitted before
/// any of it is published, so every admission probe misses. A request
/// must start above the shared pages committed when its first prefill
/// span is scheduled, not those committed at admission. With one row per
/// step the first request prefills its whole prompt before any other is
/// selected, so each of the other N-1 attaches the full system prompt.
#[tokio::test]
async fn burst_sharing_a_system_prompt_attaches_it_once_published() {
    const REQUESTS: usize = 4;
    let model = ModelInstanceId::new(941);
    let executor = UnifiedExecutor::new_for_test(Box::new(ManagedDecodeExecutor {
        incremental_prefill: true,
        ..Default::default()
    }));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            enable_chunked_prefill: true,
            chunked_prefill_threshold: 32,
            block_size: 16,
            max_blocks: 512,
            max_batch_size: 1,
            enable_prefix_caching: true,
            managed_prefix_cache_salt: Some("burst".to_string()),
            backend: BackendKind::Cpu,
            ..Default::default()
        },
        executor,
    )
    .expect("core");
    core.retry_policy.execution_backoff_base = Duration::ZERO;
    core.retry_policy.execution_backoff_max = Duration::ZERO;
    let contract = crate::kv::test_contract();
    let capability = crate::kv::InferenceStateCapability::Managed(contract.clone());
    let physical = core
        .load_managed_model_cache(model, &capability, None)
        .expect("load managed state")
        .expect("physical managed state");
    let system = (1..=64).collect::<Vec<u32>>();
    for index in 0..REQUESTS {
        let id = format!("burst-{index}");
        let mut request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: id.clone(),
        }])
        .with_model_variant(ModelVariant::Qwen306B);
        request.id = id;
        request.params.max_tokens = 2;
        let mut prompt = system.clone();
        prompt.extend((0..8).map(|token| 1_000 + 8 * index as u32 + token));
        request
            .install_chat_execution_preparation(ModelVariant::Qwen306B, prompt, None, 4096)
            .expect("prepare prompt");
        // What a chat prompt with a system message declares, so every
        // prefill span inside the system prompt publishes its pages.
        request.managed_preamble_boundary = Some(system.len() as u32);
        request.bind_model_instance(model).expect("model instance");
        bind_managed_test_runtime(
            &mut request,
            model,
            contract.clone(),
            physical.clone(),
            PrefillMode::Incremental,
        );
        core.add_request(request).expect("add");
    }

    let mut cached = HashMap::new();
    for _ in 0..500 {
        for output in core.step().await.expect("step") {
            assert!(output.error.is_none(), "{:?}", output.error);
            if output.is_finished {
                cached.insert(
                    output.request_id.clone(),
                    output.token_stats.cached_prefix_tokens,
                );
            }
        }
        if cached.len() == REQUESTS {
            break;
        }
    }
    assert_eq!(cached.len(), REQUESTS, "{cached:?}");
    for index in 0..REQUESTS {
        let expected = (index > 0).then_some(64);
        assert_eq!(
            cached[&format!("burst-{index}")],
            expected,
            "request {index}: {cached:?}"
        );
    }
    let telemetry = core.managed_kv_cache.telemetry_snapshot();
    let followers = REQUESTS as u64 - 1;
    assert_eq!(
        (telemetry.prefix_hits, telemetry.prefix_hit_tokens),
        (followers, 64 * followers)
    );
}

/// Managed decode rows dispatched as one continuous native batch of up
/// to `width` rows, for the host-overhead benchmark below.
struct BatchedDecodeExecutor {
    width: usize,
    prefill: PrefillMode,
    last_decode_rows: Arc<AtomicUsize>,
    /// Simulated device time of every model call.
    device_time: Duration,
}

impl ModelExecutor for BatchedDecodeExecutor {
    fn execution_profile(&self, request: &EngineCoreRequest) -> Option<ExecutionProfile> {
        let mut profile = ManagedReceiptExecutor.execution_profile(request)?;
        profile.prefill = self.prefill;
        profile.incremental_decode = true;
        profile.prefill_batch = NativeBatchMode::Continuous;
        profile.decode_batch = NativeBatchMode::Continuous;
        profile.concurrency = ConcurrencyClass::Batchable;
        profile.max_batch_size = self.width;
        profile.resolved_from_loaded_model = true;
        Some(profile)
    }

    fn cleanup_session(
        &self,
        session: &super::super::SessionKey,
    ) -> super::super::executor::CacheReleaseReport {
        ManagedReceiptExecutor.cleanup_session(session)
    }

    fn execute_physical_batch(
        &self,
        execution: super::super::executor::PhysicalBatchExecution<'_>,
    ) -> super::super::executor::PhysicalDispatchResult {
        if !execution.is_prefill() {
            self.last_decode_rows
                .store(execution.scheduled.len(), Ordering::Relaxed);
        }
        std::thread::sleep(self.device_time);
        ManagedDecodeExecutor::default().execute_physical_batch(execution)
    }

    fn execute_prefill(
        &self,
        _requests: &[&EngineCoreRequest],
        _scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        unreachable!("physical dispatch is overridden")
    }

    fn execute_decode(
        &self,
        _requests: &[&EngineCoreRequest],
        _scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        unreachable!("physical dispatch is overridden")
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn initialize(&mut self) -> Result<()> {
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }

    fn cleanup_request(&self, _request_id: &str) -> CacheReleaseReport {
        CacheReleaseReport::confirmed(0)
    }
}

/// A core over the no-op batched executor, and what it takes to add
/// benchmark requests: 8-token prompts with 4,000-token budgets.
struct BenchCore {
    core: EngineCore,
    last_decode_rows: Arc<AtomicUsize>,
    model: ModelInstanceId,
    contract: crate::kv::v2::InferenceStateContract,
    physical: Arc<super::super::ManagedKvModelRuntime>,
    prefill: PrefillMode,
    mixed: bool,
    width: usize,
}

impl BenchCore {
    /// Room for `rows` requests in one native batch of `rows` rows. A
    /// mixed core binds one mixed stage, which runs resumable prefill.
    fn new(rows: usize, prefill: PrefillMode, mixed: bool) -> Self {
        Self::with_device(rows, prefill, mixed, None)
    }

    /// A bench core whose model calls run on `device` and take
    /// `device_time` there.
    fn with_device(
        rows: usize,
        prefill: PrefillMode,
        mixed: bool,
        device: Option<(Arc<crate::backends::DeviceThread>, Duration)>,
    ) -> Self {
        let prefill = if mixed {
            PrefillMode::Incremental
        } else {
            prefill
        };
        let model = ModelInstanceId::new(101);
        let last_decode_rows = Arc::new(AtomicUsize::new(0));
        let mut executor = UnifiedExecutor::new_for_test(Box::new(BatchedDecodeExecutor {
            width: rows,
            prefill,
            last_decode_rows: last_decode_rows.clone(),
            device_time: device.as_ref().map_or(Duration::ZERO, |(_, time)| *time),
        }));
        if let Some((device, _)) = device {
            executor = executor.with_device_for_test(device);
        }
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                enable_chunked_prefill: prefill == PrefillMode::Incremental && !mixed,
                block_size: 16,
                // Full-request claims of 8 prompt + 4,000 output tokens.
                max_blocks: rows * 256,
                max_batch_size: rows,
                max_tokens_per_step: rows * 16,
                max_retained_sequences: rows,
                max_staged_transactions: rows,
                managed_chat_claim_quantum_tokens: 0,
                backend: BackendKind::Cpu,
                ..Default::default()
            },
            executor,
        )
        .expect("core");
        core.retry_policy.execution_backoff_base = Duration::ZERO;
        core.retry_policy.execution_backoff_max = Duration::ZERO;
        let contract = crate::kv::test_contract();
        let capability = crate::kv::InferenceStateCapability::Managed(contract.clone());
        let physical = core
            .load_managed_model_cache(model, &capability, None)
            .expect("load managed state")
            .expect("physical managed state");
        Self {
            core,
            last_decode_rows,
            model,
            contract,
            physical,
            prefill,
            mixed,
            width: rows,
        }
    }

    fn add(&mut self, id: String) {
        let mut request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: id.clone(),
        }])
        .with_model_variant(ModelVariant::Qwen306B);
        request.id = id;
        request.params.max_tokens = 4_000;
        request
            .install_chat_execution_preparation(
                ModelVariant::Qwen306B,
                (1..=8).collect(),
                None,
                8_192,
            )
            .expect("prepare prompt");
        request
            .bind_model_instance(self.model)
            .expect("model instance");
        bind_managed_test_runtime_with_batch(
            &mut request,
            self.model,
            self.contract.clone(),
            self.physical.clone(),
            self.prefill,
            if self.mixed {
                NativeBatchMode::Mixed
            } else {
                NativeBatchMode::Continuous
            },
            self.width,
        );
        self.core.add_request(request).expect("add");
    }

    /// Step until one decode batch holds `rows` rows.
    async fn warm(&mut self, rows: usize) {
        // Full prefill admits one row per starvation window of nine steps.
        for _ in 0..rows * 12 {
            for output in self.core.step().await.expect("warmup step") {
                assert!(output.error.is_none(), "{:?}", output.error);
            }
            if self.last_decode_rows.load(Ordering::Relaxed) == rows {
                return;
            }
        }
        panic!("{rows} rows never decoded together");
    }

    /// One step's prepare and whole host time.
    async fn timed_step(&mut self) -> (Duration, Duration) {
        let started = Instant::now();
        let prepared = self
            .core
            .prepare_step()
            .await
            .expect("prepare")
            .expect("work");
        let prepare = started.elapsed();
        for output in self
            .core
            .execute_and_commit_prepared_with_progress(prepared)
            .await
            .expect("execute and commit")
        {
            assert!(output.error.is_none(), "{:?}", output.error);
        }
        (prepare, started.elapsed())
    }
}

fn mean_us(elapsed: Duration, steps: usize) -> f64 {
    elapsed.as_secs_f64() * 1e6 / steps as f64
}

/// The production step driver (`Engine::step`) over `rows` decoding
/// requests whose model calls take `device_time` on the CPU device
/// thread: the mean device idle gap between steps and the mean step wall
/// time, both in microseconds, over `steps` steps.
async fn measure_device_idle(rows: usize, steps: usize, device_time: Duration) -> (f64, f64) {
    let device = crate::backends::DeviceThread::for_location(candle_core::DeviceLocation::Cpu)
        .expect("device thread");
    let mut bench =
        BenchCore::with_device(rows, PrefillMode::Full, false, Some((device, device_time)));
    for index in 0..rows {
        bench.add(format!("idle-{index}"));
    }
    bench.warm(rows).await;
    let engine = super::super::tests::engine_around_core(bench.core);
    // The first measured step follows the warm-up's last step.
    engine.step().await.expect("step");
    let (count_before, idle_before) = super::super::metrics::engine_device_idle_totals("cpu");
    let started = Instant::now();
    for _ in 0..steps {
        for output in engine.step().await.expect("step") {
            assert!(output.error.is_none(), "{:?}", output.error);
        }
    }
    let wall = started.elapsed();
    let (count, idle) = super::super::metrics::engine_device_idle_totals("cpu");
    assert_eq!(count - count_before, steps as u64, "one idle gap per step");
    (
        (idle - idle_before) * 1e6 / steps as f64,
        mean_us(wall, steps),
    )
}

/// Mean host microseconds per decode step (prepare, then execute and
/// commit) with `rows` running requests in one native batch and a
/// no-op executor, measured over `steps` steps after every row decodes.
async fn measure_step_host_overhead(rows: usize, steps: usize) -> (f64, f64) {
    let mut bench = BenchCore::new(rows, PrefillMode::Full, false);
    for index in 0..rows {
        bench.add(format!("bench-{index}"));
    }
    bench.warm(rows).await;

    let mut prepare = Duration::ZERO;
    let mut total = Duration::ZERO;
    for _ in 0..steps {
        bench.last_decode_rows.store(0, Ordering::Relaxed);
        let (step_prepare, step_total) = bench.timed_step().await;
        prepare += step_prepare;
        total += step_total;
        assert_eq!(bench.last_decode_rows.load(Ordering::Relaxed), rows);
    }
    (mean_us(prepare, steps), mean_us(total, steps))
}

/// A burst of `burst` prompts arriving while `rows` requests decode in
/// one native batch: the steps and model calls until every burst prompt
/// has prefilled, and the mean prepare and step host microseconds.
async fn measure_prefill_burst(
    rows: usize,
    burst: usize,
    prefill: PrefillMode,
    mixed: bool,
) -> (usize, usize, f64, f64) {
    let mut bench = BenchCore::new(rows + burst, prefill, mixed);
    for index in 0..rows {
        bench.add(format!("decode-{index}"));
    }
    bench.warm(rows).await;
    let burst_ids = (0..burst)
        .map(|index| format!("burst-{index}"))
        .collect::<Vec<_>>();
    for id in &burst_ids {
        bench.add(id.clone());
    }
    bench.core.step_model_calls.clear();
    let mut steps = 0;
    let mut prepare = Duration::ZERO;
    let mut total = Duration::ZERO;
    while !burst_ids.iter().all(|id| {
        bench
            .core
            .request_phase_timings
            .get(id)
            .is_some_and(|timing| timing.prefill_steps > 0)
    }) {
        assert!(steps < burst * 64, "the burst never prefilled");
        let (step_prepare, step_total) = bench.timed_step().await;
        prepare += step_prepare;
        total += step_total;
        steps += 1;
    }
    let calls = bench.core.step_model_calls.iter().sum();
    (steps, calls, mean_us(prepare, steps), mean_us(total, steps))
}

/// SCH-F9 microbenchmark: engine host time per decode step at 64 and 256
/// rows with a no-op executor, then a burst of 8 prompts under 64 decode
/// rows with full and resumable prefill on the split path and in mixed
/// steps: the steps and model calls to prefill the burst. Run in release:
/// `cargo test --release -p izwi-core --lib step_host_overhead_bench -- --ignored --nocapture`
#[tokio::test]
#[ignore = "host-overhead microbenchmark; run explicitly in release"]
async fn step_host_overhead_bench() {
    use std::io::Write;
    let steps = std::env::var("IZWI_STEP_BENCH_STEPS")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(200);
    for rows in [64, 256] {
        let (prepare, total) = measure_step_host_overhead(rows, steps).await;
        let _ = writeln!(
            std::io::stderr(),
            "step_host_overhead rows={rows} steps={steps} prepare_us={prepare:.1} \
                 step_us={total:.1} per_row_us={:.2}",
            total / rows as f64
        );
    }
    for rows in [64, 256] {
        let device_time = Duration::from_millis(2);
        let (idle, step) = measure_device_idle(rows, steps, device_time).await;
        let _ = writeln!(
            std::io::stderr(),
            "device_idle rows={rows} steps={steps} device_us={} idle_us={idle:.1} \
                 step_us={step:.1}",
            device_time.as_micros()
        );
    }
    for (prefill, mixed) in [
        (PrefillMode::Full, false),
        (PrefillMode::Incremental, false),
        (PrefillMode::Incremental, true),
    ] {
        let (steps, calls, prepare, total) = measure_prefill_burst(64, 8, prefill, mixed).await;
        let _ = writeln!(
            std::io::stderr(),
            "prefill_burst decode_rows=64 burst=8 prefill={prefill:?} mixed={mixed} \
                 steps={steps} model_calls={calls} prepare_us={prepare:.1} step_us={total:.1}"
        );
    }
}

/// The split path serves a prompt arriving under decode in its own model
/// call: every step makes exactly one call, and the burst takes at least
/// one step per prompt plus the decode steps between them.
#[tokio::test]
async fn split_path_prefills_each_prompt_in_its_own_model_call() {
    for prefill in [PrefillMode::Full, PrefillMode::Incremental] {
        let (steps, calls, _, _) = measure_prefill_burst(4, 3, prefill, false).await;
        assert_eq!(calls, steps, "{prefill:?}: one model call per step");
        assert!(steps >= 3 + 2, "{prefill:?}: {steps} steps");
    }
}

#[test]
fn mixed_batches_isolate_only_multi_token_decode_quanta() {
    let step = |phase, tokens: usize| WorkUnit::SequenceStep {
        phase,
        input: super::super::InputRange {
            start: 0,
            end: tokens,
        },
        max_output_steps: tokens,
        auxiliary_state: None,
    };
    let cost = |units| WorkCost::new(units, units, 0);
    let isolated = PhysicalBatchAssembly::is_isolated_quantum;
    // A model-authored multi-token decode quantum (MTP) stays alone.
    assert!(isolated(
        NativeBatchMode::Mixed,
        &step(SequencePhase::Decode, 4),
        &cost(4)
    ));
    assert!(!isolated(
        NativeBatchMode::Mixed,
        &step(SequencePhase::Decode, 1),
        &cost(1)
    ));
    // Prefill spans of any length share a mixed batch.
    assert!(!isolated(
        NativeBatchMode::Mixed,
        &step(SequencePhase::Prefill, 16),
        &cost(16)
    ));
    // Continuous envelopes keep isolating every multi-token row.
    assert!(isolated(
        NativeBatchMode::Continuous,
        &step(SequencePhase::Prefill, 16),
        &cost(16)
    ));
}

/// In mixed steps the same burst prefills beside the decode rows in one
/// model call.
#[tokio::test]
async fn mixed_steps_prefill_a_burst_beside_decode_in_one_call() {
    let (steps, calls, _, _) =
        measure_prefill_burst(4, 3, PrefillMode::Incremental, true).await;
    assert_eq!((steps, calls), (1, 1));
}

/// H2: a row whose managed state can never be prepared fails alone. The
/// step used to return the error for every scheduled row, and the run loop
/// rescheduled the same row forever, starving all other requests.
#[tokio::test]
async fn managed_prepare_failure_fails_only_its_own_request() {
    let model_instance = ModelInstanceId::new(94);
    let executor = UnifiedExecutor::new_for_test(Box::new(ManagedReceiptExecutor));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            enable_chunked_prefill: false,
            block_size: 16,
            max_blocks: 1,
            max_batch_size: 2,
            backend: BackendKind::Cpu,
            ..Default::default()
        },
        executor,
    )
    .expect("core");
    core.retry_policy.execution_backoff_base = Duration::ZERO;
    core.retry_policy.execution_backoff_max = Duration::ZERO;
    // 40 prompt tokens need three 16-token pages; the arena holds one, so
    // the full-request claim is `Overloaded` on every attempt.
    let oversized =
        managed_test_request(&mut core, model_instance, "oversized", (1..=40).collect());
    let healthy = managed_test_request(&mut core, model_instance, "healthy", vec![1, 2, 3, 4]);
    core.add_request(oversized).expect("add oversized");
    core.add_request(healthy).expect("add healthy");

    let mut outputs = Vec::new();
    for _ in 0..4 {
        outputs.extend(
            core.step()
                .await
                .expect("one row's preparation failure must not fail the engine step"),
        );
    }

    let failed = outputs
        .iter()
        .find(|output| output.request_id == "oversized" && output.is_finished)
        .expect("the oversized request reaches a terminal output");
    assert!(
        failed
            .error
            .as_deref()
            .is_some_and(|error| error.contains("managed KV state could not be prepared")),
        "{:?}",
        failed.error
    );
    assert!(!core.has_request(&"oversized".to_string()));
    let healthy_session = core
        .get_session_key(&"healthy".to_string())
        .expect("healthy session");
    let snapshot = core
        .managed_kv_cache
        .snapshot(
            model_instance,
            &healthy_session,
            crate::kv::CacheDomainId::new(1),
        )
        .expect("healthy request committed its prefill");
    // The healthy row kept running while the oversized one failed: its
    // whole prompt is committed (decode steps may have advanced further).
    assert!(
        snapshot.committed_tokens >= 4,
        "{}",
        snapshot.committed_tokens
    );
}

#[tokio::test]
async fn managed_capacity_preemption_reclaims_only_unpublished_recompute_safe_owner() {
    let model_instance = ModelInstanceId::new(921);
    let executor = UnifiedExecutor::new_for_test(Box::new(ManagedReceiptExecutor));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            enable_chunked_prefill: false,
            enable_preemption: true,
            scheduling_policy: super::super::scheduler::SchedulingPolicy::Priority,
            block_size: 16,
            max_blocks: 1,
            max_batch_size: 1,
            backend: BackendKind::Cpu,
            ..Default::default()
        },
        executor,
    )
    .expect("core");
    core.retry_policy.execution_backoff_base = Duration::ZERO;
    core.retry_policy.execution_backoff_max = Duration::ZERO;
    let mut owner = managed_test_request(
        &mut core,
        model_instance,
        "preemptible-capacity-owner",
        vec![1, 2, 3, 4],
    );
    owner.priority = super::super::Priority::Low;
    core.add_request(owner).expect("add owner");
    core.step().await.expect("commit owner prefill");
    let owner_session = core
        .get_session_key(&"preemptible-capacity-owner".to_string())
        .expect("owner session");

    let mut waiter = managed_test_request(
        &mut core,
        model_instance,
        "priority-capacity-waiter",
        vec![5, 6, 7, 8],
    );
    waiter.priority = super::super::Priority::High;
    core.add_request(waiter).expect("add waiter");
    assert!(core
        .prepare_step()
        .await
        .expect("preemption is controlled backpressure")
        .is_none());
    assert!(core
        .managed_kv_cache
        .snapshot(
            model_instance,
            &owner_session,
            crate::kv::CacheDomainId::new(1),
        )
        .is_none());
    assert_eq!(
        core.managed_kv_cache
            .runtime_snapshot()
            .totals
            .coordinator
            .admission_claimed_pages,
        0
    );

    core.step().await.expect("higher priority waiter runs next");
    let waiter_session = core
        .get_session_key(&"priority-capacity-waiter".to_string())
        .expect("waiter session");
    assert!(core
        .managed_kv_cache
        .snapshot(
            model_instance,
            &waiter_session,
            crate::kv::CacheDomainId::new(1),
        )
        .is_some());
}

#[tokio::test]
async fn managed_kv_backpressure_defers_only_the_blocked_row() {
    let blocked_model = ModelInstanceId::new(93);
    let runnable_model = ModelInstanceId::new(94);
    let executor = UnifiedExecutor::new_for_test(Box::new(ManagedReceiptExecutor));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            enable_chunked_prefill: false,
            block_size: 16,
            max_blocks: 1,
            max_batch_size: 2,
            backend: BackendKind::Cpu,
            ..Default::default()
        },
        executor,
    )
    .expect("core");
    core.retry_policy.execution_backoff_base = Duration::ZERO;
    core.retry_policy.execution_backoff_max = Duration::ZERO;

    let owner = managed_test_request(
        &mut core,
        blocked_model,
        "isolated-capacity-owner",
        vec![1, 2, 3, 4],
    );
    core.add_request(owner).expect("add capacity owner");
    core.step().await.expect("commit capacity owner");
    core.scheduler
        .finish_request(&"isolated-capacity-owner".to_string());

    let blocked = managed_test_request(
        &mut core,
        blocked_model,
        "isolated-capacity-blocked",
        vec![5, 6, 7, 8],
    );
    let runnable = managed_test_request(
        &mut core,
        runnable_model,
        "isolated-capacity-runnable",
        vec![9, 10, 11, 12],
    );
    core.add_request(blocked).expect("add blocked row");
    core.add_request(runnable).expect("add runnable row");

    core.step().await.expect("execute independent runnable row");
    let runnable_session = core
        .get_session_key(&"isolated-capacity-runnable".to_string())
        .expect("runnable session");
    assert!(core
        .managed_kv_cache
        .snapshot(
            runnable_model,
            &runnable_session,
            crate::kv::CacheDomainId::new(1),
        )
        .is_some());
    let blocked_session = core
        .get_session_key(&"isolated-capacity-blocked".to_string())
        .expect("blocked session");
    assert!(
        core.managed_kv_cache
            .snapshot(
                blocked_model,
                &blocked_session,
                crate::kv::CacheDomainId::new(1),
            )
            .is_none(),
        "full-request admission must not register a table for a blocked row"
    );
    assert!(core
        .active_plans
        .values()
        .all(|plan| plan.session != blocked_session));
}

#[tokio::test]
async fn blocked_prefill_does_not_discard_an_already_prepared_decode() {
    let saturated_model = ModelInstanceId::new(95);
    let decode_model = ModelInstanceId::new(96);
    let executor = UnifiedExecutor::new_for_test(Box::new(ManagedReceiptExecutor));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            enable_chunked_prefill: false,
            block_size: 16,
            max_blocks: 1,
            max_batch_size: 2,
            backend: BackendKind::Cpu,
            ..Default::default()
        },
        executor,
    )
    .expect("core");
    core.retry_policy.execution_backoff_base = Duration::ZERO;
    core.retry_policy.execution_backoff_max = Duration::ZERO;

    let owner = managed_test_request(
        &mut core,
        saturated_model,
        "decode-isolation-owner",
        vec![1, 2, 3, 4],
    );
    let decoder = managed_test_request(
        &mut core,
        decode_model,
        "decode-isolation-runnable",
        vec![5, 6, 7, 8],
    );
    core.add_request(owner).expect("add capacity owner");
    core.add_request(decoder).expect("add decoder");
    core.step().await.expect("commit first prefill quantum");
    core.scheduler
        .finish_request(&"decode-isolation-owner".to_string());
    core.step().await.expect("commit second prefill quantum");

    let blocked = managed_test_request(
        &mut core,
        saturated_model,
        "decode-isolation-blocked",
        vec![9, 10, 11, 12],
    );
    core.add_request(blocked).expect("add blocked prefill");
    core.step()
        .await
        .expect("commit decode despite later blocked prefill");

    let decode_session = core
        .get_session_key(&"decode-isolation-runnable".to_string())
        .expect("decode session");
    assert_eq!(
        core.managed_kv_cache
            .snapshot(
                decode_model,
                &decode_session,
                crate::kv::CacheDomainId::new(1),
            )
            .expect("decode snapshot")
            .committed_tokens,
        5
    );
    let blocked_session = core
        .get_session_key(&"decode-isolation-blocked".to_string())
        .expect("blocked session");
    assert!(
        core.managed_kv_cache
            .snapshot(
                saturated_model,
                &blocked_session,
                crate::kv::CacheDomainId::new(1),
            )
            .is_none(),
        "phase isolation must defer the blocked prefill before staging KV state"
    );
    assert_eq!(
        core.scheduler
            .get_status(&"decode-isolation-blocked".to_string()),
        Some(RequestStatus::Waiting)
    );
}

#[tokio::test]
async fn asr_output_carries_prepared_input_duration_when_executor_audio_is_empty() {
    let executor = UnifiedExecutor::new_for_test(Box::new(ImmediateFinishExecutor));
    let mut core =
        EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor).unwrap();
    let variant = ModelVariant::ParakeetTdt06BV3;
    let mut request = EngineCoreRequest::asr_bytes(vec![1, 2, 3]).with_model_variant(variant);
    request.id = "asr-input-duration".to_string();
    core.add_request(request).unwrap();
    // Admission of prepared audio requires the loaded model lease that
    // runtime preparation installs alongside it. Install the decoded audio
    // after admission to model a prepared row whose executor reports none.
    Arc::get_mut(core.requests.get_mut("asr-input-duration").unwrap())
        .expect("the core owns the admitted request")
        .install_prepared_asr_audio(variant, vec![0.0; 36_000], 24_000)
        .unwrap();

    let outputs = core.step().await.unwrap();

    assert_eq!(outputs.len(), 1);
    assert!(outputs[0].error.is_none(), "{:?}", outputs[0].error);
    assert_eq!(outputs[0].audio.duration_secs, 0.0);
    assert_eq!(outputs[0].input_audio_duration_secs, Some(1.5));
}

#[test]
fn unbound_request_does_not_allocate_managed_cache() {
    let executor = UnifiedExecutor::new_for_test(Box::new(ImmediateFinishExecutor));
    let mut core =
        EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor).unwrap();
    let mut request = EngineCoreRequest::chat(vec![ChatMessage {
        role: ChatRole::User,
        content: "unbound".to_string(),
    }])
    .with_model_variant(ModelVariant::Qwen306B);
    request
        .install_chat_execution_preparation(ModelVariant::Qwen306B, vec![1], None, 4096)
        .unwrap();
    request
        .bind_model_instance(ModelInstanceId::new(43))
        .unwrap();

    core.add_request(request).unwrap();
    assert_eq!(core.managed_kv_cache.model_count(), 0);
    assert!(core
        .requests
        .values()
        .next()
        .unwrap()
        .managed_cache_runtime()
        .is_none());
}

#[test]
fn managed_model_cannot_publish_ready_after_failed_load_time_negotiation() {
    let executor = UnifiedExecutor::new_for_test(Box::new(ImmediateFinishExecutor));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            max_blocks: 0,
            ..Default::default()
        },
        executor,
    )
    .unwrap();
    let error = core
        .load_managed_model_cache(
            ModelInstanceId::new(44),
            &crate::kv::InferenceStateCapability::Managed(crate::kv::test_contract()),
            None,
        )
        .unwrap_err();
    assert!(error.to_string().contains("capacity"));
    assert!(core.requests.is_empty());
    assert_eq!(core.managed_kv_cache.model_count(), 0);
}

#[test]
fn core_rejects_unprepared_and_mismatched_chat_accounting() {
    let executor = UnifiedExecutor::new_for_test(Box::new(ImmediateFinishExecutor));
    let mut core =
        EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor).unwrap();
    let mut request = EngineCoreRequest::chat(vec![ChatMessage {
        role: ChatRole::User,
        content: "Hello".to_string(),
    }])
    .with_model_variant(ModelVariant::Qwen306B);
    request.prompt_tokens = vec![11, 22, 33, 44];

    let error = core
        .add_request(request)
        .expect_err("public prompt tokens must not authorize scheduler accounting");
    assert!(error
        .to_string()
        .contains("missing exact model prompt preparation"));

    let mut mismatched = EngineCoreRequest::chat(vec![ChatMessage {
        role: ChatRole::User,
        content: "Hello".to_string(),
    }])
    .with_model_variant(ModelVariant::Qwen306B);
    mismatched
        .install_chat_execution_preparation(ModelVariant::Qwen306B, vec![11, 22], None, 4096)
        .unwrap();
    mismatched.prompt_tokens[0] = 99;
    let error = core
        .add_request(mismatched)
        .expect_err("mutated exact preparation must not reach the scheduler");
    assert!(error
        .to_string()
        .contains("changed after exact prompt preparation"));
}

#[test]
fn core_enforces_exact_chat_context_after_preparation() {
    let executor = UnifiedExecutor::new_for_test(Box::new(ImmediateFinishExecutor));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            max_seq_len: 4,
            ..EngineCoreConfig::default()
        },
        executor,
    )
    .unwrap();

    let mut full = EngineCoreRequest::chat(vec![ChatMessage {
        role: ChatRole::User,
        content: "full".to_string(),
    }])
    .with_model_variant(ModelVariant::Qwen306B);
    full.install_chat_execution_preparation(ModelVariant::Qwen306B, vec![1, 2, 3, 4], None, 4)
        .unwrap();
    assert!(core
        .add_request(full)
        .expect_err("a full context must leave no output allocation")
        .to_string()
        .contains("leaves no output capacity"));

    let mut bounded = EngineCoreRequest::chat(vec![ChatMessage {
        role: ChatRole::User,
        content: "bounded".to_string(),
    }])
    .with_model_variant(ModelVariant::Qwen306B);
    bounded.id = "bounded-context".to_string();
    bounded.params.max_tokens = 100;
    bounded
        .install_chat_execution_preparation(ModelVariant::Qwen306B, vec![1, 2], None, 4)
        .unwrap();
    core.add_request(bounded).unwrap();
    assert_eq!(core.requests["bounded-context"].params.max_tokens, 2);
}

#[tokio::test]
async fn incompatible_physical_batches_commit_in_separate_engine_steps() {
    let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
        Mutex::new(Vec::new()),
    ))));
    let config = EngineCoreConfig {
        max_batch_size: 2,
        max_tokens_per_step: 2,
        min_tokens_per_step: 1,
        enable_chunked_prefill: false,
        enable_adaptive_batching: false,
        block_size: 1,
        max_blocks: 8,
        ..Default::default()
    };
    let mut core = EngineCore::new_with_unified_executor(config, executor).unwrap();
    for request_id in ["fast-batch", "slow-batch"] {
        let mut request = EngineCoreRequest::tts(request_id);
        request.id = request_id.to_string();
        request.prompt_tokens = vec![1];
        core.add_request(request).unwrap();
    }

    let mut prefill_ms = HashMap::new();
    for _ in 0..=super::super::scheduler::MAX_DECODE_ONLY_STEPS_WITH_WAITING_FULL_PREFILL + 1 {
        let prepared = core
            .prepare_step()
            .await
            .unwrap()
            .expect("prepared physical batch");
        let mut executed = core.execute_prepared_with_progress(prepared).await.unwrap();
        assert_eq!(executed.batches.len(), 1);
        let batch = &mut executed.batches[0];
        let request_id = batch.results[0].session.request_id.as_str();
        let elapsed = match request_id {
            "fast-batch" => Duration::from_millis(7),
            "slow-batch" => Duration::from_millis(31),
            other => panic!("unexpected batch row {other}"),
        };
        batch.report.elapsed = elapsed;
        for row in &mut batch.report.rows {
            row.execution.elapsed = elapsed;
        }
        let committed = core.commit_step(executed).await.unwrap();
        for output in &committed.outputs {
            prefill_ms.insert(
                output.request_id.clone(),
                output
                    .latency_breakdown
                    .as_ref()
                    .expect("latency breakdown")
                    .prefill_ms,
            );
        }
        if prefill_ms.contains_key("slow-batch") {
            break;
        }
    }

    assert!((prefill_ms["fast-batch"] - 7.0).abs() < 0.001);
    assert!((prefill_ms["slow-batch"] - 31.0).abs() < 0.001);
}

#[tokio::test]
async fn shadow_mode_owns_multiple_candidate_tickets_but_executes_them_serially() {
    let active = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let max_active = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    struct ShadowExecutor {
        active: Arc<std::sync::atomic::AtomicUsize>,
        max_active: Arc<std::sync::atomic::AtomicUsize>,
    }
    impl ModelExecutor for ShadowExecutor {
        fn execute_physical_batch(
            &self,
            execution: super::super::PhysicalBatchExecution<'_>,
        ) -> super::super::PhysicalDispatchResult {
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_active.fetch_max(active, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(10));
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(execution
                .scheduled
                .iter()
                .map(|scheduled| {
                    ExecutorStepResult::new(
                        scheduled,
                        ExecutorOutput::terminal(scheduled.request_id.clone()),
                    )
                    .with_dispatch(execution.expected_dispatch())
                })
                .collect())
        }

        fn execute_prefill(
            &self,
            _requests: &[&EngineCoreRequest],
            _scheduled: &[super::super::scheduler::ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            unreachable!("physical boundary must own dispatch")
        }

        fn execute_decode(
            &self,
            _requests: &[&EngineCoreRequest],
            _scheduled: &[super::super::scheduler::ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            unreachable!("physical boundary must own dispatch")
        }

        fn is_ready(&self) -> bool {
            true
        }

        fn initialize(&mut self) -> Result<()> {
            Ok(())
        }

        fn shutdown(&mut self) -> Result<()> {
            Ok(())
        }
    }

    let executor = UnifiedExecutor::new_for_test(Box::new(ShadowExecutor {
        active,
        max_active: max_active.clone(),
    }));
    let config = EngineCoreConfig {
        max_batch_size: 2,
        max_tokens_per_step: 2,
        min_tokens_per_step: 1,
        enable_chunked_prefill: false,
        enable_adaptive_batching: false,
        block_size: 1,
        max_blocks: 8,
        physical_execution_mode: crate::config::PhysicalExecutionMode::Shadow,
        max_physical_in_flight: crate::config::PhysicalInFlightLimit::new(2).unwrap(),
        ..Default::default()
    };
    let mut core = EngineCore::new_with_unified_executor(config, executor).unwrap();
    for request_id in ["shadow-a", "shadow-b"] {
        let mut request = EngineCoreRequest::tts(request_id);
        request.id = request_id.to_string();
        request.prompt_tokens = vec![1];
        core.add_request(request).unwrap();
    }

    let prepared = core
        .prepare_step()
        .await
        .unwrap()
        .expect("shadow candidate tickets");
    assert_eq!(core.in_flight_dispatches.len(), 2);
    let executed = core.execute_prepared_with_progress(prepared).await.unwrap();
    assert_eq!(executed.batches.len(), 2);
    assert_eq!(max_active.load(Ordering::SeqCst), 1);
    core.commit_step(executed).await.unwrap();
    assert!(core.in_flight_dispatches.is_empty());
}

#[tokio::test]
async fn in_flight_quantum_blocks_duplicate_preparation_and_clears_on_commit_and_rollback() {
    let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
        Mutex::new(Vec::new()),
    ))));
    let config = EngineCoreConfig {
        max_batch_size: 1,
        max_tokens_per_step: 1,
        min_tokens_per_step: 1,
        enable_chunked_prefill: false,
        enable_adaptive_batching: false,
        block_size: 1,
        max_blocks: 8,
        ..Default::default()
    };
    let mut core = EngineCore::new_with_unified_executor(config, executor).unwrap();
    let mut request = EngineCoreRequest::tts("one physical quantum at a time");
    request.id = "ticket-serial".to_string();
    request.prompt_tokens = vec![1];
    core.add_request(request).unwrap();

    let prefill = core
        .prepare_step()
        .await
        .unwrap()
        .expect("prefill dispatch");
    assert_eq!(core.in_flight_dispatches.len(), 1);
    assert_eq!(core.active_plans.len(), 1);
    assert_eq!(
        core.in_flight_dispatches.values().next().unwrap().phase,
        ExecutionPhase::Prefill
    );
    assert!(core.prepare_step().await.unwrap().is_none());
    assert_eq!(core.in_flight_dispatches.len(), 1);
    assert_eq!(core.active_plans.len(), 1);

    let executed = core.execute_prepared_with_progress(prefill).await.unwrap();
    core.commit_step(executed).await.unwrap();
    assert!(core.in_flight_dispatches.is_empty());
    assert!(core.active_plans.is_empty());
    assert!(core.active_stream_batches.is_empty());

    let decode = core.prepare_step().await.unwrap().expect("decode dispatch");
    assert_eq!(core.in_flight_dispatches.len(), 1);
    assert_eq!(core.active_plans.len(), 1);
    assert_eq!(
        core.in_flight_dispatches.values().next().unwrap().phase,
        ExecutionPhase::Decode
    );
    assert!(core.prepare_step().await.unwrap().is_none());
    assert_eq!(core.in_flight_dispatches.len(), 1);
    assert_eq!(core.active_plans.len(), 1);

    assert_eq!(core.rollback_all_in_flight_dispatches(), 1);
    assert!(core.in_flight_dispatches.is_empty());
    assert!(core.active_plans.is_empty());
    assert!(core.active_managed_cache.is_empty());
    assert!(core.active_stream_batches.is_empty());
    drop(decode);
}

/// Chat rows decode one token per call; the long-form ASR row transcribes
/// `windows` audio windows, one per dispatch when `by_chunk`, all of
/// them in one dispatch otherwise. Every window and chat call is logged.
struct LongFormMockExecutor {
    windows: usize,
    by_chunk: bool,
    done: Mutex<usize>,
    log: Arc<Mutex<Vec<&'static str>>>,
}

impl ModelExecutor for LongFormMockExecutor {
    fn execution_profile(&self, request: &EngineCoreRequest) -> Option<ExecutionProfile> {
        let mode = if request.task_type == TaskType::ASR {
            ExecutionMode::Atomic
        } else {
            ExecutionMode::Sequence
        };
        Some(ExecutionProfile::fail_closed(
            BackendKind::Cpu,
            request.model_variant,
            mode,
        ))
    }

    fn execute_prefill(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        Ok(scheduled
            .iter()
            .map(|row| {
                let asr = requests
                    .iter()
                    .any(|r| r.id == row.request_id && r.task_type == TaskType::ASR);
                if !asr {
                    self.log.lock().unwrap().push("chat-prefill");
                    return ExecutorStepResult::new(
                        row,
                        MockExecutor::build_outputs(std::slice::from_ref(row)).remove(0),
                    );
                }
                let mut done = self.done.lock().unwrap();
                let run = if self.by_chunk {
                    1
                } else {
                    self.windows - *done
                };
                for _ in 0..run {
                    self.log.lock().unwrap().push("asr-window");
                }
                *done += run;
                let mut output = ExecutorOutput::terminal(row.request_id.clone());
                if *done < self.windows {
                    output.finished = false;
                    return ExecutorStepResult::from_session(
                        row,
                        ModelSessionResult::yielded(
                            output,
                            super::super::YieldReason::QuantumExhausted,
                        ),
                    );
                }
                output.text = Some("transcript".into());
                ExecutorStepResult::new(row, output)
            })
            .collect())
    }

    fn execute_decode(
        &self,
        _requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        self.log.lock().unwrap().push("chat");
        Ok(wrap_outputs(
            scheduled,
            MockExecutor::build_outputs(scheduled),
        ))
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn initialize(&mut self) -> Result<()> {
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }

    fn cleanup_request(&self, _request_id: &str) -> super::super::executor::CacheReleaseReport {
        super::super::executor::CacheReleaseReport::confirmed(1)
    }
}

/// The most long-form windows the device ran between two chat steps
/// while a long-form transcription ran beside chat streams, and whether
/// a chat request that arrived during the transcription began before it
/// ended.
async fn long_form_beside_chat(by_chunk: bool) -> (usize, bool) {
    let windows = 12;
    let log = Arc::new(Mutex::new(Vec::new()));
    let executor = UnifiedExecutor::new_for_test(Box::new(LongFormMockExecutor {
        windows,
        by_chunk,
        done: Mutex::new(0),
        log: log.clone(),
    }));
    let config = EngineCoreConfig {
        max_batch_size: 4,
        enable_chunked_prefill: false,
        enable_adaptive_batching: false,
        ..Default::default()
    };
    let mut core = EngineCore::new_with_unified_executor(config, executor).unwrap();
    for id in ["chat-a", "chat-b"] {
        let mut request = EngineCoreRequest::tts(id);
        request.id = id.to_string();
        request.prompt_tokens = vec![1];
        request.params.max_tokens = 1_000;
        request
            .bind_model_instance(ModelInstanceId::new(1))
            .unwrap();
        core.add_request(request).unwrap();
    }
    for _ in 0..4 {
        core.step().await.unwrap();
    }

    let variant = ModelVariant::WhisperLargeV3Turbo;
    let model_instance = ModelInstanceId::new(2);
    let profile =
        ExecutionProfile::fail_closed(BackendKind::Cpu, Some(variant), ExecutionMode::Atomic);
    let mut stage = super::super::StageDescriptor::from_execution_profile(
        StageId::new(3),
        "asr.long_form",
        &profile,
        NativeBatchMode::None,
    );
    stage.selector = super::super::StageWorkSelector::Atomic;
    if by_chunk {
        stage.progress = super::super::StageProgressKind::Iterative;
        stage.membership_safe_point = super::super::MembershipSafePoint::QuantumBoundary;
    }
    let mut asr = EngineCoreRequest::asr("long-form").with_model_variant(variant);
    asr.id = "long-form".to_string();
    asr.prompt_tokens = vec![1];
    asr.bind_execution_adapter(super::super::ExecutionAdapterBinding {
        execution_group_id: super::super::ExecutionGroupId::new(1),
        model_instance_id: model_instance,
        adapter_instance_id: super::super::AdapterInstanceId::new(3),
        adapter_abi_revision: super::super::AdapterAbiRevision::new(1),
        model_variant: variant,
        capability_id: "asr".to_string(),
        stages: Arc::from([stage]),
    })
    .unwrap();
    core.add_request(asr).unwrap();
    log.lock().unwrap().clear();

    let (mut transcribed, mut late_added) = (false, false);
    for _ in 0..1_000 {
        // A chat request arrives once the transcription has begun.
        if !late_added && log.lock().unwrap().contains(&"asr-window") {
            late_added = true;
            log.lock().unwrap().push("chat-late-arrived");
            let mut late = EngineCoreRequest::tts("chat-late");
            late.id = "chat-late".to_string();
            late.prompt_tokens = vec![1];
            late.params.max_tokens = 1_000;
            late.bind_model_instance(ModelInstanceId::new(1)).unwrap();
            core.add_request(late).unwrap();
        }
        for output in core.step().await.unwrap() {
            if output.request_id == "long-form" && output.is_finished {
                assert!(output.error.is_none(), "{:?}", output.error);
                transcribed = true;
            }
        }
        if transcribed {
            break;
        }
    }
    assert!(transcribed, "the long-form transcription finished");
    let log = log.lock().unwrap();
    assert_eq!(
        log.iter().filter(|event| **event == "asr-window").count(),
        windows
    );
    let last_window = log.iter().rposition(|event| *event == "asr-window");
    let arrived = log
        .iter()
        .position(|event| *event == "chat-late-arrived")
        .expect("the late chat request arrived");
    let late_prefill = log
        .iter()
        .skip(arrived)
        .position(|event| *event == "chat-prefill")
        .map(|offset| arrived + offset);
    let model_calls = log
        .iter()
        .filter(|event| **event != "chat-late-arrived")
        .collect::<Vec<_>>();
    let windows = model_calls
        .split(|event| **event != "asr-window")
        .map(<[_]>::len)
        .max()
        .unwrap_or_default();
    let late_chat_began = late_prefill
        .zip(last_window)
        .is_some_and(|(prefill, window)| prefill < window);
    (windows, late_chat_began)
}

/// Phase 3c exit test: a long-form transcription beside chat streams
/// holds the device for at most one window between two chat steps, so
/// the streams keep advancing with bounded inter-token gaps.
#[tokio::test]
async fn long_form_asr_beside_chat_holds_the_device_one_window_at_a_time() {
    let (windows, late_chat_began) = long_form_beside_chat(true).await;
    assert_eq!(windows, 1, "windows between two chat model calls");
    assert!(
        late_chat_began,
        "a chat request arriving mid-transcription waited for it"
    );
}

#[tokio::test]
async fn in_flight_fence_holds_only_its_own_lane() {
    let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
        Mutex::new(Vec::new()),
    ))));
    let config = EngineCoreConfig {
        max_batch_size: 1,
        max_tokens_per_step: 1,
        min_tokens_per_step: 1,
        enable_chunked_prefill: false,
        enable_adaptive_batching: false,
        block_size: 1,
        max_blocks: 8,
        ..Default::default()
    };
    let mut core = EngineCore::new_with_unified_executor(config, executor).unwrap();
    for (id, model) in [("lane-a", 501), ("lane-b", 502)] {
        let mut request = EngineCoreRequest::tts(id);
        request.id = id.to_string();
        request.prompt_tokens = vec![1];
        request
            .bind_model_instance(ModelInstanceId::new(model))
            .unwrap();
        core.add_request(request).unwrap();
    }

    let first = core.prepare_step().await.unwrap().expect("first lane");
    assert_eq!(core.in_flight_dispatches.len(), 1);
    // The other lane prepares while the first lane's ticket is in flight;
    // the first lane is not polled again.
    let second = core.prepare_step().await.unwrap().expect("second lane");
    assert_eq!(core.in_flight_dispatches.len(), 2);
    let lanes = core
        .in_flight_dispatches
        .values()
        .flat_map(|dispatch| dispatch.lanes.clone())
        .collect::<HashSet<_>>();
    assert_eq!(
        lanes,
        HashSet::from([
            Some(ModelInstanceId::new(501)),
            Some(ModelInstanceId::new(502)),
        ])
    );
    assert!(core.prepare_step().await.unwrap().is_none());

    for prepared in [first, second] {
        let executed = core.execute_prepared_with_progress(prepared).await.unwrap();
        core.commit_step(executed).await.unwrap();
    }
    assert!(core.in_flight_dispatches.is_empty());
    assert!(core.active_plans.is_empty());
}

#[tokio::test]
async fn formation_stages_transactions_only_for_rows_the_step_launches() {
    let model_instance = ModelInstanceId::new(993);
    let executor = UnifiedExecutor::new_for_test(Box::new(ManagedReceiptExecutor));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            max_batch_size: 4,
            enable_chunked_prefill: false,
            enable_adaptive_batching: false,
            block_size: 16,
            max_blocks: 8,
            backend: BackendKind::Cpu,
            ..Default::default()
        },
        executor,
    )
    .expect("core");
    let ids = ["launch-a", "launch-b", "launch-c", "launch-d"];
    for (id, token) in ids.into_iter().zip(11..) {
        let request = managed_test_request(&mut core, model_instance, id, vec![token]);
        core.add_request(request).expect("add managed request");
    }

    // Four scalar rows are scheduled, and a serial step launches one.
    let prepared = core
        .prepare_step()
        .await
        .expect("prepare")
        .expect("one launch");
    assert_eq!(prepared.recovery().batch_ids().len(), 1);
    assert_eq!(core.active_managed_cache.len(), 1);
    let snapshot = core.managed_kv_cache.runtime_snapshot();
    assert_eq!(snapshot.totals.coordinator.active_transactions, 1);
    assert_eq!(
        snapshot.counters.transaction_aborts, 0,
        "rows the step cannot launch must not stage a managed transaction"
    );
    drop(prepared);
}

#[test]
fn automatic_staged_rows_do_not_cap_a_model_fitted_staging_width() {
    let contract = crate::kv::test_contract();
    let core_with = |automatic| {
        let executor = UnifiedExecutor::new_for_test(Box::new(ImmediateFinishExecutor));
        let mut config = EngineCoreConfig {
            max_retained_sequences: 8,
            max_staged_transactions: 2,
            backend: BackendKind::Cpu,
            ..Default::default()
        };
        config.automatic_row_limits.staged_transactions = automatic;
        EngineCore::new_with_unified_executor(config, executor).unwrap()
    };

    let load = |core: &mut EngineCore, model, retained, staged| {
        core.load_managed_model_state_with_row_limits(
            ModelInstanceId::new(model),
            &contract,
            None,
            retained,
            staged,
        )
    };

    assert!(load(&mut core_with(false), 46, 8, 6).is_err());
    let mut automatic = core_with(true);
    load(&mut automatic, 47, 8, 6)
        .expect("an automatic staged limit must not cap a model-fitted width");
    assert!(load(&mut automatic, 48, 9, 6).is_err());
}

#[tokio::test]
async fn exact_prepared_recovery_rolls_back_only_its_cache_plan_and_ticket() {
    let model_instance = ModelInstanceId::new(992);
    let executor = UnifiedExecutor::new_for_test(Box::new(ManagedReceiptExecutor));
    let mut core = EngineCore::new_with_unified_executor(
        EngineCoreConfig {
            max_batch_size: 1,
            max_tokens_per_step: 1,
            min_tokens_per_step: 1,
            enable_chunked_prefill: false,
            enable_adaptive_batching: false,
            block_size: 16,
            max_blocks: 8,
            backend: BackendKind::Cpu,
            ..Default::default()
        },
        executor,
    )
    .expect("core");
    for (id, token) in [("recovery-first", 11), ("recovery-later", 22)] {
        let request = managed_test_request(&mut core, model_instance, id, vec![token]);
        core.add_request(request).expect("add managed request");
    }

    let first = core
        .prepare_step()
        .await
        .expect("prepare first")
        .expect("first ticket");
    let first_recovery = first.recovery();
    assert_eq!(first_recovery.batch_ids().len(), 1);
    let first_batch_id = first_recovery.batch_ids()[0];
    let first_ticket = core
        .in_flight_dispatches
        .remove(&first_batch_id)
        .expect("first ticket registration");
    let first_plan_id = first_ticket.scheduled[0].plan_id;

    // Simulate recovery reacquiring the core after another prepared step
    // has registered a newer ticket.
    let later = core
        .prepare_step()
        .await
        .expect("prepare later")
        .expect("later ticket");
    let later_recovery = later.recovery();
    assert_eq!(later_recovery.batch_ids().len(), 1);
    let later_batch_id = later_recovery.batch_ids()[0];
    let later_plan_id = core.in_flight_dispatches[&later_batch_id].scheduled[0].plan_id;
    core.in_flight_dispatches
        .insert(first_batch_id, first_ticket);

    assert_eq!(core.in_flight_dispatches.len(), 2);
    assert_eq!(core.active_plans.len(), 2);
    assert_eq!(core.active_managed_cache.len(), 2);
    assert_eq!(
        core.managed_kv_cache
            .runtime_snapshot()
            .totals
            .coordinator
            .active_transactions,
        2
    );

    assert_eq!(
        core.rollback_in_flight_dispatches(first_recovery.batch_ids()),
        1
    );
    assert!(!core.in_flight_dispatches.contains_key(&first_batch_id));
    assert!(!core.active_plans.contains_key(&first_plan_id));
    assert!(!core.active_managed_cache.contains_key(&first_plan_id));
    assert!(core.in_flight_dispatches.contains_key(&later_batch_id));
    assert!(core.active_plans.contains_key(&later_plan_id));
    assert!(core.active_managed_cache.contains_key(&later_plan_id));
    assert_eq!(
        core.managed_kv_cache
            .runtime_snapshot()
            .totals
            .coordinator
            .active_transactions,
        1,
        "exact recovery must leave the later managed transaction active"
    );

    assert_eq!(
        core.rollback_in_flight_dispatches(later_recovery.batch_ids()),
        1
    );
    assert!(core.in_flight_dispatches.is_empty());
    assert!(core.active_plans.is_empty());
    assert!(core.active_managed_cache.is_empty());
    drop((first, later));
}

#[tokio::test]
async fn stale_and_duplicate_physical_completion_cannot_cross_the_dispatch_registry() {
    let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
        Mutex::new(Vec::new()),
    ))));
    let config = EngineCoreConfig {
        max_batch_size: 1,
        max_tokens_per_step: 1,
        min_tokens_per_step: 1,
        enable_chunked_prefill: false,
        enable_adaptive_batching: false,
        block_size: 1,
        max_blocks: 8,
        ..Default::default()
    };
    let mut core = EngineCore::new_with_unified_executor(config, executor).unwrap();
    let mut request = EngineCoreRequest::tts("completion identity fence");
    request.id = "ticket-completion".to_string();
    request.prompt_tokens = vec![1];
    core.add_request(request).unwrap();

    let prepared = core
        .prepare_step()
        .await
        .unwrap()
        .expect("registered dispatch");
    let registered_batch_id = *core.in_flight_dispatches.keys().next().unwrap();
    let exact = core.execute_prepared_with_progress(prepared).await.unwrap();
    let duplicate = exact.clone();
    let mut stale = exact.clone();
    stale.batches[0].physical_batch.batch_id = BatchId::new(u64::MAX);
    stale.batches[0].report.batch_id = BatchId::new(u64::MAX);

    let stale_commit = core.commit_step(stale).await.unwrap();
    assert!(stale_commit.outputs.is_empty());
    assert!(core.in_flight_dispatches.contains_key(&registered_batch_id));
    assert_eq!(core.active_plans.len(), 1);

    core.commit_step(exact).await.unwrap();
    assert!(core.in_flight_dispatches.is_empty());
    assert!(core.active_plans.is_empty());

    let duplicate_commit = core.commit_step(duplicate).await.unwrap();
    assert!(duplicate_commit.outputs.is_empty());
    assert!(core.in_flight_dispatches.is_empty());
    assert!(core.active_plans.is_empty());
}

#[tokio::test]
async fn latency_breakdown_reports_time_between_runtime_admission_and_insertion() {
    let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
        Mutex::new(Vec::new()),
    ))));
    let config = EngineCoreConfig {
        enable_chunked_prefill: false,
        enable_adaptive_batching: false,
        block_size: 1,
        max_blocks: 8,
        ..Default::default()
    };
    let mut core = EngineCore::new_with_unified_executor(config, executor).unwrap();
    let mut request = EngineCoreRequest::tts("pre-engine timing");
    request.id = "pre-engine".into();
    request.prompt_tokens = vec![1];
    request.runtime_admitted_at = Some(Instant::now() - Duration::from_millis(250));
    core.add_request(request).unwrap();
    core.step().await.unwrap();
    let prepared = core.prepare_step().await.unwrap().unwrap();
    let executed = core.execute_prepared_with_progress(prepared).await.unwrap();
    let committed = core.commit_step(executed).await.unwrap();
    let timing = committed.outputs[0].latency_breakdown.as_ref().unwrap();
    let pre_engine_ms = timing.pre_engine_ms.expect("runtime admission was stamped");
    // Measured at insertion, so it cannot include the engine's own time.
    assert!(pre_engine_ms >= 250.0, "{pre_engine_ms}");
    assert!(pre_engine_ms < 250.0 + 5_000.0, "{pre_engine_ms}");
}

#[tokio::test]
async fn decode_timing_commits_once_and_ignores_duplicate_completion() {
    let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
        Mutex::new(Vec::new()),
    ))));
    let config = EngineCoreConfig {
        enable_chunked_prefill: false,
        enable_adaptive_batching: false,
        block_size: 1,
        max_blocks: 8,
        ..Default::default()
    };
    let mut core = EngineCore::new_with_unified_executor(config, executor).unwrap();
    let mut request = EngineCoreRequest::tts("commit timing");
    request.id = "commit-timing".into();
    request.prompt_tokens = vec![1];
    core.add_request(request).unwrap();
    core.step().await.unwrap(); // prefill commits no output token in this fixture
    let prepared = core.prepare_step().await.unwrap().unwrap();
    let executed = core.execute_prepared_with_progress(prepared).await.unwrap();
    let duplicate = executed.clone();
    let committed = core.commit_step(executed).await.unwrap();
    let timing = committed.outputs[0].latency_breakdown.as_ref().unwrap();
    assert_eq!(timing.decode_tokens, Some(1));
    assert!(timing.decode_wall_ms.unwrap() > 0.0);
    assert!(timing.decode_wall_ms.unwrap() <= timing.total_ms);
    // Token stats carry the measured prefill, not a placeholder.
    assert!(timing.prefill_ms > 0.0);
    assert_eq!(
        committed.outputs[0].token_stats.prefill_time_ms,
        timing.prefill_ms as f32
    );
    // A direct engine caller has no runtime admission before insertion.
    assert_eq!(timing.pre_engine_ms, None);
    assert!(core
        .commit_step(duplicate)
        .await
        .unwrap()
        .outputs
        .is_empty());
    assert_eq!(core.request_phase_timings["commit-timing"].decode_tokens, 1);
}

#[tokio::test]
async fn test_step_preserves_optional_executor_phase_timings() {
    let executor = UnifiedExecutor::new_for_test(Box::new(PhaseTimingExecutor));
    let config = EngineCoreConfig {
        enable_chunked_prefill: false,
        block_size: 1,
        max_blocks: 8,
        ..Default::default()
    };
    let mut core = EngineCore::new_with_unified_executor(config, executor).unwrap();

    let mut request = EngineCoreRequest::tts("phase timing");
    request.id = "phase-req".to_string();
    request.prompt_tokens = vec![1, 2, 3];

    core.add_request(request).unwrap();
    let outputs = core.step().await.unwrap();

    let latency = outputs[0]
        .latency_breakdown
        .as_ref()
        .expect("latency breakdown");
    assert_eq!(latency.media_decode_ms, Some(12.5));
    assert_eq!(latency.sampling_ms, Some(1.25));
    assert!(
        latency.prefill_ms >= 0.0,
        "scheduler prefill timing remains available"
    );
    assert_eq!(latency.decode_ms, 0.0);
}

#[test]
fn load_owned_composite_survives_weak_request_upgrade_until_ordered_unload() {
    let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
        Mutex::new(Vec::new()),
    ))));
    let config = EngineCoreConfig {
        block_size: 16,
        max_blocks: 8,
        max_retained_sequences: 2,
        max_staged_transactions: 2,
        ..Default::default()
    };
    let mut core = EngineCore::new_with_unified_executor(config, executor).unwrap();
    let model = ModelInstanceId::new(76);
    let loaded = core
        .load_composite_retained_state(
            model,
            &crate::engine::cache::composite::tests::contract(),
            crate::kv::v2::StateDomainId::new(2),
            Some(32),
            0,
        )
        .unwrap();
    let request_weak = Arc::downgrade(&loaded);
    drop(loaded);
    let request_upgrade = request_weak
        .upgrade()
        .expect("Core load owner keeps weak request identity live");
    assert!(core.unload_managed_model_cache(model).is_err());
    drop(request_upgrade);
    assert!(core.unload_managed_model_cache(model).unwrap());
    assert!(request_weak.upgrade().is_none());
}
