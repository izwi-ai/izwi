use crate::backends::state::PhysicalStateTransactionId;
use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use crate::models::architectures::vibevoice::asr::VibeVoiceAsrRetainedTokenizerQuantum;
use crate::models::architectures::vibevoice::asr::{
    VibeVoiceAsrPreparedTokenizerSpan, VibeVoiceAsrRetainedPrefillBatchRow,
};
use crate::models::architectures::whisper::asr::WhisperTerminalTransition;
use crate::models::registry::{
    NativeAsrDecodeCheckpoint, NativeAsrDecodeStep, NativeAsrGenerationOptions,
};
use crate::runtime::granite_auto_asr_max_tokens_for_duration;
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;

use super::super::request::{EngineCoreRequest, RealtimeAsrOperationPayload};
use super::super::scheduler::ScheduledRequest;
use super::super::types::AudioOutput;
use super::super::{SequenceRestartReason, SessionKey};
use super::audio::{decode_request_audio_with_rate, AsrChunkTranscription};
use super::state::{ActiveAsrDecode, ActiveVoxtralRealtime};
use super::{
    ExecutorOutput, ExecutorPhaseTiming, ExecutorStateLease, ModelSessionResult, NativeExecutor,
};

const MAX_ASR_NEW_TOKENS: usize = 512;
const GRANITE_ASR_PREFIX_REPLAY_WORDS: usize = 0;
const GRANITE_ASR_PREFIX_REPLAY_WORDS_MAX: usize = 240;

enum AsrExecutionAudio {
    Prepared(Arc<[f32]>),
    Decoded(Vec<f32>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WhisperTerminalAction {
    Publish,
    Restart,
}

fn whisper_prefill_boundary_step(last_tokens_generated: usize) -> NativeAsrDecodeStep {
    NativeAsrDecodeStep {
        delta: String::new(),
        text: String::new(),
        tokens_generated: last_tokens_generated,
        finished: false,
    }
}

fn vibevoice_prefill_boundary_step(
    complete: bool,
    staged: Option<NativeAsrDecodeStep>,
    last_tokens_generated: usize,
) -> Result<NativeAsrDecodeStep> {
    match (complete, staged) {
        (true, Some(step)) => Ok(step),
        (true, None) => Err(Error::InferenceError(
            "final VibeVoice ASR prefill produced no staged decoder output".into(),
        )),
        (false, None) => Ok(NativeAsrDecodeStep {
            delta: String::new(),
            text: String::new(),
            tokens_generated: last_tokens_generated,
            finished: false,
        }),
        (false, Some(_)) => Err(Error::InferenceError(
            "non-final VibeVoice ASR prefill produced premature decoder output".into(),
        )),
    }
}

fn apply_whisper_terminal_transition(
    step: &mut NativeAsrDecodeStep,
    transition: WhisperTerminalTransition,
    reservation_generation: crate::engine::ManagedSessionGeneration,
) -> Result<WhisperTerminalAction> {
    match transition {
        WhisperTerminalTransition::Accept { text, .. } => {
            step.delta = text.clone();
            step.text = text;
            Ok(WhisperTerminalAction::Publish)
        }
        WhisperTerminalTransition::SkipNoSpeech { .. } => {
            step.delta.clear();
            step.text.clear();
            Ok(WhisperTerminalAction::Publish)
        }
        WhisperTerminalTransition::RetryRequired {
            next_temperature,
            reasons,
            expected_generation,
            new_generation,
        } => {
            let reservation_generation = reservation_generation.get();
            let required_new_generation = expected_generation.checked_add(1).ok_or_else(|| {
                Error::InferenceError("Whisper fallback session generation overflowed".into())
            })?;
            if expected_generation != reservation_generation
                || new_generation != required_new_generation
            {
                return Err(Error::InferenceError(format!(
                    "Whisper fallback generation {expected_generation}->{new_generation} does not continue authenticated reservation generation {reservation_generation} for temperature {next_temperature} ({})",
                    reasons.join(",")
                )));
            }
            step.delta.clear();
            step.text.clear();
            Ok(WhisperTerminalAction::Restart)
        }
    }
}

fn resolve_whisper_terminal_action(
    request: &EngineCoreRequest,
    step: &mut NativeAsrDecodeStep,
    reservation_generation: crate::engine::ManagedSessionGeneration,
    resolve: impl FnOnce() -> Result<WhisperTerminalTransition>,
) -> Result<Option<WhisperTerminalAction>> {
    if request.is_cancelled() {
        return Ok(None);
    }
    let transition = resolve()?;
    if request.is_cancelled() {
        return Ok(None);
    }
    apply_whisper_terminal_transition(step, transition, reservation_generation).map(Some)
}

fn begins_whisper_managed_generation(
    scheduled: &ScheduledRequest,
    generation: crate::engine::ManagedSessionGeneration,
) -> bool {
    scheduled.is_prefill
        && scheduled.num_computed_tokens == 0
        && generation != crate::engine::ManagedSessionGeneration::INITIAL
}

impl AsrExecutionAudio {
    fn samples(&self) -> &[f32] {
        match self {
            Self::Prepared(samples) => samples,
            Self::Decoded(samples) => samples,
        }
    }
}

fn resolve_asr_execution_audio(
    request: &EngineCoreRequest,
    family: ModelFamily,
    decode: impl FnOnce() -> Result<(Vec<f32>, u32)>,
) -> Result<(AsrExecutionAudio, u32, f64)> {
    if matches!(
        family,
        ModelFamily::Qwen3Asr
            | ModelFamily::WhisperAsr
            | ModelFamily::VibeVoiceAsr
            | ModelFamily::GraniteSpeechAsr
    ) {
        if let Some((samples, sample_rate)) = request.prepared_asr_audio_for_executor()? {
            return Ok((AsrExecutionAudio::Prepared(samples), sample_rate, 0.0));
        }
        if matches!(
            family,
            ModelFamily::Qwen3Asr | ModelFamily::VibeVoiceAsr | ModelFamily::GraniteSpeechAsr
        ) {
            return Err(Error::InferenceError(format!(
                "{} execution lost its prepared decoded-audio artifact",
                match family {
                    ModelFamily::Qwen3Asr => "Qwen3 ASR",
                    ModelFamily::VibeVoiceAsr => "VibeVoice ASR",
                    ModelFamily::GraniteSpeechAsr => "Granite Speech ASR",
                    _ => unreachable!("guarded prepared-audio family"),
                }
            )));
        }
    }
    let started = Instant::now();
    let (samples, sample_rate) = decode()?;
    Ok((
        AsrExecutionAudio::Decoded(samples),
        sample_rate,
        started.elapsed().as_secs_f64() * 1000.0,
    ))
}

fn begins_resumable_asr_prefill_state(
    scheduled: &ScheduledRequest,
    resumable_prefill: bool,
) -> bool {
    scheduled.is_prefill && resumable_prefill && scheduled.num_computed_tokens == 0
}

fn resumable_asr_prefill_span(
    scheduled: &ScheduledRequest,
    prompt_tokens: usize,
) -> Result<(usize, usize)> {
    let start = scheduled.num_computed_tokens;
    let end = start.checked_add(scheduled.num_tokens).ok_or_else(|| {
        Error::InvalidInput("resumable ASR prefill span overflowed prompt accounting".into())
    })?;
    let crate::engine::WorkUnit::SequenceStep { phase, input, .. } = &scheduled.work else {
        return Err(Error::InvalidInput(
            "resumable ASR prefill requires sequence-prefill work".into(),
        ));
    };
    if *phase != crate::engine::SequencePhase::Prefill
        || input.start != start
        || input.end != end
        || start >= end
        || end > prompt_tokens
    {
        return Err(Error::InvalidInput(format!(
            "resumable ASR prefill work [{}, {}) disagrees with scheduler span [{start}, {end}) for {prompt_tokens} prompt tokens",
            input.start, input.end
        )));
    }
    Ok((start, end))
}

/// Run one physical ASR model call and observe cooperative cancellation before
/// any caller can publish output or detach the transaction checkpoint. Native
/// device calls are not preemptible, so this post-call edge is the first safe
/// point at which a cancellation that arrived during the call can be honored.
fn run_asr_model_call<T>(
    request: &EngineCoreRequest,
    run: impl FnOnce() -> Result<T>,
) -> Result<Option<T>> {
    let output = run()?;
    Ok((!request.is_cancelled()).then_some(output))
}

fn validate_continuous_asr_batch_shape(scheduled: &[ScheduledRequest]) -> Result<()> {
    if scheduled.is_empty()
        || scheduled
            .iter()
            .any(|scheduled| scheduled.is_prefill || scheduled.num_tokens != 1)
    {
        return Err(Error::InvalidInput(
            "continuous ASR execution requires one decode token per row".to_string(),
        ));
    }
    Ok(())
}

fn late_cancelled_batch_row(cancelled: bool, checkpoint_armed: bool) -> bool {
    cancelled && checkpoint_armed
}

struct ContinuousAsrStateBatch<'a> {
    rows: Vec<(
        usize,
        SessionKey,
        ExecutorStateLease<'a, ActiveAsrDecode>,
        Option<(NativeAsrDecodeCheckpoint, usize, usize)>,
    )>,
    armed: bool,
}

impl<'a> ContinuousAsrStateBatch<'a> {
    fn new(rows: Vec<(usize, SessionKey, ExecutorStateLease<'a, ActiveAsrDecode>)>) -> Self {
        Self {
            rows: rows
                .into_iter()
                .map(|(index, session, lease)| (index, session, lease, None))
                .collect(),
            armed: true,
        }
    }

    fn commit(
        mut self,
    ) -> Result<Vec<(usize, SessionKey, ExecutorStateLease<'a, ActiveAsrDecode>)>> {
        for (_, _, lease, checkpoint) in &mut self.rows {
            if let Some((checkpoint, _, _)) = checkpoint.take() {
                lease
                    .require_state_mut()?
                    .state
                    .commit_managed_quantum(checkpoint)?;
            }
        }
        self.armed = false;
        Ok(std::mem::take(&mut self.rows)
            .into_iter()
            .map(|(index, session, lease, _)| (index, session, lease))
            .collect())
    }

    fn rollback_row(&mut self, row: usize) -> Result<usize> {
        let (index, _, lease, checkpoint) = self.rows.get_mut(row).ok_or_else(|| {
            Error::InferenceError("continuous ASR rollback row is out of range".to_string())
        })?;
        let (checkpoint, last_tokens_generated, stream_sequence) =
            checkpoint.take().ok_or_else(|| {
                Error::InferenceError(
                    "continuous ASR row has no armed checkpoint to roll back".to_string(),
                )
            })?;
        let state = lease.require_state_mut()?;
        state.state.rollback_managed_quantum(checkpoint)?;
        state.last_tokens_generated = last_tokens_generated;
        state.stream_sequence = stream_sequence;
        lease.mark_clean();
        Ok(*index)
    }
}

impl Drop for ContinuousAsrStateBatch<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        for (_, session, lease, checkpoint) in &mut self.rows {
            if let Some((checkpoint, last_tokens_generated, stream_sequence)) = checkpoint.take() {
                let rollback = lease.require_state_mut().and_then(|state| {
                    state.state.rollback_managed_quantum(checkpoint)?;
                    state.last_tokens_generated = last_tokens_generated;
                    state.stream_sequence = stream_sequence;
                    Ok(())
                });
                match rollback {
                    Ok(()) => lease.mark_clean(),
                    Err(error) => {
                        tracing::error!(
                            request_id = %session.request_id,
                            epoch = session.epoch,
                            %error,
                            "continuous ASR rollback failed; state fenced until cleanup"
                        );
                    }
                }
            }
        }
        self.rows.clear();
    }
}

fn rollback_scalar_asr_quantum(
    state_lease: &mut ExecutorStateLease<'_, ActiveAsrDecode>,
    checkpoint: &mut Option<NativeAsrDecodeCheckpoint>,
    outer_checkpoint: Option<(usize, usize)>,
    fresh_state: bool,
) -> Result<()> {
    if let Some(checkpoint) = checkpoint.take() {
        let active_state = state_lease.require_state_mut()?;
        active_state.state.rollback_managed_quantum(checkpoint)?;
        if let Some((last_tokens_generated, stream_sequence)) = outer_checkpoint {
            active_state.last_tokens_generated = last_tokens_generated;
            active_state.stream_sequence = stream_sequence;
        }
        state_lease.mark_clean();
    } else if fresh_state {
        state_lease.discard_state();
        state_lease.mark_clean();
    }
    Ok(())
}

fn with_single_invocation_cache<T>(
    request: &EngineCoreRequest,
    scheduled: &ScheduledRequest,
    run: impl FnOnce(&mut crate::models::shared::attention::physical::PhysicalPagedKvCache) -> Result<T>,
) -> Result<T> {
    let mut leases = super::invocation_paged_leases_for_atomic_scalar_row(request, scheduled)?;
    let domains = leases.domains().collect::<Vec<_>>();
    let [domain] = domains.as_slice() else {
        return Err(Error::InferenceError(format!(
            "ASR stage requires exactly one invocation KV domain, found {}",
            domains.len()
        )));
    };
    let output = run(leases.cache_mut(*domain)?)?;
    let _completions = leases.release()?;
    Ok(output)
}

fn with_whisper_invocation_state<T>(
    request: &EngineCoreRequest,
    scheduled: &ScheduledRequest,
    run: impl FnOnce(
        &mut crate::models::shared::attention::physical::PhysicalPagedKvCache,
        &mut crate::engine::InvocationStaticAttentionLease,
    ) -> Result<T>,
) -> Result<T> {
    let mut leases = super::invocation_workspace_leases_for_atomic_scalar_row(request, scheduled)?;
    let output = {
        let (paged, cross) = leases.lease_exact_kind_pair_mut(
            crate::kv::v2::InvocationStateBackingKindV2::PagedAttention,
            crate::kv::v2::InvocationStateBackingKindV2::StaticAttention,
        )?;
        let cache = paged.paged_cache_mut()?;
        let cross = cross.typed_mut::<crate::engine::InvocationStaticAttentionLease>()?;
        run(cache, cross)?
    };
    let completions = leases.release()?;
    if completions.len() != 2 {
        return Err(Error::InferenceError(
            "Whisper ASR physical state returned an incomplete completion set".to_string(),
        ));
    }
    Ok(output)
}

fn with_vibevoice_invocation_state<T>(
    request: &EngineCoreRequest,
    scheduled: &ScheduledRequest,
    run: impl FnOnce(&mut crate::kv::v2::InvocationWorkspaceLeaseSetV2) -> Result<T>,
) -> Result<T> {
    let mut leases = super::invocation_workspace_leases_for_atomic_scalar_row(request, scheduled)?;
    let output = run(&mut leases)?;
    let completions = leases.release()?;
    if completions.len() != 3 {
        return Err(Error::InferenceError(
            "VibeVoice ASR physical state returned an incomplete completion set".to_string(),
        ));
    }
    Ok(output)
}

fn with_single_invocation_tensor<T>(
    request: &EngineCoreRequest,
    scheduled: &ScheduledRequest,
    run: impl FnOnce(&mut crate::engine::InvocationTensorLease) -> Result<T>,
) -> Result<T> {
    let mut leases = super::invocation_workspace_leases_for_atomic_scalar_row(request, scheduled)?;
    let output = {
        let tensor =
            leases.lease_exact_kind_mut(crate::kv::v2::InvocationStateBackingKindV2::Tensor)?;
        run(tensor.typed_mut::<crate::engine::InvocationTensorLease>()?)?
    };
    let completions = leases.release()?;
    if completions.len() != 1 {
        return Err(Error::InferenceError(
            "atomic ASR tensor state returned an incomplete completion set".to_string(),
        ));
    }
    Ok(output)
}

fn with_nemotron_offline_state<T>(
    request: &EngineCoreRequest,
    scheduled: &ScheduledRequest,
    run: impl FnOnce(
        &mut crate::engine::InvocationTensorLease,
        &mut crate::engine::InvocationTensorLease,
    ) -> Result<T>,
) -> Result<T> {
    let mut leases = super::invocation_workspace_leases_for_atomic_scalar_row(request, scheduled)?;
    let output = {
        let (predictor, acoustic) = leases.lease_exact_kind_pair_mut(
            crate::kv::v2::InvocationStateBackingKindV2::Tensor,
            crate::kv::v2::InvocationStateBackingKindV2::StaticTensor,
        )?;
        run(
            predictor.typed_mut::<crate::engine::InvocationTensorLease>()?,
            acoustic.typed_mut::<crate::engine::InvocationTensorLease>()?,
        )?
    };
    let completions = leases.release()?;
    if completions.len() != 2 {
        return Err(Error::InferenceError(
            "Nemotron offline state returned an incomplete completion set".to_string(),
        ));
    }
    Ok(output)
}

impl NativeExecutor {
    pub(super) fn transcribe_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
    ) -> Result<ModelSessionResult> {
        self.transcribe_request_with_managed_cache(request, scheduled, None)
    }

    fn voxtral_realtime_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        mut managed_state: super::RetainedRowManagedState,
    ) -> Result<ModelSessionResult> {
        let variant = Self::resolve_variant(request)?;
        if variant.family() != ModelFamily::Voxtral || !request.is_realtime_asr_session() {
            return Err(Error::InvalidInput(
                "Voxtral realtime work crossed its exact ASR session route".into(),
            ));
        }
        if request.asr_prompt_for_execution().is_some() {
            return Err(Error::InvalidInput(
                "Voxtral realtime ASR does not support an initial text prompt".into(),
            ));
        }
        if managed_state.tensor_state.is_some() {
            return Err(Error::InferenceError(
                "Voxtral realtime row unexpectedly received tensor state".into(),
            ));
        }
        let mut cache = managed_state.take_only_paged()?;
        managed_state.ensure_all_paged_consumed()?;
        let (operation_id, max_output_steps, max_cache_append) = match &scheduled.work {
            crate::engine::WorkUnit::RealtimePush {
                operation_id,
                max_output_steps,
                max_cache_append,
                ..
            }
            | crate::engine::WorkUnit::RealtimeFinish {
                operation_id,
                max_output_steps,
                max_cache_append,
            } => (*operation_id, *max_output_steps, *max_cache_append),
            _ => {
                return Err(Error::InvalidInput(
                    "Voxtral realtime executor received non-realtime work".into(),
                ));
            }
        };
        let payload = request.realtime_asr_operation(operation_id)?;
        let session = scheduled.session_key();
        let mut state_lease = ExecutorStateLease::checkout(
            &self.voxtral_realtime_states,
            session,
            "Voxtral realtime ASR",
        )?;
        if state_lease
            .state()
            .is_some_and(|active| active.variant != variant)
        {
            state_lease.discard_state();
        }
        if state_lease.state().is_none() {
            if !matches!(payload, RealtimeAsrOperationPayload::Push { .. }) {
                return Err(Error::InferenceError(
                    "Voxtral realtime session cannot start with finish".into(),
                ));
            }
            let model = self.with_registry(|registry| {
                registry.try_get_voxtral_lease(variant).ok_or_else(|| {
                    Error::ModelNotFound(format!(
                        "Voxtral model {variant} is not loaded in registry"
                    ))
                })
            })?;
            let state = model.start_realtime_state(request.asr_language_for_execution());
            state_lease.install_state(ActiveVoxtralRealtime {
                variant,
                model,
                state,
                last_tokens_generated: 0,
                stream_sequence: 0,
                input_sample_rate: 0,
            })?;
        }

        let model_max_output_steps = state_lease
            .require_state_mut()?
            .model
            .realtime_max_output_steps()?;
        if max_output_steps == 0 || max_output_steps > model_max_output_steps {
            return Err(Error::InvalidInput(format!(
                "Voxtral realtime quantum requested {max_output_steps} output steps outside the model ceiling {model_max_output_steps}"
            )));
        }
        if max_cache_append == 0 {
            return Err(Error::InvalidInput(
                "Voxtral realtime quantum requires a non-zero cache append bound".into(),
            ));
        }

        let (prior_tokens, prior_stream_sequence, prior_input_sample_rate, prior_cache_len) = {
            let active = state_lease.require_state_mut()?;
            (
                active.last_tokens_generated,
                active.stream_sequence,
                active.input_sample_rate,
                cache.context_len(),
            )
        };
        let active = state_lease.require_state_mut()?;
        let mut checkpoint = active
            .model
            .begin_realtime_quantum(&mut active.state, &cache)?;
        state_lease.mark_dirty();

        let execution = (|| {
            let active = state_lease.require_state_mut()?;
            let mut should_cancel = || request.is_cancelled();
            let steps = match (&scheduled.work, payload) {
                (
                    crate::engine::WorkUnit::RealtimePush { input, .. },
                    RealtimeAsrOperationPayload::Push {
                        samples,
                        sample_rate,
                    },
                ) if input.start == active.state.source_sample_count()
                    && input.len() == samples.len()
                    && sample_rate > 0 =>
                {
                    if active.input_sample_rate != 0 && active.input_sample_rate != sample_rate {
                        return Err(Error::InvalidInput(
                            "Voxtral realtime sample rate changed within one session".into(),
                        ));
                    }
                    active.input_sample_rate = sample_rate;
                    active.model.apply_realtime_push_physical(
                        &mut active.state,
                        &mut cache,
                        samples.as_ref(),
                        sample_rate,
                        max_output_steps,
                        &mut should_cancel,
                    )?
                }
                (
                    crate::engine::WorkUnit::RealtimeFinish { .. },
                    RealtimeAsrOperationPayload::Finish,
                ) => active.model.apply_realtime_finish_physical(
                    &mut active.state,
                    &mut cache,
                    max_output_steps,
                    &mut should_cancel,
                )?,
                _ => {
                    return Err(Error::InferenceError(
                        "Voxtral realtime work and retained operation payload disagree".into(),
                    ));
                }
            };
            let appended = cache
                .context_len()
                .checked_sub(prior_cache_len)
                .ok_or_else(|| {
                    Error::InferenceError("Voxtral realtime cache cursor regressed".into())
                })?;
            if appended > max_cache_append {
                return Err(Error::InferenceError(format!(
                    "Voxtral realtime quantum appended {appended} cache tokens beyond its {max_cache_append}-token bound"
                )));
            }
            if request.is_cancelled() {
                return Err(Error::Cancelled(
                    "Voxtral realtime quantum cancelled before commit".into(),
                ));
            }
            if let Some(tx) = Self::stream_sender(request) {
                for step in &steps {
                    if !step.delta.is_empty() {
                        Self::stream_text_with_policy(
                            &tx,
                            request.stream_policy,
                            &request.id,
                            &mut active.stream_sequence,
                            step.delta.clone(),
                        )?;
                    }
                    if step.finished {
                        Self::stream_final_marker_with_policy(
                            &tx,
                            request.stream_policy,
                            &request.id,
                            &mut active.stream_sequence,
                        )?;
                    }
                }
            }
            if request.is_cancelled() {
                return Err(Error::Cancelled(
                    "Voxtral realtime quantum cancelled before seal".into(),
                ));
            }
            let staged = request.take_staged_stream_outputs()?;
            if request.is_cancelled() {
                return Err(Error::Cancelled(
                    "Voxtral realtime quantum cancelled during seal".into(),
                ));
            }
            Ok(staged)
        })();

        let staged = match execution {
            Ok(staged) => staged,
            Err(error) => {
                let _ = request.take_staged_stream_outputs();
                let active = state_lease.require_state_mut()?;
                active.stream_sequence = prior_stream_sequence;
                active.last_tokens_generated = prior_tokens;
                active.input_sample_rate = prior_input_sample_rate;
                active.model.rollback_realtime_quantum(
                    &mut active.state,
                    &mut cache,
                    &mut checkpoint,
                )?;
                let cancelled = matches!(error, Error::Cancelled(_));
                if cancelled {
                    state_lease.release()?;
                    return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                        request.id.clone(),
                    )));
                }
                state_lease.restore()?;
                return Err(error);
            }
        };

        if request.is_cancelled() {
            let _ = request.take_staged_stream_outputs();
            let active = state_lease.require_state_mut()?;
            active.stream_sequence = prior_stream_sequence;
            active.last_tokens_generated = prior_tokens;
            active.input_sample_rate = prior_input_sample_rate;
            active.model.rollback_realtime_quantum(
                &mut active.state,
                &mut cache,
                &mut checkpoint,
            )?;
            state_lease.release()?;
            return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                request.id.clone(),
            )));
        }

        let (tokens_generated, text, finished, sample_rate, sample_count) = {
            let active = state_lease.require_state_mut()?;
            active
                .model
                .commit_realtime_quantum(&mut active.state, &cache, &mut checkpoint)?;
            let tokens_generated = active.state.tokens_generated().saturating_sub(prior_tokens);
            active.last_tokens_generated = active.state.tokens_generated();
            (
                tokens_generated,
                active.state.text().to_string(),
                active.state.is_finished(),
                active.input_sample_rate,
                active.state.source_sample_count(),
            )
        };
        let completions = cache.take_completed_writes();
        if finished {
            state_lease.release()?;
        } else {
            state_lease.restore()?;
        }
        Ok(ModelSessionResult::sequence(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(AudioOutput {
                samples: Vec::new(),
                sample_rate,
                duration_secs: if sample_rate == 0 {
                    0.0
                } else {
                    sample_count as f32 / sample_rate as f32
                },
            }),
            text: Some(text),
            input_transcription: None,
            tokens_processed: match &scheduled.work {
                crate::engine::WorkUnit::RealtimePush { input, .. } => input.len(),
                crate::engine::WorkUnit::RealtimeFinish { .. } => 0,
                _ => unreachable!("realtime work authenticated above"),
            },
            tokens_generated,
            finished,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
        })
        .with_staged_stream_outputs(staged)
        .with_managed_cache_completions(completions)
        .with_managed_cache_append(cache.context_len().saturating_sub(prior_cache_len)))
    }

    fn granite_speech_asr_sequence_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        mut managed_state: Option<super::RetainedRowManagedState>,
    ) -> Result<ModelSessionResult> {
        let variant = Self::resolve_variant(request)?;
        let model = request.prepared_asr_model_for_executor()?.ok_or_else(|| {
            Error::InferenceError("Granite Speech sequence lost its loaded model".into())
        })?;
        if variant.family() != ModelFamily::GraniteSpeechAsr
            || !model.supports_resumable_prefill()
            || !model.supports_incremental_decode()
        {
            return Err(Error::InvalidInput(
                "loaded Granite Speech model has no retained sequence contract".into(),
            ));
        }
        let mut retained = managed_state.take().ok_or_else(|| {
            Error::InferenceError("Granite Speech sequence lost retained paged state".into())
        })?;
        if retained.tensor_state.is_some() {
            return Err(Error::InferenceError(
                "Granite Speech retained sequence unexpectedly received tensor state".into(),
            ));
        }
        let cache = retained.take_only_paged()?;
        let prompt_tokens = request.num_prompt_tokens();
        let resumable_span = scheduled
            .is_prefill
            .then(|| resumable_asr_prefill_span(scheduled, prompt_tokens))
            .transpose()?;
        let session = scheduled.session_key();
        let mut state_lease = ExecutorStateLease::checkout(
            &self.asr_decode_states,
            session,
            "Granite Speech ASR decode",
        )?;
        if state_lease
            .state()
            .is_some_and(|active| active.variant != variant || !Arc::ptr_eq(&active.model, &model))
        {
            state_lease.discard_state();
        }

        let mut checkpoint = None;
        let mut outer_checkpoint = None;
        let mut fresh_state = false;
        if state_lease.state().is_some() {
            let active = state_lease.require_state_mut()?;
            outer_checkpoint = Some((active.last_tokens_generated, active.stream_sequence));
            checkpoint = Some(active.state.begin_managed_quantum(cache)?);
            state_lease.mark_dirty();
        } else {
            if !scheduled.is_prefill || scheduled.num_computed_tokens != 0 {
                return Err(Error::InferenceError(
                    "Granite Speech sequence lost state before initial prefill".into(),
                ));
            }
            if request.is_cancelled() {
                state_lease.release()?;
                return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    request.id.clone(),
                )));
            }
            let artifact = request
                .prepared_granite_speech_artifact_for_executor()?
                .ok_or_else(|| {
                    Error::InferenceError(
                        "Granite Speech sequence lost its prepared prompt artifact".into(),
                    )
                })?;
            let (samples, sample_rate) =
                request.prepared_asr_audio_for_executor()?.ok_or_else(|| {
                    Error::InferenceError("Granite Speech sequence lost decoded audio".into())
                })?;
            let options = Self::asr_chunk_generation_options(
                request,
                ModelFamily::GraniteSpeechAsr,
                samples.len(),
                sample_rate,
                &Self::asr_generation_options(request),
            );
            let Some(decode_state) = run_asr_model_call(request, || {
                Self::run_blocking(|| {
                    model.start_granite_speech_resumable_prefill_managed(artifact, options, cache)
                })
            })?
            else {
                state_lease.release()?;
                return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    request.id.clone(),
                )));
            };
            if decode_state.prefill_token_count() != Some(prompt_tokens)
                || decode_state.prefill_progress() != Some(0)
                || decode_state.sequence_position() != Some(0)
            {
                return Err(Error::InferenceError(
                    "Granite Speech prompt geometry differs from scheduler admission".into(),
                ));
            }
            let model_lease = request
                .prepared_asr_model_lease_for_executor()?
                .ok_or_else(|| {
                    Error::InferenceError("Granite Speech sequence lost model residency".into())
                })?;
            state_lease.install_state(ActiveAsrDecode {
                variant,
                model: model.clone(),
                _model_lease: model_lease,
                state: decode_state,
                last_tokens_generated: 0,
                stream_sequence: 0,
                input_sample_rate: sample_rate,
                input_sample_count: samples.len(),
            })?;
            fresh_state = true;
        }

        if request.is_cancelled() {
            rollback_scalar_asr_quantum(
                &mut state_lease,
                &mut checkpoint,
                outer_checkpoint,
                fresh_state,
            )?;
            state_lease.release()?;
            return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                request.id.clone(),
            )));
        }

        state_lease.mark_dirty();
        let execution = (|| {
            let active = state_lease.require_state_mut()?;
            let iterations = if scheduled.is_prefill {
                1
            } else {
                scheduled.num_tokens.max(1)
            };
            let mut generated = 0usize;
            let mut text = String::new();
            let mut finished = false;
            let mut cancelled = false;
            let mut events = Vec::new();
            for _ in 0..iterations {
                if request.is_cancelled() {
                    cancelled = true;
                    break;
                }
                let step = if let Some((start, end)) = resumable_span {
                    let Some(_complete) = run_asr_model_call(request, || {
                        Self::run_blocking(|| {
                            active
                                .model
                                .continue_resumable_prefill(&mut active.state, start, end)
                        })
                    })?
                    else {
                        cancelled = true;
                        break;
                    };
                    crate::models::registry::NativeAsrDecodeStep {
                        delta: String::new(),
                        text: String::new(),
                        tokens_generated: active.last_tokens_generated,
                        finished: false,
                    }
                } else {
                    let Some(step) = run_asr_model_call(request, || {
                        Self::run_blocking(|| active.model.decode_step(&mut active.state))
                    })?
                    else {
                        cancelled = true;
                        break;
                    };
                    step
                };
                if request.is_cancelled() {
                    cancelled = true;
                    break;
                }
                generated = generated.saturating_add(
                    step.tokens_generated
                        .saturating_sub(active.last_tokens_generated),
                );
                active.last_tokens_generated = step.tokens_generated;
                text = step.text.clone();
                if !scheduled.is_prefill {
                    events.push((step.delta, step.finished));
                }
                if step.finished {
                    finished = true;
                    break;
                }
            }
            if !cancelled {
                if let Some(tx) = Self::stream_sender(request) {
                    for (delta, event_finished) in events {
                        if request.is_cancelled() {
                            cancelled = true;
                            break;
                        }
                        if !delta.is_empty() {
                            Self::stream_text_with_policy(
                                &tx,
                                request.stream_policy,
                                &request.id,
                                &mut active.stream_sequence,
                                delta,
                            )?;
                        }
                        if event_finished {
                            Self::stream_final_marker_with_policy(
                                &tx,
                                request.stream_policy,
                                &request.id,
                                &mut active.stream_sequence,
                            )?;
                        }
                    }
                }
                cancelled |= request.is_cancelled();
            }
            if cancelled {
                let _ = request.take_staged_stream_outputs()?;
            }
            let completions = if cancelled {
                Vec::new()
            } else {
                active.state.take_managed_write_completions()
            };
            Ok((
                generated,
                text,
                finished,
                cancelled,
                active.input_sample_rate,
                active.input_sample_count,
                completions,
            ))
        })();
        let (generated, text, finished, cancelled, sample_rate, sample_count, completions) =
            match execution {
                Ok(value) => value,
                Err(error) => {
                    let _ = request.take_staged_stream_outputs()?;
                    rollback_scalar_asr_quantum(
                        &mut state_lease,
                        &mut checkpoint,
                        outer_checkpoint,
                        fresh_state,
                    )?;
                    return Err(error);
                }
            };
        if cancelled || request.is_cancelled() {
            let _ = request.take_staged_stream_outputs()?;
            rollback_scalar_asr_quantum(
                &mut state_lease,
                &mut checkpoint,
                outer_checkpoint,
                fresh_state,
            )?;
            state_lease.release()?;
            return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                request.id.clone(),
            )));
        }
        if let Some(checkpoint) = checkpoint.take() {
            state_lease
                .require_state_mut()?
                .state
                .commit_managed_quantum(checkpoint)?;
        }
        let tokens_processed = if scheduled.is_prefill {
            scheduled.num_tokens
        } else {
            scheduled.num_tokens.max(1)
        };
        if finished {
            state_lease.release()?;
        } else {
            state_lease.restore()?;
        }
        Ok(ModelSessionResult::sequence(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(AudioOutput {
                samples: Vec::new(),
                sample_rate,
                duration_secs: if sample_rate == 0 {
                    0.0
                } else {
                    sample_count as f32 / sample_rate as f32
                },
            }),
            text: Some(text),
            input_transcription: None,
            tokens_processed,
            tokens_generated: generated,
            finished,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
        })
        .with_managed_cache_completions(completions))
    }

    fn whisper_asr_sequence_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        mut managed_state: Option<super::RetainedRowManagedState>,
    ) -> Result<ModelSessionResult> {
        let variant = Self::resolve_variant(request)?;
        let model = request.prepared_asr_model_for_executor()?.ok_or_else(|| {
            Error::InferenceError("Whisper sequence request lost its model identity".into())
        })?;
        let mut retained = managed_state.take().ok_or_else(|| {
            Error::InferenceError("Whisper sequence request lost retained paged state".into())
        })?;
        if retained.tensor_state.is_some() {
            return Err(Error::InferenceError(
                "Whisper retained sequence unexpectedly received tensor state".into(),
            ));
        }
        let session_generation = retained.session_generation();
        let cache = retained.take_only_paged()?;
        let cross_runtime = request
            .v2_state_runtime()
            .and_then(|runtime| runtime.retained_static_attention_runtime())
            .ok_or_else(|| {
                Error::InferenceError(
                    "Whisper sequence request lost retained cross-attention state".into(),
                )
            })?;
        let prompt_tokens = request.num_prompt_tokens();
        let resumable_span = scheduled
            .is_prefill
            .then(|| resumable_asr_prefill_span(scheduled, prompt_tokens))
            .transpose()?;
        let session = scheduled.session_key();
        let mut state_lease =
            ExecutorStateLease::checkout(&self.asr_decode_states, session, "Whisper decode")?;
        if state_lease
            .state()
            .is_some_and(|state| state.variant != variant || !Arc::ptr_eq(&state.model, &model))
        {
            state_lease.discard_state();
        }
        let mut checkpoint = None;
        let mut fresh = false;
        if state_lease.state().is_some() {
            let state = &mut state_lease.require_state_mut()?.state;
            checkpoint = Some(
                if begins_whisper_managed_generation(scheduled, session_generation) {
                    state.begin_whisper_managed_generation(cache, session_generation)?
                } else {
                    state.begin_managed_quantum(cache)?
                },
            );
            state_lease.mark_dirty();
        } else {
            if !scheduled.is_prefill || scheduled.num_computed_tokens != 0 {
                return Err(Error::InferenceError(
                    "Whisper sequence lost its state before initial prefill".into(),
                ));
            }
            let prepared = request
                .prepared_whisper_window_for_executor()?
                .ok_or_else(|| {
                    Error::InferenceError("Whisper sequence lost its prepared window".into())
                })?;
            let Some(decode_state) = run_asr_model_call(request, || {
                Self::run_blocking(|| {
                    // Registration and model ownership transfer are one synchronous
                    // boundary: after registration succeeds, the model call owns release
                    // on both success and failure.
                    let cross_sequence = cross_runtime.register_sequence()?;
                    model.start_whisper_resumable_prefill(
                        prepared.as_ref(),
                        request.asr_language_for_execution(),
                        request.asr_prompt_for_execution(),
                        Some(request.params.max_tokens.clamp(1, MAX_ASR_NEW_TOKENS)),
                        cache,
                        cross_runtime,
                        cross_sequence,
                    )
                })
            })?
            else {
                state_lease.release()?;
                return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    request.id.clone(),
                )));
            };
            if decode_state.prefill_token_count() != Some(prompt_tokens) {
                return Err(Error::InferenceError(
                    "Whisper prompt geometry differs from scheduler admission".into(),
                ));
            }
            let (samples, sample_rate) =
                request.prepared_asr_audio_for_executor()?.ok_or_else(|| {
                    Error::InferenceError("Whisper sequence lost decoded audio".into())
                })?;
            let model_lease = request
                .prepared_asr_model_lease_for_executor()?
                .ok_or_else(|| {
                    Error::InferenceError("Whisper sequence lost model residency".into())
                })?;
            state_lease.install_state(ActiveAsrDecode {
                variant,
                model: model.clone(),
                _model_lease: model_lease,
                state: decode_state,
                last_tokens_generated: 0,
                stream_sequence: 0,
                input_sample_rate: sample_rate,
                input_sample_count: samples.len(),
            })?;
            fresh = true;
        }

        if request.is_cancelled() {
            if let Some(checkpoint) = checkpoint.take() {
                state_lease
                    .require_state_mut()?
                    .state
                    .rollback_managed_quantum(checkpoint)?;
                state_lease.mark_clean();
            } else if fresh {
                state_lease.discard_state();
            }
            state_lease.release()?;
            return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                request.id.clone(),
            )));
        }

        state_lease.mark_dirty();
        let execution = (|| {
            let active = state_lease.require_state_mut()?;
            let iterations = if scheduled.is_prefill {
                1
            } else {
                scheduled.num_tokens.max(1)
            };
            let mut generated = 0usize;
            let mut text = String::new();
            let mut finished = false;
            let mut cancelled = false;
            let mut restart = false;
            let mut events = Vec::new();
            for _ in 0..iterations {
                if request.is_cancelled() {
                    cancelled = true;
                    break;
                }
                let mut step = if let Some((start, end)) = resumable_span {
                    let Some(_complete) = run_asr_model_call(request, || {
                        Self::run_blocking(|| {
                            active.model.continue_whisper_resumable_prefill(
                                &mut active.state,
                                start,
                                end,
                            )
                        })
                    })?
                    else {
                        cancelled = true;
                        break;
                    };
                    // Prefill completion is a scheduling boundary. Never consume the first
                    // decode token in the final prefill quantum: it needs its own reservation
                    // and cancellation/transaction fence.
                    whisper_prefill_boundary_step(active.last_tokens_generated)
                } else {
                    let Some(step) = run_asr_model_call(request, || {
                        Self::run_blocking(|| {
                            active.model.decode_whisper_retained_step(&mut active.state)
                        })
                    })?
                    else {
                        cancelled = true;
                        break;
                    };
                    step
                };
                if step.finished {
                    let action = resolve_whisper_terminal_action(
                        request,
                        &mut step,
                        session_generation,
                        || {
                            active
                                .model
                                .resolve_whisper_terminal_transition(&mut active.state)
                        },
                    )?;
                    let Some(action) = action else {
                        cancelled = true;
                        break;
                    };
                    restart = matches!(action, WhisperTerminalAction::Restart);
                } else {
                    // Whisper fallback policy can reject an entire temperature attempt at
                    // EOS. Do not publish provisional text before the policy accepts it.
                    step.delta.clear();
                    step.text.clear();
                }
                if request.is_cancelled() {
                    cancelled = true;
                    break;
                }
                if restart {
                    break;
                }
                generated = generated.saturating_add(
                    step.tokens_generated
                        .saturating_sub(active.last_tokens_generated),
                );
                active.last_tokens_generated = step.tokens_generated;
                text = step.text.clone();
                if step.finished {
                    events.push((step.delta, true));
                }
                if step.finished {
                    finished = true;
                    break;
                }
            }
            if !cancelled {
                if let Some(tx) = Self::stream_sender(request) {
                    for (delta, event_finished) in events {
                        if request.is_cancelled() {
                            cancelled = true;
                            break;
                        }
                        if !delta.is_empty() {
                            Self::stream_text_with_policy(
                                &tx,
                                request.stream_policy,
                                &request.id,
                                &mut active.stream_sequence,
                                delta,
                            )?;
                        }
                        if event_finished {
                            Self::stream_final_marker_with_policy(
                                &tx,
                                request.stream_policy,
                                &request.id,
                                &mut active.stream_sequence,
                            )?;
                        }
                    }
                }
                cancelled |= request.is_cancelled();
            }
            if cancelled {
                let _ = request.take_staged_stream_outputs()?;
            }
            if restart {
                let _ = request.take_staged_stream_outputs()?;
                // The scheduler will abort this attempt's physical reservation before
                // advancing the session generation. Drain and drop its sealed writes so
                // no receipt can escape and the model can authenticate the next generation.
                drop(active.state.take_managed_write_completions());
            }
            let completions = if cancelled || restart {
                Vec::new()
            } else {
                active.state.take_managed_write_completions()
            };
            Ok((
                generated,
                text,
                finished,
                cancelled,
                active.input_sample_rate,
                active.input_sample_count,
                completions,
                restart,
            ))
        })();
        let (generated, text, finished, cancelled, sample_rate, sample_count, completions, restart) =
            match execution {
                Ok(value) => value,
                Err(error) => {
                    let _ = request.take_staged_stream_outputs()?;
                    if let Some(checkpoint) = checkpoint.take() {
                        state_lease
                            .require_state_mut()?
                            .state
                            .rollback_managed_quantum(checkpoint)?;
                        state_lease.mark_clean();
                    } else if fresh {
                        state_lease.discard_state();
                    }
                    return Err(error);
                }
            };
        if cancelled {
            if let Some(checkpoint) = checkpoint.take() {
                state_lease
                    .require_state_mut()?
                    .state
                    .rollback_managed_quantum(checkpoint)?;
                state_lease.mark_clean();
            } else if fresh {
                state_lease.discard_state();
            }
            state_lease.release()?;
            return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                request.id.clone(),
            )));
        }
        if restart {
            let checkpoint = checkpoint.take().ok_or_else(|| {
                Error::InferenceError(
                    "Whisper fallback restart has no active managed checkpoint".into(),
                )
            })?;
            let active = state_lease.require_state_mut()?;
            active.state.commit_managed_quantum(checkpoint)?;
            active.last_tokens_generated = 0;
            active.stream_sequence = 0;
            if request.is_cancelled() {
                state_lease.discard_state();
                state_lease.release()?;
                return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    request.id.clone(),
                )));
            }
            state_lease.restore()?;
            return Ok(ModelSessionResult::restart_sequence(
                request.id.clone(),
                SequenceRestartReason::ModelFallback,
            ));
        }
        if let Some(checkpoint) = checkpoint.take() {
            state_lease
                .require_state_mut()?
                .state
                .commit_managed_quantum(checkpoint)?;
        }
        let tokens_processed = if scheduled.is_prefill {
            scheduled.num_tokens
        } else {
            generated
        };
        if finished {
            state_lease.release()?;
        } else {
            state_lease.restore()?;
        }
        Ok(ModelSessionResult::sequence(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(AudioOutput {
                samples: Vec::new(),
                sample_rate,
                duration_secs: if sample_rate == 0 {
                    0.0
                } else {
                    sample_count as f32 / sample_rate as f32
                },
            }),
            text: Some(text),
            input_transcription: None,
            tokens_processed,
            tokens_generated: generated,
            finished,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
        })
        .with_managed_cache_completions(completions))
    }

    fn vibevoice_asr_sequence_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        managed_state: Option<super::RetainedRowManagedState>,
    ) -> Result<ModelSessionResult> {
        self.vibevoice_asr_sequence_request_inner(request, scheduled, managed_state, None)
    }

    fn vibevoice_asr_sequence_request_inner(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        mut managed_state: Option<super::RetainedRowManagedState>,
        prepared_tokenizer: Option<&VibeVoiceAsrPreparedTokenizerSpan>,
    ) -> Result<ModelSessionResult> {
        let variant = Self::resolve_variant(request)?;
        let model = request.prepared_asr_model_for_executor()?.ok_or_else(|| {
            Error::InferenceError(
                "VibeVoice ASR sequence request lost its loaded model identity".into(),
            )
        })?;
        if variant.family() != ModelFamily::VibeVoiceAsr
            || !model.supports_resumable_prefill()
            || !model.supports_continuous_decode_batch()
        {
            return Err(Error::InvalidInput(
                "loaded VibeVoice ASR model has no retained sequence contract".into(),
            ));
        }
        let mut retained = managed_state.take().ok_or_else(|| {
            Error::InferenceError("VibeVoice ASR sequence lost retained paged state".into())
        })?;
        let tensor_reservation = retained.tensor_state.clone();
        let auxiliary_spans = match &scheduled.work {
            crate::engine::WorkUnit::SequenceStep {
                auxiliary_state: Some(spans),
                ..
            } => spans.as_ref(),
            _ => &[],
        };
        if auxiliary_spans.len() > 1 || auxiliary_spans.is_empty() != tensor_reservation.is_none() {
            return Err(Error::InferenceError(
                "VibeVoice ASR tokenizer reservation does not match its exact auxiliary span"
                    .into(),
            ));
        }
        let tensor_arena = request
            .managed_cache_runtime()
            .and_then(|runtime| runtime.tensor_state())
            .cloned();
        if tensor_reservation.is_some() && tensor_arena.is_none() {
            return Err(Error::InferenceError(
                "VibeVoice ASR tokenizer reservation lost its retained tensor arena".into(),
            ));
        }
        let tokenizer_transaction = tensor_reservation
            .as_ref()
            .map(|_| PhysicalStateTransactionId::new(scheduled.plan_id))
            .transpose()?;
        let tokenizer_quantum = match (
            tensor_arena.as_ref(),
            tokenizer_transaction,
            auxiliary_spans.first(),
        ) {
            (Some(arena), Some(transaction), Some(span)) => Some(
                VibeVoiceAsrRetainedTokenizerQuantum::new(arena.clone(), transaction, span.clone()),
            ),
            (_, None, None) => None,
            _ => {
                return Err(Error::InferenceError(
                    "VibeVoice ASR tokenizer quantum identity is incomplete".into(),
                ));
            }
        };
        let cache = retained.take_only_paged()?;
        let prompt_tokens = request.num_prompt_tokens();
        let resumable_span = scheduled
            .is_prefill
            .then(|| resumable_asr_prefill_span(scheduled, prompt_tokens))
            .transpose()?;
        let session = scheduled.session_key();
        let mut state_lease =
            ExecutorStateLease::checkout(&self.asr_decode_states, session, "VibeVoice ASR decode")?;
        if state_lease
            .state()
            .is_some_and(|active| active.variant != variant || !Arc::ptr_eq(&active.model, &model))
        {
            state_lease.discard_state();
        }

        let mut checkpoint = None;
        let mut outer_checkpoint = None;
        let mut fresh_state = false;
        if state_lease.state().is_some() {
            let active = state_lease.require_state_mut()?;
            outer_checkpoint = Some((active.last_tokens_generated, active.stream_sequence));
            checkpoint = Some(active.state.begin_managed_quantum(cache)?);
            state_lease.mark_dirty();
        } else {
            if !scheduled.is_prefill || scheduled.num_computed_tokens != 0 {
                return Err(Error::InferenceError(
                    "VibeVoice ASR sequence lost state before initial prefill".into(),
                ));
            }
            if request.is_cancelled() {
                state_lease.release()?;
                return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    request.id.clone(),
                )));
            }
            let artifact = request
                .prepared_vibevoice_artifact_for_executor()?
                .ok_or_else(|| {
                    Error::InferenceError(
                        "VibeVoice ASR sequence lost its prepared encoder artifact".into(),
                    )
                })?;
            let (samples, sample_rate) =
                request.prepared_asr_audio_for_executor()?.ok_or_else(|| {
                    Error::InferenceError(
                        "VibeVoice ASR sequence lost its prepared decoded audio".into(),
                    )
                })?;
            let Some(decode_state) = run_asr_model_call(request, || {
                Self::run_blocking(|| {
                    model.start_vibevoice_resumable_prefill_managed(
                        artifact,
                        Self::asr_generation_options(request),
                        cache,
                    )
                })
            })?
            else {
                state_lease.release()?;
                return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    request.id.clone(),
                )));
            };
            if decode_state.prefill_token_count() != Some(prompt_tokens)
                || decode_state.prefill_progress() != Some(0)
                || decode_state.sequence_position() != Some(0)
            {
                return Err(Error::InferenceError(
                    "VibeVoice ASR prepared prompt does not match scheduler admission".into(),
                ));
            }
            let model_lease = request
                .prepared_asr_model_lease_for_executor()?
                .ok_or_else(|| {
                    Error::InferenceError("VibeVoice ASR sequence lost model residency".into())
                })?;
            state_lease.install_state(ActiveAsrDecode {
                variant,
                model: model.clone(),
                _model_lease: model_lease,
                state: decode_state,
                last_tokens_generated: 0,
                stream_sequence: 0,
                input_sample_rate: sample_rate,
                input_sample_count: samples.len(),
            })?;
            fresh_state = true;
        }

        if request.is_cancelled() {
            rollback_scalar_asr_quantum(
                &mut state_lease,
                &mut checkpoint,
                outer_checkpoint,
                fresh_state,
            )?;
            state_lease.release()?;
            return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                request.id.clone(),
            )));
        }

        state_lease.mark_dirty();
        let execution = (|| {
            let active = state_lease.require_state_mut()?;
            let iterations = if scheduled.is_prefill {
                1
            } else {
                scheduled.num_tokens.max(1)
            };
            let mut generated = 0usize;
            let mut text = String::new();
            let mut finished = false;
            let mut cancelled = false;
            let mut events = Vec::new();
            for _ in 0..iterations {
                if request.is_cancelled() {
                    cancelled = true;
                    break;
                }
                let step = if let Some((start, end)) = resumable_span {
                    let Some(complete) = run_asr_model_call(request, || {
                        Self::run_blocking(|| {
                            match prepared_tokenizer {
                            Some(prepared) => active
                                .model
                                .continue_vibevoice_resumable_prefill_prepared(
                                    &mut active.state,
                                    start,
                                    end,
                                    tokenizer_quantum.clone().ok_or_else(|| {
                                        Error::InferenceError(
                                            "prepared VibeVoice tokenizer span lost its selected transaction"
                                                .into(),
                                        )
                                    })?,
                                    prepared,
                                ),
                            None => active.model.continue_vibevoice_resumable_prefill_retained(
                                &mut active.state,
                                start,
                                end,
                                tokenizer_quantum.clone(),
                            ),
                        }
                        })
                    })?
                    else {
                        cancelled = true;
                        break;
                    };
                    let staged = active.state.take_staged_asr_decode_step();
                    vibevoice_prefill_boundary_step(complete, staged, active.last_tokens_generated)?
                } else {
                    let Some(step) = run_asr_model_call(request, || {
                        Self::run_blocking(|| active.model.decode_step(&mut active.state))
                    })?
                    else {
                        cancelled = true;
                        break;
                    };
                    step
                };
                if request.is_cancelled() {
                    cancelled = true;
                    break;
                }
                let row_generated = step
                    .tokens_generated
                    .saturating_sub(active.last_tokens_generated);
                active.last_tokens_generated = step.tokens_generated;
                generated = generated.saturating_add(row_generated);
                text = step.text.clone();
                events.push((step.delta, step.finished));
                if step.finished {
                    finished = true;
                    break;
                }
            }

            if !cancelled {
                if let Some(tx) = Self::stream_sender(request) {
                    for (delta, event_finished) in events {
                        if request.is_cancelled() {
                            cancelled = true;
                            break;
                        }
                        if !delta.is_empty() {
                            Self::stream_text_with_policy(
                                &tx,
                                request.stream_policy,
                                &request.id,
                                &mut active.stream_sequence,
                                delta,
                            )?;
                        }
                        if event_finished {
                            Self::stream_final_marker_with_policy(
                                &tx,
                                request.stream_policy,
                                &request.id,
                                &mut active.stream_sequence,
                            )?;
                        }
                    }
                }
                cancelled |= request.is_cancelled();
            }
            if cancelled {
                let _ = request.take_staged_stream_outputs()?;
            }
            let completions = if cancelled {
                Vec::new()
            } else {
                active.state.take_managed_write_completions()
            };
            Ok((
                generated,
                text,
                finished,
                cancelled,
                active.input_sample_rate,
                active.input_sample_count,
                completions,
            ))
        })();
        let (generated, text, finished, cancelled, sample_rate, sample_count, completions) =
            match execution {
                Ok(value) => value,
                Err(error) => {
                    let _ = request.take_staged_stream_outputs()?;
                    rollback_scalar_asr_quantum(
                        &mut state_lease,
                        &mut checkpoint,
                        outer_checkpoint,
                        fresh_state,
                    )?;
                    return Err(error);
                }
            };
        if cancelled {
            rollback_scalar_asr_quantum(
                &mut state_lease,
                &mut checkpoint,
                outer_checkpoint,
                fresh_state,
            )?;
            state_lease.release()?;
            return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                request.id.clone(),
            )));
        }
        // Cancellation can race the bookkeeping between the last staged event
        // and checkpoint commit. Output and completions are still private here,
        // so honor that edge by rolling back the entire scalar quantum.
        if request.is_cancelled() {
            let _ = request.take_staged_stream_outputs()?;
            rollback_scalar_asr_quantum(
                &mut state_lease,
                &mut checkpoint,
                outer_checkpoint,
                fresh_state,
            )?;
            state_lease.release()?;
            return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                request.id.clone(),
            )));
        }
        let clocked_state_completion = match (tensor_arena.as_ref(), tokenizer_transaction) {
            (Some(arena), Some(transaction)) => match arena.seal_selected_completion(transaction) {
                Ok(completion) => Some(completion),
                Err(error) => {
                    let _ = request.take_staged_stream_outputs()?;
                    rollback_scalar_asr_quantum(
                        &mut state_lease,
                        &mut checkpoint,
                        outer_checkpoint,
                        fresh_state,
                    )?;
                    return Err(error);
                }
            },
            (_, None) => None,
            _ => {
                rollback_scalar_asr_quantum(
                    &mut state_lease,
                    &mut checkpoint,
                    outer_checkpoint,
                    fresh_state,
                )?;
                return Err(Error::InferenceError(
                    "VibeVoice ASR tokenizer completion identity is incomplete".into(),
                ));
            }
        };
        if request.is_cancelled() {
            let _ = request.take_staged_stream_outputs()?;
            rollback_scalar_asr_quantum(
                &mut state_lease,
                &mut checkpoint,
                outer_checkpoint,
                fresh_state,
            )?;
            state_lease.release()?;
            return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                request.id.clone(),
            )));
        }
        if let Some(checkpoint) = checkpoint.take() {
            if let Err(error) = state_lease
                .require_state_mut()?
                .state
                .commit_managed_quantum(checkpoint)
            {
                let _ = request.take_staged_stream_outputs()?;
                return Err(error);
            }
        }
        let tokens_processed = if scheduled.is_prefill {
            scheduled.num_tokens
        } else {
            scheduled.num_tokens.max(1)
        };
        if finished {
            state_lease.release()?;
        } else {
            state_lease.restore()?;
        }
        let result = ModelSessionResult::sequence(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(AudioOutput {
                samples: Vec::new(),
                sample_rate,
                duration_secs: if sample_rate == 0 {
                    0.0
                } else {
                    sample_count as f32 / sample_rate as f32
                },
            }),
            text: Some(text),
            input_transcription: None,
            tokens_processed,
            tokens_generated: generated,
            finished,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
        })
        .with_managed_cache_completions(completions);
        Ok(match clocked_state_completion {
            Some(completion) => result.with_clocked_state_completion(completion),
            None => result,
        })
    }

    fn qwen3_asr_sequence_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        mut managed_state: Option<super::RetainedRowManagedState>,
    ) -> Result<ModelSessionResult> {
        let variant = Self::resolve_variant(request)?;
        let prepared_model = request.prepared_asr_model_for_executor()?.ok_or_else(|| {
            Error::InferenceError(
                "Qwen3 ASR sequence request has no exact loaded model identity".to_string(),
            )
        })?;
        if !prepared_model.supports_incremental_decode() {
            return Err(Error::InvalidInput(
                "loaded Qwen3 ASR model has no retained sequence decoder".to_string(),
            ));
        }
        if request.managed_cache_runtime().is_none() || managed_state.is_none() {
            return Err(Error::InferenceError(
                "Qwen3 ASR sequence execution requires scheduler-owned retained state".to_string(),
            ));
        }
        let mut retained = managed_state.take().expect("validated retained ASR state");
        let tensor_reservation = retained.tensor_state.clone();
        let mut managed_cache = Some(retained.take_only_paged()?);
        let tensor_arena = request
            .managed_cache_runtime()
            .and_then(|runtime| runtime.tensor_state());
        if tensor_arena.is_some() != tensor_reservation.is_some() {
            return Err(Error::InferenceError(
                "Qwen3 ASR sequence lost its prepared-input tensor-state reservation".to_string(),
            ));
        }

        let resumable_prefill =
            self.config.enable_chunked_prefill && prepared_model.supports_resumable_prefill();
        let prompt_tokens = request.num_prompt_tokens();
        let resumable_span = (scheduled.is_prefill && resumable_prefill)
            .then(|| resumable_asr_prefill_span(scheduled, prompt_tokens))
            .transpose()?;
        if scheduled.is_prefill
            && !resumable_prefill
            && (scheduled.num_computed_tokens != 0 || scheduled.num_tokens != prompt_tokens)
        {
            return Err(Error::InvalidInput(
                "managed Qwen3 ASR full prefill requires one exact multimodal prompt quantum"
                    .to_string(),
            ));
        }

        let session = scheduled.session_key();
        let mut state_lease =
            ExecutorStateLease::checkout(&self.asr_decode_states, session, "ASR decode")?;
        if state_lease
            .state()
            .map(|state| state.variant != variant)
            .unwrap_or(false)
        {
            state_lease.discard_state();
        }

        let mut checkpoint = None;
        let mut outer_checkpoint = None;
        let mut fresh_state = false;
        let initial_media_decode_ms = None;
        if let Some(active) = state_lease.state() {
            if !Arc::ptr_eq(&active.model, &prepared_model) {
                return Err(Error::InferenceError(
                    "Qwen3 ASR sequence state belongs to a different loaded model instance"
                        .to_string(),
                ));
            }
            let cache = managed_cache.take().ok_or_else(|| {
                Error::InferenceError(
                    "Qwen3 ASR continuation lost its managed-cache reservation".to_string(),
                )
            })?;
            outer_checkpoint = Some((active.last_tokens_generated, active.stream_sequence));
            let next_checkpoint = state_lease
                .require_state_mut()?
                .state
                .begin_managed_quantum(cache)?;
            checkpoint = Some(next_checkpoint);
            if let (Some(arena), Some(reservation)) = (tensor_arena, tensor_reservation) {
                let hydration = state_lease
                    .require_state_mut()?
                    .state
                    .bind_qwen3_tensor_sequence(reservation.sequence)
                    .and_then(|()| {
                        state_lease
                            .require_state_mut()?
                            .state
                            .restore_qwen3_prepared_tensor_state(arena)
                    });
                if let Err(error) = hydration {
                    state_lease
                        .require_state_mut()?
                        .state
                        .rollback_managed_quantum(
                            checkpoint
                                .take()
                                .expect("checkpoint installed before hydration"),
                        )?;
                    state_lease.mark_clean();
                    return Err(error);
                }
            }
            state_lease.mark_dirty();
        } else {
            if !scheduled.is_prefill {
                return Err(Error::InferenceError(format!(
                    "Qwen3 ASR request {} lost its decode state before continuation",
                    request.id
                )));
            }
            if resumable_prefill && scheduled.num_computed_tokens != 0 {
                return Err(Error::InferenceError(format!(
                    "resumable Qwen3 ASR request {} lost its state before prompt span continuation",
                    request.id
                )));
            }
            if request.is_cancelled() {
                state_lease.release()?;
                return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    request.id.clone(),
                )));
            }

            let (samples, sample_rate) =
                request.prepared_asr_audio_for_executor()?.ok_or_else(|| {
                    Error::InferenceError(
                        "Qwen3 ASR sequence request lost its prepared decoded audio".to_string(),
                    )
                })?;
            let prepared_audio = request
                .prepared_asr_encoder_artifact_for_executor()?
                .ok_or_else(|| {
                    Error::InferenceError(
                        "Qwen3 ASR sequence request lost its prepared encoder artifact".to_string(),
                    )
                })?;
            let samples_len = samples.len();
            let chunk_plan = Self::asr_chunk_plan(
                samples.as_ref(),
                sample_rate,
                prepared_model.max_audio_seconds_hint(),
                false,
                false,
            );
            if chunk_plan.requires_chunk_path() {
                return Err(Error::InvalidInput(
                    "managed Qwen3 ASR cannot switch a retained sequence row to the long-audio chunk executor"
                        .to_string(),
                ));
            }

            let max_new_tokens = request.params.max_tokens.clamp(1, MAX_ASR_NEW_TOKENS);
            let cache = managed_cache.take().ok_or_else(|| {
                Error::InferenceError(
                    "Qwen3 ASR prefill lost its managed-cache reservation".to_string(),
                )
            })?;
            let decode_state = if begins_resumable_asr_prefill_state(scheduled, resumable_prefill) {
                run_asr_model_call(request, || {
                    Self::run_blocking(|| {
                        prepared_model.start_resumable_prefill_from_prepared_audio_managed(
                            prepared_audio.as_ref(),
                            request.asr_language_for_execution(),
                            request.asr_prompt_for_execution(),
                            max_new_tokens,
                            cache,
                        )
                    })
                })?
            } else {
                run_asr_model_call(request, || {
                    Self::run_blocking(|| {
                        prepared_model.start_decode_state_from_prepared_audio_managed(
                            prepared_audio.as_ref(),
                            request.asr_language_for_execution(),
                            request.asr_prompt_for_execution(),
                            max_new_tokens,
                            cache,
                        )
                    })
                })?
            };
            let Some(mut decode_state) = decode_state else {
                state_lease.release()?;
                return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    request.id.clone(),
                )));
            };
            if decode_state.prefill_token_count() != Some(prompt_tokens) {
                return Err(Error::InferenceError(
                    "Qwen3 ASR prepared multimodal span does not match scheduler admission"
                        .to_string(),
                ));
            }
            let expected_position = if resumable_prefill { 0 } else { prompt_tokens };
            if decode_state.sequence_position() != Some(expected_position)
                || decode_state.prefill_progress() != Some(expected_position)
            {
                return Err(Error::InferenceError(
                    "Qwen3 ASR decoder state started at an unexpected prefill cursor".to_string(),
                ));
            }
            if let Some(reservation) = tensor_reservation {
                decode_state.bind_qwen3_tensor_sequence(reservation.sequence)?;
            }
            let model_lease = request
                .prepared_asr_model_lease_for_executor()?
                .ok_or_else(|| {
                    Error::InferenceError(
                        "Qwen3 ASR sequence request lost its model residency lease".to_string(),
                    )
                })?;
            state_lease.install_state(ActiveAsrDecode {
                variant,
                model: prepared_model.clone(),
                _model_lease: model_lease,
                state: decode_state,
                last_tokens_generated: 0,
                stream_sequence: 0,
                input_sample_rate: sample_rate,
                input_sample_count: samples_len,
            })?;
            fresh_state = true;
        }

        if request.is_cancelled() {
            rollback_scalar_asr_quantum(
                &mut state_lease,
                &mut checkpoint,
                outer_checkpoint,
                fresh_state,
            )?;
            state_lease.release()?;
            return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                request.id.clone(),
            )));
        }

        state_lease.mark_dirty();
        let execution = (|| {
            let active_state = state_lease.require_state_mut()?;
            let mut decode_steps_ran = 0usize;
            let mut total_tokens_generated = 0usize;
            let mut final_text = String::new();
            let mut finished = false;
            let mut cancelled = false;
            let mut stream_events = Vec::new();
            let iterations = if scheduled.is_prefill {
                1
            } else {
                scheduled.num_tokens.max(1)
            };

            for _ in 0..iterations {
                if request.is_cancelled() {
                    cancelled = true;
                    break;
                }
                let step = if let Some((span_start, span_end)) = resumable_span {
                    let Some(prefill_complete) = run_asr_model_call(request, || {
                        Self::run_blocking(|| {
                            active_state.model.continue_resumable_prefill(
                                &mut active_state.state,
                                span_start,
                                span_end,
                            )
                        })
                    })?
                    else {
                        cancelled = true;
                        break;
                    };
                    if prefill_complete {
                        let Some(step) = run_asr_model_call(request, || {
                            Self::run_blocking(|| {
                                active_state.model.decode_step(&mut active_state.state)
                            })
                        })?
                        else {
                            cancelled = true;
                            break;
                        };
                        step
                    } else {
                        NativeAsrDecodeStep {
                            delta: String::new(),
                            text: String::new(),
                            tokens_generated: active_state.last_tokens_generated,
                            finished: false,
                        }
                    }
                } else {
                    let Some(step) = run_asr_model_call(request, || {
                        Self::run_blocking(|| {
                            active_state.model.decode_step(&mut active_state.state)
                        })
                    })?
                    else {
                        cancelled = true;
                        break;
                    };
                    step
                };
                if request.is_cancelled() {
                    cancelled = true;
                    break;
                }
                if !scheduled.is_prefill
                    || resumable_span.is_some_and(|(_, end)| end == prompt_tokens)
                {
                    decode_steps_ran = decode_steps_ran.saturating_add(1);
                }
                let step_tokens_generated = step
                    .tokens_generated
                    .saturating_sub(active_state.last_tokens_generated);
                active_state.last_tokens_generated = step.tokens_generated;
                total_tokens_generated =
                    total_tokens_generated.saturating_add(step_tokens_generated);
                final_text = step.text.clone();
                stream_events.push((step.delta, step.finished));
                if step.finished {
                    finished = true;
                    break;
                }
            }

            if !cancelled {
                if let Some(arena) = tensor_arena {
                    active_state
                        .state
                        .stage_qwen3_prepared_tensor_state(arena, scheduled.plan_id)?;
                }
                cancelled = request.is_cancelled();
            }
            if !cancelled {
                if let Some(tx) = Self::stream_sender(request) {
                    for (delta, event_finished) in stream_events {
                        if request.is_cancelled() {
                            cancelled = true;
                            break;
                        }
                        if !delta.is_empty() {
                            Self::stream_text_with_policy(
                                &tx,
                                request.stream_policy,
                                &request.id,
                                &mut active_state.stream_sequence,
                                delta,
                            )?;
                        }
                        if event_finished {
                            Self::stream_final_marker_with_policy(
                                &tx,
                                request.stream_policy,
                                &request.id,
                                &mut active_state.stream_sequence,
                            )?;
                        }
                    }
                }
                cancelled |= request.is_cancelled();
            }
            if cancelled {
                // All Qwen3 ASR output is commit-fenced. Drain this quantum's
                // staged events before returning cancellation so no text from a
                // rolled-back physical call can be attached to its result.
                let _ = request.take_staged_stream_outputs()?;
            }
            let completions = if cancelled {
                Vec::new()
            } else {
                active_state.state.take_managed_write_completions()
            };
            Ok((
                decode_steps_ran,
                total_tokens_generated,
                final_text,
                finished,
                cancelled,
                active_state.input_sample_rate,
                active_state.input_sample_count,
                completions,
            ))
        })();

        let (
            decode_steps_ran,
            total_tokens_generated,
            final_text,
            finished,
            cancelled,
            input_sample_rate,
            input_sample_count,
            managed_cache_completions,
        ) = match execution {
            Ok(execution) => execution,
            Err(error) => {
                rollback_scalar_asr_quantum(
                    &mut state_lease,
                    &mut checkpoint,
                    outer_checkpoint,
                    fresh_state,
                )?;
                return Err(error);
            }
        };

        if cancelled {
            rollback_scalar_asr_quantum(
                &mut state_lease,
                &mut checkpoint,
                outer_checkpoint,
                fresh_state,
            )?;
            state_lease.release()?;
            return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                request.id.clone(),
            )));
        }
        drop(checkpoint);

        let tokens_processed = if scheduled.is_prefill {
            scheduled.num_tokens
        } else {
            decode_steps_ran
        };
        if finished {
            state_lease.release()?;
        } else {
            state_lease.restore()?;
        }

        Ok(ModelSessionResult::sequence(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(AudioOutput {
                samples: Vec::new(),
                sample_rate: input_sample_rate,
                duration_secs: if input_sample_rate > 0 {
                    input_sample_count as f32 / input_sample_rate as f32
                } else {
                    0.0
                },
            }),
            text: Some(final_text),
            input_transcription: None,
            tokens_processed,
            tokens_generated: total_tokens_generated,
            finished,
            phase_timing_override: initial_media_decode_ms
                .map(ExecutorPhaseTiming::with_media_decode_ms),
            asr_diagnostics: None,
            error: None,
        })
        .with_managed_cache_completions(managed_cache_completions))
    }

    pub(super) fn vibevoice_prefill_batch_with_managed(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        mut managed_caches: Vec<Option<super::RetainedRowManagedState>>,
    ) -> Result<(Vec<ModelSessionResult>, bool)> {
        if scheduled.is_empty()
            || managed_caches.len() != scheduled.len()
            || scheduled.iter().any(|row| !row.is_prefill)
        {
            return Err(Error::InvalidInput(
                "static VibeVoice prefill requires one managed prefill row per request".into(),
            ));
        }
        let ordered_requests = scheduled
            .iter()
            .map(|scheduled| {
                requests
                    .iter()
                    .copied()
                    .find(|request| request.id == scheduled.request_id)
                    .ok_or_else(|| {
                        Error::InferenceError(format!(
                            "VibeVoice prefill request {} is missing its snapshot",
                            scheduled.request_id
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut outputs = (0..scheduled.len())
            .map(|_| None)
            .collect::<Vec<Option<ModelSessionResult>>>();
        let live_indices = ordered_requests
            .iter()
            .enumerate()
            .filter_map(|(index, request)| {
                if request.is_cancelled() {
                    outputs[index] = Some(ModelSessionResult::cancelled_before_dispatch(
                        ExecutorOutput::cancelled(request.id.clone()),
                    ));
                    None
                } else {
                    Some(index)
                }
            })
            .collect::<Vec<_>>();
        if live_indices.is_empty() {
            let outputs = outputs
                .into_iter()
                .map(|output| {
                    output.ok_or_else(|| {
                        Error::InferenceError(
                            "cancelled VibeVoice prefill row produced no result".into(),
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            return Ok((outputs, false));
        }

        let model = ordered_requests[live_indices[0]]
            .prepared_asr_model_for_executor()?
            .ok_or_else(|| {
                Error::InferenceError(
                    "VibeVoice prefill batch lost its loaded model identity".into(),
                )
            })?;
        for index in live_indices.iter().copied() {
            let request = ordered_requests[index];
            if Self::resolve_variant(request)?.family() != ModelFamily::VibeVoiceAsr {
                return Err(Error::InvalidInput(
                    "static VibeVoice prefill batch crossed ASR families".into(),
                ));
            }
            let row_model = request.prepared_asr_model_for_executor()?.ok_or_else(|| {
                Error::InferenceError("VibeVoice prefill row lost its loaded model identity".into())
            })?;
            if !Arc::ptr_eq(&model, &row_model) {
                return Err(Error::InvalidInput(
                    "static VibeVoice prefill batch spans loaded model instances".into(),
                ));
            }
            if managed_caches[index].is_none() {
                return Err(Error::InferenceError(
                    "VibeVoice prefill row lost its managed reservation".into(),
                ));
            }
        }

        let mut tokenizer_inputs = Vec::new();
        let mut tokenizer_indices = Vec::new();
        for index in live_indices.iter().copied() {
            let request = ordered_requests[index];
            let row = managed_caches[index]
                .as_ref()
                .expect("validated VibeVoice managed row");
            let spans = match &scheduled[index].work {
                crate::engine::WorkUnit::SequenceStep {
                    auxiliary_state: Some(spans),
                    ..
                } => spans.as_ref(),
                _ => &[],
            };
            if spans.len() > 1 || spans.is_empty() != row.tensor_state.is_none() {
                return Err(Error::InferenceError(
                    "VibeVoice prefill row tensor reservation does not match its projected span"
                        .into(),
                ));
            }
            let Some(span) = spans.first() else {
                continue;
            };
            let arena = request
                .managed_cache_runtime()
                .and_then(|runtime| runtime.tensor_state())
                .cloned()
                .ok_or_else(|| {
                    Error::InferenceError(
                        "VibeVoice prefill row lost its retained tokenizer arena".into(),
                    )
                })?;
            let transaction = PhysicalStateTransactionId::new(scheduled[index].plan_id)?;
            let decoder_span =
                resumable_asr_prefill_span(&scheduled[index], request.num_prompt_tokens())?;
            let artifact = if scheduled[index].num_computed_tokens == 0 {
                request
                    .prepared_vibevoice_artifact_for_executor()?
                    .ok_or_else(|| {
                        Error::InferenceError(
                            "initial VibeVoice prefill lost its prepared artifact".into(),
                        )
                    })?
            } else {
                let session = scheduled[index].session_key();
                let lease = ExecutorStateLease::checkout(
                    &self.asr_decode_states,
                    session,
                    "VibeVoice prefill artifact snapshot",
                )?;
                let active = lease.state().ok_or_else(|| {
                    Error::InferenceError(
                        "continuing VibeVoice prefill has no active decode state".into(),
                    )
                })?;
                if active.variant.family() != ModelFamily::VibeVoiceAsr
                    || !Arc::ptr_eq(&active.model, &model)
                {
                    return Err(Error::InferenceError(
                        "continuing VibeVoice prefill state crossed model identity".into(),
                    ));
                }
                let artifact = active.state.vibevoice_prepared_artifact().ok_or_else(|| {
                    Error::InferenceError(
                        "continuing VibeVoice prefill state lost its artifact".into(),
                    )
                })?;
                lease.restore()?;
                artifact
            };
            tokenizer_inputs.push(VibeVoiceAsrRetainedPrefillBatchRow {
                artifact,
                span_start: decoder_span.0,
                span_end: decoder_span.1,
                tokenizer_quantum: VibeVoiceAsrRetainedTokenizerQuantum::new(
                    arena,
                    transaction,
                    span.clone(),
                ),
            });
            tokenizer_indices.push(index);
        }

        let used_native_tokenizer_batch = tokenizer_inputs.len() > 1;
        let prepared_tokenizer = if tokenizer_inputs.is_empty() {
            Vec::new()
        } else {
            Self::run_blocking(|| {
                model.prepare_vibevoice_retained_tokenizer_batch(&tokenizer_inputs)
            })?
        };
        if prepared_tokenizer.len() != tokenizer_indices.len() {
            return Err(Error::InferenceError(
                "VibeVoice tokenizer batch returned the wrong number of prepared rows".into(),
            ));
        }
        if !tokenizer_inputs.is_empty() {
            crate::engine::metrics::record_engine_model_call(if tokenizer_inputs.len() > 1 {
                crate::engine::metrics::EngineModelCall::NativeTensor {
                    mode: crate::engine::NativeBatchMode::Static,
                    rows: tokenizer_inputs.len(),
                }
            } else {
                crate::engine::metrics::EngineModelCall::ScalarRows {
                    envelope: crate::engine::NativeBatchMode::Static,
                    rows: 1,
                }
            });
        }
        let mut prepared_by_index = (0..scheduled.len())
            .map(|_| None)
            .collect::<Vec<Option<VibeVoiceAsrPreparedTokenizerSpan>>>();
        for (index, prepared) in tokenizer_indices.into_iter().zip(prepared_tokenizer) {
            prepared_by_index[index] = Some(prepared);
        }

        let mut scalar_calls = 0usize;
        for index in live_indices {
            scalar_calls += 1;
            let result = self
                .vibevoice_asr_sequence_request_inner(
                    ordered_requests[index],
                    &scheduled[index],
                    managed_caches[index].take(),
                    prepared_by_index[index].as_ref(),
                )
                .unwrap_or_else(|error| {
                    ModelSessionResult::sequence(ExecutorOutput::error(
                        ordered_requests[index].id.clone(),
                        error.to_string(),
                    ))
                });
            outputs[index] = Some(result);
        }
        // The first scalar decoder row may finish while a later row is still
        // executing. Recheck the entire cohort only after the final row so a
        // cancellation that arrived during a peer call cannot publish the
        // earlier row's staged stream events or completion-bearing result.
        for (index, request) in ordered_requests.iter().enumerate() {
            if !late_cancelled_batch_row(request.is_cancelled(), outputs[index].is_some()) {
                continue;
            }
            let _ = request.take_staged_stream_outputs()?;
            let _ = crate::engine::ModelExecutor::cleanup_session(
                self,
                &scheduled[index].session_key(),
            );
            outputs[index] = Some(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                request.id.clone(),
            )));
        }
        if scalar_calls > 0 {
            crate::engine::metrics::record_engine_model_call(
                crate::engine::metrics::EngineModelCall::ScalarRows {
                    envelope: crate::engine::NativeBatchMode::Static,
                    rows: scalar_calls,
                },
            );
        }
        let outputs = outputs
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| {
                    Error::InferenceError("VibeVoice prefill row produced no result".into())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((outputs, used_native_tokenizer_batch))
    }

    pub(super) fn asr_decode_batch_with_managed(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        managed_caches: Vec<Option<super::RetainedRowManagedState>>,
    ) -> Result<Vec<ModelSessionResult>> {
        validate_continuous_asr_batch_shape(scheduled)?;
        if managed_caches.len() != scheduled.len() {
            return Err(Error::InvalidInput(
                "continuous ASR managed-cache rows do not match batch width".to_string(),
            ));
        }
        let ordered_requests = scheduled
            .iter()
            .map(|scheduled| {
                requests
                    .iter()
                    .copied()
                    .find(|request| request.id == scheduled.request_id)
                    .ok_or_else(|| {
                        Error::InferenceError(format!(
                            "continuous ASR request {} is missing its snapshot",
                            scheduled.request_id
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let live_indices = ordered_requests
            .iter()
            .enumerate()
            .filter_map(|(index, request)| (!request.is_cancelled()).then_some(index))
            .collect::<Vec<_>>();
        let mut outputs = (0..scheduled.len())
            .map(|_| None)
            .collect::<Vec<Option<ModelSessionResult>>>();
        for (index, request) in ordered_requests.iter().enumerate() {
            if request.is_cancelled() {
                outputs[index] = Some(ModelSessionResult::cancelled_before_dispatch(
                    ExecutorOutput::cancelled(request.id.clone()),
                ));
            }
        }
        if live_indices.is_empty() {
            return outputs
                .into_iter()
                .map(|output| {
                    output.ok_or_else(|| {
                        Error::InferenceError("cancelled ASR row produced no result".into())
                    })
                })
                .collect();
        }

        let model = ordered_requests[live_indices[0]]
            .prepared_asr_model_for_executor()?
            .ok_or_else(|| {
                Error::InferenceError(
                    "continuous ASR request has no exact loaded model identity".to_string(),
                )
            })?;
        let batch_family = Self::resolve_variant(ordered_requests[live_indices[0]])?.family();
        if !matches!(
            batch_family,
            ModelFamily::Qwen3Asr | ModelFamily::VibeVoiceAsr | ModelFamily::GraniteSpeechAsr
        ) {
            return Err(Error::InvalidInput(
                "continuous ASR batch contains a model without retained tensor decode".into(),
            ));
        }
        if !model.supports_continuous_decode_batch() {
            return Err(Error::InvalidInput(
                "loaded ASR model has no continuous tensor decode adapter".to_string(),
            ));
        }
        for index in live_indices.iter().copied().skip(1) {
            let row_model = ordered_requests[index]
                .prepared_asr_model_for_executor()?
                .ok_or_else(|| {
                    Error::InferenceError(
                        "continuous ASR row has no exact loaded model identity".to_string(),
                    )
                })?;
            if !Arc::ptr_eq(&model, &row_model) {
                return Err(Error::InferenceError(
                    "continuous ASR batch spans different loaded model instances".to_string(),
                ));
            }
        }

        let mut checked_out_states = Vec::with_capacity(live_indices.len());
        for index in live_indices.iter().copied() {
            let request = ordered_requests[index];
            let session = scheduled[index].session_key();
            let expected_variant = Self::resolve_variant(request)?;
            if expected_variant.family() != batch_family {
                return Err(Error::InvalidInput(
                    "continuous ASR batch spans different ASR families".to_string(),
                ));
            }
            let lease = ExecutorStateLease::checkout(
                &self.asr_decode_states,
                session.clone(),
                "continuous ASR decode",
            )?;
            let state = lease.state().ok_or_else(|| {
                Error::InferenceError(format!(
                    "continuous ASR session {}:{} has no active decode state",
                    session.request_id, session.epoch
                ))
            })?;
            if state.variant != expected_variant || !Arc::ptr_eq(&state.model, &model) {
                return Err(Error::InferenceError(
                    "continuous ASR state identity does not match its request".to_string(),
                ));
            }
            checked_out_states.push((index, session, lease));
        }

        let mut active_states = ContinuousAsrStateBatch::new(checked_out_states);
        let mut managed_caches = managed_caches;
        for (index, _, lease, checkpoint) in &mut active_states.rows {
            let request = ordered_requests[*index];
            let managed_cache = managed_caches[*index].take();
            if request.managed_cache_runtime().is_some() != managed_cache.is_some() {
                return Err(Error::InferenceError(
                    "continuous ASR row lost its managed-cache reservation".to_string(),
                ));
            }
            let mut views = managed_cache.ok_or_else(|| {
                Error::InferenceError(
                    "continuous ASR decode requires retained physical KV".to_string(),
                )
            })?;
            let tensor_reservation = views.tensor_state.clone();
            let tensor_arena = request
                .managed_cache_runtime()
                .and_then(|runtime| runtime.tensor_state());
            if batch_family != ModelFamily::VibeVoiceAsr
                && tensor_arena.is_some() != tensor_reservation.is_some()
            {
                return Err(Error::InferenceError(
                    "continuous ASR row lost its prepared-input tensor-state reservation"
                        .to_string(),
                ));
            }
            if batch_family == ModelFamily::VibeVoiceAsr && tensor_reservation.is_some() {
                return Err(Error::InferenceError(
                    "normal VibeVoice ASR decode received forbidden tokenizer tensor state".into(),
                ));
            }
            let cache = views.take_only_paged()?;
            let state = lease.require_state_mut()?;
            let last_tokens_generated = state.last_tokens_generated;
            let stream_sequence = state.stream_sequence;
            let native_checkpoint = state.state.begin_managed_quantum(cache)?;
            *checkpoint = Some((native_checkpoint, last_tokens_generated, stream_sequence));
            if let (Some(arena), Some(reservation)) = (tensor_arena, tensor_reservation) {
                state
                    .state
                    .bind_qwen3_tensor_sequence(reservation.sequence)?;
                state.state.restore_qwen3_prepared_tensor_state(arena)?;
            }
            lease.mark_dirty();
        }

        // Cancellation may arrive after reservations were rebound but before
        // the native launch. Roll those rows back now and exclude them from the
        // physical batch; the remaining rows keep their independent checkpoints.
        for row in 0..active_states.rows.len() {
            let index = active_states.rows[row].0;
            let request = ordered_requests[index];
            if request.is_cancelled() {
                let index = active_states.rollback_row(row)?;
                let _ = request.take_staged_stream_outputs()?;
                outputs[index] = Some(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    request.id.clone(),
                )));
            }
        }
        let call_rows = active_states
            .rows
            .iter()
            .enumerate()
            .filter_map(|(row, (index, _, _, _))| outputs[*index].is_none().then_some(row))
            .collect::<Vec<_>>();
        let mut state_refs = active_states
            .rows
            .iter_mut()
            .filter(|(index, _, _, _)| outputs[*index].is_none())
            .map(|(_, _, lease, _)| lease.require_state_mut().map(|state| &mut state.state))
            .collect::<Result<Vec<_>>>()?;
        let live_width = state_refs.len();
        let steps = if state_refs.is_empty() {
            Vec::new()
        } else if state_refs.len() == 1 {
            vec![Self::run_blocking(|| model.decode_step(state_refs[0]))?]
        } else {
            Self::run_blocking(|| model.decode_step_batch(&mut state_refs))?
        };
        drop(state_refs);
        // One native batch call may outlive cancellation of any subset of its
        // rows. Observe every row before staging tensor state, emitting text,
        // or detaching write completions while all checkpoints remain armed.
        let cancelled_after_model = call_rows
            .iter()
            .map(|row| ordered_requests[active_states.rows[*row].0].is_cancelled())
            .collect::<Vec<_>>();
        if steps.len() != call_rows.len() {
            return Err(Error::InferenceError(
                "continuous ASR model returned the wrong number of rows".to_string(),
            ));
        }
        let model_call = if live_width > 1 && model.continuous_decode_is_tensor_batched() {
            crate::engine::metrics::EngineModelCall::NativeTensor {
                mode: crate::engine::NativeBatchMode::Continuous,
                rows: live_width,
            }
        } else {
            crate::engine::metrics::EngineModelCall::ScalarRows {
                envelope: crate::engine::NativeBatchMode::Continuous,
                rows: live_width,
            }
        };
        crate::engine::metrics::record_engine_model_call(model_call);

        for (row, cancelled) in call_rows
            .iter()
            .copied()
            .zip(cancelled_after_model.into_iter())
        {
            if !cancelled {
                continue;
            }
            let index = active_states.rollback_row(row)?;
            let request = ordered_requests[index];
            let _ = request.take_staged_stream_outputs()?;
            outputs[index] = Some(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                request.id.clone(),
            )));
        }

        for row in 0..active_states.rows.len() {
            let index = active_states.rows[row].0;
            if outputs[index].is_some() {
                continue;
            }
            let request = ordered_requests[index];
            if batch_family == ModelFamily::Qwen3Asr {
                if let Some(arena) = request
                    .managed_cache_runtime()
                    .and_then(|runtime| runtime.tensor_state())
                {
                    active_states.rows[row]
                        .2
                        .require_state_mut()?
                        .state
                        .stage_qwen3_prepared_tensor_state(arena, scheduled[index].plan_id)?;
                }
            }
            if request.is_cancelled() {
                let index = active_states.rollback_row(row)?;
                let request = ordered_requests[index];
                let _ = request.take_staged_stream_outputs()?;
                outputs[index] = Some(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    request.id.clone(),
                )));
            }
        }

        let mut continuing = vec![false; scheduled.len()];
        for (row, step) in call_rows.iter().copied().zip(steps.into_iter()) {
            let index = active_states.rows[row].0;
            if outputs[index].is_some() {
                continue;
            }
            let request = ordered_requests[index];
            if request.is_cancelled() {
                let index = active_states.rollback_row(row)?;
                let request = ordered_requests[index];
                let _ = request.take_staged_stream_outputs()?;
                outputs[index] = Some(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    request.id.clone(),
                )));
                continue;
            }

            let (step_tokens_generated, input_sample_rate, input_sample_count, stream_result) = {
                let active_state = active_states.rows[row].2.require_state_mut()?;
                let step_tokens_generated = step
                    .tokens_generated
                    .saturating_sub(active_state.last_tokens_generated);
                active_state.last_tokens_generated = step.tokens_generated;
                let stream_result = (|| -> Result<()> {
                    let Some(tx) = Self::stream_sender(request) else {
                        return Ok(());
                    };
                    if !step.delta.is_empty() {
                        Self::stream_text_with_policy(
                            &tx,
                            request.stream_policy,
                            &request.id,
                            &mut active_state.stream_sequence,
                            step.delta.clone(),
                        )?;
                    }
                    if step.finished {
                        Self::stream_final_marker_with_policy(
                            &tx,
                            request.stream_policy,
                            &request.id,
                            &mut active_state.stream_sequence,
                        )?;
                    }
                    Ok(())
                })();
                (
                    step_tokens_generated,
                    active_state.input_sample_rate,
                    active_state.input_sample_count,
                    stream_result,
                )
            };
            if let Err(error) = stream_result {
                let _ = request.take_staged_stream_outputs();
                if batch_family == ModelFamily::VibeVoiceAsr {
                    for active_index in live_indices.iter().copied() {
                        let _ = ordered_requests[active_index].take_staged_stream_outputs();
                    }
                    return Err(Error::InferenceError(format!(
                        "continuous VibeVoice ASR stream staging failed: {error}"
                    )));
                }
                let index = active_states.rollback_row(row)?;
                outputs[index] = Some(ModelSessionResult::sequence(ExecutorOutput::error(
                    request.id.clone(),
                    format!("continuous ASR stream staging failed: {error}"),
                )));
                continue;
            }
            if request.is_cancelled() {
                let _ = request.take_staged_stream_outputs()?;
                let index = active_states.rollback_row(row)?;
                outputs[index] = Some(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    request.id.clone(),
                )));
                continue;
            }

            let managed_cache_completions = active_states.rows[row]
                .2
                .require_state_mut()?
                .state
                .take_managed_write_completions();
            outputs[index] = Some(
                ModelSessionResult::sequence(ExecutorOutput {
                    request_id: request.id.clone(),
                    audio: Some(AudioOutput {
                        samples: Vec::new(),
                        sample_rate: input_sample_rate,
                        duration_secs: if input_sample_rate > 0 {
                            input_sample_count as f32 / input_sample_rate as f32
                        } else {
                            0.0
                        },
                    }),
                    text: Some(step.text),
                    input_transcription: None,
                    tokens_processed: 1,
                    tokens_generated: step_tokens_generated,
                    finished: step.finished,
                    phase_timing_override: None,
                    asr_diagnostics: None,
                    error: None,
                })
                .with_managed_cache_completions(managed_cache_completions),
            );
            if !step.finished {
                continuing[index] = true;
            }
        }

        // Recheck every still-armed row immediately before committing any
        // checkpoint. Earlier rows may be cancelled while later rows are being
        // sampled or staged; their private outputs and detached completions must
        // be discarded with the rollback.
        for row in 0..active_states.rows.len() {
            let index = active_states.rows[row].0;
            if late_cancelled_batch_row(
                ordered_requests[index].is_cancelled(),
                active_states.rows[row].3.is_some(),
            ) {
                let _ = ordered_requests[index].take_staged_stream_outputs()?;
                let index = active_states.rollback_row(row)?;
                continuing[index] = false;
                outputs[index] = Some(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    ordered_requests[index].id.clone(),
                )));
            }
        }

        let committed_states = match active_states.commit() {
            Ok(states) => states,
            Err(error) => {
                for index in live_indices.iter().copied() {
                    let _ = ordered_requests[index].take_staged_stream_outputs();
                }
                return Err(error);
            }
        };
        for (index, _, lease) in committed_states {
            let transition = if continuing[index] {
                lease.restore()
            } else {
                lease.release()
            };
            if let Err(error) = transition {
                outputs[index] = Some(ModelSessionResult::sequence(ExecutorOutput::error(
                    ordered_requests[index].id.clone(),
                    format!("continuous ASR state transition failed: {error}"),
                )));
            }
        }
        outputs
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| {
                    Error::InferenceError("continuous ASR row produced no result".into())
                })
            })
            .collect()
    }

    pub(super) fn transcribe_request_with_managed_cache(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        mut managed_state: Option<super::RetainedRowManagedState>,
    ) -> Result<ModelSessionResult> {
        if request.managed_cache_runtime().is_some() != managed_state.is_some() {
            return Err(Error::InferenceError(
                "Qwen3 ASR request and scheduler reservation disagree on retained-state authority"
                    .to_string(),
            ));
        }
        let variant = Self::resolve_variant(request)?;
        let family = variant.family();
        if family == ModelFamily::Voxtral && request.is_realtime_asr_session() {
            let retained = managed_state.take().ok_or_else(|| {
                Error::InferenceError("Voxtral realtime ASR lost retained paged state".into())
            })?;
            return self.voxtral_realtime_request(request, scheduled, retained);
        }
        if family == ModelFamily::Qwen3Asr && request.uses_asr_retained_sequence() {
            return self.qwen3_asr_sequence_request(request, scheduled, managed_state.take());
        }
        if family == ModelFamily::WhisperAsr && request.uses_asr_retained_sequence() {
            return self.whisper_asr_sequence_request(request, scheduled, managed_state.take());
        }
        if family == ModelFamily::VibeVoiceAsr && request.uses_asr_retained_sequence() {
            return self.vibevoice_asr_sequence_request(request, scheduled, managed_state.take());
        }
        if family == ModelFamily::GraniteSpeechAsr && request.uses_asr_retained_sequence() {
            return self.granite_speech_asr_sequence_request(
                request,
                scheduled,
                managed_state.take(),
            );
        }
        if managed_state.is_some() {
            return Err(Error::InferenceError(
                "retained ASR state was routed outside a retained sequence executor".to_string(),
            ));
        }
        let mut managed_cache = None;
        let language = request.asr_language_for_execution();
        let asr_prompt = request.asr_prompt_for_execution();
        let generation_options = Self::asr_generation_options(request);
        let stream_tx = Self::stream_sender(request);
        let stream_policy = request.stream_policy;
        let session = scheduled.session_key();

        if let Some(tx) = stream_tx.as_ref() {
            if !matches!(family, ModelFamily::Voxtral) {
                let mut state_lease =
                    ExecutorStateLease::checkout(&self.asr_decode_states, session, "ASR decode")?;
                if state_lease
                    .state()
                    .map(|state| state.variant != variant)
                    .unwrap_or(false)
                {
                    state_lease.discard_state();
                }
                let (model, new_model_lease) = if let Some(state) = state_lease.state() {
                    (state.model.clone(), None)
                } else {
                    let (model, lease) = self.asr_model_for_request(request, variant)?;
                    (model, Some(lease))
                };

                if model.supports_incremental_decode()
                    && !matches!(family, ModelFamily::NemotronAsr)
                    && !request.uses_asr_long_form_atomic()
                {
                    let mut initial_media_decode_ms = None;
                    if state_lease.state().is_some() {
                        if let Some(cache) = managed_cache.take() {
                            state_lease.mark_dirty();
                            state_lease
                                .require_state_mut()?
                                .state
                                .install_qwen3_managed_reservation(cache)?;
                        }
                    } else {
                        if request.is_cancelled() {
                            state_lease.release()?;
                            return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                                request.id.clone(),
                            )));
                        }
                        let audio_decode_started = Instant::now();
                        let (samples, sample_rate) =
                            Self::run_blocking(|| decode_request_audio_with_rate(request))?;
                        let audio_decode_ms = audio_decode_started.elapsed().as_secs_f64() * 1000.0;
                        initial_media_decode_ms = Some(audio_decode_ms);
                        let samples_len = samples.len();

                        let chunk_plan = Self::asr_chunk_plan(
                            &samples,
                            sample_rate,
                            model.max_audio_seconds_hint(),
                            false,
                            matches!(family, ModelFamily::WhisperAsr),
                        );
                        if chunk_plan.requires_chunk_path() {
                            if managed_cache.is_some() {
                                return Err(Error::InvalidInput(
                                    "managed Qwen3 ASR cannot switch a scheduled sequence row to the chunked executor"
                                        .to_string(),
                                ));
                            }
                            let mut sequence = 0usize;
                            let chunk_stream_options = if matches!(family, ModelFamily::Qwen3Asr) {
                                Self::qwen_asr_chunk_stream_options()
                            } else {
                                Default::default()
                            };
                            let chunked = Self::run_blocking(|| {
                                Self::transcribe_with_chunk_plan_with_details_and_options(
                                    &request.id,
                                    Some(tx),
                                    stream_policy,
                                    &mut sequence,
                                    &samples,
                                    sample_rate,
                                    &chunk_plan.chunks,
                                    &chunk_plan.config,
                                    chunk_stream_options,
                                    |chunk_audio, sr, _prefix_text| {
                                        let details = model
                                            .transcribe_with_details_and_prompt_and_options(
                                                chunk_audio,
                                                sr,
                                                language,
                                                asr_prompt,
                                                generation_options.clone(),
                                            )?;
                                        Ok(AsrChunkTranscription {
                                            text: details.text,
                                            diagnostics: details.diagnostics,
                                        })
                                    },
                                )
                            })?;
                            let diagnostics = Self::with_audio_decode_timing(
                                Some(chunk_plan.diagnostics_with_chunk_transcriptions(
                                    chunked.chunk_diagnostics,
                                )),
                                audio_decode_ms,
                            );

                            return Ok(ModelSessionResult::atomic(ExecutorOutput {
                                request_id: request.id.clone(),
                                audio: Some(AudioOutput {
                                    samples: Vec::new(),
                                    sample_rate,
                                    duration_secs: if sample_rate > 0 {
                                        samples_len as f32 / sample_rate as f32
                                    } else {
                                        0.0
                                    },
                                }),
                                text: Some(chunked.text),
                                input_transcription: None,
                                tokens_processed: request.num_prompt_tokens(),
                                tokens_generated: (samples_len / 256).max(1),
                                finished: true,
                                phase_timing_override: Some(
                                    ExecutorPhaseTiming::with_media_decode_ms(audio_decode_ms),
                                ),
                                asr_diagnostics: diagnostics,
                                error: None,
                            }));
                        }

                        // Keep ASR decode bounded. If EOS is missed, very high caps
                        // produce runaway gibberish and extreme latency.
                        let max_new_tokens = request.params.max_tokens.clamp(1, MAX_ASR_NEW_TOKENS);
                        let cache = managed_cache.take().ok_or_else(|| {
                            Error::InferenceError(
                                "incremental Qwen3 ASR requires scheduler-owned physical KV"
                                    .to_string(),
                            )
                        })?;
                        let decode_state = Self::run_blocking(|| {
                            model.start_decode_state_with_prompt_managed(
                                &samples,
                                sample_rate,
                                language,
                                asr_prompt,
                                max_new_tokens,
                                cache,
                            )
                        })?;
                        if decode_state.sequence_position() != Some(request.num_prompt_tokens()) {
                            return Err(Error::InferenceError(
                                "Qwen3 ASR prepared multimodal span does not match model prefill"
                                    .to_string(),
                            ));
                        }
                        state_lease.install_state(ActiveAsrDecode {
                            variant,
                            model: model.clone(),
                            _model_lease: new_model_lease
                                .expect("a new ASR decode state must retain its model lease"),
                            state: decode_state,
                            last_tokens_generated: 0,
                            stream_sequence: 0,
                            input_sample_rate: sample_rate,
                            input_sample_count: samples_len,
                        })?;
                    }

                    if request.is_cancelled() {
                        state_lease.release()?;
                        return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                            request.id.clone(),
                        )));
                    }

                    let decode_iterations = if scheduled.is_prefill {
                        1
                    } else {
                        scheduled.num_tokens.max(1)
                    };
                    state_lease.mark_dirty();
                    let (
                        tokens_processed,
                        total_tokens_generated,
                        final_text,
                        finished,
                        input_sample_rate,
                        input_sample_count,
                        managed_cache_completions,
                    ) = {
                        let active_state = state_lease.require_state_mut()?;
                        let mut decode_steps_ran = 0usize;
                        let mut total_tokens_generated = 0usize;
                        let mut final_text = String::new();
                        let mut finished = false;

                        for _ in 0..decode_iterations {
                            if request.is_cancelled() {
                                return Ok(ModelSessionResult::cancelled(
                                    ExecutorOutput::cancelled(request.id.clone()),
                                ));
                            }
                            let step = Self::run_blocking(|| {
                                active_state.model.decode_step(&mut active_state.state)
                            })?;
                            if request.is_cancelled() {
                                return Ok(ModelSessionResult::cancelled(
                                    ExecutorOutput::cancelled(request.id.clone()),
                                ));
                            }
                            decode_steps_ran = decode_steps_ran.saturating_add(1);
                            let step_tokens_generated = step
                                .tokens_generated
                                .saturating_sub(active_state.last_tokens_generated);
                            active_state.last_tokens_generated = step.tokens_generated;
                            total_tokens_generated =
                                total_tokens_generated.saturating_add(step_tokens_generated);
                            final_text = step.text.clone();

                            if !step.delta.is_empty() {
                                Self::stream_text_with_policy(
                                    tx,
                                    stream_policy,
                                    &request.id,
                                    &mut active_state.stream_sequence,
                                    step.delta,
                                )?;
                            }
                            if step.finished {
                                Self::stream_final_marker_with_policy(
                                    tx,
                                    stream_policy,
                                    &request.id,
                                    &mut active_state.stream_sequence,
                                )?;
                                finished = true;
                                break;
                            }
                        }

                        let tokens_processed = if scheduled.is_prefill {
                            request.num_prompt_tokens()
                        } else {
                            decode_steps_ran.max(1)
                        };
                        let input_sample_rate = active_state.input_sample_rate;
                        let input_sample_count = active_state.input_sample_count;
                        let managed_cache_completions =
                            active_state.state.take_managed_write_completions();
                        (
                            tokens_processed,
                            total_tokens_generated,
                            final_text,
                            finished,
                            input_sample_rate,
                            input_sample_count,
                            managed_cache_completions,
                        )
                    };

                    if finished {
                        state_lease.release()?;
                    } else {
                        state_lease.restore()?;
                    }

                    return Ok(ModelSessionResult::sequence(ExecutorOutput {
                        request_id: request.id.clone(),
                        audio: Some(AudioOutput {
                            samples: Vec::new(),
                            sample_rate: input_sample_rate,
                            duration_secs: if input_sample_rate > 0 {
                                input_sample_count as f32 / input_sample_rate as f32
                            } else {
                                0.0
                            },
                        }),
                        text: Some(final_text),
                        input_transcription: None,
                        tokens_processed,
                        tokens_generated: total_tokens_generated,
                        finished,
                        phase_timing_override: initial_media_decode_ms
                            .map(ExecutorPhaseTiming::with_media_decode_ms),
                        asr_diagnostics: None,
                        error: None,
                    })
                    .with_managed_cache_completions(managed_cache_completions));
                }
                if managed_cache.is_some() {
                    return Err(Error::InferenceError(
                        "managed Qwen3 ASR reservation reached a non-incremental model".to_string(),
                    ));
                }
            }
        }

        if managed_cache.is_some() {
            return Err(Error::InferenceError(
                "managed Qwen3 ASR reservation requires the incremental streaming graph"
                    .to_string(),
            ));
        }

        let (execution_audio, sample_rate, audio_decode_ms) =
            resolve_asr_execution_audio(request, family, || {
                Self::run_blocking(|| decode_request_audio_with_rate(request))
            })?;
        let samples = execution_audio.samples();
        let samples_len = samples.len();

        let (text, asr_diagnostics) = Self::run_blocking(|| {
            let mut sequence = 0usize;
            if matches!(family, ModelFamily::Voxtral) {
                let model = self.with_registry(|registry| {
                    registry.try_get_voxtral_lease(variant).ok_or_else(|| {
                        Error::ModelNotFound(format!(
                            "Voxtral model {variant} is not loaded in registry"
                        ))
                    })
                })?;

                let chunk_plan = Self::asr_chunk_plan(&samples, sample_rate, None, false, false);
                if chunk_plan.requires_chunk_path() {
                    let chunked = Self::transcribe_with_chunk_plan_with_details(
                        &request.id,
                        stream_tx.as_ref(),
                        stream_policy,
                        &mut sequence,
                        &samples,
                        sample_rate,
                        &chunk_plan.chunks,
                        &chunk_plan.config,
                        |chunk_audio, sr| {
                            with_single_invocation_cache(request, scheduled, |cache| {
                                model.transcribe_with_details_physical(
                                    chunk_audio,
                                    sr,
                                    language,
                                    cache,
                                )
                            })
                            .map(|details| AsrChunkTranscription {
                                text: details.text,
                                diagnostics: details.diagnostics,
                            })
                        },
                    )?;
                    return Ok((
                        chunked.text,
                        Some(
                            chunk_plan
                                .diagnostics_with_chunk_transcriptions(chunked.chunk_diagnostics),
                        ),
                    ));
                }

                if request.streaming {
                    if let Some(tx) = stream_tx.as_ref() {
                        let mut stream_err: Option<Error> = None;
                        let mut emit = |delta: &str| {
                            if stream_err.is_none() {
                                if let Err(err) = Self::stream_text_with_policy(
                                    tx,
                                    stream_policy,
                                    &request.id,
                                    &mut sequence,
                                    delta.to_string(),
                                ) {
                                    stream_err = Some(err);
                                }
                            }
                        };
                        let text = with_single_invocation_cache(request, scheduled, |cache| {
                            model.transcribe_with_callback_physical(
                                &samples,
                                sample_rate,
                                language,
                                &mut emit,
                                cache,
                            )
                        })?;
                        if let Some(err) = stream_err {
                            return Err(err);
                        }
                        Self::stream_final_marker_with_policy(
                            tx,
                            stream_policy,
                            &request.id,
                            &mut sequence,
                        )?;
                        return Ok((text, None));
                    }
                }
                let details = with_single_invocation_cache(request, scheduled, |cache| {
                    model.transcribe_with_details_physical(&samples, sample_rate, language, cache)
                })?;
                return Ok((details.text, details.diagnostics));
            }

            let (model, _model_lease) = self.asr_model_for_request(request, variant)?;

            let chunk_plan = Self::asr_chunk_plan(
                &samples,
                sample_rate,
                model.max_audio_seconds_hint(),
                request.streaming && !model.supports_incremental_decode(),
                matches!(family, ModelFamily::WhisperAsr),
            );
            if chunk_plan.requires_chunk_path() {
                let chunked = Self::transcribe_with_chunk_plan_with_context_and_details(
                    &request.id,
                    stream_tx.as_ref(),
                    stream_policy,
                    &mut sequence,
                    &samples,
                    sample_rate,
                    &chunk_plan.chunks,
                    &chunk_plan.config,
                    |chunk_audio, sr, prefix_text| {
                        let bounded_prefix_text = matches!(family, ModelFamily::GraniteSpeechAsr)
                            .then(|| Self::granite_asr_prefix_replay_text(prefix_text));
                        let prefix_text = bounded_prefix_text
                            .as_deref()
                            .filter(|value| !value.trim().is_empty());
                        let chunk_generation_options = Self::asr_chunk_generation_options(
                            request,
                            family,
                            chunk_audio.len(),
                            sr,
                            &generation_options,
                        );
                        let mut details = match family {
                            ModelFamily::Qwen3Asr => {
                                with_single_invocation_cache(request, scheduled, |cache| {
                                    model.transcribe_qwen3_with_details_and_prompt_physical(
                                        chunk_audio,
                                        sr,
                                        language,
                                        asr_prompt,
                                        cache,
                                    )
                                })?
                            }
                            ModelFamily::VibeVoiceAsr => {
                                with_vibevoice_invocation_state(request, scheduled, |leases| {
                                    model.transcribe_vibevoice_with_details_and_prompt_and_options_physical(
                                        chunk_audio,
                                        sr,
                                        language,
                                        asr_prompt,
                                        chunk_generation_options.clone(),
                                        leases,
                                    )
                                })?
                            }
                            ModelFamily::GraniteSpeechAsr => {
                                with_single_invocation_cache(request, scheduled, |cache| {
                                    model.transcribe_granite_speech_with_details_prompt_prefix_and_options_physical(
                                        chunk_audio,
                                        sr,
                                        language,
                                        asr_prompt,
                                        prefix_text,
                                        chunk_generation_options.clone(),
                                        cache,
                                    )
                                })?
                            }
                            ModelFamily::WhisperAsr => with_whisper_invocation_state(
                                request,
                                scheduled,
                                |self_kv, cross_kv| {
                                    model.transcribe_whisper_with_details_and_prompt_physical(
                                        chunk_audio,
                                        sr,
                                        language,
                                        asr_prompt,
                                        self_kv,
                                        cross_kv,
                                    )
                                },
                            )?,
                            ModelFamily::ParakeetAsr => {
                                with_single_invocation_tensor(request, scheduled, |state| {
                                    model.transcribe_parakeet_with_details_physical(
                                        chunk_audio,
                                        sr,
                                        language,
                                        state,
                                    )
                                })?
                            }
                            ModelFamily::NemotronAsr => with_nemotron_offline_state(
                                request,
                                scheduled,
                                |predictor, acoustic| {
                                    model.transcribe_nemotron_with_details_and_prompt_physical(
                                        chunk_audio,
                                        sr,
                                        language,
                                        asr_prompt,
                                        predictor,
                                        acoustic,
                                    )
                                },
                            )?,
                            _ => model.transcribe_with_details_prompt_prefix_and_options(
                                chunk_audio,
                                sr,
                                language,
                                asr_prompt,
                                prefix_text,
                                chunk_generation_options.clone(),
                            )?,
                        };
                        if matches!(family, ModelFamily::GraniteSpeechAsr) {
                            details = Self::recover_granite_chunk_loop(
                                &model,
                                chunk_audio,
                                sr,
                                language,
                                asr_prompt,
                                prefix_text,
                                &chunk_generation_options,
                                details,
                                request,
                                scheduled,
                            )?;
                        }
                        Ok(AsrChunkTranscription {
                            text: details.text,
                            diagnostics: details.diagnostics,
                        })
                    },
                )?;
                return Ok((
                    chunked.text,
                    Some(
                        chunk_plan.diagnostics_with_chunk_transcriptions(chunked.chunk_diagnostics),
                    ),
                ));
            }

            // Granite's public auto budget is admitted at its conservative
            // ceiling. Once the audio has been decoded under that admission,
            // narrow a single-segment decode to the actual duration just as
            // the chunked path does for each chunk.
            let segment_generation_options = Self::asr_chunk_generation_options(
                request,
                family,
                samples.len(),
                sample_rate,
                &generation_options,
            );

            if request.streaming {
                if let Some(tx) = stream_tx.as_ref() {
                    let mut stream_err: Option<Error> = None;
                    let mut emit = |delta: &str| {
                        if stream_err.is_none() {
                            if let Err(err) = Self::stream_text_with_policy(
                                tx,
                                stream_policy,
                                &request.id,
                                &mut sequence,
                                delta.to_string(),
                            ) {
                                stream_err = Some(err);
                            }
                        }
                    };
                    let text = match family {
                        ModelFamily::VibeVoiceAsr => {
                            with_vibevoice_invocation_state(request, scheduled, |leases| {
                                model
                                    .transcribe_vibevoice_with_callback_and_prompt_and_options_physical(
                                        &samples,
                                        sample_rate,
                                        language,
                                        asr_prompt,
                                        segment_generation_options.clone(),
                                        leases,
                                        &mut emit,
                                    )
                            })?
                        }
                        ModelFamily::GraniteSpeechAsr => {
                            with_single_invocation_cache(request, scheduled, |cache| {
                                model
                                    .transcribe_granite_speech_with_callback_and_prompt_and_options_physical(
                                        &samples,
                                        sample_rate,
                                        language,
                                        asr_prompt,
                                        segment_generation_options.clone(),
                                        cache,
                                        &mut emit,
                                    )
                            })?
                        }
                        ModelFamily::WhisperAsr => with_whisper_invocation_state(
                            request,
                            scheduled,
                            |self_kv, cross_kv| {
                                model.transcribe_whisper_with_callback_and_prompt_physical(
                                    &samples,
                                    sample_rate,
                                    language,
                                    asr_prompt,
                                    self_kv,
                                    cross_kv,
                                    &mut emit,
                                )
                            },
                        )?,
                        ModelFamily::ParakeetAsr => {
                            with_single_invocation_tensor(request, scheduled, |state| {
                                model.transcribe_parakeet_with_callback_physical(
                                    &samples,
                                    sample_rate,
                                    language,
                                    state,
                                    &mut emit,
                                )
                            })?
                        }
                        ModelFamily::NemotronAsr => with_nemotron_offline_state(
                            request,
                            scheduled,
                            |predictor, acoustic| {
                                model.transcribe_nemotron_with_callback_and_prompt_physical(
                                    &samples,
                                    sample_rate,
                                    language,
                                    asr_prompt,
                                    predictor,
                                    acoustic,
                                    &mut emit,
                                )
                            },
                        )?,
                        _ => model.transcribe_with_callback_and_prompt_and_options(
                            &samples,
                            sample_rate,
                            language,
                            asr_prompt,
                            segment_generation_options.clone(),
                            &mut emit,
                        )?,
                    };
                    if let Some(err) = stream_err {
                        return Err(err);
                    }
                    Self::stream_final_marker_with_policy(
                        tx,
                        stream_policy,
                        &request.id,
                        &mut sequence,
                    )?;
                    return Ok((text, None));
                }
            }
            let details = match family {
                ModelFamily::Qwen3Asr => {
                    with_single_invocation_cache(request, scheduled, |cache| {
                        model.transcribe_qwen3_with_details_and_prompt_physical(
                            &samples,
                            sample_rate,
                            language,
                            asr_prompt,
                            cache,
                        )
                    })?
                }
                ModelFamily::VibeVoiceAsr => {
                    with_vibevoice_invocation_state(request, scheduled, |leases| {
                        model.transcribe_vibevoice_with_details_and_prompt_and_options_physical(
                            &samples,
                            sample_rate,
                            language,
                            asr_prompt,
                            segment_generation_options.clone(),
                            leases,
                        )
                    })?
                }
                ModelFamily::GraniteSpeechAsr => {
                    with_single_invocation_cache(request, scheduled, |cache| {
                        model
                            .transcribe_granite_speech_with_details_and_prompt_and_options_physical(
                                &samples,
                                sample_rate,
                                language,
                                asr_prompt,
                                segment_generation_options.clone(),
                                cache,
                            )
                    })?
                }
                ModelFamily::WhisperAsr => {
                    with_whisper_invocation_state(request, scheduled, |self_kv, cross_kv| {
                        model.transcribe_whisper_with_details_and_prompt_physical(
                            &samples,
                            sample_rate,
                            language,
                            asr_prompt,
                            self_kv,
                            cross_kv,
                        )
                    })?
                }
                ModelFamily::ParakeetAsr => {
                    with_single_invocation_tensor(request, scheduled, |state| {
                        model.transcribe_parakeet_with_details_physical(
                            &samples,
                            sample_rate,
                            language,
                            state,
                        )
                    })?
                }
                ModelFamily::NemotronAsr => {
                    with_nemotron_offline_state(request, scheduled, |predictor, acoustic| {
                        model.transcribe_nemotron_with_details_and_prompt_physical(
                            &samples,
                            sample_rate,
                            language,
                            asr_prompt,
                            predictor,
                            acoustic,
                        )
                    })?
                }
                _ => model.transcribe_with_details_and_prompt_and_options(
                    &samples,
                    sample_rate,
                    language,
                    asr_prompt,
                    segment_generation_options,
                )?,
            };
            Ok((details.text, details.diagnostics))
        })?;
        let asr_diagnostics = Self::with_audio_decode_timing(asr_diagnostics, audio_decode_ms);

        Ok(ModelSessionResult::atomic(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(AudioOutput {
                samples: Vec::new(),
                sample_rate,
                duration_secs: samples_len as f32 / sample_rate as f32,
            }),
            text: Some(text),
            input_transcription: None,
            tokens_processed: request.num_prompt_tokens(),
            tokens_generated: (samples_len / 256).max(1),
            finished: true,
            phase_timing_override: Some(ExecutorPhaseTiming::with_media_decode_ms(audio_decode_ms)),
            asr_diagnostics,
            error: None,
        }))
    }

    fn with_audio_decode_timing(
        diagnostics: Option<serde_json::Value>,
        audio_decode_ms: f64,
    ) -> Option<serde_json::Value> {
        let mut payload = diagnostics.unwrap_or_else(|| serde_json::json!({}));
        if !payload.is_object() {
            payload = serde_json::json!({
                "model_diagnostics": payload
            });
        }

        if let Some(root) = payload.as_object_mut() {
            let timings = root
                .entry("timings_ms")
                .or_insert_with(|| serde_json::json!({}));
            if let Some(timings_obj) = timings.as_object_mut() {
                timings_obj.insert(
                    "audio_decode".to_string(),
                    serde_json::json!(audio_decode_ms),
                );
            } else {
                root.insert(
                    "timings_ms".to_string(),
                    serde_json::json!({ "audio_decode": audio_decode_ms }),
                );
            }
        }

        Some(payload)
    }

    fn asr_generation_options(request: &EngineCoreRequest) -> NativeAsrGenerationOptions {
        NativeAsrGenerationOptions {
            max_new_tokens: request.params.max_tokens.max(1),
            stop_token_ids: request.params.stop_token_ids.clone(),
            stop_sequences: request.params.stop_sequences.clone(),
        }
    }

    fn asr_chunk_generation_options(
        request: &EngineCoreRequest,
        family: ModelFamily,
        chunk_sample_count: usize,
        sample_rate: u32,
        base: &NativeAsrGenerationOptions,
    ) -> NativeAsrGenerationOptions {
        let mut options = base.clone();
        if request.asr_auto_max_tokens
            && matches!(family, ModelFamily::GraniteSpeechAsr)
            && sample_rate > 0
        {
            let chunk_seconds = chunk_sample_count as f32 / sample_rate as f32;
            options.max_new_tokens =
                granite_auto_asr_max_tokens_for_duration(chunk_seconds).min(base.max_new_tokens);
        }
        options.max_new_tokens = options.max_new_tokens.max(1);
        options
    }

    fn granite_asr_prefix_replay_words() -> usize {
        Self::env_usize("IZWI_GRANITE_ASR_PREFIX_REPLAY_WORDS")
            .unwrap_or(GRANITE_ASR_PREFIX_REPLAY_WORDS)
            .min(GRANITE_ASR_PREFIX_REPLAY_WORDS_MAX)
    }

    fn granite_asr_prefix_replay_text(prefix_text: &str) -> String {
        recent_word_suffix(prefix_text, Self::granite_asr_prefix_replay_words())
    }

    #[allow(clippy::too_many_arguments)]
    fn recover_granite_chunk_loop(
        model: &crate::models::registry::NativeAsrModel,
        chunk_audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        asr_prompt: Option<&str>,
        prefix_text: Option<&str>,
        base_options: &NativeAsrGenerationOptions,
        mut details: crate::models::registry::NativeAsrTranscription,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
    ) -> Result<crate::models::registry::NativeAsrTranscription> {
        let Some(original_loop) =
            granite_chunk_loop_signal(&details.text, details.diagnostics.as_ref())
        else {
            return Ok(details);
        };
        let retry_max_tokens = granite_loop_recovery_max_tokens(
            chunk_audio.len(),
            sample_rate,
            base_options.max_new_tokens,
        );
        if retry_max_tokens >= base_options.max_new_tokens {
            let trim = trim_granite_looping_chunk_text(&mut details.text, &original_loop);
            details.diagnostics = with_granite_loop_recovery_diagnostics(
                details.diagnostics,
                &original_loop,
                None,
                base_options.max_new_tokens,
                retry_max_tokens,
                false,
                if trim.is_some() {
                    "not_retried_budget_floor_trimmed_original"
                } else {
                    "not_retried_budget_floor"
                },
                trim.as_ref(),
            );
            return Ok(details);
        }

        let mut retry_options = base_options.clone();
        retry_options.max_new_tokens = retry_max_tokens;
        let mut retry = with_single_invocation_cache(request, scheduled, |cache| {
            model.transcribe_granite_speech_with_details_prompt_prefix_and_options_physical(
                chunk_audio,
                sample_rate,
                language,
                asr_prompt,
                prefix_text,
                retry_options,
                cache,
            )
        })?;
        let retry_loop = granite_chunk_loop_signal(&retry.text, retry.diagnostics.as_ref());
        let retry_trim = retry_loop
            .as_ref()
            .and_then(|loop_signal| trim_granite_looping_chunk_text(&mut retry.text, loop_signal));
        let use_retry = !retry.text.trim().is_empty()
            && (retry_trim.is_some()
                || retry_loop
                    .as_ref()
                    .map(|loop_signal| loop_signal.score() < original_loop.score())
                    .unwrap_or(true));

        if use_retry {
            retry.diagnostics = with_granite_loop_recovery_diagnostics(
                retry.diagnostics,
                &original_loop,
                retry_loop.as_ref(),
                base_options.max_new_tokens,
                retry_max_tokens,
                true,
                if retry_trim.is_some() {
                    "retry_selected_trimmed"
                } else {
                    "retry_selected"
                },
                retry_trim.as_ref(),
            );
            Ok(retry)
        } else {
            let original_trim = trim_granite_looping_chunk_text(&mut details.text, &original_loop);
            details.diagnostics = with_granite_loop_recovery_diagnostics(
                details.diagnostics,
                &original_loop,
                retry_loop.as_ref(),
                base_options.max_new_tokens,
                retry_max_tokens,
                false,
                if original_trim.is_some() {
                    "retry_not_better_trimmed_original"
                } else {
                    "retry_not_better"
                },
                original_trim.as_ref(),
            );
            Ok(details)
        }
    }
}

fn recent_word_suffix(text: &str, max_words: usize) -> String {
    if max_words == 0 {
        return String::new();
    }
    let words = text.split_whitespace().collect::<Vec<_>>();
    if words.len() <= max_words {
        return text.trim().to_string();
    }
    words[words.len() - max_words..].join(" ")
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GranitePhraseLoop {
    phrase: String,
    phrase_words: usize,
    repeat_count: usize,
    start_word: usize,
    trailing_words: usize,
}

impl GranitePhraseLoop {
    fn score(&self) -> usize {
        self.phrase_words.saturating_mul(self.repeat_count)
    }
}

fn granite_chunk_loop_signal(
    text: &str,
    diagnostics: Option<&serde_json::Value>,
) -> Option<GranitePhraseLoop> {
    if diagnostics
        .and_then(|value| value.pointer("/decode/stop_reason"))
        .and_then(|value| value.as_str())
        != Some("max_tokens")
    {
        return None;
    }
    repeated_phrase_loop(text)
}

fn repeated_phrase_loop(text: &str) -> Option<GranitePhraseLoop> {
    let words = normalized_words(text);
    if words.len() < 9 {
        return None;
    }
    let mut best: Option<GranitePhraseLoop> = None;
    let max_phrase_words = 12.min(words.len() / 3);
    for phrase_words in 3..=max_phrase_words {
        let mut idx = 0usize;
        while idx + phrase_words * 3 <= words.len() {
            let phrase = &words[idx..idx + phrase_words];
            let mut repeats = 1usize;
            while idx + phrase_words * (repeats + 1) <= words.len()
                && words[idx + phrase_words * repeats..idx + phrase_words * (repeats + 1)]
                    == *phrase
            {
                repeats += 1;
            }
            if repeats >= 3 {
                let repeated_words = phrase_words.saturating_mul(repeats);
                let candidate = GranitePhraseLoop {
                    phrase: phrase.join(" "),
                    phrase_words,
                    repeat_count: repeats,
                    start_word: idx,
                    trailing_words: words.len().saturating_sub(idx + repeated_words),
                };
                if best
                    .as_ref()
                    .map(|current| candidate.score() > current.score())
                    .unwrap_or(true)
                {
                    best = Some(candidate);
                }
                idx += phrase_words * repeats;
            } else {
                idx += 1;
            }
        }
    }
    best
}

fn normalized_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|ch| ch.is_alphanumeric() || matches!(ch, '\'' | '-'))
                .flat_map(|ch| ch.to_lowercase())
                .collect::<String>()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

#[derive(Debug, Clone)]
struct NormalizedWordSpan {
    end_byte: usize,
}

fn normalized_word_spans(text: &str) -> Vec<NormalizedWordSpan> {
    let mut spans = Vec::new();
    let mut current = String::new();
    let mut current_end = 0usize;
    for (idx, ch) in text.char_indices() {
        if ch.is_alphanumeric() || matches!(ch, '\'' | '-') {
            current.extend(ch.to_lowercase());
            current_end = idx + ch.len_utf8();
        } else if !current.is_empty() {
            spans.push(NormalizedWordSpan {
                end_byte: current_end,
            });
            current.clear();
        }
    }
    if !current.is_empty() {
        spans.push(NormalizedWordSpan {
            end_byte: current_end,
        });
    }
    spans
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GraniteLoopTrim {
    original_chars: usize,
    trimmed_chars: usize,
    original_words: usize,
    trimmed_words: usize,
}

fn trim_repeated_phrase_tail(text: &str, loop_signal: &GranitePhraseLoop) -> Option<String> {
    if loop_signal.repeat_count < 3 || loop_signal.trailing_words > loop_signal.phrase_words.max(4)
    {
        return None;
    }
    let spans = normalized_word_spans(text);
    let keep_words = loop_signal
        .start_word
        .saturating_add(loop_signal.phrase_words);
    let trim_end = spans.get(keep_words.saturating_sub(1))?.end_byte;
    let trimmed = text[..trim_end].trim_end();
    if trimmed.len() < text.trim_end().len() {
        Some(trimmed.to_string())
    } else {
        None
    }
}

fn trim_granite_looping_chunk_text(
    text: &mut String,
    loop_signal: &GranitePhraseLoop,
) -> Option<GraniteLoopTrim> {
    let original_chars = text.chars().count();
    let original_words = normalized_words(text).len();
    let trimmed = trim_repeated_phrase_tail(text, loop_signal)?;
    let trimmed_chars = trimmed.chars().count();
    let trimmed_words = normalized_words(&trimmed).len();
    *text = trimmed;
    Some(GraniteLoopTrim {
        original_chars,
        trimmed_chars,
        original_words,
        trimmed_words,
    })
}

fn granite_loop_recovery_max_tokens(
    chunk_sample_count: usize,
    sample_rate: u32,
    original_max_tokens: usize,
) -> usize {
    if original_max_tokens <= 1 {
        return original_max_tokens;
    }
    let chunk_seconds = if sample_rate > 0 {
        chunk_sample_count as f32 / sample_rate as f32
    } else {
        0.0
    };
    let auto_for_chunk = granite_auto_asr_max_tokens_for_duration(chunk_seconds);
    let reduced = original_max_tokens.saturating_mul(3) / 4;
    reduced
        .min(auto_for_chunk)
        .max(24.min(original_max_tokens.saturating_sub(1)))
        .min(original_max_tokens.saturating_sub(1))
        .max(1)
}

fn with_granite_loop_recovery_diagnostics(
    diagnostics: Option<serde_json::Value>,
    original_loop: &GranitePhraseLoop,
    retry_loop: Option<&GranitePhraseLoop>,
    original_max_tokens: usize,
    retry_max_tokens: usize,
    selected_retry: bool,
    decision: &str,
    trim: Option<&GraniteLoopTrim>,
) -> Option<serde_json::Value> {
    let mut diagnostics = diagnostics.unwrap_or_else(|| json!({}));
    if !diagnostics.is_object() {
        diagnostics = json!({ "model_diagnostics": diagnostics });
    }
    if let Some(root) = diagnostics.as_object_mut() {
        root.insert(
            "chunk_loop_recovery".to_string(),
            json!({
                "triggered": true,
                "decision": decision,
                "selected_retry": selected_retry,
                "original_max_new_tokens": original_max_tokens,
                "retry_max_new_tokens": retry_max_tokens,
                "original_loop": {
                    "phrase": original_loop.phrase,
                    "phrase_words": original_loop.phrase_words,
                    "repeat_count": original_loop.repeat_count,
                },
                "retry_loop": retry_loop.map(|loop_signal| json!({
                    "phrase": loop_signal.phrase,
                    "phrase_words": loop_signal.phrase_words,
                    "repeat_count": loop_signal.repeat_count,
                })),
                "trim": trim.map(|trim| json!({
                    "original_chars": trim.original_chars,
                    "trimmed_chars": trim.trimmed_chars,
                    "original_words": trim.original_words,
                    "trimmed_words": trim.trimmed_words,
                })),
            }),
        );
    }
    Some(diagnostics)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    use super::NativeExecutor;
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
    fn vibevoice_legacy_execution_reuses_prepared_audio_and_fails_closed_without_it() {
        let variant = ModelVariant::VibeVoiceAsr;
        let mut request = EngineCoreRequest::asr_bytes(vec![1, 2, 3]).with_model_variant(variant);
        request
            .install_prepared_asr_audio(variant, vec![0.25, -0.5, 0.75], 24_000)
            .unwrap();
        request.install_prepared_asr_long_form_atomic().unwrap();
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
        };

        let options = NativeExecutor::asr_chunk_generation_options(
            &request,
            ModelFamily::GraniteSpeechAsr,
            16_000 * 30,
            16_000,
            &base,
        );

        assert_eq!(options.max_new_tokens, 84);
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
        };

        let options = NativeExecutor::asr_chunk_generation_options(
            &request,
            ModelFamily::GraniteSpeechAsr,
            16_000 * 30,
            16_000,
            &base,
        );

        assert_eq!(options.max_new_tokens, 84);
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
        };

        let options = NativeExecutor::asr_chunk_generation_options(
            &request,
            ModelFamily::GraniteSpeechAsr,
            16_000 * 30,
            16_000,
            &base,
        );

        assert_eq!(options.max_new_tokens, 2048);
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
}
