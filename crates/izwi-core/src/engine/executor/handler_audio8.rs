//! Audio8 TTS execution: prefill and decode on the family's retained state,
//! codec frames decoded in batched quanta, and a finalize stage that flushes
//! the stream.
//!
//! Every model quantum runs inside the family's quantum checkpoint, and the
//! checkpoint and each codec candidate wait in the family's pending
//! coordinator for the core's commit or abort. Cancellation and the deadline
//! are checked before and after every model call.

use std::sync::Arc;
use std::time::Instant;

use crate::catalog::ModelFamily;
use crate::engine::InvocationPagedKvLease;
use crate::error::{Error, Result};

use super::super::request::EngineCoreRequest;
use super::super::scheduler::ScheduledRequest;
use super::super::types::AudioOutput;
use super::super::SessionKey;
use super::handler_tts::retained_tts_batch_model_call;
use super::state::ActiveAudio8TtsDecode;
use super::{ExecutorOutput, ExecutorStateLease, ModelSessionResult, NativeExecutor};

// If any later row rejects staging, roll back every candidate already handed
// to the coordinator before this physical invocation reports collective failure.
struct Audio8PendingBatchGuard<'a> {
    coordinator: &'a super::Audio8StateCoordinator,
    staged: Vec<(crate::engine::PlanId, SessionKey)>,
    armed: bool,
}
impl Drop for Audio8PendingBatchGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            for (plan, session) in &self.staged {
                super::PendingQuantumFinalizer::discard(self.coordinator, *plan, session);
            }
        }
    }
}

struct ContinuousAudio8Row<'a> {
    index: usize,
    lease: Option<ExecutorStateLease<'a, ActiveAudio8TtsDecode>>,
    checkpoint: Option<crate::models::architectures::audio8::Audio8RetainedCheckpoint>,
    fast: Option<InvocationPagedKvLease>,
    sampling_before: f64,
    fresh: bool,
}

impl ContinuousAudio8Row<'_> {
    fn rollback(&mut self) -> Result<()> {
        if self.fresh {
            // Initial prefill has no earlier managed cache to restore.
            // Discard its unpublished state instead of failing the cohort.
            if let Some(lease) = &mut self.lease {
                lease.discard_state();
            }
            self.checkpoint = None;
            self.fresh = false;
            return Ok(());
        }
        if let (Some(lease), Some(checkpoint)) = (&mut self.lease, &mut self.checkpoint) {
            lease
                .require_state_mut()?
                .state
                .rollback_managed_quantum(checkpoint)?;
            lease.mark_clean();
            self.checkpoint = None;
        }
        Ok(())
    }
}

impl Drop for ContinuousAudio8Row<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.rollback() {
            tracing::error!(%error, "Audio8 batch rollback failed; state remains fenced")
        }
    }
}

impl NativeExecutor {
    pub(super) fn audio8_tts_batch_with_managed(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        mut managed: Vec<Option<super::RetainedRowManagedState>>,
    ) -> Result<Vec<ModelSessionResult>> {
        if requests.is_empty()
            || requests.len() != scheduled.len()
            || managed.len() != scheduled.len()
        {
            return Err(Error::InvalidInput("Audio8 batch row counts differ".into()));
        }
        let model = requests[0]
            .prepared_audio8_tts_model_lease_for_executor()?
            .ok_or_else(|| Error::InferenceError("Audio8 batch lost model".into()))?;
        let variant = Self::resolve_variant(requests[0])?;
        let is_prefill = scheduled[0].is_prefill;
        let mut outputs: Vec<Option<ModelSessionResult>> =
            (0..requests.len()).map(|_| None).collect();
        let mut rows = Vec::with_capacity(requests.len());
        for (index, (request, scheduled_row)) in requests.iter().zip(scheduled).enumerate() {
            if scheduled_row.request_id != request.id
                || scheduled_row.is_prefill != is_prefill
                || Self::resolve_variant(request)? != variant
                || request
                    .execution_adapter_binding()
                    .map(|b| b.adapter_instance_id)
                    != requests[0]
                        .execution_adapter_binding()
                        .map(|b| b.adapter_instance_id)
            {
                return Err(Error::InvalidInput(
                    "Audio8 batch crossed request, phase or adapter identity".into(),
                ));
            }
            let row_model = request
                .prepared_audio8_tts_model_lease_for_executor()?
                .ok_or_else(|| Error::InferenceError("Audio8 batch row lost model".into()))?;
            if !Arc::ptr_eq(&row_model.model_arc(), &model.model_arc()) {
                return Err(Error::InvalidInput(
                    "Audio8 batch crossed model generation".into(),
                ));
            }
            if let Some(terminal) = audio8_row_terminal(request, false) {
                outputs[index] = Some(terminal);
                continue;
            }
            let mut retained = managed[index]
                .take()
                .ok_or_else(|| Error::InferenceError("Audio8 batch lost managed state".into()))?;
            let slow = retained
                .take_paged_domain(crate::kv::CacheDomainId::new(1), true)?
                .expect("required Audio8 slow cache");
            retained.ensure_all_paged_consumed()?;
            let mut lease = ExecutorStateLease::checkout(
                &self.audio8_tts_decode_states,
                scheduled_row.session_key(),
                variant,
                "Audio8 batch",
            )?;
            if lease
                .state()
                .is_some_and(|active| !Arc::ptr_eq(&active.model.model_arc(), &model.model_arc()))
            {
                return Err(Error::InvalidInput(
                    "Audio8 batch retained state crossed model".into(),
                ));
            }
            let fresh = lease.state().is_none();
            let checkpoint = if fresh {
                if !scheduled_row.is_prefill || scheduled_row.num_computed_tokens != 0 {
                    return Err(Error::InferenceError(
                        "Audio8 TTS lost state before initial prefill".into(),
                    ));
                }
                let artifact = request
                    .prepared_audio8_tts_artifact_for_executor()?
                    .ok_or_else(|| Error::InferenceError("Audio8 TTS lost prompt".into()))?;
                let params = request
                    .audio8_tts_generation_params_for_executor()?
                    .ok_or_else(|| Error::InferenceError("Audio8 TTS lost geometry".into()))?;
                let runtime = request.managed_cache_runtime().ok_or_else(|| {
                    Error::InferenceError("Audio8 TTS lost managed context limit".into())
                })?;
                let max_sequence_tokens = usize::try_from(runtime.maximum_sequence_tokens())
                    .map_err(|_| Error::InvalidInput("Audio8 context exceeds usize".into()))?;
                let (state, checkpoint) = model.new_retained_state_in_quantum(
                    artifact,
                    params,
                    slow,
                    max_sequence_tokens,
                )?;
                lease.install_state(ActiveAudio8TtsDecode {
                    variant,
                    model: model.clone(),
                    state,
                    last_frames_generated: 0,
                    stream_sequence: 0,
                    codec: Default::default(),
                    audio_samples: Vec::new(),
                    total_audio_samples: 0,
                    collect_audio_samples: request.collect_audio_samples,
                    codec_ms: 0.0,
                    execution_started: Instant::now(),
                    first_audio_ms: None,
                })?;
                checkpoint
            } else {
                lease
                    .require_state_mut()?
                    .state
                    .begin_managed_quantum(slow)?
            };

            let sampling_before = lease.require_state_mut()?.state.sampling_and_steps().0;
            lease.mark_dirty();
            let mut row = ContinuousAudio8Row {
                index,
                lease: Some(lease),
                checkpoint: Some(checkpoint),
                fast: None,
                sampling_before,
                fresh,
            };
            if !is_prefill {
                row.fast = Some(super::invocation_paged_lease_for_row(
                    request,
                    scheduled_row,
                )?);
            }
            rows.push(row);
        }
        let mut pending_guard = Audio8PendingBatchGuard {
            coordinator: &self.audio8_pending,
            staged: Vec::new(),
            armed: true,
        };
        if !rows.is_empty() {
            let width = rows.len();
            let mut states = Vec::with_capacity(width);
            let mut fast = Vec::with_capacity(width);
            let mut spans = Vec::with_capacity(width);
            for row in &mut rows {
                states.push(
                    &mut row
                        .lease
                        .as_mut()
                        .expect("Audio8 lease")
                        .require_state_mut()?
                        .state,
                );
                if let Some(cache) = &mut row.fast {
                    fast.push(cache.cache_mut());
                }
                spans.push(scheduled[row.index].num_tokens);
            }
            let steps = if is_prefill {
                model.retained_prefill_batch(&mut states, &spans)?
            } else {
                model.retained_decode_batch(&mut states, &mut fast)?
            };
            if steps.len() != width {
                return Err(Error::InferenceError(
                    "Audio8 batch returned wrong width".into(),
                ));
            }
            if let Some(call) = retained_tts_batch_model_call(
                if is_prefill {
                    crate::engine::NativeBatchMode::Static
                } else {
                    crate::engine::NativeBatchMode::Continuous
                },
                width,
            ) {
                crate::engine::metrics::record_engine_model_call(call);
            }
            // Codec frames due this step, gathered over every row.
            let due = audio8_due_frames(
                &rows
                    .iter()
                    .map(|row| {
                        let active = row
                            .lease
                            .as_ref()
                            .and_then(|lease| lease.state())
                            .ok_or_else(|| {
                                Error::InferenceError("Audio8 batch lost state".into())
                            })?;
                        Ok(Audio8CodecProgress {
                            decoded: active.codec.decoded_frames(),
                            generated: active.state.frames_generated(),
                            finished: active.state.finished(),
                            streamed: requests[row.index].streaming,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            );
            for ((row, step), due) in rows.iter_mut().zip(steps).zip(due) {
                let index = row.index;
                let request = requests[index];
                let scheduled_row = &scheduled[index];
                if let Some(cache) = row.fast.take() {
                    let _ = cache.release()?;
                }
                if let Some(terminal) = audio8_row_terminal(request, true) {
                    row.rollback()?;
                    outputs[index] = Some(terminal);
                    continue;
                }
                let lease = row.lease.as_mut().expect("Audio8 lease");
                let active = lease.require_state_mut()?;
                if is_prefill
                    && !matches!(step, crate::models::architectures::audio8::Audio8RetainedStep::Prefill { consumed, .. } if consumed == scheduled_row.num_tokens)
                {
                    return Err(Error::InferenceError(
                        "Audio8 prefill progress differs from scheduler".into(),
                    ));
                }
                let completions = active.state.take_managed_write_completions();
                if active.state.take_staged_step().as_ref() != Some(&step) {
                    return Err(Error::InferenceError(
                        "Audio8 batch staged output mismatch".into(),
                    ));
                }
                let frames = active.state.frames_generated();
                let generated = frames.saturating_sub(active.last_frames_generated);
                let timing = audio8_execution_timing(
                    active,
                    0.0,
                    active.state.sampling_and_steps().0 - row.sampling_before,
                    active.first_audio_ms,
                );
                let output = ExecutorOutput {
                    request_id: request.id.clone(),
                    audio: Some(AudioOutput::new(
                        Vec::new(),
                        model.diagnostics().sample_rate,
                    )),
                    text: None,
                    input_transcription: None,
                    tokens_processed: scheduled_row.num_tokens,
                    tokens_generated: generated,
                    finished: false,
                    phase_timing_override: Some(timing),
                    asr_diagnostics: None,
                    error: None,
                    logprobs: Vec::new(),
                };
                let result = if due > 0 {
                    ModelSessionResult::yielded(
                        output,
                        crate::engine::YieldReason::AwaitingAudioDecode { max_frames: due },
                    )
                } else if active.state.finished() {
                    ModelSessionResult::yielded(
                        output,
                        crate::engine::YieldReason::AwaitingFinalization,
                    )
                } else {
                    ModelSessionResult::sequence(output)
                };
                self.audio8_pending.stage_ar(
                    scheduled_row.plan_id,
                    scheduled_row.session_key(),
                    row.lease.take().expect("Audio8 lease"),
                    row.checkpoint.take().expect("Audio8 checkpoint"),
                )?;
                pending_guard
                    .staged
                    .push((scheduled_row.plan_id, scheduled_row.session_key()));
                outputs[index] = Some(
                    result
                        .with_managed_cache_completions(completions)
                        .requiring_pending_quantum(),
                );
            }
        }
        let result = outputs
            .into_iter()
            .map(|out| out.ok_or_else(|| Error::InferenceError("Audio8 batch lost output".into())))
            .collect::<Result<Vec<_>>>()?;
        pending_guard.armed = false;
        Ok(result)
    }

    pub(super) fn audio8_tts_request_with_managed_cache(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        retained: Option<super::RetainedRowManagedState>,
    ) -> Result<ModelSessionResult> {
        self.audio8_tts_batch_with_managed(
            &[request],
            std::slice::from_ref(scheduled),
            vec![retained],
        )?
        .pop()
        .ok_or_else(|| Error::InferenceError("Audio8 scalar quantum produced no result".into()))
    }

    pub(super) fn audio8_tts_audio_decode_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
    ) -> Result<ModelSessionResult> {
        self.audio8_tts_audio_decode_batch(&[request], std::slice::from_ref(scheduled))?
            .pop()
            .ok_or_else(|| Error::InferenceError("Audio8 codec lost scalar output".into()))
    }

    pub(super) fn audio8_tts_audio_decode_batch(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ModelSessionResult>> {
        if requests.is_empty() || requests.len() != scheduled.len() {
            return Err(Error::InvalidInput(
                "Audio8 codec batch row counts differ".into(),
            ));
        }
        let model = requests[0]
            .prepared_audio8_tts_model_lease_for_executor()?
            .ok_or_else(|| Error::InferenceError("Audio8 codec batch lost model".into()))?;
        let mut outputs: Vec<Option<ModelSessionResult>> =
            (0..requests.len()).map(|_| None).collect();
        let mut rows = Vec::new();
        let mut codecs = Vec::new();
        let mut frames = Vec::new();
        for (index, (request, scheduled_row)) in requests.iter().zip(scheduled).enumerate() {
            let crate::engine::WorkUnit::SequenceAudioDecode { max_frames } = scheduled_row.work
            else {
                return Err(Error::InvalidInput(
                    "Audio8 codec batch requires audio work".into(),
                ));
            };
            if request.id != scheduled_row.request_id
                || request
                    .execution_adapter_binding()
                    .map(|b| b.adapter_instance_id)
                    != requests[0]
                        .execution_adapter_binding()
                        .map(|b| b.adapter_instance_id)
            {
                return Err(Error::InvalidInput(
                    "Audio8 codec batch crossed adapter identity".into(),
                ));
            }
            if let Some(terminal) = audio8_row_terminal(request, false) {
                outputs[index] = Some(terminal);
                continue;
            }
            let variant = Self::resolve_variant(request)?;
            let row_model = request
                .prepared_audio8_tts_model_lease_for_executor()?
                .ok_or_else(|| Error::InferenceError("Audio8 codec row lost model".into()))?;
            let mut lease = ExecutorStateLease::checkout(
                &self.audio8_tts_decode_states,
                scheduled_row.session_key(),
                variant,
                "Audio8 codec batch",
            )?;
            let active = lease.require_state_mut()?;
            if active.variant != variant
                || !Arc::ptr_eq(&active.model.model_arc(), &model.model_arc())
                || !Arc::ptr_eq(&row_model.model_arc(), &model.model_arc())
            {
                return Err(Error::InvalidInput(
                    "Audio8 codec batch crossed model generation".into(),
                ));
            }
            let output_credit = if Self::stream_sender(request).is_some() {
                let bytes = max_frames
                    .checked_mul(2048 * std::mem::size_of::<f32>())
                    .ok_or_else(|| {
                        Error::Overloaded("Audio8 codec output credit overflow".into())
                    })?;
                let Some(credit) = request.try_reserve_audio_output(bytes)? else {
                    outputs[index] = Some(ModelSessionResult::yielded(
                        ExecutorOutput {
                            request_id: request.id.clone(),
                            audio: Some(AudioOutput::empty(model.diagnostics().sample_rate)),
                            text: None,
                            input_transcription: None,
                            tokens_processed: 0,
                            tokens_generated: 0,
                            finished: false,
                            phase_timing_override: None,
                            asr_diagnostics: None,
                            error: None,
                            logprobs: Vec::new(),
                        },
                        crate::engine::YieldReason::AwaitingAudioOutput { max_frames },
                    ));
                    continue;
                };
                Some(credit)
            } else {
                None
            };
            codecs.push(active.codec.clone());
            frames.push(max_frames);
            rows.push((index, lease, output_credit));
        }
        let mut pending_guard = Audio8PendingBatchGuard {
            coordinator: &self.audio8_pending,
            staged: Vec::new(),
            armed: true,
        };
        if !rows.is_empty() {
            let states = rows
                .iter_mut()
                .map(|(_, lease, _)| lease.require_state_mut().map(|active| &active.state))
                .collect::<Result<Vec<_>>>()?;
            let started = Instant::now();
            let chunks =
                model.decode_retained_audio_batch(&states, &mut codecs, &frames, &|| Ok(()))?;
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            if chunks.len() != rows.len() {
                return Err(Error::InferenceError(
                    "Audio8 codec returned wrong width".into(),
                ));
            }
            for (((index, mut lease, _credit), codec), samples) in
                rows.into_iter().zip(codecs).zip(chunks)
            {
                let request = requests[index];
                let scheduled_row = &scheduled[index];
                if let Some(terminal) = audio8_row_terminal(request, true) {
                    // The model wrote only candidate codec state. Dropping this
                    // clean lease and output credit preserves committed state.
                    outputs[index] = Some(terminal);
                    continue;
                }
                let active = lease.require_state_mut()?;
                let sample_rate = model.diagnostics().sample_rate;
                let mut sequence = active.stream_sequence;
                if let Some(tx) = Self::stream_sender(request) {
                    Self::stream_audio_with_policy(
                        &tx,
                        request.stream_policy,
                        &request.id,
                        &mut sequence,
                        samples.clone(),
                        sample_rate,
                        false,
                    )?;
                }
                let done = active.state.finished();
                // More whole chunks than one quantum decodes, or the tail.
                let due = Audio8CodecProgress {
                    decoded: codec.decoded_frames(),
                    generated: active.state.frames_generated(),
                    finished: done,
                    streamed: request.streaming,
                }
                .due_frames();
                let first_audio_ms = active
                    .first_audio_ms
                    .or_else(|| Some(active.execution_started.elapsed().as_secs_f64() * 1000.0));
                let timing = audio8_execution_timing(active, elapsed_ms, 0.0, first_audio_ms);
                self.audio8_pending.stage(
                    scheduled_row.plan_id,
                    scheduled_row.session_key(),
                    lease,
                    super::Audio8CodecCommit {
                        codec,
                        samples,
                        stream_sequence: sequence,
                        codec_ms: elapsed_ms,
                        first_audio_ms,
                    },
                )?;
                let output = ExecutorOutput {
                    request_id: request.id.clone(),
                    audio: Some(AudioOutput::empty(sample_rate)),
                    text: None,
                    input_transcription: None,
                    tokens_processed: 0,
                    tokens_generated: 0,
                    finished: false,
                    phase_timing_override: Some(timing),
                    asr_diagnostics: None,
                    error: None,
                    logprobs: Vec::new(),
                };
                let reason = if due > 0 {
                    crate::engine::YieldReason::AwaitingAudioDecode { max_frames: due }
                } else if done {
                    crate::engine::YieldReason::AwaitingFinalization
                } else {
                    crate::engine::YieldReason::QuantumExhausted
                };
                pending_guard
                    .staged
                    .push((scheduled_row.plan_id, scheduled_row.session_key()));
                outputs[index] =
                    Some(ModelSessionResult::yielded(output, reason).requiring_pending_quantum());
            }
        }
        let result = outputs
            .into_iter()
            .map(|out| {
                out.ok_or_else(|| Error::InferenceError("Audio8 codec batch lost output".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        pending_guard.armed = false;
        Ok(result)
    }

    pub(super) fn audio8_tts_finalize_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
    ) -> Result<ModelSessionResult> {
        if !matches!(
            scheduled.work,
            crate::engine::WorkUnit::SequenceFinalize {
                max_output_steps: 1
            }
        ) || scheduled.is_prefill
        {
            return Err(Error::InvalidInput(
                "Audio8 codec requires an exact sequence-finalize quantum".into(),
            ));
        }
        let variant = Self::resolve_variant(request)?;
        if variant.family() != ModelFamily::Audio8Tts {
            return Err(Error::InvalidInput("foreign Audio8 codec request".into()));
        }
        let model = request
            .prepared_audio8_tts_model_lease_for_executor()?
            .ok_or_else(|| Error::InferenceError("Audio8 codec lost model residency".into()))?;
        let mut lease = ExecutorStateLease::checkout(
            &self.audio8_tts_decode_states,
            scheduled.session_key(),
            variant,
            "Audio8 codec",
        )?;
        let active = lease.require_state_mut()?;
        if active.variant != variant || !Arc::ptr_eq(&active.model.model_arc(), &model.model_arc())
        {
            return Err(Error::InferenceError(
                "Audio8 codec crossed its model fence".into(),
            ));
        }
        if active.state.frames_generated() == 0
            || !active.state.finished()
            || active.codec.decoded_frames() != active.state.frames_generated()
        {
            return Err(Error::InferenceError(
                "Audio8 finalization has undecoded frames".into(),
            ));
        }
        active.state.require_complete()?;
        check_audio8_codec_request(request)?;
        let _tail = active.codec.flush();
        let sample_rate = model.diagnostics().sample_rate;
        let timing = audio8_execution_timing(active, 0.0, 0.0, active.first_audio_ms);
        active.state.trace_timings(&request.id, active.codec_ms);
        if let Some(tx) = Self::stream_sender(request) {
            Self::stream_audio_with_policy(
                &tx,
                request.stream_policy,
                &request.id,
                &mut active.stream_sequence,
                Vec::new(),
                sample_rate,
                true,
            )?;
        }
        let audio = if active.collect_audio_samples {
            AudioOutput::new(std::mem::take(&mut active.audio_samples), sample_rate)
        } else {
            AudioOutput::streamed(active.total_audio_samples, sample_rate)
        };
        lease.release()?;
        Ok(ModelSessionResult::sequence(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(audio),
            text: None,
            input_transcription: None,
            tokens_processed: 0,
            tokens_generated: 0,
            finished: true,
            phase_timing_override: Some(timing),
            asr_diagnostics: None,
            error: None,
            logprobs: Vec::new(),
        }))
    }
}

/// Terminal conditions belong to one row, even when its device work shared a
/// launch with peers. Callers discard that row's uncommitted candidate state.
fn audio8_row_terminal(
    request: &EngineCoreRequest,
    dispatched: bool,
) -> Option<ModelSessionResult> {
    if request.is_cancelled() {
        let output = ExecutorOutput::cancelled(request.id.clone());
        return Some(if dispatched {
            ModelSessionResult::cancelled(output)
        } else {
            ModelSessionResult::cancelled_before_dispatch(output)
        });
    }
    if request
        .deadline
        .is_none_or(|deadline| Instant::now() < deadline)
    {
        return None;
    }
    let mut result = ModelSessionResult::sequence(ExecutorOutput::error(
        request.id.clone(),
        Error::Timeout(request.id.clone()).to_string(),
    ));
    result.disposition =
        crate::engine::ExecutionDisposition::Finished(crate::engine::FinishReason::TimedOut);
    result.provenance = crate::engine::OutcomeProvenance::deadline(
        if dispatched {
            crate::engine::DeadlinePhase::ModelExecution
        } else {
            crate::engine::DeadlinePhase::DispatchWait
        },
        if dispatched {
            crate::engine::DispatchState::Started
        } else {
            crate::engine::DispatchState::NotStarted
        },
    );
    Some(result)
}

/// One row's codec progress after a decode step.
#[derive(Debug, Clone, Copy)]
struct Audio8CodecProgress {
    decoded: usize,
    generated: usize,
    finished: bool,
    /// Whether the request streams its audio.
    streamed: bool,
}

impl Audio8CodecProgress {
    /// Whether this row calls its step's rows to the codec: it finished
    /// with frames pending, its first audio is due, or it has a chunk
    /// (streamed) or [`AUDIO8_BUFFERED_AUDIO_FRAMES`] pending.
    ///
    /// [`AUDIO8_BUFFERED_AUDIO_FRAMES`]: crate::models::architectures::audio8::AUDIO8_BUFFERED_AUDIO_FRAMES
    fn calls_codec(self) -> bool {
        use crate::models::architectures::audio8::{
            AUDIO8_AUDIO_CHUNK_FRAMES, AUDIO8_BUFFERED_AUDIO_FRAMES, AUDIO8_FIRST_AUDIO_FRAMES,
        };
        let pending = self.generated.saturating_sub(self.decoded);
        (self.finished && pending > 0)
            || (self.decoded == 0 && pending >= AUDIO8_FIRST_AUDIO_FRAMES)
            || pending
                >= if self.streamed {
                    AUDIO8_AUDIO_CHUNK_FRAMES
                } else {
                    AUDIO8_BUFFERED_AUDIO_FRAMES
                }
    }

    /// The frames this row decodes in its next audio quantum: its whole
    /// pending chunks, at most one quantum's, or a finished row's tail.
    fn due_frames(self) -> usize {
        crate::models::architectures::audio8::audio8_codec_quantum_end(
            self.decoded,
            self.generated,
            self.finished,
            crate::models::architectures::audio8::AUDIO8_CODEC_QUANTUM_FRAMES,
        )
        .saturating_sub(self.decoded)
    }
}

/// Each row's frames due after a decode step. When any row calls the codec
/// every row with a whole chunk pending yields it, so one audio quantum
/// gathers the step's rows instead of each row's chunk running alone.
fn audio8_due_frames(rows: &[Audio8CodecProgress]) -> Vec<usize> {
    let called = rows.iter().any(|row| row.calls_codec());
    rows.iter()
        .map(|row| if called { row.due_frames() } else { 0 })
        .collect()
}

fn check_audio8_codec_request(request: &EngineCoreRequest) -> Result<()> {
    if request.is_cancelled() {
        return Err(Error::Cancelled(request.id.clone()));
    }
    if request
        .deadline
        .is_some_and(|deadline| Instant::now() >= deadline)
    {
        return Err(Error::Timeout(request.id.clone()));
    }
    Ok(())
}

fn audio8_execution_timing(
    active: &ActiveAudio8TtsDecode,
    codec_ms: f64,
    sampling_ms: f64,
    first_audio_ms: Option<f64>,
) -> super::ExecutorPhaseTiming {
    let (prefill_ms, decode_ms) = active.state.phase_timings();
    let (_, prefill_steps, decode_steps) = active.state.sampling_and_steps();
    super::ExecutorPhaseTiming {
        prefill_ms: Some(prefill_ms),
        decode_ms: Some(decode_ms),
        sampling_ms: Some(sampling_ms),
        codec_ms: Some(codec_ms),
        prefill_steps: Some(prefill_steps),
        decode_steps: Some(decode_steps),
        first_output_ms_since_start: first_audio_ms,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::{audio8_due_frames, Audio8CodecProgress};

    fn row(decoded: usize, generated: usize, finished: bool, streamed: bool) -> Audio8CodecProgress {
        Audio8CodecProgress {
            decoded,
            generated,
            finished,
            streamed,
        }
    }

    /// A row's first audio, a streamed row's chunk, a buffered row's 48
    /// frames or a finished row's tail call the codec; then every row with a
    /// whole chunk pending yields its whole chunks (one quantum's at most),
    /// and rows short of a chunk keep decoding.
    #[test]
    fn audio8_codec_gathers_every_row_with_a_whole_chunk() {
        // Nothing due: a buffered row's 47 frames and a streamed row's 15.
        assert_eq!(
            audio8_due_frames(&[row(4, 51, false, false), row(20, 35, false, true)]),
            [0, 0]
        );
        // A new row's first four frames call the codec for everyone.
        assert_eq!(
            audio8_due_frames(&[
                row(0, 4, false, true),
                row(4, 51, false, false),
                row(20, 35, false, true),
                row(0, 3, false, false),
            ]),
            [4, 32, 0, 0]
        );
        // A streamed row's chunk calls; a buffered row with 100 pending
        // decodes one quantum's whole chunks (4 + 3 * 16 from frame 0).
        assert_eq!(
            audio8_due_frames(&[row(20, 36, false, true), row(0, 100, false, false)]),
            [16, 52]
        );
        assert_eq!(audio8_due_frames(&[row(52, 100, false, false)]), [48]);
        // A buffered row calls at 48 pending.
        assert_eq!(audio8_due_frames(&[row(20, 68, false, false)]), [48]);
        // A finished row's tail calls and is decoded whole; past one
        // quantum it decodes whole chunks first.
        assert_eq!(
            audio8_due_frames(&[row(20, 27, true, false), row(4, 30, false, true)]),
            [7, 16]
        );
        assert_eq!(audio8_due_frames(&[row(20, 100, true, false)]), [64]);
        assert_eq!(audio8_due_frames(&[row(84, 100, true, false)]), [16]);
        // Finished with nothing pending: nothing due.
        assert_eq!(audio8_due_frames(&[row(27, 27, true, false)]), [0]);
    }

    #[test]
    fn audio8_fresh_terminal_row_discards_state_without_poisoning_lease() {
        use crate::engine::executor::PendingQuantumFinalizer;
        let (coordinator, session) = super::super::audio8_pending::tests::staged();
        coordinator.discard(17, &session);
        let mut lease = super::ExecutorStateLease::checkout(
            &coordinator.states,
            session.clone(),
            crate::model::ModelVariant::Audio8TtsRealisticAir,
            "fresh rollback test",
        )
        .unwrap();
        lease.mark_dirty();
        let mut row = super::ContinuousAudio8Row {
            index: 0,
            lease: Some(lease),
            checkpoint: None,
            fast: None,
            sampling_before: 0.0,
            fresh: true,
        };
        row.rollback().unwrap();
        assert!(row.lease.as_ref().unwrap().state().is_none());
        drop(row);
        assert!(!coordinator.states.lock().unwrap().contains_key(&session));
    }

    #[test]
    fn audio8_deadline_results_are_row_local_and_preserve_dispatch_provenance() {
        use crate::engine::{DeadlinePhase, DispatchState, ExecutionDisposition, FinishReason};
        let mut expired = super::EngineCoreRequest::tts("expired");
        expired.deadline = Some(std::time::Instant::now());
        let healthy = super::EngineCoreRequest::tts("healthy");
        for dispatched in [false, true] {
            let results =
                [&expired, &healthy].map(|request| super::audio8_row_terminal(request, dispatched));
            assert!(results[1].is_none(), "peer remains executable");
            let timed_out = results[0].as_ref().unwrap();
            assert_eq!(timed_out.output.request_id, expired.id);
            assert_eq!(
                timed_out.disposition,
                ExecutionDisposition::Finished(FinishReason::TimedOut)
            );
            assert_eq!(
                timed_out.provenance.deadline_phase,
                Some(if dispatched {
                    DeadlinePhase::ModelExecution
                } else {
                    DeadlinePhase::DispatchWait
                })
            );
            assert_eq!(
                timed_out.provenance.dispatch_state,
                if dispatched {
                    DispatchState::Started
                } else {
                    DispatchState::NotStarted
                }
            );
            assert!(timed_out.output.finished);
            assert_eq!(timed_out.output.tokens_processed, 0);
            assert_eq!(timed_out.output.tokens_generated, 0);
            assert!(timed_out.staged_stream_outputs.is_empty());
            assert!(timed_out.managed_cache_completions.is_empty());
            assert!(!timed_out.pending_quantum_required);
        }
    }

    #[test]
    fn audio8_batch_guard_discards_prior_staging_when_a_later_row_fails() {
        let (coordinator, session) = super::super::audio8_pending::tests::staged();
        let failure: crate::error::Result<()> = {
            let _guard = super::Audio8PendingBatchGuard {
                coordinator: &coordinator,
                staged: vec![(17, session.clone())],
                armed: true,
            };
            Err(crate::error::Error::InferenceError(
                "later row rejected after first staged".into(),
            ))
        };
        assert!(failure.is_err());
        super::super::audio8_pending::tests::assert_ready(&coordinator, &session, false);
        assert!(!super::super::PendingQuantumFinalizer::contains(
            &coordinator,
            17,
            &session
        ));
    }
}
