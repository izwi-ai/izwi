//! Chatterbox TTS execution: batched T3 prefill and decode over each
//! request's two retained paged caches, then the acoustic stage (S3Gen
//! encoder, each Euler step, HiFT) one `SequenceAudioDecode` quantum per
//! unit, so other requests' work runs between them, and a `SequenceFinalize`
//! quantum that hands over the waveform. A streamed request renders windows
//! the same way while T3 runs, each window's samples staged on its stream.

use std::sync::Arc;

use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use crate::models::architectures::chatterbox::retained::{
    ChatterboxQuantumCheckpoint, ChatterboxStepRow,
};
use crate::models::architectures::chatterbox::{
    ChatterboxAcousticProgress, CHATTERBOX_COND_DOMAIN, CHATTERBOX_UNCOND_DOMAIN, S3GEN_SAMPLE_RATE,
};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

use super::super::request::EngineCoreRequest;
use super::super::scheduler::ScheduledRequest;
use super::super::types::AudioOutput;
use super::state::ActiveChatterboxTtsDecode;
use super::{ExecutorOutput, ExecutorStateLease, ModelSessionResult, NativeExecutor};

struct ChatterboxRow<'a> {
    index: usize,
    lease: Option<ExecutorStateLease<'a, ActiveChatterboxTtsDecode>>,
    conditional: PhysicalPagedKvCache,
    unconditional: Option<PhysicalPagedKvCache>,
    checkpoint: Option<ChatterboxQuantumCheckpoint>,
}

impl ChatterboxRow<'_> {
    fn rollback(&mut self) -> Result<()> {
        let checkpoint = self.checkpoint.take().ok_or_else(|| {
            Error::InferenceError("Chatterbox row has no rollback checkpoint".into())
        })?;
        let lease = self.lease.as_mut().expect("armed Chatterbox lease");
        lease.require_state_mut()?.state.rollback_quantum(
            &mut self.conditional,
            self.unconditional.as_mut(),
            &checkpoint,
        )?;
        lease.mark_clean();
        Ok(())
    }
}

/// Rolls back every armed row if the quantum does not reach its commit.
struct ChatterboxBatch<'a> {
    rows: Vec<ChatterboxRow<'a>>,
    armed: bool,
}

impl Drop for ChatterboxBatch<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        for row in &mut self.rows {
            if row.checkpoint.is_some() {
                if let Err(error) = row.rollback() {
                    tracing::error!(%error, "Chatterbox rollback failed; state remains fenced");
                }
            }
        }
    }
}

impl NativeExecutor {
    pub(super) fn chatterbox_tts_request_with_managed_cache(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        retained: Option<super::RetainedRowManagedState>,
    ) -> Result<ModelSessionResult> {
        let mut outputs = self.chatterbox_tts_batch_with_managed(
            &[request],
            std::slice::from_ref(scheduled),
            vec![retained],
        )?;
        outputs
            .pop()
            .ok_or_else(|| Error::InferenceError("Chatterbox row lost its output".into()))
    }

    /// One T3 quantum for every row: a prefill span each, or one decode
    /// step each, in a single batched forward.
    pub(super) fn chatterbox_tts_batch_with_managed(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        mut managed: Vec<Option<super::RetainedRowManagedState>>,
    ) -> Result<Vec<ModelSessionResult>> {
        if requests.is_empty()
            || requests.len() != scheduled.len()
            || managed.len() != scheduled.len()
        {
            return Err(Error::InvalidInput(
                "Chatterbox batch rows do not match".into(),
            ));
        }
        let mut outputs = (0..scheduled.len()).map(|_| None).collect::<Vec<_>>();
        let live = requests
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
        if live.is_empty() {
            return collect_outputs(outputs);
        }
        let is_prefill = scheduled[live[0]].is_prefill;
        if live
            .iter()
            .any(|&index| scheduled[index].is_prefill != is_prefill)
        {
            return Err(Error::InvalidInput("Chatterbox batch mixed phases".into()));
        }
        let model = requests[live[0]]
            .prepared_chatterbox_tts_model_lease_for_executor()?
            .ok_or_else(|| Error::InferenceError("Chatterbox batch lost model".into()))?;
        let model_arc = model.model_arc();

        let mut rows = Vec::with_capacity(live.len());
        for &index in &live {
            let request = requests[index];
            let variant = Self::resolve_variant(request)?;
            let row_model = request
                .prepared_chatterbox_tts_model_lease_for_executor()?
                .ok_or_else(|| Error::InferenceError("Chatterbox row lost model".into()))?;
            if variant.family() != ModelFamily::ChatterboxTts
                || !Arc::ptr_eq(&model_arc, &row_model.model_arc())
            {
                return Err(Error::InvalidInput(
                    "Chatterbox batch crossed model identity".into(),
                ));
            }
            let mut retained = managed[index].take().ok_or_else(|| {
                Error::InferenceError("Chatterbox row lost its retained state".into())
            })?;
            let conditional = retained
                .take_paged_domain(CHATTERBOX_COND_DOMAIN, true)?
                .expect("required Chatterbox conditional cache");
            // Unreserved for an unguided request (`project_paged_state_input`).
            let unconditional = retained.take_paged_domain(CHATTERBOX_UNCOND_DOMAIN, false)?;
            retained.ensure_all_paged_consumed()?;
            let mut lease = ExecutorStateLease::checkout(
                &self.chatterbox_tts_decode_states,
                scheduled[index].session_key(),
                variant,
                "Chatterbox TTS",
            )?;
            if lease.state().is_some_and(|active| {
                active.variant != variant || !Arc::ptr_eq(&active.model.model_arc(), &model_arc)
            }) {
                lease.discard_state();
            }
            if lease.state().is_none() {
                if !is_prefill || scheduled[index].num_computed_tokens != 0 {
                    return Err(Error::InferenceError(
                        "Chatterbox row lost its initial state".into(),
                    ));
                }
                let artifact = request
                    .prepared_chatterbox_tts_artifact_for_executor()?
                    .ok_or_else(|| {
                        Error::InferenceError("Chatterbox row lost its prompt".into())
                    })?;
                let params = request
                    .chatterbox_tts_generation_params_for_executor()?
                    .ok_or_else(|| {
                        Error::InferenceError("Chatterbox row lost its parameters".into())
                    })?;
                lease.install_state(ActiveChatterboxTtsDecode {
                    variant,
                    model: model.clone(),
                    state: model.new_retained_state(artifact, params, request.streaming)?,
                    stream_sequence: 0,
                    streamed_samples: 0,
                })?;
            }
            let active = lease.require_state_mut()?;
            if active.state.guided() != unconditional.is_some() {
                return Err(Error::InferenceError(
                    "Chatterbox row's unconditional cache does not match its guidance".into(),
                ));
            }
            if is_prefill {
                let (start, _) = super::handler_tts::resumable_tts_prefill_span(
                    &scheduled[index],
                    active.state.prompt_tokens(),
                )?;
                if active.state.prefill_cursor() != start {
                    return Err(Error::InferenceError(
                        "Chatterbox scheduler and retained prefill cursors diverged".into(),
                    ));
                }
            }
            let checkpoint = active
                .state
                .begin_quantum(&conditional, unconditional.as_ref())?;
            lease.mark_dirty();
            rows.push(ChatterboxRow {
                index,
                lease: Some(lease),
                conditional,
                unconditional,
                checkpoint: Some(checkpoint),
            });
        }
        let mut batch = ChatterboxBatch { rows, armed: true };
        for row in &mut batch.rows {
            if requests[row.index].is_cancelled() {
                row.rollback()?;
                outputs[row.index] = Some(ModelSessionResult::cancelled(
                    ExecutorOutput::cancelled(requests[row.index].id.clone()),
                ));
            }
        }

        let call_rows = batch
            .rows
            .iter()
            .enumerate()
            .filter_map(|(position, row)| row.checkpoint.is_some().then_some(position))
            .collect::<Vec<_>>();
        if !call_rows.is_empty() {
            let mut step_rows = Vec::with_capacity(call_rows.len());
            let mut spans = Vec::with_capacity(call_rows.len());
            for (position, row) in batch.rows.iter_mut().enumerate() {
                if !call_rows.contains(&position) {
                    continue;
                }
                spans.push(scheduled[row.index].num_tokens);
                step_rows.push(ChatterboxStepRow {
                    state: &mut row
                        .lease
                        .as_mut()
                        .expect("armed Chatterbox lease")
                        .require_state_mut()?
                        .state,
                    conditional: &mut row.conditional,
                    unconditional: row.unconditional.as_mut(),
                });
            }
            let finished = if is_prefill {
                let steps =
                    Self::run_blocking(|| model.retained_prefill_batch(&mut step_rows, &spans))?;
                for (step, &span) in steps.iter().zip(&spans) {
                    if step.consumed_tokens != span {
                        return Err(Error::InferenceError(
                            "Chatterbox prefill progress differs from the scheduler".into(),
                        ));
                    }
                }
                vec![false; steps.len()]
            } else {
                Self::run_blocking(|| model.retained_decode_batch(&mut step_rows))?
                    .into_iter()
                    .map(|step| step.finished)
                    .collect::<Vec<_>>()
            };
            let width = step_rows.len();
            drop(step_rows);
            let mode = if is_prefill {
                crate::engine::NativeBatchMode::Static
            } else {
                crate::engine::NativeBatchMode::Continuous
            };
            if let Some(call) = super::handler_tts::retained_tts_batch_model_call(mode, width) {
                crate::engine::metrics::record_engine_model_call(call);
            }

            // Cancellation is checked again before the all-row commit.
            for row in &mut batch.rows {
                if row.checkpoint.is_some() && requests[row.index].is_cancelled() {
                    row.rollback()?;
                    outputs[row.index] = Some(ModelSessionResult::cancelled(
                        ExecutorOutput::cancelled(requests[row.index].id.clone()),
                    ));
                }
            }
            // Validate every commit before consuming any checkpoint.
            for row in &mut batch.rows {
                let Some(checkpoint) = row.checkpoint.as_ref() else {
                    continue;
                };
                row.lease
                    .as_mut()
                    .expect("armed Chatterbox lease")
                    .require_state_mut()?
                    .state
                    .commit_quantum(&row.conditional, row.unconditional.as_ref(), checkpoint)?;
            }
            for (position, done) in call_rows.into_iter().zip(finished) {
                let row = &mut batch.rows[position];
                if row.checkpoint.is_none() {
                    continue;
                }
                let index = row.index;
                let mut completions = row.conditional.take_completed_writes();
                if let Some(unconditional) = row.unconditional.as_mut() {
                    completions.extend(unconditional.take_completed_writes());
                }
                row.checkpoint = None;
                let mut lease = row.lease.take().expect("armed Chatterbox lease");
                let window_due =
                    !done && model.retained_window_due(&lease.require_state_mut()?.state);
                lease.mark_clean();
                lease.restore()?;
                // A row that sampled its stop token this step does not run
                // it through T3: its caches stay where they were.
                let output = ExecutorOutput {
                    request_id: requests[index].id.clone(),
                    audio: Some(AudioOutput::new(Vec::new(), S3GEN_SAMPLE_RATE)),
                    text: None,
                    input_transcription: None,
                    tokens_processed: if done { 0 } else { scheduled[index].num_tokens },
                    tokens_generated: usize::from(!is_prefill),
                    finished: false,
                    phase_timing_override: None,
                    asr_diagnostics: None,
                    error: None,
                    logprobs: Vec::new(),
                };
                let result = if done {
                    ModelSessionResult::yielded(
                        output,
                        crate::engine::YieldReason::AwaitingAudioDecode { max_frames: 1 },
                    )
                    .with_managed_cache_append(0)
                } else if window_due {
                    // A streamed window's tokens are in: render it before
                    // the next decode step.
                    ModelSessionResult::yielded(
                        output,
                        crate::engine::YieldReason::AwaitingAudioDecode { max_frames: 1 },
                    )
                } else {
                    ModelSessionResult::sequence(output)
                };
                outputs[index] = Some(result.with_managed_cache_completions(completions));
            }
        }
        batch.armed = false;
        collect_outputs(outputs)
    }

    /// One acoustic unit: the S3Gen encoder, an Euler step, or HiFT, of a
    /// final-only request whose T3 stopped or of a streamed request's
    /// window. Yields for the next unit until the waveform or window is
    /// done; a finished window's samples go on the stream, then T3 resumes,
    /// another window runs, or the request finalizes.
    pub(super) fn chatterbox_tts_acoustic_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
    ) -> Result<ModelSessionResult> {
        if !matches!(
            scheduled.work,
            crate::engine::WorkUnit::SequenceAudioDecode { max_frames: 1 }
        ) || scheduled.is_prefill
        {
            return Err(Error::InvalidInput(
                "Chatterbox acoustic stage requires one audio-decode unit".into(),
            ));
        }
        let (model, mut lease) = self.checkout_chatterbox_acoustic(request, scheduled)?;
        lease.mark_dirty();
        let started = std::time::Instant::now();
        let progress = Self::run_blocking(|| {
            model.retained_acoustic_step(&mut lease.require_state_mut()?.state)
        });
        // A failed unit leaves the lease dirty, so its partial state is
        // discarded with the request.
        let progress = progress?;
        tracing::debug!(
            request_id = %request.id,
            elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
            window = matches!(progress, ChatterboxAcousticProgress::Window(_)),
            rendered = matches!(progress, ChatterboxAcousticProgress::Rendered),
            "Chatterbox acoustic unit"
        );
        if request.is_cancelled() {
            return Err(Error::Cancelled(request.id.clone()));
        }
        let reason = match progress {
            ChatterboxAcousticProgress::Working => {
                crate::engine::YieldReason::AwaitingAudioDecode { max_frames: 1 }
            }
            ChatterboxAcousticProgress::Rendered => {
                crate::engine::YieldReason::AwaitingFinalization
            }
            ChatterboxAcousticProgress::Window(samples) => {
                let active = lease.require_state_mut()?;
                active.streamed_samples += samples.len();
                if let Some(tx) = Self::stream_sender(request) {
                    Self::stream_audio_with_policy(
                        &tx,
                        request.stream_policy,
                        &request.id,
                        &mut active.stream_sequence,
                        samples,
                        S3GEN_SAMPLE_RATE,
                        false,
                    )?;
                }
                if active
                    .state
                    .stream
                    .as_ref()
                    .is_some_and(|stream| stream.done())
                {
                    crate::engine::YieldReason::AwaitingFinalization
                } else if model.retained_window_due(&active.state) {
                    crate::engine::YieldReason::AwaitingAudioDecode { max_frames: 1 }
                } else {
                    // Back to T3 for the next window's tokens.
                    crate::engine::YieldReason::QuantumExhausted
                }
            }
        };
        lease.mark_clean();
        lease.restore()?;
        Ok(ModelSessionResult::yielded(
            ExecutorOutput {
                request_id: request.id.clone(),
                audio: Some(AudioOutput::empty(S3GEN_SAMPLE_RATE)),
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
            reason,
        ))
    }

    /// Hand over the rendered waveform: the whole utterance in one output,
    /// or, for a stream whose windows sent it all, the end-of-stream marker.
    pub(super) fn chatterbox_tts_finalize_request(
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
                "Chatterbox finalization requires an exact sequence-finalize quantum".into(),
            ));
        }
        if request.is_cancelled() {
            return Err(Error::Cancelled(request.id.clone()));
        }
        let (model, mut lease) = self.checkout_chatterbox_acoustic(request, scheduled)?;
        let active = lease.require_state_mut()?;
        let samples = model.retained_acoustic_finalize(&mut active.state)?;
        let audio = if active.state.stream.is_some() {
            if let Some(tx) = Self::stream_sender(request) {
                Self::stream_audio_with_policy(
                    &tx,
                    request.stream_policy,
                    &request.id,
                    &mut active.stream_sequence,
                    Vec::new(),
                    S3GEN_SAMPLE_RATE,
                    true,
                )?;
            }
            AudioOutput::streamed(active.streamed_samples, S3GEN_SAMPLE_RATE)
        } else {
            AudioOutput::new(samples, S3GEN_SAMPLE_RATE)
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
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
            logprobs: Vec::new(),
        }))
    }

    /// The model and the request's retained state for an acoustic quantum,
    /// fenced to the model that generated its tokens.
    fn checkout_chatterbox_acoustic<'a>(
        &'a self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
    ) -> Result<(
        crate::models::registry::ChatterboxTtsModelLease,
        ExecutorStateLease<'a, ActiveChatterboxTtsDecode>,
    )> {
        let variant = Self::resolve_variant(request)?;
        if variant.family() != ModelFamily::ChatterboxTts {
            return Err(Error::InvalidInput(
                "foreign Chatterbox acoustic request".into(),
            ));
        }
        let model = request
            .prepared_chatterbox_tts_model_lease_for_executor()?
            .ok_or_else(|| Error::InferenceError("Chatterbox acoustic lost model".into()))?;
        let mut lease = ExecutorStateLease::checkout(
            &self.chatterbox_tts_decode_states,
            scheduled.session_key(),
            variant,
            "Chatterbox acoustic",
        )?;
        let active = lease.require_state_mut()?;
        if active.variant != variant || !Arc::ptr_eq(&active.model.model_arc(), &model.model_arc())
        {
            return Err(Error::InferenceError(
                "Chatterbox acoustic state crossed its model fence".into(),
            ));
        }
        Ok((model, lease))
    }
}

fn collect_outputs(outputs: Vec<Option<ModelSessionResult>>) -> Result<Vec<ModelSessionResult>> {
    outputs
        .into_iter()
        .map(|output| {
            output.ok_or_else(|| Error::InferenceError("Chatterbox row lost its output".into()))
        })
        .collect()
}
