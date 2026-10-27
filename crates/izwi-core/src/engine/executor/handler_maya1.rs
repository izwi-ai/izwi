//! Maya1 TTS execution: batched decoder prefill and decode over each
//! request's retained paged cache, an audio-window quantum
//! (`SequenceAudioDecode`) whenever a SNAC window is due, and a
//! `SequenceFinalize` quantum that hands over the audio. Streamed and
//! final-only requests decode the same windows; a stream sends each one as
//! it settles, a final-only request keeps them until it finishes.

use std::sync::Arc;

use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use crate::models::architectures::maya1::retained::{Maya1QuantumCheckpoint, Maya1StepRow};
use crate::models::architectures::maya1::{Maya1AudioProgress, MAYA1_DOMAIN, SAMPLE_RATE};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

use super::super::request::EngineCoreRequest;
use super::super::scheduler::ScheduledRequest;
use super::super::types::AudioOutput;
use super::state::ActiveMaya1TtsDecode;
use super::{ExecutorOutput, ExecutorStateLease, ModelSessionResult, NativeExecutor};

struct Maya1Row<'a> {
    index: usize,
    lease: Option<ExecutorStateLease<'a, ActiveMaya1TtsDecode>>,
    cache: PhysicalPagedKvCache,
    checkpoint: Option<Maya1QuantumCheckpoint>,
}

impl Maya1Row<'_> {
    fn rollback(&mut self) -> Result<()> {
        let checkpoint = self
            .checkpoint
            .take()
            .ok_or_else(|| Error::InferenceError("Maya1 row has no rollback checkpoint".into()))?;
        let lease = self.lease.as_mut().expect("armed Maya1 lease");
        lease
            .require_state_mut()?
            .state
            .rollback_quantum(&mut self.cache, &checkpoint)?;
        lease.mark_clean();
        Ok(())
    }
}

/// Rolls back every armed row if the quantum does not reach its commit.
struct Maya1Batch<'a> {
    rows: Vec<Maya1Row<'a>>,
    armed: bool,
}

impl Drop for Maya1Batch<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        for row in &mut self.rows {
            if row.checkpoint.is_some() {
                if let Err(error) = row.rollback() {
                    tracing::error!(%error, "Maya1 rollback failed; state remains fenced");
                }
            }
        }
    }
}

fn empty_output(request: &EngineCoreRequest) -> ExecutorOutput {
    ExecutorOutput {
        request_id: request.id.clone(),
        audio: Some(AudioOutput::empty(SAMPLE_RATE)),
        text: None,
        input_transcription: None,
        tokens_processed: 0,
        tokens_generated: 0,
        finished: false,
        phase_timing_override: None,
        asr_diagnostics: None,
        error: None,
        logprobs: Vec::new(),
    }
}

impl NativeExecutor {
    pub(super) fn maya1_tts_request_with_managed_cache(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        retained: Option<super::RetainedRowManagedState>,
    ) -> Result<ModelSessionResult> {
        let mut outputs = self.maya1_tts_batch_with_managed(
            &[request],
            std::slice::from_ref(scheduled),
            vec![retained],
        )?;
        outputs
            .pop()
            .ok_or_else(|| Error::InferenceError("Maya1 row lost its output".into()))
    }

    /// One decoder quantum for every row: a prefill span each, or one decode
    /// step each, in a single batched forward.
    pub(super) fn maya1_tts_batch_with_managed(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        mut managed: Vec<Option<super::RetainedRowManagedState>>,
    ) -> Result<Vec<ModelSessionResult>> {
        if requests.is_empty()
            || requests.len() != scheduled.len()
            || managed.len() != scheduled.len()
        {
            return Err(Error::InvalidInput("Maya1 batch rows do not match".into()));
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
            return Err(Error::InvalidInput("Maya1 batch mixed phases".into()));
        }
        let model = requests[live[0]]
            .prepared_maya1_tts_model_lease_for_executor()?
            .ok_or_else(|| Error::InferenceError("Maya1 batch lost model".into()))?;
        let model_arc = model.model_arc();

        let mut rows = Vec::with_capacity(live.len());
        for &index in &live {
            let request = requests[index];
            let variant = Self::resolve_variant(request)?;
            let row_model = request
                .prepared_maya1_tts_model_lease_for_executor()?
                .ok_or_else(|| Error::InferenceError("Maya1 row lost model".into()))?;
            if variant.family() != ModelFamily::Maya1Tts
                || !Arc::ptr_eq(&model_arc, &row_model.model_arc())
            {
                return Err(Error::InvalidInput(
                    "Maya1 batch crossed model identity".into(),
                ));
            }
            let mut retained = managed[index]
                .take()
                .ok_or_else(|| Error::InferenceError("Maya1 row lost its retained state".into()))?;
            let cache = retained
                .take_paged_domain(MAYA1_DOMAIN, true)?
                .expect("required Maya1 cache");
            retained.ensure_all_paged_consumed()?;
            let mut lease = ExecutorStateLease::checkout(
                &self.maya1_tts_decode_states,
                scheduled[index].session_key(),
                variant,
                "Maya1 TTS",
            )?;
            if lease.state().is_some_and(|active| {
                active.variant != variant || !Arc::ptr_eq(&active.model.model_arc(), &model_arc)
            }) {
                lease.discard_state();
            }
            if lease.state().is_none() {
                if !is_prefill || scheduled[index].num_computed_tokens != 0 {
                    return Err(Error::InferenceError(
                        "Maya1 row lost its initial state".into(),
                    ));
                }
                let artifact = request
                    .prepared_maya1_tts_artifact_for_executor()?
                    .ok_or_else(|| Error::InferenceError("Maya1 row lost its prompt".into()))?;
                let params = request
                    .maya1_tts_generation_params_for_executor()?
                    .ok_or_else(|| Error::InferenceError("Maya1 row lost its parameters".into()))?;
                lease.install_state(ActiveMaya1TtsDecode {
                    variant,
                    model: model.clone(),
                    state: model.new_retained_state(artifact, params)?,
                    stream_sequence: 0,
                    rendered: Vec::new(),
                })?;
            }
            let active = lease.require_state_mut()?;
            if is_prefill {
                let (start, _) = super::handler_tts::resumable_tts_prefill_span(
                    &scheduled[index],
                    active.state.prompt_tokens(),
                )?;
                if active.state.prefill_cursor() != start {
                    return Err(Error::InferenceError(
                        "Maya1 scheduler and retained prefill cursors diverged".into(),
                    ));
                }
            }
            let checkpoint = active.state.begin_quantum(&cache)?;
            lease.mark_dirty();
            rows.push(Maya1Row {
                index,
                lease: Some(lease),
                cache,
                checkpoint: Some(checkpoint),
            });
        }
        let mut batch = Maya1Batch { rows, armed: true };
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
                step_rows.push(Maya1StepRow {
                    state: &mut row
                        .lease
                        .as_mut()
                        .expect("armed Maya1 lease")
                        .require_state_mut()?
                        .state,
                    cache: &mut row.cache,
                });
            }
            let finished = if is_prefill {
                let steps =
                    Self::run_blocking(|| model.retained_prefill_batch(&mut step_rows, &spans))?;
                for (step, &span) in steps.iter().zip(&spans) {
                    if step.consumed_tokens != span {
                        return Err(Error::InferenceError(
                            "Maya1 prefill progress differs from the scheduler".into(),
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
                    .expect("armed Maya1 lease")
                    .require_state_mut()?
                    .state
                    .commit_quantum(&row.cache, checkpoint)?;
            }
            for (position, done) in call_rows.into_iter().zip(finished) {
                let row = &mut batch.rows[position];
                if row.checkpoint.is_none() {
                    continue;
                }
                let index = row.index;
                let completions = row.cache.take_completed_writes();
                row.checkpoint = None;
                let mut lease = row.lease.take().expect("armed Maya1 lease");
                let window_due = model.retained_window_due(&lease.require_state_mut()?.state);
                lease.mark_clean();
                lease.restore()?;
                // A row that sampled its stop token this step does not run
                // it through the decoder: its cache stays where it was.
                let output = ExecutorOutput {
                    request_id: requests[index].id.clone(),
                    audio: Some(AudioOutput::new(Vec::new(), SAMPLE_RATE)),
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
                    // A window's frames are in: decode it before the next
                    // decoder step.
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

    /// One audio window of a request: decoded with SNAC, then sent on its
    /// stream or kept for the final output. Yields for another window when
    /// one is due, back to the decoder while it runs, or to finalization
    /// once the last window is out.
    pub(super) fn maya1_tts_audio_request(
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
                "Maya1 audio stage requires one audio-decode unit".into(),
            ));
        }
        let (model, mut lease) = self.checkout_maya1_audio(request, scheduled)?;
        lease.mark_dirty();
        let started = std::time::Instant::now();
        let progress =
            Self::run_blocking(|| model.retained_audio_step(&mut lease.require_state_mut()?.state));
        // A failed window leaves the lease dirty, so its partial state is
        // discarded with the request.
        let progress = progress?;
        tracing::debug!(
            request_id = %request.id,
            elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
            "Maya1 audio window"
        );
        if request.is_cancelled() {
            return Err(Error::Cancelled(request.id.clone()));
        }
        let active = lease.require_state_mut()?;
        if let Maya1AudioProgress::Window(samples) = progress {
            if !samples.is_empty() {
                if let Some(tx) = Self::stream_sender(request) {
                    Self::stream_audio_with_policy(
                        &tx,
                        request.stream_policy,
                        &request.id,
                        &mut active.stream_sequence,
                        samples,
                        SAMPLE_RATE,
                        false,
                    )?;
                } else {
                    active.rendered.extend(samples);
                }
            }
        }
        let reason = if active.state.audio.done() {
            crate::engine::YieldReason::AwaitingFinalization
        } else if model.retained_window_due(&active.state) {
            crate::engine::YieldReason::AwaitingAudioDecode { max_frames: 1 }
        } else if active.state.stopped() {
            return Err(Error::InferenceError(
                "Maya1 stopped with no audio window left to run".into(),
            ));
        } else {
            // Back to the decoder for the next window's frames.
            crate::engine::YieldReason::QuantumExhausted
        };
        lease.mark_clean();
        lease.restore()?;
        Ok(ModelSessionResult::yielded(empty_output(request), reason))
    }

    /// Hand over the audio: the whole utterance in one output, or, for a
    /// stream whose windows sent it all, the end-of-stream marker.
    pub(super) fn maya1_tts_finalize_request(
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
                "Maya1 finalization requires an exact sequence-finalize quantum".into(),
            ));
        }
        if request.is_cancelled() {
            return Err(Error::Cancelled(request.id.clone()));
        }
        let (model, mut lease) = self.checkout_maya1_audio(request, scheduled)?;
        let active = lease.require_state_mut()?;
        model.retained_finalize(&mut active.state)?;
        let audio = if let Some(tx) = Self::stream_sender(request) {
            Self::stream_audio_with_policy(
                &tx,
                request.stream_policy,
                &request.id,
                &mut active.stream_sequence,
                Vec::new(),
                SAMPLE_RATE,
                true,
            )?;
            AudioOutput::streamed(active.state.audio.samples(), SAMPLE_RATE)
        } else {
            AudioOutput::new(std::mem::take(&mut active.rendered), SAMPLE_RATE)
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

    /// The model and the request's retained state for an audio quantum,
    /// fenced to the model that generated its tokens.
    fn checkout_maya1_audio<'a>(
        &'a self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
    ) -> Result<(
        crate::models::registry::Maya1TtsModelLease,
        ExecutorStateLease<'a, ActiveMaya1TtsDecode>,
    )> {
        let variant = Self::resolve_variant(request)?;
        if variant.family() != ModelFamily::Maya1Tts {
            return Err(Error::InvalidInput("foreign Maya1 audio request".into()));
        }
        let model = request
            .prepared_maya1_tts_model_lease_for_executor()?
            .ok_or_else(|| Error::InferenceError("Maya1 audio lost model".into()))?;
        let mut lease = ExecutorStateLease::checkout(
            &self.maya1_tts_decode_states,
            scheduled.session_key(),
            variant,
            "Maya1 audio",
        )?;
        let active = lease.require_state_mut()?;
        if active.variant != variant || !Arc::ptr_eq(&active.model.model_arc(), &model.model_arc())
        {
            return Err(Error::InferenceError(
                "Maya1 audio state crossed its model fence".into(),
            ));
        }
        Ok((model, lease))
    }
}

fn collect_outputs(outputs: Vec<Option<ModelSessionResult>>) -> Result<Vec<ModelSessionResult>> {
    outputs
        .into_iter()
        .map(|output| {
            output.ok_or_else(|| Error::InferenceError("Maya1 row lost its output".into()))
        })
        .collect()
}
