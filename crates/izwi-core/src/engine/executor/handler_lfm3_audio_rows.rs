//! LFM3-Audio requests as retained continuous rows: each quantum prefills a
//! row's prompt span, or steps every row one slot with the language model in
//! one batched forward, over each row's retained paged KV and ShortConv ring.
//! A speech row streams its audio as the codec decodes it (a final-only
//! request gets it when its row finishes); a transcription row returns its
//! text so far and streams each quantum's new text; a voice-chat row streams
//! its reply's text and speech in the order they decode.

use std::sync::Arc;

use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use crate::models::architectures::lfm3_audio::retained::{
    Lfm3AudioRetainedCheckpoint, Lfm3AudioRetainedState, Lfm3AudioStepRow,
};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

use super::super::request::EngineCoreRequest;
use super::super::scheduler::ScheduledRequest;
use super::super::types::{AudioOutput, TaskType};
use super::handler_tts::{retained_tts_batch_model_call, stream_committed_tts_audio};
use super::state::ActiveLfm3AudioDecode;
use super::{ExecutorOutput, ExecutorStateLease, ModelSessionResult, NativeExecutor};

const SAMPLE_RATE: u32 = 24_000;

struct Lfm3AudioBatchRow<'a> {
    index: usize,
    lease: Option<ExecutorStateLease<'a, ActiveLfm3AudioDecode>>,
    cache: PhysicalPagedKvCache,
    cache_start: usize,
    checkpoint: Option<Lfm3AudioRetainedCheckpoint>,
}

impl Lfm3AudioBatchRow<'_> {
    fn state_mut(&mut self) -> Result<&mut Lfm3AudioRetainedState> {
        Ok(&mut self
            .lease
            .as_mut()
            .ok_or_else(|| Error::InferenceError("LFM3-Audio row lease is absent".into()))?
            .require_state_mut()?
            .state)
    }

    fn rollback(&mut self) -> Result<()> {
        let checkpoint = self.checkpoint.take().ok_or_else(|| {
            Error::InferenceError("LFM3-Audio row has no rollback checkpoint".into())
        })?;
        let lease = self.lease.as_mut().expect("armed LFM3-Audio lease");
        lease
            .require_state_mut()?
            .state
            .rollback_quantum(&mut self.cache, &checkpoint)?;
        lease.mark_clean();
        Ok(())
    }
}

/// Rolls back every armed row if the quantum does not reach its commit.
struct Lfm3AudioBatch<'a> {
    rows: Vec<Lfm3AudioBatchRow<'a>>,
    armed: bool,
}

impl Drop for Lfm3AudioBatch<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        for row in &mut self.rows {
            if row.checkpoint.is_some() {
                if let Err(error) = row.rollback() {
                    tracing::error!(%error, "LFM3-Audio rollback failed; state remains fenced");
                }
            }
        }
    }
}

impl NativeExecutor {
    pub(super) fn lfm3_audio_request_with_managed_cache(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        retained: Option<super::RetainedRowManagedState>,
    ) -> Result<ModelSessionResult> {
        let mut outputs = self.lfm3_audio_batch_with_managed(
            &[request],
            std::slice::from_ref(scheduled),
            vec![retained],
        )?;
        outputs
            .pop()
            .ok_or_else(|| Error::InferenceError("LFM3-Audio row lost its output".into()))
    }

    /// One quantum for every row: a prompt span each, or one slot each with
    /// the language model batched across rows.
    pub(super) fn lfm3_audio_batch_with_managed(
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
                "LFM3-Audio batch rows do not match".into(),
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
            return Err(Error::InvalidInput("LFM3-Audio batch mixed phases".into()));
        }
        let model = requests[live[0]]
            .prepared_lfm3_audio_model_lease_for_executor()?
            .ok_or_else(|| Error::InferenceError("LFM3-Audio batch lost its model".into()))?;
        let model_arc = model.model_arc();
        let lfm3 = model.lfm3_audio()?;

        let mut rows = Vec::with_capacity(live.len());
        for &index in &live {
            let request = requests[index];
            let variant = Self::resolve_variant(request)?;
            let row_model = request
                .prepared_lfm3_audio_model_lease_for_executor()?
                .ok_or_else(|| Error::InferenceError("LFM3-Audio row lost its model".into()))?;
            if variant.family() != ModelFamily::Lfm3Audio
                || !Arc::ptr_eq(&model_arc, &row_model.model_arc())
            {
                return Err(Error::InvalidInput(
                    "LFM3-Audio batch crossed model identity".into(),
                ));
            }
            let mut retained = managed[index].take().ok_or_else(|| {
                Error::InferenceError("LFM3-Audio row lost its retained state".into())
            })?;
            let tensor = retained.tensor_state.clone().ok_or_else(|| {
                Error::InferenceError("LFM3-Audio row lost its ShortConv reservation".into())
            })?;
            let arena = request
                .managed_cache_runtime()
                .and_then(|runtime| runtime.tensor_state())
                .ok_or_else(|| {
                    Error::InferenceError("LFM3-Audio row lost its ShortConv arena".into())
                })?;
            let cache = retained.take_only_paged()?;
            let cache_start = cache.context_len();
            retained.ensure_all_paged_consumed()?;
            let mut lease = ExecutorStateLease::checkout(
                &self.lfm3_audio_decode_states,
                scheduled[index].session_key(),
                variant,
                "LFM3-Audio row",
            )?;
            if lease.state().is_some_and(|active| {
                active.variant != variant || !Arc::ptr_eq(&active.model.model_arc(), &model_arc)
            }) {
                lease.discard_state();
            }
            if lease.state().is_none() {
                if !is_prefill || scheduled[index].num_computed_tokens != 0 {
                    return Err(Error::InferenceError(
                        "LFM3-Audio row lost its initial state".into(),
                    ));
                }
                let artifact = request
                    .prepared_lfm3_audio_artifact_for_executor()?
                    .ok_or_else(|| {
                        Error::InferenceError("LFM3-Audio row lost its prompt".into())
                    })?;
                let input_audio = match request.task_type {
                    TaskType::ASR => {
                        let (samples, rate) =
                            request.prepared_asr_audio_for_executor()?.ok_or_else(|| {
                                Error::InferenceError("LFM3-Audio row lost its audio".into())
                            })?;
                        Some((rate, samples.len()))
                    }
                    _ => None,
                };
                // The row holds its prompt and every slot it may decode.
                let context = artifact.prompt_tokens + request.params.max_tokens;
                lease.install_state(ActiveLfm3AudioDecode {
                    variant,
                    model: model.clone(),
                    state: Lfm3AudioRetainedState::new(lfm3, artifact, context)?,
                    stream_sequence: 0,
                    streamed_samples: 0,
                    rendered: Vec::new(),
                    input_audio,
                })?;
            }
            let active = lease.require_state_mut()?;
            active.state.bind_tensor_sequence(tensor.sequence)?;
            active.state.restore_shortconv(arena)?;
            if is_prefill && active.state.prefill_cursor() != scheduled[index].num_computed_tokens {
                return Err(Error::InferenceError(
                    "LFM3-Audio scheduler and retained prefill cursors diverged".into(),
                ));
            }
            let checkpoint = active.state.begin_quantum(&cache)?;
            lease.mark_dirty();
            rows.push(Lfm3AudioBatchRow {
                index,
                lease: Some(lease),
                cache,
                cache_start,
                checkpoint: Some(checkpoint),
            });
        }
        let mut batch = Lfm3AudioBatch { rows, armed: true };
        for row in &mut batch.rows {
            if requests[row.index].is_cancelled() {
                row.rollback()?;
                outputs[row.index] = Some(ModelSessionResult::cancelled(
                    ExecutorOutput::cancelled(requests[row.index].id.clone()),
                ));
            }
        }

        let armed = batch
            .rows
            .iter()
            .filter(|row| row.checkpoint.is_some())
            .count();
        if armed > 0 {
            if is_prefill {
                for row in &mut batch.rows {
                    if row.checkpoint.is_none() {
                        continue;
                    }
                    let span = scheduled[row.index].num_tokens;
                    let Lfm3AudioBatchRow { lease, cache, .. } = row;
                    let state = &mut lease
                        .as_mut()
                        .expect("armed LFM3-Audio lease")
                        .require_state_mut()?
                        .state;
                    Self::run_blocking(|| lfm3.retained_prefill(state, cache, span))?;
                }
            } else {
                let mut step_rows = Vec::with_capacity(armed);
                for row in &mut batch.rows {
                    if row.checkpoint.is_none() {
                        continue;
                    }
                    let Lfm3AudioBatchRow { lease, cache, .. } = row;
                    step_rows.push(Lfm3AudioStepRow {
                        state: &mut lease
                            .as_mut()
                            .expect("armed LFM3-Audio lease")
                            .require_state_mut()?
                            .state,
                        cache,
                    });
                }
                Self::run_blocking(|| lfm3.retained_decode_batch(&mut step_rows))?;
            }
            let mode = if is_prefill {
                crate::engine::NativeBatchMode::Static
            } else {
                crate::engine::NativeBatchMode::Continuous
            };
            if let Some(call) = retained_tts_batch_model_call(mode, armed) {
                crate::engine::metrics::record_engine_model_call(call);
            }
        }

        // Stage every ring, then check cancellation again before the
        // all-row commit.
        for row in &mut batch.rows {
            if row.checkpoint.is_none() {
                continue;
            }
            let index = row.index;
            let arena = requests[index]
                .managed_cache_runtime()
                .and_then(|runtime| runtime.tensor_state())
                .ok_or_else(|| {
                    Error::InferenceError("LFM3-Audio row lost its ShortConv arena".into())
                })?;
            row.state_mut()?
                .stage_shortconv(arena, scheduled[index].plan_id)?;
        }
        for row in &mut batch.rows {
            if row.checkpoint.is_some() && requests[row.index].is_cancelled() {
                row.rollback()?;
                outputs[row.index] = Some(ModelSessionResult::cancelled(
                    ExecutorOutput::cancelled(requests[row.index].id.clone()),
                ));
            }
        }
        for row in &mut batch.rows {
            let Some(checkpoint) = row.checkpoint.as_ref() else {
                continue;
            };
            row.lease
                .as_mut()
                .expect("armed LFM3-Audio lease")
                .require_state_mut()?
                .state
                .commit_quantum(&row.cache, checkpoint)?;
        }
        for row in &mut batch.rows {
            if row.checkpoint.take().is_none() {
                continue;
            }
            let index = row.index;
            let request = requests[index];
            let completions = row.cache.take_completed_writes();
            let append = row
                .cache
                .context_len()
                .checked_sub(row.cache_start)
                .ok_or_else(|| {
                    Error::InferenceError("LFM3-Audio row cache cursor regressed".into())
                })?;
            let mut lease = row.lease.take().expect("armed LFM3-Audio lease");
            let active = lease.require_state_mut()?;
            let finished = active.state.finished();
            let (audio, text) = match (request.task_type, active.input_audio) {
                (TaskType::ASR, Some((sample_rate, input_samples))) => {
                    let delta = active.state.take_text();
                    if let Some(tx) = Self::stream_sender(request) {
                        if !delta.is_empty() {
                            Self::stream_text_with_policy(
                                &tx,
                                request.stream_policy,
                                &request.id,
                                &mut active.stream_sequence,
                                delta,
                            )?;
                        }
                        if finished {
                            Self::stream_final_marker_with_policy(
                                &tx,
                                request.stream_policy,
                                &request.id,
                                &mut active.stream_sequence,
                            )?;
                        }
                    }
                    let text = active
                        .state
                        .output()
                        .map(|output| output.text.clone())
                        .unwrap_or_default();
                    let audio = AudioOutput {
                        streamed_samples: None,
                        samples: Vec::new(),
                        sample_rate,
                        duration_secs: input_samples as f32 / sample_rate.max(1) as f32,
                    };
                    (audio, Some(text))
                }
                (TaskType::ASR, None) => {
                    return Err(Error::InferenceError(
                        "LFM3-Audio transcription row lost its audio".into(),
                    ))
                }
                (TaskType::SpeechToSpeech, _) => {
                    self.lfm3_audio_reply_output(request, active, is_prefill)?
                }
                _ => (
                    self.lfm3_audio_speech_output(request, active, is_prefill)?,
                    None,
                ),
            };
            lease.mark_clean();
            if finished {
                lease.release()?;
            } else {
                lease.restore()?;
            }
            let output = ExecutorOutput {
                request_id: request.id.clone(),
                audio: Some(audio),
                text,
                input_transcription: None,
                tokens_processed: if is_prefill {
                    scheduled[index].num_tokens
                } else {
                    append
                },
                tokens_generated: usize::from(!is_prefill),
                finished,
                phase_timing_override: None,
                asr_diagnostics: None,
                error: None,
                logprobs: Vec::new(),
            };
            outputs[index] = Some(
                ModelSessionResult::sequence(output)
                    .with_managed_cache_completions(completions)
                    .with_managed_cache_append(append),
            );
        }
        batch.armed = false;
        collect_outputs(outputs)
    }
}

impl NativeExecutor {
    /// A speech row's audio for this quantum: streamed as the codec decoded
    /// it, or kept until a final-only request finishes.
    fn lfm3_audio_speech_output(
        &self,
        request: &EngineCoreRequest,
        active: &mut ActiveLfm3AudioDecode,
        is_prefill: bool,
    ) -> Result<AudioOutput> {
        if is_prefill {
            return Ok(AudioOutput::new(Vec::new(), SAMPLE_RATE));
        }
        let finished = active.state.finished();
        if finished
            && active
                .state
                .output()
                .is_some_and(|output| output.audio_frames == 0)
        {
            return Err(Error::InferenceError(format!(
                "LFM3-Audio spoke no audio for request {}",
                request.id
            )));
        }
        let samples = active.state.take_samples();
        Ok(match Self::stream_sender(request) {
            Some(tx) => stream_committed_tts_audio(
                &tx,
                request.stream_policy,
                &request.id,
                &mut active.stream_sequence,
                &mut active.streamed_samples,
                samples,
                SAMPLE_RATE,
                finished,
            )?,
            None => {
                active.rendered.extend(samples);
                if finished {
                    AudioOutput::new(std::mem::take(&mut active.rendered), SAMPLE_RATE)
                } else {
                    AudioOutput::new(Vec::new(), SAMPLE_RATE)
                }
            }
        })
    }
}

impl NativeExecutor {
    /// A voice-chat row's reply for this quantum: its text and speech
    /// streamed as they decode, or kept until a final-only turn finishes.
    /// The finished quantum returns the whole reply's text.
    fn lfm3_audio_reply_output(
        &self,
        request: &EngineCoreRequest,
        active: &mut ActiveLfm3AudioDecode,
        is_prefill: bool,
    ) -> Result<(AudioOutput, Option<String>)> {
        if is_prefill {
            return Ok((AudioOutput::new(Vec::new(), SAMPLE_RATE), None));
        }
        let finished = active.state.finished();
        let delta = active.state.take_text();
        let samples = active.state.take_samples();
        let audio = match Self::stream_sender(request) {
            Some(tx) => {
                if !delta.is_empty() {
                    Self::stream_text_with_policy(
                        &tx,
                        request.stream_policy,
                        &request.id,
                        &mut active.stream_sequence,
                        delta,
                    )?;
                }
                stream_committed_tts_audio(
                    &tx,
                    request.stream_policy,
                    &request.id,
                    &mut active.stream_sequence,
                    &mut active.streamed_samples,
                    samples,
                    SAMPLE_RATE,
                    finished,
                )?
            }
            None => {
                active.rendered.extend(samples);
                if finished {
                    AudioOutput::new(std::mem::take(&mut active.rendered), SAMPLE_RATE)
                } else {
                    AudioOutput::new(Vec::new(), SAMPLE_RATE)
                }
            }
        };
        // The reply's text is trimmed for the voice turn's transcript.
        let text = finished.then(|| {
            active
                .state
                .output()
                .map(|output| output.text.as_str())
                .unwrap_or_default()
                .trim()
                .to_string()
        });
        Ok((audio, text))
    }
}

fn collect_outputs(outputs: Vec<Option<ModelSessionResult>>) -> Result<Vec<ModelSessionResult>> {
    outputs
        .into_iter()
        .map(|output| {
            output.ok_or_else(|| Error::InferenceError("LFM3-Audio row produced no output".into()))
        })
        .collect()
}
