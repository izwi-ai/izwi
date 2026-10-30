//! Nemotron 3.5 Transcribe requests as retained continuous rows: each
//! quantum prefills a row's prompt span (encoding the perception windows its
//! audio positions need) or steps every row one token with the decoder
//! batched across rows, over each row's retained paged KV and Mamba state.
//! The Mamba state is staged into the tensor-state arena every quantum and
//! committed with the KV pages, so a failed or cancelled quantum rolls the
//! row back to where it started.

use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use crate::models::architectures::nemotron_transcribe::{
    NemotronRetainedCheckpoint, NemotronRetainedState, NemotronStepRow,
};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

use super::super::request::EngineCoreRequest;
use super::super::scheduler::ScheduledRequest;
use super::super::types::AudioOutput;
use super::state::ActiveNemotronTranscribeDecode;
use super::{ExecutorOutput, ExecutorStateLease, ModelSessionResult, NativeExecutor};

const SAMPLE_RATE: u32 = 16_000;

struct NemotronBatchRow<'a> {
    index: usize,
    lease: Option<ExecutorStateLease<'a, ActiveNemotronTranscribeDecode>>,
    cache: PhysicalPagedKvCache,
    cache_start: usize,
    checkpoint: Option<NemotronRetainedCheckpoint>,
}

impl NemotronBatchRow<'_> {
    fn state_mut(&mut self) -> Result<&mut NemotronRetainedState> {
        Ok(&mut self
            .lease
            .as_mut()
            .ok_or_else(|| {
                Error::InferenceError("Nemotron 3.5 Transcribe row lease is absent".into())
            })?
            .require_state_mut()?
            .state)
    }

    fn rollback(&mut self) -> Result<()> {
        let checkpoint = self.checkpoint.take().ok_or_else(|| {
            Error::InferenceError("Nemotron 3.5 Transcribe row has no rollback checkpoint".into())
        })?;
        let lease = self
            .lease
            .as_mut()
            .expect("armed Nemotron 3.5 Transcribe lease");
        lease
            .require_state_mut()?
            .state
            .rollback_quantum(&mut self.cache, &checkpoint)?;
        lease.mark_clean();
        Ok(())
    }
}

/// Rolls back every armed row if the quantum does not reach its commit.
struct NemotronBatch<'a> {
    rows: Vec<NemotronBatchRow<'a>>,
    armed: bool,
}

impl Drop for NemotronBatch<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        for row in &mut self.rows {
            if row.checkpoint.is_some() {
                if let Err(error) = row.rollback() {
                    tracing::error!(
                        %error,
                        "Nemotron 3.5 Transcribe rollback failed; state remains fenced"
                    );
                }
            }
        }
    }
}

impl NativeExecutor {
    pub(super) fn nemotron_transcribe_request_with_managed_cache(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        retained: Option<super::RetainedRowManagedState>,
    ) -> Result<ModelSessionResult> {
        let mut outputs = self.nemotron_transcribe_batch_with_managed(
            &[request],
            std::slice::from_ref(scheduled),
            vec![retained],
        )?;
        outputs.pop().ok_or_else(|| {
            Error::InferenceError("Nemotron 3.5 Transcribe row lost its output".into())
        })
    }

    /// One quantum for every row: a prompt span each, or one token each with
    /// the decoder batched across rows.
    pub(super) fn nemotron_transcribe_batch_with_managed(
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
                "Nemotron 3.5 Transcribe batch rows do not match".into(),
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
            return Err(Error::InvalidInput(
                "Nemotron 3.5 Transcribe batch mixed phases".into(),
            ));
        }
        let model_lease = requests[live[0]]
            .prepared_asr_model_lease_for_executor()?
            .ok_or_else(|| {
                Error::InferenceError("Nemotron 3.5 Transcribe batch lost its model".into())
            })?;
        let model_arc = model_lease.model_arc();
        let nemotron = model_arc.nemotron_transcribe()?;

        let mut rows = Vec::with_capacity(live.len());
        for &index in &live {
            let request = requests[index];
            let variant = Self::resolve_variant(request)?;
            let row_lease = request
                .prepared_asr_model_lease_for_executor()?
                .ok_or_else(|| {
                    Error::InferenceError("Nemotron 3.5 Transcribe row lost its model".into())
                })?;
            if variant.family() != ModelFamily::NemotronTranscribeAsr
                || !std::sync::Arc::ptr_eq(&model_arc, &row_lease.model_arc())
            {
                return Err(Error::InvalidInput(
                    "Nemotron 3.5 Transcribe batch crossed model identity".into(),
                ));
            }
            let mut retained = managed[index].take().ok_or_else(|| {
                Error::InferenceError("Nemotron 3.5 Transcribe row lost its retained state".into())
            })?;
            let tensor = retained.tensor_state.clone().ok_or_else(|| {
                Error::InferenceError(
                    "Nemotron 3.5 Transcribe row lost its Mamba state reservation".into(),
                )
            })?;
            let arena = request
                .managed_cache_runtime()
                .and_then(|runtime| runtime.tensor_state())
                .ok_or_else(|| {
                    Error::InferenceError(
                        "Nemotron 3.5 Transcribe row lost its tensor-state arena".into(),
                    )
                })?;
            let cache = retained.take_only_paged()?;
            let cache_start = cache.context_len();
            retained.ensure_all_paged_consumed()?;
            let mut lease = ExecutorStateLease::checkout(
                &self.nemotron_transcribe_decode_states,
                scheduled[index].session_key(),
                variant,
                "Nemotron 3.5 Transcribe row",
            )?;
            if lease.state().is_some_and(|active| {
                active.variant != variant
                    || !std::sync::Arc::ptr_eq(&active.model.model_arc(), &model_arc)
            }) {
                lease.discard_state();
            }
            if lease.state().is_none() {
                if !is_prefill || scheduled[index].num_computed_tokens != 0 {
                    return Err(Error::InferenceError(
                        "Nemotron 3.5 Transcribe row lost its initial state".into(),
                    ));
                }
                let artifact = request
                    .prepared_nemotron_transcribe_artifact_for_executor()?
                    .ok_or_else(|| {
                        Error::InferenceError("Nemotron 3.5 Transcribe row lost its prompt".into())
                    })?;
                // The row holds its prompt and every token it may decode.
                let context = artifact.prompt_tokens() + request.params.max_tokens;
                lease.install_state(ActiveNemotronTranscribeDecode {
                    variant,
                    model: row_lease,
                    state: nemotron.new_row(artifact, context)?,
                    stream_sequence: 0,
                })?;
            }
            let active = lease.require_state_mut()?;
            active.state.bind_tensor_sequence(tensor.sequence)?;
            active.state.restore_mamba(arena, nemotron)?;
            if is_prefill && active.state.prefill_cursor() != scheduled[index].num_computed_tokens {
                return Err(Error::InferenceError(
                    "Nemotron 3.5 Transcribe scheduler and retained prefill cursors diverged"
                        .into(),
                ));
            }
            let checkpoint = active.state.begin_quantum(&cache);
            lease.mark_dirty();
            rows.push(NemotronBatchRow {
                index,
                lease: Some(lease),
                cache,
                cache_start,
                checkpoint: Some(checkpoint),
            });
        }
        let mut batch = NemotronBatch { rows, armed: true };
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
                    let NemotronBatchRow { lease, cache, .. } = row;
                    let state = &mut lease
                        .as_mut()
                        .expect("armed Nemotron 3.5 Transcribe lease")
                        .require_state_mut()?
                        .state;
                    Self::run_blocking(|| nemotron.retained_prefill(state, cache, span))?;
                }
            } else {
                let mut step_rows = Vec::with_capacity(armed);
                for row in &mut batch.rows {
                    if row.checkpoint.is_none() {
                        continue;
                    }
                    let NemotronBatchRow { lease, cache, .. } = row;
                    step_rows.push(NemotronStepRow {
                        state: &mut lease
                            .as_mut()
                            .expect("armed Nemotron 3.5 Transcribe lease")
                            .require_state_mut()?
                            .state,
                        cache,
                    });
                }
                Self::run_blocking(|| nemotron.retained_decode_batch(&mut step_rows))?;
            }
            let mode = if is_prefill {
                crate::engine::NativeBatchMode::Static
            } else {
                crate::engine::NativeBatchMode::Continuous
            };
            if let Some(call) = super::handler_tts::retained_tts_batch_model_call(mode, armed) {
                crate::engine::metrics::record_engine_model_call(call);
            }
        }

        // Stage every row's Mamba state, then check cancellation again before
        // the all-row commit.
        for row in &mut batch.rows {
            if row.checkpoint.is_none() {
                continue;
            }
            let index = row.index;
            let arena = requests[index]
                .managed_cache_runtime()
                .and_then(|runtime| runtime.tensor_state())
                .ok_or_else(|| {
                    Error::InferenceError(
                        "Nemotron 3.5 Transcribe row lost its tensor-state arena".into(),
                    )
                })?;
            row.state_mut()?
                .stage_mamba(arena, scheduled[index].plan_id)?;
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
            if row.checkpoint.is_none() {
                continue;
            }
            row.lease
                .as_mut()
                .expect("armed Nemotron 3.5 Transcribe lease")
                .require_state_mut()?
                .state
                .commit_quantum(&row.cache)?;
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
                    Error::InferenceError(
                        "Nemotron 3.5 Transcribe row cache cursor regressed".into(),
                    )
                })?;
            let mut lease = row
                .lease
                .take()
                .expect("armed Nemotron 3.5 Transcribe lease");
            let active = lease.require_state_mut()?;
            let finished = active.state.finished();
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
            let artifact = active.state.artifact().clone();
            let (text, asr_diagnostics) = if finished {
                let transcription = nemotron.finish_row(&active.state)?;
                let mut diagnostics = serde_json::json!({
                    "family": "nemotron_transcribe_asr",
                    "finish_reason": transcription.finish_reason.as_str(),
                    "raw_text": transcription.transcript.raw,
                    // Timed from the CTC words when the request asked for
                    // timestamps and the stage aligned them.
                    "speaker_turns": transcription
                        .timed_turns()
                        .iter()
                        .map(|turn| {
                            serde_json::json!({
                                "speaker": turn.speaker,
                                "text": turn.text,
                                "start": turn.start,
                                "end": turn.end,
                            })
                        })
                        .collect::<Vec<_>>(),
                    "prompt_tokens": transcription.prompt_tokens,
                    "audio_tokens": transcription.audio_tokens,
                    "generated_tokens": transcription.generated_tokens,
                    "perception_windows": transcription.windows,
                    "greedy_near_ties": active
                        .state
                        .decode_loop()
                        .map_or(0, |decode| decode.near_ties()),
                });
                // NeMo's public `ctc_timestamps` object, and its words as the
                // `timestamp_words` the transcription routes read, without
                // the event and style tags the returned text leaves out.
                if let Some(timestamps) = &transcription.timestamps {
                    let words = timestamps
                        .words
                        .iter()
                        .filter(|word| !(word.word.starts_with('<') && word.word.ends_with('>')))
                        .collect::<Vec<_>>();
                    diagnostics["timestamp_words"] = serde_json::to_value(words)
                        .map_err(|err| Error::InferenceError(err.to_string()))?;
                    diagnostics["ctc_timestamps"] = serde_json::to_value(timestamps)
                        .map_err(|err| Error::InferenceError(err.to_string()))?;
                }
                (Some(transcription.transcript.text), Some(diagnostics))
            } else {
                (Some(String::new()), None)
            };
            let audio = AudioOutput {
                streamed_samples: None,
                samples: Vec::new(),
                sample_rate: SAMPLE_RATE,
                duration_secs: artifact.duration_secs,
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
                asr_diagnostics,
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

fn collect_outputs(outputs: Vec<Option<ModelSessionResult>>) -> Result<Vec<ModelSessionResult>> {
    outputs
        .into_iter()
        .map(|output| {
            output.ok_or_else(|| {
                Error::InferenceError("Nemotron 3.5 Transcribe row produced no output".into())
            })
        })
        .collect()
}
