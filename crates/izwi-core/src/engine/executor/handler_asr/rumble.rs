//! Rumble's batched prefill on CUDA: the retained rows one step prefills
//! run as one mixed trunk step. A row's first span starts its retained
//! state as the shared retained sequence handler (`granite.rs`) does; a
//! later span resumes it under a checkpoint. Every row's quantum commits
//! only after the whole step succeeds, and a row cancelled around the call
//! rolls back alone.

use super::*;

/// How a prefill row undoes its quantum.
enum RumblePrefillUndo {
    /// Roll the resumed state back to before this step.
    Checkpoint(NativeAsrDecodeCheckpoint),
    /// Drop the state this step started.
    Fresh,
}

struct RumblePrefillRow<'a> {
    index: usize,
    span: (usize, usize),
    lease: ExecutorStateLease<'a, ActiveAsrDecode>,
    /// `None` once the row committed or rolled back.
    undo: Option<RumblePrefillUndo>,
}

impl RumblePrefillRow<'_> {
    fn rollback(&mut self) -> Result<()> {
        match self.undo.take() {
            Some(RumblePrefillUndo::Checkpoint(checkpoint)) => {
                self.lease
                    .require_state_mut()?
                    .state
                    .rollback_managed_quantum(checkpoint)?;
                self.lease.mark_clean();
            }
            Some(RumblePrefillUndo::Fresh) => {
                self.lease.discard_state();
                self.lease.mark_clean();
            }
            None => {}
        }
        Ok(())
    }

    fn commit(&mut self) -> Result<()> {
        if let Some(RumblePrefillUndo::Checkpoint(checkpoint)) = self.undo.take() {
            self.lease
                .require_state_mut()?
                .state
                .commit_managed_quantum(checkpoint)?;
        }
        Ok(())
    }
}

/// The rows of one prefill step. Dropped before they commit (an error),
/// every row rolls back; a row whose rollback fails stays dirty, and its
/// lease fences the session until cleanup.
struct RumblePrefillBatch<'a> {
    rows: Vec<RumblePrefillRow<'a>>,
}

impl Drop for RumblePrefillBatch<'_> {
    fn drop(&mut self) {
        for row in &mut self.rows {
            if let Err(error) = row.rollback() {
                tracing::error!(
                    row = row.index,
                    %error,
                    "Rumble prefill rollback failed; state fenced until cleanup"
                );
            }
        }
    }
}

impl NativeExecutor {
    pub(in crate::engine::executor) fn rumble_asr_prefill_batch_with_managed(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        mut managed: Vec<Option<super::super::RetainedRowManagedState>>,
    ) -> Result<Vec<ModelSessionResult>> {
        if requests.len() != scheduled.len() || managed.len() != scheduled.len() {
            return Err(Error::InvalidInput(
                "Rumble prefill batch geometry is inconsistent".into(),
            ));
        }
        let mut outputs = (0..requests.len())
            .map(|_| None)
            .collect::<Vec<Option<ModelSessionResult>>>();
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
        let collect = |outputs: Vec<Option<ModelSessionResult>>| {
            outputs
                .into_iter()
                .map(|output| {
                    output.ok_or_else(|| Error::InferenceError("Rumble prefill lost a row".into()))
                })
                .collect::<Result<Vec<_>>>()
        };
        if live.is_empty() {
            return collect(outputs);
        }
        let model = requests[live[0]]
            .prepared_asr_model_for_executor()?
            .ok_or_else(|| Error::InferenceError("Rumble prefill lost its loaded model".into()))?;
        if !model.supports_static_prefill_batch() {
            return Err(Error::InvalidInput(
                "loaded Rumble model does not prefill rows together".into(),
            ));
        }

        let mut batch = RumblePrefillBatch {
            rows: Vec::with_capacity(live.len()),
        };
        for index in live {
            let request = requests[index];
            let variant = Self::resolve_variant(request)?;
            let row_model = request.prepared_asr_model_for_executor()?.ok_or_else(|| {
                Error::InferenceError("Rumble prefill row lost its loaded model".into())
            })?;
            if variant.family() != ModelFamily::RumbleAsr
                || !Arc::ptr_eq(&row_model, &model)
                || !scheduled[index].is_prefill
            {
                return Err(Error::InvalidInput(
                    "Rumble prefill batch crossed its loaded sequence identity".into(),
                ));
            }
            let prompt_tokens = request.num_prompt_tokens();
            let span = resumable_asr_prefill_span(&scheduled[index], prompt_tokens)?;
            let mut retained = managed[index].take().ok_or_else(|| {
                Error::InferenceError("Rumble prefill row lost retained paged state".into())
            })?;
            if retained.tensor_state.is_some() {
                return Err(Error::InferenceError(
                    "Rumble prefill row unexpectedly received tensor state".into(),
                ));
            }
            let cache = retained.take_only_paged()?;
            let mut lease = ExecutorStateLease::checkout(
                &self.asr_decode_states,
                scheduled[index].session_key(),
                variant,
                "Rumble ASR decode",
            )?;
            if lease.state().is_some_and(|active| {
                active.variant != variant || !Arc::ptr_eq(&active.model, &model)
            }) {
                lease.discard_state();
            }
            let undo = if lease.state().is_some() {
                let checkpoint = lease
                    .require_state_mut()?
                    .state
                    .begin_managed_quantum(cache)?;
                RumblePrefillUndo::Checkpoint(checkpoint)
            } else {
                if span.0 != 0 {
                    return Err(Error::InferenceError(
                        "Rumble sequence lost state before initial prefill".into(),
                    ));
                }
                let artifact = request
                    .prepared_rumble_artifact_for_executor()?
                    .ok_or_else(|| {
                        Error::InferenceError(
                            "Rumble sequence lost its prepared prompt artifact".into(),
                        )
                    })?;
                let (samples, sample_rate) =
                    request.prepared_asr_audio_for_executor()?.ok_or_else(|| {
                        Error::InferenceError("Rumble sequence lost decoded audio".into())
                    })?;
                let options = Self::asr_chunk_generation_options(
                    request,
                    variant.family(),
                    samples.len(),
                    sample_rate,
                    &Self::asr_generation_options(request),
                    self.asr_effective_context(variant)
                        .zip(model.audio_token_rate()),
                );
                let state =
                    model.start_rumble_resumable_prefill_managed(artifact, options, cache)?;
                if state.prefill_token_count() != Some(prompt_tokens)
                    || state.prefill_progress() != Some(0)
                    || state.sequence_position() != Some(0)
                {
                    return Err(Error::InferenceError(
                        "Rumble prompt geometry differs from scheduler admission".into(),
                    ));
                }
                let model_lease = request
                    .prepared_asr_model_lease_for_executor()?
                    .ok_or_else(|| {
                        Error::InferenceError("Rumble sequence lost model residency".into())
                    })?;
                lease.install_state(ActiveAsrDecode {
                    variant,
                    model: model.clone(),
                    _model_lease: model_lease,
                    state,
                    last_tokens_generated: 0,
                    stream_sequence: 0,
                    input_sample_rate: sample_rate,
                    input_sample_count: samples.len(),
                    clock: Default::default(),
                })?;
                RumblePrefillUndo::Fresh
            };
            lease.mark_dirty();
            batch.rows.push(RumblePrefillRow {
                index,
                span,
                lease,
                undo: Some(undo),
            });
        }

        // A row cancelled while the batch was assembled rolls back and stays
        // out of the model call.
        let cancel = |row: &mut RumblePrefillRow<'_>,
                      outputs: &mut Vec<Option<ModelSessionResult>>|
         -> Result<()> {
            row.rollback()?;
            let request = requests[row.index];
            let _ = request.take_staged_stream_outputs()?;
            outputs[row.index] = Some(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                request.id.clone(),
            )));
            Ok(())
        };
        for row in &mut batch.rows {
            if requests[row.index].is_cancelled() {
                cancel(row, &mut outputs)?;
            }
        }
        let started = Instant::now();
        let called = {
            let mut rows = batch
                .rows
                .iter_mut()
                .filter(|row| outputs[row.index].is_none())
                .map(|row| {
                    let (start, end) = row.span;
                    row.lease
                        .require_state_mut()
                        .map(|active| (&mut active.state, start, end))
                })
                .collect::<Result<Vec<_>>>()?;
            if !rows.is_empty() {
                let complete =
                    Self::run_blocking(|| model.continue_resumable_prefill_batch(&mut rows))?;
                if complete.len() != rows.len() {
                    return Err(Error::InferenceError(
                        "Rumble prefill batch returned the wrong row count".into(),
                    ));
                }
            }
            rows.len()
        };
        let elapsed = started.elapsed();
        if let Some(call) =
            retained_asr_batch_model_call(crate::engine::NativeBatchMode::Static, called)
        {
            crate::engine::metrics::record_engine_model_call(call);
        }

        for row in &mut batch.rows {
            if outputs[row.index].is_some() {
                continue;
            }
            let request = requests[row.index];
            if request.is_cancelled() {
                cancel(row, &mut outputs)?;
                continue;
            }
            let active = row.lease.require_state_mut()?;
            active.clock.record_prefill(elapsed);
            let sample_rate = active.input_sample_rate;
            let completions = active.state.take_managed_write_completions();
            outputs[row.index] = Some(
                ModelSessionResult::sequence(ExecutorOutput {
                    request_id: request.id.clone(),
                    audio: Some(AudioOutput {
                        streamed_samples: None,
                        samples: Vec::new(),
                        sample_rate,
                        duration_secs: if sample_rate == 0 {
                            0.0
                        } else {
                            active.input_sample_count as f32 / sample_rate as f32
                        },
                    }),
                    text: Some(String::new()),
                    input_transcription: None,
                    tokens_processed: scheduled[row.index].num_tokens,
                    tokens_generated: 0,
                    finished: false,
                    phase_timing_override: None,
                    asr_diagnostics: None,
                    error: None,
                    logprobs: Vec::new(),
                })
                .with_managed_cache_completions(completions),
            );
        }

        // Recheck every armed row immediately before committing any: a row
        // cancelled while later rows were staged drops its output and
        // completions with its rollback.
        for row in &mut batch.rows {
            if late_cancelled_batch_row(requests[row.index].is_cancelled(), row.undo.is_some()) {
                cancel(row, &mut outputs)?;
            }
        }
        let mut rows = std::mem::take(&mut batch.rows);
        drop(batch);
        let mut continuing = Vec::with_capacity(rows.len());
        for row in &mut rows {
            continuing.push(row.undo.is_some());
            row.commit()?;
        }
        for (row, continuing) in rows.into_iter().zip(continuing) {
            let index = row.index;
            let transition = if continuing {
                row.lease.restore()
            } else {
                row.lease.release()
            };
            if let Err(error) = transition {
                outputs[index] = Some(ModelSessionResult::sequence(ExecutorOutput::error(
                    requests[index].id.clone(),
                    format!("Rumble prefill state transition failed: {error}"),
                )));
            }
        }
        collect(outputs)
    }
}
