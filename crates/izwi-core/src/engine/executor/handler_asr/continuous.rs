//! Continuous ASR decode: one batched decode step over retained rows,
//! dispatching LFM2.5-Audio and Parakeet rows to their own batches.

use super::*;

pub(super) fn validate_continuous_asr_batch_shape(scheduled: &[ScheduledRequest]) -> Result<()> {
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

pub(super) struct ContinuousAsrStateBatch<'a> {
    pub(super) rows: Vec<(
        usize,
        SessionKey,
        ExecutorStateLease<'a, ActiveAsrDecode>,
        Option<(NativeAsrDecodeCheckpoint, usize, usize)>,
    )>,
    pub(super) armed: bool,
}

impl<'a> ContinuousAsrStateBatch<'a> {
    pub(super) fn new(rows: Vec<(usize, SessionKey, ExecutorStateLease<'a, ActiveAsrDecode>)>) -> Self {
        Self {
            rows: rows
                .into_iter()
                .map(|(index, session, lease)| (index, session, lease, None))
                .collect(),
            armed: true,
        }
    }

    pub(super) fn commit(
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

    pub(super) fn rollback_row(&mut self, row: usize) -> Result<usize> {
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

impl NativeExecutor {
    pub(in crate::engine::executor) fn asr_decode_batch_with_managed(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        managed_caches: Vec<Option<super::super::RetainedRowManagedState>>,
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

        if Self::resolve_variant(ordered_requests[live_indices[0]])?.family()
            == ModelFamily::Lfm25Audio
        {
            return self.lfm25_audio_asr_decode_batch_with_managed(
                &ordered_requests,
                scheduled,
                managed_caches,
            );
        }
        if Self::resolve_variant(ordered_requests[live_indices[0]])?.family()
            == ModelFamily::Lfm3Audio
        {
            return self.lfm3_audio_batch_with_managed(
                &ordered_requests,
                scheduled,
                managed_caches,
            );
        }
        if Self::resolve_variant(ordered_requests[live_indices[0]])?.family()
            == ModelFamily::ParakeetAsr
        {
            return self.parakeet_asr_decode_batch_with_managed(
                &ordered_requests,
                scheduled,
                managed_caches,
            );
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
                expected_variant,
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

        for (row, cancelled) in call_rows.iter().copied().zip(cancelled_after_model) {
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
        for (row, step) in call_rows.iter().copied().zip(steps) {
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
                        streamed_samples: None,
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
                    logprobs: Vec::new(),
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
}
