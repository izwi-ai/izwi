//! Qwen3-ASR: the retained sequence path.

use super::*;

impl NativeExecutor {
    pub(super) fn qwen3_asr_sequence_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        mut managed_state: Option<super::super::RetainedRowManagedState>,
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
            ExecutorStateLease::checkout(&self.asr_decode_states, session, variant, "ASR decode")?;
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
                streamed_samples: None,
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
            logprobs: Vec::new(),
        })
        .with_managed_cache_completions(managed_cache_completions))
    }
}
