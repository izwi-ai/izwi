//! VibeVoice ASR: the retained sequence path and the managed prefill batch.

use super::*;

pub(super) fn vibevoice_prefill_boundary_step(
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

pub(super) fn with_vibevoice_invocation_state<T>(
    request: &EngineCoreRequest,
    scheduled: &ScheduledRequest,
    run: impl FnOnce(&mut crate::kv::v2::InvocationWorkspaceLeaseSetV2) -> Result<T>,
) -> Result<T> {
    let mut leases = super::super::invocation_workspace_leases_for_atomic_scalar_row(request, scheduled)?;
    let result = run(&mut leases);
    let output = leases.fence_on_error(result)?;
    let completions = leases.release()?;
    if completions.len() != 3 {
        return Err(Error::InferenceError(
            "VibeVoice ASR physical state returned an incomplete completion set".to_string(),
        ));
    }
    Ok(output)
}

impl NativeExecutor {
    pub(super) fn vibevoice_asr_sequence_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        managed_state: Option<super::super::RetainedRowManagedState>,
    ) -> Result<ModelSessionResult> {
        self.vibevoice_asr_sequence_request_inner(request, scheduled, managed_state, None)
    }

    pub(super) fn vibevoice_asr_sequence_request_inner(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        mut managed_state: Option<super::super::RetainedRowManagedState>,
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
        let mut state_lease = ExecutorStateLease::checkout(
            &self.asr_decode_states,
            session,
            variant,
            "VibeVoice ASR decode",
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
                streamed_samples: None,
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
            logprobs: Vec::new(),
        })
        .with_managed_cache_completions(completions);
        Ok(match clocked_state_completion {
            Some(completion) => result.with_clocked_state_completion(completion),
            None => result,
        })
    }

    pub(in crate::engine::executor) fn vibevoice_prefill_batch_with_managed(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        mut managed_caches: Vec<Option<super::super::RetainedRowManagedState>>,
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
                    Self::resolve_variant(request)?,
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
}
