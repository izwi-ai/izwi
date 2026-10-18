//! Whisper ASR: the retained sequence path and its terminal transitions.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WhisperTerminalAction {
    Publish,
    Restart,
}

pub(super) fn whisper_prefill_boundary_step(last_tokens_generated: usize) -> NativeAsrDecodeStep {
    NativeAsrDecodeStep {
        delta: String::new(),
        text: String::new(),
        tokens_generated: last_tokens_generated,
        finished: false,
    }
}

pub(super) fn apply_whisper_terminal_transition(
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

pub(super) fn resolve_whisper_terminal_action(
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

pub(super) fn begins_whisper_managed_generation(
    scheduled: &ScheduledRequest,
    generation: crate::engine::ManagedSessionGeneration,
) -> bool {
    scheduled.is_prefill
        && scheduled.num_computed_tokens == 0
        && generation != crate::engine::ManagedSessionGeneration::INITIAL
}

pub(super) fn with_whisper_invocation_state<T>(
    request: &EngineCoreRequest,
    scheduled: &ScheduledRequest,
    run: impl FnOnce(
        &mut crate::models::shared::attention::physical::PhysicalPagedKvCache,
        &mut crate::engine::InvocationStaticAttentionLease,
    ) -> Result<T>,
) -> Result<T> {
    let mut leases = super::super::invocation_workspace_leases_for_atomic_scalar_row(request, scheduled)?;
    let output = {
        let (paged, cross) = leases.lease_exact_kind_pair_mut(
            crate::kv::v2::InvocationStateBackingKindV2::PagedAttention,
            crate::kv::v2::InvocationStateBackingKindV2::StaticAttention,
        )?;
        let cache = paged.paged_cache_mut()?;
        let cross = cross.typed_mut::<crate::engine::InvocationStaticAttentionLease>()?;
        run(cache, cross)
    };
    let output = leases.fence_on_error(output)?;
    let completions = leases.release()?;
    if completions.len() != 2 {
        return Err(Error::InferenceError(
            "Whisper ASR physical state returned an incomplete completion set".to_string(),
        ));
    }
    Ok(output)
}

impl NativeExecutor {
    pub(super) fn whisper_asr_sequence_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        mut managed_state: Option<super::super::RetainedRowManagedState>,
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
        let mut state_lease = ExecutorStateLease::checkout(
            &self.asr_decode_states,
            session,
            variant,
            "Whisper decode",
        )?;
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
            let mut diagnostics = None;
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
                    if !restart {
                        diagnostics = active.model.whisper_terminal_diagnostics(&active.state)?;
                    }
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
                diagnostics,
            ))
        })();
        let (
            generated,
            text,
            finished,
            cancelled,
            sample_rate,
            sample_count,
            completions,
            restart,
            diagnostics,
        ) = match execution {
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
            asr_diagnostics: diagnostics,
            error: None,
            logprobs: Vec::new(),
        })
        .with_managed_cache_completions(completions))
    }
}
