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

/// What a retained Whisper row does once one step has run.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum WhisperStepSettlement {
    /// The attempt goes on and publishes nothing yet.
    Continue,
    /// The terminal policy accepted or skipped the attempt: publish the
    /// step's text with these diagnostics.
    Publish {
        diagnostics: Option<serde_json::Value>,
    },
    /// The fallback policy rejected the attempt: restart the sequence at
    /// its next temperature.
    Restart,
    /// The request was cancelled before the outcome could publish.
    Cancelled,
}

/// Settle one row's `step`, as the scalar and the batched decode both do. An
/// unfinished step publishes no provisional text (the fallback policy can
/// still reject the attempt at EOS); a finished one runs the terminal
/// policy on `row` (`resolve`), and an accepted or skipped attempt reads
/// its diagnostics (`diagnostics`). Cancellation observed around the
/// policy wins.
pub(super) fn settle_whisper_step<R>(
    request: &EngineCoreRequest,
    step: &mut NativeAsrDecodeStep,
    reservation_generation: crate::engine::ManagedSessionGeneration,
    row: &mut R,
    resolve: impl FnOnce(&mut R) -> Result<WhisperTerminalTransition>,
    diagnostics: impl FnOnce(&R) -> Result<Option<serde_json::Value>>,
) -> Result<WhisperStepSettlement> {
    if !step.finished {
        step.delta.clear();
        step.text.clear();
        return Ok(WhisperStepSettlement::Continue);
    }
    Ok(
        match resolve_whisper_terminal_action(request, step, reservation_generation, || {
            resolve(row)
        })? {
            None => WhisperStepSettlement::Cancelled,
            Some(WhisperTerminalAction::Restart) => WhisperStepSettlement::Restart,
            Some(WhisperTerminalAction::Publish) => WhisperStepSettlement::Publish {
                diagnostics: diagnostics(row)?,
            },
        },
    )
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
                clock: Default::default(),
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
                match settle_whisper_step(
                    request,
                    &mut step,
                    session_generation,
                    &mut *active,
                    |active| {
                        active
                            .model
                            .resolve_whisper_terminal_transition(&mut active.state)
                    },
                    |active| active.model.whisper_terminal_diagnostics(&active.state),
                )? {
                    WhisperStepSettlement::Continue => {}
                    WhisperStepSettlement::Cancelled => {
                        cancelled = true;
                        break;
                    }
                    WhisperStepSettlement::Restart => restart = true,
                    WhisperStepSettlement::Publish {
                        diagnostics: published,
                    } => diagnostics = published,
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

impl NativeExecutor {
    /// One continuous decode step of retained Whisper rows: every live row
    /// decodes one token in one batched model call, then settles alone. A
    /// row the terminal policy accepts or skips publishes its transcript and
    /// releases its state; a row the fallback policy rejects commits its
    /// retry intent and restarts its sequence (the scheduler then prefills
    /// the next temperature's attempt on the scalar prefill stage); a
    /// cancelled row rolls back to its checkpoint. Rows never fail each
    /// other once the model call succeeded.
    pub(super) fn whisper_asr_decode_batch_with_managed(
        &self,
        ordered_requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        mut managed_caches: Vec<Option<super::super::RetainedRowManagedState>>,
    ) -> Result<Vec<ModelSessionResult>> {
        let mut outputs = (0..scheduled.len())
            .map(|_| None)
            .collect::<Vec<Option<ModelSessionResult>>>();
        let mut live_indices = Vec::with_capacity(scheduled.len());
        for (index, request) in ordered_requests.iter().enumerate() {
            if request.is_cancelled() {
                outputs[index] = Some(ModelSessionResult::cancelled_before_dispatch(
                    ExecutorOutput::cancelled(request.id.clone()),
                ));
            } else {
                live_indices.push(index);
            }
        }
        let Some(&first) = live_indices.first() else {
            return collect_whisper_batch_outputs(outputs);
        };
        let model = ordered_requests[first]
            .prepared_asr_model_for_executor()?
            .ok_or_else(|| {
                Error::InferenceError("continuous Whisper row lost its model identity".into())
            })?;
        let mut checked_out = Vec::with_capacity(live_indices.len());
        for index in live_indices.iter().copied() {
            let request = ordered_requests[index];
            let variant = Self::resolve_variant(request)?;
            let row_model = request.prepared_asr_model_for_executor()?.ok_or_else(|| {
                Error::InferenceError("continuous Whisper row lost its model identity".into())
            })?;
            if variant.family() != ModelFamily::WhisperAsr || !Arc::ptr_eq(&model, &row_model) {
                return Err(Error::InferenceError(
                    "continuous Whisper batch spans different loaded models".into(),
                ));
            }
            let session = scheduled[index].session_key();
            let lease = ExecutorStateLease::checkout(
                &self.asr_decode_states,
                session.clone(),
                variant,
                "continuous Whisper decode",
            )?;
            let state = lease.state().ok_or_else(|| {
                Error::InferenceError(format!(
                    "continuous Whisper session {}:{} has no active decode state",
                    session.request_id, session.epoch
                ))
            })?;
            if state.variant != variant || !Arc::ptr_eq(&state.model, &model) {
                return Err(Error::InferenceError(
                    "continuous Whisper state identity does not match its request".into(),
                ));
            }
            checked_out.push((index, session, lease));
        }

        let mut active_states = super::continuous::ContinuousAsrStateBatch::new(checked_out);
        let mut generations =
            vec![crate::engine::ManagedSessionGeneration::INITIAL; scheduled.len()];
        for (index, _, lease, checkpoint) in &mut active_states.rows {
            let mut views = managed_caches[*index].take().ok_or_else(|| {
                Error::InferenceError(
                    "continuous Whisper decode requires retained physical KV".into(),
                )
            })?;
            if views.tensor_state.is_some() {
                return Err(Error::InferenceError(
                    "continuous Whisper row unexpectedly received tensor state".into(),
                ));
            }
            generations[*index] = views.session_generation();
            let cache = views.take_only_paged()?;
            let state = lease.require_state_mut()?;
            let restore = (state.last_tokens_generated, state.stream_sequence);
            *checkpoint = Some((
                state.state.begin_managed_quantum(cache)?,
                restore.0,
                restore.1,
            ));
            lease.mark_dirty();
        }

        // Cancellation after the reservations rebound: roll those rows back
        // and leave them out of the model call.
        for row in 0..active_states.rows.len() {
            let index = active_states.rows[row].0;
            if ordered_requests[index].is_cancelled() {
                cancel_whisper_batch_row(&mut active_states, row, ordered_requests, &mut outputs)?;
            }
        }
        let call_rows = (0..active_states.rows.len())
            .filter(|row| outputs[active_states.rows[*row].0].is_none())
            .collect::<Vec<_>>();
        let steps = if call_rows.is_empty() {
            Vec::new()
        } else {
            let mut state_refs = active_states
                .rows
                .iter_mut()
                .filter(|(index, _, _, _)| outputs[*index].is_none())
                .map(|(_, _, lease, _)| lease.require_state_mut().map(|state| &mut state.state))
                .collect::<Result<Vec<_>>>()?;
            let steps =
                Self::run_blocking(|| model.decode_whisper_retained_step_batch(&mut state_refs))?;
            if let Some(call) = retained_asr_batch_model_call(
                crate::engine::NativeBatchMode::Continuous,
                state_refs.len(),
            ) {
                crate::engine::metrics::record_engine_model_call(call);
            }
            steps
        };
        if steps.len() != call_rows.len() {
            return Err(Error::InferenceError(
                "continuous Whisper model returned the wrong number of rows".into(),
            ));
        }

        let mut continuing = vec![false; scheduled.len()];
        let mut restarting = vec![false; scheduled.len()];
        for (row, mut step) in call_rows.into_iter().zip(steps) {
            let index = active_states.rows[row].0;
            let request = ordered_requests[index];
            if request.is_cancelled() {
                cancel_whisper_batch_row(&mut active_states, row, ordered_requests, &mut outputs)?;
                continue;
            }
            let active = active_states.rows[row].2.require_state_mut()?;
            let settlement = settle_whisper_step(
                request,
                &mut step,
                generations[index],
                &mut *active,
                |active| {
                    active
                        .model
                        .resolve_whisper_terminal_transition(&mut active.state)
                },
                |active| active.model.whisper_terminal_diagnostics(&active.state),
            );
            let diagnostics = match settlement {
                Ok(WhisperStepSettlement::Cancelled) => {
                    cancel_whisper_batch_row(
                        &mut active_states,
                        row,
                        ordered_requests,
                        &mut outputs,
                    )?;
                    continue;
                }
                Ok(WhisperStepSettlement::Restart) => {
                    // The scheduler aborts this attempt's reservation before
                    // advancing the session generation: no write receipt of
                    // the rejected attempt may escape.
                    drop(active.state.take_managed_write_completions());
                    active.last_tokens_generated = 0;
                    active.stream_sequence = 0;
                    restarting[index] = true;
                    outputs[index] = Some(ModelSessionResult::restart_sequence(
                        request.id.clone(),
                        SequenceRestartReason::ModelFallback,
                    ));
                    continue;
                }
                Ok(WhisperStepSettlement::Continue) => None,
                Ok(WhisperStepSettlement::Publish { diagnostics }) => diagnostics,
                Err(error) => {
                    let _ = request.take_staged_stream_outputs();
                    active_states.rollback_row(row)?;
                    outputs[index] = Some(ModelSessionResult::sequence(ExecutorOutput::error(
                        request.id.clone(),
                        format!("continuous Whisper terminal policy failed: {error}"),
                    )));
                    continue;
                }
            };
            let generated = step
                .tokens_generated
                .saturating_sub(active.last_tokens_generated);
            active.last_tokens_generated = step.tokens_generated;
            let streamed = (|| -> Result<()> {
                let Some(tx) = Self::stream_sender(request) else {
                    return Ok(());
                };
                if !step.delta.is_empty() {
                    Self::stream_text_with_policy(
                        &tx,
                        request.stream_policy,
                        &request.id,
                        &mut active.stream_sequence,
                        step.delta.clone(),
                    )?;
                }
                if step.finished {
                    Self::stream_final_marker_with_policy(
                        &tx,
                        request.stream_policy,
                        &request.id,
                        &mut active.stream_sequence,
                    )?;
                }
                Ok(())
            })();
            if let Err(error) = streamed {
                let _ = request.take_staged_stream_outputs();
                active_states.rollback_row(row)?;
                outputs[index] = Some(ModelSessionResult::sequence(ExecutorOutput::error(
                    request.id.clone(),
                    format!("continuous Whisper stream staging failed: {error}"),
                )));
                continue;
            }
            let (sample_rate, sample_count) = (active.input_sample_rate, active.input_sample_count);
            let completions = active.state.take_managed_write_completions();
            continuing[index] = !step.finished;
            outputs[index] = Some(
                ModelSessionResult::sequence(ExecutorOutput {
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
                    text: Some(step.text),
                    input_transcription: None,
                    // A row that goes on wrote its token's K/V; a finishing
                    // step (EOT, budget or repetition) decodes nothing more.
                    tokens_processed: usize::from(!step.finished),
                    tokens_generated: generated,
                    finished: step.finished,
                    phase_timing_override: None,
                    asr_diagnostics: diagnostics,
                    error: None,
                    logprobs: Vec::new(),
                })
                .with_managed_cache_completions(completions),
            );
        }

        // Rows cancelled while later rows settled roll back with their
        // private outputs and detached completions.
        for row in 0..active_states.rows.len() {
            let index = active_states.rows[row].0;
            if late_cancelled_batch_row(
                ordered_requests[index].is_cancelled(),
                active_states.rows[row].3.is_some(),
            ) {
                continuing[index] = false;
                restarting[index] = false;
                cancel_whisper_batch_row(&mut active_states, row, ordered_requests, &mut outputs)?;
            }
        }

        let committed = match active_states.commit() {
            Ok(states) => states,
            Err(error) => {
                for index in live_indices {
                    let _ = ordered_requests[index].take_staged_stream_outputs();
                }
                return Err(error);
            }
        };
        for (index, _, mut lease) in committed {
            let request = ordered_requests[index];
            let transition = if restarting[index] && request.is_cancelled() {
                // Cancelled after its restart committed: nothing restarts.
                outputs[index] = Some(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    request.id.clone(),
                )));
                lease.discard_state();
                lease.release()
            } else if restarting[index] || continuing[index] {
                lease.restore()
            } else {
                lease.release()
            };
            if let Err(error) = transition {
                outputs[index] = Some(ModelSessionResult::sequence(ExecutorOutput::error(
                    request.id.clone(),
                    format!("continuous Whisper state transition failed: {error}"),
                )));
            }
        }
        collect_whisper_batch_outputs(outputs)
    }
}

/// Roll `row` back to its checkpoint and answer it as cancelled.
fn cancel_whisper_batch_row(
    states: &mut super::continuous::ContinuousAsrStateBatch<'_>,
    row: usize,
    requests: &[&EngineCoreRequest],
    outputs: &mut [Option<ModelSessionResult>],
) -> Result<()> {
    let index = states.rollback_row(row)?;
    let request = requests[index];
    let _ = request.take_staged_stream_outputs()?;
    outputs[index] = Some(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
        request.id.clone(),
    )));
    Ok(())
}

fn collect_whisper_batch_outputs(
    outputs: Vec<Option<ModelSessionResult>>,
) -> Result<Vec<ModelSessionResult>> {
    outputs
        .into_iter()
        .map(|output| {
            output.ok_or_else(|| {
                Error::InferenceError("continuous Whisper row produced no result".into())
            })
        })
        .collect()
}
