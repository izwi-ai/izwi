use std::sync::Arc;

use crate::error::{Error, Result};
use crate::models::registry::{
    MixedInput, MixedRow, MixedRowStep, NativeChatDecodeCheckpoint, NativeChatDecodeState,
    NativeChatDecodeStep, NativeChatModel,
};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::chat::ChatGenerationConfig;
use crate::models::shared::chat::ChatMessage;

use super::super::request::EngineCoreRequest;
use super::super::scheduler::ScheduledRequest;
use super::super::types::AudioOutput;
use super::super::SessionKey;
use super::state::ActiveChatDecode;
use super::stop_sequences::StopSequenceFilter;
use super::{ExecutorOutput, ExecutorStateLease, ModelSessionResult, NativeExecutor};

fn begins_resumable_prefill_state(
    scheduled: &ScheduledRequest,
    resumable_prefill: bool,
    request: &EngineCoreRequest,
) -> bool {
    scheduled.is_prefill
        && resumable_prefill
        && (scheduled.num_computed_tokens == 0
            // DS1.5: a jumped request starts its first span at the probed
            // prefix cursor before it has ever executed.
            || Some(
                u32::try_from(scheduled.num_computed_tokens).unwrap_or(u32::MAX),
            ) == request.managed_prefix_cursor())
}

fn finish_resumable_prefill_step(
    prefill_complete: bool,
    last_tokens_generated: usize,
    publish_bootstrap: impl FnOnce() -> Result<NativeChatDecodeStep>,
) -> Result<NativeChatDecodeStep> {
    if prefill_complete {
        // Publish the already-computed first token in the same transaction as
        // the final prompt span. No additional prompt KV write is performed.
        return publish_bootstrap();
    }
    Ok(NativeChatDecodeStep {
        delta: String::new(),
        text: String::new(),
        tokens_generated: last_tokens_generated,
        input_tokens_committed: 0,
        finished: false,
        logprobs: Vec::new(),
    })
}

fn resumable_prefill_span(
    scheduled: &ScheduledRequest,
    prompt_tokens: usize,
) -> Result<(usize, usize)> {
    let start = scheduled.num_computed_tokens;
    let end = start.checked_add(scheduled.num_tokens).ok_or_else(|| {
        Error::InvalidInput("resumable prefill span overflowed prompt accounting".into())
    })?;
    let crate::engine::WorkUnit::SequenceStep { phase, input, .. } = &scheduled.work else {
        return Err(Error::InvalidInput(
            "resumable prefill requires sequence-prefill work".into(),
        ));
    };
    if *phase != crate::engine::SequencePhase::Prefill
        || input.start != start
        || input.end != end
        || start >= end
        || end > prompt_tokens
    {
        return Err(Error::InvalidInput(format!(
            "resumable prefill work [{}, {}) disagrees with scheduler span [{start}, {end}) for {prompt_tokens} prompt tokens",
            input.start, input.end
        )));
    }
    Ok((start, end))
}

struct ContinuousChatStateBatch<'a> {
    rows: Vec<(
        usize,
        SessionKey,
        ExecutorStateLease<'a, ActiveChatDecode>,
        Option<NativeChatDecodeCheckpoint>,
    )>,
    armed: bool,
}

impl<'a> ContinuousChatStateBatch<'a> {
    fn new(rows: Vec<(usize, SessionKey, ExecutorStateLease<'a, ActiveChatDecode>)>) -> Self {
        Self {
            rows: rows
                .into_iter()
                .map(|(index, session, lease)| (index, session, lease, None))
                .collect(),
            armed: true,
        }
    }

    fn commit(mut self) -> Vec<(usize, SessionKey, ExecutorStateLease<'a, ActiveChatDecode>)> {
        self.armed = false;
        std::mem::take(&mut self.rows)
            .into_iter()
            .map(|(index, session, lease, _)| (index, session, lease))
            .collect()
    }
}

impl Drop for ContinuousChatStateBatch<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        for (_, session, lease, checkpoint) in &mut self.rows {
            if let Some(checkpoint) = checkpoint.take() {
                match lease
                    .require_state_mut()
                    .and_then(|state| state.state.rollback_continuous_quantum(checkpoint))
                {
                    Ok(()) => lease.mark_clean(),
                    Err(error) => {
                        tracing::error!(
                            request_id = %session.request_id,
                            epoch = session.epoch,
                            %error,
                            "continuous chat rollback failed; state fenced until cleanup"
                        );
                    }
                }
            }
        }
        // Dropping each lease restores clean/rolled-back rows and poisons any
        // row whose physical mutation could not be rolled back.
        self.rows.clear();
    }
}

// Correlate model-local diagnostics with the serving request without logging
// prompt/response content. Error paths already carry the request's executor ID.
fn record_lfm_completion(
    state: &NativeChatDecodeState,
    request: &EngineCoreRequest,
    step: &NativeChatDecodeStep,
) {
    if let NativeChatDecodeState::Lfm2(state) = state {
        if step.finished {
            tracing::info!(request_id = %request.id, model = ?request.model_variant,
                stop_reason = state.stop_reason(), generated_tokens = step.tokens_generated,
                prompt_tokens = request.num_prompt_tokens(), max_new_tokens = request.params.max_tokens,
                raw_chars = step.text.chars().count(), "LFM2 generation completed");
        }
    }
}

/// Whether a family's multi-token quantum is speculative (MTP drafts
/// verified several tokens per forward). Such a quantum cannot stop between
/// tokens, so requests with stop strings decode one token per quantum there.
pub(super) fn decodes_speculative_quanta(model: &NativeChatModel) -> bool {
    matches!(
        model,
        NativeChatModel::Qwen38(_) | NativeChatModel::Qwen36Moe(_)
    )
}

/// Fold the next one-token step of a quantum into the quantum's step.
fn append_decode_step(quantum: &mut NativeChatDecodeStep, next: NativeChatDecodeStep) {
    quantum.delta.push_str(&next.delta);
    quantum.text = next.text;
    quantum.tokens_generated = next.tokens_generated;
    quantum.input_tokens_committed = quantum
        .input_tokens_committed
        .saturating_add(next.input_tokens_committed);
    quantum.finished = next.finished;
    quantum.logprobs.extend(next.logprobs);
}

fn canonical_chat_terminal_text(streamed_text: &str, terminal_text: String) -> String {
    if streamed_text.is_empty() {
        terminal_text
    } else {
        streamed_text.to_string()
    }
}

impl NativeExecutor {
    pub(super) fn suspend_chat_session(&self, session: &SessionKey) -> Result<Option<usize>> {
        let mut states = self
            .chat_decode_states
            .lock()
            .map_err(|_| Error::InferenceError("chat state mutex poisoned".into()))?;
        let Some(super::ExecutorStateSlot::Ready { state, .. }) = states.get(session) else {
            return Ok(None);
        };
        let checkpoint = state.state.replay_checkpoint()?;
        let tokens = checkpoint.replay_tokens();
        let mut suspended = self
            .suspended_chat_states
            .lock()
            .map_err(|_| Error::InferenceError("suspended chat mutex poisoned".into()))?;
        if suspended.contains_key(session) {
            return Err(Error::InferenceError(
                "duplicate suspended chat identity".into(),
            ));
        }
        suspended.insert(
            session.clone(),
            super::state::SuspendedChatDecode {
                variant: state.variant,
                checkpoint,
                last_tokens_generated: state.last_tokens_generated,
                stream_sequence: state.stream_sequence,
                streamed_text: state.streamed_text.clone(),
                logprobs: state.logprobs.clone(),
                stop: state.stop.clone(),
            },
        );
        // All CPU continuation state is installed before dropping GPU owners.
        states.remove(session);
        Ok(Some(tokens))
    }

    /// Whether `session` is a capacity replay that has not finished
    /// rebuilding its state. Its prefill quanta are replay spans, whatever the
    /// engine's prefill mode, and they run through the scalar handler.
    pub(super) fn chat_session_replaying(&self, session: &SessionKey) -> Result<bool> {
        if self
            .suspended_chat_states
            .lock()
            .map_err(|_| Error::InferenceError("suspended chat mutex poisoned".into()))?
            .contains_key(session)
        {
            return Ok(true);
        }
        let states = self
            .chat_decode_states
            .lock()
            .map_err(|_| Error::InferenceError("chat state mutex poisoned".into()))?;
        Ok(matches!(
            states.get(session),
            Some(super::ExecutorStateSlot::Ready { state, .. })
                if state.state.replay_tokens().is_some()
        ))
    }

    pub(super) fn chat_generation_config(request: &EngineCoreRequest) -> ChatGenerationConfig {
        request.chat_generation_config()
    }

    pub(super) fn chat_request_seed(request_id: &str) -> u64 {
        EngineCoreRequest::chat_request_seed(request_id)
    }

    fn chat_messages(request: &EngineCoreRequest) -> Result<&[ChatMessage]> {
        request
            .chat_messages
            .as_deref()
            .ok_or_else(|| Error::InvalidInput("Chat request missing messages".to_string()))
    }

    pub(super) fn chat_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
    ) -> Result<ModelSessionResult> {
        self.chat_request_with_managed_cache(request, scheduled, None, None, None)
    }

    pub(super) fn chat_request_with_managed_cache(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        mut managed_cache: Option<PhysicalPagedKvCache>,
        mut mtp_cache: Option<PhysicalPagedKvCache>,
        tensor_reservation: Option<crate::engine::ManagedTensorStateReservation>,
    ) -> Result<ModelSessionResult> {
        if managed_cache.is_none() && mtp_cache.is_some() {
            return Err(Error::InferenceError(
                "managed Qwen3.8 MTP cache has no target-cache authority".into(),
            ));
        }
        if request.managed_cache_runtime().is_some() != managed_cache.is_some() {
            return Err(Error::InferenceError(
                "managed Qwen3 execution requires its exact row reservation".to_string(),
            ));
        }
        let model = request.prepared_chat_model_for_executor()?;
        let resumable_prefill = (self.config.enable_chunked_prefill
            || super::binds_mixed_step(request))
            && model.supports_resumable_prefill();
        let replaying =
            scheduled.is_prefill && self.chat_session_replaying(&scheduled.session_key())?;
        if managed_cache.is_some()
            && scheduled.is_prefill
            && !resumable_prefill
            && !replaying
            && (scheduled.num_computed_tokens != 0
                || scheduled.num_tokens != request.num_prompt_tokens())
        {
            return Err(Error::InvalidInput(
                "managed Qwen3 chat requires one full-prompt prefill quantum".to_string(),
            ));
        }
        let tensor_arena = request
            .managed_cache_runtime()
            .and_then(|runtime| runtime.tensor_state().cloned());
        if tensor_arena.is_some() != tensor_reservation.is_some() {
            return Err(Error::InferenceError(
                "managed chat tensor state requires its exact row reservation".into(),
            ));
        }
        let prepared_chat_prompt = request.prepared_chat_prompt_for_executor()?;
        let variant = Self::resolve_variant(request)?;
        let messages = Self::chat_messages(request)?;
        let max_new_tokens = request.params.max_tokens.max(1);
        let stream_tx = Self::stream_sender(request);
        let stream_policy = request.stream_policy;
        let generation_config = Self::chat_generation_config(request);
        let session = scheduled.session_key();
        if mtp_cache.is_some() && !matches!(model.as_ref(), NativeChatModel::Qwen38(_)) {
            return Err(Error::InferenceError(
                "managed Qwen3.8 MTP cache was routed to another model family".into(),
            ));
        }
        // Prefill scheduling can happen after preemption; only recover state
        // owned by this exact request incarnation. The in-flight marker remains
        // visible to cleanup until this quantum is committed or released.
        let mut state_lease = ExecutorStateLease::checkout(
            &self.chat_decode_states,
            session.clone(),
            variant,
            "chat decode",
        )?;
        if state_lease
            .state()
            .map(|state| state.variant != variant)
            .unwrap_or(false)
        {
            state_lease.discard_state();
        }

        let mut started_replay = false;
        if state_lease.state().is_none() {
            let mut suspended = self
                .suspended_chat_states
                .lock()
                .map_err(|_| Error::InferenceError("suspended chat mutex poisoned".into()))?;
            if let Some(saved) = suspended.get(&session) {
                if !scheduled.is_prefill
                    || scheduled.num_computed_tokens != 0
                    || saved.variant != variant
                {
                    return Err(Error::InferenceError(
                        "invalid chat replay restart boundary".into(),
                    ));
                }
                let cache = managed_cache
                    .take()
                    .ok_or_else(|| Error::InferenceError("replay lost target cache".into()))?;
                let mut state =
                    model.begin_replay_state_managed(&saved.checkpoint, cache, mtp_cache.take())?;
                if let Some(reservation) = tensor_reservation.as_ref() {
                    state.bind_hybrid_tensor_sequence(reservation.sequence)?;
                }
                state_lease.install_state(ActiveChatDecode {
                    variant,
                    state,
                    last_tokens_generated: saved.last_tokens_generated,
                    stream_sequence: saved.stream_sequence,
                    streamed_text: saved.streamed_text.clone(),
                    logprobs: saved.logprobs.clone(),
                    stop: saved.stop.clone(),
                })?;
                suspended.remove(&session);
                started_replay = true;
            }
        }

        if state_lease.state().is_some() {
            match managed_cache.take() {
                Some(cache) => {
                    state_lease.mark_dirty();
                    state_lease
                        .require_state_mut()?
                        .state
                        .install_managed_reservations(cache, mtp_cache.take())?;
                }
                None if !started_replay
                    && state_lease
                        .state()
                        .is_some_and(|state| state.state.uses_managed_kv()) =>
                {
                    return Err(Error::InferenceError(
                        "managed chat session lost its physical cache authority".to_string(),
                    ))
                }
                None => {}
            }
            if let (Some(arena), Some(reservation)) = (tensor_arena.as_ref(), tensor_reservation) {
                state_lease.mark_dirty();
                state_lease
                    .require_state_mut()?
                    .state
                    .bind_hybrid_tensor_sequence(reservation.sequence)?;
                if !started_replay {
                    state_lease
                        .require_state_mut()?
                        .state
                        .restore_hybrid_tensor_state(arena)?;
                }
            }
        } else {
            if request.is_cancelled() {
                state_lease.release()?;
                return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    request.id.clone(),
                )));
            }
            if scheduled.is_prefill
                && resumable_prefill
                && scheduled.num_computed_tokens > 0
                && !begins_resumable_prefill_state(scheduled, resumable_prefill, request)
            {
                return Err(Error::InferenceError(format!(
                    "resumable prefill request {} lost its decode state before span continuation; retry requires a fresh prompt",
                    request.id
                )));
            }
            let mut decode_state = match managed_cache.take() {
                Some(cache)
                    if begins_resumable_prefill_state(scheduled, resumable_prefill, request) =>
                {
                    Self::run_blocking(|| {
                        model.start_resumable_prefill_state_managed(
                            messages,
                            max_new_tokens,
                            &generation_config,
                            prepared_chat_prompt,
                            &request.prompt_tokens,
                            cache,
                            mtp_cache.take(),
                        )
                    })?
                }
                // Each family names its own managed start, so a new chat
                // family fails to compile here instead of falling through
                // to another family's.
                Some(cache) => match model.as_ref() {
                    NativeChatModel::Qwen3(_) => Self::run_blocking(|| {
                        model.start_qwen3_decode_state_managed(
                            &request.prompt_tokens,
                            max_new_tokens,
                            &generation_config,
                            cache,
                        )
                    })?,
                    NativeChatModel::Qwen35(_) => Self::run_blocking(|| {
                        model.start_qwen35_decode_state_managed(
                            messages,
                            max_new_tokens,
                            &generation_config,
                            prepared_chat_prompt.and_then(|prepared| prepared.as_qwen35()),
                            cache,
                        )
                    })?,
                    NativeChatModel::Qwen36Moe(_) => Self::run_blocking(|| {
                        model.start_qwen36_moe_decode_state_managed(
                            messages,
                            max_new_tokens,
                            &generation_config,
                            prepared_chat_prompt.and_then(|prepared| prepared.as_qwen36_moe()),
                            cache,
                        )
                    })?,
                    NativeChatModel::Qwen38(_) => Self::run_blocking(|| {
                        model.start_qwen38_decode_state_managed(
                            messages,
                            max_new_tokens,
                            &generation_config,
                            prepared_chat_prompt.and_then(|prepared| prepared.as_qwen38()),
                            cache,
                            mtp_cache.take(),
                        )
                    })?,
                    NativeChatModel::Gemma3(_) => Self::run_blocking(|| {
                        model.start_gemma3_decode_state_managed(
                            &request.prompt_tokens,
                            max_new_tokens,
                            &generation_config,
                            cache,
                        )
                    })?,
                    NativeChatModel::Lfm2(_) => Self::run_blocking(|| {
                        model.start_lfm2_decode_state_managed(
                            &request.prompt_tokens,
                            max_new_tokens,
                            &generation_config,
                            cache,
                        )
                    })?,
                },
                None => {
                    return Err(Error::InferenceError(
                        "incremental chat execution requires scheduler-owned physical state".into(),
                    ))
                }
            };
            if let Some(reservation) = tensor_reservation {
                decode_state.bind_hybrid_tensor_sequence(reservation.sequence)?;
                // Only a resumable begin starts above zero on an attached
                // prefix: adopt the forked hybrid tensor state before the
                // first span. A monolithic begin already consumed its prompt
                // inside the begin call, and an empty fresh sequence restores
                // nothing, so both must skip the restore.
                if begins_resumable_prefill_state(scheduled, resumable_prefill, request) {
                    if let Some(arena) = tensor_arena.as_ref() {
                        decode_state.restore_hybrid_tensor_state(arena)?;
                    }
                }
            }
            state_lease.install_state(ActiveChatDecode {
                variant,
                state: decode_state,
                last_tokens_generated: 0,
                stream_sequence: 0,
                streamed_text: String::new(),
                logprobs: Vec::new(),
                stop: StopSequenceFilter::new(&request.params.stop_sequences),
            })?;
        }

        let input_budget = if scheduled.is_prefill {
            1
        } else {
            scheduled.num_tokens.max(1)
        };
        let mut total_tokens_generated = 0usize;
        if request.is_cancelled() {
            state_lease.release()?;
            return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                request.id.clone(),
            )));
        }
        let resumable_prefill_quantum = scheduled.is_prefill && (resumable_prefill || replaying);
        let resumable_span_tokens = resumable_prefill_quantum.then_some(scheduled.num_tokens);
        let replay_tokens = state_lease
            .state()
            .and_then(|active| active.state.replay_tokens());
        let resumable_span = resumable_prefill_quantum
            .then(|| {
                resumable_prefill_span(
                    scheduled,
                    replay_tokens.unwrap_or(request.num_prompt_tokens()),
                )
            })
            .transpose()?;
        state_lease.mark_dirty();
        let (step, final_text, finished, managed_cache_completions, terminal_logprobs) = {
            let active_state = state_lease.require_state_mut()?;
            if matches!(active_state.state, NativeChatDecodeState::Lfm2(_))
                && crate::models::architectures::lfm2::diagnostics::enabled()
            {
                tracing::info!(request_id = %request.id, model = ?request.model_variant,
                    prompt_tokens = request.num_prompt_tokens(), max_new_tokens = request.params.max_tokens,
                    prefill = scheduled.is_prefill, "LFM2 quantum diagnostic context");
            }
            // A stop-string quantum has its stop strings applied token by
            // token below and carries the logprob entries to stream.
            let mut stopped_logprobs = None;
            let mut step = if let (Some(_), Some((start, end))) = (replay_tokens, resumable_span) {
                Self::run_blocking(|| {
                    model.continue_replay(
                        &mut active_state.state,
                        &request.prompt_tokens,
                        prepared_chat_prompt,
                        start,
                        end,
                    )
                })?;
                crate::engine::metrics::record_capacity_replay(end - start);
                NativeChatDecodeStep {
                    delta: String::new(),
                    text: active_state.streamed_text.clone(),
                    tokens_generated: active_state.last_tokens_generated,
                    input_tokens_committed: 0,
                    finished: false,
                    logprobs: Vec::new(),
                }
            } else if let Some((scheduled_start, span_end)) = resumable_span {
                // A managed prefix attach leaves the state's logical prefill
                // cursor at the attached physical cursor, which can sit above
                // the scheduler's span start. The model always feeds from its
                // own cursor; the scheduler still receives the scheduled span
                // length as progress.
                let span_start = active_state
                    .state
                    .prefill_progress()
                    .unwrap_or(scheduled_start)
                    .max(scheduled_start);
                let prefill_complete = Self::run_blocking(|| {
                    model.continue_resumable_prefill(
                        &mut active_state.state,
                        messages,
                        &generation_config,
                        prepared_chat_prompt,
                        &request.prompt_tokens,
                        span_start,
                        span_end,
                        request.num_prompt_tokens(),
                    )
                })?;
                finish_resumable_prefill_step(
                    prefill_complete,
                    active_state.last_tokens_generated,
                    || Self::run_blocking(|| model.decode_quantum(&mut active_state.state, 1)),
                )?
            } else if active_state.stop.is_some() && input_budget > 1 {
                // Stop strings are checked after every token, so the
                // quantum ends at a stop string like an end-of-sequence
                // token: no token past it is sampled, counted or reported.
                let mut quantum: Option<NativeChatDecodeStep> = None;
                let mut stream_logprobs = Vec::new();
                for _ in 0..input_budget {
                    let mut token =
                        Self::run_blocking(|| model.decode_quantum(&mut active_state.state, 1))?;
                    stream_logprobs.extend(active_state.apply_stop_sequences(&mut token));
                    match quantum.as_mut() {
                        Some(quantum) => append_decode_step(quantum, token),
                        None => quantum = Some(token),
                    }
                    if quantum.as_ref().is_some_and(|quantum| quantum.finished) {
                        break;
                    }
                }
                stopped_logprobs = Some(stream_logprobs);
                quantum.expect("a quantum decodes at least one token")
            } else {
                Self::run_blocking(|| model.decode_quantum(&mut active_state.state, input_budget))?
            };
            if request.is_cancelled() {
                return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    request.id.clone(),
                )));
            }

            record_lfm_completion(&active_state.state, request, &step);
            let stream_logprobs = match stopped_logprobs {
                Some(stream_logprobs) => stream_logprobs,
                None => active_state.apply_stop_sequences(&mut step),
            };
            let step_tokens_generated = step
                .tokens_generated
                .saturating_sub(active_state.last_tokens_generated);
            active_state.last_tokens_generated = step.tokens_generated;
            total_tokens_generated = total_tokens_generated.saturating_add(step_tokens_generated);
            let mut final_text = step.text.clone();
            let finished = step.finished;

            // Durable state must be fully staged before any externally visible
            // token is emitted. A staging failure then remains an atomic row
            // failure instead of leaking text from an uncommittable quantum.
            if let Some(arena) = tensor_arena.as_ref() {
                active_state
                    .state
                    .stage_hybrid_tensor_state(arena, scheduled.plan_id)?;
            }

            let terminal_logprobs = active_state.record_step_logprobs(&step.logprobs, finished);
            if let Some(tx) = stream_tx.as_ref() {
                // A step whose text a stop string discarded still publishes
                // its logprob entries, so the stream carries the terminal
                // list whole.
                if !step.delta.is_empty() || !stream_logprobs.is_empty() {
                    Self::stream_chat_text_with_policy(
                        tx,
                        stream_policy,
                        &request.id,
                        &mut active_state.stream_sequence,
                        step.delta.clone(),
                        stream_logprobs,
                    )?;
                    active_state.streamed_text.push_str(&step.delta);
                }
                if step.finished {
                    Self::stream_final_marker_with_policy(
                        tx,
                        stream_policy,
                        &request.id,
                        &mut active_state.stream_sequence,
                    )?;
                    final_text =
                        canonical_chat_terminal_text(&active_state.streamed_text, final_text);
                }
            }
            let managed_cache_completions = active_state.state.take_managed_write_completions();
            (
                step,
                final_text,
                finished,
                managed_cache_completions,
                terminal_logprobs,
            )
        };

        let tokens_processed = if let Some(span_tokens) = resumable_span_tokens {
            span_tokens
        } else if scheduled.is_prefill {
            request.num_prompt_tokens()
        } else {
            step.input_tokens_committed
        };
        if finished {
            state_lease.release()?;
        } else {
            state_lease.restore()?;
        }

        Ok(ModelSessionResult::sequence(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(AudioOutput::empty(24_000)),
            text: Some(final_text),
            input_transcription: None,
            tokens_processed,
            tokens_generated: total_tokens_generated,
            finished,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
            logprobs: terminal_logprobs,
        })
        .with_managed_cache_completions(managed_cache_completions))
    }

    pub(super) fn chat_decode_batch_with_managed(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        managed_caches: Vec<Option<super::RetainedRowManagedState>>,
    ) -> Result<Vec<ModelSessionResult>> {
        if scheduled.is_empty()
            || scheduled
                .iter()
                .any(|scheduled| scheduled.is_prefill || scheduled.num_tokens < 1)
        {
            return Err(Error::InvalidInput(
                "continuous chat execution requires a positive decode token count per row".to_string(),
            ));
        }
        // DS9.4: a speculative envelope is homogeneous — every row carries the
        // same multi-token quantum. Mixed widths cannot arise because the
        // quantum gates are per-model and per-step global.
        let speculative_width = {
            let width = scheduled[0].num_tokens;
            if width > 1
                && scheduled
                    .iter()
                    .any(|scheduled| scheduled.num_tokens != width)
            {
                return Err(Error::InvalidInput(
                    "continuous chat execution requires homogeneous speculative quanta".to_string(),
                ));
            }
            (width > 1).then_some(width)
        };
        if managed_caches.len() != scheduled.len() {
            return Err(Error::InvalidInput(
                "continuous chat managed-cache rows do not match batch width".to_string(),
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
                            "continuous chat request {} is missing its snapshot",
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
                        Error::InferenceError("cancelled chat row produced no result".into())
                    })
                })
                .collect();
        }

        let model = ordered_requests[live_indices[0]].prepared_chat_model_for_executor()?;
        if !model.supports_continuous_decode_batch() {
            return Err(Error::InvalidInput(
                "loaded chat model has no continuous tensor decode adapter".to_string(),
            ));
        }
        for index in live_indices.iter().copied().skip(1) {
            let request = ordered_requests[index];
            let row_model = request.prepared_chat_model_for_executor()?;
            if !Arc::ptr_eq(&model, &row_model) {
                return Err(Error::InferenceError(
                    "continuous chat batch spans different loaded model instances".to_string(),
                ));
            }
        }

        let mut checked_out_states = Vec::with_capacity(live_indices.len());
        for index in live_indices.iter().copied() {
            let request = ordered_requests[index];
            let session = scheduled[index].session_key();
            let expected_variant = Self::resolve_variant(request)?;
            let lease = ExecutorStateLease::checkout(
                &self.chat_decode_states,
                session.clone(),
                expected_variant,
                "continuous chat decode",
            )?;
            let state = lease.state().ok_or_else(|| {
                Error::InferenceError(format!(
                    "continuous chat session {}:{} has no active decode state",
                    session.request_id, session.epoch
                ))
            })?;
            if state.variant != expected_variant {
                return Err(Error::InferenceError(
                    "continuous chat state variant does not match its request".to_string(),
                ));
            }
            checked_out_states.push((index, session, lease));
        }
        let mut active_states = ContinuousChatStateBatch::new(checked_out_states);
        let mut managed_caches = managed_caches;

        for (index, _, lease, checkpoint) in &mut active_states.rows {
            let request = ordered_requests[*index];
            let managed_cache = managed_caches[*index].take();
            if request.managed_cache_runtime().is_some() != managed_cache.is_some() {
                return Err(Error::InferenceError(
                    "continuous managed Qwen3 row lost its reservation".to_string(),
                ));
            }
            match managed_cache {
                Some(mut views) => {
                    let tensor_reservation = views.tensor_state.clone();
                    let (cache, mtp_cache) = Self::continuous_row_caches(request, &mut views)?;
                    let tensor_arena = request
                        .managed_cache_runtime()
                        .and_then(|runtime| runtime.tensor_state());
                    if tensor_arena.is_some() != tensor_reservation.is_some() {
                        return Err(Error::InferenceError(
                            "continuous hybrid row lost its tensor-state reservation".into(),
                        ));
                    }
                    lease.mark_dirty();
                    let active_state = lease.require_state_mut()?;
                    if let (Some(arena), Some(reservation)) = (tensor_arena, tensor_reservation) {
                        active_state
                            .state
                            .bind_hybrid_tensor_sequence(reservation.sequence)?;
                        active_state.state.restore_hybrid_tensor_state(arena)?;
                    }
                    *checkpoint = Some(
                        active_state
                            .state
                            .begin_continuous_quantum(cache, mtp_cache)?,
                    );
                }
                None if lease
                    .state()
                    .is_some_and(|state| state.state.uses_managed_kv()) =>
                {
                    return Err(Error::InferenceError(
                        "continuous chat row lost its managed-cache reservation".to_string(),
                    ))
                }
                None => {}
            }
        }

        for (_, _, lease, _) in &mut active_states.rows {
            lease.mark_dirty();
        }
        if matches!(model.as_ref(), NativeChatModel::Lfm2(_))
            && crate::models::architectures::lfm2::diagnostics::enabled()
        {
            for (row, (index, _, _, _)) in active_states.rows.iter().enumerate() {
                let request = ordered_requests[*index];
                tracing::info!(request_id = %request.id, row, prompt_tokens = request.num_prompt_tokens(),
                    max_new_tokens = request.params.max_tokens, "LFM2 batch diagnostic row");
            }
        }
        let mut state_refs = active_states
            .rows
            .iter_mut()
            .map(|(_, _, lease, _)| lease.require_state_mut().map(|state| &mut state.state))
            .collect::<Result<Vec<_>>>()?;
        let live_width = state_refs.len();
        let steps = match speculative_width {
            Some(budget) => {
                if !matches!(
                    model.as_ref(),
                    NativeChatModel::Qwen38(_) | NativeChatModel::Qwen36Moe(_)
                ) {
                    return Err(Error::InvalidInput(
                        "speculative envelopes require an MTP speculative model".to_string(),
                    ));
                }
                Self::run_blocking(|| {
                    model.decode_speculative_batch(&mut state_refs, budget)
                })?
            }
            None => Self::run_blocking(|| model.decode_step_batch(&mut state_refs))?,
        };
        drop(state_refs);
        if steps.len() != active_states.rows.len() {
            return Err(Error::InferenceError(
                "continuous chat model returned the wrong number of rows".to_string(),
            ));
        }
        let model_call = if model.continuous_decode_is_tensor_batched() {
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

        for (index, _, lease, _) in &mut active_states.rows {
            if let Some(arena) = ordered_requests[*index]
                .managed_cache_runtime()
                .and_then(|runtime| runtime.tensor_state())
            {
                let active_state = lease.require_state_mut()?;
                active_state
                    .state
                    .stage_hybrid_tensor_state(arena, scheduled[*index].plan_id)?;
            }
        }

        let mut continuing = vec![false; scheduled.len()];
        for ((index, _, lease, _), step) in active_states.rows.iter_mut().zip(steps) {
            let (output, continues) = Self::publish_shared_row_step(
                ordered_requests[*index],
                lease.require_state_mut()?,
                step,
                1,
            );
            outputs[*index] = Some(output);
            continuing[*index] = continues;
        }

        let committed_states = active_states.commit();
        for (index, _, lease) in committed_states {
            let transition = if continuing[index] {
                lease.restore()
            } else {
                lease.release()
            };
            if let Err(error) = transition {
                outputs[index] = Some(ModelSessionResult::sequence(ExecutorOutput::error(
                    ordered_requests[index].id.clone(),
                    format!("continuous chat state transition failed: {error}"),
                )));
            }
        }
        outputs
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| {
                    Error::InferenceError("continuous chat row produced no result".into())
                })
            })
            .collect()
    }

    /// The target cache, and the MTP cache of a family with one, of a
    /// continuous chat row's managed views.
    fn continuous_row_caches(
        request: &EngineCoreRequest,
        views: &mut super::RetainedRowManagedState,
    ) -> Result<(PhysicalPagedKvCache, Option<PhysicalPagedKvCache>)> {
        let family = request.model_variant.map(|variant| variant.family());
        let (target, mtp, name) = if family == Some(crate::catalog::ModelFamily::Qwen38Chat) {
            (
                super::QWEN38_TARGET_ATTENTION_DOMAIN,
                super::QWEN38_MTP_ATTENTION_DOMAIN,
                "Qwen3.8",
            )
        } else if family == Some(crate::catalog::ModelFamily::Qwen35MoeChat) {
            (
                super::QWEN36_MOE_TARGET_ATTENTION_DOMAIN,
                super::QWEN36_MOE_MTP_ATTENTION_DOMAIN,
                "Qwen3.6-MoE",
            )
        } else {
            return Ok((views.take_only_paged()?, None));
        };
        let target = views.take_paged_domain(target, true)?;
        let mtp = views.take_paged_domain(mtp, false)?;
        views.ensure_all_paged_consumed()?;
        let target = target.ok_or_else(|| {
            Error::InferenceError(format!("continuous {name} row lost its target cache"))
        })?;
        Ok((target, mtp))
    }

    /// Publish one row's step of a shared model call: stop strings,
    /// logprobs, stream deltas and the final marker, then the row's output.
    /// Returns the output and whether the row continues.
    fn publish_shared_row_step(
        request: &EngineCoreRequest,
        active_state: &mut ActiveChatDecode,
        mut step: NativeChatDecodeStep,
        tokens_processed: usize,
    ) -> (ModelSessionResult, bool) {
        record_lfm_completion(&active_state.state, request, &step);
        let stream_logprobs = active_state.apply_stop_sequences(&mut step);
        let step_tokens_generated = step
            .tokens_generated
            .saturating_sub(active_state.last_tokens_generated);
        active_state.last_tokens_generated = step.tokens_generated;
        // Recorded before the stream branch: a non-streamed row has no
        // sender, and its terminal output must still carry every entry.
        let terminal_logprobs = active_state.record_step_logprobs(&step.logprobs, step.finished);
        let stream_result = (|| -> Result<()> {
            let Some(tx) = Self::stream_sender(request) else {
                return Ok(());
            };
            if !step.delta.is_empty() || !stream_logprobs.is_empty() {
                Self::stream_chat_text_with_policy(
                    &tx,
                    request.stream_policy,
                    &request.id,
                    &mut active_state.stream_sequence,
                    step.delta.clone(),
                    stream_logprobs,
                )?;
                active_state.streamed_text.push_str(&step.delta);
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
        if let Err(error) = stream_result {
            return (
                ModelSessionResult::sequence(ExecutorOutput::error(
                    request.id.clone(),
                    format!("continuous chat stream staging failed: {error}"),
                )),
                false,
            );
        }

        let managed_cache_completions = active_state.state.take_managed_write_completions();
        let output = ModelSessionResult::sequence(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(AudioOutput::empty(24_000)),
            text: Some(if step.finished {
                canonical_chat_terminal_text(&active_state.streamed_text, step.text)
            } else {
                step.text
            }),
            input_transcription: None,
            tokens_processed,
            tokens_generated: step_tokens_generated,
            finished: step.finished,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
            logprobs: terminal_logprobs,
        })
        .with_managed_cache_completions(managed_cache_completions);
        (output, !step.finished)
    }

    /// Run one mixed physical batch of a chat model whose loaded family runs
    /// mixed steps: one-token decode rows first, then resumable prefill
    /// spans, in one model call and one sampling readback.
    ///
    /// Every row binds its managed views before the call. A row with state
    /// checkpoints its quantum, so a failed call rolls it back; a prefill row
    /// starting its first span begins its state first, as the scalar
    /// handler does. Each row then publishes like a continuous decode row: a
    /// non-final span reports its span and no token; a final span publishes
    /// its bootstrap token.
    pub(super) fn chat_mixed_step_with_managed(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        managed_caches: Vec<Option<super::RetainedRowManagedState>>,
    ) -> Result<Vec<ModelSessionResult>> {
        if scheduled.is_empty()
            || managed_caches.len() != scheduled.len()
            || scheduled
                .windows(2)
                .any(|pair| pair[0].is_prefill && !pair[1].is_prefill)
            || scheduled
                .iter()
                .any(|row| !row.is_prefill && row.num_tokens != 1)
        {
            return Err(Error::InvalidInput(
                "a mixed chat step takes one-token decode rows, then prefill spans, each with its managed views".into(),
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
                            "mixed chat request {} is missing its snapshot",
                            scheduled.request_id
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut outputs = ordered_requests
            .iter()
            .map(|request| {
                request.is_cancelled().then(|| {
                    ModelSessionResult::cancelled_before_dispatch(ExecutorOutput::cancelled(
                        request.id.clone(),
                    ))
                })
            })
            .collect::<Vec<_>>();
        let live_indices = (0..scheduled.len())
            .filter(|index| outputs[*index].is_none())
            .collect::<Vec<_>>();
        let Some(&first_live) = live_indices.first() else {
            return outputs
                .into_iter()
                .map(|output| {
                    output.ok_or_else(|| {
                        Error::InferenceError("cancelled chat row produced no result".into())
                    })
                })
                .collect();
        };
        let model = ordered_requests[first_live].prepared_chat_model_for_executor()?;
        if !model.supports_mixed_step() {
            return Err(Error::InvalidInput(
                "loaded chat model does not run mixed prefill and decode steps".into(),
            ));
        }
        for index in live_indices.iter().copied().skip(1) {
            let row_model = ordered_requests[index].prepared_chat_model_for_executor()?;
            if !Arc::ptr_eq(&model, &row_model) {
                return Err(Error::InferenceError(
                    "mixed chat batch spans different loaded model instances".into(),
                ));
            }
        }

        let mut checked_out = Vec::with_capacity(live_indices.len());
        for index in live_indices.iter().copied() {
            let request = ordered_requests[index];
            let session = scheduled[index].session_key();
            let variant = Self::resolve_variant(request)?;
            let lease = ExecutorStateLease::checkout(
                &self.chat_decode_states,
                session.clone(),
                variant,
                "mixed chat step",
            )?;
            if lease.state().is_some_and(|state| state.variant != variant) {
                return Err(Error::InferenceError(
                    "mixed chat state variant does not match its request".into(),
                ));
            }
            checked_out.push((index, session, lease));
        }
        let mut active_states = ContinuousChatStateBatch::new(checked_out);
        let mut managed_caches = managed_caches;
        for (index, session, lease, checkpoint) in &mut active_states.rows {
            let request = ordered_requests[*index];
            let row = &scheduled[*index];
            let mut views = managed_caches[*index].take().ok_or_else(|| {
                Error::InferenceError("mixed chat row has no managed-cache reservation".into())
            })?;
            let tensor_reservation = views.tensor_state.clone();
            let (cache, mtp_cache) = Self::continuous_row_caches(request, &mut views)?;
            let tensor_arena = request
                .managed_cache_runtime()
                .and_then(|runtime| runtime.tensor_state());
            if tensor_arena.is_some() != tensor_reservation.is_some() {
                return Err(Error::InferenceError(
                    "mixed hybrid row lost its tensor-state reservation".into(),
                ));
            }
            lease.mark_dirty();
            if lease.state().is_some() {
                let active_state = lease.require_state_mut()?;
                if let (Some(arena), Some(reservation)) = (tensor_arena, tensor_reservation) {
                    active_state
                        .state
                        .bind_hybrid_tensor_sequence(reservation.sequence)?;
                    active_state.state.restore_hybrid_tensor_state(arena)?;
                }
                *checkpoint = Some(
                    active_state
                        .state
                        .begin_continuous_quantum(cache, mtp_cache)?,
                );
                continue;
            }
            if !begins_resumable_prefill_state(row, true, request)
                || self
                    .suspended_chat_states
                    .lock()
                    .map_err(|_| Error::InferenceError("suspended chat mutex poisoned".into()))?
                    .contains_key(session)
            {
                return Err(Error::InferenceError(format!(
                    "mixed chat row {} has no decode state to continue",
                    request.id
                )));
            }
            let messages = Self::chat_messages(request)?;
            let generation_config = Self::chat_generation_config(request);
            let mut state = Self::run_blocking(|| {
                model.start_resumable_prefill_state_managed(
                    messages,
                    request.params.max_tokens.max(1),
                    &generation_config,
                    request.prepared_chat_prompt_for_executor()?,
                    &request.prompt_tokens,
                    cache,
                    mtp_cache,
                )
            })?;
            if let Some(reservation) = tensor_reservation {
                state.bind_hybrid_tensor_sequence(reservation.sequence)?;
                if let Some(arena) = tensor_arena {
                    state.restore_hybrid_tensor_state(arena)?;
                }
            }
            lease.install_state(ActiveChatDecode {
                variant: Self::resolve_variant(request)?,
                state,
                last_tokens_generated: 0,
                stream_sequence: 0,
                streamed_text: String::new(),
                logprobs: Vec::new(),
                stop: StopSequenceFilter::new(&request.params.stop_sequences),
            })?;
        }

        let mut spans = Vec::with_capacity(active_states.rows.len());
        for (index, _, lease, _) in &active_states.rows {
            let request = ordered_requests[*index];
            let row = &scheduled[*index];
            spans.push(if row.is_prefill {
                let (start, end) = resumable_prefill_span(row, request.num_prompt_tokens())?;
                // An attached prefix can leave the state's cursor above the
                // scheduled start; the model feeds from its own cursor.
                let start = lease
                    .state()
                    .and_then(|active| active.state.prefill_progress())
                    .unwrap_or(start)
                    .max(start);
                Some((start, end))
            } else {
                None
            });
        }
        let mut rows = Vec::with_capacity(active_states.rows.len());
        for ((index, _, lease, _), span) in active_states.rows.iter_mut().zip(&spans) {
            let request = ordered_requests[*index];
            let input = match *span {
                Some((span_start, span_end)) => MixedInput::Prefill {
                    prompt: &request.prompt_tokens,
                    prepared: request.prepared_chat_prompt_for_executor()?,
                    span_start,
                    span_end,
                },
                None => MixedInput::Decode,
            };
            rows.push(MixedRow {
                state: &mut lease.require_state_mut()?.state,
                input,
            });
        }
        let live_width = rows.len();
        let steps = Self::run_blocking(|| model.step_mixed(&mut rows))?;
        drop(rows);
        if steps.len() != live_width {
            return Err(Error::InferenceError(
                "mixed chat model returned the wrong number of rows".into(),
            ));
        }
        crate::engine::metrics::record_engine_model_call(
            crate::engine::metrics::EngineModelCall::NativeTensor {
                mode: crate::engine::NativeBatchMode::Mixed,
                rows: live_width,
            },
        );

        for (index, _, lease, _) in &mut active_states.rows {
            if let Some(arena) = ordered_requests[*index]
                .managed_cache_runtime()
                .and_then(|runtime| runtime.tensor_state())
            {
                lease
                    .require_state_mut()?
                    .state
                    .stage_hybrid_tensor_state(arena, scheduled[*index].plan_id)?;
            }
        }

        let mut continuing = vec![false; scheduled.len()];
        for (((index, _, lease, _), step), span) in
            active_states.rows.iter_mut().zip(steps).zip(&spans)
        {
            let row = &scheduled[*index];
            let active_state = lease.require_state_mut()?;
            let (step, tokens_processed) = match step {
                // The model reports the span it fed, which an attached prefix
                // can shorten; the scheduler receives the scheduled span.
                MixedRowStep::Progress { input_tokens } => {
                    if span.is_none_or(|(start, end)| input_tokens != end - start) {
                        return Err(Error::InferenceError(
                            "mixed chat row reported progress that is not its prefill span".into(),
                        ));
                    }
                    let step = NativeChatDecodeStep {
                        delta: String::new(),
                        text: String::new(),
                        tokens_generated: active_state.last_tokens_generated,
                        input_tokens_committed: 0,
                        finished: false,
                        logprobs: Vec::new(),
                    };
                    (step, row.num_tokens)
                }
                MixedRowStep::Token(step) if row.is_prefill => (step, row.num_tokens),
                MixedRowStep::Token(step) => {
                    let tokens = step.input_tokens_committed;
                    (step, tokens)
                }
            };
            let (output, continues) = Self::publish_shared_row_step(
                ordered_requests[*index],
                active_state,
                step,
                tokens_processed,
            );
            outputs[*index] = Some(output);
            continuing[*index] = continues;
        }

        for (index, _, lease) in active_states.commit() {
            let transition = if continuing[index] {
                lease.restore()
            } else {
                lease.release()
            };
            if let Err(error) = transition {
                outputs[index] = Some(ModelSessionResult::sequence(ExecutorOutput::error(
                    ordered_requests[index].id.clone(),
                    format!("mixed chat state transition failed: {error}"),
                )));
            }
        }
        outputs
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| {
                    Error::InferenceError("mixed chat row produced no result".into())
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{GenerationParams, InputRange, SequencePhase, WorkUnit};
    use crate::model::ModelVariant;
    use crate::models::registry::NativeChatPreparedPrompt;
    use crate::models::shared::chat::{ChatMessage, ChatRole};

    #[test]
    fn final_first_span_still_bootstraps_resumable_prefill_state() {
        let scheduled = ScheduledRequest {
            plan_id: 1,
            request_id: "short-resumable-prompt".to_string(),
            sequence_id: 1,
            num_tokens: 16,
            is_prefill: true,
            num_computed_tokens: 0,
            work: WorkUnit::SequenceStep {
                phase: SequencePhase::Prefill,
                input: InputRange { start: 0, end: 16 },
                max_output_steps: 16,
                auxiliary_state: None,
            },
        };

        let request = EngineCoreRequest::tts("resumable-prefill-state");
        assert!(begins_resumable_prefill_state(&scheduled, true, &request));
        assert!(!begins_resumable_prefill_state(&scheduled, false, &request));
        assert_eq!(resumable_prefill_span(&scheduled, 16).unwrap(), (0, 16));
    }

    #[test]
    fn resumable_prefill_rejects_scheduler_work_cursor_mismatch() {
        let scheduled = ScheduledRequest {
            plan_id: 1,
            request_id: "bad-resumable-span".to_string(),
            sequence_id: 1,
            num_tokens: 8,
            is_prefill: true,
            num_computed_tokens: 8,
            work: WorkUnit::SequenceStep {
                phase: SequencePhase::Prefill,
                input: InputRange { start: 7, end: 16 },
                max_output_steps: 8,
                auxiliary_state: None,
            },
        };

        assert!(resumable_prefill_span(&scheduled, 16).is_err());
    }

    #[test]
    fn final_resumable_prefill_span_publishes_its_bootstrap_token() {
        let published = std::cell::Cell::new(false);
        let step = finish_resumable_prefill_step(true, 0, || {
            published.set(true);
            Ok(NativeChatDecodeStep {
                delta: "token".to_string(),
                text: "token".to_string(),
                tokens_generated: 1,
                input_tokens_committed: 0,
                finished: false,
                logprobs: Vec::new(),
            })
        })
        .unwrap();

        assert!(published.get());
        assert_eq!(step.delta, "token");
        assert_eq!(step.tokens_generated, 1);
        assert_eq!(step.input_tokens_committed, 0);
    }

    #[test]
    fn incomplete_resumable_prefill_span_does_not_publish_a_token() {
        let published = std::cell::Cell::new(false);
        let step = finish_resumable_prefill_step(false, 3, || {
            published.set(true);
            unreachable!("an incomplete prefill cannot publish its bootstrap")
        })
        .unwrap();

        assert!(!published.get());
        assert_eq!(step.tokens_generated, 3);
        assert!(step.delta.is_empty());
        assert_eq!(step.input_tokens_committed, 0);
    }

    #[test]
    fn chat_handler_rejects_unprepared_public_prompt_tokens() {
        let executor = NativeExecutor::new(super::super::WorkerConfig::default());
        let mut request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "hello".to_string(),
        }])
        .with_model_variant(ModelVariant::Qwen306B);
        request.prompt_tokens = vec![1, 2, 3];
        let scheduled = ScheduledRequest {
            plan_id: 1,
            request_id: request.id.clone(),
            sequence_id: 1,
            num_tokens: 3,
            is_prefill: true,
            num_computed_tokens: 0,
            work: WorkUnit::SequenceStep {
                phase: SequencePhase::Prefill,
                input: InputRange { start: 0, end: 3 },
                max_output_steps: 1,
                auxiliary_state: None,
            },
        };

        let error = executor
            .chat_request(&request, &scheduled)
            .expect_err("chat execution must require the private preparation marker");
        assert!(error
            .to_string()
            .contains("missing exact model prompt preparation"));

        request
            .install_chat_execution_preparation(ModelVariant::Qwen306B, vec![1, 2, 3], None, 4096)
            .unwrap();
        request.prompt_tokens[0] = 99;
        let error = executor
            .chat_request(&request, &scheduled)
            .expect_err("mutated preparation must fail before model lookup");
        assert!(error
            .to_string()
            .contains("changed after exact prompt preparation"));
    }

    #[test]
    fn chat_generation_config_preserves_request_sampling_controls() {
        let mut request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "hello".to_string(),
        }]);
        request.id = "req-sampling".to_string();
        request.params = GenerationParams {
            temperature: 0.85,
            top_p: 0.92,
            top_k: 40,
            repetition_penalty: 1.2,
            presence_penalty: 1.5,
            stop_token_ids: vec![7, 9],
            ..GenerationParams::default()
        };

        let config = NativeExecutor::chat_generation_config(&request);
        assert_eq!(config.temperature, 0.85);
        assert_eq!(config.top_p, 0.92);
        assert_eq!(config.top_k, 40);
        assert_eq!(config.repetition_penalty, 1.2);
        assert_eq!(config.presence_penalty, 1.5);
        assert_eq!(config.stop_token_ids, vec![7, 9]);
        assert_eq!(
            config.seed,
            NativeExecutor::chat_request_seed("req-sampling")
        );
        assert_eq!(config.request, request.chat_config);
    }

    /// EXE-F9: a caller's seed, not the request id, seeds sampling, so the
    /// same seed reproduces across requests. Seed 0 must not reach the
    /// samplers as 0, which they treat as "seed from the clock".
    #[test]
    fn chat_generation_config_honours_a_request_seed() {
        let seeded = |id: &str, seed: Option<u64>| {
            let mut request = EngineCoreRequest::chat(vec![ChatMessage {
                role: ChatRole::User,
                content: "hello".to_string(),
            }]);
            request.id = id.to_string();
            request.params.seed = seed;
            NativeExecutor::chat_generation_config(&request).seed
        };
        assert_eq!(seeded("req-a", Some(42)), seeded("req-b", Some(42)));
        assert_ne!(seeded("req-a", Some(42)), seeded("req-a", Some(43)));
        assert_ne!(seeded("req-a", Some(0)), 0);
        assert_ne!(seeded("req-a", Some(0)), seeded("req-a", Some(1)));
        assert_eq!(
            seeded("req-a", None),
            NativeExecutor::chat_request_seed("req-a")
        );
    }

    #[test]
    fn chat_request_seed_is_stable_for_same_request_id() {
        let first = NativeExecutor::chat_request_seed("req-123");
        let second = NativeExecutor::chat_request_seed("req-123");
        let other = NativeExecutor::chat_request_seed("req-456");

        assert_eq!(first, second);
        assert_ne!(first, other);
    }

    #[test]
    fn visible_chat_deltas_are_the_canonical_terminal_text() {
        assert_eq!(
            canonical_chat_terminal_text(" raw visible text ", "raw visible text".to_string()),
            " raw visible text "
        );
        assert_eq!(
            canonical_chat_terminal_text("prefix replacement", "prefix rewritten".to_string()),
            "prefix replacement"
        );
        assert_eq!(
            canonical_chat_terminal_text("", "terminal-only".to_string()),
            "terminal-only"
        );
    }

    #[test]
    fn hybrid_handler_suspension_replays_physical_receipts_without_republishing() {
        use crate::backends::BackendKind;
        use crate::engine::cache::managed::ManagedKvCacheManager;
        use crate::engine::ModelInstanceId;
        use crate::kv::InferenceStateContractProvider;
        use crate::models::architectures::qwen38::chat::recovery_tests::{
            model_fixture, prepared_prompt,
        };
        use crate::models::registry::{ChatModelLease, NativeChatPreparedPrompt};

        let model = model_fixture(true);
        let capability = model.inference_state_contract().unwrap();
        let lease = ChatModelLease::for_test(NativeChatModel::Qwen38(model));
        let instance = ModelInstanceId::new(9191);
        let mut actual_cache = ManagedKvCacheManager::new(None);
        let mut baseline_cache = ManagedKvCacheManager::new(None);
        let mut request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "fixture".into(),
        }])
        .with_model_variant(ModelVariant::Qwen3827BFp8)
        .with_streaming(true);
        request.model_instance_id = Some(instance);
        request.params.max_tokens = 12;
        request.params.temperature = 0.8;
        let prepared = prepared_prompt();
        request
            .install_chat_execution_preparation_with_model(
                ModelVariant::Qwen3827BFp8,
                prepared.prompt_ids().to_vec(),
                Some(NativeChatPreparedPrompt::Qwen38(prepared)),
                lease,
                32,
            )
            .unwrap();
        let runtime = actual_cache
            .bind_request(instance, BackendKind::Cpu, 16, 4, &capability)
            .unwrap()
            .unwrap();
        request
            .install_managed_cache_runtime(runtime.clone())
            .unwrap();
        let mut baseline_request = request.clone();
        let baseline_runtime = baseline_cache
            .bind_request(instance, BackendKind::Cpu, 16, 4, &capability)
            .unwrap()
            .unwrap();
        // Separate managers intentionally own separate model runtime allocations.
        baseline_request.managed_cache_runtime = Some(baseline_runtime.clone());
        let config = super::super::WorkerConfig {
            enable_chunked_prefill: true,
            ..Default::default()
        };
        let actual = NativeExecutor::new(config.clone());
        let baseline = NativeExecutor::new(config);
        let scheduled = |plan, start, end, prefill| ScheduledRequest {
            plan_id: plan,
            request_id: request.id.clone(),
            sequence_id: 1,
            num_tokens: end - start,
            num_computed_tokens: start,
            is_prefill: prefill,
            work: WorkUnit::SequenceStep {
                phase: if prefill {
                    SequencePhase::Prefill
                } else {
                    SequencePhase::Decode
                },
                input: InputRange { start, end },
                max_output_steps: 1,
                auxiliary_state: None,
            },
        };
        let run = |executor: &NativeExecutor,
                   manager: &mut ManagedKvCacheManager,
                   request: &EngineCoreRequest,
                   row: &ScheduledRequest| {
            let runtime = request.managed_cache_runtime().unwrap();
            let reservation = manager
                .prepare_incremental(
                    runtime,
                    row.plan_id,
                    &row.session_key(),
                    &row.work,
                    Some(request),
                )
                .unwrap()
                .unwrap();
            let caches =
                super::super::qwen38_managed_caches_for_row(request, row, &reservation).unwrap();
            request.begin_stream_staging().unwrap();
            let result = executor
                .chat_request_with_managed_cache(
                    request,
                    row,
                    Some(caches.target),
                    caches.mtp,
                    reservation.clocked_state.clone(),
                )
                .unwrap();
            let receipt = reservation
                .completed_write_receipt(&result.managed_cache_completions)
                .unwrap();
            manager
                .finalize(&reservation, Some(&receipt), true)
                .unwrap();
            for group in &runtime.plan().groups {
                assert_eq!(
                    manager
                        .snapshot(instance, &row.session_key(), group.domain)
                        .unwrap()
                        .committed_tokens,
                    (row.num_computed_tokens + row.num_tokens) as u32
                );
            }
            let tensor_sequence = crate::backends::state::PhysicalStateSequenceId::new(
                reservation.clocked_state.as_ref().unwrap().sequence,
            )
            .unwrap();
            let tensors = runtime
                .state_plan_v2()
                .non_paged
                .iter()
                .flat_map(|domain| {
                    runtime
                        .tensor_state()
                        .unwrap()
                        .read(tensor_sequence, domain.domain())
                        .unwrap()
                        .unwrap()
                        .components
                        .iter()
                        .map(|component| {
                            component
                                .tensor
                                .as_ref()
                                .unwrap()
                                .flatten_all()
                                .unwrap()
                                .to_vec1::<f32>()
                                .unwrap()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            assert_eq!(
                tensors.len(),
                2,
                "fixture must exercise recurrent and convolution domains"
            );
            assert!(
                tensors
                    .iter()
                    .all(|values| values.iter().any(|value| value.abs() > 1e-6)),
                "hybrid state must be nonzero"
            );
            let events = request.take_staged_stream_outputs().unwrap();
            (result.output, events, tensors)
        };
        let compare_tensors = |actual: &Vec<Vec<f32>>, expected: &Vec<Vec<f32>>| {
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                assert_eq!(actual.len(), expected.len());
                for (actual, expected) in actual.iter().zip(expected) {
                    assert!(
                        (actual - expected).abs() < 1e-5,
                        "hybrid state mismatch: {actual} vs {expected}"
                    );
                }
            }
        };
        let mut baseline_tensors = Vec::new();
        for (plan, start, end, prefill) in [(1, 0, 2, true), (2, 2, 3, false), (3, 3, 4, false)] {
            let row = scheduled(plan, start, end, prefill);
            let (_, actual_events, actual_tensors) =
                run(&actual, &mut actual_cache, &request, &row);
            let (_, expected_events, expected_tensors) =
                run(&baseline, &mut baseline_cache, &baseline_request, &row);
            compare_tensors(&actual_tensors, &expected_tensors);
            baseline_tensors = expected_tensors;
            assert_eq!(
                actual_events
                    .iter()
                    .map(|e| (&e.text, e.sequence))
                    .collect::<Vec<_>>(),
                expected_events
                    .iter()
                    .map(|e| (&e.text, e.sequence))
                    .collect::<Vec<_>>()
            );
        }
        let session = scheduled(3, 3, 4, false).session_key();
        assert_eq!(actual.suspend_chat_session(&session).unwrap(), Some(4));
        actual_cache.release_session(&session).unwrap();
        assert!(!actual
            .chat_decode_states
            .lock()
            .unwrap()
            .contains_key(&session));
        for cursor in 0..4 {
            let row = scheduled(10 + cursor as u64, cursor, cursor + 1, true);
            let (output, events, replay_tensors) = run(&actual, &mut actual_cache, &request, &row);
            if cursor == 3 {
                compare_tensors(&replay_tensors, &baseline_tensors);
            }
            assert_eq!(output.tokens_generated, 0);
            assert_eq!(output.tokens_processed, 1);
            assert!(
                events.is_empty(),
                "replay must not publish any duplicate token"
            );
        }
        assert!(!actual
            .suspended_chat_states
            .lock()
            .unwrap()
            .contains_key(&session));
        for cursor in 4..7 {
            let row = scheduled(20 + cursor as u64, cursor, cursor + 1, false);
            let (_, actual_events, actual_tensors) =
                run(&actual, &mut actual_cache, &request, &row);
            let (_, expected_events, expected_tensors) =
                run(&baseline, &mut baseline_cache, &baseline_request, &row);
            compare_tensors(&actual_tensors, &expected_tensors);
            assert_eq!(
                actual_events
                    .iter()
                    .map(|e| (&e.text, e.sequence))
                    .collect::<Vec<_>>(),
                expected_events
                    .iter()
                    .map(|e| (&e.text, e.sequence))
                    .collect::<Vec<_>>()
            );
        }
    }

    /// D2/D3: a streaming chat session suspended for capacity after
    /// `suspend_after` decode quanta is rebuilt the way CPU and Metal serve it
    /// (chunked prefill off, so the whole replay is one prefill span in a fresh
    /// reservation). The replay span publishes nothing and generates nothing,
    /// and the resumed session streams the same deltas, sequence numbers,
    /// final text and token counts as an unpreempted run with the same seed.
    fn assert_replay_resumes_the_stream(
        lease: &crate::models::registry::ChatModelLease,
        capability: &crate::kv::InferenceStateCapability,
        variant: ModelVariant,
        prompt: &[u32],
        suspend_after: usize,
    ) {
        use crate::backends::BackendKind;
        use crate::engine::ModelInstanceId;

        let instance = ModelInstanceId::new(9393);
        let mut manager = crate::engine::cache::managed::ManagedKvCacheManager::new(None);
        let runtime = manager
            .bind_request(instance, BackendKind::Cpu, 16, 8, capability)
            .unwrap()
            .unwrap();
        let request = |id: &str| {
            let mut request = EngineCoreRequest::chat(vec![ChatMessage {
                role: ChatRole::User,
                content: id.into(),
            }])
            .with_model_variant(variant)
            .with_streaming(true);
            request.id = id.into();
            request.model_instance_id = Some(instance);
            request.params.max_tokens = 16;
            request.params.temperature = 0.8;
            request.params.seed = Some(7);
            request
                .install_chat_execution_preparation_with_model(
                    variant,
                    prompt.to_vec(),
                    None,
                    lease.clone(),
                    64,
                )
                .unwrap();
            request
                .install_managed_cache_runtime(runtime.clone())
                .unwrap();
            request
        };
        let executor = NativeExecutor::new(super::super::WorkerConfig::default());
        assert!(!executor.config.enable_chunked_prefill);
        let mut plan = 0;
        let mut run = |manager: &mut crate::engine::cache::managed::ManagedKvCacheManager,
                       request: &EngineCoreRequest,
                       start: usize,
                       end: usize,
                       prefill: bool| {
            plan += 1;
            let row = MixedWorld::row(request, plan, start, end, prefill);
            let reservation = manager
                .prepare_incremental(
                    &runtime,
                    row.plan_id,
                    &row.session_key(),
                    &row.work,
                    Some(request),
                )
                .unwrap()
                .unwrap();
            request.begin_stream_staging().unwrap();
            let result = executor.execute_single_request(&[request], &row, Some(&reservation));
            assert!(result.output.error.is_none(), "{:?}", result.output.error);
            let receipt = reservation
                .completed_write_receipt(&result.managed_cache_completions)
                .unwrap();
            manager
                .finalize(&reservation, Some(&receipt), true)
                .unwrap();
            let events = request
                .take_staged_stream_outputs()
                .unwrap()
                .into_iter()
                .map(|event| (event.text, event.sequence, event.is_final))
                .collect::<Vec<_>>();
            (result.output, events)
        };
        // Each run's events, generated tokens and terminal output.
        let mut streams = Vec::new();
        for (id, suspend) in [("baseline", None), ("preempted", Some(suspend_after))] {
            let request = request(id);
            let (output, mut events) = run(&mut manager, &request, 0, prompt.len(), true);
            let mut generated = output.tokens_generated;
            let mut cursor = prompt.len();
            let mut decoded = 0;
            let terminal = loop {
                if suspend == Some(decoded) {
                    let session = MixedWorld::row(&request, 0, 0, 0, false).session_key();
                    let replay_tokens = executor.suspend_chat_session(&session).unwrap().unwrap();
                    assert_eq!(replay_tokens, cursor);
                    manager.release_session(&session).unwrap();
                    let (output, replay_events) =
                        run(&mut manager, &request, 0, replay_tokens, true);
                    assert!(replay_events.is_empty(), "replay republished output");
                    assert_eq!(output.tokens_generated, 0);
                    assert_eq!(output.tokens_processed, replay_tokens);
                    assert!(!output.finished);
                }
                let (output, step_events) = run(&mut manager, &request, cursor, cursor + 1, false);
                events.extend(step_events);
                generated += output.tokens_generated;
                cursor += 1;
                decoded += 1;
                if output.finished {
                    break output;
                }
            };
            assert!(
                decoded > suspend_after,
                "the run ended before its suspension"
            );
            streams.push((events, generated, terminal.text, terminal.tokens_processed));
        }
        assert_eq!(streams[1], streams[0]);
    }

    #[test]
    fn replayed_dense_sessions_resume_their_streams_without_republishing() {
        use crate::kv::InferenceStateContractProvider;
        use crate::models::architectures::gemma3::chat::tiny_chat_model_for_test as tiny_gemma3;
        use crate::models::architectures::qwen3::chat::tiny_chat_model_for_test as tiny_qwen3;
        use crate::models::registry::ChatModelLease;

        let qwen3 = tiny_qwen3(&candle_core::Device::Cpu);
        let capability = qwen3.inference_state_contract().unwrap();
        let lease = ChatModelLease::for_test(NativeChatModel::Qwen3(qwen3));
        assert_replay_resumes_the_stream(
            &lease,
            &capability,
            ModelVariant::Qwen306B,
            &[0, 1, 7, 2, 5],
            6,
        );
        let gemma3 = tiny_gemma3(&candle_core::Device::Cpu);
        let capability = gemma3.inference_state_contract().unwrap();
        let lease = ChatModelLease::for_test(NativeChatModel::Gemma3(gemma3));
        assert_replay_resumes_the_stream(
            &lease,
            &capability,
            ModelVariant::Gemma34BIt,
            &[0, 1, 2, 9, 4],
            6,
        );
    }

    /// The hybrid LFM2 replay rebuilds its ShortConv tensor state through
    /// the managed tensor-state reservation as well as its KV pages.
    #[test]
    fn replayed_hybrid_sessions_resume_their_streams_without_republishing() {
        use crate::kv::InferenceStateContractProvider;
        use crate::models::architectures::lfm2::chat::tiny_lfm_model_for_test;
        use crate::models::registry::ChatModelLease;

        let lfm2 = tiny_lfm_model_for_test(3);
        let capability = lfm2.inference_state_contract().unwrap();
        let lease = ChatModelLease::for_test(NativeChatModel::Lfm2(lfm2));
        assert_replay_resumes_the_stream(
            &lease,
            &capability,
            ModelVariant::Lfm2512BInstructGguf,
            &[1, 3, 4, 5, 6],
            5,
        );
    }

    /// Two chat requests over one tiny Qwen3.8 model, managed cache and
    /// executor: "decoder" already prefilled, "prompt" not started.
    struct MixedWorld {
        executor: NativeExecutor,
        manager: crate::engine::cache::managed::ManagedKvCacheManager,
        runtime: Arc<crate::engine::ManagedKvModelRuntime>,
        decoder: EngineCoreRequest,
        prompt: EngineCoreRequest,
    }

    impl MixedWorld {
        /// Two Qwen3.8 requests with the fixture's prepared prompt.
        fn new(
            lease: &crate::models::registry::ChatModelLease,
            capability: &crate::kv::InferenceStateCapability,
        ) -> Self {
            use crate::models::architectures::qwen38::chat::recovery_tests::prepared_prompt;

            Self::with_prompts(lease, capability, ModelVariant::Qwen3827BFp8, |_| {
                let prepared = prepared_prompt();
                (
                    prepared.prompt_ids().to_vec(),
                    Some(NativeChatPreparedPrompt::Qwen38(prepared)),
                )
            })
        }

        /// A `decoder` request, prefilled and ready to decode, and a `prompt`
        /// request, each with the prompt `prompt(id)` returns.
        fn with_prompts(
            lease: &crate::models::registry::ChatModelLease,
            capability: &crate::kv::InferenceStateCapability,
            variant: ModelVariant,
            prompt: impl Fn(&str) -> (Vec<u32>, Option<NativeChatPreparedPrompt>),
        ) -> Self {
            use crate::backends::BackendKind;
            use crate::engine::ModelInstanceId;

            let instance = ModelInstanceId::new(9292);
            let mut manager = crate::engine::cache::managed::ManagedKvCacheManager::new(None);
            let runtime = manager
                .bind_request(instance, BackendKind::Cpu, 16, 4, capability)
                .unwrap()
                .unwrap();
            let request = |id: &str| {
                let mut request = EngineCoreRequest::chat(vec![ChatMessage {
                    role: ChatRole::User,
                    content: id.into(),
                }])
                .with_model_variant(variant);
                request.id = id.into();
                request.model_instance_id = Some(instance);
                request.params.max_tokens = 12;
                request.params.temperature = 0.8;
                let (prompt_ids, prepared) = prompt(id);
                request
                    .install_chat_execution_preparation_with_model(
                        variant,
                        prompt_ids,
                        prepared,
                        lease.clone(),
                        32,
                    )
                    .unwrap();
                request
                    .install_managed_cache_runtime(runtime.clone())
                    .unwrap();
                request
            };
            let mut world = Self {
                executor: NativeExecutor::new(super::super::WorkerConfig {
                    enable_chunked_prefill: true,
                    ..Default::default()
                }),
                manager,
                runtime: runtime.clone(),
                decoder: request("decoder"),
                prompt: request("prompt"),
            };
            let decoder = world.decoder.clone();
            let prefill = Self::row(&decoder, 1, 0, decoder.num_prompt_tokens(), true);
            let reservation = world.reserve(&decoder, &prefill);
            let result =
                world
                    .executor
                    .execute_single_request(&[&decoder], &prefill, Some(&reservation));
            world.commit(&reservation, &result);
            world
        }

        fn row(
            request: &EngineCoreRequest,
            plan: u64,
            start: usize,
            end: usize,
            prefill: bool,
        ) -> ScheduledRequest {
            ScheduledRequest {
                plan_id: plan,
                request_id: request.id.clone(),
                sequence_id: 1,
                num_tokens: end - start,
                is_prefill: prefill,
                num_computed_tokens: start,
                work: WorkUnit::SequenceStep {
                    phase: if prefill {
                        SequencePhase::Prefill
                    } else {
                        SequencePhase::Decode
                    },
                    input: InputRange { start, end },
                    max_output_steps: end - start,
                    auxiliary_state: None,
                },
            }
        }

        fn reserve(
            &mut self,
            request: &EngineCoreRequest,
            row: &ScheduledRequest,
        ) -> crate::engine::ManagedCacheReservation {
            self.manager
                .prepare_incremental(
                    &self.runtime,
                    row.plan_id,
                    &row.session_key(),
                    &row.work,
                    Some(request),
                )
                .unwrap()
                .unwrap()
        }

        fn commit(
            &mut self,
            reservation: &crate::engine::ManagedCacheReservation,
            result: &ModelSessionResult,
        ) {
            assert!(result.output.error.is_none(), "{:?}", result.output.error);
            let receipt = reservation
                .completed_write_receipt(&result.managed_cache_completions)
                .unwrap();
            self.manager
                .finalize(reservation, Some(&receipt), true)
                .unwrap();
        }

        fn views(
            &self,
            requests: &[&EngineCoreRequest],
            rows: &[ScheduledRequest],
            reservations: &[crate::engine::ManagedCacheReservation],
        ) -> Vec<Option<super::super::RetainedRowManagedState>> {
            rows.iter()
                .zip(requests)
                .zip(reservations)
                .map(|((row, request), reservation)| {
                    Some(
                        super::super::retained_row_managed_state_for_row(request, row, reservation)
                            .unwrap(),
                    )
                })
                .collect()
        }

        /// A ready row carrying `reservation`; only the reservation is read.
        fn ready(
            row: &ScheduledRequest,
            reservation: &crate::engine::ManagedCacheReservation,
        ) -> crate::engine::ReadyQuantum {
            crate::engine::ReadyQuantum {
                plan_id: row.plan_id,
                session: row.session_key(),
                lane: crate::engine::BatchLaneKey {
                    execution_group: crate::engine::ExecutionGroupId::new(1),
                    model_instance: crate::engine::ModelInstanceId::new(9292),
                    adapter_instance: crate::engine::AdapterInstanceId::new(1),
                    adapter_abi: crate::engine::AdapterAbiRevision::new(1),
                    capability_id: "chat".into(),
                    stage_id: crate::engine::StageId::new(3),
                    backend: crate::backends::BackendKind::Cpu,
                    device_ordinal: None,
                    compute_dtype: "f32".into(),
                    state_dtype: "f32".into(),
                    tensor_layout: "ragged".into(),
                    quantization: "adapter-owned".into(),
                    state_schema: "test".into(),
                    kernel_mode: "chat.step.mixed".into(),
                    semantic_mode: "chat.sequence_step".into(),
                    shape_bucket: "ragged".into(),
                },
                work: row.work.clone(),
                cost: crate::engine::WorkCost::new(row.num_tokens as u64, row.num_tokens as u64, 0),
                managed_cache: Some(reservation.clone()),
            }
        }
    }

    fn session_summary(
        result: &ModelSessionResult,
    ) -> (String, usize, usize, Option<String>, bool) {
        (
            result.output.request_id.clone(),
            result.output.tokens_processed,
            result.output.tokens_generated,
            result.output.text.clone(),
            result.output.finished,
        )
    }

    /// `IZWI_MIXED_FORCE_SPLIT` runs a mixed batch (a decode row, then a
    /// final prefill span) through the split handlers with each row's own
    /// managed reservation, in batch order: every row's output, and the next
    /// decode step, equal those of the same rows run one handler at a time.
    /// The mixed call itself refuses a model without a mixed forward before
    /// it touches any row's state.
    #[test]
    fn forced_split_mixed_step_runs_the_split_handlers_in_batch_order() {
        use crate::kv::InferenceStateContractProvider;
        use crate::models::architectures::qwen38::chat::recovery_tests::model_fixture;
        use crate::models::registry::ChatModelLease;

        let model = model_fixture(true);
        let capability = model.inference_state_contract().unwrap();
        let lease = ChatModelLease::for_test(NativeChatModel::Qwen38(model));
        let mut oracle = MixedWorld::new(&lease, &capability);
        let mut reference = MixedWorld::new(&lease, &capability);
        let (decoder, prompt) = (oracle.decoder.clone(), oracle.prompt.clone());
        let decode_at = decoder.num_prompt_tokens();
        let span_tokens = prompt.num_prompt_tokens();
        let scheduled = [
            MixedWorld::row(&decoder, 2, decode_at, decode_at + 1, false),
            MixedWorld::row(&prompt, 3, 0, span_tokens, true),
        ];

        // Aborted transactions stay terminal, so the refused attempt takes
        // its own plan ids.
        let refused_rows = scheduled.clone().map(|mut row| {
            row.plan_id += 100;
            row
        });
        let reservations = refused_rows
            .iter()
            .zip([&decoder, &prompt])
            .map(|(row, request)| oracle.reserve(request, row))
            .collect::<Vec<_>>();
        let views = oracle.views(&[&decoder, &prompt], &refused_rows, &reservations);
        let refused = oracle
            .executor
            .chat_mixed_step_with_managed(&[&decoder, &prompt], &refused_rows, views)
            .unwrap_err();
        assert!(refused.to_string().contains("mixed"), "{refused}");
        for reservation in &reservations {
            oracle.manager.finalize(reservation, None, false).unwrap();
        }

        let reservations = scheduled
            .iter()
            .zip([&decoder, &prompt])
            .map(|(row, request)| oracle.reserve(request, row))
            .collect::<Vec<_>>();
        let rows = scheduled
            .iter()
            .zip(&reservations)
            .map(|(row, reservation)| MixedWorld::ready(row, reservation))
            .collect::<Vec<_>>();
        let mixed = oracle
            .executor
            .chat_mixed_step_split(&[&decoder, &prompt], &scheduled, &rows)
            .unwrap();
        assert_eq!(mixed.len(), 2);
        for (reservation, result) in reservations.iter().zip(&mixed) {
            oracle.commit(reservation, result);
        }

        let (reference_decoder, reference_prompt) =
            (reference.decoder.clone(), reference.prompt.clone());
        let reservation = reference.reserve(&reference_decoder, &scheduled[0]);
        let views = reference.views(
            &[&reference_decoder],
            &scheduled[..1],
            std::slice::from_ref(&reservation),
        );
        let mut split = reference
            .executor
            .chat_decode_batch_with_managed(&[&reference_decoder], &scheduled[..1], views)
            .unwrap();
        reference.commit(&reservation, &split[0]);
        let reservation = reference.reserve(&reference_prompt, &scheduled[1]);
        split.push(reference.executor.execute_single_request(
            &[&reference_prompt],
            &scheduled[1],
            Some(&reservation),
        ));
        reference.commit(&reservation, &split[1]);

        assert_eq!(
            mixed.iter().map(session_summary).collect::<Vec<_>>(),
            split.iter().map(session_summary).collect::<Vec<_>>()
        );
        assert_eq!(
            (
                mixed[0].output.tokens_processed,
                mixed[1].output.tokens_processed
            ),
            (1, span_tokens)
        );

        // Both worlds continue identically: one more decode row for every
        // request that has not finished.
        let next = [
            MixedWorld::row(&decoder, 4, decode_at + 1, decode_at + 2, false),
            MixedWorld::row(&prompt, 5, span_tokens, span_tokens + 1, false),
        ]
        .into_iter()
        .zip(&mixed)
        .filter(|(_, result)| !result.output.finished)
        .map(|(row, _)| row)
        .collect::<Vec<_>>();
        assert!(!next.is_empty(), "the fixture finished every row at once");
        let step = |world: &mut MixedWorld| {
            let requests = next
                .iter()
                .map(|row| {
                    if row.request_id == world.decoder.id {
                        world.decoder.clone()
                    } else {
                        world.prompt.clone()
                    }
                })
                .collect::<Vec<_>>();
            let refs = requests.iter().collect::<Vec<_>>();
            let reservations = next
                .iter()
                .zip(&requests)
                .map(|(row, request)| world.reserve(request, row))
                .collect::<Vec<_>>();
            let views = world.views(&refs, &next, &reservations);
            let results = world
                .executor
                .chat_decode_batch_with_managed(&refs, &next, views)
                .unwrap();
            for (reservation, result) in reservations.iter().zip(&results) {
                world.commit(reservation, result);
            }
            results.iter().map(session_summary).collect::<Vec<_>>()
        };
        assert_eq!(step(&mut oracle), step(&mut reference));
    }

    /// Steps of a mixed batch through the loaded family's mixed forward,
    /// against the same steps through the forced-split oracle
    /// (`IZWI_MIXED_FORCE_SPLIT`) from the same start: the decoder decodes
    /// while the prompt prefills in a non-final and a final span, then both
    /// decode. Every row's output of every step must be equal.
    fn assert_mixed_steps_match_forced_split(world: impl Fn() -> MixedWorld) {
        let (mut mixed, mut split) = (world(), world());
        // Each world's requests carry that world's managed runtime.
        let (decoder, prompt) = (mixed.decoder.clone(), mixed.prompt.clone());
        let split_requests = (split.decoder.clone(), split.prompt.clone());
        let split_requests = [&split_requests.0, &split_requests.1];
        let decode_at = decoder.num_prompt_tokens();
        let prompt_len = prompt.num_prompt_tokens();
        let first = prompt_len / 2;
        let steps = [
            [
                MixedWorld::row(&decoder, 2, decode_at, decode_at + 1, false),
                MixedWorld::row(&prompt, 3, 0, first, true),
            ],
            [
                MixedWorld::row(&decoder, 4, decode_at + 1, decode_at + 2, false),
                MixedWorld::row(&prompt, 5, first, prompt_len, true),
            ],
            [
                MixedWorld::row(&decoder, 6, decode_at + 2, decode_at + 3, false),
                MixedWorld::row(&prompt, 7, prompt_len, prompt_len + 1, false),
            ],
        ];
        let requests = [&decoder, &prompt];
        for (step, scheduled) in steps.iter().enumerate() {
            let reservations = scheduled
                .iter()
                .zip(requests)
                .map(|(row, request)| mixed.reserve(request, row))
                .collect::<Vec<_>>();
            let views = mixed.views(&requests, scheduled, &reservations);
            let actual = mixed
                .executor
                .chat_mixed_step_with_managed(&requests, scheduled, views)
                .unwrap();
            for (reservation, result) in reservations.iter().zip(&actual) {
                mixed.commit(reservation, result);
            }

            let reservations = scheduled
                .iter()
                .zip(split_requests)
                .map(|(row, request)| split.reserve(request, row))
                .collect::<Vec<_>>();
            let rows = scheduled
                .iter()
                .zip(&reservations)
                .map(|(row, reservation)| MixedWorld::ready(row, reservation))
                .collect::<Vec<_>>();
            let expected = split
                .executor
                .chat_mixed_step_split(&split_requests, scheduled, &rows)
                .unwrap();
            for (reservation, result) in reservations.iter().zip(&expected) {
                split.commit(reservation, result);
            }
            assert!(
                actual.iter().all(|result| !result.output.finished),
                "step {step}: the fixture finished a row"
            );
            assert_eq!(
                actual.iter().map(session_summary).collect::<Vec<_>>(),
                expected.iter().map(session_summary).collect::<Vec<_>>(),
                "step {step}"
            );
        }
    }

    /// Qwen3 mixed steps commit exactly what the split handlers commit.
    #[test]
    fn qwen3_mixed_steps_match_the_forced_split_oracle() {
        use crate::kv::InferenceStateContractProvider;
        use crate::models::architectures::qwen3::chat::tiny_chat_model_for_test;
        use crate::models::registry::ChatModelLease;

        let model = tiny_chat_model_for_test(&candle_core::Device::Cpu);
        let capability = model.inference_state_contract().unwrap();
        let lease = ChatModelLease::for_test(NativeChatModel::Qwen3(model));
        assert!(lease.supports_mixed_step());
        assert_mixed_steps_match_forced_split(|| {
            MixedWorld::with_prompts(&lease, &capability, ModelVariant::Qwen306B, |id| {
                let prompt = match id {
                    "decoder" => vec![0, 1, 7, 2],
                    _ => vec![3, 5, 0, 6, 1, 4, 2, 7, 5],
                };
                (prompt, None)
            })
        });
    }

    /// LFM2 mixed steps (each prefill span's ShortConv segment on its own
    /// state) commit exactly what the split handlers commit.
    #[test]
    fn lfm2_mixed_steps_match_the_forced_split_oracle() {
        use crate::kv::InferenceStateContractProvider;
        use crate::models::architectures::lfm2::chat::tiny_lfm_model_for_test;
        use crate::models::registry::ChatModelLease;

        let model = tiny_lfm_model_for_test(3);
        let capability = model.inference_state_contract().unwrap();
        let lease = ChatModelLease::for_test(NativeChatModel::Lfm2(model));
        assert!(lease.supports_mixed_step());
        assert_mixed_steps_match_forced_split(|| {
            MixedWorld::with_prompts(
                &lease,
                &capability,
                ModelVariant::Lfm2512BInstructGguf,
                |id| {
                    let prompt = match id {
                        "decoder" => vec![1, 3, 4, 5],
                        _ => vec![1, 4, 3, 6, 5, 3, 4, 6, 5],
                    };
                    (prompt, None)
                },
            )
        });
    }
}
