use std::sync::Arc;
use std::time::Instant;

use crate::error::{Error, Result};
use crate::models::registry::{
    NativeChatDecodeCheckpoint, NativeChatDecodeState, NativeChatDecodeStep, NativeChatModel,
};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::chat::ChatGenerationConfig;
use crate::models::shared::chat::ChatMessage;

use super::super::request::EngineCoreRequest;
use super::super::scheduler::ScheduledRequest;
use super::super::types::AudioOutput;
use super::super::SessionKey;
use super::state::ActiveChatDecode;
use super::{
    ExecutorOutput, ExecutorPhaseTiming, ExecutorStateLease, ModelSessionResult, NativeExecutor,
};

const FALLBACK_CHAT_STREAM_BATCH_PIECES: usize = 4;
const FALLBACK_CHAT_STREAM_BATCH_BYTES: usize = 32;

fn begins_resumable_prefill_state(scheduled: &ScheduledRequest, resumable_prefill: bool) -> bool {
    scheduled.is_prefill && resumable_prefill && scheduled.num_computed_tokens == 0
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

#[derive(Debug, Default)]
struct StreamDeltaBatch {
    emitted_first: bool,
    pending: String,
    pending_pieces: usize,
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

impl StreamDeltaBatch {
    fn push(&mut self, delta: &str) -> Option<String> {
        if delta.is_empty() {
            return None;
        }
        if !self.emitted_first {
            self.emitted_first = true;
            return Some(delta.to_string());
        }

        self.pending.push_str(delta);
        self.pending_pieces += 1;
        if self.pending_pieces >= FALLBACK_CHAT_STREAM_BATCH_PIECES
            || self.pending.len() >= FALLBACK_CHAT_STREAM_BATCH_BYTES
            || delta.ends_with('\n')
        {
            return self.take_pending();
        }
        None
    }

    fn finish(&mut self) -> Option<String> {
        self.take_pending()
    }

    fn take_pending(&mut self) -> Option<String> {
        if self.pending.is_empty() {
            return None;
        }
        self.pending_pieces = 0;
        Some(std::mem::take(&mut self.pending))
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
        let NativeChatDecodeState::Qwen38(model_state) = &state.state else {
            return Ok(None);
        };
        let checkpoint = model_state.replay_checkpoint()?;
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
            },
        );
        // All CPU continuation state is installed before dropping GPU owners.
        states.remove(session);
        Ok(Some(tokens))
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
        let resumable_prefill =
            self.config.enable_chunked_prefill && model.supports_resumable_prefill();
        if managed_cache.is_some()
            && scheduled.is_prefill
            && !resumable_prefill
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
        // Fallback path for chat backends that do not expose incremental decode state.
        if !model.supports_incremental_decode() {
            let mut phase_timing_override: Option<ExecutorPhaseTiming> = None;
            let output = Self::run_blocking(|| {
                let generation_started = Instant::now();
                let mut first_output_ms_since_start: Option<f64> = None;
                let mut sequence = 0usize;
                let mut stream_err: Option<Error> = None;
                let mut stream_batch = StreamDeltaBatch::default();
                let mut streamed_text = String::new();

                let mut emit = |delta: &str| {
                    if first_output_ms_since_start.is_none() && !delta.is_empty() {
                        first_output_ms_since_start =
                            Some(generation_started.elapsed().as_secs_f64() * 1000.0);
                    }
                    if let Some(tx) = stream_tx.as_ref() {
                        if stream_err.is_none() {
                            streamed_text.push_str(delta);
                            if let Some(chunk) = stream_batch.push(delta) {
                                if let Err(err) = Self::stream_text_with_policy(
                                    tx,
                                    stream_policy,
                                    &request.id,
                                    &mut sequence,
                                    chunk,
                                ) {
                                    stream_err = Some(err);
                                }
                            }
                        }
                    }
                };

                let mut output = model.generate_with_callback_and_config(
                    messages,
                    max_new_tokens,
                    &generation_config,
                    &mut emit,
                )?;

                if let Some(tx) = stream_tx.as_ref() {
                    if stream_err.is_none() {
                        if let Some(chunk) = stream_batch.finish() {
                            if let Err(err) = Self::stream_text_with_policy(
                                tx,
                                stream_policy,
                                &request.id,
                                &mut sequence,
                                chunk,
                            ) {
                                stream_err = Some(err);
                            }
                        }
                    }
                }
                if let Some(err) = stream_err {
                    return Err(err);
                }
                if let Some(tx) = stream_tx.as_ref() {
                    Self::stream_final_marker_with_policy(
                        tx,
                        stream_policy,
                        &request.id,
                        &mut sequence,
                    )?;
                    output.text = canonical_chat_terminal_text(&streamed_text, output.text);
                }

                let total_ms = generation_started.elapsed().as_secs_f64() * 1000.0;
                let prefill_ms = first_output_ms_since_start.unwrap_or(total_ms);
                let decode_ms = (total_ms - prefill_ms).max(0.0);
                let decode_steps = if decode_ms > 0.0 {
                    u32::try_from(output.tokens_generated.max(1)).unwrap_or(u32::MAX)
                } else {
                    0
                };
                phase_timing_override = Some(ExecutorPhaseTiming {
                    prefill_ms: Some(prefill_ms.max(0.0)),
                    decode_ms: Some(decode_ms),
                    first_output_ms_since_start,
                    prefill_steps: Some(1),
                    decode_steps: Some(decode_steps),
                    ..ExecutorPhaseTiming::default()
                });

                Ok(output)
            })?;

            return Ok(ModelSessionResult::atomic(ExecutorOutput {
                request_id: request.id.clone(),
                audio: Some(AudioOutput::empty(24_000)),
                text: Some(output.text),
                input_transcription: None,
                tokens_processed: request.num_prompt_tokens(),
                tokens_generated: output.tokens_generated.max(1),
                finished: true,
                phase_timing_override,
                asr_diagnostics: None,
                error: None,
            }));
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
                let NativeChatModel::Qwen38(qwen) = model.as_ref() else {
                    return Err(Error::InferenceError(
                        "replay model does not match session".into(),
                    ));
                };
                let cache = managed_cache
                    .take()
                    .ok_or_else(|| Error::InferenceError("replay lost target cache".into()))?;
                let mut state = NativeChatDecodeState::Qwen38(qwen.begin_replay_state_physical(
                    &saved.checkpoint,
                    cache,
                    mtp_cache.take(),
                )?);
                if let Some(reservation) = tensor_reservation.as_ref() {
                    state.bind_hybrid_tensor_sequence(reservation.sequence)?;
                }
                state_lease.install_state(ActiveChatDecode {
                    variant,
                    state,
                    last_tokens_generated: saved.last_tokens_generated,
                    stream_sequence: saved.stream_sequence,
                    streamed_text: saved.streamed_text.clone(),
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
            if scheduled.is_prefill && resumable_prefill && scheduled.num_computed_tokens > 0 {
                return Err(Error::InferenceError(format!(
                    "resumable prefill request {} lost its decode state before span continuation; retry requires a fresh prompt",
                    request.id
                )));
            }
            let mut decode_state = match managed_cache.take() {
                Some(cache) if begins_resumable_prefill_state(scheduled, resumable_prefill) => {
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
                Some(cache) if matches!(model.as_ref(), NativeChatModel::Qwen35(_)) => {
                    Self::run_blocking(|| {
                        model.start_qwen35_decode_state_managed(
                            messages,
                            max_new_tokens,
                            &generation_config,
                            prepared_chat_prompt.and_then(|prepared| prepared.as_qwen35()),
                            cache,
                        )
                    })?
                }
                Some(cache) if matches!(model.as_ref(), NativeChatModel::Qwen38(_)) => {
                    Self::run_blocking(|| {
                        model.start_qwen38_decode_state_managed(
                            messages,
                            max_new_tokens,
                            &generation_config,
                            prepared_chat_prompt.and_then(|prepared| prepared.as_qwen38()),
                            cache,
                            mtp_cache.take(),
                        )
                    })?
                }
                Some(cache) if matches!(model.as_ref(), NativeChatModel::Gemma3(_)) => {
                    Self::run_blocking(|| {
                        model.start_gemma3_decode_state_managed(
                            messages,
                            max_new_tokens,
                            &generation_config,
                            cache,
                        )
                    })?
                }
                Some(cache) if matches!(model.as_ref(), NativeChatModel::Lfm2(_)) => {
                    Self::run_blocking(|| {
                        model.start_lfm2_decode_state_managed(
                            &request.prompt_tokens,
                            max_new_tokens,
                            &generation_config,
                            cache,
                        )
                    })?
                }
                Some(cache) => Self::run_blocking(|| {
                    model.start_qwen3_decode_state_managed(
                        messages,
                        max_new_tokens,
                        &generation_config,
                        cache,
                    )
                })?,
                None => {
                    return Err(Error::InferenceError(
                        "incremental chat execution requires scheduler-owned physical state".into(),
                    ))
                }
            };
            if let Some(reservation) = tensor_reservation {
                decode_state.bind_hybrid_tensor_sequence(reservation.sequence)?;
            }
            state_lease.install_state(ActiveChatDecode {
                variant,
                state: decode_state,
                last_tokens_generated: 0,
                stream_sequence: 0,
                streamed_text: String::new(),
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
        let resumable_prefill_quantum = scheduled.is_prefill && resumable_prefill;
        let resumable_span_tokens = resumable_prefill_quantum.then_some(scheduled.num_tokens);
        let replay_tokens = state_lease.state().and_then(|active| match &active.state {
            NativeChatDecodeState::Qwen38(state) => state.replay_tokens(),
            _ => None,
        });
        let resumable_span = resumable_prefill_quantum
            .then(|| {
                resumable_prefill_span(
                    scheduled,
                    replay_tokens.unwrap_or(request.num_prompt_tokens()),
                )
            })
            .transpose()?;
        state_lease.mark_dirty();
        let (step, final_text, finished, managed_cache_completions) = {
            let active_state = state_lease.require_state_mut()?;
            if matches!(active_state.state, NativeChatDecodeState::Lfm2(_))
                && crate::models::architectures::lfm2::diagnostics::enabled()
            {
                tracing::info!(request_id = %request.id, model = ?request.model_variant,
                    prompt_tokens = request.num_prompt_tokens(), max_new_tokens = request.params.max_tokens,
                    prefill = scheduled.is_prefill, "LFM2 quantum diagnostic context");
            }
            let step = if let (Some(_), Some((start, end))) = (replay_tokens, resumable_span) {
                let (NativeChatModel::Qwen38(qwen), NativeChatDecodeState::Qwen38(state)) =
                    (model.as_ref(), &mut active_state.state)
                else {
                    return Err(Error::InferenceError(
                        "chat replay crossed model family".into(),
                    ));
                };
                Self::run_blocking(|| qwen.continue_replay_physical(state, start, end))?;
                crate::engine::metrics::record_capacity_replay(end - start);
                NativeChatDecodeStep {
                    delta: String::new(),
                    text: active_state.streamed_text.clone(),
                    tokens_generated: active_state.last_tokens_generated,
                    input_tokens_committed: 0,
                    finished: false,
                }
            } else if let Some((span_start, span_end)) = resumable_span {
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
            } else {
                Self::run_blocking(|| model.decode_quantum(&mut active_state.state, input_budget))?
            };
            if request.is_cancelled() {
                return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    request.id.clone(),
                )));
            }

            record_lfm_completion(&active_state.state, request, &step);
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

            if let Some(tx) = stream_tx.as_ref() {
                if !step.delta.is_empty() {
                    Self::stream_text_with_policy(
                        tx,
                        stream_policy,
                        &request.id,
                        &mut active_state.stream_sequence,
                        step.delta.clone(),
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
            (step, final_text, finished, managed_cache_completions)
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
        })
        .with_managed_cache_completions(managed_cache_completions))
    }

    pub(super) fn chat_decode_batch(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ModelSessionResult>> {
        self.chat_decode_batch_with_managed(
            requests,
            scheduled,
            (0..scheduled.len()).map(|_| None).collect(),
        )
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
                .any(|scheduled| scheduled.is_prefill || scheduled.num_tokens != 1)
        {
            return Err(Error::InvalidInput(
                "continuous chat execution requires one decode token per row".to_string(),
            ));
        }
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
                    let (cache, mtp_cache) = if request.model_variant.is_some_and(|variant| {
                        variant.family() == crate::catalog::ModelFamily::Qwen38Chat
                    }) {
                        let target =
                            views.take_paged_domain(super::QWEN38_TARGET_ATTENTION_DOMAIN, true)?;
                        let mtp =
                            views.take_paged_domain(super::QWEN38_MTP_ATTENTION_DOMAIN, false)?;
                        views.ensure_all_paged_consumed()?;
                        (
                            target.ok_or_else(|| {
                                Error::InferenceError(
                                    "continuous Qwen3.8 row lost its target cache".into(),
                                )
                            })?,
                            mtp,
                        )
                    } else {
                        (views.take_only_paged()?, None)
                    };
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
        let steps = Self::run_blocking(|| model.decode_step_batch(&mut state_refs))?;
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
            let request = ordered_requests[*index];
            let active_state = lease.require_state_mut()?;
            record_lfm_completion(&active_state.state, request, &step);
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
                outputs[*index] = Some(ModelSessionResult::sequence(ExecutorOutput::error(
                    request.id.clone(),
                    format!("continuous chat stream staging failed: {error}"),
                )));
                continue;
            }

            let managed_cache_completions = active_state.state.take_managed_write_completions();
            outputs[*index] = Some(
                ModelSessionResult::sequence(ExecutorOutput {
                    request_id: request.id.clone(),
                    audio: Some(AudioOutput::empty(24_000)),
                    text: Some(if step.finished {
                        canonical_chat_terminal_text(&active_state.streamed_text, step.text)
                    } else {
                        step.text
                    }),
                    input_transcription: None,
                    tokens_processed: 1,
                    tokens_generated: step_tokens_generated,
                    finished: step.finished,
                    phase_timing_override: None,
                    asr_diagnostics: None,
                    error: None,
                })
                .with_managed_cache_completions(managed_cache_completions),
            );
            if !step.finished {
                continuing[*index] = true;
            }
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{GenerationParams, InputRange, SequencePhase, WorkUnit};
    use crate::model::ModelVariant;
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

        assert!(begins_resumable_prefill_state(&scheduled, true));
        assert!(!begins_resumable_prefill_state(&scheduled, false));
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

    #[test]
    fn chat_request_seed_is_stable_for_same_request_id() {
        let first = NativeExecutor::chat_request_seed("req-123");
        let second = NativeExecutor::chat_request_seed("req-123");
        let other = NativeExecutor::chat_request_seed("req-456");

        assert_eq!(first, second);
        assert_ne!(first, other);
    }

    #[test]
    fn stream_delta_batch_emits_first_delta_immediately_then_batches() {
        let mut batch = StreamDeltaBatch::default();

        assert_eq!(batch.push("A"), Some("A".to_string()));
        assert_eq!(batch.push("b"), None);
        assert_eq!(batch.push("c"), None);
        assert_eq!(batch.push("d"), None);
        assert_eq!(batch.push("e"), Some("bcde".to_string()));
    }

    #[test]
    fn stream_delta_batch_flushes_pending_on_finish() {
        let mut batch = StreamDeltaBatch::default();

        assert_eq!(batch.push("hello"), Some("hello".to_string()));
        assert_eq!(batch.push(" "), None);
        assert_eq!(batch.push("world"), None);
        assert_eq!(batch.finish(), Some(" world".to_string()));
        assert_eq!(batch.finish(), None);
    }

    #[test]
    fn stream_delta_batch_flushes_on_newline_boundary() {
        let mut batch = StreamDeltaBatch::default();

        assert_eq!(batch.push("intro"), Some("intro".to_string()));
        assert_eq!(batch.push(" line"), None);
        assert_eq!(batch.push("\n"), Some(" line\n".to_string()));
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
}
