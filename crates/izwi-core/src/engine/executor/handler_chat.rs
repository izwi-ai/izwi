use std::mem::size_of;
use std::sync::Arc;
use std::time::Instant;

use tracing::debug;

use crate::engine::resources::{ReservationClass, ReservationOwner, ResourceLease};
use crate::error::{Error, Result};
use crate::models::architectures::qwen3::core::Qwen3ManagedCache;
use crate::models::architectures::qwen35::chat::{Qwen35PrefixSnapshot, Qwen35PreparedPrompt};
use crate::models::registry::NativeChatModel;
use crate::models::shared::chat::ChatGenerationConfig;
use crate::models::shared::chat::ChatMessage;

use super::super::request::EngineCoreRequest;
use super::super::scheduler::ScheduledRequest;
use super::super::types::AudioOutput;
use super::prefix_cache::{ExactPrefixHandle, ExactPrefixScope};
use super::state::ActiveChatDecode;
use super::{ExecutorOutput, ExecutorPhaseTiming, ModelSessionResult, NativeExecutor};

const FALLBACK_CHAT_STREAM_BATCH_PIECES: usize = 4;
const FALLBACK_CHAT_STREAM_BATCH_BYTES: usize = 32;

struct PendingPrefixAuthorization {
    max_bytes: u64,
    lease: ResourceLease,
}

#[derive(Debug, Default)]
struct StreamDeltaBatch {
    emitted_first: bool,
    pending: String,
    pending_pieces: usize,
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

fn canonical_chat_terminal_text(streamed_text: &str, terminal_text: String) -> String {
    if streamed_text.is_empty() {
        terminal_text
    } else {
        streamed_text.to_string()
    }
}

impl NativeExecutor {
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
        self.chat_request_with_managed_cache(request, scheduled, None, None)
    }

    pub(super) fn chat_request_with_managed_cache(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        mut managed_cache: Option<Qwen3ManagedCache>,
        tensor_reservation: Option<crate::engine::ManagedTensorStateReservation>,
    ) -> Result<ModelSessionResult> {
        if request.managed_cache_runtime().is_some() != managed_cache.is_some() {
            return Err(Error::InferenceError(
                "managed Qwen3 execution requires its exact row reservation".to_string(),
            ));
        }
        if managed_cache.is_some()
            && scheduled.is_prefill
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
        let prepared_qwen35_prompt = request.prepared_qwen35_prompt_for_executor()?;
        let variant = Self::resolve_variant(request)?;
        let messages = Self::chat_messages(request)?;
        let max_new_tokens = request.params.max_tokens.max(1);
        let stream_tx = Self::stream_sender(request);
        let stream_policy = request.stream_policy;
        let generation_config = Self::chat_generation_config(request);
        let session = scheduled.session_key();
        let model = request.prepared_chat_model_for_executor()?;
        let prefix_scope = ExactPrefixScope {
            variant,
            backend: self.config.backend,
            activation_dtype: self.config.dtype.clone(),
            kv_cache_dtype: self.config.kv_cache_dtype.clone(),
        };
        let prefix_cache_enabled = managed_cache.is_none()
            && self.qwen35_prefix_cache_enabled(prepared_qwen35_prompt, model.as_ref());

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

        let mut active_state = {
            let mut guard = self.chat_decode_states.lock().map_err(|_| {
                Error::InferenceError("Chat decode state mutex poisoned".to_string())
            })?;
            // Prefill scheduling can happen after preemption; only recover state
            // owned by this exact request incarnation.
            guard.remove(&session)
        };

        if active_state
            .as_ref()
            .map(|state| state.variant != variant)
            .unwrap_or(false)
        {
            active_state = None;
        }

        let mut active_state = if let Some(mut state) = active_state {
            match managed_cache.take() {
                Some(cache) => state.state.install_managed_reservation(cache)?,
                None if state.state.uses_managed_kv() => {
                    return Err(Error::InferenceError(
                        "managed chat session lost its physical cache authority".to_string(),
                    ))
                }
                None => {}
            }
            if let (Some(arena), Some(reservation)) = (tensor_arena.as_ref(), tensor_reservation) {
                state
                    .state
                    .bind_qwen35_tensor_sequence(reservation.sequence)?;
                state.state.restore_qwen35_tensor_state(arena)?;
            }
            state
        } else {
            if request.is_cancelled() {
                return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    request.id.clone(),
                )));
            }
            let cached_prefix = prefix_cache_enabled
                .then(|| {
                    let prepared = prepared_qwen35_prompt
                        .expect("enabled Qwen3.5 prefix cache requires prepared prompt");
                    self.qwen35_prefix_cache.lookup(
                        &model,
                        &prefix_scope,
                        prepared.prompt_ids(),
                        prepared.prompt_positions(),
                    )
                })
                .flatten();
            let pending_prefix_authorization = prefix_cache_enabled
                .then(|| {
                    self.preauthorize_qwen35_prefix_snapshot(
                        request,
                        &prefix_scope,
                        prepared_qwen35_prompt
                            .expect("enabled Qwen3.5 prefix cache requires prepared prompt"),
                    )
                })
                .flatten();
            let capture_prefix_max_bytes = pending_prefix_authorization
                .as_ref()
                .map(|authorization| authorization.max_bytes);
            let mut decode_state = match managed_cache.take() {
                Some(cache) if matches!(model.as_ref(), NativeChatModel::Qwen35(_)) => {
                    Self::run_blocking(|| {
                        model.start_qwen35_decode_state_managed(
                            messages,
                            max_new_tokens,
                            &generation_config,
                            prepared_qwen35_prompt,
                            cache,
                        )
                    })?
                }
                Some(cache) => Self::run_blocking(|| {
                    model.start_qwen3_decode_state_managed(messages, max_new_tokens, cache)
                })?,
                None => Self::run_blocking(|| {
                    model.start_decode_state_with_prefix(
                        messages,
                        max_new_tokens,
                        &generation_config,
                        prepared_qwen35_prompt,
                        cached_prefix.as_ref().map(|cached| cached.snapshot()),
                        capture_prefix_max_bytes,
                    )
                })?,
            };
            if let Some(reservation) = tensor_reservation {
                decode_state.bind_qwen35_tensor_sequence(reservation.sequence)?;
            }
            let reused_prefix_tokens = decode_state.reused_qwen35_prefix_tokens();
            if reused_prefix_tokens > 0 {
                debug!(
                    model = %variant,
                    reused_prefix_tokens,
                    prompt_tokens = request.num_prompt_tokens(),
                    "Qwen3.5 exact-prefix cache hit"
                );
            }
            let pending_prefix_snapshot = match (
                decode_state.take_pending_qwen35_prefix_snapshot(),
                pending_prefix_authorization,
            ) {
                (Some(snapshot), Some(authorization)) => {
                    self.materialize_qwen35_prefix_snapshot(&prefix_scope, snapshot, authorization)
                }
                _ => None,
            };
            ActiveChatDecode {
                variant,
                state: decode_state,
                last_tokens_generated: 0,
                stream_sequence: 0,
                streamed_text: String::new(),
                pending_prefix_snapshot,
            }
        };

        let decode_iterations = if scheduled.is_prefill {
            1
        } else {
            scheduled.num_tokens.max(1)
        };
        let mut total_tokens_generated = 0usize;
        let mut decode_steps_ran = 0usize;
        let mut final_text = String::new();
        let mut finished = false;

        for _ in 0..decode_iterations {
            if request.is_cancelled() {
                return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    request.id.clone(),
                )));
            }
            let step = Self::run_blocking(|| model.decode_step(&mut active_state.state))?;
            if request.is_cancelled() {
                return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    request.id.clone(),
                )));
            }
            decode_steps_ran = decode_steps_ran.saturating_add(1);

            let step_tokens_generated = step
                .tokens_generated
                .saturating_sub(active_state.last_tokens_generated);
            active_state.last_tokens_generated = step.tokens_generated;
            total_tokens_generated = total_tokens_generated.saturating_add(step_tokens_generated);
            final_text = step.text.clone();

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

            if step.finished {
                if let Some(snapshot) = active_state.pending_prefix_snapshot.take() {
                    let _ = self
                        .qwen35_prefix_cache
                        .insert(&model, prefix_scope.clone(), snapshot);
                }
                finished = true;
                break;
            }
        }

        let tokens_processed = if scheduled.is_prefill {
            request.num_prompt_tokens()
        } else {
            decode_steps_ran.max(1)
        };
        if let Some(arena) = tensor_arena.as_ref() {
            active_state
                .state
                .stage_qwen35_tensor_state(arena, scheduled.plan_id)?;
        }
        let managed_cache_completions = active_state.state.take_managed_write_completions();
        if !finished {
            let mut guard = self.chat_decode_states.lock().map_err(|_| {
                Error::InferenceError("Chat decode state mutex poisoned".to_string())
            })?;
            guard.insert(session, active_state);
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
        managed_caches: Vec<Option<Qwen3ManagedCache>>,
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
        let model = ordered_requests[0].prepared_chat_model_for_executor()?;
        if !model.supports_continuous_decode_batch() {
            return Err(Error::InvalidInput(
                "loaded chat model has no continuous tensor decode adapter".to_string(),
            ));
        }
        for request in ordered_requests.iter().skip(1) {
            let row_model = request.prepared_chat_model_for_executor()?;
            if !Arc::ptr_eq(&model, &row_model) {
                return Err(Error::InferenceError(
                    "continuous chat batch spans different loaded model instances".to_string(),
                ));
            }
        }

        let mut active_states = {
            let mut guard = self.chat_decode_states.lock().map_err(|_| {
                Error::InferenceError("Chat decode state mutex poisoned".to_string())
            })?;
            for (request, scheduled) in ordered_requests.iter().zip(scheduled) {
                let session = scheduled.session_key();
                let expected_variant = Self::resolve_variant(request)?;
                let state = guard.get(&session).ok_or_else(|| {
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
            }
            scheduled
                .iter()
                .map(|scheduled| {
                    guard
                        .remove(&scheduled.session_key())
                        .expect("continuous chat state was validated under the same lock")
                })
                .collect::<Vec<_>>()
        };

        for ((request, active_state), managed_cache) in ordered_requests
            .iter()
            .zip(active_states.iter_mut())
            .zip(managed_caches)
        {
            if request.managed_cache_runtime().is_some() != managed_cache.is_some() {
                return Err(Error::InferenceError(
                    "continuous managed Qwen3 row lost its reservation".to_string(),
                ));
            }
            match managed_cache {
                Some(cache) => active_state.state.install_managed_reservation(cache)?,
                None if active_state.state.uses_managed_kv() => {
                    return Err(Error::InferenceError(
                        "continuous managed Qwen3 session changed cache authority".to_string(),
                    ))
                }
                None => {}
            }
        }

        let mut state_refs = active_states
            .iter_mut()
            .map(|state| &mut state.state)
            .collect::<Vec<_>>();
        let steps = Self::run_blocking(|| model.decode_step_batch(&mut state_refs))?;
        drop(state_refs);
        if steps.len() != active_states.len() {
            return Err(Error::InferenceError(
                "continuous chat model returned the wrong number of rows".to_string(),
            ));
        }

        let mut outputs = Vec::with_capacity(steps.len());
        let mut continuing = Vec::new();
        for (((request, scheduled), mut active_state), step) in ordered_requests
            .into_iter()
            .zip(scheduled)
            .zip(active_states)
            .zip(steps)
        {
            if request.is_cancelled() {
                outputs.push(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    request.id.clone(),
                )));
                continue;
            }

            let step_tokens_generated = step
                .tokens_generated
                .saturating_sub(active_state.last_tokens_generated);
            active_state.last_tokens_generated = step.tokens_generated;
            if let Some(tx) = Self::stream_sender(request).as_ref() {
                if !step.delta.is_empty() {
                    Self::stream_text_with_policy(
                        tx,
                        request.stream_policy,
                        &request.id,
                        &mut active_state.stream_sequence,
                        step.delta.clone(),
                    )?;
                    active_state.streamed_text.push_str(&step.delta);
                }
                if step.finished {
                    Self::stream_final_marker_with_policy(
                        tx,
                        request.stream_policy,
                        &request.id,
                        &mut active_state.stream_sequence,
                    )?;
                }
            }

            let managed_cache_completions = active_state.state.take_managed_write_completions();
            outputs.push(
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
                continuing.push((scheduled.session_key(), active_state));
            }
        }

        if !continuing.is_empty() {
            let mut guard = self.chat_decode_states.lock().map_err(|_| {
                Error::InferenceError("Chat decode state mutex poisoned".to_string())
            })?;
            for (session, state) in continuing {
                if guard.insert(session, state).is_some() {
                    return Err(Error::InferenceError(
                        "continuous chat state collided during commit".to_string(),
                    ));
                }
            }
        }
        Ok(outputs)
    }

    fn qwen35_prefix_cache_enabled(
        &self,
        prepared: Option<&Qwen35PreparedPrompt>,
        model: &NativeChatModel,
    ) -> bool {
        self.config.resource_authority.is_some()
            && self.qwen35_prefix_cache.max_retained_bytes() > 0
            && matches!(model, NativeChatModel::Qwen35(_))
            && prepared.is_some_and(Qwen35PreparedPrompt::supports_exact_prefix_reuse)
    }

    fn preauthorize_qwen35_prefix_snapshot(
        &self,
        request: &EngineCoreRequest,
        scope: &ExactPrefixScope,
        prepared: &Qwen35PreparedPrompt,
    ) -> Option<PendingPrefixAuthorization> {
        let state_bytes = self.authorized_session_cache_bytes(request).ok()?;
        let metadata_per_token = size_of::<u32>().checked_add(size_of::<[usize; 3]>())?;
        let metadata_bytes = prepared
            .prompt_ids()
            .len()
            .checked_mul(metadata_per_token)?
            .checked_add(size_of::<Qwen35PrefixSnapshot>())?;
        let max_bytes = state_bytes.checked_add(u64::try_from(metadata_bytes).ok()?)?;
        if max_bytes == 0 || max_bytes > self.qwen35_prefix_cache.max_retained_bytes() {
            return None;
        }
        let Some(authority) = self.config.resource_authority.as_ref() else {
            return None;
        };
        let resources = super::cache_resource_vector(scope.backend, max_bytes);
        let owner = ReservationOwner::new(
            ReservationClass::Cache,
            format!("qwen35-prefix-pending:{}", scope.variant),
        );
        let lease = authority.reserve(owner, resources).ok()?;
        Some(PendingPrefixAuthorization { max_bytes, lease })
    }

    fn materialize_qwen35_prefix_snapshot(
        &self,
        scope: &ExactPrefixScope,
        snapshot: Qwen35PrefixSnapshot,
        mut authorization: PendingPrefixAuthorization,
    ) -> Option<Arc<ExactPrefixHandle<Qwen35PrefixSnapshot>>> {
        let Some(bytes) = snapshot.retained_bytes() else {
            drop(snapshot);
            drop(authorization);
            return None;
        };
        if bytes == 0 || bytes > authorization.max_bytes {
            drop(snapshot);
            drop(authorization);
            return None;
        }
        let resources = super::cache_resource_vector(scope.backend, bytes);
        if authorization
            .lease
            .record_materialized_usage(resources)
            .is_err()
        {
            drop(snapshot);
            drop(authorization);
            return None;
        }
        // The copy is complete and its exact backing size is known. Relinquish
        // unused preauthorization while keeping the materialized snapshot fully
        // covered for its retained lifetime.
        let _ = authorization.lease.resize(resources);
        Some(ExactPrefixHandle::new(
            scope.backend,
            snapshot,
            Some(authorization.lease),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{GenerationParams, InputRange, SequencePhase, WorkUnit};
    use crate::model::ModelVariant;
    use crate::models::shared::chat::{ChatMessage, ChatRole};

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
            },
        };

        let error = executor
            .chat_request(&request, &scheduled)
            .expect_err("chat execution must require the private preparation marker");
        assert!(error
            .to_string()
            .contains("missing exact model prompt preparation"));

        request
            .install_chat_execution_preparation(ModelVariant::Qwen306B, vec![1, 2, 3], None)
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
}
