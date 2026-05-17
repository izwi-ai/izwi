use std::time::Instant;

use crate::backends::BackendKind;
use crate::error::{Error, Result};
use crate::models::shared::chat::ChatGenerationConfig;
use crate::models::shared::chat::ChatMessage;

use super::super::request::EngineCoreRequest;
use super::super::scheduler::ScheduledRequest;
use super::super::types::AudioOutput;
use super::state::{ActiveChatDecode, ActiveChatState};
use super::{ExecutorOutput, ExecutorPhaseTiming, NativeExecutor};

const FALLBACK_CHAT_STREAM_BATCH_PIECES: usize = 4;
const FALLBACK_CHAT_STREAM_BATCH_BYTES: usize = 32;

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

impl NativeExecutor {
    fn chat_generation_config(request: &EngineCoreRequest) -> ChatGenerationConfig {
        ChatGenerationConfig {
            temperature: request.params.temperature.max(0.0),
            top_p: request.params.top_p.clamp(0.0, 1.0),
            top_k: request.params.top_k,
            repetition_penalty: request.params.repetition_penalty.max(1.0),
            presence_penalty: request.params.presence_penalty.clamp(-2.0, 2.0),
            stop_token_ids: request.params.stop_token_ids.clone(),
            seed: Self::chat_request_seed(&request.id),
            request: request.chat_config.clone(),
        }
    }

    pub(super) fn chat_request_seed(request_id: &str) -> u64 {
        const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;

        let mut hash = FNV_OFFSET_BASIS;
        for byte in request_id.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    fn chat_messages(request: &EngineCoreRequest) -> Result<&[ChatMessage]> {
        request
            .chat_messages
            .as_deref()
            .ok_or_else(|| Error::InvalidInput("Chat request missing messages".to_string()))
    }

    fn should_use_qwen35_cuda_continuation_prefill(
        backend: BackendKind,
        model_supports_continuation_prefill: bool,
        scheduled: &ScheduledRequest,
        prompt_tokens: usize,
        has_active_prefill: bool,
    ) -> bool {
        backend == BackendKind::Cuda
            && model_supports_continuation_prefill
            && scheduled.is_prefill
            && (has_active_prefill || scheduled.num_tokens < prompt_tokens.max(1))
    }

    pub(super) fn chat_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
    ) -> Result<ExecutorOutput> {
        let variant = Self::resolve_variant(request)?;
        let messages = Self::chat_messages(request)?;
        let max_new_tokens = request.params.max_tokens.max(1);
        let stream_tx = Self::stream_sender(request);
        let stream_policy = request.stream_policy;
        let generation_config = Self::chat_generation_config(request);

        let model = self.with_registry(|registry| {
            registry
                .try_get_chat(variant)
                .ok_or_else(|| Error::ModelNotFound(format!("Chat model {variant} is not loaded")))
        })?;

        // Fallback path for chat backends that do not expose incremental decode state.
        if !model.supports_incremental_decode() {
            let mut phase_timing_override: Option<ExecutorPhaseTiming> = None;
            let output = Self::run_blocking(|| {
                let generation_started = Instant::now();
                let mut first_output_ms_since_start: Option<f64> = None;
                let mut sequence = 0usize;
                let mut stream_err: Option<Error> = None;
                let mut stream_batch = StreamDeltaBatch::default();

                let mut emit = |delta: &str| {
                    if first_output_ms_since_start.is_none() && !delta.is_empty() {
                        first_output_ms_since_start =
                            Some(generation_started.elapsed().as_secs_f64() * 1000.0);
                    }
                    if let Some(tx) = stream_tx.as_ref() {
                        if stream_err.is_none() {
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

                let output = model.generate_with_callback_and_config(
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
                    prefill_ms: prefill_ms.max(0.0),
                    decode_ms,
                    first_output_ms_since_start,
                    prefill_steps: 1,
                    decode_steps,
                });

                Ok(output)
            })?;

            return Ok(ExecutorOutput {
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
            });
        }

        let mut active_state = {
            let mut guard = self.chat_decode_states.lock().map_err(|_| {
                Error::InferenceError("Chat decode state mutex poisoned".to_string())
            })?;
            if scheduled.is_prefill {
                // Prefill scheduling can happen after preemption; reset stale state.
                guard.remove(&request.id)
            } else {
                guard.remove(&request.id)
            }
        };

        if active_state
            .as_ref()
            .map(|state| state.variant != variant)
            .unwrap_or(false)
        {
            active_state = None;
        }

        let has_active_prefill = active_state
            .as_ref()
            .map(|state| matches!(&state.state, ActiveChatState::Prefilling(_)))
            .unwrap_or(false);
        let use_continuation_prefill = Self::should_use_qwen35_cuda_continuation_prefill(
            self.config.backend,
            model.supports_continuation_prefill(),
            scheduled,
            request.num_prompt_tokens(),
            has_active_prefill,
        );

        let mut active_state = if let Some(state) = active_state {
            state
        } else if use_continuation_prefill {
            let prefill_state = Self::run_blocking(|| {
                model.start_prefill_state_with_config(messages, max_new_tokens, &generation_config)
            })?;
            ActiveChatDecode {
                variant,
                state: ActiveChatState::Prefilling(prefill_state),
                prompt_accounted: true,
                last_tokens_generated: 0,
                stream_sequence: 0,
            }
        } else {
            let decode_state = Self::run_blocking(|| {
                model.start_decode_state_with_config(messages, max_new_tokens, &generation_config)
            })?;
            ActiveChatDecode {
                variant,
                state: ActiveChatState::Decoding(decode_state),
                prompt_accounted: false,
                last_tokens_generated: 0,
                stream_sequence: 0,
            }
        };

        if use_continuation_prefill {
            let mut prefill_state = match active_state.state {
                ActiveChatState::Prefilling(state) => state,
                ActiveChatState::Decoding(_) => {
                    return Err(Error::InferenceError(
                        "Qwen3.5 CUDA continuation prefill received decode state during prefill"
                            .to_string(),
                    ));
                }
            };
            let step = Self::run_blocking(|| {
                model.prefill_next_chunk(&mut prefill_state, scheduled.num_tokens.max(1))
            })?;
            active_state.prompt_accounted = true;
            active_state.state = if step.finished {
                ActiveChatState::Decoding(Self::run_blocking(|| {
                    model.finish_prefill_state(prefill_state)
                })?)
            } else {
                ActiveChatState::Prefilling(prefill_state)
            };

            let mut guard = self.chat_decode_states.lock().map_err(|_| {
                Error::InferenceError("Chat decode state mutex poisoned".to_string())
            })?;
            guard.insert(request.id.clone(), active_state);

            return Ok(ExecutorOutput {
                request_id: request.id.clone(),
                audio: Some(AudioOutput::empty(24_000)),
                text: Some(String::new()),
                input_transcription: None,
                tokens_processed: step.tokens_processed,
                tokens_generated: 0,
                finished: false,
                phase_timing_override: None,
                asr_diagnostics: None,
                error: None,
            });
        }

        if matches!(&active_state.state, ActiveChatState::Prefilling(_)) {
            return Err(Error::InferenceError(
                "Qwen3.5 CUDA continuation prefill was still incomplete during decode scheduling"
                    .to_string(),
            ));
        }

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
            let step = match &mut active_state.state {
                ActiveChatState::Decoding(state) => {
                    Self::run_blocking(|| model.decode_step(state))?
                }
                ActiveChatState::Prefilling(_) => {
                    return Err(Error::InferenceError(
                        "Chat decode state is still prefilling".to_string(),
                    ));
                }
            };
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
                }
                if step.finished {
                    Self::stream_final_marker_with_policy(
                        tx,
                        stream_policy,
                        &request.id,
                        &mut active_state.stream_sequence,
                    )?;
                }
            }

            if step.finished {
                finished = true;
                break;
            }
        }

        let mut tokens_processed = if scheduled.is_prefill {
            scheduled.num_tokens.max(1)
        } else {
            decode_steps_ran.max(1)
        };
        if !active_state.prompt_accounted {
            active_state.prompt_accounted = true;
            tokens_processed = tokens_processed.saturating_add(request.num_prompt_tokens());
        }

        if !finished {
            let mut guard = self.chat_decode_states.lock().map_err(|_| {
                Error::InferenceError("Chat decode state mutex poisoned".to_string())
            })?;
            guard.insert(request.id.clone(), active_state);
        }

        Ok(ExecutorOutput {
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::GenerationParams;
    use crate::engine::scheduler::ScheduledRequest;
    use crate::models::shared::chat::{ChatMessage, ChatRole};

    fn scheduled_chat_prefill(num_tokens: usize) -> ScheduledRequest {
        ScheduledRequest {
            request_id: "req-chat".to_string(),
            sequence_id: 1,
            num_tokens,
            is_prefill: true,
            block_ids: Vec::new(),
            num_computed_tokens: 0,
        }
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
    fn qwen35_cuda_continuation_prefill_decision_is_cuda_only_and_chunked() {
        let scheduled = scheduled_chat_prefill(512);

        assert!(
            !NativeExecutor::should_use_qwen35_cuda_continuation_prefill(
                BackendKind::Cpu,
                true,
                &scheduled,
                4096,
                false,
            )
        );
        assert!(
            !NativeExecutor::should_use_qwen35_cuda_continuation_prefill(
                BackendKind::Metal,
                true,
                &scheduled,
                4096,
                false,
            )
        );
        assert!(
            !NativeExecutor::should_use_qwen35_cuda_continuation_prefill(
                BackendKind::Cuda,
                false,
                &scheduled,
                4096,
                false,
            )
        );
        assert!(NativeExecutor::should_use_qwen35_cuda_continuation_prefill(
            BackendKind::Cuda,
            true,
            &scheduled,
            4096,
            false,
        ));
    }

    #[test]
    fn qwen35_cuda_continuation_prefill_keeps_active_prefill_state() {
        let scheduled = scheduled_chat_prefill(4096);

        assert!(
            !NativeExecutor::should_use_qwen35_cuda_continuation_prefill(
                BackendKind::Cuda,
                true,
                &scheduled,
                4096,
                false,
            )
        );
        assert!(NativeExecutor::should_use_qwen35_cuda_continuation_prefill(
            BackendKind::Cuda,
            true,
            &scheduled,
            4096,
            true,
        ));
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
}
