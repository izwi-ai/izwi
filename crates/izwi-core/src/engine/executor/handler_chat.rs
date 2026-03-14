use crate::error::{Error, Result};
use crate::models::shared::chat::ChatGenerationConfig;
use crate::models::shared::chat::ChatMessage;

use super::super::request::EngineCoreRequest;
use super::super::scheduler::ScheduledRequest;
use super::super::types::AudioOutput;
use super::state::ActiveChatDecode;
use super::{ExecutorOutput, NativeExecutor};

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

    pub(super) fn chat_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
    ) -> Result<ExecutorOutput> {
        let variant = Self::resolve_variant(request)?;
        let messages = Self::chat_messages(request)?;
        let max_new_tokens = request.params.max_tokens.max(1);
        let stream_tx = Self::stream_sender(request);
        let generation_config = Self::chat_generation_config(request);

        let model = self.with_registry(|registry| {
            registry
                .try_get_chat(variant)
                .ok_or_else(|| Error::ModelNotFound(format!("Chat model {variant} is not loaded")))
        })?;

        // Fallback path for chat backends that do not expose incremental decode state.
        if !model.supports_incremental_decode() {
            let output = Self::run_blocking(|| {
                if let Some(tx) = stream_tx.as_ref() {
                    let mut sequence = 0usize;
                    let mut stream_err: Option<Error> = None;
                    let mut emit = |delta: &str| {
                        if stream_err.is_none() {
                            if let Err(err) =
                                Self::stream_text(tx, &request.id, &mut sequence, delta.to_string())
                            {
                                stream_err = Some(err);
                            }
                        }
                    };
                    let output = model.generate_with_callback_and_config(
                        messages,
                        max_new_tokens,
                        &generation_config,
                        &mut emit,
                    )?;
                    if let Some(err) = stream_err {
                        return Err(err);
                    }
                    Self::stream_final_marker(tx, &request.id, &mut sequence)?;
                    Ok(output)
                } else {
                    model.generate_with_config(messages, max_new_tokens, &generation_config)
                }
            })?;

            return Ok(ExecutorOutput {
                request_id: request.id.clone(),
                audio: Some(AudioOutput::empty(24_000)),
                text: Some(output.text),
                input_transcription: None,
                tokens_processed: request.num_prompt_tokens(),
                tokens_generated: output.tokens_generated.max(1),
                finished: true,
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

        let mut active_state = if let Some(state) = active_state {
            state
        } else {
            let decode_state = Self::run_blocking(|| {
                model.start_decode_state_with_config(messages, max_new_tokens, &generation_config)
            })?;
            ActiveChatDecode {
                variant,
                state: decode_state,
                prompt_accounted: false,
                last_tokens_generated: 0,
                stream_sequence: 0,
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
            let step = Self::run_blocking(|| model.decode_step(&mut active_state.state))?;
            decode_steps_ran = decode_steps_ran.saturating_add(1);

            let step_tokens_generated = step
                .tokens_generated
                .saturating_sub(active_state.last_tokens_generated);
            active_state.last_tokens_generated = step.tokens_generated;
            total_tokens_generated = total_tokens_generated.saturating_add(step_tokens_generated);
            final_text = step.text.clone();

            if let Some(tx) = stream_tx.as_ref() {
                if !step.delta.is_empty() {
                    Self::stream_text(
                        tx,
                        &request.id,
                        &mut active_state.stream_sequence,
                        step.delta.clone(),
                    )?;
                }
                if step.finished {
                    Self::stream_final_marker(tx, &request.id, &mut active_state.stream_sequence)?;
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
            error: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::GenerationParams;
    use crate::models::shared::chat::{ChatMessage, ChatRole};

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
}
