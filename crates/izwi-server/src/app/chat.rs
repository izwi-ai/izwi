use std::time::Duration;

use tokio::sync::mpsc;

use crate::error::ApiError;
use crate::state::AppState;
use izwi_core::{
    parse_chat_model_variant, qwen35_recommended_generation_params, ChatGeneration, ChatMessage,
    GenerationParams, ModelVariant,
};

#[derive(Debug, Clone)]
pub struct ChatExecutionRequest {
    pub variant: ModelVariant,
    pub messages: Vec<ChatMessage>,
    pub max_completion_tokens: Option<usize>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub correlation_id: Option<String>,
}

impl ChatExecutionRequest {
    fn resolved_max_new_tokens(&self) -> usize {
        max_new_tokens(self.variant, self.max_completion_tokens, self.max_tokens)
    }

    fn resolved_generation_params(&self) -> GenerationParams {
        let max_new_tokens = self.resolved_max_new_tokens();
        let mut params =
            qwen35_recommended_generation_params(self.variant, &self.messages, max_new_tokens)
                .unwrap_or_else(|| {
                    let mut params = GenerationParams::default();
                    params.max_tokens = max_new_tokens;
                    params
                });

        if let Some(temperature) = self.temperature {
            params.temperature = temperature;
        }
        if let Some(top_p) = self.top_p {
            params.top_p = top_p;
        }
        if let Some(presence_penalty) = self.presence_penalty {
            params.presence_penalty = presence_penalty;
        }
        params.max_tokens = max_new_tokens;
        params
    }
}

#[derive(Debug, Clone)]
pub enum ChatStreamEvent {
    Started,
    Delta(String),
    Completed(ChatGeneration),
    Failed(String),
    TimedOut,
    ShuttingDown,
}

pub fn max_new_tokens(
    variant: ModelVariant,
    max_completion_tokens: Option<usize>,
    max_tokens: Option<usize>,
) -> usize {
    let requested = max_completion_tokens.or(max_tokens);

    let default = match variant {
        ModelVariant::Gemma34BIt => 4096,
        ModelVariant::Gemma31BIt => 4096,
        ModelVariant::Lfm2512BInstructGguf => 4096,
        ModelVariant::Lfm2512BThinkingGguf => 4096,
        ModelVariant::Qwen3508B => 4096,
        ModelVariant::Qwen352B => 4096,
        ModelVariant::Qwen354B => 4096,
        ModelVariant::Qwen359B => 4096,
        _ => 1536,
    };

    requested.unwrap_or(default).clamp(1, 4096)
}

pub fn parse_chat_model(model_id: &str) -> Result<ModelVariant, ApiError> {
    parse_chat_model_variant(Some(model_id)).map_err(|err| ApiError::bad_request(err.to_string()))
}

pub fn is_qwen35_chat_variant(variant: ModelVariant) -> bool {
    matches!(
        variant,
        ModelVariant::Qwen3508B
            | ModelVariant::Qwen352B
            | ModelVariant::Qwen354B
            | ModelVariant::Qwen359B
    )
}

pub async fn generate_chat(
    state: &AppState,
    request: ChatExecutionRequest,
) -> Result<ChatGeneration, ApiError> {
    let params = request.resolved_generation_params();
    let _permit = state.acquire_permit().await;

    state
        .runtime
        .chat_generate_with_generation_params_and_correlation(
            request.variant,
            request.messages,
            params,
            request.correlation_id.as_deref(),
        )
        .await
        .map_err(ApiError::from)
}

pub fn spawn_chat_stream(
    state: AppState,
    request: ChatExecutionRequest,
) -> mpsc::UnboundedReceiver<ChatStreamEvent> {
    let timeout = Duration::from_secs(state.request_timeout_secs);
    let semaphore = state.request_semaphore.clone();
    let runtime = state.runtime.clone();
    let params = request.resolved_generation_params();
    let variant = request.variant;
    let messages = request.messages;
    let correlation_id = request.correlation_id;

    let (event_tx, event_rx) = mpsc::unbounded_channel();

    tokio::spawn(async move {
        let _permit = match semaphore.acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => {
                let _ = event_tx.send(ChatStreamEvent::ShuttingDown);
                return;
            }
        };

        let _ = event_tx.send(ChatStreamEvent::Started);

        let delta_tx = event_tx.clone();
        let generation = tokio::time::timeout(timeout, async {
            runtime
                .chat_generate_streaming_with_generation_params_and_correlation(
                    variant,
                    messages,
                    params,
                    correlation_id.as_deref(),
                    move |delta| {
                        let _ = delta_tx.send(ChatStreamEvent::Delta(delta));
                    },
                )
                .await
        })
        .await;

        match generation {
            Ok(Ok(generation)) => {
                let _ = event_tx.send(ChatStreamEvent::Completed(generation));
            }
            Ok(Err(err)) => {
                let _ = event_tx.send(ChatStreamEvent::Failed(err.to_string()));
            }
            Err(_) => {
                let _ = event_tx.send(ChatStreamEvent::TimedOut);
            }
        }
    });

    event_rx
}

#[cfg(test)]
mod tests {
    use super::*;
    use izwi_core::{qwen35_thinking_control_content, ChatRole};

    fn user_message() -> Vec<ChatMessage> {
        vec![ChatMessage {
            role: ChatRole::User,
            content: "hello".to_string(),
        }]
    }

    #[test]
    fn qwen35_large_models_use_recommended_non_thinking_defaults() {
        let request = ChatExecutionRequest {
            variant: ModelVariant::Qwen354B,
            messages: user_message(),
            max_completion_tokens: Some(64),
            max_tokens: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            correlation_id: None,
        };

        let params = request.resolved_generation_params();
        assert_eq!(params.temperature, 0.7);
        assert_eq!(params.top_p, 0.8);
        assert_eq!(params.top_k, 20);
        assert_eq!(params.presence_penalty, 0.0);
        assert_eq!(params.max_tokens, 64);
    }

    #[test]
    fn qwen35_thinking_control_switches_to_thinking_defaults() {
        let request = ChatExecutionRequest {
            variant: ModelVariant::Qwen354B,
            messages: vec![
                ChatMessage {
                    role: ChatRole::System,
                    content: qwen35_thinking_control_content(true),
                },
                ChatMessage {
                    role: ChatRole::User,
                    content: "hello".to_string(),
                },
            ],
            max_completion_tokens: Some(64),
            max_tokens: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            correlation_id: None,
        };

        let params = request.resolved_generation_params();
        assert_eq!(params.temperature, 0.6);
        assert_eq!(params.top_p, 0.95);
        assert_eq!(params.top_k, 20);
        assert_eq!(params.presence_penalty, 0.0);
    }

    #[test]
    fn explicit_overrides_win_over_qwen35_defaults() {
        let request = ChatExecutionRequest {
            variant: ModelVariant::Qwen354B,
            messages: vec![
                ChatMessage {
                    role: ChatRole::System,
                    content: qwen35_thinking_control_content(false),
                },
                ChatMessage {
                    role: ChatRole::User,
                    content: "hello".to_string(),
                },
            ],
            max_completion_tokens: None,
            max_tokens: Some(32),
            temperature: Some(0.42),
            top_p: Some(0.73),
            presence_penalty: Some(0.25),
            correlation_id: None,
        };

        let params = request.resolved_generation_params();
        assert_eq!(params.temperature, 0.42);
        assert_eq!(params.top_p, 0.73);
        assert_eq!(params.top_k, 20);
        assert_eq!(params.presence_penalty, 0.25);
        assert_eq!(params.max_tokens, 32);
    }
}
