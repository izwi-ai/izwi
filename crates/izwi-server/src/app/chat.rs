use std::time::Duration;

use tokio::sync::mpsc;

use crate::error::ApiError;
use crate::state::AppState;
use izwi_core::{parse_chat_model_variant, ChatGeneration, ChatMessage, ModelVariant};

#[derive(Debug, Clone)]
pub struct ChatExecutionRequest {
    pub variant: ModelVariant,
    pub messages: Vec<ChatMessage>,
    pub max_completion_tokens: Option<usize>,
    pub max_tokens: Option<usize>,
    pub correlation_id: Option<String>,
}

impl ChatExecutionRequest {
    fn resolved_max_new_tokens(&self) -> usize {
        max_new_tokens(self.variant, self.max_completion_tokens, self.max_tokens)
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
    let max_new_tokens = request.resolved_max_new_tokens();
    let _permit = state.acquire_permit().await;

    state
        .runtime
        .chat_generate_with_correlation(
            request.variant,
            request.messages,
            max_new_tokens,
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
    let max_new_tokens = request.resolved_max_new_tokens();
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
                .chat_generate_streaming_with_correlation(
                    variant,
                    messages,
                    max_new_tokens,
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
