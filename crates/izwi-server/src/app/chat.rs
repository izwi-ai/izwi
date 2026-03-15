use std::{future::Future, sync::Arc};

use tokio::sync::mpsc;

use crate::error::ApiError;
use crate::state::AppState;
use izwi_core::{
    parse_chat_model_variant, ChatGeneration, ChatMessage, ChatRequestConfig, GenerationParams,
    ModelVariant,
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
    pub chat_config: ChatRequestConfig,
    pub correlation_id: Option<String>,
}

impl ChatExecutionRequest {
    fn resolved_max_new_tokens(&self) -> usize {
        max_new_tokens(self.variant, self.max_completion_tokens, self.max_tokens)
    }

    fn resolved_generation_params(&self) -> GenerationParams {
        let max_new_tokens = self.resolved_max_new_tokens();
        let mut params = GenerationParams::default();
        params.max_tokens = max_new_tokens;

        if let Some(temperature) = self.temperature {
            params.temperature = temperature;
        }
        if let Some(top_p) = self.top_p {
            params.top_p = top_p;
        }
        if let Some(presence_penalty) = self.presence_penalty {
            params.presence_penalty = presence_penalty;
        }
        params
    }

    fn resolved_chat_config(&self) -> ChatRequestConfig {
        self.chat_config.clone()
    }
}

#[derive(Debug, Clone)]
pub enum ChatStreamEvent {
    Started,
    Delta(String),
    Completed(ChatGeneration),
    Failed(String),
    ShuttingDown,
}

pub fn max_new_tokens(
    _variant: ModelVariant,
    max_completion_tokens: Option<usize>,
    max_tokens: Option<usize>,
) -> usize {
    let requested = max_completion_tokens.or(max_tokens);

    requested.unwrap_or(4096).clamp(1, 4096)
}

pub fn parse_chat_model(model_id: &str) -> Result<ModelVariant, ApiError> {
    parse_chat_model_variant(Some(model_id)).map_err(|err| ApiError::bad_request(err.to_string()))
}

pub async fn generate_chat(
    state: &AppState,
    request: ChatExecutionRequest,
) -> Result<ChatGeneration, ApiError> {
    let params = request.resolved_generation_params();
    let chat_config = request.resolved_chat_config();
    let variant = request.variant;
    let messages = request.messages;
    let correlation_id = request.correlation_id;
    let _permit = state.acquire_permit().await;

    state
        .runtime
        .chat_generate_with_generation_params_and_chat_config_and_correlation(
            variant,
            messages,
            params,
            chat_config,
            correlation_id.as_deref(),
        )
        .await
        .map_err(ApiError::from)
}

pub fn spawn_chat_stream(
    state: AppState,
    request: ChatExecutionRequest,
) -> mpsc::UnboundedReceiver<ChatStreamEvent> {
    let semaphore = state.request_semaphore.clone();
    let runtime = state.runtime.clone();
    let params = request.resolved_generation_params();
    let chat_config = request.resolved_chat_config();
    let variant = request.variant;
    let messages = request.messages;
    let correlation_id = request.correlation_id;

    // Streamed chat should be allowed to finish once generation starts; a hard
    // wall-clock timeout cuts off active responses mid-stream.
    spawn_chat_stream_with_task(semaphore, move |event_tx| async move {
        runtime
            .chat_generate_streaming_with_generation_params_and_chat_config_and_correlation(
                variant,
                messages,
                params,
                chat_config,
                correlation_id.as_deref(),
                move |delta| {
                    let _ = event_tx.send(ChatStreamEvent::Delta(delta));
                },
            )
            .await
            .map_err(|err| err.to_string())
    })
}

fn spawn_chat_stream_with_task<G, Fut>(
    semaphore: Arc<tokio::sync::Semaphore>,
    generation_task: G,
) -> mpsc::UnboundedReceiver<ChatStreamEvent>
where
    G: FnOnce(mpsc::UnboundedSender<ChatStreamEvent>) -> Fut + Send + 'static,
    Fut: Future<Output = Result<ChatGeneration, String>> + Send + 'static,
{
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

        match generation_task(event_tx.clone()).await {
            Ok(generation) => {
                let _ = event_tx.send(ChatStreamEvent::Completed(generation));
            }
            Err(err) => {
                let _ = event_tx.send(ChatStreamEvent::Failed(err));
            }
        }
    });

    event_rx
}

#[cfg(test)]
mod tests {
    use super::*;
    use izwi_core::ChatRole;
    use std::time::Duration;
    use tokio::sync::Semaphore;

    #[test]
    fn explicit_overrides_win_over_default_generation_params() {
        let request = ChatExecutionRequest {
            variant: ModelVariant::Qwen34BGguf,
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "hello".to_string(),
            }],
            max_completion_tokens: None,
            max_tokens: Some(32),
            temperature: Some(0.42),
            top_p: Some(0.73),
            presence_penalty: Some(0.25),
            chat_config: ChatRequestConfig::default(),
            correlation_id: None,
        };

        let params = request.resolved_generation_params();
        assert_eq!(params.temperature, 0.42);
        assert_eq!(params.top_p, 0.73);
        assert_eq!(params.top_k, 0);
        assert_eq!(params.presence_penalty, 0.25);
        assert_eq!(params.max_tokens, 32);
    }

    #[test]
    fn chat_models_default_to_4096_max_tokens_when_request_omits_limits() {
        for variant in [
            ModelVariant::Gemma34BIt,
            ModelVariant::Lfm2512BInstructGguf,
            ModelVariant::Qwen306BGguf,
            ModelVariant::Qwen317BGguf,
            ModelVariant::Qwen34BGguf,
            ModelVariant::Qwen38BGguf,
            ModelVariant::Qwen314BGguf,
            ModelVariant::Qwen352BGguf,
        ] {
            assert_eq!(max_new_tokens(variant, None, None), 4096);
        }
    }

    #[tokio::test]
    async fn streaming_chat_allows_long_running_generations_to_complete() {
        let semaphore = Arc::new(Semaphore::new(1));
        let mut event_rx = spawn_chat_stream_with_task(semaphore, |event_tx| async move {
            let _ = event_tx.send(ChatStreamEvent::Delta("Hello".to_string()));
            tokio::time::sleep(Duration::from_millis(25)).await;
            let _ = event_tx.send(ChatStreamEvent::Delta(" world".to_string()));
            Ok(ChatGeneration {
                text: "Hello world".to_string(),
                prompt_tokens: 12,
                tokens_generated: 2,
                generation_time_ms: 25.0,
            })
        });

        match event_rx.recv().await {
            Some(ChatStreamEvent::Started) => {}
            other => panic!("expected stream start event, got {other:?}"),
        }

        match event_rx.recv().await {
            Some(ChatStreamEvent::Delta(delta)) => assert_eq!(delta, "Hello"),
            other => panic!("expected first delta event, got {other:?}"),
        }

        match event_rx.recv().await {
            Some(ChatStreamEvent::Delta(delta)) => assert_eq!(delta, " world"),
            other => panic!("expected second delta event, got {other:?}"),
        }

        match event_rx.recv().await {
            Some(ChatStreamEvent::Completed(generation)) => {
                assert_eq!(generation.text, "Hello world");
                assert_eq!(generation.tokens_generated, 2);
            }
            other => panic!("expected completed event, got {other:?}"),
        }
    }
}
