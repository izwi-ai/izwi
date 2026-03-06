//! Chat runtime methods routed through the unified core engine.

use crate::engine::EngineCoreRequest;
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::shared::chat::ChatMessage;
use crate::runtime::service::RuntimeService;
use crate::runtime::types::ChatGeneration;

impl RuntimeService {
    async fn build_chat_request(
        &self,
        variant: ModelVariant,
        messages: Vec<ChatMessage>,
        max_new_tokens: usize,
        correlation_id: Option<&str>,
    ) -> Result<EngineCoreRequest> {
        self.load_model(variant).await?;

        let prompt_tokens = self
            .model_registry
            .get_chat(variant)
            .await
            .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?
            .prompt_token_ids(&messages)?;

        let mut request = EngineCoreRequest::chat(messages);
        request.model_variant = Some(variant);
        request.params.max_tokens = max_new_tokens.max(1);
        request.correlation_id = correlation_id.map(|s| s.to_string());
        request.prompt_tokens = prompt_tokens;
        Ok(request)
    }

    pub async fn chat_generate(
        &self,
        variant: ModelVariant,
        messages: Vec<ChatMessage>,
        max_new_tokens: usize,
    ) -> Result<ChatGeneration> {
        self.chat_generate_with_correlation(variant, messages, max_new_tokens, None)
            .await
    }

    pub async fn chat_generate_with_correlation(
        &self,
        variant: ModelVariant,
        messages: Vec<ChatMessage>,
        max_new_tokens: usize,
        correlation_id: Option<&str>,
    ) -> Result<ChatGeneration> {
        let request = self
            .build_chat_request(variant, messages, max_new_tokens, correlation_id)
            .await?;
        let output = self.run_request(request).await?;
        Ok(ChatGeneration {
            text: output.text.unwrap_or_default(),
            prompt_tokens: output.token_stats.prompt_tokens,
            tokens_generated: output.num_tokens,
            generation_time_ms: output.generation_time.as_secs_f64() * 1000.0,
        })
    }

    pub async fn chat_generate_streaming<F>(
        &self,
        variant: ModelVariant,
        messages: Vec<ChatMessage>,
        max_new_tokens: usize,
        on_delta: F,
    ) -> Result<ChatGeneration>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.chat_generate_streaming_with_correlation(
            variant,
            messages,
            max_new_tokens,
            None,
            on_delta,
        )
        .await
    }

    pub async fn chat_generate_streaming_with_correlation<F>(
        &self,
        variant: ModelVariant,
        messages: Vec<ChatMessage>,
        max_new_tokens: usize,
        correlation_id: Option<&str>,
        mut on_delta: F,
    ) -> Result<ChatGeneration>
    where
        F: FnMut(String) + Send + 'static,
    {
        let request = self
            .build_chat_request(variant, messages, max_new_tokens, correlation_id)
            .await?;
        let mut streamed_text = String::new();
        let output = self
            .run_streaming_request(request, |chunk| {
                if let Some(delta) = chunk.text {
                    if !delta.is_empty() {
                        streamed_text.push_str(&delta);
                        on_delta(delta);
                    }
                }
                std::future::ready(Ok(()))
            })
            .await?;

        Ok(ChatGeneration {
            text: output.text.unwrap_or(streamed_text),
            prompt_tokens: output.token_stats.prompt_tokens,
            tokens_generated: output.num_tokens,
            generation_time_ms: output.generation_time.as_secs_f64() * 1000.0,
        })
    }
}
