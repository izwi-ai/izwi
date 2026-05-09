//! Chat runtime methods routed through the unified core engine.

use crate::engine::EngineCoreRequest;
use crate::engine::GenerationParams;
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::shared::chat::{ChatGenerationConfig, ChatMessage, ChatRequestConfig};
use crate::runtime::request::ChatRuntimeRequest;
use crate::runtime::service::RuntimeService;
use crate::runtime::types::ChatGeneration;

impl RuntimeService {
    fn prompt_token_config(
        params: &GenerationParams,
        chat_config: &ChatRequestConfig,
    ) -> ChatGenerationConfig {
        ChatGenerationConfig {
            temperature: params.temperature.max(0.0),
            top_p: params.top_p.clamp(0.0, 1.0),
            top_k: params.top_k,
            repetition_penalty: params.repetition_penalty.max(1.0),
            presence_penalty: params.presence_penalty.clamp(-2.0, 2.0),
            stop_token_ids: params.stop_token_ids.clone(),
            seed: 0,
            request: chat_config.clone(),
        }
    }

    async fn build_chat_request_with_params_and_config(
        &self,
        variant: ModelVariant,
        messages: Vec<ChatMessage>,
        mut params: GenerationParams,
        chat_config: ChatRequestConfig,
        correlation_id: Option<&str>,
    ) -> Result<EngineCoreRequest> {
        self.load_model(variant).await?;
        let _lease = self.acquire_model_residency_lease(variant);

        let prompt_config = Self::prompt_token_config(&params, &chat_config);

        let prompt_tokens = self
            .model_registry
            .get_chat(variant)
            .await
            .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?
            .prompt_token_ids_with_config(&messages, &prompt_config)?;

        params.max_tokens = params.max_tokens.max(1);
        Ok(ChatRuntimeRequest::from_messages(
            variant,
            messages,
            params,
            chat_config,
            prompt_tokens,
            correlation_id.map(ToOwned::to_owned),
        )?
        .into_engine_request())
    }

    async fn build_chat_request_with_params(
        &self,
        variant: ModelVariant,
        messages: Vec<ChatMessage>,
        params: GenerationParams,
        correlation_id: Option<&str>,
    ) -> Result<EngineCoreRequest> {
        self.build_chat_request_with_params_and_config(
            variant,
            messages,
            params,
            ChatRequestConfig::default(),
            correlation_id,
        )
        .await
    }

    async fn build_chat_request(
        &self,
        variant: ModelVariant,
        messages: Vec<ChatMessage>,
        max_new_tokens: usize,
        correlation_id: Option<&str>,
    ) -> Result<EngineCoreRequest> {
        let mut params = GenerationParams::default();
        params.max_tokens = max_new_tokens.max(1);
        self.build_chat_request_with_params(variant, messages, params, correlation_id)
            .await
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

    pub async fn chat_generate_with_generation_params(
        &self,
        variant: ModelVariant,
        messages: Vec<ChatMessage>,
        params: GenerationParams,
    ) -> Result<ChatGeneration> {
        self.chat_generate_with_generation_params_and_correlation(variant, messages, params, None)
            .await
    }

    pub async fn chat_generate_with_generation_params_and_correlation(
        &self,
        variant: ModelVariant,
        messages: Vec<ChatMessage>,
        params: GenerationParams,
        correlation_id: Option<&str>,
    ) -> Result<ChatGeneration> {
        self.chat_generate_with_generation_params_and_chat_config_and_correlation(
            variant,
            messages,
            params,
            ChatRequestConfig::default(),
            correlation_id,
        )
        .await
    }

    pub async fn chat_generate_with_generation_params_and_chat_config_and_correlation(
        &self,
        variant: ModelVariant,
        messages: Vec<ChatMessage>,
        params: GenerationParams,
        chat_config: ChatRequestConfig,
        correlation_id: Option<&str>,
    ) -> Result<ChatGeneration> {
        let request = self
            .build_chat_request_with_params_and_config(
                variant,
                messages,
                params,
                chat_config,
                correlation_id,
            )
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

    pub async fn chat_generate_streaming_with_generation_params<F>(
        &self,
        variant: ModelVariant,
        messages: Vec<ChatMessage>,
        params: GenerationParams,
        on_delta: F,
    ) -> Result<ChatGeneration>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.chat_generate_streaming_with_generation_params_and_correlation(
            variant, messages, params, None, on_delta,
        )
        .await
    }

    pub async fn chat_generate_streaming_with_generation_params_and_correlation<F>(
        &self,
        variant: ModelVariant,
        messages: Vec<ChatMessage>,
        params: GenerationParams,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<ChatGeneration>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.chat_generate_streaming_with_generation_params_and_chat_config_and_correlation(
            variant,
            messages,
            params,
            ChatRequestConfig::default(),
            correlation_id,
            on_delta,
        )
        .await
    }

    pub async fn chat_generate_streaming_with_generation_params_and_chat_config_and_correlation<F>(
        &self,
        variant: ModelVariant,
        messages: Vec<ChatMessage>,
        params: GenerationParams,
        chat_config: ChatRequestConfig,
        correlation_id: Option<&str>,
        mut on_delta: F,
    ) -> Result<ChatGeneration>
    where
        F: FnMut(String) + Send + 'static,
    {
        let request = self
            .build_chat_request_with_params_and_config(
                variant,
                messages,
                params,
                chat_config,
                correlation_id,
            )
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
