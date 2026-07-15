//! Chat runtime methods routed through the unified core engine.

use crate::catalog::ModelFamily;
use crate::engine::{GenerationParams, TaskType};
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::architectures::qwen35::media_resource_estimate;
use crate::models::shared::chat::{ChatGenerationConfig, ChatMessage, ChatRequestConfig};
use crate::runtime::request::ChatRuntimeRequest;
use crate::runtime::service::{
    media_preparation_resources, retained_chat_preparation_input_bytes, AdmittedEngineRequest,
    RuntimeService,
};
use crate::runtime::types::{ChatGeneration, RuntimeRequestContext};

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
        runtime_context: RuntimeRequestContext,
        streaming: bool,
    ) -> Result<AdmittedEngineRequest> {
        if messages.is_empty() {
            return Err(Error::InvalidInput(
                "Chat request missing messages".to_string(),
            ));
        }
        if !chat_config.media_inputs.is_empty() && variant.family() != ModelFamily::Qwen35Chat {
            return Err(Error::InvalidInput(format!(
                "Chat model {variant} does not support Qwen3.5 media inputs"
            )));
        }
        let correlation_id = correlation_id.map(ToOwned::to_owned);
        let input_bytes = retained_chat_preparation_input_bytes(
            &messages,
            messages.capacity(),
            &chat_config,
            &params,
            correlation_id.as_ref(),
        )?;
        let media_estimate = media_resource_estimate(&chat_config.media_inputs)?;
        let media_resources = media_preparation_resources(
            self.backend_router.context().backend_kind,
            media_estimate,
        )?;
        self.prepare_engine_request_blocking(
            variant,
            TaskType::Chat,
            streaming,
            runtime_context,
            input_bytes,
            media_resources,
            move |registry| {
                let prompt_config = Self::prompt_token_config(&params, &chat_config);
                let model = registry
                    .blocking_get_chat(variant)
                    .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?;
                let (prompt_tokens, prepared_qwen35_prompt) =
                    model.prepare_prompt_for_execution(&messages, &prompt_config)?;

                params.max_tokens = params.max_tokens.max(1);
                let mut request = ChatRuntimeRequest::from_messages(
                    variant,
                    messages,
                    params,
                    chat_config,
                    prompt_tokens,
                    correlation_id,
                    runtime_context,
                )?
                .into_engine_request();
                let exact_prompt_tokens = std::mem::take(&mut request.prompt_tokens);
                request.install_chat_execution_preparation_with_model(
                    variant,
                    exact_prompt_tokens,
                    prepared_qwen35_prompt,
                    model,
                )?;
                Ok(request)
            },
        )
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
        self.chat_generate_with_correlation_and_runtime_context(
            variant,
            messages,
            max_new_tokens,
            correlation_id,
            RuntimeRequestContext::default(),
        )
        .await
    }

    pub async fn chat_generate_with_correlation_and_runtime_context(
        &self,
        variant: ModelVariant,
        messages: Vec<ChatMessage>,
        max_new_tokens: usize,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
    ) -> Result<ChatGeneration> {
        let mut params = GenerationParams::default();
        params.max_tokens = max_new_tokens.max(1);
        let admitted = self
            .build_chat_request_with_params_and_config(
                variant,
                messages,
                params,
                ChatRequestConfig::default(),
                correlation_id,
                runtime_context,
                false,
            )
            .await?;
        let output = self.run_admitted_request(admitted).await?;
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
        self.chat_generate_with_runtime_context(
            variant,
            messages,
            params,
            chat_config,
            correlation_id,
            RuntimeRequestContext::default(),
        )
        .await
    }

    pub async fn chat_generate_with_runtime_context(
        &self,
        variant: ModelVariant,
        messages: Vec<ChatMessage>,
        params: GenerationParams,
        chat_config: ChatRequestConfig,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
    ) -> Result<ChatGeneration> {
        let admitted = self
            .build_chat_request_with_params_and_config(
                variant,
                messages,
                params,
                chat_config,
                correlation_id,
                runtime_context,
                false,
            )
            .await?;
        let output = self.run_admitted_request(admitted).await?;
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
        on_delta: F,
    ) -> Result<ChatGeneration>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.chat_generate_streaming_with_correlation_and_runtime_context(
            variant,
            messages,
            max_new_tokens,
            correlation_id,
            RuntimeRequestContext::default(),
            on_delta,
        )
        .await
    }

    pub async fn chat_generate_streaming_with_correlation_and_runtime_context<F>(
        &self,
        variant: ModelVariant,
        messages: Vec<ChatMessage>,
        max_new_tokens: usize,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
        mut on_delta: F,
    ) -> Result<ChatGeneration>
    where
        F: FnMut(String) + Send + 'static,
    {
        let mut params = GenerationParams::default();
        params.max_tokens = max_new_tokens.max(1);
        let admitted = self
            .build_chat_request_with_params_and_config(
                variant,
                messages,
                params,
                ChatRequestConfig::default(),
                correlation_id,
                runtime_context,
                true,
            )
            .await?;
        let mut streamed_text = String::new();
        let output = self
            .run_admitted_streaming_request(admitted, |chunk| {
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
        on_delta: F,
    ) -> Result<ChatGeneration>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.chat_generate_streaming_with_runtime_context(
            variant,
            messages,
            params,
            chat_config,
            correlation_id,
            RuntimeRequestContext::default(),
            on_delta,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn chat_generate_streaming_with_runtime_context<F>(
        &self,
        variant: ModelVariant,
        messages: Vec<ChatMessage>,
        params: GenerationParams,
        chat_config: ChatRequestConfig,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
        mut on_delta: F,
    ) -> Result<ChatGeneration>
    where
        F: FnMut(String) + Send + 'static,
    {
        let admitted = self
            .build_chat_request_with_params_and_config(
                variant,
                messages,
                params,
                chat_config,
                correlation_id,
                runtime_context,
                true,
            )
            .await?;
        let mut streamed_text = String::new();
        let output = self
            .run_admitted_streaming_request(admitted, |chunk| {
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
