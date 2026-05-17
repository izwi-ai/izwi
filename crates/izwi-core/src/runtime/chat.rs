//! Chat runtime methods routed through the unified core engine.

use crate::backends::BackendKind;
use crate::catalog::ModelFamily;
use crate::engine::EngineCoreRequest;
use crate::engine::GenerationParams;
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::shared::chat::{ChatGenerationConfig, ChatMessage, ChatRequestConfig};
use crate::runtime::request::ChatRuntimeRequest;
use crate::runtime::service::RuntimeService;
use crate::runtime::types::ChatGeneration;

const DEFAULT_QWEN35_CUDA_SAFE_PREFILL_TOKENS: usize = 8192;
const DEFAULT_QWEN35_CUDA_PREFILL_CHUNK_TOKENS: usize = 4096;
const QWEN35_CUDA_SAFE_PREFILL_ENV: &str = "IZWI_QWEN35_CUDA_SAFE_PREFILL_TOKENS";
const QWEN35_CUDA_PREFILL_CHUNK_ENV: &str = "IZWI_QWEN35_CUDA_PREFILL_CHUNK_TOKENS";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35CudaContextBudget {
    pub total_context_tokens: usize,
    pub prefill_chunk_tokens: usize,
    pub runtime_max_sequence_length: usize,
    pub model_context_length: Option<usize>,
}

impl RuntimeService {
    pub fn qwen35_cuda_prefill_chunk_tokens(&self) -> usize {
        qwen35_cuda_prefill_chunk_tokens()
    }

    pub fn qwen35_cuda_safe_prefill_tokens(&self) -> usize {
        qwen35_cuda_safe_prefill_tokens()
    }

    pub fn qwen35_cuda_context_budget(
        &self,
        model_context_length: Option<usize>,
    ) -> Qwen35CudaContextBudget {
        qwen35_cuda_context_budget(self.config.max_sequence_length, model_context_length)
    }

    pub async fn qwen35_cuda_context_budget_for_variant(
        &self,
        variant: ModelVariant,
    ) -> Result<Option<Qwen35CudaContextBudget>> {
        if self.backend_context().backend_kind != BackendKind::Cuda
            || variant.family() != ModelFamily::Qwen35Chat
        {
            return Ok(None);
        }

        self.load_model(variant).await?;
        let _lease = self.acquire_model_residency_lease(variant);
        let chat_model = self
            .model_registry
            .get_chat(variant)
            .await
            .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?;
        Ok(Some(self.qwen35_cuda_context_budget(
            chat_model.qwen35_context_length_hint(),
        )))
    }

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

        let chat_model = self
            .model_registry
            .get_chat(variant)
            .await
            .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?;
        let prompt_tokens = chat_model.prompt_token_ids_with_config(&messages, &prompt_config)?;

        params.max_tokens = params.max_tokens.max(1);
        self.validate_qwen35_cuda_context_budget(
            variant,
            prompt_tokens.len(),
            params.max_tokens,
            chat_model.qwen35_context_length_hint(),
        )?;
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

    pub async fn chat_prompt_token_count_with_generation_params_and_chat_config(
        &self,
        variant: ModelVariant,
        messages: &[ChatMessage],
        params: &GenerationParams,
        chat_config: &ChatRequestConfig,
    ) -> Result<usize> {
        self.load_model(variant).await?;
        let _lease = self.acquire_model_residency_lease(variant);

        let prompt_config = Self::prompt_token_config(params, chat_config);
        let chat_model = self
            .model_registry
            .get_chat(variant)
            .await
            .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?;
        chat_model
            .prompt_token_ids_with_config(messages, &prompt_config)
            .map(|tokens| tokens.len())
    }

    pub async fn chat_prompt_token_count_with_generation_params(
        &self,
        variant: ModelVariant,
        messages: &[ChatMessage],
        params: &GenerationParams,
    ) -> Result<usize> {
        self.chat_prompt_token_count_with_generation_params_and_chat_config(
            variant,
            messages,
            params,
            &ChatRequestConfig::default(),
        )
        .await
    }

    fn validate_qwen35_cuda_context_budget(
        &self,
        variant: ModelVariant,
        prompt_tokens: usize,
        max_new_tokens: usize,
        model_context_length: Option<usize>,
    ) -> Result<()> {
        if self.backend_context().backend_kind != BackendKind::Cuda
            || variant.family() != ModelFamily::Qwen35Chat
        {
            return Ok(());
        }

        let limit = validate_qwen35_cuda_context_budget(
            prompt_tokens,
            max_new_tokens,
            self.config.max_sequence_length,
            model_context_length,
        )?;
        tracing::debug!(
            target: "izwi.qwen35",
            variant = %variant.display_name(),
            prompt_tokens,
            max_new_tokens = max_new_tokens.max(1),
            runtime_max_sequence_length = self.config.max_sequence_length.max(1),
            model_context_length = model_context_length.unwrap_or_default(),
            limit_tokens = limit,
            safe_prefill_tokens = qwen35_cuda_safe_prefill_tokens(),
            prefill_chunk_tokens = qwen35_cuda_prefill_chunk_tokens(),
            "Validated Qwen3.5 CUDA chat prompt budget"
        );
        Ok(())
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

fn qwen35_cuda_safe_prefill_tokens() -> usize {
    std::env::var(QWEN35_CUDA_SAFE_PREFILL_ENV)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_QWEN35_CUDA_SAFE_PREFILL_TOKENS)
}

pub(super) fn qwen35_cuda_prefill_chunk_tokens() -> usize {
    std::env::var(QWEN35_CUDA_PREFILL_CHUNK_ENV)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .or_else(|| {
            std::env::var(QWEN35_CUDA_SAFE_PREFILL_ENV)
                .ok()
                .and_then(|raw| raw.trim().parse::<usize>().ok())
                .filter(|value| *value > 0)
        })
        .unwrap_or(DEFAULT_QWEN35_CUDA_PREFILL_CHUNK_TOKENS)
}

fn qwen35_cuda_context_budget_limit(
    runtime_max_sequence_length: usize,
    model_context_length: Option<usize>,
) -> usize {
    let runtime_limit = runtime_max_sequence_length.max(1);
    model_context_length
        .filter(|limit| *limit > 0)
        .map(|model_limit| runtime_limit.min(model_limit))
        .unwrap_or(runtime_limit)
}

fn qwen35_cuda_context_budget(
    runtime_max_sequence_length: usize,
    model_context_length: Option<usize>,
) -> Qwen35CudaContextBudget {
    let total_context_tokens =
        qwen35_cuda_context_budget_limit(runtime_max_sequence_length, model_context_length);
    let prefill_chunk_tokens = qwen35_cuda_prefill_chunk_tokens()
        .max(1)
        .min(total_context_tokens.max(1));
    Qwen35CudaContextBudget {
        total_context_tokens,
        prefill_chunk_tokens,
        runtime_max_sequence_length: runtime_max_sequence_length.max(1),
        model_context_length,
    }
}

fn validate_qwen35_cuda_context_budget(
    prompt_tokens: usize,
    max_new_tokens: usize,
    runtime_max_sequence_length: usize,
    model_context_length: Option<usize>,
) -> Result<usize> {
    let max_new_tokens = max_new_tokens.max(1);
    let requested_tokens = prompt_tokens.saturating_add(max_new_tokens);
    let limit = qwen35_cuda_context_budget_limit(runtime_max_sequence_length, model_context_length);
    if requested_tokens <= limit {
        return Ok(limit);
    }

    Err(Error::InvalidInput(format!(
        "Qwen3.5 CUDA prompt budget exceeded: prompt_tokens={prompt_tokens}, \
         max_new_tokens={max_new_tokens}, requested_tokens={requested_tokens}, \
         limit_tokens={limit}, runtime_max_sequence_length={}, model_context_length={}. \
         Reduce the prompt/max tokens or use hierarchical chunking for long summaries.",
        runtime_max_sequence_length.max(1),
        model_context_length.unwrap_or_default()
    )))
}

#[cfg(test)]
mod tests {
    use super::{
        QWEN35_CUDA_PREFILL_CHUNK_ENV, QWEN35_CUDA_SAFE_PREFILL_ENV, qwen35_cuda_context_budget,
        qwen35_cuda_context_budget_limit, qwen35_cuda_prefill_chunk_tokens,
        qwen35_cuda_safe_prefill_tokens, validate_qwen35_cuda_context_budget,
    };
    use crate::error::Error;

    #[test]
    fn qwen35_cuda_context_budget_uses_runtime_and_model_minimum() {
        assert_eq!(qwen35_cuda_context_budget_limit(4096, Some(262_144)), 4096);
        assert_eq!(qwen35_cuda_context_budget_limit(262_144, Some(4096)), 4096);
        assert_eq!(qwen35_cuda_context_budget_limit(8192, None), 8192);
        assert_eq!(qwen35_cuda_context_budget_limit(0, Some(0)), 1);
    }

    #[test]
    fn qwen35_cuda_context_budget_rejects_oversized_prompt_before_prefill() {
        let err = validate_qwen35_cuda_context_budget(4090, 16, 4096, Some(262_144))
            .expect_err("prompt should exceed runtime limit");
        assert!(matches!(
            err,
            Error::InvalidInput(message)
                if message.contains("Qwen3.5 CUDA prompt budget exceeded")
                    && message.contains("requested_tokens=4106")
                    && message.contains("limit_tokens=4096")
        ));
    }

    #[test]
    fn qwen35_cuda_context_budget_accepts_prompt_within_limit() {
        let limit = validate_qwen35_cuda_context_budget(4000, 96, 4096, Some(262_144))
            .expect("prompt fits");
        assert_eq!(limit, 4096);
    }

    #[test]
    fn qwen35_cuda_context_budget_separates_total_context_from_prefill_chunk() {
        let _guard = crate::env_test_lock().lock().expect("env lock");
        std::env::set_var(QWEN35_CUDA_PREFILL_CHUNK_ENV, "2048");

        let budget = qwen35_cuda_context_budget(65_536, Some(262_144));

        assert_eq!(budget.total_context_tokens, 65_536);
        assert_eq!(budget.prefill_chunk_tokens, 2048);
        assert_eq!(budget.runtime_max_sequence_length, 65_536);
        assert_eq!(budget.model_context_length, Some(262_144));

        std::env::remove_var(QWEN35_CUDA_PREFILL_CHUNK_ENV);
    }

    #[test]
    fn qwen35_cuda_prefill_chunk_tokens_are_clamped_to_total_context_budget() {
        let _guard = crate::env_test_lock().lock().expect("env lock");
        std::env::set_var(QWEN35_CUDA_PREFILL_CHUNK_ENV, "8192");

        let budget = qwen35_cuda_context_budget(4096, Some(262_144));

        assert_eq!(budget.total_context_tokens, 4096);
        assert_eq!(budget.prefill_chunk_tokens, 4096);

        std::env::remove_var(QWEN35_CUDA_PREFILL_CHUNK_ENV);
    }

    #[test]
    fn qwen35_cuda_prefill_chunk_tokens_use_new_env_before_legacy_safe_env() {
        let _guard = crate::env_test_lock().lock().expect("env lock");
        std::env::remove_var(QWEN35_CUDA_PREFILL_CHUNK_ENV);
        std::env::remove_var(QWEN35_CUDA_SAFE_PREFILL_ENV);
        assert_eq!(qwen35_cuda_prefill_chunk_tokens(), 4096);

        std::env::set_var(QWEN35_CUDA_SAFE_PREFILL_ENV, "8192");
        assert_eq!(qwen35_cuda_prefill_chunk_tokens(), 8192);

        std::env::set_var(QWEN35_CUDA_PREFILL_CHUNK_ENV, "2048");
        assert_eq!(qwen35_cuda_prefill_chunk_tokens(), 2048);

        std::env::set_var(QWEN35_CUDA_PREFILL_CHUNK_ENV, "0");
        assert_eq!(qwen35_cuda_prefill_chunk_tokens(), 8192);

        std::env::remove_var(QWEN35_CUDA_PREFILL_CHUNK_ENV);
        std::env::remove_var(QWEN35_CUDA_SAFE_PREFILL_ENV);
    }

    #[test]
    fn qwen35_cuda_safe_prefill_tokens_uses_positive_env_override() {
        let _guard = crate::env_test_lock().lock().expect("env lock");

        std::env::remove_var(QWEN35_CUDA_PREFILL_CHUNK_ENV);
        std::env::remove_var(QWEN35_CUDA_SAFE_PREFILL_ENV);
        assert_eq!(qwen35_cuda_safe_prefill_tokens(), 8192);

        std::env::set_var(QWEN35_CUDA_SAFE_PREFILL_ENV, "16384");
        assert_eq!(qwen35_cuda_safe_prefill_tokens(), 16_384);

        std::env::set_var(QWEN35_CUDA_SAFE_PREFILL_ENV, "0");
        assert_eq!(qwen35_cuda_safe_prefill_tokens(), 8192);

        std::env::set_var(QWEN35_CUDA_SAFE_PREFILL_ENV, "not-a-number");
        assert_eq!(qwen35_cuda_safe_prefill_tokens(), 8192);

        std::env::remove_var(QWEN35_CUDA_SAFE_PREFILL_ENV);
    }
}
