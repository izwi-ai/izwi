//! Chat runtime methods routed through the unified core engine.

use crate::catalog::ModelFamily;
use crate::engine::{resolve_backend_model_context, GenerationParams, TaskType};
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::architectures::qwen35::media_resource_estimate;
use crate::models::shared::chat::{
    ChatGenerationConfig, ChatMessage, ChatRequestConfig, ChatTruncation,
};
use crate::runtime::chat_request::CHAT_MAX_TOKENS_UNSET;
use crate::runtime::request::ChatRuntimeRequest;
use crate::runtime::service::{
    media_preparation_resources, retained_chat_preparation_input_bytes, AdmittedEngineRequest,
    RuntimeChatInvocation, RuntimeChatInvocationRequest, RuntimeService,
};
use crate::runtime::types::{ChatGeneration, RuntimeRequestContext};
use tracing::warn;

/// Return prefix endpoints for complete historical user/assistant turns while
/// preserving a leading system message and the newest user query. Candidate
/// prompts are still prepared by the loaded model; these boundaries only avoid
/// O(turns) repeated one-pair tokenization.
fn complete_chat_turn_prefix_ends(messages: &[ChatMessage]) -> Vec<usize> {
    let protected_prefix = usize::from(
        messages
            .first()
            .is_some_and(|message| message.role == crate::models::shared::chat::ChatRole::System),
    );
    let Some(last_user_index) = messages
        .iter()
        .rposition(|message| message.role == crate::models::shared::chat::ChatRole::User)
    else {
        return Vec::new();
    };
    let mut ends = Vec::new();
    let mut cursor = protected_prefix;
    while cursor < last_user_index {
        let Some(user_offset) = messages[cursor..last_user_index]
            .iter()
            .position(|message| message.role == crate::models::shared::chat::ChatRole::User)
        else {
            break;
        };
        let user_index = cursor + user_offset;
        let Some(assistant_offset) = messages[(user_index + 1)..last_user_index]
            .iter()
            .position(|message| message.role == crate::models::shared::chat::ChatRole::Assistant)
        else {
            cursor = user_index + 1;
            continue;
        };
        let assistant_index = user_index + 1 + assistant_offset;
        ends.push(assistant_index);
        cursor = assistant_index + 1;
    }
    ends
}

fn chat_messages_after_prefix(messages: &[ChatMessage], prefix_end: usize) -> Vec<ChatMessage> {
    let protected_prefix = usize::from(
        messages
            .first()
            .is_some_and(|message| message.role == crate::models::shared::chat::ChatRole::System),
    );
    let mut compacted = Vec::with_capacity(
        protected_prefix.saturating_add(messages.len().saturating_sub(prefix_end + 1)),
    );
    compacted.extend_from_slice(&messages[..protected_prefix]);
    compacted.extend_from_slice(&messages[(prefix_end + 1)..]);
    compacted
}

/// The answer of a server-side generation with any reasoning split off:
/// `None` when nothing is left (the budget ran out inside `<think>`).
fn auxiliary_answer(text: &str, opened_in_prompt: bool) -> Option<String> {
    let (_, answer) = izwi_model_catalog::chat::reasoning::split_reasoning(text, opened_in_prompt);
    let answer = answer.trim();
    (!answer.is_empty()).then(|| answer.to_string())
}

fn prompt_too_long(prompt_tokens: usize, context_limit: usize) -> Error {
    Error::ContextLengthExceeded(format!(
        "This model's maximum context length is {context_limit} tokens. However, your messages resulted in {prompt_tokens} tokens. Please reduce the length of the messages."
    ))
}

/// The output budget a chat request runs with (D8). The prompt must leave at
/// least one token. An unset budget takes the rest of the context, as in
/// vLLM. An explicit budget that does not fit is an error unless the request
/// opted into truncation, which shortens it to the remaining context.
pub(crate) fn chat_completion_budget(
    prompt_tokens: usize,
    requested_max_tokens: usize,
    context_limit: usize,
    truncation: ChatTruncation,
) -> Result<usize> {
    if prompt_tokens >= context_limit {
        return Err(prompt_too_long(prompt_tokens, context_limit));
    }
    let requested_max_tokens = requested_max_tokens.max(1);
    let remaining = context_limit - prompt_tokens;
    if requested_max_tokens <= remaining
        || requested_max_tokens == CHAT_MAX_TOKENS_UNSET
        || truncation == ChatTruncation::Auto
    {
        return Ok(requested_max_tokens.min(remaining));
    }
    Err(Error::ContextLengthExceeded(format!(
        "This model's maximum context length is {context_limit} tokens. However, you requested {} tokens ({prompt_tokens} in the messages, {requested_max_tokens} in the completion). Please reduce the length of the messages or completion.",
        prompt_tokens + requested_max_tokens
    )))
}

fn reconcile_streamed_chat_text(
    streamed_text: String,
    terminal_text: Option<String>,
) -> Result<String> {
    let Some(terminal_text) = terminal_text.filter(|text| !text.is_empty()) else {
        return Ok(streamed_text);
    };
    if streamed_text.is_empty() {
        return Ok(terminal_text);
    }
    if streamed_text == terminal_text {
        return Ok(streamed_text);
    }

    Err(Error::InferenceError(format!(
        "Streaming chat text did not match terminal output (streamed {} bytes, terminal {} bytes)",
        streamed_text.len(),
        terminal_text.len()
    )))
}

/// A streaming chat request that passed admission and has not started; see
/// [`RuntimeService::admit_streaming_chat`]. Dropping it releases the
/// admission.
pub struct AdmittedChatStream(AdmittedEngineRequest);

impl RuntimeService {
    /// Admit one chat request and return its worker-facing execution handle.
    ///
    /// Success is authoritative acceptance: coordinator capacity, a pinned
    /// model generation and an exact Engine session have all been established.
    pub async fn start_chat_invocation(
        &self,
        invocation: RuntimeChatInvocationRequest,
    ) -> Result<RuntimeChatInvocation> {
        let RuntimeChatInvocationRequest {
            variant,
            messages,
            params,
            chat_config,
            correlation_id,
            runtime_context,
            streaming,
            max_tokens_ceiling,
        } = invocation;
        let admitted = self
            .build_chat_request_with_params_and_config(
                variant,
                messages,
                params,
                chat_config,
                correlation_id.as_deref(),
                runtime_context,
                streaming,
                max_tokens_ceiling,
            )
            .await?;
        self.start_admitted_chat_invocation(admitted).await
    }

    /// The chat config of the `max_new_tokens` helpers. Their callers are
    /// product features (agent turns, voice sessions, transcript summaries)
    /// that grow histories and pick budgets without the exact prompt size, so
    /// they opt into truncation.
    fn sized_budget_chat_config() -> ChatRequestConfig {
        ChatRequestConfig {
            truncation: ChatTruncation::Auto,
            ..ChatRequestConfig::default()
        }
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
            logprobs: params.logprobs || params.top_logprobs > 0,
            top_logprobs: params.top_logprobs,
            constrain_json_object: params.constrain_json_object,
            skip_special_tokens: params.skip_special_tokens,
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
        max_tokens_ceiling: Option<usize>,
    ) -> Result<AdmittedEngineRequest> {
        if variant.family() == ModelFamily::Lfm3Audio {
            return self
                .build_lfm3_audio_chat_request(
                    variant,
                    messages,
                    params,
                    chat_config,
                    correlation_id,
                    runtime_context,
                    streaming,
                    max_tokens_ceiling,
                )
                .await;
        }
        if messages.is_empty() {
            return Err(Error::InvalidInput(
                "Chat request missing messages".to_string(),
            ));
        }
        // Qwen3.5/3.6/3.8 templates accept a system message only first.
        let messages = if matches!(
            variant.family(),
            ModelFamily::Qwen35Chat | ModelFamily::Qwen35MoeChat | ModelFamily::Qwen38Chat
        ) {
            izwi_model_catalog::chat::fold_system_messages(messages)
        } else {
            messages
        };
        if !chat_config.media_inputs.is_empty() && variant.family() != ModelFamily::Qwen35Chat {
            return Err(Error::InvalidInput(format!(
                "Chat model {variant} does not support Qwen3.5 media inputs"
            )));
        }
        let correlation_id = correlation_id.map(ToOwned::to_owned);
        let backend = self.backend_router.context().backend_kind;
        let configured_context_limit = self.config.portable_context_ceiling();
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
                let context_limit = registry.effective_context(variant).unwrap_or(
                    resolve_backend_model_context(
                        backend,
                        configured_context_limit,
                        model.max_context_tokens()?,
                    )?,
                );
                let original_messages = messages;
                let initial =
                    model.prepare_prompt_for_execution(&original_messages, &prompt_config)?;
                let (messages, prompt_tokens, prepared_chat_prompt, trimmed_messages) =
                    if initial.0.len() < context_limit {
                        (original_messages, initial.0, initial.1, 0usize)
                    } else {
                    if chat_config.truncation != ChatTruncation::Auto {
                        return Err(prompt_too_long(initial.0.len(), context_limit));
                    }
                    if !chat_config.media_inputs.is_empty() {
                        return Err(Error::ContextLengthExceeded(format!(
                            "Chat prompt has {} tokens in a {context_limit}-token context; automatic history compaction is disabled for media turns so inputs cannot become misaligned",
                            initial.0.len()
                        )));
                    }
                    let prefix_ends = complete_chat_turn_prefix_ends(&original_messages);
                    if prefix_ends.is_empty() {
                        return Err(Error::ContextLengthExceeded(format!(
                            "Chat prompt has {} tokens in a {context_limit}-token context and no older complete turn can be compacted",
                            initial.0.len()
                        )));
                    }
                    let protected_prefix = usize::from(
                        original_messages.first().is_some_and(|message| {
                            message.role == crate::models::shared::chat::ChatRole::System
                        }),
                    );
                    let mut low = 0usize;
                    let mut high = prefix_ends.len() - 1;
                    let mut selected = None;
                    while low <= high {
                        let middle = low + (high - low) / 2;
                        let prefix_end = prefix_ends[middle];
                        let candidate =
                            chat_messages_after_prefix(&original_messages, prefix_end);
                        let prepared =
                            model.prepare_prompt_for_execution(&candidate, &prompt_config)?;
                        if prepared.0.len() < context_limit {
                            selected = Some((candidate, prepared, prefix_end));
                            if middle == 0 {
                                break;
                            }
                            high = middle - 1;
                        } else {
                            low = middle + 1;
                        }
                    }
                    let Some((messages, prepared, prefix_end)) = selected else {
                        return Err(Error::ContextLengthExceeded(format!(
                            "Chat prompt has {} tokens in a {context_limit}-token context and remains too long after compacting all older complete turns",
                            initial.0.len()
                        )));
                    };
                    (
                        messages,
                        prepared.0,
                        prepared.1,
                        prefix_end + 1 - protected_prefix,
                    )
                };
                if trimmed_messages > 0 {
                    warn!(
                        model = %variant,
                        trimmed_messages,
                        remaining_messages = messages.len(),
                        prompt_tokens = prompt_tokens.len(),
                        context_limit,
                        "compacted oldest complete chat turns to fit the configured context"
                    );
                }

                let requested_max_tokens = params.max_tokens.max(1);
                let budget = chat_completion_budget(
                    prompt_tokens.len(),
                    requested_max_tokens,
                    context_limit,
                    chat_config.truncation,
                )?;
                if budget < requested_max_tokens && requested_max_tokens != CHAT_MAX_TOKENS_UNSET {
                    warn!(
                        model = %variant,
                        requested_max_tokens,
                        max_tokens = budget,
                        prompt_tokens = prompt_tokens.len(),
                        context_limit,
                        "clamped chat completion budget to the remaining context"
                    );
                }
                params.max_tokens = budget.min(max_tokens_ceiling.unwrap_or(usize::MAX));
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
                    prepared_chat_prompt,
                    model,
                    context_limit,
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
        let params = GenerationParams {
            max_tokens: max_new_tokens.max(1),
            ..Default::default()
        };
        let admitted = self
            .build_chat_request_with_params_and_config(
                variant,
                messages,
                params,
                Self::sized_budget_chat_config(),
                correlation_id,
                runtime_context,
                false,
                None,
            )
            .await?;
        let output = self.run_admitted_request(admitted).await?;
        Ok(ChatGeneration {
            latency_breakdown: output.latency_breakdown,
            finish_reason: output.finish_reason,
            text: output.text.unwrap_or_default(),
            prompt_tokens: output.token_stats.prompt_tokens,
            tokens_generated: output.num_tokens,
            generation_time_ms: output.generation_time.as_secs_f64() * 1000.0,
            cached_prompt_tokens: output.token_stats.cached_prefix_tokens.map(u64::from),
            logprobs: output.logprobs,
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
                None,
            )
            .await?;
        let output = self.run_admitted_request(admitted).await?;
        Ok(ChatGeneration {
            latency_breakdown: output.latency_breakdown,
            finish_reason: output.finish_reason,
            text: output.text.unwrap_or_default(),
            prompt_tokens: output.token_stats.prompt_tokens,
            tokens_generated: output.num_tokens,
            generation_time_ms: output.generation_time.as_secs_f64() * 1000.0,
            cached_prompt_tokens: output.token_stats.cached_prefix_tokens.map(u64::from),
            logprobs: output.logprobs,
        })
    }

    /// The chat config of the server's own generations (summaries, titles,
    /// memory): thinking off and truncation on, whatever the model defaults to.
    pub fn auxiliary_chat_config() -> ChatRequestConfig {
        ChatRequestConfig {
            enable_thinking: Some(false),
            truncation: ChatTruncation::Auto,
            ..ChatRequestConfig::default()
        }
    }

    /// A generation the server makes for its own features. Thinking is off,
    /// and only the answer comes back: a model that reasons anyway (LFM2.5
    /// Thinking ignores the switch) has its `<think>` block split off, and an
    /// answer left empty, because the budget ran out while reasoning, is an
    /// error instead of reasoning stored as the result.
    pub async fn auxiliary_generate(
        &self,
        variant: ModelVariant,
        messages: Vec<ChatMessage>,
        params: GenerationParams,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
    ) -> Result<ChatGeneration> {
        let chat_config = Self::auxiliary_chat_config();
        let opened_in_prompt = izwi_model_catalog::chat::reasoning::reasoning_opened_in_prompt(
            variant,
            chat_config.enable_thinking,
        );
        let mut generation = self
            .chat_generate_with_runtime_context(
                variant,
                messages,
                params,
                chat_config,
                correlation_id,
                runtime_context,
            )
            .await?;
        generation.text =
            auxiliary_answer(&generation.text, opened_in_prompt).ok_or_else(|| {
                Error::InferenceError(format!(
                    "{variant} returned no answer (its output budget ran out while reasoning)"
                ))
            })?;
        Ok(generation)
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
        let params = GenerationParams {
            max_tokens: max_new_tokens.max(1),
            ..Default::default()
        };
        self.chat_generate_streaming_tokens_with_runtime_context(
            variant,
            messages,
            params,
            Self::sized_budget_chat_config(),
            correlation_id,
            runtime_context,
            move |delta, _logprobs| on_delta(delta),
        )
        .await
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
        self.chat_generate_streaming_tokens_with_runtime_context(
            variant,
            messages,
            params,
            chat_config,
            correlation_id,
            runtime_context,
            move |delta, _logprobs| on_delta(delta),
        )
        .await
    }

    /// DS9.3: streaming chat whose delta callback also carries the per-token
    /// logprob entries observed for that chunk (empty unless the request
    /// asked for logprobs).
    #[allow(clippy::too_many_arguments)]
    pub async fn chat_generate_streaming_tokens_with_runtime_context<F>(
        &self,
        variant: ModelVariant,
        messages: Vec<ChatMessage>,
        params: GenerationParams,
        chat_config: ChatRequestConfig,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
        on_delta: F,
    ) -> Result<ChatGeneration>
    where
        F: FnMut(String, Vec<crate::engine::TokenLogprob>) + Send + 'static,
    {
        let admitted = self
            .admit_streaming_chat(
                variant,
                messages,
                params,
                chat_config,
                correlation_id,
                runtime_context,
            )
            .await?;
        self.run_admitted_streaming_chat(admitted, on_delta).await
    }

    /// Admit a streaming chat request without starting it: the prompt fits
    /// the context (or the request fails with `ContextLengthExceeded`), and
    /// the request holds its coordinator capacity and model residency until
    /// [`Self::run_admitted_streaming_chat`] runs it or it is dropped. A
    /// server admits before it commits to a streamed response, so an
    /// admission failure is still a plain HTTP error.
    pub async fn admit_streaming_chat(
        &self,
        variant: ModelVariant,
        messages: Vec<ChatMessage>,
        params: GenerationParams,
        chat_config: ChatRequestConfig,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
    ) -> Result<AdmittedChatStream> {
        self.build_chat_request_with_params_and_config(
            variant,
            messages,
            params,
            chat_config,
            correlation_id,
            runtime_context,
            true,
            None,
        )
        .await
        .map(AdmittedChatStream)
    }

    /// Run an admitted streaming chat request; `on_delta` receives each
    /// text delta with its logprob entries.
    pub async fn run_admitted_streaming_chat<F>(
        &self,
        admitted: AdmittedChatStream,
        mut on_delta: F,
    ) -> Result<ChatGeneration>
    where
        F: FnMut(String, Vec<crate::engine::TokenLogprob>) + Send + 'static,
    {
        let AdmittedChatStream(admitted) = admitted;
        let mut streamed_text = String::new();
        let output = self
            .run_admitted_streaming_request(admitted, |chunk| {
                let crate::engine::StreamingOutput { text, logprobs, .. } = chunk;
                if let Some(delta) = text {
                    if !delta.is_empty() || !logprobs.is_empty() {
                        streamed_text.push_str(&delta);
                        on_delta(delta, logprobs);
                    }
                }
                std::future::ready(Ok(()))
            })
            .await?;

        let text = reconcile_streamed_chat_text(streamed_text, output.text)?;
        Ok(ChatGeneration {
            latency_breakdown: output.latency_breakdown,
            finish_reason: output.finish_reason,
            text,
            prompt_tokens: output.token_stats.prompt_tokens,
            tokens_generated: output.num_tokens,
            generation_time_ms: output.generation_time.as_secs_f64() * 1000.0,
            cached_prompt_tokens: output.token_stats.cached_prefix_tokens.map(u64::from),
            logprobs: output.logprobs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        auxiliary_answer, chat_completion_budget, chat_messages_after_prefix,
        complete_chat_turn_prefix_ends, reconcile_streamed_chat_text, CHAT_MAX_TOKENS_UNSET,
    };
    use crate::error::Error;
    use crate::models::shared::chat::{ChatMessage, ChatRole, ChatTruncation};

    fn message(role: ChatRole, content: &str) -> ChatMessage {
        ChatMessage {
            role,
            content: content.to_string(),
        }
    }

    #[test]
    fn auxiliary_answers_never_carry_reasoning() {
        // Thinking off: the answer passes through.
        assert_eq!(
            auxiliary_answer("  Tips for winter houseplants \n", false).as_deref(),
            Some("Tips for winter houseplants")
        );
        // A model that reasons anyway writes its own opener (LFM2.5 Thinking).
        assert_eq!(
            auxiliary_answer(
                "<think>Okay, the user wants</think>\n\nWinter plant care",
                false
            )
            .as_deref(),
            Some("Winter plant care")
        );
        // A block the prompt opened (Qwen3.5 with thinking on).
        assert_eq!(
            auxiliary_answer(
                "Thinking Process:\n1. **Analyze**</think>\n\nA summary.",
                true
            )
            .as_deref(),
            Some("A summary.")
        );
        // The budget ran out while reasoning: nothing is returned, so the
        // reasoning is never stored as the result.
        assert_eq!(
            auxiliary_answer("Thinking Process:\n\n1.  **Analyze the Request:**", true),
            None
        );
        assert_eq!(auxiliary_answer("<think> Okay, let's see", false), None);
        assert_eq!(auxiliary_answer("   ", false), None);
    }

    #[test]
    fn compaction_preserves_system_and_latest_query() {
        let messages = vec![
            message(ChatRole::System, "system"),
            message(ChatRole::User, "u1"),
            message(ChatRole::Assistant, "a1"),
            message(ChatRole::User, "u2"),
            message(ChatRole::Assistant, "a2"),
            message(ChatRole::User, "u3"),
        ];

        let ends = complete_chat_turn_prefix_ends(&messages);
        assert_eq!(ends, vec![2, 4]);
        let compacted = chat_messages_after_prefix(&messages, ends[0]);
        assert_eq!(compacted.len(), 4);
        assert_eq!(compacted[0].role, ChatRole::System);
        assert_eq!(compacted[1].content, "u2");
        assert_eq!(compacted[3].content, "u3");
    }

    #[test]
    fn compaction_never_drops_the_only_user_query() {
        let messages = vec![
            message(ChatRole::System, "system"),
            message(ChatRole::User, "latest"),
        ];
        assert!(complete_chat_turn_prefix_ends(&messages).is_empty());
        assert_eq!(messages.len(), 2);
    }

    /// D8: like OpenAI and vLLM, a request that does not fit is a 400 with
    /// the token counts; only an unset budget or an explicit truncation
    /// opt-in is shortened to the remaining context.
    #[test]
    fn completion_budget_rejects_what_does_not_fit_unless_truncation_is_on() {
        let disabled = ChatTruncation::Disabled;
        let budget = |prompt, requested, truncation| {
            chat_completion_budget(prompt, requested, 100, truncation)
        };
        assert_eq!(budget(90, 10, disabled).unwrap(), 10);
        assert_eq!(budget(90, 0, disabled).unwrap(), 1);
        assert_eq!(budget(90, CHAT_MAX_TOKENS_UNSET, disabled).unwrap(), 10);
        assert_eq!(budget(90, 32, ChatTruncation::Auto).unwrap(), 10);

        match budget(90, 32, disabled) {
            Err(Error::ContextLengthExceeded(message)) => assert!(
                message.contains("maximum context length is 100 tokens")
                    && message.contains(
                        "you requested 122 tokens (90 in the messages, 32 in the completion)"
                    ),
                "{message}"
            ),
            other => panic!("expected a context-length error, got {other:?}"),
        }
        for truncation in [disabled, ChatTruncation::Auto] {
            match budget(100, CHAT_MAX_TOKENS_UNSET, truncation) {
                Err(Error::ContextLengthExceeded(message)) => {
                    assert!(message.contains("resulted in 100 tokens"), "{message}")
                }
                other => panic!("expected a context-length error, got {other:?}"),
            }
        }
    }

    #[test]
    fn streamed_chat_text_accepts_matching_terminal_text() {
        let text = reconcile_streamed_chat_text(
            "complete response".to_string(),
            Some("complete response".to_string()),
        )
        .expect("matching text should reconcile");

        assert_eq!(text, "complete response");
    }

    #[test]
    fn streamed_chat_text_rejects_terminal_mismatch() {
        let err = reconcile_streamed_chat_text(
            "streamed response".to_string(),
            Some("different response".to_string()),
        )
        .expect_err("conflicting public and terminal text must fail");

        assert!(err.to_string().contains("did not match terminal output"));
    }

    #[test]
    fn streamed_chat_text_accepts_terminal_only_adapters() {
        let text =
            reconcile_streamed_chat_text(String::new(), Some("terminal response".to_string()))
                .expect("terminal-only text should reconcile");

        assert_eq!(text, "terminal response");
    }

    #[test]
    fn streamed_chat_text_accepts_delta_only_adapters() {
        let text = reconcile_streamed_chat_text("delta response".to_string(), None)
            .expect("delta-only text should reconcile");
        let empty_terminal =
            reconcile_streamed_chat_text("delta response".to_string(), Some(String::new()))
                .expect("an empty terminal payload should not discard deltas");

        assert_eq!(text, "delta response");
        assert_eq!(empty_terminal, "delta response");
    }
}
