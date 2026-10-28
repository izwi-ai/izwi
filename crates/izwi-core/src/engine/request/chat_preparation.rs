//! Chat execution preparation: the context window, prepared prompts and the
//! chat generation config.

use super::*;

impl EngineCoreRequest {
    pub(crate) fn has_chat_execution_preparation(&self) -> bool {
        self.chat_execution_ready.is_some()
    }

    /// Bound an unprepared direct chat request before it is cloned or passed
    /// to a tokenizer. Runtime-prepared requests are resource-admitted before
    /// this boundary; public `Engine` callers otherwise have no admission
    /// lease protecting the blocking tokenizer pool from oversized payloads.
    pub(crate) fn validate_direct_chat_preparation_input(&self, max_seq_len: usize) -> Result<()> {
        if self.task_type != TaskType::Chat || self.has_chat_execution_preparation() {
            return Ok(());
        }
        if !self.chat_config.media_inputs.is_empty() {
            return Err(Error::InvalidInput(
                "Direct Engine multimodal chat is not supported; use RuntimeService so media preparation is resource-admitted"
                    .to_string(),
            ));
        }

        let context_limit = max_seq_len.max(1);
        let input_limit = context_limit
            .saturating_mul(DIRECT_CHAT_INPUT_BYTES_PER_CONTEXT_TOKEN)
            .clamp(DIRECT_CHAT_MIN_INPUT_BYTES, DIRECT_CHAT_MAX_INPUT_BYTES);
        let messages = self.chat_messages.as_deref().ok_or_else(|| {
            Error::InvalidInput(format!("Chat request {} is missing messages", self.id))
        })?;
        if messages.len() > context_limit {
            return Err(Error::InvalidInput(format!(
                "Direct chat request contains {} messages, exceeding the {context_limit}-token context guard",
                messages.len()
            )));
        }

        let mut total = 0usize;
        for message in messages {
            add_bounded_chat_input_bytes(
                &mut total,
                message.content.len(),
                input_limit,
                "message",
            )?;
        }

        if self.chat_config.tools.len() > context_limit {
            return Err(Error::InvalidInput(format!(
                "Direct chat request contains {} tools, exceeding the {context_limit}-token context guard",
                self.chat_config.tools.len()
            )));
        }
        add_bounded_chat_input_bytes(
            &mut total,
            self.chat_config
                .tools
                .len()
                .checked_mul(DIRECT_CHAT_JSON_NODE_BYTES)
                .ok_or_else(|| {
                    Error::InvalidInput("Direct chat tool count overflow".to_string())
                })?,
            input_limit,
            "tool",
        )?;

        // Walk JSON iteratively so an adversarially nested tool schema cannot
        // overflow this thread's stack before serde reaches its own guard.
        let mut pending = self
            .chat_config
            .tools
            .iter()
            .map(|value| (value, 1usize))
            .collect::<Vec<_>>();
        while let Some((value, depth)) = pending.pop() {
            if depth > DIRECT_CHAT_MAX_TOOL_JSON_DEPTH {
                return Err(Error::InvalidInput(format!(
                    "Direct chat tool JSON exceeds the {DIRECT_CHAT_MAX_TOOL_JSON_DEPTH}-level nesting limit"
                )));
            }
            match value {
                serde_json::Value::Null | serde_json::Value::Bool(_) => {}
                serde_json::Value::Number(_) => add_bounded_chat_input_bytes(
                    &mut total,
                    DIRECT_CHAT_JSON_NODE_BYTES,
                    input_limit,
                    "tool number",
                )?,
                serde_json::Value::String(value) => add_bounded_chat_input_bytes(
                    &mut total,
                    escaped_json_string_upper_bound(value)?,
                    input_limit,
                    "tool string",
                )?,
                serde_json::Value::Array(values) => {
                    add_bounded_chat_input_bytes(
                        &mut total,
                        values
                            .len()
                            .checked_mul(DIRECT_CHAT_JSON_NODE_BYTES)
                            .ok_or_else(|| {
                                Error::InvalidInput(
                                    "Direct chat tool array size overflow".to_string(),
                                )
                            })?,
                        input_limit,
                        "tool array",
                    )?;
                    pending.extend(values.iter().map(|value| (value, depth + 1)));
                }
                serde_json::Value::Object(values) => {
                    add_bounded_chat_input_bytes(
                        &mut total,
                        values
                            .len()
                            .checked_mul(DIRECT_CHAT_JSON_NODE_BYTES)
                            .ok_or_else(|| {
                                Error::InvalidInput(
                                    "Direct chat tool object size overflow".to_string(),
                                )
                            })?,
                        input_limit,
                        "tool object",
                    )?;
                    for (key, value) in values {
                        add_bounded_chat_input_bytes(
                            &mut total,
                            escaped_json_string_upper_bound(key)?,
                            input_limit,
                            "tool key",
                        )?;
                        pending.push((value, depth + 1));
                    }
                }
            }
        }
        Ok(())
    }

    /// Enforce the configured context contract after exact tokenization. The
    /// prompt must leave room for at least one output token, and the requested
    /// output budget is reduced to the remaining context instead of allowing
    /// the scheduler/model cache to exceed the configured sequence length.
    pub(crate) fn enforce_chat_context_window(&mut self, max_seq_len: usize) -> Result<()> {
        if self.task_type != TaskType::Chat {
            return Ok(());
        }
        let context_limit = self
            .chat_execution_ready
            .as_ref()
            .map(|ready| ready.context_limit)
            .unwrap_or(max_seq_len);
        let prompt_tokens = self.prompt_tokens.len();
        if context_limit == 0 || prompt_tokens >= context_limit {
            return Err(Error::InvalidInput(format!(
                "Chat request {} exact prompt has {prompt_tokens} tokens and leaves no output capacity in the resolved {context_limit}-token context",
                self.id
            )));
        }
        self.params.max_tokens = self
            .params
            .max_tokens
            .max(1)
            .min(context_limit - prompt_tokens);
        Ok(())
    }

    /// Install the private scheduler/executor authorization for an exact chat
    /// prompt. Callers must supply tokens produced from this request's current
    /// messages/configuration by the loaded model adapter.
    pub(crate) fn install_chat_execution_preparation_with_model(
        &mut self,
        model_variant: ModelVariant,
        prompt_tokens: Vec<TokenId>,
        prepared_chat_prompt: Option<NativeChatPreparedPrompt>,
        model: ChatModelLease,
        context_limit: usize,
    ) -> Result<()> {
        self.install_chat_execution_preparation_inner(
            model_variant,
            prompt_tokens,
            prepared_chat_prompt,
            PreparedChatModel::Exact(model),
            context_limit,
        )
    }

    #[cfg(test)]
    pub(crate) fn install_chat_execution_preparation(
        &mut self,
        model_variant: ModelVariant,
        prompt_tokens: Vec<TokenId>,
        prepared_chat_prompt: Option<NativeChatPreparedPrompt>,
        context_limit: usize,
    ) -> Result<()> {
        self.install_chat_execution_preparation_inner(
            model_variant,
            prompt_tokens,
            prepared_chat_prompt,
            PreparedChatModel::ValidationOnly,
            context_limit,
        )
    }

    pub(super) fn install_chat_execution_preparation_inner(
        &mut self,
        model_variant: ModelVariant,
        prompt_tokens: Vec<TokenId>,
        prepared_chat_prompt: Option<NativeChatPreparedPrompt>,
        model: PreparedChatModel,
        context_limit: usize,
    ) -> Result<()> {
        self.chat_execution_ready = None;
        if self.task_type != TaskType::Chat {
            return Err(Error::InvalidInput(format!(
                "Request {} is not a chat request",
                self.id
            )));
        }
        if self.model_variant != Some(model_variant) {
            return Err(Error::InvalidInput(format!(
                "Chat request {} preparation model does not match its routed model",
                self.id
            )));
        }
        let messages = self.chat_messages.as_deref().ok_or_else(|| {
            Error::InvalidInput(format!("Chat request {} is missing messages", self.id))
        })?;
        if messages.is_empty() {
            return Err(Error::InvalidInput(format!(
                "Chat request {} has no messages to prepare",
                self.id
            )));
        }
        if prompt_tokens.is_empty() {
            return Err(Error::InvalidInput(format!(
                "Chat request {} preparation produced no prompt tokens",
                self.id
            )));
        }
        if context_limit == 0 {
            return Err(Error::InvalidInput(format!(
                "Chat request {} preparation has a zero context limit",
                self.id
            )));
        }

        let family = model_variant.family();
        let needs_prepared_prompt =
            matches!(family, ModelFamily::Qwen35Chat | ModelFamily::Qwen38Chat);
        match prepared_chat_prompt.as_ref() {
            Some(prepared) if prepared.family() != family => {
                return Err(Error::InvalidInput(format!(
                    "Chat request {} routed a mismatched prepared prompt artifact to {model_variant}",
                    self.id
                )));
            }
            Some(prepared) if prepared.prompt_ids() != prompt_tokens => {
                return Err(Error::InvalidInput(format!(
                    "Chat request {} prepared prompt artifact does not match its exact prompt tokens",
                    self.id
                )));
            }
            None if needs_prepared_prompt => {
                return Err(Error::InvalidInput(format!(
                    "Chat request {} is missing its prepared hybrid-Qwen prompt artifact",
                    self.id
                )));
            }
            _ => {}
        }

        if prepared_chat_prompt.is_some() {
            // The opaque artifact owns all decoded/encoded vision state needed
            // by execution. Do not retain or repeatedly fingerprint multi-MB
            // data URLs (or signed remote URLs) on the scheduler hot path.
            for media in &mut self.chat_config.media_inputs {
                media.source.clear();
                media.source.shrink_to_fit();
            }
        }

        let prepared_continuous_cost =
            Self::continuous_chat_stage_cost(self.execution_adapter_binding.as_ref(), &model)?;

        let fingerprint =
            chat_execution_fingerprint(model_variant, messages, &self.chat_config, &prompt_tokens)?;
        self.prepared_stage_costs.clear();
        if let Some((stage_id, cost)) = prepared_continuous_cost {
            self.install_prepared_stage_cost(stage_id, cost)?;
        }
        // A prepared prompt declares its own boundaries; a model that shares
        // prefixes without one (LFM chat) finds them in the prompt tokens.
        let history_boundary = prepared_chat_prompt
            .as_ref()
            .and_then(NativeChatPreparedPrompt::managed_snapshot_boundary)
            .or_else(|| match &model {
                PreparedChatModel::Exact(model) => model.managed_history_boundary(&prompt_tokens),
                #[cfg(test)]
                PreparedChatModel::ValidationOnly => None,
            });
        let preamble_boundary = history_boundary.and_then(|history| {
            crate::models::registry::managed_preamble_boundary(&prompt_tokens, history)
        });
        self.managed_snapshot_boundary = history_boundary
            .filter(|boundary| *boundary <= prompt_tokens.len())
            .and_then(|boundary| u32::try_from(boundary).ok());
        self.managed_preamble_boundary = preamble_boundary
            .filter(|boundary| *boundary <= prompt_tokens.len())
            .and_then(|boundary| u32::try_from(boundary).ok());
        self.prompt_tokens = prompt_tokens;
        self.sync_task_from_fields();
        self.chat_execution_ready = Some(ChatExecutionReady {
            model_variant,
            model,
            fingerprint,
            prepared_chat_prompt,
            context_limit,
            core_validated: false,
        });
        Ok(())
    }

    pub(super) fn continuous_chat_stage_cost(
        binding: Option<&super::super::ExecutionAdapterBinding>,
        model: &PreparedChatModel,
    ) -> Result<Option<(StageId, WorkCost)>> {
        let Some(stage) = binding.and_then(|binding| {
            binding
                .stages
                .iter()
                .find(|stage| stage.batch_mode == super::super::NativeBatchMode::Continuous)
        }) else {
            return Ok(None);
        };
        #[allow(clippy::infallible_destructuring_match)]
        let model = match model {
            PreparedChatModel::Exact(model) => model,
            #[cfg(test)]
            PreparedChatModel::ValidationOnly => return Ok(None),
        };
        if !model.supports_continuous_decode_batch() {
            return Err(Error::InvalidInput(
                "loaded adapter selected continuous decode for an incompatible chat model"
                    .to_string(),
            ));
        }
        let accelerator_bytes = model.continuous_decode_batch_workspace_per_row_bytes()?;
        let cost = WorkCost::with_workspace(
            1,
            1,
            super::super::continuous_chat_workspace_per_row(accelerator_bytes)?,
        );
        let required_bytes = cost.workspace.workspace_bytes()?;
        if required_bytes > stage.max_workspace_bytes {
            return Err(Error::Overloaded(format!(
                "continuous decode workspace exceeds its loaded adapter budget (required {} bytes, available {} bytes)",
                required_bytes, stage.max_workspace_bytes,
            )));
        }
        Ok(Some((stage.id, cost)))
    }

    /// Validate the private preparation marker and both legacy/typed payload
    /// views before any scheduler or executor trusts prompt-token accounting.
    pub(crate) fn validate_chat_execution_preparation(&self) -> Result<()> {
        if self.task_type != TaskType::Chat {
            return Err(Error::InvalidInput(format!(
                "Request {} carries chat preparation for a non-chat task",
                self.id
            )));
        }
        let ready = self.chat_execution_ready.as_ref().ok_or_else(|| {
            Error::InvalidInput(format!(
                "Chat request {} is missing exact model prompt preparation",
                self.id
            ))
        })?;
        let model_variant = self.model_variant.ok_or_else(|| {
            Error::InvalidInput(format!(
                "Chat request {} is missing its routed model",
                self.id
            ))
        })?;
        if ready.model_variant != model_variant {
            return Err(Error::InvalidInput(format!(
                "Chat request {} changed model after prompt preparation",
                self.id
            )));
        }
        let messages = self.chat_messages.as_deref().ok_or_else(|| {
            Error::InvalidInput(format!("Chat request {} is missing messages", self.id))
        })?;
        let field_fingerprint = chat_execution_fingerprint(
            model_variant,
            messages,
            &self.chat_config,
            &self.prompt_tokens,
        )?;
        if field_fingerprint != ready.fingerprint {
            return Err(Error::InvalidInput(format!(
                "Chat request {} changed after exact prompt preparation",
                self.id
            )));
        }

        let EngineTask::Chat(input) = &self.task else {
            return Err(Error::InvalidInput(format!(
                "Chat request {} has a mismatched typed task payload",
                self.id
            )));
        };
        let task_fingerprint = chat_execution_fingerprint(
            model_variant,
            &input.messages,
            &input.chat_config,
            &input.prompt_tokens,
        )?;
        if task_fingerprint != ready.fingerprint {
            return Err(Error::InvalidInput(format!(
                "Chat request {} typed payload does not match its exact prompt preparation",
                self.id
            )));
        }

        match ready.prepared_chat_prompt.as_ref() {
            Some(prepared)
                if prepared.family() != model_variant.family()
                    || prepared.prompt_ids() != self.prompt_tokens =>
            {
                Err(Error::InvalidInput(format!(
                    "Chat request {} has a mismatched prepared prompt artifact",
                    self.id
                )))
            }
            None if matches!(
                model_variant.family(),
                ModelFamily::Qwen35Chat | ModelFamily::Qwen38Chat
            ) =>
            {
                Err(Error::InvalidInput(format!(
                    "Chat request {} is missing its prepared hybrid-Qwen prompt artifact",
                    self.id
                )))
            }
            _ => Ok(()),
        }
    }

    pub(crate) fn validate_chat_execution_for_executor(&self) -> Result<()> {
        if self
            .chat_execution_ready
            .as_ref()
            .is_some_and(|ready| ready.core_validated)
        {
            if self.task_type != TaskType::Chat {
                return Err(Error::InvalidInput(format!(
                    "Request {} carries sealed chat preparation for a non-chat task",
                    self.id
                )));
            }
            return Ok(());
        }
        self.validate_chat_execution_preparation()
    }

    pub(crate) fn prepared_chat_prompt_for_executor(
        &self,
    ) -> Result<Option<&NativeChatPreparedPrompt>> {
        self.validate_chat_execution_for_executor()?;
        Ok(self
            .chat_execution_ready
            .as_ref()
            .and_then(|ready| ready.prepared_chat_prompt.as_ref()))
    }

    pub(crate) fn prepared_chat_model_for_executor(&self) -> Result<Arc<NativeChatModel>> {
        self.validate_chat_execution_for_executor()?;
        match &self
            .chat_execution_ready
            .as_ref()
            .expect("validated chat preparation must exist")
            .model
        {
            PreparedChatModel::Exact(model) => Ok(model.model_arc()),
            #[cfg(test)]
            PreparedChatModel::ValidationOnly => Err(Error::InferenceError(
                "Test-only chat preparation does not carry a model instance".to_string(),
            )),
        }
    }

    pub(crate) fn chat_generation_config(&self) -> ChatGenerationConfig {
        ChatGenerationConfig {
            temperature: self.params.temperature.max(0.0),
            top_p: self.params.top_p.clamp(0.0, 1.0),
            top_k: self.params.top_k,
            repetition_penalty: self.params.repetition_penalty.max(1.0),
            presence_penalty: self.params.presence_penalty.clamp(-2.0, 2.0),
            stop_token_ids: self.params.stop_token_ids.clone(),
            seed: self
                .params
                .seed
                .map_or_else(|| Self::chat_request_seed(&self.id), Self::chat_user_seed),
            request: self.chat_config.clone(),
            logprobs: self.params.logprobs || self.params.top_logprobs > 0,
            top_logprobs: self.params.top_logprobs,
            constrain_json_object: self.params.constrain_json_object,
            skip_special_tokens: self.params.skip_special_tokens,
        }
    }

    pub(crate) fn chat_request_seed(request_id: &str) -> u64 {
        const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;

        let mut hash = FNV_OFFSET_BASIS;
        for byte in request_id.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    /// The sampler seed for a caller's `seed`. The chat samplers seed from
    /// the clock on 0, so a caller's seed is mixed (SplitMix64's finalizer,
    /// a bijection) rather than passed through, and a 0 result becomes 1.
    pub(crate) fn chat_user_seed(seed: u64) -> u64 {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)).max(1)
    }
}
