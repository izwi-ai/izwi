//! Offline and streaming transcription entry points and the ASR request they
//! build.

use super::*;

impl RuntimeService {
    pub(super) async fn build_asr_request(
        &self,
        variant: ModelVariant,
        audio_input: AsrAudioInput<'_>,
        language: Option<&str>,
        prompt: Option<&str>,
        task: AsrTaskOptions,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
        streaming: bool,
        native_word_timestamps: bool,
    ) -> Result<AdmittedEngineRequest> {
        validate_asr_language(variant, language)?;
        validate_asr_task(variant, language, prompt, &task)?;
        let audio_bytes = match audio_input {
            AsrAudioInput::Base64("") => {
                return Err(Error::InvalidInput(
                    "ASR request missing audio input".to_string(),
                ));
            }
            AsrAudioInput::Bytes([]) => {
                return Err(Error::InvalidInput(
                    "ASR request missing audio bytes".to_string(),
                ));
            }
            AsrAudioInput::Base64(audio) => audio.len(),
            AsrAudioInput::Bytes(audio) => audio.len(),
        };
        audio_input.validate_retained_size()?;
        let input_bytes = audio_bytes
            .checked_add(language.map(str::len).unwrap_or_default())
            .and_then(|bytes| bytes.checked_add(prompt.map(str::len).unwrap_or_default()))
            .and_then(|bytes| bytes.checked_add(correlation_id.map(str::len).unwrap_or_default()))
            .and_then(|bytes| bytes.checked_add(task.retained_bytes()))
            .ok_or_else(|| Error::Overloaded("ASR preparation input overflowed".to_string()))?;
        self.prepare_engine_request_blocking_with_input(
            variant,
            TaskType::ASR,
            streaming,
            runtime_context,
            input_bytes,
            ResourceVector::zero(),
            move |job| async move {
                let audio_input = audio_input.into_owned_for_job(&job).await?;
                let language =
                    copy_optional_preparation_string(&job, language, "ASR language metadata")
                        .await?;
                let prompt =
                    copy_optional_preparation_string(&job, prompt, "ASR prompt metadata").await?;
                let correlation_id = copy_optional_preparation_string(
                    &job,
                    correlation_id,
                    "ASR correlation metadata",
                )
                .await?;
                let retained_metadata_bytes = language
                    .as_ref()
                    .map_or(0, String::capacity)
                    .checked_add(prompt.as_ref().map_or(0, String::capacity))
                    .and_then(|bytes| {
                        bytes.checked_add(correlation_id.as_ref().map_or(0, String::capacity))
                    })
                    .ok_or_else(|| {
                        Error::Overloaded("ASR retained metadata overflowed".to_string())
                    })?;
                job.record_materialized_usage(retained_host_observation(&[
                    input_bytes,
                    audio_input.retained_bytes(),
                    retained_metadata_bytes,
                ])?)?;
                Ok((audio_input, language, prompt, correlation_id))
            },
            move |_registry, (audio_input, language, prompt, correlation_id)| {
                audio_input.validate_source_size()?;
                let runtime_request = match audio_input {
                    OwnedAsrAudioInput::Base64(audio_base64) => AsrRuntimeRequest::from_base64(
                        audio_base64,
                        variant,
                        language,
                        correlation_id,
                        runtime_context,
                    )?,
                    OwnedAsrAudioInput::Bytes(audio_bytes) => AsrRuntimeRequest::from_bytes(
                        audio_bytes,
                        variant,
                        language,
                        correlation_id,
                        runtime_context,
                    )?,
                }
                .with_prompt(prompt)
                .with_task(task)
                .with_native_word_timestamps(native_word_timestamps);
                let mut request = runtime_request.into_engine_request();
                apply_asr_max_tokens(&mut request, variant, max_tokens);
                Ok(request)
            },
        )
        .await
    }

    pub(crate) async fn asr_transcribe_with_variant(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        language: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_with_variant_and_prompt(
            variant,
            audio_base64,
            language,
            None,
            correlation_id,
        )
        .await
    }

    pub(crate) async fn asr_transcribe_with_variant_and_prompt(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        language: Option<&str>,
        prompt: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_with_variant_and_prompt_options(
            variant,
            audio_base64,
            language,
            prompt,
            None,
            correlation_id,
        )
        .await
    }

    pub(crate) async fn asr_transcribe_with_variant_and_prompt_options(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        let admitted = self
            .build_asr_request(
                variant,
                AsrAudioInput::Base64(audio_base64),
                language,
                prompt,
                AsrTaskOptions::default(),
                max_tokens,
                correlation_id,
                RuntimeRequestContext::default(),
                false,
                false,
            )
            .await?;
        let output = self.run_admitted_request(admitted).await?;
        let duration_secs = asr_transcription_duration_secs(&output);
        let text = asr_transcript_text(&output.text.unwrap_or_default());

        Ok(AsrTranscription {
            text,
            language: asr_response_language(language, None, output.asr_diagnostics.as_ref()),
            duration_secs,
            asr_diagnostics: output.asr_diagnostics,
        })
    }

    pub(crate) async fn asr_transcribe_with_variant_streaming<F>(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        language: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_with_variant_streaming_and_prompt(
            variant,
            audio_base64,
            language,
            None,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub(crate) async fn asr_transcribe_with_variant_streaming_and_prompt<F>(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        language: Option<&str>,
        prompt: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_with_variant_streaming_and_prompt_options(
            variant,
            audio_base64,
            language,
            prompt,
            None,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub(crate) async fn asr_transcribe_with_variant_streaming_and_prompt_options<F>(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        let admitted = self
            .build_asr_request(
                variant,
                AsrAudioInput::Base64(audio_base64),
                language,
                prompt,
                AsrTaskOptions::default(),
                max_tokens,
                correlation_id,
                RuntimeRequestContext::default(),
                true,
                false,
            )
            .await?;
        self.stream_admitted_asr_request(admitted, variant, true, language, None, on_delta, |_| {})
            .await
    }

    pub(crate) async fn asr_transcribe_bytes_with_variant_and_prompt(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        prompt: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_bytes_with_variant_and_prompt_options(
            variant,
            audio_bytes,
            language,
            prompt,
            None,
            correlation_id,
        )
        .await
    }

    pub(crate) async fn asr_transcribe_bytes_with_variant_and_prompt_options(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_bytes_with_variant_and_prompt_options_and_runtime_context(
            variant,
            audio_bytes,
            language,
            prompt,
            AsrTaskOptions::default(),
            max_tokens,
            correlation_id,
            RuntimeRequestContext::default(),
            false,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn asr_transcribe_bytes_with_variant_and_prompt_options_and_runtime_context(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        prompt: Option<&str>,
        task: AsrTaskOptions,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
        native_word_timestamps: bool,
    ) -> Result<AsrTranscription> {
        let translate_to = task.translate_to.clone();
        let admitted = self
            .build_asr_request(
                variant,
                AsrAudioInput::Bytes(audio_bytes),
                language,
                prompt,
                task,
                max_tokens,
                correlation_id,
                runtime_context,
                false,
                native_word_timestamps,
            )
            .await?;
        let output = self.run_admitted_request(admitted).await?;
        let duration_secs = asr_transcription_duration_secs(&output);
        let text = asr_transcript_text(&output.text.unwrap_or_default());

        Ok(AsrTranscription {
            text,
            language: asr_response_language(
                language,
                translate_to.as_deref(),
                output.asr_diagnostics.as_ref(),
            ),
            duration_secs,
            asr_diagnostics: output.asr_diagnostics,
        })
    }

    pub(crate) async fn asr_transcribe_bytes_with_variant_streaming_and_prompt<F>(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        prompt: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_bytes_with_variant_streaming_and_prompt_options(
            variant,
            audio_bytes,
            language,
            prompt,
            None,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub(crate) async fn asr_transcribe_bytes_with_variant_streaming_and_prompt_options<F>(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_bytes_with_variant_streaming_and_prompt_options_with_progress(
            variant,
            audio_bytes,
            language,
            prompt,
            max_tokens,
            correlation_id,
            on_delta,
            |_progress| {},
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn asr_transcribe_bytes_with_variant_streaming_and_prompt_options_with_progress<
        F,
        P,
    >(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        on_delta: F,
        on_progress: P,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
        P: FnMut(AsrProgress) + Send + 'static,
    {
        self.asr_transcribe_bytes_with_variant_callback_and_prompt_options_with_progress(
            variant,
            audio_bytes,
            language,
            prompt,
            AsrTaskOptions::default(),
            max_tokens,
            correlation_id,
            on_delta,
            on_progress,
            true,
            RuntimeRequestContext::default(),
            false,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn asr_transcribe_bytes_with_variant_callback_and_prompt_options_with_progress<F, P>(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        prompt: Option<&str>,
        task: AsrTaskOptions,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        on_delta: F,
        on_progress: P,
        broker_streaming_required: bool,
        runtime_context: RuntimeRequestContext,
        native_word_timestamps: bool,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
        P: FnMut(AsrProgress) + Send + 'static,
    {
        let translate_to = task.translate_to.clone();
        let admitted = self
            .build_asr_request(
                variant,
                AsrAudioInput::Bytes(audio_bytes),
                language,
                prompt,
                task,
                max_tokens,
                correlation_id,
                runtime_context,
                true,
                native_word_timestamps,
            )
            .await?;
        self.stream_admitted_asr_request(
            admitted,
            variant,
            broker_streaming_required,
            language,
            translate_to.as_deref(),
            on_delta,
            on_progress,
        )
        .await
    }

    /// Run an admitted ASR request as a stream of text deltas.
    ///
    /// `native_stream_requested` asks for the model's native streaming
    /// contract. A model whose ASR adapter declares no streaming mode is
    /// offline-only, so its stream runs on the offline route instead: the
    /// transport still carries the deltas its decode emits (retained per-step
    /// text, chunk partials). Either way the deltas end with whatever part of
    /// the final transcript they did not cover, so they concatenate to it.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn stream_admitted_asr_request<F, P>(
        &self,
        admitted: AdmittedEngineRequest,
        variant: ModelVariant,
        native_stream_requested: bool,
        language: Option<&str>,
        translate_to: Option<&str>,
        mut on_delta: F,
        mut on_progress: P,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
        P: FnMut(AsrProgress) + Send + 'static,
    {
        let mut streamed_text = String::new();
        let handle_chunk = |chunk: crate::engine::StreamingOutput| {
            if let Some(progress) = chunk.asr_progress {
                on_progress(progress);
            }
            if let Some(delta) = chunk.text {
                if !delta.is_empty() {
                    streamed_text.push_str(&delta);
                    on_delta(delta);
                }
            }
            std::future::ready(Ok(()))
        };
        let output =
            if native_stream_requested && asr_streams_natively(&self.adapter_registry, variant) {
                self.run_admitted_streaming_request(admitted, handle_chunk)
                    .await?
            } else {
                self.run_admitted_transport_streaming_request(admitted, handle_chunk)
                    .await?
            };
        let duration_secs = asr_transcription_duration_secs(&output);
        let text = asr_transcript_text(output.text.as_deref().unwrap_or(&streamed_text));
        if let Some(tail) = unstreamed_transcript_tail(&streamed_text, &text) {
            on_delta(tail.to_string());
        }

        Ok(AsrTranscription {
            text,
            language: asr_response_language(
                language,
                translate_to,
                output.asr_diagnostics.as_ref(),
            ),
            duration_secs,
            asr_diagnostics: output.asr_diagnostics,
        })
    }

    /// Transcribe audio with Voxtral through the offline transcription path.
    pub async fn voxtral_transcribe(
        &self,
        audio_base64: &str,
        language: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_with_variant(
            ModelVariant::VoxtralMini4BRealtime2602,
            audio_base64,
            language,
            None,
        )
        .await
    }

    /// Transcribe audio with Voxtral and emit incremental deltas from offline decode.
    pub async fn voxtral_transcribe_streaming<F>(
        &self,
        audio_base64: &str,
        language: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_with_variant_streaming(
            ModelVariant::VoxtralMini4BRealtime2602,
            audio_base64,
            language,
            None,
            on_delta,
        )
        .await
    }

    /// Transcribe audio with native ASR models.
    pub async fn asr_transcribe(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_with_correlation(audio_base64, model_id, language, None)
            .await
    }

    /// Transcribe audio with request correlation metadata.
    pub async fn asr_transcribe_with_correlation(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_with_prompt_and_correlation(
            audio_base64,
            model_id,
            language,
            None,
            correlation_id,
        )
        .await
    }

    /// Transcribe audio with optional ASR initial prompt/context metadata.
    pub async fn asr_transcribe_with_prompt_and_correlation(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_with_variant_and_prompt(
            variant,
            audio_base64,
            language,
            prompt,
            correlation_id,
        )
        .await
    }

    /// Transcribe audio with optional ASR prompt and max-token decode budget.
    pub async fn asr_transcribe_with_prompt_max_tokens_and_correlation(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_with_variant_and_prompt_options(
            variant,
            audio_base64,
            language,
            prompt,
            max_tokens,
            correlation_id,
        )
        .await
    }

    pub async fn asr_transcribe_bytes(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_bytes_with_prompt(audio_bytes, model_id, language, None)
            .await
    }

    pub async fn asr_transcribe_bytes_with_prompt(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
    ) -> Result<AsrTranscription> {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_and_prompt(
            variant,
            audio_bytes,
            language,
            prompt,
            None,
        )
        .await
    }

    pub async fn asr_transcribe_bytes_with_prompt_max_tokens_and_correlation(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_and_prompt_options(
            variant,
            audio_bytes,
            language,
            prompt,
            max_tokens,
            correlation_id,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn asr_transcribe_bytes_with_runtime_context(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_bytes_with_runtime_context_and_word_timestamps(
            audio_bytes,
            model_id,
            language,
            prompt,
            AsrTaskOptions::default(),
            max_tokens,
            correlation_id,
            runtime_context,
            false,
        )
        .await
    }

    /// Like [`Self::asr_transcribe_bytes_with_runtime_context`], with a
    /// translation target or keywords (`task`; a model that cannot honour
    /// them rejects the request) and optionally asking for model-native word
    /// timestamps: a model that emits them (Granite Speech Plus) returns them
    /// as `timestamp_words` in its diagnostics; any other model transcribes
    /// as usual.
    #[allow(clippy::too_many_arguments)]
    pub async fn asr_transcribe_bytes_with_runtime_context_and_word_timestamps(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        task: AsrTaskOptions,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
        native_word_timestamps: bool,
    ) -> Result<AsrTranscription> {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_and_prompt_options_and_runtime_context(
            variant,
            audio_bytes,
            language,
            prompt,
            task,
            max_tokens,
            correlation_id,
            runtime_context,
            native_word_timestamps,
        )
        .await
    }

    /// Transcribe audio and emit deltas.
    pub async fn asr_transcribe_streaming<F>(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_streaming_with_correlation(
            audio_base64,
            model_id,
            language,
            None,
            on_delta,
        )
        .await
    }

    pub async fn asr_transcribe_streaming_bytes<F>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_streaming_bytes_with_correlation(
            audio_bytes,
            model_id,
            language,
            None,
            on_delta,
        )
        .await
    }

    /// Transcribe audio and emit deltas with request correlation metadata.
    pub async fn asr_transcribe_streaming_with_correlation<F>(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_streaming_with_prompt_and_correlation(
            audio_base64,
            model_id,
            language,
            None,
            correlation_id,
            on_delta,
        )
        .await
    }

    /// Transcribe audio with optional ASR initial prompt/context metadata and deltas.
    pub async fn asr_transcribe_streaming_with_prompt_and_correlation<F>(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_with_variant_streaming_and_prompt(
            variant,
            audio_base64,
            language,
            prompt,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub async fn asr_transcribe_streaming_with_prompt_max_tokens_and_correlation<F>(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_with_variant_streaming_and_prompt_options(
            variant,
            audio_base64,
            language,
            prompt,
            max_tokens,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub async fn asr_transcribe_streaming_bytes_with_correlation<F>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_streaming_bytes_with_prompt_and_correlation(
            audio_bytes,
            model_id,
            language,
            None,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub async fn asr_transcribe_streaming_bytes_with_prompt_and_correlation<F>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_streaming_and_prompt(
            variant,
            audio_bytes,
            language,
            prompt,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub async fn asr_transcribe_streaming_bytes_with_prompt_max_tokens_and_correlation<F>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_streaming_and_prompt_options(
            variant,
            audio_bytes,
            language,
            prompt,
            max_tokens,
            correlation_id,
            on_delta,
        )
        .await
    }

    /// Streams a transcription (or, with `task.translate_to`, a translation)
    /// as text deltas.
    #[allow(clippy::too_many_arguments)]
    pub async fn asr_transcribe_streaming_bytes_with_runtime_context<F>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        task: AsrTaskOptions,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_callback_and_prompt_options_with_progress(
            variant,
            audio_bytes,
            language,
            prompt,
            task,
            max_tokens,
            correlation_id,
            on_delta,
            |_progress| {},
            true,
            runtime_context,
            false,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn asr_transcribe_streaming_bytes_with_progress_and_correlation<F, P>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
        on_progress: P,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
        P: FnMut(AsrProgress) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_streaming_and_prompt_options_with_progress(
            variant,
            audio_bytes,
            language,
            None,
            None,
            correlation_id,
            on_delta,
            on_progress,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn asr_transcribe_streaming_bytes_with_progress_and_runtime_context<F, P>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
        on_delta: F,
        on_progress: P,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
        P: FnMut(AsrProgress) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_callback_and_prompt_options_with_progress(
            variant,
            audio_bytes,
            language,
            None,
            AsrTaskOptions::default(),
            None,
            correlation_id,
            on_delta,
            on_progress,
            true,
            runtime_context,
            false,
        )
        .await
    }

    pub async fn asr_transcribe_bytes_with_progress_and_correlation<F, P>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
        on_progress: P,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
        P: FnMut(AsrProgress) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_callback_and_prompt_options_with_progress(
            variant,
            audio_bytes,
            language,
            None,
            AsrTaskOptions::default(),
            None,
            correlation_id,
            on_delta,
            on_progress,
            false,
            RuntimeRequestContext::default(),
            false,
        )
        .await
    }

    /// Transcribes (or, with `task.translate_to`, translates) with progress
    /// reports; the transcription jobs' path.
    #[allow(clippy::too_many_arguments)]
    pub async fn asr_transcribe_bytes_with_progress_and_runtime_context<F, P>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        task: AsrTaskOptions,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
        native_word_timestamps: bool,
        on_delta: F,
        on_progress: P,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
        P: FnMut(AsrProgress) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_callback_and_prompt_options_with_progress(
            variant,
            audio_bytes,
            language,
            None,
            task,
            None,
            correlation_id,
            on_delta,
            on_progress,
            false,
            runtime_context,
            native_word_timestamps,
        )
        .await
    }
}
