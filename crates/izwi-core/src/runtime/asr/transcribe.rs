//! Offline and streaming transcription entry points and the ASR request they
//! build.

use super::*;

impl RuntimeService {
    pub(super) async fn asr_transcribe_audio_chat<F>(
        &self,
        variant: ModelVariant,
        audio_input: AsrAudioInput<'_>,
        max_tokens: Option<usize>,
        streaming_required: bool,
        on_delta: F,
        request_id: String,
        runtime_context: RuntimeRequestContext,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        let input_bytes = audio_input.input_bytes();
        if input_bytes == 0 {
            return Err(Error::InvalidInput(
                "ASR request missing audio input".to_string(),
            ));
        }
        audio_input.validate_retained_size()?;
        let spec = self.coordinator_job_for_audio_input(
            request_id,
            CoordinatorLane::Atomic,
            runtime_context,
            input_bytes,
        )?;
        let job = self
            .coordinator
            .admit_observed(spec, host_input_observation(input_bytes)?)
            .await?;

        // Establish an owned packet only after admission. The caller-owned
        // allocation remains live while this future runs, so both allocations
        // are reconciled until the blocking decoder consumes the owned copy.
        let audio_input = audio_input.into_owned_for_job(&job).await?;
        job.record_materialized_usage(retained_host_observation(&[
            input_bytes,
            audio_input.retained_bytes(),
        ])?)?;

        let (residency_lease, execution_contract, state_binding) = self
            .load_capability_with_state_for_job(
                &job,
                variant,
                CapabilityKind::Asr,
                streaming_required,
                ExecutionTargetKind::DirectModel,
                CapabilityLoadOptions::default(),
            )
            .await?;
        let model = self
            .model_registry
            .get_audio_chat(variant)
            .await
            .ok_or_else(|| {
                Error::ModelNotFound(format!("Audio-chat model {variant} is not loaded"))
            })?;
        let observation_job = job.clone();
        self.coordinator
            .run_loaded_blocking_stage_with_invocation_workspace(
                &job,
                execution_contract,
                state_binding,
                WorkUnit::AtomicJob {
                    kind: "asr.audio_chat".to_string(),
                },
                move |leases| {
                    let _residency_lease = residency_lease;
                    let retained_audio_bytes = audio_input.retained_bytes();
                    let (samples, sample_rate) = audio_input.decode()?;
                    let steady_usage = decoded_audio_observation(input_bytes, samples.capacity())?;
                    observation_job.record_materialized_usage(add_retained_host_bytes(
                        steady_usage,
                        retained_audio_bytes,
                    )?)?;
                    observation_job.prepare_materialized_release(steady_usage)?;
                    drop(audio_input);
                    let duration_secs = if sample_rate > 0 {
                        samples.len() as f32 / sample_rate as f32
                    } else {
                        0.0
                    };
                    let mut on_delta = on_delta;
                    let mut delta_sink = |delta: &str| {
                        if !delta.is_empty() {
                            on_delta(delta.to_string());
                        }
                    };
                    let output = model
                        .transcribe_with_callback_and_max_tokens_from_invocation_workspace(
                            &samples,
                            sample_rate,
                            max_tokens,
                            leases,
                            &mut delta_sink,
                        )?;

                    Ok(AsrTranscription {
                        text: asr_transcript_text(&output.text),
                        language: output.language,
                        duration_secs,
                        asr_diagnostics: output.diagnostics,
                    })
                },
            )
            .await
    }

    pub(super) async fn build_asr_request(
        &self,
        variant: ModelVariant,
        audio_input: AsrAudioInput<'_>,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
        streaming: bool,
        native_word_timestamps: bool,
    ) -> Result<AdmittedEngineRequest> {
        validate_asr_language(variant, language)?;
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
        if variant.is_audio_chat() && variant.family() != ModelFamily::Lfm25Audio {
            self.observe_broker_capability_request(CapabilityKind::Asr, Some(variant), false)?;
            let context = RuntimeRequestContext::default();
            return self
                .asr_transcribe_audio_chat(
                    variant,
                    AsrAudioInput::Base64(audio_base64),
                    max_tokens,
                    false,
                    |_delta| {},
                    correlation_id
                        .map(ToOwned::to_owned)
                        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                    context,
                )
                .await;
        }

        let admitted = self
            .build_asr_request(
                variant,
                AsrAudioInput::Base64(audio_base64),
                language,
                prompt,
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
            language: asr_response_language(language, output.asr_diagnostics.as_ref()),
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
        if variant.is_audio_chat() && variant.family() != ModelFamily::Lfm25Audio {
            self.observe_broker_capability_request(CapabilityKind::Asr, Some(variant), true)?;
            let context = RuntimeRequestContext::new(crate::engine::WorkloadClass::Streaming);
            return self
                .asr_transcribe_audio_chat(
                    variant,
                    AsrAudioInput::Base64(audio_base64),
                    max_tokens,
                    true,
                    on_delta,
                    correlation_id
                        .map(ToOwned::to_owned)
                        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                    context,
                )
                .await;
        }

        let admitted = self
            .build_asr_request(
                variant,
                AsrAudioInput::Base64(audio_base64),
                language,
                prompt,
                max_tokens,
                correlation_id,
                RuntimeRequestContext::default(),
                true,
                false,
            )
            .await?;
        self.stream_admitted_asr_request(admitted, variant, true, language, on_delta, |_| {})
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
            max_tokens,
            correlation_id,
            RuntimeRequestContext::default(),
            false,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn asr_transcribe_bytes_with_variant_and_prompt_options_and_runtime_context(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
        native_word_timestamps: bool,
    ) -> Result<AsrTranscription> {
        if variant.is_audio_chat() && variant.family() != ModelFamily::Lfm25Audio {
            self.observe_broker_capability_request(CapabilityKind::Asr, Some(variant), false)?;
            return self
                .asr_transcribe_audio_chat(
                    variant,
                    AsrAudioInput::Bytes(audio_bytes),
                    max_tokens,
                    false,
                    |_delta| {},
                    correlation_id
                        .map(ToOwned::to_owned)
                        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                    runtime_context,
                )
                .await;
        }

        let admitted = self
            .build_asr_request(
                variant,
                AsrAudioInput::Bytes(audio_bytes),
                language,
                prompt,
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
            language: asr_response_language(language, output.asr_diagnostics.as_ref()),
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
            max_tokens,
            correlation_id,
            on_delta,
            on_progress,
            true,
            RuntimeRequestContext::default(),
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
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        on_delta: F,
        on_progress: P,
        broker_streaming_required: bool,
        runtime_context: RuntimeRequestContext,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
        P: FnMut(AsrProgress) + Send + 'static,
    {
        if variant.is_audio_chat() && variant.family() != ModelFamily::Lfm25Audio {
            self.observe_broker_capability_request(
                CapabilityKind::Asr,
                Some(variant),
                broker_streaming_required,
            )?;
            return self
                .asr_transcribe_audio_chat(
                    variant,
                    AsrAudioInput::Bytes(audio_bytes),
                    max_tokens,
                    broker_streaming_required,
                    on_delta,
                    correlation_id
                        .map(ToOwned::to_owned)
                        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                    runtime_context,
                )
                .await;
        }

        let admitted = self
            .build_asr_request(
                variant,
                AsrAudioInput::Bytes(audio_bytes),
                language,
                prompt,
                max_tokens,
                correlation_id,
                runtime_context,
                true,
                false,
            )
            .await?;
        self.stream_admitted_asr_request(
            admitted,
            variant,
            broker_streaming_required,
            language,
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
    pub(super) async fn stream_admitted_asr_request<F, P>(
        &self,
        admitted: AdmittedEngineRequest,
        variant: ModelVariant,
        native_stream_requested: bool,
        language: Option<&str>,
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
            language: asr_response_language(language, output.asr_diagnostics.as_ref()),
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
            max_tokens,
            correlation_id,
            runtime_context,
            false,
        )
        .await
    }

    /// Like [`Self::asr_transcribe_bytes_with_runtime_context`], optionally
    /// asking for model-native word timestamps: a model that emits them
    /// (Granite Speech Plus) returns them as `timestamp_words` in its
    /// diagnostics; any other model transcribes as usual.
    #[allow(clippy::too_many_arguments)]
    pub async fn asr_transcribe_bytes_with_runtime_context_and_word_timestamps(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
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

    #[allow(clippy::too_many_arguments)]
    pub async fn asr_transcribe_streaming_bytes_with_runtime_context<F>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
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
            max_tokens,
            correlation_id,
            on_delta,
            |_progress| {},
            true,
            runtime_context,
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
            None,
            correlation_id,
            on_delta,
            on_progress,
            true,
            runtime_context,
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
            None,
            correlation_id,
            on_delta,
            on_progress,
            false,
            RuntimeRequestContext::default(),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn asr_transcribe_bytes_with_progress_and_runtime_context<F, P>(
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
            None,
            correlation_id,
            on_delta,
            on_progress,
            false,
            runtime_context,
        )
        .await
    }
}
