//! TTS execution preparation: loaded models, prepared artifacts, generation
//! parameters and stage costs.

use super::*;

impl EngineCoreRequest {
    pub(super) fn lfm25_audio_tts_stage_costs(
        binding: Option<&super::super::ExecutionAdapterBinding>,
        model: &crate::models::registry::NativeAudioChatModel,
        prompt_tokens: usize,
        max_new_tokens: usize,
    ) -> Result<Vec<(StageId, WorkCost)>> {
        let Some(binding) = binding else {
            return Ok(Vec::new());
        };
        let mut costs = Vec::new();
        for stage in binding.stages.iter().filter(|stage| {
            matches!(
                stage.selector,
                super::super::StageWorkSelector::SequencePrefill
                    | super::super::StageWorkSelector::SequenceDecode
            )
        }) {
            let envelope = if stage.selector == super::super::StageWorkSelector::SequencePrefill {
                model.lfm25_audio_tts_prefill_resource_envelope(0, prompt_tokens, prompt_tokens)?
            } else {
                let position = prompt_tokens
                    .checked_add(max_new_tokens.max(1))
                    .and_then(|value| value.checked_sub(1))
                    .ok_or_else(|| {
                        Error::Overloaded("LFM2.5 Audio TTS position overflowed".into())
                    })?;
                model.lfm25_audio_tts_decode_resource_envelope(position, true)?
            };
            let cost = WorkCost::with_workspace(
                envelope.work_units,
                envelope.materialized_tensor_elements,
                ResourceVector {
                    host_bytes: ResourceAmount::Known(envelope.host_workspace_bytes),
                    device_bytes: ResourceAmount::Known(envelope.device_workspace_bytes),
                    unified_bytes: ResourceAmount::Known(envelope.unified_workspace_bytes),
                    ..ResourceVector::zero()
                },
            );
            if !cost
                .workspace
                .workspace_bytes()
                .is_ok_and(|bytes| bytes <= stage.max_workspace_bytes)
            {
                return Err(Error::Overloaded(
                    "LFM2.5 Audio TTS workspace exceeds its loaded adapter budget".into(),
                ));
            }
            costs.push((stage.id, cost));
        }
        Ok(costs)
    }

    pub(super) fn continuous_tts_stage_cost(
        binding: Option<&super::super::ExecutionAdapterBinding>,
        model: &Qwen3TtsModel,
        prefill_tokens: usize,
        max_frames: usize,
    ) -> Result<Option<(StageId, WorkCost)>> {
        let Some(stage) = binding.and_then(|binding| {
            binding
                .stages
                .iter()
                .find(|stage| stage.batch_mode == super::super::NativeBatchMode::Continuous)
        }) else {
            return Ok(None);
        };
        if !model.supports_continuous_decode_batch() {
            return Err(Error::InvalidInput(
                "loaded adapter selected continuous decode for an incompatible TTS model".into(),
            ));
        }
        let (host_bytes, accelerator_bytes) =
            model.continuous_decode_bounded_workspace_per_row_bytes(prefill_tokens, max_frames)?;
        let cost = WorkCost::with_workspace(
            1,
            1,
            ResourceVector {
                host_bytes: ResourceAmount::Known(host_bytes),
                temporary_bytes: ResourceAmount::Known(accelerator_bytes),
                ..ResourceVector::zero()
            },
        );
        if !cost
            .workspace
            .workspace_bytes()
            .is_ok_and(|bytes| bytes <= stage.max_workspace_bytes)
        {
            return Err(Error::Overloaded(
                "continuous TTS decode workspace exceeds its loaded adapter budget".into(),
            ));
        }
        Ok(Some((stage.id, cost)))
    }

    pub(super) fn kokoro_tts_stage_cost(
        binding: Option<&super::super::ExecutionAdapterBinding>,
        prepared: &KokoroPreparedText,
    ) -> Result<Option<(StageId, WorkCost)>> {
        let Some(stage) = binding.and_then(|binding| {
            binding.stages.iter().find(|stage| {
                stage.selector == super::super::StageWorkSelector::Atomic
                    && stage.batch_mode == super::super::NativeBatchMode::Static
            })
        }) else {
            return Ok(None);
        };
        let tensor_elements = prepared
            .chunks
            .iter()
            .try_fold(0usize, |total, chunk| {
                total
                    .checked_add(chunk.token_ids.len())?
                    .checked_add(chunk.ref_style.elem_count())
            })
            .and_then(|elements| u64::try_from(elements).ok())
            .ok_or_else(|| Error::Overloaded("Kokoro prepared tensor size overflowed".into()))?;
        Ok(Some((stage.id, WorkCost::new(1, tensor_elements, 0))))
    }

    pub(super) fn fit_qwen_tts_layout_to_loaded_context(
        mut layout: crate::models::architectures::qwen3::tts::Qwen3TtsPhysicalSessionLayout,
        max_sequence_tokens: usize,
        request_id: &str,
    ) -> Result<crate::models::architectures::qwen3::tts::Qwen3TtsPhysicalSessionLayout> {
        let output_capacity = max_sequence_tokens
            .checked_sub(layout.prefill_tokens)
            .filter(|capacity| *capacity > 0)
            .ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Qwen TTS request {request_id} exact prefill has {} tokens and leaves no output capacity in the configured {max_sequence_tokens}-token context",
                    layout.prefill_tokens
                ))
            })?;
        layout.max_frames = layout.max_frames.min(output_capacity);
        Ok(layout)
    }

    pub(super) fn resumable_tts_prefill_stage_cost(
        binding: Option<&super::super::ExecutionAdapterBinding>,
        model: &Qwen3TtsModel,
        prefill_tokens: usize,
        max_frames: usize,
        has_reference: bool,
    ) -> Result<Option<(StageId, WorkCost)>> {
        let Some(stage) =
            Self::native_prepared_stage(binding, super::super::StageWorkSelector::SequencePrefill)
        else {
            return Ok(None);
        };
        let (host_bytes, accelerator_bytes) =
            model.resumable_prefill_workspace_bytes(prefill_tokens, max_frames, has_reference)?;
        let cost = WorkCost::with_workspace(
            1,
            1,
            ResourceVector {
                host_bytes: ResourceAmount::Known(host_bytes),
                temporary_bytes: ResourceAmount::Known(accelerator_bytes),
                ..ResourceVector::zero()
            },
        );
        if !cost
            .workspace
            .workspace_bytes()
            .is_ok_and(|bytes| bytes <= stage.max_workspace_bytes)
        {
            return Err(Error::Overloaded(
                "resumable TTS prefill workspace exceeds its loaded adapter budget".into(),
            ));
        }
        Ok(Some((stage.id, cost)))
    }

    /// Clamp TTS generation parameters once, before any preparation reads
    /// them. Preparation records the speed it prepared for and execution
    /// checks it against the request, so a later clamp (engine admission)
    /// failed every request outside the range with "changed after
    /// preparation" (KOK-20).
    pub(crate) fn normalize_tts_params_before_preparation(&mut self) -> Result<()> {
        if self.task_type == TaskType::TTS {
            self.params.speed = clamp_tts_speed(self.params.speed);
            self.params.speech.validate()?;
        }
        Ok(())
    }

    pub(crate) fn lfm25_audio_tts_messages_for_preparation(&self) -> Result<Vec<ChatMessage>> {
        let text = self
            .text
            .as_deref()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| {
                Error::InvalidInput("LFM2.5 Audio TTS request is missing text".into())
            })?;
        let speaker = self
            .params
            .speaker
            .as_deref()
            .or(self.params.voice.as_deref());
        Ok(vec![
            ChatMessage {
                role: ChatRole::System,
                content: lfm25_audio_tts_system_prompt(speaker).to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: text.trim().to_string(),
            },
        ])
    }

    pub(crate) fn install_lfm25_audio_tts_execution_model(
        &mut self,
        model_variant: ModelVariant,
        model: Lfm25AudioModelLease,
        artifact: Arc<Lfm25AudioPreparedTtsArtifact>,
        max_sequence_tokens: usize,
    ) -> Result<()> {
        if model_variant.family() != ModelFamily::Lfm25Audio
            || self.task_type != TaskType::TTS
            || self.model_variant != Some(model_variant)
        {
            return Err(Error::InvalidInput(format!(
                "LFM2.5 Audio TTS request {} model preparation does not match its routed task/model",
                self.id
            )));
        }
        let expected = self.lfm25_audio_tts_messages_for_preparation()?;
        if artifact.source_messages.len() != expected.len()
            || artifact
                .source_messages
                .iter()
                .zip(&expected)
                .any(|(actual, expected)| {
                    actual.role != expected.role || actual.content != expected.content
                })
            || artifact.prompt_tokens == 0
            || artifact.prompt_tokens >= max_sequence_tokens
        {
            return Err(Error::InvalidInput(format!(
                "LFM2.5 Audio TTS request {} prepared prompt does not match its exact input/context",
                self.id
            )));
        }
        let available = max_sequence_tokens - artifact.prompt_tokens;
        self.params.max_tokens = if self.params.max_tokens == 0 {
            1_024.min(available)
        } else {
            self.params.max_tokens.min(available)
        };
        if self.params.max_tokens == 0 {
            return Err(Error::InvalidInput(
                "LFM2.5 Audio TTS request leaves no output capacity".into(),
            ));
        }
        let stage_costs = Self::lfm25_audio_tts_stage_costs(
            self.execution_adapter_binding.as_ref(),
            &model,
            artifact.prompt_tokens,
            self.params.max_tokens,
        )?;
        self.prepared_stage_costs.clear();
        self.prepared_sequence_input_tokens = Some(artifact.prompt_tokens);
        self.incremental_model_execution_ready = Some(IncrementalModelExecutionReady {
            model_variant,
            model: PreparedIncrementalModel::Lfm25AudioTts(model),
            qwen_tts: None,
            lfm25_audio_tts: Some(artifact),
            vibevoice_tts: None,
            vibevoice_tts_params: None,
            fish_s2_tts: None,
            fish_s2_tts_params: None,
            voxtral_tts: None,
            voxtral_tts_params: None,
            chatterbox_tts: None,
            chatterbox_tts_params: None,
            maya1_tts: None,
            maya1_tts_params: None,
            maya2_tts: None,
            maya2_tts_params: None,
            lfm3_audio: None,
            kokoro_tts: None,
        });
        for (stage, cost) in stage_costs {
            self.install_prepared_stage_cost(stage, cost)?;
        }
        Ok(())
    }

    pub(crate) fn lfm25_audio_tts_generation_config(&self) -> Lfm25AudioGenerationConfig {
        // Unset fields keep the reference 0.8 / top-k 64; the seed defaults
        // to one derived from the request id, as audio chat does.
        let speech = &self.params.speech;
        Lfm25AudioGenerationConfig::tts(
            self.params.audio_temperature.or(speech.temperature),
            self.params.audio_top_k.or(speech.top_k),
            speech
                .seed
                .unwrap_or_else(|| Self::chat_request_seed(&self.id)),
        )
    }

    pub(crate) fn install_vibevoice_tts_execution_model(
        &mut self,
        model_variant: ModelVariant,
        model: VibeVoiceTtsModelLease,
        artifact: Arc<VibeVoiceTtsPreparedArtifact>,
        params: VibeVoiceTtsGenerationParams,
        max_sequence_tokens: usize,
    ) -> Result<()> {
        if self.task_type != TaskType::TTS
            || self.model_variant != Some(model_variant)
            || model_variant.family() != ModelFamily::VibeVoiceTts
            || artifact.prompt_tokens() == 0
            || artifact.prompt_tokens() >= max_sequence_tokens
            || params.max_frames == 0
        {
            return Err(Error::InvalidInput(format!(
                "VibeVoice TTS request {} preparation does not match its routed model/context",
                self.id
            )));
        }
        self.prepared_stage_costs.clear();
        self.prepared_sequence_input_tokens = Some(artifact.prompt_tokens());
        self.params.max_tokens = params.max_frames;
        self.incremental_model_execution_ready = Some(IncrementalModelExecutionReady {
            model_variant,
            model: PreparedIncrementalModel::VibeVoiceTts(model),
            qwen_tts: None,
            lfm25_audio_tts: None,
            vibevoice_tts: Some(artifact),
            vibevoice_tts_params: Some(params),
            fish_s2_tts: None,
            fish_s2_tts_params: None,
            voxtral_tts: None,
            voxtral_tts_params: None,
            chatterbox_tts: None,
            chatterbox_tts_params: None,
            maya1_tts: None,
            maya1_tts_params: None,
            maya2_tts: None,
            maya2_tts_params: None,
            lfm3_audio: None,
            kokoro_tts: None,
        });
        Ok(())
    }

    pub(crate) fn install_fish_s2_tts_execution_model(
        &mut self,
        model_variant: ModelVariant,
        model: FishS2TtsModelLease,
        artifact: Arc<FishS2PreparedArtifact>,
        params: FishS2GenerationParams,
        max_sequence_tokens: usize,
    ) -> Result<()> {
        if self.task_type != TaskType::TTS
            || self.model_variant != Some(model_variant)
            || model_variant.family() != ModelFamily::FishS2Tts
            || artifact.prompt_tokens() == 0
            || artifact.prompt_tokens() >= max_sequence_tokens
            || params.max_frames == 0
        {
            return Err(Error::InvalidInput(format!(
                "Fish S2 TTS request {} preparation does not match its routed model/context",
                self.id
            )));
        }
        // Validate exact reference + text + full output room before admission.
        crate::models::architectures::fish_s2::effective_frame_budget(
            artifact.prompt_tokens(),
            max_sequence_tokens,
            max_sequence_tokens,
            params.max_frames,
        )?;
        self.prepared_stage_costs.clear();
        self.prepared_sequence_input_tokens = Some(artifact.prompt_tokens());
        self.params.max_tokens = params.max_frames;
        self.incremental_model_execution_ready = Some(IncrementalModelExecutionReady {
            model_variant,
            model: PreparedIncrementalModel::FishS2Tts(model),
            qwen_tts: None,
            lfm25_audio_tts: None,
            vibevoice_tts: None,
            vibevoice_tts_params: None,
            fish_s2_tts: Some(artifact),
            fish_s2_tts_params: Some(params),
            voxtral_tts: None,
            voxtral_tts_params: None,
            chatterbox_tts: None,
            chatterbox_tts_params: None,
            maya1_tts: None,
            maya1_tts_params: None,
            maya2_tts: None,
            maya2_tts_params: None,
            lfm3_audio: None,
            kokoro_tts: None,
        });
        Ok(())
    }

    pub(crate) fn install_voxtral_tts_execution_model(
        &mut self,
        model_variant: ModelVariant,
        model: VoxtralTtsModelLease,
        artifact: Arc<VoxtralTtsPreparedArtifact>,
        params: VoxtralTtsGenerationParams,
        max_sequence_tokens: usize,
    ) -> Result<()> {
        if self.task_type != TaskType::TTS
            || self.model_variant != Some(model_variant)
            || model_variant.family() != ModelFamily::VoxtralTts
            || artifact.prompt_tokens == 0
            || artifact.prompt_tokens >= max_sequence_tokens
            || params.max_frames == 0
        {
            return Err(Error::InvalidInput(format!(
                "Voxtral TTS request {} preparation does not match its routed model/context",
                self.id
            )));
        }
        self.prepared_stage_costs.clear();
        self.prepared_sequence_input_tokens = Some(artifact.prompt_tokens);
        self.params.max_tokens = params.max_frames;
        self.incremental_model_execution_ready = Some(IncrementalModelExecutionReady {
            model_variant,
            model: PreparedIncrementalModel::VoxtralTts(model),
            qwen_tts: None,
            lfm25_audio_tts: None,
            vibevoice_tts: None,
            vibevoice_tts_params: None,
            fish_s2_tts: None,
            fish_s2_tts_params: None,
            voxtral_tts: Some(artifact),
            voxtral_tts_params: Some(params),
            chatterbox_tts: None,
            chatterbox_tts_params: None,
            maya1_tts: None,
            maya1_tts_params: None,
            maya2_tts: None,
            maya2_tts_params: None,
            lfm3_audio: None,
            kokoro_tts: None,
        });
        Ok(())
    }

    pub(crate) fn install_chatterbox_tts_execution_model(
        &mut self,
        model_variant: ModelVariant,
        model: ChatterboxTtsModelLease,
        artifact: Arc<ChatterboxPreparedArtifact>,
        params: ChatterboxGenerationParams,
        max_sequence_tokens: usize,
    ) -> Result<()> {
        if self.task_type != TaskType::TTS
            || self.model_variant != Some(model_variant)
            || model_variant.family() != ModelFamily::ChatterboxTts
            || artifact.prompt_tokens == 0
            || artifact.prompt_tokens + params.max_speech_tokens > max_sequence_tokens + 1
            || params.max_speech_tokens == 0
        {
            return Err(Error::InvalidInput(format!(
                "Chatterbox request {} preparation does not match its routed model/context",
                self.id
            )));
        }
        self.prepared_stage_costs.clear();
        self.prepared_sequence_input_tokens = Some(artifact.prompt_tokens);
        self.params.max_tokens = params.max_speech_tokens;
        self.incremental_model_execution_ready = Some(IncrementalModelExecutionReady {
            model_variant,
            model: PreparedIncrementalModel::ChatterboxTts(model),
            qwen_tts: None,
            lfm25_audio_tts: None,
            vibevoice_tts: None,
            vibevoice_tts_params: None,
            fish_s2_tts: None,
            fish_s2_tts_params: None,
            voxtral_tts: None,
            voxtral_tts_params: None,
            chatterbox_tts: Some(artifact),
            chatterbox_tts_params: Some(params),
            maya1_tts: None,
            maya1_tts_params: None,
            maya2_tts: None,
            maya2_tts_params: None,
            lfm3_audio: None,
            kokoro_tts: None,
        });
        Ok(())
    }

    pub(crate) fn install_maya1_tts_execution_model(
        &mut self,
        model_variant: ModelVariant,
        model: Maya1TtsModelLease,
        artifact: Arc<Maya1PreparedArtifact>,
        params: Maya1GenerationParams,
        max_sequence_tokens: usize,
    ) -> Result<()> {
        if self.task_type != TaskType::TTS
            || self.model_variant != Some(model_variant)
            || model_variant.family() != ModelFamily::Maya1Tts
            || artifact.prompt.is_empty()
            || artifact.prompt.len() + params.max_new_tokens > max_sequence_tokens + 1
            || params.max_new_tokens == 0
        {
            return Err(Error::InvalidInput(format!(
                "Maya1 request {} preparation does not match its routed model/context",
                self.id
            )));
        }
        self.prepared_stage_costs.clear();
        self.prepared_sequence_input_tokens = Some(artifact.prompt.len());
        self.params.max_tokens = params.max_new_tokens;
        self.incremental_model_execution_ready = Some(IncrementalModelExecutionReady {
            model_variant,
            model: PreparedIncrementalModel::Maya1Tts(model),
            qwen_tts: None,
            lfm25_audio_tts: None,
            vibevoice_tts: None,
            vibevoice_tts_params: None,
            fish_s2_tts: None,
            fish_s2_tts_params: None,
            voxtral_tts: None,
            voxtral_tts_params: None,
            chatterbox_tts: None,
            chatterbox_tts_params: None,
            maya1_tts: Some(artifact),
            maya1_tts_params: Some(params),
            maya2_tts: None,
            maya2_tts_params: None,
            lfm3_audio: None,
            kokoro_tts: None,
        });
        Ok(())
    }

    pub(crate) fn install_maya2_tts_execution_model(
        &mut self,
        model_variant: ModelVariant,
        model: Maya2TtsModelLease,
        artifact: Arc<Maya2PreparedArtifact>,
        params: Maya2GenerationParams,
        max_sequence_tokens: usize,
    ) -> Result<()> {
        let prompt = artifact.prompt.items.len();
        let frames = artifact.budget.cap;
        if self.task_type != TaskType::TTS
            || self.model_variant != Some(model_variant)
            || model_variant.family() != ModelFamily::Maya2Tts
            || prompt == 0
            || frames == 0
            || prompt + frames > max_sequence_tokens + 1
        {
            return Err(Error::InvalidInput(format!(
                "Maya2 request {} preparation does not match its routed model/context",
                self.id
            )));
        }
        self.prepared_stage_costs.clear();
        self.prepared_sequence_input_tokens = Some(prompt);
        self.params.max_tokens = frames;
        self.incremental_model_execution_ready = Some(IncrementalModelExecutionReady {
            model_variant,
            model: PreparedIncrementalModel::Maya2Tts(model),
            qwen_tts: None,
            lfm25_audio_tts: None,
            vibevoice_tts: None,
            vibevoice_tts_params: None,
            fish_s2_tts: None,
            fish_s2_tts_params: None,
            voxtral_tts: None,
            voxtral_tts_params: None,
            chatterbox_tts: None,
            chatterbox_tts_params: None,
            maya1_tts: None,
            maya1_tts_params: None,
            maya2_tts: Some(artifact),
            maya2_tts_params: Some(params),
            lfm3_audio: None,
            kokoro_tts: None,
        });
        Ok(())
    }

    /// The request's LFM3-Audio speech sampling: the model card's
    /// (temperature 1, top-k 128, a 1.1 repetition penalty over 4,096 frames)
    /// unless it sets its own.
    pub(crate) fn lfm3_audio_sampling(
        &self,
    ) -> crate::models::architectures::lfm3_audio::config::Lfm3AudioSampling {
        let defaults =
            crate::models::architectures::lfm3_audio::config::Lfm3AudioSampling::default();
        let speech = &self.params.speech;
        crate::models::architectures::lfm3_audio::config::Lfm3AudioSampling {
            temperature: self
                .params
                .audio_temperature
                .or(speech.temperature)
                .unwrap_or(defaults.temperature)
                .max(0.0),
            top_k: self
                .params
                .audio_top_k
                .or(speech.top_k)
                .unwrap_or(defaults.top_k),
            repetition_penalty: speech
                .repetition_penalty
                .unwrap_or(defaults.repetition_penalty),
            repetition_window: defaults.repetition_window,
        }
    }

    /// The LFM3-Audio speech this TTS request asks for: its text in its voice
    /// (`female` or `male`) and locale, at most its frame cap (from the text
    /// when unset), with its sampling and seed (else one derived from its id).
    /// Read it before preparation seals the request's slots.
    pub(crate) fn lfm3_audio_tts_request(
        &self,
    ) -> Result<crate::models::architectures::lfm3_audio::model::Lfm3AudioRequest> {
        use crate::models::architectures::lfm3_audio::model::{
            lfm3_audio_tts_auto_max_frames, tts_request,
        };
        let text = self
            .tts_text_for_execution()
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .ok_or_else(|| Error::InvalidInput("LFM3-Audio TTS request is missing text".into()))?;
        let voice = crate::models::architectures::lfm3_audio::prompts::Lfm3AudioVoice::resolve(
            self.params
                .voice
                .as_deref()
                .or(self.params.speaker.as_deref()),
            self.language.as_deref(),
        )?;
        let max_frames = match self.params.max_tokens {
            0 => lfm3_audio_tts_auto_max_frames(text),
            frames => frames,
        };
        let seed = self
            .params
            .speech
            .seed
            .unwrap_or_else(|| Self::chat_request_seed(&self.id));
        tts_request(text, voice, max_frames, self.lfm3_audio_sampling(), seed)
    }

    /// Seal an LFM3-Audio request as a retained row: its prompt embeddings
    /// are computed, and it decodes at most its request's slots (speech
    /// frames or text tokens, and `<|im_end|>`). A long-form transcription
    /// has no row: it keeps only the model, and chunks its prepared audio.
    pub(crate) fn install_lfm3_audio_execution_model(
        &mut self,
        model_variant: ModelVariant,
        model: Lfm25AudioModelLease,
        artifact: Option<Arc<Lfm3AudioPreparedArtifact>>,
        max_sequence_tokens: usize,
    ) -> Result<()> {
        let routed = self.model_variant == Some(model_variant)
            && model_variant.family() == ModelFamily::Lfm3Audio;
        let shaped = match (self.task_type, artifact.as_ref()) {
            (TaskType::TTS, Some(_)) => true,
            (TaskType::ASR, Some(_)) => !self.uses_asr_long_form_atomic(),
            (TaskType::ASR, None) => self.uses_asr_long_form_atomic(),
            _ => false,
        };
        let fits = artifact.as_ref().is_none_or(|artifact| {
            artifact.prompt_tokens > 0
                && artifact.request.max_slots > 0
                && artifact.prompt_tokens < max_sequence_tokens
        });
        if !routed || !shaped || !fits {
            return Err(Error::InvalidInput(format!(
                "LFM3-Audio request {} preparation does not match its routed model/context",
                self.id
            )));
        }
        self.prepared_stage_costs.clear();
        if let Some(artifact) = artifact.as_ref() {
            let prompt = artifact.prompt_tokens;
            self.prepared_sequence_input_tokens = Some(prompt);
            self.params.max_tokens = artifact.request.max_slots.min(max_sequence_tokens - prompt);
            if self.task_type == TaskType::ASR {
                self.prepared_asr_execution_shape =
                    Some(PreparedAsrExecutionShape::RetainedSequence {
                        input_tokens: prompt,
                    });
            }
        }
        self.incremental_model_execution_ready = Some(IncrementalModelExecutionReady {
            model_variant,
            model: PreparedIncrementalModel::Lfm3Audio(model),
            qwen_tts: None,
            lfm25_audio_tts: None,
            vibevoice_tts: None,
            vibevoice_tts_params: None,
            fish_s2_tts: None,
            fish_s2_tts_params: None,
            voxtral_tts: None,
            voxtral_tts_params: None,
            chatterbox_tts: None,
            chatterbox_tts_params: None,
            maya1_tts: None,
            maya1_tts_params: None,
            maya2_tts: None,
            maya2_tts_params: None,
            lfm3_audio: artifact,
            kokoro_tts: None,
        });
        Ok(())
    }

    pub(crate) fn install_kokoro_tts_execution_model(
        &mut self,
        model_variant: ModelVariant,
        model: KokoroTtsModelLease,
        artifact: Arc<KokoroPreparedText>,
        max_sequence_tokens: usize,
    ) -> Result<()> {
        let text = self
            .text
            .as_deref()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| Error::InvalidInput("Kokoro TTS request is missing text".into()))?;
        let speaker = self
            .params
            .speaker
            .as_deref()
            .or(self.params.voice.as_deref());
        let input_tokens = artifact.input_tokens()?;
        if self.task_type != TaskType::TTS
            || self.model_variant != Some(model_variant)
            || model_variant.family() != ModelFamily::KokoroTts
            || !artifact.matches_request(text, speaker, self.language.as_deref(), self.params.speed)
            || !artifact.is_well_formed()
            || artifact.longest_chunk_tokens() > max_sequence_tokens
        {
            return Err(Error::InvalidInput(format!(
                "Kokoro TTS request {} preparation does not match its routed input/model/context",
                self.id
            )));
        }
        let prepared_stage_cost =
            Self::kokoro_tts_stage_cost(self.execution_adapter_binding.as_ref(), &artifact)?;
        self.prepared_stage_costs.clear();
        self.prepared_sequence_input_tokens = Some(input_tokens);
        self.incremental_model_execution_ready = Some(IncrementalModelExecutionReady {
            model_variant,
            model: PreparedIncrementalModel::KokoroTts(model),
            qwen_tts: None,
            lfm25_audio_tts: None,
            vibevoice_tts: None,
            vibevoice_tts_params: None,
            fish_s2_tts: None,
            fish_s2_tts_params: None,
            voxtral_tts: None,
            voxtral_tts_params: None,
            chatterbox_tts: None,
            chatterbox_tts_params: None,
            maya1_tts: None,
            maya1_tts_params: None,
            maya2_tts: None,
            maya2_tts_params: None,
            lfm3_audio: None,
            kokoro_tts: Some(artifact),
        });
        if let Some((stage, cost)) = prepared_stage_cost {
            self.install_prepared_stage_cost(stage, cost)?;
        }
        Ok(())
    }

    pub(crate) fn install_qwen_tts_execution_model(
        &mut self,
        model_variant: ModelVariant,
        model: QwenTtsModelLease,
        reference: Option<Arc<SpeakerReference>>,
        max_sequence_tokens: usize,
    ) -> Result<()> {
        if self.task_type != TaskType::TTS
            || self.model_variant != Some(model_variant)
            || model_variant.family() != ModelFamily::Qwen3Tts
        {
            return Err(Error::InvalidInput(format!(
                "Qwen TTS request {} model preparation does not match its routed task/model",
                self.id
            )));
        }
        let model_arc = model.model_arc();
        self.prepared_stage_costs.clear();
        let text = self
            .text
            .as_deref()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| Error::InvalidInput("Qwen TTS request is missing text".to_string()))?;
        let speakers = model_arc.available_speakers();
        let speaker = if reference.is_some() || speakers.is_empty() {
            None
        } else {
            let requested = self
                .params
                .speaker
                .as_deref()
                .or(self.params.voice.as_deref())
                .filter(|speaker| !speaker.trim().is_empty())
                .unwrap_or_else(|| speakers[0].as_str());
            let resolved = speakers
                .iter()
                .find(|speaker| speaker.eq_ignore_ascii_case(requested))
                .ok_or_else(|| {
                    Error::InvalidInput(format!(
                        "Unknown speaker '{requested}'. Available speakers: {}",
                        speakers
                            .iter()
                            .map(|speaker| speaker.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                })?;
            Some((*resolved).clone())
        };
        let params = self.qwen_tts_generation_params();
        let layout = model_arc.physical_session_layout(TtsSessionCacheRequest {
            text,
            reference: reference.as_deref(),
            language: self.language.as_deref(),
            instruct: self.voice_description.as_deref(),
            uses_preset_speaker: speaker.is_some(),
            max_frames: params.max_frames,
        })?;
        let layout =
            Self::fit_qwen_tts_layout_to_loaded_context(layout, max_sequence_tokens, &self.id)?;
        let prepared_continuous_cost = Self::continuous_tts_stage_cost(
            self.execution_adapter_binding.as_ref(),
            &model_arc,
            layout.prefill_tokens,
            layout.max_frames,
        )?;
        let prepared_prefill_cost = Self::resumable_tts_prefill_stage_cost(
            self.execution_adapter_binding.as_ref(),
            &model_arc,
            layout.prefill_tokens,
            layout.max_frames,
            reference.is_some(),
        )?;
        if layout.prefill_tokens == 0 {
            return Err(Error::InvalidInput(format!(
                "Qwen TTS request {} exact prefill is empty",
                self.id
            )));
        }
        self.params.max_tokens = layout.max_frames;
        self.prepared_sequence_input_tokens = Some(layout.prefill_tokens);
        self.incremental_model_execution_ready = Some(IncrementalModelExecutionReady {
            model_variant,
            model: PreparedIncrementalModel::QwenTts(model),
            qwen_tts: Some(PreparedQwenTtsInput {
                reference,
                speaker,
                prefill_tokens: layout.prefill_tokens,
                max_frames: layout.max_frames,
            }),
            lfm25_audio_tts: None,
            vibevoice_tts: None,
            vibevoice_tts_params: None,
            fish_s2_tts: None,
            fish_s2_tts_params: None,
            voxtral_tts: None,
            voxtral_tts_params: None,
            chatterbox_tts: None,
            chatterbox_tts_params: None,
            maya1_tts: None,
            maya1_tts_params: None,
            maya2_tts: None,
            maya2_tts_params: None,
            lfm3_audio: None,
            kokoro_tts: None,
        });
        if let Some((stage_id, cost)) = prepared_continuous_cost {
            self.install_prepared_stage_cost(stage_id, cost)?;
        }
        if let Some((stage_id, cost)) = prepared_prefill_cost {
            self.install_prepared_stage_cost(stage_id, cost)?;
        }
        Ok(())
    }

    pub(crate) fn qwen_tts_generation_params(&self) -> TtsGenerationParams {
        let model_max_frames = self
            .model_variant
            .and_then(|variant| variant.tts_max_output_frames_hint())
            .unwrap_or(ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES);
        let prepared_max_frames = self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| ready.qwen_tts.as_ref())
            .map(|prepared| prepared.max_frames);
        let max_frames = if let Some(max_frames) = prepared_max_frames {
            max_frames
        } else if self.params.max_tokens == 0 {
            model_max_frames
        } else {
            self.params.max_tokens.clamp(16, model_max_frames.max(16))
        };
        TtsGenerationParams::for_request(&self.params.speech, max_frames)
    }

    pub(crate) fn prepared_lfm25_audio_tts_model_lease_for_executor(
        &self,
    ) -> Result<Option<Lfm25AudioModelLease>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::Lfm25AudioTts(model) => Some(model.clone()),
                _ => None,
            }))
    }

    pub(crate) fn prepared_lfm25_audio_tts_artifact_for_executor(
        &self,
    ) -> Result<Option<Arc<Lfm25AudioPreparedTtsArtifact>>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| ready.lfm25_audio_tts.clone()))
    }

    pub(crate) fn prepared_vibevoice_tts_model_lease_for_executor(
        &self,
    ) -> Result<Option<VibeVoiceTtsModelLease>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::VibeVoiceTts(model) => Some(model.clone()),
                _ => None,
            }))
    }

    pub(crate) fn prepared_vibevoice_tts_artifact_for_executor(
        &self,
    ) -> Result<Option<Arc<VibeVoiceTtsPreparedArtifact>>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| ready.vibevoice_tts.clone()))
    }

    pub(crate) fn vibevoice_tts_generation_params_for_executor(
        &self,
    ) -> Result<Option<VibeVoiceTtsGenerationParams>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| ready.vibevoice_tts_params.clone()))
    }

    pub(crate) fn prepared_fish_s2_tts_model_lease_for_executor(
        &self,
    ) -> Result<Option<FishS2TtsModelLease>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::FishS2Tts(model) => Some(model.clone()),
                _ => None,
            }))
    }

    pub(crate) fn prepared_fish_s2_tts_artifact_for_executor(
        &self,
    ) -> Result<Option<Arc<FishS2PreparedArtifact>>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| ready.fish_s2_tts.clone()))
    }

    pub(crate) fn fish_s2_tts_generation_params_for_executor(
        &self,
    ) -> Result<Option<FishS2GenerationParams>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| ready.fish_s2_tts_params.clone()))
    }

    pub(crate) fn prepared_voxtral_tts_model_lease_for_executor(
        &self,
    ) -> Result<Option<VoxtralTtsModelLease>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::VoxtralTts(model) => Some(model.clone()),
                _ => None,
            }))
    }

    pub(crate) fn prepared_voxtral_tts_artifact_for_executor(
        &self,
    ) -> Result<Option<Arc<VoxtralTtsPreparedArtifact>>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| ready.voxtral_tts.clone()))
    }

    pub(crate) fn voxtral_tts_generation_params_for_executor(
        &self,
    ) -> Result<Option<VoxtralTtsGenerationParams>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| ready.voxtral_tts_params))
    }

    pub(crate) fn prepared_chatterbox_tts_model_lease_for_executor(
        &self,
    ) -> Result<Option<ChatterboxTtsModelLease>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::ChatterboxTts(model) => Some(model.clone()),
                _ => None,
            }))
    }

    pub(crate) fn prepared_chatterbox_tts_artifact_for_executor(
        &self,
    ) -> Result<Option<Arc<ChatterboxPreparedArtifact>>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| ready.chatterbox_tts.clone()))
    }

    pub(crate) fn chatterbox_tts_generation_params_for_executor(
        &self,
    ) -> Result<Option<ChatterboxGenerationParams>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| ready.chatterbox_tts_params.clone()))
    }

    pub(crate) fn prepared_maya1_tts_model_lease_for_executor(
        &self,
    ) -> Result<Option<Maya1TtsModelLease>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::Maya1Tts(model) => Some(model.clone()),
                _ => None,
            }))
    }

    pub(crate) fn prepared_maya1_tts_artifact_for_executor(
        &self,
    ) -> Result<Option<Arc<Maya1PreparedArtifact>>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| ready.maya1_tts.clone()))
    }

    pub(crate) fn maya1_tts_generation_params_for_executor(
        &self,
    ) -> Result<Option<Maya1GenerationParams>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| ready.maya1_tts_params))
    }

    pub(crate) fn prepared_maya2_tts_model_lease_for_executor(
        &self,
    ) -> Result<Option<Maya2TtsModelLease>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::Maya2Tts(model) => Some(model.clone()),
                _ => None,
            }))
    }

    pub(crate) fn prepared_maya2_tts_artifact_for_executor(
        &self,
    ) -> Result<Option<Arc<Maya2PreparedArtifact>>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| ready.maya2_tts.clone()))
    }

    pub(crate) fn maya2_tts_generation_params_for_executor(
        &self,
    ) -> Result<Option<Maya2GenerationParams>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| ready.maya2_tts_params))
    }

    pub(crate) fn prepared_lfm3_audio_model_lease_for_executor(
        &self,
    ) -> Result<Option<Lfm25AudioModelLease>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::Lfm3Audio(model) => Some(model.clone()),
                _ => None,
            }))
    }

    pub(crate) fn prepared_lfm3_audio_artifact_for_executor(
        &self,
    ) -> Result<Option<Arc<Lfm3AudioPreparedArtifact>>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| ready.lfm3_audio.clone()))
    }

    pub(crate) fn prepared_kokoro_tts_model_lease_for_executor(
        &self,
    ) -> Result<Option<KokoroTtsModelLease>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::KokoroTts(model) => Some(model.clone()),
                _ => None,
            }))
    }

    pub(crate) fn prepared_kokoro_tts_artifact_for_executor(
        &self,
    ) -> Result<Option<Arc<KokoroPreparedText>>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| ready.kokoro_tts.clone()))
    }

    pub(crate) fn prepared_qwen_tts_model_for_executor(
        &self,
    ) -> Result<Option<Arc<Qwen3TtsModel>>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::QwenTts(model) => Some(model.model_arc()),
                PreparedIncrementalModel::Asr(_)
                | PreparedIncrementalModel::Lfm25AudioAsr(_)
                | PreparedIncrementalModel::Lfm25AudioTts(_) => None,
                PreparedIncrementalModel::VibeVoiceTts(_)
                | PreparedIncrementalModel::FishS2Tts(_)
                | PreparedIncrementalModel::VoxtralTts(_)
                | PreparedIncrementalModel::ChatterboxTts(_)
                | PreparedIncrementalModel::Maya1Tts(_)
                | PreparedIncrementalModel::Maya2Tts(_)
                | PreparedIncrementalModel::Lfm3Audio(_)
                | PreparedIncrementalModel::KokoroTts(_) => None,
            }))
    }

    pub(crate) fn prepared_qwen_tts_model_lease_for_executor(
        &self,
    ) -> Result<Option<QwenTtsModelLease>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::QwenTts(model) => Some(model.clone()),
                PreparedIncrementalModel::Asr(_)
                | PreparedIncrementalModel::Lfm25AudioAsr(_)
                | PreparedIncrementalModel::Lfm25AudioTts(_) => None,
                PreparedIncrementalModel::VibeVoiceTts(_)
                | PreparedIncrementalModel::FishS2Tts(_)
                | PreparedIncrementalModel::VoxtralTts(_)
                | PreparedIncrementalModel::ChatterboxTts(_)
                | PreparedIncrementalModel::Maya1Tts(_)
                | PreparedIncrementalModel::Maya2Tts(_)
                | PreparedIncrementalModel::Lfm3Audio(_)
                | PreparedIncrementalModel::KokoroTts(_) => None,
            }))
    }

    pub(crate) fn prepared_qwen_tts_input_for_executor(&self) -> Result<&PreparedQwenTtsInput> {
        self.validate_incremental_model_execution_preparation()?;
        self.incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| ready.qwen_tts.as_ref())
            .ok_or_else(|| {
                Error::InferenceError(
                    "Qwen TTS request is missing exact host preparation".to_string(),
                )
            })
    }
}
