//! ASR execution preparation: loaded models, prepared audio and encoder
//! artifacts, and stage costs.

use super::*;

impl EngineCoreRequest {
    pub(crate) fn install_asr_execution_model(
        &mut self,
        model_variant: ModelVariant,
        model: AsrModelLease,
    ) -> Result<()> {
        if self.task_type != TaskType::ASR || self.model_variant != Some(model_variant) {
            return Err(Error::InvalidInput(format!(
                "ASR request {} model preparation does not match its routed task/model",
                self.id
            )));
        }
        let prepared_continuous_cost = self
            .uses_asr_retained_sequence()
            .then(|| {
                Self::continuous_asr_stage_cost(self.execution_adapter_binding.as_ref(), &model)
            })
            .transpose()?
            .flatten();
        self.prepared_stage_costs.clear();
        self.incremental_model_execution_ready = Some(IncrementalModelExecutionReady {
            model_variant,
            model: PreparedIncrementalModel::Asr(model),
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
            kokoro_tts: None,
        });
        if let Some((stage_id, cost)) = prepared_continuous_cost {
            self.install_prepared_stage_cost(stage_id, cost)?;
        }
        Ok(())
    }

    pub(crate) fn install_lfm25_audio_asr_execution_model(
        &mut self,
        model_variant: ModelVariant,
        model: Lfm25AudioModelLease,
    ) -> Result<()> {
        if model_variant.family() != ModelFamily::Lfm25Audio
            || self.task_type != TaskType::ASR
            || self.model_variant != Some(model_variant)
        {
            return Err(Error::InvalidInput(format!(
                "LFM2.5 Audio ASR request {} model preparation does not match its routed task/model",
                self.id
            )));
        }
        // The service path installs the encoder artifact before the model
        // lease, when the stage costs cannot be priced yet; price them here
        // from the artifact's prompt so the stages are never admitted as free
        // (LFA-2).
        let prepared_costs = match self.prepared_lfm25_audio_asr_artifact_for_executor()? {
            Some(artifact) => Self::lfm25_audio_asr_stage_costs(
                self.execution_adapter_binding.as_ref(),
                &model,
                artifact.prompt_tokens,
                self.params.max_tokens,
            )?,
            None => Vec::new(),
        };
        self.prepared_stage_costs.clear();
        self.incremental_model_execution_ready = Some(IncrementalModelExecutionReady {
            model_variant,
            model: PreparedIncrementalModel::Lfm25AudioAsr(model),
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
            kokoro_tts: None,
        });
        for (stage_id, cost) in prepared_costs {
            self.install_prepared_stage_cost(stage_id, cost)?;
        }
        Ok(())
    }

    pub(super) fn continuous_asr_stage_cost(
        binding: Option<&super::super::ExecutionAdapterBinding>,
        model: &NativeAsrModel,
    ) -> Result<Option<(StageId, WorkCost)>> {
        let Some(stage) = binding.and_then(|binding| {
            binding
                .stages
                .iter()
                .find(|stage| stage.batch_mode == super::super::NativeBatchMode::Continuous)
        }) else {
            return Ok(None);
        };
        if matches!(model, NativeAsrModel::Parakeet(_)) {
            let cost = WorkCost::new(
                1,
                1,
                crate::models::architectures::parakeet::asr::PARAKEET_RETAINED_WORKSPACE_PER_ROW_BYTES,
            );
            if !cost
                .workspace
                .workspace_bytes()
                .is_ok_and(|bytes| bytes <= stage.max_workspace_bytes)
            {
                return Err(Error::Overloaded(
                    "Parakeet retained decode workspace exceeds its loaded adapter budget"
                        .to_string(),
                ));
            }
            return Ok(Some((stage.id, cost)));
        }
        if !model.supports_continuous_decode_batch() {
            return Err(Error::InvalidInput(
                "loaded adapter selected continuous decode for an incompatible ASR model"
                    .to_string(),
            ));
        }
        let accelerator_bytes = model.continuous_decode_batch_workspace_per_row_bytes()?;
        let host_bytes = super::super::continuous_asr_host_workspace_per_row_bytes()?;
        let cost = WorkCost::with_workspace(
            1,
            1,
            ResourceVector {
                host_bytes: ResourceAmount::Known(host_bytes),
                temporary_bytes: ResourceAmount::Known(accelerator_bytes),
                ..ResourceVector::zero()
            },
        );
        if super::super::continuous_asr_workspace_per_row_bytes(accelerator_bytes)?
            > stage.max_workspace_bytes
        {
            return Err(Error::Overloaded(
                "continuous ASR decode workspace exceeds its loaded adapter budget".to_string(),
            ));
        }
        Ok(Some((stage.id, cost)))
    }

    pub(super) fn lfm25_audio_asr_stage_costs(
        binding: Option<&super::super::ExecutionAdapterBinding>,
        model: &crate::models::registry::NativeAudioChatModel,
        prompt_tokens: usize,
        max_new_tokens: usize,
    ) -> Result<Vec<(StageId, WorkCost)>> {
        Self::lfm25_audio_asr_stage_costs_from_envelopes(binding, |selector| {
            // Reserve a complete prompt span even when the scheduler divides
            // it into smaller quanta. This also bounds singleton span prefill;
            // the per-row cost is aggregated once by physical batch admission.
            if selector == super::super::StageWorkSelector::SequencePrefill {
                model.lfm25_audio_asr_prefill_resource_envelope(0, prompt_tokens, prompt_tokens)
            } else {
                let position = prompt_tokens
                    .checked_add(max_new_tokens.max(1))
                    .and_then(|value| value.checked_sub(1))
                    .ok_or_else(|| {
                        Error::Overloaded("LFM2.5 Audio ASR decode position overflowed".into())
                    })?;
                model.lfm25_audio_asr_decode_resource_envelope(position)
            }
        })
    }

    pub(super) fn lfm25_audio_asr_stage_costs_from_envelopes(
        binding: Option<&super::super::ExecutionAdapterBinding>,
        mut envelope_for: impl FnMut(
            super::super::StageWorkSelector,
        ) -> Result<
            crate::models::architectures::lfm25_audio::model::Lfm25AudioAsrStepResourceEnvelope,
        >,
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
            let envelope = envelope_for(stage.selector)?;
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
                    "LFM2.5 Audio ASR workspace exceeds its loaded adapter budget".into(),
                ));
            }
            costs.push((stage.id, cost));
        }
        Ok(costs)
    }

    pub(crate) fn install_prepared_asr_audio(
        &mut self,
        model_variant: ModelVariant,
        samples: Vec<f32>,
        sample_rate: u32,
    ) -> Result<()> {
        if sample_rate == 0 || samples.is_empty() {
            return Err(Error::InvalidInput(format!(
                "ASR request {} decoded to an empty or zero-rate audio artifact",
                self.id
            )));
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(Error::InvalidInput(format!(
                "ASR request {} decoded to non-finite audio samples",
                self.id
            )));
        }
        let source_fingerprint = self.refresh_asr_execution_source_fingerprint(model_variant)?;
        if let Some(current) = self.prepared_asr_audio.as_ref() {
            if current.model_variant == model_variant
                && current.sample_rate == sample_rate
                && current.source_fingerprint == source_fingerprint
                && current.samples.as_ref() == samples.as_slice()
            {
                return Ok(());
            }
            return Err(Error::InvalidInput(format!(
                "ASR request {} changed its prepared decoded-audio artifact",
                self.id
            )));
        }
        self.prepared_asr_audio = Some(PreparedAsrAudio {
            model_variant,
            samples: Arc::from(samples.into_boxed_slice()),
            sample_rate,
            source_fingerprint,
        });
        Ok(())
    }

    pub(crate) fn prepared_asr_audio_for_executor(&self) -> Result<Option<(Arc<[f32]>, u32)>> {
        let Some(prepared) = self.prepared_asr_audio.as_ref() else {
            return Ok(None);
        };
        if self.model_variant != Some(prepared.model_variant)
            || self.asr_execution_source_fingerprint(prepared.model_variant)?
                != prepared.source_fingerprint
        {
            return Err(Error::InvalidInput(format!(
                "ASR request {} changed after decoded-audio preparation",
                self.id
            )));
        }
        Ok(Some((prepared.samples.clone(), prepared.sample_rate)))
    }

    /// Duration of the canonical input audio decoded at preparation. ASR
    /// executors report generated audio, which is empty for transcription, so
    /// this is the request's record of how much audio it transcribes.
    pub(crate) fn prepared_asr_input_duration_secs(&self) -> Option<f32> {
        self.prepared_asr_audio
            .as_ref()
            .map(|prepared| (prepared.samples.len() as f64 / prepared.sample_rate as f64) as f32)
    }

    pub(crate) fn prepared_asr_audio_retained_bytes(&self) -> Result<usize> {
        self.prepared_asr_audio.as_ref().map_or(Ok(0), |prepared| {
            prepared
                .samples
                .len()
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| {
                    Error::Overloaded("prepared ASR decoded-audio storage overflowed".to_string())
                })
        })
    }

    pub(crate) fn install_prepared_asr_encoder_artifact(
        &mut self,
        model_variant: ModelVariant,
        artifact: Arc<Qwen3AsrPreparedAudio>,
    ) -> Result<()> {
        if self.task_type != TaskType::ASR || self.model_variant != Some(model_variant) {
            return Err(Error::InvalidInput(format!(
                "ASR request {} encoder artifact does not match its routed task/model",
                self.id
            )));
        }
        if self.uses_asr_long_form_atomic() {
            return Err(Error::InvalidInput(format!(
                "long-form ASR request {} cannot retain a normal-route encoder artifact",
                self.id
            )));
        }
        let source_fingerprint = self.refresh_asr_execution_source_fingerprint(model_variant)?;
        if let Some(current) = self.prepared_asr_encoder_artifact.as_ref() {
            if current.model_variant == model_variant
                && current.source_fingerprint == source_fingerprint
                && matches!(&current.artifact, PreparedAsrEncoderArtifactValue::Qwen3(value) if Arc::ptr_eq(value, &artifact))
            {
                return Ok(());
            }
            return Err(Error::InvalidInput(format!(
                "ASR request {} changed its prepared encoder artifact",
                self.id
            )));
        }
        self.prepared_asr_encoder_artifact = Some(PreparedAsrEncoderArtifact {
            model_variant,
            artifact: PreparedAsrEncoderArtifactValue::Qwen3(artifact),
            source_fingerprint,
        });
        Ok(())
    }

    pub(crate) fn prepared_asr_encoder_artifact_for_executor(
        &self,
    ) -> Result<Option<Arc<Qwen3AsrPreparedAudio>>> {
        let Some(prepared) = self.prepared_asr_encoder_artifact.as_ref() else {
            return Ok(None);
        };
        if self.model_variant != Some(prepared.model_variant)
            || self.asr_execution_source_fingerprint(prepared.model_variant)?
                != prepared.source_fingerprint
            || self.uses_asr_long_form_atomic()
        {
            return Err(Error::InvalidInput(format!(
                "ASR request {} changed after encoder preparation",
                self.id
            )));
        }
        match &prepared.artifact {
            PreparedAsrEncoderArtifactValue::Qwen3(value) => Ok(Some(value.clone())),
            PreparedAsrEncoderArtifactValue::Whisper(_)
            | PreparedAsrEncoderArtifactValue::VibeVoice(_)
            | PreparedAsrEncoderArtifactValue::GraniteSpeech(_)
            | PreparedAsrEncoderArtifactValue::Lfm25Audio(_)
            | PreparedAsrEncoderArtifactValue::Parakeet(_) => Err(Error::InvalidInput(
                "foreign encoder artifact was requested through the Qwen3 accessor".into(),
            )),
        }
    }

    pub(crate) fn install_prepared_whisper_window(
        &mut self,
        model_variant: ModelVariant,
        artifact: Arc<WhisperPreparedWindow>,
    ) -> Result<()> {
        if model_variant.family() != crate::catalog::ModelFamily::WhisperAsr
            || self.task_type != TaskType::ASR
            || self.model_variant != Some(model_variant)
            || self.uses_asr_long_form_atomic()
        {
            return Err(Error::InvalidInput(format!(
                "ASR request {} Whisper artifact does not match its normal routed model",
                self.id
            )));
        }
        let source_fingerprint = self.refresh_asr_execution_source_fingerprint(model_variant)?;
        if self.prepared_asr_encoder_artifact.is_some() {
            return Err(Error::InvalidInput(format!(
                "ASR request {} already owns an encoder artifact",
                self.id
            )));
        }
        self.prepared_asr_encoder_artifact = Some(PreparedAsrEncoderArtifact {
            model_variant,
            artifact: PreparedAsrEncoderArtifactValue::Whisper(artifact),
            source_fingerprint,
        });
        Ok(())
    }

    pub(crate) fn prepared_whisper_window_for_executor(
        &self,
    ) -> Result<Option<Arc<WhisperPreparedWindow>>> {
        let Some(prepared) = self.prepared_asr_encoder_artifact.as_ref() else {
            return Ok(None);
        };
        if self.model_variant != Some(prepared.model_variant)
            || self.asr_execution_source_fingerprint(prepared.model_variant)?
                != prepared.source_fingerprint
            || self.uses_asr_long_form_atomic()
        {
            return Err(Error::InvalidInput(format!(
                "ASR request {} changed after Whisper encoder preparation",
                self.id
            )));
        }
        match &prepared.artifact {
            PreparedAsrEncoderArtifactValue::Whisper(value) => Ok(Some(value.clone())),
            PreparedAsrEncoderArtifactValue::Qwen3(_)
            | PreparedAsrEncoderArtifactValue::VibeVoice(_)
            | PreparedAsrEncoderArtifactValue::GraniteSpeech(_)
            | PreparedAsrEncoderArtifactValue::Lfm25Audio(_)
            | PreparedAsrEncoderArtifactValue::Parakeet(_) => Err(Error::InvalidInput(
                "foreign encoder artifact was requested through the Whisper accessor".into(),
            )),
        }
    }

    pub(crate) fn install_prepared_vibevoice_artifact(
        &mut self,
        model_variant: ModelVariant,
        artifact: Arc<VibeVoiceAsrPreparedArtifact>,
    ) -> Result<()> {
        if model_variant.family() != ModelFamily::VibeVoiceAsr
            || self.task_type != TaskType::ASR
            || self.model_variant != Some(model_variant)
            || self.uses_asr_long_form_atomic()
        {
            return Err(Error::InvalidInput(format!(
                "ASR request {} VibeVoice artifact does not match its normal routed model",
                self.id
            )));
        }
        let source_fingerprint = self.refresh_asr_execution_source_fingerprint(model_variant)?;
        if self.prepared_asr_encoder_artifact.is_some() {
            return Err(Error::InvalidInput(format!(
                "ASR request {} already owns an encoder artifact",
                self.id
            )));
        }
        self.prepared_asr_encoder_artifact = Some(PreparedAsrEncoderArtifact {
            model_variant,
            artifact: PreparedAsrEncoderArtifactValue::VibeVoice(artifact),
            source_fingerprint,
        });
        Ok(())
    }

    pub(crate) fn prepared_vibevoice_artifact_for_executor(
        &self,
    ) -> Result<Option<Arc<VibeVoiceAsrPreparedArtifact>>> {
        let Some(prepared) = self.prepared_asr_encoder_artifact.as_ref() else {
            return Ok(None);
        };
        if self.model_variant != Some(prepared.model_variant)
            || self.asr_execution_source_fingerprint(prepared.model_variant)?
                != prepared.source_fingerprint
            || self.uses_asr_long_form_atomic()
        {
            return Err(Error::InvalidInput(format!(
                "ASR request {} changed after VibeVoice encoder preparation",
                self.id
            )));
        }
        match &prepared.artifact {
            PreparedAsrEncoderArtifactValue::VibeVoice(value) => Ok(Some(value.clone())),
            PreparedAsrEncoderArtifactValue::Qwen3(_)
            | PreparedAsrEncoderArtifactValue::Whisper(_)
            | PreparedAsrEncoderArtifactValue::GraniteSpeech(_)
            | PreparedAsrEncoderArtifactValue::Lfm25Audio(_)
            | PreparedAsrEncoderArtifactValue::Parakeet(_) => Err(Error::InvalidInput(
                "foreign encoder artifact was requested through the VibeVoice accessor".into(),
            )),
        }
    }

    pub(crate) fn install_prepared_granite_speech_artifact(
        &mut self,
        model_variant: ModelVariant,
        artifact: Arc<GraniteSpeechPreparedPromptArtifact>,
    ) -> Result<()> {
        if model_variant.family() != ModelFamily::GraniteSpeechAsr
            || self.task_type != TaskType::ASR
            || self.model_variant != Some(model_variant)
            || self.uses_asr_long_form_atomic()
        {
            return Err(Error::InvalidInput(format!(
                "ASR request {} Granite Speech artifact does not match its normal routed model",
                self.id
            )));
        }
        let source_fingerprint = self.refresh_asr_execution_source_fingerprint(model_variant)?;
        if self.prepared_asr_encoder_artifact.is_some() {
            return Err(Error::InvalidInput(format!(
                "ASR request {} already owns an encoder artifact",
                self.id
            )));
        }
        self.prepared_asr_encoder_artifact = Some(PreparedAsrEncoderArtifact {
            model_variant,
            artifact: PreparedAsrEncoderArtifactValue::GraniteSpeech(artifact),
            source_fingerprint,
        });
        Ok(())
    }

    pub(crate) fn prepared_granite_speech_artifact_for_executor(
        &self,
    ) -> Result<Option<Arc<GraniteSpeechPreparedPromptArtifact>>> {
        let Some(prepared) = self.prepared_asr_encoder_artifact.as_ref() else {
            return Ok(None);
        };
        if self.model_variant != Some(prepared.model_variant)
            || self.asr_execution_source_fingerprint(prepared.model_variant)?
                != prepared.source_fingerprint
            || self.uses_asr_long_form_atomic()
        {
            return Err(Error::InvalidInput(format!(
                "ASR request {} changed after Granite Speech preparation",
                self.id
            )));
        }
        match &prepared.artifact {
            PreparedAsrEncoderArtifactValue::GraniteSpeech(value) => Ok(Some(value.clone())),
            PreparedAsrEncoderArtifactValue::Qwen3(_)
            | PreparedAsrEncoderArtifactValue::Whisper(_)
            | PreparedAsrEncoderArtifactValue::VibeVoice(_)
            | PreparedAsrEncoderArtifactValue::Lfm25Audio(_)
            | PreparedAsrEncoderArtifactValue::Parakeet(_) => Err(Error::InvalidInput(
                "foreign encoder artifact was requested through the Granite Speech accessor".into(),
            )),
        }
    }

    pub(crate) fn install_prepared_lfm25_audio_asr_artifact(
        &mut self,
        model_variant: ModelVariant,
        artifact: Arc<Lfm25AudioPreparedAsrArtifact>,
    ) -> Result<()> {
        if model_variant.family() != ModelFamily::Lfm25Audio
            || self.task_type != TaskType::ASR
            || self.model_variant != Some(model_variant)
            || self.uses_asr_long_form_atomic()
        {
            return Err(Error::InvalidInput(format!(
                "ASR request {} LFM2.5 Audio artifact does not match its normal routed model",
                self.id
            )));
        }
        let source_fingerprint = self.refresh_asr_execution_source_fingerprint(model_variant)?;
        if self.prepared_asr_encoder_artifact.is_some() {
            return Err(Error::InvalidInput(format!(
                "ASR request {} already owns an encoder artifact",
                self.id
            )));
        }
        let prepared_costs = self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::Lfm25AudioAsr(model) => Some(model),
                _ => None,
            })
            .map(|model| {
                Self::lfm25_audio_asr_stage_costs(
                    self.execution_adapter_binding.as_ref(),
                    model,
                    artifact.prompt_tokens,
                    self.params.max_tokens,
                )
            })
            .transpose()?
            .unwrap_or_default();
        self.prepared_asr_encoder_artifact = Some(PreparedAsrEncoderArtifact {
            model_variant,
            artifact: PreparedAsrEncoderArtifactValue::Lfm25Audio(artifact),
            source_fingerprint,
        });
        for (stage_id, cost) in prepared_costs {
            self.install_prepared_stage_cost(stage_id, cost)?;
        }
        Ok(())
    }

    pub(crate) fn prepared_lfm25_audio_asr_artifact_for_executor(
        &self,
    ) -> Result<Option<Arc<Lfm25AudioPreparedAsrArtifact>>> {
        let Some(prepared) = self.prepared_asr_encoder_artifact.as_ref() else {
            return Ok(None);
        };
        if self.model_variant != Some(prepared.model_variant)
            || self.asr_execution_source_fingerprint(prepared.model_variant)?
                != prepared.source_fingerprint
            || self.uses_asr_long_form_atomic()
        {
            return Err(Error::InvalidInput(format!(
                "ASR request {} changed after LFM2.5 Audio preparation",
                self.id
            )));
        }
        match &prepared.artifact {
            PreparedAsrEncoderArtifactValue::Lfm25Audio(value) => Ok(Some(value.clone())),
            PreparedAsrEncoderArtifactValue::Qwen3(_)
            | PreparedAsrEncoderArtifactValue::Whisper(_)
            | PreparedAsrEncoderArtifactValue::VibeVoice(_)
            | PreparedAsrEncoderArtifactValue::GraniteSpeech(_)
            | PreparedAsrEncoderArtifactValue::Parakeet(_) => Err(Error::InvalidInput(
                "foreign encoder artifact was requested through the LFM2.5 Audio accessor".into(),
            )),
        }
    }

    pub(crate) fn install_prepared_parakeet_artifact(
        &mut self,
        model_variant: ModelVariant,
        artifact: Arc<ParakeetPreparedEncoderArtifact>,
    ) -> Result<()> {
        if model_variant.family() != ModelFamily::ParakeetAsr
            || self.task_type != TaskType::ASR
            || self.model_variant != Some(model_variant)
            || self.uses_asr_long_form_atomic()
        {
            return Err(Error::InvalidInput(format!(
                "ASR request {} Parakeet artifact does not match its normal routed model",
                self.id
            )));
        }
        let source_fingerprint = self.refresh_asr_execution_source_fingerprint(model_variant)?;
        if self.prepared_asr_encoder_artifact.is_some() {
            return Err(Error::InvalidInput(format!(
                "ASR request {} already owns an encoder artifact",
                self.id
            )));
        }
        self.prepared_asr_encoder_artifact = Some(PreparedAsrEncoderArtifact {
            model_variant,
            artifact: PreparedAsrEncoderArtifactValue::Parakeet(artifact),
            source_fingerprint,
        });
        Ok(())
    }

    pub(crate) fn prepared_parakeet_artifact_for_executor(
        &self,
    ) -> Result<Option<Arc<ParakeetPreparedEncoderArtifact>>> {
        let Some(prepared) = self.prepared_asr_encoder_artifact.as_ref() else {
            return Ok(None);
        };
        if self.model_variant != Some(prepared.model_variant)
            || self.asr_execution_source_fingerprint(prepared.model_variant)?
                != prepared.source_fingerprint
            || self.uses_asr_long_form_atomic()
        {
            return Err(Error::InvalidInput(format!(
                "ASR request {} changed after Parakeet preparation",
                self.id
            )));
        }
        match &prepared.artifact {
            PreparedAsrEncoderArtifactValue::Parakeet(value) => Ok(Some(value.clone())),
            _ => Err(Error::InvalidInput(
                "foreign encoder artifact was requested through the Parakeet accessor".into(),
            )),
        }
    }

    #[cfg(test)]
    pub(crate) fn prepared_asr_encoder_artifact_retained_bytes(&self) -> Result<u64> {
        self.prepared_asr_encoder_artifact
            .as_ref()
            .map_or(Ok(0), |prepared| {
                prepared
                    .artifact
                    .resident_resources()
                    .map(|(_, device)| device)
            })
    }

    pub(crate) fn prepared_asr_encoder_artifact_retained_resources(&self) -> Result<(u64, u64)> {
        self.prepared_asr_encoder_artifact
            .as_ref()
            .map_or(Ok((0, 0)), |prepared| {
                prepared.artifact.resident_resources()
            })
    }

    /// Route the request to the long-form atomic executor, which runs `plan`
    /// as planned here.
    pub(crate) fn install_prepared_asr_long_form_atomic(
        &mut self,
        plan: PlannedAsrChunks,
    ) -> Result<()> {
        if self.task_type != TaskType::ASR {
            return Err(Error::InvalidInput(format!(
                "Request {} cannot install an ASR long-form route for {:?}",
                self.id, self.task_type
            )));
        }
        if self.prepared_sequence_input_tokens.is_some()
            || matches!(
                self.prepared_asr_execution_shape,
                Some(PreparedAsrExecutionShape::RetainedSequence { .. })
            )
        {
            return Err(Error::InvalidInput(format!(
                "ASR request {} cannot change its prepared retained sequence route to long-form execution",
                self.id
            )));
        }
        self.prepared_asr_execution_shape = Some(PreparedAsrExecutionShape::LongFormAtomic);
        self.prepared_asr_route_plan = Some(Arc::new(plan));
        Ok(())
    }

    /// The segmentation preparation planned for a long-form row.
    pub(crate) fn prepared_asr_route_plan(&self) -> Option<Arc<PlannedAsrChunks>> {
        self.prepared_asr_route_plan.clone()
    }

    pub(crate) fn prepared_asr_execution_shape(&self) -> Option<PreparedAsrExecutionShape> {
        self.prepared_asr_execution_shape
    }

    pub(crate) fn uses_asr_retained_sequence(&self) -> bool {
        matches!(
            self.prepared_asr_execution_shape,
            Some(PreparedAsrExecutionShape::RetainedSequence { .. })
        )
    }

    pub(crate) fn uses_asr_long_form_atomic(&self) -> bool {
        self.prepared_asr_execution_shape == Some(PreparedAsrExecutionShape::LongFormAtomic)
    }

    pub(crate) fn prepared_asr_model_for_executor(&self) -> Result<Option<Arc<NativeAsrModel>>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::Asr(model) => Some(model.model_arc()),
                PreparedIncrementalModel::Lfm25AudioAsr(_)
                | PreparedIncrementalModel::Lfm25AudioTts(_)
                | PreparedIncrementalModel::QwenTts(_) => None,
                PreparedIncrementalModel::VibeVoiceTts(_)
                | PreparedIncrementalModel::FishS2Tts(_)
                | PreparedIncrementalModel::VoxtralTts(_)
                | PreparedIncrementalModel::ChatterboxTts(_)
                | PreparedIncrementalModel::KokoroTts(_) => None,
            }))
    }

    pub(crate) fn prepared_asr_model_lease_for_executor(&self) -> Result<Option<AsrModelLease>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::Asr(model) => Some(model.clone()),
                PreparedIncrementalModel::Lfm25AudioAsr(_)
                | PreparedIncrementalModel::Lfm25AudioTts(_)
                | PreparedIncrementalModel::QwenTts(_) => None,
                PreparedIncrementalModel::VibeVoiceTts(_)
                | PreparedIncrementalModel::FishS2Tts(_)
                | PreparedIncrementalModel::VoxtralTts(_)
                | PreparedIncrementalModel::ChatterboxTts(_)
                | PreparedIncrementalModel::KokoroTts(_) => None,
            }))
    }

    pub(crate) fn prepared_lfm25_audio_asr_model_lease_for_executor(
        &self,
    ) -> Result<Option<Lfm25AudioModelLease>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::Lfm25AudioAsr(model) => Some(model.clone()),
                PreparedIncrementalModel::Asr(_)
                | PreparedIncrementalModel::Lfm25AudioTts(_)
                | PreparedIncrementalModel::QwenTts(_) => None,
                PreparedIncrementalModel::VibeVoiceTts(_)
                | PreparedIncrementalModel::FishS2Tts(_)
                | PreparedIncrementalModel::VoxtralTts(_)
                | PreparedIncrementalModel::ChatterboxTts(_)
                | PreparedIncrementalModel::KokoroTts(_) => None,
            }))
    }
}
