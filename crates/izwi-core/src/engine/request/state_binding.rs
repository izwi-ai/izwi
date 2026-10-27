//! Binding a request to its execution adapter and state: projected state
//! inputs, prepared stage costs and the managed cache runtime.

use super::*;

impl EngineCoreRequest {
    pub(crate) fn validate_execution_preparation(&self) -> Result<()> {
        let chat_result = if self.task_type == TaskType::Chat {
            self.validate_chat_execution_preparation()
        } else if self.chat_execution_ready.is_some() {
            Err(Error::InvalidInput(format!(
                "Non-chat request {} carries chat execution preparation",
                self.id
            )))
        } else {
            Ok(())
        };
        chat_result?;
        if self.task_type == TaskType::Chat && self.incremental_model_execution_ready.is_some() {
            return Err(Error::InvalidInput(format!(
                "Chat request {} carries non-chat incremental model preparation",
                self.id
            )));
        }
        self.validate_incremental_model_execution_preparation()?;
        self.validate_prepared_stage_costs()
    }

    /// Validate once at the mutable core admission boundary. Requests are
    /// immutable after insertion, so executor dispatch can then check this
    /// sealed state without re-hashing a long prompt on every decode quantum.
    pub(crate) fn seal_execution_preparation(&mut self) -> Result<()> {
        self.validate_execution_preparation()?;
        if let Some(ready) = self.chat_execution_ready.as_mut() {
            ready.core_validated = true;
        }
        Ok(())
    }

    pub(super) fn native_prepared_stage(
        binding: Option<&super::super::ExecutionAdapterBinding>,
        selector: super::super::StageWorkSelector,
    ) -> Option<&super::super::StageDescriptor> {
        binding.and_then(|binding| {
            binding.stages.iter().find(|stage| {
                stage.selector == selector && stage.batch_mode != super::super::NativeBatchMode::None
            })
        })
    }

    /// Derive independently clocked retained-state work from immutable
    /// preparation metadata. This method never chooses a model or a stage:
    /// the already-bound stage authorizes exact group/clock pairs, while Core
    /// authenticates the returned spans against the physical state plan.
    pub(crate) fn project_auxiliary_state_spans(
        &self,
        stage: &StageDescriptor,
        input: InputRange,
    ) -> Result<Option<Arc<[ClockedStateSpan]>>> {
        let Some(authorized) = stage.retained_state_selections.as_deref() else {
            return Ok(None);
        };
        if self
            .incremental_model_execution_ready
            .as_ref()
            .is_some_and(|ready| {
                matches!(ready.model, PreparedIncrementalModel::VibeVoiceTts(_))
                    && ready.vibevoice_tts.is_some()
            })
        {
            if authorized.is_empty() {
                return Ok(Some(Arc::from([])));
            }
            let selection = super::super::ClockedStateSelection::new(
                crate::models::architectures::vibevoice::VIBEVOICE_TTS_TOKENIZER_GROUP,
                crate::kv::v2::StateClock::CodecFrames,
            )?;
            if authorized != [selection.clone()] {
                return Err(Error::InvalidInput(format!(
                    "stage {} does not authorize the VibeVoice TTS tokenizer clock",
                    stage.name
                )));
            }
            let prompt_tokens = self.prepared_sequence_input_tokens.ok_or_else(|| {
                Error::InferenceError(
                    "prepared VibeVoice TTS request lost its prompt boundary".into(),
                )
            })?;
            let start = input.start.checked_sub(prompt_tokens).ok_or_else(|| {
                Error::InvalidInput(
                    "VibeVoice TTS tokenizer state cannot advance during prompt prefill".into(),
                )
            })?;
            let end = input.end.checked_sub(prompt_tokens).ok_or_else(|| {
                Error::InvalidInput(
                    "VibeVoice TTS tokenizer state crossed its prompt boundary".into(),
                )
            })?;
            return Ok(Some(Arc::from([ClockedStateSpan::new(
                selection.group(),
                selection.clock().clone(),
                InputRange::new(start, end)?,
            )?])));
        }
        let projections = if let Some(prepared) = self.prepared_asr_encoder_artifact.as_ref() {
            if self.model_variant != Some(prepared.model_variant)
                || self.asr_execution_source_fingerprint(prepared.model_variant)?
                    != prepared.source_fingerprint
                || self.uses_asr_long_form_atomic()
            {
                return Err(Error::InvalidInput(format!(
                    "ASR request {} changed after auxiliary state projection preparation",
                    self.id
                )));
            }
            prepared.artifact.clocked_state_projections()
        } else {
            &[]
        };

        let mut previous = None;
        let mut spans = Vec::new();
        for projection in projections {
            let group = projection.selection().group();
            let primary = projection.primary();
            if let Some((previous_group, previous_clock, previous_end)) = previous.as_ref() {
                if group.get() < *previous_group
                    || (group.get() == *previous_group
                        && (projection.selection().clock() != previous_clock
                            || primary.start < *previous_end))
                {
                    return Err(Error::InvalidInput(format!(
                        "request {} has non-canonical clocked state projections",
                        self.id
                    )));
                }
            }
            previous = Some((
                group.get(),
                projection.selection().clock().clone(),
                primary.end,
            ));

            let Some(span) = projection.project(input)? else {
                continue;
            };
            if !authorized
                .iter()
                .any(|selection| selection == projection.selection())
            {
                return Err(Error::InvalidInput(format!(
                    "stage {} does not authorize request {} auxiliary group {} {:?}",
                    stage.name,
                    self.id,
                    group.get(),
                    projection.selection().clock()
                )));
            }
            spans.push(span);
        }

        Ok(Some(Arc::from(spans)))
    }

    pub(crate) fn project_paged_state_input(
        &self,
        domain: crate::kv::CacheDomainId,
        phase: super::super::SequencePhase,
        input: InputRange,
    ) -> Result<InputRange> {
        // An unguided Chatterbox request never runs its unconditional
        // sequence; an empty span leaves that cache unreserved.
        if domain == crate::models::architectures::chatterbox::CHATTERBOX_UNCOND_DOMAIN
            && self
                .incremental_model_execution_ready
                .as_ref()
                .and_then(|ready| ready.chatterbox_tts.as_ref())
                .is_some_and(|artifact| artifact.unconditional.is_none())
        {
            return InputRange::new(0, 0);
        }
        let is_vibevoice_tts =
            self.incremental_model_execution_ready
                .as_ref()
                .is_some_and(|ready| {
                    matches!(ready.model, PreparedIncrementalModel::VibeVoiceTts(_))
                        && ready.vibevoice_tts.is_some()
                });
        if !is_vibevoice_tts
            || domain != crate::models::architectures::vibevoice::VIBEVOICE_TTS_NEGATIVE_DOMAIN
        {
            return Ok(input);
        }
        match phase {
            super::super::SequencePhase::Prefill if input.start == 0 => InputRange::new(0, 1),
            super::super::SequencePhase::Prefill => Err(Error::InferenceError(
                "VibeVoice TTS requires its exact prompt prefill in one quantum".into(),
            )),
            super::super::SequencePhase::Decode => {
                let prompt_tokens = self.prepared_sequence_input_tokens.ok_or_else(|| {
                    Error::InferenceError(
                        "prepared VibeVoice TTS request lost its prompt boundary".into(),
                    )
                })?;
                let start = input
                    .start
                    .checked_sub(prompt_tokens)
                    .and_then(|frames| frames.checked_add(1))
                    .ok_or_else(|| {
                        Error::InvalidInput(
                            "VibeVoice TTS negative cache start crossed its prompt boundary".into(),
                        )
                    })?;
                let end = input
                    .end
                    .checked_sub(prompt_tokens)
                    .and_then(|frames| frames.checked_add(1))
                    .ok_or_else(|| {
                        Error::InvalidInput(
                            "VibeVoice TTS negative cache end crossed its prompt boundary".into(),
                        )
                    })?;
                InputRange::new(start, end)
            }
        }
    }

    pub(crate) fn install_prepared_sequence_input_tokens(
        &mut self,
        input_tokens: usize,
        max_sequence_tokens: usize,
    ) -> Result<()> {
        if self.task_type != TaskType::ASR {
            return Err(Error::InvalidInput(format!(
                "Request {} cannot install multimodal sequence shape for {:?}",
                self.id, self.task_type
            )));
        }
        if self.prepared_asr_execution_shape == Some(PreparedAsrExecutionShape::LongFormAtomic) {
            return Err(Error::InvalidInput(format!(
                "ASR request {} cannot change its prepared long-form route to retained sequence execution",
                self.id
            )));
        }
        if input_tokens == 0 {
            return Err(Error::InvalidInput(format!(
                "ASR request {} has an empty exact prefill",
                self.id
            )));
        }
        if input_tokens >= max_sequence_tokens {
            // The load fitted a context too small for this request's audio:
            // a capacity refusal (503); longer audio is chunked upstream.
            return Err(Error::Overloaded(format!(
                "ASR request {} exact prefill has {input_tokens} tokens and leaves no output capacity in the configured {max_sequence_tokens}-token context",
                self.id
            )));
        }
        if self
            .prepared_sequence_input_tokens
            .is_some_and(|current| current != input_tokens)
        {
            return Err(Error::InvalidInput(format!(
                "ASR request {} changed its prepared multimodal sequence shape",
                self.id
            )));
        }
        self.params.max_tokens = self
            .params
            .max_tokens
            .max(1)
            .min(max_sequence_tokens - input_tokens);
        self.prepared_sequence_input_tokens = Some(input_tokens);
        self.prepared_asr_execution_shape =
            Some(PreparedAsrExecutionShape::RetainedSequence { input_tokens });
        Ok(())
    }

    pub(super) fn validate_incremental_model_execution_preparation(&self) -> Result<()> {
        let Some(ready) = self.incremental_model_execution_ready.as_ref() else {
            if self.prepared_sequence_input_tokens.is_some()
                || self.prepared_asr_execution_shape.is_some()
                || self.prepared_asr_audio.is_some()
                || self.prepared_asr_encoder_artifact.is_some()
            {
                return Err(Error::InvalidInput(format!(
                    "Request {} carries prepared multimodal input without an exact loaded model",
                    self.id
                )));
            }
            return Ok(());
        };
        if self.model_variant != Some(ready.model_variant) {
            return Err(Error::InvalidInput(format!(
                "Request {} changed model after incremental model preparation",
                self.id
            )));
        }
        if matches!(&ready.model, PreparedIncrementalModel::Asr(_))
            && matches!(
                ready.model_variant.family(),
                ModelFamily::Qwen3Asr
                    | ModelFamily::WhisperAsr
                    | ModelFamily::VibeVoiceAsr
                    | ModelFamily::GraniteSpeechAsr
                    | ModelFamily::ParakeetAsr
            )
        {
            if self.prepared_asr_audio_for_executor()?.is_none() {
                return Err(Error::InvalidInput(format!(
                    "Qwen3 ASR request {} is missing its exact decoded-audio artifact",
                    self.id
                )));
            }
            match self.prepared_asr_execution_shape {
                Some(PreparedAsrExecutionShape::RetainedSequence { input_tokens })
                    if self.prepared_sequence_input_tokens == Some(input_tokens)
                        && match ready.model_variant.family() {
                            ModelFamily::Qwen3Asr => {
                                self.prepared_asr_encoder_artifact_for_executor()?.is_some()
                            }
                            ModelFamily::WhisperAsr => {
                                self.prepared_whisper_window_for_executor()?.is_some()
                            }
                            ModelFamily::VibeVoiceAsr => {
                                self.prepared_vibevoice_artifact_for_executor()?.is_some()
                            }
                            ModelFamily::GraniteSpeechAsr => self
                                .prepared_granite_speech_artifact_for_executor()?
                                .is_some(),
                            ModelFamily::ParakeetAsr => {
                                self.prepared_parakeet_artifact_for_executor()?.is_some()
                            }
                            _ => false,
                        } => {}
                Some(PreparedAsrExecutionShape::LongFormAtomic)
                    if self.prepared_sequence_input_tokens.is_none()
                        && self.prepared_asr_encoder_artifact.is_none() => {}
                _ => {
                    return Err(Error::InvalidInput(format!(
                        "ASR request {} is missing a consistent exact media execution shape",
                        self.id
                    )));
                }
            }
        }
        if matches!(&ready.model, PreparedIncrementalModel::Lfm25AudioAsr(_)) {
            if ready.model_variant.family() != ModelFamily::Lfm25Audio
                || self.prepared_asr_audio_for_executor()?.is_none()
            {
                return Err(Error::InvalidInput(format!(
                    "LFM2.5 Audio ASR request {} is missing its exact decoded-audio artifact",
                    self.id
                )));
            }
            match self.prepared_asr_execution_shape {
                Some(PreparedAsrExecutionShape::RetainedSequence { input_tokens })
                    if self.prepared_sequence_input_tokens == Some(input_tokens)
                        && self
                            .prepared_lfm25_audio_asr_artifact_for_executor()?
                            .is_some() => {}
                Some(PreparedAsrExecutionShape::LongFormAtomic)
                    if self.prepared_sequence_input_tokens.is_none()
                        && self.prepared_asr_encoder_artifact.is_none() => {}
                _ => {
                    return Err(Error::InvalidInput(format!(
                        "LFM2.5 Audio ASR request {} is missing a consistent exact media execution shape",
                        self.id
                    )));
                }
            }
        }
        if matches!(&ready.model, PreparedIncrementalModel::Lfm25AudioTts(_)) {
            let expected = self.lfm25_audio_tts_messages_for_preparation()?;
            let prepared = ready.lfm25_audio_tts.as_ref().ok_or_else(|| {
                Error::InvalidInput(format!(
                    "LFM2.5 Audio TTS request {} is missing its prepared prompt",
                    self.id
                ))
            })?;
            if ready.model_variant.family() != ModelFamily::Lfm25Audio
                || self.prepared_sequence_input_tokens != Some(prepared.prompt_tokens)
                || prepared.source_messages.len() != expected.len()
                || prepared
                    .source_messages
                    .iter()
                    .zip(&expected)
                    .any(|(actual, expected)| {
                        actual.role != expected.role || actual.content != expected.content
                    })
            {
                return Err(Error::InvalidInput(format!(
                    "LFM2.5 Audio TTS request {} changed after preparation",
                    self.id
                )));
            }
        }
        if matches!(&ready.model, PreparedIncrementalModel::VibeVoiceTts(_)) {
            let prepared = ready.vibevoice_tts.as_ref().ok_or_else(|| {
                Error::InvalidInput(format!(
                    "VibeVoice TTS request {} is missing its prepared prompt",
                    self.id
                ))
            })?;
            let params = ready.vibevoice_tts_params.as_ref().ok_or_else(|| {
                Error::InvalidInput(format!(
                    "VibeVoice TTS request {} is missing its sealed generation geometry",
                    self.id
                ))
            })?;
            if ready.model_variant.family() != ModelFamily::VibeVoiceTts
                || self.prepared_sequence_input_tokens != Some(prepared.prompt_tokens())
                || self.params.max_tokens != params.max_frames
            {
                return Err(Error::InvalidInput(format!(
                    "VibeVoice TTS request {} changed after preparation",
                    self.id
                )));
            }
        }
        if matches!(&ready.model, PreparedIncrementalModel::FishS2Tts(_)) {
            let prepared = ready.fish_s2_tts.as_ref().ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Fish S2 TTS request {} is missing its prepared prompt",
                    self.id
                ))
            })?;
            let params = ready.fish_s2_tts_params.as_ref().ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Fish S2 TTS request {} is missing its sealed generation geometry",
                    self.id
                ))
            })?;
            if ready.model_variant.family() != ModelFamily::FishS2Tts
                || self.prepared_sequence_input_tokens != Some(prepared.prompt_tokens())
                || self.params.max_tokens != params.max_frames
            {
                return Err(Error::InvalidInput(format!(
                    "Fish S2 TTS request {} changed after preparation",
                    self.id
                )));
            }
        }
        if matches!(&ready.model, PreparedIncrementalModel::VoxtralTts(_)) {
            let prepared = ready.voxtral_tts.as_ref().ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Voxtral TTS request {} is missing its prepared prompt",
                    self.id
                ))
            })?;
            let params = ready.voxtral_tts_params.as_ref().ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Voxtral TTS request {} is missing its sealed generation geometry",
                    self.id
                ))
            })?;
            if ready.model_variant.family() != ModelFamily::VoxtralTts
                || self.prepared_sequence_input_tokens != Some(prepared.prompt_tokens)
                || self.params.max_tokens != params.max_frames
            {
                return Err(Error::InvalidInput(format!(
                    "Voxtral TTS request {} changed after preparation",
                    self.id
                )));
            }
        }
        if matches!(&ready.model, PreparedIncrementalModel::ChatterboxTts(_)) {
            let prepared = ready.chatterbox_tts.as_ref().ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Chatterbox request {} is missing its prepared prompt",
                    self.id
                ))
            })?;
            let params = ready.chatterbox_tts_params.as_ref().ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Chatterbox request {} is missing its sealed generation parameters",
                    self.id
                ))
            })?;
            if ready.model_variant.family() != ModelFamily::ChatterboxTts
                || self.prepared_sequence_input_tokens != Some(prepared.prompt_tokens)
                || self.params.max_tokens != params.max_speech_tokens
            {
                return Err(Error::InvalidInput(format!(
                    "Chatterbox request {} changed after preparation",
                    self.id
                )));
            }
        }
        if matches!(&ready.model, PreparedIncrementalModel::Maya1Tts(_)) {
            let prepared = ready.maya1_tts.as_ref().ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Maya1 request {} is missing its prepared prompt",
                    self.id
                ))
            })?;
            let params = ready.maya1_tts_params.as_ref().ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Maya1 request {} is missing its sealed generation parameters",
                    self.id
                ))
            })?;
            if ready.model_variant.family() != ModelFamily::Maya1Tts
                || self.prepared_sequence_input_tokens != Some(prepared.prompt.len())
                || self.params.max_tokens != params.max_new_tokens
            {
                return Err(Error::InvalidInput(format!(
                    "Maya1 request {} changed after preparation",
                    self.id
                )));
            }
        }
        if matches!(&ready.model, PreparedIncrementalModel::KokoroTts(_)) {
            let prepared = ready.kokoro_tts.as_ref().ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Kokoro TTS request {} is missing its prepared input",
                    self.id
                ))
            })?;
            let text = self.text.as_deref().unwrap_or_default();
            let speaker = self
                .params
                .speaker
                .as_deref()
                .or(self.params.voice.as_deref());
            let input_tokens = prepared.input_tokens()?;
            if ready.model_variant.family() != ModelFamily::KokoroTts
                || !prepared.matches_request(
                    text,
                    speaker,
                    self.language.as_deref(),
                    self.params.speed,
                )
                || !prepared.is_well_formed()
                || self.prepared_sequence_input_tokens != Some(input_tokens)
            {
                return Err(Error::InvalidInput(format!(
                    "Kokoro TTS request {} changed after preparation",
                    self.id
                )));
            }
        }
        match (&ready.model, self.task_type) {
            (PreparedIncrementalModel::Asr(_), TaskType::ASR) if ready.qwen_tts.is_none() => Ok(()),
            (PreparedIncrementalModel::Lfm25AudioAsr(_), TaskType::ASR)
                if ready.qwen_tts.is_none() =>
            {
                Ok(())
            }
            (PreparedIncrementalModel::Lfm25AudioTts(_), TaskType::TTS)
                if ready.qwen_tts.is_none() && ready.lfm25_audio_tts.is_some() =>
            {
                Ok(())
            }
            (PreparedIncrementalModel::QwenTts(_), TaskType::TTS)
                if ready.model_variant.family() == ModelFamily::Qwen3Tts
                    && ready.qwen_tts.as_ref().is_some_and(|prepared| {
                        self.prepared_sequence_input_tokens == Some(prepared.prefill_tokens)
                            && self.params.max_tokens == prepared.max_frames
                    }) =>
            {
                Ok(())
            }
            (PreparedIncrementalModel::VibeVoiceTts(_), TaskType::TTS)
                if ready.model_variant.family() == ModelFamily::VibeVoiceTts
                    && ready.vibevoice_tts.is_some()
                    && ready.vibevoice_tts_params.is_some() =>
            {
                Ok(())
            }
            (PreparedIncrementalModel::FishS2Tts(_), TaskType::TTS)
                if ready.model_variant.family() == ModelFamily::FishS2Tts
                    && ready.fish_s2_tts.is_some()
                    && ready.fish_s2_tts_params.is_some() =>
            {
                Ok(())
            }
            (PreparedIncrementalModel::VoxtralTts(_), TaskType::TTS)
                if ready.model_variant.family() == ModelFamily::VoxtralTts
                    && ready.voxtral_tts.is_some()
                    && ready.voxtral_tts_params.is_some() =>
            {
                Ok(())
            }
            (PreparedIncrementalModel::ChatterboxTts(_), TaskType::TTS)
                if ready.model_variant.family() == ModelFamily::ChatterboxTts
                    && ready.chatterbox_tts.is_some()
                    && ready.chatterbox_tts_params.is_some() =>
            {
                Ok(())
            }
            (PreparedIncrementalModel::Maya1Tts(_), TaskType::TTS)
                if ready.model_variant.family() == ModelFamily::Maya1Tts
                    && ready.maya1_tts.is_some()
                    && ready.maya1_tts_params.is_some() =>
            {
                Ok(())
            }
            (PreparedIncrementalModel::KokoroTts(_), TaskType::TTS)
                if ready.model_variant.family() == ModelFamily::KokoroTts
                    && ready.kokoro_tts.is_some() =>
            {
                Ok(())
            }
            _ => Err(Error::InvalidInput(format!(
                "Request {} carries incremental preparation for a different task",
                self.id
            ))),
        }
    }

    pub(crate) fn bind_model_instance(
        &mut self,
        model_instance_id: super::super::ModelInstanceId,
    ) -> Result<()> {
        if self
            .model_instance_id
            .is_some_and(|current| current != model_instance_id)
        {
            return Err(Error::InvalidInput(
                "engine request is already bound to a different model instance".to_string(),
            ));
        }
        self.model_instance_id = Some(model_instance_id);
        Ok(())
    }

    pub(crate) fn model_instance_id(&self) -> Option<super::super::ModelInstanceId> {
        self.model_instance_id
    }

    pub(crate) fn bind_execution_adapter(
        &mut self,
        binding: super::super::ExecutionAdapterBinding,
    ) -> Result<()> {
        binding.validate()?;
        if self.model_variant != Some(binding.model_variant) {
            return Err(Error::InvalidInput(
                "engine request model does not match its execution adapter".to_string(),
            ));
        }
        self.bind_model_instance(binding.model_instance_id)?;
        if self
            .execution_adapter_binding
            .as_ref()
            .is_some_and(|current| current != &binding)
        {
            return Err(Error::InvalidInput(
                "engine request is already bound to a different execution adapter".to_string(),
            ));
        }
        let prepared_continuous_cost = self
            .chat_execution_ready
            .as_ref()
            .map(|ready| Self::continuous_chat_stage_cost(Some(&binding), &ready.model))
            .transpose()?
            .flatten();
        let prepared_asr_continuous_cost = self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::Asr(model) => Some(model),
                PreparedIncrementalModel::Lfm25AudioAsr(_)
                | PreparedIncrementalModel::Lfm25AudioTts(_)
                | PreparedIncrementalModel::QwenTts(_) => None,
                PreparedIncrementalModel::VibeVoiceTts(_)
                | PreparedIncrementalModel::FishS2Tts(_)
                | PreparedIncrementalModel::VoxtralTts(_)
                | PreparedIncrementalModel::ChatterboxTts(_)
                | PreparedIncrementalModel::Maya1Tts(_)
                | PreparedIncrementalModel::KokoroTts(_) => None,
            })
            .filter(|_| self.uses_asr_retained_sequence())
            .map(|model| Self::continuous_asr_stage_cost(Some(&binding), model))
            .transpose()?
            .flatten();
        let prepared_lfm25_asr_costs = self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::Lfm25AudioAsr(model) => Some(model),
                _ => None,
            })
            .zip(self.prepared_asr_encoder_artifact.as_ref())
            .and_then(|(model, prepared)| match &prepared.artifact {
                PreparedAsrEncoderArtifactValue::Lfm25Audio(artifact) => {
                    Some((model, artifact.prompt_tokens))
                }
                _ => None,
            })
            .filter(|_| self.uses_asr_retained_sequence())
            .map(|(model, prompt_tokens)| {
                Self::lfm25_audio_asr_stage_costs(
                    Some(&binding),
                    model,
                    prompt_tokens,
                    self.params.max_tokens,
                )
            })
            .transpose()?
            .unwrap_or_default();
        let prepared_lfm25_tts_costs = self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(
                |ready| match (&ready.model, ready.lfm25_audio_tts.as_ref()) {
                    (PreparedIncrementalModel::Lfm25AudioTts(model), Some(artifact)) => {
                        Some((model, artifact.prompt_tokens))
                    }
                    _ => None,
                },
            )
            .map(|(model, prompt_tokens)| {
                Self::lfm25_audio_tts_stage_costs(
                    Some(&binding),
                    model,
                    prompt_tokens,
                    self.params.max_tokens,
                )
            })
            .transpose()?
            .unwrap_or_default();
        let prepared_tts_continuous_cost = self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match (&ready.model, ready.qwen_tts.as_ref()) {
                (PreparedIncrementalModel::QwenTts(model), Some(prepared)) => Some((
                    model.model_arc(),
                    prepared.prefill_tokens,
                    prepared.max_frames,
                )),
                (
                    PreparedIncrementalModel::Asr(_) | PreparedIncrementalModel::Lfm25AudioAsr(_),
                    _,
                ) => None,
                _ => None,
            })
            .map(|(model, prefill_tokens, max_frames)| {
                Self::continuous_tts_stage_cost(Some(&binding), &model, prefill_tokens, max_frames)
            })
            .transpose()?
            .flatten();
        let prepared_tts_prefill_cost = self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match (&ready.model, ready.qwen_tts.as_ref()) {
                (PreparedIncrementalModel::QwenTts(model), Some(prepared)) => Some((
                    model.model_arc(),
                    prepared.prefill_tokens,
                    prepared.max_frames,
                    prepared.reference.is_some(),
                )),
                _ => None,
            })
            .map(|(model, prefill_tokens, max_frames, has_reference)| {
                Self::resumable_tts_prefill_stage_cost(
                    Some(&binding),
                    &model,
                    prefill_tokens,
                    max_frames,
                    has_reference,
                )
            })
            .transpose()?
            .flatten();
        let prepared_kokoro_tts_cost = self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match (&ready.model, ready.kokoro_tts.as_ref()) {
                (PreparedIncrementalModel::KokoroTts(_), Some(prepared)) => Some(prepared),
                _ => None,
            })
            .map(|prepared| Self::kokoro_tts_stage_cost(Some(&binding), prepared))
            .transpose()?
            .flatten();
        self.execution_adapter_binding = Some(binding);
        if let Some((stage_id, cost)) = prepared_continuous_cost {
            self.install_prepared_stage_cost(stage_id, cost)?;
        }
        if let Some((stage_id, cost)) = prepared_asr_continuous_cost {
            self.install_prepared_stage_cost(stage_id, cost)?;
        }
        for (stage_id, cost) in prepared_lfm25_asr_costs {
            self.install_prepared_stage_cost(stage_id, cost)?;
        }
        for (stage_id, cost) in prepared_lfm25_tts_costs {
            self.install_prepared_stage_cost(stage_id, cost)?;
        }
        if let Some((stage_id, cost)) = prepared_tts_continuous_cost {
            self.install_prepared_stage_cost(stage_id, cost)?;
        }
        if let Some((stage_id, cost)) = prepared_tts_prefill_cost {
            self.install_prepared_stage_cost(stage_id, cost)?;
        }
        if let Some((stage_id, cost)) = prepared_kokoro_tts_cost {
            self.install_prepared_stage_cost(stage_id, cost)?;
        }
        Ok(())
    }

    pub(crate) fn execution_adapter_binding(&self) -> Option<&super::super::ExecutionAdapterBinding> {
        self.execution_adapter_binding.as_ref()
    }

    pub(crate) fn bind_v2_state_descriptor(
        &mut self,
        descriptor: crate::kv::v2::CapabilityStateDescriptorV2,
        fingerprint: [u8; 32],
    ) -> Result<()> {
        if self
            .v2_state_descriptor
            .as_ref()
            .is_some_and(|current| current != &descriptor)
        {
            return Err(Error::InvalidInput(
                "engine request is already bound to a different state ABI v2 contract".to_string(),
            ));
        }
        if self
            .v2_state_fingerprint
            .is_some_and(|current| current != fingerprint)
        {
            return Err(Error::InvalidInput(
                "engine request is already bound to a different state ABI v2 fingerprint"
                    .to_string(),
            ));
        }
        if let Some(execution) = &self.execution_adapter_binding {
            let expected = descriptor.fingerprint(&execution.stages)?;
            if expected != fingerprint {
                return Err(Error::InvalidInput(
                    "state ABI v2 fingerprint does not match the bound execution stages"
                        .to_string(),
                ));
            }
        }
        self.v2_state_descriptor = Some(descriptor);
        self.v2_state_fingerprint = Some(fingerprint);
        Ok(())
    }

    pub(crate) fn bind_v2_state_runtime(
        &mut self,
        runtime: Arc<crate::kv::v2::CapabilityStateRuntimeV2>,
        fingerprint: [u8; 32],
        backend: crate::backends::BackendKind,
    ) -> Result<()> {
        if runtime.state_fingerprint != fingerprint {
            return Err(Error::InvalidInput(
                "state ABI v2 runtime fingerprint does not match its capability binding"
                    .to_string(),
            ));
        }
        let execution = self.execution_adapter_binding.as_ref().ok_or_else(|| {
            Error::InvalidInput(
                "state ABI v2 runtime requires a bound execution adapter".to_string(),
            )
        })?;
        runtime.validate_against(backend, execution)?;
        if self
            .v2_state_runtime
            .as_ref()
            .is_some_and(|current| current.id != runtime.id)
        {
            return Err(Error::InvalidInput(
                "engine request is already bound to a different state ABI v2 runtime".to_string(),
            ));
        }
        if self
            .v2_state_runtime
            .as_ref()
            .is_some_and(|current| current.id == runtime.id)
        {
            return Ok(());
        }
        self.bind_v2_state_descriptor(runtime.descriptor.clone(), fingerprint)?;
        self.v2_state_runtime = Some(runtime);
        Ok(())
    }

    pub(crate) fn v2_state_runtime(&self) -> Option<&Arc<crate::kv::v2::CapabilityStateRuntimeV2>> {
        self.v2_state_runtime.as_ref()
    }

    pub(crate) fn v2_state_descriptor(
        &self,
    ) -> Option<&crate::kv::v2::CapabilityStateDescriptorV2> {
        self.v2_state_descriptor.as_ref()
    }

    #[cfg(test)]
    pub(crate) fn v2_state_fingerprint(&self) -> Option<[u8; 32]> {
        self.v2_state_fingerprint
    }

    pub(crate) fn install_managed_cache_runtime(
        &mut self,
        runtime: Arc<super::super::cache::managed::ManagedKvModelRuntime>,
    ) -> Result<()> {
        if self.model_instance_id != Some(runtime.plan().model_instance) {
            return Err(Error::InvalidInput(
                "managed-cache runtime does not match the request model instance".to_string(),
            ));
        }
        if self
            .managed_cache_runtime
            .as_ref()
            .is_some_and(|current| current.plan().id != runtime.plan().id)
        {
            return Err(Error::InvalidInput(
                "engine request is already bound to a different managed-cache plan".to_string(),
            ));
        }
        self.managed_cache_runtime = Some(runtime);
        Ok(())
    }

    /// Records the admission-time managed prefix cursor for the first prefill
    /// span. Must be called before the request enters the scheduler.
    pub(crate) fn set_managed_prefix_cursor(&mut self, cursor: Option<u32>) {
        self.managed_prefix_cursor = cursor;
    }

    pub(crate) fn managed_prefix_cursor(&self) -> Option<u32> {
        self.managed_prefix_cursor
    }

    /// Phase 5a: the prepared prompt's stable-history boundary (tokens).
    pub(crate) fn managed_snapshot_boundary(&self) -> Option<u32> {
        self.managed_snapshot_boundary
    }

    pub(crate) fn managed_preamble_boundary(&self) -> Option<u32> {
        self.managed_preamble_boundary
    }

    #[cfg(test)]
    pub(crate) fn set_managed_snapshot_boundary_for_test(&mut self, boundary: Option<u32>) {
        self.managed_snapshot_boundary = boundary;
    }

    pub(crate) fn managed_cache_runtime(
        &self,
    ) -> Option<&Arc<super::super::cache::managed::ManagedKvModelRuntime>> {
        self.managed_cache_runtime.as_ref()
    }

    pub(super) fn validate_prepared_stage_costs(&self) -> Result<()> {
        if self.prepared_stage_costs.is_empty() {
            return Ok(());
        }
        let binding = self.execution_adapter_binding.as_ref().ok_or_else(|| {
            Error::InvalidInput(format!(
                "Request {} carries prepared stage costs without a loaded adapter binding",
                self.id
            ))
        })?;
        let mut stage_ids = std::collections::HashSet::new();
        for prepared in &self.prepared_stage_costs {
            if !stage_ids.insert(prepared.stage_id) {
                return Err(Error::InvalidInput(format!(
                    "Request {} carries duplicate prepared costs for one stage",
                    self.id
                )));
            }
            let stage = binding
                .stages
                .iter()
                .find(|stage| stage.id == prepared.stage_id)
                .ok_or_else(|| {
                    Error::InvalidInput(format!(
                        "Request {} prepared an unknown adapter stage",
                        self.id
                    ))
                })?;
            if stage.batch_mode == super::super::NativeBatchMode::None
                || prepared.cost.logical_units == 0
                || prepared.cost.tensor_elements == 0
                || !prepared
                    .cost
                    .workspace
                    .workspace_bytes()
                    .is_ok_and(|bytes| bytes <= stage.max_workspace_bytes)
            {
                return Err(Error::InvalidInput(format!(
                    "Request {} carries an invalid prepared tensor-stage cost",
                    self.id
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn install_prepared_stage_cost(
        &mut self,
        stage_id: StageId,
        cost: WorkCost,
    ) -> Result<()> {
        self.prepared_stage_costs
            .retain(|prepared| prepared.stage_id != stage_id);
        self.prepared_stage_costs
            .push(PreparedStageCost { stage_id, cost });
        self.validate_prepared_stage_costs()
    }

    pub(crate) fn prepared_stage_cost(&self, stage_id: StageId) -> Option<WorkCost> {
        self.prepared_stage_costs
            .iter()
            .find(|prepared| prepared.stage_id == stage_id)
            .map(|prepared| prepared.cost)
    }
}
