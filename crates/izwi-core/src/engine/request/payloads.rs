//! Direct payload selection, canonicalisation and validation.

use super::*;

impl EngineCoreRequest {
    pub(super) fn sync_task_from_fields(&mut self) {
        if self.task_type == TaskType::Chat {
            self.task = EngineTask::Chat(ChatEngineInput {
                messages: self.chat_messages.clone().unwrap_or_default(),
                chat_config: self.chat_config.clone(),
                prompt_tokens: self.prompt_tokens.clone(),
            });
        }
    }

    pub(super) fn empty_audio_for_compatibility_fields(
        audio_input: Option<&String>,
        audio_bytes: Option<&Vec<u8>>,
    ) -> EngineAudioInput {
        if audio_bytes.is_some() {
            EngineAudioInput::Bytes(Vec::new())
        } else if audio_input.is_some() {
            EngineAudioInput::Base64(String::new())
        } else {
            EngineAudioInput::Bytes(Vec::new())
        }
    }

    pub(super) fn selected_audio_input(&self) -> Option<BorrowedEngineAudioInput<'_>> {
        if let Some(audio) = self.audio_bytes.as_ref() {
            return Some(BorrowedEngineAudioInput::Bytes(audio));
        }
        if let Some(audio) = self.audio_input.as_ref() {
            return Some(BorrowedEngineAudioInput::Base64(audio));
        }
        let audio = match (&self.task, self.task_type) {
            (EngineTask::Asr(input), TaskType::ASR) => &input.audio,
            (EngineTask::SpeechToSpeech(input), TaskType::SpeechToSpeech) => &input.audio,
            _ => return None,
        };
        (!audio.is_empty()).then_some(match audio {
            EngineAudioInput::Base64(value) => BorrowedEngineAudioInput::Base64(value),
            EngineAudioInput::Bytes(value) => BorrowedEngineAudioInput::Bytes(value),
        })
    }

    pub(super) fn selected_tts_reference(&self) -> (Option<&String>, Option<&String>) {
        if self.reference_audio.is_some() || self.reference_text.is_some() {
            return (self.reference_audio.as_ref(), self.reference_text.as_ref());
        }
        match (&self.task, self.task_type) {
            (EngineTask::Tts(input), TaskType::TTS) => (
                input.reference_audio.as_ref(),
                input.reference_text.as_ref(),
            ),
            _ => (None, None),
        }
    }

    pub(super) fn selected_asr_language(&self) -> Option<&String> {
        self.language
            .as_ref()
            .or(match (&self.task, self.task_type) {
                (EngineTask::Asr(input), TaskType::ASR) => input.language.as_ref(),
                _ => None,
            })
    }

    pub(super) fn selected_asr_prompt(&self) -> Option<&String> {
        self.asr_prompt
            .as_ref()
            .or(match (&self.task, self.task_type) {
                (EngineTask::Asr(input), TaskType::ASR) => input.prompt.as_ref(),
                _ => None,
            })
    }

    pub(super) fn selected_speech_messages(&self) -> &[ChatMessage] {
        if let Some(messages) = self.chat_messages.as_deref() {
            return messages;
        }
        match (&self.task, self.task_type) {
            (EngineTask::SpeechToSpeech(input), TaskType::SpeechToSpeech) => &input.messages,
            _ => &[],
        }
    }

    pub(super) fn selected_speech_messages_capacity(&self) -> usize {
        if let Some(messages) = self.chat_messages.as_ref() {
            return messages.capacity();
        }
        match (&self.task, self.task_type) {
            (EngineTask::SpeechToSpeech(input), TaskType::SpeechToSpeech) => {
                input.messages.capacity()
            }
            _ => 0,
        }
    }

    pub(super) fn selected_speech_system_prompt(&self) -> Option<&String> {
        self.system_prompt
            .as_ref()
            .or(match (&self.task, self.task_type) {
                (EngineTask::SpeechToSpeech(input), TaskType::SpeechToSpeech) => {
                    input.system_prompt.as_ref()
                }
                _ => None,
            })
    }

    pub(crate) fn audio_bytes_for_execution(&self) -> Option<&[u8]> {
        match self.selected_audio_input()? {
            BorrowedEngineAudioInput::Bytes(value) => Some(value.as_slice()),
            BorrowedEngineAudioInput::Base64(_) => None,
        }
    }

    pub(crate) fn audio_base64_for_execution(&self) -> Option<&str> {
        match self.selected_audio_input()? {
            BorrowedEngineAudioInput::Base64(value) => Some(value.as_str()),
            BorrowedEngineAudioInput::Bytes(_) => None,
        }
    }

    pub(crate) fn tts_reference_audio_for_execution(&self) -> Option<&str> {
        self.selected_tts_reference().0.map(String::as_str)
    }

    pub(crate) fn tts_text_for_execution(&self) -> Option<&str> {
        self.text.as_deref()
    }

    pub(crate) fn tts_speaker_for_execution(&self) -> Option<&str> {
        self.params
            .speaker
            .as_deref()
            .or(self.params.voice.as_deref())
    }

    pub(crate) fn tts_reference_text_for_execution(&self) -> Option<&str> {
        self.selected_tts_reference().1.map(String::as_str)
    }

    pub(crate) fn has_tts_reference_for_execution(&self) -> bool {
        let (audio, text) = self.selected_tts_reference();
        audio.is_some() || text.is_some()
    }

    pub(crate) fn asr_language_for_execution(&self) -> Option<&str> {
        self.selected_asr_language().map(String::as_str)
    }

    pub(crate) fn asr_prompt_for_execution(&self) -> Option<&str> {
        self.selected_asr_prompt().map(String::as_str)
    }

    /// Identity of an ASR request's routed source: model, encoded upload,
    /// language, prompt and timestamp option. Prepared artifacts record it
    /// and every executor accessor re-checks it. Hashing a large upload is
    /// the expensive part, so accessors reuse the digest that preparation
    /// computed while the same upload buffer is in place, and hash in full
    /// only when the buffer was replaced (or on a clone of the request).
    pub(super) fn asr_execution_source_fingerprint(&self, model_variant: ModelVariant) -> Result<u64> {
        let audio = self.asr_execution_source_audio(model_variant)?;
        let identity = AsrSourceAudioIdentity::of(audio);
        let digest = match self.asr_source_audio_digest {
            Some(cached) if cached.identity == identity => cached.digest,
            _ => asr_source_audio_digest(audio),
        };
        Ok(self.asr_execution_source_fingerprint_with(model_variant, digest))
    }

    /// Hash the upload in full, cache the digest for the accessors, and
    /// return the source fingerprint. Every preparation install calls this,
    /// so an artifact always records the fingerprint of the bytes as they
    /// were when it was prepared.
    pub(super) fn refresh_asr_execution_source_fingerprint(
        &mut self,
        model_variant: ModelVariant,
    ) -> Result<u64> {
        let audio = self.asr_execution_source_audio(model_variant)?;
        let cached = AsrSourceAudioDigest {
            identity: AsrSourceAudioIdentity::of(audio),
            digest: asr_source_audio_digest(audio),
        };
        self.asr_source_audio_digest = Some(cached);
        Ok(self.asr_execution_source_fingerprint_with(model_variant, cached.digest))
    }

    pub(super) fn asr_execution_source_audio(
        &self,
        model_variant: ModelVariant,
    ) -> Result<BorrowedEngineAudioInput<'_>> {
        if self.task_type != TaskType::ASR || self.model_variant != Some(model_variant) {
            return Err(Error::InvalidInput(format!(
                "ASR request {} source identity does not match its routed task/model",
                self.id
            )));
        }
        self.selected_audio_input().ok_or_else(|| {
            Error::InvalidInput(format!("ASR request {} is missing audio input", self.id))
        })
    }

    pub(super) fn asr_execution_source_fingerprint_with(
        &self,
        model_variant: ModelVariant,
        audio_digest: u64,
    ) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        model_variant.hash(&mut hasher);
        audio_digest.hash(&mut hasher);
        self.asr_language_for_execution().hash(&mut hasher);
        self.asr_prompt_for_execution().hash(&mut hasher);
        self.asr_native_word_timestamps.hash(&mut hasher);
        self.asr_task.hash(&mut hasher);
        hasher.finish()
    }

    pub(crate) fn speech_messages_for_execution(&self) -> &[ChatMessage] {
        self.selected_speech_messages()
    }

    pub(crate) fn speech_system_prompt_for_execution(&self) -> Option<&str> {
        self.selected_speech_system_prompt().map(String::as_str)
    }

    /// Apply backend-independent source limits and collapse compatibility and
    /// typed payload views into one owner before request processing can clone
    /// the request or an executor can decode it.
    pub(crate) fn canonicalize_direct_payloads(&mut self, max_seq_len: usize) -> Result<()> {
        self.canonicalize_direct_payloads_with_limits(DirectPayloadLimits::production(max_seq_len)?)
    }

    pub(super) fn canonicalize_direct_payloads_with_limits(
        &mut self,
        limits: DirectPayloadLimits,
    ) -> Result<()> {
        let dynamic_metadata_bytes = self.validate_direct_dynamic_allocations(limits)?;
        match self.task_type {
            TaskType::ASR => {
                let language = self.selected_asr_language();
                let prompt = self.selected_asr_prompt();
                Self::validate_direct_metadata(
                    "ASR",
                    dynamic_metadata_bytes,
                    [language, prompt]
                        .into_iter()
                        .flatten()
                        .map(String::capacity),
                    limits.metadata_bytes,
                )?;
                if let Some(audio) = self.selected_audio_input() {
                    Self::validate_direct_audio(audio, limits.audio_source_bytes, "input audio")?;
                }
                let already_canonical = self.audio_input.is_none()
                    && self.audio_bytes.is_none()
                    && matches!(
                        &self.task,
                        EngineTask::Asr(input)
                            if !input.audio.is_empty()
                                && input.language.as_deref() == language.map(String::as_str)
                                && input.prompt.as_deref() == prompt.map(String::as_str)
                    );
                if !already_canonical {
                    let language = language.map(String::as_str).map(str::to_owned);
                    let prompt = prompt.map(String::as_str).map(str::to_owned);
                    let audio = if self.selected_audio_input().is_some() {
                        self.take_selected_audio_input()
                            .expect("validated selected audio must remain available")
                    } else {
                        Self::empty_audio_for_compatibility_fields(
                            self.audio_input.as_ref(),
                            self.audio_bytes.as_ref(),
                        )
                    };
                    self.task = EngineTask::Asr(AsrEngineInput {
                        audio,
                        language,
                        prompt,
                    });
                }
            }
            TaskType::SpeechToSpeech => {
                let messages = self.selected_speech_messages();
                let system_prompt = self.selected_speech_system_prompt();
                if messages.len() > limits.collection_items {
                    return Err(Error::InvalidInput(format!(
                        "Direct Engine speech-to-speech request contains {} messages, exceeding the {}-item context-derived limit",
                        messages.len(), limits.collection_items
                    )));
                }
                let message_storage = self
                    .selected_speech_messages_capacity()
                    .checked_mul(std::mem::size_of::<ChatMessage>())
                    .ok_or_else(|| {
                        Error::InvalidInput(
                            "Direct Engine speech-to-speech metadata size overflow".to_string(),
                        )
                    })?;
                Self::validate_direct_metadata(
                    "speech-to-speech",
                    dynamic_metadata_bytes,
                    std::iter::once(message_storage)
                        .chain(messages.iter().map(|message| message.content.capacity()))
                        .chain(system_prompt.map(String::capacity)),
                    limits.metadata_bytes,
                )?;
                if let Some(audio) = self.selected_audio_input() {
                    Self::validate_direct_audio(audio, limits.audio_source_bytes, "input audio")?;
                }
                let already_canonical = self.audio_input.is_none()
                    && self.audio_bytes.is_none()
                    && matches!(
                        &self.task,
                        EngineTask::SpeechToSpeech(input)
                            if !input.audio.is_empty()
                                && Self::chat_messages_match(&input.messages, messages)
                                && input.system_prompt.as_deref()
                                    == system_prompt.map(String::as_str)
                    );
                if !already_canonical {
                    let canonical_messages = messages.to_vec();
                    let system_prompt = system_prompt.map(String::as_str).map(str::to_owned);
                    let audio = if self.selected_audio_input().is_some() {
                        self.take_selected_audio_input()
                            .expect("validated selected audio must remain available")
                    } else {
                        Self::empty_audio_for_compatibility_fields(
                            self.audio_input.as_ref(),
                            self.audio_bytes.as_ref(),
                        )
                    };
                    self.task = EngineTask::SpeechToSpeech(AudioChatEngineInput {
                        audio,
                        messages: canonical_messages,
                        system_prompt,
                    });
                }
            }
            TaskType::TTS => {
                let (reference_audio, reference_text) = self.selected_tts_reference();
                Self::validate_direct_metadata(
                    "TTS",
                    dynamic_metadata_bytes,
                    [
                        self.text.as_ref(),
                        reference_text,
                        self.voice_description.as_ref(),
                        self.language.as_ref(),
                    ]
                    .into_iter()
                    .flatten()
                    .map(String::capacity),
                    limits.metadata_bytes,
                )?;
                if reference_audio.is_some() || reference_text.is_some() {
                    let reference_audio = reference_audio.ok_or_else(|| {
                        Error::InvalidInput(
                            "reference_audio and reference_text must both be provided".to_string(),
                        )
                    })?;
                    let reference_text = reference_text.ok_or_else(|| {
                        Error::InvalidInput(
                            "reference_audio and reference_text must both be provided".to_string(),
                        )
                    })?;
                    if reference_text.trim().is_empty() {
                        return Err(Error::InvalidInput(
                            "reference_text cannot be empty".to_string(),
                        ));
                    }
                    validate_base64_audio_retained_size(
                        reference_audio.capacity(),
                        limits.reference_source_bytes,
                    )?;
                    Self::validate_direct_base64(
                        reference_audio,
                        limits.reference_source_bytes,
                        "reference audio",
                    )?;
                    let reference_chars = reference_text.chars().count();
                    if reference_chars > limits.reference_text_chars {
                        return Err(Error::InvalidInput(format!(
                            "TTS reference_text contains {reference_chars} characters, exceeding the {}-character direct Engine limit",
                            limits.reference_text_chars
                        )));
                    }
                }
                let already_canonical = self.reference_audio.is_none()
                    && self.reference_text.is_none()
                    && matches!(
                        &self.task,
                        EngineTask::Tts(input)
                            if input.text == self.text.as_deref().unwrap_or_default()
                                && input.voice_description == self.voice_description
                    );
                if !already_canonical {
                    let (reference_audio, reference_text) = self.take_selected_tts_reference();
                    self.task = EngineTask::Tts(TtsEngineInput {
                        text: self.text.clone().unwrap_or_default(),
                        reference_audio,
                        reference_text,
                        voice_description: self.voice_description.clone(),
                    });
                }
            }
            TaskType::Chat => {}
        }
        self.sync_task_from_fields();
        Ok(())
    }

    pub(super) fn validate_direct_dynamic_allocations(&self, limits: DirectPayloadLimits) -> Result<usize> {
        for (label, count) in [
            ("stop sequences", self.params.stop_sequences.len()),
            ("stop token IDs", self.params.stop_token_ids.len()),
        ] {
            if count > limits.collection_items {
                return Err(Error::InvalidInput(format!(
                    "Direct Engine request contains {count} {label}, exceeding the {}-item context-derived limit",
                    limits.collection_items
                )));
            }
        }
        if self.task_type != TaskType::Chat && self.prompt_tokens.len() > limits.collection_items {
            return Err(Error::InvalidInput(format!(
                "Direct Engine request contains {} prompt tokens, exceeding the {}-item context-derived limit",
                self.prompt_tokens.len(),
                limits.collection_items
            )));
        }

        let stop_sequence_storage = self
            .params
            .stop_sequences
            .capacity()
            .checked_mul(std::mem::size_of::<String>())
            .ok_or_else(|| {
                Error::InvalidInput("Direct Engine stop-sequence storage size overflow".to_string())
            })?;
        let stop_token_storage = self
            .params
            .stop_token_ids
            .capacity()
            .checked_mul(std::mem::size_of::<TokenId>())
            .ok_or_else(|| {
                Error::InvalidInput("Direct Engine stop-token storage size overflow".to_string())
            })?;
        let prompt_token_storage = if self.task_type == TaskType::Chat {
            0
        } else {
            self.prompt_tokens
                .capacity()
                .checked_mul(std::mem::size_of::<TokenId>())
                .ok_or_else(|| {
                    Error::InvalidInput(
                        "Direct Engine prompt-token storage size overflow".to_string(),
                    )
                })?
        };
        let fixed = [
            self.params
                .speaker
                .as_ref()
                .map(String::capacity)
                .unwrap_or(0),
            self.params
                .voice
                .as_ref()
                .map(String::capacity)
                .unwrap_or(0),
            stop_sequence_storage,
            stop_token_storage,
            prompt_token_storage,
        ];
        Self::validate_direct_metadata(
            "request",
            0,
            fixed
                .into_iter()
                .chain(self.params.stop_sequences.iter().map(String::capacity)),
            limits.metadata_bytes,
        )
    }

    pub(super) fn chat_messages_match(left: &[ChatMessage], right: &[ChatMessage]) -> bool {
        left.len() == right.len()
            && left
                .iter()
                .zip(right)
                .all(|(left, right)| left.role == right.role && left.content == right.content)
    }

    pub(super) fn validate_direct_metadata(
        task: &str,
        initial: usize,
        fields: impl IntoIterator<Item = usize>,
        limit: usize,
    ) -> Result<usize> {
        if initial > limit {
            return Err(Error::InvalidInput(format!(
                "Direct Engine {task} metadata is {initial} bytes, exceeding the {limit}-byte context-derived limit"
            )));
        }
        let mut total = initial;
        for bytes in fields {
            total = total.checked_add(bytes).ok_or_else(|| {
                Error::InvalidInput(format!("Direct Engine {task} metadata size overflow"))
            })?;
            if total > limit {
                return Err(Error::InvalidInput(format!(
                    "Direct Engine {task} metadata is {total} bytes, exceeding the {limit}-byte context-derived limit"
                )));
            }
        }
        Ok(total)
    }

    pub(super) fn validate_direct_audio(
        audio: BorrowedEngineAudioInput<'_>,
        max_source_bytes: usize,
        label: &str,
    ) -> Result<()> {
        match audio {
            BorrowedEngineAudioInput::Base64(value) => {
                validate_base64_audio_retained_size(value.capacity(), max_source_bytes)?;
                Self::validate_direct_base64(value, max_source_bytes, label)
            }
            BorrowedEngineAudioInput::Bytes(value) => {
                if value.is_empty() {
                    return Err(Error::InvalidInput(format!(
                        "Direct Engine {label} cannot be empty"
                    )));
                }
                if value.capacity() > max_source_bytes {
                    return Err(Error::InvalidInput(format!(
                        "Direct Engine {label} retains {} bytes, exceeding the {max_source_bytes}-byte source limit",
                        value.capacity()
                    )));
                }
                Ok(())
            }
        }
    }

    pub(super) fn validate_direct_base64(value: &str, max_source_bytes: usize, label: &str) -> Result<()> {
        if value.is_empty() {
            return Err(Error::InvalidInput(format!(
                "Direct Engine {label} cannot be empty"
            )));
        }
        validate_base64_audio_source_input(value, max_source_bytes)
    }

    pub(super) fn take_selected_audio_input(&mut self) -> Option<EngineAudioInput> {
        if let Some(audio) = self.audio_bytes.take() {
            self.audio_input = None;
            return Some(EngineAudioInput::Bytes(audio));
        }
        if let Some(audio) = self.audio_input.take() {
            return Some(EngineAudioInput::Base64(audio));
        }
        match (&mut self.task, self.task_type) {
            (EngineTask::Asr(input), TaskType::ASR) => Self::take_nonempty_audio(&mut input.audio),
            (EngineTask::SpeechToSpeech(input), TaskType::SpeechToSpeech) => {
                Self::take_nonempty_audio(&mut input.audio)
            }
            _ => None,
        }
    }

    pub(super) fn take_nonempty_audio(audio: &mut EngineAudioInput) -> Option<EngineAudioInput> {
        if audio.is_empty() {
            return None;
        }
        Some(match audio {
            EngineAudioInput::Base64(value) => EngineAudioInput::Base64(std::mem::take(value)),
            EngineAudioInput::Bytes(value) => EngineAudioInput::Bytes(std::mem::take(value)),
        })
    }

    pub(super) fn take_selected_tts_reference(&mut self) -> (Option<String>, Option<String>) {
        if self.reference_audio.is_some() || self.reference_text.is_some() {
            return (self.reference_audio.take(), self.reference_text.take());
        }
        match (&mut self.task, self.task_type) {
            (EngineTask::Tts(input), TaskType::TTS) => {
                (input.reference_audio.take(), input.reference_text.take())
            }
            _ => (None, None),
        }
    }
}
