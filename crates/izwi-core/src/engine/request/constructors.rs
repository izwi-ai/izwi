//! EngineCoreRequest constructors per task and the builder-style setters.

use super::*;

impl EngineCoreRequest {
    /// Create a new TTS request.
    pub fn tts(text: impl Into<String>) -> Self {
        let text = text.into();
        Self {
            id: Uuid::new_v4().to_string(),
            task: EngineTask::Tts(TtsEngineInput {
                // Large direct metadata is validated before the canonical
                // typed view is populated during request processing.
                text: String::new(),
                reference_audio: None,
                reference_text: None,
                voice_description: None,
            }),
            task_type: TaskType::TTS,
            model_variant: None,
            model_instance_id: None,
            execution_adapter_binding: None,
            v2_state_descriptor: None,
            v2_state_fingerprint: None,
            v2_state_runtime: None,
            managed_cache_runtime: None,
            managed_prefix_cursor: None,
            managed_snapshot_boundary: None,
            managed_preamble_boundary: None,
            prepared_stage_costs: Vec::new(),
            prepared_sequence_input_tokens: None,
            prepared_asr_execution_shape: None,
            prepared_asr_route_plan: None,
            prepared_asr_audio: None,
            asr_source_audio_digest: None,
            prepared_asr_encoder_artifact: None,
            stream_staging: StreamStagingBuffer::default(),
            collect_audio_samples: true,
            tenant_key: None,
            audio_stream_external_queue_capacity: 0,
            audio_stream_engine_queue_capacity: None,
            text: Some(text),
            chat_messages: None,
            chat_config: ChatRequestConfig::default(),
            language: None,
            correlation_id: None,
            audio_input: None,
            audio_bytes: None,
            asr_prompt: None,
            asr_auto_max_tokens: false,
            asr_native_word_timestamps: false,
            asr_task: AsrTaskOptions::default(),
            reference_audio: None,
            reference_text: None,
            reference_language: None,
            voice_description: None,
            system_prompt: None,
            params: GenerationParams::default(),
            priority: Priority::Normal,
            workload_class: WorkloadClass::Online,
            admission_ms: None,
            arrival_time: Instant::now(),
            runtime_admitted_at: None,
            deadline: None,
            prompt_tokens: Vec::new(),
            chat_execution_ready: None,
            incremental_model_execution_ready: None,
            realtime_asr_ingress: None,
            streaming: false,
            stream_policy: EngineStreamPolicy::default(),
            streaming_tx: None,
            cancellation: None,
        }
    }

    /// Create a new ASR request.
    pub fn asr(audio_base64: impl Into<String>) -> Self {
        let audio_base64 = audio_base64.into();
        Self {
            id: Uuid::new_v4().to_string(),
            task: EngineTask::Asr(AsrEngineInput {
                // The compatibility field below owns the caller buffer until
                // bounded processing moves it into this typed representation.
                audio: EngineAudioInput::Base64(String::new()),
                language: None,
                prompt: None,
            }),
            task_type: TaskType::ASR,
            model_variant: None,
            model_instance_id: None,
            execution_adapter_binding: None,
            v2_state_descriptor: None,
            v2_state_fingerprint: None,
            v2_state_runtime: None,
            managed_cache_runtime: None,
            managed_prefix_cursor: None,
            managed_snapshot_boundary: None,
            managed_preamble_boundary: None,
            prepared_stage_costs: Vec::new(),
            prepared_sequence_input_tokens: None,
            prepared_asr_execution_shape: None,
            prepared_asr_route_plan: None,
            prepared_asr_audio: None,
            asr_source_audio_digest: None,
            prepared_asr_encoder_artifact: None,
            stream_staging: StreamStagingBuffer::default(),
            collect_audio_samples: true,
            tenant_key: None,
            audio_stream_external_queue_capacity: 0,
            audio_stream_engine_queue_capacity: None,
            text: None,
            chat_messages: None,
            chat_config: ChatRequestConfig::default(),
            language: None,
            correlation_id: None,
            audio_input: Some(audio_base64),
            audio_bytes: None,
            asr_prompt: None,
            asr_auto_max_tokens: false,
            asr_native_word_timestamps: false,
            asr_task: AsrTaskOptions::default(),
            reference_audio: None,
            reference_text: None,
            reference_language: None,
            voice_description: None,
            system_prompt: None,
            params: GenerationParams::default(),
            priority: Priority::Normal,
            workload_class: WorkloadClass::Online,
            admission_ms: None,
            arrival_time: Instant::now(),
            runtime_admitted_at: None,
            deadline: None,
            prompt_tokens: Vec::new(),
            chat_execution_ready: None,
            incremental_model_execution_ready: None,
            realtime_asr_ingress: None,
            streaming: false,
            stream_policy: EngineStreamPolicy::default(),
            streaming_tx: None,
            cancellation: None,
        }
    }

    /// Create a new ASR request from already-decoded audio bytes.
    pub fn asr_bytes(audio_bytes: impl Into<Vec<u8>>) -> Self {
        let audio_bytes = audio_bytes.into();
        Self {
            id: Uuid::new_v4().to_string(),
            task: EngineTask::Asr(AsrEngineInput {
                // Keep construction allocation-free beyond the caller-owned
                // buffer; processing installs it here after source preflight.
                audio: EngineAudioInput::Bytes(Vec::new()),
                language: None,
                prompt: None,
            }),
            task_type: TaskType::ASR,
            model_variant: None,
            model_instance_id: None,
            execution_adapter_binding: None,
            v2_state_descriptor: None,
            v2_state_fingerprint: None,
            v2_state_runtime: None,
            managed_cache_runtime: None,
            managed_prefix_cursor: None,
            managed_snapshot_boundary: None,
            managed_preamble_boundary: None,
            prepared_stage_costs: Vec::new(),
            prepared_sequence_input_tokens: None,
            prepared_asr_execution_shape: None,
            prepared_asr_route_plan: None,
            prepared_asr_audio: None,
            asr_source_audio_digest: None,
            prepared_asr_encoder_artifact: None,
            stream_staging: StreamStagingBuffer::default(),
            collect_audio_samples: true,
            tenant_key: None,
            audio_stream_external_queue_capacity: 0,
            audio_stream_engine_queue_capacity: None,
            text: None,
            chat_messages: None,
            chat_config: ChatRequestConfig::default(),
            language: None,
            correlation_id: None,
            audio_input: None,
            audio_bytes: Some(audio_bytes),
            asr_prompt: None,
            asr_auto_max_tokens: false,
            asr_native_word_timestamps: false,
            asr_task: AsrTaskOptions::default(),
            reference_audio: None,
            reference_text: None,
            reference_language: None,
            voice_description: None,
            system_prompt: None,
            params: GenerationParams::default(),
            priority: Priority::Normal,
            workload_class: WorkloadClass::Online,
            admission_ms: None,
            arrival_time: Instant::now(),
            runtime_admitted_at: None,
            deadline: None,
            prompt_tokens: Vec::new(),
            chat_execution_ready: None,
            incremental_model_execution_ready: None,
            realtime_asr_ingress: None,
            streaming: false,
            stream_policy: EngineStreamPolicy::default(),
            streaming_tx: None,
            cancellation: None,
        }
    }

    /// Create a new chat request.
    pub fn chat(messages: Vec<ChatMessage>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            task: EngineTask::Chat(ChatEngineInput {
                messages: messages.clone(),
                chat_config: ChatRequestConfig::default(),
                prompt_tokens: Vec::new(),
            }),
            task_type: TaskType::Chat,
            model_variant: None,
            model_instance_id: None,
            execution_adapter_binding: None,
            v2_state_descriptor: None,
            v2_state_fingerprint: None,
            v2_state_runtime: None,
            managed_cache_runtime: None,
            managed_prefix_cursor: None,
            managed_snapshot_boundary: None,
            managed_preamble_boundary: None,
            prepared_stage_costs: Vec::new(),
            prepared_sequence_input_tokens: None,
            prepared_asr_execution_shape: None,
            prepared_asr_route_plan: None,
            prepared_asr_audio: None,
            asr_source_audio_digest: None,
            prepared_asr_encoder_artifact: None,
            stream_staging: StreamStagingBuffer::default(),
            collect_audio_samples: true,
            tenant_key: None,
            audio_stream_external_queue_capacity: 0,
            audio_stream_engine_queue_capacity: None,
            text: None,
            chat_messages: Some(messages),
            chat_config: ChatRequestConfig::default(),
            language: None,
            correlation_id: None,
            audio_input: None,
            audio_bytes: None,
            asr_prompt: None,
            asr_auto_max_tokens: false,
            asr_native_word_timestamps: false,
            asr_task: AsrTaskOptions::default(),
            reference_audio: None,
            reference_text: None,
            reference_language: None,
            voice_description: None,
            system_prompt: None,
            params: GenerationParams::default(),
            priority: Priority::Normal,
            workload_class: WorkloadClass::Online,
            admission_ms: None,
            arrival_time: Instant::now(),
            runtime_admitted_at: None,
            deadline: None,
            prompt_tokens: Vec::new(),
            chat_execution_ready: None,
            incremental_model_execution_ready: None,
            realtime_asr_ingress: None,
            streaming: false,
            stream_policy: EngineStreamPolicy::default(),
            streaming_tx: None,
            cancellation: None,
        }
    }

    /// Create a new speech-to-speech request.
    pub fn speech_to_speech(audio_base64: impl Into<String>) -> Self {
        let audio_base64 = audio_base64.into();
        Self {
            id: Uuid::new_v4().to_string(),
            task: EngineTask::SpeechToSpeech(AudioChatEngineInput {
                audio: EngineAudioInput::Base64(String::new()),
                messages: Vec::new(),
                system_prompt: None,
            }),
            task_type: TaskType::SpeechToSpeech,
            model_variant: None,
            model_instance_id: None,
            execution_adapter_binding: None,
            v2_state_descriptor: None,
            v2_state_fingerprint: None,
            v2_state_runtime: None,
            managed_cache_runtime: None,
            managed_prefix_cursor: None,
            managed_snapshot_boundary: None,
            managed_preamble_boundary: None,
            prepared_stage_costs: Vec::new(),
            prepared_sequence_input_tokens: None,
            prepared_asr_execution_shape: None,
            prepared_asr_route_plan: None,
            prepared_asr_audio: None,
            asr_source_audio_digest: None,
            prepared_asr_encoder_artifact: None,
            stream_staging: StreamStagingBuffer::default(),
            collect_audio_samples: true,
            tenant_key: None,
            audio_stream_external_queue_capacity: 0,
            audio_stream_engine_queue_capacity: None,
            text: None,
            chat_messages: None,
            chat_config: ChatRequestConfig::default(),
            language: None,
            correlation_id: None,
            audio_input: Some(audio_base64),
            audio_bytes: None,
            asr_prompt: None,
            asr_auto_max_tokens: false,
            asr_native_word_timestamps: false,
            asr_task: AsrTaskOptions::default(),
            reference_audio: None,
            reference_text: None,
            reference_language: None,
            voice_description: None,
            system_prompt: None,
            params: GenerationParams::default(),
            priority: Priority::Normal,
            workload_class: WorkloadClass::Online,
            admission_ms: None,
            arrival_time: Instant::now(),
            runtime_admitted_at: None,
            deadline: None,
            prompt_tokens: Vec::new(),
            chat_execution_ready: None,
            incremental_model_execution_ready: None,
            realtime_asr_ingress: None,
            streaming: false,
            stream_policy: EngineStreamPolicy::default(),
            streaming_tx: None,
            cancellation: None,
        }
    }

    /// Create a new speech-to-speech request from already-decoded audio bytes.
    pub fn speech_to_speech_bytes(audio_bytes: impl Into<Vec<u8>>) -> Self {
        let audio_bytes = audio_bytes.into();
        Self {
            id: Uuid::new_v4().to_string(),
            task: EngineTask::SpeechToSpeech(AudioChatEngineInput {
                audio: EngineAudioInput::Bytes(Vec::new()),
                messages: Vec::new(),
                system_prompt: None,
            }),
            task_type: TaskType::SpeechToSpeech,
            model_variant: None,
            model_instance_id: None,
            execution_adapter_binding: None,
            v2_state_descriptor: None,
            v2_state_fingerprint: None,
            v2_state_runtime: None,
            managed_cache_runtime: None,
            managed_prefix_cursor: None,
            managed_snapshot_boundary: None,
            managed_preamble_boundary: None,
            prepared_stage_costs: Vec::new(),
            prepared_sequence_input_tokens: None,
            prepared_asr_execution_shape: None,
            prepared_asr_route_plan: None,
            prepared_asr_audio: None,
            asr_source_audio_digest: None,
            prepared_asr_encoder_artifact: None,
            stream_staging: StreamStagingBuffer::default(),
            collect_audio_samples: true,
            tenant_key: None,
            audio_stream_external_queue_capacity: 0,
            audio_stream_engine_queue_capacity: None,
            text: None,
            chat_messages: None,
            chat_config: ChatRequestConfig::default(),
            language: None,
            correlation_id: None,
            audio_input: None,
            audio_bytes: Some(audio_bytes),
            asr_prompt: None,
            asr_auto_max_tokens: false,
            asr_native_word_timestamps: false,
            asr_task: AsrTaskOptions::default(),
            reference_audio: None,
            reference_text: None,
            reference_language: None,
            voice_description: None,
            system_prompt: None,
            params: GenerationParams::default(),
            priority: Priority::Normal,
            workload_class: WorkloadClass::Online,
            admission_ms: None,
            arrival_time: Instant::now(),
            runtime_admitted_at: None,
            deadline: None,
            prompt_tokens: Vec::new(),
            chat_execution_ready: None,
            incremental_model_execution_ready: None,
            realtime_asr_ingress: None,
            streaming: false,
            stream_policy: EngineStreamPolicy::default(),
            streaming_tx: None,
            cancellation: None,
        }
    }

    /// Set model variant.
    pub fn with_model_variant(mut self, model_variant: ModelVariant) -> Self {
        self.model_variant = Some(model_variant);
        self
    }

    /// Set generation parameters.
    pub fn with_params(mut self, params: GenerationParams) -> Self {
        self.params = params;
        self
    }

    /// Set priority.
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Set an absolute end-to-end deadline.
    pub fn with_deadline(mut self, deadline: Option<Instant>) -> Self {
        self.deadline = deadline;
        self
    }

    /// Set workload class.
    pub fn with_workload_class(mut self, workload_class: WorkloadClass) -> Self {
        self.workload_class = workload_class;
        self
    }

    /// Enable streaming.
    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
        if streaming && self.workload_class == WorkloadClass::Online {
            self.workload_class = WorkloadClass::Streaming;
        }
        self
    }

    /// Set streaming backpressure policy.
    pub fn with_stream_policy(mut self, stream_policy: EngineStreamPolicy) -> Self {
        self.stream_policy = stream_policy;
        self
    }

    /// Set voice/speaker.
    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.params.voice = Some(voice.into());
        self
    }

    /// Set reference audio for voice cloning.
    pub fn with_reference(mut self, audio: impl Into<String>, text: impl Into<String>) -> Self {
        self.reference_audio = Some(audio.into());
        self.reference_text = Some(text.into());
        self.sync_task_from_fields();
        self
    }

    /// Set voice description.
    pub fn with_voice_description(mut self, description: impl Into<String>) -> Self {
        self.voice_description = Some(description.into());
        self.sync_task_from_fields();
        self
    }

    /// Set language hint.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self.sync_task_from_fields();
        self
    }

    /// Ask for model-native word timestamps where the model emits them.
    pub fn with_asr_native_word_timestamps(mut self, enabled: bool) -> Self {
        self.asr_native_word_timestamps = enabled;
        self
    }

    /// Set the ASR translation target and keyword list.
    pub fn with_asr_task(mut self, task: AsrTaskOptions) -> Self {
        self.asr_task = task;
        self
    }

    /// Set ASR initial prompt/context hint.
    pub fn with_asr_prompt(mut self, prompt: impl Into<String>) -> Self {
        let prompt = prompt.into();
        let trimmed = prompt.trim();
        self.asr_prompt = (!trimmed.is_empty()).then_some(trimmed.to_string());
        self.sync_task_from_fields();
        self
    }

    /// Set request correlation ID.
    pub fn with_correlation_id(mut self, correlation_id: impl Into<String>) -> Self {
        self.correlation_id = Some(correlation_id.into());
        self
    }

    /// Set speech-to-speech system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self.sync_task_from_fields();
        self
    }

    /// Set chat-specific prompt/runtime configuration.
    pub fn with_chat_config(mut self, chat_config: ChatRequestConfig) -> Self {
        self.chat_config = chat_config;
        self.sync_task_from_fields();
        self
    }
}
