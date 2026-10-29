//! Runtime orchestration layer.
//!
//! This is the canonical request lifecycle module (similar to runtime engines
//! in vLLM/TGI/llama.cpp style systems), while legacy `inference` paths are
//! maintained as compatibility shims.

mod adapters;
mod asr;
mod asr_no_speech;
pub(crate) mod audio_io;
mod broker;
mod capabilities;
pub(crate) mod chat;
mod chat_request;
mod conformance;
mod coordinator;
mod diarization;
pub(crate) mod fish_capacity;
mod forced_alignment;
mod health;
mod kokoro;
mod lfm3_audio_chat;
#[cfg(test)]
mod lfm3_audio_tests;
mod lifecycle;
mod pipeline;
mod request;
pub(crate) mod rollout;
mod routing;
mod service;
mod speech_to_speech;
mod telemetry;
mod tts;
mod types;
mod voice_metrics;
mod voice_session;

pub(crate) use asr::vibevoice_asr_planned_max_tokens;
pub(crate) use asr::{
    asr_automatic_context, asr_sequence_ceiling, compose_invocation_audio_limit,
    ASR_TEXT_PROMPT_TOKEN_RESERVE,
};

pub(crate) use adapters::capability_rows;
pub use adapters::CapabilityKind;
pub(crate) use asr::{
    granite_asr_planned_max_tokens, granite_auto_asr_max_tokens_for_duration,
    granite_speech5_asr_planned_max_tokens, invocation_decode_room,
};
pub use asr::{
    validate_asr_language, validate_asr_task, RuntimeAsrRealtimeEvent, RuntimeAsrRealtimeStream,
};
pub use chat::AdmittedChatStream;
pub use chat_request::{
    chat_supports_json_object, chat_supports_tools, resolve_chat_max_tokens, resolve_chat_sampling,
    ChatGenerationControls, ChatSamplingRequest, ResolvedChatSampling, CHAT_MAX_TOKENS_UNSET,
};
pub use conformance::{
    capability_conformance_cases, required_conformance_capabilities, CapabilityConformanceCase,
    ConformanceExecutionClass,
};
pub use coordinator::{
    system_memory_snapshot, CoordinatorLane, CoordinatorSnapshot, InferenceCoordinator,
    JobCancellation, JobSpec,
};
pub(crate) use coordinator::{PhysicalExecutionAdmission, PhysicalExecutionLease};
pub use forced_alignment::{AlignedWindow, AlignmentSource, AlignmentWindow, WindowedAlignment};
pub use health::RuntimeHealth;
pub(crate) use health::{fatal_device_fault, fatal_device_fault_message};
pub use lifecycle::SessionModelLeases;
pub use service::{
    stream_slow_consumer_timeout, RuntimeChatInvocation, RuntimeChatInvocationEvent,
    RuntimeChatInvocationRequest, RuntimeChatTeardown, RuntimeChatTeardownDisposition,
    RuntimeService, DEFAULT_STREAM_SLOW_CONSUMER_TIMEOUT,
};
pub use telemetry::histogram as prometheus;
pub use telemetry::{
    runtime_trace_contracts, sanitized_replay_record, trace_contract_for_phase,
    EngineRuntimeTelemetrySnapshot, InferenceBrokerRuntimeTelemetrySnapshot,
    PipelineRuntimeTelemetrySnapshot, ReplayRedaction, RuntimeLatencyStats,
    RuntimeObservabilityTelemetrySnapshot, RuntimeObservationContext, RuntimeReplayRecord,
    RuntimeStageObservation, RuntimeStageOutcome, RuntimeStageOutputCounters, RuntimeStageTiming,
    RuntimeTelemetrySnapshot, RuntimeTraceContract, RuntimeTracePhase,
    RuntimeWorkloadClassTelemetrySnapshot, VoiceRuntimeTelemetrySnapshot, RUNTIME_REPLAY_REDACTION,
    RUNTIME_TRACE_CONTRACTS, TRACE_CAPABILITY, TRACE_CORRELATION_ID, TRACE_ERROR_KIND,
    TRACE_EXECUTION_TARGET, TRACE_MODEL_VARIANT, TRACE_PIPELINE_KIND, TRACE_PIPELINE_STAGE,
    TRACE_REQUEST_ID, TRACE_STREAMING_MODE,
};
pub use types::{
    AsrTranscription, AudioChunk, ChatFinishReason, ChatGeneration, ChunkStats, DiarizationConfig,
    DiarizationPipelineOptions, DiarizationResult, DiarizationSegment, DiarizationTranscriptResult,
    DiarizationUtterance, DiarizationWord, GenerationConfig, GenerationRequest, GenerationResult,
    InferenceOptions, RuntimeRequestContext, SpeakerAttributedAsrResult,
    SpeakerAttributedAsrStatus, SpeakerAttributedAsrTurn, SpeechToSpeechGeneration, TtsStopReason,
    TtsStreamingMode,
};
pub use voice_metrics::{
    prometheus_voice_metric_name, prometheus_voice_metric_type, voice_metric_catalog,
    voice_metric_prometheus_contract, VoiceLatency, VoiceMetricDescriptor, VOICE_ASR_FINAL_SECONDS,
    VOICE_ASR_FIRST_PARTIAL_SECONDS, VOICE_AUDIO_EGRESS_UNDERRUNS_TOTAL,
    VOICE_AUDIO_INGRESS_DROPPED_FRAMES_TOTAL, VOICE_BARGE_IN_LATENCY_SECONDS, VOICE_BARGE_IN_TOTAL,
    VOICE_ENDPOINTING_LATENCY_SECONDS, VOICE_LLM_FIRST_TOKEN_SECONDS, VOICE_METRIC_CATALOG,
    VOICE_SESSION_CLOSED_TOTAL, VOICE_SESSION_DURATION_SECONDS, VOICE_SESSION_INTERRUPTED_TOTAL,
    VOICE_SESSION_STARTED_TOTAL, VOICE_SPEECH_REJECTED_TOTAL, VOICE_STREAM_BACKPRESSURE_TOTAL,
    VOICE_TTS_FIRST_AUDIO_SECONDS, VOICE_TURN_FIRST_AUDIO_SECONDS,
};
pub use voice_session::{VoiceSession, VoiceSessionPhase};

// Allow executor regression tests to validate the real loaded adapter join.
#[cfg(test)]
pub(crate) use adapters::{LoadedModelBundleDraft, RuntimeAdapterRegistry};
