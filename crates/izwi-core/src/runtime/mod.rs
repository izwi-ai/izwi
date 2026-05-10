//! Runtime orchestration layer.
//!
//! This is the canonical request lifecycle module (similar to runtime engines
//! in vLLM/TGI/llama.cpp style systems), while legacy `inference` paths are
//! maintained as compatibility shims.

mod adapters;
mod asr;
pub(crate) mod audio_io;
mod broker;
mod chat;
mod conformance;
mod diarization;
mod kokoro;
mod lifecycle;
mod pipeline;
mod request;
mod service;
mod speech_to_speech;
mod tts;
mod types;
mod voice_metrics;
mod voice_session;

pub use service::{
    EngineRuntimeTelemetrySnapshot, InferenceBrokerRuntimeTelemetrySnapshot,
    PipelineRuntimeTelemetrySnapshot, RuntimeService, RuntimeTelemetrySnapshot,
    VoiceRuntimeTelemetrySnapshot,
};
pub use conformance::{
    capability_conformance_cases, required_conformance_capabilities,
    CapabilityConformanceCase, ConformanceCapability, ConformanceExecutionClass,
};
pub use types::{
    AsrTranscription, AudioChunk, ChatGeneration, ChunkStats, DiarizationConfig, DiarizationResult,
    DiarizationSegment, DiarizationTranscriptResult, DiarizationUtterance, DiarizationWord,
    GenerationConfig, GenerationRequest, GenerationResult, InferenceOptions,
    SpeechToSpeechGeneration,
};
pub use voice_metrics::{
    prometheus_voice_metric_name, prometheus_voice_metric_type, voice_metric_catalog,
    voice_metric_prometheus_contract, VoiceMetricDescriptor, VOICE_ASR_FINAL_MS,
    VOICE_ASR_FIRST_PARTIAL_MS, VOICE_AUDIO_EGRESS_UNDERRUNS_TOTAL,
    VOICE_AUDIO_INGRESS_DROPPED_FRAMES_TOTAL, VOICE_AUDIO_INGRESS_JITTER_MS,
    VOICE_BARGE_IN_LATENCY_MS, VOICE_BARGE_IN_TOTAL, VOICE_ENDPOINTING_LATENCY_MS,
    VOICE_LLM_FIRST_TOKEN_MS, VOICE_METRIC_CATALOG, VOICE_MODEL_READY_TOTAL,
    VOICE_SESSION_CLOSED_TOTAL, VOICE_SESSION_DURATION_MS, VOICE_SESSION_INTERRUPTED_TOTAL,
    VOICE_SESSION_STARTED_TOTAL, VOICE_STREAM_BACKPRESSURE_TOTAL, VOICE_TTS_FIRST_AUDIO_MS,
    VOICE_VAD_SPEECH_START_MS,
};
pub use voice_session::{VoiceSession, VoiceSessionPhase};
