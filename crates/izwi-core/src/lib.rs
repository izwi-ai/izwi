//! Izwi Core - High-Performance Audio Inference Engine
//!
//! This crate provides a production-ready inference engine for audio models
//! (Qwen3-TTS, Kokoro) on Apple Silicon and CPU devices.
//!
//! # Architecture
//!
//! The engine follows vLLM's architecture patterns with:
//! - Request scheduling with FCFS/priority policies
//! - Paged KV-cache memory management
//! - Continuous batching support
//! - Streaming output
//!
//! # Example
//!
//! ```ignore
//! use izwi_core::engine::{Engine, EngineCoreConfig, EngineCoreRequest};
//!
//! let config = EngineCoreConfig::default();
//! let engine = Engine::new(config)?;
//!
//! let request = EngineCoreRequest::tts("Hello, world!");
//! let output = engine.generate(request).await?;
//! ```
#![allow(dead_code)]
// Inference entry points intentionally expose the complete tensor/state contract,
// model registries intentionally store heterogeneous implementations inline, and
// numerical kernels use explicit index loops where coordinates participate in
// multiple tensor strides. Refactoring these solely to satisfy shape/style lints
// would add indirection to hot paths without improving correctness; all other
// Clippy warnings remain denied.
#![allow(
    clippy::large_enum_variant,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

pub mod artifacts;
pub mod audio;
pub mod backends;
pub mod catalog;
pub mod codecs;
pub mod config;
pub mod engine;
pub mod error;
pub mod kernels;
pub mod kv;
pub mod model;
mod models;
pub mod residency;
pub mod runtime;
pub mod runtime_models;
pub mod serve_runtime;
pub mod tokenizer;

// Re-export main types from the new engine module
pub use engine::{
    AdapterAbiRevision, AdapterInstanceId, AsrEngineInput, AsrProgress, AsrProgressPhase,
    AudioChatEngineInput, BatchBudget, BatchId, BatchKey, BatchLaneKey, ChatEngineInput, Engine,
    EngineAudioInput, EngineCore, EngineCoreConfig, EngineCoreRequest, EngineMetrics, EngineOutput,
    EngineTask, ExecutionCapabilities, ExecutionDisposition, ExecutionDomain, ExecutionFailure,
    ExecutionGroupId, ExecutionMode, ExecutionPlan, ExecutionProfile, ExecutionReport,
    ExecutionState, ExecutorStepResult, FailureKind, FailureScope, FinishReason, GenerationParams,
    HealthImpact, ManagedKvArenaRuntimeSnapshot, ManagedKvCoordinatorSnapshot,
    ManagedKvModelRuntimeSnapshot, ManagedKvOperationSnapshot, ManagedKvRuntimeSnapshot,
    ManagedKvRuntimeTotalsSnapshot, ManagedKvTelemetrySnapshot, MembershipSafePoint, ModelExecutor,
    ModelInstanceId, ModelSessionResult, OutputProcessor, OutputVisibility, PhysicalBatch,
    PhysicalBatchReport, PhysicalBatchRowReport, PlanId, Priority, ReadyQuantum, RequestProcessor,
    RequestStatus, ResourceAmount, ResourceEstimate, ResourceLedger, ResourceReservation,
    ResourceVector, RetryDisposition, Scheduler, SchedulerConfig, SchedulingPolicy, SessionEpoch,
    SessionKey, StageDescriptor, StageId, StageProgressKind, StageShapePolicy, StateDisposition,
    StreamingOutput, TerminalOutcome, TokenId, TtsEngineInput, WorkCost, WorkUnit, WorkloadClass,
    YieldReason,
};

// Legacy re-exports for backward compatibility
pub use config::EngineConfig;
pub use error::{Error, Result};
pub use models::shared::telemetry::KernelPathTelemetrySnapshot;
pub use runtime::{
    capability_conformance_cases, prometheus_voice_metric_name, prometheus_voice_metric_type,
    required_conformance_capabilities, voice_metric_catalog, voice_metric_prometheus_contract,
    CapabilityConformanceCase, ConformanceCapability, ConformanceExecutionClass,
    VoiceMetricDescriptor, VOICE_ASR_FINAL_MS, VOICE_ASR_FIRST_PARTIAL_MS,
    VOICE_AUDIO_EGRESS_UNDERRUNS_TOTAL, VOICE_AUDIO_INGRESS_DROPPED_FRAMES_TOTAL,
    VOICE_AUDIO_INGRESS_JITTER_MS, VOICE_BARGE_IN_LATENCY_MS, VOICE_BARGE_IN_TOTAL,
    VOICE_ENDPOINTING_LATENCY_MS, VOICE_LLM_FIRST_TOKEN_MS, VOICE_METRIC_CATALOG,
    VOICE_MODEL_READY_TOTAL, VOICE_SESSION_CLOSED_TOTAL, VOICE_SESSION_DURATION_MS,
    VOICE_SESSION_INTERRUPTED_TOTAL, VOICE_SESSION_STARTED_TOTAL, VOICE_STREAM_BACKPRESSURE_TOTAL,
    VOICE_TTS_FIRST_AUDIO_MS, VOICE_VAD_SPEECH_START_MS,
};
pub use runtime::{
    runtime_trace_contracts, sanitized_replay_record, trace_contract_for_phase, AudioChunk,
    CoordinatorSnapshot, EngineRuntimeTelemetrySnapshot, GenerationConfig,
    InferenceBrokerRuntimeTelemetrySnapshot, InferenceOptions, PipelineRuntimeTelemetrySnapshot,
    ReplayRedaction, RuntimeAsrRealtimeEvent, RuntimeAsrRealtimeStream, RuntimeLatencyStats,
    RuntimeObservabilityTelemetrySnapshot, RuntimeObservationContext, RuntimeReplayRecord,
    RuntimeRequestContext, RuntimeService, RuntimeStageObservation, RuntimeStageOutcome,
    RuntimeStageOutputCounters, RuntimeStageTiming, RuntimeTelemetrySnapshot, RuntimeTraceContract,
    RuntimeTracePhase, RuntimeWorkloadClassTelemetrySnapshot, SpeechToSpeechGeneration,
    VoiceRuntimeTelemetrySnapshot, VoiceSession, VoiceSessionPhase, RUNTIME_REPLAY_REDACTION,
    RUNTIME_TRACE_CONTRACTS, TRACE_CAPABILITY, TRACE_CORRELATION_ID, TRACE_ERROR_KIND,
    TRACE_EXECUTION_TARGET, TRACE_MODEL_VARIANT, TRACE_PIPELINE_KIND, TRACE_PIPELINE_STAGE,
    TRACE_REQUEST_ID, TRACE_STREAMING_MODE,
};
pub use runtime::{
    AsrTranscription, ChatGeneration, ChunkStats, DiarizationConfig, DiarizationResult,
    DiarizationSegment, DiarizationTranscriptResult, DiarizationUtterance, DiarizationWord,
    GenerationRequest, GenerationResult,
};
pub use serve_runtime::{ServeRuntimeConfig, ServeRuntimeConfigOverrides};

// Canonical catalog/artifact/runtime-model re-exports
pub use artifacts::{
    DownloadProgress, ModelArtifactState, ModelDownloader, ModelLifecycleSnapshot, ModelManager,
    ModelResidencyState, ModelWeights,
};
pub use catalog::{
    parse_chat_model_variant, parse_model_variant, parse_tts_model_variant,
    resolve_asr_model_variant, resolve_diarization_model_variant, CudaEvidenceLevel,
    CudaExecutionStatus, CudaOperatorCapability, CudaOperatorKind, CudaProviderClass,
    CudaQuantizationInfo, CudaQuantizationSupportLevel, CudaSupportInfo, CudaSupportLevel,
    ModelInfo, ModelStatus, ModelVariant, SpeechModelCapabilities,
};
pub use runtime_models::shared::chat::{
    ChatMediaInput, ChatMediaKind, ChatMessage, ChatRequestConfig, ChatRole,
};

// Canonical native registry/device exports.
pub use backends::{DeviceProfile, DeviceSelector};
pub use runtime_models::{
    model_family_registrations, registration_for_variant, registrations_for_capability,
    AsrModelLease, ChatModelLease, FamilyRegistration, LoadedModelDiagnostics, LoadedModelRegistry,
    ModelRegistry, QwenTtsModelLease, MODEL_FAMILY_REGISTRATIONS,
};

#[cfg(test)]
pub(crate) fn env_test_lock() -> &'static std::sync::Mutex<()> {
    use std::sync::{Mutex, OnceLock};

    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}
