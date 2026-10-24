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

// Dead-code analysis is restored module by module: every module without an
// attribute below is checked by the compiler (and so by `clippy -D warnings`).
// `backends`, `kernels` and `models` still opt out: much of their code is
// reachable only from CUDA builds, which this gate cannot compile on every
// lane. Remove an attribute once its module compiles warning-free on every
// feature lane, including `cargo-cuda-compile`.
pub mod artifacts;
pub mod audio;
#[allow(dead_code)]
pub mod backends;
pub mod catalog;
pub mod codecs;
pub mod config;
pub mod engine;
pub mod env;
pub mod error;
#[allow(dead_code)]
pub mod kernels;
pub mod kv;
pub mod model;
#[allow(dead_code)]
mod models;
pub mod performance;
pub mod residency;
pub mod runtime;
pub mod runtime_models;
pub mod serve_runtime;
pub mod tokenizer;

// Re-export main types from the new engine module
pub use engine::{
    AdapterAbiRevision, AdapterInstanceId, AsrEngineInput, AsrProgress, AsrProgressPhase,
    AsrTaskOptions, AudioChatEngineInput, BatchBudget, BatchId, BatchKey, BatchLaneKey,
    CacheReleaseOutcome, ChatEngineInput, Engine, EngineAudioInput, EngineCore, EngineCoreConfig,
    EngineCoreRequest, EngineMetrics, EngineOutput, EngineTask, ExecutionCapabilities,
    ExecutionDisposition, ExecutionDomain, ExecutionFailure, ExecutionGroupId, ExecutionMode,
    ExecutionPlan, ExecutionProfile, ExecutionReport, ExecutionState, ExecutorStepResult,
    FailureKind, FailureScope, FinishReason, GenerationParams, HealthImpact,
    ManagedKvArenaRuntimeSnapshot, ManagedKvCoordinatorSnapshot, ManagedKvModelRuntimeSnapshot,
    ManagedKvOperationSnapshot, ManagedKvRuntimeSnapshot, ManagedKvRuntimeTotalsSnapshot,
    ManagedKvTelemetrySnapshot, MembershipSafePoint, ModelExecutor, ModelInstanceId,
    ModelSessionResult, OutputProcessor, OutputVisibility, PhysicalBatch, PhysicalBatchReport,
    PhysicalBatchRowReport, PlanId, Priority, ReadyQuantum, RequestProcessor, RequestStatus,
    ResourceAmount, ResourceEstimate, ResourceLedger, ResourceReservation, ResourceVector,
    RetryDisposition, Scheduler, SchedulerConfig, SchedulingPolicy, SessionEpoch, SessionKey,
    SpeechSamplingOptions, StageDescriptor, StageId, StageProgressKind, StageShapePolicy,
    StateDisposition, StreamingOutput, TerminalOutcome, TokenId, TtsEngineInput, WorkCost,
    WorkUnit, WorkloadClass, YieldReason,
};

// Legacy re-exports for backward compatibility
pub use config::{
    BatchSizePreference, ContextLengthPreference, EngineConfig, GraniteSpeechOptions,
    PhysicalExecutionCapacity, PhysicalExecutionMode, PhysicalInFlightLimit,
    RealtimeAsrSessionConfig,
};
pub use error::{Error, ErrorKind, Result};
/// The separator between two transcript pieces: none where a script is
/// written without spaces (CJK, Thai) or punctuation closes, else a space.
pub use izwi_asr_toolkit::spacing_between;
pub use models::shared::telemetry::KernelPathTelemetrySnapshot;
pub use performance::{
    CpuPerformanceConfig, CpuPerformanceConfigOverrides, CpuProjectionBackend,
    CudaPerformanceConfig, CudaPerformanceConfigOverrides, CudaProjectionBackend,
    LoadingIoStrategy, LoadingPerformanceConfig, LoadingPerformanceConfigOverrides,
    MetalPerformanceConfig, MetalPerformanceConfigOverrides, MetalProjectionBackend,
    OptimizationMode, PerformanceConfig, PerformanceConfigOverrides,
};
pub use runtime::{
    capability_conformance_cases, prometheus_voice_metric_name, prometheus_voice_metric_type,
    required_conformance_capabilities, voice_metric_catalog, voice_metric_prometheus_contract,
    CapabilityConformanceCase, CapabilityKind, ConformanceExecutionClass, VoiceLatency,
    VoiceMetricDescriptor, VOICE_ASR_FINAL_SECONDS, VOICE_ASR_FIRST_PARTIAL_SECONDS,
    VOICE_AUDIO_EGRESS_UNDERRUNS_TOTAL, VOICE_AUDIO_INGRESS_DROPPED_FRAMES_TOTAL,
    VOICE_BARGE_IN_LATENCY_SECONDS, VOICE_BARGE_IN_TOTAL, VOICE_ENDPOINTING_LATENCY_SECONDS,
    VOICE_LLM_FIRST_TOKEN_SECONDS, VOICE_METRIC_CATALOG, VOICE_SESSION_CLOSED_TOTAL,
    VOICE_SESSION_DURATION_SECONDS, VOICE_SESSION_INTERRUPTED_TOTAL, VOICE_SESSION_STARTED_TOTAL,
    VOICE_SPEECH_REJECTED_TOTAL, VOICE_STREAM_BACKPRESSURE_TOTAL, VOICE_TTS_FIRST_AUDIO_SECONDS, VOICE_TURN_FIRST_AUDIO_SECONDS,
};
pub use runtime::{
    chat_supports_json_object, chat_supports_tools, resolve_chat_max_tokens, resolve_chat_sampling,
    validate_asr_language, validate_asr_task, AlignedWindow, AlignmentSource, AlignmentWindow,
    AsrTranscription, ChatFinishReason, ChatGeneration, ChatGenerationControls,
    ChatSamplingRequest, ChunkStats, DiarizationConfig, DiarizationPipelineOptions,
    DiarizationResult, DiarizationSegment, DiarizationTranscriptResult, DiarizationUtterance,
    DiarizationWord, GenerationRequest, GenerationResult, JobCancellation, ResolvedChatSampling,
    TtsStopReason, TtsStreamingMode, WindowedAlignment, CHAT_MAX_TOKENS_UNSET,
};
pub use runtime::{
    runtime_trace_contracts, sanitized_replay_record, stream_slow_consumer_timeout,
    trace_contract_for_phase, AdmittedChatStream, AudioChunk, CoordinatorSnapshot,
    EngineRuntimeTelemetrySnapshot, GenerationConfig, InferenceBrokerRuntimeTelemetrySnapshot,
    InferenceOptions, PipelineRuntimeTelemetrySnapshot, ReplayRedaction, RuntimeAsrRealtimeEvent,
    RuntimeAsrRealtimeStream, RuntimeChatInvocation, RuntimeChatInvocationEvent,
    RuntimeChatInvocationRequest, RuntimeChatTeardown, RuntimeChatTeardownDisposition,
    RuntimeHealth, RuntimeLatencyStats, RuntimeObservabilityTelemetrySnapshot,
    RuntimeObservationContext, RuntimeReplayRecord, RuntimeRequestContext, RuntimeService,
    RuntimeStageObservation, RuntimeStageOutcome, RuntimeStageOutputCounters, RuntimeStageTiming,
    RuntimeTelemetrySnapshot, RuntimeTraceContract, RuntimeTracePhase,
    RuntimeWorkloadClassTelemetrySnapshot, SessionModelLeases, SpeechToSpeechGeneration,
    VoiceRuntimeTelemetrySnapshot, VoiceSession, VoiceSessionPhase,
    DEFAULT_STREAM_SLOW_CONSUMER_TIMEOUT, RUNTIME_REPLAY_REDACTION, RUNTIME_TRACE_CONTRACTS,
    TRACE_CAPABILITY, TRACE_CORRELATION_ID, TRACE_ERROR_KIND, TRACE_EXECUTION_TARGET,
    TRACE_MODEL_VARIANT, TRACE_PIPELINE_KIND, TRACE_PIPELINE_STAGE, TRACE_REQUEST_ID,
    TRACE_STREAMING_MODE,
};
pub use serve_runtime::{ServeRuntimeConfig, ServeRuntimeConfigOverrides};

// Canonical catalog/artifact/runtime-model re-exports
pub use artifacts::{
    DownloadProgress, ModelArtifactState, ModelDownloader, ModelLifecycleSnapshot, ModelManager,
    ModelResidencyState, ModelWeights,
};
pub use catalog::{
    normalize_stored_model_id, parse_chat_model_variant, parse_model_variant,
    parse_tts_model_variant, removed_model_variant, resolve_asr_model_variant,
    resolve_diarization_asr_model_variant, resolve_diarization_llm_variant,
    resolve_diarization_model_variant, resolve_diarization_model_variant_strict,
    validate_diarization_speaker_limits, ChatModelCapabilities, CudaEvidenceLevel,
    CudaExecutionStatus, CudaOperatorCapability, CudaOperatorKind, CudaProviderClass,
    CudaQuantizationInfo, CudaQuantizationSupportLevel, CudaSupportInfo, CudaSupportLevel,
    ModelFamily, ModelInfo, ModelStatus, ModelVariant, RemovedModelVariant,
    SpeechModelCapabilities, REMOVED_MODEL_VARIANTS,
};
pub use runtime_models::shared::chat::{
    ChatMediaInput, ChatMediaKind, ChatMessage, ChatReasoningEffort, ChatRequestConfig, ChatRole,
    ChatTemplateKwargs, ChatTruncation,
};

/// How `<think>` reasoning splits from a chat answer, shared with the gateway.
pub use izwi_model_catalog::chat::reasoning as chat_reasoning;

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
    let lock = LOCK.get_or_init(|| Mutex::new(()));
    // The lock orders tests that touch the process environment; it guards no
    // data. A test that panics while holding it must fail alone, not poison
    // every later test into a PoisonError.
    lock.clear_poison();
    lock
}
