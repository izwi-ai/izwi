//! Izwi Core - High-Performance Audio Inference Engine
//!
//! This crate provides a production-ready inference engine for audio models
//! (Qwen3-TTS, LFM2-Audio) on Apple Silicon and CPU devices.
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
//! let config = EngineCoreConfig::for_lfm2();
//! let engine = Engine::new(config)?;
//!
//! let request = EngineCoreRequest::tts("Hello, world!");
//! let output = engine.generate(request).await?;
//! ```
#![allow(dead_code)]

pub mod artifacts;
pub mod audio;
pub mod backends;
pub mod catalog;
pub mod codecs;
pub mod config;
pub mod engine;
pub mod error;
pub mod model;
mod models;
pub mod runtime;
pub mod runtime_models;
pub mod serve_runtime;
pub mod tokenizer;

// Re-export main types from the new engine module
pub use engine::{
    CacheResidency, Engine, EngineCore, EngineCoreConfig, EngineCoreRequest, EngineMetrics,
    EngineOutput, GenerationParams, KVCacheManager, ModelExecutor, OutputProcessor,
    PinnedBlockHandle, RequestProcessor, RequestStatus, Scheduler, SchedulerConfig,
    SchedulingPolicy, StreamingOutput,
};

// Legacy re-exports for backward compatibility
pub use config::EngineConfig;
pub use error::{Error, Result};
pub use runtime::{
    AsrTranscription, ChatGeneration, ChunkStats, DiarizationConfig, DiarizationResult,
    DiarizationSegment, DiarizationTranscriptResult, DiarizationUtterance, DiarizationWord,
    GenerationRequest, GenerationResult,
};
pub use runtime::{
    AudioChunk, GenerationConfig, InferenceOptions, RuntimeService, RuntimeTelemetrySnapshot,
    SpeechToSpeechGeneration,
};
pub use serve_runtime::{ServeRuntimeConfig, ServeRuntimeConfigOverrides};

// Canonical catalog/artifact/runtime-model re-exports
pub use artifacts::{DownloadProgress, ModelDownloader, ModelManager, ModelWeights};
pub use catalog::{
    parse_chat_model_variant, parse_model_variant, parse_tts_model_variant,
    resolve_asr_model_variant, resolve_diarization_model_variant, ModelInfo, ModelStatus,
    ModelVariant,
};
pub use runtime_models::shared::chat::{
    parse_qwen35_multimodal_control_content, parse_qwen35_thinking_control_content,
    parse_qwen35_tools_control_content, qwen35_multimodal_control_content,
    qwen35_recommended_generation_params, qwen35_thinking_control_content,
    qwen35_tools_control_content, ChatMessage, ChatRole, Qwen35MultimodalInput,
    Qwen35MultimodalKind,
};

// Canonical native registry/device exports.
pub use backends::{DeviceProfile, DeviceSelector};
pub use runtime_models::ModelRegistry;

#[cfg(test)]
pub(crate) fn env_test_lock() -> &'static std::sync::Mutex<()> {
    use std::sync::{Mutex, OnceLock};

    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}
