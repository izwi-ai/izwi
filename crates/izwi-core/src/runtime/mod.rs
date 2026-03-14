//! Runtime orchestration layer.
//!
//! This is the canonical request lifecycle module (similar to runtime engines
//! in vLLM/TGI/llama.cpp style systems), while legacy `inference` paths are
//! maintained as compatibility shims.

mod asr;
pub(crate) mod audio_io;
mod chat;
mod diarization;
mod kokoro;
mod lifecycle;
mod service;
mod tts;
mod types;

pub use service::{RuntimeService, RuntimeTelemetrySnapshot};
pub use types::{
    AsrTranscription, AudioChunk, ChatGeneration, ChunkStats, DiarizationConfig, DiarizationResult,
    DiarizationSegment, DiarizationTranscriptResult, DiarizationUtterance, DiarizationWord,
    GenerationConfig, GenerationRequest, GenerationResult, InferenceOptions,
    SpeechToSpeechGeneration,
};
