//! Voxtral Realtime model implementation.
//!
//! Architecture:
//! - ≈3.4B Language Model (Mistral-based)
//! - ≈0.6B Audio Encoder (Whisper-based with causal attention)
//! - Block pooling (pool_size=4) for audio representations
//! - Time embedding for delay token conditioning
//!
//! Reference: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/voxtral_realtime.py

mod audio;
mod config;
pub(crate) mod model;
pub(crate) mod streaming;
mod tokenizer;

pub use model::VoxtralRealtimeModel;
pub(crate) use model::{
    VoxtralRealtimeCheckpoint, VoxtralRealtimeDecodeBatchRow, VoxtralRealtimeStep,
};
pub(crate) use streaming::{VoxtralRealtimeResourceUsage, VoxtralRealtimeState};
