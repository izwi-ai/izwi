//! Concrete model architecture implementations grouped by family.
//!
//! Adding a new model should follow this pattern:
//! 1. Create a new family folder (or extend an existing one).
//! 2. Export that family here.
//! 3. Wire loading in `crate::models::registry`.

pub mod chatterbox;
pub mod fish_s2;
pub mod gemma3;
pub mod granite_speech;
pub mod granite_speech5;
pub mod kokoro;
pub mod lfm2;
pub mod lfm25_audio;
pub mod lfm3;
pub mod lfm3_audio;
pub mod maya1;
pub mod maya2;
pub mod nemotron;
pub mod nemotron_transcribe;
pub mod parakeet;
pub mod qwen3;
pub mod qwen35;
pub mod qwen36moe;
pub mod qwen38;
pub mod sortformer;
pub mod vibevoice;
pub mod voxtral;
pub mod whisper;
