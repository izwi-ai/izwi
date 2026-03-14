//! Concrete model architecture implementations grouped by family.
//!
//! Adding a new model should follow this pattern:
//! 1. Create a new family folder (or extend an existing one).
//! 2. Export that family here.
//! 3. Wire loading in `crate::models::registry`.

pub mod gemma3;
pub mod kokoro;
pub mod lfm2;
pub mod lfm25_audio;
pub mod parakeet;
pub mod qwen3;
pub mod qwen35;
pub mod sortformer;
pub mod voxtral;
pub mod whisper;
