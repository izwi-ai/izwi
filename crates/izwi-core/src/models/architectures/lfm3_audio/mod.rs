//! LiquidAI LFM3-Audio-2B: speech recognition, speech synthesis and voice
//! chat with one model.
//!
//! The language model is the LFM3 trunk (`lfm3`) on the LFM2 backbone. The
//! causal speech encoder runs on the LFM2 backbone too, under its own tensor
//! names in the mmproj GGUF, and so does LiquidCodec's decoder trunk; the
//! MaskGIT depth decoder and the FSQ/ISTFT stages are this module's own.

pub(crate) mod codec;
pub(crate) mod config;
pub(crate) mod depth;
pub(crate) mod encoder;
pub(crate) mod model;
pub(crate) mod physical;
pub(crate) mod preprocessor;
pub mod prompts;
pub(crate) mod retained;
pub(crate) mod sampling;

pub use model::{lfm3_audio_tts_auto_max_frames, Lfm3AudioModel, Lfm3AudioOutput};
pub use prompts::{
    Lfm3AudioVoice, LFM3_AUDIO_ASR_LANGUAGES, LFM3_AUDIO_TTS_LOCALES, LFM3_AUDIO_VOICES,
};
