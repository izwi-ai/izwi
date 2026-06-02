//! Audio processing utilities for TTS output

mod codec;
mod encoder;
mod inspection;
mod preprocessing;
mod resampling;
mod streaming;

pub use codec::{AudioCodec, CodecConfig};
pub use encoder::{AudioEncoder, AudioFormat};
pub use inspection::{AudioInspection, inspect_audio_bytes};
pub use preprocessing::{MelConfig, MelSpectrogram};
pub use resampling::{resample_mono_high_quality, target_sample_count};
pub use streaming::{AudioChunkBuffer, StreamingConfig};
