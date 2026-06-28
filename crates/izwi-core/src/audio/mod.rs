//! Audio processing utilities for TTS output

mod codec;
mod encoder;
mod inspection;
mod preprocessing;
mod resampling;
mod streaming;

pub use codec::{AudioCodec, CodecConfig};
pub use encoder::{AudioEncoder, AudioFormat};
pub use inspection::{decode_audio_bytes_to_mono, inspect_audio_bytes, AudioInspection};
pub use preprocessing::{MelConfig, MelNorm, MelScale, MelSpectrogram};
pub use resampling::{resample_mono_high_quality, target_sample_count};
pub use streaming::{AudioChunkBuffer, StreamingConfig};
