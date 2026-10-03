//! Audio processing utilities for TTS output

mod codec;
mod encoder;
mod inspection;
mod preprocessing;
mod resampling;
mod streaming;

pub use codec::{AudioCodec, CodecConfig};
pub use encoder::{AudioEncoder, AudioFormat};
pub use inspection::{
    decode_and_inspect_audio_bytes, decode_and_inspect_audio_bytes_canonical,
    decode_audio_bytes_to_mono, inspect_audio_bytes, inspect_audio_bytes_canonical,
    AudioInspection, AudioSourceMetadata, DecodedAudio,
};
pub use preprocessing::{MelConfig, MelNorm, MelScale, MelSpectrogram};
pub use resampling::{resample_mono_high_quality, target_sample_count};
pub(crate) use resampling::{align_resampled_length, HighQualityResampler};
pub use streaming::{AudioChunkBuffer, StreamingConfig};
