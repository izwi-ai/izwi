//! Voxtral neural codec decode contract helpers.

use std::ops::Range;

use crate::error::{Error, Result};

use super::acoustic::{strip_audio_token_offset, AudioCodeValue};
use super::config::VoxtralTtsConfig;

pub const VOXTRAL_CODEC_CHUNK_FRAMES: usize = 375;

#[derive(Debug, Clone, PartialEq)]
pub struct VoxtralCodecConfig {
    pub channels: usize,
    pub sample_rate: usize,
    pub frame_rate: f32,
    pub patch_size: usize,
    pub decoder_strides: Vec<usize>,
    pub semantic_dim: usize,
    pub acoustic_dim: usize,
    pub latent_dim: usize,
}

impl VoxtralCodecConfig {
    pub fn from_config(config: &VoxtralTtsConfig) -> Result<Self> {
        let tokenizer = &config.multimodal.audio_tokenizer_args;
        Ok(Self {
            channels: tokenizer.channels,
            sample_rate: tokenizer.sampling_rate,
            frame_rate: config.frame_rate(),
            patch_size: tokenizer.pretransform_patch_size,
            decoder_strides: config.decoder_conv_strides()?,
            semantic_dim: tokenizer.semantic_dim,
            acoustic_dim: tokenizer.acoustic_dim,
            latent_dim: tokenizer.semantic_dim + tokenizer.acoustic_dim,
        })
    }

    pub fn downsample_factor(&self) -> Result<usize> {
        let stride_product = self
            .decoder_strides
            .iter()
            .try_fold(1usize, |acc, stride| {
                acc.checked_mul(*stride).ok_or_else(|| {
                    Error::ConfigError("Voxtral codec stride product overflowed".to_string())
                })
            })?;
        self.patch_size.checked_mul(stride_product).ok_or_else(|| {
            Error::ConfigError("Voxtral codec downsample factor overflowed".to_string())
        })
    }

    pub fn samples_for_frames(&self, frames: usize) -> Result<usize> {
        self.downsample_factor()?
            .checked_mul(frames)
            .ok_or_else(|| Error::AudioError("Voxtral codec sample count overflowed".to_string()))
    }

    pub fn frame_count_for_samples_ceil(&self, samples: usize) -> Result<usize> {
        let factor = self.downsample_factor()?;
        Ok((samples + factor - 1) / factor)
    }

    pub fn chunk_ranges(&self, frames: usize) -> Vec<Range<usize>> {
        chunk_ranges(frames, VOXTRAL_CODEC_CHUNK_FRAMES)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxtralCodecTimeline {
    pub shifted_codebooks: Vec<Vec<u32>>,
}

impl VoxtralCodecTimeline {
    pub fn new(shifted_codebooks: Vec<Vec<u32>>) -> Result<Self> {
        if shifted_codebooks.is_empty() {
            return Err(Error::AudioError(
                "Voxtral codec timeline must contain at least one codebook".to_string(),
            ));
        }
        let frames = shifted_codebooks[0].len();
        if shifted_codebooks
            .iter()
            .any(|codebook| codebook.len() != frames)
        {
            return Err(Error::AudioError(
                "Voxtral codec codebooks must all have the same frame length".to_string(),
            ));
        }
        Ok(Self { shifted_codebooks })
    }

    pub fn codebook_count(&self) -> usize {
        self.shifted_codebooks.len()
    }

    pub fn generated_frame_count(&self) -> usize {
        self.shifted_codebooks.first().map(Vec::len).unwrap_or(0)
    }

    pub fn audible_frame_count(&self) -> usize {
        self.shifted_codebooks
            .first()
            .and_then(|semantic| {
                semantic
                    .iter()
                    .position(|token| strip_audio_token_offset(*token) == AudioCodeValue::End)
            })
            .unwrap_or_else(|| self.generated_frame_count())
    }

    pub fn trim_at_end_audio(&self) -> Self {
        let frames = self.audible_frame_count();
        let shifted_codebooks = self
            .shifted_codebooks
            .iter()
            .map(|codebook| codebook[..frames].to_vec())
            .collect();
        Self { shifted_codebooks }
    }

    pub fn unshifted_codebooks(&self) -> Vec<Vec<Option<u32>>> {
        self.shifted_codebooks
            .iter()
            .map(|codebook| {
                codebook
                    .iter()
                    .map(|token| match strip_audio_token_offset(*token) {
                        AudioCodeValue::Code(code) => Some(code),
                        AudioCodeValue::Empty | AudioCodeValue::End => None,
                    })
                    .collect()
            })
            .collect()
    }
}

pub fn chunk_ranges(frames: usize, chunk_frames: usize) -> Vec<Range<usize>> {
    if frames == 0 {
        return Vec::new();
    }
    let chunk_frames = chunk_frames.max(1);
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < frames {
        let end = (start + chunk_frames).min(frames);
        ranges.push(start..end);
        start = end;
    }
    ranges
}

pub fn trim_samples_to_frames(
    samples: &mut Vec<f32>,
    frames: usize,
    config: &VoxtralCodecConfig,
) -> Result<()> {
    let expected = config.samples_for_frames(frames)?;
    if samples.len() > expected {
        samples.truncate(expected);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::architectures::voxtral::tts::acoustic::{
        apply_audio_token_offset, AudioSpecialToken,
    };
    use crate::models::architectures::voxtral::tts::config::{fixture_json, VoxtralTtsConfig};

    #[test]
    fn derives_codec_shape_from_params() {
        let config = VoxtralTtsConfig::from_json_str(fixture_json()).unwrap();
        let codec = VoxtralCodecConfig::from_config(&config).unwrap();
        assert_eq!(codec.sample_rate, 24_000);
        assert_eq!(codec.patch_size, 240);
        assert_eq!(codec.decoder_strides, vec![1, 2, 2, 2]);
        assert_eq!(codec.downsample_factor().unwrap(), 1920);
        assert_eq!(codec.samples_for_frames(12).unwrap(), 23_040);
        assert_eq!(codec.frame_count_for_samples_ceil(1921).unwrap(), 2);
        assert_eq!(codec.latent_dim, 292);
    }

    #[test]
    fn cuts_timeline_at_end_audio_on_semantic_codebook() {
        let semantic = vec![
            apply_audio_token_offset(3),
            apply_audio_token_offset(4),
            AudioSpecialToken::End.id(),
            apply_audio_token_offset(5),
        ];
        let acoustic = vec![apply_audio_token_offset(1); 4];
        let timeline = VoxtralCodecTimeline::new(vec![semantic, acoustic]).unwrap();
        assert_eq!(timeline.generated_frame_count(), 4);
        assert_eq!(timeline.audible_frame_count(), 2);
        assert_eq!(timeline.trim_at_end_audio().generated_frame_count(), 2);
    }

    #[test]
    fn chunks_like_vllm_omni_decoder_helper() {
        let ranges = chunk_ranges(751, VOXTRAL_CODEC_CHUNK_FRAMES);
        assert_eq!(ranges, vec![0..375, 375..750, 750..751]);
    }

    #[test]
    fn unshifts_special_audio_tokens_to_none() {
        let timeline = VoxtralCodecTimeline::new(vec![vec![0, 1, 2, 8]]).unwrap();
        assert_eq!(
            timeline.unshifted_codebooks(),
            vec![vec![None, None, Some(0), Some(6)]]
        );
    }
}
