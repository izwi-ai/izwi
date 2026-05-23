//! Flow-matching acoustic-token shape helpers.

use crate::error::{Error, Result};

use super::config::{VoxtralTtsConfig, DEFAULT_CFG_ALPHA, DEFAULT_N_DECODING_STEPS};

pub const AUDIO_SPECIAL_TOKEN_COUNT: u32 = 2;
pub const ACOUSTIC_CODEBOOK_OFFSET: usize = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioSpecialToken {
    Empty,
    End,
}

impl AudioSpecialToken {
    pub fn id(self) -> u32 {
        match self {
            Self::Empty => 0,
            Self::End => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioCodeValue {
    Empty,
    End,
    Code(u32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcousticCodeFrame {
    pub semantic: u32,
    pub acoustic: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcousticGenerationConfig {
    pub semantic_codebook_size: usize,
    pub acoustic_codebook_size: usize,
    pub n_acoustic_codebooks: usize,
    pub num_codebooks: usize,
    pub n_decoding_steps: usize,
    pub cfg_alpha: f32,
    pub sigma: f32,
}

impl AcousticGenerationConfig {
    pub fn from_config(config: &VoxtralTtsConfig) -> Self {
        let transformer = &config.multimodal.audio_model_args.acoustic_transformer_args;
        Self {
            semantic_codebook_size: config.semantic_codebook_size(),
            acoustic_codebook_size: config.acoustic_codebook_size(),
            n_acoustic_codebooks: config.n_acoustic_codebooks(),
            num_codebooks: config.num_codebooks(),
            n_decoding_steps: transformer
                .n_decoding_steps
                .unwrap_or(DEFAULT_N_DECODING_STEPS),
            cfg_alpha: transformer.cfg_alpha.unwrap_or(DEFAULT_CFG_ALPHA),
            sigma: transformer.sigma,
        }
    }

    pub fn validate_frame(&self, frame: &AcousticCodeFrame) -> Result<()> {
        if frame.semantic as usize >= self.semantic_codebook_size {
            return Err(Error::InferenceError(format!(
                "Voxtral semantic code {} exceeds codebook size {}",
                frame.semantic, self.semantic_codebook_size
            )));
        }
        if frame.acoustic.len() != self.n_acoustic_codebooks {
            return Err(Error::InferenceError(format!(
                "Voxtral acoustic frame has {} codebooks, expected {}",
                frame.acoustic.len(),
                self.n_acoustic_codebooks
            )));
        }
        for (idx, code) in frame.acoustic.iter().enumerate() {
            if *code as usize >= self.acoustic_codebook_size {
                return Err(Error::InferenceError(format!(
                    "Voxtral acoustic codebook {idx} value {code} exceeds codebook size {}",
                    self.acoustic_codebook_size
                )));
            }
        }
        Ok(())
    }
}

impl AcousticCodeFrame {
    pub fn new(
        semantic: u32,
        acoustic: Vec<u32>,
        config: &AcousticGenerationConfig,
    ) -> Result<Self> {
        let frame = Self { semantic, acoustic };
        config.validate_frame(&frame)?;
        Ok(frame)
    }

    pub fn shifted_codes(&self) -> Vec<u32> {
        let mut codes = Vec::with_capacity(self.acoustic.len() + ACOUSTIC_CODEBOOK_OFFSET);
        codes.push(apply_audio_token_offset(self.semantic));
        codes.extend(
            self.acoustic
                .iter()
                .map(|code| apply_audio_token_offset(*code)),
        );
        codes
    }
}

pub fn apply_audio_token_offset(code: u32) -> u32 {
    code + AUDIO_SPECIAL_TOKEN_COUNT
}

pub fn strip_audio_token_offset(shifted: u32) -> AudioCodeValue {
    match shifted {
        0 => AudioCodeValue::Empty,
        1 => AudioCodeValue::End,
        value => AudioCodeValue::Code(value - AUDIO_SPECIAL_TOKEN_COUNT),
    }
}

pub fn fsq_unit_to_code(value: f32, codebook_size: usize) -> u32 {
    if codebook_size <= 1 {
        return 0;
    }
    let clamped = value.clamp(-1.0, 1.0);
    let scaled = (clamped + 1.0) * 0.5 * (codebook_size as f32 - 1.0);
    scaled.round() as u32
}

pub fn fsq_code_to_unit(code: u32, codebook_size: usize) -> f32 {
    if codebook_size <= 1 {
        return 0.0;
    }
    let clamped = code.min(codebook_size as u32 - 1) as f32;
    (clamped / (codebook_size as f32 - 1.0)) * 2.0 - 1.0
}

pub fn cfg_velocity_blend(conditional: f32, unconditional: f32, alpha: f32) -> f32 {
    alpha.mul_add(conditional, (1.0 - alpha) * unconditional)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::architectures::voxtral::tts::config::{fixture_json, VoxtralTtsConfig};

    #[test]
    fn derives_generation_config_from_params() {
        let config = VoxtralTtsConfig::from_json_str(fixture_json()).unwrap();
        let acoustic = AcousticGenerationConfig::from_config(&config);
        assert_eq!(acoustic.semantic_codebook_size, 8192);
        assert_eq!(acoustic.acoustic_codebook_size, 21);
        assert_eq!(acoustic.n_acoustic_codebooks, 36);
        assert_eq!(acoustic.num_codebooks, 37);
        assert_eq!(acoustic.n_decoding_steps, 7);
        assert_eq!(acoustic.cfg_alpha, 1.2);
    }

    #[test]
    fn applies_and_strips_vllm_omni_audio_token_offsets() {
        assert_eq!(apply_audio_token_offset(0), 2);
        assert_eq!(strip_audio_token_offset(0), AudioCodeValue::Empty);
        assert_eq!(strip_audio_token_offset(1), AudioCodeValue::End);
        assert_eq!(strip_audio_token_offset(8193), AudioCodeValue::Code(8191));
    }

    #[test]
    fn quantizes_fsq_unit_range_to_21_levels() {
        assert_eq!(fsq_unit_to_code(-2.0, 21), 0);
        assert_eq!(fsq_unit_to_code(-1.0, 21), 0);
        assert_eq!(fsq_unit_to_code(0.0, 21), 10);
        assert_eq!(fsq_unit_to_code(1.0, 21), 20);
        assert_eq!(fsq_unit_to_code(2.0, 21), 20);
        assert!((fsq_code_to_unit(10, 21) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn validates_code_frame_shapes() {
        let config = VoxtralTtsConfig::from_json_str(fixture_json()).unwrap();
        let acoustic = AcousticGenerationConfig::from_config(&config);
        let frame = AcousticCodeFrame::new(7, vec![3; 36], &acoustic).unwrap();
        assert_eq!(frame.shifted_codes().len(), 37);
        assert!(AcousticCodeFrame::new(8192, vec![3; 36], &acoustic).is_err());
        assert!(AcousticCodeFrame::new(7, vec![3; 35], &acoustic).is_err());
        assert!(AcousticCodeFrame::new(7, vec![21; 36], &acoustic).is_err());
    }

    #[test]
    fn blends_classifier_free_guidance_velocity() {
        assert!((cfg_velocity_blend(2.0, 0.5, 1.2) - 2.3).abs() < 1e-6);
    }
}
