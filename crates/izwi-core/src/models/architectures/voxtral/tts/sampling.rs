//! Runtime generation parameters for Voxtral TTS.

use crate::catalog::ModelVariant;

use super::config::{DEFAULT_CFG_ALPHA, DEFAULT_N_DECODING_STEPS};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoxtralTtsGenerationParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub cfg_alpha: f32,
    pub n_decoding_steps: usize,
    pub max_frames: usize,
}

impl Default for VoxtralTtsGenerationParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            cfg_alpha: DEFAULT_CFG_ALPHA,
            n_decoding_steps: DEFAULT_N_DECODING_STEPS,
            max_frames: ModelVariant::VOXTRAL_TTS_MAX_OUTPUT_FRAMES,
        }
    }
}

impl VoxtralTtsGenerationParams {
    pub fn from_generation_config(config: &crate::runtime::GenerationConfig) -> Self {
        let opts = &config.options;
        Self {
            temperature: opts.temperature.max(0.0),
            top_p: opts.top_p.clamp(0.0, 1.0),
            top_k: opts.top_k,
            cfg_alpha: DEFAULT_CFG_ALPHA,
            n_decoding_steps: DEFAULT_N_DECODING_STEPS,
            max_frames: if opts.max_tokens == 0 {
                ModelVariant::VOXTRAL_TTS_MAX_OUTPUT_FRAMES
            } else {
                opts.max_tokens
                    .clamp(1, ModelVariant::VOXTRAL_TTS_MAX_OUTPUT_FRAMES)
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::GenerationConfig;

    #[test]
    fn defaults_match_voxtral_generation_contract() {
        let params = VoxtralTtsGenerationParams::default();
        assert_eq!(params.temperature, 0.0);
        assert_eq!(params.top_p, 1.0);
        assert_eq!(params.cfg_alpha, 1.2);
        assert_eq!(params.n_decoding_steps, 7);
        assert_eq!(
            params.max_frames,
            ModelVariant::VOXTRAL_TTS_MAX_OUTPUT_FRAMES
        );
    }

    #[test]
    fn converts_runtime_generation_config() {
        let mut config = GenerationConfig::default();
        config.options.temperature = 0.7;
        config.options.top_p = 0.8;
        config.options.top_k = 40;
        config.options.max_tokens = ModelVariant::VOXTRAL_TTS_MAX_OUTPUT_FRAMES + 10;
        let params = VoxtralTtsGenerationParams::from_generation_config(&config);
        assert_eq!(params.temperature, 0.7);
        assert_eq!(params.top_p, 0.8);
        assert_eq!(params.top_k, 40);
        assert_eq!(
            params.max_frames,
            ModelVariant::VOXTRAL_TTS_MAX_OUTPUT_FRAMES
        );
    }
}
