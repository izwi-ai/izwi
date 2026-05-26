//! Runtime generation parameters for Voxtral TTS.

use crate::catalog::ModelVariant;

use super::config::{DEFAULT_CFG_ALPHA, DEFAULT_N_DECODING_STEPS};

pub const VOXTRAL_TTS_AUTO_MIN_OUTPUT_FRAMES: usize = 12;
pub const VOXTRAL_TTS_AUTO_MAX_OUTPUT_FRAMES: usize = 96;
const VOXTRAL_TTS_WORDS_PER_SECOND: f32 = 2.5;
const VOXTRAL_TTS_AUTO_PADDING_SECONDS: f32 = 0.6;
const VOXTRAL_TTS_AUTO_PADDING_FRAMES: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoxtralTtsGenerationParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub cfg_alpha: f32,
    pub n_decoding_steps: usize,
    pub max_frames: usize,
    pub auto_frame_budget: bool,
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
            auto_frame_budget: false,
        }
    }
}

impl VoxtralTtsGenerationParams {
    pub fn from_generation_config(config: &crate::runtime::GenerationConfig) -> Self {
        Self::from_generation_config_with_auto_frames(config, None)
    }

    pub fn from_generation_config_for_text(
        config: &crate::runtime::GenerationConfig,
        text: &str,
    ) -> Self {
        Self::from_generation_config_with_auto_frames(
            config,
            Some(voxtral_tts_auto_max_frames_for_text(text)),
        )
    }

    fn from_generation_config_with_auto_frames(
        config: &crate::runtime::GenerationConfig,
        auto_max_frames: Option<usize>,
    ) -> Self {
        let opts = &config.options;
        let auto_frame_budget = opts.max_tokens == 0;
        let max_frames = if auto_frame_budget {
            auto_max_frames.unwrap_or(ModelVariant::VOXTRAL_TTS_MAX_OUTPUT_FRAMES)
        } else {
            opts.max_tokens
                .clamp(1, ModelVariant::VOXTRAL_TTS_MAX_OUTPUT_FRAMES)
        };
        Self {
            temperature: opts.temperature.max(0.0),
            top_p: opts.top_p.clamp(0.0, 1.0),
            top_k: opts.top_k,
            cfg_alpha: DEFAULT_CFG_ALPHA,
            n_decoding_steps: DEFAULT_N_DECODING_STEPS,
            max_frames,
            auto_frame_budget,
        }
    }
}

pub fn voxtral_tts_auto_max_frames_for_text(text: &str) -> usize {
    let word_count = text
        .split_whitespace()
        .filter(|word| !word.is_empty())
        .count();
    let char_count = text.chars().filter(|ch| !ch.is_whitespace()).count();
    let effective_words = if word_count > 0 {
        word_count as f32
    } else if char_count > 0 {
        (char_count as f32 / 4.0).ceil()
    } else {
        1.0
    };
    let estimated_secs =
        VOXTRAL_TTS_AUTO_PADDING_SECONDS + effective_words / VOXTRAL_TTS_WORDS_PER_SECOND;
    let estimated_frames = (estimated_secs * ModelVariant::VOXTRAL_TTS_FRAME_RATE_HZ).ceil()
        as usize
        + VOXTRAL_TTS_AUTO_PADDING_FRAMES;
    estimated_frames.clamp(
        VOXTRAL_TTS_AUTO_MIN_OUTPUT_FRAMES,
        VOXTRAL_TTS_AUTO_MAX_OUTPUT_FRAMES,
    )
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
        assert!(!params.auto_frame_budget);
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
        assert!(!params.auto_frame_budget);
    }

    #[test]
    fn auto_frame_budget_scales_with_text_and_caps() {
        let mut config = GenerationConfig::default();
        config.options.max_tokens = 0;

        let short = VoxtralTtsGenerationParams::from_generation_config_for_text(
            &config,
            "The costs split cleanly into three buckets",
        );
        assert!(short.auto_frame_budget);
        assert_eq!(short.max_frames, 47);

        let long = VoxtralTtsGenerationParams::from_generation_config_for_text(
            &config,
            &"word ".repeat(500),
        );
        assert_eq!(long.max_frames, VOXTRAL_TTS_AUTO_MAX_OUTPUT_FRAMES);
    }
}
