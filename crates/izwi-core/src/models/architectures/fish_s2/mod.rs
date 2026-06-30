//! Fish Audio S2 Pro TTS architecture boundary.
//!
//! This module currently validates the public Hugging Face artifact layout and
//! config shape. Native synthesis still needs the Candle slow transformer,
//! fast audio decoder, and Fish codec loader before `generate_with_reference`
//! can produce audio.

use std::path::Path;

use crate::catalog::ModelVariant;
use crate::error::{Error, Result};

pub mod artifacts;
pub mod codec;
pub mod config;
pub mod contracts;
pub mod fast;
pub mod slow;
pub mod tokenizer;
pub mod weights;

pub use artifacts::FishS2ArtifactManifest;
pub use codec::{FishS2CodecArtifact, FishS2CodecSupport};
pub use config::{FishS2AudioDecoderConfig, FishS2Config, FishS2TextConfig};
pub use contracts::{
    build_semantic_allowed_mask, remap_fish_qwen3_omni_key, semantic_code_from_token_id,
    semantic_token_id, FishS2DacContract, FishS2PromptTensorShape,
};
pub use fast::{
    FishS2FastCache, FishS2FastConfig, FishS2FastDecoder, FishS2GeneratedFrame, FishS2Sampler,
};
pub use slow::{FishS2SlowCache, FishS2SlowConfig, FishS2SlowOutput, FishS2SlowTransformer};
pub use tokenizer::{
    FishS2ConditioningPrompt, FishS2PromptTokenizer, FishS2SpecialTokens, FishS2VqCodes,
};
pub use weights::{FishS2TensorSpec, FishS2WeightIndex, FishS2Weights};

#[derive(Debug, Clone)]
pub struct FishS2TtsModel {
    variant: ModelVariant,
    config: FishS2Config,
    artifacts: FishS2ArtifactManifest,
    codec: FishS2CodecArtifact,
}

#[derive(Debug, Clone)]
pub struct FishS2Reference {
    pub audio_samples: Vec<f32>,
    pub sample_rate: u32,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct FishS2GenerationParams {
    pub max_frames: usize,
    pub temperature: f32,
    pub top_p: f32,
}

#[derive(Debug, Clone)]
pub struct FishS2GenerationOutput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub frames_generated: usize,
}

impl FishS2TtsModel {
    pub fn load_metadata(model_dir: &Path, variant: ModelVariant) -> Result<Self> {
        if variant != ModelVariant::FishAudioS2Pro {
            return Err(Error::InvalidInput(format!(
                "Unsupported Fish S2 TTS variant: {variant}"
            )));
        }
        let config = FishS2Config::load(model_dir)?;
        let artifacts = FishS2ArtifactManifest::load(model_dir)?;
        let codec = FishS2CodecArtifact::load(model_dir)?;
        Ok(Self {
            variant,
            config,
            artifacts,
            codec,
        })
    }

    pub fn variant(&self) -> ModelVariant {
        self.variant
    }

    pub fn config(&self) -> &FishS2Config {
        &self.config
    }

    pub fn artifacts(&self) -> &FishS2ArtifactManifest {
        &self.artifacts
    }

    pub fn codec(&self) -> &FishS2CodecArtifact {
        &self.codec
    }

    pub fn generate_with_reference(
        &self,
        _text: &str,
        _reference: FishS2Reference,
        _params: FishS2GenerationParams,
    ) -> Result<FishS2GenerationOutput> {
        self.codec.ensure_native_supported()?;
        Err(Error::ModelLoadError(
            "Fish Audio S2 Pro native generation is not implemented yet".to_string(),
        ))
    }
}

impl Default for FishS2GenerationParams {
    fn default() -> Self {
        Self {
            max_frames: ModelVariant::FISH_S2_PRO_MAX_OUTPUT_FRAMES,
            temperature: 0.8,
            top_p: 0.8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_params_default_to_s2_frame_budget() {
        let params = FishS2GenerationParams::default();
        assert_eq!(
            params.max_frames,
            ModelVariant::FISH_S2_PRO_MAX_OUTPUT_FRAMES
        );
        assert_eq!(params.temperature, 0.8);
        assert_eq!(params.top_p, 0.8);
    }
}
