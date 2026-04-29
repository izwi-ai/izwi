//! CUDA support contract for model variants.
//!
//! This is intentionally descriptive in its first pass: it records the current
//! execution surface without changing backend routing or model loading.

use serde::{Deserialize, Serialize};

use super::{ModelFamily, ModelVariant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CudaSupportLevel {
    NativeCuda,
    CandleCudaGeneric,
    CpuOnly,
    Disabled,
    Unknown,
}

impl CudaSupportLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NativeCuda => "native_cuda",
            Self::CandleCudaGeneric => "candle_cuda_generic",
            Self::CpuOnly => "cpu_only",
            Self::Disabled => "disabled",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CudaSupportInfo {
    pub level: CudaSupportLevel,
    pub reason: &'static str,
}

impl CudaSupportInfo {
    pub const fn new(level: CudaSupportLevel, reason: &'static str) -> Self {
        Self { level, reason }
    }
}

impl ModelVariant {
    pub fn cuda_support(&self) -> CudaSupportInfo {
        if !self.is_enabled() {
            return CudaSupportInfo::new(
                CudaSupportLevel::Disabled,
                "variant is disabled in the application catalog",
            );
        }

        match self.family() {
            ModelFamily::Tokenizer => CudaSupportInfo::new(
                CudaSupportLevel::CpuOnly,
                "tokenizer-only artifact does not run an inference backend",
            ),
            ModelFamily::SortformerDiarization => CudaSupportInfo::new(
                CudaSupportLevel::CpuOnly,
                "Sortformer currently hardcodes CPU tensors in its loader",
            ),
            ModelFamily::Qwen3Tts
            | ModelFamily::KokoroTts
            | ModelFamily::ParakeetAsr
            | ModelFamily::WhisperAsr
            | ModelFamily::Qwen3Asr
            | ModelFamily::Qwen3Chat
            | ModelFamily::Qwen35Chat
            | ModelFamily::Lfm2Chat
            | ModelFamily::Lfm25Audio
            | ModelFamily::Gemma3Chat
            | ModelFamily::Qwen3ForcedAligner
            | ModelFamily::Voxtral => CudaSupportInfo::new(
                CudaSupportLevel::CandleCudaGeneric,
                "model can be loaded on a CUDA Candle device but still needs CUDA-specific validation",
            ),
        }
    }

    pub fn cuda_support_level(&self) -> CudaSupportLevel {
        self.cuda_support().level
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_support_inventory_covers_every_variant() {
        for variant in ModelVariant::all() {
            let info = variant.cuda_support();
            assert!(
                !info.reason.trim().is_empty(),
                "{variant} must include a CUDA support reason"
            );
        }
    }

    #[test]
    fn disabled_variants_are_reported_as_disabled() {
        for variant in ModelVariant::all()
            .iter()
            .copied()
            .filter(|variant| !variant.is_enabled())
        {
            assert_eq!(
                variant.cuda_support_level(),
                CudaSupportLevel::Disabled,
                "{variant} should be marked disabled for CUDA support"
            );
        }
    }

    #[test]
    fn known_cpu_only_families_are_explicit() {
        assert_eq!(
            ModelVariant::DiarStreamingSortformer4SpkV21.cuda_support_level(),
            CudaSupportLevel::CpuOnly
        );
        assert_eq!(
            ModelVariant::Qwen3TtsTokenizer12Hz.cuda_support_level(),
            CudaSupportLevel::CpuOnly
        );
    }

    #[test]
    fn enabled_model_families_are_not_unknown() {
        for variant in ModelVariant::all()
            .iter()
            .copied()
            .filter(ModelVariant::is_enabled)
        {
            assert_ne!(
                variant.cuda_support_level(),
                CudaSupportLevel::Unknown,
                "{variant} should have an explicit CUDA support level"
            );
        }
    }
}
