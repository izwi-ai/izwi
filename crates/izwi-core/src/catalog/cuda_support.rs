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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CudaSupportInfo {
    pub level: CudaSupportLevel,
    pub reason: &'static str,
}

impl CudaSupportInfo {
    pub const fn new(level: CudaSupportLevel, reason: &'static str) -> Self {
        Self { level, reason }
    }
}

impl Default for CudaSupportInfo {
    fn default() -> Self {
        Self::new(
            CudaSupportLevel::Unknown,
            "CUDA support was not recorded in serialized model metadata",
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CudaQuantizationSupportLevel {
    Dense,
    CandleQuantizedGeneric,
    DenseDequantizedFallback,
    CpuOnly,
    Disabled,
    Unknown,
}

impl CudaQuantizationSupportLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::CandleQuantizedGeneric => "candle_quantized_generic",
            Self::DenseDequantizedFallback => "dense_dequantized_fallback",
            Self::CpuOnly => "cpu_only",
            Self::Disabled => "disabled",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CudaQuantizationInfo {
    pub level: CudaQuantizationSupportLevel,
    pub reason: &'static str,
}

impl CudaQuantizationInfo {
    pub const fn new(level: CudaQuantizationSupportLevel, reason: &'static str) -> Self {
        Self { level, reason }
    }

    pub fn is_allowed_for_cuda(self) -> bool {
        !matches!(
            self.level,
            CudaQuantizationSupportLevel::CpuOnly
                | CudaQuantizationSupportLevel::Disabled
                | CudaQuantizationSupportLevel::Unknown
        )
    }

    pub fn uses_dense_dequantized_fallback(self) -> bool {
        matches!(
            self.level,
            CudaQuantizationSupportLevel::DenseDequantizedFallback
        )
    }
}

impl Default for CudaQuantizationInfo {
    fn default() -> Self {
        Self::new(
            CudaQuantizationSupportLevel::Unknown,
            "CUDA quantization support was not recorded in serialized model metadata",
        )
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
                CudaSupportLevel::CandleCudaGeneric,
                "Sortformer uses Candle CUDA tensor kernels for inference when selected; preprocessing/postprocessing remain host-side orchestration",
            ),
            ModelFamily::Qwen3Tts
            | ModelFamily::KokoroTts
            | ModelFamily::VoxtralTts
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
                "model uses Candle CUDA tensor kernels, with CUDA-only Candle FlashAttention fast paths when shape and dtype support them",
            ),
        }
    }

    pub fn cuda_support_level(&self) -> CudaSupportLevel {
        self.cuda_support().level
    }

    pub fn cuda_quantization(&self) -> CudaQuantizationInfo {
        if !self.is_enabled() {
            return CudaQuantizationInfo::new(
                CudaQuantizationSupportLevel::Disabled,
                "variant is disabled in the application catalog",
            );
        }

        match self.family() {
            ModelFamily::Tokenizer => CudaQuantizationInfo::new(
                CudaQuantizationSupportLevel::CpuOnly,
                "tokenizer-only artifact does not run quantized CUDA inference",
            ),
            ModelFamily::SortformerDiarization => CudaQuantizationInfo::new(
                CudaQuantizationSupportLevel::Dense,
                "Sortformer checkpoint is dense F32 when loaded on CUDA",
            ),
            _ if self.is_qwen_chat_gguf()
                || self.is_qwen35_chat_gguf()
                || self.is_lfm2_chat_gguf() =>
            {
                CudaQuantizationInfo::new(
                    CudaQuantizationSupportLevel::CandleQuantizedGeneric,
                    "GGUF text model uses Candle quantized weights on the selected device",
                )
            }
            _ if self.is_qwen_asr_gguf() || self.is_lfm25_audio_gguf() => {
                CudaQuantizationInfo::new(
                    CudaQuantizationSupportLevel::DenseDequantizedFallback,
                    "GGUF speech/audio bundle is loaded through dense VarBuilder paths",
                )
            }
            _ if self.is_quantized() => CudaQuantizationInfo::new(
                CudaQuantizationSupportLevel::DenseDequantizedFallback,
                "quantized safetensors are dequantized into dense tensors before CUDA execution",
            ),
            _ => CudaQuantizationInfo::new(
                CudaQuantizationSupportLevel::Dense,
                "dense checkpoint uses the selected CUDA dtype policy",
            ),
        }
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

    #[test]
    fn enabled_inference_families_report_candle_cuda_kernel_coverage() {
        for variant in ModelVariant::all()
            .iter()
            .copied()
            .filter(ModelVariant::is_enabled)
            .filter(|variant| variant.family() != ModelFamily::Tokenizer)
        {
            let info = variant.cuda_support();
            assert_eq!(
                info.level,
                CudaSupportLevel::CandleCudaGeneric,
                "{variant} should use the Candle CUDA support class"
            );
            assert!(
                info.reason.contains("Candle CUDA"),
                "{variant} CUDA reason should name Candle CUDA coverage: {}",
                info.reason
            );
        }
    }

    #[test]
    fn cuda_quantization_inventory_covers_every_variant() {
        for variant in ModelVariant::all() {
            let info = variant.cuda_quantization();
            assert!(
                !info.reason.trim().is_empty(),
                "{variant} must include a CUDA quantization reason"
            );
        }
    }

    #[test]
    fn cuda_quantization_marks_dequantized_and_candle_paths() {
        assert_eq!(
            ModelVariant::Qwen34BGguf.cuda_quantization().level,
            CudaQuantizationSupportLevel::CandleQuantizedGeneric
        );
        assert_eq!(
            ModelVariant::Qwen3Asr06BGguf.cuda_quantization().level,
            CudaQuantizationSupportLevel::DenseDequantizedFallback
        );
        assert_eq!(
            ModelVariant::Qwen3Tts12Hz06BBase4Bit
                .cuda_quantization()
                .level,
            CudaQuantizationSupportLevel::DenseDequantizedFallback
        );
    }

    #[test]
    fn cuda_quantization_policy_distinguishes_allowed_and_fallback_modes() {
        let dense = ModelVariant::WhisperLargeV3Turbo.cuda_quantization();
        assert!(dense.is_allowed_for_cuda());
        assert!(!dense.uses_dense_dequantized_fallback());

        let dequant = ModelVariant::Qwen3Tts12Hz06BBase4Bit.cuda_quantization();
        assert!(dequant.is_allowed_for_cuda());
        assert!(dequant.uses_dense_dequantized_fallback());

        let tokenizer = ModelVariant::Qwen3TtsTokenizer12Hz.cuda_quantization();
        assert!(!tokenizer.is_allowed_for_cuda());
    }
}
