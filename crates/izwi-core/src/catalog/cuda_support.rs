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

/// Runtime maturity of the CUDA execution path advertised for a model.
///
/// This is intentionally independent from [`CudaSupportLevel`]. The legacy
/// level describes what kind of implementation exists, while this status says
/// how far that implementation has progressed through validation and rollout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CudaExecutionStatus {
    Unsupported,
    /// Portable Candle CUDA tensor graph; this is a provider class, not proof
    /// that CUDA compilation or execution was observed on the current host.
    Portable,
    /// An optimized provider is source-eligible but lacks runtime validation.
    EligibleUnverified,
    CandleOptimized,
    CustomOptimized,
    Certified,
}

impl CudaExecutionStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unsupported => "unsupported",
            Self::Portable => "portable",
            Self::EligibleUnverified => "eligible_unverified",
            Self::CandleOptimized => "candle_optimized",
            Self::CustomOptimized => "custom_optimized",
            Self::Certified => "certified",
        }
    }

    pub const fn is_optimized(self) -> bool {
        matches!(
            self,
            Self::CandleOptimized | Self::CustomOptimized | Self::Certified
        )
    }
}

/// Highest CUDA evidence state actually observed for a support claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CudaEvidenceLevel {
    NotObserved,
    SourceReviewed,
    PortableVerified,
    CudaCompiled,
    CudaRuntimeValidated,
    CudaPerformanceCertified,
}

impl CudaEvidenceLevel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NotObserved => "not_observed",
            Self::SourceReviewed => "source_reviewed",
            Self::PortableVerified => "portable_verified",
            Self::CudaCompiled => "cuda_compiled",
            Self::CudaRuntimeValidated => "cuda_runtime_validated",
            Self::CudaPerformanceCertified => "cuda_performance_certified",
        }
    }

    pub const fn proves_cuda_runtime(self) -> bool {
        matches!(
            self,
            Self::CudaRuntimeValidated | Self::CudaPerformanceCertified
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CudaSupportInfo {
    /// Legacy implementation classification retained for API compatibility.
    pub level: CudaSupportLevel,
    pub execution_status: CudaExecutionStatus,
    pub evidence: CudaEvidenceLevel,
    pub reason: &'static str,
}

impl CudaSupportInfo {
    pub const fn new(level: CudaSupportLevel, reason: &'static str) -> Self {
        let (execution_status, evidence) = match level {
            CudaSupportLevel::NativeCuda | CudaSupportLevel::CandleCudaGeneric => (
                CudaExecutionStatus::EligibleUnverified,
                CudaEvidenceLevel::SourceReviewed,
            ),
            CudaSupportLevel::CpuOnly | CudaSupportLevel::Disabled => (
                CudaExecutionStatus::Unsupported,
                CudaEvidenceLevel::SourceReviewed,
            ),
            CudaSupportLevel::Unknown => (
                CudaExecutionStatus::Unsupported,
                CudaEvidenceLevel::NotObserved,
            ),
        };
        Self {
            level,
            execution_status,
            evidence,
            reason,
        }
    }

    pub const fn try_with_evidence(
        level: CudaSupportLevel,
        execution_status: CudaExecutionStatus,
        evidence: CudaEvidenceLevel,
        reason: &'static str,
    ) -> Option<Self> {
        let info = Self {
            level,
            execution_status,
            evidence,
            reason,
        };
        if info.evidence_is_sufficient() {
            Some(info)
        } else {
            None
        }
    }

    pub const fn evidence_is_sufficient(self) -> bool {
        match self.execution_status {
            CudaExecutionStatus::Unsupported => true,
            CudaExecutionStatus::Portable | CudaExecutionStatus::EligibleUnverified => {
                !matches!(self.evidence, CudaEvidenceLevel::NotObserved)
            }
            CudaExecutionStatus::CandleOptimized | CudaExecutionStatus::CustomOptimized => {
                self.evidence.proves_cuda_runtime()
            }
            CudaExecutionStatus::Certified => {
                matches!(self.evidence, CudaEvidenceLevel::CudaPerformanceCertified)
            }
        }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CudaOperatorKind {
    DenseGemm,
    QuantizedGemm,
    Convolution,
    Attention,
    PagedAttention,
    Rope,
    Normalization,
    Sampling,
    State,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CudaProviderClass {
    NotApplicable,
    HostOrchestration,
    CandleTensor,
    CandleCudnnEligible,
    CandleFlashAttentionEligible,
    IzwiCudaEligible,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CudaOperatorCapability {
    pub operator: CudaOperatorKind,
    pub provider: CudaProviderClass,
    pub evidence: CudaEvidenceLevel,
    pub reason: &'static str,
}

impl CudaOperatorCapability {
    const fn source_reviewed(
        operator: CudaOperatorKind,
        provider: CudaProviderClass,
        reason: &'static str,
    ) -> Self {
        Self {
            operator,
            provider,
            evidence: CudaEvidenceLevel::SourceReviewed,
            reason,
        }
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
    pub fn cuda_operator_capabilities(&self) -> [CudaOperatorCapability; 9] {
        use CudaOperatorKind as Operator;
        use CudaProviderClass as Provider;

        if !self.is_enabled() {
            return std::array::from_fn(|index| {
                let operator = CUDA_OPERATOR_ORDER[index];
                CudaOperatorCapability::source_reviewed(
                    operator,
                    Provider::NotApplicable,
                    "variant is disabled in the application catalog",
                )
            });
        }

        let family = self.family();
        let convolution = match family {
            ModelFamily::Qwen3Tts
            | ModelFamily::KokoroTts
            | ModelFamily::VoxtralTts
            | ModelFamily::VibeVoiceTts
            | ModelFamily::FishS2Tts
            | ModelFamily::ParakeetAsr
            | ModelFamily::WhisperAsr
            | ModelFamily::Qwen3Asr
            | ModelFamily::VibeVoiceAsr
            | ModelFamily::NemotronAsr
            | ModelFamily::GraniteSpeechAsr
            | ModelFamily::SortformerDiarization
            | ModelFamily::Lfm2Chat
            | ModelFamily::Lfm25Audio
            | ModelFamily::Qwen3ForcedAligner
            | ModelFamily::Voxtral
            | ModelFamily::Tokenizer => CudaOperatorCapability::source_reviewed(
                Operator::Convolution,
                Provider::CandleCudnnEligible,
                "convolution uses Candle tensor operators and is eligible for cuDNN only when build, layout, dtype, and grouping constraints match",
            ),
            ModelFamily::Qwen35Chat => CudaOperatorCapability::source_reviewed(
                Operator::Convolution,
                Provider::IzwiCudaEligible,
                "Qwen3.5 recurrent blocks have an existing Izwi CUDA causal-convolution provider with a Candle fallback",
            ),
            ModelFamily::Qwen3Chat | ModelFamily::Gemma3Chat => {
                CudaOperatorCapability::source_reviewed(
                    Operator::Convolution,
                    Provider::NotApplicable,
                    "text decoder graph has no convolutional hot path",
                )
            }
        };
        let attention_provider = match family {
            ModelFamily::KokoroTts
            | ModelFamily::ParakeetAsr
            | ModelFamily::NemotronAsr
            | ModelFamily::SortformerDiarization
            | ModelFamily::Lfm25Audio => Provider::CandleTensor,
            _ => Provider::CandleFlashAttentionEligible,
        };
        let paged_provider = match family {
            ModelFamily::Qwen3Chat
            | ModelFamily::Qwen35Chat
            | ModelFamily::Lfm2Chat
            | ModelFamily::Gemma3Chat
            | ModelFamily::Qwen3Tts
            | ModelFamily::VoxtralTts
            | ModelFamily::VibeVoiceTts
            | ModelFamily::FishS2Tts
            | ModelFamily::WhisperAsr
            | ModelFamily::Qwen3Asr
            | ModelFamily::VibeVoiceAsr
            | ModelFamily::NemotronAsr
            | ModelFamily::GraniteSpeechAsr
            | ModelFamily::Lfm25Audio
            | ModelFamily::Qwen3ForcedAligner
            | ModelFamily::Voxtral => Provider::IzwiCudaEligible,
            ModelFamily::KokoroTts
            | ModelFamily::ParakeetAsr
            | ModelFamily::SortformerDiarization
            | ModelFamily::Tokenizer => Provider::NotApplicable,
        };
        let rope_provider = match family {
            ModelFamily::KokoroTts
            | ModelFamily::ParakeetAsr
            | ModelFamily::NemotronAsr
            | ModelFamily::SortformerDiarization => Provider::NotApplicable,
            _ => Provider::CandleTensor,
        };
        let sampling_provider = match family {
            ModelFamily::KokoroTts
            | ModelFamily::SortformerDiarization
            | ModelFamily::Tokenizer => Provider::NotApplicable,
            _ => Provider::CandleTensor,
        };
        let quantization_provider = match self.cuda_quantization().level {
            CudaQuantizationSupportLevel::CandleQuantizedGeneric => Provider::CandleTensor,
            CudaQuantizationSupportLevel::DenseDequantizedFallback => Provider::HostOrchestration,
            CudaQuantizationSupportLevel::Dense => Provider::NotApplicable,
            CudaQuantizationSupportLevel::CpuOnly
            | CudaQuantizationSupportLevel::Disabled
            | CudaQuantizationSupportLevel::Unknown => Provider::NotApplicable,
        };

        [
            CudaOperatorCapability::source_reviewed(
                Operator::DenseGemm,
                Provider::CandleTensor,
                "dense projections use Candle matmul/linear dispatch on the selected device",
            ),
            CudaOperatorCapability::source_reviewed(
                Operator::QuantizedGemm,
                quantization_provider,
                "provider reflects the checkpoint loading and quantized projection path; it is not CUDA runtime proof",
            ),
            convolution,
            CudaOperatorCapability::source_reviewed(
                Operator::Attention,
                attention_provider,
                "FlashAttention eligibility remains conditional on explicit policy, build, device, dtype, shape, and mask semantics",
            ),
            CudaOperatorCapability::source_reviewed(
                Operator::PagedAttention,
                paged_provider,
                "iterative decoders use the shared physical paged-attention contract; other graphs mark this operator not applicable",
            ),
            CudaOperatorCapability::source_reviewed(
                Operator::Rope,
                rope_provider,
                "RoPE uses Candle tensor kernels where the architecture has rotary position encoding",
            ),
            CudaOperatorCapability::source_reviewed(
                Operator::Normalization,
                Provider::CandleTensor,
                "normalization has a Candle tensor reference path on the selected device",
            ),
            CudaOperatorCapability::source_reviewed(
                Operator::Sampling,
                sampling_provider,
                "autoregressive selection uses Candle reductions where implemented and explicit host orchestration otherwise",
            ),
            CudaOperatorCapability::source_reviewed(
                Operator::State,
                paged_provider,
                "state provider reflects load-sealed physical/invocation ownership, not observed CUDA execution",
            ),
        ]
    }

    pub fn cuda_support(&self) -> CudaSupportInfo {
        if !self.is_enabled() {
            return CudaSupportInfo::new(
                CudaSupportLevel::Disabled,
                "variant is disabled in the application catalog",
            );
        }

        match self.family() {
            ModelFamily::Tokenizer => CudaSupportInfo::new(
                CudaSupportLevel::CandleCudaGeneric,
                "Qwen3 TTS tokenizer is a neural speech codec whose encoder, RVQ, transformer, and decoder use Candle CUDA tensor kernels when selected",
            ),
            ModelFamily::SortformerDiarization => CudaSupportInfo::new(
                CudaSupportLevel::CandleCudaGeneric,
                "Sortformer uses Candle CUDA tensor kernels for inference when selected; preprocessing/postprocessing remain host-side orchestration",
            ),
            ModelFamily::VoxtralTts => CudaSupportInfo::new(
                CudaSupportLevel::CandleCudaGeneric,
                "Voxtral TTS uses dense Candle CUDA tensor kernels for the LM, flow-matching acoustic transformer, and codec; progressive streaming remains final-only until CUDA-only chunked decode is proven",
            ),
            ModelFamily::VibeVoiceTts => CudaSupportInfo::new(
                CudaSupportLevel::CandleCudaGeneric,
                "VibeVoice TTS uses dense Candle CUDA tensor kernels for the Qwen decoder, continuous tokenizers, and diffusion head; long-form generation is final-only until chunked decode is proven",
            ),
            ModelFamily::NemotronAsr => CudaSupportInfo::new(
                CudaSupportLevel::CandleCudaGeneric,
                "Nemotron ASR uses dense Candle CUDA tensor kernels once the native FastConformer-RNNT loader is selected; cache-aware streaming remains disabled until encoder cache state is proven",
            ),
            ModelFamily::GraniteSpeechAsr => CudaSupportInfo::new(
                CudaSupportLevel::CandleCudaGeneric,
                "Granite Speech ASR uses dense Candle CUDA tensor kernels for the audio encoder, Q-Former projector, and Granite decoder once the native loader is selected",
            ),
            ModelFamily::Qwen3Tts
            | ModelFamily::KokoroTts
            | ModelFamily::FishS2Tts
            | ModelFamily::ParakeetAsr
            | ModelFamily::WhisperAsr
            | ModelFamily::Qwen3Asr
            | ModelFamily::VibeVoiceAsr
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
                CudaQuantizationSupportLevel::Dense,
                "Qwen3 TTS tokenizer is a dense neural speech codec; CUDA dtype policy, not text-tokenizer orchestration, controls its execution",
            ),
            ModelFamily::SortformerDiarization => CudaQuantizationInfo::new(
                CudaQuantizationSupportLevel::Dense,
                "Sortformer checkpoint is dense F32 when loaded on CUDA",
            ),
            ModelFamily::VoxtralTts => CudaQuantizationInfo::new(
                CudaQuantizationSupportLevel::Dense,
                "Voxtral TTS checkpoint is dense safetensors; CUDA dtype policy, not quantization, controls memory/performance tradeoffs",
            ),
            ModelFamily::NemotronAsr => CudaQuantizationInfo::new(
                CudaQuantizationSupportLevel::Dense,
                "Nemotron ASR ships a dense .nemo checkpoint; CUDA dtype policy, not quantization, controls memory/performance tradeoffs",
            ),
            ModelFamily::GraniteSpeechAsr => CudaQuantizationInfo::new(
                CudaQuantizationSupportLevel::Dense,
                "Granite Speech ships dense BF16 safetensors; CUDA dtype policy, not quantization, controls memory/performance tradeoffs",
            ),
            ModelFamily::VibeVoiceTts | ModelFamily::VibeVoiceAsr => CudaQuantizationInfo::new(
                CudaQuantizationSupportLevel::Dense,
                "VibeVoice checkpoint is dense safetensors; CUDA dtype policy, not quantization, controls memory/performance tradeoffs",
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

const CUDA_OPERATOR_ORDER: [CudaOperatorKind; 9] = [
    CudaOperatorKind::DenseGemm,
    CudaOperatorKind::QuantizedGemm,
    CudaOperatorKind::Convolution,
    CudaOperatorKind::Attention,
    CudaOperatorKind::PagedAttention,
    CudaOperatorKind::Rope,
    CudaOperatorKind::Normalization,
    CudaOperatorKind::Sampling,
    CudaOperatorKind::State,
];

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
    fn cuda_operator_inventory_is_complete_and_source_reviewed() {
        use std::collections::HashSet;

        for variant in ModelVariant::all() {
            let capabilities = variant.cuda_operator_capabilities();
            assert_eq!(capabilities.len(), CUDA_OPERATOR_ORDER.len());
            assert_eq!(
                capabilities
                    .iter()
                    .map(|capability| capability.operator)
                    .collect::<HashSet<_>>()
                    .len(),
                CUDA_OPERATOR_ORDER.len(),
                "{variant} has duplicate CUDA operator records"
            );
            for capability in capabilities {
                assert_eq!(capability.evidence, CudaEvidenceLevel::SourceReviewed);
                assert!(!capability.reason.trim().is_empty());
            }
        }
    }

    #[test]
    fn neural_tokenizer_reports_codec_operator_coverage() {
        let capabilities = ModelVariant::Qwen3TtsTokenizer12Hz.cuda_operator_capabilities();
        let convolution = capabilities
            .iter()
            .find(|capability| capability.operator == CudaOperatorKind::Convolution)
            .expect("convolution capability");
        assert_eq!(convolution.provider, CudaProviderClass::CandleCudnnEligible);
        let sampling = capabilities
            .iter()
            .find(|capability| capability.operator == CudaOperatorKind::Sampling)
            .expect("sampling capability");
        assert_eq!(sampling.provider, CudaProviderClass::NotApplicable);
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
    fn neural_speech_tokenizer_is_cuda_eligible_but_unverified() {
        let support = ModelVariant::Qwen3TtsTokenizer12Hz.cuda_support();
        assert_eq!(support.level, CudaSupportLevel::CandleCudaGeneric);
        assert_eq!(
            support.execution_status,
            CudaExecutionStatus::EligibleUnverified
        );
        assert_eq!(support.evidence, CudaEvidenceLevel::SourceReviewed);
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
    fn enabled_inference_families_report_source_reviewed_unverified_cuda() {
        for variant in ModelVariant::all()
            .iter()
            .copied()
            .filter(ModelVariant::is_enabled)
        {
            let info = variant.cuda_support();
            assert_eq!(
                info.execution_status,
                CudaExecutionStatus::EligibleUnverified,
                "{variant} must not claim CUDA validation from source review"
            );
            assert_eq!(
                info.evidence,
                CudaEvidenceLevel::SourceReviewed,
                "{variant} should not claim unobserved CUDA runtime evidence"
            );
            assert!(
                info.reason.contains("Candle CUDA"),
                "{variant} CUDA reason should name Candle CUDA coverage: {}",
                info.reason
            );
        }
    }

    #[test]
    fn support_metadata_serializes_execution_and_evidence_independently() {
        let value = serde_json::to_value(ModelVariant::Qwen34BGguf.cuda_support())
            .expect("serialize CUDA support");

        assert_eq!(value["level"], "candle_cuda_generic");
        assert_eq!(value["execution_status"], "eligible_unverified");
        assert_eq!(value["evidence"], "source_reviewed");
    }

    #[test]
    fn default_support_metadata_does_not_invent_cuda_evidence() {
        let info = CudaSupportInfo::default();
        assert_eq!(info.execution_status, CudaExecutionStatus::Unsupported);
        assert_eq!(info.evidence, CudaEvidenceLevel::NotObserved);
        assert!(!info.evidence.proves_cuda_runtime());
    }

    #[test]
    fn optimized_status_requires_runtime_evidence_by_contract() {
        for status in [
            CudaExecutionStatus::CandleOptimized,
            CudaExecutionStatus::CustomOptimized,
            CudaExecutionStatus::Certified,
        ] {
            assert!(status.is_optimized());
        }
        assert!(!CudaExecutionStatus::EligibleUnverified.is_optimized());
        assert!(CudaEvidenceLevel::CudaRuntimeValidated.proves_cuda_runtime());
        assert!(CudaEvidenceLevel::CudaPerformanceCertified.proves_cuda_runtime());
        assert!(!CudaEvidenceLevel::CudaCompiled.proves_cuda_runtime());

        let invalid = CudaSupportInfo::try_with_evidence(
            CudaSupportLevel::NativeCuda,
            CudaExecutionStatus::CandleOptimized,
            CudaEvidenceLevel::CudaCompiled,
            "compile-only evidence",
        );
        assert!(invalid.is_none());

        let certified = CudaSupportInfo::try_with_evidence(
            CudaSupportLevel::NativeCuda,
            CudaExecutionStatus::Certified,
            CudaEvidenceLevel::CudaPerformanceCertified,
            "runtime and performance evidence",
        )
        .expect("performance evidence should permit a certified claim");
        assert!(certified.evidence_is_sufficient());
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
        assert_eq!(
            ModelVariant::Voxtral4BTts2603.cuda_quantization().level,
            CudaQuantizationSupportLevel::Dense
        );
        assert_eq!(
            ModelVariant::Nemotron35AsrStreaming06B
                .cuda_quantization()
                .level,
            CudaQuantizationSupportLevel::Dense
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
        assert!(tokenizer.is_allowed_for_cuda());
        assert_eq!(tokenizer.level, CudaQuantizationSupportLevel::Dense);
    }
}
