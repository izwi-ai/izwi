//! Model identities, families, metadata and backend support levels.

mod cuda_support;
mod metadata;
mod prefix_reuse;
mod variant;

pub use cuda_support::{
    CudaEvidenceLevel, CudaExecutionStatus, CudaOperatorCapability, CudaOperatorKind,
    CudaProviderClass, CudaQuantizationInfo, CudaQuantizationSupportLevel, CudaSupportInfo,
    CudaSupportLevel,
};
pub use metadata::{
    ChatModelCapabilities, ChatReasoningEffort, ModelInfo, ModelStatus, ModelVariant,
    SpeechControl, SpeechControlRange, SpeechModelCapabilities,
};
pub use prefix_reuse::{
    prefix_reuse_engages, PrefixReuseEvidenceLevel, PrefixReuseMode, PrefixReuseSupport,
    PrefixReuseSupportLevel,
};
pub use variant::{
    normalize_stored_model_id, parse_chat_model_variant, parse_model_variant,
    parse_tts_model_variant, removed_model_variant, resolve_asr_model_variant,
    resolve_diarization_asr_model_variant, resolve_diarization_llm_variant,
    resolve_diarization_model_variant, resolve_diarization_model_variant_strict,
    validate_diarization_speaker_limits, InferenceBackendHint, ModelFamily, ModelTask,
    ParseModelVariantError, RemovedModelVariant, REMOVED_MODEL_VARIANTS,
};
