//! Model catalog surface aligned with inference-engine style organization.
//!
//! This module is the canonical place for model metadata, capabilities,
//! status records, and identifier parsing. Artifact-management types now live
//! under `crate::artifacts`, while legacy `crate::model` paths remain
//! available for backward compatibility.

mod batch_width;
mod cuda_support;
mod metadata;
mod prefix_reuse;
mod variant;

pub(crate) use batch_width::{
    certified_decode_width, engine_row_width, max_certified_decode_width,
};
pub use cuda_support::{
    CudaEvidenceLevel, CudaExecutionStatus, CudaOperatorCapability, CudaOperatorKind,
    CudaProviderClass, CudaQuantizationInfo, CudaQuantizationSupportLevel, CudaSupportInfo,
    CudaSupportLevel,
};
pub use metadata::{
    ChatModelCapabilities, ChatReasoningEffort, ModelInfo, ModelStatus, ModelVariant,
    SpeechModelCapabilities,
};
pub use prefix_reuse::{
    prefix_reuse_engages, PrefixReuseEvidenceLevel, PrefixReuseMode, PrefixReuseSupport,
    PrefixReuseSupportLevel,
};
pub use variant::{
    normalize_stored_model_id, parse_chat_model_variant, parse_model_variant,
    parse_tts_model_variant, removed_model_variant, resolve_asr_model_variant,
    resolve_diarization_llm_variant, resolve_diarization_model_variant,
    resolve_diarization_model_variant_strict, InferenceBackendHint, ModelFamily, ModelTask,
    ParseModelVariantError, RemovedModelVariant, REMOVED_MODEL_VARIANTS,
};
