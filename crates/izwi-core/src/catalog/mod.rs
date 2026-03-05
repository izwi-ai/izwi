//! Model catalog surface aligned with inference-engine style organization.
//!
//! This module is the canonical place for model metadata, capabilities,
//! status records, and identifier parsing. Artifact-management types now live
//! under `crate::artifacts`, while legacy `crate::model` paths remain
//! available for backward compatibility.

mod metadata;
mod variant;

pub use metadata::{ModelInfo, ModelStatus, ModelVariant};
pub use variant::{
    parse_chat_model_variant, parse_model_variant, parse_tts_model_variant,
    resolve_asr_model_variant, resolve_diarization_llm_variant, resolve_diarization_model_variant,
    InferenceBackendHint, ModelFamily, ModelTask, ParseModelVariantError,
};
