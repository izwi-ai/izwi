//! Model catalog surface aligned with inference-engine style organization.
//!
//! This module is the canonical place for model metadata, capabilities,
//! status records, and identifier parsing. Artifact-management types now live
//! under `crate::artifacts`, while legacy `crate::model` paths remain
//! available for backward compatibility. The accelerator-free catalog itself
//! lives in `izwi-model-catalog` and is re-exported here; the certified
//! widths and mixed steps stay with the engine they configure.

mod batch_width;
mod mixed_step;

pub(crate) use batch_width::{
    certified_decode_width, engine_row_width, max_certified_decode_width, non_chat_row_width,
    EXPLICIT_WIDTH_CEILING,
};
pub use izwi_model_catalog::catalog::*;
pub(crate) use mixed_step::mixed_step_enabled;
