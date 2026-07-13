//! Executable capability contracts for runtime adapters.
//!
//! The first implementation layer plans and validates capability execution
//! without taking over dispatch. That keeps current runtime behavior stable
//! while giving the broker one concrete place to ask: can this model execute
//! this capability under this stream mode?

use crate::backends::BackendKind;
use crate::catalog::{ModelFamily, ModelVariant};
use crate::engine::{CacheMode, ExecutionMode, ExecutionProfile, NativeBatchMode, PrefillMode};
use crate::error::{Error, Result};
use crate::runtime::adapters::{
    AdapterMetadata, CapabilityKind, ExecutionTargetKind, RuntimeAdapterRegistry, StreamingMode,
};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CapabilityExecutionRequest {
    pub(crate) capability: CapabilityKind,
    pub(crate) model_variant: ModelVariant,
    pub(crate) streaming_required: bool,
    pub(crate) backend_kind: BackendKind,
}

impl CapabilityExecutionRequest {
    pub(crate) fn new(
        capability: CapabilityKind,
        model_variant: ModelVariant,
        backend_kind: BackendKind,
    ) -> Self {
        Self {
            capability,
            model_variant,
            streaming_required: false,
            backend_kind,
        }
    }

    pub(crate) fn with_streaming_required(mut self, streaming_required: bool) -> Self {
        self.streaming_required = streaming_required;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct CapabilityExecutionPlan {
    pub(crate) adapter_id: &'static str,
    pub(crate) capability: CapabilityKind,
    pub(crate) model_variant: ModelVariant,
    pub(crate) streaming_mode: StreamingMode,
    pub(crate) execution_target: ExecutionTargetKind,
    pub(crate) execution_profile: ExecutionProfile,
}

impl CapabilityExecutionPlan {
    fn from_metadata(
        metadata: AdapterMetadata,
        backend_kind: BackendKind,
        streaming_required: bool,
    ) -> Self {
        let mut execution_profile =
            route_execution_profile(metadata, backend_kind, streaming_required);
        execution_profile.resolved_from_loaded_model = false;
        Self {
            adapter_id: metadata.id,
            capability: metadata.capability,
            model_variant: metadata.model_variant,
            streaming_mode: metadata.streaming_mode,
            execution_target: metadata.execution_target,
            execution_profile,
        }
    }
}

fn route_execution_profile(
    metadata: AdapterMetadata,
    backend_kind: BackendKind,
    streaming_required: bool,
) -> ExecutionProfile {
    let variant = metadata.model_variant;
    let family = variant.family();
    let sequence = match metadata.capability {
        CapabilityKind::Chat => {
            matches!(family, ModelFamily::Qwen35Chat)
                || matches!(
                    variant,
                    ModelVariant::Qwen306B
                        | ModelVariant::Qwen306B4Bit
                        | ModelVariant::Qwen317B
                        | ModelVariant::Qwen317B4Bit
                )
        }
        CapabilityKind::Tts | CapabilityKind::StreamingTts => family == ModelFamily::Qwen3Tts,
        CapabilityKind::Asr => family == ModelFamily::Qwen3Asr && streaming_required,
        _ => false,
    };
    let mode = if sequence {
        ExecutionMode::Sequence
    } else {
        match metadata.execution_target {
            ExecutionTargetKind::RealtimeRunner => ExecutionMode::Realtime,
            ExecutionTargetKind::PipelineRunner => ExecutionMode::Pipeline,
            ExecutionTargetKind::Artifact => ExecutionMode::Artifact,
            ExecutionTargetKind::TokenEngine
            | ExecutionTargetKind::BatchRunner
            | ExecutionTargetKind::DirectModel => ExecutionMode::Atomic,
        }
    };
    let mut profile = ExecutionProfile::fail_closed(backend_kind, Some(variant), mode);
    if sequence {
        profile.prefill = PrefillMode::Full;
        profile.incremental_decode = true;
        profile.cache_mode = CacheMode::OpaqueModelOwned;
    }
    // The existing Qwen TTS batch API is static and request-shape dependent;
    // the runtime route cannot prove those conditions, so it remains disabled
    // here and is enabled only by the loaded executor profile.
    profile.prefill_batch = NativeBatchMode::None;
    profile.compute_dtype = "loaded_model_default".to_string();
    profile.kv_dtype = if sequence {
        "loaded_model_default".to_string()
    } else {
        "none".to_string()
    };
    profile.cache_namespace =
        sequence.then(|| format!("{}:{}:opaque", variant, backend_kind.as_str()));
    profile
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CapabilityExecutionRegistry<'a> {
    adapters: &'a RuntimeAdapterRegistry,
}

impl<'a> CapabilityExecutionRegistry<'a> {
    pub(crate) fn new(adapters: &'a RuntimeAdapterRegistry) -> Self {
        Self { adapters }
    }

    pub(crate) fn plan(
        &self,
        request: CapabilityExecutionRequest,
    ) -> Result<CapabilityExecutionPlan> {
        let metadata = *self
            .adapters
            .require(request.capability, request.model_variant)?;

        if request.streaming_required && metadata.streaming_mode == StreamingMode::None {
            return Err(Error::InvalidInput(format!(
                "Model {} supports {:?}, but not streaming execution for that capability",
                request.model_variant, request.capability
            )));
        }

        Ok(CapabilityExecutionPlan::from_metadata(
            metadata,
            request.backend_kind,
            request.streaming_required,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelVariant;

    #[test]
    fn execution_registry_plans_supported_capability() {
        let adapters = RuntimeAdapterRegistry::built_in();
        let registry = CapabilityExecutionRegistry::new(&adapters);

        let plan = registry
            .plan(CapabilityExecutionRequest::new(
                CapabilityKind::Chat,
                ModelVariant::Qwen38BGguf,
                BackendKind::Cpu,
            ))
            .expect("chat plan");

        assert_eq!(plan.adapter_id, "builtin.chat");
        assert_eq!(plan.capability, CapabilityKind::Chat);
        assert_eq!(plan.execution_target, ExecutionTargetKind::TokenEngine);
    }

    #[test]
    fn execution_registry_rejects_unsupported_capability() {
        let adapters = RuntimeAdapterRegistry::built_in();
        let registry = CapabilityExecutionRegistry::new(&adapters);

        let err = registry
            .plan(CapabilityExecutionRequest::new(
                CapabilityKind::Chat,
                ModelVariant::Kokoro82M,
                BackendKind::Cpu,
            ))
            .unwrap_err();

        assert!(err.to_string().contains("does not support"));
    }

    #[test]
    fn execution_registry_rejects_required_streaming_when_adapter_is_batch_only() {
        let adapters = RuntimeAdapterRegistry::built_in();
        let registry = CapabilityExecutionRegistry::new(&adapters);

        let err = registry
            .plan(
                CapabilityExecutionRequest::new(
                    CapabilityKind::Asr,
                    ModelVariant::WhisperLargeV3Turbo,
                    BackendKind::Cpu,
                )
                .with_streaming_required(true),
            )
            .unwrap_err();

        assert!(err.to_string().contains("not streaming execution"));
    }

    #[test]
    fn route_profiles_match_the_execution_adapters_that_exist_today() {
        let adapters = RuntimeAdapterRegistry::built_in();
        let registry = CapabilityExecutionRegistry::new(&adapters);

        for (variant, expected_mode) in [
            (ModelVariant::Qwen306B, ExecutionMode::Sequence),
            (ModelVariant::Qwen306BGguf, ExecutionMode::Atomic),
            (ModelVariant::Qwen354BGguf, ExecutionMode::Sequence),
            (ModelVariant::Lfm2512BInstructGguf, ExecutionMode::Atomic),
            (ModelVariant::Gemma34BIt, ExecutionMode::Atomic),
        ] {
            let plan = registry
                .plan(CapabilityExecutionRequest::new(
                    CapabilityKind::Chat,
                    variant,
                    BackendKind::Cpu,
                ))
                .expect("chat route plan");
            assert_eq!(plan.execution_profile.mode, expected_mode, "{variant}");
            assert!(!plan.execution_profile.resolved_from_loaded_model);
        }
    }

    #[test]
    fn qwen_asr_route_only_claims_sequence_execution_for_streaming_today() {
        let adapters = RuntimeAdapterRegistry::built_in();
        let registry = CapabilityExecutionRegistry::new(&adapters);
        let offline = registry
            .plan(CapabilityExecutionRequest::new(
                CapabilityKind::Asr,
                ModelVariant::Qwen3Asr06BGguf,
                BackendKind::Metal,
            ))
            .expect("offline ASR plan");
        let streaming = registry
            .plan(
                CapabilityExecutionRequest::new(
                    CapabilityKind::Asr,
                    ModelVariant::Qwen3Asr06BGguf,
                    BackendKind::Metal,
                )
                .with_streaming_required(true),
            )
            .expect("streaming ASR plan");

        assert_eq!(offline.execution_profile.mode, ExecutionMode::Atomic);
        assert_eq!(streaming.execution_profile.mode, ExecutionMode::Sequence);
        assert_eq!(streaming.execution_profile.backend, BackendKind::Metal);
    }
}
