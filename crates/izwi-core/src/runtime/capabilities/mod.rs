//! Executable capability contracts for runtime adapters.
//!
//! The first implementation layer plans and validates capability execution
//! without taking over dispatch. That keeps current runtime behavior stable
//! while giving the broker one concrete place to ask: can this model execute
//! this capability under this stream mode?

use crate::catalog::ModelVariant;
use crate::error::{Error, Result};
use crate::runtime::adapters::{
    AdapterMetadata, CapabilityKind, ExecutionTargetKind, RuntimeAdapterRegistry, StreamingMode,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CapabilityExecutionRequest {
    pub(crate) capability: CapabilityKind,
    pub(crate) model_variant: ModelVariant,
    pub(crate) streaming_required: bool,
}

impl CapabilityExecutionRequest {
    pub(crate) fn new(capability: CapabilityKind, model_variant: ModelVariant) -> Self {
        Self {
            capability,
            model_variant,
            streaming_required: false,
        }
    }

    pub(crate) fn with_streaming_required(mut self, streaming_required: bool) -> Self {
        self.streaming_required = streaming_required;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CapabilityExecutionPlan {
    pub(crate) adapter_id: &'static str,
    pub(crate) capability: CapabilityKind,
    pub(crate) model_variant: ModelVariant,
    pub(crate) streaming_mode: StreamingMode,
    pub(crate) execution_target: ExecutionTargetKind,
}

impl From<AdapterMetadata> for CapabilityExecutionPlan {
    fn from(metadata: AdapterMetadata) -> Self {
        Self {
            adapter_id: metadata.id,
            capability: metadata.capability,
            model_variant: metadata.model_variant,
            streaming_mode: metadata.streaming_mode,
            execution_target: metadata.execution_target,
        }
    }
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

        Ok(metadata.into())
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
                )
                .with_streaming_required(true),
            )
            .unwrap_err();

        assert!(err.to_string().contains("not streaming execution"));
    }
}
