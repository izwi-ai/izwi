use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::backends::BackendKind;
use crate::engine::{
    AdapterAbiRevision, AdapterInstanceId, ExecutionAdapterBinding, ExecutionGroupId,
    ModelInstanceId,
};
use crate::error::{Error, Result};
use crate::model::ModelVariant;

use super::{stage_graph_fingerprint, CapabilityStateDescriptorV2};

const STATELESS_RUNTIME_FINGERPRINT_DOMAIN: &[u8] = b"izwi.inference-state.stateless-runtime.v2\0";

/// Canonical identity shared by every retained/workspace/runtime plan for one
/// exact loaded capability. Pool sharing, when introduced, must be an explicit
/// authorization between identities rather than an accidental fingerprint
/// collision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct CapabilityRuntimeIdentityV2 {
    pub(crate) execution_group: ExecutionGroupId,
    pub(crate) model_instance: ModelInstanceId,
    pub(crate) model_variant: ModelVariant,
    pub(crate) backend: BackendKind,
    pub(crate) capability_id: String,
    pub(crate) adapter_instance: AdapterInstanceId,
    pub(crate) adapter_abi: AdapterAbiRevision,
}

impl CapabilityRuntimeIdentityV2 {
    pub(crate) fn seal(backend: BackendKind, execution: &ExecutionAdapterBinding) -> Result<Self> {
        execution.validate()?;
        Ok(Self {
            execution_group: execution.execution_group_id,
            model_instance: execution.model_instance_id,
            model_variant: execution.model_variant,
            backend,
            capability_id: execution.capability_id.clone(),
            adapter_instance: execution.adapter_instance_id,
            adapter_abi: execution.adapter_abi_revision,
        })
    }

    pub(crate) fn validate_against(
        &self,
        backend: BackendKind,
        execution: &ExecutionAdapterBinding,
    ) -> Result<()> {
        execution.validate()?;
        if self.backend != backend
            || self.execution_group != execution.execution_group_id
            || self.model_instance != execution.model_instance_id
            || self.model_variant != execution.model_variant
            || self.capability_id != execution.capability_id
            || self.adapter_instance != execution.adapter_instance_id
            || self.adapter_abi != execution.adapter_abi_revision
        {
            return Err(invalid(
                "capability runtime identity does not match the selected loaded adapter",
            ));
        }
        Ok(())
    }
}

/// Immutable request-selectable proof that one exact loaded capability needs
/// neither retained physical state nor invocation workspace for this stage
/// graph. Stateful runtime variants are added only with backend-owned storage
/// and allocation receipts; descriptor metadata alone is never a runtime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct StatelessCapabilityRuntimeV2 {
    pub(crate) id: [u8; 32],
    pub(crate) identity: CapabilityRuntimeIdentityV2,
    pub(crate) stage_graph_fingerprint: [u8; 32],
    pub(crate) state_fingerprint: [u8; 32],
    pub(crate) descriptor: CapabilityStateDescriptorV2,
}

/// Request-facing inference-state runtime. Callers bind this model-neutral
/// handle and never branch on whether the backing is stateless, paged KV, or a
/// future tensor/ring arena. The backing kind remains private to the state
/// runtime so adding a physical domain cannot create another request ABI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CapabilityStateRuntimeV2 {
    pub(crate) id: [u8; 32],
    pub(crate) state_fingerprint: [u8; 32],
    pub(crate) descriptor: CapabilityStateDescriptorV2,
    backing: CapabilityStateRuntimeBackingV2,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CapabilityStateRuntimeBackingV2 {
    Stateless(StatelessCapabilityRuntimeV2),
}

impl CapabilityStateRuntimeV2 {
    pub(crate) fn stateless(runtime: StatelessCapabilityRuntimeV2) -> Self {
        Self {
            id: runtime.id,
            state_fingerprint: runtime.state_fingerprint,
            descriptor: runtime.descriptor.clone(),
            backing: CapabilityStateRuntimeBackingV2::Stateless(runtime),
        }
    }

    pub(crate) fn validate_against(
        &self,
        backend: BackendKind,
        execution: &ExecutionAdapterBinding,
    ) -> Result<()> {
        match &self.backing {
            CapabilityStateRuntimeBackingV2::Stateless(runtime) => {
                runtime.validate_against(backend, execution)?;
                if self.id != runtime.id
                    || self.state_fingerprint != runtime.state_fingerprint
                    || self.descriptor != runtime.descriptor
                {
                    return Err(invalid(
                        "state ABI v2 runtime wrapper does not match its sealed backing",
                    ));
                }
                Ok(())
            }
        }
    }
}

impl StatelessCapabilityRuntimeV2 {
    pub(crate) fn seal(
        backend: BackendKind,
        execution: &ExecutionAdapterBinding,
        descriptor: CapabilityStateDescriptorV2,
    ) -> Result<Self> {
        execution.validate()?;
        descriptor.validate_against_stages(&execution.stages)?;
        if !descriptor.is_stateless()
            || !descriptor.has_zero_invocation_workspace_for(&execution.stages)?
        {
            return Err(invalid(
                "only stateless zero-workspace state ABI v2 capabilities can be sealed without a physical backend runtime",
            ));
        }
        let stage_graph_fingerprint = stage_graph_fingerprint(&execution.stages)?;
        let state_fingerprint = descriptor.fingerprint(&execution.stages)?;
        let mut runtime = Self {
            id: [0; 32],
            identity: CapabilityRuntimeIdentityV2::seal(backend, execution)?,
            stage_graph_fingerprint,
            state_fingerprint,
            descriptor,
        };
        runtime.id = runtime.compute_id()?;
        runtime.validate_against(backend, execution)?;
        Ok(runtime)
    }

    pub(crate) fn validate_against(
        &self,
        backend: BackendKind,
        execution: &ExecutionAdapterBinding,
    ) -> Result<()> {
        execution.validate()?;
        self.identity.validate_against(backend, execution)?;
        if self.stage_graph_fingerprint != stage_graph_fingerprint(&execution.stages)?
            || self.state_fingerprint != self.descriptor.fingerprint(&execution.stages)?
            || !self.descriptor.is_stateless()
            || !self
                .descriptor
                .has_zero_invocation_workspace_for(&execution.stages)?
            || self.id != self.compute_id()?
        {
            return Err(invalid(
                "stateless state ABI v2 runtime does not match the selected loaded capability",
            ));
        }
        Ok(())
    }

    fn compute_id(&self) -> Result<[u8; 32]> {
        #[derive(Serialize)]
        struct Payload<'a> {
            identity: &'a CapabilityRuntimeIdentityV2,
            stage_graph_fingerprint: [u8; 32],
            state_fingerprint: [u8; 32],
        }

        let encoded = serde_json::to_vec(&Payload {
            identity: &self.identity,
            stage_graph_fingerprint: self.stage_graph_fingerprint,
            state_fingerprint: self.state_fingerprint,
        })
        .map_err(|error| invalid(format!("failed to encode stateless runtime: {error}")))?;
        let mut hasher = Sha256::new();
        hasher.update(STATELESS_RUNTIME_FINGERPRINT_DOMAIN);
        hasher.update(encoded);
        Ok(hasher.finalize().into())
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::engine::{
        ExecutionMode, ExecutionProfile, NativeBatchMode, StageDescriptor, StageId,
    };

    fn binding() -> ExecutionAdapterBinding {
        let variant = ModelVariant::Kokoro82M;
        let profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, Some(variant), ExecutionMode::Atomic);
        let stage = StageDescriptor::from_execution_profile(
            StageId::new(1),
            "tts.stateless",
            &profile,
            NativeBatchMode::None,
        );
        ExecutionAdapterBinding {
            execution_group_id: ExecutionGroupId::new(2),
            model_instance_id: ModelInstanceId::new(3),
            adapter_instance_id: AdapterInstanceId::new(4),
            adapter_abi_revision: AdapterAbiRevision::new(5),
            model_variant: variant,
            capability_id: "tts".to_string(),
            stages: Arc::from([stage]),
        }
    }

    #[test]
    fn stateless_runtime_seals_the_complete_execution_identity() {
        let binding = binding();
        let descriptor = CapabilityStateDescriptorV2::stateless_for_stages_test(&binding.stages);
        let runtime =
            StatelessCapabilityRuntimeV2::seal(BackendKind::Cpu, &binding, descriptor).unwrap();
        runtime
            .validate_against(BackendKind::Cpu, &binding)
            .unwrap();
        assert!(runtime
            .validate_against(BackendKind::Cuda, &binding)
            .is_err());

        let mut changed = binding.clone();
        changed.execution_group_id = ExecutionGroupId::new(20);
        assert!(runtime
            .validate_against(BackendKind::Cpu, &changed)
            .is_err());
        let mut changed = binding.clone();
        changed.model_instance_id = ModelInstanceId::new(30);
        assert!(runtime
            .validate_against(BackendKind::Cpu, &changed)
            .is_err());
        let mut changed = binding.clone();
        changed.model_variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        assert!(runtime
            .validate_against(BackendKind::Cpu, &changed)
            .is_err());
        let mut changed = binding.clone();
        changed.adapter_instance_id = AdapterInstanceId::new(40);
        assert!(runtime
            .validate_against(BackendKind::Cpu, &changed)
            .is_err());
        let mut changed = binding.clone();
        changed.adapter_abi_revision = AdapterAbiRevision::new(50);
        assert!(runtime
            .validate_against(BackendKind::Cpu, &changed)
            .is_err());
        let mut changed = binding.clone();
        changed.capability_id = "streaming_tts".to_string();
        assert!(runtime
            .validate_against(BackendKind::Cpu, &changed)
            .is_err());
        let mut changed = binding;
        Arc::make_mut(&mut changed.stages)[0].name = "tts.changed".to_string();
        assert!(runtime
            .validate_against(BackendKind::Cpu, &changed)
            .is_err());
    }
}
