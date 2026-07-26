use serde::{Deserialize, Serialize};

use crate::error::Result;

use super::v2::InferenceStateContract;

/// Inference-state behavior published by an exact loaded-model
/// implementation.
///
/// Stateful models publish the same ABI-v2 semantic contract consumed by
/// backend negotiation and physical allocation. Stateless models promise that
/// no mutable inference state survives an invocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", content = "contract", rename_all = "snake_case")]
pub(crate) enum InferenceStateCapability {
    Stateless,
    Managed(InferenceStateContract),
}

impl InferenceStateCapability {
    pub(crate) fn validate(&self) -> Result<()> {
        match self {
            Self::Stateless => Ok(()),
            Self::Managed(contract) => contract.validate(),
        }
    }

    pub(crate) fn managed_contract(&self) -> Option<&InferenceStateContract> {
        match self {
            Self::Managed(contract) => Some(contract),
            Self::Stateless => None,
        }
    }
}

/// Implemented by the loaded adapter/model boundary, never by catalog
/// entries.
pub(crate) trait InferenceStateContractProvider {
    fn inference_state_contract(&self) -> Result<InferenceStateCapability>;
}
