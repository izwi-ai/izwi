//! Explicit residency/transfer state reserved for real tiered KV arenas.
//!
//! Managed local arenas do not instantiate this state machine. It exists so a
//! future offload backend cannot label a page resident before a destination
//! buffer and acknowledged transfer actually exist.

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KvTransferId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvStorageTier {
    Host,
    Device,
    Unified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum KvResidencyState {
    Resident {
        tier: KvStorageTier,
    },
    Loading {
        operation: KvTransferId,
        from: KvStorageTier,
        to: KvStorageTier,
    },
    Offloading {
        operation: KvTransferId,
        from: KvStorageTier,
        to: KvStorageTier,
    },
    BothDuringTransfer {
        operation: KvTransferId,
        source: KvStorageTier,
        destination: KvStorageTier,
    },
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum KvResidencyError {
    #[error("source and destination KV tiers must differ")]
    SameTier,
    #[error("KV transfer operation does not match the active operation")]
    WrongOperation,
    #[error("KV residency transition is invalid for the current state")]
    InvalidTransition,
}

impl KvResidencyState {
    pub fn begin_loading(
        self,
        operation: KvTransferId,
        destination: KvStorageTier,
    ) -> Result<Self, KvResidencyError> {
        let Self::Resident { tier: source } = self else {
            return Err(KvResidencyError::InvalidTransition);
        };
        if source == destination {
            return Err(KvResidencyError::SameTier);
        }
        Ok(Self::Loading {
            operation,
            from: source,
            to: destination,
        })
    }

    /// A destination allocation exists after this transition, but the source
    /// remains authoritative and pinned until copy completion is acknowledged.
    pub fn destination_allocated(self, operation: KvTransferId) -> Result<Self, KvResidencyError> {
        match self {
            Self::Loading {
                operation: active,
                from,
                to,
            }
            | Self::Offloading {
                operation: active,
                from,
                to,
            } if active == operation => Ok(Self::BothDuringTransfer {
                operation,
                source: from,
                destination: to,
            }),
            Self::Loading { .. } | Self::Offloading { .. } => Err(KvResidencyError::WrongOperation),
            Self::Resident { .. } | Self::BothDuringTransfer { .. } => {
                Err(KvResidencyError::InvalidTransition)
            }
        }
    }

    /// Only an acknowledged transfer may make the destination authoritative.
    pub fn acknowledge(self, operation: KvTransferId) -> Result<Self, KvResidencyError> {
        match self {
            Self::BothDuringTransfer {
                operation: active,
                destination,
                ..
            } if active == operation => Ok(Self::Resident { tier: destination }),
            Self::BothDuringTransfer { .. } => Err(KvResidencyError::WrongOperation),
            _ => Err(KvResidencyError::InvalidTransition),
        }
    }

    /// A failed copy discards the destination and restores source authority.
    pub fn abort(self, operation: KvTransferId) -> Result<Self, KvResidencyError> {
        match self {
            Self::Loading {
                operation: active,
                from,
                ..
            }
            | Self::Offloading {
                operation: active,
                from,
                ..
            } if active == operation => Ok(Self::Resident { tier: from }),
            Self::BothDuringTransfer {
                operation: active,
                source,
                ..
            } if active == operation => Ok(Self::Resident { tier: source }),
            Self::Loading { .. } | Self::Offloading { .. } | Self::BothDuringTransfer { .. } => {
                Err(KvResidencyError::WrongOperation)
            }
            Self::Resident { .. } => Err(KvResidencyError::InvalidTransition),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn destination_is_not_resident_before_acknowledgement() {
        let operation = KvTransferId(9);
        let loading = KvResidencyState::Resident {
            tier: KvStorageTier::Host,
        }
        .begin_loading(operation, KvStorageTier::Device)
        .unwrap();
        let copying = loading.destination_allocated(operation).unwrap();
        assert_eq!(
            copying,
            KvResidencyState::BothDuringTransfer {
                operation,
                source: KvStorageTier::Host,
                destination: KvStorageTier::Device
            }
        );
        assert_eq!(
            copying.acknowledge(operation).unwrap(),
            KvResidencyState::Resident {
                tier: KvStorageTier::Device
            }
        );
    }

    #[test]
    fn failed_or_stale_transfers_preserve_source_authority() {
        let operation = KvTransferId(4);
        let loading = KvResidencyState::Resident {
            tier: KvStorageTier::Device,
        }
        .begin_loading(operation, KvStorageTier::Host)
        .unwrap();
        assert_eq!(
            loading.acknowledge(operation).unwrap_err(),
            KvResidencyError::InvalidTransition
        );
        assert_eq!(
            loading.abort(KvTransferId(5)).unwrap_err(),
            KvResidencyError::WrongOperation
        );
        assert_eq!(
            loading.abort(operation).unwrap(),
            KvResidencyState::Resident {
                tier: KvStorageTier::Device
            }
        );
    }
}
