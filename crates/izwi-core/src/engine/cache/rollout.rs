//! Pre-session managed-cache rollout and authority selection.
//!
//! Cache ownership is immutable for a session. Runtime kernel failures may
//! circuit-break a resolved plan for *new* sessions, but an admitted session
//! never falls back to a second model-owned history.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::engine::ModelInstanceId;
use crate::kv::{CacheCapability, KvPlanFingerprint};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KvRolloutState {
    Legacy,
    ArenaShadow,
    ArenaOptIn,
    Auto,
    ArenaRequired,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvSessionCacheMode {
    None,
    OpaqueModelOwned,
    ArenaShadow { plan: KvPlanFingerprint },
    Managed { plan: KvPlanFingerprint },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvSessionCacheAuthority {
    pub model_instance: ModelInstanceId,
    pub mode: KvSessionCacheMode,
}

impl KvSessionCacheAuthority {
    pub fn require_same_authority(&self, candidate: &Self) -> Result<(), KvRolloutError> {
        if self == candidate {
            Ok(())
        } else {
            Err(KvRolloutError::AuthorityChanged)
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum KvRolloutError {
    #[error("managed KV cache is required but the loaded adapter is not managed")]
    ManagedContractRequired,
    #[error("managed KV cache is required but backend negotiation failed: {0}")]
    NegotiationRequired(String),
    #[error("managed KV cache plan is circuit-broken for new sessions")]
    CircuitOpen,
    #[error("a running session cannot change cache authority")]
    AuthorityChanged,
}

#[derive(Debug, Default)]
pub struct KvPlanCircuitBreaker {
    open: HashSet<KvPlanFingerprint>,
}

impl KvPlanCircuitBreaker {
    pub fn open(&mut self, plan: KvPlanFingerprint) -> bool {
        self.open.insert(plan)
    }

    pub fn close(&mut self, plan: KvPlanFingerprint) -> bool {
        self.open.remove(&plan)
    }

    pub fn is_open(&self, plan: KvPlanFingerprint) -> bool {
        self.open.contains(&plan)
    }
}

/// Select cache authority once, before a session is registered with either the
/// coordinator or the opaque model cache.
pub fn select_session_cache_authority(
    model_instance: ModelInstanceId,
    rollout: KvRolloutState,
    capability: &CacheCapability,
    negotiated: Result<KvPlanFingerprint, String>,
    circuit_breaker: &KvPlanCircuitBreaker,
) -> Result<KvSessionCacheAuthority, KvRolloutError> {
    if matches!(capability, CacheCapability::None) {
        return Ok(KvSessionCacheAuthority {
            model_instance,
            mode: KvSessionCacheMode::None,
        });
    }

    let managed = matches!(capability, CacheCapability::Managed(_));
    if !managed || rollout == KvRolloutState::Legacy {
        if rollout == KvRolloutState::ArenaRequired {
            return Err(KvRolloutError::ManagedContractRequired);
        }
        return Ok(KvSessionCacheAuthority {
            model_instance,
            mode: KvSessionCacheMode::OpaqueModelOwned,
        });
    }

    let plan = match negotiated {
        Ok(plan) => plan,
        Err(reason) if rollout == KvRolloutState::ArenaRequired => {
            return Err(KvRolloutError::NegotiationRequired(reason));
        }
        Err(_) => {
            return Ok(KvSessionCacheAuthority {
                model_instance,
                mode: KvSessionCacheMode::OpaqueModelOwned,
            });
        }
    };
    if circuit_breaker.is_open(plan) {
        if rollout == KvRolloutState::ArenaRequired {
            return Err(KvRolloutError::CircuitOpen);
        }
        return Ok(KvSessionCacheAuthority {
            model_instance,
            mode: KvSessionCacheMode::OpaqueModelOwned,
        });
    }

    let mode = match rollout {
        KvRolloutState::ArenaShadow => KvSessionCacheMode::ArenaShadow { plan },
        KvRolloutState::ArenaOptIn | KvRolloutState::Auto | KvRolloutState::ArenaRequired => {
            KvSessionCacheMode::Managed { plan }
        }
        KvRolloutState::Legacy => unreachable!("legacy returned before negotiation"),
    };
    Ok(KvSessionCacheAuthority {
        model_instance,
        mode,
    })
}

#[cfg(test)]
mod tests {
    use crate::kv::KvPlanFingerprint;

    use super::*;

    fn managed_capability() -> CacheCapability {
        // Selection intentionally needs only the capability discriminator; the
        // contract is validated before negotiation.
        use crate::kv::{KvCacheContract, CURRENT_KV_CONTRACT_ABI};
        CacheCapability::Managed(KvCacheContract {
            abi: CURRENT_KV_CONTRACT_ABI,
            domains: Vec::new(),
        })
    }

    #[test]
    fn shadow_and_managed_are_explicit_pre_session_choices() {
        let model = ModelInstanceId::new(4);
        let plan = KvPlanFingerprint::new([7; 32]);
        let breaker = KvPlanCircuitBreaker::default();
        let shadow = select_session_cache_authority(
            model,
            KvRolloutState::ArenaShadow,
            &managed_capability(),
            Ok(plan),
            &breaker,
        )
        .unwrap();
        let managed = select_session_cache_authority(
            model,
            KvRolloutState::Auto,
            &managed_capability(),
            Ok(plan),
            &breaker,
        )
        .unwrap();
        assert_eq!(shadow.mode, KvSessionCacheMode::ArenaShadow { plan });
        assert_eq!(managed.mode, KvSessionCacheMode::Managed { plan });
        assert_eq!(
            shadow.require_same_authority(&managed).unwrap_err(),
            KvRolloutError::AuthorityChanged
        );
    }

    #[test]
    fn required_mode_fails_closed_and_circuit_breaker_affects_only_new_selection() {
        let model = ModelInstanceId::new(4);
        let plan = KvPlanFingerprint::new([7; 32]);
        let mut breaker = KvPlanCircuitBreaker::default();
        let admitted = select_session_cache_authority(
            model,
            KvRolloutState::Auto,
            &managed_capability(),
            Ok(plan),
            &breaker,
        )
        .unwrap();
        breaker.open(plan);
        assert_eq!(
            select_session_cache_authority(
                model,
                KvRolloutState::ArenaRequired,
                &managed_capability(),
                Ok(plan),
                &breaker,
            )
            .unwrap_err(),
            KvRolloutError::CircuitOpen
        );
        assert_eq!(admitted.mode, KvSessionCacheMode::Managed { plan });
    }

    #[test]
    fn unsupported_or_failed_optional_negotiation_stays_opaque() {
        let breaker = KvPlanCircuitBreaker::default();
        let selected = select_session_cache_authority(
            ModelInstanceId::new(1),
            KvRolloutState::Auto,
            &CacheCapability::OpaqueModelOwned,
            Err("unsupported dtype".into()),
            &breaker,
        )
        .unwrap();
        assert_eq!(selected.mode, KvSessionCacheMode::OpaqueModelOwned);
    }
}
