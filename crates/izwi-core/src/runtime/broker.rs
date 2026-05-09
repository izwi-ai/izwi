//! Rollout-aware inference broker skeleton.
//!
//! Phase 3 keeps this deliberately inert: the broker can be configured and
//! inspected, but runtime execution still follows existing `RuntimeService`
//! paths until later phases explicitly opt capabilities in.

use serde::Serialize;

const BROKER_MODE_ENV: &str = "IZWI_INFERENCE_BROKER";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum InferenceBrokerMode {
    Off,
    Shadow,
    On,
}

impl InferenceBrokerMode {
    fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "shadow" | "audit" => Self::Shadow,
            "on" | "enabled" | "true" | "1" => Self::On,
            _ => Self::Off,
        }
    }

    fn from_env() -> Self {
        std::env::var(BROKER_MODE_ENV)
            .ok()
            .as_deref()
            .map(Self::parse)
            .unwrap_or(Self::Off)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct InferenceBrokerSnapshot {
    pub(crate) mode: InferenceBrokerMode,
    pub(crate) shadow_enabled: bool,
    pub(crate) execution_enabled: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct InferenceBroker {
    mode: InferenceBrokerMode,
}

impl InferenceBroker {
    pub(crate) fn from_env() -> Self {
        Self {
            mode: InferenceBrokerMode::from_env(),
        }
    }

    pub(crate) fn with_mode(mode: InferenceBrokerMode) -> Self {
        Self { mode }
    }

    pub(crate) fn mode(&self) -> InferenceBrokerMode {
        self.mode
    }

    pub(crate) fn shadow_enabled(&self) -> bool {
        matches!(self.mode, InferenceBrokerMode::Shadow | InferenceBrokerMode::On)
    }

    pub(crate) fn execution_enabled(&self) -> bool {
        matches!(self.mode, InferenceBrokerMode::On)
    }

    pub(crate) fn snapshot(&self) -> InferenceBrokerSnapshot {
        InferenceBrokerSnapshot {
            mode: self.mode(),
            shadow_enabled: self.shadow_enabled(),
            execution_enabled: self.execution_enabled(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broker_mode_defaults_unknown_values_to_off() {
        assert_eq!(InferenceBrokerMode::parse(""), InferenceBrokerMode::Off);
        assert_eq!(
            InferenceBrokerMode::parse("something-else"),
            InferenceBrokerMode::Off
        );
    }

    #[test]
    fn broker_mode_accepts_shadow_alias() {
        assert_eq!(InferenceBrokerMode::parse("shadow"), InferenceBrokerMode::Shadow);
        assert_eq!(InferenceBrokerMode::parse("audit"), InferenceBrokerMode::Shadow);
    }

    #[test]
    fn broker_snapshot_separates_shadow_from_execution() {
        let off = InferenceBroker::with_mode(InferenceBrokerMode::Off).snapshot();
        assert!(!off.shadow_enabled);
        assert!(!off.execution_enabled);

        let shadow = InferenceBroker::with_mode(InferenceBrokerMode::Shadow).snapshot();
        assert!(shadow.shadow_enabled);
        assert!(!shadow.execution_enabled);

        let on = InferenceBroker::with_mode(InferenceBrokerMode::On).snapshot();
        assert!(on.shadow_enabled);
        assert!(on.execution_enabled);
    }
}
