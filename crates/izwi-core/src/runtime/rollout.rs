//! Fail-closed rollout controls for scheduler execution modes.
//!
//! Rollout is keyed by both the concrete model and the selected backend. This
//! prevents enabling an optimization validated on one device from implicitly
//! enabling it on CPU, Metal, or CUDA peers with different execution behavior.

use std::collections::HashMap;

use serde::Serialize;

use crate::backends::BackendKind;
use crate::catalog::{parse_model_variant, ModelVariant};
use crate::error::{Error, Result};

const EXECUTION_ROLLOUT_ENV: &str = "IZWI_EXECUTION_ROLLOUT";
const EXECUTION_ROLLOUT_OVERRIDES_ENV: &str = "IZWI_EXECUTION_ROLLOUT_OVERRIDES";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ExecutionRolloutMode {
    #[default]
    Off,
    Shadow,
    Static,
    Continuous,
}

impl ExecutionRolloutMode {
    fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "off" => Ok(Self::Off),
            "shadow" => Ok(Self::Shadow),
            "static" => Ok(Self::Static),
            "continuous" => Ok(Self::Continuous),
            value => Err(Error::InvalidInput(format!(
                "Unsupported execution rollout mode `{value}`"
            ))),
        }
    }

    pub(crate) fn observes(self) -> bool {
        !matches!(self, Self::Off)
    }

    pub(crate) fn executes(self) -> bool {
        matches!(self, Self::Static | Self::Continuous)
    }

    pub(crate) fn continuous_batching(self) -> bool {
        matches!(self, Self::Continuous)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ExecutionRolloutKey {
    model_variant: ModelVariant,
    backend_kind: BackendKind,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ExecutionRolloutPolicy {
    default: ExecutionRolloutMode,
    overrides: HashMap<ExecutionRolloutKey, ExecutionRolloutMode>,
}

impl ExecutionRolloutPolicy {
    /// Reads rollout configuration without ever enabling execution on invalid
    /// input. Callers that need diagnostics can use `try_from_raw` directly.
    pub(crate) fn from_env() -> Self {
        let default = std::env::var(EXECUTION_ROLLOUT_ENV).ok();
        let overrides = std::env::var(EXECUTION_ROLLOUT_OVERRIDES_ENV).ok();
        Self::try_from_raw(default.as_deref(), overrides.as_deref()).unwrap_or_default()
    }

    pub(crate) fn try_from_raw(default: Option<&str>, overrides: Option<&str>) -> Result<Self> {
        let default = default
            .map(ExecutionRolloutMode::parse)
            .transpose()?
            .unwrap_or_default();
        if default.executes() {
            return Err(Error::InvalidInput(
                "Execution rollout defaults may only be `off` or `shadow`; static and continuous execution require exact model@backend overrides"
                    .to_string(),
            ));
        }
        let mut policy = Self {
            default,
            overrides: HashMap::new(),
        };

        for entry in overrides.unwrap_or_default().split(',') {
            let entry = entry.trim();
            if entry.is_empty() {
                continue;
            }

            let (scope, raw_mode) = entry.split_once('=').ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Invalid execution rollout override `{entry}`; expected model@backend=mode"
                ))
            })?;
            let (raw_model, raw_backend) = scope.split_once('@').ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Invalid execution rollout scope `{scope}`; expected model@backend"
                ))
            })?;
            let key = ExecutionRolloutKey {
                model_variant: parse_model_variant(raw_model)
                    .map_err(|err| Error::InvalidInput(err.to_string()))?,
                backend_kind: parse_backend(raw_backend)?,
            };
            let mode = ExecutionRolloutMode::parse(raw_mode)?;
            if policy.overrides.insert(key, mode).is_some() {
                return Err(Error::InvalidInput(format!(
                    "Duplicate execution rollout override for {scope}"
                )));
            }
        }

        Ok(policy)
    }

    pub(crate) fn mode_for(
        &self,
        model_variant: ModelVariant,
        backend_kind: BackendKind,
    ) -> ExecutionRolloutMode {
        self.overrides
            .get(&ExecutionRolloutKey {
                model_variant,
                backend_kind,
            })
            .copied()
            .unwrap_or(self.default)
    }
}

fn parse_backend(raw: &str) -> Result<BackendKind> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "cpu" => Ok(BackendKind::Cpu),
        "metal" => Ok(BackendKind::Metal),
        "cuda" => Ok(BackendKind::Cuda),
        value => Err(Error::InvalidInput(format!(
            "Unsupported execution rollout backend `{value}`"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BACKENDS: [BackendKind; 3] = [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda];

    #[test]
    fn rollout_defaults_every_model_and_backend_to_off() {
        let policy = ExecutionRolloutPolicy::default();

        for model_variant in ModelVariant::all().iter().copied() {
            for backend_kind in BACKENDS {
                assert_eq!(
                    policy.mode_for(model_variant, backend_kind),
                    ExecutionRolloutMode::Off,
                    "unexpected rollout for {model_variant:?} on {backend_kind:?}"
                );
            }
        }
    }

    #[test]
    fn rollout_modes_only_enable_their_declared_behavior() {
        assert!(!ExecutionRolloutMode::Off.observes());
        assert!(!ExecutionRolloutMode::Off.executes());
        assert!(!ExecutionRolloutMode::Off.continuous_batching());

        assert!(ExecutionRolloutMode::Shadow.observes());
        assert!(!ExecutionRolloutMode::Shadow.executes());
        assert!(!ExecutionRolloutMode::Shadow.continuous_batching());

        assert!(ExecutionRolloutMode::Static.observes());
        assert!(ExecutionRolloutMode::Static.executes());
        assert!(!ExecutionRolloutMode::Static.continuous_batching());

        assert!(ExecutionRolloutMode::Continuous.observes());
        assert!(ExecutionRolloutMode::Continuous.executes());
        assert!(ExecutionRolloutMode::Continuous.continuous_batching());
    }

    #[test]
    fn exact_model_backend_override_wins_over_default() {
        let policy = ExecutionRolloutPolicy::try_from_raw(
            Some("shadow"),
            Some("Qwen3-0.6B@cuda=continuous,Qwen3-0.6B@metal=static"),
        )
        .expect("valid rollout policy");

        assert_eq!(
            policy.mode_for(ModelVariant::Qwen306B, BackendKind::Cpu),
            ExecutionRolloutMode::Shadow
        );
        assert_eq!(
            policy.mode_for(ModelVariant::Qwen306B, BackendKind::Metal),
            ExecutionRolloutMode::Static
        );
        assert_eq!(
            policy.mode_for(ModelVariant::Qwen306B, BackendKind::Cuda),
            ExecutionRolloutMode::Continuous
        );
    }

    #[test]
    fn malformed_configuration_cannot_enable_execution() {
        for (default, overrides) in [
            (Some("enabled"), None),
            (Some("static"), None),
            (Some("continuous"), None),
            (Some("off"), Some("Qwen3-0.6B@cuda=enabled")),
            (Some("off"), Some("unknown-model@cuda=continuous")),
            (Some("off"), Some("Qwen3-0.6B@gpu=continuous")),
            (Some("off"), Some("Qwen3-0.6B@cuda")),
        ] {
            let policy =
                ExecutionRolloutPolicy::try_from_raw(default, overrides).unwrap_or_default();
            assert_eq!(
                policy.mode_for(ModelVariant::Qwen306B, BackendKind::Cuda),
                ExecutionRolloutMode::Off
            );
        }
    }

    #[test]
    fn blanket_execution_defaults_are_rejected() {
        for mode in ["static", "continuous"] {
            let err = ExecutionRolloutPolicy::try_from_raw(Some(mode), None)
                .expect_err("execution must be scoped to an exact model and backend");
            assert!(err.to_string().contains("exact model@backend"));
        }
    }

    #[test]
    fn duplicate_override_is_rejected_instead_of_using_order() {
        let err = ExecutionRolloutPolicy::try_from_raw(
            Some("off"),
            Some("Qwen3-0.6B@cpu=static,Qwen3-0.6B@cpu=continuous"),
        )
        .expect_err("duplicate override must fail closed");

        assert!(err.to_string().contains("Duplicate"));
    }
}
