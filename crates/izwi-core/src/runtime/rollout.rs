//! Fail-closed rollout policy for physical inference-state providers.
//!
//! Provider selection is part of the resolved state plan, not a best-effort
//! runtime fallback. The one process-level override supported here can only
//! demote a certified optimized provider to its certified portable provider.

use std::ffi::OsStr;

use crate::backends::BackendKind;
use crate::catalog::{ModelFamily, ModelVariant};
use crate::error::{Error, Result};
use crate::kv::v2::{PagedOperationImplementation, ResolvedStatePlan};
use crate::runtime::adapters::CapabilityKind;

pub(crate) const DISABLE_OPTIMIZED_KV_PROVIDER_ENV: &str = "IZWI_KV_DISABLE_OPTIMIZED_PROVIDER";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KvProviderRollout {
    optimized_provider_enabled: bool,
}

impl KvProviderRollout {
    pub(crate) fn from_process_env() -> Result<Self> {
        Self::from_disable_optimized_value(
            std::env::var_os(DISABLE_OPTIMIZED_KV_PROVIDER_ENV).as_deref(),
        )
    }

    fn from_disable_optimized_value(value: Option<&OsStr>) -> Result<Self> {
        let disable_optimized = match value {
            None => false,
            Some(value) => {
                let value = value.to_str().ok_or_else(|| {
                    invalid(format!(
                        "{DISABLE_OPTIMIZED_KV_PROVIDER_ENV} must contain UTF-8"
                    ))
                })?;
                match value.trim().to_ascii_lowercase().as_str() {
                    "1" | "true" | "yes" | "on" => true,
                    "0" | "false" | "no" | "off" => false,
                    _ => {
                        return Err(invalid(format!(
                            "{DISABLE_OPTIMIZED_KV_PROVIDER_ENV} must be one of \
                             1/true/yes/on or 0/false/no/off, got `{value}`"
                        )))
                    }
                }
            }
        };
        Ok(Self {
            optimized_provider_enabled: !disable_optimized,
        })
    }

    pub(crate) const fn optimized_provider_enabled(self) -> bool {
        self.optimized_provider_enabled
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvProviderEligibility {
    PortableRouteValidated,
    OptimizedEligibleUnverified,
}

/// Validate that the loaded-model route has a complete provider after backend
/// negotiation and before the model generation can become Ready.
///
/// An `Optimized` operation in the resolved plan is source/build eligibility,
/// not CUDA runtime or performance evidence. Only retained NVIDIA artifacts
/// may promote that separate evidence state.
pub(crate) fn validate_managed_state_plan_eligibility(
    variant: ModelVariant,
    capability: CapabilityKind,
    plan: &ResolvedStatePlan,
) -> Result<KvProviderEligibility> {
    let route_validated = matches!(
        (variant.family(), capability),
        (
            ModelFamily::Qwen3Chat
                | ModelFamily::Qwen35Chat
                | ModelFamily::Qwen38Chat
                | ModelFamily::Gemma3Chat
                | ModelFamily::Lfm2Chat,
            CapabilityKind::Chat
        ) | (ModelFamily::Qwen3Asr, CapabilityKind::Asr)
            | (ModelFamily::Lfm25Audio, CapabilityKind::Asr)
            | (
                ModelFamily::Qwen3Tts,
                CapabilityKind::Tts | CapabilityKind::StreamingTts
            )
    );
    if !route_validated {
        return Err(Error::ModelLoadError(format!(
            "managed KV provider route is not validated for model {variant}, capability {}",
            capability.as_str()
        )));
    }

    let optimized = plan.paged_attention.iter().any(|group| {
        let implementations = group.operations.implementations;
        implementations.write == PagedOperationImplementation::Optimized
            || implementations.prefill == PagedOperationImplementation::Optimized
            || implementations.decode == PagedOperationImplementation::Optimized
    });
    if optimized && plan.backend != BackendKind::Cuda {
        return Err(Error::ModelLoadError(format!(
            "optimized managed KV provider is not eligible for {:?}",
            plan.backend
        )));
    }

    Ok(if optimized {
        KvProviderEligibility::OptimizedEligibleUnverified
    } else {
        KvProviderEligibility::PortableRouteValidated
    })
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::state::{negotiate_state_plan, StateBackendPlanRequest};
    use crate::kv::v2::test_contract;

    #[test]
    fn optimized_provider_is_enabled_without_an_operator_override() {
        assert!(KvProviderRollout::from_disable_optimized_value(None)
            .unwrap()
            .optimized_provider_enabled());
    }

    #[test]
    fn optimized_provider_kill_switch_accepts_only_explicit_booleans() {
        for value in ["1", "true", "YES", "on"] {
            assert!(
                !KvProviderRollout::from_disable_optimized_value(Some(OsStr::new(value)))
                    .unwrap()
                    .optimized_provider_enabled(),
                "{value}"
            );
        }
        for value in ["0", "false", "NO", "off"] {
            assert!(
                KvProviderRollout::from_disable_optimized_value(Some(OsStr::new(value)))
                    .unwrap()
                    .optimized_provider_enabled(),
                "{value}"
            );
        }

        let error =
            KvProviderRollout::from_disable_optimized_value(Some(OsStr::new("maybe"))).unwrap_err();
        assert!(error
            .to_string()
            .contains(DISABLE_OPTIMIZED_KV_PROVIDER_ENV));
        assert!(error.to_string().contains("1/true/yes/on"));
    }

    #[test]
    fn managed_provider_promotion_is_bound_to_exact_model_capability_cells() {
        let plan = negotiate_state_plan(
            &test_contract(),
            &StateBackendPlanRequest {
                backend: BackendKind::Cpu,
                device_ordinal: None,
                page_tokens_hint: Some(16),
                storage_dtype_hint: None,
            },
        )
        .unwrap();

        assert_eq!(
            validate_managed_state_plan_eligibility(
                ModelVariant::Qwen306B,
                CapabilityKind::Chat,
                &plan,
            )
            .unwrap(),
            KvProviderEligibility::PortableRouteValidated
        );
        assert_eq!(
            validate_managed_state_plan_eligibility(
                ModelVariant::Lfm2512BInstructGguf,
                CapabilityKind::Chat,
                &plan,
            )
            .unwrap(),
            KvProviderEligibility::PortableRouteValidated
        );
        assert_eq!(
            validate_managed_state_plan_eligibility(
                ModelVariant::Lfm25Audio15BGguf,
                CapabilityKind::Asr,
                &plan,
            )
            .unwrap(),
            KvProviderEligibility::PortableRouteValidated
        );
        let error = validate_managed_state_plan_eligibility(
            ModelVariant::WhisperLargeV3Turbo,
            CapabilityKind::Asr,
            &plan,
        )
        .unwrap_err();
        assert!(error.to_string().contains("not validated"));
    }
}
