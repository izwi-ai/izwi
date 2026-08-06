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
pub(crate) enum KvProviderCertification {
    PortableCertified,
    OptimizedCertified,
}

/// Certify the exact loaded-model route after backend negotiation and before
/// the model generation can become Ready.
///
/// Backend negotiation has already validated dtype, page size, attention
/// semantics, layout, and build availability. This final gate binds that
/// exact physical plan to the model/capability cells migrated to managed v2.
pub(crate) fn certify_managed_state_plan(
    variant: ModelVariant,
    capability: CapabilityKind,
    plan: &ResolvedStatePlan,
) -> Result<KvProviderCertification> {
    let route_certified = matches!(
        (variant.family(), capability),
        (
            ModelFamily::Qwen3Chat | ModelFamily::Qwen35Chat | ModelFamily::Gemma3Chat,
            CapabilityKind::Chat
        ) | (ModelFamily::Qwen3Asr, CapabilityKind::Asr)
            | (
                ModelFamily::Qwen3Tts,
                CapabilityKind::Tts | CapabilityKind::StreamingTts
            )
    );
    if !route_certified {
        return Err(Error::ModelLoadError(format!(
            "managed KV provider is not certified for model {variant}, capability {}",
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
            "optimized managed KV provider is not certified for {:?}",
            plan.backend
        )));
    }

    Ok(if optimized {
        KvProviderCertification::OptimizedCertified
    } else {
        KvProviderCertification::PortableCertified
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
            certify_managed_state_plan(ModelVariant::Qwen306B, CapabilityKind::Chat, &plan)
                .unwrap(),
            KvProviderCertification::PortableCertified
        );
        let error = certify_managed_state_plan(
            ModelVariant::WhisperLargeV3Turbo,
            CapabilityKind::Asr,
            &plan,
        )
        .unwrap_err();
        assert!(error.to_string().contains("not certified"));
    }
}
