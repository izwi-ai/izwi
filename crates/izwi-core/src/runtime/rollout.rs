//! Fail-closed rollout policy for physical inference-state providers.
//!
//! Provider selection is part of the resolved state plan, not a best-effort
//! runtime fallback. The process-level overrides here only fall back: one
//! demotes a certified optimized provider to its certified portable provider,
//! the other restores a host synchronization after every Metal KV mutation.

use std::ffi::OsStr;

use crate::backends::BackendKind;
use crate::catalog::{ModelFamily, ModelVariant};
use crate::error::{Error, Result};
use crate::kv::v2::{PagedOperationImplementation, ResolvedStatePlan};
use crate::runtime::adapters::CapabilityKind;

pub(crate) const DISABLE_OPTIMIZED_KV_PROVIDER_ENV: &str = "IZWI_KV_DISABLE_OPTIMIZED_PROVIDER";
/// Kill switch for Metal KV epoch fences: when set, every Metal KV mutation
/// drains the device before returning its fence, as before epoch fences.
pub(crate) const METAL_EAGER_KV_FENCES_ENV: &str = "IZWI_KV_METAL_EAGER_FENCES";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KvProviderRollout {
    optimized_provider_enabled: bool,
    metal_eager_fences: bool,
}

impl KvProviderRollout {
    pub(crate) fn from_process_env() -> Result<Self> {
        Self::from_values(
            std::env::var_os(DISABLE_OPTIMIZED_KV_PROVIDER_ENV).as_deref(),
            std::env::var_os(METAL_EAGER_KV_FENCES_ENV).as_deref(),
        )
    }

    fn from_values(
        disable_optimized: Option<&OsStr>,
        metal_eager_fences: Option<&OsStr>,
    ) -> Result<Self> {
        Ok(Self {
            optimized_provider_enabled: !strict_bool(
                DISABLE_OPTIMIZED_KV_PROVIDER_ENV,
                disable_optimized,
            )?,
            metal_eager_fences: strict_bool(METAL_EAGER_KV_FENCES_ENV, metal_eager_fences)?,
        })
    }

    pub(crate) const fn optimized_provider_enabled(self) -> bool {
        self.optimized_provider_enabled
    }

    // Read by the accelerator KV backend, which only Metal and CUDA builds compile.
    #[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
    pub(crate) const fn metal_eager_fences(self) -> bool {
        self.metal_eager_fences
    }
}

/// An unset or empty switch is off; a set one must be an explicit boolean.
fn strict_bool(name: &str, value: Option<&OsStr>) -> Result<bool> {
    let Some(value) = value else {
        return Ok(false);
    };
    let value = value
        .to_str()
        .ok_or_else(|| invalid(format!("{name} must contain UTF-8")))?;
    if value.trim().is_empty() {
        return Ok(false);
    }
    izwi_env::parse_bool(name, value).map_err(|error| invalid(error.to_string()))
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
            ModelFamily::Qwen3MoeChat
                | ModelFamily::Qwen3Chat
                | ModelFamily::Qwen35Chat
                | ModelFamily::Qwen35MoeChat
                | ModelFamily::Qwen38Chat
                | ModelFamily::Gemma3Chat
                | ModelFamily::Lfm2Chat,
            CapabilityKind::Chat
        ) | (ModelFamily::Qwen3Asr, CapabilityKind::Asr)
            | (ModelFamily::VibeVoiceAsr, CapabilityKind::Asr)
            | (
                ModelFamily::Lfm25Audio,
                CapabilityKind::Asr | CapabilityKind::Tts
            )
            | (
                ModelFamily::Qwen3Tts,
                CapabilityKind::Tts | CapabilityKind::StreamingTts
            )
            | (ModelFamily::VibeVoiceTts, CapabilityKind::Tts)
            | (ModelFamily::VoxtralTts, CapabilityKind::Tts)
            | (
                ModelFamily::FishS2Tts,
                CapabilityKind::Tts | CapabilityKind::StreamingTts
            )
            | (ModelFamily::ChatterboxTts, CapabilityKind::Tts)
            | (ModelFamily::Maya1Tts, CapabilityKind::Tts)
            | (ModelFamily::GraniteSpeechAsr, CapabilityKind::Asr)
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
        let rollout = KvProviderRollout::from_values(None, None).unwrap();
        assert!(rollout.optimized_provider_enabled());
        assert!(!rollout.metal_eager_fences());
    }

    #[test]
    fn optimized_provider_kill_switch_accepts_only_explicit_booleans() {
        for value in ["1", "true", "YES", "on"] {
            assert!(
                !KvProviderRollout::from_values(Some(OsStr::new(value)), None)
                    .unwrap()
                    .optimized_provider_enabled(),
                "{value}"
            );
        }
        for value in ["0", "false", "NO", "off"] {
            assert!(
                KvProviderRollout::from_values(Some(OsStr::new(value)), None)
                    .unwrap()
                    .optimized_provider_enabled(),
                "{value}"
            );
        }

        let error = KvProviderRollout::from_values(Some(OsStr::new("maybe")), None).unwrap_err();
        assert!(error
            .to_string()
            .contains(DISABLE_OPTIMIZED_KV_PROVIDER_ENV));
        assert!(error.to_string().contains("yes, no, on or off"));
    }

    #[test]
    fn metal_eager_fence_kill_switch_accepts_only_explicit_booleans() {
        for value in ["1", "true", "YES", "on"] {
            let rollout = KvProviderRollout::from_values(None, Some(OsStr::new(value))).unwrap();
            assert!(rollout.metal_eager_fences(), "{value}");
            assert!(rollout.optimized_provider_enabled(), "{value}");
        }
        for value in ["0", "false", "NO", "off"] {
            assert!(
                !KvProviderRollout::from_values(None, Some(OsStr::new(value)))
                    .unwrap()
                    .metal_eager_fences(),
                "{value}"
            );
        }

        let error = KvProviderRollout::from_values(None, Some(OsStr::new("maybe"))).unwrap_err();
        assert!(error.to_string().contains(METAL_EAGER_KV_FENCES_ENV));
        assert!(error.to_string().contains("yes, no, on or off"));
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
                ModelVariant::Qwen36Moe35BA3BFp8,
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
        assert_eq!(
            validate_managed_state_plan_eligibility(
                ModelVariant::Lfm25Audio15BGguf,
                CapabilityKind::Tts,
                &plan,
            )
            .unwrap(),
            KvProviderEligibility::PortableRouteValidated
        );
        assert_eq!(
            validate_managed_state_plan_eligibility(
                ModelVariant::VibeVoice15BTts,
                CapabilityKind::Tts,
                &plan,
            )
            .unwrap(),
            KvProviderEligibility::PortableRouteValidated
        );
        assert_eq!(
            validate_managed_state_plan_eligibility(
                ModelVariant::FishAudioS2Pro,
                CapabilityKind::Tts,
                &plan,
            )
            .unwrap(),
            KvProviderEligibility::PortableRouteValidated
        );
        assert_eq!(
            validate_managed_state_plan_eligibility(
                ModelVariant::FishAudioS2Pro,
                CapabilityKind::StreamingTts,
                &plan,
            )
            .unwrap(),
            KvProviderEligibility::PortableRouteValidated
        );
        assert_eq!(
            validate_managed_state_plan_eligibility(
                ModelVariant::GraniteSpeech412BPlus,
                CapabilityKind::Asr,
                &plan,
            )
            .unwrap(),
            KvProviderEligibility::PortableRouteValidated
        );
        assert_eq!(
            validate_managed_state_plan_eligibility(
                ModelVariant::Voxtral4BTts2603,
                CapabilityKind::Tts,
                &plan,
            )
            .unwrap(),
            KvProviderEligibility::PortableRouteValidated
        );
        assert_eq!(
            validate_managed_state_plan_eligibility(
                ModelVariant::VibeVoiceAsr,
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
