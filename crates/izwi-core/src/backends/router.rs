use crate::catalog::{InferenceBackendHint, ModelVariant};

use super::capabilities::BackendCapabilities;
use super::device::{DeviceProfile, DeviceSelector};
use super::types::{BackendContext, BackendPreference, BackendSelectionSource, ExecutionBackend};

#[derive(Debug, Clone)]
pub struct BackendPlan {
    pub backend: ExecutionBackend,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct BackendRouter {
    context: BackendContext,
}

impl Default for BackendRouter {
    fn default() -> Self {
        Self::from_context(Self::resolve_context(
            BackendPreference::Auto,
            BackendSelectionSource::Default,
        ))
    }
}

impl BackendRouter {
    pub fn from_context(context: BackendContext) -> Self {
        Self { context }
    }

    pub fn with_default(default_backend: ExecutionBackend) -> Self {
        Self::from_context(Self::resolve_context(
            BackendPreference::from(default_backend.kind()),
            BackendSelectionSource::Default,
        ))
    }

    pub fn env_preference() -> Option<BackendPreference> {
        std::env::var("IZWI_BACKEND")
            .ok()
            .as_deref()
            .and_then(BackendPreference::parse)
            .or_else(|| {
                std::env::var("IZWI_USE_METAL").ok().and_then(|raw| {
                    let value = raw.trim().to_ascii_lowercase();
                    if matches!(value.as_str(), "1" | "true" | "yes" | "on") {
                        Some(BackendPreference::Metal)
                    } else {
                        None
                    }
                })
            })
    }

    pub fn resolve_context(
        preference: BackendPreference,
        source: BackendSelectionSource,
    ) -> BackendContext {
        let capabilities = BackendCapabilities::detect();
        let device = DeviceSelector::detect_for_preference(preference)
            .unwrap_or_else(|_| DeviceProfile::cpu());
        let actual_backend = ExecutionBackend::from_kind(device.kind.into());
        let reason = match preference.requested_kind() {
            None => format!(
                "Selected {} backend via {} auto detection",
                actual_backend.kind().as_str(),
                source.as_str()
            ),
            Some(requested) if requested == actual_backend.kind() => format!(
                "Selected {} backend from {} {} preference",
                requested.as_str(),
                source.as_str(),
                preference.as_str()
            ),
            Some(requested) => format!(
                "Requested {} backend from {} {} preference was unavailable; fell back to {}",
                requested.as_str(),
                source.as_str(),
                preference.as_str(),
                actual_backend.kind().as_str()
            ),
        };

        BackendContext::new(preference, source, capabilities, device, reason)
    }

    pub fn resolve_context_for_kind(
        kind: super::types::BackendKind,
        source: BackendSelectionSource,
    ) -> BackendContext {
        Self::resolve_context(BackendPreference::from(kind), source)
    }

    pub fn resolve_context_from_env_or(
        preference: BackendPreference,
        source: BackendSelectionSource,
    ) -> BackendContext {
        Self::env_preference()
            .map(|env_preference| {
                Self::resolve_context(env_preference, BackendSelectionSource::Env)
            })
            .unwrap_or_else(|| Self::resolve_context(preference, source))
    }

    pub fn from_env_with_default(default_backend: ExecutionBackend) -> Self {
        Self::from_context(Self::env_preference().map_or_else(
            || {
                Self::resolve_context(
                    BackendPreference::from(default_backend.kind()),
                    BackendSelectionSource::Default,
                )
            },
            |preference| Self::resolve_context(preference, BackendSelectionSource::Env),
        ))
    }

    pub fn from_env() -> Self {
        Self::from_context(Self::resolve_context_from_env_or(
            BackendPreference::Auto,
            BackendSelectionSource::Default,
        ))
    }

    pub fn from_preference(preference: BackendPreference) -> Self {
        Self::from_context(Self::resolve_context(
            preference,
            BackendSelectionSource::Config,
        ))
    }

    pub fn context(&self) -> &BackendContext {
        &self.context
    }

    pub fn default_backend(&self) -> ExecutionBackend {
        self.context.execution_backend
    }

    pub fn select(&self, variant: ModelVariant) -> BackendPlan {
        let default_desc = match self.context.execution_backend {
            ExecutionBackend::CandleMetal => "Metal backend",
            ExecutionBackend::CandleNative => "native CPU backend",
            ExecutionBackend::CandleCuda => "CUDA backend",
        };

        match variant.backend_hint() {
            InferenceBackendHint::CandleNative => BackendPlan {
                backend: self.context.execution_backend,
                reason: format!(
                    "{} targets the native Candle execution path ({})",
                    variant.dir_name(),
                    default_desc
                ),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::ModelVariant;
    use crate::env_test_lock;

    #[test]
    fn preference_maps_to_expected_default_backend() {
        assert_eq!(
            BackendRouter::from_preference(BackendPreference::Cpu).default_backend(),
            ExecutionBackend::CandleNative
        );
        assert_eq!(
            BackendRouter::from_preference(BackendPreference::Metal).default_backend(),
            ExecutionBackend::CandleMetal
        );
        assert_eq!(
            BackendRouter::from_preference(BackendPreference::Cuda).default_backend(),
            ExecutionBackend::CandleCuda
        );
    }

    #[test]
    fn select_uses_router_default_backend() {
        let router = BackendRouter::from_preference(BackendPreference::Cpu);
        let plan = router.select(ModelVariant::Qwen3Tts12Hz06BBase);
        assert_eq!(plan.backend, ExecutionBackend::CandleNative);
    }

    #[test]
    fn auto_preference_selects_supported_backend() {
        let backend = BackendRouter::from_preference(BackendPreference::Auto).default_backend();
        assert!(matches!(
            backend,
            ExecutionBackend::CandleNative
                | ExecutionBackend::CandleMetal
                | ExecutionBackend::CandleCuda
        ));
    }

    #[test]
    fn env_auto_keeps_runtime_default_backend() {
        let _guard = env_test_lock().lock().expect("env lock poisoned");
        std::env::set_var("IZWI_BACKEND", "auto");
        std::env::remove_var("IZWI_USE_METAL");

        let router = BackendRouter::from_env_with_default(ExecutionBackend::CandleMetal);
        assert!(matches!(
            router.default_backend(),
            ExecutionBackend::CandleNative
                | ExecutionBackend::CandleMetal
                | ExecutionBackend::CandleCuda
        ));
        assert_eq!(router.context().preference, BackendPreference::Auto);
        assert_eq!(router.context().source, BackendSelectionSource::Env);

        std::env::remove_var("IZWI_BACKEND");
    }

    #[test]
    fn context_tracks_requested_backend_match() {
        let context =
            BackendRouter::resolve_context(BackendPreference::Cpu, BackendSelectionSource::Config);
        assert!(context.matches_preference());
        assert_eq!(context.backend_kind, crate::backends::BackendKind::Cpu);
    }
}
