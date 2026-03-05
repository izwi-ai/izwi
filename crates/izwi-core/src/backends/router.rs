use crate::catalog::{InferenceBackendHint, ModelVariant};

use super::device::DeviceSelector;
use super::types::{BackendPreference, ExecutionBackend};

#[derive(Debug, Clone)]
pub struct BackendPlan {
    pub backend: ExecutionBackend,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct BackendRouter {
    default_backend: ExecutionBackend,
}

impl Default for BackendRouter {
    fn default() -> Self {
        Self {
            default_backend: ExecutionBackend::CandleNative,
        }
    }
}

impl BackendRouter {
    pub fn with_default(default_backend: ExecutionBackend) -> Self {
        Self { default_backend }
    }

    pub fn from_env_with_default(default_backend: ExecutionBackend) -> Self {
        let override_backend = std::env::var("IZWI_BACKEND")
            .ok()
            .as_deref()
            .and_then(BackendPreference::parse)
            .and_then(|preference| match preference {
                BackendPreference::Auto => None,
                _ => Some(Self::backend_for_preference(preference)),
            })
            .or_else(|| {
                std::env::var("IZWI_USE_METAL").ok().and_then(|raw| {
                    let value = raw.trim().to_ascii_lowercase();
                    if matches!(value.as_str(), "1" | "true" | "yes" | "on") {
                        Some(ExecutionBackend::CandleMetal)
                    } else {
                        None
                    }
                })
            });

        Self::with_default(override_backend.unwrap_or(default_backend))
    }

    pub fn from_env() -> Self {
        Self::from_env_with_default(ExecutionBackend::CandleNative)
    }

    pub fn from_preference(preference: BackendPreference) -> Self {
        Self::with_default(Self::backend_for_preference(preference))
    }

    fn backend_for_preference(preference: BackendPreference) -> ExecutionBackend {
        match preference {
            BackendPreference::Auto => Self::detect_auto_backend(),
            BackendPreference::Cpu => ExecutionBackend::CandleNative,
            BackendPreference::Metal => ExecutionBackend::CandleMetal,
            BackendPreference::Cuda => ExecutionBackend::CandleCuda,
        }
    }

    fn detect_auto_backend() -> ExecutionBackend {
        if let Ok(profile) = DeviceSelector::detect() {
            if profile.kind.is_metal() {
                return ExecutionBackend::CandleMetal;
            }
            if profile.kind.is_cuda() {
                return ExecutionBackend::CandleCuda;
            }
        }
        ExecutionBackend::CandleNative
    }

    pub fn default_backend(&self) -> ExecutionBackend {
        self.default_backend
    }

    pub fn select(&self, variant: ModelVariant) -> BackendPlan {
        let default_desc = match self.default_backend {
            ExecutionBackend::CandleMetal => "Metal backend",
            ExecutionBackend::CandleNative => "native CPU backend",
            ExecutionBackend::CandleCuda => "CUDA backend",
        };

        match variant.backend_hint() {
            InferenceBackendHint::CandleNative => BackendPlan {
                backend: self.default_backend,
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
        assert_eq!(router.default_backend(), ExecutionBackend::CandleMetal);

        std::env::remove_var("IZWI_BACKEND");
    }
}
