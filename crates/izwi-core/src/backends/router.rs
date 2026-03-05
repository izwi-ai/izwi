use crate::catalog::{InferenceBackendHint, ModelVariant};

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
    pub fn from_env_with_default(default_backend: ExecutionBackend) -> Self {
        let override_backend = std::env::var("IZWI_BACKEND")
            .ok()
            .as_deref()
            .and_then(BackendPreference::parse)
            .map(Self::backend_for_preference)
            .or_else(|| {
                std::env::var("IZWI_USE_METAL")
                    .ok()
                    .and_then(|raw| {
                        let value = raw.trim().to_ascii_lowercase();
                        if matches!(value.as_str(), "1" | "true" | "yes" | "on") {
                            Some(ExecutionBackend::CandleMetal)
                        } else {
                            None
                        }
                    })
            });

        Self {
            default_backend: override_backend.unwrap_or(default_backend),
        }
    }

    pub fn from_env() -> Self {
        Self::from_env_with_default(ExecutionBackend::CandleNative)
    }

    pub fn from_preference(preference: BackendPreference) -> Self {
        Self {
            default_backend: Self::backend_for_preference(preference),
        }
    }

    fn backend_for_preference(preference: BackendPreference) -> ExecutionBackend {
        match preference {
            BackendPreference::Auto => ExecutionBackend::CandleNative,
            BackendPreference::Cpu => ExecutionBackend::CandleNative,
            BackendPreference::Metal => ExecutionBackend::CandleMetal,
            BackendPreference::Cuda => ExecutionBackend::CandleCuda,
        }
    }

    pub fn default_backend(&self) -> ExecutionBackend {
        self.default_backend
    }

    pub fn select(&self, variant: ModelVariant) -> BackendPlan {
        let default_desc = match self.default_backend {
            ExecutionBackend::CandleMetal => "Metal backend",
            ExecutionBackend::CandleNative => "native CPU backend",
            ExecutionBackend::CandleCuda => "CUDA backend",
            ExecutionBackend::MlxNative => "MLX runtime",
        };

        match variant.backend_hint() {
            InferenceBackendHint::MlxCandidate => BackendPlan {
                backend: self.default_backend,
                reason: format!(
                    "{} is MLX-compatible; using {}",
                    variant.dir_name(),
                    default_desc
                ),
            },
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
