use std::path::PathBuf;

use tracing::info;

use crate::backends::BackendPlan;
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::runtime::service::RuntimeService;

pub(super) struct ResolvedModelLoad {
    pub variant: ModelVariant,
    pub backend_plan: BackendPlan,
}

pub(super) struct AcquiredModelLoad {
    pub variant: ModelVariant,
    pub model_path: PathBuf,
}

impl RuntimeService {
    pub(super) async fn resolve_model_load(
        &self,
        variant: ModelVariant,
    ) -> Result<ResolvedModelLoad> {
        let backend_plan = self.backend_router.select(variant);
        info!(
            "Selected backend {:?} for {} ({})",
            backend_plan.backend, variant, backend_plan.reason
        );

        Ok(ResolvedModelLoad {
            variant,
            backend_plan,
        })
    }

    pub(super) async fn acquire_model_artifacts(
        &self,
        resolved: ResolvedModelLoad,
    ) -> Result<AcquiredModelLoad> {
        let variant = resolved.variant;
        let _backend = resolved.backend_plan.backend;

        if let Some(model_path) = self
            .model_manager
            .get_model_info(variant)
            .await
            .and_then(|i| i.local_path)
        {
            return Ok(AcquiredModelLoad {
                variant,
                model_path,
            });
        }

        if self.model_manager.is_download_active(variant).await {
            if let Some(model_path) = self.model_manager.wait_for_download(variant).await? {
                return Ok(AcquiredModelLoad {
                    variant,
                    model_path,
                });
            }
        }

        Err(Error::ModelNotFound(format!(
            "Model {} not downloaded. Please download it first.",
            variant
        )))
    }

    pub(super) async fn clear_active_tts_variant(&self) {
        let mut path_guard = self.loaded_model_path.write().await;
        *path_guard = None;

        let mut variant_guard = self.loaded_tts_variant.write().await;
        *variant_guard = None;
    }

    pub(super) async fn set_active_tts_variant(&self, variant: ModelVariant, model_path: PathBuf) {
        let mut path_guard = self.loaded_model_path.write().await;
        *path_guard = Some(model_path);

        let mut variant_guard = self.loaded_tts_variant.write().await;
        *variant_guard = Some(variant);
    }
}
