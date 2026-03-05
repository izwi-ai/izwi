//! Model lifecycle management

use futures::stream::{self, StreamExt};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use tracing::info;

use crate::artifacts::ModelWeights;
use crate::catalog::{ModelInfo, ModelStatus, ModelVariant};
use crate::config::EngineConfig;
use crate::error::{Error, Result};
use crate::model::download::{DownloadProgress, DownloadState, ModelDownloader};

/// Manages model downloading, loading, and lifecycle
pub struct ModelManager {
    _config: EngineConfig,
    downloader: ModelDownloader,
    models: RwLock<HashMap<ModelVariant, ModelState>>,
}

struct ModelState {
    info: ModelInfo,
    weights: Option<Arc<ModelWeights>>,
}

impl ModelManager {
    /// Create a new model manager
    pub fn new(config: EngineConfig) -> Result<Self> {
        let downloader = ModelDownloader::new(config.models_dir.clone())?;

        // Initialize model states
        let mut models = HashMap::new();
        for variant in ModelVariant::all() {
            let mut info = ModelInfo::new(*variant);

            // Check if already downloaded
            if downloader.is_downloaded(*variant) {
                info.status = ModelStatus::Downloaded;
                info.local_path = Some(downloader.model_path(*variant));
                info.size_bytes = downloader.get_cached_size(*variant);
            } else {
                // Use built-in estimates until Hugging Face totals are resolved.
                info.size_bytes = Some(variant.estimated_size());
            }

            models.insert(
                *variant,
                ModelState {
                    info,
                    weights: None,
                },
            );
        }

        Ok(Self {
            _config: config,
            downloader,
            models: RwLock::new(models),
        })
    }

    async fn refresh_model_states(&self) {
        let variants: Vec<(ModelVariant, ModelStatus)> = {
            let models = self.models.read().await;
            models
                .iter()
                .map(|(variant, state)| (*variant, state.info.status))
                .collect()
        };

        let updates = stream::iter(variants.into_iter())
            .filter(|(_, status)| {
                futures::future::ready(
                    *status != ModelStatus::Ready && *status != ModelStatus::Loading,
                )
            })
            .map(|(variant, _)| async move {
                let download_state = self.downloader.state_manager().get_state(variant).await;
                let latest_progress = if download_state == DownloadState::Downloading {
                    self.downloader.get_latest_progress(variant).await
                } else {
                    None
                };

                let is_downloaded = self.downloader.is_downloaded(variant);
                let size_bytes = if is_downloaded {
                    self.downloader.get_cached_size(variant)
                } else if let Some(progress) = &latest_progress {
                    if progress.total_bytes > 0 {
                        Some(progress.total_bytes)
                    } else {
                        Some(self.downloader.expected_size_bytes(variant).await)
                    }
                } else {
                    Some(self.downloader.expected_size_bytes(variant).await)
                };

                (
                    variant,
                    download_state,
                    latest_progress,
                    is_downloaded,
                    size_bytes,
                )
            })
            .buffer_unordered(8)
            .collect::<Vec<_>>()
            .await;

        let mut models = self.models.write().await;
        for (variant, download_state, latest_progress, is_downloaded, size_bytes) in updates {
            let Some(state) = models.get_mut(&variant) else {
                continue;
            };

            if state.info.status == ModelStatus::Ready || state.info.status == ModelStatus::Loading
            {
                continue;
            }

            if download_state == DownloadState::Downloading {
                state.info.status = ModelStatus::Downloading;
                if let Some(progress) = latest_progress {
                    state.info.download_progress = Some(progress.total_percent());
                } else if state.info.download_progress.is_none() {
                    state.info.download_progress = Some(0.0);
                }
                state.info.size_bytes = size_bytes.or(Some(variant.estimated_size()));
                state.info.error_message = None;
                continue;
            }

            if is_downloaded {
                state.info.status = ModelStatus::Downloaded;
                state.info.local_path = Some(self.downloader.model_path(variant));
                state.info.size_bytes = size_bytes;
                state.info.download_progress = Some(100.0);
                state.info.error_message = None;
                continue;
            }

            match download_state {
                DownloadState::Error => {
                    state.info.status = ModelStatus::Error;
                    state.info.local_path = None;
                    state.info.size_bytes = size_bytes.or(Some(variant.estimated_size()));
                    state.info.download_progress = None;
                    state.info.error_message = Some("Download failed".to_string());
                }
                DownloadState::Downloaded | DownloadState::NotDownloaded => {
                    state.info.status = ModelStatus::NotDownloaded;
                    state.info.local_path = None;
                    state.info.size_bytes = size_bytes.or(Some(variant.estimated_size()));
                    state.info.download_progress = None;
                    state.info.error_message = None;
                }
                DownloadState::Downloading => {}
            }
        }
    }

    /// Get list of all available models with their status
    pub async fn list_models(&self) -> Vec<ModelInfo> {
        self.refresh_model_states().await;
        let models = self.models.read().await;
        models.values().map(|s| s.info.clone()).collect()
    }

    /// Get info for a specific model
    pub async fn get_model_info(&self, variant: ModelVariant) -> Option<ModelInfo> {
        self.refresh_model_states().await;
        let models = self.models.read().await;
        models.get(&variant).map(|s| s.info.clone())
    }

    /// Download a model from HuggingFace
    pub async fn download_model(&self, variant: ModelVariant) -> Result<PathBuf> {
        if !variant.is_enabled() {
            return Err(Error::ConfigError(format!(
                "Model {} is disabled and cannot be downloaded",
                variant.dir_name()
            )));
        }

        // Update status to downloading
        {
            let mut models = self.models.write().await;
            if let Some(state) = models.get_mut(&variant) {
                state.info.status = ModelStatus::Downloading;
                state.info.download_progress = Some(0.0);
            }
        }

        // Perform download (now async - no spawn_blocking needed)
        let result = self.downloader.download(variant).await?;

        // Update status
        {
            let mut models = self.models.write().await;
            if let Some(state) = models.get_mut(&variant) {
                state.info.status = ModelStatus::Downloaded;
                state.info.local_path = Some(result.clone());
                state.info.download_progress = Some(100.0);
                state.info.size_bytes = self.downloader.get_cached_size(variant);
            }
        }

        Ok(result)
    }

    /// Download a model with progress reporting
    pub async fn download_model_with_progress(
        &self,
        variant: ModelVariant,
        progress_tx: broadcast::Sender<DownloadProgress>,
    ) -> Result<PathBuf> {
        if !variant.is_enabled() {
            return Err(Error::ConfigError(format!(
                "Model {} is disabled and cannot be downloaded",
                variant.dir_name()
            )));
        }

        // Update status
        {
            let mut models = self.models.write().await;
            if let Some(state) = models.get_mut(&variant) {
                state.info.status = ModelStatus::Downloading;
            }
        }

        let result = self
            .downloader
            .download_with_progress(variant, progress_tx)
            .await?;

        // Update status
        {
            let mut models = self.models.write().await;
            if let Some(state) = models.get_mut(&variant) {
                state.info.status = ModelStatus::Downloaded;
                state.info.local_path = Some(result.clone());
                state.info.size_bytes = self.downloader.get_cached_size(variant);
            }
        }

        Ok(result)
    }

    /// Load a model into memory
    pub async fn load_model(&self, variant: ModelVariant) -> Result<Arc<ModelWeights>> {
        // Check if already loaded
        {
            let models = self.models.read().await;
            if let Some(state) = models.get(&variant) {
                if let Some(ref weights) = state.weights {
                    return Ok(weights.clone());
                }
            }
        }

        // Get model path
        let model_path = {
            let models = self.models.read().await;
            models
                .get(&variant)
                .and_then(|s| s.info.local_path.clone())
                .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?
        };

        // Update status
        {
            let mut models = self.models.write().await;
            if let Some(state) = models.get_mut(&variant) {
                state.info.status = ModelStatus::Loading;
            }
        }

        info!("Loading model {} from {:?}", variant, model_path);

        // Load weights (blocking operation)
        let weights = tokio::task::spawn_blocking(move || ModelWeights::load(&model_path))
            .await
            .map_err(|e| Error::ModelLoadError(e.to_string()))??;

        let weights = Arc::new(weights);

        // Store loaded weights
        {
            let mut models = self.models.write().await;
            if let Some(state) = models.get_mut(&variant) {
                state.info.status = ModelStatus::Ready;
                state.weights = Some(weights.clone());
            }
        }

        info!("Model {} loaded successfully", variant);
        Ok(weights)
    }

    /// Unload a model from memory
    pub async fn unload_model(&self, variant: ModelVariant) -> Result<()> {
        let mut models = self.models.write().await;
        if let Some(state) = models.get_mut(&variant) {
            state.weights = None;
            state.info.status = if state.info.local_path.is_some() {
                ModelStatus::Downloaded
            } else {
                ModelStatus::NotDownloaded
            };
        }
        Ok(())
    }

    /// Mark a model as loaded without storing weights (native implementations).
    pub async fn mark_loaded(&self, variant: ModelVariant) {
        let mut models = self.models.write().await;
        if let Some(state) = models.get_mut(&variant) {
            state.info.status = ModelStatus::Ready;
            state.weights = None;
        }
    }

    /// Get loaded model weights
    pub async fn get_weights(&self, variant: ModelVariant) -> Option<Arc<ModelWeights>> {
        let models = self.models.read().await;
        models.get(&variant).and_then(|s| s.weights.clone())
    }

    /// Check if model is ready for inference
    pub async fn is_ready(&self, variant: ModelVariant) -> bool {
        let models = self.models.read().await;
        models
            .get(&variant)
            .map(|s| s.info.status == ModelStatus::Ready)
            .unwrap_or(false)
    }

    /// Delete downloaded model files
    pub async fn delete_model(&self, variant: ModelVariant) -> Result<()> {
        // Ensure background download is stopped before we remove files.
        if self.downloader.is_download_active(variant).await {
            let _ = self.downloader.cancel_download(variant).await;
        }

        // Unload first
        self.unload_model(variant).await?;

        let model_path = self.downloader.model_path(variant);
        if model_path.exists() {
            std::fs::remove_dir_all(&model_path)?;
        }

        // Update status
        {
            let mut models = self.models.write().await;
            if let Some(state) = models.get_mut(&variant) {
                state.info = ModelInfo::new(variant);
            }
        }

        Ok(())
    }

    /// Start a non-blocking background download
    pub async fn spawn_download(
        &self,
        variant: ModelVariant,
    ) -> Result<broadcast::Receiver<DownloadProgress>> {
        if !variant.is_enabled() {
            return Err(Error::ConfigError(format!(
                "Model {} is disabled and cannot be downloaded",
                variant.dir_name()
            )));
        }

        // Update status
        {
            let mut models = self.models.write().await;
            if let Some(state) = models.get_mut(&variant) {
                state.info.status = ModelStatus::Downloading;
            }
        }

        let progress_rx = self.downloader.spawn_download(variant).await?;

        Ok(progress_rx)
    }

    /// Subscribe to progress updates for an active download
    pub async fn subscribe_progress(
        &self,
        variant: ModelVariant,
    ) -> Result<broadcast::Receiver<DownloadProgress>> {
        self.downloader.subscribe_progress(variant).await
    }

    /// Cancel an active download
    pub async fn cancel_download(&self, variant: ModelVariant) -> Result<()> {
        // Update status first
        {
            let mut models = self.models.write().await;
            if let Some(state) = models.get_mut(&variant) {
                state.info.status = ModelStatus::NotDownloaded;
                state.info.download_progress = None;
            }
        }

        self.downloader.cancel_download(variant).await
    }

    /// Check if a download is currently active
    pub async fn is_download_active(&self, variant: ModelVariant) -> bool {
        self.downloader.is_download_active(variant).await
    }

    /// Wait for an active download to complete
    pub async fn wait_for_download(&self, variant: ModelVariant) -> Result<Option<PathBuf>> {
        self.downloader.wait_for_download(variant).await
    }

    /// Get download state
    pub async fn get_download_state(
        &self,
        variant: ModelVariant,
    ) -> super::download::DownloadState {
        self.downloader.state_manager().get_state(variant).await
    }
}

// Make downloader cloneable for async tasks
impl Clone for ModelDownloader {
    fn clone(&self) -> Self {
        ModelDownloader::new(self.models_dir.clone()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EngineConfig;
    use std::path::PathBuf;
    use uuid::Uuid;

    #[tokio::test]
    async fn refresh_keeps_downloading_status_when_partial_files_exist() {
        let temp_dir = std::env::temp_dir().join(format!("izwi-manager-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_dir).unwrap();

        let config = EngineConfig {
            models_dir: PathBuf::from(&temp_dir),
            ..EngineConfig::default()
        };
        let manager = ModelManager::new(config).unwrap();
        let variant = ModelVariant::Qwen306B4Bit;

        let model_dir = temp_dir.join(variant.dir_name());
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("config.json"), "{}").unwrap();
        std::fs::write(model_dir.join("vocab.json"), "{}").unwrap();
        std::fs::write(model_dir.join("merges.txt"), "").unwrap();
        std::fs::write(model_dir.join("model.safetensors"), [0u8]).unwrap();

        manager
            .downloader
            .state_manager()
            .set_state(variant, DownloadState::Downloading)
            .await;

        let models = manager.list_models().await;
        let model = models.into_iter().find(|m| m.variant == variant).unwrap();

        assert_eq!(model.status, ModelStatus::Downloading);
        assert_eq!(model.download_progress, Some(0.0));

        std::fs::remove_dir_all(&temp_dir).unwrap();
    }
}
