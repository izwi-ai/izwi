use crate::catalog::ModelFamily;
use crate::error::Result;
use crate::model::ModelStatus;
use crate::model::ModelVariant;
use crate::models::architectures::qwen35::chat::clear_qwen35_transform_cache_for_model_dir;
use crate::models::shared::memory::metal::MetalPoolManager;
use crate::runtime::service::RuntimeService;
use tracing::warn;

impl RuntimeService {
    /// Unload a model from memory.
    pub async fn unload_model(&self, variant: ModelVariant) -> Result<()> {
        let model_dir = self
            .model_manager
            .get_model_info(variant)
            .await
            .and_then(|info| info.local_path);
        let _ = self.core_engine.abort_requests_for_variant(variant).await;

        match variant.family() {
            ModelFamily::Qwen3Asr
            | ModelFamily::ParakeetAsr
            | ModelFamily::WhisperAsr
            | ModelFamily::Qwen3ForcedAligner => {
                self.model_registry.unload_asr(variant).await;
            }
            ModelFamily::SortformerDiarization => {
                self.model_registry.unload_diarization(variant).await;
            }
            ModelFamily::Qwen3Chat
            | ModelFamily::Qwen35Chat
            | ModelFamily::Lfm2Chat
            | ModelFamily::Gemma3Chat => {
                self.model_registry.unload_chat(variant).await;
            }
            ModelFamily::Lfm2Audio => {
                self.model_registry.unload_lfm2(variant).await;
                self.clear_active_tts_variant(variant).await;
            }
            ModelFamily::Voxtral => {
                self.model_registry.unload_voxtral(variant).await;
            }
            ModelFamily::Qwen3Tts => {
                self.model_registry.unload_qwen_tts(variant).await;
                self.clear_active_tts_variant(variant).await;
            }
            ModelFamily::KokoroTts => {
                self.model_registry.unload_kokoro(variant).await;
                self.clear_active_tts_variant(variant).await;
            }
            ModelFamily::Tokenizer => {
                let mut tokenizer_guard = self.tokenizer.write().await;
                *tokenizer_guard = None;
            }
        }

        self.model_manager.unload_model(variant).await?;

        if matches!(variant.family(), ModelFamily::Qwen35Chat) {
            if let Some(model_dir) = model_dir.as_deref() {
                match clear_qwen35_transform_cache_for_model_dir(model_dir) {
                    Ok(cleared) if cleared > 0 => {
                        tracing::info!(
                            "Cleared {} cached Qwen3.5 transformed tensors while unloading {}",
                            cleared,
                            variant
                        );
                    }
                    Ok(_) => {}
                    Err(err) => warn!(
                        "Failed to clear scoped Qwen3.5 transform cache for {} ({}): {}",
                        variant,
                        model_dir.display(),
                        err
                    ),
                }
            }
        }

        let has_other_loaded_models =
            self.model_manager
                .list_models()
                .await
                .into_iter()
                .any(|info| {
                    info.variant != variant
                        && matches!(info.status, ModelStatus::Ready | ModelStatus::Loading)
                });
        if !has_other_loaded_models {
            MetalPoolManager::global().clear_all();
        }

        Ok(())
    }

    /// Unload every model currently resident in memory.
    pub async fn unload_all_models(&self) -> Result<usize> {
        let loaded_variants = self
            .model_manager
            .list_models()
            .await
            .into_iter()
            .filter(|info| matches!(info.status, ModelStatus::Ready | ModelStatus::Loading))
            .map(|info| info.variant)
            .collect::<Vec<_>>();

        let mut unloaded_count = 0usize;
        for variant in loaded_variants {
            self.unload_model(variant).await?;
            unloaded_count += 1;
        }

        Ok(unloaded_count)
    }
}
