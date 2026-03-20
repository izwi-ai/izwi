use crate::catalog::ModelFamily;
use crate::error::Result;
use crate::model::ModelStatus;
use crate::model::ModelVariant;
use crate::models::shared::memory::metal::MetalPoolManager;
use crate::runtime::service::RuntimeService;

impl RuntimeService {
    /// Unload a model from memory.
    pub async fn unload_model(&self, variant: ModelVariant) -> Result<()> {
        let _ = self.core_engine.abort_requests_for_variant(variant).await;

        match variant.family() {
            ModelFamily::ParakeetAsr
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
            ModelFamily::Voxtral => {
                self.model_registry.unload_voxtral(variant).await;
            }
            ModelFamily::Lfm25Audio => {
                self.model_registry.unload_audio_chat(variant).await;
                self.clear_active_tts_variant(variant).await;
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

        self.forget_model_usage(variant).await;

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
