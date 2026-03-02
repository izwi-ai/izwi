use crate::catalog::ModelFamily;
use crate::error::Result;
use crate::model::ModelStatus;
use crate::model::ModelVariant;
use crate::runtime::service::RuntimeService;

impl RuntimeService {
    /// Unload a model from memory.
    pub async fn unload_model(&self, variant: ModelVariant) -> Result<()> {
        match variant.family() {
            ModelFamily::Qwen3Asr | ModelFamily::ParakeetAsr | ModelFamily::Qwen3ForcedAligner => {
                self.model_registry.unload_asr(variant).await;
            }
            ModelFamily::SortformerDiarization => {
                self.model_registry.unload_diarization(variant).await;
            }
            ModelFamily::Qwen3Chat | ModelFamily::Qwen35Chat | ModelFamily::Gemma3Chat => {
                self.model_registry.unload_chat(variant).await;
            }
            ModelFamily::Lfm2Audio => {
                self.model_registry.unload_lfm2(variant).await;
                self.clear_active_tts_variant().await;
            }
            ModelFamily::Voxtral => {
                self.model_registry.unload_voxtral(variant).await;
            }
            ModelFamily::Qwen3Tts => {
                let mut model_guard = self.tts_model.write().await;
                *model_guard = None;
                drop(model_guard);
                self.clear_active_tts_variant().await;
            }
            ModelFamily::KokoroTts => {
                self.model_registry.unload_kokoro(variant).await;
                self.clear_active_tts_variant().await;
            }
            ModelFamily::Tokenizer => {
                let mut tokenizer_guard = self.tokenizer.write().await;
                *tokenizer_guard = None;
            }
        }

        self.model_manager.unload_model(variant).await
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
