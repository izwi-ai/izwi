use crate::catalog::ModelFamily;
use crate::error::Result;
use crate::runtime::lifecycle::instantiate::{InstantiatedModelLoad, InstantiatedPayload};
use crate::runtime::service::RuntimeService;

impl RuntimeService {
    pub(super) async fn publish_loaded_model(
        &self,
        instantiated: InstantiatedModelLoad,
    ) -> Result<()> {
        let InstantiatedModelLoad {
            family,
            variant,
            model_path,
            payload,
        } = instantiated;

        match family {
            ModelFamily::Qwen3Asr
            | ModelFamily::ParakeetAsr
            | ModelFamily::Qwen3ForcedAligner
            | ModelFamily::SortformerDiarization
            | ModelFamily::Qwen3Chat
            | ModelFamily::Qwen35Chat
            | ModelFamily::Gemma3Chat
            | ModelFamily::Voxtral => {
                self.model_manager.mark_loaded(variant).await;
            }
            ModelFamily::Lfm2Audio => {
                // LFM2 owns active TTS routing and does not use the legacy Qwen slot.
                let mut tts_guard = self.tts_model.write().await;
                *tts_guard = None;
                drop(tts_guard);

                self.set_active_tts_variant(variant, model_path).await;
                self.model_manager.mark_loaded(variant).await;
            }
            ModelFamily::Qwen3Tts => {
                if let InstantiatedPayload::TtsModel(model) = payload {
                    let mut model_guard = self.tts_model.write().await;
                    *model_guard = Some(model);
                    drop(model_guard);
                }
                self.set_active_tts_variant(variant, model_path).await;
                self.model_manager.mark_loaded(variant).await;
            }
            ModelFamily::KokoroTts => {
                // Kokoro owns active TTS routing and does not use the legacy Qwen slot.
                let mut model_guard = self.tts_model.write().await;
                *model_guard = None;
                drop(model_guard);
                self.set_active_tts_variant(variant, model_path).await;
                self.model_manager.mark_loaded(variant).await;
            }
            ModelFamily::Tokenizer => {
                if let InstantiatedPayload::Tokenizer(Some(tokenizer)) = payload {
                    let mut guard = self.tokenizer.write().await;
                    *guard = Some(tokenizer);
                }

                let mut codec_guard = self.codec.write().await;
                codec_guard.load_weights(&model_path)?;

                self.model_manager.mark_loaded(variant).await;
            }
        }

        Ok(())
    }
}
