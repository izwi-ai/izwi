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
            | ModelFamily::WhisperAsr
            | ModelFamily::Qwen3ForcedAligner
            | ModelFamily::SortformerDiarization
            | ModelFamily::Qwen3Chat
            | ModelFamily::Qwen35Chat
            | ModelFamily::Lfm2Chat
            | ModelFamily::Gemma3Chat
            | ModelFamily::Voxtral
            | ModelFamily::Lfm25Audio
            | ModelFamily::Qwen3Tts
            | ModelFamily::KokoroTts => {
                if matches!(
                    family,
                    ModelFamily::Qwen3Tts | ModelFamily::KokoroTts | ModelFamily::Lfm25Audio
                ) {
                    self.set_active_tts_variant(variant).await;
                }
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
