use tracing::info;

use crate::catalog::ModelFamily;
use crate::catalog::ModelVariant;
use crate::error::Result;
use crate::runtime::lifecycle::phases::AcquiredModelLoad;
use crate::runtime::service::RuntimeService;
use crate::tokenizer::Tokenizer;

pub(super) enum InstantiatedPayload {
    None,
    Tokenizer(Option<Tokenizer>),
}

pub(super) struct InstantiatedModelLoad {
    pub family: ModelFamily,
    pub variant: ModelVariant,
    pub model_path: std::path::PathBuf,
    pub payload: InstantiatedPayload,
}

impl RuntimeService {
    pub(super) async fn instantiate_model(
        &self,
        acquired: AcquiredModelLoad,
    ) -> Result<InstantiatedModelLoad> {
        let AcquiredModelLoad {
            variant,
            model_path,
        } = acquired;
        let family = variant.family();

        let payload = match family {
            ModelFamily::Qwen3Asr
            | ModelFamily::ParakeetAsr
            | ModelFamily::WhisperAsr
            | ModelFamily::Qwen3ForcedAligner => {
                self.model_registry.load_asr(variant, &model_path).await?;
                InstantiatedPayload::None
            }
            ModelFamily::SortformerDiarization => {
                self.model_registry
                    .load_diarization(variant, &model_path)
                    .await?;
                InstantiatedPayload::None
            }
            ModelFamily::Qwen3Chat
            | ModelFamily::Qwen35Chat
            | ModelFamily::Lfm2Chat
            | ModelFamily::Gemma3Chat => {
                self.model_registry.load_chat(variant, &model_path).await?;
                InstantiatedPayload::None
            }
            ModelFamily::Voxtral => {
                self.model_registry
                    .load_voxtral(variant, &model_path)
                    .await?;
                InstantiatedPayload::None
            }
            ModelFamily::KokoroTts => {
                self.model_registry
                    .load_kokoro(variant, &model_path)
                    .await?;
                InstantiatedPayload::None
            }
            ModelFamily::Qwen3Tts => {
                info!("Loading native TTS model {variant} via shared model registry");
                self.model_registry
                    .load_qwen_tts(
                        variant,
                        &model_path,
                        self.config.kv_page_size,
                        &self.config.kv_cache_dtype,
                    )
                    .await?;
                InstantiatedPayload::None
            }
            ModelFamily::Tokenizer => {
                let tokenizer = match Tokenizer::from_path(&model_path) {
                    Ok(tokenizer) => Some(tokenizer),
                    Err(err) => {
                        tracing::warn!("Failed to load tokenizer from model directory: {}", err);
                        None
                    }
                };
                InstantiatedPayload::Tokenizer(tokenizer)
            }
        };

        Ok(InstantiatedModelLoad {
            family,
            variant,
            model_path,
            payload,
        })
    }
}
