use tracing::info;

use crate::catalog::ModelFamily;
use crate::catalog::ModelVariant;
use crate::error::{Error, Result};
use crate::runtime::lifecycle::phases::AcquiredModelLoad;
use crate::runtime::service::RuntimeService;
use crate::runtime_models::architectures::qwen3::tts::Qwen3TtsModel;
use crate::tokenizer::Tokenizer;

type TtsLoaderFn =
    fn(&std::path::Path, crate::backends::DeviceProfile, usize, &str) -> Result<Qwen3TtsModel>;

struct TtsLoaderRegistration {
    name: &'static str,
    family: ModelFamily,
    loader: TtsLoaderFn,
}

fn load_qwen_tts_model(
    model_dir: &std::path::Path,
    device: crate::backends::DeviceProfile,
    kv_page_size: usize,
    kv_cache_dtype: &str,
) -> Result<Qwen3TtsModel> {
    Qwen3TtsModel::load(model_dir, device, kv_page_size, kv_cache_dtype)
}

const TTS_LOADER_REGISTRY: &[TtsLoaderRegistration] = &[TtsLoaderRegistration {
    name: "qwen3_tts",
    family: ModelFamily::Qwen3Tts,
    loader: load_qwen_tts_model,
}];

fn resolve_tts_loader_registration(family: ModelFamily) -> Option<&'static TtsLoaderRegistration> {
    TTS_LOADER_REGISTRY
        .iter()
        .find(|registration| registration.family == family)
}

pub(super) enum InstantiatedPayload {
    None,
    TtsModel(Qwen3TtsModel),
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
            ModelFamily::Lfm2Audio => {
                self.model_registry.load_lfm2(variant, &model_path).await?;
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
                let registration = resolve_tts_loader_registration(family).ok_or_else(|| {
                    Error::InvalidInput(format!("Unsupported TTS model variant: {variant}"))
                })?;
                if self.is_tts_model_already_loaded(&model_path).await {
                    InstantiatedPayload::None
                } else {
                    info!(
                        "Loading native TTS model {variant} ({}) from {:?}",
                        registration.name, model_path
                    );
                    let model = (registration.loader)(
                        &model_path,
                        self.device.clone(),
                        self.config.kv_page_size,
                        &self.config.kv_cache_dtype,
                    )?;
                    InstantiatedPayload::TtsModel(model)
                }
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

    async fn is_tts_model_already_loaded(&self, model_path: &std::path::Path) -> bool {
        let loaded_path = self.loaded_model_path.read().await;
        let tts_model = self.tts_model.read().await;

        tts_model.is_some()
            && loaded_path
                .as_ref()
                .map(|p| p.as_path() == model_path)
                .unwrap_or(false)
    }
}
