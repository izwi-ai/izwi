use std::path::Path;

use tracing::info;

use crate::catalog::ModelFamily;
use crate::catalog::ModelVariant;
use crate::error::Result;
use crate::kernels::numeric_check::PrecisionReloadGate;
use crate::models::registry::ModelRegistry;
use crate::runtime::lifecycle::controller::ModelLifecycleController;
use crate::runtime::lifecycle::phases::AcquiredModelLoad;
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

impl ModelLifecycleController {
    /// Loads the acquired model's weights. `reload_gate` admits the reload
    /// of an ASR component whose lowered precision fails its load-time probe.
    pub(super) async fn instantiate_model(
        &self,
        acquired: AcquiredModelLoad,
        reload_gate: PrecisionReloadGate,
    ) -> Result<InstantiatedModelLoad> {
        let AcquiredModelLoad {
            variant,
            model_path,
        } = acquired;
        let payload = load_into_registry(
            &self.model_registry,
            variant,
            &model_path,
            Some(reload_gate),
            self.config.kv_page_size,
            &self.config.kv_cache_dtype,
        )
        .await?;
        Ok(InstantiatedModelLoad {
            family: variant.family(),
            variant,
            model_path,
            payload,
        })
    }
}

/// Loads `variant`'s weights from `model_path` into `registry`, dispatching
/// on its family: the one load path both the lifecycle and the Metal
/// footprint harness (`load/footprint_tests.rs`) run. Without `reload_gate`,
/// an ASR component whose lowered precision fails its probe reloads
/// unadmitted.
pub(in crate::runtime::lifecycle) async fn load_into_registry(
    registry: &ModelRegistry,
    variant: ModelVariant,
    model_path: &Path,
    reload_gate: Option<PrecisionReloadGate>,
    kv_page_size: usize,
    kv_cache_dtype: &str,
) -> Result<InstantiatedPayload> {
    Ok(match variant.family() {
        ModelFamily::ParakeetAsr
        | ModelFamily::WhisperAsr
        | ModelFamily::Qwen3Asr
        | ModelFamily::VibeVoiceAsr
        | ModelFamily::NemotronAsr
        | ModelFamily::GraniteSpeechAsr
        | ModelFamily::GraniteSpeech5Asr
        | ModelFamily::Qwen3ForcedAligner => {
            registry.load_asr(variant, model_path, reload_gate).await?;
            InstantiatedPayload::None
        }
        ModelFamily::SortformerDiarization => {
            registry.load_diarization(variant, model_path).await?;
            InstantiatedPayload::None
        }
        ModelFamily::Qwen3Chat
        | ModelFamily::Qwen3MoeChat
        | ModelFamily::Qwen35Chat
        | ModelFamily::Qwen35MoeChat
        | ModelFamily::Qwen38Chat
        | ModelFamily::Lfm2Chat
        | ModelFamily::Lfm3Chat
        | ModelFamily::Gemma3Chat => {
            registry.load_chat(variant, model_path).await?;
            InstantiatedPayload::None
        }
        ModelFamily::Voxtral => {
            registry.load_voxtral(variant, model_path).await?;
            InstantiatedPayload::None
        }
        ModelFamily::Lfm25Audio | ModelFamily::Lfm3Audio => {
            registry.load_audio_chat(variant, model_path).await?;
            InstantiatedPayload::None
        }
        ModelFamily::KokoroTts => {
            registry.load_kokoro(variant, model_path).await?;
            InstantiatedPayload::None
        }
        ModelFamily::Qwen3Tts => {
            info!("Loading native TTS model {variant} via shared model registry");
            registry
                .load_qwen_tts(variant, model_path, kv_page_size, kv_cache_dtype)
                .await?;
            InstantiatedPayload::None
        }
        ModelFamily::VoxtralTts => {
            registry.load_voxtral_tts(variant, model_path).await?;
            InstantiatedPayload::None
        }
        ModelFamily::VibeVoiceTts => {
            registry.load_vibevoice_tts(variant, model_path).await?;
            InstantiatedPayload::None
        }
        ModelFamily::FishS2Tts => {
            registry.load_fish_s2_tts(variant, model_path).await?;
            InstantiatedPayload::None
        }
        ModelFamily::ChatterboxTts => {
            registry.load_chatterbox_tts(variant, model_path).await?;
            InstantiatedPayload::None
        }
        ModelFamily::Maya1Tts => {
            registry.load_maya1_tts(variant, model_path).await?;
            InstantiatedPayload::None
        }
        ModelFamily::Maya2Tts => {
            registry.load_maya2_tts(variant, model_path).await?;
            InstantiatedPayload::None
        }
        ModelFamily::Tokenizer => {
            let tokenizer = match Tokenizer::from_path(model_path) {
                Ok(tokenizer) => Some(tokenizer),
                Err(err) => {
                    tracing::warn!("Failed to load tokenizer from model directory: {}", err);
                    None
                }
            };
            InstantiatedPayload::Tokenizer(tokenizer)
        }
    })
}
