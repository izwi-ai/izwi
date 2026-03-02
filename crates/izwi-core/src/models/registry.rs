//! Model registry to ensure models are loaded once and shared across the app.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{OnceCell, RwLock};
use tracing::info;

use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::architectures::gemma3::chat::Gemma3ChatModel;
use crate::models::architectures::kokoro::KokoroTtsModel;
use crate::models::architectures::lfm2::audio::Lfm2AudioModel;
use crate::models::architectures::parakeet::asr::ParakeetAsrModel;
use crate::models::architectures::qwen3::asr::{
    AsrDecodeState as Qwen3AsrDecodeState, AsrDecodeStep as Qwen3AsrDecodeStep,
    AsrTranscriptionOutput as Qwen3AsrTranscriptionOutput, Qwen3AsrModel,
};
use crate::models::architectures::qwen3::chat::{
    ChatDecodeState as Qwen3ChatDecodeState, ChatGenerationOutput, Qwen3ChatModel,
};
use crate::models::architectures::qwen35::chat::{
    ChatDecodeState as Qwen35ChatDecodeState, Qwen35ChatModel,
};
use crate::models::architectures::sortformer::diarization::SortformerDiarizerModel;
use crate::models::architectures::voxtral::realtime::VoxtralRealtimeModel;
use crate::models::shared::chat::ChatMessage;
use crate::models::shared::device::DeviceProfile;
use crate::runtime::{DiarizationConfig, DiarizationResult};

type AsrLoaderFn = fn(&Path, ModelVariant, DeviceProfile) -> Result<NativeAsrModel>;
type ChatLoaderFn = fn(&Path, ModelVariant, DeviceProfile) -> Result<NativeChatModel>;
type DiarizationLoaderFn = fn(&Path, ModelVariant) -> Result<NativeDiarizationModel>;
type VoxtralLoaderFn = fn(&Path, ModelVariant, DeviceProfile) -> Result<VoxtralRealtimeModel>;
type Lfm2LoaderFn = fn(&Path, ModelVariant, DeviceProfile) -> Result<Lfm2AudioModel>;
type KokoroLoaderFn = fn(&Path, ModelVariant, DeviceProfile) -> Result<KokoroTtsModel>;

struct AsrLoaderRegistration {
    name: &'static str,
    family: ModelFamily,
    loader: AsrLoaderFn,
}

struct ChatLoaderRegistration {
    name: &'static str,
    family: ModelFamily,
    loader: ChatLoaderFn,
}

struct DiarizationLoaderRegistration {
    name: &'static str,
    family: ModelFamily,
    loader: DiarizationLoaderFn,
}

struct VoxtralLoaderRegistration {
    name: &'static str,
    family: ModelFamily,
    loader: VoxtralLoaderFn,
}

struct Lfm2LoaderRegistration {
    name: &'static str,
    family: ModelFamily,
    loader: Lfm2LoaderFn,
}

struct KokoroLoaderRegistration {
    name: &'static str,
    family: ModelFamily,
    loader: KokoroLoaderFn,
}

fn load_qwen_asr_model(
    model_dir: &Path,
    _variant: ModelVariant,
    device: DeviceProfile,
) -> Result<NativeAsrModel> {
    Ok(NativeAsrModel::Qwen3(Qwen3AsrModel::load(
        model_dir, device,
    )?))
}

fn load_parakeet_asr_model(
    model_dir: &Path,
    variant: ModelVariant,
    device: DeviceProfile,
) -> Result<NativeAsrModel> {
    Ok(NativeAsrModel::Parakeet(ParakeetAsrModel::load(
        model_dir, variant, device,
    )?))
}

fn load_qwen_chat_model(
    model_dir: &Path,
    variant: ModelVariant,
    device: DeviceProfile,
) -> Result<NativeChatModel> {
    Ok(NativeChatModel::Qwen3(Qwen3ChatModel::load(
        model_dir, variant, device,
    )?))
}

fn load_qwen35_chat_model(
    model_dir: &Path,
    _variant: ModelVariant,
    device: DeviceProfile,
) -> Result<NativeChatModel> {
    Ok(NativeChatModel::Qwen35(Qwen35ChatModel::load(
        model_dir, device,
    )?))
}

fn load_gemma_chat_model(
    model_dir: &Path,
    variant: ModelVariant,
    device: DeviceProfile,
) -> Result<NativeChatModel> {
    Ok(NativeChatModel::Gemma3(Gemma3ChatModel::load(
        model_dir, variant, device,
    )?))
}

fn load_sortformer_diarization_model(
    model_dir: &Path,
    variant: ModelVariant,
) -> Result<NativeDiarizationModel> {
    Ok(NativeDiarizationModel::Sortformer(
        SortformerDiarizerModel::load(model_dir, variant)?,
    ))
}

fn load_voxtral_model(
    model_dir: &Path,
    _variant: ModelVariant,
    device: DeviceProfile,
) -> Result<VoxtralRealtimeModel> {
    VoxtralRealtimeModel::load(model_dir, device)
}

fn load_lfm2_model(
    model_dir: &Path,
    _variant: ModelVariant,
    device: DeviceProfile,
) -> Result<Lfm2AudioModel> {
    Lfm2AudioModel::load(model_dir, device)
}

fn load_kokoro_model(
    model_dir: &Path,
    _variant: ModelVariant,
    device: DeviceProfile,
) -> Result<KokoroTtsModel> {
    KokoroTtsModel::load(model_dir, device)
}

const ASR_LOADER_REGISTRY: &[AsrLoaderRegistration] = &[
    AsrLoaderRegistration {
        name: "parakeet_asr",
        family: ModelFamily::ParakeetAsr,
        loader: load_parakeet_asr_model,
    },
    AsrLoaderRegistration {
        name: "qwen_asr",
        family: ModelFamily::Qwen3Asr,
        loader: load_qwen_asr_model,
    },
];

const CHAT_LOADER_REGISTRY: &[ChatLoaderRegistration] = &[
    ChatLoaderRegistration {
        name: "qwen_chat",
        family: ModelFamily::Qwen3Chat,
        loader: load_qwen_chat_model,
    },
    ChatLoaderRegistration {
        name: "qwen35_chat",
        family: ModelFamily::Qwen35Chat,
        loader: load_qwen35_chat_model,
    },
    ChatLoaderRegistration {
        name: "gemma_chat",
        family: ModelFamily::Gemma3Chat,
        loader: load_gemma_chat_model,
    },
];

const DIARIZATION_LOADER_REGISTRY: &[DiarizationLoaderRegistration] =
    &[DiarizationLoaderRegistration {
        name: "sortformer_diarization",
        family: ModelFamily::SortformerDiarization,
        loader: load_sortformer_diarization_model,
    }];

const VOXTRAL_LOADER_REGISTRY: &[VoxtralLoaderRegistration] = &[VoxtralLoaderRegistration {
    name: "voxtral_realtime",
    family: ModelFamily::Voxtral,
    loader: load_voxtral_model,
}];

const LFM2_LOADER_REGISTRY: &[Lfm2LoaderRegistration] = &[Lfm2LoaderRegistration {
    name: "lfm2_audio",
    family: ModelFamily::Lfm2Audio,
    loader: load_lfm2_model,
}];

const KOKORO_LOADER_REGISTRY: &[KokoroLoaderRegistration] = &[KokoroLoaderRegistration {
    name: "kokoro_tts",
    family: ModelFamily::KokoroTts,
    loader: load_kokoro_model,
}];

fn resolve_asr_loader_registration(
    variant: ModelVariant,
) -> Option<&'static AsrLoaderRegistration> {
    let family = match variant.family() {
        ModelFamily::Qwen3Asr | ModelFamily::Qwen3ForcedAligner => ModelFamily::Qwen3Asr,
        ModelFamily::ParakeetAsr => ModelFamily::ParakeetAsr,
        _ => return None,
    };

    ASR_LOADER_REGISTRY
        .iter()
        .find(|registration| registration.family == family)
}

fn resolve_chat_loader_registration(
    variant: ModelVariant,
) -> Option<&'static ChatLoaderRegistration> {
    let family = variant.family();
    CHAT_LOADER_REGISTRY
        .iter()
        .find(|registration| registration.family == family)
}

fn resolve_diarization_loader_registration(
    variant: ModelVariant,
) -> Option<&'static DiarizationLoaderRegistration> {
    let family = variant.family();
    DIARIZATION_LOADER_REGISTRY
        .iter()
        .find(|registration| registration.family == family)
}

fn resolve_voxtral_loader_registration(
    variant: ModelVariant,
) -> Option<&'static VoxtralLoaderRegistration> {
    let family = variant.family();
    VOXTRAL_LOADER_REGISTRY
        .iter()
        .find(|registration| registration.family == family)
}

fn resolve_lfm2_loader_registration(
    variant: ModelVariant,
) -> Option<&'static Lfm2LoaderRegistration> {
    let family = variant.family();
    LFM2_LOADER_REGISTRY
        .iter()
        .find(|registration| registration.family == family)
}

fn resolve_kokoro_loader_registration(
    variant: ModelVariant,
) -> Option<&'static KokoroLoaderRegistration> {
    let family = variant.family();
    KOKORO_LOADER_REGISTRY
        .iter()
        .find(|registration| registration.family == family)
}

pub enum NativeAsrModel {
    Qwen3(Qwen3AsrModel),
    Parakeet(ParakeetAsrModel),
}

pub enum NativeAsrDecodeState {
    Qwen3(Qwen3AsrDecodeState),
}

pub enum NativeDiarizationModel {
    Sortformer(SortformerDiarizerModel),
}

#[derive(Debug, Clone)]
pub struct NativeAsrDecodeStep {
    pub delta: String,
    pub text: String,
    pub tokens_generated: usize,
    pub finished: bool,
}

#[derive(Debug, Clone)]
pub struct NativeAsrTranscription {
    pub text: String,
    pub language: Option<String>,
}

impl NativeAsrModel {
    pub fn transcribe(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<String> {
        let mut no_op = |_delta: &str| {};
        self.transcribe_with_callback(audio, sample_rate, language, &mut no_op)
    }

    pub fn transcribe_with_callback(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        match self {
            Self::Qwen3(model) => {
                model.transcribe_with_callback(audio, sample_rate, language, on_delta)
            }
            Self::Parakeet(model) => {
                model.transcribe_with_callback(audio, sample_rate, language, on_delta)
            }
        }
    }

    pub fn transcribe_with_details(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<NativeAsrTranscription> {
        match self {
            Self::Qwen3(model) => {
                let Qwen3AsrTranscriptionOutput { text, language } =
                    model.transcribe_with_details(audio, sample_rate, language)?;
                Ok(NativeAsrTranscription { text, language })
            }
            Self::Parakeet(model) => Ok(NativeAsrTranscription {
                text: model.transcribe(audio, sample_rate, language)?,
                language: language.map(|value| value.to_string()),
            }),
        }
    }

    pub fn force_align(
        &self,
        audio: &[f32],
        sample_rate: u32,
        reference_text: &str,
        language: Option<&str>,
    ) -> Result<Vec<(String, u32, u32)>> {
        match self {
            Self::Qwen3(model) => model.force_align(audio, sample_rate, reference_text, language),
            Self::Parakeet(_) => Err(Error::InvalidInput(
                "Forced alignment is only available for Qwen3-ForcedAligner models".to_string(),
            )),
        }
    }

    pub fn supports_incremental_decode(&self) -> bool {
        matches!(self, Self::Qwen3(_))
    }

    pub fn max_audio_seconds_hint(&self) -> Option<f32> {
        match self {
            Self::Qwen3(model) => model.max_audio_seconds_hint(),
            Self::Parakeet(_) => None,
        }
    }

    pub fn start_decode_state(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        max_new_tokens: usize,
    ) -> Result<NativeAsrDecodeState> {
        match self {
            Self::Qwen3(model) => Ok(NativeAsrDecodeState::Qwen3(model.start_decode(
                audio,
                sample_rate,
                language,
                max_new_tokens,
            )?)),
            Self::Parakeet(_) => Err(Error::InvalidInput(
                "Incremental decode state is not available for this ASR model".to_string(),
            )),
        }
    }

    pub fn decode_step(&self, state: &mut NativeAsrDecodeState) -> Result<NativeAsrDecodeStep> {
        match (self, state) {
            (Self::Qwen3(model), NativeAsrDecodeState::Qwen3(state)) => {
                let step: Qwen3AsrDecodeStep = model.decode_step(state)?;
                Ok(NativeAsrDecodeStep {
                    delta: step.delta,
                    text: step.text,
                    tokens_generated: step.tokens_generated,
                    finished: step.finished,
                })
            }
            _ => Err(Error::InvalidInput(
                "ASR decode state does not match loaded ASR model".to_string(),
            )),
        }
    }
}

impl NativeDiarizationModel {
    pub fn diarize(
        &self,
        audio: &[f32],
        sample_rate: u32,
        config: &DiarizationConfig,
    ) -> Result<DiarizationResult> {
        match self {
            Self::Sortformer(model) => model.diarize(audio, sample_rate, config),
        }
    }
}

pub enum NativeChatModel {
    Qwen3(Qwen3ChatModel),
    Qwen35(Qwen35ChatModel),
    Gemma3(Gemma3ChatModel),
}

pub enum NativeChatDecodeState {
    Qwen3(Qwen3ChatDecodeState),
    Qwen35(Qwen35ChatDecodeState),
}

#[derive(Debug, Clone)]
pub struct NativeChatDecodeStep {
    pub delta: String,
    pub text: String,
    pub tokens_generated: usize,
    pub finished: bool,
}

impl NativeChatModel {
    pub fn generate(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
    ) -> Result<ChatGenerationOutput> {
        match self {
            Self::Qwen3(model) => model.generate(messages, max_new_tokens),
            Self::Qwen35(model) => {
                let output = model.generate(messages, max_new_tokens)?;
                Ok(ChatGenerationOutput {
                    text: output.text,
                    tokens_generated: output.tokens_generated,
                })
            }
            Self::Gemma3(model) => {
                let output = model.generate(messages, max_new_tokens)?;
                Ok(ChatGenerationOutput {
                    text: output.text,
                    tokens_generated: output.tokens_generated,
                })
            }
        }
    }

    pub fn generate_with_callback(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<ChatGenerationOutput> {
        match self {
            Self::Qwen3(model) => model.generate_with_callback(messages, max_new_tokens, on_delta),
            Self::Qwen35(model) => {
                let output = model.generate_with_callback(messages, max_new_tokens, on_delta)?;
                Ok(ChatGenerationOutput {
                    text: output.text,
                    tokens_generated: output.tokens_generated,
                })
            }
            Self::Gemma3(model) => {
                let output = model.generate_with_callback(messages, max_new_tokens, on_delta)?;
                Ok(ChatGenerationOutput {
                    text: output.text,
                    tokens_generated: output.tokens_generated,
                })
            }
        }
    }

    pub fn supports_incremental_decode(&self) -> bool {
        match self {
            Self::Qwen3(model) => model.supports_incremental_decode(),
            Self::Qwen35(model) => model.supports_incremental_decode(),
            Self::Gemma3(_) => false,
        }
    }

    pub fn start_decode_state(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
    ) -> Result<NativeChatDecodeState> {
        match self {
            Self::Qwen3(model) => Ok(NativeChatDecodeState::Qwen3(
                model.start_decode(messages, max_new_tokens)?,
            )),
            Self::Qwen35(model) => Ok(NativeChatDecodeState::Qwen35(
                model.start_decode(messages, max_new_tokens)?,
            )),
            Self::Gemma3(_) => Err(Error::InvalidInput(
                "Incremental decode state is not available for this chat model".to_string(),
            )),
        }
    }

    pub fn decode_step(&self, state: &mut NativeChatDecodeState) -> Result<NativeChatDecodeStep> {
        match (self, state) {
            (Self::Qwen3(model), NativeChatDecodeState::Qwen3(state)) => {
                let step = model.decode_step(state)?;
                Ok(NativeChatDecodeStep {
                    delta: step.delta,
                    text: step.text,
                    tokens_generated: step.tokens_generated,
                    finished: step.finished,
                })
            }
            (Self::Qwen35(model), NativeChatDecodeState::Qwen35(state)) => {
                let step = model.decode_step(state)?;
                Ok(NativeChatDecodeStep {
                    delta: step.delta,
                    text: step.text,
                    tokens_generated: step.tokens_generated,
                    finished: step.finished,
                })
            }
            _ => Err(Error::InvalidInput(
                "Chat decode state does not match loaded chat model".to_string(),
            )),
        }
    }
}

#[derive(Clone)]
pub struct ModelRegistry {
    models_dir: PathBuf,
    device: DeviceProfile,
    asr_models: Arc<RwLock<HashMap<ModelVariant, Arc<OnceCell<Arc<NativeAsrModel>>>>>>,
    diarization_models:
        Arc<RwLock<HashMap<ModelVariant, Arc<OnceCell<Arc<NativeDiarizationModel>>>>>>,
    chat_models: Arc<RwLock<HashMap<ModelVariant, Arc<OnceCell<Arc<NativeChatModel>>>>>>,
    voxtral_models: Arc<RwLock<HashMap<ModelVariant, Arc<OnceCell<Arc<VoxtralRealtimeModel>>>>>>,
    lfm2_models: Arc<RwLock<HashMap<ModelVariant, Arc<OnceCell<Arc<Lfm2AudioModel>>>>>>,
    kokoro_models: Arc<RwLock<HashMap<ModelVariant, Arc<OnceCell<Arc<KokoroTtsModel>>>>>>,
}

impl ModelRegistry {
    pub fn new(models_dir: PathBuf, device: DeviceProfile) -> Self {
        Self {
            models_dir,
            device,
            asr_models: Arc::new(RwLock::new(HashMap::new())),
            diarization_models: Arc::new(RwLock::new(HashMap::new())),
            chat_models: Arc::new(RwLock::new(HashMap::new())),
            voxtral_models: Arc::new(RwLock::new(HashMap::new())),
            lfm2_models: Arc::new(RwLock::new(HashMap::new())),
            kokoro_models: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn device(&self) -> &DeviceProfile {
        &self.device
    }

    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    pub async fn load_asr(
        &self,
        variant: ModelVariant,
        model_dir: &Path,
    ) -> Result<Arc<NativeAsrModel>> {
        let registration = resolve_asr_loader_registration(variant).ok_or_else(|| {
            Error::InvalidInput(format!(
                "Unsupported ASR/ForcedAligner model variant: {variant}"
            ))
        })?;

        let cell = {
            let mut guard = self.asr_models.write().await;
            guard
                .entry(variant)
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone()
        };

        info!(
            "Loading native ASR/ForcedAligner model {variant} ({}) from {model_dir:?}",
            registration.name
        );

        let model = cell
            .get_or_try_init({
                let model_dir = model_dir.to_path_buf();
                let device = self.device.clone();
                let loader = registration.loader;
                move || async move {
                    tokio::task::spawn_blocking(move || {
                        let model = loader(&model_dir, variant, device)?;
                        Ok::<NativeAsrModel, Error>(model)
                    })
                    .await
                    .map_err(|e| Error::ModelLoadError(e.to_string()))?
                    .map(Arc::new)
                }
            })
            .await?;

        Ok(model.clone())
    }

    pub async fn load_chat(
        &self,
        variant: ModelVariant,
        model_dir: &Path,
    ) -> Result<Arc<NativeChatModel>> {
        let registration = resolve_chat_loader_registration(variant).ok_or_else(|| {
            Error::InvalidInput(format!("Unsupported chat model variant: {variant}"))
        })?;

        let cell = {
            let mut guard = self.chat_models.write().await;
            guard
                .entry(variant)
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone()
        };

        info!(
            "Loading native chat model {variant} ({}) from {model_dir:?}",
            registration.name
        );

        let model = cell
            .get_or_try_init({
                let model_dir = model_dir.to_path_buf();
                let device = self.device.clone();
                let loader = registration.loader;
                move || async move {
                    tokio::task::spawn_blocking(move || {
                        let model = loader(&model_dir, variant, device)?;
                        Ok::<NativeChatModel, Error>(model)
                    })
                    .await
                    .map_err(|e| Error::ModelLoadError(e.to_string()))?
                    .map(Arc::new)
                }
            })
            .await?;

        Ok(model.clone())
    }

    pub async fn load_diarization(
        &self,
        variant: ModelVariant,
        model_dir: &Path,
    ) -> Result<Arc<NativeDiarizationModel>> {
        let registration = resolve_diarization_loader_registration(variant).ok_or_else(|| {
            Error::InvalidInput(format!("Unsupported diarization model variant: {variant}"))
        })?;

        let cell = {
            let mut guard = self.diarization_models.write().await;
            guard
                .entry(variant)
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone()
        };

        info!(
            "Loading native diarization model {variant} ({}) from {model_dir:?}",
            registration.name
        );

        let model = cell
            .get_or_try_init({
                let model_dir = model_dir.to_path_buf();
                let loader = registration.loader;
                move || async move {
                    tokio::task::spawn_blocking(move || {
                        let model = loader(&model_dir, variant)?;
                        Ok::<NativeDiarizationModel, Error>(model)
                    })
                    .await
                    .map_err(|e| Error::ModelLoadError(e.to_string()))?
                    .map(Arc::new)
                }
            })
            .await?;

        Ok(model.clone())
    }

    pub async fn load_voxtral(
        &self,
        variant: ModelVariant,
        model_dir: &Path,
    ) -> Result<Arc<VoxtralRealtimeModel>> {
        let registration = resolve_voxtral_loader_registration(variant).ok_or_else(|| {
            Error::InvalidInput(format!("Unsupported Voxtral model variant: {variant}"))
        })?;

        let cell = {
            let mut guard = self.voxtral_models.write().await;
            guard
                .entry(variant)
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone()
        };

        info!(
            "Loading native Voxtral model {variant} ({}) from {model_dir:?}",
            registration.name
        );

        let model = cell
            .get_or_try_init({
                let model_dir = model_dir.to_path_buf();
                let device = self.device.clone();
                let loader = registration.loader;
                move || async move {
                    tokio::task::spawn_blocking(move || loader(&model_dir, variant, device))
                        .await
                        .map_err(|e| Error::ModelLoadError(e.to_string()))?
                        .map(Arc::new)
                }
            })
            .await?;

        Ok(model.clone())
    }

    pub async fn load_lfm2(
        &self,
        variant: ModelVariant,
        model_dir: &Path,
    ) -> Result<Arc<Lfm2AudioModel>> {
        let registration = resolve_lfm2_loader_registration(variant).ok_or_else(|| {
            Error::InvalidInput(format!("Unsupported LFM2 model variant: {variant}"))
        })?;

        let cell = {
            let mut guard = self.lfm2_models.write().await;
            guard
                .entry(variant)
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone()
        };

        info!(
            "Loading LFM2 model {variant} ({}) from {model_dir:?}",
            registration.name
        );

        let model = cell
            .get_or_try_init({
                let model_dir = model_dir.to_path_buf();
                let device = self.device.clone();
                let loader = registration.loader;
                move || async move {
                    tokio::task::spawn_blocking(move || loader(&model_dir, variant, device))
                        .await
                        .map_err(|e| Error::ModelLoadError(e.to_string()))?
                        .map(Arc::new)
                }
            })
            .await?;

        Ok(model.clone())
    }

    pub async fn load_kokoro(
        &self,
        variant: ModelVariant,
        model_dir: &Path,
    ) -> Result<Arc<KokoroTtsModel>> {
        let registration = resolve_kokoro_loader_registration(variant).ok_or_else(|| {
            Error::InvalidInput(format!("Unsupported Kokoro model variant: {variant}"))
        })?;

        let cell = {
            let mut guard = self.kokoro_models.write().await;
            guard
                .entry(variant)
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone()
        };

        info!(
            "Loading Kokoro model {variant} ({}) from {model_dir:?}",
            registration.name
        );

        let model = cell
            .get_or_try_init({
                let model_dir = model_dir.to_path_buf();
                let device = self.device.clone();
                let loader = registration.loader;
                move || async move {
                    tokio::task::spawn_blocking(move || loader(&model_dir, variant, device))
                        .await
                        .map_err(|e| Error::ModelLoadError(e.to_string()))?
                        .map(Arc::new)
                }
            })
            .await?;

        Ok(model.clone())
    }

    pub async fn get_asr(&self, variant: ModelVariant) -> Option<Arc<NativeAsrModel>> {
        let guard = self.asr_models.read().await;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub fn try_get_asr(&self, variant: ModelVariant) -> Option<Arc<NativeAsrModel>> {
        let guard = self.asr_models.try_read().ok()?;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub async fn get_diarization(
        &self,
        variant: ModelVariant,
    ) -> Option<Arc<NativeDiarizationModel>> {
        let guard = self.diarization_models.read().await;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub fn try_get_diarization(
        &self,
        variant: ModelVariant,
    ) -> Option<Arc<NativeDiarizationModel>> {
        let guard = self.diarization_models.try_read().ok()?;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub async fn get_chat(&self, variant: ModelVariant) -> Option<Arc<NativeChatModel>> {
        let guard = self.chat_models.read().await;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub fn try_get_chat(&self, variant: ModelVariant) -> Option<Arc<NativeChatModel>> {
        let guard = self.chat_models.try_read().ok()?;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub async fn get_voxtral(&self, variant: ModelVariant) -> Option<Arc<VoxtralRealtimeModel>> {
        let guard = self.voxtral_models.read().await;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub fn try_get_voxtral(&self, variant: ModelVariant) -> Option<Arc<VoxtralRealtimeModel>> {
        let guard = self.voxtral_models.try_read().ok()?;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub async fn get_lfm2(&self, variant: ModelVariant) -> Option<Arc<Lfm2AudioModel>> {
        let guard = self.lfm2_models.read().await;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub fn try_get_lfm2(&self, variant: ModelVariant) -> Option<Arc<Lfm2AudioModel>> {
        let guard = self.lfm2_models.try_read().ok()?;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub async fn get_kokoro(&self, variant: ModelVariant) -> Option<Arc<KokoroTtsModel>> {
        let guard = self.kokoro_models.read().await;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub fn try_get_kokoro(&self, variant: ModelVariant) -> Option<Arc<KokoroTtsModel>> {
        let guard = self.kokoro_models.try_read().ok()?;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub async fn unload_asr(&self, variant: ModelVariant) {
        let mut guard = self.asr_models.write().await;
        guard.remove(&variant);
    }

    pub async fn unload_diarization(&self, variant: ModelVariant) {
        let mut guard = self.diarization_models.write().await;
        guard.remove(&variant);
    }

    pub async fn unload_chat(&self, variant: ModelVariant) {
        let mut guard = self.chat_models.write().await;
        guard.remove(&variant);
    }

    pub async fn unload_voxtral(&self, variant: ModelVariant) {
        let mut guard = self.voxtral_models.write().await;
        guard.remove(&variant);
    }

    pub async fn unload_lfm2(&self, variant: ModelVariant) {
        let mut guard = self.lfm2_models.write().await;
        guard.remove(&variant);
    }

    pub async fn unload_kokoro(&self, variant: ModelVariant) {
        let mut guard = self.kokoro_models.write().await;
        guard.remove(&variant);
    }
}
