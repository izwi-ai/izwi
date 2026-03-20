//! Model registry to ensure models are loaded once and shared across the app.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{OnceCell, RwLock};
use tracing::info;

use crate::backends::DeviceProfile;
use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::architectures::gemma3::chat::Gemma3ChatModel;
use crate::models::architectures::kokoro::KokoroTtsModel;
use crate::models::architectures::lfm2::chat::Lfm2ChatModel;
use crate::models::architectures::lfm25_audio::{
    Lfm25AudioGenerationConfig, Lfm25AudioModel, Lfm25AudioStreamConfig,
};
use crate::models::architectures::parakeet::asr::ParakeetAsrModel;
use crate::models::architectures::qwen3::asr::{
    AsrDecodeState as Qwen3AsrDecodeState, AsrDecodeStep as Qwen3AsrDecodeStep,
    AsrTranscriptionOutput as Qwen3AsrTranscriptionOutput, Qwen3AsrModel,
};
use crate::models::architectures::qwen3::chat::{
    ChatDecodeState as Qwen3ChatDecodeState, ChatGenerationOutput, Qwen3ChatModel,
};
use crate::models::architectures::qwen3::tts::Qwen3TtsModel;
use crate::models::architectures::qwen35::chat::{
    ChatDecodeState as Qwen35ChatDecodeState, Qwen35ChatModel,
};
use crate::models::architectures::sortformer::diarization::SortformerDiarizerModel;
use crate::models::architectures::voxtral::realtime::VoxtralRealtimeModel;
use crate::models::architectures::whisper::asr::{
    AsrTranscriptionOutput as WhisperAsrTranscriptionOutput, WhisperTurboAsrModel,
};
use crate::models::shared::chat::{ChatGenerationConfig, ChatMessage};
use crate::runtime::{DiarizationConfig, DiarizationResult};

type AsrLoaderFn = fn(&Path, ModelVariant, DeviceProfile) -> Result<NativeAsrModel>;
type AudioChatLoaderFn = fn(&Path, ModelVariant, DeviceProfile) -> Result<NativeAudioChatModel>;
type ChatLoaderFn = fn(&Path, ModelVariant, DeviceProfile) -> Result<NativeChatModel>;
type DiarizationLoaderFn = fn(&Path, ModelVariant) -> Result<NativeDiarizationModel>;
type VoxtralLoaderFn = fn(&Path, ModelVariant, DeviceProfile) -> Result<VoxtralRealtimeModel>;
type QwenTtsLoaderFn = fn(&Path, ModelVariant, DeviceProfile, usize, &str) -> Result<Qwen3TtsModel>;
type KokoroLoaderFn = fn(&Path, ModelVariant, DeviceProfile) -> Result<KokoroTtsModel>;

struct AsrLoaderRegistration {
    name: &'static str,
    family: ModelFamily,
    loader: AsrLoaderFn,
}

struct AudioChatLoaderRegistration {
    name: &'static str,
    family: ModelFamily,
    loader: AudioChatLoaderFn,
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

struct QwenTtsLoaderRegistration {
    name: &'static str,
    family: ModelFamily,
    loader: QwenTtsLoaderFn,
}

struct KokoroLoaderRegistration {
    name: &'static str,
    family: ModelFamily,
    loader: KokoroLoaderFn,
}

fn load_qwen_forced_aligner_model(
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

fn load_whisper_asr_model(
    model_dir: &Path,
    _variant: ModelVariant,
    device: DeviceProfile,
) -> Result<NativeAsrModel> {
    Ok(NativeAsrModel::WhisperTurbo(WhisperTurboAsrModel::load(
        model_dir, device,
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

fn load_lfm2_chat_model(
    model_dir: &Path,
    variant: ModelVariant,
    device: DeviceProfile,
) -> Result<NativeChatModel> {
    Ok(NativeChatModel::Lfm2(Lfm2ChatModel::load(
        model_dir, variant, device,
    )?))
}

fn load_qwen35_chat_model(
    model_dir: &Path,
    variant: ModelVariant,
    device: DeviceProfile,
) -> Result<NativeChatModel> {
    Ok(NativeChatModel::Qwen35(Qwen35ChatModel::load(
        model_dir, variant, device,
    )?))
}

fn load_lfm25_audio_model(
    model_dir: &Path,
    variant: ModelVariant,
    device: DeviceProfile,
) -> Result<NativeAudioChatModel> {
    Ok(NativeAudioChatModel::Lfm25Audio(Lfm25AudioModel::load(
        model_dir, variant, device,
    )?))
}

fn load_voxtral_model(
    model_dir: &Path,
    _variant: ModelVariant,
    device: DeviceProfile,
) -> Result<VoxtralRealtimeModel> {
    VoxtralRealtimeModel::load(model_dir, device)
}

fn load_qwen_tts_model(
    model_dir: &Path,
    _variant: ModelVariant,
    device: DeviceProfile,
    kv_page_size: usize,
    kv_cache_dtype: &str,
) -> Result<Qwen3TtsModel> {
    Qwen3TtsModel::load(model_dir, device, kv_page_size, kv_cache_dtype)
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
        name: "whisper_asr",
        family: ModelFamily::WhisperAsr,
        loader: load_whisper_asr_model,
    },
    AsrLoaderRegistration {
        name: "qwen_forced_aligner",
        family: ModelFamily::Qwen3ForcedAligner,
        loader: load_qwen_forced_aligner_model,
    },
];

const AUDIO_CHAT_LOADER_REGISTRY: &[AudioChatLoaderRegistration] = &[AudioChatLoaderRegistration {
    name: "lfm25_audio",
    family: ModelFamily::Lfm25Audio,
    loader: load_lfm25_audio_model,
}];

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
    ChatLoaderRegistration {
        name: "lfm2_chat",
        family: ModelFamily::Lfm2Chat,
        loader: load_lfm2_chat_model,
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

const QWEN_TTS_LOADER_REGISTRY: &[QwenTtsLoaderRegistration] = &[QwenTtsLoaderRegistration {
    name: "qwen3_tts",
    family: ModelFamily::Qwen3Tts,
    loader: load_qwen_tts_model,
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
        ModelFamily::Qwen3ForcedAligner => ModelFamily::Qwen3ForcedAligner,
        ModelFamily::ParakeetAsr => ModelFamily::ParakeetAsr,
        ModelFamily::WhisperAsr => ModelFamily::WhisperAsr,
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

fn resolve_audio_chat_loader_registration(
    variant: ModelVariant,
) -> Option<&'static AudioChatLoaderRegistration> {
    let family = variant.family();
    AUDIO_CHAT_LOADER_REGISTRY
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

fn resolve_qwen_tts_loader_registration(
    variant: ModelVariant,
) -> Option<&'static QwenTtsLoaderRegistration> {
    let family = variant.family();
    QWEN_TTS_LOADER_REGISTRY
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
    WhisperTurbo(WhisperTurboAsrModel),
}

pub enum NativeAudioChatModel {
    Lfm25Audio(Lfm25AudioModel),
}

#[derive(Debug, Clone)]
pub struct NativeAudioChatGeneration {
    pub text: String,
    pub prompt_tokens: usize,
    pub tokens_generated: usize,
    pub audio_frames_generated: usize,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
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
            Self::WhisperTurbo(model) => {
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
            Self::WhisperTurbo(model) => {
                let WhisperAsrTranscriptionOutput { text, language } =
                    model.transcribe_with_details(audio, sample_rate, language)?;
                Ok(NativeAsrTranscription { text, language })
            }
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
            Self::WhisperTurbo(_) => Err(Error::InvalidInput(
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
            Self::WhisperTurbo(model) => model.max_audio_seconds_hint(),
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
            Self::WhisperTurbo(_) => Err(Error::InvalidInput(
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

impl NativeAudioChatModel {
    pub fn generate_sequential(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
    ) -> Result<NativeAudioChatGeneration> {
        self.generate_sequential_with_callback(messages, max_new_tokens, &mut |_delta| {})
    }

    pub fn generate_sequential_with_callback(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        on_text_delta: &mut dyn FnMut(&str),
    ) -> Result<NativeAudioChatGeneration> {
        match self {
            Self::Lfm25Audio(model) => {
                let output = model.generate_sequential_with_callback(
                    messages,
                    max_new_tokens,
                    on_text_delta,
                )?;
                Ok(NativeAudioChatGeneration {
                    text: output.text,
                    prompt_tokens: output.prompt_tokens,
                    tokens_generated: output.tokens_generated,
                    audio_frames_generated: output.audio_frames_generated,
                    samples: output.samples,
                    sample_rate: output.sample_rate,
                })
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_interleaved_with_config_and_callback(
        &self,
        history_messages: &[ChatMessage],
        audio: &[f32],
        sample_rate: u32,
        max_new_tokens: usize,
        system_prompt: Option<&str>,
        generation_config: &Lfm25AudioGenerationConfig,
        stream_config: &Lfm25AudioStreamConfig,
        on_text_delta: &mut dyn FnMut(&str),
        on_audio_samples: &mut dyn FnMut(&[f32]),
    ) -> Result<NativeAudioChatGeneration> {
        match self {
            Self::Lfm25Audio(model) => {
                let output = model.generate_interleaved_with_config_and_callback(
                    history_messages,
                    audio,
                    sample_rate,
                    max_new_tokens,
                    system_prompt,
                    generation_config,
                    stream_config,
                    on_text_delta,
                    on_audio_samples,
                )?;
                Ok(NativeAudioChatGeneration {
                    text: output.text,
                    prompt_tokens: output.prompt_tokens,
                    tokens_generated: output.tokens_generated,
                    audio_frames_generated: output.audio_frames_generated,
                    samples: output.samples,
                    sample_rate: output.sample_rate,
                })
            }
        }
    }

    pub fn transcribe(&self, audio: &[f32], sample_rate: u32) -> Result<NativeAsrTranscription> {
        self.transcribe_with_callback(audio, sample_rate, &mut |_delta| {})
    }

    pub fn transcribe_with_callback(
        &self,
        audio: &[f32],
        sample_rate: u32,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<NativeAsrTranscription> {
        match self {
            Self::Lfm25Audio(model) => {
                let output =
                    model.transcribe_to_output_with_callback(audio, sample_rate, 1024, on_delta)?;
                Ok(NativeAsrTranscription {
                    text: output.text,
                    language: None,
                })
            }
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
    Lfm2(Lfm2ChatModel),
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
    pub fn prompt_token_ids(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        self.prompt_token_ids_with_config(messages, &ChatGenerationConfig::default())
    }

    pub fn prompt_token_ids_with_config(
        &self,
        messages: &[ChatMessage],
        config: &ChatGenerationConfig,
    ) -> Result<Vec<u32>> {
        match self {
            Self::Qwen3(model) => model.prompt_token_ids(messages),
            Self::Qwen35(model) => model.prompt_token_ids_with_config(messages, config),
            Self::Gemma3(model) => model.prompt_token_ids(messages),
            Self::Lfm2(model) => model.prompt_token_ids(messages),
        }
    }

    pub fn generate(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
    ) -> Result<ChatGenerationOutput> {
        let config = ChatGenerationConfig::default();
        self.generate_with_config(messages, max_new_tokens, &config)
    }

    pub fn generate_with_config(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
    ) -> Result<ChatGenerationOutput> {
        match self {
            Self::Qwen3(model) => model.generate(messages, max_new_tokens),
            Self::Qwen35(model) => {
                let output = model.generate_with_config(messages, max_new_tokens, config)?;
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
            Self::Lfm2(model) => {
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
        let config = ChatGenerationConfig::default();
        self.generate_with_callback_and_config(messages, max_new_tokens, &config, on_delta)
    }

    pub fn generate_with_callback_and_config(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<ChatGenerationOutput> {
        match self {
            Self::Qwen3(model) => model.generate_with_callback(messages, max_new_tokens, on_delta),
            Self::Qwen35(model) => {
                let output = model.generate_with_callback_and_config(
                    messages,
                    max_new_tokens,
                    config,
                    on_delta,
                )?;
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
            Self::Lfm2(model) => {
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
            Self::Lfm2(model) => model.supports_incremental_decode(),
        }
    }

    pub fn start_decode_state(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
    ) -> Result<NativeChatDecodeState> {
        let config = ChatGenerationConfig::default();
        self.start_decode_state_with_config(messages, max_new_tokens, &config)
    }

    pub fn start_decode_state_with_config(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        _config: &ChatGenerationConfig,
    ) -> Result<NativeChatDecodeState> {
        match self {
            Self::Qwen3(model) => Ok(NativeChatDecodeState::Qwen3(
                model.start_decode(messages, max_new_tokens)?,
            )),
            Self::Qwen35(model) => Ok(NativeChatDecodeState::Qwen35(
                model.start_decode_state_with_config(messages, max_new_tokens, _config)?,
            )),
            Self::Gemma3(_) => Err(Error::InvalidInput(
                "Incremental decode state is not available for this chat model".to_string(),
            )),
            Self::Lfm2(_) => Err(Error::InvalidInput(
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
    audio_chat_models: Arc<RwLock<HashMap<ModelVariant, Arc<OnceCell<Arc<NativeAudioChatModel>>>>>>,
    diarization_models:
        Arc<RwLock<HashMap<ModelVariant, Arc<OnceCell<Arc<NativeDiarizationModel>>>>>>,
    chat_models: Arc<RwLock<HashMap<ModelVariant, Arc<OnceCell<Arc<NativeChatModel>>>>>>,
    voxtral_models: Arc<RwLock<HashMap<ModelVariant, Arc<OnceCell<Arc<VoxtralRealtimeModel>>>>>>,
    qwen_tts_models: Arc<RwLock<HashMap<ModelVariant, Arc<OnceCell<Arc<Qwen3TtsModel>>>>>>,
    kokoro_models: Arc<RwLock<HashMap<ModelVariant, Arc<OnceCell<Arc<KokoroTtsModel>>>>>>,
}

impl ModelRegistry {
    pub fn new(models_dir: PathBuf, device: DeviceProfile) -> Self {
        Self {
            models_dir,
            device,
            asr_models: Arc::new(RwLock::new(HashMap::new())),
            audio_chat_models: Arc::new(RwLock::new(HashMap::new())),
            diarization_models: Arc::new(RwLock::new(HashMap::new())),
            chat_models: Arc::new(RwLock::new(HashMap::new())),
            voxtral_models: Arc::new(RwLock::new(HashMap::new())),
            qwen_tts_models: Arc::new(RwLock::new(HashMap::new())),
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

    pub async fn load_audio_chat(
        &self,
        variant: ModelVariant,
        model_dir: &Path,
    ) -> Result<Arc<NativeAudioChatModel>> {
        let registration = resolve_audio_chat_loader_registration(variant).ok_or_else(|| {
            Error::InvalidInput(format!("Unsupported audio-chat model variant: {variant}"))
        })?;

        let cell = {
            let mut guard = self.audio_chat_models.write().await;
            guard
                .entry(variant)
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone()
        };

        info!(
            "Loading native audio-chat model {variant} ({}) from {model_dir:?}",
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
                        Ok::<NativeAudioChatModel, Error>(model)
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

    pub async fn load_qwen_tts(
        &self,
        variant: ModelVariant,
        model_dir: &Path,
        kv_page_size: usize,
        kv_cache_dtype: &str,
    ) -> Result<Arc<Qwen3TtsModel>> {
        let registration = resolve_qwen_tts_loader_registration(variant).ok_or_else(|| {
            Error::InvalidInput(format!("Unsupported Qwen TTS model variant: {variant}"))
        })?;

        let cell = {
            let mut guard = self.qwen_tts_models.write().await;
            guard
                .entry(variant)
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone()
        };

        info!(
            "Loading Qwen TTS model {variant} ({}) from {model_dir:?}",
            registration.name
        );

        let model = cell
            .get_or_try_init({
                let model_dir = model_dir.to_path_buf();
                let device = self.device.clone();
                let loader = registration.loader;
                let kv_cache_dtype = kv_cache_dtype.to_string();
                move || async move {
                    tokio::task::spawn_blocking(move || {
                        loader(
                            &model_dir,
                            variant,
                            device,
                            kv_page_size.max(1),
                            &kv_cache_dtype,
                        )
                    })
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

    pub async fn get_audio_chat(&self, variant: ModelVariant) -> Option<Arc<NativeAudioChatModel>> {
        let guard = self.audio_chat_models.read().await;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub fn try_get_chat(&self, variant: ModelVariant) -> Option<Arc<NativeChatModel>> {
        let guard = self.chat_models.try_read().ok()?;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub fn try_get_audio_chat(&self, variant: ModelVariant) -> Option<Arc<NativeAudioChatModel>> {
        let guard = self.audio_chat_models.try_read().ok()?;
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

    pub async fn get_qwen_tts(&self, variant: ModelVariant) -> Option<Arc<Qwen3TtsModel>> {
        let guard = self.qwen_tts_models.read().await;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub fn try_get_qwen_tts(&self, variant: ModelVariant) -> Option<Arc<Qwen3TtsModel>> {
        let guard = self.qwen_tts_models.try_read().ok()?;
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

    pub async fn unload_audio_chat(&self, variant: ModelVariant) {
        let mut guard = self.audio_chat_models.write().await;
        guard.remove(&variant);
    }

    pub async fn unload_voxtral(&self, variant: ModelVariant) {
        let mut guard = self.voxtral_models.write().await;
        guard.remove(&variant);
    }

    pub async fn unload_qwen_tts(&self, variant: ModelVariant) {
        let mut guard = self.qwen_tts_models.write().await;
        guard.remove(&variant);
    }

    pub async fn unload_kokoro(&self, variant: ModelVariant) {
        let mut guard = self.kokoro_models.write().await;
        guard.remove(&variant);
    }
}
