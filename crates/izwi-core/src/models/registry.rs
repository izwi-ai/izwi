//! Model registry to ensure models are loaded once and shared across the app.

use izwi_asr_toolkit::{plan_audio_chunks, AsrLongFormConfig, TranscriptAssembler};
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::{Notify, OnceCell, RwLock};
use tracing::info;

use crate::backends::state::PhysicalStateTransactionId;
use crate::backends::{DTypeSelectionRequest, DeviceProfile};
use crate::catalog::{ModelFamily, ModelTask};
use crate::engine::{
    InvocationStaticAttentionLease, InvocationTensorLease, RetainedTensorStateRuntimeV2,
    StageDescriptor,
};
use crate::error::{Error, Result};
use crate::kv::v2::{InvocationStateBackingKindV2, InvocationWorkspaceLeaseSetV2};
use crate::kv::{InferenceStateCapability, InferenceStateContractProvider};
use crate::model::ModelVariant;
use crate::models::architectures::fish_s2::FishS2TtsModel;
use crate::models::architectures::gemma3::chat::{
    ChatDecodeCheckpoint as Gemma3ChatDecodeCheckpoint, ChatDecodeState as Gemma3ChatDecodeState,
    Gemma3ChatModel,
};
use crate::models::architectures::granite_speech::asr::{
    GraniteSpeechAsrGenerationOptions, GraniteSpeechAsrModel, GraniteSpeechAsrTranscriptionOutput,
    GraniteSpeechPhysicalStateSpec, GraniteSpeechTask,
};
use crate::models::architectures::kokoro::KokoroTtsModel;
use crate::models::architectures::lfm2::chat::Lfm2ChatModel;
use crate::models::architectures::lfm2::physical::Lfm2PhysicalStateSpec;
use crate::models::architectures::lfm25_audio::{
    physical::{
        Lfm25AudioPhysicalStateSpec, Lfm25AudioStateMode, LFM25_DEPTHFORMER_STATE_DOMAIN,
        LFM25_MAIN_ATTENTION_STATE_DOMAIN, LFM25_MAIN_SHORTCONV_STATE_DOMAIN,
    },
    Lfm25AudioGenerationConfig, Lfm25AudioModel, Lfm25AudioStreamConfig,
};
use crate::models::architectures::nemotron::asr::{
    NemotronAsrDecodeStep, NemotronAsrModel, NemotronAsrTranscriptionOutput,
    NemotronOfflinePhysicalStateSpec, NemotronRealtimePhysicalStateSpec,
    NemotronRealtimeResourceReservation, NemotronStreamingState,
};
use crate::models::architectures::parakeet::asr::{
    ParakeetAsrModel, ParakeetAsrTranscriptionOutput, ParakeetPhysicalStateSpec,
};
use crate::models::architectures::qwen3::asr::{
    AsrDecodeState as Qwen3AsrDecodeState, AsrDecodeStep as Qwen3AsrDecodeStep,
    AsrTranscriptionOutput as Qwen3AsrTranscriptionOutput, Qwen3AsrModel,
    Qwen3AsrPhysicalStateSpec,
};
use crate::models::architectures::qwen3::chat::{
    ChatDecodeCheckpoint as Qwen3ChatDecodeCheckpoint, ChatDecodeState as Qwen3ChatDecodeState,
    ChatGenerationOutput, Qwen3ChatModel,
};
use crate::models::architectures::qwen3::tts::Qwen3TtsModel;
use crate::models::architectures::qwen35::chat::{
    ChatDecodeState as Qwen35ChatDecodeState, Qwen35ChatModel, Qwen35PreparedPrompt,
};
use crate::models::architectures::qwen38::chat::{
    ChatDecodeState as Qwen38ChatDecodeState, Qwen38ChatModel, Qwen38PreparedPrompt,
};
use crate::models::architectures::sortformer::diarization::{
    SortformerDiarizerModel, SortformerPhysicalStateSpec, SortformerWorkspaceEstimate,
    SortformerWorkspaceEvent,
};
use crate::models::architectures::vibevoice::asr::{
    VibeVoiceAsrGenerationOptions, VibeVoiceAsrModel, VibeVoiceAsrTranscriptionOutput,
};
use crate::models::architectures::vibevoice::tts::VibeVoiceTtsModel;
use crate::models::architectures::vibevoice::VibeVoicePhysicalStateSpec;
use crate::models::architectures::voxtral::realtime::VoxtralRealtimeModel;
use crate::models::architectures::voxtral::tts::VoxtralTtsModel;
use crate::models::architectures::whisper::asr::{
    AsrTranscriptionOutput as WhisperAsrTranscriptionOutput, WhisperTurboAsrModel,
};
use crate::models::architectures::whisper::WhisperPhysicalStateSpec;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::chat::{ChatGenerationConfig, ChatMessage};
use crate::runtime::{DiarizationConfig, DiarizationResult};

type AsrLoaderFn = fn(&Path, ModelVariant, DeviceProfile) -> Result<NativeAsrModel>;
type AudioChatLoaderFn = fn(&Path, ModelVariant, DeviceProfile) -> Result<NativeAudioChatModel>;
type ChatLoaderFn = fn(&Path, ModelVariant, DeviceProfile) -> Result<NativeChatModel>;
type DiarizationLoaderFn = fn(&Path, ModelVariant, DeviceProfile) -> Result<NativeDiarizationModel>;
type VoxtralLoaderFn = fn(&Path, ModelVariant, DeviceProfile) -> Result<VoxtralRealtimeModel>;
type VoxtralTtsLoaderFn = fn(&Path, ModelVariant, DeviceProfile) -> Result<VoxtralTtsModel>;
type VibeVoiceTtsLoaderFn = fn(&Path, ModelVariant, DeviceProfile) -> Result<VibeVoiceTtsModel>;
type FishS2TtsLoaderFn = fn(&Path, ModelVariant, DeviceProfile) -> Result<FishS2TtsModel>;
type QwenTtsLoaderFn = fn(&Path, ModelVariant, DeviceProfile, usize, &str) -> Result<Qwen3TtsModel>;
type KokoroLoaderFn = fn(&Path, ModelVariant, DeviceProfile) -> Result<KokoroTtsModel>;

const LFM25_AUDIO_ASR_DEFAULT_MAX_NEW_TOKENS: usize = 1024;
const LFM25_AUDIO_ASR_MIN_CHUNK_NEW_TOKENS: usize = 128;
const LFM25_AUDIO_ASR_MAX_CHUNK_NEW_TOKENS: usize = 256;
const LFM25_AUDIO_ASR_TOKENS_PER_SECOND: f32 = 8.0;

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

struct VoxtralTtsLoaderRegistration {
    name: &'static str,
    family: ModelFamily,
    loader: VoxtralTtsLoaderFn,
}

struct VibeVoiceTtsLoaderRegistration {
    name: &'static str,
    family: ModelFamily,
    loader: VibeVoiceTtsLoaderFn,
}

struct FishS2TtsLoaderRegistration {
    name: &'static str,
    family: ModelFamily,
    loader: FishS2TtsLoaderFn,
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
    variant: ModelVariant,
    device: DeviceProfile,
) -> Result<NativeAsrModel> {
    Ok(NativeAsrModel::Qwen3(Qwen3AsrModel::load(
        model_dir, variant, device,
    )?))
}

fn load_qwen_asr_model(
    model_dir: &Path,
    variant: ModelVariant,
    device: DeviceProfile,
) -> Result<NativeAsrModel> {
    Ok(NativeAsrModel::Qwen3(Qwen3AsrModel::load(
        model_dir, variant, device,
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

fn load_nemotron_asr_model(
    model_dir: &Path,
    variant: ModelVariant,
    device: DeviceProfile,
) -> Result<NativeAsrModel> {
    Ok(NativeAsrModel::Nemotron(NemotronAsrModel::load(
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

fn load_vibevoice_asr_model(
    model_dir: &Path,
    variant: ModelVariant,
    device: DeviceProfile,
) -> Result<NativeAsrModel> {
    Ok(NativeAsrModel::VibeVoice(VibeVoiceAsrModel::load(
        model_dir, variant, device,
    )?))
}

fn load_granite_speech_asr_model(
    model_dir: &Path,
    variant: ModelVariant,
    device: DeviceProfile,
) -> Result<NativeAsrModel> {
    Ok(NativeAsrModel::GraniteSpeech(GraniteSpeechAsrModel::load(
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
    device: DeviceProfile,
) -> Result<NativeDiarizationModel> {
    Ok(NativeDiarizationModel::Sortformer(
        SortformerDiarizerModel::load(model_dir, variant, device)?,
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

fn load_qwen38_chat_model(
    model_dir: &Path,
    variant: ModelVariant,
    device: DeviceProfile,
) -> Result<NativeChatModel> {
    Ok(NativeChatModel::Qwen38(Qwen38ChatModel::load(
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

fn load_voxtral_tts_model(
    model_dir: &Path,
    _variant: ModelVariant,
    device: DeviceProfile,
) -> Result<VoxtralTtsModel> {
    VoxtralTtsModel::load(model_dir, device)
}

fn load_vibevoice_tts_model(
    model_dir: &Path,
    variant: ModelVariant,
    device: DeviceProfile,
) -> Result<VibeVoiceTtsModel> {
    VibeVoiceTtsModel::load(model_dir, variant, device)
}

fn load_fish_s2_tts_model(
    model_dir: &Path,
    variant: ModelVariant,
    device: DeviceProfile,
) -> Result<FishS2TtsModel> {
    FishS2TtsModel::load(model_dir, variant, device)
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
        name: "nemotron_asr",
        family: ModelFamily::NemotronAsr,
        loader: load_nemotron_asr_model,
    },
    AsrLoaderRegistration {
        name: "whisper_asr",
        family: ModelFamily::WhisperAsr,
        loader: load_whisper_asr_model,
    },
    AsrLoaderRegistration {
        name: "qwen_asr",
        family: ModelFamily::Qwen3Asr,
        loader: load_qwen_asr_model,
    },
    AsrLoaderRegistration {
        name: "qwen_forced_aligner",
        family: ModelFamily::Qwen3ForcedAligner,
        loader: load_qwen_forced_aligner_model,
    },
    AsrLoaderRegistration {
        name: "vibevoice_asr",
        family: ModelFamily::VibeVoiceAsr,
        loader: load_vibevoice_asr_model,
    },
    AsrLoaderRegistration {
        name: "granite_speech_asr",
        family: ModelFamily::GraniteSpeechAsr,
        loader: load_granite_speech_asr_model,
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
        name: "qwen38_chat",
        family: ModelFamily::Qwen38Chat,
        loader: load_qwen38_chat_model,
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

const VOXTRAL_TTS_LOADER_REGISTRY: &[VoxtralTtsLoaderRegistration] =
    &[VoxtralTtsLoaderRegistration {
        name: "voxtral_tts",
        family: ModelFamily::VoxtralTts,
        loader: load_voxtral_tts_model,
    }];

const VIBEVOICE_TTS_LOADER_REGISTRY: &[VibeVoiceTtsLoaderRegistration] =
    &[VibeVoiceTtsLoaderRegistration {
        name: "vibevoice_tts",
        family: ModelFamily::VibeVoiceTts,
        loader: load_vibevoice_tts_model,
    }];

const FISH_S2_TTS_LOADER_REGISTRY: &[FishS2TtsLoaderRegistration] =
    &[FishS2TtsLoaderRegistration {
        name: "fish_s2_tts",
        family: ModelFamily::FishS2Tts,
        loader: load_fish_s2_tts_model,
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
        ModelFamily::Qwen3Asr => ModelFamily::Qwen3Asr,
        ModelFamily::Qwen3ForcedAligner => ModelFamily::Qwen3ForcedAligner,
        ModelFamily::ParakeetAsr => ModelFamily::ParakeetAsr,
        ModelFamily::NemotronAsr => ModelFamily::NemotronAsr,
        ModelFamily::WhisperAsr => ModelFamily::WhisperAsr,
        ModelFamily::VibeVoiceAsr => ModelFamily::VibeVoiceAsr,
        ModelFamily::GraniteSpeechAsr => ModelFamily::GraniteSpeechAsr,
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

fn resolve_voxtral_tts_loader_registration(
    variant: ModelVariant,
) -> Option<&'static VoxtralTtsLoaderRegistration> {
    let family = variant.family();
    VOXTRAL_TTS_LOADER_REGISTRY
        .iter()
        .find(|registration| registration.family == family)
}

fn resolve_vibevoice_tts_loader_registration(
    variant: ModelVariant,
) -> Option<&'static VibeVoiceTtsLoaderRegistration> {
    let family = variant.family();
    VIBEVOICE_TTS_LOADER_REGISTRY
        .iter()
        .find(|registration| registration.family == family)
}

fn resolve_fish_s2_tts_loader_registration(
    variant: ModelVariant,
) -> Option<&'static FishS2TtsLoaderRegistration> {
    let family = variant.family();
    FISH_S2_TTS_LOADER_REGISTRY
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
    Nemotron(NemotronAsrModel),
    WhisperTurbo(WhisperTurboAsrModel),
    VibeVoice(VibeVoiceAsrModel),
    GraniteSpeech(GraniteSpeechAsrModel),
}

impl InferenceStateContractProvider for NativeAsrModel {
    fn inference_state_contract(&self) -> Result<InferenceStateCapability> {
        match self {
            Self::Qwen3(model) => model.inference_state_contract(),
            Self::Parakeet(_)
            | Self::Nemotron(_)
            | Self::WhisperTurbo(_)
            | Self::VibeVoice(_)
            | Self::GraniteSpeech(_) => Ok(InferenceStateCapability::Stateless),
        }
    }
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
    pub diagnostics: Option<serde_json::Value>,
}

pub enum NativeAsrDecodeState {
    Qwen3(Qwen3AsrDecodeState),
    Nemotron(NemotronStreamingState),
}

impl NativeAsrDecodeState {
    pub(crate) fn uses_managed_qwen3_kv(&self) -> bool {
        matches!(self, Self::Qwen3(state) if state.uses_managed_kv())
    }

    pub(crate) fn take_managed_write_completions(
        &mut self,
    ) -> Vec<Arc<crate::backends::kv::KvWriteBatchCompletion>> {
        match self {
            Self::Qwen3(state) => state.take_managed_write_completions(),
            Self::Nemotron(_) => Vec::new(),
        }
    }

    pub(crate) fn sequence_position(&self) -> Option<usize> {
        match self {
            Self::Qwen3(state) => Some(state.sequence_position()),
            Self::Nemotron(_) => None,
        }
    }

    pub(crate) fn install_qwen3_managed_reservation(
        &mut self,
        cache: PhysicalPagedKvCache,
    ) -> Result<()> {
        match self {
            Self::Qwen3(state) => state.install_managed_reservation(cache),
            Self::Nemotron(_) => Err(Error::InvalidInput(
                "managed Qwen3 KV cache was supplied to a non-Qwen3 ASR state".to_string(),
            )),
        }
    }
}

pub enum NativeAsrRealtimeState {
    Nemotron(NemotronStreamingState),
}

impl NativeAsrRealtimeState {
    pub fn resource_usage(&self) -> Option<(u64, u64)> {
        match self {
            Self::Nemotron(state) => {
                let usage = state.session_resource_usage()?;
                Some((usage.host_bytes, usage.tensor_bytes))
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeAsrRealtimeResourceReservation {
    Nemotron(NemotronRealtimeResourceReservation),
}

impl NativeAsrRealtimeResourceReservation {
    pub fn max_samples(self) -> usize {
        match self {
            Self::Nemotron(reservation) => reservation.max_samples,
        }
    }

    pub fn host_bytes(self) -> u64 {
        match self {
            Self::Nemotron(reservation) => reservation.host_bytes,
        }
    }

    pub fn tensor_bytes(self) -> u64 {
        match self {
            Self::Nemotron(reservation) => reservation.tensor_bytes,
        }
    }
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
pub struct NativeAsrRealtimeEvent {
    pub delta: String,
    pub text: String,
    pub is_final: bool,
    pub chunk_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeAsrGenerationOptions {
    pub max_new_tokens: usize,
    pub stop_token_ids: Vec<u32>,
    pub stop_sequences: Vec<String>,
}

impl Default for NativeAsrGenerationOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 768,
            stop_token_ids: Vec::new(),
            stop_sequences: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NativeAsrTranscription {
    pub text: String,
    pub language: Option<String>,
    pub diagnostics: Option<serde_json::Value>,
}

fn vibevoice_asr_options(options: NativeAsrGenerationOptions) -> VibeVoiceAsrGenerationOptions {
    VibeVoiceAsrGenerationOptions {
        max_new_tokens: options.max_new_tokens,
        stop_token_ids: options.stop_token_ids,
        stop_sequences: options.stop_sequences,
    }
}

fn granite_speech_asr_options(
    options: NativeAsrGenerationOptions,
) -> GraniteSpeechAsrGenerationOptions {
    GraniteSpeechAsrGenerationOptions {
        max_new_tokens: options.max_new_tokens,
        stop_token_ids: options.stop_token_ids,
        stop_sequences: options.stop_sequences,
    }
}

impl NativeAsrModel {
    /// Execute an atomic ASR operation through its complete lifecycle-owned
    /// invocation workspace. This is the model-adapter boundary used by direct
    /// runtime pipelines; callers never select or omit physical domains.
    pub(crate) fn transcribe_with_details_and_prompt_and_options_from_invocation_workspace(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        options: NativeAsrGenerationOptions,
        leases: &mut InvocationWorkspaceLeaseSetV2,
    ) -> Result<NativeAsrTranscription> {
        match self {
            Self::Qwen3(_) => {
                let cache = leases
                    .lease_exact_kind_mut(InvocationStateBackingKindV2::PagedAttention)?
                    .paged_cache_mut()?;
                self.transcribe_qwen3_with_details_and_prompt_physical(
                    audio,
                    sample_rate,
                    language,
                    prompt,
                    cache,
                )
            }
            Self::WhisperTurbo(_) => {
                let (self_attention, cross_attention) = leases.lease_exact_kind_pair_mut(
                    InvocationStateBackingKindV2::PagedAttention,
                    InvocationStateBackingKindV2::StaticAttention,
                )?;
                let self_kv = self_attention.paged_cache_mut()?;
                let cross_kv = cross_attention.typed_mut::<InvocationStaticAttentionLease>()?;
                self.transcribe_whisper_with_details_and_prompt_physical(
                    audio,
                    sample_rate,
                    language,
                    prompt,
                    self_kv,
                    cross_kv,
                )
            }
            Self::VibeVoice(_) => self
                .transcribe_vibevoice_with_details_and_prompt_and_options_physical(
                    audio,
                    sample_rate,
                    language,
                    prompt,
                    options,
                    leases,
                ),
            Self::GraniteSpeech(_) => {
                let cache = leases
                    .lease_exact_kind_mut(InvocationStateBackingKindV2::PagedAttention)?
                    .paged_cache_mut()?;
                self.transcribe_granite_speech_with_details_and_prompt_and_options_physical(
                    audio,
                    sample_rate,
                    language,
                    prompt,
                    options,
                    cache,
                )
            }
            Self::Parakeet(_) => {
                let state = leases
                    .lease_exact_kind_mut(InvocationStateBackingKindV2::Tensor)?
                    .typed_mut::<InvocationTensorLease>()?;
                self.transcribe_parakeet_with_details_physical(audio, sample_rate, language, state)
            }
            Self::Nemotron(_) => {
                let (predictor, acoustic) = leases.lease_exact_kind_pair_mut(
                    InvocationStateBackingKindV2::Tensor,
                    InvocationStateBackingKindV2::StaticTensor,
                )?;
                self.transcribe_nemotron_with_details_and_prompt_physical(
                    audio,
                    sample_rate,
                    language,
                    prompt,
                    predictor.typed_mut::<InvocationTensorLease>()?,
                    acoustic.typed_mut::<InvocationTensorLease>()?,
                )
            }
        }
    }

    pub(crate) fn nemotron_offline_physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<NemotronOfflinePhysicalStateSpec> {
        match self {
            Self::Nemotron(model) => model.offline_physical_state_spec(stage_graphs),
            _ => Err(Error::ModelLoadError(
                "non-Nemotron ASR model cannot author offline Nemotron physical state".to_string(),
            )),
        }
    }

    pub(crate) fn transcribe_nemotron_with_details_and_prompt_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        predictor: &mut InvocationTensorLease,
        acoustic: &mut InvocationTensorLease,
    ) -> Result<NativeAsrTranscription> {
        let Self::Nemotron(model) = self else {
            return Err(Error::InferenceError(
                "Nemotron offline physical state was routed to a different model".to_string(),
            ));
        };
        let NemotronAsrTranscriptionOutput {
            text,
            language,
            diagnostics,
        } = model.transcribe_with_details_and_prompt_physical(
            audio,
            sample_rate,
            language,
            prompt,
            predictor,
            acoustic,
        )?;
        Ok(NativeAsrTranscription {
            text,
            language,
            diagnostics,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn transcribe_nemotron_with_callback_and_prompt_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        predictor: &mut InvocationTensorLease,
        acoustic: &mut InvocationTensorLease,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let Self::Nemotron(model) = self else {
            return Err(Error::InferenceError(
                "Nemotron offline physical state was routed to a different model".to_string(),
            ));
        };
        model.transcribe_with_callback_and_prompt_physical(
            audio,
            sample_rate,
            language,
            prompt,
            predictor,
            acoustic,
            on_delta,
        )
    }

    pub(crate) fn parakeet_physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<ParakeetPhysicalStateSpec> {
        match self {
            Self::Parakeet(model) => model.physical_state_spec(stage_graphs),
            _ => Err(Error::ModelLoadError(
                "non-Parakeet ASR model cannot author Parakeet physical state".to_string(),
            )),
        }
    }

    pub(crate) fn transcribe_parakeet_with_details_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        state: &mut InvocationTensorLease,
    ) -> Result<NativeAsrTranscription> {
        let Self::Parakeet(model) = self else {
            return Err(Error::InferenceError(
                "Parakeet physical ASR state was routed to a different model".to_string(),
            ));
        };
        let ParakeetAsrTranscriptionOutput {
            text,
            language,
            diagnostics,
        } = model.transcribe_with_details_physical(audio, sample_rate, language, state)?;
        Ok(NativeAsrTranscription {
            text,
            language,
            diagnostics,
        })
    }

    pub(crate) fn transcribe_parakeet_with_callback_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        state: &mut InvocationTensorLease,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let Self::Parakeet(model) = self else {
            return Err(Error::InferenceError(
                "Parakeet physical ASR state was routed to a different model".to_string(),
            ));
        };
        model.transcribe_with_callback_physical(audio, sample_rate, language, state, on_delta)
    }

    pub(crate) fn whisper_physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<WhisperPhysicalStateSpec> {
        match self {
            Self::WhisperTurbo(model) => model.physical_state_spec(stage_graphs),
            _ => Err(Error::ModelLoadError(
                "non-Whisper ASR model cannot author Whisper physical state".to_string(),
            )),
        }
    }

    pub(crate) fn transcribe_whisper_with_details_and_prompt_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        self_kv: &mut PhysicalPagedKvCache,
        cross_kv: &mut InvocationStaticAttentionLease,
    ) -> Result<NativeAsrTranscription> {
        let Self::WhisperTurbo(model) = self else {
            return Err(Error::InferenceError(
                "Whisper physical ASR state was routed to a different model".to_string(),
            ));
        };
        let WhisperAsrTranscriptionOutput {
            text,
            language,
            diagnostics,
        } = model.transcribe_with_details_and_prompt_physical(
            audio,
            sample_rate,
            language,
            prompt,
            self_kv,
            cross_kv,
        )?;
        Ok(NativeAsrTranscription {
            text,
            language,
            diagnostics,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn transcribe_whisper_with_callback_and_prompt_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        self_kv: &mut PhysicalPagedKvCache,
        cross_kv: &mut InvocationStaticAttentionLease,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let Self::WhisperTurbo(model) = self else {
            return Err(Error::InferenceError(
                "Whisper physical ASR state was routed to a different model".to_string(),
            ));
        };
        model.transcribe_with_callback_and_prompt_physical(
            audio,
            sample_rate,
            language,
            prompt,
            self_kv,
            cross_kv,
            on_delta,
        )
    }

    pub(crate) fn granite_speech_physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<GraniteSpeechPhysicalStateSpec> {
        match self {
            Self::GraniteSpeech(model) => model.physical_state_spec(stage_graphs),
            _ => Err(Error::ModelLoadError(
                "non-Granite ASR model cannot author Granite Speech physical state".to_string(),
            )),
        }
    }

    pub(crate) fn transcribe_granite_speech_with_details_and_prompt_and_options_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        options: NativeAsrGenerationOptions,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<NativeAsrTranscription> {
        let Self::GraniteSpeech(model) = self else {
            return Err(Error::InferenceError(
                "Granite Speech physical ASR pages were routed to a different model".to_string(),
            ));
        };
        let GraniteSpeechAsrTranscriptionOutput {
            text,
            language,
            diagnostics,
        } = model.transcribe_with_details_and_prompt_and_options_physical(
            audio,
            sample_rate,
            language,
            prompt,
            granite_speech_asr_options(options),
            cache,
        )?;
        Ok(NativeAsrTranscription {
            text,
            language,
            diagnostics,
        })
    }

    pub(crate) fn transcribe_granite_speech_with_details_prompt_prefix_and_options_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        prefix_text: Option<&str>,
        options: NativeAsrGenerationOptions,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<NativeAsrTranscription> {
        let Self::GraniteSpeech(model) = self else {
            return Err(Error::InferenceError(
                "Granite Speech physical ASR pages were routed to a different model".to_string(),
            ));
        };
        let GraniteSpeechAsrTranscriptionOutput {
            text,
            language,
            diagnostics,
        } = model.transcribe_with_details_and_prompt_prefix_and_options_physical(
            audio,
            sample_rate,
            language,
            prompt,
            prefix_text,
            granite_speech_asr_options(options),
            cache,
        )?;
        Ok(NativeAsrTranscription {
            text,
            language,
            diagnostics,
        })
    }

    pub(crate) fn transcribe_granite_speech_with_callback_and_prompt_and_options_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        options: NativeAsrGenerationOptions,
        cache: &mut PhysicalPagedKvCache,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let Self::GraniteSpeech(model) = self else {
            return Err(Error::InferenceError(
                "Granite Speech physical ASR pages were routed to a different model".to_string(),
            ));
        };
        model
            .transcribe_with_callback_and_prompt_and_options_physical(
                audio,
                sample_rate,
                language,
                prompt,
                granite_speech_asr_options(options),
                cache,
                on_delta,
            )
            .map(|output| output.text)
    }

    pub(crate) fn transcribe_granite_speech_task_and_options_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        task: GraniteSpeechTask,
        prefix_text: Option<&str>,
        options: NativeAsrGenerationOptions,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<NativeAsrTranscription> {
        let Self::GraniteSpeech(model) = self else {
            return Err(Error::InferenceError(
                "Granite Speech physical ASR pages were routed to a different model".to_string(),
            ));
        };
        let GraniteSpeechAsrTranscriptionOutput {
            text,
            language,
            diagnostics,
        } = model.transcribe_with_details_task_prefix_and_options_physical(
            audio,
            sample_rate,
            language,
            task,
            prefix_text,
            granite_speech_asr_options(options),
            cache,
        )?;
        Ok(NativeAsrTranscription {
            text,
            language,
            diagnostics,
        })
    }

    pub(crate) fn vibevoice_physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<VibeVoicePhysicalStateSpec> {
        match self {
            Self::VibeVoice(model) => model.physical_state_spec(stage_graphs),
            _ => Err(Error::ModelLoadError(
                "non-VibeVoice ASR model cannot author VibeVoice physical state".to_string(),
            )),
        }
    }

    pub(crate) fn transcribe_vibevoice_with_details_and_prompt_and_options_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        options: NativeAsrGenerationOptions,
        leases: &mut InvocationWorkspaceLeaseSetV2,
    ) -> Result<NativeAsrTranscription> {
        let Self::VibeVoice(model) = self else {
            return Err(Error::InferenceError(
                "VibeVoice physical ASR pages were routed to a different model".to_string(),
            ));
        };
        let VibeVoiceAsrTranscriptionOutput {
            text,
            language,
            diagnostics,
        } = model.transcribe_with_details_and_prompt_and_options_physical(
            audio,
            sample_rate,
            language,
            prompt,
            vibevoice_asr_options(options),
            leases,
        )?;
        Ok(NativeAsrTranscription {
            text,
            language,
            diagnostics,
        })
    }

    pub(crate) fn transcribe_vibevoice_with_callback_and_prompt_and_options_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        options: NativeAsrGenerationOptions,
        leases: &mut InvocationWorkspaceLeaseSetV2,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let Self::VibeVoice(model) = self else {
            return Err(Error::InferenceError(
                "VibeVoice physical ASR pages were routed to a different model".to_string(),
            ));
        };
        model.transcribe_with_callback_and_prompt_and_options_physical(
            audio,
            sample_rate,
            language,
            prompt,
            vibevoice_asr_options(options),
            leases,
            on_delta,
        )
    }

    pub(crate) fn qwen3_physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<Qwen3AsrPhysicalStateSpec> {
        match self {
            Self::Qwen3(model) => model.physical_state_spec(stage_graphs),
            _ => Err(Error::ModelLoadError(
                "non-Qwen ASR model cannot author Qwen3 physical state".to_string(),
            )),
        }
    }

    pub(crate) fn transcribe_qwen3_with_details_and_prompt_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<NativeAsrTranscription> {
        let Self::Qwen3(model) = self else {
            return Err(Error::InferenceError(
                "Qwen3 physical ASR pages were routed to a non-Qwen model".to_string(),
            ));
        };
        let Qwen3AsrTranscriptionOutput {
            text,
            language,
            diagnostics,
        } = model.transcribe_with_details_and_prompt_physical(
            audio,
            sample_rate,
            language,
            prompt,
            cache,
        )?;
        Ok(NativeAsrTranscription {
            text,
            language,
            diagnostics,
        })
    }

    pub(crate) fn transcribe_qwen3_with_callback_and_prompt_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<String> {
        let Self::Qwen3(model) = self else {
            return Err(Error::InferenceError(
                "Qwen3 physical ASR pages were routed to a non-Qwen model".to_string(),
            ));
        };
        model.transcribe_with_callback_and_prompt_physical(
            audio,
            sample_rate,
            language,
            prompt,
            on_delta,
            cache,
        )
    }

    pub fn transcribe(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<String> {
        self.transcribe_with_prompt(audio, sample_rate, language, None)
    }

    pub fn transcribe_with_prompt(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
    ) -> Result<String> {
        let mut no_op = |_delta: &str| {};
        self.transcribe_with_callback_and_prompt(audio, sample_rate, language, prompt, &mut no_op)
    }

    pub fn transcribe_with_callback(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        self.transcribe_with_callback_and_prompt(audio, sample_rate, language, None, on_delta)
    }

    pub fn transcribe_with_callback_and_prompt(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        match self {
            Self::Qwen3(model) => model.transcribe_with_callback_and_prompt(
                audio,
                sample_rate,
                language,
                prompt,
                on_delta,
            ),
            Self::Parakeet(model) => {
                model.transcribe_with_callback(audio, sample_rate, language, on_delta)
            }
            Self::Nemotron(model) => model.transcribe_with_callback_and_prompt(
                audio,
                sample_rate,
                language,
                prompt,
                on_delta,
            ),
            Self::WhisperTurbo(model) => model.transcribe_with_callback_and_prompt(
                audio,
                sample_rate,
                language,
                prompt,
                on_delta,
            ),
            Self::VibeVoice(model) => model.transcribe_with_callback_and_prompt(
                audio,
                sample_rate,
                language,
                prompt,
                on_delta,
            ),
            Self::GraniteSpeech(model) => model.transcribe_with_callback_and_prompt(
                audio,
                sample_rate,
                language,
                prompt,
                on_delta,
            ),
        }
    }

    pub fn transcribe_with_callback_and_prompt_and_options(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        options: NativeAsrGenerationOptions,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        match self {
            Self::VibeVoice(model) => model.transcribe_with_callback_and_prompt_and_options(
                audio,
                sample_rate,
                language,
                prompt,
                vibevoice_asr_options(options),
                on_delta,
            ),
            Self::GraniteSpeech(model) => model
                .transcribe_with_callback_and_prompt_and_options(
                    audio,
                    sample_rate,
                    language,
                    prompt,
                    granite_speech_asr_options(options),
                    on_delta,
                )
                .map(|output| output.text),
            _ => self.transcribe_with_callback_and_prompt(
                audio,
                sample_rate,
                language,
                prompt,
                on_delta,
            ),
        }
    }

    pub fn transcribe_with_details(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<NativeAsrTranscription> {
        self.transcribe_with_details_and_prompt(audio, sample_rate, language, None)
    }

    pub fn transcribe_with_details_and_prompt(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
    ) -> Result<NativeAsrTranscription> {
        match self {
            Self::Qwen3(model) => {
                let Qwen3AsrTranscriptionOutput {
                    text,
                    language,
                    diagnostics,
                } = model.transcribe_with_details_and_prompt(
                    audio,
                    sample_rate,
                    language,
                    prompt,
                )?;
                Ok(NativeAsrTranscription {
                    text,
                    language,
                    diagnostics,
                })
            }
            Self::Parakeet(model) => {
                let ParakeetAsrTranscriptionOutput {
                    text,
                    language,
                    diagnostics,
                } = model.transcribe_with_details(audio, sample_rate, language)?;
                Ok(NativeAsrTranscription {
                    text,
                    language,
                    diagnostics,
                })
            }
            Self::Nemotron(model) => {
                let NemotronAsrTranscriptionOutput {
                    text,
                    language,
                    diagnostics,
                } = model.transcribe_with_details_and_prompt(
                    audio,
                    sample_rate,
                    language,
                    prompt,
                )?;
                Ok(NativeAsrTranscription {
                    text,
                    language,
                    diagnostics,
                })
            }
            Self::WhisperTurbo(model) => {
                let WhisperAsrTranscriptionOutput {
                    text,
                    language,
                    diagnostics,
                } = model.transcribe_with_details_and_prompt(
                    audio,
                    sample_rate,
                    language,
                    prompt,
                )?;
                Ok(NativeAsrTranscription {
                    text,
                    language,
                    diagnostics,
                })
            }
            Self::VibeVoice(model) => {
                let VibeVoiceAsrTranscriptionOutput {
                    text,
                    language,
                    diagnostics,
                } = model.transcribe_with_details_and_prompt(
                    audio,
                    sample_rate,
                    language,
                    prompt,
                )?;
                Ok(NativeAsrTranscription {
                    text,
                    language,
                    diagnostics,
                })
            }
            Self::GraniteSpeech(model) => {
                let GraniteSpeechAsrTranscriptionOutput {
                    text,
                    language,
                    diagnostics,
                } = model.transcribe_with_details_and_prompt(
                    audio,
                    sample_rate,
                    language,
                    prompt,
                )?;
                Ok(NativeAsrTranscription {
                    text,
                    language,
                    diagnostics,
                })
            }
        }
    }

    pub fn transcribe_with_details_and_prompt_and_options(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        options: NativeAsrGenerationOptions,
    ) -> Result<NativeAsrTranscription> {
        match self {
            Self::VibeVoice(model) => {
                let VibeVoiceAsrTranscriptionOutput {
                    text,
                    language,
                    diagnostics,
                } = model.transcribe_with_details_and_prompt_and_options(
                    audio,
                    sample_rate,
                    language,
                    prompt,
                    vibevoice_asr_options(options),
                )?;
                Ok(NativeAsrTranscription {
                    text,
                    language,
                    diagnostics,
                })
            }
            Self::GraniteSpeech(model) => {
                let GraniteSpeechAsrTranscriptionOutput {
                    text,
                    language,
                    diagnostics,
                } = model.transcribe_with_details_and_prompt_and_options(
                    audio,
                    sample_rate,
                    language,
                    prompt,
                    granite_speech_asr_options(options),
                )?;
                Ok(NativeAsrTranscription {
                    text,
                    language,
                    diagnostics,
                })
            }
            _ => self.transcribe_with_details_and_prompt(audio, sample_rate, language, prompt),
        }
    }

    pub fn transcribe_with_details_prompt_prefix_and_options(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        prefix_text: Option<&str>,
        options: NativeAsrGenerationOptions,
    ) -> Result<NativeAsrTranscription> {
        match self {
            Self::GraniteSpeech(model) => {
                let GraniteSpeechAsrTranscriptionOutput {
                    text,
                    language,
                    diagnostics,
                } = model.transcribe_with_details_and_prompt_prefix_and_options(
                    audio,
                    sample_rate,
                    language,
                    prompt,
                    prefix_text,
                    granite_speech_asr_options(options),
                )?;
                Ok(NativeAsrTranscription {
                    text,
                    language,
                    diagnostics,
                })
            }
            _ => self.transcribe_with_details_and_prompt_and_options(
                audio,
                sample_rate,
                language,
                prompt,
                options,
            ),
        }
    }

    pub fn transcribe_with_granite_speech_task_and_options(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        task: GraniteSpeechTask,
        prefix_text: Option<&str>,
        options: NativeAsrGenerationOptions,
    ) -> Result<NativeAsrTranscription> {
        match self {
            Self::GraniteSpeech(model) => {
                let GraniteSpeechAsrTranscriptionOutput {
                    text,
                    language,
                    diagnostics,
                } = model.transcribe_with_details_task_prefix_and_options(
                    audio,
                    sample_rate,
                    language,
                    task,
                    prefix_text,
                    granite_speech_asr_options(options),
                )?;
                Ok(NativeAsrTranscription {
                    text,
                    language,
                    diagnostics,
                })
            }
            _ => Err(Error::InvalidInput(
                "Granite Speech task transcription requires a Granite Speech model".to_string(),
            )),
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
            Self::Nemotron(_) => Err(Error::InvalidInput(
                "Forced alignment is only available for Qwen3-ForcedAligner models".to_string(),
            )),
            Self::WhisperTurbo(_) => Err(Error::InvalidInput(
                "Forced alignment is only available for Qwen3-ForcedAligner models".to_string(),
            )),
            Self::VibeVoice(_) => Err(Error::InvalidInput(
                "Forced alignment is only available for Qwen3-ForcedAligner models".to_string(),
            )),
            Self::GraniteSpeech(_) => Err(Error::InvalidInput(
                "Forced alignment is only available for Qwen3-ForcedAligner models".to_string(),
            )),
        }
    }

    pub fn supports_incremental_decode(&self) -> bool {
        matches!(self, Self::Qwen3(_))
    }

    pub fn supports_realtime_stream_decode(&self) -> bool {
        matches!(self, Self::Nemotron(_))
    }

    pub fn conservative_realtime_stream_resource_reservation(
        variant: ModelVariant,
        language: Option<&str>,
        prompt: Option<&str>,
        right_context_frames: Option<usize>,
    ) -> Result<NativeAsrRealtimeResourceReservation> {
        if variant.family() != ModelFamily::NemotronAsr {
            return Err(Error::InvalidInput(
                "Realtime resource reservation is not available for this ASR model".to_string(),
            ));
        }
        Ok(NativeAsrRealtimeResourceReservation::Nemotron(
            NemotronAsrModel::conservative_realtime_stream_resource_reservation(
                language,
                prompt,
                right_context_frames,
            )?,
        ))
    }

    pub fn realtime_stream_resource_reservation(
        &self,
        language: Option<&str>,
        prompt: Option<&str>,
        right_context_frames: Option<usize>,
    ) -> Result<NativeAsrRealtimeResourceReservation> {
        match self {
            Self::Nemotron(model) => Ok(NativeAsrRealtimeResourceReservation::Nemotron(
                model.realtime_stream_resource_reservation(
                    language,
                    prompt,
                    right_context_frames,
                )?,
            )),
            _ => Err(Error::InvalidInput(
                "Realtime resource reservation is not available for this ASR model".to_string(),
            )),
        }
    }

    pub fn start_realtime_stream_state(
        &self,
        language: Option<&str>,
        prompt: Option<&str>,
        right_context_frames: Option<usize>,
    ) -> Result<NativeAsrRealtimeState> {
        match self {
            Self::Nemotron(model) => Ok(NativeAsrRealtimeState::Nemotron(
                model.start_stream_state(language, prompt, right_context_frames)?,
            )),
            _ => Err(Error::InvalidInput(
                "Realtime audio stream state is not available for this ASR model".to_string(),
            )),
        }
    }

    pub(crate) fn realtime_physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<NemotronRealtimePhysicalStateSpec> {
        match self {
            Self::Nemotron(model) => model.realtime_physical_state_spec(stage_graphs),
            _ => Err(Error::InvalidInput(
                "Realtime physical state is not available for this ASR model".into(),
            )),
        }
    }

    pub(crate) fn hydrate_realtime_physical_state(
        &self,
        state: &mut NativeAsrRealtimeState,
        runtime: &RetainedTensorStateRuntimeV2,
        transaction: PhysicalStateTransactionId,
    ) -> Result<()> {
        match (self, state) {
            (Self::Nemotron(model), NativeAsrRealtimeState::Nemotron(state)) => {
                model.hydrate_realtime_physical_state(state, runtime, transaction)
            }
            _ => Err(Error::InvalidInput(
                "ASR realtime physical state does not match the loaded model".into(),
            )),
        }
    }

    pub(crate) fn stage_realtime_physical_state(
        &self,
        state: &mut NativeAsrRealtimeState,
        runtime: &RetainedTensorStateRuntimeV2,
        transaction: PhysicalStateTransactionId,
        target_cursor: u64,
    ) -> Result<()> {
        match (self, state) {
            (Self::Nemotron(model), NativeAsrRealtimeState::Nemotron(state)) => {
                model.stage_realtime_physical_state(state, runtime, transaction, target_cursor)
            }
            _ => Err(Error::InvalidInput(
                "ASR realtime physical state does not match the loaded model".into(),
            )),
        }
    }

    pub(crate) fn clear_realtime_tensor_handles(
        &self,
        state: &mut NativeAsrRealtimeState,
    ) -> Result<()> {
        match (self, state) {
            (Self::Nemotron(model), NativeAsrRealtimeState::Nemotron(state)) => {
                model.clear_realtime_tensor_handles(state)
            }
            _ => Err(Error::InvalidInput(
                "ASR realtime physical state does not match the loaded model".into(),
            )),
        }
    }

    pub fn start_realtime_stream_state_with_reservation(
        &self,
        language: Option<&str>,
        prompt: Option<&str>,
        right_context_frames: Option<usize>,
        reservation: NativeAsrRealtimeResourceReservation,
    ) -> Result<NativeAsrRealtimeState> {
        match (self, reservation) {
            (
                Self::Nemotron(model),
                NativeAsrRealtimeResourceReservation::Nemotron(reservation),
            ) => Ok(NativeAsrRealtimeState::Nemotron(
                model.start_stream_state_with_reservation(
                    language,
                    prompt,
                    right_context_frames,
                    reservation,
                )?,
            )),
            _ => Err(Error::InvalidInput(
                "Realtime resource reservation does not match the loaded ASR model".to_string(),
            )),
        }
    }

    pub fn push_realtime_stream_samples(
        &self,
        state: &mut NativeAsrRealtimeState,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<NativeAsrRealtimeEvent>> {
        match (self, state) {
            (Self::Nemotron(model), NativeAsrRealtimeState::Nemotron(state)) => Ok(model
                .push_stream_samples(state, samples, sample_rate)?
                .into_iter()
                .map(|event| NativeAsrRealtimeEvent {
                    delta: event.delta,
                    text: event.text,
                    is_final: event.is_final,
                    chunk_index: event.chunk_index,
                })
                .collect()),
            _ => Err(Error::InvalidInput(
                "ASR realtime stream state does not match loaded ASR model".to_string(),
            )),
        }
    }

    pub fn finish_realtime_stream(
        &self,
        state: &mut NativeAsrRealtimeState,
    ) -> Result<Vec<NativeAsrRealtimeEvent>> {
        match (self, state) {
            (Self::Nemotron(model), NativeAsrRealtimeState::Nemotron(state)) => Ok(model
                .finish_stream(state)?
                .into_iter()
                .map(|event| NativeAsrRealtimeEvent {
                    delta: event.delta,
                    text: event.text,
                    is_final: event.is_final,
                    chunk_index: event.chunk_index,
                })
                .collect()),
            _ => Err(Error::InvalidInput(
                "ASR realtime stream state does not match loaded ASR model".to_string(),
            )),
        }
    }

    pub fn max_audio_seconds_hint(&self) -> Option<f32> {
        match self {
            Self::Qwen3(model) => model.max_audio_seconds_hint(),
            Self::Parakeet(_) => None,
            Self::Nemotron(model) => model.max_audio_seconds_hint(),
            Self::WhisperTurbo(model) => model.max_audio_seconds_hint(),
            Self::VibeVoice(model) => model.max_audio_seconds_hint(),
            Self::GraniteSpeech(model) => model.max_audio_seconds_hint(),
        }
    }

    pub(crate) fn incremental_prompt_token_count(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
    ) -> Result<usize> {
        match self {
            Self::Qwen3(model) => {
                model.incremental_prompt_token_count(audio, sample_rate, language, prompt)
            }
            _ => Err(Error::InvalidInput(
                "Loaded ASR model does not expose multimodal sequence shape preparation"
                    .to_string(),
            )),
        }
    }

    pub fn start_decode_state(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        max_new_tokens: usize,
    ) -> Result<NativeAsrDecodeState> {
        self.start_decode_state_with_prompt(audio, sample_rate, language, None, max_new_tokens)
    }

    pub fn start_decode_state_with_prompt(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        max_new_tokens: usize,
    ) -> Result<NativeAsrDecodeState> {
        match self {
            Self::Qwen3(_) => Err(Error::InvalidInput(
                "Incremental Qwen3 ASR requires scheduler-owned physical state".to_string(),
            )),
            Self::Parakeet(_) => Err(Error::InvalidInput(
                "Incremental decode state is not available for this ASR model".to_string(),
            )),
            Self::Nemotron(model) => Ok(NativeAsrDecodeState::Nemotron(
                model.start_decode_with_prompt(
                    audio,
                    sample_rate,
                    language,
                    prompt,
                    max_new_tokens,
                )?,
            )),
            Self::WhisperTurbo(_) => Err(Error::InvalidInput(
                "Incremental decode state is not available for this ASR model".to_string(),
            )),
            Self::VibeVoice(_) => Err(Error::InvalidInput(
                "Incremental decode state is not available for this ASR model".to_string(),
            )),
            Self::GraniteSpeech(_) => Err(Error::InvalidInput(
                "Incremental decode state is not available for this ASR model".to_string(),
            )),
        }
    }

    /// Start Qwen3 ASR with an exact scheduler-owned cache reservation.
    /// Other ASR architectures fail closed until they provide their own
    /// managed-cache adapter instead of being selected by family inference.
    pub(crate) fn start_decode_state_with_prompt_managed(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        max_new_tokens: usize,
        cache: PhysicalPagedKvCache,
    ) -> Result<NativeAsrDecodeState> {
        match self {
            Self::Qwen3(model) => Ok(NativeAsrDecodeState::Qwen3(
                model.start_decode_with_prompt_managed(
                    audio,
                    sample_rate,
                    language,
                    prompt,
                    max_new_tokens,
                    cache,
                )?,
            )),
            _ => Err(Error::InvalidInput(
                "managed Qwen3 KV cache was supplied to a non-Qwen3 ASR model".to_string(),
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
            (Self::Nemotron(model), NativeAsrDecodeState::Nemotron(state)) => {
                let step: NemotronAsrDecodeStep = model.decode_step(state)?;
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

fn lfm25_audio_asr_long_form_config() -> AsrLongFormConfig {
    let mut cfg = AsrLongFormConfig::default();
    if let Some(value) = env_positive_f32("IZWI_ASR_CHUNK_TARGET_SECS") {
        cfg.target_chunk_secs = value;
    }
    if let Some(value) = env_positive_f32("IZWI_ASR_CHUNK_MAX_SECS") {
        cfg.hard_max_chunk_secs = value;
    }
    if let Some(value) = env_positive_f32("IZWI_ASR_CHUNK_OVERLAP_SECS") {
        cfg.overlap_secs = value;
    }
    if let Some(value) = env_positive_f32("IZWI_LFM25_ASR_CHUNK_TARGET_SECS") {
        cfg.target_chunk_secs = value;
    }
    if let Some(value) = env_positive_f32("IZWI_LFM25_ASR_CHUNK_MAX_SECS") {
        cfg.hard_max_chunk_secs = value;
    }
    if let Some(value) = env_positive_f32("IZWI_LFM25_ASR_CHUNK_OVERLAP_SECS") {
        cfg.overlap_secs = value;
    }
    if cfg.hard_max_chunk_secs < cfg.min_chunk_secs {
        cfg.hard_max_chunk_secs = cfg.min_chunk_secs;
    }
    cfg.target_chunk_secs = cfg
        .target_chunk_secs
        .max(cfg.min_chunk_secs.max(1.0))
        .min(cfg.hard_max_chunk_secs);
    cfg.overlap_secs = cfg.overlap_secs.clamp(0.0, cfg.target_chunk_secs * 0.45);
    cfg
}

fn lfm25_audio_asr_single_pass_max_new_tokens(max_tokens: Option<usize>) -> usize {
    max_tokens
        .or_else(|| env_positive_usize("IZWI_LFM25_ASR_MAX_NEW_TOKENS"))
        .unwrap_or(LFM25_AUDIO_ASR_DEFAULT_MAX_NEW_TOKENS)
        .max(1)
}

fn lfm25_audio_asr_chunk_max_new_tokens(duration_secs: f32, max_tokens: Option<usize>) -> usize {
    if let Some(max_tokens) = max_tokens {
        return max_tokens.max(1);
    }
    if let Some(max_tokens) = env_positive_usize("IZWI_LFM25_ASR_CHUNK_MAX_NEW_TOKENS") {
        return max_tokens;
    }

    let duration_budget = if duration_secs.is_finite() && duration_secs > 0.0 {
        (duration_secs * LFM25_AUDIO_ASR_TOKENS_PER_SECOND).ceil() as usize
    } else {
        0
    };
    duration_budget.clamp(
        LFM25_AUDIO_ASR_MIN_CHUNK_NEW_TOKENS,
        LFM25_AUDIO_ASR_MAX_CHUNK_NEW_TOKENS,
    )
}

fn audio_duration_secs(audio: &[f32], sample_rate: u32) -> f32 {
    if sample_rate > 0 {
        audio.len() as f32 / sample_rate as f32
    } else {
        0.0
    }
}

fn samples_to_seconds_f64(samples: usize, sample_rate: u32) -> f64 {
    if sample_rate > 0 {
        samples as f64 / sample_rate as f64
    } else {
        0.0
    }
}

fn env_positive_f32(key: &str) -> Option<f32> {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn env_positive_usize(key: &str) -> Option<usize> {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
}

impl NativeAudioChatModel {
    pub(crate) fn physical_state_spec(
        &self,
        mode: Lfm25AudioStateMode,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<Lfm25AudioPhysicalStateSpec> {
        match self {
            Self::Lfm25Audio(model) => model.physical_state_spec(mode, stage_graphs),
        }
    }

    pub(crate) fn generate_sequential_with_callback_from_invocation_workspace(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        leases: &mut InvocationWorkspaceLeaseSetV2,
        on_text_delta: &mut dyn FnMut(&str),
    ) -> Result<NativeAudioChatGeneration> {
        match self {
            Self::Lfm25Audio(model) => {
                let (attention, shortconv, depthformer) = leases.lease_triplet_mut(
                    LFM25_MAIN_ATTENTION_STATE_DOMAIN,
                    LFM25_MAIN_SHORTCONV_STATE_DOMAIN,
                    LFM25_DEPTHFORMER_STATE_DOMAIN,
                )?;
                let output = model.generate_sequential_with_config_and_callback_physical(
                    messages,
                    max_new_tokens,
                    &Lfm25AudioGenerationConfig::default(),
                    attention.paged_cache_mut()?,
                    shortconv.typed_mut::<InvocationTensorLease>()?,
                    depthformer.paged_cache_mut()?,
                    on_text_delta,
                )?;
                Ok(NativeAudioChatGeneration {
                    text: output.text,
                    prompt_tokens: output.prompt_tokens,
                    tokens_generated: output.tokens_generated,
                    audio_frames_generated: output.audio_frames_generated,
                    samples: output.samples,
                    sample_rate: output.sample_rate,
                    diagnostics: output.diagnostics,
                })
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_interleaved_with_config_and_callback_from_invocation_workspace(
        &self,
        history_messages: &[ChatMessage],
        audio: &[f32],
        sample_rate: u32,
        max_new_tokens: usize,
        system_prompt: Option<&str>,
        generation_config: &Lfm25AudioGenerationConfig,
        stream_config: &Lfm25AudioStreamConfig,
        leases: &mut InvocationWorkspaceLeaseSetV2,
        on_text_delta: &mut dyn FnMut(&str),
        on_audio_samples: &mut dyn FnMut(&[f32]),
    ) -> Result<NativeAudioChatGeneration> {
        match self {
            Self::Lfm25Audio(model) => {
                let (attention, shortconv, depthformer) = leases.lease_triplet_mut(
                    LFM25_MAIN_ATTENTION_STATE_DOMAIN,
                    LFM25_MAIN_SHORTCONV_STATE_DOMAIN,
                    LFM25_DEPTHFORMER_STATE_DOMAIN,
                )?;
                let output = model.generate_interleaved_with_config_and_callback_physical(
                    history_messages,
                    audio,
                    sample_rate,
                    max_new_tokens,
                    system_prompt,
                    generation_config,
                    stream_config,
                    attention.paged_cache_mut()?,
                    shortconv.typed_mut::<InvocationTensorLease>()?,
                    depthformer.paged_cache_mut()?,
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
                    diagnostics: output.diagnostics,
                })
            }
        }
    }

    pub(crate) fn transcribe_with_callback_and_max_tokens_from_invocation_workspace(
        &self,
        audio: &[f32],
        sample_rate: u32,
        max_tokens: Option<usize>,
        leases: &mut InvocationWorkspaceLeaseSetV2,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<NativeAsrTranscription> {
        self.transcribe_long_form_with_callback_from_invocation_workspace(
            audio,
            sample_rate,
            max_tokens,
            leases,
            on_delta,
        )
    }

    pub(crate) fn transcribe_single_pass_with_callback_and_options_from_invocation_workspace(
        &self,
        audio: &[f32],
        sample_rate: u32,
        options: NativeAsrGenerationOptions,
        leases: &mut InvocationWorkspaceLeaseSetV2,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<NativeAsrTranscription> {
        match self {
            Self::Lfm25Audio(model) => {
                let (attention, shortconv) = leases.lease_pair_mut(
                    LFM25_MAIN_ATTENTION_STATE_DOMAIN,
                    LFM25_MAIN_SHORTCONV_STATE_DOMAIN,
                )?;
                let output = model.transcribe_to_output_with_callback_physical(
                    audio,
                    sample_rate,
                    options.max_new_tokens.max(1),
                    attention.paged_cache_mut()?,
                    shortconv.typed_mut::<InvocationTensorLease>()?,
                    on_delta,
                )?;
                Ok(NativeAsrTranscription {
                    text: output.text,
                    language: None,
                    diagnostics: output.diagnostics,
                })
            }
        }
    }

    pub(crate) fn transcribe_long_form_with_callback_from_invocation_workspace(
        &self,
        audio: &[f32],
        sample_rate: u32,
        max_tokens: Option<usize>,
        leases: &mut InvocationWorkspaceLeaseSetV2,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<NativeAsrTranscription> {
        let duration_secs = audio_duration_secs(audio, sample_rate);
        let chunk_cfg = lfm25_audio_asr_long_form_config();
        let chunks = plan_audio_chunks(audio, sample_rate, &chunk_cfg, None);
        if chunks.len() <= 1 {
            return self
                .transcribe_single_pass_with_callback_and_options_from_invocation_workspace(
                    audio,
                    sample_rate,
                    NativeAsrGenerationOptions {
                        max_new_tokens: lfm25_audio_asr_single_pass_max_new_tokens(max_tokens),
                        ..NativeAsrGenerationOptions::default()
                    },
                    leases,
                    on_delta,
                );
        }

        let mut assembler = TranscriptAssembler::new(chunk_cfg.clone());
        let mut chunk_diagnostics = Vec::with_capacity(chunks.len());
        for (idx, chunk) in chunks.iter().enumerate() {
            if chunk.end_sample <= chunk.start_sample || chunk.end_sample > audio.len() {
                chunk_diagnostics.push(serde_json::json!({
                    "index": idx,
                    "skipped": true,
                    "skip_reason": "invalid_bounds",
                    "start_sample": chunk.start_sample,
                    "end_sample": chunk.end_sample
                }));
                continue;
            }

            let chunk_audio = &audio[chunk.start_sample..chunk.end_sample];
            let chunk_duration_secs = audio_duration_secs(chunk_audio, sample_rate);
            let chunk_max_tokens =
                lfm25_audio_asr_chunk_max_new_tokens(chunk_duration_secs, max_tokens);
            let output = self
                .transcribe_single_pass_with_callback_and_options_from_invocation_workspace(
                    chunk_audio,
                    sample_rate,
                    NativeAsrGenerationOptions {
                        max_new_tokens: chunk_max_tokens,
                        ..NativeAsrGenerationOptions::default()
                    },
                    leases,
                    &mut |_delta| {},
                )?;
            let delta = assembler.push_chunk_text(&output.text);
            if !delta.is_empty() {
                on_delta(&delta);
            }

            chunk_diagnostics.push(serde_json::json!({
                "index": idx,
                "start_sample": chunk.start_sample,
                "end_sample": chunk.end_sample,
                "start_seconds": samples_to_seconds_f64(chunk.start_sample, sample_rate),
                "end_seconds": samples_to_seconds_f64(chunk.end_sample, sample_rate),
                "duration_seconds": chunk_duration_secs,
                "max_new_tokens": chunk_max_tokens,
                "text_chars": output.text.len(),
                "model_diagnostics": output.diagnostics
            }));
        }

        Ok(NativeAsrTranscription {
            text: assembler.text().to_string(),
            language: None,
            diagnostics: Some(serde_json::json!({
                "model": "lfm25_audio",
                "task": "asr",
                "chunking": {
                    "enabled": true,
                    "planner": "duration",
                    "input_samples": audio.len(),
                    "input_sample_rate": sample_rate,
                    "duration_seconds": duration_secs,
                    "target_chunk_seconds": chunk_cfg.target_chunk_secs,
                    "hard_max_chunk_seconds": chunk_cfg.hard_max_chunk_secs,
                    "overlap_seconds": chunk_cfg.overlap_secs,
                    "chunks": chunks.iter().map(|chunk| serde_json::json!({
                        "start_sample": chunk.start_sample,
                        "end_sample": chunk.end_sample,
                        "start_seconds": samples_to_seconds_f64(chunk.start_sample, sample_rate),
                        "end_seconds": samples_to_seconds_f64(chunk.end_sample, sample_rate),
                    })).collect::<Vec<_>>(),
                    "chunk_transcriptions": chunk_diagnostics
                },
                "decode": {
                    "requested_max_tokens": max_tokens,
                    "default_single_pass_max_tokens": LFM25_AUDIO_ASR_DEFAULT_MAX_NEW_TOKENS,
                    "chunk_tokens_per_second": LFM25_AUDIO_ASR_TOKENS_PER_SECOND,
                    "chunk_min_new_tokens": LFM25_AUDIO_ASR_MIN_CHUNK_NEW_TOKENS,
                    "chunk_max_new_tokens": LFM25_AUDIO_ASR_MAX_CHUNK_NEW_TOKENS
                }
            })),
        })
    }
}

impl NativeDiarizationModel {
    pub(crate) fn physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<SortformerPhysicalStateSpec> {
        match self {
            Self::Sortformer(model) => model.physical_state_spec(stage_graphs),
        }
    }

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

    pub fn workspace_estimate(
        &self,
        target_sample_count: usize,
    ) -> Result<SortformerWorkspaceEstimate> {
        match self {
            Self::Sortformer(model) => model.workspace_estimate(target_sample_count),
        }
    }

    pub fn diarize_with_workspace_observer<F>(
        &self,
        audio: &[f32],
        sample_rate: u32,
        config: &DiarizationConfig,
        observer: F,
    ) -> Result<DiarizationResult>
    where
        F: FnMut(SortformerWorkspaceEvent) -> Result<()>,
    {
        match self {
            Self::Sortformer(model) => {
                model.diarize_with_workspace_observer(audio, sample_rate, config, observer)
            }
        }
    }

    pub(crate) fn diarize_with_workspace_observer_physical<F>(
        &self,
        audio: &[f32],
        sample_rate: u32,
        config: &DiarizationConfig,
        state: &mut InvocationTensorLease,
        observer: F,
    ) -> Result<DiarizationResult>
    where
        F: FnMut(SortformerWorkspaceEvent) -> Result<()>,
    {
        match self {
            Self::Sortformer(model) => model.diarize_with_workspace_observer_physical(
                audio,
                sample_rate,
                config,
                state,
                observer,
            ),
        }
    }
}

pub enum NativeChatModel {
    Qwen3(Qwen3ChatModel),
    Qwen35(Qwen35ChatModel),
    Qwen38(Qwen38ChatModel),
    Gemma3(Gemma3ChatModel),
    Lfm2(Lfm2ChatModel),
}

impl NativeChatModel {
    /// Native context carried by the exact loaded checkpoint. This deliberately
    /// excludes optional rope-scaling modes that the local adapter has not
    /// implemented.
    pub fn max_context_tokens(&self) -> Result<usize> {
        match self {
            Self::Qwen3(model) => model.max_context_tokens(),
            Self::Qwen35(model) => model.max_context_tokens(),
            Self::Qwen38(model) => model.max_context_tokens(),
            Self::Gemma3(model) => model.max_context_tokens(),
            Self::Lfm2(model) => model.max_context_tokens(),
        }
    }
}

impl InferenceStateContractProvider for NativeChatModel {
    fn inference_state_contract(&self) -> Result<InferenceStateCapability> {
        match self {
            Self::Qwen3(model) => model.inference_state_contract(),
            Self::Qwen35(model) => model.inference_state_contract(),
            Self::Qwen38(model) => model.inference_state_contract(),
            Self::Gemma3(model) => model.inference_state_contract(),
            Self::Lfm2(_) => Ok(InferenceStateCapability::Stateless),
        }
    }
}

#[derive(Default)]
struct ModelUseState {
    active: AtomicUsize,
    idle: Notify,
}

impl ModelUseState {
    fn acquire(self: &Arc<Self>) -> Option<Arc<ModelUseGuard>> {
        self.active
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |active| {
                active.checked_add(1)
            })
            .ok()?;
        Some(Arc::new(ModelUseGuard {
            state: self.clone(),
        }))
    }

    async fn wait_until_idle(&self) {
        loop {
            let notified = self.idle.notified();
            if self.active.load(Ordering::Acquire) == 0 {
                return;
            }
            notified.await;
        }
    }
}

struct ModelUseGuard {
    state: Arc<ModelUseState>,
}

impl Drop for ModelUseGuard {
    fn drop(&mut self) {
        if self.state.active.fetch_sub(1, Ordering::AcqRel) == 1 {
            // `notify_one` retains a permit when an unload waiter has not yet
            // reached its await, avoiding a lost zero-use transition.
            self.state.idle.notify_one();
        }
    }
}

struct TrackedModelEntry<T> {
    model: OnceCell<Arc<T>>,
    uses: Arc<ModelUseState>,
}

impl<T> Default for TrackedModelEntry<T> {
    fn default() -> Self {
        Self {
            model: OnceCell::new(),
            uses: Arc::new(ModelUseState::default()),
        }
    }
}

impl<T> TrackedModelEntry<T> {
    fn acquire(&self) -> Option<TrackedModelLease<T>> {
        let guard = self.uses.acquire()?;
        let model = self.model.get()?.clone();
        Some(TrackedModelLease {
            model,
            _guard: guard,
        })
    }
}

struct TrackedModelLease<T> {
    model: Arc<T>,
    _guard: Arc<ModelUseGuard>,
}

impl<T> Clone for TrackedModelLease<T> {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            _guard: self._guard.clone(),
        }
    }
}

impl<T> Deref for TrackedModelLease<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.model.as_ref()
    }
}

/// A loaded chat model handle that keeps lifecycle/resource residency active.
///
/// Clones share one use guard. The registry removes the model from discovery
/// first during unload, then waits for the last lease clone before allowing the
/// runtime's physical resource authorization to be released.
#[derive(Clone)]
pub struct ChatModelLease {
    inner: TrackedModelLease<NativeChatModel>,
}

impl ChatModelLease {
    pub(crate) fn model_arc(&self) -> Arc<NativeChatModel> {
        self.inner.model.clone()
    }
}

impl Deref for ChatModelLease {
    type Target = NativeChatModel;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// A loaded ASR model handle that fences registry unload for its exact model
/// instance until the final clone is dropped.
#[derive(Clone)]
pub struct AsrModelLease {
    inner: TrackedModelLease<NativeAsrModel>,
}

impl AsrModelLease {
    pub(crate) fn model_arc(&self) -> Arc<NativeAsrModel> {
        self.inner.model.clone()
    }
}

impl Deref for AsrModelLease {
    type Target = NativeAsrModel;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// A loaded Qwen TTS model handle that fences registry unload for its exact
/// model instance until the final clone is dropped.
#[derive(Clone)]
pub struct QwenTtsModelLease {
    inner: TrackedModelLease<Qwen3TtsModel>,
}

impl QwenTtsModelLease {
    pub(crate) fn model_arc(&self) -> Arc<Qwen3TtsModel> {
        self.inner.model.clone()
    }
}

impl Deref for QwenTtsModelLease {
    type Target = Qwen3TtsModel;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub enum NativeChatDecodeState {
    Qwen3(Qwen3ChatDecodeState),
    Qwen35(Qwen35ChatDecodeState),
    Qwen38(Qwen38ChatDecodeState),
    Gemma3(Gemma3ChatDecodeState),
}

pub(crate) enum NativeChatDecodeCheckpoint {
    Qwen3(Qwen3ChatDecodeCheckpoint),
    Gemma3(Gemma3ChatDecodeCheckpoint),
}

#[derive(Debug, Clone)]
pub enum NativeChatPreparedPrompt {
    Qwen35(Qwen35PreparedPrompt),
    Qwen38(Qwen38PreparedPrompt),
}

impl NativeChatPreparedPrompt {
    pub fn prompt_ids(&self) -> &[u32] {
        match self {
            Self::Qwen35(prepared) => prepared.prompt_ids(),
            Self::Qwen38(prepared) => prepared.prompt_ids(),
        }
    }

    pub fn family(&self) -> ModelFamily {
        match self {
            Self::Qwen35(_) => ModelFamily::Qwen35Chat,
            Self::Qwen38(_) => ModelFamily::Qwen38Chat,
        }
    }

    pub(crate) fn as_qwen35(&self) -> Option<&Qwen35PreparedPrompt> {
        match self {
            Self::Qwen35(prepared) => Some(prepared),
            Self::Qwen38(_) => None,
        }
    }

    pub(crate) fn as_qwen38(&self) -> Option<&Qwen38PreparedPrompt> {
        match self {
            Self::Qwen38(prepared) => Some(prepared),
            Self::Qwen35(_) => None,
        }
    }
}

impl NativeChatDecodeState {
    pub(crate) fn begin_continuous_quantum(
        &mut self,
        cache: PhysicalPagedKvCache,
    ) -> Result<NativeChatDecodeCheckpoint> {
        match self {
            Self::Qwen3(state) => state
                .begin_managed_quantum(cache)
                .map(NativeChatDecodeCheckpoint::Qwen3),
            Self::Gemma3(state) => state
                .begin_managed_quantum(cache)
                .map(NativeChatDecodeCheckpoint::Gemma3),
            Self::Qwen35(_) | Self::Qwen38(_) => Err(Error::InvalidInput(
                "scalar chat state entered a continuous tensor quantum".into(),
            )),
        }
    }

    pub(crate) fn rollback_continuous_quantum(
        &mut self,
        checkpoint: NativeChatDecodeCheckpoint,
    ) -> Result<()> {
        match (self, checkpoint) {
            (Self::Qwen3(state), NativeChatDecodeCheckpoint::Qwen3(checkpoint)) => {
                state.rollback_managed_quantum(checkpoint);
                Ok(())
            }
            (Self::Gemma3(state), NativeChatDecodeCheckpoint::Gemma3(checkpoint)) => {
                state.rollback_managed_quantum(checkpoint);
                Ok(())
            }
            _ => Err(Error::InferenceError(
                "continuous chat rollback checkpoint changed model family".into(),
            )),
        }
    }

    pub(crate) fn install_managed_reservations(
        &mut self,
        cache: PhysicalPagedKvCache,
        mtp_cache: Option<PhysicalPagedKvCache>,
    ) -> Result<()> {
        match self {
            Self::Qwen38(state) => {
                if state.uses_mtp_physical_kv() != mtp_cache.is_some() {
                    return Err(Error::InferenceError(
                        "Qwen3.8 managed MTP reservation does not match the decode state policy"
                            .into(),
                    ));
                }
                state.install_physical_reservation(cache)?;
                if let Some(mtp_cache) = mtp_cache {
                    state.install_mtp_physical_reservation(mtp_cache)?;
                }
                Ok(())
            }
            Self::Qwen3(state) => {
                if mtp_cache.is_some() {
                    return Err(Error::InvalidInput(
                        "Qwen3.8 MTP reservation was routed to a Qwen3 state".into(),
                    ));
                }
                state.install_managed_reservation(cache)
            }
            Self::Qwen35(state) => {
                if mtp_cache.is_some() {
                    return Err(Error::InvalidInput(
                        "Qwen3.8 MTP reservation was routed to a Qwen3.5 state".into(),
                    ));
                }
                state.install_physical_reservation(cache)
            }
            Self::Gemma3(state) => {
                if mtp_cache.is_some() {
                    return Err(Error::InvalidInput(
                        "Qwen3.8 MTP reservation was routed to a Gemma3 state".into(),
                    ));
                }
                state.install_physical_reservation(cache)
            }
        }
    }

    pub(crate) fn uses_managed_kv(&self) -> bool {
        match self {
            Self::Qwen3(state) => state.uses_managed_kv(),
            Self::Qwen35(state) => state.uses_physical_kv(),
            Self::Qwen38(state) => state.uses_physical_kv(),
            Self::Gemma3(_) => true,
        }
    }

    pub(crate) fn take_managed_write_completions(
        &mut self,
    ) -> Vec<Arc<crate::backends::kv::KvWriteBatchCompletion>> {
        match self {
            Self::Qwen3(state) => state.take_managed_write_completions(),
            Self::Qwen35(state) => state.take_physical_write_completions(),
            Self::Qwen38(state) => state.take_physical_write_completions(),
            Self::Gemma3(state) => state.take_physical_write_completions(),
        }
    }

    pub(crate) fn bind_hybrid_tensor_sequence(&mut self, sequence: u64) -> Result<()> {
        match self {
            Self::Qwen35(state) => state.bind_tensor_sequence(sequence),
            Self::Qwen38(state) => state.bind_tensor_sequence(sequence),
            Self::Qwen3(_) => Err(Error::InvalidInput(
                "tensor-state reservation was routed to a dense Qwen3 model".into(),
            )),
            Self::Gemma3(_) => Err(Error::InvalidInput(
                "tensor-state reservation was routed to a Gemma3 model".into(),
            )),
        }
    }

    pub(crate) fn restore_hybrid_tensor_state(
        &mut self,
        arena: &crate::backends::state::TensorStateArena,
    ) -> Result<()> {
        match self {
            Self::Qwen35(state) => state.restore_tensor_state(arena),
            Self::Qwen38(state) => state.restore_tensor_state(arena),
            Self::Qwen3(_) => Err(Error::InvalidInput(
                "tensor-state arena was routed to a dense Qwen3 model".into(),
            )),
            Self::Gemma3(_) => Err(Error::InvalidInput(
                "tensor-state arena was routed to a Gemma3 model".into(),
            )),
        }
    }

    pub(crate) fn stage_hybrid_tensor_state(
        &mut self,
        arena: &crate::backends::state::TensorStateArena,
        transaction: u64,
    ) -> Result<()> {
        match self {
            Self::Qwen35(state) => state.stage_tensor_state(arena, transaction),
            Self::Qwen38(state) => state.stage_tensor_state(arena, transaction),
            Self::Qwen3(_) => Err(Error::InvalidInput(
                "tensor-state arena was routed to a dense Qwen3 model".into(),
            )),
            Self::Gemma3(_) => Err(Error::InvalidInput(
                "tensor-state arena was routed to a Gemma3 model".into(),
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NativeChatDecodeStep {
    pub delta: String,
    pub text: String,
    pub tokens_generated: usize,
    pub input_tokens_committed: usize,
    pub finished: bool,
}

impl NativeChatModel {
    pub(crate) fn lfm2_physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<Lfm2PhysicalStateSpec> {
        match self {
            Self::Lfm2(model) => model.physical_state_spec(stage_graphs),
            _ => Err(Error::InvalidInput(
                "loaded chat model is not an LFM2 physical-state provider".into(),
            )),
        }
    }

    pub(crate) fn generate_lfm2_with_callback_physical(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        cache: &mut PhysicalPagedKvCache,
        shortconv: &mut InvocationTensorLease,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<ChatGenerationOutput> {
        match self {
            Self::Lfm2(model) => {
                let output = model.generate_with_callback_physical(
                    messages,
                    max_new_tokens,
                    cache,
                    shortconv,
                    on_delta,
                )?;
                Ok(ChatGenerationOutput {
                    text: output.text,
                    tokens_generated: output.tokens_generated,
                })
            }
            _ => Err(Error::InvalidInput("loaded chat model is not LFM2".into())),
        }
    }

    /// Prepare the exact prompt consumed by execution. Hybrid Qwen families
    /// return their independently typed reusable artifacts.
    pub fn prepare_prompt_for_execution(
        &self,
        messages: &[ChatMessage],
        config: &ChatGenerationConfig,
    ) -> Result<(Vec<u32>, Option<NativeChatPreparedPrompt>)> {
        match self {
            Self::Qwen35(model) => {
                let prepared = model.prepare_prompt_for_execution(messages, config)?;
                Ok((
                    prepared.prompt_ids().to_vec(),
                    Some(NativeChatPreparedPrompt::Qwen35(prepared)),
                ))
            }
            Self::Qwen38(model) => {
                let prepared = model.prepare_prompt_for_execution(messages, config)?;
                Ok((
                    prepared.prompt_ids().to_vec(),
                    Some(NativeChatPreparedPrompt::Qwen38(prepared)),
                ))
            }
            _ => Ok((self.prompt_token_ids_with_config(messages, config)?, None)),
        }
    }

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
            Self::Qwen38(model) => model.prompt_token_ids_with_config(messages, config),
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
        _config: &ChatGenerationConfig,
    ) -> Result<ChatGenerationOutput> {
        match self {
            Self::Qwen3(model) => model.generate(messages, max_new_tokens),
            Self::Qwen35(_) => Err(Error::InvalidInput(
                "Qwen3.5 chat requires scheduler-owned physical state".to_string(),
            )),
            Self::Qwen38(_) => Err(Error::InvalidInput(
                "Qwen3.8 chat requires scheduler-owned physical state".to_string(),
            )),
            Self::Gemma3(model) => {
                let output = model.generate(messages, max_new_tokens)?;
                Ok(ChatGenerationOutput {
                    text: output.text,
                    tokens_generated: output.tokens_generated,
                })
            }
            Self::Lfm2(_) => Err(Error::InvalidInput(
                "LFM2 chat requires invocation-owned physical state".into(),
            )),
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
        _config: &ChatGenerationConfig,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<ChatGenerationOutput> {
        match self {
            Self::Qwen3(model) => model.generate_with_callback(messages, max_new_tokens, on_delta),
            Self::Qwen35(_) => Err(Error::InvalidInput(
                "Qwen3.5 chat requires scheduler-owned physical state".to_string(),
            )),
            Self::Qwen38(_) => Err(Error::InvalidInput(
                "Qwen3.8 chat requires scheduler-owned physical state".to_string(),
            )),
            Self::Gemma3(model) => {
                let output = model.generate_with_callback(messages, max_new_tokens, on_delta)?;
                Ok(ChatGenerationOutput {
                    text: output.text,
                    tokens_generated: output.tokens_generated,
                })
            }
            Self::Lfm2(_) => Err(Error::InvalidInput(
                "LFM2 chat requires invocation-owned physical state".into(),
            )),
        }
    }

    pub fn supports_incremental_decode(&self) -> bool {
        match self {
            Self::Qwen3(model) => model.supports_incremental_decode(),
            Self::Qwen35(model) => model.supports_incremental_decode(),
            Self::Qwen38(model) => model.supports_incremental_decode(),
            Self::Gemma3(model) => model.supports_incremental_decode(),
            Self::Lfm2(model) => model.supports_incremental_decode(),
        }
    }

    pub fn supports_continuous_decode_batch(&self) -> bool {
        match self {
            Self::Qwen3(model) => model.supports_continuous_decode_batch(),
            Self::Gemma3(model) => model.supports_continuous_decode_batch(),
            Self::Qwen38(model) => model.supports_continuous_decode_batch(),
            Self::Qwen35(_) | Self::Lfm2(_) => false,
        }
    }

    pub fn continuous_decode_batch_workspace_per_row_bytes(&self) -> Result<u64> {
        match self {
            Self::Qwen3(model) => model.continuous_decode_batch_workspace_per_row_bytes(),
            Self::Gemma3(model) => model.continuous_decode_batch_workspace_per_row_bytes(),
            Self::Qwen38(model) => model.continuous_decode_batch_workspace_per_row_bytes(),
            Self::Qwen35(_) | Self::Lfm2(_) => Err(Error::InvalidInput(
                "loaded chat model has no continuous decode workspace contract".to_string(),
            )),
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

    /// Native Qwen3 entry point used once the engine has reserved a physical
    /// block table. Other families remain on their existing compatibility
    /// cache until they publish an equivalent managed adapter.
    pub fn start_qwen3_decode_state_managed(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
        cache: PhysicalPagedKvCache,
    ) -> Result<NativeChatDecodeState> {
        match self {
            Self::Qwen3(model) => Ok(NativeChatDecodeState::Qwen3(model.start_decode_managed(
                messages,
                max_new_tokens,
                config,
                cache,
            )?)),
            _ => Err(Error::InvalidInput(
                "managed Qwen3 KV cache was routed to another model family".to_string(),
            )),
        }
    }

    pub(crate) fn start_qwen35_decode_state_managed(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
        prepared: Option<&Qwen35PreparedPrompt>,
        cache: PhysicalPagedKvCache,
    ) -> Result<NativeChatDecodeState> {
        match self {
            Self::Qwen35(model) => Ok(NativeChatDecodeState::Qwen35(
                model.start_decode_state_physical(
                    messages,
                    max_new_tokens,
                    config,
                    prepared,
                    cache,
                )?,
            )),
            _ => Err(Error::InvalidInput(
                "managed Qwen3.5 state was routed to another model family".into(),
            )),
        }
    }

    pub(crate) fn start_qwen38_decode_state_managed(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
        prepared: Option<&Qwen38PreparedPrompt>,
        target_cache: PhysicalPagedKvCache,
        mtp_cache: Option<PhysicalPagedKvCache>,
    ) -> Result<NativeChatDecodeState> {
        match self {
            Self::Qwen38(model) => Ok(NativeChatDecodeState::Qwen38(
                model.start_decode_state_physical(
                    messages,
                    max_new_tokens,
                    config,
                    prepared,
                    target_cache,
                    mtp_cache,
                )?,
            )),
            _ => Err(Error::InvalidInput(
                "managed Qwen3.8 state was routed to another model family".into(),
            )),
        }
    }

    pub(crate) fn start_gemma3_decode_state_managed(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
        cache: PhysicalPagedKvCache,
    ) -> Result<NativeChatDecodeState> {
        match self {
            Self::Gemma3(model) => Ok(NativeChatDecodeState::Gemma3(model.start_decode_managed(
                messages,
                max_new_tokens,
                config,
                cache,
            )?)),
            _ => Err(Error::InvalidInput(
                "managed Gemma3 state was routed to another model family".into(),
            )),
        }
    }

    pub fn start_decode_state_with_config(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
    ) -> Result<NativeChatDecodeState> {
        self.start_decode_state_with_prepared(messages, max_new_tokens, config, None)
    }

    pub fn start_decode_state_with_prepared(
        &self,
        _messages: &[ChatMessage],
        _max_new_tokens: usize,
        _config: &ChatGenerationConfig,
        prepared: Option<&NativeChatPreparedPrompt>,
    ) -> Result<NativeChatDecodeState> {
        match self {
            Self::Qwen3(_) => {
                if prepared.is_some() {
                    return Err(Error::InvalidInput(
                        "Qwen3.5 prepared prompt was routed to a Qwen3 model".to_string(),
                    ));
                }
                Err(Error::InvalidInput(
                    "incremental Qwen3 chat requires scheduler-owned physical state".to_string(),
                ))
            }
            Self::Qwen35(_) => Err(Error::InvalidInput(
                "incremental Qwen3.5 chat requires scheduler-owned physical state".into(),
            )),
            Self::Qwen38(_) => Err(Error::InvalidInput(
                "incremental Qwen3.8 chat requires scheduler-owned physical state".into(),
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
                    input_tokens_committed: 1,
                    finished: step.finished,
                })
            }
            (Self::Qwen35(model), NativeChatDecodeState::Qwen35(state)) => {
                let step = model.decode_step(state)?;
                Ok(NativeChatDecodeStep {
                    delta: step.delta,
                    text: step.text,
                    tokens_generated: step.tokens_generated,
                    input_tokens_committed: 1,
                    finished: step.finished,
                })
            }
            (Self::Qwen38(model), NativeChatDecodeState::Qwen38(state)) => {
                let step = model.decode_step(state)?;
                Ok(NativeChatDecodeStep {
                    delta: step.delta,
                    text: step.text,
                    tokens_generated: step.tokens_generated,
                    input_tokens_committed: step.input_tokens_committed,
                    finished: step.finished,
                })
            }
            (Self::Gemma3(model), NativeChatDecodeState::Gemma3(state)) => {
                let step = model.decode_step(state)?;
                Ok(NativeChatDecodeStep {
                    delta: step.delta,
                    text: step.text,
                    tokens_generated: step.tokens_generated,
                    input_tokens_committed: 1,
                    finished: step.finished,
                })
            }
            _ => Err(Error::InvalidInput(
                "Chat decode state does not match loaded chat model".to_string(),
            )),
        }
    }

    pub(crate) fn decode_quantum(
        &self,
        state: &mut NativeChatDecodeState,
        input_budget: usize,
    ) -> Result<NativeChatDecodeStep> {
        if let (Self::Qwen38(model), NativeChatDecodeState::Qwen38(state)) = (self, &mut *state) {
            let step = model.decode_quantum(state, input_budget.max(1))?;
            return Ok(NativeChatDecodeStep {
                delta: step.delta,
                text: step.text,
                tokens_generated: step.tokens_generated,
                input_tokens_committed: step.input_tokens_committed,
                finished: step.finished,
            });
        }
        let mut delta = String::new();
        let mut text = String::new();
        let mut tokens_generated = 0usize;
        let mut input_tokens_committed = 0usize;
        let mut finished = false;
        for _ in 0..input_budget.max(1) {
            let step = self.decode_step(state)?;
            delta.push_str(&step.delta);
            text = step.text;
            tokens_generated = step.tokens_generated;
            input_tokens_committed =
                input_tokens_committed.saturating_add(step.input_tokens_committed);
            finished = step.finished;
            if finished {
                break;
            }
        }
        Ok(NativeChatDecodeStep {
            delta,
            text,
            tokens_generated,
            input_tokens_committed,
            finished,
        })
    }

    pub fn decode_step_batch(
        &self,
        states: &mut [&mut NativeChatDecodeState],
    ) -> Result<Vec<NativeChatDecodeStep>> {
        let convert = |steps: Vec<crate::models::architectures::qwen3::chat::ChatDecodeStep>| {
            steps
                .into_iter()
                .map(|step| NativeChatDecodeStep {
                    delta: step.delta,
                    text: step.text,
                    tokens_generated: step.tokens_generated,
                    input_tokens_committed: 1,
                    finished: step.finished,
                })
                .collect()
        };
        match self {
            Self::Qwen3(model) => {
                let mut typed = Vec::with_capacity(states.len());
                for state in states.iter_mut() {
                    match &mut **state {
                        NativeChatDecodeState::Qwen3(state) => typed.push(state),
                        _ => {
                            return Err(Error::InvalidInput(
                                "Qwen3 continuous batch received another model's state".into(),
                            ))
                        }
                    }
                }
                model.decode_step_batch(&mut typed).map(convert)
            }
            Self::Gemma3(model) => {
                let mut typed = Vec::with_capacity(states.len());
                for state in states.iter_mut() {
                    match &mut **state {
                        NativeChatDecodeState::Gemma3(state) => typed.push(state),
                        _ => {
                            return Err(Error::InvalidInput(
                                "Gemma3 continuous batch received another model's state".into(),
                            ))
                        }
                    }
                }
                model.decode_step_batch(&mut typed).map(|steps| {
                    steps
                        .into_iter()
                        .map(|step| NativeChatDecodeStep {
                            delta: step.delta,
                            text: step.text,
                            tokens_generated: step.tokens_generated,
                            input_tokens_committed: 1,
                            finished: step.finished,
                        })
                        .collect()
                })
            }
            Self::Qwen38(model) => {
                let mut typed = Vec::with_capacity(states.len());
                for state in states.iter_mut() {
                    match &mut **state {
                        NativeChatDecodeState::Qwen38(state) => typed.push(state),
                        _ => {
                            return Err(Error::InvalidInput(
                                "Qwen3.8 continuous batch received another model's state".into(),
                            ))
                        }
                    }
                }
                model.decode_step_batch(&mut typed).map(|steps| {
                    steps
                        .into_iter()
                        .map(|step| NativeChatDecodeStep {
                            delta: step.delta,
                            text: step.text,
                            tokens_generated: step.tokens_generated,
                            input_tokens_committed: 1,
                            finished: step.finished,
                        })
                        .collect()
                })
            }
            Self::Qwen35(_) | Self::Lfm2(_) => Err(Error::InvalidInput(
                "Loaded chat model has no continuous tensor decode adapter".to_string(),
            )),
        }
    }
}

#[derive(Clone)]
pub struct ModelRegistry {
    models_dir: PathBuf,
    device: DeviceProfile,
    asr_models: Arc<RwLock<HashMap<ModelVariant, Arc<TrackedModelEntry<NativeAsrModel>>>>>,
    audio_chat_models: Arc<RwLock<HashMap<ModelVariant, Arc<OnceCell<Arc<NativeAudioChatModel>>>>>>,
    diarization_models:
        Arc<RwLock<HashMap<ModelVariant, Arc<OnceCell<Arc<NativeDiarizationModel>>>>>>,
    chat_models: Arc<RwLock<HashMap<ModelVariant, Arc<TrackedModelEntry<NativeChatModel>>>>>,
    voxtral_models: Arc<RwLock<HashMap<ModelVariant, Arc<OnceCell<Arc<VoxtralRealtimeModel>>>>>>,
    voxtral_tts_models: Arc<RwLock<HashMap<ModelVariant, Arc<OnceCell<Arc<VoxtralTtsModel>>>>>>,
    vibevoice_tts_models: Arc<RwLock<HashMap<ModelVariant, Arc<OnceCell<Arc<VibeVoiceTtsModel>>>>>>,
    fish_s2_tts_models: Arc<RwLock<HashMap<ModelVariant, Arc<OnceCell<Arc<FishS2TtsModel>>>>>>,
    qwen_tts_models: Arc<RwLock<HashMap<ModelVariant, Arc<TrackedModelEntry<Qwen3TtsModel>>>>>,
    kokoro_models: Arc<RwLock<HashMap<ModelVariant, Arc<OnceCell<Arc<KokoroTtsModel>>>>>>,
    effective_contexts: Arc<std::sync::RwLock<HashMap<ModelVariant, usize>>>,
}

/// Explicit name for the in-memory registry of loaded native model handles.
///
/// `ModelRegistry` remains the compatibility name used throughout the current
/// runtime. New architecture work should prefer `LoadedModelRegistry` when the
/// distinction from catalog and artifact registries matters.
pub type LoadedModelRegistry = ModelRegistry;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LoadedModelDiagnostics {
    pub variant_id: String,
    pub variant: String,
    pub family: &'static str,
    pub task: &'static str,
    pub handle_kind: &'static str,
    pub loaded_model_kind: &'static str,
    pub backend_kind: String,
    pub device_kind: String,
    pub actual_device_kind: Option<String>,
    pub actual_compute_dtype: Option<String>,
    pub default_compute_dtype: String,
    pub default_dtype_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effective_context_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_incremental_decode: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_realtime_stream_decode: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family_diagnostics: Option<Value>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct LoadedModelActualRuntime {
    device_kind: Option<String>,
    compute_dtype: Option<String>,
}

impl LoadedModelActualRuntime {
    fn from_values(device_kind: Option<&str>, compute_dtype: Option<&str>) -> Self {
        Self {
            device_kind: device_kind.and_then(normalize_observed_runtime_value),
            compute_dtype: compute_dtype.and_then(normalize_observed_runtime_value),
        }
    }

    fn from_diagnostics(diagnostics: &Value, device_pointer: &str, dtype_pointer: &str) -> Self {
        Self::from_values(
            diagnostics.pointer(device_pointer).and_then(Value::as_str),
            diagnostics.pointer(dtype_pointer).and_then(Value::as_str),
        )
    }
}

fn normalize_observed_runtime_value(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_ascii_lowercase())
}

fn loaded_model_diagnostics_entry(
    device: &DeviceProfile,
    variant: ModelVariant,
    handle_kind: &'static str,
    loaded_model_kind: &'static str,
    actual_runtime: LoadedModelActualRuntime,
    supports_incremental_decode: Option<bool>,
    supports_realtime_stream_decode: Option<bool>,
    family_diagnostics: Option<Value>,
) -> LoadedModelDiagnostics {
    let family = variant.family();
    let dtype_selection =
        device.resolve_dtype(DTypeSelectionRequest::new(None).with_model_family(family));

    LoadedModelDiagnostics {
        variant_id: variant.dir_name().to_string(),
        variant: variant.to_string(),
        family: model_family_name(family),
        task: model_task_name(variant.primary_task()),
        handle_kind,
        loaded_model_kind,
        backend_kind: format!("{:?}", device.kind).to_ascii_lowercase(),
        device_kind: format!("{:?}", device.kind),
        actual_device_kind: actual_runtime.device_kind,
        actual_compute_dtype: actual_runtime.compute_dtype,
        default_compute_dtype: format!("{:?}", dtype_selection.dtype).to_ascii_lowercase(),
        default_dtype_reason: dtype_selection.reason.into_owned(),
        effective_context_tokens: None,
        supports_incremental_decode,
        supports_realtime_stream_decode,
        family_diagnostics,
    }
}

fn model_family_name(family: ModelFamily) -> &'static str {
    match family {
        ModelFamily::Qwen3Tts => "qwen3_tts",
        ModelFamily::KokoroTts => "kokoro_tts",
        ModelFamily::VoxtralTts => "voxtral_tts",
        ModelFamily::VibeVoiceTts => "vibevoice_tts",
        ModelFamily::FishS2Tts => "fish_s2_tts",
        ModelFamily::ParakeetAsr => "parakeet_asr",
        ModelFamily::WhisperAsr => "whisper_asr",
        ModelFamily::Qwen3Asr => "qwen3_asr",
        ModelFamily::VibeVoiceAsr => "vibevoice_asr",
        ModelFamily::NemotronAsr => "nemotron_asr",
        ModelFamily::GraniteSpeechAsr => "granite_speech_asr",
        ModelFamily::SortformerDiarization => "sortformer_diarization",
        ModelFamily::Qwen3Chat => "qwen3_chat",
        ModelFamily::Qwen35Chat => "qwen35_chat",
        ModelFamily::Qwen38Chat => "qwen38_chat",
        ModelFamily::Lfm2Chat => "lfm2_chat",
        ModelFamily::Lfm25Audio => "lfm25_audio",
        ModelFamily::Gemma3Chat => "gemma3_chat",
        ModelFamily::Qwen3ForcedAligner => "qwen3_forced_aligner",
        ModelFamily::Voxtral => "voxtral",
        ModelFamily::Tokenizer => "tokenizer",
    }
}

fn model_task_name(task: ModelTask) -> &'static str {
    match task {
        ModelTask::Tts => "tts",
        ModelTask::Asr => "asr",
        ModelTask::Diarization => "diarization",
        ModelTask::Chat => "chat",
        ModelTask::ForcedAlign => "forced_align",
        ModelTask::AudioChat => "audio_chat",
        ModelTask::Tokenizer => "tokenizer",
    }
}

fn native_asr_model_kind(model: &NativeAsrModel) -> &'static str {
    match model {
        NativeAsrModel::Qwen3(_) => "qwen3_asr",
        NativeAsrModel::Parakeet(_) => "parakeet_asr",
        NativeAsrModel::Nemotron(_) => "nemotron_asr",
        NativeAsrModel::WhisperTurbo(_) => "whisper_turbo_asr",
        NativeAsrModel::VibeVoice(_) => "vibevoice_asr",
        NativeAsrModel::GraniteSpeech(_) => "granite_speech_asr",
    }
}

fn native_asr_runtime_diagnostics(
    model: &NativeAsrModel,
) -> (LoadedModelActualRuntime, Option<Value>) {
    match model {
        NativeAsrModel::Nemotron(model) => {
            let diagnostics = model.diagnostics();
            let actual_runtime = LoadedModelActualRuntime::from_diagnostics(
                &diagnostics,
                "/device",
                "/dtype_plan/activations",
            );
            (actual_runtime, Some(diagnostics))
        }
        NativeAsrModel::GraniteSpeech(model) => {
            let diagnostics = model.diagnostics_summary();
            let actual_runtime =
                LoadedModelActualRuntime::from_diagnostics(&diagnostics, "/device_kind", "/dtype");
            (actual_runtime, Some(diagnostics))
        }
        _ => (LoadedModelActualRuntime::default(), None),
    }
}

fn native_audio_chat_model_kind(model: &NativeAudioChatModel) -> &'static str {
    match model {
        NativeAudioChatModel::Lfm25Audio(_) => "lfm25_audio",
    }
}

fn native_diarization_model_kind(model: &NativeDiarizationModel) -> &'static str {
    match model {
        NativeDiarizationModel::Sortformer(_) => "sortformer_diarization",
    }
}

fn native_chat_model_kind(model: &NativeChatModel) -> &'static str {
    match model {
        NativeChatModel::Qwen3(_) => "qwen3_chat",
        NativeChatModel::Qwen35(_) => "qwen35_chat",
        NativeChatModel::Qwen38(_) => "qwen38_chat",
        NativeChatModel::Gemma3(_) => "gemma3_chat",
        NativeChatModel::Lfm2(_) => "lfm2_chat",
    }
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
            voxtral_tts_models: Arc::new(RwLock::new(HashMap::new())),
            vibevoice_tts_models: Arc::new(RwLock::new(HashMap::new())),
            fish_s2_tts_models: Arc::new(RwLock::new(HashMap::new())),
            qwen_tts_models: Arc::new(RwLock::new(HashMap::new())),
            kokoro_models: Arc::new(RwLock::new(HashMap::new())),
            effective_contexts: Arc::new(std::sync::RwLock::new(HashMap::new())),
        }
    }

    pub(crate) fn publish_effective_context(
        &self,
        variant: ModelVariant,
        tokens: u64,
    ) -> Result<()> {
        let tokens = usize::try_from(tokens)
            .map_err(|_| Error::ModelLoadError("effective context exceeds usize".into()))?;
        if tokens == 0 {
            return Err(Error::ModelLoadError(
                "effective context must be greater than zero".into(),
            ));
        }
        self.effective_contexts
            .write()
            .unwrap_or_else(|poison| poison.into_inner())
            .insert(variant, tokens);
        Ok(())
    }

    pub(crate) fn effective_context(&self, variant: ModelVariant) -> Option<usize> {
        self.effective_contexts
            .read()
            .unwrap_or_else(|poison| poison.into_inner())
            .get(&variant)
            .copied()
    }

    pub(crate) fn clear_effective_context(&self, variant: ModelVariant) {
        self.effective_contexts
            .write()
            .unwrap_or_else(|poison| poison.into_inner())
            .remove(&variant);
    }

    pub fn device(&self) -> &DeviceProfile {
        &self.device
    }

    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    pub async fn loaded_model_diagnostics(&self) -> Vec<LoadedModelDiagnostics> {
        let mut diagnostics = Vec::new();

        {
            let guard = self.asr_models.read().await;
            for (variant, entry) in guard.iter() {
                let Some(model) = entry.model.get() else {
                    continue;
                };
                let (actual_runtime, family_diagnostics) = native_asr_runtime_diagnostics(model);
                diagnostics.push(loaded_model_diagnostics_entry(
                    &self.device,
                    *variant,
                    "native_asr",
                    native_asr_model_kind(model),
                    actual_runtime,
                    Some(model.supports_incremental_decode()),
                    Some(model.supports_realtime_stream_decode()),
                    family_diagnostics,
                ));
            }
        }

        {
            let guard = self.audio_chat_models.read().await;
            for (variant, cell) in guard.iter() {
                let Some(model) = cell.get() else {
                    continue;
                };
                diagnostics.push(loaded_model_diagnostics_entry(
                    &self.device,
                    *variant,
                    "native_audio_chat",
                    native_audio_chat_model_kind(model),
                    match model.as_ref() {
                        NativeAudioChatModel::Lfm25Audio(model) => {
                            LoadedModelActualRuntime::from_values(
                                Some(&format!("{:?}", model.device().kind)),
                                None,
                            )
                        }
                    },
                    None,
                    None,
                    None,
                ));
            }
        }

        {
            let guard = self.diarization_models.read().await;
            for (variant, cell) in guard.iter() {
                let Some(model) = cell.get() else {
                    continue;
                };
                diagnostics.push(loaded_model_diagnostics_entry(
                    &self.device,
                    *variant,
                    "native_diarization",
                    native_diarization_model_kind(model),
                    LoadedModelActualRuntime::default(),
                    None,
                    None,
                    None,
                ));
            }
        }

        {
            let guard = self.chat_models.read().await;
            for (variant, entry) in guard.iter() {
                let Some(model) = entry.model.get() else {
                    continue;
                };
                diagnostics.push(loaded_model_diagnostics_entry(
                    &self.device,
                    *variant,
                    "native_chat",
                    native_chat_model_kind(model),
                    match model.as_ref() {
                        NativeChatModel::Qwen3(model) => LoadedModelActualRuntime::from_values(
                            Some(model.runtime_device_kind().as_str()),
                            model.runtime_compute_dtype().as_deref(),
                        ),
                        NativeChatModel::Qwen38(model) => LoadedModelActualRuntime::from_values(
                            Some(model.device_kind().as_str()),
                            model.runtime_compute_dtype(),
                        ),
                        _ => LoadedModelActualRuntime::default(),
                    },
                    Some(model.supports_incremental_decode()),
                    None,
                    match model.as_ref() {
                        NativeChatModel::Qwen38(model) => Some(model.runtime_diagnostics()),
                        _ => None,
                    },
                ));
            }
        }

        {
            let guard = self.voxtral_models.read().await;
            for (variant, cell) in guard.iter() {
                if cell.get().is_some() {
                    diagnostics.push(loaded_model_diagnostics_entry(
                        &self.device,
                        *variant,
                        "voxtral_realtime",
                        "voxtral_realtime",
                        LoadedModelActualRuntime::default(),
                        None,
                        Some(true),
                        None,
                    ));
                }
            }
        }

        {
            let guard = self.voxtral_tts_models.read().await;
            for (variant, cell) in guard.iter() {
                if cell.get().is_some() {
                    diagnostics.push(loaded_model_diagnostics_entry(
                        &self.device,
                        *variant,
                        "voxtral_tts",
                        "voxtral_tts",
                        LoadedModelActualRuntime::default(),
                        None,
                        None,
                        None,
                    ));
                }
            }
        }

        {
            let guard = self.vibevoice_tts_models.read().await;
            for (variant, cell) in guard.iter() {
                let Some(model) = cell.get() else {
                    continue;
                };
                let model_diagnostics = model.diagnostics();
                let actual_runtime = LoadedModelActualRuntime::from_values(
                    Some(&model_diagnostics.device_kind),
                    Some(&model_diagnostics.dtype),
                );
                diagnostics.push(loaded_model_diagnostics_entry(
                    &self.device,
                    *variant,
                    "vibevoice_tts",
                    "vibevoice_tts",
                    actual_runtime,
                    None,
                    None,
                    serde_json::to_value(model_diagnostics).ok(),
                ));
            }
        }

        {
            let guard = self.fish_s2_tts_models.read().await;
            for (variant, cell) in guard.iter() {
                let Some(model) = cell.get() else {
                    continue;
                };
                diagnostics.push(loaded_model_diagnostics_entry(
                    &self.device,
                    *variant,
                    "fish_s2_tts",
                    "fish_s2_tts",
                    LoadedModelActualRuntime::default(),
                    None,
                    None,
                    serde_json::to_value(model.diagnostics()).ok(),
                ));
            }
        }

        {
            let guard = self.qwen_tts_models.read().await;
            for (variant, entry) in guard.iter() {
                let Some(model) = entry.model.get() else {
                    continue;
                };
                let model_diagnostics = model.diagnostics();
                let actual_runtime = LoadedModelActualRuntime::from_values(
                    Some(&model_diagnostics.device_kind),
                    Some(&model_diagnostics.talker_dtype),
                );
                diagnostics.push(loaded_model_diagnostics_entry(
                    &self.device,
                    *variant,
                    "qwen3_tts",
                    "qwen3_tts",
                    actual_runtime,
                    None,
                    None,
                    serde_json::to_value(model_diagnostics).ok(),
                ));
            }
        }

        {
            let guard = self.kokoro_models.read().await;
            for (variant, cell) in guard.iter() {
                if cell.get().is_some() {
                    diagnostics.push(loaded_model_diagnostics_entry(
                        &self.device,
                        *variant,
                        "kokoro_tts",
                        "kokoro_tts",
                        LoadedModelActualRuntime::default(),
                        None,
                        None,
                        None,
                    ));
                }
            }
        }

        for entry in &mut diagnostics {
            entry.effective_context_tokens = self
                .effective_contexts
                .read()
                .unwrap_or_else(|poison| poison.into_inner())
                .iter()
                .find_map(|(variant, tokens)| {
                    (variant.dir_name() == entry.variant_id).then_some(*tokens)
                });
        }
        diagnostics.sort_by(|left, right| {
            left.variant_id
                .cmp(&right.variant_id)
                .then_with(|| left.handle_kind.cmp(right.handle_kind))
        });
        diagnostics
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

        let (entry, loading_guard) = {
            let mut guard = self.asr_models.write().await;
            let entry = guard
                .entry(variant)
                .or_insert_with(|| Arc::new(TrackedModelEntry::default()))
                .clone();
            let loading_guard = entry.uses.acquire().ok_or_else(|| {
                Error::ModelLoadError(format!(
                    "ASR model {variant} exceeded its active-use accounting capacity"
                ))
            })?;
            (entry, loading_guard)
        };

        info!(
            "Loading native ASR/ForcedAligner model {variant} ({}) from {model_dir:?}",
            registration.name
        );

        entry
            .model
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

        let model = {
            let guard = self.asr_models.read().await;
            guard
                .get(&variant)
                .filter(|current| Arc::ptr_eq(current, &entry))
                .and_then(|current| current.model.get().cloned())
        };
        drop(loading_guard);
        model.ok_or_else(|| {
            Error::ModelLoadError(format!(
                "ASR model {variant} load was superseded before publication"
            ))
        })
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
    ) -> Result<ChatModelLease> {
        let registration = resolve_chat_loader_registration(variant).ok_or_else(|| {
            Error::InvalidInput(format!("Unsupported chat model variant: {variant}"))
        })?;

        let (entry, loading_guard) = {
            let mut guard = self.chat_models.write().await;
            let entry = guard
                .entry(variant)
                .or_insert_with(|| Arc::new(TrackedModelEntry::default()))
                .clone();
            let loading_guard = entry.uses.acquire().ok_or_else(|| {
                Error::ModelLoadError(format!(
                    "Chat model {variant} exceeded its active-use accounting capacity"
                ))
            })?;
            (entry, loading_guard)
        };

        info!(
            "Loading native chat model {variant} ({}) from {model_dir:?}",
            registration.name
        );

        entry
            .model
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

        // Acquire the published handle while holding the registry read lock.
        // If an unload removed or superseded this entry while loading, fail
        // rather than returning an untracked model handle.
        let lease = {
            let guard = self.chat_models.read().await;
            guard
                .get(&variant)
                .filter(|current| Arc::ptr_eq(current, &entry))
                .and_then(|current| current.acquire())
                .map(|inner| ChatModelLease { inner })
        };
        drop(loading_guard);
        lease.ok_or_else(|| {
            Error::ModelLoadError(format!(
                "Chat model {variant} load was superseded before publication"
            ))
        })
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
                let device = self.device.clone();
                let loader = registration.loader;
                move || async move {
                    tokio::task::spawn_blocking(move || {
                        let model = loader(&model_dir, variant, device)?;
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

        let (entry, loading_guard) = {
            let mut guard = self.qwen_tts_models.write().await;
            let entry = guard
                .entry(variant)
                .or_insert_with(|| Arc::new(TrackedModelEntry::default()))
                .clone();
            let loading_guard = entry.uses.acquire().ok_or_else(|| {
                Error::ModelLoadError(format!(
                    "Qwen TTS model {variant} exceeded its active-use accounting capacity"
                ))
            })?;
            (entry, loading_guard)
        };

        info!(
            "Loading Qwen TTS model {variant} ({}) from {model_dir:?}",
            registration.name
        );

        entry
            .model
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

        let model = {
            let guard = self.qwen_tts_models.read().await;
            guard
                .get(&variant)
                .filter(|current| Arc::ptr_eq(current, &entry))
                .and_then(|current| current.model.get().cloned())
        };
        drop(loading_guard);
        model.ok_or_else(|| {
            Error::ModelLoadError(format!(
                "Qwen TTS model {variant} load was superseded before publication"
            ))
        })
    }

    pub async fn load_voxtral_tts(
        &self,
        variant: ModelVariant,
        model_dir: &Path,
    ) -> Result<Arc<VoxtralTtsModel>> {
        let registration = resolve_voxtral_tts_loader_registration(variant).ok_or_else(|| {
            Error::InvalidInput(format!("Unsupported Voxtral TTS model variant: {variant}"))
        })?;

        let cell = {
            let mut guard = self.voxtral_tts_models.write().await;
            guard
                .entry(variant)
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone()
        };

        info!(
            "Loading Voxtral TTS model {variant} ({}) from {model_dir:?}",
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

    pub async fn load_vibevoice_tts(
        &self,
        variant: ModelVariant,
        model_dir: &Path,
    ) -> Result<Arc<VibeVoiceTtsModel>> {
        let registration = resolve_vibevoice_tts_loader_registration(variant).ok_or_else(|| {
            Error::InvalidInput(format!(
                "Unsupported VibeVoice TTS model variant: {variant}"
            ))
        })?;

        let cell = {
            let mut guard = self.vibevoice_tts_models.write().await;
            guard
                .entry(variant)
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone()
        };

        info!(
            "Loading VibeVoice TTS model {variant} ({}) from {model_dir:?}",
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

    pub async fn load_fish_s2_tts(
        &self,
        variant: ModelVariant,
        model_dir: &Path,
    ) -> Result<Arc<FishS2TtsModel>> {
        let registration = resolve_fish_s2_tts_loader_registration(variant).ok_or_else(|| {
            Error::InvalidInput(format!("Unsupported Fish S2 TTS model variant: {variant}"))
        })?;

        let cell = {
            let mut guard = self.fish_s2_tts_models.write().await;
            guard
                .entry(variant)
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone()
        };

        info!(
            "Loading Fish S2 TTS model metadata {variant} ({}) from {model_dir:?}",
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
        guard
            .get(&variant)
            .and_then(|entry| entry.model.get().cloned())
    }

    pub fn try_get_asr(&self, variant: ModelVariant) -> Option<Arc<NativeAsrModel>> {
        let guard = self.asr_models.try_read().ok()?;
        guard
            .get(&variant)
            .and_then(|entry| entry.model.get().cloned())
    }

    pub async fn get_asr_lease(&self, variant: ModelVariant) -> Option<AsrModelLease> {
        let guard = self.asr_models.read().await;
        guard
            .get(&variant)
            .and_then(|entry| entry.acquire())
            .map(|inner| AsrModelLease { inner })
    }

    pub fn try_get_asr_lease(&self, variant: ModelVariant) -> Option<AsrModelLease> {
        let guard = self.asr_models.try_read().ok()?;
        guard
            .get(&variant)
            .and_then(|entry| entry.acquire())
            .map(|inner| AsrModelLease { inner })
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

    pub async fn get_chat(&self, variant: ModelVariant) -> Option<ChatModelLease> {
        let guard = self.chat_models.read().await;
        guard
            .get(&variant)
            .and_then(|entry| entry.acquire())
            .map(|inner| ChatModelLease { inner })
    }

    /// Resolve a loaded chat model from a dedicated blocking worker.
    ///
    /// Callers must not invoke this on an async runtime thread. Runtime prompt
    /// preparation uses it from `spawn_blocking` so transient registry lock
    /// contention cannot be misreported as a missing model.
    pub(crate) fn blocking_get_chat(&self, variant: ModelVariant) -> Option<ChatModelLease> {
        let guard = self.chat_models.blocking_read();
        guard
            .get(&variant)
            .and_then(|entry| entry.acquire())
            .map(|inner| ChatModelLease { inner })
    }

    pub async fn get_audio_chat(&self, variant: ModelVariant) -> Option<Arc<NativeAudioChatModel>> {
        let guard = self.audio_chat_models.read().await;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub fn try_get_chat(&self, variant: ModelVariant) -> Option<ChatModelLease> {
        let guard = self.chat_models.try_read().ok()?;
        guard
            .get(&variant)
            .and_then(|entry| entry.acquire())
            .map(|inner| ChatModelLease { inner })
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

    pub async fn get_voxtral_tts(&self, variant: ModelVariant) -> Option<Arc<VoxtralTtsModel>> {
        let guard = self.voxtral_tts_models.read().await;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub fn try_get_voxtral_tts(&self, variant: ModelVariant) -> Option<Arc<VoxtralTtsModel>> {
        let guard = self.voxtral_tts_models.try_read().ok()?;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub async fn get_vibevoice_tts(&self, variant: ModelVariant) -> Option<Arc<VibeVoiceTtsModel>> {
        let guard = self.vibevoice_tts_models.read().await;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub fn try_get_vibevoice_tts(&self, variant: ModelVariant) -> Option<Arc<VibeVoiceTtsModel>> {
        let guard = self.vibevoice_tts_models.try_read().ok()?;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub async fn get_fish_s2_tts(&self, variant: ModelVariant) -> Option<Arc<FishS2TtsModel>> {
        let guard = self.fish_s2_tts_models.read().await;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub fn try_get_fish_s2_tts(&self, variant: ModelVariant) -> Option<Arc<FishS2TtsModel>> {
        let guard = self.fish_s2_tts_models.try_read().ok()?;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub async fn get_qwen_tts(&self, variant: ModelVariant) -> Option<Arc<Qwen3TtsModel>> {
        let guard = self.qwen_tts_models.read().await;
        guard
            .get(&variant)
            .and_then(|entry| entry.model.get().cloned())
    }

    pub fn try_get_qwen_tts(&self, variant: ModelVariant) -> Option<Arc<Qwen3TtsModel>> {
        let guard = self.qwen_tts_models.try_read().ok()?;
        guard
            .get(&variant)
            .and_then(|entry| entry.model.get().cloned())
    }

    pub async fn get_qwen_tts_lease(&self, variant: ModelVariant) -> Option<QwenTtsModelLease> {
        let guard = self.qwen_tts_models.read().await;
        guard
            .get(&variant)
            .and_then(|entry| entry.acquire())
            .map(|inner| QwenTtsModelLease { inner })
    }

    pub fn try_get_qwen_tts_lease(&self, variant: ModelVariant) -> Option<QwenTtsModelLease> {
        let guard = self.qwen_tts_models.try_read().ok()?;
        guard
            .get(&variant)
            .and_then(|entry| entry.acquire())
            .map(|inner| QwenTtsModelLease { inner })
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
        let entry = {
            let mut guard = self.asr_models.write().await;
            guard.remove(&variant)
        };
        if let Some(entry) = entry {
            entry.uses.wait_until_idle().await;
        }
    }

    pub async fn unload_diarization(&self, variant: ModelVariant) {
        let mut guard = self.diarization_models.write().await;
        guard.remove(&variant);
    }

    pub async fn unload_chat(&self, variant: ModelVariant) {
        let entry = {
            let mut guard = self.chat_models.write().await;
            guard.remove(&variant)
        };
        if let Some(entry) = entry {
            entry.uses.wait_until_idle().await;
        }
    }

    pub async fn unload_audio_chat(&self, variant: ModelVariant) {
        let mut guard = self.audio_chat_models.write().await;
        guard.remove(&variant);
    }

    pub async fn unload_voxtral(&self, variant: ModelVariant) {
        let mut guard = self.voxtral_models.write().await;
        guard.remove(&variant);
    }

    pub async fn unload_voxtral_tts(&self, variant: ModelVariant) {
        let mut guard = self.voxtral_tts_models.write().await;
        guard.remove(&variant);
    }

    pub async fn unload_vibevoice_tts(&self, variant: ModelVariant) {
        let mut guard = self.vibevoice_tts_models.write().await;
        guard.remove(&variant);
    }

    pub async fn unload_fish_s2_tts(&self, variant: ModelVariant) {
        let mut guard = self.fish_s2_tts_models.write().await;
        guard.remove(&variant);
    }

    pub async fn unload_qwen_tts(&self, variant: ModelVariant) {
        let entry = {
            let mut guard = self.qwen_tts_models.write().await;
            guard.remove(&variant)
        };
        if let Some(entry) = entry {
            entry.uses.wait_until_idle().await;
        }
    }

    pub async fn unload_kokoro(&self, variant: ModelVariant) {
        let mut guard = self.kokoro_models.write().await;
        guard.remove(&variant);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn loaded_model_diagnostics_entry_reports_backend_family_and_dtype_policy() {
        let diagnostics = loaded_model_diagnostics_entry(
            &DeviceProfile::cpu(),
            ModelVariant::Qwen306BGguf,
            "native_chat",
            "qwen3_chat",
            LoadedModelActualRuntime::default(),
            Some(true),
            None,
            None,
        );

        assert_eq!(diagnostics.variant_id, "Qwen3-0.6B-GGUF");
        assert_eq!(diagnostics.family, "qwen3_chat");
        assert_eq!(diagnostics.task, "chat");
        assert_eq!(diagnostics.handle_kind, "native_chat");
        assert_eq!(diagnostics.loaded_model_kind, "qwen3_chat");
        assert_eq!(diagnostics.backend_kind, "cpu");
        assert_eq!(diagnostics.device_kind, "Cpu");
        assert_eq!(diagnostics.actual_device_kind, None);
        assert_eq!(diagnostics.actual_compute_dtype, None);
        assert_eq!(diagnostics.default_compute_dtype, "f32");
        assert_eq!(diagnostics.supports_incremental_decode, Some(true));
        assert!(diagnostics.default_dtype_reason.contains("CPU"));
    }

    #[test]
    fn loaded_model_diagnostics_keeps_observed_runtime_separate_from_policy() {
        let diagnostics = loaded_model_diagnostics_entry(
            &DeviceProfile::cpu(),
            ModelVariant::Qwen306BGguf,
            "native_chat",
            "qwen3_chat",
            LoadedModelActualRuntime::from_values(Some("CUDA"), Some("BF16")),
            Some(true),
            None,
            None,
        );

        assert_eq!(diagnostics.actual_device_kind.as_deref(), Some("cuda"));
        assert_eq!(diagnostics.actual_compute_dtype.as_deref(), Some("bf16"));
        assert_eq!(diagnostics.default_compute_dtype, "f32");
    }

    #[tokio::test]
    async fn loaded_model_diagnostics_empty_until_handles_are_initialized() {
        let registry = ModelRegistry::new(PathBuf::from("/tmp/models"), DeviceProfile::cpu());

        assert!(registry.loaded_model_diagnostics().await.is_empty());
    }

    #[tokio::test]
    async fn model_use_state_fences_unload_until_last_shared_lease_drops() {
        let state = Arc::new(ModelUseState::default());
        let lease = state.acquire().expect("first model-use lease");
        let lease_clone = lease.clone();
        let waiter_state = state.clone();
        let waiter = tokio::spawn(async move {
            waiter_state.wait_until_idle().await;
        });

        tokio::task::yield_now().await;
        assert!(!waiter.is_finished());
        drop(lease);
        tokio::task::yield_now().await;
        assert!(
            !waiter.is_finished(),
            "shared lease clone must retain usage"
        );
        drop(lease_clone);

        tokio::time::timeout(std::time::Duration::from_secs(1), waiter)
            .await
            .expect("unload fence should observe the final lease drop")
            .expect("unload fence task should complete");
        assert_eq!(state.active.load(Ordering::Acquire), 0);
    }

    async fn assert_session_lease_survives_concurrent_reload(old: &'static str, new: &'static str) {
        let models = Arc::new(RwLock::new(HashMap::<
            &'static str,
            Arc<TrackedModelEntry<&'static str>>,
        >::new()));
        let old_entry = Arc::new(TrackedModelEntry::default());
        old_entry.model.set(Arc::new(old)).expect("seed old model");
        models.write().await.insert("model", old_entry.clone());
        let session_lease = old_entry.acquire().expect("acquire old session lease");

        let (removed_tx, removed_rx) = tokio::sync::oneshot::channel();
        let unload_models = models.clone();
        let unload = tokio::spawn(async move {
            let removed = unload_models
                .write()
                .await
                .remove("model")
                .expect("old entry remains discoverable until unload");
            removed_tx.send(()).expect("signal registry removal");
            removed.uses.wait_until_idle().await;
        });
        removed_rx.await.expect("observe registry removal");

        let replacement = Arc::new(TrackedModelEntry::default());
        replacement
            .model
            .set(Arc::new(new))
            .expect("seed replacement model");
        models.write().await.insert("model", replacement.clone());
        let replacement_lease = replacement.acquire().expect("acquire replacement lease");

        assert_eq!(*session_lease, old);
        assert_eq!(*replacement_lease, new);
        assert!(
            !unload.is_finished(),
            "old unload must wait for its session"
        );
        drop(session_lease);
        tokio::time::timeout(std::time::Duration::from_secs(1), unload)
            .await
            .expect("old unload should finish after its exact session drops")
            .expect("unload task should complete");
    }

    #[tokio::test]
    async fn asr_and_qwen_tts_sessions_keep_exact_identity_across_unload_reload() {
        assert_session_lease_survives_concurrent_reload("old-asr", "new-asr").await;
        assert_session_lease_survives_concurrent_reload("old-qwen-tts", "new-qwen-tts").await;
    }

    #[test]
    fn resolves_vibevoice_asr_loader_registration() {
        let registration = resolve_asr_loader_registration(ModelVariant::VibeVoiceAsr)
            .expect("VibeVoice-ASR loader should be registered");

        assert_eq!(registration.name, "vibevoice_asr");
        assert_eq!(registration.family, ModelFamily::VibeVoiceAsr);
    }

    #[test]
    fn resolves_nemotron_asr_loader_registration() {
        let registration = resolve_asr_loader_registration(ModelVariant::Nemotron35AsrStreaming06B)
            .expect("Nemotron ASR loader should be registered");

        assert_eq!(registration.name, "nemotron_asr");
        assert_eq!(registration.family, ModelFamily::NemotronAsr);
    }

    #[test]
    fn resolves_granite_speech_asr_loader_registration() {
        let registration = resolve_asr_loader_registration(ModelVariant::GraniteSpeech412BPlus)
            .expect("Granite Speech ASR loader should be registered");

        assert_eq!(registration.name, "granite_speech_asr");
        assert_eq!(registration.family, ModelFamily::GraniteSpeechAsr);
    }

    #[test]
    fn vibevoice_tts_is_not_registered_as_asr() {
        assert!(resolve_asr_loader_registration(ModelVariant::VibeVoice15BTts).is_none());
    }

    #[test]
    fn resolves_vibevoice_tts_loader_registration() {
        let registration = resolve_vibevoice_tts_loader_registration(ModelVariant::VibeVoice15BTts)
            .expect("VibeVoice TTS loader should be registered");

        assert_eq!(registration.name, "vibevoice_tts");
        assert_eq!(registration.family, ModelFamily::VibeVoiceTts);
    }

    #[test]
    fn resolves_fish_s2_tts_loader_registration() {
        let registration = resolve_fish_s2_tts_loader_registration(ModelVariant::FishAudioS2Pro)
            .expect("Fish S2 TTS loader should be registered");

        assert_eq!(registration.name, "fish_s2_tts");
        assert_eq!(registration.family, ModelFamily::FishS2Tts);
    }

    #[test]
    fn lfm_audio_asr_chunk_budget_scales_and_caps() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let previous = std::env::var("IZWI_LFM25_ASR_CHUNK_MAX_NEW_TOKENS").ok();
        std::env::remove_var("IZWI_LFM25_ASR_CHUNK_MAX_NEW_TOKENS");

        assert_eq!(
            lfm25_audio_asr_chunk_max_new_tokens(0.0, None),
            LFM25_AUDIO_ASR_MIN_CHUNK_NEW_TOKENS
        );
        assert_eq!(lfm25_audio_asr_chunk_max_new_tokens(24.0, None), 192);
        assert_eq!(
            lfm25_audio_asr_chunk_max_new_tokens(60.0, None),
            LFM25_AUDIO_ASR_MAX_CHUNK_NEW_TOKENS
        );
        assert_eq!(lfm25_audio_asr_chunk_max_new_tokens(24.0, Some(42)), 42);

        if let Some(previous) = previous {
            std::env::set_var("IZWI_LFM25_ASR_CHUNK_MAX_NEW_TOKENS", previous);
        }
    }

    #[test]
    fn lfm_audio_asr_single_pass_preserves_legacy_default_cap() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let previous = std::env::var("IZWI_LFM25_ASR_MAX_NEW_TOKENS").ok();
        std::env::remove_var("IZWI_LFM25_ASR_MAX_NEW_TOKENS");

        assert_eq!(
            lfm25_audio_asr_single_pass_max_new_tokens(None),
            LFM25_AUDIO_ASR_DEFAULT_MAX_NEW_TOKENS
        );
        assert_eq!(lfm25_audio_asr_single_pass_max_new_tokens(Some(64)), 64);

        if let Some(previous) = previous {
            std::env::set_var("IZWI_LFM25_ASR_MAX_NEW_TOKENS", previous);
        }
    }
}
