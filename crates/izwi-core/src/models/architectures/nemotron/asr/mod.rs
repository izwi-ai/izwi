//! Nemotron 3.5 ASR artifact and native inference support.

pub mod config;
mod metal_kernels;
pub mod nemo;
mod network;
mod physical;

use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde_json::json;
use sha2::{Digest, Sha256};
use tracing::info;

use crate::backends::state::{PhysicalStateTransactionId, StateComponentValue};
use crate::backends::{DTypeSelection, DTypeSelectionRequest, DeviceProfile};
use crate::catalog::ModelFamily;
use crate::engine::{InvocationTensorLease, RetainedTensorStateRuntimeV2, StageDescriptor};
use crate::error::{Error, Result};
use crate::kv::v2::{
    AppendStateDomainSpec, BoundedShape, CapabilityStateDescriptorV2, CheckpointPolicy,
    InferenceStateContract, PlacementPolicy, PrefixPolicy, RingStateDomainSpec, ShapeAxis,
    ShapeDimension, ShapeExtent, StateClock, StateComponentId, StateDType, StateDomainHeader,
    StateDomainId, StateDomainSpec, StateGroupId, StateGroupSpec, StateScope, TensorComponentSpec,
    TensorRole, TensorStateDomainSpec, CURRENT_INFERENCE_STATE_ABI,
};
use crate::model::ModelVariant;
use crate::models::shared::memory::accounting::TensorStorageAccounting;
use crate::tokenizer::Tokenizer;

pub use config::NemotronConfigInventory;
pub use nemo::{ensure_nemotron_artifacts, NemotronArtifacts, NEMOTRON_NEMO_FILENAME};
pub(crate) use network::NEMOTRON_MODEL_MEMO_MAX_BYTES;
use network::{
    default_realtime_state_shape, resample_linear, NemotronEncodeProfile, NemotronNetwork,
    NemotronRealtimeStateShape, NemotronRnntStreamState, NemotronStreamingEncoderState,
    NemotronStreamingFeatureState, NemotronStreamingPreEncodeState,
};
pub(crate) use physical::NemotronOfflinePhysicalStateSpec;

const SAMPLE_RATE: u32 = 16_000;
const DEFAULT_STRIP_LANG_TAGS: bool = true;
const DEFAULT_MAX_AUDIO_SECONDS_HINT: f32 = 30.0;
const STREAMING_FRAME_MS: usize = 80;
const NEMOTRON_ASR_DTYPE_ENV: &str = "IZWI_NEMOTRON_ASR_DTYPE";
const NEMOTRON_REALTIME_MAX_SECONDS_ENV: &str = "IZWI_NEMOTRON_REALTIME_MAX_SECONDS";
const DEFAULT_NEMOTRON_REALTIME_MAX_SECONDS: usize = 300;
const CONSERVATIVE_DECODED_TOKEN_EXPANSION: usize = 4;
const CONSERVATIVE_NEMOTRON_DECODER_TOKEN_BYTES: usize = 256;
const REALTIME_HOST_FIXED_OVERHEAD_BYTES: u64 = 64 * 1024;
const REALTIME_PEAK_TEXT_COPIES: u64 = 4;
pub(crate) const NEMOTRON_REALTIME_STAGE_WORKSPACE_BYTES: u64 = 64 * 1024 * 1024;
pub(crate) const NEMOTRON_REALTIME_ENCODER_STAGE: &str = "asr.realtime.encoder";
pub(crate) const NEMOTRON_REALTIME_RNNT_STAGE: &str = "asr.realtime.rnnt";
pub(crate) const NEMOTRON_REALTIME_FALLBACK_STAGE: &str = "asr.realtime.scalar_fallback";
pub(crate) const NEMOTRON_ENCODER_STATE_GROUP: StateGroupId = StateGroupId::new(1);
pub(crate) const NEMOTRON_RNNT_STATE_GROUP: StateGroupId = StateGroupId::new(2);
const NEMOTRON_FEATURE_HISTORY_DOMAIN: StateDomainId = StateDomainId::new(1);
const NEMOTRON_PENDING_ENCODER_DOMAIN: StateDomainId = StateDomainId::new(2);
const NEMOTRON_ATTENTION_HISTORY_DOMAIN: StateDomainId = StateDomainId::new(3);
const NEMOTRON_CONVOLUTION_HISTORY_DOMAIN: StateDomainId = StateDomainId::new(4);
const NEMOTRON_RNNT_STATE_DOMAIN: StateDomainId = StateDomainId::new(5);
const SUPPORTED_TARGET_LANGS: &[&str] = &[
    "auto", "en-US", "en-GB", "es-US", "es-ES", "fr-FR", "fr-CA", "it-IT", "pt-BR", "pt-PT",
    "nl-NL", "de-DE", "tr-TR", "ru-RU", "ar-AR", "hi-IN", "ja-JP", "ko-KR", "vi-VN", "uk-UA",
    "pl-PL", "sv-SE", "cs-CZ", "nb-NO", "da-DK", "bg-BG", "fi-FI", "hr-HR", "sk-SK", "zh-CN",
    "hu-HU", "ro-RO", "et-EE", "el-GR", "lt-LT", "lv-LV", "mt-MT", "sl-SI", "he-IL", "th-TH",
    "nn-NO",
];

#[derive(Debug, Clone)]
pub struct NemotronAsrTranscriptionOutput {
    pub text: String,
    pub language: Option<String>,
    pub diagnostics: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct NemotronAsrDecodeStep {
    pub delta: String,
    pub text: String,
    pub tokens_generated: usize,
    pub finished: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NemotronRealtimeResourceReservation {
    pub max_samples: usize,
    pub host_bytes: u64,
    pub tensor_bytes: u64,
    max_output_tokens: usize,
    max_text_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NemotronRealtimeResourceUsage {
    pub host_bytes: u64,
    pub tensor_bytes: u64,
}

pub(crate) struct NemotronRealtimePhysicalStateSpec {
    pub(crate) retained: InferenceStateContract,
    pub(crate) descriptor: CapabilityStateDescriptorV2,
}

pub struct NemotronAsrModel {
    variant: ModelVariant,
    artifacts: NemotronArtifacts,
    decoder: NemotronDecoder,
    network: NemotronNetwork,
    runtime_plan: NemotronRuntimePlan,
    device_profile: DeviceProfile,
    dtype_selection: DTypeSelection,
}

struct NemotronOfflinePhysicalState<'a> {
    predictor: &'a mut InvocationTensorLease,
    acoustic: &'a mut InvocationTensorLease,
    source_identity: [u8; 32],
}

enum NemotronDecoder {
    HfTokenizer(Tokenizer),
    ConfigLabels(Vec<String>),
    Vocab(Vec<String>),
}

impl NemotronDecoder {
    fn load(artifacts: &NemotronArtifacts) -> Result<Self> {
        if !artifacts.config_inventory.output_vocabulary.is_empty() {
            return Ok(Self::ConfigLabels(
                artifacts.config_inventory.output_vocabulary.clone(),
            ));
        }

        if let Ok(tokenizer) = Tokenizer::from_path(&artifacts.extracted_dir) {
            return Ok(Self::HfTokenizer(tokenizer));
        }

        for path in &artifacts.tokenizer_paths {
            if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name == "tokenizer.vocab" || name == "vocab.txt")
            {
                return Ok(Self::Vocab(load_tokenizer_vocab(path)?));
            }
        }

        let asset_list = artifacts
            .tokenizer_paths
            .iter()
            .filter_map(|path| path.file_name().and_then(|name| name.to_str()))
            .collect::<Vec<_>>()
            .join(", ");
        Err(Error::ModelLoadError(format!(
            "Nemotron tokenizer assets do not include a supported decoder at {} (found: {})",
            artifacts.extracted_dir.display(),
            if asset_list.is_empty() {
                "none"
            } else {
                &asset_list
            }
        )))
    }

    fn decode(&self, ids: &[usize]) -> String {
        match self {
            Self::HfTokenizer(tokenizer) => {
                let ids = ids.iter().map(|id| *id as u32).collect::<Vec<_>>();
                tokenizer.decode(&ids).unwrap_or_default()
            }
            Self::ConfigLabels(vocab) => decode_vocab_tokens(ids, vocab),
            Self::Vocab(vocab) => decode_vocab_tokens(ids, vocab),
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            Self::HfTokenizer(tokenizer) => tokenizer.vocab_size(),
            Self::ConfigLabels(vocab) => vocab.len(),
            Self::Vocab(vocab) => vocab.len(),
        }
    }

    fn source(&self) -> &'static str {
        match self {
            Self::HfTokenizer(_) => "huggingface_tokenizer",
            Self::ConfigLabels(_) => "config_labels",
            Self::Vocab(_) => "vocab_file",
        }
    }

    fn max_token_bytes(&self) -> usize {
        match self {
            Self::HfTokenizer(tokenizer) => tokenizer
                .vocab()
                .keys()
                .map(|token| token.len())
                .max()
                .unwrap_or(1),
            Self::ConfigLabels(vocab) | Self::Vocab(vocab) => {
                vocab.iter().map(String::len).max().unwrap_or(1)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NemotronRuntimePlan {
    sample_rate: u32,
    feature_bins: Option<usize>,
    n_fft: Option<usize>,
    window_length: Option<usize>,
    hop_length: Option<usize>,
    normalize: Option<String>,
    encoder_layers: Option<usize>,
    encoder_dim: Option<usize>,
    encoder_heads: Option<usize>,
    subsampling_factor: Option<usize>,
    subsampling_conv_channels: Option<usize>,
    ff_expansion_factor: Option<usize>,
    conv_kernel_size: Option<usize>,
    predictor_hidden: Option<usize>,
    predictor_layers: Option<usize>,
    joint_hidden: Option<usize>,
    prompt_dim: Option<usize>,
    prompt_dictionary_size: usize,
    vocab_size: Option<usize>,
    default_streaming_profile: NemotronStreamingProfile,
    streaming_profiles: Vec<NemotronStreamingProfile>,
}

impl NemotronRuntimePlan {
    fn from_inventory(inventory: &NemotronConfigInventory) -> Result<Self> {
        let sample_rate = inventory.sample_rate.unwrap_or(SAMPLE_RATE as usize);
        if sample_rate != SAMPLE_RATE as usize {
            return Err(Error::ModelLoadError(format!(
                "Nemotron config advertises sample_rate={sample_rate}, expected {SAMPLE_RATE}"
            )));
        }

        let streaming_profiles = NemotronStreamingProfile::profiles_from_inventory(inventory)?;
        let default_streaming_profile = streaming_profiles.last().cloned().ok_or_else(|| {
            Error::ModelLoadError("Nemotron config did not yield a streaming profile".to_string())
        })?;

        Ok(Self {
            sample_rate: SAMPLE_RATE,
            feature_bins: inventory.features,
            n_fft: inventory.n_fft,
            window_length: inventory.window_length,
            hop_length: inventory.hop_length,
            normalize: inventory.normalize.clone(),
            encoder_layers: inventory.encoder_layers,
            encoder_dim: inventory.encoder_dim,
            encoder_heads: inventory.encoder_heads,
            subsampling_factor: inventory.subsampling_factor,
            subsampling_conv_channels: inventory.subsampling_conv_channels,
            ff_expansion_factor: inventory.ff_expansion_factor,
            conv_kernel_size: inventory.conv_kernel_size,
            predictor_hidden: inventory.predictor_hidden,
            predictor_layers: inventory.predictor_layers,
            joint_hidden: inventory.joint_hidden,
            prompt_dim: inventory.prompt_dim,
            prompt_dictionary_size: inventory.prompt_dictionary.len(),
            vocab_size: inventory.vocab_size,
            default_streaming_profile,
            streaming_profiles,
        })
    }

    fn diagnostics(&self) -> serde_json::Value {
        json!({
            "sample_rate": self.sample_rate,
            "feature_bins": self.feature_bins,
            "n_fft": self.n_fft,
            "window_length": self.window_length,
            "hop_length": self.hop_length,
            "normalize": self.normalize,
            "encoder_layers": self.encoder_layers,
            "encoder_dim": self.encoder_dim,
            "encoder_heads": self.encoder_heads,
            "subsampling_factor": self.subsampling_factor,
            "subsampling_conv_channels": self.subsampling_conv_channels,
            "ff_expansion_factor": self.ff_expansion_factor,
            "conv_kernel_size": self.conv_kernel_size,
            "predictor_hidden": self.predictor_hidden,
            "predictor_layers": self.predictor_layers,
            "joint_hidden": self.joint_hidden,
            "prompt_dim": self.prompt_dim,
            "prompt_dictionary_size": self.prompt_dictionary_size,
            "vocab_size": self.vocab_size,
            "streaming_profile": self.default_streaming_profile.diagnostics(),
            "streaming_profiles": self.streaming_profiles.iter().map(|profile| profile.diagnostics()).collect::<Vec<_>>(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NemotronStreamingProfile {
    pub left_context_frames: usize,
    pub right_context_frames: usize,
    pub chunk_frames: usize,
    pub chunk_ms: usize,
}

impl NemotronStreamingProfile {
    fn profiles_from_inventory(inventory: &NemotronConfigInventory) -> Result<Vec<Self>> {
        let left_context_frames = inventory.left_context_frames.unwrap_or(56);
        let mut right_context_frames = if inventory.right_context_frames.is_empty() {
            vec![0, 1, 3, 6, 13]
        } else {
            inventory.right_context_frames.clone()
        };
        right_context_frames.sort_unstable();
        right_context_frames.dedup();
        right_context_frames
            .into_iter()
            .map(|right| Self::new(left_context_frames, right))
            .collect()
    }

    pub fn new(left_context_frames: usize, right_context_frames: usize) -> Result<Self> {
        if left_context_frames != 56 {
            return Err(Error::ModelLoadError(format!(
                "Nemotron 3.5 ASR currently expects 56 left-context frames, got {left_context_frames}"
            )));
        }
        if !matches!(right_context_frames, 0 | 1 | 3 | 6 | 13) {
            return Err(Error::ModelLoadError(format!(
                "Unsupported Nemotron right-context profile {right_context_frames}; expected one of 0, 1, 3, 6, 13"
            )));
        }

        let chunk_frames = right_context_frames + 1;
        Ok(Self {
            left_context_frames,
            right_context_frames,
            chunk_frames,
            chunk_ms: chunk_frames * STREAMING_FRAME_MS,
        })
    }

    pub fn chunk_samples(&self, sample_rate: u32) -> usize {
        ms_to_samples(self.chunk_ms, sample_rate)
    }

    pub fn left_context_samples(&self, sample_rate: u32) -> usize {
        ms_to_samples(self.left_context_frames * STREAMING_FRAME_MS, sample_rate)
    }

    pub fn right_context_samples(&self, sample_rate: u32) -> usize {
        ms_to_samples(self.right_context_frames * STREAMING_FRAME_MS, sample_rate)
    }

    fn diagnostics(&self) -> serde_json::Value {
        json!({
            "att_context_size": [self.left_context_frames, self.right_context_frames],
            "chunk_frames": self.chunk_frames,
            "chunk_ms": self.chunk_ms,
            "cache_reuse_ready": true,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NemotronStreamChunkRange {
    pub chunk_index: usize,
    pub start_sample: usize,
    pub end_sample: usize,
    pub is_final: bool,
}

impl NemotronStreamChunkRange {
    pub fn len_samples(&self) -> usize {
        self.end_sample.saturating_sub(self.start_sample)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NemotronRealtimeStreamConfig {
    pub language: Option<String>,
    pub prompt: Option<String>,
    pub right_context_frames: Option<usize>,
    pub emit_partials: bool,
}

impl Default for NemotronRealtimeStreamConfig {
    fn default() -> Self {
        Self {
            language: None,
            prompt: None,
            right_context_frames: None,
            emit_partials: true,
        }
    }
}

impl NemotronRealtimeStreamConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    fn with_optional_language(mut self, language: Option<&str>) -> Self {
        self.language = language.map(ToOwned::to_owned);
        self
    }

    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    fn with_optional_prompt(mut self, prompt: Option<&str>) -> Self {
        self.prompt = prompt.map(ToOwned::to_owned);
        self
    }

    pub fn with_right_context_frames(mut self, right_context_frames: usize) -> Self {
        self.right_context_frames = Some(right_context_frames);
        self
    }

    fn with_optional_right_context_frames(mut self, right_context_frames: Option<usize>) -> Self {
        self.right_context_frames = right_context_frames;
        self
    }

    pub fn with_emit_partials(mut self, emit_partials: bool) -> Self {
        self.emit_partials = emit_partials;
        self
    }

    fn prompt_condition(&self) -> Result<NemotronPromptCondition> {
        NemotronPromptCondition::resolve(self.language.as_deref(), self.prompt.as_deref())
    }

    fn diagnostics(&self) -> serde_json::Value {
        let prompt = self.prompt_condition().ok();
        json!({
            "target_lang": prompt.as_ref().map(|prompt| prompt.target_lang.as_str()).unwrap_or("auto"),
            "context_prompt_present": prompt
                .as_ref()
                .is_some_and(|prompt| prompt.context_prompt.is_some()),
            "right_context_frames": self.right_context_frames,
            "emit_partials": self.emit_partials,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NemotronRealtimeStreamEvent {
    pub text: String,
    pub delta: String,
    pub is_final: bool,
    pub chunk_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NemotronStreamingCacheStatus {
    PhysicalStateV2,
}

#[derive(Clone)]
pub struct NemotronStreamingState {
    profile: NemotronStreamingProfile,
    prompt: NemotronPromptCondition,
    sample_rate: u32,
    max_samples: usize,
    max_text_bytes: usize,
    emit_partials: bool,
    resampler: NemotronStreamingResampler,
    samples: Vec<f32>,
    buffered_samples: usize,
    consumed_samples: usize,
    chunks_processed: usize,
    events_emitted: usize,
    input_finished: bool,
    final_event_emitted: bool,
    cache_status: NemotronStreamingCacheStatus,
    feature_state: NemotronStreamingFeatureState,
    pre_encode_state: NemotronStreamingPreEncodeState,
    encoder_state: NemotronStreamingEncoderState,
    rnnt_state: Option<NemotronRnntStreamState>,
    assembled_text: String,
    emitted_tokens: usize,
}

pub(crate) enum NemotronRealtimeBatchInput<'a> {
    Push {
        samples: &'a [f32],
        sample_rate: u32,
    },
    Finish,
}

pub(crate) struct NemotronRealtimeBatchRow<'a> {
    pub(crate) state: &'a mut NemotronStreamingState,
    pub(crate) input: NemotronRealtimeBatchInput<'a>,
}

#[derive(Clone, Debug, Default)]
struct NemotronStreamingResampler {
    source_rate: Option<u32>,
    source_samples: usize,
    projected_output_samples: usize,
    emitted_output_samples: usize,
    retained_source_samples: [Option<(usize, f32)>; 2],
    last_source_sample: Option<f32>,
}

impl NemotronStreamingResampler {
    fn resample_chunk(
        &self,
        samples: &[f32],
        source_rate: u32,
        target_rate: u32,
        max_output_samples: usize,
    ) -> Result<(Self, Vec<f32>)> {
        if source_rate == 0 || target_rate == 0 {
            return Err(Error::InvalidInput(
                "Audio sample rate must be greater than zero".to_string(),
            ));
        }
        if let Some(expected_rate) = self.source_rate {
            if expected_rate != source_rate {
                return Err(Error::InvalidInput(format!(
                    "Nemotron realtime stream sample rate changed from {expected_rate} Hz to {source_rate} Hz"
                )));
            }
        }

        let source_samples = self
            .source_samples
            .checked_add(samples.len())
            .ok_or_else(|| {
                Error::InvalidInput("Nemotron realtime source sample count overflowed".to_string())
            })?;
        let projected_output_samples =
            cumulative_resampled_output_len(source_samples, source_rate, target_rate)?;
        if projected_output_samples > max_output_samples {
            return Err(realtime_sample_limit_error(max_output_samples, target_rate));
        }

        let mut next = self.clone();
        next.source_rate = Some(source_rate);
        next.source_samples = source_samples;
        next.projected_output_samples = projected_output_samples;
        let output_capacity = projected_output_samples
            .checked_sub(next.emitted_output_samples)
            .ok_or_else(realtime_resampler_invariant)?;
        let mut output = Vec::with_capacity(output_capacity);
        let chunk_start = self.source_samples;

        while next.emitted_output_samples < projected_output_samples {
            let output_index = next.emitted_output_samples;
            let source_position = (output_index as u128)
                .checked_mul(u128::from(source_rate))
                .ok_or_else(realtime_resampler_invariant)?;
            let left_index = usize::try_from(source_position / u128::from(target_rate))
                .map_err(|_| realtime_resampler_invariant())?;
            let remainder = source_position % u128::from(target_rate);
            let needs_right_sample = remainder != 0;
            if left_index >= source_samples
                || (needs_right_sample && left_index.saturating_add(1) >= source_samples)
            {
                break;
            }

            let left = stream_source_sample(self, samples, chunk_start, left_index)?;
            let value = if needs_right_sample {
                let right = stream_source_sample(self, samples, chunk_start, left_index + 1)?;
                let fraction = (remainder as f64 / target_rate as f64) as f32;
                left * (1.0 - fraction) + right * fraction
            } else {
                left
            };
            output.push(value);
            next.emitted_output_samples = next.emitted_output_samples.saturating_add(1);
        }

        if let Some(last) = samples.last() {
            next.last_source_sample = Some(*last);
        }
        let next_source_position = (next.emitted_output_samples as u128)
            .checked_mul(u128::from(source_rate))
            .ok_or_else(realtime_resampler_invariant)?;
        let next_left_index = usize::try_from(next_source_position / u128::from(target_rate))
            .map_err(|_| realtime_resampler_invariant())?;
        next.retained_source_samples = [None, None];
        for (slot, sample_index) in [next_left_index, next_left_index.saturating_add(1)]
            .into_iter()
            .enumerate()
        {
            if sample_index < source_samples {
                next.retained_source_samples[slot] = Some((
                    sample_index,
                    stream_source_sample(self, samples, chunk_start, sample_index)?,
                ));
            }
        }
        Ok((next, output))
    }

    fn finish(&self) -> Result<(Self, Vec<f32>)> {
        let remaining = self
            .projected_output_samples
            .checked_sub(self.emitted_output_samples)
            .ok_or_else(realtime_resampler_invariant)?;
        if remaining == 0 {
            return Ok((self.clone(), Vec::new()));
        }
        let last_sample = self
            .last_source_sample
            .ok_or_else(realtime_resampler_invariant)?;
        let mut next = self.clone();
        next.emitted_output_samples = next.projected_output_samples;
        Ok((next, vec![last_sample; remaining]))
    }
}

fn cumulative_resampled_output_len(
    input_samples: usize,
    source_rate: u32,
    target_rate: u32,
) -> Result<usize> {
    if source_rate == 0 || target_rate == 0 {
        return Err(Error::InvalidInput(
            "Audio sample rate must be greater than zero".to_string(),
        ));
    }
    if input_samples == 0 {
        return Ok(0);
    }
    let numerator = (input_samples as u128)
        .checked_mul(u128::from(target_rate))
        .and_then(|value| value.checked_add(u128::from(source_rate / 2)))
        .ok_or_else(realtime_resampler_invariant)?;
    usize::try_from((numerator / u128::from(source_rate)).max(1))
        .map_err(|_| realtime_resampler_invariant())
}

fn stream_source_sample(
    state: &NemotronStreamingResampler,
    samples: &[f32],
    chunk_start: usize,
    sample_index: usize,
) -> Result<f32> {
    if sample_index < chunk_start {
        for (retained_index, retained_sample) in state.retained_source_samples.into_iter().flatten()
        {
            if retained_index == sample_index {
                return Ok(retained_sample);
            }
        }
        return Err(Error::InferenceError(format!(
            "Nemotron realtime resampler requested unretained source sample {sample_index} before chunk {chunk_start}"
        )));
    }
    samples
        .get(sample_index - chunk_start)
        .copied()
        .ok_or_else(|| {
            Error::InferenceError(format!(
                "Nemotron realtime resampler requested source sample {sample_index} outside chunk {chunk_start}..{}",
                chunk_start.saturating_add(samples.len())
            ))
        })
}

fn realtime_resampler_invariant() -> Error {
    Error::InferenceError("Nemotron realtime resampler state is inconsistent".to_string())
}

fn realtime_sample_limit_error(max_samples: usize, sample_rate: u32) -> Error {
    Error::InvalidInput(format!(
        "Nemotron realtime stream exceeds the configured limit of {} samples ({:.3} seconds at {} Hz)",
        max_samples,
        max_samples as f64 / sample_rate as f64,
        sample_rate
    ))
}

struct NemotronRealtimeEventBatch {
    previous_text: String,
    changed: bool,
    is_final: bool,
}

impl NemotronRealtimeEventBatch {
    fn new(state: &NemotronStreamingState) -> Self {
        Self {
            previous_text: state.assembled_text.clone(),
            changed: false,
            is_final: false,
        }
    }

    fn record_decoded_text(
        &mut self,
        state: &mut NemotronStreamingState,
        text: String,
        is_final: bool,
    ) -> Result<()> {
        if text.len() > state.max_text_bytes {
            return Err(Error::InferenceError(format!(
                "Nemotron realtime decoder output exceeded its model-derived limit of {} bytes",
                state.max_text_bytes
            )));
        }
        self.changed |= state.assembled_text != text;
        state.assembled_text = text;
        self.is_final |= is_final;
        Ok(())
    }

    fn mark_final(&mut self) {
        self.is_final = true;
    }

    #[cfg(test)]
    fn pending_event_count(&self) -> usize {
        usize::from(self.changed || self.is_final)
    }

    #[cfg(test)]
    fn retained_text_bytes(&self) -> usize {
        self.previous_text.capacity()
    }

    fn into_events(self, state: &mut NemotronStreamingState) -> Vec<NemotronRealtimeStreamEvent> {
        if (self.is_final && state.final_event_emitted)
            || (!self.is_final && (!state.emit_partials || !self.changed))
        {
            return Vec::new();
        }

        let delta = if self.is_final && !state.emit_partials && state.events_emitted == 0 {
            state.assembled_text.clone()
        } else {
            text_delta(&self.previous_text, &state.assembled_text)
        };
        let event = NemotronRealtimeStreamEvent {
            text: state.assembled_text.clone(),
            delta,
            is_final: self.is_final,
            chunk_index: state.events_emitted,
        };
        state.final_event_emitted |= self.is_final;
        state.events_emitted = state.events_emitted.saturating_add(1);
        vec![event]
    }
}

impl NemotronStreamingState {
    fn new(
        profile: NemotronStreamingProfile,
        prompt: NemotronPromptCondition,
        sample_rate: u32,
        max_samples: usize,
        max_text_bytes: usize,
        emit_partials: bool,
    ) -> Self {
        let encoder_state = NemotronStreamingEncoderState::new(
            profile.left_context_frames,
            profile.right_context_frames,
        );
        Self {
            profile,
            prompt,
            sample_rate,
            max_samples,
            max_text_bytes,
            emit_partials,
            resampler: NemotronStreamingResampler::default(),
            samples: Vec::new(),
            buffered_samples: 0,
            consumed_samples: 0,
            chunks_processed: 0,
            events_emitted: 0,
            input_finished: false,
            final_event_emitted: false,
            cache_status: NemotronStreamingCacheStatus::PhysicalStateV2,
            feature_state: NemotronStreamingFeatureState::new(),
            pre_encode_state: NemotronStreamingPreEncodeState::new(),
            encoder_state,
            rnnt_state: None,
            assembled_text: String::new(),
            emitted_tokens: 0,
        }
    }

    fn attach_rnnt_state(&mut self, rnnt_state: NemotronRnntStreamState) {
        self.rnnt_state = Some(rnnt_state);
    }

    pub fn profile(&self) -> &NemotronStreamingProfile {
        &self.profile
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn buffered_samples(&self) -> usize {
        self.buffered_samples
    }

    pub fn max_samples(&self) -> usize {
        self.max_samples
    }

    pub fn consumed_samples(&self) -> usize {
        self.consumed_samples
    }

    pub fn chunks_processed(&self) -> usize {
        self.chunks_processed
    }

    pub fn text(&self) -> &str {
        &self.assembled_text
    }

    pub fn emitted_tokens(&self) -> usize {
        self.emitted_tokens
    }

    /// Complete retained backing allocation for this exact stream session.
    pub fn session_cache_bytes(&self) -> Option<u64> {
        let usage = self.session_resource_usage()?;
        usage.host_bytes.checked_add(usage.tensor_bytes)
    }

    pub fn session_resource_usage(&self) -> Option<NemotronRealtimeResourceUsage> {
        let mut host = TensorStorageAccounting::default();
        host.add_bytes(retained_capacity_bytes::<f32>(self.samples.capacity())?)?;
        host.add_bytes(retained_capacity_bytes::<u8>(
            self.assembled_text.capacity(),
        )?)?;
        host.add_bytes(retained_capacity_bytes::<u8>(
            self.prompt.target_lang.capacity(),
        )?)?;
        if let Some(prompt) = &self.prompt.context_prompt {
            host.add_bytes(retained_capacity_bytes::<u8>(prompt.capacity())?)?;
        }
        self.feature_state.account_host_storage(&mut host)?;
        let mut tensors = TensorStorageAccounting::default();
        self.pre_encode_state.account_tensor_storage(&mut tensors)?;
        self.encoder_state.account_tensor_storage(&mut tensors)?;
        if let Some(rnnt_state) = &self.rnnt_state {
            rnnt_state.account_host_storage(&mut host)?;
            rnnt_state.account_tensor_storage(&mut tensors)?;
        }
        Some(NemotronRealtimeResourceUsage {
            host_bytes: host.bytes(),
            tensor_bytes: tensors.bytes(),
        })
    }

    pub fn push_samples(&mut self, samples: &[f32]) -> Result<()> {
        if self.input_finished {
            return Err(Error::InvalidInput(
                "Cannot push audio into a finalized Nemotron streaming state".to_string(),
            ));
        }
        let next_samples = self.ensure_can_accept_samples(samples.len())?;
        self.samples.extend_from_slice(samples);
        self.buffered_samples = next_samples;
        Ok(())
    }

    fn ensure_can_accept_samples(&self, additional_samples: usize) -> Result<usize> {
        let next_samples = self
            .buffered_samples
            .checked_add(additional_samples)
            .ok_or_else(|| {
                Error::InvalidInput("Nemotron realtime stream sample count overflowed".to_string())
            })?;
        if next_samples > self.max_samples {
            return Err(realtime_sample_limit_error(
                self.max_samples,
                self.sample_rate,
            ));
        }
        Ok(next_samples)
    }

    pub fn finish_input(&mut self) {
        self.input_finished = true;
    }

    pub fn next_ready_chunk(&self) -> Option<NemotronStreamChunkRange> {
        if self.consumed_samples >= self.buffered_samples {
            return None;
        }

        let chunk_samples = self.profile.chunk_samples(self.sample_rate);
        let planned_end = self.consumed_samples.saturating_add(chunk_samples);
        if planned_end <= self.buffered_samples {
            return Some(NemotronStreamChunkRange {
                chunk_index: self.chunks_processed,
                start_sample: self.consumed_samples,
                end_sample: planned_end,
                is_final: self.input_finished && planned_end == self.buffered_samples,
            });
        }

        self.input_finished.then_some(NemotronStreamChunkRange {
            chunk_index: self.chunks_processed,
            start_sample: self.consumed_samples,
            end_sample: self.buffered_samples,
            is_final: true,
        })
    }

    pub fn mark_chunk_consumed(&mut self, chunk: &NemotronStreamChunkRange) -> Result<()> {
        if chunk.chunk_index != self.chunks_processed {
            return Err(Error::InvalidInput(format!(
                "Nemotron stream chunk index mismatch: got {}, expected {}",
                chunk.chunk_index, self.chunks_processed
            )));
        }
        if chunk.start_sample != self.consumed_samples
            || chunk.end_sample <= chunk.start_sample
            || chunk.end_sample > self.buffered_samples
        {
            return Err(Error::InvalidInput(format!(
                "Invalid Nemotron stream chunk range {}..{} for consumed={} buffered={}",
                chunk.start_sample, chunk.end_sample, self.consumed_samples, self.buffered_samples
            )));
        }

        self.consumed_samples = chunk.end_sample;
        self.chunks_processed = self.chunks_processed.saturating_add(1);
        Ok(())
    }

    pub fn diagnostics(&self) -> serde_json::Value {
        json!({
            "profile": self.profile.diagnostics(),
            "prompt": self.prompt.diagnostics(),
            "sample_rate": self.sample_rate,
            "max_samples": self.max_samples,
            "max_duration_seconds": self.max_samples as f64 / self.sample_rate as f64,
            "emit_partials": self.emit_partials,
            "source_sample_rate": self.resampler.source_rate,
            "source_samples": self.resampler.source_samples,
            "projected_output_samples": self.resampler.projected_output_samples,
            "emitted_output_samples": self.resampler.emitted_output_samples,
            "buffered_samples": self.buffered_samples,
            "consumed_samples": self.consumed_samples,
            "chunks_processed": self.chunks_processed,
            "events_emitted": self.events_emitted,
            "input_finished": self.input_finished,
            "final_event_emitted": self.final_event_emitted,
            "emitted_tokens": self.emitted_tokens,
            "cache_status": format!("{:?}", self.cache_status),
            "supports_realtime_cache_decode": true,
            "supports_realtime_stream_decode": self.rnnt_state.is_some(),
        })
    }
}

fn retained_capacity_bytes<T>(capacity: usize) -> Option<u64> {
    u64::try_from(capacity.checked_mul(std::mem::size_of::<T>())?).ok()
}

fn realtime_max_samples_for_seconds(sample_rate: u32, seconds: usize) -> Result<usize> {
    if sample_rate == 0 || seconds == 0 {
        return Err(Error::ConfigError(
            "Nemotron realtime stream duration and sample rate must be greater than zero"
                .to_string(),
        ));
    }
    (sample_rate as usize).checked_mul(seconds).ok_or_else(|| {
        Error::ConfigError("Nemotron realtime stream sample limit overflowed".to_string())
    })
}

fn configured_realtime_max_samples(sample_rate: u32) -> Result<usize> {
    let seconds = match std::env::var(NEMOTRON_REALTIME_MAX_SECONDS_ENV) {
        Ok(raw) => raw.trim().parse::<usize>().map_err(|_| {
            Error::ConfigError(format!(
                "{NEMOTRON_REALTIME_MAX_SECONDS_ENV} must be a positive whole number of seconds"
            ))
        })?,
        Err(std::env::VarError::NotPresent) => DEFAULT_NEMOTRON_REALTIME_MAX_SECONDS,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(Error::ConfigError(format!(
                "{NEMOTRON_REALTIME_MAX_SECONDS_ENV} must contain valid UTF-8"
            )));
        }
    };
    realtime_max_samples_for_seconds(sample_rate, seconds)
}

fn state_dtype(dtype: DType) -> Result<StateDType> {
    match dtype {
        DType::F32 => Ok(StateDType::F32),
        DType::F16 => Ok(StateDType::F16),
        DType::BF16 => Ok(StateDType::Bf16),
        other => Err(Error::ModelLoadError(format!(
            "Nemotron realtime physical state does not support {other:?}"
        ))),
    }
}

fn state_header(id: StateDomainId) -> StateDomainHeader {
    StateDomainHeader {
        id,
        scope: StateScope::Retained,
        clock: StateClock::Custom("realtime_operation_revision".into()),
        placement: PlacementPolicy::BackendLocal,
        prefix: PrefixPolicy::Disabled,
        checkpoint: CheckpointPolicy::Transactional,
    }
}

fn fixed_component(
    id: u32,
    role: TensorRole,
    axis: ShapeAxis,
    elements: usize,
    dtype: StateDType,
) -> Result<TensorComponentSpec> {
    Ok(TensorComponentSpec {
        id: StateComponentId::new(id),
        role,
        shape: BoundedShape {
            dimensions: vec![ShapeDimension {
                axis,
                extent: ShapeExtent::Fixed {
                    value: u64::try_from(elements).map_err(|_| realtime_reservation_overflow())?,
                },
            }],
        },
        accepted_dtypes: vec![dtype],
    })
}

fn nemotron_realtime_state_contract(
    max_samples: usize,
    shape: NemotronRealtimeStateShape,
    left_context_frames: usize,
    dtype: StateDType,
) -> Result<InferenceStateContract> {
    if max_samples == 0
        || shape.hop_length == 0
        || shape.subsampling_factor == 0
        || shape.encoder_layers == 0
        || left_context_frames == 0
        || shape.conv_kernel_size <= 1
    {
        return Err(Error::ModelLoadError(
            "Nemotron realtime physical state contains a zero capacity".into(),
        ));
    }
    let feature_frames = max_samples.div_ceil(shape.hop_length).max(1);
    let encoded_frames = feature_frames.div_ceil(shape.subsampling_factor).max(1);
    let feature = fixed_component(
        1,
        TensorRole::AudioHistory,
        ShapeAxis::Channels,
        shape.feature_bins,
        dtype,
    )?;
    let encoder = fixed_component(
        1,
        TensorRole::EncoderMemory,
        ShapeAxis::Hidden,
        shape.encoder_dim,
        dtype,
    )?;
    let attention = (0..shape.encoder_layers)
        .map(|layer| {
            fixed_component(
                u32::try_from(layer + 1).map_err(|_| realtime_reservation_overflow())?,
                TensorRole::EncoderMemory,
                ShapeAxis::Hidden,
                shape.encoder_dim,
                dtype,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let convolution = (0..shape.encoder_layers)
        .map(|layer| {
            fixed_component(
                u32::try_from(layer + 1).map_err(|_| realtime_reservation_overflow())?,
                TensorRole::ConvolutionState,
                ShapeAxis::Hidden,
                shape.encoder_dim,
                dtype,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let rnnt = vec![
        fixed_component(
            1,
            TensorRole::RecurrentHidden,
            ShapeAxis::Hidden,
            shape.predictor_hidden,
            dtype,
        )?,
        fixed_component(
            2,
            TensorRole::RecurrentCell,
            ShapeAxis::Hidden,
            shape.predictor_hidden,
            dtype,
        )?,
        fixed_component(
            3,
            TensorRole::RecurrentHidden,
            ShapeAxis::Hidden,
            shape.predictor_hidden,
            dtype,
        )?,
        fixed_component(
            4,
            TensorRole::RecurrentCell,
            ShapeAxis::Hidden,
            shape.predictor_hidden,
            dtype,
        )?,
        fixed_component(
            5,
            TensorRole::RetainedEmbedding,
            ShapeAxis::Hidden,
            shape.predictor_hidden,
            dtype,
        )?,
        fixed_component(
            6,
            TensorRole::Custom("rnnt_predictor_projection".into()),
            ShapeAxis::Hidden,
            shape.joint_hidden,
            dtype,
        )?,
    ];
    let domains = vec![
        StateDomainSpec::Append(AppendStateDomainSpec {
            header: state_header(NEMOTRON_FEATURE_HISTORY_DOMAIN),
            components_per_step: vec![feature],
            max_steps: u64::try_from(feature_frames)
                .map_err(|_| realtime_reservation_overflow())?,
        }),
        StateDomainSpec::Append(AppendStateDomainSpec {
            header: state_header(NEMOTRON_PENDING_ENCODER_DOMAIN),
            components_per_step: vec![encoder],
            max_steps: u64::try_from(encoded_frames)
                .map_err(|_| realtime_reservation_overflow())?,
        }),
        StateDomainSpec::Ring(RingStateDomainSpec {
            header: state_header(NEMOTRON_ATTENTION_HISTORY_DOMAIN),
            components_per_step: attention,
            capacity_steps: u64::try_from(left_context_frames)
                .map_err(|_| realtime_reservation_overflow())?,
        }),
        StateDomainSpec::Ring(RingStateDomainSpec {
            header: state_header(NEMOTRON_CONVOLUTION_HISTORY_DOMAIN),
            components_per_step: convolution,
            capacity_steps: u64::try_from(shape.conv_kernel_size - 1)
                .map_err(|_| realtime_reservation_overflow())?,
        }),
        StateDomainSpec::Tensor(TensorStateDomainSpec {
            header: state_header(NEMOTRON_RNNT_STATE_DOMAIN),
            components: rnnt,
        }),
    ];
    let contract = InferenceStateContract {
        abi: CURRENT_INFERENCE_STATE_ABI,
        groups: vec![
            StateGroupSpec {
                id: NEMOTRON_ENCODER_STATE_GROUP,
                domains: vec![
                    NEMOTRON_FEATURE_HISTORY_DOMAIN,
                    NEMOTRON_PENDING_ENCODER_DOMAIN,
                    NEMOTRON_ATTENTION_HISTORY_DOMAIN,
                    NEMOTRON_CONVOLUTION_HISTORY_DOMAIN,
                ],
                prefix_shareable: false,
            },
            StateGroupSpec {
                id: NEMOTRON_RNNT_STATE_GROUP,
                domains: vec![NEMOTRON_RNNT_STATE_DOMAIN],
                prefix_shareable: false,
            },
        ],
        domains,
    };
    contract.validate()?;
    Ok(contract)
}

fn physical_components(
    runtime: &RetainedTensorStateRuntimeV2,
    transaction: PhysicalStateTransactionId,
    domain: StateDomainId,
    expected_components: usize,
) -> Result<Vec<Option<Tensor>>> {
    let snapshot = runtime
        .read_transaction_base(transaction, domain)?
        .ok_or_else(|| {
            Error::InferenceError(format!(
                "Nemotron physical domain {} has no committed snapshot",
                domain.get()
            ))
        })?;
    if snapshot.components.len() != expected_components {
        return Err(Error::InferenceError(format!(
            "Nemotron physical domain {} expected {} components, found {}",
            domain.get(),
            expected_components,
            snapshot.components.len()
        )));
    }
    snapshot
        .components
        .iter()
        .enumerate()
        .map(|(index, value)| {
            if value.component != StateComponentId::new((index + 1) as u32) {
                return Err(Error::InferenceError(format!(
                    "Nemotron physical domain {} has non-canonical components",
                    domain.get()
                )));
            }
            Ok(value.tensor.clone())
        })
        .collect()
}

fn stage_physical_components(
    runtime: &RetainedTensorStateRuntimeV2,
    transaction: PhysicalStateTransactionId,
    domain: StateDomainId,
    target_cursor: u64,
    tensors: Vec<Option<Tensor>>,
) -> Result<()> {
    let expected_cursor = runtime
        .read_transaction_base(transaction, domain)?
        .map_or(0, |snapshot| snapshot.cursor);
    let values = tensors
        .into_iter()
        .enumerate()
        .map(|(index, tensor)| {
            Ok(StateComponentValue {
                component: StateComponentId::new(
                    u32::try_from(index + 1).map_err(|_| realtime_reservation_overflow())?,
                ),
                tensor,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    runtime.stage_replace(transaction, domain, expected_cursor, target_cursor, values)
}

fn estimate_realtime_resource_reservation(
    max_samples: usize,
    max_decoder_token_bytes: usize,
    profile: &NemotronStreamingProfile,
    shape: NemotronRealtimeStateShape,
    prompt: &NemotronPromptCondition,
) -> Result<NemotronRealtimeResourceReservation> {
    if max_samples == 0 || shape.hop_length == 0 || shape.subsampling_factor == 0 {
        return Err(Error::ConfigError(
            "Nemotron realtime resource shape contains a zero limit".to_string(),
        ));
    }

    let feature_frames = max_samples.div_ceil(shape.hop_length).max(1);
    let encoded_frames = feature_frames.div_ceil(shape.subsampling_factor).max(1);
    let max_output_tokens = encoded_frames
        .checked_mul(shape.max_symbols_per_frame)
        .ok_or_else(realtime_reservation_overflow)?;
    let decoded_token_bytes = max_decoder_token_bytes
        .max(1)
        .checked_mul(CONSERVATIVE_DECODED_TOKEN_EXPANSION)
        .ok_or_else(realtime_reservation_overflow)?
        .max(16);
    let max_text_bytes = max_output_tokens
        .checked_mul(decoded_token_bytes)
        .ok_or_else(realtime_reservation_overflow)?;

    let audio_capacity = conservative_vec_capacity(max_samples, 4)?;
    let token_capacity = conservative_vec_capacity(max_output_tokens, 4)?;
    let text_capacity = conservative_vec_capacity(max_text_bytes, 8)?;
    let target_lang_capacity = conservative_vec_capacity(prompt.target_lang.len(), 8)?;
    let context_prompt_capacity = conservative_vec_capacity(
        prompt
            .context_prompt
            .as_deref()
            .map(str::len)
            .unwrap_or_default(),
        8,
    )?;

    let audio_capacity_bytes = checked_element_bytes(audio_capacity, std::mem::size_of::<f32>())?;
    let mut host_bytes = REALTIME_HOST_FIXED_OVERHEAD_BYTES;
    // One conservative capacity covers retained target-rate stream audio. The
    // second covers the ordered owned input packet plus transient resampling
    // output, so moving packet work onto a blocking worker cannot exceed the
    // immutable job authorization.
    host_bytes = checked_add_bytes(
        host_bytes,
        audio_capacity_bytes
            .checked_mul(2)
            .ok_or_else(realtime_reservation_overflow)?,
    )?;
    host_bytes = checked_add_bytes(
        host_bytes,
        checked_element_bytes(token_capacity, std::mem::size_of::<usize>())?,
    )?;
    host_bytes = checked_add_bytes(
        host_bytes,
        checked_element_bytes(token_capacity, std::mem::size_of::<u32>())?,
    )?;
    host_bytes = checked_add_bytes(
        host_bytes,
        u64::try_from(text_capacity)
            .map_err(|_| realtime_reservation_overflow())?
            .checked_mul(REALTIME_PEAK_TEXT_COPIES)
            .ok_or_else(realtime_reservation_overflow)?,
    )?;
    host_bytes = checked_add_bytes(
        host_bytes,
        u64::try_from(target_lang_capacity).map_err(|_| realtime_reservation_overflow())?,
    )?;
    host_bytes = checked_add_bytes(
        host_bytes,
        u64::try_from(context_prompt_capacity).map_err(|_| realtime_reservation_overflow())?,
    )?;

    let feature_elements = checked_product(&[feature_frames, shape.feature_bins])?;
    let pending_encoder_elements = checked_product(&[encoded_frames, shape.encoder_dim])?;
    let layer_cache_frames = profile
        .left_context_frames
        .checked_add(shape.conv_kernel_size.saturating_sub(1))
        .ok_or_else(realtime_reservation_overflow)?;
    let layer_cache_elements =
        checked_product(&[shape.encoder_layers, layer_cache_frames, shape.encoder_dim])?;
    let predictor_state_tensors = shape
        .predictor_layers
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(realtime_reservation_overflow)?;
    let predictor_elements = checked_product(&[predictor_state_tensors, shape.predictor_hidden])?
        .checked_add(shape.joint_hidden)
        .ok_or_else(realtime_reservation_overflow)?;
    let tensor_elements = feature_elements
        .checked_add(pending_encoder_elements)
        .and_then(|value| value.checked_add(layer_cache_elements))
        .and_then(|value| value.checked_add(predictor_elements))
        .ok_or_else(realtime_reservation_overflow)?;
    let tensor_bytes = checked_element_bytes(tensor_elements, std::mem::size_of::<f32>())?;

    Ok(NemotronRealtimeResourceReservation {
        max_samples,
        host_bytes,
        tensor_bytes,
        max_output_tokens,
        max_text_bytes,
    })
}

fn conservative_vec_capacity(length: usize, minimum_nonzero: usize) -> Result<usize> {
    if length == 0 {
        return Ok(0);
    }
    length
        .checked_mul(2)
        .map(|capacity| capacity.max(minimum_nonzero))
        .ok_or_else(realtime_reservation_overflow)
}

fn checked_product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(realtime_reservation_overflow)
    })
}

fn checked_element_bytes(elements: usize, element_size: usize) -> Result<u64> {
    let bytes = elements
        .checked_mul(element_size)
        .ok_or_else(realtime_reservation_overflow)?;
    u64::try_from(bytes).map_err(|_| realtime_reservation_overflow())
}

fn checked_add_bytes(left: u64, right: u64) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(realtime_reservation_overflow)
}

fn realtime_reservation_overflow() -> Error {
    Error::ConfigError("Nemotron realtime resource reservation overflowed".to_string())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NemotronPromptCondition {
    target_lang: String,
    strip_lang_tags: bool,
    context_prompt: Option<String>,
}

impl NemotronPromptCondition {
    fn resolve(language: Option<&str>, prompt: Option<&str>) -> Result<Self> {
        let language_hint = language.and_then(non_empty_trimmed).or_else(|| {
            prompt
                .and_then(non_empty_trimmed)
                .filter(|value| looks_like_lang(value))
        });
        let target_lang = normalize_target_lang(language_hint.unwrap_or("auto"))?;
        let context_prompt = prompt
            .and_then(non_empty_trimmed)
            .filter(|value| !looks_like_lang(value))
            .map(ToOwned::to_owned);

        Ok(Self {
            target_lang,
            strip_lang_tags: DEFAULT_STRIP_LANG_TAGS,
            context_prompt,
        })
    }

    fn diagnostics(&self) -> serde_json::Value {
        json!({
            "target_lang": self.target_lang,
            "strip_lang_tags": self.strip_lang_tags,
            "context_prompt_present": self.context_prompt.is_some(),
        })
    }
}

#[derive(Debug, Clone)]
struct NemotronDecodeRequest {
    samples: usize,
    input_sample_rate: u32,
    target_sample_rate: u32,
    prompt: NemotronPromptCondition,
}

#[derive(Debug, Clone, Default)]
struct NemotronStageTimings {
    resample: Duration,
    encode: Duration,
    encode_profile: Option<NemotronEncodeProfile>,
    rnnt_decode: Duration,
    text_assembly: Duration,
}

impl NemotronStageTimings {
    fn diagnostics(&self) -> serde_json::Value {
        let profile = self.encode_profile.as_ref();
        let profile_timing =
            |f: fn(&network::NemotronEncodeProfile) -> Duration| profile.map(f).map(duration_ms);
        let feature_extract = profile_timing(|profile| profile.timings.feature_extract);
        let feature_upload = profile_timing(|profile| profile.timings.feature_upload);
        let dtype_cast = profile_timing(|profile| profile.timings.dtype_cast);
        let subsample = profile_timing(|profile| profile.timings.subsample);
        let pos_emb = profile_timing(|profile| profile.timings.pos_emb);
        let att_mask = profile_timing(|profile| profile.timings.att_mask);
        let encoder_ffn = profile_timing(|profile| profile.timings.encoder_ffn);
        let encoder_attention = profile_timing(|profile| profile.timings.encoder_attention);
        let encoder_conv = profile_timing(|profile| profile.timings.encoder_conv);
        let encoder_norm = profile_timing(|profile| profile.timings.encoder_norm);
        let prompt_kernel = profile_timing(|profile| profile.timings.prompt_kernel);
        let encoder_forward = profile_timing(|profile| profile.timings.encoder_layers_total());
        let mel_prepare = profile.map(|profile| {
            duration_ms(profile.timings.feature_extract + profile.timings.feature_upload)
        });
        let model_total = self.resample + self.encode + self.rnnt_decode + self.text_assembly;

        json!({
            "resample": duration_ms(self.resample),
            "feature_extract": feature_extract,
            "feature_upload": feature_upload,
            "mel_prepare": mel_prepare,
            "mel": feature_extract,
            "dtype_cast": dtype_cast,
            "subsample": subsample,
            "pos_emb": pos_emb,
            "att_mask": att_mask,
            "encoder_forward": encoder_forward,
            "encoder_ffn": encoder_ffn,
            "encoder_attention": encoder_attention,
            "encoder_conv": encoder_conv,
            "encoder_norm": encoder_norm,
            "prompt_kernel": prompt_kernel,
            "audio_encode": duration_ms(self.encode),
            "prefill": duration_ms(model_total),
            "decode": duration_ms(self.rnnt_decode),
            "model_total": duration_ms(model_total),
            "resample_ms": duration_ms(self.resample),
            "encode_ms": duration_ms(self.encode),
            "rnnt_decode_ms": duration_ms(self.rnnt_decode),
            "text_assembly_ms": duration_ms(self.text_assembly),
        })
    }
}

impl NemotronDecodeRequest {
    fn diagnostics(&self) -> serde_json::Value {
        json!({
            "samples": self.samples,
            "input_sample_rate": self.input_sample_rate,
            "target_sample_rate": self.target_sample_rate,
            "prompt": self.prompt.diagnostics(),
        })
    }
}

fn nemotron_offline_source_identity(
    audio: &[f32],
    sample_rate: u32,
    request: &NemotronDecodeRequest,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"izwi.nemotron.offline-acoustic.v2\0");
    hasher.update(sample_rate.to_le_bytes());
    hasher.update((audio.len() as u64).to_le_bytes());
    hasher.update(request.prompt.target_lang.as_bytes());
    if let Some(prompt) = request.prompt.context_prompt.as_deref() {
        hasher.update(prompt.as_bytes());
    }
    for sample in audio {
        hasher.update(sample.to_bits().to_le_bytes());
    }
    let mut identity: [u8; 32] = hasher.finalize().into();
    if identity.iter().all(|byte| *byte == 0) {
        identity[0] = 1;
    }
    identity
}

impl NemotronAsrModel {
    pub(crate) fn offline_physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<NemotronOfflinePhysicalStateSpec> {
        physical::nemotron_offline_physical_state_spec(
            self.network.realtime_state_shape(),
            state_dtype(self.network.dtype())?,
            stage_graphs,
        )
    }

    pub(crate) fn realtime_physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<NemotronRealtimePhysicalStateSpec> {
        let retained = nemotron_realtime_state_contract(
            configured_realtime_max_samples(self.runtime_plan.sample_rate)?,
            self.network.realtime_state_shape(),
            self.runtime_plan
                .streaming_profiles
                .iter()
                .map(|profile| profile.left_context_frames)
                .max()
                .unwrap_or(56),
            state_dtype(self.network.dtype())?,
        )?;
        let descriptor =
            CapabilityStateDescriptorV2::managed_for_stage_graphs(retained.clone(), stage_graphs)?;
        Ok(NemotronRealtimePhysicalStateSpec {
            retained,
            descriptor,
        })
    }

    pub(crate) fn hydrate_realtime_physical_state(
        &self,
        state: &mut NemotronStreamingState,
        runtime: &RetainedTensorStateRuntimeV2,
        transaction: PhysicalStateTransactionId,
    ) -> Result<()> {
        let feature =
            physical_components(runtime, transaction, NEMOTRON_FEATURE_HISTORY_DOMAIN, 1)?;
        let pending =
            physical_components(runtime, transaction, NEMOTRON_PENDING_ENCODER_DOMAIN, 1)?;
        let attention = physical_components(
            runtime,
            transaction,
            NEMOTRON_ATTENTION_HISTORY_DOMAIN,
            self.network.realtime_state_shape().encoder_layers,
        )?;
        let convolution = physical_components(
            runtime,
            transaction,
            NEMOTRON_CONVOLUTION_HISTORY_DOMAIN,
            self.network.realtime_state_shape().encoder_layers,
        )?;
        let rnnt = physical_components(runtime, transaction, NEMOTRON_RNNT_STATE_DOMAIN, 6)?;

        state
            .pre_encode_state
            .install_retained_tensor(feature.into_iter().next().flatten());
        state.encoder_state.install_retained_tensors(
            pending.into_iter().next().flatten(),
            attention,
            convolution,
        )?;
        let rnnt_state = state.rnnt_state.as_mut().ok_or_else(|| {
            Error::InferenceError("Nemotron realtime state has no RNNT control state".into())
        })?;
        rnnt_state.install_retained_tensors(rnnt.try_into().map_err(|_| {
            Error::InferenceError("Nemotron physical RNNT state is incomplete".into())
        })?);
        Ok(())
    }

    pub(crate) fn stage_realtime_physical_state(
        &self,
        state: &mut NemotronStreamingState,
        runtime: &RetainedTensorStateRuntimeV2,
        transaction: PhysicalStateTransactionId,
        target_cursor: u64,
    ) -> Result<()> {
        let feature = vec![state.pre_encode_state.retained_tensor()];
        let (pending, attention, convolution) = state.encoder_state.retained_tensors();
        let rnnt_state = state.rnnt_state.as_ref().ok_or_else(|| {
            Error::InferenceError("Nemotron realtime state has no RNNT control state".into())
        })?;
        let rnnt = rnnt_state
            .retained_tensors()
            .into_iter()
            .collect::<Vec<_>>();

        stage_physical_components(
            runtime,
            transaction,
            NEMOTRON_FEATURE_HISTORY_DOMAIN,
            target_cursor,
            feature,
        )?;
        stage_physical_components(
            runtime,
            transaction,
            NEMOTRON_PENDING_ENCODER_DOMAIN,
            target_cursor,
            vec![pending],
        )?;
        stage_physical_components(
            runtime,
            transaction,
            NEMOTRON_ATTENTION_HISTORY_DOMAIN,
            target_cursor,
            attention,
        )?;
        stage_physical_components(
            runtime,
            transaction,
            NEMOTRON_CONVOLUTION_HISTORY_DOMAIN,
            target_cursor,
            convolution,
        )?;
        stage_physical_components(
            runtime,
            transaction,
            NEMOTRON_RNNT_STATE_DOMAIN,
            target_cursor,
            rnnt,
        )?;

        self.clear_realtime_tensor_handles(state)
    }

    pub(crate) fn clear_realtime_tensor_handles(
        &self,
        state: &mut NemotronStreamingState,
    ) -> Result<()> {
        // The committed arena or active transaction owns the retained
        // handles. Between operations the native state keeps only
        // host/control metadata.
        state.pre_encode_state.install_retained_tensor(None);
        let layers = self.network.realtime_state_shape().encoder_layers;
        state.encoder_state.install_retained_tensors(
            None,
            vec![None; layers],
            vec![None; layers],
        )?;
        state
            .rnnt_state
            .as_mut()
            .expect("validated above")
            .install_retained_tensors(std::array::from_fn(|_| None));
        Ok(())
    }

    pub fn diagnostics(&self) -> serde_json::Value {
        json!({
            "variant": self.variant.dir_name(),
            "repo_id": self.variant.repo_id(),
            "device": format!("{:?}", self.device_profile.kind),
            "nemo_file": public_artifact_filename(&self.artifacts.nemo_path),
            "checkpoint_file": public_artifact_filename(&self.artifacts.checkpoint_path),
            "model_config_file": public_artifact_filename(&self.artifacts.model_config_path),
            "tokenizer_vocab_size": self.decoder.vocab_size(),
            "decoder_vocabulary_size": self.decoder.vocab_size(),
            "decoder_source": self.decoder.source(),
            "runtime": self.runtime_plan.diagnostics(),
            "dtype_plan": nemotron_dtype_diagnostics(
                &self.dtype_selection,
                &self.device_profile,
                self.network.dtype()
            ),
            "blank_id": self.network.blank_idx(),
            "native_forward_status": "enabled_offline_fastconformer_rnnt",
            "supports_realtime_cache_decode": true,
            "realtime_state_abi": "physical_v2",
            "supports_realtime_stream_decode": true,
        })
    }

    pub fn load(
        model_dir: &Path,
        variant: ModelVariant,
        device_profile: DeviceProfile,
    ) -> Result<Self> {
        if variant != ModelVariant::Nemotron35AsrStreaming06B {
            return Err(Error::InvalidInput(format!(
                "Variant {} is not a Nemotron 3.5 ASR model",
                variant.dir_name()
            )));
        }

        let artifacts = ensure_nemotron_artifacts(model_dir, variant)?;
        let runtime_plan = NemotronRuntimePlan::from_inventory(&artifacts.config_inventory)?;
        validate_config_output_vocabulary(&artifacts.config_inventory)?;
        let decoder = NemotronDecoder::load(&artifacts)?;
        let device = select_device_for_nemotron(&device_profile);
        let dtype_override = std::env::var(NEMOTRON_ASR_DTYPE_ENV).ok();
        let dtype_selection =
            select_nemotron_asr_dtype(&device_profile, dtype_override.as_deref())?;
        let checkpoint_display = artifacts.checkpoint_path.display();
        let vb = match VarBuilder::from_pth(
            &artifacts.checkpoint_path,
            dtype_selection.dtype,
            &device,
        ) {
            Ok(vb) => vb,
            Err(e) => {
                return Err(Error::ModelLoadError(format!(
                    "Failed to load Nemotron checkpoint {checkpoint_display}: {e}"
                )));
            }
        };
        let network = NemotronNetwork::load(&vb, &artifacts.config_inventory)?;
        info!(
            "Loaded Nemotron ASR model on {:?} with dtype {:?} ({})",
            device_profile.kind,
            dtype_selection.dtype,
            dtype_selection.reason.as_ref()
        );

        Ok(Self {
            variant,
            artifacts,
            decoder,
            network,
            runtime_plan,
            device_profile,
            dtype_selection,
        })
    }

    pub fn transcribe(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<String> {
        let request = self.prepare_decode_request(audio, sample_rate, language, None)?;
        let output = self.decode_offline_final(audio, &request, None)?;
        Ok(output.text)
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
        let request = self.prepare_decode_request(audio, sample_rate, language, prompt)?;
        let output = self.decode_offline(audio, &request, None, on_delta)?;
        Ok(output.text)
    }

    pub(crate) fn transcribe_with_callback_and_prompt_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        predictor: &mut InvocationTensorLease,
        acoustic: &mut InvocationTensorLease,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let request = self.prepare_decode_request(audio, sample_rate, language, prompt)?;
        let source_identity = nemotron_offline_source_identity(audio, sample_rate, &request);
        let output = self.decode_offline(
            audio,
            &request,
            Some(NemotronOfflinePhysicalState {
                predictor,
                acoustic,
                source_identity,
            }),
            on_delta,
        )?;
        Ok(output.text)
    }

    pub fn transcribe_with_details_and_prompt(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
    ) -> Result<NemotronAsrTranscriptionOutput> {
        let request = self.prepare_decode_request(audio, sample_rate, language, prompt)?;
        self.decode_offline_final(audio, &request, None)
    }

    pub(crate) fn transcribe_with_details_and_prompt_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        predictor: &mut InvocationTensorLease,
        acoustic: &mut InvocationTensorLease,
    ) -> Result<NemotronAsrTranscriptionOutput> {
        let request = self.prepare_decode_request(audio, sample_rate, language, prompt)?;
        let source_identity = nemotron_offline_source_identity(audio, sample_rate, &request);
        self.decode_offline_final(
            audio,
            &request,
            Some(NemotronOfflinePhysicalState {
                predictor,
                acoustic,
                source_identity,
            }),
        )
    }

    pub fn max_audio_seconds_hint(&self) -> Option<f32> {
        Some(DEFAULT_MAX_AUDIO_SECONDS_HINT)
    }

    pub fn available_streaming_profiles(&self) -> &[NemotronStreamingProfile] {
        &self.runtime_plan.streaming_profiles
    }

    pub fn start_stream_state(
        &self,
        language: Option<&str>,
        prompt: Option<&str>,
        right_context_frames: Option<usize>,
    ) -> Result<NemotronStreamingState> {
        self.start_stream_state_with_config(
            &NemotronRealtimeStreamConfig::new()
                .with_emit_partials(true)
                .with_optional_language(language)
                .with_optional_prompt(prompt)
                .with_optional_right_context_frames(right_context_frames),
        )
    }

    pub fn realtime_stream_resource_reservation(
        &self,
        language: Option<&str>,
        prompt: Option<&str>,
        right_context_frames: Option<usize>,
    ) -> Result<NemotronRealtimeResourceReservation> {
        self.realtime_stream_resource_reservation_with_config(
            &NemotronRealtimeStreamConfig::new()
                .with_emit_partials(true)
                .with_optional_language(language)
                .with_optional_prompt(prompt)
                .with_optional_right_context_frames(right_context_frames),
        )
    }

    pub fn conservative_realtime_stream_resource_reservation(
        language: Option<&str>,
        prompt: Option<&str>,
        right_context_frames: Option<usize>,
    ) -> Result<NemotronRealtimeResourceReservation> {
        let prompt = NemotronPromptCondition::resolve(language, prompt)?;
        let profile = NemotronStreamingProfile::new(56, right_context_frames.unwrap_or(13))?;
        let max_samples = configured_realtime_max_samples(SAMPLE_RATE)?;
        estimate_realtime_resource_reservation(
            max_samples,
            CONSERVATIVE_NEMOTRON_DECODER_TOKEN_BYTES,
            &profile,
            default_realtime_state_shape(),
            &prompt,
        )
    }

    fn realtime_stream_resource_reservation_with_config(
        &self,
        config: &NemotronRealtimeStreamConfig,
    ) -> Result<NemotronRealtimeResourceReservation> {
        let prompt = config.prompt_condition()?;
        let profile = self.resolve_streaming_profile(config.right_context_frames)?;
        let max_samples = configured_realtime_max_samples(self.runtime_plan.sample_rate)?;
        estimate_realtime_resource_reservation(
            max_samples,
            self.decoder.max_token_bytes(),
            &profile,
            self.network.realtime_state_shape(),
            &prompt,
        )
    }

    pub fn start_stream_state_with_config(
        &self,
        config: &NemotronRealtimeStreamConfig,
    ) -> Result<NemotronStreamingState> {
        let reservation = self.realtime_stream_resource_reservation_with_config(config)?;
        self.start_stream_state_with_config_and_reservation(config, reservation)
    }

    pub fn start_stream_state_with_reservation(
        &self,
        language: Option<&str>,
        prompt: Option<&str>,
        right_context_frames: Option<usize>,
        reservation: NemotronRealtimeResourceReservation,
    ) -> Result<NemotronStreamingState> {
        self.start_stream_state_with_config_and_reservation(
            &NemotronRealtimeStreamConfig::new()
                .with_emit_partials(true)
                .with_optional_language(language)
                .with_optional_prompt(prompt)
                .with_optional_right_context_frames(right_context_frames),
            reservation,
        )
    }

    fn start_stream_state_with_config_and_reservation(
        &self,
        config: &NemotronRealtimeStreamConfig,
        reservation: NemotronRealtimeResourceReservation,
    ) -> Result<NemotronStreamingState> {
        let prompt = config.prompt_condition()?;
        let profile = self.resolve_streaming_profile(config.right_context_frames)?;
        let required_reservation = estimate_realtime_resource_reservation(
            reservation.max_samples,
            self.decoder.max_token_bytes(),
            &profile,
            self.network.realtime_state_shape(),
            &prompt,
        )?;
        if reservation.host_bytes < required_reservation.host_bytes
            || reservation.tensor_bytes < required_reservation.tensor_bytes
            || reservation.max_output_tokens < required_reservation.max_output_tokens
            || reservation.max_text_bytes < required_reservation.max_text_bytes
        {
            return Err(Error::InvalidInput(
                "Nemotron realtime reservation is smaller than the loaded model's retained-state bound"
                    .to_string(),
            ));
        }
        let rnnt_state = self
            .network
            .start_rnnt_stream(reservation.max_output_tokens)?;

        let mut state = NemotronStreamingState::new(
            profile,
            prompt,
            self.runtime_plan.sample_rate,
            reservation.max_samples,
            reservation.max_text_bytes,
            config.emit_partials,
        );
        state.attach_rnnt_state(rnnt_state);
        Ok(state)
    }

    fn resolve_streaming_profile(
        &self,
        right_context_frames: Option<usize>,
    ) -> Result<NemotronStreamingProfile> {
        Ok(match right_context_frames {
            Some(right_context_frames) => self
                .runtime_plan
                .streaming_profiles
                .iter()
                .find(|profile| profile.right_context_frames == right_context_frames)
                .cloned()
                .ok_or_else(|| {
                    Error::InvalidInput(format!(
                        "Nemotron streaming profile with right-context {right_context_frames} is not available"
                    ))
                })?,
            None => self.runtime_plan.default_streaming_profile.clone(),
        })
    }

    pub fn push_stream_samples(
        &self,
        state: &mut NemotronStreamingState,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<NemotronRealtimeStreamEvent>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        self.run_realtime_batch(&mut [NemotronRealtimeBatchRow {
            state,
            input: NemotronRealtimeBatchInput::Push {
                samples,
                sample_rate,
            },
        }])
        .map(|mut rows| rows.pop().unwrap_or_default())
    }

    pub fn finish_stream(
        &self,
        state: &mut NemotronStreamingState,
    ) -> Result<Vec<NemotronRealtimeStreamEvent>> {
        self.run_realtime_batch(&mut [NemotronRealtimeBatchRow {
            state,
            input: NemotronRealtimeBatchInput::Finish,
        }])
        .map(|mut rows| rows.pop().unwrap_or_default())
    }

    pub(crate) fn run_realtime_batch(
        &self,
        rows: &mut [NemotronRealtimeBatchRow<'_>],
    ) -> Result<Vec<Vec<NemotronRealtimeStreamEvent>>> {
        if rows.is_empty() {
            return Err(Error::InvalidInput(
                "Nemotron realtime cohort cannot be empty".into(),
            ));
        }
        let checkpoints = rows.iter().map(|row| row.state.clone()).collect::<Vec<_>>();
        let result = self.run_realtime_batch_inner(rows);
        if result.is_err() {
            for (row, checkpoint) in rows.iter_mut().zip(checkpoints) {
                *row.state = checkpoint;
            }
        }
        result
    }

    fn run_realtime_batch_inner(
        &self,
        rows: &mut [NemotronRealtimeBatchRow<'_>],
    ) -> Result<Vec<Vec<NemotronRealtimeStreamEvent>>> {
        let finishing = rows
            .iter()
            .map(|row| matches!(row.input, NemotronRealtimeBatchInput::Finish))
            .collect::<Vec<_>>();
        let mut event_batches = rows
            .iter()
            .map(|row| NemotronRealtimeEventBatch::new(row.state))
            .collect::<Vec<_>>();
        for row in rows.iter_mut() {
            match &row.input {
                NemotronRealtimeBatchInput::Push {
                    samples,
                    sample_rate,
                } => {
                    if samples.is_empty() {
                        continue;
                    }
                    if row.state.input_finished {
                        return Err(Error::InvalidInput(
                            "Cannot push audio into a finalized Nemotron streaming state".into(),
                        ));
                    }
                    let (next_resampler, samples) = row.state.resampler.resample_chunk(
                        samples,
                        *sample_rate,
                        row.state.sample_rate,
                        row.state.max_samples,
                    )?;
                    row.state.push_samples(&samples)?;
                    row.state.resampler = next_resampler;
                }
                NemotronRealtimeBatchInput::Finish => {
                    let (next_resampler, tail) = row.state.resampler.finish()?;
                    if !tail.is_empty() {
                        row.state.push_samples(&tail)?;
                    }
                    row.state.resampler = next_resampler;
                    row.state.finish_input();
                }
            }
        }

        loop {
            let mut consumed_chunk = false;
            for row in rows.iter_mut() {
                let Some(chunk) = row.state.next_ready_chunk() else {
                    continue;
                };
                let chunk_samples =
                    row.state.samples[chunk.start_sample..chunk.end_sample].to_vec();
                row.state.feature_state.push_samples(&chunk_samples)?;
                if chunk.is_final {
                    row.state.feature_state.finish_input();
                }
                row.state.mark_chunk_consumed(&chunk)?;
                consumed_chunk = true;
            }
            self.drain_streaming_cohort(rows, &mut event_batches)?;
            if !consumed_chunk {
                break;
            }
        }

        let mut outputs = Vec::with_capacity(rows.len());
        for ((row, mut batch), finishing) in rows.iter_mut().zip(event_batches).zip(finishing) {
            if finishing && !row.state.final_event_emitted {
                batch.mark_final();
            }
            outputs.push(batch.into_events(row.state));
        }
        Ok(outputs)
    }

    pub fn start_decode_with_prompt(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        _max_new_tokens: usize,
    ) -> Result<NemotronStreamingState> {
        let mut state = self.start_stream_state(language, prompt, None)?;
        let audio_16khz = if sample_rate == self.runtime_plan.sample_rate {
            audio.to_vec()
        } else {
            resample_linear(audio, sample_rate, self.runtime_plan.sample_rate)
        };
        state.push_samples(&audio_16khz)?;
        state.finish_input();
        Ok(state)
    }

    pub fn decode_step(&self, state: &mut NemotronStreamingState) -> Result<NemotronAsrDecodeStep> {
        let events = self.decode_next_stream_chunk(state)?;
        let mut delta = String::new();
        let mut finished = false;
        for event in events {
            delta.push_str(&event.delta);
            finished |= event.is_final;
        }
        if state.input_finished && state.consumed_samples >= state.buffered_samples && !finished {
            let final_events = self.finish_stream(state)?;
            for event in final_events {
                delta.push_str(&event.delta);
                finished |= event.is_final;
            }
        }
        Ok(NemotronAsrDecodeStep {
            delta,
            text: state.assembled_text.clone(),
            tokens_generated: state.emitted_tokens,
            finished,
        })
    }

    fn decode_ready_stream_chunks(
        &self,
        state: &mut NemotronStreamingState,
    ) -> Result<Vec<NemotronRealtimeStreamEvent>> {
        let mut batch = NemotronRealtimeEventBatch::new(state);
        self.decode_ready_stream_chunks_into(state, &mut batch)?;
        Ok(batch.into_events(state))
    }

    fn decode_ready_stream_chunks_into(
        &self,
        state: &mut NemotronStreamingState,
        batch: &mut NemotronRealtimeEventBatch,
    ) -> Result<()> {
        while state.next_ready_chunk().is_some() {
            self.decode_next_stream_chunk_into(state, batch)?;
        }
        Ok(())
    }

    fn decode_next_stream_chunk(
        &self,
        state: &mut NemotronStreamingState,
    ) -> Result<Vec<NemotronRealtimeStreamEvent>> {
        let mut batch = NemotronRealtimeEventBatch::new(state);
        self.decode_next_stream_chunk_into(state, &mut batch)?;
        Ok(batch.into_events(state))
    }

    fn decode_next_stream_chunk_into(
        &self,
        state: &mut NemotronStreamingState,
        batch: &mut NemotronRealtimeEventBatch,
    ) -> Result<()> {
        let Some(chunk) = state.next_ready_chunk() else {
            return Ok(());
        };
        let chunk_samples = state.samples[chunk.start_sample..chunk.end_sample].to_vec();
        state.feature_state.push_samples(&chunk_samples)?;
        if chunk.is_final {
            state.feature_state.finish_input();
        }
        state.mark_chunk_consumed(&chunk)?;
        let prompt_id = self.network.prompt_id(&state.prompt.target_lang)?;
        self.drain_streaming_network(state, prompt_id, batch)
    }

    fn drain_streaming_network(
        &self,
        state: &mut NemotronStreamingState,
        prompt_id: usize,
        batch: &mut NemotronRealtimeEventBatch,
    ) -> Result<()> {
        loop {
            let mut progressed = false;

            if let Some(feature_chunk) = self
                .network
                .compute_streaming_features(&mut state.feature_state)?
            {
                state.pre_encode_state.push_features(feature_chunk)?;
                progressed = true;
            } else if state.input_finished {
                state.pre_encode_state.finish_input();
            }

            if let Some(pre_encoded) = self
                .network
                .pre_encode_streaming_chunk(&mut state.pre_encode_state)?
            {
                state.encoder_state.push_pre_encoded(pre_encoded)?;
                progressed = true;
            } else if state.input_finished {
                state.encoder_state.finish_input();
            }

            if let Some(encoder_chunk) = self
                .network
                .encode_streaming_chunk(&mut state.encoder_state, prompt_id)?
            {
                let (decoded, text) = {
                    let rnnt_state = state.rnnt_state.as_mut().ok_or_else(|| {
                        Error::InferenceError("Nemotron stream is missing RNNT state".to_string())
                    })?;
                    let mut ignored = |_token_id: usize| {};
                    let decoded = self.network.decode_rnnt_streaming_chunk(
                        rnnt_state,
                        &encoder_chunk.encoded,
                        encoder_chunk.frames,
                        &mut ignored,
                    )?;
                    let text = self.decoder.decode(rnnt_state.token_ids());
                    (decoded, text)
                };
                state.emitted_tokens = decoded.stats.emitted_tokens;
                batch.record_decoded_text(state, text, encoder_chunk.is_final)?;
                progressed = true;
            }

            if !progressed {
                break;
            }
        }
        Ok(())
    }

    fn drain_streaming_cohort(
        &self,
        rows: &mut [NemotronRealtimeBatchRow<'_>],
        event_batches: &mut [NemotronRealtimeEventBatch],
    ) -> Result<()> {
        if rows.len() != event_batches.len() {
            return Err(Error::InferenceError(
                "Nemotron realtime event cohort geometry mismatch".into(),
            ));
        }
        loop {
            let mut progressed = false;
            let mut encoded_chunks = (0..rows.len()).map(|_| None).collect::<Vec<_>>();
            for (index, row) in rows.iter_mut().enumerate() {
                let state = &mut row.state;
                if let Some(feature_chunk) = self
                    .network
                    .compute_streaming_features(&mut state.feature_state)?
                {
                    state.pre_encode_state.push_features(feature_chunk)?;
                    progressed = true;
                } else if state.input_finished {
                    state.pre_encode_state.finish_input();
                }

                if let Some(pre_encoded) = self
                    .network
                    .pre_encode_streaming_chunk(&mut state.pre_encode_state)?
                {
                    state.encoder_state.push_pre_encoded(pre_encoded)?;
                    progressed = true;
                } else if state.input_finished {
                    state.encoder_state.finish_input();
                }

                let prompt_id = self.network.prompt_id(&state.prompt.target_lang)?;
                if let Some(encoded) = self
                    .network
                    .encode_streaming_chunk(&mut state.encoder_state, prompt_id)?
                {
                    encoded_chunks[index] = Some(encoded);
                    progressed = true;
                }
            }

            if encoded_chunks.iter().any(Option::is_some) {
                let mut rnnt_states = Vec::new();
                let mut encoded = Vec::new();
                let mut encoded_lens = Vec::new();
                for (row, chunk) in rows.iter_mut().zip(encoded_chunks.iter()) {
                    let Some(chunk) = chunk.as_ref() else {
                        continue;
                    };
                    rnnt_states.push(row.state.rnnt_state.as_mut().ok_or_else(|| {
                        Error::InferenceError(
                            "Nemotron realtime cohort row has no RNNT state".into(),
                        )
                    })?);
                    encoded.push(&chunk.encoded);
                    encoded_lens.push(chunk.frames);
                }
                self.network.decode_rnnt_streaming_cohort(
                    &mut rnnt_states,
                    &encoded,
                    &encoded_lens,
                )?;
                drop(rnnt_states);

                for ((row, batch), chunk) in rows
                    .iter_mut()
                    .zip(event_batches.iter_mut())
                    .zip(encoded_chunks.iter())
                {
                    let Some(chunk) = chunk.as_ref() else {
                        continue;
                    };
                    let rnnt_state = row.state.rnnt_state.as_ref().ok_or_else(|| {
                        Error::InferenceError("Nemotron realtime cohort row lost RNNT state".into())
                    })?;
                    row.state.emitted_tokens = rnnt_state.token_ids().len();
                    batch.record_decoded_text(
                        row.state,
                        self.decoder.decode(rnnt_state.token_ids()),
                        chunk.is_final,
                    )?;
                }
            }

            if !progressed {
                break;
            }
        }
        Ok(())
    }

    pub fn diagnostics_for_prompt(
        &self,
        language: Option<&str>,
        prompt: Option<&str>,
    ) -> Result<serde_json::Value> {
        let prompt = NemotronPromptCondition::resolve(language, prompt)?;
        let prompt_id = self.network.prompt_id(&prompt.target_lang)?;
        Ok(json!({
            "variant": self.variant.dir_name(),
            "repo_id": self.variant.repo_id(),
            "device": format!("{:?}", self.device_profile.kind),
            "nemo_file": public_artifact_filename(&self.artifacts.nemo_path),
            "checkpoint_file": public_artifact_filename(&self.artifacts.checkpoint_path),
            "model_config_file": public_artifact_filename(&self.artifacts.model_config_path),
            "tokenizer_vocab_size": self.decoder.vocab_size(),
            "decoder_vocabulary_size": self.decoder.vocab_size(),
            "decoder_source": self.decoder.source(),
            "runtime": self.runtime_plan.diagnostics(),
            "dtype_plan": nemotron_dtype_diagnostics(&self.dtype_selection, &self.device_profile, self.network.dtype()),
            "prompt": prompt.diagnostics(),
            "prompt_id": prompt_id,
            "blank_id": self.network.blank_idx(),
            "native_forward_status": "enabled_offline_fastconformer_rnnt",
            "supports_realtime_cache_decode": true,
        }))
    }

    fn prepare_decode_request(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
    ) -> Result<NemotronDecodeRequest> {
        if audio.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }
        if sample_rate == 0 {
            return Err(Error::InvalidInput(
                "Audio sample rate must be greater than zero".to_string(),
            ));
        }

        Ok(NemotronDecodeRequest {
            samples: audio.len(),
            input_sample_rate: sample_rate,
            target_sample_rate: self.runtime_plan.sample_rate,
            prompt: NemotronPromptCondition::resolve(language, prompt)?,
        })
    }

    fn decode_offline(
        &self,
        audio: &[f32],
        request: &NemotronDecodeRequest,
        physical: Option<NemotronOfflinePhysicalState<'_>>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<NemotronAsrTranscriptionOutput> {
        let mut timings = NemotronStageTimings::default();
        let resample_start = Instant::now();
        let audio_16khz = if request.input_sample_rate == request.target_sample_rate {
            audio.to_vec()
        } else {
            resample_linear(audio, request.input_sample_rate, request.target_sample_rate)
        };
        timings.resample = resample_start.elapsed();
        let prompt_id = self.network.prompt_id(&request.prompt.target_lang)?;
        let encode_start = Instant::now();
        let (encoded, encoded_len, encode_profile) = self
            .network
            .encode_with_prompt_profile(&audio_16khz, prompt_id)?;
        timings.encode = encode_start.elapsed();
        timings.encode_profile = Some(encode_profile);

        let mut token_ids = Vec::<usize>::new();
        let mut assembled = String::new();
        let decoded = {
            let mut on_token = |token_id: usize| {
                token_ids.push(token_id);
                let text_start = Instant::now();
                let decoded = self.decoder.decode(&token_ids);
                let delta = text_delta(&assembled, &decoded);
                if !delta.is_empty() {
                    on_delta(delta.as_str());
                }
                assembled = decoded;
                timings.text_assembly += text_start.elapsed();
            };
            let decode_start = Instant::now();
            let decoded = match physical {
                Some(physical) => self.network.decode_rnnt_greedy_physical(
                    &encoded,
                    encoded_len,
                    physical.source_identity,
                    physical.predictor,
                    physical.acoustic,
                    &mut on_token,
                )?,
                None => self
                    .network
                    .decode_rnnt_greedy(&encoded, encoded_len, &mut on_token)?,
            };
            timings.rnnt_decode = decode_start.elapsed();
            decoded
        };
        if assembled.is_empty() {
            let text_start = Instant::now();
            assembled = self.decoder.decode(&decoded.token_ids);
            timings.text_assembly += text_start.elapsed();
        }

        Ok(self.build_decode_output(
            assembled,
            request,
            prompt_id,
            audio_16khz.len(),
            decoded,
            timings,
            "callback_delta",
        ))
    }

    fn decode_offline_final(
        &self,
        audio: &[f32],
        request: &NemotronDecodeRequest,
        physical: Option<NemotronOfflinePhysicalState<'_>>,
    ) -> Result<NemotronAsrTranscriptionOutput> {
        let mut timings = NemotronStageTimings::default();
        let resample_start = Instant::now();
        let audio_16khz = if request.input_sample_rate == request.target_sample_rate {
            audio.to_vec()
        } else {
            resample_linear(audio, request.input_sample_rate, request.target_sample_rate)
        };
        timings.resample = resample_start.elapsed();

        let prompt_id = self.network.prompt_id(&request.prompt.target_lang)?;
        let encode_start = Instant::now();
        let (encoded, encoded_len, encode_profile) = self
            .network
            .encode_with_prompt_profile(&audio_16khz, prompt_id)?;
        timings.encode = encode_start.elapsed();
        timings.encode_profile = Some(encode_profile);

        let mut no_op = |_token_id: usize| {};
        let decode_start = Instant::now();
        let decoded = match physical {
            Some(physical) => self.network.decode_rnnt_greedy_physical(
                &encoded,
                encoded_len,
                physical.source_identity,
                physical.predictor,
                physical.acoustic,
                &mut no_op,
            )?,
            None => self
                .network
                .decode_rnnt_greedy(&encoded, encoded_len, &mut no_op)?,
        };
        timings.rnnt_decode = decode_start.elapsed();

        let text_start = Instant::now();
        let assembled = self.decoder.decode(&decoded.token_ids);
        timings.text_assembly = text_start.elapsed();

        Ok(self.build_decode_output(
            assembled,
            request,
            prompt_id,
            audio_16khz.len(),
            decoded,
            timings,
            "final_only",
        ))
    }

    fn build_decode_output(
        &self,
        text: String,
        request: &NemotronDecodeRequest,
        prompt_id: usize,
        resampled_samples: usize,
        decoded: network::NemotronDecodedTokens,
        timings: NemotronStageTimings,
        decode_mode: &'static str,
    ) -> NemotronAsrTranscriptionOutput {
        let encode_diagnostics = timings
            .encode_profile
            .as_ref()
            .map(|profile| profile.stats.diagnostics());
        let feature_frames = timings
            .encode_profile
            .as_ref()
            .map(|profile| profile.stats.feature_frames);
        NemotronAsrTranscriptionOutput {
            text,
            language: Some(request.prompt.target_lang.clone()),
            diagnostics: Some(json!({
                "audio": {
                    "input_sample_rate": request.input_sample_rate,
                    "target_sample_rate": request.target_sample_rate,
                    "input_samples": request.samples,
                    "resampled_samples": resampled_samples,
                    "acoustic_frames": feature_frames,
                },
                "encode": encode_diagnostics,
                "prompt": request.prompt.diagnostics(),
                "prompt_id": prompt_id,
                "blank_id": self.network.blank_idx(),
                "dtype_plan": nemotron_dtype_diagnostics(&self.dtype_selection, &self.device_profile, self.network.dtype()),
                "native_forward_status": "enabled_offline_fastconformer_rnnt",
                "decode_mode": decode_mode,
                "decode": decoded.stats.diagnostics(),
                "timings_ms": timings.diagnostics(),
                "supports_realtime_cache_decode": true,
            })),
        }
    }
}

fn public_artifact_filename(path: &Path) -> Option<String> {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
}

fn select_device_for_nemotron(device_profile: &DeviceProfile) -> Device {
    device_profile.device.clone()
}

fn select_nemotron_asr_dtype(
    device_profile: &DeviceProfile,
    dtype_override: Option<&str>,
) -> Result<DTypeSelection> {
    let requested = dtype_override.map(str::trim).filter(|raw| !raw.is_empty());
    let request = DTypeSelectionRequest::new(if device_profile.kind.is_cuda() {
        requested
    } else {
        None
    })
    .with_model_family(ModelFamily::NemotronAsr);

    if device_profile.kind.is_cuda() && requested.is_some() {
        return device_profile.try_resolve_dtype(request).map_err(|err| {
            Error::InvalidInput(format!("Invalid CUDA Nemotron ASR dtype override: {err}"))
        });
    }

    Ok(device_profile.resolve_dtype(request))
}

fn nemotron_dtype_diagnostics(
    selection: &DTypeSelection,
    device_profile: &DeviceProfile,
    actual_network_dtype: DType,
) -> serde_json::Value {
    let cuda_compute_capability = device_profile
        .capabilities
        .cuda_compute_capability
        .map(|(major, minor)| format!("{major}.{minor}"));
    json!({
        "model_weights": format!("{:?}", selection.dtype),
        "activations": format!("{:?}", actual_network_dtype),
        "reason": selection.reason.to_string(),
        "device": format!("{:?}", device_profile.kind),
        "supports_bf16": device_profile.capabilities.supports_bf16,
        "supports_f16": device_profile.capabilities.supports_f16,
        "cuda_compute_capability": cuda_compute_capability,
        "cuda_device_name": device_profile.capabilities.cuda_device_name.as_deref(),
    })
}

fn validate_config_output_vocabulary(inventory: &NemotronConfigInventory) -> Result<()> {
    let Some(expected) = inventory.vocab_size else {
        return Ok(());
    };
    if inventory.output_vocabulary.is_empty() {
        return Ok(());
    }

    let actual = inventory.output_vocabulary.len();
    if actual != expected {
        return Err(Error::ModelLoadError(format!(
            "Nemotron output vocabulary length does not match config: labels={actual}, config={expected}"
        )));
    }
    Ok(())
}

fn normalize_target_lang(value: &str) -> Result<String> {
    let normalized = value.trim().replace('_', "-");
    if normalized.eq_ignore_ascii_case("auto") {
        return Ok("auto".to_string());
    }

    if let Some(locale) = SUPPORTED_TARGET_LANGS
        .iter()
        .copied()
        .find(|candidate| candidate.eq_ignore_ascii_case(&normalized))
    {
        return Ok(locale.to_string());
    }

    let alias_key = language_alias_key(&normalized);
    if let Some(locale) = default_locale_for_language_name(&alias_key) {
        return Ok(locale.to_string());
    }

    if let Some(locale) = default_locale_for_short_code(&normalized.to_ascii_lowercase()) {
        return Ok(locale.to_string());
    }

    Err(Error::InvalidInput(format!(
        "Unsupported Nemotron target_lang '{value}'. Use 'auto', a supported language name, or one of: {}",
        SUPPORTED_TARGET_LANGS.join(", ")
    )))
}

fn language_alias_key(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut last_was_space = true;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_was_space = false;
        } else if !last_was_space {
            out.push(' ');
            last_was_space = true;
        }
    }
    out.trim().to_string()
}

fn default_locale_for_language_name(name: &str) -> Option<&'static str> {
    match name {
        "english" | "american english" | "us english" | "u s english" | "united states english" => {
            Some("en-US")
        }
        "british english" | "uk english" | "u k english" | "united kingdom english" => {
            Some("en-GB")
        }
        "spanish" | "castilian" | "european spanish" | "spain spanish" => Some("es-ES"),
        "us spanish" | "u s spanish" | "united states spanish" | "american spanish" => {
            Some("es-US")
        }
        "french" | "european french" | "france french" => Some("fr-FR"),
        "canadian french" | "canada french" => Some("fr-CA"),
        "italian" => Some("it-IT"),
        "portuguese" | "brazilian portuguese" | "brazil portuguese" => Some("pt-BR"),
        "european portuguese" | "portugal portuguese" => Some("pt-PT"),
        "dutch" => Some("nl-NL"),
        "german" => Some("de-DE"),
        "turkish" => Some("tr-TR"),
        "russian" => Some("ru-RU"),
        "arabic" => Some("ar-AR"),
        "hindi" => Some("hi-IN"),
        "japanese" => Some("ja-JP"),
        "korean" => Some("ko-KR"),
        "vietnamese" => Some("vi-VN"),
        "ukrainian" => Some("uk-UA"),
        "polish" => Some("pl-PL"),
        "swedish" => Some("sv-SE"),
        "czech" => Some("cs-CZ"),
        "norwegian" | "norwegian bokmal" | "bokmal" => Some("nb-NO"),
        "norwegian nynorsk" | "nynorsk" => Some("nn-NO"),
        "danish" => Some("da-DK"),
        "bulgarian" => Some("bg-BG"),
        "finnish" => Some("fi-FI"),
        "croatian" => Some("hr-HR"),
        "slovak" => Some("sk-SK"),
        "chinese" | "mandarin" | "mandarin chinese" | "simplified chinese" => Some("zh-CN"),
        "hungarian" => Some("hu-HU"),
        "romanian" => Some("ro-RO"),
        "estonian" => Some("et-EE"),
        "greek" => Some("el-GR"),
        "lithuanian" => Some("lt-LT"),
        "latvian" => Some("lv-LV"),
        "maltese" => Some("mt-MT"),
        "slovenian" | "slovene" => Some("sl-SI"),
        "hebrew" => Some("he-IL"),
        "thai" => Some("th-TH"),
        _ => None,
    }
}

fn default_locale_for_short_code(code: &str) -> Option<&'static str> {
    match code {
        "en" => Some("en-US"),
        "es" => Some("es-ES"),
        "fr" => Some("fr-FR"),
        "it" => Some("it-IT"),
        "pt" => Some("pt-BR"),
        "nl" => Some("nl-NL"),
        "de" => Some("de-DE"),
        "tr" => Some("tr-TR"),
        "ru" => Some("ru-RU"),
        "ar" => Some("ar-AR"),
        "hi" => Some("hi-IN"),
        "ja" => Some("ja-JP"),
        "ko" => Some("ko-KR"),
        "vi" => Some("vi-VN"),
        "uk" => Some("uk-UA"),
        "pl" => Some("pl-PL"),
        "sv" => Some("sv-SE"),
        "cs" => Some("cs-CZ"),
        "no" | "nb" => Some("nb-NO"),
        "da" => Some("da-DK"),
        "bg" => Some("bg-BG"),
        "fi" => Some("fi-FI"),
        "hr" => Some("hr-HR"),
        "sk" => Some("sk-SK"),
        "zh" => Some("zh-CN"),
        "hu" => Some("hu-HU"),
        "ro" => Some("ro-RO"),
        "et" => Some("et-EE"),
        "el" => Some("el-GR"),
        "lt" => Some("lt-LT"),
        "lv" => Some("lv-LV"),
        "mt" => Some("mt-MT"),
        "sl" => Some("sl-SI"),
        "he" => Some("he-IL"),
        "th" => Some("th-TH"),
        "nn" => Some("nn-NO"),
        _ => None,
    }
}

fn looks_like_lang(value: &str) -> bool {
    let trimmed = value.trim();
    trimmed.eq_ignore_ascii_case("auto")
        || trimmed.len() == 2
        || (trimmed.len() == 5
            && trimmed
                .as_bytes()
                .get(2)
                .is_some_and(|separator| *separator == b'-' || *separator == b'_'))
}

fn non_empty_trimmed(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then_some(trimmed)
}

fn load_tokenizer_vocab(path: &Path) -> Result<Vec<String>> {
    let raw = fs::read_to_string(path).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed to read Nemotron tokenizer vocab {}: {}",
            path.display(),
            e
        ))
    })?;

    let vocab = raw
        .lines()
        .filter_map(|line| line.split('\t').next())
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    if vocab.is_empty() {
        return Err(Error::ModelLoadError(format!(
            "Nemotron tokenizer vocab at {} is empty",
            path.display()
        )));
    }
    Ok(vocab)
}

fn decode_vocab_tokens(ids: &[usize], vocab: &[String]) -> String {
    let mut out = String::new();

    for &id in ids {
        let Some(token) = vocab.get(id) else {
            continue;
        };
        if should_skip_token(token) {
            continue;
        }
        if token.starts_with('<') && token.ends_with('>') {
            continue;
        }
        if token.starts_with('▁') {
            let piece = token.trim_start_matches('▁');
            if !out.is_empty() && !out.ends_with(' ') {
                out.push(' ');
            }
            out.push_str(piece);
            continue;
        }
        if let Some(piece) = token.strip_prefix("##") {
            out.push_str(piece);
            continue;
        }
        out.push_str(token);
    }

    normalize_decoded_text(out)
}

fn text_delta(previous: &str, current: &str) -> String {
    if let Some(delta) = current.strip_prefix(previous) {
        return delta.to_string();
    }
    let common = previous
        .chars()
        .zip(current.chars())
        .take_while(|(a, b)| a == b)
        .count();
    current.chars().skip(common).collect()
}

fn should_skip_token(token: &str) -> bool {
    matches!(
        token,
        "<unk>" | "<pad>" | "<blank>" | "<s>" | "</s>" | "[UNK]" | "[PAD]" | "[BLANK]"
    )
}

fn normalize_decoded_text(mut text: String) -> String {
    text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    for punct in [".", ",", "!", "?", ":", ";"] {
        text = text.replace(&format!(" {punct}"), punct);
    }
    text.trim().to_string()
}

fn ms_to_samples(ms: usize, sample_rate: u32) -> usize {
    ((sample_rate as usize).saturating_mul(ms)) / 1000
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::PathBuf;

    use crate::backends::{DeviceCapabilities, DeviceKind, DeviceProfile, DeviceSelector};
    use uuid::Uuid;

    #[test]
    fn prompt_condition_defaults_to_auto_language() {
        let prompt = NemotronPromptCondition::resolve(None, None).unwrap();

        assert_eq!(prompt.target_lang, "auto");
        assert!(prompt.strip_lang_tags);
        assert!(prompt.context_prompt.is_none());
    }

    fn test_device_profile(
        kind: DeviceKind,
        supports_bf16: bool,
        supports_f16: bool,
    ) -> DeviceProfile {
        DeviceProfile {
            device: Device::Cpu,
            kind,
            capabilities: DeviceCapabilities {
                prefers_f32: kind.is_metal(),
                supports_bf16,
                supports_f16,
                cuda_compute_capability: kind.is_cuda().then_some((8, 9)),
                cuda_device_name: kind.is_cuda().then_some("test-cuda".to_string()),
                ..Default::default()
            },
            memory_pool: None,
        }
    }

    #[test]
    fn nemotron_dtype_plan_keeps_cpu_and_metal_f32() {
        let cpu = test_device_profile(DeviceKind::Cpu, true, true);
        let metal = test_device_profile(DeviceKind::Metal, true, true);

        assert_eq!(
            select_nemotron_asr_dtype(&cpu, None).unwrap().dtype,
            DType::F32
        );
        assert_eq!(
            select_nemotron_asr_dtype(&cpu, Some("bf16")).unwrap().dtype,
            DType::F32
        );
        assert_eq!(
            select_nemotron_asr_dtype(&metal, None).unwrap().dtype,
            DType::F32
        );
        assert_eq!(
            select_nemotron_asr_dtype(&metal, Some("f16"))
                .unwrap()
                .dtype,
            DType::F32
        );
    }

    #[test]
    fn nemotron_dtype_plan_uses_cuda_capability_order_and_diagnostics() {
        let cuda_bf16 = test_device_profile(DeviceKind::Cuda, true, true);
        let cuda_f16 = test_device_profile(DeviceKind::Cuda, false, true);
        let cuda_f32 = test_device_profile(DeviceKind::Cuda, false, false);

        let selection = select_nemotron_asr_dtype(&cuda_bf16, None).unwrap();
        assert_eq!(selection.dtype, DType::BF16);
        assert_eq!(
            select_nemotron_asr_dtype(&cuda_f16, None).unwrap().dtype,
            DType::F16
        );
        assert_eq!(
            select_nemotron_asr_dtype(&cuda_f32, None).unwrap().dtype,
            DType::F32
        );

        let diagnostics = nemotron_dtype_diagnostics(&selection, &cuda_bf16, DType::BF16);
        assert_eq!(diagnostics["model_weights"], "BF16");
        assert_eq!(diagnostics["activations"], "BF16");
        assert_eq!(diagnostics["device"], "Cuda");
        assert_eq!(diagnostics["cuda_compute_capability"], "8.9");
        assert_eq!(diagnostics["cuda_device_name"], "test-cuda");
    }

    #[test]
    fn nemotron_dtype_plan_rejects_bad_cuda_overrides() {
        let cuda = test_device_profile(DeviceKind::Cuda, false, true);

        let err = select_nemotron_asr_dtype(&cuda, Some("bf16")).unwrap_err();
        assert!(err.to_string().contains("Invalid CUDA Nemotron ASR"));

        let err = select_nemotron_asr_dtype(&cuda, Some("float8")).unwrap_err();
        assert!(err.to_string().contains("expected one of"));
    }

    #[test]
    fn stage_timings_diagnostics_report_milliseconds() {
        let timings = NemotronStageTimings {
            resample: Duration::from_micros(1_500),
            encode: Duration::from_millis(2),
            rnnt_decode: Duration::from_micros(3_250),
            text_assembly: Duration::from_millis(4),
            ..Default::default()
        };
        let diagnostics = timings.diagnostics();

        assert_eq!(diagnostics["resample"], 1.5);
        assert_eq!(diagnostics["audio_encode"], 2.0);
        assert_eq!(diagnostics["decode"], 3.25);
        assert_eq!(diagnostics["model_total"], 10.75);
        assert_eq!(diagnostics["resample_ms"], 1.5);
        assert_eq!(diagnostics["encode_ms"], 2.0);
        assert_eq!(diagnostics["rnnt_decode_ms"], 3.25);
        assert_eq!(diagnostics["text_assembly_ms"], 4.0);
    }

    #[test]
    fn public_artifact_diagnostics_only_expose_filenames() {
        let path = Path::new("/private/models/Nemotron/native/model_weights.ckpt");

        assert_eq!(
            public_artifact_filename(path).as_deref(),
            Some("model_weights.ckpt")
        );
        assert!(!public_artifact_filename(path)
            .expect("artifact filename")
            .contains("/private/models"));
    }

    #[test]
    fn prompt_condition_accepts_short_language_aliases() {
        let prompt = NemotronPromptCondition::resolve(Some("de"), None).unwrap();
        assert_eq!(prompt.target_lang, "de-DE");

        let prompt = NemotronPromptCondition::resolve(None, Some("en_US")).unwrap();
        assert_eq!(prompt.target_lang, "en-US");
        assert!(prompt.context_prompt.is_none());
    }

    #[test]
    fn prompt_condition_accepts_public_language_names() {
        let prompt = NemotronPromptCondition::resolve(Some("English"), None).unwrap();
        assert_eq!(prompt.target_lang, "en-US");

        let prompt = NemotronPromptCondition::resolve(Some("Auto"), None).unwrap();
        assert_eq!(prompt.target_lang, "auto");

        let prompt = NemotronPromptCondition::resolve(Some("British English"), None).unwrap();
        assert_eq!(prompt.target_lang, "en-GB");

        let prompt = NemotronPromptCondition::resolve(Some("Canadian French"), None).unwrap();
        assert_eq!(prompt.target_lang, "fr-CA");

        let prompt = NemotronPromptCondition::resolve(Some("European Portuguese"), None).unwrap();
        assert_eq!(prompt.target_lang, "pt-PT");

        let prompt = NemotronPromptCondition::resolve(Some("Mandarin"), None).unwrap();
        assert_eq!(prompt.target_lang, "zh-CN");
    }

    #[test]
    fn prompt_condition_preserves_non_language_prompt_as_context() {
        let prompt =
            NemotronPromptCondition::resolve(Some("fr-CA"), Some("medical dictation")).unwrap();

        assert_eq!(prompt.target_lang, "fr-CA");
        assert_eq!(prompt.context_prompt.as_deref(), Some("medical dictation"));
    }

    #[test]
    fn prompt_condition_rejects_unknown_language() {
        let err = NemotronPromptCondition::resolve(Some("xx-YY"), None).unwrap_err();

        assert!(err.to_string().contains("Unsupported Nemotron target_lang"));
    }

    #[test]
    fn prompt_condition_rejects_unsupported_public_language_name() {
        let err = NemotronPromptCondition::resolve(Some("Cantonese"), None).unwrap_err();
        let msg = err.to_string();

        assert!(msg.contains("Unsupported Nemotron target_lang 'Cantonese'"));
        assert!(msg.contains("supported language name"));
    }

    #[test]
    fn streaming_profile_maps_right_context_to_chunk_ms() {
        let profile = NemotronStreamingProfile::new(56, 13).unwrap();

        assert_eq!(profile.chunk_frames, 14);
        assert_eq!(profile.chunk_ms, 1120);
        assert_eq!(profile.chunk_samples(16_000), 17_920);
        assert_eq!(profile.right_context_samples(16_000), 16_640);
    }

    #[test]
    fn streaming_profiles_from_inventory_cover_all_model_card_profiles() {
        let inventory = NemotronConfigInventory {
            left_context_frames: Some(56),
            right_context_frames: vec![13, 0, 6, 1, 3],
            ..Default::default()
        };

        let profiles = NemotronStreamingProfile::profiles_from_inventory(&inventory).unwrap();
        let chunk_ms = profiles
            .iter()
            .map(|profile| profile.chunk_ms)
            .collect::<Vec<_>>();

        assert_eq!(chunk_ms, vec![80, 160, 320, 560, 1120]);
    }

    #[test]
    fn realtime_stream_config_resolves_prompt_and_profile_contract() {
        let literal = NemotronRealtimeStreamConfig {
            language: Some("German".to_string()),
            prompt: Some("medical dictation".to_string()),
            right_context_frames: Some(3),
            emit_partials: true,
        };
        let config = NemotronRealtimeStreamConfig::new()
            .with_language("German")
            .with_prompt("medical dictation")
            .with_right_context_frames(3);

        assert_eq!(literal, config);
        let prompt = config.prompt_condition().unwrap();
        let diagnostics = config.diagnostics();

        assert_eq!(prompt.target_lang, "de-DE");
        assert_eq!(prompt.context_prompt.as_deref(), Some("medical dictation"));
        assert_eq!(config.right_context_frames, Some(3));
        assert_eq!(diagnostics["target_lang"], "de-DE");
        assert_eq!(diagnostics["right_context_frames"], 3);
        assert_eq!(diagnostics["emit_partials"], true);
    }

    #[test]
    fn realtime_duration_limit_resolves_to_target_samples() {
        assert_eq!(
            realtime_max_samples_for_seconds(16_000, 300).unwrap(),
            4_800_000
        );
        assert!(realtime_max_samples_for_seconds(16_000, 0).is_err());
        assert!(realtime_max_samples_for_seconds(0, 300).is_err());
    }

    #[test]
    fn realtime_resource_reservation_prices_model_state_growth() {
        let profile = NemotronStreamingProfile::new(56, 3).unwrap();
        let prompt = NemotronPromptCondition::resolve(Some("en-US"), None).unwrap();
        let shape = NemotronRealtimeStateShape {
            feature_bins: 128,
            hop_length: 160,
            encoder_layers: 24,
            encoder_dim: 1_024,
            conv_kernel_size: 9,
            subsampling_factor: 8,
            predictor_hidden: 640,
            predictor_layers: 2,
            joint_hidden: 640,
            max_symbols_per_frame: 10,
        };

        let reservation =
            estimate_realtime_resource_reservation(16_000, 8, &profile, shape, &prompt).unwrap();

        assert_eq!(reservation.max_samples, 16_000);
        assert_eq!(reservation.max_output_tokens, 130);
        assert_eq!(reservation.max_text_bytes, 4_160);
        assert_eq!(reservation.host_bytes, 357_946);
        assert_eq!(reservation.tensor_bytes, 6_411_264);
        let retained_and_worker_audio_bytes =
            16_000_u64 * 2 * std::mem::size_of::<f32>() as u64 * 2;
        assert!(
            reservation.host_bytes
                >= REALTIME_HOST_FIXED_OVERHEAD_BYTES + retained_and_worker_audio_bytes
        );
    }

    #[test]
    fn realtime_event_batch_coalesces_large_chunk_bursts_to_one_bounded_event() {
        let profile = NemotronStreamingProfile::new(56, 0).unwrap();
        let prompt = NemotronPromptCondition::resolve(Some("en-US"), None).unwrap();
        let max_text_bytes = 8_192;
        let mut state =
            NemotronStreamingState::new(profile, prompt, 16_000, 16_000, max_text_bytes, true);
        let mut batch = NemotronRealtimeEventBatch::new(&state);
        let mut cumulative = String::new();

        for _ in 0..4_096 {
            cumulative.push('a');
            batch
                .record_decoded_text(&mut state, cumulative.clone(), false)
                .unwrap();
            assert_eq!(batch.pending_event_count(), 1);
            assert!(batch.retained_text_bytes() <= max_text_bytes);
        }

        let events = batch.into_events(&mut state);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].chunk_index, 0);
        assert_eq!(events[0].text.len(), 4_096);
        assert_eq!(events[0].delta.len(), 4_096);
        assert!(events[0].text.len() + events[0].delta.len() <= max_text_bytes * 2);
        assert_eq!(state.events_emitted, 1);
    }

    #[test]
    fn realtime_event_batch_honors_partial_policy_and_contiguous_indices() {
        let profile = NemotronStreamingProfile::new(56, 0).unwrap();
        let prompt = NemotronPromptCondition::resolve(Some("en-US"), None).unwrap();
        let mut suppressed =
            NemotronStreamingState::new(profile.clone(), prompt.clone(), 16_000, 128, 128, false);

        let mut partial = NemotronRealtimeEventBatch::new(&suppressed);
        partial
            .record_decoded_text(&mut suppressed, "hello".to_string(), false)
            .unwrap();
        assert!(partial.into_events(&mut suppressed).is_empty());
        assert_eq!(suppressed.text(), "hello");
        assert_eq!(suppressed.events_emitted, 0);

        let mut final_batch = NemotronRealtimeEventBatch::new(&suppressed);
        final_batch.mark_final();
        let final_events = final_batch.into_events(&mut suppressed);
        assert_eq!(final_events.len(), 1);
        assert!(final_events[0].is_final);
        assert_eq!(final_events[0].text, "hello");
        assert_eq!(final_events[0].delta, "hello");
        assert_eq!(final_events[0].chunk_index, 0);

        let mut partials = NemotronStreamingState::new(profile, prompt, 16_000, 128, 128, true);
        let mut first_batch = NemotronRealtimeEventBatch::new(&partials);
        first_batch
            .record_decoded_text(&mut partials, "hello".to_string(), false)
            .unwrap();
        let first_events = first_batch.into_events(&mut partials);
        assert_eq!(first_events.len(), 1);
        assert_eq!(first_events[0].chunk_index, 0);
        assert!(!first_events[0].is_final);

        let mut second_batch = NemotronRealtimeEventBatch::new(&partials);
        second_batch
            .record_decoded_text(&mut partials, "hello world".to_string(), true)
            .unwrap();
        let second_events = second_batch.into_events(&mut partials);
        assert_eq!(second_events.len(), 1);
        assert_eq!(second_events[0].chunk_index, 1);
        assert_eq!(second_events[0].delta, " world");
        assert!(second_events[0].is_final);
    }

    fn streaming_resample(
        samples: &[f32],
        source_rate: u32,
        target_rate: u32,
        chunk_sizes: &[usize],
        max_output_samples: usize,
    ) -> Result<Vec<f32>> {
        let mut state = NemotronStreamingResampler::default();
        let mut output = Vec::new();
        let mut offset = 0usize;
        let mut chunk_index = 0usize;
        while offset < samples.len() {
            let chunk_len = chunk_sizes[chunk_index % chunk_sizes.len()]
                .max(1)
                .min(samples.len() - offset);
            let (next, chunk) = state.resample_chunk(
                &samples[offset..offset + chunk_len],
                source_rate,
                target_rate,
                max_output_samples,
            )?;
            state = next;
            output.extend(chunk);
            offset += chunk_len;
            chunk_index += 1;
        }
        let (next, tail) = state.finish()?;
        state = next;
        output.extend(tail);
        assert_eq!(output.len(), state.projected_output_samples);
        Ok(output)
    }

    #[test]
    fn realtime_resampler_is_phase_continuous_across_randomized_chunks() {
        let samples = (0..4_411)
            .map(|index| ((index * 37 % 1_003) as f32 - 501.0) / 501.0)
            .collect::<Vec<_>>();
        let one_shot =
            streaming_resample(&samples, 44_100, 16_000, &[samples.len()], 16_000).unwrap();

        let mut seed = 0x9e37_79b9u32;
        let chunk_sizes = (0..256)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed as usize % 97) + 1
            })
            .collect::<Vec<_>>();
        let chunked = streaming_resample(&samples, 44_100, 16_000, &chunk_sizes, 16_000).unwrap();

        assert_eq!(chunked, one_shot);
        assert_eq!(chunked.len(), 1_600);
    }

    #[test]
    fn realtime_resampler_counts_repeated_single_samples_against_the_cap() {
        let samples = (0..8).map(|value| value as f32).collect::<Vec<_>>();
        let expected = streaming_resample(&samples, 8_000, 16_000, &[8], 16).unwrap();
        let mut state = NemotronStreamingResampler::default();
        let mut output = Vec::new();

        for sample in &samples {
            let (next, chunk) = state.resample_chunk(&[*sample], 8_000, 16_000, 16).unwrap();
            state = next;
            output.extend(chunk);
        }
        let (finished, tail) = state.finish().unwrap();
        state = finished;
        output.extend(tail);

        assert_eq!(output, expected);
        assert_eq!(output.len(), 16);
        assert_eq!(state.projected_output_samples, 16);
        let before = (
            state.source_samples,
            state.projected_output_samples,
            state.emitted_output_samples,
            state.retained_source_samples,
            state.last_source_sample,
        );
        let err = state.resample_chunk(&[8.0], 8_000, 16_000, 16).unwrap_err();
        assert!(err.to_string().contains("configured limit of 16 samples"));
        assert_eq!(
            before,
            (
                state.source_samples,
                state.projected_output_samples,
                state.emitted_output_samples,
                state.retained_source_samples,
                state.last_source_sample,
            )
        );
    }

    #[test]
    fn streaming_state_reports_physical_v2_cache_wiring() {
        let profile = NemotronStreamingProfile::new(56, 3).unwrap();
        let prompt = NemotronPromptCondition::resolve(Some("auto"), None).unwrap();
        let state = NemotronStreamingState::new(profile, prompt, 16_000, 4_096, 4_096, true);
        let diagnostics = state.diagnostics();

        assert_eq!(diagnostics["supports_realtime_cache_decode"], true);
        assert_eq!(diagnostics["cache_status"], "PhysicalStateV2");
        assert_eq!(diagnostics["profile"]["cache_reuse_ready"], true);
    }

    #[test]
    fn realtime_physical_contract_covers_every_tensor_without_paged_state() {
        let shape = default_realtime_state_shape();
        let contract =
            nemotron_realtime_state_contract(16_000 * 300, shape, 56, StateDType::F16).unwrap();

        assert_eq!(contract.domains.len(), 5);
        assert_eq!(contract.groups.len(), 2);
        assert_eq!(contract.groups[0].id, NEMOTRON_ENCODER_STATE_GROUP);
        assert_eq!(contract.groups[0].domains.len(), 4);
        assert_eq!(contract.groups[1].id, NEMOTRON_RNNT_STATE_GROUP);
        assert_eq!(contract.groups[1].domains, vec![NEMOTRON_RNNT_STATE_DOMAIN]);
        assert!(matches!(contract.domains[0], StateDomainSpec::Append(_)));
        assert!(matches!(contract.domains[1], StateDomainSpec::Append(_)));
        assert!(matches!(contract.domains[2], StateDomainSpec::Ring(_)));
        assert!(matches!(contract.domains[3], StateDomainSpec::Ring(_)));
        assert!(matches!(contract.domains[4], StateDomainSpec::Tensor(_)));
        assert!(!contract.domains.iter().any(|domain| matches!(
            domain,
            StateDomainSpec::PagedAttention(_) | StateDomainSpec::StaticAttention(_)
        )));
        let StateDomainSpec::Ring(attention) = &contract.domains[2] else {
            unreachable!()
        };
        let StateDomainSpec::Ring(convolution) = &contract.domains[3] else {
            unreachable!()
        };
        let StateDomainSpec::Tensor(rnnt) = &contract.domains[4] else {
            unreachable!()
        };
        assert_eq!(attention.components_per_step.len(), shape.encoder_layers);
        assert_eq!(attention.capacity_steps, 56);
        assert_eq!(convolution.components_per_step.len(), shape.encoder_layers);
        assert_eq!(
            convolution.capacity_steps,
            (shape.conv_kernel_size - 1) as u64
        );
        assert_eq!(rnnt.components.len(), 6);
        assert!(contract.domains.iter().all(|domain| match domain {
            StateDomainSpec::Append(spec) => spec
                .components_per_step
                .iter()
                .all(|component| component.accepted_dtypes == vec![StateDType::F16]),
            StateDomainSpec::Ring(spec) => spec
                .components_per_step
                .iter()
                .all(|component| component.accepted_dtypes == vec![StateDType::F16]),
            StateDomainSpec::Tensor(spec) => spec
                .components
                .iter()
                .all(|component| component.accepted_dtypes == vec![StateDType::F16]),
            _ => false,
        }));
    }

    #[test]
    fn streaming_state_retains_audio_samples_for_native_pipeline() {
        let profile = NemotronStreamingProfile::new(56, 0).unwrap();
        let prompt = NemotronPromptCondition::resolve(Some("en-US"), None).unwrap();
        let mut state = NemotronStreamingState::new(profile, prompt, 16_000, 4_096, 4_096, true);
        let empty_bytes = state.session_cache_bytes().unwrap();
        let empty_usage = state.session_resource_usage().unwrap();

        state.push_samples(&[0.1, 0.2, 0.3]).unwrap();
        state.push_samples(&[0.4]).unwrap();
        let usage = state.session_resource_usage().unwrap();

        assert_eq!(state.buffered_samples(), 4);
        assert_eq!(state.samples, vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(state.text(), "");
        assert_eq!(state.emitted_tokens(), 0);
        assert!(state.session_cache_bytes().unwrap() >= empty_bytes + 4 * 4);
        assert!(usage.host_bytes >= empty_usage.host_bytes + 4 * 4);
        assert_eq!(usage.tensor_bytes, 0);
        assert_eq!(
            state.session_cache_bytes(),
            usage.host_bytes.checked_add(usage.tensor_bytes)
        );
        assert_eq!(
            state.diagnostics()["supports_realtime_stream_decode"],
            false
        );
    }

    #[test]
    fn streaming_state_rejects_sample_limit_without_mutation() {
        let profile = NemotronStreamingProfile::new(56, 0).unwrap();
        let prompt = NemotronPromptCondition::resolve(Some("en-US"), None).unwrap();
        let mut state = NemotronStreamingState::new(profile, prompt, 16_000, 4, 4_096, true);
        state.push_samples(&[0.1, 0.2, 0.3]).unwrap();
        let samples_before = state.samples.clone();
        let buffered_before = state.buffered_samples();
        let bytes_before = state.session_cache_bytes();

        let err = state.push_samples(&[0.4, 0.5]).unwrap_err();

        assert!(err.to_string().contains("configured limit of 4 samples"));
        assert_eq!(state.samples, samples_before);
        assert_eq!(state.buffered_samples(), buffered_before);
        assert_eq!(state.session_cache_bytes(), bytes_before);
    }

    #[test]
    fn streaming_state_emits_non_overlapping_ready_chunks() {
        let profile = NemotronStreamingProfile::new(56, 1).unwrap();
        let prompt = NemotronPromptCondition::resolve(Some("en-US"), None).unwrap();
        let mut state = NemotronStreamingState::new(profile, prompt, 16_000, 4_096, 4_096, true);

        state.push_samples(&vec![0.0; 2_560]).unwrap();
        let first = state.next_ready_chunk().expect("first chunk");
        assert_eq!(first.start_sample, 0);
        assert_eq!(first.end_sample, 2_560);
        assert!(!first.is_final);
        state.mark_chunk_consumed(&first).unwrap();

        state.push_samples(&vec![0.0; 1_280]).unwrap();
        assert!(state.next_ready_chunk().is_none());
        state.finish_input();
        let tail = state.next_ready_chunk().expect("final tail chunk");
        assert_eq!(tail.start_sample, 2_560);
        assert_eq!(tail.end_sample, 3_840);
        assert!(tail.is_final);
    }

    #[test]
    fn streaming_state_rejects_out_of_order_chunk_accounting() {
        let profile = NemotronStreamingProfile::new(56, 0).unwrap();
        let prompt = NemotronPromptCondition::resolve(None, None).unwrap();
        let mut state = NemotronStreamingState::new(profile, prompt, 16_000, 4_096, 4_096, true);
        state.push_samples(&vec![0.0; 1_280]).unwrap();

        let mut chunk = state.next_ready_chunk().unwrap();
        chunk.chunk_index = 3;
        let err = state.mark_chunk_consumed(&chunk).unwrap_err();

        assert!(err.to_string().contains("chunk index mismatch"));
    }

    #[test]
    fn decoder_prefers_config_labels_over_short_vocab_txt() {
        let temp_dir = std::env::temp_dir().join(format!("nemotron-decoder-{}", Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).unwrap();
        let vocab_path = temp_dir.join("vocab.txt");
        fs::write(&vocab_path, "##hello\n##world\n").unwrap();
        let artifacts = NemotronArtifacts {
            nemo_path: temp_dir.join("model.nemo"),
            extracted_dir: temp_dir.clone(),
            model_config_path: temp_dir.join("model_config.yaml"),
            checkpoint_path: temp_dir.join("model_weights.ckpt"),
            tokenizer_paths: vec![vocab_path],
            config_inventory: NemotronConfigInventory {
                vocab_size: Some(4),
                output_vocabulary: vec![
                    "<unk>".to_string(),
                    "<en-US>".to_string(),
                    "▁hello".to_string(),
                    "▁world".to_string(),
                ],
                ..Default::default()
            },
        };

        validate_config_output_vocabulary(&artifacts.config_inventory).unwrap();
        let decoder = NemotronDecoder::load(&artifacts).unwrap();

        assert_eq!(decoder.source(), "config_labels");
        assert_eq!(decoder.vocab_size(), 4);
        assert_eq!(decoder.decode(&[0, 1, 2, 3]), "hello world");

        fs::remove_dir_all(temp_dir).unwrap();
    }

    #[test]
    fn config_output_vocabulary_must_match_config_vocab_size() {
        let inventory = NemotronConfigInventory {
            vocab_size: Some(3),
            output_vocabulary: vec!["<unk>".to_string(), "hello".to_string()],
            ..Default::default()
        };

        let err = validate_config_output_vocabulary(&inventory).unwrap_err();

        assert!(err
            .to_string()
            .contains("output vocabulary length does not match config"));
    }

    #[test]
    fn vocab_decoder_skips_control_and_language_tags() {
        let vocab = vec![
            "<blank>".to_string(),
            "▁Hello".to_string(),
            ",".to_string(),
            "▁world".to_string(),
            "!".to_string(),
            "<en-US>".to_string(),
        ];

        assert_eq!(
            decode_vocab_tokens(&[0, 1, 2, 3, 4, 5], &vocab),
            "Hello, world!"
        );
    }

    #[test]
    #[ignore = "requires local Nemotron-3.5-ASR-Streaming-0.6B assets and loads a 2.4 GB checkpoint"]
    fn nemotron_local_silence_forward_smoke_if_available() {
        let models_root = std::env::var("IZWI_MODELS_DIR")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                dirs::data_local_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join("izwi")
                    .join("models")
            });
        let model_dir = models_root.join(ModelVariant::Nemotron35AsrStreaming06B.dir_name());
        let ckpt_path = model_dir.join("nemotron-native").join("model_weights.ckpt");
        if !ckpt_path.exists() {
            eprintln!(
                "Skipping local Nemotron smoke test, checkpoint not found at {}",
                ckpt_path.display()
            );
            return;
        }

        let backend = std::env::var("IZWI_NEMOTRON_ASR_SMOKE_BACKEND")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| "cpu".to_string());
        let device =
            DeviceSelector::detect_with_preference(Some(&backend)).expect("requested device");
        if backend.eq_ignore_ascii_case("metal") && device.kind != DeviceKind::Metal {
            eprintln!("Skipping local Nemotron Metal smoke test, Metal device was not selected");
            return;
        }
        let model =
            NemotronAsrModel::load(&model_dir, ModelVariant::Nemotron35AsrStreaming06B, device)
                .expect("Nemotron ASR model should load");
        let silence = vec![0.0f32; 1_600];
        let output = model
            .transcribe_with_details_and_prompt(&silence, 16_000, Some("English"), None)
            .expect("Nemotron silent forward should run");

        assert_eq!(output.language.as_deref(), Some("en-US"));
        let diagnostics = output.diagnostics.expect("diagnostics");
        assert_eq!(
            diagnostics["native_forward_status"],
            "enabled_offline_fastconformer_rnnt"
        );
        assert_eq!(diagnostics["prompt_id"], 0);
        assert!(diagnostics["decode"]["encoded_frames"]
            .as_u64()
            .is_some_and(|frames| frames > 0));
    }
}
