//! Nemotron 3.5 ASR artifact and native inference support.

pub mod config;
pub mod nemo;
mod network;

use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use serde_json::json;
use tracing::info;

use crate::backends::{DTypeSelection, DTypeSelectionRequest, DeviceProfile};
use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::tokenizer::Tokenizer;

pub use config::NemotronConfigInventory;
pub use nemo::{ensure_nemotron_artifacts, NemotronArtifacts, NEMOTRON_NEMO_FILENAME};
use network::{
    resample_linear, NemotronNetwork, NemotronRnntStreamState, NemotronStreamingEncoderState,
    NemotronStreamingFeatureState, NemotronStreamingPreEncodeState,
};

const SAMPLE_RATE: u32 = 16_000;
const DEFAULT_STRIP_LANG_TAGS: bool = true;
const DEFAULT_MAX_AUDIO_SECONDS_HINT: f32 = 30.0;
const STREAMING_FRAME_MS: usize = 80;
const NEMOTRON_ASR_DTYPE_ENV: &str = "IZWI_NEMOTRON_ASR_DTYPE";
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

pub struct NemotronAsrModel {
    variant: ModelVariant,
    artifacts: NemotronArtifacts,
    decoder: NemotronDecoder,
    network: NemotronNetwork,
    runtime_plan: NemotronRuntimePlan,
    device_profile: DeviceProfile,
    dtype_selection: DTypeSelection,
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
            "cache_reuse_ready": false,
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
    PendingNativeCacheImplementation,
}

pub struct NemotronStreamingState {
    profile: NemotronStreamingProfile,
    prompt: NemotronPromptCondition,
    sample_rate: u32,
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

impl NemotronStreamingState {
    fn new(
        profile: NemotronStreamingProfile,
        prompt: NemotronPromptCondition,
        sample_rate: u32,
    ) -> Self {
        let encoder_state = NemotronStreamingEncoderState::new(
            profile.left_context_frames,
            profile.right_context_frames,
        );
        Self {
            profile,
            prompt,
            sample_rate,
            samples: Vec::new(),
            buffered_samples: 0,
            consumed_samples: 0,
            chunks_processed: 0,
            events_emitted: 0,
            input_finished: false,
            final_event_emitted: false,
            cache_status: NemotronStreamingCacheStatus::PendingNativeCacheImplementation,
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

    pub fn push_samples(&mut self, samples: &[f32]) -> Result<()> {
        if self.input_finished {
            return Err(Error::InvalidInput(
                "Cannot push audio into a finalized Nemotron streaming state".to_string(),
            ));
        }
        self.samples.extend_from_slice(samples);
        self.buffered_samples = self.buffered_samples.saturating_add(samples.len());
        Ok(())
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
            "buffered_samples": self.buffered_samples,
            "consumed_samples": self.consumed_samples,
            "chunks_processed": self.chunks_processed,
            "events_emitted": self.events_emitted,
            "input_finished": self.input_finished,
            "final_event_emitted": self.final_event_emitted,
            "emitted_tokens": self.emitted_tokens,
            "cache_status": format!("{:?}", self.cache_status),
            "supports_realtime_cache_decode": false,
            "supports_realtime_stream_decode": self.rnnt_state.is_some(),
        })
    }
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
    rnnt_decode: Duration,
    text_assembly: Duration,
}

impl NemotronStageTimings {
    fn diagnostics(&self) -> serde_json::Value {
        json!({
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

impl NemotronAsrModel {
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
        let output = self.decode_offline_final(audio, &request)?;
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
        let output = self.decode_offline(audio, &request, on_delta)?;
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
        self.decode_offline_final(audio, &request)
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

    pub fn start_stream_state_with_config(
        &self,
        config: &NemotronRealtimeStreamConfig,
    ) -> Result<NemotronStreamingState> {
        let prompt = config.prompt_condition()?;
        let profile = self.resolve_streaming_profile(config.right_context_frames)?;
        let rnnt_state = self.network.start_rnnt_stream()?;

        let mut state = NemotronStreamingState::new(profile, prompt, self.runtime_plan.sample_rate);
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
        if sample_rate == 0 {
            return Err(Error::InvalidInput(
                "Audio sample rate must be greater than zero".to_string(),
            ));
        }
        let samples = if sample_rate == state.sample_rate {
            samples.to_vec()
        } else {
            resample_linear(samples, sample_rate, state.sample_rate)
        };
        state.push_samples(&samples)?;
        self.decode_ready_stream_chunks(state)
    }

    pub fn finish_stream(
        &self,
        state: &mut NemotronStreamingState,
    ) -> Result<Vec<NemotronRealtimeStreamEvent>> {
        state.finish_input();
        let mut events = self.decode_ready_stream_chunks(state)?;
        if !state.final_event_emitted {
            state.final_event_emitted = true;
            let event = NemotronRealtimeStreamEvent {
                text: state.assembled_text.clone(),
                delta: String::new(),
                is_final: true,
                chunk_index: state.events_emitted,
            };
            state.events_emitted = state.events_emitted.saturating_add(1);
            events.push(event);
        }
        Ok(events)
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
        let mut events = Vec::new();
        while state.next_ready_chunk().is_some() {
            events.extend(self.decode_next_stream_chunk(state)?);
        }
        Ok(events)
    }

    fn decode_next_stream_chunk(
        &self,
        state: &mut NemotronStreamingState,
    ) -> Result<Vec<NemotronRealtimeStreamEvent>> {
        let Some(chunk) = state.next_ready_chunk() else {
            return Ok(Vec::new());
        };
        let chunk_samples = state.samples[chunk.start_sample..chunk.end_sample].to_vec();
        state.feature_state.push_samples(&chunk_samples)?;
        if chunk.is_final {
            state.feature_state.finish_input();
        }
        state.mark_chunk_consumed(&chunk)?;
        let prompt_id = self.network.prompt_id(&state.prompt.target_lang)?;
        self.drain_streaming_network(state, prompt_id)
    }

    fn drain_streaming_network(
        &self,
        state: &mut NemotronStreamingState,
        prompt_id: usize,
    ) -> Result<Vec<NemotronRealtimeStreamEvent>> {
        let mut events = Vec::new();
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
                state.emitted_tokens = decoded.stats.emitted_tokens;
                let text = self.decoder.decode(&decoded.token_ids);
                let delta = text_delta(&state.assembled_text, &text);
                state.assembled_text = text.clone();
                if !delta.is_empty() || encoder_chunk.is_final {
                    let is_final = encoder_chunk.is_final;
                    state.final_event_emitted |= is_final;
                    events.push(NemotronRealtimeStreamEvent {
                        text,
                        delta,
                        is_final,
                        chunk_index: state.events_emitted,
                    });
                    state.events_emitted = state.events_emitted.saturating_add(1);
                }
                progressed = true;
            }

            if !progressed {
                break;
            }
        }
        Ok(events)
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
            "nemo_path": self.artifacts.nemo_path.display().to_string(),
            "checkpoint_path": self.artifacts.checkpoint_path.display().to_string(),
            "model_config_path": self.artifacts.model_config_path.display().to_string(),
            "tokenizer_vocab_size": self.decoder.vocab_size(),
            "decoder_vocabulary_size": self.decoder.vocab_size(),
            "decoder_source": self.decoder.source(),
            "runtime": self.runtime_plan.diagnostics(),
            "dtype_plan": nemotron_dtype_diagnostics(&self.dtype_selection, &self.device_profile, self.network.dtype()),
            "prompt": prompt.diagnostics(),
            "prompt_id": prompt_id,
            "blank_id": self.network.blank_idx(),
            "native_forward_status": "enabled_offline_fastconformer_rnnt",
            "supports_realtime_cache_decode": false,
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
        let (encoded, encoded_len) = self.network.encode_with_prompt(&audio_16khz, prompt_id)?;
        timings.encode = encode_start.elapsed();

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
            let decoded = self
                .network
                .decode_rnnt_greedy(&encoded, encoded_len, &mut on_token)?;
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
        let (encoded, encoded_len) = self.network.encode_with_prompt(&audio_16khz, prompt_id)?;
        timings.encode = encode_start.elapsed();

        let mut no_op = |_token_id: usize| {};
        let decode_start = Instant::now();
        let decoded = self
            .network
            .decode_rnnt_greedy(&encoded, encoded_len, &mut no_op)?;
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
        NemotronAsrTranscriptionOutput {
            text,
            language: Some(request.prompt.target_lang.clone()),
            diagnostics: Some(json!({
                "audio": {
                    "input_sample_rate": request.input_sample_rate,
                    "target_sample_rate": request.target_sample_rate,
                    "input_samples": request.samples,
                    "resampled_samples": resampled_samples,
                },
                "prompt": request.prompt.diagnostics(),
                "prompt_id": prompt_id,
                "blank_id": self.network.blank_idx(),
                "dtype_plan": nemotron_dtype_diagnostics(&self.dtype_selection, &self.device_profile, self.network.dtype()),
                "native_forward_status": "enabled_offline_fastconformer_rnnt",
                "decode_mode": decode_mode,
                "decode": decoded.stats.diagnostics(),
                "timings_ms": timings.diagnostics(),
                "supports_realtime_cache_decode": false,
            })),
        }
    }
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
        };
        let diagnostics = timings.diagnostics();

        assert_eq!(diagnostics["resample_ms"], 1.5);
        assert_eq!(diagnostics["encode_ms"], 2.0);
        assert_eq!(diagnostics["rnnt_decode_ms"], 3.25);
        assert_eq!(diagnostics["text_assembly_ms"], 4.0);
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
        let config = NemotronRealtimeStreamConfig::new()
            .with_language("German")
            .with_prompt("medical dictation")
            .with_right_context_frames(3);

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
    fn streaming_state_contract_does_not_claim_native_cache_before_wiring() {
        let profile = NemotronStreamingProfile::new(56, 3).unwrap();
        let prompt = NemotronPromptCondition::resolve(Some("auto"), None).unwrap();
        let state = NemotronStreamingState::new(profile, prompt, 16_000);
        let diagnostics = state.diagnostics();

        assert_eq!(diagnostics["supports_realtime_cache_decode"], false);
        assert_eq!(
            diagnostics["cache_status"],
            "PendingNativeCacheImplementation"
        );
        assert_eq!(diagnostics["profile"]["cache_reuse_ready"], false);
    }

    #[test]
    fn streaming_state_retains_audio_samples_for_native_pipeline() {
        let profile = NemotronStreamingProfile::new(56, 0).unwrap();
        let prompt = NemotronPromptCondition::resolve(Some("en-US"), None).unwrap();
        let mut state = NemotronStreamingState::new(profile, prompt, 16_000);

        state.push_samples(&[0.1, 0.2, 0.3]).unwrap();
        state.push_samples(&[0.4]).unwrap();

        assert_eq!(state.buffered_samples(), 4);
        assert_eq!(state.samples, vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(state.text(), "");
        assert_eq!(state.emitted_tokens(), 0);
        assert_eq!(state.diagnostics()["supports_realtime_stream_decode"], false);
    }

    #[test]
    fn streaming_state_emits_non_overlapping_ready_chunks() {
        let profile = NemotronStreamingProfile::new(56, 1).unwrap();
        let prompt = NemotronPromptCondition::resolve(Some("en-US"), None).unwrap();
        let mut state = NemotronStreamingState::new(profile, prompt, 16_000);

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
        let mut state = NemotronStreamingState::new(profile, prompt, 16_000);
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
