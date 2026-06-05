//! Nemotron 3.5 ASR artifact and native inference support.

pub mod config;
pub mod nemo;

use std::fs;
use std::path::Path;

use serde_json::json;

use crate::backends::DeviceProfile;
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::tokenizer::Tokenizer;

pub use config::NemotronConfigInventory;
pub use nemo::{ensure_nemotron_artifacts, NemotronArtifacts, NEMOTRON_NEMO_FILENAME};

const SAMPLE_RATE: u32 = 16_000;
const DEFAULT_STRIP_LANG_TAGS: bool = true;
const DEFAULT_MAX_AUDIO_SECONDS_HINT: f32 = 30.0;
const STREAMING_FRAME_MS: usize = 80;
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

pub struct NemotronAsrModel {
    variant: ModelVariant,
    artifacts: NemotronArtifacts,
    decoder: NemotronDecoder,
    runtime_plan: NemotronRuntimePlan,
    device_profile: DeviceProfile,
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
    encoder_layers: Option<usize>,
    encoder_dim: Option<usize>,
    encoder_heads: Option<usize>,
    predictor_hidden: Option<usize>,
    joint_hidden: Option<usize>,
    prompt_dim: Option<usize>,
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
        let default_streaming_profile = streaming_profiles
            .last()
            .cloned()
            .ok_or_else(|| {
                Error::ModelLoadError("Nemotron config did not yield a streaming profile".to_string())
            })?;

        Ok(Self {
            sample_rate: SAMPLE_RATE,
            feature_bins: inventory.features,
            encoder_layers: inventory.encoder_layers,
            encoder_dim: inventory.encoder_dim,
            encoder_heads: inventory.encoder_heads,
            predictor_hidden: inventory.predictor_hidden,
            joint_hidden: inventory.joint_hidden,
            prompt_dim: inventory.prompt_dim,
            vocab_size: inventory.vocab_size,
            default_streaming_profile,
            streaming_profiles,
        })
    }

    fn diagnostics(&self) -> serde_json::Value {
        json!({
            "sample_rate": self.sample_rate,
            "feature_bins": self.feature_bins,
            "encoder_layers": self.encoder_layers,
            "encoder_dim": self.encoder_dim,
            "encoder_heads": self.encoder_heads,
            "predictor_hidden": self.predictor_hidden,
            "joint_hidden": self.joint_hidden,
            "prompt_dim": self.prompt_dim,
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
pub enum NemotronStreamingCacheStatus {
    PendingNativeCacheImplementation,
}

#[derive(Debug, Clone)]
pub struct NemotronStreamingState {
    profile: NemotronStreamingProfile,
    prompt: NemotronPromptCondition,
    sample_rate: u32,
    buffered_samples: usize,
    consumed_samples: usize,
    chunks_processed: usize,
    input_finished: bool,
    cache_status: NemotronStreamingCacheStatus,
}

impl NemotronStreamingState {
    fn new(
        profile: NemotronStreamingProfile,
        prompt: NemotronPromptCondition,
        sample_rate: u32,
    ) -> Self {
        Self {
            profile,
            prompt,
            sample_rate,
            buffered_samples: 0,
            consumed_samples: 0,
            chunks_processed: 0,
            input_finished: false,
            cache_status: NemotronStreamingCacheStatus::PendingNativeCacheImplementation,
        }
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

    pub fn push_samples(&mut self, samples: &[f32]) -> Result<()> {
        if self.input_finished {
            return Err(Error::InvalidInput(
                "Cannot push audio into a finalized Nemotron streaming state".to_string(),
            ));
        }
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
            "input_finished": self.input_finished,
            "cache_status": format!("{:?}", self.cache_status),
            "supports_realtime_cache_decode": false,
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
        let language_hint = language
            .and_then(non_empty_trimmed)
            .or_else(|| prompt.and_then(non_empty_trimmed).filter(|value| looks_like_lang(value)));
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

        Ok(Self {
            variant,
            artifacts,
            decoder,
            runtime_plan,
            device_profile,
        })
    }

    pub fn transcribe(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<String> {
        let mut no_op = |_delta: &str| {};
        self.transcribe_with_callback_and_prompt(audio, sample_rate, language, None, &mut no_op)
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
        _on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let request = self.prepare_decode_request(audio, sample_rate, language, prompt)?;
        Err(self.native_forward_not_ready_error(&request))
    }

    pub fn transcribe_with_details_and_prompt(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
    ) -> Result<NemotronAsrTranscriptionOutput> {
        let request = self.prepare_decode_request(audio, sample_rate, language, prompt)?;
        Err(self.native_forward_not_ready_error(&request))
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
        let prompt = NemotronPromptCondition::resolve(language, prompt)?;
        let profile = match right_context_frames {
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
        };

        Ok(NemotronStreamingState::new(
            profile,
            prompt,
            self.runtime_plan.sample_rate,
        ))
    }

    pub fn diagnostics_for_prompt(
        &self,
        language: Option<&str>,
        prompt: Option<&str>,
    ) -> Result<serde_json::Value> {
        let prompt = NemotronPromptCondition::resolve(language, prompt)?;
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
            "prompt": prompt.diagnostics(),
            "native_forward_status": "pending_fastconformer_rnnt_weight_mapping",
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

    fn native_forward_not_ready_error(&self, request: &NemotronDecodeRequest) -> Error {
        Error::InferenceError(format!(
            "Nemotron 3.5 ASR native FastConformer-RNNT forward pass is not enabled yet; loaded artifacts and prompt diagnostics: {}",
            request.diagnostics()
        ))
    }
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

    if let Some(locale) = default_locale_for_short_code(&normalized.to_ascii_lowercase()) {
        return Ok(locale.to_string());
    }

    Err(Error::InvalidInput(format!(
        "Unsupported Nemotron target_lang '{value}'. Use 'auto' or one of: {}",
        SUPPORTED_TARGET_LANGS.join(", ")
    )))
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

#[cfg(test)]
mod tests {
    use super::*;

    use uuid::Uuid;

    #[test]
    fn prompt_condition_defaults_to_auto_language() {
        let prompt = NemotronPromptCondition::resolve(None, None).unwrap();

        assert_eq!(prompt.target_lang, "auto");
        assert!(prompt.strip_lang_tags);
        assert!(prompt.context_prompt.is_none());
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
    fn prompt_condition_preserves_non_language_prompt_as_context() {
        let prompt = NemotronPromptCondition::resolve(Some("fr-CA"), Some("medical dictation"))
            .unwrap();

        assert_eq!(prompt.target_lang, "fr-CA");
        assert_eq!(prompt.context_prompt.as_deref(), Some("medical dictation"));
    }

    #[test]
    fn prompt_condition_rejects_unknown_language() {
        let err = NemotronPromptCondition::resolve(Some("xx-YY"), None).unwrap_err();

        assert!(err.to_string().contains("Unsupported Nemotron target_lang"));
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

        assert_eq!(decode_vocab_tokens(&[0, 1, 2, 3, 4, 5], &vocab), "Hello, world!");
    }
}
