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
    Vocab(Vec<String>),
}

impl NemotronDecoder {
    fn load(artifacts: &NemotronArtifacts) -> Result<Self> {
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
            Self::Vocab(vocab) => decode_vocab_tokens(ids, vocab),
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            Self::HfTokenizer(tokenizer) => tokenizer.vocab_size(),
            Self::Vocab(vocab) => vocab.len(),
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
}

impl NemotronRuntimePlan {
    fn from_inventory(inventory: &NemotronConfigInventory) -> Result<Self> {
        let sample_rate = inventory.sample_rate.unwrap_or(SAMPLE_RATE as usize);
        if sample_rate != SAMPLE_RATE as usize {
            return Err(Error::ModelLoadError(format!(
                "Nemotron config advertises sample_rate={sample_rate}, expected {SAMPLE_RATE}"
            )));
        }

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
            default_streaming_profile: NemotronStreamingProfile::from_inventory(inventory)?,
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
    fn from_inventory(inventory: &NemotronConfigInventory) -> Result<Self> {
        let left_context_frames = inventory.left_context_frames.unwrap_or(56);
        let right_context_frames = inventory
            .right_context_frames
            .iter()
            .copied()
            .max()
            .unwrap_or(13);
        Self::new(left_context_frames, right_context_frames)
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
            chunk_ms: chunk_frames * 80,
        })
    }

    fn diagnostics(&self) -> serde_json::Value {
        json!({
            "att_context_size": [self.left_context_frames, self.right_context_frames],
            "chunk_frames": self.chunk_frames,
            "chunk_ms": self.chunk_ms,
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
        let decoder = NemotronDecoder::load(&artifacts)?;
        if let Some(expected) = runtime_plan.vocab_size {
            let actual = decoder.vocab_size();
            if actual < expected {
                return Err(Error::ModelLoadError(format!(
                    "Nemotron tokenizer vocabulary is smaller than config: tokenizer={actual}, config={expected}"
                )));
            }
        }

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

#[cfg(test)]
mod tests {
    use super::*;

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
