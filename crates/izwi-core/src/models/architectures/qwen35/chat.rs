//! Native Qwen3.5 chat loader scaffolding.
//!
//! Phase 2 intentionally stops at asset/config loading. The hybrid text decoder
//! and multimodal execution land in later commits.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use candle_core::quantized::gguf_file::Value as GgufValue;
use serde::Deserialize;
use tracing::info;

use crate::backends::{BackendKind, DeviceProfile};
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::shared::chat::{ChatMessage, ChatRole};
use crate::models::shared::weights::gguf::{GgufLoader, GgufModelInfo};
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone)]
pub struct ChatGenerationOutput {
    pub text: String,
    pub tokens_generated: usize,
}

#[derive(Debug, Clone)]
pub struct Qwen35TextConfig {
    pub architecture: String,
    pub block_count: usize,
    pub context_length: usize,
    pub embedding_length: usize,
    pub feed_forward_length: usize,
    pub attention_head_count: usize,
    pub attention_head_count_kv: usize,
    pub attention_key_length: usize,
    pub attention_value_length: usize,
    pub rope_dimension_sections: Vec<usize>,
    pub rope_dimension_count: usize,
    pub rope_freq_base: f64,
    pub attention_layer_norm_rms_epsilon: f64,
    pub ssm_conv_kernel: usize,
    pub ssm_state_size: usize,
    pub ssm_group_count: usize,
    pub ssm_time_step_rank: usize,
    pub ssm_inner_size: usize,
    pub full_attention_interval: usize,
}

#[derive(Debug, Clone)]
struct SpecialTokenIds {
    _im_start: u32,
    _im_end: u32,
    _eos: u32,
}

#[derive(Debug, Deserialize)]
struct TokenizerConfigFile {
    #[serde(default)]
    added_tokens_decoder: HashMap<String, AddedToken>,
    #[serde(default)]
    bos_token: Option<String>,
    #[serde(default)]
    eos_token: Option<String>,
    chat_template: String,
}

#[derive(Debug, Deserialize)]
struct AddedToken {
    content: String,
}

struct Qwen35Tokenizer {
    inner: Tokenizer,
    _vocab_size: usize,
    _specials: SpecialTokenIds,
    chat_template: String,
    default_enable_thinking: bool,
    _bos_token: Option<String>,
}

impl Qwen35Tokenizer {
    fn load(model_dir: &Path, variant: ModelVariant) -> Result<Self> {
        let inner = Tokenizer::from_path(model_dir)?;
        let vocab_size = inner.vocab_size();

        let config_path = model_dir.join("tokenizer_config.json");
        let config_str = fs::read_to_string(config_path)?;
        let config: TokenizerConfigFile = serde_json::from_str(&config_str)?;

        let id_for = |token: &str| -> Option<u32> {
            config.added_tokens_decoder.iter().find_map(|(id, entry)| {
                if entry.content == token {
                    id.parse().ok()
                } else {
                    None
                }
            })
        };

        let im_start = id_for("<|im_start|>")
            .ok_or_else(|| Error::TokenizationError("Missing <|im_start|> token id".to_string()))?;
        let im_end = id_for("<|im_end|>")
            .ok_or_else(|| Error::TokenizationError("Missing <|im_end|> token id".to_string()))?;
        let eos = config
            .eos_token
            .as_deref()
            .and_then(id_for)
            .unwrap_or(im_end);
        let default_enable_thinking =
            resolve_default_enable_thinking(&config.chat_template, variant);

        Ok(Self {
            inner,
            _vocab_size: vocab_size,
            _specials: SpecialTokenIds {
                _im_start: im_start,
                _im_end: im_end,
                _eos: eos,
            },
            chat_template: config.chat_template,
            default_enable_thinking,
            _bos_token: config.bos_token,
        })
    }

    fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        self.inner.encode(text)
    }
}

pub struct Qwen35ChatModel {
    device: DeviceProfile,
    variant: ModelVariant,
    tokenizer: Qwen35Tokenizer,
    text_config: Qwen35TextConfig,
    text_checkpoint: GgufModelInfo,
    projector_checkpoint: GgufModelInfo,
}

impl Qwen35ChatModel {
    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        let gguf_path = model_dir.join(qwen35_gguf_filename(variant)?);
        let mmproj_path = model_dir.join("mmproj-F16.gguf");

        if !gguf_path.exists() {
            return Err(Error::ModelLoadError(format!(
                "Qwen3.5 GGUF checkpoint not found: {}",
                gguf_path.display()
            )));
        }
        if !mmproj_path.exists() {
            return Err(Error::ModelLoadError(format!(
                "Qwen3.5 projector checkpoint not found: {}",
                mmproj_path.display()
            )));
        }

        let backend = BackendKind::from(device.kind);
        let text_loader = GgufLoader::from_path_with_backend(&gguf_path, backend)?;
        let text_checkpoint = text_loader.get_model_info();
        let architecture = text_checkpoint
            .architecture
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        if architecture != "qwen35" {
            return Err(Error::ModelLoadError(format!(
                "Expected general.architecture=qwen35 for {}, found {}",
                gguf_path.display(),
                architecture
            )));
        }

        let text_config = parse_text_config(&text_loader)?;
        let tokenizer = Qwen35Tokenizer::load(model_dir, variant)?;

        let projector_loader = GgufLoader::from_path_with_backend(&mmproj_path, backend)?;
        let projector_checkpoint = projector_loader.get_model_info();

        info!(
            "Loaded Qwen3.5 chat assets for {} on {:?} ({} text tensors, {} projector tensors)",
            variant.display_name(),
            device.kind,
            text_loader.tensor_count(),
            projector_loader.tensor_count()
        );

        Ok(Self {
            device,
            variant,
            tokenizer,
            text_config,
            text_checkpoint,
            projector_checkpoint,
        })
    }

    pub fn variant(&self) -> ModelVariant {
        self.variant
    }

    pub fn text_config(&self) -> &Qwen35TextConfig {
        &self.text_config
    }

    pub fn chat_template(&self) -> &str {
        &self.tokenizer.chat_template
    }

    pub fn default_enable_thinking(&self) -> bool {
        self.tokenizer.default_enable_thinking
    }

    pub fn text_checkpoint(&self) -> &GgufModelInfo {
        &self.text_checkpoint
    }

    pub fn projector_checkpoint(&self) -> &GgufModelInfo {
        &self.projector_checkpoint
    }

    pub fn prompt_token_ids(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        let prompt = render_fallback_prompt(messages);
        self.tokenizer.encode_text(&prompt)
    }

    pub fn generate(
        &self,
        _messages: &[ChatMessage],
        _max_new_tokens: usize,
    ) -> Result<ChatGenerationOutput> {
        Err(not_ready_generation_error())
    }

    pub fn generate_with_callback(
        &self,
        _messages: &[ChatMessage],
        _max_new_tokens: usize,
        _on_delta: &mut dyn FnMut(&str),
    ) -> Result<ChatGenerationOutput> {
        Err(not_ready_generation_error())
    }

    pub fn supports_incremental_decode(&self) -> bool {
        false
    }

    pub fn device_kind(&self) -> BackendKind {
        BackendKind::from(self.device.kind)
    }
}

fn render_fallback_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for message in messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(match message.role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        });
        prompt.push('\n');
        prompt.push_str(&message.content);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

fn not_ready_generation_error() -> Error {
    Error::InvalidInput(
        "Qwen3.5 asset loading is available, but generation lands in a later phase".to_string(),
    )
}

fn qwen35_gguf_filename(variant: ModelVariant) -> Result<&'static str> {
    match variant {
        ModelVariant::Qwen3508BGguf => Ok("Qwen3.5-0.8B-Q4_K_M.gguf"),
        ModelVariant::Qwen352BGguf => Ok("Qwen3.5-2B-Q4_K_M.gguf"),
        ModelVariant::Qwen354BGguf => Ok("Qwen3.5-4B-Q4_K_M.gguf"),
        ModelVariant::Qwen359BGguf => Ok("Qwen3.5-9B-Q4_K_M.gguf"),
        _ => Err(Error::ModelLoadError(format!(
            "Unsupported Qwen3.5 GGUF chat variant: {variant}"
        ))),
    }
}

fn resolve_default_enable_thinking(chat_template: &str, variant: ModelVariant) -> bool {
    if chat_template.contains("enable_thinking is defined and enable_thinking is false") {
        true
    } else if chat_template.contains("enable_thinking is defined and enable_thinking is true") {
        false
    } else {
        matches!(variant, ModelVariant::Qwen354BGguf | ModelVariant::Qwen359BGguf)
    }
}

fn parse_text_config(loader: &GgufLoader) -> Result<Qwen35TextConfig> {
    Ok(Qwen35TextConfig {
        architecture: loader
            .get_metadata_string("general.architecture")
            .unwrap_or_else(|| "qwen35".to_string()),
        block_count: required_usize(loader, "qwen35.block_count")?,
        context_length: required_usize(loader, "qwen35.context_length")?,
        embedding_length: required_usize(loader, "qwen35.embedding_length")?,
        feed_forward_length: required_usize(loader, "qwen35.feed_forward_length")?,
        attention_head_count: required_usize(loader, "qwen35.attention.head_count")?,
        attention_head_count_kv: required_usize(loader, "qwen35.attention.head_count_kv")?,
        attention_key_length: required_usize(loader, "qwen35.attention.key_length")?,
        attention_value_length: required_usize(loader, "qwen35.attention.value_length")?,
        rope_dimension_sections: required_usize_array(loader, "qwen35.rope.dimension_sections")?,
        rope_dimension_count: required_usize(loader, "qwen35.rope.dimension_count")?,
        rope_freq_base: required_f64(loader, "qwen35.rope.freq_base")?,
        attention_layer_norm_rms_epsilon: required_f64(
            loader,
            "qwen35.attention.layer_norm_rms_epsilon",
        )?,
        ssm_conv_kernel: required_usize(loader, "qwen35.ssm.conv_kernel")?,
        ssm_state_size: required_usize(loader, "qwen35.ssm.state_size")?,
        ssm_group_count: required_usize(loader, "qwen35.ssm.group_count")?,
        ssm_time_step_rank: required_usize(loader, "qwen35.ssm.time_step_rank")?,
        ssm_inner_size: required_usize(loader, "qwen35.ssm.inner_size")?,
        full_attention_interval: required_usize(loader, "qwen35.full_attention_interval")?,
    })
}

fn required_usize(loader: &GgufLoader, key: &str) -> Result<usize> {
    loader
        .get_metadata_u64(key)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| Error::ModelLoadError(format!("Missing or invalid GGUF metadata: {key}")))
}

fn required_f64(loader: &GgufLoader, key: &str) -> Result<f64> {
    let value = loader
        .metadata_value(key)
        .and_then(gguf_to_f64)
        .ok_or_else(|| Error::ModelLoadError(format!("Missing or invalid GGUF metadata: {key}")))?;
    Ok(value)
}

fn required_usize_array(loader: &GgufLoader, key: &str) -> Result<Vec<usize>> {
    let value = loader.metadata_value(key).ok_or_else(|| {
        Error::ModelLoadError(format!("Missing or invalid GGUF metadata: {key}"))
    })?;
    let GgufValue::Array(items) = value else {
        return Err(Error::ModelLoadError(format!(
            "Expected GGUF array metadata for {key}"
        )));
    };

    let mut values = Vec::with_capacity(items.len());
    for item in items {
        let Some(raw) = gguf_to_u64(item) else {
            return Err(Error::ModelLoadError(format!(
                "Expected integer array values for {key}"
            )));
        };
        let value = usize::try_from(raw).map_err(|_| {
            Error::ModelLoadError(format!("Array value out of range for {key}: {raw}"))
        })?;
        values.push(value);
    }
    Ok(values)
}

fn gguf_to_u64(value: &GgufValue) -> Option<u64> {
    match value {
        GgufValue::U64(n) => Some(*n),
        GgufValue::I64(n) => Some(*n as u64),
        GgufValue::U32(n) => Some(*n as u64),
        GgufValue::I32(n) => Some(*n as u64),
        GgufValue::U16(n) => Some(*n as u64),
        GgufValue::I16(n) => Some(*n as u64),
        GgufValue::U8(n) => Some(*n as u64),
        GgufValue::I8(n) => Some(*n as u64),
        _ => None,
    }
}

fn gguf_to_f64(value: &GgufValue) -> Option<f64> {
    match value {
        GgufValue::F64(n) => Some(*n),
        GgufValue::F32(n) => Some(*n as f64),
        GgufValue::U64(n) => Some(*n as f64),
        GgufValue::I64(n) => Some(*n as f64),
        GgufValue::U32(n) => Some(*n as f64),
        GgufValue::I32(n) => Some(*n as f64),
        GgufValue::U16(n) => Some(*n as f64),
        GgufValue::I16(n) => Some(*n as f64),
        GgufValue::U8(n) => Some(*n as f64),
        GgufValue::I8(n) => Some(*n as f64),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::DeviceProfile;
    use std::path::PathBuf;

    fn local_model_dir(name: &str) -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        PathBuf::from(home)
            .join("Library/Application Support/izwi/models")
            .join(name)
    }

    #[test]
    fn resolve_default_thinking_matches_variant_templates() {
        let small = "{%- if add_generation_prompt %}{%- if enable_thinking is defined and enable_thinking is true %}<think>\n{%- endif %}{%- endif %}";
        let large = "{%- if add_generation_prompt %}{%- if enable_thinking is defined and enable_thinking is false %}{%- else %}<think>\n{%- endif %}{%- endif %}";

        assert!(!resolve_default_enable_thinking(
            small,
            ModelVariant::Qwen3508BGguf
        ));
        assert!(resolve_default_enable_thinking(
            large,
            ModelVariant::Qwen354BGguf
        ));
    }

    #[test]
    fn load_local_qwen35_assets_smoke_if_available() {
        let model_dir = local_model_dir("Qwen3.5-4B");
        if !model_dir.exists() {
            return;
        }

        let model =
            Qwen35ChatModel::load(&model_dir, ModelVariant::Qwen354BGguf, DeviceProfile::cpu())
                .expect("qwen3.5 assets should load");

        assert_eq!(model.variant(), ModelVariant::Qwen354BGguf);
        assert_eq!(model.text_config().architecture, "qwen35");
        assert_eq!(model.text_config().full_attention_interval, 4);
        assert!(model.default_enable_thinking());
        assert_eq!(model.text_checkpoint().architecture.as_deref(), Some("qwen35"));
        assert!(model.projector_checkpoint().path.ends_with("mmproj-F16.gguf"));
    }
}
