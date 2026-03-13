//! Native Qwen3.5 chat model loader and text generation.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use candle_core::quantized::gguf_file::Value as GgufValue;
use candle_core::{DType, IndexOp, Tensor, D};
use serde::Deserialize;
use tracing::info;

use crate::backends::{BackendKind, DeviceProfile};
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::shared::chat::{ChatGenerationConfig, ChatMessage, ChatRole};
use crate::models::shared::weights::gguf::{GgufLoader, GgufModelInfo};
use crate::tokenizer::Tokenizer;

use super::text::Qwen35TextModel;

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
    im_start: u32,
    im_end: u32,
    eos: u32,
    eos_alt: Option<u32>,
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
    vocab_size: usize,
    specials: SpecialTokenIds,
    chat_template: String,
    default_enable_thinking: bool,
    bos_token: Option<String>,
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
        let eos_alt = id_for("<|endoftext|>");
        let default_enable_thinking =
            resolve_default_enable_thinking(&config.chat_template, variant);

        Ok(Self {
            inner,
            vocab_size,
            specials: SpecialTokenIds {
                im_start,
                im_end,
                eos,
                eos_alt,
            },
            chat_template: config.chat_template,
            default_enable_thinking,
            bos_token: config.bos_token,
        })
    }

    fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        self.inner.encode(text)
    }

    fn decode_text(&self, ids: &[u32]) -> Result<String> {
        let filtered: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|id| (*id as usize) < self.vocab_size)
            .collect();
        self.inner.decode(&filtered)
    }
}

pub struct Qwen35ChatModel {
    device_kind: BackendKind,
    variant: ModelVariant,
    tokenizer: Qwen35Tokenizer,
    text_config: Qwen35TextConfig,
    text_checkpoint: GgufModelInfo,
    projector_checkpoint: GgufModelInfo,
    text_model: Qwen35TextModel,
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
        let text_model = Qwen35TextModel::load(&text_loader, &text_config, &device.device)?;

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
            device_kind: backend,
            variant,
            tokenizer,
            text_config,
            text_checkpoint,
            projector_checkpoint,
            text_model,
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
        messages: &[ChatMessage],
        max_new_tokens: usize,
    ) -> Result<ChatGenerationOutput> {
        let config = ChatGenerationConfig::default();
        self.generate_with_config(messages, max_new_tokens, &config)
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

    pub fn generate_with_config(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
    ) -> Result<ChatGenerationOutput> {
        let mut no_op = |_delta: &str| {};
        self.generate_with_callback_and_config(messages, max_new_tokens, config, &mut no_op)
    }

    pub fn generate_with_callback_and_config(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<ChatGenerationOutput> {
        let prompt_ids = self.prompt_token_ids(messages)?;
        if prompt_ids.is_empty() {
            return Err(Error::InvalidInput(
                "Chat request must include at least one tokenizable message".to_string(),
            ));
        }

        let mut state = self.text_model.new_state();
        let mut logits: Option<Tensor> = None;
        for token_id in prompt_ids {
            logits = Some(self.text_model.forward_token_id(token_id, &mut state)?);
        }
        let mut logits = logits.ok_or_else(|| {
            Error::InferenceError("Qwen3.5 prompt prefill did not produce logits".to_string())
        })?;

        let max_new_tokens = max_new_tokens.max(1);
        let mut generated_ids = Vec::new();
        let mut assembled = String::new();
        let mut rng = SimpleRng::new(config.seed);

        while generated_ids.len() < max_new_tokens {
            let next = sample_next_token(&logits, config, &generated_ids, &mut rng)?;
            if self.is_stop_token(next, config) {
                break;
            }

            generated_ids.push(next);
            let decoded = self.tokenizer.decode_text(&generated_ids)?;
            let delta = text_delta(&assembled, &decoded);
            if !delta.is_empty() {
                on_delta(&delta);
            }
            assembled = decoded;
            logits = self.text_model.forward_token_id(next, &mut state)?;
        }

        Ok(ChatGenerationOutput {
            text: assembled.trim().to_string(),
            tokens_generated: generated_ids.len(),
        })
    }

    pub fn supports_incremental_decode(&self) -> bool {
        false
    }

    pub fn device_kind(&self) -> BackendKind {
        self.device_kind
    }

    fn is_stop_token(&self, token_id: u32, config: &ChatGenerationConfig) -> bool {
        token_id == self.tokenizer.specials.im_end
            || token_id == self.tokenizer.specials.eos
            || self.tokenizer.specials.eos_alt == Some(token_id)
            || config.stop_token_ids.contains(&token_id)
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
        matches!(
            variant,
            ModelVariant::Qwen354BGguf | ModelVariant::Qwen359BGguf
        )
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
    let value = loader
        .metadata_value(key)
        .ok_or_else(|| Error::ModelLoadError(format!("Missing or invalid GGUF metadata: {key}")))?;
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

fn sample_next_token(
    logits: &Tensor,
    config: &ChatGenerationConfig,
    history: &[u32],
    rng: &mut SimpleRng,
) -> Result<u32> {
    let mut values = logits_to_vec(logits)?;

    if config.repetition_penalty > 1.0 && !history.is_empty() {
        let mut seen = vec![false; values.len()];
        for &token in history {
            let idx = token as usize;
            if idx < seen.len() {
                seen[idx] = true;
            }
        }

        for (idx, seen_flag) in seen.iter().enumerate() {
            if !*seen_flag {
                continue;
            }
            let value = &mut values[idx];
            if !value.is_finite() {
                continue;
            }
            if *value > 0.0 {
                *value /= config.repetition_penalty;
            } else {
                *value *= config.repetition_penalty;
            }
        }
    }

    if config.presence_penalty.abs() > f32::EPSILON && !history.is_empty() {
        let mut seen = vec![false; values.len()];
        for &token in history {
            let idx = token as usize;
            if idx < seen.len() {
                seen[idx] = true;
            }
        }

        for (idx, seen_flag) in seen.iter().enumerate() {
            if *seen_flag && values[idx].is_finite() {
                values[idx] -= config.presence_penalty;
            }
        }
    }

    if config.temperature <= 1e-5 {
        return argmax_values(&values);
    }

    let temperature = config.temperature.max(1e-5);
    for value in &mut values {
        if value.is_finite() {
            *value /= temperature;
        }
    }

    let mut candidates: Vec<usize> = values
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| value.is_finite().then_some(idx))
        .collect();
    if candidates.is_empty() {
        return argmax_values(&values);
    }

    if config.top_k > 0 && config.top_k < candidates.len() {
        candidates.sort_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap_or(Ordering::Equal));
        candidates.truncate(config.top_k);
    }

    let max_logit = candidates
        .iter()
        .map(|&idx| values[idx])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = candidates
        .iter()
        .map(|&idx| (idx, (values[idx] - max_logit).exp()))
        .collect();

    let mut sum: f32 = probs.iter().map(|(_, prob)| *prob).sum();
    if !sum.is_finite() || sum <= 0.0 {
        return argmax_values(&values);
    }
    for (_, prob) in &mut probs {
        *prob /= sum;
    }

    if config.top_p < 1.0 {
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        let cutoff = config.top_p.max(1e-6);
        let mut cumulative = 0.0f32;
        let mut keep = 0usize;
        for (_, prob) in &probs {
            cumulative += *prob;
            keep += 1;
            if cumulative >= cutoff {
                break;
            }
        }
        probs.truncate(keep.max(1));
        sum = probs.iter().map(|(_, prob)| *prob).sum();
        if sum > 0.0 {
            for (_, prob) in &mut probs {
                *prob /= sum;
            }
        }
    }

    let sample = rng.next_f32();
    let mut cumulative = 0.0f32;
    for (idx, prob) in &probs {
        cumulative += *prob;
        if sample <= cumulative {
            return Ok(*idx as u32);
        }
    }

    probs
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
        .map(|(idx, _)| *idx as u32)
        .ok_or_else(|| Error::InferenceError("Failed to sample Qwen3.5 token".to_string()))
}

fn logits_to_vec(logits: &Tensor) -> Result<Vec<f32>> {
    let logits = match logits.rank() {
        1 => logits.clone(),
        2 => {
            let (rows, _cols) = logits.dims2()?;
            if rows != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected Qwen3.5 logits shape for sampling: {:?}",
                    logits.shape().dims()
                )));
            }
            logits.i(0)?
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected Qwen3.5 logits rank for sampling: {rank}"
            )))
        }
    };

    logits
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()
        .map_err(Error::from)
}

fn argmax_values(values: &[f32]) -> Result<u32> {
    let mut max_idx = None;
    let mut max_value = f32::NEG_INFINITY;

    for (idx, value) in values.iter().enumerate() {
        if value.is_finite() && *value > max_value {
            max_value = *value;
            max_idx = Some(idx);
        }
    }

    max_idx
        .map(|idx| idx as u32)
        .ok_or_else(|| Error::InferenceError("No valid Qwen3.5 logits to sample".to_string()))
}

fn argmax(logits: &Tensor) -> Result<u32> {
    let logits = match logits.rank() {
        1 => logits.clone(),
        2 => {
            let (rows, _cols) = logits.dims2()?;
            if rows != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected Qwen3.5 logits shape for argmax: {:?}",
                    logits.shape().dims()
                )));
            }
            logits.i(0)?
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected Qwen3.5 logits rank for argmax: {rank}"
            )))
        }
    };

    let idx = logits.argmax(D::Minus1)?;
    let idx = if idx.rank() == 0 {
        idx
    } else {
        idx.squeeze(0)?
    };
    idx.to_dtype(DType::U32)?
        .to_scalar::<u32>()
        .map_err(Error::from)
}

fn text_delta(previous: &str, current: &str) -> String {
    if let Some(delta) = current.strip_prefix(previous) {
        return delta.to_string();
    }

    let common = previous
        .chars()
        .zip(current.chars())
        .take_while(|(left, right)| left == right)
        .count();
    current.chars().skip(common).collect()
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        let seed = if seed == 0 {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos() as u64)
                .unwrap_or(0x9E37_79B9_7F4A_7C15)
        } else {
            seed
        };
        Self {
            state: seed ^ 0xA076_1D64_78BD_642F,
        }
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f64 / (u32::MAX as f64 + 1.0)) as f32
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
        assert_eq!(
            model.text_checkpoint().architecture.as_deref(),
            Some("qwen35")
        );
        assert!(model
            .projector_checkpoint()
            .path
            .ends_with("mmproj-F16.gguf"));
    }

    #[test]
    fn generate_local_qwen35_text_smoke_if_available() {
        let model_dir = local_model_dir("Qwen3.5-0.8B");
        if !model_dir.exists() {
            return;
        }

        let model = Qwen35ChatModel::load(
            &model_dir,
            ModelVariant::Qwen3508BGguf,
            DeviceProfile::cpu(),
        )
        .expect("qwen3.5 assets should load");
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            content: "Reply with one short word.".to_string(),
        }];
        let config = ChatGenerationConfig {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            stop_token_ids: Vec::new(),
            seed: 7,
        };

        let output = model
            .generate_with_config(&messages, 4, &config)
            .expect("qwen3.5 text generation should run");

        assert!(output.tokens_generated <= 4);
    }
}
