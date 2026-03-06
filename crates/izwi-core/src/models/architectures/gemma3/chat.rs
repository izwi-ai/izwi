//! Native Gemma 3 text-chat model loader and generation.

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::Path;
use std::sync::Mutex;

use candle_core::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::gemma3::{Config as Gemma3Config, Model as Gemma3Model};
use serde_json::Value;
use tracing::{info, warn};

use crate::backends::DeviceProfile;
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::shared::attention::flash::should_enable_flash_attention_v2;
use crate::models::shared::chat::{ChatMessage, ChatRole};
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone)]
pub struct ChatGenerationOutput {
    pub text: String,
    pub tokens_generated: usize,
}

#[derive(Debug, Clone)]
struct SpecialTokenIds {
    bos: Option<u32>,
    eos: u32,
    start_of_turn: u32,
    end_of_turn: u32,
}

struct GemmaTokenizer {
    inner: Tokenizer,
    vocab_size: usize,
    specials: SpecialTokenIds,
}

impl GemmaTokenizer {
    fn load(model_dir: &Path) -> Result<Self> {
        let inner = Tokenizer::from_path(model_dir)?;
        let vocab_size = inner.vocab_size();

        let token_id = |token: &str| -> Result<u32> {
            inner.token_to_id(token).ok_or_else(|| {
                Error::TokenizationError(format!("Missing Gemma special token: {token}"))
            })
        };

        let start_of_turn = token_id("<start_of_turn>")?;
        let end_of_turn = token_id("<end_of_turn>")?;
        let eos = inner.token_to_id("<eos>").unwrap_or(end_of_turn);
        let bos = inner.token_to_id("<bos>");

        Ok(Self {
            inner,
            vocab_size,
            specials: SpecialTokenIds {
                bos,
                eos,
                start_of_turn,
                end_of_turn,
            },
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
        let decoded = self.inner.decode(&filtered)?;
        Ok(strip_unused_placeholders(&decoded))
    }
}

struct GemmaDefaults {
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    head_dim: usize,
}

fn defaults_for_variant(variant: ModelVariant) -> GemmaDefaults {
    match variant {
        ModelVariant::Gemma31BIt => GemmaDefaults {
            hidden_size: 1152,
            intermediate_size: 6912,
            num_attention_heads: 4,
            num_hidden_layers: 26,
            num_key_value_heads: 1,
            head_dim: 256,
        },
        ModelVariant::Gemma34BIt => GemmaDefaults {
            hidden_size: 2560,
            intermediate_size: 10240,
            num_attention_heads: 8,
            num_hidden_layers: 34,
            num_key_value_heads: 4,
            head_dim: 256,
        },
        _ => GemmaDefaults {
            hidden_size: 2560,
            intermediate_size: 10240,
            num_attention_heads: 8,
            num_hidden_layers: 34,
            num_key_value_heads: 4,
            head_dim: 256,
        },
    }
}

fn parse_gemma3_config(
    config_str: &str,
    variant: ModelVariant,
    tokenizer_vocab_size: usize,
    checkpoint_vocab_size: Option<usize>,
) -> Result<Gemma3Config> {
    let root_value: Value = serde_json::from_str(config_str)?;
    let has_text_config = root_value.get("text_config").is_some();
    let source = root_value
        .get("text_config")
        .cloned()
        .unwrap_or_else(|| root_value.clone());

    let mut object: BTreeMap<String, Value> = source
        .as_object()
        .ok_or_else(|| Error::InvalidInput("Invalid Gemma config.json format".to_string()))?
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    let defaults = defaults_for_variant(variant);

    let mut set_default = |key: &str, value: Value| {
        if !object.contains_key(key) {
            object.insert(key.to_string(), value);
        }
    };

    set_default("attention_bias", Value::Bool(false));
    set_default(
        "hidden_activation",
        Value::String("gelu_pytorch_tanh".to_string()),
    );
    set_default("hidden_size", Value::from(defaults.hidden_size as u64));
    set_default(
        "intermediate_size",
        Value::from(defaults.intermediate_size as u64),
    );
    // MLX gemma3.py applies these multimodal defaults for Gemma 3 4B configs
    // when they are omitted from text_config.
    if has_text_config {
        set_default("num_attention_heads", Value::from(8u64));
        set_default("num_key_value_heads", Value::from(4u64));
    }

    set_default(
        "num_attention_heads",
        Value::from(defaults.num_attention_heads as u64),
    );
    set_default(
        "num_hidden_layers",
        Value::from(defaults.num_hidden_layers as u64),
    );
    set_default(
        "num_key_value_heads",
        Value::from(defaults.num_key_value_heads as u64),
    );
    set_default("head_dim", Value::from(defaults.head_dim as u64));
    set_default("rms_norm_eps", Value::from(1e-6f64));
    set_default("rope_theta", Value::from(1_000_000f64));
    set_default("rope_local_base_freq", Value::from(10_000f64));
    set_default("query_pre_attn_scalar", Value::from(256u64));
    set_default("sliding_window", Value::from(512u64));
    set_default("sliding_window_pattern", Value::from(6u64));
    set_default("max_position_embeddings", Value::from(32_768u64));
    let resolved_vocab_size = checkpoint_vocab_size.unwrap_or_else(|| {
        if has_text_config {
            262_208
        } else {
            tokenizer_vocab_size
        }
    });
    if let Some(config_vocab_size) = object.get("vocab_size").and_then(|value| value.as_u64()) {
        if config_vocab_size as usize != resolved_vocab_size {
            info!(
                "Overriding Gemma vocab_size from config {} to {}",
                config_vocab_size, resolved_vocab_size
            );
        }
    }
    object.insert(
        "vocab_size".to_string(),
        Value::from(resolved_vocab_size as u64),
    );

    let config =
        serde_json::from_value::<Gemma3Config>(Value::Object(object.into_iter().collect()))
            .map_err(Error::from)?;

    Ok(config)
}

fn should_use_language_model_prefix(config_str: &str) -> bool {
    let Ok(root) = serde_json::from_str::<Value>(config_str) else {
        return false;
    };

    if root
        .get("architectures")
        .and_then(|v| v.as_array())
        .is_some_and(|architectures| {
            architectures.iter().any(|entry| {
                entry
                    .as_str()
                    .is_some_and(|name| name == "Gemma3ForConditionalGeneration")
            })
        })
    {
        return true;
    }

    root.get("model_type")
        .and_then(|v| v.as_str())
        .is_some_and(|model_type| model_type == "gemma3")
}

const MAX_SAFE_TENSORS_HEADER_SIZE: usize = 100_000_000;

fn tensor_shape_from_safetensors_header(
    safetensors_path: &Path,
    tensor_name: &str,
) -> Result<Option<Vec<usize>>> {
    let mut file = fs::File::open(safetensors_path)?;

    let mut n_buf = [0u8; 8];
    file.read_exact(&mut n_buf)?;
    let header_len_u64 = u64::from_le_bytes(n_buf);
    let header_len: usize = header_len_u64
        .try_into()
        .map_err(|_| Error::InvalidInput("Invalid safetensors header length".to_string()))?;
    if header_len > MAX_SAFE_TENSORS_HEADER_SIZE {
        return Err(Error::InvalidInput(format!(
            "Safetensors header too large: {header_len}"
        )));
    }

    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)?;

    let metadata: Value = serde_json::from_slice(&header_buf)?;
    let tensor_entry = match metadata.get(tensor_name) {
        Some(entry) => entry,
        None => return Ok(None),
    };
    let shape = match tensor_entry.get("shape").and_then(|shape| shape.as_array()) {
        Some(shape) => shape,
        None => return Ok(None),
    };

    let dims = shape
        .iter()
        .map(|dim| dim.as_u64().map(|value| value as usize))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            Error::InvalidInput(format!(
                "Invalid shape metadata for tensor {tensor_name} in {}",
                safetensors_path.display()
            ))
        })?;

    Ok(Some(dims))
}

fn infer_embed_vocab_size_from_safetensors(
    safetensors_path: &Path,
    tensor_name: &str,
) -> Result<Option<usize>> {
    let shape = match tensor_shape_from_safetensors_header(safetensors_path, tensor_name)? {
        Some(shape) => shape,
        None => return Ok(None),
    };

    let Some(vocab_size) = shape.first().copied() else {
        return Ok(None);
    };

    Ok(Some(vocab_size))
}

pub struct Gemma3ChatModel {
    variant: ModelVariant,
    device: DeviceProfile,
    tokenizer: GemmaTokenizer,
    text_model: Mutex<Gemma3Model>,
}

impl Gemma3ChatModel {
    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        let tokenizer = GemmaTokenizer::load(model_dir)?;

        let config_path = model_dir.join("config.json");
        let config_str = fs::read_to_string(config_path)?;
        let mut use_language_model_prefix = should_use_language_model_prefix(&config_str);
        let mut inferred_vocab_size: Option<usize> = None;
        let dtype = if device.kind.is_metal() {
            // Prefer F16 for Gemma on Metal to keep memory use in check on larger checkpoints.
            DType::F16
        } else {
            device.select_dtype(None)
        };

        let index_path = model_dir.join("model.safetensors.index.json");
        let vb_base = if index_path.exists() {
            let index_data = fs::read_to_string(&index_path)?;
            let index: Value = serde_json::from_str(&index_data)?;
            let weight_map = index
                .get("weight_map")
                .and_then(|m| m.as_object())
                .ok_or_else(|| {
                    Error::InvalidInput("Invalid model.safetensors.index.json format".to_string())
                })?;
            if weight_map
                .keys()
                .any(|tensor_name| tensor_name.starts_with("language_model."))
            {
                use_language_model_prefix = true;
            }

            let embed_tensor_name = if use_language_model_prefix {
                "language_model.model.embed_tokens.weight"
            } else {
                "model.embed_tokens.weight"
            };
            if let Some(shard_name) = weight_map.get(embed_tensor_name).and_then(|v| v.as_str()) {
                let shard_path = model_dir.join(shard_name);
                inferred_vocab_size =
                    infer_embed_vocab_size_from_safetensors(&shard_path, embed_tensor_name)?;
            } else {
                let fallback_tensor_name =
                    if embed_tensor_name == "language_model.model.embed_tokens.weight" {
                        "model.embed_tokens.weight"
                    } else {
                        "language_model.model.embed_tokens.weight"
                    };
                if let Some(shard_name) = weight_map
                    .get(fallback_tensor_name)
                    .and_then(|v| v.as_str())
                {
                    let shard_path = model_dir.join(shard_name);
                    inferred_vocab_size =
                        infer_embed_vocab_size_from_safetensors(&shard_path, fallback_tensor_name)?;
                    if fallback_tensor_name.starts_with("language_model.") {
                        use_language_model_prefix = true;
                    }
                }
            }

            let mut shard_files: Vec<String> = weight_map
                .values()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            shard_files.sort();
            shard_files.dedup();

            let shard_paths: Vec<std::path::PathBuf> =
                shard_files.iter().map(|f| model_dir.join(f)).collect();
            unsafe { VarBuilder::from_mmaped_safetensors(&shard_paths, dtype, &device.device)? }
        } else {
            let weights_path = model_dir.join("model.safetensors");
            for tensor_name in [
                "model.embed_tokens.weight",
                "language_model.model.embed_tokens.weight",
            ] {
                if let Some(vocab_size) =
                    infer_embed_vocab_size_from_safetensors(&weights_path, tensor_name)?
                {
                    inferred_vocab_size = Some(vocab_size);
                    if tensor_name.starts_with("language_model.") {
                        use_language_model_prefix = true;
                    }
                    break;
                }
            }
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device.device)? }
        };

        if let Some(vocab_size) = inferred_vocab_size {
            info!(
                "Gemma {} using checkpoint embedding vocab size {}",
                variant.dir_name(),
                vocab_size
            );
        } else {
            warn!(
                "Could not infer Gemma {} embedding vocab size from checkpoint, falling back to tokenizer vocab {}",
                variant.dir_name(),
                tokenizer.vocab_size
            );
        }

        let config = parse_gemma3_config(
            &config_str,
            variant,
            tokenizer.vocab_size,
            inferred_vocab_size,
        )?;
        info!(
            "Gemma {} config resolved: layers={}, heads={}, kv_heads={}, head_dim={}, hidden_size={}, sliding_window={}, sliding_window_pattern={}, max_position_embeddings={}, dtype={:?}",
            variant.dir_name(),
            config.num_hidden_layers,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.hidden_size,
            config.sliding_window,
            config.sliding_window_pattern,
            config.max_position_embeddings,
            dtype
        );

        let vb = if use_language_model_prefix {
            vb_base.pp("language_model")
        } else {
            vb_base
        };

        let use_flash_attn = should_enable_flash_attention_v2(&device.device);
        let text_model = Gemma3Model::new(use_flash_attn, &config, vb).map_err(Error::from)?;

        info!(
            "Loaded Gemma chat model {} on {:?} (flash_attn={})",
            variant.dir_name(),
            device.kind,
            use_flash_attn
        );

        Ok(Self {
            variant,
            device,
            tokenizer,
            text_model: Mutex::new(text_model),
        })
    }

    pub fn generate(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
    ) -> Result<ChatGenerationOutput> {
        let mut no_op = |_delta: &str| {};
        self.generate_with_callback(messages, max_new_tokens, &mut no_op)
    }

    pub fn generate_with_callback(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<ChatGenerationOutput> {
        let prompt_ids = self.build_prompt(messages)?;
        if prompt_ids.is_empty() {
            return Err(Error::InvalidInput(
                "Gemma prompt produced no tokens".to_string(),
            ));
        }
        let mut input_ids = Tensor::from_vec(vec![prompt_ids[0]], (1, 1), &self.device.device)?;
        let mut seqlen_offset = 0usize;

        let mut generated_ids = Vec::new();
        let mut assembled = String::new();
        let mut stagnant_steps = 0usize;

        let mut model = self
            .text_model
            .lock()
            .map_err(|_| Error::InferenceError("Gemma model mutex poisoned".to_string()))?;
        model.clear_kv_cache();

        // Keep prefill token-by-token to avoid known Candle slice_set issues
        // with some Metal execution paths on Gemma.
        for &token in prompt_ids.iter().skip(1) {
            model
                .forward(&input_ids, seqlen_offset)
                .map_err(Error::from)?;
            seqlen_offset += 1;
            input_ids = Tensor::from_vec(vec![token], (1, 1), &self.device.device)?;
        }

        for _ in 0..max_new_tokens {
            let logits = model
                .forward(&input_ids, seqlen_offset)
                .map_err(Error::from)?;
            let next = select_next_token(&logits, self.tokenizer.vocab_size)?;

            if next == self.tokenizer.specials.end_of_turn
                || next == self.tokenizer.specials.eos
                || next == self.tokenizer.specials.start_of_turn
                || self.tokenizer.specials.bos.is_some_and(|bos| next == bos)
            {
                break;
            }

            generated_ids.push(next);

            let decoded = self.tokenizer.decode_text(&generated_ids)?;
            let delta = text_delta(&assembled, &decoded);
            for ch in delta.chars() {
                let mut buf = [0u8; 4];
                on_delta(ch.encode_utf8(&mut buf));
            }
            if decoded == assembled {
                stagnant_steps += 1;
                if stagnant_steps >= 4 {
                    break;
                }
            } else {
                stagnant_steps = 0;
                assembled = decoded;
            }

            seqlen_offset += 1;
            input_ids = Tensor::from_vec(vec![next], (1, 1), &self.device.device)?;
        }

        Ok(ChatGenerationOutput {
            text: assembled.trim().to_string(),
            tokens_generated: generated_ids.len(),
        })
    }

    pub fn prompt_token_ids(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        self.build_prompt(messages)
    }

    fn build_prompt(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        if messages.is_empty() {
            return Err(Error::InvalidInput(
                "Chat request must include at least one message".to_string(),
            ));
        }

        let mut system_parts = Vec::new();
        let mut turns: Vec<(ChatRole, String)> = Vec::new();

        for message in messages {
            let content = if matches!(message.role, ChatRole::Assistant) {
                strip_think_blocks(message.content.trim())
            } else {
                message.content.trim().to_string()
            };
            if content.is_empty() {
                continue;
            }

            if matches!(message.role, ChatRole::System) {
                system_parts.push(content);
            } else {
                turns.push((message.role.clone(), content));
            }
        }

        let system = if system_parts.is_empty() {
            "You are a helpful assistant.".to_string()
        } else {
            system_parts.join("\n\n")
        };

        if let Some((role, first_content)) = turns.first_mut() {
            if matches!(role, ChatRole::User) {
                *first_content = format!("{system}\n\n{first_content}");
            } else {
                turns.insert(0, (ChatRole::User, system));
            }
        } else {
            turns.push((ChatRole::User, system));
        }

        let mut ids = Vec::new();
        if let Some(bos) = self.tokenizer.specials.bos {
            ids.push(bos);
        }

        for (role, content) in &turns {
            let role_name = match role {
                ChatRole::Assistant => "model",
                ChatRole::User | ChatRole::System => "user",
            };

            ids.push(self.tokenizer.specials.start_of_turn);
            ids.extend(self.tokenizer.encode_text(role_name)?);
            ids.extend(self.tokenizer.encode_text("\n")?);
            ids.extend(self.tokenizer.encode_text(content)?);
            ids.push(self.tokenizer.specials.end_of_turn);
            ids.extend(self.tokenizer.encode_text("\n")?);
        }

        ids.push(self.tokenizer.specials.start_of_turn);
        ids.extend(self.tokenizer.encode_text("model")?);
        ids.extend(self.tokenizer.encode_text("\n")?);

        Ok(ids)
    }

    pub fn variant(&self) -> ModelVariant {
        self.variant
    }
}

fn strip_think_blocks(input: &str) -> String {
    let mut output = input.to_string();
    let open = "<think>";
    let close = "</think>";

    loop {
        let Some(start) = output.find(open) else {
            break;
        };

        let search_from = start + open.len();
        if let Some(end_rel) = output[search_from..].find(close) {
            let end = search_from + end_rel + close.len();
            output.replace_range(start..end, "");
            continue;
        }

        output.truncate(start);
        break;
    }

    output.trim().to_string()
}

fn strip_unused_placeholders(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut cursor = 0usize;

    while cursor < input.len() {
        let tail = &input[cursor..];
        if let Some(after_prefix) = tail.strip_prefix("<unused") {
            let mut digit_len = 0usize;
            for ch in after_prefix.chars() {
                if ch.is_ascii_digit() {
                    digit_len += ch.len_utf8();
                } else {
                    break;
                }
            }

            if digit_len > 0 {
                let rest = &after_prefix[digit_len..];
                if let Some(after_marker) = rest.strip_prefix('>') {
                    let consumed = input.len() - after_marker.len() - cursor;
                    cursor += consumed;
                    continue;
                }
            }
        }

        let ch = tail.chars().next().unwrap_or_default();
        out.push(ch);
        cursor += ch.len_utf8();
    }

    out
}

fn argmax(logits: &Tensor, vocab_limit: usize) -> Result<u32> {
    let values = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let capped = vocab_limit.min(values.len());
    if capped == 0 {
        return Err(Error::InferenceError(
            "No valid logits in constrained vocabulary".to_string(),
        ));
    }
    let (idx, _) = values
        .iter()
        .take(capped)
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| Error::InferenceError("Empty logits".to_string()))?;
    Ok(idx as u32)
}

fn select_next_token(logits: &Tensor, vocab_limit: usize) -> Result<u32> {
    match logits.rank() {
        // [vocab]
        1 => argmax(logits, vocab_limit),
        // [seq, vocab]
        2 => {
            let seq_len = logits.dim(0)?;
            argmax(&logits.i(seq_len.saturating_sub(1))?, vocab_limit)
        }
        // [batch, seq, vocab]
        3 => {
            let seq_len = logits.dim(1)?;
            argmax(&logits.i((0, seq_len.saturating_sub(1)))?, vocab_limit)
        }
        _ => Err(Error::InferenceError(format!(
            "Unexpected Gemma logits rank: {} with dims {:?}",
            logits.rank(),
            logits.dims()
        ))),
    }
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

#[cfg(test)]
mod tests {
    use super::strip_unused_placeholders;

    #[test]
    fn strip_unused_placeholders_removes_marker_tokens() {
        let input = "<unused6421> hello <unused9>world<unused123>";
        let output = strip_unused_placeholders(input);
        assert_eq!(output, " hello world");
    }

    #[test]
    fn strip_unused_placeholders_keeps_normal_angle_bracket_text() {
        let input = "a <unused> b <unusedx12> c";
        let output = strip_unused_placeholders(input);
        assert_eq!(output, input);
    }
}
