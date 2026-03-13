//! Native Qwen3 text-chat model loader and generation.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Mutex;

use candle_core::quantized::gguf_file;
use candle_core::{DType, IndexOp, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::quantized_qwen3::ModelWeights as QuantizedQwen3Model;
use serde::Deserialize;
use serde_json::Value;
use tracing::info;

use crate::backends::DeviceProfile;
use crate::backends::{open_gguf_reader, BackendKind};
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::architectures::qwen3::core::{Qwen3Cache, Qwen3Config, Qwen3Model};
use crate::models::shared::chat::{ChatMessage, ChatRole};
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone)]
pub struct ChatGenerationOutput {
    pub text: String,
    pub tokens_generated: usize,
}

pub struct ChatDecodeState {
    cache: Qwen3Cache,
    embeds: Tensor,
    pos: usize,
    generated_ids: Vec<u32>,
    assembled: String,
    max_new_tokens: usize,
    finished: bool,
}

#[derive(Debug, Clone)]
pub struct ChatDecodeStep {
    pub delta: String,
    pub text: String,
    pub tokens_generated: usize,
    pub finished: bool,
}

#[derive(Debug, Clone)]
struct SpecialTokenIds {
    im_start: u32,
    im_end: u32,
    eos: u32,
    eos_alt: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    #[serde(default)]
    added_tokens_decoder: HashMap<String, AddedToken>,
    #[serde(default)]
    eos_token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AddedToken {
    content: String,
}

struct ChatTokenizer {
    inner: Tokenizer,
    vocab_size: usize,
    specials: SpecialTokenIds,
}

impl ChatTokenizer {
    fn load(model_dir: &Path, expected_vocab_size: Option<usize>) -> Result<Self> {
        let inner = Tokenizer::from_path_with_expected_vocab(model_dir, expected_vocab_size)?;
        let vocab_size = inner.vocab_size();

        let config_path = model_dir.join("tokenizer_config.json");
        let config_str = fs::read_to_string(config_path)?;
        let config: TokenizerConfig = serde_json::from_str(&config_str)?;

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

        Ok(Self {
            inner,
            vocab_size,
            specials: SpecialTokenIds {
                im_start,
                im_end,
                eos,
                eos_alt,
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
        self.inner.decode(&filtered)
    }
}

enum Qwen3ChatBackend {
    Native {
        text_model: Qwen3Model,
    },
    Gguf {
        text_model: Mutex<QuantizedQwen3Model>,
        gguf_file: String,
    },
}

pub struct Qwen3ChatModel {
    device: DeviceProfile,
    tokenizer: ChatTokenizer,
    backend: Qwen3ChatBackend,
}

impl Qwen3ChatModel {
    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        if variant.is_qwen_chat_gguf() {
            return Self::load_gguf(model_dir, variant, device);
        }
        Self::load_safetensors(model_dir, device)
    }

    fn load_safetensors(model_dir: &Path, device: DeviceProfile) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_str = fs::read_to_string(config_path)?;
        let config = parse_qwen3_config(&config_str)?;

        let tokenizer = ChatTokenizer::load(model_dir, Some(config.vocab_size))?;
        let dtype_override = std::env::var("IZWI_CHAT_DTYPE")
            .ok()
            .or_else(|| std::env::var("IZWI_QWEN_DTYPE").ok());
        let dtype = match dtype_override.as_deref().map(str::trim) {
            Some(raw) if !raw.is_empty() => device.select_dtype(Some(raw)),
            _ if device.kind.is_metal() => {
                // Keep chat memory/latency practical on Apple Silicon.
                DType::F16
            }
            _ => device.select_dtype(None),
        };

        let index_path = model_dir.join("model.safetensors.index.json");
        let vb = if index_path.exists() {
            let index_data = fs::read_to_string(&index_path)?;
            let index: Value = serde_json::from_str(&index_data)?;
            let weight_map = index
                .get("weight_map")
                .and_then(|m| m.as_object())
                .ok_or_else(|| {
                    Error::InvalidInput("Invalid model.safetensors.index.json format".to_string())
                })?;

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
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device.device)? }
        };

        let text_model = Qwen3Model::load(config, vb)?;

        info!(
            "Loaded Qwen3 chat model on {:?} with dtype {:?}",
            device.kind, dtype
        );

        Ok(Self {
            device,
            tokenizer,
            backend: Qwen3ChatBackend::Native { text_model },
        })
    }

    fn load_gguf(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        let gguf_name = match variant {
            ModelVariant::Qwen306BGguf => "Qwen3-0.6B-Q8_0.gguf",
            ModelVariant::Qwen317BGguf => "Qwen3-1.7B-Q8_0.gguf",
            ModelVariant::Qwen34BGguf => "Qwen3-4B-Q4_K_M.gguf",
            ModelVariant::Qwen38BGguf => "Qwen3-8B-Q4_K_M.gguf",
            ModelVariant::Qwen314BGguf => "Qwen3-14B-Q4_K_M.gguf",
            _ => {
                return Err(Error::ModelLoadError(format!(
                    "Unsupported GGUF chat variant: {variant}"
                )))
            }
        };
        let gguf_path = model_dir.join(gguf_name);
        if !gguf_path.exists() {
            return Err(Error::ModelLoadError(format!(
                "GGUF checkpoint not found: {}",
                gguf_path.display()
            )));
        }

        let tokenizer = ChatTokenizer::load(model_dir, None)?;
        let mut reader = open_gguf_reader(&gguf_path, BackendKind::from(device.kind))?;
        let content = gguf_file::Content::read(&mut reader)
            .map_err(|e| Error::ModelLoadError(format!("Failed to parse GGUF header: {e}")))?;
        let text_model = QuantizedQwen3Model::from_gguf(content, &mut reader, &device.device)
            .map_err(|e| {
                Error::ModelLoadError(format!("Failed to load quantized Qwen3 GGUF model: {e}"))
            })?;

        info!(
            "Loaded Qwen3 GGUF chat model on {:?} from {}",
            device.kind,
            gguf_path.display()
        );

        Ok(Self {
            device,
            tokenizer,
            backend: Qwen3ChatBackend::Gguf {
                text_model: Mutex::new(text_model),
                gguf_file: gguf_name.to_string(),
            },
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
        match &self.backend {
            Qwen3ChatBackend::Native { .. } => {
                let mut state = self.start_decode(messages, max_new_tokens)?;
                loop {
                    let step = self.decode_step(&mut state)?;
                    if !step.delta.is_empty() {
                        for ch in step.delta.chars() {
                            let mut buf = [0u8; 4];
                            on_delta(ch.encode_utf8(&mut buf));
                        }
                    }
                    if step.finished {
                        return Ok(ChatGenerationOutput {
                            text: step.text,
                            tokens_generated: step.tokens_generated,
                        });
                    }
                }
            }
            Qwen3ChatBackend::Gguf { .. } => {
                self.generate_with_callback_gguf(messages, max_new_tokens, on_delta)
            }
        }
    }

    pub fn start_decode(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
    ) -> Result<ChatDecodeState> {
        let text_model = match &self.backend {
            Qwen3ChatBackend::Native { text_model } => text_model,
            Qwen3ChatBackend::Gguf { gguf_file, .. } => {
                return Err(Error::InvalidInput(format!(
                    "Incremental chat decode is unavailable for GGUF model {}",
                    gguf_file
                )))
            }
        };

        let prompt_ids = self.build_prompt(messages)?;
        let input_ids = Tensor::from_vec(
            prompt_ids.clone(),
            (1, prompt_ids.len()),
            &self.device.device,
        )?;

        let mut cache = Qwen3Cache::new(text_model.num_layers());
        let embeds = text_model.forward(&input_ids, 0, Some(&mut cache))?;
        let pos = embeds.dim(1)?;

        Ok(ChatDecodeState {
            cache,
            embeds,
            pos,
            generated_ids: Vec::new(),
            assembled: String::new(),
            max_new_tokens: max_new_tokens.max(1),
            finished: false,
        })
    }

    pub fn decode_step(&self, state: &mut ChatDecodeState) -> Result<ChatDecodeStep> {
        let text_model = match &self.backend {
            Qwen3ChatBackend::Native { text_model } => text_model,
            Qwen3ChatBackend::Gguf { gguf_file, .. } => {
                return Err(Error::InvalidInput(format!(
                    "Incremental chat decode is unavailable for GGUF model {}",
                    gguf_file
                )))
            }
        };

        if state.finished || state.generated_ids.len() >= state.max_new_tokens {
            state.finished = true;
            return Ok(ChatDecodeStep {
                delta: String::new(),
                text: state.assembled.trim().to_string(),
                tokens_generated: state.generated_ids.len(),
                finished: true,
            });
        }

        let logits = state.embeds.i((0, state.embeds.dim(1)? - 1))?;
        let next = argmax(&logits)?;

        if next == self.tokenizer.specials.im_end
            || next == self.tokenizer.specials.eos
            || self.tokenizer.specials.eos_alt == Some(next)
        {
            state.finished = true;
            return Ok(ChatDecodeStep {
                delta: String::new(),
                text: state.assembled.trim().to_string(),
                tokens_generated: state.generated_ids.len(),
                finished: true,
            });
        }

        state.generated_ids.push(next);
        let decoded = self.tokenizer.decode_text(&state.generated_ids)?;
        let delta = text_delta(&state.assembled, &decoded);
        state.assembled = decoded;

        let next_tensor = Tensor::from_vec(vec![next], (1, 1), &self.device.device)?;
        state.embeds = text_model.forward(&next_tensor, state.pos, Some(&mut state.cache))?;
        state.pos += 1;

        if state.generated_ids.len() >= state.max_new_tokens {
            state.finished = true;
        }

        Ok(ChatDecodeStep {
            delta,
            text: state.assembled.trim().to_string(),
            tokens_generated: state.generated_ids.len(),
            finished: state.finished,
        })
    }

    pub fn supports_incremental_decode(&self) -> bool {
        matches!(&self.backend, Qwen3ChatBackend::Native { .. })
    }

    fn generate_with_callback_gguf(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<ChatGenerationOutput> {
        let prompt_ids = self.build_prompt(messages)?;
        let mut model = match &self.backend {
            Qwen3ChatBackend::Gguf { text_model, .. } => text_model.lock().map_err(|_| {
                Error::InferenceError("Qwen3 GGUF model mutex poisoned".to_string())
            })?,
            Qwen3ChatBackend::Native { .. } => {
                return Err(Error::InferenceError(
                    "Internal error: GGUF generation called on safetensors backend".to_string(),
                ))
            }
        };

        model.clear_kv_cache();
        let input_ids = Tensor::from_vec(
            prompt_ids.clone(),
            (1, prompt_ids.len()),
            &self.device.device,
        )?;
        let mut logits = model
            .forward(&input_ids, 0)
            .map_err(|e| Error::InferenceError(format!("Qwen3 GGUF forward failed: {e}")))?;
        let mut position = prompt_ids.len();

        let mut generated_ids = Vec::new();
        let mut assembled = String::new();
        let max_new_tokens = max_new_tokens.max(1);

        while generated_ids.len() < max_new_tokens {
            let next = argmax(&logits)?;
            if next == self.tokenizer.specials.im_end
                || next == self.tokenizer.specials.eos
                || self.tokenizer.specials.eos_alt == Some(next)
            {
                break;
            }

            generated_ids.push(next);
            let decoded = self.tokenizer.decode_text(&generated_ids)?;
            let delta = text_delta(&assembled, &decoded);
            if !delta.is_empty() {
                for ch in delta.chars() {
                    let mut buf = [0u8; 4];
                    on_delta(ch.encode_utf8(&mut buf));
                }
            }
            assembled = decoded;

            let next_tensor = Tensor::from_vec(vec![next], (1, 1), &self.device.device)?;
            logits = model
                .forward(&next_tensor, position)
                .map_err(|e| Error::InferenceError(format!("Qwen3 GGUF decode failed: {e}")))?;
            position += 1;
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

        let mut prompt_messages = messages.to_vec();
        if !matches!(
            prompt_messages.first().map(|m| &m.role),
            Some(ChatRole::System)
        ) {
            prompt_messages.insert(
                0,
                ChatMessage {
                    role: ChatRole::System,
                    content: "You are a helpful assistant.".to_string(),
                },
            );
        }

        let mut ids = Vec::new();
        for message in &prompt_messages {
            let content = if matches!(message.role, ChatRole::Assistant) {
                strip_think_blocks(message.content.trim())
            } else {
                message.content.trim().to_string()
            };

            if content.is_empty() {
                continue;
            }

            ids.push(self.tokenizer.specials.im_start);
            ids.extend(
                self.tokenizer
                    .encode_text(&format!("{}\n", message.role.as_prompt_role()))?,
            );
            ids.extend(self.tokenizer.encode_text(&content)?);
            ids.push(self.tokenizer.specials.im_end);
            ids.extend(self.tokenizer.encode_text("\n")?);
        }

        ids.push(self.tokenizer.specials.im_start);
        ids.extend(self.tokenizer.encode_text("assistant\n")?);

        Ok(ids)
    }
}

fn strip_think_blocks(input: &str) -> String {
    let mut output = input.to_string();
    let open = "<think>";
    let close = "</think>";

    if let Some(close_idx) = output.find(close) {
        let has_open_before_close = output[..close_idx].find(open).is_some();
        if !has_open_before_close {
            let start = close_idx + close.len();
            output = output[start..].to_string();
        }
    }

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

    output.replace(close, " ").trim().to_string()
}

fn parse_qwen3_config(config_str: &str) -> Result<Qwen3Config> {
    let value: Value = serde_json::from_str(config_str)?;
    if let Some(text_config) = value.get("text_config") {
        serde_json::from_value(text_config.clone()).map_err(Error::from)
    } else {
        serde_json::from_value(value).map_err(Error::from)
    }
}

fn argmax(logits: &Tensor) -> Result<u32> {
    let logits = match logits.rank() {
        1 => logits.clone(),
        2 => {
            let (batch, _vocab) = logits.dims2()?;
            if batch != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected batched logits for argmax: expected batch=1, got {batch}"
                )));
            }
            logits.i(0)?
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected logits rank for argmax: {rank}"
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
        .take_while(|(a, b)| a == b)
        .count();
    current.chars().skip(common).collect()
}

#[cfg(test)]
mod tests {
    use super::strip_think_blocks;

    #[test]
    fn strip_think_blocks_handles_explicit_tags() {
        let stripped = strip_think_blocks("<think>reasoning</think>\nFinal answer");
        assert_eq!(stripped, "Final answer");
    }

    #[test]
    fn strip_think_blocks_handles_implicit_open_pattern() {
        let stripped = strip_think_blocks("reasoning only</think>\nFinal answer");
        assert_eq!(stripped, "Final answer");
    }
}
