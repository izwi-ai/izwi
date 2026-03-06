//! LFM2/LFM2.5 GGUF text-chat model loader and generation.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Mutex;

use candle_core::quantized::gguf_file;
use candle_core::{DType, IndexOp, Tensor, D};
use candle_transformers::models::quantized_lfm2::ModelWeights as QuantizedLfm2Model;
use serde::Deserialize;
use tracing::info;

use crate::backends::DeviceProfile;
use crate::backends::{open_gguf_reader, BackendKind};
use crate::error::{Error, Result};
use crate::model::ModelVariant;
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
    bos_token: Option<String>,
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
    fn load(model_dir: &Path) -> Result<Self> {
        let inner = Tokenizer::from_path(model_dir)?;
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
        let bos = config
            .bos_token
            .as_deref()
            .and_then(id_for)
            .or_else(|| id_for("<|startoftext|>"));
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
                bos,
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

pub struct Lfm2ChatModel {
    device: DeviceProfile,
    tokenizer: ChatTokenizer,
    text_model: Mutex<QuantizedLfm2Model>,
}

impl Lfm2ChatModel {
    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        let gguf_name = match variant {
            ModelVariant::Lfm2512BInstructGguf => "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
            ModelVariant::Lfm2512BThinkingGguf => "LFM2.5-1.2B-Thinking-Q4_K_M.gguf",
            _ => {
                return Err(Error::ModelLoadError(format!(
                    "Unsupported LFM2 GGUF chat variant: {variant}"
                )));
            }
        };
        let gguf_path = model_dir.join(gguf_name);
        if !gguf_path.exists() {
            return Err(Error::ModelLoadError(format!(
                "GGUF checkpoint not found: {}",
                gguf_path.display()
            )));
        }

        let tokenizer = ChatTokenizer::load(model_dir)?;
        let mut reader = open_gguf_reader(&gguf_path, BackendKind::from(device.kind))?;
        let content = gguf_file::Content::read(&mut reader)
            .map_err(|e| Error::ModelLoadError(format!("Failed to parse GGUF header: {e}")))?;
        let text_model = QuantizedLfm2Model::from_gguf(content, &mut reader, &device.device)
            .map_err(|e| Error::ModelLoadError(format!("Failed to load LFM2 GGUF model: {e}")))?;

        info!(
            "Loaded LFM2 GGUF chat model on {:?} from {}",
            device.kind,
            gguf_path.display()
        );

        Ok(Self {
            device,
            tokenizer,
            text_model: Mutex::new(text_model),
        })
    }

    pub fn supports_incremental_decode(&self) -> bool {
        false
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
        let mut model = self
            .text_model
            .lock()
            .map_err(|_| Error::InferenceError("LFM2 GGUF model mutex poisoned".to_string()))?;

        let input_ids = Tensor::from_vec(
            prompt_ids.clone(),
            (1, prompt_ids.len()),
            &self.device.device,
        )?;
        let mut logits = model
            .forward(&input_ids, 0)
            .map_err(|e| Error::InferenceError(format!("LFM2 GGUF forward failed: {e}")))?;
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

            if has_token_repetition_loop(&generated_ids) {
                break;
            }

            let next_tensor = Tensor::from_vec(vec![next], (1, 1), &self.device.device)?;
            logits = model
                .forward(&next_tensor, position)
                .map_err(|e| Error::InferenceError(format!("LFM2 GGUF decode failed: {e}")))?;
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
        if let Some(bos) = self.tokenizer.specials.bos {
            ids.push(bos);
        }

        let last_assistant_index = prompt_messages
            .iter()
            .rposition(|message| matches!(message.role, ChatRole::Assistant));

        for (index, message) in prompt_messages.iter().enumerate() {
            let content = if matches!(message.role, ChatRole::Assistant) {
                if Some(index) == last_assistant_index {
                    message.content.trim().to_string()
                } else {
                    strip_past_assistant_thinking(message.content.trim())
                }
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
            )));
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

fn strip_past_assistant_thinking(input: &str) -> String {
    if let Some((_reasoning, tail)) = input.rsplit_once("</think>") {
        tail.trim().to_string()
    } else {
        input.trim().to_string()
    }
}

fn has_suffix_repeat(ids: &[u32], span: usize, repeats: usize) -> bool {
    if span == 0 || repeats < 2 || ids.len() < span * repeats {
        return false;
    }
    let tail_start = ids.len() - span;
    let tail = &ids[tail_start..];
    (2..=repeats).all(|rep| {
        let start = ids.len() - (span * rep);
        &ids[start..start + span] == tail
    })
}

fn has_token_repetition_loop(ids: &[u32]) -> bool {
    // Catch common degenerate loops from greedy decode where the same token span
    // is emitted repeatedly (frequent in tiny reasoning models).
    if ids.len() < 48 {
        return false;
    }
    const PATTERNS: &[(usize, usize)] = &[(24, 3), (16, 3), (12, 3), (8, 4), (6, 5)];
    PATTERNS
        .iter()
        .any(|(span, repeats)| has_suffix_repeat(ids, *span, *repeats))
}

#[cfg(test)]
mod tests {
    use super::{has_token_repetition_loop, strip_past_assistant_thinking};

    #[test]
    fn strip_past_assistant_thinking_keeps_only_tail_after_close_tag() {
        let input = "<think>reasoning</think>\nFinal answer";
        assert_eq!(strip_past_assistant_thinking(input), "Final answer");
    }

    #[test]
    fn strip_past_assistant_thinking_keeps_unclosed_content() {
        let input = "<think>still reasoning";
        assert_eq!(
            strip_past_assistant_thinking(input),
            "<think>still reasoning"
        );
    }

    #[test]
    fn detects_token_repetition_loop() {
        let mut ids = Vec::new();
        let phrase = vec![1, 2, 3, 4, 5, 6, 7, 8];
        for _ in 0..5 {
            ids.extend(phrase.iter().copied());
        }
        ids.splice(0..0, vec![42; 16]);
        assert!(has_token_repetition_loop(&ids));
    }

    #[test]
    fn does_not_flag_short_sequences_as_loop() {
        let ids: Vec<u32> = (1..30).collect();
        assert!(!has_token_repetition_loop(&ids));
    }
}
