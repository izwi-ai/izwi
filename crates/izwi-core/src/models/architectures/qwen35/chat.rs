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

use super::text::{Qwen35TextModel, Qwen35TextRuntimeState};
use super::vision::{PreparedVisionInputs, Qwen35VisionModel};

const IMAGE_PAD_PLACEHOLDER: &str = "<|image_pad|>";
const VIDEO_PAD_PLACEHOLDER: &str = "<|video_pad|>";

#[derive(Debug)]
struct PreparedPrompt {
    prompt_ids: Vec<u32>,
    prompt_positions: Vec<[usize; 3]>,
    next_text_position: usize,
    vision_inputs: Option<PreparedVisionInputs>,
}

#[derive(Debug, Clone)]
pub struct ChatGenerationOutput {
    pub text: String,
    pub tokens_generated: usize,
}

pub struct ChatDecodeState {
    text_state: Qwen35TextRuntimeState,
    logits: Tensor,
    generated_ids: Vec<u32>,
    assembled: String,
    max_new_tokens: usize,
    finished: bool,
    next_text_position: usize,
    config: ChatGenerationConfig,
    rng: SimpleRng,
}

#[derive(Debug, Clone)]
pub struct ChatDecodeStep {
    pub delta: String,
    pub text: String,
    pub tokens_generated: usize,
    pub finished: bool,
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
    image_pad: u32,
    video_pad: u32,
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
    #[serde(default)]
    chat_template: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AddedToken {
    content: String,
}

struct Qwen35Tokenizer {
    inner: Tokenizer,
    vocab_size: usize,
    specials: SpecialTokenIds,
    literal_special_tokens: Vec<(String, u32)>,
    chat_template: String,
    default_enable_thinking: bool,
    bos_token: Option<String>,
}

#[derive(Debug)]
struct GgufTokenizerMetadata {
    tokens: Vec<String>,
    merges: Vec<String>,
    pre_tokenizer: Option<String>,
    chat_template: String,
    eos_token_id: Option<u32>,
}

impl Qwen35Tokenizer {
    fn load(model_dir: &Path, variant: ModelVariant, loader: &GgufLoader) -> Result<Self> {
        let gguf_meta = parse_gguf_tokenizer_metadata(loader)?;
        let config = load_tokenizer_config_file(model_dir)?;
        let inner = match Tokenizer::from_path(model_dir) {
            Ok(inner) => inner,
            Err(_) => Tokenizer::from_gguf_bpe(
                &gguf_meta.tokens,
                &gguf_meta.merges,
                gguf_meta.pre_tokenizer.as_deref(),
                false,
            )?,
        };
        let vocab_size = inner.vocab_size();

        let mut token_to_id: HashMap<String, u32> = HashMap::new();
        if let Some(cfg) = &config {
            for (id, entry) in &cfg.added_tokens_decoder {
                if let Ok(parsed) = id.parse::<u32>() {
                    token_to_id.insert(entry.content.clone(), parsed);
                }
            }
        }
        for (idx, token) in gguf_meta.tokens.iter().enumerate() {
            let id = u32::try_from(idx).map_err(|_| {
                Error::TokenizationError(format!("GGUF tokenizer id out of range: {idx}"))
            })?;
            token_to_id.entry(token.clone()).or_insert(id);
        }

        let id_for = |token: &str| token_to_id.get(token).copied();
        let im_start = id_for("<|im_start|>")
            .ok_or_else(|| Error::TokenizationError("Missing <|im_start|> token id".to_string()))?;
        let im_end = id_for("<|im_end|>")
            .ok_or_else(|| Error::TokenizationError("Missing <|im_end|> token id".to_string()))?;
        let image_pad = id_for("<|image_pad|>").ok_or_else(|| {
            Error::TokenizationError("Missing <|image_pad|> token id".to_string())
        })?;
        let video_pad = id_for("<|video_pad|>").ok_or_else(|| {
            Error::TokenizationError("Missing <|video_pad|> token id".to_string())
        })?;

        let eos = config
            .as_ref()
            .and_then(|cfg| cfg.eos_token.as_deref())
            .and_then(id_for)
            .or(gguf_meta.eos_token_id)
            .unwrap_or(im_end);
        let eos_alt = id_for("<|endoftext|>");

        let chat_template = config
            .as_ref()
            .and_then(|cfg| cfg.chat_template.clone())
            .unwrap_or_else(|| gguf_meta.chat_template.clone());
        let default_enable_thinking = resolve_default_enable_thinking(&chat_template, variant);

        let mut literal_special_tokens: Vec<(String, u32)> = token_to_id
            .iter()
            .filter_map(|(token, id)| {
                (token.starts_with("<|") && token.ends_with("|>")).then_some((token.clone(), *id))
            })
            .collect();
        literal_special_tokens.sort_by(|(left, _), (right, _)| {
            right.len().cmp(&left.len()).then_with(|| left.cmp(right))
        });

        Ok(Self {
            inner,
            vocab_size,
            specials: SpecialTokenIds {
                im_start,
                im_end,
                image_pad,
                video_pad,
                eos,
                eos_alt,
            },
            literal_special_tokens,
            chat_template,
            default_enable_thinking,
            bos_token: config.and_then(|cfg| cfg.bos_token),
        })
    }

    fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        if self.literal_special_tokens.is_empty() {
            return self.inner.encode(text);
        }

        let mut ids = Vec::new();
        let mut offset = 0usize;
        while offset < text.len() {
            let tail = &text[offset..];
            let mut next_match: Option<(usize, &str, u32)> = None;
            for (token, token_id) in &self.literal_special_tokens {
                if let Some(rel_idx) = tail.find(token) {
                    let candidate = (rel_idx, token.as_str(), *token_id);
                    match next_match {
                        None => next_match = Some(candidate),
                        Some((best_idx, best_token, _)) => {
                            if rel_idx < best_idx
                                || (rel_idx == best_idx && token.len() > best_token.len())
                            {
                                next_match = Some(candidate);
                            }
                        }
                    }
                }
            }

            let Some((rel_idx, matched_token, matched_id)) = next_match else {
                ids.extend(self.inner.encode(tail)?);
                break;
            };

            if rel_idx > 0 {
                ids.extend(self.inner.encode(&tail[..rel_idx])?);
            }
            ids.push(matched_id);
            offset += rel_idx + matched_token.len();
        }

        Ok(ids)
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
    vision_model: Qwen35VisionModel,
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
        let tokenizer = Qwen35Tokenizer::load(model_dir, variant, &text_loader)?;
        let text_model = Qwen35TextModel::load(&text_loader, &text_config, &device.device)?;

        let projector_loader = GgufLoader::from_path_with_backend(&mmproj_path, backend)?;
        let projector_checkpoint = projector_loader.get_model_info();
        let vision_model =
            Qwen35VisionModel::load(&projector_loader, &device.device, text_model.hidden_size())?;

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
            vision_model,
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
        self.prompt_token_ids_with_config(messages, &ChatGenerationConfig::default())
    }

    pub fn prompt_token_ids_with_config(
        &self,
        messages: &[ChatMessage],
        config: &ChatGenerationConfig,
    ) -> Result<Vec<u32>> {
        Ok(self.prepare_prompt(messages, config)?.prompt_ids)
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
        let mut state = self.start_decode_state_with_config(messages, max_new_tokens, config)?;
        loop {
            let step = self.decode_step(&mut state)?;
            if !step.delta.is_empty() {
                on_delta(&step.delta);
            }
            if step.finished {
                return Ok(ChatGenerationOutput {
                    text: step.text,
                    tokens_generated: step.tokens_generated,
                });
            }
        }
    }

    pub fn supports_incremental_decode(&self) -> bool {
        true
    }

    pub fn device_kind(&self) -> BackendKind {
        self.device_kind
    }

    pub fn start_decode_state(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
    ) -> Result<ChatDecodeState> {
        self.start_decode_state_with_config(
            messages,
            max_new_tokens,
            &ChatGenerationConfig::default(),
        )
    }

    pub fn start_decode_state_with_config(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
    ) -> Result<ChatDecodeState> {
        let prepared_prompt = self.prepare_prompt(messages, config)?;
        if prepared_prompt.prompt_ids.is_empty() {
            return Err(Error::InvalidInput(
                "Chat request must include at least one tokenizable message".to_string(),
            ));
        }

        let mut text_state = self.text_model.new_state();
        let logits = self.prefill_prompt(&prepared_prompt, &mut text_state)?;

        Ok(ChatDecodeState {
            text_state,
            logits,
            generated_ids: Vec::new(),
            assembled: String::new(),
            max_new_tokens: max_new_tokens.max(1),
            finished: false,
            next_text_position: prepared_prompt.next_text_position,
            config: config.clone(),
            rng: SimpleRng::new(config.seed),
        })
    }

    pub fn decode_step(&self, state: &mut ChatDecodeState) -> Result<ChatDecodeStep> {
        if state.finished || state.generated_ids.len() >= state.max_new_tokens {
            state.finished = true;
            return Ok(ChatDecodeStep {
                delta: String::new(),
                text: state.assembled.trim().to_string(),
                tokens_generated: state.generated_ids.len(),
                finished: true,
            });
        }

        let next = sample_next_token(
            &state.logits,
            self.tokenizer.vocab_size,
            &state.config,
            &state.generated_ids,
            &mut state.rng,
        )?;
        if self.is_stop_token(next, &state.config) {
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
        state.logits = self.text_model.forward_token_id_at(
            next,
            [state.next_text_position; 3],
            &mut state.text_state,
        )?;
        state.next_text_position += 1;
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

    fn is_stop_token(&self, token_id: u32, config: &ChatGenerationConfig) -> bool {
        token_id == self.tokenizer.specials.im_end
            || token_id == self.tokenizer.specials.eos
            || self.tokenizer.specials.eos_alt == Some(token_id)
            || config.stop_token_ids.contains(&token_id)
    }

    fn prefill_prompt(
        &self,
        prepared_prompt: &PreparedPrompt,
        text_state: &mut Qwen35TextRuntimeState,
    ) -> Result<Tensor> {
        let mut logits: Option<Tensor> = None;
        let mut vision_embedding_index = 0usize;
        let mut idx = 0usize;
        while idx < prepared_prompt.prompt_ids.len() {
            let token_id = prepared_prompt.prompt_ids[idx];
            let position_ids = prepared_prompt.prompt_positions[idx];
            let is_last = idx + 1 == prepared_prompt.prompt_ids.len();
            if token_id == self.tokenizer.specials.image_pad {
                let vision_inputs = prepared_prompt.vision_inputs.as_ref().ok_or_else(|| {
                    Error::InvalidInput(
                        "Qwen3.5 image placeholders require paired media inputs".to_string(),
                    )
                })?;
                if vision_embedding_index >= vision_inputs.embeddings.dim(0)? {
                    return Err(Error::InferenceError(
                        "Qwen3.5 prompt consumed more image placeholders than vision embeddings"
                            .to_string(),
                    ));
                }
                let embedding = vision_inputs
                    .embeddings
                    .narrow(0, vision_embedding_index, 1)?
                    .reshape((1, 1, self.text_model.hidden_size()))?;
                vision_embedding_index += 1;
                if is_last {
                    logits = Some(self.text_model.forward_input_embedding_at(
                        &embedding,
                        position_ids,
                        text_state,
                    )?);
                } else {
                    self.text_model.forward_input_embedding_hidden_at(
                        &embedding,
                        position_ids,
                        text_state,
                    )?;
                }
            } else if token_id == self.tokenizer.specials.video_pad {
                return Err(Error::InvalidInput(
                    "Qwen3.5 video inputs are not implemented yet".to_string(),
                ));
            } else {
                let mut run_end = idx + 1;
                while run_end < prepared_prompt.prompt_ids.len() {
                    let candidate = prepared_prompt.prompt_ids[run_end];
                    if candidate == self.tokenizer.specials.image_pad
                        || candidate == self.tokenizer.specials.video_pad
                    {
                        break;
                    }
                    run_end += 1;
                }

                let compute_logits = run_end == prepared_prompt.prompt_ids.len();
                if let Some(run_logits) = self.text_model.prefill_token_ids(
                    &prepared_prompt.prompt_ids[idx..run_end],
                    &prepared_prompt.prompt_positions[idx..run_end],
                    text_state,
                    compute_logits,
                )? {
                    logits = Some(run_logits);
                }
                idx = run_end;
                continue;
            }

            idx += 1;
        }

        if let Some(vision_inputs) = prepared_prompt.vision_inputs.as_ref() {
            if vision_embedding_index != vision_inputs.embeddings.dim(0)? {
                return Err(Error::InferenceError(format!(
                    "Qwen3.5 prompt consumed {vision_embedding_index} image embeddings, expected {}",
                    vision_inputs.embeddings.dim(0)?
                )));
            }
        }

        logits.ok_or_else(|| {
            Error::InferenceError("Qwen3.5 prompt prefill did not produce logits".to_string())
        })
    }

    fn prepare_prompt(
        &self,
        messages: &[ChatMessage],
        config: &ChatGenerationConfig,
    ) -> Result<PreparedPrompt> {
        let prompt = render_prompt(messages, config, self.default_enable_thinking())?;
        let image_placeholders = prompt.matches(IMAGE_PAD_PLACEHOLDER).count();
        let video_placeholders = prompt.matches(VIDEO_PAD_PLACEHOLDER).count();
        if video_placeholders > 0 {
            return Err(Error::InvalidInput(
                "Qwen3.5 video inputs are not implemented yet".to_string(),
            ));
        }

        let Some(vision_inputs) = self
            .vision_model
            .encode_media(&config.request.media_inputs)?
        else {
            if image_placeholders > 0 {
                return Err(Error::InvalidInput(
                    "Qwen3.5 image placeholders require paired media inputs".to_string(),
                ));
            }
            let prompt_ids = self.tokenizer.encode_text(&prompt)?;
            let prompt_positions = build_text_positions(prompt_ids.len());
            return Ok(PreparedPrompt {
                next_text_position: prompt_positions.len(),
                prompt_ids,
                prompt_positions,
                vision_inputs: None,
            });
        };

        if image_placeholders == 0 {
            return Err(Error::InvalidInput(
                "Qwen3.5 media inputs require <|image_pad|> placeholders in the rendered prompt"
                    .to_string(),
            ));
        }
        if image_placeholders != vision_inputs.token_counts.len() {
            return Err(Error::InvalidInput(format!(
                "Qwen3.5 prompt/media mismatch: rendered {image_placeholders} image placeholders for {} media inputs",
                vision_inputs.token_counts.len()
            )));
        }

        let expanded_prompt = expand_image_placeholders(&prompt, &vision_inputs.token_counts)?;
        let prompt_ids = self.tokenizer.encode_text(&expanded_prompt)?;
        let (prompt_positions, next_text_position) = build_prompt_positions(
            &prompt_ids,
            &vision_inputs,
            self.tokenizer.specials.image_pad,
            self.tokenizer.specials.video_pad,
            self.vision_model.spatial_merge_size(),
        )?;
        Ok(PreparedPrompt {
            prompt_ids,
            prompt_positions,
            next_text_position,
            vision_inputs: Some(vision_inputs),
        })
    }
}

fn build_text_positions(token_count: usize) -> Vec<[usize; 3]> {
    (0..token_count).map(|idx| [idx; 3]).collect()
}

fn expand_image_placeholders(prompt: &str, token_counts: &[usize]) -> Result<String> {
    let segments: Vec<&str> = prompt.split(IMAGE_PAD_PLACEHOLDER).collect();
    if segments.len() != token_counts.len() + 1 {
        return Err(Error::InvalidInput(format!(
            "Qwen3.5 prompt/media mismatch while expanding image placeholders: found {} placeholders for {} media inputs",
            segments.len().saturating_sub(1),
            token_counts.len()
        )));
    }

    let mut expanded = String::with_capacity(
        prompt.len()
            + token_counts
                .iter()
                .map(|count| IMAGE_PAD_PLACEHOLDER.len() * count.saturating_sub(1))
                .sum::<usize>(),
    );
    for (idx, segment) in segments.iter().enumerate() {
        expanded.push_str(segment);
        if let Some(&token_count) = token_counts.get(idx) {
            if token_count == 0 {
                return Err(Error::InvalidInput(
                    "Qwen3.5 image inputs must contribute at least one LLM token".to_string(),
                ));
            }
            expanded.push_str(&IMAGE_PAD_PLACEHOLDER.repeat(token_count));
        }
    }
    Ok(expanded)
}

fn build_prompt_positions(
    prompt_ids: &[u32],
    vision_inputs: &PreparedVisionInputs,
    image_pad_id: u32,
    video_pad_id: u32,
    spatial_merge_size: usize,
) -> Result<(Vec<[usize; 3]>, usize)> {
    let mut prompt_positions = Vec::with_capacity(prompt_ids.len());
    let mut current_pos = 0usize;
    let mut idx = 0usize;
    let mut image_idx = 0usize;

    while idx < prompt_ids.len() {
        let token_id = prompt_ids[idx];
        if token_id == image_pad_id {
            let Some(grid) = vision_inputs.grids.get(image_idx).copied() else {
                return Err(Error::InvalidInput(
                    "Qwen3.5 prompt consumed more image placeholder runs than media inputs"
                        .to_string(),
                ));
            };
            let expected_tokens = *vision_inputs.token_counts.get(image_idx).ok_or_else(|| {
                Error::InvalidInput(
                    "Qwen3.5 image token counts are missing for a prepared media input".to_string(),
                )
            })?;
            let run_start = idx;
            while idx < prompt_ids.len() && prompt_ids[idx] == image_pad_id {
                idx += 1;
            }
            let actual_tokens = idx - run_start;
            if actual_tokens != expected_tokens {
                return Err(Error::InvalidInput(format!(
                    "Qwen3.5 image placeholder count mismatch for media input {}: prompt has {}, encoder expects {}",
                    image_idx + 1,
                    actual_tokens,
                    expected_tokens
                )));
            }
            prompt_positions.extend(vision_position_ids(current_pos, grid, spatial_merge_size)?);
            current_pos += grid[1].max(grid[2]) / spatial_merge_size;
            image_idx += 1;
            continue;
        }
        if token_id == video_pad_id {
            return Err(Error::InvalidInput(
                "Qwen3.5 video inputs are not implemented yet".to_string(),
            ));
        }

        let run_start = idx;
        while idx < prompt_ids.len()
            && prompt_ids[idx] != image_pad_id
            && prompt_ids[idx] != video_pad_id
        {
            idx += 1;
        }
        let text_len = idx - run_start;
        prompt_positions.extend((0..text_len).map(|offset| {
            let position = current_pos + offset;
            [position; 3]
        }));
        current_pos += text_len;
    }

    if image_idx != vision_inputs.grids.len() {
        return Err(Error::InvalidInput(format!(
            "Qwen3.5 prompt consumed {} image placeholder runs for {} media inputs",
            image_idx,
            vision_inputs.grids.len()
        )));
    }
    if prompt_positions.len() != prompt_ids.len() {
        return Err(Error::InferenceError(format!(
            "Qwen3.5 prompt position count mismatch: {} positions for {} tokens",
            prompt_positions.len(),
            prompt_ids.len()
        )));
    }

    Ok((prompt_positions, current_pos))
}

fn vision_position_ids(
    start_position: usize,
    grid_thw: [usize; 3],
    spatial_merge_size: usize,
) -> Result<Vec<[usize; 3]>> {
    if spatial_merge_size == 0 {
        return Err(Error::InvalidInput(
            "Qwen3.5 spatial merge size must be non-zero".to_string(),
        ));
    }

    let llm_grid_t = grid_thw[0];
    let llm_grid_h = grid_thw[1] / spatial_merge_size;
    let llm_grid_w = grid_thw[2] / spatial_merge_size;
    let image_seq_length = llm_grid_t * llm_grid_h * llm_grid_w;
    let mut positions = Vec::with_capacity(image_seq_length);
    for _temporal_idx in 0..llm_grid_t {
        for height_idx in 0..llm_grid_h {
            for width_idx in 0..llm_grid_w {
                positions.push([
                    start_position,
                    start_position + height_idx,
                    start_position + width_idx,
                ]);
            }
        }
    }
    Ok(positions)
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

fn render_prompt(
    messages: &[ChatMessage],
    config: &ChatGenerationConfig,
    default_enable_thinking: bool,
) -> Result<String> {
    if messages.is_empty() {
        return Err(Error::InvalidInput(
            "Qwen3.5 chat prompt requires at least one message".to_string(),
        ));
    }

    let mut prompt = String::new();
    let leading_system =
        matches!(messages.first(), Some(message) if message.role == ChatRole::System);
    let system_content = if leading_system {
        messages[0].content.trim()
    } else {
        ""
    };

    if !config.request.tools.is_empty() {
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for tool in &config.request.tools {
            prompt.push('\n');
            prompt.push_str(&serde_json::to_string(tool)?);
        }
        prompt.push_str("\n</tools>");
        prompt.push_str(TOOL_PROMPT_SUFFIX);
        if !system_content.is_empty() {
            prompt.push_str("\n\n");
            prompt.push_str(system_content);
        }
        prompt.push_str("<|im_end|>\n");
    } else if leading_system {
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str(system_content);
        prompt.push_str("<|im_end|>\n");
    }

    let last_query_index = last_query_index(messages)?;
    for (index, message) in messages.iter().enumerate() {
        if message.role == ChatRole::System {
            if index != 0 {
                return Err(Error::InvalidInput(
                    "Qwen3.5 system message must be the first message".to_string(),
                ));
            }
            continue;
        }

        match message.role {
            ChatRole::User => {
                prompt.push_str("<|im_start|>user\n");
                prompt.push_str(message.content.trim());
                prompt.push_str("<|im_end|>\n");
            }
            ChatRole::Assistant => {
                let (reasoning_content, content) = split_assistant_reasoning(&message.content);
                prompt.push_str("<|im_start|>assistant\n");
                if index > last_query_index {
                    prompt.push_str("<think>\n");
                    prompt.push_str(reasoning_content.trim());
                    prompt.push_str("\n</think>\n\n");
                    prompt.push_str(content.trim_start());
                } else {
                    prompt.push_str(content.trim());
                }
                prompt.push_str("<|im_end|>\n");
            }
            ChatRole::System => {}
        }
    }

    prompt.push_str("<|im_start|>assistant\n");
    if config
        .request
        .enable_thinking
        .unwrap_or(default_enable_thinking)
    {
        prompt.push_str("<think>\n");
    } else {
        prompt.push_str("<think>\n\n</think>\n\n");
    }
    Ok(prompt)
}

fn last_query_index(messages: &[ChatMessage]) -> Result<usize> {
    messages
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, message)| {
            (message.role == ChatRole::User && !is_tool_response(&message.content)).then_some(index)
        })
        .ok_or_else(|| {
            Error::InvalidInput("Qwen3.5 prompt requires at least one user query".to_string())
        })
}

fn is_tool_response(content: &str) -> bool {
    let content = content.trim();
    content.starts_with("<tool_response>") && content.ends_with("</tool_response>")
}

fn split_assistant_reasoning(content: &str) -> (&str, &str) {
    let Some(end_idx) = content.find("</think>") else {
        return ("", content);
    };
    let reasoning_prefix = &content[..end_idx];
    let reasoning = reasoning_prefix
        .rsplit_once("<think>")
        .map(|(_, reasoning)| reasoning)
        .unwrap_or(reasoning_prefix);
    let answer = content[(end_idx + "</think>".len())..].trim_start_matches('\n');
    (reasoning.trim_matches('\n'), answer)
}

const TOOL_PROMPT_SUFFIX: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

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

fn parse_gguf_tokenizer_metadata(loader: &GgufLoader) -> Result<GgufTokenizerMetadata> {
    Ok(GgufTokenizerMetadata {
        tokens: required_string_array(loader, "tokenizer.ggml.tokens")?,
        merges: required_string_array(loader, "tokenizer.ggml.merges")?,
        pre_tokenizer: loader.get_metadata_string("tokenizer.ggml.pre"),
        chat_template: loader
            .get_metadata_string("tokenizer.chat_template")
            .ok_or_else(|| {
                Error::ModelLoadError(
                    "Missing or invalid GGUF metadata: tokenizer.chat_template".to_string(),
                )
            })?,
        eos_token_id: loader
            .get_metadata_u64("tokenizer.ggml.eos_token_id")
            .and_then(|value| u32::try_from(value).ok()),
    })
}

fn load_tokenizer_config_file(model_dir: &Path) -> Result<Option<TokenizerConfigFile>> {
    let config_path = model_dir.join("tokenizer_config.json");
    if !config_path.exists() {
        return Ok(None);
    }
    let config_str = fs::read_to_string(config_path)?;
    let config: TokenizerConfigFile = serde_json::from_str(&config_str)?;
    Ok(Some(config))
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

fn required_string_array(loader: &GgufLoader, key: &str) -> Result<Vec<String>> {
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
        let Some(raw) = gguf_to_string(item) else {
            return Err(Error::ModelLoadError(format!(
                "Expected string array values for {key}"
            )));
        };
        values.push(raw);
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

fn gguf_to_string(value: &GgufValue) -> Option<String> {
    match value {
        GgufValue::String(s) => Some(s.clone()),
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
    vocab_size: usize,
    config: &ChatGenerationConfig,
    history: &[u32],
    rng: &mut SimpleRng,
) -> Result<u32> {
    let mut values = logits_to_vec(logits)?;
    clamp_logits_to_vocab(&mut values, vocab_size);

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

fn clamp_logits_to_vocab(values: &mut [f32], vocab_size: usize) {
    if vocab_size < values.len() {
        values[vocab_size..].fill(f32::NEG_INFINITY);
    }
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
    use crate::backends::{DeviceKind, DeviceProfile, DeviceSelector};
    use crate::models::shared::chat::{ChatMediaInput, ChatMediaKind, ChatRequestConfig};
    use std::path::PathBuf;

    fn local_model_dir(name: &str) -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        PathBuf::from(home)
            .join("Library/Application Support/izwi/models")
            .join(name)
    }

    fn local_metal_device() -> Option<DeviceProfile> {
        let Ok(device) = DeviceSelector::detect_with_preference(Some("metal")) else {
            return None;
        };
        if device.kind != DeviceKind::Metal {
            return None;
        }
        Some(device)
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
    fn render_prompt_injects_tool_system_preamble() {
        let config = ChatGenerationConfig {
            request: ChatRequestConfig {
                enable_thinking: Some(true),
                tools: vec![serde_json::json!({"type":"function","function":{"name":"lookup"}})],
                media_inputs: Vec::new(),
            },
            ..ChatGenerationConfig::default()
        };
        let prompt = render_prompt(
            &[
                ChatMessage {
                    role: ChatRole::System,
                    content: "Be precise.".to_string(),
                },
                ChatMessage {
                    role: ChatRole::User,
                    content: "Hi".to_string(),
                },
            ],
            &config,
            false,
        )
        .expect("prompt should render");

        assert!(prompt.contains("# Tools"));
        assert!(prompt.contains("\"name\":\"lookup\""));
        assert!(prompt.contains("Be precise."));
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n"));
    }

    #[test]
    fn render_prompt_can_force_closed_think_block() {
        let config = ChatGenerationConfig {
            request: ChatRequestConfig {
                enable_thinking: Some(false),
                tools: Vec::new(),
                media_inputs: Vec::new(),
            },
            ..ChatGenerationConfig::default()
        };
        let prompt = render_prompt(
            &[ChatMessage {
                role: ChatRole::User,
                content: "Answer briefly.".to_string(),
            }],
            &config,
            true,
        )
        .expect("prompt should render");

        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn split_assistant_reasoning_handles_implicit_open_pattern() {
        assert_eq!(
            split_assistant_reasoning("reasoning first</think>\nFinal answer"),
            ("reasoning first", "Final answer")
        );
    }

    #[test]
    fn render_prompt_strips_prior_assistant_reasoning_from_history() {
        let config = ChatGenerationConfig {
            request: ChatRequestConfig {
                enable_thinking: Some(true),
                tools: Vec::new(),
                media_inputs: Vec::new(),
            },
            ..ChatGenerationConfig::default()
        };
        let prompt = render_prompt(
            &[
                ChatMessage {
                    role: ChatRole::User,
                    content: "First question".to_string(),
                },
                ChatMessage {
                    role: ChatRole::Assistant,
                    content: "reasoning first</think>\nFinal answer".to_string(),
                },
                ChatMessage {
                    role: ChatRole::User,
                    content: "Follow-up".to_string(),
                },
            ],
            &config,
            true,
        )
        .expect("prompt should render");

        assert!(prompt.contains("<|im_start|>assistant\nFinal answer<|im_end|>\n"));
        assert!(!prompt.contains("reasoning first"));
    }

    #[test]
    fn sample_next_token_masks_logits_above_vocab_limit() {
        let logits = Tensor::from_vec(
            vec![0.1f32, 0.2, 0.3, 12.0, 9.0],
            (5,),
            &candle_core::Device::Cpu,
        )
        .expect("logits");
        let config = ChatGenerationConfig {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            stop_token_ids: Vec::new(),
            seed: 7,
            request: ChatRequestConfig::default(),
        };
        let mut rng = SimpleRng::new(7);
        let token = sample_next_token(&logits, 3, &config, &[], &mut rng).expect("sample token");
        assert_eq!(token, 2);
    }

    #[test]
    fn sample_next_token_errors_when_vocab_limit_is_zero() {
        let logits = Tensor::from_vec(vec![0.1f32, 0.2, 0.3], (3,), &candle_core::Device::Cpu)
            .expect("logits");
        let config = ChatGenerationConfig {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            stop_token_ids: Vec::new(),
            seed: 7,
            request: ChatRequestConfig::default(),
        };
        let mut rng = SimpleRng::new(7);
        let result = sample_next_token(&logits, 0, &config, &[], &mut rng);
        assert!(result.is_err());
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
        assert_eq!(
            model.default_enable_thinking(),
            resolve_default_enable_thinking(model.chat_template(), ModelVariant::Qwen354BGguf)
        );
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
            request: ChatRequestConfig::default(),
        };

        let output = model
            .generate_with_config(&messages, 4, &config)
            .expect("qwen3.5 text generation should run");

        assert!(output.tokens_generated <= 4);
    }

    fn generate_local_qwen35_variant_text_smoke_if_available(
        model_name: &str,
        variant: ModelVariant,
    ) {
        let model_dir = local_model_dir(model_name);
        if !model_dir.exists() {
            return;
        }

        let model =
            Qwen35ChatModel::load(&model_dir, variant, DeviceProfile::cpu()).expect("model loads");
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            content: "Who made you? Reply in one short sentence.".to_string(),
        }];
        let config = ChatGenerationConfig {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            stop_token_ids: Vec::new(),
            seed: 7,
            request: ChatRequestConfig {
                enable_thinking: Some(false),
                tools: Vec::new(),
                media_inputs: Vec::new(),
            },
        };

        let output = model
            .generate_with_config(&messages, 32, &config)
            .expect("qwen3.5 text generation should run");
        let text = output.text.trim();

        assert!(output.tokens_generated <= 32);
        assert!(!text.is_empty(), "qwen3.5 output should not be empty");
        assert!(
            !text.starts_with("</think>") && !text.starts_with("think>"),
            "qwen3.5 output should not start with malformed think closing tags: {text}"
        );
    }

    #[test]
    fn generate_local_qwen35_4b_text_smoke_if_available() {
        generate_local_qwen35_variant_text_smoke_if_available(
            "Qwen3.5-4B",
            ModelVariant::Qwen354BGguf,
        );
    }

    #[test]
    fn generate_local_qwen35_text_metal_smoke_if_available() {
        let model_dir = local_model_dir("Qwen3.5-0.8B");
        if !model_dir.exists() {
            return;
        }

        let Some(device) = local_metal_device() else {
            return;
        };
        let model = Qwen35ChatModel::load(&model_dir, ModelVariant::Qwen3508BGguf, device)
            .expect("qwen3.5 assets should load on Metal");
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
            request: ChatRequestConfig::default(),
        };

        let output = model
            .generate_with_config(&messages, 4, &config)
            .expect("qwen3.5 text generation should run on Metal");

        assert!(output.tokens_generated <= 4);
    }

    #[test]
    fn expand_image_placeholders_repeats_each_media_slot() {
        let prompt = concat!(
            "<|vision_start|><|image_pad|><|vision_end|>",
            " then ",
            "<|vision_start|><|image_pad|><|vision_end|>"
        );
        let expanded = expand_image_placeholders(prompt, &[4, 2]).expect("expand");
        assert_eq!(expanded.matches(IMAGE_PAD_PLACEHOLDER).count(), 6);
        assert!(expanded.contains(
            "<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>"
        ));
        assert!(expanded.contains("then <|vision_start|><|image_pad|><|image_pad|><|vision_end|>"));
    }

    #[test]
    fn build_prompt_positions_matches_qwen_multimodal_rope_layout() {
        let image_pad_id = 248056;
        let prompt_ids = vec![
            11,
            image_pad_id,
            image_pad_id,
            image_pad_id,
            image_pad_id,
            12,
        ];
        let vision_inputs = PreparedVisionInputs {
            embeddings: Tensor::zeros((4, 8), DType::F32, &candle_core::Device::Cpu)
                .expect("embeddings"),
            grids: vec![[1, 4, 4]],
            token_counts: vec![4],
        };

        let (positions, next_text_position) =
            build_prompt_positions(&prompt_ids, &vision_inputs, image_pad_id, 248057, 2)
                .expect("positions");

        assert_eq!(
            positions,
            vec![
                [0, 0, 0],
                [1, 1, 1],
                [1, 1, 2],
                [1, 2, 1],
                [1, 2, 2],
                [3, 3, 3],
            ]
        );
        assert_eq!(next_text_position, 4);
    }

    #[test]
    fn generate_local_qwen35_image_smoke_if_available() {
        let model_dir = local_model_dir("Qwen3.5-0.8B");
        if !model_dir.exists() {
            return;
        }
        let image_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../izwi-desktop/icons/32x32.png");
        if !image_path.exists() {
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
            content: format!(
                "{}{}{} Reply with one short word.",
                "<|vision_start|>", IMAGE_PAD_PLACEHOLDER, "<|vision_end|>"
            ),
        }];
        let config = ChatGenerationConfig {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            stop_token_ids: Vec::new(),
            seed: 7,
            request: ChatRequestConfig {
                enable_thinking: Some(false),
                tools: Vec::new(),
                media_inputs: vec![ChatMediaInput {
                    kind: ChatMediaKind::Image,
                    source: image_path.display().to_string(),
                }],
            },
        };

        let output = model
            .generate_with_config(&messages, 2, &config)
            .expect("qwen3.5 image generation should run");

        assert!(output.tokens_generated <= 2);
    }

    #[test]
    fn generate_local_qwen35_image_metal_smoke_if_available() {
        let model_dir = local_model_dir("Qwen3.5-0.8B");
        if !model_dir.exists() {
            return;
        }
        let image_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../izwi-desktop/icons/32x32.png");
        if !image_path.exists() {
            return;
        }

        let Some(device) = local_metal_device() else {
            return;
        };
        let model = Qwen35ChatModel::load(&model_dir, ModelVariant::Qwen3508BGguf, device)
            .expect("qwen3.5 assets should load on Metal");
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            content: format!(
                "{}{}{} Reply with one short word.",
                "<|vision_start|>", IMAGE_PAD_PLACEHOLDER, "<|vision_end|>"
            ),
        }];
        let config = ChatGenerationConfig {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            stop_token_ids: Vec::new(),
            seed: 7,
            request: ChatRequestConfig {
                enable_thinking: Some(false),
                tools: Vec::new(),
                media_inputs: vec![ChatMediaInput {
                    kind: ChatMediaKind::Image,
                    source: image_path.display().to_string(),
                }],
            },
        };

        let output = model
            .generate_with_config(&messages, 2, &config)
            .expect("qwen3.5 image generation should run on Metal");

        assert!(output.tokens_generated <= 2);
    }
}
