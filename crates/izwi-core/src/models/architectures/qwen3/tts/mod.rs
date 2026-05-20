//! Native Qwen3-TTS model loader and inference.
//!
//! This module provides native Rust implementation for Qwen3-TTS models,
//! supporting both CustomVoice (preset speakers) and voice cloning modes.

mod config;
mod predictor;
mod rope;
mod speech_tokenizer;
mod talker;
mod tokenizer;

pub use config::Qwen3TtsConfig;
pub use predictor::{CodePredictor, CodePredictorCache};
pub use speech_tokenizer::SpeechTokenizerDecoder;
pub use talker::{TalkerCache, TalkerModel};
pub use tokenizer::{SpeakerReference, TtsSpecialTokens, TtsTokenizer};

use candle_core::{DType, IndexOp, Tensor, D};
use candle_nn::VarBuilder;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, info};

use crate::backends::DeviceProfile;
use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use crate::models::shared::attention::paged::{default_kv_page_size, KvCacheQuantization};

const NEWLINE_TOKEN_ID: u32 = 198;
const ENV_QWEN_TTS_CUDA_CHUNKED_CODEC_STREAM: &str = "IZWI_QWEN_TTS_CUDA_CHUNKED_CODEC_STREAM";
const MIN_QWEN_TTS_TOKENS_BEFORE_EOS: usize = 8;

/// Runtime generation settings for semantic token sampling.
#[derive(Debug, Clone)]
pub struct TtsGenerationParams {
    /// Semantic token temperature. <= 0 means greedy.
    pub temperature: f32,
    /// Top-p nucleus sampling threshold.
    pub top_p: f32,
    /// Top-k sampling cutoff. 0 means disabled.
    pub top_k: usize,
    /// Repetition penalty for previously sampled semantic tokens.
    pub repetition_penalty: f32,
    /// Maximum generated codec frames. `0` means auto (model maximum).
    pub max_frames: usize,
}

impl Default for TtsGenerationParams {
    fn default() -> Self {
        // Mirrors the official generation_config defaults.
        Self {
            temperature: 0.9,
            top_p: 1.0,
            top_k: 50,
            repetition_penalty: 1.05,
            max_frames: crate::model::ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES,
        }
    }
}

/// Runtime configuration for progressive TTS audio emission.
#[derive(Debug, Clone, Copy)]
pub struct TtsStreamingConfig {
    /// Minimum codec frames before emitting first audio chunk.
    pub min_frames_before_stream: usize,
    /// Minimum newly generated codec frames before decoding again.
    pub decode_interval_frames: usize,
    /// Keep a small decode lookahead to reduce boundary artifacts.
    pub decode_lookahead_frames: usize,
}

impl Default for TtsStreamingConfig {
    fn default() -> Self {
        Self {
            min_frames_before_stream: 6,
            decode_interval_frames: 4,
            decode_lookahead_frames: 2,
        }
    }
}

impl TtsStreamingConfig {
    /// Decode audio only at completion.
    ///
    /// This avoids repeatedly decoding the full codec timeline for non-streaming
    /// generation paths, which materially improves long-form performance.
    pub fn final_only() -> Self {
        Self {
            min_frames_before_stream: usize::MAX,
            decode_interval_frames: usize::MAX,
            decode_lookahead_frames: 0,
        }
    }
}

struct ProgressiveStreamState<'a> {
    config: TtsStreamingConfig,
    emitted_frames: usize,
    emitted_samples: usize,
    decode_raw_token_scratch: Vec<Vec<u32>>,
    on_chunk: &'a mut dyn FnMut(Vec<f32>) -> Result<()>,
}

/// Incremental decode state for scheduler-integrated Qwen3 TTS execution.
pub struct TtsDecodeState {
    talker_cache: TalkerCache,
    predictor_cache: CodePredictorCache,
    text_vocab_size: u32,
    acoustic_vocab_size: u32,
    semantic_vocab_size: u32,
    trailing_text_hidden: Tensor,
    trailing_text_len: usize,
    tts_pad_embed: Tensor,
    max_frames: usize,
    frame_idx: usize,
    offset: usize,
    all_code_groups: Vec<Vec<u32>>,
    semantic_history: Vec<u32>,
    last_hidden: Tensor,
    last_logits: Tensor,
    rng: SimpleRng,
    params: TtsGenerationParams,
    stream_config: TtsStreamingConfig,
    emitted_frames: usize,
    emitted_samples: usize,
    decode_raw_token_scratch: Vec<Vec<u32>>,
    finished: bool,
}

#[derive(Debug, Clone)]
pub struct TtsDecodeStep {
    pub samples: Vec<f32>,
    pub frames_generated: usize,
    pub finished: bool,
}

/// Batch input for CustomVoice (preset speaker) generation.
#[derive(Debug, Clone)]
pub struct BatchedSpeakerRequest {
    pub text: String,
    pub speaker: String,
    pub language: Option<String>,
    pub instruct: Option<String>,
    pub params: TtsGenerationParams,
}

impl TtsGenerationParams {
    /// Convert external generation config to TTS sampling params.
    pub fn from_generation_config(cfg: &crate::runtime::GenerationConfig) -> Self {
        let opts = &cfg.options;
        Self {
            temperature: opts.temperature.max(0.0),
            top_p: opts.top_p.clamp(0.0, 1.0),
            top_k: if opts.top_k == 0 { 50 } else { opts.top_k },
            repetition_penalty: opts.repetition_penalty.max(1.0),
            max_frames: if opts.max_tokens == 0 {
                crate::model::ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES
            } else {
                opts.max_tokens
                    .clamp(16, crate::model::ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES)
            },
        }
    }
}

/// Qwen3-TTS Model for speech synthesis
pub struct Qwen3TtsModel {
    /// Device configuration
    device: DeviceProfile,
    /// Primary transformer data type for inference.
    dtype: DType,
    /// Data type used by the acoustic code predictor.
    code_predictor_dtype: DType,
    /// Data type used by the speech tokenizer decoder.
    speech_tokenizer_dtype: DType,
    /// Tokenizer for text and codec tokens
    tokenizer: TtsTokenizer,
    /// Special token IDs
    specials: TtsSpecialTokens,
    /// Main talker (LLM) model
    talker: TalkerModel,
    /// Code predictor for multi-codebook generation
    code_predictor: CodePredictor,
    /// Speech tokenizer decoder for codec to audio conversion
    speech_tokenizer: SpeechTokenizerDecoder,
    /// Model configuration
    config: Qwen3TtsConfig,
    /// Decode-time KV page size.
    kv_page_size: usize,
    /// KV cache quantization mode.
    kv_quantization: KvCacheQuantization,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Qwen3TtsDTypePlan {
    talker: DType,
    code_predictor: DType,
    speech_tokenizer: DType,
}

fn select_qwen3_tts_dtypes(
    device: &DeviceProfile,
    dtype_override: Option<&str>,
    is_custom_voice_model: bool,
    is_voice_clone_model: bool,
) -> Result<Qwen3TtsDTypePlan> {
    if let Some(raw) = dtype_override.map(str::trim).filter(|raw| !raw.is_empty()) {
        let dtype =
            device.select_model_dtype_checked(ModelFamily::Qwen3Tts, Some(raw), "Qwen3 TTS")?;
        return Ok(Qwen3TtsDTypePlan {
            talker: dtype,
            code_predictor: dtype,
            speech_tokenizer: dtype,
        });
    }

    let legacy_dtype = if is_custom_voice_model || is_voice_clone_model {
        // Voice-clone and CustomVoice generation were historically kept in
        // F32 by default. Preserve that outside CUDA, and keep the decoder at
        // F32 for CUDA unless the user explicitly overrides the dtype.
        DType::F32
    } else if device.kind.is_metal() {
        // Reduce model residency on Apple Silicon while preserving speed.
        DType::F16
    } else {
        device.select_model_dtype(ModelFamily::Qwen3Tts, None)
    };

    if device.kind.is_cuda() {
        let transformer_dtype = device.select_model_dtype(ModelFamily::Qwen3Tts, None);
        let speech_tokenizer_dtype = if is_custom_voice_model || is_voice_clone_model {
            DType::F32
        } else {
            legacy_dtype
        };
        Ok(Qwen3TtsDTypePlan {
            talker: transformer_dtype,
            code_predictor: transformer_dtype,
            speech_tokenizer: speech_tokenizer_dtype,
        })
    } else {
        Ok(Qwen3TtsDTypePlan {
            talker: legacy_dtype,
            code_predictor: legacy_dtype,
            speech_tokenizer: legacy_dtype,
        })
    }
}

fn qwen_tts_uses_cuda_sampling(device: &DeviceProfile) -> bool {
    device.kind.is_cuda()
}

fn qwen_tts_allows_eos(frames_generated: usize) -> bool {
    frames_generated >= MIN_QWEN_TTS_TOKENS_BEFORE_EOS
}

impl Qwen3TtsModel {
    /// Load a Qwen3-TTS model from the specified directory
    pub fn load(
        model_dir: &Path,
        device: DeviceProfile,
        kv_page_size: usize,
        kv_cache_dtype: &str,
    ) -> Result<Self> {
        info!("Loading Qwen3-TTS model from {:?}", model_dir);

        // Load configuration
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
        let config: Qwen3TtsConfig = serde_json::from_str(&config_str)?;

        info!("Model type: {}", config.tts_model_type);
        info!("Model size: {}", config.tts_model_size);

        let model_type_normalized = config
            .tts_model_type
            .trim()
            .to_ascii_lowercase()
            .replace(['-', '_'], "");
        let is_custom_voice_model = model_type_normalized == "customvoice";
        let is_voice_clone_model = model_type_normalized == "base"
            || model_type_normalized == "voiceclone"
            || model_type_normalized == "voicecloning"
            || config.talker_config.spk_id.is_empty();
        let dtype_override = std::env::var("IZWI_QWEN_TTS_DTYPE")
            .ok()
            .or_else(|| std::env::var("IZWI_QWEN_DTYPE").ok());
        let dtype_plan = select_qwen3_tts_dtypes(
            &device,
            dtype_override.as_deref(),
            is_custom_voice_model,
            is_voice_clone_model,
        )?;

        // Load tokenizer
        let specials = TtsSpecialTokens::from_configs(&config, &config.talker_config);
        let tokenizer = TtsTokenizer::load(model_dir, specials.clone(), &config.talker_config)?;

        // Load model weights
        let weights_path = model_dir.join("model.safetensors");
        let talker_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&weights_path),
                dtype_plan.talker,
                &device.device,
            )?
        };
        let code_predictor_vb = if dtype_plan.code_predictor == dtype_plan.talker {
            talker_vb.clone()
        } else {
            unsafe {
                VarBuilder::from_mmaped_safetensors(
                    std::slice::from_ref(&weights_path),
                    dtype_plan.code_predictor,
                    &device.device,
                )?
            }
        };

        // Load talker model
        info!("Loading talker model...");
        let talker = TalkerModel::load(config.talker_config.clone(), talker_vb.pp("talker"))?;

        // Load code predictor
        info!("Loading code predictor...");
        let num_code_groups = config.talker_config.num_code_groups;
        // For 1.7B model, codec embeddings use talker.text_hidden_size (2048)
        // For 0.6B model, codec embeddings use code_predictor.hidden_size (1024)
        // Detect 1.7B by checking if talker.hidden_size differs from code_predictor.hidden_size
        let mut code_predictor_config = config.talker_config.code_predictor_config.clone();
        if config.talker_config.hidden_size != code_predictor_config.hidden_size {
            // 1.7B case: codec embeddings use text_hidden_size dimension
            code_predictor_config.text_hidden_size = Some(config.talker_config.text_hidden_size);
        }
        let code_predictor = CodePredictor::load(
            code_predictor_config,
            code_predictor_vb.pp("talker.code_predictor"),
            num_code_groups,
        )?;

        // Load speech tokenizer decoder
        info!("Loading speech tokenizer decoder...");
        let speech_tokenizer_path = model_dir.join("speech_tokenizer");
        let speech_tokenizer = SpeechTokenizerDecoder::load(
            &speech_tokenizer_path,
            device.device.clone(),
            dtype_plan.speech_tokenizer,
        )?;

        info!(
            "Qwen3-TTS model loaded successfully on {:?} (talker {:?}, predictor {:?}, speech tokenizer {:?})",
            device.kind, dtype_plan.talker, dtype_plan.code_predictor, dtype_plan.speech_tokenizer
        );
        let kv_quantization = KvCacheQuantization::from_dtype_hint(kv_cache_dtype);

        Ok(Self {
            device,
            dtype: dtype_plan.talker,
            code_predictor_dtype: dtype_plan.code_predictor,
            speech_tokenizer_dtype: dtype_plan.speech_tokenizer,
            tokenizer,
            specials,
            talker,
            code_predictor,
            speech_tokenizer,
            config,
            kv_page_size: kv_page_size.max(1),
            kv_quantization,
        })
    }

    /// Generate speech using a preset speaker (CustomVoice mode)
    pub fn generate_with_speaker(
        &self,
        text: &str,
        speaker: &str,
        language: Option<&str>,
        instruct: Option<&str>,
    ) -> Result<Vec<f32>> {
        self.generate_with_speaker_params(
            text,
            speaker,
            language,
            instruct,
            &TtsGenerationParams::default(),
        )
    }

    /// Generate speech with a preset speaker (CustomVoice mode) and explicit sampling params.
    pub fn generate_with_speaker_params(
        &self,
        text: &str,
        speaker: &str,
        language: Option<&str>,
        instruct: Option<&str>,
        params: &TtsGenerationParams,
    ) -> Result<Vec<f32>> {
        info!("Generating speech with speaker: {}", speaker);

        let prompt_ids = self.encode_assistant_prompt_ids(text)?;
        let speaker_id = self.tokenizer.get_speaker_id(speaker).ok_or_else(|| {
            Error::InvalidInput(format!(
                "Unknown speaker '{speaker}'. Available speakers: {}",
                self.tokenizer
                    .available_speakers()
                    .into_iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })?;
        let language_id = self.resolve_language_id(language);
        let instruct_ids = self.encode_instruction_ids(instruct)?;

        debug!(
            "Prompt token length: {}, speaker_id: {}, language_id: {:?}, has_instruction: {}",
            prompt_ids.len(),
            speaker_id,
            language_id,
            instruct_ids.is_some()
        );

        let codec_tokens = self.generate_codec_tokens_conditioned(
            &prompt_ids,
            Some(speaker_id),
            language_id,
            instruct_ids.as_deref(),
            params,
            None,
        )?;

        // Decode to audio using speech tokenizer
        self.codec_to_audio(&codec_tokens)
    }

    /// Generate speech with progressive audio chunk callbacks.
    pub fn generate_with_speaker_params_streaming(
        &self,
        text: &str,
        speaker: &str,
        language: Option<&str>,
        instruct: Option<&str>,
        params: &TtsGenerationParams,
        stream_config: TtsStreamingConfig,
        on_chunk: &mut dyn FnMut(Vec<f32>) -> Result<()>,
    ) -> Result<()> {
        info!("Streaming speech generation with speaker: {}", speaker);

        let prompt_ids = self.encode_assistant_prompt_ids(text)?;
        let speaker_id = self.tokenizer.get_speaker_id(speaker).ok_or_else(|| {
            Error::InvalidInput(format!(
                "Unknown speaker '{speaker}'. Available speakers: {}",
                self.tokenizer
                    .available_speakers()
                    .into_iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })?;
        let language_id = self.resolve_language_id(language);
        let instruct_ids = self.encode_instruction_ids(instruct)?;
        let mut stream_state = ProgressiveStreamState {
            config: stream_config,
            emitted_frames: 0,
            emitted_samples: 0,
            decode_raw_token_scratch: Vec::new(),
            on_chunk,
        };

        let _ = self.generate_codec_tokens_conditioned(
            &prompt_ids,
            Some(speaker_id),
            language_id,
            instruct_ids.as_deref(),
            params,
            Some(&mut stream_state),
        )?;
        Ok(())
    }

    /// Generate speech for a batch of preset-speaker requests.
    pub fn generate_with_speaker_params_batch(
        &self,
        requests: &[BatchedSpeakerRequest],
    ) -> Result<Vec<Vec<f32>>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        #[derive(Debug)]
        struct PreparedRequest {
            index: usize,
            params: TtsGenerationParams,
            prefill_embeds: Tensor,
            prefill_len: usize,
            trailing_text_hidden: Tensor,
            trailing_text_len: usize,
            tts_pad_embed: Tensor,
            max_frames: usize,
        }

        let text_vocab_size = self.tokenizer.text_vocab_size() as u32;
        let acoustic_vocab_size = self.tokenizer.codec_vocab_size() as u32;
        let talker_codec_vocab_size = self.config.talker_config.vocab_size as u32;
        let semantic_vocab_size = talker_codec_vocab_size.saturating_sub(1024);

        let mut prepared = Vec::with_capacity(requests.len());
        for (idx, req) in requests.iter().enumerate() {
            let prompt_ids = self.encode_assistant_prompt_ids(&req.text)?;
            let speaker_id = self.tokenizer.get_speaker_id(&req.speaker).ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Unknown speaker '{}'. Available speakers: {}",
                    req.speaker,
                    self.tokenizer
                        .available_speakers()
                        .into_iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })?;
            let language_id = self.resolve_language_id(req.language.as_deref());
            let instruct_ids = self.encode_instruction_ids(req.instruct.as_deref())?;

            let prefill_embeds = self.build_conditioned_prefill_embeddings(
                &prompt_ids,
                Some(speaker_id),
                language_id,
                instruct_ids.as_deref(),
            )?;
            let prefill_len = prefill_embeds.dim(1)?;

            let context_budget = self
                .config
                .talker_config
                .max_position_embeddings
                .saturating_sub(prefill_len + 1);
            if context_budget == 0 {
                return Err(Error::InferenceError(
                    "Input is too long for model context; no room left for audio generation"
                        .to_string(),
                ));
            }
            let max_frames = if req.params.max_frames == 0 {
                context_budget
            } else {
                req.params.max_frames.max(1).min(context_budget)
            };

            let (trailing_text_hidden, trailing_text_len, tts_pad_embed) =
                self.build_trailing_text_embeddings_from_prompt(&prompt_ids, max_frames)?;

            prepared.push(PreparedRequest {
                index: idx,
                params: req.params.clone(),
                prefill_embeds,
                prefill_len,
                trailing_text_hidden,
                trailing_text_len,
                tts_pad_embed,
                max_frames,
            });
        }

        let mut groups: HashMap<usize, Vec<PreparedRequest>> = HashMap::new();
        for item in prepared {
            groups.entry(item.prefill_len).or_default().push(item);
        }

        let mut outputs: Vec<Option<Vec<f32>>> = vec![None; requests.len()];

        for (_prefill_len, group) in groups {
            let batch_size = group.len();
            if batch_size == 0 {
                continue;
            }

            let embeds: Vec<Tensor> = group.iter().map(|req| req.prefill_embeds.clone()).collect();
            let batch_embeds = Tensor::cat(&embeds, 0)?;

            let mut talker_cache = TalkerCache::with_page_size_and_quantization(
                self.talker.num_layers(),
                self.kv_page_size,
                self.kv_quantization,
            );
            let (last_hidden_batch, last_logits_batch) =
                self.talker
                    .prefill_with_embeds(&batch_embeds, &mut talker_cache, None)?;

            struct DecodeState {
                index: usize,
                params: TtsGenerationParams,
                trailing_text_hidden: Tensor,
                trailing_text_len: usize,
                tts_pad_embed: Tensor,
                max_frames: usize,
                all_code_groups: Vec<Vec<u32>>,
                semantic_history: Vec<u32>,
                last_hidden: Tensor,
                last_logits: Tensor,
                predictor_cache: CodePredictorCache,
                rng: SimpleRng,
                finished: bool,
            }

            let mut states = Vec::with_capacity(batch_size);
            for (batch_idx, req) in group.iter().enumerate() {
                let last_hidden = last_hidden_batch.i(batch_idx)?.unsqueeze(0)?;
                let last_logits = last_logits_batch.i(batch_idx)?.unsqueeze(0)?;
                states.push(DecodeState {
                    index: req.index,
                    params: req.params.clone(),
                    trailing_text_hidden: req.trailing_text_hidden.clone(),
                    trailing_text_len: req.trailing_text_len,
                    tts_pad_embed: req.tts_pad_embed.clone(),
                    max_frames: req.max_frames,
                    all_code_groups: vec![Vec::new(); self.config.talker_config.num_code_groups],
                    semantic_history: Vec::new(),
                    last_hidden,
                    last_logits,
                    predictor_cache: CodePredictorCache::with_page_size_and_quantization(
                        self.code_predictor.num_layers(),
                        self.kv_page_size,
                        self.kv_quantization,
                    ),
                    rng: SimpleRng::new(),
                    finished: false,
                });
            }

            let mut offset = group[0].prefill_len;
            let max_frames = states.iter().map(|s| s.max_frames).max().unwrap_or(0);

            for frame_idx in 0..max_frames {
                let mut step_inputs = Vec::with_capacity(batch_size);
                let mut any_active = false;

                for state in states.iter_mut() {
                    if state.finished || frame_idx >= state.max_frames {
                        step_inputs.push(state.tts_pad_embed.clone());
                        continue;
                    }

                    any_active = true;
                    let allow_eos = qwen_tts_allows_eos(state.all_code_groups[0].len());
                    let semantic_token = sample_semantic(
                        &state.last_logits.i((0, 0))?,
                        semantic_vocab_size,
                        self.specials.codec_eos_token_id,
                        allow_eos,
                        &state.params,
                        &state.semantic_history,
                        &mut state.rng,
                        qwen_tts_uses_cuda_sampling(&self.device),
                    )?;
                    if allow_eos && semantic_token == self.specials.codec_eos_token_id {
                        state.finished = true;
                        debug!(
                            frames_generated = state.all_code_groups[0].len(),
                            device = ?self.device.kind,
                            "Qwen3-TTS batch generation reached semantic EOS"
                        );
                        step_inputs.push(state.tts_pad_embed.clone());
                        continue;
                    }

                    state.semantic_history.push(semantic_token);
                    if state.semantic_history.len() > 256 {
                        let drain = state.semantic_history.len() - 256;
                        state.semantic_history.drain(0..drain);
                    }

                    state.all_code_groups[0].push(text_vocab_size + semantic_token);

                    let semantic_embed = self.talker.get_codec_embedding(semantic_token)?;
                    let acoustic_codes = self.code_predictor.generate_acoustic_codes(
                        &state.last_hidden,
                        &semantic_embed,
                        &mut state.predictor_cache,
                    )?;
                    let acoustic_embed_sum = self
                        .code_predictor
                        .get_acoustic_embeddings_sum(&acoustic_codes)?;
                    for (acoustic_idx, &group_token) in acoustic_codes.iter().enumerate() {
                        let group_idx = acoustic_idx + 1;
                        if group_idx < state.all_code_groups.len() {
                            let combined_token = text_vocab_size
                                + group_token
                                + (group_idx as u32 * acoustic_vocab_size);
                            state.all_code_groups[group_idx].push(combined_token);
                        }
                    }

                    let text_addition = if frame_idx < state.trailing_text_len {
                        state
                            .trailing_text_hidden
                            .i((.., frame_idx..frame_idx + 1, ..))?
                    } else {
                        state.tts_pad_embed.clone()
                    };

                    let step_input = semantic_embed
                        .broadcast_add(&acoustic_embed_sum)?
                        .broadcast_add(&text_addition)?;
                    step_inputs.push(step_input);
                }

                if !any_active {
                    break;
                }

                let batch_input = Tensor::cat(&step_inputs, 0)?;
                let (batch_hidden, batch_logits) = self.talker.generate_step_with_embed(
                    &batch_input,
                    &mut talker_cache,
                    offset,
                )?;

                for (idx, state) in states.iter_mut().enumerate() {
                    state.last_hidden = batch_hidden.i(idx)?.unsqueeze(0)?;
                    state.last_logits = batch_logits.i(idx)?.unsqueeze(0)?;
                }

                offset += 1;
            }

            for state in states {
                let samples = self.codec_to_audio(&state.all_code_groups)?;
                outputs[state.index] = Some(samples);
            }
        }

        Ok(outputs
            .into_iter()
            .map(|out| {
                out.ok_or_else(|| {
                    Error::InferenceError("Missing output for batched request".to_string())
                })
            })
            .collect::<Result<Vec<_>>>()?)
    }

    /// Generate speech with voice cloning
    pub fn generate_with_voice_clone(
        &self,
        text: &str,
        reference: &SpeakerReference,
        language: Option<&str>,
    ) -> Result<Vec<f32>> {
        info!("Generating speech with voice cloning");

        let ref_codec_tokens = self.encode_reference_audio(reference)?;
        if ref_codec_tokens.is_empty() || ref_codec_tokens[0].is_empty() {
            return Err(Error::ModelError(
                "Voice cloning reference encoder produced no conditioning tokens".to_string(),
            ));
        }

        let params = TtsGenerationParams::default();
        let prompt_ids = self.encode_assistant_prompt_ids(text)?;
        let ref_prompt_ids = self.encode_reference_prompt_ids(reference.text.as_str())?;
        let language_id = self.resolve_language_id(language);

        let codec_tokens = self.generate_codec_tokens_voice_clone_conditioned(
            &prompt_ids,
            &ref_prompt_ids,
            &ref_codec_tokens,
            language_id,
            &params,
            None,
        )?;

        // Decode to audio
        self.codec_to_audio(&codec_tokens)
    }

    /// Generate voice-cloned speech with progressive audio chunk callbacks.
    pub fn generate_with_voice_clone_streaming(
        &self,
        text: &str,
        reference: &SpeakerReference,
        language: Option<&str>,
        params: &TtsGenerationParams,
        stream_config: TtsStreamingConfig,
        on_chunk: &mut dyn FnMut(Vec<f32>) -> Result<()>,
    ) -> Result<()> {
        let ref_codec_tokens = self.encode_reference_audio(reference)?;
        if ref_codec_tokens.is_empty() || ref_codec_tokens[0].is_empty() {
            return Err(Error::ModelError(
                "Voice cloning reference encoder produced no conditioning tokens".to_string(),
            ));
        }

        let prompt_ids = self.encode_assistant_prompt_ids(text)?;
        let ref_prompt_ids = self.encode_reference_prompt_ids(reference.text.as_str())?;
        let language_id = self.resolve_language_id(language);
        let mut stream_state = ProgressiveStreamState {
            config: stream_config,
            emitted_frames: 0,
            emitted_samples: 0,
            decode_raw_token_scratch: Vec::new(),
            on_chunk,
        };

        let _ = self.generate_codec_tokens_voice_clone_conditioned(
            &prompt_ids,
            &ref_prompt_ids,
            &ref_codec_tokens,
            language_id,
            params,
            Some(&mut stream_state),
        )?;
        Ok(())
    }

    /// Start incremental decode state for voice-clone synthesis.
    pub fn start_decode_with_voice_clone_params(
        &self,
        text: &str,
        reference: &SpeakerReference,
        language: Option<&str>,
        params: &TtsGenerationParams,
        stream_config: TtsStreamingConfig,
    ) -> Result<TtsDecodeState> {
        let ref_codec_tokens = self.encode_reference_audio(reference)?;
        if ref_codec_tokens.is_empty() || ref_codec_tokens[0].is_empty() {
            return Err(Error::ModelError(
                "Voice cloning reference encoder produced no conditioning tokens".to_string(),
            ));
        }

        let prompt_ids = self.encode_assistant_prompt_ids(text)?;
        let ref_prompt_ids = self.encode_reference_prompt_ids(reference.text.as_str())?;
        let target_text_ids: Vec<u32> = if prompt_ids.len() > 8 {
            prompt_ids[3..prompt_ids.len() - 5].to_vec()
        } else {
            Vec::new()
        };
        let reference_text_ids: Vec<u32> = if ref_prompt_ids.len() > 5 {
            ref_prompt_ids[3..ref_prompt_ids.len() - 2].to_vec()
        } else {
            Vec::new()
        };
        if target_text_ids.is_empty() || reference_text_ids.is_empty() {
            return Err(Error::InvalidInput(
                "Voice cloning requires non-empty target/reference transcript tokens".to_string(),
            ));
        }

        let language_id = self.resolve_language_id(language);
        let base_prefill =
            self.build_conditioned_prefill_embeddings(&[], None, language_id, None)?;
        let tts_pad_embed = self
            .talker
            .get_projected_special_embed(self.specials.tts_pad_token_id)?;
        let tts_eos_embed = self
            .talker
            .get_projected_special_embed(self.specials.tts_eos_token_id)?;
        let (icl_embed, trailing_text_hidden) = self.build_voice_clone_icl_embeddings(
            &target_text_ids,
            &reference_text_ids,
            &ref_codec_tokens,
            &tts_pad_embed,
            &tts_eos_embed,
            false,
        )?;
        let prefill_embeds = Tensor::cat(&[&base_prefill, &icl_embed], 1)?;
        let prefill_len = prefill_embeds.dim(1)?;
        let context_budget = self
            .config
            .talker_config
            .max_position_embeddings
            .saturating_sub(prefill_len + 1);
        if context_budget == 0 {
            return Err(Error::InferenceError(
                "Voice-clone prompt exceeds model context window".to_string(),
            ));
        }

        let resolved_max_frames = if params.max_frames == 0 {
            context_budget
        } else {
            params.max_frames.max(1).min(context_budget)
        };
        let resolved_params = TtsGenerationParams {
            max_frames: resolved_max_frames,
            ..params.clone()
        };
        self.start_decode_from_prefill(
            prefill_embeds,
            trailing_text_hidden,
            tts_pad_embed,
            &resolved_params,
            stream_config,
        )
    }

    /// Generate speech without requiring a preset speaker table.
    ///
    /// This path is used for base checkpoints that do not ship `spk_id`
    /// mappings in config, while still supporting plain text synthesis.
    pub fn generate_with_text_params(
        &self,
        text: &str,
        language: Option<&str>,
        instruct: Option<&str>,
        params: &TtsGenerationParams,
    ) -> Result<Vec<f32>> {
        let prompt_ids = self.encode_assistant_prompt_ids(text)?;
        let language_id = self.resolve_language_id(language);
        let instruct_ids = self.encode_instruction_ids(instruct)?;

        let codec_tokens = self.generate_codec_tokens_conditioned(
            &prompt_ids,
            None,
            language_id,
            instruct_ids.as_deref(),
            params,
            None,
        )?;
        self.codec_to_audio(&codec_tokens)
    }

    /// Generate plain text speech with progressive audio chunk callbacks.
    pub fn generate_with_text_params_streaming(
        &self,
        text: &str,
        language: Option<&str>,
        instruct: Option<&str>,
        params: &TtsGenerationParams,
        stream_config: TtsStreamingConfig,
        on_chunk: &mut dyn FnMut(Vec<f32>) -> Result<()>,
    ) -> Result<()> {
        let prompt_ids = self.encode_assistant_prompt_ids(text)?;
        let language_id = self.resolve_language_id(language);
        let instruct_ids = self.encode_instruction_ids(instruct)?;
        let mut stream_state = ProgressiveStreamState {
            config: stream_config,
            emitted_frames: 0,
            emitted_samples: 0,
            decode_raw_token_scratch: Vec::new(),
            on_chunk,
        };

        let _ = self.generate_codec_tokens_conditioned(
            &prompt_ids,
            None,
            language_id,
            instruct_ids.as_deref(),
            params,
            Some(&mut stream_state),
        )?;
        Ok(())
    }

    /// Start incremental decode state for preset-speaker TTS.
    pub fn start_decode_with_speaker_params(
        &self,
        text: &str,
        speaker: &str,
        language: Option<&str>,
        instruct: Option<&str>,
        params: &TtsGenerationParams,
        stream_config: TtsStreamingConfig,
    ) -> Result<TtsDecodeState> {
        let prompt_ids = self.encode_assistant_prompt_ids(text)?;
        let speaker_id = self.tokenizer.get_speaker_id(speaker).ok_or_else(|| {
            Error::InvalidInput(format!(
                "Unknown speaker '{speaker}'. Available speakers: {}",
                self.tokenizer
                    .available_speakers()
                    .into_iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })?;
        let language_id = self.resolve_language_id(language);
        let instruct_ids = self.encode_instruction_ids(instruct)?;

        let prefill_embeds = self.build_conditioned_prefill_embeddings(
            &prompt_ids,
            Some(speaker_id),
            language_id,
            instruct_ids.as_deref(),
        )?;
        let prefill_len = prefill_embeds.dim(1)?;
        let context_budget = self
            .config
            .talker_config
            .max_position_embeddings
            .saturating_sub(prefill_len + 1);
        if context_budget == 0 {
            return Err(Error::InferenceError(
                "Input is too long for model context; no room left for audio generation"
                    .to_string(),
            ));
        }
        let resolved_max_frames = if params.max_frames == 0 {
            context_budget
        } else {
            params.max_frames.max(1).min(context_budget)
        };
        let resolved_params = TtsGenerationParams {
            max_frames: resolved_max_frames,
            ..params.clone()
        };
        let (trailing_text_hidden, _, tts_pad_embed) =
            self.build_trailing_text_embeddings_from_prompt(&prompt_ids, resolved_max_frames)?;
        self.start_decode_from_prefill(
            prefill_embeds,
            trailing_text_hidden,
            tts_pad_embed,
            &resolved_params,
            stream_config,
        )
    }

    /// Start incremental decode state for text-only TTS (no preset speaker table).
    pub fn start_decode_with_text_params(
        &self,
        text: &str,
        language: Option<&str>,
        instruct: Option<&str>,
        params: &TtsGenerationParams,
        stream_config: TtsStreamingConfig,
    ) -> Result<TtsDecodeState> {
        let prompt_ids = self.encode_assistant_prompt_ids(text)?;
        let language_id = self.resolve_language_id(language);
        let instruct_ids = self.encode_instruction_ids(instruct)?;
        let prefill_embeds = self.build_conditioned_prefill_embeddings(
            &prompt_ids,
            None,
            language_id,
            instruct_ids.as_deref(),
        )?;
        let prefill_len = prefill_embeds.dim(1)?;
        let context_budget = self
            .config
            .talker_config
            .max_position_embeddings
            .saturating_sub(prefill_len + 1);
        if context_budget == 0 {
            return Err(Error::InferenceError(
                "Input is too long for model context; no room left for audio generation"
                    .to_string(),
            ));
        }
        let resolved_max_frames = if params.max_frames == 0 {
            context_budget
        } else {
            params.max_frames.max(1).min(context_budget)
        };
        let resolved_params = TtsGenerationParams {
            max_frames: resolved_max_frames,
            ..params.clone()
        };
        let (trailing_text_hidden, _, tts_pad_embed) =
            self.build_trailing_text_embeddings_from_prompt(&prompt_ids, resolved_max_frames)?;
        self.start_decode_from_prefill(
            prefill_embeds,
            trailing_text_hidden,
            tts_pad_embed,
            &resolved_params,
            stream_config,
        )
    }

    /// Execute one incremental decode step and optionally emit a new audio chunk.
    pub fn tts_decode_step(&self, state: &mut TtsDecodeState) -> Result<TtsDecodeStep> {
        if state.finished {
            return Ok(TtsDecodeStep {
                samples: Vec::new(),
                frames_generated: state.all_code_groups.first().map(|g| g.len()).unwrap_or(0),
                finished: true,
            });
        }

        if state.frame_idx >= state.max_frames {
            state.finished = true;
            let final_samples = self.collect_incremental_audio(state, true)?;
            return Ok(TtsDecodeStep {
                samples: final_samples,
                frames_generated: state.all_code_groups.first().map(|g| g.len()).unwrap_or(0),
                finished: true,
            });
        }

        let step_start = Instant::now();
        let allow_eos = qwen_tts_allows_eos(state.all_code_groups[0].len());
        let semantic_start = Instant::now();
        let semantic_token = sample_semantic(
            &state.last_logits.i((0, 0))?,
            state.semantic_vocab_size,
            self.specials.codec_eos_token_id,
            allow_eos,
            &state.params,
            &state.semantic_history,
            &mut state.rng,
            qwen_tts_uses_cuda_sampling(&self.device),
        )?;
        let semantic_ms = semantic_start.elapsed().as_secs_f64() * 1000.0;

        if allow_eos && semantic_token == self.specials.codec_eos_token_id {
            state.finished = true;
            debug!(
                frames_generated = state.all_code_groups.first().map(|g| g.len()).unwrap_or(0),
                device = ?self.device.kind,
                "Qwen3-TTS decode reached semantic EOS"
            );
            let final_samples = self.collect_incremental_audio(state, true)?;
            return Ok(TtsDecodeStep {
                samples: final_samples,
                frames_generated: state.all_code_groups.first().map(|g| g.len()).unwrap_or(0),
                finished: true,
            });
        }

        state.semantic_history.push(semantic_token);
        if state.semantic_history.len() > 256 {
            let drain = state.semantic_history.len() - 256;
            state.semantic_history.drain(0..drain);
        }

        state.all_code_groups[0].push(state.text_vocab_size + semantic_token);

        let predictor_start = Instant::now();
        let semantic_embed = self.talker.get_codec_embedding(semantic_token)?;
        let acoustic_codes = self.code_predictor.generate_acoustic_codes(
            &state.last_hidden,
            &semantic_embed,
            &mut state.predictor_cache,
        )?;
        let acoustic_embed_sum = self
            .code_predictor
            .get_acoustic_embeddings_sum(&acoustic_codes)?;
        let predictor_ms = predictor_start.elapsed().as_secs_f64() * 1000.0;
        for (acoustic_idx, &group_token) in acoustic_codes.iter().enumerate() {
            let group_idx = acoustic_idx + 1;
            if group_idx < state.all_code_groups.len() {
                let combined_token = state.text_vocab_size
                    + group_token
                    + (group_idx as u32 * state.acoustic_vocab_size);
                state.all_code_groups[group_idx].push(combined_token);
            }
        }

        let text_addition = if state.frame_idx < state.trailing_text_len {
            state
                .trailing_text_hidden
                .i((.., state.frame_idx..state.frame_idx + 1, ..))?
        } else {
            state.tts_pad_embed.clone()
        };

        let step_input = semantic_embed
            .broadcast_add(&acoustic_embed_sum)?
            .broadcast_add(&text_addition)?;
        let talker_start = Instant::now();
        let (new_hidden, new_logits) = self.talker.generate_step_with_embed(
            &step_input,
            &mut state.talker_cache,
            state.offset,
        )?;
        let talker_ms = talker_start.elapsed().as_secs_f64() * 1000.0;
        state.last_hidden = new_hidden;
        state.last_logits = new_logits;
        state.frame_idx = state.frame_idx.saturating_add(1);
        state.offset = state.offset.saturating_add(1);

        let audio_start = Instant::now();
        let mut samples = self.collect_incremental_audio(state, false)?;
        let audio_ms = audio_start.elapsed().as_secs_f64() * 1000.0;
        if state.frame_idx >= state.max_frames {
            state.finished = true;
            let final_samples = self.collect_incremental_audio(state, true)?;
            samples.extend(final_samples);
        }

        if self.device.kind.is_cuda() {
            debug!(
                frame_idx = state.frame_idx,
                semantic_ms,
                predictor_ms,
                talker_ms,
                audio_ms,
                total_ms = step_start.elapsed().as_secs_f64() * 1000.0,
                emitted_samples = samples.len(),
                "Qwen3-TTS CUDA decode step timings"
            );
        }

        Ok(TtsDecodeStep {
            samples,
            frames_generated: state.all_code_groups.first().map(|g| g.len()).unwrap_or(0),
            finished: state.finished,
        })
    }

    fn resolve_language_id(&self, language: Option<&str>) -> Option<u32> {
        let normalized = language.map(str::trim).filter(|s| !s.is_empty());
        match normalized {
            Some(lang) if lang.eq_ignore_ascii_case("auto") => None,
            Some(lang) => Some(self.tokenizer.get_language_id(lang)),
            None => None,
        }
    }

    fn encode_instruction_ids(&self, instruct: Option<&str>) -> Result<Option<Vec<u32>>> {
        let Some(text) = instruct.map(str::trim).filter(|s| !s.is_empty()) else {
            return Ok(None);
        };
        // Align instruction prompt shape with upstream VoiceDesign/CustomVoice API.
        let prompt = format!("<|im_start|>user\n{text}<|im_end|>\n");
        let ids = self.tokenizer.encode_text(&prompt, None)?;
        if ids.is_empty() {
            Ok(None)
        } else {
            Ok(Some(ids))
        }
    }

    fn encode_assistant_prompt_ids(&self, text: &str) -> Result<Vec<u32>> {
        // Mirror upstream prompting:
        // <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
        let prompt = format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n");
        self.tokenizer.encode_text(&prompt, None)
    }

    fn encode_reference_prompt_ids(&self, reference_text: &str) -> Result<Vec<u32>> {
        // Mirror upstream voice-clone reference prompt:
        // <|im_start|>assistant\n{reference_text}<|im_end|>\n
        let prompt = format!("<|im_start|>assistant\n{reference_text}<|im_end|>\n");
        self.tokenizer.encode_text(&prompt, None)
    }

    /// Legacy codec generation path used by not-yet-updated flows (e.g. voice cloning).
    fn generate_codec_tokens_legacy(
        &self,
        input_ids: &[u32],
        max_length: usize,
        params: &TtsGenerationParams,
    ) -> Result<Vec<Vec<u32>>> {
        let mut talker_cache = TalkerCache::with_page_size_and_quantization(
            self.talker.num_layers(),
            self.kv_page_size,
            self.kv_quantization,
        );
        let mut predictor_cache = CodePredictorCache::with_page_size_and_quantization(
            self.code_predictor.num_layers(),
            self.kv_page_size,
            self.kv_quantization,
        );
        let text_vocab_size = self.tokenizer.text_vocab_size() as u32;
        let acoustic_vocab_size = self.tokenizer.codec_vocab_size() as u32;
        // Talker logits are over full codec vocab (semantic + control).
        let talker_codec_vocab_size = self.config.talker_config.vocab_size as u32;
        // Official suppression keeps semantic IDs [0, vocab-1024) and EOS.
        let semantic_vocab_size = talker_codec_vocab_size.saturating_sub(1024);

        // Convert input to tensor
        let input_tensor = Tensor::from_vec(
            input_ids.to_vec(),
            (1, input_ids.len()),
            &self.device.device,
        )?;

        // Initial forward pass through talker
        let mut logits = self
            .talker
            .forward(&input_tensor, 0, Some(&mut talker_cache))?;

        // Collect generated tokens
        let mut all_code_groups: Vec<Vec<u32>> =
            vec![Vec::new(); self.config.talker_config.num_code_groups];
        let mut pos = input_ids.len();
        let context_budget = self
            .config
            .talker_config
            .max_position_embeddings
            .saturating_sub(input_ids.len() + 1);
        if context_budget == 0 {
            return Err(Error::InferenceError(
                "Input is too long for model context; no room left for audio generation"
                    .to_string(),
            ));
        }
        let max_length = if max_length == 0 {
            context_budget
        } else {
            max_length.max(1).min(context_budget)
        };
        let mut rng = SimpleRng::new();
        let mut semantic_history: Vec<u32> = Vec::new();

        for _step in 0..max_length {
            // Get last position logits
            let last_logits = logits.i((0, logits.dim(1)? - 1))?;

            // Sample first codebook token (semantic) from valid semantic ids plus EOS.
            let allow_eos = qwen_tts_allows_eos(all_code_groups[0].len());
            let first_codebook_token = sample_semantic(
                &last_logits,
                semantic_vocab_size,
                self.specials.codec_eos_token_id,
                allow_eos,
                params,
                &semantic_history,
                &mut rng,
                qwen_tts_uses_cuda_sampling(&self.device),
            )?;

            // Check for end of sequence
            if allow_eos && first_codebook_token == self.specials.codec_eos_token_id {
                debug!(
                    frames_generated = all_code_groups[0].len(),
                    device = ?self.device.kind,
                    "Qwen3-TTS generation reached semantic EOS"
                );
                break;
            }
            semantic_history.push(first_codebook_token);
            if semantic_history.len() > 256 {
                let drain = semantic_history.len() - 256;
                semantic_history.drain(0..drain);
            }
            let first_combined_token = text_vocab_size + first_codebook_token;

            // Store semantic token in combined-vocab format.
            all_code_groups[0].push(first_combined_token);

            // Generate remaining codebooks using code predictor
            let first_codebook_tensor =
                Tensor::from_vec(vec![first_codebook_token], (1, 1), &self.device.device)?;
            let predictor_logits = self.code_predictor.forward(
                &first_codebook_tensor,
                pos,
                Some(&mut predictor_cache),
            )?;

            for (acoustic_idx, group_logits) in predictor_logits.iter().enumerate() {
                let group_idx = acoustic_idx + 1;
                if group_idx >= all_code_groups.len() {
                    break;
                }
                let group_token = argmax(&group_logits.i((0, 0))?)?;
                let combined_token =
                    text_vocab_size + group_token + (group_idx as u32 * acoustic_vocab_size);
                all_code_groups[group_idx].push(combined_token);
            }

            // Prepare next input token for talker
            let next_token_tensor =
                Tensor::from_vec(vec![first_combined_token], (1, 1), &self.device.device)?;
            logits = self
                .talker
                .forward(&next_token_tensor, pos, Some(&mut talker_cache))?;
            pos += 1;
        }

        let semantic_len = all_code_groups.first().map(|g| g.len()).unwrap_or(0);
        let semantic_unique = all_code_groups
            .first()
            .map(|g| g.iter().copied().collect::<HashSet<_>>().len())
            .unwrap_or(0);
        let semantic_preview: Vec<u32> = all_code_groups
            .first()
            .map(|g| g.iter().take(24).copied().collect())
            .unwrap_or_default();
        debug!(
            "Voice clone legacy decode stats: semantic_len={}, semantic_unique={}, preview={:?}",
            semantic_len, semantic_unique, semantic_preview
        );

        Ok(all_code_groups)
    }

    /// Official-style CustomVoice generation path:
    /// prefill with fused role/codec/text embeddings, then per-frame semantic+acoustic+trailing-text fusion.
    fn generate_codec_tokens_conditioned(
        &self,
        prompt_ids: &[u32],
        speaker_id: Option<u32>,
        language_id: Option<u32>,
        instruct_ids: Option<&[u32]>,
        params: &TtsGenerationParams,
        stream_state: Option<&mut ProgressiveStreamState<'_>>,
    ) -> Result<Vec<Vec<u32>>> {
        let prefill_embeds = self.build_conditioned_prefill_embeddings(
            prompt_ids,
            speaker_id,
            language_id,
            instruct_ids,
        )?;
        let prefill_len = prefill_embeds.dim(1)?;
        let context_budget = self
            .config
            .talker_config
            .max_position_embeddings
            .saturating_sub(prefill_len + 1);
        if context_budget == 0 {
            return Err(Error::InferenceError(
                "Input is too long for model context; no room left for audio generation"
                    .to_string(),
            ));
        }

        let max_frames = if params.max_frames == 0 {
            context_budget
        } else {
            params.max_frames.max(1).min(context_budget)
        };
        let (trailing_text_hidden, _trailing_text_len, tts_pad_embed) =
            self.build_trailing_text_embeddings_from_prompt(prompt_ids, max_frames)?;

        self.generate_codec_tokens_from_prefill(
            prefill_embeds,
            trailing_text_hidden,
            tts_pad_embed,
            max_frames,
            params,
            stream_state,
        )
    }

    fn generate_codec_tokens_voice_clone_conditioned(
        &self,
        prompt_ids: &[u32],
        ref_prompt_ids: &[u32],
        ref_codec_tokens: &[Vec<u32>],
        language_id: Option<u32>,
        params: &TtsGenerationParams,
        stream_state: Option<&mut ProgressiveStreamState<'_>>,
    ) -> Result<Vec<Vec<u32>>> {
        let target_text_ids: Vec<u32> = if prompt_ids.len() > 8 {
            prompt_ids[3..prompt_ids.len() - 5].to_vec()
        } else {
            Vec::new()
        };
        let reference_text_ids: Vec<u32> = if ref_prompt_ids.len() > 5 {
            ref_prompt_ids[3..ref_prompt_ids.len() - 2].to_vec()
        } else {
            Vec::new()
        };
        if target_text_ids.is_empty() || reference_text_ids.is_empty() {
            return Err(Error::InvalidInput(
                "Voice cloning requires non-empty target/reference transcript tokens".to_string(),
            ));
        }

        let base_prefill =
            self.build_conditioned_prefill_embeddings(&[], None, language_id, None)?;
        let tts_pad_embed = self
            .talker
            .get_projected_special_embed(self.specials.tts_pad_token_id)?;
        let tts_eos_embed = self
            .talker
            .get_projected_special_embed(self.specials.tts_eos_token_id)?;
        let (icl_embed, trailing_text_hidden) = self.build_voice_clone_icl_embeddings(
            &target_text_ids,
            &reference_text_ids,
            ref_codec_tokens,
            &tts_pad_embed,
            &tts_eos_embed,
            false,
        )?;
        let prefill_embeds = Tensor::cat(&[&base_prefill, &icl_embed], 1)?;

        let prefill_len = prefill_embeds.dim(1)?;
        let context_budget = self
            .config
            .talker_config
            .max_position_embeddings
            .saturating_sub(prefill_len + 1);
        if context_budget == 0 {
            return Err(Error::InferenceError(
                "Voice-clone prompt exceeds model context window".to_string(),
            ));
        }
        let max_frames = if params.max_frames == 0 {
            context_budget
        } else {
            params.max_frames.max(1).min(context_budget)
        };

        self.generate_codec_tokens_from_prefill(
            prefill_embeds,
            trailing_text_hidden,
            tts_pad_embed,
            max_frames,
            params,
            stream_state,
        )
    }

    fn build_voice_clone_icl_embeddings(
        &self,
        target_text_ids: &[u32],
        reference_text_ids: &[u32],
        ref_codec_tokens: &[Vec<u32>],
        tts_pad_embed: &Tensor,
        tts_eos_embed: &Tensor,
        non_streaming_mode: bool,
    ) -> Result<(Tensor, Tensor)> {
        let mut all_text_ids = Vec::with_capacity(reference_text_ids.len() + target_text_ids.len());
        all_text_ids.extend_from_slice(reference_text_ids);
        all_text_ids.extend_from_slice(target_text_ids);
        let text_embed = self.talker.get_projected_text_embeddings(&all_text_ids)?;
        let text_embed = Tensor::cat(&[&text_embed, tts_eos_embed], 1)?;

        let codec_embed = self.build_ref_codec_embeddings(ref_codec_tokens)?;

        let text_lens = text_embed.dim(1)?;
        let codec_lens = codec_embed.dim(1)?;
        if codec_lens == 0 {
            return Err(Error::ModelError(
                "Reference codec conditioning is empty".to_string(),
            ));
        }

        if non_streaming_mode {
            let codec_pad_ids = vec![self.specials.codec_pad_id; text_lens];
            let codec_pad_embed = self.talker.get_codec_embedding_batch(&codec_pad_ids)?;
            let icl_input = text_embed.broadcast_add(&codec_pad_embed)?;
            let icl_input =
                Tensor::cat(&[&icl_input, &codec_embed.broadcast_add(tts_pad_embed)?], 1)?;
            return Ok((icl_input, tts_pad_embed.clone()));
        }

        if text_lens > codec_lens {
            let text_prefix = text_embed.i((.., ..codec_lens, ..))?;
            let trailing = text_embed.i((.., codec_lens.., ..))?;
            let icl_input = text_prefix.broadcast_add(&codec_embed)?;
            Ok((icl_input, trailing))
        } else {
            let mut padded_parts: Vec<Tensor> = vec![text_embed];
            for _ in 0..codec_lens.saturating_sub(text_lens) {
                padded_parts.push(tts_pad_embed.clone());
            }
            let padded_refs: Vec<&Tensor> = padded_parts.iter().collect();
            let padded_text = Tensor::cat(&padded_refs, 1)?;
            let icl_input = padded_text.broadcast_add(&codec_embed)?;
            Ok((icl_input, tts_pad_embed.clone()))
        }
    }

    fn build_ref_codec_embeddings(&self, ref_codec_tokens: &[Vec<u32>]) -> Result<Tensor> {
        let num_code_groups = self.config.talker_config.num_code_groups;
        if ref_codec_tokens.len() < num_code_groups {
            return Err(Error::InvalidInput(format!(
                "Reference codec groups mismatch: got {}, expected at least {}",
                ref_codec_tokens.len(),
                num_code_groups
            )));
        }

        let num_acoustic_groups = self.code_predictor.num_acoustic_groups();
        let usable_groups = (1 + num_acoustic_groups).min(num_code_groups);
        let mut frame_len = usize::MAX;
        for group in ref_codec_tokens.iter().take(usable_groups) {
            frame_len = frame_len.min(group.len());
        }
        if frame_len == usize::MAX || frame_len == 0 {
            return Err(Error::ModelError(
                "Reference codec conditioning has no usable frames".to_string(),
            ));
        }

        let max_ref_frames = 320usize;
        frame_len = frame_len.min(max_ref_frames);

        let codec_vocab = self.tokenizer.codec_vocab_size() as u32;
        let mut semantic_codes = Vec::with_capacity(frame_len);
        let mut acoustic_embed_steps = Vec::with_capacity(frame_len);

        for frame_idx in 0..frame_len {
            semantic_codes.push(ref_codec_tokens[0][frame_idx] % codec_vocab);

            let mut acoustic_codes = Vec::with_capacity(num_acoustic_groups);
            for group_idx in 0..num_acoustic_groups {
                let source_group = group_idx + 1;
                let code = ref_codec_tokens
                    .get(source_group)
                    .and_then(|group| group.get(frame_idx))
                    .copied()
                    .unwrap_or(0)
                    % codec_vocab;
                acoustic_codes.push(code);
            }
            acoustic_embed_steps.push(
                self.code_predictor
                    .get_acoustic_embeddings_sum(&acoustic_codes)?,
            );
        }

        let semantic_embed = self.talker.get_codec_embedding_batch(&semantic_codes)?;
        let acoustic_refs: Vec<&Tensor> = acoustic_embed_steps.iter().collect();
        let acoustic_embed = Tensor::cat(&acoustic_refs, 1)?;
        let codec_embed = semantic_embed.broadcast_add(&acoustic_embed)?;

        let codec_bos = self
            .talker
            .get_codec_embedding_batch(&[self.specials.codec_bos_id])?;
        Tensor::cat(&[&codec_bos, &codec_embed], 1).map_err(Error::from)
    }

    fn generate_codec_tokens_from_prefill(
        &self,
        prefill_embeds: Tensor,
        trailing_text_hidden: Tensor,
        tts_pad_embed: Tensor,
        max_frames: usize,
        params: &TtsGenerationParams,
        mut stream_state: Option<&mut ProgressiveStreamState<'_>>,
    ) -> Result<Vec<Vec<u32>>> {
        let stream_config = stream_state
            .as_ref()
            .map(|s| s.config)
            .unwrap_or_else(TtsStreamingConfig::final_only);
        let mut state = self.start_decode_from_prefill(
            prefill_embeds,
            trailing_text_hidden,
            tts_pad_embed,
            &TtsGenerationParams {
                max_frames,
                ..params.clone()
            },
            stream_config,
        )?;

        loop {
            let step = self.tts_decode_step(&mut state)?;
            if let Some(progressive) = stream_state.as_deref_mut() {
                if !step.samples.is_empty() {
                    (progressive.on_chunk)(step.samples)?;
                }
            }
            if step.finished {
                break;
            }
        }

        Ok(state.all_code_groups)
    }

    fn start_decode_from_prefill(
        &self,
        prefill_embeds: Tensor,
        trailing_text_hidden: Tensor,
        tts_pad_embed: Tensor,
        params: &TtsGenerationParams,
        stream_config: TtsStreamingConfig,
    ) -> Result<TtsDecodeState> {
        let mut talker_cache = TalkerCache::with_page_size_and_quantization(
            self.talker.num_layers(),
            self.kv_page_size,
            self.kv_quantization,
        );
        let predictor_cache = CodePredictorCache::with_page_size_and_quantization(
            self.code_predictor.num_layers(),
            self.kv_page_size,
            self.kv_quantization,
        );
        let text_vocab_size = self.tokenizer.text_vocab_size() as u32;
        let acoustic_vocab_size = self.tokenizer.codec_vocab_size() as u32;
        let talker_codec_vocab_size = self.config.talker_config.vocab_size as u32;
        let semantic_vocab_size = talker_codec_vocab_size.saturating_sub(1024);

        let prefill_len = prefill_embeds.dim(1)?;
        let trailing_text_len = trailing_text_hidden.dim(1)?;
        let prefill_start = Instant::now();
        let (last_hidden, last_logits) =
            self.talker
                .prefill_with_embeds(&prefill_embeds, &mut talker_cache, None)?;
        if self.device.kind.is_cuda() {
            debug!(
                prefill_len,
                trailing_text_len,
                talker_dtype = ?self.dtype,
                predictor_dtype = ?self.code_predictor_dtype,
                speech_tokenizer_dtype = ?self.speech_tokenizer_dtype,
                prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0,
                "Qwen3-TTS CUDA prefill timings"
            );
        }

        Ok(TtsDecodeState {
            talker_cache,
            predictor_cache,
            text_vocab_size,
            acoustic_vocab_size,
            semantic_vocab_size,
            trailing_text_hidden,
            trailing_text_len,
            tts_pad_embed,
            max_frames: params.max_frames.max(1),
            frame_idx: 0,
            offset: prefill_len,
            all_code_groups: vec![Vec::new(); self.config.talker_config.num_code_groups],
            semantic_history: Vec::new(),
            last_hidden,
            last_logits,
            rng: SimpleRng::new(),
            params: params.clone(),
            stream_config,
            emitted_frames: 0,
            emitted_samples: 0,
            decode_raw_token_scratch: Vec::new(),
            finished: false,
        })
    }

    fn collect_incremental_audio(
        &self,
        state: &mut TtsDecodeState,
        force: bool,
    ) -> Result<Vec<f32>> {
        let total_frames = state.all_code_groups.first().map(|g| g.len()).unwrap_or(0);
        if total_frames == 0 {
            return Ok(Vec::new());
        }

        if !force {
            if total_frames < state.stream_config.min_frames_before_stream {
                return Ok(Vec::new());
            }
            let newly_generated = total_frames.saturating_sub(state.emitted_frames);
            if newly_generated < state.stream_config.decode_interval_frames {
                return Ok(Vec::new());
            }
        }

        let lookahead = if force {
            0
        } else {
            state.stream_config.decode_lookahead_frames
        };
        let target_frames = total_frames.saturating_sub(lookahead);
        if target_frames <= state.emitted_frames {
            return Ok(Vec::new());
        }

        for group in &state.all_code_groups {
            if group.len() < target_frames {
                return Ok(Vec::new());
            }
        }

        if self.device.kind.is_cuda() && !force && qwen_tts_cuda_chunked_codec_stream_enabled() {
            let (samples, emitted_frames, emitted_samples) = self.decode_cuda_stream_chunk(
                &state.all_code_groups,
                state.emitted_frames,
                state.emitted_samples,
                target_frames,
                state.stream_config,
                &mut state.decode_raw_token_scratch,
            )?;
            state.emitted_frames = emitted_frames;
            state.emitted_samples = emitted_samples;
            return Ok(samples);
        }

        self.fill_raw_codec_scratch(
            &state.all_code_groups,
            target_frames,
            &mut state.decode_raw_token_scratch,
        )?;
        let decoded = self.decode_raw_codec_tokens(&state.decode_raw_token_scratch)?;
        if decoded.len() <= state.emitted_samples {
            state.emitted_frames = target_frames;
            return Ok(Vec::new());
        }

        let new_samples = decoded[state.emitted_samples..].to_vec();
        state.emitted_samples = decoded.len();
        state.emitted_frames = target_frames;
        Ok(new_samples)
    }

    fn maybe_emit_progressive_audio_chunk(
        &self,
        codec_groups: &[Vec<u32>],
        state: &mut ProgressiveStreamState<'_>,
        force: bool,
    ) -> Result<()> {
        let total_frames = codec_groups.first().map(|g| g.len()).unwrap_or(0);
        if total_frames == 0 {
            return Ok(());
        }

        if !force {
            if total_frames < state.config.min_frames_before_stream {
                return Ok(());
            }
            let newly_generated = total_frames.saturating_sub(state.emitted_frames);
            if newly_generated < state.config.decode_interval_frames {
                return Ok(());
            }
        }

        let lookahead = if force {
            0
        } else {
            state.config.decode_lookahead_frames
        };
        let target_frames = total_frames.saturating_sub(lookahead);
        if target_frames <= state.emitted_frames {
            return Ok(());
        }

        for group in codec_groups {
            if group.len() < target_frames {
                return Ok(());
            }
        }

        if self.device.kind.is_cuda() && !force && qwen_tts_cuda_chunked_codec_stream_enabled() {
            let (new_samples, emitted_frames, emitted_samples) = self.decode_cuda_stream_chunk(
                codec_groups,
                state.emitted_frames,
                state.emitted_samples,
                target_frames,
                state.config,
                &mut state.decode_raw_token_scratch,
            )?;
            state.emitted_frames = emitted_frames;
            state.emitted_samples = emitted_samples;
            if !new_samples.is_empty() {
                (state.on_chunk)(new_samples)?;
            }
            return Ok(());
        }

        self.fill_raw_codec_scratch(
            codec_groups,
            target_frames,
            &mut state.decode_raw_token_scratch,
        )?;
        let decoded = self.decode_raw_codec_tokens(&state.decode_raw_token_scratch)?;
        if decoded.len() <= state.emitted_samples {
            state.emitted_frames = target_frames;
            return Ok(());
        }

        let new_samples = decoded[state.emitted_samples..].to_vec();
        state.emitted_samples = decoded.len();
        state.emitted_frames = target_frames;
        if !new_samples.is_empty() {
            (state.on_chunk)(new_samples)?;
        }

        Ok(())
    }

    fn build_conditioned_prefill_embeddings(
        &self,
        prompt_ids: &[u32],
        speaker_id: Option<u32>,
        language_id: Option<u32>,
        instruct_ids: Option<&[u32]>,
    ) -> Result<Tensor> {
        let role_prefix = self.talker.get_projected_text_embeddings(&[
            self.specials.im_start_token_id,
            self.specials.assistant_token_id,
            NEWLINE_TOKEN_ID,
        ])?;

        let mut codec_prefix_ids = Vec::new();
        if let Some(language_id) = language_id {
            codec_prefix_ids.extend_from_slice(&[
                self.specials.codec_think_id,
                self.specials.codec_think_bos_id,
                language_id,
                self.specials.codec_think_eos_id,
            ]);
        } else {
            codec_prefix_ids.extend_from_slice(&[
                self.specials.codec_nothink_id,
                self.specials.codec_think_bos_id,
                self.specials.codec_think_eos_id,
            ]);
        }
        if let Some(speaker_id) = speaker_id {
            codec_prefix_ids.push(speaker_id);
        }
        codec_prefix_ids.push(self.specials.codec_pad_id);
        codec_prefix_ids.push(self.specials.codec_bos_id);

        let prefix_len = codec_prefix_ids.len();
        if prefix_len < 2 {
            return Err(Error::InferenceError(
                "Invalid codec prefix while building conditioned prefill".to_string(),
            ));
        }

        let codec_prefix = self.talker.get_codec_embedding_batch(&codec_prefix_ids)?;
        let codec_without_last = codec_prefix.i((.., ..prefix_len - 1, ..))?;

        let mut tts_overlay_ids = vec![self.specials.tts_pad_token_id; prefix_len - 2];
        tts_overlay_ids.push(self.specials.tts_bos_token_id);
        let tts_overlay = self
            .talker
            .get_projected_text_embeddings(&tts_overlay_ids)?;
        let codec_hidden = tts_overlay.broadcast_add(&codec_without_last)?;

        let mut hidden = Tensor::cat(&[&role_prefix, &codec_hidden], 1)?;

        if let Some(instruct_ids) = instruct_ids {
            if !instruct_ids.is_empty() {
                let instruct_hidden = self.talker.get_projected_text_embeddings(instruct_ids)?;
                hidden = Tensor::cat(&[&instruct_hidden, &hidden], 1)?;
            }
        }

        let first_text_id = if prompt_ids.len() > 3 {
            Some(prompt_ids[3])
        } else {
            prompt_ids.first().copied()
        };

        if let Some(first_text_id) = first_text_id {
            let first_text_proj = self
                .talker
                .get_projected_text_embeddings(&[first_text_id])?;
            let codec_bos_embed = codec_prefix.i((.., prefix_len - 1..prefix_len, ..))?;
            let first_combined = first_text_proj.broadcast_add(&codec_bos_embed)?;
            hidden = Tensor::cat(&[&hidden, &first_combined], 1)?;
        }

        Ok(hidden)
    }

    fn build_trailing_text_embeddings_from_prompt(
        &self,
        prompt_ids: &[u32],
        max_frames: usize,
    ) -> Result<(Tensor, usize, Tensor)> {
        // Upstream uses prompt layout:
        // [role(3), first_text(1), trailing_text(...), trailer(5)]
        let trailing_ids: &[u32] = if prompt_ids.len() > 9 {
            &prompt_ids[4..prompt_ids.len() - 5]
        } else if prompt_ids.len() > 4 {
            &prompt_ids[4..]
        } else {
            &[]
        };

        let trailing = if !trailing_ids.is_empty() {
            let trailing_end = max_frames.min(trailing_ids.len());
            let remaining = self
                .talker
                .get_projected_text_embeddings(&trailing_ids[..trailing_end])?;
            let eos_embed = self
                .talker
                .get_projected_special_embed(self.specials.tts_eos_token_id)?;
            Tensor::cat(&[&remaining, &eos_embed], 1)?
        } else {
            self.talker
                .get_projected_special_embed(self.specials.tts_eos_token_id)?
        };
        let trailing_len = trailing.dim(1)?;
        let tts_pad = self
            .talker
            .get_projected_special_embed(self.specials.tts_pad_token_id)?;
        Ok((trailing, trailing_len, tts_pad))
    }

    /// Encode reference audio to codec tokens for voice cloning
    fn encode_reference_audio(&self, reference: &SpeakerReference) -> Result<Vec<Vec<u32>>> {
        let codec_tokens = self
            .speech_tokenizer
            .encode_reference_audio(&reference.audio_samples, reference.sample_rate)?;
        let num_groups = codec_tokens.len();
        let num_frames = codec_tokens.first().map(|g| g.len()).unwrap_or(0);
        debug!(
            "Reference audio encoded for voice cloning ({} groups x {} frames, transcript chars: {})",
            num_groups,
            num_frames,
            reference.text.len()
        );
        Ok(codec_tokens)
    }

    /// Convert codec tokens to audio waveform
    fn codec_to_audio(&self, codec_tokens: &[Vec<u32>]) -> Result<Vec<f32>> {
        let target_frames = codec_tokens.first().map(|g| g.len()).unwrap_or(0);
        let mut raw_codec_tokens: Vec<Vec<u32>> = Vec::new();
        self.fill_raw_codec_scratch(codec_tokens, target_frames, &mut raw_codec_tokens)?;
        self.decode_raw_codec_tokens(&raw_codec_tokens)
    }

    fn decode_cuda_stream_chunk(
        &self,
        codec_tokens: &[Vec<u32>],
        emitted_frames: usize,
        emitted_samples: usize,
        target_frames: usize,
        stream_config: TtsStreamingConfig,
        scratch: &mut Vec<Vec<u32>>,
    ) -> Result<(Vec<f32>, usize, usize)> {
        let context_frames = emitted_frames.min(
            stream_config
                .decode_lookahead_frames
                .max(stream_config.decode_interval_frames)
                .max(4),
        );
        let start_frame = emitted_frames.saturating_sub(context_frames);
        let chunk_frames = target_frames.saturating_sub(start_frame);
        if chunk_frames == 0 {
            return Ok((Vec::new(), emitted_frames, emitted_samples));
        }

        self.fill_raw_codec_scratch_range(codec_tokens, start_frame, target_frames, scratch)?;
        let decoded = self.decode_raw_codec_tokens(scratch)?;
        let skip_samples = decoded
            .len()
            .saturating_mul(emitted_frames.saturating_sub(start_frame))
            / chunk_frames;
        let new_samples = decoded.get(skip_samples..).unwrap_or(&[]).to_vec();
        let emitted_samples = emitted_samples.saturating_add(new_samples.len());
        Ok((new_samples, target_frames, emitted_samples))
    }

    fn fill_raw_codec_scratch(
        &self,
        codec_tokens: &[Vec<u32>],
        target_frames: usize,
        scratch: &mut Vec<Vec<u32>>,
    ) -> Result<()> {
        self.fill_raw_codec_scratch_range(codec_tokens, 0, target_frames, scratch)
    }

    fn fill_raw_codec_scratch_range(
        &self,
        codec_tokens: &[Vec<u32>],
        start_frame: usize,
        end_frame: usize,
        scratch: &mut Vec<Vec<u32>>,
    ) -> Result<()> {
        if codec_tokens.is_empty() || end_frame <= start_frame {
            scratch.clear();
            return Ok(());
        }

        let target_frames = end_frame - start_frame;
        let text_vocab_size = self.tokenizer.text_vocab_size() as u32;
        let codec_vocab_size = self.tokenizer.codec_vocab_size() as u32;

        if scratch.len() != codec_tokens.len() {
            scratch.resize_with(codec_tokens.len(), Vec::new);
        }

        for (group_idx, group_tokens) in codec_tokens.iter().enumerate() {
            if group_tokens.len() < end_frame {
                return Err(Error::InvalidInput(
                    "Insufficient codec frames for requested decode slice".to_string(),
                ));
            }
            let raw_tokens = &mut scratch[group_idx];
            raw_tokens.clear();
            raw_tokens.reserve(target_frames);

            for &token in group_tokens.iter().take(end_frame).skip(start_frame) {
                raw_tokens.push(raw_codec_token(
                    token,
                    group_idx,
                    text_vocab_size,
                    codec_vocab_size,
                ));
            }
        }
        Ok(())
    }

    fn decode_raw_codec_tokens(&self, raw_codec_tokens: &[Vec<u32>]) -> Result<Vec<f32>> {
        if raw_codec_tokens.is_empty() || raw_codec_tokens[0].is_empty() {
            return Ok(Vec::new());
        }
        let mut audio = self.speech_tokenizer.decode(raw_codec_tokens)?;
        normalize_audio(&mut audio);
        Ok(audio)
    }

    /// List available preset speakers
    pub fn available_speakers(&self) -> Vec<&String> {
        self.tokenizer.available_speakers()
    }

    /// List available languages
    pub fn available_languages(&self) -> Vec<&String> {
        self.tokenizer.available_languages()
    }

    /// Get the model configuration
    pub fn config(&self) -> &Qwen3TtsConfig {
        &self.config
    }

    /// Get the device
    pub fn device(&self) -> &DeviceProfile {
        &self.device
    }
}

fn raw_codec_token(
    token: u32,
    group_idx: usize,
    text_vocab_size: u32,
    codec_vocab_size: u32,
) -> u32 {
    if group_idx == 0 {
        if token >= text_vocab_size {
            token - text_vocab_size
        } else {
            token
        }
    } else {
        let offset = text_vocab_size + (group_idx as u32 * codec_vocab_size);
        if token >= offset {
            token - offset
        } else {
            token
        }
    }
}

pub(crate) fn qwen_tts_cuda_chunked_codec_stream_enabled() -> bool {
    qwen_tts_cuda_chunked_codec_stream_enabled_from(
        std::env::var(ENV_QWEN_TTS_CUDA_CHUNKED_CODEC_STREAM)
            .ok()
            .as_deref(),
    )
}

fn qwen_tts_cuda_chunked_codec_stream_enabled_from(raw: Option<&str>) -> bool {
    matches!(
        raw.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

/// Argmax sampling for greedy decoding
fn argmax(logits: &Tensor) -> Result<u32> {
    let logits = match logits.rank() {
        1 => logits.clone(),
        2 => {
            let (rows, _cols) = logits.dims2()?;
            if rows != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected Qwen3-TTS logits shape for argmax: {:?}",
                    logits.shape().dims()
                )));
            }
            logits.i(0)?
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected Qwen3-TTS logits rank for argmax: {rank}"
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

fn logit_scalar_f32(logits: &Tensor, idx: u32) -> Result<f32> {
    logits
        .i(idx as usize)?
        .to_dtype(DType::F32)?
        .to_scalar::<f32>()
        .map_err(Error::from)
}

fn argmax_semantic(
    logits: &Tensor,
    semantic_vocab_size: u32,
    eos_token_id: u32,
    allow_eos: bool,
) -> Result<u32> {
    let logits = match logits.rank() {
        1 => logits.clone(),
        2 => {
            let (rows, _cols) = logits.dims2()?;
            if rows != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected Qwen3-TTS semantic logits shape: {:?}",
                    logits.shape().dims()
                )));
            }
            logits.i(0)?
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected Qwen3-TTS semantic logits rank: {rank}"
            )))
        }
    };

    let vocab_len = logits.dim(0)?;
    let semantic_len = (semantic_vocab_size as usize).min(vocab_len);
    let mut best_idx = if semantic_len > 0 {
        let semantic_logits = logits.narrow(0, 0, semantic_len)?;
        Some(argmax(&semantic_logits)?)
    } else {
        None
    };
    let best_val = if let Some(idx) = best_idx {
        logit_scalar_f32(&logits, idx)?
    } else {
        f32::NEG_INFINITY
    };

    let eos_idx = eos_token_id as usize;
    if allow_eos && eos_idx < vocab_len && eos_idx >= semantic_len {
        let eos_val = logit_scalar_f32(&logits, eos_token_id)?;
        if eos_val > best_val {
            best_idx = Some(eos_token_id);
        }
    }

    if let Some(idx) = best_idx {
        Ok(idx)
    } else if allow_eos {
        Ok(eos_token_id)
    } else {
        Ok(0)
    }
}

fn argmax_semantic_reference(
    logits: &Tensor,
    semantic_vocab_size: u32,
    eos_token_id: u32,
    allow_eos: bool,
) -> Result<u32> {
    let logits = logits.to_dtype(DType::F32)?;
    let logits = match logits.rank() {
        1 => logits,
        2 => {
            let (rows, _cols) = logits.dims2()?;
            if rows != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected Qwen3-TTS semantic logits shape: {:?}",
                    logits.shape().dims()
                )));
            }
            logits.i(0)?
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected Qwen3-TTS semantic logits rank: {rank}"
            )))
        }
    };
    let values = logits.to_vec1::<f32>()?;
    let mut max_idx: Option<usize> = None;
    let mut max_val = f32::NEG_INFINITY;

    for (idx, &val) in values.iter().enumerate() {
        let token_id = idx as u32;
        let allowed = token_id < semantic_vocab_size || (allow_eos && token_id == eos_token_id);
        if !allowed {
            continue;
        }
        if val > max_val {
            max_val = val;
            max_idx = Some(idx);
        }
    }

    if let Some(idx) = max_idx {
        Ok(idx as u32)
    } else if allow_eos {
        Ok(eos_token_id)
    } else {
        Ok(0)
    }
}

fn sample_semantic(
    logits: &Tensor,
    semantic_vocab_size: u32,
    eos_token_id: u32,
    allow_eos: bool,
    params: &TtsGenerationParams,
    history: &[u32],
    rng: &mut SimpleRng,
    prefer_device_sampling: bool,
) -> Result<u32> {
    if !prefer_device_sampling {
        return sample_semantic_reference(
            logits,
            semantic_vocab_size,
            eos_token_id,
            allow_eos,
            params,
            history,
            rng,
        );
    }

    // Greedy fallback stays on device until the selected scalar is copied back.
    if params.temperature <= 1e-5 {
        return argmax_semantic(logits, semantic_vocab_size, eos_token_id, allow_eos);
    }

    let logits = logits.to_dtype(DType::F32)?;
    let logits = match logits.rank() {
        1 => logits,
        2 => {
            let (rows, _cols) = logits.dims2()?;
            if rows != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected Qwen3-TTS semantic logits shape: {:?}",
                    logits.shape().dims()
                )));
            }
            logits.i(0)?
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected Qwen3-TTS semantic logits rank: {rank}"
            )))
        }
    };
    let vocab_len = logits.dim(0)?;
    let semantic_len = (semantic_vocab_size as usize).min(vocab_len);
    let mut values = collect_semantic_sampling_values(&logits, semantic_len, params, history)?;

    let eos_idx = eos_token_id as usize;
    if allow_eos && eos_idx < vocab_len && eos_idx >= semantic_len {
        values.push((eos_token_id, logit_scalar_f32(&logits, eos_token_id)?));
    }

    // Repetition penalty over recent semantic history.
    if params.repetition_penalty > 1.0 && !history.is_empty() {
        let seen: HashSet<u32> = history.iter().copied().collect();
        for (token_id, v) in values.iter_mut() {
            if !seen.contains(token_id) {
                continue;
            }
            if !v.is_finite() {
                continue;
            }
            if *v > 0.0 {
                *v /= params.repetition_penalty;
            } else {
                *v *= params.repetition_penalty;
            }
        }
    }

    let temperature = params.temperature.max(1e-5);
    for (_, v) in values.iter_mut() {
        if v.is_finite() {
            *v /= temperature;
        }
    }

    let mut candidates: Vec<(u32, f32)> = values
        .into_iter()
        .filter(|(_, value)| value.is_finite())
        .collect();
    if candidates.is_empty() {
        return Ok(if allow_eos { eos_token_id } else { 0 });
    }

    // Top-k filtering.
    if params.top_k > 0 && params.top_k < candidates.len() {
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        candidates.truncate(params.top_k);
    }

    let max_logit = candidates
        .iter()
        .map(|(_, value)| *value)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(u32, f32)> = candidates
        .iter()
        .map(|(token_id, value)| (*token_id, (*value - max_logit).exp()))
        .collect();

    let mut sum: f32 = probs.iter().map(|(_, p)| *p).sum();
    if !sum.is_finite() || sum <= 0.0 {
        return argmax_semantic(&logits, semantic_vocab_size, eos_token_id, allow_eos);
    }
    for (_, p) in probs.iter_mut() {
        *p /= sum;
    }

    // Top-p filtering over normalized probabilities.
    if params.top_p < 1.0 {
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        let cutoff = params.top_p.max(1e-6);
        let mut cumsum = 0.0f32;
        let mut keep = 0usize;
        for (_, p) in probs.iter() {
            cumsum += *p;
            keep += 1;
            if cumsum >= cutoff {
                break;
            }
        }
        probs.truncate(keep.max(1));
        sum = probs.iter().map(|(_, p)| *p).sum();
        if sum > 0.0 {
            for (_, p) in probs.iter_mut() {
                *p /= sum;
            }
        }
    }

    let r = rng.next_f32();
    let mut acc = 0.0f32;
    for (token_id, p) in probs.iter() {
        acc += *p;
        if r <= acc {
            return Ok(*token_id);
        }
    }

    // Numerical fallback: pick max probability candidate.
    probs
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
        .map(|(idx, _)| *idx)
        .or_else(|| Some(if allow_eos { eos_token_id } else { 0 }))
        .ok_or_else(|| Error::InferenceError("Failed to sample semantic token".to_string()))
}

fn sample_semantic_reference(
    logits: &Tensor,
    semantic_vocab_size: u32,
    eos_token_id: u32,
    allow_eos: bool,
    params: &TtsGenerationParams,
    history: &[u32],
    rng: &mut SimpleRng,
) -> Result<u32> {
    let logits = logits.to_dtype(DType::F32)?;
    let logits = match logits.rank() {
        1 => logits,
        2 => {
            let (rows, _cols) = logits.dims2()?;
            if rows != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected Qwen3-TTS semantic logits shape: {:?}",
                    logits.shape().dims()
                )));
            }
            logits.i(0)?
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected Qwen3-TTS semantic logits rank: {rank}"
            )))
        }
    };
    let mut values = logits.to_vec1::<f32>()?;

    // Token suppression: keep semantic range and optional EOS only.
    for (idx, v) in values.iter_mut().enumerate() {
        let token_id = idx as u32;
        let allowed = token_id < semantic_vocab_size || (allow_eos && token_id == eos_token_id);
        if !allowed {
            *v = f32::NEG_INFINITY;
        }
    }

    // Repetition penalty over recent semantic history.
    if params.repetition_penalty > 1.0 && !history.is_empty() {
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
            let v = &mut values[idx];
            if !v.is_finite() {
                continue;
            }
            if *v > 0.0 {
                *v /= params.repetition_penalty;
            } else {
                *v *= params.repetition_penalty;
            }
        }
    }

    // Greedy fallback when sampling is effectively disabled.
    if params.temperature <= 1e-5 {
        return argmax_semantic_reference(&logits, semantic_vocab_size, eos_token_id, allow_eos);
    }

    let temperature = params.temperature.max(1e-5);
    for v in values.iter_mut() {
        if v.is_finite() {
            *v /= temperature;
        }
    }

    let mut candidates: Vec<usize> = values
        .iter()
        .enumerate()
        .filter_map(|(idx, &v)| if v.is_finite() { Some(idx) } else { None })
        .collect();
    if candidates.is_empty() {
        return Ok(if allow_eos { eos_token_id } else { 0 });
    }

    // Top-k filtering.
    if params.top_k > 0 && params.top_k < candidates.len() {
        candidates.sort_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap_or(Ordering::Equal));
        candidates.truncate(params.top_k);
    }

    let max_logit = candidates
        .iter()
        .map(|&idx| values[idx])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = candidates
        .iter()
        .map(|&idx| (idx, (values[idx] - max_logit).exp()))
        .collect();

    let mut sum: f32 = probs.iter().map(|(_, p)| *p).sum();
    if !sum.is_finite() || sum <= 0.0 {
        return argmax_semantic_reference(&logits, semantic_vocab_size, eos_token_id, allow_eos);
    }
    for (_, p) in probs.iter_mut() {
        *p /= sum;
    }

    // Top-p filtering over normalized probabilities.
    if params.top_p < 1.0 {
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        let cutoff = params.top_p.max(1e-6);
        let mut cumsum = 0.0f32;
        let mut keep = 0usize;
        for (_, p) in probs.iter() {
            cumsum += *p;
            keep += 1;
            if cumsum >= cutoff {
                break;
            }
        }
        probs.truncate(keep.max(1));
        sum = probs.iter().map(|(_, p)| *p).sum();
        if sum > 0.0 {
            for (_, p) in probs.iter_mut() {
                *p /= sum;
            }
        }
    }

    let r = rng.next_f32();
    let mut acc = 0.0f32;
    for (idx, p) in probs.iter() {
        acc += *p;
        if r <= acc {
            return Ok(*idx as u32);
        }
    }

    // Numerical fallback: pick max probability candidate.
    probs
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
        .map(|(idx, _)| *idx as u32)
        .or_else(|| Some(if allow_eos { eos_token_id } else { 0 }))
        .ok_or_else(|| Error::InferenceError("Failed to sample semantic token".to_string()))
}

fn collect_semantic_sampling_values(
    logits: &Tensor,
    semantic_len: usize,
    params: &TtsGenerationParams,
    history: &[u32],
) -> Result<Vec<(u32, f32)>> {
    if semantic_len == 0 {
        return Ok(Vec::new());
    }

    let semantic_logits = logits.narrow(0, 0, semantic_len)?;
    if params.top_k > 0 && params.top_k < semantic_len {
        let penalty_extra = if params.repetition_penalty > 1.0 && !history.is_empty() {
            history
                .iter()
                .copied()
                .filter(|token_id| (*token_id as usize) < semantic_len)
                .collect::<HashSet<_>>()
                .len()
        } else {
            0
        };
        let prefetch = params.top_k.saturating_add(penalty_extra).min(semantic_len);
        let (sorted_values, sorted_indices) = semantic_logits.sort_last_dim(false)?;
        let values = sorted_values.narrow(0, 0, prefetch)?.to_vec1::<f32>()?;
        let indices = sorted_indices
            .narrow(0, 0, prefetch)?
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()?;
        return Ok(indices.into_iter().zip(values).collect());
    }

    Ok(semantic_logits
        .to_vec1::<f32>()?
        .into_iter()
        .enumerate()
        .map(|(idx, value)| (idx as u32, value))
        .collect())
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15);
        Self {
            state: seed ^ 0xA076_1D64_78BD_642F,
        }
    }

    fn next_u32(&mut self) -> u32 {
        // xorshift64*
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

fn normalize_audio(samples: &mut [f32]) {
    if samples.is_empty() {
        return;
    }

    // Drop non-finite values and remove DC offset.
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for s in samples.iter_mut() {
        if !s.is_finite() {
            *s = 0.0;
            continue;
        }
        sum += *s as f64;
        count += 1;
    }
    if count > 0 {
        let mean = (sum / count as f64) as f32;
        for s in samples.iter_mut() {
            *s -= mean;
        }
    }

    // Peak normalize to avoid hard clipping in WAV encoder.
    let mut peak = 0.0f32;
    for &s in samples.iter() {
        let a = s.abs();
        if a > peak {
            peak = a;
        }
    }
    if peak > 0.95 {
        let scale = 0.95 / peak;
        for s in samples.iter_mut() {
            *s *= scale;
        }
    }

    // Keep output loudness within a practical band for WAV playback.
    let mut power = 0.0f64;
    for &s in samples.iter() {
        power += (s as f64) * (s as f64);
    }
    let rms = (power / samples.len() as f64).sqrt() as f32;
    let max_rms = 0.12f32;
    let min_rms = 0.04f32;
    if rms > max_rms && rms > 1e-6 {
        let scale = max_rms / rms;
        for s in samples.iter_mut() {
            *s *= scale;
        }
    } else if rms < min_rms && rms > 1e-6 {
        let scale = (min_rms / rms).min(8.0);
        for s in samples.iter_mut() {
            *s *= scale;
        }

        // Re-apply peak guard after boosting.
        let mut peak = 0.0f32;
        for &s in samples.iter() {
            let a = s.abs();
            if a > peak {
                peak = a;
            }
        }
        if peak > 0.95 {
            let scale = 0.95 / peak;
            for s in samples.iter_mut() {
                *s *= scale;
            }
        }
    }
}

/// Load a Qwen3-TTS model
pub fn load_model(model_path: &Path, device: DeviceProfile) -> Result<Qwen3TtsModel> {
    let kv_cache_dtype =
        std::env::var("IZWI_KV_CACHE_DTYPE").unwrap_or_else(|_| "float16".to_string());
    Qwen3TtsModel::load(model_path, device, default_kv_page_size(), &kv_cache_dtype)
}

#[cfg(test)]
mod tests {
    use crate::backends::{DeviceCapabilities, DeviceKind};

    use super::config::{CodePredictorConfig, TalkerConfig};
    use super::*;

    fn dtype_test_profile(
        kind: DeviceKind,
        supports_bf16: bool,
        supports_f16: bool,
    ) -> DeviceProfile {
        DeviceProfile {
            device: candle_core::Device::Cpu,
            kind,
            capabilities: DeviceCapabilities {
                supports_bf16,
                supports_f16,
                ..Default::default()
            },
            memory_pool: None,
        }
    }

    #[test]
    fn test_special_tokens_creation() {
        let main_config = Qwen3TtsConfig {
            architectures: vec!["Qwen3TTSForConditionalGeneration".to_string()],
            model_type: "qwen3_tts".to_string(),
            tokenizer_type: "qwen3_tts_tokenizer_12hz".to_string(),
            tts_model_size: "0b6".to_string(),
            tts_model_type: "custom_voice".to_string(),
            assistant_token_id: 77091,
            im_end_token_id: 151645,
            im_start_token_id: 151644,
            tts_bos_token_id: 151672,
            tts_eos_token_id: 151673,
            tts_pad_token_id: 151671,
            talker_config: TalkerConfig {
                model_type: "qwen3_tts_talker".to_string(),
                hidden_size: 1024,
                intermediate_size: 3072,
                num_hidden_layers: 28,
                num_attention_heads: 16,
                num_key_value_heads: 8,
                head_dim: 128,
                max_position_embeddings: 32768,
                vocab_size: 3072,
                text_vocab_size: 151936,
                text_hidden_size: 2048,
                num_code_groups: 16,
                rms_norm_eps: 1e-6,
                rope_theta: 1_000_000.0,
                hidden_act: "silu".to_string(),
                use_cache: true,
                position_id_per_seconds: 13,
                rope_scaling: None,
                sliding_window: None,
                code_predictor_config: CodePredictorConfig {
                    model_type: "qwen3_tts_talker_code_predictor".to_string(),
                    hidden_size: 1024,
                    intermediate_size: 3072,
                    num_hidden_layers: 5,
                    num_attention_heads: 16,
                    num_key_value_heads: 8,
                    head_dim: 128,
                    max_position_embeddings: 65536,
                    vocab_size: 2048,
                    num_code_groups: 16,
                    rms_norm_eps: 1e-6,
                    rope_theta: 1_000_000.0,
                    hidden_act: "silu".to_string(),
                    use_cache: true,
                    layer_types: vec![],
                    text_hidden_size: None,
                },
                codec_bos_id: 2149,
                codec_eos_token_id: 2150,
                codec_think_id: 2154,
                codec_nothink_id: 2155,
                codec_pad_id: 2148,
                codec_think_bos_id: 2156,
                codec_think_eos_id: 2157,
                spk_id: std::collections::HashMap::new(),
                spk_is_dialect: std::collections::HashMap::new(),
                codec_language_id: std::collections::HashMap::new(),
            },
        };

        let specials = TtsSpecialTokens::from_configs(&main_config, &main_config.talker_config);
        assert_eq!(specials.codec_bos_id, 2149);
        assert_eq!(specials.codec_eos_token_id, 2150);
    }

    #[test]
    fn cuda_base_tts_uses_half_transformers_without_changing_decoder_default() {
        let profile = dtype_test_profile(DeviceKind::Cuda, true, true);
        let plan = select_qwen3_tts_dtypes(&profile, None, false, true).unwrap();

        assert_eq!(plan.talker, DType::BF16);
        assert_eq!(plan.code_predictor, DType::BF16);
        assert_eq!(plan.speech_tokenizer, DType::F32);
    }

    #[test]
    fn cuda_custom_tts_falls_back_to_f16_transformers_without_bf16() {
        let profile = dtype_test_profile(DeviceKind::Cuda, false, true);
        let plan = select_qwen3_tts_dtypes(&profile, None, true, false).unwrap();

        assert_eq!(plan.talker, DType::F16);
        assert_eq!(plan.code_predictor, DType::F16);
        assert_eq!(plan.speech_tokenizer, DType::F32);
    }

    #[test]
    fn cpu_and_metal_qwen_tts_dtype_policy_stays_legacy() {
        let cpu = dtype_test_profile(DeviceKind::Cpu, false, false);
        let cpu_plan = select_qwen3_tts_dtypes(&cpu, None, true, false).unwrap();
        assert_eq!(
            cpu_plan,
            Qwen3TtsDTypePlan {
                talker: DType::F32,
                code_predictor: DType::F32,
                speech_tokenizer: DType::F32,
            }
        );

        let metal = dtype_test_profile(DeviceKind::Metal, false, false);
        let metal_custom = select_qwen3_tts_dtypes(&metal, None, true, false).unwrap();
        assert_eq!(metal_custom.talker, DType::F32);
        assert_eq!(metal_custom.code_predictor, DType::F32);
        assert_eq!(metal_custom.speech_tokenizer, DType::F32);

        let metal_voice_design = select_qwen3_tts_dtypes(&metal, None, false, false).unwrap();
        assert_eq!(metal_voice_design.talker, DType::F16);
        assert_eq!(metal_voice_design.code_predictor, DType::F16);
        assert_eq!(metal_voice_design.speech_tokenizer, DType::F16);
    }

    #[test]
    fn explicit_qwen_tts_dtype_override_applies_to_all_components() {
        let profile = dtype_test_profile(DeviceKind::Cuda, true, true);
        let plan = select_qwen3_tts_dtypes(&profile, Some("f32"), true, false).unwrap();

        assert_eq!(
            plan,
            Qwen3TtsDTypePlan {
                talker: DType::F32,
                code_predictor: DType::F32,
                speech_tokenizer: DType::F32,
            }
        );
    }

    #[test]
    fn qwen_tts_optimized_sampling_is_cuda_only() {
        let cpu = dtype_test_profile(DeviceKind::Cpu, false, false);
        let metal = dtype_test_profile(DeviceKind::Metal, false, false);
        let cuda = dtype_test_profile(DeviceKind::Cuda, true, true);

        assert!(!qwen_tts_uses_cuda_sampling(&cpu));
        assert!(!qwen_tts_uses_cuda_sampling(&metal));
        assert!(qwen_tts_uses_cuda_sampling(&cuda));
    }

    #[test]
    fn qwen_tts_eos_gate_matches_reference_minimum() {
        assert!(!qwen_tts_allows_eos(
            MIN_QWEN_TTS_TOKENS_BEFORE_EOS.saturating_sub(1)
        ));
        assert!(qwen_tts_allows_eos(MIN_QWEN_TTS_TOKENS_BEFORE_EOS));
        assert!(qwen_tts_allows_eos(MIN_QWEN_TTS_TOKENS_BEFORE_EOS + 1));
    }

    #[test]
    fn semantic_argmax_masks_invalid_tokens_and_gates_eos() {
        let logits = Tensor::new(
            vec![0.0f32, 2.0, 4.0, 1.0, 100.0, 3.0, 5.0],
            &candle_core::Device::Cpu,
        )
        .unwrap();

        assert_eq!(argmax_semantic(&logits, 3, 6, false).unwrap(), 2);
        assert_eq!(argmax_semantic(&logits, 3, 6, true).unwrap(), 6);
    }

    #[test]
    fn greedy_semantic_sampling_uses_device_argmax_path() {
        let logits = Tensor::new(
            vec![0.0f32, 2.0, 4.0, 1.0, 100.0, 3.0, 5.0],
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let params = TtsGenerationParams {
            temperature: 0.0,
            ..Default::default()
        };
        let mut rng = SimpleRng {
            state: 0x1234_5678_9abc_def0,
        };

        assert_eq!(
            sample_semantic(&logits, 3, 6, true, &params, &[], &mut rng, true).unwrap(),
            6
        );
    }

    #[test]
    fn reference_semantic_sampling_suppresses_invalid_tokens() {
        let logits = Tensor::new(
            vec![0.0f32, 2.0, 4.0, 1.0, 100.0, 3.0, 5.0],
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let params = TtsGenerationParams {
            temperature: 0.0,
            ..Default::default()
        };
        let mut rng = SimpleRng {
            state: 0x1234_5678_9abc_def0,
        };

        assert_eq!(
            sample_semantic(&logits, 3, 6, false, &params, &[], &mut rng, false).unwrap(),
            2
        );
        assert_eq!(
            sample_semantic(&logits, 3, 6, true, &params, &[], &mut rng, false).unwrap(),
            6
        );
    }

    #[test]
    fn semantic_sampling_reference_and_device_paths_match_simple_top_k() {
        let logits = Tensor::new(
            vec![10.0f32, 9.0, 8.0, 0.0, -5.0],
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let params = TtsGenerationParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 2,
            repetition_penalty: 1.1,
            ..Default::default()
        };
        let mut reference_rng = SimpleRng {
            state: 0x1234_5678_9abc_def0,
        };
        let mut device_rng = SimpleRng {
            state: 0x1234_5678_9abc_def0,
        };

        let reference = sample_semantic(
            &logits,
            5,
            99,
            false,
            &params,
            &[0],
            &mut reference_rng,
            false,
        )
        .unwrap();
        let device =
            sample_semantic(&logits, 5, 99, false, &params, &[0], &mut device_rng, true).unwrap();

        assert_eq!(reference, device);
    }

    #[test]
    fn top_k_semantic_sampling_keeps_penalty_replacement_candidates() {
        let logits = Tensor::new(vec![10.0f32, 9.0, 0.0, -5.0], &candle_core::Device::Cpu).unwrap();
        let params = TtsGenerationParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 1,
            repetition_penalty: 2.0,
            ..Default::default()
        };
        let mut rng = SimpleRng {
            state: 0x1234_5678_9abc_def0,
        };

        assert_eq!(
            sample_semantic(&logits, 4, 99, false, &params, &[0], &mut rng, true).unwrap(),
            1
        );
    }

    #[test]
    fn raw_codec_token_mapping_preserves_unoffset_tokens() {
        assert_eq!(raw_codec_token(151_936 + 7, 0, 151_936, 2048), 7);
        assert_eq!(raw_codec_token(7, 0, 151_936, 2048), 7);
        assert_eq!(
            raw_codec_token(151_936 + (3 * 2048) + 19, 3, 151_936, 2048),
            19
        );
        assert_eq!(raw_codec_token(19, 3, 151_936, 2048), 19);
    }

    #[test]
    fn cuda_chunked_codec_streaming_is_explicit_opt_in() {
        assert!(!qwen_tts_cuda_chunked_codec_stream_enabled_from(None));
        for raw in ["", "0", "false", "off", "no"] {
            assert!(!qwen_tts_cuda_chunked_codec_stream_enabled_from(Some(raw)));
        }
        for raw in ["1", "true", "YES", " on "] {
            assert!(qwen_tts_cuda_chunked_codec_stream_enabled_from(Some(raw)));
        }
    }
}
