use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use candle_core::{IndexOp, Tensor};
use tracing::info;

use crate::backends::{BackendKind, DeviceProfile};
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::shared::chat::{ChatMessage, ChatRole};

use super::audio_output::Lfm25AudioHead;
use super::backbone::QuantizedLfm2Backbone;
use super::bundle::{Lfm25AudioBundle, Lfm25AudioBundleInfo};
use super::config::{
    parse_audio_decoder_config, parse_audio_encoder_config, parse_detokenizer_config,
    parse_main_backbone_config, Lfm25AudioDecoderConfig, Lfm25AudioEncoderConfig,
    Lfm2BackboneConfig,
};
use super::conformer::Lfm25AudioEncoder;
use super::detokenizer::Lfm25AudioDetokenizer;
use super::preprocessor::Lfm25AudioPreprocessor;
use super::sampling::{
    greedy_from_logits, greedy_token_tensor_from_logits, sample_from_logits,
    Lfm25AudioGenerationConfig, SimpleRng,
};
use super::tokenizer::{Lfm25SpecialTokenIds, Lfm25TextTokenizer};
use super::LFM25_AUDIO_DEFAULT_INTERLEAVED_SYSTEM_PROMPT;

const DEFAULT_MAX_NEW_TOKENS: usize = 1024;
const DEFAULT_AUDIO_STREAM_DECODE_STRIDE_FRAMES: usize = 6;
const DEFAULT_AUDIO_STREAM_HOLDBACK_FRAMES: usize = 2;
const DEFAULT_ASR_STOP_CHECK_INTERVAL: usize = 96;

#[derive(Debug, Clone)]
pub struct Lfm25AudioTextOutput {
    pub text: String,
    pub prompt_tokens: usize,
    pub tokens_generated: usize,
    pub diagnostics: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct Lfm25AudioGenerationOutput {
    pub text: String,
    pub prompt_tokens: usize,
    pub tokens_generated: usize,
    pub audio_frames_generated: usize,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub diagnostics: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy)]
pub struct Lfm25AudioStreamConfig {
    pub decode_stride_frames: usize,
    pub holdback_frames: usize,
}

impl Default for Lfm25AudioStreamConfig {
    fn default() -> Self {
        Self {
            decode_stride_frames: DEFAULT_AUDIO_STREAM_DECODE_STRIDE_FRAMES,
            holdback_frames: DEFAULT_AUDIO_STREAM_HOLDBACK_FRAMES,
        }
    }
}

pub struct Lfm25AudioModel {
    device: DeviceProfile,
    bundle_info: Lfm25AudioBundleInfo,
    tokenizer: Lfm25TextTokenizer,
    main_config: Lfm2BackboneConfig,
    detokenizer_config: Lfm2BackboneConfig,
    encoder_config: Lfm25AudioEncoderConfig,
    decoder_config: Lfm25AudioDecoderConfig,
    preprocessor: Lfm25AudioPreprocessor,
    encoder: Lfm25AudioEncoder,
    audio_head: Lfm25AudioHead,
    detokenizer: Lfm25AudioDetokenizer,
    main_backbone: Mutex<QuantizedLfm2Backbone>,
}

#[derive(Debug, Clone, Copy, Default)]
struct Lfm25AsrProfile {
    prompt_embed_ms: f64,
    prompt_concat_ms: f64,
    main_prefill_ms: f64,
    decode_loop_ms: f64,
    decode_argmax_ms: f64,
    decode_host_read_ms: f64,
    decode_token_tensor_ms: f64,
    decode_forward_ms: f64,
    tokenizer_decode_ms: f64,
    token_select_reads: u64,
    host_token_reads: u64,
    host_read_chunks: u64,
    device_token_steps: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct Lfm25TtsProfile {
    prompt_embed_ms: f64,
    main_prefill_ms: f64,
    text_sampling_ms: f64,
    tokenizer_decode_ms: f64,
    text_forward_ms: f64,
    audio_head_ms: f64,
    audio_head_depth_linear_ms: f64,
    audio_head_depth_reshape_ms: f64,
    audio_head_cache_setup_ms: f64,
    audio_head_codebook_input_ms: f64,
    audio_head_depthformer_ms: f64,
    audio_head_sample_ms: f64,
    audio_head_embed_step_ms: f64,
    audio_head_materialize_ms: f64,
    audio_head_materialize_pack_ms: f64,
    audio_head_materialize_readback_ms: f64,
    audio_embed_ms: f64,
    audio_forward_ms: f64,
    detokenizer_embedding_ms: f64,
    detokenizer_upsample_ms: f64,
    detokenizer_backbone_ms: f64,
    detokenizer_projection_ms: f64,
    detokenizer_waveform_prepare_ms: f64,
    detokenizer_readback_ms: f64,
    detokenizer_istft_ms: f64,
    audio_head_calls: u64,
    audio_head_codebook_steps: u64,
    text_sample_calls: u64,
}

impl Lfm25AudioModel {
    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        if !matches!(variant, ModelVariant::Lfm25Audio15BGguf) {
            return Err(Error::ModelLoadError(format!(
                "Unsupported LFM2.5 Audio variant: {variant}"
            )));
        }

        let backend = BackendKind::from(device.kind);
        let bundle = Lfm25AudioBundle::load(model_dir, backend)?;
        let bundle_info = bundle.info();

        let tokenizer = Lfm25TextTokenizer::load(&bundle.main)?;
        let main_config = parse_main_backbone_config(&bundle.main)?;
        let detokenizer_config = parse_detokenizer_config(&bundle.tokenizer)?;
        let encoder_config = parse_audio_encoder_config(&bundle.mmproj)?;
        let decoder_config = parse_audio_decoder_config(&bundle.vocoder)?;
        let preprocessor = Lfm25AudioPreprocessor::load()?;

        let main_backbone =
            QuantizedLfm2Backbone::load(&bundle.main, main_config.clone(), &device.device)?;
        let encoder =
            Lfm25AudioEncoder::load(&bundle.mmproj, encoder_config.clone(), &device.device)?;
        let audio_head = Lfm25AudioHead::load(
            &bundle.vocoder,
            &decoder_config,
            main_config.embedding_length,
            &device.device,
        )?;
        let detokenizer = Lfm25AudioDetokenizer::load(
            &bundle.tokenizer,
            &bundle.vocoder,
            detokenizer_config.clone(),
            &decoder_config,
            &device.device,
        )?;

        info!(
            "Loaded LFM2.5 Audio GGUF bundle on {:?} from {}",
            device.kind,
            model_dir.display()
        );

        Ok(Self {
            device,
            bundle_info,
            tokenizer,
            main_config,
            detokenizer_config,
            encoder_config,
            decoder_config,
            preprocessor,
            encoder,
            audio_head,
            detokenizer,
            main_backbone: Mutex::new(main_backbone),
        })
    }

    pub fn device(&self) -> &DeviceProfile {
        &self.device
    }

    pub fn bundle_info(&self) -> &Lfm25AudioBundleInfo {
        &self.bundle_info
    }

    pub fn tokenizer(&self) -> &Lfm25TextTokenizer {
        &self.tokenizer
    }

    pub fn main_config(&self) -> &Lfm2BackboneConfig {
        &self.main_config
    }

    pub fn detokenizer_config(&self) -> &Lfm2BackboneConfig {
        &self.detokenizer_config
    }

    pub fn encoder_config(&self) -> &Lfm25AudioEncoderConfig {
        &self.encoder_config
    }

    pub fn decoder_config(&self) -> &Lfm25AudioDecoderConfig {
        &self.decoder_config
    }

    pub fn transcribe(&self, audio: &[f32], sample_rate: u32) -> Result<String> {
        let mut no_op = |_delta: &str| {};
        self.transcribe_with_callback(audio, sample_rate, &mut no_op)
    }

    pub fn transcribe_with_callback(
        &self,
        audio: &[f32],
        sample_rate: u32,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        Ok(self
            .transcribe_to_output_with_callback(
                audio,
                sample_rate,
                DEFAULT_MAX_NEW_TOKENS,
                on_delta,
            )?
            .text)
    }

    pub fn transcribe_to_output(
        &self,
        audio: &[f32],
        sample_rate: u32,
        max_new_tokens: usize,
    ) -> Result<Lfm25AudioTextOutput> {
        let mut no_op = |_delta: &str| {};
        self.transcribe_to_output_with_callback(audio, sample_rate, max_new_tokens, &mut no_op)
    }

    pub fn transcribe_to_output_with_callback(
        &self,
        audio: &[f32],
        sample_rate: u32,
        max_new_tokens: usize,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<Lfm25AudioTextOutput> {
        if audio.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }

        let total_started = Instant::now();
        let resample_started = Instant::now();
        let mono_16khz = if sample_rate == super::config::LFM25_AUDIO_INPUT_SAMPLE_RATE {
            audio.to_vec()
        } else {
            resample_linear(
                audio,
                sample_rate,
                super::config::LFM25_AUDIO_INPUT_SAMPLE_RATE,
            )
        };
        let resample_ms = elapsed_ms(resample_started);

        let feature_started = Instant::now();
        let (features, feature_frames) = self
            .preprocessor
            .compute_features(&mono_16khz, &self.device.device)?;
        let feature_extract_ms = elapsed_ms(feature_started);

        let encoder_started = Instant::now();
        let audio_embeds = self.encoder.encode(&features, feature_frames)?;
        let encoder_forward_ms = elapsed_ms(encoder_started);
        let audio_tokens = audio_embeds.dim(1)?;

        let prompt_started = Instant::now();
        let (prefix_ids, suffix_ids) = self.build_asr_prompt_segments()?;
        let prompt_build_ms = elapsed_ms(prompt_started);
        let vocab_limit = self.tokenizer.vocab_size();
        let specials = self.tokenizer.specials().clone();

        let main_started = Instant::now();
        let (mut output, profile) = self.with_main_backbone(|main_backbone| {
            let mut profile = Lfm25AsrProfile::default();
            main_backbone.reset_state();

            let prompt_embed_started = Instant::now();
            let prefix_embeds = embed_token_ids(main_backbone, &self.device.device, &prefix_ids)?;
            let suffix_embeds = embed_token_ids(main_backbone, &self.device.device, &suffix_ids)?;
            profile.prompt_embed_ms = elapsed_ms(prompt_embed_started);

            let prompt_concat_started = Instant::now();
            let prompt_embeds = Tensor::cat(&[&prefix_embeds, &audio_embeds, &suffix_embeds], 1)?;
            let prompt_tokens = prompt_embeds.dim(1)?;
            profile.prompt_concat_ms = elapsed_ms(prompt_concat_started);

            let prefill_started = Instant::now();
            let hidden = main_backbone.forward_embeds(&prompt_embeds, 0)?;
            let mut logits = main_backbone.project_last_hidden(&hidden)?;
            profile.main_prefill_ms = elapsed_ms(prefill_started);
            let mut position = prompt_tokens;
            let mut generated_ids = Vec::new();
            let mut assembled = String::new();
            let max_new_tokens = max_new_tokens.max(1);
            let stop_check_interval = lfm25_asr_stop_check_interval();
            let use_deferred_device_decode = (self.device.device.is_metal()
                || self.device.device.is_cuda())
                && stop_check_interval > 1;

            let decode_started = Instant::now();
            if use_deferred_device_decode {
                while generated_ids.len() < max_new_tokens {
                    let remaining = max_new_tokens.saturating_sub(generated_ids.len());
                    let chunk_len = remaining.min(stop_check_interval);
                    let mut chunk_tokens = Vec::with_capacity(chunk_len);

                    for _ in 0..chunk_len {
                        let argmax_started = Instant::now();
                        let next_token = greedy_token_tensor_from_logits(&logits, vocab_limit)?
                            .ok_or_else(|| {
                                Error::InferenceError(
                                    "Device ASR argmax returned no token tensor".to_string(),
                                )
                            })?;
                        profile.decode_argmax_ms += elapsed_ms(argmax_started);
                        profile.token_select_reads = profile.token_select_reads.saturating_add(1);
                        profile.device_token_steps = profile.device_token_steps.saturating_add(1);

                        let token_tensor_started = Instant::now();
                        let next_tensor = next_token.reshape((1, 1))?;
                        profile.decode_token_tensor_ms += elapsed_ms(token_tensor_started);

                        let decode_forward_started = Instant::now();
                        logits = main_backbone.forward_tokens(&next_tensor, position)?;
                        profile.decode_forward_ms += elapsed_ms(decode_forward_started);
                        position += 1;
                        chunk_tokens.push(next_token);
                    }

                    let read_started = Instant::now();
                    let token_refs = chunk_tokens.iter().collect::<Vec<_>>();
                    let host_tokens = Tensor::cat(&token_refs, 0)?
                        .to_vec1::<u32>()
                        .map_err(Error::from)?;
                    profile.decode_host_read_ms += elapsed_ms(read_started);
                    profile.host_read_chunks = profile.host_read_chunks.saturating_add(1);
                    profile.host_token_reads = profile
                        .host_token_reads
                        .saturating_add(u64::try_from(host_tokens.len()).unwrap_or(u64::MAX));

                    let mut should_stop = false;
                    for next in host_tokens {
                        if is_asr_stop_token(next, &specials) {
                            should_stop = true;
                            break;
                        }

                        if append_asr_text_token(
                            &self.tokenizer,
                            &mut generated_ids,
                            &mut assembled,
                            next,
                            &mut profile,
                            on_delta,
                        )? {
                            should_stop = true;
                            break;
                        }
                    }
                    if should_stop {
                        break;
                    }
                }
            } else {
                while generated_ids.len() < max_new_tokens {
                    let argmax_started = Instant::now();
                    let next = greedy_from_logits(&logits, vocab_limit)?;
                    profile.decode_argmax_ms += elapsed_ms(argmax_started);
                    profile.token_select_reads = profile.token_select_reads.saturating_add(1);
                    profile.host_token_reads = profile.host_token_reads.saturating_add(1);
                    if is_asr_stop_token(next, &specials) {
                        break;
                    }

                    if append_asr_text_token(
                        &self.tokenizer,
                        &mut generated_ids,
                        &mut assembled,
                        next,
                        &mut profile,
                        on_delta,
                    )? {
                        break;
                    }

                    let token_tensor_started = Instant::now();
                    let next_tensor = Tensor::from_vec(vec![next], (1, 1), &self.device.device)?;
                    profile.decode_token_tensor_ms += elapsed_ms(token_tensor_started);
                    let decode_forward_started = Instant::now();
                    logits = main_backbone.forward_tokens(&next_tensor, position)?;
                    profile.decode_forward_ms += elapsed_ms(decode_forward_started);
                    position += 1;
                }
            }
            profile.decode_loop_ms = elapsed_ms(decode_started);

            Ok((
                Lfm25AudioTextOutput {
                    text: assembled.trim().to_string(),
                    prompt_tokens,
                    tokens_generated: generated_ids.len(),
                    diagnostics: None,
                },
                profile,
            ))
        })?;
        let main_backbone_ms = elapsed_ms(main_started);
        let model_total_ms = elapsed_ms(total_started);
        let device_token_select_reads = profile.device_token_steps;
        let host_argmax_reads = if device_token_select_reads > 0 {
            0
        } else {
            profile.host_token_reads
        };
        let stop_check_interval = lfm25_asr_stop_check_interval();
        let deferred_stop_check = device_token_select_reads > 0 && stop_check_interval > 1;
        output.diagnostics = Some(serde_json::json!({
            "model": "lfm25_audio",
            "task": "asr",
            "timings_ms": {
                "resample": resample_ms,
                "feature_extract": feature_extract_ms,
                "mel": feature_extract_ms,
                "encoder_forward": encoder_forward_ms,
                "audio_encode": encoder_forward_ms,
                "prompt_build": prompt_build_ms,
                "prompt_embed": profile.prompt_embed_ms,
                "prompt_concat": profile.prompt_concat_ms,
                "prefill": profile.main_prefill_ms,
                "main_prefill": profile.main_prefill_ms,
                "decode": profile.decode_loop_ms,
                "decode_argmax": profile.decode_argmax_ms,
                "decode_host_read": profile.decode_host_read_ms,
                "decode_token_tensor": profile.decode_token_tensor_ms,
                "decode_forward": profile.decode_forward_ms,
                "tokenizer_decode": profile.tokenizer_decode_ms,
                "main_backbone": main_backbone_ms,
                "model_total": model_total_ms
            },
            "prompt": {
                "prompt_tokens": output.prompt_tokens,
                "prefix_tokens": prefix_ids.len(),
                "suffix_tokens": suffix_ids.len()
            },
            "audio": {
                "input_samples": audio.len(),
                "input_sample_rate": sample_rate,
                "resampled_samples": mono_16khz.len(),
                "feature_frames": feature_frames,
                "audio_tokens": audio_tokens
            },
            "decode": {
                "generated_tokens": output.tokens_generated,
                "max_new_tokens": max_new_tokens,
                "token_select_reads": profile.token_select_reads,
                "device_argmax_reads": device_token_select_reads,
                "host_argmax_reads": host_argmax_reads,
                "host_read_chunks": profile.host_read_chunks,
                "host_token_reads": profile.host_token_reads,
                "device_token_steps": profile.device_token_steps,
                "profile": {
                    "enabled": true,
                    "sampling_ms": profile.decode_argmax_ms + profile.decode_host_read_ms,
                    "argmax_ms": profile.decode_argmax_ms,
                    "host_read_ms": profile.decode_host_read_ms,
                    "host_read_chunks": profile.host_read_chunks,
                    "token_tensor_ms": profile.decode_token_tensor_ms,
                    "decoder_forward_ms": profile.decode_forward_ms,
                    "tokenizer_decode_ms": profile.tokenizer_decode_ms,
                    "step_total_ms": profile.decode_loop_ms
                }
            },
            "execution": {
                "deferred_stop_check": deferred_stop_check,
                "chunked_stop_check": deferred_stop_check,
                "stop_check_interval": stop_check_interval
            }
        }));
        Ok(output)
    }

    pub fn generate_sequential(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
    ) -> Result<Lfm25AudioGenerationOutput> {
        let mut no_op = |_delta: &str| {};
        self.generate_sequential_with_callback(messages, max_new_tokens, &mut no_op)
    }

    pub fn generate_sequential_with_callback(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        on_text_delta: &mut dyn FnMut(&str),
    ) -> Result<Lfm25AudioGenerationOutput> {
        self.generate_sequential_with_config_and_callback(
            messages,
            max_new_tokens,
            &Lfm25AudioGenerationConfig::default(),
            on_text_delta,
        )
    }

    pub fn generate_sequential_with_config_and_callback(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        generation_config: &Lfm25AudioGenerationConfig,
        on_text_delta: &mut dyn FnMut(&str),
    ) -> Result<Lfm25AudioGenerationOutput> {
        let total_started = Instant::now();
        let prompt_build_started = Instant::now();
        let prompt_ids = self.build_chat_prompt(messages)?;
        let prompt_build_ms = elapsed_ms(prompt_build_started);
        let vocab_limit = self.tokenizer.vocab_size();
        let specials = self.tokenizer.specials().clone();
        let codebooks = self.decoder_config.codebooks;

        let main_started = Instant::now();
        let (text, prompt_tokens, tokens_generated, audio_codes, mut profile) = self
            .with_main_backbone(|main_backbone| {
                let mut profile = Lfm25TtsProfile::default();
                let mut rng = SimpleRng::new(generation_config.seed);
                main_backbone.reset_state();

                let prompt_embed_started = Instant::now();
                let prompt_embeds =
                    embed_token_ids(main_backbone, &self.device.device, &prompt_ids)?;
                profile.prompt_embed_ms = elapsed_ms(prompt_embed_started);
                let prompt_tokens = prompt_embeds.dim(1)?;
                let prefill_started = Instant::now();
                let prompt_hidden = main_backbone.forward_embeds(&prompt_embeds, 0)?;
                let mut last_hidden = last_hidden_state(&prompt_hidden)?;
                let mut logits = main_backbone.project_last_hidden(&prompt_hidden)?;
                profile.main_prefill_ms = elapsed_ms(prefill_started);
                let mut position = prompt_tokens;
                let mut visible_text_ids = Vec::new();
                let mut visible_text = String::new();
                let mut audio_codes = vec![Vec::new(); codebooks];
                let mut tokens_generated = 0usize;
                let mut in_audio = false;
                let max_new_tokens = max_new_tokens.max(1);

                while tokens_generated < max_new_tokens {
                    if !in_audio {
                        let sampling_started = Instant::now();
                        let next = sample_from_logits(
                            &logits,
                            vocab_limit,
                            &generation_config.text,
                            &mut rng,
                        )?;
                        profile.text_sampling_ms += elapsed_ms(sampling_started);
                        profile.text_sample_calls = profile.text_sample_calls.saturating_add(1);
                        tokens_generated += 1;

                        if next == specials.im_end
                            || next == specials.eos
                            || specials.eos_alt == Some(next)
                        {
                            break;
                        }

                        if next == specials.audio_start {
                            in_audio = true;
                        } else if next != specials.text_end {
                            visible_text_ids.push(next);
                            let tokenizer_started = Instant::now();
                            let decoded = self.tokenizer.decode_text(&visible_text_ids)?;
                            profile.tokenizer_decode_ms += elapsed_ms(tokenizer_started);
                            let delta = text_delta(&visible_text, &decoded);
                            if !delta.is_empty() {
                                for ch in delta.chars() {
                                    let mut buf = [0u8; 4];
                                    on_text_delta(ch.encode_utf8(&mut buf));
                                }
                            }
                            visible_text = decoded;
                        }

                        let text_forward_started = Instant::now();
                        let next_embed =
                            embed_token_ids(main_backbone, &self.device.device, &[next])?;
                        let step_hidden = main_backbone.forward_embeds(&next_embed, position)?;
                        position += 1;
                        last_hidden = last_hidden_state(&step_hidden)?;
                        logits = main_backbone.project_last_hidden(&step_hidden)?;
                        profile.text_forward_ms += elapsed_ms(text_forward_started);

                        if has_token_repetition_loop(&visible_text_ids) {
                            break;
                        }
                    } else {
                        let audio_head_started = Instant::now();
                        let (frame, audio_head_profile) =
                            self.audio_head.sample_audio_frame_with_profile(
                                &last_hidden,
                                &generation_config.audio,
                                &mut rng,
                            )?;
                        profile.audio_head_ms += elapsed_ms(audio_head_started);
                        profile.audio_head_depth_linear_ms += audio_head_profile.depth_linear_ms;
                        profile.audio_head_depth_reshape_ms += audio_head_profile.depth_reshape_ms;
                        profile.audio_head_cache_setup_ms += audio_head_profile.cache_setup_ms;
                        profile.audio_head_codebook_input_ms +=
                            audio_head_profile.codebook_input_ms;
                        profile.audio_head_depthformer_ms += audio_head_profile.depthformer_ms;
                        profile.audio_head_sample_ms += audio_head_profile.sample_ms;
                        profile.audio_head_embed_step_ms += audio_head_profile.embed_ms;
                        profile.audio_head_materialize_ms += audio_head_profile.materialize_ms;
                        profile.audio_head_materialize_pack_ms +=
                            audio_head_profile.materialize_pack_ms;
                        profile.audio_head_materialize_readback_ms +=
                            audio_head_profile.materialize_readback_ms;
                        profile.audio_head_calls = profile.audio_head_calls.saturating_add(1);
                        profile.audio_head_codebook_steps = profile
                            .audio_head_codebook_steps
                            .saturating_add(audio_head_profile.codebook_steps);
                        tokens_generated += 1;
                        let is_end =
                            frame.first().copied() == Some(self.audio_head.audio_end_token_id());
                        if !is_end {
                            for (codebook_idx, token) in frame.iter().copied().enumerate() {
                                audio_codes[codebook_idx].push(token);
                            }
                        }

                        let audio_embed_started = Instant::now();
                        let audio_embed = self
                            .audio_head
                            .embed_audio_frame(&frame, &self.device.device)?;
                        profile.audio_embed_ms += elapsed_ms(audio_embed_started);
                        let audio_forward_started = Instant::now();
                        let step_hidden = main_backbone.forward_embeds(&audio_embed, position)?;
                        position += 1;
                        last_hidden = last_hidden_state(&step_hidden)?;
                        profile.audio_forward_ms += elapsed_ms(audio_forward_started);

                        if is_end {
                            in_audio = false;
                            logits = main_backbone.project_last_hidden(&step_hidden)?;
                        }
                    }
                }

                Ok((
                    visible_text.trim().to_string(),
                    prompt_tokens,
                    tokens_generated,
                    audio_codes,
                    profile,
                ))
            })?;
        let main_backbone_ms = elapsed_ms(main_started);

        let detokenizer_started = Instant::now();
        let (samples, detokenizer_profile) = self
            .detokenizer
            .decode_with_profile(&audio_codes, &self.device.device)?;
        let detokenizer_ms = elapsed_ms(detokenizer_started);
        profile.detokenizer_embedding_ms = detokenizer_profile.embedding_ms;
        profile.detokenizer_upsample_ms = detokenizer_profile.upsample_ms;
        profile.detokenizer_backbone_ms = detokenizer_profile.backbone_forward_ms;
        profile.detokenizer_projection_ms = detokenizer_profile.projection_ms;
        profile.detokenizer_waveform_prepare_ms = detokenizer_profile.waveform_prepare_ms;
        profile.detokenizer_readback_ms = detokenizer_profile.readback_ms;
        profile.detokenizer_istft_ms = detokenizer_profile.istft_ms;
        let model_total_ms = elapsed_ms(total_started);
        let audio_frames_generated = audio_codes.first().map(Vec::len).unwrap_or(0);
        let samples_len = samples.len();
        Ok(Lfm25AudioGenerationOutput {
            text,
            prompt_tokens,
            tokens_generated,
            audio_frames_generated,
            samples,
            sample_rate: self.decoder_config.output_sample_rate,
            diagnostics: Some(serde_json::json!({
                "model": "lfm25_audio",
                "task": "tts",
                "timings_ms": {
                    "prompt_build": prompt_build_ms,
                    "prompt_embed": profile.prompt_embed_ms,
                    "prefill": profile.main_prefill_ms,
                    "main_prefill": profile.main_prefill_ms,
                    "text_sampling": profile.text_sampling_ms,
                    "tokenizer_decode": profile.tokenizer_decode_ms,
                    "text_forward": profile.text_forward_ms,
                    "audio_head": profile.audio_head_ms,
                    "audio_head_depth_linear": profile.audio_head_depth_linear_ms,
                    "audio_head_depth_reshape": profile.audio_head_depth_reshape_ms,
                    "audio_head_cache_setup": profile.audio_head_cache_setup_ms,
                    "audio_head_codebook_input": profile.audio_head_codebook_input_ms,
                    "audio_head_depthformer": profile.audio_head_depthformer_ms,
                    "audio_head_sample": profile.audio_head_sample_ms,
                    "audio_head_embed_step": profile.audio_head_embed_step_ms,
                    "audio_head_materialize": profile.audio_head_materialize_ms,
                    "audio_head_materialize_pack": profile.audio_head_materialize_pack_ms,
                    "audio_head_materialize_readback": profile.audio_head_materialize_readback_ms,
                    "audio_embed": profile.audio_embed_ms,
                    "audio_forward": profile.audio_forward_ms,
                    "main_backbone": main_backbone_ms,
                    "detokenizer": detokenizer_ms,
                    "detokenizer_embedding": profile.detokenizer_embedding_ms,
                    "detokenizer_upsample": profile.detokenizer_upsample_ms,
                    "detokenizer_backbone": profile.detokenizer_backbone_ms,
                    "detokenizer_projection": profile.detokenizer_projection_ms,
                    "detokenizer_waveform_prepare": profile.detokenizer_waveform_prepare_ms,
                    "detokenizer_readback": profile.detokenizer_readback_ms,
                    "detokenizer_istft": profile.detokenizer_istft_ms,
                    "model_total": model_total_ms
                },
                "prompt": {
                    "prompt_tokens": prompt_tokens
                },
                "decode": {
                    "generated_tokens": tokens_generated,
                    "max_new_tokens": max_new_tokens,
                    "text_sample_calls": profile.text_sample_calls,
                    "audio_head_calls": profile.audio_head_calls,
                    "audio_head_codebook_steps": profile.audio_head_codebook_steps
                },
                "audio": {
                    "audio_frames": audio_frames_generated,
                    "sample_rate": self.decoder_config.output_sample_rate,
                    "samples": samples_len
                }
            })),
        })
    }

    pub fn generate_interleaved(
        &self,
        history_messages: &[ChatMessage],
        audio: &[f32],
        sample_rate: u32,
        max_new_tokens: usize,
    ) -> Result<Lfm25AudioGenerationOutput> {
        let mut no_text = |_delta: &str| {};
        let mut no_audio = |_samples: &[f32]| {};
        self.generate_interleaved_with_config_and_callback(
            history_messages,
            audio,
            sample_rate,
            max_new_tokens,
            None,
            &Lfm25AudioGenerationConfig::default(),
            &Lfm25AudioStreamConfig::default(),
            &mut no_text,
            &mut no_audio,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_interleaved_with_config_and_callback(
        &self,
        history_messages: &[ChatMessage],
        audio: &[f32],
        sample_rate: u32,
        max_new_tokens: usize,
        system_prompt: Option<&str>,
        generation_config: &Lfm25AudioGenerationConfig,
        stream_config: &Lfm25AudioStreamConfig,
        on_text_delta: &mut dyn FnMut(&str),
        on_audio_samples: &mut dyn FnMut(&[f32]),
    ) -> Result<Lfm25AudioGenerationOutput> {
        if audio.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }

        let audio_embeds = self.encode_audio_input(audio, sample_rate)?;
        let (prefix_ids, suffix_ids) =
            self.build_audio_chat_prompt(history_messages, system_prompt)?;
        let vocab_limit = self.tokenizer.vocab_size();
        let specials = self.tokenizer.specials().clone();
        let codebooks = self.decoder_config.codebooks;
        let stride_frames = stream_config.decode_stride_frames.max(1);
        let holdback_samples = self.audio_stream_holdback_samples(stream_config);

        let (text, prompt_tokens, tokens_generated, audio_codes, samples) = self
            .with_main_backbone(|main_backbone| {
                let mut rng = SimpleRng::new(generation_config.seed);
                let mut emitted_audio_samples = 0usize;

                main_backbone.reset_state();
                let prefix_embeds =
                    embed_token_ids(main_backbone, &self.device.device, &prefix_ids)?;
                let suffix_embeds =
                    embed_token_ids(main_backbone, &self.device.device, &suffix_ids)?;
                let prompt_embeds =
                    Tensor::cat(&[&prefix_embeds, &audio_embeds, &suffix_embeds], 1)?;
                let prompt_tokens = prompt_embeds.dim(1)?;
                let prompt_hidden = main_backbone.forward_embeds(&prompt_embeds, 0)?;
                let mut last_hidden = last_hidden_state(&prompt_hidden)?;
                let mut logits = main_backbone.project_last_hidden(&prompt_hidden)?;
                let mut position = prompt_tokens;
                let mut visible_text_ids = Vec::new();
                let mut visible_text = String::new();
                let mut audio_codes = vec![Vec::new(); codebooks];
                let mut tokens_generated = 0usize;
                let mut in_audio = false;
                let mut text_done = false;
                let mut modality_left = self.decoder_config.interleaved_n_text.max(1);
                let max_new_tokens = max_new_tokens.max(1);

                while tokens_generated < max_new_tokens {
                    modality_left = modality_left.saturating_sub(1);
                    if !in_audio {
                        let next = sample_from_logits(
                            &logits,
                            vocab_limit,
                            &generation_config.text,
                            &mut rng,
                        )?;
                        tokens_generated += 1;

                        if next == specials.im_end
                            || next == specials.eos
                            || specials.eos_alt == Some(next)
                        {
                            break;
                        }

                        if next == specials.text_end {
                            text_done = true;
                        } else {
                            visible_text_ids.push(next);
                            let decoded = self.tokenizer.decode_text(&visible_text_ids)?;
                            let delta = text_delta(&visible_text, &decoded);
                            if !delta.is_empty() {
                                for ch in delta.chars() {
                                    let mut buf = [0u8; 4];
                                    on_text_delta(ch.encode_utf8(&mut buf));
                                }
                            }
                            visible_text = decoded;
                        }

                        if modality_left == 0 || text_done {
                            in_audio = true;
                            modality_left = self.decoder_config.interleaved_n_audio.max(1);
                        }

                        let next_embed =
                            embed_token_ids(main_backbone, &self.device.device, &[next])?;
                        let step_hidden = main_backbone.forward_embeds(&next_embed, position)?;
                        position += 1;
                        last_hidden = last_hidden_state(&step_hidden)?;
                        logits = main_backbone.project_last_hidden(&step_hidden)?;

                        if has_token_repetition_loop(&visible_text_ids) {
                            break;
                        }
                    } else {
                        let mut frame = self.audio_head.sample_audio_frame(
                            &last_hidden,
                            &generation_config.audio,
                            &mut rng,
                        )?;
                        tokens_generated += 1;
                        let is_end =
                            frame.first().copied() == Some(self.audio_head.audio_end_token_id());
                        if is_end {
                            frame.fill(self.audio_head.audio_end_token_id());
                            in_audio = false;
                        } else {
                            for (codebook_idx, token) in frame.iter().copied().enumerate() {
                                audio_codes[codebook_idx].push(token);
                            }
                            if modality_left == 0 && !text_done {
                                in_audio = false;
                                modality_left = self.decoder_config.interleaved_n_text.max(1);
                            }
                        }

                        let audio_embed = self
                            .audio_head
                            .embed_audio_frame(&frame, &self.device.device)?;
                        let step_hidden = main_backbone.forward_embeds(&audio_embed, position)?;
                        position += 1;
                        last_hidden = last_hidden_state(&step_hidden)?;
                        logits = main_backbone.project_last_hidden(&step_hidden)?;

                        let should_decode_partial = !audio_codes[0].is_empty()
                            && (is_end
                                || !in_audio
                                || audio_codes[0].len() % stride_frames == 0
                                || tokens_generated >= max_new_tokens);
                        if should_decode_partial {
                            let partial =
                                self.detokenizer.decode(&audio_codes, &self.device.device)?;
                            let delta = next_audio_delta_stable(
                                &partial,
                                &mut emitted_audio_samples,
                                if is_end || !in_audio {
                                    0
                                } else {
                                    holdback_samples
                                },
                                is_end || tokens_generated >= max_new_tokens,
                            );
                            if !delta.is_empty() {
                                on_audio_samples(&delta);
                            }
                        }
                    }
                }

                let samples = self.detokenizer.decode(&audio_codes, &self.device.device)?;
                let final_delta =
                    next_audio_delta_stable(&samples, &mut emitted_audio_samples, 0, true);
                if !final_delta.is_empty() {
                    on_audio_samples(&final_delta);
                }

                Ok((
                    visible_text
                        .trim()
                        .trim_end_matches(super::config::LFM25_AUDIO_TEXT_END_TOKEN)
                        .trim()
                        .to_string(),
                    prompt_tokens,
                    tokens_generated,
                    audio_codes,
                    samples,
                ))
            })?;

        Ok(Lfm25AudioGenerationOutput {
            text,
            prompt_tokens,
            tokens_generated,
            audio_frames_generated: audio_codes.first().map(Vec::len).unwrap_or(0),
            samples,
            sample_rate: self.decoder_config.output_sample_rate,
            diagnostics: None,
        })
    }

    pub fn with_main_backbone<T>(
        &self,
        f: impl FnOnce(&mut QuantizedLfm2Backbone) -> Result<T>,
    ) -> Result<T> {
        let mut guard = self.main_backbone.lock().map_err(|_| {
            Error::InferenceError("LFM2.5 Audio backbone mutex poisoned".to_string())
        })?;
        f(&mut guard)
    }

    fn encode_audio_input(&self, audio: &[f32], sample_rate: u32) -> Result<Tensor> {
        let mono_16khz = if sample_rate == super::config::LFM25_AUDIO_INPUT_SAMPLE_RATE {
            audio.to_vec()
        } else {
            resample_linear(
                audio,
                sample_rate,
                super::config::LFM25_AUDIO_INPUT_SAMPLE_RATE,
            )
        };

        let (features, feature_frames) = self
            .preprocessor
            .compute_features(&mono_16khz, &self.device.device)?;
        self.encoder.encode(&features, feature_frames)
    }

    fn audio_stream_holdback_samples(&self, stream_config: &Lfm25AudioStreamConfig) -> usize {
        self.decoder_config
            .output_hop_length
            .saturating_mul(self.decoder_config.detokenizer_upsample_factor)
            .saturating_mul(stream_config.holdback_frames)
    }

    fn build_asr_prompt_segments(&self) -> Result<(Vec<u32>, Vec<u32>)> {
        let specials = self.tokenizer.specials();
        let mut prefix = Vec::new();
        if let Some(bos) = specials.bos {
            prefix.push(bos);
        }
        prefix.push(specials.im_start);
        prefix.extend(self.tokenizer.encode_text("system\n")?);
        prefix.extend(self.tokenizer.encode_text("Perform ASR.")?);
        prefix.push(specials.im_end);
        prefix.extend(self.tokenizer.encode_text("\n")?);
        prefix.push(specials.im_start);
        prefix.extend(self.tokenizer.encode_text("user\n")?);

        let mut suffix = Vec::new();
        suffix.push(specials.im_end);
        suffix.extend(self.tokenizer.encode_text("\n")?);
        suffix.push(specials.im_start);
        suffix.extend(self.tokenizer.encode_text("assistant\n")?);

        Ok((prefix, suffix))
    }

    fn build_audio_chat_prompt(
        &self,
        history_messages: &[ChatMessage],
        system_prompt: Option<&str>,
    ) -> Result<(Vec<u32>, Vec<u32>)> {
        let specials = self.tokenizer.specials();
        let mut prompt_messages = history_messages.to_vec();
        let explicit_system_prompt = system_prompt
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned);

        if let Some(prompt) = explicit_system_prompt {
            if let Some(first) = prompt_messages.first_mut() {
                if matches!(first.role, ChatRole::System) {
                    first.content = prompt;
                } else {
                    prompt_messages.insert(
                        0,
                        ChatMessage {
                            role: ChatRole::System,
                            content: prompt,
                        },
                    );
                }
            } else {
                prompt_messages.insert(
                    0,
                    ChatMessage {
                        role: ChatRole::System,
                        content: prompt,
                    },
                );
            }
        } else if !matches!(
            prompt_messages.first().map(|message| &message.role),
            Some(ChatRole::System)
        ) {
            prompt_messages.insert(
                0,
                ChatMessage {
                    role: ChatRole::System,
                    content: LFM25_AUDIO_DEFAULT_INTERLEAVED_SYSTEM_PROMPT.to_string(),
                },
            );
        }

        let last_assistant_index = prompt_messages
            .iter()
            .rposition(|message| matches!(message.role, ChatRole::Assistant));

        let mut prefix = Vec::new();
        if let Some(bos) = specials.bos {
            prefix.push(bos);
        }

        for (idx, message) in prompt_messages.iter().enumerate() {
            let content = if matches!(message.role, ChatRole::Assistant) {
                if Some(idx) == last_assistant_index {
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

            prefix.push(specials.im_start);
            prefix.extend(
                self.tokenizer
                    .encode_text(&format!("{}\n", message.role.as_prompt_role()))?,
            );
            prefix.extend(self.tokenizer.encode_text(&content)?);
            prefix.push(specials.im_end);
            prefix.extend(self.tokenizer.encode_text("\n")?);
        }

        prefix.push(specials.im_start);
        prefix.extend(self.tokenizer.encode_text("user\n")?);

        let mut suffix = Vec::new();
        suffix.push(specials.im_end);
        suffix.extend(self.tokenizer.encode_text("\n")?);
        suffix.push(specials.im_start);
        suffix.extend(self.tokenizer.encode_text("assistant\n")?);

        Ok((prefix, suffix))
    }

    fn build_chat_prompt(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        if messages.is_empty() {
            return Err(Error::InvalidInput(
                "Chat request must include at least one message".to_string(),
            ));
        }

        let mut prompt_messages = messages.to_vec();
        if !matches!(
            prompt_messages.first().map(|message| &message.role),
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

        let specials = self.tokenizer.specials();
        let last_assistant_index = prompt_messages
            .iter()
            .rposition(|message| matches!(message.role, ChatRole::Assistant));

        let mut ids = Vec::new();
        if let Some(bos) = specials.bos {
            ids.push(bos);
        }

        for (idx, message) in prompt_messages.iter().enumerate() {
            let content = if matches!(message.role, ChatRole::Assistant) {
                if Some(idx) == last_assistant_index {
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

            ids.push(specials.im_start);
            ids.extend(
                self.tokenizer
                    .encode_text(&format!("{}\n", message.role.as_prompt_role()))?,
            );
            ids.extend(self.tokenizer.encode_text(&content)?);
            ids.push(specials.im_end);
            ids.extend(self.tokenizer.encode_text("\n")?);
        }

        ids.push(specials.im_start);
        ids.extend(self.tokenizer.encode_text("assistant\n")?);

        Ok(ids)
    }
}

fn embed_token_ids(
    backbone: &QuantizedLfm2Backbone,
    device: &candle_core::Device,
    token_ids: &[u32],
) -> Result<Tensor> {
    let ids = Tensor::from_vec(token_ids.to_vec(), (1, token_ids.len()), device)?;
    backbone.embed_tokens(&ids)
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn last_hidden_state(hidden_states: &Tensor) -> Result<Tensor> {
    let seq_len = hidden_states.dim(1)?;
    hidden_states
        .i((0, seq_len.saturating_sub(1)))
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

fn append_asr_text_token(
    tokenizer: &Lfm25TextTokenizer,
    generated_ids: &mut Vec<u32>,
    assembled: &mut String,
    next: u32,
    profile: &mut Lfm25AsrProfile,
    on_delta: &mut dyn FnMut(&str),
) -> Result<bool> {
    generated_ids.push(next);
    let tokenizer_started = Instant::now();
    let decoded = tokenizer.decode_text(generated_ids)?;
    profile.tokenizer_decode_ms += elapsed_ms(tokenizer_started);
    let delta = text_delta(assembled, &decoded);
    if !delta.is_empty() {
        for ch in delta.chars() {
            let mut buf = [0u8; 4];
            on_delta(ch.encode_utf8(&mut buf));
        }
    }
    *assembled = decoded;
    Ok(has_token_repetition_loop(generated_ids))
}

fn is_asr_stop_token(next: u32, specials: &Lfm25SpecialTokenIds) -> bool {
    next == specials.im_end
        || next == specials.eos
        || specials.eos_alt == Some(next)
        || next == specials.text_end
        || next == specials.audio_start
}

fn lfm25_asr_stop_check_interval() -> usize {
    lfm25_asr_stop_check_interval_from_env(std::env::var("IZWI_LFM25_ASR_STOP_CHECK_INTERVAL").ok())
}

fn lfm25_asr_stop_check_interval_from_env(value: Option<String>) -> usize {
    value
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_ASR_STOP_CHECK_INTERVAL)
        .clamp(1, 128)
}

fn next_audio_delta_stable(
    all_samples: &[f32],
    emitted_samples: &mut usize,
    holdback_samples: usize,
    is_final: bool,
) -> Vec<f32> {
    let stable_end = if is_final {
        all_samples.len()
    } else {
        all_samples.len().saturating_sub(holdback_samples)
    };
    let start = (*emitted_samples).min(stable_end);
    let delta = all_samples[start..stable_end].to_vec();
    *emitted_samples = stable_end;
    delta
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
    if ids.len() < 48 {
        return false;
    }
    const PATTERNS: &[(usize, usize)] = &[(24, 3), (16, 3), (12, 3), (8, 4), (6, 5)];
    PATTERNS
        .iter()
        .any(|(span, repeats)| has_suffix_repeat(ids, *span, *repeats))
}

fn resample_linear(audio: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || audio.len() < 2 {
        return audio.to_vec();
    }

    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = ((audio.len() as f64) * ratio).round().max(1.0) as usize;
    let mut out = Vec::with_capacity(out_len);

    for idx in 0..out_len {
        let src_pos = idx as f64 / ratio;
        let left = src_pos.floor() as usize;
        let right = left
            .min(audio.len() - 1)
            .saturating_add(1)
            .min(audio.len() - 1);
        let frac = (src_pos - left as f64) as f32;
        let left_sample = audio[left.min(audio.len() - 1)];
        let right_sample = audio[right];
        out.push(left_sample + (right_sample - left_sample) * frac);
    }

    out
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::backends::DeviceProfile;
    use crate::model::ModelVariant;

    fn local_model_dir(name: &str) -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        PathBuf::from(home)
            .join("Library/Application Support/izwi/models")
            .join(name)
    }

    #[test]
    fn next_audio_delta_stable_holds_back_tail_until_final() {
        let mut emitted = 0usize;
        let all = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
        let delta = next_audio_delta_stable(&all, &mut emitted, 2, false);
        assert_eq!(delta, vec![0.1, 0.2, 0.3]);
        assert_eq!(emitted, 3);

        let delta_final = next_audio_delta_stable(&all, &mut emitted, 0, true);
        assert_eq!(delta_final, vec![0.4, 0.5]);
        assert_eq!(emitted, 5);
    }

    #[test]
    fn strip_past_assistant_thinking_keeps_final_answer_only() {
        assert_eq!(
            strip_past_assistant_thinking("<think>plan</think>final answer"),
            "final answer"
        );
    }

    #[test]
    fn asr_stop_check_interval_defaults_to_ninety_six() {
        assert_eq!(lfm25_asr_stop_check_interval_from_env(None), 96);
        assert_eq!(
            lfm25_asr_stop_check_interval_from_env(Some("bad".to_string())),
            96
        );
    }

    #[test]
    fn asr_stop_check_interval_clamps_override() {
        assert_eq!(
            lfm25_asr_stop_check_interval_from_env(Some("0".to_string())),
            1
        );
        assert_eq!(
            lfm25_asr_stop_check_interval_from_env(Some("128".to_string())),
            128
        );
        assert_eq!(
            lfm25_asr_stop_check_interval_from_env(Some("256".to_string())),
            128
        );
        assert_eq!(
            lfm25_asr_stop_check_interval_from_env(Some("96".to_string())),
            96
        );
    }

    #[test]
    fn load_local_lfm25_audio_model_smoke_if_available() {
        let model_dir = local_model_dir("LFM2.5-Audio-1.5B-GGUF");
        if !model_dir.exists() {
            return;
        }

        let model = Lfm25AudioModel::load(
            &model_dir,
            ModelVariant::Lfm25Audio15BGguf,
            DeviceProfile::cpu(),
        )
        .expect("lfm2.5 audio assets should load");

        assert_eq!(model.main_config().architecture, "lfm2");
        assert_eq!(model.encoder_config().embedding_length, 512);
        assert_eq!(model.encoder_config().feed_forward_length, 2048);
        assert_eq!(model.decoder_config().codebooks, 8);
    }
}
