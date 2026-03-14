use std::path::Path;
use std::sync::Mutex;

use candle_core::{DType, IndexOp, Tensor, D};
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
use super::sampling::{sample_from_logits, Lfm25AudioGenerationConfig, SimpleRng};
use super::tokenizer::Lfm25TextTokenizer;

const DEFAULT_MAX_NEW_TOKENS: usize = 1024;
const DEFAULT_INTERLEAVED_SYSTEM_PROMPT: &str = "Respond with interleaved text and audio.";
const DEFAULT_AUDIO_STREAM_DECODE_STRIDE_FRAMES: usize = 6;
const DEFAULT_AUDIO_STREAM_HOLDBACK_FRAMES: usize = 2;

#[derive(Debug, Clone)]
pub struct Lfm25AudioTextOutput {
    pub text: String,
    pub prompt_tokens: usize,
    pub tokens_generated: usize,
}

#[derive(Debug, Clone)]
pub struct Lfm25AudioGenerationOutput {
    pub text: String,
    pub prompt_tokens: usize,
    pub tokens_generated: usize,
    pub audio_frames_generated: usize,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
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
        let audio_embeds = self.encoder.encode(&features, feature_frames)?;
        let (prefix_ids, suffix_ids) = self.build_asr_prompt_segments()?;
        let vocab_limit = self.tokenizer.vocab_size();
        let specials = self.tokenizer.specials().clone();

        self.with_main_backbone(|main_backbone| {
            main_backbone.reset_state();

            let prefix_embeds = embed_token_ids(main_backbone, &self.device.device, &prefix_ids)?;
            let suffix_embeds = embed_token_ids(main_backbone, &self.device.device, &suffix_ids)?;
            let prompt_embeds = Tensor::cat(&[&prefix_embeds, &audio_embeds, &suffix_embeds], 1)?;
            let prompt_tokens = prompt_embeds.dim(1)?;

            let hidden = main_backbone.forward_embeds(&prompt_embeds, 0)?;
            let mut logits = main_backbone.project_last_hidden(&hidden)?;
            let mut position = prompt_tokens;
            let mut generated_ids = Vec::new();
            let mut assembled = String::new();
            let max_new_tokens = max_new_tokens.max(1);

            while generated_ids.len() < max_new_tokens {
                let next = argmax(&logits, vocab_limit)?;
                if next == specials.im_end
                    || next == specials.eos
                    || specials.eos_alt == Some(next)
                    || next == specials.text_end
                    || next == specials.audio_start
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
                logits = main_backbone.forward_tokens(&next_tensor, position)?;
                position += 1;
            }

            Ok(Lfm25AudioTextOutput {
                text: assembled.trim().to_string(),
                prompt_tokens,
                tokens_generated: generated_ids.len(),
            })
        })
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
        let prompt_ids = self.build_chat_prompt(messages)?;
        let vocab_limit = self.tokenizer.vocab_size();
        let specials = self.tokenizer.specials().clone();
        let codebooks = self.decoder_config.codebooks;

        let (text, prompt_tokens, tokens_generated, audio_codes) =
            self.with_main_backbone(|main_backbone| {
                let mut rng = SimpleRng::new(generation_config.seed);
                main_backbone.reset_state();

                let prompt_embeds =
                    embed_token_ids(main_backbone, &self.device.device, &prompt_ids)?;
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
                let max_new_tokens = max_new_tokens.max(1);

                while tokens_generated < max_new_tokens {
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

                        if next == specials.audio_start {
                            in_audio = true;
                        } else if next != specials.text_end {
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
                        let frame = self.audio_head.sample_audio_frame(
                            &last_hidden,
                            &generation_config.audio,
                            &mut rng,
                        )?;
                        tokens_generated += 1;
                        let is_end =
                            frame.first().copied() == Some(self.audio_head.audio_end_token_id());
                        if !is_end {
                            for (codebook_idx, token) in frame.iter().copied().enumerate() {
                                audio_codes[codebook_idx].push(token);
                            }
                        }

                        let audio_embed = self
                            .audio_head
                            .embed_audio_frame(&frame, &self.device.device)?;
                        let step_hidden = main_backbone.forward_embeds(&audio_embed, position)?;
                        position += 1;
                        last_hidden = last_hidden_state(&step_hidden)?;

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
                ))
            })?;

        let samples = self.detokenizer.decode(&audio_codes, &self.device.device)?;
        Ok(Lfm25AudioGenerationOutput {
            text,
            prompt_tokens,
            tokens_generated,
            audio_frames_generated: audio_codes.first().map(Vec::len).unwrap_or(0),
            samples,
            sample_rate: self.decoder_config.output_sample_rate,
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
                    content: DEFAULT_INTERLEAVED_SYSTEM_PROMPT.to_string(),
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

fn last_hidden_state(hidden_states: &Tensor) -> Result<Tensor> {
    let seq_len = hidden_states.dim(1)?;
    hidden_states
        .i((0, seq_len.saturating_sub(1)))
        .map_err(Error::from)
}

fn argmax(logits: &Tensor, vocab_limit: usize) -> Result<u32> {
    if vocab_limit == 0 {
        return Err(Error::InferenceError(
            "Cannot sample from zero-sized vocabulary".to_string(),
        ));
    }

    let logits = match logits.rank() {
        1 => {
            let vocab = logits.dim(0)?;
            logits.narrow(0, 0, vocab.min(vocab_limit))?
        }
        2 => {
            let (batch, vocab) = logits.dims2()?;
            if batch != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected batched logits for argmax: expected batch=1, got {batch}"
                )));
            }
            logits.i(0)?.narrow(0, 0, vocab.min(vocab_limit))?
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
        .take_while(|(left, right)| left == right)
        .count();
    current.chars().skip(common).collect()
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
