use std::path::Path;
use std::sync::Mutex;

use candle_core::{DType, IndexOp, Tensor, D};
use tracing::info;

use crate::backends::{BackendKind, DeviceProfile};
use crate::error::{Error, Result};
use crate::model::ModelVariant;

use super::backbone::QuantizedLfm2Backbone;
use super::bundle::{Lfm25AudioBundle, Lfm25AudioBundleInfo};
use super::conformer::Lfm25AudioEncoder;
use super::config::{
    parse_audio_decoder_config, parse_audio_encoder_config, parse_detokenizer_config,
    parse_main_backbone_config, Lfm25AudioDecoderConfig, Lfm25AudioEncoderConfig,
    Lfm2BackboneConfig,
};
use super::preprocessor::Lfm25AudioPreprocessor;
use super::tokenizer::Lfm25TextTokenizer;

const DEFAULT_MAX_NEW_TOKENS: usize = 1024;

#[derive(Debug, Clone)]
pub struct Lfm25AudioTextOutput {
    pub text: String,
    pub prompt_tokens: usize,
    pub tokens_generated: usize,
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
    main_backbone: Mutex<QuantizedLfm2Backbone>,
    detokenizer_backbone: Mutex<QuantizedLfm2Backbone>,
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
        let detokenizer_backbone = QuantizedLfm2Backbone::load(
            &bundle.tokenizer,
            detokenizer_config.clone(),
            &device.device,
        )?;
        let encoder = Lfm25AudioEncoder::load(&bundle.mmproj, encoder_config.clone(), &device.device)?;

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
            main_backbone: Mutex::new(main_backbone),
            detokenizer_backbone: Mutex::new(detokenizer_backbone),
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

        let (features, feature_frames) =
            self.preprocessor.compute_features(&mono_16khz, &self.device.device)?;
        let audio_embeds = self.encoder.encode(&features, feature_frames)?;
        let (prefix_ids, suffix_ids) = self.build_asr_prompt_segments()?;
        let vocab_limit = self.tokenizer.vocab_size();
        let specials = self.tokenizer.specials().clone();

        self.with_main_backbone(|main_backbone| {
            main_backbone.reset_state();

            let prefix_embeds =
                embed_token_ids(main_backbone, &self.device.device, &prefix_ids)?;
            let suffix_embeds =
                embed_token_ids(main_backbone, &self.device.device, &suffix_ids)?;
            let prompt_embeds = Tensor::cat(
                &[&prefix_embeds, &audio_embeds, &suffix_embeds],
                1,
            )?;
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

                let next_tensor =
                    Tensor::from_vec(vec![next], (1, 1), &self.device.device)?;
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

    pub fn with_main_backbone<T>(
        &self,
        f: impl FnOnce(&mut QuantizedLfm2Backbone) -> Result<T>,
    ) -> Result<T> {
        let mut guard = self.main_backbone.lock().map_err(|_| {
            Error::InferenceError("LFM2.5 Audio backbone mutex poisoned".to_string())
        })?;
        f(&mut guard)
    }

    pub fn with_detokenizer_backbone<T>(
        &self,
        f: impl FnOnce(&mut QuantizedLfm2Backbone) -> Result<T>,
    ) -> Result<T> {
        let mut guard = self.detokenizer_backbone.lock().map_err(|_| {
            Error::InferenceError("LFM2.5 Audio detokenizer mutex poisoned".to_string())
        })?;
        f(&mut guard)
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
}

fn embed_token_ids(
    backbone: &QuantizedLfm2Backbone,
    device: &candle_core::Device,
    token_ids: &[u32],
) -> Result<Tensor> {
    let ids = Tensor::from_vec(token_ids.to_vec(), (1, token_ids.len()), device)?;
    backbone.embed_tokens(&ids)
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
    let idx = if idx.rank() == 0 { idx } else { idx.squeeze(0)? };
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
        let right = left.min(audio.len() - 1).saturating_add(1).min(audio.len() - 1);
        let frac = (src_pos - left as f64) as f32;
        let left_sample = audio[left.min(audio.len() - 1)];
        let right_sample = audio[right];
        out.push(left_sample + (right_sample - left_sample) * frac);
    }

    out
}
