//! Native VibeVoice-ASR model path.

use std::path::{Path, PathBuf};

use candle_core::{DType, IndexOp, Tensor};
use serde_json::json;
use tracing::info;

use crate::backends::DeviceProfile;
use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::architectures::qwen3::core::{Qwen3Cache, Qwen3Model, Qwen3WeightLayout};
use crate::models::architectures::vibevoice::config::{
    VibeVoiceConfig, VibeVoicePreprocessorConfig,
};
use crate::models::architectures::vibevoice::connector::SpeechConnector;
use crate::models::architectures::vibevoice::prompt::VibeVoicePromptTokenizer;
use crate::models::architectures::vibevoice::tokenizer::{
    VibeVoiceAcousticTokenizer, VibeVoiceSemanticTokenizer,
};
use crate::models::shared::weights::gguf::load_model_weights;

const DEFAULT_MAX_NEW_TOKENS: usize = 768;

#[derive(Debug, Clone)]
struct VibeVoiceAsrPreprocessStats {
    normalized: bool,
    target_db_fs: f32,
    rms_before: f32,
    gain: f32,
    clipping_divisor: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VibeVoiceAsrTranscriptionOutput {
    pub text: String,
    pub language: Option<String>,
    pub diagnostics: Option<serde_json::Value>,
}

pub struct VibeVoiceAsrModel {
    model_dir: PathBuf,
    device: DeviceProfile,
    dtype: DType,
    config: VibeVoiceConfig,
    preprocessor: VibeVoicePreprocessorConfig,
    tokenizer: VibeVoicePromptTokenizer,
    acoustic_tokenizer: VibeVoiceAcousticTokenizer,
    semantic_tokenizer: VibeVoiceSemanticTokenizer,
    acoustic_connector: SpeechConnector,
    semantic_connector: SpeechConnector,
    language_model: Qwen3Model,
}

impl VibeVoiceAsrModel {
    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        if variant != ModelVariant::VibeVoiceAsr {
            return Err(Error::InvalidInput(format!(
                "VibeVoiceAsrModel cannot load non-ASR variant {variant}"
            )));
        }
        let config = VibeVoiceConfig::load(model_dir)?;
        if config.is_tts() {
            return Err(Error::ModelLoadError(
                "VibeVoice-ASR loader received a TTS config".to_string(),
            ));
        }
        let preprocessor = VibeVoicePreprocessorConfig::load(model_dir)?;
        let dtype = std::env::var("IZWI_VIBEVOICE_ASR_DTYPE")
            .ok()
            .as_deref()
            .map(str::trim)
            .filter(|raw| !raw.is_empty())
            .map(|raw| {
                device.select_model_dtype_checked(
                    ModelFamily::VibeVoiceAsr,
                    Some(raw),
                    "VibeVoice ASR",
                )
            })
            .transpose()?
            .unwrap_or_else(|| device.select_model_dtype(ModelFamily::VibeVoiceAsr, None));
        let vb = load_model_weights(model_dir, dtype, &device.device)?;
        let tokenizer =
            VibeVoicePromptTokenizer::load(model_dir, config.decoder_config.vocab_size)?;
        let acoustic_tokenizer = VibeVoiceAcousticTokenizer::load(
            &config.acoustic_tokenizer_config,
            vb.pp("model.acoustic_tokenizer"),
        )?;
        let semantic_tokenizer = VibeVoiceSemanticTokenizer::load(
            &config.semantic_tokenizer_config,
            vb.pp("model.semantic_tokenizer"),
        )?;
        let acoustic_connector = SpeechConnector::load(
            config.acoustic_vae_dim(),
            config.decoder_config.hidden_size,
            vb.pp("model.acoustic_connector"),
        )?;
        let semantic_connector = SpeechConnector::load(
            config.semantic_vae_dim(),
            config.decoder_config.hidden_size,
            vb.pp("model.semantic_connector"),
        )?;
        let language_model = Qwen3Model::load_with_layout(
            config.decoder_config.clone(),
            vb,
            Qwen3WeightLayout::VIBEVOICE,
        )?;
        info!(
            "Loaded VibeVoice-ASR from {:?} on {:?} with dtype {:?}",
            model_dir, device.kind, dtype
        );
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            device,
            dtype,
            config,
            preprocessor,
            tokenizer,
            acoustic_tokenizer,
            semantic_tokenizer,
            acoustic_connector,
            semantic_connector,
            language_model,
        })
    }

    pub fn transcribe_with_details_and_prompt(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
    ) -> Result<VibeVoiceAsrTranscriptionOutput> {
        let mut no_op = |_delta: &str| {};
        self.transcribe_internal(audio, sample_rate, language, prompt, &mut no_op)
    }

    pub fn transcribe_with_callback_and_prompt(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        Ok(self
            .transcribe_internal(audio, sample_rate, language, prompt, on_delta)?
            .text)
    }

    pub fn max_audio_seconds_hint(&self) -> Option<f32> {
        Some(60.0 * 60.0)
    }

    fn transcribe_internal(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<VibeVoiceAsrTranscriptionOutput> {
        if audio.is_empty() {
            return Err(Error::InvalidInput(
                "VibeVoice-ASR audio input cannot be empty".to_string(),
            ));
        }
        let (processed_audio, preprocess_stats) =
            preprocess_asr_audio(audio, sample_rate, &self.preprocessor)?;
        if processed_audio.is_empty() {
            return Err(Error::InvalidInput(
                "VibeVoice-ASR audio input produced no samples after preprocessing".to_string(),
            ));
        }
        let target_sample_rate = self.preprocessor.target_sample_rate();
        let audio_seconds = processed_audio.len() as f32 / target_sample_rate as f32;
        let compress_ratio = self.preprocessor.speech_tok_compress_ratio.max(1);
        let expected_acoustic_frames = asr_placeholder_count(processed_audio.len(), compress_ratio);
        let mut encoder_audio = processed_audio.clone();
        let encoder_samples = expected_acoustic_frames.saturating_mul(compress_ratio);
        if encoder_audio.len() < encoder_samples {
            encoder_audio.resize(encoder_samples, 0.0);
        }
        let speech = Tensor::from_vec(encoder_audio, (1, 1, encoder_samples), &self.device.device)?
            .to_dtype(self.dtype)?;
        let speech_features = self.encode_speech(&speech)?;
        let acoustic_frames = speech_features.dim(1)?;
        if acoustic_frames != expected_acoustic_frames {
            return Err(Error::InferenceError(format!(
                "VibeVoice-ASR tokenizer produced {acoustic_frames} frames but processor reserved {expected_acoustic_frames}"
            )));
        }
        let extra = prompt_instruction(language, prompt);
        let prompt = self.tokenizer.build_asr_prompt(
            audio_seconds,
            expected_acoustic_frames,
            extra.as_deref(),
        )?;
        let input_ids = Tensor::from_vec(
            prompt.input_ids.clone(),
            (1, prompt.input_ids.len()),
            &self.device.device,
        )?;
        let input_embeds = self.language_model.embeddings(&input_ids)?;
        let input_embeds = replace_range_with_features(
            &input_embeds,
            prompt.acoustic_input_range.clone(),
            &speech_features.to_dtype(input_embeds.dtype())?,
        )?;

        let mut cache = Qwen3Cache::new(self.language_model.num_layers());
        let logits =
            self.language_model
                .forward_with_embeds(&input_embeds, 0, Some(&mut cache), None)?;
        let mut pos = prompt.input_ids.len();
        let mut next = argmax_last_logits(&logits)?;
        let mut generated = Vec::new();
        let mut assembled = String::new();
        let stop_tokens = [
            self.tokenizer.specials().im_end,
            self.tokenizer.specials().endoftext,
        ];

        for _ in 0..DEFAULT_MAX_NEW_TOKENS {
            if stop_tokens.contains(&next) {
                break;
            }
            generated.push(next);
            let decoded = self.tokenizer.decode(&generated)?;
            if decoded.len() > assembled.len() {
                on_delta(&decoded[assembled.len()..]);
            }
            assembled = decoded;

            let token = Tensor::from_vec(vec![next], (1, 1), &self.device.device)?;
            let logits = self.language_model.forward(&token, pos, Some(&mut cache))?;
            pos += 1;
            next = argmax_last_logits(&logits)?;
        }

        let text = cleanup_transcript_text(&assembled);
        Ok(VibeVoiceAsrTranscriptionOutput {
            text,
            language: language.map(ToOwned::to_owned),
            diagnostics: Some(json!({
                "model_family": "vibevoice_asr",
                "model_dir": self.model_dir.display().to_string(),
                "audio": {
                    "input_sample_rate": sample_rate,
                    "input_samples": audio.len(),
                    "resampled_sample_rate": target_sample_rate,
                    "resampled_samples": processed_audio.len(),
                    "encoder_samples": encoder_samples,
                    "duration_seconds": audio_seconds,
                    "acoustic_frames": acoustic_frames,
                    "expected_acoustic_frames": expected_acoustic_frames,
                    "speech_tok_compress_ratio": compress_ratio,
                    "normalized": preprocess_stats.normalized,
                    "target_db_fs": preprocess_stats.target_db_fs,
                    "rms_before_normalization": preprocess_stats.rms_before,
                    "normalization_gain": preprocess_stats.gain,
                    "clipping_divisor": preprocess_stats.clipping_divisor,
                },
                "prompt": {
                    "tokens": prompt.prompt_token_count,
                    "acoustic_input_tokens": prompt.acoustic_input_range.end.saturating_sub(prompt.acoustic_input_range.start),
                    "language": language,
                    "extra_prompt": extra,
                },
                "decode": {
                    "generated_tokens": generated.len(),
                    "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
                },
                "execution": {
                    "dtype": format!("{:?}", self.dtype),
                    "device_kind": format!("{:?}", self.device.kind),
                    "decoder_layers": self.config.decoder_config.num_hidden_layers,
                }
            })),
        })
    }

    fn encode_speech(&self, speech: &Tensor) -> Result<Tensor> {
        let acoustic = self.acoustic_tokenizer.encode(speech)?;
        let acoustic = self.acoustic_tokenizer.sample(&acoustic)?;
        let acoustic = self.acoustic_connector.forward(&acoustic)?;

        let semantic = self.semantic_tokenizer.encode(speech)?.mode();
        let semantic = self.semantic_connector.forward(&semantic)?;

        if acoustic.dims() != semantic.dims() {
            return Err(Error::InferenceError(format!(
                "VibeVoice-ASR acoustic/semantic feature shape mismatch: {:?} vs {:?}",
                acoustic.dims(),
                semantic.dims()
            )));
        }
        acoustic.broadcast_add(&semantic).map_err(Error::from)
    }
}

fn prompt_instruction(language: Option<&str>, prompt: Option<&str>) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(language) = language.filter(|value| {
        let value = value.trim();
        !value.is_empty() && !value.eq_ignore_ascii_case("auto")
    }) {
        parts.push(format!("The spoken language is {}.", language.trim()));
    }
    if let Some(prompt) = prompt.filter(|value| !value.trim().is_empty()) {
        parts.push(prompt.trim().to_string());
    }
    (!parts.is_empty()).then(|| parts.join(" "))
}

fn replace_range_with_features(
    embeds: &Tensor,
    range: std::ops::Range<usize>,
    features: &Tensor,
) -> Result<Tensor> {
    let seq_len = embeds.dim(1)?;
    let feature_len = features.dim(1)?;
    if feature_len != range.end.saturating_sub(range.start) {
        return Err(Error::InferenceError(format!(
            "VibeVoice prompt reserved {} acoustic tokens but encoder produced {feature_len}",
            range.end.saturating_sub(range.start)
        )));
    }
    let mut parts = Vec::new();
    if range.start > 0 {
        parts.push(embeds.narrow(1, 0, range.start)?);
    }
    parts.push(features.clone());
    if range.end < seq_len {
        parts.push(embeds.narrow(1, range.end, seq_len - range.end)?);
    }
    Tensor::cat(&parts, 1).map_err(Error::from)
}

fn argmax_last_logits(logits: &Tensor) -> Result<u32> {
    let seq_len = logits.dim(1)?;
    let row = logits.i((0, seq_len - 1))?.to_dtype(DType::F32)?;
    let values = row.to_vec1::<f32>()?;
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx as u32)
        .ok_or_else(|| Error::InferenceError("VibeVoice-ASR logits row was empty".to_string()))
}

fn cleanup_transcript_text(raw: &str) -> String {
    raw.replace("<|im_end|>", "")
        .replace("<|endoftext|>", "")
        .trim()
        .to_string()
}

fn preprocess_asr_audio(
    audio: &[f32],
    sample_rate: u32,
    config: &VibeVoicePreprocessorConfig,
) -> Result<(Vec<f32>, VibeVoiceAsrPreprocessStats)> {
    let mut resampled = resample_linear(audio, sample_rate, config.target_sample_rate())?;
    for sample in &mut resampled {
        if !sample.is_finite() {
            *sample = 0.0;
        }
    }
    let stats = if config.normalize_audio {
        normalize_asr_loudness(&mut resampled, config.target_db_fs, config.eps)
    } else {
        VibeVoiceAsrPreprocessStats {
            normalized: false,
            target_db_fs: config.target_db_fs,
            rms_before: audio_rms(&resampled),
            gain: 1.0,
            clipping_divisor: 1.0,
        }
    };
    Ok((resampled, stats))
}

fn normalize_asr_loudness(
    samples: &mut [f32],
    target_db_fs: f32,
    eps: f32,
) -> VibeVoiceAsrPreprocessStats {
    if samples.is_empty() {
        return VibeVoiceAsrPreprocessStats {
            normalized: true,
            target_db_fs,
            rms_before: 0.0,
            gain: 1.0,
            clipping_divisor: 1.0,
        };
    }
    let rms = audio_rms(samples);
    let gain = 10f32.powf(target_db_fs / 20.0) / (rms + eps.max(0.0));
    for sample in samples.iter_mut() {
        *sample *= gain;
    }

    let peak = samples.iter().fold(0.0f32, |peak, &s| peak.max(s.abs()));
    let clipping_divisor = if peak > 1.0 { peak + eps.max(0.0) } else { 1.0 };
    if clipping_divisor != 1.0 {
        for sample in samples.iter_mut() {
            *sample /= clipping_divisor;
        }
    }

    VibeVoiceAsrPreprocessStats {
        normalized: true,
        target_db_fs,
        rms_before: rms,
        gain,
        clipping_divisor,
    }
}

fn audio_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples
        .iter()
        .map(|&sample| (sample as f64) * (sample as f64))
        .sum::<f64>()
        / samples.len() as f64)
        .sqrt() as f32
}

fn asr_placeholder_count(samples: usize, speech_tok_compress_ratio: usize) -> usize {
    let ratio = speech_tok_compress_ratio.max(1);
    samples.saturating_add(ratio - 1) / ratio
}

fn resample_linear(audio: &[f32], src_rate: u32, dst_rate: u32) -> Result<Vec<f32>> {
    if src_rate == 0 || dst_rate == 0 {
        return Err(Error::InvalidInput(
            "Sample rates must be positive for VibeVoice-ASR resampling".to_string(),
        ));
    }
    if src_rate == dst_rate {
        return Ok(audio.to_vec());
    }
    if audio.is_empty() {
        return Ok(Vec::new());
    }
    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = ((audio.len() as f64) * ratio).round().max(1.0) as usize;
    let mut out = Vec::with_capacity(out_len);
    for idx in 0..out_len {
        let src_pos = idx as f64 / ratio;
        let left = src_pos.floor() as usize;
        let right = (left + 1).min(audio.len() - 1);
        let frac = (src_pos - left as f64) as f32;
        out.push(audio[left] * (1.0 - frac) + audio[right] * frac);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_linear_preserves_identity_rate() {
        let audio = vec![0.0, 0.5, -0.25];
        assert_eq!(resample_linear(&audio, 24_000, 24_000).unwrap(), audio);
    }

    #[test]
    fn asr_placeholder_count_ceil_divides_samples_by_compress_ratio() {
        assert_eq!(asr_placeholder_count(0, 3_200), 0);
        assert_eq!(asr_placeholder_count(1, 3_200), 1);
        assert_eq!(asr_placeholder_count(3_200, 3_200), 1);
        assert_eq!(asr_placeholder_count(3_201, 3_200), 2);
        assert_eq!(asr_placeholder_count(9_599, 3_200), 3);
        assert_eq!(asr_placeholder_count(9_600, 3_200), 3);
    }

    #[test]
    fn normalize_asr_loudness_matches_reference_dbfs_formula() {
        let mut audio = vec![0.5, -0.5, 0.5, -0.5];
        let stats = normalize_asr_loudness(&mut audio, -20.0, 1e-6);

        let rms = audio_rms(&audio);
        assert!((rms - 0.1).abs() < 1e-5);
        assert!(stats.normalized);
        assert!((stats.rms_before - 0.5).abs() < 1e-6);
        assert!((stats.gain - 0.2).abs() < 1e-5);
        assert_eq!(stats.clipping_divisor, 1.0);
    }

    #[test]
    fn normalize_asr_loudness_avoids_clipping() {
        let mut audio = vec![1.0, -1.0];
        let stats = normalize_asr_loudness(&mut audio, 6.0, 1e-6);

        assert!(stats.clipping_divisor > 1.0);
        assert!(audio.iter().all(|sample| sample.abs() <= 1.0));
    }

    #[test]
    fn preprocess_asr_audio_sanitizes_and_resamples() {
        let config = VibeVoicePreprocessorConfig {
            sampling_rate: 4,
            speech_tok_compress_ratio: 2,
            normalize_audio: false,
            target_db_fs: -25.0,
            eps: 1e-6,
        };
        let (audio, stats) =
            preprocess_asr_audio(&[0.0, f32::NAN, 1.0], 2, &config).expect("preprocess");

        assert_eq!(audio.len(), 6);
        assert!(audio.iter().all(|sample| sample.is_finite()));
        assert!(!stats.normalized);
    }

    #[test]
    fn replace_range_preserves_prompt_length() {
        let device = candle_core::Device::Cpu;
        let embeds = Tensor::zeros((1, 5, 3), DType::F32, &device).unwrap();
        let features = Tensor::ones((1, 2, 3), DType::F32, &device).unwrap();
        let replaced = replace_range_with_features(&embeds, 2..4, &features).unwrap();
        assert_eq!(replaced.dims(), &[1, 5, 3]);
        assert_eq!(
            replaced.i((0, 2, ..)).unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0, 1.0, 1.0]
        );
    }
}
