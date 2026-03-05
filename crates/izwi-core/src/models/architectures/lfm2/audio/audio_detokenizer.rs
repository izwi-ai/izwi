use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;
use serde::Deserialize;

use crate::error::{Error, Result};

use super::config::LfmConfig;
use super::lfm_backbone::LfmBackbone;

const DEFAULT_NUM_CODEBOOKS: usize = 8;
const DEFAULT_AUDIO_VOCAB_SIZE: usize = 2048;
const DEFAULT_OUTPUT_SIZE: usize = 1282;
const DEFAULT_SLIDING_WINDOW: usize = 30;
const DEFAULT_N_FFT: usize = 1280;
const DEFAULT_HOP_LENGTH: usize = 320;
const DEFAULT_UPSAMPLE_FACTOR: usize = 6;
const DEFAULT_EOS_TOKEN_ID: usize = 7;

#[derive(Debug, Clone, Deserialize)]
struct AudioDetokenizerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub layer_types: Vec<String>,
    pub rope_theta: f64,
    #[serde(alias = "conv_L_cache")]
    pub conv_l_cache: usize,
    pub norm_eps: f64,
    pub vocab_size: usize,
    #[serde(default = "default_output_size")]
    pub output_size: usize,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,
    #[serde(default = "default_num_codebooks")]
    pub num_codebooks: usize,
    #[serde(default = "default_audio_vocab_size")]
    pub audio_vocab_size: usize,
    #[serde(default = "default_n_fft")]
    pub n_fft: usize,
    #[serde(default = "default_hop_length")]
    pub hop_length: usize,
    #[serde(default = "default_upsample_factor")]
    pub upsample_factor: usize,
    #[serde(default = "default_eos_token_id")]
    pub eos_token_id: usize,
}

fn default_output_size() -> usize {
    DEFAULT_OUTPUT_SIZE
}
fn default_sliding_window() -> usize {
    DEFAULT_SLIDING_WINDOW
}
fn default_num_codebooks() -> usize {
    DEFAULT_NUM_CODEBOOKS
}
fn default_audio_vocab_size() -> usize {
    DEFAULT_AUDIO_VOCAB_SIZE
}
fn default_n_fft() -> usize {
    DEFAULT_N_FFT
}
fn default_hop_length() -> usize {
    DEFAULT_HOP_LENGTH
}
fn default_upsample_factor() -> usize {
    DEFAULT_UPSAMPLE_FACTOR
}
fn default_eos_token_id() -> usize {
    DEFAULT_EOS_TOKEN_ID
}

pub struct AudioDetokenizer {
    codebook_embedding: Embedding,
    lfm: LfmBackbone,
    lin: Linear,
    n_fft: usize,
    hop_length: usize,
    upsample_factor: usize,
    sliding_window: usize,
    num_codebooks: usize,
    audio_vocab_size: usize,
    window: Vec<f32>,
    device: Device,
}

impl AudioDetokenizer {
    pub fn load(model_dir: &Path, device: &Device) -> Result<Self> {
        let detok_dir = model_dir.join("audio_detokenizer");
        if !detok_dir.exists() {
            return Err(Error::ModelLoadError(format!(
                "Missing LFM2.5 audio_detokenizer directory: {}",
                detok_dir.display()
            )));
        }

        let cfg_path = detok_dir.join("config.json");
        let cfg: AudioDetokenizerConfig = serde_json::from_str(
            std::fs::read_to_string(&cfg_path)
                .map_err(|e| {
                    Error::ModelLoadError(format!(
                        "Failed reading LFM2.5 detokenizer config {}: {}",
                        cfg_path.display(),
                        e
                    ))
                })?
                .as_str(),
        )
        .map_err(|e| {
            Error::ModelLoadError(format!(
                "Invalid LFM2.5 detokenizer config {}: {}",
                cfg_path.display(),
                e
            ))
        })?;

        let lfm_cfg = LfmConfig {
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            num_hidden_layers: cfg.num_hidden_layers,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            max_position_embeddings: cfg.max_position_embeddings,
            layer_types: cfg
                .layer_types
                .iter()
                .map(|layer| {
                    if layer == "sliding_attention" {
                        "full_attention".to_string()
                    } else {
                        layer.to_string()
                    }
                })
                .collect(),
            rope_theta: cfg.rope_theta,
            conv_l_cache: cfg.conv_l_cache,
            norm_eps: cfg.norm_eps,
            vocab_size: cfg.vocab_size,
            eos_token_id: cfg.eos_token_id,
        };

        let weight_path = detok_dir.join("model.safetensors");
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weight_path], DType::F32, device)? };

        let codebook_embedding = candle_nn::embedding(
            cfg.num_codebooks * cfg.audio_vocab_size,
            cfg.hidden_size,
            vb.pp("emb.emb"),
        )?;
        let lfm = LfmBackbone::load(lfm_cfg, vb.pp("lfm"))?;
        let lin = candle_nn::linear(cfg.hidden_size, cfg.output_size, vb.pp("lin"))?;

        Ok(Self {
            codebook_embedding,
            lfm,
            lin,
            n_fft: cfg.n_fft,
            hop_length: cfg.hop_length,
            upsample_factor: cfg.upsample_factor,
            sliding_window: cfg.sliding_window,
            num_codebooks: cfg.num_codebooks,
            audio_vocab_size: cfg.audio_vocab_size,
            window: hann_window(cfg.n_fft),
            device: device.clone(),
        })
    }

    pub fn decode_tokens(&self, codebooks: &[Vec<u32>]) -> Result<Vec<f32>> {
        if codebooks.is_empty() || codebooks[0].is_empty() {
            return Ok(Vec::new());
        }
        if codebooks.len() != self.num_codebooks {
            return Err(Error::InferenceError(format!(
                "Expected {} codebooks for LFM2.5 detokenizer, got {}",
                self.num_codebooks,
                codebooks.len()
            )));
        }

        let frames = codebooks[0].len();
        if codebooks.iter().any(|c| c.len() != frames) {
            return Err(Error::InferenceError(
                "Inconsistent LFM2.5 audio token frame lengths".to_string(),
            ));
        }

        let mut offset_codes = Vec::with_capacity(self.num_codebooks * frames);
        for (codebook_idx, row) in codebooks.iter().enumerate() {
            let offset = (codebook_idx * self.audio_vocab_size) as u32;
            for &tok in row {
                if tok >= self.audio_vocab_size as u32 {
                    return Err(Error::InferenceError(format!(
                        "Invalid LFM2.5 audio code {} (expected < {})",
                        tok, self.audio_vocab_size
                    )));
                }
                offset_codes.push(tok + offset);
            }
        }

        let codes = Tensor::from_vec(offset_codes, (1, self.num_codebooks, frames), &self.device)?
            .to_dtype(DType::U32)?;
        let emb = self.codebook_embedding.forward(&codes)?; // [1, C, T, D]
        let emb = emb.sum(1)?.broadcast_div(
            &Tensor::new(self.num_codebooks as f32, &self.device)?.reshape((1, 1, 1))?,
        )?; // [1, T, D]
        let emb = self.upsample_nearest(&emb, self.upsample_factor)?;

        let seq_len = emb.dim(1)?;
        let mask = sliding_causal_mask(seq_len, self.sliding_window, emb.device())?;
        let hidden = self.lfm.forward_embeds(&emb, Some(&mask))?;
        let spec = self.lin.forward(&hidden)?; // [1, T', 1282]

        self.istft_from_detokenizer_spec(&spec)
    }

    fn upsample_nearest(&self, x: &Tensor, factor: usize) -> Result<Tensor> {
        upsample_nearest_time(x, factor)
    }

    fn istft_from_detokenizer_spec(&self, spec: &Tensor) -> Result<Vec<f32>> {
        let n_bins = self.n_fft / 2 + 1;
        let out_size = spec.dim(2)?;
        if out_size < n_bins * 2 {
            return Err(Error::InferenceError(format!(
                "Detokenizer output dim {} too small for n_fft {}",
                out_size, self.n_fft
            )));
        }

        let log_abs = spec.narrow(2, 0, n_bins)?.squeeze(0)?.to_vec2::<f32>()?;
        let angle = spec
            .narrow(2, n_bins, n_bins)?
            .squeeze(0)?
            .to_vec2::<f32>()?;
        let frames = log_abs.len();
        if frames == 0 {
            return Ok(Vec::new());
        }

        let output_len = (frames - 1) * self.hop_length + self.n_fft;
        let mut output = vec![0.0f32; output_len];
        let mut envelope = vec![0.0f32; output_len];

        let mut planner = FftPlanner::<f32>::new();
        let ifft = planner.plan_fft_inverse(self.n_fft);
        let mut spectrum = vec![Complex32::new(0.0, 0.0); self.n_fft];

        for frame_idx in 0..frames {
            spectrum.fill(Complex32::new(0.0, 0.0));
            for k in 0..n_bins {
                let mag = log_abs[frame_idx][k].exp();
                let ph = angle[frame_idx][k];
                spectrum[k] = Complex32::from_polar(mag, ph);
            }
            for k in 1..(n_bins - 1) {
                spectrum[self.n_fft - k] = spectrum[k].conj();
            }

            ifft.process(&mut spectrum);

            let frame_start = frame_idx * self.hop_length;
            for n in 0..self.n_fft {
                let sample = (spectrum[n].re / self.n_fft as f32) * self.window[n];
                let idx = frame_start + n;
                output[idx] += sample;
                envelope[idx] += self.window[n] * self.window[n];
            }
        }

        for i in 0..output.len() {
            if envelope[i] > 1e-8 {
                output[i] /= envelope[i];
            }
        }

        let pad = (self.n_fft.saturating_sub(self.hop_length)) / 2;
        let mut trimmed = if output.len() > 2 * pad {
            output[pad..output.len() - pad].to_vec()
        } else {
            output
        };
        normalize_detokenized_audio(&mut trimmed);

        Ok(trimmed)
    }
}

fn normalize_detokenized_audio(samples: &mut [f32]) {
    if samples.is_empty() {
        return;
    }

    let mut sum = 0.0f64;
    let mut finite_count = 0usize;
    for sample in samples.iter_mut() {
        if !sample.is_finite() {
            *sample = 0.0;
            continue;
        }
        sum += *sample as f64;
        finite_count += 1;
    }
    if finite_count > 0 {
        let mean = (sum / finite_count as f64) as f32;
        for sample in samples.iter_mut() {
            *sample -= mean;
        }
    }

    let mut peak = 0.0f32;
    for &sample in samples.iter() {
        peak = peak.max(sample.abs());
    }
    if peak > 0.95 {
        let scale = 0.95 / peak;
        for sample in samples.iter_mut() {
            *sample *= scale;
        }
    }

    let power = samples
        .iter()
        .map(|&sample| {
            let sample = sample as f64;
            sample * sample
        })
        .sum::<f64>();
    let rms = (power / samples.len() as f64).sqrt() as f32;
    let max_rms = 0.25f32;
    if rms > max_rms {
        let scale = max_rms / rms;
        for sample in samples.iter_mut() {
            *sample *= scale;
        }
    }

    for sample in samples.iter_mut() {
        *sample = sample.clamp(-1.0, 1.0);
    }
}

fn sliding_causal_mask(seq_len: usize, sliding_window: usize, device: &Device) -> Result<Tensor> {
    let mut mask = Vec::with_capacity(seq_len * seq_len);
    for i in 0..seq_len {
        for j in 0..seq_len {
            let valid = j <= i && (i - j) < sliding_window;
            mask.push(u8::from(!valid));
        }
    }
    Tensor::from_vec(mask, (seq_len, seq_len), device).map_err(Error::from)
}

fn hann_window(win_length: usize) -> Vec<f32> {
    if win_length <= 1 {
        return vec![1.0; win_length.max(1)];
    }

    (0..win_length)
        .map(|i| {
            let x = (2.0 * std::f32::consts::PI * i as f32) / win_length as f32;
            0.5 - 0.5 * x.cos()
        })
        .collect()
}

fn upsample_nearest_time(x: &Tensor, factor: usize) -> Result<Tensor> {
    if factor <= 1 {
        return Ok(x.clone());
    }
    let (_b, t, _d) = x.dims3()?;
    let indices: Vec<u32> = (0..t * factor).map(|i| (i / factor) as u32).collect();
    let indices = Tensor::from_vec(indices, t * factor, x.device())?.to_dtype(DType::U32)?;

    let x = x.transpose(1, 2)?.contiguous()?; // [B, D, T]
    let x = x.index_select(&indices, 2)?; // [B, D, T*factor]
    x.transpose(1, 2).map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use super::{normalize_detokenized_audio, upsample_nearest_time};
    use candle_core::{Device, Tensor};

    #[test]
    fn upsample_nearest_time_repeats_frames_without_contiguity_errors() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 2, 2), &device).unwrap();
        let upsampled = upsample_nearest_time(&x, 3).unwrap();

        let values = upsampled.to_vec3::<f32>().unwrap();
        assert_eq!(
            values,
            vec![vec![
                vec![1.0, 2.0],
                vec![1.0, 2.0],
                vec![1.0, 2.0],
                vec![3.0, 4.0],
                vec![3.0, 4.0],
                vec![3.0, 4.0],
            ]]
        );
    }

    #[test]
    fn normalize_detokenized_audio_sanitizes_invalid_and_hot_samples() {
        let mut samples = vec![
            f32::NAN,
            f32::INFINITY,
            -f32::INFINITY,
            3.5,
            -2.75,
            0.2,
            -0.2,
        ];
        normalize_detokenized_audio(&mut samples);

        assert!(samples.iter().all(|s| s.is_finite()));

        let peak = samples
            .iter()
            .fold(0.0f32, |acc, sample| acc.max(sample.abs()));
        assert!(peak <= 0.95 + 1e-6);
    }

    #[test]
    fn normalize_detokenized_audio_preserves_reasonable_levels() {
        let mut samples = vec![0.05f32, -0.08, 0.11, -0.07, 0.03];
        let before = samples.clone();
        normalize_detokenized_audio(&mut samples);

        for (lhs, rhs) in samples.iter().zip(before.iter()) {
            assert!((*lhs - *rhs).abs() < 0.2);
        }
    }
}
