use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, Module};
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use crate::error::{Error, Result};
use crate::models::shared::weights::gguf::GgufLoader;

use super::backbone::QuantizedLfm2Backbone;
use super::config::{Lfm25AudioDecoderConfig, Lfm2BackboneConfig, LFM25_AUDIO_AUDIO_VOCAB_SIZE};

pub struct Lfm25AudioDetokenizer {
    fused_embedding: Embedding,
    backbone: Mutex<QuantizedLfm2Backbone>,
    istft_window: Vec<f32>,
    istft_window_sq: Vec<f32>,
    ifft: Arc<dyn Fft<f32>>,
    upsample_factor: usize,
    n_fft: usize,
    hop_length: usize,
    codebooks: usize,
    codebook_offsets: Vec<u32>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Lfm25DetokenizerProfile {
    pub embedding_ms: f64,
    pub upsample_ms: f64,
    pub backbone_forward_ms: f64,
    pub projection_ms: f64,
    pub waveform_prepare_ms: f64,
    pub readback_ms: f64,
    pub istft_ms: f64,
}

impl Lfm25AudioDetokenizer {
    pub fn load(
        backbone_loader: &GgufLoader,
        decoder_loader: &GgufLoader,
        backbone_config: Lfm2BackboneConfig,
        decoder_config: &Lfm25AudioDecoderConfig,
        device: &Device,
    ) -> Result<Self> {
        let fused_embedding =
            load_embedding_any(decoder_loader, device, &["emb.emb.weight".to_string()])?;
        let backbone = QuantizedLfm2Backbone::load(backbone_loader, backbone_config, device)?;
        let istft_window = load_vector_any(
            decoder_loader,
            device,
            &["istft.window".to_string()],
            decoder_config.output_n_fft,
        )?
        .to_vec1::<f32>()?;
        let istft_window_sq = istft_window
            .iter()
            .map(|value| value * value)
            .collect::<Vec<_>>();
        let mut planner = FftPlanner::<f32>::new();
        let ifft = planner.plan_fft_inverse(decoder_config.output_n_fft);
        let codebook_offsets = (0..decoder_config.codebooks)
            .map(|idx| {
                u32::try_from(idx * (LFM25_AUDIO_AUDIO_VOCAB_SIZE - 1)).map_err(|_| {
                    Error::ModelLoadError(format!(
                        "Detokenizer codebook offset out of range for codebook {idx}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            fused_embedding,
            backbone: Mutex::new(backbone),
            istft_window,
            istft_window_sq,
            ifft,
            upsample_factor: decoder_config.detokenizer_upsample_factor,
            n_fft: decoder_config.output_n_fft,
            hop_length: decoder_config.output_hop_length,
            codebooks: decoder_config.codebooks,
            codebook_offsets,
        })
    }

    pub fn decode(&self, audio_codes: &[Vec<u32>], device: &Device) -> Result<Vec<f32>> {
        self.decode_with_profile(audio_codes, device)
            .map(|(samples, _profile)| samples)
    }

    pub fn decode_with_profile(
        &self,
        audio_codes: &[Vec<u32>],
        device: &Device,
    ) -> Result<(Vec<f32>, Lfm25DetokenizerProfile)> {
        let mut profile = Lfm25DetokenizerProfile::default();
        if audio_codes.len() != self.codebooks {
            return Err(Error::InvalidInput(format!(
                "Expected {} audio codebooks, received {}",
                self.codebooks,
                audio_codes.len()
            )));
        }

        let frames = audio_codes.first().map(Vec::len).unwrap_or(0);
        if frames == 0 {
            return Ok((Vec::new(), profile));
        }
        if audio_codes.iter().any(|codes| codes.len() != frames) {
            return Err(Error::InvalidInput(
                "All audio codebooks must have the same frame length".to_string(),
            ));
        }
        if audio_codes
            .iter()
            .flat_map(|codes| codes.iter().copied())
            .any(|code| code >= 2048)
        {
            return Err(Error::InvalidInput(
                "Detokenizer expects audio codes in the range [0, 2047]".to_string(),
            ));
        }

        let embedding_started = Instant::now();
        let embeds = self.fused_embeddings(audio_codes, device)?;
        profile.embedding_ms = elapsed_ms(embedding_started);

        let upsample_started = Instant::now();
        let upsampled = self.upsample_nearest(&embeds)?;
        profile.upsample_ms = elapsed_ms(upsample_started);

        let projected = {
            let mut guard = self.backbone.lock().map_err(|_| {
                Error::InferenceError("LFM2.5 Audio detokenizer mutex poisoned".to_string())
            })?;
            guard.reset_state();
            let backbone_started = Instant::now();
            let hidden = guard.forward_embeds(&upsampled, 0)?;
            profile.backbone_forward_ms = elapsed_ms(backbone_started);
            let projection_started = Instant::now();
            let projected = guard.project_hidden(&hidden)?;
            profile.projection_ms = elapsed_ms(projection_started);
            projected
        };

        let (samples, waveform_profile) = self.projected_to_waveform_with_profile(&projected)?;
        profile.waveform_prepare_ms = waveform_profile.waveform_prepare_ms;
        profile.readback_ms = waveform_profile.readback_ms;
        profile.istft_ms = waveform_profile.istft_ms;
        Ok((samples, profile))
    }

    fn fused_embeddings(&self, audio_codes: &[Vec<u32>], device: &Device) -> Result<Tensor> {
        let frames = audio_codes[0].len();
        let mut flat = Vec::with_capacity(frames * self.codebooks);
        for frame_idx in 0..frames {
            for (codebook_idx, codes) in audio_codes.iter().enumerate() {
                flat.push(codes[frame_idx].saturating_add(self.codebook_offsets[codebook_idx]));
            }
        }

        let ids = Tensor::from_vec(flat, (frames, self.codebooks), device)?;
        let embeds = self.fused_embedding.forward(&ids)?; // [T, C, D]
        let embeds = embeds.sum(1)?.affine(1.0 / self.codebooks as f64, 0.0)?;
        embeds.unsqueeze(0).map_err(Error::from)
    }

    fn upsample_nearest(&self, embeds: &Tensor) -> Result<Tensor> {
        let (batch, frames, hidden) = embeds.dims3()?;
        embeds
            .unsqueeze(2)?
            .broadcast_as((batch, frames, self.upsample_factor, hidden))?
            .reshape((batch, frames * self.upsample_factor, hidden))
            .map_err(Error::from)
    }

    fn projected_to_waveform(&self, projected: &Tensor) -> Result<Vec<f32>> {
        self.projected_to_waveform_with_profile(projected)
            .map(|(samples, _profile)| samples)
    }

    fn projected_to_waveform_with_profile(
        &self,
        projected: &Tensor,
    ) -> Result<(Vec<f32>, Lfm25DetokenizerProfile)> {
        let mut profile = Lfm25DetokenizerProfile::default();
        let waveform_prepare_started = Instant::now();
        let projected = projected.i(0)?.to_dtype(DType::F32)?;
        let (_frames, width) = projected.dims2()?;
        let freq_bins = self.n_fft / 2 + 1;
        if width != freq_bins * 2 {
            return Err(Error::InferenceError(format!(
                "Detokenizer output width mismatch: expected {}, found {width}",
                freq_bins * 2
            )));
        }
        profile.waveform_prepare_ms = elapsed_ms(waveform_prepare_started);

        let readback_started = Instant::now();
        let projected = projected.to_vec2::<f32>()?;
        profile.readback_ms = elapsed_ms(readback_started);

        let istft_started = Instant::now();
        let samples = self.istft_same(&projected, freq_bins)?;
        profile.istft_ms = elapsed_ms(istft_started);
        Ok((samples, profile))
    }

    fn istft_same(&self, projected: &[Vec<f32>], freq_bins: usize) -> Result<Vec<f32>> {
        let frames = projected.len();
        if freq_bins != self.n_fft / 2 + 1 || frames == 0 {
            return Ok(Vec::new());
        }

        let pad = (self.n_fft - self.hop_length) / 2;
        let output_size = (frames - 1) * self.hop_length + self.n_fft;
        let mut output = vec![0.0f32; output_size];
        let mut envelope = vec![0.0f32; output_size];
        let mut full = vec![Complex::<f32>::new(0.0, 0.0); self.n_fft];

        for frame_idx in 0..frames {
            full.fill(Complex::new(0.0, 0.0));
            let frame = &projected[frame_idx];
            for freq_idx in 0..freq_bins {
                full[freq_idx] =
                    Complex::from_polar(frame[freq_idx].exp(), frame[freq_bins + freq_idx]);
            }
            for freq_idx in 1..(freq_bins - 1) {
                full[self.n_fft - freq_idx] = full[freq_idx].conj();
            }

            self.ifft.process(&mut full);
            for sample_idx in 0..self.n_fft {
                let sample =
                    full[sample_idx].re / self.n_fft as f32 * self.istft_window[sample_idx];
                let out_idx = frame_idx * self.hop_length + sample_idx;
                output[out_idx] += sample;
                envelope[out_idx] += self.istft_window_sq[sample_idx];
            }
        }

        let mut cropped = output[pad..output_size - pad].to_vec();
        for (sample, env) in cropped
            .iter_mut()
            .zip(envelope[pad..output_size - pad].iter().copied())
        {
            if env > 1e-11 {
                *sample /= env;
            } else {
                *sample = 0.0;
            }
        }

        Ok(cropped)
    }
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn load_embedding_any(loader: &GgufLoader, device: &Device, names: &[String]) -> Result<Embedding> {
    let weight = load_tensor_any(loader, device, names, DType::F32)?;
    let (_, dim) = weight.dims2()?;
    Ok(Embedding::new(weight, dim))
}

fn load_vector_any(
    loader: &GgufLoader,
    device: &Device,
    names: &[String],
    expected_len: usize,
) -> Result<Tensor> {
    let tensor = load_tensor_any(loader, device, names, DType::F32)?;
    if tensor.elem_count() != expected_len {
        return Err(Error::ModelLoadError(format!(
            "Vector shape mismatch for {}: expected {expected_len}, found {}",
            names.join(" | "),
            tensor.elem_count()
        )));
    }
    if tensor.rank() == 1 {
        Ok(tensor)
    } else {
        tensor.flatten_all().map_err(Error::from)
    }
}

fn load_tensor_any(
    loader: &GgufLoader,
    device: &Device,
    names: &[String],
    dtype: DType,
) -> Result<Tensor> {
    for name in names {
        if loader.has_tensor(name) {
            return loader.load_tensor(name, dtype, device);
        }
    }
    Err(Error::ModelLoadError(format!(
        "Missing GGUF tensor; tried {}",
        names.join(" | ")
    )))
}
