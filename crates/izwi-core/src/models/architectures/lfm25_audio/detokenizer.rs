use std::sync::Mutex;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, Module};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::error::{Error, Result};
use crate::models::shared::weights::gguf::GgufLoader;

use super::backbone::QuantizedLfm2Backbone;
use super::config::{Lfm25AudioDecoderConfig, Lfm2BackboneConfig, LFM25_AUDIO_AUDIO_VOCAB_SIZE};

pub struct Lfm25AudioDetokenizer {
    fused_embedding: Embedding,
    backbone: Mutex<QuantizedLfm2Backbone>,
    istft_window: Vec<f32>,
    upsample_factor: usize,
    n_fft: usize,
    hop_length: usize,
    codebooks: usize,
    codebook_offsets: Vec<u32>,
}

impl Lfm25AudioDetokenizer {
    pub fn load(
        backbone_loader: &GgufLoader,
        decoder_loader: &GgufLoader,
        backbone_config: Lfm2BackboneConfig,
        decoder_config: &Lfm25AudioDecoderConfig,
        device: &Device,
    ) -> Result<Self> {
        let fused_embedding = load_embedding_any(
            decoder_loader,
            device,
            &[
                "emb.emb.weight".to_string(),
            ],
        )?;
        let backbone = QuantizedLfm2Backbone::load(backbone_loader, backbone_config, device)?;
        let istft_window = load_vector_any(
            decoder_loader,
            device,
            &["istft.window".to_string()],
            decoder_config.output_n_fft,
        )?
        .to_vec1::<f32>()?;
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
            upsample_factor: decoder_config.detokenizer_upsample_factor,
            n_fft: decoder_config.output_n_fft,
            hop_length: decoder_config.output_hop_length,
            codebooks: decoder_config.codebooks,
            codebook_offsets,
        })
    }

    pub fn decode(&self, audio_codes: &[Vec<u32>], device: &Device) -> Result<Vec<f32>> {
        if audio_codes.len() != self.codebooks {
            return Err(Error::InvalidInput(format!(
                "Expected {} audio codebooks, received {}",
                self.codebooks,
                audio_codes.len()
            )));
        }

        let frames = audio_codes.first().map(Vec::len).unwrap_or(0);
        if frames == 0 {
            return Ok(Vec::new());
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

        let embeds = self.fused_embeddings(audio_codes, device)?;
        let upsampled = self.upsample_nearest(&embeds)?;

        let projected = {
            let mut guard = self.backbone.lock().map_err(|_| {
                Error::InferenceError("LFM2.5 Audio detokenizer mutex poisoned".to_string())
            })?;
            guard.reset_state();
            let hidden = guard.forward_embeds(&upsampled, 0)?;
            guard.project_hidden(&hidden)?
        };

        self.projected_to_waveform(&projected)
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
        let projected = projected.i(0)?.to_dtype(DType::F32)?;
        let (frames, width) = projected.dims2()?;
        let freq_bins = self.n_fft / 2 + 1;
        if width != freq_bins * 2 {
            return Err(Error::InferenceError(format!(
                "Detokenizer output width mismatch: expected {}, found {width}",
                freq_bins * 2
            )));
        }

        let log_abs = projected.narrow(1, 0, freq_bins)?.to_vec2::<f32>()?;
        let angle = projected
            .narrow(1, freq_bins, freq_bins)?
            .to_vec2::<f32>()?;

        let mut spectrum = vec![vec![Complex::<f32>::new(0.0, 0.0); frames]; freq_bins];
        for frame_idx in 0..frames {
            for freq_idx in 0..freq_bins {
                spectrum[freq_idx][frame_idx] = Complex::from_polar(
                    log_abs[frame_idx][freq_idx].exp(),
                    angle[frame_idx][freq_idx],
                );
            }
        }

        self.istft_same(&spectrum)
    }

    fn istft_same(&self, spectrum: &[Vec<Complex<f32>>]) -> Result<Vec<f32>> {
        let freq_bins = spectrum.len();
        let frames = spectrum.first().map(Vec::len).unwrap_or(0);
        if freq_bins != self.n_fft / 2 + 1 || frames == 0 {
            return Ok(Vec::new());
        }

        let pad = (self.n_fft - self.hop_length) / 2;
        let output_size = (frames - 1) * self.hop_length + self.n_fft;
        let mut output = vec![0.0f32; output_size];
        let mut envelope = vec![0.0f32; output_size];

        let mut planner = FftPlanner::<f32>::new();
        let ifft = planner.plan_fft_inverse(self.n_fft);

        for frame_idx in 0..frames {
            let mut full = vec![Complex::<f32>::new(0.0, 0.0); self.n_fft];
            for freq_idx in 0..freq_bins {
                full[freq_idx] = spectrum[freq_idx][frame_idx];
            }
            for freq_idx in 1..(freq_bins - 1) {
                full[self.n_fft - freq_idx] = spectrum[freq_idx][frame_idx].conj();
            }

            ifft.process(&mut full);
            for sample_idx in 0..self.n_fft {
                let sample =
                    full[sample_idx].re / self.n_fft as f32 * self.istft_window[sample_idx];
                let out_idx = frame_idx * self.hop_length + sample_idx;
                output[out_idx] += sample;
                envelope[out_idx] += self.istft_window[sample_idx] * self.istft_window[sample_idx];
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
