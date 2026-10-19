use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Tensor};
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use crate::error::{Error, Result};
use crate::models::shared::weights::gguf::GgufLoader;
use crate::models::shared::weights::token_table::QuantizedTokenTable;

use super::config::{Lfm25AudioDecoderConfig, Lfm2BackboneConfig, LFM25_AUDIO_AUDIO_VOCAB_SIZE};
use crate::models::architectures::lfm2::backbone::{Lfm2StatelessStream, QuantizedLfm2Backbone};

pub struct Lfm25AudioDetokenizer {
    fused_embedding: QuantizedTokenTable,
    backbone: Mutex<QuantizedLfm2Backbone>,
    istft_window: Vec<f32>,
    istft_window_sq: Vec<f32>,
    ifft: Arc<dyn Fft<f32>>,
    upsample_factor: usize,
    n_fft: usize,
    hop_length: usize,
    codebooks: usize,
    codebook_offsets: Vec<u32>,
    codebook_offsets_tensor: Tensor,
}

/// Incremental detokenizer state ([`Lfm25AudioDetokenizer::decode_stream_step`]):
/// the code frames consumed, the backbone's per-layer contexts and the
/// ISTFT's unfinished overlap. A clone is an exact checkpoint.
#[derive(Debug, Clone, Default)]
pub(crate) struct Lfm25DetokenizerStream {
    frames: usize,
    backbone: Lfm2StatelessStream,
    istft: IstftStream,
}

impl Lfm25DetokenizerStream {
    /// Code frames detokenized so far.
    pub(crate) fn frames(&self) -> usize {
        self.frames
    }
}

/// The streaming ISTFT's overlap-add buffers from absolute sample `base`.
#[derive(Debug, Clone, Default)]
struct IstftStream {
    spectral_frames: usize,
    base: usize,
    emitted: usize,
    output: Vec<f32>,
    envelope: Vec<f32>,
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
        // Kept quantized: rows are gathered and dequantized on demand.
        let fused_embedding =
            QuantizedTokenTable::new(decoder_loader.load_qtensor("emb.emb.weight", device)?)?;
        let backbone =
            QuantizedLfm2Backbone::load_for_embeds(backbone_loader, backbone_config, device)?;
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
        let codebook_offsets_tensor = Tensor::from_vec(
            codebook_offsets.clone(),
            (decoder_config.codebooks,),
            device,
        )?;

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
            codebook_offsets_tensor,
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
            let guard = self.backbone.lock().map_err(|_| {
                Error::InferenceError("LFM2.5 Audio detokenizer mutex poisoned".to_string())
            })?;
            let backbone_started = Instant::now();
            let hidden = guard.forward_embeds_stateless(&upsampled)?;
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

    pub fn decode_token_ids_with_profile(
        &self,
        audio_codes: &Tensor,
    ) -> Result<(Vec<f32>, Lfm25DetokenizerProfile)> {
        let mut profile = Lfm25DetokenizerProfile::default();
        let (frames, codebooks) = audio_codes.dims2()?;
        if codebooks != self.codebooks {
            return Err(Error::InvalidInput(format!(
                "Expected {} audio codebooks, received {codebooks}",
                self.codebooks
            )));
        }
        if frames == 0 {
            return Ok((Vec::new(), profile));
        }

        let embedding_started = Instant::now();
        let embeds = self.fused_embeddings_from_token_ids(audio_codes)?;
        profile.embedding_ms = elapsed_ms(embedding_started);

        let upsample_started = Instant::now();
        let upsampled = self.upsample_nearest(&embeds)?;
        profile.upsample_ms = elapsed_ms(upsample_started);

        let projected = {
            let guard = self.backbone.lock().map_err(|_| {
                Error::InferenceError("LFM2.5 Audio detokenizer mutex poisoned".to_string())
            })?;
            let backbone_started = Instant::now();
            let hidden = guard.forward_embeds_stateless(&upsampled)?;
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

    /// Detokenize the next code frames, `new_codes[codebook][frame]` (the
    /// frames after those `stream` has consumed), and return the samples of
    /// the whole-utterance [`Self::decode`] that no later frame can change
    /// (LFT-5). The detokenizer is causal (ShortConv plus a 30-position
    /// window), so the backbone runs only the new positions against bounded
    /// per-layer contexts, and the ISTFT emits each sample once every frame
    /// overlapping it has been added. [`Self::finish_stream`] emits the rest.
    pub(crate) fn decode_stream_step(
        &self,
        stream: &mut Lfm25DetokenizerStream,
        new_codes: &[Vec<u32>],
        device: &Device,
    ) -> Result<Vec<f32>> {
        if new_codes.len() != self.codebooks {
            return Err(Error::InvalidInput(format!(
                "Expected {} audio codebooks, received {}",
                self.codebooks,
                new_codes.len()
            )));
        }
        let frames = new_codes.first().map(Vec::len).unwrap_or(0);
        if frames == 0 {
            return Ok(Vec::new());
        }
        if new_codes.iter().any(|codes| codes.len() != frames)
            || new_codes.iter().flatten().any(|code| *code >= 2048)
        {
            return Err(Error::InvalidInput(
                "Detokenizer stream codes must share a frame length and lie in [0, 2047]".into(),
            ));
        }
        let embeds = self.fused_embeddings(new_codes, device)?;
        let upsampled = self.upsample_nearest(&embeds)?;
        let projected = {
            let guard = self.backbone.lock().map_err(|_| {
                Error::InferenceError("LFM2.5 Audio detokenizer mutex poisoned".to_string())
            })?;
            let hidden = guard.forward_embeds_stream(&mut stream.backbone, &upsampled)?;
            guard.project_hidden(&hidden)?
        };
        let freq_bins = self.n_fft / 2 + 1;
        let projected = projected.i(0)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        for frame in &projected {
            if frame.len() != freq_bins * 2 {
                return Err(Error::InferenceError(format!(
                    "Detokenizer output width mismatch: expected {}, found {}",
                    freq_bins * 2,
                    frame.len()
                )));
            }
            self.istft_add_frame(&mut stream.istft, frame, freq_bins);
        }
        stream.frames += frames;
        let ready = stream.istft.spectral_frames * self.hop_length;
        Ok(self.istft_emit(&mut stream.istft, ready))
    }

    /// The samples after the last [`Self::decode_stream_step`]: the tail the
    /// whole-utterance ISTFT ends with.
    pub(crate) fn finish_stream(&self, stream: &mut Lfm25DetokenizerStream) -> Vec<f32> {
        let frames = stream.istft.spectral_frames;
        if frames == 0 {
            return Vec::new();
        }
        let pad = (self.n_fft - self.hop_length) / 2;
        let end = (frames - 1) * self.hop_length + self.n_fft - pad;
        self.istft_emit(&mut stream.istft, end)
    }

    /// Overlap-add one spectral frame, as [`Self::istft_same`] does.
    fn istft_add_frame(&self, istft: &mut IstftStream, frame: &[f32], freq_bins: usize) {
        let mut full = vec![Complex::<f32>::new(0.0, 0.0); self.n_fft];
        for freq_idx in 0..freq_bins {
            full[freq_idx] =
                Complex::from_polar(frame[freq_idx].exp(), frame[freq_bins + freq_idx]);
        }
        for freq_idx in 1..(freq_bins - 1) {
            full[self.n_fft - freq_idx] = full[freq_idx].conj();
        }
        self.ifft.process(&mut full);
        let first = istft.spectral_frames * self.hop_length;
        let needed = first + self.n_fft - istft.base;
        if istft.output.len() < needed {
            istft.output.resize(needed, 0.0);
            istft.envelope.resize(needed, 0.0);
        }
        for sample_idx in 0..self.n_fft {
            let sample = full[sample_idx].re / self.n_fft as f32 * self.istft_window[sample_idx];
            let index = first + sample_idx - istft.base;
            istft.output[index] += sample;
            istft.envelope[index] += self.istft_window_sq[sample_idx];
        }
        istft.spectral_frames += 1;
    }

    /// Normalized samples from the last emitted one up to absolute index
    /// `end` (exclusive), cropped by the "same" padding like
    /// [`Self::istft_same`]; the emitted prefix leaves the buffers.
    fn istft_emit(&self, istft: &mut IstftStream, end: usize) -> Vec<f32> {
        let pad = (self.n_fft - self.hop_length) / 2;
        let start = istft.emitted.max(pad);
        if end <= start {
            return Vec::new();
        }
        let samples = (start..end)
            .map(|index| {
                let local = index - istft.base;
                let env = istft.envelope.get(local).copied().unwrap_or(0.0);
                if env > 1e-11 {
                    istft.output[local] / env
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        istft.emitted = end;
        let drop = end - istft.base;
        istft.output.drain(..drop.min(istft.output.len()));
        istft.envelope.drain(..drop.min(istft.envelope.len()));
        istft.base = end;
        samples
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
        let embeds = self.fused_embedding.embed(&ids)?; // [T, C, D]
        let embeds = embeds.sum(1)?.affine(1.0 / self.codebooks as f64, 0.0)?;
        embeds.unsqueeze(0).map_err(Error::from)
    }

    fn fused_embeddings_from_token_ids(&self, audio_codes: &Tensor) -> Result<Tensor> {
        let ids = audio_codes
            .broadcast_add(&self.codebook_offsets_tensor)?
            .contiguous()?;
        let embeds = self.fused_embedding.embed(&ids)?; // [T, C, D]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::numeric_check::{max_abs_diff, xorshift_tensor};

    /// A tiny detokenizer: two codebooks, the tiny windowed LFM2 backbone
    /// (window 30) with a 6-wide head, and a 4-point ISTFT with hop 2.
    fn tiny_detokenizer() -> Lfm25AudioDetokenizer {
        let device = Device::Cpu;
        let codebooks = 2;
        let rows = codebooks * (LFM25_AUDIO_AUDIO_VOCAB_SIZE - 1) + 1;
        let table = xorshift_tensor(&[rows, 8], 0xDE70, 1.0, &device).unwrap();
        let n_fft = 4;
        let window = vec![0.1f32, 0.6, 1.0, 0.6];
        let codebook_offsets = (0..codebooks)
            .map(|idx| (idx * (LFM25_AUDIO_AUDIO_VOCAB_SIZE - 1)) as u32)
            .collect::<Vec<_>>();
        Lfm25AudioDetokenizer {
            fused_embedding: QuantizedTokenTable::new(
                candle_core::quantized::QTensor::quantize(
                    &table,
                    candle_core::quantized::GgmlDType::F32,
                )
                .unwrap(),
            )
            .unwrap(),
            backbone: Mutex::new(
                crate::models::architectures::lfm2::backbone::tests::tiny_windowed_backbone_with_head(
                    Some(30),
                    6,
                ),
            ),
            istft_window_sq: window.iter().map(|v| v * v).collect(),
            istft_window: window,
            ifft: FftPlanner::<f32>::new().plan_fft_inverse(n_fft),
            upsample_factor: 6,
            n_fft,
            hop_length: 2,
            codebooks,
            codebook_offsets_tensor: Tensor::from_vec(
                codebook_offsets.clone(),
                codebooks,
                &device,
            )
            .unwrap(),
            codebook_offsets,
        }
    }

    /// LFT-5: streaming the detokenizer in uneven steps, then finishing,
    /// reproduces the whole-utterance decode sample for sample: backbone
    /// contexts and the ISTFT overlap carry across steps.
    #[test]
    fn streamed_detokenizer_matches_the_whole_decode() {
        let detokenizer = tiny_detokenizer();
        let mut state = 0x1234_5678u64;
        let codes = (0..2)
            .map(|_| {
                (0..40)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        (state % 2048) as u32
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let whole = detokenizer.decode(&codes, &Device::Cpu).unwrap();
        let mut stream = Lfm25DetokenizerStream::default();
        let mut streamed = Vec::new();
        let mut start = 0;
        for step in [1usize, 5, 3, 9, 22] {
            let next = codes
                .iter()
                .map(|c| c[start..start + step].to_vec())
                .collect::<Vec<_>>();
            let chunk = detokenizer
                .decode_stream_step(&mut stream, &next, &Device::Cpu)
                .unwrap();
            assert!(!chunk.is_empty());
            streamed.extend(chunk);
            start += step;
        }
        assert_eq!(stream.frames(), 40);
        streamed.extend(detokenizer.finish_stream(&mut stream));
        assert_eq!(streamed.len(), whole.len());
        let scale = whole.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(scale > 1e-3, "degenerate output");
        let error = max_abs_diff(&streamed, &whole);
        assert!(
            error <= 1e-5 * scale.max(1.0),
            "streamed vs whole max abs {error}"
        );
    }
}
