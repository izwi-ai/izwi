//! The LiquidCodec decoder (`modeling_liquid_codec.py`): 12 FSQ codes per
//! 80 ms frame to 24 kHz audio.
//!
//! Each code's four FSQ digits (levels 8, 7, 6, 6) give a 48-dim latent per
//! frame; `project_out` (48 -> 256) and `from_quantizer` (256 -> 512) map it
//! to the decoder width, a nearest x6 upsample gives 75 rows per second, an
//! 8-layer LFM3 trunk (width-7 ShortConvs, gated attention over the last 30
//! rows, sandwich norms) runs over them, and a linear layer gives 641 log
//! magnitudes and phases per row for an ISTFT (n_fft 1280, hop 320, Hann)
//! with overlap-add normalized by the summed squared window.
//!
//! The decoder is causal with bounded reach, so a stream decodes exactly
//! what the whole utterance does: the trunk keeps each layer's last inputs,
//! and the ISTFT emits a sample once every frame that overlaps it has been
//! added.
//!
//! In the GGUF bundle the tensors are the mmproj's `a.gen.wav.*`.

use std::sync::Arc;

use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Module, Tensor};
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use crate::error::{Error, Result};
use crate::models::architectures::lfm2::backbone::{Lfm2StatelessStream, QuantizedLfm2Backbone};
use crate::models::architectures::lfm2::config::{Lfm2BackboneConfig, Lfm3TrunkFeatures};
use crate::models::shared::weights::gguf::GgufLoader;

use super::config::{AUDIO_EOS_CODE, CODEBOOKS};

const FSQ_LEVELS: [u32; 4] = [8, 7, 6, 6];
const LATENT: usize = CODEBOOKS * FSQ_LEVELS.len();
const WIDTH: usize = 512;
const UPSAMPLE: usize = 6;
const N_FFT: usize = 1_280;
const HOP: usize = 320;
const BINS: usize = N_FFT / 2 + 1;
/// Trunk rows (75 per second) the RoPE tables cover: about 14.5 minutes of
/// one stream.
const CONTEXT_ROWS: usize = 65_536;

pub(crate) struct LiquidCodecDecoder {
    /// `[48, 256]`.
    project_out_t: Tensor,
    project_out_bias: Tensor,
    from_quantizer: QMatMul,
    from_quantizer_bias: Tensor,
    trunk: QuantizedLfm2Backbone,
    out_proj: QMatMul,
    out_proj_bias: Tensor,
    ifft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    window_sq: Vec<f32>,
    device: Device,
}

/// One utterance's decoder state.
#[derive(Debug, Clone, Default)]
pub(crate) struct LiquidCodecStream {
    trunk: Lfm2StatelessStream,
    /// Overlap-add sums from absolute sample `base` on.
    output: Vec<f32>,
    envelope: Vec<f32>,
    base: usize,
    emitted: usize,
    rows: usize,
}

/// The decoder trunk's tensor names under the LFM2 backbone's GGUF names.
fn codec_tensor_name(stored: &str) -> Option<String> {
    if stored == "a.gen.wav.tfm.output_norm.weight" {
        return Some("output_norm.weight".into());
    }
    let (layer, leaf) = stored.strip_prefix("a.gen.wav.tfm.blk.")?.split_once('.')?;
    let leaf = match leaf {
        "ln1.weight" => "attn_norm.weight",
        "ln2.weight" => "ffn_norm.weight",
        "attn_post_norm.weight" => "post_attention_norm.weight",
        "ffn_post_norm.weight" => "post_ffw_norm.weight",
        "conv_dw.weight" => "shortconv.conv.weight",
        "conv_pw1.weight" => "shortconv.in_proj.weight",
        "conv_pw2.weight" => "shortconv.out_proj.weight",
        "attn_out.weight" => "attn_output.weight",
        other => other,
    };
    Some(format!("blk.{layer}.{leaf}"))
}

/// FSQ digits of one code as latents in [-1, 1]: digit `k` is
/// `code / basis_k mod levels_k`, centred and scaled by `levels_k / 2`.
pub(crate) fn fsq_latents(code: u32) -> [f32; 4] {
    let mut basis = 1u32;
    let mut out = [0f32; 4];
    for (slot, levels) in out.iter_mut().zip(FSQ_LEVELS) {
        let half = (levels / 2) as f32;
        *slot = (((code / basis) % levels) as f32 - half) / half;
        basis *= levels;
    }
    out
}

impl LiquidCodecStream {
    /// Whether any frame has been decoded into this stream.
    pub(crate) fn has_rows(&self) -> bool {
        self.rows > 0
    }
}

impl LiquidCodecDecoder {
    pub(crate) fn load(mmproj: &GgufLoader, device: &Device) -> Result<Self> {
        let dense = |name: &str| mmproj.load_tensor(name, DType::F32, device);
        let qmatmul = |name: &str| -> Result<QMatMul> {
            Ok(QMatMul::from_qtensor(mmproj.load_qtensor(name, device)?)?)
        };
        let attention_head_count_kv = (0..8)
            .map(|layer| {
                if mmproj.has_tensor(&format!("a.gen.wav.tfm.blk.{layer}.attn_q.weight")) {
                    2
                } else {
                    0
                }
            })
            .collect::<Vec<_>>();
        let config = Lfm2BackboneConfig {
            architecture: "liquid_codec".into(),
            block_count: 8,
            context_length: CONTEXT_ROWS,
            embedding_length: WIDTH,
            embedding_length_out: None,
            feed_forward_length: Some(2_304),
            attention_head_count: 8,
            attention_head_count_kv,
            attention_layer_norm_rms_epsilon: 1e-5,
            attention_sliding_window: Some(30),
            rope_freq_base: 1_000_000.0,
            shortconv_l_cache: 7,
            lfm3: Some(Lfm3TrunkFeatures {
                attention_output_gate: true,
                sandwich_norm: true,
                ..Lfm3TrunkFeatures::default()
            }),
        };
        let trunk =
            QuantizedLfm2Backbone::load_trunk(&mmproj.renamed(codec_tensor_name), config, device)?;
        let window = (0..N_FFT)
            .map(|n| {
                (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / N_FFT as f64).cos()) as f32
            })
            .collect::<Vec<_>>();
        Ok(Self {
            project_out_t: dense("a.gen.wav.quant.first.out_proj.weight")?
                .t()?
                .contiguous()?,
            project_out_bias: dense("a.gen.wav.quant.first.out_proj.bias")?,
            from_quantizer: qmatmul("a.gen.wav.tfm.in_proj.weight")?,
            from_quantizer_bias: dense("a.gen.wav.tfm.in_proj.bias")?,
            trunk,
            out_proj: qmatmul("a.gen.wav.tfm.out_proj.weight")?,
            out_proj_bias: dense("a.gen.wav.tfm.out_proj.bias")?,
            ifft: FftPlanner::new().plan_fft_inverse(N_FFT),
            window_sq: window.iter().map(|w| w * w).collect(),
            window,
            device: device.clone(),
        })
    }

    /// The samples `frames` complete in `stream` (`finalize` flushes the
    /// tail). Decoding an utterance in any split gives the samples of
    /// decoding it whole.
    pub(crate) fn decode_stream(
        &self,
        stream: &mut LiquidCodecStream,
        frames: &[[u32; CODEBOOKS]],
        finalize: bool,
    ) -> Result<Vec<f32>> {
        if frames.iter().flatten().any(|code| *code >= AUDIO_EOS_CODE) {
            return Err(Error::InvalidInput(
                "LiquidCodec frames must hold speech codes only".into(),
            ));
        }
        if !frames.is_empty() {
            let rows = self.spectral_rows(&mut stream.trunk, frames)?;
            for row in &rows {
                self.overlap_add(stream, row);
            }
        }
        let end = if finalize {
            if stream.rows == 0 {
                return Ok(Vec::new());
            }
            (stream.rows - 1) * HOP + N_FFT - (N_FFT - HOP) / 2
        } else {
            stream.rows * HOP
        };
        Ok(self.emit(stream, end))
    }

    /// The whole utterance's samples.
    pub(crate) fn decode(&self, frames: &[[u32; CODEBOOKS]]) -> Result<Vec<f32>> {
        self.decode_stream(&mut LiquidCodecStream::default(), frames, true)
    }

    /// `[frames * 6]` rows of 1282 log magnitudes and phases.
    fn spectral_rows(
        &self,
        trunk: &mut Lfm2StatelessStream,
        frames: &[[u32; CODEBOOKS]],
    ) -> Result<Vec<Vec<f32>>> {
        let latents = frames
            .iter()
            .flat_map(|frame| frame.iter().flat_map(|code| fsq_latents(*code)))
            .collect::<Vec<_>>();
        let latents = Tensor::from_vec(latents, (frames.len(), LATENT), &self.device)?;
        let x = latents
            .matmul(&self.project_out_t)?
            .broadcast_add(&self.project_out_bias)?;
        let x = self
            .from_quantizer
            .forward(&x)?
            .broadcast_add(&self.from_quantizer_bias)?;
        // `interpolate(..., mode="nearest-exact")` by an integer factor
        // repeats each frame.
        let x = x
            .unsqueeze(1)?
            .broadcast_as((frames.len(), UPSAMPLE, WIDTH))?
            .reshape((1, frames.len() * UPSAMPLE, WIDTH))?
            .contiguous()?;
        let hidden = self.trunk.forward_embeds_stream(trunk, &x)?;
        let projected = self
            .out_proj
            .forward(&hidden)?
            .broadcast_add(&self.out_proj_bias)?
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?;
        Ok(projected.to_vec2::<f32>()?)
    }

    fn overlap_add(&self, stream: &mut LiquidCodecStream, row: &[f32]) {
        let mut full = vec![Complex::<f32>::new(0.0, 0.0); N_FFT];
        for bin in 0..BINS {
            full[bin] = Complex::from_polar(row[bin].exp(), row[BINS + bin]);
        }
        for bin in 1..BINS - 1 {
            full[N_FFT - bin] = full[bin].conj();
        }
        // The real part of a full inverse FFT is `irfft`'s output: the DC and
        // Nyquist imaginary parts only reach the imaginary output.
        self.ifft.process(&mut full);
        let first = stream.rows * HOP;
        let needed = first + N_FFT - stream.base;
        if stream.output.len() < needed {
            stream.output.resize(needed, 0.0);
            stream.envelope.resize(needed, 0.0);
        }
        for (index, value) in full.iter().enumerate() {
            let local = first + index - stream.base;
            stream.output[local] += value.re / N_FFT as f32 * self.window[index];
            stream.envelope[local] += self.window_sq[index];
        }
        stream.rows += 1;
    }

    /// Normalized samples from the last emitted one up to absolute sample
    /// `end`, past the leading `(n_fft - hop) / 2` trim.
    fn emit(&self, stream: &mut LiquidCodecStream, end: usize) -> Vec<f32> {
        let start = stream.emitted.max((N_FFT - HOP) / 2);
        if end <= start {
            return Vec::new();
        }
        let samples = (start..end)
            .map(|index| {
                let local = index - stream.base;
                let envelope = stream.envelope.get(local).copied().unwrap_or(0.0);
                if envelope > 1e-11 {
                    stream.output[local] / envelope
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        stream.emitted = end;
        let drop = (end - stream.base).min(stream.output.len());
        stream.output.drain(..drop);
        stream.envelope.drain(..drop);
        stream.base = end;
        samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fsq_digits_cover_every_code_once() {
        let mut seen = std::collections::HashSet::new();
        for code in 0..AUDIO_EOS_CODE {
            let latents = fsq_latents(code);
            assert!(latents.iter().all(|value| (-1.0..=1.0).contains(value)));
            assert!(
                seen.insert(latents.map(f32::to_bits)),
                "code {code} repeats"
            );
        }
        // Code 0 is every digit's lowest level.
        assert_eq!(fsq_latents(0), [-1.0, -1.0, -1.0, -1.0]);
        // Basis [1, 8, 56, 336]: 8 + 56 + 336 is the second level of the
        // last three digits.
        assert_eq!(fsq_latents(400), [-1.0, -2.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0]);
    }

    #[test]
    fn codec_tensors_take_the_backbone_names() {
        assert_eq!(
            codec_tensor_name("a.gen.wav.tfm.blk.2.attn_post_norm.weight").unwrap(),
            "blk.2.post_attention_norm.weight"
        );
        assert_eq!(
            codec_tensor_name("a.gen.wav.tfm.blk.7.ffn_post_norm.weight").unwrap(),
            "blk.7.post_ffw_norm.weight"
        );
        assert_eq!(
            codec_tensor_name("a.gen.wav.tfm.blk.4.attn_gate.weight").unwrap(),
            "blk.4.attn_gate.weight"
        );
        assert_eq!(codec_tensor_name("a.gen.wav.tfm.in_proj.weight"), None);
    }
}
