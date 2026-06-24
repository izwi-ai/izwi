use candle_core::{
    CpuStorage, CustomOp3, DType, IndexOp, Layout, Result as CandleResult, Shape, Tensor,
};
use candle_nn::rnn::Direction;
use candle_nn::{
    ops, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Linear, Module, VarBuilder,
};
use candle_nn::{LSTMConfig, RNN};
use rayon::prelude::*;

use crate::error::{Error, Result};

use super::config::KokoroConfig;

#[derive(Debug, Clone)]
pub struct KokoroProsodyDebugOutput {
    pub duration_frames: Vec<u32>,
    pub expanded_frames: usize,
    pub f0_shape: Vec<usize>,
    pub n_shape: Vec<usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct KokoroProsodyOutput {
    pub duration_frames: Vec<u32>,
    pub expanded_frames: usize,
    pub f0: Tensor,
    pub n: Tensor,
}

#[derive(Debug)]
pub struct KokoroProsodyPredictor {
    duration_encoder: DurationEncoder,
    duration_lstm: BiLstm1,
    duration_proj: Linear,
    shared_lstm: BiLstm1,
    f0_blocks: Vec<AdainResBlk1d>,
    n_blocks: Vec<AdainResBlk1d>,
    f0_proj: Conv1d,
    n_proj: Conv1d,
    hidden_dim: usize,
    style_dim: usize,
}

impl KokoroProsodyPredictor {
    pub fn load(cfg: &KokoroConfig, vb: VarBuilder) -> Result<Self> {
        let root = vb.pp("module");
        let hidden_dim = cfg.hidden_dim;
        let style_dim = cfg.style_dim;
        let duration_encoder =
            DurationEncoder::load(style_dim, hidden_dim, cfg.n_layer, root.pp("text_encoder"))?;
        let duration_lstm = BiLstm1::load(hidden_dim + style_dim, hidden_dim / 2, root.pp("lstm"))?;
        let duration_proj = candle_nn::linear(
            hidden_dim,
            cfg.max_dur,
            root.pp("duration_proj.linear_layer"),
        )
        .map_err(Error::from)?;
        let shared_lstm = BiLstm1::load(hidden_dim + style_dim, hidden_dim / 2, root.pp("shared"))?;

        let mut f0_blocks = Vec::with_capacity(3);
        f0_blocks.push(AdainResBlk1d::load(
            hidden_dim,
            hidden_dim,
            style_dim,
            false,
            root.pp("F0.0"),
        )?);
        f0_blocks.push(AdainResBlk1d::load(
            hidden_dim,
            hidden_dim / 2,
            style_dim,
            true,
            root.pp("F0.1"),
        )?);
        f0_blocks.push(AdainResBlk1d::load(
            hidden_dim / 2,
            hidden_dim / 2,
            style_dim,
            false,
            root.pp("F0.2"),
        )?);

        let mut n_blocks = Vec::with_capacity(3);
        n_blocks.push(AdainResBlk1d::load(
            hidden_dim,
            hidden_dim,
            style_dim,
            false,
            root.pp("N.0"),
        )?);
        n_blocks.push(AdainResBlk1d::load(
            hidden_dim,
            hidden_dim / 2,
            style_dim,
            true,
            root.pp("N.1"),
        )?);
        n_blocks.push(AdainResBlk1d::load(
            hidden_dim / 2,
            hidden_dim / 2,
            style_dim,
            false,
            root.pp("N.2"),
        )?);

        let f0_proj = load_plain_conv1d(root.pp("F0_proj"), Conv1dConfig::default())?;
        let n_proj = load_plain_conv1d(root.pp("N_proj"), Conv1dConfig::default())?;

        Ok(Self {
            duration_encoder,
            duration_lstm,
            duration_proj,
            shared_lstm,
            f0_blocks,
            n_blocks,
            f0_proj,
            n_proj,
            hidden_dim,
            style_dim,
        })
    }

    pub(crate) fn forward(
        &self,
        d_en: &Tensor,      // [B, hidden_dim, T]
        ref_style: &Tensor, // [B, 256]
        speed: f32,
    ) -> Result<KokoroProsodyOutput> {
        let (_b, c, t) = d_en.dims3().map_err(Error::from)?;
        if c != self.hidden_dim {
            return Err(Error::InferenceError(format!(
                "Kokoro prosody debug expected d_en channels {}, got {}",
                self.hidden_dim, c
            )));
        }
        let (_b_style, style_total) = ref_style.dims2().map_err(Error::from)?;
        if style_total < self.style_dim * 2 {
            return Err(Error::InferenceError(format!(
                "Kokoro ref_style must have >= {} dims, got {}",
                self.style_dim * 2,
                style_total
            )));
        }

        let style = ref_style
            .i((.., self.style_dim..(self.style_dim * 2)))
            .map_err(Error::from)?; // predictor uses second half (128 dims)

        let d = self.duration_encoder.forward(d_en, &style)?; // [B, T, hidden+style]
        let x = self.duration_lstm.forward(&d)?; // [B, T, hidden]
        let duration_logits = self.duration_proj.forward(&x).map_err(Error::from)?; // [B,T,max_dur]
        let duration = ops::sigmoid(&duration_logits).map_err(Error::from)?;
        let duration = duration.sum_keepdim(2).map_err(Error::from)?; // [B,T,1]
        let speed = (speed as f64).max(0.1);
        let duration = (duration / speed).map_err(Error::from)?;
        let duration = duration.squeeze(2).map_err(Error::from)?; // [B,T]

        // Batch size >1 is not used in Kokoro TTS inference; keep implementation scoped.
        let (b, _t2) = duration.dims2().map_err(Error::from)?;
        if b != 1 {
            return Err(Error::InferenceError(
                "Kokoro prosody debug currently supports batch size 1 only".to_string(),
            ));
        }
        let duration_vec = duration.to_vec2::<f32>().map_err(Error::from)?;
        let dur_row = duration_vec.first().cloned().unwrap_or_default();
        let mut pred_dur = Vec::with_capacity(t);
        for v in dur_row {
            let r = v.round().max(1.0) as u32;
            pred_dur.push(r);
        }
        let expanded_frames: usize = pred_dur.iter().map(|&v| v as usize).sum();
        let pred_aln = build_alignment_matrix(&pred_dur, d.device())?; // [1,T,frames]

        let d_t = d.transpose(1, 2).map_err(Error::from)?; // [B, hidden+style, T]
        let en = d_t.matmul(&pred_aln).map_err(Error::from)?; // [B, hidden+style, frames]
        let (f0, n) = self.f0n_train(&en, &style)?;

        Ok(KokoroProsodyOutput {
            duration_frames: pred_dur,
            expanded_frames,
            f0,
            n,
        })
    }

    pub fn forward_debug(
        &self,
        d_en: &Tensor,      // [B, hidden_dim, T]
        ref_style: &Tensor, // [B, 256]
        speed: f32,
    ) -> Result<KokoroProsodyDebugOutput> {
        let out = self.forward(d_en, ref_style, speed)?;
        Ok(KokoroProsodyDebugOutput {
            duration_frames: out.duration_frames,
            expanded_frames: out.expanded_frames,
            f0_shape: out.f0.shape().dims().to_vec(),
            n_shape: out.n.shape().dims().to_vec(),
        })
    }

    fn f0n_train(&self, x: &Tensor, style: &Tensor) -> Result<(Tensor, Tensor)> {
        let x_bt = x.transpose(1, 2).map_err(Error::from)?; // [B,T,C]
        let shared = self.shared_lstm.forward(&x_bt)?; // [B,T,hidden]
        let mut f0 = shared.transpose(1, 2).map_err(Error::from)?; // [B,H,T]
        for block in &self.f0_blocks {
            f0 = block.forward(&f0, style)?;
        }
        let f0 = self.f0_proj.forward(&f0).map_err(Error::from)?; // [B,1,T]
        let f0 = f0.squeeze(1).map_err(Error::from)?; // [B,T]

        let mut n = shared.transpose(1, 2).map_err(Error::from)?;
        for block in &self.n_blocks {
            n = block.forward(&n, style)?;
        }
        let n = self.n_proj.forward(&n).map_err(Error::from)?;
        let n = n.squeeze(1).map_err(Error::from)?;
        Ok((f0, n))
    }
}

#[derive(Debug)]
struct DurationEncoder {
    blocks: Vec<DurationBlock>,
    style_dim: usize,
    hidden_dim: usize,
}

#[derive(Debug)]
enum DurationBlock {
    Lstm(BiLstm1),
    Ada(AdaLayerNorm),
}

impl DurationEncoder {
    fn load(style_dim: usize, hidden_dim: usize, nlayers: usize, vb: VarBuilder) -> Result<Self> {
        let mut blocks = Vec::with_capacity(nlayers * 2);
        for i in 0..nlayers {
            blocks.push(DurationBlock::Lstm(BiLstm1::load(
                hidden_dim + style_dim,
                hidden_dim / 2,
                vb.pp(format!("lstms.{}", i * 2)),
            )?));
            blocks.push(DurationBlock::Ada(AdaLayerNorm::load(
                style_dim,
                hidden_dim,
                vb.pp(format!("lstms.{}", i * 2 + 1)),
            )?));
        }
        Ok(Self {
            blocks,
            style_dim,
            hidden_dim,
        })
    }

    fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        let (b, c, t) = x.dims3().map_err(Error::from)?;
        if c != self.hidden_dim {
            return Err(Error::InferenceError(format!(
                "DurationEncoder expected hidden_dim {}, got {}",
                self.hidden_dim, c
            )));
        }
        let style_bt = style
            .unsqueeze(1)
            .map_err(Error::from)?
            .broadcast_as((b, t, self.style_dim))
            .map_err(Error::from)?;
        let style_ch = style_bt.transpose(1, 2).map_err(Error::from)?; // [B,S,T]

        let mut cur = x.transpose(1, 2).map_err(Error::from)?; // [B,T,H]
        cur = Tensor::cat(&[cur, style_bt.clone()], 2).map_err(Error::from)?; // [B,T,H+S]
        let mut cur = cur.transpose(1, 2).map_err(Error::from)?; // [B,H+S,T]

        for block in &self.blocks {
            match block {
                DurationBlock::Lstm(lstm) => {
                    let bt = cur.transpose(1, 2).map_err(Error::from)?;
                    let out = lstm.forward(&bt)?; // [B,T,H]
                    cur = out.transpose(1, 2).map_err(Error::from)?; // [B,H,T]
                }
                DurationBlock::Ada(ada) => {
                    let bt = cur.transpose(1, 2).map_err(Error::from)?; // [B,T,H]
                    let out = ada.forward(&bt, style)?; // [B,T,H]
                    let out = out.transpose(1, 2).map_err(Error::from)?; // [B,H,T]
                    cur = Tensor::cat(&[out, style_ch.clone()], 1).map_err(Error::from)?;
                    // [B,H+S,T]
                }
            }
        }

        cur.transpose(1, 2).map_err(Error::from) // [B,T,H+S]
    }
}

#[derive(Debug)]
struct AdaLayerNorm {
    channels: usize,
    eps: f64,
    fc: Linear,
}

impl AdaLayerNorm {
    fn load(style_dim: usize, channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            channels,
            eps: 1e-5,
            fc: candle_nn::linear(style_dim, channels * 2, vb.pp("fc")).map_err(Error::from)?,
        })
    }

    fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        let (_b, _t, c) = x.dims3().map_err(Error::from)?;
        if c != self.channels {
            return Err(Error::InferenceError(format!(
                "AdaLayerNorm expected channels {}, got {}",
                self.channels, c
            )));
        }
        let mean = x.mean_keepdim(2).map_err(Error::from)?;
        let var = x.var_keepdim(2).map_err(Error::from)?;
        let denom = (var + self.eps)
            .map_err(Error::from)?
            .sqrt()
            .map_err(Error::from)?;
        let xhat = (x.broadcast_sub(&mean).map_err(Error::from)?)
            .broadcast_div(&denom)
            .map_err(Error::from)?;

        let h = self.fc.forward(style).map_err(Error::from)?; // [B,2C]
        let chunks = h.chunk(2, 1).map_err(Error::from)?;
        let gamma = chunks[0].unsqueeze(1).map_err(Error::from)?; // [B,1,C]
        let beta = chunks[1].unsqueeze(1).map_err(Error::from)?;
        let y = xhat
            .broadcast_mul(&(gamma + 1.0f64).map_err(Error::from)?)
            .map_err(Error::from)?
            .broadcast_add(&beta)
            .map_err(Error::from)?;
        Ok(y)
    }
}

#[derive(Debug)]
pub(crate) struct AdaIN1d {
    channels: usize,
    eps: f64,
    fc: Linear,
}

impl AdaIN1d {
    pub(crate) fn load(style_dim: usize, channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            channels,
            eps: 1e-5,
            fc: candle_nn::linear(style_dim, channels * 2, vb.pp("fc")).map_err(Error::from)?,
        })
    }

    pub(crate) fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        let (_b, c, _t) = x.dims3().map_err(Error::from)?;
        if c != self.channels {
            return Err(Error::InferenceError(format!(
                "AdaIN1d expected channels {}, got {}",
                self.channels, c
            )));
        }
        let h = self.fc.forward(style).map_err(Error::from)?;
        let chunks = h.chunk(2, 1).map_err(Error::from)?;
        let gamma = chunks[0].unsqueeze(2).map_err(Error::from)?;
        let beta = chunks[1].unsqueeze(2).map_err(Error::from)?;

        if x.device().is_cpu()
            && x.dtype() == DType::F32
            && gamma.dtype() == DType::F32
            && beta.dtype() == DType::F32
        {
            if let Ok(out) = x.apply_op3_no_bwd(&gamma, &beta, &AdaIN1dCpuOp { eps: self.eps }) {
                return Ok(out);
            }
        }

        let mean = x.mean_keepdim(2).map_err(Error::from)?;
        let var = x.var_keepdim(2).map_err(Error::from)?;
        let denom = (var + self.eps)
            .map_err(Error::from)?
            .sqrt()
            .map_err(Error::from)?;
        let xhat = (x.broadcast_sub(&mean).map_err(Error::from)?)
            .broadcast_div(&denom)
            .map_err(Error::from)?;
        xhat.broadcast_mul(&(gamma + 1.0f64).map_err(Error::from)?)
            .map_err(Error::from)?
            .broadcast_add(&beta)
            .map_err(Error::from)
    }

    pub(crate) fn forward_snake(
        &self,
        x: &Tensor,
        style: &Tensor,
        alpha: &Tensor,
    ) -> Result<Tensor> {
        let (_b, c, _t) = x.dims3().map_err(Error::from)?;
        if c != self.channels {
            return Err(Error::InferenceError(format!(
                "AdaIN1d expected channels {}, got {}",
                self.channels, c
            )));
        }
        let h = self.fc.forward(style).map_err(Error::from)?;
        if x.device().is_cpu()
            && x.dtype() == DType::F32
            && h.dtype() == DType::F32
            && alpha.dtype() == DType::F32
        {
            if let Ok(out) = x.apply_op3_no_bwd(&h, alpha, &AdaIN1dSnakeCpuOp { eps: self.eps }) {
                return Ok(out);
            }
        }

        let normalized = self.forward(x, style)?;
        snake1d_expr(&normalized, alpha)
    }
}

#[derive(Debug, Clone, Copy)]
struct AdaIN1dCpuOp {
    eps: f64,
}

impl CustomOp3 for AdaIN1dCpuOp {
    fn name(&self) -> &'static str {
        "kokoro-adain1d-cpu"
    }

    fn cpu_fwd(
        &self,
        x_storage: &CpuStorage,
        x_layout: &Layout,
        gamma_storage: &CpuStorage,
        gamma_layout: &Layout,
        beta_storage: &CpuStorage,
        beta_layout: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        let x = match x_storage {
            CpuStorage::F32(values) => values,
            _ => candle_core::bail!("kokoro-adain1d-cpu only supports F32 input"),
        };
        let gamma = match gamma_storage {
            CpuStorage::F32(values) => values,
            _ => candle_core::bail!("kokoro-adain1d-cpu only supports F32 gamma"),
        };
        let beta = match beta_storage {
            CpuStorage::F32(values) => values,
            _ => candle_core::bail!("kokoro-adain1d-cpu only supports F32 beta"),
        };
        let x = match x_layout.contiguous_offsets() {
            Some((start, end)) => &x[start..end],
            None => candle_core::bail!("kokoro-adain1d-cpu requires contiguous input"),
        };
        let gamma = match gamma_layout.contiguous_offsets() {
            Some((start, end)) => &gamma[start..end],
            None => candle_core::bail!("kokoro-adain1d-cpu requires contiguous gamma"),
        };
        let beta = match beta_layout.contiguous_offsets() {
            Some((start, end)) => &beta[start..end],
            None => candle_core::bail!("kokoro-adain1d-cpu requires contiguous beta"),
        };

        let dims = x_layout.shape().dims();
        if dims.len() != 3 {
            candle_core::bail!("kokoro-adain1d-cpu expects [B,C,T] input")
        }
        let batch = dims[0];
        let channels = dims[1];
        let time = dims[2];
        if time < 2 {
            candle_core::bail!("kokoro-adain1d-cpu requires time length >= 2")
        }
        validate_adain_affine_shape(gamma_layout.shape().dims(), batch, channels, "gamma")?;
        validate_adain_affine_shape(beta_layout.shape().dims(), batch, channels, "beta")?;
        if gamma.len() != batch * channels || beta.len() != batch * channels {
            candle_core::bail!(
                "kokoro-adain1d-cpu affine storage length mismatch: gamma={}, beta={}, expected={}",
                gamma.len(),
                beta.len(),
                batch * channels
            )
        }

        let mut out = vec![0.0f32; x.len()];
        let denom_scale = (time - 1) as f32;
        out.par_chunks_mut(time)
            .enumerate()
            .for_each(|(row, out_row)| {
                let b = row / channels;
                let c = row % channels;
                let base = row * time;
                let values = &x[base..base + time];
                let mean = values.iter().copied().sum::<f32>() / time as f32;
                let var = values
                    .iter()
                    .map(|v| {
                        let d = *v - mean;
                        d * d
                    })
                    .sum::<f32>()
                    / denom_scale;
                let inv_std = 1.0f32 / (var + self.eps as f32).sqrt();
                let affine_idx = b * channels + c;
                let scale = gamma[affine_idx] + 1.0;
                let bias = beta[affine_idx];
                for t in 0..time {
                    let v = values[t];
                    out_row[t] = (v - mean) * inv_std * scale + bias;
                }
            });
        Ok((CpuStorage::F32(out), x_layout.shape().clone()))
    }
}

fn validate_adain_affine_shape(
    dims: &[usize],
    batch: usize,
    channels: usize,
    name: &str,
) -> CandleResult<()> {
    match dims {
        [b, c, 1] if *b == batch && *c == channels => Ok(()),
        [b, c] if *b == batch && *c == channels => Ok(()),
        _ => candle_core::bail!(
            "kokoro-adain1d-cpu {name} shape {:?} cannot broadcast to [{batch},{channels},T]",
            dims
        ),
    }
}

#[derive(Debug, Clone, Copy)]
struct AdaIN1dSnakeCpuOp {
    eps: f64,
}

impl CustomOp3 for AdaIN1dSnakeCpuOp {
    fn name(&self) -> &'static str {
        "kokoro-adain1d-snake-cpu"
    }

    fn cpu_fwd(
        &self,
        x_storage: &CpuStorage,
        x_layout: &Layout,
        h_storage: &CpuStorage,
        h_layout: &Layout,
        alpha_storage: &CpuStorage,
        alpha_layout: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        let x = match x_storage {
            CpuStorage::F32(values) => values,
            _ => candle_core::bail!("kokoro-adain1d-snake-cpu only supports F32 input"),
        };
        let h = match h_storage {
            CpuStorage::F32(values) => values,
            _ => candle_core::bail!("kokoro-adain1d-snake-cpu only supports F32 affine"),
        };
        let alpha = match alpha_storage {
            CpuStorage::F32(values) => values,
            _ => candle_core::bail!("kokoro-adain1d-snake-cpu only supports F32 alpha"),
        };
        let x = match x_layout.contiguous_offsets() {
            Some((start, end)) => &x[start..end],
            None => candle_core::bail!("kokoro-adain1d-snake-cpu requires contiguous input"),
        };
        let h = match h_layout.contiguous_offsets() {
            Some((start, end)) => &h[start..end],
            None => candle_core::bail!("kokoro-adain1d-snake-cpu requires contiguous affine"),
        };
        let alpha = match alpha_layout.contiguous_offsets() {
            Some((start, end)) => &alpha[start..end],
            None => candle_core::bail!("kokoro-adain1d-snake-cpu requires contiguous alpha"),
        };

        let dims = x_layout.shape().dims();
        if dims.len() != 3 {
            candle_core::bail!("kokoro-adain1d-snake-cpu expects [B,C,T] input")
        }
        let batch = dims[0];
        let channels = dims[1];
        let time = dims[2];
        if time < 2 {
            candle_core::bail!("kokoro-adain1d-snake-cpu requires time length >= 2")
        }
        match h_layout.shape().dims() {
            [b, two_c] if *b == batch && *two_c == channels * 2 => {}
            other => candle_core::bail!(
                "kokoro-adain1d-snake-cpu affine shape {:?} cannot broadcast to [{batch},{channels},T]",
                other
            ),
        }
        let alpha_by_channel = validate_snake_alpha_shape(alpha_layout.shape().dims(), channels)?;
        if alpha_by_channel && alpha.len() != channels {
            candle_core::bail!(
                "kokoro-adain1d-snake-cpu alpha storage length {}, expected {}",
                alpha.len(),
                channels
            )
        }
        if !alpha_by_channel && alpha.len() != 1 {
            candle_core::bail!("kokoro-adain1d-snake-cpu scalar alpha has invalid storage length")
        }

        let mut out = vec![0.0f32; x.len()];
        let denom_scale = (time - 1) as f32;
        out.par_chunks_mut(time)
            .enumerate()
            .for_each(|(row, out_row)| {
                let b = row / channels;
                let c = row % channels;
                let base = row * time;
                let values = &x[base..base + time];
                let mean = values.iter().copied().sum::<f32>() / time as f32;
                let var = values
                    .iter()
                    .map(|v| {
                        let d = *v - mean;
                        d * d
                    })
                    .sum::<f32>()
                    / denom_scale;
                let inv_std = 1.0f32 / (var + self.eps as f32).sqrt();
                let h_base = b * channels * 2;
                let scale = h[h_base + c] + 1.0;
                let bias = h[h_base + channels + c];
                let alpha = if alpha_by_channel { alpha[c] } else { alpha[0] };
                for t in 0..time {
                    let y = (values[t] - mean) * inv_std * scale + bias;
                    let s = (y * alpha).sin();
                    out_row[t] = y + (s * s) / alpha;
                }
            });
        Ok((CpuStorage::F32(out), x_layout.shape().clone()))
    }
}

fn validate_snake_alpha_shape(dims: &[usize], channels: usize) -> CandleResult<bool> {
    match dims {
        [1] => Ok(false),
        [c] if *c == channels => Ok(true),
        [1, c, 1] if *c == channels => Ok(true),
        [c, 1] if *c == channels => Ok(true),
        _ => candle_core::bail!(
            "kokoro-adain1d-snake-cpu alpha shape {:?} cannot broadcast to channel count {}",
            dims,
            channels
        ),
    }
}

fn snake1d_expr(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    let ax = x.broadcast_mul(alpha).map_err(Error::from)?;
    let sin_sq = ax.sin().map_err(Error::from)?.sqr().map_err(Error::from)?;
    let scaled = sin_sq.broadcast_div(alpha).map_err(Error::from)?;
    x.broadcast_add(&scaled).map_err(Error::from)
}

#[derive(Debug)]
pub(crate) struct AdainResBlk1d {
    norm1: AdaIN1d,
    norm2: AdaIN1d,
    conv1: Conv1d,
    conv2: Conv1d,
    conv1x1: Option<Conv1d>,
    pool: Option<ConvTranspose1d>,
    upsample: bool,
    learned_sc: bool,
}

impl AdainResBlk1d {
    pub(crate) fn load(
        dim_in: usize,
        dim_out: usize,
        style_dim: usize,
        upsample: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let learned_sc = dim_in != dim_out;
        let conv1 = load_weight_norm_conv1d(
            vb.pp("conv1"),
            Conv1dConfig {
                padding: 1,
                stride: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
        )?;
        let conv2 = load_weight_norm_conv1d(
            vb.pp("conv2"),
            Conv1dConfig {
                padding: 1,
                stride: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
        )?;
        let conv1x1 = if learned_sc {
            Some(load_weight_norm_conv1d(
                vb.pp("conv1x1"),
                Conv1dConfig::default(),
            )?)
        } else {
            None
        };
        let pool = if upsample {
            Some(load_weight_norm_conv_transpose1d(
                vb.pp("pool"),
                ConvTranspose1dConfig {
                    padding: 1,
                    output_padding: 1,
                    stride: 2,
                    dilation: 1,
                    groups: dim_in,
                },
            )?)
        } else {
            None
        };

        Ok(Self {
            norm1: AdaIN1d::load(style_dim, dim_in, vb.pp("norm1"))?,
            norm2: AdaIN1d::load(style_dim, dim_out, vb.pp("norm2"))?,
            conv1,
            conv2,
            conv1x1,
            pool,
            upsample,
            learned_sc,
        })
    }

    pub(crate) fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        let shortcut = self.shortcut(x)?;
        let residual = self.residual(x, style)?;
        ((shortcut + residual).map_err(Error::from)? * (1.0f64 / 2.0f64.sqrt()))
            .map_err(Error::from)
    }

    fn shortcut(&self, x: &Tensor) -> Result<Tensor> {
        let mut y = if self.upsample {
            upsample_nearest_2x_1d(x)?
        } else {
            x.clone()
        };
        if self.learned_sc {
            y = self
                .conv1x1
                .as_ref()
                .expect("conv1x1 present when learned_sc")
                .forward(&y)
                .map_err(Error::from)?;
        }
        Ok(y)
    }

    fn residual(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        let mut y = self.norm1.forward(x, style)?;
        y = ops::leaky_relu(&y, 0.2).map_err(Error::from)?;
        if let Some(pool) = &self.pool {
            y = pool.forward(&y).map_err(Error::from)?;
        }
        y = self.conv1.forward(&y).map_err(Error::from)?;
        y = self.norm2.forward(&y, style)?;
        y = ops::leaky_relu(&y, 0.2).map_err(Error::from)?;
        self.conv2.forward(&y).map_err(Error::from)
    }
}

#[derive(Debug)]
pub(crate) struct BiLstm1 {
    fwd: candle_nn::LSTM,
    bwd: candle_nn::LSTM,
    hidden_dim: usize,
}

impl BiLstm1 {
    pub(crate) fn load(input_dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        let mut cfg_f = LSTMConfig::default();
        cfg_f.direction = Direction::Forward;
        let mut cfg_b = LSTMConfig::default();
        cfg_b.direction = Direction::Backward;
        Ok(Self {
            fwd: candle_nn::lstm(input_dim, hidden_dim, cfg_f, vb.clone()).map_err(Error::from)?,
            bwd: candle_nn::lstm(input_dim, hidden_dim, cfg_b, vb).map_err(Error::from)?,
            hidden_dim,
        })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _c) = x.dims3().map_err(Error::from)?;
        let x = x.contiguous().map_err(Error::from)?;
        let fwd_states = self.fwd.seq(&x).map_err(Error::from)?;
        let fwd = self
            .fwd
            .states_to_tensor(&fwd_states)
            .map_err(Error::from)?;

        let rev_idx: Vec<u32> = (0..t as u32).rev().collect();
        let rev_idx = Tensor::new(&rev_idx[..], x.device()).map_err(Error::from)?;
        let x_rev = x
            .index_select(&rev_idx, 1)
            .map_err(Error::from)?
            .contiguous()
            .map_err(Error::from)?;
        let bwd_states = self.bwd.seq(&x_rev).map_err(Error::from)?;
        let bwd_rev = self
            .bwd
            .states_to_tensor(&bwd_states)
            .map_err(Error::from)?;
        let bwd = bwd_rev
            .contiguous()
            .map_err(Error::from)?
            .index_select(&rev_idx, 1)
            .map_err(Error::from)?;

        let out = Tensor::cat(&[fwd, bwd], 2).map_err(Error::from)?;
        let (_, _, out_c) = out.dims3().map_err(Error::from)?;
        if out_c != self.hidden_dim * 2 {
            return Err(Error::InferenceError(format!(
                "BiLstm1 unexpected output channels {} (expected {}) for batch {}",
                out_c,
                self.hidden_dim * 2,
                b
            )));
        }
        Ok(out)
    }
}

pub(crate) fn build_alignment_matrix(
    durations: &[u32],
    device: &candle_core::Device,
) -> Result<Tensor> {
    let t = durations.len();
    let frames: usize = durations.iter().map(|&v| v as usize).sum();
    let frames = frames.max(1);
    let mut data = vec![0.0f32; t * frames];
    let mut col = 0usize;
    for (row, &dur) in durations.iter().enumerate() {
        let dur = dur.max(1) as usize;
        for _ in 0..dur {
            if col >= frames {
                break;
            }
            data[row * frames + col] = 1.0;
            col += 1;
        }
    }
    Tensor::from_vec(data, (1, t, frames), device).map_err(Error::from)
}

fn upsample_nearest_2x_1d(x: &Tensor) -> Result<Tensor> {
    let (b, c, t) = x.dims3().map_err(Error::from)?;
    x.unsqueeze(3)
        .map_err(Error::from)?
        .broadcast_as((b, c, t, 2))
        .map_err(Error::from)?
        .reshape((b, c, t * 2))
        .map_err(Error::from)
}

pub(crate) fn load_plain_conv1d(vb: VarBuilder, cfg: Conv1dConfig) -> Result<Conv1d> {
    let w = vb
        .get_unchecked_dtype("weight", DType::F32)
        .map_err(Error::from)?;
    let b = if vb.contains_tensor("bias") {
        Some(
            vb.get_unchecked_dtype("bias", DType::F32)
                .map_err(Error::from)?,
        )
    } else {
        None
    };
    Ok(Conv1d::new(w, b, cfg))
}

pub(crate) fn load_weight_norm_conv1d(vb: VarBuilder, cfg: Conv1dConfig) -> Result<Conv1d> {
    let wv = vb
        .get_unchecked_dtype("weight_v", DType::F32)
        .map_err(Error::from)?;
    let wg = vb
        .get_unchecked_dtype("weight_g", DType::F32)
        .map_err(Error::from)?;
    let b = if vb.contains_tensor("bias") {
        Some(
            vb.get_unchecked_dtype("bias", DType::F32)
                .map_err(Error::from)?,
        )
    } else {
        None
    };
    let w = fuse_weight_norm_dim0(&wv, &wg)?;
    Ok(Conv1d::new(w, b, cfg))
}

pub(crate) fn load_weight_norm_conv_transpose1d(
    vb: VarBuilder,
    cfg: ConvTranspose1dConfig,
) -> Result<ConvTranspose1d> {
    let wv = vb
        .get_unchecked_dtype("weight_v", DType::F32)
        .map_err(Error::from)?;
    let wg = vb
        .get_unchecked_dtype("weight_g", DType::F32)
        .map_err(Error::from)?;
    let b = if vb.contains_tensor("bias") {
        Some(
            vb.get_unchecked_dtype("bias", DType::F32)
                .map_err(Error::from)?,
        )
    } else {
        None
    };
    let w = fuse_weight_norm_dim0(&wv, &wg)?;
    Ok(ConvTranspose1d::new(w, b, cfg))
}

pub(crate) fn fuse_weight_norm_dim0(weight_v: &Tensor, weight_g: &Tensor) -> Result<Tensor> {
    let rank = weight_v.rank();
    let sq = (weight_v * weight_v).map_err(Error::from)?;
    let norm = match rank {
        2 => sq.sum_keepdim(1).map_err(Error::from)?,
        3 => sq.sum_keepdim((1, 2)).map_err(Error::from)?,
        _ => {
            return Err(Error::ModelLoadError(format!(
                "Unsupported weight_norm tensor rank {} for Kokoro loader",
                rank
            )))
        }
    }
    .sqrt()
    .map_err(Error::from)?;
    let scale = weight_g.broadcast_div(&norm).map_err(Error::from)?;
    weight_v.broadcast_mul(&scale).map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adain1d_cpu_op_matches_candle_expression() {
        let device = candle_core::Device::Cpu;
        let eps = 1e-5;
        let x = Tensor::from_vec(
            vec![-0.4f32, 0.0, 0.25, 0.7, -1.2, 0.3, 0.9, 1.4],
            (1, 2, 4),
            &device,
        )
        .expect("x tensor");
        let gamma =
            Tensor::from_vec(vec![0.2f32, -0.15], (1, 2, 1), &device).expect("gamma tensor");
        let beta = Tensor::from_vec(vec![0.05f32, -0.2], (1, 2, 1), &device).expect("beta tensor");

        let fused = x
            .apply_op3_no_bwd(&gamma, &beta, &AdaIN1dCpuOp { eps })
            .expect("fused adain")
            .flatten_all()
            .expect("flatten fused")
            .to_vec1::<f32>()
            .expect("fused values");

        let mean = x.mean_keepdim(2).expect("mean");
        let var = x.var_keepdim(2).expect("var");
        let denom = (var + eps).expect("eps").sqrt().expect("sqrt");
        let xhat = x
            .broadcast_sub(&mean)
            .expect("sub mean")
            .broadcast_div(&denom)
            .expect("div denom");
        let reference = xhat
            .broadcast_mul(&(gamma + 1.0f64).expect("gamma plus one"))
            .expect("mul gamma")
            .broadcast_add(&beta)
            .expect("add beta")
            .flatten_all()
            .expect("flatten reference")
            .to_vec1::<f32>()
            .expect("reference values");

        assert_eq!(fused.len(), reference.len());
        for (got, expected) in fused.iter().zip(reference.iter()) {
            assert!((got - expected).abs() < 1e-5, "{got} != {expected}");
        }
    }

    #[test]
    fn adain1d_snake_cpu_op_matches_separate_expression() {
        let device = candle_core::Device::Cpu;
        let eps = 1e-5;
        let x = Tensor::from_vec(
            vec![-0.4f32, 0.0, 0.25, 0.7, -1.2, 0.3, 0.9, 1.4],
            (1, 2, 4),
            &device,
        )
        .expect("x tensor");
        let affine = Tensor::from_vec(vec![0.2f32, -0.15, 0.05, -0.2], (1, 4), &device)
            .expect("affine tensor");
        let alpha = Tensor::from_vec(vec![0.7f32, 1.3], (1, 2, 1), &device).expect("alpha tensor");

        let fused = x
            .apply_op3_no_bwd(&affine, &alpha, &AdaIN1dSnakeCpuOp { eps })
            .expect("fused adain snake")
            .flatten_all()
            .expect("flatten fused")
            .to_vec1::<f32>()
            .expect("fused values");

        let gamma = affine
            .narrow(1, 0, 2)
            .expect("gamma")
            .unsqueeze(2)
            .expect("gamma unsqueeze");
        let beta = affine
            .narrow(1, 2, 2)
            .expect("beta")
            .unsqueeze(2)
            .expect("beta unsqueeze");
        let mean = x.mean_keepdim(2).expect("mean");
        let var = x.var_keepdim(2).expect("var");
        let denom = (var + eps).expect("eps").sqrt().expect("sqrt");
        let xhat = x
            .broadcast_sub(&mean)
            .expect("sub mean")
            .broadcast_div(&denom)
            .expect("div denom");
        let adain = xhat
            .broadcast_mul(&(gamma + 1.0f64).expect("gamma plus one"))
            .expect("mul gamma")
            .broadcast_add(&beta)
            .expect("add beta");
        let reference = snake1d_expr(&adain, &alpha)
            .expect("snake")
            .flatten_all()
            .expect("flatten reference")
            .to_vec1::<f32>()
            .expect("reference values");

        assert_eq!(fused.len(), reference.len());
        for (got, expected) in fused.iter().zip(reference.iter()) {
            assert!((got - expected).abs() < 1e-5, "{got} != {expected}");
        }
    }
}
