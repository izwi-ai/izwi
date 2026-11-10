//! Linear layers of the family, in each backend's resident form.
//!
//! | Weight | CPU | Metal | CUDA |
//! |---|---|---|---|
//! | FP8 (ModelOpt) | Q8_0 (D4) | raw E4M3 + scale grid, F32 activations | raw E4M3 + scale grid, BF16 activations |
//! | BF16 | F32 | F32 | BF16 |
//!
//! FP8 weights are W8A16 by default: `decode(byte) * weight_scale` exactly,
//! no activation quantization. Metal keeps F32 activations, so relu² expert
//! intermediates never meet F16's range. CUDA's FP8 provider needs
//! `rows % 64 == 0` and `cols % 128 == 0`, so CUDA pads both with zeros
//! (exact: zero rows give zero outputs, zero columns read nothing); an MoE
//! expert keeps the padded width between its up and down projections.
//!
//! Every layer takes and returns F32 activations `[m, in]` → `[m, out]`.
//! A [`Pass::Step`] (decode) always runs the row-wise kernels and a
//! [`Pass::Span`] (prefill) the tile kernels, so a row's arithmetic never
//! depends on how many other rows share its step.

use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{DType, Device, Module, Tensor};
use half::f16;

use super::checkpoint::{Fp8Weight, NemotronCheckpoint};
use super::precision::Fp8Mode;
use crate::error::{Error, Result};
use crate::kernels::cuda::fp8::{self, Fp8Epilogue};
use crate::kernels::metal_fp8_gemm;

/// CPU prefill expands Q8_0 weights in tiles of this many rows above
/// [`CPU_TILE_MIN_ROWS`] activation rows.
const CPU_TILE_ROWS: usize = 256;
const CPU_TILE_MIN_ROWS: usize = 5;

/// Whether a call is part of a prefill span or a decode step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pass {
    Span,
    Step,
}

/// Which backend a layer was loaded for (`Device` alone does not carry the
/// policy).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residency {
    Cpu,
    Metal,
    Cuda,
}

impl Residency {
    pub fn of(device: &Device) -> Self {
        match device {
            Device::Cpu => Self::Cpu,
            Device::Metal(_) => Self::Metal,
            Device::Cuda(_) => Self::Cuda,
        }
    }

    /// The dtype dense (non-FP8) weights are kept in.
    pub fn dense_dtype(self) -> DType {
        match self {
            Self::Cpu | Self::Metal => DType::F32,
            Self::Cuda => DType::BF16,
        }
    }

    /// The activation dtype FP8 projections take.
    pub fn fp8_activation_dtype(self) -> DType {
        match self {
            Self::Cpu | Self::Metal => DType::F32,
            Self::Cuda => DType::BF16,
        }
    }
}

/// Runs an op over packed rows (row `r`'s `lengths[r]` tokens in turn)
/// split by each row's `class` of its token count: `run(ids, class)` takes
/// the token indices of one class (`None` when every row is of that class)
/// and returns their outputs in that order. Returns every token's output in
/// token order.
pub(crate) fn by_row_class(
    lengths: &[usize],
    device: &Device,
    class: impl Fn(usize) -> bool,
    mut run: impl FnMut(Option<&Tensor>, bool) -> Result<Tensor>,
) -> Result<Tensor> {
    let classes = lengths
        .iter()
        .map(|&tokens| class(tokens))
        .collect::<Vec<_>>();
    match classes.first() {
        None => {
            return Err(Error::InferenceError(
                "a packed call needs at least one row".into(),
            ))
        }
        Some(&first) if classes.iter().all(|&c| c == first) => return run(None, first),
        Some(_) => {}
    }
    let mut ids = [Vec::new(), Vec::new()];
    let mut start = 0u32;
    for (&tokens, &c) in lengths.iter().zip(&classes) {
        let end = start
            + u32::try_from(tokens)
                .map_err(|_| Error::InferenceError("a packed row has too many tokens".into()))?;
        ids[usize::from(c)].extend(start..end);
        start = end;
    }
    let mut parts = Vec::with_capacity(2);
    let mut position = vec![0u32; start as usize];
    let mut at = 0u32;
    for (c, ids) in [false, true].into_iter().zip(&ids) {
        for &token in ids {
            position[token as usize] = at;
            at += 1;
        }
        let tensor = Tensor::from_vec(ids.clone(), ids.len(), device)?;
        parts.push(run(Some(&tensor), c)?);
    }
    let position = Tensor::from_vec(position, start as usize, device)?;
    Ok(Tensor::cat(&parts, 0)?.index_select(&position, 0)?)
}

enum Weight {
    Q8(QMatMul),
    Fp8 { bytes: Tensor, scales: Tensor },
    Dense(Tensor),
}

/// A loaded linear layer.
pub struct NtLinear {
    weight: Weight,
    bias: Option<Tensor>,
    out_features: usize,
    in_features: usize,
    /// Resident rows and columns (CUDA pads FP8 weights).
    padded_out: usize,
    padded_in: usize,
    /// The static activation scale (`vllm-fp8` parity mode only).
    input_scale: Option<f32>,
    weight_scale: Option<f32>,
}

impl std::fmt::Debug for NtLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match &self.weight {
            Weight::Q8(_) => "q8_0",
            Weight::Fp8 { .. } => "fp8",
            Weight::Dense(_) => "dense",
        };
        f.debug_struct("NtLinear")
            .field("kind", &kind)
            .field("out", &self.out_features)
            .field("in", &self.in_features)
            .finish()
    }
}

fn load_error(message: impl Into<String>) -> Error {
    Error::ModelLoadError(format!(
        "Nemotron 3.5 Transcribe linear: {}",
        message.into()
    ))
}

/// Q8_0 blocks (F16 scale, 32 signed bytes) of row-major `[rows, cols]`
/// values produced by `value(row, col)`, as ggml's reference quantizer.
fn q8_0_bytes(rows: usize, cols: usize, value: impl Fn(usize, &mut [f32]) + Sync) -> Vec<u8> {
    use rayon::prelude::*;
    let row_bytes = cols / 32 * 34;
    let mut out = vec![0u8; rows * row_bytes];
    out.par_chunks_mut(row_bytes)
        .enumerate()
        .for_each(|(row, bytes)| {
            let mut values = vec![0f32; cols];
            value(row, &mut values);
            for (block, chunk) in values.as_chunks::<32>().0.iter().enumerate() {
                let amax = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
                let d = amax / 127.0;
                let inv = if d != 0.0 { 1.0 / d } else { 0.0 };
                let dst = &mut bytes[block * 34..(block + 1) * 34];
                dst[..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
                for (slot, v) in dst[2..].iter_mut().zip(chunk) {
                    *slot = (v * inv).round() as i8 as u8;
                }
            }
        });
    out
}

fn q8_matmul(rows: usize, cols: usize, bytes: Vec<u8>) -> Result<QMatMul> {
    let storage = QStorage::from_data(
        std::borrow::Cow::Owned(bytes),
        &Device::Cpu,
        GgmlDType::Q8_0,
    )?;
    Ok(QMatMul::QTensor(std::sync::Arc::new(QTensor::new(
        storage,
        (rows, cols),
    )?)))
}

/// The E4M3 RNE fake quantization of the `vllm-fp8` mode: `x` (host F32) to
/// `s * decode(encode(clamp(x / s, ±448)))`.
pub(crate) fn fake_quantize_e4m3(values: &mut [f32], scale: f32) {
    use crate::kernels::cuda::fp8::{decode_e4m3fn, encode_e4m3fn};
    let inverse = 1.0 / scale;
    for v in values {
        *v = decode_e4m3fn(encode_e4m3fn(*v * inverse)) * scale;
    }
}

impl NtLinear {
    /// The FP8 linear `module` (`[rows, cols]`), with an optional BF16
    /// `bias`.
    pub fn load_fp8(
        checkpoint: &NemotronCheckpoint,
        module: &str,
        rows: usize,
        cols: usize,
        bias: bool,
        device: &Device,
        mode: Fp8Mode,
    ) -> Result<Self> {
        let weight = checkpoint.fp8(module, rows, cols)?;
        Self::from_fp8(weight, rows, cols, device, mode)?
            .with_bias(checkpoint, module, bias, device)
    }

    /// Builds the layer from FP8 bytes already read (also the parity mode's
    /// requantized q/k/v shards).
    pub fn from_fp8(
        weight: Fp8Weight<'_>,
        rows: usize,
        cols: usize,
        device: &Device,
        mode: Fp8Mode,
    ) -> Result<Self> {
        let residency = Residency::of(device);
        let (input_scale, weight_scale) = match mode {
            Fp8Mode::W8A16 => (None, None),
            Fp8Mode::VllmFp8 => (Some(weight.input_scale), Some(weight.weight_scale)),
        };
        let (padded_out, padded_in) = match residency {
            Residency::Cuda => (rows.next_multiple_of(64), cols.next_multiple_of(128)),
            _ => (rows, cols),
        };
        let weight_kind = match residency {
            Residency::Cpu => {
                if !cols.is_multiple_of(32) {
                    return Err(load_error(format!(
                        "Q8_0 needs whole 32-column blocks; [{rows}, {cols}]"
                    )));
                }
                let bytes = q8_0_bytes(rows, cols, |row, out| {
                    out.copy_from_slice(&weight.dequantized_row(row));
                });
                Weight::Q8(q8_matmul(rows, cols, bytes)?)
            }
            Residency::Metal | Residency::Cuda => {
                let bytes = if (padded_out, padded_in) == (rows, cols) {
                    weight.bytes.to_vec()
                } else {
                    let mut padded = vec![0u8; padded_out * padded_in];
                    for row in 0..rows {
                        padded[row * padded_in..row * padded_in + cols]
                            .copy_from_slice(&weight.bytes[row * cols..(row + 1) * cols]);
                    }
                    padded
                };
                let scales =
                    vec![weight.weight_scale; padded_out.div_ceil(128) * padded_in.div_ceil(128)];
                Weight::Fp8 {
                    bytes: Tensor::from_vec(bytes, (padded_out, padded_in), device)?,
                    scales: Tensor::from_vec(
                        scales,
                        (padded_out.div_ceil(128), padded_in.div_ceil(128)),
                        device,
                    )?,
                }
            }
        };
        Ok(Self {
            weight: weight_kind,
            bias: None,
            out_features: rows,
            in_features: cols,
            padded_out,
            padded_in,
            input_scale,
            weight_scale,
        })
    }

    /// A GPU layer from block-FP8 `bytes` (`[rows, cols]`) and one F32
    /// scale per 128x128 block, already whole tiles of the FP8 kernels
    /// (`rows % 64 == 0`, `cols % 128 == 0`): a quantized head.
    pub fn fp8_from_blocks(
        bytes: Vec<u8>,
        scales: Vec<f32>,
        rows: usize,
        cols: usize,
        device: &Device,
    ) -> Result<Self> {
        if device.is_cpu()
            || !rows.is_multiple_of(64)
            || !cols.is_multiple_of(128)
            || bytes.len() != rows * cols
            || scales.len() != rows.div_ceil(128) * cols.div_ceil(128)
        {
            return Err(load_error(format!(
                "a block-FP8 GPU linear needs whole 64x128 tiles; [{rows}, {cols}]"
            )));
        }
        Ok(Self {
            weight: Weight::Fp8 {
                bytes: Tensor::from_vec(bytes, (rows, cols), device)?,
                scales: Tensor::from_vec(scales, (rows.div_ceil(128), cols.div_ceil(128)), device)?,
            },
            bias: None,
            out_features: rows,
            in_features: cols,
            padded_out: rows,
            padded_in: cols,
            input_scale: None,
            weight_scale: None,
        })
    }

    /// The BF16 linear `module` as a dense layer (F32 on CPU and Metal,
    /// BF16 on CUDA), with an optional BF16 `bias`.
    pub fn load_dense(
        checkpoint: &NemotronCheckpoint,
        module: &str,
        rows: usize,
        cols: usize,
        bias: bool,
        device: &Device,
    ) -> Result<Self> {
        let values = checkpoint.f32_values(&format!("{module}.weight"), &[rows, cols])?;
        Self::dense_from_values(
            values,
            rows,
            cols,
            device,
            Residency::of(device).dense_dtype(),
        )?
        .with_bias(checkpoint, module, bias, device)
    }

    /// A dense layer from host values in `dtype` on `device`.
    pub fn dense_from_values(
        values: Vec<f32>,
        rows: usize,
        cols: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let weight = Tensor::from_vec(values, (rows, cols), &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(device)?;
        Ok(Self {
            weight: Weight::Dense(weight),
            bias: None,
            out_features: rows,
            in_features: cols,
            padded_out: rows,
            padded_in: cols,
            input_scale: None,
            weight_scale: None,
        })
    }

    /// A Q8_0 layer from host values (the CPU language-model head).
    pub fn q8_from_values(values: &[f32], rows: usize, cols: usize) -> Result<Self> {
        if !cols.is_multiple_of(32) {
            return Err(load_error("Q8_0 needs whole 32-column blocks"));
        }
        let bytes = q8_0_bytes(rows, cols, |row, out| {
            out.copy_from_slice(&values[row * cols..(row + 1) * cols]);
        });
        Ok(Self {
            weight: Weight::Q8(q8_matmul(rows, cols, bytes)?),
            bias: None,
            out_features: rows,
            in_features: cols,
            padded_out: rows,
            padded_in: cols,
            input_scale: None,
            weight_scale: None,
        })
    }

    pub(crate) fn with_bias(
        mut self,
        checkpoint: &NemotronCheckpoint,
        module: &str,
        bias: bool,
        device: &Device,
    ) -> Result<Self> {
        if bias {
            let values = checkpoint.f32_values(&format!("{module}.bias"), &[self.out_features])?;
            self.bias = Some(Tensor::from_vec(values, self.out_features, device)?);
        }
        Ok(self)
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// The resident output width (CUDA FP8 pads it).
    pub fn padded_out(&self) -> usize {
        self.padded_out
    }

    pub fn padded_in(&self) -> usize {
        self.padded_in
    }

    pub fn is_fp8(&self) -> bool {
        matches!(self.weight, Weight::Fp8 { .. } | Weight::Q8(_)) && self.padded_in > 0
    }

    /// Resident bytes of the weight (and bias).
    pub fn resident_bytes(&self) -> usize {
        let weight = match &self.weight {
            Weight::Q8(_) => self.out_features * self.in_features / 32 * 34,
            Weight::Fp8 { .. } => {
                self.padded_out * self.padded_in
                    + 4 * self.padded_out.div_ceil(128) * self.padded_in.div_ceil(128)
            }
            Weight::Dense(t) => t.elem_count() * t.dtype().size_in_bytes(),
        };
        weight + self.bias.as_ref().map_or(0, |b| b.elem_count() * 4)
    }

    /// `x @ W.T + b` for F32 `x` `[m, in]`, trimmed to `[m, out]`.
    pub fn forward(&self, x: &Tensor, pass: Pass) -> Result<Tensor> {
        let y = self.forward_padded(x, pass)?;
        if self.padded_out == self.out_features {
            Ok(y)
        } else {
            Ok(y.narrow(1, 0, self.out_features)?)
        }
    }

    /// As [`Self::forward`], keeping the padded output width (zeros past
    /// `out`), and accepting `x` at the padded input width.
    pub fn forward_padded(&self, x: &Tensor, pass: Pass) -> Result<Tensor> {
        let (m, width) = x.dims2()?;
        if width != self.in_features && width != self.padded_in {
            return Err(Error::InferenceError(format!(
                "linear [{}, {}] got input width {width}",
                self.out_features, self.in_features
            )));
        }
        let x = if let Some(scale) = self.input_scale {
            // vllm-fp8 parity: static per-tensor E4M3 activation quantization.
            let mut values = x.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
            fake_quantize_e4m3(&mut values, scale);
            Tensor::from_vec(values, (m, width), x.device())?
        } else {
            x.clone()
        };
        let _ = self.weight_scale;
        let y = match &self.weight {
            Weight::Q8(q) => {
                let x = x.contiguous()?;
                match (pass, q) {
                    (Pass::Span, QMatMul::QTensor(qt)) if m >= CPU_TILE_MIN_ROWS => {
                        crate::kernels::cpu_q8::dequant_tile_matmul(&x, qt, CPU_TILE_ROWS)?
                    }
                    _ => q.forward(&x)?,
                }
            }
            Weight::Fp8 { bytes, scales } => {
                let x = if width < self.padded_in {
                    x.pad_with_zeros(1, 0, self.padded_in - width)?
                } else {
                    x
                };
                match bytes.device() {
                    Device::Metal(_) => match pass {
                        Pass::Step => crate::kernels::cuda::fp8::block_fp8_projection(
                            &x.contiguous()?,
                            bytes,
                            scales,
                        )?,
                        Pass::Span
                            if metal_fp8_gemm::supported(
                                bytes.device(),
                                self.padded_out,
                                self.padded_in,
                            ) =>
                        {
                            metal_fp8_gemm::block_gemm(&x.contiguous()?, bytes, scales)?
                        }
                        Pass::Span => {
                            let w = crate::kernels::metal_fp8::dequantize_f16(bytes, scales)?
                                .to_dtype(DType::F32)?;
                            x.contiguous()?.matmul(&w.t()?)?
                        }
                    },
                    _ => {
                        let x = x.contiguous()?;
                        if pass == Pass::Step && fp8::f32_projection_vectorized(&x, bytes) {
                            // F32 in and out: no casts around the GEMV.
                            fp8::block_fp8_projection_f32(&x, bytes, scales, Fp8Epilogue::None)?
                        } else {
                            let xb = x.to_dtype(DType::BF16)?;
                            let y = match pass {
                                Pass::Step => fp8::block_fp8_projection(&xb, bytes, scales)?,
                                Pass::Span => {
                                    fp8::block_fp8_projection_dequant_gemm(&xb, bytes, scales)?
                                }
                            };
                            y.to_dtype(DType::F32)?
                        }
                    }
                }
            }
            Weight::Dense(w) => {
                let xw = x.to_dtype(w.dtype())?.contiguous()?;
                xw.matmul(&w.t()?)?.to_dtype(DType::F32)?
            }
        };
        match &self.bias {
            Some(bias) => {
                let bias = if self.padded_out > self.out_features {
                    bias.pad_with_zeros(0, 0, self.padded_out - self.out_features)?
                } else {
                    bias.clone()
                };
                Ok(y.broadcast_add(&bias)?)
            }
            None => Ok(y),
        }
    }

    /// `x @ W.T` in the weight's own dtype for a dense layer (no cast back to
    /// F32: a device greedy pick reads 16-bit logits as they are); the F32
    /// [`Self::forward`] of a decode step otherwise.
    pub fn forward_native(&self, x: &Tensor) -> Result<Tensor> {
        match (&self.weight, &self.bias) {
            (Weight::Dense(w), None) => Ok(x.to_dtype(w.dtype())?.contiguous()?.matmul(&w.t()?)?),
            _ => self.forward(x, Pass::Step),
        }
    }

    /// The dtype [`Self::forward_native`] returns.
    pub fn native_dtype(&self) -> DType {
        match (&self.weight, &self.bias) {
            (Weight::Dense(w), None) => w.dtype(),
            _ => DType::F32,
        }
    }

    /// [`Self::forward`] (or, with `relu2`, [`Self::forward_relu2`]) of
    /// several rows' spans packed into `x`, row `r`'s `lengths[r]` tokens in
    /// turn, each row through the kernel its span takes alone. Only the
    /// CPU's Q8_0 weights pick their kernel by row count (the tile kernel
    /// from `CPU_TILE_MIN_ROWS` rows, the row-wise one below): there the
    /// shorter spans run the row-wise kernel together and the others the tile
    /// kernel, so a row differs from its span alone only in the tile GEMM's
    /// rounding. Every other weight takes its span kernel in one call.
    pub fn forward_rows(&self, x: &Tensor, lengths: &[usize], relu2: bool) -> Result<Tensor> {
        let run = |x: &Tensor, pass: Pass| {
            if relu2 {
                self.forward_relu2(x, pass)
            } else {
                self.forward(x, pass)
            }
        };
        if !matches!(&self.weight, Weight::Q8(QMatMul::QTensor(_))) {
            return run(x, Pass::Span);
        }
        by_row_class(
            lengths,
            x.device(),
            |tokens| tokens >= CPU_TILE_MIN_ROWS,
            |ids, tiled| {
                // A step always takes the row-wise kernel.
                let pass = if tiled { Pass::Span } else { Pass::Step };
                match ids {
                    Some(ids) => run(&x.index_select(ids, 0)?, pass),
                    None => run(x, pass),
                }
            },
        )
    }

    /// `relu(forward_padded(x))^2`: a relu² MLP's up projection. A CUDA step
    /// applies the relu² in the GEMV's epilogue (one launch).
    pub fn forward_relu2(&self, x: &Tensor, pass: Pass) -> Result<Tensor> {
        if let (Weight::Fp8 { bytes, scales }, Pass::Step, None) =
            (&self.weight, pass, self.input_scale)
        {
            if bytes.device().is_cuda() && self.bias.is_none() {
                let (_, width) = x.dims2()?;
                let x = if width < self.padded_in {
                    x.pad_with_zeros(1, 0, self.padded_in - width)?
                } else {
                    x.clone()
                }
                .contiguous()?;
                if fp8::f32_projection_vectorized(&x, bytes) {
                    return Ok(fp8::block_fp8_projection_f32(
                        &x,
                        bytes,
                        scales,
                        Fp8Epilogue::ReluSquare,
                    )?);
                }
            }
        }
        Ok(self.forward_padded(x, pass)?.relu()?.sqr()?)
    }

    /// Compares rows `rows` of this layer against the exactly dequantized
    /// checkpoint weight on xorshift input: the self-check that the resident
    /// form (Q8_0, padding, the device kernel) computes the original
    /// weights. Returns the relative L2 error.
    pub fn check_against(&self, original: &Fp8Weight<'_>, device: &Device) -> Result<f64> {
        use crate::kernels::numeric_check::{host_f32, relative_l2, xorshift_values};
        let probe_rows = original.rows.min(64);
        let x = xorshift_values(2 * original.cols, 17, 1.0);
        let mut expected = vec![0f32; 2 * probe_rows];
        for row in 0..probe_rows {
            let w = original.dequantized_row(row);
            for m in 0..2 {
                expected[m * probe_rows + row] = x[m * original.cols..(m + 1) * original.cols]
                    .iter()
                    .zip(&w)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum::<f64>() as f32;
            }
        }
        let input = Tensor::from_vec(x, (2, original.cols), device)?;
        let y = self.forward(&input, Pass::Step)?.narrow(1, 0, probe_rows)?;
        Ok(relative_l2(&host_f32(&y)?, &expected))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::cuda::fp8::encode_e4m3fn;
    use crate::kernels::numeric_check::{host_f32, relative_l2, xorshift_values};

    fn fp8_bytes(rows: usize, cols: usize, scale: f32, seed: u64) -> Vec<u8> {
        xorshift_values(rows * cols, seed, 0.05)
            .into_iter()
            .map(|v| encode_e4m3fn(v / scale))
            .collect()
    }

    #[test]
    fn q8_resident_fp8_tracks_the_exact_dequantization() {
        let (rows, cols) = (96, 160);
        let bytes = fp8_bytes(rows, cols, 1e-3, 1);
        let weight = Fp8Weight {
            bytes: &bytes,
            rows,
            cols,
            weight_scale: 1e-3,
            input_scale: 0.02,
        };
        let layer = NtLinear::from_fp8(weight, rows, cols, &Device::Cpu, Fp8Mode::W8A16).unwrap();
        let error = layer.check_against(&weight, &Device::Cpu).unwrap();
        // Q8_0 re-quantization of E4M3 values: a small, bounded deviation.
        assert!(error < 1e-2, "{error}");
        // Span (tiled) and step (row-wise) agree closely.
        let x =
            Tensor::from_vec(xorshift_values(8 * cols, 4, 1.0), (8, cols), &Device::Cpu).unwrap();
        let span = host_f32(&layer.forward(&x, Pass::Span).unwrap()).unwrap();
        let step = host_f32(&layer.forward(&x, Pass::Step).unwrap()).unwrap();
        assert!(relative_l2(&span, &step) < 1e-2);
    }

    /// Packed spans of short (row-wise Q8_0) and long (tile) rows on the
    /// CPU: every row's output is its span's alone, the short rows bit for
    /// bit, the long ones within the tile GEMM's rounding at another row
    /// count, and each lands back at its own tokens.
    #[test]
    fn packed_rows_take_their_spans_own_q8_kernels() {
        let (rows, cols) = (48, 64);
        let layer =
            NtLinear::q8_from_values(&xorshift_values(rows * cols, 3, 0.5), rows, cols).unwrap();
        let lengths = [1usize, 7, 4, 5, 2];
        let total = lengths.iter().sum::<usize>();
        let x = Tensor::from_vec(
            xorshift_values(total * cols, 5, 1.0),
            (total, cols),
            &Device::Cpu,
        )
        .unwrap();
        for relu2 in [false, true] {
            let packed = layer.forward_rows(&x, &lengths, relu2).unwrap();
            let mut offset = 0;
            for &tokens in &lengths {
                let span = x.narrow(0, offset, tokens).unwrap();
                let alone = if relu2 {
                    layer.forward_relu2(&span, Pass::Span)
                } else {
                    layer.forward(&span, Pass::Span)
                }
                .unwrap();
                let alone = host_f32(&alone).unwrap();
                let got = host_f32(&packed.narrow(0, offset, tokens).unwrap()).unwrap();
                if tokens < CPU_TILE_MIN_ROWS {
                    assert_eq!(got, alone, "{tokens}-token row");
                } else {
                    let error = relative_l2(&got, &alone);
                    assert!(error < 1e-6, "{tokens}-token row: {error}");
                }
                offset += tokens;
            }
        }
        // The kernels differ: a short row through the tile kernel is not its
        // span alone.
        let short = x.narrow(0, 0, 1).unwrap();
        let tiled = host_f32(
            &layer
                .forward(&Tensor::cat(&[&short; 5], 0).unwrap(), Pass::Span)
                .unwrap()
                .narrow(0, 0, 1)
                .unwrap(),
        )
        .unwrap();
        assert_ne!(
            tiled,
            host_f32(&layer.forward(&short, Pass::Span).unwrap()).unwrap()
        );
    }

    #[test]
    fn e4m3_fake_quantization_saturates_and_rounds_to_the_grid() {
        let mut values = vec![1000.0, -1000.0, 0.0, 1.0, 1.0625, 1.1875];
        fake_quantize_e4m3(&mut values, 1.0);
        assert_eq!(values, vec![448.0, -448.0, 0.0, 1.0, 1.0, 1.25]);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_fp8_matches_the_exact_dequantization_on_both_passes() {
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "metal_fp8_matches_the_exact_dequantization_on_both_passes",
        ) else {
            return;
        };
        let (rows, cols) = (96, 1856);
        let bytes = fp8_bytes(rows, cols, 2e-3, 2);
        let weight = Fp8Weight {
            bytes: &bytes,
            rows,
            cols,
            weight_scale: 2e-3,
            input_scale: 0.02,
        };
        let layer = NtLinear::from_fp8(weight, rows, cols, &device, Fp8Mode::W8A16).unwrap();
        assert!(layer.check_against(&weight, &device).unwrap() < 1e-5);
        let x = Tensor::from_vec(xorshift_values(12 * cols, 4, 1.0), (12, cols), &device).unwrap();
        let span = host_f32(&layer.forward(&x, Pass::Span).unwrap()).unwrap();
        let step = host_f32(&layer.forward(&x, Pass::Step).unwrap()).unwrap();
        // The span path rounds weights to F16 once.
        assert!(relative_l2(&span, &step) < 2e-3);
    }
}
