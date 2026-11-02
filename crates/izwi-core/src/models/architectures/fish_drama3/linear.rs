//! Linear layers of the family in their resident form.
//!
//! | Weight | CPU (D4) | Metal | CUDA |
//! |---|---|---|---|
//! | Routed experts (E4M3 × F32 block scales) | Q8_0 of the exact F32 dequantization | raw E4M3 + scales (`device::trunk`) | raw E4M3 + scales (`device::trunk`) |
//! | Dense trunk projections, shared expert (BF16) | Q8_0 | F16 | BF16 |
//! | A projection class whose Q8_0 self-check fails | F32 | — | — |
//! | Router, DeltaNet gates | F32 | F32 | F32 |
//!
//! Every layer takes and returns F32 activations `[m, in]` → `[m, out]`
//! through [`FishDrama3Linear::forward`]; a device layer also runs
//! [`FishDrama3Linear::forward_device`], whose activations stay on the
//! device in the layer's activation dtype (F16 on Metal or BF16 on CUDA,
//! or F32 for a Metal layer the load-time range probe moved). The caller
//! names the [`FishDrama3KernelPhase`].
//!
//! On the CPU a Q8_0 step runs
//! Candle's row-wise Q8_0 kernel (its activations are quantized to Q8_0
//! too), so a row's arithmetic never depends on how many rows share its
//! step. A span (one request's prefill) always runs the tile kernel: weights
//! expanded to F32 a tile at a time, F32 activations, multiplied by Candle's
//! F32 matmul, even for a span of one row. That matmul takes its
//! matrix-vector path for a single row (a one-row span, or an expert group
//! of one row) and blocks the rest by shape, so a span's bits depend on how
//! its rows split: split invariance is not promised. The model loop pins
//! the prefill grid instead (`model/mod.rs`).
//!
//! On Metal a step runs one GEMV per row (§7.4: at widths 1–2 the weights
//! are read once per row either way), each over a contiguous row, so no row
//! shares a reduction with another; a span runs one tile GEMM over the
//! span's rows. On CUDA a step of several rows runs the row-invariant
//! multi-row GEMV (`kernels::cuda::row_gemv`: one fixed reduction order per
//! row, the weights read once for all of them) once its device check has
//! enabled it, else one GEMV per row as on Metal; a span runs one cuBLAS
//! GEMM.

use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{DType, Device, Module, Tensor};
use half::{bf16, f16};

use super::residency::FishDrama3WeightForm;
use crate::error::{Error, Result};

/// Q8_0 holds 32 values per block.
pub const Q8_0_BLOCK: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 34;
/// A span expands Q8_0 weights in tiles of this many rows.
const CPU_TILE_ROWS: usize = 256;

/// Which kernels a forward runs. The caller chooses it (decode or prefill),
/// never the row count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FishDrama3KernelPhase {
    /// Decode: any number of single-row requests, row-wise kernels only.
    Step,
    /// Prefill: rows of one request, tile kernels allowed.
    Span,
}

enum Weight {
    Q8_0(QMatMul),
    F32(Tensor),
    /// `[rows, cols]` 16-bit on the layer's device: F16 on Metal, BF16 on
    /// CUDA.
    Half(Tensor),
}

#[cfg(test)]
thread_local! {
    /// The RED hook of the batch-invariance tests: a step of several rows
    /// splits its reduction in two, as a batch-sized split-K GEMM would.
    static SPLIT_K_STEPS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f` with every multi-row device step of this thread split along its
/// reduction (an m-dependent kernel), for RED checks of row invariance.
#[cfg(test)]
pub(crate) fn with_split_k_steps<R>(f: impl FnOnce() -> R) -> R {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            SPLIT_K_STEPS.with(|flag| flag.set(false));
        }
    }
    SPLIT_K_STEPS.with(|flag| flag.set(true));
    let _reset = Reset;
    f()
}

/// Whether this thread's steps run the RED split (always false outside
/// tests).
fn split_k_steps() -> bool {
    #[cfg(test)]
    {
        SPLIT_K_STEPS.with(|flag| flag.get())
    }
    #[cfg(not(test))]
    {
        false
    }
}

/// A CUDA step's projection through the row-invariant multi-row GEMV, once
/// its device check has enabled it: `x` rounded to the weights' 16-bit dtype
/// (16-bit activations as they are, for F32 weights), the output in `act`.
/// `None` when the kernel is off or does not take the operands; the caller
/// then runs one GEMV per row.
fn row_invariant_step(x: &Tensor, w: &Tensor, act: DType) -> Result<Option<Tensor>> {
    use crate::kernels::cuda::row_gemv::{self, RowGemvWeight};
    if !row_gemv::enabled() {
        return Ok(None);
    }
    let input = match w.dtype() {
        DType::F16 | DType::BF16 => w.dtype(),
        _ if matches!(x.dtype(), DType::F16 | DType::BF16) => x.dtype(),
        _ => return Ok(None),
    };
    if !row_gemv::supported(input, RowGemvWeight::Dense(w), act) {
        return Ok(None);
    }
    let y = row_gemv::row_gemv(&x.to_dtype(input)?, RowGemvWeight::Dense(w), act)?;
    super::device::count_step_row_gemv();
    Ok(Some(y))
}

/// A loaded projection `[rows, cols]` (no bias: the trunk has none).
pub struct FishDrama3Linear {
    weight: Weight,
    rows: usize,
    cols: usize,
}

impl std::fmt::Debug for FishDrama3Linear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FishDrama3Linear")
            .field("form", &self.form())
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .finish()
    }
}

fn load_error(message: impl Into<String>) -> Error {
    Error::ModelLoadError(format!("Fish Audio Drama 3 linear: {}", message.into()))
}

/// Q8_0 blocks (an F16 scale, then 32 signed bytes) of the row-major
/// `[rows, cols]` matrix whose row `r` `value` writes, quantized exactly as
/// ggml's reference quantizer does (`d = amax / 127`, `round(x / d)`).
pub(crate) fn q8_0_bytes(
    rows: usize,
    cols: usize,
    value: impl Fn(usize, &mut [f32]) + Sync,
) -> Vec<u8> {
    use rayon::prelude::*;
    let row_bytes = cols / Q8_0_BLOCK * Q8_0_BLOCK_BYTES;
    let mut out = vec![0u8; rows * row_bytes];
    out.par_chunks_mut(row_bytes)
        .enumerate()
        .for_each(|(row, bytes)| {
            let mut values = vec![0f32; cols];
            value(row, &mut values);
            for (block, chunk) in values.as_chunks::<Q8_0_BLOCK>().0.iter().enumerate() {
                let amax = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
                let d = amax / 127.0;
                let inverse = if d != 0.0 { 1.0 / d } else { 0.0 };
                let dst = &mut bytes[block * Q8_0_BLOCK_BYTES..(block + 1) * Q8_0_BLOCK_BYTES];
                dst[..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
                for (slot, v) in dst[2..].iter_mut().zip(chunk) {
                    *slot = (v * inverse).round() as i8 as u8;
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

impl FishDrama3Linear {
    /// The layer whose row-major `[rows, cols]` weights are `values` (exact
    /// F32: a BF16 checkpoint tensor or an FP8 expert's dequantization), in
    /// `form` on `device`.
    pub fn from_values(
        values: Vec<f32>,
        rows: usize,
        cols: usize,
        form: FishDrama3WeightForm,
        device: &Device,
    ) -> Result<Self> {
        if values.len() != rows * cols || rows == 0 || cols == 0 {
            return Err(load_error(format!(
                "{} values do not fill [{rows}, {cols}]",
                values.len()
            )));
        }
        let weight = match form {
            FishDrama3WeightForm::Q8_0 => {
                if !device.is_cpu() {
                    return Err(load_error("Q8_0 projections are a CPU form"));
                }
                if !cols.is_multiple_of(Q8_0_BLOCK) {
                    return Err(load_error(format!(
                        "Q8_0 needs whole {Q8_0_BLOCK}-column blocks; [{rows}, {cols}]"
                    )));
                }
                let bytes = q8_0_bytes(rows, cols, |row, out| {
                    out.copy_from_slice(&values[row * cols..(row + 1) * cols]);
                });
                Weight::Q8_0(q8_matmul(rows, cols, bytes)?)
            }
            FishDrama3WeightForm::F32 => {
                Weight::F32(Tensor::from_vec(values, (rows, cols), device)?)
            }
            FishDrama3WeightForm::F16 => Weight::Half(Tensor::from_vec(
                values.into_iter().map(f16::from_f32).collect::<Vec<_>>(),
                (rows, cols),
                device,
            )?),
            FishDrama3WeightForm::Bf16 => Weight::Half(Tensor::from_vec(
                values.into_iter().map(bf16::from_f32).collect::<Vec<_>>(),
                (rows, cols),
                device,
            )?),
            other => {
                return Err(load_error(format!(
                    "the {other:?} form is not a dense projection's (experts stack per layer)"
                )))
            }
        };
        Ok(Self { weight, rows, cols })
    }

    /// A 16-bit layer over `weight`, `[rows, cols]` F16 or BF16 already on
    /// its device (a BF16 checkpoint tensor, cast to F16 on the host for
    /// Metal, and uploaded once).
    pub fn from_half(weight: Tensor) -> Result<Self> {
        let (rows, cols) = weight.dims2()?;
        if !matches!(weight.dtype(), DType::F16 | DType::BF16) || rows == 0 || cols == 0 {
            return Err(load_error(format!(
                "a 16-bit layer takes a non-empty F16 or BF16 matrix, not {:?} {:?}",
                weight.dtype(),
                weight.dims()
            )));
        }
        Ok(Self {
            weight: Weight::Half(weight),
            rows,
            cols,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn form(&self) -> FishDrama3WeightForm {
        match &self.weight {
            Weight::Q8_0(_) => FishDrama3WeightForm::Q8_0,
            Weight::F32(_) => FishDrama3WeightForm::F32,
            Weight::Half(w) if w.dtype() == DType::BF16 => FishDrama3WeightForm::Bf16,
            Weight::Half(_) => FishDrama3WeightForm::F16,
        }
    }

    /// Resident bytes of the weight.
    pub fn resident_bytes(&self) -> usize {
        self.form().bytes(self.rows, self.cols)
    }

    /// `x @ W.T` for F32 `x` `[m, cols]`: `[m, rows]` F32.
    pub fn forward(&self, x: &Tensor, phase: FishDrama3KernelPhase) -> Result<Tensor> {
        let (_, width) = x.dims2()?;
        if width != self.cols || x.dtype() != DType::F32 {
            return Err(Error::InferenceError(format!(
                "linear [{}, {}] got a {:?} input of width {width}",
                self.rows,
                self.cols,
                x.dtype()
            )));
        }
        let x = x.contiguous()?;
        Ok(match &self.weight {
            Weight::Q8_0(q) => match (phase, q) {
                (FishDrama3KernelPhase::Span, QMatMul::QTensor(qt)) => {
                    crate::kernels::cpu_q8::dequant_tile_matmul(&x, qt, CPU_TILE_ROWS)?
                }
                _ => q.forward(&x)?,
            },
            Weight::F32(w) => x.matmul(&w.t()?)?,
            // F32 activations over the widened weights.
            Weight::Half(w) => x.matmul(&w.to_dtype(DType::F32)?.t()?)?,
        })
    }

    /// `x @ W.T` on the weight's device for `x` `[m, cols]` (any float
    /// dtype), computed and returned in `act`: the 16-bit dtype for a
    /// healthy layer, F32 for an F32-gated one (the router and gates) or
    /// one Metal's range probe moved (its 16-bit weights are then widened
    /// per call). A step runs one GEMV per contiguous row, or on CUDA the
    /// row-invariant multi-row GEMV; a span one GEMM.
    pub fn forward_device(
        &self,
        x: &Tensor,
        phase: FishDrama3KernelPhase,
        act: DType,
    ) -> Result<Tensor> {
        let (m, width) = x.dims2()?;
        if width != self.cols || m == 0 {
            return Err(Error::InferenceError(format!(
                "linear [{}, {}] got {:?} rows",
                self.rows,
                self.cols,
                x.dims()
            )));
        }
        let w = match &self.weight {
            Weight::Half(w) | Weight::F32(w) => w,
            Weight::Q8_0(_) => {
                return Err(Error::InferenceError(
                    "a Q8_0 projection has no device forward".into(),
                ))
            }
        };
        if phase == FishDrama3KernelPhase::Step && w.device().is_cuda() && !split_k_steps() {
            if let Some(y) = row_invariant_step(x, w, act)? {
                return Ok(y);
            }
        }
        let w = if w.dtype() == act {
            w.clone()
        } else {
            w.to_dtype(act)?
        };
        let wt = w.t()?;
        let x = x.to_dtype(act)?;
        if phase == FishDrama3KernelPhase::Span || m == 1 {
            return Ok(x.contiguous()?.matmul(&wt)?);
        }
        if split_k_steps() {
            let half = self.cols / 2;
            let part = |start: usize, len: usize| -> Result<Tensor> {
                Ok(x.narrow(1, start, len)?
                    .contiguous()?
                    .matmul(&w.narrow(1, start, len)?.contiguous()?.t()?)?)
            };
            return Ok((part(0, half)? + part(half, self.cols - half)?)?);
        }
        if x.device().is_cuda() {
            super::device::count_step_row_loop();
        }
        let rows = (0..m)
            .map(|row| x.narrow(0, row, 1)?.contiguous()?.matmul(&wt))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Tensor::cat(&rows, 0)?)
    }

    /// `x @ W.T` with `x` rounded to the weights' 16-bit dtype and the
    /// products summed into F32 outputs: a 16-bit projection into F32
    /// logits (CUDA's codebook heads, whose BF16 outputs would keep two
    /// fewer decimal digits than sampling reads). A step runs the
    /// row-invariant GEMV once enabled, else one F32 GEMV per row over the
    /// widened weights; a span one F32 GEMM.
    pub fn forward_device_f32(&self, x: &Tensor, phase: FishDrama3KernelPhase) -> Result<Tensor> {
        let (m, width) = x.dims2()?;
        let w = match &self.weight {
            Weight::Half(w) if width == self.cols && m > 0 => w,
            _ => {
                return Err(Error::InferenceError(format!(
                    "linear [{}, {}] has no 16-bit device forward into F32 for {:?} rows",
                    self.rows,
                    self.cols,
                    x.dims()
                )))
            }
        };
        if phase == FishDrama3KernelPhase::Step && w.device().is_cuda() && !split_k_steps() {
            if let Some(y) = row_invariant_step(x, w, DType::F32)? {
                return Ok(y);
            }
        }
        let x = x.to_dtype(w.dtype())?.to_dtype(DType::F32)?;
        let wt = w.to_dtype(DType::F32)?.t()?;
        if phase == FishDrama3KernelPhase::Span || m == 1 {
            return Ok(x.contiguous()?.matmul(&wt)?);
        }
        if split_k_steps() {
            let half = self.cols / 2;
            let part = |start: usize, len: usize| -> Result<Tensor> {
                Ok(x.narrow(1, start, len)?
                    .contiguous()?
                    .matmul(&wt.narrow(0, start, len)?)?)
            };
            return Ok((part(0, half)? + part(half, self.cols - half)?)?);
        }
        if x.device().is_cuda() {
            super::device::count_step_row_loop();
        }
        let rows = (0..m)
            .map(|row| x.narrow(0, row, 1)?.contiguous()?.matmul(&wt))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Tensor::cat(&rows, 0)?)
    }

    /// [`Self::forward`] over host rows: `x` holds `m` rows of `cols`.
    pub fn forward_host(
        &self,
        x: &[f32],
        m: usize,
        phase: FishDrama3KernelPhase,
    ) -> Result<Vec<f32>> {
        if x.len() != m * self.cols {
            return Err(Error::InferenceError(format!(
                "linear [{}, {}] got {} values for {m} rows",
                self.rows,
                self.cols,
                x.len()
            )));
        }
        let device = match &self.weight {
            Weight::F32(w) | Weight::Half(w) => w.device().clone(),
            Weight::Q8_0(_) => Device::Cpu,
        };
        let input = Tensor::from_slice(x, (m, self.cols), &device)?;
        Ok(self
            .forward(&input, phase)?
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::numeric_check::{relative_l2, xorshift_values};

    fn reference(values: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let m = x.len() / cols;
        let mut out = vec![0f32; m * rows];
        for i in 0..m {
            for r in 0..rows {
                out[i * rows + r] = (0..cols)
                    .map(|c| f64::from(x[i * cols + c]) * f64::from(values[r * cols + c]))
                    .sum::<f64>() as f32;
            }
        }
        out
    }

    #[test]
    fn the_q8_0_bytes_match_candles_reference_quantizer() {
        let (rows, cols) = (5, 96);
        let values = xorshift_values(rows * cols, 3, 0.7);
        let ours = q8_0_bytes(rows, cols, |row, out| {
            out.copy_from_slice(&values[row * cols..(row + 1) * cols])
        });
        let tensor = Tensor::from_vec(values, (rows, cols), &Device::Cpu).unwrap();
        let candle = QTensor::quantize(&tensor, GgmlDType::Q8_0).unwrap();
        assert_eq!(ours, candle.data().unwrap().to_vec());
    }

    #[test]
    fn both_forms_compute_the_projection_on_both_phases() {
        let (rows, cols, m) = (48, 128, 7);
        let values = xorshift_values(rows * cols, 5, 0.1);
        let x = xorshift_values(m * cols, 6, 1.0);
        let expected = reference(&values, &x, rows, cols);
        let f32_layer = FishDrama3Linear::from_values(
            values.clone(),
            rows,
            cols,
            FishDrama3WeightForm::F32,
            &Device::Cpu,
        )
        .unwrap();
        let q8 = FishDrama3Linear::from_values(
            values,
            rows,
            cols,
            FishDrama3WeightForm::Q8_0,
            &Device::Cpu,
        )
        .unwrap();
        assert_eq!(q8.resident_bytes(), rows * cols / 32 * 34);
        assert_eq!(f32_layer.resident_bytes(), rows * cols * 4);
        for phase in [FishDrama3KernelPhase::Step, FishDrama3KernelPhase::Span] {
            let y = f32_layer.forward_host(&x, m, phase).unwrap();
            assert!(relative_l2(&y, &expected) < 1e-6, "{phase:?}");
            // Q8_0 weights (and, on the row-wise path, Q8_0 activations).
            let y = q8.forward_host(&x, m, phase).unwrap();
            assert!(relative_l2(&y, &expected) < 1.5e-2, "{phase:?}");
        }
    }

    #[test]
    fn bf16_layers_load_and_raw_fp8_is_not_a_dense_form() {
        let values = xorshift_values(2 * 32, 9, 0.5);
        let layer = FishDrama3Linear::from_values(
            values.clone(),
            2,
            32,
            FishDrama3WeightForm::Bf16,
            &Device::Cpu,
        )
        .unwrap();
        assert_eq!(layer.form(), FishDrama3WeightForm::Bf16);
        assert_eq!(layer.resident_bytes(), 2 * 32 * 2);
        // The host forward over the widened BF16 weights.
        let x = xorshift_values(32, 10, 1.0);
        let rounded = values
            .iter()
            .map(|v| bf16::from_f32(*v).to_f32())
            .collect::<Vec<_>>();
        let y = layer
            .forward_host(&x, 1, FishDrama3KernelPhase::Step)
            .unwrap();
        assert!(relative_l2(&y, &reference(&rounded, &x, 2, 32)) < 1e-6);
        assert!(FishDrama3Linear::from_values(
            vec![0.0; 64],
            2,
            32,
            FishDrama3WeightForm::RawFp8,
            &Device::Cpu
        )
        .is_err());
        assert!(FishDrama3Linear::from_values(
            vec![0.0; 40],
            2,
            20,
            FishDrama3WeightForm::Q8_0,
            &Device::Cpu
        )
        .is_err());
    }
}
