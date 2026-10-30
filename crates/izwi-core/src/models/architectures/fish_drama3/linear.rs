//! Linear layers of the family in their resident form.
//!
//! | Weight | CPU (D4) | Metal (Phase 3) | CUDA (Phase 3) |
//! |---|---|---|---|
//! | Routed experts (E4M3 × F32 block scales) | Q8_0 of the exact F32 dequantization | raw E4M3 + scales | raw E4M3 + scales |
//! | Dense trunk projections, shared expert (BF16) | Q8_0 | F16 | BF16 |
//! | A projection class whose Q8_0 self-check fails | F32 | — | — |
//!
//! Only the CPU forms exist so far; [`FishDrama3WeightForm`] already names
//! the device ones so their loads slot in without another API.
//!
//! Every layer takes and returns F32 activations `[m, in]` → `[m, out]`. The
//! caller names the [`FishDrama3KernelPhase`]. On the CPU a Q8_0 step runs
//! Candle's row-wise Q8_0 kernel (its activations are quantized to Q8_0
//! too), so a row's arithmetic never depends on how many rows share its
//! step. A span (one request's prefill) always runs the tile kernel: weights
//! expanded to F32 a tile at a time, F32 activations, multiplied by Candle's
//! F32 matmul, even for a span of one row. That matmul takes its
//! matrix-vector path for a single row (a one-row span, or an expert group
//! of one row) and blocks the rest by shape, so a span's bits depend on how
//! its rows split: split invariance is not promised. The model loop pins
//! the prefill grid instead (`model/mod.rs`).

use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{DType, Device, Module, Tensor};
use half::f16;

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
            other => {
                return Err(load_error(format!(
                    "the {other:?} form has no implementation yet (Phase 3)"
                )))
            }
        };
        Ok(Self { weight, rows, cols })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn form(&self) -> FishDrama3WeightForm {
        match self.weight {
            Weight::Q8_0(_) => FishDrama3WeightForm::Q8_0,
            Weight::F32(_) => FishDrama3WeightForm::F32,
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
        })
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
            Weight::F32(w) => w.device().clone(),
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
    fn device_forms_are_refused_until_phase_3() {
        for form in [
            FishDrama3WeightForm::F16,
            FishDrama3WeightForm::Bf16,
            FishDrama3WeightForm::RawFp8,
        ] {
            assert!(
                FishDrama3Linear::from_values(vec![0.0; 64], 2, 32, form, &Device::Cpu).is_err()
            );
        }
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
