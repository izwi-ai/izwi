//! Qwen3.5 projections.
//!
//! Each projection keeps Candle's `QMatMul` and, on CUDA, an izwi-owned view
//! of the same GGUF weight for the K-quant MMVQ kernels
//! (`kernels/cuda/kquant.rs`). The view is used only after a self-check shows
//! it reproduces `QMatMul` exactly. It makes the projection capturable in a
//! CUDA graph, and it lets projections that read one input share a single
//! Q8_1 quantization ([`forward_shared`]).

use candle_core::quantized::QMatMul;
use candle_core::{Module, Tensor};

use crate::error::{Error, Result};
use crate::kernels::cuda::kquant::{self, KQuantWeight};

use super::fast_path::{legacy_requested, xorshift_values, Qwen35FastPath};

/// Environment switch that keeps every projection on `QMatMul`.
pub(crate) const KQUANT_ENV: &str = "IZWI_QWEN35_KQUANT";

pub(crate) struct Qwen35Linear {
    matmul: QMatMul,
    kquant: Option<KQuantWeight>,
    kquant_path: Qwen35FastPath,
    /// On Metal, a Q4_K / Q6_K mat-vec that reads the weight once for 2..=8
    /// decode rows; Candle's `QMatMul` streams it once per row.
    #[cfg(feature = "metal")]
    small_batch: Option<crate::kernels::metal_kquant_mv::MetalKQuantMatVec>,
}

impl Qwen35Linear {
    pub(crate) fn new(matmul: QMatMul) -> Self {
        Self {
            matmul,
            kquant: None,
            kquant_path: Qwen35FastPath::legacy("unresolved"),
            #[cfg(feature = "metal")]
            small_batch: None,
        }
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        if let Some(small_batch) = &self.small_batch {
            if small_batch.accepts(x) {
                return Ok(small_batch.forward(x)?);
            }
        }
        match &self.kquant {
            Some(weight) if kquant::accepts(x) => {
                Ok(kquant::mat_vec(weight, &kquant::quantize(x)?)?)
            }
            _ => Ok(self.matmul.forward(x)?),
        }
    }

    pub(crate) fn kquant_path(&self) -> &Qwen35FastPath {
        &self.kquant_path
    }

    pub(crate) fn matmul(&self) -> &QMatMul {
        &self.matmul
    }

    /// Take `matmul` (the same weight in a buffer `kernel` shares) and the
    /// kernel, once it passes the self-check; a failed check keeps `matmul`
    /// on Candle.
    #[cfg(feature = "metal")]
    pub(crate) fn adopt_metal_small_batch(
        &mut self,
        matmul: QMatMul,
        kernel: crate::kernels::metal_kquant_mv::MetalKQuantMatVec,
    ) {
        self.matmul = matmul;
        self.small_batch = None;
        match self.small_batch_self_check(&kernel) {
            Ok(()) => {
                self.small_batch = Some(kernel);
                self.kquant_path = Qwen35FastPath::Fused;
            }
            Err(error) => {
                tracing::warn!(%error, "Qwen3.5 Metal small-batch self-check failed; keeping QMatMul");
                self.kquant_path = Qwen35FastPath::legacy(format!("self-check failed: {error}"));
            }
        }
    }

    /// Resolve the Metal small-batch mat-vec for this weight: a Q4_K, Q5_K,
    /// Q6_K or Q8_0 Metal weight whose output for several decode rows
    /// matches `QMatMul` to float rounding (the sums only regroup). One row
    /// stays on Candle.
    #[cfg(feature = "metal")]
    pub(crate) fn resolve_metal_small_batch(&mut self) {
        self.small_batch = None;
        if legacy_requested(KQUANT_ENV) {
            return;
        }
        let kernel = match crate::kernels::metal_kquant_mv::share_qmatmul_weight(&mut self.matmul) {
            Ok(Some(kernel)) => kernel,
            Ok(None) => return,
            Err(error) => {
                tracing::warn!(%error, "Qwen3.5 Metal small-batch weight upload failed");
                return;
            }
        };
        match self.small_batch_self_check(&kernel) {
            Ok(()) => {
                self.small_batch = Some(kernel);
                self.kquant_path = Qwen35FastPath::Fused;
            }
            Err(error) => {
                tracing::warn!(%error, "Qwen3.5 Metal small-batch self-check failed; keeping QMatMul");
                self.kquant_path = Qwen35FastPath::legacy(format!("self-check failed: {error}"));
            }
        }
    }

    #[cfg(feature = "metal")]
    fn small_batch_self_check(
        &self,
        kernel: &crate::kernels::metal_kquant_mv::MetalKQuantMatVec,
    ) -> Result<()> {
        let device = match &self.matmul {
            QMatMul::QTensor(qtensor) => qtensor.device(),
            _ => return Err(Error::InferenceError("not a quantized weight".into())),
        };
        let cols = match &self.matmul {
            QMatMul::QTensor(qtensor) => qtensor.shape().dims2()?.1,
            _ => unreachable!("checked above"),
        };
        for (rows, seed) in [(2usize, 41u64), (5, 42)] {
            let x = Tensor::from_vec(
                xorshift_values(rows * cols, seed, 1.0),
                (rows, 1, cols),
                &device,
            )?;
            let expected = self.matmul.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
            let actual = kernel.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
            let scale = expected.iter().fold(0f32, |max, e| max.max(e.abs()));
            if actual.len() != expected.len()
                || !scale.is_finite()
                || actual
                    .iter()
                    .zip(&expected)
                    .any(|(a, e)| !a.is_finite() || (a - e).abs() > 1e-5 * scale.max(1e-6))
            {
                return Err(Error::InferenceError(format!(
                    "small-batch mat-vec at {rows} rows differs from QMatMul"
                )));
            }
        }
        Ok(())
    }

    /// Resolve the K-quant MMVQ path for this weight. It needs a CUDA Q4_K,
    /// Q5_K, Q6_K or Q8_0 tensor and a self-check that matches `QMatMul` bit
    /// for bit at one and several input rows.
    pub(crate) fn resolve_kquant(&mut self, switch_off: Option<&'static str>) {
        self.kquant = None;
        self.kquant_path = if legacy_requested(KQUANT_ENV) {
            Qwen35FastPath::legacy(format!("{KQUANT_ENV}=legacy"))
        } else if let Some(reason) = switch_off {
            Qwen35FastPath::legacy(reason)
        } else {
            match KQuantWeight::from_qmatmul(&self.matmul) {
                None => Qwen35FastPath::legacy(
                    "K-quant MMVQ needs a CUDA Q4_K, Q5_K, Q6_K or Q8_0 weight",
                ),
                Some(weight) => match self.kquant_self_check(&weight) {
                    Ok(()) => {
                        self.kquant = Some(weight);
                        Qwen35FastPath::Fused
                    }
                    Err(error) => {
                        tracing::warn!(
                            %error,
                            "Qwen3.5 K-quant MMVQ self-check failed; keeping QMatMul"
                        );
                        Qwen35FastPath::legacy(format!("self-check failed: {error}"))
                    }
                },
            }
        };
    }

    fn kquant_self_check(&self, weight: &KQuantWeight) -> Result<()> {
        let device = weight.device();
        for (rows, seed) in [(1usize, 31u64), (3, 32)] {
            let x = Tensor::from_vec(
                xorshift_values(rows * weight.cols(), seed, 1.0),
                (rows, weight.cols()),
                &device,
            )?;
            let expected = self.matmul.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
            let actual = kquant::mat_vec(weight, &kquant::quantize(&x)?)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            if actual.len() != expected.len()
                || actual
                    .iter()
                    .zip(&expected)
                    .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
            {
                return Err(Error::InferenceError(format!(
                    "{:?} [{}, {}] at {rows} rows differs from QMatMul",
                    weight.dtype(),
                    weight.rows(),
                    weight.cols()
                )));
            }
        }
        Ok(())
    }
}

/// Projections that read the same input. On the K-quant path the input is
/// quantized to Q8_1 once and each weight runs one MMVQ launch; otherwise
/// each projection runs on its own.
pub(crate) fn forward_shared<const N: usize>(
    x: &Tensor,
    linears: [&Qwen35Linear; N],
) -> Result<[Tensor; N]> {
    let mut outputs = Vec::with_capacity(N);
    if kquant::accepts(x) && linears.iter().all(|linear| linear.kquant.is_some()) {
        let activations = kquant::quantize(x)?;
        for linear in linears {
            let weight = linear.kquant.as_ref().expect("checked above");
            outputs.push(kquant::mat_vec(weight, &activations)?);
        }
    } else {
        for linear in linears {
            outputs.push(linear.forward(x)?);
        }
    }
    outputs
        .try_into()
        .map_err(|_| Error::InferenceError("shared projection output count".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn shared_projections_fall_back_to_each_matmul_off_cuda() {
        let device = Device::Cpu;
        let linear = |seed: u64, rows: usize| {
            Qwen35Linear::new(QMatMul::Tensor(
                Tensor::from_vec(xorshift_values(rows * 8, seed, 1.0), (rows, 8), &device).unwrap(),
            ))
        };
        let mut first = linear(1, 4);
        let second = linear(2, 3);
        first.resolve_kquant(None);
        assert!(matches!(first.kquant_path(), Qwen35FastPath::Legacy { .. }));
        let x = Tensor::from_vec(xorshift_values(16, 3, 1.0), (2, 8), &device).unwrap();
        let [a, b] = forward_shared(&x, [&first, &second]).unwrap();
        assert_eq!(a.dims(), &[2, 4]);
        assert_eq!(b.dims(), &[2, 3]);
        let expected = first.matmul.forward(&x).unwrap();
        assert_eq!(
            a.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn kquant_resolution_honours_its_switches() {
        let _guard = crate::env_test_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut linear = Qwen35Linear::new(QMatMul::Tensor(
            Tensor::zeros((2, 8), DType::F32, &Device::Cpu).unwrap(),
        ));
        std::env::set_var(KQUANT_ENV, "legacy");
        linear.resolve_kquant(None);
        std::env::remove_var(KQUANT_ENV);
        assert_eq!(
            linear.kquant_path(),
            &Qwen35FastPath::legacy("IZWI_QWEN35_KQUANT=legacy")
        );
        linear.resolve_kquant(Some("cuda.packed_projections=off"));
        assert_eq!(
            linear.kquant_path(),
            &Qwen35FastPath::legacy("cuda.packed_projections=off")
        );
    }
}
