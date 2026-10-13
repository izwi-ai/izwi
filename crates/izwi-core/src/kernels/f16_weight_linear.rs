//! Weight-only F16 linear layers: F16 weights, F32 activations, F32
//! accumulation and F32 output.
//!
//! This is whisper.cpp's recipe for F32 NeMo checkpoints (its Parakeet
//! converter stores only the 2-D linear and pointwise weights in F16 and keeps
//! activations in F32). On Metal the weight is an F16 ggml `QTensor` applied as
//! a Candle `CustomOp1`, which dispatches ggml's `kernel_mul_mm_f16_f32`:
//! weights staged as half, activations read as float, accumulation and output
//! in float. Candle 0.11 sends a single-row input to `kernel_mul_mv_f16_f32`
//! with zero weight strides (`nb01 = 0`), so every output would read weight
//! row 0; a single row is padded to two rows so it also takes `mul_mm`.
//! `QMatMul::from_arc` must not be used here: it dequantizes F16 QTensors to a
//! dense F32 tensor and keeps none of the memory saving.
//!
//! Elsewhere (CPU, CUDA, and Metal shapes the ggml kernel cannot take) the
//! weight is a dense F16 tensor cast to F32 per call, which gives the same
//! numbers. Candle's CPU `QTensor` op would quantize the activations to F16,
//! so it is not used.

use std::sync::Arc;

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor};

/// ggml's `mul_mm` reads the reduction dimension in blocks of 32 with no tail.
const GGML_MUL_MM_K_BLOCK: usize = 32;

/// How an [`F16WeightLinear`] multiplies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F16WeightLinearPath {
    /// ggml `mul_mm_f16_f32` on Metal.
    GgmlF16F32,
    /// A dense F16 weight cast to F32 per call.
    DenseF16Upcast,
}

impl F16WeightLinearPath {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::GgmlF16F32 => "ggml_f16_f32",
            Self::DenseF16Upcast => "dense_f16_upcast",
        }
    }
}

#[derive(Debug, Clone)]
enum F16Weight {
    Ggml(Arc<QTensor>),
    Dense(Tensor),
}

/// `y = x W^T + b` with `W` stored in F16, `x`, `b` and `y` in F32.
#[derive(Debug, Clone)]
pub struct F16WeightLinear {
    weight: F16Weight,
    bias: Option<Tensor>,
    out_features: usize,
    in_features: usize,
}

impl F16WeightLinear {
    /// Converts a `[out, in]` weight (any float dtype, any device) to F16 on
    /// the host and places it on `device`. `bias` stays F32 on `device`.
    pub fn new(weight: &Tensor, bias: Option<&Tensor>, device: &Device) -> Result<Self> {
        let (out_features, in_features) = weight.dims2()?;
        let host = weight.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let weight = if device.is_metal() && in_features % GGML_MUL_MM_K_BLOCK == 0 {
            F16Weight::Ggml(Arc::new(QTensor::quantize_onto(
                &host,
                GgmlDType::F16,
                device,
            )?))
        } else {
            F16Weight::Dense(host.to_dtype(DType::F16)?.to_device(device)?)
        };
        let bias = bias
            .map(|bias| bias.to_dtype(DType::F32)?.to_device(device))
            .transpose()?;
        Ok(Self {
            weight,
            bias,
            out_features,
            in_features,
        })
    }

    /// Converts a pointwise (kernel-size 1) `Conv1d` weight `[out, in, 1]`.
    pub fn from_pointwise_conv1d(
        weight: &Tensor,
        bias: Option<&Tensor>,
        device: &Device,
    ) -> Result<Self> {
        let (out_features, in_features, kernel) = weight.dims3()?;
        if kernel != 1 {
            candle_core::bail!("pointwise conv1d weight must have kernel size 1, got {kernel}");
        }
        Self::new(&weight.reshape((out_features, in_features))?, bias, device)
    }

    pub fn path(&self) -> F16WeightLinearPath {
        match self.weight {
            F16Weight::Ggml(_) => F16WeightLinearPath::GgmlF16F32,
            F16Weight::Dense(_) => F16WeightLinearPath::DenseF16Upcast,
        }
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Resident bytes of the F16 weight and the F32 bias.
    pub fn resident_bytes(&self) -> usize {
        self.out_features * self.in_features * DType::F16.size_in_bytes()
            + self
                .bias
                .as_ref()
                .map_or(0, |bias| bias.elem_count() * DType::F32.size_in_bytes())
    }

    /// The stored weight widened to F32, `[out, in]`.
    pub fn dequantized_weight(&self) -> Result<Tensor> {
        match &self.weight {
            F16Weight::Ggml(weight) => weight.dequantize(&weight.device()),
            F16Weight::Dense(weight) => weight.to_dtype(DType::F32),
        }
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl Module for F16WeightLinear {
    /// `x` is `[..., in]` in any float dtype; the output is F32 `[..., out]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let Some((&in_features, leading)) = dims.split_last() else {
            candle_core::bail!("F16WeightLinear input must have at least one dimension");
        };
        if in_features != self.in_features {
            candle_core::bail!(
                "F16WeightLinear input has {in_features} features, weight expects {}",
                self.in_features
            );
        }
        let rows = leading.iter().product::<usize>();
        let x = x
            .to_dtype(DType::F32)?
            .reshape((rows, in_features))?
            .contiguous()?;
        let y = match &self.weight {
            // The ggml op reads the weight as `[n, k]` and the input as
            // `[m, k]` row-major, writing F32 `[m, n]`.
            F16Weight::Ggml(weight) if rows == 1 => Tensor::cat(&[&x, &x], 0)?
                .apply_op1_no_bwd(weight.as_ref())?
                .narrow(0, 0, 1)?,
            F16Weight::Ggml(weight) => x.apply_op1_no_bwd(weight.as_ref())?,
            F16Weight::Dense(weight) => x.matmul(&weight.to_dtype(DType::F32)?.t()?)?,
        };
        let y = match &self.bias {
            Some(bias) => y.broadcast_add(bias)?,
            None => y,
        };
        let mut out_dims = leading.to_vec();
        out_dims.push(self.out_features);
        y.reshape(out_dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::numeric_check::{host_f32, max_abs_diff, relative_l2, xorshift_tensor};
    use candle_nn::Linear;

    /// A dense F32 reference and the F16 layer built from the same weights.
    fn layers(
        out_features: usize,
        in_features: usize,
        with_bias: bool,
        device: &Device,
    ) -> (Linear, F16WeightLinear) {
        let weight = xorshift_tensor(&[out_features, in_features], 11, 0.2, &Device::Cpu).unwrap();
        let bias =
            with_bias.then(|| xorshift_tensor(&[out_features], 12, 0.5, &Device::Cpu).unwrap());
        let dense = Linear::new(weight.clone(), bias.clone());
        let f16 = F16WeightLinear::new(&weight, bias.as_ref(), device).unwrap();
        (dense, f16)
    }

    fn compare(dense: &Linear, f16: &F16WeightLinear, x: &Tensor) -> (f64, f32) {
        let reference =
            host_f32(&dense.forward(&x.to_device(&Device::Cpu).unwrap()).unwrap()).unwrap();
        let y = f16.forward(x).unwrap();
        assert_eq!(y.dtype(), DType::F32);
        let values = host_f32(&y).unwrap();
        (
            relative_l2(&values, &reference),
            max_abs_diff(&values, &reference),
        )
    }

    #[test]
    fn cpu_matches_the_dense_f32_linear_within_f16_weight_rounding() {
        let device = Device::Cpu;
        for (out_features, in_features, with_bias) in
            [(96, 64, true), (40, 24, false), (7, 1024, true)]
        {
            let (dense, f16) = layers(out_features, in_features, with_bias, &device);
            assert_eq!(f16.path(), F16WeightLinearPath::DenseF16Upcast);
            for shape in [
                vec![1, 5, in_features],
                vec![3, in_features],
                vec![1, 1, in_features],
            ] {
                let x = xorshift_tensor(&shape, 21, 1.0, &device).unwrap();
                let (rel, _) = compare(&dense, &f16, &x);
                assert!(
                    rel < 1e-3,
                    "{out_features}x{in_features} {shape:?}: {rel:e}"
                );
                // F16 rounds weights, so the result is close but not exact.
                assert!(rel > 0.0, "the weight was not rounded to F16");
            }
            let y = f16
                .forward(&xorshift_tensor(&[2, 3, in_features], 22, 1.0, &device).unwrap())
                .unwrap();
            assert_eq!(y.dims(), &[2, 3, out_features]);
        }
    }

    #[test]
    fn stores_the_weight_in_half_and_widens_it_exactly() {
        let (dense, f16) = layers(32, 64, true, &Device::Cpu);
        assert_eq!(f16.resident_bytes(), 32 * 64 * 2 + 32 * 4);
        let widened = host_f32(&f16.dequantized_weight().unwrap()).unwrap();
        let rounded = host_f32(
            &dense
                .weight()
                .to_dtype(DType::F16)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(widened, rounded);
    }

    #[test]
    fn pointwise_conv1d_weights_load_as_linear() {
        let weight = xorshift_tensor(&[8, 4, 1], 3, 1.0, &Device::Cpu).unwrap();
        let layer = F16WeightLinear::from_pointwise_conv1d(&weight, None, &Device::Cpu).unwrap();
        assert_eq!((layer.out_features(), layer.in_features()), (8, 4));
        let bad = xorshift_tensor(&[8, 4, 3], 3, 1.0, &Device::Cpu).unwrap();
        assert!(F16WeightLinear::from_pointwise_conv1d(&bad, None, &Device::Cpu).is_err());
    }

    #[test]
    fn rejects_mismatched_input_features() {
        let (_, f16) = layers(8, 32, false, &Device::Cpu);
        let x = xorshift_tensor(&[2, 16], 1, 1.0, &Device::Cpu).unwrap();
        assert!(f16.forward(&x).is_err());
    }

    /// The Metal layer must run ggml's f16 x f32 kernel (not a dense
    /// fallback) for matrix shapes (partial 64x32 tiles included) and for a
    /// single row (which Candle would send to its broken `mul_mv`), and agree
    /// with the F32 reference and the CPU path.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_uses_the_ggml_f16_f32_kernel_and_matches_f32() {
        let Some(device) = crate::backends::metal_device_if_available(0) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        for (out_features, in_features, with_bias) in
            [(1024, 1024, false), (96, 64, true), (640, 4096, false)]
        {
            let (dense, f16) = layers(out_features, in_features, with_bias, &device);
            assert_eq!(f16.path(), F16WeightLinearPath::GgmlF16F32);
            let cpu = F16WeightLinear::new(dense.weight(), dense.bias(), &Device::Cpu).unwrap();
            for shape in [
                vec![1, 37, in_features],
                vec![130, in_features],
                vec![1, 1, in_features],
                vec![2, 1, in_features],
            ] {
                let x = xorshift_tensor(&shape, 31, 1.0, &device).unwrap();
                let (rel, max) = compare(&dense, &f16, &x);
                assert!(
                    rel < 1e-3,
                    "{out_features}x{in_features} {shape:?}: rel {rel:e} max {max:e}"
                );
                let metal = host_f32(&f16.forward(&x).unwrap()).unwrap();
                let host =
                    host_f32(&cpu.forward(&x.to_device(&Device::Cpu).unwrap()).unwrap()).unwrap();
                let rel_cpu = relative_l2(&metal, &host);
                assert!(
                    rel_cpu < 1e-5,
                    "Metal vs CPU F16 path {shape:?}: {rel_cpu:e}"
                );
            }
        }
        // A reduction dimension ggml's mul_mm cannot take falls back densely.
        let (dense, f16) = layers(16, 40, true, &device);
        assert_eq!(f16.path(), F16WeightLinearPath::DenseF16Upcast);
        let x = xorshift_tensor(&[3, 40], 5, 1.0, &device).unwrap();
        let (rel, _) = compare(&dense, &f16, &x);
        assert!(rel < 1e-3, "{rel:e}");
    }
}
