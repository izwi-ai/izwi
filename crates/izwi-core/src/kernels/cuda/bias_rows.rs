//! A linear layer's bias added in place, optionally followed by the exact
//! (erf) GELU, on CUDA without uploading anything.
//!
//! Candle adds a bias as a broadcast add: its strided kernel uploads the
//! operands' dims and strides on every launch, and on a deep stream that
//! small upload can wait for the queue. For F32 layers, where cuBLASLt's
//! epilogue would switch the GEMM to TF32 ([`super::lt_linear`] serves the
//! half types), this kernel replaces the add, and the GELU after it, with
//! one launch whose geometry rides as arguments. Bits match the eager chain:
//! the sum is rounded to the output's dtype, then GELU is Candle's
//! `x * normcdf(x)` in F32.

use candle_core::{bail, CpuStorage, InplaceOp2, Layout, Result, Tensor};

/// `y` ([..., cols], contiguous, fresh) += `bias` ([cols]), then GELU (erf)
/// when `gelu`; in place. CUDA only: see [`serves`].
pub fn add_bias(y: &Tensor, bias: &Tensor, gelu: bool) -> Result<()> {
    let cols = y.dim(candle_core::D::Minus1)?;
    if bias.dims() != [cols] || bias.dtype() != y.dtype() || !y.is_contiguous() {
        bail!(
            "izwi-bias-rows: y {:?} {:?} (contiguous {}), bias {:?} {:?}",
            y.dims(),
            y.dtype(),
            y.is_contiguous(),
            bias.dims(),
            bias.dtype()
        )
    }
    y.inplace_op2(&bias.contiguous()?, &BiasRows { gelu })
}

/// Whether [`add_bias`] takes `dtype` on `device`.
pub fn serves(device: &candle_core::Device, dtype: candle_core::DType) -> bool {
    use candle_core::DType;
    cfg!(feature = "cuda")
        && device.is_cuda()
        && matches!(dtype, DType::F32 | DType::F16 | DType::BF16)
}

/// `x` [..., in] through `linear` exactly as Candle's `Linear::forward`
/// multiplies, the bias (and GELU) through [`add_bias`] where it serves;
/// elsewhere Candle's ops (`gelu_erf` for the GELU).
pub fn linear_forward(linear: &candle_nn::Linear, x: &Tensor, gelu: bool) -> Result<Tensor> {
    use candle_nn::Module;
    match linear.bias() {
        Some(bias) if serves(x.device(), x.dtype()) => {
            // Candle's `Linear::forward` matmul, case for case.
            let weight = linear.weight();
            let y = match *x.dims() {
                [b1, b2, m, k] if x.is_contiguous() => x
                    .reshape((b1 * b2 * m, k))?
                    .matmul(&weight.t()?)?
                    .reshape((b1, b2, m, ()))?,
                [b1, b2, _, _] => x.matmul(&weight.broadcast_left((b1, b2))?.t()?)?,
                [b, m, k] if x.is_contiguous() => x
                    .reshape((b * m, k))?
                    .matmul(&weight.t()?)?
                    .reshape((b, m, ()))?,
                [b, _, _] => x.matmul(&weight.broadcast_left(b)?.t()?)?,
                _ => x.matmul(&weight.t()?)?,
            };
            let y = y.contiguous()?;
            add_bias(&y, bias, gelu)?;
            Ok(y)
        }
        _ => {
            let y = linear.forward(x)?;
            if gelu {
                y.gelu_erf()
            } else {
                Ok(y)
            }
        }
    }
}

struct BiasRows {
    gelu: bool,
}

impl InplaceOp2 for BiasRows {
    fn name(&self) -> &'static str {
        "izwi-bias-rows"
    }

    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<()> {
        bail!("izwi-bias-rows runs on CUDA only")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        y: &mut candle_core::CudaStorage,
        yl: &Layout,
        bias: &candle_core::CudaStorage,
        bl: &Layout,
    ) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::{CudaStorageSlice, WrapErr};
        let Some((y_start, y_end)) = yl.contiguous_offsets() else {
            bail!("izwi-bias-rows needs a contiguous output")
        };
        let cols = bl.shape().elem_count();
        let total = y_end - y_start;
        if cols == 0 || total % cols != 0 {
            bail!("izwi-bias-rows: {total} values do not split into rows of {cols}")
        }
        let rows = (total / cols) as i64;
        let device = y.device.clone();
        let config = LaunchConfig {
            grid_dim: (total.div_ceil(256).clamp(1, 16384) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let gelu = i32::from(self.gelu);
        macro_rules! run {
            ($variant:ident, $name:literal) => {{
                let (CudaStorageSlice::$variant(ys), CudaStorageSlice::$variant(bs)) =
                    (&mut y.slice, &bias.slice)
                else {
                    bail!("izwi-bias-rows operands must share a dtype")
                };
                let mut ys = ys.slice_mut(y_start..y_end);
                let bs = bs.slice(bl.start_offset()..bl.start_offset() + cols);
                let func = device.get_or_load_custom_func(
                    $name,
                    "izwi_bias_rows",
                    super::cuda_ptx::BIAS_ROWS,
                )?;
                let mut builder = func.builder();
                builder.arg(&mut ys);
                builder.arg(&bs);
                candle_core::builder_arg!(builder, rows, cols as i64, gelu);
                // SAFETY: the arguments follow `izwi_bias_rows_*`; the views
                // cover rows * cols outputs and cols biases.
                unsafe { builder.launch(config) }.w()?;
            }};
        }
        match &y.slice {
            CudaStorageSlice::F32(_) => run!(F32, "izwi_bias_rows_f32"),
            CudaStorageSlice::F16(_) => run!(F16, "izwi_bias_rows_f16"),
            CudaStorageSlice::BF16(_) => run!(BF16, "izwi_bias_rows_bf16"),
            _ => bail!("izwi-bias-rows takes F32, F16 or BF16"),
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn cuda_linear_matches_candle_bit_for_bit() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        let values =
            |len: usize, seed: u64| crate::kernels::numeric_check::xorshift_values(len, seed, 0.5);
        let (input, out) = (64, 40);
        let x = Tensor::from_vec(values(3 * 5 * input, 1), (3, 5, input), &device).unwrap();
        let w = Tensor::from_vec(values(out * input, 2), (out, input), &device).unwrap();
        let b = Tensor::from_vec(values(out, 3), out, &device).unwrap();
        let linear = candle_nn::Linear::new(w, Some(b));
        for gelu in [false, true] {
            let got = linear_forward(&linear, &x, gelu).unwrap();
            let mut want = candle_nn::Module::forward(&linear, &x).unwrap();
            if gelu {
                want = want.gelu_erf().unwrap();
            }
            let diff = (got - want)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert_eq!(diff, 0.0, "gelu {gelu}");
        }
    }
}
