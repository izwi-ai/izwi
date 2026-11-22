//! A linear layer as one cuBLASLt GEMM with its bias (and optionally the
//! tanh GELU after it) fused into the epilogue.
//!
//! Candle runs `x · Wᵀ + b` as a GEMM and a broadcast add (Candle's strided
//! binary kernel), and a GELU after it as a third launch. Small models made
//! of many such layers (Ming's flow head is eight blocks of six biased
//! linears, ten Euler steps a latent) spend as much time in those adds as in
//! the GEMMs. cuBLASLt applies the bias, and the GELU, as the GEMM writes
//! its output. One handle per device, on Candle's stream, so the call is
//! ordered with Candle's kernels and records into CUDA graphs.

use candle_core::{CpuStorage, CustomOp3, Layout, Result, Shape, Tensor};

/// `x` [..., in] · `weight` [out, in]ᵀ + `bias` [out], then the tanh GELU
/// when `gelu`: [..., out] in the operands' dtype (F16 or BF16, CUDA only).
pub fn linear_bias(x: &Tensor, weight: &Tensor, bias: &Tensor, gelu: bool) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let (out, input) = weight.dims2()?;
    if dims.last() != Some(&input)
        || bias.dims() != [out]
        || x.dtype() != weight.dtype()
        || x.dtype() != bias.dtype()
    {
        candle_core::bail!(
            "izwi-lt-linear operands disagree: x {:?} {:?}, weight {:?} {:?}, bias {:?} {:?}",
            dims,
            x.dtype(),
            weight.dims(),
            weight.dtype(),
            bias.dims(),
            bias.dtype()
        )
    }
    let rows = dims[..dims.len() - 1].iter().product::<usize>();
    let y = x.reshape((rows, input))?.contiguous()?.apply_op3_no_bwd(
        &weight.contiguous()?,
        &bias.contiguous()?,
        &LtLinear { gelu },
    )?;
    let mut shape = dims;
    *shape.last_mut().expect("a linear input has a last dim") = out;
    y.reshape(shape)
}

/// Whether [`linear_bias`] takes tensors of `dtype` on `device`.
pub fn serves(device: &candle_core::Device, dtype: candle_core::DType) -> bool {
    cfg!(feature = "cuda")
        && device.is_cuda()
        && matches!(dtype, candle_core::DType::BF16 | candle_core::DType::F16)
}

struct LtLinear {
    gelu: bool,
}

impl CustomOp3 for LtLinear {
    fn name(&self) -> &'static str {
        "izwi-lt-linear"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("izwi-lt-linear runs on CUDA only")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        x: &candle_core::CudaStorage,
        xl: &Layout,
        w: &candle_core::CudaStorage,
        wl: &Layout,
        b: &candle_core::CudaStorage,
        bl: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::cublaslt::{Activation, Matmul, MatmulConfig};
        use candle_core::cuda_backend::CudaStorageSlice;
        let (rows, input) = xl.shape().dims2()?;
        let (out, _) = wl.shape().dims2()?;
        let device = x.device().clone();
        let handle = handle(&device)?;
        let config = MatmulConfig {
            // Column-major: y[out, rows] = W ([input, out] as stored,
            // transposed) times x ([input, rows]).
            transa: true,
            transb: false,
            transc: false,
            m: out as u64,
            n: rows as u64,
            k: input as u64,
            alpha: 1.0,
            lda: input as i64,
            ldb: input as i64,
            beta: 0.0,
            ldc: out as i64,
            stride_a: None,
            stride_b: None,
            stride_c: None,
            stride_bias: None,
            batch_size: None,
        };
        let act = self.gelu.then_some(Activation::Gelu);
        macro_rules! run {
            ($variant:ident, $ty:ty) => {{
                let (
                    CudaStorageSlice::$variant(xs),
                    CudaStorageSlice::$variant(ws),
                    CudaStorageSlice::$variant(bs),
                ) = (&x.slice, &w.slice, &b.slice)
                else {
                    candle_core::bail!("izwi-lt-linear operands must share a dtype")
                };
                let xs = xs.slice(xl.start_offset()..xl.start_offset() + rows * input);
                let ws = ws.slice(wl.start_offset()..wl.start_offset() + out * input);
                let bs = bs.slice(bl.start_offset()..bl.start_offset() + out);
                // SAFETY: cuBLASLt writes every element of the output.
                let mut y = unsafe { device.alloc::<$ty>(rows * out)? };
                // SAFETY: the views cover [rows, input], [out, input] and [out]
                // at the leading dimensions given, and the output [rows, out].
                unsafe { handle.matmul(config, &ws, &xs, &mut y, Some(&bs), act.as_ref()) }
                    .map_err(|e| candle_core::Error::Msg(format!("izwi-lt-linear: {e:?}")))?;
                CudaStorageSlice::$variant(y)
            }};
        }
        let slice = match &x.slice {
            CudaStorageSlice::BF16(_) => run!(BF16, half::bf16),
            CudaStorageSlice::F16(_) => run!(F16, half::f16),
            _ => candle_core::bail!("izwi-lt-linear takes F16 or BF16"),
        };
        Ok((
            candle_core::CudaStorage { slice, device },
            Shape::from((rows, out)),
        ))
    }
}

/// The device's cuBLASLt handle on Candle's stream, made on first use (with
/// its workspace, so a graph capture never allocates one).
#[cfg(feature = "cuda")]
fn handle(
    device: &candle_core::CudaDevice,
) -> Result<std::sync::Arc<candle_core::cuda_backend::cudarc::cublaslt::CudaBlasLT>> {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};
    type Handles = Mutex<
        HashMap<
            candle_core::cuda_backend::DeviceId,
            Arc<candle_core::cuda_backend::cudarc::cublaslt::CudaBlasLT>,
        >,
    >;
    static HANDLES: OnceLock<Handles> = OnceLock::new();
    let mut handles = HANDLES
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| candle_core::Error::Msg("izwi-lt-linear handles were poisoned".into()))?;
    if let Some(handle) = handles.get(&device.id()) {
        return Ok(handle.clone());
    }
    let handle = Arc::new(
        candle_core::cuda_backend::cudarc::cublaslt::CudaBlasLT::new(device.cuda_stream())
            .map_err(|e| candle_core::Error::Msg(format!("izwi-lt-linear handle: {e:?}")))?,
    );
    handles.insert(device.id(), handle.clone());
    Ok(handle)
}

/// Makes the device's handle now, outside any graph capture.
pub fn prepare(device: &candle_core::Device) -> Result<()> {
    #[cfg(feature = "cuda")]
    if let candle_core::Device::Cuda(device) = device {
        handle(device)?;
    }
    let _ = device;
    Ok(())
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn cuda_lt_linear_matches_matmul_bias_and_gelu() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        let values =
            |len: usize, seed: u64| crate::kernels::numeric_check::xorshift_values(len, seed, 0.5);
        let (rows, input, out) = (6, 64, 48);
        let x = Tensor::from_vec(values(rows * input, 1), (2, 3, input), &Device::Cpu).unwrap();
        let w = Tensor::from_vec(values(out * input, 2), (out, input), &Device::Cpu).unwrap();
        let b = Tensor::from_vec(values(out, 3), out, &Device::Cpu).unwrap();
        let on = |t: &Tensor| t.to_dtype(DType::BF16).unwrap().to_device(&device).unwrap();
        let round = |t: &Tensor| {
            t.to_dtype(DType::BF16)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
        };
        for gelu in [false, true] {
            let got = linear_bias(&on(&x), &on(&w), &on(&b), gelu)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let mut want = round(&x)
                .broadcast_matmul(&round(&w).t().unwrap())
                .unwrap()
                .broadcast_add(&round(&b))
                .unwrap();
            if gelu {
                want = want.gelu().unwrap();
            }
            let want = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for (index, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= 2e-2 * w.abs().max(1.0),
                    "{gelu} {index}: {g} {w}"
                );
            }
        }
    }
}
