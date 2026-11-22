//! An F32 linear layer's product on tensor cores in TF32, for one caller.
//!
//! Candle runs F32 GEMMs in full FP32 (no tensor cores) unless a
//! process-wide switch puts every model's F32 GEMMs in TF32. This op is the
//! per-call choice: `x · Wᵀ` through `cublasGemmEx` with
//! `CUBLAS_COMPUTE_32F_FAST_TF32`, F32 in and out (inputs rounded to a
//! 10-bit mantissa inside the multiply, F32 accumulation). A fixed shape
//! picks a fixed algorithm, so a row's bits still depend on the call's shape
//! alone.

use candle_core::{CpuStorage, CustomOp2, Layout, Result, Shape, Tensor};

/// `x` [rows, in] · `weight` [out, in]ᵀ: [rows, out], F32 on CUDA only (see
/// [`serves`]).
pub fn matmul_t(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (_, input) = x.dims2()?;
    let (_, w_input) = weight.dims2()?;
    if input != w_input
        || x.dtype() != candle_core::DType::F32
        || weight.dtype() != candle_core::DType::F32
    {
        candle_core::bail!(
            "izwi-tf32-gemm operands disagree: x {:?} {:?}, weight {:?} {:?}",
            x.dims(),
            x.dtype(),
            weight.dims(),
            weight.dtype()
        )
    }
    x.contiguous()?
        .apply_op2_no_bwd(&weight.contiguous()?, &Tf32Gemm)
}

/// Whether [`matmul_t`] takes tensors of `dtype` on `device`.
pub fn serves(device: &candle_core::Device, dtype: candle_core::DType) -> bool {
    cfg!(feature = "cuda") && device.is_cuda() && dtype == candle_core::DType::F32
}

struct Tf32Gemm;

impl CustomOp2 for Tf32Gemm {
    fn name(&self) -> &'static str {
        "izwi-tf32-gemm"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("izwi-tf32-gemm runs on CUDA only")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        x: &candle_core::CudaStorage,
        xl: &Layout,
        w: &candle_core::CudaStorage,
        wl: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::cublas::sys as cublas_sys;
        use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
        use candle_core::cuda_backend::CudaStorageSlice;
        let (rows, input) = xl.shape().dims2()?;
        let (out, _) = wl.shape().dims2()?;
        if [rows, input, out].iter().any(|v| *v > i32::MAX as usize) {
            candle_core::bail!("izwi-tf32-gemm size overflow")
        }
        let (CudaStorageSlice::F32(xs), CudaStorageSlice::F32(ws)) = (&x.slice, &w.slice) else {
            candle_core::bail!("izwi-tf32-gemm takes F32")
        };
        let device = x.device().clone();
        let stream = device.cuda_stream();
        let blas = device.cublas_handle();
        let xs = xs.slice(xl.start_offset()..xl.start_offset() + rows * input);
        let ws = ws.slice(wl.start_offset()..wl.start_offset() + out * input);
        // SAFETY: the GEMM writes every element of the output (beta = 0).
        let mut y = unsafe { device.alloc::<f32>(rows * out)? };
        {
            let (x_ptr, _x) = xs.device_ptr(&stream);
            let (w_ptr, _w) = ws.device_ptr(&stream);
            let (y_ptr, _y) = y.device_ptr_mut(&stream);
            let (alpha, beta) = (1f32, 0f32);
            // Column-major: y[out, rows] = W ([input, out] as stored,
            // transposed) times x ([input, rows]).
            // SAFETY: the pointers cover [out, input], [rows, input] and
            // [rows, out] F32 at the leading dimensions given, on the
            // handle's stream.
            let status = unsafe {
                cublas_sys::cublasGemmEx(
                    *blas.handle(),
                    cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    out as i32,
                    rows as i32,
                    input as i32,
                    (&alpha) as *const f32 as *const _,
                    w_ptr as *const _,
                    cublas_sys::cudaDataType_t::CUDA_R_32F,
                    input as i32,
                    x_ptr as *const _,
                    cublas_sys::cudaDataType_t::CUDA_R_32F,
                    input as i32,
                    (&beta) as *const f32 as *const _,
                    y_ptr as *mut _,
                    cublas_sys::cudaDataType_t::CUDA_R_32F,
                    out as i32,
                    cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
                    cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
            };
            if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                candle_core::bail!("izwi-tf32-gemm failed: {status:?}")
            }
        }
        Ok((
            candle_core::CudaStorage {
                slice: CudaStorageSlice::F32(y),
                device,
            },
            Shape::from((rows, out)),
        ))
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn cuda_tf32_gemm_tracks_the_f32_product() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        let values =
            |len: usize, seed: u64| crate::kernels::numeric_check::xorshift_values(len, seed, 1.0);
        let (rows, input, out) = (37, 256, 96);
        let x = Tensor::from_vec(values(rows * input, 1), (rows, input), &device).unwrap();
        let w = Tensor::from_vec(values(out * input, 2), (out, input), &device).unwrap();
        let got = matmul_t(&x, &w).unwrap().to_device(&Device::Cpu).unwrap();
        let want = x
            .matmul(&w.t().unwrap())
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap();
        let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let want = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let error = crate::kernels::numeric_check::relative_l2(&got, &want);
        // TF32 keeps 10 mantissa bits: about 1e-3 relative over 256 terms.
        assert!(error < 5e-3, "relative error {error}");
        assert!(error > 0.0, "TF32 must differ from full F32 somewhere");
    }
}
