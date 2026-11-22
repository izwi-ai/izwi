//! Row groups through their own weight matrices in one cuBLAS call.
//!
//! `y[rows of group g] = x[rows of group g] · W[e_g]ᵀ`: `x` is `[P, K]` with
//! each group's rows contiguous (a mixture of experts' tokens sorted by
//! expert), `W` is `[E, N, K]` (each expert's rows of a linear layer), and
//! every group is a GEMM of exactly its own rows against its expert's
//! resident weights, all in one `cublasGemmGroupedBatchedEx` on tensor
//! cores. Nothing is padded to the busiest group and no weight is copied.
//! BF16 and F16 on CUDA built against toolkit 12.5 or later (cuBLAS's
//! grouped GEMM; see [`available_on_device`]); the CPU computes the same
//! products in F32 (a test reference), and other devices are refused.

use candle_core::{CpuStorage, CustomOp2, DType, Layout, Result, Shape, Tensor};

/// One group: the expert whose weights it uses, its first row in `x` and
/// its row count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowGroup {
    pub expert: usize,
    pub first_row: usize,
    pub rows: usize,
}

/// `[P, N]`: each group's rows of `x` (`[P, K]`) times its expert's weights
/// (`weights` `[E, N, K]`) transposed. Rows in no group come back zero.
pub fn grouped_rows_gemm(x: &Tensor, weights: &Tensor, groups: &[RowGroup]) -> Result<Tensor> {
    let (rows, k) = x.dims2()?;
    let (experts, n, wk) = weights.dims3()?;
    if wk != k || x.dtype() != weights.dtype() {
        candle_core::bail!(
            "grouped GEMM operands disagree: x {:?} {:?}, weights {:?} {:?}",
            x.dims(),
            x.dtype(),
            weights.dims(),
            weights.dtype()
        )
    }
    for group in groups {
        if group.expert >= experts
            || group.rows == 0
            || group
                .first_row
                .checked_add(group.rows)
                .is_none_or(|end| end > rows)
        {
            candle_core::bail!("grouped GEMM group {group:?} is out of range")
        }
    }
    x.contiguous()?.apply_op2_no_bwd(
        &weights.contiguous()?,
        &GroupedRowsGemm {
            groups: groups.to_vec(),
            n,
            k,
        },
    )
}

struct GroupedRowsGemm {
    groups: Vec<RowGroup>,
    n: usize,
    k: usize,
}

impl CustomOp2 for GroupedRowsGemm {
    fn name(&self) -> &'static str {
        "izwi-grouped-rows-gemm"
    }

    /// The reference: F32 products of the operands as stored, rounded once
    /// to the operands' dtype.
    fn cpu_fwd(
        &self,
        x: &CpuStorage,
        xl: &Layout,
        w: &CpuStorage,
        wl: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (n, k) = (self.n, self.k);
        let rows = xl.shape().elem_count() / k;
        let to_f32 = |s: &CpuStorage, l: &Layout, len: usize| -> Result<Vec<f32>> {
            let range = l.start_offset()..l.start_offset() + len;
            Ok(match s {
                CpuStorage::F32(v) => v[range].to_vec(),
                CpuStorage::F16(v) => v[range].iter().map(|x| x.to_f32()).collect(),
                CpuStorage::BF16(v) => v[range].iter().map(|x| x.to_f32()).collect(),
                _ => candle_core::bail!("grouped GEMM takes F32, F16 or BF16"),
            })
        };
        let xs = to_f32(x, xl, rows * k)?;
        let ws = to_f32(w, wl, wl.shape().elem_count())?;
        let mut y = vec![0f32; rows * n];
        for group in &self.groups {
            let weights = &ws[group.expert * n * k..(group.expert + 1) * n * k];
            for row in group.first_row..group.first_row + group.rows {
                let input = &xs[row * k..(row + 1) * k];
                for (column, out) in y[row * n..(row + 1) * n].iter_mut().enumerate() {
                    *out = input
                        .iter()
                        .zip(&weights[column * k..(column + 1) * k])
                        .fold(0f32, |total, (a, b)| a.mul_add(*b, total));
                }
            }
        }
        let storage = match x {
            CpuStorage::F16(_) => CpuStorage::F16(y.into_iter().map(half::f16::from_f32).collect()),
            CpuStorage::BF16(_) => {
                CpuStorage::BF16(y.into_iter().map(half::bf16::from_f32).collect())
            }
            _ => CpuStorage::F32(y),
        };
        Ok((storage, Shape::from((rows, n))))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        x: &candle_core::CudaStorage,
        xl: &Layout,
        w: &candle_core::CudaStorage,
        wl: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        #[cfg(not(izwi_cublas_grouped))]
        {
            let _ = (x, xl, w, wl);
            candle_core::bail!("this build's CUDA toolkit has no cuBLAS grouped GEMM (12.5+)")
        }
        #[cfg(izwi_cublas_grouped)]
        self.cuda_grouped(x, xl, w, wl)
    }
}

impl GroupedRowsGemm {
    #[cfg(all(feature = "cuda", izwi_cublas_grouped))]
    fn cuda_grouped(
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
        let device = x.device();
        let (n, k) = (self.n, self.k);
        let rows = xl.shape().elem_count() / k;
        if [rows, n, k].iter().any(|v| *v > i32::MAX as usize) {
            candle_core::bail!("grouped GEMM size overflow")
        }
        let stream = device.cuda_stream();
        let blas = device.cublas_handle();
        macro_rules! run {
            ($v:ident, $ty:ty, $data:ident) => {{
                let (CudaStorageSlice::$v(xs), CudaStorageSlice::$v(ws)) = (&x.slice, &w.slice)
                else {
                    candle_core::bail!("grouped GEMM operands must share a dtype")
                };
                let xs = xs.slice(xl.start_offset()..xl.start_offset() + rows * k);
                let ws = ws.slice(wl.start_offset()..wl.start_offset() + wl.shape().elem_count());
                let mut out = device.alloc_zeros::<$ty>(rows * n)?;
                if !self.groups.is_empty() {
                    let bytes = std::mem::size_of::<$ty>() as u64;
                    let (x_ptr, _x) = xs.device_ptr(&stream);
                    let (w_ptr, _w) = ws.device_ptr(&stream);
                    let (y_ptr, _y) = out.device_ptr_mut(&stream);
                    let groups = self.groups.len();
                    // Row-major y[r, n] = x[r, k] W[e]ᵀ is column-major
                    // Y[n, r] = W[e] (as stored, [k, n] column-major,
                    // transposed) times X ([k, r] column-major).
                    let a = self
                        .groups
                        .iter()
                        .map(|g| w_ptr + (g.expert * n * k) as u64 * bytes)
                        .collect::<Vec<u64>>();
                    let b = self
                        .groups
                        .iter()
                        .map(|g| x_ptr + (g.first_row * k) as u64 * bytes)
                        .collect::<Vec<u64>>();
                    let c = self
                        .groups
                        .iter()
                        .map(|g| y_ptr + (g.first_row * n) as u64 * bytes)
                        .collect::<Vec<u64>>();
                    let a_dev = device.clone_htod(&a)?;
                    let b_dev = device.clone_htod(&b)?;
                    let c_dev = device.clone_htod(&c)?;
                    let (a_dev_ptr, _a) = a_dev.device_ptr(&stream);
                    let (b_dev_ptr, _b) = b_dev.device_ptr(&stream);
                    let (c_dev_ptr, _c) = c_dev.device_ptr(&stream);
                    let trans_a = vec![cublas_sys::cublasOperation_t::CUBLAS_OP_T; groups];
                    let trans_b = vec![cublas_sys::cublasOperation_t::CUBLAS_OP_N; groups];
                    let m_array = vec![n as i32; groups];
                    let n_array = self
                        .groups
                        .iter()
                        .map(|g| g.rows as i32)
                        .collect::<Vec<_>>();
                    let k_array = vec![k as i32; groups];
                    let alpha = vec![1f32; groups];
                    let beta = vec![0f32; groups];
                    let lda = vec![k as i32; groups];
                    let ldb = vec![k as i32; groups];
                    let ldc = vec![n as i32; groups];
                    let sizes = vec![1i32; groups];
                    // SAFETY: the pointer arrays are device copies of live
                    // device addresses on the handle's stream; each group's
                    // A, B and C cover its expert's [n, k] weights, its rows
                    // of x and its rows of y at the leading dimensions
                    // given, and the host arrays hold one entry a group.
                    let status = unsafe {
                        cublas_sys::cublasGemmGroupedBatchedEx(
                            *blas.handle(),
                            trans_a.as_ptr(),
                            trans_b.as_ptr(),
                            m_array.as_ptr(),
                            n_array.as_ptr(),
                            k_array.as_ptr(),
                            alpha.as_ptr() as *const _,
                            a_dev_ptr as *const *const _,
                            cublas_sys::cudaDataType_t::$data,
                            lda.as_ptr(),
                            b_dev_ptr as *const *const _,
                            cublas_sys::cudaDataType_t::$data,
                            ldb.as_ptr(),
                            beta.as_ptr() as *const _,
                            c_dev_ptr as *const *mut _,
                            cublas_sys::cudaDataType_t::$data,
                            ldc.as_ptr(),
                            groups as i32,
                            sizes.as_ptr(),
                            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        )
                    };
                    if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                        candle_core::bail!("grouped GEMM failed: {status:?}")
                    }
                }
                CudaStorageSlice::$v(out)
            }};
        }
        let slice = match &x.slice {
            CudaStorageSlice::BF16(_) => run!(BF16, half::bf16, CUDA_R_16BF),
            CudaStorageSlice::F16(_) => run!(F16, half::f16, CUDA_R_16F),
            _ => candle_core::bail!("the CUDA grouped GEMM takes F16 or BF16"),
        };
        Ok((
            candle_core::CudaStorage {
                slice,
                device: device.clone(),
            },
            Shape::from((rows, n)),
        ))
    }
}

/// Whether a CUDA tensor of `dtype` can take the grouped GEMM: F16 or
/// BF16, in a build whose toolkit binds cuBLAS's grouped GEMM.
pub fn available_on_device(dtype: DType) -> bool {
    cfg!(all(feature = "cuda", izwi_cublas_grouped)) && matches!(dtype, DType::BF16 | DType::F16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn reference(x: &Tensor, w: &Tensor, groups: &[RowGroup]) -> Vec<Vec<f32>> {
        let (rows, _) = x.dims2().unwrap();
        let n = w.dim(1).unwrap();
        let mut out = vec![vec![0f32; n]; rows];
        for g in groups {
            let block = x.narrow(0, g.first_row, g.rows).unwrap();
            let y = block
                .matmul(&w.get(g.expert).unwrap().t().unwrap())
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            for (offset, row) in y.into_iter().enumerate() {
                out[g.first_row + offset] = row;
            }
        }
        out
    }

    fn values(len: usize, seed: u64) -> Vec<f32> {
        crate::kernels::numeric_check::xorshift_values(len, seed, 1.0)
    }

    #[test]
    fn each_group_multiplies_only_its_own_rows_by_its_expert() {
        let device = Device::Cpu;
        let (rows, n, k, experts) = (9, 6, 8, 4);
        let x = Tensor::from_vec(values(rows * k, 3), (rows, k), &device).unwrap();
        let w = Tensor::from_vec(values(experts * n * k, 5), (experts, n, k), &device).unwrap();
        let groups = [
            RowGroup {
                expert: 2,
                first_row: 0,
                rows: 3,
            },
            RowGroup {
                expert: 0,
                first_row: 3,
                rows: 1,
            },
            RowGroup {
                expert: 3,
                first_row: 5,
                rows: 4,
            },
        ];
        let got = grouped_rows_gemm(&x, &w, &groups)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let want = reference(&x, &w, &groups);
        for (row, (a, b)) in got.iter().zip(&want).enumerate() {
            for (u, v) in a.iter().zip(b) {
                assert!((u - v).abs() < 1e-4, "row {row}: {u} {v}");
            }
        }
        // Row 4 is in no group.
        assert!(got[4].iter().all(|v| *v == 0.0));
    }

    #[test]
    fn out_of_range_groups_are_refused() {
        let device = Device::Cpu;
        let x = Tensor::zeros((4, 8), DType::F32, &device).unwrap();
        let w = Tensor::zeros((2, 6, 8), DType::F32, &device).unwrap();
        let bad = |g: RowGroup| grouped_rows_gemm(&x, &w, &[g]).is_err();
        assert!(bad(RowGroup {
            expert: 2,
            first_row: 0,
            rows: 1
        }));
        assert!(bad(RowGroup {
            expert: 0,
            first_row: 3,
            rows: 2
        }));
        assert!(bad(RowGroup {
            expert: 0,
            first_row: 0,
            rows: 0
        }));
    }

    #[cfg(all(feature = "cuda", izwi_cublas_grouped))]
    #[test]
    fn cuda_grouped_gemm_matches_the_reference() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        let (rows, n, k, experts) = (37, 64, 128, 6);
        let x = Tensor::from_vec(values(rows * k, 7), (rows, k), &Device::Cpu).unwrap();
        let w =
            Tensor::from_vec(values(experts * n * k, 9), (experts, n, k), &Device::Cpu).unwrap();
        let groups = [
            RowGroup {
                expert: 5,
                first_row: 0,
                rows: 20,
            },
            RowGroup {
                expert: 1,
                first_row: 20,
                rows: 1,
            },
            RowGroup {
                expert: 3,
                first_row: 21,
                rows: 16,
            },
        ];
        let on = |t: &Tensor| t.to_dtype(DType::BF16).unwrap().to_device(&device).unwrap();
        let got = grouped_rows_gemm(&on(&x), &on(&w), &groups)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let round = |t: &Tensor| {
            t.to_dtype(DType::BF16)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
        };
        let want = reference(&round(&x), &round(&w), &groups);
        for (row, (a, b)) in got.iter().zip(&want).enumerate() {
            for (u, v) in a.iter().zip(b) {
                assert!(
                    (u - v).abs() <= 2e-2 * v.abs().max(1.0),
                    "row {row}: {u} {v}"
                );
            }
        }
    }
}
