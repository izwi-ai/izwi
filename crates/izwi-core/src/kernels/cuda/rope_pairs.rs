//! Interleaved-pair rotary embedding in one launch on CUDA, in the
//! `[batch, sequence, heads, dim]` layout a projection writes.
//!
//! Rotating a half-precision projection in F32 through Candle's `rope_i`
//! takes a cast, a transpose to `[batch, heads, sequence, dim]`, the
//! rotation, a transpose back and a cast: five launches for q and five for
//! k in every layer. This kernel reads the projection where it is, rotates
//! each pair in F32 with F32 tables and rounds once to the input's dtype,
//! the rounding the chain had.

use candle_core::{CpuStorage, CustomOp3, Layout, Result, Shape, Tensor};

/// `x` [batch, sequence, heads, dim] rotated by `cos`/`sin` (F32
/// [sequence, dim / 2]) in adjacent pairs, in `x`'s dtype. CUDA only.
pub fn rotate(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (batch, sequence, heads, dim) = x.dims4()?;
    let table = [sequence, dim / 2];
    if !dim.is_multiple_of(2)
        || cos.dims() != table
        || sin.dims() != table
        || cos.dtype() != candle_core::DType::F32
        || sin.dtype() != candle_core::DType::F32
    {
        candle_core::bail!(
            "izwi-rope-pairs: x {:?}, cos {:?} {:?}, sin {:?}",
            x.dims(),
            cos.dims(),
            cos.dtype(),
            sin.dims()
        )
    }
    x.contiguous()?.apply_op3_no_bwd(
        &cos.contiguous()?,
        &sin.contiguous()?,
        &RopePairs {
            tokens: batch * sequence,
            sequence,
            heads,
            half: dim / 2,
        },
    )
}

/// Whether [`rotate`] takes `dtype` on `device`.
pub fn serves(device: &candle_core::Device, dtype: candle_core::DType) -> bool {
    use candle_core::DType;
    cfg!(feature = "cuda")
        && device.is_cuda()
        && matches!(dtype, DType::F32 | DType::F16 | DType::BF16)
}

struct RopePairs {
    tokens: usize,
    sequence: usize,
    heads: usize,
    half: usize,
}

impl CustomOp3 for RopePairs {
    fn name(&self) -> &'static str {
        "izwi-rope-pairs"
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
        candle_core::bail!("izwi-rope-pairs runs on CUDA only")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        x: &candle_core::CudaStorage,
        xl: &Layout,
        cos: &candle_core::CudaStorage,
        cl: &Layout,
        sin: &candle_core::CudaStorage,
        sl: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::{CudaStorageSlice, WrapErr};
        let device = x.device().clone();
        let len = self.tokens * self.heads * self.half * 2;
        let table = self.sequence * self.half;
        let (CudaStorageSlice::F32(c), CudaStorageSlice::F32(s)) = (&cos.slice, &sin.slice) else {
            candle_core::bail!("izwi-rope-pairs takes F32 tables")
        };
        let c = c.slice(cl.start_offset()..cl.start_offset() + table);
        let s = s.slice(sl.start_offset()..sl.start_offset() + table);
        let pairs = (len / 2) as u64;
        let config = LaunchConfig {
            grid_dim: (pairs.div_ceil(256).clamp(1, 16384) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        macro_rules! run {
            ($variant:ident, $ty:ty, $name:literal) => {{
                let CudaStorageSlice::$variant(xs) = &x.slice else {
                    unreachable!()
                };
                let xs = xs.slice(xl.start_offset()..xl.start_offset() + len);
                // SAFETY: every element of the output is written (one thread
                // per pair, both members).
                let out = unsafe { device.alloc::<$ty>(len)? };
                let func = device.get_or_load_custom_func(
                    $name,
                    "izwi_rope_pairs",
                    super::cuda_ptx::ROPE_PAIRS,
                )?;
                let mut builder = func.builder();
                builder.arg(&xs);
                builder.arg(&c);
                builder.arg(&s);
                builder.arg(&out);
                candle_core::builder_arg!(
                    builder,
                    self.tokens as i64,
                    self.sequence as i32,
                    self.heads as i32,
                    self.half as i32
                );
                // SAFETY: the arguments follow `izwi_rope_pairs_*`, and the
                // views cover the geometry checked at the call.
                unsafe { builder.launch(config) }.w()?;
                CudaStorageSlice::$variant(out)
            }};
        }
        let slice = match &x.slice {
            CudaStorageSlice::BF16(_) => run!(BF16, half::bf16, "izwi_rope_pairs_bf16"),
            CudaStorageSlice::F16(_) => run!(F16, half::f16, "izwi_rope_pairs_f16"),
            CudaStorageSlice::F32(_) => run!(F32, f32, "izwi_rope_pairs_f32"),
            _ => candle_core::bail!("izwi-rope-pairs takes F32, F16 or BF16"),
        };
        Ok((
            candle_core::CudaStorage { slice, device },
            xl.shape().clone(),
        ))
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn cuda_rope_pairs_match_the_f32_rope_i_chain() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        let values =
            |len: usize, seed: u64| crate::kernels::numeric_check::xorshift_values(len, seed, 1.0);
        let (b, t, h, d) = (2, 5, 3, 16);
        let x = Tensor::from_vec(values(b * t * h * d, 1), (b, t, h, d), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let angles = values(t * d / 2, 2);
        let table = |f: fn(f32) -> f32| {
            Tensor::from_vec(
                angles.iter().map(|a| f(*a)).collect::<Vec<_>>(),
                (t, d / 2),
                &device,
            )
            .unwrap()
        };
        let (cos, sin) = (table(f32::cos), table(f32::sin));
        let want = candle_nn::rotary_emb::rope_i(
            &x.to_dtype(DType::F32)
                .unwrap()
                .transpose(1, 2)
                .unwrap()
                .contiguous()
                .unwrap(),
            &cos,
            &sin,
        )
        .unwrap()
        .transpose(1, 2)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let got = rotate(&x, &cos, &sin).unwrap();
        let diff = (got.to_dtype(DType::F32).unwrap() - want.to_dtype(DType::F32).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        // One BF16 rounding of the same F32 values; an FMA contraction may
        // move a value across a rounding boundary.
        assert!(diff <= 1.6e-2, "max difference {diff}");
    }
}
