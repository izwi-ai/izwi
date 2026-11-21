//! Bidirectional attention over a few positions, straight from a fused
//! q/k/v projection, with the interleaved rotation applied on the way in.
//!
//! Ming's flow transformers attend over two or three positions. Done with
//! Candle's ops that is a rotation per q and k, four head-major copies and
//! a flash-attention launch sized for long sequences (about 23 µs on the
//! H100 for three positions). This kernel reads q, k and v where the fused
//! projection wrote them and writes the output position-major, one launch
//! per layer. The CPU computes the same thing with Candle's ops (the test
//! reference and the eager path elsewhere).

use candle_core::{CpuStorage, CustomOp3, Layout, Result, Shape, Tensor};

/// The most positions the kernel takes.
pub const MAX_POSITIONS: usize = 4;
const MAX_DIM: usize = 256;

/// `qkv` [batch, positions, 3 · heads · dim] (q, k, v each heads · dim),
/// `cos`/`sin` [positions, dim / 2] in `qkv`'s dtype → the attention output,
/// [batch, positions, heads · dim]. CUDA only; see [`serves`].
pub fn rope_attention(
    qkv: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let (batch, positions, width) = qkv.dims3()?;
    if width != 3 * heads * head_dim
        || positions == 0
        || positions > MAX_POSITIONS
        || !head_dim.is_multiple_of(2)
        || head_dim > MAX_DIM
        || cos.dims() != [positions, head_dim / 2]
        || sin.dims() != [positions, head_dim / 2]
        || cos.dtype() != qkv.dtype()
        || sin.dtype() != qkv.dtype()
    {
        candle_core::bail!(
            "izwi-small-rope-attention: qkv {:?}, cos {:?}, {heads} heads of {head_dim}",
            qkv.dims(),
            cos.dims()
        )
    }
    qkv.contiguous()?.apply_op3_no_bwd(
        &cos.contiguous()?,
        &sin.contiguous()?,
        &SmallRopeAttention {
            batch,
            positions,
            heads,
            head_dim,
        },
    )
}

/// Whether [`rope_attention`] takes this geometry on `device`.
pub fn serves(device: &candle_core::Device, positions: usize, head_dim: usize) -> bool {
    cfg!(feature = "cuda")
        && device.is_cuda()
        && (1..=MAX_POSITIONS).contains(&positions)
        && head_dim.is_multiple_of(2)
        && head_dim <= MAX_DIM
}

struct SmallRopeAttention {
    batch: usize,
    positions: usize,
    heads: usize,
    head_dim: usize,
}

impl CustomOp3 for SmallRopeAttention {
    fn name(&self) -> &'static str {
        "izwi-small-rope-attention"
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
        candle_core::bail!("izwi-small-rope-attention runs on CUDA only")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        qkv: &candle_core::CudaStorage,
        ql: &Layout,
        cos: &candle_core::CudaStorage,
        cl: &Layout,
        sin: &candle_core::CudaStorage,
        sl: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::{CudaStorageSlice, WrapErr};
        let device = qkv.device().clone();
        let (batch, positions, heads, dim) =
            (self.batch, self.positions, self.heads, self.head_dim);
        let out_len = batch * positions * heads * dim;
        let blocks = u32::try_from(batch * heads)
            .map_err(|_| candle_core::Error::Msg("izwi-small-rope-attention grid".into()))?;
        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (dim.div_ceil(32) as u32 * 32, 1, 1),
            shared_mem_bytes: 0,
        };
        let scale = 1.0 / (dim as f32).sqrt();
        macro_rules! run {
            ($variant:ident, $ty:ty, $name:literal) => {{
                let (
                    CudaStorageSlice::$variant(q),
                    CudaStorageSlice::$variant(c),
                    CudaStorageSlice::$variant(s),
                ) = (&qkv.slice, &cos.slice, &sin.slice)
                else {
                    candle_core::bail!("izwi-small-rope-attention operands must share a dtype")
                };
                let q = q.slice(ql.start_offset()..ql.start_offset() + 3 * out_len);
                let c = c.slice(cl.start_offset()..cl.start_offset() + positions * dim / 2);
                let s = s.slice(sl.start_offset()..sl.start_offset() + positions * dim / 2);
                // SAFETY: every output element is written by its block.
                let out = unsafe { device.alloc::<$ty>(out_len)? };
                let func = device.get_or_load_custom_func(
                    $name,
                    "izwi_small_attention",
                    super::cuda_ptx::SMALL_ATTENTION,
                )?;
                let mut builder = func.builder();
                builder.arg(&q);
                builder.arg(&c);
                builder.arg(&s);
                builder.arg(&out);
                candle_core::builder_arg!(
                    builder,
                    positions as i32,
                    heads as i32,
                    dim as i32,
                    scale
                );
                // SAFETY: the arguments follow `izwi_small_rope_attention_*`
                // and the views cover the geometry checked at the call.
                unsafe { builder.launch(config) }.w()?;
                CudaStorageSlice::$variant(out)
            }};
        }
        let slice = match &qkv.slice {
            CudaStorageSlice::BF16(_) => run!(BF16, half::bf16, "izwi_small_rope_attention_bf16"),
            CudaStorageSlice::F16(_) => run!(F16, half::f16, "izwi_small_rope_attention_f16"),
            CudaStorageSlice::F32(_) => run!(F32, f32, "izwi_small_rope_attention_f32"),
            _ => candle_core::bail!("izwi-small-rope-attention takes F32, F16 or BF16"),
        };
        Ok((
            candle_core::CudaStorage { slice, device },
            Shape::from((batch, positions, heads * dim)),
        ))
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use candle_core::{DType, Device, D};

    /// Candle's ops: rope_i on q and k, softmax attention in F32.
    fn reference(qkv: &Tensor, cos: &Tensor, sin: &Tensor, heads: usize, dim: usize) -> Tensor {
        let (batch, positions, _) = qkv.dims3().unwrap();
        let part = |index: usize| {
            qkv.narrow(2, index * heads * dim, heads * dim)
                .unwrap()
                .reshape((batch, positions, heads, dim))
                .unwrap()
                .transpose(1, 2)
                .unwrap()
                .contiguous()
                .unwrap()
        };
        let q = candle_nn::rotary_emb::rope_i(&part(0), cos, sin).unwrap();
        let k = candle_nn::rotary_emb::rope_i(&part(1), cos, sin).unwrap();
        let v = part(2);
        let scores = (q.matmul(&k.t().unwrap()).unwrap() / (dim as f64).sqrt()).unwrap();
        let weights = candle_nn::ops::softmax(&scores, D::Minus1).unwrap();
        weights
            .matmul(&v)
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .reshape((batch, positions, heads * dim))
            .unwrap()
    }

    #[test]
    fn cuda_small_attention_matches_rope_and_softmax() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        let values =
            |len: usize, seed: u64| crate::kernels::numeric_check::xorshift_values(len, seed, 1.0);
        for (batch, positions, heads, dim) in [(5, 3, 4, 64), (2, 2, 3, 128)] {
            let qkv = Tensor::from_vec(
                values(batch * positions * 3 * heads * dim, 1),
                (batch, positions, 3 * heads * dim),
                &Device::Cpu,
            )
            .unwrap();
            let angles = values(positions * dim / 2, 2);
            let cos = Tensor::from_vec(
                angles.iter().map(|a| a.cos()).collect::<Vec<_>>(),
                (positions, dim / 2),
                &Device::Cpu,
            )
            .unwrap();
            let sin = Tensor::from_vec(
                angles.iter().map(|a| a.sin()).collect::<Vec<_>>(),
                (positions, dim / 2),
                &Device::Cpu,
            )
            .unwrap();
            let want = reference(&qkv, &cos, &sin, heads, dim)
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            for dtype in [DType::F32, DType::BF16] {
                let on = |t: &Tensor| t.to_dtype(dtype).unwrap().to_device(&device).unwrap();
                let got = rope_attention(&on(&qkv), &on(&cos), &on(&sin), heads, dim)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
                    .to_device(&Device::Cpu)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let tolerance = if dtype == DType::F32 { 1e-4 } else { 3e-2 };
                for (index, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        (g - w).abs() <= tolerance * w.abs().max(1.0),
                        "{dtype:?} {index}: {g} {w}"
                    );
                }
            }
        }
    }
}
