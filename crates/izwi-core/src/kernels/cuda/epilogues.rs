//! Typed epilogues; all reductions and nonlinear arithmetic accumulate in F32.
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::{
    backend::BackendStorage,
    cuda_backend::{CudaStorageSlice, WrapErr},
    CudaStorage, Layout, Result, Shape,
};

fn offsets(layout: &Layout) -> Result<std::ops::Range<usize>> {
    let (a, b) = layout
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("epilogue requires contiguous storage".into()))?;
    Ok(a..b)
}

pub(super) fn silu(
    a: &CudaStorage,
    al: &Layout,
    b: &CudaStorage,
    bl: &Layout,
) -> Result<(CudaStorage, Shape)> {
    let n = al.shape().elem_count();
    if n == 0 || n > i32::MAX as usize || al.shape() != bl.shape() {
        candle_core::bail!("invalid SiLU shape")
    }
    let device = a.device();
    macro_rules! run {
        ($variant:ident,$ty:ty,$symbol:literal) => {{
            let (CudaStorageSlice::$variant(a), CudaStorageSlice::$variant(b)) =
                (&a.slice, &b.slice)
            else {
                candle_core::bail!("SiLU dtype mismatch")
            };
            let a = a.slice(offsets(al)?);
            let b = b.slice(offsets(bl)?);
            // SAFETY: every output is written, and bounds and dtypes match the ABI.
            let mut out = unsafe { device.alloc::<$ty>(n)? };
            let f = device.get_or_load_custom_func(
                $symbol,
                "izwi_qwen38_decode_epilogues",
                super::cuda_ptx::QWEN38,
            )?;
            let mut launch = f.builder();
            launch.arg(&a);
            launch.arg(&b);
            launch.arg(&mut out);
            candle_core::builder_arg!(launch, n as i32);
            unsafe { launch.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
            CudaStorageSlice::$variant(out)
        }};
    }
    let slice = match &a.slice {
        CudaStorageSlice::F32(_) => run!(F32, f32, "qwen38_silu_mul_decode_f32"),
        CudaStorageSlice::F16(_) => run!(F16, half::f16, "qwen38_silu_mul_decode_f16"),
        CudaStorageSlice::BF16(_) => run!(BF16, half::bf16, "qwen38_silu_mul_decode_bf16"),
        _ => candle_core::bail!("unsupported epilogue dtype"),
    };
    Ok((
        CudaStorage {
            slice,
            device: device.clone(),
        },
        al.shape().clone(),
    ))
}

pub(super) fn l2(
    a: &CudaStorage,
    al: &Layout,
    rows: usize,
    width: usize,
    eps: f32,
) -> Result<(CudaStorage, Shape)> {
    norm(a, al, None, None, rows, width, eps)
}
#[allow(clippy::too_many_arguments)]
pub(super) fn rms(
    a: &CudaStorage,
    al: &Layout,
    b: &CudaStorage,
    bl: &Layout,
    w: &CudaStorage,
    wl: &Layout,
    rows: usize,
    width: usize,
    eps: f32,
) -> Result<(CudaStorage, Shape)> {
    norm(a, al, Some((b, bl)), Some((w, wl)), rows, width, eps)
}
fn norm(
    a: &CudaStorage,
    al: &Layout,
    b: Option<(&CudaStorage, &Layout)>,
    w: Option<(&CudaStorage, &Layout)>,
    rows: usize,
    width: usize,
    eps: f32,
) -> Result<(CudaStorage, Shape)> {
    let n = al.shape().elem_count();
    if rows == 0
        || width == 0
        || n > i32::MAX as usize
        || rows.checked_mul(width) != Some(n)
        || b.is_some_and(|(_, l)| l.shape() != al.shape())
        || w.is_some_and(|(_, l)| l.shape().elem_count() != width)
    {
        candle_core::bail!("invalid normalization shape")
    }
    let device = a.device();
    macro_rules! run {
        ($variant:ident,$ty:ty,$l2:literal,$rms:literal) => {{
            let CudaStorageSlice::$variant(a) = &a.slice else {
                unreachable!()
            };
            let a = a.slice(offsets(al)?);
            let bv = if let Some((b, l)) = b {
                let CudaStorageSlice::$variant(b) = &b.slice else {
                    candle_core::bail!("gate dtype mismatch")
                };
                Some(b.slice(offsets(l)?))
            } else {
                None
            };
            let wv = if let Some((w, l)) = w {
                let CudaStorageSlice::$variant(w) = &w.slice else {
                    candle_core::bail!("weight dtype mismatch")
                };
                Some(w.slice(offsets(l)?))
            } else {
                None
            };
            // SAFETY: one block writes each validated complete row.
            let mut out = unsafe { device.alloc::<$ty>(n)? };
            let f = device.get_or_load_custom_func(
                if b.is_some() { $rms } else { $l2 },
                "izwi_qwen38_decode_epilogues",
                super::cuda_ptx::QWEN38,
            )?;
            let mut launch = f.builder();
            launch.arg(&a);
            if let Some(b) = &bv {
                launch.arg(b);
            }
            if let Some(w) = &wv {
                launch.arg(w);
            }
            launch.arg(&mut out);
            candle_core::builder_arg!(launch, width as i32, eps);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 1024,
                })
            }
            .w()?;
            CudaStorageSlice::$variant(out)
        }};
    }
    let slice = match &a.slice {
        CudaStorageSlice::F32(_) => run!(
            F32,
            f32,
            "qwen38_l2_norm_decode_f32",
            "qwen38_gated_rms_norm_decode_f32"
        ),
        CudaStorageSlice::F16(_) => run!(
            F16,
            half::f16,
            "qwen38_l2_norm_decode_f16",
            "qwen38_gated_rms_norm_decode_f16"
        ),
        CudaStorageSlice::BF16(_) => run!(
            BF16,
            half::bf16,
            "qwen38_l2_norm_decode_bf16",
            "qwen38_gated_rms_norm_decode_bf16"
        ),
        _ => candle_core::bail!("unsupported norm dtype"),
    };
    Ok((
        CudaStorage {
            slice,
            device: device.clone(),
        },
        al.shape().clone(),
    ))
}
