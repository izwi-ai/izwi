//! A residual add fused with RMSNorm, and RMSNorm alone, over rows of any mix
//! of F32/F16/BF16 activations, on CPU, Metal and CUDA.
//!
//! ```text
//! sum    = R(float(residual) + float(delta))       R: the residual's dtype
//! n      = O(float(sum))                           O: the output's dtype
//! normed = O(float(n) * rsqrt(mean(n * n) + eps) * gain)
//! ```
//!
//! That is the eager chain each decoder runs between its layers: add the
//! mixer's output into the residual stream in F32 and round it once to the
//! stream's dtype, cast the stream to the layer's activation dtype, then
//! normalize in F32 with an F32 gain. One launch replaces the three to six of
//! the chain. Nemotron-H keeps an F32 stream and F32 activations; Drama 3 a
//! BF16 stream on CUDA and an F32 stream with F16 activations on Metal.
//!
//! Every backend sums a row's squares in its own fixed order (one thread per
//! row on the CPU, a fixed per-thread stride and tree on the GPU), so a row's
//! bits depend on that row alone, never on the rows beside it; the backends
//! agree with each other and with Candle's `rms_norm` to rounding.
use candle_core::{DType, Device, Result, Tensor, D};

fn dtype_tag(dtype: DType) -> Result<&'static str> {
    Ok(match dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        other => candle_core::bail!("residual RMSNorm does not take {other:?}"),
    })
}

fn check(x: &Tensor, gain: &Tensor) -> Result<(usize, usize)> {
    let hidden = x.dim(D::Minus1)?;
    if hidden == 0 || gain.dtype() != DType::F32 || gain.elem_count() != hidden {
        candle_core::bail!(
            "invalid residual RMSNorm operands: x {:?} {:?}, gain {:?} {:?}",
            x.dims(),
            x.dtype(),
            gain.dims(),
            gain.dtype()
        )
    }
    dtype_tag(x.dtype())?;
    Ok((x.elem_count() / hidden, hidden))
}

/// `residual + delta` rounded to the residual's dtype, and its RMSNorm times
/// `gain` (F32, `[hidden]`) in `out`. `delta` has the residual's shape and any
/// float dtype.
pub fn add_rms_norm(
    residual: &Tensor,
    delta: &Tensor,
    gain: &Tensor,
    eps: f32,
    out: DType,
) -> Result<(Tensor, Tensor)> {
    let (rows, hidden) = check(residual, gain)?;
    if delta.dims() != residual.dims() || !delta.device().same_device(residual.device()) {
        candle_core::bail!(
            "residual RMSNorm delta {:?} does not match the residual {:?}",
            delta.dims(),
            residual.dims()
        )
    }
    dtype_tag(delta.dtype())?;
    dtype_tag(out)?;
    let (sum, normed) = run(residual, Some(delta), gain, eps, out, rows, hidden)?;
    Ok((sum.expect("a residual add returns its sum"), normed))
}

/// RMSNorm of `x` times `gain` in `out` (`x` cast to `out` first).
pub fn rms_norm(x: &Tensor, gain: &Tensor, eps: f32, out: DType) -> Result<Tensor> {
    let (rows, hidden) = check(x, gain)?;
    dtype_tag(out)?;
    Ok(run(x, None, gain, eps, out, rows, hidden)?.1)
}

fn run(
    residual: &Tensor,
    delta: Option<&Tensor>,
    gain: &Tensor,
    eps: f32,
    out: DType,
    rows: usize,
    hidden: usize,
) -> Result<(Option<Tensor>, Tensor)> {
    let residual = residual.contiguous()?;
    let delta = delta.map(Tensor::contiguous).transpose()?;
    let gain = gain.contiguous()?;
    match residual.device() {
        #[cfg(feature = "metal")]
        Device::Metal(_) => {
            if let Some(result) =
                metal_impl::run(&residual, delta.as_ref(), &gain, eps, out, rows, hidden)?
            {
                return Ok(result);
            }
            reference(&residual, delta.as_ref(), &gain, eps, out)
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => cuda_impl::run(&residual, delta.as_ref(), &gain, eps, out, rows, hidden),
        Device::Cpu => cpu(&residual, delta.as_ref(), &gain, eps, out, rows, hidden),
        #[allow(unreachable_patterns)]
        _ => reference(&residual, delta.as_ref(), &gain, eps, out),
    }
}

/// `v` rounded through `dtype`.
fn round(v: f32, dtype: DType) -> f32 {
    match dtype {
        DType::F16 => half::f16::from_f32(v).to_f32(),
        DType::BF16 => half::bf16::from_f32(v).to_f32(),
        _ => v,
    }
}

fn cpu(
    residual: &Tensor,
    delta: Option<&Tensor>,
    gain: &Tensor,
    eps: f32,
    out: DType,
    rows: usize,
    hidden: usize,
) -> Result<(Option<Tensor>, Tensor)> {
    use rayon::prelude::*;
    let r_dtype = residual.dtype();
    let r = residual
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let d = delta
        .map(|d| d.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>())
        .transpose()?;
    let g = gain.to_vec1::<f32>()?;
    let mut sum = vec![0f32; if d.is_some() { rows * hidden } else { 0 }];
    let mut normed = vec![0f32; rows * hidden];
    let row_sum = |row: usize, s: &mut [f32]| {
        for (i, v) in s.iter_mut().enumerate() {
            let at = row * hidden + i;
            *v = round(r[at] + d.as_ref().map_or(0.0, |d| d[at]), r_dtype);
        }
    };
    if d.is_some() {
        sum.par_chunks_mut(hidden)
            .enumerate()
            .for_each(|(row, s)| row_sum(row, s));
    }
    normed
        .par_chunks_mut(hidden)
        .enumerate()
        .for_each(|(row, n)| {
            let source = if d.is_some() {
                &sum[row * hidden..(row + 1) * hidden]
            } else {
                &r[row * hidden..(row + 1) * hidden]
            };
            let mut acc = 0f32;
            for (slot, &s) in n.iter_mut().zip(source) {
                *slot = round(s, out);
                acc += *slot * *slot;
            }
            let scale = 1.0 / (acc / hidden as f32 + eps).sqrt();
            for (slot, &w) in n.iter_mut().zip(&g) {
                *slot = round(*slot * scale * w, out);
            }
        });
    let shape = residual.shape().clone();
    let sum = d
        .is_some()
        .then(|| Tensor::from_vec(sum, shape.clone(), &Device::Cpu)?.to_dtype(r_dtype))
        .transpose()?;
    let normed = Tensor::from_vec(normed, shape, &Device::Cpu)?.to_dtype(out)?;
    Ok((sum, normed))
}

/// The eager chain on any device, for devices without a kernel and for tests.
pub fn reference(
    residual: &Tensor,
    delta: Option<&Tensor>,
    gain: &Tensor,
    eps: f32,
    out: DType,
) -> Result<(Option<Tensor>, Tensor)> {
    let sum = delta
        .map(|delta| {
            (residual.to_dtype(DType::F32)? + delta.to_dtype(DType::F32)?)?
                .to_dtype(residual.dtype())
        })
        .transpose()?;
    let source = sum.as_ref().unwrap_or(residual);
    let n = source.to_dtype(out)?.to_dtype(DType::F32)?;
    let normed = candle_nn::ops::rms_norm(&n, gain, eps)?.to_dtype(out)?;
    Ok((sum, normed))
}

#[cfg(feature = "metal")]
mod metal_impl {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    use candle_core::backend::BackendStorage;
    use candle_core::op::BackpropOp;
    use candle_core::{DType, MetalStorage, Result, Storage, Tensor};
    use candle_metal_kernels::metal::{ComputePipeline, Device as MetalDevice};

    use super::dtype_tag;
    use crate::kernels::metal_encoder::IzwiMetalCommandEncoderExt;

    const THREADS: usize = 256;

    /// The kernel for this dtype mix, or `None` when Metal has no form of it
    /// (BF16 runs the reference chain).
    pub(super) fn run(
        residual: &Tensor,
        delta: Option<&Tensor>,
        gain: &Tensor,
        eps: f32,
        out: DType,
        rows: usize,
        hidden: usize,
    ) -> Result<Option<(Option<Tensor>, Tensor)>> {
        let d_dtype = delta.map_or(residual.dtype(), Tensor::dtype);
        if [residual.dtype(), d_dtype, out].contains(&DType::BF16) {
            return Ok(None);
        }
        let name = format!(
            "izwi_residual_norm_{}_{}_{}",
            dtype_tag(residual.dtype())?,
            dtype_tag(d_dtype)?,
            dtype_tag(out)?
        );
        let (r_storage, r_layout) = residual.storage_and_layout();
        let (g_storage, g_layout) = gain.storage_and_layout();
        let (Storage::Metal(r_m), Storage::Metal(g_m)) = (&*r_storage, &*g_storage) else {
            candle_core::bail!("the Metal residual RMSNorm needs Metal tensors")
        };
        let device = r_m.device().clone();
        let Some(pipeline) = pipeline(device.metal_device(), &name)? else {
            return Ok(None);
        };
        let elements = rows * hidden;
        let r_size = residual.dtype().size_in_bytes();
        let sum = delta
            .map(|_| device.new_buffer(elements, residual.dtype(), "izwi-residual-sum"))
            .transpose()?;
        let normed = device.new_buffer(elements, out, "izwi-residual-normed")?;
        let delta_storage = delta.map(|d| (d.storage_and_layout(), d.dtype()));
        let encoder = device.command_encoder()?;
        encoder.set_label("izwi-residual-norm");
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_input_buffer(0, Some(r_m.buffer()), r_layout.start_offset() * r_size);
        match &delta_storage {
            Some(((d_storage, d_layout), d_dtype)) => {
                let Storage::Metal(d_m) = &**d_storage else {
                    candle_core::bail!("the Metal residual RMSNorm needs a Metal delta")
                };
                encoder.set_input_buffer(
                    1,
                    Some(d_m.buffer()),
                    d_layout.start_offset() * d_dtype.size_in_bytes(),
                );
            }
            None => encoder.set_input_buffer(1, Some(r_m.buffer()), 0),
        }
        encoder.set_input_buffer(2, Some(g_m.buffer()), g_layout.start_offset() * 4);
        encoder.set_output_buffer(3, Some(sum.as_ref().unwrap_or(&normed)), 0);
        encoder.set_output_buffer(4, Some(&normed), 0);
        let params: [u32; 3] = [hidden as u32, u32::from(delta.is_some()), eps.to_bits()];
        encoder.set_bytes(5, &params);
        encoder.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: rows,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: THREADS,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);
        let shape = residual.shape().clone();
        let wrap = |buffer, dtype| {
            Tensor::from_storage(
                Storage::Metal(MetalStorage::new(buffer, device.clone(), elements, dtype)),
                shape.clone(),
                BackpropOp::none(),
                false,
            )
        };
        Ok(Some((
            sum.map(|buffer| wrap(buffer, residual.dtype())),
            wrap(normed, out),
        )))
    }

    fn pipeline(device: &MetalDevice, function: &str) -> Result<Option<ComputePipeline>> {
        static PIPELINES: OnceLock<Mutex<HashMap<(u64, String), ComputePipeline>>> =
            OnceLock::new();
        let key = (device.registry_id(), function.to_string());
        let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(pipeline) = pipelines
            .lock()
            .map_err(|err| candle_core::Error::Msg(err.to_string()))?
            .get(&key)
            .cloned()
        {
            return Ok(Some(pipeline));
        }
        let library = device
            .new_library_with_source(SOURCE, None)
            .map_err(candle_core::Error::wrap)?;
        let Ok(function) = library.get_function(function, None) else {
            return Ok(None);
        };
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(candle_core::Error::wrap)?;
        pipelines
            .lock()
            .map_err(|err| candle_core::Error::Msg(err.to_string()))?
            .insert(key, pipeline.clone());
        Ok(Some(pipeline))
    }

    const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// One threadgroup of 256 per row: sum = R(r + d), n = O(sum),
// normed = O(n * rsqrt(mean(n * n) + eps) * gain); without a delta the input
// is normalized and nothing is summed. params: hidden, has_delta, eps bits.
#define IZWI_RESIDUAL_NORM(NAME, R, D, O)                                         \
kernel void izwi_residual_norm_##NAME(                                            \
    device const R* residual [[buffer(0)]],                                       \
    device const D* delta [[buffer(1)]],                                          \
    device const float* gain [[buffer(2)]],                                       \
    device R* sum [[buffer(3)]],                                                  \
    device O* out [[buffer(4)]],                                                  \
    constant uint* params [[buffer(5)]],                                          \
    uint tid [[thread_position_in_threadgroup]],                                  \
    uint row [[threadgroup_position_in_grid]]                                     \
) {                                                                               \
    threadgroup float partial[256];                                               \
    const uint hidden = params[0];                                                \
    const bool has_delta = params[1] != 0;                                        \
    const float eps = as_type<float>(params[2]);                                  \
    const ulong base = ulong(row) * hidden;                                       \
    float acc = 0.0f;                                                             \
    for (uint i = tid; i < hidden; i += 256) {                                    \
        float s = float(residual[base + i]);                                      \
        if (has_delta) {                                                          \
            s = float(R(s + float(delta[base + i])));                             \
            sum[base + i] = R(s);                                                 \
        }                                                                         \
        const float n = float(O(s));                                              \
        acc += n * n;                                                             \
    }                                                                             \
    partial[tid] = acc;                                                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);                              \
    for (uint stride = 128; stride > 0; stride >>= 1) {                           \
        if (tid < stride) {                                                       \
            partial[tid] += partial[tid + stride];                                \
        }                                                                         \
        threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    }                                                                             \
    const float scale = precise::rsqrt(partial[0] / float(hidden) + eps);         \
    for (uint i = tid; i < hidden; i += 256) {                                    \
        const float s = has_delta ? float(sum[base + i]) : float(residual[base + i]); \
        const float n = float(O(s));                                              \
        out[base + i] = O(n * scale * gain[i]);                                   \
    }                                                                             \
}

IZWI_RESIDUAL_NORM(f32_f32_f32, float, float, float)
IZWI_RESIDUAL_NORM(f32_f32_f16, float, float, half)
IZWI_RESIDUAL_NORM(f32_f16_f16, float, half, half)
IZWI_RESIDUAL_NORM(f16_f32_f16, half, float, half)
IZWI_RESIDUAL_NORM(f16_f16_f16, half, half, half)
"#;
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use candle_core::cuda_backend::cudarc::driver::{
        CudaSlice, CudaView, LaunchConfig, PushKernelArg,
    };
    use candle_core::cuda_backend::{CudaDType, CudaDevice, WrapErr};
    use candle_core::op::BackpropOp;
    use candle_core::{CudaStorage, DType, Result, Shape, Storage, Tensor};

    use super::dtype_tag;

    const MODULE: &str = "izwi_residual_norm";
    const THREADS: u32 = 256;

    fn view<'a, T: CudaDType>(storage: &'a Storage, tensor: &Tensor) -> Result<CudaView<'a, T>> {
        let Storage::Cuda(cuda) = storage else {
            candle_core::bail!("the CUDA residual RMSNorm needs CUDA tensors")
        };
        let Some((start, end)) = tensor.layout().contiguous_offsets() else {
            candle_core::bail!("the CUDA residual RMSNorm needs contiguous tensors")
        };
        Ok(cuda.as_cuda_slice::<T>()?.slice(start..end))
    }

    fn wrap<T: CudaDType>(slice: CudaSlice<T>, device: &CudaDevice, shape: Shape) -> Tensor {
        Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(slice, device.clone())),
            shape,
            BackpropOp::none(),
            false,
        )
    }

    /// The kernel name for the dtype mix, or the nearest exported one a plain
    /// norm (no delta) can use: its delta pointer is null.
    fn kernel(r: DType, d: DType, out: DType) -> Result<String> {
        let name = format!("{}_{}_{}", dtype_tag(r)?, dtype_tag(d)?, dtype_tag(out)?);
        let exported = [
            "f32_f32_f32",
            "f32_f32_f16",
            "f32_f16_f16",
            "f32_f32_bf16",
            "bf16_f32_bf16",
            "bf16_bf16_bf16",
            "f16_f32_f16",
            "f16_f16_f16",
        ];
        if !exported.contains(&name.as_str()) {
            candle_core::bail!("the CUDA residual RMSNorm has no {name} form")
        }
        Ok(format!("izwi_residual_norm_{name}"))
    }

    pub(super) fn run(
        residual: &Tensor,
        delta: Option<&Tensor>,
        gain: &Tensor,
        eps: f32,
        out: DType,
        rows: usize,
        hidden: usize,
    ) -> Result<(Option<Tensor>, Tensor)> {
        let device = residual.device().as_cuda_device()?;
        let r_dtype = residual.dtype();
        let d_dtype = delta.map_or(r_dtype, Tensor::dtype);
        let name = kernel(r_dtype, d_dtype, out)?;
        let function = device.get_or_load_custom_func(
            &name,
            MODULE,
            super::super::cuda::cuda_ptx_residual_norm(),
        )?;
        let elements = rows * hidden;
        let shape = residual.shape().clone();
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(rows)
                    .map_err(|_| candle_core::Error::Msg("residual RMSNorm grid".into()))?,
                1,
                1,
            ),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let (r_storage, _) = residual.storage_and_layout();
        let (g_storage, _) = gain.storage_and_layout();
        let g = view::<f32>(&g_storage, gain)?;
        let null = 0u64;
        macro_rules! go {
            ($r:ty, $d:ty, $o:ty) => {{
                let r = view::<$r>(&r_storage, residual)?;
                // SAFETY: the kernel writes every element it is given.
                let normed = unsafe { device.alloc::<$o>(elements)? };
                let sum = match delta {
                    Some(_) => Some(unsafe { device.alloc::<$r>(elements)? }),
                    None => None,
                };
                let delta_storage = delta.map(|d| d.storage_and_layout());
                let d_view = match (&delta_storage, delta) {
                    (Some((storage, _)), Some(d)) => Some(view::<$d>(storage, d)?),
                    _ => None,
                };
                let mut builder = function.builder();
                builder.arg(&r);
                match &d_view {
                    Some(d) => builder.arg(d),
                    None => builder.arg(&null),
                };
                builder.arg(&g);
                match &sum {
                    Some(sum) => builder.arg(sum),
                    None => builder.arg(&null),
                };
                builder.arg(&normed);
                candle_core::builder_arg!(builder, hidden as i32, eps);
                // SAFETY: argument order and types match the kernel.
                unsafe { builder.launch(config) }.w()?;
                (
                    sum.map(|sum| wrap(sum, device, shape.clone())),
                    wrap(normed, device, shape.clone()),
                )
            }};
        }
        use half::{bf16, f16};
        Ok(match (r_dtype, d_dtype, out) {
            (DType::F32, DType::F32, DType::F32) => go!(f32, f32, f32),
            (DType::F32, DType::F32, DType::F16) => go!(f32, f32, f16),
            (DType::F32, DType::F16, DType::F16) => go!(f32, f16, f16),
            (DType::F32, DType::F32, DType::BF16) => go!(f32, f32, bf16),
            (DType::BF16, DType::F32, DType::BF16) => go!(bf16, f32, bf16),
            (DType::BF16, DType::BF16, DType::BF16) => go!(bf16, bf16, bf16),
            (DType::F16, DType::F32, DType::F16) => go!(f16, f32, f16),
            (DType::F16, DType::F16, DType::F16) => go!(f16, f16, f16),
            (r, d, o) => {
                candle_core::bail!("the CUDA residual RMSNorm has no {r:?}/{d:?}/{o:?} form")
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::numeric_check::xorshift_values;

    fn tensor(values: Vec<f32>, shape: (usize, usize), dtype: DType, device: &Device) -> Tensor {
        Tensor::from_vec(values, shape, &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
            .to_device(device)
            .unwrap()
    }

    fn host(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    /// The dtype mixes the decoders run: Nemotron-H's F32 stream, Drama 3's
    /// BF16 stream with F32 and BF16 deltas (CUDA) and F32 stream with F16
    /// activations and F16 or F32 deltas (Metal).
    const MIXES: [(DType, DType, DType); 6] = [
        (DType::F32, DType::F32, DType::F32),
        (DType::F32, DType::F32, DType::F16),
        (DType::F32, DType::F16, DType::F16),
        (DType::BF16, DType::F32, DType::BF16),
        (DType::BF16, DType::BF16, DType::BF16),
        (DType::F16, DType::F16, DType::F16),
    ];

    fn operands(
        rows: usize,
        hidden: usize,
        (r, d, _): (DType, DType, DType),
        device: &Device,
    ) -> (Tensor, Tensor, Tensor) {
        let residual = tensor(
            xorshift_values(rows * hidden, 3, 4.0),
            (rows, hidden),
            r,
            device,
        );
        let delta = tensor(
            xorshift_values(rows * hidden, 5, 2.0),
            (rows, hidden),
            d,
            device,
        );
        let gain = Tensor::from_vec(
            xorshift_values(hidden, 7, 0.5)
                .iter()
                .map(|v| 1.0 + v)
                .collect::<Vec<_>>(),
            hidden,
            &Device::Cpu,
        )
        .unwrap()
        .to_device(device)
        .unwrap();
        (residual, delta, gain)
    }

    /// The CPU kernel against the eager chain: the sum bit for bit, the norm
    /// to rounding (the square sums run in another order).
    #[test]
    fn the_cpu_kernel_matches_the_eager_chain() {
        for mix in MIXES {
            for (rows, hidden) in [(1usize, 2688usize), (3, 2048), (2, 7)] {
                let (r, d, g) = operands(rows, hidden, mix, &Device::Cpu);
                let (sum, normed) = add_rms_norm(&r, &d, &g, 1e-5, mix.2).unwrap();
                let (ref_sum, ref_normed) = reference(&r, Some(&d), &g, 1e-5, mix.2).unwrap();
                assert_eq!(sum.dtype(), mix.0);
                assert_eq!(normed.dtype(), mix.2);
                assert_eq!(host(&sum), host(&ref_sum.unwrap()), "{mix:?} sum");
                let tolerance = if mix.2 == DType::F32 { 1e-5 } else { 1e-2 };
                let (a, e) = (host(&normed), host(&ref_normed));
                for (a, e) in a.iter().zip(&e) {
                    assert!(
                        (a - e).abs() <= tolerance * e.abs().max(1.0),
                        "{mix:?}: {a} {e}"
                    );
                }
                let plain = rms_norm(&r, &g, 1e-5, mix.2).unwrap();
                let (_, ref_plain) = reference(&r, None, &g, 1e-5, mix.2).unwrap();
                for (a, e) in host(&plain).iter().zip(&host(&ref_plain)) {
                    assert!(
                        (a - e).abs() <= tolerance * e.abs().max(1.0),
                        "{mix:?} plain"
                    );
                }
            }
        }
    }

    /// A row's output does not depend on how many rows share the call.
    #[test]
    fn a_row_is_the_same_alone_and_in_a_batch() {
        for mix in MIXES {
            let (r, d, g) = operands(5, 2688, mix, &Device::Cpu);
            let (sum, normed) = add_rms_norm(&r, &d, &g, 1e-5, mix.2).unwrap();
            let (one_sum, one) = add_rms_norm(
                &r.narrow(0, 3, 1).unwrap(),
                &d.narrow(0, 3, 1).unwrap(),
                &g,
                1e-5,
                mix.2,
            )
            .unwrap();
            assert_eq!(host(&normed.narrow(0, 3, 1).unwrap()), host(&one));
            assert_eq!(host(&sum.narrow(0, 3, 1).unwrap()), host(&one_sum));
        }
    }

    #[test]
    fn malformed_operands_are_refused() {
        let (r, d, g) = operands(2, 64, MIXES[0], &Device::Cpu);
        assert!(add_rms_norm(&r, &d.narrow(1, 0, 32).unwrap(), &g, 1e-5, DType::F32).is_err());
        assert!(rms_norm(&r, &g.narrow(0, 0, 32).unwrap(), 1e-5, DType::F32).is_err());
        assert!(rms_norm(&r, &g.to_dtype(DType::F16).unwrap(), 1e-5, DType::F32).is_err());
        assert!(rms_norm(&r, &g, 1e-5, DType::U32).is_err());
    }

    /// A GPU backend agrees with the CPU kernel: the sum bit for bit, the norm
    /// to rounding, and every row alone as in its batch.
    #[allow(dead_code)]
    fn device_matches_the_cpu(device: &Device, mixes: &[(DType, DType, DType)]) {
        for &mix in mixes {
            for (rows, hidden) in [(1usize, 2688usize), (4, 2048), (3, 300)] {
                let (r, d, g) = operands(rows, hidden, mix, &Device::Cpu);
                let (cpu_sum, cpu_normed) = add_rms_norm(&r, &d, &g, 1e-5, mix.2).unwrap();
                let on = |t: &Tensor| t.to_device(device).unwrap();
                let (sum, normed) = add_rms_norm(&on(&r), &on(&d), &on(&g), 1e-5, mix.2).unwrap();
                assert_eq!(host(&sum), host(&cpu_sum), "{mix:?} sum");
                let tolerance = if mix.2 == DType::F32 { 1e-5 } else { 1e-2 };
                for (a, e) in host(&normed).iter().zip(&host(&cpu_normed)) {
                    assert!(
                        (a - e).abs() <= tolerance * e.abs().max(1.0),
                        "{mix:?}: {a} {e}"
                    );
                }
                let plain = rms_norm(&on(&r), &on(&g), 1e-5, mix.2).unwrap();
                let cpu_plain = rms_norm(&r, &g, 1e-5, mix.2).unwrap();
                for (a, e) in host(&plain).iter().zip(&host(&cpu_plain)) {
                    assert!(
                        (a - e).abs() <= tolerance * e.abs().max(1.0),
                        "{mix:?} plain"
                    );
                }
                if rows > 1 {
                    let (_, one) = add_rms_norm(
                        &on(&r.narrow(0, 1, 1).unwrap()),
                        &on(&d.narrow(0, 1, 1).unwrap()),
                        &on(&g),
                        1e-5,
                        mix.2,
                    )
                    .unwrap();
                    assert_eq!(host(&normed.narrow(0, 1, 1).unwrap()), host(&one));
                }
            }
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_matches_the_cpu_kernel() {
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "residual_norm_metal_matches_the_cpu_kernel",
        ) else {
            return;
        };
        // Metal runs F32 and F16 activations; it has no BF16 form.
        let mixes = MIXES
            .into_iter()
            .filter(|mix| ![mix.0, mix.1, mix.2].contains(&DType::BF16))
            .collect::<Vec<_>>();
        device_matches_the_cpu(&device, &mixes);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_matches_the_cpu_kernel() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        device_matches_the_cpu(&device, &MIXES);
    }
}
