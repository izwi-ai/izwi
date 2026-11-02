//! Per-head q/k RMSNorm and partial RoPE from host-built cos/sin tables, read
//! straight from a packed attention projection, on CPU, Metal and CUDA.
//!
//! One launch replaces the chain a gated-attention layer runs per step:
//! contiguous copies of q, k and v out of the packed projection, a norm per
//! head of q and k (each a cast and a norm), and a RoPE of each (an F32 cast,
//! two table uploads, the rotation, the pass-through `cat` and a cast back).
//! With `d` the head width, `r` the rotated dims and `A` the activation dtype:
//!
//! ```text
//! n_i   = A(x_i * rsqrt(mean(x * x) + eps) * gain_i)
//! out_a = A(n_a * c - n_b * s),  out_b = A(n_a * s + n_b * c)   each pair (a, b) below r
//! out_i = n_i                                                   i >= r
//! ```
//!
//! The pairs are `(p, p + r / 2)` ([`RopeLayout::HalfSplit`], NeoX) or
//! `(2p, 2p + 1)` ([`RopeLayout::Interleaved`]); `(c, s)` is row `position`
//! of the tables, which the caller builds on the host (precise angles, never
//! device trigonometry) and uploads once. Values pass through `v` unchanged.
//! A row's bits depend on that row alone.
use candle_core::{DType, Device, Result, Tensor, D};

/// Which dims a RoPE pair joins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeLayout {
    /// `(p, p + r / 2)`: NeoX's rotate-half.
    HalfSplit,
    /// `(2p, 2p + 1)`.
    Interleaved,
}

/// Where the heads sit in a packed projection row, and how to rotate them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QkNormRopeSpec {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dims: usize,
    pub layout: RopeLayout,
    /// Column of query head 0, and the columns between query heads (`2 d`
    /// when each head's query is followed by its gate).
    pub q_offset: usize,
    pub q_stride: usize,
    /// Columns of key head 0 and value head 0 (heads `d` apart).
    pub k_offset: usize,
    pub v_offset: usize,
    pub eps: f32,
}

impl QkNormRopeSpec {
    fn validate(&self, width: usize) -> Result<()> {
        let d = self.head_dim;
        let ok = self.heads > 0
            && self.kv_heads > 0
            && d > 0
            && d <= 512
            && self.rotary_dims <= d
            && self.rotary_dims.is_multiple_of(2)
            && self.q_stride >= d
            && self.q_offset + (self.heads - 1) * self.q_stride + d <= width
            && self.k_offset + self.kv_heads * d <= width
            && self.v_offset + self.kv_heads * d <= width;
        if !ok {
            candle_core::bail!("invalid q/k norm + RoPE geometry {self:?} over {width} columns")
        }
        Ok(())
    }
}

/// `(q [n, heads, d], k [n, kv_heads, d], v [n, kv_heads, d])` in `packed`'s
/// dtype from `packed` `[n, width]`, the F32 gains `[d]`, the F32 tables
/// `[positions, r / 2]` and each row's position (`[n]` U32).
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope(
    packed: &Tensor,
    q_gain: &Tensor,
    k_gain: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: &Tensor,
    spec: QkNormRopeSpec,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (rows, width) = packed.dims2()?;
    spec.validate(width)?;
    let d = spec.head_dim;
    let half = spec.rotary_dims / 2;
    let device = packed.device();
    if !matches!(packed.dtype(), DType::F32 | DType::F16 | DType::BF16)
        || [q_gain, k_gain]
            .iter()
            .any(|g| g.dims() != [d] || g.dtype() != DType::F32)
        || [cos, sin].iter().any(|t| {
            t.dims().len() != 2
                || t.dim(D::Minus1).ok() != Some(half.max(1))
                || t.dtype() != DType::F32
        })
        || cos.dims() != sin.dims()
        || positions.dims() != [rows]
        || positions.dtype() != DType::U32
        || [q_gain, k_gain, cos, sin, positions]
            .iter()
            .any(|t| !t.device().same_device(device))
    {
        candle_core::bail!("invalid q/k norm + RoPE operands for {spec:?}")
    }
    let packed = packed.contiguous()?;
    let (q_gain, k_gain) = (q_gain.contiguous()?, k_gain.contiguous()?);
    let (cos, sin) = (cos.contiguous()?, sin.contiguous()?);
    let positions = positions.contiguous()?;
    match device {
        Device::Cpu => cpu(&packed, &q_gain, &k_gain, &cos, &sin, &positions, spec),
        #[cfg(feature = "metal")]
        Device::Metal(_) => {
            metal_impl::run(&packed, &q_gain, &k_gain, &cos, &sin, &positions, spec)
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => cuda_impl::run(&packed, &q_gain, &k_gain, &cos, &sin, &positions, spec),
        #[allow(unreachable_patterns)]
        _ => candle_core::bail!("q/k norm + RoPE has no kernel for this device"),
    }
}

fn round(v: f32, dtype: DType) -> f32 {
    match dtype {
        DType::F16 => half::f16::from_f32(v).to_f32(),
        DType::BF16 => half::bf16::from_f32(v).to_f32(),
        _ => v,
    }
}

/// The pair `(a, b)` dim `i` belongs to and its table column `p`.
fn pair(i: usize, half: usize, layout: RopeLayout) -> (usize, usize, usize) {
    match layout {
        RopeLayout::Interleaved => (2 * (i / 2), 2 * (i / 2) + 1, i / 2),
        RopeLayout::HalfSplit => {
            let p = if i < half { i } else { i - half };
            (p, p + half, p)
        }
    }
}

/// One head: `x` normalized with `gain`, then rotated at table row `row`.
fn head(
    x: &[f32],
    gain: &[f32],
    cos: &[f32],
    sin: &[f32],
    spec: &QkNormRopeSpec,
    dtype: DType,
    out: &mut [f32],
) {
    let d = spec.head_dim;
    let half = spec.rotary_dims / 2;
    let mut acc = 0f32;
    for v in x {
        acc += v * v;
    }
    let scale = 1.0 / (acc / d as f32 + spec.eps).sqrt();
    let normed = x
        .iter()
        .zip(gain)
        .map(|(v, w)| round(v * scale * w, dtype))
        .collect::<Vec<_>>();
    for i in 0..d {
        out[i] = if i < spec.rotary_dims {
            let (a, b, p) = pair(i, half, spec.layout);
            let (c, s) = (cos[p], sin[p]);
            let value = if i == a {
                normed[a] * c - normed[b] * s
            } else {
                normed[a] * s + normed[b] * c
            };
            round(value, dtype)
        } else {
            normed[i]
        };
    }
}

fn cpu(
    packed: &Tensor,
    q_gain: &Tensor,
    k_gain: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: &Tensor,
    spec: QkNormRopeSpec,
) -> Result<(Tensor, Tensor, Tensor)> {
    use rayon::prelude::*;
    let dtype = packed.dtype();
    let (rows, width) = packed.dims2()?;
    let x = packed
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let (qg, kg) = (q_gain.to_vec1::<f32>()?, k_gain.to_vec1::<f32>()?);
    let half = spec.rotary_dims / 2;
    let (cos, sin) = (
        cos.flatten_all()?.to_vec1::<f32>()?,
        sin.flatten_all()?.to_vec1::<f32>()?,
    );
    let positions = positions.to_vec1::<u32>()?;
    let d = spec.head_dim;
    let (h, kv) = (spec.heads, spec.kv_heads);
    let mut outs = vec![0f32; rows * (h + kv) * d];
    outs.par_chunks_mut(d).enumerate().for_each(|(index, out)| {
        let (row, j) = (index / (h + kv), index % (h + kv));
        let base = row * width;
        let (src, gain) = if j < h {
            (base + spec.q_offset + j * spec.q_stride, &qg)
        } else {
            (base + spec.k_offset + (j - h) * d, &kg)
        };
        let table = positions[row] as usize * half;
        let (c, s) = if half == 0 {
            (&cos[..0], &sin[..0])
        } else {
            (&cos[table..table + half], &sin[table..table + half])
        };
        head(&x[src..src + d], gain, c, s, &spec, dtype, out);
    });
    let mut q = Vec::with_capacity(rows * h * d);
    let mut k = Vec::with_capacity(rows * kv * d);
    let mut v = Vec::with_capacity(rows * kv * d);
    for row in 0..rows {
        let heads = &outs[row * (h + kv) * d..(row + 1) * (h + kv) * d];
        q.extend_from_slice(&heads[..h * d]);
        k.extend_from_slice(&heads[h * d..]);
        let vsrc = row * width + spec.v_offset;
        v.extend_from_slice(&x[vsrc..vsrc + kv * d]);
    }
    let make = |values: Vec<f32>, heads: usize| {
        Tensor::from_vec(values, (rows, heads, d), &Device::Cpu)?.to_dtype(dtype)
    };
    Ok((make(q, h)?, make(k, kv)?, make(v, kv)?))
}

#[cfg(feature = "metal")]
mod metal_impl {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    use candle_core::backend::BackendStorage;
    use candle_core::op::BackpropOp;
    use candle_core::{DType, MetalStorage, Result, Shape, Storage, Tensor};
    use candle_metal_kernels::metal::{ComputePipeline, Device as MetalDevice};

    use super::{QkNormRopeSpec, RopeLayout};
    use crate::kernels::metal_encoder::IzwiMetalCommandEncoderExt;

    const THREADS: usize = 128;

    pub(super) fn run(
        packed: &Tensor,
        q_gain: &Tensor,
        k_gain: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        positions: &Tensor,
        spec: QkNormRopeSpec,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let dtype = packed.dtype();
        let name = match dtype {
            DType::F32 => "izwi_qk_norm_rope_f32",
            DType::F16 => "izwi_qk_norm_rope_f16",
            // Metal runs F32 and F16 activations only.
            _ => candle_core::bail!("the Metal q/k norm + RoPE has no {dtype:?} form"),
        };
        let (rows, width) = packed.dims2()?;
        let storages =
            [packed, q_gain, k_gain, cos, sin, positions].map(|t| t.storage_and_layout());
        let mut buffers = Vec::with_capacity(6);
        for (tensor, (storage, layout)) in [packed, q_gain, k_gain, cos, sin, positions]
            .iter()
            .zip(&storages)
        {
            let Storage::Metal(m) = &**storage else {
                candle_core::bail!("the Metal q/k norm + RoPE needs Metal tensors")
            };
            buffers.push((m, layout.start_offset() * tensor.dtype().size_in_bytes()));
        }
        let device = buffers[0].0.device().clone();
        let pipeline = pipeline(device.metal_device(), name)?;
        let d = spec.head_dim;
        let (h, kv) = (spec.heads, spec.kv_heads);
        let q = device.new_buffer(rows * h * d, dtype, "izwi-qnr-q")?;
        let k = device.new_buffer(rows * kv * d, dtype, "izwi-qnr-k")?;
        let v = device.new_buffer(rows * kv * d, dtype, "izwi-qnr-v")?;
        let params: [u32; 12] = [
            width as u32,
            h as u32,
            kv as u32,
            d as u32,
            spec.rotary_dims as u32,
            u32::from(spec.layout == RopeLayout::Interleaved),
            spec.q_offset as u32,
            spec.q_stride as u32,
            spec.k_offset as u32,
            spec.v_offset as u32,
            (spec.rotary_dims / 2) as u32,
            spec.eps.to_bits(),
        ];
        let encoder = device.command_encoder()?;
        encoder.set_label("izwi-qk-norm-rope");
        encoder.set_compute_pipeline_state(&pipeline);
        for (index, (buffer, offset)) in buffers.iter().enumerate() {
            encoder.set_input_buffer(index, Some(buffer.buffer()), *offset);
        }
        encoder.set_output_buffer(6, Some(&q), 0);
        encoder.set_output_buffer(7, Some(&k), 0);
        encoder.set_output_buffer(8, Some(&v), 0);
        encoder.set_bytes(9, &params);
        encoder.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: rows,
                height: h + kv,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: THREADS,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);
        let wrap = |buffer, heads: usize| {
            Tensor::from_storage(
                Storage::Metal(MetalStorage::new(
                    buffer,
                    device.clone(),
                    rows * heads * d,
                    dtype,
                )),
                Shape::from((rows, heads, d)),
                BackpropOp::none(),
                false,
            )
        };
        Ok((wrap(q, h), wrap(k, kv), wrap(v, kv)))
    }

    fn pipeline(device: &MetalDevice, function: &str) -> Result<ComputePipeline> {
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
            return Ok(pipeline);
        }
        let library = device
            .new_library_with_source(SOURCE, None)
            .map_err(candle_core::Error::wrap)?;
        let function = library
            .get_function(function, None)
            .map_err(candle_core::Error::wrap)?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(candle_core::Error::wrap)?;
        pipelines
            .lock()
            .map_err(|err| candle_core::Error::Msg(err.to_string()))?
            .insert(key, pipeline.clone());
        Ok(pipeline)
    }

    const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// One threadgroup of 128 per (row, head); heads [0, H) are queries, the
// rest keys, and a key threadgroup also copies its value head. params:
// width, heads, kv_heads, d, rotary, interleaved, q_offset, q_stride,
// k_offset, v_offset, rotary / 2, eps bits.
#define IZWI_QK_NORM_ROPE(NAME, T)                                                \
kernel void izwi_qk_norm_rope_##NAME(                                             \
    device const T* packed [[buffer(0)]],                                         \
    device const float* q_gain [[buffer(1)]],                                     \
    device const float* k_gain [[buffer(2)]],                                     \
    device const float* cos_t [[buffer(3)]],                                      \
    device const float* sin_t [[buffer(4)]],                                      \
    device const uint* positions [[buffer(5)]],                                   \
    device T* q [[buffer(6)]],                                                    \
    device T* k [[buffer(7)]],                                                    \
    device T* v [[buffer(8)]],                                                    \
    constant uint* params [[buffer(9)]],                                          \
    uint2 local [[thread_position_in_threadgroup]],                               \
    uint2 group [[threadgroup_position_in_grid]]                                  \
) {                                                                               \
    const uint tid = local.x;                                                     \
    threadgroup float partial[128];                                               \
    threadgroup float normed[512];                                                \
    const uint width = params[0], heads = params[1], kv_heads = params[2];        \
    const uint d = params[3], rotary = params[4];                                 \
    const bool interleaved = params[5] != 0;                                      \
    const uint half_r = params[10];                                               \
    const float eps = as_type<float>(params[11]);                                 \
    const uint row = group.x, head = group.y;                                     \
    const bool is_q = head < heads;                                               \
    const uint kv = head - heads;                                                  \
    const ulong src = ulong(row) * width +                                        \
        (is_q ? params[6] + ulong(head) * params[7] : params[8] + ulong(kv) * d); \
    device const float* gain = is_q ? q_gain : k_gain;                            \
    device T* dst = is_q ? q + (ulong(row) * heads + head) * d                    \
                         : k + (ulong(row) * kv_heads + kv) * d;                  \
    float acc = 0.0f;                                                             \
    for (uint i = tid; i < d; i += 128) {                                         \
        const float x = float(packed[src + i]);                                   \
        acc += x * x;                                                             \
    }                                                                             \
    partial[tid] = acc;                                                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);                              \
    for (uint stride = 64; stride > 0; stride >>= 1) {                            \
        if (tid < stride) {                                                       \
            partial[tid] += partial[tid + stride];                                \
        }                                                                         \
        threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    }                                                                             \
    const float scale = precise::rsqrt(partial[0] / float(d) + eps);              \
    for (uint i = tid; i < d; i += 128) {                                         \
        normed[i] = float(T(float(packed[src + i]) * scale * gain[i]));           \
    }                                                                             \
    threadgroup_barrier(mem_flags::mem_threadgroup);                              \
    const ulong table = ulong(positions[row]) * half_r;                           \
    for (uint i = tid; i < d; i += 128) {                                         \
        float out = normed[i];                                                    \
        if (i < rotary) {                                                         \
            const uint p = interleaved ? i / 2 : (i < half_r ? i : i - half_r);   \
            const uint a = interleaved ? 2 * p : p;                               \
            const uint b = interleaved ? 2 * p + 1 : p + half_r;                  \
            const float c = cos_t[table + p], s = sin_t[table + p];               \
            out = (i == a) ? normed[a] * c - normed[b] * s                        \
                           : normed[a] * s + normed[b] * c;                       \
        }                                                                         \
        dst[i] = T(out);                                                          \
    }                                                                             \
    if (!is_q) {                                                                  \
        const ulong vsrc = ulong(row) * width + params[9] + ulong(kv) * d;        \
        device T* vdst = v + (ulong(row) * kv_heads + kv) * d;                    \
        for (uint i = tid; i < d; i += 128) {                                     \
            vdst[i] = packed[vsrc + i];                                           \
        }                                                                         \
    }                                                                             \
}

IZWI_QK_NORM_ROPE(f32, float)
IZWI_QK_NORM_ROPE(f16, half)
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

    use super::{QkNormRopeSpec, RopeLayout};

    const MODULE: &str = "izwi_qk_norm_rope";

    fn view<'a, T: CudaDType>(storage: &'a Storage, tensor: &Tensor) -> Result<CudaView<'a, T>> {
        let Storage::Cuda(cuda) = storage else {
            candle_core::bail!("the CUDA q/k norm + RoPE needs CUDA tensors")
        };
        let Some((start, end)) = tensor.layout().contiguous_offsets() else {
            candle_core::bail!("the CUDA q/k norm + RoPE needs contiguous tensors")
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

    pub(super) fn run(
        packed: &Tensor,
        q_gain: &Tensor,
        k_gain: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        positions: &Tensor,
        spec: QkNormRopeSpec,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let device = packed.device().as_cuda_device()?;
        let (rows, width) = packed.dims2()?;
        let d = spec.head_dim;
        let (h, kv) = (spec.heads, spec.kv_heads);
        let name = match packed.dtype() {
            DType::F32 => "izwi_qk_norm_rope_f32",
            DType::F16 => "izwi_qk_norm_rope_f16",
            DType::BF16 => "izwi_qk_norm_rope_bf16",
            other => candle_core::bail!("the CUDA q/k norm + RoPE has no {other:?} form"),
        };
        let function = device.get_or_load_custom_func(
            name,
            MODULE,
            super::super::cuda::cuda_ptx_qk_norm_rope(),
        )?;
        let storages = [q_gain, k_gain, cos, sin].map(|t| t.storage_and_layout());
        let views = [q_gain, k_gain, cos, sin]
            .iter()
            .zip(&storages)
            .map(|(t, (s, _))| view::<f32>(s, t))
            .collect::<Result<Vec<_>>>()?;
        let (p_storage, _) = positions.storage_and_layout();
        let p = view::<u32>(&p_storage, positions)?;
        let (x_storage, _) = packed.storage_and_layout();
        let config = LaunchConfig {
            grid_dim: (rows as u32, (h + kv) as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        macro_rules! go {
            ($t:ty) => {{
                let x = view::<$t>(&x_storage, packed)?;
                // SAFETY: the kernel writes every element of the outputs.
                let q = unsafe { device.alloc::<$t>(rows * h * d)? };
                let k = unsafe { device.alloc::<$t>(rows * kv * d)? };
                let v = unsafe { device.alloc::<$t>(rows * kv * d)? };
                let mut builder = function.builder();
                builder.arg(&x);
                for view in &views {
                    builder.arg(view);
                }
                builder.arg(&p);
                builder.arg(&q);
                builder.arg(&k);
                builder.arg(&v);
                candle_core::builder_arg!(
                    builder,
                    width as i32,
                    h as i32,
                    kv as i32,
                    d as i32,
                    spec.rotary_dims as i32,
                    i32::from(spec.layout == RopeLayout::Interleaved),
                    spec.q_offset as i32,
                    spec.q_stride as i32,
                    spec.k_offset as i32,
                    spec.v_offset as i32,
                    spec.eps
                );
                // SAFETY: argument order and types match the kernel.
                unsafe { builder.launch(config) }.w()?;
                (
                    wrap(q, device, Shape::from((rows, h, d))),
                    wrap(k, device, Shape::from((rows, kv, d))),
                    wrap(v, device, Shape::from((rows, kv, d))),
                )
            }};
        }
        Ok(match packed.dtype() {
            DType::F32 => go!(f32),
            DType::F16 => go!(half::f16),
            _ => go!(half::bf16),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::numeric_check::xorshift_values;

    /// Tables of `positions` rows for the rotated pairs, F64 angles rounded
    /// once to F32, as a family builds them.
    fn tables(positions: usize, rotary: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
        let half = rotary / 2;
        let mut cos = Vec::with_capacity(positions * half);
        let mut sin = Vec::with_capacity(positions * half);
        for position in 0..positions {
            for i in 0..half {
                let angle = position as f64 * theta.powf(-((2 * i) as f64) / rotary as f64);
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
        (cos, sin)
    }

    /// The gated-attention geometry: `[q_h | gate_h]` per head, then K, then
    /// V, a quarter of each head rotated.
    fn spec(layout: RopeLayout) -> QkNormRopeSpec {
        let (heads, kv_heads, d) = (4, 2, 256);
        QkNormRopeSpec {
            heads,
            kv_heads,
            head_dim: d,
            rotary_dims: 64,
            layout,
            q_offset: 0,
            q_stride: 2 * d,
            k_offset: 2 * heads * d,
            v_offset: 2 * heads * d + kv_heads * d,
            eps: 1e-6,
        }
    }

    struct Case {
        packed: Tensor,
        q_gain: Tensor,
        k_gain: Tensor,
        cos: Tensor,
        sin: Tensor,
        positions: Tensor,
    }

    fn case(spec: &QkNormRopeSpec, rows: usize, dtype: DType, device: &Device) -> Case {
        let width = spec.v_offset + spec.kv_heads * spec.head_dim;
        let d = spec.head_dim;
        let (cos, sin) = tables(4096, spec.rotary_dims, 1e7);
        let half = spec.rotary_dims / 2;
        let positions = (0..rows)
            .map(|r| (r * 997 + 3) as u32 % 4096)
            .collect::<Vec<_>>();
        let gain = |seed| {
            Tensor::from_vec(
                xorshift_values(d, seed, 0.3)
                    .iter()
                    .map(|v| 1.0 + v)
                    .collect::<Vec<_>>(),
                d,
                device,
            )
            .unwrap()
        };
        Case {
            packed: Tensor::from_vec(
                xorshift_values(rows * width, 11, 3.0),
                (rows, width),
                &Device::Cpu,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
            .to_device(device)
            .unwrap(),
            q_gain: gain(12),
            k_gain: gain(13),
            cos: Tensor::from_vec(cos, (4096, half), device).unwrap(),
            sin: Tensor::from_vec(sin, (4096, half), device).unwrap(),
            positions: Tensor::from_vec(positions, rows, device).unwrap(),
        }
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

    fn run(case: &Case, spec: QkNormRopeSpec) -> (Tensor, Tensor, Tensor) {
        qk_norm_rope(
            &case.packed,
            &case.q_gain,
            &case.k_gain,
            &case.cos,
            &case.sin,
            &case.positions,
            spec,
        )
        .unwrap()
    }

    /// The eager chain a gated-attention layer runs: copy the head out, norm it
    /// (rounded to the activation dtype), rotate it in F32 with Candle's RoPE
    /// over the table rows, and round once.
    fn eager(case: &Case, spec: QkNormRopeSpec) -> (Tensor, Tensor, Tensor) {
        let (rows, _) = case.packed.dims2().unwrap();
        let dtype = case.packed.dtype();
        let d = spec.head_dim;
        let (h, kv) = (spec.heads, spec.kv_heads);
        let packed = case.packed.to_device(&Device::Cpu).unwrap();
        let heads_of = |offset: usize, stride: usize, count: usize| {
            let parts = (0..count)
                .map(|j| packed.narrow(1, offset + j * stride, d).unwrap())
                .collect::<Vec<_>>();
            Tensor::stack(&parts, 1).unwrap()
        };
        let positions = case
            .positions
            .to_device(&Device::Cpu)
            .unwrap()
            .to_vec1::<u32>()
            .unwrap();
        let half = spec.rotary_dims / 2;
        let index = Tensor::from_vec(positions, rows, &Device::Cpu).unwrap();
        let cos = case
            .cos
            .to_device(&Device::Cpu)
            .unwrap()
            .index_select(&index, 0)
            .unwrap();
        let sin = case
            .sin
            .to_device(&Device::Cpu)
            .unwrap()
            .index_select(&index, 0)
            .unwrap();
        let normed = |x: Tensor, gain: &Tensor, count: usize| {
            let flat = x
                .reshape((rows * count, d))
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            candle_nn::ops::rms_norm(&flat, &gain.to_device(&Device::Cpu).unwrap(), spec.eps)
                .unwrap()
                .to_dtype(dtype)
                .unwrap()
                .reshape((rows, count, d))
                .unwrap()
        };
        let rope = |x: Tensor| {
            let xf = x.to_dtype(DType::F32).unwrap();
            let part = xf
                .narrow(2, 0, spec.rotary_dims)
                .unwrap()
                .contiguous()
                .unwrap();
            let rotated = match spec.layout {
                RopeLayout::HalfSplit => {
                    candle_nn::rotary_emb::rope_thd(&part.unsqueeze(0).unwrap(), &cos, &sin)
                        .unwrap()
                        .squeeze(0)
                        .unwrap()
                }
                RopeLayout::Interleaved => candle_nn::rotary_emb::rope_i(
                    &part
                        .transpose(0, 1)
                        .unwrap()
                        .unsqueeze(0)
                        .unwrap()
                        .contiguous()
                        .unwrap(),
                    &cos,
                    &sin,
                )
                .unwrap()
                .squeeze(0)
                .unwrap()
                .transpose(0, 1)
                .unwrap(),
            };
            Tensor::cat(
                &[
                    &rotated,
                    &xf.narrow(2, spec.rotary_dims, d - spec.rotary_dims)
                        .unwrap(),
                ],
                2,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
        };
        let _ = half;
        let q = rope(normed(
            heads_of(spec.q_offset, spec.q_stride, h),
            &case.q_gain,
            h,
        ));
        let k = rope(normed(heads_of(spec.k_offset, d, kv), &case.k_gain, kv));
        let v = heads_of(spec.v_offset, d, kv);
        (q, k, v)
    }

    fn close(label: &str, actual: &Tensor, expected: &Tensor, dtype: DType) {
        assert_eq!(actual.dims(), expected.dims(), "{label}");
        let tolerance = match dtype {
            DType::F32 => 1e-5,
            DType::F16 => 2e-3,
            _ => 1.6e-2,
        };
        for (i, (a, e)) in host(actual).iter().zip(&host(expected)).enumerate() {
            assert!(
                (a - e).abs() <= tolerance * e.abs().max(1.0),
                "{label} {dtype:?} [{i}]: {a} vs {e}"
            );
        }
    }

    #[test]
    fn the_cpu_kernel_matches_the_eager_chain() {
        for layout in [RopeLayout::HalfSplit, RopeLayout::Interleaved] {
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                let spec = spec(layout);
                let case = case(&spec, 3, dtype, &Device::Cpu);
                let (q, k, v) = run(&case, spec);
                let (eq, ek, ev) = eager(&case, spec);
                close("q", &q, &eq, dtype);
                close("k", &k, &ek, dtype);
                assert_eq!(host(&v), host(&ev), "v is copied bit for bit");
            }
        }
    }

    #[test]
    fn a_row_is_the_same_alone_and_in_a_batch() {
        let spec = spec(RopeLayout::HalfSplit);
        let case = case(&spec, 4, DType::BF16, &Device::Cpu);
        let (q, _, _) = run(&case, spec);
        let one = Case {
            packed: case.packed.narrow(0, 2, 1).unwrap(),
            positions: case.positions.narrow(0, 2, 1).unwrap(),
            q_gain: case.q_gain.clone(),
            k_gain: case.k_gain.clone(),
            cos: case.cos.clone(),
            sin: case.sin.clone(),
        };
        let (q1, _, _) = run(&one, spec);
        assert_eq!(host(&q.narrow(0, 2, 1).unwrap()), host(&q1));
    }

    #[test]
    fn malformed_geometry_is_refused() {
        let mut bad = spec(RopeLayout::HalfSplit);
        let case = case(&bad, 1, DType::F32, &Device::Cpu);
        bad.rotary_dims = 3;
        assert!(qk_norm_rope(
            &case.packed,
            &case.q_gain,
            &case.k_gain,
            &case.cos,
            &case.sin,
            &case.positions,
            bad
        )
        .is_err());
        bad = spec(RopeLayout::HalfSplit);
        bad.v_offset += 1;
        assert!(qk_norm_rope(
            &case.packed,
            &case.q_gain,
            &case.k_gain,
            &case.cos,
            &case.sin,
            &case.positions,
            bad
        )
        .is_err());
    }

    /// A GPU agrees with the CPU kernel: q and k to rounding, v bit for bit.
    #[allow(dead_code)]
    fn device_matches_the_cpu(device: &Device, dtypes: &[DType]) {
        for layout in [RopeLayout::HalfSplit, RopeLayout::Interleaved] {
            for &dtype in dtypes {
                let spec = spec(layout);
                let cpu = case(&spec, 3, dtype, &Device::Cpu);
                let gpu = case(&spec, 3, dtype, device);
                let (q, k, v) = run(&cpu, spec);
                let (gq, gk, gv) = run(&gpu, spec);
                close("q", &gq, &q, dtype);
                close("k", &gk, &k, dtype);
                assert_eq!(host(&gv), host(&v));
            }
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_matches_the_cpu_kernel() {
        let Some((device, _lease)) =
            crate::backends::metal_test_device_or_skip("qk_norm_rope_metal_matches_the_cpu_kernel")
        else {
            return;
        };
        device_matches_the_cpu(&device, &[DType::F32, DType::F16]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_matches_the_cpu_kernel() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        device_matches_the_cpu(&device, &[DType::F32, DType::F16, DType::BF16]);
    }
}
