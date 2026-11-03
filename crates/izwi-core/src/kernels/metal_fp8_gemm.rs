//! Tiled FP8-weight GEMMs for Metal spans (prefill).
//!
//! A span used to expand each FP8 weight matrix into a transient F16 or F32
//! matrix and then multiply: per matrix and span, a write and a read of the
//! expanded weights on top of the GEMM's own reads. These kernels read the
//! E4M3FN bytes straight into a threadgroup tile, decode them there, and
//! accumulate with simdgroup 8x8 F32 matrices, so the expanded weights never
//! exist in device memory.
//!
//! Two forms, both `out[m, n] = sum_k x[m, k] * W[n, k]` in F32:
//!
//! - [`block_gemm`]: one `[n, k]` matrix with one F32 scale per 128x128 block;
//!   each weight is `f32(f16(decode(b) * scale))`, exactly the value the F16
//!   expansion (`kernels::metal_fp8::dequantize_f16`) holds.
//! - [`grouped_gemm`]: stacked `[experts, n, k]` matrices with one scale per
//!   expert; rows of `x` are grouped expert-major and each group multiplies
//!   its own expert (`decode(b) * scale` in F32, as
//!   `kernels::moe_relu2::dequantize_expert` expands it). An optional relu²
//!   epilogue and per-row output scale fuse a sparse MoE's activation and
//!   combine weights.
//!
//! Each output element sums its products in a fixed order that depends only
//! on `k`, never on `m` or on the other rows, so a row's result does not
//! depend on the span it shares. That order differs from the expanded path's
//! GEMM, so the two agree to rounding, not bit for bit. `k` must be a
//! multiple of 32; `n` and `m` are free. The CPU references define the
//! expected values for the device tests.

use candle_core::{DType, Device, Result, Tensor};

use super::cuda::fp8::decode_e4m3fn;

/// Rows of `x` per threadgroup tile.
pub const TILE_ROWS: usize = 32;
/// Output columns per threadgroup tile.
pub const TILE_COLUMNS: usize = 64;
/// Reduction depth per tile step; `k` must be a multiple of it.
pub const TILE_DEPTH: usize = 32;

/// What [`grouped_gemm`] applies to each output before the row scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Epilogue {
    None,
    /// `max(v, 0)^2`.
    ReluSquare,
}

/// Whether the tiled kernels run a `[n, k]` weight matrix on `device`.
pub fn supported(device: &Device, n: usize, k: usize) -> bool {
    device.is_metal()
        && n > 0
        && k > 0
        && k.is_multiple_of(TILE_DEPTH)
        && u32::try_from(n.saturating_mul(k)).is_ok()
}

/// `x @ W.T` for F32 `x` `[m, k]` and block-scaled FP8 `w` `[n, k]` with
/// `scales` `[ceil(n/128), ceil(k/128)]` F32 → `[m, n]` F32. CPU tensors run
/// the reference.
pub fn block_gemm(x: &Tensor, w: &Tensor, scales: &Tensor) -> Result<Tensor> {
    let (m, k) = x.dims2()?;
    let (n, wk) = w.dims2()?;
    if wk != k
        || x.dtype() != DType::F32
        || w.dtype() != DType::U8
        || scales.dtype() != DType::F32
        || scales.dims() != [n.div_ceil(128), k.div_ceil(128)]
        || !x.device().same_device(w.device())
        || !w.device().same_device(scales.device())
    {
        candle_core::bail!(
            "invalid block-FP8 GEMM: x {:?} {:?}, w {:?} {:?}, scales {:?}",
            x.dims(),
            x.dtype(),
            w.dims(),
            w.dtype(),
            scales.dims()
        )
    }
    #[cfg(feature = "metal")]
    if x.device().is_metal() {
        if !supported(x.device(), n, k) {
            candle_core::bail!("block-FP8 GEMM needs k a multiple of {TILE_DEPTH}, got {k}")
        }
        return metal::block(&aligned(x)?, &aligned(w)?, &aligned(scales)?, m, n, k);
    }
    if !x.device().is_cpu() {
        candle_core::bail!("the tiled FP8 GEMMs run on Metal (or the CPU reference)")
    }
    let columns = k.div_ceil(128);
    let scales = scales.flatten_all()?.to_vec1::<f32>()?;
    let weights: Vec<f32> = w
        .flatten_all()?
        .to_vec1::<u8>()?
        .into_iter()
        .enumerate()
        .map(|(i, b)| {
            let scale = scales[(i / k / 128) * columns + (i % k) / 128];
            half::f16::from_f32(decode_e4m3fn(b) * scale).to_f32()
        })
        .collect();
    let out = reference(
        &x.flatten_all()?.to_vec1::<f32>()?,
        &weights,
        &[(0, 0, m)],
        n,
        k,
        Epilogue::None,
        None,
    );
    Tensor::from_vec(out, (m, n), x.device())
}

/// The rows of each expert in `x`, in order: `groups[i] = (expert, rows)`,
/// and the groups cover `x`'s rows exactly.
///
/// `x` `[rows, k]` F32, `w` `[experts, n, k]` FP8 bytes, `scales` `[experts]`
/// F32, `row_scale` `[rows]` F32 → `[rows, n]` F32:
/// `epilogue(x[r] @ W[expert(r)].T) * row_scale[r]`.
pub fn grouped_gemm(
    x: &Tensor,
    w: &Tensor,
    scales: &Tensor,
    groups: &[(usize, usize)],
    epilogue: Epilogue,
    row_scale: Option<&Tensor>,
) -> Result<Tensor> {
    let (rows, k) = x.dims2()?;
    let (experts, n, wk) = w.dims3()?;
    let covered = groups.iter().map(|&(_, count)| count).sum::<usize>();
    if wk != k
        || x.dtype() != DType::F32
        || w.dtype() != DType::U8
        || scales.dims() != [experts]
        || scales.dtype() != DType::F32
        || covered != rows
        || groups.iter().any(|&(expert, _)| expert >= experts)
        || row_scale.is_some_and(|s| s.dims() != [rows] || s.dtype() != DType::F32)
        || !x.device().same_device(w.device())
        || !w.device().same_device(scales.device())
    {
        candle_core::bail!(
            "invalid grouped FP8 GEMM: x {:?} {:?}, w {:?} {:?}, scales {:?}, {} groups over {covered} rows",
            x.dims(),
            x.dtype(),
            w.dims(),
            w.dtype(),
            scales.dims(),
            groups.len()
        )
    }
    let mut spans = Vec::with_capacity(groups.len());
    let mut at = 0;
    for &(expert, count) in groups {
        if count > 0 {
            spans.push((expert, at, at + count));
        }
        at += count;
    }
    #[cfg(feature = "metal")]
    if x.device().is_metal() {
        if !supported(x.device(), n, k) || u32::try_from(experts * n * k).is_err() {
            candle_core::bail!("grouped FP8 GEMM needs k a multiple of {TILE_DEPTH}, got {k}")
        }
        return metal::grouped(
            &aligned(x)?,
            &aligned(w)?,
            &aligned(scales)?,
            &spans,
            epilogue,
            row_scale.map(aligned).transpose()?.as_ref(),
            rows,
            n,
            k,
        );
    }
    if !x.device().is_cpu() {
        candle_core::bail!("the tiled FP8 GEMMs run on Metal (or the CPU reference)")
    }
    let scales = scales.to_vec1::<f32>()?;
    let bytes = w.flatten_all()?.to_vec1::<u8>()?;
    let weights: Vec<f32> = bytes
        .iter()
        .enumerate()
        .map(|(i, &b)| decode_e4m3fn(b) * scales[i / (n * k)])
        .collect();
    let row_scale = row_scale.map(Tensor::to_vec1::<f32>).transpose()?;
    let out = reference(
        &x.flatten_all()?.to_vec1::<f32>()?,
        &weights,
        &spans,
        n,
        k,
        epilogue,
        row_scale.as_deref(),
    );
    Tensor::from_vec(out, (rows, n), x.device())
}

/// `t` contiguous and starting on a 16-byte boundary, as the kernels' vector
/// loads need: a view that starts elsewhere is copied, never refused.
#[cfg(feature = "metal")]
fn aligned(t: &Tensor) -> Result<Tensor> {
    let t = t.contiguous()?;
    let offset = t.layout().start_offset() * t.dtype().size_in_bytes();
    if offset.is_multiple_of(16) {
        Ok(t)
    } else {
        // `copy` keeps the view's offset; this lands in a fresh buffer.
        t.force_contiguous()
    }
}

/// The CPU reference: `spans` of `(matrix, first row, end row)` over
/// expanded `weights` (`[matrices, n, k]`), each dot product in F32 in `k`
/// order.
fn reference(
    x: &[f32],
    weights: &[f32],
    spans: &[(usize, usize, usize)],
    n: usize,
    k: usize,
    epilogue: Epilogue,
    row_scale: Option<&[f32]>,
) -> Vec<f32> {
    let rows = spans.iter().map(|&(_, _, end)| end).max().unwrap_or(0);
    let mut out = vec![0f32; rows * n];
    for &(matrix, first, end) in spans {
        let w = &weights[matrix * n * k..(matrix + 1) * n * k];
        for r in first..end {
            let xr = &x[r * k..(r + 1) * k];
            for (c, wr) in w.chunks_exact(k).enumerate() {
                let mut v = xr
                    .iter()
                    .zip(wr)
                    .fold(0f32, |acc, (a, b)| a.mul_add(*b, acc));
                if epilogue == Epilogue::ReluSquare {
                    v = v.max(0.0);
                    v *= v;
                }
                if let Some(scale) = row_scale {
                    v *= scale[r];
                }
                out[r * n + c] = v;
            }
        }
    }
    out
}

#[cfg(feature = "metal")]
mod metal {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    use candle_core::op::BackpropOp;
    use candle_core::{bail, DType, MetalStorage, Result, Shape, Storage, Tensor};
    use candle_metal_kernels::metal::{Buffer, ComputePipeline, Device as MetalDevice};
    use objc2_metal::MTLSize;

    use super::{Epilogue, TILE_COLUMNS, TILE_ROWS};
    use crate::kernels::metal_encoder::IzwiMetalCommandEncoderExt;

    const SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint BM = 32;
constant constexpr uint BN = 64;
constant constexpr uint BK = 32;

// E4M3FN byte -> value * 2^-8 through an F16 lane (exact for finite bytes).
inline float fg_e4m3(uint b) {
    ushort h = ushort(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7));
    return float(as_type<half>(h));
}

struct FgParams {
    uint n;
    uint k;
    uint scale_columns;
    uint epilogue;
    uint row_scaled;
};

// One 32x64 output tile: rows [row0, row_end) of x against weight rows
// [n0, n0 + 64) of the matrix at byte offset `w_offset`. Four simdgroups,
// each owning 16 output columns of all 32 rows. BLOCK: 128x128 block
// scales and the F16 rounding of the dense expansion; otherwise one F32
// scale for the whole matrix.
template <bool BLOCK>
inline void fg_tile(
    device const float* x, device const uchar* w, device const float* scales,
    device float* out, device const float* row_scale, constant FgParams& p,
    uint row0, uint row_end, uint n0, ulong w_offset, float matrix_scale,
    threadgroup float* shared, uint tid, uint sg) {
    threadgroup float* As = shared;              // [BM][BK]
    threadgroup float* Bs = shared + BM * BK;    // [BK][BN]
    simdgroup_float8x8 acc[4][2];
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 2; ++j) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    const uint a_row = tid >> 2;
    const uint a_col = (tid & 3u) << 3;
    const uint b_row = tid >> 1;
    const uint b_col = (tid & 1u) << 4;
    const uint gm = row0 + a_row;
    const uint gn = n0 + b_row;
    for (uint k0 = 0; k0 < p.k; k0 += BK) {
        float4 a0 = float4(0.0f), a1 = float4(0.0f);
        if (gm < row_end) {
            device const float4* src = (device const float4*)(x + (ulong)gm * p.k + k0 + a_col);
            a0 = src[0];
            a1 = src[1];
        }
        threadgroup float4* a_dst = (threadgroup float4*)(As + a_row * BK + a_col);
        a_dst[0] = a0;
        a_dst[1] = a1;
        float values[16];
        if (gn < p.n) {
            const uint4 q = *(device const uint4*)(w + w_offset + (ulong)gn * p.k + k0 + b_col);
            const uint words[4] = {q.x, q.y, q.z, q.w};
            if (BLOCK) {
                const float scale =
                    scales[(gn >> 7) * p.scale_columns + ((k0 + b_col) >> 7)] * 256.0f;
                for (uint i = 0; i < 16; ++i) {
                    values[i] = float(half(fg_e4m3((words[i >> 2] >> ((i & 3u) * 8)) & 0xFFu) * scale));
                }
            } else {
                const float scale = matrix_scale * 256.0f;
                for (uint i = 0; i < 16; ++i) {
                    values[i] = fg_e4m3((words[i >> 2] >> ((i & 3u) * 8)) & 0xFFu) * scale;
                }
            }
        } else {
            for (uint i = 0; i < 16; ++i) {
                values[i] = 0.0f;
            }
        }
        for (uint i = 0; i < 16; ++i) {
            Bs[(b_col + i) * BN + b_row] = values[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 a[4];
            simdgroup_float8x8 b[2];
            for (uint i = 0; i < 4; ++i) {
                simdgroup_load(a[i], As + (i * 8) * BK + kk, BK);
            }
            for (uint j = 0; j < 2; ++j) {
                simdgroup_load(b[j], Bs + kk * BN + sg * 16 + j * 8, BN);
            }
            for (uint i = 0; i < 4; ++i) {
                for (uint j = 0; j < 2; ++j) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float* Cs = shared;              // [BM][BN], reusing As and Bs
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 2; ++j) {
            simdgroup_store(acc[i][j], Cs + (i * 8) * BN + sg * 16 + j * 8, BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < BM * BN; e += 128) {
        const uint r = row0 + e / BN;
        const uint c = n0 + e % BN;
        if (r < row_end && c < p.n) {
            float v = Cs[e];
            if (p.epilogue == 1u) {
                v = max(v, 0.0f);
                v = v * v;
            }
            if (p.row_scaled != 0u) {
                v = v * row_scale[r];
            }
            out[(ulong)r * p.n + c] = v;
        }
    }
}

// grid (ceil(n / 64), ceil(m / 32)) threadgroups of 128 threads.
kernel void izwi_fp8_gemm_block(
    device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],
    device const float* scales [[buffer(2)]], device float* out [[buffer(3)]],
    constant FgParams& p [[buffer(4)]], constant uint& m [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float shared[BM * BK + BK * BN];
    fg_tile<true>(x, w, scales, out, scales, p, tg.y * BM, m, tg.x * BN, 0, 1.0f,
                  shared, tid, sg);
}

// tiles: (matrix, first row, end row, unused) per 32-row tile; grid
// (ceil(n / 64), tiles) threadgroups of 128 threads.
kernel void izwi_fp8_gemm_grouped(
    device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],
    device const float* scales [[buffer(2)]], device float* out [[buffer(3)]],
    constant FgParams& p [[buffer(4)]], device const uint4* tiles [[buffer(5)]],
    device const float* row_scale [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float shared[BM * BK + BK * BN];
    const uint4 tile = tiles[tg.y];
    const ulong offset = (ulong)tile.x * p.n * p.k;
    fg_tile<false>(x, w, scales, out, row_scale, p, tile.y, tile.z, tg.x * BN, offset,
                   scales[tile.x], shared, tid, sg);
}
"#;

    fn pipeline(device: &MetalDevice, name: &'static str) -> Result<ComputePipeline> {
        static PIPELINES: OnceLock<Mutex<HashMap<(u64, &'static str), ComputePipeline>>> =
            OnceLock::new();
        let key = (device.registry_id(), name);
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
            .get_function(name, None)
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

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Params {
        n: u32,
        k: u32,
        scale_columns: u32,
        epilogue: u32,
        row_scaled: u32,
    }

    fn u32_of(value: usize, what: &str) -> Result<u32> {
        u32::try_from(value)
            .map_err(|_| candle_core::Error::Msg(format!("FP8 GEMM {what} {value} exceeds u32")))
    }

    /// A contiguous Metal tensor's buffer and byte offset.
    struct Bound<'a> {
        guard: candle_core::StorageRef<'a>,
        offset: usize,
    }

    impl Bound<'_> {
        fn buffer(&self) -> Result<&Buffer> {
            match &*self.guard {
                Storage::Metal(storage) => Ok(storage.buffer()),
                _ => bail!("the FP8 GEMM Metal kernels need Metal tensors"),
            }
        }
    }

    fn bind<'a>(tensor: &'a Tensor, name: &str) -> Result<Bound<'a>> {
        let (guard, layout) = tensor.storage_and_layout();
        if !layout.is_contiguous() {
            bail!("FP8 GEMM input {name} must be contiguous")
        }
        let offset = layout.start_offset() * tensor.dtype().size_in_bytes();
        if !offset.is_multiple_of(16) {
            bail!("FP8 GEMM input {name} must start on a 16-byte boundary")
        }
        Ok(Bound { guard, offset })
    }

    fn metal_device(tensor: &Tensor) -> Result<candle_core::MetalDevice> {
        match tensor.device() {
            candle_core::Device::Metal(device) => Ok(device.clone()),
            _ => bail!("the FP8 GEMM Metal kernels need Metal tensors"),
        }
    }

    fn wrap(device: &candle_core::MetalDevice, buffer: Arc<Buffer>, shape: Shape) -> Tensor {
        let count = shape.elem_count();
        Tensor::from_storage(
            Storage::Metal(MetalStorage::new(buffer, device.clone(), count, DType::F32)),
            shape,
            BackpropOp::none(),
            false,
        )
    }

    fn threadgroups(width: usize, height: usize) -> MTLSize {
        MTLSize {
            width,
            height,
            depth: 1,
        }
    }

    pub(super) fn block(
        x: &Tensor,
        w: &Tensor,
        scales: &Tensor,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Tensor> {
        let device = metal_device(x)?;
        let out = device.new_buffer(m * n, DType::F32, "izwi-fp8-gemm-block")?;
        if m == 0 {
            return Ok(wrap(&device, out, Shape::from((m, n))));
        }
        let params = Params {
            n: u32_of(n, "n")?,
            k: u32_of(k, "k")?,
            scale_columns: u32_of(k.div_ceil(128), "scale columns")?,
            epilogue: 0,
            row_scaled: 0,
        };
        let (xb, wb, sb) = (bind(x, "x")?, bind(w, "w")?, bind(scales, "scales")?);
        let encoder = device.command_encoder()?;
        encoder.set_label("izwi-fp8-gemm-block");
        encoder
            .set_compute_pipeline_state(&pipeline(device.metal_device(), "izwi_fp8_gemm_block")?);
        encoder.set_input_buffer(0, Some(xb.buffer()?), xb.offset);
        encoder.set_input_buffer(1, Some(wb.buffer()?), wb.offset);
        encoder.set_input_buffer(2, Some(sb.buffer()?), sb.offset);
        encoder.set_output_buffer(3, Some(&out), 0);
        encoder.set_bytes(4, &params);
        encoder.set_bytes(5, &u32_of(m, "m")?);
        encoder.dispatch_thread_groups(
            threadgroups(n.div_ceil(TILE_COLUMNS), m.div_ceil(TILE_ROWS)),
            threadgroups(128, 1),
        );
        drop(encoder);
        drop((xb, wb, sb));
        Ok(wrap(&device, out, Shape::from((m, n))))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn grouped(
        x: &Tensor,
        w: &Tensor,
        scales: &Tensor,
        spans: &[(usize, usize, usize)],
        epilogue: Epilogue,
        row_scale: Option<&Tensor>,
        rows: usize,
        n: usize,
        k: usize,
    ) -> Result<Tensor> {
        let device = metal_device(x)?;
        let out = device.new_buffer(rows * n, DType::F32, "izwi-fp8-gemm-grouped")?;
        let mut tiles = Vec::new();
        for &(matrix, first, end) in spans {
            for start in (first..end).step_by(TILE_ROWS) {
                tiles.extend([
                    u32_of(matrix, "expert")?,
                    u32_of(start, "row")?,
                    u32_of(end.min(start + TILE_ROWS), "row")?,
                    0,
                ]);
            }
        }
        if tiles.is_empty() {
            return Ok(wrap(&device, out, Shape::from((rows, n))));
        }
        let count = tiles.len() / 4;
        let tiles = Tensor::from_vec(tiles, (count, 4), x.device())?;
        let params = Params {
            n: u32_of(n, "n")?,
            k: u32_of(k, "k")?,
            scale_columns: 0,
            epilogue: match epilogue {
                Epilogue::None => 0,
                Epilogue::ReluSquare => 1,
            },
            row_scaled: u32::from(row_scale.is_some()),
        };
        let (xb, wb, sb, tb) = (
            bind(x, "x")?,
            bind(w, "w")?,
            bind(scales, "scales")?,
            bind(&tiles, "tiles")?,
        );
        let rb = row_scale.map(|s| bind(s, "row scale")).transpose()?;
        let encoder = device.command_encoder()?;
        encoder.set_label("izwi-fp8-gemm-grouped");
        encoder
            .set_compute_pipeline_state(&pipeline(device.metal_device(), "izwi_fp8_gemm_grouped")?);
        encoder.set_input_buffer(0, Some(xb.buffer()?), xb.offset);
        encoder.set_input_buffer(1, Some(wb.buffer()?), wb.offset);
        encoder.set_input_buffer(2, Some(sb.buffer()?), sb.offset);
        encoder.set_output_buffer(3, Some(&out), 0);
        encoder.set_bytes(4, &params);
        encoder.set_input_buffer(5, Some(tb.buffer()?), tb.offset);
        // An unscaled call binds the scales again: the kernel never reads it.
        match &rb {
            Some(rb) => encoder.set_input_buffer(6, Some(rb.buffer()?), rb.offset),
            None => encoder.set_input_buffer(6, Some(sb.buffer()?), sb.offset),
        }
        encoder.dispatch_thread_groups(
            threadgroups(n.div_ceil(TILE_COLUMNS), count),
            threadgroups(128, 1),
        );
        drop(encoder);
        drop((xb, wb, sb, tb, rb));
        Ok(wrap(&device, out, Shape::from((rows, n))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(count: usize, seed: u32) -> Vec<u8> {
        (0..count as u32)
            .map(
                |i| match ((i ^ seed).wrapping_mul(0x9E37_79B1) >> 13) as u8 {
                    // No NaN bytes (0x7f / 0xff).
                    b if b & 0x7f == 0x7f => b ^ 0x10,
                    b => b,
                },
            )
            .collect()
    }

    fn values(count: usize, seed: f32) -> Vec<f32> {
        (0..count)
            .map(|i| ((i as f32 + seed) * 0.618_034).sin() * (1.0 + (i % 5) as f32 * 0.2))
            .collect()
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    #[test]
    fn the_block_reference_multiplies_the_f16_expansion() {
        let (m, n, k) = (5usize, 200usize, 256usize);
        let w = Tensor::from_vec(bytes(n * k, 3), (n, k), &Device::Cpu).unwrap();
        let s = Tensor::from_vec(
            values(n.div_ceil(128) * k.div_ceil(128), 1.0)
                .into_iter()
                .map(|v| 1e-2 * (1.5 + v))
                .collect(),
            (n.div_ceil(128), k.div_ceil(128)),
            &Device::Cpu,
        )
        .unwrap();
        let x = Tensor::from_vec(values(m * k, 2.0), (m, k), &Device::Cpu).unwrap();
        let expanded = crate::kernels::metal_fp8::dequantize_f16(&w, &s)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let expected = flat(&x.matmul(&expanded.t().unwrap()).unwrap());
        let actual = flat(&block_gemm(&x, &w, &s).unwrap());
        for (a, e) in actual.iter().zip(&expected) {
            assert!((a - e).abs() <= 1e-4 * (1.0 + e.abs()), "{a} vs {e}");
        }
    }

    #[test]
    fn the_grouped_reference_applies_each_experts_scale_epilogue_and_row_scale() {
        let (experts, n, k) = (3usize, 40usize, 64usize);
        let w = Tensor::from_vec(bytes(experts * n * k, 7), (experts, n, k), &Device::Cpu).unwrap();
        let s = Tensor::from_vec(vec![0.01f32, 0.02, 0.005], experts, &Device::Cpu).unwrap();
        let x = Tensor::from_vec(values(6 * k, 4.0), (6, k), &Device::Cpu).unwrap();
        let rs = Tensor::from_vec(values(6, 9.0), 6, &Device::Cpu).unwrap();
        let groups = [(2usize, 2usize), (0, 3), (1, 1)];
        let out =
            flat(&grouped_gemm(&x, &w, &s, &groups, Epilogue::ReluSquare, Some(&rs)).unwrap());
        let (xs, rs) = (flat(&x), flat(&rs));
        let mut row = 0;
        for &(e, count) in &groups {
            let we = crate::kernels::moe_relu2::dequantize_expert(&w, &s, e, DType::F32).unwrap();
            let we = flat(&we);
            for _ in 0..count {
                for c in 0..n {
                    let mut v = 0f32;
                    for i in 0..k {
                        v = xs[row * k + i].mul_add(we[c * k + i], v);
                    }
                    let v = v.max(0.0).powi(2) * rs[row];
                    assert_eq!(out[row * n + c].to_bits(), v.to_bits(), "row {row} col {c}");
                }
                row += 1;
            }
        }
        assert!(grouped_gemm(&x, &w, &s, &[(0, 5)], Epilogue::None, None).is_err());
        assert!(grouped_gemm(&x, &w, &s, &[(3, 6)], Epilogue::None, None).is_err());
    }

    #[cfg(feature = "metal")]
    fn close(actual: &[f32], expected: &[f32], what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}");
        let scale = expected.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= 2e-5 * scale + 1e-5 * e.abs(),
                "{what}: element {i}: {a} vs {e} (scale {scale})"
            );
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_block_gemm_matches_the_reference_at_ragged_shapes() {
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "metal_block_gemm_matches_the_reference_at_ragged_shapes",
        ) else {
            return;
        };
        for (m, n, k) in [
            (1usize, 64usize, 32usize),
            (37, 200, 256),
            (64, 130, 416),
            (5, 2688, 1856),
        ] {
            let w = Tensor::from_vec(bytes(n * k, m as u32), (n, k), &Device::Cpu).unwrap();
            let s = Tensor::from_vec(
                values(n.div_ceil(128) * k.div_ceil(128), 3.0)
                    .into_iter()
                    .map(|v| 1e-2 * (1.5 + v))
                    .collect(),
                (n.div_ceil(128), k.div_ceil(128)),
                &Device::Cpu,
            )
            .unwrap();
            let x = Tensor::from_vec(values(m * k, 5.0), (m, k), &Device::Cpu).unwrap();
            let expected = flat(&block_gemm(&x, &w, &s).unwrap());
            let on = |t: &Tensor| t.to_device(&device).unwrap();
            let actual = block_gemm(&on(&x), &on(&w), &on(&s)).unwrap();
            close(&flat(&actual), &expected, &format!("[{m}, {n}, {k}]"));
            // A row's value does not depend on the rows beside it.
            let alone = block_gemm(&on(&x.narrow(0, m - 1, 1).unwrap()), &on(&w), &on(&s)).unwrap();
            assert_eq!(
                flat(&alone),
                flat(&actual.narrow(0, m - 1, 1).unwrap()),
                "[{m}, {n}, {k}] last row alone"
            );
        }
    }

    /// A view that starts off a 16-byte boundary is copied, not refused.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_block_gemm_takes_a_misaligned_view() {
        let Some((device, _lease)) =
            crate::backends::metal_test_device_or_skip("metal_block_gemm_takes_a_misaligned_view")
        else {
            return;
        };
        let (m, n, k) = (3usize, 64usize, 64usize);
        let w = Tensor::from_vec(bytes(n * k, 5), (n, k), &Device::Cpu).unwrap();
        let s = Tensor::from_vec(vec![1e-2f32; 1], (1, 1), &Device::Cpu).unwrap();
        // One leading value: the view starts 4 bytes in.
        let buffer = Tensor::from_vec(values(1 + m * k, 2.0), 1 + m * k, &Device::Cpu).unwrap();
        let x = buffer.narrow(0, 1, m * k).unwrap().reshape((m, k)).unwrap();
        let expected = flat(&block_gemm(&x, &w, &s).unwrap());
        let on = |t: &Tensor| t.to_device(&device).unwrap();
        let x = on(&buffer)
            .narrow(0, 1, m * k)
            .unwrap()
            .reshape((m, k))
            .unwrap();
        let actual = block_gemm(&x, &on(&w), &on(&s)).unwrap();
        close(&flat(&actual), &expected, "misaligned");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_grouped_gemm_matches_the_reference() {
        let Some((device, _lease)) =
            crate::backends::metal_test_device_or_skip("metal_grouped_gemm_matches_the_reference")
        else {
            return;
        };
        let (experts, n, k) = (5usize, 96usize, 128usize);
        let w =
            Tensor::from_vec(bytes(experts * n * k, 11), (experts, n, k), &Device::Cpu).unwrap();
        let s = Tensor::from_vec(
            vec![0.01f32, 0.02, 0.005, 0.03, 0.015],
            experts,
            &Device::Cpu,
        )
        .unwrap();
        let groups = [(4usize, 40usize), (0, 1), (2, 33), (3, 0), (1, 7)];
        let rows = groups.iter().map(|g| g.1).sum::<usize>();
        let x = Tensor::from_vec(values(rows * k, 6.0), (rows, k), &Device::Cpu).unwrap();
        let rs = Tensor::from_vec(values(rows, 8.0), rows, &Device::Cpu).unwrap();
        let on = |t: &Tensor| t.to_device(&device).unwrap();
        for (epilogue, row_scale) in [
            (Epilogue::None, None),
            (Epilogue::ReluSquare, None),
            (Epilogue::None, Some(&rs)),
        ] {
            let expected = flat(&grouped_gemm(&x, &w, &s, &groups, epilogue, row_scale).unwrap());
            let actual = grouped_gemm(
                &on(&x),
                &on(&w),
                &on(&s),
                &groups,
                epilogue,
                row_scale.map(on).as_ref(),
            )
            .unwrap();
            close(&flat(&actual), &expected, &format!("{epilogue:?}"));
        }
    }

    /// The tiled kernel against the expansion it replaces, at Nemotron's
    /// span shapes. `cargo test --release --features metal -- --ignored
    /// metal_fp8_gemm_timing --nocapture`.
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "timing on a Metal device"]
    fn metal_fp8_gemm_timing() {
        let Some((device, _lease)) =
            crate::backends::metal_test_device_or_skip("metal_fp8_gemm_timing")
        else {
            return;
        };
        let sync = |t: &Tensor| {
            t.sum_all().unwrap().to_vec0::<f32>().unwrap();
        };
        for (m, n, k) in [
            (512usize, 2688usize, 4096usize),
            (512, 10304, 2688),
            (512, 1856, 2688),
        ] {
            let w = Tensor::from_vec(bytes(n * k, 1), (n, k), &Device::Cpu)
                .unwrap()
                .to_device(&device)
                .unwrap();
            let s = Tensor::ones((n.div_ceil(128), k.div_ceil(128)), DType::F32, &device)
                .unwrap()
                .affine(1e-2, 0.0)
                .unwrap();
            let x = Tensor::from_vec(values(m * k, 1.0), (m, k), &Device::Cpu)
                .unwrap()
                .to_device(&device)
                .unwrap();
            let expanded = || {
                let w = crate::kernels::metal_fp8::dequantize_f16(&w, &s)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap();
                x.matmul(&w.t().unwrap()).unwrap()
            };
            let tiled = || block_gemm(&x, &w, &s).unwrap();
            for (name, run) in [
                ("expanded", &expanded as &dyn Fn() -> Tensor),
                ("tiled", &tiled),
            ] {
                sync(&run());
                let start = std::time::Instant::now();
                for _ in 0..5 {
                    sync(&run());
                }
                eprintln!(
                    "[{m}, {n}, {k}] {name}: {:.2} ms",
                    start.elapsed().as_secs_f64() * 1e3 / 5.0
                );
            }
        }
    }
}
