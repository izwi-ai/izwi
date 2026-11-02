//! Greedy picks of logits rows on the device: each row's largest value (ties
//! to the lowest id, NaN never picked) and its margin over the runner-up,
//! which counts a duplicate of the best (so a tie has margin 0).
//!
//! A decoder that samples greedily reads its whole logits row to the host
//! only to scan it; on a GPU this kernel scans it in place and the step reads
//! back 8 bytes a row. The merge compares and takes maxima only, so the pick
//! and margin are exactly the host scan's on every backend, whatever the
//! partition of the row.
use candle_core::{DType, Result, Tensor};

/// `(id, margin)` of each row of `logits` `[rows, vocab]` (F32, F16 or
/// BF16), as [`host_pick`] scans them.
pub fn greedy(logits: &Tensor) -> Result<Vec<(u32, f32)>> {
    Ok(unpack(
        &greedy_packed(logits)?.flatten_all()?.to_vec1::<u32>()?,
    ))
}

/// [`greedy`] left on the logits' device: `[rows, 2]` U32 `{id, margin
/// bits}` per row, for a caller that reads it back with other values (or
/// captures it in a CUDA graph). A CPU tensor on the CPU.
pub fn greedy_packed(logits: &Tensor) -> Result<Tensor> {
    let (rows, vocab) = logits.dims2()?;
    if rows == 0 || vocab == 0 || !matches!(logits.dtype(), DType::F32 | DType::F16 | DType::BF16) {
        candle_core::bail!(
            "greedy picks take [rows, vocab] float logits, got {:?} {:?}",
            logits.dims(),
            logits.dtype()
        )
    }
    let logits = logits.contiguous()?;
    match logits.device() {
        #[cfg(feature = "metal")]
        candle_core::Device::Metal(_) => metal_impl::run(&logits, rows, vocab),
        #[cfg(feature = "cuda")]
        candle_core::Device::Cuda(_) => cuda_impl::run(&logits, rows, vocab),
        _ => {
            let values = logits
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let packed = values
                .chunks_exact(vocab)
                .map(host_pick)
                .flat_map(|(id, margin)| [id, margin.to_bits()])
                .collect::<Vec<_>>();
            Tensor::from_vec(packed, (rows, 2), &candle_core::Device::Cpu)
        }
    }
}

/// The host values of a [`greedy_packed`] result as `(id, margin)` pairs.
pub fn unpack(packed: &[u32]) -> Vec<(u32, f32)> {
    packed
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&[id, margin]| (id, f32::from_bits(margin)))
        .collect()
}

/// The host scan: ties to the lowest id, and the runner-up counts a
/// duplicate of the best.
pub fn host_pick(row: &[f32]) -> (u32, f32) {
    let mut best = (0usize, f32::NEG_INFINITY);
    let mut second = f32::NEG_INFINITY;
    for (id, &v) in row.iter().enumerate() {
        if v > best.1 {
            second = best.1;
            best = (id, v);
        } else if v > second {
            second = v;
        }
    }
    (best.0 as u32, best.1 - second)
}

#[cfg(feature = "metal")]
mod metal_impl {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    use candle_core::backend::BackendStorage;
    use candle_core::op::BackpropOp;
    use candle_core::{DType, MetalStorage, Result, Shape, Storage, Tensor};
    use candle_metal_kernels::metal::{ComputePipeline, Device as MetalDevice};

    use crate::kernels::metal_encoder::IzwiMetalCommandEncoderExt;

    pub(super) fn run(logits: &Tensor, rows: usize, vocab: usize) -> Result<Tensor> {
        let name = match logits.dtype() {
            DType::F32 => "izwi_greedy_f32",
            DType::F16 => "izwi_greedy_f16",
            other => candle_core::bail!("the Metal greedy pick has no {other:?} form"),
        };
        let (storage, layout) = logits.storage_and_layout();
        let Storage::Metal(m) = &*storage else {
            candle_core::bail!("the Metal greedy pick needs a Metal tensor")
        };
        let device = m.device().clone();
        let pipeline = pipeline(device.metal_device(), name)?;
        let out = device.new_buffer(rows * 2, DType::U32, "izwi-greedy")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("izwi-greedy");
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_input_buffer(
            0,
            Some(m.buffer()),
            layout.start_offset() * logits.dtype().size_in_bytes(),
        );
        encoder.set_output_buffer(1, Some(&out), 0);
        encoder.set_bytes(2, &(vocab as u32));
        encoder.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: rows,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);
        Ok(Tensor::from_storage(
            Storage::Metal(MetalStorage::new(out, device.clone(), rows * 2, DType::U32)),
            Shape::from((rows, 2)),
            BackpropOp::none(),
            false,
        ))
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

#define GREEDY_NONE 0xFFFFFFFFu

struct GreedyTop { float best; uint id; float second; };

static inline GreedyTop greedy_merge(GreedyTop a, GreedyTop b) {
    const bool a_wins = b.id == GREEDY_NONE ||
        (a.id != GREEDY_NONE && (a.best > b.best || (a.best == b.best && a.id < b.id)));
    GreedyTop w = a_wins ? a : b;
    const GreedyTop l = a_wins ? b : a;
    const float runner = l.id == GREEDY_NONE ? l.second : max(l.best, l.second);
    w.second = max(w.second, runner);
    return w;
}

// One threadgroup of 256 per row; out[row] = {id, margin bits}.
#define IZWI_GREEDY(NAME, T)                                                      \
kernel void izwi_greedy_##NAME(                                                   \
    device const T* logits [[buffer(0)]],                                         \
    device uint* out [[buffer(1)]],                                               \
    constant uint& vocab [[buffer(2)]],                                           \
    uint tid [[thread_position_in_threadgroup]],                                  \
    uint row [[threadgroup_position_in_grid]]                                     \
) {                                                                               \
    threadgroup float s_best[256];                                                \
    threadgroup uint s_id[256];                                                   \
    threadgroup float s_second[256];                                              \
    const ulong base = ulong(row) * vocab;                                        \
    GreedyTop top = {-INFINITY, GREEDY_NONE, -INFINITY};                          \
    for (uint i = tid; i < vocab; i += 256) {                                     \
        const float v = float(logits[base + i]);                                  \
        if (top.id == GREEDY_NONE ? v > -INFINITY : v > top.best) {               \
            top.second = top.id == GREEDY_NONE ? top.second : top.best;           \
            top.best = v;                                                         \
            top.id = i;                                                           \
        } else if (v > top.second) {                                              \
            top.second = v;                                                       \
        }                                                                         \
    }                                                                             \
    s_best[tid] = top.best;                                                       \
    s_id[tid] = top.id;                                                           \
    s_second[tid] = top.second;                                                   \
    threadgroup_barrier(mem_flags::mem_threadgroup);                              \
    for (uint stride = 128; stride > 0; stride >>= 1) {                           \
        if (tid < stride) {                                                       \
            const GreedyTop a = {s_best[tid], s_id[tid], s_second[tid]};          \
            const GreedyTop b = {s_best[tid + stride], s_id[tid + stride],        \
                                 s_second[tid + stride]};                         \
            const GreedyTop m = greedy_merge(a, b);                               \
            s_best[tid] = m.best;                                                 \
            s_id[tid] = m.id;                                                     \
            s_second[tid] = m.second;                                             \
        }                                                                         \
        threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    }                                                                             \
    if (tid == 0) {                                                               \
        const bool none = s_id[0] == GREEDY_NONE;                                 \
        out[2 * row] = none ? 0u : s_id[0];                                       \
        out[2 * row + 1] = as_type<uint>(none ? NAN : s_best[0] - s_second[0]);   \
    }                                                                             \
}

IZWI_GREEDY(f32, float)
IZWI_GREEDY(f16, half)
"#;
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::{CudaStorage, WrapErr};
    use candle_core::op::BackpropOp;
    use candle_core::{DType, Result, Shape, Storage, Tensor};

    const MODULE: &str = "izwi_greedy";

    pub(super) fn run(logits: &Tensor, rows: usize, vocab: usize) -> Result<Tensor> {
        let device = logits.device().as_cuda_device()?;
        let (storage, _) = logits.storage_and_layout();
        let Storage::Cuda(cuda) = &*storage else {
            candle_core::bail!("the CUDA greedy pick needs a CUDA tensor")
        };
        let Some((start, end)) = logits.layout().contiguous_offsets() else {
            candle_core::bail!("the CUDA greedy pick needs contiguous logits")
        };
        // SAFETY: the kernel writes both words of every row.
        let out = unsafe { device.alloc::<u32>(rows * 2)? };
        let config = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        macro_rules! go {
            ($t:ty, $name:literal) => {{
                let function = device.get_or_load_custom_func(
                    $name,
                    MODULE,
                    super::super::cuda::cuda_ptx_greedy(),
                )?;
                let view = cuda.as_cuda_slice::<$t>()?.slice(start..end);
                let mut builder = function.builder();
                builder.arg(&view);
                builder.arg(&out);
                candle_core::builder_arg!(builder, vocab as i32);
                // SAFETY: argument order and types match the kernel.
                unsafe { builder.launch(config) }.w()?;
            }};
        }
        match logits.dtype() {
            DType::F32 => go!(f32, "izwi_greedy_f32"),
            DType::F16 => go!(half::f16, "izwi_greedy_f16"),
            DType::BF16 => go!(half::bf16, "izwi_greedy_bf16"),
            other => candle_core::bail!("the CUDA greedy pick has no {other:?} form"),
        }
        Ok(Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(out, device.clone())),
            Shape::from((rows, 2)),
            BackpropOp::none(),
            false,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::numeric_check::xorshift_values;
    use candle_core::Device;

    /// Rows with the cases that matter: a clear winner, a duplicated best
    /// (margin 0, lowest id wins), NaNs, and a near tie.
    fn rows(vocab: usize) -> Vec<Vec<f32>> {
        let mut clear = xorshift_values(vocab, 3, 5.0);
        clear[vocab - 7] = 40.0;
        let mut tied = xorshift_values(vocab, 4, 5.0);
        tied[vocab / 3] = 30.0;
        tied[vocab / 2] = 30.0;
        let mut nan = xorshift_values(vocab, 5, 5.0);
        nan[0] = f32::NAN;
        nan[11] = 25.0;
        nan[12] = f32::NAN;
        let mut near = xorshift_values(vocab, 6, 5.0);
        near[100] = 20.0;
        near[vocab - 1] = 20.000_002;
        vec![clear, tied, nan, near]
    }

    fn picks(device: &Device, dtype: DType, vocab: usize) -> (Vec<(u32, f32)>, Vec<(u32, f32)>) {
        let rows = rows(vocab);
        let flat = rows.concat();
        let logits = Tensor::from_vec(flat, (rows.len(), vocab), &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let host_rows = logits
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expected = host_rows.chunks_exact(vocab).map(host_pick).collect();
        let got = greedy(&logits.to_device(device).unwrap()).unwrap();
        (got, expected)
    }

    fn same(got: &[(u32, f32)], expected: &[(u32, f32)]) {
        for (g, e) in got.iter().zip(expected) {
            assert_eq!(g.0, e.0);
            assert_eq!(g.1.to_bits(), e.1.to_bits(), "{g:?} vs {e:?}");
        }
    }

    #[test]
    fn the_host_scan_ties_to_the_lowest_id_and_counts_duplicates() {
        let (got, _) = picks(&Device::Cpu, DType::F32, 1000);
        assert_eq!(got[0].0, 993);
        assert_eq!(got[1], (333, 0.0));
        assert_eq!(got[2].0, 11);
        assert_eq!(got[3].0, 999);
        assert!(got[3].1 > 0.0 && got[3].1 < 1e-4);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_picks_exactly_as_the_host_scan() {
        let Some((device, _lease)) =
            crate::backends::metal_test_device_or_skip("metal_picks_exactly_as_the_host_scan")
        else {
            return;
        };
        for dtype in [DType::F32, DType::F16] {
            for vocab in [1000, 131_072] {
                let (got, expected) = picks(&device, dtype, vocab);
                same(&got, &expected);
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_picks_exactly_as_the_host_scan() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for vocab in [1000, 131_072] {
                let (got, expected) = picks(&device, dtype, vocab);
                same(&got, &expected);
            }
        }
    }
}
