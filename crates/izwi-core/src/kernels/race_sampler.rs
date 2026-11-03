//! The exponential-race sampler (temperature, top-k, top-p) as one launch
//! per draw, on CPU, Metal and CUDA. Over candidates `first..vocab` of a
//! logits row:
//!
//! ```text
//! s_i    = logit_i / temperature
//! rank   = s descending, ties by candidate index (a stable sort)
//! p_r    = exp(s_r - max) / sum_i exp(s_i - max)
//! keep r = r == 0 || (r < cap && cumsum(p)[r] <= top_p)
//! pick   = the first kept rank maximising p_r / noise[candidate_r]
//! ```
//!
//! That is fish-speech's sampler: the noise holds one exponential per
//! candidate, so a recorded rollout replays whatever the sort does, and the
//! race draws from the kept distribution exactly. A device draw used to be
//! about 47 Candle ops (a bounded sort, masks, a cumulative-sum matmul over a
//! freshly built triangle, the race); here it is one launch per row: a radix
//! select finds the R-th largest `s` (`R <= 512` ranks), the candidates above
//! it and the lowest-index ties at it are bitonic-sorted, and one thread
//! walks the ranks. When all `R < candidates` ranks are kept and the cap
//! allows more, the kept prefix runs past them: a second radix descent finds
//! its end by mass (the largest `s` whose candidates at or above it hold more
//! than `top_p`), a count select finds the cap's, the ties at both bounds are
//! kept in candidate order, and the race runs over every kept candidate in
//! parallel. No draw needs the host.
//!
//! Sums run in F32 in a fixed order on every backend, so a row's draw depends
//! on that row alone and repeats bit for bit; against a host sampler that
//! sums in F64 it agrees except where rounding splits a near-tie.
use candle_core::{DType, Device, Result, Tensor};

/// The most ranks one draw sorts; a longer kept prefix is bounded by mass.
pub const MAX_RANKS: usize = 512;

/// One draw's policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RacePolicy {
    /// Already raised to the sampler's floor.
    pub temperature: f32,
    pub top_p: f32,
    /// `None` keeps every rank.
    pub top_k: Option<usize>,
}

impl RacePolicy {
    fn ranks(&self) -> usize {
        self.top_k.unwrap_or(MAX_RANKS).clamp(1, MAX_RANKS)
    }
}

/// One row's draw: the picked candidate (an index into the whole row) and
/// the row's non-finite logits (when non-zero the pick is meaningless).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RaceDraw {
    pub pick: u32,
    pub non_finite: u32,
}

/// Draws one candidate of each row of `logits` `[rows, vocab]` (F32) with
/// its row of `noise` (`[rows, vocab]` F32 exponentials, by candidate), over
/// candidates `first..vocab`. Returns `[rows, 2]` U32 `{pick, non_finite}`
/// on the logits' device, for one batched readback.
pub fn race_sample(
    logits: &Tensor,
    noise: &Tensor,
    first: usize,
    policy: RacePolicy,
) -> Result<Tensor> {
    let (rows, vocab) = logits.dims2()?;
    if rows == 0
        || first >= vocab
        || noise.dims() != [rows, vocab]
        || logits.dtype() != DType::F32
        || noise.dtype() != DType::F32
        || !noise.device().same_device(logits.device())
        || policy.temperature.is_nan()
        || policy.temperature <= 0.0
        || policy.top_p.is_nan()
        || policy.top_p <= 0.0
        || vocab > u32::MAX as usize
    {
        candle_core::bail!(
            "invalid race sample operands: logits {:?}, noise {:?}, first {first}, {policy:?}",
            logits.dims(),
            noise.dims()
        )
    }
    let (logits, noise) = (logits.contiguous()?, noise.contiguous()?);
    match logits.device() {
        #[cfg(feature = "metal")]
        Device::Metal(_) => metal_impl::run(&logits, &noise, first, policy),
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => cuda_impl::run(&logits, &noise, first, policy),
        _ => {
            let l = logits.flatten_all()?.to_vec1::<f32>()?;
            let n = noise.flatten_all()?.to_vec1::<f32>()?;
            let out = l
                .chunks_exact(vocab)
                .zip(n.chunks_exact(vocab))
                .flat_map(|(row, noise)| {
                    let d = draw_row(row, noise, first, policy);
                    [d.pick, d.non_finite]
                })
                .collect::<Vec<_>>();
            Tensor::from_vec(out, (rows, 2), &Device::Cpu)
        }
    }
}

/// The `[rows, 2]` result of [`race_sample`] as draws.
pub fn draws(packed: &[u32]) -> Vec<RaceDraw> {
    packed
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&[pick, non_finite]| RaceDraw { pick, non_finite })
        .collect()
}

/// One row on the host, in the kernels' arithmetic: the softmax total
/// summed in F32 in candidate order, ranks by a stable sort, the kept prefix
/// walked in F32 to its end (a device that bounds a long prefix by mass sums
/// it in another order, so it agrees except where rounding splits a
/// near-tie at the bound).
pub fn draw_row(row: &[f32], noise: &[f32], first: usize, policy: RacePolicy) -> RaceDraw {
    let non_finite = row.iter().filter(|v| !v.is_finite()).count() as u32;
    if non_finite > 0 {
        return RaceDraw {
            pick: first as u32,
            non_finite,
        };
    }
    let t = policy.temperature;
    let scaled = |i: usize| row[i] / t;
    let max = (first..row.len())
        .map(scaled)
        .fold(f32::NEG_INFINITY, f32::max);
    let total = (first..row.len())
        .map(|i| (scaled(i) - max).exp())
        .sum::<f32>();
    let inverse = 1.0 / total;
    let mut order = (first..row.len()).collect::<Vec<_>>();
    order.sort_by(|&a, &b| scaled(b).total_cmp(&scaled(a)));
    let limit = policy.top_k.unwrap_or(usize::MAX);
    let mut cumulative = 0f32;
    let mut kept = 0;
    for (r, &i) in order.iter().enumerate() {
        cumulative += (scaled(i) - max).exp() * inverse;
        if r > 0 && (r >= limit || cumulative > policy.top_p) {
            break;
        }
        kept += 1;
    }
    let mut best = (order[0], f32::NEG_INFINITY);
    for &i in &order[..kept] {
        let score = (scaled(i) - max).exp() * inverse / noise[i];
        if score > best.1 {
            best = (i, score);
        }
    }
    RaceDraw {
        pick: best.0 as u32,
        non_finite: 0,
    }
}

#[cfg(feature = "metal")]
mod metal_impl {
    use std::sync::OnceLock;

    use candle_core::backend::BackendStorage;
    use candle_core::op::BackpropOp;
    use candle_core::{DType, MetalStorage, Result, Shape, Storage, Tensor};
    use candle_metal_kernels::metal::ComputePipeline;

    use super::RacePolicy;
    use crate::kernels::metal_encoder::IzwiMetalCommandEncoderExt;

    pub(super) fn run(
        logits: &Tensor,
        noise: &Tensor,
        first: usize,
        policy: RacePolicy,
    ) -> Result<Tensor> {
        let (rows, vocab) = logits.dims2()?;
        let (l_storage, l_layout) = logits.storage_and_layout();
        let (n_storage, n_layout) = noise.storage_and_layout();
        let (Storage::Metal(l), Storage::Metal(n)) = (&*l_storage, &*n_storage) else {
            candle_core::bail!("the Metal race sampler needs Metal tensors")
        };
        let device = l.device().clone();
        static PIPELINE: OnceLock<std::result::Result<ComputePipeline, String>> = OnceLock::new();
        let pipeline = PIPELINE
            .get_or_init(|| {
                let metal = device.metal_device();
                let library = metal
                    .new_library_with_source(SOURCE, None)
                    .map_err(|e| e.to_string())?;
                let function = library
                    .get_function("izwi_race_sample", None)
                    .map_err(|e| e.to_string())?;
                metal
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(|e| e.to_string())
            })
            .clone()
            .map_err(candle_core::Error::Msg)?;
        let out = device.new_buffer(rows * 2, DType::U32, "izwi-race-sample")?;
        let params: [u32; 6] = [
            vocab as u32,
            first as u32,
            policy.temperature.to_bits(),
            policy.top_k.map_or(0, |k| k.min(u32::MAX as usize) as u32),
            policy.top_p.to_bits(),
            policy.ranks() as u32,
        ];
        let encoder = device.command_encoder()?;
        encoder.set_label("izwi-race-sample");
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_input_buffer(0, Some(l.buffer()), l_layout.start_offset() * 4);
        encoder.set_input_buffer(1, Some(n.buffer()), n_layout.start_offset() * 4);
        encoder.set_output_buffer(2, Some(&out), 0);
        encoder.set_bytes(3, &params);
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

    const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

static inline uint rs_key(float v) {
    const uint bits = as_type<uint>(v);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

static inline bool rs_before(uint ka, uint ia, uint kb, uint ib) {
    return ka > kb || (ka == kb && ia < ib);
}

// One threadgroup of 256 per row. params: vocab, first, temperature bits,
// cap (0 none), top_p bits, ranks. out[row] = {pick, non_finite}. A kept
// prefix past the sorted ranks is bounded by mass and by count, as the CUDA
// kernel's (`cuda/race_sampler.cu`).
kernel void izwi_race_sample(
    device const float* logits [[buffer(0)]],
    device const float* noise [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant uint* params [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row_index [[threadgroup_position_in_grid]]
) {
    threadgroup uint keys[512];
    threadgroup uint idx[512];
    threadgroup float f_scratch[256];
    threadgroup uint u_scratch[256];
    threadgroup atomic_uint slot;
    threadgroup int spill;
    const int vocab = int(params[0]), first = int(params[1]);
    const float temperature = as_type<float>(params[2]);
    const int cap = int(params[3]);
    const float top_p = as_type<float>(params[4]);
    const int ranks = int(params[5]);
    const ulong base = ulong(row_index) * vocab;
    device const float* row = logits + base;
    device const float* nz = noise + base;
    const int candidates = vocab - first;
    const int r_count = min(ranks, candidates);

    uint bad = 0;
    for (int i = int(tid); i < vocab; i += 256) {
        bad += isfinite(row[i]) ? 0u : 1u;
    }
    u_scratch[tid] = bad;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) { u_scratch[tid] += u_scratch[tid + s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint non_finite = u_scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (non_finite > 0 || candidates <= 0) {
        if (tid == 0) {
            out[2 * row_index] = uint(first);
            out[2 * row_index + 1] = non_finite;
        }
        return;
    }

    float m = -INFINITY;
    for (int i = first + int(tid); i < vocab; i += 256) {
        m = max(m, precise::divide(row[i], temperature));
    }
    f_scratch[tid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) { f_scratch[tid] = max(f_scratch[tid], f_scratch[tid + s]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_s = f_scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0.0f;
    for (int i = first + int(tid); i < vocab; i += 256) {
        acc += precise::exp(precise::divide(row[i], temperature) - max_s);
    }
    f_scratch[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) { f_scratch[tid] += f_scratch[tid + s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inverse = precise::divide(1.0f, f_scratch[0]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint prefix = 0;
    for (int bit = 31; bit >= 0; --bit) {
        const uint t = prefix | (1u << bit);
        uint count = 0;
        for (int i = first + int(tid); i < vocab; i += 256) {
            count += rs_key(precise::divide(row[i], temperature)) >= t ? 1u : 0u;
        }
        u_scratch[tid] = count;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 128; s > 0; s >>= 1) {
            if (tid < s) { u_scratch[tid] += u_scratch[tid + s]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (u_scratch[0] >= uint(r_count)) { prefix = t; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) { atomic_store_explicit(&slot, 0u, memory_order_relaxed); }
    for (uint i = tid; i < 512; i += 256) {
        keys[i] = 0u;
        idx[i] = 0xFFFFFFFFu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = first + int(tid); i < vocab; i += 256) {
        const uint k = rs_key(precise::divide(row[i], temperature));
        if (k > prefix) {
            const uint at = atomic_fetch_add_explicit(&slot, 1u, memory_order_relaxed);
            keys[at] = k;
            idx[at] = uint(i);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint above = atomic_load_explicit(&slot, memory_order_relaxed);
    const int chunk = (candidates + 255) / 256;
    const int lo = first + int(tid) * chunk;
    const int hi = min(vocab, lo + chunk);
    uint ties = 0;
    for (int i = lo; i < hi; ++i) {
        ties += rs_key(precise::divide(row[i], temperature)) == prefix ? 1u : 0u;
    }
    u_scratch[tid] = ties;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint running = 0;
        for (uint t = 0; t < 256; ++t) {
            const uint c = u_scratch[t];
            u_scratch[t] = running;
            running += c;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint at = above + u_scratch[tid];
    for (int i = lo; i < hi && at < uint(r_count); ++i) {
        const uint k = rs_key(precise::divide(row[i], temperature));
        if (k == prefix) {
            keys[at] = k;
            idx[at] = uint(i);
            ++at;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int width = 1;
    while (width < r_count) { width <<= 1; }
    for (int size = 2; size <= width; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = int(tid); i < width; i += 256) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool ascending = (i & size) == 0;
                    const bool swap = ascending ? rs_before(keys[j], idx[j], keys[i], idx[i])
                                                : rs_before(keys[i], idx[i], keys[j], idx[j]);
                    if (swap) {
                        const uint tk = keys[i], ti = idx[i];
                        keys[i] = keys[j]; idx[i] = idx[j];
                        keys[j] = tk; idx[j] = ti;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const int limit = cap > 0 ? cap : 0x7FFFFFFF;
    if (tid == 0) {
        float cumulative = 0.0f;
        int kept = 0;
        for (int r = 0; r < r_count; ++r) {
            cumulative += precise::exp(precise::divide(row[idx[r]], temperature) - max_s) * inverse;
            if (r > 0 && (r >= limit || cumulative > top_p)) { break; }
            ++kept;
        }
        spill = (kept == r_count && r_count < candidates && limit > r_count) ? 1 : 0;
        if (spill == 0) {
            uint best = idx[0];
            float best_score = -INFINITY;
            for (int r = 0; r < kept; ++r) {
                const float score = precise::divide(
                    precise::exp(precise::divide(row[idx[r]], temperature) - max_s) * inverse,
                    nz[idx[r]]);
                if (score > best_score) {
                    best_score = score;
                    best = idx[r];
                }
            }
            out[2 * row_index] = best;
            out[2 * row_index + 1] = 0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (spill == 0) { return; }

    uint cap_key = 0;
    uint cap_room = 0xFFFFFFFFu;
    if (limit < candidates) {
        for (int bit = 31; bit >= 0; --bit) {
            const uint t = cap_key | (1u << bit);
            uint count = 0;
            for (int i = first + int(tid); i < vocab; i += 256) {
                count += rs_key(precise::divide(row[i], temperature)) >= t ? 1u : 0u;
            }
            u_scratch[tid] = count;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) { u_scratch[tid] += u_scratch[tid + s]; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (u_scratch[0] >= uint(limit)) { cap_key = t; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        uint count = 0;
        for (int i = first + int(tid); i < vocab; i += 256) {
            count += rs_key(precise::divide(row[i], temperature)) > cap_key ? 1u : 0u;
        }
        u_scratch[tid] = count;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 128; s > 0; s >>= 1) {
            if (tid < s) { u_scratch[tid] += u_scratch[tid + s]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        cap_room = uint(limit) - u_scratch[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint mass_key = 0;
    for (int bit = 31; bit >= 0; --bit) {
        const uint t = mass_key | (1u << bit);
        float mass = 0.0f;
        for (int i = first + int(tid); i < vocab; i += 256) {
            const float v = precise::divide(row[i], temperature);
            if (rs_key(v) >= t) { mass += precise::exp(v - max_s) * inverse; }
        }
        f_scratch[tid] = mass;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 128; s > 0; s >>= 1) {
            if (tid < s) { f_scratch[tid] += f_scratch[tid + s]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (f_scratch[0] > top_p) { mass_key = t; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mass = 0.0f;
    for (int i = first + int(tid); i < vocab; i += 256) {
        const float v = precise::divide(row[i], temperature);
        if (rs_key(v) > mass_key) { mass += precise::exp(v - max_s) * inverse; }
    }
    f_scratch[tid] = mass;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) { f_scratch[tid] += f_scratch[tid + s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mass_above = f_scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint cap_ties = 0;
    float mass_ties = 0.0f;
    for (int i = lo; i < hi; ++i) {
        const float v = precise::divide(row[i], temperature);
        const uint k = rs_key(v);
        cap_ties += k == cap_key ? 1u : 0u;
        if (k == mass_key) { mass_ties += precise::exp(v - max_s) * inverse; }
    }
    u_scratch[tid] = cap_ties;
    f_scratch[tid] = mass_ties;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint running = 0;
        float total = mass_above;
        for (uint t = 0; t < 256; ++t) {
            const uint c = u_scratch[t];
            const float m = f_scratch[t];
            u_scratch[t] = running;
            f_scratch[t] = total;
            running += c;
            total += m;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint seen = u_scratch[tid];
    float cumulative = f_scratch[tid];
    float best_score = -INFINITY;
    uint best_key = 0, best = 0xFFFFFFFFu;
    for (int i = lo; i < hi; ++i) {
        const float v = precise::divide(row[i], temperature);
        const uint k = rs_key(v);
        const float p = precise::exp(v - max_s) * inverse;
        bool counted = k > cap_key;
        if (k == cap_key) { counted = seen < cap_room; ++seen; }
        bool massed = k > mass_key;
        if (k == mass_key) {
            cumulative += p;
            massed = cumulative <= top_p;
        }
        if (counted && massed) {
            const float score = precise::divide(p, nz[i]);
            if (score > best_score || (score == best_score && rs_before(k, uint(i), best_key, best))) {
                best_score = score;
                best_key = k;
                best = uint(i);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    f_scratch[tid] = best_score;
    keys[tid] = best_key;
    idx[tid] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            const float o = f_scratch[tid + s];
            if (o > f_scratch[tid]
                || (o == f_scratch[tid] && rs_before(keys[tid + s], idx[tid + s], keys[tid], idx[tid]))) {
                f_scratch[tid] = o;
                keys[tid] = keys[tid + s];
                idx[tid] = idx[tid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        out[2 * row_index] = idx[0];
        out[2 * row_index + 1] = 0;
    }
}
"#;
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::{CudaStorage, WrapErr};
    use candle_core::op::BackpropOp;
    use candle_core::{Result, Shape, Storage, Tensor};

    use super::RacePolicy;

    pub(super) fn run(
        logits: &Tensor,
        noise: &Tensor,
        first: usize,
        policy: RacePolicy,
    ) -> Result<Tensor> {
        let device = logits.device().as_cuda_device()?;
        let (rows, vocab) = logits.dims2()?;
        let (l_storage, l_layout) = logits.storage_and_layout();
        let (n_storage, n_layout) = noise.storage_and_layout();
        let (Storage::Cuda(l_cuda), Storage::Cuda(n_cuda)) = (&*l_storage, &*n_storage) else {
            candle_core::bail!("the CUDA race sampler needs CUDA tensors")
        };
        let (ls, ns) = (l_layout.start_offset(), n_layout.start_offset());
        let l = l_cuda.as_cuda_slice::<f32>()?.slice(ls..ls + rows * vocab);
        let n = n_cuda.as_cuda_slice::<f32>()?.slice(ns..ns + rows * vocab);
        // SAFETY: the kernel writes both words of every row.
        let out = unsafe { device.alloc::<u32>(rows * 2)? };
        let function = device.get_or_load_custom_func(
            "izwi_race_sample",
            "izwi_race_sampler",
            super::super::cuda::cuda_ptx_race_sampler(),
        )?;
        let mut builder = function.builder();
        builder.arg(&l);
        builder.arg(&n);
        builder.arg(&out);
        candle_core::builder_arg!(
            builder,
            vocab as i32,
            first as i32,
            policy.temperature,
            policy.top_k.map_or(0, |k| k.min(i32::MAX as usize) as i32),
            policy.top_p,
            policy.ranks() as i32
        );
        let config = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: argument order and types match the kernel.
        unsafe { builder.launch(config) }.w()?;
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

    pub(crate) fn exponentials(n: usize, seed: u64) -> Vec<f32> {
        xorshift_values(n, seed, 0.5)
            .into_iter()
            .map(|u| -((u + 0.5).clamp(1e-6, 1.0 - 1e-6)).ln())
            .collect()
    }

    /// Peaked rows, as a trained head's: a few large values over noise.
    pub(crate) fn peaked(width: usize, seed: u64) -> Vec<f32> {
        xorshift_values(width, 100 + seed, 4.0)
            .into_iter()
            .enumerate()
            .map(|(i, v)| if i % 97 == seed as usize { v + 9.0 } else { v })
            .collect()
    }

    pub(crate) const POLICIES: [RacePolicy; 4] = [
        RacePolicy {
            temperature: 1.0,
            top_p: 0.9,
            top_k: None,
        },
        RacePolicy {
            temperature: 1.0,
            top_p: 0.95,
            top_k: Some(30),
        },
        RacePolicy {
            temperature: 0.7,
            top_p: 0.7,
            top_k: Some(30),
        },
        RacePolicy {
            temperature: 1e-5,
            top_p: 1.0,
            top_k: Some(5),
        },
    ];

    /// The host form: ties ranked by candidate index, and a kept prefix
    /// walked to its end however long.
    #[test]
    fn the_host_form_ranks_ties_by_candidate_and_walks_the_whole_prefix() {
        let width = 4097;
        for seed in 0..6u64 {
            let row = peaked(width, seed);
            for policy in POLICIES {
                let noise = exponentials(width, seed + 3);
                let d = draw_row(&row, &noise, 1, policy);
                assert_eq!(d.non_finite, 0);
                assert!(d.pick >= 1);
            }
        }
        // All-equal logits: the first ranks are the lowest candidates, so a
        // greedy draw picks the first candidate.
        let flat = vec![0.5f32; width];
        let noise = vec![1.0f32; width];
        let greedy = RacePolicy {
            temperature: 1.0,
            top_p: 1.0,
            top_k: Some(1),
        };
        assert_eq!(draw_row(&flat, &noise, 1, greedy).pick, 1);
        // An uncapped flat row keeps candidates 1..=3686 under top_p 0.9.
        let mut late = noise.clone();
        late[3000] = 1e-3;
        late[4000] = 1e-4;
        assert_eq!(draw_row(&flat, &late, 1, POLICIES[0]).pick, 3000);
        let mut bad = flat.clone();
        bad[7] = f32::NAN;
        bad[9] = f32::INFINITY;
        assert_eq!(draw_row(&bad, &noise, 0, POLICIES[1]).non_finite, 2);
    }

    /// A device's draws equal the host form's, row by row, batched.
    #[allow(dead_code)]
    fn device_matches_the_host_form(device: &Device) {
        let width = 4097;
        let rows = 6;
        for (p, policy) in POLICIES.iter().enumerate() {
            for first in [0usize, 1] {
                let logits = (0..rows as u64)
                    .flat_map(|seed| peaked(width, seed + 7 * p as u64))
                    .collect::<Vec<_>>();
                let noise = (0..rows as u64)
                    .flat_map(|seed| exponentials(width, seed * 5 + p as u64))
                    .collect::<Vec<_>>();
                let expected = logits
                    .chunks_exact(width)
                    .zip(noise.chunks_exact(width))
                    .map(|(l, n)| draw_row(l, n, first, *policy))
                    .collect::<Vec<_>>();
                let got = race_sample(
                    &Tensor::from_vec(logits, (rows, width), device).unwrap(),
                    &Tensor::from_vec(noise, (rows, width), device).unwrap(),
                    first,
                    *policy,
                )
                .unwrap();
                let got = draws(&got.flatten_all().unwrap().to_vec1::<u32>().unwrap());
                assert_eq!(got, expected, "policy {p}, first {first}");
            }
        }
        // Kept prefixes past the sorted ranks, bounded by mass, by a cap
        // above them, and both: flat rows (every candidate tied), and rows
        // whose top 512 ranks hold too little mass.
        let spills = [
            POLICIES[0],
            RacePolicy {
                temperature: 1.0,
                top_p: 0.5,
                top_k: None,
            },
            RacePolicy {
                temperature: 1.0,
                top_p: 1.0,
                top_k: Some(1000),
            },
            RacePolicy {
                temperature: 1.3,
                top_p: 0.95,
                top_k: Some(600),
            },
        ];
        for (p, policy) in spills.iter().enumerate() {
            let rows = 4;
            let logits = (0..rows as u64)
                .flat_map(|seed| match seed {
                    0 => vec![0.5f32; width],
                    _ => xorshift_values(width, 40 + seed + 7 * p as u64, 0.5),
                })
                .collect::<Vec<_>>();
            let noise = exponentials(rows * width, 50 + p as u64);
            let expected = logits
                .chunks_exact(width)
                .zip(noise.chunks_exact(width))
                .map(|(l, n)| draw_row(l, n, 1, *policy))
                .collect::<Vec<_>>();
            let got = race_sample(
                &Tensor::from_vec(logits, (rows, width), device).unwrap(),
                &Tensor::from_vec(noise, (rows, width), device).unwrap(),
                1,
                *policy,
            )
            .unwrap();
            let got = draws(&got.flatten_all().unwrap().to_vec1::<u32>().unwrap());
            assert_eq!(got, expected, "spill policy {p}");
        }
        let mut bad = vec![0.5f32; width];
        bad[7] = f32::NAN;
        let got = race_sample(
            &Tensor::from_vec(bad, (1, width), device).unwrap(),
            &Tensor::from_vec(exponentials(width, 1), (1, width), device).unwrap(),
            0,
            POLICIES[1],
        )
        .unwrap();
        assert_eq!(
            draws(&got.flatten_all().unwrap().to_vec1::<u32>().unwrap())[0].non_finite,
            1
        );
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_draws_as_the_host_form() {
        let Some((device, _lease)) =
            crate::backends::metal_test_device_or_skip("metal_draws_as_the_host_form")
        else {
            return;
        };
        device_matches_the_host_form(&device);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_draws_as_the_host_form() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        device_matches_the_host_form(&device);
    }
}
