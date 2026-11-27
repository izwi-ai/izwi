//! Paged decode attention with an additive position bias and a sink logit,
//! read straight from a paged KV arena.
//!
//! A query at the end of its context scores the key at backward distance `d`
//! (0 is the query's own key) as
//!
//! ```text
//! dot(q[h], k[t][g]) * scale + (d < extent ? profile[h][d] : 0)
//! ```
//!
//! over the keys with `d < min(context, window)`, where query head `h` reads
//! KV head `g = h / (heads / kv_heads)`. One extra logit per (query, head),
//! the sink, joins the softmax and carries no value, so the output is the
//! weighted values of the visible keys with weights that sum to less than
//! one. Queries, profiles, sinks and the output are F32; keys and values stay
//! in the arena's storage dtype and its `[page, token, kv_head, dim]` layout.
//!
//! The bias is zero past `extent`, not clamped to its last entry, and a key
//! outside the window is skipped rather than given a score of minus
//! infinity. The two agree: a skipped key has exactly zero weight either way,
//! and the sink keeps the softmax finite when no key is visible.
//!
//! Block tables are packed as `[context[T] | first_page_offset[T] |
//! table[T x max_blocks]]`, the layout the arenas' decode plans already hold
//! on the device. Each table serves `queries_per_table` consecutive rows: the
//! last sees the table's whole context, the one before it one token less,
//! and so on. One query per table is a batch of decode rows; one table with
//! `seq` queries is a prefill span of one sequence.
//!
//! # Arithmetic
//!
//! [`reference`] is the kernels' arithmetic on the host, operation for
//! operation, and the CPU arena's implementation:
//!
//! - `lanes` lanes (32, or 16 for wide value rows, see [`lanes`]). Lane `l`
//!   owns the keys at distances `l, l + lanes, ...`, nearest first.
//! - A dot product runs eight chains of fused multiply-adds (dim `i` joins
//!   chain `i % 8`), summed `((c0 + c1) + (c2 + c3)) + ((c4 + c5) + (c6 +
//!   c7))`. The score is `fma(dot, scale, bias)`.
//! - Each lane keeps an online softmax over its own keys: a running maximum,
//!   a running sum and one accumulator per value dim, all rescaled by
//!   [`det_exp`] of the change in maximum.
//! - The merge starts from the sink (maximum = sink, sum = 1 before
//!   rescaling, no value) and folds the lanes in lane order, then divides.
//!
//! A row's arithmetic depends on its own context, window and extent only, so
//! a row has the same bits alone, in a batch and as a prefill row, and
//! whatever its table's first page is. Against a dense evaluation in another
//! order (cuBLAS, Candle's softmax) it agrees within [`TOLERANCE`].

use candle_core::{CpuStorage, DType, Device, Storage, Tensor};
use rayon::prelude::*;

use super::relpos_attention::det_exp;
use crate::error::{Error, Result};

/// The most lanes a kernel runs, `PBS_MAX_LANES` in the sources.
pub const MAX_LANES: usize = 32;
/// The widest value row the kernels take.
pub const MAX_VALUE_DIM: usize = 496;
/// The bound on `|fused - dense| / max |dense|` the tests and the self-check
/// hold every backend to, for a dense evaluation of the same stored keys and
/// values.
pub const TOLERANCE: f64 = 1e-5;

/// Lanes for a value row of `value_dim`: 32 while the lane states fit a
/// Metal thread group's 32 KiB of shared memory, 16 for wider rows, `None`
/// past [`MAX_VALUE_DIM`]. Every backend uses the same count, so the
/// summation order does not depend on the device.
pub fn lanes(value_dim: usize) -> Option<usize> {
    match value_dim {
        0 => None,
        1..=240 => Some(32),
        241..=MAX_VALUE_DIM => Some(16),
        _ => None,
    }
}

/// The shape of one call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geometry {
    /// Queries: `tables * queries_per_table`.
    pub rows: usize,
    pub queries_per_table: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    /// Entries of a bias profile.
    pub extent: usize,
    pub page_tokens: usize,
    /// Width of the packed block table.
    pub max_blocks: usize,
    /// Pages the key and value storage holds.
    pub capacity_pages: usize,
    /// The window in tokens, the query's own included.
    pub window: Option<u32>,
    pub scale: f32,
}

fn invalid(message: impl std::fmt::Display) -> Error {
    Error::InferenceError(format!("paged bias/sink attention: {message}"))
}

impl Geometry {
    pub fn tables(&self) -> usize {
        self.rows / self.queries_per_table.max(1)
    }

    pub fn metadata_len(&self) -> usize {
        self.tables() * (2 + self.max_blocks)
    }

    /// The window as the kernels take it: zero for none.
    #[cfg(any(feature = "cuda", feature = "metal"))]
    fn window_arg(&self) -> u32 {
        self.window.unwrap_or(0)
    }

    /// Checks the shape and every table of `metadata`: each context covers
    /// its rows, starts inside its first page and names resident pages only.
    pub fn validate(&self, metadata: &[u32]) -> Result<usize> {
        let lanes = lanes(self.value_dim).ok_or_else(|| {
            invalid(format!(
                "value head dim {} is outside 1..={MAX_VALUE_DIM}",
                self.value_dim
            ))
        })?;
        if self.rows == 0
            || self.queries_per_table == 0
            || !self.rows.is_multiple_of(self.queries_per_table)
            || self.heads == 0
            || self.kv_heads == 0
            || !self.heads.is_multiple_of(self.kv_heads)
            || self.key_dim == 0
            || self.extent == 0
            || self.page_tokens == 0
            || self.max_blocks == 0
            || self.capacity_pages == 0
            || self.window == Some(0)
            || !self.scale.is_finite()
        {
            return Err(invalid(format!("unsupported shape {self:?}")));
        }
        // The kernels take their sizes as 32-bit integers.
        let rows_heads = self.rows.checked_mul(self.heads);
        if rows_heads.is_none_or(|count| i32::try_from(count).is_err())
            || [
                self.key_dim,
                self.extent,
                self.page_tokens,
                self.max_blocks,
                self.capacity_pages,
            ]
            .iter()
            .any(|value| i32::try_from(*value).is_err())
        {
            return Err(invalid(format!("shape {self:?} exceeds 32-bit sizes")));
        }
        let tables = self.tables();
        if metadata.len() != self.metadata_len() {
            return Err(invalid(format!(
                "metadata has {} words, expected {}",
                metadata.len(),
                self.metadata_len()
            )));
        }
        for table in 0..tables {
            let context = metadata[table] as usize;
            let offset = metadata[tables + table] as usize;
            let pages = (context + offset).div_ceil(self.page_tokens);
            let blocks = &metadata[2 * tables + table * self.max_blocks..][..self.max_blocks];
            if context < self.queries_per_table
                || offset >= self.page_tokens
                || pages > self.max_blocks
                || blocks[..pages]
                    .iter()
                    .any(|page| *page as usize >= self.capacity_pages)
            {
                return Err(invalid(format!(
                    "table {table} (context {context}, first-page offset {offset}) does not fit \
                     {} queries, pages of {} tokens or {} resident pages",
                    self.queries_per_table, self.page_tokens, self.capacity_pages
                )));
            }
        }
        Ok(lanes)
    }
}

/// `a * b + c` as the kernels compute it: fused wherever the build has a
/// fused multiply-add instruction. Without one (x86 built without `fma`) the
/// library fallback would cost more than the attention, so the product
/// rounds first and the host agrees with the kernels within [`TOLERANCE`]
/// rather than bit for bit.
#[inline(always)]
fn fma(a: f32, b: f32, c: f32) -> f32 {
    if cfg!(any(target_arch = "aarch64", target_feature = "fma")) {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// `q . k` in eight chains, as the kernels sum it.
#[inline(always)]
fn dot<K: Copy>(q: &[f32], k: &[K]) -> f32
where
    f32: From<K>,
{
    let mut chains = [0f32; 8];
    let mut q_blocks = q.chunks_exact(8);
    let mut k_blocks = k.chunks_exact(8);
    for (q, k) in (&mut q_blocks).zip(&mut k_blocks) {
        for chain in 0..8 {
            chains[chain] = fma(q[chain], f32::from(k[chain]), chains[chain]);
        }
    }
    for (chain, (q, k)) in q_blocks
        .remainder()
        .iter()
        .zip(k_blocks.remainder())
        .enumerate()
    {
        chains[chain] = fma(*q, f32::from(*k), chains[chain]);
    }
    ((chains[0] + chains[1]) + (chains[2] + chains[3]))
        + ((chains[4] + chains[5]) + (chains[6] + chains[7]))
}

/// The op on host slices: `queries` `[rows, heads, key_dim]`, `profiles`
/// `[rows, heads, extent]`, `sinks` `[rows, heads]`, `keys` and `values`
/// `[capacity_pages, page_tokens, kv_heads, dim]`, `metadata` packed as the
/// module describes. Returns `[rows, heads, value_dim]`.
///
/// This is the kernels' arithmetic (see the module docs for the order) and
/// the CPU arena's implementation; the (row, head) units run in parallel and
/// never share a sum.
pub fn reference<K>(
    geometry: &Geometry,
    queries: &[f32],
    profiles: &[f32],
    sinks: &[f32],
    keys: &[K],
    values: &[K],
    metadata: &[u32],
) -> Result<Vec<f32>>
where
    K: Copy + Sync,
    f32: From<K>,
{
    let g = *geometry;
    let lanes = g.validate(metadata)?;
    let rows_heads = g.rows * g.heads;
    let slots = g.capacity_pages * g.page_tokens * g.kv_heads;
    if queries.len() != rows_heads * g.key_dim
        || profiles.len() != rows_heads * g.extent
        || sinks.len() != rows_heads
        || keys.len() < slots * g.key_dim
        || values.len() < slots * g.value_dim
    {
        return Err(invalid("operand sizes do not match the shape"));
    }
    let tables = g.tables();
    let group = g.heads / g.kv_heads;
    let mut output = vec![0f32; rows_heads * g.value_dim];
    output
        .par_chunks_mut(g.value_dim)
        .enumerate()
        .for_each(|(row_head, out)| {
            let (row, head) = (row_head / g.heads, row_head % g.heads);
            let kv_head = head / group;
            let table = row / g.queries_per_table;
            let later = g.queries_per_table - 1 - row % g.queries_per_table;
            let context = metadata[table] as usize - later;
            let first_page_offset = metadata[tables + table] as usize;
            let blocks = &metadata[2 * tables + table * g.max_blocks..][..g.max_blocks];
            let visible = g
                .window
                .map_or(context, |window| context.min(window as usize));
            let q = &queries[row_head * g.key_dim..][..g.key_dim];
            let profile = &profiles[row_head * g.extent..][..g.extent];

            let mut maxima = [f32::NEG_INFINITY; MAX_LANES];
            let mut sums = [0f32; MAX_LANES];
            let mut accumulators = vec![0f32; lanes * g.value_dim];
            // Walking the distances in order visits each lane's keys in the
            // lane's own order; the lanes never mix before the merge.
            for distance in 0..visible {
                let lane = distance % lanes;
                let token = first_page_offset + (context - 1 - distance);
                let slot =
                    blocks[token / g.page_tokens] as usize * g.page_tokens + token % g.page_tokens;
                let kv_row = slot * g.kv_heads + kv_head;
                let bias = if distance < g.extent {
                    profile[distance]
                } else {
                    0.0
                };
                let score = fma(
                    dot(q, &keys[kv_row * g.key_dim..][..g.key_dim]),
                    g.scale,
                    bias,
                );
                let (mut previous_weight, mut token_weight) = (1f32, 1f32);
                if score > maxima[lane] {
                    previous_weight = det_exp(maxima[lane] - score);
                    maxima[lane] = score;
                } else {
                    token_weight = det_exp(score - maxima[lane]);
                }
                sums[lane] = fma(sums[lane], previous_weight, token_weight);
                let value = &values[kv_row * g.value_dim..][..g.value_dim];
                let accumulator = &mut accumulators[lane * g.value_dim..][..g.value_dim];
                for (acc, v) in accumulator.iter_mut().zip(value) {
                    *acc = fma(*acc, previous_weight, f32::from(*v) * token_weight);
                }
            }

            // The sink first, then the lanes in order.
            let sink = sinks[row_head];
            let mut top = sink;
            for lane_max in &maxima[..lanes] {
                if *lane_max > top {
                    top = *lane_max;
                }
            }
            let mut weights = [0f32; MAX_LANES];
            let mut total = det_exp(sink - top);
            for lane in 0..lanes {
                weights[lane] = det_exp(maxima[lane] - top);
                total = fma(sums[lane], weights[lane], total);
            }
            for (dim, out) in out.iter_mut().enumerate() {
                let mut sum = 0f32;
                for lane in 0..lanes {
                    sum = fma(accumulators[lane * g.value_dim + dim], weights[lane], sum);
                }
                *out = sum / total;
            }
        });
    Ok(output)
}

/// One call on the operands' device.
#[derive(Debug, Clone, Copy)]
pub struct PagedBiasSink<'a> {
    /// `[rows, heads, key_dim]` F32.
    pub queries: &'a Tensor,
    /// `[rows, heads, extent]` F32, indexed by backward distance.
    pub profiles: &'a Tensor,
    /// `[rows, heads]` F32.
    pub sinks: &'a Tensor,
    /// `[pages, page_tokens, kv_heads, key_dim]` in the arena's storage.
    pub keys: &'a Tensor,
    /// `[pages, page_tokens, kv_heads, value_dim]`, the keys' dtype.
    pub values: &'a Tensor,
    /// The packed block tables on the host, validated on every call.
    pub metadata: &'a [u32],
    /// The same words as a U32 tensor on the operands' device; a CPU call
    /// reads the host copy and takes `None`.
    pub device_metadata: Option<&'a Tensor>,
    pub max_blocks: usize,
    pub queries_per_table: usize,
    pub scale: f32,
    pub window: Option<u32>,
}

impl PagedBiasSink<'_> {
    /// The call's shape, after checking the operands against each other.
    pub fn geometry(&self) -> Result<Geometry> {
        let (rows, heads, key_dim) = self.queries.dims3()?;
        let (pages, page_tokens, kv_heads, stored_key_dim) = self.keys.dims4()?;
        let (value_pages, value_page_tokens, value_heads, value_dim) = self.values.dims4()?;
        let profile_dims = self.profiles.dims();
        if profile_dims.len() != 3
            || profile_dims[..2] != [rows, heads]
            || self.sinks.dims() != [rows, heads]
            || stored_key_dim != key_dim
            || (value_pages, value_page_tokens, value_heads) != (pages, page_tokens, kv_heads)
        {
            return Err(invalid(format!(
                "queries {:?}, profiles {:?}, sinks {:?}, keys {:?} and values {:?} do not agree",
                self.queries.dims(),
                profile_dims,
                self.sinks.dims(),
                self.keys.dims(),
                self.values.dims()
            )));
        }
        let float_inputs = [self.queries, self.profiles, self.sinks];
        if float_inputs.iter().any(|t| t.dtype() != DType::F32) {
            return Err(invalid("queries, profiles and sinks must be F32"));
        }
        if self.keys.dtype() != self.values.dtype() {
            return Err(invalid("keys and values must share a storage dtype"));
        }
        let location = self.queries.device().location();
        let operands = [self.profiles, self.sinks, self.keys, self.values];
        if operands
            .iter()
            .chain(self.device_metadata.iter())
            .any(|t| t.device().location() != location)
        {
            return Err(invalid("operands are on different devices"));
        }
        if float_inputs
            .iter()
            .chain([self.keys, self.values].iter())
            .any(|t| !t.is_contiguous())
        {
            return Err(invalid("operands must be contiguous"));
        }
        Ok(Geometry {
            rows,
            queries_per_table: self.queries_per_table,
            heads,
            kv_heads,
            key_dim,
            value_dim,
            extent: profile_dims[2],
            page_tokens,
            max_blocks: self.max_blocks,
            capacity_pages: pages,
            window: self.window,
            scale: self.scale,
        })
    }
}

/// The attention of `spec`, F32 `[rows, heads, value_dim]` on the operands'
/// device: [`reference`] on the CPU (F32, F16 or BF16 storage), one launch on
/// Metal (F32 or F16 storage) and on CUDA (F32, F16 or BF16 storage).
pub fn attend(spec: &PagedBiasSink<'_>) -> Result<Tensor> {
    let geometry = spec.geometry()?;
    geometry.validate(spec.metadata)?;
    let shape = (geometry.rows, geometry.heads, geometry.value_dim);
    match spec.queries.device() {
        Device::Cpu => {
            let output = host_attend(spec, &geometry)?;
            Ok(Tensor::from_vec(output, shape, &Device::Cpu)?)
        }
        Device::Metal(_) => {
            #[cfg(feature = "metal")]
            {
                Ok(metal_impl::run(spec, &geometry)?)
            }
            #[cfg(not(feature = "metal"))]
            Err(invalid("this build has no Metal kernels"))
        }
        Device::Cuda(_) => {
            #[cfg(feature = "cuda")]
            {
                Ok(cuda_impl::run(spec, &geometry)?)
            }
            #[cfg(not(feature = "cuda"))]
            Err(invalid("this build has no CUDA kernels"))
        }
    }
}

/// [`reference`] over the CPU tensors' own storage.
fn host_attend(spec: &PagedBiasSink<'_>, geometry: &Geometry) -> Result<Vec<f32>> {
    fn floats<'a>(storage: &'a Storage, tensor: &Tensor) -> Result<&'a [f32]> {
        match (storage, tensor.layout().contiguous_offsets()) {
            (Storage::Cpu(CpuStorage::F32(values)), Some((start, end))) => Ok(&values[start..end]),
            _ => Err(invalid("expected a contiguous F32 CPU tensor")),
        }
    }
    let (queries, _) = spec.queries.storage_and_layout();
    let (profiles, _) = spec.profiles.storage_and_layout();
    let (sinks, _) = spec.sinks.storage_and_layout();
    let (keys, _) = spec.keys.storage_and_layout();
    let (values, _) = spec.values.storage_and_layout();
    let (key_range, value_range) = match (
        spec.keys.layout().contiguous_offsets(),
        spec.values.layout().contiguous_offsets(),
    ) {
        (Some(keys), Some(values)) => (keys.0..keys.1, values.0..values.1),
        _ => return Err(invalid("keys and values must be contiguous")),
    };
    let queries = floats(&queries, spec.queries)?;
    let profiles = floats(&profiles, spec.profiles)?;
    let sinks = floats(&sinks, spec.sinks)?;
    macro_rules! typed {
        ($keys:expr, $values:expr) => {
            reference(
                geometry,
                queries,
                profiles,
                sinks,
                &$keys[key_range],
                &$values[value_range],
                spec.metadata,
            )
        };
    }
    match (&*keys, &*values) {
        (Storage::Cpu(CpuStorage::F32(keys)), Storage::Cpu(CpuStorage::F32(values))) => {
            typed!(keys, values)
        }
        (Storage::Cpu(CpuStorage::F16(keys)), Storage::Cpu(CpuStorage::F16(values))) => {
            typed!(keys, values)
        }
        (Storage::Cpu(CpuStorage::BF16(keys)), Storage::Cpu(CpuStorage::BF16(values))) => {
            typed!(keys, values)
        }
        _ => Err(invalid("keys and values must be F32, F16 or BF16")),
    }
}

/// What [`self_check`] measured.
#[derive(Debug, Clone, PartialEq)]
pub struct SelfCheck {
    pub values: usize,
    /// Outputs whose bits differ from the host reference.
    pub differing: usize,
    /// `max |device - host| / max |host|`.
    pub worst_ratio: f64,
    /// Why the device result is not usable, if it is not.
    pub failure: Option<String>,
}

/// Runs the op on `device` over a fixed case (grouped heads, a shuffled
/// table with a first-page offset, a window on half the calls, a bias
/// shorter than the context, a span of several queries) with keys and values
/// stored as `kv_dtype`, and compares it with [`reference`] over the same
/// stored values. A caller may gate the fused path on `failure` being
/// `None`; nothing here runs it implicitly.
pub fn self_check(device: &Device, kv_dtype: DType) -> Result<SelfCheck> {
    use super::numeric_check::{host_f32, xorshift_tensor, xorshift_values};

    let (heads, kv_heads, key_dim, value_dim, extent) = (6usize, 2usize, 128usize, 128usize, 24);
    let (page_tokens, pages, span) = (16usize, 8usize, 3usize);
    let table: [u32; 6] = [5, 2, 7, 0, 3, 6];
    let (first_page_offset, context) = (9u32, 81u32);
    let mut metadata = vec![context, first_page_offset];
    metadata.extend_from_slice(&table);
    let stored = |dim: usize, seed: u64| -> Result<(Tensor, Vec<f32>)> {
        let tensor = xorshift_tensor(&[pages, page_tokens, kv_heads, dim], seed, 1.0, device)?
            .to_dtype(kv_dtype)?;
        let host = host_f32(&tensor)?;
        Ok((tensor, host))
    };
    let (keys, host_keys) = stored(key_dim, 11)?;
    let (values, host_values) = stored(value_dim, 12)?;
    let host_queries = xorshift_values(span * heads * key_dim, 13, 1.0);
    let host_profiles = xorshift_values(span * heads * extent, 14, 2.0);
    let host_sinks = xorshift_values(span * heads, 15, 3.0);
    let queries = Tensor::from_vec(host_queries.clone(), (span, heads, key_dim), device)?;
    let profiles = Tensor::from_vec(host_profiles.clone(), (span, heads, extent), device)?;
    let sinks = Tensor::from_vec(host_sinks.clone(), (span, heads), device)?;
    let device_metadata = Tensor::from_vec(metadata.clone(), metadata.len(), device)?;

    let mut check = SelfCheck {
        values: 0,
        differing: 0,
        worst_ratio: 0.0,
        failure: None,
    };
    for window in [None, Some(40u32)] {
        let spec = PagedBiasSink {
            queries: &queries,
            profiles: &profiles,
            sinks: &sinks,
            keys: &keys,
            values: &values,
            metadata: &metadata,
            device_metadata: (!device.is_cpu()).then_some(&device_metadata),
            max_blocks: table.len(),
            queries_per_table: span,
            scale: (key_dim as f32).powf(-0.5),
            window,
        };
        let got = host_f32(&attend(&spec)?)?;
        let want = reference(
            &spec.geometry()?,
            &host_queries,
            &host_profiles,
            &host_sinks,
            &host_keys,
            &host_values,
            &metadata,
        )?;
        let peak = want.iter().fold(1e-6f64, |m, v| m.max(f64::from(v.abs())));
        for (got, want) in got.iter().zip(&want) {
            check.values += 1;
            check.differing += usize::from(got.to_bits() != want.to_bits());
            let error = f64::from((got - want).abs()) / peak;
            check.worst_ratio = if error.is_finite() {
                check.worst_ratio.max(error)
            } else {
                f64::INFINITY
            };
        }
    }
    if check.worst_ratio > TOLERANCE {
        check.failure = Some(format!(
            "device and host differ by {:e} of the peak, past {TOLERANCE:e}",
            check.worst_ratio
        ));
    }
    Ok(check)
}

/// One launch: a block per (row, head), [`lanes`] threads.
#[cfg(feature = "cuda")]
mod cuda_impl {
    use candle_core::cuda_backend::cudarc::driver::{CudaView, LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::{CudaDType, CudaStorage, WrapErr};
    use candle_core::op::BackpropOp;
    use candle_core::{DType, Result, Shape, Storage, Tensor};

    use super::{Geometry, PagedBiasSink};

    const MODULE: &str = "izwi_paged_bias_sink";

    fn view<'a, T: CudaDType>(
        storage: &'a Storage,
        tensor: &Tensor,
        name: &str,
    ) -> Result<CudaView<'a, T>> {
        let Storage::Cuda(cuda) = storage else {
            candle_core::bail!("paged bias/sink attention: {name} must be a CUDA tensor")
        };
        let Some((start, end)) = tensor.layout().contiguous_offsets() else {
            candle_core::bail!("paged bias/sink attention: {name} must be contiguous")
        };
        Ok(cuda.as_cuda_slice::<T>()?.slice(start..end))
    }

    fn int(value: usize) -> Result<i32> {
        i32::try_from(value).map_err(|_| {
            candle_core::Error::Msg(format!("paged bias/sink attention: {value} exceeds i32"))
        })
    }

    pub(super) fn run(spec: &PagedBiasSink<'_>, geometry: &Geometry) -> Result<Tensor> {
        let g = geometry;
        let Some(lanes) = super::lanes(g.value_dim) else {
            candle_core::bail!("paged bias/sink attention: unsupported value head dim")
        };
        let Some(metadata) = spec.device_metadata else {
            candle_core::bail!("paged bias/sink attention: CUDA needs the tables on the device")
        };
        if metadata.dtype() != DType::U32 || metadata.elem_count() != spec.metadata.len() {
            candle_core::bail!("paged bias/sink attention: device metadata does not match")
        }
        let device = spec.queries.device().as_cuda_device()?;
        let (qs, _) = spec.queries.storage_and_layout();
        let (ps, _) = spec.profiles.storage_and_layout();
        let (ss, _) = spec.sinks.storage_and_layout();
        let (ks, _) = spec.keys.storage_and_layout();
        let (vs, _) = spec.values.storage_and_layout();
        let (ms, _) = metadata.storage_and_layout();
        let qv = view::<f32>(&qs, spec.queries, "queries")?;
        let pv = view::<f32>(&ps, spec.profiles, "profiles")?;
        let sv = view::<f32>(&ss, spec.sinks, "sinks")?;
        let mv = view::<u32>(&ms, metadata, "metadata")?;
        let count = g.rows * g.heads * g.value_dim;
        // SAFETY: every block writes its (row, head)'s whole value row.
        let out = unsafe { device.alloc::<f32>(count)? };
        let name = match spec.keys.dtype() {
            DType::F32 => "izwi_paged_bias_sink_f32",
            DType::F16 => "izwi_paged_bias_sink_f16",
            DType::BF16 => "izwi_paged_bias_sink_bf16",
            other => candle_core::bail!(
                "paged bias/sink attention: CUDA keys and values must be F32, F16 or BF16, \
                 got {other:?}"
            ),
        };
        let ptx = super::super::cuda::cuda_ptx_paged_bias_sink();
        let function = device.get_or_load_custom_func(name, MODULE, ptx)?;
        // One pair of views per storage type; the builder borrows the pair
        // it is given until the launch.
        let (kf, vf, kh, vh, kb, vb);
        let mut builder = function.builder();
        builder.arg(&qv);
        builder.arg(&pv);
        builder.arg(&sv);
        match spec.keys.dtype() {
            DType::F32 => {
                kf = view::<f32>(&ks, spec.keys, "keys")?;
                vf = view::<f32>(&vs, spec.values, "values")?;
                builder.arg(&kf);
                builder.arg(&vf);
            }
            DType::F16 => {
                kh = view::<half::f16>(&ks, spec.keys, "keys")?;
                vh = view::<half::f16>(&vs, spec.values, "values")?;
                builder.arg(&kh);
                builder.arg(&vh);
            }
            _ => {
                kb = view::<half::bf16>(&ks, spec.keys, "keys")?;
                vb = view::<half::bf16>(&vs, spec.values, "values")?;
                builder.arg(&kb);
                builder.arg(&vb);
            }
        }
        builder.arg(&mv);
        builder.arg(&out);
        candle_core::builder_arg!(
            builder,
            int(g.rows)?,
            int(g.queries_per_table)?,
            int(g.heads)?,
            int(g.kv_heads)?,
            int(g.page_tokens)?,
            int(g.max_blocks)?,
            int(g.key_dim)?,
            int(g.value_dim)?,
            int(g.extent)?,
            int(g.capacity_pages)?,
            g.window_arg(),
            g.scale
        );
        let launch = LaunchConfig {
            grid_dim: (int(g.rows * g.heads)? as u32, 1, 1),
            block_dim: (lanes as u32, 1, 1),
            // Each lane's maximum, sum and value accumulators.
            shared_mem_bytes: (lanes * (g.value_dim + 2) * std::mem::size_of::<f32>()) as u32,
        };
        // SAFETY: argument order matches `izwi_paged_bias_sink_*`; the host
        // validated the tables against the storage this call reads.
        unsafe { builder.launch(launch) }.w()?;
        drop(builder);
        Ok(Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(out, device.clone())),
            Shape::from((g.rows, g.heads, g.value_dim)),
            BackpropOp::none(),
            false,
        ))
    }
}

/// One dispatch: a thread group per (row, head), [`lanes`] threads.
#[cfg(feature = "metal")]
mod metal_impl {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    use candle_core::op::BackpropOp;
    use candle_core::{bail, DType, MetalStorage, Result, Shape, Storage, Tensor};
    use candle_metal_kernels::metal::{Buffer, ComputePipeline, Device as MetalDevice};
    use objc2_metal::{MTLCompileOptions, MTLSize};

    use super::{Geometry, PagedBiasSink};
    use crate::kernels::metal_encoder::IzwiMetalCommandEncoderExt;

    // `cuda/paged_bias_sink.cu` in Metal: the same lanes, chains and merge,
    // `fma` where the CUDA source has `fmaf`. The storage type, the lane
    // count and the value width are compiled in (`PBS_KV`, `PBS_LANES`,
    // `PBS_VALUE_DIM`) because thread-group arrays need constant sizes.
    const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// `relpos_attention::det_exp`: the same literals and operations.
inline float pbs_exp(float x) {
    if (isnan(x)) {
        return x;
    }
    if (x <= -104.0f) {
        return 0.0f;
    }
    const float n = rint(x * 1.442695f);
    float r = fma(-n, 0.6933594f, x);
    r = fma(-n, -0.00021219444f, r);
    float p = 0.00019875691f;
    p = fma(p, r, 0.0013981999f);
    p = fma(p, r, 0.008333452f);
    p = fma(p, r, 0.041665796f);
    p = fma(p, r, 0.16666666f);
    p = fma(p, r, 0.5f);
    const float rr = r * r;
    const float y = fma(p, rr, r) + 1.0f;
    const int e = int(n);
    if (e >= -126) {
        return y * as_type<float>(uint(e + 127) << 23);
    }
    const float low = y * as_type<float>(uint(e + 191) << 23);
    return low * as_type<float>(uint(63) << 23);
}

// q . k in eight chains (dim d joins chain d % 8), summed pairwise.
inline float pbs_dot(device const float* q, device const PBS_KV* k, uint dim) {
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    float c4 = 0.0f, c5 = 0.0f, c6 = 0.0f, c7 = 0.0f;
    uint d = 0;
    for (; d + 8 <= dim; d += 8) {
        c0 = fma(q[d], float(k[d]), c0);
        c1 = fma(q[d + 1], float(k[d + 1]), c1);
        c2 = fma(q[d + 2], float(k[d + 2]), c2);
        c3 = fma(q[d + 3], float(k[d + 3]), c3);
        c4 = fma(q[d + 4], float(k[d + 4]), c4);
        c5 = fma(q[d + 5], float(k[d + 5]), c5);
        c6 = fma(q[d + 6], float(k[d + 6]), c6);
        c7 = fma(q[d + 7], float(k[d + 7]), c7);
    }
    if (d < dim) c0 = fma(q[d], float(k[d]), c0);
    if (d + 1 < dim) c1 = fma(q[d + 1], float(k[d + 1]), c1);
    if (d + 2 < dim) c2 = fma(q[d + 2], float(k[d + 2]), c2);
    if (d + 3 < dim) c3 = fma(q[d + 3], float(k[d + 3]), c3);
    if (d + 4 < dim) c4 = fma(q[d + 4], float(k[d + 4]), c4);
    if (d + 5 < dim) c5 = fma(q[d + 5], float(k[d + 5]), c5);
    if (d + 6 < dim) c6 = fma(q[d + 6], float(k[d + 6]), c6);
    const float a = c0 + c1;
    const float b = c2 + c3;
    const float c = c4 + c5;
    const float e = c6 + c7;
    const float ab = a + b;
    const float ce = c + e;
    return ab + ce;
}

// dims: rows, span, heads, kv_heads, page_tokens, max_blocks, key_dim,
// extent, window (0 for none). One thread group per (head, row), PBS_LANES
// threads. The host validated the tables.
kernel void pbs_attend(
    device const float* queries [[buffer(0)]],
    device const float* profiles [[buffer(1)]],
    device const float* sinks [[buffer(2)]],
    device const PBS_KV* keys [[buffer(3)]],
    device const PBS_KV* values [[buffer(4)]],
    device const uint* metadata [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint* dims [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float lane_state[PBS_LANES * 2];
    threadgroup float lane_acc[PBS_LANES * PBS_VALUE_DIM];

    const uint rows = dims[0], span = dims[1], heads = dims[2], kv_heads = dims[3];
    const uint page_tokens = dims[4], max_blocks = dims[5], key_dim = dims[6];
    const uint extent = dims[7], window = dims[8];
    const uint head = group.x;
    const uint row = group.y;
    if (row >= rows || head >= heads) {
        return;
    }
    const ulong row_head = (ulong)row * heads + head;
    const uint kv_head = head / (heads / kv_heads);
    const uint tables = rows / span;
    const uint table = row / span;
    // Tokens of the table's context that come after this row's query.
    const uint later = span - 1 - (row - table * span);
    const uint context = metadata[table] - later;
    const uint first_page_offset = metadata[tables + table];
    device const uint* blocks = metadata + 2 * (ulong)tables + (ulong)table * max_blocks;
    const uint visible = (window != 0 && window < context) ? window : context;
    device const float* q = queries + row_head * key_dim;
    device const float* profile = profiles + row_head * extent;

    float acc[PBS_VALUE_DIM];
    for (uint dim = 0; dim < PBS_VALUE_DIM; ++dim) {
        acc[dim] = 0.0f;
    }
    float lane_max = -INFINITY;
    float lane_sum = 0.0f;
    for (uint distance = lane; distance < visible; distance += PBS_LANES) {
        const uint token = first_page_offset + (context - 1 - distance);
        const uint logical_page = token / page_tokens;
        const ulong slot = (ulong)blocks[logical_page] * page_tokens
            + (token - logical_page * page_tokens);
        const ulong kv_row = slot * kv_heads + kv_head;
        const float dot = pbs_dot(q, keys + kv_row * key_dim, key_dim);
        const float bias = distance < extent ? profile[distance] : 0.0f;
        const float score = fma(dot, scale, bias);
        float previous_weight = 1.0f;
        float token_weight = 1.0f;
        if (score > lane_max) {
            previous_weight = pbs_exp(lane_max - score);
            lane_max = score;
        } else {
            token_weight = pbs_exp(score - lane_max);
        }
        lane_sum = fma(lane_sum, previous_weight, token_weight);
        device const PBS_KV* value = values + kv_row * PBS_VALUE_DIM;
        for (uint dim = 0; dim < PBS_VALUE_DIM; ++dim) {
            const float weighted = float(value[dim]) * token_weight;
            acc[dim] = fma(acc[dim], previous_weight, weighted);
        }
    }
    lane_state[lane * 2] = lane_max;
    lane_state[lane * 2 + 1] = lane_sum;
    for (uint dim = 0; dim < PBS_VALUE_DIM; ++dim) {
        lane_acc[lane * PBS_VALUE_DIM + dim] = acc[dim];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The sink first, then the lanes in order.
    const float sink = sinks[row_head];
    float top = sink;
    for (uint l = 0; l < PBS_LANES; ++l) {
        const float other = lane_state[l * 2];
        if (other > top) {
            top = other;
        }
    }
    float weights[PBS_LANES];
    float total = pbs_exp(sink - top);
    for (uint l = 0; l < PBS_LANES; ++l) {
        weights[l] = pbs_exp(lane_state[l * 2] - top);
        total = fma(lane_state[l * 2 + 1], weights[l], total);
    }
    device float* out = output + row_head * PBS_VALUE_DIM;
    for (uint dim = lane; dim < PBS_VALUE_DIM; dim += PBS_LANES) {
        float sum = 0.0f;
        for (uint l = 0; l < PBS_LANES; ++l) {
            sum = fma(lane_acc[l * PBS_VALUE_DIM + dim], weights[l], sum);
        }
        out[dim] = sum / total;
    }
}
"#;

    /// `(device, half storage, lanes, value_dim)`.
    type PipelineKey = (u64, bool, usize, usize);

    fn pipeline(
        device: &MetalDevice,
        half_storage: bool,
        lanes: usize,
        value_dim: usize,
    ) -> Result<ComputePipeline> {
        static PIPELINES: OnceLock<Mutex<HashMap<PipelineKey, ComputePipeline>>> = OnceLock::new();
        let key = (device.registry_id(), half_storage, lanes, value_dim);
        let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(pipeline) = pipelines
            .lock()
            .map_err(|err| candle_core::Error::Msg(err.to_string()))?
            .get(&key)
            .cloned()
        {
            return Ok(pipeline);
        }
        let source = format!(
            "#define PBS_KV {}\n#define PBS_LANES {lanes}\n#define PBS_VALUE_DIM {value_dim}\n{SOURCE}",
            if half_storage { "half" } else { "float" },
        );
        // Metal compiles with fast math unless told otherwise, which lets the
        // compiler reorder the sums and split the fused multiply-adds. The
        // kernel promises the reference's order, so it is off.
        let options = MTLCompileOptions::new();
        #[allow(deprecated)] // `mathMode` needs macOS 15; this covers every release we run on.
        options.setFastMathEnabled(false);
        let library = device
            .new_library_with_source(&source, Some(&options))
            .map_err(candle_core::Error::wrap)?;
        let function = library
            .get_function("pbs_attend", None)
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

    /// A contiguous Metal tensor's buffer and byte offset.
    struct Bound<'a> {
        guard: candle_core::StorageRef<'a>,
        offset: usize,
    }

    impl Bound<'_> {
        fn buffer(&self) -> Result<&Buffer> {
            match &*self.guard {
                Storage::Metal(storage) => Ok(storage.buffer()),
                _ => bail!("paged bias/sink attention: Metal kernels need Metal tensors"),
            }
        }
    }

    fn bind<'a>(tensor: &'a Tensor, name: &str) -> Result<Bound<'a>> {
        let (guard, layout) = tensor.storage_and_layout();
        if !layout.is_contiguous() {
            bail!("paged bias/sink attention: {name} must be contiguous")
        }
        let offset = layout.start_offset() * tensor.dtype().size_in_bytes();
        Ok(Bound { guard, offset })
    }

    pub(super) fn run(spec: &PagedBiasSink<'_>, geometry: &Geometry) -> Result<Tensor> {
        let g = geometry;
        let Some(lanes) = super::lanes(g.value_dim) else {
            bail!("paged bias/sink attention: unsupported value head dim")
        };
        let half_storage = match spec.keys.dtype() {
            DType::F32 => false,
            DType::F16 => true,
            other => bail!(
                "paged bias/sink attention: Metal keys and values must be F32 or F16, got \
                 {other:?}"
            ),
        };
        let Some(metadata) = spec.device_metadata else {
            bail!("paged bias/sink attention: Metal needs the tables on the device")
        };
        if metadata.dtype() != DType::U32 || metadata.elem_count() != spec.metadata.len() {
            bail!("paged bias/sink attention: device metadata does not match")
        }
        let candle_core::Device::Metal(device) = spec.queries.device() else {
            bail!("paged bias/sink attention: Metal kernels need Metal tensors")
        };
        let mut dims = [0u32; 9];
        for (slot, value) in dims.iter_mut().zip([
            g.rows,
            g.queries_per_table,
            g.heads,
            g.kv_heads,
            g.page_tokens,
            g.max_blocks,
            g.key_dim,
            g.extent,
            g.window_arg() as usize,
        ]) {
            *slot = u32::try_from(value).map_err(|_| {
                candle_core::Error::Msg(format!("paged bias/sink attention: {value} exceeds u32"))
            })?;
        }
        let pipeline = pipeline(device.metal_device(), half_storage, lanes, g.value_dim)?;
        let qb = bind(spec.queries, "queries")?;
        let pb = bind(spec.profiles, "profiles")?;
        let sb = bind(spec.sinks, "sinks")?;
        let kb = bind(spec.keys, "keys")?;
        let vb = bind(spec.values, "values")?;
        let mb = bind(metadata, "metadata")?;
        let count = g.rows * g.heads * g.value_dim;
        let out = device.new_buffer(count, DType::F32, "pbs-out")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("paged-bias-sink");
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_input_buffer(0, Some(qb.buffer()?), qb.offset);
        encoder.set_input_buffer(1, Some(pb.buffer()?), pb.offset);
        encoder.set_input_buffer(2, Some(sb.buffer()?), sb.offset);
        encoder.set_input_buffer(3, Some(kb.buffer()?), kb.offset);
        encoder.set_input_buffer(4, Some(vb.buffer()?), vb.offset);
        encoder.set_input_buffer(5, Some(mb.buffer()?), mb.offset);
        encoder.set_output_buffer(6, Some(&out), 0);
        encoder.set_bytes(7, &dims);
        encoder.set_bytes(8, &g.scale);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: g.heads,
                height: g.rows,
                depth: 1,
            },
            MTLSize {
                width: lanes,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);
        drop((qb, pb, sb, kb, vb, mb));
        Ok(Tensor::from_storage(
            Storage::Metal(MetalStorage::new(out, device.clone(), count, DType::F32)),
            Shape::from((g.rows, g.heads, g.value_dim)),
            BackpropOp::none(),
            false,
        ))
    }
}

/// Scenes shared by the CPU, Metal and CUDA tests: an arena with every slot
/// written, the op through `KvArena::paged_decode_biased`, and a dense F64
/// evaluation of the same stored values.
#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::backends::kv::{
        CpuKvArena, KvArena, KvArenaConfig, KvLayerConfig, KvWriteArgs, PagedKvBiasedDecodeArgs,
    };
    use crate::backends::BackendKind;
    use crate::engine::ModelInstanceId;
    use crate::kernels::numeric_check::{host_f32, xorshift_values};
    use crate::kv::{
        CacheBlockRef, KvArenaId, KvDecodeBatchMetadata, KvGroupId, KvLayerBinding,
        KvSequenceBlockTable, KvSlotRef,
    };

    const GROUP: KvGroupId = KvGroupId::new(3);
    const LAYER: KvLayerBinding = KvLayerBinding {
        model_layer: 5,
        physical_layer: 0,
    };

    /// `(physical pages in logical order, first-page offset, context)`.
    pub(crate) type Table = (Vec<u32>, u32, u32);

    #[derive(Debug, Clone)]
    pub(crate) struct Scene {
        pub name: &'static str,
        pub heads: usize,
        pub kv_heads: usize,
        pub key_dim: usize,
        pub value_dim: usize,
        pub extent: usize,
        pub page_tokens: usize,
        pub pages: usize,
        pub tables: Vec<Table>,
        pub span: usize,
        pub window: Option<u32>,
    }

    impl Scene {
        pub(crate) fn rows(&self) -> usize {
            self.tables.len() * self.span
        }

        fn scale(&self) -> f32 {
            (self.key_dim as f32).powf(-0.5)
        }
    }

    /// A table over `pages` shuffled physical pages that ends `context`
    /// tokens after `offset`.
    fn shuffled(pages: usize, stride: usize, start: usize, offset: u32, context: u32) -> Table {
        let needed = (offset + context).div_ceil(16) as usize;
        (
            (0..needed)
                .map(|page| ((start + page * stride) % pages) as u32)
                .collect(),
            offset,
            context,
        )
    }

    /// The case matrix: the published grouped geometry (24 heads over 8 KV
    /// heads, 128 dims) with ragged shuffled tables, a window shorter than
    /// the contexts, a bias shorter than the contexts without a window, a
    /// tiny geometry whose key and value widths differ, one-token contexts,
    /// spans of several queries, and a value row wide enough for 16 lanes.
    pub(crate) fn scenes() -> Vec<Scene> {
        let ragged = vec![
            shuffled(13, 5, 2, 9, 150),
            shuffled(13, 3, 7, 0, 33),
            shuffled(13, 1, 4, 15, 1),
            shuffled(13, 4, 0, 3, 97),
        ];
        let grouped = Scene {
            name: "grouped, bias shorter than the context",
            heads: 24,
            kv_heads: 8,
            key_dim: 128,
            value_dim: 128,
            extent: 40,
            page_tokens: 16,
            pages: 13,
            tables: ragged.clone(),
            span: 1,
            window: None,
        };
        let tiny_tables = vec![
            (vec![7, 2, 9, 0], 1, 13),
            (vec![5], 3, 1),
            (vec![11, 4, 1], 0, 9),
            (vec![3, 10, 6, 8], 2, 14),
        ];
        let tiny = Scene {
            name: "tiny, unequal key and value widths",
            heads: 6,
            kv_heads: 2,
            key_dim: 6,
            value_dim: 3,
            extent: 5,
            page_tokens: 4,
            pages: 12,
            tables: tiny_tables,
            span: 1,
            window: None,
        };
        vec![
            grouped.clone(),
            Scene {
                name: "grouped, window shorter than the context",
                extent: 64,
                window: Some(64),
                ..grouped.clone()
            },
            Scene {
                name: "grouped, window longer than the bias",
                extent: 24,
                window: Some(50),
                ..grouped.clone()
            },
            tiny.clone(),
            Scene {
                name: "tiny, window",
                window: Some(4),
                ..tiny.clone()
            },
            Scene {
                name: "one-token contexts",
                tables: vec![shuffled(13, 1, 4, 15, 1), shuffled(13, 1, 9, 0, 1)],
                ..grouped.clone()
            },
            Scene {
                name: "spans of five queries",
                heads: 6,
                kv_heads: 2,
                key_dim: 32,
                value_dim: 32,
                tables: vec![shuffled(13, 5, 2, 9, 150), shuffled(13, 3, 7, 14, 5)],
                span: 5,
                ..grouped.clone()
            },
            Scene {
                name: "spans of nine queries, window",
                heads: 4,
                kv_heads: 4,
                key_dim: 20,
                value_dim: 12,
                extent: 16,
                tables: vec![shuffled(13, 2, 1, 6, 77)],
                span: 9,
                window: Some(16),
                ..grouped.clone()
            },
            Scene {
                name: "sixteen lanes",
                heads: 4,
                kv_heads: 2,
                key_dim: 256,
                value_dim: 256,
                tables: vec![shuffled(13, 5, 2, 9, 150), shuffled(13, 3, 7, 0, 33)],
                ..grouped
            },
        ]
    }

    /// An arena holding a scene's keys and values, and what it stores.
    pub(crate) struct Stored {
        pub arena: Arc<dyn KvArena>,
        id: KvArenaId,
        /// The stored keys and values widened back to F32.
        pub keys: Vec<f32>,
        pub values: Vec<f32>,
    }

    /// Writes every slot of a fresh arena on `device` storing `dtype`.
    /// `rounded` are the values to write when they must already be exact in
    /// another arena's dtype (the CPU reference of an F16 or BF16 arena).
    pub(crate) fn store(
        scene: &Scene,
        device: &Device,
        dtype: DType,
        rounded: Option<(&[f32], &[f32])>,
    ) -> Stored {
        let backend = if device.is_metal() {
            BackendKind::Metal
        } else if device.is_cuda() {
            BackendKind::Cuda
        } else {
            BackendKind::Cpu
        };
        let id = KvArenaId {
            model_instance: ModelInstanceId::new(77),
            backend,
            device_ordinal: (backend != BackendKind::Cpu).then_some(0),
            generation: 1,
        };
        let config = KvArenaConfig {
            id,
            group: GROUP,
            page_tokens: scene.page_tokens as u32,
            capacity_pages: scene.pages as u32,
            growth: None,
            dtype,
            layers: vec![KvLayerConfig {
                binding: LAYER,
                num_kv_heads: scene.kv_heads as u32,
                key_head_dim: scene.key_dim as u32,
                value_head_dim: scene.value_dim as u32,
            }],
        };
        let arena: Arc<dyn KvArena> = if backend == BackendKind::Cpu {
            Arc::new(CpuKvArena::new(config).unwrap())
        } else {
            #[cfg(any(feature = "cuda", feature = "metal"))]
            {
                Arc::new(
                    crate::backends::kv::CandleAcceleratorKvArena::new_mutation_only(
                        config,
                        device.clone(),
                    )
                    .unwrap(),
                )
            }
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            unreachable!("an accelerator device without an accelerator feature")
        };
        let tokens = scene.pages * scene.page_tokens;
        let slots = (0..tokens)
            .map(|token| KvSlotRef {
                block: block(id, (token / scene.page_tokens) as u32),
                offset: (token % scene.page_tokens) as u32,
            })
            .collect::<Vec<_>>();
        let slots = arena.lower_slots(&slots).unwrap();
        let tensor = |dim: usize, seed: u64, given: Option<&[f32]>| {
            let values = given.map_or_else(
                || xorshift_values(tokens * scene.kv_heads * dim, seed, 1.0),
                <[f32]>::to_vec,
            );
            Tensor::from_vec(values, (tokens, scene.kv_heads, dim), device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap()
        };
        let keys = tensor(scene.key_dim, 101, rounded.map(|(keys, _)| keys));
        let values = tensor(scene.value_dim, 102, rounded.map(|(_, values)| values));
        arena
            .write_slots(
                LAYER,
                KvWriteArgs {
                    keys: &keys,
                    values: &values,
                    slots: slots.as_ref(),
                },
            )
            .unwrap()
            .wait()
            .unwrap();
        Stored {
            arena,
            id,
            keys: host_f32(&keys).unwrap(),
            values: host_f32(&values).unwrap(),
        }
    }

    fn block(arena: KvArenaId, index: u32) -> CacheBlockRef {
        CacheBlockRef {
            arena,
            group: GROUP,
            index,
            slot_generation: 1,
        }
    }

    /// A scene's queries (sharp enough that the running maxima move),
    /// profiles and sinks.
    pub(crate) struct Inputs {
        pub queries: Vec<f32>,
        pub profiles: Vec<f32>,
        pub sinks: Vec<f32>,
    }

    pub(crate) fn inputs(scene: &Scene) -> Inputs {
        let rows_heads = scene.rows() * scene.heads;
        Inputs {
            queries: xorshift_values(rows_heads * scene.key_dim, 201, 6.0),
            profiles: xorshift_values(rows_heads * scene.extent, 202, 2.0),
            sinks: xorshift_values(rows_heads, 203, 3.0),
        }
    }

    /// The op through the arena over `tables`, `span` queries each, taking
    /// rows `first_row..` of `inputs`.
    pub(crate) fn decode(
        scene: &Scene,
        stored: &Stored,
        device: &Device,
        tables: &[Table],
        span: usize,
        first_row: usize,
        inputs: &Inputs,
    ) -> Vec<f32> {
        let call = Call::new(scene, stored, device, tables, span, first_row, inputs);
        let output = call.run(scene, stored);
        assert_eq!(
            output.dims(),
            [tables.len() * span, scene.heads, scene.value_dim]
        );
        assert_eq!(output.dtype(), DType::F32);
        host_f32(&output).unwrap()
    }

    /// One call's operands on the device.
    pub(crate) struct Call {
        queries: Tensor,
        profiles: Tensor,
        sinks: Tensor,
        batch: KvDecodeBatchMetadata,
        span: usize,
    }

    impl Call {
        pub(crate) fn new(
            scene: &Scene,
            stored: &Stored,
            device: &Device,
            tables: &[Table],
            span: usize,
            first_row: usize,
            inputs: &Inputs,
        ) -> Self {
            let rows = tables.len() * span;
            let slice = |values: &[f32], width: usize| {
                let per_row = scene.heads * width;
                values[first_row * per_row..(first_row + rows) * per_row].to_vec()
            };
            let queries = Tensor::from_vec(
                slice(&inputs.queries, scene.key_dim),
                (rows, scene.heads, scene.key_dim),
                device,
            )
            .unwrap();
            let profiles = Tensor::from_vec(
                slice(&inputs.profiles, scene.extent),
                (rows, scene.heads, scene.extent),
                device,
            )
            .unwrap();
            let sinks =
                Tensor::from_vec(slice(&inputs.sinks, 1), (rows, scene.heads), device).unwrap();
            let batch = KvDecodeBatchMetadata {
                sequences: tables
                    .iter()
                    .map(|(pages, offset, context)| KvSequenceBlockTable {
                        blocks: pages.iter().map(|page| block(stored.id, *page)).collect(),
                        first_page_offset: *offset,
                        context_len: *context,
                    })
                    .collect(),
            };
            Self {
                queries,
                profiles,
                sinks,
                batch,
                span,
            }
        }

        /// Submits the call; the output stays on the device.
        pub(crate) fn run(&self, scene: &Scene, stored: &Stored) -> Tensor {
            stored
                .arena
                .paged_decode_biased(
                    LAYER,
                    PagedKvBiasedDecodeArgs {
                        queries: &self.queries,
                        profiles: &self.profiles,
                        sinks: &self.sinks,
                        batch: &self.batch,
                        queries_per_sequence: self.span as u32,
                        softmax_scale: scene.scale(),
                        window_tokens: scene.window,
                    },
                )
                .unwrap()
        }
    }

    pub(crate) fn decode_scene(
        scene: &Scene,
        stored: &Stored,
        device: &Device,
        inputs: &Inputs,
    ) -> Vec<f32> {
        decode(scene, stored, device, &scene.tables, scene.span, 0, inputs)
    }

    /// The table of row `row` alone: its sequence's table cut to the row's
    /// own context.
    pub(crate) fn table_of_row(scene: &Scene, row: usize) -> Table {
        let (pages, offset, context) = &scene.tables[row / scene.span];
        let context = context - (scene.span - 1 - row % scene.span) as u32;
        let needed = (offset + context).div_ceil(scene.page_tokens as u32) as usize;
        (pages[..needed].to_vec(), *offset, context)
    }

    /// The formula over dense keys and values in F64: every visible key's
    /// score, one softmax over the scores and the sink, the weighted values.
    pub(crate) fn dense(scene: &Scene, keys: &[f32], values: &[f32], inputs: &Inputs) -> Vec<f64> {
        let group = scene.heads / scene.kv_heads;
        let mut output = Vec::with_capacity(scene.rows() * scene.heads * scene.value_dim);
        for row in 0..scene.rows() {
            let (pages, offset, context) = table_of_row(scene, row);
            // Gather the row's context oldest first.
            let slots = (0..context as usize)
                .map(|token| {
                    let physical = offset as usize + token;
                    pages[physical / scene.page_tokens] as usize * scene.page_tokens
                        + physical % scene.page_tokens
                })
                .collect::<Vec<_>>();
            for head in 0..scene.heads {
                let row_head = row * scene.heads + head;
                let kv_head = head / group;
                let query = &inputs.queries[row_head * scene.key_dim..][..scene.key_dim];
                let mut scores = Vec::new();
                let mut visible = Vec::new();
                for (token, slot) in slots.iter().enumerate() {
                    let distance = slots.len() - 1 - token;
                    if scene
                        .window
                        .is_some_and(|window| distance >= window as usize)
                    {
                        continue;
                    }
                    let kv_row = slot * scene.kv_heads + kv_head;
                    let key = &keys[kv_row * scene.key_dim..][..scene.key_dim];
                    let dot = query
                        .iter()
                        .zip(key)
                        .map(|(q, k)| f64::from(*q) * f64::from(*k))
                        .sum::<f64>();
                    let bias = if distance < scene.extent {
                        f64::from(inputs.profiles[row_head * scene.extent + distance])
                    } else {
                        0.0
                    };
                    scores.push(dot * f64::from(scene.scale()) + bias);
                    visible.push(kv_row);
                }
                let sink = f64::from(inputs.sinks[row_head]);
                let top = scores.iter().copied().fold(sink, f64::max);
                let total =
                    (sink - top).exp() + scores.iter().map(|s| (s - top).exp()).sum::<f64>();
                for dim in 0..scene.value_dim {
                    output.push(
                        scores
                            .iter()
                            .zip(&visible)
                            .map(|(score, kv_row)| {
                                (score - top).exp() / total
                                    * f64::from(values[kv_row * scene.value_dim + dim])
                            })
                            .sum::<f64>(),
                    );
                }
            }
        }
        output
    }

    /// `max |got - want| / max |want|`, infinite for a non-finite difference.
    pub(crate) fn worst_ratio(got: &[f32], want: &[f64]) -> f64 {
        assert_eq!(got.len(), want.len());
        let peak = want
            .iter()
            .fold(1e-6f64, |peak, value| peak.max(value.abs()));
        got.iter().zip(want).fold(0f64, |worst, (got, want)| {
            let error = (f64::from(*got) - want).abs() / peak;
            if error.is_finite() {
                worst.max(error)
            } else {
                f64::INFINITY
            }
        })
    }

    pub(crate) fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    /// Every scene on `device` with `dtype` storage: within [`TOLERANCE`] of
    /// the dense evaluation and of the CPU reference over the same stored
    /// values, and each row alone has the bits it has in the batch.
    pub(crate) fn check_device(device: &Device, dtype: DType) {
        for scene in scenes() {
            let stored = store(&scene, device, dtype, None);
            let inputs = inputs(&scene);
            let got = decode_scene(&scene, &stored, device, &inputs);
            let exact = dense(&scene, &stored.keys, &stored.values, &inputs);
            let versus_dense = worst_ratio(&got, &exact);

            // The CPU reference over the values the device arena stores.
            let cpu = store(
                &scene,
                &Device::Cpu,
                DType::F32,
                Some((&stored.keys, &stored.values)),
            );
            assert_eq!(cpu.keys, stored.keys);
            let reference = decode_scene(&scene, &cpu, &Device::Cpu, &inputs);
            let reference_f64 = reference.iter().map(|v| f64::from(*v)).collect::<Vec<_>>();
            let versus_cpu = worst_ratio(&got, &reference_f64);
            let differing = bits(&got)
                .iter()
                .zip(bits(&reference))
                .filter(|(got, want)| **got != *want)
                .count();
            eprintln!(
                "paged bias/sink {:?} {dtype:?} `{}`: {:.3e} from dense, {:.3e} from the CPU \
                 reference, {differing} of {} values differ in bits",
                device.location(),
                scene.name,
                versus_dense,
                versus_cpu,
                got.len()
            );
            assert!(
                versus_dense <= TOLERANCE,
                "{}: {versus_dense:e}",
                scene.name
            );
            assert!(versus_cpu <= TOLERANCE, "{}: {versus_cpu:e}", scene.name);

            let row_len = scene.heads * scene.value_dim;
            for row in 0..scene.rows() {
                let alone = decode(
                    &scene,
                    &stored,
                    device,
                    &[table_of_row(&scene, row)],
                    1,
                    row,
                    &inputs,
                );
                assert_eq!(
                    bits(&alone),
                    bits(&got[row * row_len..(row + 1) * row_len]),
                    "{}: row {row} alone differs from the batch",
                    scene.name
                );
            }
        }
    }

    #[test]
    fn cpu_matches_a_dense_evaluation_and_rows_are_batch_invariant() {
        check_device(&Device::Cpu, DType::F32);
    }

    /// The CPU arena reads F16 and BF16 storage as the accelerators do.
    #[test]
    fn cpu_reads_half_width_storage() {
        for dtype in [DType::F16, DType::BF16] {
            let scene = scenes().remove(1);
            let stored = store(&scene, &Device::Cpu, dtype, None);
            let inputs = inputs(&scene);
            let got = decode_scene(&scene, &stored, &Device::Cpu, &inputs);
            let exact = dense(&scene, &stored.keys, &stored.values, &inputs);
            assert!(worst_ratio(&got, &exact) <= TOLERANCE);
            let widened = store(
                &scene,
                &Device::Cpu,
                DType::F32,
                Some((&stored.keys, &stored.values)),
            );
            assert_eq!(
                bits(&got),
                bits(&decode_scene(&scene, &widened, &Device::Cpu, &inputs))
            );
        }
    }

    /// A query has the same bits as a decode row whose table starts at its
    /// own window and as a prefill row whose table starts pages earlier.
    #[test]
    fn a_row_does_not_depend_on_where_its_table_starts() {
        let scene = Scene {
            window: Some(20),
            extent: 20,
            ..scenes().remove(0)
        };
        let stored = store(&scene, &Device::Cpu, DType::F32, None);
        let inputs = inputs(&scene);
        let got = decode_scene(&scene, &stored, &Device::Cpu, &inputs);
        // Row 0: 150 tokens after offset 9. Its last 20 tokens start at
        // physical token 139, page 8 of its table, offset 11.
        let (pages, _, _) = &scene.tables[0];
        let windowed = (pages[8..].to_vec(), 11, 20);
        let alone = decode(&scene, &stored, &Device::Cpu, &[windowed], 1, 0, &inputs);
        let row_len = scene.heads * scene.value_dim;
        assert_eq!(bits(&alone), bits(&got[..row_len]));
    }

    /// With one key and a dominant sink the output vanishes; with a
    /// negligible sink it is the key's value.
    #[test]
    fn only_the_sink_competes_with_a_single_key() {
        let scene = scenes().remove(5);
        let stored = store(&scene, &Device::Cpu, DType::F32, None);
        let mut inputs = inputs(&scene);
        let row_len = scene.heads * scene.value_dim;
        inputs.sinks.fill(80.0);
        let drowned = decode_scene(&scene, &stored, &Device::Cpu, &inputs);
        assert!(drowned.iter().all(|value| value.abs() < 1e-20));
        inputs.sinks.fill(-80.0);
        let alone = decode_scene(&scene, &stored, &Device::Cpu, &inputs);
        // Row 0 reads physical page 4, offset 15; head h reads KV head h / 3.
        let slot = 4 * scene.page_tokens + 15;
        for head in 0..scene.heads {
            let kv_row = slot * scene.kv_heads + head / 3;
            assert_eq!(
                alone[head * scene.value_dim..(head + 1) * scene.value_dim],
                stored.values[kv_row * scene.value_dim..][..scene.value_dim]
            );
        }
        assert_eq!(alone.len(), 2 * row_len);
    }

    #[test]
    fn shapes_and_tables_the_op_does_not_take_are_refused() {
        let scene = scenes().remove(3);
        let stored = store(&scene, &Device::Cpu, DType::F32, None);
        let inputs = inputs(&scene);
        let call = |queries: &Tensor, span: u32, window: Option<u32>, scale: f32| {
            let rows = scene.rows();
            let profiles =
                Tensor::zeros((rows, scene.heads, scene.extent), DType::F32, &Device::Cpu).unwrap();
            let sinks = Tensor::zeros((rows, scene.heads), DType::F32, &Device::Cpu).unwrap();
            let batch = KvDecodeBatchMetadata {
                sequences: scene
                    .tables
                    .iter()
                    .map(|(pages, offset, context)| KvSequenceBlockTable {
                        blocks: pages.iter().map(|page| block(stored.id, *page)).collect(),
                        first_page_offset: *offset,
                        context_len: *context,
                    })
                    .collect(),
            };
            stored.arena.paged_decode_biased(
                LAYER,
                PagedKvBiasedDecodeArgs {
                    queries,
                    profiles: &profiles,
                    sinks: &sinks,
                    batch: &batch,
                    queries_per_sequence: span,
                    softmax_scale: scale,
                    window_tokens: window,
                },
            )
        };
        let shape = (scene.rows(), scene.heads, scene.key_dim);
        let queries = Tensor::from_vec(inputs.queries.clone(), shape, &Device::Cpu).unwrap();
        assert!(call(&queries, 1, None, 0.5).is_ok());
        assert!(call(&queries, 1, Some(0), 0.5).is_err(), "a zero window");
        assert!(call(&queries, 1, None, f32::NAN).is_err(), "a NaN scale");
        // Four rows over four tables cannot be two queries per table, and
        // table 1 holds one token.
        assert!(call(&queries, 2, None, 0.5).is_err());
        let half = queries.to_dtype(DType::F16).unwrap();
        assert!(call(&half, 1, None, 0.5).is_err(), "queries must be F32");
        let wide = Tensor::zeros(
            (4, scene.heads, scene.key_dim + 1),
            DType::F32,
            &Device::Cpu,
        );
        assert!(call(&wide.unwrap(), 1, None, 0.5).is_err());
        assert_eq!(lanes(240), Some(32));
        assert_eq!(lanes(256), Some(16));
        assert_eq!(lanes(MAX_VALUE_DIM + 1), None);
    }

    /// Writes every scene for `scripts/cuda-host-emulation/paged_bias_sink.cpp`,
    /// which replays them through the CUDA kernel bodies and requires this
    /// reference's bits. Keys and values are multiples of 1/32, exact in
    /// F32, F16 and BF16, so one expected output serves the three kernels.
    #[test]
    #[ignore = "writes emulator cases to IZWI_PAGED_BIAS_SINK_EMULATOR_DIR"]
    fn writes_the_cuda_emulator_cases() {
        let Ok(dir) = std::env::var("IZWI_PAGED_BIAS_SINK_EMULATOR_DIR") else {
            return;
        };
        let scenes = scenes();
        let mut words: Vec<u32> = vec![0x3153_4250, scenes.len() as u32];
        for scene in &scenes {
            let tables = scene.tables.len();
            let max_blocks = scene.tables.iter().map(|t| t.0.len()).max().unwrap();
            let mut metadata = vec![0u32; tables * (2 + max_blocks)];
            for (index, (pages, offset, context)) in scene.tables.iter().enumerate() {
                metadata[index] = *context;
                metadata[tables + index] = *offset;
                metadata[2 * tables + index * max_blocks..][..pages.len()].copy_from_slice(pages);
            }
            let geometry = Geometry {
                rows: scene.rows(),
                queries_per_table: scene.span,
                heads: scene.heads,
                kv_heads: scene.kv_heads,
                key_dim: scene.key_dim,
                value_dim: scene.value_dim,
                extent: scene.extent,
                page_tokens: scene.page_tokens,
                max_blocks,
                capacity_pages: scene.pages,
                window: scene.window,
                scale: scene.scale(),
            };
            let lanes = geometry.validate(&metadata).unwrap();
            let coarse = |len: usize, seed: u64| {
                xorshift_values(len, seed, 1.0)
                    .into_iter()
                    .map(|value| (value * 32.0).round() / 32.0)
                    .collect::<Vec<_>>()
            };
            let slots = scene.pages * scene.page_tokens * scene.kv_heads;
            let keys = coarse(slots * scene.key_dim, 301);
            let values = coarse(slots * scene.value_dim, 302);
            let inputs = inputs(scene);
            let expected = reference(
                &geometry,
                &inputs.queries,
                &inputs.profiles,
                &inputs.sinks,
                &keys,
                &values,
                &metadata,
            )
            .unwrap();
            words.extend(
                [
                    geometry.rows,
                    scene.span,
                    scene.heads,
                    scene.kv_heads,
                    scene.page_tokens,
                    max_blocks,
                    scene.key_dim,
                    scene.value_dim,
                    scene.extent,
                    scene.pages,
                    scene.window.unwrap_or(0) as usize,
                    lanes,
                ]
                .map(|value| value as u32),
            );
            words.push(geometry.scale.to_bits());
            words.extend_from_slice(&metadata);
            for floats in [
                &inputs.queries,
                &inputs.profiles,
                &inputs.sinks,
                &keys,
                &values,
                &expected,
            ] {
                words.extend(bits(floats));
            }
        }
        let bytes = words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        let path = std::path::Path::new(&dir).join("paged_bias_sink_cases.bin");
        std::fs::write(&path, bytes).unwrap();
        eprintln!("wrote {} cases to {}", scenes.len(), path.display());
    }

    #[test]
    fn the_self_check_passes_on_the_cpu() {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let check = self_check(&Device::Cpu, dtype).unwrap();
            assert_eq!(check.failure, None);
            assert_eq!(check.differing, 0);
            assert_eq!(check.values, 2 * 3 * 6 * 128);
        }
    }
}

/// On a Metal device: the kernel through an accelerator arena against the
/// dense evaluation and the CPU reference.
#[cfg(all(test, feature = "metal"))]
mod metal_tests {
    use super::tests::check_device;
    use super::*;

    fn metal(test: &str) -> Option<(Device, impl Sized)> {
        crate::backends::metal_test_device_or_skip(test)
    }

    #[test]
    fn metal_f32_storage_matches_the_cpu_reference_and_rows_are_batch_invariant() {
        let Some((device, _lease)) = metal("paged bias/sink attention, F32 storage") else {
            return;
        };
        check_device(&device, DType::F32);
    }

    /// The CPU reference reads the F16-rounded keys and values.
    #[test]
    fn metal_f16_storage_matches_the_cpu_reference_and_rows_are_batch_invariant() {
        let Some((device, _lease)) = metal("paged bias/sink attention, F16 storage") else {
            return;
        };
        check_device(&device, DType::F16);
    }

    /// Rough cost of a call at the published geometry (24 heads over 8 KV
    /// heads, 128 dims, F16 storage) with its operands already on the
    /// device: read back after every call, and 48 calls (one per layer of a
    /// deep stack) queued before one readback. The CPU arena runs the same
    /// scene; its figure is only meaningful in an optimized build.
    #[test]
    #[ignore = "timing; run with --ignored --nocapture"]
    fn metal_timing_at_the_published_geometry() {
        use super::tests::{inputs, store, Call, Scene};
        use crate::kernels::numeric_check::host_f32;

        let Some((device, _lease)) = metal("paged bias/sink attention timing") else {
            return;
        };
        for (context, span) in [(1_024u32, 1usize), (8_192, 1), (1_024, 64)] {
            let pages = (context as usize).div_ceil(16);
            let scene = Scene {
                name: "timing",
                heads: 24,
                kv_heads: 8,
                key_dim: 128,
                value_dim: 128,
                extent: 1_024,
                page_tokens: 16,
                pages,
                tables: vec![((0..pages as u32).rev().collect(), 0, context)],
                span,
                window: None,
            };
            let inputs = inputs(&scene);
            for (target, dtype) in [(&device, DType::F16), (&Device::Cpu, DType::F32)] {
                let stored = store(&scene, target, dtype, None);
                let call = Call::new(&scene, &stored, target, &scene.tables, span, 0, &inputs);
                host_f32(&call.run(&scene, &stored)).unwrap();
                let calls = 48;
                let started = std::time::Instant::now();
                for _ in 0..calls {
                    host_f32(&call.run(&scene, &stored)).unwrap();
                }
                let synced = started.elapsed().as_secs_f64() * 1e3 / calls as f64;
                let started = std::time::Instant::now();
                let outputs = (0..calls)
                    .map(|_| call.run(&scene, &stored))
                    .collect::<Vec<_>>();
                host_f32(outputs.last().unwrap()).unwrap();
                let queued = started.elapsed().as_secs_f64() * 1e3 / calls as f64;
                eprintln!(
                    "paged bias/sink timing {:?} {dtype:?}: context {context}, {span} queries: \
                     {synced:.3} ms per call read back each time, {queued:.3} ms per call with \
                     {calls} queued",
                    target.location(),
                );
            }
        }
    }

    #[test]
    fn metal_passes_the_self_check() {
        let Some((device, _lease)) = metal("paged bias/sink attention self-check") else {
            return;
        };
        for dtype in [DType::F32, DType::F16] {
            let check = self_check(&device, dtype).unwrap();
            eprintln!("paged bias/sink Metal self-check {dtype:?}: {check:?}");
            assert_eq!(check.failure, None);
        }
    }
}

/// On a CUDA device: the kernel through an accelerator arena against the
/// dense evaluation and the CPU reference.
#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::tests::check_device;
    use super::*;

    #[test]
    fn cuda_f32_storage_matches_the_cpu_reference_and_rows_are_batch_invariant() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        check_device(&device, DType::F32);
    }

    /// The CPU reference reads the rounded keys and values.
    #[test]
    fn cuda_half_width_storage_matches_the_cpu_reference_and_rows_are_batch_invariant() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        check_device(&device, DType::BF16);
        check_device(&device, DType::F16);
    }

    #[test]
    fn cuda_passes_the_self_check() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let check = self_check(&device, dtype).unwrap();
            eprintln!("paged bias/sink CUDA self-check {dtype:?}: {check:?}");
            assert_eq!(check.failure, None);
        }
    }
}
