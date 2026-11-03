//! Grouped-query split decode attention over the paged KV arena.
//!
//! FlashAttention-2 without splits and the native one-pass kernel run one block
//! per query head over the whole context, so B = 1 decode occupies Hq SMs and
//! attention time grows linearly with context from the first token. This
//! kernel gives each block one (row, KV head, partition): it reads every K/V
//! row of its token range once for all query heads sharing that KV head, and
//! a second kernel merges the partitions. It reads the native metadata layout
//! `[context[B] | first_page_offset[B] | block_table[B x max_blocks]]`, and
//! derives token ranges from the device context lengths.
//!
//! How many partitions a row gets is the caller's choice
//! ([`SplitDecodePartitioning`]). [`partitions`] sizes them to fill the GPU,
//! from the batch size and the batch's longest row, so a row's partition
//! count, token ranges and merge depend on the rows beside it. The opt-in
//! [`SplitDecodePartitioning::RowLength`] cuts every row into partitions of
//! [`ROW_PARTITION_TOKENS`] tokens of its own context instead, so a row's
//! output is the same whatever it is batched with (batch-invariant decode).
//!
//! On by default: `IZWI_CUDA_SPLIT_DECODE=0` turns it off, and so does
//! `IZWI_KV_DISABLE_OPTIMIZED_PROVIDER`. The first use of each geometry runs
//! [`self_check`] and compares one live call against the current provider;
//! the caller keeps the provider on any failure.
use candle_core::{DType, Device, Result, Tensor};

/// Kill switch (any false boolean, e.g. `0` or `off`), read once per KV arena.
pub const SPLIT_DECODE_ENV: &str = "IZWI_CUDA_SPLIT_DECODE";

const MAX_GROUP: usize = 8;
/// The wide-group kernel's cap: 16 query heads per KV head, at head dim 128.
const WIDE_GROUP: usize = 16;
const MAX_PARTITIONS: usize = 1024;
/// One 32-key tile per warp of a 4-warp block.
const MIN_PARTITION_TOKENS: usize = 128;
/// About four resident blocks per SM on a 132-SM H100. With at most 16 query
/// heads per KV head this also bounds the F32 partials to about 16 MiB until
/// the batch alone fills the GPU.
const TARGET_BLOCKS: usize = 512;

/// Whether the switch leaves the kernel on (the default).
pub fn enabled() -> bool {
    crate::env::flag_or(SPLIT_DECODE_ENV, true)
}

/// Whether the kernel covers this geometry: F16/BF16 queries and KV, equal key
/// and value head dims of 128 or 256, and at most 8 query heads per KV head
/// (16 at head dim 128).
pub fn supported(
    dtype: DType,
    query_heads: usize,
    kv_heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> bool {
    matches!(dtype, DType::F16 | DType::BF16)
        && key_dim == value_dim
        && matches!(key_dim, 128 | 256)
        && kv_heads > 0
        && query_heads.is_multiple_of(kv_heads)
        && (1..=group_cap(key_dim)).contains(&(query_heads / kv_heads))
}

/// The largest query-head group a head dim's kernels serve.
fn group_cap(head_dim: usize) -> usize {
    if head_dim == 128 {
        WIDE_GROUP
    } else {
        MAX_GROUP
    }
}

/// Partitions for the longest row: enough blocks to fill the GPU, but at least
/// one key tile per warp.
pub fn partitions(batch: usize, kv_heads: usize, max_context: usize) -> usize {
    let by_context = max_context.div_ceil(MIN_PARTITION_TOKENS);
    let by_occupancy = TARGET_BLOCKS.div_ceil(batch.saturating_mul(kv_heads).max(1));
    by_context.min(by_occupancy).clamp(1, MAX_PARTITIONS)
}

/// How a split decode divides each row's context into partitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SplitDecodePartitioning {
    /// [`partitions`]`(batch, kv_heads, longest row)` partitions for every
    /// row, enough blocks to fill the GPU: a row's partition count, token
    /// ranges and merge depend on its batch.
    #[default]
    Occupancy,
    /// Partitions of [`ROW_PARTITION_TOKENS`] tokens over each row's own
    /// context (at most 1,024 partitions, after which each grows): a row's
    /// partitions, their token ranges and their merge come from its own
    /// length only, so its output does not depend on the rows beside it.
    RowLength,
}

/// A [`SplitDecodePartitioning::RowLength`] partition's tokens: one key tile
/// per warp of the 4-warp block.
pub const ROW_PARTITION_TOKENS: usize = MIN_PARTITION_TOKENS;

/// The launch's partition count and the per-row partition length the
/// kernel takes (`min_partition_tokens`, 0 for "every row uses them all")
/// under `mode`, for a batch whose longest row holds `max_context` tokens.
/// Under [`SplitDecodePartitioning::RowLength`] the launch has room for the
/// longest row's own count, so every row uses exactly
/// [`row_partitions`]`(its length)`.
pub fn launch_partitions(
    mode: SplitDecodePartitioning,
    batch: usize,
    kv_heads: usize,
    max_context: usize,
) -> (usize, usize) {
    match mode {
        SplitDecodePartitioning::Occupancy => (partitions(batch, kv_heads, max_context), 0),
        SplitDecodePartitioning::RowLength => (
            max_context
                .div_ceil(ROW_PARTITION_TOKENS)
                .clamp(1, MAX_PARTITIONS),
            ROW_PARTITION_TOKENS,
        ),
    }
}

/// The partitions a row of `context_len` tokens uses in a launch of
/// `num_partitions` with `min_partition_tokens` (the kernel's
/// `izwi_split_decode_partitions`): all of them when that is 0, else one per
/// `min_partition_tokens` of its own context, at least one.
pub fn row_partitions(
    context_len: usize,
    num_partitions: usize,
    min_partition_tokens: usize,
) -> usize {
    if min_partition_tokens == 0 {
        return num_partitions;
    }
    context_len
        .div_ceil(min_partition_tokens)
        .min(num_partitions)
        .max(1)
}

/// `[batch, query_heads, head_dim]` attention output for one decode token per
/// row. `metadata` must already be validated against the arena, as the
/// accelerator's decode plan does.
#[allow(clippy::too_many_arguments)]
pub fn paged_decode(
    queries: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    metadata: &Tensor,
    batch: usize,
    query_heads: usize,
    kv_heads: usize,
    page_tokens: usize,
    max_blocks: usize,
    softmax_scale: f32,
    softcap: Option<f32>,
    partitions: usize,
) -> Result<Tensor> {
    paged_decode_with(
        queries,
        keys,
        values,
        metadata,
        [batch, query_heads, kv_heads, page_tokens, max_blocks],
        softmax_scale,
        softcap,
        partitions,
        0,
    )
}

/// [`paged_decode`] with one launch geometry for every context length: each
/// row reads its length from `metadata` on the device and uses one partition
/// per 128 tokens, up to `max_partitions`. Nothing depends on host-side
/// lengths, so the launch can be captured in a CUDA graph and replayed as the
/// context grows. Size `max_partitions` with [`partitions`] at the longest
/// context the graph serves.
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_adaptive(
    queries: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    metadata: &Tensor,
    batch: usize,
    query_heads: usize,
    kv_heads: usize,
    page_tokens: usize,
    max_blocks: usize,
    softmax_scale: f32,
    softcap: Option<f32>,
    max_partitions: usize,
) -> Result<Tensor> {
    paged_decode_with(
        queries,
        keys,
        values,
        metadata,
        [batch, query_heads, kv_heads, page_tokens, max_blocks],
        softmax_scale,
        softcap,
        max_partitions,
        MIN_PARTITION_TOKENS,
    )
}

/// [`paged_decode`] partitioned by `mode` for a batch whose longest row
/// holds `max_context` tokens ([`launch_partitions`]).
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_partitioned(
    queries: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    metadata: &Tensor,
    batch: usize,
    query_heads: usize,
    kv_heads: usize,
    page_tokens: usize,
    max_blocks: usize,
    softmax_scale: f32,
    softcap: Option<f32>,
    mode: SplitDecodePartitioning,
    max_context: usize,
) -> Result<Tensor> {
    let (partitions, min_partition_tokens) = launch_partitions(mode, batch, kv_heads, max_context);
    paged_decode_with(
        queries,
        keys,
        values,
        metadata,
        [batch, query_heads, kv_heads, page_tokens, max_blocks],
        softmax_scale,
        softcap,
        partitions,
        min_partition_tokens,
    )
}

#[allow(clippy::too_many_arguments)]
fn paged_decode_with(
    queries: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    metadata: &Tensor,
    [batch, query_heads, kv_heads, page_tokens, max_blocks]: [usize; 5],
    softmax_scale: f32,
    softcap: Option<f32>,
    partitions: usize,
    min_partition_tokens: usize,
) -> Result<Tensor> {
    let head_dim = queries.dim(candle_core::D::Minus1)?;
    let dtype = queries.dtype();
    let location = queries.device().location();
    if !supported(dtype, query_heads, kv_heads, head_dim, head_dim)
        || [keys, values, metadata]
            .iter()
            .any(|t| t.device().location() != location)
        || queries.dims() != [batch, query_heads, head_dim]
        || keys.dtype() != dtype
        || values.dtype() != dtype
        || keys.dims().len() != 4
        || keys.dims()[1..] != [page_tokens, kv_heads, head_dim]
        || values.dims() != keys.dims()
        || metadata.dtype() != DType::U32
        || metadata.dims() != [batch * (2 + max_blocks)]
        || batch == 0
        || page_tokens == 0
        || max_blocks == 0
        || !(1..=MAX_PARTITIONS).contains(&partitions)
        || !softmax_scale.is_finite()
        || softmax_scale <= 0.0
        || softcap.is_some_and(|cap| !cap.is_finite() || cap <= 0.0)
    {
        candle_core::bail!(
            "invalid split decode contract: queries {:?} {dtype:?}, keys {:?} {:?}, values {:?}, metadata {:?}, \
             {query_heads}/{kv_heads} heads, {page_tokens}-token pages, {max_blocks} blocks, {partitions} partitions",
            queries.dims(),
            keys.dims(),
            keys.dtype(),
            values.dims(),
            metadata.dims()
        )
    }
    if [keys.elem_count(), metadata.elem_count()]
        .iter()
        .any(|&count| count > i32::MAX as usize)
    {
        candle_core::bail!("split decode exceeds the signed 32-bit kernel index ABI")
    }
    #[cfg(feature = "cuda")]
    if queries.device().is_cuda() {
        return cuda_impl::launch(
            queries,
            keys,
            values,
            metadata,
            [
                batch,
                query_heads,
                kv_heads,
                page_tokens,
                max_blocks,
                head_dim,
                partitions,
                min_partition_tokens,
            ],
            softmax_scale,
            softcap.unwrap_or(0.0),
        );
    }
    let _ = min_partition_tokens;
    candle_core::bail!("split decode attention runs on CUDA only")
}

/// Fails unless `actual` is within the output dtype's rounding of `expected`,
/// relative to the largest reference magnitude.
pub fn compare(actual: &Tensor, expected: &Tensor) -> Result<()> {
    if actual.dims() != expected.dims() {
        candle_core::bail!(
            "split decode shape {:?} differs from the reference {:?}",
            actual.dims(),
            expected.dims()
        )
    }
    let host =
        |t: &Tensor| -> Result<Vec<f32>> { t.to_dtype(DType::F32)?.flatten_all()?.to_vec1() };
    let (actual_values, expected_values) = (host(actual)?, host(expected)?);
    let max_ref = expected_values.iter().fold(0f32, |m, v| m.max(v.abs()));
    // `f32::max` drops NaN, so a non-finite difference saturates the error.
    let max_err = actual_values
        .iter()
        .zip(&expected_values)
        .map(|(a, e)| (a - e).abs())
        .fold(0f32, |m, d| {
            if d.is_finite() {
                m.max(d)
            } else {
                f32::INFINITY
            }
        });
    let relative = match expected.dtype() {
        DType::BF16 => 1.0 / 64.0,
        _ => 1.0 / 256.0,
    };
    if !max_err.is_finite() || max_err > relative * max_ref.max(1e-3) {
        candle_core::bail!(
            "split decode differs from the reference: max error {max_err} vs max |ref| {max_ref}"
        )
    }
    Ok(())
}

/// Runs a synthetic two-row probe with page offsets, a shuffled block table,
/// a ragged last tile and four partitions, and compares it with the native
/// one-pass kernel.
pub fn self_check(
    device: &Device,
    dtype: DType,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    if !supported(dtype, query_heads, kv_heads, head_dim, head_dim) {
        candle_core::bail!(
            "split decode does not cover {dtype:?} with {query_heads}/{kv_heads} heads of {head_dim}"
        )
    }
    let probe = Probe::new(device, dtype, query_heads, kv_heads, head_dim)?;
    let reference = probe.reference()?;
    // Four fixed partitions, then the adaptive geometry with room for eight:
    // the rows use four and two of them.
    for (launch, partitions) in [
        (paged_decode as Launch, 4),
        (paged_decode_adaptive as Launch, 8),
    ] {
        let actual = launch(
            &probe.queries,
            &probe.keys,
            &probe.values,
            &probe.metadata,
            Probe::CONTEXTS.len(),
            query_heads,
            kv_heads,
            Probe::PAGE_TOKENS,
            probe.max_blocks,
            probe.softmax_scale,
            None,
            partitions,
        )?;
        compare(&actual, &reference)?;
    }
    Ok(())
}

type Launch = fn(
    &Tensor,
    &Tensor,
    &Tensor,
    &Tensor,
    usize,
    usize,
    usize,
    usize,
    usize,
    f32,
    Option<f32>,
    usize,
) -> Result<Tensor>;

struct Probe {
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    metadata: Tensor,
    max_blocks: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    softmax_scale: f32,
}

impl Probe {
    const PAGE_TOKENS: usize = 16;
    const CONTEXTS: [usize; 2] = [421, 150];
    const OFFSETS: [usize; 2] = [5, 11];

    fn wave(n: usize, seed: f32, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 + seed) * 0.754_877_7).sin() * scale)
            .collect()
    }

    fn new(
        device: &Device,
        dtype: DType,
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let rows = Self::CONTEXTS.len();
        let blocks =
            |row: usize| (Self::CONTEXTS[row] + Self::OFFSETS[row]).div_ceil(Self::PAGE_TOKENS);
        let max_blocks = (0..rows).map(blocks).max().unwrap_or(1);
        let capacity = rows * max_blocks + 3;
        // Rows take interleaved pages in reverse, so logical order never
        // matches physical order.
        let mut metadata = Vec::with_capacity(rows * (2 + max_blocks));
        metadata.extend(Self::CONTEXTS.iter().map(|&c| c as u32));
        metadata.extend(Self::OFFSETS.iter().map(|&o| o as u32));
        for row in 0..rows {
            metadata
                .extend((0..max_blocks).map(|block| (capacity - 1 - (block * rows + row)) as u32));
        }
        let kv_dims = [capacity, Self::PAGE_TOKENS, kv_heads, head_dim];
        let kv_len = kv_dims.iter().product();
        let tensor = |values: Vec<f32>, shape: &[usize]| {
            Tensor::from_vec(values, shape, &Device::Cpu)?
                .to_dtype(dtype)?
                .to_device(device)
        };
        Ok(Self {
            queries: tensor(
                Self::wave(rows * query_heads * head_dim, 1.0, 1.0),
                &[rows, query_heads, head_dim],
            )?,
            keys: tensor(Self::wave(kv_len, 2.0, 2.0), &kv_dims)?,
            values: tensor(Self::wave(kv_len, 3.0, 1.0), &kv_dims)?,
            metadata: Tensor::from_vec(metadata, rows * (2 + max_blocks), device)?,
            max_blocks,
            query_heads,
            kv_heads,
            head_dim,
            softmax_scale: 2.0 / (head_dim as f32).sqrt(),
        })
    }

    fn reference(&self) -> Result<Tensor> {
        super::paged_decode_attention(
            &self.queries,
            &self.keys,
            &self.values,
            &self.metadata,
            Self::CONTEXTS.len(),
            self.query_heads,
            self.kv_heads,
            Self::PAGE_TOKENS,
            self.max_blocks,
            self.head_dim,
            self.head_dim,
            self.softmax_scale,
            None,
            Self::CONTEXTS[0],
            None,
        )
    }
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use candle_core::cuda_backend::cudarc::driver::{CudaView, LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::{CudaDType, WrapErr};
    use candle_core::op::BackpropOp;
    use candle_core::{CudaStorage, DType, Result, Storage, Tensor};

    const MODULE: &str = "izwi_physical_state";
    const BLOCK_THREADS: u32 = 128;
    const REDUCE_THREADS: u32 = 256;

    /// Contiguous view. K/V rows load in 16-byte vectors, so their views must
    /// start 16-byte aligned.
    fn view<'a, T: CudaDType>(
        storage: &'a Storage,
        tensor: &Tensor,
        vector_loads: bool,
    ) -> Result<CudaView<'a, T>> {
        let Storage::Cuda(cuda) = storage else {
            candle_core::bail!("split decode inputs must be CUDA tensors")
        };
        let Some((start, end)) = tensor.layout().contiguous_offsets() else {
            candle_core::bail!("split decode inputs must be contiguous")
        };
        if vector_loads && !(start * std::mem::size_of::<T>()).is_multiple_of(16) {
            candle_core::bail!("split decode K/V views must start 16-byte aligned")
        }
        Ok(cuda.as_cuda_slice::<T>()?.slice(start..end))
    }

    fn int(value: usize) -> Result<i32> {
        i32::try_from(value)
            .map_err(|_| candle_core::Error::Msg(format!("split decode argument {value}")))
    }

    pub(super) fn launch(
        queries: &Tensor,
        keys: &Tensor,
        values: &Tensor,
        metadata: &Tensor,
        geometry: [usize; 8],
        softmax_scale: f32,
        softcap: f32,
    ) -> Result<Tensor> {
        let [batch, query_heads, kv_heads, page_tokens, max_blocks, head_dim, partitions, min_partition_tokens] =
            geometry;
        let device = queries.device().as_cuda_device()?;
        let capacity_pages = keys.dims()[0];
        let row_heads = batch * query_heads;
        let queries = queries.contiguous()?;
        let (q_storage, _) = queries.storage_and_layout();
        let (k_storage, _) = keys.storage_and_layout();
        let (v_storage, _) = values.storage_and_layout();
        let (m_storage, _) = metadata.storage_and_layout();
        let meta = view::<u32>(&m_storage, metadata, false)?;
        // SAFETY: the partition kernel writes every partial the reduction
        // reads, and the reduction writes every output element.
        let partials = unsafe { device.alloc::<f32>(row_heads * partitions * (head_dim + 2))? };
        let partition_config = LaunchConfig {
            grid_dim: (
                u32::try_from(batch * kv_heads)
                    .map_err(|_| candle_core::Error::Msg("split decode grid".into()))?,
                partitions as u32,
                1,
            ),
            block_dim: (BLOCK_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let reduce_config = LaunchConfig {
            grid_dim: (
                u32::try_from(row_heads)
                    .map_err(|_| candle_core::Error::Msg("split decode reduce grid".into()))?,
                1,
                1,
            ),
            block_dim: (REDUCE_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        macro_rules! run {
            ($ty:ty, $suffix:literal) => {{
                let q = view::<$ty>(&q_storage, &queries, false)?;
                let k = view::<$ty>(&k_storage, keys, true)?;
                let v = view::<$ty>(&v_storage, values, true)?;
                let partition_name = match (head_dim, query_heads / kv_heads > super::MAX_GROUP) {
                    (128, true) => concat!("physical_paged_decode_split_", $suffix, "_d128_g16"),
                    (128, false) => concat!("physical_paged_decode_split_", $suffix, "_d128"),
                    _ => concat!("physical_paged_decode_split_", $suffix, "_d256"),
                };
                let function = device.get_or_load_custom_func(
                    partition_name,
                    MODULE,
                    super::super::cuda_ptx::PHYSICAL_STATE,
                )?;
                let mut builder = function.builder();
                builder.arg(&q);
                builder.arg(&k);
                builder.arg(&v);
                builder.arg(&meta);
                builder.arg(&partials);
                candle_core::builder_arg!(
                    builder,
                    int(batch)?,
                    int(query_heads)?,
                    int(kv_heads)?,
                    int(page_tokens)?,
                    int(max_blocks)?,
                    int(capacity_pages)?,
                    int(partitions)?,
                    int(min_partition_tokens)?,
                    softmax_scale,
                    softcap
                );
                // SAFETY: argument order and types match the kernel, and the
                // validated metadata bounds every page it reads.
                unsafe { builder.launch(partition_config) }.w()?;

                // SAFETY: written in full by the reduction below.
                let out = unsafe { device.alloc::<$ty>(row_heads * head_dim)? };
                let function = device.get_or_load_custom_func(
                    concat!("physical_paged_decode_split_reduce_", $suffix),
                    MODULE,
                    super::super::cuda_ptx::PHYSICAL_STATE,
                )?;
                let mut builder = function.builder();
                builder.arg(&partials);
                builder.arg(&meta);
                builder.arg(&out);
                candle_core::builder_arg!(
                    builder,
                    int(batch)?,
                    int(query_heads)?,
                    int(head_dim)?,
                    int(partitions)?,
                    int(min_partition_tokens)?
                );
                // SAFETY: the partition launch on the same stream initializes
                // every partial.
                unsafe { builder.launch(reduce_config) }.w()?;
                CudaStorage::wrap_cuda_slice(out, device.clone())
            }};
        }
        let storage = match queries.dtype() {
            DType::BF16 => run!(half::bf16, "bf16"),
            DType::F16 => run!(half::f16, "f16"),
            other => candle_core::bail!("split decode does not support {other:?}"),
        };
        Ok(Tensor::from_storage(
            Storage::Cuda(storage),
            (batch, query_heads, head_dim),
            BackpropOp::none(),
            false,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_geometries_cover_grouped_f16_and_bf16_heads_of_128_and_256() {
        assert!(supported(DType::BF16, 16, 2, 256, 256), "Qwen3.6");
        assert!(supported(DType::F16, 32, 8, 128, 128));
        assert!(supported(DType::F16, 4, 4, 128, 128), "no grouping");
        assert!(!supported(DType::F32, 16, 2, 256, 256));
        assert!(
            !supported(DType::BF16, 16, 2, 256, 128),
            "split key/value dims"
        );
        assert!(!supported(DType::BF16, 16, 2, 64, 64));
        assert!(
            supported(DType::BF16, 32, 2, 128, 128),
            "Nemotron-H: 16 at 128"
        );
        assert!(!supported(DType::BF16, 32, 2, 256, 256), "16 only at 128");
        assert!(!supported(DType::BF16, 64, 2, 128, 128), "group of 32");
        assert!(!supported(DType::BF16, 12, 8, 128, 128), "ragged groups");
    }

    #[test]
    fn partitions_fill_the_gpu_without_splitting_below_a_tile_per_warp() {
        // Qwen3.6 at B = 1 (2 KV heads): one partition per 128 tokens, capped
        // by occupancy.
        assert_eq!(partitions(1, 2, 1), 1);
        assert_eq!(partitions(1, 2, 4_096), 32);
        assert_eq!(partitions(1, 2, 32_768), 256);
        assert_eq!(partitions(1, 2, 1_000_000), 256);
        // Large batches fill the GPU with rows instead.
        assert_eq!(partitions(64, 2, 32_768), 4);
        assert_eq!(partitions(512, 2, 32_768), 1);
        assert_eq!(partitions(1, 1, 1_000_000), 512);
    }

    /// One row of the split algorithm on the host, as the kernels divide
    /// it: `used` partitions of `ceil(L / used)` tokens, each an online
    /// softmax over its tokens in order (max, sum, unnormalized output),
    /// then the reduction's merge. `keys` and `values` are `[L, dim]` for
    /// one KV head, `query` `[dim]`.
    fn split_row(query: &[f32], keys: &[f32], values: &[f32], used: usize, scale: f32) -> Vec<f32> {
        let dim = query.len();
        let len = keys.len() / dim;
        let span = len.div_ceil(used);
        let mut partials = Vec::with_capacity(used);
        for p in 0..used {
            let (start, end) = ((p * span).min(len), (p * span + span).min(len));
            let (mut max, mut sum, mut out) = (f32::NEG_INFINITY, 0f32, vec![0f32; dim]);
            for t in start..end {
                let score = query
                    .iter()
                    .zip(&keys[t * dim..(t + 1) * dim])
                    .fold(0f32, |acc, (q, k)| q.mul_add(*k, acc))
                    * scale;
                let next = max.max(score);
                let rescale = (max - next).exp();
                let weight = (score - next).exp();
                sum = sum * rescale + weight;
                for (o, v) in out.iter_mut().zip(&values[t * dim..(t + 1) * dim]) {
                    *o = o.mul_add(rescale, weight * v);
                }
                max = next;
            }
            partials.push((max, sum, out));
        }
        let global = partials
            .iter()
            .filter(|(_, sum, _)| *sum > 0.0)
            .fold(f32::NEG_INFINITY, |m, (max, ..)| m.max(*max));
        let weights = partials
            .iter()
            .map(|(max, sum, _)| {
                if *sum > 0.0 {
                    (max - global).exp()
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        let total = partials
            .iter()
            .zip(&weights)
            .fold(0f32, |t, ((_, sum, _), w)| t + sum * w);
        (0..dim)
            .map(|d| {
                partials
                    .iter()
                    .zip(&weights)
                    .fold(0f32, |acc, ((.., out), w)| acc + w * out[d])
                    / total
            })
            .collect()
    }

    /// Under [`SplitDecodePartitioning::RowLength`] a row's partitions come
    /// from its own length, so its output is the same alone, beside a longer
    /// row and beside a shorter one; under today's occupancy partitioning
    /// (RED) a longer row beside it changes its partitions and its bits.
    #[test]
    fn row_length_partitions_make_a_rows_output_independent_of_its_batch() {
        let (dim, kv_heads) = (64usize, 2usize);
        let row_len = 700usize;
        let wave = |n: usize, seed: f32| {
            (0..n)
                .map(|i| ((i as f32 + seed) * 0.754_877_7).sin())
                .collect::<Vec<_>>()
        };
        let (query, keys, values) = (
            wave(dim, 1.0),
            wave(row_len * dim, 2.0),
            wave(row_len * dim, 3.0),
        );
        let scale = 1.0 / (dim as f32).sqrt();
        let output = |mode: SplitDecodePartitioning, batch: &[usize]| {
            let longest = batch.iter().copied().max().unwrap();
            let (launch, min_tokens) = launch_partitions(mode, batch.len(), kv_heads, longest);
            let used = row_partitions(row_len, launch, min_tokens);
            let bits = split_row(&query, &keys, &values, used, scale)
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>();
            (used, bits)
        };
        // Alone, beside a longer row, beside a shorter one, in either order.
        let batches: [&[usize]; 5] = [
            &[row_len],
            &[row_len, 3_000],
            &[3_000, row_len],
            &[200, row_len],
            &[row_len, 200, 3_000, 40_000],
        ];
        let alone = output(SplitDecodePartitioning::RowLength, batches[0]);
        assert_eq!(alone.0, row_len.div_ceil(ROW_PARTITION_TOKENS));
        for batch in &batches[1..] {
            assert_eq!(
                output(SplitDecodePartitioning::RowLength, batch),
                alone,
                "{batch:?}"
            );
        }
        // RED: occupancy gives the row 6 partitions alone and 24 beside a
        // 3,000-token row, and the merge changes its bits.
        let occupancy = output(SplitDecodePartitioning::Occupancy, batches[0]);
        let beside = output(SplitDecodePartitioning::Occupancy, batches[1]);
        assert_ne!(occupancy.0, beside.0);
        assert_ne!(
            occupancy.1, beside.1,
            "the RED partitioning must change the row's bits"
        );
        // Every output is the attention itself.
        let reference = split_row(&query, &keys, &values, 1, scale);
        let bits = |values: &[u32]| {
            values
                .iter()
                .map(|&v| f32::from_bits(v))
                .collect::<Vec<_>>()
        };
        for (a, b) in bits(&alone.1).iter().zip(&reference) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        // Existing callers keep their geometry exactly.
        assert_eq!(
            launch_partitions(SplitDecodePartitioning::Occupancy, 3, 2, 4_096),
            (partitions(3, 2, 4_096), 0)
        );
        assert_eq!(
            SplitDecodePartitioning::default(),
            SplitDecodePartitioning::Occupancy
        );
        // Past 1,024 partitions a row's partitions grow, still from its own
        // length.
        assert_eq!(
            launch_partitions(SplitDecodePartitioning::RowLength, 1, 2, 1_000_000),
            (MAX_PARTITIONS, ROW_PARTITION_TOKENS)
        );
        assert_eq!(
            row_partitions(1_000_000, MAX_PARTITIONS, ROW_PARTITION_TOKENS),
            MAX_PARTITIONS
        );
        assert_eq!(row_partitions(1, 7, ROW_PARTITION_TOKENS), 1);
        assert_eq!(row_partitions(300, 7, 0), 7);
    }

    #[test]
    fn compare_accepts_rounding_and_rejects_drift() {
        let expected = Tensor::new(&[1.0f32, -2.0, 0.5], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        compare(&expected, &expected).unwrap();
        let drift = Tensor::new(&[1.0f32, -1.8, 0.5], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        assert!(compare(&drift, &expected).is_err());
        let nan = Tensor::new(&[f32::NAN, -2.0, 0.5], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        assert!(compare(&nan, &expected).is_err());
    }

    /// On the device, a row's output under
    /// [`SplitDecodePartitioning::RowLength`] is the same alone, beside a
    /// longer row and beside a shorter one, at any slot of the batch (the
    /// RED against occupancy partitioning runs on the emulator, which reads
    /// the merge in F32: a BF16 store rounds most of its differences away).
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_row_length_split_decode_is_batch_invariant() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        let (query_heads, kv_heads, head_dim, page_tokens) = (16usize, 2usize, 256usize, 16usize);
        // (context, first-page offset, query seed): the subject, a longer
        // row and a shorter one, each on its own pages handed out in reverse.
        let rows = [(700usize, 5usize, 1.0f32), (3_000, 0, 2.0), (200, 11, 3.0)];
        let mut next = 0u32;
        let tables = rows
            .iter()
            .map(|&(context, offset, _)| {
                let pages = (context + offset).div_ceil(page_tokens) as u32;
                let table = (next..next + pages).rev().collect::<Vec<_>>();
                next += pages;
                table
            })
            .collect::<Vec<_>>();
        let capacity = next as usize;
        let wave = |n: usize, seed: f32, scale: f32| {
            (0..n)
                .map(|i| ((i as f32 + seed) * 0.754_877_7).sin() * scale)
                .collect::<Vec<_>>()
        };
        let kv_dims = (capacity, page_tokens, kv_heads, head_dim);
        let kv_len = capacity * page_tokens * kv_heads * head_dim;
        let cache = |seed: f32, scale: f32| {
            Tensor::from_vec(wave(kv_len, seed, scale), kv_dims, &Device::Cpu)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
                .to_device(&device)
                .unwrap()
        };
        let (keys, values) = (cache(5.0, 2.0), cache(7.0, 1.0));
        let subject_bits = |batch: &[usize]| -> Vec<u32> {
            let max_blocks = batch.iter().map(|&r| tables[r].len()).max().unwrap();
            let mut metadata = batch.iter().map(|&r| rows[r].0 as u32).collect::<Vec<_>>();
            metadata.extend(batch.iter().map(|&r| rows[r].1 as u32));
            for &r in batch {
                let mut table = tables[r].clone();
                table.resize(max_blocks, 0);
                metadata.extend(table);
            }
            let queries = batch
                .iter()
                .flat_map(|&r| wave(query_heads * head_dim, rows[r].2, 1.0))
                .collect::<Vec<_>>();
            let queries =
                Tensor::from_vec(queries, (batch.len(), query_heads, head_dim), &Device::Cpu)
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap()
                    .to_device(&device)
                    .unwrap();
            let metadata =
                Tensor::from_vec(metadata, batch.len() * (2 + max_blocks), &device).unwrap();
            let longest = batch.iter().map(|&r| rows[r].0).max().unwrap();
            let out = paged_decode_partitioned(
                &queries,
                &keys,
                &values,
                &metadata,
                batch.len(),
                query_heads,
                kv_heads,
                page_tokens,
                max_blocks,
                1.0 / (head_dim as f32).sqrt(),
                None,
                SplitDecodePartitioning::RowLength,
                longest,
            )
            .unwrap();
            let slot = batch.iter().position(|&r| r == 0).unwrap();
            out.get(slot)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect()
        };
        let alone = subject_bits(&[0]);
        for batch in [&[0usize, 1][..], &[1, 0], &[2, 0], &[2, 0, 1]] {
            assert_eq!(subject_bits(batch), alone, "{batch:?}");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_split_decode_matches_the_one_pass_kernel() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        for (dtype, query_heads, kv_heads, head_dim) in [
            (DType::BF16, 16, 2, 256),
            (DType::F16, 6, 2, 128),
            (DType::BF16, 4, 4, 128),
        ] {
            self_check(&device, dtype, query_heads, kv_heads, head_dim)
                .unwrap_or_else(|e| panic!("{dtype:?} {query_heads}/{kv_heads}x{head_dim}: {e}"));
        }
    }
}
