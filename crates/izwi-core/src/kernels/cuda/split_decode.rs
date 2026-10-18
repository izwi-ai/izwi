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
//! On by default: `IZWI_CUDA_SPLIT_DECODE=0` turns it off, and so does
//! `IZWI_KV_DISABLE_OPTIMIZED_PROVIDER`. The first use of each geometry runs
//! [`self_check`] and compares one live call against the current provider;
//! the caller keeps the provider on any failure.
use candle_core::{DType, Device, Result, Tensor};

/// Kill switch (any false boolean, e.g. `0` or `off`), read once per KV arena.
pub const SPLIT_DECODE_ENV: &str = "IZWI_CUDA_SPLIT_DECODE";

const MAX_GROUP: usize = 8;
const MAX_PARTITIONS: usize = 1024;
/// One 32-key tile per warp of a 4-warp block.
const MIN_PARTITION_TOKENS: usize = 128;
/// About four resident blocks per SM on a 132-SM H100. With at most 8 query
/// heads per KV head this also bounds the F32 partials to about 8 MiB until
/// the batch alone fills the GPU.
const TARGET_BLOCKS: usize = 512;

/// Whether the switch leaves the kernel on (the default).
pub fn enabled() -> bool {
    crate::env::flag_or(SPLIT_DECODE_ENV, true)
}

/// Whether the kernel covers this geometry: F16/BF16 queries and KV, equal key
/// and value head dims of 128 or 256, and at most 8 query heads per KV head.
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
        && (1..=MAX_GROUP).contains(&(query_heads / kv_heads))
}

/// Partitions for the longest row: enough blocks to fill the GPU, but at least
/// one key tile per warp.
pub fn partitions(batch: usize, kv_heads: usize, max_context: usize) -> usize {
    let by_context = max_context.div_ceil(MIN_PARTITION_TOKENS);
    let by_occupancy = TARGET_BLOCKS.div_ceil(batch.saturating_mul(kv_heads).max(1));
    by_context.min(by_occupancy).clamp(1, MAX_PARTITIONS)
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
                let partition_name = match head_dim {
                    128 => concat!("physical_paged_decode_split_", $suffix, "_d128"),
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
        assert!(!supported(DType::BF16, 64, 4, 128, 128), "group of 16");
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
