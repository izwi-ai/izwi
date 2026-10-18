//! Paged attention for one mixed step: decode rows and prefill spans of
//! several sequences in one flattened token batch.
//!
//! Rows are laid out decode rows first, then prefill rows, each row's tokens
//! contiguous. Every layer writes all tokens' K/V with one `write_slots`, then
//! attends in at most two calls on the same pages: `paged_decode` over the
//! decode prefix (the tuned decode kernels) and `paged_prefill` over the
//! prefill suffix. A pure-decode step is therefore exactly today's batched
//! decode call and a one-row prefill step exactly today's prefill call.
//! Generalised from `fish_s2/batch.rs`, which keeps its own copy.

use std::sync::Arc;

use candle_core::Tensor;

use crate::backends::kv::{
    submit_ordered_after_write, KvSlotMap, KvWriteArgs, KvWriteCompletionCollector,
    PagedKvDecodeArgs, PagedKvPrefillArgs, PagedKvPrefillRow,
};
use crate::error::{Error, Result};
use crate::kv::{KvDecodeBatchMetadata, KvSequenceBlockTable};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

/// One row of a mixed step: `q_len` new tokens appended at `start_pos`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MixedRowSpan {
    pub(crate) start_pos: usize,
    pub(crate) q_len: usize,
}

/// Token layout of a mixed step: `decode_rows` one-token rows first, then
/// prefill rows, each row's tokens contiguous in the flattened batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MixedRowLayout {
    rows: Vec<MixedRowSpan>,
    decode_rows: usize,
    total_tokens: usize,
}

impl MixedRowLayout {
    pub(crate) fn new(rows: Vec<MixedRowSpan>, decode_rows: usize) -> Result<Self> {
        if rows.is_empty() || decode_rows > rows.len() {
            return Err(Error::InvalidInput(
                "mixed step layout needs rows and at most as many decode rows".into(),
            ));
        }
        if rows.iter().any(|row| row.q_len == 0)
            || rows[..decode_rows].iter().any(|row| row.q_len != 1)
        {
            return Err(Error::InvalidInput(
                "mixed step decode rows take one token and prefill rows at least one".into(),
            ));
        }
        let total_tokens = rows
            .iter()
            .try_fold(0usize, |total, row| total.checked_add(row.q_len))
            .ok_or_else(|| Error::InvalidInput("mixed step token count overflow".into()))?;
        Ok(Self {
            rows,
            decode_rows,
            total_tokens,
        })
    }

    pub(crate) fn rows(&self) -> &[MixedRowSpan] {
        &self.rows
    }

    pub(crate) fn decode_rows(&self) -> usize {
        self.decode_rows
    }

    pub(crate) fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Flattened index of `row`'s first token.
    pub(crate) fn token_offset(&self, row: usize) -> usize {
        self.rows[..row].iter().map(|row| row.q_len).sum()
    }

    /// Flattened index of `row`'s last token, the one that samples.
    pub(crate) fn last_token(&self, row: usize) -> usize {
        self.token_offset(row) + self.rows[row].q_len - 1
    }

    fn decode_tokens(&self) -> usize {
        self.decode_rows
    }
}

/// Attention views of one window size: decode tables and prefill rows.
struct MixedWindowViews {
    window: Option<usize>,
    decode: KvDecodeBatchMetadata,
    prefill: Vec<PagedKvPrefillRow>,
}

/// One mixed step lowered once and reused by every layer: the write slots of
/// all tokens, per-window attention views, and the write completions.
pub(crate) struct PreparedPhysicalMixedStep {
    layout: MixedRowLayout,
    slots: Arc<dyn KvSlotMap>,
    views: Vec<MixedWindowViews>,
    completions: KvWriteCompletionCollector,
}

impl PreparedPhysicalMixedStep {
    /// Lower the slots of every row in one call. `caches[i]` is row `i`'s
    /// view; all must share one arena and layer bindings, and each row must
    /// append at its cache's committed context.
    pub(crate) fn new(caches: &[&PhysicalPagedKvCache], layout: MixedRowLayout) -> Result<Self> {
        if caches.len() != layout.rows.len() {
            return Err(Error::InvalidInput(
                "mixed step needs one cache per row".into(),
            ));
        }
        let first = caches[0];
        let layers = first.arena().config().layers.len();
        for cache in &caches[1..] {
            if !Arc::ptr_eq(first.arena(), cache.arena()) {
                return Err(Error::InvalidInput(
                    "mixed step rows must share one physical arena".into(),
                ));
            }
            for layer in 0..layers {
                if cache.layer_binding(layer)? != first.layer_binding(layer)? {
                    return Err(Error::InvalidInput(
                        "mixed step rows have different layer bindings".into(),
                    ));
                }
            }
        }
        let mut logical_slots = Vec::with_capacity(layout.total_tokens);
        for (cache, row) in caches.iter().zip(&layout.rows) {
            logical_slots.extend(cache.slots_for_append(row.start_pos, row.q_len)?);
        }
        let slots = first.arena().lower_slots(&logical_slots)?;
        let completions =
            KvWriteCompletionCollector::new(first.arena().config(), slots.logical_slots())?;
        let views = vec![Self::views(caches, &layout, None)?];
        Ok(Self {
            layout,
            slots,
            views,
            completions,
        })
    }

    pub(crate) fn layout(&self) -> &MixedRowLayout {
        &self.layout
    }

    /// Decode tables and rebased prefill rows for `window` (the causal window
    /// including the query token). A row's table starts at the first token its
    /// first query sees, so one canonical prefill row serves every query.
    fn views(
        caches: &[&PhysicalPagedKvCache],
        layout: &MixedRowLayout,
        window: Option<usize>,
    ) -> Result<MixedWindowViews> {
        let table =
            |cache: &PhysicalPagedKvCache, row: MixedRowSpan| -> Result<KvSequenceBlockTable> {
                let end = row
                    .start_pos
                    .checked_add(row.q_len)
                    .ok_or_else(|| Error::InvalidInput("mixed step context overflow".into()))?;
                match window {
                    None => cache.sequence_table(end),
                    Some(window) => cache.sequence_table_with_window(end, window + row.q_len - 1),
                }
            };
        let mut decode = Vec::with_capacity(layout.decode_rows);
        let mut prefill = Vec::with_capacity(layout.rows.len() - layout.decode_rows);
        let mut query_start = 0u32;
        for (index, (cache, row)) in caches.iter().zip(&layout.rows).enumerate() {
            let table = table(cache, *row)?;
            if index < layout.decode_rows {
                decode.push(table);
                continue;
            }
            let query_len = u32::try_from(row.q_len)
                .map_err(|_| Error::InvalidInput("mixed step query length exceeds u32".into()))?;
            prefill.push(PagedKvPrefillRow {
                blocks: table.blocks,
                first_page_offset: table.first_page_offset,
                query_start,
                query_len,
                context_len: table.context_len,
            });
            query_start = query_start
                .checked_add(query_len)
                .ok_or_else(|| Error::InvalidInput("mixed step query offset overflow".into()))?;
        }
        Ok(MixedWindowViews {
            window,
            decode: KvDecodeBatchMetadata { sequences: decode },
            prefill,
        })
    }

    /// Write layer `layer`'s K/V for every token, then attend. Queries are
    /// `[tokens, query_heads, dim]` and keys/values `[tokens, kv_heads, dim]`
    /// in layout order; the output is `[tokens, query_heads, value_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_and_attend(
        &mut self,
        caches: &[&PhysicalPagedKvCache],
        layer: usize,
        queries: &Tensor,
        keys: &Tensor,
        values: &Tensor,
        softmax_scale: f32,
        softcap: Option<f32>,
        window: Option<usize>,
    ) -> Result<Tensor> {
        let tokens = self.layout.total_tokens;
        if queries.dim(0)? != tokens || keys.dim(0)? != tokens || values.dim(0)? != tokens {
            return Err(Error::InvalidInput(
                "mixed step attention inputs do not match the step's tokens".into(),
            ));
        }
        if caches.len() != self.layout.rows.len() {
            return Err(Error::InvalidInput(
                "mixed step needs one cache per row".into(),
            ));
        }
        if window == Some(0) {
            return Err(Error::InvalidInput(
                "mixed step sliding window cannot be zero".into(),
            ));
        }
        let window_u32 = window
            .map(|window| {
                u32::try_from(window)
                    .map_err(|_| Error::InvalidInput("mixed step window exceeds u32".into()))
            })
            .transpose()?;
        let views = match self.views.iter().position(|views| views.window == window) {
            Some(index) => index,
            None => {
                self.views.push(Self::views(caches, &self.layout, window)?);
                self.views.len() - 1
            }
        };
        let cache = caches[0];
        let binding = cache.layer_binding(layer)?;
        let completion = cache.arena().write_slots(
            binding,
            KvWriteArgs {
                keys,
                values,
                slots: self.slots.as_ref(),
            },
        )?;
        if completion.arena() != cache.arena().id()
            || completion.layer() != binding
            || completion.slots() != tokens
        {
            return Err(Error::InferenceError(
                "mixed step write returned a mismatched backend completion".into(),
            ));
        }
        let views = &self.views[views];
        let decode_tokens = self.layout.decode_tokens();
        let (output, completion) = submit_ordered_after_write(completion, || {
            let decode = (decode_tokens > 0)
                .then(|| {
                    let queries = queries.narrow(0, 0, decode_tokens)?.contiguous()?;
                    cache.arena().paged_decode(
                        binding,
                        PagedKvDecodeArgs {
                            queries: &queries,
                            batch: &views.decode,
                            softmax_scale,
                            softcap,
                        },
                    )
                })
                .transpose()?;
            let prefill = (decode_tokens < tokens)
                .then(|| {
                    let queries = queries
                        .narrow(0, decode_tokens, tokens - decode_tokens)?
                        .contiguous()?;
                    cache.arena().paged_prefill(
                        binding,
                        PagedKvPrefillArgs {
                            queries: &queries,
                            rows: &views.prefill,
                            softmax_scale,
                            softcap,
                            window_tokens: window_u32,
                        },
                    )
                })
                .transpose()?;
            match (decode, prefill) {
                (Some(decode), Some(prefill)) => Ok(Tensor::cat(&[decode, prefill], 0)?),
                (Some(output), None) | (None, Some(output)) => Ok(output),
                (None, None) => Err(Error::InvalidInput("mixed step has no tokens".into())),
            }
        })?;
        self.completions.collect(completion)?;
        Ok(output)
    }

    /// Seal the step's writes and advance every row's cache by its span, or,
    /// when the forward failed, drain the writes and return the error.
    pub(crate) fn finish<T>(
        self,
        caches: &mut [&mut PhysicalPagedKvCache],
        result: Result<T>,
    ) -> Result<T> {
        let output = match result {
            Ok(output) => output,
            Err(error) => {
                return match self.completions.drain() {
                    Ok(()) => Err(error),
                    Err(drain) => Err(Error::InferenceError(format!(
                        "mixed step failed: {error}; write-fence drain failed: {drain}"
                    ))),
                }
            }
        };
        if caches.len() != self.layout.rows.len() {
            return Err(Error::InvalidInput(
                "mixed step needs one cache per row".into(),
            ));
        }
        let completion = Arc::new(self.completions.seal()?);
        for (cache, row) in caches.iter_mut().zip(&self.layout.rows) {
            cache.commit_shared_completion(row.start_pos, row.q_len, completion.clone())?;
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::kv::{
        arena_page_bytes, CpuKvArena, KvArena, KvArenaConfig, KvLayerConfig,
    };
    use crate::backends::BackendKind;
    use crate::engine::ModelInstanceId;
    use crate::kernels::numeric_check::xorshift_values;
    use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
    use candle_core::{DType, Device};

    const LAYERS: usize = 2;
    const HEADS: usize = 4;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 8;
    const PAGE_TOKENS: usize = 16;
    const PAGES_PER_ROW: usize = 4;

    /// Rows (prefix tokens already cached, new tokens): three decode rows, a
    /// fresh prefill and a prefill continuing a 33-token prefix.
    const ROWS: [(usize, usize); 5] = [(5, 1), (20, 1), (40, 1), (0, 5), (33, 17)];
    const DECODE_ROWS: usize = 3;

    /// One arena whose rows own pages in a shuffled order, so a table that
    /// is lowered for the wrong row reads another row's K/V.
    fn shuffled_caches(device: &Device) -> (Arc<dyn KvArena>, Vec<PhysicalPagedKvCache>) {
        let id = KvArenaId {
            model_instance: ModelInstanceId::new(4_243),
            backend: if device.is_metal() {
                BackendKind::Metal
            } else {
                BackendKind::Cpu
            },
            device_ordinal: device.is_metal().then_some(0),
            generation: 1,
        };
        let group = KvGroupId::new(1);
        let bindings = (0..LAYERS as u32)
            .map(|layer| KvLayerBinding {
                model_layer: layer,
                physical_layer: layer,
            })
            .collect::<Vec<_>>();
        let pages = ROWS.len() * PAGES_PER_ROW;
        let config = KvArenaConfig {
            id,
            group,
            page_tokens: PAGE_TOKENS as u32,
            capacity_pages: pages as u32,
            growth: None,
            dtype: DType::F32,
            layers: bindings
                .iter()
                .map(|&binding| KvLayerConfig {
                    binding,
                    num_kv_heads: KV_HEADS as u32,
                    key_head_dim: HEAD_DIM as u32,
                    value_head_dim: HEAD_DIM as u32,
                })
                .collect(),
        };
        let arena: Arc<dyn KvArena> = if device.is_metal() {
            #[cfg(feature = "metal")]
            {
                Arc::new(
                    crate::backends::kv::CandleAcceleratorKvArena::new_mutation_only(
                        config,
                        device.clone(),
                    )
                    .unwrap(),
                )
            }
            #[cfg(not(feature = "metal"))]
            unreachable!("a Metal device without the metal feature")
        } else {
            Arc::new(CpuKvArena::new(config).unwrap())
        };
        // Page p of row r is physical page (7 * (r * PAGES_PER_ROW + p) + 3)
        // mod pages: a fixed permutation interleaving the rows.
        let caches = (0..ROWS.len())
            .map(|row| {
                let blocks = (0..PAGES_PER_ROW)
                    .map(|page| CacheBlockRef {
                        arena: id,
                        group,
                        index: ((7 * (row * PAGES_PER_ROW + page) + 3) % pages) as u32,
                        slot_generation: 1,
                    })
                    .collect();
                PhysicalPagedKvCache::new(arena.clone(), bindings.clone(), blocks, 0).unwrap()
            })
            .collect();
        (arena, caches)
    }

    fn random(shape: (usize, usize, usize), seed: u64, device: &Device) -> Tensor {
        Tensor::from_vec(
            xorshift_values(shape.0 * shape.1 * shape.2, seed, 1.0),
            shape,
            device,
        )
        .unwrap()
    }

    /// Fill each row's prefix with the same K/V in every world.
    fn write_prefixes(caches: &mut [PhysicalPagedKvCache], device: &Device) {
        for (row, (cache, (prefix, _))) in caches.iter_mut().zip(ROWS).enumerate() {
            if prefix == 0 {
                continue;
            }
            let mut prepared = cache.prepare_append(0, prefix).unwrap();
            for layer in 0..LAYERS {
                let seed = 1_000 + 10 * row as u64 + layer as u64;
                let q = random((prefix, HEADS, HEAD_DIM), seed, device);
                let k = random((prefix, KV_HEADS, HEAD_DIM), seed + 100, device);
                let v = random((prefix, KV_HEADS, HEAD_DIM), seed + 200, device);
                cache
                    .write_and_attend(layer, &mut prepared, &q, &k, &v, 0.5)
                    .unwrap();
            }
            cache.commit_prepared(prepared).unwrap();
        }
    }

    struct StepInputs {
        queries: Vec<Tensor>,
        keys: Vec<Tensor>,
        values: Vec<Tensor>,
    }

    fn step_inputs(tokens: usize, device: &Device) -> StepInputs {
        let per_layer = |seed: u64, heads: usize| {
            (0..LAYERS)
                .map(|layer| random((tokens, heads, HEAD_DIM), seed + layer as u64, device))
                .collect::<Vec<_>>()
        };
        StepInputs {
            queries: per_layer(7_000, HEADS),
            keys: per_layer(8_000, KV_HEADS),
            values: per_layer(9_000, KV_HEADS),
        }
    }

    fn layout() -> MixedRowLayout {
        MixedRowLayout::new(
            ROWS.iter()
                .map(|&(start_pos, q_len)| MixedRowSpan { start_pos, q_len })
                .collect(),
            DECODE_ROWS,
        )
        .unwrap()
    }

    /// The mixed step's per-layer outputs.
    fn run_mixed(
        caches: &mut [PhysicalPagedKvCache],
        inputs: &StepInputs,
        softcap: Option<f32>,
        window: Option<usize>,
    ) -> Vec<Tensor> {
        let refs = caches.iter().collect::<Vec<_>>();
        let mut step = PreparedPhysicalMixedStep::new(&refs, layout()).unwrap();
        let outputs = (0..LAYERS)
            .map(|layer| {
                step.write_and_attend(
                    &refs,
                    layer,
                    &inputs.queries[layer],
                    &inputs.keys[layer],
                    &inputs.values[layer],
                    0.5,
                    softcap,
                    window,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let mut refs = caches.iter_mut().collect::<Vec<_>>();
        step.finish(&mut refs, Ok(outputs)).unwrap()
    }

    /// The same step as one `write_and_attend` per row, the path every family
    /// takes today.
    fn run_per_row(
        caches: &mut [PhysicalPagedKvCache],
        inputs: &StepInputs,
        softcap: Option<f32>,
        window: Option<usize>,
    ) -> Vec<Tensor> {
        let layout = layout();
        let mut outputs = vec![Vec::new(); LAYERS];
        for (row, cache) in caches.iter_mut().enumerate() {
            let span = layout.rows()[row];
            let offset = layout.token_offset(row);
            let mut prepared = cache.prepare_append(span.start_pos, span.q_len).unwrap();
            for (layer, layer_outputs) in outputs.iter_mut().enumerate() {
                let slice = |tensor: &Tensor| tensor.narrow(0, offset, span.q_len).unwrap();
                layer_outputs.push(
                    cache
                        .write_and_attend_with_semantics(
                            layer,
                            &mut prepared,
                            &slice(&inputs.queries[layer]),
                            &slice(&inputs.keys[layer]),
                            &slice(&inputs.values[layer]),
                            0.5,
                            softcap,
                            window,
                        )
                        .unwrap(),
                );
            }
            cache.commit_prepared(prepared).unwrap();
        }
        outputs
            .into_iter()
            .map(|rows| Tensor::cat(&rows, 0).unwrap())
            .collect()
    }

    fn pages(arena: &Arc<dyn KvArena>) -> Vec<Vec<u8>> {
        let config = arena.config();
        (0..config.capacity_pages)
            .map(|index| {
                let mut bytes = vec![0u8; arena_page_bytes(config) as usize];
                arena
                    .capture_page(
                        CacheBlockRef {
                            arena: config.id,
                            group: config.group,
                            index,
                            slot_generation: 1,
                        },
                        &mut bytes,
                    )
                    .unwrap();
                bytes
            })
            .collect()
    }

    /// Largest absolute difference; a non-finite difference counts as
    /// infinite, so NaN never reads as a match.
    fn max_abs_diff(left: &Tensor, right: &Tensor) -> f32 {
        let left = left.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let right = right.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(left.len(), right.len());
        left.iter()
            .zip(&right)
            .map(|(a, b)| {
                let diff = (a - b).abs();
                if diff.is_finite() {
                    diff
                } else {
                    f32::INFINITY
                }
            })
            .fold(0.0, |max, diff| if diff > max { diff } else { max })
    }

    /// Run both paths on fresh arenas of `device`; the per-layer max output
    /// difference, and whether every page ends up byte-identical.
    fn compare(device: &Device, softcap: Option<f32>, window: Option<usize>) -> (f32, bool) {
        let tokens = layout().total_tokens();
        let inputs = step_inputs(tokens, device);
        let (mixed_arena, mut mixed) = shuffled_caches(device);
        let (row_arena, mut per_row) = shuffled_caches(device);
        write_prefixes(&mut mixed, device);
        write_prefixes(&mut per_row, device);
        let mixed_out = run_mixed(&mut mixed, &inputs, softcap, window);
        let row_out = run_per_row(&mut per_row, &inputs, softcap, window);
        for (cache, (prefix, q_len)) in mixed.iter().zip(ROWS) {
            assert_eq!(cache.context_len(), prefix + q_len);
        }
        let diff = mixed_out
            .iter()
            .zip(&row_out)
            .map(|(mixed, row)| {
                assert_eq!(mixed.dims(), [tokens, HEADS, HEAD_DIM]);
                max_abs_diff(mixed, row)
            })
            .fold(0.0, f32::max);
        (diff, pages(&mixed_arena) == pages(&row_arena))
    }

    #[test]
    fn tolerance_helper_reads_nan_as_infinite() {
        let device = Device::Cpu;
        let finite = Tensor::new(&[1.0f32, 2.0], &device).unwrap();
        let nan = Tensor::new(&[1.0f32, f32::NAN], &device).unwrap();
        assert_eq!(max_abs_diff(&finite, &nan), f32::INFINITY);
        assert_eq!(max_abs_diff(&finite, &finite), 0.0);
    }

    #[test]
    fn layout_rejects_multi_token_decode_rows() {
        let span = |start_pos, q_len| MixedRowSpan { start_pos, q_len };
        assert!(MixedRowLayout::new(vec![span(3, 2), span(0, 4)], 1).is_err());
        assert!(MixedRowLayout::new(vec![span(3, 1), span(0, 0)], 1).is_err());
        assert!(MixedRowLayout::new(vec![span(3, 1)], 2).is_err());
        let layout = MixedRowLayout::new(vec![span(3, 1), span(9, 1), span(0, 4)], 2).unwrap();
        assert_eq!(layout.total_tokens(), 6);
        assert_eq!(
            (0..3).map(|row| layout.last_token(row)).collect::<Vec<_>>(),
            [0, 1, 5]
        );
    }

    /// On the CPU the mixed step runs the same kernels per subset as one call
    /// per row, so outputs and written pages are bit-identical.
    #[test]
    fn mixed_step_matches_per_row_attention_exactly_on_cpu() {
        for (softcap, window) in [(None, None), (Some(30.0), Some(8))] {
            let (diff, pages_equal) = compare(&Device::Cpu, softcap, window);
            assert_eq!(diff, 0.0, "softcap {softcap:?} window {window:?}");
            assert!(pages_equal, "softcap {softcap:?} window {window:?}");
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn mixed_step_matches_per_row_attention_on_metal() {
        let Some((device, _guard)) = crate::backends::metal_test_device_or_skip(
            "mixed_step_matches_per_row_attention_on_metal",
        ) else {
            return;
        };
        for (softcap, window) in [(None, None), (Some(30.0), Some(8))] {
            let (diff, pages_equal) = compare(&device, softcap, window);
            assert!(
                diff <= 1e-5,
                "softcap {softcap:?} window {window:?}: {diff}"
            );
            assert!(pages_equal, "softcap {softcap:?} window {window:?}");
        }
    }
}
