//! Packed, authenticated physical attention shared by Audio8's two AR clocks.
use crate::backends::kv::{
    submit_ordered_after_write, KvSlotMap, KvWriteArgs, KvWriteCompletionCollector,
    PagedKvDecodeArgs, PagedKvPrefillArgs, PagedKvPrefillRow,
};
use crate::error::{Error, Result};
use crate::kv::KvDecodeBatchMetadata;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use candle_core::Tensor;
use std::sync::Arc;

pub(super) struct Audio8PhysicalBatch {
    starts: Vec<usize>,
    counts: Vec<usize>,
    slots: Arc<dyn KvSlotMap>,
    decode: KvDecodeBatchMetadata,
    prefill: Vec<PagedKvPrefillRow>,
    completions: KvWriteCompletionCollector,
}

impl Audio8PhysicalBatch {
    pub(super) fn new(
        caches: &[&mut PhysicalPagedKvCache],
        counts: &[usize],
        layers: usize,
    ) -> Result<Self> {
        if caches.is_empty() || caches.len() != counts.len() || counts.contains(&0) {
            return Err(Error::InvalidInput(
                "Audio8 physical batch requires matching nonempty rows".into(),
            ));
        }
        let first = &*caches[0];
        for cache in caches {
            if !Arc::ptr_eq(first.arena(), cache.arena()) {
                return Err(Error::InvalidInput(
                    "Audio8 batch rows must share an authenticated physical arena".into(),
                ));
            }
            for layer in 0..layers {
                if cache.layer_binding(layer)? != first.layer_binding(layer)? {
                    return Err(Error::InvalidInput(
                        "Audio8 batch layer bindings differ".into(),
                    ));
                }
            }
        }
        let starts = caches.iter().map(|c| c.context_len()).collect::<Vec<_>>();
        let mut logical_slots = Vec::new();
        let mut sequences = Vec::new();
        let mut prefill = Vec::new();
        let mut offset = 0usize;
        for ((cache, &start), &count) in caches.iter().zip(&starts).zip(counts) {
            logical_slots.extend(cache.slots_for_append(start, count)?);
            let table =
                cache.sequence_table(start.checked_add(count).ok_or_else(|| {
                    Error::InvalidInput("Audio8 batch context overflow".into())
                })?)?;
            prefill.push(PagedKvPrefillRow {
                blocks: table.blocks.clone(),
                first_page_offset: table.first_page_offset,
                query_start: u32::try_from(offset)
                    .map_err(|_| Error::InvalidInput("Audio8 packed offset exceeds u32".into()))?,
                query_len: u32::try_from(count)
                    .map_err(|_| Error::InvalidInput("Audio8 packed length exceeds u32".into()))?,
                context_len: table.context_len,
            });
            offset = offset
                .checked_add(count)
                .ok_or_else(|| Error::InvalidInput("Audio8 packed tokens overflow".into()))?;
            sequences.push(table);
        }
        let slots = first.arena().lower_slots(&logical_slots)?;
        let completions =
            KvWriteCompletionCollector::new(first.arena().config(), slots.logical_slots())?;
        Ok(Self {
            starts,
            counts: counts.to_vec(),
            slots,
            decode: KvDecodeBatchMetadata { sequences },
            prefill,
            completions,
        })
    }
    pub(super) fn attend(
        &mut self,
        cache: &PhysicalPagedKvCache,
        layer: usize,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f32,
    ) -> Result<Tensor> {
        let binding = cache.layer_binding(layer)?;
        let completion = cache.arena().write_slots(
            binding,
            KvWriteArgs {
                keys: k,
                values: v,
                slots: self.slots.as_ref(),
            },
        )?;
        let (output, completion) = submit_ordered_after_write(completion, || {
            if self.counts.iter().all(|&n| n == 1) {
                cache.arena().paged_decode(
                    binding,
                    PagedKvDecodeArgs {
                        queries: q,
                        batch: &self.decode,
                        softmax_scale: scale,
                        softcap: None,
                    },
                )
            } else {
                cache.arena().paged_prefill(
                    binding,
                    PagedKvPrefillArgs {
                        queries: q,
                        rows: &self.prefill,
                        softmax_scale: scale,
                        softcap: None,
                        window_tokens: None,
                    },
                )
            }
        })?;
        self.completions.collect(completion)?;
        Ok(output)
    }
    pub(super) fn finish<T>(
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
                        "Audio8 batch failed: {error}; write-fence drain failed: {drain}"
                    ))),
                }
            }
        };
        let completion = Arc::new(self.completions.seal()?);
        for ((cache, start), count) in caches.iter_mut().zip(self.starts).zip(self.counts) {
            cache.commit_shared_completion(start, count, completion.clone())?;
        }
        Ok(output)
    }
}
