//! One decode step's paged attention for every row at once: each layer writes
//! every row's key and value through one slot map and attends every row in
//! one row-invariant decode call, where the step had attended row by row
//! (30 layers times the rows: 240 eager calls a step at 8 rows, 21 of the
//! step's 34 ms on the H100). The sliding layers attend each row's window
//! view, the full layers its whole context; the writes commit together.

use std::sync::Arc;

use candle_core::Tensor;

use crate::backends::kv::{
    submit_ordered_after_write, KvSlotMap, KvWriteArgs, KvWriteCompletionCollector,
    PagedKvDecodeArgs,
};
use crate::error::{Error, Result};
use crate::kv::KvDecodeBatchMetadata;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

pub(super) struct SovDecodeBatch {
    starts: Vec<usize>,
    slots: Arc<dyn KvSlotMap>,
    /// Every row's whole context, for the full-attention layers.
    full: KvDecodeBatchMetadata,
    /// Every row's sliding-window view, for the sliding layers.
    windowed: Option<KvDecodeBatchMetadata>,
    completions: KvWriteCompletionCollector,
}

impl SovDecodeBatch {
    /// One token for each cache at its cursor. The caches share one arena
    /// and layer bindings (rows of one model).
    pub(super) fn new(
        caches: &[&PhysicalPagedKvCache],
        window: Option<usize>,
        layers: usize,
    ) -> Result<Self> {
        let Some(first) = caches.first() else {
            return Err(Error::InvalidInput("a decode batch needs a row".into()));
        };
        for cache in caches {
            if !Arc::ptr_eq(first.arena(), cache.arena()) {
                return Err(Error::InvalidInput(
                    "SOV-Transcribe decode rows must share one physical arena".into(),
                ));
            }
            for layer in 0..layers {
                if cache.layer_binding(layer)? != first.layer_binding(layer)? {
                    return Err(Error::InvalidInput(
                        "SOV-Transcribe decode rows bind different layers".into(),
                    ));
                }
            }
        }
        let mut starts = Vec::with_capacity(caches.len());
        let mut logical_slots = Vec::with_capacity(caches.len());
        let mut full = Vec::with_capacity(caches.len());
        let mut windowed = Vec::with_capacity(caches.len());
        for cache in caches {
            let start = cache.context_len();
            starts.push(start);
            logical_slots.extend(cache.slots_for_append(start, 1)?);
            let end = start
                .checked_add(1)
                .ok_or_else(|| Error::InvalidInput("SOV-Transcribe context overflow".into()))?;
            full.push(cache.sequence_table(end)?);
            if let Some(window) = window {
                let (view, _) = cache.window_attention_views(start, 1, window)?;
                windowed.extend(view.sequences);
            }
        }
        let slots = first.arena().lower_slots(&logical_slots)?;
        let completions =
            KvWriteCompletionCollector::new(first.arena().config(), slots.logical_slots())?;
        Ok(Self {
            starts,
            slots,
            full: KvDecodeBatchMetadata { sequences: full },
            windowed: window.map(|_| KvDecodeBatchMetadata {
                sequences: windowed,
            }),
            completions,
        })
    }

    /// Writes every row's `k` and `v` (`[rows, kv_heads, d]`) at `layer` and
    /// attends every row's `q` over its context, or its window when
    /// `windowed`. Returns `[rows, heads, d]` in the KV dtype.
    pub(super) fn attend(
        &mut self,
        cache: &PhysicalPagedKvCache,
        layer: usize,
        [q, k, v]: [&Tensor; 3],
        windowed: bool,
    ) -> Result<Tensor> {
        let batch = match (windowed, &self.windowed) {
            (true, Some(view)) => view,
            (true, None) => {
                return Err(Error::InvalidInput(
                    "SOV-Transcribe decode batch has no window views".into(),
                ))
            }
            (false, _) => &self.full,
        };
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
            cache.arena().paged_decode_row_invariant(
                binding,
                PagedKvDecodeArgs {
                    queries: q,
                    batch,
                    softmax_scale: 1.0,
                    softcap: None,
                },
            )
        })?;
        self.completions.collect(completion)?;
        Ok(output)
    }

    /// Commits every row's token after a successful step; after a failure
    /// only drains the submitted writes.
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
                        "SOV-Transcribe decode failed: {error}; write-fence drain failed: {drain}"
                    ))),
                }
            }
        };
        let completion = Arc::new(self.completions.seal()?);
        for (cache, start) in caches.iter_mut().zip(self.starts) {
            cache.commit_shared_completion(start, 1, completion.clone())?;
        }
        Ok(output)
    }
}
