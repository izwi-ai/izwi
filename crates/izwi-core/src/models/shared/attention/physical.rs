//! Model-neutral session view over scheduler-owned paged-attention storage.

use std::collections::HashSet;
use std::sync::Arc;

use candle_core::Tensor;

use crate::backends::kv::{
    KvArena, KvSlotMap, KvWriteArgs, KvWriteBatchCompletion, KvWriteCompletionCollector,
    PagedKvDecodeArgs, PagedKvPrefillArgs, PagedKvPrefillRow,
};
use crate::error::{Error, Result};
use crate::kv::{
    CacheBlockRef, KvArenaId, KvDecodeBatchMetadata, KvLayerBinding, KvSequenceBlockTable,
    KvSlotRef,
};

/// One immutable append/decode view lowered once and reused by every model
/// layer in an execution quantum.
pub(crate) struct PreparedPhysicalPagedStep {
    arena: KvArenaId,
    logical_generation: u32,
    start_pos: usize,
    token_count: usize,
    slots: Arc<dyn KvSlotMap>,
    decode: KvDecodeBatchMetadata,
    prefill: Vec<PagedKvPrefillRow>,
    completions: KvWriteCompletionCollector,
}

/// A generation-pinned logical block table over one physical paged-attention
/// arena. Models retain only this view; K/V tensors remain backend-owned.
pub struct PhysicalPagedKvCache {
    pub(crate) arena: Arc<dyn KvArena>,
    layer_bindings: Vec<KvLayerBinding>,
    pub(crate) blocks: Vec<CacheBlockRef>,
    window_start: usize,
    context_len: usize,
    logical_generation: u32,
    completed_writes: Vec<Arc<KvWriteBatchCompletion>>,
}

impl PhysicalPagedKvCache {
    pub fn new(
        arena: Arc<dyn KvArena>,
        layer_bindings: Vec<KvLayerBinding>,
        blocks: Vec<CacheBlockRef>,
        context_len: usize,
    ) -> Result<Self> {
        Self::new_windowed(arena, layer_bindings, blocks, 0, context_len)
    }

    pub fn new_windowed(
        arena: Arc<dyn KvArena>,
        layer_bindings: Vec<KvLayerBinding>,
        blocks: Vec<CacheBlockRef>,
        window_start: usize,
        context_len: usize,
    ) -> Result<Self> {
        if layer_bindings.is_empty() {
            return Err(Error::InvalidInput(
                "physical paged cache has no layer bindings".to_string(),
            ));
        }
        if blocks.is_empty() {
            return Err(Error::InvalidInput(
                "physical paged cache has no physical blocks".to_string(),
            ));
        }
        let arena_id = arena.id();
        let group = arena.config().group;
        let mut unique_blocks = HashSet::with_capacity(blocks.len());
        for block in &blocks {
            if block.arena != arena_id || block.group != group {
                return Err(Error::InvalidInput(
                    "physical paged cache block belongs to another arena or group".to_string(),
                ));
            }
            if block.index >= arena.config().capacity_pages {
                return Err(Error::InvalidInput(format!(
                    "physical paged cache block {} exceeds arena capacity {}",
                    block.index,
                    arena.config().capacity_pages
                )));
            }
            if !unique_blocks.insert(*block) {
                return Err(Error::InvalidInput(
                    "physical paged cache block table contains a duplicate block".to_string(),
                ));
            }
        }
        if window_start > context_len {
            return Err(Error::InvalidInput(
                "physical paged cache window starts after its context".to_string(),
            ));
        }
        let page_tokens = arena.config().page_tokens as usize;
        let first_page_start = (window_start / page_tokens)
            .checked_mul(page_tokens)
            .ok_or_else(|| Error::InvalidInput("physical page start overflow".into()))?;
        let capacity_end = blocks
            .len()
            .checked_mul(page_tokens)
            .and_then(|capacity| first_page_start.checked_add(capacity))
            .ok_or_else(|| Error::InvalidInput("physical paged cache capacity overflow".into()))?;
        if context_len > capacity_end {
            return Err(Error::InvalidInput(format!(
                "physical paged cache context {context_len} exceeds capacity end {capacity_end}"
            )));
        }
        let mut previous_model_layer = None;
        for (expected_physical, binding) in layer_bindings.iter().enumerate() {
            if previous_model_layer.is_some_and(|previous| binding.model_layer <= previous)
                || binding.physical_layer as usize != expected_physical
            {
                return Err(Error::InvalidInput(format!(
                    "physical layer bindings must have increasing model layers and dense physical ordinals; got {}:{} at ordinal {}",
                    binding.model_layer, binding.physical_layer, expected_physical
                )));
            }
            previous_model_layer = Some(binding.model_layer);
        }
        Ok(Self {
            arena,
            layer_bindings,
            blocks,
            window_start,
            context_len,
            logical_generation: 1,
            completed_writes: Vec::new(),
        })
    }

    pub fn context_len(&self) -> usize {
        self.context_len
    }

    pub fn capacity_tokens(&self) -> usize {
        let page_tokens = self.arena.config().page_tokens as usize;
        (self.window_start / page_tokens) * page_tokens + self.blocks.len() * page_tokens
    }

    pub fn window_start(&self) -> usize {
        self.window_start
    }

    pub fn arena(&self) -> &Arc<dyn KvArena> {
        &self.arena
    }

    pub(crate) fn validate_model(
        &self,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        if self.layer_bindings.len() != num_layers || self.arena.config().layers.len() != num_layers
        {
            return Err(Error::InvalidInput(format!(
                "physical paged cache has {} layers for a {num_layers}-layer model",
                self.layer_bindings.len()
            )));
        }
        for (binding, layer) in self
            .layer_bindings
            .iter()
            .zip(self.arena.config().layers.iter())
        {
            if layer.binding != *binding
                || layer.num_kv_heads as usize != num_kv_heads
                || layer.key_head_dim as usize != head_dim
                || layer.value_head_dim as usize != head_dim
            {
                return Err(Error::InvalidInput(
                    "physical paged cache geometry does not match the loaded model".to_string(),
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn validate_sparse_model(
        &self,
        model_layers: &[u32],
        num_kv_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
    ) -> Result<()> {
        let layers = model_layers
            .iter()
            .map(|model_layer| (*model_layer, num_kv_heads, key_head_dim, value_head_dim))
            .collect::<Vec<_>>();
        self.validate_sparse_model_layers(&layers)
    }

    pub(crate) fn validate_sparse_model_layers(
        &self,
        model_layers: &[(u32, usize, usize, usize)],
    ) -> Result<()> {
        if self.layer_bindings.len() != model_layers.len()
            || self.arena.config().layers.len() != model_layers.len()
        {
            return Err(Error::InvalidInput(
                "physical paged cache does not cover every sparse attention layer".into(),
            ));
        }
        for ((binding, layer), (model_layer, num_kv_heads, key_head_dim, value_head_dim)) in self
            .layer_bindings
            .iter()
            .zip(self.arena.config().layers.iter())
            .zip(model_layers)
        {
            if layer.binding != *binding
                || binding.model_layer != *model_layer
                || layer.num_kv_heads as usize != *num_kv_heads
                || layer.key_head_dim as usize != *key_head_dim
                || layer.value_head_dim as usize != *value_head_dim
            {
                return Err(Error::InvalidInput(
                    "physical paged cache geometry does not match the sparse attention model"
                        .into(),
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn slots_for_append(
        &self,
        start_pos: usize,
        token_count: usize,
    ) -> Result<Vec<KvSlotRef>> {
        if start_pos != self.context_len {
            return Err(Error::InvalidInput(format!(
                "physical paged append starts at {start_pos}, expected {}",
                self.context_len
            )));
        }
        let end = start_pos
            .checked_add(token_count)
            .ok_or_else(|| Error::InvalidInput("physical paged context overflow".into()))?;
        if end > self.capacity_tokens() {
            return Err(Error::InvalidInput(format!(
                "physical paged append ends at {end}, beyond capacity {}",
                self.capacity_tokens()
            )));
        }
        let page_tokens = self.arena.config().page_tokens as usize;
        let first_logical_page = self.window_start / page_tokens;
        (start_pos..end)
            .map(|position| {
                let logical_page = position / page_tokens;
                let table_index =
                    logical_page
                        .checked_sub(first_logical_page)
                        .ok_or_else(|| {
                            Error::InvalidInput(
                                "physical paged append precedes its cache window".into(),
                            )
                        })?;
                Ok(KvSlotRef {
                    block: *self.blocks.get(table_index).ok_or_else(|| {
                        Error::InvalidInput("physical paged append exceeds its block table".into())
                    })?,
                    offset: u32::try_from(position % page_tokens).map_err(|_| {
                        Error::InvalidInput("physical page offset exceeds u32".into())
                    })?,
                })
            })
            .collect()
    }

    pub(crate) fn sequence_table(&self, context_len: usize) -> Result<KvSequenceBlockTable> {
        self.sequence_table_from(self.window_start, context_len)
    }

    fn sequence_table_from(
        &self,
        visible_start: usize,
        context_len: usize,
    ) -> Result<KvSequenceBlockTable> {
        if visible_start < self.window_start
            || context_len <= visible_start
            || context_len > self.capacity_tokens()
        {
            return Err(Error::InvalidInput(format!(
                "physical paged decode context {context_len} is outside cache capacity"
            )));
        }
        let page_tokens = self.arena.config().page_tokens as usize;
        let allocated_first_page = self.window_start / page_tokens;
        let visible_first_page = visible_start / page_tokens;
        let first_block = visible_first_page
            .checked_sub(allocated_first_page)
            .ok_or_else(|| {
                Error::InvalidInput("physical visible window precedes its allocation".into())
            })?;
        let first_page_offset = visible_start % page_tokens;
        let visible_tokens = context_len - visible_start;
        let required_pages = (first_page_offset + visible_tokens).div_ceil(page_tokens);
        let end_block = first_block
            .checked_add(required_pages)
            .ok_or_else(|| Error::InvalidInput("physical visible page range overflow".into()))?;
        Ok(KvSequenceBlockTable {
            blocks: self
                .blocks
                .get(first_block..end_block)
                .ok_or_else(|| {
                    Error::InvalidInput("physical visible window exceeds its block table".into())
                })?
                .to_vec(),
            first_page_offset: u32::try_from(first_page_offset).map_err(|_| {
                Error::InvalidInput("physical first-page offset exceeds u32".into())
            })?,
            context_len: u32::try_from(visible_tokens)
                .map_err(|_| Error::InvalidInput("physical context length exceeds u32".into()))?,
        })
    }

    pub(crate) fn layer_binding(&self, layer_idx: usize) -> Result<KvLayerBinding> {
        self.layer_bindings.get(layer_idx).copied().ok_or_else(|| {
            Error::InvalidInput(format!(
                "physical paged cache has no binding for layer {layer_idx}"
            ))
        })
    }

    pub(crate) fn prepare_append(
        &self,
        start_pos: usize,
        token_count: usize,
    ) -> Result<PreparedPhysicalPagedStep> {
        self.prepare_append_visible(start_pos, token_count, self.window_start)
    }

    /// Prepare an exact one-token sliding-window append without changing the
    /// cache allocation or write coordinates. The returned attention table
    /// begins at `end_pos - window_tokens` while the new K/V still lands in
    /// its absolute physical slot.
    pub(crate) fn prepare_append_with_window(
        &self,
        start_pos: usize,
        token_count: usize,
        window_tokens: usize,
    ) -> Result<PreparedPhysicalPagedStep> {
        if window_tokens == 0 {
            return Err(Error::InvalidInput(
                "physical paged sliding window cannot be zero".into(),
            ));
        }
        if token_count != 1 {
            return Err(Error::InvalidInput(
                "physical paged sliding-window append currently requires one token".into(),
            ));
        }
        let end_pos = start_pos
            .checked_add(token_count)
            .ok_or_else(|| Error::InvalidInput("physical paged context overflow".into()))?;
        let visible_start = end_pos.saturating_sub(window_tokens).max(self.window_start);
        self.prepare_append_visible(start_pos, token_count, visible_start)
    }

    fn prepare_append_visible(
        &self,
        start_pos: usize,
        token_count: usize,
        visible_start: usize,
    ) -> Result<PreparedPhysicalPagedStep> {
        if token_count == 0 {
            return Err(Error::InvalidInput(
                "physical paged append cannot prepare zero tokens".into(),
            ));
        }
        let end_pos = start_pos
            .checked_add(token_count)
            .ok_or_else(|| Error::InvalidInput("physical paged context overflow".into()))?;
        let slots = self
            .arena
            .lower_slots(&self.slots_for_append(start_pos, token_count)?)?;
        let table = self.sequence_table_from(visible_start, end_pos)?;
        let query_len = u32::try_from(token_count)
            .map_err(|_| Error::InvalidInput("physical paged query length exceeds u32".into()))?;
        let completions =
            KvWriteCompletionCollector::new(self.arena.config(), slots.logical_slots())?;
        Ok(PreparedPhysicalPagedStep {
            arena: self.arena.id(),
            logical_generation: self.logical_generation,
            start_pos,
            token_count,
            slots,
            decode: KvDecodeBatchMetadata {
                sequences: vec![table.clone()],
            },
            prefill: vec![PagedKvPrefillRow {
                blocks: table.blocks,
                first_page_offset: table.first_page_offset,
                query_start: 0,
                query_len,
                context_len: table.context_len,
            }],
            completions,
        })
    }

    /// Write one layer's projected K/V directly into its prepared physical
    /// slots and execute causal attention against the same authoritative pages.
    ///
    /// The tensors are token-major: queries are `[tokens, query_heads, dim]`
    /// and keys/values are `[tokens, kv_heads, dim]`. Multi-token calls use the
    /// page-native ragged prefill/extend operation, including non-zero-prefix
    /// continuation; one-token calls use the batched decode operation.
    pub(crate) fn write_and_attend(
        &self,
        layer_idx: usize,
        prepared: &mut PreparedPhysicalPagedStep,
        queries: &Tensor,
        keys: &Tensor,
        values: &Tensor,
        softmax_scale: f32,
    ) -> Result<Tensor> {
        let token_count = queries.dim(0)?;
        if token_count == 0 || keys.dim(0)? != token_count || values.dim(0)? != token_count {
            return Err(Error::InvalidInput(
                "physical paged attention requires matching non-empty token dimensions".into(),
            ));
        }
        if prepared.arena != self.arena.id()
            || prepared.logical_generation != self.logical_generation
            || prepared.start_pos != self.context_len
            || prepared.token_count != token_count
            || prepared.slots.len() != token_count
        {
            return Err(Error::InvalidInput(
                "physical paged attention received a stale or incompatible prepared step".into(),
            ));
        }
        let binding = self.layer_binding(layer_idx)?;
        let completion = self.arena.write_slots(
            binding,
            KvWriteArgs {
                keys,
                values,
                slots: prepared.slots.as_ref(),
            },
        )?;
        if completion.arena() != self.arena.id()
            || completion.layer() != binding
            || completion.slots() != token_count
        {
            return Err(Error::InferenceError(
                "physical paged write returned a mismatched backend completion".into(),
            ));
        }
        completion.wait()?;
        prepared.completions.collect(completion)?;

        if token_count == 1 {
            return self.arena.paged_decode(
                binding,
                PagedKvDecodeArgs {
                    queries,
                    batch: &prepared.decode,
                    softmax_scale,
                },
            );
        }

        self.arena.paged_prefill(
            binding,
            PagedKvPrefillArgs {
                queries,
                rows: &prepared.prefill,
                softmax_scale,
            },
        )
    }

    pub(crate) fn commit_prepared(&mut self, prepared: PreparedPhysicalPagedStep) -> Result<()> {
        if prepared.arena != self.arena.id()
            || prepared.logical_generation != self.logical_generation
            || prepared.start_pos != self.context_len
        {
            return Err(Error::InvalidInput(
                "physical paged commit received a stale prepared step".into(),
            ));
        }
        let completion = Arc::new(prepared.completions.seal()?);
        self.commit_shared_completion(prepared.start_pos, prepared.token_count, completion)
    }

    pub(crate) fn commit_shared_completion(
        &mut self,
        start_pos: usize,
        token_count: usize,
        completion: Arc<KvWriteBatchCompletion>,
    ) -> Result<()> {
        let expected = self.slots_for_append(start_pos, token_count)?;
        if completion.arena() != self.arena.id()
            || completion.layers() != self.layer_bindings.as_slice()
            || completion.page_tokens() != self.arena.config().page_tokens
            || expected
                .iter()
                .any(|slot| !completion.slots().contains(slot))
        {
            return Err(Error::InferenceError(
                "physical paged completion does not authenticate this append".into(),
            ));
        }
        self.context_len = self
            .context_len
            .checked_add(token_count)
            .ok_or_else(|| Error::InvalidInput("physical paged context overflow".into()))?;
        self.completed_writes.push(completion);
        Ok(())
    }

    pub(crate) fn take_completed_writes(&mut self) -> Vec<Arc<KvWriteBatchCompletion>> {
        std::mem::take(&mut self.completed_writes)
    }

    /// Reuse one invocation-exclusive page range for a new nested logical
    /// sequence. Every previously committed backend write was already waited
    /// and authenticated before it entered `completed_writes`; dropping those
    /// receipts cannot expose unfinished device work. A monotonically
    /// increasing generation invalidates prepared steps created before reset.
    ///
    /// Physical pages are not materialized or reallocated here. Subsequent
    /// attention can address only the new logical context and overwrites each
    /// visible slot before reading it. The owning invocation pool still zeros
    /// and fences the complete range between independent leases.
    pub(crate) fn reset_invocation(&mut self) -> Result<()> {
        let logical_generation = self.logical_generation.checked_add(1).ok_or_else(|| {
            Error::InvalidInput("physical paged reset generation overflow".into())
        })?;
        self.completed_writes.clear();
        self.window_start = 0;
        self.context_len = 0;
        self.logical_generation = logical_generation;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    use crate::backends::kv::{CpuKvArena, KvArenaConfig, KvLayerConfig};
    use crate::backends::BackendKind;
    use crate::engine::ModelInstanceId;
    use crate::kv::{KvGroupId, KvLayerBinding};

    #[test]
    fn sparse_model_layers_bind_to_dense_physical_ordinals() {
        let arena_id = KvArenaId {
            model_instance: ModelInstanceId::new(9),
            backend: BackendKind::Cpu,
            device_ordinal: None,
            generation: 1,
        };
        let group = KvGroupId::new(1);
        let bindings = vec![
            KvLayerBinding {
                model_layer: 3,
                physical_layer: 0,
            },
            KvLayerBinding {
                model_layer: 7,
                physical_layer: 1,
            },
        ];
        let arena = Arc::new(
            CpuKvArena::new(KvArenaConfig {
                id: arena_id,
                group,
                page_tokens: 4,
                capacity_pages: 1,
                dtype: DType::F32,
                layers: bindings
                    .iter()
                    .copied()
                    .map(|binding| KvLayerConfig {
                        binding,
                        num_kv_heads: 2,
                        key_head_dim: 4,
                        value_head_dim: 4,
                    })
                    .collect(),
            })
            .unwrap(),
        );
        let mut cache = PhysicalPagedKvCache::new(
            arena,
            bindings,
            vec![CacheBlockRef {
                arena: arena_id,
                group,
                index: 0,
                slot_generation: 1,
            }],
            2,
        )
        .unwrap();

        cache.validate_sparse_model(&[3, 7], 2, 4, 4).unwrap();
        assert!(cache.validate_sparse_model(&[3, 6], 2, 4, 4).is_err());
        let prepared = cache.prepare_append(2, 1).unwrap();
        cache.reset_invocation().unwrap();
        assert_eq!(cache.context_len(), 0);
        assert_eq!(cache.window_start(), 0);
        assert!(cache.commit_prepared(prepared).is_err());
    }

    #[test]
    fn one_token_sliding_append_reads_only_its_exact_visible_pages() {
        let arena_id = KvArenaId {
            model_instance: ModelInstanceId::new(10),
            backend: BackendKind::Cpu,
            device_ordinal: None,
            generation: 1,
        };
        let group = KvGroupId::new(1);
        let binding = KvLayerBinding {
            model_layer: 2,
            physical_layer: 0,
        };
        let arena = Arc::new(
            CpuKvArena::new(KvArenaConfig {
                id: arena_id,
                group,
                page_tokens: 4,
                capacity_pages: 4,
                dtype: DType::F32,
                layers: vec![KvLayerConfig {
                    binding,
                    num_kv_heads: 2,
                    key_head_dim: 4,
                    value_head_dim: 4,
                }],
            })
            .unwrap(),
        );
        let blocks = (0..4)
            .map(|index| CacheBlockRef {
                arena: arena_id,
                group,
                index,
                slot_generation: 1,
            })
            .collect::<Vec<_>>();
        let cache = PhysicalPagedKvCache::new(arena, vec![binding], blocks.clone(), 9).unwrap();

        let prepared = cache.prepare_append_with_window(9, 1, 4).unwrap();
        let table = &prepared.decode.sequences[0];
        assert_eq!(table.blocks, blocks[1..3]);
        assert_eq!(table.first_page_offset, 2);
        assert_eq!(table.context_len, 4);
        assert_eq!(prepared.prefill[0].blocks, blocks[1..3]);
        assert_eq!(prepared.slots.logical_slots()[0].block, blocks[2]);
        assert_eq!(prepared.slots.logical_slots()[0].offset, 1);

        assert!(cache.prepare_append_with_window(9, 2, 4).is_err());
        assert!(cache.prepare_append_with_window(9, 1, 0).is_err());
    }
}
