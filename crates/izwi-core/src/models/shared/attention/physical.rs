//! Model-neutral session view over scheduler-owned paged-attention storage.

use std::collections::HashSet;
use std::sync::Arc;

use candle_core::Tensor;

use crate::backends::kv::{
    submit_ordered_after_write, KvArena, KvSlotMap, KvWriteArgs, KvWriteBatchCompletion,
    KvWriteCompletionCollector, PagedKvDecodeArgs, PagedKvPrefillArgs, PagedKvPrefillRow,
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

    /// Recycle fully invisible leading pages before a sliding-window append.
    /// The arena keeps one spare page beyond the logical attention window so
    /// the partially visible leading and newly written trailing pages can
    /// coexist. Page identities rotate; absolute positions and context length
    /// remain monotonic.
    pub(crate) fn advance_sliding_window_for_append(
        &mut self,
        start_pos: usize,
        token_count: usize,
        window_tokens: usize,
    ) -> Result<()> {
        if start_pos != self.context_len || token_count == 0 || window_tokens == 0 {
            return Err(Error::InvalidInput(
                "physical sliding-window rotation received an invalid append".into(),
            ));
        }
        let end_pos = start_pos
            .checked_add(token_count)
            .ok_or_else(|| Error::InvalidInput("physical paged context overflow".into()))?;
        let page_tokens = self.arena.config().page_tokens as usize;
        let first_query_end = start_pos
            .checked_add(1)
            .ok_or_else(|| Error::InvalidInput("physical paged context overflow".into()))?;
        let visible_start = first_query_end.saturating_sub(window_tokens);

        while end_pos > self.capacity_tokens() {
            let next_window_start = self
                .window_start
                .checked_add(page_tokens)
                .ok_or_else(|| Error::InvalidInput("physical window overflow".into()))?;
            if next_window_start > visible_start {
                return Err(Error::InvalidInput(
                    "physical sliding-window cache needs one spare page beyond the visible window"
                        .into(),
                ));
            }
            self.blocks.rotate_left(1);
            self.window_start = next_window_start;
            self.logical_generation = self
                .logical_generation
                .checked_add(1)
                .ok_or_else(|| Error::InvalidInput("physical window generation overflow".into()))?;
        }
        Ok(())
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

    pub(crate) fn sequence_table_with_window(
        &self,
        context_len: usize,
        window_tokens: usize,
    ) -> Result<KvSequenceBlockTable> {
        if window_tokens == 0 {
            return Err(Error::InvalidInput(
                "physical paged sliding window cannot be zero".into(),
            ));
        }
        self.sequence_table_from(
            context_len
                .saturating_sub(window_tokens)
                .max(self.window_start),
            context_len,
        )
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

    /// Prepare a sliding-window append without changing cache allocation or
    /// write coordinates. Multi-token appends carry one canonical prefill row;
    /// backend lowering advances its left edge independently for each query.
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
        let mut prepared = self.prepare_append(start_pos, token_count)?;
        let (decode, prefill) =
            self.window_attention_views(start_pos, token_count, window_tokens)?;
        prepared.decode = decode;
        prepared.prefill = prefill;
        Ok(prepared)
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

    fn window_attention_views(
        &self,
        start_pos: usize,
        token_count: usize,
        window_tokens: usize,
    ) -> Result<(KvDecodeBatchMetadata, Vec<PagedKvPrefillRow>)> {
        if token_count == 0 || window_tokens == 0 {
            return Err(Error::InvalidInput(
                "physical paged window requires non-zero tokens and window size".into(),
            ));
        }
        let end_pos = start_pos
            .checked_add(token_count)
            .ok_or_else(|| Error::InvalidInput("physical paged context overflow".into()))?;
        let first_query_end = start_pos
            .checked_add(1)
            .ok_or_else(|| Error::InvalidInput("physical paged context overflow".into()))?;
        let visible_start = first_query_end
            .saturating_sub(window_tokens)
            .max(self.window_start);
        let table = self.sequence_table_from(visible_start, end_pos)?;
        let query_len = u32::try_from(token_count)
            .map_err(|_| Error::InvalidInput("physical paged query length exceeds u32".into()))?;
        let prefill = PagedKvPrefillRow {
            blocks: table.blocks.clone(),
            first_page_offset: table.first_page_offset,
            query_start: 0,
            query_len,
            context_len: table.context_len,
        };
        Ok((
            KvDecodeBatchMetadata {
                sequences: vec![table],
            },
            vec![prefill],
        ))
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
        self.write_and_attend_with_semantics(
            layer_idx,
            prepared,
            queries,
            keys,
            values,
            softmax_scale,
            None,
            None,
        )
    }

    /// Write and attend with model-specific attention semantics while keeping
    /// physical slot coordinates and completion authentication unchanged.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_and_attend_with_semantics(
        &self,
        layer_idx: usize,
        prepared: &mut PreparedPhysicalPagedStep,
        queries: &Tensor,
        keys: &Tensor,
        values: &Tensor,
        softmax_scale: f32,
        softcap: Option<f32>,
        window_tokens: Option<usize>,
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
        if window_tokens == Some(0) {
            return Err(Error::InvalidInput(
                "physical paged sliding window cannot be zero".into(),
            ));
        }
        let window_tokens_u32 = window_tokens
            .map(|window| {
                u32::try_from(window)
                    .map_err(|_| Error::InvalidInput("physical paged window exceeds u32".into()))
            })
            .transpose()?;
        let window_views = window_tokens
            .map(|window| self.window_attention_views(prepared.start_pos, token_count, window))
            .transpose()?;
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
        let (output, completion) = submit_ordered_after_write(completion, || {
            if token_count == 1 {
                let decode = window_views
                    .as_ref()
                    .map(|(decode, _)| decode)
                    .unwrap_or(&prepared.decode);
                return self.arena.paged_decode(
                    binding,
                    PagedKvDecodeArgs {
                        queries,
                        batch: decode,
                        softmax_scale,
                        softcap,
                    },
                );
            }

            let prefill = window_views
                .as_ref()
                .map(|(_, prefill)| prefill.as_slice())
                .unwrap_or(&prepared.prefill);
            self.arena.paged_prefill(
                binding,
                PagedKvPrefillArgs {
                    queries,
                    rows: prefill,
                    softmax_scale,
                    softcap,
                    window_tokens: window_tokens_u32,
                },
            )
        })?;
        prepared.completions.collect(completion)?;
        Ok(output)
    }

    /// Append with a layer-specific visible window. The block table and write
    /// slots remain absolute; only each query's attention view is narrowed.
    pub(crate) fn write_and_attend_with_window(
        &self,
        layer_idx: usize,
        prepared: &mut PreparedPhysicalPagedStep,
        queries: &Tensor,
        keys: &Tensor,
        values: &Tensor,
        softmax_scale: f32,
        window_tokens: usize,
    ) -> Result<Tensor> {
        self.write_and_attend_with_semantics(
            layer_idx,
            prepared,
            queries,
            keys,
            values,
            softmax_scale,
            None,
            Some(window_tokens),
        )
    }

    pub(crate) fn commit_prepared(&mut self, prepared: PreparedPhysicalPagedStep) -> Result<()> {
        let token_count = prepared.token_count;
        self.commit_prepared_prefix(prepared, token_count)
    }

    /// Commit an exact logical prefix of one fully executed physical append.
    ///
    /// Every backend completion for the original append is sealed and fenced
    /// before this method returns, including writes for a rejected suffix. Only
    /// the accepted prefix advances the logical cursor or enters the completion
    /// receipts exposed to the executor. Passing zero therefore aborts the
    /// prepared append without exposing any of its physical writes.
    pub(crate) fn commit_prepared_prefix(
        &mut self,
        prepared: PreparedPhysicalPagedStep,
        accepted_token_count: usize,
    ) -> Result<()> {
        let is_compatible = prepared.arena == self.arena.id()
            && prepared.logical_generation == self.logical_generation
            && prepared.start_pos == self.context_len;
        let accepted_count_is_valid = accepted_token_count <= prepared.token_count;
        let start_pos = prepared.start_pos;
        let completion = prepared.completions.seal();

        // The collector is consumed and every submitted fence is drained above,
        // even when the logical commit request itself is invalid. This prevents
        // a rejected or stale prepared step from orphaning writes before its
        // provisional slots are reused.
        if !is_compatible {
            let message = "physical paged commit received a stale prepared step";
            return match completion {
                Ok(_) => Err(Error::InvalidInput(message.into())),
                Err(drain_error) => Err(Error::InferenceError(format!(
                    "{message}; prepared write drain also failed: {drain_error}"
                ))),
            };
        }
        if !accepted_count_is_valid {
            let message = format!(
                "physical paged accepted prefix {accepted_token_count} exceeds prepared token count {}",
                prepared.token_count
            );
            return match completion {
                Ok(_) => Err(Error::InvalidInput(message)),
                Err(drain_error) => Err(Error::InferenceError(format!(
                    "{message}; prepared write drain also failed: {drain_error}"
                ))),
            };
        }

        let completion = completion?;
        if accepted_token_count == 0 {
            return Ok(());
        }
        let completion = Arc::new(completion.into_slot_prefix(accepted_token_count)?);
        self.commit_shared_completion(start_pos, accepted_token_count, completion)
    }

    /// Fence and discard every write in one prepared append.
    pub(crate) fn abort_prepared(&mut self, prepared: PreparedPhysicalPagedStep) -> Result<()> {
        self.commit_prepared_prefix(prepared, 0)
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
    use candle_core::{DType, Device};

    use crate::backends::kv::{CpuKvArena, KvArenaConfig, KvLayerConfig};
    use crate::backends::BackendKind;
    use crate::engine::ModelInstanceId;
    use crate::kv::{KvGroupId, KvLayerBinding};

    fn prefix_test_cache(generation: u32) -> PhysicalPagedKvCache {
        let arena_id = KvArenaId {
            model_instance: ModelInstanceId::new(20),
            backend: BackendKind::Cpu,
            device_ordinal: None,
            generation,
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
                capacity_pages: 2,
                growth: None,
                dtype: DType::F32,
                layers: bindings
                    .iter()
                    .copied()
                    .map(|binding| KvLayerConfig {
                        binding,
                        num_kv_heads: 1,
                        key_head_dim: 2,
                        value_head_dim: 2,
                    })
                    .collect(),
            })
            .unwrap(),
        );
        let blocks = (0..2)
            .map(|index| CacheBlockRef {
                arena: arena_id,
                group,
                index,
                slot_generation: 1,
            })
            .collect();
        PhysicalPagedKvCache::new(arena, bindings, blocks, 0).unwrap()
    }

    fn submit_prepared_writes(
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        layer_count: usize,
    ) {
        let keys = Tensor::zeros((prepared.token_count, 1, 2), DType::F32, &Device::Cpu).unwrap();
        let values = keys.clone();
        let slots = prepared.slots.clone();
        for binding in cache.layer_bindings.iter().copied().take(layer_count) {
            let completion = cache
                .arena
                .write_slots(
                    binding,
                    KvWriteArgs {
                        keys: &keys,
                        values: &values,
                        slots: slots.as_ref(),
                    },
                )
                .unwrap();
            prepared.completions.collect(completion).unwrap();
        }
    }

    fn fully_written_append(
        cache: &PhysicalPagedKvCache,
        token_count: usize,
    ) -> PreparedPhysicalPagedStep {
        let mut prepared = cache
            .prepare_append(cache.context_len(), token_count)
            .unwrap();
        submit_prepared_writes(cache, &mut prepared, cache.layer_bindings.len());
        prepared
    }

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
                growth: None,
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
    fn multi_token_sliding_append_uses_one_canonical_prefill_table() {
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
                growth: None,
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

        let prepared = cache.prepare_append_with_window(9, 3, 4).unwrap();
        let table = &prepared.decode.sequences[0];
        assert_eq!(table.blocks, blocks[1..3]);
        assert_eq!(table.first_page_offset, 2);
        assert_eq!(table.context_len, 6);
        assert_eq!(prepared.prefill.len(), 1);
        assert_eq!(prepared.prefill[0].blocks, blocks[1..3]);
        assert_eq!(prepared.prefill[0].query_len, 3);
        assert_eq!(prepared.prefill[0].context_len, 6);
        assert_eq!(prepared.slots.logical_slots()[0].block, blocks[2]);
        assert_eq!(prepared.slots.logical_slots()[0].offset, 1);
        assert_eq!(prepared.slots.logical_slots()[2].offset, 3);

        assert!(cache.prepare_append_with_window(9, 1, 0).is_err());
    }

    #[test]
    fn sliding_window_rotation_recycles_only_fully_invisible_pages() {
        let arena_id = KvArenaId {
            model_instance: ModelInstanceId::new(11),
            backend: BackendKind::Cpu,
            device_ordinal: None,
            generation: 1,
        };
        let group = KvGroupId::new(1);
        let binding = KvLayerBinding {
            model_layer: 0,
            physical_layer: 0,
        };
        let arena = Arc::new(
            CpuKvArena::new(KvArenaConfig {
                id: arena_id,
                group,
                page_tokens: 4,
                capacity_pages: 3,
                growth: None,
                dtype: DType::F32,
                layers: vec![KvLayerConfig {
                    binding,
                    num_kv_heads: 1,
                    key_head_dim: 2,
                    value_head_dim: 2,
                }],
            })
            .unwrap(),
        );
        let blocks = (0..3)
            .map(|index| CacheBlockRef {
                arena: arena_id,
                group,
                index,
                slot_generation: 1,
            })
            .collect::<Vec<_>>();
        let mut cache =
            PhysicalPagedKvCache::new(arena, vec![binding], blocks.clone(), 12).unwrap();

        cache.advance_sliding_window_for_append(12, 1, 8).unwrap();

        assert_eq!(cache.window_start(), 4);
        assert_eq!(cache.capacity_tokens(), 16);
        assert_eq!(cache.slots_for_append(12, 1).unwrap()[0].block, blocks[0]);
    }

    #[test]
    fn prepared_append_commits_every_exact_prefix_without_exposing_suffix() {
        for accepted in 0..=3 {
            let mut cache = prefix_test_cache(accepted as u32 + 1);
            let prepared = fully_written_append(&cache, 3);
            let prepared_slots = prepared.slots.logical_slots().to_vec();

            cache
                .commit_prepared_prefix(prepared, accepted)
                .expect("accepted prefix commit");

            assert_eq!(cache.context_len(), accepted);
            let writes = cache.take_completed_writes();
            if accepted == 0 {
                assert!(writes.is_empty());
            } else {
                assert_eq!(writes.len(), 1);
                assert_eq!(writes[0].slots_per_layer(), accepted);
                assert_eq!(writes[0].slots(), &prepared_slots[..accepted]);
            }
            if accepted < prepared_slots.len() {
                assert_eq!(
                    cache.slots_for_append(accepted, 1).unwrap()[0],
                    prepared_slots[accepted]
                );
            }
        }
    }

    #[test]
    fn abort_prepared_fences_and_discards_the_complete_append() {
        let mut cache = prefix_test_cache(10);
        let prepared = fully_written_append(&cache, 3);
        let first_slot = prepared.slots.logical_slots()[0];

        cache.abort_prepared(prepared).unwrap();

        assert_eq!(cache.context_len(), 0);
        assert!(cache.take_completed_writes().is_empty());
        assert_eq!(cache.slots_for_append(0, 1).unwrap()[0], first_slot);
    }

    #[test]
    fn partial_commit_receipt_composes_with_rewritten_suffix() {
        let mut cache = prefix_test_cache(11);
        let first = fully_written_append(&cache, 3);
        let original_slots = first.slots.logical_slots().to_vec();
        cache.commit_prepared_prefix(first, 1).unwrap();

        let second = fully_written_append(&cache, 2);
        assert_eq!(second.slots.logical_slots().as_ref(), &original_slots[1..]);
        cache.commit_prepared(second).unwrap();

        assert_eq!(cache.context_len(), 3);
        let writes = cache.take_completed_writes();
        assert_eq!(writes.len(), 2);
        assert_eq!(writes[0].slots(), &original_slots[..1]);
        assert_eq!(writes[1].slots(), &original_slots[1..]);
    }

    #[test]
    fn full_prefix_preserves_commit_prepared_behavior() {
        let mut legacy = prefix_test_cache(12);
        let mut prefix = prefix_test_cache(13);
        let legacy_prepared = fully_written_append(&legacy, 3);
        let prefix_prepared = fully_written_append(&prefix, 3);

        legacy.commit_prepared(legacy_prepared).unwrap();
        prefix.commit_prepared_prefix(prefix_prepared, 3).unwrap();

        assert_eq!(legacy.context_len(), 3);
        assert_eq!(prefix.context_len(), 3);
        let legacy_writes = legacy.take_completed_writes();
        let prefix_writes = prefix.take_completed_writes();
        assert_eq!(legacy_writes.len(), 1);
        assert_eq!(prefix_writes.len(), 1);
        assert_eq!(legacy_writes[0].slots_per_layer(), 3);
        assert_eq!(prefix_writes[0].slots_per_layer(), 3);
    }

    #[test]
    fn prefix_commit_rejects_out_of_range_without_mutating_cursor_or_receipts() {
        let mut cache = prefix_test_cache(14);
        let prepared = fully_written_append(&cache, 3);

        let error = cache.commit_prepared_prefix(prepared, 4).unwrap_err();

        assert!(error.to_string().contains("exceeds prepared token count"));
        assert_eq!(cache.context_len(), 0);
        assert!(cache.take_completed_writes().is_empty());
        assert!(cache.prepare_append(0, 3).is_ok());
    }

    #[test]
    fn every_prefix_requires_a_complete_authenticated_layer_batch() {
        for accepted in 0..=3 {
            let mut cache = prefix_test_cache(20 + accepted as u32);
            let mut prepared = cache.prepare_append(0, 3).unwrap();
            submit_prepared_writes(&cache, &mut prepared, 1);

            let error = cache
                .commit_prepared_prefix(prepared, accepted)
                .unwrap_err();

            assert!(error.to_string().contains("missing layer bindings"));
            assert_eq!(cache.context_len(), 0);
            assert!(cache.take_completed_writes().is_empty());
        }
    }

    #[test]
    fn stale_prefix_commit_drains_but_never_advances_the_reset_cache() {
        let mut cache = prefix_test_cache(30);
        let prepared = fully_written_append(&cache, 3);
        cache.reset_invocation().unwrap();

        let error = cache.commit_prepared_prefix(prepared, 2).unwrap_err();

        assert!(error.to_string().contains("stale prepared step"));
        assert_eq!(cache.context_len(), 0);
        assert!(cache.take_completed_writes().is_empty());
    }
}
