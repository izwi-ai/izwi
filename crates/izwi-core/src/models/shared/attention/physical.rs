//! Model-neutral session view over scheduler-owned paged-attention storage.

use std::collections::HashSet;
use std::sync::Arc;

use crate::backends::kv::KvArena;
use crate::error::{Error, Result};
use crate::kv::{CacheBlockRef, KvLayerBinding, KvSequenceBlockTable, KvSlotRef};

/// A generation-pinned logical block table over one physical paged-attention
/// arena. Models retain only this view; K/V tensors remain backend-owned.
pub struct PhysicalPagedKvCache {
    pub(crate) arena: Arc<dyn KvArena>,
    layer_bindings: Vec<KvLayerBinding>,
    pub(crate) blocks: Vec<CacheBlockRef>,
    window_start: usize,
    context_len: usize,
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
        for (expected, binding) in layer_bindings.iter().enumerate() {
            if binding.model_layer as usize != expected {
                return Err(Error::InvalidInput(format!(
                    "physical layer binding {} maps model layer {}, expected {}",
                    binding.physical_layer, binding.model_layer, expected
                )));
            }
        }
        Ok(Self {
            arena,
            layer_bindings,
            blocks,
            window_start,
            context_len,
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
        if context_len <= self.window_start || context_len > self.capacity_tokens() {
            return Err(Error::InvalidInput(format!(
                "physical paged decode context {context_len} is outside cache capacity"
            )));
        }
        let page_tokens = self.arena.config().page_tokens as usize;
        let first_page_offset = self.window_start % page_tokens;
        let visible_tokens = context_len - self.window_start;
        let required_pages = (first_page_offset + visible_tokens).div_ceil(page_tokens);
        Ok(KvSequenceBlockTable {
            blocks: self.blocks[..required_pages].to_vec(),
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

    pub(crate) fn commit_append(&mut self, start_pos: usize, token_count: usize) -> Result<()> {
        self.slots_for_append(start_pos, token_count)?;
        self.context_len += token_count;
        Ok(())
    }
}
