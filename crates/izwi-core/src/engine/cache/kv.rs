//! KV Cache Manager with paged attention support.
//!
//! Implements memory-efficient KV cache management following vLLM's paged attention
//! design. Key features:
//! - Block-based memory allocation
//! - Efficient free block tracking with doubly-linked list
//! - Sequence-to-block mapping
//! - Memory usage tracking

use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use tracing::debug;

use crate::engine::types::{BlockId, RequestId};

/// Physical residency for a KV cache block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheResidency {
    /// Host-resident cache block.
    Cpu,
    /// Device-resident cache block.
    Gpu,
    /// Host block pinned for fast DMA upload to GPU.
    PinnedCpu,
}

/// Handle returned when pinning blocks for a backend execution pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedBlockHandle {
    /// Physical block ID.
    pub block_id: BlockId,
    /// Requested residency for this handle.
    pub residency: CacheResidency,
}

/// Configuration for the KV cache.
#[derive(Debug, Clone)]
pub struct KVCacheConfig {
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads per layer
    pub num_heads: usize,
    /// Dimension of each attention head
    pub head_dim: usize,
    /// Number of tokens per block
    pub block_size: usize,
    /// Maximum number of blocks to allocate
    pub max_blocks: usize,
    /// Data type size in bytes (2 for float16, 4 for float32)
    pub dtype_bytes: usize,
}

impl Default for KVCacheConfig {
    fn default() -> Self {
        Self {
            num_layers: 24,
            num_heads: 16,
            head_dim: 64,
            block_size: 16,
            max_blocks: 1024,
            dtype_bytes: 2, // float16
        }
    }
}

impl KVCacheConfig {
    /// Calculate memory per block in bytes.
    pub fn block_memory_bytes(&self) -> usize {
        // 2 (K+V) * block_size * num_heads * head_dim * dtype_bytes * num_layers
        2 * self.block_size * self.num_heads * self.head_dim * self.dtype_bytes * self.num_layers
    }

    /// Calculate total memory for all blocks.
    pub fn total_memory_bytes(&self) -> usize {
        self.block_memory_bytes() * self.max_blocks
    }

    /// Calculate number of blocks needed for a sequence length.
    pub fn blocks_for_tokens(&self, num_tokens: usize) -> usize {
        (num_tokens + self.block_size - 1) / self.block_size
    }
}

/// A single KV cache block.
#[derive(Debug, Clone)]
pub struct KVBlock {
    /// Block ID
    pub id: BlockId,
    /// Number of tokens stored in this block
    pub num_tokens: usize,
    /// Reference count (for copy-on-write / prefix caching)
    pub ref_count: usize,
    /// Hash of block content (for prefix caching)
    pub content_hash: Option<u64>,
    /// Where this block currently resides.
    pub residency: CacheResidency,
    /// Number of outstanding pins for backend execution.
    pub pin_count: usize,
}

impl KVBlock {
    fn new(id: BlockId) -> Self {
        Self {
            id,
            num_tokens: 0,
            ref_count: 1,
            content_hash: None,
            residency: CacheResidency::Cpu,
            pin_count: 0,
        }
    }

    fn reset(&mut self) {
        self.num_tokens = 0;
        self.ref_count = 1;
        self.content_hash = None;
        self.residency = CacheResidency::Cpu;
        self.pin_count = 0;
    }
}

/// Block allocator using a free list.
pub struct BlockAllocator {
    config: KVCacheConfig,
    /// All blocks
    blocks: Vec<KVBlock>,
    /// Free block IDs (LIFO for cache locality)
    free_list: VecDeque<BlockId>,
    /// Number of allocated blocks
    num_allocated: usize,
    /// Soft allocation limit used by adaptive tuning.
    soft_max_blocks: usize,
}

impl BlockAllocator {
    /// Create a new block allocator.
    pub fn new(config: KVCacheConfig) -> Self {
        let max_blocks = config.max_blocks;
        let blocks: Vec<KVBlock> = (0..max_blocks).map(KVBlock::new).collect();
        let free_list: VecDeque<BlockId> = (0..max_blocks).collect();

        Self {
            config,
            blocks,
            free_list,
            num_allocated: 0,
            soft_max_blocks: max_blocks,
        }
    }

    /// Check if n blocks can be allocated.
    pub fn can_allocate(&self, n: usize) -> bool {
        self.free_list.len() >= n && self.num_allocated + n <= self.soft_max_blocks
    }

    /// Allocate n blocks, returning their IDs.
    /// Uses LIFO allocation - recently freed blocks are reused first for better cache locality.
    pub fn allocate(&mut self, n: usize) -> Option<Vec<BlockId>> {
        if !self.can_allocate(n) {
            return None;
        }

        let mut block_ids = Vec::with_capacity(n);
        for _ in 0..n {
            // Use pop_back for LIFO - recently freed blocks are at the back
            // This improves CPU cache locality as recently used memory is more likely to be hot
            if let Some(id) = self.free_list.pop_back() {
                self.blocks[id].reset();
                block_ids.push(id);
                self.num_allocated += 1;
            }
        }

        Some(block_ids)
    }

    /// Free a single block.
    pub fn free(&mut self, block_id: BlockId) {
        if block_id < self.blocks.len() {
            let block = &mut self.blocks[block_id];
            if block.ref_count == 0 {
                return;
            }
            block.ref_count -= 1;

            if block.ref_count == 0 {
                self.free_list.push_back(block_id);
                self.num_allocated = self.num_allocated.saturating_sub(1);
            }
        }
    }

    /// Increase the reference count of an allocated block.
    pub fn incref(&mut self, block_id: BlockId) -> bool {
        if let Some(block) = self.blocks.get_mut(block_id) {
            if block.ref_count == 0 {
                return false;
            }
            block.ref_count += 1;
            return true;
        }
        false
    }

    /// Free multiple blocks.
    pub fn free_blocks(&mut self, block_ids: &[BlockId]) {
        for &id in block_ids {
            self.free(id);
        }
    }

    /// Get block by ID.
    pub fn get_block(&self, block_id: BlockId) -> Option<&KVBlock> {
        self.blocks.get(block_id)
    }

    /// Get mutable block by ID.
    pub fn get_block_mut(&mut self, block_id: BlockId) -> Option<&mut KVBlock> {
        self.blocks.get_mut(block_id)
    }

    /// Get number of free blocks.
    pub fn num_free(&self) -> usize {
        self.free_list.len()
    }

    /// Get number of allocated blocks.
    pub fn num_allocated(&self) -> usize {
        self.num_allocated
    }

    /// Get total memory used in bytes.
    pub fn memory_used_bytes(&self) -> usize {
        self.num_allocated * self.config.block_memory_bytes()
    }

    /// Get total memory capacity in bytes.
    pub fn memory_capacity_bytes(&self) -> usize {
        self.config.total_memory_bytes()
    }

    /// Current adaptive soft cap.
    pub fn soft_max_blocks(&self) -> usize {
        self.soft_max_blocks
    }

    /// Update adaptive soft cap.
    pub fn set_soft_max_blocks(&mut self, limit: usize) {
        self.soft_max_blocks = limit.clamp(1, self.config.max_blocks);
    }
}

/// KV Cache Manager - manages KV cache for all sequences.
pub struct KVCacheManager {
    config: KVCacheConfig,
    /// Block allocator
    allocator: BlockAllocator,
    /// Mapping from request ID to allocated block IDs
    request_blocks: HashMap<RequestId, Vec<BlockId>>,
    /// Block table: maps (request_id, block_index) to physical block ID
    /// This enables non-contiguous block allocation
    block_table: HashMap<RequestId, Vec<BlockId>>,
    /// Shared prefix cache: prefix hash -> canonical block table.
    shared_prefixes: HashMap<u64, Vec<BlockId>>,
    /// Prefix hash associated with each request when sharing is active.
    request_prefix_hash: HashMap<RequestId, u64>,
    /// Number of active request mappings per shared prefix.
    prefix_ref_counts: HashMap<u64, usize>,
    /// Advanced prefix index keyed by aligned prefix token length.
    shared_prefix_levels: HashMap<usize, HashMap<u64, Vec<BlockId>>>,
    /// Reference counts for advanced prefix index entries.
    shared_prefix_level_ref_counts: HashMap<(usize, u64), usize>,
    /// Prefix index entries registered by request (for cleanup).
    request_prefix_entries: HashMap<RequestId, Vec<(usize, u64)>>,
    /// Persistent prefix snapshots retained across completed request lifetimes.
    persistent_prefix_entries: HashMap<(usize, u64), PersistentPrefixEntry>,
    /// LRU order for persistent prefix snapshots.
    persistent_prefix_lru: VecDeque<(usize, u64)>,
    /// Max persistent prefix snapshots to retain.
    persistent_prefix_max_entries: usize,
    /// Prefix snapshot lifetime in cache-operation ticks.
    persistent_prefix_ttl_ops: u64,
    /// Minimum soft-cap floor for adaptive tuning.
    min_soft_blocks: usize,
    /// Last operation count sampled for churn tuning.
    last_tuned_ops: u64,
    /// Churn and sharing telemetry.
    telemetry: KVCacheTelemetry,
}

#[derive(Debug, Clone)]
struct PersistentPrefixEntry {
    block_ids: Vec<BlockId>,
    last_touched_ops: u64,
}

/// KV cache runtime telemetry.
#[derive(Debug, Clone)]
pub struct KVCacheTelemetry {
    /// Number of fresh physical block allocations.
    pub total_allocations: u64,
    /// Number of physical block frees (refcount reached zero).
    pub total_frees: u64,
    /// Number of shared-prefix hits.
    pub shared_prefix_hits: u64,
    /// Number of copy-on-write splits.
    pub copy_on_write_splits: u64,
    /// Last sampled churn ratio used for tuning.
    pub last_churn_ratio: f64,
    /// Current adaptive soft cap.
    pub soft_max_blocks: usize,
}

impl KVCacheTelemetry {
    fn total_ops(&self) -> u64 {
        self.total_allocations + self.total_frees + self.copy_on_write_splits
    }
}

impl KVCacheManager {
    /// Create a new KV cache manager.
    pub fn new(config: KVCacheConfig) -> Self {
        let allocator = BlockAllocator::new(config.clone());
        let min_soft_blocks = (config.max_blocks / 4)
            .max(32)
            .min(config.max_blocks.max(1));
        let soft_max_blocks = allocator.soft_max_blocks();

        Self {
            config,
            allocator,
            request_blocks: HashMap::new(),
            block_table: HashMap::new(),
            shared_prefixes: HashMap::new(),
            request_prefix_hash: HashMap::new(),
            prefix_ref_counts: HashMap::new(),
            shared_prefix_levels: HashMap::new(),
            shared_prefix_level_ref_counts: HashMap::new(),
            request_prefix_entries: HashMap::new(),
            persistent_prefix_entries: HashMap::new(),
            persistent_prefix_lru: VecDeque::new(),
            persistent_prefix_max_entries: 64,
            persistent_prefix_ttl_ops: 512,
            min_soft_blocks,
            last_tuned_ops: 0,
            telemetry: KVCacheTelemetry {
                total_allocations: 0,
                total_frees: 0,
                shared_prefix_hits: 0,
                copy_on_write_splits: 0,
                last_churn_ratio: 0.0,
                soft_max_blocks,
            },
        }
    }

    /// Check if n blocks can be allocated, reclaiming persistent snapshots on pressure.
    pub fn can_allocate(&mut self, n: usize) -> bool {
        self.ensure_capacity_for(n)
    }

    /// Allocate blocks for a request.
    pub fn allocate(&mut self, request_id: &RequestId, num_blocks: usize) -> Vec<BlockId> {
        self.allocate_with_prefix(request_id, num_blocks, None)
    }

    /// Allocate blocks and try to reuse shared prefix blocks when available.
    pub fn allocate_with_prefix(
        &mut self,
        request_id: &RequestId,
        num_blocks: usize,
        prefix_hash: Option<u64>,
    ) -> Vec<BlockId> {
        if num_blocks == 0 {
            return Vec::new();
        }

        let mut block_ids = Vec::with_capacity(num_blocks);
        let mut prefix_used = None;

        if let Some(hash) = prefix_hash {
            if let Some(shared) = self.shared_prefixes.get(&hash).cloned() {
                for block_id in shared.into_iter().take(num_blocks) {
                    if self.allocator.incref(block_id) {
                        block_ids.push(block_id);
                    }
                }
                if !block_ids.is_empty() {
                    self.telemetry.shared_prefix_hits += 1;
                    prefix_used = Some(hash);
                }
            }
        }

        let remaining = num_blocks.saturating_sub(block_ids.len());
        if remaining > 0 {
            if !self.ensure_capacity_for(remaining) {
                self.allocator.free_blocks(&block_ids);
                return Vec::new();
            }
            if let Some(mut fresh_blocks) = self.allocator.allocate(remaining) {
                self.telemetry.total_allocations += fresh_blocks.len() as u64;
                block_ids.append(&mut fresh_blocks);
            } else {
                // Roll back shared refs if we couldn't allocate tail blocks.
                self.allocator.free_blocks(&block_ids);
                return Vec::new();
            }
        }

        if block_ids.is_empty() {
            return block_ids;
        }

        self.request_blocks
            .entry(request_id.clone())
            .or_default()
            .extend(block_ids.iter().copied());
        self.block_table
            .entry(request_id.clone())
            .or_default()
            .extend(block_ids.iter().copied());

        // Register this request with shared-prefix bookkeeping if applicable.
        if let Some(hash) = prefix_hash {
            if !self.shared_prefixes.contains_key(&hash) {
                self.shared_prefixes.insert(hash, block_ids.clone());
                prefix_used = Some(hash);
            }
            if let Some(active_hash) = prefix_used {
                if !self.request_prefix_hash.contains_key(request_id) {
                    self.request_prefix_hash
                        .insert(request_id.clone(), active_hash);
                    *self.prefix_ref_counts.entry(active_hash).or_insert(0) += 1;
                }
            }
        }

        self.maybe_tune_soft_limit();

        debug!(
            "Allocated {} blocks for request {}: {:?}",
            num_blocks, request_id, block_ids
        );

        block_ids
    }

    /// Allocate additional blocks for an existing request (for extension during decode).
    pub fn extend(&mut self, request_id: &RequestId, additional_blocks: usize) -> Vec<BlockId> {
        self.allocate(request_id, additional_blocks)
    }

    fn blocks_are_live(&self, candidate_blocks: &[BlockId], required_blocks: usize) -> bool {
        if candidate_blocks.len() < required_blocks {
            return false;
        }
        candidate_blocks
            .iter()
            .take(required_blocks)
            .all(|block_id| {
                self.allocator
                    .get_block(*block_id)
                    .map(|block| block.ref_count > 0)
                    .unwrap_or(false)
            })
    }

    fn lookup_prefix_candidate_blocks(
        &mut self,
        token_len: usize,
        prefix_hash: u64,
    ) -> Option<Vec<BlockId>> {
        if let Some(level) = self.shared_prefix_levels.get(&token_len) {
            if let Some(candidate_blocks) = level.get(&prefix_hash) {
                return Some(candidate_blocks.clone());
            }
        }

        let key = (token_len, prefix_hash);
        let now_ops = self.telemetry.total_ops();
        if let Some(entry) = self.persistent_prefix_entries.get_mut(&key) {
            entry.last_touched_ops = now_ops;
            self.persistent_prefix_lru
                .retain(|existing| existing != &key);
            self.persistent_prefix_lru.push_back(key);
            return Some(entry.block_ids.clone());
        }
        None
    }

    /// Estimate how many leading prompt blocks can be reused from shared-prefix cache.
    pub fn estimate_prefix_reuse_blocks(&self, prompt_tokens: &[u32], num_blocks: usize) -> usize {
        if num_blocks == 0 || prompt_tokens.is_empty() {
            return 0;
        }

        let block_size = self.config.block_size.max(1);
        let max_reusable_blocks = (prompt_tokens.len() / block_size).min(num_blocks);

        if max_reusable_blocks == 0 {
            return 0;
        }

        for blocks in (1..=max_reusable_blocks).rev() {
            let prefix_tokens = blocks * block_size;
            let prefix_hash = Self::hash_prefix_tokens(&prompt_tokens[..prefix_tokens]);
            if let Some(level) = self.shared_prefix_levels.get(&prefix_tokens) {
                if let Some(candidate_blocks) = level.get(&prefix_hash) {
                    if self.blocks_are_live(candidate_blocks, blocks) {
                        return blocks;
                    }
                }
            }
            if let Some(entry) = self
                .persistent_prefix_entries
                .get(&(prefix_tokens, prefix_hash))
            {
                if self.blocks_are_live(&entry.block_ids, blocks) {
                    return blocks;
                }
            }
        }

        0
    }

    /// Allocate blocks with block-granular prefix matching.
    ///
    /// Unlike `allocate_with_prefix`, this can reuse partial prefixes at block granularity
    /// by matching progressively shorter aligned token prefixes.
    pub fn allocate_with_prefix_tokens(
        &mut self,
        request_id: &RequestId,
        num_blocks: usize,
        prompt_tokens: &[u32],
    ) -> Vec<BlockId> {
        if num_blocks == 0 {
            return Vec::new();
        }

        let block_size = self.config.block_size.max(1);
        let mut reused = Vec::new();
        let max_reusable_blocks = (prompt_tokens.len() / block_size).min(num_blocks);

        if max_reusable_blocks > 0 {
            for blocks in (1..=max_reusable_blocks).rev() {
                let prefix_tokens = blocks * block_size;
                let prefix_hash = Self::hash_prefix_tokens(&prompt_tokens[..prefix_tokens]);
                if let Some(candidate_blocks) =
                    self.lookup_prefix_candidate_blocks(prefix_tokens, prefix_hash)
                {
                    let mut acquired = Vec::with_capacity(blocks);
                    let mut ok = true;
                    for block_id in candidate_blocks.iter().take(blocks) {
                        if self.allocator.incref(*block_id) {
                            acquired.push(*block_id);
                        } else {
                            ok = false;
                            break;
                        }
                    }
                    if ok && acquired.len() == blocks {
                        reused = acquired;
                        self.telemetry.shared_prefix_hits += 1;
                        break;
                    }
                    if !acquired.is_empty() {
                        self.allocator.free_blocks(&acquired);
                    }
                }
            }
        }

        let mut block_ids = reused;
        let remaining = num_blocks.saturating_sub(block_ids.len());
        if remaining > 0 {
            if !self.ensure_capacity_for(remaining) {
                self.allocator.free_blocks(&block_ids);
                return Vec::new();
            }
            if let Some(mut fresh_blocks) = self.allocator.allocate(remaining) {
                self.telemetry.total_allocations += fresh_blocks.len() as u64;
                block_ids.append(&mut fresh_blocks);
            } else {
                self.allocator.free_blocks(&block_ids);
                return Vec::new();
            }
        }

        if block_ids.is_empty() {
            return block_ids;
        }

        self.request_blocks
            .entry(request_id.clone())
            .or_default()
            .extend(block_ids.iter().copied());
        self.block_table
            .entry(request_id.clone())
            .or_default()
            .extend(block_ids.iter().copied());

        self.register_prefix_levels_for_request(request_id, &block_ids, prompt_tokens);
        self.maybe_tune_soft_limit();

        debug!(
            "Allocated {} blocks for request {} with advanced prefix reuse: {:?}",
            num_blocks, request_id, block_ids
        );

        block_ids
    }

    /// Free all blocks for a request.
    pub fn free(&mut self, request_id: &RequestId) {
        let released_blocks = self.request_blocks.remove(request_id).unwrap_or_default();
        self.persist_request_prefix_snapshot(request_id, &released_blocks);
        if !released_blocks.is_empty() {
            let allocated_before = self.allocator.num_allocated();
            debug!(
                "Freeing {} blocks for request {}: {:?}",
                released_blocks.len(),
                request_id,
                released_blocks
            );
            self.allocator.free_blocks(&released_blocks);
            let freed = allocated_before.saturating_sub(self.allocator.num_allocated());
            self.telemetry.total_frees += freed as u64;
        }
        self.cleanup_prefix_levels_for_request(request_id);
        if let Some(prefix_hash) = self.request_prefix_hash.remove(request_id) {
            if let Some(ref_count) = self.prefix_ref_counts.get_mut(&prefix_hash) {
                *ref_count = ref_count.saturating_sub(1);
                if *ref_count == 0 {
                    self.prefix_ref_counts.remove(&prefix_hash);
                    self.shared_prefixes.remove(&prefix_hash);
                }
            }
        }
        self.block_table.remove(request_id);
        self.maybe_tune_soft_limit();
    }

    fn hash_prefix_tokens(tokens: &[u32]) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        tokens.len().hash(&mut hasher);
        for token in tokens {
            token.hash(&mut hasher);
        }
        hasher.finish()
    }

    fn register_prefix_levels_for_request(
        &mut self,
        request_id: &RequestId,
        block_ids: &[BlockId],
        prompt_tokens: &[u32],
    ) {
        if block_ids.is_empty() || prompt_tokens.is_empty() {
            return;
        }
        let block_size = self.config.block_size.max(1);
        let max_prefix_blocks = (prompt_tokens.len() / block_size).min(block_ids.len());
        if max_prefix_blocks == 0 {
            return;
        }
        let mut request_entries = Vec::with_capacity(max_prefix_blocks);

        for blocks in 1..=max_prefix_blocks {
            let token_len = blocks * block_size;
            let hash = Self::hash_prefix_tokens(&prompt_tokens[..token_len]);
            self.shared_prefix_levels
                .entry(token_len)
                .or_default()
                .entry(hash)
                .or_insert_with(|| block_ids[..blocks].to_vec());
            *self
                .shared_prefix_level_ref_counts
                .entry((token_len, hash))
                .or_insert(0) += 1;
            request_entries.push((token_len, hash));
        }

        self.request_prefix_entries
            .entry(request_id.clone())
            .or_default()
            .extend(request_entries);
    }

    fn cleanup_prefix_levels_for_request(&mut self, request_id: &RequestId) {
        let Some(entries) = self.request_prefix_entries.remove(request_id) else {
            return;
        };

        for (token_len, hash) in entries {
            let key = (token_len, hash);
            if let Some(ref_count) = self.shared_prefix_level_ref_counts.get_mut(&key) {
                *ref_count = ref_count.saturating_sub(1);
                if *ref_count == 0 {
                    self.shared_prefix_level_ref_counts.remove(&key);
                    let mut level_empty = false;
                    if let Some(level) = self.shared_prefix_levels.get_mut(&token_len) {
                        level.remove(&hash);
                        level_empty = level.is_empty();
                    }
                    if level_empty {
                        self.shared_prefix_levels.remove(&token_len);
                    }
                }
            }
        }
    }

    fn persist_request_prefix_snapshot(
        &mut self,
        request_id: &RequestId,
        released_blocks: &[BlockId],
    ) {
        if released_blocks.is_empty() {
            return;
        }
        let Some(entries) = self.request_prefix_entries.get(request_id) else {
            return;
        };
        let Some((token_len, hash)) = entries.iter().max_by_key(|(token_len, _)| *token_len) else {
            return;
        };

        let block_size = self.config.block_size.max(1);
        let prefix_blocks = token_len / block_size;
        if prefix_blocks == 0 || prefix_blocks > released_blocks.len() {
            return;
        }

        let block_ids = released_blocks[..prefix_blocks].to_vec();
        self.insert_persistent_prefix_entry(*token_len, *hash, block_ids);
    }

    fn insert_persistent_prefix_entry(
        &mut self,
        token_len: usize,
        hash: u64,
        block_ids: Vec<BlockId>,
    ) {
        if block_ids.is_empty() {
            return;
        }
        let key = (token_len, hash);
        let now_ops = self.telemetry.total_ops();

        if let Some(entry) = self.persistent_prefix_entries.get_mut(&key) {
            entry.last_touched_ops = now_ops;
            self.persistent_prefix_lru
                .retain(|existing| existing != &key);
            self.persistent_prefix_lru.push_back(key);
            return;
        }

        let mut acquired = Vec::with_capacity(block_ids.len());
        for block_id in &block_ids {
            if self.allocator.incref(*block_id) {
                acquired.push(*block_id);
            } else {
                if !acquired.is_empty() {
                    self.allocator.free_blocks(&acquired);
                }
                return;
            }
        }

        self.persistent_prefix_entries.insert(
            key,
            PersistentPrefixEntry {
                block_ids,
                last_touched_ops: now_ops,
            },
        );
        self.persistent_prefix_lru
            .retain(|existing| existing != &key);
        self.persistent_prefix_lru.push_back(key);
        self.evict_persistent_prefix_entries_if_needed();
    }

    fn evict_persistent_prefix_entry(&mut self, key: (usize, u64)) {
        let Some(entry) = self.persistent_prefix_entries.remove(&key) else {
            return;
        };
        self.persistent_prefix_lru
            .retain(|existing| existing != &key);
        let allocated_before = self.allocator.num_allocated();
        self.allocator.free_blocks(&entry.block_ids);
        let freed = allocated_before.saturating_sub(self.allocator.num_allocated());
        self.telemetry.total_frees += freed as u64;
    }

    fn evict_persistent_prefix_entries_if_needed(&mut self) {
        while self.persistent_prefix_entries.len() > self.persistent_prefix_max_entries {
            let Some(key) = self.persistent_prefix_lru.pop_front() else {
                break;
            };
            self.evict_persistent_prefix_entry(key);
        }
    }

    fn evict_persistent_prefixes_for_capacity(&mut self, required_blocks: usize) {
        if required_blocks == 0 {
            return;
        }
        while !self.allocator.can_allocate(required_blocks) {
            let Some(key) = self.persistent_prefix_lru.front().copied() else {
                break;
            };
            self.evict_persistent_prefix_entry(key);
        }
    }

    fn ensure_capacity_for(&mut self, required_blocks: usize) -> bool {
        if required_blocks == 0 {
            return true;
        }
        if self.allocator.can_allocate(required_blocks) {
            return true;
        }

        // First drop stale snapshots, then LRU-evict snapshots until headroom exists.
        self.maybe_evict_stale_persistent_prefixes();
        if self.allocator.can_allocate(required_blocks) {
            return true;
        }
        self.evict_persistent_prefixes_for_capacity(required_blocks);
        self.allocator.can_allocate(required_blocks)
    }

    fn maybe_evict_stale_persistent_prefixes(&mut self) {
        if self.persistent_prefix_entries.is_empty() {
            return;
        }

        let now_ops = self.telemetry.total_ops();
        let stale_keys: Vec<(usize, u64)> = self
            .persistent_prefix_entries
            .iter()
            .filter_map(|(key, entry)| {
                if now_ops.saturating_sub(entry.last_touched_ops) > self.persistent_prefix_ttl_ops {
                    Some(*key)
                } else {
                    None
                }
            })
            .collect();
        for key in stale_keys {
            self.evict_persistent_prefix_entry(key);
        }
    }

    /// Get blocks allocated to a request.
    pub fn get_blocks(&self, request_id: &RequestId) -> Option<&[BlockId]> {
        self.request_blocks.get(request_id).map(|v| v.as_slice())
    }

    /// Get the block table for a request.
    pub fn get_block_table(&self, request_id: &RequestId) -> Option<&[BlockId]> {
        self.block_table.get(request_id).map(|v| v.as_slice())
    }

    /// Update token count in a block.
    pub fn update_block_tokens(&mut self, block_id: BlockId, num_tokens: usize) {
        if let Some(block) = self.allocator.get_block_mut(block_id) {
            block.num_tokens = num_tokens;
        }
    }

    /// Set residency for a physical block.
    pub fn set_block_residency(&mut self, block_id: BlockId, residency: CacheResidency) -> bool {
        if let Some(block) = self.allocator.get_block_mut(block_id) {
            block.residency = residency;
            return true;
        }
        false
    }

    /// Set residency for all blocks mapped to a request.
    pub fn set_request_residency(&mut self, request_id: &RequestId, residency: CacheResidency) {
        let block_ids = self
            .block_table
            .get(request_id)
            .cloned()
            .unwrap_or_default();
        for block_id in block_ids {
            let _ = self.set_block_residency(block_id, residency);
        }
    }

    /// Get residency for a specific block.
    pub fn block_residency(&self, block_id: BlockId) -> Option<CacheResidency> {
        self.allocator.get_block(block_id).map(|b| b.residency)
    }

    /// Pin all blocks for a request and return backend handles.
    pub fn pin_request_blocks(
        &mut self,
        request_id: &RequestId,
        residency: CacheResidency,
    ) -> Vec<PinnedBlockHandle> {
        let block_ids = self
            .block_table
            .get(request_id)
            .cloned()
            .unwrap_or_default();
        let mut handles = Vec::with_capacity(block_ids.len());

        for block_id in block_ids {
            if let Some(block) = self.allocator.get_block_mut(block_id) {
                block.pin_count += 1;
                block.residency = residency;
                handles.push(PinnedBlockHandle {
                    block_id,
                    residency,
                });
            }
        }

        handles
    }

    /// Release pinned block handles.
    pub fn unpin_blocks(&mut self, handles: &[PinnedBlockHandle]) {
        for handle in handles {
            if let Some(block) = self.allocator.get_block_mut(handle.block_id) {
                block.pin_count = block.pin_count.saturating_sub(1);
                if block.pin_count == 0 && block.residency == CacheResidency::PinnedCpu {
                    block.residency = CacheResidency::Cpu;
                }
            }
        }
    }

    /// Ensure the selected block is writable by this request (copy-on-write).
    pub fn ensure_writable_block(
        &mut self,
        request_id: &RequestId,
        logical_block_idx: usize,
    ) -> Option<BlockId> {
        let block_id = *self.block_table.get(request_id)?.get(logical_block_idx)?;

        let shared = self
            .allocator
            .get_block(block_id)
            .map(|b| b.ref_count > 1)
            .unwrap_or(false);
        if !shared {
            return Some(block_id);
        }
        if !self.ensure_capacity_for(1) {
            return None;
        }

        let mut new_blocks = self.allocator.allocate(1)?;
        let new_block_id = new_blocks.pop()?;
        let (num_tokens, content_hash, residency) = self
            .allocator
            .get_block(block_id)
            .map(|b| (b.num_tokens, b.content_hash, b.residency))?;
        if let Some(new_block) = self.allocator.get_block_mut(new_block_id) {
            new_block.num_tokens = num_tokens;
            new_block.content_hash = content_hash;
            new_block.residency = residency;
        }

        if let Some(table) = self.block_table.get_mut(request_id) {
            if logical_block_idx < table.len() {
                table[logical_block_idx] = new_block_id;
            }
        }
        if let Some(request_blocks) = self.request_blocks.get_mut(request_id) {
            if logical_block_idx < request_blocks.len() {
                request_blocks[logical_block_idx] = new_block_id;
            }
        }

        self.allocator.free(block_id);
        self.telemetry.total_allocations += 1;
        self.telemetry.copy_on_write_splits += 1;
        self.maybe_tune_soft_limit();

        Some(new_block_id)
    }

    /// Ensure the tail block for a request is writable (for append during decode).
    pub fn ensure_writable_last_block(&mut self, request_id: &RequestId) -> Option<BlockId> {
        let last_idx = self
            .block_table
            .get(request_id)
            .and_then(|table| table.len().checked_sub(1))?;
        self.ensure_writable_block(request_id, last_idx)
    }

    /// Get number of blocks needed for a number of tokens.
    pub fn blocks_for_tokens(&self, num_tokens: usize) -> usize {
        self.config.blocks_for_tokens(num_tokens)
    }

    /// Get statistics.
    pub fn stats(&self) -> KVCacheStats {
        let mut gpu_resident_blocks = 0;
        let mut pinned_blocks = 0;
        for block in &self.allocator.blocks {
            if block.ref_count == 0 {
                continue;
            }
            if block.residency == CacheResidency::Gpu {
                gpu_resident_blocks += 1;
            }
            if block.pin_count > 0 || block.residency == CacheResidency::PinnedCpu {
                pinned_blocks += 1;
            }
        }

        KVCacheStats {
            total_blocks: self.config.max_blocks,
            soft_max_blocks: self.allocator.soft_max_blocks(),
            allocated_blocks: self.allocator.num_allocated(),
            free_blocks: self.allocator.num_free(),
            num_sequences: self.request_blocks.len(),
            memory_used_bytes: self.allocator.memory_used_bytes(),
            memory_capacity_bytes: self.allocator.memory_capacity_bytes(),
            gpu_resident_blocks,
            pinned_blocks,
            shared_prefixes: self.shared_prefixes.len() + self.persistent_prefix_entries.len(),
            telemetry: self.telemetry.clone(),
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &KVCacheConfig {
        &self.config
    }

    /// Telemetry snapshot.
    pub fn telemetry(&self) -> &KVCacheTelemetry {
        &self.telemetry
    }

    /// Get the current soft maximum block limit.
    pub fn soft_max_blocks(&self) -> usize {
        self.allocator.soft_max_blocks()
    }

    /// Set the soft maximum block limit.
    pub fn set_soft_max_blocks(&mut self, limit: usize) {
        self.allocator.set_soft_max_blocks(limit);
        self.telemetry.soft_max_blocks = limit;
    }

    /// Get the number of shared prefixes.
    pub fn shared_prefix_count(&self) -> usize {
        self.shared_prefixes.len() + self.persistent_prefix_entries.len()
    }

    /// Compact shared prefixes by removing unused ones.
    pub fn compact_shared_prefixes(&mut self) {
        let before_count = self.shared_prefixes.len();

        // Remove prefixes that are not actually shared (only 1 reference)
        self.shared_prefixes
            .retain(|hash, _blocks| self.prefix_ref_counts.get(hash).copied().unwrap_or(0) > 1);

        let after_count = self.shared_prefixes.len();
        if before_count != after_count {
            tracing::debug!(
                "Compacted shared prefixes: {} -> {}",
                before_count,
                after_count
            );
        }
    }

    fn maybe_tune_soft_limit(&mut self) {
        self.maybe_evict_stale_persistent_prefixes();
        self.evict_persistent_prefix_entries_if_needed();

        let total_ops = self.telemetry.total_ops();
        let delta_ops = total_ops.saturating_sub(self.last_tuned_ops);
        if delta_ops < 64 {
            self.telemetry.soft_max_blocks = self.allocator.soft_max_blocks();
            return;
        }
        self.last_tuned_ops = total_ops;

        let allocated = self.allocator.num_allocated().max(1);
        let churn_ratio = delta_ops as f64 / allocated as f64;
        self.telemetry.last_churn_ratio = churn_ratio;

        let step = (self.config.max_blocks / 16).max(1);
        let util =
            self.allocator.num_allocated() as f64 / self.allocator.soft_max_blocks().max(1) as f64;
        let mut new_soft_limit = self.allocator.soft_max_blocks();

        if churn_ratio > 2.0 && util > 0.7 {
            new_soft_limit = (new_soft_limit + step).min(self.config.max_blocks);
        } else if churn_ratio < 0.4 && util < 0.35 {
            new_soft_limit = new_soft_limit
                .saturating_sub(step)
                .max(self.min_soft_blocks);
        }

        self.allocator.set_soft_max_blocks(new_soft_limit);
        self.telemetry.soft_max_blocks = self.allocator.soft_max_blocks();
    }
}

/// KV cache statistics.
#[derive(Debug, Clone)]
pub struct KVCacheStats {
    pub total_blocks: usize,
    pub soft_max_blocks: usize,
    pub allocated_blocks: usize,
    pub free_blocks: usize,
    pub num_sequences: usize,
    pub memory_used_bytes: usize,
    pub memory_capacity_bytes: usize,
    pub gpu_resident_blocks: usize,
    pub pinned_blocks: usize,
    pub shared_prefixes: usize,
    pub telemetry: KVCacheTelemetry,
}

impl KVCacheStats {
    /// Get utilization as a percentage (0.0 - 1.0).
    pub fn utilization(&self) -> f64 {
        if self.total_blocks == 0 {
            return 0.0;
        }
        self.allocated_blocks as f64 / self.total_blocks as f64
    }
}

// ============================================================================
// Streaming KV Cache for Continuous Audio Prefill
// ============================================================================

/// Configuration for the streaming KV cache.
#[derive(Debug, Clone)]
pub struct StreamingKVCacheConfig {
    /// Base KV cache configuration
    pub base_config: KVCacheConfig,
    /// Maximum context window size (tokens)
    pub max_context_tokens: usize,
    /// Sliding window size for context (tokens)
    pub sliding_window_tokens: usize,
    /// Enable token eviction when window is full
    pub enable_eviction: bool,
    /// Number of tokens to evict at once when needed
    pub eviction_batch_size: usize,
}

impl Default for StreamingKVCacheConfig {
    fn default() -> Self {
        Self {
            base_config: KVCacheConfig::default(),
            max_context_tokens: 4096,
            sliding_window_tokens: 2048,
            enable_eviction: true,
            eviction_batch_size: 256,
        }
    }
}

/// Token position in the streaming context.
#[derive(Debug, Clone, Copy)]
pub struct TokenPosition {
    /// Absolute position since stream start
    pub absolute: usize,
    /// Position within current window
    pub window: usize,
    /// Block index containing this token
    pub block_idx: usize,
    /// Offset within the block
    pub block_offset: usize,
}

/// Streaming sequence state for continuous prefill.
#[derive(Debug, Clone)]
pub struct StreamingSequence {
    /// Request ID
    pub request_id: String,
    /// Current token count in the sequence
    pub token_count: usize,
    /// Absolute token position (total tokens seen)
    pub absolute_position: usize,
    /// Window start position (for sliding window)
    pub window_start: usize,
    /// Allocated block IDs
    pub block_ids: Vec<BlockId>,
    /// Whether this sequence is in prefill mode
    pub in_prefill: bool,
    /// Last update timestamp
    pub last_update: std::time::Instant,
}

impl StreamingSequence {
    /// Create a new streaming sequence.
    pub fn new(request_id: String) -> Self {
        Self {
            request_id,
            token_count: 0,
            absolute_position: 0,
            window_start: 0,
            block_ids: Vec::new(),
            in_prefill: true,
            last_update: std::time::Instant::now(),
        }
    }

    /// Get current window size.
    pub fn window_size(&self) -> usize {
        self.absolute_position - self.window_start
    }

    /// Check if window needs to slide.
    pub fn needs_window_slide(&self, max_window: usize) -> bool {
        self.window_size() >= max_window
    }
}

/// Streaming KV Cache Manager for continuous audio prefill.
///
/// Unlike the standard KV cache which grows linearly, this cache
/// implements a sliding window approach suitable for streaming audio:
/// - Continuous token ingestion during speech
/// - Window-based context management
/// - Automatic eviction of old tokens
pub struct StreamingKVCacheManager {
    config: StreamingKVCacheConfig,
    /// Base allocator
    allocator: BlockAllocator,
    /// Active streaming sequences
    sequences: HashMap<String, StreamingSequence>,
    /// Block table for each sequence
    block_table: HashMap<String, Vec<BlockId>>,
    /// Statistics
    total_tokens_processed: u64,
    total_evictions: u64,
}

impl StreamingKVCacheManager {
    /// Create a new streaming KV cache manager.
    pub fn new(config: StreamingKVCacheConfig) -> Self {
        let allocator = BlockAllocator::new(config.base_config.clone());

        Self {
            config,
            allocator,
            sequences: HashMap::new(),
            block_table: HashMap::new(),
            total_tokens_processed: 0,
            total_evictions: 0,
        }
    }

    /// Start a new streaming sequence.
    pub fn start_sequence(&mut self, request_id: &str) -> bool {
        if self.sequences.contains_key(request_id) {
            return false;
        }

        let sequence = StreamingSequence::new(request_id.to_string());
        self.sequences.insert(request_id.to_string(), sequence);
        self.block_table.insert(request_id.to_string(), Vec::new());
        true
    }

    /// Append tokens to a streaming sequence.
    ///
    /// This is the core method for continuous prefill - it handles:
    /// - Block allocation for new tokens
    /// - Window sliding when needed
    /// - Eviction of old tokens
    pub fn append_tokens(&mut self, request_id: &str, num_tokens: usize) -> Option<Vec<BlockId>> {
        let config = self.config.clone();

        // Check if sequence exists and if we need eviction
        let needs_eviction = {
            let sequence = self.sequences.get_mut(request_id)?;
            sequence.last_update = std::time::Instant::now();
            sequence.needs_window_slide(config.sliding_window_tokens) && config.enable_eviction
        };

        // Evict if needed (separate borrow scope)
        if needs_eviction {
            self.evict_old_tokens(request_id);
        }

        // Calculate blocks needed
        let (window_size, current_blocks) = {
            let sequence = self.sequences.get(request_id)?;
            let current = self
                .block_table
                .get(request_id)
                .map(|b| b.len())
                .unwrap_or(0);
            (sequence.window_size(), current)
        };

        let total_tokens_after = window_size + num_tokens;
        let blocks_needed = config.base_config.blocks_for_tokens(total_tokens_after);

        let additional_blocks = if blocks_needed > current_blocks {
            blocks_needed - current_blocks
        } else {
            0
        };

        // Allocate additional blocks if needed
        if additional_blocks > 0 {
            if !self.allocator.can_allocate(additional_blocks) {
                // Try to evict from other sequences
                if !self.try_evict_for_space(additional_blocks, request_id) {
                    return None;
                }
            }

            if let Some(blocks) = self.allocator.allocate(additional_blocks) {
                if let Some(table) = self.block_table.get_mut(request_id) {
                    table.extend(blocks);
                }
            } else {
                return None;
            }
        }

        // Update sequence state
        if let Some(seq) = self.sequences.get_mut(request_id) {
            seq.token_count += num_tokens;
            seq.absolute_position += num_tokens;
            seq.block_ids = self
                .block_table
                .get(request_id)
                .cloned()
                .unwrap_or_default();
        }

        self.total_tokens_processed += num_tokens as u64;

        Some(
            self.block_table
                .get(request_id)
                .cloned()
                .unwrap_or_default(),
        )
    }

    /// Evict old tokens from a sequence to make room for new ones.
    fn evict_old_tokens(&mut self, request_id: &str) {
        let config = self.config.clone();

        let Some(sequence) = self.sequences.get_mut(request_id) else {
            return;
        };

        let tokens_to_evict = config.eviction_batch_size;
        let blocks_to_free = config.base_config.blocks_for_tokens(tokens_to_evict);

        // Free oldest blocks
        if let Some(table) = self.block_table.get_mut(request_id) {
            let blocks_freed: Vec<BlockId> =
                table.drain(..blocks_to_free.min(table.len())).collect();
            for block_id in blocks_freed {
                self.allocator.free(block_id);
            }
        }

        // Update window start
        sequence.window_start += tokens_to_evict;
        sequence.token_count = sequence.token_count.saturating_sub(tokens_to_evict);

        self.total_evictions += 1;
    }

    /// Try to evict from other sequences to make space.
    fn try_evict_for_space(&mut self, blocks_needed: usize, exclude_request: &str) -> bool {
        let mut blocks_freed = 0;

        // Find candidates for eviction (oldest sequences first)
        let mut candidates: Vec<_> = self
            .sequences
            .iter()
            .filter(|(id, _)| *id != exclude_request)
            .map(|(id, seq)| (id.clone(), seq.last_update, seq.block_ids.len()))
            .collect();

        candidates.sort_by_key(|(_, time, _)| *time);

        for (request_id, _, _num_blocks) in candidates {
            if blocks_freed >= blocks_needed {
                break;
            }

            // Evict entire sequence if needed
            if let Some(table) = self.block_table.get(&request_id) {
                for &block_id in table {
                    self.allocator.free(block_id);
                    blocks_freed += 1;
                }
            }
            self.block_table.remove(&request_id);
            self.sequences.remove(&request_id);
        }

        blocks_freed >= blocks_needed
    }

    /// End a streaming sequence and free all resources.
    pub fn end_sequence(&mut self, request_id: &str) {
        if let Some(table) = self.block_table.remove(request_id) {
            for block_id in table {
                self.allocator.free(block_id);
            }
        }
        self.sequences.remove(request_id);
    }

    /// Get sequence info.
    pub fn get_sequence(&self, request_id: &str) -> Option<&StreamingSequence> {
        self.sequences.get(request_id)
    }

    /// Get block table for a sequence.
    pub fn get_block_table(&self, request_id: &str) -> Option<&[BlockId]> {
        self.block_table.get(request_id).map(|v| v.as_slice())
    }

    /// Mark sequence as transitioning from prefill to decode.
    pub fn end_prefill(&mut self, request_id: &str) {
        if let Some(seq) = self.sequences.get_mut(request_id) {
            seq.in_prefill = false;
        }
    }

    /// Check if sequence is in prefill mode.
    pub fn is_in_prefill(&self, request_id: &str) -> bool {
        self.sequences
            .get(request_id)
            .map(|s| s.in_prefill)
            .unwrap_or(false)
    }

    /// Get statistics.
    pub fn stats(&self) -> StreamingKVCacheStats {
        StreamingKVCacheStats {
            base_stats: KVCacheStats {
                total_blocks: self.config.base_config.max_blocks,
                soft_max_blocks: self.allocator.soft_max_blocks(),
                allocated_blocks: self.allocator.num_allocated(),
                free_blocks: self.allocator.num_free(),
                num_sequences: self.sequences.len(),
                memory_used_bytes: self.allocator.memory_used_bytes(),
                memory_capacity_bytes: self.allocator.memory_capacity_bytes(),
                gpu_resident_blocks: 0,
                pinned_blocks: 0,
                shared_prefixes: 0,
                telemetry: KVCacheTelemetry {
                    total_allocations: 0,
                    total_frees: 0,
                    shared_prefix_hits: 0,
                    copy_on_write_splits: 0,
                    last_churn_ratio: 0.0,
                    soft_max_blocks: self.allocator.soft_max_blocks(),
                },
            },
            total_tokens_processed: self.total_tokens_processed,
            total_evictions: self.total_evictions,
            active_sequences: self.sequences.len(),
            sequences_in_prefill: self.sequences.values().filter(|s| s.in_prefill).count(),
        }
    }
}

/// Statistics for the streaming KV cache.
#[derive(Debug, Clone)]
pub struct StreamingKVCacheStats {
    /// Base cache statistics
    pub base_stats: KVCacheStats,
    /// Total tokens processed across all sequences
    pub total_tokens_processed: u64,
    /// Total number of eviction operations
    pub total_evictions: u64,
    /// Number of active streaming sequences
    pub active_sequences: usize,
    /// Number of sequences currently in prefill mode
    pub sequences_in_prefill: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_allocator() {
        let config = KVCacheConfig {
            max_blocks: 10,
            ..Default::default()
        };
        let mut allocator = BlockAllocator::new(config);

        assert_eq!(allocator.num_free(), 10);
        assert_eq!(allocator.num_allocated(), 0);

        // Allocate 3 blocks
        let blocks = allocator.allocate(3).unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(allocator.num_free(), 7);
        assert_eq!(allocator.num_allocated(), 3);

        // Free 1 block
        allocator.free(blocks[0]);
        assert_eq!(allocator.num_free(), 8);
        assert_eq!(allocator.num_allocated(), 2);
    }

    #[test]
    fn test_kv_cache_manager() {
        let config = KVCacheConfig {
            max_blocks: 100,
            block_size: 16,
            ..Default::default()
        };
        let mut manager = KVCacheManager::new(config);

        // Allocate for request 1
        let blocks1 = manager.allocate(&"req1".to_string(), 5);
        assert_eq!(blocks1.len(), 5);

        // Allocate for request 2
        let blocks2 = manager.allocate(&"req2".to_string(), 3);
        assert_eq!(blocks2.len(), 3);

        let stats = manager.stats();
        assert_eq!(stats.allocated_blocks, 8);
        assert_eq!(stats.num_sequences, 2);

        // Free request 1
        manager.free(&"req1".to_string());
        let stats = manager.stats();
        assert_eq!(stats.allocated_blocks, 3);
        assert_eq!(stats.num_sequences, 1);
    }

    #[test]
    fn test_shared_prefix_reuses_blocks() {
        let config = KVCacheConfig {
            max_blocks: 32,
            block_size: 16,
            ..Default::default()
        };
        let mut manager = KVCacheManager::new(config);
        let prefix_hash = Some(0xDEAD_BEEF_u64);

        let blocks1 = manager.allocate_with_prefix(&"req1".to_string(), 4, prefix_hash);
        assert_eq!(blocks1.len(), 4);
        let stats_after_req1 = manager.stats();
        assert_eq!(stats_after_req1.allocated_blocks, 4);

        let blocks2 = manager.allocate_with_prefix(&"req2".to_string(), 4, prefix_hash);
        assert_eq!(blocks2, blocks1);
        let stats_after_req2 = manager.stats();
        assert_eq!(stats_after_req2.allocated_blocks, 4);
        assert!(stats_after_req2.telemetry.shared_prefix_hits >= 1);

        manager.free(&"req1".to_string());
        assert_eq!(manager.stats().allocated_blocks, 4);
        manager.free(&"req2".to_string());
        assert_eq!(manager.stats().allocated_blocks, 0);
    }

    #[test]
    fn test_copy_on_write_split_for_shared_block() {
        let config = KVCacheConfig {
            max_blocks: 32,
            block_size: 16,
            ..Default::default()
        };
        let mut manager = KVCacheManager::new(config);
        let prefix_hash = Some(0xABCD_u64);

        let req1 = "req-cow-1".to_string();
        let req2 = "req-cow-2".to_string();
        let blocks1 = manager.allocate_with_prefix(&req1, 2, prefix_hash);
        let blocks2 = manager.allocate_with_prefix(&req2, 2, prefix_hash);
        assert_eq!(blocks1, blocks2);

        let old_last = blocks2[1];
        let new_last = manager.ensure_writable_last_block(&req2).unwrap();
        assert_ne!(old_last, new_last);

        let req1_last = manager.get_block_table(&req1).unwrap()[1];
        let req2_last = manager.get_block_table(&req2).unwrap()[1];
        assert_ne!(req1_last, req2_last);
        assert_eq!(manager.stats().telemetry.copy_on_write_splits, 1);
    }

    #[test]
    fn test_advanced_prefix_cache_reuses_partial_blocks() {
        let config = KVCacheConfig {
            max_blocks: 32,
            block_size: 4,
            ..Default::default()
        };
        let mut manager = KVCacheManager::new(config);

        let prompt_a = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let prompt_b = vec![1, 2, 3, 4, 5, 6, 7, 8, 44, 45, 46, 47];

        let blocks_a = manager.allocate_with_prefix_tokens(&"req-a".to_string(), 3, &prompt_a);
        assert_eq!(blocks_a.len(), 3);

        let blocks_b = manager.allocate_with_prefix_tokens(&"req-b".to_string(), 3, &prompt_b);
        assert_eq!(blocks_b.len(), 3);
        assert_eq!(blocks_b[..2], blocks_a[..2]);
        assert_ne!(blocks_b[2], blocks_a[2]);

        let stats = manager.stats();
        assert_eq!(stats.allocated_blocks, 4);
        assert!(stats.telemetry.shared_prefix_hits >= 1);
    }

    #[test]
    fn test_advanced_prefix_cleanup_preserves_live_entries() {
        let config = KVCacheConfig {
            max_blocks: 32,
            block_size: 4,
            ..Default::default()
        };
        let mut manager = KVCacheManager::new(config);

        let prompt = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let blocks1 = manager.allocate_with_prefix_tokens(&"req-1".to_string(), 2, &prompt);
        let blocks2 = manager.allocate_with_prefix_tokens(&"req-2".to_string(), 2, &prompt);
        assert_eq!(blocks1, blocks2);

        manager.free(&"req-1".to_string());

        // req-2 still references the shared prefix. A new request should reuse it.
        let blocks3 = manager.allocate_with_prefix_tokens(&"req-3".to_string(), 2, &prompt);
        assert_eq!(blocks3, blocks2);
    }

    #[test]
    fn test_persistent_prefix_snapshot_reuses_after_request_finishes() {
        let config = KVCacheConfig {
            max_blocks: 2,
            block_size: 2,
            ..Default::default()
        };
        let mut manager = KVCacheManager::new(config);

        let prompt = vec![21, 22, 23, 24];
        let req1 = "persist-1".to_string();
        let blocks1 = manager.allocate_with_prefix_tokens(&req1, 2, &prompt);
        assert_eq!(blocks1.len(), 2);

        // Free the original request; persistent snapshot retains reusable block refs.
        manager.free(&req1);
        assert_eq!(manager.stats().free_blocks, 0);

        // Even with no free blocks, identical prefix can still be admitted via persistent reuse.
        let req2 = "persist-2".to_string();
        let blocks2 = manager.allocate_with_prefix_tokens(&req2, 2, &prompt);
        assert_eq!(blocks2.len(), 2);
        assert_eq!(blocks2, blocks1);
        assert!(manager.stats().telemetry.shared_prefix_hits >= 1);
    }
}
