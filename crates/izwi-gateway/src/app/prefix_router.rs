//! Approximate prefix-aware routing for remote chat.
//!
//! The gateway cannot see a worker's KV cache, so it remembers what it sent
//! where. A prompt is cut into fixed-size blocks of its message stream and
//! each block gets a chained hash (block `i` hashes block `i - 1`'s hash with
//! its own bytes), so one hash names a whole prefix. Each worker keeps a
//! bounded LRU of the block hashes it was recently routed; the longest prefix
//! a worker holds is the longest chained hash it remembers. This is the
//! approximate radix tree of SGLang's cache-aware router, flattened into hash
//! sets. Only hashes are kept: raw prompt text never leaves the request.
//!
//! The registry decides whether a match is worth its load (see
//! [`crate::worker_registry::PrefixAffinity`]); this module only answers
//! "how many leading blocks does this worker probably hold".

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

use izwi_model_catalog::ChatMessage;

use crate::error::ApiError;
use crate::worker_registry::{PrefixAffinity, WorkerInstanceKey};

/// Upper bounds for operator configuration.
pub const MAX_PREFIX_BLOCK_BYTES: usize = 64 * 1024;
pub const MAX_PREFIX_BLOCKS: usize = 4_096;
pub const MAX_PREFIX_ENTRIES_PER_WORKER: usize = 1_048_576;
/// Worker incarnations whose history is kept. Replaced incarnations age out
/// once this many newer ones were routed to.
const MAX_TRACKED_WORKERS: usize = 1_024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixAffinityConfig {
    /// Bytes of message stream per hashed block (about four bytes a token).
    pub block_bytes: usize,
    /// Leading blocks considered per prompt; longer prompts share routing
    /// with any prompt that has the same first `block_bytes * max_blocks`
    /// bytes.
    pub max_blocks: usize,
    /// Block hashes remembered per worker before the least recently routed
    /// ones are forgotten.
    pub max_entries_per_worker: usize,
}

impl Default for PrefixAffinityConfig {
    fn default() -> Self {
        Self {
            block_bytes: 256,
            max_blocks: 128,
            max_entries_per_worker: 16_384,
        }
    }
}

impl PrefixAffinityConfig {
    pub fn validate(&self) -> Result<(), ApiError> {
        if self.block_bytes == 0
            || self.block_bytes > MAX_PREFIX_BLOCK_BYTES
            || self.max_blocks == 0
            || self.max_blocks > MAX_PREFIX_BLOCKS
            || self.max_entries_per_worker == 0
            || self.max_entries_per_worker > MAX_PREFIX_ENTRIES_PER_WORKER
        {
            return Err(ApiError::internal(
                "Prefix affinity block size, depth or per-worker entries are outside the supported range",
            ));
        }
        Ok(())
    }
}

/// The chained block hashes of one prompt, shortest prefix first.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PromptBlocks(Vec<u64>);

#[cfg(test)]
impl PromptBlocks {
    fn len(&self) -> usize {
        self.0.len()
    }
}

/// Bounded per-worker memory of routed prompt prefixes.
#[derive(Debug)]
pub struct PrefixRouter {
    config: PrefixAffinityConfig,
    workers: Mutex<WorkerHistories>,
}

#[derive(Debug, Default)]
struct WorkerHistories {
    by_worker: BTreeMap<WorkerInstanceKey, (u64, BlockLru)>,
    next_use: u64,
}

impl PrefixRouter {
    pub fn new(config: PrefixAffinityConfig) -> Self {
        Self {
            config,
            workers: Mutex::new(WorkerHistories::default()),
        }
    }

    /// Hashes the role-tagged message stream in full blocks. A trailing
    /// partial block is not hashed: it cannot be shared by a longer prompt.
    pub fn blocks(&self, messages: &[ChatMessage]) -> PromptBlocks {
        let block_bytes = self.config.block_bytes;
        let mut blocks = Vec::new();
        let mut chain = 0_u64;
        let mut block = Vec::with_capacity(block_bytes);
        let stream = messages.iter().flat_map(|message| {
            message
                .role
                .as_prompt_role()
                .as_bytes()
                .iter()
                .chain([0x1f_u8].iter())
                .chain(message.content.as_bytes())
                .chain([0x1e_u8].iter())
        });
        for byte in stream {
            block.push(*byte);
            if block.len() == block_bytes {
                let mut hasher = DefaultHasher::new();
                chain.hash(&mut hasher);
                block.hash(&mut hasher);
                chain = hasher.finish();
                blocks.push(chain);
                block.clear();
                if blocks.len() == self.config.max_blocks {
                    break;
                }
            }
        }
        PromptBlocks(blocks)
    }

    /// Remembers that `worker` now holds every prefix of `blocks`.
    pub fn record(&self, worker: &WorkerInstanceKey, blocks: &PromptBlocks) {
        if blocks.0.is_empty() {
            return;
        }
        let mut workers = self.lock();
        workers.next_use += 1;
        let used = workers.next_use;
        if !workers.by_worker.contains_key(worker) && workers.by_worker.len() >= MAX_TRACKED_WORKERS
        {
            let oldest = workers
                .by_worker
                .iter()
                .min_by_key(|(_, (last_used, _))| *last_used)
                .map(|(key, _)| key.clone());
            if let Some(oldest) = oldest {
                workers.by_worker.remove(&oldest);
            }
        }
        let max_entries = self.config.max_entries_per_worker;
        let (last_used, history) = workers
            .by_worker
            .entry(worker.clone())
            .or_insert_with(|| (used, BlockLru::new(max_entries)));
        *last_used = used;
        // Longest prefix first, so shorter prefixes are the most recent and
        // are forgotten last: a remembered block implies its predecessors.
        for hash in blocks.0.iter().rev() {
            history.touch(*hash);
        }
    }

    /// Leading blocks of `blocks` that `worker` was recently routed.
    pub fn matched_blocks(&self, worker: &WorkerInstanceKey, blocks: &PromptBlocks) -> usize {
        let workers = self.lock();
        let Some((_, history)) = workers.by_worker.get(worker) else {
            return 0;
        };
        blocks
            .0
            .iter()
            .rposition(|hash| history.contains(*hash))
            .map_or(0, |index| index + 1)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, WorkerHistories> {
        self.workers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// One request's prompt blocks, answering the registry's affinity question.
pub struct RequestPrefix<'a> {
    pub router: &'a PrefixRouter,
    pub blocks: &'a PromptBlocks,
}

impl PrefixAffinity for RequestPrefix<'_> {
    fn matched_blocks(&self, worker: &WorkerInstanceKey) -> usize {
        self.router.matched_blocks(worker, self.blocks)
    }
}

/// Bounded LRU set with amortized O(1) touch: a recency queue with lazy
/// deletion, compacted once stale entries outnumber live ones.
#[derive(Debug)]
struct BlockLru {
    capacity: usize,
    stamps: HashMap<u64, u64>,
    order: VecDeque<(u64, u64)>,
    clock: u64,
}

impl BlockLru {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            stamps: HashMap::new(),
            order: VecDeque::new(),
            clock: 0,
        }
    }

    fn contains(&self, hash: u64) -> bool {
        self.stamps.contains_key(&hash)
    }

    fn touch(&mut self, hash: u64) {
        self.clock += 1;
        self.stamps.insert(hash, self.clock);
        self.order.push_back((hash, self.clock));
        while self.stamps.len() > self.capacity {
            let Some((oldest, stamp)) = self.order.pop_front() else {
                break;
            };
            if self.stamps.get(&oldest) == Some(&stamp) {
                self.stamps.remove(&oldest);
            }
        }
        if self.order.len() > self.capacity.saturating_mul(2).max(64) {
            let stamps = &self.stamps;
            self.order
                .retain(|(hash, stamp)| stamps.get(hash) == Some(stamp));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use izwi_model_catalog::ChatRole;
    use izwi_serving_protocol::{IncarnationId, WorkerId};

    fn worker(name: &str) -> WorkerInstanceKey {
        WorkerInstanceKey {
            worker_id: WorkerId::new(name).unwrap(),
            incarnation_id: IncarnationId::new("inc").unwrap(),
        }
    }

    fn message(role: ChatRole, content: &str) -> ChatMessage {
        ChatMessage {
            role,
            content: content.into(),
        }
    }

    fn router(block_bytes: usize, max_blocks: usize, entries: usize) -> PrefixRouter {
        PrefixRouter::new(PrefixAffinityConfig {
            block_bytes,
            max_blocks,
            max_entries_per_worker: entries,
        })
    }

    #[test]
    fn shared_leading_blocks_match_and_divergence_stops_the_match() {
        let router = router(16, 64, 1024);
        let system = message(ChatRole::System, &"s".repeat(64));
        let first = router.blocks(&[system.clone(), message(ChatRole::User, "alpha question")]);
        let second = router.blocks(&[system.clone(), message(ChatRole::User, "beta question")]);
        let other = router.blocks(&[message(ChatRole::System, &"t".repeat(64))]);
        assert!(first.len() >= 4);
        router.record(&worker("a"), &first);

        // The system prompt's whole blocks are shared; the user turn is not.
        let shared = router.matched_blocks(&worker("a"), &second);
        assert!(shared >= 4 && shared < second.len(), "{shared}");
        assert_eq!(router.matched_blocks(&worker("a"), &first), first.len());
        assert_eq!(router.matched_blocks(&worker("a"), &other), 0);
        assert_eq!(router.matched_blocks(&worker("b"), &first), 0);
    }

    #[test]
    fn roles_are_part_of_the_prefix() {
        let router = router(8, 64, 1024);
        let as_system = router.blocks(&[message(ChatRole::System, &"x".repeat(32))]);
        let as_user = router.blocks(&[message(ChatRole::User, &"x".repeat(32))]);
        router.record(&worker("a"), &as_system);
        assert_eq!(router.matched_blocks(&worker("a"), &as_user), 0);
    }

    #[test]
    fn prompts_hash_at_most_the_configured_depth() {
        let router = router(8, 3, 1024);
        let long = router.blocks(&[message(ChatRole::User, &"y".repeat(400))]);
        assert_eq!(long.len(), 3);
        let short = router.blocks(&[message(ChatRole::User, "a")]);
        assert_eq!(short.len(), 0, "a partial block is never hashed");
    }

    #[test]
    fn worker_history_is_a_bounded_lru_that_forgets_long_prefixes_first() {
        let router = router(8, 64, 4);
        let first = router.blocks(&[message(ChatRole::User, &"a".repeat(64))]);
        assert!(first.len() > 4);
        router.record(&worker("a"), &first);
        // Only the four shortest prefixes survive, and a held block still
        // implies all of its predecessors.
        assert_eq!(router.matched_blocks(&worker("a"), &first), 4);

        let second = router.blocks(&[message(ChatRole::User, &"b".repeat(64))]);
        router.record(&worker("a"), &second);
        assert_eq!(router.matched_blocks(&worker("a"), &first), 0);
        assert_eq!(router.matched_blocks(&worker("a"), &second), 4);
    }

    #[test]
    fn lru_touch_keeps_its_queue_bounded() {
        let mut lru = BlockLru::new(8);
        for round in 0..10_000_u64 {
            lru.touch(round % 5);
        }
        assert_eq!(lru.stamps.len(), 5);
        assert!(lru.order.len() <= 64 + 1, "{}", lru.order.len());
        for hash in 0..5 {
            assert!(lru.contains(hash));
        }
    }
}
