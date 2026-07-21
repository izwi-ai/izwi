//! Committed full-page prefix index for managed physical KV arenas.
//!
//! The index owns no page storage and performs no refcount mutation. Publishing
//! a page is paired with the coordinator's prefix retain during cache commit;
//! eviction returns the exact block whose prefix retain the caller must release.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use super::telemetry::ManagedKvTelemetry;
use crate::engine::cache::coordinator::{KvCacheCoordinator, KvCoordinatorError, KvSnapshot};
use crate::engine::{ModelInstanceId, PlanId};
use crate::kv::{CacheBlockRef, KvPlanFingerprint};

const PREFIX_NAMESPACE_DOMAIN: &[u8] = b"izwi.kv.prefix-namespace.v1\0";
const PREFIX_PAGE_DOMAIN: &[u8] = b"izwi.kv.prefix-page.v1\0";

/// Everything that can change the numerical meaning of projected K/V.
///
/// A model reload may share pages only when the loaded adapter explicitly
/// supplies the same compatibility digest. `model_instance` still prevents
/// accidental cross-arena sharing by default.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvPrefixNamespace {
    pub model_instance: ModelInstanceId,
    pub model_revision: [u8; 32],
    pub adapter_abi: [u8; 32],
    pub tokenizer_or_input_encoding: [u8; 32],
    pub position_semantics: [u8; 32],
    pub plan: KvPlanFingerprint,
    pub multimodal_artifact: Option<[u8; 32]>,
    pub cache_salt: [u8; 32],
}

impl KvPrefixNamespace {
    pub fn fingerprint(&self) -> Result<[u8; 32], KvPrefixIndexError> {
        let encoded = serde_json::to_vec(self).map_err(|error| {
            KvPrefixIndexError::Encoding(format!("failed to encode prefix namespace: {error}"))
        })?;
        let mut hasher = Sha256::new();
        hasher.update(PREFIX_NAMESPACE_DOMAIN);
        hasher.update(encoded);
        Ok(hasher.finalize().into())
    }
}

/// Exact full-page identity. Chaining the previous digest makes a matching
/// token page at a different prefix location a different entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvPrefixPageKey {
    pub namespace: [u8; 32],
    pub previous_page: Option<[u8; 32]>,
    pub start_position: u64,
    pub tokens: Vec<u32>,
    digest: [u8; 32],
}

impl KvPrefixPageKey {
    pub fn new(
        namespace: &KvPrefixNamespace,
        previous_page: Option<[u8; 32]>,
        start_position: u64,
        tokens: Vec<u32>,
    ) -> Result<Self, KvPrefixIndexError> {
        if tokens.is_empty() {
            return Err(KvPrefixIndexError::PartialPage);
        }
        let namespace = namespace.fingerprint()?;
        let mut hasher = Sha256::new();
        hasher.update(PREFIX_PAGE_DOMAIN);
        hasher.update(namespace);
        match previous_page {
            Some(previous) => {
                hasher.update([1]);
                hasher.update(previous);
            }
            None => hasher.update([0]),
        }
        hasher.update(start_position.to_le_bytes());
        hasher.update((tokens.len() as u64).to_le_bytes());
        for token in &tokens {
            hasher.update(token.to_le_bytes());
        }
        let digest = hasher.finalize().into();
        Ok(Self {
            namespace,
            previous_page,
            start_position,
            tokens,
            digest,
        })
    }

    pub const fn digest(&self) -> [u8; 32] {
        self.digest
    }

    pub fn is_complete_page(&self, page_tokens: u32) -> bool {
        page_tokens != 0 && self.tokens.len() == page_tokens as usize
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvPrefixHit {
    pub block: CacheBlockRef,
    pub digest: [u8; 32],
    pub token_count: u32,
}

#[derive(Debug, Clone)]
struct PrefixEntry {
    key: KvPrefixPageKey,
    block: CacheBlockRef,
    last_access: u64,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum KvPrefixIndexError {
    #[error("prefix pages must contain one complete non-empty page")]
    PartialPage,
    #[error("prefix digest is already bound to different tokens, position, namespace, or page")]
    DigestConflict,
    #[error("one physical page cannot be published under multiple prefix identities")]
    BlockConflict,
    #[error("prefix index encoding failed: {0}")]
    Encoding(String),
    #[error("prefix index access counter overflow")]
    CounterOverflow,
    #[error(transparent)]
    Coordinator(#[from] KvCoordinatorError),
}

/// Bounded exact-match index. It deliberately starts with a page hash rather
/// than a radix tree; chained lookups recover the longest committed prefix.
#[derive(Debug, Clone)]
pub struct CommittedPrefixIndex {
    capacity_pages: usize,
    clock: u64,
    entries: HashMap<[u8; 32], PrefixEntry>,
}

impl CommittedPrefixIndex {
    pub fn new(capacity_pages: usize) -> Self {
        Self {
            capacity_pages,
            clock: 0,
            entries: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn capacity_pages(&self) -> usize {
        self.capacity_pages
    }

    /// Insert a page only after its cache transaction has committed and the
    /// coordinator has retained a prefix reference.
    pub fn publish(
        &mut self,
        key: KvPrefixPageKey,
        page_tokens: u32,
        block: CacheBlockRef,
    ) -> Result<Vec<CacheBlockRef>, KvPrefixIndexError> {
        if !key.is_complete_page(page_tokens) {
            return Err(KvPrefixIndexError::PartialPage);
        }
        let digest = key.digest();
        if let Some(existing) = self.entries.get(&digest) {
            if existing.key != key || existing.block != block {
                return Err(KvPrefixIndexError::DigestConflict);
            }
            return Ok(Vec::new());
        }
        if self.entries.values().any(|entry| entry.block == block) {
            return Err(KvPrefixIndexError::BlockConflict);
        }

        let last_access = self.tick()?;
        self.entries.insert(
            digest,
            PrefixEntry {
                key,
                block,
                last_access,
            },
        );

        let mut evicted = Vec::new();
        while self.entries.len() > self.capacity_pages {
            let removed = self.evict_lru();
            if removed.is_empty() {
                break;
            }
            evicted.extend(removed);
        }
        Ok(evicted)
    }

    /// Exact verification is intentional even though SHA-256 collisions are
    /// impractical: lookup correctness must not depend on digest uniqueness.
    pub fn lookup(
        &mut self,
        key: &KvPrefixPageKey,
    ) -> Result<Option<KvPrefixHit>, KvPrefixIndexError> {
        let digest = key.digest();
        let access = self.tick()?;
        let Some(entry) = self.entries.get_mut(&digest) else {
            return Ok(None);
        };
        if entry.key != *key {
            return Ok(None);
        }
        entry.last_access = access;
        Ok(Some(KvPrefixHit {
            block: entry.block,
            digest,
            token_count: entry.key.tokens.len() as u32,
        }))
    }

    /// Remove the least recently used index entry. The returned physical page
    /// may remain alive through table or execution refs; the coordinator owns
    /// that lifetime decision.
    pub fn evict_lru(&mut self) -> Vec<CacheBlockRef> {
        let Some(digest) = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_access)
            .map(|(digest, _)| *digest)
        else {
            return Vec::new();
        };
        self.remove_chain_from(digest)
    }

    pub fn remove(&mut self, digest: [u8; 32]) -> Option<CacheBlockRef> {
        self.entries.remove(&digest).map(|entry| entry.block)
    }

    fn tick(&mut self) -> Result<u64, KvPrefixIndexError> {
        self.clock = self
            .clock
            .checked_add(1)
            .ok_or(KvPrefixIndexError::CounterOverflow)?;
        Ok(self.clock)
    }

    fn remove_chain_from(&mut self, root: [u8; 32]) -> Vec<CacheBlockRef> {
        let mut pending = vec![root];
        let mut removed = Vec::new();
        while let Some(digest) = pending.pop() {
            pending.extend(
                self.entries
                    .iter()
                    .filter_map(|(child_digest, entry)| {
                        (entry.key.previous_page == Some(digest)).then_some(*child_digest)
                    })
                    .collect::<Vec<_>>(),
            );
            if let Some(entry) = self.entries.remove(&digest) {
                removed.push(entry.block);
            }
        }
        removed
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvPrefixPublication {
    pub key: KvPrefixPageKey,
    pub block: CacheBlockRef,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct KvPrefixMatch {
    pub blocks: Vec<CacheBlockRef>,
    pub page_digests: Vec<[u8; 32]>,
    pub reused_tokens: u32,
}

/// Couples index visibility to coordinator ownership. Index changes are built
/// on a private clone, the coordinator atomically commits table/ref mutations,
/// and only then is the staged index made visible to admission lookups.
#[derive(Debug, Clone)]
pub struct CoordinatedPrefixIndex {
    index: CommittedPrefixIndex,
    telemetry: Arc<ManagedKvTelemetry>,
}

impl CoordinatedPrefixIndex {
    pub fn new(capacity_pages: usize) -> Self {
        Self::with_telemetry(capacity_pages, Arc::new(ManagedKvTelemetry::default()))
    }

    pub fn with_telemetry(capacity_pages: usize, telemetry: Arc<ManagedKvTelemetry>) -> Self {
        Self {
            index: CommittedPrefixIndex::new(capacity_pages),
            telemetry,
        }
    }

    pub fn telemetry(&self) -> &Arc<ManagedKvTelemetry> {
        &self.telemetry
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Commit a written cache transaction and publish complete pages in one
    /// control-plane operation. Any validation or CAS failure leaves the live
    /// prefix index unchanged.
    pub fn commit_transaction(
        &mut self,
        coordinator: &mut KvCacheCoordinator,
        txn_id: PlanId,
        page_tokens: u32,
        publications: &[KvPrefixPublication],
    ) -> Result<KvSnapshot, KvPrefixIndexError> {
        let before = self
            .index
            .entries
            .values()
            .map(|entry| entry.block)
            .collect::<std::collections::HashSet<_>>();
        let mut staged = self.index.clone();
        for publication in publications {
            staged.publish(publication.key.clone(), page_tokens, publication.block)?;
        }
        let after = staged
            .entries
            .values()
            .map(|entry| entry.block)
            .collect::<std::collections::HashSet<_>>();
        let retained = after.difference(&before).copied().collect::<Vec<_>>();
        let released = before.difference(&after).copied().collect::<Vec<_>>();
        let snapshot = coordinator.commit_with_prefix_updates(txn_id, &retained, &released)?;
        self.index = staged;
        for _ in &released {
            self.telemetry.record_prefix_eviction();
        }
        Ok(snapshot)
    }

    /// Return the longest exact chain of committed, complete pages. Partial
    /// prompt tails are deliberately excluded and remain request-private.
    pub fn lookup_longest(
        &mut self,
        namespace: &KvPrefixNamespace,
        tokens: &[u32],
        page_tokens: u32,
    ) -> Result<KvPrefixMatch, KvPrefixIndexError> {
        if page_tokens == 0 {
            return Err(KvPrefixIndexError::PartialPage);
        }
        let page_tokens = page_tokens as usize;
        let mut matched = KvPrefixMatch::default();
        let mut previous = None;
        for (page_index, page) in tokens.chunks_exact(page_tokens).enumerate() {
            let start_position = u64::try_from(page_index)
                .ok()
                .and_then(|page| page.checked_mul(page_tokens as u64))
                .ok_or(KvPrefixIndexError::CounterOverflow)?;
            let key = KvPrefixPageKey::new(namespace, previous, start_position, page.to_vec())?;
            let Some(hit) = self.index.lookup(&key)? else {
                break;
            };
            matched.blocks.push(hit.block);
            matched.page_digests.push(hit.digest);
            matched.reused_tokens = matched
                .reused_tokens
                .checked_add(hit.token_count)
                .ok_or(KvPrefixIndexError::CounterOverflow)?;
            previous = Some(hit.digest);
        }
        if matched.reused_tokens == 0 {
            self.telemetry.record_prefix_miss();
        } else {
            self.telemetry
                .record_prefix_hit(u64::from(matched.reused_tokens));
        }
        Ok(matched)
    }

    /// Evict one LRU entry and release exactly its durable prefix reference.
    /// A failure leaves the live index untouched.
    pub fn evict_lru(
        &mut self,
        coordinator: &mut KvCacheCoordinator,
    ) -> Result<Vec<CacheBlockRef>, KvPrefixIndexError> {
        let mut staged = self.index.clone();
        let blocks = staged.evict_lru();
        if blocks.is_empty() {
            return Ok(Vec::new());
        }
        coordinator.release_prefixes(&blocks)?;
        self.index = staged;
        for _ in &blocks {
            self.telemetry.record_prefix_eviction();
        }
        Ok(blocks)
    }

    /// Evict the least-recently-used chain that does not contain any block in
    /// `protected`. Admission uses this after a prefix hit so reclaiming old
    /// prefixes cannot invalidate the exact shared pages it is about to
    /// reserve.
    pub fn evict_lru_excluding(
        &mut self,
        coordinator: &mut KvCacheCoordinator,
        protected: &std::collections::HashSet<CacheBlockRef>,
    ) -> Result<Vec<CacheBlockRef>, KvPrefixIndexError> {
        let mut candidates = self
            .index
            .entries
            .iter()
            .map(|(digest, entry)| (*digest, entry.last_access))
            .collect::<Vec<_>>();
        candidates.sort_by_key(|(_, last_access)| *last_access);
        for (digest, _) in candidates {
            let mut staged = self.index.clone();
            let blocks = staged.remove_chain_from(digest);
            if blocks.is_empty() || blocks.iter().any(|block| protected.contains(block)) {
                continue;
            }
            coordinator.release_prefixes(&blocks)?;
            self.index = staged;
            for _ in &blocks {
                self.telemetry.record_prefix_eviction();
            }
            return Ok(blocks);
        }
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use crate::backends::BackendKind;
    use crate::engine::cache::coordinator::{
        KvBlockIntent, KvGroupReservation, KvReserveRequest, KvWriteReceipt,
    };
    use crate::engine::{ModelInstanceId, SessionKey};
    use crate::kv::{CacheDomainId, KvArenaId, KvGroupId, KvPlanFingerprint};

    use super::*;

    fn namespace(salt: u8) -> KvPrefixNamespace {
        KvPrefixNamespace {
            model_instance: ModelInstanceId::new(7),
            model_revision: [1; 32],
            adapter_abi: [2; 32],
            tokenizer_or_input_encoding: [3; 32],
            position_semantics: [4; 32],
            plan: KvPlanFingerprint::new([5; 32]),
            multimodal_artifact: None,
            cache_salt: [salt; 32],
        }
    }

    fn block(index: u32) -> CacheBlockRef {
        CacheBlockRef {
            arena: KvArenaId {
                model_instance: ModelInstanceId::new(7),
                backend: BackendKind::Cpu,
                device_ordinal: None,
                generation: 1,
            },
            group: KvGroupId::new(0),
            index,
            slot_generation: 1,
        }
    }

    fn written_fresh_transaction(
        coordinator: &mut KvCacheCoordinator,
        txn_id: PlanId,
        snapshot: KvSnapshot,
        tokens: u32,
    ) -> CacheBlockRef {
        coordinator
            .reserve(KvReserveRequest {
                txn_id,
                expected: snapshot,
                target_committed_tokens: tokens,
                target_window_start: 0,
                groups: vec![KvGroupReservation {
                    group: KvGroupId::new(0),
                    blocks: vec![KvBlockIntent::Fresh],
                }],
            })
            .unwrap();
        let prepared = coordinator.prepare(txn_id).unwrap();
        coordinator
            .complete_write(KvWriteReceipt {
                txn_id,
                committed_tokens: tokens,
                written_blocks: prepared.writable_blocks.clone(),
            })
            .unwrap();
        prepared.writable_blocks[0]
    }

    #[test]
    fn namespace_and_chain_are_part_of_page_identity() {
        let first = KvPrefixPageKey::new(&namespace(9), None, 0, vec![1, 2]).unwrap();
        let chained =
            KvPrefixPageKey::new(&namespace(9), Some(first.digest()), 2, vec![3, 4]).unwrap();
        let unchained = KvPrefixPageKey::new(&namespace(9), None, 2, vec![3, 4]).unwrap();
        let other_namespace = KvPrefixPageKey::new(&namespace(8), None, 0, vec![1, 2]).unwrap();
        assert_ne!(chained.digest(), unchained.digest());
        assert_ne!(first.digest(), other_namespace.digest());
    }

    #[test]
    fn partial_pages_are_never_published() {
        let key = KvPrefixPageKey::new(&namespace(1), None, 0, vec![1]).unwrap();
        let error = CommittedPrefixIndex::new(1)
            .publish(key, 2, block(0))
            .unwrap_err();
        assert_eq!(error, KvPrefixIndexError::PartialPage);
    }

    #[test]
    fn lookup_is_exact_and_lru_eviction_returns_the_retained_block() {
        let mut index = CommittedPrefixIndex::new(2);
        let first = KvPrefixPageKey::new(&namespace(1), None, 0, vec![1, 2]).unwrap();
        let second =
            KvPrefixPageKey::new(&namespace(1), Some(first.digest()), 2, vec![3, 4]).unwrap();
        let third =
            KvPrefixPageKey::new(&namespace(1), Some(second.digest()), 4, vec![5, 6]).unwrap();

        assert!(index
            .publish(first.clone(), 2, block(0))
            .unwrap()
            .is_empty());
        assert!(index
            .publish(second.clone(), 2, block(1))
            .unwrap()
            .is_empty());
        assert_eq!(index.lookup(&first).unwrap().unwrap().block, block(0));
        assert_eq!(
            index.publish(third, 2, block(2)).unwrap(),
            vec![block(1), block(2)]
        );
        assert!(index.lookup(&second).unwrap().is_none());
        assert_eq!(index.lookup(&first).unwrap().unwrap().token_count, 2);
    }

    #[test]
    fn duplicate_publication_is_idempotent_but_conflicting_binding_fails() {
        let mut index = CommittedPrefixIndex::new(2);
        let key = KvPrefixPageKey::new(&namespace(1), None, 0, vec![1, 2]).unwrap();
        index.publish(key.clone(), 2, block(0)).unwrap();
        assert!(index.publish(key.clone(), 2, block(0)).unwrap().is_empty());
        assert_eq!(
            index.publish(key, 2, block(1)).unwrap_err(),
            KvPrefixIndexError::DigestConflict
        );
    }

    #[test]
    fn publication_is_invisible_until_the_written_transaction_commits() {
        let arena = block(0).arena;
        let session = SessionKey::new("prefix-atomic".into(), 1);
        let mut coordinator = KvCacheCoordinator::new(arena, 2);
        let initial = coordinator
            .register_table(session.clone(), CacheDomainId::new(0))
            .unwrap();
        let published = written_fresh_transaction(&mut coordinator, 11, initial, 2);
        let key = KvPrefixPageKey::new(&namespace(1), None, 0, vec![1, 2]).unwrap();
        let mut index = CoordinatedPrefixIndex::new(2);

        assert!(index
            .lookup_longest(&namespace(1), &[1, 2], 2)
            .unwrap()
            .blocks
            .is_empty());
        index
            .commit_transaction(
                &mut coordinator,
                11,
                2,
                &[KvPrefixPublication {
                    key,
                    block: published,
                }],
            )
            .unwrap();
        assert_eq!(
            index
                .lookup_longest(&namespace(1), &[1, 2, 9], 2)
                .unwrap()
                .blocks,
            vec![published]
        );
        assert_eq!(coordinator.stats().prefix_refs, 1);

        assert_eq!(index.evict_lru(&mut coordinator).unwrap(), vec![published]);
        assert_eq!(coordinator.stats().prefix_refs, 0);
        assert_eq!(coordinator.stats().table_refs, 1);
        coordinator
            .release_table(&session, CacheDomainId::new(0))
            .unwrap();
        assert_eq!(coordinator.stats().allocated_pages, 0);
        coordinator.check_invariants().unwrap();
    }

    #[test]
    fn failed_or_aborted_work_never_reaches_prefix_lookup() {
        let arena = block(0).arena;
        let session = SessionKey::new("prefix-abort".into(), 1);
        let mut coordinator = KvCacheCoordinator::new(arena, 2);
        let initial = coordinator
            .register_table(session, CacheDomainId::new(0))
            .unwrap();
        let candidate = written_fresh_transaction(&mut coordinator, 12, initial, 1);
        let key = KvPrefixPageKey::new(&namespace(1), None, 0, vec![7]).unwrap();
        let mut index = CoordinatedPrefixIndex::new(2);
        assert_eq!(
            index
                .commit_transaction(
                    &mut coordinator,
                    12,
                    2,
                    &[KvPrefixPublication {
                        key,
                        block: candidate,
                    }],
                )
                .unwrap_err(),
            KvPrefixIndexError::PartialPage
        );
        assert!(index.is_empty());
        assert!(coordinator.abort(12).unwrap());
        assert!(index
            .lookup_longest(&namespace(1), &[7, 8], 2)
            .unwrap()
            .blocks
            .is_empty());
        assert_eq!(coordinator.stats().prefix_refs, 0);
        coordinator.check_invariants().unwrap();
    }

    #[test]
    fn evicting_a_chain_ancestor_cannot_leave_unreachable_descendants() {
        let mut index = CommittedPrefixIndex::new(3);
        let first = KvPrefixPageKey::new(&namespace(1), None, 0, vec![1, 2]).unwrap();
        let second =
            KvPrefixPageKey::new(&namespace(1), Some(first.digest()), 2, vec![3, 4]).unwrap();
        index.publish(first, 2, block(0)).unwrap();
        index.publish(second, 2, block(1)).unwrap();
        assert_eq!(index.evict_lru(), vec![block(0), block(1)]);
        assert!(index.is_empty());
    }
}
