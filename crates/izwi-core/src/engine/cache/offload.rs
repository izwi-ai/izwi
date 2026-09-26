//! DS4 hierarchical KV offload: bounded host-tier storage for committed
//! prefix pages.
//!
//! Demotion selects the same victim an eviction would (the LRU index
//! subtree), requires every page to hold no ownership beyond its durable
//! prefix reference, and moves the subtree's bytes into a byte-budgeted host
//! pool through a two-phase pinned transfer (`KvResidencyState`): pin, mark
//! `Offloading`, capture, acknowledge, then swap index ownership. Steps run
//! synchronously at manager safe points and are bounded per tick; the
//! transfer records keep the structure an async copy worker needs if a later
//! backend makes deferred application worthwhile.
//!
//! On CUDA the host pool is real additional capacity across PCIe. On Metal
//! and CPU the pool is charged to the shared host/unified ledger (DINV-05):
//! offload there is working-set trimming — retention of cold prefixes plus
//! admission headroom — never extra capacity.

use std::collections::HashMap;
use std::time::Instant;

use super::coordinator::KvCacheCoordinator;
use super::prefix::{
    CoordinatedPrefixIndex, KvHostTailMatch, KvPrefixIndexError, KvPrefixMatch, KvPrefixNamespace,
    KvPrefixPageKey, LruSubtreeView,
};
use super::telemetry::ManagedKvTelemetry;
use crate::backends::kv::{KvArena, KvHostPool};
use crate::engine::EngineCoreConfig;
use crate::engine::cache::managed::coordinator_error;
use crate::error::{Error, Result};
use crate::kv::{CacheBlockRef, KvResidencyState, KvStorageTier, KvTransferId};

/// Resolved DS4 engagement policy. A zero budget keeps offload fully dormant;
/// explicit enablement with an invalid watermark pair fails closed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct HostOffloadPolicy {
    pub budget_bytes: u64,
    pub high_watermark: f32,
    pub low_watermark: f32,
    pub max_in_flight_pages: usize,
    pub max_promotion_pages: usize,
}

impl HostOffloadPolicy {
    /// Resolves the policy from the engine core configuration. `None` means
    /// offload stays disabled for this engine (no host pool budget).
    pub fn resolve(config: &EngineCoreConfig) -> Result<Option<Self>> {
        if config.kv_host_pool_budget_bytes == 0 {
            return Ok(None);
        }
        let policy = Self {
            budget_bytes: config.kv_host_pool_budget_bytes,
            high_watermark: config.kv_offload_high_watermark,
            low_watermark: config.kv_offload_low_watermark,
            max_in_flight_pages: config.kv_offload_max_in_flight_pages.max(1),
            max_promotion_pages: config.kv_offload_max_promotion_pages,
        };
        if !(policy.low_watermark > 0.0
            && policy.high_watermark > policy.low_watermark
            && policy.high_watermark <= 1.0)
        {
            return Err(Error::InferenceError(
                "KV offload watermarks must satisfy 0 < low < high <= 1".to_string(),
            ));
        }
        Ok(Some(policy))
    }
}

/// Host-resident entry for one page: its exact prefix identity and where its
/// bytes live in the [`KvHostPool`].
#[derive(Debug, Clone)]
struct HostChainEntry {
    key: super::prefix::KvPrefixPageKey,
    slot: usize,
    last_access: u64,
}

/// Digest-chained index of host-resident prefix pages. Keyed exactly like the
/// device index (chained page digests under one namespace fingerprint) so a
/// lookup can walk device pages and then continue into host pages seamlessly.
#[derive(Debug, Default)]
pub(crate) struct HostChainIndex {
    entries: HashMap<[u8; 32], HostChainEntry>,
    clock: u64,
}

impl HostChainIndex {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Exact, key-verified lookup of one page (mirrors the device index rule
    /// that digest uniqueness is never trusted for correctness).
    pub fn lookup(&mut self, key: &super::prefix::KvPrefixPageKey) -> Option<usize> {
        let digest = key.digest();
        let entry = self.entries.get_mut(&digest)?;
        if entry.key != *key {
            return None;
        }
        self.clock = self.clock.wrapping_add(1);
        entry.last_access = self.clock;
        Some(entry.slot)
    }

    /// Registers a demoted page under its exact identity.
    pub fn insert(&mut self, key: super::prefix::KvPrefixPageKey, slot: usize) {
        self.clock = self.clock.wrapping_add(1);
        let digest = key.digest();
        self.entries.insert(
            digest,
            HostChainEntry {
                key,
                slot,
                last_access: self.clock,
            },
        );
    }

    /// Removes one entry, returning its host slot for release.
    pub fn remove(&mut self, digest: &[u8; 32]) -> Option<usize> {
        self.entries.remove(digest).map(|entry| entry.slot)
    }

    /// Removes one entry and every host page that chains below it, freeing
    /// their slots (DS4 promotion failure purge and commit reclaim). Returns
    /// the number of freed slots.
    pub fn remove_chain_from(&mut self, pool: &mut KvHostPool, root: &[u8; 32]) -> usize {
        let mut pending = vec![*root];
        let mut freed = 0;
        while let Some(digest) = pending.pop() {
            pending.extend(
                self.entries
                    .iter()
                    .filter_map(|(child_digest, child)| {
                        (child.key.previous_page == Some(digest)).then_some(*child_digest)
                    })
                    .collect::<Vec<_>>(),
            );
            if let Some(slot) = self.remove(&digest) {
                if pool.release_slot(slot).is_ok() {
                    freed += 1;
                }
            }
        }
        freed
    }

    /// Evicts LRU host subtrees (an entry and its whole descendant chain)
    /// until `wanted` additional pages fit the pool, returning the number of
    /// freed slots.
    pub fn evict_until_fits(&mut self, pool: &mut KvHostPool, wanted: usize) -> usize {
        let mut freed = 0;
        while !pool.can_admit_pages(wanted) {
            let Some(root) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_access)
                .map(|(digest, _)| *digest)
            else {
                return freed;
            };
            let mut pending = vec![root];
            while let Some(digest) = pending.pop() {
                pending.extend(
                    self.entries
                        .iter()
                        .filter_map(|(child_digest, child)| {
                            (child.key.previous_page == Some(digest)).then_some(*child_digest)
                        })
                        .collect::<Vec<_>>(),
                );
                if let Some(slot) = self.remove(&digest) {
                    if pool.release_slot(slot).is_ok() {
                        freed += 1;
                    }
                }
            }
        }
        freed
    }
}

/// Per-arena host-tier state for one model.
#[derive(Debug)]
pub(crate) struct ArenaOffload {
    pub pool: KvHostPool,
    pub chain: HostChainIndex,
    next_transfer_id: u64,
}

impl ArenaOffload {
    pub fn new(page_bytes: u64, budget_bytes: u64) -> Result<Self> {
        Ok(Self {
            pool: KvHostPool::new(page_bytes, budget_bytes)?,
            chain: HostChainIndex::default(),
            next_transfer_id: 0,
        })
    }
}

/// Host-tier state for one model: one bounded pool + chain index per arena.
#[derive(Debug)]
pub(crate) struct ModelOffload {
    pub arenas: HashMap<crate::kv::KvArenaId, ArenaOffload>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DemoteOutcome {
    /// The LRU subtree moved to the host tier.
    Demoted { pages: usize },
    /// The victim was unusable (referenced pages, exhausted budget); the
    /// caller may try its next step or fall back to eviction.
    Skipped,
    /// The device index has nothing left to demote.
    Exhausted,
}

/// One demotion step: pick the LRU subtree, verify eligibility, move its
/// bytes to the host pool, and swap index ownership. Failure at any point
/// leaves device authority untouched (source bytes are never invalidated).
pub(crate) fn demote_step(
    coordinator: &mut KvCacheCoordinator,
    index: &mut CoordinatedPrefixIndex,
    arena: &dyn KvArena,
    offload: &mut ArenaOffload,
    telemetry: &ManagedKvTelemetry,
) -> Result<DemoteOutcome> {
    let Some(LruSubtreeView {
        digests,
        keys,
        blocks,
        ..
    }) = index.lru_subtree()
    else {
        return Ok(DemoteOutcome::Exhausted);
    };
    if blocks.is_empty() {
        return Ok(DemoteOutcome::Skipped);
    }
    // Active-request ownership excludes the page: those keep resolving
    // through the existing preemption ladder instead of demotion.
    for block in &blocks {
        if !coordinator
            .page_is_offload_candidate(*block)
            .map_err(coordinator_error)?
        {
            return Ok(DemoteOutcome::Skipped);
        }
    }
    // Host budget: retire cold host subtrees first; a full pool with nothing
    // evictable leaves the victim to ordinary eviction.
    offload
        .chain
        .evict_until_fits(&mut offload.pool, blocks.len());
    if !offload.pool.can_admit_pages(blocks.len()) {
        return Ok(DemoteOutcome::Skipped);
    }

    // Two-phase transfer: pin + mark Offloading, capture, then acknowledge
    // and swap ownership. A failed or stale step aborts back to source
    // authority and returns the host slots.
    coordinator
        .pin_transfer(&blocks)
        .map_err(coordinator_error)?;
    let mut slots = Vec::with_capacity(keys.len());
    let mut residency = Vec::with_capacity(keys.len());
    let mut failure = None;
    for (_key, block) in keys.iter().zip(&blocks) {
        offload.next_transfer_id = offload.next_transfer_id.wrapping_add(1);
        let transfer = KvTransferId(offload.next_transfer_id);
        let resident = KvResidencyState::Resident {
            tier: KvStorageTier::Device,
        };
        let state = match resident
            .begin_loading(transfer, KvStorageTier::Host)
            .and_then(|loading| loading.destination_allocated(transfer))
        {
            Ok(state) => state,
            Err(_) => {
                failure = Some(Error::InferenceError(
                    "KV offload residency transition rejected a demotion".into(),
                ));
                break;
            }
        };
        let Some(slot) = offload.pool.allocate_slot() else {
            failure = Some(Error::InferenceError(
                "KV host pool refused a demoted page".into(),
            ));
            break;
        };
        let capture = offload
            .pool
            .page_mut(slot)
            .and_then(|buffer| arena.capture_page(*block, buffer));
        if let Err(error) = capture {
            offload.pool.release_slot(slot).ok();
            failure = Some(error);
            break;
        }
        residency.push((transfer, state));
        slots.push(slot);
    }
    if let Some(error) = failure {
        for slot in slots {
            offload.pool.release_slot(slot).ok();
        }
        coordinator
            .unpin_transfer(&blocks)
            .map_err(coordinator_error)?;
        return Err(error);
    }
    for (transfer, state) in &residency {
        state
            .acknowledge(*transfer)
            .map_err(|_| Error::InferenceError("KV offload acknowledge rejected".into()))?;
    }
    index
        .remove_subtree(coordinator, &digests)
        .map_err(managed_prefix_error)?;
    for (key, slot) in keys.into_iter().zip(slots) {
        offload.chain.insert(key, slot);
    }
    coordinator
        .unpin_transfer(&blocks)
        .map_err(coordinator_error)?;
    telemetry.record_demotion(blocks.len());
    Ok(DemoteOutcome::Demoted {
        pages: blocks.len(),
    })
}

fn managed_prefix_error(error: super::prefix::KvPrefixIndexError) -> Error {
    Error::InferenceError(format!("KV offload prefix update failed: {error}"))
}

/// Device-prefix lookup extended into the host tier (DS4.3). The device walk
/// runs to its first miss, then the chain continues through the host index
/// under the same key identities. The host extension is capped at
/// `max_host_pages` so admission probes and prepare truncation agree; a zero
/// cap or a missing chain index keeps the lookup fully device-resident.
pub(crate) fn lookup_longest_with_host(
    index: &mut CoordinatedPrefixIndex,
    chain: &mut HostChainIndex,
    namespace: &KvPrefixNamespace,
    tokens: &[u32],
    page_tokens: u32,
    max_host_pages: usize,
) -> std::result::Result<KvPrefixMatch, KvPrefixIndexError> {
    let mut matched = index.lookup_longest(namespace, tokens, page_tokens)?;
    if max_host_pages == 0 {
        return Ok(matched);
    }
    let page = page_tokens as usize;
    let complete_pages = tokens.len() / page;
    let mut previous = matched.page_digests.last().copied();
    let mut digests = Vec::new();
    let mut slots = Vec::new();
    for page_index in matched.blocks.len()..complete_pages {
        if digests.len() >= max_host_pages {
            break;
        }
        let start = u64::try_from(page_index)
            .ok()
            .and_then(|page| page.checked_mul(page_tokens as u64))
            .ok_or(KvPrefixIndexError::CounterOverflow)?;
        let start = start as usize;
        let key = KvPrefixPageKey::new(
            namespace,
            previous,
            start as u64,
            tokens[start..start + page].to_vec(),
        )?;
        let Some(slot) = chain.lookup(&key) else {
            break;
        };
        previous = Some(key.digest());
        digests.push(key.digest());
        slots.push(slot);
    }
    if digests.is_empty() {
        return Ok(matched);
    }
    let tail_tokens = u32::try_from(digests.len() * page)
        .ok()
        .and_then(|tokens| matched.reused_tokens.checked_add(tokens))
        .ok_or(KvPrefixIndexError::CounterOverflow)?;
    matched.host_tail = Some(KvHostTailMatch {
        device_end_tokens: u32::try_from(matched.blocks.len() * page)
            .map_err(|_| KvPrefixIndexError::CounterOverflow)?,
        digests,
        slots,
    });
    matched.reused_tokens = tail_tokens;
    Ok(matched)
}

/// How much of a host-resident tail the promotion copy actually restored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PromotionCopy {
    pub restored_pages: usize,
    pub latency_ns: u64,
}

/// Restores the host-resident tail of a matched prefix into the fresh device
/// pages a preparing transaction reserved for that span (DS4.3 promotion
/// copy; the inverse of [`demote_step`]'s capture).
///
/// A page that cannot be restored truncates the promotion there: the failing
/// page and its host descendants are purged from the chain index (they would
/// be unreachable through it once their ancestor is gone), and the caller
/// falls back to the shorter device-resident prefix. Source authority never
/// depends on the copy succeeding.
pub(crate) fn promote_tail(
    arena: &dyn KvArena,
    offload: &mut ArenaOffload,
    blocks: &[CacheBlockRef],
    digests: &[[u8; 32]],
    slots: &[usize],
) -> PromotionCopy {
    let started = Instant::now();
    let mut restored = 0;
    while restored < blocks.len() {
        offload.next_transfer_id = offload.next_transfer_id.wrapping_add(1);
        let transfer = KvTransferId(offload.next_transfer_id);
        let resident = KvResidencyState::Resident {
            tier: KvStorageTier::Host,
        };
        let state = match resident
            .begin_loading(transfer, KvStorageTier::Device)
            .and_then(|loading| loading.destination_allocated(transfer))
        {
            Ok(state) => state,
            Err(_) => break,
        };
        let slot = slots[restored];
        let copied = offload
            .pool
            .page(slot)
            .and_then(|bytes| arena.restore_page(blocks[restored], bytes));
        if let Err(error) = copied {
            eprintln!("DS4DBG promote copy failed page {restored}: {error}");
            tracing::debug!(
                error = %error,
                "KV offload promotion copy failed; truncating the host tail"
            );
            let _ = state.abort(transfer);
            offload
                .chain
                .remove_chain_from(&mut offload.pool, &digests[restored]);
            break;
        }
        if state.acknowledge(transfer).is_err() {
            offload
                .chain
                .remove_chain_from(&mut offload.pool, &digests[restored]);
            break;
        }
        restored += 1;
    }
    PromotionCopy {
        restored_pages: restored,
        latency_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
    }
}

#[cfg(test)]
mod tests;
