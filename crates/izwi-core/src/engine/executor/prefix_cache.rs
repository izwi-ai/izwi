use std::collections::VecDeque;
use std::sync::{Arc, Mutex, Weak};

use crate::backends::BackendKind;
use crate::engine::resources::ResourceLease;
use crate::model::ModelVariant;
use crate::models::architectures::qwen35::chat::Qwen35PrefixSnapshot;

const DEFAULT_QWEN35_PREFIX_CACHE_BYTES: u64 = 0;
const MAX_QWEN35_PREFIX_CACHE_BYTES: u64 = 4 * 1024 * 1024 * 1024;

pub(super) fn configured_qwen35_prefix_cache_bytes() -> u64 {
    std::env::var("IZWI_QWEN35_PREFIX_CACHE_BYTES")
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_QWEN35_PREFIX_CACHE_BYTES)
        .min(MAX_QWEN35_PREFIX_CACHE_BYTES)
}

pub(super) trait ExactPrefixSnapshot: Send + Sync + 'static {
    fn token_ids(&self) -> &[u32];

    fn positions(&self) -> &[[usize; 3]];

    /// Storage whose lifetime this cache entry can prolong, including shared
    /// tensor backing allocations and owned token/position metadata.
    fn retained_bytes(&self) -> Option<u64>;
}

impl ExactPrefixSnapshot for Qwen35PrefixSnapshot {
    fn token_ids(&self) -> &[u32] {
        self.token_ids()
    }

    fn positions(&self) -> &[[usize; 3]] {
        self.positions()
    }

    fn retained_bytes(&self) -> Option<u64> {
        self.retained_bytes()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ExactPrefixScope {
    pub(super) variant: ModelVariant,
    pub(super) backend: BackendKind,
    pub(super) activation_dtype: String,
    pub(super) kv_cache_dtype: String,
}

pub(super) struct ExactPrefixHandle<S> {
    backend: BackendKind,
    // Snapshot precedes its lease so storage is dropped before authorization.
    snapshot: S,
    lease: Option<ResourceLease>,
}

impl<S> ExactPrefixHandle<S> {
    pub(super) fn new(
        backend: BackendKind,
        snapshot: S,
        lease: Option<ResourceLease>,
    ) -> Arc<Self> {
        Arc::new(Self {
            backend,
            snapshot,
            lease,
        })
    }

    pub(super) fn snapshot(&self) -> &S {
        &self.snapshot
    }
}

impl<S> Drop for ExactPrefixHandle<S> {
    fn drop(&mut self) {
        if let Some(lease) = self.lease.as_ref() {
            let _ =
                lease.prepare_materialized_release(super::cache_resource_vector(self.backend, 0));
        }
    }
}

struct ExactPrefixEntry<O, S> {
    scope: ExactPrefixScope,
    owner: Weak<O>,
    cached: Arc<ExactPrefixHandle<S>>,
    retained_bytes: u64,
}

struct ExactPrefixCacheState<O, S> {
    entries: VecDeque<ExactPrefixEntry<O, S>>,
    retained_bytes: u64,
}

impl<O, S> Default for ExactPrefixCacheState<O, S> {
    fn default() -> Self {
        Self {
            entries: VecDeque::new(),
            retained_bytes: 0,
        }
    }
}

/// Bounded LRU of immutable exact-prefix snapshots.
///
/// The owner is weakly referenced so an unloaded model cannot keep cache state
/// discoverable. Lookups clone only an `Arc` while holding the mutex; restored
/// Qwen state structurally shares immutable Candle tensor storage.
pub(super) struct ExactPrefixCache<O, S> {
    max_retained_bytes: u64,
    state: Mutex<ExactPrefixCacheState<O, S>>,
}

impl<O, S> ExactPrefixCache<O, S>
where
    O: Send + Sync + 'static,
    S: ExactPrefixSnapshot,
{
    pub(super) fn new(max_retained_bytes: u64) -> Self {
        Self {
            max_retained_bytes,
            state: Mutex::new(ExactPrefixCacheState::default()),
        }
    }

    pub(super) fn max_retained_bytes(&self) -> u64 {
        self.max_retained_bytes
    }

    pub(super) fn lookup(
        &self,
        owner: &Arc<O>,
        scope: &ExactPrefixScope,
        prompt_ids: &[u32],
        prompt_positions: &[[usize; 3]],
    ) -> Option<Arc<ExactPrefixHandle<S>>> {
        if self.max_retained_bytes == 0 || prompt_ids.len() != prompt_positions.len() {
            return None;
        }

        let requested_owner = Arc::downgrade(owner);
        let (cached, stale) = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            let stale = detach_stale_entries(&mut state);

            let best = state
                .entries
                .iter()
                .enumerate()
                .filter(|(_, entry)| {
                    entry.scope == *scope
                        && Weak::ptr_eq(&entry.owner, &requested_owner)
                        && exact_prefix_matches(
                            entry.cached.snapshot(),
                            prompt_ids,
                            prompt_positions,
                        )
                })
                .max_by_key(|(_, entry)| entry.cached.snapshot().token_ids().len())
                .map(|(index, _)| index);

            let cached = best.map(|best| {
                let entry = state
                    .entries
                    .remove(best)
                    .expect("selected exact-prefix entry must still exist");
                let cached = Arc::clone(&entry.cached);
                state.entries.push_back(entry);
                cached
            });
            (cached, stale)
        };
        // Resource leases and Candle/Metal storage must be released outside
        // the cache mutex to avoid lock inversion and long device frees while
        // other requests access the LRU.
        drop(stale);
        cached
    }

    /// Insert one immutable snapshot. Oversized or unaccountable snapshots are
    /// skipped instead of weakening the byte bound or failing inference.
    pub(super) fn insert(
        &self,
        owner: &Arc<O>,
        scope: ExactPrefixScope,
        cached: Arc<ExactPrefixHandle<S>>,
    ) -> bool {
        let Some(retained_bytes) = cached.snapshot().retained_bytes() else {
            return false;
        };
        if self.max_retained_bytes == 0 || retained_bytes > self.max_retained_bytes {
            return false;
        }

        let owner = Arc::downgrade(owner);
        let (inserted, detached) = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            let mut detached = detach_stale_entries(&mut state);
            let mut remove = vec![false; state.entries.len()];
            let replacement = cached.snapshot();

            // Plan replacement and ancestor compaction before mutating the
            // LRU. A failed fit must leave every useful entry intact.
            for (index, entry) in state.entries.iter().enumerate() {
                if entry.scope != scope || !Weak::ptr_eq(&entry.owner, &owner) {
                    continue;
                }
                let candidate = entry.cached.snapshot();
                let exact = candidate.token_ids() == replacement.token_ids()
                    && candidate.positions() == replacement.positions();
                if exact && Arc::strong_count(&entry.cached) > 1 {
                    drop(state);
                    drop(detached);
                    return false;
                }
                let ancestor = candidate.token_ids().len() < replacement.token_ids().len()
                    && replacement.token_ids().starts_with(candidate.token_ids())
                    && replacement.positions().starts_with(candidate.positions());
                if exact || (ancestor && Arc::strong_count(&entry.cached) == 1) {
                    remove[index] = true;
                }
            }

            let mut projected = state.retained_bytes;
            for (index, entry) in state.entries.iter().enumerate() {
                if remove[index] {
                    projected = projected.saturating_sub(entry.retained_bytes);
                }
            }

            while projected.saturating_add(retained_bytes) > self.max_retained_bytes {
                let Some((index, entry)) =
                    state.entries.iter().enumerate().find(|(index, entry)| {
                        !remove[*index] && Arc::strong_count(&entry.cached) == 1
                    })
                else {
                    drop(state);
                    drop(detached);
                    return false;
                };
                remove[index] = true;
                projected = projected.saturating_sub(entry.retained_bytes);
            }

            for index in (0..remove.len()).rev() {
                if !remove[index] {
                    continue;
                }
                let entry = state
                    .entries
                    .remove(index)
                    .expect("planned exact-prefix removal must still exist");
                state.retained_bytes = state.retained_bytes.saturating_sub(entry.retained_bytes);
                detached.push(entry);
            }

            state.retained_bytes = state.retained_bytes.saturating_add(retained_bytes);
            state.entries.push_back(ExactPrefixEntry {
                scope,
                owner,
                cached,
                retained_bytes,
            });
            (true, detached)
        };
        drop(detached);
        inserted
    }

    pub(super) fn retained_bytes(&self) -> u64 {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .retained_bytes
    }

    /// Purge all checkpoints owned by one model variant.
    ///
    /// Model unload calls this after active requests have been aborted and
    /// before the model residency lease is released. Any external lookup
    /// handle remains valid and keeps its own resource lease until it drops.
    pub(super) fn purge_variant(&self, variant: ModelVariant) -> usize {
        let detached = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            detach_matching_entries(&mut state, |entry| entry.scope.variant == variant)
        };
        let removed = detached.len();
        drop(detached);
        removed
    }

    pub(super) fn clear(&self) {
        let detached = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            state.retained_bytes = 0;
            std::mem::take(&mut state.entries)
        };
        drop(detached);
    }
}

fn exact_prefix_matches<S: ExactPrefixSnapshot + ?Sized>(
    snapshot: &S,
    prompt_ids: &[u32],
    prompt_positions: &[[usize; 3]],
) -> bool {
    let cached_ids = snapshot.token_ids();
    let cached_positions = snapshot.positions();
    !cached_ids.is_empty()
        && cached_ids.len() == cached_positions.len()
        && prompt_ids.starts_with(cached_ids)
        && prompt_positions.starts_with(cached_positions)
}

fn detach_stale_entries<O, S>(
    state: &mut ExactPrefixCacheState<O, S>,
) -> Vec<ExactPrefixEntry<O, S>> {
    // A lookup handle owns the snapshot and its lease independently of the LRU
    // entry, so a dead model owner can always be detached immediately.
    detach_matching_entries(state, |entry| entry.owner.strong_count() == 0)
}

fn detach_matching_entries<O, S>(
    state: &mut ExactPrefixCacheState<O, S>,
    mut should_detach: impl FnMut(&ExactPrefixEntry<O, S>) -> bool,
) -> Vec<ExactPrefixEntry<O, S>> {
    let mut detached = Vec::new();
    let mut index = 0usize;
    while index < state.entries.len() {
        if should_detach(&state.entries[index]) {
            let entry = state
                .entries
                .remove(index)
                .expect("selected exact-prefix entry must still exist");
            state.retained_bytes = state.retained_bytes.saturating_sub(entry.retained_bytes);
            detached.push(entry);
        } else {
            index += 1;
        }
    }
    detached
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::resources::{
        PhysicalCapacityProvider, PhysicalCapacitySnapshot, ReservationClass, ReservationOwner,
        ResourceAuthority,
    };

    #[derive(Debug)]
    struct TestSnapshot {
        ids: Vec<u32>,
        positions: Vec<[usize; 3]>,
        bytes: u64,
        label: &'static str,
    }

    #[derive(Debug)]
    struct TestCapacityProvider {
        snapshot: PhysicalCapacitySnapshot,
    }

    impl PhysicalCapacityProvider for TestCapacityProvider {
        fn snapshot(&self) -> PhysicalCapacitySnapshot {
            self.snapshot
        }
    }

    impl ExactPrefixSnapshot for TestSnapshot {
        fn token_ids(&self) -> &[u32] {
            &self.ids
        }

        fn positions(&self) -> &[[usize; 3]] {
            &self.positions
        }

        fn retained_bytes(&self) -> Option<u64> {
            Some(self.bytes)
        }
    }

    #[test]
    fn qwen35_prefix_cache_is_opt_in_and_bounded() {
        let _guard = crate::env_test_lock().lock().expect("env lock");
        std::env::remove_var("IZWI_QWEN35_PREFIX_CACHE_BYTES");
        assert_eq!(configured_qwen35_prefix_cache_bytes(), 0);

        std::env::set_var("IZWI_QWEN35_PREFIX_CACHE_BYTES", "1048576");
        assert_eq!(configured_qwen35_prefix_cache_bytes(), 1_048_576);

        std::env::set_var("IZWI_QWEN35_PREFIX_CACHE_BYTES", u64::MAX.to_string());
        assert_eq!(
            configured_qwen35_prefix_cache_bytes(),
            MAX_QWEN35_PREFIX_CACHE_BYTES
        );
        std::env::remove_var("IZWI_QWEN35_PREFIX_CACHE_BYTES");
    }

    fn scope_for(variant: ModelVariant) -> ExactPrefixScope {
        ExactPrefixScope {
            variant,
            backend: BackendKind::Cpu,
            activation_dtype: "float32".to_string(),
            kv_cache_dtype: "float16".to_string(),
        }
    }

    fn scope() -> ExactPrefixScope {
        scope_for(ModelVariant::Qwen354BGguf)
    }

    fn snapshot(
        ids: &[u32],
        bytes: u64,
        label: &'static str,
    ) -> Arc<ExactPrefixHandle<TestSnapshot>> {
        ExactPrefixHandle::new(
            BackendKind::Cpu,
            TestSnapshot {
                ids: ids.to_vec(),
                positions: (0..ids.len()).map(|index| [index; 3]).collect(),
                bytes,
                label,
            },
            None,
        )
    }

    fn positions(len: usize) -> Vec<[usize; 3]> {
        (0..len).map(|index| [index; 3]).collect()
    }

    #[test]
    fn lookup_returns_the_longest_exact_token_and_position_prefix() {
        let cache = ExactPrefixCache::<(), TestSnapshot>::new(128);
        let owner = Arc::new(());
        assert!(cache.insert(&owner, scope(), snapshot(&[1, 2], 10, "short")));
        assert!(cache.insert(&owner, scope(), snapshot(&[1, 2, 3], 10, "long"),));

        let hit = cache
            .lookup(&owner, &scope(), &[1, 2, 3, 4], &positions(4))
            .expect("longest exact prefix");
        assert_eq!(hit.snapshot().label, "long");
    }

    #[test]
    fn lookup_rejects_token_position_scope_and_model_mismatches() {
        let cache = ExactPrefixCache::<(), TestSnapshot>::new(128);
        let owner = Arc::new(());
        assert!(cache.insert(&owner, scope(), snapshot(&[1, 2, 3], 10, "entry"),));

        assert!(cache
            .lookup(&owner, &scope(), &[1, 9, 3, 4], &positions(4))
            .is_none());
        let mut wrong_positions = positions(4);
        wrong_positions[2] = [99; 3];
        assert!(cache
            .lookup(&owner, &scope(), &[1, 2, 3, 4], &wrong_positions)
            .is_none());

        let mut wrong_scope = scope();
        wrong_scope.backend = BackendKind::Metal;
        assert!(cache
            .lookup(&owner, &wrong_scope, &[1, 2, 3, 4], &positions(4))
            .is_none());
        assert!(cache
            .lookup(&Arc::new(()), &scope(), &[1, 2, 3, 4], &positions(4),)
            .is_none());
    }

    #[test]
    fn byte_bound_evicts_the_least_recently_used_entry() {
        let cache = ExactPrefixCache::<(), TestSnapshot>::new(12);
        let owner = Arc::new(());
        assert!(cache.insert(&owner, scope(), snapshot(&[1], 4, "first")));
        assert!(cache.insert(&owner, scope(), snapshot(&[2], 4, "second")));
        assert!(cache
            .lookup(&owner, &scope(), &[1, 9], &positions(2))
            .is_some());
        assert!(cache.insert(&owner, scope(), snapshot(&[3], 6, "third")));

        assert!(cache
            .lookup(&owner, &scope(), &[2, 9], &positions(2))
            .is_none());
        assert!(cache
            .lookup(&owner, &scope(), &[1, 9], &positions(2))
            .is_some());
        assert!(cache
            .lookup(&owner, &scope(), &[3, 9], &positions(2))
            .is_some());
        assert_eq!(cache.retained_bytes(), 10);
    }

    #[test]
    fn monotonic_lineage_supersedes_unpinned_ancestors_but_preserves_branches() {
        let cache = ExactPrefixCache::<(), TestSnapshot>::new(128);
        let owner = Arc::new(());
        assert!(cache.insert(&owner, scope(), snapshot(&[1], 4, "root")));
        assert!(cache.insert(&owner, scope(), snapshot(&[1, 2], 5, "middle")));
        assert!(cache.insert(&owner, scope(), snapshot(&[9, 8], 6, "branch")));
        assert!(cache.insert(&owner, scope(), snapshot(&[1, 2, 3], 7, "tip")));

        assert!(cache
            .lookup(&owner, &scope(), &[1, 2, 9], &positions(3))
            .is_none());
        assert_eq!(
            cache
                .lookup(&owner, &scope(), &[1, 2, 3, 4], &positions(4))
                .unwrap()
                .snapshot()
                .label,
            "tip"
        );
        assert_eq!(
            cache
                .lookup(&owner, &scope(), &[9, 8, 7], &positions(3))
                .unwrap()
                .snapshot()
                .label,
            "branch"
        );
        assert_eq!(cache.retained_bytes(), 13);
    }

    #[test]
    fn failed_insert_does_not_destroy_existing_ancestors() {
        let cache = ExactPrefixCache::<(), TestSnapshot>::new(8);
        let owner = Arc::new(());
        assert!(cache.insert(&owner, scope(), snapshot(&[1], 8, "root")));
        let pinned = cache
            .lookup(&owner, &scope(), &[1, 2], &positions(2))
            .unwrap();

        assert!(!cache.insert(&owner, scope(), snapshot(&[1, 2], 8, "tip")));
        assert_eq!(cache.retained_bytes(), 8);
        assert_eq!(
            cache
                .lookup(&owner, &scope(), &[1, 3], &positions(2))
                .unwrap()
                .snapshot()
                .label,
            "root"
        );
        drop(pinned);
    }

    #[test]
    fn stale_model_owner_is_pruned_and_oversized_entries_are_skipped() {
        let cache = ExactPrefixCache::<(), TestSnapshot>::new(8);
        let owner = Arc::new(());
        assert!(cache.insert(&owner, scope(), snapshot(&[1], 8, "stale")));
        drop(owner);

        let replacement_owner = Arc::new(());
        assert!(cache
            .lookup(&replacement_owner, &scope(), &[1, 2], &positions(2),)
            .is_none());
        assert_eq!(cache.retained_bytes(), 0);
        assert!(!cache.insert(&replacement_owner, scope(), snapshot(&[2], 9, "oversized"),));
    }

    #[test]
    fn lookup_handle_keeps_exact_resource_lease_after_cache_clear() {
        let capacity = super::super::cache_resource_vector(BackendKind::Cpu, 64);
        let authority = Arc::new(ResourceAuthority::new(Arc::new(TestCapacityProvider {
            snapshot: PhysicalCapacitySnapshot {
                capacity,
                available: capacity,
                source: crate::engine::resources::CapacitySource::Test,
            },
        })));
        let resources = super::super::cache_resource_vector(BackendKind::Cpu, 4);
        let lease = authority
            .reserve_with_initial_materialized(
                ReservationOwner::new(ReservationClass::Cache, "prefix"),
                resources,
                resources,
            )
            .unwrap();
        let cache = ExactPrefixCache::<(), TestSnapshot>::new(8);
        let owner = Arc::new(());
        let cached = ExactPrefixHandle::new(
            BackendKind::Cpu,
            TestSnapshot {
                ids: vec![1],
                positions: vec![[0; 3]],
                bytes: 4,
                label: "leased",
            },
            Some(lease),
        );
        assert!(cache.insert(&owner, scope(), cached));
        let hit = cache
            .lookup(&owner, &scope(), &[1, 2], &positions(2))
            .unwrap();

        cache.clear();
        assert_eq!(authority.snapshot().reservations, 1);
        assert_eq!(hit.snapshot().label, "leased");
        drop(hit);
        assert_eq!(authority.snapshot().reservations, 0);
    }

    #[test]
    fn dropping_unpublished_pending_handle_releases_exact_authorization() {
        let capacity = super::super::cache_resource_vector(BackendKind::Cpu, 64);
        let authority = Arc::new(ResourceAuthority::new(Arc::new(TestCapacityProvider {
            snapshot: PhysicalCapacitySnapshot {
                capacity,
                available: capacity,
                source: crate::engine::resources::CapacitySource::Test,
            },
        })));
        let resources = super::super::cache_resource_vector(BackendKind::Cpu, 4);
        let lease = authority
            .reserve_with_initial_materialized(
                ReservationOwner::new(ReservationClass::Cache, "pending-prefix"),
                resources,
                resources,
            )
            .unwrap();
        let pending = ExactPrefixHandle::new(
            BackendKind::Cpu,
            TestSnapshot {
                ids: vec![1],
                positions: vec![[0; 3]],
                bytes: 4,
                label: "pending",
            },
            Some(lease),
        );
        assert_eq!(authority.snapshot().reservations, 1);
        drop(pending);
        assert_eq!(authority.snapshot().reservations, 0);
    }

    #[test]
    fn variant_purge_releases_only_matching_unpinned_leases() {
        let capacity = super::super::cache_resource_vector(BackendKind::Cpu, 64);
        let authority = Arc::new(ResourceAuthority::new(Arc::new(TestCapacityProvider {
            snapshot: PhysicalCapacitySnapshot {
                capacity,
                available: capacity,
                source: crate::engine::resources::CapacitySource::Test,
            },
        })));
        let cache = ExactPrefixCache::<(), TestSnapshot>::new(64);
        let owner = Arc::new(());

        for (variant, id, label) in [
            (ModelVariant::Qwen354BGguf, 1, "four"),
            (ModelVariant::Qwen352BGguf, 2, "two"),
        ] {
            let resources = super::super::cache_resource_vector(BackendKind::Cpu, 4);
            let lease = authority
                .reserve_with_initial_materialized(
                    ReservationOwner::new(ReservationClass::Cache, label),
                    resources,
                    resources,
                )
                .unwrap();
            assert!(cache.insert(
                &owner,
                scope_for(variant),
                ExactPrefixHandle::new(
                    BackendKind::Cpu,
                    TestSnapshot {
                        ids: vec![id],
                        positions: vec![[0; 3]],
                        bytes: 4,
                        label,
                    },
                    Some(lease),
                ),
            ));
        }
        assert_eq!(authority.snapshot().reservations, 2);

        assert_eq!(cache.purge_variant(ModelVariant::Qwen354BGguf), 1);
        assert_eq!(cache.retained_bytes(), 4);
        assert_eq!(authority.snapshot().reservations, 1);
        assert!(cache
            .lookup(
                &owner,
                &scope_for(ModelVariant::Qwen354BGguf),
                &[1, 3],
                &positions(2),
            )
            .is_none());
        assert!(cache
            .lookup(
                &owner,
                &scope_for(ModelVariant::Qwen352BGguf),
                &[2, 3],
                &positions(2),
            )
            .is_some());
    }
}
