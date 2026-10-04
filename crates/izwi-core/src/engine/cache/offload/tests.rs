use super::{
    demote_step, lookup_longest_with_host, promote_tail, ArenaOffload, DemoteOutcome,
    HostChainIndex, HostOffloadPolicy,
};
use crate::backends::kv::{arena_page_bytes, CpuKvArena, KvArena};
use crate::backends::BackendKind;
use crate::engine::cache::coordinator::KvSnapshot;
use crate::engine::cache::coordinator::{
    KvBlockIntent, KvCacheCoordinator, KvGroupReservation, KvReserveRequest,
};
use crate::engine::cache::prefix::{
    CoordinatedPrefixIndex, KvPrefixNamespace, KvPrefixPageKey, KvPrefixPublication,
};
use crate::engine::EngineCoreConfig;
use crate::engine::{ModelInstanceId, SessionKey};
use crate::kv::{CacheBlockRef, CacheDomainId, KvArenaId, KvGroupId, KvPlanFingerprint};

fn arena_id() -> KvArenaId {
    KvArenaId {
        model_instance: ModelInstanceId::new(7),
        backend: BackendKind::Cpu,
        device_ordinal: None,
        generation: 1,
    }
}

fn block(index: u32) -> CacheBlockRef {
    CacheBlockRef {
        arena: arena_id(),
        group: KvGroupId::new(0),
        index,
        slot_generation: 1,
    }
}

fn namespace() -> KvPrefixNamespace {
    KvPrefixNamespace {
        model_instance: ModelInstanceId::new(7),
        model_revision: [1; 32],
        adapter_abi: [2; 32],
        tokenizer_or_input_encoding: [3; 32],
        position_semantics: [4; 32],
        plan: KvPlanFingerprint::new([5; 32]),
        multimodal_artifact: None,
        cache_salt: [9; 32],
    }
}

fn arena() -> CpuKvArena {
    CpuKvArena::new(crate::backends::kv::KvArenaConfig {
        id: arena_id(),
        group: KvGroupId::new(0),
        page_tokens: 2,
        capacity_pages: 4,
        growth: None,
        dtype: candle_core::DType::F32,
        layers: vec![crate::backends::kv::KvLayerConfig {
            binding: crate::kv::KvLayerBinding {
                model_layer: 0,
                physical_layer: 0,
            },
            num_kv_heads: 1,
            key_head_dim: 2,
            value_head_dim: 1,
        }],
    })
    .unwrap()
}

/// Publishes one chained two-page prefix whose pages carry distinctive
/// f32 bit patterns, returning (keys, blocks, expected page bytes).
fn publish_seeded_prefix(
    coordinator: &mut KvCacheCoordinator,
    index: &mut CoordinatedPrefixIndex,
    physical: &CpuKvArena,
) -> (
    Vec<KvPrefixPageKey>,
    Vec<CacheBlockRef>,
    Vec<Vec<u8>>,
    KvSnapshot,
) {
    let first = KvPrefixPageKey::new(&namespace(), None, 0, vec![1, 2]).unwrap();
    let second = KvPrefixPageKey::new(&namespace(), Some(first.digest()), 2, vec![3, 4]).unwrap();
    let snapshot = coordinator
        .register_table(
            SessionKey::new("session-a".to_string(), 0),
            CacheDomainId::new(0),
        )
        .unwrap();
    coordinator
        .reserve(KvReserveRequest {
            txn_id: 1,
            expected: snapshot,
            target_committed_tokens: 4,
            target_window_start: 0,
            groups: vec![KvGroupReservation {
                group: KvGroupId::new(0),
                blocks: vec![KvBlockIntent::Fresh, KvBlockIntent::Fresh],
            }],
        })
        .unwrap();
    let prepared = coordinator.prepare(1).unwrap();
    let blocks = prepared.writable_blocks.clone();
    let page_bytes = arena_page_bytes(physical.config()) as usize;
    let mut seeds = Vec::new();
    for (page, block) in blocks.iter().enumerate() {
        let mut bytes = vec![0_u8; page_bytes];
        for element in 0..page_bytes / 4 {
            let bits = (0x3f00_0000_u32 + (page * 64 + element) as u32).to_ne_bytes();
            bytes[element * 4..(element + 1) * 4].copy_from_slice(&bits);
        }
        physical.restore_page(*block, &bytes).unwrap();
        seeds.push(bytes);
    }
    coordinator
        .complete_write(crate::engine::cache::coordinator::KvWriteReceipt {
            txn_id: 1,
            committed_tokens: 4,
            written_blocks: blocks.clone(),
        })
        .unwrap();
    let publications = vec![
        KvPrefixPublication {
            key: first.clone(),
            block: blocks[0],
        },
        KvPrefixPublication {
            key: second.clone(),
            block: blocks[1],
        },
    ];
    let snapshot = index
        .commit_transaction(coordinator, 1, 2, &publications)
        .unwrap();
    (vec![first, second], blocks, seeds, snapshot)
}

#[test]
fn demotion_moves_the_lru_subtree_to_the_host_pool() {
    let physical = arena();
    let mut coordinator = KvCacheCoordinator::new(arena_id(), 4);
    let mut index = CoordinatedPrefixIndex::new(8);
    let (keys, _blocks, seeds, snapshot) =
        publish_seeded_prefix(&mut coordinator, &mut index, &physical);
    assert_eq!(coordinator.stats().prefix_refs, 2);

    // The committing request's table still holds its pages; releasing it is
    // what leaves them demotion candidates (durable prefix refs only).
    coordinator
        .release_table(
            &SessionKey::new("session-a".to_string(), 0),
            CacheDomainId::new(0),
        )
        .unwrap();
    let _ = snapshot;

    let mut offload = ArenaOffload::new(arena_page_bytes(physical.config()), 4096).unwrap();
    let outcome = demote_step(
        &mut coordinator,
        &mut index,
        &physical,
        &mut offload,
        &Default::default(),
    )
    .unwrap();
    assert_eq!(outcome, DemoteOutcome::Demoted { pages: 2 });

    // Device side: the subtree left the index and its pages recycled.
    let stats = coordinator.stats();
    assert_eq!(stats.allocated_pages, 0);
    assert_eq!(stats.prefix_refs, 0);
    assert_eq!(index.len(), 0);

    // Host side: the pool holds the pages with their exact bytes, findable
    // through the chain index under the original identities.
    assert_eq!(offload.pool.resident_pages(), 2);
    assert_eq!(offload.chain.len(), 2);
    for (key, seed) in keys.iter().zip(seeds.iter()) {
        let slot = offload.chain.lookup(key).expect("host entry exists");
        assert_eq!(offload.pool.page(slot).unwrap(), seed.as_slice());
    }
}

#[test]
fn demotion_skips_subtrees_with_referenced_pages() {
    let physical = arena();
    let mut coordinator = KvCacheCoordinator::new(arena_id(), 4);
    let mut index = CoordinatedPrefixIndex::new(8);
    let (_, blocks, _, _) = publish_seeded_prefix(&mut coordinator, &mut index, &physical);
    coordinator
        .release_table(
            &SessionKey::new("session-a".to_string(), 0),
            CacheDomainId::new(0),
        )
        .unwrap();

    // Attach the subtree root to an active table: ownership beyond the
    // durable prefix reference excludes the page from demotion.
    let snapshot = coordinator
        .register_table(
            SessionKey::new("session-b".to_string(), 0),
            CacheDomainId::new(0),
        )
        .unwrap();
    coordinator
        .reserve(KvReserveRequest {
            txn_id: 2,
            expected: snapshot,
            target_committed_tokens: 6,
            target_window_start: 0,
            groups: vec![KvGroupReservation {
                group: KvGroupId::new(0),
                blocks: vec![KvBlockIntent::Shared(blocks[0]), KvBlockIntent::Fresh],
            }],
        })
        .unwrap();

    let mut offload = ArenaOffload::new(arena_page_bytes(physical.config()), 4096).unwrap();
    let outcome = demote_step(
        &mut coordinator,
        &mut index,
        &physical,
        &mut offload,
        &Default::default(),
    )
    .unwrap();
    assert_eq!(outcome, DemoteOutcome::Skipped);
    assert_eq!(offload.pool.resident_pages(), 0);
    assert_eq!(index.len(), 2, "device index untouched");
}

#[test]
fn demotion_falls_back_when_the_host_budget_is_exhausted() {
    let physical = arena();
    let mut coordinator = KvCacheCoordinator::new(arena_id(), 4);
    let mut index = CoordinatedPrefixIndex::new(8);
    let (_, _, _, _) = publish_seeded_prefix(&mut coordinator, &mut index, &physical);

    // A pool that cannot hold even one page skips the step; the victim stays
    // available to ordinary eviction.
    let mut offload = ArenaOffload::new(arena_page_bytes(physical.config()), 4).unwrap();
    let outcome = demote_step(
        &mut coordinator,
        &mut index,
        &physical,
        &mut offload,
        &Default::default(),
    )
    .unwrap();
    assert_eq!(outcome, DemoteOutcome::Skipped);
    assert_eq!(index.len(), 2);
    assert_eq!(coordinator.stats().prefix_refs, 2);
}

#[test]
fn host_chain_eviction_frees_slots_until_the_pool_fits() {
    let mut pool = crate::backends::kv::KvHostPool::new(16, 3 * 16).unwrap();
    let mut chain = HostChainIndex::default();
    let ns = namespace();
    let mut keys = Vec::new();
    for page in 0..3_u64 {
        // Independent roots: each subtree is exactly one page.
        let key = KvPrefixPageKey::new(&ns, None, page * 2, vec![1, 2]).unwrap();
        let slot = pool.allocate_slot().expect("slot fits");
        chain.insert(key.clone(), slot);
        keys.push(key);
    }
    assert_eq!(chain.len(), 3);

    // Fitting two more pages evicts the two LRU roots.
    let freed = chain.evict_until_fits(&mut pool, 2);
    assert_eq!(freed, 2);
    assert!(pool.can_admit_pages(2));

    // The most recently inserted entry stays key-verifiable.
    assert!(chain.lookup(&keys[2]).is_some());
    assert!(chain.lookup(&keys[0]).is_none(), "LRU root was evicted");
    assert!(
        chain.lookup(&keys[1]).is_none(),
        "second LRU root was evicted"
    );
}

#[test]
fn offload_policy_resolves_engagement_and_watermark_sanity() {
    let mut config = EngineCoreConfig::default();
    assert_eq!(HostOffloadPolicy::resolve(&config).unwrap(), None);

    config.kv_host_pool_budget_bytes = 1024;
    let policy = HostOffloadPolicy::resolve(&config).unwrap().unwrap();
    assert_eq!(policy.budget_bytes, 1024);
    assert_eq!(policy.high_watermark, 0.85);
    assert_eq!(policy.low_watermark, 0.70);

    config.kv_offload_high_watermark = 0.5;
    config.kv_offload_low_watermark = 0.7;
    assert!(HostOffloadPolicy::resolve(&config).is_err());
}

/// Demotes a seeded two-page prefix and releases its table, leaving the pages
/// reachable only through the host chain.
fn demoted_prefix() -> (
    CpuKvArena,
    KvCacheCoordinator,
    CoordinatedPrefixIndex,
    ArenaOffload,
    Vec<KvPrefixPageKey>,
    Vec<Vec<u8>>,
) {
    let physical = arena();
    let mut coordinator = KvCacheCoordinator::new(arena_id(), 4);
    let mut index = CoordinatedPrefixIndex::new(8);
    let (keys, _, seeds, _) = publish_seeded_prefix(&mut coordinator, &mut index, &physical);
    coordinator
        .release_table(
            &SessionKey::new("session-a".to_string(), 0),
            CacheDomainId::new(0),
        )
        .unwrap();
    let mut offload = ArenaOffload::new(arena_page_bytes(physical.config()), 4096).unwrap();
    assert_eq!(
        demote_step(
            &mut coordinator,
            &mut index,
            &physical,
            &mut offload,
            &Default::default(),
        )
        .unwrap(),
        DemoteOutcome::Demoted { pages: 2 }
    );
    (physical, coordinator, index, offload, keys, seeds)
}

#[test]
fn lookup_continues_from_the_device_index_into_the_host_chain() {
    let (_, _coordinator, mut index, mut offload, keys, _) = demoted_prefix();

    let tokens = vec![1, 2, 3, 4, 5];
    let device_only = index.lookup_longest(&namespace(), &tokens, 2).unwrap();
    assert!(device_only.blocks.is_empty());
    assert!(device_only.host_tail.is_none());

    let matched =
        lookup_longest_with_host(&mut index, &mut offload.chain, &namespace(), &tokens, 2, 64)
            .unwrap();
    assert_eq!(matched.reused_tokens, 4);
    assert!(matched.blocks.is_empty(), "device head is empty here");
    let tail = matched.host_tail.expect("host tail extends the match");
    assert_eq!(tail.device_end_tokens, 0);
    assert_eq!(tail.digests, vec![keys[0].digest(), keys[1].digest()]);
    assert_eq!(tail.slots.len(), 2);
    assert_eq!(
        matched.page_digests,
        vec![keys[0].digest(), keys[1].digest()],
        "the digest chain stays complete across the tier boundary"
    );
}

#[test]
fn host_continuation_truncates_at_the_promotion_ceiling() {
    let (_, _, mut index, mut offload, keys, _) = demoted_prefix();

    let matched = lookup_longest_with_host(
        &mut index,
        &mut offload.chain,
        &namespace(),
        &[1, 2, 3, 4],
        2,
        1,
    )
    .unwrap();
    assert_eq!(matched.reused_tokens, 2);
    let tail = matched.host_tail.expect("one host page fits the ceiling");
    assert_eq!(tail.digests, vec![keys[0].digest()]);
    assert_eq!(tail.device_end_tokens, 0);

    let capped_out = lookup_longest_with_host(
        &mut index,
        &mut offload.chain,
        &namespace(),
        &[1, 2, 3, 4],
        2,
        0,
    )
    .unwrap();
    assert_eq!(capped_out.reused_tokens, 0);
    assert!(
        capped_out.host_tail.is_none(),
        "zero ceiling disables the tail"
    );
}

#[test]
fn promotion_restores_host_bytes_into_fresh_device_pages() {
    let (physical, mut coordinator, mut index, mut offload, keys, seeds) = demoted_prefix();

    // A preparing transaction reserves fresh pages for the whole span, the
    // way prepare_inner does before restoring the tail into them.
    let snapshot = coordinator
        .register_table(
            SessionKey::new("session-b".to_string(), 0),
            CacheDomainId::new(0),
        )
        .unwrap();
    coordinator
        .reserve(KvReserveRequest {
            txn_id: 2,
            expected: snapshot,
            target_committed_tokens: 6,
            target_window_start: 0,
            groups: vec![KvGroupReservation {
                group: KvGroupId::new(0),
                blocks: vec![
                    KvBlockIntent::Fresh,
                    KvBlockIntent::Fresh,
                    KvBlockIntent::Fresh,
                ],
            }],
        })
        .unwrap();
    let prepared = coordinator.prepare(2).unwrap();
    let blocks = prepared.provisional_groups[0].blocks.clone();
    assert_eq!(blocks.len(), 3);

    let tail = lookup_longest_with_host(
        &mut index,
        &mut offload.chain,
        &namespace(),
        &[1, 2, 3, 4],
        2,
        64,
    )
    .unwrap()
    .host_tail
    .expect("matched tail");
    let copy = promote_tail(
        &physical,
        &mut offload,
        &blocks[..tail.slots.len()],
        &tail.digests,
        &tail.slots,
    );
    assert_eq!(copy.restored_pages, 2);

    // The device pages now carry the exact bytes the host slots held.
    let page_bytes = arena_page_bytes(physical.config()) as usize;
    let mut readback = vec![0_u8; page_bytes];
    for (index, block) in blocks.iter().take(2).enumerate() {
        physical.capture_page(*block, &mut readback).unwrap();
        assert_eq!(readback, seeds[index]);
    }

    // Commit publishes the restored pages into the device index and reclaims
    // the host entries, leaving the chain empty and the pool released.
    coordinator
        .complete_write(crate::engine::cache::coordinator::KvWriteReceipt {
            txn_id: 2,
            committed_tokens: 6,
            written_blocks: blocks.clone(),
        })
        .unwrap();
    index
        .commit_transaction(
            &mut coordinator,
            2,
            2,
            &[
                KvPrefixPublication {
                    key: keys[0].clone(),
                    block: blocks[0],
                },
                KvPrefixPublication {
                    key: keys[1].clone(),
                    block: blocks[1],
                },
            ],
        )
        .unwrap();
    for digest in &tail.digests {
        if let Some(slot) = offload.chain.remove(digest) {
            offload.pool.release_slot(slot).unwrap();
        }
    }
    assert_eq!(offload.chain.len(), 0);
    assert_eq!(offload.pool.resident_pages(), 0);

    let device_match = index
        .lookup_longest(&namespace(), &[1, 2, 3, 4], 2)
        .unwrap();
    assert_eq!(device_match.blocks, vec![blocks[0], blocks[1]]);
    assert!(device_match.host_tail.is_none());
}

#[test]
fn a_failed_promotion_copy_purges_the_tail_from_the_host_chain() {
    let (physical, mut coordinator, _index, mut offload, keys, _) = demoted_prefix();

    // One real fresh page so the first copy succeeds; an out-of-range block
    // makes the second copy fail mid-tail.
    let snapshot = coordinator
        .register_table(
            SessionKey::new("session-b".to_string(), 0),
            CacheDomainId::new(0),
        )
        .unwrap();
    coordinator
        .reserve(KvReserveRequest {
            txn_id: 2,
            expected: snapshot,
            target_committed_tokens: 2,
            target_window_start: 0,
            groups: vec![KvGroupReservation {
                group: KvGroupId::new(0),
                blocks: vec![KvBlockIntent::Fresh],
            }],
        })
        .unwrap();
    let prepared = coordinator.prepare(2).unwrap();
    let valid = prepared.provisional_groups[0].blocks[0];

    let slots = keys
        .iter()
        .map(|key| offload.chain.lookup(key).expect("host entry"))
        .collect::<Vec<_>>();
    let copy = promote_tail(
        &physical,
        &mut offload,
        &[valid, block(99)],
        &keys.iter().map(|key| key.digest()).collect::<Vec<_>>(),
        &slots,
    );
    assert_eq!(copy.restored_pages, 1);
    assert_eq!(
        offload.chain.len(),
        1,
        "failing page and descendants purged"
    );
    assert!(offload.chain.lookup(&keys[0]).is_some());
    assert!(offload.chain.lookup(&keys[1]).is_none());
    assert_eq!(offload.pool.resident_pages(), 1);
}
