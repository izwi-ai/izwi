use super::{demote_step, ArenaOffload, DemoteOutcome, HostChainIndex, HostOffloadPolicy};
use crate::backends::kv::{arena_page_bytes, CpuKvArena, KvArena};
use crate::backends::BackendKind;
use crate::engine::cache::coordinator::KvSnapshot;
use crate::engine::cache::coordinator::{
    KvBlockIntent, KvCacheCoordinator, KvGroupReservation, KvReserveRequest,
};
use crate::engine::cache::prefix::{
    CoordinatedPrefixIndex, KvPrefixNamespace, KvPrefixPageKey, KvPrefixPublication,
};
use crate::engine::execution::PlanId;
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
    let (keys, blocks, seeds, snapshot) =
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
