use super::*;
use crate::backends::state::{
    PhysicalStateSequenceId, PhysicalStateTransactionId, StateComponentValue,
};
use crate::engine::{
    AdapterAbiRevision, AdapterInstanceId, CapacitySource, ClockedStateSpan,
    ExecutionAdapterBinding, ExecutionGroupId, ExecutionMode, ExecutionProfile, InputRange,
    NativeBatchMode, PhysicalCapacityProvider, PhysicalCapacitySnapshot, SequencePhase,
    StageDescriptor, StageId,
};
use crate::kv::v2::{
    BoundedShape, PageSizeConstraint, ShapeAxis, ShapeDimension, ShapeExtent, StateClock,
    StateComponentId, StateDType, StateDomainHeader, StateDomainId, StateGroupId,
    StateGroupSpec, StateScope, TensorComponentSpec, TensorRole, TensorStateDomainSpec,
};
use crate::kv::{
    CacheBlockRef, InferenceStateCapability as CacheCapability, KvSlotRef, test_contract,
};
use crate::model::ModelVariant;
use crate::models::shared::chat::{ChatMessage, ChatRole};
use candle_core::Tensor;

#[derive(Debug)]
struct TestCapacityProvider {
    snapshot: PhysicalCapacitySnapshot,
}

impl PhysicalCapacityProvider for TestCapacityProvider {
    fn snapshot(&self) -> PhysicalCapacitySnapshot {
        self.snapshot
    }
}

fn authority_with_capacity(bytes: u64) -> Arc<ResourceAuthority> {
    let capacity = ResourceVector {
        host_bytes: ResourceAmount::Known(bytes),
        device_bytes: ResourceAmount::Known(bytes),
        unified_bytes: ResourceAmount::Known(bytes),
        ..ResourceVector::zero()
    };
    Arc::new(ResourceAuthority::new(Arc::new(TestCapacityProvider {
        snapshot: PhysicalCapacitySnapshot {
            capacity,
            available: capacity,
            source: CapacitySource::Test,
        },
    })))
}

fn advisory_authority_with_capacity(bytes: u64) -> Arc<ResourceAuthority> {
    let capacity = ResourceVector {
        host_bytes: ResourceAmount::Known(bytes),
        device_bytes: ResourceAmount::Known(bytes),
        unified_bytes: ResourceAmount::Known(bytes),
        ..ResourceVector::zero()
    };
    Arc::new(ResourceAuthority::new_advisory(Arc::new(
        TestCapacityProvider {
            snapshot: PhysicalCapacitySnapshot {
                capacity,
                available: capacity,
                source: CapacitySource::Test,
            },
        },
    )))
}

#[test]
fn managed_runtime_rejects_a_backend_device_mismatch() {
    let mut manager = ManagedKvCacheManager::for_worker(None, BackendKind::Metal, Device::Cpu);
    let error = manager
        .bind_request(
            ModelInstanceId::new(700),
            BackendKind::Metal,
            2,
            16,
            &CacheCapability::Managed(test_contract()),
        )
        .unwrap_err();
    assert!(error.to_string().contains("cannot bind worker device"));
    assert_eq!(manager.model_count(), 0);
}

#[cfg(feature = "metal")]
#[test]
fn managed_metal_runtime_allocates_on_the_exact_worker_device() -> Result<()> {
    let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
        "managed_metal_runtime_allocates_on_the_exact_worker_device",
    ) else {
        return Ok(());
    };
    let expected_location = device.location();
    let mut manager =
        ManagedKvCacheManager::for_worker(None, BackendKind::Metal, device.clone());
    let runtime = manager
        .bind_request(
            ModelInstanceId::new(701),
            BackendKind::Metal,
            2,
            16,
            &CacheCapability::Managed(test_contract()),
        )?
        .expect("managed contract should activate on compiled Metal");
    assert_eq!(runtime.plan().backend, BackendKind::Metal);
    assert_eq!(
        runtime.plan().device_ordinal,
        managed_device_ordinal(&device)
    );
    for group in &runtime.plan().groups {
        let arena = runtime.arena(group.arena).expect("resolved arena");
        assert_eq!(arena.backend_kind(), BackendKind::Metal);
        assert_eq!(arena.device_location(), expected_location);
    }
    Ok(())
}

fn sequence_work(start: usize, end: usize) -> WorkUnit {
    WorkUnit::SequenceStep {
        phase: SequencePhase::Prefill,
        input: InputRange { start, end },
        max_output_steps: end.saturating_sub(start).max(1),
        auxiliary_state: None,
    }
}

fn selected_sequence_work(
    start: usize,
    end: usize,
    group: StateGroupId,
    clock: StateClock,
    state_start: usize,
    state_end: usize,
) -> WorkUnit {
    WorkUnit::SequenceStep {
        phase: SequencePhase::Prefill,
        input: InputRange { start, end },
        max_output_steps: end.saturating_sub(start).max(1),
        auxiliary_state: Some(
            vec![
                ClockedStateSpan::new(
                    group,
                    clock,
                    InputRange {
                        start: state_start,
                        end: state_end,
                    },
                )
                .unwrap(),
            ]
            .into(),
        ),
    }
}

fn sliding_contract(window_tokens: u32) -> InferenceStateContract {
    let mut contract = test_contract();
    let StateDomainSpec::PagedAttention(domain) = &mut contract.domains[0] else {
        unreachable!()
    };
    for layer in &mut domain.layers {
        layer.pattern = AttentionPattern::SlidingWindow { window_tokens };
    }
    domain.header.prefix = PrefixPolicy::Disabled;
    domain.header.checkpoint = crate::kv::v2::CheckpointPolicy::Transactional;
    contract.groups[0].prefix_shareable = false;
    contract
}

fn mixed_attention_contract(window_tokens: u32) -> InferenceStateContract {
    let mut contract = test_contract();
    let StateDomainSpec::PagedAttention(domain) = &mut contract.domains[0] else {
        unreachable!()
    };
    domain.layers[0].pattern = AttentionPattern::SlidingWindow { window_tokens };
    let mut global = domain.layers[0].clone();
    global.model_layer = 1;
    global.pattern = AttentionPattern::Full;
    domain.layers.push(global);
    contract
}

#[test]
fn mixed_local_and_global_layers_keep_one_full_physical_table() {
    let contract = mixed_attention_contract(32);
    assert_eq!(
        sliding_window_for_domain(&contract, CacheDomainId::new(1)).unwrap(),
        None
    );
    assert!(prefix_enabled_for_domain(&contract, CacheDomainId::new(1)));
    let mut manager = ManagedKvCacheManager::default();
    assert!(
        manager
            .bind_request(
                ModelInstanceId::new(702),
                BackendKind::Cpu,
                4,
                16,
                &CacheCapability::Managed(contract),
            )
            .unwrap()
            .is_some()
    );
}

#[test]
fn homogeneous_sliding_contract_is_admitted_by_every_managed_backend() {
    let contract = sliding_contract(31);
    for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
        validate_sliding_contract(&contract, backend).unwrap_or_else(|error| {
            panic!("{backend:?} should admit managed sliding-window KV: {error}")
        });
    }
}

#[test]
fn sliding_contract_admission_still_rejects_an_empty_paged_domain() {
    let mut contract = sliding_contract(31);
    let StateDomainSpec::PagedAttention(domain) = &mut contract.domains[0] else {
        unreachable!()
    };
    domain.layers.clear();

    let error = validate_sliding_contract(&contract, BackendKind::Cuda)
        .expect_err("an empty paged-attention domain must remain invalid");
    assert!(
        error
            .to_string()
            .contains("managed paged-attention domain has no layers")
    );
}

#[test]
fn prefix_index_capacity_is_independent_from_active_arena_capacity() {
    let model = ModelInstanceId::new(703);
    let mut manager = ManagedKvCacheManager::with_prefix_cache_policy(None, Some([3; 32]), 2);
    manager
        .bind_request(
            model,
            BackendKind::Cpu,
            8,
            16,
            &CacheCapability::Managed(test_contract()),
        )
        .unwrap()
        .expect("managed runtime");

    let state = manager.models.get(&model).expect("registered model");
    assert!(
        state
            .prefix_indexes
            .values()
            .all(|index| index.capacity_pages() == 2)
    );
    assert!(
        state
            .coordinators
            .values()
            .all(|coordinator| coordinator.stats().capacity_pages == 8)
    );
    let snapshot = manager.runtime_snapshot();
    assert_eq!(snapshot.models[0].arenas[0].prefix_capacity_pages, 2);
}

/// An unset prefix page budget takes a quarter of a CUDA arena that is
/// allocated in full at load, never below the default; everywhere else,
/// and whenever the operator set a budget (zero included), the resolved
/// budget applies as is. No budget exceeds its arena.
#[test]
fn automatic_prefix_budget_takes_a_quarter_of_a_cuda_arena_allocated_at_load() {
    use crate::config::DEFAULT_MAX_PREFIX_CACHE_PAGES as DEFAULT;
    use BackendKind::{Cpu, Cuda, Metal};

    // Qwen3.8 on an H100: 9,769 pages keep 2,442 (156,288 tokens).
    assert_eq!(prefix_index_pages(DEFAULT, true, Cuda, true, 9_769), 2_442);
    assert_eq!(prefix_index_pages(DEFAULT, true, Cuda, true, 400), DEFAULT);
    assert_eq!(prefix_index_pages(DEFAULT, true, Cuda, true, 100), 100);
    // A CUDA arena that grows on demand, and CPU and Metal arenas.
    assert_eq!(
        prefix_index_pages(DEFAULT, true, Cuda, false, 9_769),
        DEFAULT
    );
    for backend in [Cpu, Metal] {
        for materialized in [false, true] {
            assert_eq!(
                prefix_index_pages(DEFAULT, true, backend, materialized, 9_769),
                DEFAULT,
                "{backend:?}"
            );
        }
    }
    for backend in [Cpu, Metal, Cuda] {
        for pages in [0, 64, 4_096] {
            assert_eq!(
                prefix_index_pages(pages, false, backend, true, 9_769),
                pages,
                "{backend:?}"
            );
        }
        assert_eq!(
            prefix_index_pages(4_096, false, backend, true, 1_000),
            1_000
        );
        // Disabled reuse stays disabled.
        assert_eq!(prefix_index_pages(0, true, backend, true, 9_769), 0);
    }

    // CPU binding keeps the resolved budget when it is automatic.
    let model = ModelInstanceId::new(705);
    let mut manager = ManagedKvCacheManager::with_prefix_cache_policy(None, Some([4; 32]), 2);
    manager.set_automatic_prefix_cache_pages(true);
    manager
        .bind_request(
            model,
            BackendKind::Cpu,
            16,
            16,
            &CacheCapability::Managed(test_contract()),
        )
        .unwrap()
        .expect("managed runtime");
    let snapshot = manager.runtime_snapshot();
    assert_eq!(snapshot.models[0].arenas[0].prefix_capacity_pages, 2);
}

/// Pages held only by the prefix index never hold a request back: when
/// the arena has no free page, admission evicts the least recently used
/// history and the request proceeds, while newer history stays reusable.
#[test]
fn admission_reclaims_pages_held_only_by_the_prefix_index() {
    fn run(
        manager: &mut ManagedKvCacheManager,
        runtime: &ManagedKvModelRuntime,
        txn_id: PlanId,
        name: &str,
        tokens: Vec<u32>,
    ) -> ManagedCacheReservation {
        let request = prefix_request(ModelInstanceId::new(706), tokens.clone());
        let session = SessionKey::new(name.into(), 1);
        let reservation = manager
            .prepare(
                runtime,
                txn_id,
                &session,
                &sequence_work(0, tokens.len()),
                Some(&request),
            )
            .unwrap_or_else(|error| panic!("{name} was not admitted: {error}"))
            .expect("managed reservation");
        manager
            .finalize(
                &reservation,
                Some(&reservation.completed_write_receipt_for_test()),
                true,
            )
            .expect("commit");
        manager.release_session(&session).expect("release");
        reservation
    }

    let mut manager = ManagedKvCacheManager::with_prefix_cache_salt(None, Some([12; 32]));
    let runtime = manager
        .bind_request(
            ModelInstanceId::new(706),
            BackendKind::Cpu,
            4,
            32,
            &CacheCapability::Managed(test_contract()),
        )
        .unwrap()
        .expect("managed runtime");
    // Two finished conversations: the index alone holds 3 of 4 pages.
    let older = (0..64).collect::<Vec<u32>>();
    let newer = (500..532).collect::<Vec<u32>>();
    run(&mut manager, &runtime, 41, "older", older);
    run(&mut manager, &runtime, 42, "newer", newer.clone());
    let retained = manager.runtime_snapshot();
    assert_eq!(retained.counters.prefix_retained_pages, 3);
    assert_eq!(retained.totals.coordinator.free_pages, 1);

    // An unrelated prompt needs two pages; only one is free.
    run(
        &mut manager,
        &runtime,
        43,
        "unrelated",
        (900..964).collect(),
    );
    assert_eq!(manager.telemetry_snapshot().prefix_evictions, 2);

    // The newer conversation's history survived and still attaches.
    let mut next_turn = newer;
    next_turn.push(7);
    let resumed = run(&mut manager, &runtime, 44, "newer-next", next_turn);
    assert_eq!(resumed.domains[0].execution_start_tokens, 32);
}

#[test]
fn prefix_telemetry_counts_each_prepare_once_and_never_the_probe() {
    let model = ModelInstanceId::new(707);
    let mut manager = ManagedKvCacheManager::with_prefix_cache_salt(None, Some([13; 32]));
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            4,
            32,
            &CacheCapability::Managed(test_contract()),
        )
        .unwrap()
        .expect("managed runtime");
    let history = (0..64).collect::<Vec<u32>>();
    let first = SessionKey::new("history".into(), 1);
    let reservation = manager
        .prepare(
            &runtime,
            51,
            &first,
            &sequence_work(0, history.len()),
            Some(&prefix_request(model, history.clone())),
        )
        .unwrap()
        .expect("managed reservation");
    manager
        .finalize(
            &reservation,
            Some(&reservation.completed_write_receipt_for_test()),
            true,
        )
        .unwrap();
    manager.release_session(&first).unwrap();
    let cold = manager.telemetry_snapshot();
    assert_eq!((cold.prefix_hits, cold.prefix_misses), (0, 1));
    assert_eq!(cold.prefix_query_tokens, 63);

    let mut next_turn = history;
    next_turn.push(7);
    let request = prefix_request(model, next_turn.clone());
    assert_eq!(
        manager.probe_managed_prefix(&runtime, &request).unwrap(),
        Some(64)
    );
    assert_eq!(
        manager.telemetry_snapshot(),
        cold,
        "admission probes are not lookups"
    );

    let second = SessionKey::new("next-turn".into(), 1);
    manager
        .prepare(
            &runtime,
            52,
            &second,
            &sequence_work(0, next_turn.len()),
            Some(&request),
        )
        .unwrap()
        .expect("managed reservation");
    let warm = manager.telemetry_snapshot();
    assert_eq!((warm.prefix_hits, warm.prefix_misses), (1, 1));
    assert_eq!(warm.prefix_query_tokens, 63 + 64);
    assert_eq!(warm.prefix_hit_tokens, 64);
}

#[test]
fn schedule_time_probe_offers_a_cursor_only_where_prepare_attaches_it() {
    // SCH-F7: a session that committed tokens, or restarted into a
    // subordinate generation, must rebuild from context 0; offering it
    // a cursor would only be lost again at prepare.
    let model = ModelInstanceId::new(708);
    let mut manager = ManagedKvCacheManager::with_prefix_cache_salt(None, Some([13; 32]));
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            8,
            32,
            &CacheCapability::Managed(test_contract()),
        )
        .unwrap()
        .expect("managed runtime");
    let history = (0..64).collect::<Vec<u32>>();
    let publisher = SessionKey::new("publisher".into(), 1);
    let reservation = manager
        .prepare(
            &runtime,
            61,
            &publisher,
            &sequence_work(0, history.len()),
            Some(&prefix_request(model, history.clone())),
        )
        .unwrap()
        .expect("managed reservation");
    manager
        .finalize(
            &reservation,
            Some(&reservation.completed_write_receipt_for_test()),
            true,
        )
        .unwrap();
    manager.release_session(&publisher).unwrap();

    let mut next_turn = history;
    next_turn.push(7);
    let request = prefix_request(model, next_turn);
    let fresh = SessionKey::new("fresh".into(), 1);
    assert_eq!(
        manager
            .probe_managed_prefix_for_session(&runtime, &request, &fresh)
            .unwrap(),
        Some(64)
    );

    let restarted = SessionKey::new("restarted".into(), 1);
    let first = manager
        .prepare(
            &runtime,
            62,
            &restarted,
            &sequence_work(0, 5),
            Some(&request),
        )
        .unwrap()
        .expect("managed reservation");
    manager
        .finalize(
            &first,
            Some(&first.completed_write_receipt_for_test()),
            true,
        )
        .unwrap();
    assert_eq!(
        manager
            .probe_managed_prefix_for_session(&runtime, &request, &restarted)
            .unwrap(),
        None,
        "a session with committed tokens"
    );
    manager
        .reset_session_generation(&runtime, &restarted, ManagedSessionGeneration::INITIAL)
        .unwrap();
    assert_eq!(
        manager
            .probe_managed_prefix_for_session(&runtime, &request, &restarted)
            .unwrap(),
        None,
        "a subordinate session generation"
    );
}

#[test]
fn admission_skips_prefix_chains_whose_pages_are_still_owned() {
    // C1: evicting a chain whose pages a live table still owns frees no
    // memory. Admission used to drop such chains first (least recently
    // used) and keep going, so under load it flushed the index without
    // gaining a page. Only chains that free at least one page are evicted.
    let model = ModelInstanceId::new(707);
    let mut manager = ManagedKvCacheManager::with_prefix_cache_salt(None, Some([13; 32]));
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            6,
            32,
            &CacheCapability::Managed(test_contract()),
        )
        .unwrap()
        .expect("managed runtime");
    let commit = |manager: &mut ManagedKvCacheManager,
                  txn_id: PlanId,
                  session: &SessionKey,
                  tokens: Vec<u32>| {
        let request = prefix_request(model, tokens.clone());
        let reservation = manager
            .prepare(
                &runtime,
                txn_id,
                session,
                &sequence_work(0, tokens.len()),
                Some(&request),
            )
            .expect("admitted")
            .expect("managed reservation");
        manager
            .finalize(
                &reservation,
                Some(&reservation.completed_write_receipt_for_test()),
                true,
            )
            .expect("commit");
    };

    // Least recently used: a live session whose published page it still owns.
    let live = SessionKey::new("live".into(), 1);
    commit(&mut manager, 51, &live, (0..33).collect());
    // Newer: a finished conversation held only by the index (two pages).
    let finished = SessionKey::new("finished".into(), 1);
    commit(&mut manager, 52, &finished, (500..565).collect());
    manager.release_session(&finished).expect("release");
    let before = manager.runtime_snapshot();
    assert_eq!(before.counters.prefix_retained_pages, 3);

    // An unrelated prompt needs four pages while three are free.
    let unrelated = SessionKey::new("unrelated".into(), 1);
    commit(&mut manager, 53, &unrelated, (900..1000).collect());

    // Only the index-only chain went; the live session's page is still
    // published for reuse.
    assert_eq!(manager.telemetry_snapshot().prefix_evictions, 2);
    let after = manager.runtime_snapshot();
    assert!(after.counters.prefix_retained_pages >= 1);
}

fn composite_tensor_contract() -> InferenceStateContract {
    let mut contract = test_contract();
    let domain = CacheDomainId::new(2);
    contract
        .domains
        .push(StateDomainSpec::Tensor(TensorStateDomainSpec {
            header: StateDomainHeader {
                id: domain,
                scope: StateScope::Retained,
                clock: StateClock::DecoderTokens,
                placement: crate::kv::v2::PlacementPolicy::BackendLocalWithHostOffload,
                prefix: PrefixPolicy::Disabled,
                checkpoint: crate::kv::v2::CheckpointPolicy::Transactional,
            },
            components: vec![TensorComponentSpec {
                id: StateComponentId::new(1),
                role: TensorRole::RecurrentHidden,
                shape: BoundedShape {
                    dimensions: vec![ShapeDimension {
                        axis: ShapeAxis::Hidden,
                        extent: ShapeExtent::Fixed { value: 4 },
                    }],
                },
                accepted_dtypes: vec![KvStorageDType::F32],
            }],
        }));
    contract.groups = vec![StateGroupSpec {
        id: crate::kv::v2::StateGroupId::new(1),
        domains: vec![CacheDomainId::new(1), domain],
        prefix_shareable: false,
    }];
    contract.validate().unwrap();
    contract
}

fn two_paged_tensor_contract() -> InferenceStateContract {
    let mut contract = composite_tensor_contract();
    let mut second_paged = contract.domains[0].clone();
    if let StateDomainSpec::PagedAttention(domain) = &mut second_paged {
        domain.header.id = CacheDomainId::new(3);
    }
    contract.domains.push(second_paged);
    contract.groups.push(StateGroupSpec {
        id: crate::kv::v2::StateGroupId::new(2),
        domains: vec![CacheDomainId::new(3)],
        prefix_shareable: false,
    });
    contract.validate().unwrap();
    contract
}

fn hybrid_snapshot_contract() -> InferenceStateContract {
    let mut contract = test_contract();
    let domain = CacheDomainId::new(2);
    contract
        .domains
        .push(StateDomainSpec::Tensor(TensorStateDomainSpec {
            header: StateDomainHeader {
                id: domain,
                scope: StateScope::Retained,
                clock: StateClock::DecoderTokens,
                placement: crate::kv::v2::PlacementPolicy::BackendLocalWithHostOffload,
                prefix: PrefixPolicy::CommittedSnapshots { interval_steps: 32 },
                checkpoint: crate::kv::v2::CheckpointPolicy::Transactional,
            },
            components: vec![TensorComponentSpec {
                id: StateComponentId::new(1),
                role: TensorRole::RecurrentHidden,
                shape: BoundedShape {
                    dimensions: vec![ShapeDimension {
                        axis: ShapeAxis::Hidden,
                        extent: ShapeExtent::Fixed { value: 4 },
                    }],
                },
                accepted_dtypes: vec![KvStorageDType::F32],
            }],
        }));
    contract.groups = vec![StateGroupSpec {
        id: crate::kv::v2::StateGroupId::new(1),
        domains: vec![CacheDomainId::new(1), domain],
        prefix_shareable: true,
    }];
    contract.validate().unwrap();
    contract
}

fn independently_clocked_tensor_contract() -> InferenceStateContract {
    let mut contract = composite_tensor_contract();
    contract.groups = vec![
        StateGroupSpec {
            id: StateGroupId::new(1),
            domains: vec![StateDomainId::new(1)],
            prefix_shareable: false,
        },
        StateGroupSpec {
            id: StateGroupId::new(2),
            domains: vec![StateDomainId::new(2)],
            prefix_shareable: false,
        },
    ];
    let StateDomainSpec::Tensor(tensor) = &mut contract.domains[1] else {
        panic!("composite test contract tensor domain changed kind");
    };
    tensor.header.clock = StateClock::AudioFrames;
    contract.validate().unwrap();
    contract
}

fn two_paged_contract() -> InferenceStateContract {
    let mut contract = test_contract();
    let mut second_paged = contract.domains[0].clone();
    if let StateDomainSpec::PagedAttention(domain) = &mut second_paged {
        domain.header.id = CacheDomainId::new(3);
    }
    contract.domains.push(second_paged);
    contract.groups.push(StateGroupSpec {
        id: crate::kv::v2::StateGroupId::new(2),
        domains: vec![CacheDomainId::new(3)],
        prefix_shareable: false,
    });
    contract.validate().unwrap();
    contract
}

fn qwen35_9b_tensor_contract() -> InferenceStateContract {
    let mut contract = test_contract();
    let recurrent_domain = CacheDomainId::new(2);
    let convolution_domain = CacheDomainId::new(3);
    let tensor_domain = |domain: CacheDomainId, role: TensorRole, elements: u64| {
        StateDomainSpec::Tensor(TensorStateDomainSpec {
            header: StateDomainHeader {
                id: domain,
                scope: StateScope::Retained,
                clock: StateClock::DecoderTokens,
                placement: crate::kv::v2::PlacementPolicy::BackendLocalWithHostOffload,
                prefix: PrefixPolicy::Disabled,
                checkpoint: crate::kv::v2::CheckpointPolicy::Transactional,
            },
            // Qwen3.5 9B has 24 linear-attention layers. Each recurrent
            // layer owns 128 * 4,096 F32 cells and a 24,576-cell
            // convolution history.
            components: (1..=24)
                .map(|id| TensorComponentSpec {
                    id: StateComponentId::new(id),
                    role: role.clone(),
                    shape: BoundedShape {
                        dimensions: vec![ShapeDimension {
                            axis: ShapeAxis::Hidden,
                            extent: ShapeExtent::Fixed { value: elements },
                        }],
                    },
                    accepted_dtypes: vec![KvStorageDType::F32],
                })
                .collect(),
        })
    };
    contract.domains.push(tensor_domain(
        recurrent_domain,
        TensorRole::RecurrentHidden,
        524_288,
    ));
    contract.domains.push(tensor_domain(
        convolution_domain,
        TensorRole::ConvolutionState,
        24_576,
    ));
    contract.groups = vec![StateGroupSpec {
        id: crate::kv::v2::StateGroupId::new(1),
        domains: vec![CacheDomainId::new(1), recurrent_domain, convolution_domain],
        prefix_shareable: false,
    }];
    contract.validate().unwrap();
    contract
}

fn qwen38_27b_tensor_contract() -> InferenceStateContract {
    let mut contract = test_contract();
    let StateDomainSpec::PagedAttention(attention) = &mut contract.domains[0] else {
        unreachable!("test contract is paged")
    };
    let layer = attention.layers[0].clone();
    attention.layers = (0..16)
        .map(|attention_layer| crate::kv::v2::PagedAttentionLayerSpec {
            model_layer: attention_layer * 4 + 3,
            query_heads: 24,
            kv_heads: 4,
            key_head_dim: 256,
            value_head_dim: 256,
            ..layer.clone()
        })
        .collect();
    attention.header.prefix = PrefixPolicy::Disabled;
    attention.header.checkpoint = crate::kv::v2::CheckpointPolicy::Transactional;
    attention.accepted_dtypes = vec![StateDType::F16];

    let recurrent_domain = CacheDomainId::new(2);
    let convolution_domain = CacheDomainId::new(3);
    let tensor_domain = |domain: CacheDomainId, role: TensorRole, elements: u64| {
        StateDomainSpec::Tensor(TensorStateDomainSpec {
            header: StateDomainHeader {
                id: domain,
                scope: StateScope::Retained,
                clock: StateClock::DecoderTokens,
                placement: crate::kv::v2::PlacementPolicy::BackendLocalWithHostOffload,
                prefix: PrefixPolicy::Disabled,
                checkpoint: crate::kv::v2::CheckpointPolicy::Transactional,
            },
            components: (1..=48)
                .map(|id| TensorComponentSpec {
                    id: StateComponentId::new(id),
                    role: role.clone(),
                    shape: BoundedShape {
                        dimensions: vec![ShapeDimension {
                            axis: ShapeAxis::Hidden,
                            extent: ShapeExtent::Fixed { value: elements },
                        }],
                    },
                    accepted_dtypes: vec![StateDType::F32],
                })
                .collect(),
        })
    };
    // Each of the 48 DeltaNet layers owns a 128 x 128 x 48 F32
    // recurrence and 10,240 x 3 F32 convolution history.
    contract.domains.push(tensor_domain(
        recurrent_domain,
        TensorRole::RecurrentHidden,
        786_432,
    ));
    contract.domains.push(tensor_domain(
        convolution_domain,
        TensorRole::ConvolutionState,
        30_720,
    ));
    contract.groups = vec![StateGroupSpec {
        id: StateGroupId::new(1),
        domains: vec![CacheDomainId::new(1), recurrent_domain, convolution_domain],
        prefix_shareable: false,
    }];
    contract.validate().unwrap();
    contract
}

fn heterogeneous_paged_contract() -> InferenceStateContract {
    let mut contract = test_contract();
    let StateDomainSpec::PagedAttention(first) = &contract.domains[0] else {
        unreachable!()
    };
    let mut second = first.clone();
    second.header.id = CacheDomainId::new(2);
    second.page_size = PageSizeConstraint {
        min_tokens: 32,
        preferred_tokens: 32,
        max_tokens: 32,
        multiple_of: 32,
    };
    contract
        .domains
        .push(StateDomainSpec::PagedAttention(second));
    contract.groups.push(StateGroupSpec {
        id: crate::kv::v2::StateGroupId::new(2),
        domains: vec![CacheDomainId::new(2)],
        prefix_shareable: true,
    });
    contract.validate().unwrap();
    contract
}

#[test]
fn capacity_planner_gives_heterogeneous_groups_equal_token_reach() {
    let state_plan = negotiate_state_plan(
        &heterogeneous_paged_contract(),
        &StateBackendPlanRequest {
            backend: BackendKind::Cpu,
            device_ordinal: None,
            page_tokens_hint: Some(16),
            storage_dtype_hint: None,
        },
    )
    .unwrap();
    let (allocation, tensor) = plan_managed_state_capacity(
        &state_plan,
        ModelInstanceId::new(800),
        ManagedStateCapacityRequest {
            total_paged_pages: 7,
            logical_token_reach: None,
            retained_sequence_rows: 3,
            staged_transaction_rows: 3,
        },
    )
    .unwrap();
    assert!(tensor.is_none());
    let capacities = state_plan
        .paged_attention
        .iter()
        .map(|group| {
            let capacity = allocation
                .group_capacity(group.group, group.domain)
                .unwrap();
            (group.page_tokens, capacity.strategy.maximum_blocks())
        })
        .collect::<Vec<_>>();
    assert_eq!(capacities, vec![(16, 4), (32, 2)]);
    assert_eq!(capacities.iter().map(|(_, pages)| pages).sum::<u32>(), 6);
    assert_eq!(capacities[0].0 * capacities[0].1, 64);
    assert_eq!(capacities[1].0 * capacities[1].1, 64);
    assert_eq!(
        allocation.hard_limit,
        allocation.maximum_resources(&state_plan).unwrap()
    );
}

#[test]
fn capacity_planner_rejects_a_budget_that_cannot_back_every_group() {
    let state_plan = negotiate_state_plan(
        &heterogeneous_paged_contract(),
        &StateBackendPlanRequest {
            backend: BackendKind::Cpu,
            device_ordinal: None,
            page_tokens_hint: Some(16),
            storage_dtype_hint: None,
        },
    )
    .unwrap();
    assert!(
        plan_managed_state_capacity(
            &state_plan,
            ModelInstanceId::new(801),
            ManagedStateCapacityRequest {
                total_paged_pages: 1,
                logical_token_reach: None,
                retained_sequence_rows: 1,
                staged_transaction_rows: 1,
            },
        )
        .is_err()
    );
}

#[test]
fn capacity_planner_grows_cuda_paged_state_but_keeps_cpu_fixed() {
    for (backend, expected_initial) in
        [(BackendKind::Cuda, 64_u32), (BackendKind::Cpu, 256_u32)]
    {
        let strategy = managed_paged_capacity_strategy(backend, 256, false);
        assert_eq!(strategy.initial_blocks(), expected_initial);
        assert_eq!(strategy.maximum_blocks(), 256);
        assert_eq!(
            matches!(strategy, CapacityStrategy::AdmissionGrowable { .. }),
            backend == BackendKind::Cuda
        );
    }
    assert_eq!(
        managed_paged_capacity_strategy(BackendKind::Cuda, 256, true),
        CapacityStrategy::Fixed { blocks: 256 }
    );
}

#[test]
fn loaded_model_context_resolves_exact_paged_capacity_without_a_cuda_device() {
    let state_plan = negotiate_state_plan(
        &crate::kv::test_contract(),
        &StateBackendPlanRequest {
            backend: BackendKind::Cpu,
            device_ordinal: None,
            page_tokens_hint: Some(64),
            storage_dtype_hint: None,
        },
    )
    .unwrap();

    for (model_context, expected_pages) in [
        (32_768_u64, 512_u32),
        (40_960, 640),
        (131_072, 2_048),
        (262_144, 4_096),
    ] {
        let (allocation, _) = plan_managed_state_capacity(
            &state_plan,
            ModelInstanceId::new(model_context),
            ManagedStateCapacityRequest {
                // The model-derived reach is authoritative in this mode;
                // this legacy aggregate value must not inflate Qwen3 to
                // the largest catalog context.
                total_paged_pages: 4_096,
                logical_token_reach: Some(model_context),
                retained_sequence_rows: 8,
                staged_transaction_rows: 8,
            },
        )
        .unwrap();
        let group = &state_plan.paged_attention[0];
        let capacity = allocation
            .group_capacity(group.group, group.domain)
            .unwrap();
        assert_eq!(capacity.strategy.maximum_blocks(), expected_pages);
    }
}

#[test]
fn portable_auto_fit_selects_the_largest_exact_page_reach() {
    const RESERVE: u64 = 1024 * 1024 * 1024;
    let contract = test_contract();
    let state_plan = negotiate_state_plan(
        &contract,
        &StateBackendPlanRequest {
            backend: BackendKind::Cpu,
            device_ordinal: None,
            page_tokens_hint: Some(64),
            storage_dtype_hint: None,
        },
    )
    .unwrap();
    let (allocation, _) = plan_managed_state_capacity(
        &state_plan,
        ModelInstanceId::new(900),
        ManagedStateCapacityRequest {
            total_paged_pages: u32::MAX,
            logical_token_reach: Some(4_096),
            retained_sequence_rows: 8,
            staged_transaction_rows: 8,
        },
    )
    .unwrap();
    let exact = managed_state_resources(
        BackendKind::Cpu,
        allocation.maximum_resources(&state_plan).unwrap(),
    )
    .unwrap();
    let ResourceAmount::Known(exact_bytes) = exact.host_bytes else {
        panic!("test plan must have known host bytes");
    };
    let manager = ManagedKvCacheManager::for_worker(
        Some(advisory_authority_with_capacity(RESERVE + exact_bytes)),
        BackendKind::Cpu,
        Device::Cpu,
    );
    assert_eq!(
        manager
            .fit_portable_logical_token_reach(
                ModelInstanceId::new(900),
                &contract,
                40_960,
                RESERVE,
                64,
                8,
                8,
                1,
                PortableFitHeadroom::Planning { minimum_tokens: 0 },
            )
            .unwrap(),
        4_096
    );

    let duplicated = ManagedKvCacheManager::for_worker(
        Some(advisory_authority_with_capacity(RESERVE + exact_bytes)),
        BackendKind::Cpu,
        Device::Cpu,
    );
    let selected = duplicated
        .fit_portable_logical_token_reach(
            ModelInstanceId::new(901),
            &contract,
            40_960,
            RESERVE,
            64,
            8,
            8,
            2,
            PortableFitHeadroom::Planning { minimum_tokens: 0 },
        )
        .unwrap();
    assert!(selected < 4_096);
    assert!(selected >= 64);

    // With a chat floor the same squeeze is a capacity refusal (503),
    // not a published sub-floor context; a floor that fits is honoured.
    let floored = |minimum_tokens| {
        duplicated.fit_portable_logical_token_reach(
            ModelInstanceId::new(902),
            &contract,
            40_960,
            RESERVE,
            64,
            8,
            8,
            2,
            PortableFitHeadroom::Planning { minimum_tokens },
        )
    };
    assert!(matches!(floored(4_096), Err(Error::Overloaded(_))));
    let reach = floored(1_024).unwrap();
    assert!((1_024..4_096).contains(&reach), "{reach}");
}

/// ASR state fits spend live memory, not only ledger headroom: with the
/// ledger empty but most of the machine in use elsewhere, the live fit
/// shrinks while the planning fit still takes the whole ceiling.
#[test]
fn live_portable_fit_shrinks_to_live_headroom_below_planning() {
    const RESERVE: u64 = 64 * 1024 * 1024;
    let contract = test_contract();
    let bytes = |tokens: u64| {
        let state_plan = negotiate_state_plan(
            &contract,
            &StateBackendPlanRequest {
                backend: BackendKind::Cpu,
                device_ordinal: None,
                page_tokens_hint: Some(64),
                storage_dtype_hint: None,
            },
        )
        .unwrap();
        let (allocation, _) = plan_managed_state_capacity(
            &state_plan,
            ModelInstanceId::new(902),
            ManagedStateCapacityRequest {
                total_paged_pages: u32::MAX,
                logical_token_reach: Some(tokens),
                retained_sequence_rows: 8,
                staged_transaction_rows: 8,
            },
        )
        .unwrap();
        let resources = managed_state_resources(
            BackendKind::Cpu,
            allocation.maximum_resources(&state_plan).unwrap(),
        )
        .unwrap();
        let ResourceAmount::Known(bytes) = resources.host_bytes else {
            panic!("test plan must have known host bytes");
        };
        bytes
    };
    // The ledger admits 4,096 tokens; live memory only 1,024.
    let capacity = host_bytes_vector(RESERVE + bytes(4_096));
    let available = host_bytes_vector(RESERVE + bytes(1_024));
    let manager = ManagedKvCacheManager::for_worker(
        Some(Arc::new(ResourceAuthority::new_advisory(Arc::new(
            TestCapacityProvider {
                snapshot: PhysicalCapacitySnapshot {
                    capacity,
                    available,
                    source: CapacitySource::Test,
                },
            },
        )))),
        BackendKind::Cpu,
        Device::Cpu,
    );
    let fit = |headroom| {
        manager
            .fit_portable_logical_token_reach(
                ModelInstanceId::new(902),
                &contract,
                4_096,
                RESERVE,
                64,
                8,
                8,
                1,
                headroom,
            )
            .unwrap()
    };
    assert_eq!(fit(PortableFitHeadroom::Planning { minimum_tokens: 0 }), 4_096);
    // S3: pre-load admission prices a floor exactly as the fit does, the
    // page minimum included.
    assert_eq!(
        manager
            .portable_state_bytes(&contract, 1_307, 64, (8, 8), 1)
            .unwrap(),
        Some(bytes(1_307))
    );
    assert_eq!(
        manager
            .portable_state_bytes(&contract, 0, 64, (8, 8), 1)
            .unwrap(),
        Some(bytes(64))
    );
    let live = fit(PortableFitHeadroom::Live {
        minimum_tokens: 0,
        reserve: StateFitReserve::MeasuredUntracked,
        later_minimum_bytes: 0,
    });
    assert!((1_024..1_024 + 64).contains(&live), "live fit {live}");

    // Right after a large load live memory can read below the reserve
    // (Granite on Metal: 0.8 GB live beside a 1 GiB reserve). The
    // minimum is admitted against planning headroom, so the fit returns
    // it instead of failing the load; only growth is capped by live.
    let fit_with = |capacity: u64, available: u64, floor: u64| {
        ManagedKvCacheManager::for_worker(
            Some(Arc::new(ResourceAuthority::new_advisory(Arc::new(
                TestCapacityProvider {
                    snapshot: PhysicalCapacitySnapshot {
                        capacity: host_bytes_vector(capacity),
                        available: host_bytes_vector(available),
                        source: CapacitySource::Test,
                    },
                },
            )))),
            BackendKind::Cpu,
            Device::Cpu,
        )
        .fit_portable_logical_token_reach(
            ModelInstanceId::new(902),
            &contract,
            4_096,
            RESERVE,
            64,
            8,
            8,
            1,
            PortableFitHeadroom::Live {
                minimum_tokens: floor,
                reserve: StateFitReserve::MeasuredUntracked,
                later_minimum_bytes: 0,
            },
        )
    };
    assert_eq!(
        fit_with(RESERVE + bytes(4_096), RESERVE / 2, 0).unwrap(),
        64
    );
    // A minimum that planning headroom cannot hold is a capacity refusal.
    assert!(matches!(
        fit_with(RESERVE + bytes(64) - 1, RESERVE + bytes(64) - 1, 0),
        Err(Error::Overloaded(_))
    ));
    // An ASR floor of one chunk per sequence (Qwen3-ASR's 1,307 tokens):
    // with live memory below the reserve the fit still reaches the floor
    // (never a context smaller than one chunk), or refuses with
    // Overloaded when planning headroom cannot hold it.
    let reach = fit_with(RESERVE + bytes(4_096), RESERVE / 2, 1_307).unwrap();
    assert!((1_307..1_307 + 64).contains(&reach), "floor fit {reach}");
    assert!(matches!(
        fit_with(RESERVE + bytes(1_024), RESERVE / 2, 1_307),
        Err(Error::Overloaded(_))
    ));
    // A floor above the model's own context is capped by it.
    assert_eq!(
        fit_with(RESERVE + bytes(4_096), RESERVE / 2, 1 << 20).unwrap(),
        4_096
    );

    // S2: the minimums of the state groups loaded after this one stay free
    // beside the reserve. Holding back 1,024 tokens' worth caps growth that
    // much lower, and a floor that would leave them no room is refused.
    let fit_holding = |later_minimum_bytes: u64, floor: u64| {
        let total = host_bytes_vector(RESERVE + bytes(4_096));
        ManagedKvCacheManager::for_worker(
            Some(Arc::new(ResourceAuthority::new_advisory(Arc::new(
                TestCapacityProvider {
                    snapshot: PhysicalCapacitySnapshot {
                        capacity: total,
                        available: total,
                        source: CapacitySource::Test,
                    },
                },
            )))),
            BackendKind::Cpu,
            Device::Cpu,
        )
        .fit_portable_logical_token_reach(
            ModelInstanceId::new(903),
            &contract,
            4_096,
            RESERVE,
            64,
            8,
            8,
            1,
            PortableFitHeadroom::Live {
                minimum_tokens: floor,
                reserve: StateFitReserve::MeasuredUntracked,
                later_minimum_bytes,
            },
        )
    };
    assert_eq!(fit_holding(0, 0).unwrap(), 4_096);
    assert_eq!(fit_holding(bytes(1_024), 0).unwrap(), 3_072);
    assert_eq!(fit_holding(bytes(1_024), 3_072).unwrap(), 3_072);
    let refused = fit_holding(bytes(1_024) + 1, 3_072).unwrap_err();
    assert!(matches!(refused, Error::Overloaded(_)), "{refused}");
    assert!(
        refused
            .to_string()
            .contains("the working state loaded after it"),
        "{refused}"
    );
}

fn host_bytes_vector(bytes: u64) -> ResourceVector {
    ResourceVector {
        host_bytes: ResourceAmount::Known(bytes),
        device_bytes: ResourceAmount::Known(bytes),
        unified_bytes: ResourceAmount::Known(bytes),
        ..ResourceVector::zero()
    }
}

#[test]
fn qwen3_four_and_eight_billion_geometry_materializes_576_mib_at_4096() {
    let mut contract = test_contract();
    let StateDomainSpec::PagedAttention(domain) = &mut contract.domains[0] else {
        unreachable!("test contract is paged")
    };
    let layer = domain.layers[0].clone();
    domain.layers = (0..36)
        .map(|model_layer| crate::kv::v2::PagedAttentionLayerSpec {
            model_layer,
            query_heads: 32,
            kv_heads: 8,
            key_head_dim: 128,
            value_head_dim: 128,
            ..layer.clone()
        })
        .collect();
    let state_plan = negotiate_state_plan(
        &contract,
        &StateBackendPlanRequest {
            backend: BackendKind::Cpu,
            device_ordinal: None,
            page_tokens_hint: Some(64),
            storage_dtype_hint: Some(StateDType::F16),
        },
    )
    .unwrap();
    let (allocation, _) = plan_managed_state_capacity(
        &state_plan,
        ModelInstanceId::new(901),
        ManagedStateCapacityRequest {
            total_paged_pages: 1_024,
            logical_token_reach: Some(4_096),
            retained_sequence_rows: 8,
            staged_transaction_rows: 8,
        },
    )
    .unwrap();
    let resources = allocation.maximum_resources(&state_plan).unwrap();
    assert_eq!(resources.host_bytes, 576 * 1024 * 1024);
    let capacity = allocation
        .group_capacity(StateGroupId::new(1), StateDomainId::new(1))
        .unwrap();
    assert_eq!(capacity.strategy.maximum_blocks(), 64);
}

#[test]
fn mixed_runtime_uses_one_allocation_plan_for_paged_and_tensor_capacity() {
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request_with_capacity(
            ModelInstanceId::new(802),
            BackendKind::Cpu,
            ManagedStateCapacityRequest {
                total_paged_pages: 4,
                logical_token_reach: None,
                retained_sequence_rows: 2,
                staged_transaction_rows: 2,
            },
            16,
            &CacheCapability::Managed(composite_tensor_contract()),
        )
        .unwrap()
        .unwrap();
    runtime
        .plan()
        .validate_against_allocation(runtime.state_plan_v2(), runtime.allocation_plan())
        .unwrap();
    let tensor = runtime.tensor_state().unwrap().capacity();
    assert_eq!(tensor.sequence_capacity(), 2);
    assert_eq!(tensor.transaction_capacity(), 2);
    let non_paged = &runtime.state_plan_v2().non_paged[0];
    assert_eq!(
        runtime
            .allocation_plan()
            .group_capacity(non_paged.group(), non_paged.domain())
            .unwrap()
            .strategy,
        CapacityStrategy::BoundedLazy { max_blocks: 4 }
    );
    let snapshot = manager.runtime_snapshot();
    assert_eq!(
        snapshot.totals.resident_paged_bytes,
        runtime.resident_paged_bytes()
    );
    assert_eq!(
        snapshot.totals.authorized_tensor_bytes,
        tensor.authorized_bytes()
    );
    assert_eq!(
        snapshot.totals.physical_bytes,
        snapshot
            .totals
            .resident_paged_bytes
            .saturating_add(snapshot.totals.authorized_tensor_bytes)
    );
}

#[test]
fn logical_reach_sequence_rows_and_transaction_rows_are_independent_axes() {
    let state_plan = negotiate_state_plan(
        &composite_tensor_contract(),
        &StateBackendPlanRequest {
            backend: BackendKind::Cpu,
            device_ordinal: None,
            page_tokens_hint: Some(64),
            storage_dtype_hint: None,
        },
    )
    .unwrap();

    for (logical_tokens, expected_pages) in [(32_768_u64, 512_u32), (262_144_u64, 4_096_u32)] {
        let (allocation, tensor) = plan_managed_state_capacity(
            &state_plan,
            ModelInstanceId::new(logical_tokens),
            ManagedStateCapacityRequest {
                total_paged_pages: expected_pages,
                logical_token_reach: Some(logical_tokens),
                retained_sequence_rows: 8,
                staged_transaction_rows: 3,
            },
        )
        .unwrap();
        let tensor = tensor.expect("composite state has retained tensors");
        assert_eq!(tensor.sequence_capacity(), 8);
        assert_eq!(tensor.transaction_capacity(), 3);
        assert_eq!(tensor.authorized_bytes(), tensor.per_sequence_bytes() * 11);
        assert_eq!(
            allocation
                .group_capacity(
                    state_plan.paged_attention[0].group,
                    state_plan.paged_attention[0].domain,
                )
                .unwrap()
                .strategy
                .maximum_blocks(),
            expected_pages
        );
        let non_paged = &state_plan.non_paged[0];
        assert_eq!(
            allocation
                .group_capacity(non_paged.group(), non_paged.domain())
                .unwrap()
                .strategy,
            CapacityStrategy::BoundedLazy { max_blocks: 11 }
        );
    }
}

#[test]
fn transaction_rows_cannot_exceed_retained_sequence_rows() {
    let state_plan = negotiate_state_plan(
        &composite_tensor_contract(),
        &StateBackendPlanRequest {
            backend: BackendKind::Cpu,
            device_ordinal: None,
            page_tokens_hint: Some(64),
            storage_dtype_hint: None,
        },
    )
    .unwrap();
    let error = plan_managed_state_capacity(
        &state_plan,
        ModelInstanceId::new(805),
        ManagedStateCapacityRequest {
            total_paged_pages: 64,
            logical_token_reach: Some(4_096),
            retained_sequence_rows: 1,
            staged_transaction_rows: 2,
        },
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("transaction rows cannot exceed retained sequence rows")
    );
}

#[test]
fn long_context_pages_do_not_multiply_retained_tensor_sequences() {
    let state_plan = negotiate_state_plan(
        &qwen35_9b_tensor_contract(),
        &StateBackendPlanRequest {
            backend: BackendKind::Cpu,
            device_ordinal: None,
            page_tokens_hint: Some(64),
            storage_dtype_hint: None,
        },
    )
    .unwrap();
    let (allocation, tensor) = plan_managed_state_capacity(
        &state_plan,
        ModelInstanceId::new(803),
        ManagedStateCapacityRequest {
            total_paged_pages: 4_096,
            logical_token_reach: Some(262_144),
            retained_sequence_rows: 16,
            staged_transaction_rows: 16,
        },
    )
    .unwrap();
    let tensor = tensor.expect("composite state has retained tensors");

    assert_eq!(tensor.sequence_capacity(), 16);
    assert_eq!(tensor.transaction_capacity(), 16);
    assert_eq!(tensor.authorized_bytes(), tensor.per_sequence_bytes() * 32);
    assert_eq!(tensor.per_sequence_bytes(), 52_690_944);
    let initial_paged_bytes = 128 * 1024 * 1024;
    let legacy_claim = tensor
        .per_sequence_bytes()
        .checked_mul(4_096 + 16)
        .unwrap()
        .checked_add(initial_paged_bytes)
        .unwrap();
    assert_eq!(legacy_claim, 216_799_379_456);
    assert_eq!(
        tensor.authorized_bytes() + initial_paged_bytes,
        1_820_327_936
    );
    let non_paged = &state_plan.non_paged[0];
    assert_eq!(
        allocation
            .group_capacity(non_paged.group(), non_paged.domain())
            .unwrap()
            .strategy,
        CapacityStrategy::BoundedLazy { max_blocks: 32 }
    );
    assert_eq!(
        allocation
            .group_capacity(
                state_plan.paged_attention[0].group,
                state_plan.paged_attention[0].domain,
            )
            .unwrap()
            .strategy
            .maximum_blocks(),
        4_096
    );
}

#[test]
fn qwen38_staging_width_cost_is_independent_of_retained_sessions() {
    let state_plan = negotiate_state_plan(
        &qwen38_27b_tensor_contract(),
        &StateBackendPlanRequest {
            backend: BackendKind::Cpu,
            device_ordinal: None,
            page_tokens_hint: Some(64),
            storage_dtype_hint: Some(StateDType::F16),
        },
    )
    .unwrap();
    let capacity = |staged_transaction_rows| {
        plan_managed_state_capacity(
            &state_plan,
            ModelInstanceId::new(838),
            ManagedStateCapacityRequest {
                total_paged_pages: 4_096,
                logical_token_reach: Some(262_144),
                retained_sequence_rows: 16,
                staged_transaction_rows,
            },
        )
        .unwrap()
        .1
        .expect("Qwen3.8 has retained tensor state")
    };

    let scalar = capacity(1);
    let legacy = capacity(16);
    assert_eq!(scalar.sequence_capacity(), 16);
    assert_eq!(scalar.transaction_capacity(), 1);
    assert_eq!(scalar.per_sequence_bytes(), 156_893_184);
    assert_eq!(scalar.authorized_bytes(), 2_667_184_128);
    assert_eq!(legacy.authorized_bytes(), 5_020_581_888);
    assert_eq!(
        legacy.authorized_bytes() - scalar.authorized_bytes(),
        2_353_397_760
    );
}

#[test]
fn cuda_contiguous_growth_peak_prices_old_and_new_backing() {
    assert_eq!(cuda_contiguous_replacement_peak_pages(64).unwrap(), 64);
    assert_eq!(cuda_contiguous_replacement_peak_pages(128).unwrap(), 192);
    assert_eq!(
        cuda_contiguous_replacement_peak_pages(1_024).unwrap(),
        1_536
    );
    assert_eq!(
        cuda_contiguous_replacement_peak_pages(4_096).unwrap(),
        6_144
    );
}

#[test]
fn cuda_durable_reservation_prices_the_largest_replacement_extra() {
    assert_eq!(
        cuda_largest_contiguous_replacement_extra([
            (1_024, 4 * 1024 * 1024),
            (1_024, 256 * 1024),
        ])
        .unwrap(),
        512 * 4 * 1024 * 1024
    );
    assert_eq!(
        cuda_largest_contiguous_replacement_extra(std::iter::empty()).unwrap(),
        0
    );
}

#[test]
fn qwen38_contiguous_context_fitter_selects_the_largest_admissible_page() {
    const TENSOR_BYTES: u64 = 2_667_184_128;
    const BYTES_PER_PAGE: u64 = 4 * 1024 * 1024;
    const PEAK_PAGES_AT_64K: u64 = 1_536;
    let exact_peak = TENSOR_BYTES + BYTES_PER_PAGE * PEAK_PAGES_AT_64K;
    let paged = [CudaContiguousPagedGeometry {
        page_tokens: 64,
        bytes_per_page: BYTES_PER_PAGE,
    }];
    assert_eq!(
        fit_cuda_contiguous_token_reach(262_144, 64, &paged, TENSOR_BYTES, exact_peak).unwrap(),
        65_536
    );
}

#[test]
fn qwen38_resident_context_fitter_uses_steady_backing_without_replacement_copy() {
    const TENSOR_BYTES: u64 = 2_667_184_128;
    const BYTES_PER_PAGE: u64 = 4 * 1024 * 1024;
    const STEADY_PAGES_AT_64K: u64 = 1_024;
    let exact_resident = TENSOR_BYTES + BYTES_PER_PAGE * STEADY_PAGES_AT_64K;
    let paged = [CudaContiguousPagedGeometry {
        page_tokens: 64,
        bytes_per_page: BYTES_PER_PAGE,
    }];

    assert_eq!(
        fit_cuda_resident_token_reach(262_144, 64, &paged, TENSOR_BYTES, exact_resident,)
            .unwrap(),
        65_536
    );
    assert!(
        cuda_contiguous_replacement_required_bytes(65_536, &paged, TENSOR_BYTES).unwrap()
            > exact_resident
    );
}

/// A paged attach beside an un-snapshotted ShortConv ring would restart
/// the ring at zero at cursor N. Even if LFM2's attention domain opted
/// into committed-page sharing, the gate must refuse it.
#[test]
fn prefix_gate_rejects_paged_attach_beside_an_unsnapshotted_ring() {
    let config = crate::models::architectures::lfm2::config::Lfm2BackboneConfig {
        architecture: "lfm2".into(),
        block_count: 4,
        context_length: 64,
        embedding_length: 16,
        embedding_length_out: None,
        feed_forward_length: Some(32),
        attention_head_count: 4,
        attention_head_count_kv: vec![0, 2, 0, 2],
        attention_layer_norm_rms_epsilon: 1e-5,
        attention_sliding_window: None,
        rope_freq_base: 1_000_000.0,
        shortconv_l_cache: 3,
        lfm3: None,
    };
    let mut contract =
        crate::models::architectures::lfm2::physical::lfm2_managed_cache_contract(
            &config,
            crate::kv::v2::StateDType::F32,
        )
        .unwrap();
    let mut attention = None;
    for domain in &mut contract.domains {
        if let StateDomainSpec::PagedAttention(spec) = domain {
            spec.header.prefix = PrefixPolicy::CommittedPages {
                positions: crate::kv::v2::PositionSemantics::Absolute,
            };
            attention = Some(spec.header.id);
        }
    }
    let attention = attention.expect("LFM2 has a paged attention domain");
    assert!(contract
        .domains
        .iter()
        .any(|domain| matches!(domain, StateDomainSpec::Ring(_))));
    assert!(!prefix_enabled_for_domain(&contract, attention));
    // A paged-only contract keeps plain paged sharing.
    let paged_only = crate::kv::v2::test_contract();
    assert!(prefix_enabled_for_domain(
        &paged_only,
        paged_only.domains[0].id()
    ));
}

/// LFM2.5 chat on CUDA fits one 32K context (6 attention layers x 8 KV
/// heads x 64 x K/V x F16 = 12,288 B/token, 64-token pages) instead of
/// growing a 128K arena: the whole context fits a small GPU after the
/// context reserve, and a tighter budget yields a smaller reach rather
/// than an overcommitted arena.
#[test]
fn lfm25_chat_cuda_fit_is_bounded_by_one_context() {
    let paged = [CudaContiguousPagedGeometry {
        page_tokens: 64,
        bytes_per_page: 64 * 12_288,
    }];
    // Ring state for 95 retained + 8 staged rows (10 layers x 3 x 2048 x F32).
    let fixed = 103 * 245_760;
    let one_context = cuda_resident_required_bytes(32_768, &paged, fixed).unwrap();
    assert_eq!(one_context, 402_653_184 + fixed);
    // A 6 GiB GPU after ~1.4 GB of weights and a 1 GiB context reserve.
    let roomy = 6 * 1024 * 1024 * 1024 - 1_400_000_000 - 1024 * 1024 * 1024;
    assert_eq!(
        fit_cuda_resident_token_reach(32_768, 64, &paged, fixed, roomy).unwrap(),
        32_768
    );
    // A crowded device: the reach shrinks and still fits.
    let tight = 300 * 1024 * 1024;
    let tokens = fit_cuda_resident_token_reach(32_768, 64, &paged, fixed, tight).unwrap();
    assert!((4_096..32_768).contains(&tokens), "{tokens}");
    assert!(cuda_resident_required_bytes(tokens, &paged, fixed).unwrap() <= tight);
}

#[test]
fn qwen38_resident_context_leaves_room_for_decode_and_safety_headroom() {
    let fixed = 2_667_184_128;
    let paged = [
        CudaContiguousPagedGeometry {
            page_tokens: 64,
            bytes_per_page: 4 * 1024 * 1024,
        },
        CudaContiguousPagedGeometry {
            page_tokens: 64,
            bytes_per_page: 256 * 1024,
        },
    ];
    let safety = 1024 * 1024 * 1024;
    let headroom = cuda_resident_required_bytes(65_536, &paged, fixed).unwrap() + safety;
    // Exact shipped adaptive MTP estimate plus host collation. Context
    // fitting conservatively protects the full mixed-domain stage budget.
    let row = crate::engine::continuous_chat_workspace_per_row(475_144_192)
        .unwrap()
        .workspace_bytes()
        .unwrap();
    for width in [1, 8] {
        let workspace = row * width;
        let budget = headroom - safety - workspace;
        let tokens = fit_cuda_resident_token_reach(262_144, 64, &paged, fixed, budget).unwrap();
        assert!(tokens < 65_536, "KV must yield space to decode workspace");
        let retained = cuda_resident_required_bytes(tokens, &paged, fixed).unwrap();
        assert!(retained + workspace + safety <= headroom);
        let next = cuda_resident_required_bytes(tokens + 1, &paged, fixed).unwrap();
        assert!(next + workspace + safety > headroom);
    }
}

#[test]
fn qwen38_mtp_context_fitter_prices_both_paged_domains() {
    const TENSOR_BYTES: u64 = 2_667_184_128;
    const TARGET_BYTES_PER_PAGE: u64 = 4 * 1024 * 1024;
    const MTP_BYTES_PER_PAGE: u64 = 256 * 1024;
    const STEADY_PAGES_AT_64K: u64 = 1_024;
    const PEAK_PAGES_AT_64K: u64 = 1_536;
    let paged = [
        CudaContiguousPagedGeometry {
            page_tokens: 64,
            bytes_per_page: TARGET_BYTES_PER_PAGE,
        },
        CudaContiguousPagedGeometry {
            page_tokens: 64,
            bytes_per_page: MTP_BYTES_PER_PAGE,
        },
    ];
    let exact_peak = TENSOR_BYTES
        + TARGET_BYTES_PER_PAGE * PEAK_PAGES_AT_64K
        + MTP_BYTES_PER_PAGE * STEADY_PAGES_AT_64K;

    assert_eq!(
        cuda_contiguous_replacement_required_bytes(65_536, &paged, TENSOR_BYTES).unwrap(),
        exact_peak
    );
    assert_eq!(
        fit_cuda_contiguous_token_reach(262_144, 64, &paged, TENSOR_BYTES, exact_peak).unwrap(),
        65_536
    );
}

#[test]
fn qwen38_cuda_context_budget_uses_live_device_headroom() {
    const TENSOR_BYTES: u64 = 2_667_184_128;
    const TARGET_BYTES_PER_PAGE: u64 = 4 * 1024 * 1024;
    const MTP_BYTES_PER_PAGE: u64 = 256 * 1024;
    const STEADY_PAGES_AT_64K: u64 = 1_024;
    const PEAK_PAGES_AT_64K: u64 = 1_536;
    let live_peak = TENSOR_BYTES
        + TARGET_BYTES_PER_PAGE * PEAK_PAGES_AT_64K
        + MTP_BYTES_PER_PAGE * STEADY_PAGES_AT_64K;
    let capacity = ResourceVector {
        device_bytes: ResourceAmount::Known(live_peak * 4),
        ..ResourceVector::zero()
    };
    let available = ResourceVector {
        device_bytes: ResourceAmount::Known(live_peak),
        ..ResourceVector::zero()
    };
    let authority = Arc::new(ResourceAuthority::new(Arc::new(TestCapacityProvider {
        snapshot: PhysicalCapacitySnapshot {
            capacity,
            available,
            source: CapacitySource::Test,
        },
    })));
    let ResourceAmount::Known(budget) = authority
        .planning_headroom_bytes(BackendKind::Cuda)
        .unwrap()
    else {
        panic!("CUDA test budget must be known");
    };
    let paged = [
        CudaContiguousPagedGeometry {
            page_tokens: 64,
            bytes_per_page: TARGET_BYTES_PER_PAGE,
        },
        CudaContiguousPagedGeometry {
            page_tokens: 64,
            bytes_per_page: MTP_BYTES_PER_PAGE,
        },
    ];

    assert_eq!(
        fit_cuda_contiguous_token_reach(262_144, 64, &paged, TENSOR_BYTES, budget,).unwrap(),
        65_536
    );
}

/// Run the CUDA resident row and context policy for one model on a device
/// of `device_bytes` whose resident weights are already materialized.
/// Negotiating a CUDA plan needs the CUDA build; the policy reads only
/// plan geometry, which a CPU plan of the same contract reproduces.
fn cuda_admission_fit(
    contract: &InferenceStateContract,
    device_bytes: u64,
    weights_bytes: u64,
    safety_reserve_bytes: u64,
    rows: CudaResidentRowRequest,
) -> Result<CudaResidentFit> {
    cuda_admission_fit_with_snapshots(
        contract,
        device_bytes,
        weights_bytes,
        safety_reserve_bytes,
        rows,
        0,
    )
}

fn cuda_admission_fit_with_snapshots(
    contract: &InferenceStateContract,
    device_bytes: u64,
    weights_bytes: u64,
    safety_reserve_bytes: u64,
    rows: CudaResidentRowRequest,
    prefix_snapshot_rows: u32,
) -> Result<CudaResidentFit> {
    let state_plan = negotiate_state_plan(
        contract,
        &StateBackendPlanRequest {
            backend: BackendKind::Cpu,
            device_ordinal: None,
            page_tokens_hint: Some(64),
            storage_dtype_hint: None,
        },
    )?;
    fit_resident_rows_and_reach(
        &state_plan,
        ModelInstanceId::new(3_838),
        |rows| Ok(262_144 * u64::from(rows)),
        device_bytes.saturating_sub(CUDA_CONTEXT_BYTES + weights_bytes),
        safety_reserve_bytes,
        64,
        rows,
        prefix_snapshot_rows,
    )
}

/// Driver context and allocator pools present before any weight loads.
const CUDA_CONTEXT_BYTES: u64 = 512 * 1024 * 1024;

#[test]
fn cuda_admission_matrix_fits_automatic_rows_to_hybrid_recurrent_state() {
    const GIB: u64 = 1024 * 1024 * 1024;
    const H100_BYTES: u64 = 85_017_624_576;
    const DECODE_WIDTH: u32 = 8;
    const STATE_SHARE: f64 = 0.25;
    use crate::models::architectures::qwen36moe::{
        cache::qwen35_composite_cache_contract_with_mtp, native::pinned_native_config,
        native_model::qwen35_text_config_from_native,
    };
    let qwen36 = qwen35_composite_cache_contract_with_mtp(
        &qwen35_text_config_from_native(&pinned_native_config().text),
        DType::BF16,
        64,
        false,
        false,
    )
    .unwrap();
    let workspace = |per_row| {
        crate::engine::continuous_chat_workspace_per_row(per_row)
            .unwrap()
            .workspace_bytes()
            .unwrap()
            * u64::from(DECODE_WIDTH)
    };
    // CUDA resident weights from the load-time plans: Qwen3.8 Q8_0 + BF16
    // (`qwen38_representation_memory_estimate`) and Qwen3.6 block FP8
    // including its graph pool (`qwen36moe_memory::resource_plan`). Decode
    // reserves: Qwen3.8's adaptive-MTP row, and a generous bound for
    // Qwen3.6's small rows.
    let models = [
        (
            "Qwen3.8",
            qwen38_27b_tensor_contract(),
            32_407_348_704_u64,
            156_893_184_u64,
            workspace(475_144_192),
        ),
        (
            "Qwen3.6",
            qwen36,
            35_337_864_704,
            65_863_680,
            workspace(64 * 1024 * 1024),
        ),
    ];
    // The CPU plans driving the policy carry the CUDA page geometry:
    // 64-token BF16-width pages (Qwen3.8 16 layers x 4 KV heads, Qwen3.6
    // 10 layers x 2 KV heads, head dim 256).
    for ((_, contract, ..), kv_bytes_per_token) in models.iter().zip([65_536, 20_480]) {
        let plan = negotiate_state_plan(
            contract,
            &StateBackendPlanRequest {
                backend: BackendKind::Cpu,
                device_ordinal: None,
                page_tokens_hint: Some(64),
                storage_dtype_hint: None,
            },
        )
        .unwrap();
        assert_eq!(plan.paged_attention.len(), 1);
        assert_eq!(plan.paged_attention[0].page_tokens, 64);
        assert_eq!(
            plan.paged_attention[0].bytes_per_page,
            64 * kv_bytes_per_token
        );
    }
    for (name, contract, weights, per_sequence, decode_reserve) in &models {
        for device_bytes in [24 * GIB, 40 * GIB, 48 * GIB, H100_BYTES, 141 * GIB] {
            let gib = device_bytes / GIB;
            let safety = GIB + decode_reserve;
            // The memory-derived automatic ceiling (device / 256 MiB).
            let ceiling = u32::try_from(device_bytes / (256 * 1024 * 1024)).unwrap();
            let fit = cuda_admission_fit(
                contract,
                device_bytes,
                *weights,
                safety,
                CudaResidentRowRequest {
                    retained_sequence_rows: ceiling,
                    staged_transaction_rows: DECODE_WIDTH,
                    automatic_state_budget_fraction: Some(STATE_SHARE),
                },
            );
            let headroom = device_bytes.saturating_sub(CUDA_CONTEXT_BYTES + weights);
            let floor_state = per_sequence * u64::from(2 * DECODE_WIDTH);
            if headroom < safety + floor_state + 8 * 1024 * 1024 {
                let error = fit.expect_err("weights leave no room for the decode width");
                let message = error.to_string();
                assert!(
                    message.contains("IZWI_MAX_STAGED_TRANSACTIONS"),
                    "{message}"
                );
                // The client sees the short shortfall; the geometry
                // (per-sequence bytes, rows) is logged.
                assert!(
                    message.contains(" GB, ") && !message.contains("per_sequence_state_bytes="),
                    "{message}"
                );
                continue;
            }
            let fit = fit.unwrap_or_else(|error| panic!("{name} at {gib} GiB: {error}"));
            let budget = headroom - safety;
            let state_cap = (budget as f64 * STATE_SHARE) as u64;
            let rows = fit.retained_sequence_rows;
            assert!(rows >= DECODE_WIDTH, "{name} at {gib} GiB kept {rows} rows");
            assert!(rows <= ceiling);
            assert_eq!(fit.staged_transaction_rows, DECODE_WIDTH);
            assert_eq!(fit.per_sequence_state_bytes, *per_sequence);
            assert_eq!(
                fit.resident_state_bytes,
                per_sequence * u64::from(rows + DECODE_WIDTH)
            );
            // Above the decode width, state stays within its share and
            // takes the most rows that do.
            assert!(
                rows == DECODE_WIDTH || fit.resident_state_bytes <= state_cap,
                "{name} at {gib} GiB reserved {} of {state_cap}",
                fit.resident_state_bytes
            );
            assert!(
                rows == ceiling
                    || per_sequence * u64::from(rows + 1 + DECODE_WIDTH) > state_cap,
                "{name} at {gib} GiB left rows unused"
            );
            assert!(fit.logical_token_reach >= 64);
        }
    }

    // The pre-fix resolution charged every memory-derived row as both
    // retained and staged state, refusing Qwen3.8 on every device.
    for device_bytes in [40 * GIB, 48 * GIB, H100_BYTES, 141 * GIB] {
        let ceiling = u32::try_from(device_bytes / (256 * 1024 * 1024)).unwrap();
        let (_, contract, weights, _, decode_reserve) = &models[0];
        let error = cuda_admission_fit(
            contract,
            device_bytes,
            *weights,
            GIB + decode_reserve,
            CudaResidentRowRequest {
                retained_sequence_rows: ceiling,
                staged_transaction_rows: ceiling,
                automatic_state_budget_fraction: None,
            },
        )
        .expect_err("explicit rows are a hard contract");
        assert!(
            error.to_string().contains("IZWI_MAX_RETAINED_SEQUENCES"),
            "{error}"
        );
    }
}

#[test]
fn exhausted_retained_rows_wait_for_a_released_session() {
    // A request holds one retained row from its first prepare until its
    // terminal release; rows are never kept between requests. With every
    // row held, a new session gets Backpressure, which the engine defers
    // and retries rather than failing, and it binds once a holder ends.
    let model = ModelInstanceId::new(606);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_model_state_with_capacity(
            model,
            BackendKind::Cpu,
            ManagedStateCapacityRequest {
                total_paged_pages: 12,
                logical_token_reach: None,
                retained_sequence_rows: 2,
                staged_transaction_rows: 1,
            },
            32,
            &hybrid_snapshot_contract(),
        )
        .unwrap();
    let arena = runtime.tensor_state().unwrap().clone();
    let request = prefix_request(model, (0..32_u32).collect());
    let commit = |manager: &mut ManagedKvCacheManager, plan: u64, session: &SessionKey| {
        let reservation = manager
            .prepare(
                &runtime,
                plan,
                session,
                &sequence_work(0, 32),
                Some(&request),
            )?
            .expect("managed reservation");
        arena.stage_replace(
            PhysicalStateTransactionId::new(plan).unwrap(),
            StateDomainId::new(2),
            0,
            32,
            hybrid_tensor_values(&[1.0; 4]),
        )?;
        manager.finalize(
            &reservation,
            Some(&reservation.completed_write_receipt_for_test()),
            true,
        )
    };
    let holders = [
        SessionKey::new("row-holder-a".into(), 1),
        SessionKey::new("row-holder-b".into(), 1),
    ];
    commit(&mut manager, 71, &holders[0]).unwrap();
    commit(&mut manager, 72, &holders[1]).unwrap();

    let waiter = SessionKey::new("row-waiter".into(), 1);
    let error = commit(&mut manager, 73, &waiter).unwrap_err();
    assert!(
        matches!(&error, Error::Backpressure(reason) if reason.contains("sequence capacity")),
        "{error}"
    );
    manager.release_session(&holders[0]).unwrap();
    commit(&mut manager, 74, &waiter).unwrap();
    assert_eq!(arena.occupancy().unwrap().active_sequences, 2);
}

#[test]
fn multi_domain_cuda_context_fitter_handles_heterogeneous_page_geometry() {
    let paged = [
        CudaContiguousPagedGeometry {
            page_tokens: 64,
            bytes_per_page: 100,
        },
        CudaContiguousPagedGeometry {
            page_tokens: 32,
            bytes_per_page: 10,
        },
    ];
    // At 65,536 tokens the arenas have 1,024 and 2,048 steady pages.
    // Their replacement extras are respectively 51,200 and 10,240 bytes,
    // so only the larger extra is simultaneous with aggregate steady state.
    assert_eq!(
        cuda_contiguous_replacement_required_bytes(65_536, &paged, 7).unwrap(),
        7 + 102_400 + 20_480 + 51_200
    );
}

#[test]
fn multi_domain_cuda_context_fitter_rejects_invalid_geometry() {
    assert!(cuda_contiguous_replacement_required_bytes(64, &[], 0).is_err());
    assert!(
        cuda_contiguous_replacement_required_bytes(
            64,
            &[CudaContiguousPagedGeometry {
                page_tokens: 0,
                bytes_per_page: 1,
            }],
            0,
        )
        .is_err()
    );
    assert!(
        cuda_contiguous_replacement_required_bytes(
            64,
            &[CudaContiguousPagedGeometry {
                page_tokens: 1,
                bytes_per_page: u64::MAX,
            }],
            0,
        )
        .is_err()
    );
}

fn prefix_request(model: ModelInstanceId, tokens: Vec<u32>) -> EngineCoreRequest {
    let variant = ModelVariant::Qwen306B;
    let profile =
        ExecutionProfile::fail_closed(BackendKind::Cpu, Some(variant), ExecutionMode::Sequence);
    let stage = StageDescriptor::from_execution_profile(
        StageId::new(1),
        "qwen3.managed",
        &profile,
        NativeBatchMode::None,
    );
    let mut request = EngineCoreRequest::chat(vec![ChatMessage {
        role: ChatRole::User,
        content: "prefix".into(),
    }])
    .with_model_variant(variant);
    request.prompt_tokens = tokens;
    request.params.max_tokens = 1;
    request
        .bind_execution_adapter(ExecutionAdapterBinding {
            execution_group_id: ExecutionGroupId::new(1),
            model_instance_id: model,
            adapter_instance_id: AdapterInstanceId::new(2),
            adapter_abi_revision: AdapterAbiRevision::new(9),
            model_variant: variant,
            capability_id: "chat".into(),
            stages: Arc::from([stage]),
        })
        .expect("adapter binding");
    request
}

#[test]
fn incremental_admission_shares_uncapped_requests_and_waits_for_next_step() {
    let model = ModelInstanceId::new(840);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            3,
            8,
            &CacheCapability::Managed(test_contract()),
        )
        .unwrap()
        .unwrap();
    let mut request = prefix_request(model, vec![1, 2, 3]);
    request.params.max_tokens = runtime.logical_token_reach() as usize - 3;
    let sessions = (0..3)
        .map(|i| SessionKey::new(format!("incremental-{i}"), 1))
        .collect::<Vec<_>>();
    for (index, session) in sessions.iter().enumerate() {
        let reservation = manager
            .prepare_incremental(
                &runtime,
                8400 + index as u64,
                session,
                &sequence_work(0, 3),
                Some(&request),
            )
            .unwrap()
            .unwrap();
        manager
            .finalize(
                &reservation,
                Some(&reservation.completed_write_receipt_for_test()),
                true,
            )
            .unwrap();
    }
    assert_eq!(
        manager.runtime_snapshot().models[0].incremental_claim_sessions,
        3
    );
    let before = manager.models[&model].capacity_claims.clone();
    assert!(matches!(
        manager.prepare_incremental(
            &runtime,
            8410,
            &sessions[0],
            &sequence_work(3, 9),
            Some(&request)
        ),
        Err(Error::Backpressure(_))
    ));
    assert_eq!(manager.models[&model].capacity_claims, before);
    manager.release_session(&sessions[1]).unwrap();
    let next = manager
        .prepare_incremental(
            &runtime,
            8411,
            &sessions[0],
            &sequence_work(3, 9),
            Some(&request),
        )
        .unwrap()
        .unwrap();
    manager.finalize(&next, None, false).unwrap();
    for session in &sessions {
        manager.release_session(session).unwrap();
    }
    assert!(manager.models[&model].capacity_claims.is_empty());
    assert_eq!(
        manager.runtime_snapshot().models[0].incremental_claim_sessions,
        0
    );
    for coordinator in manager.models[&model].coordinators.values() {
        assert_eq!(coordinator.stats().allocated_pages, 0);
        assert_eq!(coordinator.stats().active_transactions, 0);
        coordinator.check_invariants().unwrap();
    }
}

#[test]
fn incremental_prepare_failure_restores_existing_claim() {
    let model = ModelInstanceId::new(841);
    let session = SessionKey::new("incremental-rollback".into(), 1);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            4,
            8,
            &CacheCapability::Managed(test_contract()),
        )
        .unwrap()
        .unwrap();
    let request = prefix_request(model, vec![1, 2, 3]);
    let reservation = manager
        .prepare_incremental(
            &runtime,
            8420,
            &session,
            &sequence_work(0, 3),
            Some(&request),
        )
        .unwrap()
        .unwrap();
    manager
        .finalize(
            &reservation,
            Some(&reservation.completed_write_receipt_for_test()),
            true,
        )
        .unwrap();
    let before = manager.models[&model].capacity_claims[&session].clone();
    // A newly known prompt grows the claim, but a regressed execution
    // target must reject prepare and restore the earlier claim.
    let mut expanded_request = request.clone();
    expanded_request.prompt_tokens = vec![1; 20];
    assert!(
        manager
            .prepare_incremental(
                &runtime,
                8421,
                &session,
                &sequence_work(0, 2),
                Some(&expanded_request)
            )
            .is_err()
    );
    assert_eq!(manager.models[&model].capacity_claims[&session], before);
    for coordinator in manager.models[&model].coordinators.values() {
        assert_eq!(coordinator.stats().active_transactions, 0);
        coordinator.check_invariants().unwrap();
    }
}

#[test]
fn incremental_claim_covers_known_prompt_next_token_and_replay_history() {
    let model = ModelInstanceId::new(844);
    let session = SessionKey::new("incremental-step-boundary".into(), 1);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            4,
            8,
            &CacheCapability::Managed(test_contract()),
        )
        .unwrap()
        .unwrap();
    let mut request = prefix_request(model, vec![1; 8]);
    request.params.max_tokens = 24;
    let first = manager
        .prepare_incremental(
            &runtime,
            8440,
            &session,
            &sequence_work(0, 2),
            Some(&request),
        )
        .unwrap()
        .unwrap();
    // The whole known prompt is admitted, although this chunk is shorter.
    assert_eq!(manager.models[&model].capacity_claims[&session][0].1, 1);
    manager
        .finalize(
            &first,
            Some(&first.completed_write_receipt_for_test()),
            true,
        )
        .unwrap();
    let rest = manager
        .prepare_incremental(
            &runtime,
            8441,
            &session,
            &sequence_work(2, 8),
            Some(&request),
        )
        .unwrap()
        .unwrap();
    manager
        .finalize(&rest, Some(&rest.completed_write_receipt_for_test()), true)
        .unwrap();
    let decode = WorkUnit::SequenceStep {
        phase: SequencePhase::Decode,
        input: InputRange { start: 8, end: 9 },
        max_output_steps: 1,
        auxiliary_state: None,
    };
    let next = manager
        .prepare_incremental(&runtime, 8442, &session, &decode, Some(&request))
        .unwrap()
        .unwrap();
    assert_eq!(manager.models[&model].capacity_claims[&session][0].1, 2);
    manager
        .finalize(&next, Some(&next.completed_write_receipt_for_test()), true)
        .unwrap();
    manager.release_session(&session).unwrap();
    // Suspension frees both pages. Replay spans include generated history
    // while the immutable request still carries its eight-token prompt.
    let replay = manager
        .prepare_incremental(
            &runtime,
            8443,
            &session,
            &sequence_work(0, 17),
            Some(&request),
        )
        .unwrap()
        .unwrap();
    assert_eq!(request.num_prompt_tokens(), 8);
    assert_eq!(manager.models[&model].capacity_claims[&session][0].1, 3);
    assert_eq!(replay.domains[0].target_committed_tokens, 17);
    manager.finalize(&replay, None, false).unwrap();
    manager.release_session(&session).unwrap();
    assert!(manager.models[&model].capacity_claims.is_empty());
}

#[test]
fn incremental_multi_arena_growth_is_atomic() {
    let model = ModelInstanceId::new(843);
    let session = SessionKey::new("incremental-atomic".into(), 1);
    let other = SessionKey::new("incremental-other".into(), 1);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            6,
            8,
            &CacheCapability::Managed(two_paged_tensor_contract()),
        )
        .unwrap()
        .unwrap();
    let request = prefix_request(model, vec![1, 2, 3]);
    let state = manager.models.get_mut(&model).unwrap();
    ensure_incremental_capacity_claim(state, &session, &request, &sequence_work(0, 3)).unwrap();
    let second = runtime.plan().groups[1].arena;
    state.capacity_claims.insert(other, vec![(second, 2)]);
    let before = state.capacity_claims.clone();
    assert!(matches!(
        ensure_incremental_capacity_claim(state, &session, &request, &sequence_work(3, 9)),
        Err(Error::Backpressure(_))
    ));
    assert_eq!(state.capacity_claims, before);
}

#[test]
fn incremental_claim_aggregates_groups_sharing_an_arena() {
    let model = ModelInstanceId::new(845);
    let session = SessionKey::new("incremental-shared-arena".into(), 1);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            6,
            8,
            &CacheCapability::Managed(two_paged_tensor_contract()),
        )
        .unwrap()
        .unwrap();
    drop(runtime);
    let state = manager.models.get_mut(&model).unwrap();
    // Live binding currently gives each group an exclusive arena. Exercise
    // admission independently with pooled geometry so this invariant is
    // not required for correct aggregate claims.
    let runtime = Arc::get_mut(&mut state.runtime).unwrap();
    let plan = Arc::make_mut(&mut runtime.plan);
    let shared_arena = plan.groups[0].arena;
    plan.groups[1].arena = shared_arena;
    let request = prefix_request(model, vec![1; 3]);
    ensure_incremental_capacity_claim(state, &session, &request, &sequence_work(0, 3)).unwrap();
    assert_eq!(state.capacity_claims[&session], vec![(shared_arena, 2)]);
    assert!(matches!(
        ensure_incremental_capacity_claim(state, &session, &request, &sequence_work(3, 9)),
        Err(Error::Overloaded(_))
    ));
    assert_eq!(state.capacity_claims[&session], vec![(shared_arena, 2)]);
}

#[test]
fn model_context_metadata_is_independent_of_shared_pool() {
    let model = ModelInstanceId::new(842);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            8,
            8,
            &CacheCapability::Managed(test_contract()),
        )
        .unwrap()
        .unwrap();
    runtime.set_maximum_sequence_tokens(16);
    assert_eq!(runtime.maximum_sequence_tokens(), 16);
    assert_eq!(runtime.logical_token_reach(), 64);
    assert_eq!(
        full_context_sequence_capacity(runtime.plan(), runtime.maximum_sequence_tokens()),
        4
    );
}

#[test]
fn live_manager_commits_aborts_and_releases_exact_session_tables() {
    let model = ModelInstanceId::new(41);
    let session = SessionKey::new("managed-live".to_string(), 7);
    let domain = CacheDomainId::new(1);
    let mut manager = ManagedKvCacheManager::with_prefix_cache_salt(None, Some([5; 32]));
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            4,
            16,
            &CacheCapability::Managed(test_contract()),
        )
        .expect("bind")
        .expect("managed runtime");
    assert!(runtime.physical_bytes() > 0);

    let first = manager
        .prepare(&runtime, 1, &session, &sequence_work(0, 5), None)
        .expect("prepare")
        .expect("reservation");
    assert_eq!(first.domains.len(), 1);
    assert_eq!(first.domains[0].writable_blocks.len(), 1);
    manager
        .finalize(
            &first,
            Some(&first.completed_write_receipt_for_test()),
            true,
        )
        .expect("commit");
    let snapshot = manager.snapshot(model, &session, domain).expect("snapshot");
    assert_eq!(snapshot.version, 1);
    assert_eq!(snapshot.committed_tokens, 5);

    let second = manager
        .prepare(&runtime, 2, &session, &sequence_work(5, 17), None)
        .expect("prepare")
        .expect("reservation");
    assert_eq!(second.domains[0].writable_blocks.len(), 2);
    manager.finalize(&second, None, false).expect("abort");
    let unchanged = manager.snapshot(model, &session, domain).expect("snapshot");
    assert_eq!(unchanged.version, 1);
    assert_eq!(unchanged.committed_tokens, 5);

    manager.release_session(&session).expect("release");
    assert!(manager.snapshot(model, &session, domain).is_none());
}

#[test]
fn retained_session_reset_preserves_claim_and_rejects_stale_reservation_receipt() {
    let model = ModelInstanceId::new(410);
    let session = SessionKey::new("managed-reset".to_string(), 3);
    let domain = CacheDomainId::new(1);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            4,
            16,
            &CacheCapability::Managed(test_contract()),
        )
        .unwrap()
        .unwrap();
    let mut request = prefix_request(model, vec![1, 2, 3, 4, 5]);
    request.params.max_tokens = 3;
    let first = manager
        .prepare(
            &runtime,
            4101,
            &session,
            &sequence_work(0, 5),
            Some(&request),
        )
        .unwrap()
        .unwrap();
    assert_eq!(first.session_generation, ManagedSessionGeneration::INITIAL);
    let stale_receipt = first.completed_write_receipt_for_test();
    manager
        .finalize(&first, Some(&stale_receipt), true)
        .unwrap();
    let claim_before = manager.models[&model].capacity_claims[&session].clone();
    let registered_before = manager.models[&model].registered_sessions.len();

    let next = manager
        .reset_session_generation(&runtime, &session, ManagedSessionGeneration::INITIAL)
        .unwrap();
    assert_eq!(next.get(), 2);
    let reset = manager.snapshot(model, &session, domain).unwrap();
    assert_eq!(reset.committed_tokens, 0);
    assert_eq!(reset.window_start, 0);
    assert!(reset.groups.is_empty());
    assert_eq!(reset.version, 2);
    assert_eq!(
        manager.models[&model].capacity_claims[&session],
        claim_before
    );
    assert_eq!(
        manager.models[&model].registered_sessions.len(),
        registered_before
    );

    let stale = manager.finalize(&first, Some(&stale_receipt), true);
    assert!(matches!(stale, Err(Error::InferenceError(message)) if message.contains("stale")));

    let replacement = manager
        .prepare(&runtime, 4102, &session, &sequence_work(0, 3), None)
        .unwrap()
        .unwrap();
    assert_eq!(replacement.session_generation, next);
    manager
        .finalize(
            &replacement,
            Some(&replacement.completed_write_receipt_for_test()),
            true,
        )
        .unwrap();
    assert_eq!(
        manager
            .snapshot(model, &session, domain)
            .unwrap()
            .committed_tokens,
        3
    );
}

#[test]
fn retained_session_reset_fails_closed_while_a_row_transaction_is_active() {
    let model = ModelInstanceId::new(411);
    let session = SessionKey::new("managed-reset-active".to_string(), 4);
    let domain = CacheDomainId::new(1);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            4,
            16,
            &CacheCapability::Managed(test_contract()),
        )
        .unwrap()
        .unwrap();
    let active = manager
        .prepare(&runtime, 4111, &session, &sequence_work(0, 4), None)
        .unwrap()
        .unwrap();

    let reset =
        manager.reset_session_generation(&runtime, &session, ManagedSessionGeneration::INITIAL);
    assert!(reset.is_err());
    let unchanged = manager.snapshot(model, &session, domain).unwrap();
    assert_eq!(unchanged.version, 0);
    assert_eq!(unchanged.committed_tokens, 0);
    assert_eq!(
        manager.models[&model].session_generations[&session],
        ManagedSessionGeneration::INITIAL
    );

    manager.finalize(&active, None, false).unwrap();
    let next = manager
        .reset_session_generation(&runtime, &session, ManagedSessionGeneration::INITIAL)
        .unwrap();
    assert_eq!(next.get(), 2);
}

#[test]
fn retained_session_reset_rejects_stale_expected_generation_without_mutation() {
    let model = ModelInstanceId::new(412);
    let session = SessionKey::new("managed-reset-generation".to_string(), 5);
    let domain = CacheDomainId::new(1);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            4,
            16,
            &CacheCapability::Managed(test_contract()),
        )
        .unwrap()
        .unwrap();
    let reservation = manager
        .prepare(&runtime, 4121, &session, &sequence_work(0, 2), None)
        .unwrap()
        .unwrap();
    manager.finalize(&reservation, None, false).unwrap();
    let next = manager
        .reset_session_generation(&runtime, &session, ManagedSessionGeneration::INITIAL)
        .unwrap();
    let version = manager.snapshot(model, &session, domain).unwrap().version;

    assert!(
        manager
            .reset_session_generation(&runtime, &session, ManagedSessionGeneration::INITIAL,)
            .is_err()
    );
    assert_eq!(
        manager.snapshot(model, &session, domain).unwrap().version,
        version
    );
    assert_eq!(manager.models[&model].session_generations[&session], next);
}

#[test]
fn retained_session_reset_disables_published_prefix_reuse_for_restarted_generation() {
    let model = ModelInstanceId::new(413);
    let tokens = (0..65).collect::<Vec<u32>>();
    let request = prefix_request(model, tokens.clone());
    let source_session = SessionKey::new("managed-reset-prefix-source".into(), 1);
    let restarted_session = SessionKey::new("managed-reset-prefix-target".into(), 1);
    let mut manager = ManagedKvCacheManager::with_prefix_cache_salt(None, Some([11; 32]));
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            4,
            32,
            &CacheCapability::Managed(test_contract()),
        )
        .unwrap()
        .unwrap();

    let source = manager
        .prepare(
            &runtime,
            4131,
            &source_session,
            &sequence_work(0, tokens.len()),
            Some(&request),
        )
        .unwrap()
        .unwrap();
    manager
        .finalize(
            &source,
            Some(&source.completed_write_receipt_for_test()),
            true,
        )
        .unwrap();
    manager.release_session(&source_session).unwrap();
    assert_eq!(manager.runtime_snapshot().counters.prefix_retained_pages, 2);

    let initial = manager
        .prepare(
            &runtime,
            4132,
            &restarted_session,
            &sequence_work(0, 1),
            None,
        )
        .unwrap()
        .unwrap();
    manager
        .finalize(
            &initial,
            Some(&initial.completed_write_receipt_for_test()),
            true,
        )
        .unwrap();
    let next = manager
        .reset_session_generation(
            &runtime,
            &restarted_session,
            ManagedSessionGeneration::INITIAL,
        )
        .unwrap();
    assert_eq!(next.get(), 2);
    let hits_before = manager.telemetry_snapshot().prefix_hits;

    let restarted = manager
        .prepare(
            &runtime,
            4133,
            &restarted_session,
            &sequence_work(0, tokens.len()),
            Some(&request),
        )
        .unwrap()
        .unwrap();
    assert_eq!(restarted.session_generation, next);
    assert!(
        restarted
            .domains
            .iter()
            .all(|domain| domain.execution_start_tokens == 0)
    );
    assert_eq!(manager.telemetry_snapshot().prefix_hits, hits_before);
    manager.finalize(&restarted, None, false).unwrap();
}

#[test]
fn retained_session_reset_stages_every_domain_before_mutation() {
    let model = ModelInstanceId::new(414);
    let session = SessionKey::new("managed-reset-two-domains".into(), 1);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            4,
            16,
            &CacheCapability::Managed(two_paged_contract()),
        )
        .unwrap()
        .unwrap();
    let mut request = prefix_request(model, vec![1, 2, 3, 4]);
    request.params.max_tokens = 4;
    let committed = manager
        .prepare(
            &runtime,
            4141,
            &session,
            &sequence_work(0, 4),
            Some(&request),
        )
        .unwrap()
        .unwrap();
    manager
        .finalize(
            &committed,
            Some(&committed.completed_write_receipt_for_test()),
            true,
        )
        .unwrap();

    let first_group = runtime.plan.groups[0].clone();
    let second_group = runtime.plan.groups[1].clone();
    let first_before = manager
        .snapshot(model, &session, first_group.domain)
        .unwrap();
    let second_before = manager
        .snapshot(model, &session, second_group.domain)
        .unwrap();
    let claim_before = manager.models[&model].capacity_claims[&session].clone();

    // Hold only the second domain active. Reset staging visits the first
    // domain successfully, then must reject the second without applying
    // either replacement.
    {
        let state = manager.models.get_mut(&model).unwrap();
        let coordinator = state.coordinators.get_mut(&second_group.arena).unwrap();
        let snapshot = coordinator.snapshot(&session, second_group.domain).unwrap();
        let target = snapshot.committed_tokens + 1;
        let group = reservation_for_group(
            second_group.id,
            second_group.page_tokens,
            &snapshot,
            target,
            &[],
        )
        .unwrap();
        coordinator
            .reserve(KvReserveRequest {
                txn_id: 4142,
                expected: snapshot,
                target_committed_tokens: target,
                target_window_start: 0,
                groups: vec![group],
            })
            .unwrap();
    }
    let runtime_before = manager.runtime_snapshot();

    assert!(
        manager
            .reset_session_generation(&runtime, &session, ManagedSessionGeneration::INITIAL)
            .is_err()
    );
    assert_eq!(
        manager
            .snapshot(model, &session, first_group.domain)
            .unwrap(),
        first_before
    );
    assert_eq!(
        manager
            .snapshot(model, &session, second_group.domain)
            .unwrap(),
        second_before
    );
    assert_eq!(
        manager.models[&model].capacity_claims[&session],
        claim_before
    );
    assert_eq!(manager.runtime_snapshot(), runtime_before);
    assert_eq!(
        manager.models[&model].session_generations[&session],
        ManagedSessionGeneration::INITIAL
    );

    manager
        .models
        .get_mut(&model)
        .unwrap()
        .coordinators
        .get_mut(&second_group.arena)
        .unwrap()
        .abort(4142)
        .unwrap();
}

#[test]
fn full_request_claim_prevents_late_decode_overcommit_and_releases_atomically() {
    let model = ModelInstanceId::new(42);
    let owner_session = SessionKey::new("capacity-owner".into(), 1);
    let waiter_session = SessionKey::new("capacity-waiter".into(), 1);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            4,
            16,
            &CacheCapability::Managed(test_contract()),
        )
        .unwrap()
        .unwrap();
    let mut owner = prefix_request(model, (0..32).collect());
    owner.params.max_tokens = 32;
    let owner_reservation = manager
        .prepare(
            &runtime,
            101,
            &owner_session,
            &sequence_work(0, 32),
            Some(&owner),
        )
        .unwrap()
        .unwrap();

    let waiter = prefix_request(model, vec![7]);
    let blocked = manager.prepare(
        &runtime,
        102,
        &waiter_session,
        &sequence_work(0, 1),
        Some(&waiter),
    );
    assert!(
        matches!(blocked, Err(Error::Backpressure(message)) if message.contains("full-request admission"))
    );
    let snapshot = manager.runtime_snapshot();
    assert_eq!(
        snapshot.models[0].arenas[0]
            .coordinator
            .admission_claimed_pages,
        4
    );
    assert_eq!(
        snapshot.models[0].arenas[0]
            .coordinator
            .admission_available_pages,
        0
    );

    manager.finalize(&owner_reservation, None, false).unwrap();
    manager.release_session(&owner_session).unwrap();
    let admitted = manager
        .prepare(
            &runtime,
            103,
            &waiter_session,
            &sequence_work(0, 1),
            Some(&waiter),
        )
        .unwrap()
        .unwrap();
    manager.finalize(&admitted, None, false).unwrap();
    manager.release_session(&waiter_session).unwrap();
    assert_eq!(
        manager.runtime_snapshot().models[0].arenas[0]
            .coordinator
            .admission_claimed_pages,
        0
    );
}

#[test]
fn realtime_physical_token_demand_claims_the_full_rotating_window() {
    let model = ModelInstanceId::new(421);
    let session = SessionKey::new("realtime-full-window-claim".into(), 1);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            4,
            16,
            &CacheCapability::Managed(sliding_contract(64)),
        )
        .unwrap()
        .unwrap();
    let logical_reach = usize::try_from(runtime.logical_token_reach()).unwrap();
    let mut request = EngineCoreRequest::asr_bytes(Vec::new())
        .with_model_variant(ModelVariant::VoxtralMini4BRealtime2602);
    request.params.max_tokens = logical_reach;

    let state = manager.models.get_mut(&model).unwrap();
    assert!(ensure_capacity_claim(state, &session, &request).unwrap());
    let claims = &state.capacity_claims[&session];
    assert!(!claims.is_empty());
    for (arena, claimed_pages) in claims {
        let capacity_pages = state
            .runtime
            .plan
            .groups
            .iter()
            .find(|group| group.arena == *arena)
            .map(|group| group.capacity_pages)
            .unwrap();
        assert_eq!(*claimed_pages, capacity_pages);
    }
}

#[test]
fn one_request_larger_than_the_arena_fails_before_page_dispatch() {
    let model = ModelInstanceId::new(43);
    let session = SessionKey::new("oversized-capacity".into(), 1);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            2,
            16,
            &CacheCapability::Managed(test_contract()),
        )
        .unwrap()
        .unwrap();
    let request = prefix_request(model, (0..32).collect());

    let result = manager.prepare(
        &runtime,
        104,
        &session,
        &sequence_work(0, 32),
        Some(&request),
    );

    assert!(
        matches!(result, Err(Error::Overloaded(message)) if message.contains("prompt-plus-output"))
    );
    assert_eq!(manager.runtime_snapshot().totals.registered_sessions, 0);
    assert_eq!(
        manager
            .runtime_snapshot()
            .totals
            .coordinator
            .admission_claims,
        0
    );
}

#[test]
fn session_release_is_retryable_while_a_row_transaction_is_active() {
    let model = ModelInstanceId::new(53);
    let session = SessionKey::new("managed-release-retry".into(), 1);
    let domain = CacheDomainId::new(1);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            2,
            16,
            &CacheCapability::Managed(test_contract()),
        )
        .unwrap()
        .unwrap();
    let reservation = manager
        .prepare(&runtime, 41, &session, &sequence_work(0, 1), None)
        .unwrap()
        .unwrap();

    assert!(manager.release_session(&session).is_err());
    assert!(
        manager.models[&model]
            .registered_sessions
            .contains(&session)
    );
    assert!(manager.snapshot(model, &session, domain).is_some());

    manager.finalize(&reservation, None, false).unwrap();
    manager.release_session(&session).unwrap();
    assert!(
        !manager.models[&model]
            .registered_sessions
            .contains(&session)
    );
    assert!(manager.snapshot(model, &session, domain).is_none());
}

#[test]
fn runtime_snapshot_reports_exact_physical_state_and_serializes() {
    let model = ModelInstanceId::new(44);
    let session = SessionKey::new("managed-telemetry".to_string(), 3);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            2,
            16,
            &CacheCapability::Managed(test_contract()),
        )
        .expect("bind")
        .expect("managed runtime");
    let reservation = manager
        .prepare(&runtime, 11, &session, &sequence_work(0, 1), None)
        .expect("prepare")
        .expect("reservation");

    let prepared = manager.runtime_snapshot();
    assert_eq!(
        prepared.memory_accounting,
        "resident_paged_plus_authorized_tensor"
    );
    assert_eq!(prepared.totals.models, 1);
    assert_eq!(prepared.totals.arenas, 1);
    assert_eq!(prepared.totals.physical_bytes, runtime.physical_bytes());
    assert_eq!(
        prepared.totals.resident_paged_bytes,
        runtime.resident_paged_bytes()
    );
    assert_eq!(prepared.totals.authorized_tensor_bytes, 0);
    assert_eq!(prepared.totals.coordinator.capacity_pages, 2);
    assert_eq!(prepared.totals.coordinator.allocated_pages, 1);
    assert_eq!(prepared.totals.coordinator.active_transactions, 1);
    assert_eq!(prepared.totals.operations.page_zero_dispatches, 1);
    assert_eq!(prepared.totals.operations.backing_allocations, 2);
    assert_eq!(
        prepared
            .totals
            .operations
            .backing_allocations_observed_arenas,
        1
    );
    assert_eq!(prepared.totals.operations.workspace_bytes, 0);
    assert_eq!(
        prepared.totals.operations.workspace_bytes_observed_arenas,
        1
    );
    assert_eq!(prepared.counters.pages_zeroed, 1);
    assert_eq!(prepared.counters.backing_allocations, 1);
    assert_eq!(prepared.models[0].model_instance, model);
    assert_eq!(
        prepared.models[0].arenas[0].physical_bytes,
        runtime.physical_bytes()
    );

    let encoded = serde_json::to_value(&prepared).expect("serialize managed KV telemetry");
    assert_eq!(
        encoded["memory_accounting"],
        "resident_paged_plus_authorized_tensor"
    );
    assert_eq!(encoded["totals"]["coordinator"]["allocated_pages"], 1);
    assert_eq!(encoded["totals"]["operations"]["backing_allocations"], 2);
    assert_eq!(
        encoded["totals"]["operations"]["workspace_allocations_observed_arenas"],
        1
    );
    assert_eq!(encoded["models"][0]["backend"], "cpu");

    manager.finalize(&reservation, None, false).expect("abort");
    let aborted = manager.runtime_snapshot();
    assert_eq!(aborted.counters.transaction_aborts, 1);
    assert_eq!(aborted.totals.coordinator.allocated_pages, 0);
    assert_eq!(aborted.totals.coordinator.active_transactions, 0);
}

#[test]
fn live_sliding_window_table_stays_bounded_and_carries_first_page_offset() {
    let model = ModelInstanceId::new(48);
    let session = SessionKey::new("managed-window".into(), 1);
    let domain = CacheDomainId::new(1);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            4,
            16,
            &CacheCapability::Managed(sliding_contract(32)),
        )
        .expect("bind")
        .expect("managed runtime");

    let mut observed_nonzero_offset = false;
    for target in 1..=256_usize {
        let reservation = manager
            .prepare(
                &runtime,
                target as u64,
                &session,
                &sequence_work(target - 1, target),
                None,
            )
            .expect("prepare")
            .expect("reservation");
        let row = &reservation.domains[0];
        assert_eq!(
            row.target_window_start,
            u32::try_from(target.saturating_sub(32)).unwrap()
        );
        assert_eq!(row.first_page_offset, row.target_window_start % 16);
        observed_nonzero_offset |= row.first_page_offset != 0;
        assert!(row.provisional_groups[0].blocks.len() <= 3);
        manager
            .finalize(
                &reservation,
                Some(&reservation.completed_write_receipt_for_test()),
                true,
            )
            .expect("commit");
    }

    assert!(observed_nonzero_offset);
    let snapshot = manager.snapshot(model, &session, domain).expect("snapshot");
    assert_eq!(snapshot.committed_tokens, 256);
    assert_eq!(snapshot.window_start, 224);
    assert!(snapshot.groups[0].blocks.len() <= 3);
}

#[test]
fn chunked_sliding_window_span_keeps_the_earliest_querys_lookback() {
    let model = ModelInstanceId::new(49);
    let session = SessionKey::new("managed-window-chunks".into(), 1);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            8,
            16,
            &CacheCapability::Managed(sliding_contract(32)),
        )
        .expect("bind")
        .expect("managed runtime");

    let chunk = 8_usize;
    for (step, start) in (0..256_usize).step_by(chunk).enumerate() {
        let target = start + chunk;
        let reservation = manager
            .prepare(
                &runtime,
                step as u64 + 1,
                &session,
                &sequence_work(start, target),
                None,
            )
            .expect("prepare")
            .expect("reservation");
        let row = &reservation.domains[0];
        // The query at `start` attends keys `start + 1 - 32 ..= start`;
        // `target - 32` would have dropped seven of them.
        assert_eq!(
            row.target_window_start,
            u32::try_from((start + 1).saturating_sub(32)).unwrap()
        );
        assert_eq!(row.first_page_offset, row.target_window_start % 16);
        // 32 + 8 - 1 visible tokens plus a page offset fit in four pages.
        assert!(row.provisional_groups[0].blocks.len() <= 4);
        manager
            .finalize(
                &reservation,
                Some(&reservation.completed_write_receipt_for_test()),
                true,
            )
            .expect("commit");
    }
}

/// B3: page zeroing, host-offload capture and promotion restores of a
/// prepare run on the arena's device thread, not on the preparing thread.
#[test]
fn prepare_runs_kv_page_ops_on_the_device_thread() {
    let model = ModelInstanceId::new(50);
    let mut manager = ManagedKvCacheManager::with_prefix_cache_salt(None, Some([13; 32]));
    manager.set_host_offload_policy(Some(HostOffloadPolicy {
        budget_bytes: 4 << 20,
        high_watermark: 0.85,
        low_watermark: 0.70,
        max_in_flight_pages: 8,
        max_promotion_pages: 64,
    }));
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            12,
            32,
            &CacheCapability::Managed(test_contract()),
        )
        .expect("bind")
        .expect("runtime");
    let tokens = (0..353).collect::<Vec<u32>>();
    let watch = Arc::new(std::sync::atomic::AtomicU64::new(0));
    crate::backends::device_thread::watch_off_thread_ops(Some(watch.clone()));

    let source_session = SessionKey::new("b3-source".into(), 1);
    let source = manager
        .prepare(
            &runtime,
            51,
            &source_session,
            &sequence_work(0, tokens.len()),
            Some(&prefix_request(model, tokens.clone())),
        )
        .expect("source prepare")
        .expect("source reservation");
    manager
        .finalize(
            &source,
            Some(&source.completed_write_receipt_for_test()),
            true,
        )
        .expect("source commit");
    manager.release_session(&source_session).expect("release");
    let resumed_session = SessionKey::new("b3-resumed".into(), 1);
    let resumed = manager
        .prepare(
            &runtime,
            52,
            &resumed_session,
            &sequence_work(0, tokens.len()),
            Some(&prefix_request(model, tokens.clone())),
        )
        .expect("resumed prepare")
        .expect("resumed reservation");

    crate::backends::device_thread::watch_off_thread_ops(None);
    let telemetry = manager.telemetry_snapshot();
    assert!(telemetry.pages_zeroed > 0);
    assert_eq!(telemetry.demotions_total, 11);
    assert_eq!(resumed.domains[0].execution_start_tokens, 352);
    assert_eq!(watch.load(std::sync::atomic::Ordering::Relaxed), 0);
    manager.finalize(&resumed, None, false).expect("abort");
}

#[test]
fn demoted_prefix_pages_promote_back_into_a_preparing_transaction() {
    let model = ModelInstanceId::new(49);
    let mut manager = ManagedKvCacheManager::with_prefix_cache_salt(None, Some([12; 32]));
    manager.set_host_offload_policy(Some(HostOffloadPolicy {
        budget_bytes: 4 << 20,
        high_watermark: 0.85,
        low_watermark: 0.70,
        max_in_flight_pages: 8,
        max_promotion_pages: 64,
    }));
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            12,
            32,
            &CacheCapability::Managed(test_contract()),
        )
        .expect("bind")
        .expect("runtime");
    // 353 tokens: eleven complete published pages plus one private tail
    // token, and prompt+output fits the twelve-page arena's admission.
    let tokens = (0..353).collect::<Vec<u32>>();

    // Commit and release an eleven-page prefix: every published page keeps
    // only its durable prefix reference, pushing the arena over the
    // watermark (11/12 = 0.92).
    let source_session = SessionKey::new("ds4-source".into(), 1);
    let source = manager
        .prepare(
            &runtime,
            41,
            &source_session,
            &sequence_work(0, tokens.len()),
            Some(&prefix_request(model, tokens.clone())),
        )
        .expect("source prepare")
        .expect("source reservation");
    manager
        .finalize(
            &source,
            Some(&source.completed_write_receipt_for_test()),
            true,
        )
        .expect("source commit");
    manager.release_session(&source_session).expect("release");

    // The next prepare's offload tick crosses the high watermark and
    // demotes the whole chain; its own lookup then continues into the
    // host tier and restores the tail into its fresh pages.
    let resumed_session = SessionKey::new("ds4-resumed".into(), 1);
    let resumed = manager
        .prepare(
            &runtime,
            42,
            &resumed_session,
            &sequence_work(0, tokens.len()),
            Some(&prefix_request(model, tokens.clone())),
        )
        .expect("resumed prepare")
        .expect("resumed reservation");
    let telemetry = manager.telemetry_snapshot();
    assert_eq!(telemetry.demotions_total, 11, "whole chain demoted");
    assert_eq!(
        manager.runtime_snapshot().counters.kv_host_pages,
        11,
        "host pool holds the demoted chain"
    );
    // Lookups are counted once per prepare across both tiers: the source
    // prepare's cold miss, then this hit served entirely from the host
    // chain.
    assert_eq!(telemetry.prefix_misses, 1);
    assert_eq!(telemetry.prefix_hits, 1);
    assert_eq!(telemetry.prefix_hit_tokens, 352);
    assert_eq!(telemetry.reused_tokens, 352);
    assert_eq!(telemetry.prefix_query_tokens, 2 * (tokens.len() as u64 - 1));
    assert_eq!(
        resumed.domains[0].execution_start_tokens, 352,
        "execution resumes above the restored host tail"
    );
    assert_eq!(resumed.domains[0].writable_blocks.len(), 12);

    // Commit: the restored pages publish into the device index and every
    // host entry they replace is reclaimed (the reusable window covers
    // all eleven pages). The executor-visible write range starts at the
    // restored cursor, so the receipt carries the prefix-accepted shape.
    manager
        .finalize(
            &resumed,
            Some(
                &resumed
                    .completed_write_receipt_for_prefix_for_test(tokens.len() as u32, 32)
                    .expect("prefix receipt"),
            ),
            true,
        )
        .expect("resumed commit");
    let telemetry = manager.telemetry_snapshot();
    assert_eq!(telemetry.promotions_total, 11);
    assert!(
        telemetry.promotion_latency_ns_total > 0,
        "promotion latency recorded"
    );
    assert_eq!(
        manager.runtime_snapshot().counters.kv_host_pages,
        0,
        "host tier fully reclaimed at commit"
    );
    manager.release_session(&resumed_session).expect("release");

    // Releasing the committed session leaves its published pages
    // unreferenced over the watermark again: the cycle repeats and the
    // next request re-promotes them. An aborted reservation leaves the
    // host entries valid.
    let cycle_session = SessionKey::new("ds4-cycle".into(), 1);
    let cycle = manager
        .prepare(
            &runtime,
            43,
            &cycle_session,
            &sequence_work(0, tokens.len()),
            Some(&prefix_request(model, tokens.clone())),
        )
        .expect("cycle prepare")
        .expect("cycle reservation");
    let telemetry = manager.telemetry_snapshot();
    assert_eq!(telemetry.demotions_total, 22, "chain re-demoted");
    assert_eq!(
        manager.runtime_snapshot().counters.kv_host_pages,
        11,
        "re-demoted chain is host-resident again"
    );
    assert_eq!(cycle.domains[0].execution_start_tokens, 352);
    manager.finalize(&cycle, None, false).expect("abort");
    assert_eq!(
        manager.runtime_snapshot().counters.kv_host_pages,
        11,
        "abort keeps the host entries authoritative"
    );
}

#[test]
fn salted_prefix_reuse_attaches_shared_pages_and_skips_their_prefill() {
    let model = ModelInstanceId::new(45);
    let mut manager = ManagedKvCacheManager::with_prefix_cache_salt(None, Some([9; 32]));
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            4,
            32,
            &CacheCapability::Managed(test_contract()),
        )
        .expect("bind")
        .expect("runtime");
    let tokens = (0..65).collect::<Vec<u32>>();
    let first_request = prefix_request(model, tokens.clone());
    let first_session = SessionKey::new("prefix-first".into(), 1);
    let first = manager
        .prepare(
            &runtime,
            21,
            &first_session,
            &sequence_work(0, tokens.len()),
            Some(&first_request),
        )
        .expect("first prepare")
        .expect("first reservation");
    assert_eq!(first.domains[0].execution_start_tokens, 0);
    manager
        .finalize(
            &first,
            Some(&first.completed_write_receipt_for_test()),
            true,
        )
        .expect("first commit");
    manager
        .release_session(&first_session)
        .expect("first release");
    let retained = manager.runtime_snapshot();
    assert_eq!(retained.counters.prefix_retained_pages, 2);
    assert_eq!(
        retained.counters.prefix_retained_pages,
        retained.totals.coordinator.prefix_refs
    );

    let mut second_tokens = tokens;
    *second_tokens.last_mut().unwrap() = 999;
    let second_request = prefix_request(model, second_tokens.clone());
    let second_session = SessionKey::new("prefix-second".into(), 1);
    let second = manager
        .prepare(
            &runtime,
            22,
            &second_session,
            &sequence_work(0, second_tokens.len()),
            Some(&second_request),
        )
        .expect("second prepare")
        .expect("second reservation");
    assert_eq!(second.domains[0].execution_start_tokens, 64);
    assert_eq!(second.domains[0].writable_blocks.len(), 1);
    let telemetry = manager.telemetry_snapshot();
    assert_eq!(telemetry.prefix_hits, 1);
    assert_eq!(telemetry.reused_tokens, 64);
    assert_eq!(telemetry.avoided_prefill_tokens, 64);
    manager.finalize(&second, None, false).expect("abort");
    assert_eq!(manager.telemetry_snapshot().transaction_aborts, 1);
}

#[test]
fn prefix_reuse_is_scoped_to_the_trusted_tenant() {
    // H9: a worker serves many gateway tenants from one process salt. A
    // tenant must never hit (and so time) another tenant's prefix pages.
    let model = ModelInstanceId::new(453);
    let mut manager = ManagedKvCacheManager::with_prefix_cache_salt(None, Some([14; 32]));
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            8,
            32,
            &CacheCapability::Managed(test_contract()),
        )
        .expect("bind")
        .expect("runtime");
    let tokens = (0..65).collect::<Vec<u32>>();
    let tenant_request = |tenant: u8| {
        let mut request = prefix_request(model, tokens.clone());
        request.tenant_key = Some([tenant; 32]);
        request
    };
    let run = |manager: &mut ManagedKvCacheManager, txn_id, name: &str, tenant| {
        let session = SessionKey::new(name.into(), 1);
        let reservation = manager
            .prepare(
                &runtime,
                txn_id,
                &session,
                &sequence_work(0, tokens.len()),
                Some(&tenant_request(tenant)),
            )
            .expect("prepare")
            .expect("reservation");
        let start = reservation.domains[0].execution_start_tokens;
        manager
            .finalize(
                &reservation,
                Some(&reservation.completed_write_receipt_for_test()),
                true,
            )
            .expect("commit");
        manager.release_session(&session).expect("release");
        start
    };

    assert_eq!(run(&mut manager, 61, "tenant-a-first", 1), 0);
    assert_eq!(
        run(&mut manager, 62, "tenant-b", 2),
        0,
        "no cross-tenant hit"
    );
    assert_eq!(
        run(&mut manager, 63, "tenant-a-again", 1),
        64,
        "same tenant reuses"
    );
}

#[test]
fn host_offload_budget_is_shared_by_every_arena_of_one_model() {
    // H5: the lease is charged one budget per model; pools may not hold
    // more than that in total.
    assert_eq!(split_host_offload_budget(1_000, 1), 1_000);
    assert_eq!(split_host_offload_budget(1_000, 2), 500);
    assert_eq!(split_host_offload_budget(1_000, 3), 333);
    assert!(split_host_offload_budget(1_000, 3) * 3 <= 1_000);
    assert_eq!(split_host_offload_budget(1_000, 0), 1_000);
}

#[test]
fn full_prefill_rows_never_attach_shared_prefix_pages() {
    // H10: a full-prefill adapter runs its whole prompt from position 0.
    // Attaching published pages would hand it a non-empty cache and the
    // model start would reject the row (or the engine would replan it as a
    // partial quantum forever), so preparation must recompute instead.
    let model = ModelInstanceId::new(452);
    let mut manager = ManagedKvCacheManager::with_prefix_cache_salt(None, Some([11; 32]));
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            4,
            32,
            &CacheCapability::Managed(test_contract()),
        )
        .expect("bind")
        .expect("runtime");
    let tokens = (0..65).collect::<Vec<u32>>();
    let first_session = SessionKey::new("full-first".into(), 1);
    let first = manager
        .prepare(
            &runtime,
            31,
            &first_session,
            &sequence_work(0, tokens.len()),
            Some(&prefix_request(model, tokens.clone())),
        )
        .expect("first prepare")
        .expect("first reservation");
    manager
        .finalize(
            &first,
            Some(&first.completed_write_receipt_for_test()),
            true,
        )
        .expect("first commit");
    manager
        .release_session(&first_session)
        .expect("first release");

    let second_session = SessionKey::new("full-second".into(), 1);
    let second = manager
        .prepare_row(
            &runtime,
            32,
            &second_session,
            &sequence_work(0, tokens.len()),
            Some(&prefix_request(model, tokens.clone())),
            ManagedClaimPolicy::Full,
            false,
        )
        .expect("second prepare")
        .expect("second reservation");
    assert_eq!(second.domains[0].execution_start_tokens, 0);
    assert_eq!(manager.telemetry_snapshot().prefix_hits, 0);
    manager.finalize(&second, None, false).expect("abort");
}

#[test]
fn salted_prefix_reuse_is_bounded_to_the_first_incremental_prefill_target() {
    let model = ModelInstanceId::new(451);
    let mut manager = ManagedKvCacheManager::with_prefix_cache_salt(None, Some([10; 32]));
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            4,
            32,
            &CacheCapability::Managed(test_contract()),
        )
        .expect("bind")
        .expect("runtime");
    let tokens = (0..65).collect::<Vec<u32>>();
    let source_request = prefix_request(model, tokens.clone());
    let source_session = SessionKey::new("chunk-prefix-source".into(), 1);
    let source = manager
        .prepare(
            &runtime,
            211,
            &source_session,
            &sequence_work(0, tokens.len()),
            Some(&source_request),
        )
        .expect("source prepare")
        .expect("source reservation");
    manager
        .finalize(
            &source,
            Some(&source.completed_write_receipt_for_test()),
            true,
        )
        .expect("source commit");
    manager.release_session(&source_session).expect("release");

    let mut resumed_tokens = tokens;
    *resumed_tokens.last_mut().expect("last token") = 999;
    let resumed_request = prefix_request(model, resumed_tokens);
    let resumed_session = SessionKey::new("chunk-prefix-resumed".into(), 1);
    let first_chunk = manager
        .prepare(
            &runtime,
            212,
            &resumed_session,
            &sequence_work(0, 49),
            Some(&resumed_request),
        )
        .expect("chunk prepare")
        .expect("chunk reservation");

    assert_eq!(first_chunk.domains[0].target_committed_tokens, 49);
    assert_eq!(first_chunk.domains[0].execution_start_tokens, 32);
    assert!(first_chunk.domains[0].execution_start_tokens < 49);
    let telemetry = manager.telemetry_snapshot();
    assert_eq!(telemetry.prefix_hits, 1);
    assert_eq!(telemetry.reused_tokens, 32);
    manager.finalize(&first_chunk, None, false).expect("abort");
}

#[test]
fn managed_prefix_reuse_is_disabled_without_an_explicit_salt() {
    let model = ModelInstanceId::new(46);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            4,
            32,
            &CacheCapability::Managed(test_contract()),
        )
        .expect("bind")
        .expect("runtime");
    let request = prefix_request(model, (0..33).collect());
    let session = SessionKey::new("prefix-disabled".into(), 1);
    let reservation = manager
        .prepare(
            &runtime,
            23,
            &session,
            &sequence_work(0, 33),
            Some(&request),
        )
        .expect("prepare")
        .expect("reservation");
    assert_eq!(reservation.domains[0].execution_start_tokens, 0);
    assert_eq!(manager.telemetry_snapshot().prefix_hits, 0);
    assert_eq!(manager.telemetry_snapshot().prefix_misses, 0);
    assert_eq!(manager.telemetry_snapshot().prefix_rejections, 1);
}

#[test]
fn aborted_managed_prefill_never_publishes_prefix_pages() {
    let model = ModelInstanceId::new(47);
    let mut manager = ManagedKvCacheManager::with_prefix_cache_salt(None, Some([7; 32]));
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            3,
            32,
            &CacheCapability::Managed(test_contract()),
        )
        .expect("bind")
        .expect("runtime");
    let request = prefix_request(model, (100..133).collect());
    let first_session = SessionKey::new("aborted-prefix".into(), 1);
    let first = manager
        .prepare(
            &runtime,
            24,
            &first_session,
            &sequence_work(0, 33),
            Some(&request),
        )
        .expect("prepare")
        .expect("reservation");
    manager.finalize(&first, None, false).expect("abort");
    manager.release_session(&first_session).expect("release");

    let second_session = SessionKey::new("after-abort".into(), 1);
    let second = manager
        .prepare(
            &runtime,
            25,
            &second_session,
            &sequence_work(0, 33),
            Some(&request),
        )
        .expect("prepare after abort")
        .expect("reservation");
    assert_eq!(second.domains[0].execution_start_tokens, 0);
    let telemetry = manager.telemetry_snapshot();
    assert_eq!(telemetry.prefix_hits, 0);
    assert_eq!(telemetry.prefix_misses, 2);
}

#[test]
fn one_model_instance_cannot_change_contract_after_arena_allocation() {
    let model = ModelInstanceId::new(42);
    let mut manager = ManagedKvCacheManager::default();
    manager
        .bind_request(
            model,
            BackendKind::Cpu,
            2,
            16,
            &CacheCapability::Managed(test_contract()),
        )
        .expect("first binding");
    let mut changed = test_contract();
    if let StateDomainSpec::PagedAttention(domain) = &mut changed.domains[0] {
        domain.layers[0].kv_heads = 2;
    }
    assert!(
        manager
            .bind_request(
                model,
                BackendKind::Cpu,
                2,
                16,
                &CacheCapability::Managed(changed),
            )
            .is_err()
    );
}

#[test]
fn composite_domain_receipt_publishes_every_table_under_one_row_fence() {
    let model = ModelInstanceId::new(43);
    let session = SessionKey::new("managed-composite".to_string(), 3);
    let mut contract = test_contract();
    let mut second = contract.domains[0].clone();
    if let StateDomainSpec::PagedAttention(domain) = &mut second {
        domain.header.id = CacheDomainId::new(2);
    }
    contract.domains.push(second);
    contract.groups.push(StateGroupSpec {
        id: crate::kv::v2::StateGroupId::new(2),
        domains: vec![CacheDomainId::new(2)],
        prefix_shareable: true,
    });
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            2,
            16,
            &CacheCapability::Managed(contract),
        )
        .expect("bind")
        .expect("runtime");
    let reservation = manager
        .prepare(&runtime, 8, &session, &sequence_work(0, 8), None)
        .expect("prepare")
        .expect("reservation");
    assert_eq!(reservation.domains.len(), 2);
    manager
        .finalize(
            &reservation,
            Some(&reservation.completed_write_receipt_for_test()),
            true,
        )
        .expect("composite commit");
    for domain in [CacheDomainId::new(1), CacheDomainId::new(2)] {
        let snapshot = manager.snapshot(model, &session, domain).expect("snapshot");
        assert_eq!(snapshot.version, 1);
        assert_eq!(snapshot.committed_tokens, 8);
    }
}

#[test]
fn composite_domain_failure_publishes_no_table() {
    let model = ModelInstanceId::new(45);
    let session = SessionKey::new("managed-composite-failure".to_string(), 1);
    let mut contract = test_contract();
    let mut second = contract.domains[0].clone();
    if let StateDomainSpec::PagedAttention(domain) = &mut second {
        domain.header.id = CacheDomainId::new(2);
    }
    contract.domains.push(second);
    contract.groups.push(StateGroupSpec {
        id: crate::kv::v2::StateGroupId::new(2),
        domains: vec![CacheDomainId::new(2)],
        prefix_shareable: true,
    });
    let mut manager = ManagedKvCacheManager::with_prefix_cache_salt(None, Some([5; 32]));
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            4,
            16,
            &CacheCapability::Managed(contract),
        )
        .expect("bind")
        .expect("runtime");
    let request = prefix_request(model, (0..16).collect());
    let reservation = manager
        .prepare(&runtime, 9, &session, &sequence_work(0, 16), Some(&request))
        .expect("prepare")
        .expect("reservation");
    let receipt = reservation.completed_write_receipt_for_test();
    let state = manager.models.get_mut(&model).expect("model state");
    let pending = state
        .pending_prefixes
        .get_mut(&9)
        .expect("pending prefixes");
    assert_eq!(pending.len(), 2);
    assert!(!pending[1].publications.is_empty());
    pending[1].publications[0].block = reservation.domains[0].writable_blocks[0];

    assert!(
        manager
            .finalize(&reservation, Some(&receipt), true)
            .is_err()
    );
    for domain in [CacheDomainId::new(1), CacheDomainId::new(2)] {
        let snapshot = manager.snapshot(model, &session, domain).expect("snapshot");
        assert_eq!(snapshot.version, 0);
        assert_eq!(snapshot.committed_tokens, 0);
    }
    assert!(
        manager
            .runtime_snapshot()
            .models
            .iter()
            .flat_map(|model| &model.arenas)
            .all(|arena| arena.coordinator.active_transactions == 0)
    );
}

#[test]
fn paged_and_tensor_state_commit_or_abort_under_one_row_fence() {
    let model = ModelInstanceId::new(52);
    let session = SessionKey::new("managed-tensor-composite".into(), 1);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            2,
            16,
            &CacheCapability::Managed(composite_tensor_contract()),
        )
        .unwrap()
        .unwrap();
    let arena = runtime.tensor_state().unwrap().clone();
    let tensor_domain = crate::kv::v2::StateDomainId::new(2);
    let paged_bytes = runtime
        .plan()
        .groups
        .iter()
        .map(|group| group.bytes_per_page * u64::from(group.capacity_pages))
        .sum::<u64>();
    assert_eq!(
        runtime.physical_bytes(),
        paged_bytes + arena.capacity().authorized_bytes()
    );

    let aborted = manager
        .prepare(&runtime, 31, &session, &sequence_work(0, 1), None)
        .unwrap()
        .unwrap();
    let sequence =
        PhysicalStateSequenceId::new(aborted.clocked_state.as_ref().unwrap().sequence())
            .unwrap();
    let transaction = PhysicalStateTransactionId::new(aborted.txn_id).unwrap();
    arena
        .stage_replace(
            transaction,
            tensor_domain,
            0,
            1,
            vec![StateComponentValue {
                component: crate::kv::v2::StateComponentId::new(1),
                tensor: Some(Tensor::from_slice(&[1.0_f32], 1, &Device::Cpu).unwrap()),
            }],
        )
        .unwrap();
    manager.finalize(&aborted, None, false).unwrap();
    let next_sequence_after_first_prepare = manager.next_tensor_sequence;
    assert!(arena.read(sequence, tensor_domain).unwrap().is_none());
    assert_eq!(
        manager
            .snapshot(model, &session, CacheDomainId::new(1))
            .unwrap()
            .committed_tokens,
        0
    );

    let committed = manager
        .prepare(&runtime, 32, &session, &sequence_work(0, 1), None)
        .unwrap()
        .unwrap();
    assert_eq!(
        manager.next_tensor_sequence, next_sequence_after_first_prepare,
        "an existing session must not consume another tensor sequence identity"
    );
    arena
        .stage_replace(
            PhysicalStateTransactionId::new(committed.txn_id).unwrap(),
            tensor_domain,
            0,
            1,
            vec![StateComponentValue {
                component: crate::kv::v2::StateComponentId::new(1),
                tensor: Some(Tensor::from_slice(&[2.0_f32], 1, &Device::Cpu).unwrap()),
            }],
        )
        .unwrap();
    manager
        .finalize(
            &committed,
            Some(&committed.completed_write_receipt_for_test()),
            true,
        )
        .unwrap();
    assert_eq!(
        arena
            .read(sequence, tensor_domain)
            .unwrap()
            .unwrap()
            .components[0]
            .tensor
            .as_ref()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        vec![2.0]
    );
    assert_eq!(
        manager
            .snapshot(model, &session, CacheDomainId::new(1))
            .unwrap()
            .committed_tokens,
        1
    );
}

#[test]
fn selected_clocked_state_commits_with_paged_state_under_one_fence() {
    let model = ModelInstanceId::new(5210);
    let session = SessionKey::new("managed-selected-clock".into(), 1);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            8,
            16,
            &CacheCapability::Managed(independently_clocked_tensor_contract()),
        )
        .unwrap()
        .unwrap();
    let arena = runtime.tensor_state().unwrap().clone();
    let reservation = manager
        .prepare(
            &runtime,
            5211,
            &session,
            &selected_sequence_work(
                0,
                1,
                StateGroupId::new(2),
                StateClock::AudioFrames,
                0,
                160,
            ),
            None,
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        reservation
            .clocked_state
            .as_ref()
            .unwrap()
            .selections()
            .unwrap()[0]
            .target_cursor,
        160
    );
    let transaction = PhysicalStateTransactionId::new(reservation.txn_id).unwrap();
    arena
        .stage_replace(
            transaction,
            StateDomainId::new(2),
            0,
            160,
            vec![StateComponentValue {
                component: StateComponentId::new(1),
                tensor: None,
            }],
        )
        .unwrap();
    let completion = arena.seal_selected_completion(transaction).unwrap();
    let receipt = reservation
        .completed_write_receipt_for_test()
        .with_clocked_state_completion(completion)
        .unwrap();
    manager
        .finalize(&reservation, Some(&receipt), true)
        .unwrap();
    let sequence =
        PhysicalStateSequenceId::new(reservation.clocked_state.as_ref().unwrap().sequence())
            .unwrap();
    assert_eq!(
        arena
            .read(sequence, StateDomainId::new(2))
            .unwrap()
            .unwrap()
            .cursor,
        160
    );
    assert_eq!(arena.occupancy().unwrap().active_transactions, 0);
}

#[test]
fn missing_selected_completion_aborts_paged_and_tensor_state() {
    let model = ModelInstanceId::new(5220);
    let session = SessionKey::new("managed-missing-clock-proof".into(), 1);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            8,
            16,
            &CacheCapability::Managed(independently_clocked_tensor_contract()),
        )
        .unwrap()
        .unwrap();
    let arena = runtime.tensor_state().unwrap().clone();
    let reservation = manager
        .prepare(
            &runtime,
            5221,
            &session,
            &selected_sequence_work(
                0,
                1,
                StateGroupId::new(2),
                StateClock::AudioFrames,
                0,
                160,
            ),
            None,
        )
        .unwrap()
        .unwrap();
    let transaction = PhysicalStateTransactionId::new(reservation.txn_id).unwrap();
    arena
        .stage_replace(
            transaction,
            StateDomainId::new(2),
            0,
            160,
            vec![StateComponentValue {
                component: StateComponentId::new(1),
                tensor: None,
            }],
        )
        .unwrap();
    let receipt = reservation.completed_write_receipt_for_test();
    assert!(
        manager
            .finalize(&reservation, Some(&receipt), true)
            .is_err()
    );
    assert_eq!(arena.occupancy().unwrap().active_transactions, 0);
    let sequence =
        PhysicalStateSequenceId::new(reservation.clocked_state.as_ref().unwrap().sequence())
            .unwrap();
    assert!(
        arena
            .read(sequence, StateDomainId::new(2))
            .unwrap()
            .is_none()
    );
    assert!(
        manager
            .runtime_snapshot()
            .models
            .iter()
            .flat_map(|model| &model.arenas)
            .all(|arena| arena.coordinator.active_transactions == 0)
    );
}

#[test]
fn explicit_empty_clock_selection_does_not_open_tensor_transaction() {
    let model = ModelInstanceId::new(5230);
    let session = SessionKey::new("managed-no-clocked-work".into(), 1);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            8,
            16,
            &CacheCapability::Managed(composite_tensor_contract()),
        )
        .unwrap()
        .unwrap();
    let arena = runtime.tensor_state().unwrap().clone();
    let work = WorkUnit::SequenceStep {
        phase: SequencePhase::Prefill,
        input: InputRange { start: 0, end: 1 },
        max_output_steps: 1,
        auxiliary_state: Some(Arc::from([])),
    };
    let reservation = manager
        .prepare(&runtime, 5231, &session, &work, None)
        .unwrap()
        .unwrap();
    assert!(reservation.clocked_state.is_none());
    assert_eq!(arena.occupancy().unwrap().active_transactions, 0);
    manager.finalize(&reservation, None, false).unwrap();
}

#[test]
fn selected_clocked_state_can_commit_without_a_paged_append() {
    let model = ModelInstanceId::new(5240);
    let session = SessionKey::new("managed-clocked-only".into(), 1);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            8,
            16,
            &CacheCapability::Managed(independently_clocked_tensor_contract()),
        )
        .unwrap()
        .unwrap();
    let arena = runtime.tensor_state().unwrap().clone();
    let reservation = manager
        .prepare(
            &runtime,
            5241,
            &session,
            &selected_sequence_work(0, 0, StateGroupId::new(2), StateClock::AudioFrames, 0, 80),
            None,
        )
        .unwrap()
        .unwrap();
    assert!(
        reservation
            .domains
            .iter()
            .all(|domain| { domain.target_committed_tokens == domain.execution_start_tokens })
    );
    let transaction = PhysicalStateTransactionId::new(reservation.txn_id).unwrap();
    arena
        .stage_replace(
            transaction,
            StateDomainId::new(2),
            0,
            80,
            vec![StateComponentValue {
                component: StateComponentId::new(1),
                tensor: None,
            }],
        )
        .unwrap();
    let receipt = reservation
        .completed_write_receipt_for_test()
        .with_clocked_state_completion(arena.seal_selected_completion(transaction).unwrap())
        .unwrap();
    manager
        .finalize(&reservation, Some(&receipt), true)
        .unwrap();
    let sequence =
        PhysicalStateSequenceId::new(reservation.clocked_state.as_ref().unwrap().sequence())
            .unwrap();
    assert_eq!(
        arena
            .read(sequence, StateDomainId::new(2))
            .unwrap()
            .unwrap()
            .cursor,
        80
    );
}

#[test]
fn realtime_selected_tensor_state_commits_without_implicit_paged_advance() {
    let model = ModelInstanceId::new(5242);
    let session = SessionKey::new("managed-realtime-clocked-only".into(), 1);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            8,
            16,
            &CacheCapability::Managed(independently_clocked_tensor_contract()),
        )
        .unwrap()
        .unwrap();
    let arena = runtime.tensor_state().unwrap().clone();
    let span = ClockedStateSpan::new(
        StateGroupId::new(2),
        StateClock::AudioFrames,
        InputRange::new(0, 1).unwrap(),
    )
    .unwrap();
    let work = WorkUnit::RealtimePreparation {
        operation_id: crate::engine::RealtimeOperationId::new(1),
        mode: crate::engine::RealtimePreparationMode::Push,
        input: InputRange::new(0, 160).unwrap(),
        max_output_steps: 2,
        max_cache_append: 4,
        retained_state_input: InputRange::new(0, 1).unwrap(),
        auxiliary_state: Some(Arc::from([span])),
    };
    let reservation = manager
        .prepare(&runtime, 5243, &session, &work, None)
        .unwrap()
        .unwrap();
    assert!(
        reservation
            .domains
            .iter()
            .all(|domain| { domain.target_committed_tokens == domain.execution_start_tokens })
    );
    let transaction = PhysicalStateTransactionId::new(reservation.txn_id).unwrap();
    arena
        .stage_replace(
            transaction,
            StateDomainId::new(2),
            0,
            1,
            vec![StateComponentValue {
                component: StateComponentId::new(1),
                tensor: None,
            }],
        )
        .unwrap();
    let receipt = reservation
        .completed_write_receipt_for_test()
        .with_clocked_state_completion(arena.seal_selected_completion(transaction).unwrap())
        .unwrap();
    manager
        .finalize(&reservation, Some(&receipt), true)
        .unwrap();
    let sequence =
        PhysicalStateSequenceId::new(reservation.clocked_state.as_ref().unwrap().sequence())
            .unwrap();
    assert_eq!(
        arena
            .read(sequence, StateDomainId::new(2))
            .unwrap()
            .unwrap()
            .cursor,
        1
    );
}

#[test]
fn accepted_prefix_reconciles_two_paged_domains_and_tensor_cursor() {
    const MAX_RESERVED: u32 = 9;

    for accepted in 1..=MAX_RESERVED {
        let model = ModelInstanceId::new(520 + u64::from(accepted));
        let session = SessionKey::new(format!("managed-prefix-{accepted}"), 1);
        let mut manager = ManagedKvCacheManager::default();
        let runtime = manager
            .bind_request(
                model,
                BackendKind::Cpu,
                8,
                8,
                &CacheCapability::Managed(two_paged_tensor_contract()),
            )
            .unwrap()
            .unwrap();
        let page_tokens = runtime.plan().groups[0].page_tokens;
        assert!(
            runtime
                .plan()
                .groups
                .iter()
                .all(|group| group.page_tokens == page_tokens)
        );
        let tensor_arena = runtime.tensor_state().unwrap().clone();
        let reservation = manager
            .prepare(
                &runtime,
                500 + u64::from(accepted),
                &session,
                &sequence_work(0, MAX_RESERVED as usize),
                None,
            )
            .unwrap()
            .unwrap();
        assert_eq!(reservation.domains.len(), 2);
        let sequence = PhysicalStateSequenceId::new(
            reservation
                .clocked_state
                .as_ref()
                .expect("tensor reservation")
                .sequence(),
        )
        .unwrap();
        tensor_arena
            .stage_replace(
                PhysicalStateTransactionId::new(reservation.txn_id).unwrap(),
                StateDomainId::new(2),
                0,
                u64::from(accepted),
                vec![StateComponentValue {
                    component: StateComponentId::new(1),
                    tensor: Some(
                        Tensor::from_slice(&[accepted as f32; 4], 4, &Device::Cpu).unwrap(),
                    ),
                }],
            )
            .unwrap();
        let receipt = reservation
            .completed_write_receipt_for_prefix_for_test(accepted, page_tokens)
            .unwrap();
        assert_eq!(receipt.accepted_prefix(), Some(accepted));
        manager
            .finalize(&reservation, Some(&receipt), true)
            .unwrap();

        for domain in [CacheDomainId::new(1), CacheDomainId::new(3)] {
            let snapshot = manager.snapshot(model, &session, domain).unwrap();
            assert_eq!(snapshot.committed_tokens, accepted);
            assert_eq!(snapshot.version, 1);
            assert_eq!(
                snapshot.groups[0].blocks.len(),
                accepted.div_ceil(page_tokens) as usize
            );
        }
        let tensor = tensor_arena
            .read(sequence, StateDomainId::new(2))
            .unwrap()
            .unwrap();
        assert_eq!(tensor.cursor, u64::from(accepted));
        assert_eq!(
            tensor.components[0]
                .tensor
                .as_ref()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![accepted as f32; 4]
        );
        let state = manager.models.get(&model).unwrap();
        for coordinator in state.coordinators.values() {
            assert_eq!(coordinator.stats().active_transactions, 0);
            assert_eq!(
                coordinator.stats().allocated_pages,
                accepted.div_ceil(page_tokens) as usize
            );
            coordinator.check_invariants().unwrap();
        }
    }
}

fn hybrid_tensor_values(sequence_values: &[f32]) -> Vec<StateComponentValue> {
    vec![StateComponentValue {
        component: StateComponentId::new(1),
        tensor: Some(
            Tensor::from_slice(sequence_values, sequence_values.len(), &Device::Cpu).unwrap(),
        ),
    }]
}

fn read_hybrid_tensor(
    arena: &crate::backends::state::TensorStateArena,
    sequence: u64,
) -> (u64, Vec<f32>) {
    let snapshot = arena
        .read(
            PhysicalStateSequenceId::new(sequence).unwrap(),
            StateDomainId::new(2),
        )
        .unwrap()
        .expect("tensor domain state");
    let values = snapshot.components[0]
        .tensor
        .as_ref()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    (snapshot.cursor, values)
}

#[test]
fn tensor_snapshot_publishes_on_aligned_commit_and_forks_attaching_state() {
    let model = ModelInstanceId::new(601);
    let mut manager = ManagedKvCacheManager::with_prefix_cache_salt(None, Some([21; 32]));
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            12,
            32,
            &CacheCapability::Managed(hybrid_snapshot_contract()),
        )
        .unwrap()
        .unwrap();
    let arena = runtime.tensor_state().unwrap().clone();

    // Session A commits an aligned 64-token prompt with tensor state.
    let tokens = (0..64_u32).collect::<Vec<u32>>();
    let session_a = SessionKey::new("snapshot-first".into(), 1);
    let request_a = prefix_request(model, tokens.clone());
    let first = manager
        .prepare(
            &runtime,
            61,
            &session_a,
            &sequence_work(0, 64),
            Some(&request_a),
        )
        .unwrap()
        .unwrap();
    assert_eq!(first.domains[0].target_committed_tokens, 64);
    arena
        .stage_replace(
            PhysicalStateTransactionId::new(61).unwrap(),
            StateDomainId::new(2),
            0,
            64,
            hybrid_tensor_values(&[7.0; 4]),
        )
        .unwrap();
    manager
        .finalize(
            &first,
            Some(&first.completed_write_receipt_for_test()),
            true,
        )
        .unwrap();
    let sharing = manager.models[&model]
        .tensor_snapshots
        .as_ref()
        .expect("hybrid contract shares snapshots");
    assert_eq!(sharing.index.len(), 1);
    assert_eq!(manager.telemetry_snapshot().tensor_snapshot_publishes, 1);

    // Session B shares the first 64 tokens and forks the snapshot.
    let mut longer = tokens;
    longer.extend([777, 778, 779]);
    let session_b = SessionKey::new("snapshot-second".into(), 1);
    let request_b = prefix_request(model, longer);
    let second = manager
        .prepare(
            &runtime,
            62,
            &session_b,
            &sequence_work(0, 67),
            Some(&request_b),
        )
        .unwrap()
        .unwrap();
    assert_eq!(second.domains[0].execution_start_tokens, 64);
    let telemetry = manager.telemetry_snapshot();
    assert_eq!(telemetry.tensor_snapshot_attaches, 1);
    assert_eq!(telemetry.tensor_snapshot_truncations, 0);
    assert_eq!(telemetry.reused_tokens, 64);
    let sequence_b = second
        .clocked_state
        .as_ref()
        .expect("tensor reservation")
        .sequence();
    let (cursor, values) = read_hybrid_tensor(&arena, sequence_b);
    assert_eq!(cursor, 64);
    assert_eq!(values, vec![7.0; 4], "fork seeds the committed values");

    // Session A may keep advancing; the fork and the published snapshot
    // hold their own component handles.
    let continuation = manager
        .prepare(
            &runtime,
            63,
            &session_a,
            &sequence_work(64, 72),
            Some(&request_a),
        )
        .unwrap()
        .unwrap();
    arena
        .stage_replace(
            PhysicalStateTransactionId::new(63).unwrap(),
            StateDomainId::new(2),
            64,
            72,
            hybrid_tensor_values(&[5.0; 4]),
        )
        .unwrap();
    manager
        .finalize(
            &continuation,
            Some(&continuation.completed_write_receipt_for_test()),
            true,
        )
        .unwrap();
    let (cursor_b, values_b) = read_hybrid_tensor(&arena, sequence_b);
    assert_eq!(cursor_b, 64);
    assert_eq!(values_b, vec![7.0; 4], "fork is isolated from the source");

    // Session B advances from the fork cursor and commits its own span.
    arena
        .stage_replace(
            PhysicalStateTransactionId::new(62).unwrap(),
            StateDomainId::new(2),
            64,
            67,
            hybrid_tensor_values(&[9.0; 4]),
        )
        .unwrap();
    let receipt = second
        .completed_write_receipt_for_prefix_for_test(67, 32)
        .unwrap();
    manager.finalize(&second, Some(&receipt), true).unwrap();
    let (cursor_b, values_b) = read_hybrid_tensor(&arena, sequence_b);
    assert_eq!(cursor_b, 67);
    assert_eq!(values_b, vec![9.0; 4]);
    manager.models[&model]
        .coordinators
        .values()
        .for_each(|coordinator| coordinator.check_invariants().unwrap());
}

/// Phase 5a: a request declaring its stable-history boundary publishes
/// pages from every prefill span and a tensor snapshot only on the
/// boundary's interval floor, so each conversation turn attaches at the
/// previous turn's boundary and the chain keeps growing.
#[test]
fn declared_history_boundary_advances_the_hybrid_chain_across_turns() {
    let model = ModelInstanceId::new(611);
    let mut manager = ManagedKvCacheManager::with_prefix_cache_salt(None, Some([31; 32]));
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            12,
            32,
            &CacheCapability::Managed(hybrid_snapshot_contract()),
        )
        .unwrap()
        .unwrap();
    let arena = runtime.tensor_state().unwrap().clone();
    let request = |tokens: Vec<u32>, boundary: u32| {
        let mut request = prefix_request(model, tokens);
        request.set_managed_snapshot_boundary_for_test(Some(boundary));
        request
    };
    let mut txn = 700_u64;
    let mut commit = |manager: &mut ManagedKvCacheManager,
                      session: &SessionKey,
                      request: &EngineCoreRequest,
                      start: usize,
                      end: usize,
                      value: f32|
     -> u32 {
        txn += 1;
        let reservation = manager
            .prepare(
                &runtime,
                txn,
                session,
                &sequence_work(start, end),
                Some(request),
            )
            .unwrap()
            .unwrap();
        let execution_start = reservation.domains[0].execution_start_tokens;
        let target = reservation.domains[0].target_committed_tokens;
        arena
            .stage_replace(
                PhysicalStateTransactionId::new(txn).unwrap(),
                StateDomainId::new(2),
                u64::from(execution_start),
                u64::from(target),
                hybrid_tensor_values(&[value; 4]),
            )
            .unwrap();
        let receipt = if execution_start == start as u32 {
            reservation.completed_write_receipt_for_test()
        } else {
            reservation
                .completed_write_receipt_for_prefix_for_test(target, 32)
                .unwrap()
        };
        manager
            .finalize(&reservation, Some(&receipt), true)
            .unwrap();
        execution_start
    };

    // Turn 1: 100 prompt tokens, stable history 70 (interval floor 64).
    // The scheduler ends the first span on the floor.
    let turn1 = (0..100_u32).collect::<Vec<_>>();
    let request1 = request(turn1.clone(), 70);
    let session1 = SessionKey::new("turn-1".into(), 1);
    assert_eq!(commit(&mut manager, &session1, &request1, 0, 64, 1.0), 0);
    assert_eq!(manager.telemetry_snapshot().tensor_snapshot_publishes, 1);
    // The later span publishes its pages but no snapshot.
    assert_eq!(commit(&mut manager, &session1, &request1, 64, 100, 2.0), 64);
    assert_eq!(manager.telemetry_snapshot().tensor_snapshot_publishes, 1);
    // The turn finishes; its published pages and snapshot stay shared.
    manager.release_session(&session1).unwrap();

    // Turn 2 shares only turn 1's first 70 tokens (the reasoning of an
    // answer is not re-rendered); its own history ends at 110 (floor 96).
    let mut turn2 = turn1[..70].to_vec();
    turn2.extend(500..560_u32);
    let request2 = request(turn2.clone(), 110);
    let session2 = SessionKey::new("turn-2".into(), 1);
    assert_eq!(
        commit(&mut manager, &session2, &request2, 0, 96, 3.0),
        64,
        "turn 2 attaches at turn 1's boundary"
    );
    let sequence2 = manager.telemetry_snapshot().tensor_snapshot_attaches;
    assert_eq!(sequence2, 1);
    assert_eq!(manager.telemetry_snapshot().tensor_snapshot_publishes, 2);
    // An aligned later span (96..128) is not its boundary: no snapshot.
    assert_eq!(commit(&mut manager, &session2, &request2, 96, 128, 4.0), 96);
    assert_eq!(
        commit(&mut manager, &session2, &request2, 128, 130, 5.0),
        128
    );
    assert_eq!(manager.telemetry_snapshot().tensor_snapshot_publishes, 2);
    manager.release_session(&session2).unwrap();

    // Turn 3 attaches at turn 2's boundary: pages 0-1 from turn 1, page
    // 2 from turn 2, and the snapshot at 96.
    let mut turn3 = turn2[..110].to_vec();
    turn3.extend(900..940_u32);
    let request3 = request(turn3, 140);
    let session3 = SessionKey::new("turn-3".into(), 1);
    assert_eq!(
        commit(&mut manager, &session3, &request3, 0, 128, 6.0),
        96,
        "turn 3 attaches at turn 2's boundary"
    );
    let telemetry = manager.telemetry_snapshot();
    assert_eq!(telemetry.tensor_snapshot_attaches, 2);
    assert_eq!(telemetry.reused_tokens, 64 + 96);
    assert_eq!(
        telemetry.tensor_snapshot_publishes, 3,
        "turn 3 snapshots at 128"
    );
    manager.models[&model]
        .coordinators
        .values()
        .for_each(|coordinator| coordinator.check_invariants().unwrap());
}

/// Without a declared boundary nothing changes: later spans publish no
/// pages, so a second request cannot attach past the first chunk.
#[test]
fn undeclared_requests_keep_first_span_only_publication() {
    let model = ModelInstanceId::new(612);
    let mut manager = ManagedKvCacheManager::with_prefix_cache_salt(None, Some([32; 32]));
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            12,
            32,
            &CacheCapability::Managed(hybrid_snapshot_contract()),
        )
        .unwrap()
        .unwrap();
    let tokens = (0..128_u32).collect::<Vec<_>>();
    let first = prefix_request(model, tokens.clone());
    let session = SessionKey::new("undeclared".into(), 1);
    let arena = runtime.tensor_state().unwrap().clone();
    for (txn, (start, end)) in [(801_u64, (0, 64)), (802, (64, 128))] {
        let reservation = manager
            .prepare(
                &runtime,
                txn,
                &session,
                &sequence_work(start, end),
                Some(&first),
            )
            .unwrap()
            .unwrap();
        arena
            .stage_replace(
                PhysicalStateTransactionId::new(txn).unwrap(),
                StateDomainId::new(2),
                start as u64,
                end as u64,
                hybrid_tensor_values(&[1.0; 4]),
            )
            .unwrap();
        manager
            .finalize(
                &reservation,
                Some(&reservation.completed_write_receipt_for_test()),
                true,
            )
            .unwrap();
    }
    let mut longer = tokens;
    longer.extend([1, 2, 3]);
    let second = prefix_request(model, longer);
    let reservation = manager
        .prepare(
            &runtime,
            803,
            &SessionKey::new("undeclared-2".into(), 1),
            &sequence_work(0, 131),
            Some(&second),
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        reservation.domains[0].execution_start_tokens, 64,
        "only the first span's pages and snapshot are shared"
    );
}

#[test]
fn tensor_snapshot_attach_truncates_to_zero_without_a_published_boundary() {
    let model = ModelInstanceId::new(602);
    let mut manager = ManagedKvCacheManager::with_prefix_cache_salt(None, Some([22; 32]));
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            12,
            32,
            &CacheCapability::Managed(hybrid_snapshot_contract()),
        )
        .unwrap()
        .unwrap();
    let arena = runtime.tensor_state().unwrap().clone();

    let tokens = (0..64_u32).collect::<Vec<u32>>();
    let session_a = SessionKey::new("truncate-first".into(), 1);
    let request_a = prefix_request(model, tokens.clone());
    let first = manager
        .prepare(
            &runtime,
            64,
            &session_a,
            &sequence_work(0, 64),
            Some(&request_a),
        )
        .unwrap()
        .unwrap();
    arena
        .stage_replace(
            PhysicalStateTransactionId::new(64).unwrap(),
            StateDomainId::new(2),
            0,
            64,
            hybrid_tensor_values(&[7.0; 4]),
        )
        .unwrap();
    manager
        .finalize(
            &first,
            Some(&first.completed_write_receipt_for_test()),
            true,
        )
        .unwrap();

    // Only the boundary at 64 was published; a request sharing just the
    // first page cannot fork and must not attach paged pages either.
    let diverging = {
        let mut tokens = tokens;
        tokens.truncate(32);
        tokens.extend([900, 901, 902, 903]);
        tokens
    };
    let session_b = SessionKey::new("truncate-second".into(), 1);
    let request_b = prefix_request(model, diverging);
    let second = manager
        .prepare(
            &runtime,
            65,
            &session_b,
            &sequence_work(0, 36),
            Some(&request_b),
        )
        .unwrap()
        .unwrap();
    assert_eq!(second.domains[0].execution_start_tokens, 0);
    let telemetry = manager.telemetry_snapshot();
    assert_eq!(telemetry.tensor_snapshot_attaches, 0);
    assert_eq!(telemetry.tensor_snapshot_truncations, 1);
    manager.finalize(&second, None, false).unwrap();
}

#[test]
fn tensor_snapshot_publication_requires_the_tenant_salt() {
    let model = ModelInstanceId::new(603);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            4,
            32,
            &CacheCapability::Managed(hybrid_snapshot_contract()),
        )
        .unwrap()
        .unwrap();
    let arena = runtime.tensor_state().unwrap().clone();
    let tokens = (0..64_u32).collect::<Vec<u32>>();
    let session = SessionKey::new("unsalted".into(), 1);
    let request = prefix_request(model, tokens);
    let reservation = manager
        .prepare(
            &runtime,
            66,
            &session,
            &sequence_work(0, 64),
            Some(&request),
        )
        .unwrap()
        .unwrap();
    arena
        .stage_replace(
            PhysicalStateTransactionId::new(66).unwrap(),
            StateDomainId::new(2),
            0,
            64,
            hybrid_tensor_values(&[7.0; 4]),
        )
        .unwrap();
    manager
        .finalize(
            &reservation,
            Some(&reservation.completed_write_receipt_for_test()),
            true,
        )
        .unwrap();
    let sharing = manager.models[&model]
        .tensor_snapshots
        .as_ref()
        .expect("sharing surface exists for the contract");
    assert_eq!(sharing.index.len(), 0, "no salt publishes nothing");
    assert_eq!(manager.telemetry_snapshot().tensor_snapshot_publishes, 0);
    assert_eq!(manager.telemetry_snapshot().prefix_rejections, 1);
}

#[test]
fn aborted_hybrid_commit_publishes_no_tensor_snapshot() {
    let model = ModelInstanceId::new(604);
    let mut manager = ManagedKvCacheManager::with_prefix_cache_salt(None, Some([23; 32]));
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            4,
            32,
            &CacheCapability::Managed(hybrid_snapshot_contract()),
        )
        .unwrap()
        .unwrap();
    let arena = runtime.tensor_state().unwrap().clone();
    let tokens = (0..64_u32).collect::<Vec<u32>>();
    let session = SessionKey::new("aborted".into(), 1);
    let request = prefix_request(model, tokens);
    let reservation = manager
        .prepare(
            &runtime,
            67,
            &session,
            &sequence_work(0, 64),
            Some(&request),
        )
        .unwrap()
        .unwrap();
    arena
        .stage_replace(
            PhysicalStateTransactionId::new(67).unwrap(),
            StateDomainId::new(2),
            0,
            64,
            hybrid_tensor_values(&[7.0; 4]),
        )
        .unwrap();
    manager.finalize(&reservation, None, false).unwrap();
    let sharing = manager.models[&model]
        .tensor_snapshots
        .as_ref()
        .expect("sharing surface exists for the contract");
    assert_eq!(sharing.index.len(), 0);
    assert_eq!(manager.telemetry_snapshot().tensor_snapshot_publishes, 0);
    assert_eq!(manager.telemetry_snapshot().transaction_aborts, 1);
}

#[test]
fn hybrid_first_chunk_target_aligns_to_the_snapshot_interval() {
    let model = ModelInstanceId::new(605);
    let mut manager = ManagedKvCacheManager::with_prefix_cache_salt(None, Some([24; 32]));
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            4,
            32,
            &CacheCapability::Managed(hybrid_snapshot_contract()),
        )
        .unwrap()
        .unwrap();
    let arena = runtime.tensor_state().unwrap().clone();
    // The scheduler's first-chunk alignment clips the executed span to the
    // snapshot boundary (64 of the 65-token prompt); prepare must accept
    // that span and publish at the aligned cursor.
    let tokens = (0..65_u32).collect::<Vec<u32>>();
    let session = SessionKey::new("aligned".into(), 1);
    let request = prefix_request(model, tokens);
    let reservation = manager
        .prepare(
            &runtime,
            68,
            &session,
            &sequence_work(0, 64),
            Some(&request),
        )
        .unwrap()
        .unwrap();
    assert_eq!(reservation.domains[0].target_committed_tokens, 64);
    arena
        .stage_replace(
            PhysicalStateTransactionId::new(68).unwrap(),
            StateDomainId::new(2),
            0,
            64,
            hybrid_tensor_values(&[7.0; 4]),
        )
        .unwrap();
    let receipt = reservation
        .completed_write_receipt_for_prefix_for_test(64, 32)
        .unwrap();
    manager
        .finalize(&reservation, Some(&receipt), true)
        .unwrap();
    let sharing = manager.models[&model]
        .tensor_snapshots
        .as_ref()
        .expect("sharing surface exists for the contract");
    assert_eq!(sharing.index.len(), 1);
    assert_eq!(manager.telemetry_snapshot().tensor_snapshot_publishes, 1);
}

#[test]
fn hybrid_commit_target_never_falls_below_the_scheduled_span() {
    // A span the scheduler did not align (65 tokens, interval 64) must
    // commit in full: truncating the commit target below the appended span
    // would leave the model writing pages past its block table.
    let model = ModelInstanceId::new(607);
    let mut manager = ManagedKvCacheManager::with_prefix_cache_salt(None, Some([26; 32]));
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            4,
            32,
            &CacheCapability::Managed(hybrid_snapshot_contract()),
        )
        .unwrap()
        .unwrap();
    let tokens = (0..65_u32).collect::<Vec<u32>>();
    let session = SessionKey::new("unaligned".into(), 1);
    let request = prefix_request(model, tokens);
    let reservation = manager
        .prepare(
            &runtime,
            69,
            &session,
            &sequence_work(0, 65),
            Some(&request),
        )
        .unwrap()
        .unwrap();
    assert_eq!(reservation.domains[0].target_committed_tokens, 65);
    assert_eq!(manager.telemetry_snapshot().tensor_snapshot_publishes, 0);
}

#[test]
fn snapshot_declarations_stay_inert_for_non_sharing_contracts() {
    let model = ModelInstanceId::new(606);
    let mut manager = ManagedKvCacheManager::with_prefix_cache_salt(None, Some([25; 32]));
    // The composite fixture keeps its tensor prefix disabled, so the gate
    // rejects it and nothing clamps or publishes.
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            4,
            32,
            &CacheCapability::Managed(composite_tensor_contract()),
        )
        .unwrap()
        .unwrap();
    assert!(manager.models[&model].tensor_snapshots.is_none());
    let tokens = (0..65_u32).collect::<Vec<u32>>();
    let session = SessionKey::new("inert".into(), 1);
    let request = prefix_request(model, tokens);
    let reservation = manager
        .prepare(
            &runtime,
            69,
            &session,
            &sequence_work(0, 65),
            Some(&request),
        )
        .unwrap()
        .unwrap();
    assert_eq!(reservation.domains[0].target_committed_tokens, 65);
    manager.finalize(&reservation, None, false).unwrap();
}

#[test]
fn realtime_relative_reservation_can_commit_an_unchanged_prefix() {
    let model = ModelInstanceId::new(777);
    let session = SessionKey::new("managed-realtime-zero-append".into(), 1);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            8,
            8,
            &CacheCapability::Managed(test_contract()),
        )
        .unwrap()
        .unwrap();
    let work = WorkUnit::RealtimePush {
        operation_id: crate::engine::RealtimeOperationId::new(1),
        input: crate::engine::InputRange::new(0, 160).unwrap(),
        max_output_steps: 2,
        max_cache_append: 4,
    };
    let reservation = manager
        .prepare(&runtime, 7771, &session, &work, None)
        .unwrap()
        .unwrap();
    assert!(reservation.allow_unchanged_prefix);
    assert!(reservation.domains.iter().all(|domain| {
        domain.execution_start_tokens == 0 && domain.target_committed_tokens == 4
    }));

    let receipt = reservation
        .completed_write_receipt_for_prefix(&[], 0)
        .expect("zero append receipt");
    manager
        .finalize(&reservation, Some(&receipt), true)
        .expect("unchanged prefix commit");

    for group in &runtime.plan().groups {
        let snapshot = manager.snapshot(model, &session, group.domain).unwrap();
        assert_eq!(snapshot.committed_tokens, 0);
        assert_eq!(snapshot.version, 1);
    }
    assert!(
        manager.models[&model]
            .coordinators
            .values()
            .all(|coordinator| coordinator.stats().active_transactions == 0)
    );
}

#[test]
fn realtime_subphase_reservations_are_exact_and_stage_scoped() {
    let model = ModelInstanceId::new(778);
    let session = SessionKey::new("managed-realtime-subphases".into(), 1);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            8,
            8,
            &CacheCapability::Managed(test_contract()),
        )
        .unwrap()
        .unwrap();
    let operation_id = crate::engine::RealtimeOperationId::new(1);

    for (txn_id, work) in [
        (
            7781,
            WorkUnit::RealtimePreparation {
                operation_id,
                mode: crate::engine::RealtimePreparationMode::Push,
                input: crate::engine::InputRange::new(0, 160).unwrap(),
                max_output_steps: 2,
                max_cache_append: 4,
                retained_state_input: crate::engine::InputRange::new(0, 1).unwrap(),
                auxiliary_state: None,
            },
        ),
        (7782, WorkUnit::RealtimeCompletion { operation_id }),
    ] {
        let reservation = manager
            .prepare(&runtime, txn_id, &session, &work, None)
            .unwrap()
            .expect("unchanged-prefix reservation");
        assert!(reservation.allow_unchanged_prefix);
        assert!(
            reservation.domains.iter().all(|domain| {
                domain.target_committed_tokens == domain.execution_start_tokens
            })
        );
        let receipt = reservation
            .completed_write_receipt_for_prefix(
                &[],
                reservation.domains[0].execution_start_tokens,
            )
            .expect("unchanged-prefix receipt");
        manager
            .finalize(&reservation, Some(&receipt), true)
            .expect("unchanged-prefix commit");
    }

    let prompt = manager
        .prepare(
            &runtime,
            7783,
            &session,
            &WorkUnit::RealtimePromptPrefill {
                operation_id,
                max_output_steps: 2,
                cache_append: 3,
            },
            None,
        )
        .unwrap()
        .unwrap();
    assert!(!prompt.allow_unchanged_prefix);
    assert!(
        prompt.domains.iter().all(|domain| {
            domain.target_committed_tokens - domain.execution_start_tokens == 3
        })
    );
    let partial = prompt
        .completed_write_receipt_for_prefix_for_test(1, 8)
        .expect("partial prompt receipt");
    assert!(manager.finalize(&prompt, Some(&partial), true).is_err());

    let prompt = manager
        .prepare(
            &runtime,
            7784,
            &session,
            &WorkUnit::RealtimePromptPrefill {
                operation_id,
                max_output_steps: 2,
                cache_append: 3,
            },
            None,
        )
        .unwrap()
        .unwrap();
    let prompt_receipt = prompt.completed_write_receipt_for_test();
    manager
        .finalize(&prompt, Some(&prompt_receipt), true)
        .expect("exact prompt commit");

    let decode = manager
        .prepare(
            &runtime,
            7785,
            &session,
            &WorkUnit::RealtimeDecodeContinuation {
                operation_id,
                max_output_steps: 1,
                max_cache_append: 1,
                retained_state_input: crate::engine::InputRange::new(0, 1).unwrap(),
                auxiliary_state: None,
            },
            None,
        )
        .unwrap()
        .unwrap();
    assert!(!decode.allow_unchanged_prefix);
    assert!(
        decode.domains.iter().all(|domain| {
            domain.target_committed_tokens - domain.execution_start_tokens == 1
        })
    );
    let decode_receipt = decode.completed_write_receipt_for_test();
    manager
        .finalize(&decode, Some(&decode_receipt), true)
        .expect("exact decode commit");

    assert!(
        manager
            .prepare(
                &runtime,
                7786,
                &session,
                &WorkUnit::RealtimeDecodeContinuation {
                    operation_id,
                    max_output_steps: 1,
                    max_cache_append: 2,
                    retained_state_input: crate::engine::InputRange::new(1, 2).unwrap(),
                    auxiliary_state: None,
                },
                None,
            )
            .is_err()
    );
}

#[test]
fn zero_accepted_prefix_aborts_every_paged_domain_and_tensor_state() {
    let model = ModelInstanceId::new(540);
    let session = SessionKey::new("managed-prefix-zero".into(), 1);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            8,
            8,
            &CacheCapability::Managed(two_paged_tensor_contract()),
        )
        .unwrap()
        .unwrap();
    let tensor_arena = runtime.tensor_state().unwrap().clone();
    let reservation = manager
        .prepare(&runtime, 550, &session, &sequence_work(0, 9), None)
        .unwrap()
        .unwrap();
    let sequence =
        PhysicalStateSequenceId::new(reservation.clocked_state.as_ref().unwrap().sequence())
            .unwrap();
    tensor_arena
        .stage_replace(
            PhysicalStateTransactionId::new(reservation.txn_id).unwrap(),
            StateDomainId::new(2),
            0,
            9,
            vec![StateComponentValue {
                component: StateComponentId::new(1),
                tensor: Some(Tensor::from_slice(&[9.0_f32; 4], 4, &Device::Cpu).unwrap()),
            }],
        )
        .unwrap();

    manager.finalize(&reservation, None, false).unwrap();
    for domain in [CacheDomainId::new(1), CacheDomainId::new(3)] {
        let snapshot = manager.snapshot(model, &session, domain).unwrap();
        assert_eq!(snapshot.committed_tokens, 0);
        assert_eq!(snapshot.version, 0);
    }
    assert!(
        tensor_arena
            .read(sequence, StateDomainId::new(2))
            .unwrap()
            .is_none()
    );
    let state = manager.models.get(&model).unwrap();
    for coordinator in state.coordinators.values() {
        assert_eq!(coordinator.stats().active_transactions, 0);
        assert_eq!(coordinator.stats().allocated_pages, 0);
        coordinator.check_invariants().unwrap();
    }
}

#[test]
fn failed_tensor_begin_rolls_back_a_new_managed_sequence() {
    let model = ModelInstanceId::new(53);
    let session = SessionKey::new("tensor-begin-rollback".into(), 1);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            2,
            16,
            &CacheCapability::Managed(composite_tensor_contract()),
        )
        .unwrap();
    let runtime = runtime.unwrap();
    let arena = runtime.tensor_state().unwrap().clone();
    let existing_sequence = PhysicalStateSequenceId::new(900).unwrap();
    let conflicting_transaction = PhysicalStateTransactionId::new(41).unwrap();
    arena.register(existing_sequence).unwrap();
    arena
        .begin(conflicting_transaction, existing_sequence)
        .unwrap();
    assert_eq!(
        arena.occupancy().unwrap(),
        crate::backends::state::TensorStateOccupancy {
            active_sequences: 1,
            active_transactions: 1,
        }
    );

    // The independently active tensor transaction forces begin() to fail
    // only after the managed session has registered its new sequence.
    assert!(
        manager
            .prepare(&runtime, 41, &session, &sequence_work(0, 1), None)
            .is_err()
    );
    assert_eq!(
        arena.occupancy().unwrap(),
        crate::backends::state::TensorStateOccupancy {
            active_sequences: 1,
            active_transactions: 1,
        }
    );
    assert!(
        !manager.models[&model]
            .tensor_sequences
            .contains_key(&session)
    );

    arena.abort(conflicting_transaction).unwrap();
    arena.release(existing_sequence).unwrap();
    manager.release_session(&session).unwrap();
    assert_eq!(
        arena.occupancy().unwrap(),
        crate::backends::state::TensorStateOccupancy {
            active_sequences: 0,
            active_transactions: 0,
        }
    );
}

#[test]
fn arena_accounting_is_once_per_model_and_survives_session_release() {
    let model = ModelInstanceId::new(44);
    let session = SessionKey::new("managed-accounting".to_string(), 1);
    let authority = authority_with_capacity(u64::MAX);
    let mut manager = ManagedKvCacheManager::new(Some(authority.clone()));
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            2,
            16,
            &CacheCapability::Managed(test_contract()),
        )
        .expect("bind")
        .expect("runtime");
    let physical_bytes = runtime.physical_bytes();
    assert_eq!(authority.snapshot().reservations, 1);
    assert_eq!(
        authority.snapshot().reserved.host_bytes,
        ResourceAmount::Known(physical_bytes)
    );

    let same = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            2,
            16,
            &CacheCapability::Managed(test_contract()),
        )
        .expect("repeat bind")
        .expect("same runtime");
    assert_eq!(same.plan().id, runtime.plan().id);
    assert_eq!(authority.snapshot().reservations, 1);

    let reservation = manager
        .prepare(&runtime, 10, &session, &sequence_work(0, 1), None)
        .expect("prepare")
        .expect("reservation");
    manager.finalize(&reservation, None, false).expect("abort");
    drop(same);
    drop(runtime);
    assert!(manager.unload_model(model).is_err());
    manager.release_session(&session).expect("session release");
    assert_eq!(authority.snapshot().reservations, 1);
    assert_eq!(
        authority.snapshot().reserved.host_bytes,
        ResourceAmount::Known(physical_bytes)
    );

    assert!(manager.unload_model(model).expect("model unload"));
    assert_eq!(authority.snapshot().reservations, 0);
    assert_eq!(
        authority.snapshot().reserved.host_bytes,
        ResourceAmount::Known(0)
    );
}

#[test]
fn unload_refuses_while_a_physical_cache_still_holds_the_arena() {
    let model = ModelInstanceId::new(46);
    let mut manager = ManagedKvCacheManager::default();
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            2,
            16,
            &CacheCapability::Managed(test_contract()),
        )
        .expect("bind")
        .expect("managed runtime");
    let arena_id = runtime.plan().groups[0].arena;
    // What an executor step's PhysicalPagedKvCache holds.
    let leaked = Arc::clone(runtime.arena(arena_id).expect("resolved arena"));
    drop(runtime);

    let error = manager.unload_model(model).expect_err("a live arena clone");
    assert!(error.to_string().contains("owners, expected 1"), "{error}");
    drop(leaked);
    assert!(manager
        .unload_model(model)
        .expect("unload after the clone drops"));
}

#[test]
fn replacement_arena_rejects_handles_from_the_unloaded_generation() {
    let model = ModelInstanceId::new(45);
    let mut manager = ManagedKvCacheManager::default();
    let old_runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            2,
            16,
            &CacheCapability::Managed(test_contract()),
        )
        .expect("old bind")
        .expect("old runtime");
    let old_group = old_runtime.plan().groups[0].clone();
    assert!(manager.unload_model(model).is_err());
    drop(old_runtime);
    assert!(manager.unload_model(model).expect("old unload"));

    let replacement = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            2,
            16,
            &CacheCapability::Managed(test_contract()),
        )
        .expect("replacement bind")
        .expect("replacement runtime");
    let replacement_group = &replacement.plan().groups[0];
    assert_ne!(old_group.arena, replacement_group.arena);
    let stale = KvSlotRef {
        block: CacheBlockRef {
            arena: old_group.arena,
            group: old_group.id,
            index: 0,
            slot_generation: 1,
        },
        offset: 0,
    };
    assert!(
        replacement
            .arena(replacement_group.arena)
            .expect("replacement arena")
            .lower_slots(&[stale])
            .is_err()
    );
}

/// Live capacity a driver reports: total fixed, free memory set by the
/// test as it simulates allocations and frees.
#[derive(Debug)]
struct DriverFreeProvider {
    capacity: u64,
    available: std::sync::atomic::AtomicU64,
}

impl DriverFreeProvider {
    fn set_available(&self, bytes: u64) {
        self.available.store(bytes, Ordering::SeqCst);
    }

    fn available(&self) -> u64 {
        self.available.load(Ordering::SeqCst)
    }
}

impl PhysicalCapacityProvider for DriverFreeProvider {
    fn snapshot(&self) -> PhysicalCapacitySnapshot {
        let host = |bytes| ResourceVector {
            host_bytes: ResourceAmount::Known(bytes),
            ..ResourceVector::zero()
        };
        PhysicalCapacitySnapshot {
            capacity: host(self.capacity),
            available: host(self.available()),
            source: CapacitySource::Test,
        }
    }
}

fn host_bytes(bytes: u64) -> ResourceVector {
    ResourceVector {
        host_bytes: ResourceAmount::Known(bytes),
        ..ResourceVector::zero()
    }
}

/// Whether a batch workspace of `bytes` passes guarded admission now.
fn workspace_fits(authority: &Arc<ResourceAuthority>, bytes: u64) -> Result<bool> {
    match authority.reserve_batch_workspace(
        ExecutionGroupId::new(1),
        crate::engine::BatchId::new(1),
        host_bytes(bytes),
    ) {
        Ok(_) => Ok(true),
        Err(Error::Overloaded(_)) => Ok(false),
        Err(error) => Err(error),
    }
}

/// Qwen3.8-27B recurrent state per row: 48 DeltaNet layers x (3 MiB
/// recurrent + 120 KiB conv history), F32.
const QWEN38_STATE_ROW_BYTES: u64 = 156_893_184;

/// The composite contract with one tensor component the size of a
/// Qwen3.8-27B state row.
fn qwen38_sized_tensor_contract() -> InferenceStateContract {
    let mut contract = composite_tensor_contract();
    let StateDomainSpec::Tensor(tensor) = &mut contract.domains[1] else {
        panic!("composite test contract tensor domain changed kind");
    };
    tensor.components[0].shape.dimensions[0].extent = ShapeExtent::Fixed {
        value: QWEN38_STATE_ROW_BYTES / 4,
    };
    contract.validate().unwrap();
    contract
}

fn stage_state_row(
    arena: &TensorStateArena,
    reservation: &ManagedCacheReservation,
    cursor: u64,
) -> Tensor {
    let row = Tensor::zeros(
        (QWEN38_STATE_ROW_BYTES / 4) as usize,
        DType::F32,
        &Device::Cpu,
    )
    .unwrap();
    arena
        .stage_replace(
            PhysicalStateTransactionId::new(reservation.txn_id).unwrap(),
            StateDomainId::new(2),
            cursor,
            cursor + 1,
            vec![StateComponentValue {
                component: StateComponentId::new(1),
                tensor: Some(row.clone()),
            }],
        )
        .unwrap();
    row
}

/// The H100 field failure on build b1d51b89 (Qwen3.8-27B-FP8, 78 retained
/// and 8 staged rows): a thread's second turn streamed one token, then
/// every decode step's 401,408 B workspace was refused with
/// `existing_pending=14,097,711,104` against `physical_available=
/// 14,072,741,888`. Model pending was exactly the lazy state envelope (86
/// rows x 156,893,184 B) plus the graph-decode claim (537,788,416 B):
/// committed rows shrank driver-free memory while staying pending.
/// Host bytes stand in for the device domain on this CPU runtime.
#[test]
fn committed_tensor_state_moves_from_pending_to_materialized_on_the_model_lease() -> Result<()>
{
    const CAPACITY: u64 = 85_064_286_208;
    const MODEL_RESERVED: u64 = 83_371_414_656;
    const MODEL_MATERIALIZED: u64 = 69_340_812_416;
    const GRAPH_DECODE_PENDING: u64 = 537_788_416;
    const REQUEST_PENDING: u64 = 67_108_864;
    const FIELD_AVAILABLE: u64 = 14_072_741_888;
    const WORKSPACE: u64 = 401_408;

    let provider = Arc::new(DriverFreeProvider {
        capacity: CAPACITY,
        available: std::sync::atomic::AtomicU64::new(CAPACITY),
    });
    let authority = Arc::new(ResourceAuthority::new(provider.clone()));
    let model = ModelInstanceId::new(3_800);
    let session = SessionKey::new("qwen38-second-turn".into(), 1);
    let mut manager = ManagedKvCacheManager::new(Some(authority.clone()));
    let runtime = manager
        .bind_request_with_capacity(
            model,
            BackendKind::Cpu,
            ManagedStateCapacityRequest {
                // A retained row needs a page; pages bound the rows.
                total_paged_pages: 128,
                logical_token_reach: None,
                retained_sequence_rows: 78,
                staged_transaction_rows: 8,
            },
            16,
            &CacheCapability::Managed(qwen38_sized_tensor_contract()),
        )?
        .expect("managed runtime");
    let arena = runtime.tensor_state().expect("tensor arena").clone();
    let envelope = arena.capacity().authorized_bytes();
    assert_eq!(
        arena.capacity().per_sequence_bytes(),
        QWEN38_STATE_ROW_BYTES
    );
    assert_eq!(envelope, 13_492_813_824);
    let paged = runtime.resident_paged_bytes();

    // The weights lease: everything but the graph-decode claim is
    // resident, so the two Model leases match the field's class totals.
    let weights_materialized = MODEL_MATERIALIZED - paged;
    let weights = authority.reserve_with_initial_materialized(
        ReservationOwner::new(ReservationClass::Model, "qwen38-weights"),
        host_bytes(weights_materialized + GRAPH_DECODE_PENDING),
        host_bytes(weights_materialized),
    )?;
    assert_eq!(
        weights_materialized + GRAPH_DECODE_PENDING + paged + envelope,
        MODEL_RESERVED
    );
    let request = authority.reserve(
        ReservationOwner::new(ReservationClass::Request, "second-turn"),
        host_bytes(REQUEST_PENDING),
    )?;
    let pending_without_state = envelope + GRAPH_DECODE_PENDING + REQUEST_PENDING;
    assert_eq!(pending_without_state, 14_097_711_104);
    let initial_reserved = authority.snapshot().reserved;

    // Prefill commits the row's first state. Driver-free memory reflects
    // the allocation from the moment the model creates it.
    provider.set_available(FIELD_AVAILABLE + QWEN38_STATE_ROW_BYTES);
    let prefill = manager
        .prepare(&runtime, 1, &session, &sequence_work(0, 1), None)?
        .expect("prefill reservation");
    let first_row = stage_state_row(&arena, &prefill, 0);
    provider.set_available(FIELD_AVAILABLE);
    // Staged state stays pending until it commits: the field arithmetic,
    // to the byte. 14,097,711,104 + 401,408 > 14,072,741,888.
    assert!(!workspace_fits(&authority, WORKSPACE)?);
    manager.finalize(
        &prefill,
        Some(&prefill.completed_write_receipt_for_test()),
        true,
    )?;
    assert!(
        workspace_fits(&authority, WORKSPACE)?,
        "the committed row must leave the pending claim"
    );
    assert_eq!(
        manager.runtime_snapshot().models[0].materialized_tensor_bytes,
        QWEN38_STATE_ROW_BYTES
    );
    // Exactly one row moved: live headroom is free minus the rest.
    let headroom = FIELD_AVAILABLE - (pending_without_state - QWEN38_STATE_ROW_BYTES);
    assert!(workspace_fits(&authority, headroom)?);
    assert!(!workspace_fits(&authority, headroom + 1)?);

    // A decode step replaces the row: the new state is allocated beside
    // the old, the commit retires the old, and the totals do not move.
    let decode = manager
        .prepare(&runtime, 2, &session, &sequence_work(1, 2), None)?
        .expect("decode reservation");
    stage_state_row(&arena, &decode, 1);
    provider.set_available(FIELD_AVAILABLE - QWEN38_STATE_ROW_BYTES);
    manager.finalize(
        &decode,
        Some(&decode.completed_write_receipt_for_test()),
        true,
    )?;
    drop(first_row);
    provider.set_available(FIELD_AVAILABLE);
    assert_eq!(
        manager.runtime_snapshot().models[0].materialized_tensor_bytes,
        QWEN38_STATE_ROW_BYTES
    );
    assert!(workspace_fits(&authority, headroom)?);
    assert!(!workspace_fits(&authority, headroom + 1)?);

    // An aborted step frees its staged state and changes nothing.
    let aborted = manager
        .prepare(&runtime, 3, &session, &sequence_work(2, 3), None)?
        .expect("aborted reservation");
    drop(stage_state_row(&arena, &aborted, 2));
    manager.finalize(&aborted, None, false)?;
    assert_eq!(
        manager.runtime_snapshot().models[0].materialized_tensor_bytes,
        QWEN38_STATE_ROW_BYTES
    );
    assert!(workspace_fits(&authority, headroom)?);
    assert!(!workspace_fits(&authority, headroom + 1)?);

    // Releasing the session restores the whole envelope as pending
    // before the row is freed; then the driver reports it free.
    manager.release_session(&session)?;
    assert_eq!(
        manager.runtime_snapshot().models[0].materialized_tensor_bytes,
        0
    );
    assert!(!workspace_fits(&authority, headroom)?);
    provider.set_available(FIELD_AVAILABLE + QWEN38_STATE_ROW_BYTES);
    let released_headroom = FIELD_AVAILABLE + QWEN38_STATE_ROW_BYTES - pending_without_state;
    assert!(workspace_fits(&authority, released_headroom)?);
    assert!(!workspace_fits(&authority, released_headroom + 1)?);
    assert_eq!(authority.snapshot().reserved, initial_reserved);

    // Unload returns the model lease exactly once.
    drop(arena);
    drop(runtime);
    assert!(manager.unload_model(model)?);
    drop(weights);
    drop(request);
    assert_eq!(authority.snapshot().reservations, 0);
    assert_eq!(authority.snapshot().reserved, ResourceVector::zero());
    Ok(())
}

/// Live Qwen3-TTS 0.6B: every request was refused because ten lazy
/// encoder-memory rows sized at 32,768 positions (1,342,382,080 B) sat
/// in `existing_pending`. Rows are now sized from the TTS state ceiling
/// (2,524 positions), and because each request reserves the row it
/// creates in its prefill workspace, the unused rows are no pending
/// claim: a request whose own need fits live memory is admitted.
#[test]
fn qwen_tts_tensor_rows_follow_the_tts_ceiling_and_never_block_a_fitting_request() -> Result<()>
{
    use crate::models::architectures::qwen3::tts::{
        qwen3_tts_inference_state_contract, qwen3_tts_retained_talker_contract,
        tests::cache_test_config,
    };
    const AVAILABLE: u64 = 2_304_168_748;
    let provider = Arc::new(DriverFreeProvider {
        capacity: 16 * 1024 * 1024 * 1024,
        available: std::sync::atomic::AtomicU64::new(AVAILABLE),
    });
    let authority = Arc::new(ResourceAuthority::new(provider.clone()));
    let config = cache_test_config();
    let full = qwen3_tts_inference_state_contract(
        &config,
        candle_core::DType::F32,
        candle_core::DType::F32,
        16,
    )?;
    let retained = qwen3_tts_retained_talker_contract(&full, &config, candle_core::DType::F32)?;
    let model = ModelInstanceId::new(606);
    let mut manager = ManagedKvCacheManager::new(Some(authority.clone()));
    let runtime = manager.bind_model_state_with_capacity(
        model,
        BackendKind::Cpu,
        ManagedStateCapacityRequest {
            total_paged_pages: 8 * 160,
            logical_token_reach: Some(2_560),
            retained_sequence_rows: 8,
            staged_transaction_rows: 2,
        },
        16,
        &retained,
    )?;
    let arena = runtime.tensor_state().expect("tensor arena").clone();
    // 2,524 x 1,024 F32 encoder memory plus the embedding, hidden and
    // logits rows; ten rows are about 0.1 GB.
    assert_eq!(arena.capacity().per_sequence_bytes(), 10_358_784);
    let envelope = arena.capacity().authorized_bytes();
    assert_eq!(envelope, 103_587_840);
    // Bound alone, the unused rows are pending and refuse a request
    // that fits by itself.
    assert!(!workspace_fits(&authority, AVAILABLE)?);
    assert!(workspace_fits(&authority, AVAILABLE - envelope)?);

    let reserved = authority.snapshot().reserved;
    manager.price_tensor_rows_with_requests(model)?;
    manager.price_tensor_rows_with_requests(model)?;
    assert!(workspace_fits(&authority, AVAILABLE)?);
    assert!(!workspace_fits(&authority, AVAILABLE + 1)?);
    // Planning still holds the whole envelope.
    assert_eq!(authority.snapshot().reserved, reserved);
    assert_eq!(
        manager.runtime_snapshot().models[0].materialized_tensor_bytes,
        envelope
    );

    // A request's committed row and its release leave the envelope
    // recorded: the request paid for the row, the lease never re-pends it.
    let session = SessionKey::new("qwen-tts-request".into(), 1);
    let prefill = manager
        .prepare(&runtime, 1, &session, &sequence_work(0, 12), None)?
        .expect("prefill reservation");
    let row = |shape: (usize, usize, usize)| {
        Some(Tensor::zeros(shape, DType::F32, &Device::Cpu).unwrap())
    };
    arena.stage_replace(
        PhysicalStateTransactionId::new(prefill.txn_id)?,
        StateDomainId::new(2),
        0,
        12,
        [
            (1, 1, 40, 1_024),
            (2, 1, 1, 1_024),
            (3, 1, 1, 1_024),
            (4, 1, 1, 3_072),
        ]
        .into_iter()
        .map(|(id, batch, sequence, width)| StateComponentValue {
            component: StateComponentId::new(id),
            tensor: row((batch, sequence, width)),
        })
        .collect(),
    )?;
    manager.finalize(
        &prefill,
        Some(&prefill.completed_write_receipt_for_test()),
        true,
    )?;
    assert!(workspace_fits(&authority, AVAILABLE)?);
    manager.release_session(&session)?;
    assert!(workspace_fits(&authority, AVAILABLE)?);
    assert_eq!(
        manager.runtime_snapshot().models[0].materialized_tensor_bytes,
        envelope
    );

    drop(arena);
    drop(runtime);
    assert!(manager.unload_model(model)?);
    assert_eq!(authority.snapshot().reservations, 0);
    Ok(())
}

/// Committed prefix snapshots outlive their publisher's state, so they
/// are an allocation after the load-time fit. Their budget is reserved
/// on the model lease at load; an indexed snapshot stays recorded as
/// materialized after its publisher advances or releases (each backing
/// allocation once, whoever holds it); eviction and unload restore its
/// pending claim before the tensors drop. Reserved = materialized +
/// pending throughout, and materialized equals what is physically held.
#[test]
fn prefix_snapshots_are_reserved_at_load_and_stay_materialized_until_evicted() -> Result<()> {
    use crate::models::shared::memory::accounting::tensor_backing;
    const ROW: u64 = 16; // four F32 values per hybrid row
    let provider = Arc::new(DriverFreeProvider {
        capacity: 1 << 40,
        available: std::sync::atomic::AtomicU64::new(1 << 40),
    });
    let authority = Arc::new(ResourceAuthority::new(provider.clone()));
    let model = ModelInstanceId::new(3_801);
    let mut manager =
        ManagedKvCacheManager::with_prefix_cache_salt(Some(authority.clone()), Some([41; 32]));
    manager.set_prefix_snapshot_rows(1);
    let contract = hybrid_snapshot_contract();
    assert_eq!(manager.prefix_snapshot_rows_for(&contract), 1);
    assert_eq!(manager.prefix_snapshot_rows_for(&test_contract()), 0);
    let runtime = manager
        .bind_request(
            model,
            BackendKind::Cpu,
            12,
            32,
            &CacheCapability::Managed(contract),
        )?
        .expect("managed runtime");
    let arena = runtime.tensor_state().expect("tensor arena").clone();
    let capacity = arena.capacity();
    assert_eq!(capacity.per_sequence_bytes(), ROW);
    assert_eq!(capacity.snapshot_capacity(), 1);
    assert_eq!(capacity.snapshot_capacity_bytes(), ROW);
    assert_eq!(
        capacity.authorized_bytes(),
        ROW * u64::from(capacity.sequence_capacity() + capacity.transaction_capacity() + 1)
    );
    // The lease authorizes the snapshot budget at load, beside the rows.
    let reserved_at_load = authority.snapshot().reserved;
    assert_eq!(
        reserved_at_load,
        host_bytes(runtime.resident_paged_bytes() + capacity.authorized_bytes())
    );
    let model_snapshot =
        |manager: &ManagedKvCacheManager| manager.runtime_snapshot().models[0].clone();
    assert_eq!(model_snapshot(&manager).prefix_snapshot_capacity, Some(1));
    assert_eq!(model_snapshot(&manager).prefix_snapshot_budget_bytes, ROW);
    // Reserved = materialized + pending, and materialized tensor bytes
    // equal the distinct allocations live sessions and the index hold.
    let check = |manager: &ManagedKvCacheManager, held: &[&Tensor]| -> Result<()> {
        let ledger = authority.snapshot();
        assert_eq!(ledger.reserved, reserved_at_load);
        assert_eq!(
            ledger.materialized.checked_add(ledger.pending)?,
            ledger.reserved
        );
        let mut keys = HashSet::new();
        let physically_held = held
            .iter()
            .filter(|tensor| keys.insert(tensor_backing(tensor).0))
            .map(|tensor| tensor_backing(tensor).1.unwrap())
            .sum::<u64>();
        let snapshot = model_snapshot(manager);
        assert_eq!(snapshot.materialized_tensor_bytes, physically_held);
        assert_eq!(
            ledger.materialized,
            host_bytes(runtime.resident_paged_bytes() + physically_held)
        );
        Ok(())
    };
    let commit = |manager: &mut ManagedKvCacheManager,
                  plan: u64,
                  session: &SessionKey,
                  request: &EngineCoreRequest,
                  span: (usize, usize),
                  values: [f32; 4]|
     -> Result<Tensor> {
        let reservation = manager
            .prepare(
                &runtime,
                plan,
                session,
                &sequence_work(span.0, span.1),
                Some(request),
            )?
            .expect("managed reservation");
        let state = hybrid_tensor_values(&values);
        let tensor = state[0].tensor.clone().unwrap();
        arena.stage_replace(
            PhysicalStateTransactionId::new(plan).unwrap(),
            StateDomainId::new(2),
            span.0 as u64,
            span.1 as u64,
            state,
        )?;
        manager.finalize(
            &reservation,
            Some(&reservation.completed_write_receipt_for_test()),
            true,
        )?;
        Ok(tensor)
    };

    // A commits an aligned 64-token prompt and publishes its snapshot,
    // which is A's own committed state: one allocation, counted once.
    let session_a = SessionKey::new("snapshot-publisher".into(), 1);
    let mut tokens_a = (0..64_u32).collect::<Vec<u32>>();
    tokens_a.push(999);
    let request_a = prefix_request(model, tokens_a);
    let published = commit(&mut manager, 1, &session_a, &request_a, (0, 64), [7.0; 4])?;
    assert_eq!(manager.telemetry_snapshot().tensor_snapshot_publishes, 1);
    assert_eq!(model_snapshot(&manager).prefix_snapshots, 1);
    assert_eq!(model_snapshot(&manager).prefix_snapshot_bytes, ROW);
    check(&manager, &[&published])?;

    // A advances: its new state is a second allocation, and the indexed
    // snapshot stays materialized (RED without the pin: it would read
    // as a pending claim while still holding memory).
    let advanced = commit(&mut manager, 2, &session_a, &request_a, (64, 65), [8.0; 4])?;
    check(&manager, &[&published, &advanced])?;
    assert_eq!(model_snapshot(&manager).materialized_tensor_bytes, 2 * ROW);

    // A finishes: only the snapshot remains held.
    manager.release_session(&session_a)?;
    drop(advanced);
    check(&manager, &[&published])?;

    // B publishes a different prefix. The one-row budget evicts A's
    // snapshot: its claim returns to pending before it is freed.
    let session_b = SessionKey::new("snapshot-evictor".into(), 1);
    let mut tokens_b = (1_000..1_064_u32).collect::<Vec<u32>>();
    tokens_b.push(999);
    let request_b = prefix_request(model, tokens_b);
    let replacement = commit(&mut manager, 3, &session_b, &request_b, (0, 64), [9.0; 4])?;
    assert_eq!(manager.telemetry_snapshot().tensor_snapshot_evictions, 1);
    assert_eq!(model_snapshot(&manager).prefix_snapshots, 1);
    drop(published);
    check(&manager, &[&replacement])?;
    manager.release_session(&session_b)?;
    check(&manager, &[&replacement])?;

    // Unload frees the last snapshot with its claim, then the lease.
    drop(arena);
    drop(runtime);
    assert!(manager.unload_model(model)?);
    assert_eq!(authority.snapshot().reservations, 0);
    assert_eq!(authority.snapshot().reserved, ResourceVector::zero());
    Ok(())
}

/// The H100 numbers for Qwen3.8-27B-FP8 (80 GB, field capacity
/// 85,064,286,208 B). The resident footprint is anchored to the field
/// deploy, which fitted 78 retained rows with the old 1 GiB reserve; the
/// next deploy runs the 3% reserve (2,551,928,586 B). The default
/// 8-snapshot budget (8 x 156,893,184 B) comes out of the automatic rows'
/// state share, so retained rows shrink by 8 and the KV reach is
/// unchanged; beside explicit rows the KV reach pays for it instead.
#[test]
fn cuda_fit_prices_the_prefix_snapshot_budget_inside_the_state_share() {
    const GIB: u64 = 1024 * 1024 * 1024;
    const H100_BYTES: u64 = 85_064_286_208;
    const DECODE_WIDTH: u32 = 8;
    const DECODE_RESERVE: u64 = 8 * 401_408;
    // Everything resident before the fit (weights, graph pools, context)
    // that puts the field's post-reserve budget mid-window of 86 rows.
    const RESIDENT: u64 = 29_165_420_544;
    let contract = qwen38_27b_tensor_contract();
    let fit = |reserve: u64, snapshots: u32, automatic: bool| {
        cuda_admission_fit_with_snapshots(
            &contract,
            H100_BYTES,
            RESIDENT,
            reserve + DECODE_RESERVE,
            CudaResidentRowRequest {
                retained_sequence_rows: if automatic { 316 } else { 64 },
                staged_transaction_rows: DECODE_WIDTH,
                automatic_state_budget_fraction: automatic.then_some(0.25),
            },
            snapshots,
        )
        .unwrap()
    };
    let field = fit(GIB, 0, true);
    assert_eq!(field.retained_sequence_rows, 78, "anchored to the field");
    let reserve = 2_551_928_586;
    let plain = fit(reserve, 0, true);
    let shared = fit(reserve, DEFAULT_PREFIX_SNAPSHOT_ROWS, true);
    assert_eq!(plain.per_sequence_state_bytes, QWEN38_STATE_ROW_BYTES);
    assert_eq!(plain.retained_sequence_rows, 76);
    assert_eq!(shared.prefix_snapshot_rows, 8);
    assert_eq!(shared.retained_sequence_rows, 68);
    assert_eq!(shared.staged_transaction_rows, DECODE_WIDTH);
    assert_eq!(shared.resident_state_bytes, plain.resident_state_bytes);
    assert_eq!(shared.resident_state_bytes, 84 * QWEN38_STATE_ROW_BYTES);
    assert_eq!(shared.logical_token_reach, plain.logical_token_reach);
    // Explicit rows keep their count; the snapshots cost KV instead.
    let explicit_plain = fit(reserve, 0, false);
    let explicit_shared = fit(reserve, 8, false);
    assert_eq!(explicit_shared.retained_sequence_rows, 64);
    assert_eq!(
        explicit_shared.resident_state_bytes,
        explicit_plain.resident_state_bytes + 8 * QWEN38_STATE_ROW_BYTES
    );
    let lost_kv =
        (explicit_plain.logical_token_reach - explicit_shared.logical_token_reach) * 65_536;
    assert!(
        lost_kv.abs_diff(8 * QWEN38_STATE_ROW_BYTES) <= 2 * 64 * 65_536,
        "explicit rows gave up {lost_kv} B of KV"
    );
    eprintln!(
        "H100 fit (Qwen3.8-27B-FP8): field 1 GiB reserve {} rows reach {}; 3% reserve {} + {} rows reach {}; with {} snapshots {} + {} rows, state {} B, reach {} tokens",
        field.retained_sequence_rows,
        field.logical_token_reach,
        plain.retained_sequence_rows,
        plain.staged_transaction_rows,
        plain.logical_token_reach,
        shared.prefix_snapshot_rows,
        shared.retained_sequence_rows,
        shared.staged_transaction_rows,
        shared.resident_state_bytes,
        shared.logical_token_reach,
    );
}

#[test]
fn cuda_arena_accounting_is_guarded_while_cpu_and_metal_are_advisory() {
    for backend in [BackendKind::Cpu, BackendKind::Metal] {
        let authority = advisory_authority_with_capacity(1);
        let resources = managed_state_resources(
            backend,
            StateResourceVector {
                host_bytes: 2,
                ..StateResourceVector::default()
            },
        )
        .unwrap();
        let lease =
            reserve_managed_arena(&authority, ModelInstanceId::new(46), backend, resources)
                .expect("advisory arena accounting");
        assert_eq!(lease.resources(), resources);
    }

    let authority = authority_with_capacity(1);
    assert!(
        reserve_managed_arena(
            &authority,
            ModelInstanceId::new(47),
            BackendKind::Cuda,
            managed_state_resources(
                BackendKind::Cuda,
                StateResourceVector {
                    device_bytes: 2,
                    ..StateResourceVector::default()
                },
            )
            .unwrap(),
        )
        .is_err()
    );
}
