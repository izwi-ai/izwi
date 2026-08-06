//! Explicitly-invoked deterministic lifecycle soak for the managed KV cache.
//!
//! The test is ignored so ordinary local and PR unit-test invocations remain
//! fast. `scripts/ci/run-kv-lifecycle-soak.sh` selects a bounded profile and
//! invokes it directly.

use std::env;
use std::time::{Duration, Instant};

use super::*;
use crate::engine::{
    AdapterAbiRevision, AdapterInstanceId, ExecutionAdapterBinding, ExecutionGroupId,
    ExecutionMode, ExecutionProfile, InputRange, NativeBatchMode, SequencePhase, StageDescriptor,
    StageId,
};
use crate::kv::v2::{AttentionLogitSoftcap, PrefixPolicy};
use crate::kv::{test_contract, InferenceStateCapability as CacheCapability, KvSlotRef};
use crate::model::ModelVariant;
use crate::models::shared::chat::{ChatMessage, ChatRole};

#[derive(Clone, Copy)]
struct SoakConfig {
    minimum_iterations: u64,
    duration: Duration,
}

impl SoakConfig {
    fn from_env() -> Self {
        let minimum_iterations = parse_env_u64("IZWI_KV_SOAK_ITERATIONS", 2);
        let duration = Duration::from_secs(parse_env_u64("IZWI_KV_SOAK_DURATION_SECONDS", 0));
        assert!(
            minimum_iterations > 0 || !duration.is_zero(),
            "KV lifecycle soak requires iterations or a duration"
        );
        Self {
            minimum_iterations,
            duration,
        }
    }
}

fn parse_env_u64(name: &str, default: u64) -> u64 {
    match env::var(name) {
        Ok(raw) => raw
            .parse::<u64>()
            .unwrap_or_else(|_| panic!("{name} must be an unsigned integer, got {raw:?}")),
        Err(env::VarError::NotPresent) => default,
        Err(error) => panic!("failed to read {name}: {error}"),
    }
}

fn sequence_work(start: usize, end: usize) -> WorkUnit {
    WorkUnit::SequenceStep {
        phase: if start == 0 {
            SequencePhase::Prefill
        } else {
            SequencePhase::Decode
        },
        input: InputRange { start, end },
        max_output_steps: end.saturating_sub(start).max(1),
    }
}

fn prefix_request(model: ModelInstanceId, tokens: Vec<u32>) -> EngineCoreRequest {
    let variant = ModelVariant::Qwen306B;
    let profile =
        ExecutionProfile::fail_closed(BackendKind::Cpu, Some(variant), ExecutionMode::Sequence);
    let stage = StageDescriptor::from_execution_profile(
        StageId::new(1),
        "qwen3.managed-soak",
        &profile,
        NativeBatchMode::None,
    );
    let mut request = EngineCoreRequest::chat(vec![ChatMessage {
        role: ChatRole::User,
        content: "managed KV lifecycle soak".into(),
    }])
    .with_model_variant(variant);
    request.prompt_tokens = tokens;
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
        .expect("bind soak request adapter");
    request
}

fn windowed_softcap_contract(window_tokens: u32) -> InferenceStateContract {
    let mut contract = test_contract();
    let StateDomainSpec::PagedAttention(domain) = &mut contract.domains[0] else {
        unreachable!("test contract must contain paged attention")
    };
    for layer in &mut domain.layers {
        layer.pattern = AttentionPattern::SlidingWindow { window_tokens };
        layer.attention_logit_softcap =
            Some(AttentionLogitSoftcap::new(30.0).expect("finite positive test softcap"));
    }
    domain.header.prefix = PrefixPolicy::Disabled;
    contract.groups[0].prefix_shareable = false;
    contract.validate().expect("windowed soak contract");
    contract
}

fn assert_quiescent_and_bounded(manager: &ManagedKvCacheManager) {
    let snapshot = manager.runtime_snapshot();
    let coordinator = &snapshot.totals.coordinator;
    assert_eq!(coordinator.active_transactions, 0);
    assert_eq!(coordinator.reservations, 0);
    assert_eq!(coordinator.execution_pins, 0);
    assert_eq!(coordinator.transfer_pins, 0);
    assert!(coordinator.allocated_pages <= coordinator.capacity_pages);
    assert!(coordinator.free_pages <= coordinator.capacity_pages);
    assert!(coordinator.table_refs <= coordinator.capacity_pages);
    assert!(coordinator.prefix_refs <= coordinator.capacity_pages);
    assert_eq!(
        coordinator.allocated_pages + coordinator.free_pages,
        coordinator.capacity_pages
    );
    assert_eq!(
        snapshot.counters.prefix_retained_pages,
        coordinator.prefix_refs
    );
    let exact_backing = snapshot
        .models
        .iter()
        .map(|model| model.physical_bytes)
        .sum::<u64>();
    assert_eq!(snapshot.totals.physical_bytes, exact_backing);
}

fn exercise_prefix_abort_and_reuse(
    manager: &mut ManagedKvCacheManager,
    runtime: &ManagedKvModelRuntime,
    model: ModelInstanceId,
    cycle: u64,
    next_txn: &mut u64,
) {
    let page_tokens = runtime.plan().groups[0].page_tokens as usize;
    let prompt_len = page_tokens * 2 + 1;
    let tokens = (0..prompt_len)
        .map(|index| index as u32 + (cycle as u32).wrapping_mul(1_000))
        .collect::<Vec<_>>();
    let publisher = SessionKey::new(format!("prefix-publisher-{cycle}"), cycle * 10 + 1);
    let request = prefix_request(model, tokens.clone());
    let reservation = manager
        .prepare(
            runtime,
            *next_txn,
            &publisher,
            &sequence_work(0, prompt_len),
            Some(&request),
        )
        .expect("prepare prefix publisher")
        .expect("prefix publisher reservation");
    *next_txn += 1;
    manager
        .finalize(
            &reservation,
            Some(&reservation.completed_write_receipt_for_test()),
            true,
        )
        .expect("commit prefix publisher");

    // Extend the partially-filled committed tail before releasing the source
    // session. Explicit physical tail COW is covered by the coordinator's
    // source-pin/copy/commit test; prefix lookup below exercises shared-page
    // retention and abort cleanup through the manager boundary.
    let continuation = manager
        .prepare(
            runtime,
            *next_txn,
            &publisher,
            &sequence_work(prompt_len, prompt_len + 1),
            None,
        )
        .expect("prepare prefix continuation")
        .expect("prefix continuation reservation");
    *next_txn += 1;
    manager
        .finalize(
            &continuation,
            Some(&continuation.completed_write_receipt_for_test()),
            true,
        )
        .expect("commit prefix continuation");
    manager
        .release_session(&publisher)
        .expect("release prefix publisher");

    let mut divergent = tokens;
    *divergent.last_mut().expect("non-empty prompt") ^= 0x55aa;
    let consumer = SessionKey::new(format!("prefix-consumer-{cycle}"), cycle * 10 + 2);
    let request = prefix_request(model, divergent);
    let reused = manager
        .prepare(
            runtime,
            *next_txn,
            &consumer,
            &sequence_work(0, prompt_len),
            Some(&request),
        )
        .expect("prepare prefix consumer")
        .expect("prefix consumer reservation");
    *next_txn += 1;
    assert_eq!(
        reused.domains[0].execution_start_tokens as usize,
        page_tokens * 2
    );
    manager
        .finalize(&reused, None, false)
        .expect("abort prefix consumer");
    manager
        .release_session(&consumer)
        .expect("release prefix consumer");
}

fn exercise_saturation_and_generation_reuse(
    manager: &mut ManagedKvCacheManager,
    runtime: &ManagedKvModelRuntime,
    model: ModelInstanceId,
    cycle: u64,
    next_txn: &mut u64,
) {
    let page_tokens = runtime.plan().groups[0].page_tokens as usize;
    let capacity = runtime.plan().groups[0].capacity_pages as usize;
    let first = SessionKey::new(format!("saturated-owner-{cycle}"), cycle * 10 + 3);
    let held = manager
        .prepare(
            runtime,
            *next_txn,
            &first,
            &sequence_work(0, page_tokens * capacity),
            None,
        )
        .expect("prepare capacity owner")
        .expect("capacity owner reservation");
    *next_txn += 1;

    let contender = SessionKey::new(format!("saturated-contender-{cycle}"), cycle * 10 + 4);
    let error = manager
        .prepare(runtime, *next_txn, &contender, &sequence_work(0, 1), None)
        .expect_err("full arena must apply backpressure");
    *next_txn += 1;
    assert!(matches!(error, Error::Backpressure(_)));
    manager
        .finalize(&held, None, false)
        .expect("cancel capacity owner");
    manager.release_session(&first).expect("release owner");
    manager
        .release_session(&contender)
        .expect("release contender");

    let reuse = SessionKey::new(format!("generation-reuse-{cycle}"), cycle * 10 + 5);
    let first_lease = manager
        .prepare(runtime, *next_txn, &reuse, &sequence_work(0, 1), None)
        .expect("prepare first generation")
        .expect("first generation reservation");
    *next_txn += 1;
    let old = first_lease.domains[0].writable_blocks[0];
    manager
        .finalize(&first_lease, None, false)
        .expect("abort first generation");
    let second_lease = manager
        .prepare(runtime, *next_txn, &reuse, &sequence_work(0, 1), None)
        .expect("prepare reused generation")
        .expect("reused generation reservation");
    *next_txn += 1;
    let new = second_lease.domains[0].writable_blocks[0];
    assert_eq!(old.index, new.index);
    assert!(new.slot_generation > old.slot_generation);
    manager
        .finalize(&second_lease, None, false)
        .expect("abort reused generation");
    manager.release_session(&reuse).expect("release reuse row");

    assert_eq!(runtime.plan().model_instance, model);
}

fn exercise_windowed_ragged_rows(
    manager: &mut ManagedKvCacheManager,
    runtime: &ManagedKvModelRuntime,
    model: ModelInstanceId,
    cycle: u64,
    next_txn: &mut u64,
) {
    let window_tokens = 31usize;
    let lengths = [67usize, 131, 257];
    let sessions = lengths
        .iter()
        .enumerate()
        .map(|(row, _)| {
            SessionKey::new(
                format!("window-ragged-{cycle}-{row}"),
                cycle * 100 + row as u64 + 1,
            )
        })
        .collect::<Vec<_>>();
    let mut committed = [0usize; 3];
    let mut pending = Vec::with_capacity(lengths.len());
    for (row, session) in sessions.iter().enumerate() {
        let end = row * 2 + 1;
        let reservation = manager
            .prepare(runtime, *next_txn, session, &sequence_work(0, end), None)
            .expect("prepare concurrent ragged row")
            .expect("concurrent ragged reservation");
        *next_txn += 1;
        committed[row] = end;
        pending.push(reservation);
    }
    assert_eq!(
        manager
            .runtime_snapshot()
            .totals
            .coordinator
            .active_transactions,
        lengths.len() as u64
    );
    for reservation in pending.iter().rev() {
        manager
            .finalize(
                reservation,
                Some(&reservation.completed_write_receipt_for_test()),
                true,
            )
            .expect("commit concurrent ragged row");
    }

    for (row, (&target, session)) in lengths.iter().zip(&sessions).enumerate() {
        while committed[row] < target {
            let stride = 1 + ((cycle as usize + row + committed[row]) % 11);
            let end = (committed[row] + stride).min(target);
            let reservation = manager
                .prepare(
                    runtime,
                    *next_txn,
                    session,
                    &sequence_work(committed[row], end),
                    None,
                )
                .expect("prepare ragged window row")
                .expect("ragged window reservation");
            *next_txn += 1;
            manager
                .finalize(
                    &reservation,
                    Some(&reservation.completed_write_receipt_for_test()),
                    true,
                )
                .expect("commit ragged window row");
            committed[row] = end;
            let snapshot = manager
                .snapshot(model, session, CacheDomainId::new(1))
                .expect("window snapshot");
            assert_eq!(snapshot.committed_tokens as usize, committed[row]);
            assert_eq!(
                snapshot.window_start as usize,
                committed[row].saturating_sub(window_tokens)
            );
            assert!(snapshot.groups[0].blocks.len() <= 3);
        }
        manager
            .release_session(session)
            .expect("release ragged window row");
    }
}

fn run_cycle(cycle: u64, next_txn: &mut u64) {
    let page_tokens = if cycle % 2 == 0 { 16 } else { 32 };
    let prefix_model = ModelInstanceId::new(cycle * 2 + 1);
    let window_model = ModelInstanceId::new(cycle * 2 + 2);
    let mut manager = ManagedKvCacheManager::with_prefix_cache_policy(None, Some([0x5a; 32]), 2);

    let prefix_runtime = manager
        .bind_request(
            prefix_model,
            BackendKind::Cpu,
            6,
            page_tokens,
            &CacheCapability::Managed(test_contract()),
        )
        .expect("bind prefix model")
        .expect("prefix model runtime");
    let first_arena = prefix_runtime.plan().groups[0].arena;
    exercise_prefix_abort_and_reuse(&mut manager, &prefix_runtime, prefix_model, cycle, next_txn);
    exercise_saturation_and_generation_reuse(
        &mut manager,
        &prefix_runtime,
        prefix_model,
        cycle,
        next_txn,
    );

    let window_runtime = manager
        .bind_request(
            window_model,
            BackendKind::Cpu,
            12,
            page_tokens,
            &CacheCapability::Managed(windowed_softcap_contract(31)),
        )
        .expect("bind window model")
        .expect("window model runtime");
    exercise_windowed_ragged_rows(&mut manager, &window_runtime, window_model, cycle, next_txn);
    assert_quiescent_and_bounded(&manager);

    drop(prefix_runtime);
    drop(window_runtime);
    assert!(manager
        .unload_model(prefix_model)
        .expect("unload prefix model"));
    assert!(manager
        .unload_model(window_model)
        .expect("unload window model"));
    let unloaded = manager.runtime_snapshot();
    assert_eq!(unloaded.totals.models, 0);
    assert_eq!(unloaded.totals.arenas, 0);
    assert_eq!(unloaded.totals.physical_bytes, 0);

    let replacement = manager
        .bind_request(
            prefix_model,
            BackendKind::Cpu,
            2,
            page_tokens,
            &CacheCapability::Managed(test_contract()),
        )
        .expect("reload prefix model")
        .expect("replacement runtime");
    let replacement_group = &replacement.plan().groups[0];
    assert_ne!(first_arena, replacement_group.arena);
    assert!(replacement
        .arena(replacement_group.arena)
        .expect("replacement arena")
        .lower_slots(&[KvSlotRef {
            block: crate::kv::CacheBlockRef {
                arena: first_arena,
                group: replacement_group.id,
                index: 0,
                slot_generation: 1,
            },
            offset: 0,
        }])
        .is_err());
    drop(replacement);
    assert!(manager
        .unload_model(prefix_model)
        .expect("unload replacement"));
    assert_eq!(manager.model_count(), 0);
}

#[test]
#[ignore = "run via scripts/ci/run-kv-lifecycle-soak.sh with an explicit profile"]
fn managed_kv_lifecycle_soak() {
    let config = SoakConfig::from_env();
    let started = Instant::now();
    let mut completed = 0u64;
    let mut next_txn = 1u64;
    while completed < config.minimum_iterations || started.elapsed() < config.duration {
        run_cycle(completed, &mut next_txn);
        completed += 1;
    }
    println!(
        "managed-kv-lifecycle-soak: iterations={completed} elapsed_ms={}",
        started.elapsed().as_millis()
    );
}
