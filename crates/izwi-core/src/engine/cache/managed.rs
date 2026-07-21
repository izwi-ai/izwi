//! Live binding between loaded-model KV contracts and physical engine state.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use candle_core::DType;

use super::coordinator::{
    KvBlockIntent, KvCacheCoordinator, KvCoordinatorError, KvGroupReservation, KvReserveRequest,
    KvSnapshot, KvWriteReceipt,
};
use crate::backends::kv::{
    CpuKvBackendRuntime, KvArena, KvArenaConfig, KvBackendPlanRequest, KvBackendRuntime,
    KvLayerConfig,
};
use crate::backends::BackendKind;
use crate::engine::{
    ManagedCacheDomainReservation, ManagedCacheReceipt, ManagedCacheReservation, ModelInstanceId,
    PlanId, SessionKey, WorkUnit,
};
use crate::error::{Error, Result};
#[cfg(test)]
use crate::kv::CacheDomainId;
use crate::kv::{
    CacheCapability, KvArenaId, KvCacheContract, KvDomainSpec, KvGroupId, KvStorageDType,
    ResolvedKvGroupKind, ResolvedKvPlan,
};

/// Immutable model-level plan and physical arenas shared by all its sessions.
pub(crate) struct ManagedKvModelRuntime {
    plan: Arc<ResolvedKvPlan>,
    arenas: HashMap<KvArenaId, Arc<dyn KvArena>>,
    physical_bytes: u64,
}

impl fmt::Debug for ManagedKvModelRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagedKvModelRuntime")
            .field("plan", &self.plan.id)
            .field("arena_count", &self.arenas.len())
            .field("physical_bytes", &self.physical_bytes)
            .finish()
    }
}

impl ManagedKvModelRuntime {
    pub(crate) fn plan(&self) -> &ResolvedKvPlan {
        &self.plan
    }

    pub(crate) fn arena(&self, id: KvArenaId) -> Option<&Arc<dyn KvArena>> {
        self.arenas.get(&id)
    }

    pub(crate) fn physical_bytes(&self) -> u64 {
        self.physical_bytes
    }
}

struct ManagedKvModelState {
    contract: KvCacheContract,
    runtime: Arc<ManagedKvModelRuntime>,
    coordinators: HashMap<KvArenaId, KvCacheCoordinator>,
    registered_sessions: HashSet<SessionKey>,
}

/// Engine-owned managed-cache registry. Arena backing is allocated once per
/// exact model instance; row transactions only change page ownership.
#[derive(Default)]
pub(crate) struct ManagedKvCacheManager {
    models: HashMap<ModelInstanceId, ManagedKvModelState>,
}

impl ManagedKvCacheManager {
    pub(crate) fn bind_request(
        &mut self,
        model_instance: ModelInstanceId,
        backend: BackendKind,
        capacity_pages: usize,
        page_tokens_hint: usize,
        capability: &CacheCapability,
    ) -> Result<Option<Arc<ManagedKvModelRuntime>>> {
        let Some(contract) = capability.managed_contract() else {
            return Ok(None);
        };
        if let Some(state) = self.models.get(&model_instance) {
            if &state.contract != contract || state.runtime.plan.backend != backend {
                return Err(Error::InvalidInput(
                    "one loaded model instance published incompatible managed KV contracts"
                        .to_string(),
                ));
            }
            return Ok(Some(state.runtime.clone()));
        }
        if backend != BackendKind::Cpu {
            return Err(Error::InvalidInput(format!(
                "managed KV arenas are not implemented for {backend:?}"
            )));
        }
        if contract
            .domains
            .iter()
            .any(|domain| matches!(domain, KvDomainSpec::ModelState(_)))
        {
            return Err(Error::InvalidInput(
                "managed model-state arenas are not implemented by the CPU KV runtime".to_string(),
            ));
        }
        let capacity_pages = u32::try_from(capacity_pages)
            .map_err(|_| Error::InvalidInput("managed KV page capacity exceeds u32".to_string()))?;
        let page_tokens_hint = u32::try_from(page_tokens_hint)
            .map_err(|_| Error::InvalidInput("managed KV page size exceeds u32".to_string()))?;
        let backend_runtime = CpuKvBackendRuntime;
        let plan = backend_runtime.negotiate(
            contract,
            &KvBackendPlanRequest {
                model_instance,
                backend,
                device_ordinal: None,
                capacity_pages,
                page_tokens_hint: Some(page_tokens_hint),
                storage_dtype_hint: None,
                first_arena_generation: 1,
            },
        )?;

        let mut arenas = HashMap::with_capacity(plan.groups.len());
        let mut coordinators = HashMap::with_capacity(plan.groups.len());
        let mut physical_bytes = 0_u64;
        for group in &plan.groups {
            let config = arena_config(contract, group)?;
            let arena = backend_runtime.allocate_arena(config)?;
            if arenas.insert(group.arena, arena).is_some() {
                return Err(Error::InferenceError(
                    "resolved KV plan reused one arena identity".to_string(),
                ));
            }
            coordinators.insert(
                group.arena,
                KvCacheCoordinator::new(group.arena, group.capacity_pages as usize),
            );
            physical_bytes = physical_bytes
                .checked_add(
                    group
                        .bytes_per_page
                        .checked_mul(u64::from(group.capacity_pages))
                        .ok_or_else(|| {
                            Error::Overloaded("managed KV byte total overflow".into())
                        })?,
                )
                .ok_or_else(|| Error::Overloaded("managed KV byte total overflow".into()))?;
        }
        let runtime = Arc::new(ManagedKvModelRuntime {
            plan: Arc::new(plan),
            arenas,
            physical_bytes,
        });
        self.models.insert(
            model_instance,
            ManagedKvModelState {
                contract: contract.clone(),
                runtime: runtime.clone(),
                coordinators,
                registered_sessions: HashSet::new(),
            },
        );
        Ok(Some(runtime))
    }

    pub(crate) fn prepare(
        &mut self,
        runtime: &ManagedKvModelRuntime,
        txn_id: PlanId,
        session: &SessionKey,
        work: &WorkUnit,
    ) -> Result<Option<ManagedCacheReservation>> {
        let WorkUnit::SequenceStep { input, .. } = work else {
            return Ok(None);
        };
        let target_committed_tokens = u32::try_from(input.end).map_err(|_| {
            Error::InvalidInput("managed KV token position exceeds u32".to_string())
        })?;
        let state = self
            .models
            .get_mut(&runtime.plan.model_instance)
            .ok_or_else(|| Error::InferenceError("managed KV model runtime is missing".into()))?;
        if state.runtime.plan.id != runtime.plan.id {
            return Err(Error::InferenceError(
                "request carries a stale managed KV runtime".to_string(),
            ));
        }
        ensure_session_tables(state, session)?;

        let mut domains = Vec::with_capacity(runtime.plan.groups.len());
        for group in &runtime.plan.groups {
            let coordinator = state
                .coordinators
                .get_mut(&group.arena)
                .expect("resolved arena has a coordinator");
            let snapshot = coordinator
                .snapshot(session, group.domain)
                .map_err(coordinator_error)?;
            if target_committed_tokens < snapshot.committed_tokens {
                abort_domains(state, txn_id, &domains);
                return Err(Error::InferenceError(
                    "scheduled KV target regressed behind the committed cache table".to_string(),
                ));
            }
            let reservation = reservation_for_group(
                group.id,
                group.page_tokens,
                &snapshot,
                target_committed_tokens,
            )?;
            let request = KvReserveRequest {
                txn_id,
                expected: snapshot.clone(),
                target_committed_tokens,
                target_window_start: 0,
                groups: vec![reservation],
            };
            let reserved = match coordinator.reserve(request.clone()) {
                Err(KvCoordinatorError::WriteConflict) => {
                    let mut copy_on_write = request;
                    for intent in &mut copy_on_write.groups[0].blocks {
                        if let KvBlockIntent::Writable(source) = *intent {
                            *intent = KvBlockIntent::CopyOnWrite(source);
                        }
                    }
                    coordinator.reserve(copy_on_write)
                }
                result => result,
            };
            if let Err(error) = reserved {
                abort_domains(state, txn_id, &domains);
                return Err(coordinator_error(error));
            }
            let prepared = match coordinator.prepare(txn_id) {
                Ok(prepared) => prepared,
                Err(error) => {
                    let _ = coordinator.abort(txn_id);
                    abort_domains(state, txn_id, &domains);
                    return Err(coordinator_error(error));
                }
            };
            let old = prepared
                .expected
                .groups
                .iter()
                .flat_map(|group| group.blocks.iter().copied())
                .collect::<HashSet<_>>();
            let fresh = prepared
                .provisional_groups
                .iter()
                .flat_map(|group| group.blocks.iter().copied())
                .filter(|block| !old.contains(block))
                .collect::<Vec<_>>();
            if !fresh.is_empty() {
                let arena = runtime
                    .arena(group.arena)
                    .expect("resolved arena allocated");
                if let Err(error) = arena.zero_pages(&fresh).and_then(|fence| fence.wait()) {
                    let _ = coordinator.abort(txn_id);
                    abort_domains(state, txn_id, &domains);
                    return Err(error);
                }
            }
            if !prepared.page_copies.is_empty() {
                let arena = runtime
                    .arena(group.arena)
                    .expect("resolved arena allocated");
                if let Err(error) = arena
                    .copy_pages(&prepared.page_copies)
                    .and_then(|fence| fence.wait())
                {
                    let _ = coordinator.abort(txn_id);
                    abort_domains(state, txn_id, &domains);
                    return Err(error);
                }
            }
            domains.push(ManagedCacheDomainReservation {
                arena: group.arena,
                domain: group.domain,
                expected_version: prepared.expected.version,
                expected_committed_tokens: prepared.expected.committed_tokens,
                target_committed_tokens: prepared.target_committed_tokens,
                target_window_start: prepared.target_window_start,
                provisional_groups: prepared.provisional_groups,
                writable_blocks: prepared.writable_blocks,
            });
        }
        Ok(Some(ManagedCacheReservation {
            txn_id,
            session: session.clone(),
            domains,
        }))
    }

    pub(crate) fn finalize(
        &mut self,
        reservation: &ManagedCacheReservation,
        receipt: Option<&ManagedCacheReceipt>,
        commit: bool,
    ) -> Result<()> {
        let model_instance = reservation
            .domains
            .first()
            .map(|domain| domain.arena.model_instance)
            .ok_or_else(|| Error::InferenceError("managed KV reservation is empty".into()))?;
        let state = self
            .models
            .get_mut(&model_instance)
            .ok_or_else(|| Error::InferenceError("managed KV model state is missing".into()))?;
        if !commit {
            abort_reservation(state, reservation);
            return Ok(());
        }
        let receipt = receipt.ok_or_else(|| {
            Error::InferenceError("committing managed KV row omitted its write receipt".into())
        })?;
        if &receipt.reservation != reservation {
            abort_reservation(state, reservation);
            return Err(Error::InferenceError(
                "managed KV receipt crossed a row transaction fence".to_string(),
            ));
        }

        // Validate every backend acknowledgement before publishing any table.
        for domain in &reservation.domains {
            let Some(written) = receipt
                .domains
                .iter()
                .find(|receipt| receipt.arena == domain.arena && receipt.domain == domain.domain)
            else {
                abort_reservation(state, reservation);
                return Err(Error::InferenceError(
                    "managed KV receipt omitted a cache domain".into(),
                ));
            };
            let completed = state
                .coordinators
                .get_mut(&domain.arena)
                .expect("reservation arena has a coordinator")
                .complete_write(KvWriteReceipt {
                    txn_id: reservation.txn_id,
                    committed_tokens: domain.target_committed_tokens,
                    written_blocks: written.written_blocks.clone(),
                });
            if let Err(error) = completed {
                abort_reservation(state, reservation);
                return Err(coordinator_error(error));
            }
        }
        // The engine serializes this loop under its state lock. Every domain
        // has already passed version/write validation, so no competing table
        // update can interleave between these publications.
        for domain in &reservation.domains {
            state
                .coordinators
                .get_mut(&domain.arena)
                .expect("reservation arena has a coordinator")
                .commit(reservation.txn_id, &[])
                .map_err(coordinator_error)?;
        }
        Ok(())
    }

    pub(crate) fn release_session(&mut self, session: &SessionKey) -> Result<()> {
        for state in self.models.values_mut() {
            if !state.registered_sessions.remove(session) {
                continue;
            }
            for group in &state.runtime.plan.groups {
                state
                    .coordinators
                    .get_mut(&group.arena)
                    .expect("resolved arena has a coordinator")
                    .release_table(session, group.domain)
                    .map_err(coordinator_error)?;
            }
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn snapshot(
        &self,
        model_instance: ModelInstanceId,
        session: &SessionKey,
        domain: CacheDomainId,
    ) -> Option<KvSnapshot> {
        let state = self.models.get(&model_instance)?;
        state
            .coordinators
            .values()
            .find_map(|coordinator| coordinator.snapshot(session, domain).ok())
    }
}

fn ensure_session_tables(state: &mut ManagedKvModelState, session: &SessionKey) -> Result<()> {
    if !state.registered_sessions.insert(session.clone()) {
        return Ok(());
    }
    let mut registered = Vec::new();
    for group in &state.runtime.plan.groups {
        let coordinator = state
            .coordinators
            .get_mut(&group.arena)
            .expect("resolved arena has a coordinator");
        if let Err(error) = coordinator.register_table(session.clone(), group.domain) {
            for (arena, domain) in registered {
                let _ = state
                    .coordinators
                    .get_mut(&arena)
                    .expect("registered arena exists")
                    .release_table(session, domain);
            }
            state.registered_sessions.remove(session);
            return Err(coordinator_error(error));
        }
        registered.push((group.arena, group.domain));
    }
    Ok(())
}

fn reservation_for_group(
    group: KvGroupId,
    page_tokens: u32,
    snapshot: &KvSnapshot,
    target_tokens: u32,
) -> Result<KvGroupReservation> {
    let required_pages = target_tokens
        .checked_add(page_tokens - 1)
        .ok_or_else(|| Error::Overloaded("managed KV page count overflow".into()))?
        / page_tokens;
    let required_pages = usize::try_from(required_pages)
        .map_err(|_| Error::Overloaded("managed KV page count exceeds usize".into()))?;
    let existing = snapshot
        .groups
        .iter()
        .find(|table| table.group == group)
        .map(|table| table.blocks.as_slice())
        .unwrap_or_default();
    let mut blocks = Vec::with_capacity(required_pages);
    for (index, block) in existing.iter().take(required_pages).copied().enumerate() {
        let is_partial_tail = snapshot.committed_tokens % page_tokens != 0
            && index + 1 == existing.len()
            && target_tokens > snapshot.committed_tokens;
        blocks.push(if is_partial_tail {
            KvBlockIntent::Writable(block)
        } else {
            KvBlockIntent::Existing(block)
        });
    }
    blocks.extend(
        std::iter::repeat(KvBlockIntent::Fresh).take(required_pages.saturating_sub(blocks.len())),
    );
    Ok(KvGroupReservation { group, blocks })
}

fn abort_domains(
    state: &mut ManagedKvModelState,
    txn_id: PlanId,
    domains: &[ManagedCacheDomainReservation],
) {
    for domain in domains {
        let _ = state
            .coordinators
            .get_mut(&domain.arena)
            .expect("reservation arena has a coordinator")
            .abort(txn_id);
    }
}

fn abort_reservation(state: &mut ManagedKvModelState, reservation: &ManagedCacheReservation) {
    abort_domains(state, reservation.txn_id, &reservation.domains);
}

fn arena_config(
    contract: &KvCacheContract,
    group: &crate::kv::ResolvedKvGroup,
) -> Result<KvArenaConfig> {
    let domain = contract
        .domains
        .iter()
        .find(|domain| domain.id() == group.domain)
        .ok_or_else(|| {
            Error::InferenceError("resolved KV group lost its semantic domain".into())
        })?;
    let (spec, bindings) = match (domain, &group.kind) {
        (KvDomainSpec::PagedAttention(spec), ResolvedKvGroupKind::PagedAttention { layers }) => {
            (spec, layers)
        }
        _ => {
            return Err(Error::InvalidInput(
                "CPU KV arena requires a paged-attention domain".to_string(),
            ));
        }
    };
    let mut layers = Vec::with_capacity(bindings.len());
    for binding in bindings {
        let layer = spec
            .layers
            .iter()
            .find(|layer| layer.model_layer == binding.model_layer)
            .ok_or_else(|| Error::InferenceError("resolved KV layer binding is stale".into()))?;
        layers.push(KvLayerConfig {
            binding: *binding,
            num_kv_heads: layer.num_kv_heads,
            key_head_dim: layer.key_head_dim,
            value_head_dim: layer.value_head_dim,
        });
    }
    Ok(KvArenaConfig {
        id: group.arena,
        group: group.id,
        page_tokens: group.page_tokens,
        capacity_pages: group.capacity_pages,
        dtype: candle_dtype(group.storage.dtype())?,
        layers,
    })
}

fn candle_dtype(dtype: KvStorageDType) -> Result<DType> {
    match dtype {
        KvStorageDType::F32 => Ok(DType::F32),
        KvStorageDType::F16 => Ok(DType::F16),
        KvStorageDType::Bf16 => Ok(DType::BF16),
        KvStorageDType::I8 | KvStorageDType::Q4 => Err(Error::InvalidInput(
            "CPU KV arena cannot allocate quantized storage".to_string(),
        )),
    }
}

fn coordinator_error(error: impl fmt::Display) -> Error {
    Error::InferenceError(format!(
        "managed KV coordinator rejected transaction: {error}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{InputRange, SequencePhase};
    use crate::kv::test_contract;

    fn sequence_work(start: usize, end: usize) -> WorkUnit {
        WorkUnit::SequenceStep {
            phase: SequencePhase::Prefill,
            input: InputRange { start, end },
            max_output_steps: end.saturating_sub(start).max(1),
        }
    }

    #[test]
    fn live_manager_commits_aborts_and_releases_exact_session_tables() {
        let model = ModelInstanceId::new(41);
        let session = SessionKey::new("managed-live".to_string(), 7);
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
            .expect("bind")
            .expect("managed runtime");
        assert!(runtime.physical_bytes() > 0);

        let first = manager
            .prepare(&runtime, 1, &session, &sequence_work(0, 5))
            .expect("prepare")
            .expect("reservation");
        assert_eq!(first.domains.len(), 1);
        assert_eq!(first.domains[0].writable_blocks.len(), 1);
        manager
            .finalize(&first, Some(&first.completed_write_receipt()), true)
            .expect("commit");
        let snapshot = manager.snapshot(model, &session, domain).expect("snapshot");
        assert_eq!(snapshot.version, 1);
        assert_eq!(snapshot.committed_tokens, 5);

        let second = manager
            .prepare(&runtime, 2, &session, &sequence_work(5, 17))
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
        if let KvDomainSpec::PagedAttention(domain) = &mut changed.domains[0] {
            domain.layers[0].num_kv_heads = 2;
        }
        assert!(manager
            .bind_request(
                model,
                BackendKind::Cpu,
                2,
                16,
                &CacheCapability::Managed(changed),
            )
            .is_err());
    }

    #[test]
    fn composite_domain_receipt_publishes_every_table_under_one_row_fence() {
        let model = ModelInstanceId::new(43);
        let session = SessionKey::new("managed-composite".to_string(), 3);
        let mut contract = test_contract();
        let mut second = contract.domains[0].clone();
        if let KvDomainSpec::PagedAttention(domain) = &mut second {
            domain.id = CacheDomainId::new(2);
        }
        contract.domains.push(second);
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
            .prepare(&runtime, 8, &session, &sequence_work(0, 8))
            .expect("prepare")
            .expect("reservation");
        assert_eq!(reservation.domains.len(), 2);
        manager
            .finalize(
                &reservation,
                Some(&reservation.completed_write_receipt()),
                true,
            )
            .expect("composite commit");
        for domain in [CacheDomainId::new(1), CacheDomainId::new(2)] {
            let snapshot = manager.snapshot(model, &session, domain).expect("snapshot");
            assert_eq!(snapshot.version, 1);
            assert_eq!(snapshot.committed_tokens, 8);
        }
    }
}
