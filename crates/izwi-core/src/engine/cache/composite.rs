//! Composite retained state for paged decoder KV plus immutable static memory.

use std::collections::HashSet;
use std::sync::Arc;

use serde::Serialize;

use super::managed::{ManagedKvCacheManager, ManagedKvModelRuntime, ManagedStateCapacityRequest};
use super::physical::PhysicalStateManager;
use super::retained_static_attention::RetainedStaticAttentionRuntimeV2;
use crate::backends::BackendKind;
use crate::engine::ModelInstanceId;
use crate::error::{Error, Result};
use crate::kv::v2::{
    InferenceStateContract, ResolvedStatePlan, StateDomainId, StateDomainSpec, StatePlanId,
    StateScope,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub(crate) struct CompositeRetainedStateRuntimeIdV2 {
    pub(crate) model_instance: ModelInstanceId,
    pub(crate) full_state_plan: StatePlanId,
    pub(crate) paged_runtime: crate::kv::KvPlanId,
    pub(crate) static_runtime: super::retained_static_attention::RetainedStaticAttentionRuntimeIdV2,
    pub(crate) static_domain: StateDomainId,
}

/// One load-owned retained runtime whose two physical backings are sealed to
/// one full semantic contract. Static memory is installed before decoder
/// execution and then remains immutable while paged transactions advance.
#[derive(Debug)]
pub(crate) struct CompositeRetainedStateRuntimeV2 {
    id: CompositeRetainedStateRuntimeIdV2,
    full_state_plan: Arc<ResolvedStatePlan>,
    paged: Arc<ManagedKvModelRuntime>,
    static_attention: Arc<RetainedStaticAttentionRuntimeV2>,
}

impl CompositeRetainedStateRuntimeV2 {
    pub(crate) fn new(
        full_contract: &InferenceStateContract,
        full_state_plan: Arc<ResolvedStatePlan>,
        paged: Arc<ManagedKvModelRuntime>,
        static_attention: Arc<RetainedStaticAttentionRuntimeV2>,
        static_domain: StateDomainId,
    ) -> Result<Self> {
        full_contract.validate()?;
        let fingerprint = full_contract.fingerprint()?;
        if full_state_plan.contract_fingerprint != fingerprint
            || static_attention.state_plan_v2().contract_fingerprint != fingerprint
            || static_attention.state_plan_v2().id != full_state_plan.id
        {
            return Err(invalid(
                "composite retained state does not share one full contract/plan identity",
            ));
        }
        if full_state_plan.backend != paged.state_plan_v2().backend
            || full_state_plan.device_ordinal != paged.state_plan_v2().device_ordinal
            || full_state_plan.backend != static_attention.state_plan_v2().backend
            || full_state_plan.device_ordinal != static_attention.state_plan_v2().device_ordinal
        {
            return Err(invalid(
                "composite retained state backings target different physical devices",
            ));
        }
        let model_instance = paged.plan().model_instance;
        if static_attention.id().model_instance != model_instance
            || static_attention.id().domain != static_domain
        {
            return Err(invalid(
                "composite retained state backings have different model/domain identities",
            ));
        }
        if static_attention.sequence_capacity() == 0 || paged.logical_token_reach() == 0 {
            return Err(invalid(
                "composite retained state has zero sequence or token capacity",
            ));
        }
        let selected = full_state_plan
            .non_paged
            .iter()
            .find(|domain| domain.domain() == static_domain)
            .ok_or_else(|| invalid("composite full plan is missing its selected static domain"))?;
        if !matches!(
            selected,
            crate::kv::v2::ResolvedNonPagedDomainPlan::StaticAttention(_)
        ) || full_state_plan.non_paged.len() != 1
        {
            return Err(invalid(
                "composite full plan must contain exactly one selected static-attention domain",
            ));
        }
        if full_state_plan.paged_attention != paged.state_plan_v2().paged_attention
            || !paged.state_plan_v2().non_paged.is_empty()
        {
            return Err(invalid(
                "composite paged projection does not exactly match the full resolved plan",
            ));
        }
        Ok(Self {
            id: CompositeRetainedStateRuntimeIdV2 {
                model_instance,
                full_state_plan: full_state_plan.id,
                paged_runtime: paged.plan().id,
                static_runtime: static_attention.id(),
                static_domain,
            },
            full_state_plan,
            paged,
            static_attention,
        })
    }

    pub(crate) const fn id(&self) -> CompositeRetainedStateRuntimeIdV2 {
        self.id
    }

    pub(crate) fn state_plan_v2(&self) -> &ResolvedStatePlan {
        &self.full_state_plan
    }

    pub(crate) fn paged(&self) -> Arc<ManagedKvModelRuntime> {
        self.paged.clone()
    }

    pub(crate) fn static_attention(&self) -> Arc<RetainedStaticAttentionRuntimeV2> {
        self.static_attention.clone()
    }
}

/// Canonical paged-only projection for a full retained paged+static contract.
/// Mixed consistency groups are rejected because projecting one would weaken
/// its authored transaction closure.
pub(crate) fn project_composite_paged_contract(
    contract: &InferenceStateContract,
    static_domain: StateDomainId,
) -> Result<InferenceStateContract> {
    contract.validate()?;
    let mut paged_ids = HashSet::new();
    let mut selected_static = false;
    for domain in &contract.domains {
        if domain.scope() != StateScope::Retained {
            return Err(invalid(
                "composite retained contract contains a non-retained domain",
            ));
        }
        match domain {
            StateDomainSpec::PagedAttention(_) => {
                paged_ids.insert(domain.id());
            }
            StateDomainSpec::StaticAttention(_) if domain.id() == static_domain => {
                selected_static = true;
            }
            _ => {
                return Err(invalid(
                    "composite retained contract supports paged domains plus one selected static-attention domain",
                ));
            }
        }
    }
    if paged_ids.is_empty() || !selected_static {
        return Err(invalid(
            "composite retained contract is missing paged or selected static state",
        ));
    }
    for group in &contract.groups {
        let has_paged = group
            .domains
            .iter()
            .any(|domain| paged_ids.contains(domain));
        let has_static = group.domains.contains(&static_domain);
        if has_paged && has_static {
            return Err(invalid(
                "composite retained projection cannot split one consistency group",
            ));
        }
    }
    let projected = InferenceStateContract {
        abi: contract.abi,
        domains: contract
            .domains
            .iter()
            .filter(|domain| paged_ids.contains(&domain.id()))
            .cloned()
            .collect(),
        groups: contract
            .groups
            .iter()
            .filter(|group| {
                group
                    .domains
                    .iter()
                    .all(|domain| paged_ids.contains(domain))
            })
            .cloned()
            .collect(),
    };
    projected.validate()?;
    Ok(projected)
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn allocate_composite_retained_state(
    managed: &mut ManagedKvCacheManager,
    physical: &mut PhysicalStateManager,
    model_instance: ModelInstanceId,
    backend: BackendKind,
    capacity: ManagedStateCapacityRequest,
    page_tokens_hint: usize,
    full_contract: &InferenceStateContract,
    static_domain: StateDomainId,
) -> Result<Arc<CompositeRetainedStateRuntimeV2>> {
    if managed.contains_model(model_instance) || physical.contains_model(model_instance) {
        return Err(invalid(
            "composite retained state requires a fresh model generation",
        ));
    }
    let paged_contract = project_composite_paged_contract(full_contract, static_domain)?;
    let page_tokens_u32 = u32::try_from(page_tokens_hint)
        .map_err(|_| invalid("composite retained page size exceeds u32"))?;
    let full_plan = physical.resolve_state_plan(full_contract, Some(page_tokens_u32))?;
    let paged = managed.bind_model_state_with_capacity(
        model_instance,
        backend,
        capacity,
        page_tokens_hint,
        &paged_contract,
    )?;
    #[cfg(test)]
    let synchronize = if managed.take_composite_synchronize_failure() {
        Err(Error::InferenceError(
            "injected composite paged synchronization failure".into(),
        ))
    } else {
        paged.synchronize_backing()
    };
    #[cfg(not(test))]
    let synchronize = paged.synchronize_backing();
    if let Err(error) = synchronize {
        drop(paged);
        managed.unload_model(model_instance).map_err(|rollback| {
            Error::InferenceError(format!(
                "composite paged synchronization failed ({error}); paged rollback failed: {rollback}"
            ))
        })?;
        return Err(error);
    }
    let static_attention = match physical.allocate_retained_static_attention_with_plan(
        model_instance,
        full_contract,
        full_plan.clone(),
        static_domain,
        capacity.retained_sequence_rows,
    ) {
        Ok(runtime) => runtime,
        Err(error) => {
            drop(paged);
            managed.unload_model(model_instance).map_err(|rollback| {
                Error::InferenceError(format!(
                    "composite static allocation failed ({error}); paged rollback failed: {rollback}"
                ))
            })?;
            return Err(error);
        }
    };
    match CompositeRetainedStateRuntimeV2::new(
        full_contract,
        full_plan,
        paged.clone(),
        static_attention.clone(),
        static_domain,
    ) {
        Ok(runtime) => Ok(Arc::new(runtime)),
        Err(error) => {
            drop((paged, static_attention));
            let physical_rollback = physical.unload_model(model_instance);
            let paged_rollback = managed.unload_model(model_instance);
            match (physical_rollback, paged_rollback) {
                (Ok(_), Ok(_)) => Err(error),
                (physical, paged) => Err(Error::InferenceError(format!(
                    "composite identity validation failed ({error}); physical rollback={physical:?}; paged rollback={paged:?}"
                ))),
            }
        }
    }
}

pub(crate) fn unload_composite_retained_state(
    managed: &mut ManagedKvCacheManager,
    physical: &mut PhysicalStateManager,
    model_instance: ModelInstanceId,
) -> Result<bool> {
    unload_composite_retained_state_with_runtime_owners(managed, physical, model_instance, 1)
}

pub(crate) fn unload_composite_retained_state_with_runtime_owners(
    managed: &mut ManagedKvCacheManager,
    physical: &mut PhysicalStateManager,
    model_instance: ModelInstanceId,
    expected_paged_runtime_owners: usize,
) -> Result<bool> {
    let physical_prepared = physical.prepare_unload_model(model_instance);
    let paged_prepared = managed
        .prepare_unload_model_with_runtime_owners(model_instance, expected_paged_runtime_owners);
    let (physical_present, paged_present) = match (physical_prepared, paged_prepared) {
        (Ok(physical), Ok(paged)) => (physical, paged),
        (physical, paged) => {
            return Err(Error::InferenceError(format!(
            "composite unload did not fence every backing: physical={physical:?}; paged={paged:?}"
        )))
        }
    };
    let physical_removed =
        physical_present && physical.commit_prepared_unload_model(model_instance);
    let paged_removed = paged_present && managed.commit_prepared_unload_model(model_instance);
    Ok(physical_removed || paged_removed)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    use crate::backends::state::StaticAttentionLayerValue;
    use crate::engine::{
        CapacitySource, InputRange, PhysicalCapacityProvider, PhysicalCapacitySnapshot,
        ResourceAmount, ResourceAuthority, ResourceVector, SequencePhase, SessionKey, WorkUnit,
    };
    use crate::kv::v2::{
        test_contract, CheckpointPolicy, KeyEncoding, PlacementPolicy, PrefixPolicy, StateClock,
        StateDType, StateDomainHeader, StateGroupId, StateGroupSpec, StaticAttentionDomainSpec,
        StaticAttentionLayerSpec,
    };

    #[derive(Debug)]
    struct TestCapacityProvider {
        capacity: ResourceVector,
    }

    impl PhysicalCapacityProvider for TestCapacityProvider {
        fn snapshot(&self) -> PhysicalCapacitySnapshot {
            PhysicalCapacitySnapshot {
                capacity: self.capacity,
                available: self.capacity,
                source: CapacitySource::Test,
            }
        }
    }

    pub(crate) fn contract() -> InferenceStateContract {
        let mut contract = test_contract();
        let StateDomainSpec::PagedAttention(paged) = &mut contract.domains[0] else {
            unreachable!("test contract owns paged state")
        };
        paged.header.scope = StateScope::Retained;
        paged.header.prefix = PrefixPolicy::Disabled;
        paged.header.checkpoint = CheckpointPolicy::None;
        paged.accepted_dtypes = vec![StateDType::F32];
        paged.layers[0].query_heads = 2;
        paged.layers[0].kv_heads = 2;
        paged.layers[0].key_head_dim = 2;
        paged.layers[0].value_head_dim = 2;
        paged.layers[0].key_encoding = KeyEncoding::Raw;
        contract.groups[0].prefix_shareable = false;
        contract.domains.push(StateDomainSpec::StaticAttention(
            StaticAttentionDomainSpec {
                header: StateDomainHeader {
                    id: StateDomainId::new(2),
                    scope: StateScope::Retained,
                    clock: StateClock::EncoderTokens,
                    placement: PlacementPolicy::BackendLocal,
                    prefix: PrefixPolicy::Disabled,
                    checkpoint: CheckpointPolicy::None,
                },
                layers: vec![StaticAttentionLayerSpec {
                    model_layer: 0,
                    query_heads: 2,
                    kv_heads: 2,
                    key_head_dim: 2,
                    value_head_dim: 2,
                    key_encoding: KeyEncoding::Raw,
                }],
                max_memory_tokens: 4,
                accepted_dtypes: vec![StateDType::F32],
            },
        ));
        contract.groups.push(StateGroupSpec {
            id: StateGroupId::new(2),
            domains: vec![StateDomainId::new(2)],
            prefix_shareable: false,
        });
        contract.validate().unwrap();
        contract
    }

    fn capacity() -> ManagedStateCapacityRequest {
        ManagedStateCapacityRequest {
            total_paged_pages: 8,
            logical_token_reach: Some(32),
            retained_sequence_rows: 2,
            staged_transaction_rows: 2,
        }
    }

    fn static_layer() -> StaticAttentionLayerValue {
        StaticAttentionLayerValue {
            model_layer: 0,
            keys: Tensor::from_vec(
                vec![1.0_f32, 0.0, 0.0, 1.0, 0.5, 0.5, -0.5, 0.5],
                (2, 2, 2),
                &Device::Cpu,
            )
            .unwrap(),
            values: Tensor::from_vec(
                vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                (2, 2, 2),
                &Device::Cpu,
            )
            .unwrap(),
        }
    }

    fn prefill_work() -> WorkUnit {
        WorkUnit::SequenceStep {
            phase: SequencePhase::Prefill,
            input: InputRange { start: 0, end: 1 },
            max_output_steps: 1,
        }
    }

    #[test]
    fn projection_preserves_paged_identity_and_rejects_split_groups() {
        let full = contract();
        let projected = project_composite_paged_contract(&full, StateDomainId::new(2)).unwrap();
        assert_eq!(projected.domains, vec![full.domains[0].clone()]);
        assert_eq!(projected.groups, vec![full.groups[0].clone()]);

        let mut split = full;
        split.groups = vec![StateGroupSpec {
            id: StateGroupId::new(1),
            domains: vec![StateDomainId::new(1), StateDomainId::new(2)],
            prefix_shareable: false,
        }];
        assert!(project_composite_paged_contract(&split, StateDomainId::new(2)).is_err());
    }

    #[test]
    fn mixed_runtime_authenticates_static_install_and_paged_transaction() {
        let model = ModelInstanceId::new(70);
        let mut managed = ManagedKvCacheManager::default();
        let mut physical = PhysicalStateManager::cpu(None);
        let runtime = allocate_composite_retained_state(
            &mut managed,
            &mut physical,
            model,
            BackendKind::Cpu,
            capacity(),
            16,
            &contract(),
            StateDomainId::new(2),
        )
        .unwrap();
        assert_eq!(runtime.id().model_instance, model);
        assert_eq!(
            runtime.state_plan_v2().paged_attention,
            runtime.paged().state_plan_v2().paged_attention
        );
        let static_runtime = runtime.static_attention();
        let static_sequence = static_runtime.register_sequence().unwrap();
        static_runtime
            .install(static_sequence, [8; 32], vec![static_layer()])
            .unwrap();
        let paged = runtime.paged();
        let session = SessionKey::new("composite-row".into(), 1);
        let reservation = managed
            .prepare(&paged, 91, &session, &prefill_work(), None)
            .unwrap()
            .unwrap();
        managed.finalize(&reservation, None, false).unwrap();
        managed.release_session(&session).unwrap();
        static_runtime.release_sequence(static_sequence).unwrap();
        drop((paged, static_runtime, runtime));
        assert!(unload_composite_retained_state(&mut managed, &mut physical, model).unwrap());
        assert!(!managed.contains_model(model));
        assert!(!physical.contains_model(model));
    }

    #[test]
    fn failed_second_backing_allocation_rolls_back_first_backing_and_resources() {
        let authority = Arc::new(ResourceAuthority::new(Arc::new(TestCapacityProvider {
            capacity: ResourceVector::zero(),
        })));
        let model = ModelInstanceId::new(71);
        let mut managed = ManagedKvCacheManager::default();
        let mut physical = PhysicalStateManager::cpu(Some(authority.clone()));
        assert!(allocate_composite_retained_state(
            &mut managed,
            &mut physical,
            model,
            BackendKind::Cpu,
            capacity(),
            16,
            &contract(),
            StateDomainId::new(2),
        )
        .is_err());
        assert!(!managed.contains_model(model));
        assert!(!physical.contains_model(model));
        assert_eq!(authority.snapshot().reservations, 0);
        assert_eq!(
            authority.snapshot().reserved.host_bytes,
            ResourceAmount::Known(0)
        );
    }

    #[test]
    fn synchronize_failure_rolls_back_managed_backing_and_exact_resources() {
        let capacity_vector = ResourceVector {
            host_bytes: ResourceAmount::Known(u64::MAX),
            ..ResourceVector::zero()
        };
        let authority = Arc::new(ResourceAuthority::new(Arc::new(TestCapacityProvider {
            capacity: capacity_vector,
        })));
        let model = ModelInstanceId::new(73);
        let mut managed = ManagedKvCacheManager::new(Some(authority.clone()));
        managed.inject_composite_synchronize_failure();
        let mut physical = PhysicalStateManager::cpu(None);
        assert!(allocate_composite_retained_state(
            &mut managed,
            &mut physical,
            model,
            BackendKind::Cpu,
            capacity(),
            16,
            &contract(),
            StateDomainId::new(2),
        )
        .is_err());
        assert!(!managed.contains_model(model));
        assert!(!physical.contains_model(model));
        assert_eq!(authority.snapshot().reservations, 0);
        assert_eq!(
            authority.snapshot().reserved.host_bytes,
            ResourceAmount::Known(0)
        );
    }

    #[test]
    fn two_phase_unload_removes_neither_backing_when_paged_owner_is_live() {
        let model = ModelInstanceId::new(72);
        let mut managed = ManagedKvCacheManager::default();
        let mut physical = PhysicalStateManager::cpu(None);
        let runtime = allocate_composite_retained_state(
            &mut managed,
            &mut physical,
            model,
            BackendKind::Cpu,
            capacity(),
            16,
            &contract(),
            StateDomainId::new(2),
        )
        .unwrap();
        assert!(unload_composite_retained_state(&mut managed, &mut physical, model).is_err());
        assert!(managed.contains_model(model));
        assert!(physical.contains_model(model));
        assert!(runtime.static_attention().register_sequence().is_err());
        drop(runtime);
        assert!(unload_composite_retained_state(&mut managed, &mut physical, model).unwrap());
    }

    #[test]
    fn active_static_owner_still_fences_managed_backing_on_failed_unload() {
        let model = ModelInstanceId::new(74);
        let mut managed = ManagedKvCacheManager::default();
        let mut physical = PhysicalStateManager::cpu(None);
        let runtime = allocate_composite_retained_state(
            &mut managed,
            &mut physical,
            model,
            BackendKind::Cpu,
            capacity(),
            16,
            &contract(),
            StateDomainId::new(2),
        )
        .unwrap();
        let static_runtime = runtime.static_attention();
        let sequence = static_runtime.register_sequence().unwrap();
        drop(runtime);
        assert!(unload_composite_retained_state(&mut managed, &mut physical, model).is_err());
        assert!(managed.contains_model(model));
        assert!(physical.contains_model(model));
        assert!(managed
            .require_loaded_runtime(
                model,
                BackendKind::Cpu,
                &crate::kv::InferenceStateCapability::Managed(
                    project_composite_paged_contract(&contract(), StateDomainId::new(2)).unwrap(),
                ),
            )
            .is_err());
        static_runtime.release_sequence(sequence).unwrap();
        drop(static_runtime);
        assert!(unload_composite_retained_state(&mut managed, &mut physical, model).unwrap());
    }

    #[test]
    fn reallocation_changes_both_allocation_qualified_runtime_identities() {
        let model = ModelInstanceId::new(75);
        let mut managed = ManagedKvCacheManager::default();
        let mut physical = PhysicalStateManager::cpu(None);
        let first = allocate_composite_retained_state(
            &mut managed,
            &mut physical,
            model,
            BackendKind::Cpu,
            capacity(),
            16,
            &contract(),
            StateDomainId::new(2),
        )
        .unwrap();
        let first_id = first.id();
        drop(first);
        unload_composite_retained_state(&mut managed, &mut physical, model).unwrap();
        let second = allocate_composite_retained_state(
            &mut managed,
            &mut physical,
            model,
            BackendKind::Cpu,
            capacity(),
            16,
            &contract(),
            StateDomainId::new(2),
        )
        .unwrap();
        assert_ne!(first_id.paged_runtime, second.id().paged_runtime);
        assert_ne!(first_id.static_runtime, second.id().static_runtime);
        drop(second);
        unload_composite_retained_state(&mut managed, &mut physical, model).unwrap();
    }
}
