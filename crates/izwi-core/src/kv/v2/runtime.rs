use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{Arc, Weak};

use crate::backends::state::StateBackendRegistry;
use crate::backends::BackendKind;
use crate::engine::{
    AdapterAbiRevision, AdapterInstanceId, ExecutionAdapterBinding, ExecutionGroupId,
    ModelInstanceId,
};
use crate::engine::{InvocationPagedKvLease, InvocationPagedKvPoolHandle, InvocationPagedKvPoolId};
use crate::engine::{ManagedKvModelRuntime, StageId};
use crate::error::{Error, Result};
use crate::model::ModelVariant;

use super::{
    stage_graph_fingerprint, CapabilityStateDescriptorV2, InvocationWorkspaceDomain,
    InvocationWorkspaceSet, ResolvedStatePlan, RetainedStateCapability, StateDomainId,
};

const STATELESS_RUNTIME_FINGERPRINT_DOMAIN: &[u8] = b"izwi.inference-state.stateless-runtime.v2\0";
const MANAGED_RUNTIME_FINGERPRINT_DOMAIN: &[u8] = b"izwi.inference-state.managed-runtime.v2\0";

/// Whether the selected execution graph actually acquires the capability's
/// retained physical state. A capability may own one load-scoped arena while
/// some of its exact request graphs remain cacheless (for example offline ASR
/// versus incremental streaming ASR).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum RetainedStateUseV2 {
    Inactive,
    ExternalPaged,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub(crate) struct InvocationPagedWorkspaceKeyV2 {
    pub(crate) stage_graph: [u8; 32],
    pub(crate) stage: StageId,
    pub(crate) domain: StateDomainId,
}

#[derive(Debug, Clone)]
pub(crate) struct InvocationPagedWorkspaceBindingV2 {
    pub(crate) key: InvocationPagedWorkspaceKeyV2,
    pub(crate) pool: InvocationPagedKvPoolHandle,
}

/// Load-sealed physical invocation page pools. Reusing one pool across exact
/// graph/stage keys is explicit because each key must be published here.
#[derive(Debug, Clone, Default)]
pub(crate) struct InvocationPagedWorkspaceRuntimeV2 {
    pools: HashMap<InvocationPagedWorkspaceKeyV2, InvocationPagedKvPoolHandle>,
}

impl InvocationPagedWorkspaceRuntimeV2 {
    pub(crate) fn new(bindings: Vec<InvocationPagedWorkspaceBindingV2>) -> Result<Self> {
        let mut pools = HashMap::with_capacity(bindings.len());
        for binding in bindings {
            if binding.key.stage_graph.iter().all(|byte| *byte == 0)
                || binding.key.stage.get() == 0
                || binding.key.domain.get() == 0
                || binding.pool.id().domain != binding.key.domain
            {
                return Err(invalid(
                    "invocation paged workspace binding has an incomplete or mismatched identity",
                ));
            }
            if pools.insert(binding.key, binding.pool).is_some() {
                return Err(invalid(
                    "invocation paged workspace repeats one graph/stage/domain binding",
                ));
            }
        }
        Ok(Self { pools })
    }

    fn validate_for(
        &self,
        descriptor: &CapabilityStateDescriptorV2,
        stages: &[crate::engine::StageDescriptor],
    ) -> Result<()> {
        let graph = stage_graph_fingerprint(stages)?;
        let expected = match &descriptor.invocation {
            InvocationWorkspaceSet::None { .. } => HashMap::new(),
            InvocationWorkspaceSet::Bounded { profiles } => profiles
                .iter()
                .find(|profile| profile.stage_graph_fingerprint == graph)
                .ok_or_else(|| invalid("invocation runtime has no selected descriptor profile"))?
                .stages
                .iter()
                .flat_map(|stage| {
                    stage.domains.iter().filter_map(move |domain| match domain {
                        InvocationWorkspaceDomain::State {
                            state: super::StateDomainSpec::PagedAttention(_),
                            ..
                        } => Some((
                            InvocationPagedWorkspaceKeyV2 {
                                stage_graph: graph,
                                stage: stage.stage,
                                domain: domain.id(),
                            },
                            domain,
                        )),
                        _ => None,
                    })
                })
                .collect::<HashMap<_, _>>(),
        };
        let actual = self
            .pools
            .iter()
            .filter(|(key, _)| key.stage_graph == graph)
            .collect::<HashMap<_, _>>();
        if actual.len() != expected.len() {
            return Err(invalid(
                "invocation paged workspace backing does not cover the selected descriptor",
            ));
        }
        for (key, domain) in expected {
            let pool = self.pools.get(&key).ok_or_else(|| {
                invalid("invocation paged workspace is missing a graph/stage/domain pool")
            })?;
            pool.validate_live()?;
            if pool.workspace_domain() != domain {
                return Err(invalid(
                    "invocation paged workspace pool does not match its authored domain",
                ));
            }
        }
        Ok(())
    }

    fn pool_ids_for_graph(
        &self,
        graph: [u8; 32],
    ) -> Vec<(InvocationPagedWorkspaceKeyV2, InvocationPagedKvPoolId)> {
        let mut ids = self
            .pools
            .iter()
            .filter(|(key, _)| key.stage_graph == graph)
            .map(|(key, pool)| (*key, pool.id()))
            .collect::<Vec<_>>();
        ids.sort_unstable_by_key(|(key, _)| (key.stage, key.domain));
        ids
    }

    fn lease(
        &self,
        graph: [u8; 32],
        stage: StageId,
        domain: StateDomainId,
    ) -> Result<InvocationPagedKvLease> {
        self.pools
            .get(&InvocationPagedWorkspaceKeyV2 {
                stage_graph: graph,
                stage,
                domain,
            })
            .ok_or_else(|| invalid("selected invocation paged workspace is not load-sealed"))?
            .lease()
    }
}

/// Canonical identity shared by every retained/workspace/runtime plan for one
/// exact loaded capability. Pool sharing, when introduced, must be an explicit
/// authorization between identities rather than an accidental fingerprint
/// collision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct CapabilityRuntimeIdentityV2 {
    pub(crate) execution_group: ExecutionGroupId,
    pub(crate) model_instance: ModelInstanceId,
    pub(crate) model_variant: ModelVariant,
    pub(crate) backend: BackendKind,
    pub(crate) capability_id: String,
    pub(crate) adapter_instance: AdapterInstanceId,
    pub(crate) adapter_abi: AdapterAbiRevision,
}

impl CapabilityRuntimeIdentityV2 {
    pub(crate) fn seal(backend: BackendKind, execution: &ExecutionAdapterBinding) -> Result<Self> {
        execution.validate()?;
        Ok(Self {
            execution_group: execution.execution_group_id,
            model_instance: execution.model_instance_id,
            model_variant: execution.model_variant,
            backend,
            capability_id: execution.capability_id.clone(),
            adapter_instance: execution.adapter_instance_id,
            adapter_abi: execution.adapter_abi_revision,
        })
    }

    pub(crate) fn validate_against(
        &self,
        backend: BackendKind,
        execution: &ExecutionAdapterBinding,
    ) -> Result<()> {
        execution.validate()?;
        if self.backend != backend
            || self.execution_group != execution.execution_group_id
            || self.model_instance != execution.model_instance_id
            || self.model_variant != execution.model_variant
            || self.capability_id != execution.capability_id
            || self.adapter_instance != execution.adapter_instance_id
            || self.adapter_abi != execution.adapter_abi_revision
        {
            return Err(invalid(
                "capability runtime identity does not match the selected loaded adapter",
            ));
        }
        Ok(())
    }
}

/// Immutable request-selectable proof that one exact loaded capability has no
/// retained session state. Its invocation workspace may still be bounded by
/// the descriptor and is leased by the engine's physical-batch authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct StatelessCapabilityRuntimeV2 {
    pub(crate) id: [u8; 32],
    pub(crate) identity: CapabilityRuntimeIdentityV2,
    pub(crate) stage_graph_fingerprint: [u8; 32],
    pub(crate) state_fingerprint: [u8; 32],
    pub(crate) descriptor: CapabilityStateDescriptorV2,
}

/// Request-facing inference-state runtime. Callers bind this model-neutral
/// handle and never branch on whether the backing is stateless, paged KV, or a
/// future tensor/ring arena. The backing kind remains private to the state
/// runtime so adding a physical domain cannot create another request ABI.
#[derive(Debug, Clone)]
pub(crate) struct CapabilityStateRuntimeV2 {
    pub(crate) id: [u8; 32],
    pub(crate) state_fingerprint: [u8; 32],
    pub(crate) descriptor: CapabilityStateDescriptorV2,
    backing: CapabilityStateRuntimeBackingV2,
}

#[derive(Debug, Clone)]
enum CapabilityStateRuntimeBackingV2 {
    Stateless(StatelessCapabilityRuntimeV2),
    Managed(ManagedCapabilityRuntimeV2),
}

impl PartialEq for CapabilityStateRuntimeV2 {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.state_fingerprint == other.state_fingerprint
            && self.descriptor == other.descriptor
    }
}

impl Eq for CapabilityStateRuntimeV2 {}

impl CapabilityStateRuntimeV2 {
    pub(crate) fn stateless(runtime: StatelessCapabilityRuntimeV2) -> Self {
        Self {
            id: runtime.id,
            state_fingerprint: runtime.state_fingerprint,
            descriptor: runtime.descriptor.clone(),
            backing: CapabilityStateRuntimeBackingV2::Stateless(runtime),
        }
    }

    pub(crate) fn managed(runtime: ManagedCapabilityRuntimeV2) -> Self {
        Self {
            id: runtime.id,
            state_fingerprint: runtime.state_fingerprint,
            descriptor: runtime.descriptor.clone(),
            backing: CapabilityStateRuntimeBackingV2::Managed(runtime),
        }
    }

    pub(crate) fn managed_kv_runtime(&self) -> Option<Arc<ManagedKvModelRuntime>> {
        match &self.backing {
            CapabilityStateRuntimeBackingV2::Stateless(_) => None,
            CapabilityStateRuntimeBackingV2::Managed(runtime)
                if runtime.retained_state_use == RetainedStateUseV2::ExternalPaged =>
            {
                runtime.physical.upgrade()
            }
            CapabilityStateRuntimeBackingV2::Managed(_) => None,
        }
    }

    pub(crate) fn lease_invocation_paged(
        &self,
        stage: StageId,
        domain: StateDomainId,
    ) -> Result<InvocationPagedKvLease> {
        match &self.backing {
            CapabilityStateRuntimeBackingV2::Stateless(_) => Err(invalid(
                "stateless runtime has no load-sealed paged invocation workspace",
            )),
            CapabilityStateRuntimeBackingV2::Managed(runtime) => {
                runtime
                    .invocation_paged
                    .lease(runtime.stage_graph_fingerprint, stage, domain)
            }
        }
    }

    pub(crate) fn validate_against(
        &self,
        backend: BackendKind,
        execution: &ExecutionAdapterBinding,
    ) -> Result<()> {
        match &self.backing {
            CapabilityStateRuntimeBackingV2::Stateless(runtime) => {
                runtime.validate_against(backend, execution)?;
                if self.id != runtime.id
                    || self.state_fingerprint != runtime.state_fingerprint
                    || self.descriptor != runtime.descriptor
                {
                    return Err(invalid(
                        "state ABI v2 runtime wrapper does not match its sealed backing",
                    ));
                }
                Ok(())
            }
            CapabilityStateRuntimeBackingV2::Managed(runtime) => {
                runtime.validate_against(backend, execution)?;
                if self.id != runtime.id
                    || self.state_fingerprint != runtime.state_fingerprint
                    || self.descriptor != runtime.descriptor
                {
                    return Err(invalid(
                        "state ABI v2 runtime wrapper does not match its sealed backing",
                    ));
                }
                Ok(())
            }
        }
    }
}

/// Load-sealed proof that one exact capability owns a backend-resolved state
/// plan and the already allocated physical paged arena implementing it.
#[derive(Debug, Clone)]
pub(crate) struct ManagedCapabilityRuntimeV2 {
    pub(crate) id: [u8; 32],
    pub(crate) identity: CapabilityRuntimeIdentityV2,
    pub(crate) stage_graph_fingerprint: [u8; 32],
    pub(crate) state_fingerprint: [u8; 32],
    pub(crate) descriptor: CapabilityStateDescriptorV2,
    pub(crate) state_plan: Arc<ResolvedStatePlan>,
    physical_plan: crate::kv::KvPlanId,
    retained_state_use: RetainedStateUseV2,
    /// The lifecycle manager is the physical owner. A sealed adapter proves
    /// the exact generation without pinning that generation through unload;
    /// admitted requests upgrade this weak handle while holding residency.
    physical: Weak<ManagedKvModelRuntime>,
    invocation_paged: InvocationPagedWorkspaceRuntimeV2,
}

impl ManagedCapabilityRuntimeV2 {
    pub(crate) fn seal(
        backend: BackendKind,
        execution: &ExecutionAdapterBinding,
        descriptor: CapabilityStateDescriptorV2,
        physical: Arc<ManagedKvModelRuntime>,
        retained_state_use: RetainedStateUseV2,
    ) -> Result<Self> {
        Self::seal_with_invocation_paged(
            backend,
            execution,
            descriptor,
            physical,
            retained_state_use,
            InvocationPagedWorkspaceRuntimeV2::default(),
        )
    }

    pub(crate) fn seal_with_invocation_paged(
        backend: BackendKind,
        execution: &ExecutionAdapterBinding,
        descriptor: CapabilityStateDescriptorV2,
        physical: Arc<ManagedKvModelRuntime>,
        retained_state_use: RetainedStateUseV2,
        invocation_paged: InvocationPagedWorkspaceRuntimeV2,
    ) -> Result<Self> {
        execution.validate()?;
        descriptor.validate_against_stages(&execution.stages)?;
        let RetainedStateCapability::Managed { contract } = &descriptor.retained else {
            return Err(invalid(
                "managed state ABI v2 runtime requires retained physical state",
            ));
        };
        if execution.model_instance_id != physical.plan().model_instance {
            return Err(invalid(
                "managed state ABI v2 runtime targets a different model instance",
            ));
        }
        let state_plan = Arc::new(physical.state_plan_v2().clone());
        let registry = StateBackendRegistry::new(state_plan.backend, state_plan.device_ordinal)?;
        state_plan.validate_against(contract, &registry)?;
        if backend != state_plan.backend
            || state_plan.id != physical.state_plan_v2().id
            || state_plan.contract_fingerprint != contract.fingerprint()?
        {
            return Err(invalid(
                "managed state ABI v2 runtime does not match its physical state plan",
            ));
        }
        let stage_graph_fingerprint = stage_graph_fingerprint(&execution.stages)?;
        let state_fingerprint = descriptor.fingerprint(&execution.stages)?;
        let mut runtime = Self {
            id: [0; 32],
            identity: CapabilityRuntimeIdentityV2::seal(backend, execution)?,
            stage_graph_fingerprint,
            state_fingerprint,
            descriptor,
            state_plan,
            physical_plan: physical.plan().id,
            retained_state_use,
            physical: Arc::downgrade(&physical),
            invocation_paged,
        };
        runtime.id = runtime.compute_id()?;
        runtime.validate_against(backend, execution)?;
        Ok(runtime)
    }

    fn validate_against(
        &self,
        backend: BackendKind,
        execution: &ExecutionAdapterBinding,
    ) -> Result<()> {
        self.identity.validate_against(backend, execution)?;
        self.descriptor.validate_against_stages(&execution.stages)?;
        self.invocation_paged
            .validate_for(&self.descriptor, &execution.stages)?;
        let RetainedStateCapability::Managed { contract } = &self.descriptor.retained else {
            return Err(invalid("managed runtime lost its retained-state contract"));
        };
        let registry =
            StateBackendRegistry::new(self.state_plan.backend, self.state_plan.device_ordinal)?;
        self.state_plan.validate_against(contract, &registry)?;
        let physical = self.physical.upgrade().ok_or_else(|| {
            invalid("managed state ABI v2 runtime refers to an unloaded physical generation")
        })?;
        if self.stage_graph_fingerprint != stage_graph_fingerprint(&execution.stages)?
            || self.state_fingerprint != self.descriptor.fingerprint(&execution.stages)?
            || self.state_plan.id != physical.state_plan_v2().id
            || self.physical_plan != physical.plan().id
            || execution.model_instance_id != physical.plan().model_instance
            || self.id != self.compute_id()?
        {
            return Err(invalid(
                "managed state ABI v2 runtime does not match the selected loaded capability",
            ));
        }
        Ok(())
    }

    fn compute_id(&self) -> Result<[u8; 32]> {
        #[derive(Serialize)]
        struct Payload<'a> {
            identity: &'a CapabilityRuntimeIdentityV2,
            stage_graph_fingerprint: [u8; 32],
            state_fingerprint: [u8; 32],
            state_plan: super::StatePlanId,
            physical_plan: crate::kv::KvPlanId,
            retained_state_use: RetainedStateUseV2,
            invocation_paged: Vec<(InvocationPagedWorkspaceKeyV2, InvocationPagedKvPoolId)>,
        }
        let encoded = serde_json::to_vec(&Payload {
            identity: &self.identity,
            stage_graph_fingerprint: self.stage_graph_fingerprint,
            state_fingerprint: self.state_fingerprint,
            state_plan: self.state_plan.id,
            physical_plan: self.physical_plan,
            retained_state_use: self.retained_state_use,
            invocation_paged: self
                .invocation_paged
                .pool_ids_for_graph(self.stage_graph_fingerprint),
        })
        .map_err(|error| invalid(format!("failed to encode managed runtime: {error}")))?;
        let mut hasher = Sha256::new();
        hasher.update(MANAGED_RUNTIME_FINGERPRINT_DOMAIN);
        hasher.update(encoded);
        Ok(hasher.finalize().into())
    }
}

impl StatelessCapabilityRuntimeV2 {
    pub(crate) fn seal(
        backend: BackendKind,
        execution: &ExecutionAdapterBinding,
        descriptor: CapabilityStateDescriptorV2,
    ) -> Result<Self> {
        execution.validate()?;
        descriptor.validate_against_stages(&execution.stages)?;
        if !descriptor.is_stateless() {
            return Err(invalid(
                "stateless state ABI v2 runtime cannot seal retained physical state",
            ));
        }
        let stage_graph_fingerprint = stage_graph_fingerprint(&execution.stages)?;
        let state_fingerprint = descriptor.fingerprint(&execution.stages)?;
        let mut runtime = Self {
            id: [0; 32],
            identity: CapabilityRuntimeIdentityV2::seal(backend, execution)?,
            stage_graph_fingerprint,
            state_fingerprint,
            descriptor,
        };
        runtime.id = runtime.compute_id()?;
        runtime.validate_against(backend, execution)?;
        Ok(runtime)
    }

    pub(crate) fn validate_against(
        &self,
        backend: BackendKind,
        execution: &ExecutionAdapterBinding,
    ) -> Result<()> {
        execution.validate()?;
        self.identity.validate_against(backend, execution)?;
        if self.stage_graph_fingerprint != stage_graph_fingerprint(&execution.stages)?
            || self.state_fingerprint != self.descriptor.fingerprint(&execution.stages)?
            || !self.descriptor.is_stateless()
            || self.id != self.compute_id()?
        {
            return Err(invalid(
                "stateless state ABI v2 runtime does not match the selected loaded capability",
            ));
        }
        Ok(())
    }

    fn compute_id(&self) -> Result<[u8; 32]> {
        #[derive(Serialize)]
        struct Payload<'a> {
            identity: &'a CapabilityRuntimeIdentityV2,
            stage_graph_fingerprint: [u8; 32],
            state_fingerprint: [u8; 32],
        }

        let encoded = serde_json::to_vec(&Payload {
            identity: &self.identity,
            stage_graph_fingerprint: self.stage_graph_fingerprint,
            state_fingerprint: self.state_fingerprint,
        })
        .map_err(|error| invalid(format!("failed to encode stateless runtime: {error}")))?;
        let mut hasher = Sha256::new();
        hasher.update(STATELESS_RUNTIME_FINGERPRINT_DOMAIN);
        hasher.update(encoded);
        Ok(hasher.finalize().into())
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use candle_core::DType;

    use super::*;
    use crate::backends::kv::{CpuKvArena, KvArena, KvArenaConfig, KvLayerConfig};
    use crate::backends::state::{negotiate_state_plan, StateBackendPlanRequest};
    use crate::engine::{
        EngineCore, EngineCoreConfig, ExecutionMode, ExecutionProfile, InvocationPagedKvPoolOwner,
        NativeBatchMode, StageDescriptor, StageId,
    };
    use crate::kv::v2::{test_contract, upgrade_kv_contract_v1};
    use crate::kv::v2::{
        CheckpointPolicy, InvocationStageWorkspace, InvocationStateCapacity,
        InvocationWorkspaceProfile, PlacementPolicy, PrefixPolicy, StateDType, StateDomainSpec,
        StateGroupId, StateGroupSpec, StateScope, WorkspaceFormula, CURRENT_INFERENCE_STATE_ABI,
    };
    use crate::kv::{CacheCapability, KvArenaId, KvGroupId, KvLayerBinding};

    fn binding() -> ExecutionAdapterBinding {
        let variant = ModelVariant::Kokoro82M;
        let profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, Some(variant), ExecutionMode::Atomic);
        let stage = StageDescriptor::from_execution_profile(
            StageId::new(1),
            "tts.stateless",
            &profile,
            NativeBatchMode::None,
        );
        ExecutionAdapterBinding {
            execution_group_id: ExecutionGroupId::new(2),
            model_instance_id: ModelInstanceId::new(3),
            adapter_instance_id: AdapterInstanceId::new(4),
            adapter_abi_revision: AdapterAbiRevision::new(5),
            model_variant: variant,
            capability_id: "tts".to_string(),
            stages: Arc::from([stage]),
        }
    }

    fn invocation_pool(
        model_instance: ModelInstanceId,
    ) -> (InvocationPagedKvPoolOwner, InvocationWorkspaceDomain) {
        let mut contract = test_contract();
        let StateDomainSpec::PagedAttention(domain) = &mut contract.domains[0] else {
            unreachable!()
        };
        domain.header.scope = StateScope::Invocation;
        domain.header.prefix = PrefixPolicy::Disabled;
        domain.header.checkpoint = CheckpointPolicy::None;
        domain.accepted_dtypes = vec![StateDType::F32];
        domain.layers[0].query_heads = 2;
        domain.layers[0].kv_heads = 2;
        domain.layers[0].key_head_dim = 4;
        domain.layers[0].value_head_dim = 4;
        domain.layers[0].key_encoding = super::super::KeyEncoding::Rotary { rotary_dim: 4 };
        contract.groups[0].prefix_shareable = false;
        let workspace_domain = InvocationWorkspaceDomain::State {
            state: contract.domains[0].clone(),
            capacity: InvocationStateCapacity::PagedTokens { max_tokens: 16 },
            placement: PlacementPolicy::BackendLocalWithHostOffload,
            formula: WorkspaceFormula {
                fixed_bytes: 1024 * 1024,
                dimensions: vec![],
                terms: vec![],
            },
        };
        let plan = negotiate_state_plan(
            &contract,
            &StateBackendPlanRequest {
                backend: BackendKind::Cpu,
                device_ordinal: None,
                page_tokens_hint: Some(16),
                storage_dtype_hint: Some(StateDType::F32),
            },
        )
        .unwrap();
        let resolved = &plan.paged_attention[0];
        let arena: Arc<dyn KvArena> = Arc::new(
            CpuKvArena::new(KvArenaConfig {
                id: KvArenaId {
                    model_instance,
                    backend: BackendKind::Cpu,
                    device_ordinal: None,
                    generation: 19,
                },
                group: KvGroupId::new(resolved.group.get()),
                page_tokens: resolved.page_tokens,
                capacity_pages: 1,
                dtype: DType::F32,
                layers: resolved
                    .layers
                    .iter()
                    .map(|binding| KvLayerConfig {
                        binding: KvLayerBinding {
                            model_layer: binding.model_layer,
                            physical_layer: binding.physical_layer,
                        },
                        num_kv_heads: 2,
                        key_head_dim: 4,
                        value_head_dim: 4,
                    })
                    .collect(),
            })
            .unwrap(),
        );
        (
            InvocationPagedKvPoolOwner::new(&plan, &workspace_domain, arena, 0, 1, 1, 23).unwrap(),
            workspace_domain,
        )
    }

    fn managed_invocation_fixture() -> (
        ExecutionAdapterBinding,
        CapabilityStateDescriptorV2,
        Arc<ManagedKvModelRuntime>,
        InvocationPagedKvPoolOwner,
    ) {
        let model_instance = ModelInstanceId::new(37);
        let mut execution = binding();
        execution.model_instance_id = model_instance;
        Arc::make_mut(&mut execution.stages)[0].max_workspace_bytes = 1024 * 1024;
        let graph = stage_graph_fingerprint(&execution.stages).unwrap();
        let (pool, workspace_domain) = invocation_pool(model_instance);
        let retained_contract = upgrade_kv_contract_v1(&crate::kv::test_contract()).unwrap();
        let descriptor = CapabilityStateDescriptorV2 {
            abi: CURRENT_INFERENCE_STATE_ABI,
            retained: RetainedStateCapability::Managed {
                contract: retained_contract,
            },
            invocation: InvocationWorkspaceSet::Bounded {
                profiles: vec![InvocationWorkspaceProfile {
                    stage_graph_fingerprint: graph,
                    stages: vec![InvocationStageWorkspace {
                        stage: execution.stages[0].id,
                        groups: vec![StateGroupSpec {
                            id: StateGroupId::new(1),
                            domains: vec![StateDomainId::new(1)],
                            prefix_shareable: false,
                        }],
                        domains: vec![workspace_domain],
                    }],
                }],
            },
        };
        descriptor
            .validate_against_stages(&execution.stages)
            .unwrap();
        let mut core = EngineCore::new(EngineCoreConfig {
            max_blocks: 4,
            block_size: 16,
            ..EngineCoreConfig::default()
        })
        .unwrap();
        let physical = core
            .load_managed_model_cache(
                model_instance,
                &CacheCapability::Managed(crate::kv::test_contract()),
            )
            .unwrap()
            .expect("managed physical cache");
        (execution, descriptor, physical, pool)
    }

    #[test]
    fn stateless_runtime_seals_the_complete_execution_identity() {
        let binding = binding();
        let descriptor = CapabilityStateDescriptorV2::stateless_for_stages_test(&binding.stages);
        let runtime =
            StatelessCapabilityRuntimeV2::seal(BackendKind::Cpu, &binding, descriptor).unwrap();
        runtime
            .validate_against(BackendKind::Cpu, &binding)
            .unwrap();
        assert!(runtime
            .validate_against(BackendKind::Cuda, &binding)
            .is_err());

        let mut changed = binding.clone();
        changed.execution_group_id = ExecutionGroupId::new(20);
        assert!(runtime
            .validate_against(BackendKind::Cpu, &changed)
            .is_err());
        let mut changed = binding.clone();
        changed.model_instance_id = ModelInstanceId::new(30);
        assert!(runtime
            .validate_against(BackendKind::Cpu, &changed)
            .is_err());
        let mut changed = binding.clone();
        changed.model_variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        assert!(runtime
            .validate_against(BackendKind::Cpu, &changed)
            .is_err());
        let mut changed = binding.clone();
        changed.adapter_instance_id = AdapterInstanceId::new(40);
        assert!(runtime
            .validate_against(BackendKind::Cpu, &changed)
            .is_err());
        let mut changed = binding.clone();
        changed.adapter_abi_revision = AdapterAbiRevision::new(50);
        assert!(runtime
            .validate_against(BackendKind::Cpu, &changed)
            .is_err());
        let mut changed = binding.clone();
        changed.capability_id = "streaming_tts".to_string();
        assert!(runtime
            .validate_against(BackendKind::Cpu, &changed)
            .is_err());
        let mut changed = binding;
        Arc::make_mut(&mut changed.stages)[0].name = "tts.changed".to_string();
        assert!(runtime
            .validate_against(BackendKind::Cpu, &changed)
            .is_err());
    }

    #[test]
    fn stateless_runtime_seals_bounded_invocation_workspace() {
        let mut binding = binding();
        Arc::make_mut(&mut binding.stages)[0].max_workspace_bytes = 4096;
        let descriptor =
            CapabilityStateDescriptorV2::stateless_for_stage_graphs(&[binding.stages.as_ref()])
                .unwrap();
        assert!(!descriptor
            .has_zero_invocation_workspace_for(&binding.stages)
            .unwrap());
        StatelessCapabilityRuntimeV2::seal(BackendKind::Cpu, &binding, descriptor)
            .unwrap()
            .validate_against(BackendKind::Cpu, &binding)
            .unwrap();
    }

    #[test]
    fn invocation_runtime_rejects_duplicate_or_mismatched_bindings() {
        let (execution, _, _, pool) = managed_invocation_fixture();
        let handle = pool.handle();
        let key = InvocationPagedWorkspaceKeyV2 {
            stage_graph: stage_graph_fingerprint(&execution.stages).unwrap(),
            stage: execution.stages[0].id,
            domain: StateDomainId::new(1),
        };
        assert!(InvocationPagedWorkspaceRuntimeV2::new(vec![
            InvocationPagedWorkspaceBindingV2 {
                key,
                pool: handle.clone(),
            },
            InvocationPagedWorkspaceBindingV2 {
                key,
                pool: handle.clone(),
            },
        ])
        .is_err());
        let wrong_key = InvocationPagedWorkspaceKeyV2 {
            domain: StateDomainId::new(99),
            ..key
        };
        assert!(
            InvocationPagedWorkspaceRuntimeV2::new(vec![InvocationPagedWorkspaceBindingV2 {
                key: wrong_key,
                pool: handle,
            },])
            .is_err()
        );
    }

    #[test]
    fn managed_runtime_requires_exact_invocation_backing_before_seal() {
        let (execution, descriptor, physical, pool) = managed_invocation_fixture();
        assert!(ManagedCapabilityRuntimeV2::seal_with_invocation_paged(
            BackendKind::Cpu,
            &execution,
            descriptor.clone(),
            physical.clone(),
            RetainedStateUseV2::ExternalPaged,
            InvocationPagedWorkspaceRuntimeV2::default(),
        )
        .is_err());

        let stage = execution.stages[0].id;
        let domain = StateDomainId::new(1);
        let invocation =
            InvocationPagedWorkspaceRuntimeV2::new(vec![InvocationPagedWorkspaceBindingV2 {
                key: InvocationPagedWorkspaceKeyV2 {
                    stage_graph: stage_graph_fingerprint(&execution.stages).unwrap(),
                    stage,
                    domain,
                },
                pool: pool.handle(),
            }])
            .unwrap();
        let runtime = CapabilityStateRuntimeV2::managed(
            ManagedCapabilityRuntimeV2::seal_with_invocation_paged(
                BackendKind::Cpu,
                &execution,
                descriptor,
                physical,
                RetainedStateUseV2::ExternalPaged,
                invocation,
            )
            .unwrap(),
        );
        let lease = runtime.lease_invocation_paged(stage, domain).unwrap();
        assert_eq!(lease.cache().context_len(), 0);
        lease.release().unwrap();
        drop(pool);
        assert!(runtime.lease_invocation_paged(stage, domain).is_err());
        assert!(runtime
            .validate_against(BackendKind::Cpu, &execution)
            .is_err());
    }
}
