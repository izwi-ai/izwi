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
use crate::engine::{
    InvocationPagedKvCompletion, InvocationPagedKvLease, InvocationPagedKvPoolHandle,
    InvocationPagedKvPoolId,
};
use crate::engine::{
    ManagedKvModelRuntime, RetainedTensorStateRuntimeIdV2, RetainedTensorStateRuntimeV2, StageId,
};
use crate::error::{Error, Result};
use crate::model::ModelVariant;

use super::{
    stage_graph_fingerprint, CapabilityStateDescriptorV2, InvocationWorkspaceDomain,
    InvocationWorkspaceSet, ResolvedStatePlan, RetainedStateCapability, StateDomainId,
};

const STATELESS_RUNTIME_FINGERPRINT_DOMAIN: &[u8] = b"izwi.inference-state.stateless-runtime.v2\0";
const INVOCATION_RUNTIME_FINGERPRINT_DOMAIN: &[u8] =
    b"izwi.inference-state.invocation-runtime.v2\0";
const MANAGED_RUNTIME_FINGERPRINT_DOMAIN: &[u8] = b"izwi.inference-state.managed-runtime.v2\0";

/// Whether the selected execution graph actually acquires the capability's
/// retained physical state. A capability may own one load-scoped arena while
/// some of its exact request graphs remain cacheless (for example offline ASR
/// versus incremental streaming ASR).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum RetainedStateUseV2 {
    Inactive,
    ExternalPaged,
    ExternalTensor,
}

#[derive(Debug, Clone)]
pub(crate) enum RetainedStateRuntimeV2 {
    Paged(Arc<ManagedKvModelRuntime>),
    Tensor(Arc<RetainedTensorStateRuntimeV2>),
}

impl From<Arc<ManagedKvModelRuntime>> for RetainedStateRuntimeV2 {
    fn from(runtime: Arc<ManagedKvModelRuntime>) -> Self {
        Self::Paged(runtime)
    }
}

impl From<Arc<RetainedTensorStateRuntimeV2>> for RetainedStateRuntimeV2 {
    fn from(runtime: Arc<RetainedTensorStateRuntimeV2>) -> Self {
        Self::Tensor(runtime)
    }
}

impl RetainedStateRuntimeV2 {
    pub(crate) fn model_instance(&self) -> ModelInstanceId {
        match self {
            Self::Paged(runtime) => runtime.plan().model_instance,
            Self::Tensor(runtime) => runtime.id().model_instance,
        }
    }

    pub(crate) fn state_plan_v2(&self) -> &ResolvedStatePlan {
        match self {
            Self::Paged(runtime) => runtime.state_plan_v2(),
            Self::Tensor(runtime) => runtime.state_plan_v2(),
        }
    }

    pub(crate) const fn is_tensor_only(&self) -> bool {
        matches!(self, Self::Tensor(_))
    }

    fn downgrade(&self) -> WeakRetainedStateRuntimeV2 {
        match self {
            Self::Paged(runtime) => WeakRetainedStateRuntimeV2::Paged(Arc::downgrade(runtime)),
            Self::Tensor(runtime) => WeakRetainedStateRuntimeV2::Tensor(Arc::downgrade(runtime)),
        }
    }

    fn identity(&self) -> RetainedStateRuntimeIdentityV2 {
        match self {
            Self::Paged(runtime) => RetainedStateRuntimeIdentityV2::Paged {
                plan: runtime.plan().id,
            },
            Self::Tensor(runtime) => RetainedStateRuntimeIdentityV2::Tensor {
                runtime: runtime.id(),
            },
        }
    }
}

#[derive(Debug, Clone)]
enum WeakRetainedStateRuntimeV2 {
    Paged(Weak<ManagedKvModelRuntime>),
    Tensor(Weak<RetainedTensorStateRuntimeV2>),
}

impl WeakRetainedStateRuntimeV2 {
    fn upgrade(&self) -> Option<RetainedStateRuntimeV2> {
        match self {
            Self::Paged(runtime) => runtime.upgrade().map(RetainedStateRuntimeV2::Paged),
            Self::Tensor(runtime) => runtime.upgrade().map(RetainedStateRuntimeV2::Tensor),
        }
    }
}

fn retained_use_matches(
    retained_state_use: RetainedStateUseV2,
    runtime: &RetainedStateRuntimeV2,
) -> bool {
    match retained_state_use {
        RetainedStateUseV2::Inactive => true,
        RetainedStateUseV2::ExternalPaged => {
            matches!(runtime, RetainedStateRuntimeV2::Paged(_))
        }
        RetainedStateUseV2::ExternalTensor => {
            matches!(runtime, RetainedStateRuntimeV2::Tensor(_))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
enum RetainedStateRuntimeIdentityV2 {
    Paged {
        plan: crate::kv::KvPlanId,
    },
    Tensor {
        runtime: RetainedTensorStateRuntimeIdV2,
    },
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

/// One atomic row's canonically ordered set of exact invocation-domain leases.
/// Partially acquired sets release through each lease's Drop implementation.
#[derive(Debug)]
pub(crate) struct InvocationPagedLeaseSetV2 {
    stage: StageId,
    leases: Vec<(StateDomainId, InvocationPagedKvLease)>,
}

#[derive(Debug)]
pub(crate) struct InvocationPagedDomainCompletionV2 {
    pub(crate) stage: StageId,
    pub(crate) domain: StateDomainId,
    pub(crate) physical: InvocationPagedKvCompletion,
}

impl InvocationPagedLeaseSetV2 {
    pub(crate) const fn stage(&self) -> StageId {
        self.stage
    }

    pub(crate) fn domains(&self) -> impl ExactSizeIterator<Item = StateDomainId> + '_ {
        self.leases.iter().map(|(domain, _)| *domain)
    }

    pub(crate) fn cache(
        &self,
        domain: StateDomainId,
    ) -> Result<&crate::models::shared::attention::physical::PhysicalPagedKvCache> {
        self.leases
            .iter()
            .find(|(candidate, _)| *candidate == domain)
            .map(|(_, lease)| lease.cache())
            .ok_or_else(|| {
                invalid("atomic invocation lease set does not contain the requested domain")
            })
    }

    pub(crate) fn cache_mut(
        &mut self,
        domain: StateDomainId,
    ) -> Result<&mut crate::models::shared::attention::physical::PhysicalPagedKvCache> {
        self.leases
            .iter_mut()
            .find(|(candidate, _)| *candidate == domain)
            .map(|(_, lease)| lease.cache_mut())
            .ok_or_else(|| {
                invalid("atomic invocation lease set does not contain the requested domain")
            })
    }

    pub(crate) fn cache_pair_mut(
        &mut self,
        first: StateDomainId,
        second: StateDomainId,
    ) -> Result<(
        &mut crate::models::shared::attention::physical::PhysicalPagedKvCache,
        &mut crate::models::shared::attention::physical::PhysicalPagedKvCache,
    )> {
        if first == second {
            return Err(invalid(
                "invocation cache pair requires two distinct domains",
            ));
        }
        let first_index = self
            .leases
            .iter()
            .position(|(domain, _)| *domain == first)
            .ok_or_else(|| invalid("invocation lease set is missing its first cache domain"))?;
        let second_index = self
            .leases
            .iter()
            .position(|(domain, _)| *domain == second)
            .ok_or_else(|| invalid("invocation lease set is missing its second cache domain"))?;
        if first_index < second_index {
            let (left, right) = self.leases.split_at_mut(second_index);
            Ok((left[first_index].1.cache_mut(), right[0].1.cache_mut()))
        } else {
            let (left, right) = self.leases.split_at_mut(first_index);
            Ok((right[0].1.cache_mut(), left[second_index].1.cache_mut()))
        }
    }

    /// Release every domain even if one completion fails authentication. A
    /// failed set never exposes a partial collection of completions.
    pub(crate) fn release(mut self) -> Result<Vec<InvocationPagedDomainCompletionV2>> {
        let leases = std::mem::take(&mut self.leases);
        let mut completions = Vec::with_capacity(leases.len());
        let mut first_error = None;
        for (domain, lease) in leases {
            match lease.release() {
                Ok(physical) => completions.push(InvocationPagedDomainCompletionV2 {
                    stage: self.stage,
                    domain,
                    physical,
                }),
                Err(error) if first_error.is_none() => first_error = Some(error),
                Err(_) => {}
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(completions),
        }
    }
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

    fn lease_set(
        &self,
        graph: [u8; 32],
        stage: StageId,
        domains: &[StateDomainId],
    ) -> Result<InvocationPagedLeaseSetV2> {
        if domains.is_empty() {
            return Err(invalid(
                "atomic invocation lease set requires at least one state domain",
            ));
        }
        let mut canonical = domains.to_vec();
        canonical.sort_unstable();
        if canonical.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(invalid(
                "atomic invocation lease set repeats a state domain",
            ));
        }
        let mut leases = Vec::with_capacity(canonical.len());
        for domain in canonical {
            // If a later domain is unavailable, every lease already pushed
            // here is released by Drop before the error escapes.
            leases.push((domain, self.lease(graph, stage, domain)?));
        }
        Ok(InvocationPagedLeaseSetV2 { stage, leases })
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

/// Load-sealed physical invocation state for a capability with no retained
/// session state. This is the normal runtime for atomic ASR/TTS pipelines.
#[derive(Debug, Clone)]
pub(crate) struct InvocationCapabilityRuntimeV2 {
    pub(crate) id: [u8; 32],
    pub(crate) identity: CapabilityRuntimeIdentityV2,
    pub(crate) stage_graph_fingerprint: [u8; 32],
    pub(crate) state_fingerprint: [u8; 32],
    pub(crate) descriptor: CapabilityStateDescriptorV2,
    invocation_paged: InvocationPagedWorkspaceRuntimeV2,
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
    Invocation(InvocationCapabilityRuntimeV2),
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

    pub(crate) fn invocation(runtime: InvocationCapabilityRuntimeV2) -> Self {
        Self {
            id: runtime.id,
            state_fingerprint: runtime.state_fingerprint,
            descriptor: runtime.descriptor.clone(),
            backing: CapabilityStateRuntimeBackingV2::Invocation(runtime),
        }
    }

    pub(crate) fn managed_kv_runtime(&self) -> Option<Arc<ManagedKvModelRuntime>> {
        match &self.backing {
            CapabilityStateRuntimeBackingV2::Stateless(_) => None,
            CapabilityStateRuntimeBackingV2::Invocation(_) => None,
            CapabilityStateRuntimeBackingV2::Managed(runtime)
                if runtime.retained_state_use == RetainedStateUseV2::ExternalPaged =>
            {
                match runtime.retained.upgrade()? {
                    RetainedStateRuntimeV2::Paged(physical) => Some(physical),
                    RetainedStateRuntimeV2::Tensor(_) => None,
                }
            }
            CapabilityStateRuntimeBackingV2::Managed(_) => None,
        }
    }

    pub(crate) fn retained_tensor_state_runtime(
        &self,
    ) -> Option<Arc<RetainedTensorStateRuntimeV2>> {
        match &self.backing {
            CapabilityStateRuntimeBackingV2::Managed(runtime)
                if runtime.retained_state_use == RetainedStateUseV2::ExternalTensor =>
            {
                match runtime.retained.upgrade()? {
                    RetainedStateRuntimeV2::Tensor(physical) => Some(physical),
                    RetainedStateRuntimeV2::Paged(_) => None,
                }
            }
            CapabilityStateRuntimeBackingV2::Stateless(_)
            | CapabilityStateRuntimeBackingV2::Invocation(_)
            | CapabilityStateRuntimeBackingV2::Managed(_) => None,
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
            CapabilityStateRuntimeBackingV2::Invocation(runtime) => {
                runtime
                    .invocation_paged
                    .lease(runtime.stage_graph_fingerprint, stage, domain)
            }
            CapabilityStateRuntimeBackingV2::Managed(runtime) => {
                runtime
                    .invocation_paged
                    .lease(runtime.stage_graph_fingerprint, stage, domain)
            }
        }
    }

    pub(crate) fn lease_invocation_paged_set(
        &self,
        stage: StageId,
        domains: &[StateDomainId],
    ) -> Result<InvocationPagedLeaseSetV2> {
        match &self.backing {
            CapabilityStateRuntimeBackingV2::Stateless(_) => Err(invalid(
                "stateless runtime has no load-sealed paged invocation workspace",
            )),
            CapabilityStateRuntimeBackingV2::Invocation(runtime) => runtime
                .invocation_paged
                .lease_set(runtime.stage_graph_fingerprint, stage, domains),
            CapabilityStateRuntimeBackingV2::Managed(runtime) => runtime
                .invocation_paged
                .lease_set(runtime.stage_graph_fingerprint, stage, domains),
        }
    }

    /// Lease every paged domain authored for one exact stage. Domain selection
    /// is sealed into the runtime descriptor so direct and engine runners
    /// cannot accidentally execute with a partial cache set.
    pub(crate) fn lease_complete_invocation_paged_set(
        &self,
        stage: StageId,
    ) -> Result<InvocationPagedLeaseSetV2> {
        let graph = match &self.backing {
            CapabilityStateRuntimeBackingV2::Stateless(_) => {
                return Err(invalid(
                    "stateless runtime has no load-sealed paged invocation workspace",
                ));
            }
            CapabilityStateRuntimeBackingV2::Invocation(runtime) => runtime.stage_graph_fingerprint,
            CapabilityStateRuntimeBackingV2::Managed(runtime) => runtime.stage_graph_fingerprint,
        };
        let InvocationWorkspaceSet::Bounded { profiles } = &self.descriptor.invocation else {
            return Err(invalid(
                "physical invocation runtime has no bounded workspace profile",
            ));
        };
        let workspace = profiles
            .iter()
            .find(|profile| profile.stage_graph_fingerprint == graph)
            .and_then(|profile| {
                profile
                    .stages
                    .iter()
                    .find(|workspace| workspace.stage == stage)
            })
            .ok_or_else(|| invalid("physical invocation runtime has no exact stage workspace"))?;
        let domains = workspace
            .domains
            .iter()
            .filter_map(|domain| match domain {
                InvocationWorkspaceDomain::State {
                    state: super::StateDomainSpec::PagedAttention(state),
                    capacity: super::InvocationStateCapacity::PagedTokens { .. },
                    ..
                } => Some(state.header.id),
                _ => None,
            })
            .collect::<Vec<_>>();
        self.lease_invocation_paged_set(stage, &domains)
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
            CapabilityStateRuntimeBackingV2::Invocation(runtime) => {
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
/// plan and the already allocated physical retained arena implementing it.
#[derive(Debug, Clone)]
pub(crate) struct ManagedCapabilityRuntimeV2 {
    pub(crate) id: [u8; 32],
    pub(crate) identity: CapabilityRuntimeIdentityV2,
    pub(crate) stage_graph_fingerprint: [u8; 32],
    pub(crate) state_fingerprint: [u8; 32],
    pub(crate) descriptor: CapabilityStateDescriptorV2,
    pub(crate) state_plan: Arc<ResolvedStatePlan>,
    retained_identity: RetainedStateRuntimeIdentityV2,
    retained_state_use: RetainedStateUseV2,
    /// The lifecycle manager is the physical owner. A sealed adapter proves
    /// the exact generation without pinning that generation through unload;
    /// admitted requests upgrade this weak handle while holding residency.
    retained: WeakRetainedStateRuntimeV2,
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
        physical: impl Into<RetainedStateRuntimeV2>,
        retained_state_use: RetainedStateUseV2,
        invocation_paged: InvocationPagedWorkspaceRuntimeV2,
    ) -> Result<Self> {
        execution.validate()?;
        descriptor.validate_against_stages(&execution.stages)?;
        let physical = physical.into();
        let RetainedStateCapability::Managed { contract } = &descriptor.retained else {
            return Err(invalid(
                "managed state ABI v2 runtime requires retained physical state",
            ));
        };
        if execution.model_instance_id != physical.model_instance() {
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
            retained_identity: physical.identity(),
            retained_state_use,
            retained: physical.downgrade(),
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
        let physical = self.retained.upgrade().ok_or_else(|| {
            invalid("managed state ABI v2 runtime refers to an unloaded physical generation")
        })?;
        if self.stage_graph_fingerprint != stage_graph_fingerprint(&execution.stages)?
            || self.state_fingerprint != self.descriptor.fingerprint(&execution.stages)?
            || self.state_plan.id != physical.state_plan_v2().id
            || self.retained_identity != physical.identity()
            || execution.model_instance_id != physical.model_instance()
            || !retained_use_matches(self.retained_state_use, &physical)
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
            retained_identity: RetainedStateRuntimeIdentityV2,
            retained_state_use: RetainedStateUseV2,
            invocation_paged: Vec<(InvocationPagedWorkspaceKeyV2, InvocationPagedKvPoolId)>,
        }
        let encoded = serde_json::to_vec(&Payload {
            identity: &self.identity,
            stage_graph_fingerprint: self.stage_graph_fingerprint,
            state_fingerprint: self.state_fingerprint,
            state_plan: self.state_plan.id,
            retained_identity: self.retained_identity,
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

impl InvocationCapabilityRuntimeV2 {
    pub(crate) fn seal(
        backend: BackendKind,
        execution: &ExecutionAdapterBinding,
        descriptor: CapabilityStateDescriptorV2,
        invocation_paged: InvocationPagedWorkspaceRuntimeV2,
    ) -> Result<Self> {
        execution.validate()?;
        descriptor.validate_against_stages(&execution.stages)?;
        if !descriptor.is_stateless()
            || descriptor.has_zero_invocation_workspace_for(&execution.stages)?
        {
            return Err(invalid(
                "invocation state ABI v2 runtime requires physical invocation state and no retained state",
            ));
        }
        invocation_paged.validate_for(&descriptor, &execution.stages)?;
        let stage_graph_fingerprint = stage_graph_fingerprint(&execution.stages)?;
        let state_fingerprint = descriptor.fingerprint(&execution.stages)?;
        let mut runtime = Self {
            id: [0; 32],
            identity: CapabilityRuntimeIdentityV2::seal(backend, execution)?,
            stage_graph_fingerprint,
            state_fingerprint,
            descriptor,
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
        if !self.descriptor.is_stateless()
            || self
                .descriptor
                .has_zero_invocation_workspace_for(&execution.stages)?
            || self.stage_graph_fingerprint != stage_graph_fingerprint(&execution.stages)?
            || self.state_fingerprint != self.descriptor.fingerprint(&execution.stages)?
            || self.id != self.compute_id()?
        {
            return Err(invalid(
                "invocation state ABI v2 runtime does not match the selected loaded capability",
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
            invocation_paged: Vec<(InvocationPagedWorkspaceKeyV2, InvocationPagedKvPoolId)>,
        }
        let encoded = serde_json::to_vec(&Payload {
            identity: &self.identity,
            stage_graph_fingerprint: self.stage_graph_fingerprint,
            state_fingerprint: self.state_fingerprint,
            invocation_paged: self
                .invocation_paged
                .pool_ids_for_graph(self.stage_graph_fingerprint),
        })
        .map_err(|error| invalid(format!("failed to encode invocation runtime: {error}")))?;
        let mut hasher = Sha256::new();
        hasher.update(INVOCATION_RUNTIME_FINGERPRINT_DOMAIN);
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
        if !descriptor.is_stateless()
            || !descriptor.has_zero_invocation_workspace_for(&execution.stages)?
        {
            return Err(invalid(
                "stateless state ABI v2 runtime cannot seal retained or invocation physical state",
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
            || !self
                .descriptor
                .has_zero_invocation_workspace_for(&execution.stages)?
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
        NativeBatchMode, PhysicalStateManager, StageDescriptor, StageId,
    };
    use crate::kv::v2::{test_contract, upgrade_kv_contract_v1};
    use crate::kv::v2::{
        BoundedShape, CheckpointPolicy, InferenceStateContract, InvocationStageWorkspace,
        InvocationStateCapacity, InvocationWorkspaceProfile, PlacementPolicy, PrefixPolicy,
        ShapeAxis, ShapeDimension, ShapeExtent, StateClock, StateComponentId, StateDType,
        StateDomainHeader, StateDomainSpec, StateGroupId, StateGroupSpec, StateScope,
        TensorComponentSpec, TensorRole, TensorStateDomainSpec, WorkspaceFormula,
        CURRENT_INFERENCE_STATE_ABI,
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

    fn tensor_contract() -> InferenceStateContract {
        InferenceStateContract {
            abi: CURRENT_INFERENCE_STATE_ABI,
            domains: vec![StateDomainSpec::Tensor(TensorStateDomainSpec {
                header: StateDomainHeader {
                    id: StateDomainId::new(1),
                    scope: StateScope::Retained,
                    clock: StateClock::DecoderTokens,
                    placement: PlacementPolicy::BackendLocal,
                    prefix: PrefixPolicy::Disabled,
                    checkpoint: CheckpointPolicy::Transactional,
                },
                components: vec![TensorComponentSpec {
                    id: StateComponentId::new(1),
                    role: TensorRole::RecurrentHidden,
                    shape: BoundedShape {
                        dimensions: vec![ShapeDimension {
                            axis: ShapeAxis::Hidden,
                            extent: ShapeExtent::RuntimeBounded { min: 1, max: 8 },
                        }],
                    },
                    accepted_dtypes: vec![StateDType::F32],
                }],
            })],
            groups: vec![StateGroupSpec {
                id: StateGroupId::new(1),
                domains: vec![StateDomainId::new(1)],
                prefix_shareable: false,
            }],
        }
    }

    fn invocation_pool(
        model_instance: ModelInstanceId,
    ) -> (InvocationPagedKvPoolOwner, InvocationWorkspaceDomain) {
        invocation_pool_for_domain(model_instance, StateDomainId::new(1), 19)
    }

    fn invocation_pool_for_domain(
        model_instance: ModelInstanceId,
        domain_id: StateDomainId,
        arena_generation: u32,
    ) -> (InvocationPagedKvPoolOwner, InvocationWorkspaceDomain) {
        let mut contract = test_contract();
        let StateDomainSpec::PagedAttention(domain) = &mut contract.domains[0] else {
            unreachable!()
        };
        domain.header.id = domain_id;
        domain.header.scope = StateScope::Invocation;
        domain.header.prefix = PrefixPolicy::Disabled;
        domain.header.checkpoint = CheckpointPolicy::None;
        domain.accepted_dtypes = vec![StateDType::F32];
        domain.layers[0].query_heads = 2;
        domain.layers[0].kv_heads = 2;
        domain.layers[0].key_head_dim = 4;
        domain.layers[0].value_head_dim = 4;
        domain.layers[0].key_encoding = super::super::KeyEncoding::Rotary { rotary_dim: 4 };
        contract.groups[0].domains = vec![domain_id];
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
                    generation: arena_generation,
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
                        lease_scope: super::super::InvocationLeaseScope::PerStageBatch,
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
    fn tensor_only_retained_runtime_seals_and_invalidates_on_unload() {
        let binding = binding();
        let contract = tensor_contract();
        let descriptor = CapabilityStateDescriptorV2::managed_for_stage_graphs(
            contract.clone(),
            &[binding.stages.as_ref()],
        )
        .unwrap();
        let mut physical = PhysicalStateManager::cpu(None);
        let retained = physical
            .allocate_retained_tensor(binding.model_instance_id, &contract, 1)
            .unwrap();

        assert!(ManagedCapabilityRuntimeV2::seal_with_invocation_paged(
            BackendKind::Cpu,
            &binding,
            descriptor.clone(),
            retained.clone(),
            RetainedStateUseV2::ExternalPaged,
            InvocationPagedWorkspaceRuntimeV2::default(),
        )
        .is_err());

        let runtime = CapabilityStateRuntimeV2::managed(
            ManagedCapabilityRuntimeV2::seal_with_invocation_paged(
                BackendKind::Cpu,
                &binding,
                descriptor,
                retained.clone(),
                RetainedStateUseV2::ExternalTensor,
                InvocationPagedWorkspaceRuntimeV2::default(),
            )
            .unwrap(),
        );
        assert!(runtime.managed_kv_runtime().is_none());
        assert_eq!(
            runtime
                .retained_tensor_state_runtime()
                .expect("tensor-only retained backing")
                .id(),
            retained.id()
        );
        runtime
            .validate_against(BackendKind::Cpu, &binding)
            .unwrap();

        drop(retained);
        assert!(physical.unload_model(binding.model_instance_id).unwrap());
        assert!(runtime.retained_tensor_state_runtime().is_none());
        assert!(runtime
            .validate_against(BackendKind::Cpu, &binding)
            .is_err());
    }

    #[test]
    fn invocation_only_runtime_owns_bounded_physical_workspace() {
        let (binding, mut descriptor, _, pool) = managed_invocation_fixture();
        descriptor.retained = RetainedStateCapability::Stateless;
        assert!(!descriptor
            .has_zero_invocation_workspace_for(&binding.stages)
            .unwrap());
        assert!(
            StatelessCapabilityRuntimeV2::seal(BackendKind::Cpu, &binding, descriptor.clone())
                .is_err()
        );
        let stage = binding.stages[0].id;
        let domain = StateDomainId::new(1);
        let pools =
            InvocationPagedWorkspaceRuntimeV2::new(vec![InvocationPagedWorkspaceBindingV2 {
                key: InvocationPagedWorkspaceKeyV2 {
                    stage_graph: stage_graph_fingerprint(&binding.stages).unwrap(),
                    stage,
                    domain,
                },
                pool: pool.handle(),
            }])
            .unwrap();
        let runtime =
            InvocationCapabilityRuntimeV2::seal(BackendKind::Cpu, &binding, descriptor, pools)
                .unwrap();
        runtime
            .validate_against(BackendKind::Cpu, &binding)
            .unwrap();
        let runtime = CapabilityStateRuntimeV2::invocation(runtime);
        runtime
            .lease_invocation_paged(stage, domain)
            .unwrap()
            .release()
            .unwrap();
        let leases = runtime
            .lease_complete_invocation_paged_set(stage)
            .expect("complete descriptor-authored lease set");
        assert_eq!(leases.domains().collect::<Vec<_>>(), vec![domain]);
        leases.release().unwrap();
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

    #[test]
    fn invocation_lease_set_is_canonical_and_returns_every_completion() {
        let graph = [7; 32];
        let stage = StageId::new(4);
        let model = ModelInstanceId::new(51);
        let (first, _) = invocation_pool_for_domain(model, StateDomainId::new(1), 31);
        let (second, _) = invocation_pool_for_domain(model, StateDomainId::new(2), 32);
        let runtime = InvocationPagedWorkspaceRuntimeV2::new(vec![
            InvocationPagedWorkspaceBindingV2 {
                key: InvocationPagedWorkspaceKeyV2 {
                    stage_graph: graph,
                    stage,
                    domain: StateDomainId::new(1),
                },
                pool: first.handle(),
            },
            InvocationPagedWorkspaceBindingV2 {
                key: InvocationPagedWorkspaceKeyV2 {
                    stage_graph: graph,
                    stage,
                    domain: StateDomainId::new(2),
                },
                pool: second.handle(),
            },
        ])
        .unwrap();

        let mut leases = runtime
            .lease_set(
                graph,
                stage,
                &[StateDomainId::new(2), StateDomainId::new(1)],
            )
            .unwrap();
        assert_eq!(leases.stage(), stage);
        assert_eq!(
            leases.domains().collect::<Vec<_>>(),
            vec![StateDomainId::new(1), StateDomainId::new(2)]
        );
        assert_eq!(
            leases.cache(StateDomainId::new(1)).unwrap().context_len(),
            0
        );
        assert_eq!(
            leases
                .cache_mut(StateDomainId::new(2))
                .unwrap()
                .context_len(),
            0
        );

        let completions = leases.release().unwrap();
        assert_eq!(completions.len(), 2);
        assert!(completions.iter().all(|completion| {
            completion.stage == stage
                && completion.domain == completion.physical.slot.pool.domain
                && completion.physical.writes.is_empty()
        }));
    }

    #[test]
    fn invocation_lease_set_rolls_back_partial_acquisition_on_error() {
        let graph = [9; 32];
        let stage = StageId::new(5);
        let model = ModelInstanceId::new(52);
        let (owner, _) = invocation_pool_for_domain(model, StateDomainId::new(1), 33);
        let runtime =
            InvocationPagedWorkspaceRuntimeV2::new(vec![InvocationPagedWorkspaceBindingV2 {
                key: InvocationPagedWorkspaceKeyV2 {
                    stage_graph: graph,
                    stage,
                    domain: StateDomainId::new(1),
                },
                pool: owner.handle(),
            }])
            .unwrap();

        assert!(runtime
            .lease_set(
                graph,
                stage,
                &[StateDomainId::new(1), StateDomainId::new(99)],
            )
            .is_err());
        runtime
            .lease_set(graph, stage, &[StateDomainId::new(1)])
            .expect("the first lease must be released after later acquisition fails")
            .release()
            .unwrap();
        assert!(runtime
            .lease_set(
                graph,
                stage,
                &[StateDomainId::new(1), StateDomainId::new(1)],
            )
            .is_err());
    }
}
