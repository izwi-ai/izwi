use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::any::Any;
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

/// Model-neutral identity for one exact invocation workspace publication.
///
/// The historical paged name remains an alias below so existing adapters can
/// migrate independently without creating a second runtime contract.
pub(crate) type InvocationWorkspaceKeyV2 = InvocationPagedWorkspaceKeyV2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub(crate) enum InvocationStateBackingKindV2 {
    PagedAttention,
    StaticAttention,
    Tensor,
    Append,
    Ring,
    StaticTensor,
}

impl InvocationStateBackingKindV2 {
    fn for_workspace(domain: &InvocationWorkspaceDomain) -> Option<Self> {
        let InvocationWorkspaceDomain::State { state, .. } = domain else {
            return None;
        };
        Some(match state {
            super::StateDomainSpec::PagedAttention(_) => Self::PagedAttention,
            super::StateDomainSpec::StaticAttention(_) => Self::StaticAttention,
            super::StateDomainSpec::Tensor(_) => Self::Tensor,
            super::StateDomainSpec::Append(_) => Self::Append,
            super::StateDomainSpec::Ring(_) => Self::Ring,
            super::StateDomainSpec::StaticTensor(_) => Self::StaticTensor,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub(crate) enum InvocationWorkspaceBackingIdentityV2 {
    Paged(InvocationPagedKvPoolId),
    Typed {
        kind: InvocationStateBackingKindV2,
        model_instance: ModelInstanceId,
        backend: BackendKind,
        domain: StateDomainId,
        allocation_generation: u32,
    },
}

impl InvocationWorkspaceBackingIdentityV2 {
    const fn kind(self) -> InvocationStateBackingKindV2 {
        match self {
            Self::Paged(_) => InvocationStateBackingKindV2::PagedAttention,
            Self::Typed { kind, .. } => kind,
        }
    }

    fn validates_kind(self, authored: InvocationStateBackingKindV2) -> bool {
        match (self, authored) {
            (Self::Paged(_), InvocationStateBackingKindV2::PagedAttention) => true,
            (
                Self::Typed {
                    kind,
                    model_instance,
                    domain,
                    allocation_generation,
                    ..
                },
                InvocationStateBackingKindV2::StaticAttention
                | InvocationStateBackingKindV2::Tensor
                | InvocationStateBackingKindV2::Append
                | InvocationStateBackingKindV2::Ring
                | InvocationStateBackingKindV2::StaticTensor,
            ) => {
                kind == authored
                    && model_instance.get() != 0
                    && domain.get() != 0
                    && allocation_generation != 0
            }
            _ => false,
        }
    }

    fn validates_owner(
        self,
        key: InvocationWorkspaceKeyV2,
        backend: BackendKind,
        model_instance: ModelInstanceId,
    ) -> bool {
        match self {
            Self::Paged(pool) => {
                pool.domain == key.domain
                    && pool.arena.model_instance == model_instance
                    && pool.arena.backend == backend
            }
            Self::Typed {
                model_instance: owner,
                backend: owner_backend,
                domain,
                ..
            } => owner == model_instance && owner_backend == backend && domain == key.domain,
        }
    }
}

#[derive(Debug)]
pub(crate) enum InvocationWorkspacePhysicalCompletionV2 {
    Paged(InvocationPagedKvCompletion),
    Typed {
        backing: InvocationWorkspaceBackingIdentityV2,
        /// Backing-owned authenticated completion receipt. The runtime binds
        /// it to the exact published backing identity before exposing it.
        authentication: [u8; 32],
    },
}

impl InvocationWorkspacePhysicalCompletionV2 {
    fn backing_identity(&self) -> InvocationWorkspaceBackingIdentityV2 {
        match self {
            Self::Paged(completion) => {
                InvocationWorkspaceBackingIdentityV2::Paged(completion.slot.pool)
            }
            Self::Typed { backing, .. } => *backing,
        }
    }
}

/// One physical invocation lease. Implementations own domain-specific
/// transaction authentication; this runtime owns graph/stage/domain binding
/// and completion authentication.
pub(crate) trait InvocationWorkspacePhysicalLeaseV2: std::fmt::Debug + Send {
    fn backing_identity(&self) -> InvocationWorkspaceBackingIdentityV2;

    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn into_any(self: Box<Self>) -> Box<dyn Any>;

    fn paged_cache(
        &self,
    ) -> Option<&crate::models::shared::attention::physical::PhysicalPagedKvCache> {
        None
    }

    fn paged_cache_mut(
        &mut self,
    ) -> Option<&mut crate::models::shared::attention::physical::PhysicalPagedKvCache> {
        None
    }

    /// Complete the physical operation and return its authenticated receipt.
    /// Implementations must remain abortable when this returns an error.
    fn complete(&mut self) -> Result<InvocationWorkspacePhysicalCompletionV2>;

    /// Idempotently abandon the physical operation and return its slot/state
    /// to the load-owned pool. Runtime-owned guards invoke this on every
    /// partial acquisition, operation error, and unwind path.
    fn abort(&mut self);
}

/// A load-owned physical backing published for one typed invocation domain.
/// Scratch workspaces intentionally have no implementation of this contract:
/// their formula is accounted by the stage workspace allocation itself.
pub(crate) trait InvocationWorkspaceBackingV2: std::fmt::Debug + Send + Sync {
    fn identity(&self) -> InvocationWorkspaceBackingIdentityV2;

    fn workspace_domain(&self) -> &InvocationWorkspaceDomain;

    fn validate_live(&self) -> Result<()>;

    fn lease(&self) -> Result<Box<dyn InvocationWorkspacePhysicalLeaseV2>>;

    fn authenticate_completion(
        &self,
        completion: &InvocationWorkspacePhysicalCompletionV2,
    ) -> Result<()>;
}

#[derive(Debug, Clone)]
pub(crate) struct InvocationWorkspaceBindingV2 {
    pub(crate) key: InvocationWorkspaceKeyV2,
    pub(crate) backing: Arc<dyn InvocationWorkspaceBackingV2>,
}

/// Keep the concrete paged wrapper private to the runtime while allowing the
/// physical manager to publish the generic workspace contract.
pub(crate) fn invocation_paged_workspace_backing_v2(
    pool: InvocationPagedKvPoolHandle,
) -> Arc<dyn InvocationWorkspaceBackingV2> {
    Arc::new(InvocationPagedWorkspaceBackingV2 { pool })
}

#[derive(Debug, Clone)]
struct InvocationPagedWorkspaceBackingV2 {
    pool: InvocationPagedKvPoolHandle,
}

impl InvocationWorkspaceBackingV2 for InvocationPagedWorkspaceBackingV2 {
    fn identity(&self) -> InvocationWorkspaceBackingIdentityV2 {
        InvocationWorkspaceBackingIdentityV2::Paged(self.pool.id())
    }

    fn workspace_domain(&self) -> &InvocationWorkspaceDomain {
        self.pool.workspace_domain()
    }

    fn validate_live(&self) -> Result<()> {
        self.pool.validate_live()
    }

    fn lease(&self) -> Result<Box<dyn InvocationWorkspacePhysicalLeaseV2>> {
        Ok(Box::new(InvocationPagedWorkspacePhysicalLeaseV2 {
            lease: Some(self.pool.lease()?),
        }))
    }

    fn authenticate_completion(
        &self,
        completion: &InvocationWorkspacePhysicalCompletionV2,
    ) -> Result<()> {
        match completion {
            InvocationWorkspacePhysicalCompletionV2::Paged(completion)
                if completion.slot.pool == self.pool.id() =>
            {
                Ok(())
            }
            _ => Err(invalid(
                "paged invocation completion does not authenticate its pool",
            )),
        }
    }
}

#[derive(Debug)]
struct InvocationPagedWorkspacePhysicalLeaseV2 {
    lease: Option<InvocationPagedKvLease>,
}

impl InvocationWorkspacePhysicalLeaseV2 for InvocationPagedWorkspacePhysicalLeaseV2 {
    fn backing_identity(&self) -> InvocationWorkspaceBackingIdentityV2 {
        InvocationWorkspaceBackingIdentityV2::Paged(
            self.lease
                .as_ref()
                .expect("active paged invocation lease")
                .slot()
                .pool,
        )
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }

    fn paged_cache(
        &self,
    ) -> Option<&crate::models::shared::attention::physical::PhysicalPagedKvCache> {
        self.lease.as_ref().map(InvocationPagedKvLease::cache)
    }

    fn paged_cache_mut(
        &mut self,
    ) -> Option<&mut crate::models::shared::attention::physical::PhysicalPagedKvCache> {
        self.lease.as_mut().map(InvocationPagedKvLease::cache_mut)
    }

    fn complete(&mut self) -> Result<InvocationWorkspacePhysicalCompletionV2> {
        let lease = self
            .lease
            .take()
            .ok_or_else(|| invalid("paged invocation lease is no longer active"))?;
        Ok(InvocationWorkspacePhysicalCompletionV2::Paged(
            lease.release()?,
        ))
    }

    fn abort(&mut self) {
        drop(self.lease.take());
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct InvocationWorkspaceRuntimeV2 {
    backings: HashMap<InvocationWorkspaceKeyV2, Arc<dyn InvocationWorkspaceBackingV2>>,
}

#[derive(Debug)]
pub(crate) struct InvocationWorkspaceLeaseV2 {
    key: InvocationWorkspaceKeyV2,
    backing_identity: InvocationWorkspaceBackingIdentityV2,
    backing: Arc<dyn InvocationWorkspaceBackingV2>,
    physical: Option<Box<dyn InvocationWorkspacePhysicalLeaseV2>>,
}

impl InvocationWorkspaceLeaseV2 {
    pub(crate) const fn key(&self) -> InvocationWorkspaceKeyV2 {
        self.key
    }

    pub(crate) const fn kind(&self) -> InvocationStateBackingKindV2 {
        self.backing_identity.kind()
    }

    pub(crate) fn paged_cache(
        &self,
    ) -> Result<&crate::models::shared::attention::physical::PhysicalPagedKvCache> {
        self.physical
            .as_ref()
            .expect("active invocation workspace lease")
            .paged_cache()
            .ok_or_else(|| invalid("invocation workspace lease is not paged attention"))
    }

    pub(crate) fn paged_cache_mut(
        &mut self,
    ) -> Result<&mut crate::models::shared::attention::physical::PhysicalPagedKvCache> {
        self.physical
            .as_mut()
            .expect("active invocation workspace lease")
            .paged_cache_mut()
            .ok_or_else(|| invalid("invocation workspace lease is not paged attention"))
    }

    pub(crate) fn typed<T: Any>(&self) -> Result<&T> {
        self.physical
            .as_ref()
            .expect("active invocation workspace lease")
            .as_any()
            .downcast_ref()
            .ok_or_else(|| invalid("invocation workspace lease has a different physical type"))
    }

    pub(crate) fn typed_mut<T: Any>(&mut self) -> Result<&mut T> {
        self.physical
            .as_mut()
            .expect("active invocation workspace lease")
            .as_any_mut()
            .downcast_mut()
            .ok_or_else(|| invalid("invocation workspace lease has a different physical type"))
    }

    pub(crate) fn release(mut self) -> Result<InvocationWorkspaceDomainCompletionV2> {
        let lease = self
            .physical
            .as_mut()
            .ok_or_else(|| invalid("invocation workspace lease is no longer active"))?;
        let physical = match lease.complete() {
            Ok(completion) => completion,
            Err(error) => {
                lease.abort();
                self.physical.take();
                return Err(error);
            }
        };
        if physical.backing_identity() != self.backing_identity {
            lease.abort();
            self.physical.take();
            return Err(invalid(
                "invocation workspace completion does not authenticate its published backing",
            ));
        }
        if let Err(error) = self.backing.authenticate_completion(&physical) {
            lease.abort();
            self.physical.take();
            return Err(error);
        }
        self.physical.take();
        Ok(InvocationWorkspaceDomainCompletionV2 {
            key: self.key,
            physical,
        })
    }

    fn into_paged(mut self) -> Result<InvocationPagedKvLease> {
        if self.backing_identity.kind() != InvocationStateBackingKindV2::PagedAttention {
            return Err(invalid("invocation workspace lease is not paged attention"));
        }
        let active = self
            .physical
            .as_ref()
            .expect("active invocation workspace lease")
            .as_any()
            .downcast_ref::<InvocationPagedWorkspacePhysicalLeaseV2>()
            .ok_or_else(|| invalid("paged invocation workspace has a mismatched lease type"))?;
        if active.lease.is_none() {
            return Err(invalid(
                "paged invocation workspace lease is no longer active",
            ));
        }
        let physical = self
            .physical
            .take()
            .expect("active invocation workspace lease")
            .into_any()
            .downcast::<InvocationPagedWorkspacePhysicalLeaseV2>()
            .expect("paged invocation lease type was checked before disarming rollback");
        Ok(physical
            .lease
            .expect("paged invocation lease activity was checked before disarming rollback"))
    }
}

impl Drop for InvocationWorkspaceLeaseV2 {
    fn drop(&mut self) {
        if let Some(mut physical) = self.physical.take() {
            physical.abort();
        }
    }
}

#[derive(Debug)]
pub(crate) struct InvocationWorkspaceLeaseSetV2 {
    stage: StageId,
    leases: Vec<(StateDomainId, InvocationWorkspaceLeaseV2)>,
}

#[derive(Debug)]
pub(crate) struct InvocationWorkspaceDomainCompletionV2 {
    pub(crate) key: InvocationWorkspaceKeyV2,
    pub(crate) physical: InvocationWorkspacePhysicalCompletionV2,
}

impl InvocationWorkspaceLeaseSetV2 {
    pub(crate) const fn stage(&self) -> StageId {
        self.stage
    }

    pub(crate) fn domains(&self) -> impl ExactSizeIterator<Item = StateDomainId> + '_ {
        self.leases.iter().map(|(domain, _)| *domain)
    }

    pub(crate) fn lease(&self, domain: StateDomainId) -> Result<&InvocationWorkspaceLeaseV2> {
        self.leases
            .iter()
            .find(|(candidate, _)| *candidate == domain)
            .map(|(_, lease)| lease)
            .ok_or_else(|| invalid("atomic invocation lease set does not contain the domain"))
    }

    pub(crate) fn lease_mut(
        &mut self,
        domain: StateDomainId,
    ) -> Result<&mut InvocationWorkspaceLeaseV2> {
        self.leases
            .iter_mut()
            .find(|(candidate, _)| *candidate == domain)
            .map(|(_, lease)| lease)
            .ok_or_else(|| invalid("atomic invocation lease set does not contain the domain"))
    }

    /// Borrow the only state domain in a complete scalar workspace and verify
    /// its physical kind. Model adapters cannot accidentally ignore a newly
    /// authored companion domain.
    pub(crate) fn lease_exact_kind_mut(
        &mut self,
        kind: InvocationStateBackingKindV2,
    ) -> Result<&mut InvocationWorkspaceLeaseV2> {
        let [(domain, _)] = self.leases.as_slice() else {
            return Err(invalid(
                "atomic invocation lease kind requires exactly one state domain",
            ));
        };
        let domain = *domain;
        let lease = self.lease_mut(domain)?;
        if lease.kind() != kind {
            return Err(invalid(
                "atomic invocation lease has a different physical backing kind",
            ));
        }
        Ok(lease)
    }

    pub(crate) fn lease_pair_mut(
        &mut self,
        first: StateDomainId,
        second: StateDomainId,
    ) -> Result<(
        &mut InvocationWorkspaceLeaseV2,
        &mut InvocationWorkspaceLeaseV2,
    )> {
        if first == second {
            return Err(invalid(
                "atomic invocation lease pair requires distinct domains",
            ));
        }
        let first_index = self
            .leases
            .iter()
            .position(|(domain, _)| *domain == first)
            .ok_or_else(|| {
                invalid("atomic invocation lease set does not contain the first domain")
            })?;
        let second_index = self
            .leases
            .iter()
            .position(|(domain, _)| *domain == second)
            .ok_or_else(|| {
                invalid("atomic invocation lease set does not contain the second domain")
            })?;
        if first_index < second_index {
            let (before_second, from_second) = self.leases.split_at_mut(second_index);
            return Ok((&mut before_second[first_index].1, &mut from_second[0].1));
        }
        let (before_first, from_first) = self.leases.split_at_mut(first_index);
        Ok((&mut from_first[0].1, &mut before_first[second_index].1))
    }

    pub(crate) fn lease_triplet_mut(
        &mut self,
        first: StateDomainId,
        second: StateDomainId,
        third: StateDomainId,
    ) -> Result<(
        &mut InvocationWorkspaceLeaseV2,
        &mut InvocationWorkspaceLeaseV2,
        &mut InvocationWorkspaceLeaseV2,
    )> {
        if first == second || first == third || second == third {
            return Err(invalid(
                "atomic invocation lease triplet requires distinct domains",
            ));
        }
        let index_for = |domain| {
            self.leases
                .iter()
                .position(|(candidate, _)| *candidate == domain)
                .ok_or_else(|| {
                    invalid("atomic invocation lease set does not contain a requested domain")
                })
        };
        let mut ordered = [
            (index_for(first)?, 0_u8),
            (index_for(second)?, 1_u8),
            (index_for(third)?, 2_u8),
        ];
        ordered.sort_unstable_by_key(|(index, _)| *index);
        let [(low_index, low_order), (middle_index, middle_order), (high_index, high_order)] =
            ordered;
        let (before_middle, from_middle) = self.leases.split_at_mut(middle_index);
        let low = &mut before_middle[low_index].1;
        let (middle_entry, after_middle) = from_middle.split_at_mut(1);
        let middle = &mut middle_entry[0].1;
        let high = &mut after_middle[high_index - middle_index - 1].1;
        match (low_order, middle_order, high_order) {
            (0, 1, 2) => Ok((low, middle, high)),
            (0, 2, 1) => Ok((low, high, middle)),
            (1, 0, 2) => Ok((middle, low, high)),
            (1, 2, 0) => Ok((high, low, middle)),
            (2, 0, 1) => Ok((middle, high, low)),
            (2, 1, 0) => Ok((high, middle, low)),
            _ => unreachable!("three distinct domain indices have one total ordering"),
        }
    }

    /// Borrow an exact two-domain mixed workspace by physical backing kind.
    /// This keeps model adapters independent of authored domain ordering while
    /// rejecting missing, duplicate, or silently extra state.
    pub(crate) fn lease_exact_kind_pair_mut(
        &mut self,
        first: InvocationStateBackingKindV2,
        second: InvocationStateBackingKindV2,
    ) -> Result<(
        &mut InvocationWorkspaceLeaseV2,
        &mut InvocationWorkspaceLeaseV2,
    )> {
        if first == second {
            return Err(invalid(
                "atomic invocation lease kind pair requires distinct backing kinds",
            ));
        }
        if self.leases.len() != 2 {
            return Err(invalid(
                "atomic invocation lease kind pair requires exactly two state domains",
            ));
        }
        let domain_for = |kind| {
            self.leases
                .iter()
                .find(|(_, lease)| lease.kind() == kind)
                .map(|(domain, _)| *domain)
                .ok_or_else(|| {
                    invalid("atomic invocation lease set is missing a requested backing kind")
                })
        };
        let first_domain = domain_for(first)?;
        let second_domain = domain_for(second)?;
        self.lease_pair_mut(first_domain, second_domain)
    }

    /// Release every domain even if one completion fails authentication. A
    /// failed set never exposes a partial collection of completions.
    pub(crate) fn release(mut self) -> Result<Vec<InvocationWorkspaceDomainCompletionV2>> {
        let leases = std::mem::take(&mut self.leases);
        let mut completions = Vec::with_capacity(leases.len());
        let mut first_error = None;
        for (_, lease) in leases {
            match lease.release() {
                Ok(completion) => completions.push(completion),
                Err(error) if first_error.is_none() => first_error = Some(error),
                Err(_) => {}
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(completions),
        }
    }

    fn into_paged(self) -> Result<InvocationPagedLeaseSetV2> {
        let mut leases = Vec::with_capacity(self.leases.len());
        for (domain, lease) in self.leases {
            leases.push((domain, lease.into_paged()?));
        }
        Ok(InvocationPagedLeaseSetV2 {
            stage: self.stage,
            leases,
        })
    }
}

impl InvocationWorkspaceRuntimeV2 {
    pub(crate) fn new(bindings: Vec<InvocationWorkspaceBindingV2>) -> Result<Self> {
        let mut backings = HashMap::with_capacity(bindings.len());
        for binding in bindings {
            let authored_kind =
                InvocationStateBackingKindV2::for_workspace(binding.backing.workspace_domain())
                    .ok_or_else(|| {
                        invalid("scratch invocation workspace cannot publish a physical backing")
                    })?;
            if binding.key.stage_graph.iter().all(|byte| *byte == 0)
                || binding.key.domain.get() == 0
                || binding.backing.workspace_domain().id() != binding.key.domain
                || !binding.backing.identity().validates_kind(authored_kind)
            {
                return Err(invalid(
                    "invocation workspace binding has an incomplete or mismatched identity",
                ));
            }
            if backings.insert(binding.key, binding.backing).is_some() {
                return Err(invalid(
                    "invocation workspace repeats one graph/stage/domain binding",
                ));
            }
        }
        Ok(Self { backings })
    }

    fn validate_for(
        &self,
        descriptor: &CapabilityStateDescriptorV2,
        execution: &ExecutionAdapterBinding,
        backend: BackendKind,
    ) -> Result<()> {
        let selected_graph = stage_graph_fingerprint(&execution.stages)?;
        let mut authored = HashMap::new();
        if let InvocationWorkspaceSet::Bounded { profiles } = &descriptor.invocation {
            for profile in profiles {
                for stage in &profile.stages {
                    for domain in &stage.domains {
                        if matches!(domain, InvocationWorkspaceDomain::State { .. }) {
                            authored.insert(
                                InvocationWorkspaceKeyV2 {
                                    stage_graph: profile.stage_graph_fingerprint,
                                    stage: stage.stage,
                                    domain: domain.id(),
                                },
                                domain,
                            );
                        }
                    }
                }
            }
        }

        for (key, backing) in &self.backings {
            let domain = authored.get(key).ok_or_else(|| {
                invalid("invocation workspace publishes an unauthored graph/stage/domain backing")
            })?;
            backing.validate_live()?;
            if backing.workspace_domain() != *domain
                || !backing.identity().validates_kind(
                    InvocationStateBackingKindV2::for_workspace(domain)
                        .expect("authored state workspace has a physical kind"),
                )
                || !backing
                    .identity()
                    .validates_owner(*key, backend, execution.model_instance_id)
            {
                return Err(invalid(
                    "invocation workspace backing does not match its authored domain",
                ));
            }
        }

        let expected = authored
            .iter()
            .filter(|(key, _)| key.stage_graph == selected_graph)
            .count();
        let actual = self
            .backings
            .keys()
            .filter(|key| key.stage_graph == selected_graph)
            .count();
        if actual != expected {
            return Err(invalid(
                "invocation workspace backing does not exactly cover the selected descriptor",
            ));
        }
        for key in authored
            .keys()
            .filter(|key| key.stage_graph == selected_graph)
        {
            if !self.backings.contains_key(key) {
                return Err(invalid(
                    "invocation workspace is missing a selected graph/stage/domain backing",
                ));
            }
        }
        Ok(())
    }

    fn backing_ids_for_graph(
        &self,
        graph: [u8; 32],
    ) -> Vec<(
        InvocationWorkspaceKeyV2,
        InvocationWorkspaceBackingIdentityV2,
    )> {
        let mut ids = self
            .backings
            .iter()
            .filter(|(key, _)| key.stage_graph == graph)
            .map(|(key, backing)| (*key, backing.identity()))
            .collect::<Vec<_>>();
        ids.sort_unstable_by_key(|(key, _)| (key.stage, key.domain));
        ids
    }

    fn lease(
        &self,
        graph: [u8; 32],
        stage: StageId,
        domain: StateDomainId,
    ) -> Result<InvocationWorkspaceLeaseV2> {
        let key = InvocationWorkspaceKeyV2 {
            stage_graph: graph,
            stage,
            domain,
        };
        let backing = self
            .backings
            .get(&key)
            .ok_or_else(|| invalid("selected invocation workspace is not load-sealed"))?;
        backing.validate_live()?;
        let mut physical = backing.lease()?;
        let identity = backing.identity();
        if physical.backing_identity() != identity {
            physical.abort();
            return Err(invalid(
                "invocation workspace lease does not authenticate its published backing",
            ));
        }
        Ok(InvocationWorkspaceLeaseV2 {
            key,
            backing_identity: identity,
            backing: backing.clone(),
            physical: Some(physical),
        })
    }

    fn lease_set(
        &self,
        graph: [u8; 32],
        stage: StageId,
        domains: &[StateDomainId],
    ) -> Result<InvocationWorkspaceLeaseSetV2> {
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
            leases.push((domain, self.lease(graph, stage, domain)?));
        }
        Ok(InvocationWorkspaceLeaseSetV2 { stage, leases })
    }
}

impl From<InvocationPagedWorkspaceRuntimeV2> for InvocationWorkspaceRuntimeV2 {
    fn from(runtime: InvocationPagedWorkspaceRuntimeV2) -> Self {
        let backings = runtime
            .pools
            .into_iter()
            .map(|(key, pool)| {
                (
                    key,
                    Arc::new(InvocationPagedWorkspaceBackingV2 { pool })
                        as Arc<dyn InvocationWorkspaceBackingV2>,
                )
            })
            .collect();
        Self { backings }
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
    invocation_workspace: InvocationWorkspaceRuntimeV2,
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

    fn lease_invocation_workspace(
        &self,
        stage: StageId,
        domain: StateDomainId,
    ) -> Result<InvocationWorkspaceLeaseV2> {
        match &self.backing {
            CapabilityStateRuntimeBackingV2::Stateless(_) => Err(invalid(
                "stateless runtime has no load-sealed invocation workspace",
            )),
            CapabilityStateRuntimeBackingV2::Invocation(runtime) => runtime
                .invocation_workspace
                .lease(runtime.stage_graph_fingerprint, stage, domain),
            CapabilityStateRuntimeBackingV2::Managed(runtime) => runtime
                .invocation_workspace
                .lease(runtime.stage_graph_fingerprint, stage, domain),
        }
    }

    fn lease_invocation_workspace_set(
        &self,
        stage: StageId,
        domains: &[StateDomainId],
    ) -> Result<InvocationWorkspaceLeaseSetV2> {
        match &self.backing {
            CapabilityStateRuntimeBackingV2::Stateless(_) => Err(invalid(
                "stateless runtime has no load-sealed invocation workspace",
            )),
            CapabilityStateRuntimeBackingV2::Invocation(runtime) => runtime
                .invocation_workspace
                .lease_set(runtime.stage_graph_fingerprint, stage, domains),
            CapabilityStateRuntimeBackingV2::Managed(runtime) => runtime
                .invocation_workspace
                .lease_set(runtime.stage_graph_fingerprint, stage, domains),
        }
    }

    /// Lease every typed state domain authored for one exact stage. Scratch is
    /// formula-only stage workspace and therefore never appears in this set.
    pub(crate) fn lease_complete_invocation_workspace_set(
        &self,
        stage: StageId,
    ) -> Result<InvocationWorkspaceLeaseSetV2> {
        let graph = match &self.backing {
            CapabilityStateRuntimeBackingV2::Stateless(_) => {
                return Err(invalid(
                    "stateless runtime has no load-sealed invocation workspace",
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
                InvocationWorkspaceDomain::State { state, .. } => Some(state.id()),
                InvocationWorkspaceDomain::Scratch { .. } => None,
            })
            .collect::<Vec<_>>();
        if domains.is_empty() {
            return Ok(InvocationWorkspaceLeaseSetV2 {
                stage,
                leases: Vec::new(),
            });
        }
        self.lease_invocation_workspace_set(stage, &domains)
    }

    pub(crate) fn lease_invocation_paged(
        &self,
        stage: StageId,
        domain: StateDomainId,
    ) -> Result<InvocationPagedKvLease> {
        let authored = self.complete_paged_domains_for_stage(stage)?;
        if authored.as_slice() != [domain] {
            return Err(invalid(
                "single-domain paged compatibility requires the complete authored stage set",
            ));
        }
        self.lease_invocation_workspace(stage, domain)?.into_paged()
    }

    fn lease_invocation_paged_set(
        &self,
        stage: StageId,
        domains: &[StateDomainId],
    ) -> Result<InvocationPagedLeaseSetV2> {
        let mut requested = domains.to_vec();
        requested.sort_unstable();
        if requested.windows(2).any(|pair| pair[0] == pair[1])
            || requested != self.complete_paged_domains_for_stage(stage)?
        {
            return Err(invalid(
                "paged compatibility requires the complete authored stage set",
            ));
        }
        self.lease_invocation_workspace_set(stage, domains)?
            .into_paged()
    }

    fn complete_paged_domains_for_stage(&self, stage: StageId) -> Result<Vec<StateDomainId>> {
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
        let mut domains = Vec::new();
        for domain in &workspace.domains {
            match domain {
                InvocationWorkspaceDomain::State {
                    state: super::StateDomainSpec::PagedAttention(state),
                    ..
                } => domains.push(state.header.id),
                InvocationWorkspaceDomain::State { .. } => {
                    return Err(invalid(
                        "paged invocation compatibility cannot lease a mixed typed workspace",
                    ));
                }
                InvocationWorkspaceDomain::Scratch { .. } => {}
            }
        }
        domains.sort_unstable();
        if domains.is_empty() {
            return Err(invalid(
                "paged invocation compatibility requires an authored paged domain",
            ));
        }
        Ok(domains)
    }

    /// Lease every paged domain authored for one exact stage. Domain selection
    /// is sealed into the runtime descriptor so direct and engine runners
    /// cannot accidentally execute with a partial cache set.
    pub(crate) fn lease_complete_invocation_paged_set(
        &self,
        stage: StageId,
    ) -> Result<InvocationPagedLeaseSetV2> {
        let domains = self.complete_paged_domains_for_stage(stage)?;
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
    invocation_workspace: InvocationWorkspaceRuntimeV2,
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
        Self::seal_with_invocation_workspace(
            backend,
            execution,
            descriptor,
            physical,
            retained_state_use,
            invocation_paged.into(),
        )
    }

    pub(crate) fn seal_with_invocation_workspace(
        backend: BackendKind,
        execution: &ExecutionAdapterBinding,
        descriptor: CapabilityStateDescriptorV2,
        physical: impl Into<RetainedStateRuntimeV2>,
        retained_state_use: RetainedStateUseV2,
        invocation_workspace: InvocationWorkspaceRuntimeV2,
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
            invocation_workspace,
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
        self.invocation_workspace
            .validate_for(&self.descriptor, execution, backend)?;
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
            invocation_workspace: Vec<(
                InvocationWorkspaceKeyV2,
                InvocationWorkspaceBackingIdentityV2,
            )>,
        }
        let encoded = serde_json::to_vec(&Payload {
            identity: &self.identity,
            stage_graph_fingerprint: self.stage_graph_fingerprint,
            state_fingerprint: self.state_fingerprint,
            state_plan: self.state_plan.id,
            retained_identity: self.retained_identity,
            retained_state_use: self.retained_state_use,
            invocation_workspace: self
                .invocation_workspace
                .backing_ids_for_graph(self.stage_graph_fingerprint),
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
        Self::seal_with_invocation_workspace(
            backend,
            execution,
            descriptor,
            invocation_paged.into(),
        )
    }

    pub(crate) fn seal_with_invocation_workspace(
        backend: BackendKind,
        execution: &ExecutionAdapterBinding,
        descriptor: CapabilityStateDescriptorV2,
        invocation_workspace: InvocationWorkspaceRuntimeV2,
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
        invocation_workspace.validate_for(&descriptor, execution, backend)?;
        let stage_graph_fingerprint = stage_graph_fingerprint(&execution.stages)?;
        let state_fingerprint = descriptor.fingerprint(&execution.stages)?;
        let mut runtime = Self {
            id: [0; 32],
            identity: CapabilityRuntimeIdentityV2::seal(backend, execution)?,
            stage_graph_fingerprint,
            state_fingerprint,
            descriptor,
            invocation_workspace,
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
        self.invocation_workspace
            .validate_for(&self.descriptor, execution, backend)?;
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
            invocation_workspace: Vec<(
                InvocationWorkspaceKeyV2,
                InvocationWorkspaceBackingIdentityV2,
            )>,
        }
        let encoded = serde_json::to_vec(&Payload {
            identity: &self.identity,
            stage_graph_fingerprint: self.stage_graph_fingerprint,
            state_fingerprint: self.state_fingerprint,
            invocation_workspace: self
                .invocation_workspace
                .backing_ids_for_graph(self.stage_graph_fingerprint),
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
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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
        AppendStateDomainSpec, BoundedShape, CheckpointPolicy, InferenceStateContract,
        InvocationStageWorkspace, InvocationStateCapacity, InvocationWorkspaceProfile,
        PlacementPolicy, PrefixPolicy, RingStateDomainSpec, ShapeAxis, ShapeDimension, ShapeExtent,
        StateClock, StateComponentId, StateDType, StateDomainHeader, StateDomainSpec, StateGroupId,
        StateGroupSpec, StateScope, StaticAttentionDomainSpec, StaticAttentionLayerSpec,
        StaticTensorDomainSpec, TensorComponentSpec, TensorRole, TensorStateDomainSpec,
        WorkspaceFormula, CURRENT_INFERENCE_STATE_ABI,
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

    fn invocation_header(id: u32) -> StateDomainHeader {
        StateDomainHeader {
            id: StateDomainId::new(id),
            scope: StateScope::Invocation,
            clock: StateClock::DecoderTokens,
            placement: PlacementPolicy::BackendLocal,
            prefix: PrefixPolicy::Disabled,
            checkpoint: CheckpointPolicy::None,
        }
    }

    fn invocation_component() -> TensorComponentSpec {
        TensorComponentSpec {
            id: StateComponentId::new(1),
            role: TensorRole::Control,
            shape: BoundedShape {
                dimensions: vec![ShapeDimension {
                    axis: ShapeAxis::Hidden,
                    extent: ShapeExtent::Fixed { value: 8 },
                }],
            },
            accepted_dtypes: vec![StateDType::F32],
        }
    }

    fn typed_invocation_domain(
        kind: InvocationStateBackingKindV2,
        id: u32,
    ) -> InvocationWorkspaceDomain {
        let header = invocation_header(id);
        let component = invocation_component();
        let state = match kind {
            InvocationStateBackingKindV2::PagedAttention => {
                panic!("paged test domains require a real physical pool")
            }
            InvocationStateBackingKindV2::StaticAttention => {
                StateDomainSpec::StaticAttention(StaticAttentionDomainSpec {
                    header,
                    layers: vec![StaticAttentionLayerSpec {
                        model_layer: 0,
                        query_heads: 4,
                        kv_heads: 2,
                        key_head_dim: 8,
                        value_head_dim: 8,
                        key_encoding: super::super::KeyEncoding::Raw,
                    }],
                    max_memory_tokens: 32,
                    accepted_dtypes: vec![StateDType::F32],
                })
            }
            InvocationStateBackingKindV2::Tensor => {
                StateDomainSpec::Tensor(TensorStateDomainSpec {
                    header,
                    components: vec![component],
                })
            }
            InvocationStateBackingKindV2::Append => {
                StateDomainSpec::Append(AppendStateDomainSpec {
                    header,
                    components_per_step: vec![component],
                    max_steps: 32,
                })
            }
            InvocationStateBackingKindV2::Ring => StateDomainSpec::Ring(RingStateDomainSpec {
                header,
                components_per_step: vec![component],
                capacity_steps: 8,
            }),
            InvocationStateBackingKindV2::StaticTensor => {
                StateDomainSpec::StaticTensor(StaticTensorDomainSpec {
                    header,
                    components: vec![component],
                })
            }
        };
        InvocationWorkspaceDomain::State {
            state,
            capacity: InvocationStateCapacity::SemanticBounded,
            placement: PlacementPolicy::BackendLocal,
            formula: WorkspaceFormula {
                fixed_bytes: 4096,
                dimensions: vec![],
                terms: vec![],
            },
        }
    }

    #[derive(Debug)]
    struct TestInvocationBackingState {
        leased: AtomicBool,
        live: AtomicBool,
        releases: AtomicUsize,
    }

    #[derive(Debug)]
    struct TestInvocationBacking {
        identity: InvocationWorkspaceBackingIdentityV2,
        workspace_domain: InvocationWorkspaceDomain,
        state: Arc<TestInvocationBackingState>,
        corrupt_completion: bool,
    }

    impl TestInvocationBacking {
        fn new(
            kind: InvocationStateBackingKindV2,
            workspace_domain: InvocationWorkspaceDomain,
            model_instance: ModelInstanceId,
            identity_byte: u8,
            corrupt_completion: bool,
        ) -> Arc<Self> {
            let domain = workspace_domain.id();
            Arc::new(Self {
                identity: InvocationWorkspaceBackingIdentityV2::Typed {
                    kind,
                    model_instance,
                    backend: BackendKind::Cpu,
                    domain,
                    allocation_generation: u32::from(identity_byte).max(1),
                },
                workspace_domain,
                state: Arc::new(TestInvocationBackingState {
                    leased: AtomicBool::new(false),
                    live: AtomicBool::new(true),
                    releases: AtomicUsize::new(0),
                }),
                corrupt_completion,
            })
        }
    }

    impl InvocationWorkspaceBackingV2 for TestInvocationBacking {
        fn identity(&self) -> InvocationWorkspaceBackingIdentityV2 {
            self.identity
        }

        fn workspace_domain(&self) -> &InvocationWorkspaceDomain {
            &self.workspace_domain
        }

        fn validate_live(&self) -> Result<()> {
            if self.state.live.load(Ordering::Acquire) {
                Ok(())
            } else {
                Err(invalid("test invocation backing is closed"))
            }
        }

        fn lease(&self) -> Result<Box<dyn InvocationWorkspacePhysicalLeaseV2>> {
            self.validate_live()?;
            self.state
                .leased
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .map_err(|_| invalid("test invocation backing is already leased"))?;
            Ok(Box::new(TestInvocationPhysicalLease {
                identity: self.identity,
                state: self.state.clone(),
                active: true,
                corrupt_completion: self.corrupt_completion,
            }))
        }

        fn authenticate_completion(
            &self,
            completion: &InvocationWorkspacePhysicalCompletionV2,
        ) -> Result<()> {
            match completion {
                InvocationWorkspacePhysicalCompletionV2::Typed {
                    backing,
                    authentication,
                } if *backing == self.identity && *authentication == [0x5a; 32] => Ok(()),
                _ => Err(invalid(
                    "test invocation completion failed backing authentication",
                )),
            }
        }
    }

    #[derive(Debug)]
    struct TestInvocationPhysicalLease {
        identity: InvocationWorkspaceBackingIdentityV2,
        state: Arc<TestInvocationBackingState>,
        active: bool,
        corrupt_completion: bool,
    }

    impl TestInvocationPhysicalLease {
        fn finish(&mut self) {
            if self.active {
                self.state.leased.store(false, Ordering::Release);
                self.state.releases.fetch_add(1, Ordering::AcqRel);
                self.active = false;
            }
        }
    }

    impl InvocationWorkspacePhysicalLeaseV2 for TestInvocationPhysicalLease {
        fn backing_identity(&self) -> InvocationWorkspaceBackingIdentityV2 {
            self.identity
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }

        fn into_any(self: Box<Self>) -> Box<dyn Any> {
            self
        }

        fn complete(&mut self) -> Result<InvocationWorkspacePhysicalCompletionV2> {
            self.finish();
            let backing = if self.corrupt_completion {
                match self.identity {
                    InvocationWorkspaceBackingIdentityV2::Typed {
                        kind,
                        model_instance,
                        backend,
                        allocation_generation,
                        ..
                    } => InvocationWorkspaceBackingIdentityV2::Typed {
                        kind,
                        model_instance,
                        backend,
                        domain: StateDomainId::new(0xff),
                        allocation_generation,
                    },
                    InvocationWorkspaceBackingIdentityV2::Paged(_) => unreachable!(),
                }
            } else {
                self.identity
            };
            Ok(InvocationWorkspacePhysicalCompletionV2::Typed {
                backing,
                authentication: [0x5a; 32],
            })
        }

        fn abort(&mut self) {
            self.finish();
        }
    }

    #[derive(Debug, Clone, Copy)]
    enum HostileInvocationBehavior {
        AcquisitionIdentityMismatch,
        AuthenticationFailure,
        PanicDuringCompletion,
    }

    #[derive(Debug)]
    struct HostileInvocationBacking {
        identity: InvocationWorkspaceBackingIdentityV2,
        workspace_domain: InvocationWorkspaceDomain,
        state: Arc<TestInvocationBackingState>,
        behavior: HostileInvocationBehavior,
    }

    impl HostileInvocationBacking {
        fn new(behavior: HostileInvocationBehavior) -> Arc<Self> {
            let domain = StateDomainId::new(70);
            Arc::new(Self {
                identity: InvocationWorkspaceBackingIdentityV2::Typed {
                    kind: InvocationStateBackingKindV2::Tensor,
                    model_instance: ModelInstanceId::new(3),
                    backend: BackendKind::Cpu,
                    domain,
                    allocation_generation: 70,
                },
                workspace_domain: typed_invocation_domain(
                    InvocationStateBackingKindV2::Tensor,
                    domain.get(),
                ),
                state: Arc::new(TestInvocationBackingState {
                    leased: AtomicBool::new(false),
                    live: AtomicBool::new(true),
                    releases: AtomicUsize::new(0),
                }),
                behavior,
            })
        }
    }

    impl InvocationWorkspaceBackingV2 for HostileInvocationBacking {
        fn identity(&self) -> InvocationWorkspaceBackingIdentityV2 {
            self.identity
        }

        fn workspace_domain(&self) -> &InvocationWorkspaceDomain {
            &self.workspace_domain
        }

        fn validate_live(&self) -> Result<()> {
            Ok(())
        }

        fn lease(&self) -> Result<Box<dyn InvocationWorkspacePhysicalLeaseV2>> {
            self.state
                .leased
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .map_err(|_| invalid("hostile invocation backing is already leased"))?;
            let identity = if matches!(
                self.behavior,
                HostileInvocationBehavior::AcquisitionIdentityMismatch
            ) {
                InvocationWorkspaceBackingIdentityV2::Typed {
                    kind: InvocationStateBackingKindV2::Tensor,
                    model_instance: ModelInstanceId::new(3),
                    backend: BackendKind::Cpu,
                    domain: StateDomainId::new(71),
                    allocation_generation: 70,
                }
            } else {
                self.identity
            };
            Ok(Box::new(HostileInvocationPhysicalLease {
                identity,
                state: self.state.clone(),
                active: true,
                behavior: self.behavior,
            }))
        }

        fn authenticate_completion(
            &self,
            completion: &InvocationWorkspacePhysicalCompletionV2,
        ) -> Result<()> {
            if matches!(
                self.behavior,
                HostileInvocationBehavior::AuthenticationFailure
            ) {
                return Err(invalid("hostile invocation authentication failed"));
            }
            match completion {
                InvocationWorkspacePhysicalCompletionV2::Typed {
                    backing,
                    authentication,
                } if *backing == self.identity && *authentication == [0x5a; 32] => Ok(()),
                _ => Err(invalid("hostile invocation completion is invalid")),
            }
        }
    }

    #[derive(Debug)]
    struct HostileInvocationPhysicalLease {
        identity: InvocationWorkspaceBackingIdentityV2,
        state: Arc<TestInvocationBackingState>,
        active: bool,
        behavior: HostileInvocationBehavior,
    }

    impl HostileInvocationPhysicalLease {
        fn abort_once(&mut self) {
            if self.active {
                self.state.leased.store(false, Ordering::Release);
                self.state.releases.fetch_add(1, Ordering::AcqRel);
                self.active = false;
            }
        }
    }

    impl InvocationWorkspacePhysicalLeaseV2 for HostileInvocationPhysicalLease {
        fn backing_identity(&self) -> InvocationWorkspaceBackingIdentityV2 {
            self.identity
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }

        fn into_any(self: Box<Self>) -> Box<dyn Any> {
            self
        }

        fn complete(&mut self) -> Result<InvocationWorkspacePhysicalCompletionV2> {
            if matches!(
                self.behavior,
                HostileInvocationBehavior::PanicDuringCompletion
            ) {
                panic!("hostile invocation completion panic");
            }
            Ok(InvocationWorkspacePhysicalCompletionV2::Typed {
                backing: self.identity,
                authentication: [0x5a; 32],
            })
        }

        fn abort(&mut self) {
            self.abort_once();
        }
    }

    fn hostile_invocation_runtime(
        behavior: HostileInvocationBehavior,
    ) -> (
        InvocationWorkspaceRuntimeV2,
        Arc<HostileInvocationBacking>,
        InvocationWorkspaceKeyV2,
    ) {
        let backing = HostileInvocationBacking::new(behavior);
        let key = InvocationWorkspaceKeyV2 {
            stage_graph: [0x70; 32],
            stage: StageId::new(70),
            domain: StateDomainId::new(70),
        };
        (
            InvocationWorkspaceRuntimeV2::new(vec![InvocationWorkspaceBindingV2 {
                key,
                backing: backing.clone(),
            }])
            .unwrap(),
            backing,
            key,
        )
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
    fn invocation_runtime_bindings_accept_the_canonical_scalar_stage_zero() {
        let typed = HostileInvocationBacking::new(HostileInvocationBehavior::AuthenticationFailure);
        InvocationWorkspaceRuntimeV2::new(vec![InvocationWorkspaceBindingV2 {
            key: InvocationWorkspaceKeyV2 {
                stage_graph: [0x71; 32],
                stage: StageId::new(0),
                domain: StateDomainId::new(70),
            },
            backing: typed,
        }])
        .unwrap();

        let (paged, _) = invocation_pool(ModelInstanceId::new(38));
        InvocationPagedWorkspaceRuntimeV2::new(vec![InvocationPagedWorkspaceBindingV2 {
            key: InvocationPagedWorkspaceKeyV2 {
                stage_graph: [0x72; 32],
                stage: StageId::new(0),
                domain: StateDomainId::new(1),
            },
            pool: paged.handle(),
        }])
        .unwrap();
    }

    #[test]
    fn invocation_runtime_accepts_authored_backings_for_other_selectable_graphs() {
        let (first_execution, mut descriptor, _, first_pool) = managed_invocation_fixture();
        descriptor.retained = RetainedStateCapability::Stateless;
        let mut second_execution = first_execution.clone();
        Arc::make_mut(&mut second_execution.stages)[0].name =
            "tts.alternate-invocation-graph".to_string();

        let first_graph = stage_graph_fingerprint(&first_execution.stages).unwrap();
        let second_graph = stage_graph_fingerprint(&second_execution.stages).unwrap();
        assert_ne!(first_graph, second_graph);

        let InvocationWorkspaceSet::Bounded { profiles } = &mut descriptor.invocation else {
            unreachable!()
        };
        let mut second_profile = profiles[0].clone();
        second_profile.stage_graph_fingerprint = second_graph;
        profiles.push(second_profile);
        profiles.sort_unstable_by_key(|profile| profile.stage_graph_fingerprint);
        descriptor
            .validate_against_stages(&first_execution.stages)
            .unwrap();
        descriptor
            .validate_against_stages(&second_execution.stages)
            .unwrap();

        let domain = StateDomainId::new(1);
        let (second_pool, _) =
            invocation_pool_for_domain(first_execution.model_instance_id, domain, 20);
        let invocation = InvocationPagedWorkspaceRuntimeV2::new(vec![
            InvocationPagedWorkspaceBindingV2 {
                key: InvocationPagedWorkspaceKeyV2 {
                    stage_graph: first_graph,
                    stage: first_execution.stages[0].id,
                    domain,
                },
                pool: first_pool.handle(),
            },
            InvocationPagedWorkspaceBindingV2 {
                key: InvocationPagedWorkspaceKeyV2 {
                    stage_graph: second_graph,
                    stage: second_execution.stages[0].id,
                    domain,
                },
                pool: second_pool.handle(),
            },
        ])
        .unwrap();
        let invocation: InvocationWorkspaceRuntimeV2 = invocation.into();

        InvocationCapabilityRuntimeV2::seal_with_invocation_workspace(
            BackendKind::Cpu,
            &first_execution,
            descriptor.clone(),
            invocation.clone(),
        )
        .unwrap();
        InvocationCapabilityRuntimeV2::seal_with_invocation_workspace(
            BackendKind::Cpu,
            &second_execution,
            descriptor,
            invocation,
        )
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

    fn all_domain_invocation_fixture(
        corrupt_domain: Option<StateDomainId>,
    ) -> (
        InvocationWorkspaceRuntimeV2,
        CapabilityStateDescriptorV2,
        InvocationPagedKvPoolOwner,
        Vec<Arc<TestInvocationBacking>>,
        ExecutionAdapterBinding,
    ) {
        let mut execution = binding();
        Arc::make_mut(&mut execution.stages)[0].max_workspace_bytes = 2048;
        let graph = stage_graph_fingerprint(&execution.stages).unwrap();
        let stage = execution.stages[0].id;
        let (paged_owner, paged_domain) =
            invocation_pool_for_domain(execution.model_instance_id, StateDomainId::new(1), 41);
        let kinds = [
            InvocationStateBackingKindV2::StaticAttention,
            InvocationStateBackingKindV2::Tensor,
            InvocationStateBackingKindV2::Append,
            InvocationStateBackingKindV2::Ring,
            InvocationStateBackingKindV2::StaticTensor,
        ];
        let mut domains = vec![paged_domain.clone()];
        let mut typed_backings = Vec::new();
        let mut bindings = vec![InvocationWorkspaceBindingV2 {
            key: InvocationWorkspaceKeyV2 {
                stage_graph: graph,
                stage,
                domain: StateDomainId::new(1),
            },
            backing: Arc::new(InvocationPagedWorkspaceBackingV2 {
                pool: paged_owner.handle(),
            }),
        }];
        for (offset, kind) in kinds.into_iter().enumerate() {
            let id = StateDomainId::new(u32::try_from(offset).unwrap() + 2);
            let domain = typed_invocation_domain(kind, id.get());
            let backing = TestInvocationBacking::new(
                kind,
                domain.clone(),
                execution.model_instance_id,
                u8::try_from(id.get()).unwrap(),
                corrupt_domain == Some(id),
            );
            domains.push(domain);
            bindings.push(InvocationWorkspaceBindingV2 {
                key: InvocationWorkspaceKeyV2 {
                    stage_graph: graph,
                    stage,
                    domain: id,
                },
                backing: backing.clone(),
            });
            typed_backings.push(backing);
        }
        domains.push(InvocationWorkspaceDomain::Scratch {
            id: StateDomainId::new(99),
            placement: PlacementPolicy::BackendLocal,
            alignment_bytes: 64,
            zero_on_release: true,
            formula: WorkspaceFormula {
                fixed_bytes: 2048,
                dimensions: vec![],
                terms: vec![],
            },
        });
        let descriptor = CapabilityStateDescriptorV2 {
            abi: CURRENT_INFERENCE_STATE_ABI,
            retained: RetainedStateCapability::Stateless,
            invocation: InvocationWorkspaceSet::Bounded {
                profiles: vec![InvocationWorkspaceProfile {
                    stage_graph_fingerprint: graph,
                    stages: vec![InvocationStageWorkspace {
                        stage,
                        lease_scope: super::super::InvocationLeaseScope::PerStageBatch,
                        groups: vec![StateGroupSpec {
                            id: StateGroupId::new(1),
                            domains: (1..=6).map(StateDomainId::new).collect(),
                            prefix_shareable: false,
                        }],
                        domains,
                    }],
                }],
            },
        };
        descriptor
            .validate_against_stages(&execution.stages)
            .unwrap();
        (
            InvocationWorkspaceRuntimeV2::new(bindings).unwrap(),
            descriptor,
            paged_owner,
            typed_backings,
            execution,
        )
    }

    #[test]
    fn invocation_workspace_runtime_covers_every_typed_domain_and_excludes_scratch() {
        let (runtime, descriptor, _paged_owner, typed, execution) =
            all_domain_invocation_fixture(None);
        runtime
            .validate_for(&descriptor, &execution, BackendKind::Cpu)
            .unwrap();
        let sealed = CapabilityStateRuntimeV2::invocation(
            InvocationCapabilityRuntimeV2::seal_with_invocation_workspace(
                BackendKind::Cpu,
                &execution,
                descriptor.clone(),
                runtime.clone(),
            )
            .unwrap(),
        );
        assert!(sealed
            .lease_complete_invocation_paged_set(execution.stages[0].id)
            .is_err());
        assert!(sealed
            .lease_invocation_paged(execution.stages[0].id, StateDomainId::new(1))
            .is_err());
        let graph = stage_graph_fingerprint(&execution.stages).unwrap();
        let stage = execution.stages[0].id;
        let mut leases = runtime
            .lease_set(
                graph,
                stage,
                &[
                    StateDomainId::new(6),
                    StateDomainId::new(5),
                    StateDomainId::new(4),
                    StateDomainId::new(3),
                    StateDomainId::new(2),
                    StateDomainId::new(1),
                ],
            )
            .unwrap();
        assert_eq!(
            leases.domains().collect::<Vec<_>>(),
            (1..=6).map(StateDomainId::new).collect::<Vec<_>>()
        );
        assert_eq!(
            leases
                .lease(StateDomainId::new(1))
                .unwrap()
                .paged_cache()
                .unwrap()
                .context_len(),
            0
        );
        assert!(leases
            .lease_mut(StateDomainId::new(2))
            .unwrap()
            .typed_mut::<TestInvocationPhysicalLease>()
            .is_ok());
        let completions = leases.release().unwrap();
        assert_eq!(completions.len(), 6);
        assert!(completions
            .iter()
            .all(|completion| completion.key.domain != StateDomainId::new(99)));
        assert_eq!(
            completions
                .iter()
                .filter(|completion| matches!(
                    &completion.physical,
                    InvocationWorkspacePhysicalCompletionV2::Typed {
                        authentication,
                        ..
                    } if *authentication == [0x5a; 32]
                ))
                .count(),
            5
        );
        assert!(typed
            .iter()
            .all(|backing| backing.state.releases.load(Ordering::Acquire) == 1));
    }

    #[test]
    fn invocation_workspace_lease_set_borrows_distinct_domain_pairs_in_requested_order() {
        let (runtime, _descriptor, _paged_owner, _typed, execution) =
            all_domain_invocation_fixture(None);
        let graph = stage_graph_fingerprint(&execution.stages).unwrap();
        let stage = execution.stages[0].id;
        let mut leases = runtime
            .lease_set(
                graph,
                stage,
                &[StateDomainId::new(1), StateDomainId::new(2)],
            )
            .unwrap();

        let (typed, paged) = leases
            .lease_pair_mut(StateDomainId::new(2), StateDomainId::new(1))
            .unwrap();
        assert_eq!(typed.kind(), InvocationStateBackingKindV2::StaticAttention);
        assert_eq!(paged.kind(), InvocationStateBackingKindV2::PagedAttention);
        assert!(typed.typed_mut::<TestInvocationPhysicalLease>().is_ok());
        assert_eq!(paged.paged_cache_mut().unwrap().context_len(), 0);

        assert!(leases
            .lease_pair_mut(StateDomainId::new(1), StateDomainId::new(1))
            .is_err());
        assert!(leases
            .lease_pair_mut(StateDomainId::new(99), StateDomainId::new(1))
            .is_err());
        assert!(leases
            .lease_pair_mut(StateDomainId::new(1), StateDomainId::new(99))
            .is_err());
        assert_eq!(leases.release().unwrap().len(), 2);
    }

    #[test]
    fn invocation_workspace_lease_set_borrows_distinct_domain_triplets_in_requested_order() {
        let (runtime, _descriptor, _paged_owner, _typed, execution) =
            all_domain_invocation_fixture(None);
        let graph = stage_graph_fingerprint(&execution.stages).unwrap();
        let stage = execution.stages[0].id;
        let mut leases = runtime
            .lease_set(
                graph,
                stage,
                &[
                    StateDomainId::new(1),
                    StateDomainId::new(2),
                    StateDomainId::new(3),
                ],
            )
            .unwrap();

        let (tensor, paged, static_attention) = leases
            .lease_triplet_mut(
                StateDomainId::new(3),
                StateDomainId::new(1),
                StateDomainId::new(2),
            )
            .unwrap();
        assert_eq!(tensor.kind(), InvocationStateBackingKindV2::Tensor);
        assert_eq!(paged.kind(), InvocationStateBackingKindV2::PagedAttention);
        assert_eq!(
            static_attention.kind(),
            InvocationStateBackingKindV2::StaticAttention
        );
        assert!(tensor.typed_mut::<TestInvocationPhysicalLease>().is_ok());
        assert_eq!(paged.paged_cache_mut().unwrap().context_len(), 0);

        assert!(leases
            .lease_triplet_mut(
                StateDomainId::new(1),
                StateDomainId::new(1),
                StateDomainId::new(2),
            )
            .is_err());
        assert!(leases
            .lease_triplet_mut(
                StateDomainId::new(1),
                StateDomainId::new(2),
                StateDomainId::new(99),
            )
            .is_err());
        assert_eq!(leases.release().unwrap().len(), 3);
    }

    #[test]
    fn invocation_workspace_lease_set_borrows_an_exact_mixed_kind_pair() {
        let (runtime, _descriptor, _paged_owner, _typed, execution) =
            all_domain_invocation_fixture(None);
        let graph = stage_graph_fingerprint(&execution.stages).unwrap();
        let stage = execution.stages[0].id;
        let mut leases = runtime
            .lease_set(
                graph,
                stage,
                &[StateDomainId::new(1), StateDomainId::new(2)],
            )
            .unwrap();

        let (paged, static_attention) = leases
            .lease_exact_kind_pair_mut(
                InvocationStateBackingKindV2::PagedAttention,
                InvocationStateBackingKindV2::StaticAttention,
            )
            .unwrap();
        assert_eq!(paged.kind(), InvocationStateBackingKindV2::PagedAttention);
        assert_eq!(
            static_attention.kind(),
            InvocationStateBackingKindV2::StaticAttention
        );
        assert!(leases
            .lease_exact_kind_pair_mut(
                InvocationStateBackingKindV2::PagedAttention,
                InvocationStateBackingKindV2::PagedAttention,
            )
            .is_err());
        assert!(leases
            .lease_exact_kind_pair_mut(
                InvocationStateBackingKindV2::Tensor,
                InvocationStateBackingKindV2::StaticAttention,
            )
            .is_err());
        assert_eq!(leases.release().unwrap().len(), 2);

        let mut extra = runtime
            .lease_set(
                graph,
                stage,
                &[
                    StateDomainId::new(1),
                    StateDomainId::new(2),
                    StateDomainId::new(3),
                ],
            )
            .unwrap();
        assert!(extra
            .lease_exact_kind_pair_mut(
                InvocationStateBackingKindV2::PagedAttention,
                InvocationStateBackingKindV2::StaticAttention,
            )
            .is_err());
        assert_eq!(extra.release().unwrap().len(), 3);
    }

    #[test]
    fn invocation_workspace_lease_set_borrows_one_exact_kind() {
        let (runtime, _descriptor, _paged_owner, _typed, execution) =
            all_domain_invocation_fixture(None);
        let graph = stage_graph_fingerprint(&execution.stages).unwrap();
        let stage = execution.stages[0].id;
        let mut leases = runtime
            .lease_set(graph, stage, &[StateDomainId::new(1)])
            .unwrap();
        assert!(leases
            .lease_exact_kind_mut(InvocationStateBackingKindV2::PagedAttention)
            .unwrap()
            .paged_cache_mut()
            .is_ok());
        assert!(leases
            .lease_exact_kind_mut(InvocationStateBackingKindV2::StaticAttention)
            .is_err());
        assert_eq!(leases.release().unwrap().len(), 1);

        let mut multiple = runtime
            .lease_set(
                graph,
                stage,
                &[StateDomainId::new(1), StateDomainId::new(2)],
            )
            .unwrap();
        assert!(multiple
            .lease_exact_kind_mut(InvocationStateBackingKindV2::PagedAttention)
            .is_err());
        assert_eq!(multiple.release().unwrap().len(), 2);
    }

    #[test]
    fn invocation_workspace_runtime_rejects_scratch_and_unauthored_publications() {
        let graph = [3; 32];
        let stage = StageId::new(9);
        let scratch = InvocationWorkspaceDomain::Scratch {
            id: StateDomainId::new(7),
            placement: PlacementPolicy::BackendLocal,
            alignment_bytes: 64,
            zero_on_release: false,
            formula: WorkspaceFormula {
                fixed_bytes: 64,
                dimensions: vec![],
                terms: vec![],
            },
        };
        let scratch_backing = TestInvocationBacking::new(
            InvocationStateBackingKindV2::Tensor,
            scratch,
            ModelInstanceId::new(3),
            7,
            false,
        );
        assert!(
            InvocationWorkspaceRuntimeV2::new(vec![InvocationWorkspaceBindingV2 {
                key: InvocationWorkspaceKeyV2 {
                    stage_graph: graph,
                    stage,
                    domain: StateDomainId::new(7),
                },
                backing: scratch_backing,
            }])
            .is_err()
        );

        let (runtime, descriptor, _owner, _typed, execution) = all_domain_invocation_fixture(None);
        let extra = TestInvocationBacking::new(
            InvocationStateBackingKindV2::Tensor,
            typed_invocation_domain(InvocationStateBackingKindV2::Tensor, 77),
            execution.model_instance_id,
            77,
            false,
        );
        let extra_runtime = InvocationWorkspaceRuntimeV2::new(vec![InvocationWorkspaceBindingV2 {
            key: InvocationWorkspaceKeyV2 {
                stage_graph: graph,
                stage,
                domain: StateDomainId::new(77),
            },
            backing: extra,
        }])
        .unwrap();
        assert!(extra_runtime
            .validate_for(&descriptor, &execution, BackendKind::Cpu)
            .is_err());
        runtime
            .validate_for(&descriptor, &execution, BackendKind::Cpu)
            .unwrap();
    }

    #[test]
    fn invocation_workspace_set_rolls_back_and_releases_all_on_authentication_error() {
        let corrupt = StateDomainId::new(2);
        let (runtime, _descriptor, _owner, typed, execution) =
            all_domain_invocation_fixture(Some(corrupt));
        let graph = stage_graph_fingerprint(&execution.stages).unwrap();
        let stage = execution.stages[0].id;
        let held = runtime.lease(graph, stage, StateDomainId::new(3)).unwrap();
        assert!(runtime
            .lease_set(
                graph,
                stage,
                &[StateDomainId::new(2), StateDomainId::new(3)],
            )
            .is_err());
        assert_eq!(typed[0].state.releases.load(Ordering::Acquire), 1);
        drop(held);

        let leases = runtime
            .lease_set(
                graph,
                stage,
                &[StateDomainId::new(2), StateDomainId::new(3)],
            )
            .unwrap();
        assert!(leases.release().is_err());
        assert_eq!(typed[0].state.releases.load(Ordering::Acquire), 2);
        assert_eq!(typed[1].state.releases.load(Ordering::Acquire), 2);
        runtime
            .lease_set(
                graph,
                stage,
                &[StateDomainId::new(2), StateDomainId::new(3)],
            )
            .expect("every domain was released despite one bad completion")
            .release()
            .unwrap_err();
    }

    #[test]
    fn invocation_runtime_aborts_a_hostile_acquisition_identity_mismatch() {
        let (runtime, backing, key) =
            hostile_invocation_runtime(HostileInvocationBehavior::AcquisitionIdentityMismatch);
        assert!(runtime
            .lease(key.stage_graph, key.stage, key.domain)
            .is_err());
        assert!(!backing.state.leased.load(Ordering::Acquire));
        assert_eq!(backing.state.releases.load(Ordering::Acquire), 1);
        assert!(runtime
            .lease(key.stage_graph, key.stage, key.domain)
            .is_err());
        assert_eq!(backing.state.releases.load(Ordering::Acquire), 2);
    }

    #[test]
    fn invocation_runtime_keeps_rollback_armed_through_authentication() {
        let (runtime, backing, key) =
            hostile_invocation_runtime(HostileInvocationBehavior::AuthenticationFailure);
        let lease = runtime
            .lease(key.stage_graph, key.stage, key.domain)
            .unwrap();
        assert!(lease.release().is_err());
        assert!(!backing.state.leased.load(Ordering::Acquire));
        assert_eq!(backing.state.releases.load(Ordering::Acquire), 1);

        let lease = runtime
            .lease(key.stage_graph, key.stage, key.domain)
            .unwrap();
        drop(lease);
        assert_eq!(backing.state.releases.load(Ordering::Acquire), 2);
    }

    #[test]
    fn invocation_runtime_aborts_when_completion_unwinds() {
        let (runtime, backing, key) =
            hostile_invocation_runtime(HostileInvocationBehavior::PanicDuringCompletion);
        let lease = runtime
            .lease(key.stage_graph, key.stage, key.domain)
            .unwrap();
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = lease.release();
        }));
        assert!(unwind.is_err());
        assert!(!backing.state.leased.load(Ordering::Acquire));
        assert_eq!(backing.state.releases.load(Ordering::Acquire), 1);
    }
}
