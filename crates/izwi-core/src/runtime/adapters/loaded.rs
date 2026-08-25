use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::backends::BackendKind;
use crate::engine::ManagedKvModelRuntime;
use crate::engine::{
    AdapterAbiRevision, AdapterInstanceId, CacheMode, CancellationGranularity, ConcurrencyClass,
    ExecutionAdapterBinding, ExecutionGroupId, ExecutionMode, ExecutionProfile, ModelInstanceId,
    NativeBatchMode, OutputVisibility, PhysicalLaunchPolicy, PrefillMode, StageDescriptor, StageId,
    StageShapePolicy, StageWorkSelector,
};
use crate::error::{Error, Result};
use crate::kv::v2::{
    stage_graph_fingerprint, CapabilityStateDescriptorV2, CapabilityStateRuntimeV2,
    InferenceStateContract, InvocationCapabilityRuntimeV2, InvocationWorkspaceRuntimeV2,
    ManagedCapabilityRuntimeV2, RetainedStateCapability, RetainedStateRuntimeV2,
    RetainedStateUseV2, StatelessCapabilityRuntimeV2,
};
use crate::model::ModelVariant;

use super::{
    scalar_execution_profile, AdapterMetadata, CapabilityKind, InferenceStateRequirement,
    RuntimeAdapterRegistry, StreamingMode,
};

const SCALAR_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(11);
const STATIC_TENSOR_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(12);
const CONTINUOUS_TENSOR_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(13);
const NEMOTRON_REALTIME_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(14);
const STATIC_TTS_MAX_BATCH_WORKSPACE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const CONTINUOUS_CHAT_MAX_BATCH_WORKSPACE_BYTES: u64 = 16 * 1024 * 1024;
// Qwen3.8 MTP supports draft depths one through three, which requires an
// isolated target quantum of depth + 1. Shared continuous batches remain one
// work unit per row; this is an aggregate stage ceiling, not a default grant.
const CONTINUOUS_CHAT_MAX_DECODE_QUANTUM: u64 = 4;
static NEXT_ADAPTER_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

/// Streaming has two independent meanings at the loaded-adapter boundary:
/// a transport may publish executor progress even when the model itself does
/// not require a native chunked/realtime decode contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StreamingRequirements {
    pub(crate) transport_output: bool,
    pub(crate) model_native: bool,
}

impl StreamingRequirements {
    pub(crate) const NONE: Self = Self {
        transport_output: false,
        model_native: false,
    };

    pub(crate) const fn native(required: bool) -> Self {
        Self {
            transport_output: required,
            model_native: required,
        }
    }

    pub(crate) const fn transport_only() -> Self {
        Self {
            transport_output: true,
            model_native: false,
        }
    }
}

fn output_visibility_for(
    transport_output: bool,
    execution_mode: ExecutionMode,
    batch_mode: NativeBatchMode,
) -> OutputVisibility {
    if batch_mode == NativeBatchMode::None
        && transport_output
        && execution_mode == ExecutionMode::Atomic
    {
        OutputVisibility::IncrementalCommitted
    } else {
        OutputVisibility::AfterQuantumCommit
    }
}

fn scalar_request_parallelism(backend_kind: BackendKind, configured: usize) -> usize {
    match backend_kind {
        BackendKind::Cpu => configured.max(1),
        // Metal serializes scalar model access. CUDA keeps scalar/per-row
        // invocation state at one resident slot as well: the wider automatic
        // tier belongs to native tensor batches, not to N fully-backed copies
        // of a model's maximum-context workspace.
        BackendKind::Metal | BackendKind::Cuda => 1,
    }
}

/// Catalog metadata, backend selection, and adapter ABI are structural identity,
/// not evidence that distinct physical model calls may overlap. No production
/// concurrency evidence is loaded at this boundary today, so every contract must
/// remain execution-group serialized.
const fn launch_policy_without_concurrency_evidence() -> PhysicalLaunchPolicy {
    PhysicalLaunchPolicy::ExecutionGroupExclusive
}

const fn scalar_row_policy_without_concurrency_evidence() -> (usize, PhysicalLaunchPolicy) {
    (1, launch_policy_without_concurrency_evidence())
}

#[derive(Debug, Clone)]
pub(crate) struct LoadedExecutionContract {
    pub(crate) execution_group_id: ExecutionGroupId,
    pub(crate) model_instance_id: ModelInstanceId,
    pub(crate) adapter_instance_id: AdapterInstanceId,
    pub(crate) adapter_abi_revision: AdapterAbiRevision,
    pub(crate) metadata: AdapterMetadata,
    pub(crate) execution_profile: ExecutionProfile,
    pub(crate) stages: Arc<[StageDescriptor]>,
}

impl LoadedExecutionContract {
    fn validate_physical_launch_policy(&self) -> Result<()> {
        if !self.execution_profile.resolved_from_loaded_model {
            return Err(Error::ModelLoadError(
                "loaded execution contract is not resolved from an exact model instance".into(),
            ));
        }
        if self.execution_profile.model_variant != Some(self.metadata.model_variant) {
            return Err(Error::ModelLoadError(
                "loaded execution contract model identity does not match adapter metadata".into(),
            ));
        }

        let declared = self.execution_profile.effective_physical_launch_policy();
        let supported = launch_policy_without_concurrency_evidence();
        if declared != supported {
            return Err(Error::ModelLoadError(format!(
                "loaded model {} capability {:?} declared unsupported physical launch policy {declared:?}; no production concurrency evidence is available, so the supported policy is {supported:?}",
                self.metadata.model_variant, self.metadata.capability,
            )));
        }
        if self
            .stages
            .iter()
            .any(|stage| stage.physical_launch_policy != declared)
        {
            return Err(Error::ModelLoadError(
                "loaded execution stage launch policy does not match its sealed profile".into(),
            ));
        }
        if matches!(declared, PhysicalLaunchPolicy::Concurrent { .. })
            && (self.execution_profile.concurrency != ConcurrencyClass::Batchable
                || self.stages.iter().any(|stage| {
                    stage.batch_mode != NativeBatchMode::None
                        || stage.concurrency != ConcurrencyClass::Batchable
                        || stage.shape_policy != StageShapePolicy::Independent
                }))
        {
            return Err(Error::ModelLoadError(
                "concurrent physical launches require independently shaped scalar rows".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn adapter_binding(&self) -> Result<ExecutionAdapterBinding> {
        self.validate_physical_launch_policy()?;
        let binding = ExecutionAdapterBinding {
            execution_group_id: self.execution_group_id,
            model_instance_id: self.model_instance_id,
            adapter_instance_id: self.adapter_instance_id,
            adapter_abi_revision: self.adapter_abi_revision,
            model_variant: self.metadata.model_variant,
            capability_id: self.metadata.capability.as_str().to_string(),
            stages: self.stages.clone(),
        };
        binding.validate()?;
        Ok(binding)
    }
}

pub(crate) trait LoadedExecutionAdapter: fmt::Debug + Send + Sync {
    fn metadata(&self) -> AdapterMetadata;
    fn adapter_instance_id(&self) -> AdapterInstanceId;
    fn adapter_abi_revision(&self) -> AdapterAbiRevision;
    fn contract(&self, streaming: StreamingRequirements) -> Result<LoadedExecutionContract>;
}

/// Loaded-state publication normalized into an immutable ABI-v2 runtime before
/// the model becomes ready.
#[derive(Debug, Clone)]
pub(crate) enum LoadedStatePublication {
    V2(CapabilityStateDescriptorV2),
    ManagedV2 {
        contract: InferenceStateContract,
        physical: Arc<ManagedKvModelRuntime>,
    },
    /// Fully authored retained + typed invocation state with all physical
    /// backing allocated before capability sealing.
    PhysicalV2 {
        descriptor: CapabilityStateDescriptorV2,
        retained: Option<RetainedStateRuntimeV2>,
        /// Exact stage-graph activation is declared independently from the
        /// KV-specific execution profile fields.
        retained_uses: HashMap<[u8; 32], RetainedStateUseV2>,
        invocation_workspace: InvocationWorkspaceRuntimeV2,
    },
}

impl LoadedStatePublication {
    fn validate(&self, stages: &[StageDescriptor]) -> Result<()> {
        match self {
            Self::V2(descriptor) => descriptor.validate_against_stages(stages),
            Self::ManagedV2 { contract, physical } => {
                contract.validate()?;
                if contract.fingerprint()? != physical.state_plan_v2().contract_fingerprint {
                    return Err(Error::ModelLoadError(
                        "managed v2 publication does not match its physical state plan".to_string(),
                    ));
                }
                Ok(())
            }
            Self::PhysicalV2 {
                descriptor,
                retained,
                ..
            } => {
                descriptor.validate_against_stages(stages)?;
                match (&descriptor.retained, retained) {
                    (RetainedStateCapability::Stateless, None) => {}
                    (RetainedStateCapability::Managed { contract }, Some(retained))
                        if contract.fingerprint()?
                            == retained.state_plan_v2().contract_fingerprint => {}
                    (RetainedStateCapability::Managed { .. }, None) => {
                        return Err(Error::ModelLoadError(
                            "physical state publication is missing its retained backing".into(),
                        ));
                    }
                    (RetainedStateCapability::Stateless, Some(_)) => {
                        return Err(Error::ModelLoadError(
                            "invocation-only publication unexpectedly owns retained backing".into(),
                        ));
                    }
                    (RetainedStateCapability::Managed { .. }, Some(_)) => {
                        return Err(Error::ModelLoadError(
                            "physical state publication does not match its retained plan".into(),
                        ));
                    }
                }
                Ok(())
            }
        }
    }
}

/// One sealed capability declaration for an exact loaded model instance.
///
/// Execution remains request-resolved because streaming requirements can
/// select a different stage contract. Cache truth is immutable for the loaded
/// capability and can no longer be overlaid after adapter selection.
#[derive(Debug, Clone)]
pub(crate) struct LoadedCapabilityDescriptor {
    execution: Arc<dyn LoadedExecutionAdapter>,
    state: LoadedStatePublication,
    v2_runtimes: HashMap<[u8; 32], Arc<CapabilityStateRuntimeV2>>,
}

fn loaded_execution_contracts(
    execution: &dyn LoadedExecutionAdapter,
) -> Result<Vec<LoadedExecutionContract>> {
    let metadata = execution.metadata();
    let mut requirements = vec![
        StreamingRequirements::NONE,
        StreamingRequirements::transport_only(),
    ];
    if metadata.streaming_mode != StreamingMode::None {
        requirements.push(StreamingRequirements {
            transport_output: false,
            model_native: true,
        });
        requirements.push(StreamingRequirements::native(true));
    }
    let contracts = requirements
        .into_iter()
        .map(|requirements| {
            let contract = execution.contract(requirements)?;
            contract.validate_physical_launch_policy()?;
            Ok(contract)
        })
        .collect::<Result<Vec<_>>>()?;
    let launch_policy = contracts
        .first()
        .map(|contract| {
            contract
                .execution_profile
                .effective_physical_launch_policy()
        })
        .ok_or_else(|| Error::ModelLoadError("loaded adapter produced no contracts".into()))?;
    if contracts.iter().any(|contract| {
        contract
            .execution_profile
            .effective_physical_launch_policy()
            != launch_policy
    }) {
        return Err(Error::ModelLoadError(
            "one loaded adapter instance produced inconsistent launch policies".into(),
        ));
    }
    Ok(contracts)
}

impl LoadedCapabilityDescriptor {
    fn new(
        execution: Arc<dyn LoadedExecutionAdapter>,
        state: Option<LoadedStatePublication>,
        backend_kind: BackendKind,
    ) -> Result<Self> {
        let contracts = loaded_execution_contracts(execution.as_ref())?;
        for contract in &contracts {
            if contract.execution_profile.backend != backend_kind {
                return Err(Error::ModelLoadError(
                    "state ABI v2 execution contract does not match the authoritative loaded backend"
                        .to_string(),
                ));
            }
        }
        let state = match state {
            Some(state) => state,
            None if execution.metadata().state_requirement
                == InferenceStateRequirement::Stateless
                && contracts.iter().all(|contract| {
                    contract.execution_profile.cache_mode == CacheMode::None
                        && contract
                            .stages
                            .iter()
                            .all(|stage| stage.max_workspace_bytes == 0)
                }) =>
            {
                let stage_graphs = contracts
                    .iter()
                    .map(|contract| contract.stages.as_ref())
                    .collect::<Vec<_>>();
                LoadedStatePublication::V2(CapabilityStateDescriptorV2::stateless_for_stage_graphs(
                    &stage_graphs,
                )?)
            }
            None => {
                return Err(Error::ModelLoadError(format!(
                    "loaded model {} capability {:?} requires an explicit load-sealed ABI-v2 state publication",
                    execution.metadata().model_variant,
                    execution.metadata().capability,
                )));
            }
        };
        let mut v2_runtimes = HashMap::new();
        let state = match state {
            LoadedStatePublication::V2(descriptor) => {
                if !descriptor.is_stateless() {
                    return Err(Error::ModelLoadError(
                        "managed state ABI v2 publication requires physical backing".to_string(),
                    ));
                }
                if execution.metadata().state_requirement.requires_retained() {
                    return Err(Error::ModelLoadError(
                        "capability requiring retained inference state cannot publish a stateless runtime"
                            .to_string(),
                    ));
                }
                for contract in &contracts {
                    if contract.execution_profile.cache_mode != CacheMode::None
                        || contract.execution_profile.cache_namespace.is_some()
                        || contract.execution_profile.kv_dtype != "none"
                    {
                        return Err(Error::ModelLoadError(
                        "stateless state ABI v2 contradicts execution that declares retained cache state"
                            .to_string(),
                        ));
                    }
                    let has_invocation =
                        !descriptor.has_zero_invocation_workspace_for(&contract.stages)?;
                    if has_invocation {
                        return Err(Error::ModelLoadError(
                            "state ABI v2 invocation workspace requires load-sealed physical backing"
                                .to_string(),
                        ));
                    }
                    if execution.metadata().state_requirement.requires_invocation() {
                        return Err(Error::ModelLoadError(
                            "capability requiring invocation state cannot publish a zero-workspace runtime"
                                .to_string(),
                        ));
                    }
                    let binding = contract.adapter_binding()?;
                    let stateless = StatelessCapabilityRuntimeV2::seal(
                        backend_kind,
                        &binding,
                        descriptor.clone(),
                    )?;
                    let graph = stateless.stage_graph_fingerprint;
                    let runtime = Arc::new(CapabilityStateRuntimeV2::stateless(stateless));
                    match v2_runtimes.entry(graph) {
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            entry.insert(runtime);
                        }
                        std::collections::hash_map::Entry::Occupied(entry) => {
                            if entry.get().as_ref() != runtime.as_ref() {
                                return Err(Error::ModelLoadError(
                                "one state ABI v2 stage graph resolved to inconsistent runtime identities"
                                    .to_string(),
                            ));
                            }
                        }
                    }
                }
                LoadedStatePublication::V2(descriptor)
            }
            LoadedStatePublication::ManagedV2 { contract, physical } => {
                if !execution.metadata().state_requirement.requires_retained() {
                    return Err(Error::ModelLoadError(
                        "capability declared without retained state published a retained physical runtime"
                            .to_string(),
                    ));
                }
                if execution.metadata().state_requirement.requires_invocation() {
                    return Err(Error::ModelLoadError(
                        "capability requiring invocation state cannot publish a retained-only physical runtime"
                            .to_string(),
                    ));
                }
                let stage_graphs = contracts
                    .iter()
                    .map(|contract| contract.stages.as_ref())
                    .collect::<Vec<_>>();
                let descriptor =
                    CapabilityStateDescriptorV2::managed_for_stage_graphs(contract, &stage_graphs)?;
                for contract in &contracts {
                    let retained_state_use = match contract.execution_profile.cache_mode {
                        CacheMode::ExternalPaged
                            if contract.execution_profile.cache_namespace.is_some()
                                && contract.execution_profile.kv_dtype != "none" =>
                        {
                            RetainedStateUseV2::ExternalPaged
                        }
                        CacheMode::None
                            if contract.execution_profile.cache_namespace.is_none()
                                && contract.execution_profile.kv_dtype == "none" =>
                        {
                            RetainedStateUseV2::Inactive
                        }
                        _ => {
                            return Err(Error::ModelLoadError(
                                "managed state ABI v2 requires each graph to declare either external paged state or no retained state"
                                    .to_string(),
                            ));
                        }
                    };
                    let binding = contract.adapter_binding()?;
                    let managed = ManagedCapabilityRuntimeV2::seal(
                        backend_kind,
                        &binding,
                        descriptor.clone(),
                        physical.clone(),
                        retained_state_use,
                    )?;
                    let graph = managed.stage_graph_fingerprint;
                    let runtime = Arc::new(CapabilityStateRuntimeV2::managed(managed));
                    match v2_runtimes.entry(graph) {
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            entry.insert(runtime);
                        }
                        std::collections::hash_map::Entry::Occupied(entry) => {
                            if entry.get().as_ref() != runtime.as_ref() {
                                return Err(Error::ModelLoadError(
                                    "one managed state ABI v2 graph resolved inconsistent runtime identities"
                                        .to_string(),
                                ));
                            }
                        }
                    }
                }
                LoadedStatePublication::V2(descriptor)
            }
            LoadedStatePublication::PhysicalV2 {
                descriptor,
                retained,
                retained_uses,
                invocation_workspace,
            } => {
                if execution.metadata().state_requirement.requires_retained() != retained.is_some()
                {
                    return Err(Error::ModelLoadError(
                        "physical retained backing does not match the capability lifetime declaration"
                            .to_string(),
                    ));
                }
                let expected_graphs = contracts
                    .iter()
                    .map(|contract| stage_graph_fingerprint(&contract.stages))
                    .collect::<Result<HashSet<_>>>()?;
                let declared_graphs = retained_uses.keys().copied().collect::<HashSet<_>>();
                if retained.is_some() {
                    if declared_graphs != expected_graphs {
                        return Err(Error::ModelLoadError(
                            "physical retained-state use must be declared for every exact stage graph"
                                .to_string(),
                        ));
                    }
                } else if !declared_graphs.is_empty() {
                    return Err(Error::ModelLoadError(
                        "invocation-only physical state cannot declare retained-state use"
                            .to_string(),
                    ));
                }
                let capability_has_invocation =
                    contracts
                        .iter()
                        .try_fold(false, |has_invocation, contract| {
                            Ok::<_, Error>(
                                has_invocation
                                    || !descriptor
                                        .has_zero_invocation_workspace_for(&contract.stages)?,
                            )
                        })?;
                if execution.metadata().state_requirement.requires_invocation()
                    != capability_has_invocation
                {
                    return Err(Error::ModelLoadError(
                        "physical invocation workspace does not match the capability lifetime declaration"
                            .to_string(),
                    ));
                }
                for contract in &contracts {
                    let binding = contract.adapter_binding()?;
                    let (graph, runtime) = if let Some(retained) = retained.as_ref() {
                        let graph = stage_graph_fingerprint(&contract.stages)?;
                        let retained_state_use =
                            retained_uses.get(&graph).copied().ok_or_else(|| {
                                Error::ModelLoadError(
                                    "physical retained-state use is missing an exact stage graph"
                                        .to_string(),
                                )
                            })?;
                        validate_retained_state_use(
                            retained,
                            retained_state_use,
                            &contract.execution_profile,
                        )?;
                        let managed = ManagedCapabilityRuntimeV2::seal_with_invocation_workspace(
                            backend_kind,
                            &binding,
                            descriptor.clone(),
                            retained.clone(),
                            retained_state_use,
                            invocation_workspace.clone(),
                        )?;
                        (
                            managed.stage_graph_fingerprint,
                            Arc::new(CapabilityStateRuntimeV2::managed(managed)),
                        )
                    } else {
                        if contract.execution_profile.cache_mode != CacheMode::None
                            || contract.execution_profile.cache_namespace.is_some()
                            || contract.execution_profile.kv_dtype != "none"
                        {
                            return Err(Error::ModelLoadError(
                                "invocation-only state ABI v2 graph declared retained cache state"
                                    .to_string(),
                            ));
                        }
                        let invocation =
                            InvocationCapabilityRuntimeV2::seal_with_invocation_workspace(
                                backend_kind,
                                &binding,
                                descriptor.clone(),
                                invocation_workspace.clone(),
                            )?;
                        (
                            invocation.stage_graph_fingerprint,
                            Arc::new(CapabilityStateRuntimeV2::invocation(invocation)),
                        )
                    };
                    match v2_runtimes.entry(graph) {
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            entry.insert(runtime);
                        }
                        std::collections::hash_map::Entry::Occupied(entry) => {
                            if entry.get().as_ref() != runtime.as_ref() {
                                return Err(Error::ModelLoadError(
                                    "one physical state ABI v2 graph resolved inconsistent runtime identities"
                                        .to_string(),
                                ));
                            }
                        }
                    }
                }
                LoadedStatePublication::V2(descriptor)
            }
        };
        Ok(Self {
            execution,
            state,
            v2_runtimes,
        })
    }

    fn contract(&self, streaming: StreamingRequirements) -> Result<LoadedExecutionContract> {
        let contract = self.execution.contract(streaming)?;
        contract.validate_physical_launch_policy()?;
        Ok(contract)
    }

    fn binding(&self, streaming: StreamingRequirements) -> Result<LoadedCapabilityBinding> {
        let contract = self.contract(streaming)?;
        self.state.validate(&contract.stages)?;
        let execution = contract.adapter_binding()?;
        let state = match &self.state {
            LoadedStatePublication::V2(_) => {
                let graph = stage_graph_fingerprint(&contract.stages)?;
                let runtime = self.v2_runtimes.get(&graph).ok_or_else(|| {
                    Error::InferenceError(
                        "selected execution graph has no load-sealed state ABI v2 runtime"
                            .to_string(),
                    )
                })?;
                runtime.validate_against(contract.execution_profile.backend, &execution)?;
                runtime.clone()
            }
            LoadedStatePublication::ManagedV2 { .. }
            | LoadedStatePublication::PhysicalV2 { .. } => {
                return Err(Error::InferenceError(
                    "managed state publication was not load-sealed".to_string(),
                ));
            }
        };
        Ok(LoadedCapabilityBinding { execution, state })
    }
}

fn validate_retained_state_use(
    retained: &RetainedStateRuntimeV2,
    retained_state_use: RetainedStateUseV2,
    profile: &ExecutionProfile,
) -> Result<()> {
    let cacheless = profile.cache_mode == CacheMode::None
        && profile.cache_namespace.is_none()
        && profile.kv_dtype == "none";
    let external_paged = profile.cache_mode == CacheMode::ExternalPaged
        && profile.cache_namespace.is_some()
        && profile.kv_dtype != "none";
    let valid = if retained.is_tensor_only() {
        cacheless
            && matches!(
                retained_state_use,
                RetainedStateUseV2::ExternalTensor | RetainedStateUseV2::Inactive
            )
    } else {
        matches!(
            retained_state_use,
            RetainedStateUseV2::ExternalPaged if external_paged
        ) || matches!(retained_state_use, RetainedStateUseV2::Inactive if cacheless)
    };
    if !valid {
        return Err(Error::ModelLoadError(
            "retained-state use does not match its physical backing and exact execution profile"
                .to_string(),
        ));
    }
    Ok(())
}

/// Request-ready projection of one sealed loaded capability descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LoadedCapabilityBinding {
    pub(crate) execution: ExecutionAdapterBinding,
    pub(crate) state: Arc<CapabilityStateRuntimeV2>,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct LoadedAdapterFactoryContext {
    execution_group_id: ExecutionGroupId,
    model_instance_id: ModelInstanceId,
    backend_kind: BackendKind,
    max_tensor_batch_size: usize,
    request_parallelism: usize,
}

pub(super) trait LoadedExecutionAdapterFactory: fmt::Debug + Send + Sync {
    fn id(&self) -> &'static str;
    fn batch_mode(&self) -> NativeBatchMode;
    fn supports(&self, metadata: AdapterMetadata, backend_kind: BackendKind) -> bool;
    fn create(
        &self,
        context: LoadedAdapterFactoryContext,
        metadata: AdapterMetadata,
    ) -> Result<Arc<dyn LoadedExecutionAdapter>>;
}

fn is_physical_qwen_tts(metadata: AdapterMetadata) -> bool {
    matches!(
        metadata.capability,
        CapabilityKind::Tts | CapabilityKind::StreamingTts
    ) && metadata.model_variant.family() == crate::catalog::ModelFamily::Qwen3Tts
}

fn is_nemotron_realtime(metadata: AdapterMetadata) -> bool {
    metadata.capability == CapabilityKind::RealtimeAsr
        && metadata.model_variant.family() == crate::catalog::ModelFamily::NemotronAsr
}

fn is_continuous_physical_chat(metadata: AdapterMetadata) -> bool {
    metadata.capability == CapabilityKind::Chat
        && matches!(
            metadata.model_variant.family(),
            crate::catalog::ModelFamily::Qwen3Chat
                | crate::catalog::ModelFamily::Qwen35Chat
                | crate::catalog::ModelFamily::Gemma3Chat
                | crate::catalog::ModelFamily::Qwen38Chat
                | crate::catalog::ModelFamily::Lfm2Chat
        )
}

#[derive(Debug, Clone, Copy)]
struct PhysicalQwenTtsAdapterFactory;

impl LoadedExecutionAdapterFactory for PhysicalQwenTtsAdapterFactory {
    fn id(&self) -> &'static str {
        "builtin.qwen_tts.physical_sequence"
    }

    fn batch_mode(&self) -> NativeBatchMode {
        NativeBatchMode::None
    }

    fn supports(&self, metadata: AdapterMetadata, _backend_kind: BackendKind) -> bool {
        is_physical_qwen_tts(metadata)
    }

    fn create(
        &self,
        context: LoadedAdapterFactoryContext,
        metadata: AdapterMetadata,
    ) -> Result<Arc<dyn LoadedExecutionAdapter>> {
        Ok(Arc::new(PhysicalQwenTtsExecutionAdapter::new(
            context.execution_group_id,
            context.model_instance_id,
            metadata,
            context.backend_kind,
        )))
    }
}

#[derive(Debug, Clone, Copy)]
struct NemotronRealtimeAdapterFactory;

impl LoadedExecutionAdapterFactory for NemotronRealtimeAdapterFactory {
    fn id(&self) -> &'static str {
        "builtin.nemotron_realtime.physical_tensor"
    }

    fn batch_mode(&self) -> NativeBatchMode {
        NativeBatchMode::None
    }

    fn supports(&self, metadata: AdapterMetadata, _backend_kind: BackendKind) -> bool {
        is_nemotron_realtime(metadata)
    }

    fn create(
        &self,
        context: LoadedAdapterFactoryContext,
        metadata: AdapterMetadata,
    ) -> Result<Arc<dyn LoadedExecutionAdapter>> {
        Ok(Arc::new(NemotronRealtimeExecutionAdapter::new(
            context.execution_group_id,
            context.model_instance_id,
            metadata,
            context.backend_kind,
        )))
    }
}

#[derive(Debug, Clone, Copy)]
struct ContinuousPhysicalChatAdapterFactory;

impl LoadedExecutionAdapterFactory for ContinuousPhysicalChatAdapterFactory {
    fn id(&self) -> &'static str {
        "builtin.physical_chat.tensor_continuous"
    }

    fn batch_mode(&self) -> NativeBatchMode {
        NativeBatchMode::Continuous
    }

    fn supports(&self, metadata: AdapterMetadata, _backend_kind: BackendKind) -> bool {
        is_continuous_physical_chat(metadata)
    }

    fn create(
        &self,
        context: LoadedAdapterFactoryContext,
        metadata: AdapterMetadata,
    ) -> Result<Arc<dyn LoadedExecutionAdapter>> {
        Ok(Arc::new(ContinuousChatExecutionAdapter::new(
            context.execution_group_id,
            context.model_instance_id,
            metadata,
            context.backend_kind,
            context.max_tensor_batch_size,
            context.request_parallelism,
        )))
    }
}

#[derive(Debug, Clone, Copy)]
struct ScalarExecutionAdapterFactory;

impl LoadedExecutionAdapterFactory for ScalarExecutionAdapterFactory {
    fn id(&self) -> &'static str {
        "builtin.scalar"
    }

    fn batch_mode(&self) -> NativeBatchMode {
        NativeBatchMode::None
    }

    fn supports(&self, metadata: AdapterMetadata, _backend_kind: BackendKind) -> bool {
        !is_physical_qwen_tts(metadata)
            && !is_nemotron_realtime(metadata)
            && !is_continuous_physical_chat(metadata)
    }

    fn create(
        &self,
        context: LoadedAdapterFactoryContext,
        metadata: AdapterMetadata,
    ) -> Result<Arc<dyn LoadedExecutionAdapter>> {
        Ok(Arc::new(ScalarExecutionAdapter::new(
            context.execution_group_id,
            context.model_instance_id,
            metadata,
            context.backend_kind,
            context.request_parallelism,
        )))
    }
}

pub(super) fn built_in_loaded_adapter_factories() -> Vec<Arc<dyn LoadedExecutionAdapterFactory>> {
    vec![
        Arc::new(PhysicalQwenTtsAdapterFactory),
        Arc::new(NemotronRealtimeAdapterFactory),
        Arc::new(ContinuousPhysicalChatAdapterFactory),
        Arc::new(ScalarExecutionAdapterFactory),
    ]
}

#[derive(Debug)]
struct ScalarExecutionAdapter {
    execution_group_id: ExecutionGroupId,
    model_instance_id: ModelInstanceId,
    adapter_instance_id: AdapterInstanceId,
    metadata: AdapterMetadata,
    backend_kind: BackendKind,
    request_parallelism: usize,
}

impl ScalarExecutionAdapter {
    fn new(
        execution_group_id: ExecutionGroupId,
        model_instance_id: ModelInstanceId,
        metadata: AdapterMetadata,
        backend_kind: BackendKind,
        request_parallelism: usize,
    ) -> Self {
        Self {
            execution_group_id,
            model_instance_id,
            adapter_instance_id: AdapterInstanceId::new(
                NEXT_ADAPTER_INSTANCE_ID.fetch_add(1, Ordering::Relaxed),
            ),
            metadata,
            backend_kind,
            request_parallelism: scalar_request_parallelism(backend_kind, request_parallelism),
        }
    }
}

impl LoadedExecutionAdapter for ScalarExecutionAdapter {
    fn metadata(&self) -> AdapterMetadata {
        self.metadata
    }

    fn adapter_instance_id(&self) -> AdapterInstanceId {
        self.adapter_instance_id
    }

    fn adapter_abi_revision(&self) -> AdapterAbiRevision {
        SCALAR_ADAPTER_ABI
    }

    fn contract(&self, streaming: StreamingRequirements) -> Result<LoadedExecutionContract> {
        scalar_contract(
            self.execution_group_id,
            self.model_instance_id,
            self.adapter_instance_id(),
            self.adapter_abi_revision(),
            self.metadata(),
            self.backend_kind,
            self.request_parallelism,
            streaming,
        )
    }
}

fn scalar_contract(
    execution_group_id: ExecutionGroupId,
    model_instance_id: ModelInstanceId,
    adapter_instance_id: AdapterInstanceId,
    adapter_abi_revision: AdapterAbiRevision,
    metadata: AdapterMetadata,
    backend_kind: BackendKind,
    _request_parallelism: usize,
    streaming: StreamingRequirements,
) -> Result<LoadedExecutionContract> {
    if streaming.model_native && metadata.streaming_mode == StreamingMode::None {
        return Err(Error::InvalidInput(format!(
            "Model {} supports {:?}, but not streaming execution for that capability",
            metadata.model_variant, metadata.capability
        )));
    }

    let mut execution_profile =
        scalar_execution_profile(metadata, backend_kind, streaming.model_native);
    if metadata.capability == CapabilityKind::Asr
        && metadata.model_variant.family() == crate::catalog::ModelFamily::Qwen3Asr
        && streaming.model_native
    {
        execution_profile.cache_mode = CacheMode::ExternalPaged;
        execution_profile.cache_namespace = Some(format!(
            "{}:{}:state-v2",
            metadata.model_variant,
            backend_kind.as_str()
        ));
        execution_profile.kv_dtype = "state_v2_resolved".to_string();
    }
    execution_profile.resolved_from_loaded_model = true;
    execution_profile.prefill_batch = NativeBatchMode::None;
    execution_profile.decode_batch = NativeBatchMode::None;
    let (row_width, physical_launch_policy) = scalar_row_policy_without_concurrency_evidence();
    execution_profile.max_batch_size = row_width;
    execution_profile.concurrency = if execution_profile.max_batch_size > 1 {
        ConcurrencyClass::Batchable
    } else {
        ConcurrencyClass::Exclusive
    };
    execution_profile.physical_launch_policy = physical_launch_policy;

    let mut stage = StageDescriptor::from_execution_profile(
        StageId::new(0),
        format!("{}.scalar", metadata.capability.as_str()),
        &execution_profile,
        NativeBatchMode::None,
    );
    stage.output_visibility = output_visibility_for(
        streaming.transport_output,
        execution_profile.mode,
        NativeBatchMode::None,
    );
    stage.validate()?;

    Ok(LoadedExecutionContract {
        execution_group_id,
        model_instance_id,
        adapter_instance_id,
        adapter_abi_revision,
        metadata,
        execution_profile,
        stages: Arc::from([stage]),
    })
}

#[derive(Debug)]
struct NemotronRealtimeExecutionAdapter {
    execution_group_id: ExecutionGroupId,
    model_instance_id: ModelInstanceId,
    adapter_instance_id: AdapterInstanceId,
    metadata: AdapterMetadata,
    backend_kind: BackendKind,
}

impl NemotronRealtimeExecutionAdapter {
    fn new(
        execution_group_id: ExecutionGroupId,
        model_instance_id: ModelInstanceId,
        metadata: AdapterMetadata,
        backend_kind: BackendKind,
    ) -> Self {
        Self {
            execution_group_id,
            model_instance_id,
            adapter_instance_id: AdapterInstanceId::new(
                NEXT_ADAPTER_INSTANCE_ID.fetch_add(1, Ordering::Relaxed),
            ),
            metadata,
            backend_kind,
        }
    }
}

impl LoadedExecutionAdapter for NemotronRealtimeExecutionAdapter {
    fn metadata(&self) -> AdapterMetadata {
        self.metadata
    }

    fn adapter_instance_id(&self) -> AdapterInstanceId {
        self.adapter_instance_id
    }

    fn adapter_abi_revision(&self) -> AdapterAbiRevision {
        NEMOTRON_REALTIME_ADAPTER_ABI
    }

    fn contract(&self, streaming: StreamingRequirements) -> Result<LoadedExecutionContract> {
        let metadata = self.metadata();
        let mut execution_profile =
            scalar_execution_profile(metadata, self.backend_kind, streaming.model_native);
        execution_profile.mode = ExecutionMode::Atomic;
        execution_profile.prefill = PrefillMode::Full;
        execution_profile.incremental_decode = false;
        execution_profile.prefill_batch = NativeBatchMode::None;
        execution_profile.decode_batch = NativeBatchMode::None;
        execution_profile.cache_mode = CacheMode::None;
        execution_profile.cache_namespace = None;
        execution_profile.kv_dtype = "none".into();
        execution_profile.cancellation = CancellationGranularity::RealtimeChunk;
        execution_profile.concurrency = ConcurrencyClass::Exclusive;
        execution_profile.recompute_safe = false;
        execution_profile.cache_release_safe = true;
        execution_profile.prefix_reuse_safe = false;
        execution_profile.max_batch_size = 1;
        execution_profile.resolved_from_loaded_model = true;

        let mut stage = StageDescriptor::from_execution_profile(
            StageId::new(1),
            "asr.realtime.physical_tensor",
            &execution_profile,
            NativeBatchMode::None,
        );
        stage.selector = StageWorkSelector::Atomic;
        stage.max_workspace_bytes =
            crate::models::architectures::nemotron::asr::NEMOTRON_REALTIME_STAGE_WORKSPACE_BYTES;
        stage.output_visibility = OutputVisibility::AfterQuantumCommit;
        stage.validate()?;

        Ok(LoadedExecutionContract {
            execution_group_id: self.execution_group_id,
            model_instance_id: self.model_instance_id,
            adapter_instance_id: self.adapter_instance_id(),
            adapter_abi_revision: self.adapter_abi_revision(),
            metadata,
            execution_profile,
            stages: Arc::from([stage]),
        })
    }
}

#[derive(Debug)]
struct PhysicalQwenTtsExecutionAdapter {
    execution_group_id: ExecutionGroupId,
    model_instance_id: ModelInstanceId,
    adapter_instance_id: AdapterInstanceId,
    metadata: AdapterMetadata,
    backend_kind: BackendKind,
}

impl PhysicalQwenTtsExecutionAdapter {
    fn new(
        execution_group_id: ExecutionGroupId,
        model_instance_id: ModelInstanceId,
        metadata: AdapterMetadata,
        backend_kind: BackendKind,
    ) -> Self {
        Self {
            execution_group_id,
            model_instance_id,
            adapter_instance_id: AdapterInstanceId::new(
                NEXT_ADAPTER_INSTANCE_ID.fetch_add(1, Ordering::Relaxed),
            ),
            metadata,
            backend_kind,
        }
    }
}

impl LoadedExecutionAdapter for PhysicalQwenTtsExecutionAdapter {
    fn metadata(&self) -> AdapterMetadata {
        self.metadata
    }

    fn adapter_instance_id(&self) -> AdapterInstanceId {
        self.adapter_instance_id
    }

    fn adapter_abi_revision(&self) -> AdapterAbiRevision {
        AdapterAbiRevision::new(3)
    }

    fn contract(&self, streaming: StreamingRequirements) -> Result<LoadedExecutionContract> {
        let metadata = self.metadata();
        if streaming.model_native && metadata.streaming_mode == StreamingMode::None {
            return Err(Error::InvalidInput(format!(
                "Model {} has no native streaming TTS contract",
                metadata.model_variant
            )));
        }
        let mut execution_profile =
            scalar_execution_profile(metadata, self.backend_kind, streaming.model_native);
        execution_profile.mode = ExecutionMode::Sequence;
        execution_profile.prefill = PrefillMode::Full;
        execution_profile.incremental_decode = true;
        execution_profile.prefill_batch = NativeBatchMode::None;
        execution_profile.decode_batch = NativeBatchMode::None;
        execution_profile.cache_mode = CacheMode::ExternalPaged;
        execution_profile.cache_namespace = Some(format!(
            "{}:{}:{}:state-v2",
            metadata.model_variant,
            metadata.capability.as_str(),
            self.backend_kind.as_str()
        ));
        execution_profile.kv_dtype = "state_v2_resolved".to_string();
        execution_profile.cancellation = CancellationGranularity::SequenceStep;
        execution_profile.concurrency = ConcurrencyClass::Exclusive;
        execution_profile.recompute_safe = false;
        execution_profile.cache_release_safe = true;
        execution_profile.prefix_reuse_safe = false;
        execution_profile.max_batch_size = 1;
        execution_profile.resolved_from_loaded_model = true;

        let mut prefill = StageDescriptor::from_execution_profile(
            StageId::new(1),
            "tts.prefill.physical",
            &execution_profile,
            NativeBatchMode::None,
        );
        prefill.selector = StageWorkSelector::SequencePrefill;
        prefill.max_workspace_bytes = 0;
        prefill.output_visibility = output_visibility_for(
            streaming.transport_output,
            execution_profile.mode,
            NativeBatchMode::None,
        );
        let mut decode = StageDescriptor::from_execution_profile(
            StageId::new(2),
            "tts.decode.physical",
            &execution_profile,
            NativeBatchMode::None,
        );
        decode.selector = StageWorkSelector::SequenceDecode;
        // Predictor KV is load-owned typed invocation state, not scheduler
        // scratch. Its physical pool is authorized and charged by lifecycle.
        decode.max_workspace_bytes = 0;
        decode.output_visibility = prefill.output_visibility;
        prefill.validate()?;
        decode.validate()?;
        Ok(LoadedExecutionContract {
            execution_group_id: self.execution_group_id,
            model_instance_id: self.model_instance_id,
            adapter_instance_id: self.adapter_instance_id(),
            adapter_abi_revision: self.adapter_abi_revision(),
            metadata,
            execution_profile,
            stages: Arc::from([prefill, decode]),
        })
    }
}

#[derive(Debug)]
struct StaticTtsExecutionAdapter {
    execution_group_id: ExecutionGroupId,
    model_instance_id: ModelInstanceId,
    adapter_instance_id: AdapterInstanceId,
    metadata: AdapterMetadata,
    backend_kind: BackendKind,
    max_batch_size: usize,
    request_parallelism: usize,
}

impl StaticTtsExecutionAdapter {
    fn new(
        execution_group_id: ExecutionGroupId,
        model_instance_id: ModelInstanceId,
        metadata: AdapterMetadata,
        backend_kind: BackendKind,
        max_batch_size: usize,
        request_parallelism: usize,
    ) -> Self {
        Self {
            execution_group_id,
            model_instance_id,
            adapter_instance_id: AdapterInstanceId::new(
                NEXT_ADAPTER_INSTANCE_ID.fetch_add(1, Ordering::Relaxed),
            ),
            metadata,
            backend_kind,
            max_batch_size: max_batch_size.max(1),
            request_parallelism: scalar_request_parallelism(backend_kind, request_parallelism),
        }
    }
}

impl LoadedExecutionAdapter for StaticTtsExecutionAdapter {
    fn metadata(&self) -> AdapterMetadata {
        self.metadata
    }

    fn adapter_instance_id(&self) -> AdapterInstanceId {
        self.adapter_instance_id
    }

    fn adapter_abi_revision(&self) -> AdapterAbiRevision {
        STATIC_TENSOR_ADAPTER_ABI
    }

    fn contract(&self, streaming: StreamingRequirements) -> Result<LoadedExecutionContract> {
        if streaming.model_native {
            return scalar_contract(
                self.execution_group_id,
                self.model_instance_id,
                self.adapter_instance_id(),
                self.adapter_abi_revision(),
                self.metadata(),
                self.backend_kind,
                self.request_parallelism,
                streaming,
            );
        }

        let metadata = self.metadata();
        let mut execution_profile = scalar_execution_profile(metadata, self.backend_kind, false);
        execution_profile.mode = ExecutionMode::Atomic;
        execution_profile.prefill = PrefillMode::None;
        execution_profile.incremental_decode = false;
        execution_profile.prefill_batch = NativeBatchMode::Static;
        execution_profile.decode_batch = NativeBatchMode::None;
        execution_profile.cache_mode = CacheMode::None;
        execution_profile.cancellation = CancellationGranularity::OperationBoundary;
        execution_profile.concurrency = ConcurrencyClass::Batchable;
        execution_profile.recompute_safe = false;
        execution_profile.cache_release_safe = false;
        execution_profile.prefix_reuse_safe = false;
        execution_profile.max_batch_size = self.max_batch_size;
        execution_profile.resolved_from_loaded_model = true;
        execution_profile.kv_dtype = "none".to_string();
        execution_profile.cache_namespace = None;

        let mut stage = StageDescriptor::from_execution_profile(
            StageId::new(1),
            "tts.generate.tensor_static",
            &execution_profile,
            NativeBatchMode::Static,
        );
        stage.selector = StageWorkSelector::Atomic;
        stage.shape_policy = crate::engine::StageShapePolicy::Exact;
        stage.max_padding_basis_points = 0;
        stage.max_work_units = u64::try_from(stage.max_batch_size).map_err(|_| {
            Error::Overloaded("static TTS batch width exceeds work accounting".to_string())
        })?;
        stage.max_workspace_bytes = STATIC_TTS_MAX_BATCH_WORKSPACE_BYTES;
        let mut scalar = StageDescriptor::from_execution_profile(
            StageId::new(0),
            "tts.generate.scalar",
            &execution_profile,
            NativeBatchMode::None,
        );
        scalar.selector = StageWorkSelector::Any;
        // The static tensor certificate applies to the single native B>1 call,
        // not to overlapping scalar fallbacks into the same loaded model.
        scalar.max_batch_size = 1;
        scalar.concurrency = ConcurrencyClass::Exclusive;
        scalar.shape_policy = crate::engine::StageShapePolicy::Exact;
        scalar.output_visibility = output_visibility_for(
            streaming.transport_output,
            execution_profile.mode,
            NativeBatchMode::None,
        );
        stage.validate()?;
        scalar.validate()?;

        Ok(LoadedExecutionContract {
            execution_group_id: self.execution_group_id,
            model_instance_id: self.model_instance_id,
            adapter_instance_id: self.adapter_instance_id(),
            adapter_abi_revision: self.adapter_abi_revision(),
            metadata,
            execution_profile,
            stages: Arc::from([stage, scalar]),
        })
    }
}

#[derive(Debug)]
struct ContinuousChatExecutionAdapter {
    execution_group_id: ExecutionGroupId,
    model_instance_id: ModelInstanceId,
    adapter_instance_id: AdapterInstanceId,
    metadata: AdapterMetadata,
    backend_kind: BackendKind,
    max_batch_size: usize,
}

impl ContinuousChatExecutionAdapter {
    fn new(
        execution_group_id: ExecutionGroupId,
        model_instance_id: ModelInstanceId,
        metadata: AdapterMetadata,
        backend_kind: BackendKind,
        max_batch_size: usize,
        _request_parallelism: usize,
    ) -> Self {
        Self {
            execution_group_id,
            model_instance_id,
            adapter_instance_id: AdapterInstanceId::new(
                NEXT_ADAPTER_INSTANCE_ID.fetch_add(1, Ordering::Relaxed),
            ),
            metadata,
            backend_kind,
            max_batch_size: max_batch_size.max(1),
        }
    }
}

impl LoadedExecutionAdapter for ContinuousChatExecutionAdapter {
    fn metadata(&self) -> AdapterMetadata {
        self.metadata
    }

    fn adapter_instance_id(&self) -> AdapterInstanceId {
        self.adapter_instance_id
    }

    fn adapter_abi_revision(&self) -> AdapterAbiRevision {
        CONTINUOUS_TENSOR_ADAPTER_ABI
    }

    fn contract(&self, streaming: StreamingRequirements) -> Result<LoadedExecutionContract> {
        let metadata = self.metadata();
        if streaming.model_native && metadata.streaming_mode == StreamingMode::None {
            return Err(Error::InvalidInput(format!(
                "Model {} has no streaming chat contract",
                metadata.model_variant
            )));
        }
        let mut execution_profile =
            scalar_execution_profile(metadata, self.backend_kind, streaming.model_native);
        execution_profile.prefill_batch = NativeBatchMode::None;
        execution_profile.decode_batch = NativeBatchMode::Continuous;
        execution_profile.cache_mode = CacheMode::ExternalPaged;
        execution_profile.cache_namespace = Some(format!(
            "{}:{}:state-v2",
            metadata.model_variant,
            self.backend_kind.as_str()
        ));
        execution_profile.kv_dtype = "state_v2_resolved".to_string();
        execution_profile.concurrency = ConcurrencyClass::Batchable;
        execution_profile.max_batch_size = self.max_batch_size;
        execution_profile.resolved_from_loaded_model = true;

        let mut prefill = StageDescriptor::from_execution_profile(
            StageId::new(1),
            "chat.prefill.scalar",
            &execution_profile,
            NativeBatchMode::None,
        );
        prefill.selector = StageWorkSelector::SequencePrefill;
        // Continuous decode is one model-owned tensor call. Prefill remains a
        // scalar model entry and has no independent-row reentrancy certificate.
        prefill.max_batch_size = 1;
        prefill.concurrency = ConcurrencyClass::Exclusive;
        prefill.shape_policy = crate::engine::StageShapePolicy::Exact;
        prefill.output_visibility = output_visibility_for(
            streaming.transport_output,
            execution_profile.mode,
            NativeBatchMode::None,
        );
        let mut decode = StageDescriptor::from_execution_profile(
            StageId::new(2),
            "chat.decode.tensor_continuous",
            &execution_profile,
            NativeBatchMode::Continuous,
        );
        decode.selector = StageWorkSelector::SequenceDecode;
        // `max_work_units` is the aggregate budget for the whole physical
        // envelope. Shared continuous decode uses one token per row; an
        // isolated model-preferred quantum may use up to four target inputs so
        // Qwen3.8 MTP draft/verify remains reachable without queue pressure.
        decode.max_work_units = u64::try_from(decode.max_batch_size)
            .map_err(|_| {
                Error::Overloaded(
                    "continuous decode batch width exceeds work accounting".to_string(),
                )
            })?
            .max(CONTINUOUS_CHAT_MAX_DECODE_QUANTUM);
        decode.max_workspace_bytes = CONTINUOUS_CHAT_MAX_BATCH_WORKSPACE_BYTES;
        prefill.validate()?;
        decode.validate()?;

        Ok(LoadedExecutionContract {
            execution_group_id: self.execution_group_id,
            model_instance_id: self.model_instance_id,
            adapter_instance_id: self.adapter_instance_id(),
            adapter_abi_revision: self.adapter_abi_revision(),
            metadata,
            execution_profile,
            stages: Arc::from([prefill, decode]),
        })
    }
}

impl RuntimeAdapterRegistry {
    pub(super) fn loaded_adapter_factory(
        &self,
        metadata: AdapterMetadata,
        backend_kind: BackendKind,
    ) -> Result<&dyn LoadedExecutionAdapterFactory> {
        let mut matches = self
            .loaded_adapter_factories
            .iter()
            .filter(|factory| factory.supports(metadata, backend_kind));
        let Some(selected) = matches.next() else {
            return Err(Error::ModelLoadError(format!(
                "loaded model {} capability {:?} has no execution adapter factory for {backend_kind:?}",
                metadata.model_variant, metadata.capability,
            )));
        };
        if let Some(ambiguous) = matches.next() {
            return Err(Error::ModelLoadError(format!(
                "loaded model {} capability {:?} matches both execution adapter factories `{}` and `{}`",
                metadata.model_variant,
                metadata.capability,
                selected.id(),
                ambiguous.id(),
            )));
        }
        Ok(selected.as_ref())
    }

    pub(super) fn loaded_native_variants(
        &self,
        backend_kind: BackendKind,
        batch_mode: NativeBatchMode,
    ) -> std::collections::HashSet<ModelVariant> {
        ModelVariant::all()
            .iter()
            .copied()
            .filter(|variant| {
                self.capabilities_for(*variant).into_iter().any(|metadata| {
                    self.loaded_adapter_factory(metadata, backend_kind)
                        .expect("factory ambiguity is rejected when the registry is built")
                        .batch_mode()
                        == batch_mode
                })
            })
            .collect()
    }

    fn create_loaded_adapter(
        &self,
        execution_group_id: ExecutionGroupId,
        model_instance_id: ModelInstanceId,
        metadata: AdapterMetadata,
        backend_kind: BackendKind,
    ) -> Result<Arc<dyn LoadedExecutionAdapter>> {
        let context = LoadedAdapterFactoryContext {
            execution_group_id,
            model_instance_id,
            backend_kind,
            max_tensor_batch_size: self.max_tensor_batch_size(),
            request_parallelism: self.request_parallelism(),
        };
        let adapter = self
            .loaded_adapter_factory(metadata, backend_kind)?
            .create(context, metadata)?;
        if adapter.metadata() != metadata {
            return Err(Error::ModelLoadError(format!(
                "loaded adapter factory returned mismatched metadata for {} capability {:?}",
                metadata.model_variant, metadata.capability
            )));
        }
        Ok(adapter)
    }
}

/// One-shot execution identity built before physical state allocation.
/// Factories run exactly once; sealing consumes the draft so a state plan can
/// never bind to a different adapter instance or selectable stage graph.
pub(crate) struct LoadedModelBundleDraft {
    execution_group_id: ExecutionGroupId,
    model_instance_id: ModelInstanceId,
    model_variant: ModelVariant,
    backend_kind: BackendKind,
    capabilities: HashMap<CapabilityKind, Arc<dyn LoadedExecutionAdapter>>,
}

impl fmt::Debug for LoadedModelBundleDraft {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LoadedModelBundleDraft")
            .field("execution_group_id", &self.execution_group_id)
            .field("model_instance_id", &self.model_instance_id)
            .field("model_variant", &self.model_variant)
            .field("backend_kind", &self.backend_kind)
            .field("capability_count", &self.capabilities.len())
            .finish()
    }
}

impl LoadedModelBundleDraft {
    pub(crate) fn build(
        registry: &RuntimeAdapterRegistry,
        execution_group_id: ExecutionGroupId,
        model_instance_id: ModelInstanceId,
        model_variant: ModelVariant,
        backend_kind: BackendKind,
    ) -> Result<Self> {
        let metadata = registry.capabilities_for(model_variant);
        if metadata.is_empty() {
            return Err(Error::ModelLoadError(format!(
                "loaded model {model_variant} has no executable capability adapter"
            )));
        }
        let mut capabilities = HashMap::with_capacity(metadata.len());
        for metadata in metadata {
            let adapter = registry.create_loaded_adapter(
                execution_group_id,
                model_instance_id,
                metadata,
                backend_kind,
            )?;
            if capabilities.insert(metadata.capability, adapter).is_some() {
                return Err(Error::ModelLoadError(format!(
                    "loaded model {model_variant} has duplicate {:?} adapters",
                    metadata.capability
                )));
            }
        }
        Ok(Self {
            execution_group_id,
            model_instance_id,
            model_variant,
            backend_kind,
            capabilities,
        })
    }

    pub(crate) fn execution_contracts(
        &self,
        capability: CapabilityKind,
    ) -> Result<Vec<LoadedExecutionContract>> {
        let execution = self.capabilities.get(&capability).ok_or_else(|| {
            Error::InvalidInput(format!(
                "loaded model {} does not expose capability {:?}",
                self.model_variant, capability
            ))
        })?;
        loaded_execution_contracts(execution.as_ref())
    }

    pub(crate) fn seal(
        self,
        mut state_publications: HashMap<CapabilityKind, LoadedStatePublication>,
    ) -> Result<LoadedModelBundle> {
        let mut unmatched = state_publications
            .keys()
            .copied()
            .filter(|capability| !self.capabilities.contains_key(capability))
            .map(CapabilityKind::as_str)
            .collect::<Vec<_>>();
        if !unmatched.is_empty() {
            unmatched.sort_unstable();
            return Err(Error::ModelLoadError(format!(
                "loaded model {} published cache truth for unregistered capabilities: {}",
                self.model_variant,
                unmatched.join(", ")
            )));
        }

        let mut capabilities = HashMap::with_capacity(self.capabilities.len());
        for (capability, execution) in self.capabilities {
            let state = state_publications.remove(&capability);
            let descriptor = LoadedCapabilityDescriptor::new(execution, state, self.backend_kind)?;
            capabilities.insert(capability, descriptor);
        }
        debug_assert!(state_publications.is_empty());
        Ok(LoadedModelBundle {
            execution_group_id: self.execution_group_id,
            model_instance_id: self.model_instance_id,
            model_variant: self.model_variant,
            backend_kind: self.backend_kind,
            capabilities,
        })
    }
}

pub(crate) struct LoadedModelBundle {
    execution_group_id: ExecutionGroupId,
    model_instance_id: ModelInstanceId,
    model_variant: ModelVariant,
    backend_kind: BackendKind,
    capabilities: HashMap<CapabilityKind, LoadedCapabilityDescriptor>,
}

impl fmt::Debug for LoadedModelBundle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LoadedModelBundle")
            .field("execution_group_id", &self.execution_group_id)
            .field("model_instance_id", &self.model_instance_id)
            .field("model_variant", &self.model_variant)
            .field("backend_kind", &self.backend_kind)
            .field("capability_count", &self.capabilities.len())
            .finish()
    }
}

impl LoadedModelBundle {
    pub(crate) fn bind(
        registry: &RuntimeAdapterRegistry,
        execution_group_id: ExecutionGroupId,
        model_instance_id: ModelInstanceId,
        model_variant: ModelVariant,
        backend_kind: BackendKind,
    ) -> Result<Self> {
        Self::bind_with_state_publications(
            registry,
            execution_group_id,
            model_instance_id,
            model_variant,
            backend_kind,
            HashMap::new(),
        )
    }

    /// Bind adapter metadata and exact loaded-model state truth into one sealed
    /// descriptor per capability. Only capabilities explicitly classified as
    /// stateless may omit a physical state publication.
    pub(crate) fn bind_with_state_publications(
        registry: &RuntimeAdapterRegistry,
        execution_group_id: ExecutionGroupId,
        model_instance_id: ModelInstanceId,
        model_variant: ModelVariant,
        backend_kind: BackendKind,
        state_publications: HashMap<CapabilityKind, LoadedStatePublication>,
    ) -> Result<Self> {
        LoadedModelBundleDraft::build(
            registry,
            execution_group_id,
            model_instance_id,
            model_variant,
            backend_kind,
        )?
        .seal(state_publications)
    }

    pub(crate) fn execution_group_id(&self) -> ExecutionGroupId {
        self.execution_group_id
    }

    pub(crate) fn model_instance_id(&self) -> ModelInstanceId {
        self.model_instance_id
    }

    pub(crate) fn model_variant(&self) -> ModelVariant {
        self.model_variant
    }

    pub(crate) fn backend_kind(&self) -> BackendKind {
        self.backend_kind
    }

    pub(crate) fn adapter_count(&self) -> usize {
        self.capabilities.len()
    }

    fn require_capability(
        &self,
        capability: CapabilityKind,
    ) -> Result<&LoadedCapabilityDescriptor> {
        self.capabilities.get(&capability).ok_or_else(|| {
            Error::InvalidInput(format!(
                "loaded model {} does not expose capability {:?}",
                self.model_variant, capability
            ))
        })
    }

    pub(crate) fn contract(
        &self,
        capability: CapabilityKind,
        streaming_required: bool,
    ) -> Result<LoadedExecutionContract> {
        self.contract_for_streaming(
            capability,
            StreamingRequirements::native(streaming_required),
        )
    }

    pub(crate) fn contract_for_streaming(
        &self,
        capability: CapabilityKind,
        streaming: StreamingRequirements,
    ) -> Result<LoadedExecutionContract> {
        self.require_capability(capability)?.contract(streaming)
    }

    pub(crate) fn capability_binding_for_streaming(
        &self,
        capability: CapabilityKind,
        streaming: StreamingRequirements,
    ) -> Result<LoadedCapabilityBinding> {
        self.require_capability(capability)?.binding(streaming)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{EngineCore, EngineCoreConfig};
    use crate::kv::InferenceStateCapability;
    use crate::runtime::adapters::ExecutionTargetKind;
    use crate::runtime::adapters::SequenceExecutionMode;

    fn chat_adapter_metadata(variant: ModelVariant) -> AdapterMetadata {
        AdapterMetadata {
            id: "test.chat.adapter",
            capability: CapabilityKind::Chat,
            model_variant: variant,
            streaming_mode: StreamingMode::Chunked,
            execution_target: ExecutionTargetKind::TokenEngine,
            sequence_execution: SequenceExecutionMode::StreamingOnly,
            state_requirement: InferenceStateRequirement::Retained,
        }
    }

    #[test]
    fn stateful_chat_models_with_batch_paths_select_the_continuous_stage() {
        let continuous = ContinuousPhysicalChatAdapterFactory;
        let scalar = ScalarExecutionAdapterFactory;

        for variant in [
            ModelVariant::Qwen3827BFp8,
            ModelVariant::Qwen3508BGguf,
            ModelVariant::Lfm2512BInstructGguf,
        ] {
            let metadata = chat_adapter_metadata(variant);
            for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
                assert!(
                    continuous.supports(metadata, backend),
                    "{variant}/{backend:?}"
                );
                assert!(!scalar.supports(metadata, backend), "{variant}/{backend:?}");
            }
        }
    }

    #[test]
    fn managed_qwen_publication_seals_a_physical_v2_runtime() {
        let registry = RuntimeAdapterRegistry::built_in();
        let model_instance = ModelInstanceId::new(8);
        let state_contract = crate::kv::test_contract();
        let capability = InferenceStateCapability::Managed(state_contract.clone());
        let mut core = EngineCore::new(EngineCoreConfig {
            backend: BackendKind::Cpu,
            max_blocks: 4,
            block_size: 32,
            ..EngineCoreConfig::default()
        })
        .unwrap();
        let physical = core
            .load_managed_model_cache(model_instance, &capability, None)
            .unwrap()
            .expect("physical managed runtime");
        let bundle = LoadedModelBundle::bind_with_state_publications(
            &registry,
            ExecutionGroupId::new(3),
            model_instance,
            ModelVariant::Qwen306B,
            BackendKind::Cpu,
            HashMap::from([(
                CapabilityKind::Chat,
                LoadedStatePublication::ManagedV2 {
                    contract: state_contract,
                    physical: physical.clone(),
                },
            )]),
        )
        .unwrap();

        let binding = bundle
            .capability_binding_for_streaming(CapabilityKind::Chat, StreamingRequirements::NONE)
            .unwrap();
        assert_eq!(binding.execution.model_instance_id, model_instance);
        let runtime = binding.state;
        assert!(runtime.managed_kv_runtime().is_some());
        let inactive = CapabilityStateRuntimeV2::managed(
            ManagedCapabilityRuntimeV2::seal(
                BackendKind::Cpu,
                &binding.execution,
                runtime.descriptor.clone(),
                physical.clone(),
                RetainedStateUseV2::Inactive,
            )
            .unwrap(),
        );
        assert!(inactive.managed_kv_runtime().is_none());
        assert_ne!(inactive.id, runtime.id);
        assert_eq!(
            runtime
                .managed_kv_runtime()
                .expect("managed backing")
                .state_plan_v2()
                .id,
            physical.state_plan_v2().id
        );
        assert_eq!(
            bundle
                .contract(CapabilityKind::Chat, false)
                .unwrap()
                .execution_profile
                .cache_mode,
            CacheMode::ExternalPaged
        );
        let mut request = crate::engine::EngineCoreRequest::chat(vec![])
            .with_model_variant(ModelVariant::Qwen306B);
        request.bind_model_instance(model_instance).unwrap();
        request.bind_execution_adapter(binding.execution).unwrap();
        request
            .bind_v2_state_runtime(runtime.clone(), runtime.state_fingerprint, BackendKind::Cpu)
            .unwrap();
        assert!(request.v2_state_runtime().is_some());
    }

    #[test]
    fn sealed_adapter_bundle_does_not_pin_an_idle_physical_generation() {
        let registry = RuntimeAdapterRegistry::built_in();
        let model_instance = ModelInstanceId::new(80);
        let state_contract = crate::kv::test_contract();
        let mut core = EngineCore::new(EngineCoreConfig {
            backend: BackendKind::Cpu,
            max_blocks: 4,
            block_size: 32,
            ..EngineCoreConfig::default()
        })
        .unwrap();
        let physical = core
            .load_managed_model_cache(
                model_instance,
                &InferenceStateCapability::Managed(state_contract.clone()),
                None,
            )
            .unwrap()
            .expect("physical managed runtime");
        let bundle = LoadedModelBundle::bind_with_state_publications(
            &registry,
            ExecutionGroupId::new(30),
            model_instance,
            ModelVariant::Qwen306B,
            BackendKind::Cpu,
            HashMap::from([(
                CapabilityKind::Chat,
                LoadedStatePublication::ManagedV2 {
                    contract: state_contract,
                    physical: physical.clone(),
                },
            )]),
        )
        .unwrap();
        let binding = bundle
            .capability_binding_for_streaming(CapabilityKind::Chat, StreamingRequirements::NONE)
            .unwrap();
        let runtime = binding.state;
        drop(physical);

        assert!(core
            .unload_managed_model_cache(model_instance)
            .expect("idle physical generation unload"));
        assert!(runtime.managed_kv_runtime().is_none());
        assert!(runtime
            .validate_against(BackendKind::Cpu, &binding.execution)
            .is_err());
    }

    #[test]
    fn qwen_asr_rejects_retained_only_publication() {
        let registry = RuntimeAdapterRegistry::built_in();
        let model_instance = ModelInstanceId::new(81);
        let state_contract = crate::kv::test_contract();
        let capability = InferenceStateCapability::Managed(state_contract.clone());
        let mut core = EngineCore::new(EngineCoreConfig {
            max_blocks: 4,
            block_size: 32,
            ..EngineCoreConfig::default()
        })
        .unwrap();
        let physical = core
            .load_managed_model_cache(model_instance, &capability, None)
            .unwrap()
            .expect("physical managed runtime");
        let error = LoadedModelBundle::bind_with_state_publications(
            &registry,
            ExecutionGroupId::new(31),
            model_instance,
            ModelVariant::Qwen3Asr06BGguf,
            BackendKind::Cpu,
            HashMap::from([(
                CapabilityKind::Asr,
                LoadedStatePublication::ManagedV2 {
                    contract: state_contract,
                    physical,
                },
            )]),
        )
        .expect_err("Qwen3 ASR needs both retained and invocation backing");
        assert!(error
            .to_string()
            .contains("invocation state cannot publish a retained-only"));
    }

    #[test]
    fn stateful_qwen_capability_fails_closed_without_physical_publication() {
        let registry = RuntimeAdapterRegistry::built_in();
        let error = LoadedModelBundle::bind(
            &registry,
            ExecutionGroupId::new(3),
            ModelInstanceId::new(10),
            ModelVariant::Qwen306B,
            BackendKind::Cpu,
        )
        .expect_err("stateful chat must not seal without physical state");
        assert!(error
            .to_string()
            .contains("requires an explicit load-sealed ABI-v2 state publication"));
    }

    #[test]
    fn lfm2_chat_fails_closed_without_managed_state_publication() {
        let registry = RuntimeAdapterRegistry::built_in();
        let error = LoadedModelBundle::bind(
            &registry,
            ExecutionGroupId::new(3),
            ModelInstanceId::new(11),
            ModelVariant::Lfm2512BInstructGguf,
            BackendKind::Cpu,
        )
        .expect_err("LFM2 chat must not seal without managed physical state");
        assert!(error
            .to_string()
            .contains("requires an explicit load-sealed ABI-v2 state publication"));
    }

    #[test]
    fn v2_state_publication_is_preserved_without_legacy_fallback() {
        let registry = RuntimeAdapterRegistry::built_in();
        let variant = ModelVariant::Qwen3TtsTokenizer12Hz;
        let capability = CapabilityKind::Tokenizer;
        let compatibility = LoadedModelBundle::bind(
            &registry,
            ExecutionGroupId::new(3),
            ModelInstanceId::new(10),
            variant,
            BackendKind::Cpu,
        )
        .unwrap();
        let offline = compatibility
            .contract_for_streaming(capability, StreamingRequirements::NONE)
            .unwrap()
            .stages;
        let transport = compatibility
            .contract_for_streaming(capability, StreamingRequirements::transport_only())
            .unwrap()
            .stages;
        let descriptor =
            crate::kv::v2::CapabilityStateDescriptorV2::stateless_for_stage_graphs_test(&[
                &offline, &transport,
            ]);
        let bundle = LoadedModelBundle::bind_with_state_publications(
            &registry,
            ExecutionGroupId::new(3),
            ModelInstanceId::new(10),
            variant,
            BackendKind::Cpu,
            HashMap::from([(capability, LoadedStatePublication::V2(descriptor.clone()))]),
        )
        .unwrap();

        let binding = bundle
            .capability_binding_for_streaming(capability, StreamingRequirements::NONE)
            .unwrap();
        let runtime = binding.state;
        assert_eq!(runtime.descriptor, descriptor);
        runtime
            .validate_against(BackendKind::Cpu, &binding.execution)
            .unwrap();
    }

    #[test]
    fn v2_runtime_reuses_one_seal_for_identical_selectable_stage_graphs() {
        let registry = RuntimeAdapterRegistry::built_in();
        let variant = ModelVariant::Qwen3TtsTokenizer12Hz;
        let capability = CapabilityKind::Tokenizer;
        let compatibility = LoadedModelBundle::bind(
            &registry,
            ExecutionGroupId::new(3),
            ModelInstanceId::new(11),
            variant,
            BackendKind::Cpu,
        )
        .unwrap();
        let offline = compatibility
            .contract_for_streaming(capability, StreamingRequirements::NONE)
            .unwrap();
        let descriptor =
            crate::kv::v2::CapabilityStateDescriptorV2::stateless_for_stages_test(&offline.stages);

        let bundle = LoadedModelBundle::bind_with_state_publications(
            &registry,
            ExecutionGroupId::new(3),
            ModelInstanceId::new(11),
            variant,
            BackendKind::Cpu,
            HashMap::from([(capability, LoadedStatePublication::V2(descriptor))]),
        )
        .unwrap();
        let offline = bundle
            .capability_binding_for_streaming(capability, StreamingRequirements::NONE)
            .unwrap();
        let transport = bundle
            .capability_binding_for_streaming(capability, StreamingRequirements::transport_only())
            .unwrap();
        assert_eq!(
            offline.state.state_fingerprint,
            transport.state.state_fingerprint
        );
    }

    #[test]
    fn stateful_capability_cannot_publish_ready_without_physical_backing() {
        let registry = RuntimeAdapterRegistry::built_in();
        let variant = ModelVariant::Nemotron35AsrStreaming06B;
        let error = LoadedModelBundle::bind(
            &registry,
            ExecutionGroupId::new(3),
            ModelInstanceId::new(14),
            variant,
            BackendKind::Cpu,
        )
        .expect_err("Nemotron realtime must fail closed without physical publication");
        assert!(error
            .to_string()
            .contains("requires an explicit load-sealed ABI-v2 state publication"));
    }

    #[test]
    fn nemotron_realtime_factory_authors_atomic_physical_tensor_workspace() {
        let draft = LoadedModelBundleDraft::build(
            &RuntimeAdapterRegistry::built_in(),
            ExecutionGroupId::new(3),
            ModelInstanceId::new(15),
            ModelVariant::Nemotron35AsrStreaming06B,
            BackendKind::Cpu,
        )
        .unwrap();
        let contracts = draft
            .execution_contracts(CapabilityKind::RealtimeAsr)
            .unwrap();

        assert!(!contracts.is_empty());
        for contract in contracts {
            assert_eq!(contract.adapter_abi_revision, NEMOTRON_REALTIME_ADAPTER_ABI);
            assert_eq!(contract.execution_profile.mode, ExecutionMode::Atomic);
            assert_eq!(contract.execution_profile.cache_mode, CacheMode::None);
            assert_eq!(contract.stages.len(), 1);
            assert_eq!(contract.stages[0].selector, StageWorkSelector::Atomic);
            assert_eq!(
                contract.stages[0].max_workspace_bytes,
                crate::models::architectures::nemotron::asr::NEMOTRON_REALTIME_STAGE_WORKSPACE_BYTES
            );
        }
    }

    #[test]
    fn managed_v2_publication_requires_a_physical_runtime_before_ready() {
        let registry = RuntimeAdapterRegistry::built_in();
        let draft = LoadedModelBundleDraft::build(
            &registry,
            ExecutionGroupId::new(3),
            ModelInstanceId::new(12),
            ModelVariant::Qwen306B,
            BackendKind::Cpu,
        )
        .unwrap();
        let stages = draft
            .execution_contracts(CapabilityKind::Chat)
            .unwrap()
            .remove(0)
            .stages;
        let descriptor = CapabilityStateDescriptorV2::managed_for_stages_test(
            crate::kv::v2::test_contract(),
            &stages,
        );

        let error = draft
            .seal(HashMap::from([(
                CapabilityKind::Chat,
                LoadedStatePublication::V2(descriptor),
            )]))
            .expect_err("managed v2 metadata alone must not publish Ready");
        assert!(error.to_string().contains("requires physical backing"));
    }

    #[test]
    fn stateless_v2_rejects_execution_that_declares_model_owned_cache() {
        let registry = RuntimeAdapterRegistry::built_in();
        let variant = ModelVariant::Qwen3508BGguf;
        let draft = LoadedModelBundleDraft::build(
            &registry,
            ExecutionGroupId::new(3),
            ModelInstanceId::new(13),
            variant,
            BackendKind::Cpu,
        )
        .unwrap();
        let stages = draft
            .execution_contracts(CapabilityKind::Chat)
            .unwrap()
            .remove(0)
            .stages;
        let descriptor = CapabilityStateDescriptorV2::stateless_for_stages_test(&stages);

        let error = draft
            .seal(HashMap::from([(
                CapabilityKind::Chat,
                LoadedStatePublication::V2(descriptor),
            )]))
            .expect_err("stateless v2 must not relabel a retained sequence cache");
        assert!(
            error
                .to_string()
                .contains("requiring retained inference state"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn sealed_state_runtime_is_scoped_to_one_capability_descriptor() {
        let registry = RuntimeAdapterRegistry::built_in();
        let bundle = LoadedModelBundle::bind(
            &registry,
            ExecutionGroupId::new(4),
            ModelInstanceId::new(11),
            ModelVariant::Kokoro82M,
            BackendKind::Cpu,
        )
        .unwrap();

        let tts = bundle
            .capability_binding_for_streaming(CapabilityKind::Tts, StreamingRequirements::NONE)
            .unwrap();
        let streaming_tts = bundle
            .capability_binding_for_streaming(
                CapabilityKind::StreamingTts,
                StreamingRequirements::NONE,
            )
            .unwrap();
        assert_ne!(
            tts.execution.capability_id,
            streaming_tts.execution.capability_id
        );
        assert_ne!(tts.state.id, streaming_tts.state.id);
    }

    #[test]
    fn cache_truth_for_an_unregistered_capability_is_rejected() {
        let state_publications = HashMap::from([(
            CapabilityKind::Asr,
            LoadedStatePublication::V2(CapabilityStateDescriptorV2::stateless_for_test()),
        )]);

        let error = LoadedModelBundle::bind_with_state_publications(
            &RuntimeAdapterRegistry::built_in(),
            ExecutionGroupId::new(5),
            ModelInstanceId::new(12),
            ModelVariant::Qwen306B,
            BackendKind::Cpu,
            state_publications,
        )
        .expect_err("an unmatched cache declaration must fail closed");

        assert!(error.to_string().contains("unregistered capabilities"));
        assert!(error.to_string().contains("asr"));
    }

    #[derive(Debug)]
    struct TestStaticTtsFactory {
        id: &'static str,
        model_variant: ModelVariant,
    }

    impl LoadedExecutionAdapterFactory for TestStaticTtsFactory {
        fn id(&self) -> &'static str {
            self.id
        }

        fn batch_mode(&self) -> NativeBatchMode {
            NativeBatchMode::Static
        }

        fn supports(&self, metadata: AdapterMetadata, _backend_kind: BackendKind) -> bool {
            metadata.model_variant == self.model_variant
                && matches!(
                    metadata.capability,
                    CapabilityKind::Tts | CapabilityKind::StreamingTts
                )
        }

        fn create(
            &self,
            context: LoadedAdapterFactoryContext,
            metadata: AdapterMetadata,
        ) -> Result<Arc<dyn LoadedExecutionAdapter>> {
            Ok(Arc::new(StaticTtsExecutionAdapter::new(
                context.execution_group_id,
                context.model_instance_id,
                metadata,
                context.backend_kind,
                context.max_tensor_batch_size,
                context.request_parallelism,
            )))
        }
    }

    #[derive(Debug)]
    struct TestScalarFactory {
        excluded_model_variant: ModelVariant,
    }

    impl LoadedExecutionAdapterFactory for TestScalarFactory {
        fn id(&self) -> &'static str {
            "test.scalar"
        }

        fn batch_mode(&self) -> NativeBatchMode {
            NativeBatchMode::None
        }

        fn supports(&self, metadata: AdapterMetadata, _backend_kind: BackendKind) -> bool {
            metadata.model_variant != self.excluded_model_variant
                && !is_physical_qwen_tts(metadata)
                && !is_nemotron_realtime(metadata)
                && !is_continuous_physical_chat(metadata)
        }

        fn create(
            &self,
            context: LoadedAdapterFactoryContext,
            metadata: AdapterMetadata,
        ) -> Result<Arc<dyn LoadedExecutionAdapter>> {
            Ok(Arc::new(ScalarExecutionAdapter::new(
                context.execution_group_id,
                context.model_instance_id,
                metadata,
                context.backend_kind,
                context.request_parallelism,
            )))
        }
    }

    #[test]
    fn every_supported_model_capability_authors_an_exact_width_one_contract() {
        let registry = RuntimeAdapterRegistry::built_in();

        for (index, variant) in ModelVariant::all().iter().copied().enumerate() {
            let instance = ModelInstanceId::new(index as u64 + 1);
            let draft = LoadedModelBundleDraft::build(
                &registry,
                ExecutionGroupId::new(7),
                instance,
                variant,
                BackendKind::Cpu,
            )
            .unwrap_or_else(|error| panic!("failed to build {variant}: {error}"));
            let metadata = registry.capabilities_for(variant);

            assert_eq!(draft.capabilities.len(), metadata.len(), "{variant}");
            for metadata in metadata {
                let execution = draft.capabilities.get(&metadata.capability).unwrap();
                let contract = execution
                    .contract(StreamingRequirements::NONE)
                    .unwrap_or_else(|error| panic!("failed to contract {variant}: {error}"));
                assert_eq!(contract.execution_group_id, ExecutionGroupId::new(7));
                assert_eq!(contract.model_instance_id, instance);
                assert_eq!(contract.metadata, metadata);
                let factory = registry
                    .loaded_adapter_factory(metadata, BackendKind::Cpu)
                    .unwrap();
                if factory.id() == "builtin.scalar" {
                    assert_eq!(contract.adapter_abi_revision, SCALAR_ADAPTER_ABI);
                    assert_eq!(contract.stages.len(), 1);
                    assert_eq!(contract.stages[0].batch_mode, NativeBatchMode::None);
                } else {
                    assert_ne!(contract.adapter_abi_revision, SCALAR_ADAPTER_ABI);
                    assert!(contract
                        .stages
                        .iter()
                        .any(|stage| stage.batch_mode == factory.batch_mode()));
                }
                assert!(contract
                    .stages
                    .iter()
                    .all(|stage| stage.max_batch_size == 1));
                assert_eq!(contract.execution_profile.max_batch_size, 1);
                assert!(contract.execution_profile.resolved_from_loaded_model);

                let transport = execution
                    .contract(StreamingRequirements::transport_only())
                    .unwrap_or_else(|error| {
                        panic!("failed transport-only contract for {variant}: {error}")
                    });
                assert_eq!(transport.metadata, metadata);

                let native_streaming = execution.contract(StreamingRequirements::native(true));
                if metadata.streaming_mode == StreamingMode::None {
                    assert!(
                        native_streaming.is_err(),
                        "{variant} {:?} unexpectedly advertised native streaming",
                        metadata.capability
                    );
                } else {
                    native_streaming.unwrap_or_else(|error| {
                        panic!("failed native-streaming contract for {variant}: {error}")
                    });
                }
            }
        }
    }

    #[test]
    fn every_stateful_capability_fails_closed_without_physical_publication() {
        let registry = RuntimeAdapterRegistry::built_in();
        let backends = [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda];

        for backend in backends {
            for (index, variant) in ModelVariant::all().iter().copied().enumerate() {
                let draft = LoadedModelBundleDraft::build(
                    &registry,
                    ExecutionGroupId::new(11),
                    ModelInstanceId::new(index as u64 + 1),
                    variant,
                    backend,
                )
                .unwrap_or_else(|error| {
                    panic!("failed to build {variant} for {backend:?}: {error}")
                });

                for execution in draft.capabilities.values() {
                    let metadata = execution.metadata();
                    let sealed = LoadedCapabilityDescriptor::new(execution.clone(), None, backend);
                    if metadata.state_requirement == InferenceStateRequirement::Stateless {
                        sealed.unwrap_or_else(|error| {
                            panic!(
                                "stateless {variant} {:?} failed to seal for {backend:?}: {error}",
                                metadata.capability
                            )
                        });
                    } else {
                        let error = sealed.expect_err(&format!(
                            "stateful {variant} {:?} sealed without physical state for {backend:?}",
                            metadata.capability
                        ));
                        assert!(
                            error.to_string().contains(
                                "requires an explicit load-sealed ABI-v2 state publication"
                            ),
                            "unexpected fail-closed error for {variant} {:?} on {backend:?}: {error}",
                            metadata.capability
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn loaded_launch_policy_matrix_is_group_exclusive_without_concurrency_evidence() {
        let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(1, 3).unwrap();

        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            for (index, variant) in ModelVariant::all().iter().copied().enumerate() {
                let draft = LoadedModelBundleDraft::build(
                    &registry,
                    ExecutionGroupId::new(41),
                    ModelInstanceId::new(index as u64 + 1),
                    variant,
                    backend,
                )
                .unwrap_or_else(|error| {
                    panic!("failed to build {variant} for {backend:?}: {error}")
                });
                for execution in draft.capabilities.values() {
                    let metadata = execution.metadata();
                    for contract in
                        loaded_execution_contracts(execution.as_ref()).unwrap_or_else(|error| {
                            panic!(
                                "failed launch contract for {variant} {:?} on {backend:?}: {error}",
                                metadata.capability
                            )
                        })
                    {
                        let expected = PhysicalLaunchPolicy::ExecutionGroupExclusive;
                        assert_eq!(
                            contract.execution_profile.physical_launch_policy, expected,
                            "{variant} {:?} on {backend:?}",
                            metadata.capability
                        );
                        assert!(contract
                            .stages
                            .iter()
                            .all(|stage| stage.physical_launch_policy == expected));
                    }
                }
            }
        }
    }

    #[test]
    fn missing_concurrency_evidence_keeps_cpu_whisper_and_tts_scalar_width_one() {
        let request_parallelism = 4;
        let registry =
            RuntimeAdapterRegistry::built_in_with_execution_limits(2, request_parallelism).unwrap();
        let whisper = LoadedModelBundleDraft::build(
            &registry,
            ExecutionGroupId::new(44),
            ModelInstanceId::new(1),
            ModelVariant::WhisperLargeV3Turbo,
            BackendKind::Cpu,
        )
        .unwrap();
        let whisper = whisper
            .execution_contracts(CapabilityKind::Asr)
            .unwrap()
            .remove(0);
        assert_eq!(whisper.execution_profile.max_batch_size, 1);
        assert_eq!(
            whisper.execution_profile.concurrency,
            ConcurrencyClass::Exclusive
        );
        assert_eq!(
            whisper.execution_profile.physical_launch_policy,
            PhysicalLaunchPolicy::ExecutionGroupExclusive
        );
        assert!(whisper.stages.iter().all(|stage| {
            stage.batch_mode == NativeBatchMode::None
                && stage.max_batch_size == 1
                && stage.concurrency == ConcurrencyClass::Exclusive
                && stage.shape_policy == StageShapePolicy::Exact
                && stage.physical_launch_policy == PhysicalLaunchPolicy::ExecutionGroupExclusive
        }));

        for (index, variant) in [
            ModelVariant::Kokoro82M,
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
        ]
        .into_iter()
        .enumerate()
        {
            let tts = LoadedModelBundleDraft::build(
                &registry,
                ExecutionGroupId::new(45),
                ModelInstanceId::new(index as u64 + 2),
                variant,
                BackendKind::Cpu,
            )
            .unwrap();
            for contract in tts.execution_contracts(CapabilityKind::Tts).unwrap() {
                assert_eq!(
                    contract.execution_profile.physical_launch_policy,
                    PhysicalLaunchPolicy::ExecutionGroupExclusive
                );
                assert_eq!(contract.execution_profile.max_batch_size, 1);
                assert_eq!(
                    contract.execution_profile.concurrency,
                    ConcurrencyClass::Exclusive
                );
                assert!(!contract.execution_profile.capabilities().native_batch);
                assert!(contract
                    .stages
                    .iter()
                    .all(|stage| stage.batch_mode == NativeBatchMode::None));
            }
        }
    }

    #[test]
    fn current_audio_adapters_remain_scalar_until_a_family_native_call_opts_in() {
        let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(8, 4).unwrap();
        let audio_capability = |capability| {
            matches!(
                capability,
                CapabilityKind::Asr
                    | CapabilityKind::SpeakerAttributedAsr
                    | CapabilityKind::RealtimeAsr
                    | CapabilityKind::Tts
                    | CapabilityKind::StreamingTts
            )
        };

        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            for (index, variant) in ModelVariant::all().iter().copied().enumerate() {
                let draft = LoadedModelBundleDraft::build(
                    &registry,
                    ExecutionGroupId::new(46),
                    ModelInstanceId::new(index as u64 + 1),
                    variant,
                    backend,
                )
                .unwrap_or_else(|error| {
                    panic!("failed to build {variant} for {backend:?}: {error}")
                });
                for execution in draft
                    .capabilities
                    .values()
                    .filter(|execution| audio_capability(execution.metadata().capability))
                {
                    for contract in
                        loaded_execution_contracts(execution.as_ref()).unwrap_or_else(|error| {
                            panic!(
                                "failed audio contract for {variant} {:?} on {backend:?}: {error}",
                                execution.metadata().capability
                            )
                        })
                    {
                        assert_eq!(contract.execution_profile.max_batch_size, 1);
                        assert_eq!(
                            contract.execution_profile.concurrency,
                            ConcurrencyClass::Exclusive
                        );
                        assert!(!contract.execution_profile.capabilities().native_batch);
                        assert!(contract.stages.iter().all(|stage| {
                            stage.batch_mode == NativeBatchMode::None
                                && stage.max_batch_size == 1
                                && stage.concurrency == ConcurrencyClass::Exclusive
                        }));
                    }
                }
            }
        }
    }

    #[test]
    fn loaded_contracts_reject_policy_without_evidence_and_stage_profile_mismatch() {
        let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(1, 3).unwrap();
        let unknown = LoadedModelBundleDraft::build(
            &registry,
            ExecutionGroupId::new(42),
            ModelInstanceId::new(1),
            ModelVariant::Kokoro82M,
            BackendKind::Cpu,
        )
        .unwrap();
        let base = unknown
            .execution_contracts(CapabilityKind::Tts)
            .unwrap()
            .remove(0);
        assert_eq!(
            base.execution_profile.concurrency,
            ConcurrencyClass::Exclusive
        );
        assert_eq!(
            base.execution_profile.physical_launch_policy,
            PhysicalLaunchPolicy::ExecutionGroupExclusive
        );

        for policy in [
            PhysicalLaunchPolicy::ModelExclusive,
            PhysicalLaunchPolicy::concurrent(3).unwrap(),
        ] {
            let mut contract = base.clone();
            contract.execution_profile.physical_launch_policy = policy;
            contract.stages = contract
                .stages
                .iter()
                .cloned()
                .map(|mut stage| {
                    stage.physical_launch_policy = policy;
                    stage
                })
                .collect::<Vec<_>>()
                .into();
            let error = contract
                .validate_physical_launch_policy()
                .expect_err("unsupported model policy must fail closed");
            assert!(error
                .to_string()
                .contains("no production concurrency evidence is available"));
        }

        let mut mismatch = base;
        let mut stages = mismatch.stages.to_vec();
        stages[0].physical_launch_policy = PhysicalLaunchPolicy::ModelExclusive;
        mismatch.stages = stages.into();
        let error = mismatch
            .validate_physical_launch_policy()
            .expect_err("stage/profile launch-policy mismatch must fail closed");
        assert!(error.to_string().contains("stage launch policy"));
    }

    #[test]
    fn whisper_metadata_and_scalar_abi_cannot_manufacture_concurrent_policy() {
        let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(1, 3).unwrap();
        let whisper = LoadedModelBundleDraft::build(
            &registry,
            ExecutionGroupId::new(43),
            ModelInstanceId::new(2),
            ModelVariant::WhisperLargeV3Turbo,
            BackendKind::Cpu,
        )
        .unwrap();
        let mut manufactured = whisper
            .execution_contracts(CapabilityKind::Asr)
            .unwrap()
            .remove(0);
        assert_eq!(manufactured.adapter_abi_revision, SCALAR_ADAPTER_ABI);
        assert_eq!(
            manufactured.metadata.model_variant,
            ModelVariant::WhisperLargeV3Turbo
        );
        assert_eq!(manufactured.metadata.capability, CapabilityKind::Asr);

        let concurrent = PhysicalLaunchPolicy::concurrent(3).unwrap();
        manufactured.execution_profile.max_batch_size = 3;
        manufactured.execution_profile.concurrency = ConcurrencyClass::Batchable;
        manufactured.execution_profile.physical_launch_policy = concurrent;
        manufactured.stages = manufactured
            .stages
            .iter()
            .cloned()
            .map(|mut stage| {
                stage.max_batch_size = 3;
                stage.max_work_units = 3;
                stage.concurrency = ConcurrencyClass::Batchable;
                stage.shape_policy = StageShapePolicy::Independent;
                stage.physical_launch_policy = concurrent;
                stage
            })
            .collect::<Vec<_>>()
            .into();

        let error = manufactured
            .validate_physical_launch_policy()
            .expect_err("metadata and adapter ABI are not concurrency evidence");
        assert!(error
            .to_string()
            .contains("no production concurrency evidence is available"));
    }

    #[test]
    fn scalar_adapters_remain_exact_width_one_without_concurrency_evidence() {
        let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(1, 3).unwrap();

        for (index, variant) in ModelVariant::all().iter().copied().enumerate() {
            let draft = LoadedModelBundleDraft::build(
                &registry,
                ExecutionGroupId::new(9),
                ModelInstanceId::new(index as u64 + 1),
                variant,
                BackendKind::Cpu,
            )
            .unwrap_or_else(|error| panic!("failed to build {variant}: {error}"));
            for metadata in registry.capabilities_for(variant) {
                if registry
                    .loaded_adapter_factory(metadata, BackendKind::Cpu)
                    .unwrap()
                    .id()
                    != "builtin.scalar"
                {
                    continue;
                }
                let contract = draft
                    .execution_contracts(metadata.capability)
                    .unwrap_or_else(|error| panic!("failed to contract {variant}: {error}"));
                for contract in contract {
                    assert_eq!(contract.adapter_abi_revision, SCALAR_ADAPTER_ABI);
                    assert_eq!(contract.execution_profile.max_batch_size, 1);
                    assert_eq!(
                        contract.execution_profile.concurrency,
                        ConcurrencyClass::Exclusive
                    );
                    assert_eq!(
                        contract.execution_profile.physical_launch_policy,
                        PhysicalLaunchPolicy::ExecutionGroupExclusive
                    );
                    assert_eq!(contract.stages[0].max_batch_size, 1);
                    assert_eq!(
                        contract.stages[0].shape_policy,
                        crate::engine::StageShapePolicy::Exact
                    );
                    assert_eq!(
                        contract.stages[0].physical_launch_policy,
                        PhysicalLaunchPolicy::ExecutionGroupExclusive
                    );
                }
            }
        }

        let metal = LoadedModelBundle::bind(
            &registry,
            ExecutionGroupId::new(9),
            ModelInstanceId::new(999),
            ModelVariant::Kokoro82M,
            BackendKind::Metal,
        )
        .unwrap();
        let contract = metal.contract(CapabilityKind::Tts, false).unwrap();
        assert_eq!(contract.execution_profile.max_batch_size, 1);
        assert_eq!(
            contract.execution_profile.concurrency,
            ConcurrencyClass::Exclusive
        );

        let cuda = LoadedModelBundleDraft::build(
            &registry,
            ExecutionGroupId::new(10),
            ModelInstanceId::new(1_000),
            ModelVariant::Lfm2512BThinkingGguf,
            BackendKind::Cuda,
        )
        .unwrap();
        let contract = cuda
            .execution_contracts(CapabilityKind::Chat)
            .unwrap()
            .pop()
            .unwrap();
        assert_eq!(contract.execution_profile.max_batch_size, 1);
        assert_eq!(contract.stages[0].max_batch_size, 1);
        assert_eq!(
            contract.execution_profile.concurrency,
            ConcurrencyClass::Batchable
        );
        assert_eq!(contract.stages.len(), 2);
        assert_eq!(contract.stages[1].batch_mode, NativeBatchMode::Continuous);
        assert_eq!(contract.stages[1].max_batch_size, 1);
    }

    #[test]
    fn voxtral_streaming_binds_to_its_exact_token_engine_adapter() {
        let variant = ModelVariant::VoxtralMini4BRealtime2602;
        let draft = LoadedModelBundleDraft::build(
            &RuntimeAdapterRegistry::built_in(),
            ExecutionGroupId::new(7),
            ModelInstanceId::new(1),
            variant,
            BackendKind::Cpu,
        )
        .unwrap();

        let contract = draft
            .execution_contracts(CapabilityKind::Asr)
            .unwrap()
            .into_iter()
            .find(|contract| {
                contract.stages[0].output_visibility == OutputVisibility::IncrementalCommitted
            })
            .expect("streaming Voxtral contract");
        assert_eq!(
            contract.metadata.execution_target,
            ExecutionTargetKind::TokenEngine
        );
        assert_eq!(contract.metadata.streaming_mode, StreamingMode::Chunked);
        assert!(contract.execution_profile.resolved_from_loaded_model);
        assert_eq!(
            contract.stages[0].output_visibility,
            OutputVisibility::IncrementalCommitted
        );
        let error = draft
            .seal(HashMap::new())
            .expect_err("Voxtral must not seal without physical invocation state");
        assert!(error
            .to_string()
            .contains("requires an explicit load-sealed ABI-v2 state publication"));
    }

    #[test]
    fn offline_asr_transport_progress_does_not_require_native_streaming() {
        let draft = LoadedModelBundleDraft::build(
            &RuntimeAdapterRegistry::built_in(),
            ExecutionGroupId::new(7),
            ModelInstanceId::new(1),
            ModelVariant::ParakeetTdt06BV3,
            BackendKind::Cpu,
        )
        .unwrap();
        let execution = draft.capabilities.get(&CapabilityKind::Asr).unwrap();
        assert!(execution
            .contract(StreamingRequirements::native(true))
            .is_err());
        let transport = execution
            .contract(StreamingRequirements::transport_only())
            .expect("offline ASR must expose atomic executor progress");
        assert_eq!(transport.metadata.streaming_mode, StreamingMode::None);
        assert_eq!(transport.execution_profile.mode, ExecutionMode::Atomic);
        assert_eq!(
            transport.stages[0].output_visibility,
            OutputVisibility::IncrementalCommitted
        );
    }

    #[test]
    fn lfm2_sequence_chat_remains_quantum_committed_when_streaming() {
        let draft = LoadedModelBundleDraft::build(
            &RuntimeAdapterRegistry::built_in(),
            ExecutionGroupId::new(7),
            ModelInstanceId::new(1),
            ModelVariant::Lfm2512BThinkingGguf,
            BackendKind::Cpu,
        )
        .unwrap();
        let adapter = draft.capabilities.get(&CapabilityKind::Chat).unwrap();

        let non_streaming = adapter.contract(StreamingRequirements::NONE).unwrap();
        assert_eq!(
            non_streaming.execution_profile.mode,
            ExecutionMode::Sequence
        );
        assert!(non_streaming
            .stages
            .iter()
            .all(|stage| stage.output_visibility == OutputVisibility::AfterQuantumCommit));

        let streaming = adapter
            .contract(StreamingRequirements::native(true))
            .unwrap();
        assert_eq!(streaming.execution_profile.mode, ExecutionMode::Sequence);
        assert!(streaming
            .stages
            .iter()
            .all(|stage| stage.output_visibility == OutputVisibility::AfterQuantumCommit));
    }

    #[test]
    fn sequence_chat_remains_quantum_committed_when_streaming() {
        let draft = LoadedModelBundleDraft::build(
            &RuntimeAdapterRegistry::built_in(),
            ExecutionGroupId::new(7),
            ModelInstanceId::new(1),
            ModelVariant::Qwen3508BGguf,
            BackendKind::Cpu,
        )
        .unwrap();

        let streaming = draft
            .capabilities
            .get(&CapabilityKind::Chat)
            .unwrap()
            .contract(StreamingRequirements::native(true))
            .unwrap();
        assert_eq!(streaming.execution_profile.mode, ExecutionMode::Sequence);
        assert!(streaming
            .stages
            .iter()
            .all(|stage| { stage.output_visibility == OutputVisibility::AfterQuantumCommit }));
    }

    #[test]
    fn adapter_instances_are_distinct_across_capabilities_and_loads() {
        let registry = RuntimeAdapterRegistry::built_in();
        let first = LoadedModelBundle::bind(
            &registry,
            ExecutionGroupId::new(1),
            ModelInstanceId::new(1),
            ModelVariant::Kokoro82M,
            BackendKind::Metal,
        )
        .expect("first bundle");
        let second = LoadedModelBundle::bind(
            &registry,
            ExecutionGroupId::new(1),
            ModelInstanceId::new(2),
            ModelVariant::Kokoro82M,
            BackendKind::Metal,
        )
        .expect("second bundle");

        let first_tts = first
            .capability_binding_for_streaming(CapabilityKind::Tts, StreamingRequirements::NONE)
            .expect("first tts")
            .execution
            .adapter_instance_id;
        let first_streaming_tts = first
            .capability_binding_for_streaming(
                CapabilityKind::StreamingTts,
                StreamingRequirements::NONE,
            )
            .expect("first streaming tts")
            .execution
            .adapter_instance_id;
        let second_tts = second
            .capability_binding_for_streaming(CapabilityKind::Tts, StreamingRequirements::NONE)
            .expect("second tts")
            .execution
            .adapter_instance_id;

        assert_ne!(first_tts, first_streaming_tts);
        assert_ne!(first_tts, second_tts);
    }

    #[test]
    fn bundle_draft_preserves_the_exact_adapter_identity_through_state_seal() {
        let registry = RuntimeAdapterRegistry::built_in();
        let draft = LoadedModelBundleDraft::build(
            &registry,
            ExecutionGroupId::new(12),
            ModelInstanceId::new(91),
            ModelVariant::Kokoro82M,
            BackendKind::Cpu,
        )
        .unwrap();
        let contracts = draft.execution_contracts(CapabilityKind::Tts).unwrap();
        let adapter = contracts[0].adapter_instance_id;
        assert!(contracts
            .iter()
            .all(|contract| contract.adapter_instance_id == adapter));

        let bundle = draft.seal(HashMap::new()).unwrap();
        let sealed = bundle
            .contract_for_streaming(CapabilityKind::Tts, StreamingRequirements::NONE)
            .unwrap();
        assert_eq!(sealed.adapter_instance_id, adapter);
    }

    #[test]
    fn replacing_the_scalar_factory_adds_an_optimized_model_without_bundle_branching() {
        let variant = ModelVariant::Kokoro82M;
        let mut registry = RuntimeAdapterRegistry::built_in();
        registry
            .loaded_adapter_factories
            .retain(|factory| factory.id() != "builtin.scalar");
        registry
            .loaded_adapter_factories
            .push(Arc::new(TestScalarFactory {
                excluded_model_variant: variant,
            }));
        registry
            .loaded_adapter_factories
            .push(Arc::new(TestStaticTtsFactory {
                id: "test.kokoro.tensor_static",
                model_variant: variant,
            }));
        registry.validate_loaded_adapter_factories().unwrap();
        let draft = LoadedModelBundleDraft::build(
            &registry,
            ExecutionGroupId::new(1),
            ModelInstanceId::new(2),
            variant,
            BackendKind::Cpu,
        )
        .unwrap();
        let contract = draft
            .capabilities
            .get(&CapabilityKind::Tts)
            .unwrap()
            .contract(StreamingRequirements::NONE)
            .unwrap();

        assert_eq!(contract.adapter_abi_revision, STATIC_TENSOR_ADAPTER_ABI);
        assert_eq!(contract.stages[0].batch_mode, NativeBatchMode::Static);
        assert!(registry
            .static_tensor_batch_variants(BackendKind::Cpu)
            .contains(&variant));
    }

    #[test]
    fn missing_loaded_factory_fails_closed() {
        let variant = ModelVariant::Kokoro82M;
        let mut registry = RuntimeAdapterRegistry::built_in();
        registry
            .loaded_adapter_factories
            .retain(|factory| factory.id() != "builtin.scalar");
        let metadata = *registry.require(CapabilityKind::Tts, variant).unwrap();

        let error = registry
            .loaded_adapter_factory(metadata, BackendKind::Cpu)
            .expect_err("every loaded capability requires exactly one factory");

        let message = error.to_string();
        assert!(message.contains("has no execution adapter factory"));
        assert!(message.contains(&variant.to_string()));
    }

    #[test]
    fn overlapping_loaded_factories_fail_closed() {
        let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        let mut registry = RuntimeAdapterRegistry::built_in_with_execution_limits(2, 1).unwrap();
        registry
            .loaded_adapter_factories
            .push(Arc::new(TestStaticTtsFactory {
                id: "test.overlapping.tensor_static",
                model_variant: variant,
            }));

        let error = LoadedModelBundle::bind(
            &registry,
            ExecutionGroupId::new(1),
            ModelInstanceId::new(2),
            variant,
            BackendKind::Cpu,
        )
        .expect_err("ambiguous factories must not depend on registration order");

        assert!(error.to_string().contains("matches both"));
        assert!(error.to_string().contains("test.overlapping.tensor_static"));
    }

    #[test]
    fn qwen_tts_factory_binds_every_capability_to_physical_sequence_stages() {
        let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(4, 1).unwrap();
        let draft = LoadedModelBundleDraft::build(
            &registry,
            ExecutionGroupId::new(1),
            ModelInstanceId::new(2),
            variant,
            BackendKind::Metal,
        )
        .unwrap();

        let tts = draft.capabilities.get(&CapabilityKind::Tts).unwrap();
        let physical = tts.contract(StreamingRequirements::NONE).unwrap();
        assert_eq!(physical.adapter_abi_revision, AdapterAbiRevision::new(3));
        assert_eq!(physical.execution_profile.mode, ExecutionMode::Sequence);
        assert_eq!(
            physical.execution_profile.cache_mode,
            CacheMode::ExternalPaged
        );
        assert_eq!(physical.execution_profile.max_batch_size, 1);
        assert_eq!(physical.stages.len(), 2);
        assert_eq!(
            physical.stages[0].selector,
            StageWorkSelector::SequencePrefill
        );
        assert_eq!(physical.stages[0].max_workspace_bytes, 0);
        assert_eq!(
            physical.stages[1].selector,
            StageWorkSelector::SequenceDecode
        );
        assert_eq!(physical.stages[1].max_workspace_bytes, 0);

        let streaming = tts.contract(StreamingRequirements::native(true)).unwrap();
        assert_eq!(streaming.adapter_abi_revision, AdapterAbiRevision::new(3));
        assert_eq!(
            streaming.execution_profile.prefill_batch,
            NativeBatchMode::None
        );
        assert_eq!(streaming.execution_profile.max_batch_size, 1);
        assert_eq!(streaming.stages[0].batch_mode, NativeBatchMode::None);

        let streaming_capability = draft
            .capabilities
            .get(&CapabilityKind::StreamingTts)
            .unwrap()
            .contract(StreamingRequirements::NONE)
            .unwrap();
        assert_eq!(
            streaming_capability.adapter_abi_revision,
            AdapterAbiRevision::new(3)
        );
        assert_eq!(
            streaming_capability.execution_profile.cache_mode,
            CacheMode::ExternalPaged
        );
    }

    #[test]
    fn qwen_tts_physical_sequence_is_enabled_on_cpu_by_default() {
        let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(4, 1).unwrap();
        let draft = LoadedModelBundleDraft::build(
            &registry,
            ExecutionGroupId::new(1),
            ModelInstanceId::new(2),
            variant,
            BackendKind::Cpu,
        )
        .unwrap();

        let contract = draft
            .capabilities
            .get(&CapabilityKind::Tts)
            .unwrap()
            .contract(StreamingRequirements::NONE)
            .unwrap();
        assert_eq!(contract.adapter_abi_revision, AdapterAbiRevision::new(3));
        assert_eq!(contract.execution_profile.max_batch_size, 1);
        assert_eq!(contract.stages[0].batch_mode, NativeBatchMode::None);
        assert_eq!(
            contract.execution_profile.cache_mode,
            CacheMode::ExternalPaged
        );
    }

    #[test]
    fn qwen_chat_native_factory_publishes_scalar_prefill_and_ragged_decode() {
        let variant = ModelVariant::Qwen306B;
        let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(8, 1).unwrap();
        let draft = LoadedModelBundleDraft::build(
            &registry,
            ExecutionGroupId::new(1),
            ModelInstanceId::new(2),
            variant,
            BackendKind::Cuda,
        )
        .unwrap();

        let contract = draft
            .capabilities
            .get(&CapabilityKind::Chat)
            .unwrap()
            .contract(StreamingRequirements::native(true))
            .unwrap();
        assert_eq!(contract.adapter_abi_revision, CONTINUOUS_TENSOR_ADAPTER_ABI);
        assert_eq!(contract.execution_profile.mode, ExecutionMode::Sequence);
        assert_eq!(
            contract.execution_profile.prefill_batch,
            NativeBatchMode::None
        );
        assert_eq!(
            contract.execution_profile.decode_batch,
            NativeBatchMode::Continuous
        );
        assert_eq!(contract.execution_profile.max_batch_size, 8);
        assert_eq!(contract.stages.len(), 2);
        assert_eq!(
            contract.stages[0].selector,
            StageWorkSelector::SequencePrefill
        );
        assert_eq!(contract.stages[0].batch_mode, NativeBatchMode::None);
        assert_eq!(contract.stages[0].max_batch_size, 1);
        assert_eq!(
            contract.stages[1].selector,
            StageWorkSelector::SequenceDecode
        );
        assert_eq!(contract.stages[1].batch_mode, NativeBatchMode::Continuous);
        assert_eq!(contract.stages[1].max_batch_size, 8);
        assert_eq!(contract.stages[1].max_work_units, 8);
        assert!(contract
            .stages
            .iter()
            .all(|stage| { stage.output_visibility == OutputVisibility::AfterQuantumCommit }));
        assert_eq!(
            contract.stages[1].max_workspace_bytes,
            CONTINUOUS_CHAT_MAX_BATCH_WORKSPACE_BYTES
        );
    }

    #[test]
    fn continuous_chat_stage_reserves_the_bounded_solo_speculation_quantum() {
        let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(1, 1).unwrap();
        let draft = LoadedModelBundleDraft::build(
            &registry,
            ExecutionGroupId::new(1),
            ModelInstanceId::new(2),
            ModelVariant::Qwen3827BFp8,
            BackendKind::Cpu,
        )
        .unwrap();
        let contract = draft
            .capabilities
            .get(&CapabilityKind::Chat)
            .unwrap()
            .contract(StreamingRequirements::native(true))
            .unwrap();

        assert_eq!(contract.stages[1].max_batch_size, 1);
        assert_eq!(
            contract.stages[1].max_work_units,
            CONTINUOUS_CHAT_MAX_DECODE_QUANTUM
        );
    }
}
