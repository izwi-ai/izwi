use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use crate::backends::BackendKind;
use crate::engine::{
    continuous_asr_workspace_per_row_bytes, continuous_chat_workspace_per_row,
    ManagedKvModelRuntime,
};
use crate::engine::{
    AdapterAbiRevision, AdapterInstanceId, CacheMode, CancellationGranularity,
    ClockedStateSelection, ConcurrencyClass, ExecutionAdapterBinding, ExecutionGroupId,
    ExecutionMode, ExecutionProfile, MembershipSafePoint, ModelInstanceId, NativeBatchMode,
    OutputVisibility, PhysicalLaunchPolicy, PrefillMode, StageDescriptor, StageId,
    StageProgressKind, StageShapePolicy, StageWorkSelector,
};
use crate::error::{Error, Result};
use crate::kv::v2::{
    stage_graph_fingerprint, CapabilityStateDescriptorV2, CapabilityStateRuntimeV2,
    InferenceStateContract, InvocationCapabilityRuntimeV2, InvocationWorkspaceRuntimeV2,
    ManagedCapabilityRuntimeV2, RetainedStateCapability, RetainedStateRuntimeV2,
    RetainedStateUseV2, StateClock, StatelessCapabilityRuntimeV2,
};
use crate::model::ModelVariant;

use super::table::{table_selects, LoadedAdapterKind};
use super::{
    scalar_execution_profile, AdapterMetadata, CapabilityKind, InferenceStateRequirement,
    RuntimeAdapterRegistry, StreamingMode,
};

const SCALAR_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(11);
const STATIC_TENSOR_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(12);
const CONTINUOUS_TENSOR_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(29);
const NEMOTRON_REALTIME_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(25);
const CONTINUOUS_ASR_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(15);
const CONTINUOUS_TTS_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(16);
const WHISPER_ASR_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(17);
const VIBEVOICE_ASR_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(19);
const GRANITE_SPEECH_ASR_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(22);
const LFM25_AUDIO_ASR_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(23);
const LFM25_AUDIO_TTS_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(24);
const VIBEVOICE_TTS_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(26);
const FISH_S2_TTS_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(30);
const VOXTRAL_TTS_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(28);
const PARAKEET_ASR_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(28);
pub(crate) const VOXTRAL_REALTIME_ADAPTER_ABI: AdapterAbiRevision =
    AdapterAbiRevision::new(crate::models::architectures::voxtral::VOXTRAL_REALTIME_EXECUTION_ABI);
// Architecture ceiling: 8,192 codec frames, 1,920 output samples/frame,
// f32-width intermediates, and up to 1,024 simultaneous channel-equivalents.
// Exact request/model-derived costs remain smaller and are installed at
// preparation; this ceiling prevents the adapter from rejecting production
// geometry before physical capacity admission.
const STATIC_TTS_MAX_BATCH_WORKSPACE_BYTES: u64 = 8 * 8_192 * 1_920 * 4 * 1_024;
const CONTINUOUS_ASR_MAX_BATCH_WORKSPACE_BYTES: u64 = 16 * 1024 * 1024;
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
    /// Exact ASR media route, independent of transport streaming. This bit is
    /// part of stage-graph/state identity so long-form atomic work cannot
    /// acquire the retained sequence runtime.
    pub(crate) asr_long_form: bool,
}

impl StreamingRequirements {
    pub(crate) const NONE: Self = Self {
        transport_output: false,
        model_native: false,
        asr_long_form: false,
    };

    pub(crate) const fn native(required: bool) -> Self {
        Self {
            transport_output: required,
            model_native: required,
            asr_long_form: false,
        }
    }

    pub(crate) const fn transport_only() -> Self {
        Self {
            transport_output: true,
            model_native: false,
            asr_long_form: false,
        }
    }

    pub(crate) const fn with_asr_long_form(mut self, required: bool) -> Self {
        self.asr_long_form = required;
        self
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

/// Make an atomic long-form ASR stage dispatch one chunk at a time when its
/// family does (`ModelFamily::dispatches_long_form_asr_by_chunk`): it is
/// renamed `chunked_name` and declares iterative progress with a safe point
/// between chunks, so the job yields after each chunk and the scheduler
/// dispatches the next one, and other work runs on the device between the
/// chunks of one file. Other families keep one dispatch for the whole file.
fn dispatch_long_form_asr_by_chunk(
    stage: &mut StageDescriptor,
    metadata: &AdapterMetadata,
    chunked_name: &str,
) {
    if metadata
        .model_variant
        .family()
        .dispatches_long_form_asr_by_chunk()
    {
        stage.name = chunked_name.to_string();
        stage.progress = StageProgressKind::Iterative;
        stage.membership_safe_point = MembershipSafePoint::QuantumBoundary;
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

    fn seal_fish_capacity(
        &self,
        _capacity: crate::runtime::fish_capacity::ResolvedFishServingCapacity,
    ) -> Result<()> {
        Err(Error::ModelLoadError(
            "loaded adapter does not support Fish capacity sealing".into(),
        ))
    }

    fn seal_chat_workspace(&self, _accelerator_bytes: u64) -> Result<()> {
        Err(Error::ModelLoadError(
            "loaded adapter does not support continuous chat workspace sealing".into(),
        ))
    }

    /// Bound and price a chat model's prefill: each span is at most
    /// `max_span_tokens`, and each token costs `per_token_bytes` of
    /// activation workspace. Unsealed adapters keep unbounded, unpriced
    /// prefill.
    fn seal_chat_prefill_workspace(&self, _per_token_bytes: u64, _max_span_tokens: u64) -> Result<()> {
        Err(Error::ModelLoadError(
            "loaded adapter does not support chat prefill workspace sealing".into(),
        ))
    }

    /// The loaded model's preferred decode quantum: the decode stage's work
    /// budget must admit an isolated row's whole quantum, or every quantum
    /// step fails its work-budget check and rolls back.
    fn seal_chat_decode_quantum(&self, _tokens: u64) -> Result<()> {
        Err(Error::ModelLoadError(
            "loaded adapter does not support chat decode quantum sealing".into(),
        ))
    }

    /// Publish one mixed prefill+decode stage of `token_budget` tokens in
    /// place of the split stages. Sealed only for a loaded model that runs
    /// mixed steps on a backend where its family is certified.
    fn seal_chat_mixed_step(&self, _token_budget: u64) -> Result<()> {
        Err(Error::ModelLoadError(
            "loaded adapter does not support mixed chat steps".into(),
        ))
    }

    fn seal_qwen3_asr_audio_preparation(
        &self,
        _model: &crate::models::architectures::qwen3::asr::Qwen3AsrModel,
    ) -> Result<()> {
        Err(Error::ModelLoadError(
            "loaded adapter does not own Qwen3 ASR audio preparation".into(),
        ))
    }

    fn seal_whisper_audio_preparation(
        &self,
        _model: &crate::models::architectures::whisper::asr::WhisperTurboAsrModel,
    ) -> Result<()> {
        Err(Error::ModelLoadError(
            "loaded adapter does not own Whisper audio preparation".into(),
        ))
    }

    fn seal_vibevoice_asr_preparation(
        &self,
        _model: &crate::models::architectures::vibevoice::asr::VibeVoiceAsrModel,
    ) -> Result<()> {
        Err(Error::ModelLoadError(
            "loaded adapter does not own VibeVoice ASR preparation".into(),
        ))
    }

    fn seal_granite_speech_asr_preparation(
        &self,
        _model: &crate::models::architectures::granite_speech::asr::GraniteSpeechAsrModel,
    ) -> Result<()> {
        Err(Error::ModelLoadError(
            "loaded adapter does not own Granite Speech ASR preparation".into(),
        ))
    }

    fn seal_lfm25_audio_asr_preparation(
        &self,
        _model: &crate::models::registry::NativeAudioChatModel,
    ) -> Result<()> {
        Err(Error::ModelLoadError(
            "loaded adapter does not own LFM2.5 Audio ASR preparation".into(),
        ))
    }

    fn seal_lfm25_audio_tts_preparation(
        &self,
        _model: &crate::models::registry::NativeAudioChatModel,
    ) -> Result<()> {
        Err(Error::ModelLoadError(
            "loaded adapter does not own LFM2.5 Audio TTS preparation".into(),
        ))
    }

    fn seal_voxtral_realtime_preparation(
        &self,
        _model: &crate::models::architectures::voxtral::realtime::VoxtralRealtimeModel,
    ) -> Result<()> {
        Err(Error::ModelLoadError(
            "loaded adapter does not own Voxtral realtime preparation".into(),
        ))
    }

    #[cfg(test)]
    fn install_test_preparation_seal(
        &self,
        _backend: BackendKind,
        _max_batch_size: usize,
    ) -> Result<()> {
        Ok(())
    }
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
            asr_long_form: false,
        });
        requirements.push(StreamingRequirements::native(true));
    }
    // Every ASR family enumerates its long-form variant: adapters that
    // implement the atomic graph produce it here, and adapters whose contract
    // ignores the flag re-seal an identical graph (consistent identities).
    if metadata.capability == CapabilityKind::Asr {
        let long_form = requirements
            .iter()
            .copied()
            .map(|requirements| requirements.with_asr_long_form(true))
            .collect::<Vec<_>>();
        requirements.extend(long_form);
    }
    // VibeVoice and Fish S2 TTS retain their public/direct atomic routes as
    // separately authenticated invocation graphs while normal requests use
    // retained sequence graphs. The internal long-form bit is only a graph-
    // enumeration discriminator here; request routing never labels TTS as ASR.
    if metadata.capability == CapabilityKind::Tts
        && matches!(
            metadata.model_variant.family(),
            crate::catalog::ModelFamily::VibeVoiceTts
        )
    {
        let atomic = requirements
            .iter()
            .copied()
            .map(|requirements| requirements.with_asr_long_form(true))
            .collect::<Vec<_>>();
        requirements.extend(atomic);
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
            RetainedStateUseV2::ExternalPaged | RetainedStateUseV2::ExternalPagedStatic
                if external_paged
        ) || matches!(retained_state_use, RetainedStateUseV2::Inactive if cacheless)
    };
    if !valid {
        return Err(Error::ModelLoadError(
            format!(
                "retained-state use {retained_state_use:?} does not match its physical backing (tensor_only={}) and exact execution profile (cache_mode={:?}, namespace={}, kv_dtype={})",
                retained.is_tensor_only(),
                profile.cache_mode,
                profile.cache_namespace.is_some(),
                profile.kv_dtype,
            ),
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

#[derive(Debug, Clone, Copy)]
struct PhysicalQwenTtsAdapterFactory;

impl LoadedExecutionAdapterFactory for PhysicalQwenTtsAdapterFactory {
    fn id(&self) -> &'static str {
        LoadedAdapterKind::QwenTtsSequence.factory_id()
    }

    fn batch_mode(&self) -> NativeBatchMode {
        NativeBatchMode::Continuous
    }

    fn supports(&self, metadata: AdapterMetadata, backend_kind: BackendKind) -> bool {
        table_selects(LoadedAdapterKind::QwenTtsSequence, metadata, backend_kind)
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
            context.max_tensor_batch_size,
        )))
    }
}

#[derive(Debug, Clone, Copy)]
struct NemotronRealtimeAdapterFactory;

#[derive(Debug, Clone, Copy)]
struct VoxtralRealtimeAdapterFactory;

impl LoadedExecutionAdapterFactory for VoxtralRealtimeAdapterFactory {
    fn id(&self) -> &'static str {
        LoadedAdapterKind::VoxtralRealtime.factory_id()
    }

    fn batch_mode(&self) -> NativeBatchMode {
        NativeBatchMode::Continuous
    }

    fn supports(&self, metadata: AdapterMetadata, backend_kind: BackendKind) -> bool {
        table_selects(LoadedAdapterKind::VoxtralRealtime, metadata, backend_kind)
    }

    fn create(
        &self,
        context: LoadedAdapterFactoryContext,
        metadata: AdapterMetadata,
    ) -> Result<Arc<dyn LoadedExecutionAdapter>> {
        Ok(Arc::new(VoxtralRealtimeExecutionAdapter::new(
            context.execution_group_id,
            context.model_instance_id,
            metadata,
            context.backend_kind,
            context.max_tensor_batch_size,
        )))
    }
}

impl LoadedExecutionAdapterFactory for NemotronRealtimeAdapterFactory {
    fn id(&self) -> &'static str {
        LoadedAdapterKind::NemotronRealtime.factory_id()
    }

    fn batch_mode(&self) -> NativeBatchMode {
        NativeBatchMode::Continuous
    }

    fn supports(&self, metadata: AdapterMetadata, backend_kind: BackendKind) -> bool {
        table_selects(LoadedAdapterKind::NemotronRealtime, metadata, backend_kind)
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
            context.max_tensor_batch_size,
        )))
    }
}

#[derive(Debug, Clone, Copy)]
struct ContinuousPhysicalChatAdapterFactory;

impl LoadedExecutionAdapterFactory for ContinuousPhysicalChatAdapterFactory {
    fn id(&self) -> &'static str {
        LoadedAdapterKind::ContinuousChat.factory_id()
    }

    fn batch_mode(&self) -> NativeBatchMode {
        NativeBatchMode::Continuous
    }

    fn supports(&self, metadata: AdapterMetadata, backend_kind: BackendKind) -> bool {
        table_selects(LoadedAdapterKind::ContinuousChat, metadata, backend_kind)
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
struct ContinuousPhysicalAsrAdapterFactory;

impl LoadedExecutionAdapterFactory for ContinuousPhysicalAsrAdapterFactory {
    fn id(&self) -> &'static str {
        LoadedAdapterKind::Qwen3AsrContinuous.factory_id()
    }

    fn batch_mode(&self) -> NativeBatchMode {
        NativeBatchMode::Continuous
    }

    fn supports(&self, metadata: AdapterMetadata, backend_kind: BackendKind) -> bool {
        table_selects(
            LoadedAdapterKind::Qwen3AsrContinuous,
            metadata,
            backend_kind,
        )
    }

    fn create(
        &self,
        context: LoadedAdapterFactoryContext,
        metadata: AdapterMetadata,
    ) -> Result<Arc<dyn LoadedExecutionAdapter>> {
        Ok(Arc::new(ContinuousAsrExecutionAdapter::new(
            context.execution_group_id,
            context.model_instance_id,
            metadata,
            context.backend_kind,
            context.max_tensor_batch_size,
        )))
    }
}

#[derive(Debug, Clone, Copy)]
struct ScalarExecutionAdapterFactory;

#[derive(Debug, Clone, Copy)]
struct WhisperPhysicalAsrAdapterFactory;

#[derive(Debug, Clone, Copy)]
struct VibeVoicePhysicalAsrAdapterFactory;

#[derive(Debug, Clone, Copy)]
struct GraniteSpeechPhysicalAsrAdapterFactory;

#[derive(Debug, Clone, Copy)]
struct Lfm25AudioPhysicalAsrAdapterFactory;

#[derive(Debug, Clone, Copy)]
struct ParakeetPhysicalAsrAdapterFactory;

#[derive(Debug, Clone, Copy)]
struct Lfm25AudioPhysicalTtsAdapterFactory;

#[derive(Debug, Clone, Copy)]
struct VibeVoicePhysicalTtsAdapterFactory;

#[derive(Debug, Clone, Copy)]
struct FishS2PhysicalTtsAdapterFactory;

#[derive(Debug, Clone, Copy)]
struct VoxtralPhysicalTtsAdapterFactory;

#[derive(Debug, Clone, Copy)]
struct KokoroStaticTtsAdapterFactory;

impl LoadedExecutionAdapterFactory for KokoroStaticTtsAdapterFactory {
    fn id(&self) -> &'static str {
        LoadedAdapterKind::KokoroStaticTts.factory_id()
    }

    fn batch_mode(&self) -> NativeBatchMode {
        NativeBatchMode::Static
    }

    fn supports(&self, metadata: AdapterMetadata, backend_kind: BackendKind) -> bool {
        table_selects(LoadedAdapterKind::KokoroStaticTts, metadata, backend_kind)
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

impl LoadedExecutionAdapterFactory for VoxtralPhysicalTtsAdapterFactory {
    fn id(&self) -> &'static str {
        LoadedAdapterKind::VoxtralTts.factory_id()
    }
    fn batch_mode(&self) -> NativeBatchMode {
        NativeBatchMode::Continuous
    }
    fn supports(&self, metadata: AdapterMetadata, backend_kind: BackendKind) -> bool {
        table_selects(LoadedAdapterKind::VoxtralTts, metadata, backend_kind)
    }
    fn create(
        &self,
        context: LoadedAdapterFactoryContext,
        metadata: AdapterMetadata,
    ) -> Result<Arc<dyn LoadedExecutionAdapter>> {
        Ok(Arc::new(VoxtralTtsExecutionAdapter::new(
            context.execution_group_id,
            context.model_instance_id,
            metadata,
            context.backend_kind,
            context.max_tensor_batch_size,
        )))
    }
}

impl LoadedExecutionAdapterFactory for FishS2PhysicalTtsAdapterFactory {
    fn id(&self) -> &'static str {
        LoadedAdapterKind::FishS2Tts.factory_id()
    }

    fn batch_mode(&self) -> NativeBatchMode {
        NativeBatchMode::Continuous
    }

    fn supports(&self, metadata: AdapterMetadata, backend_kind: BackendKind) -> bool {
        table_selects(LoadedAdapterKind::FishS2Tts, metadata, backend_kind)
    }

    fn create(
        &self,
        context: LoadedAdapterFactoryContext,
        metadata: AdapterMetadata,
    ) -> Result<Arc<dyn LoadedExecutionAdapter>> {
        Ok(Arc::new(FishS2TtsExecutionAdapter::new(
            context.execution_group_id,
            context.model_instance_id,
            metadata,
            context.backend_kind,
        )))
    }
}

impl LoadedExecutionAdapterFactory for VibeVoicePhysicalTtsAdapterFactory {
    fn id(&self) -> &'static str {
        LoadedAdapterKind::VibeVoiceTts.factory_id()
    }
    fn batch_mode(&self) -> NativeBatchMode {
        NativeBatchMode::Continuous
    }
    fn supports(&self, metadata: AdapterMetadata, backend_kind: BackendKind) -> bool {
        table_selects(LoadedAdapterKind::VibeVoiceTts, metadata, backend_kind)
    }
    fn create(
        &self,
        context: LoadedAdapterFactoryContext,
        metadata: AdapterMetadata,
    ) -> Result<Arc<dyn LoadedExecutionAdapter>> {
        Ok(Arc::new(VibeVoiceTtsExecutionAdapter::new(
            context.execution_group_id,
            context.model_instance_id,
            metadata,
            context.backend_kind,
            context.max_tensor_batch_size,
        )))
    }
}

impl LoadedExecutionAdapterFactory for Lfm25AudioPhysicalTtsAdapterFactory {
    fn id(&self) -> &'static str {
        LoadedAdapterKind::Lfm25AudioTts.factory_id()
    }

    fn batch_mode(&self) -> NativeBatchMode {
        NativeBatchMode::Continuous
    }

    fn supports(&self, metadata: AdapterMetadata, backend_kind: BackendKind) -> bool {
        table_selects(LoadedAdapterKind::Lfm25AudioTts, metadata, backend_kind)
    }

    fn create(
        &self,
        context: LoadedAdapterFactoryContext,
        metadata: AdapterMetadata,
    ) -> Result<Arc<dyn LoadedExecutionAdapter>> {
        Ok(Arc::new(Lfm25AudioTtsExecutionAdapter::new(
            context.execution_group_id,
            context.model_instance_id,
            metadata,
            context.backend_kind,
            context.max_tensor_batch_size,
        )))
    }
}

impl LoadedExecutionAdapterFactory for Lfm25AudioPhysicalAsrAdapterFactory {
    fn id(&self) -> &'static str {
        LoadedAdapterKind::Lfm25AudioAsr.factory_id()
    }

    fn batch_mode(&self) -> NativeBatchMode {
        NativeBatchMode::Continuous
    }

    fn supports(&self, metadata: AdapterMetadata, backend_kind: BackendKind) -> bool {
        table_selects(LoadedAdapterKind::Lfm25AudioAsr, metadata, backend_kind)
    }

    fn create(
        &self,
        context: LoadedAdapterFactoryContext,
        metadata: AdapterMetadata,
    ) -> Result<Arc<dyn LoadedExecutionAdapter>> {
        Ok(Arc::new(Lfm25AudioAsrExecutionAdapter::new(
            context.execution_group_id,
            context.model_instance_id,
            metadata,
            context.backend_kind,
            context.max_tensor_batch_size,
        )))
    }
}

impl LoadedExecutionAdapterFactory for ParakeetPhysicalAsrAdapterFactory {
    fn id(&self) -> &'static str {
        LoadedAdapterKind::ParakeetAsr.factory_id()
    }

    fn batch_mode(&self) -> NativeBatchMode {
        NativeBatchMode::Continuous
    }

    fn supports(&self, metadata: AdapterMetadata, backend_kind: BackendKind) -> bool {
        table_selects(LoadedAdapterKind::ParakeetAsr, metadata, backend_kind)
    }

    fn create(
        &self,
        context: LoadedAdapterFactoryContext,
        metadata: AdapterMetadata,
    ) -> Result<Arc<dyn LoadedExecutionAdapter>> {
        Ok(Arc::new(ParakeetAsrExecutionAdapter::new(
            context.execution_group_id,
            context.model_instance_id,
            metadata,
            context.backend_kind,
            context.max_tensor_batch_size,
        )))
    }
}

impl LoadedExecutionAdapterFactory for GraniteSpeechPhysicalAsrAdapterFactory {
    fn id(&self) -> &'static str {
        LoadedAdapterKind::GraniteSpeechAsr.factory_id()
    }

    fn batch_mode(&self) -> NativeBatchMode {
        NativeBatchMode::Continuous
    }

    fn supports(&self, metadata: AdapterMetadata, backend_kind: BackendKind) -> bool {
        table_selects(LoadedAdapterKind::GraniteSpeechAsr, metadata, backend_kind)
    }

    fn create(
        &self,
        context: LoadedAdapterFactoryContext,
        metadata: AdapterMetadata,
    ) -> Result<Arc<dyn LoadedExecutionAdapter>> {
        Ok(Arc::new(GraniteSpeechAsrExecutionAdapter::new(
            context.execution_group_id,
            context.model_instance_id,
            metadata,
            context.backend_kind,
            context.max_tensor_batch_size,
        )))
    }
}

impl LoadedExecutionAdapterFactory for VibeVoicePhysicalAsrAdapterFactory {
    fn id(&self) -> &'static str {
        LoadedAdapterKind::VibeVoiceAsr.factory_id()
    }

    fn batch_mode(&self) -> NativeBatchMode {
        NativeBatchMode::Continuous
    }

    fn supports(&self, metadata: AdapterMetadata, backend_kind: BackendKind) -> bool {
        table_selects(LoadedAdapterKind::VibeVoiceAsr, metadata, backend_kind)
    }

    fn create(
        &self,
        context: LoadedAdapterFactoryContext,
        metadata: AdapterMetadata,
    ) -> Result<Arc<dyn LoadedExecutionAdapter>> {
        Ok(Arc::new(VibeVoiceAsrExecutionAdapter::new(
            context.execution_group_id,
            context.model_instance_id,
            metadata,
            context.backend_kind,
            context.max_tensor_batch_size,
        )))
    }
}

impl LoadedExecutionAdapterFactory for WhisperPhysicalAsrAdapterFactory {
    fn id(&self) -> &'static str {
        LoadedAdapterKind::WhisperAsr.factory_id()
    }

    fn batch_mode(&self) -> NativeBatchMode {
        NativeBatchMode::Static
    }

    fn supports(&self, metadata: AdapterMetadata, backend_kind: BackendKind) -> bool {
        table_selects(LoadedAdapterKind::WhisperAsr, metadata, backend_kind)
    }

    fn create(
        &self,
        context: LoadedAdapterFactoryContext,
        metadata: AdapterMetadata,
    ) -> Result<Arc<dyn LoadedExecutionAdapter>> {
        Ok(Arc::new(WhisperAsrExecutionAdapter::new(
            context.execution_group_id,
            context.model_instance_id,
            metadata,
            context.backend_kind,
            context.max_tensor_batch_size,
        )))
    }
}

impl LoadedExecutionAdapterFactory for ScalarExecutionAdapterFactory {
    fn id(&self) -> &'static str {
        LoadedAdapterKind::Scalar.factory_id()
    }

    fn batch_mode(&self) -> NativeBatchMode {
        NativeBatchMode::None
    }

    fn supports(&self, metadata: AdapterMetadata, backend_kind: BackendKind) -> bool {
        table_selects(LoadedAdapterKind::Scalar, metadata, backend_kind)
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
        Arc::new(VoxtralRealtimeAdapterFactory),
        Arc::new(ContinuousPhysicalChatAdapterFactory),
        Arc::new(ContinuousPhysicalAsrAdapterFactory),
        Arc::new(WhisperPhysicalAsrAdapterFactory),
        Arc::new(VibeVoicePhysicalAsrAdapterFactory),
        Arc::new(GraniteSpeechPhysicalAsrAdapterFactory),
        Arc::new(Lfm25AudioPhysicalAsrAdapterFactory),
        Arc::new(ParakeetPhysicalAsrAdapterFactory),
        Arc::new(Lfm25AudioPhysicalTtsAdapterFactory),
        Arc::new(VibeVoicePhysicalTtsAdapterFactory),
        Arc::new(FishS2PhysicalTtsAdapterFactory),
        Arc::new(VoxtralPhysicalTtsAdapterFactory),
        Arc::new(KokoroStaticTtsAdapterFactory),
        Arc::new(ScalarExecutionAdapterFactory),
    ]
}

#[derive(Debug)]
struct VoxtralRealtimeExecutionAdapter {
    execution_group_id: ExecutionGroupId,
    model_instance_id: ModelInstanceId,
    adapter_instance_id: AdapterInstanceId,
    metadata: AdapterMetadata,
    backend_kind: BackendKind,
    max_batch_size: usize,
    preparation: OnceLock<
        crate::models::architectures::voxtral::realtime::VoxtralRealtimePreparationStageSeal,
    >,
}

impl VoxtralRealtimeExecutionAdapter {
    fn new(
        execution_group_id: ExecutionGroupId,
        model_instance_id: ModelInstanceId,
        metadata: AdapterMetadata,
        backend_kind: BackendKind,
        max_batch_size: usize,
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
            preparation: OnceLock::new(),
        }
    }

    fn install_preparation_seal(
        &self,
        seal: crate::models::architectures::voxtral::realtime::VoxtralRealtimePreparationStageSeal,
    ) -> Result<()> {
        if let Some(existing) = self.preparation.get() {
            return if existing == &seal {
                Ok(())
            } else {
                Err(Error::ModelLoadError(
                    "Voxtral realtime preparation was resealed with different geometry".into(),
                ))
            };
        }
        self.preparation.set(seal).map_err(|_| {
            Error::ModelLoadError("Voxtral realtime preparation seal raced publication".into())
        })
    }
}

impl LoadedExecutionAdapter for VoxtralRealtimeExecutionAdapter {
    fn metadata(&self) -> AdapterMetadata {
        self.metadata
    }

    fn adapter_instance_id(&self) -> AdapterInstanceId {
        self.adapter_instance_id
    }

    fn adapter_abi_revision(&self) -> AdapterAbiRevision {
        VOXTRAL_REALTIME_ADAPTER_ABI
    }

    fn seal_voxtral_realtime_preparation(
        &self,
        model: &crate::models::architectures::voxtral::realtime::VoxtralRealtimeModel,
    ) -> Result<()> {
        self.install_preparation_seal(model.realtime_preparation_stage_seal()?)
    }

    #[cfg(test)]
    fn install_test_preparation_seal(
        &self,
        _backend: BackendKind,
        max_batch_size: usize,
    ) -> Result<()> {
        if max_batch_size.max(1) != self.max_batch_size {
            return Err(Error::ModelLoadError(
                "Voxtral realtime test seal crossed its adapter batch width".into(),
            ));
        }
        self.install_preparation_seal(
            crate::models::architectures::voxtral::realtime::VoxtralRealtimePreparationStageSeal {
                max_source_samples: 32_000,
                max_work_units: 32_000,
                max_materialized_tensor_elements_per_row: 1_000_000,
                max_workspace_bytes: 4_000_000,
            },
        )
    }

    fn contract(&self, _streaming: StreamingRequirements) -> Result<LoadedExecutionContract> {
        let metadata = self.metadata();
        let mut execution_profile = scalar_execution_profile(metadata, self.backend_kind);
        execution_profile.mode = ExecutionMode::Realtime;
        execution_profile.prefill = PrefillMode::None;
        execution_profile.incremental_decode = true;
        execution_profile.prefill_batch = NativeBatchMode::Static;
        execution_profile.decode_batch = NativeBatchMode::Continuous;
        execution_profile.cache_mode = CacheMode::ExternalPaged;
        execution_profile.cache_namespace = Some(format!(
            "{}:{}:realtime-state-v2",
            metadata.model_variant,
            self.backend_kind.as_str()
        ));
        execution_profile.kv_dtype = "state_v2_resolved".into();
        execution_profile.cancellation = CancellationGranularity::RealtimeChunk;
        execution_profile.concurrency = ConcurrencyClass::Batchable;
        execution_profile.recompute_safe = false;
        execution_profile.cache_release_safe = true;
        execution_profile.prefix_reuse_safe = false;
        execution_profile.max_batch_size = self.max_batch_size;
        execution_profile.resolved_from_loaded_model = true;

        let seal = self.preparation.get().ok_or_else(|| {
            Error::ModelLoadError(
                "Voxtral realtime graph is unavailable before preparation geometry is sealed"
                    .into(),
            )
        })?;
        if seal.max_source_samples == 0
            || seal.max_work_units == 0
            || seal.max_materialized_tensor_elements_per_row == 0
            || seal.max_workspace_bytes == 0
        {
            return Err(Error::ModelLoadError(
                "Voxtral realtime preparation seal is not finite and positive".into(),
            ));
        }
        let width = u64::try_from(self.max_batch_size)
            .map_err(|_| Error::Overloaded("Voxtral realtime batch width exceeds u64".into()))?;
        let preparation_work = seal
            .max_work_units
            .checked_mul(width)
            .ok_or_else(|| Error::Overloaded("Voxtral preparation work ceiling overflow".into()))?;
        let preparation_workspace =
            seal.max_workspace_bytes.checked_mul(width).ok_or_else(|| {
                Error::Overloaded("Voxtral preparation workspace ceiling overflow".into())
            })?;

        let mut preparation = StageDescriptor::from_execution_profile(
            StageId::new(0),
            crate::models::architectures::voxtral::VOXTRAL_REALTIME_PREPARATION_STAGE,
            &execution_profile,
            NativeBatchMode::Static,
        );
        preparation.selector = StageWorkSelector::RealtimePreparation;
        preparation.max_batch_size = self.max_batch_size;
        preparation.max_work_units = preparation_work;
        preparation.workspace_per_row_bytes = seal.max_workspace_bytes;
        preparation.max_workspace_bytes = preparation_workspace;
        preparation.shape_policy = StageShapePolicy::Padded;
        preparation.output_visibility = OutputVisibility::AfterQuantumCommit;

        let mut prompt = StageDescriptor::from_execution_profile(
            StageId::new(1),
            crate::models::architectures::voxtral::VOXTRAL_REALTIME_PROMPT_PREFILL_STAGE,
            &execution_profile,
            NativeBatchMode::None,
        );
        prompt.selector = StageWorkSelector::RealtimePromptPrefill;
        prompt.max_batch_size = 1;
        prompt.max_work_units = seal.max_work_units;
        prompt.workspace_per_row_bytes = seal.max_workspace_bytes;
        prompt.max_workspace_bytes = seal.max_workspace_bytes;
        prompt.concurrency = ConcurrencyClass::Exclusive;
        prompt.shape_policy = StageShapePolicy::Exact;

        let mut decode = StageDescriptor::from_execution_profile(
            StageId::new(2),
            crate::models::architectures::voxtral::VOXTRAL_REALTIME_DECODE_STAGE,
            &execution_profile,
            NativeBatchMode::Continuous,
        );
        decode.selector = StageWorkSelector::RealtimeDecodeContinuation;
        decode.progress = StageProgressKind::Iterative;
        decode.membership_safe_point = MembershipSafePoint::QuantumBoundary;
        decode.max_batch_size = self.max_batch_size;
        decode.max_work_units = width;
        decode.workspace_per_row_bytes = seal.max_workspace_bytes;
        decode.max_workspace_bytes = preparation_workspace;
        decode.shape_policy = StageShapePolicy::Ragged;

        let mut completion = StageDescriptor::from_execution_profile(
            StageId::new(3),
            crate::models::architectures::voxtral::VOXTRAL_REALTIME_COMPLETION_STAGE,
            &execution_profile,
            NativeBatchMode::None,
        );
        completion.selector = StageWorkSelector::RealtimeCompletion;
        completion.max_batch_size = 1;
        completion.max_work_units = 1;
        completion.concurrency = ConcurrencyClass::Exclusive;
        completion.shape_policy = StageShapePolicy::Exact;

        preparation.validate()?;
        prompt.validate()?;
        decode.validate()?;
        completion.validate()?;

        Ok(LoadedExecutionContract {
            execution_group_id: self.execution_group_id,
            model_instance_id: self.model_instance_id,
            adapter_instance_id: self.adapter_instance_id(),
            adapter_abi_revision: self.adapter_abi_revision(),
            metadata,
            execution_profile,
            stages: Arc::from([preparation, prompt, decode, completion]),
        })
    }
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

    let mut execution_profile = scalar_execution_profile(metadata, backend_kind);
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
    if metadata.capability == CapabilityKind::SpeakerAttributedAsr
        && metadata.model_variant.family() == crate::catalog::ModelFamily::GraniteSpeechAsr
    {
        stage.selector = StageWorkSelector::Pipeline { ordinal: None };
    }
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
    max_batch_size: usize,
}

impl NemotronRealtimeExecutionAdapter {
    fn new(
        execution_group_id: ExecutionGroupId,
        model_instance_id: ModelInstanceId,
        metadata: AdapterMetadata,
        backend_kind: BackendKind,
        max_batch_size: usize,
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

    fn contract(&self, _streaming: StreamingRequirements) -> Result<LoadedExecutionContract> {
        let metadata = self.metadata();
        let mut execution_profile = scalar_execution_profile(metadata, self.backend_kind);
        execution_profile.mode = ExecutionMode::Realtime;
        execution_profile.prefill = PrefillMode::None;
        execution_profile.incremental_decode = true;
        execution_profile.prefill_batch = NativeBatchMode::None;
        execution_profile.decode_batch = NativeBatchMode::Continuous;
        execution_profile.cache_mode = CacheMode::None;
        execution_profile.cache_namespace = None;
        execution_profile.kv_dtype = "none".into();
        execution_profile.cancellation = CancellationGranularity::RealtimeChunk;
        execution_profile.concurrency = ConcurrencyClass::Batchable;
        execution_profile.recompute_safe = false;
        execution_profile.cache_release_safe = true;
        execution_profile.prefix_reuse_safe = false;
        execution_profile.max_batch_size = self.max_batch_size;
        execution_profile.resolved_from_loaded_model = true;

        let workspace =
            crate::models::architectures::nemotron::asr::NEMOTRON_REALTIME_STAGE_WORKSPACE_BYTES;
        let width = u64::try_from(self.max_batch_size)
            .map_err(|_| Error::Overloaded("Nemotron realtime batch width exceeds u64".into()))?;
        let batch_workspace = workspace.checked_mul(width).ok_or_else(|| {
            Error::Overloaded("Nemotron realtime batch workspace overflowed".into())
        })?;
        let max_samples = u64::try_from(
            crate::models::architectures::nemotron::asr::NemotronAsrModel::conservative_realtime_stream_resource_reservation(None, None, None)?.max_samples,
        )
        .map_err(|_| Error::Overloaded("Nemotron realtime sample ceiling exceeds u64".into()))?;

        let mut fallback = StageDescriptor::from_execution_profile(
            StageId::new(0),
            crate::models::architectures::nemotron::asr::NEMOTRON_REALTIME_FALLBACK_STAGE,
            &execution_profile,
            NativeBatchMode::None,
        );
        fallback.selector = StageWorkSelector::Atomic;
        fallback.max_batch_size = 1;
        fallback.max_work_units = max_samples;
        fallback.max_workspace_bytes = workspace;
        fallback.concurrency = ConcurrencyClass::Exclusive;
        fallback.shape_policy = StageShapePolicy::Exact;
        fallback.retained_state_selections = Some(vec![
            ClockedStateSelection::new(
                crate::models::architectures::nemotron::asr::NEMOTRON_ENCODER_STATE_GROUP,
                StateClock::Custom("realtime_operation_revision".into()),
            )?,
            ClockedStateSelection::new(
                crate::models::architectures::nemotron::asr::NEMOTRON_RNNT_STATE_GROUP,
                StateClock::DecoderTokens,
            )?,
        ]);
        fallback.output_visibility = OutputVisibility::AfterQuantumCommit;

        let mut encoder = StageDescriptor::from_execution_profile(
            StageId::new(1),
            crate::models::architectures::nemotron::asr::NEMOTRON_REALTIME_ENCODER_STAGE,
            &execution_profile,
            NativeBatchMode::None,
        );
        encoder.selector = StageWorkSelector::RealtimePreparation;
        encoder.max_batch_size = 1;
        encoder.max_work_units = max_samples;
        encoder.max_workspace_bytes = workspace;
        encoder.concurrency = ConcurrencyClass::Exclusive;
        encoder.shape_policy = StageShapePolicy::Exact;
        encoder.retained_state_selections = Some(vec![ClockedStateSelection::new(
            crate::models::architectures::nemotron::asr::NEMOTRON_ENCODER_STATE_GROUP,
            StateClock::Custom("realtime_operation_revision".into()),
        )?]);

        let mut rnnt = StageDescriptor::from_execution_profile(
            StageId::new(2),
            crate::models::architectures::nemotron::asr::NEMOTRON_REALTIME_RNNT_STAGE,
            &execution_profile,
            NativeBatchMode::Continuous,
        );
        rnnt.selector = StageWorkSelector::RealtimeDecodeContinuation;
        rnnt.progress = StageProgressKind::Iterative;
        rnnt.membership_safe_point = MembershipSafePoint::QuantumBoundary;
        rnnt.max_batch_size = self.max_batch_size;
        rnnt.max_work_units = width;
        rnnt.workspace_per_row_bytes = workspace;
        rnnt.max_workspace_bytes = batch_workspace;
        rnnt.shape_policy = StageShapePolicy::Ragged;
        rnnt.retained_state_selections = Some(vec![ClockedStateSelection::new(
            crate::models::architectures::nemotron::asr::NEMOTRON_RNNT_STATE_GROUP,
            StateClock::DecoderTokens,
        )?]);

        fallback.validate()?;
        encoder.validate()?;
        rnnt.validate()?;

        Ok(LoadedExecutionContract {
            execution_group_id: self.execution_group_id,
            model_instance_id: self.model_instance_id,
            adapter_instance_id: self.adapter_instance_id(),
            adapter_abi_revision: self.adapter_abi_revision(),
            metadata,
            execution_profile,
            stages: Arc::from([fallback, encoder, rnnt]),
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
    max_batch_size: usize,
}

impl PhysicalQwenTtsExecutionAdapter {
    fn new(
        execution_group_id: ExecutionGroupId,
        model_instance_id: ModelInstanceId,
        metadata: AdapterMetadata,
        backend_kind: BackendKind,
        max_batch_size: usize,
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

impl LoadedExecutionAdapter for PhysicalQwenTtsExecutionAdapter {
    fn metadata(&self) -> AdapterMetadata {
        self.metadata
    }

    fn adapter_instance_id(&self) -> AdapterInstanceId {
        self.adapter_instance_id
    }

    fn adapter_abi_revision(&self) -> AdapterAbiRevision {
        CONTINUOUS_TTS_ADAPTER_ABI
    }

    fn contract(&self, streaming: StreamingRequirements) -> Result<LoadedExecutionContract> {
        let metadata = self.metadata();
        if streaming.model_native && metadata.streaming_mode == StreamingMode::None {
            return Err(Error::InvalidInput(format!(
                "Model {} has no native streaming TTS contract",
                metadata.model_variant
            )));
        }
        let mut execution_profile = scalar_execution_profile(metadata, self.backend_kind);
        execution_profile.mode = ExecutionMode::Sequence;
        execution_profile.prefill = PrefillMode::Incremental;
        execution_profile.incremental_decode = true;
        execution_profile.prefill_batch = NativeBatchMode::None;
        execution_profile.decode_batch = NativeBatchMode::Continuous;
        execution_profile.cache_mode = CacheMode::ExternalPaged;
        execution_profile.cache_namespace = Some(format!(
            "{}:{}:{}:state-v2",
            metadata.model_variant,
            metadata.capability.as_str(),
            self.backend_kind.as_str()
        ));
        execution_profile.kv_dtype = "state_v2_resolved".to_string();
        execution_profile.cancellation = CancellationGranularity::SequenceStep;
        execution_profile.concurrency = ConcurrencyClass::Batchable;
        execution_profile.recompute_safe = true;
        execution_profile.cache_release_safe = true;
        execution_profile.prefix_reuse_safe = false;
        execution_profile.max_batch_size = self.max_batch_size;
        execution_profile.resolved_from_loaded_model = true;

        let mut prefill = StageDescriptor::from_execution_profile(
            StageId::new(1),
            "tts.prefill.physical",
            &execution_profile,
            NativeBatchMode::None,
        );
        prefill.selector = StageWorkSelector::SequencePrefill;
        prefill.max_batch_size = 1;
        prefill.concurrency = ConcurrencyClass::Exclusive;
        prefill.shape_policy = StageShapePolicy::Exact;
        prefill.max_workspace_bytes = STATIC_TTS_MAX_BATCH_WORKSPACE_BYTES;
        prefill.output_visibility = output_visibility_for(
            streaming.transport_output,
            execution_profile.mode,
            NativeBatchMode::None,
        );
        let mut decode = StageDescriptor::from_execution_profile(
            StageId::new(2),
            "tts.decode.physical",
            &execution_profile,
            NativeBatchMode::Continuous,
        );
        decode.selector = StageWorkSelector::SequenceDecode;
        // Predictor KV is load-owned typed invocation state, not scheduler
        // scratch. Its physical pool is authorized and charged by lifecycle.
        decode.max_work_units = u64::try_from(decode.max_batch_size).map_err(|_| {
            Error::Overloaded("continuous TTS batch width exceeds work accounting".to_string())
        })?;
        decode.max_workspace_bytes = STATIC_TTS_MAX_BATCH_WORKSPACE_BYTES;
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
        let mut execution_profile = scalar_execution_profile(metadata, self.backend_kind);
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

        let kokoro = self.metadata.model_variant.family() == crate::catalog::ModelFamily::KokoroTts;
        let mut stage = StageDescriptor::from_execution_profile(
            StageId::new(1),
            if kokoro {
                "tts.generate.kokoro.tensor_static"
            } else {
                "tts.generate.tensor_static"
            },
            &execution_profile,
            NativeBatchMode::Static,
        );
        stage.selector = StageWorkSelector::Atomic;
        stage.shape_policy = if kokoro {
            crate::engine::StageShapePolicy::Ragged
        } else {
            crate::engine::StageShapePolicy::Exact
        };
        stage.max_padding_basis_points = 0;
        stage.max_work_units = u64::try_from(stage.max_batch_size).map_err(|_| {
            Error::Overloaded("static TTS batch width exceeds work accounting".to_string())
        })?;
        // Kokoro is architecturally stateless/cacheless. Its request-shaped
        // synthesis workspace is admitted by RuntimeService and must not be
        // relabelled as load-owned invocation state by this stage contract.
        stage.max_workspace_bytes = if kokoro {
            0
        } else {
            STATIC_TTS_MAX_BATCH_WORKSPACE_BYTES
        };
        let mut scalar = StageDescriptor::from_execution_profile(
            StageId::new(0),
            if kokoro {
                "tts.generate.kokoro.scalar"
            } else {
                "tts.generate.scalar"
            },
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

        if kokoro {
            let mut preparation = StageDescriptor::from_execution_profile(
                StageId::new(2),
                "tts.prepare.kokoro",
                &execution_profile,
                NativeBatchMode::None,
            );
            preparation.selector = StageWorkSelector::PreSequencePreparation;
            preparation.max_batch_size = 1;
            preparation.max_work_units = 1;
            preparation.max_workspace_bytes = 0;
            preparation.concurrency = ConcurrencyClass::Exclusive;
            preparation.shape_policy = StageShapePolicy::Exact;
            preparation.validate()?;
            return Ok(LoadedExecutionContract {
                execution_group_id: self.execution_group_id,
                model_instance_id: self.model_instance_id,
                adapter_instance_id: self.adapter_instance_id(),
                adapter_abi_revision: self.adapter_abi_revision(),
                metadata,
                execution_profile,
                stages: Arc::from([preparation, stage, scalar]),
            });
        }

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
struct ContinuousAsrExecutionAdapter {
    execution_group_id: ExecutionGroupId,
    model_instance_id: ModelInstanceId,
    adapter_instance_id: AdapterInstanceId,
    metadata: AdapterMetadata,
    backend_kind: BackendKind,
    max_batch_size: usize,
    audio_preparation:
        OnceLock<crate::models::architectures::qwen3::asr::Qwen3AsrAudioPreparationStageSeal>,
}

impl ContinuousAsrExecutionAdapter {
    fn new(
        execution_group_id: ExecutionGroupId,
        model_instance_id: ModelInstanceId,
        metadata: AdapterMetadata,
        backend_kind: BackendKind,
        max_batch_size: usize,
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
            audio_preparation: OnceLock::new(),
        }
    }

    fn install_audio_preparation_seal(
        &self,
        seal: crate::models::architectures::qwen3::asr::Qwen3AsrAudioPreparationStageSeal,
    ) -> Result<()> {
        if let Some(existing) = self.audio_preparation.get() {
            return if existing == &seal {
                Ok(())
            } else {
                Err(Error::ModelLoadError(
                    "Qwen3 ASR audio preparation was resealed with different geometry".into(),
                ))
            };
        }
        self.audio_preparation.set(seal).map_err(|_| {
            Error::ModelLoadError("Qwen3 ASR audio preparation seal raced publication".into())
        })
    }
}

impl LoadedExecutionAdapter for ContinuousAsrExecutionAdapter {
    fn metadata(&self) -> AdapterMetadata {
        self.metadata
    }

    fn adapter_instance_id(&self) -> AdapterInstanceId {
        self.adapter_instance_id
    }

    fn adapter_abi_revision(&self) -> AdapterAbiRevision {
        CONTINUOUS_ASR_ADAPTER_ABI
    }

    fn seal_qwen3_asr_audio_preparation(
        &self,
        model: &crate::models::architectures::qwen3::asr::Qwen3AsrModel,
    ) -> Result<()> {
        let seal = model.audio_preparation_stage_seal(self.backend_kind, self.max_batch_size)?;
        self.install_audio_preparation_seal(seal)
    }

    #[cfg(test)]
    fn install_test_preparation_seal(
        &self,
        backend: BackendKind,
        max_batch_size: usize,
    ) -> Result<()> {
        self.install_audio_preparation_seal(
            crate::models::architectures::qwen3::asr::Qwen3AsrAudioPreparationStageSeal {
                backend,
                audio_dtype: "f32".into(),
                text_dtype: "f32".into(),
                max_batch_size,
                max_workspace_bytes: 64 * 1024 * 1024,
            },
        )
    }

    fn contract(&self, streaming: StreamingRequirements) -> Result<LoadedExecutionContract> {
        let metadata = self.metadata();
        if streaming.model_native && metadata.streaming_mode == StreamingMode::None {
            return Err(Error::InvalidInput(format!(
                "Model {} has no streaming ASR contract",
                metadata.model_variant
            )));
        }
        if streaming.asr_long_form {
            let mut execution_profile = scalar_execution_profile(metadata, self.backend_kind);
            execution_profile.mode = ExecutionMode::Atomic;
            execution_profile.prefill = PrefillMode::None;
            execution_profile.incremental_decode = false;
            execution_profile.prefill_batch = NativeBatchMode::None;
            execution_profile.decode_batch = NativeBatchMode::None;
            execution_profile.cache_mode = CacheMode::None;
            execution_profile.cache_namespace = None;
            execution_profile.kv_dtype = "none".to_string();
            execution_profile.cancellation = CancellationGranularity::OperationBoundary;
            execution_profile.concurrency = ConcurrencyClass::Exclusive;
            execution_profile.recompute_safe = false;
            execution_profile.cache_release_safe = false;
            execution_profile.prefix_reuse_safe = false;
            execution_profile.max_batch_size = 1;
            execution_profile.resolved_from_loaded_model = true;

            let mut stage = StageDescriptor::from_execution_profile(
                StageId::new(3),
                "asr.long_form.atomic",
                &execution_profile,
                NativeBatchMode::None,
            );
            stage.selector = StageWorkSelector::Atomic;
            stage.shape_policy = StageShapePolicy::Exact;
            stage.output_visibility = output_visibility_for(
                streaming.transport_output,
                execution_profile.mode,
                NativeBatchMode::None,
            );
            dispatch_long_form_asr_by_chunk(&mut stage, &metadata, "asr.long_form.chunked");
            stage.validate()?;
            return Ok(LoadedExecutionContract {
                execution_group_id: self.execution_group_id,
                model_instance_id: self.model_instance_id,
                adapter_instance_id: self.adapter_instance_id(),
                adapter_abi_revision: self.adapter_abi_revision(),
                metadata,
                execution_profile,
                stages: Arc::from([stage]),
            });
        }
        let mut execution_profile = scalar_execution_profile(metadata, self.backend_kind);
        let audio_preparation = self.audio_preparation.get().ok_or_else(|| {
            Error::ModelLoadError(
                "Qwen3 ASR normal execution graph is unavailable before loaded-model audio preparation is sealed"
                    .into(),
            )
        })?;
        if audio_preparation.backend != self.backend_kind
            || audio_preparation.max_batch_size != self.max_batch_size
            || audio_preparation.audio_dtype.is_empty()
            || audio_preparation.text_dtype.is_empty()
        {
            return Err(Error::ModelLoadError(
                "Qwen3 ASR audio preparation seal does not match its loaded adapter identity"
                    .into(),
            ));
        }
        execution_profile.mode = ExecutionMode::Sequence;
        execution_profile.prefill = PrefillMode::Incremental;
        execution_profile.incremental_decode = true;
        execution_profile.prefill_batch = NativeBatchMode::None;
        execution_profile.decode_batch = NativeBatchMode::Continuous;
        execution_profile.cache_mode = CacheMode::ExternalPaged;
        execution_profile.cache_namespace = Some(format!(
            "{}:{}:state-v2",
            metadata.model_variant,
            self.backend_kind.as_str()
        ));
        execution_profile.kv_dtype = "state_v2_resolved".to_string();
        execution_profile.cancellation = CancellationGranularity::SequenceStep;
        execution_profile.concurrency = ConcurrencyClass::Batchable;
        execution_profile.recompute_safe = true;
        execution_profile.cache_release_safe = true;
        execution_profile.prefix_reuse_safe = false;
        execution_profile.max_batch_size = self.max_batch_size;
        execution_profile.resolved_from_loaded_model = true;

        let mut preparation = StageDescriptor::from_execution_profile(
            StageId::new(0),
            "asr.encoder.audio",
            &execution_profile,
            NativeBatchMode::Static,
        );
        preparation.selector = StageWorkSelector::PreSequencePreparation;
        preparation.progress = StageProgressKind::Atomic;
        preparation.membership_safe_point = MembershipSafePoint::OperationBoundary;
        preparation.shape_policy = StageShapePolicy::Padded;
        preparation.max_workspace_bytes = audio_preparation.max_workspace_bytes;
        preparation.output_visibility = OutputVisibility::AfterQuantumCommit;

        let mut prefill = StageDescriptor::from_execution_profile(
            StageId::new(1),
            "asr.prefill.scalar",
            &execution_profile,
            NativeBatchMode::None,
        );
        prefill.selector = StageWorkSelector::SequencePrefill;
        prefill.max_batch_size = 1;
        prefill.concurrency = ConcurrencyClass::Exclusive;
        prefill.shape_policy = StageShapePolicy::Exact;
        prefill.output_visibility = output_visibility_for(
            streaming.transport_output,
            execution_profile.mode,
            NativeBatchMode::None,
        );

        let mut decode = StageDescriptor::from_execution_profile(
            StageId::new(2),
            "asr.decode.tensor_continuous",
            &execution_profile,
            NativeBatchMode::Continuous,
        );
        decode.selector = StageWorkSelector::SequenceDecode;
        decode.max_work_units = u64::try_from(decode.max_batch_size).map_err(|_| {
            Error::Overloaded("continuous ASR batch width exceeds work accounting".to_string())
        })?;
        decode.max_workspace_bytes = CONTINUOUS_ASR_MAX_BATCH_WORKSPACE_BYTES;
        preparation.validate()?;
        prefill.validate()?;
        decode.validate()?;

        Ok(LoadedExecutionContract {
            execution_group_id: self.execution_group_id,
            model_instance_id: self.model_instance_id,
            adapter_instance_id: self.adapter_instance_id(),
            adapter_abi_revision: self.adapter_abi_revision(),
            metadata,
            execution_profile,
            stages: Arc::from([preparation, prefill, decode]),
        })
    }
}

#[derive(Debug)]
struct VoxtralTtsExecutionAdapter {
    execution_group_id: ExecutionGroupId,
    model_instance_id: ModelInstanceId,
    adapter_instance_id: AdapterInstanceId,
    metadata: AdapterMetadata,
    backend_kind: BackendKind,
    max_batch_size: usize,
}

impl VoxtralTtsExecutionAdapter {
    fn new(
        execution_group_id: ExecutionGroupId,
        model_instance_id: ModelInstanceId,
        metadata: AdapterMetadata,
        backend_kind: BackendKind,
        max_batch_size: usize,
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

impl LoadedExecutionAdapter for VoxtralTtsExecutionAdapter {
    fn metadata(&self) -> AdapterMetadata {
        self.metadata
    }
    fn adapter_instance_id(&self) -> AdapterInstanceId {
        self.adapter_instance_id
    }
    fn adapter_abi_revision(&self) -> AdapterAbiRevision {
        VOXTRAL_TTS_ADAPTER_ABI
    }

    fn contract(&self, _streaming: StreamingRequirements) -> Result<LoadedExecutionContract> {
        let mut profile = scalar_execution_profile(self.metadata, self.backend_kind);
        profile.mode = ExecutionMode::Sequence;
        profile.prefill = PrefillMode::Incremental;
        profile.incremental_decode = true;
        profile.prefill_batch = NativeBatchMode::Static;
        profile.decode_batch = NativeBatchMode::Continuous;
        profile.cache_mode = CacheMode::ExternalPaged;
        profile.cache_namespace = Some(format!(
            "{}:tts:{}:voxtral-state-v2",
            self.metadata.model_variant,
            self.backend_kind.as_str()
        ));
        profile.kv_dtype = "state_v2_resolved".into();
        profile.cancellation = CancellationGranularity::SequenceStep;
        profile.concurrency = ConcurrencyClass::Batchable;
        profile.recompute_safe = true;
        profile.cache_release_safe = true;
        profile.max_batch_size = self.max_batch_size;
        profile.resolved_from_loaded_model = true;

        let mut preparation = StageDescriptor::from_execution_profile(
            StageId::new(0),
            "tts.prepare.voxtral",
            &profile,
            NativeBatchMode::None,
        );
        preparation.selector = StageWorkSelector::PreSequencePreparation;
        preparation.progress = StageProgressKind::Atomic;
        preparation.max_batch_size = 1;
        preparation.concurrency = ConcurrencyClass::Exclusive;
        preparation.shape_policy = StageShapePolicy::Exact;
        preparation.max_work_units = 16_384;
        preparation.max_workspace_bytes = 512 * 1024 * 1024;

        let mut prefill = StageDescriptor::from_execution_profile(
            StageId::new(1),
            "tts.prefill.voxtral.tensor_static",
            &profile,
            NativeBatchMode::Static,
        );
        prefill.selector = StageWorkSelector::SequencePrefill;
        prefill.progress = StageProgressKind::Iterative;
        prefill.shape_policy = StageShapePolicy::Ragged;
        prefill.max_padding_basis_points = 0;
        prefill.max_work_units = 16_384_u64.saturating_mul(self.max_batch_size as u64);
        prefill.max_workspace_bytes = 512_u64
            .saturating_mul(1024 * 1024)
            .saturating_mul(self.max_batch_size as u64);

        let mut decode = StageDescriptor::from_execution_profile(
            StageId::new(2),
            "tts.decode.voxtral.tensor_continuous",
            &profile,
            NativeBatchMode::Continuous,
        );
        decode.selector = StageWorkSelector::SequenceDecode;
        decode.progress = StageProgressKind::Iterative;
        decode.shape_policy = StageShapePolicy::Ragged;
        decode.max_work_units = 64_u64.saturating_mul(self.max_batch_size as u64);
        decode.max_workspace_bytes = 512_u64
            .saturating_mul(1024 * 1024)
            .saturating_mul(self.max_batch_size as u64);
        let mut finalize = StageDescriptor::from_execution_profile(
            StageId::new(3),
            "tts.codec.voxtral.scalar",
            &profile,
            NativeBatchMode::None,
        );
        finalize.selector = StageWorkSelector::SequenceFinalize;
        finalize.progress = StageProgressKind::Atomic;
        finalize.shape_policy = StageShapePolicy::Exact;
        finalize.max_batch_size = 1;
        finalize.concurrency = ConcurrencyClass::Exclusive;
        finalize.max_work_units = 1;
        finalize.max_workspace_bytes = 512 * 1024 * 1024;
        for stage in [&mut preparation, &mut prefill, &mut decode, &mut finalize] {
            stage.output_visibility = OutputVisibility::AfterQuantumCommit;
            stage.validate()?;
        }
        Ok(LoadedExecutionContract {
            execution_group_id: self.execution_group_id,
            model_instance_id: self.model_instance_id,
            adapter_instance_id: self.adapter_instance_id,
            adapter_abi_revision: self.adapter_abi_revision(),
            metadata: self.metadata,
            execution_profile: profile,
            stages: Arc::from([preparation, prefill, decode, finalize]),
        })
    }
}

#[derive(Debug)]
struct FishS2TtsExecutionAdapter {
    execution_group_id: ExecutionGroupId,
    model_instance_id: ModelInstanceId,
    adapter_instance_id: AdapterInstanceId,
    metadata: AdapterMetadata,
    backend_kind: BackendKind,
    capacity: OnceLock<crate::runtime::fish_capacity::ResolvedFishServingCapacity>,
}

impl FishS2TtsExecutionAdapter {
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
            capacity: OnceLock::new(),
        }
    }
}

impl LoadedExecutionAdapter for FishS2TtsExecutionAdapter {
    fn metadata(&self) -> AdapterMetadata {
        self.metadata
    }

    fn adapter_instance_id(&self) -> AdapterInstanceId {
        self.adapter_instance_id
    }

    fn adapter_abi_revision(&self) -> AdapterAbiRevision {
        FISH_S2_TTS_ADAPTER_ABI
    }

    fn seal_fish_capacity(
        &self,
        capacity: crate::runtime::fish_capacity::ResolvedFishServingCapacity,
    ) -> Result<()> {
        if let Some(existing) = self.capacity.get() {
            return if existing == &capacity {
                Ok(())
            } else {
                Err(Error::ModelLoadError(
                    "Fish serving capacity cannot change after sealing".into(),
                ))
            };
        }
        self.capacity
            .set(capacity)
            .map_err(|_| Error::ModelLoadError("Fish capacity seal raced publication".into()))
    }

    fn contract(&self, streaming: StreamingRequirements) -> Result<LoadedExecutionContract> {
        use crate::models::architectures::fish_s2::{
            FISH_S2_SLOW_STATE_GROUP, FISH_S2_TTS_DECODE_STAGE, FISH_S2_TTS_PREFILL_STAGE,
            FISH_S2_TTS_PREPARATION_STAGE,
        };
        if streaming.asr_long_form {
            return Err(Error::InvalidInput(
                "Fish S2 supports retained TTS execution only".into(),
            ));
        }

        let capacity = self
            .capacity
            .get()
            .cloned()
            .unwrap_or(crate::runtime::fish_capacity::ResolvedFishServingCapacity::scalar()?);
        let native = if capacity.native_batching {
            NativeBatchMode::Continuous
        } else {
            NativeBatchMode::None
        };
        let mut profile = scalar_execution_profile(self.metadata, self.backend_kind);
        profile.mode = ExecutionMode::Sequence;
        profile.prefill = PrefillMode::Incremental;
        profile.incremental_decode = true;
        profile.decode_batch = native;
        profile.cache_mode = CacheMode::ExternalPaged;
        profile.cache_namespace = Some(format!(
            "{}:tts:{}:fish-s2-state-v2",
            self.metadata.model_variant,
            self.backend_kind.as_str()
        ));
        profile.kv_dtype = "state_v2_resolved".into();
        profile.cancellation = CancellationGranularity::SequenceStep;
        profile.concurrency = ConcurrencyClass::Batchable;
        profile.recompute_safe = true;
        profile.cache_release_safe = true;
        profile.max_batch_size = capacity.ar_rows;
        profile.resolved_from_loaded_model = true;

        let mut preparation = StageDescriptor::from_execution_profile(
            StageId::new(0),
            FISH_S2_TTS_PREPARATION_STAGE,
            &profile,
            NativeBatchMode::None,
        );
        preparation.selector = StageWorkSelector::PreSequencePreparation;
        preparation.progress = StageProgressKind::Atomic;
        preparation.concurrency = ConcurrencyClass::Exclusive;
        preparation.shape_policy = StageShapePolicy::Exact;
        preparation.max_batch_size = 1;
        preparation.max_work_units = ModelVariant::FISH_S2_PRO_NATIVE_CONTEXT_TOKENS as u64;
        preparation.max_workspace_bytes =
            crate::models::architectures::fish_s2::codec::maximum_preparation_workspace_bytes()?;

        let mut prefill = StageDescriptor::from_execution_profile(
            StageId::new(1),
            FISH_S2_TTS_PREFILL_STAGE,
            &profile,
            if capacity.native_batching {
                NativeBatchMode::Static
            } else {
                NativeBatchMode::None
            },
        );
        prefill.selector = StageWorkSelector::SequencePrefill;
        prefill.concurrency = if capacity.native_batching {
            ConcurrencyClass::Batchable
        } else {
            ConcurrencyClass::Exclusive
        };
        prefill.shape_policy = StageShapePolicy::Ragged;
        prefill.max_batch_size = capacity.ar_rows.min(capacity.prefill_tokens as usize);
        prefill.max_work_units = capacity.prefill_tokens;
        prefill.workspace_per_row_bytes = 0;
        prefill.workspace_per_work_unit_bytes = capacity.prefill_workspace_per_token;
        prefill.max_workspace_bytes = capacity
            .ar_workspace_per_row
            .saturating_mul(prefill.max_batch_size as u64);
        prefill.retained_state_selections = Some(vec![ClockedStateSelection::new(
            FISH_S2_SLOW_STATE_GROUP,
            StateClock::DecoderTokens,
        )?]);

        let mut decode = StageDescriptor::from_execution_profile(
            StageId::new(2),
            FISH_S2_TTS_DECODE_STAGE,
            &profile,
            native,
        );
        decode.selector = StageWorkSelector::SequenceDecode;
        decode.concurrency = if capacity.native_batching {
            ConcurrencyClass::Batchable
        } else {
            ConcurrencyClass::Exclusive
        };
        decode.shape_policy = StageShapePolicy::Ragged;
        decode.max_work_units = capacity.ar_rows as u64;
        decode.workspace_per_row_bytes = capacity.ar_workspace_per_row;
        decode.max_workspace_bytes = capacity
            .ar_workspace_per_row
            .saturating_mul(capacity.ar_rows as u64);
        decode.retained_state_selections = Some(vec![ClockedStateSelection::new(
            FISH_S2_SLOW_STATE_GROUP,
            StateClock::DecoderTokens,
        )?]);
        let mut finalize = StageDescriptor::from_execution_profile(
            StageId::new(3),
            "tts.codec.fish_s2.scalar",
            &profile,
            NativeBatchMode::None,
        );
        finalize.selector = StageWorkSelector::SequenceFinalize;
        finalize.progress = StageProgressKind::Atomic;
        finalize.shape_policy = StageShapePolicy::Exact;
        finalize.max_batch_size = 1;
        finalize.concurrency = ConcurrencyClass::Exclusive;
        finalize.max_work_units = 1;
        finalize.max_workspace_bytes = 0;
        let mut audio_decode = StageDescriptor::from_execution_profile(
            StageId::new(4),
            crate::models::architectures::fish_s2::FISH_S2_TTS_AUDIO_DECODE_STAGE,
            &profile,
            if capacity.native_batching {
                NativeBatchMode::Static
            } else {
                NativeBatchMode::None
            },
        );
        audio_decode.max_batch_size = capacity.codec_rows;
        audio_decode.selector = StageWorkSelector::SequenceAudioDecode;
        audio_decode.progress = StageProgressKind::Iterative;
        audio_decode.shape_policy = StageShapePolicy::Exact;
        audio_decode.concurrency = if capacity.native_batching {
            ConcurrencyClass::Batchable
        } else {
            ConcurrencyClass::Exclusive
        };
        audio_decode.max_work_units =
            (crate::models::architectures::fish_s2::FISH_S2_AUDIO_CHUNK_FRAMES as u64)
                .saturating_mul(capacity.codec_rows as u64);
        audio_decode.workspace_per_row_bytes = capacity.codec_workspace_per_row;
        audio_decode.max_workspace_bytes = capacity
            .codec_workspace_per_row
            .saturating_mul(capacity.codec_rows as u64);
        for stage in [
            &mut preparation,
            &mut prefill,
            &mut decode,
            &mut finalize,
            &mut audio_decode,
        ] {
            stage.max_padding_basis_points = 0;
            stage.output_visibility = OutputVisibility::AfterQuantumCommit;
            stage.validate()?;
        }
        Ok(LoadedExecutionContract {
            execution_group_id: self.execution_group_id,
            model_instance_id: self.model_instance_id,
            adapter_instance_id: self.adapter_instance_id,
            adapter_abi_revision: self.adapter_abi_revision(),
            metadata: self.metadata,
            execution_profile: profile,
            stages: Arc::from([preparation, prefill, decode, finalize, audio_decode]),
        })
    }
}

#[derive(Debug)]
struct VibeVoiceTtsExecutionAdapter {
    execution_group_id: ExecutionGroupId,
    model_instance_id: ModelInstanceId,
    adapter_instance_id: AdapterInstanceId,
    metadata: AdapterMetadata,
    backend_kind: BackendKind,
    max_batch_size: usize,
}

impl VibeVoiceTtsExecutionAdapter {
    fn new(
        execution_group_id: ExecutionGroupId,
        model_instance_id: ModelInstanceId,
        metadata: AdapterMetadata,
        backend_kind: BackendKind,
        max_batch_size: usize,
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

impl LoadedExecutionAdapter for VibeVoiceTtsExecutionAdapter {
    fn metadata(&self) -> AdapterMetadata {
        self.metadata
    }
    fn adapter_instance_id(&self) -> AdapterInstanceId {
        self.adapter_instance_id
    }
    fn adapter_abi_revision(&self) -> AdapterAbiRevision {
        VIBEVOICE_TTS_ADAPTER_ABI
    }

    fn contract(&self, streaming: StreamingRequirements) -> Result<LoadedExecutionContract> {
        use crate::models::architectures::vibevoice::{
            VIBEVOICE_TTS_DECODE_STAGE, VIBEVOICE_TTS_LEGACY_STAGE, VIBEVOICE_TTS_PREFILL_STAGE,
            VIBEVOICE_TTS_PREPARATION_STAGE,
        };
        if streaming.asr_long_form {
            let mut profile = scalar_execution_profile(self.metadata, self.backend_kind);
            profile.mode = ExecutionMode::Atomic;
            profile.cache_mode = CacheMode::None;
            profile.cache_namespace = None;
            profile.kv_dtype = "none".into();
            profile.max_batch_size = 1;
            profile.resolved_from_loaded_model = true;
            let mut stage = StageDescriptor::from_execution_profile(
                StageId::new(0),
                VIBEVOICE_TTS_LEGACY_STAGE,
                &profile,
                NativeBatchMode::None,
            );
            stage.selector = StageWorkSelector::Atomic;
            stage.shape_policy = StageShapePolicy::Exact;
            stage.validate()?;
            return Ok(LoadedExecutionContract {
                execution_group_id: self.execution_group_id,
                model_instance_id: self.model_instance_id,
                adapter_instance_id: self.adapter_instance_id,
                adapter_abi_revision: self.adapter_abi_revision(),
                metadata: self.metadata,
                execution_profile: profile,
                stages: Arc::from([stage]),
            });
        }
        let mut profile = scalar_execution_profile(self.metadata, self.backend_kind);
        profile.mode = ExecutionMode::Sequence;
        profile.prefill = PrefillMode::Incremental;
        profile.incremental_decode = true;
        profile.decode_batch = NativeBatchMode::Continuous;
        profile.cache_mode = CacheMode::ExternalPaged;
        profile.cache_namespace = Some(format!(
            "{}:tts:{}:state-v2",
            self.metadata.model_variant,
            self.backend_kind.as_str()
        ));
        profile.kv_dtype = "state_v2_resolved".into();
        profile.cancellation = CancellationGranularity::SequenceStep;
        profile.concurrency = ConcurrencyClass::Batchable;
        profile.recompute_safe = true;
        profile.cache_release_safe = true;
        profile.max_batch_size = self.max_batch_size;
        profile.resolved_from_loaded_model = true;

        let mut preparation = StageDescriptor::from_execution_profile(
            StageId::new(0),
            VIBEVOICE_TTS_PREPARATION_STAGE,
            &profile,
            NativeBatchMode::None,
        );
        preparation.selector = StageWorkSelector::PreSequencePreparation;
        preparation.progress = StageProgressKind::Atomic;
        preparation.membership_safe_point = MembershipSafePoint::OperationBoundary;
        preparation.max_batch_size = 1;
        preparation.concurrency = ConcurrencyClass::Exclusive;
        preparation.shape_policy = StageShapePolicy::Exact;
        preparation.max_work_units = 16_384;
        preparation.max_workspace_bytes = 512 * 1024 * 1024;
        preparation.output_visibility = OutputVisibility::AfterQuantumCommit;

        let mut prefill = StageDescriptor::from_execution_profile(
            StageId::new(1),
            VIBEVOICE_TTS_PREFILL_STAGE,
            &profile,
            NativeBatchMode::None,
        );
        prefill.selector = StageWorkSelector::SequencePrefill;
        prefill.max_batch_size = 1;
        prefill.concurrency = ConcurrencyClass::Exclusive;
        prefill.shape_policy = StageShapePolicy::Exact;
        prefill.max_work_units = 16_384;
        prefill.max_workspace_bytes = 512 * 1024 * 1024;
        prefill.retained_state_selections = Some(vec![]);
        prefill.output_visibility = OutputVisibility::AfterQuantumCommit;

        let mut decode = StageDescriptor::from_execution_profile(
            StageId::new(2),
            VIBEVOICE_TTS_DECODE_STAGE,
            &profile,
            NativeBatchMode::Continuous,
        );
        decode.selector = StageWorkSelector::SequenceDecode;
        decode.shape_policy = StageShapePolicy::Ragged;
        decode.max_work_units = u64::try_from(self.max_batch_size)
            .map_err(|_| Error::Overloaded("VibeVoice TTS batch width exceeds u64".into()))?;
        decode.workspace_per_row_bytes = 512 * 1024 * 1024;
        decode.max_workspace_bytes = decode
            .workspace_per_row_bytes
            .checked_mul(decode.max_work_units)
            .ok_or_else(|| Error::Overloaded("VibeVoice TTS decode workspace overflow".into()))?;
        decode.retained_state_selections = Some(vec![ClockedStateSelection::new(
            crate::models::architectures::vibevoice::VIBEVOICE_TTS_TOKENIZER_GROUP,
            StateClock::CodecFrames,
        )?]);
        preparation.validate()?;
        prefill.validate()?;
        decode.validate()?;
        Ok(LoadedExecutionContract {
            execution_group_id: self.execution_group_id,
            model_instance_id: self.model_instance_id,
            adapter_instance_id: self.adapter_instance_id,
            adapter_abi_revision: self.adapter_abi_revision(),
            metadata: self.metadata,
            execution_profile: profile,
            stages: Arc::from([preparation, prefill, decode]),
        })
    }
}

#[derive(Debug)]
struct Lfm25AudioTtsExecutionAdapter {
    execution_group_id: ExecutionGroupId,
    model_instance_id: ModelInstanceId,
    adapter_instance_id: AdapterInstanceId,
    metadata: AdapterMetadata,
    backend_kind: BackendKind,
    max_batch_size: usize,
    ceiling: OnceLock<crate::models::architectures::lfm25_audio::model::Lfm25AudioTtsStageCeiling>,
}

impl Lfm25AudioTtsExecutionAdapter {
    fn new(
        execution_group_id: ExecutionGroupId,
        model_instance_id: ModelInstanceId,
        metadata: AdapterMetadata,
        backend_kind: BackendKind,
        max_batch_size: usize,
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
            ceiling: OnceLock::new(),
        }
    }
}

impl LoadedExecutionAdapter for Lfm25AudioTtsExecutionAdapter {
    fn metadata(&self) -> AdapterMetadata {
        self.metadata
    }
    fn adapter_instance_id(&self) -> AdapterInstanceId {
        self.adapter_instance_id
    }
    fn adapter_abi_revision(&self) -> AdapterAbiRevision {
        LFM25_AUDIO_TTS_ADAPTER_ABI
    }

    fn seal_lfm25_audio_tts_preparation(
        &self,
        model: &crate::models::registry::NativeAudioChatModel,
    ) -> Result<()> {
        let ceiling = model.lfm25_audio_tts_stage_ceiling()?;
        if let Some(existing) = self.ceiling.get() {
            return if *existing == ceiling {
                Ok(())
            } else {
                Err(Error::ModelLoadError(
                    "LFM2.5 Audio TTS adapter was resealed with different geometry".into(),
                ))
            };
        }
        self.ceiling.set(ceiling).map_err(|_| {
            Error::ModelLoadError(
                "LFM2.5 Audio TTS adapter preparation seal raced publication".into(),
            )
        })
    }

    #[cfg(test)]
    fn install_test_preparation_seal(
        &self,
        backend: BackendKind,
        _max_batch_size: usize,
    ) -> Result<()> {
        self.ceiling
            .set(
                crate::models::architectures::lfm25_audio::model::Lfm25AudioTtsStageCeiling {
                    backend,
                    max_prompt_tokens: 4_096,
                    max_codebooks: 8,
                    max_materialized_tensor_elements: 4_096 * 2_048,
                    max_retained_resident_bytes: 4_096 * 2_048 * 4,
                    max_preparation_workspace_bytes: 4_096 * 2_048 * 4,
                    max_workspace_bytes: 64 * 1024 * 1024,
                },
            )
            .map_err(|_| Error::ModelLoadError("test LFM TTS seal already installed".into()))
    }

    fn contract(&self, _streaming: StreamingRequirements) -> Result<LoadedExecutionContract> {
        let ceiling = self.ceiling.get().ok_or_else(|| {
            Error::ModelLoadError("LFM2.5 Audio TTS adapter is not preparation-sealed".into())
        })?;
        let mut profile = scalar_execution_profile(self.metadata, self.backend_kind);
        profile.mode = ExecutionMode::Sequence;
        profile.prefill = PrefillMode::Incremental;
        profile.incremental_decode = true;
        profile.prefill_batch = NativeBatchMode::Static;
        profile.decode_batch = NativeBatchMode::Continuous;
        profile.cache_mode = CacheMode::ExternalPaged;
        profile.cache_namespace = Some(format!(
            "{}:tts:{}:state-v2",
            self.metadata.model_variant,
            self.backend_kind.as_str()
        ));
        profile.kv_dtype = "state_v2_resolved".into();
        profile.cancellation = CancellationGranularity::SequenceStep;
        profile.concurrency = ConcurrencyClass::Batchable;
        profile.recompute_safe = true;
        profile.cache_release_safe = true;
        profile.max_batch_size = self.max_batch_size;
        profile.resolved_from_loaded_model = true;
        let mut preparation = StageDescriptor::from_execution_profile(
            StageId::new(0),
            "tts.prepare.lfm25_audio",
            &profile,
            NativeBatchMode::None,
        );
        preparation.selector = StageWorkSelector::PreSequencePreparation;
        preparation.max_batch_size = 1;
        preparation.concurrency = ConcurrencyClass::Exclusive;
        preparation.progress = StageProgressKind::Atomic;
        preparation.shape_policy = StageShapePolicy::Exact;
        preparation.max_work_units = u64::try_from(ceiling.max_prompt_tokens)
            .map_err(|_| Error::Overloaded("LFM2.5 Audio TTS prompt ceiling exceeds u64".into()))?;
        preparation.max_workspace_bytes = ceiling.max_preparation_workspace_bytes;
        preparation.output_visibility = OutputVisibility::AfterQuantumCommit;
        let mut prefill = StageDescriptor::from_execution_profile(
            StageId::new(1),
            "tts.prefill.lfm25_audio.tensor_static",
            &profile,
            NativeBatchMode::Static,
        );
        prefill.selector = StageWorkSelector::SequencePrefill;
        prefill.shape_policy = StageShapePolicy::Exact;
        prefill.max_padding_basis_points = 0;
        prefill.max_work_units = u64::try_from(ceiling.max_prompt_tokens)
            .map_err(|_| Error::Overloaded("LFM2.5 Audio TTS prompt ceiling exceeds u64".into()))?
            .checked_mul(u64::try_from(self.max_batch_size).map_err(|_| {
                Error::Overloaded("LFM2.5 Audio TTS batch ceiling exceeds u64".into())
            })?)
            .ok_or_else(|| Error::Overloaded("LFM2.5 Audio TTS prefill work overflow".into()))?;
        prefill.max_workspace_bytes = ceiling
            .max_workspace_bytes
            .checked_mul(u64::try_from(self.max_batch_size).map_err(|_| {
                Error::Overloaded("LFM2.5 Audio TTS batch ceiling exceeds u64".into())
            })?)
            .ok_or_else(|| {
                Error::Overloaded("LFM2.5 Audio TTS prefill workspace overflow".into())
            })?;
        prefill.output_visibility = OutputVisibility::AfterQuantumCommit;
        let mut decode = StageDescriptor::from_execution_profile(
            StageId::new(2),
            "tts.decode.lfm25_audio.tensor_continuous",
            &profile,
            NativeBatchMode::Continuous,
        );
        decode.selector = StageWorkSelector::SequenceDecode;
        decode.shape_policy = StageShapePolicy::Exact;
        decode.max_work_units = u64::try_from(ceiling.max_codebooks.saturating_add(1))
            .map_err(|_| Error::Overloaded("LFM2.5 Audio TTS codebook ceiling exceeds u64".into()))?
            .checked_mul(u64::try_from(self.max_batch_size).map_err(|_| {
                Error::Overloaded("LFM2.5 Audio TTS batch ceiling exceeds u64".into())
            })?)
            .ok_or_else(|| Error::Overloaded("LFM2.5 Audio TTS batch work overflow".into()))?;
        decode.max_workspace_bytes = ceiling
            .max_workspace_bytes
            .checked_mul(u64::try_from(self.max_batch_size).map_err(|_| {
                Error::Overloaded("LFM2.5 Audio TTS batch ceiling exceeds u64".into())
            })?)
            .ok_or_else(|| Error::Overloaded("LFM2.5 Audio TTS batch workspace overflow".into()))?;
        decode.output_visibility = OutputVisibility::AfterQuantumCommit;
        preparation.validate()?;
        prefill.validate()?;
        decode.validate()?;
        Ok(LoadedExecutionContract {
            execution_group_id: self.execution_group_id,
            model_instance_id: self.model_instance_id,
            adapter_instance_id: self.adapter_instance_id,
            adapter_abi_revision: self.adapter_abi_revision(),
            metadata: self.metadata,
            execution_profile: profile,
            stages: Arc::from([preparation, prefill, decode]),
        })
    }
}

#[derive(Debug)]
struct ParakeetAsrExecutionAdapter {
    execution_group_id: ExecutionGroupId,
    model_instance_id: ModelInstanceId,
    adapter_instance_id: AdapterInstanceId,
    metadata: AdapterMetadata,
    backend_kind: BackendKind,
    max_batch_size: usize,
}

impl ParakeetAsrExecutionAdapter {
    fn new(
        execution_group_id: ExecutionGroupId,
        model_instance_id: ModelInstanceId,
        metadata: AdapterMetadata,
        backend_kind: BackendKind,
        max_batch_size: usize,
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

impl LoadedExecutionAdapter for ParakeetAsrExecutionAdapter {
    fn metadata(&self) -> AdapterMetadata {
        self.metadata
    }

    fn adapter_instance_id(&self) -> AdapterInstanceId {
        self.adapter_instance_id
    }

    fn adapter_abi_revision(&self) -> AdapterAbiRevision {
        PARAKEET_ASR_ADAPTER_ABI
    }

    fn contract(&self, streaming: StreamingRequirements) -> Result<LoadedExecutionContract> {
        let metadata = self.metadata();
        if streaming.model_native {
            return Err(Error::InvalidInput(format!(
                "Model {} has no native streaming ASR contract",
                metadata.model_variant
            )));
        }
        // Parakeet does not implement the long-form atomic graph: its
        // retained predictor state and the scratch atomic workspace must be
        // published as separate load-sealed publications (parakeet physical
        // spec), which the single-publication load path does not support yet.
        // The flag is therefore ignored here and the pipeline's atomic
        // requirement fails closed at load with a capability error.
        let width = u64::try_from(self.max_batch_size)
            .map_err(|_| Error::Overloaded("Parakeet batch width exceeds u64".into()))?;
        let workspace_per_row =
            crate::models::architectures::parakeet::asr::PARAKEET_RETAINED_WORKSPACE_PER_ROW_BYTES;
        let batch_workspace = workspace_per_row
            .checked_mul(width)
            .ok_or_else(|| Error::Overloaded("Parakeet batch workspace overflowed".into()))?;

        let mut profile = scalar_execution_profile(metadata, self.backend_kind);
        profile.mode = ExecutionMode::Sequence;
        profile.prefill = PrefillMode::Incremental;
        profile.incremental_decode = true;
        profile.prefill_batch = NativeBatchMode::Static;
        profile.decode_batch = NativeBatchMode::Continuous;
        // Parakeet's rollback-safe recurrent state is retained by the exact
        // executor session; it has no attention pages or managed KV runtime.
        profile.cache_mode = CacheMode::None;
        profile.cache_namespace = None;
        profile.kv_dtype = "none".into();
        profile.cancellation = CancellationGranularity::SequenceStep;
        profile.concurrency = ConcurrencyClass::Batchable;
        profile.recompute_safe = true;
        profile.cache_release_safe = true;
        profile.prefix_reuse_safe = false;
        profile.max_batch_size = self.max_batch_size;
        profile.resolved_from_loaded_model = true;

        let mut preparation = StageDescriptor::from_execution_profile(
            StageId::new(0),
            "asr.encoder.parakeet",
            &profile,
            NativeBatchMode::None,
        );
        preparation.selector = StageWorkSelector::PreSequencePreparation;
        preparation.progress = StageProgressKind::Atomic;
        preparation.membership_safe_point = MembershipSafePoint::OperationBoundary;
        preparation.max_batch_size = 1;
        preparation.max_work_units = 64 * 1024 * 1024;
        preparation.max_workspace_bytes = 4 * 1024 * 1024 * 1024;
        preparation.concurrency = ConcurrencyClass::Exclusive;
        preparation.shape_policy = StageShapePolicy::Exact;
        preparation.output_visibility = OutputVisibility::AfterQuantumCommit;

        let selection = ClockedStateSelection::new(
            crate::models::architectures::parakeet::asr::PARAKEET_PREDICTOR_STATE_GROUP,
            StateClock::DecoderTokens,
        )?;
        let mut prefill = StageDescriptor::from_execution_profile(
            StageId::new(1),
            crate::models::architectures::parakeet::asr::PARAKEET_RETAINED_PREFILL_STAGE,
            &profile,
            NativeBatchMode::Static,
        );
        prefill.selector = StageWorkSelector::SequencePrefill;
        prefill.progress = StageProgressKind::Iterative;
        prefill.membership_safe_point = MembershipSafePoint::QuantumBoundary;
        prefill.max_work_units = width;
        prefill.workspace_per_row_bytes = workspace_per_row;
        prefill.max_workspace_bytes = batch_workspace;
        prefill.shape_policy = StageShapePolicy::Ragged;
        prefill.max_padding_basis_points = 0;
        prefill.retained_state_selections = Some(vec![selection.clone()]);
        prefill.output_visibility = OutputVisibility::AfterQuantumCommit;

        let mut decode = StageDescriptor::from_execution_profile(
            StageId::new(2),
            crate::models::architectures::parakeet::asr::PARAKEET_RETAINED_DECODE_STAGE,
            &profile,
            NativeBatchMode::Continuous,
        );
        decode.selector = StageWorkSelector::SequenceDecode;
        decode.progress = StageProgressKind::Iterative;
        decode.membership_safe_point = MembershipSafePoint::QuantumBoundary;
        decode.max_work_units = width;
        decode.workspace_per_row_bytes = workspace_per_row;
        decode.max_workspace_bytes = batch_workspace;
        decode.shape_policy = StageShapePolicy::Ragged;
        decode.max_padding_basis_points = 0;
        decode.retained_state_selections = Some(vec![selection]);
        decode.output_visibility = OutputVisibility::AfterQuantumCommit;

        preparation.validate()?;
        prefill.validate()?;
        decode.validate()?;
        Ok(LoadedExecutionContract {
            execution_group_id: self.execution_group_id,
            model_instance_id: self.model_instance_id,
            adapter_instance_id: self.adapter_instance_id(),
            adapter_abi_revision: self.adapter_abi_revision(),
            metadata,
            execution_profile: profile,
            stages: Arc::from([preparation, prefill, decode]),
        })
    }
}

#[derive(Debug)]
struct Lfm25AudioAsrExecutionAdapter {
    execution_group_id: ExecutionGroupId,
    model_instance_id: ModelInstanceId,
    adapter_instance_id: AdapterInstanceId,
    metadata: AdapterMetadata,
    backend_kind: BackendKind,
    max_batch_size: usize,
    preparation: OnceLock<
        crate::models::architectures::lfm25_audio::model::Lfm25AudioAsrPreparationStageCeiling,
    >,
}

impl Lfm25AudioAsrExecutionAdapter {
    fn new(
        execution_group_id: ExecutionGroupId,
        model_instance_id: ModelInstanceId,
        metadata: AdapterMetadata,
        backend_kind: BackendKind,
        max_batch_size: usize,
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
            preparation: OnceLock::new(),
        }
    }

    fn install_preparation_seal(
        &self,
        seal: crate::models::architectures::lfm25_audio::model::Lfm25AudioAsrPreparationStageCeiling,
    ) -> Result<()> {
        if let Some(existing) = self.preparation.get() {
            return if *existing == seal {
                Ok(())
            } else {
                Err(Error::ModelLoadError(
                    "LFM2.5 Audio ASR preparation was resealed with different geometry".into(),
                ))
            };
        }
        self.preparation.set(seal).map_err(|_| {
            Error::ModelLoadError("LFM2.5 Audio ASR preparation seal raced publication".into())
        })
    }
}

impl LoadedExecutionAdapter for Lfm25AudioAsrExecutionAdapter {
    fn metadata(&self) -> AdapterMetadata {
        self.metadata
    }

    fn adapter_instance_id(&self) -> AdapterInstanceId {
        self.adapter_instance_id
    }

    fn adapter_abi_revision(&self) -> AdapterAbiRevision {
        LFM25_AUDIO_ASR_ADAPTER_ABI
    }

    fn seal_lfm25_audio_asr_preparation(
        &self,
        model: &crate::models::registry::NativeAudioChatModel,
    ) -> Result<()> {
        self.install_preparation_seal(model.lfm25_audio_asr_preparation_stage_ceiling()?)
    }

    #[cfg(test)]
    fn install_test_preparation_seal(
        &self,
        backend: BackendKind,
        _max_batch_size: usize,
    ) -> Result<()> {
        self.install_preparation_seal(
            crate::models::architectures::lfm25_audio::model::Lfm25AudioAsrPreparationStageCeiling {
                backend,
                max_source_samples: 16_000 * 30,
                max_source_sample_rate: 48_000,
                max_resampled_samples: 16_000 * 30,
                max_prompt_tokens: 4_096,
                max_work_units: 1_024,
                max_materialized_tensor_elements: 4_096 * 2_048,
                max_retained_resident_bytes: 4_096 * 2_048 * 4,
                max_host_workspace_bytes: 32 * 1024 * 1024,
                max_device_workspace_bytes: 64 * 1024 * 1024,
                max_unified_workspace_bytes: 0,
                max_workspace_bytes: 96 * 1024 * 1024,
                max_decode_workspace_bytes: 16 * 1024 * 1024,
            },
        )
    }

    fn contract(&self, streaming: StreamingRequirements) -> Result<LoadedExecutionContract> {
        let metadata = self.metadata();
        if streaming.model_native && metadata.streaming_mode == StreamingMode::None {
            return Err(Error::InvalidInput(format!(
                "Model {} has no streaming ASR contract",
                metadata.model_variant
            )));
        }
        if streaming.asr_long_form {
            let mut profile = scalar_execution_profile(metadata, self.backend_kind);
            profile.mode = ExecutionMode::Atomic;
            profile.prefill = PrefillMode::None;
            profile.incremental_decode = false;
            profile.prefill_batch = NativeBatchMode::None;
            profile.decode_batch = NativeBatchMode::None;
            profile.cache_mode = CacheMode::None;
            profile.cache_namespace = None;
            profile.kv_dtype = "none".into();
            profile.cancellation = CancellationGranularity::OperationBoundary;
            profile.concurrency = ConcurrencyClass::Exclusive;
            profile.recompute_safe = false;
            profile.cache_release_safe = false;
            profile.prefix_reuse_safe = false;
            profile.max_batch_size = 1;
            profile.resolved_from_loaded_model = true;

            let mut stage = StageDescriptor::from_execution_profile(
                StageId::new(3),
                "asr.long_form.lfm25_audio.atomic",
                &profile,
                NativeBatchMode::None,
            );
            stage.selector = StageWorkSelector::Atomic;
            stage.shape_policy = StageShapePolicy::Exact;
            stage.output_visibility = OutputVisibility::AfterQuantumCommit;
            dispatch_long_form_asr_by_chunk(
                &mut stage,
                &metadata,
                "asr.long_form.lfm25_audio.chunked",
            );
            stage.validate()?;
            return Ok(LoadedExecutionContract {
                execution_group_id: self.execution_group_id,
                model_instance_id: self.model_instance_id,
                adapter_instance_id: self.adapter_instance_id(),
                adapter_abi_revision: self.adapter_abi_revision(),
                metadata,
                execution_profile: profile,
                stages: Arc::from([stage]),
            });
        }

        let seal = self.preparation.get().ok_or_else(|| {
            Error::ModelLoadError(
                "LFM2.5 Audio ASR normal graph is unavailable before preparation is sealed".into(),
            )
        })?;
        if seal.backend != self.backend_kind
            || seal.max_source_samples == 0
            || seal.max_source_sample_rate == 0
            || seal.max_resampled_samples == 0
            || seal.max_prompt_tokens == 0
            || seal.max_work_units == 0
            || seal.max_materialized_tensor_elements == 0
            || seal.max_retained_resident_bytes == 0
            || seal.max_workspace_bytes == 0
            || seal.max_decode_workspace_bytes == 0
        {
            return Err(Error::ModelLoadError(
                "LFM2.5 Audio ASR preparation seal does not match its loaded adapter".into(),
            ));
        }

        let mut profile = scalar_execution_profile(metadata, self.backend_kind);
        profile.mode = ExecutionMode::Sequence;
        profile.prefill = PrefillMode::Incremental;
        profile.incremental_decode = true;
        profile.prefill_batch = NativeBatchMode::Static;
        profile.decode_batch = NativeBatchMode::Continuous;
        profile.cache_mode = CacheMode::ExternalPaged;
        profile.cache_namespace = Some(format!(
            "{}:{}:state-v2",
            metadata.model_variant,
            self.backend_kind.as_str()
        ));
        profile.kv_dtype = "state_v2_resolved".into();
        profile.cancellation = CancellationGranularity::SequenceStep;
        profile.concurrency = ConcurrencyClass::Batchable;
        profile.recompute_safe = true;
        profile.cache_release_safe = true;
        profile.prefix_reuse_safe = false;
        profile.max_batch_size = self.max_batch_size;
        profile.resolved_from_loaded_model = true;

        let mut preparation = StageDescriptor::from_execution_profile(
            StageId::new(0),
            "asr.encoder.lfm25_audio",
            &profile,
            NativeBatchMode::None,
        );
        preparation.selector = StageWorkSelector::PreSequencePreparation;
        preparation.progress = StageProgressKind::Atomic;
        preparation.membership_safe_point = MembershipSafePoint::OperationBoundary;
        preparation.max_batch_size = 1;
        preparation.max_work_units = seal.max_work_units;
        preparation.max_workspace_bytes = seal.max_workspace_bytes;
        preparation.concurrency = ConcurrencyClass::Exclusive;
        preparation.shape_policy = StageShapePolicy::Exact;
        preparation.output_visibility = OutputVisibility::AfterQuantumCommit;

        let mut prefill = StageDescriptor::from_execution_profile(
            StageId::new(1),
            "asr.prefill.lfm25_audio.tensor_static",
            &profile,
            NativeBatchMode::Static,
        );
        prefill.selector = StageWorkSelector::SequencePrefill;
        prefill.progress = StageProgressKind::Iterative;
        prefill.max_batch_size = self.max_batch_size;
        prefill.concurrency = ConcurrencyClass::Batchable;
        prefill.shape_policy = StageShapePolicy::Ragged;
        prefill.max_padding_basis_points = 0;
        prefill.max_work_units = u64::try_from(seal.max_prompt_tokens)
            .map_err(|_| Error::Overloaded("LFM2.5 Audio ASR prompt ceiling exceeds u64".into()))?
            .checked_mul(u64::try_from(self.max_batch_size).map_err(|_| {
                Error::Overloaded("LFM2.5 Audio ASR batch ceiling exceeds u64".into())
            })?)
            .ok_or_else(|| Error::Overloaded("LFM2.5 Audio ASR prefill work overflow".into()))?;
        prefill.max_workspace_bytes = seal
            .max_workspace_bytes
            .checked_mul(u64::try_from(self.max_batch_size).map_err(|_| {
                Error::Overloaded("LFM2.5 Audio ASR batch ceiling exceeds u64".into())
            })?)
            .ok_or_else(|| {
                Error::Overloaded("LFM2.5 Audio ASR prefill workspace overflow".into())
            })?;
        prefill.output_visibility = OutputVisibility::AfterQuantumCommit;

        let mut decode = StageDescriptor::from_execution_profile(
            StageId::new(2),
            "asr.decode.lfm25_audio.continuous",
            &profile,
            NativeBatchMode::Continuous,
        );
        decode.selector = StageWorkSelector::SequenceDecode;
        decode.progress = StageProgressKind::Iterative;
        decode.shape_policy = StageShapePolicy::Ragged;
        decode.concurrency = ConcurrencyClass::Batchable;
        decode.max_work_units = u64::try_from(self.max_batch_size).map_err(|_| {
            Error::Overloaded("LFM2.5 Audio ASR batch width exceeds work accounting".into())
        })?;
        // One decode step per row at the last position the context allows: a
        // budget derived from the model, so a default request (2,048 new
        // tokens) is admitted, instead of the shared 16 MiB (LFA-2).
        decode.max_workspace_bytes = seal
            .max_decode_workspace_bytes
            .checked_mul(decode.max_work_units)
            .ok_or_else(|| {
                Error::Overloaded("LFM2.5 Audio ASR decode workspace overflow".into())
            })?;
        decode.output_visibility = OutputVisibility::AfterQuantumCommit;

        preparation.validate()?;
        prefill.validate()?;
        decode.validate()?;
        Ok(LoadedExecutionContract {
            execution_group_id: self.execution_group_id,
            model_instance_id: self.model_instance_id,
            adapter_instance_id: self.adapter_instance_id(),
            adapter_abi_revision: self.adapter_abi_revision(),
            metadata,
            execution_profile: profile,
            stages: Arc::from([preparation, prefill, decode]),
        })
    }
}

#[derive(Debug)]
struct GraniteSpeechAsrExecutionSeal {
    preparation:
        crate::models::architectures::granite_speech::asr::GraniteSpeechAsrPreparationStageSeal,
    preparation_max_batch_work_units: u64,
    preparation_max_batch_materialized_tensor_elements: u64,
    preparation_max_batch_workspace_bytes: u64,
    decode_workspace_per_row_bytes: u64,
}

#[derive(Debug)]
struct GraniteSpeechAsrExecutionAdapter {
    execution_group_id: ExecutionGroupId,
    model_instance_id: ModelInstanceId,
    adapter_instance_id: AdapterInstanceId,
    metadata: AdapterMetadata,
    backend_kind: BackendKind,
    max_batch_size: usize,
    seal: OnceLock<GraniteSpeechAsrExecutionSeal>,
}

impl GraniteSpeechAsrExecutionAdapter {
    fn new(
        execution_group_id: ExecutionGroupId,
        model_instance_id: ModelInstanceId,
        metadata: AdapterMetadata,
        backend_kind: BackendKind,
        max_batch_size: usize,
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
            seal: OnceLock::new(),
        }
    }

    fn install_preparation_seal(&self, seal: GraniteSpeechAsrExecutionSeal) -> Result<()> {
        if let Some(existing) = self.seal.get() {
            return if existing.preparation == seal.preparation
                && existing.preparation_max_batch_work_units
                    == seal.preparation_max_batch_work_units
                && existing.preparation_max_batch_materialized_tensor_elements
                    == seal.preparation_max_batch_materialized_tensor_elements
                && existing.preparation_max_batch_workspace_bytes
                    == seal.preparation_max_batch_workspace_bytes
                && existing.decode_workspace_per_row_bytes == seal.decode_workspace_per_row_bytes
            {
                Ok(())
            } else {
                Err(Error::ModelLoadError(
                    "Granite Speech ASR preparation was resealed with different geometry".into(),
                ))
            };
        }
        self.seal.set(seal).map_err(|_| {
            Error::ModelLoadError("Granite Speech ASR preparation seal raced publication".into())
        })
    }
}

impl LoadedExecutionAdapter for GraniteSpeechAsrExecutionAdapter {
    fn metadata(&self) -> AdapterMetadata {
        self.metadata
    }

    fn adapter_instance_id(&self) -> AdapterInstanceId {
        self.adapter_instance_id
    }

    fn adapter_abi_revision(&self) -> AdapterAbiRevision {
        GRANITE_SPEECH_ASR_ADAPTER_ABI
    }

    fn seal_granite_speech_asr_preparation(
        &self,
        model: &crate::models::architectures::granite_speech::asr::GraniteSpeechAsrModel,
    ) -> Result<()> {
        let preparation = model.scalar_preparation_stage_seal(self.backend_kind)?;
        let width = u64::try_from(self.max_batch_size)
            .map_err(|_| Error::Overloaded("Granite Speech batch width exceeds u64".into()))?;
        self.install_preparation_seal(GraniteSpeechAsrExecutionSeal {
            preparation_max_batch_work_units: preparation
                .max_work_units
                .checked_mul(width)
                .ok_or_else(|| {
                    Error::Overloaded("Granite Speech batch work ceiling overflow".into())
                })?,
            preparation_max_batch_materialized_tensor_elements: preparation
                .max_materialized_tensor_elements_per_row
                .checked_mul(width)
                .ok_or_else(|| {
                    Error::Overloaded(
                        "Granite Speech batch materialization ceiling overflow".into(),
                    )
                })?,
            preparation_max_batch_workspace_bytes: preparation
                .max_workspace_bytes
                .checked_mul(width)
                .ok_or_else(|| {
                    Error::Overloaded("Granite Speech batch workspace ceiling overflow".into())
                })?,
            preparation,
            decode_workspace_per_row_bytes: model.continuous_decode_workspace_per_row_bytes()?,
        })
    }

    #[cfg(test)]
    fn install_test_preparation_seal(
        &self,
        backend: BackendKind,
        max_batch_size: usize,
    ) -> Result<()> {
        if max_batch_size.max(1) != self.max_batch_size {
            return Err(Error::ModelLoadError(
                "test Granite Speech preparation width does not match its adapter".into(),
            ));
        }
        let width = u64::try_from(self.max_batch_size)
            .map_err(|_| Error::Overloaded("test Granite Speech batch width exceeds u64".into()))?;
        self.install_preparation_seal(GraniteSpeechAsrExecutionSeal {
            preparation:
                crate::models::architectures::granite_speech::asr::GraniteSpeechAsrPreparationStageSeal {
                backend,
                dtype: "f32".into(),
                max_work_units: 10_000,
                max_materialized_tensor_elements_per_row: 2_000_000,
                max_workspace_bytes: 1024 * 1024 * 1024,
            },
            preparation_max_batch_work_units: 10_000_u64
                .checked_mul(width)
                .ok_or_else(|| {
                    Error::Overloaded("test Granite Speech work ceiling overflow".into())
                })?,
            preparation_max_batch_materialized_tensor_elements: 2_000_000_u64
                .checked_mul(width)
                .ok_or_else(|| {
                    Error::Overloaded("test Granite Speech materialization ceiling overflow".into())
                })?,
            preparation_max_batch_workspace_bytes: (1024_u64 * 1024 * 1024)
                .checked_mul(width)
                .ok_or_else(|| {
                    Error::Overloaded("test Granite Speech workspace ceiling overflow".into())
                })?,
            decode_workspace_per_row_bytes: 8_192,
        })
    }

    fn contract(&self, streaming: StreamingRequirements) -> Result<LoadedExecutionContract> {
        let metadata = self.metadata();
        if streaming.model_native {
            return Err(Error::InvalidInput(format!(
                "Model {} has no model-native streaming ASR contract",
                metadata.model_variant
            )));
        }
        if streaming.asr_long_form {
            let mut profile = scalar_execution_profile(metadata, self.backend_kind);
            profile.mode = ExecutionMode::Atomic;
            profile.prefill = PrefillMode::None;
            profile.incremental_decode = false;
            profile.prefill_batch = NativeBatchMode::None;
            profile.decode_batch = NativeBatchMode::None;
            profile.cache_mode = CacheMode::None;
            profile.cache_namespace = None;
            profile.kv_dtype = "none".into();
            profile.cancellation = CancellationGranularity::OperationBoundary;
            profile.concurrency = ConcurrencyClass::Exclusive;
            profile.recompute_safe = false;
            profile.cache_release_safe = false;
            profile.prefix_reuse_safe = false;
            profile.max_batch_size = 1;
            profile.resolved_from_loaded_model = true;
            let mut stage = StageDescriptor::from_execution_profile(
                StageId::new(0),
                "asr.granite.long_form.atomic",
                &profile,
                NativeBatchMode::None,
            );
            stage.selector = StageWorkSelector::Atomic;
            stage.shape_policy = StageShapePolicy::Exact;
            stage.output_visibility = output_visibility_for(
                streaming.transport_output,
                profile.mode,
                NativeBatchMode::None,
            );
            dispatch_long_form_asr_by_chunk(&mut stage, &metadata, "asr.granite.long_form.chunked");
            stage.validate()?;
            return Ok(LoadedExecutionContract {
                execution_group_id: self.execution_group_id,
                model_instance_id: self.model_instance_id,
                adapter_instance_id: self.adapter_instance_id(),
                adapter_abi_revision: self.adapter_abi_revision(),
                metadata,
                execution_profile: profile,
                stages: Arc::from([stage]),
            });
        }

        let seal = self.seal.get().ok_or_else(|| {
            Error::ModelLoadError(
                "Granite Speech ASR normal execution graph is unavailable before preparation is sealed"
                    .into(),
            )
        })?;
        let width = u64::try_from(self.max_batch_size)
            .map_err(|_| Error::Overloaded("Granite Speech batch width exceeds u64".into()))?;
        let valid_batch_ceilings = seal
            .preparation
            .max_work_units
            .checked_mul(width)
            .is_some_and(|value| value == seal.preparation_max_batch_work_units)
            && seal
                .preparation
                .max_materialized_tensor_elements_per_row
                .checked_mul(width)
                .is_some_and(|value| {
                    value == seal.preparation_max_batch_materialized_tensor_elements
                })
            && seal
                .preparation
                .max_workspace_bytes
                .checked_mul(width)
                .is_some_and(|value| value == seal.preparation_max_batch_workspace_bytes);
        if seal.preparation.backend != self.backend_kind
            || seal.preparation.dtype.is_empty()
            || seal.decode_workspace_per_row_bytes == 0
            || seal.preparation.max_materialized_tensor_elements_per_row == 0
            || seal.preparation_max_batch_work_units == 0
            || seal.preparation_max_batch_materialized_tensor_elements == 0
            || seal.preparation_max_batch_workspace_bytes == 0
            || !valid_batch_ceilings
        {
            return Err(Error::ModelLoadError(
                "Granite Speech ASR preparation seal does not match its loaded adapter identity"
                    .into(),
            ));
        }

        let mut profile = scalar_execution_profile(metadata, self.backend_kind);
        profile.mode = ExecutionMode::Sequence;
        profile.prefill = PrefillMode::Incremental;
        profile.incremental_decode = true;
        profile.prefill_batch = NativeBatchMode::None;
        profile.decode_batch = NativeBatchMode::Continuous;
        profile.cache_mode = CacheMode::ExternalPaged;
        profile.cache_namespace = Some(format!(
            "{}:{}:state-v2",
            metadata.model_variant,
            self.backend_kind.as_str()
        ));
        profile.kv_dtype = "state_v2_resolved".into();
        profile.cancellation = CancellationGranularity::SequenceStep;
        profile.concurrency = ConcurrencyClass::Batchable;
        profile.recompute_safe = true;
        profile.cache_release_safe = true;
        profile.prefix_reuse_safe = false;
        profile.max_batch_size = self.max_batch_size;
        profile.resolved_from_loaded_model = true;

        let native_preparation = self.max_batch_size > 1;
        let mut preparation = StageDescriptor::from_execution_profile(
            StageId::new(0),
            "asr.encoder.granite_speech",
            &profile,
            if native_preparation {
                NativeBatchMode::Static
            } else {
                NativeBatchMode::None
            },
        );
        preparation.selector = StageWorkSelector::PreSequencePreparation;
        preparation.progress = StageProgressKind::Atomic;
        preparation.membership_safe_point = MembershipSafePoint::OperationBoundary;
        if native_preparation {
            preparation.max_batch_size = self.max_batch_size;
            preparation.max_work_units = seal.preparation_max_batch_work_units;
            preparation.max_workspace_bytes = seal.preparation_max_batch_workspace_bytes;
        } else {
            preparation.max_batch_size = 1;
            preparation.max_work_units = seal.preparation.max_work_units;
            preparation.max_workspace_bytes = seal.preparation.max_workspace_bytes;
            preparation.concurrency = ConcurrencyClass::Exclusive;
            preparation.shape_policy = StageShapePolicy::Exact;
        }
        preparation.output_visibility = OutputVisibility::AfterQuantumCommit;

        let mut prefill = StageDescriptor::from_execution_profile(
            StageId::new(1),
            "asr.granite.prefill.scalar",
            &profile,
            NativeBatchMode::None,
        );
        prefill.selector = StageWorkSelector::SequencePrefill;
        prefill.max_batch_size = 1;
        prefill.concurrency = ConcurrencyClass::Exclusive;
        prefill.shape_policy = StageShapePolicy::Exact;

        let mut decode = StageDescriptor::from_execution_profile(
            StageId::new(2),
            "asr.granite.decode.tensor_continuous",
            &profile,
            NativeBatchMode::Continuous,
        );
        decode.selector = StageWorkSelector::SequenceDecode;
        decode.max_work_units = u64::try_from(decode.max_batch_size).map_err(|_| {
            Error::Overloaded("Granite Speech ASR batch width exceeds work accounting".into())
        })?;
        let decode_workspace_per_row =
            continuous_asr_workspace_per_row_bytes(seal.decode_workspace_per_row_bytes)?;
        decode.workspace_per_row_bytes = decode_workspace_per_row;
        decode.max_workspace_bytes = decode_workspace_per_row
            .checked_mul(u64::try_from(self.max_batch_size).map_err(|_| {
                Error::Overloaded("Granite Speech ASR batch width exceeds u64".into())
            })?)
            .ok_or_else(|| {
                Error::Overloaded("Granite Speech ASR decode workspace overflow".into())
            })?;
        preparation.validate()?;
        prefill.validate()?;
        decode.validate()?;
        Ok(LoadedExecutionContract {
            execution_group_id: self.execution_group_id,
            model_instance_id: self.model_instance_id,
            adapter_instance_id: self.adapter_instance_id(),
            adapter_abi_revision: self.adapter_abi_revision(),
            metadata,
            execution_profile: profile,
            stages: Arc::from([preparation, prefill, decode]),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VibeVoiceAsrExecutionSeal {
    preparation: crate::models::architectures::vibevoice::asr::VibeVoiceAsrPreparationStageSeal,
    prefill_max_batch_work_units: u64,
    prefill_max_batch_workspace_bytes: u64,
    decode_workspace_per_row_bytes: u64,
}

#[derive(Debug)]
struct VibeVoiceAsrExecutionAdapter {
    execution_group_id: ExecutionGroupId,
    model_instance_id: ModelInstanceId,
    adapter_instance_id: AdapterInstanceId,
    metadata: AdapterMetadata,
    backend_kind: BackendKind,
    max_batch_size: usize,
    seal: OnceLock<VibeVoiceAsrExecutionSeal>,
}

impl VibeVoiceAsrExecutionAdapter {
    fn new(
        execution_group_id: ExecutionGroupId,
        model_instance_id: ModelInstanceId,
        metadata: AdapterMetadata,
        backend_kind: BackendKind,
        max_batch_size: usize,
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
            seal: OnceLock::new(),
        }
    }
}

impl LoadedExecutionAdapter for VibeVoiceAsrExecutionAdapter {
    fn metadata(&self) -> AdapterMetadata {
        self.metadata
    }

    fn adapter_instance_id(&self) -> AdapterInstanceId {
        self.adapter_instance_id
    }

    fn adapter_abi_revision(&self) -> AdapterAbiRevision {
        VIBEVOICE_ASR_ADAPTER_ABI
    }

    fn seal_vibevoice_asr_preparation(
        &self,
        model: &crate::models::architectures::vibevoice::asr::VibeVoiceAsrModel,
    ) -> Result<()> {
        let preparation = model.scalar_preparation_stage_seal(self.backend_kind)?;
        let width = u64::try_from(self.max_batch_size)
            .map_err(|_| Error::Overloaded("VibeVoice ASR batch width exceeds u64".into()))?;
        let seal = VibeVoiceAsrExecutionSeal {
            prefill_max_batch_work_units: preparation
                .max_work_units
                .checked_mul(width)
                .ok_or_else(|| {
                    Error::Overloaded("VibeVoice ASR prefill work ceiling overflow".into())
                })?,
            prefill_max_batch_workspace_bytes: preparation
                .max_workspace_bytes
                .checked_mul(width)
                .ok_or_else(|| {
                Error::Overloaded("VibeVoice ASR prefill workspace ceiling overflow".into())
            })?,
            preparation,
            decode_workspace_per_row_bytes: model.continuous_decode_workspace_per_row_bytes()?,
        };
        if let Some(existing) = self.seal.get() {
            return if existing == &seal {
                Ok(())
            } else {
                Err(Error::ModelLoadError(
                    "VibeVoice ASR execution was resealed with different geometry".into(),
                ))
            };
        }
        self.seal.set(seal).map_err(|_| {
            Error::ModelLoadError("VibeVoice ASR execution seal raced publication".into())
        })
    }

    #[cfg(test)]
    fn install_test_preparation_seal(
        &self,
        backend: BackendKind,
        max_batch_size: usize,
    ) -> Result<()> {
        let width = u64::try_from(max_batch_size)
            .map_err(|_| Error::Overloaded("VibeVoice ASR test batch width exceeds u64".into()))?;
        self.seal
            .set(VibeVoiceAsrExecutionSeal {
                preparation:
                    crate::models::architectures::vibevoice::asr::VibeVoiceAsrPreparationStageSeal {
                        backend,
                        dtype: "f32".into(),
                        max_work_units: 1_500,
                        max_workspace_bytes: 64 * 1024 * 1024,
                    },
                prefill_max_batch_work_units: 1_500_u64.checked_mul(width).ok_or_else(|| {
                    Error::Overloaded("VibeVoice ASR test prefill work ceiling overflow".into())
                })?,
                prefill_max_batch_workspace_bytes: (64_u64 * 1024 * 1024)
                    .checked_mul(width)
                    .ok_or_else(|| {
                        Error::Overloaded(
                            "VibeVoice ASR test prefill workspace ceiling overflow".into(),
                        )
                    })?,
                decode_workspace_per_row_bytes: 8_192,
            })
            .map_err(|_| Error::ModelLoadError("test VibeVoice seal was already installed".into()))
    }

    fn contract(&self, streaming: StreamingRequirements) -> Result<LoadedExecutionContract> {
        use crate::models::architectures::vibevoice::{
            VIBEVOICE_ASR_DECODE_STAGE, VIBEVOICE_ASR_LEGACY_STAGE, VIBEVOICE_ASR_PREFILL_STAGE,
            VIBEVOICE_ASR_PREPARATION_STAGE, VIBEVOICE_ASR_TOKENIZER_GROUP,
        };

        let metadata = self.metadata();
        if streaming.model_native {
            return Err(Error::InvalidInput(format!(
                "Model {} has no model-native streaming ASR contract",
                metadata.model_variant
            )));
        }
        if streaming.asr_long_form {
            let mut profile = scalar_execution_profile(metadata, self.backend_kind);
            profile.mode = ExecutionMode::Atomic;
            profile.prefill = PrefillMode::None;
            profile.incremental_decode = false;
            profile.prefill_batch = NativeBatchMode::None;
            profile.decode_batch = NativeBatchMode::None;
            profile.cache_mode = CacheMode::None;
            profile.cache_namespace = None;
            profile.kv_dtype = "none".into();
            profile.cancellation = CancellationGranularity::OperationBoundary;
            profile.concurrency = ConcurrencyClass::Exclusive;
            profile.recompute_safe = false;
            profile.cache_release_safe = false;
            profile.prefix_reuse_safe = false;
            profile.max_batch_size = 1;
            profile.resolved_from_loaded_model = true;
            let mut stage = StageDescriptor::from_execution_profile(
                StageId::new(0),
                VIBEVOICE_ASR_LEGACY_STAGE,
                &profile,
                NativeBatchMode::None,
            );
            stage.selector = StageWorkSelector::Atomic;
            stage.shape_policy = StageShapePolicy::Exact;
            stage.output_visibility = output_visibility_for(
                streaming.transport_output,
                profile.mode,
                NativeBatchMode::None,
            );
            stage.validate()?;
            return Ok(LoadedExecutionContract {
                execution_group_id: self.execution_group_id,
                model_instance_id: self.model_instance_id,
                adapter_instance_id: self.adapter_instance_id(),
                adapter_abi_revision: self.adapter_abi_revision(),
                metadata,
                execution_profile: profile,
                stages: Arc::from([stage]),
            });
        }

        let seal = self.seal.get().ok_or_else(|| {
            Error::ModelLoadError(
                "VibeVoice ASR normal graph is unavailable before model geometry is sealed".into(),
            )
        })?;
        if seal.preparation.backend != self.backend_kind
            || seal.preparation.dtype.is_empty()
            || seal.preparation.max_work_units == 0
            || seal.preparation.max_workspace_bytes == 0
            || seal.prefill_max_batch_work_units == 0
            || seal.prefill_max_batch_workspace_bytes == 0
            || seal.decode_workspace_per_row_bytes == 0
        {
            return Err(Error::ModelLoadError(
                "VibeVoice ASR execution seal does not match its loaded adapter".into(),
            ));
        }
        let decode_workspace_per_row =
            continuous_asr_workspace_per_row_bytes(seal.decode_workspace_per_row_bytes)?;
        let decode_workspace = decode_workspace_per_row
            .checked_mul(
                u64::try_from(self.max_batch_size).map_err(|_| {
                    Error::Overloaded("VibeVoice ASR batch width exceeds u64".into())
                })?,
            )
            .ok_or_else(|| Error::Overloaded("VibeVoice ASR decode workspace overflow".into()))?;
        let mut profile = scalar_execution_profile(metadata, self.backend_kind);
        let native_prefill = self.max_batch_size > 1;
        profile.mode = ExecutionMode::Sequence;
        profile.prefill = PrefillMode::Incremental;
        profile.incremental_decode = true;
        profile.prefill_batch = if native_prefill {
            NativeBatchMode::Static
        } else {
            NativeBatchMode::None
        };
        profile.decode_batch = NativeBatchMode::Continuous;
        profile.cache_mode = CacheMode::ExternalPaged;
        profile.cache_namespace = Some(format!(
            "{}:{}:state-v2",
            metadata.model_variant,
            self.backend_kind.as_str()
        ));
        profile.kv_dtype = "state_v2_resolved".into();
        profile.cancellation = CancellationGranularity::SequenceStep;
        profile.concurrency = ConcurrencyClass::Batchable;
        profile.recompute_safe = true;
        profile.cache_release_safe = true;
        profile.prefix_reuse_safe = false;
        profile.max_batch_size = self.max_batch_size;
        profile.resolved_from_loaded_model = true;

        let mut preparation = StageDescriptor::from_execution_profile(
            StageId::new(0),
            VIBEVOICE_ASR_PREPARATION_STAGE,
            &profile,
            NativeBatchMode::None,
        );
        preparation.selector = StageWorkSelector::PreSequencePreparation;
        preparation.progress = StageProgressKind::Atomic;
        preparation.membership_safe_point = MembershipSafePoint::OperationBoundary;
        preparation.max_batch_size = 1;
        preparation.max_work_units = seal.preparation.max_work_units;
        preparation.max_workspace_bytes = seal.preparation.max_workspace_bytes;
        preparation.concurrency = ConcurrencyClass::Exclusive;
        preparation.shape_policy = StageShapePolicy::Exact;
        preparation.output_visibility = OutputVisibility::AfterQuantumCommit;

        let mut prefill = StageDescriptor::from_execution_profile(
            StageId::new(1),
            VIBEVOICE_ASR_PREFILL_STAGE,
            &profile,
            profile.prefill_batch,
        );
        prefill.selector = StageWorkSelector::SequencePrefill;
        if native_prefill {
            prefill.max_batch_size = self.max_batch_size;
            prefill.max_work_units = seal.prefill_max_batch_work_units;
            // The largest loaded preparation envelope is a conservative upper
            // bound for one causal tokenizer row plus its two connectors. The
            // checked width product seals the padded native batch envelope while
            // Core charges this per-row ceiling before model entry.
            prefill.workspace_per_row_bytes = seal.preparation.max_workspace_bytes;
            prefill.max_workspace_bytes = seal.prefill_max_batch_workspace_bytes;
        } else {
            prefill.max_batch_size = 1;
            prefill.max_work_units = seal.preparation.max_work_units;
            prefill.max_workspace_bytes = seal.preparation.max_workspace_bytes;
            prefill.concurrency = ConcurrencyClass::Exclusive;
            prefill.shape_policy = StageShapePolicy::Exact;
        }
        prefill.retained_state_selections = Some(vec![ClockedStateSelection::new(
            VIBEVOICE_ASR_TOKENIZER_GROUP,
            StateClock::AudioSamples,
        )?]);
        prefill.output_visibility = output_visibility_for(
            streaming.transport_output,
            profile.mode,
            profile.prefill_batch,
        );

        let mut decode = StageDescriptor::from_execution_profile(
            StageId::new(2),
            VIBEVOICE_ASR_DECODE_STAGE,
            &profile,
            NativeBatchMode::Continuous,
        );
        decode.selector = StageWorkSelector::SequenceDecode;
        decode.max_work_units = u64::try_from(decode.max_batch_size).map_err(|_| {
            Error::Overloaded("VibeVoice ASR batch width exceeds work accounting".into())
        })?;
        decode.max_workspace_bytes = decode_workspace;
        decode.workspace_per_row_bytes = decode_workspace_per_row;
        decode.retained_state_selections = Some(vec![]);
        preparation.validate()?;
        prefill.validate()?;
        decode.validate()?;
        Ok(LoadedExecutionContract {
            execution_group_id: self.execution_group_id,
            model_instance_id: self.model_instance_id,
            adapter_instance_id: self.adapter_instance_id(),
            adapter_abi_revision: self.adapter_abi_revision(),
            metadata,
            execution_profile: profile,
            stages: Arc::from([preparation, prefill, decode]),
        })
    }
}

#[derive(Debug)]
struct WhisperAsrExecutionAdapter {
    execution_group_id: ExecutionGroupId,
    model_instance_id: ModelInstanceId,
    adapter_instance_id: AdapterInstanceId,
    metadata: AdapterMetadata,
    backend_kind: BackendKind,
    max_batch_size: usize,
    audio_preparation:
        OnceLock<crate::models::architectures::whisper::asr::WhisperAudioPreparationStageSeal>,
}

impl WhisperAsrExecutionAdapter {
    fn new(
        execution_group_id: ExecutionGroupId,
        model_instance_id: ModelInstanceId,
        metadata: AdapterMetadata,
        backend_kind: BackendKind,
        max_batch_size: usize,
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
            audio_preparation: OnceLock::new(),
        }
    }
}

impl LoadedExecutionAdapter for WhisperAsrExecutionAdapter {
    fn metadata(&self) -> AdapterMetadata {
        self.metadata
    }

    fn adapter_instance_id(&self) -> AdapterInstanceId {
        self.adapter_instance_id
    }

    fn adapter_abi_revision(&self) -> AdapterAbiRevision {
        WHISPER_ASR_ADAPTER_ABI
    }

    fn seal_whisper_audio_preparation(
        &self,
        model: &crate::models::architectures::whisper::asr::WhisperTurboAsrModel,
    ) -> Result<()> {
        let seal = model.window_preparation_stage_seal(self.backend_kind, self.max_batch_size)?;
        if let Some(existing) = self.audio_preparation.get() {
            return if existing == &seal {
                Ok(())
            } else {
                Err(Error::ModelLoadError(
                    "Whisper audio preparation was resealed with different geometry".into(),
                ))
            };
        }
        self.audio_preparation.set(seal).map_err(|_| {
            Error::ModelLoadError("Whisper audio preparation seal raced publication".into())
        })
    }

    #[cfg(test)]
    fn install_test_preparation_seal(
        &self,
        backend: BackendKind,
        max_batch_size: usize,
    ) -> Result<()> {
        self.audio_preparation
            .set(
                crate::models::architectures::whisper::asr::WhisperAudioPreparationStageSeal {
                    backend,
                    dtype: "f32".into(),
                    max_batch_size,
                    max_workspace_bytes: 64 * 1024 * 1024,
                },
            )
            .map_err(|_| Error::ModelLoadError("test Whisper seal was already installed".into()))
    }

    fn contract(&self, streaming: StreamingRequirements) -> Result<LoadedExecutionContract> {
        let metadata = self.metadata();
        if streaming.model_native {
            return Err(Error::InvalidInput(format!(
                "Model {} has no model-native streaming ASR contract",
                metadata.model_variant
            )));
        }
        if streaming.asr_long_form {
            let mut execution_profile = scalar_execution_profile(metadata, self.backend_kind);
            execution_profile.mode = ExecutionMode::Atomic;
            execution_profile.prefill = PrefillMode::None;
            execution_profile.incremental_decode = false;
            execution_profile.cache_mode = CacheMode::None;
            execution_profile.cache_namespace = None;
            execution_profile.kv_dtype = "none".to_string();
            execution_profile.concurrency = ConcurrencyClass::Exclusive;
            execution_profile.max_batch_size = 1;
            execution_profile.resolved_from_loaded_model = true;
            let mut stage = StageDescriptor::from_execution_profile(
                StageId::new(3),
                "asr.long_form.atomic",
                &execution_profile,
                NativeBatchMode::None,
            );
            stage.selector = StageWorkSelector::Atomic;
            stage.shape_policy = StageShapePolicy::Exact;
            stage.output_visibility = output_visibility_for(
                streaming.transport_output,
                execution_profile.mode,
                NativeBatchMode::None,
            );
            dispatch_long_form_asr_by_chunk(&mut stage, &metadata, "asr.long_form.chunked");
            stage.validate()?;
            return Ok(LoadedExecutionContract {
                execution_group_id: self.execution_group_id,
                model_instance_id: self.model_instance_id,
                adapter_instance_id: self.adapter_instance_id(),
                adapter_abi_revision: self.adapter_abi_revision(),
                metadata,
                execution_profile,
                stages: Arc::from([stage]),
            });
        }

        let seal = self.audio_preparation.get().ok_or_else(|| {
            Error::ModelLoadError(
                "Whisper normal execution graph is unavailable before audio preparation is sealed"
                    .into(),
            )
        })?;
        if seal.backend != self.backend_kind
            || seal.max_batch_size != self.max_batch_size
            || seal.dtype.is_empty()
        {
            return Err(Error::ModelLoadError(
                "Whisper audio preparation seal does not match its loaded adapter identity".into(),
            ));
        }
        let mut execution_profile = scalar_execution_profile(metadata, self.backend_kind);
        execution_profile.mode = ExecutionMode::Sequence;
        execution_profile.prefill = PrefillMode::Incremental;
        execution_profile.incremental_decode = true;
        execution_profile.prefill_batch = NativeBatchMode::None;
        execution_profile.decode_batch = NativeBatchMode::None;
        execution_profile.cache_mode = CacheMode::ExternalPaged;
        execution_profile.cache_namespace = Some(format!(
            "{}:{}:state-v2",
            metadata.model_variant,
            self.backend_kind.as_str()
        ));
        execution_profile.kv_dtype = "state_v2_resolved".to_string();
        execution_profile.cancellation = CancellationGranularity::SequenceStep;
        execution_profile.concurrency = ConcurrencyClass::Exclusive;
        execution_profile.recompute_safe = true;
        execution_profile.cache_release_safe = true;
        // The capability envelope exposes the independently batchable encoder
        // width; decoder stages below remain explicitly scalar width one.
        execution_profile.max_batch_size = self.max_batch_size;
        execution_profile.resolved_from_loaded_model = true;

        let mut preparation = StageDescriptor::from_execution_profile(
            StageId::new(0),
            "asr.encoder.whisper",
            &execution_profile,
            NativeBatchMode::Static,
        );
        preparation.selector = StageWorkSelector::PreSequencePreparation;
        preparation.progress = StageProgressKind::Atomic;
        preparation.membership_safe_point = MembershipSafePoint::OperationBoundary;
        preparation.shape_policy = StageShapePolicy::Padded;
        preparation.max_batch_size = seal.max_batch_size;
        preparation.max_workspace_bytes = seal.max_workspace_bytes;
        preparation.output_visibility = OutputVisibility::AfterQuantumCommit;

        let mut prefill = StageDescriptor::from_execution_profile(
            StageId::new(1),
            "asr.prefill.scalar",
            &execution_profile,
            NativeBatchMode::None,
        );
        prefill.selector = StageWorkSelector::SequencePrefill;
        prefill.max_batch_size = 1;
        prefill.concurrency = ConcurrencyClass::Exclusive;
        prefill.shape_policy = StageShapePolicy::Exact;

        let mut decode = StageDescriptor::from_execution_profile(
            StageId::new(2),
            "asr.decode.scalar",
            &execution_profile,
            NativeBatchMode::None,
        );
        decode.selector = StageWorkSelector::SequenceDecode;
        decode.max_batch_size = 1;
        decode.concurrency = ConcurrencyClass::Exclusive;
        decode.shape_policy = StageShapePolicy::Exact;
        preparation.validate()?;
        prefill.validate()?;
        decode.validate()?;

        Ok(LoadedExecutionContract {
            execution_group_id: self.execution_group_id,
            model_instance_id: self.model_instance_id,
            adapter_instance_id: self.adapter_instance_id(),
            adapter_abi_revision: self.adapter_abi_revision(),
            metadata,
            execution_profile,
            stages: Arc::from([preparation, prefill, decode]),
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
    workspace_per_row_bytes: OnceLock<u64>,
    /// `(per_token_bytes, max_span_tokens)` of a model that prices its
    /// prefill activations.
    prefill_workspace: OnceLock<(u64, u64)>,
    /// The loaded model's preferred decode quantum.
    decode_quantum: OnceLock<u64>,
    /// The token budget of a sealed mixed prefill+decode stage.
    mixed_step_budget: OnceLock<u64>,
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
            workspace_per_row_bytes: OnceLock::new(),
            prefill_workspace: OnceLock::new(),
            decode_quantum: OnceLock::new(),
            mixed_step_budget: OnceLock::new(),
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

    fn seal_chat_workspace(&self, accelerator_bytes: u64) -> Result<()> {
        let per_row = continuous_chat_workspace_per_row(accelerator_bytes)?.workspace_bytes()?;
        let width = u64::try_from(self.max_batch_size)
            .map_err(|_| Error::Overloaded("continuous chat batch width overflow".into()))?;
        per_row.checked_mul(width).ok_or_else(|| {
            Error::Overloaded("continuous chat batch workspace estimate overflow".into())
        })?;
        if let Some(existing) = self.workspace_per_row_bytes.get() {
            return if *existing == per_row {
                Ok(())
            } else {
                Err(Error::ModelLoadError(
                    "continuous chat workspace was resealed with different geometry".into(),
                ))
            };
        }
        self.workspace_per_row_bytes.set(per_row).map_err(|_| {
            Error::ModelLoadError("continuous chat workspace seal raced publication".into())
        })
    }

    fn seal_chat_prefill_workspace(&self, per_token_bytes: u64, max_span_tokens: u64) -> Result<()> {
        if max_span_tokens == 0 {
            return Err(Error::ModelLoadError(
                "chat prefill span must be positive".into(),
            ));
        }
        per_token_bytes
            .checked_mul(max_span_tokens)
            .ok_or_else(|| Error::Overloaded("chat prefill workspace estimate overflow".into()))?;
        let sealed = (per_token_bytes, max_span_tokens);
        if let Some(existing) = self.prefill_workspace.get() {
            return if *existing == sealed {
                Ok(())
            } else {
                Err(Error::ModelLoadError(
                    "chat prefill workspace was resealed with different geometry".into(),
                ))
            };
        }
        self.prefill_workspace.set(sealed).map_err(|_| {
            Error::ModelLoadError("chat prefill workspace seal raced publication".into())
        })
    }

    fn seal_chat_decode_quantum(&self, tokens: u64) -> Result<()> {
        if tokens == 0 {
            return Err(Error::ModelLoadError(
                "chat decode quantum must be positive".into(),
            ));
        }
        if let Some(existing) = self.decode_quantum.get() {
            return if *existing == tokens {
                Ok(())
            } else {
                Err(Error::ModelLoadError(
                    "chat decode quantum was resealed with a different size".into(),
                ))
            };
        }
        self.decode_quantum.set(tokens).map_err(|_| {
            Error::ModelLoadError("chat decode quantum seal raced publication".into())
        })
    }

    fn seal_chat_mixed_step(&self, token_budget: u64) -> Result<()> {
        if token_budget == 0 {
            return Err(Error::ModelLoadError(
                "chat mixed step budget must be positive".into(),
            ));
        }
        if let Some(existing) = self.mixed_step_budget.get() {
            return if *existing == token_budget {
                Ok(())
            } else {
                Err(Error::ModelLoadError(
                    "chat mixed step was resealed with a different budget".into(),
                ))
            };
        }
        self.mixed_step_budget
            .set(token_budget)
            .map_err(|_| Error::ModelLoadError("chat mixed step seal raced publication".into()))
    }

    #[cfg(test)]
    fn install_test_preparation_seal(&self, _backend: BackendKind, _width: usize) -> Result<()> {
        self.seal_chat_workspace(8_192)
    }

    fn contract(&self, streaming: StreamingRequirements) -> Result<LoadedExecutionContract> {
        let metadata = self.metadata();
        let workspace_per_row = *self.workspace_per_row_bytes.get().ok_or_else(|| {
            Error::ModelLoadError(
                "continuous chat execution graph is unavailable before loaded model workspace is sealed"
                    .into(),
            )
        })?;
        if streaming.model_native && metadata.streaming_mode == StreamingMode::None {
            return Err(Error::InvalidInput(format!(
                "Model {} has no streaming chat contract",
                metadata.model_variant
            )));
        }
        let mut execution_profile = scalar_execution_profile(metadata, self.backend_kind);
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
        if let Some((per_token_bytes, max_span_tokens)) = self.prefill_workspace.get().copied() {
            // The scheduler caps each prefill span at the stage's work units,
            // with or without chunked prefill, and prices every token.
            prefill.max_work_units = max_span_tokens;
            prefill.workspace_per_work_unit_bytes = per_token_bytes;
            prefill.max_workspace_bytes = per_token_bytes
                .checked_mul(max_span_tokens)
                .ok_or_else(|| Error::Overloaded("chat prefill workspace estimate overflow".into()))?;
        }
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
            .max(CONTINUOUS_CHAT_MAX_DECODE_QUANTUM)
            .max(self.decode_quantum.get().copied().unwrap_or(1));
        decode.workspace_per_row_bytes = workspace_per_row;
        decode.max_workspace_bytes =
            workspace_per_row
                .checked_mul(u64::try_from(decode.max_batch_size).map_err(|_| {
                    Error::Overloaded("continuous chat batch width overflow".into())
                })?)
                .ok_or_else(|| {
                    Error::Overloaded("continuous chat batch workspace estimate overflow".into())
                })?;
        prefill.validate()?;
        decode.validate()?;
        let stages = match self.mixed_step_budget.get().copied() {
            // One mixed stage replaces both: with the split stages beside it
            // the exact selectors would be ambiguous.
            Some(token_budget) => {
                execution_profile.mixed_step = true;
                let mut mixed = StageDescriptor::from_execution_profile(
                    StageId::new(3),
                    "chat.step.mixed",
                    &execution_profile,
                    NativeBatchMode::Mixed,
                );
                mixed.selector = StageWorkSelector::SequenceStep;
                mixed.max_work_units = token_budget;
                // Every token is priced like a prefill token and every row
                // like a decode row; the ragged batch sums them.
                mixed.workspace_per_work_unit_bytes = prefill.workspace_per_work_unit_bytes;
                mixed.workspace_per_row_bytes = workspace_per_row;
                mixed.max_workspace_bytes = prefill
                    .workspace_per_work_unit_bytes
                    .checked_mul(token_budget)
                    .and_then(|bytes| bytes.checked_add(decode.max_workspace_bytes))
                    .ok_or_else(|| {
                        Error::Overloaded("chat mixed step workspace estimate overflow".into())
                    })?;
                mixed.output_visibility = decode.output_visibility;
                mixed.validate()?;
                Arc::from([mixed])
            }
            None => Arc::from([prefill, decode]),
        };

        Ok(LoadedExecutionContract {
            execution_group_id: self.execution_group_id,
            model_instance_id: self.model_instance_id,
            adapter_instance_id: self.adapter_instance_id(),
            adapter_abi_revision: self.adapter_abi_revision(),
            metadata,
            execution_profile,
            stages,
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
            // A chat model runs at most its family's certified width.
            max_tensor_batch_size: if metadata.capability == CapabilityKind::Chat {
                self.max_chat_tensor_batch_size()
                    .min(crate::catalog::certified_decode_width(
                        metadata.model_variant.family(),
                        backend_kind,
                    ))
            } else {
                self.max_tensor_batch_size()
            },
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

    pub(crate) fn seal_fish_capacity(
        &self,
        capacity: crate::runtime::fish_capacity::ResolvedFishServingCapacity,
    ) -> Result<()> {
        for capability in [CapabilityKind::Tts, CapabilityKind::StreamingTts] {
            self.capabilities
                .get(&capability)
                .ok_or_else(|| {
                    Error::ModelLoadError("Fish capability missing during capacity seal".into())
                })?
                .seal_fish_capacity(capacity.clone())?;
        }
        Ok(())
    }

    pub(crate) fn seal_chat_workspace(&self, accelerator_bytes: u64) -> Result<()> {
        let execution = self
            .capabilities
            .get(&CapabilityKind::Chat)
            .ok_or_else(|| {
                Error::ModelLoadError("loaded bundle has no chat capability to seal".into())
            })?;
        execution.seal_chat_workspace(accelerator_bytes)
    }

    pub(crate) fn seal_chat_decode_quantum(&self, tokens: u64) -> Result<()> {
        self.capabilities
            .get(&CapabilityKind::Chat)
            .ok_or_else(|| {
                Error::ModelLoadError("loaded bundle has no chat capability to seal".into())
            })?
            .seal_chat_decode_quantum(tokens)
    }

    pub(crate) fn seal_chat_prefill_workspace(
        &self,
        per_token_bytes: u64,
        max_span_tokens: u64,
    ) -> Result<()> {
        self.capabilities
            .get(&CapabilityKind::Chat)
            .ok_or_else(|| {
                Error::ModelLoadError("loaded bundle has no chat capability to seal".into())
            })?
            .seal_chat_prefill_workspace(per_token_bytes, max_span_tokens)
    }

    pub(crate) fn seal_chat_mixed_step(&self, token_budget: u64) -> Result<()> {
        self.capabilities
            .get(&CapabilityKind::Chat)
            .ok_or_else(|| {
                Error::ModelLoadError("loaded bundle has no chat capability to seal".into())
            })?
            .seal_chat_mixed_step(token_budget)
    }

    pub(crate) fn seal_qwen3_asr_audio_preparation(
        &self,
        model: &crate::models::architectures::qwen3::asr::Qwen3AsrModel,
    ) -> Result<()> {
        let execution = self.capabilities.get(&CapabilityKind::Asr).ok_or_else(|| {
            Error::ModelLoadError("Qwen3 ASR loaded bundle has no ASR capability to seal".into())
        })?;
        execution.seal_qwen3_asr_audio_preparation(model)
    }

    pub(crate) fn seal_whisper_audio_preparation(
        &self,
        model: &crate::models::architectures::whisper::asr::WhisperTurboAsrModel,
    ) -> Result<()> {
        let execution = self.capabilities.get(&CapabilityKind::Asr).ok_or_else(|| {
            Error::ModelLoadError("Whisper loaded bundle has no ASR capability to seal".into())
        })?;
        execution.seal_whisper_audio_preparation(model)
    }

    pub(crate) fn seal_vibevoice_asr_preparation(
        &self,
        model: &crate::models::architectures::vibevoice::asr::VibeVoiceAsrModel,
    ) -> Result<()> {
        let execution = self.capabilities.get(&CapabilityKind::Asr).ok_or_else(|| {
            Error::ModelLoadError("VibeVoice loaded bundle has no ASR capability to seal".into())
        })?;
        execution.seal_vibevoice_asr_preparation(model)
    }

    pub(crate) fn seal_granite_speech_asr_preparation(
        &self,
        model: &crate::models::architectures::granite_speech::asr::GraniteSpeechAsrModel,
    ) -> Result<()> {
        let execution = self.capabilities.get(&CapabilityKind::Asr).ok_or_else(|| {
            Error::ModelLoadError(
                "Granite Speech loaded bundle has no ASR capability to seal".into(),
            )
        })?;
        execution.seal_granite_speech_asr_preparation(model)
    }

    pub(crate) fn seal_lfm25_audio_asr_preparation(
        &self,
        model: &crate::models::registry::NativeAudioChatModel,
    ) -> Result<()> {
        let execution = self.capabilities.get(&CapabilityKind::Asr).ok_or_else(|| {
            Error::ModelLoadError("LFM2.5 Audio loaded bundle has no ASR capability to seal".into())
        })?;
        execution.seal_lfm25_audio_asr_preparation(model)
    }

    pub(crate) fn seal_lfm25_audio_tts_preparation(
        &self,
        model: &crate::models::registry::NativeAudioChatModel,
    ) -> Result<()> {
        let execution = self.capabilities.get(&CapabilityKind::Tts).ok_or_else(|| {
            Error::ModelLoadError("LFM2.5 Audio loaded bundle has no TTS capability to seal".into())
        })?;
        execution.seal_lfm25_audio_tts_preparation(model)
    }

    pub(crate) fn seal_voxtral_realtime_preparation(
        &self,
        model: &crate::models::architectures::voxtral::realtime::VoxtralRealtimeModel,
    ) -> Result<()> {
        let execution = self
            .capabilities
            .get(&CapabilityKind::RealtimeAsr)
            .ok_or_else(|| {
                Error::ModelLoadError(
                    "Voxtral loaded bundle has no realtime ASR capability to seal".into(),
                )
            })?;
        execution.seal_voxtral_realtime_preparation(model)
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
    #[cfg(test)]
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
    #[cfg(test)]
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

    #[cfg(test)]
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

    #[cfg(test)]
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
#[path = "loaded_tests.rs"]
mod tests;
