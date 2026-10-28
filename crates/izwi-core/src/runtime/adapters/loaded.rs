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

mod asr;
mod bundle;
mod chat;
mod descriptor;
mod factories;
mod realtime;
mod sequence_asr;
mod tts;
use asr::{
    ContinuousAsrExecutionAdapter, Lfm25AudioAsrExecutionAdapter, Lfm3AudioAsrExecutionAdapter,
    ParakeetAsrExecutionAdapter,
};
use chat::ContinuousChatExecutionAdapter;
use descriptor::loaded_execution_contracts;
pub(super) use factories::built_in_loaded_adapter_factories;
#[cfg(test)]
use factories::{
    ContinuousPhysicalChatAdapterFactory, FishS2PhysicalTtsAdapterFactory,
    Lfm25AudioPhysicalAsrAdapterFactory, Lfm25AudioPhysicalTtsAdapterFactory,
    ScalarExecutionAdapterFactory,
};
use realtime::{NemotronRealtimeExecutionAdapter, VoxtralRealtimeExecutionAdapter};
use sequence_asr::{
    GraniteSpeechAsrExecutionAdapter, VibeVoiceAsrExecutionAdapter, WhisperAsrExecutionAdapter,
};
use tts::{
    ChatterboxTtsExecutionAdapter, FishS2TtsExecutionAdapter, Lfm25AudioTtsExecutionAdapter,
    Lfm3AudioTtsExecutionAdapter, Maya1TtsExecutionAdapter, Maya2TtsExecutionAdapter,
    PhysicalQwenTtsExecutionAdapter, StaticTtsExecutionAdapter, VibeVoiceTtsExecutionAdapter,
    VoxtralTtsExecutionAdapter,
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
const CHATTERBOX_TTS_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(1);
const MAYA1_TTS_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(1);
const MAYA2_TTS_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(1);
const LFM3_AUDIO_TTS_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(1);
const LFM3_AUDIO_ASR_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(1);
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

/// Whether a contract can run one whole job as a blocking stage: the
/// long-form ASR graph that the diarization pipeline requires.
pub(crate) fn contract_supports_atomic_blocking(contract: &LoadedExecutionContract) -> bool {
    contract.stages.iter().any(|stage| {
        matches!(
            stage.selector,
            StageWorkSelector::Atomic | StageWorkSelector::Any
        )
    })
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
    // A scalar ASR family that dispatches long audio by chunk has one graph,
    // the chunked atomic stage, for every request: each invocation leases
    // its own decoder state, and a long file yields the device between its
    // chunks. A short file finishes in its first dispatch.
    if metadata.capability == CapabilityKind::Asr
        && metadata
            .model_variant
            .family()
            .dispatches_long_form_asr_by_chunk()
    {
        stage.selector = StageWorkSelector::Atomic;
        dispatch_long_form_asr_by_chunk(&mut stage, &metadata, "asr.long_form.chunked");
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct VibeVoiceAsrExecutionSeal {
    preparation: crate::models::architectures::vibevoice::asr::VibeVoiceAsrPreparationStageSeal,
    prefill_max_batch_work_units: u64,
    prefill_max_batch_workspace_bytes: u64,
    decode_workspace_per_row_bytes: u64,
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

pub(crate) struct LoadedModelBundle {
    execution_group_id: ExecutionGroupId,
    model_instance_id: ModelInstanceId,
    model_variant: ModelVariant,
    backend_kind: BackendKind,
    capabilities: HashMap<CapabilityKind, LoadedCapabilityDescriptor>,
}

#[cfg(test)]
#[path = "loaded_tests.rs"]
mod tests;
