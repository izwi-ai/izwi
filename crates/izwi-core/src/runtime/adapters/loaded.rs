use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::backends::BackendKind;
use crate::engine::{
    AdapterAbiRevision, AdapterInstanceId, CacheMode, CancellationGranularity, ConcurrencyClass,
    ExecutionAdapterBinding, ExecutionGroupId, ExecutionMode, ExecutionProfile, ModelInstanceId,
    NativeBatchMode, OutputVisibility, PrefillMode, StageDescriptor, StageId, StageWorkSelector,
};
use crate::error::{Error, Result};
use crate::kv::v2::CapabilityStateDescriptorV2;
use crate::kv::{CacheCapability, LoadedKvCacheCapability};
use crate::model::ModelVariant;

use super::{
    compatibility_execution_profile, AdapterMetadata, CapabilityKind, RuntimeAdapterRegistry,
    StreamingMode,
};

const COMPATIBILITY_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(7);
const STATIC_TENSOR_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(8);
const CONTINUOUS_TENSOR_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(9);
const STATIC_TTS_MAX_BATCH_WORKSPACE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const CONTINUOUS_CHAT_MAX_BATCH_WORKSPACE_BYTES: u64 = 16 * 1024 * 1024;
const COMPATIBILITY_CACHE_FALLBACK_REASON: &str = "loaded_adapter_uses_model_owned_cache";
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

fn compatible_request_parallelism(backend_kind: BackendKind, configured: usize) -> usize {
    if backend_kind == BackendKind::Metal {
        1
    } else {
        configured.max(1)
    }
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
    pub(crate) fn adapter_binding(&self) -> Result<ExecutionAdapterBinding> {
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

fn compatibility_state_publication() -> LoadedStatePublication {
    LoadedStatePublication::LegacyV1(LoadedKvCacheCapability {
        capability: CacheCapability::OpaqueModelOwned,
        fallback_reason: Some(COMPATIBILITY_CACHE_FALLBACK_REASON),
    })
}

/// Additive loaded-state publication during the ABI migration. v1 remains an
/// explicit compatibility value; v2 is validated as a complete semantic
/// contract and is never converted to an opaque v1 fallback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LoadedStatePublication {
    LegacyV1(LoadedKvCacheCapability),
    V2(CapabilityStateDescriptorV2),
}

impl LoadedStatePublication {
    fn validate(&self, stages: &[StageDescriptor]) -> Result<()> {
        match self {
            Self::LegacyV1(cache) => cache.validate(),
            Self::V2(descriptor) => descriptor.validate_against_stages(stages),
        }
    }

    fn binding(&self) -> CapabilityStateBinding {
        match self {
            Self::LegacyV1(cache) => CapabilityStateBinding::LegacyV1(cache.capability.clone()),
            Self::V2(descriptor) => CapabilityStateBinding::V2(descriptor.clone()),
        }
    }

    fn fingerprint(&self, stages: &[StageDescriptor]) -> Result<Option<[u8; 32]>> {
        match self {
            Self::LegacyV1(_) => Ok(None),
            Self::V2(descriptor) => Ok(Some(descriptor.fingerprint(stages)?)),
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
}

impl LoadedCapabilityDescriptor {
    fn new(
        execution: Arc<dyn LoadedExecutionAdapter>,
        state: LoadedStatePublication,
    ) -> Result<Self> {
        let execution_contract = execution.contract(StreamingRequirements::NONE)?;
        state.validate(&execution_contract.stages)?;
        Ok(Self { execution, state })
    }

    fn contract(&self, streaming: StreamingRequirements) -> Result<LoadedExecutionContract> {
        self.execution.contract(streaming)
    }

    fn binding(&self, streaming: StreamingRequirements) -> Result<LoadedCapabilityBinding> {
        let contract = self.contract(streaming)?;
        self.state.validate(&contract.stages)?;
        Ok(LoadedCapabilityBinding {
            execution: contract.adapter_binding()?,
            state: self.state.binding(),
            state_fingerprint: self.state.fingerprint(&contract.stages)?,
        })
    }
}

/// Request-ready projection of one sealed loaded capability descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LoadedCapabilityBinding {
    pub(crate) execution: ExecutionAdapterBinding,
    pub(crate) state: CapabilityStateBinding,
    pub(crate) state_fingerprint: Option<[u8; 32]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CapabilityStateBinding {
    LegacyV1(CacheCapability),
    V2(CapabilityStateDescriptorV2),
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
struct StaticQwenTtsAdapterFactory;

impl LoadedExecutionAdapterFactory for StaticQwenTtsAdapterFactory {
    fn id(&self) -> &'static str {
        "builtin.qwen_tts.tensor_static"
    }

    fn batch_mode(&self) -> NativeBatchMode {
        NativeBatchMode::Static
    }

    fn supports(&self, metadata: AdapterMetadata, _backend_kind: BackendKind) -> bool {
        metadata.capability == CapabilityKind::Tts
            && metadata.model_variant.family() == crate::catalog::ModelFamily::Qwen3Tts
            && metadata
                .model_variant
                .speech_capabilities()
                .is_some_and(|capabilities| capabilities.supports_builtin_voices)
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

#[derive(Debug, Clone, Copy)]
struct ContinuousQwenChatAdapterFactory;

impl LoadedExecutionAdapterFactory for ContinuousQwenChatAdapterFactory {
    fn id(&self) -> &'static str {
        "builtin.qwen_chat.tensor_continuous"
    }

    fn batch_mode(&self) -> NativeBatchMode {
        NativeBatchMode::Continuous
    }

    fn supports(&self, metadata: AdapterMetadata, _backend_kind: BackendKind) -> bool {
        metadata.capability == CapabilityKind::Chat
            && matches!(
                metadata.model_variant,
                ModelVariant::Qwen306B
                    | ModelVariant::Qwen306B4Bit
                    | ModelVariant::Qwen317B
                    | ModelVariant::Qwen317B4Bit
            )
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

pub(super) fn built_in_loaded_adapter_factories() -> Vec<Arc<dyn LoadedExecutionAdapterFactory>> {
    vec![
        Arc::new(StaticQwenTtsAdapterFactory),
        Arc::new(ContinuousQwenChatAdapterFactory),
    ]
}

#[derive(Debug)]
struct CompatibilityExecutionAdapter {
    execution_group_id: ExecutionGroupId,
    model_instance_id: ModelInstanceId,
    adapter_instance_id: AdapterInstanceId,
    metadata: AdapterMetadata,
    backend_kind: BackendKind,
    request_parallelism: usize,
}

impl CompatibilityExecutionAdapter {
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
            request_parallelism: compatible_request_parallelism(backend_kind, request_parallelism),
        }
    }
}

impl LoadedExecutionAdapter for CompatibilityExecutionAdapter {
    fn metadata(&self) -> AdapterMetadata {
        self.metadata
    }

    fn adapter_instance_id(&self) -> AdapterInstanceId {
        self.adapter_instance_id
    }

    fn adapter_abi_revision(&self) -> AdapterAbiRevision {
        COMPATIBILITY_ADAPTER_ABI
    }

    fn contract(&self, streaming: StreamingRequirements) -> Result<LoadedExecutionContract> {
        compatibility_contract(
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

fn compatibility_contract(
    execution_group_id: ExecutionGroupId,
    model_instance_id: ModelInstanceId,
    adapter_instance_id: AdapterInstanceId,
    adapter_abi_revision: AdapterAbiRevision,
    metadata: AdapterMetadata,
    backend_kind: BackendKind,
    request_parallelism: usize,
    streaming: StreamingRequirements,
) -> Result<LoadedExecutionContract> {
    if streaming.model_native && metadata.streaming_mode == StreamingMode::None {
        return Err(Error::InvalidInput(format!(
            "Model {} supports {:?}, but not streaming execution for that capability",
            metadata.model_variant, metadata.capability
        )));
    }

    let mut execution_profile =
        compatibility_execution_profile(metadata, backend_kind, streaming.model_native);
    execution_profile.resolved_from_loaded_model = true;
    execution_profile.prefill_batch = NativeBatchMode::None;
    execution_profile.decode_batch = NativeBatchMode::None;
    execution_profile.max_batch_size = request_parallelism.max(1);
    execution_profile.concurrency = if execution_profile.max_batch_size > 1 {
        ConcurrencyClass::Batchable
    } else {
        ConcurrencyClass::Exclusive
    };

    let mut stage = StageDescriptor::from_execution_profile(
        StageId::new(0),
        format!("{}.compatibility", metadata.capability.as_str()),
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
            request_parallelism: compatible_request_parallelism(backend_kind, request_parallelism),
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
            return compatibility_contract(
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
        let mut execution_profile =
            compatibility_execution_profile(metadata, self.backend_kind, false);
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
        let mut compatibility = StageDescriptor::from_execution_profile(
            StageId::new(0),
            "tts.generate.compatibility",
            &execution_profile,
            NativeBatchMode::None,
        );
        compatibility.selector = StageWorkSelector::Any;
        compatibility.max_batch_size = self.request_parallelism;
        compatibility.concurrency = if self.request_parallelism > 1 {
            ConcurrencyClass::Batchable
        } else {
            ConcurrencyClass::Exclusive
        };
        compatibility.shape_policy = if self.request_parallelism > 1 {
            crate::engine::StageShapePolicy::Independent
        } else {
            crate::engine::StageShapePolicy::Exact
        };
        compatibility.output_visibility = output_visibility_for(
            streaming.transport_output,
            execution_profile.mode,
            NativeBatchMode::None,
        );
        stage.validate()?;
        compatibility.validate()?;

        Ok(LoadedExecutionContract {
            execution_group_id: self.execution_group_id,
            model_instance_id: self.model_instance_id,
            adapter_instance_id: self.adapter_instance_id(),
            adapter_abi_revision: self.adapter_abi_revision(),
            metadata,
            execution_profile,
            stages: Arc::from([stage, compatibility]),
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
    request_parallelism: usize,
}

impl ContinuousChatExecutionAdapter {
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
            request_parallelism: compatible_request_parallelism(backend_kind, request_parallelism),
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
            compatibility_execution_profile(metadata, self.backend_kind, streaming.model_native);
        execution_profile.prefill_batch = NativeBatchMode::None;
        execution_profile.decode_batch = NativeBatchMode::Continuous;
        execution_profile.concurrency = ConcurrencyClass::Batchable;
        execution_profile.max_batch_size = self.max_batch_size;
        execution_profile.resolved_from_loaded_model = true;

        let mut prefill = StageDescriptor::from_execution_profile(
            StageId::new(1),
            "chat.prefill.compatibility",
            &execution_profile,
            NativeBatchMode::None,
        );
        prefill.selector = StageWorkSelector::SequencePrefill;
        prefill.max_batch_size = self.request_parallelism;
        prefill.concurrency = if self.request_parallelism > 1 {
            ConcurrencyClass::Batchable
        } else {
            ConcurrencyClass::Exclusive
        };
        prefill.shape_policy = if self.request_parallelism > 1 {
            crate::engine::StageShapePolicy::Independent
        } else {
            crate::engine::StageShapePolicy::Exact
        };
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
        // batch, not a per-row quantum. Continuous decode schedules exactly
        // one token per row, so a width-N stage needs an N-unit budget or the
        // second row can never join the batch.
        decode.max_work_units = u64::try_from(decode.max_batch_size).map_err(|_| {
            Error::Overloaded("continuous decode batch width exceeds work accounting".to_string())
        })?;
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
    ) -> Result<Option<&dyn LoadedExecutionAdapterFactory>> {
        let mut matches = self
            .loaded_adapter_factories
            .iter()
            .filter(|factory| factory.supports(metadata, backend_kind));
        let Some(selected) = matches.next() else {
            return Ok(None);
        };
        if let Some(ambiguous) = matches.next() {
            return Err(Error::ModelLoadError(format!(
                "loaded model {} capability {:?} matches both native factories `{}` and `{}`",
                metadata.model_variant,
                metadata.capability,
                selected.id(),
                ambiguous.id(),
            )));
        }
        Ok(Some(selected.as_ref()))
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
                        .is_some_and(|factory| factory.batch_mode() == batch_mode)
                })
            })
            .collect()
    }

    fn bind_loaded_capability(
        &self,
        execution_group_id: ExecutionGroupId,
        model_instance_id: ModelInstanceId,
        metadata: AdapterMetadata,
        backend_kind: BackendKind,
        state: LoadedStatePublication,
    ) -> Result<LoadedCapabilityDescriptor> {
        let context = LoadedAdapterFactoryContext {
            execution_group_id,
            model_instance_id,
            backend_kind,
            max_tensor_batch_size: self.max_tensor_batch_size(),
            request_parallelism: self.request_parallelism(),
        };
        let adapter = match self.loaded_adapter_factory(metadata, backend_kind)? {
            Some(factory) => factory.create(context, metadata)?,
            None => Arc::new(CompatibilityExecutionAdapter::new(
                execution_group_id,
                model_instance_id,
                metadata,
                backend_kind,
                self.request_parallelism(),
            )),
        };
        if adapter.metadata() != metadata {
            return Err(Error::ModelLoadError(format!(
                "loaded adapter factory returned mismatched metadata for {} capability {:?}",
                metadata.model_variant, metadata.capability
            )));
        }
        LoadedCapabilityDescriptor::new(adapter, state)
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

    /// Bind adapter metadata and exact loaded-model cache truth into one sealed
    /// descriptor per capability. Missing declarations retain the current
    /// opaque compatibility behavior during model migration.
    pub(crate) fn bind_with_state_publications(
        registry: &RuntimeAdapterRegistry,
        execution_group_id: ExecutionGroupId,
        model_instance_id: ModelInstanceId,
        model_variant: ModelVariant,
        backend_kind: BackendKind,
        mut state_publications: HashMap<CapabilityKind, LoadedStatePublication>,
    ) -> Result<Self> {
        let metadata = registry.capabilities_for(model_variant);
        if metadata.is_empty() {
            return Err(Error::ModelLoadError(format!(
                "loaded model {model_variant} has no executable capability adapter"
            )));
        }
        let mut unmatched = state_publications
            .keys()
            .copied()
            .filter(|capability| !metadata.iter().any(|entry| entry.capability == *capability))
            .map(CapabilityKind::as_str)
            .collect::<Vec<_>>();
        if !unmatched.is_empty() {
            unmatched.sort_unstable();
            return Err(Error::ModelLoadError(format!(
                "loaded model {model_variant} published cache truth for unregistered capabilities: {}",
                unmatched.join(", ")
            )));
        }

        let mut capabilities = HashMap::with_capacity(metadata.len());
        for metadata in metadata {
            let state = state_publications
                .remove(&metadata.capability)
                .unwrap_or_else(compatibility_state_publication);
            let descriptor = registry.bind_loaded_capability(
                execution_group_id,
                model_instance_id,
                metadata,
                backend_kind,
                state,
            )?;
            if capabilities
                .insert(metadata.capability, descriptor)
                .is_some()
            {
                return Err(Error::ModelLoadError(format!(
                    "loaded model {model_variant} has duplicate {:?} adapters",
                    metadata.capability
                )));
            }
        }

        debug_assert!(state_publications.is_empty());

        Ok(Self {
            execution_group_id,
            model_instance_id,
            model_variant,
            backend_kind,
            capabilities,
        })
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
    use crate::runtime::adapters::ExecutionTargetKind;

    #[test]
    fn exact_native_qwen3_cpu_descriptor_seals_execution_and_managed_cache_truth() {
        let registry = RuntimeAdapterRegistry::built_in();
        let managed = CacheCapability::Managed(crate::kv::test_contract());
        let state_publications = HashMap::from([(
            CapabilityKind::Chat,
            LoadedStatePublication::LegacyV1(LoadedKvCacheCapability {
                capability: managed.clone(),
                fallback_reason: None,
            }),
        )]);
        let exact = LoadedModelBundle::bind_with_state_publications(
            &registry,
            ExecutionGroupId::new(3),
            ModelInstanceId::new(9),
            ModelVariant::Qwen306B,
            BackendKind::Cpu,
            state_publications,
        )
        .unwrap();
        let binding = exact
            .capability_binding_for_streaming(CapabilityKind::Chat, StreamingRequirements::NONE)
            .unwrap();
        assert_eq!(binding.execution.model_instance_id, ModelInstanceId::new(9));
        assert_eq!(binding.execution.capability_id, "chat");
        assert_eq!(binding.state, CapabilityStateBinding::LegacyV1(managed));

        let catalog_only = LoadedModelBundle::bind(
            &registry,
            ExecutionGroupId::new(3),
            ModelInstanceId::new(10),
            ModelVariant::Qwen306B,
            BackendKind::Cpu,
        )
        .unwrap();
        assert_eq!(
            catalog_only
                .capability_binding_for_streaming(
                    CapabilityKind::Chat,
                    StreamingRequirements::NONE,
                )
                .unwrap(),
            LoadedCapabilityBinding {
                execution: catalog_only
                    .contract(CapabilityKind::Chat, false)
                    .unwrap()
                    .adapter_binding()
                    .unwrap(),
                state: CapabilityStateBinding::LegacyV1(CacheCapability::OpaqueModelOwned),
                state_fingerprint: None,
            }
        );
    }

    #[test]
    fn v2_state_publication_is_preserved_without_legacy_fallback() {
        let contract = crate::kv::v2::test_contract();
        let registry = RuntimeAdapterRegistry::built_in();
        let compatibility = LoadedModelBundle::bind(
            &registry,
            ExecutionGroupId::new(3),
            ModelInstanceId::new(10),
            ModelVariant::Qwen306B,
            BackendKind::Cpu,
        )
        .unwrap();
        let stages = compatibility
            .contract(CapabilityKind::Chat, false)
            .unwrap()
            .stages;
        let descriptor =
            crate::kv::v2::CapabilityStateDescriptorV2::managed_for_stages_test(contract, &stages);
        let bundle = LoadedModelBundle::bind_with_state_publications(
            &registry,
            ExecutionGroupId::new(3),
            ModelInstanceId::new(10),
            ModelVariant::Qwen306B,
            BackendKind::Cpu,
            HashMap::from([(
                CapabilityKind::Chat,
                LoadedStatePublication::V2(descriptor.clone()),
            )]),
        )
        .unwrap();

        assert_eq!(
            bundle
                .capability_binding_for_streaming(
                    CapabilityKind::Chat,
                    StreamingRequirements::NONE,
                )
                .unwrap()
                .state,
            CapabilityStateBinding::V2(descriptor)
        );
    }

    #[test]
    fn cache_truth_is_scoped_to_one_capability_descriptor() {
        let registry = RuntimeAdapterRegistry::built_in();
        let managed = CacheCapability::Managed(crate::kv::test_contract());
        let state_publications = HashMap::from([(
            CapabilityKind::Tts,
            LoadedStatePublication::LegacyV1(LoadedKvCacheCapability {
                capability: managed.clone(),
                fallback_reason: None,
            }),
        )]);
        let bundle = LoadedModelBundle::bind_with_state_publications(
            &registry,
            ExecutionGroupId::new(4),
            ModelInstanceId::new(11),
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
            BackendKind::Cpu,
            state_publications,
        )
        .unwrap();

        let tts = bundle
            .capability_binding_for_streaming(CapabilityKind::Tts, StreamingRequirements::NONE)
            .unwrap();
        assert_eq!(tts.state, CapabilityStateBinding::LegacyV1(managed));
        assert_eq!(
            bundle
                .capability_binding_for_streaming(
                    CapabilityKind::StreamingTts,
                    StreamingRequirements::NONE,
                )
                .unwrap()
                .state,
            CapabilityStateBinding::LegacyV1(CacheCapability::OpaqueModelOwned)
        );
    }

    #[test]
    fn cache_truth_for_an_unregistered_capability_is_rejected() {
        let state_publications = HashMap::from([(
            CapabilityKind::Asr,
            LoadedStatePublication::LegacyV1(LoadedKvCacheCapability {
                capability: CacheCapability::Managed(crate::kv::test_contract()),
                fallback_reason: None,
            }),
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
                && metadata.capability == CapabilityKind::Tts
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

    #[test]
    fn every_supported_model_capability_binds_to_an_exact_width_one_contract() {
        let registry = RuntimeAdapterRegistry::built_in();

        for (index, variant) in ModelVariant::all().iter().copied().enumerate() {
            let instance = ModelInstanceId::new(index as u64 + 1);
            let bundle = LoadedModelBundle::bind(
                &registry,
                ExecutionGroupId::new(7),
                instance,
                variant,
                BackendKind::Cpu,
            )
            .unwrap_or_else(|error| panic!("failed to bind {variant}: {error}"));
            let metadata = registry.capabilities_for(variant);

            assert_eq!(bundle.adapter_count(), metadata.len(), "{variant}");
            assert_eq!(bundle.model_instance_id(), instance);
            for metadata in metadata {
                let binding = bundle
                    .capability_binding_for_streaming(
                        metadata.capability,
                        StreamingRequirements::NONE,
                    )
                    .unwrap_or_else(|error| {
                        panic!("failed to bind capability for {variant}: {error}")
                    });
                assert_eq!(
                    binding.state,
                    CapabilityStateBinding::LegacyV1(CacheCapability::OpaqueModelOwned),
                    "cache capability changed for {variant}"
                );
                let contract = bundle
                    .contract(metadata.capability, false)
                    .unwrap_or_else(|error| panic!("failed to contract {variant}: {error}"));
                assert_eq!(contract.execution_group_id, ExecutionGroupId::new(7));
                assert_eq!(contract.model_instance_id, instance);
                assert_eq!(contract.metadata, metadata);
                let native_mode = registry
                    .loaded_adapter_factory(metadata, BackendKind::Cpu)
                    .unwrap()
                    .map(|factory| factory.batch_mode());
                match native_mode {
                    Some(mode) => {
                        assert_ne!(contract.adapter_abi_revision, COMPATIBILITY_ADAPTER_ABI);
                        assert!(contract.stages.iter().any(|stage| stage.batch_mode == mode));
                    }
                    None => {
                        assert_eq!(contract.adapter_abi_revision, COMPATIBILITY_ADAPTER_ABI);
                        assert_eq!(contract.stages.len(), 1);
                        assert_eq!(contract.stages[0].batch_mode, NativeBatchMode::None);
                    }
                }
                assert!(contract
                    .stages
                    .iter()
                    .all(|stage| stage.max_batch_size == 1));
                assert_eq!(contract.execution_profile.max_batch_size, 1);
                assert!(contract.execution_profile.resolved_from_loaded_model);
                assert_eq!(binding.execution.model_variant, variant);
                assert_eq!(binding.execution.model_instance_id, instance);
                assert_eq!(
                    binding.execution.capability_id,
                    metadata.capability.as_str()
                );

                let transport = bundle
                    .contract_for_streaming(
                        metadata.capability,
                        StreamingRequirements::transport_only(),
                    )
                    .unwrap_or_else(|error| {
                        panic!("failed transport-only contract for {variant}: {error}")
                    });
                assert_eq!(transport.metadata, metadata);

                let native_streaming = bundle.contract(metadata.capability, true);
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
    fn every_compatibility_adapter_uses_the_configured_independent_row_width() {
        let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(1, 3).unwrap();

        for (index, variant) in ModelVariant::all().iter().copied().enumerate() {
            let bundle = LoadedModelBundle::bind(
                &registry,
                ExecutionGroupId::new(9),
                ModelInstanceId::new(index as u64 + 1),
                variant,
                BackendKind::Cpu,
            )
            .unwrap_or_else(|error| panic!("failed to bind {variant}: {error}"));
            for metadata in registry.capabilities_for(variant) {
                if registry
                    .loaded_adapter_factory(metadata, BackendKind::Cpu)
                    .unwrap()
                    .is_some()
                {
                    continue;
                }
                let contract = bundle
                    .contract(metadata.capability, false)
                    .unwrap_or_else(|error| panic!("failed to contract {variant}: {error}"));
                assert_eq!(contract.adapter_abi_revision, COMPATIBILITY_ADAPTER_ABI);
                assert_eq!(contract.execution_profile.max_batch_size, 3);
                assert_eq!(
                    contract.execution_profile.concurrency,
                    ConcurrencyClass::Batchable
                );
                assert_eq!(contract.stages[0].max_batch_size, 3);
                assert_eq!(
                    contract.stages[0].shape_policy,
                    crate::engine::StageShapePolicy::Independent
                );
            }
        }

        let metal = LoadedModelBundle::bind(
            &registry,
            ExecutionGroupId::new(9),
            ModelInstanceId::new(999),
            ModelVariant::Gemma31BIt,
            BackendKind::Metal,
        )
        .unwrap();
        let contract = metal.contract(CapabilityKind::Chat, false).unwrap();
        assert_eq!(contract.execution_profile.max_batch_size, 1);
        assert_eq!(
            contract.execution_profile.concurrency,
            ConcurrencyClass::Exclusive
        );
    }

    #[test]
    fn voxtral_streaming_binds_to_its_exact_token_engine_adapter() {
        let variant = ModelVariant::VoxtralMini4BRealtime2602;
        let bundle = LoadedModelBundle::bind(
            &RuntimeAdapterRegistry::built_in(),
            ExecutionGroupId::new(7),
            ModelInstanceId::new(1),
            variant,
            BackendKind::Cpu,
        )
        .unwrap();

        let contract = bundle.contract(CapabilityKind::Asr, true).unwrap();
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
    }

    #[test]
    fn offline_asr_transport_progress_does_not_require_native_streaming() {
        let bundle = LoadedModelBundle::bind(
            &RuntimeAdapterRegistry::built_in(),
            ExecutionGroupId::new(7),
            ModelInstanceId::new(1),
            ModelVariant::ParakeetTdt06BV3,
            BackendKind::Cpu,
        )
        .unwrap();

        assert!(bundle.contract(CapabilityKind::Asr, true).is_err());
        let transport = bundle
            .contract_for_streaming(CapabilityKind::Asr, StreamingRequirements::transport_only())
            .expect("offline ASR must expose atomic executor progress");
        assert_eq!(transport.metadata.streaming_mode, StreamingMode::None);
        assert_eq!(transport.execution_profile.mode, ExecutionMode::Atomic);
        assert_eq!(
            transport.stages[0].output_visibility,
            OutputVisibility::IncrementalCommitted
        );
    }

    #[test]
    fn atomic_chunked_chat_opt_in_is_streaming_specific() {
        let bundle = LoadedModelBundle::bind(
            &RuntimeAdapterRegistry::built_in(),
            ExecutionGroupId::new(7),
            ModelInstanceId::new(1),
            ModelVariant::Lfm2512BThinkingGguf,
            BackendKind::Cpu,
        )
        .unwrap();

        let non_streaming = bundle.contract(CapabilityKind::Chat, false).unwrap();
        assert_eq!(non_streaming.execution_profile.mode, ExecutionMode::Atomic);
        assert_eq!(
            non_streaming.stages[0].output_visibility,
            OutputVisibility::AfterQuantumCommit
        );

        let streaming = bundle.contract(CapabilityKind::Chat, true).unwrap();
        assert_eq!(streaming.execution_profile.mode, ExecutionMode::Atomic);
        assert_eq!(
            streaming.stages[0].output_visibility,
            OutputVisibility::IncrementalCommitted
        );
    }

    #[test]
    fn sequence_chat_remains_quantum_committed_when_streaming() {
        let bundle = LoadedModelBundle::bind(
            &RuntimeAdapterRegistry::built_in(),
            ExecutionGroupId::new(7),
            ModelInstanceId::new(1),
            ModelVariant::Qwen3508BGguf,
            BackendKind::Cpu,
        )
        .unwrap();

        let streaming = bundle.contract(CapabilityKind::Chat, true).unwrap();
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
            ModelVariant::Lfm25Audio15BGguf,
            BackendKind::Metal,
        )
        .expect("first bundle");
        let second = LoadedModelBundle::bind(
            &registry,
            ExecutionGroupId::new(1),
            ModelInstanceId::new(2),
            ModelVariant::Lfm25Audio15BGguf,
            BackendKind::Metal,
        )
        .expect("second bundle");

        let first_asr = first
            .capability_binding_for_streaming(CapabilityKind::Asr, StreamingRequirements::NONE)
            .expect("first asr")
            .execution
            .adapter_instance_id;
        let first_tts = first
            .capability_binding_for_streaming(CapabilityKind::Tts, StreamingRequirements::NONE)
            .expect("first tts")
            .execution
            .adapter_instance_id;
        let second_asr = second
            .capability_binding_for_streaming(CapabilityKind::Asr, StreamingRequirements::NONE)
            .expect("second asr")
            .execution
            .adapter_instance_id;

        assert_ne!(first_asr, first_tts);
        assert_ne!(first_asr, second_asr);
    }

    #[test]
    fn registering_a_factory_adds_an_optimized_model_without_bundle_branching() {
        let variant = ModelVariant::Kokoro82M;
        let mut registry = RuntimeAdapterRegistry::built_in();
        registry
            .loaded_adapter_factories
            .push(Arc::new(TestStaticTtsFactory {
                id: "test.kokoro.tensor_static",
                model_variant: variant,
            }));
        registry.validate_loaded_adapter_factories().unwrap();

        let bundle = LoadedModelBundle::bind(
            &registry,
            ExecutionGroupId::new(1),
            ModelInstanceId::new(2),
            variant,
            BackendKind::Cpu,
        )
        .unwrap();
        let contract = bundle.contract(CapabilityKind::Tts, false).unwrap();

        assert_eq!(contract.adapter_abi_revision, STATIC_TENSOR_ADAPTER_ABI);
        assert_eq!(contract.stages[0].batch_mode, NativeBatchMode::Static);
        assert!(registry
            .static_tensor_batch_variants(BackendKind::Cpu)
            .contains(&variant));
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
    fn qwen_tts_native_factory_binds_static_generation_but_not_streaming() {
        let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(4, 1).unwrap();
        let bundle = LoadedModelBundle::bind(
            &registry,
            ExecutionGroupId::new(1),
            ModelInstanceId::new(2),
            variant,
            BackendKind::Metal,
        )
        .unwrap();

        let batch = bundle.contract(CapabilityKind::Tts, false).unwrap();
        assert_eq!(batch.adapter_abi_revision, STATIC_TENSOR_ADAPTER_ABI);
        assert_eq!(batch.execution_profile.mode, ExecutionMode::Atomic);
        assert_eq!(
            batch.execution_profile.prefill_batch,
            NativeBatchMode::Static
        );
        assert_eq!(batch.execution_profile.max_batch_size, 4);
        assert_eq!(batch.stages.len(), 2);
        assert_eq!(batch.stages[0].selector, StageWorkSelector::Atomic);
        assert_eq!(batch.stages[0].batch_mode, NativeBatchMode::Static);
        assert_eq!(batch.stages[0].max_batch_size, 4);
        assert_eq!(
            batch.stages[0].shape_policy,
            crate::engine::StageShapePolicy::Exact
        );
        assert_eq!(batch.stages[1].selector, StageWorkSelector::Any);
        assert_eq!(batch.stages[1].batch_mode, NativeBatchMode::None);

        let streaming = bundle.contract(CapabilityKind::Tts, true).unwrap();
        assert_eq!(streaming.adapter_abi_revision, STATIC_TENSOR_ADAPTER_ABI);
        assert_eq!(
            streaming.execution_profile.prefill_batch,
            NativeBatchMode::None
        );
        assert_eq!(streaming.execution_profile.max_batch_size, 1);
        assert_eq!(streaming.stages[0].batch_mode, NativeBatchMode::None);

        let streaming_capability = bundle
            .contract(CapabilityKind::StreamingTts, false)
            .unwrap();
        assert_eq!(
            streaming_capability.adapter_abi_revision,
            COMPATIBILITY_ADAPTER_ABI
        );
    }

    #[test]
    fn qwen_tts_native_factory_is_enabled_on_cpu_by_default() {
        let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(4, 1).unwrap();
        let bundle = LoadedModelBundle::bind(
            &registry,
            ExecutionGroupId::new(1),
            ModelInstanceId::new(2),
            variant,
            BackendKind::Cpu,
        )
        .unwrap();

        let contract = bundle.contract(CapabilityKind::Tts, false).unwrap();
        assert_eq!(contract.adapter_abi_revision, STATIC_TENSOR_ADAPTER_ABI);
        assert_eq!(contract.execution_profile.max_batch_size, 4);
        assert_eq!(contract.stages[0].batch_mode, NativeBatchMode::Static);
    }

    #[test]
    fn qwen_chat_native_factory_publishes_scalar_prefill_and_ragged_decode() {
        let variant = ModelVariant::Qwen306B;
        let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(8, 1).unwrap();
        let bundle = LoadedModelBundle::bind(
            &registry,
            ExecutionGroupId::new(1),
            ModelInstanceId::new(2),
            variant,
            BackendKind::Cuda,
        )
        .unwrap();

        let contract = bundle.contract(CapabilityKind::Chat, true).unwrap();
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
}
