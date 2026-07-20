use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::backends::BackendKind;
use crate::engine::{
    AdapterAbiRevision, AdapterInstanceId, CacheMode, CancellationGranularity, ConcurrencyClass,
    ExecutionAdapterBinding, ExecutionGroupId, ExecutionMode, ExecutionProfile, ModelInstanceId,
    NativeBatchMode, PrefillMode, StageDescriptor, StageId, StageWorkSelector,
};
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::runtime::rollout::ExecutionRolloutMode;

use super::{
    compatibility_execution_profile, AdapterMetadata, CapabilityKind, RuntimeAdapterRegistry,
    StreamingMode,
};

const COMPATIBILITY_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(4);
const STATIC_TENSOR_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(5);
const CONTINUOUS_TENSOR_ADAPTER_ABI: AdapterAbiRevision = AdapterAbiRevision::new(6);
const STATIC_TTS_MAX_BATCH_WORKSPACE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const CONTINUOUS_CHAT_MAX_BATCH_WORKSPACE_BYTES: u64 = 16 * 1024 * 1024;
static NEXT_ADAPTER_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

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
    fn contract(&self, streaming_required: bool) -> Result<LoadedExecutionContract>;
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
    fn rollout_mode(&self) -> ExecutionRolloutMode;
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

    fn rollout_mode(&self) -> ExecutionRolloutMode {
        ExecutionRolloutMode::Static
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

    fn rollout_mode(&self) -> ExecutionRolloutMode {
        ExecutionRolloutMode::Continuous
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

    fn contract(&self, streaming_required: bool) -> Result<LoadedExecutionContract> {
        compatibility_contract(
            self.execution_group_id,
            self.model_instance_id,
            self.adapter_instance_id(),
            self.adapter_abi_revision(),
            self.metadata(),
            self.backend_kind,
            self.request_parallelism,
            streaming_required,
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
    streaming_required: bool,
) -> Result<LoadedExecutionContract> {
    if streaming_required && metadata.streaming_mode == StreamingMode::None {
        return Err(Error::InvalidInput(format!(
            "Model {} supports {:?}, but not streaming execution for that capability",
            metadata.model_variant, metadata.capability
        )));
    }

    let mut execution_profile =
        compatibility_execution_profile(metadata, backend_kind, streaming_required);
    execution_profile.resolved_from_loaded_model = true;
    execution_profile.prefill_batch = NativeBatchMode::None;
    execution_profile.decode_batch = NativeBatchMode::None;
    execution_profile.max_batch_size = request_parallelism.max(1);
    execution_profile.concurrency = if execution_profile.max_batch_size > 1 {
        ConcurrencyClass::Batchable
    } else {
        ConcurrencyClass::Exclusive
    };

    let stage = StageDescriptor::from_execution_profile(
        StageId::new(0),
        format!("{}.compatibility", metadata.capability.as_str()),
        &execution_profile,
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

    fn contract(&self, streaming_required: bool) -> Result<LoadedExecutionContract> {
        if streaming_required {
            return compatibility_contract(
                self.execution_group_id,
                self.model_instance_id,
                self.adapter_instance_id(),
                self.adapter_abi_revision(),
                self.metadata(),
                self.backend_kind,
                self.request_parallelism,
                true,
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

    fn contract(&self, streaming_required: bool) -> Result<LoadedExecutionContract> {
        let metadata = self.metadata();
        if streaming_required && metadata.streaming_mode == StreamingMode::None {
            return Err(Error::InvalidInput(format!(
                "Model {} has no streaming chat contract",
                metadata.model_variant
            )));
        }
        let mut execution_profile =
            compatibility_execution_profile(metadata, self.backend_kind, streaming_required);
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
    fn loaded_adapter_factory(
        &self,
        metadata: AdapterMetadata,
        backend_kind: BackendKind,
        rollout_mode: ExecutionRolloutMode,
    ) -> Result<Option<&dyn LoadedExecutionAdapterFactory>> {
        if rollout_mode == ExecutionRolloutMode::Off {
            return Ok(None);
        }
        let mut matches = self.loaded_adapter_factories.iter().filter(|factory| {
            factory.rollout_mode() == rollout_mode && factory.supports(metadata, backend_kind)
        });
        let Some(selected) = matches.next() else {
            return Ok(None);
        };
        if let Some(ambiguous) = matches.next() {
            return Err(Error::ModelLoadError(format!(
                "loaded model {} capability {:?} matches both `{}` and `{}` for {rollout_mode:?} rollout",
                metadata.model_variant,
                metadata.capability,
                selected.id(),
                ambiguous.id(),
            )));
        }
        Ok(Some(selected.as_ref()))
    }

    pub(super) fn supports_loaded_rollout(
        &self,
        model_variant: ModelVariant,
        backend_kind: BackendKind,
        rollout_mode: ExecutionRolloutMode,
    ) -> Result<bool> {
        if rollout_mode == ExecutionRolloutMode::Off {
            return Ok(true);
        }
        for metadata in self.capabilities_for(model_variant) {
            if self
                .loaded_adapter_factory(metadata, backend_kind, rollout_mode)?
                .is_some()
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(super) fn loaded_rollout_variants(
        &self,
        backend_kind: BackendKind,
        rollout_mode: ExecutionRolloutMode,
    ) -> std::collections::HashSet<ModelVariant> {
        ModelVariant::all()
            .iter()
            .copied()
            .filter(|variant| {
                self.execution_mode_for(*variant, backend_kind) == rollout_mode
                    && self
                        .supports_loaded_rollout(*variant, backend_kind, rollout_mode)
                        .unwrap_or(false)
            })
            .collect()
    }

    fn bind_loaded_adapter(
        &self,
        execution_group_id: ExecutionGroupId,
        model_instance_id: ModelInstanceId,
        metadata: AdapterMetadata,
        backend_kind: BackendKind,
    ) -> Result<Arc<dyn LoadedExecutionAdapter>> {
        let rollout_mode = self.execution_mode_for(metadata.model_variant, backend_kind);
        let context = LoadedAdapterFactoryContext {
            execution_group_id,
            model_instance_id,
            backend_kind,
            max_tensor_batch_size: self.max_tensor_batch_size(),
            request_parallelism: self.request_parallelism(),
        };
        let adapter = match self.loaded_adapter_factory(metadata, backend_kind, rollout_mode)? {
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
        adapter.contract(false)?;
        Ok(adapter)
    }
}

pub(crate) struct LoadedModelBundle {
    execution_group_id: ExecutionGroupId,
    model_instance_id: ModelInstanceId,
    model_variant: ModelVariant,
    backend_kind: BackendKind,
    adapters: HashMap<CapabilityKind, Arc<dyn LoadedExecutionAdapter>>,
}

impl fmt::Debug for LoadedModelBundle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LoadedModelBundle")
            .field("execution_group_id", &self.execution_group_id)
            .field("model_instance_id", &self.model_instance_id)
            .field("model_variant", &self.model_variant)
            .field("backend_kind", &self.backend_kind)
            .field("adapter_count", &self.adapters.len())
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
        let metadata = registry.capabilities_for(model_variant);
        if metadata.is_empty() {
            return Err(Error::ModelLoadError(format!(
                "loaded model {model_variant} has no executable capability adapter"
            )));
        }

        let mut adapters = HashMap::with_capacity(metadata.len());
        for metadata in metadata {
            let adapter = registry.bind_loaded_adapter(
                execution_group_id,
                model_instance_id,
                metadata,
                backend_kind,
            )?;
            if adapters.insert(metadata.capability, adapter).is_some() {
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
            adapters,
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
        self.adapters.len()
    }

    pub(crate) fn require_adapter(
        &self,
        capability: CapabilityKind,
    ) -> Result<Arc<dyn LoadedExecutionAdapter>> {
        self.adapters.get(&capability).cloned().ok_or_else(|| {
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
        self.require_adapter(capability)?
            .contract(streaming_required)
    }

    pub(crate) fn adapter_binding(
        &self,
        capability: CapabilityKind,
        streaming_required: bool,
    ) -> Result<ExecutionAdapterBinding> {
        self.contract(capability, streaming_required)?
            .adapter_binding()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::adapters::ExecutionTargetKind;
    use crate::runtime::rollout::ExecutionRolloutPolicy;

    #[derive(Debug)]
    struct TestStaticTtsFactory {
        id: &'static str,
        model_variant: ModelVariant,
    }

    impl LoadedExecutionAdapterFactory for TestStaticTtsFactory {
        fn id(&self) -> &'static str {
            self.id
        }

        fn rollout_mode(&self) -> ExecutionRolloutMode {
            ExecutionRolloutMode::Static
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
                let contract = bundle
                    .contract(metadata.capability, false)
                    .unwrap_or_else(|error| panic!("failed to contract {variant}: {error}"));
                assert_eq!(contract.execution_group_id, ExecutionGroupId::new(7));
                assert_eq!(contract.model_instance_id, instance);
                assert_eq!(contract.metadata, metadata);
                assert_eq!(contract.adapter_abi_revision, COMPATIBILITY_ADAPTER_ABI);
                assert_eq!(contract.stages.len(), 1);
                assert_eq!(contract.stages[0].max_batch_size, 1);
                assert_eq!(contract.stages[0].batch_mode, NativeBatchMode::None);
                assert_eq!(contract.execution_profile.max_batch_size, 1);
                assert!(contract.execution_profile.resolved_from_loaded_model);
                let binding = bundle
                    .adapter_binding(metadata.capability, false)
                    .expect("adapter binding");
                assert_eq!(binding.model_variant, variant);
                assert_eq!(binding.model_instance_id, instance);
                assert_eq!(binding.capability_id, metadata.capability.as_str());
            }
        }
    }

    #[test]
    fn every_compatibility_adapter_uses_the_configured_independent_row_width() {
        let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(
            ExecutionRolloutPolicy::default(),
            1,
            3,
        )
        .unwrap();

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
            ModelVariant::Qwen306B,
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
            .require_adapter(CapabilityKind::Asr)
            .expect("first asr")
            .adapter_instance_id();
        let first_tts = first
            .require_adapter(CapabilityKind::Tts)
            .expect("first tts")
            .adapter_instance_id();
        let second_asr = second
            .require_adapter(CapabilityKind::Asr)
            .expect("second asr")
            .adapter_instance_id();

        assert_ne!(first_asr, first_tts);
        assert_ne!(first_asr, second_asr);
    }

    #[test]
    fn registering_a_factory_adds_an_optimized_model_without_bundle_branching() {
        let variant = ModelVariant::Kokoro82M;
        let rollout = ExecutionRolloutPolicy::try_from_raw(
            Some("off"),
            Some(&format!("{variant}@cpu=static")),
        )
        .unwrap();
        let mut registry = RuntimeAdapterRegistry::built_in();
        registry.execution_rollout = rollout;
        registry
            .loaded_adapter_factories
            .push(Arc::new(TestStaticTtsFactory {
                id: "test.kokoro.tensor_static",
                model_variant: variant,
            }));
        registry.validate_execution_rollout().unwrap();

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
        let rollout = ExecutionRolloutPolicy::try_from_raw(
            Some("off"),
            Some(&format!("{variant}@cpu=static")),
        )
        .unwrap();
        let mut registry = RuntimeAdapterRegistry::built_in_with_rollout(rollout, 2).unwrap();
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
    fn exact_qwen_tts_rollout_binds_static_generation_but_not_streaming() {
        let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        let override_value = format!("{}@metal=static", variant);
        let rollout =
            ExecutionRolloutPolicy::try_from_raw(Some("off"), Some(&override_value)).unwrap();
        let registry = RuntimeAdapterRegistry::built_in_with_rollout(rollout, 4).unwrap();
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
    fn static_rollout_does_not_cross_backend_boundaries() {
        let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        let override_value = format!("{}@metal=static", variant);
        let rollout =
            ExecutionRolloutPolicy::try_from_raw(Some("off"), Some(&override_value)).unwrap();
        let registry = RuntimeAdapterRegistry::built_in_with_rollout(rollout, 4).unwrap();
        let bundle = LoadedModelBundle::bind(
            &registry,
            ExecutionGroupId::new(1),
            ModelInstanceId::new(2),
            variant,
            BackendKind::Cpu,
        )
        .unwrap();

        let contract = bundle.contract(CapabilityKind::Tts, false).unwrap();
        assert_eq!(contract.adapter_abi_revision, COMPATIBILITY_ADAPTER_ABI);
        assert_eq!(contract.execution_profile.max_batch_size, 1);
        assert_eq!(contract.stages[0].batch_mode, NativeBatchMode::None);
    }

    #[test]
    fn qwen_chat_continuous_rollout_publishes_scalar_prefill_and_ragged_decode() {
        let variant = ModelVariant::Qwen306B;
        let override_value = format!("{}@cuda=continuous", variant);
        let rollout =
            ExecutionRolloutPolicy::try_from_raw(Some("off"), Some(&override_value)).unwrap();
        let registry = RuntimeAdapterRegistry::built_in_with_rollout(rollout, 8).unwrap();
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
        assert_eq!(
            contract.stages[1].max_workspace_bytes,
            CONTINUOUS_CHAT_MAX_BATCH_WORKSPACE_BYTES
        );
    }
}
