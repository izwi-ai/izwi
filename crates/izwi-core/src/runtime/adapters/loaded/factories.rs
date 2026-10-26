//! The built-in loaded-adapter factories; each selects its capability table rows.

use super::*;

#[derive(Debug, Clone, Copy)]
pub(super) struct PhysicalQwenTtsAdapterFactory;

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
pub(super) struct NemotronRealtimeAdapterFactory;

#[derive(Debug, Clone, Copy)]
pub(super) struct VoxtralRealtimeAdapterFactory;

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
pub(super) struct ContinuousPhysicalChatAdapterFactory;

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
pub(super) struct ContinuousPhysicalAsrAdapterFactory;

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
pub(super) struct ScalarExecutionAdapterFactory;

#[derive(Debug, Clone, Copy)]
pub(super) struct WhisperPhysicalAsrAdapterFactory;

#[derive(Debug, Clone, Copy)]
pub(super) struct VibeVoicePhysicalAsrAdapterFactory;

#[derive(Debug, Clone, Copy)]
pub(super) struct GraniteSpeechPhysicalAsrAdapterFactory;

#[derive(Debug, Clone, Copy)]
pub(super) struct Lfm25AudioPhysicalAsrAdapterFactory;

#[derive(Debug, Clone, Copy)]
pub(super) struct ParakeetPhysicalAsrAdapterFactory;

#[derive(Debug, Clone, Copy)]
pub(super) struct Lfm25AudioPhysicalTtsAdapterFactory;

#[derive(Debug, Clone, Copy)]
pub(super) struct VibeVoicePhysicalTtsAdapterFactory;

#[derive(Debug, Clone, Copy)]
pub(super) struct FishS2PhysicalTtsAdapterFactory;

#[derive(Debug, Clone, Copy)]
pub(super) struct VoxtralPhysicalTtsAdapterFactory;

#[derive(Debug, Clone, Copy)]
pub(super) struct ChatterboxPhysicalTtsAdapterFactory;

impl LoadedExecutionAdapterFactory for ChatterboxPhysicalTtsAdapterFactory {
    fn id(&self) -> &'static str {
        LoadedAdapterKind::ChatterboxTts.factory_id()
    }
    fn batch_mode(&self) -> NativeBatchMode {
        NativeBatchMode::Continuous
    }
    fn supports(&self, metadata: AdapterMetadata, backend_kind: BackendKind) -> bool {
        table_selects(LoadedAdapterKind::ChatterboxTts, metadata, backend_kind)
    }
    fn create(
        &self,
        context: LoadedAdapterFactoryContext,
        metadata: AdapterMetadata,
    ) -> Result<Arc<dyn LoadedExecutionAdapter>> {
        Ok(Arc::new(ChatterboxTtsExecutionAdapter::new(
            context.execution_group_id,
            context.model_instance_id,
            metadata,
            context.backend_kind,
            context.max_tensor_batch_size,
        )))
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct KokoroStaticTtsAdapterFactory;

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
        // Continuous decode batching lives on the retained sequence graph,
        // which the adapter offers only with resumable prefill (GRN-8).
        if crate::models::architectures::granite_speech::asr::GraniteSpeechAsrModel::RESUMABLE_PREFILL
        {
            NativeBatchMode::Continuous
        } else {
            NativeBatchMode::None
        }
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

pub(in crate::runtime::adapters) fn built_in_loaded_adapter_factories() -> Vec<Arc<dyn LoadedExecutionAdapterFactory>> {
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
        Arc::new(ChatterboxPhysicalTtsAdapterFactory),
        Arc::new(KokoroStaticTtsAdapterFactory),
        Arc::new(ScalarExecutionAdapterFactory),
    ]
}
