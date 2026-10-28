//! Realtime ASR execution adapters: Voxtral and LFM3-Audio (paged) and
//! Nemotron (tensor).

use super::*;

#[derive(Debug)]
pub(super) struct VoxtralRealtimeExecutionAdapter {
    pub(super) execution_group_id: ExecutionGroupId,
    pub(super) model_instance_id: ModelInstanceId,
    pub(super) adapter_instance_id: AdapterInstanceId,
    pub(super) metadata: AdapterMetadata,
    pub(super) backend_kind: BackendKind,
    pub(super) max_batch_size: usize,
    pub(super) preparation: OnceLock<
        crate::models::architectures::voxtral::realtime::VoxtralRealtimePreparationStageSeal,
    >,
}

impl VoxtralRealtimeExecutionAdapter {
    pub(super) fn new(
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

    pub(super) fn install_preparation_seal(
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

/// LFM3-Audio LiveASR sessions as retained rows: preparation hears a push
/// (mel features and the streaming encoder, static-batched across sessions,
/// each row its own), each 0.96 s chunk prefills alone, and its text
/// decodes one token per step, batched across sessions. The language model's pages live
/// in the model's managed arena; the ShortConv ring and the encoder stream
/// are each session's own checkpointed host state, so every stage selects no
/// retained tensor state.
#[derive(Debug)]
pub(super) struct Lfm3AudioRealtimeExecutionAdapter {
    pub(super) execution_group_id: ExecutionGroupId,
    pub(super) model_instance_id: ModelInstanceId,
    pub(super) adapter_instance_id: AdapterInstanceId,
    pub(super) metadata: AdapterMetadata,
    pub(super) backend_kind: BackendKind,
    pub(super) max_batch_size: usize,
}

impl Lfm3AudioRealtimeExecutionAdapter {
    /// Positions one chunk's block feeds: the prompt or `<|text_end|>` and
    /// the chunk's 12 encoder tokens.
    const MAX_PREFILL_POSITIONS: u64 = 256;
    /// One feed's activations and a text step's logits row.
    const STEP_WORKSPACE_BYTES: u64 = 64 * 1024 * 1024;

    pub(super) fn new(
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

impl LoadedExecutionAdapter for Lfm3AudioRealtimeExecutionAdapter {
    fn metadata(&self) -> AdapterMetadata {
        self.metadata
    }

    fn adapter_instance_id(&self) -> AdapterInstanceId {
        self.adapter_instance_id
    }

    fn adapter_abi_revision(&self) -> AdapterAbiRevision {
        LFM3_AUDIO_REALTIME_ADAPTER_ABI
    }

    fn contract(&self, _streaming: StreamingRequirements) -> Result<LoadedExecutionContract> {
        use crate::models::architectures::lfm3_audio::{
            LFM3_AUDIO_REALTIME_COMPLETION_STAGE, LFM3_AUDIO_REALTIME_DECODE_STAGE,
            LFM3_AUDIO_REALTIME_MAX_PREPARATION_FRAMES, LFM3_AUDIO_REALTIME_PREPARATION_STAGE,
            LFM3_AUDIO_REALTIME_PREPARATION_WORKSPACE_BYTES,
            LFM3_AUDIO_REALTIME_PROMPT_PREFILL_STAGE,
        };
        let metadata = self.metadata();
        let mut execution_profile = scalar_execution_profile(metadata, self.backend_kind);
        execution_profile.mode = ExecutionMode::Realtime;
        execution_profile.prefill = PrefillMode::None;
        execution_profile.incremental_decode = true;
        execution_profile.prefill_batch = NativeBatchMode::Static;
        execution_profile.decode_batch = NativeBatchMode::Continuous;
        execution_profile.cache_mode = CacheMode::ExternalPaged;
        execution_profile.cache_namespace = Some(format!(
            "{}:{}:lfm3-audio-realtime-state-v2",
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
        let width = self.max_batch_size as u64;

        let mut preparation = StageDescriptor::from_execution_profile(
            StageId::new(0),
            LFM3_AUDIO_REALTIME_PREPARATION_STAGE,
            &execution_profile,
            NativeBatchMode::Static,
        );
        preparation.selector = StageWorkSelector::RealtimePreparation;
        preparation.max_batch_size = self.max_batch_size;
        preparation.max_work_units =
            LFM3_AUDIO_REALTIME_MAX_PREPARATION_FRAMES.saturating_mul(width);
        preparation.workspace_per_row_bytes = LFM3_AUDIO_REALTIME_PREPARATION_WORKSPACE_BYTES;
        preparation.max_workspace_bytes =
            LFM3_AUDIO_REALTIME_PREPARATION_WORKSPACE_BYTES.saturating_mul(width);
        preparation.shape_policy = StageShapePolicy::Ragged;
        preparation.max_padding_basis_points = 0;
        preparation.retained_state_selections = Some(vec![]);

        let mut prompt = StageDescriptor::from_execution_profile(
            StageId::new(1),
            LFM3_AUDIO_REALTIME_PROMPT_PREFILL_STAGE,
            &execution_profile,
            NativeBatchMode::None,
        );
        prompt.selector = StageWorkSelector::RealtimePromptPrefill;
        prompt.max_batch_size = 1;
        prompt.max_work_units = Self::MAX_PREFILL_POSITIONS;
        prompt.workspace_per_row_bytes = Self::STEP_WORKSPACE_BYTES;
        prompt.max_workspace_bytes = Self::STEP_WORKSPACE_BYTES;
        prompt.concurrency = ConcurrencyClass::Exclusive;
        prompt.shape_policy = StageShapePolicy::Exact;

        let mut decode = StageDescriptor::from_execution_profile(
            StageId::new(2),
            LFM3_AUDIO_REALTIME_DECODE_STAGE,
            &execution_profile,
            NativeBatchMode::Continuous,
        );
        decode.selector = StageWorkSelector::RealtimeDecodeContinuation;
        decode.progress = StageProgressKind::Iterative;
        decode.membership_safe_point = MembershipSafePoint::QuantumBoundary;
        decode.max_batch_size = self.max_batch_size;
        decode.max_work_units = width;
        decode.workspace_per_row_bytes = Self::STEP_WORKSPACE_BYTES;
        decode.max_workspace_bytes = Self::STEP_WORKSPACE_BYTES.saturating_mul(width);
        decode.shape_policy = StageShapePolicy::Ragged;
        decode.retained_state_selections = Some(vec![]);

        let mut completion = StageDescriptor::from_execution_profile(
            StageId::new(3),
            LFM3_AUDIO_REALTIME_COMPLETION_STAGE,
            &execution_profile,
            NativeBatchMode::None,
        );
        completion.selector = StageWorkSelector::RealtimeCompletion;
        completion.max_batch_size = 1;
        completion.max_work_units = 1;
        completion.concurrency = ConcurrencyClass::Exclusive;
        completion.shape_policy = StageShapePolicy::Exact;

        for stage in [&mut preparation, &mut prompt, &mut decode, &mut completion] {
            stage.output_visibility = OutputVisibility::AfterQuantumCommit;
            stage.validate()?;
        }
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
pub(super) struct NemotronRealtimeExecutionAdapter {
    pub(super) execution_group_id: ExecutionGroupId,
    pub(super) model_instance_id: ModelInstanceId,
    pub(super) adapter_instance_id: AdapterInstanceId,
    pub(super) metadata: AdapterMetadata,
    pub(super) backend_kind: BackendKind,
    pub(super) max_batch_size: usize,
}

impl NemotronRealtimeExecutionAdapter {
    pub(super) fn new(
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
