//! Continuous ASR execution adapters: Qwen3-ASR, Parakeet and LFM2.5-Audio.

use super::*;

#[derive(Debug)]
pub(super) struct ContinuousAsrExecutionAdapter {
    pub(super) execution_group_id: ExecutionGroupId,
    pub(super) model_instance_id: ModelInstanceId,
    pub(super) adapter_instance_id: AdapterInstanceId,
    pub(super) metadata: AdapterMetadata,
    pub(super) backend_kind: BackendKind,
    pub(super) max_batch_size: usize,
    pub(super) audio_preparation:
        OnceLock<crate::models::architectures::qwen3::asr::Qwen3AsrAudioPreparationStageSeal>,
}

impl ContinuousAsrExecutionAdapter {
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
            audio_preparation: OnceLock::new(),
        }
    }

    pub(super) fn install_audio_preparation_seal(
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
pub(super) struct ParakeetAsrExecutionAdapter {
    pub(super) execution_group_id: ExecutionGroupId,
    pub(super) model_instance_id: ModelInstanceId,
    pub(super) adapter_instance_id: AdapterInstanceId,
    pub(super) metadata: AdapterMetadata,
    pub(super) backend_kind: BackendKind,
    pub(super) max_batch_size: usize,
}

impl ParakeetAsrExecutionAdapter {
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
pub(super) struct Lfm25AudioAsrExecutionAdapter {
    pub(super) execution_group_id: ExecutionGroupId,
    pub(super) model_instance_id: ModelInstanceId,
    pub(super) adapter_instance_id: AdapterInstanceId,
    pub(super) metadata: AdapterMetadata,
    pub(super) backend_kind: BackendKind,
    pub(super) max_batch_size: usize,
    pub(super) preparation: OnceLock<
        crate::models::architectures::lfm25_audio::model::Lfm25AudioAsrPreparationStageCeiling,
    >,
}

impl Lfm25AudioAsrExecutionAdapter {
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
