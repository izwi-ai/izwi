//! Sequence ASR execution adapters: Granite Speech, VibeVoice and Whisper.

use super::*;

#[derive(Debug)]
pub(super) struct GraniteSpeechAsrExecutionSeal {
    pub(super) preparation:
        crate::models::architectures::granite_speech::asr::GraniteSpeechAsrPreparationStageSeal,
    pub(super) preparation_max_batch_work_units: u64,
    pub(super) preparation_max_batch_materialized_tensor_elements: u64,
    pub(super) preparation_max_batch_workspace_bytes: u64,
    pub(super) decode_workspace_per_row_bytes: u64,
}

#[derive(Debug)]
pub(super) struct GraniteSpeechAsrExecutionAdapter {
    pub(super) execution_group_id: ExecutionGroupId,
    pub(super) model_instance_id: ModelInstanceId,
    pub(super) adapter_instance_id: AdapterInstanceId,
    pub(super) metadata: AdapterMetadata,
    pub(super) backend_kind: BackendKind,
    pub(super) max_batch_size: usize,
    pub(super) seal: OnceLock<GraniteSpeechAsrExecutionSeal>,
}

impl GraniteSpeechAsrExecutionAdapter {
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
            seal: OnceLock::new(),
        }
    }

    pub(super) fn install_preparation_seal(&self, seal: GraniteSpeechAsrExecutionSeal) -> Result<()> {
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

#[derive(Debug)]
pub(super) struct VibeVoiceAsrExecutionAdapter {
    pub(super) execution_group_id: ExecutionGroupId,
    pub(super) model_instance_id: ModelInstanceId,
    pub(super) adapter_instance_id: AdapterInstanceId,
    pub(super) metadata: AdapterMetadata,
    pub(super) backend_kind: BackendKind,
    pub(super) max_batch_size: usize,
    pub(super) seal: OnceLock<VibeVoiceAsrExecutionSeal>,
}

impl VibeVoiceAsrExecutionAdapter {
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
pub(super) struct WhisperAsrExecutionAdapter {
    pub(super) execution_group_id: ExecutionGroupId,
    pub(super) model_instance_id: ModelInstanceId,
    pub(super) adapter_instance_id: AdapterInstanceId,
    pub(super) metadata: AdapterMetadata,
    pub(super) backend_kind: BackendKind,
    pub(super) max_batch_size: usize,
    pub(super) audio_preparation:
        OnceLock<crate::models::architectures::whisper::asr::WhisperAudioPreparationStageSeal>,
}

impl WhisperAsrExecutionAdapter {
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
