//! TTS execution adapters: Qwen3-TTS, static Kokoro, Voxtral, Chatterbox, Maya1,
//! Maya 2 Native, Fish S2, VibeVoice and LFM2.5-Audio.

use super::*;

#[derive(Debug)]
pub(super) struct PhysicalQwenTtsExecutionAdapter {
    pub(super) execution_group_id: ExecutionGroupId,
    pub(super) model_instance_id: ModelInstanceId,
    pub(super) adapter_instance_id: AdapterInstanceId,
    pub(super) metadata: AdapterMetadata,
    pub(super) backend_kind: BackendKind,
    pub(super) max_batch_size: usize,
}

impl PhysicalQwenTtsExecutionAdapter {
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
pub(super) struct StaticTtsExecutionAdapter {
    pub(super) execution_group_id: ExecutionGroupId,
    pub(super) model_instance_id: ModelInstanceId,
    pub(super) adapter_instance_id: AdapterInstanceId,
    pub(super) metadata: AdapterMetadata,
    pub(super) backend_kind: BackendKind,
    pub(super) max_batch_size: usize,
    pub(super) request_parallelism: usize,
}

impl StaticTtsExecutionAdapter {
    pub(super) fn new(
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
pub(super) struct VoxtralTtsExecutionAdapter {
    pub(super) execution_group_id: ExecutionGroupId,
    pub(super) model_instance_id: ModelInstanceId,
    pub(super) adapter_instance_id: AdapterInstanceId,
    pub(super) metadata: AdapterMetadata,
    pub(super) backend_kind: BackendKind,
    pub(super) max_batch_size: usize,
}

impl VoxtralTtsExecutionAdapter {
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
pub(super) struct ChatterboxTtsExecutionAdapter {
    pub(super) execution_group_id: ExecutionGroupId,
    pub(super) model_instance_id: ModelInstanceId,
    pub(super) adapter_instance_id: AdapterInstanceId,
    pub(super) metadata: AdapterMetadata,
    pub(super) backend_kind: BackendKind,
    pub(super) max_batch_size: usize,
}

impl ChatterboxTtsExecutionAdapter {
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

impl LoadedExecutionAdapter for ChatterboxTtsExecutionAdapter {
    fn metadata(&self) -> AdapterMetadata {
        self.metadata
    }
    fn adapter_instance_id(&self) -> AdapterInstanceId {
        self.adapter_instance_id
    }
    fn adapter_abi_revision(&self) -> AdapterAbiRevision {
        CHATTERBOX_TTS_ADAPTER_ABI
    }

    fn contract(&self, streaming: StreamingRequirements) -> Result<LoadedExecutionContract> {
        // One graph serves both: a streamed request's acoustic quanta render
        // windows while T3 runs and stage each window's samples.
        if streaming.model_native && self.metadata.streaming_mode == StreamingMode::None {
            return Err(Error::InvalidInput(format!(
                "Model {} has no native streaming TTS contract",
                self.metadata.model_variant
            )));
        }
        let mut profile = scalar_execution_profile(self.metadata, self.backend_kind);
        profile.mode = ExecutionMode::Sequence;
        profile.prefill = PrefillMode::Incremental;
        profile.incremental_decode = true;
        profile.prefill_batch = NativeBatchMode::Static;
        profile.decode_batch = NativeBatchMode::Continuous;
        profile.cache_mode = CacheMode::ExternalPaged;
        profile.cache_namespace = Some(format!(
            "{}:tts:{}:chatterbox-state-v2",
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
            "tts.prepare.chatterbox",
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
            "tts.prefill.chatterbox.tensor_static",
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
            "tts.decode.chatterbox.tensor_continuous",
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
        // Hands over the rendered waveform.
        let mut finalize = StageDescriptor::from_execution_profile(
            StageId::new(3),
            "tts.finalize.chatterbox.scalar",
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
        // One acoustic unit per quantum: the S3Gen encoder, an Euler step,
        // or HiFT (of the whole utterance or of a streamed window), so other
        // requests' quanta run between them.
        let mut acoustic = StageDescriptor::from_execution_profile(
            StageId::new(4),
            "tts.acoustic.chatterbox.scalar",
            &profile,
            NativeBatchMode::None,
        );
        acoustic.selector = StageWorkSelector::SequenceAudioDecode;
        acoustic.progress = StageProgressKind::Iterative;
        acoustic.shape_policy = StageShapePolicy::Exact;
        acoustic.max_batch_size = 1;
        acoustic.concurrency = ConcurrencyClass::Exclusive;
        acoustic.max_work_units = 1;
        // S3Gen's flow attention over up to 2 × 1,250 tokens of frames, both
        // CFG rows, plus the vocoder's upsampled activations.
        acoustic.max_workspace_bytes = 1024 * 1024 * 1024;
        for stage in [
            &mut preparation,
            &mut prefill,
            &mut decode,
            &mut finalize,
            &mut acoustic,
        ] {
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
            stages: Arc::from([preparation, prefill, decode, finalize, acoustic]),
        })
    }
}

#[derive(Debug)]
pub(super) struct Maya1TtsExecutionAdapter {
    pub(super) execution_group_id: ExecutionGroupId,
    pub(super) model_instance_id: ModelInstanceId,
    pub(super) adapter_instance_id: AdapterInstanceId,
    pub(super) metadata: AdapterMetadata,
    pub(super) backend_kind: BackendKind,
    pub(super) max_batch_size: usize,
}

impl Maya1TtsExecutionAdapter {
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

impl LoadedExecutionAdapter for Maya1TtsExecutionAdapter {
    fn metadata(&self) -> AdapterMetadata {
        self.metadata
    }
    fn adapter_instance_id(&self) -> AdapterInstanceId {
        self.adapter_instance_id
    }
    fn adapter_abi_revision(&self) -> AdapterAbiRevision {
        MAYA1_TTS_ADAPTER_ABI
    }

    fn contract(&self, streaming: StreamingRequirements) -> Result<LoadedExecutionContract> {
        // One graph serves both: every request's audio quanta decode SNAC
        // windows while the decoder runs; a stream sends each one.
        if streaming.model_native && self.metadata.streaming_mode == StreamingMode::None {
            return Err(Error::InvalidInput(format!(
                "Model {} has no native streaming TTS contract",
                self.metadata.model_variant
            )));
        }
        let mut profile = scalar_execution_profile(self.metadata, self.backend_kind);
        profile.mode = ExecutionMode::Sequence;
        profile.prefill = PrefillMode::Incremental;
        profile.incremental_decode = true;
        profile.prefill_batch = NativeBatchMode::Static;
        profile.decode_batch = NativeBatchMode::Continuous;
        profile.cache_mode = CacheMode::ExternalPaged;
        profile.cache_namespace = Some(format!(
            "{}:tts:{}:maya1-state-v2",
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
            "tts.prepare.maya1",
            &profile,
            NativeBatchMode::None,
        );
        preparation.selector = StageWorkSelector::PreSequencePreparation;
        preparation.progress = StageProgressKind::Atomic;
        preparation.max_batch_size = 1;
        preparation.concurrency = ConcurrencyClass::Exclusive;
        preparation.shape_policy = StageShapePolicy::Exact;
        preparation.max_work_units = 16_384;
        // Normalising and tokenizing the prompt on the host.
        preparation.max_workspace_bytes = 64 * 1024 * 1024;

        let mut prefill = StageDescriptor::from_execution_profile(
            StageId::new(1),
            "tts.prefill.maya1.tensor_static",
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
            "tts.decode.maya1.tensor_continuous",
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
        // Hands over the audio (a stream's end-of-stream marker).
        let mut finalize = StageDescriptor::from_execution_profile(
            StageId::new(3),
            "tts.finalize.maya1.scalar",
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
        // One SNAC window per quantum, so other requests' quanta run between
        // them.
        let mut acoustic = StageDescriptor::from_execution_profile(
            StageId::new(4),
            "tts.audio.maya1.scalar",
            &profile,
            NativeBatchMode::None,
        );
        acoustic.selector = StageWorkSelector::SequenceAudioDecode;
        acoustic.progress = StageProgressKind::Iterative;
        acoustic.shape_policy = StageShapePolicy::Exact;
        acoustic.max_batch_size = 1;
        acoustic.concurrency = ConcurrencyClass::Exclusive;
        acoustic.max_work_units = 1;
        // SNAC's activations over a window of up to 22 frames (45,056
        // samples), the widest 12 MB a tensor.
        acoustic.max_workspace_bytes = 256 * 1024 * 1024;
        for stage in [
            &mut preparation,
            &mut prefill,
            &mut decode,
            &mut finalize,
            &mut acoustic,
        ] {
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
            stages: Arc::from([preparation, prefill, decode, finalize, acoustic]),
        })
    }
}

#[derive(Debug)]
pub(super) struct Maya2TtsExecutionAdapter {
    pub(super) execution_group_id: ExecutionGroupId,
    pub(super) model_instance_id: ModelInstanceId,
    pub(super) adapter_instance_id: AdapterInstanceId,
    pub(super) metadata: AdapterMetadata,
    pub(super) backend_kind: BackendKind,
    pub(super) max_batch_size: usize,
}

impl Maya2TtsExecutionAdapter {
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

impl LoadedExecutionAdapter for Maya2TtsExecutionAdapter {
    fn metadata(&self) -> AdapterMetadata {
        self.metadata
    }
    fn adapter_instance_id(&self) -> AdapterInstanceId {
        self.adapter_instance_id
    }
    fn adapter_abi_revision(&self) -> AdapterAbiRevision {
        MAYA2_TTS_ADAPTER_ABI
    }

    fn contract(&self, streaming: StreamingRequirements) -> Result<LoadedExecutionContract> {
        // One graph serves both: every request's audio quanta run Mimi over
        // its newest frames while the backbone runs; a stream sends each one.
        if streaming.model_native && self.metadata.streaming_mode == StreamingMode::None {
            return Err(Error::InvalidInput(format!(
                "Model {} has no native streaming TTS contract",
                self.metadata.model_variant
            )));
        }
        let mut profile = scalar_execution_profile(self.metadata, self.backend_kind);
        profile.mode = ExecutionMode::Sequence;
        profile.prefill = PrefillMode::Incremental;
        profile.incremental_decode = true;
        profile.prefill_batch = NativeBatchMode::Static;
        profile.decode_batch = NativeBatchMode::Continuous;
        profile.cache_mode = CacheMode::ExternalPaged;
        profile.cache_namespace = Some(format!(
            "{}:tts:{}:maya2-state-v2",
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
            "tts.prepare.maya2",
            &profile,
            NativeBatchMode::None,
        );
        preparation.selector = StageWorkSelector::PreSequencePreparation;
        preparation.progress = StageProgressKind::Atomic;
        preparation.max_batch_size = 1;
        preparation.concurrency = ConcurrencyClass::Exclusive;
        preparation.shape_policy = StageShapePolicy::Exact;
        preparation.max_work_units = 16_384;
        // Tokenizing the prompt on the host; a new clone reference also runs
        // Mimi's encoder (in bounded windows) and ECAPA over up to 30 s.
        preparation.max_workspace_bytes = 768 * 1024 * 1024;

        let mut prefill = StageDescriptor::from_execution_profile(
            StageId::new(1),
            "tts.prefill.maya2.tensor_static",
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
            "tts.decode.maya2.tensor_continuous",
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
        // Hands over the audio (a stream's end-of-stream marker).
        let mut finalize = StageDescriptor::from_execution_profile(
            StageId::new(3),
            "tts.finalize.maya2.scalar",
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
        // One Mimi window per quantum, so other requests' quanta run between
        // them.
        let mut acoustic = StageDescriptor::from_execution_profile(
            StageId::new(4),
            "tts.audio.maya2.scalar",
            &profile,
            NativeBatchMode::None,
        );
        acoustic.selector = StageWorkSelector::SequenceAudioDecode;
        acoustic.progress = StageProgressKind::Iterative;
        acoustic.shape_policy = StageShapePolicy::Exact;
        acoustic.max_batch_size = 1;
        acoustic.concurrency = ConcurrencyClass::Exclusive;
        acoustic.max_work_units = 1;
        // Mimi's activations over a window of a few 80 ms frames (1,920
        // samples each at up to 1,024 channels) and its per-stream state.
        acoustic.max_workspace_bytes = 64 * 1024 * 1024;
        for stage in [
            &mut preparation,
            &mut prefill,
            &mut decode,
            &mut finalize,
            &mut acoustic,
        ] {
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
            stages: Arc::from([preparation, prefill, decode, finalize, acoustic]),
        })
    }
}

/// LFM3-Audio speech as retained continuous rows: a scalar preparation
/// (the prompt's embeddings), a static prefill of each row's prompt, and a
/// continuous decode that steps every row one slot with the language model
/// batched across rows. The codec decodes inside the decode quantum, so the
/// graph has no separate audio stage.
#[derive(Debug)]
pub(super) struct Lfm3AudioTtsExecutionAdapter {
    pub(super) execution_group_id: ExecutionGroupId,
    pub(super) model_instance_id: ModelInstanceId,
    pub(super) adapter_instance_id: AdapterInstanceId,
    pub(super) metadata: AdapterMetadata,
    pub(super) backend_kind: BackendKind,
    pub(super) max_batch_size: usize,
}

impl Lfm3AudioTtsExecutionAdapter {
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

impl LoadedExecutionAdapter for Lfm3AudioTtsExecutionAdapter {
    fn metadata(&self) -> AdapterMetadata {
        self.metadata
    }
    fn adapter_instance_id(&self) -> AdapterInstanceId {
        self.adapter_instance_id
    }
    fn adapter_abi_revision(&self) -> AdapterAbiRevision {
        LFM3_AUDIO_TTS_ADAPTER_ABI
    }

    fn contract(&self, streaming: StreamingRequirements) -> Result<LoadedExecutionContract> {
        if streaming.model_native && self.metadata.streaming_mode == StreamingMode::None {
            return Err(Error::InvalidInput(format!(
                "Model {} has no native streaming TTS contract",
                self.metadata.model_variant
            )));
        }
        let mut profile = scalar_execution_profile(self.metadata, self.backend_kind);
        profile.mode = ExecutionMode::Sequence;
        profile.prefill = PrefillMode::Incremental;
        profile.incremental_decode = true;
        profile.prefill_batch = NativeBatchMode::Static;
        profile.decode_batch = NativeBatchMode::Continuous;
        profile.cache_mode = CacheMode::ExternalPaged;
        profile.cache_namespace = Some(format!(
            "{}:tts:{}:lfm3-audio-state-v2",
            self.metadata.model_variant,
            self.backend_kind.as_str()
        ));
        profile.kv_dtype = "state_v2_resolved".into();
        profile.cancellation = CancellationGranularity::SequenceStep;
        profile.concurrency = ConcurrencyClass::Batchable;
        // A recomputed row would draw its frames again; it cannot rebuild
        // the speech it already sent.
        profile.recompute_safe = false;
        profile.cache_release_safe = true;
        profile.max_batch_size = self.max_batch_size;
        profile.resolved_from_loaded_model = true;
        let rows = self.max_batch_size as u64;

        let mut preparation = StageDescriptor::from_execution_profile(
            StageId::new(0),
            "tts.prepare.lfm3_audio",
            &profile,
            NativeBatchMode::None,
        );
        preparation.selector = StageWorkSelector::PreSequencePreparation;
        preparation.progress = StageProgressKind::Atomic;
        preparation.max_batch_size = 1;
        preparation.concurrency = ConcurrencyClass::Exclusive;
        preparation.shape_policy = StageShapePolicy::Exact;
        preparation.max_work_units = 16_384;
        // Tokenizing the prompt and gathering its embeddings.
        preparation.max_workspace_bytes = 64 * 1024 * 1024;

        let mut prefill = StageDescriptor::from_execution_profile(
            StageId::new(1),
            "tts.prefill.lfm3_audio.tensor_static",
            &profile,
            NativeBatchMode::Static,
        );
        prefill.selector = StageWorkSelector::SequencePrefill;
        prefill.progress = StageProgressKind::Iterative;
        prefill.shape_policy = StageShapePolicy::Ragged;
        prefill.max_padding_basis_points = 0;
        prefill.max_work_units = 4_096_u64.saturating_mul(rows);
        // Each row's prompt prefills alone: one span's activations.
        prefill.max_workspace_bytes = (256_u64 * 1024 * 1024).saturating_mul(rows);

        let mut decode = StageDescriptor::from_execution_profile(
            StageId::new(2),
            "tts.decode.lfm3_audio.tensor_continuous",
            &profile,
            NativeBatchMode::Continuous,
        );
        decode.selector = StageWorkSelector::SequenceDecode;
        decode.progress = StageProgressKind::Iterative;
        decode.shape_policy = StageShapePolicy::Ragged;
        decode.max_work_units = 64_u64.saturating_mul(rows);
        // The depth decoder's passes and the codec's window, per row.
        decode.max_workspace_bytes = (128_u64 * 1024 * 1024).saturating_mul(rows);
        for stage in [&mut preparation, &mut prefill, &mut decode] {
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
            stages: Arc::from([preparation, prefill, decode]),
        })
    }
}

#[derive(Debug)]
pub(super) struct FishS2TtsExecutionAdapter {
    pub(super) execution_group_id: ExecutionGroupId,
    pub(super) model_instance_id: ModelInstanceId,
    pub(super) adapter_instance_id: AdapterInstanceId,
    pub(super) metadata: AdapterMetadata,
    pub(super) backend_kind: BackendKind,
    pub(super) capacity: OnceLock<crate::runtime::fish_capacity::ResolvedFishServingCapacity>,
}

impl FishS2TtsExecutionAdapter {
    pub(super) fn new(
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
pub(super) struct VibeVoiceTtsExecutionAdapter {
    pub(super) execution_group_id: ExecutionGroupId,
    pub(super) model_instance_id: ModelInstanceId,
    pub(super) adapter_instance_id: AdapterInstanceId,
    pub(super) metadata: AdapterMetadata,
    pub(super) backend_kind: BackendKind,
    pub(super) max_batch_size: usize,
}

impl VibeVoiceTtsExecutionAdapter {
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
        preparation.max_work_units =
            crate::models::architectures::vibevoice::VIBEVOICE_TTS_STAGE_MAX_TOKENS as u64;
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
        prefill.max_work_units =
            crate::models::architectures::vibevoice::VIBEVOICE_TTS_STAGE_MAX_TOKENS as u64;
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
pub(super) struct Lfm25AudioTtsExecutionAdapter {
    pub(super) execution_group_id: ExecutionGroupId,
    pub(super) model_instance_id: ModelInstanceId,
    pub(super) adapter_instance_id: AdapterInstanceId,
    pub(super) metadata: AdapterMetadata,
    pub(super) backend_kind: BackendKind,
    pub(super) max_batch_size: usize,
    pub(super) ceiling: OnceLock<crate::models::architectures::lfm25_audio::model::Lfm25AudioTtsStageCeiling>,
}

impl Lfm25AudioTtsExecutionAdapter {
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
