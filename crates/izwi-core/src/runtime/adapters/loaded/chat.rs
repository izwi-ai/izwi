//! The continuous chat execution adapter.

use super::*;

#[derive(Debug)]
pub(super) struct ContinuousChatExecutionAdapter {
    pub(super) execution_group_id: ExecutionGroupId,
    pub(super) model_instance_id: ModelInstanceId,
    pub(super) adapter_instance_id: AdapterInstanceId,
    pub(super) metadata: AdapterMetadata,
    pub(super) backend_kind: BackendKind,
    pub(super) max_batch_size: usize,
    pub(super) workspace_per_row_bytes: OnceLock<u64>,
    /// `(per_token_bytes, max_span_tokens)` of a model that prices its
    /// prefill activations.
    pub(super) prefill_workspace: OnceLock<(u64, u64)>,
    /// The loaded model's preferred decode quantum.
    pub(super) decode_quantum: OnceLock<u64>,
    /// The token budget of a sealed mixed prefill+decode stage.
    pub(super) mixed_step_budget: OnceLock<u64>,
}

impl ContinuousChatExecutionAdapter {
    pub(super) fn new(
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
