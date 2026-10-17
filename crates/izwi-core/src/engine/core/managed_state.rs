//! Managed KV glue: loading and unloading a model's managed cache, retained
//! and invocation state, and the token reach of its contract.

use super::*;

pub(super) fn resolve_managed_token_reach(
    backend: BackendKind,
    configured_context: usize,
    loaded_context: Option<usize>,
) -> Result<Option<u64>> {
    let Some(loaded_context) = loaded_context else {
        if backend == BackendKind::Cuda {
            return Err(Error::ModelLoadError(
                "CUDA managed state requires a positive loaded-model context limit".into(),
            ));
        }
        return Ok(None);
    };
    let effective = loaded_context.min(configured_context);
    if effective == 0 {
        let backend = if backend == BackendKind::Cuda {
            "CUDA"
        } else {
            "portable"
        };
        return Err(Error::ModelLoadError(format!(
            "{backend} managed state resolved a zero loaded-model context limit"
        )));
    }
    u64::try_from(effective).map(Some).map_err(|_| {
        let backend = if backend == BackendKind::Cuda {
            "CUDA"
        } else {
            "portable"
        };
        Error::ModelLoadError(format!("{backend} model context exceeds u64"))
    })
}

/// SCH-F7: probe the managed prefix index for a resumable prefill the
/// scheduler selected before it made progress, and keep the request's own
/// cursor (which prepare and the executor read) equal to the scheduler's.
/// A probe failure plans a zero-start prefill, like the admission probe.
pub(super) fn reprobe_managed_prefix(
    managed_kv_cache: &mut ManagedKvCacheManager,
    requests: &mut HashMap<RequestId, Arc<EngineCoreRequest>>,
    session: &super::super::SessionKey,
) -> Option<u32> {
    let request = requests.get_mut(&session.request_id)?;
    let runtime = request.managed_cache_runtime()?.clone();
    let cursor = managed_kv_cache
        .probe_managed_prefix_for_session(&runtime, request, session)
        .unwrap_or_else(|error| {
            warn!(
                request_id = %session.request_id,
                error = %error,
                "managed prefix probe failed; planning a zero-start prefill"
            );
            None
        });
    if request.managed_prefix_cursor() != cursor {
        Arc::make_mut(request).set_managed_prefix_cursor(cursor);
    }
    cursor
}

impl EngineCore {
    /// Allocate and admit physical KV backing while the authoritative model
    /// generation is still Loading. Ready publication happens only after this
    /// succeeds, and request admission performs lookup only.
    #[cfg(test)]
    pub(crate) fn load_managed_model_cache(
        &mut self,
        model_instance: super::super::ModelInstanceId,
        capability: &crate::kv::InferenceStateCapability,
        logical_context_tokens: Option<usize>,
    ) -> Result<Option<Arc<super::super::ManagedKvModelRuntime>>> {
        self.load_managed_model_cache_with_capacity_policy(
            model_instance,
            capability,
            logical_context_tokens,
            None,
            CudaResidentPool::Disabled,
            0,
        )
    }

    pub(super) fn cuda_pool_upper_bound(context: u64, retained_rows: u32, incremental: bool) -> Result<u64> {
        if incremental {
            context
                .checked_mul(u64::from(retained_rows))
                .ok_or_else(|| {
                    Error::ModelLoadError("CUDA aggregate token capacity overflow".into())
                })
        } else {
            Ok(context)
        }
    }

    /// Load managed state while allowing a model lifecycle to distinguish the
    /// scheduler's retained-session capacity from its simultaneously staged
    /// transaction width. `None` preserves the engine-wide legacy behavior.
    /// CUDA context fitting leaves the largest active decode batch's workspace
    /// free in addition to safety headroom. Dispatch still owns its lease.
    pub(crate) fn load_managed_model_cache_with_capacity_policy(
        &mut self,
        model_instance: super::super::ModelInstanceId,
        capability: &crate::kv::InferenceStateCapability,
        logical_context_tokens: Option<usize>,
        staged_transaction_rows: Option<u32>,
        cuda_resident_pool: CudaResidentPool,
        decode_workspace_reserve_bytes: u64,
    ) -> Result<Option<Arc<super::super::ManagedKvModelRuntime>>> {
        let Some(contract) = capability.managed_contract() else {
            return Ok(None);
        };
        let backend = self.managed_kv_cache.worker_backend();
        let retained_sequence_rows = u32::try_from(self.config.max_retained_sequences)
            .map_err(|_| Error::InvalidInput("managed state sequence limit exceeds u32".into()))?;
        let fit_cuda_resident_context =
            backend == BackendKind::Cuda && cuda_resident_pool != CudaResidentPool::Disabled;
        let staged_transaction_rows = staged_transaction_rows.unwrap_or(
            u32::try_from(self.config.max_staged_transactions).map_err(|_| {
                Error::InvalidInput("managed state transaction limit exceeds u32".into())
            })?,
        );
        let (logical_token_reach, retained_sequence_rows, staged_transaction_rows) =
            if fit_cuda_resident_context {
                let context = logical_context_tokens.ok_or_else(|| {
                    Error::ModelLoadError(
                        "CUDA managed state requires a positive loaded-model context limit".into(),
                    )
                })?;
                let context = u64::try_from(context)
                    .map_err(|_| Error::ModelLoadError("model context exceeds u64".into()))?;
                let incremental = self.config.cuda_incremental_chat_enabled()
                    && cuda_resident_pool == CudaResidentPool::RetainedRows;
                let automatic_rows = self.config.automatic_row_limits.retained_sequences;
                let context_reserve_bytes = self
                    .config
                    .cuda_context_reserve_bytes(self.managed_kv_cache.device_capacity());
                let fit = self
                    .managed_kv_cache
                    .fit_cuda_resident_logical_token_reach(
                        model_instance,
                        contract,
                        |rows| Self::cuda_pool_upper_bound(context, rows, incremental),
                        context_reserve_bytes
                            .checked_add(decode_workspace_reserve_bytes)
                            .ok_or_else(|| {
                                Error::ModelLoadError(
                                    "CUDA context workspace reserve overflow".into(),
                                )
                            })?,
                        self.config.block_size,
                        crate::engine::cache::managed::CudaResidentRowRequest {
                            retained_sequence_rows,
                            staged_transaction_rows,
                            automatic_state_budget_fraction: automatic_rows
                                .then_some(self.config.cuda_state_budget_fraction),
                        },
                    )?;
                info!(
                    ?model_instance,
                    automatic_rows,
                    retained_sequence_rows = fit.retained_sequence_rows,
                    staged_transaction_rows = fit.staged_transaction_rows,
                    prefix_snapshot_rows = fit.prefix_snapshot_rows,
                    per_sequence_state_bytes = fit.per_sequence_state_bytes,
                    resident_state_bytes = fit.resident_state_bytes,
                    logical_token_reach = fit.logical_token_reach,
                    context_reserve_bytes,
                    decode_workspace_reserve_bytes,
                    "Fitted CUDA resident state rows and context"
                );
                (
                    Some(fit.logical_token_reach),
                    fit.retained_sequence_rows,
                    fit.staged_transaction_rows,
                )
            } else {
                // An automatically fitted portable chat context is capped: the
                // arena is fully resident from load, so an idle model would
                // otherwise hold its whole advertised context in memory.
                let cap = self.config.portable_auto_context_cap_tokens;
                let logical_context_tokens =
                    if self.config.portable_context_auto && backend != BackendKind::Cuda && cap > 0
                    {
                        logical_context_tokens.map(|tokens| tokens.min(cap))
                    } else {
                        logical_context_tokens
                    };
                (
                    self.resolve_managed_token_reach_for_contract(
                        backend,
                        model_instance,
                        contract,
                        logical_context_tokens,
                        1,
                        PortableFitHeadroom::Planning {
                            minimum_tokens: CHAT_MINIMUM_CONTEXT_TOKENS,
                        },
                    )?,
                    retained_sequence_rows,
                    staged_transaction_rows,
                )
            };
        let runtime = self
            .managed_kv_cache
            .bind_request_with_capacity_policy(
                model_instance,
                backend,
                ManagedStateCapacityRequest {
                    total_paged_pages: u32::try_from(self.config.max_blocks).map_err(|_| {
                        Error::InvalidInput("managed KV page budget exceeds u32".into())
                    })?,
                    logical_token_reach,
                    retained_sequence_rows,
                    staged_transaction_rows,
                },
                self.config.block_size,
                capability,
                fit_cuda_resident_context,
            )?
            .ok_or_else(|| {
                Error::InferenceError(
                    "managed KV load did not install a physical runtime".to_string(),
                )
            })?;
        if let Some(context) = logical_context_tokens {
            runtime
                .set_maximum_sequence_tokens((context as u64).min(runtime.logical_token_reach()));
        }
        runtime.synchronize_backing()?;
        Ok(Some(runtime))
    }

    /// Retained ASR or TTS state on the engine's own row limits, fitted
    /// within live memory above `minimum_tokens` per sequence (see
    /// [`Self::load_managed_model_state_with_portable_copies_and_rows`]).
    pub(crate) fn load_managed_model_state(
        &mut self,
        model_instance: super::super::ModelInstanceId,
        retained_state: &crate::kv::v2::InferenceStateContract,
        logical_context_tokens: Option<usize>,
        minimum_tokens: u64,
        reserve: StateFitReserve,
    ) -> Result<Arc<super::super::ManagedKvModelRuntime>> {
        let rows = (
            self.config.max_retained_sequences,
            self.config.max_staged_transactions,
        );
        self.load_managed_model_state_with_portable_copies_and_rows(
            model_instance,
            retained_state,
            logical_context_tokens,
            1,
            rows,
            minimum_tokens,
            reserve,
        )
    }

    /// Record a loaded model's tensor rows as reserved by the requests that
    /// create them (see `ManagedKvCacheManager::price_tensor_rows_with_requests`).
    pub(crate) fn price_tensor_rows_with_requests(
        &mut self,
        model_instance: super::super::ModelInstanceId,
    ) -> Result<()> {
        self.managed_kv_cache
            .price_tensor_rows_with_requests(model_instance)
    }

    /// Retained ASR or TTS state with its own `(retained, staged)` row
    /// counts, at most the engine's (ADM-6): the engine rows are sized for
    /// chat (on CUDA one per 256 MiB of device memory), far wider than ASR
    /// concurrency. `minimum_tokens` is the floor every sequence must reach
    /// (one ASR chunk, one TTS prompt plus its output budget); a floor that
    /// planning headroom cannot hold refuses the load with `Overloaded`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_managed_model_state_with_portable_copies_and_rows(
        &mut self,
        model_instance: super::super::ModelInstanceId,
        retained_state: &crate::kv::v2::InferenceStateContract,
        logical_context_tokens: Option<usize>,
        portable_state_copies: u32,
        (retained_rows, staged_rows): (usize, usize),
        minimum_tokens: u64,
        reserve: StateFitReserve,
    ) -> Result<Arc<super::super::ManagedKvModelRuntime>> {
        if retained_rows == 0
            || staged_rows == 0
            || retained_rows > self.config.max_retained_sequences
            || staged_rows > self.config.max_staged_transactions
        {
            return Err(Error::ModelLoadError(
                "retained state row limits exceed configured capacity".into(),
            ));
        }
        let backend = self.managed_kv_cache.worker_backend();
        // ASR and TTS retained state fit within live memory so the requests
        // they serve can run beside them; the floor is admitted against
        // planning headroom.
        let logical_token_reach = self.resolve_managed_token_reach_for_contract_with_rows(
            backend,
            model_instance,
            retained_state,
            logical_context_tokens,
            portable_state_copies,
            (retained_rows, staged_rows),
            PortableFitHeadroom::Live {
                minimum_tokens,
                reserve,
            },
        )?;
        let runtime = self.managed_kv_cache.bind_model_state_with_capacity(
            model_instance,
            backend,
            ManagedStateCapacityRequest {
                total_paged_pages: u32::try_from(self.config.max_blocks).map_err(|_| {
                    Error::InvalidInput("managed KV page budget exceeds u32".into())
                })?,
                logical_token_reach,
                retained_sequence_rows: u32::try_from(retained_rows).map_err(|_| {
                    Error::InvalidInput("managed state sequence limit exceeds u32".into())
                })?,
                staged_transaction_rows: u32::try_from(staged_rows).map_err(|_| {
                    Error::InvalidInput("managed state transaction limit exceeds u32".into())
                })?,
            },
            self.config.block_size,
            retained_state,
        )?;
        runtime.synchronize_backing()?;
        Ok(runtime)
    }

    /// Resolve one loaded model against its sealed row ceilings without changing
    /// engine-wide configuration or another model's fitting policy.
    pub(crate) fn load_managed_model_state_with_row_limits(
        &mut self,
        model_instance: super::super::ModelInstanceId,
        retained_state: &crate::kv::v2::InferenceStateContract,
        logical_context_tokens: Option<usize>,
        retained_rows: usize,
        staged_rows: usize,
    ) -> Result<Arc<super::super::ManagedKvModelRuntime>> {
        // An automatic staged limit is sized for tensor-width launches, not an
        // operator ceiling; a model that fits its own launch width (Fish) is
        // bounded by its retained rows instead.
        let staged_ceiling = if self.config.automatic_row_limits.staged_transactions {
            self.config.max_retained_sequences
        } else {
            self.config.max_staged_transactions
        };
        if retained_rows == 0
            || staged_rows == 0
            || retained_rows > self.config.max_retained_sequences
            || staged_rows > staged_ceiling
        {
            return Err(Error::ModelLoadError(
                "model row limits exceed configured capacity".into(),
            ));
        }
        let backend = self.managed_kv_cache.worker_backend();
        let logical_token_reach = self.resolve_managed_token_reach_for_contract_with_rows(
            backend,
            model_instance,
            retained_state,
            logical_context_tokens,
            1,
            (retained_rows, staged_rows),
            PortableFitHeadroom::Planning { minimum_tokens: 0 },
        )?;
        let runtime = self.managed_kv_cache.bind_model_state_with_capacity(
            model_instance,
            backend,
            ManagedStateCapacityRequest {
                total_paged_pages: u32::try_from(self.config.max_blocks).map_err(|_| {
                    Error::InvalidInput("managed KV page budget exceeds u32".into())
                })?,
                logical_token_reach,
                retained_sequence_rows: u32::try_from(retained_rows).map_err(|_| {
                    Error::InvalidInput("managed state sequence limit exceeds u32".into())
                })?,
                staged_transaction_rows: u32::try_from(staged_rows).map_err(|_| {
                    Error::InvalidInput("managed state transaction limit exceeds u32".into())
                })?,
            },
            self.config.block_size,
            retained_state,
        )?;
        runtime.synchronize_backing()?;
        Ok(runtime)
    }

    pub(crate) fn load_composite_retained_state(
        &mut self,
        model_instance: super::super::ModelInstanceId,
        full_contract: &crate::kv::v2::InferenceStateContract,
        static_domain: crate::kv::v2::StateDomainId,
        logical_context_tokens: Option<usize>,
        minimum_tokens: u64,
    ) -> Result<Arc<CompositeRetainedStateRuntimeV2>> {
        if self.composite_retained_state.contains_key(&model_instance) {
            return Err(Error::InvalidInput(
                "composite retained state already owns this model generation".into(),
            ));
        }
        let backend = self.managed_kv_cache.worker_backend();
        let logical_token_reach = self.resolve_managed_token_reach_for_contract(
            backend,
            model_instance,
            &project_composite_paged_contract(full_contract, static_domain)?,
            logical_context_tokens,
            1,
            // Whisper's decoder state: ASR, fitted within live memory. Its
            // load reconciles observed weight growth.
            PortableFitHeadroom::Live {
                minimum_tokens,
                reserve: StateFitReserve::MeasuredUntracked,
            },
        )?;
        let sequence_capacity = u32::try_from(self.config.max_retained_sequences)
            .map_err(|_| Error::InvalidInput("managed state sequence limit exceeds u32".into()))?;
        let runtime = allocate_composite_retained_state(
            &mut self.managed_kv_cache,
            &mut self.physical_state,
            model_instance,
            backend,
            ManagedStateCapacityRequest {
                total_paged_pages: u32::try_from(self.config.max_blocks).map_err(|_| {
                    Error::InvalidInput("managed KV page budget exceeds u32".into())
                })?,
                logical_token_reach,
                retained_sequence_rows: sequence_capacity,
                staged_transaction_rows: u32::try_from(self.config.max_staged_transactions)
                    .map_err(|_| {
                        Error::InvalidInput("managed state transaction limit exceeds u32".into())
                    })?,
            },
            self.config.block_size,
            full_contract,
            static_domain,
        )?;
        let replaced = self
            .composite_retained_state
            .insert(model_instance, runtime.clone())
            .is_some();
        debug_assert!(!replaced, "fresh composite generation replaced an owner");
        Ok(runtime)
    }

    pub(super) fn resolve_managed_token_reach_for_contract(
        &self,
        backend: BackendKind,
        model_instance: super::super::ModelInstanceId,
        contract: &crate::kv::v2::InferenceStateContract,
        loaded_context: Option<usize>,
        portable_state_copies: u32,
        headroom: PortableFitHeadroom,
    ) -> Result<Option<u64>> {
        self.resolve_managed_token_reach_for_contract_with_rows(
            backend,
            model_instance,
            contract,
            loaded_context,
            portable_state_copies,
            (
                self.config.max_retained_sequences,
                self.config.max_staged_transactions,
            ),
            headroom,
        )
    }

    pub(super) fn resolve_managed_token_reach_for_contract_with_rows(
        &self,
        backend: BackendKind,
        model_instance: super::super::ModelInstanceId,
        contract: &crate::kv::v2::InferenceStateContract,
        loaded_context: Option<usize>,
        portable_state_copies: u32,
        row_limits: (usize, usize),
        headroom: PortableFitHeadroom,
    ) -> Result<Option<u64>> {
        if !self.config.portable_context_auto
            && backend != BackendKind::Cuda
            && loaded_context.is_none()
        {
            return Ok(None);
        }
        let loaded_context = loaded_context.ok_or_else(|| {
            Error::ModelLoadError(format!(
                "{backend:?} managed state requires a positive loaded-model context limit"
            ))
        })?;
        if self.config.portable_context_auto && backend != BackendKind::Cuda {
            let maximum_tokens = u64::try_from(loaded_context)
                .map_err(|_| Error::ModelLoadError("model context exceeds u64".into()))?;
            return self
                .managed_kv_cache
                .fit_portable_logical_token_reach(
                    model_instance,
                    contract,
                    maximum_tokens,
                    self.config.portable_context_reserve_bytes,
                    self.config.block_size,
                    u32::try_from(row_limits.0).map_err(|_| {
                        Error::InvalidInput("managed state sequence limit exceeds u32".into())
                    })?,
                    u32::try_from(row_limits.1).map_err(|_| {
                        Error::InvalidInput("managed state transaction limit exceeds u32".into())
                    })?,
                    portable_state_copies,
                    headroom,
                )
                .map(Some);
        }
        resolve_managed_token_reach(backend, self.config.max_seq_len, Some(loaded_context))
    }

    pub(crate) fn load_retained_tensor_state(
        &mut self,
        model_instance: super::super::ModelInstanceId,
        contract: &crate::kv::v2::InferenceStateContract,
        sequence_capacity: u32,
    ) -> Result<Arc<super::super::RetainedTensorStateRuntimeV2>> {
        self.physical_state
            .allocate_retained_tensor(model_instance, contract, sequence_capacity)
    }

    pub(crate) fn resolve_and_load_invocation_workspace(
        &mut self,
        model_instance: super::super::ModelInstanceId,
        adapter_instance: super::super::AdapterInstanceId,
        stage_graph: [u8; 32],
        stage: super::super::StageId,
        contract: &crate::kv::v2::InferenceStateContract,
        domain: &crate::kv::v2::InvocationWorkspaceDomain,
        slot_count: u32,
    ) -> Result<Arc<dyn crate::kv::v2::InvocationWorkspaceBackingV2>> {
        self.physical_state
            .resolve_and_allocate_invocation_workspace(
                model_instance,
                InvocationPhysicalKey {
                    adapter_instance,
                    stage_graph,
                    stage,
                    domain: domain.id(),
                },
                contract,
                domain,
                slot_count,
            )
    }

    /// Retire physical managed-KV state for one exact loaded-model instance.
    /// Variant-only purge is insufficient because an unload/reload may publish
    /// a different generation of the same variant.
    pub fn unload_managed_model_cache(
        &mut self,
        model_instance: super::super::ModelInstanceId,
    ) -> Result<bool> {
        if let Some(runtime) = self.composite_retained_state.get(&model_instance) {
            if Arc::strong_count(runtime) != 1 {
                return Err(Error::InferenceError(format!(
                    "composite retained model {} still has live publication/request handles",
                    model_instance.get()
                )));
            }
            let physical_prepared = self
                .physical_state
                .prepare_unload_model_with_static_runtime_owners(model_instance, 2);
            let paged_prepared = self
                .managed_kv_cache
                .prepare_unload_model_with_runtime_owners(model_instance, 2);
            let (physical_present, paged_present) =
                match (physical_prepared, paged_prepared) {
                    (Ok(physical), Ok(paged)) => (physical, paged),
                    (physical, paged) => {
                        return Err(Error::InferenceError(format!(
                            "composite unload did not fence every backing: physical={physical:?}; paged={paged:?}"
                        )))
                    }
                };
            drop(
                self.composite_retained_state
                    .remove(&model_instance)
                    .expect("composite runtime was authenticated above"),
            );
            let physical_removed = physical_present
                && self
                    .physical_state
                    .commit_prepared_unload_model(model_instance);
            let paged_removed = paged_present
                && self
                    .managed_kv_cache
                    .commit_prepared_unload_model(model_instance);
            return Ok(physical_removed || paged_removed);
        }
        unload_composite_retained_state(
            &mut self.managed_kv_cache,
            &mut self.physical_state,
            model_instance,
        )
    }
}
