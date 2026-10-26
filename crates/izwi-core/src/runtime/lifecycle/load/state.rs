//! Per-capability state allocation of a model load: each method fits and
//! publishes the physical state one capability of the loaded model needs,
//! called in order by `run_load_attempt_locked`.

use super::*;

impl ModelLifecycleController {
    /// Chat: seal the continuous workspace and fit the chat model's retained
    /// state.
    pub(super) async fn allocate_chat_state(
        &self,
        variant: ModelVariant,
        model_instance_id: ModelInstanceId,
        backend: BackendKind,
        bundle_draft: &LoadedModelBundleDraft,
        state_publications: &mut HashMap<CapabilityKind, LoadedStatePublication>,
    ) -> Result<()> {
        if let Some(loaded) = self.model_registry.get_chat(variant).await {
            // Freeze exact MTP/collation geometry before deriving any
            // selectable stage graphs or planning physical state.
            bundle_draft.seal_chat_workspace(
                loaded.continuous_decode_batch_workspace_per_row_bytes()?,
            )?;
            bundle_draft.seal_chat_decode_quantum(
                loaded.speculative_decode_profile().0.max(1) as u64,
            )?;
            if let Some((per_token_bytes, max_span_tokens)) = loaded.prefill_workspace()? {
                bundle_draft.seal_chat_prefill_workspace(per_token_bytes, max_span_tokens)?;
            }
            if loaded.supports_mixed_step()
                && crate::catalog::mixed_step_enabled(variant.family(), backend)
            {
                let token_budget = match self.config.max_tokens_per_step {
                    0 => crate::engine::backend_step_sizes(backend).max_tokens_per_step,
                    tokens => tokens,
                };
                bundle_draft.seal_chat_mixed_step(token_budget as u64)?;
            }
            let decode_workspace_reserve_bytes = bundle_draft
                .execution_contracts(CapabilityKind::Chat)?
                .iter()
                .flat_map(|contract| contract.stages.iter())
                .map(|stage| stage.max_workspace_bytes)
                .max()
                .unwrap_or(0);
            let loaded_cache = loaded.inference_state_contract()?;
            loaded_cache.validate()?;
            let publication = match &loaded_cache {
                crate::kv::InferenceStateCapability::Managed(contract) => {
                    if !managed_kv_backend_compiled(backend) {
                        return Err(Error::ModelLoadError(format!(
                            "loaded model {variant} publishes managed KV, but the {backend:?} build has no direct paged-attention runtime"
                        )));
                    }
                    let mut capacity_policy = managed_chat_capacity_policy(variant, backend);
                    if capacity_policy.staged_transaction_rows.is_none() {
                        capacity_policy.staged_transaction_rows =
                            self.portable_chat_staged_rows(bundle_draft, backend)?;
                    }
                    let physical = self
                        .core_engine
                        .load_managed_model_cache_with_capacity_policy(
                            model_instance_id,
                            &loaded_cache,
                            Some(loaded.max_context_tokens()?),
                            capacity_policy.staged_transaction_rows,
                            capacity_policy.cuda_resident_pool,
                            decode_workspace_reserve_bytes,
                        )
                        .await?;
                    let physical = physical.ok_or_else(|| {
                        Error::ModelLoadError(
                            "managed state allocation returned no physical runtime".to_string(),
                        )
                    })?;
                    self.model_registry.publish_effective_context(
                        variant,
                        physical.maximum_sequence_tokens(),
                    )?;
                    self.model_registry.publish_kv_storage(variant, &physical);
                    crate::runtime::rollout::validate_managed_state_plan_eligibility(
                        variant,
                        CapabilityKind::Chat,
                        physical.state_plan_v2(),
                    )?;
                    Some(LoadedStatePublication::ManagedV2 {
                        contract: contract.clone(),
                        physical,
                    })
                }
                crate::kv::InferenceStateCapability::Stateless => None,
            };
            if let Some(publication) = publication {
                state_publications.insert(CapabilityKind::Chat, publication);
            }
        }
        Ok(())
    }

    /// Granite Speech: invocation state for ASR and speaker-attributed ASR.
    pub(super) async fn allocate_granite_speech_state(
        &self,
        variant: ModelVariant,
        model_instance_id: ModelInstanceId,
        backend: BackendKind,
        bundle_draft: &LoadedModelBundleDraft,
        state_publications: &mut HashMap<CapabilityKind, LoadedStatePublication>,
    ) -> Result<()> {
        if variant.family() == crate::catalog::ModelFamily::GraniteSpeechAsr {
            if !managed_kv_backend_compiled(backend) {
                return Err(Error::ModelLoadError(format!(
                    "loaded model {variant} requires physical Granite Speech invocation state, but the {backend:?} build has no direct paged-attention runtime"
                )));
            }
            let model = self
                .model_registry
                .get_loading_asr(variant)
                .await
                .ok_or_else(|| {
                    Error::ModelLoadError(format!(
                        "loaded Granite Speech model {variant} is missing from the registry"
                    ))
                })?;
            state_publications.extend(
                self.load_granite_speech_state_publications(
                    variant,
                    model_instance_id,
                    bundle_draft,
                    |graphs| model.granite_speech_physical_state_spec(graphs),
                )
                .await?,
            );
        }
        Ok(())
    }

    /// Publish the state each Granite Speech capability's stage graphs need,
    /// as `physical_state_spec` authors it for them.
    pub(super) async fn load_granite_speech_state_publications(
        &self,
        variant: ModelVariant,
        model_instance_id: ModelInstanceId,
        bundle_draft: &LoadedModelBundleDraft,
        physical_state_spec: impl Fn(
            &[&[StageDescriptor]],
        ) -> Result<
            crate::models::architectures::granite_speech::asr::GraniteSpeechPhysicalStateSpec,
        >,
    ) -> Result<HashMap<CapabilityKind, LoadedStatePublication>> {
        let mut publications = HashMap::new();
        for capability in [CapabilityKind::Asr, CapabilityKind::SpeakerAttributedAsr] {
            if self.adapter_registry.require(capability, variant).is_err() {
                continue;
            }
            let contracts = bundle_draft.execution_contracts(capability)?;
            let stage_graphs = contracts
                .iter()
                .map(|contract| contract.stages.as_ref())
                .collect::<Vec<_>>();
            let physical_spec = physical_state_spec(&stage_graphs)?;
            // Without resumable prefill the ASR capability offers no
            // retained graph, so the model authors no retained state and
            // none is allocated (GRN-8).
            let retained_route =
                capability == CapabilityKind::Asr && physical_spec.retained.is_some();
            let (retained, retained_uses) = if retained_route {
                let retained_contract = physical_spec.retained.as_ref().ok_or_else(|| {
                    Error::ModelLoadError(
                        "Granite Speech ASR graph did not author retained decoder state".into(),
                    )
                })?;
                let retained_max_tokens = physical_spec.retained_max_tokens.ok_or_else(|| {
                    Error::ModelLoadError(
                        "Granite Speech retained decoder has no context bound".into(),
                    )
                })?;
                let retained = self
                    .core_engine
                    .load_managed_model_state_with_portable_copies(
                        model_instance_id,
                        retained_contract,
                        Some(retained_max_tokens),
                        2,
                        self.asr_state_rows(),
                        asr_minimum_tokens(variant),
                    )
                    .await?;
                self.model_registry
                    .publish_effective_context(variant, retained.logical_token_reach())?;
                self.model_registry.publish_kv_storage(variant, &retained);
                crate::runtime::rollout::validate_managed_state_plan_eligibility(
                    variant,
                    CapabilityKind::Asr,
                    retained.state_plan_v2(),
                )?;
                let uses = contracts
                    .iter()
                    .map(|contract| {
                        let graph = stage_graph_fingerprint(&contract.stages)?;
                        let use_kind = match contract.execution_profile.mode {
                            crate::engine::ExecutionMode::Sequence => {
                                RetainedStateUseV2::ExternalPaged
                            }
                            crate::engine::ExecutionMode::Atomic => {
                                RetainedStateUseV2::Inactive
                            }
                            _ => {
                                return Err(Error::ModelLoadError(
                                    "Granite Speech ASR graph has an incompatible retained-state profile"
                                        .into(),
                                ));
                            }
                        };
                        Ok((graph, use_kind))
                    })
                    .collect::<Result<HashMap<_, _>>>()?;
                (Some(retained.into()), uses)
            } else {
                if physical_spec.retained.is_some() || physical_spec.retained_max_tokens.is_some() {
                    return Err(Error::ModelLoadError(
                        "speaker-attributed Granite Speech graph unexpectedly requested retained state"
                            .into(),
                    ));
                }
                (None, HashMap::new())
            };
            let publication = self
                .load_invocation_workspace_publication_with_remaining_groups(
                    model_instance_id,
                    &contracts,
                    physical_spec.descriptor,
                    &physical_spec.invocation,
                    retained,
                    retained_uses,
                    LaterStateGroups::sharing(if capability == CapabilityKind::Asr {
                        2
                    } else {
                        1
                    }),
                )
                .await?;
            publications.insert(capability, publication);
        }
        Ok(publications)
    }

    /// Offline ASR (every family but Granite Speech): retained and invocation
    /// state per loaded ASR capability.
    pub(super) async fn allocate_asr_state(
        &self,
        variant: ModelVariant,
        model_instance_id: ModelInstanceId,
        backend: BackendKind,
        bundle_draft: &LoadedModelBundleDraft,
        state_publications: &mut HashMap<CapabilityKind, LoadedStatePublication>,
    ) -> Result<()> {
        if self
            .adapter_registry
            .require(CapabilityKind::Asr, variant)
            .is_ok()
            && variant.family() != crate::catalog::ModelFamily::GraniteSpeechAsr
        {
            if variant.family() == crate::catalog::ModelFamily::Voxtral {
                if !managed_kv_backend_compiled(backend) {
                    return Err(Error::ModelLoadError(format!(
                        "loaded model {variant} requires physical ASR invocation state, but the {backend:?} build has no direct paged-attention runtime"
                    )));
                }
                let loaded = self
                    .model_registry
                    .get_loading_voxtral(variant)
                    .await
                    .ok_or_else(|| {
                    Error::ModelLoadError(format!(
                        "loaded Voxtral model {variant} is missing from the registry"
                    ))
                    })?;
                let contracts = bundle_draft.execution_contracts(CapabilityKind::Asr)?;
                let stage_graphs = contracts
                    .iter()
                    .map(|contract| contract.stages.as_ref())
                    .collect::<Vec<_>>();
                let physical_spec = loaded.physical_state_spec(&stage_graphs)?;
                let publication = self
                    .load_invocation_workspace_publication(
                        model_instance_id,
                        &contracts,
                        physical_spec.descriptor,
                        &physical_spec.invocation,
                        None,
                        HashMap::new(),
                    )
                    .await?;
                state_publications.insert(CapabilityKind::Asr, publication);
            } else if let Some(loaded) =
                self.model_registry.get_loading_asr(variant).await
            {
                let loaded_cache = loaded.inference_state_contract()?;
                loaded_cache.validate()?;
                let publication_route = loaded_asr_state_publication_route(variant);
                if publication_route == LoadedAsrStatePublicationRoute::Qwen3 {
                    if !managed_kv_backend_compiled(backend) {
                        return Err(Error::ModelLoadError(format!(
                            "loaded model {variant} requires physical ASR state, but the {backend:?} build has no direct paged-attention runtime"
                        )));
                    }
                    let contracts = bundle_draft.execution_contracts(CapabilityKind::Asr)?;
                    let stage_graphs = contracts
                        .iter()
                        .map(|contract| contract.stages.as_ref())
                        .collect::<Vec<_>>();
                    // A configured context wins; otherwise Qwen3-ASR bounds
                    // its state by one planned chunk per sequence.
                    let context_bound =
                        if self.config.max_sequence_length.explicit_tokens().is_some() {
                            Qwen3AsrContextBound::Configured
                        } else {
                            Qwen3AsrContextBound::Automatic {
                                retained_rows: self.asr_state_rows().0,
                            }
                        };
                    let physical_spec =
                        loaded.qwen3_physical_state_spec(&stage_graphs, context_bound)?;
                    let retained_max_tokens = physical_spec.retained_max_tokens;
                    let physical = self
                        .core_engine
                        .load_managed_model_state_with_portable_copies(
                            model_instance_id,
                            &physical_spec.retained,
                            Some(retained_max_tokens),
                            2,
                            self.asr_state_rows(),
                            asr_minimum_tokens(variant),
                        )
                        .await?;
                    // The retained pool serves several sequences, but one
                    // sequence or invocation never exceeds its own bound.
                    self.model_registry.publish_effective_context(
                        variant,
                        physical
                            .logical_token_reach()
                            .min(physical_spec.sequence_tokens as u64),
                    )?;
                    self.model_registry.publish_kv_storage(variant, &physical);
                    crate::runtime::rollout::validate_managed_state_plan_eligibility(
                        variant,
                        CapabilityKind::Asr,
                        physical.state_plan_v2(),
                    )?;
                    let retained_uses = contracts
                        .iter()
                        .map(|contract| {
                            let graph = stage_graph_fingerprint(&contract.stages)?;
                            let retained_use = match contract.execution_profile.cache_mode {
                                CacheMode::ExternalPaged
                                    if contract.execution_profile.cache_namespace.is_some()
                                        && contract.execution_profile.kv_dtype != "none" =>
                                {
                                    RetainedStateUseV2::ExternalPaged
                                }
                                CacheMode::None
                                    if contract.execution_profile.cache_namespace.is_none()
                                        && contract.execution_profile.kv_dtype == "none" =>
                                {
                                    RetainedStateUseV2::Inactive
                                }
                                _ => {
                                    return Err(Error::ModelLoadError(
                                        "Qwen3 ASR graph has an incompatible retained-state profile"
                                            .to_string(),
                                    ));
                                }
                            };
                            Ok((graph, retained_use))
                        })
                        .collect::<Result<HashMap<_, _>>>()?;
                    let publication = self
                        .load_invocation_workspace_publication(
                            model_instance_id,
                            &contracts,
                            physical_spec.descriptor,
                            &physical_spec.invocation,
                            Some(physical.into()),
                            retained_uses,
                        )
                        .await?;
                    state_publications.insert(
                        CapabilityKind::Asr,
                        publication,
                    );
                } else if publication_route == LoadedAsrStatePublicationRoute::VibeVoice {
                    if !managed_kv_backend_compiled(backend) {
                        return Err(Error::ModelLoadError(format!(
                            "loaded model {variant} requires physical ASR invocation state, but the {backend:?} build has no direct paged-attention runtime"
                        )));
                    }
                    let contracts = bundle_draft.execution_contracts(CapabilityKind::Asr)?;
                    let stage_graphs = contracts
                        .iter()
                        .map(|contract| contract.stages.as_ref())
                        .collect::<Vec<_>>();
                    let physical_spec = loaded.vibevoice_physical_state_spec(&stage_graphs)?;
                    let retained_contract = physical_spec.retained.as_ref().ok_or_else(|| {
                        Error::ModelLoadError(
                            "VibeVoice ASR normal graph did not author retained decoder state"
                                .into(),
                        )
                    })?;
                    let native_tokens = physical_spec.retained_max_tokens.ok_or_else(|| {
                        Error::ModelLoadError(
                            "VibeVoice ASR retained decoder state has no context bound".into(),
                        )
                    })?;
                    // The retained route takes one tokenizer chunk, so
                    // without a configured context each retained row is
                    // bounded by one chunk's prompt and duration budget
                    // (P2.5), not the decoder's native context.
                    let retained_max_tokens =
                        if self.config.max_sequence_length.explicit_tokens().is_some() {
                            native_tokens
                        } else {
                            crate::runtime::asr_automatic_context(
                                native_tokens,
                                loaded.vibevoice_retained_sequence_ceiling()?,
                                self.asr_state_rows().0,
                            )
                            .1
                        };
                    let retained = self
                        .core_engine
                        .load_managed_model_state_with_portable_copies(
                            model_instance_id,
                            retained_contract,
                            Some(retained_max_tokens),
                            2,
                            self.asr_state_rows(),
                            asr_minimum_tokens(variant),
                        )
                        .await?;
                    // The legacy invocation route keeps the duration-budget
                    // ceiling (32,768 decode tokens); its own fit bounds it
                    // by memory.
                    let invocation_ceiling = native_tokens.min(
                        crate::models::architectures::vibevoice::asr::VIBEVOICE_ASR_AUTO_MAX_TOKENS,
                    ) as u64;
                    self.model_registry.publish_effective_context(
                        variant,
                        retained.logical_token_reach().max(invocation_ceiling),
                    )?;
                    self.model_registry.publish_kv_storage(variant, &retained);
                    crate::runtime::rollout::validate_managed_state_plan_eligibility(
                        variant,
                        CapabilityKind::Asr,
                        retained.state_plan_v2(),
                    )?;
                    let retained_uses = contracts
                        .iter()
                        .map(|contract| {
                            let graph = stage_graph_fingerprint(&contract.stages)?;
                            let retained_use = match contract.execution_profile.mode {
                                crate::engine::ExecutionMode::Sequence => {
                                    RetainedStateUseV2::ExternalPaged
                                }
                                crate::engine::ExecutionMode::Atomic => {
                                    RetainedStateUseV2::Inactive
                                }
                                _ => {
                                    return Err(Error::ModelLoadError(
                                        "VibeVoice ASR graph has an incompatible retained-state profile"
                                            .into(),
                                    ));
                                }
                            };
                            Ok((graph, retained_use))
                        })
                        .collect::<Result<HashMap<_, _>>>()?;
                    let publication = self
                        .load_invocation_workspace_publication(
                            model_instance_id,
                            &contracts,
                            physical_spec.descriptor,
                            &physical_spec.invocation,
                            Some(retained.into()),
                            retained_uses,
                        )
                        .await?;
                    state_publications.insert(CapabilityKind::Asr, publication);
                } else if publication_route == LoadedAsrStatePublicationRoute::Whisper {
                    if !managed_kv_backend_compiled(backend) {
                        return Err(Error::ModelLoadError(format!(
                            "loaded model {variant} requires physical Whisper ASR invocation state, but the {backend:?} build has no direct paged-attention runtime"
                        )));
                    }
                    let contracts = bundle_draft.execution_contracts(CapabilityKind::Asr)?;
                    let stage_graphs = contracts
                        .iter()
                        .map(|contract| contract.stages.as_ref())
                        .collect::<Vec<_>>();
                    let physical_spec = loaded.whisper_physical_state_spec(&stage_graphs)?;
                    let retained = self
                        .core_engine
                        .load_composite_retained_state(
                            model_instance_id,
                            &physical_spec.retained,
                            physical_spec.retained_static_domain,
                            Some(physical_spec.retained_max_tokens),
                            asr_minimum_tokens(variant),
                        )
                        .await?;
                    let retained_uses = contracts
                        .iter()
                        .map(|contract| {
                            let graph = stage_graph_fingerprint(&contract.stages)?;
                            let retained_use = if contract.execution_profile.mode
                                == crate::engine::ExecutionMode::Sequence
                            {
                                RetainedStateUseV2::ExternalPagedStatic
                            } else if contract.execution_profile.mode
                                == crate::engine::ExecutionMode::Atomic
                            {
                                RetainedStateUseV2::Inactive
                            } else {
                                return Err(Error::ModelLoadError(
                                    "Whisper graph has an incompatible retained-state profile"
                                        .into(),
                                ));
                            };
                            Ok((graph, retained_use))
                        })
                        .collect::<Result<HashMap<_, _>>>()?;
                    let publication = self
                        .load_invocation_workspace_publication(
                            model_instance_id,
                            &contracts,
                            physical_spec.descriptor,
                            &physical_spec.invocation,
                            Some(retained.into()),
                            retained_uses,
                        )
                        .await?;
                    state_publications.insert(CapabilityKind::Asr, publication);
                } else if publication_route == LoadedAsrStatePublicationRoute::Parakeet {
                    let contracts = bundle_draft.execution_contracts(CapabilityKind::Asr)?;
                    let stage_graphs = contracts
                        .iter()
                        .map(|contract| contract.stages.as_ref())
                        .collect::<Vec<_>>();
                    let physical_spec = loaded.parakeet_physical_state_spec(&stage_graphs)?;
                    let retained_contract = physical_spec.retained.as_ref().ok_or_else(|| {
                        Error::ModelLoadError(
                            "Parakeet sequence graph did not author retained recurrent state"
                                .into(),
                        )
                    })?;
                    let retained = self
                        .core_engine
                        .load_retained_tensor_state(
                            model_instance_id,
                            retained_contract,
                            self.realtime_asr_sequence_capacity,
                        )
                        .await?;
                    let effective_context = self.config.portable_context_ceiling();
                    self.model_registry.publish_effective_context(
                        variant,
                        u64::try_from(effective_context).map_err(|_| {
                            Error::ModelLoadError(
                                "Parakeet effective context exceeds u64".into(),
                            )
                        })?,
                    )?;
                    let invocation_contract =
                        physical_spec.invocation.as_ref().ok_or_else(|| {
                            Error::ModelLoadError(
                                "Parakeet long-form graph did not author invocation predictor state"
                                    .into(),
                            )
                        })?;
                    let publication = self
                        .load_invocation_workspace_publication(
                            model_instance_id,
                            &contracts,
                            physical_spec.descriptor,
                            invocation_contract,
                            Some(retained.into()),
                            parakeet_retained_uses(&contracts)?,
                        )
                        .await?;
                    state_publications.insert(CapabilityKind::Asr, publication);
                } else if publication_route
                    == LoadedAsrStatePublicationRoute::NemotronOffline
                {
                    let contracts = bundle_draft.execution_contracts(CapabilityKind::Asr)?;
                    let stage_graphs = contracts
                        .iter()
                        .map(|contract| contract.stages.as_ref())
                        .collect::<Vec<_>>();
                    let physical_spec =
                        loaded.nemotron_offline_physical_state_spec(&stage_graphs)?;
                    let publication = self
                        .load_invocation_workspace_publication(
                            model_instance_id,
                            &contracts,
                            physical_spec.descriptor,
                            &physical_spec.invocation,
                            None,
                            HashMap::new(),
                        )
                        .await?;
                    state_publications.insert(CapabilityKind::Asr, publication);
                } else if publication_route == LoadedAsrStatePublicationRoute::GraniteSpeech5 {
                    // One atomic invocation per request on leased decoder
                    // pages of the model's full context.
                    if !managed_kv_backend_compiled(backend) {
                        return Err(Error::ModelLoadError(format!(
                            "loaded model {variant} requires physical ASR invocation state, but the {backend:?} build has no direct paged-attention runtime"
                        )));
                    }
                    let contracts = bundle_draft.execution_contracts(CapabilityKind::Asr)?;
                    let stage_graphs = contracts
                        .iter()
                        .map(|contract| contract.stages.as_ref())
                        .collect::<Vec<_>>();
                    let physical_spec =
                        loaded.granite_speech5_physical_state_spec(&stage_graphs)?;
                    let publication = self
                        .load_invocation_workspace_publication(
                            model_instance_id,
                            &contracts,
                            physical_spec.descriptor,
                            &physical_spec.invocation,
                            None,
                            HashMap::new(),
                        )
                        .await?;
                    state_publications.insert(CapabilityKind::Asr, publication);
                } else if let crate::kv::InferenceStateCapability::Managed(contract) =
                    &loaded_cache
                {
                    return Err(Error::ModelLoadError(format!(
                        "loaded non-Qwen ASR model {variant} still publishes legacy managed state with {} domains",
                        contract.domains.len()
                    )));
                }
            }
        }
        Ok(())
    }

    /// Realtime ASR: the retained session state and staging of the realtime
    /// capability.
    pub(super) async fn allocate_realtime_asr_state(
        &self,
        variant: ModelVariant,
        model_instance_id: ModelInstanceId,
        backend: BackendKind,
        state_fit_reserve: StateFitReserve,
        bundle_draft: &LoadedModelBundleDraft,
        state_publications: &mut HashMap<CapabilityKind, LoadedStatePublication>,
    ) -> Result<()> {
        if self
            .adapter_registry
            .require(CapabilityKind::RealtimeAsr, variant)
            .is_ok()
        {
            let contracts =
                bundle_draft.execution_contracts(CapabilityKind::RealtimeAsr)?;
            let stage_graphs = contracts
                .iter()
                .map(|contract| contract.stages.as_ref())
                .collect::<Vec<_>>();
            if variant.family() == crate::catalog::ModelFamily::Voxtral {
                if !managed_kv_backend_compiled(backend) {
                    return Err(Error::ModelLoadError(format!(
                        "loaded model {variant} requires physical realtime ASR state, but the {backend:?} build has no direct paged-attention runtime"
                    )));
                }
                let model = self
                    .model_registry
                    .get_loading_voxtral(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "loaded Voxtral realtime ASR model {variant} is missing from the registry"
                        ))
                    })?;
                let physical_spec = model.realtime_physical_state_spec(&stage_graphs)?;
                // One planned ASR chunk per sequence, as every other
                // ASR retained-state fit guarantees.
                let retained = self
                    .core_engine
                    .load_managed_model_state(
                        model_instance_id,
                        &physical_spec.retained,
                        Some(physical_spec.retained_max_tokens),
                        asr_minimum_tokens(variant),
                        state_fit_reserve,
                        0,
                    )
                    .await?;
                let retained_uses = contracts
                    .iter()
                    .map(|contract| {
                        Ok((
                            stage_graph_fingerprint(&contract.stages)?,
                            RetainedStateUseV2::ExternalPaged,
                        ))
                    })
                    .collect::<Result<HashMap<_, _>>>()?;
                state_publications.insert(
                    CapabilityKind::RealtimeAsr,
                    LoadedStatePublication::PhysicalV2 {
                        descriptor: physical_spec.descriptor,
                        retained: Some(retained.into()),
                        retained_uses,
                        invocation_workspace: InvocationWorkspaceRuntimeV2::default(),
                    },
                );
            } else {
                let model = self
                    .model_registry
                    .get_loading_asr(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "loaded realtime ASR model {variant} is missing from the registry"
                        ))
                    })?;
                let physical_spec = model.realtime_physical_state_spec(&stage_graphs)?;
                let retained = self
                    .core_engine
                    .load_retained_tensor_state(
                        model_instance_id,
                        &physical_spec.retained,
                        self.realtime_asr_sequence_capacity,
                    )
                    .await?;
                let retained_uses = contracts
                    .iter()
                    .map(|contract| {
                        Ok((
                            stage_graph_fingerprint(&contract.stages)?,
                            RetainedStateUseV2::ExternalTensor,
                        ))
                    })
                    .collect::<Result<HashMap<_, _>>>()?;
                state_publications.insert(
                    CapabilityKind::RealtimeAsr,
                    LoadedStatePublication::PhysicalV2 {
                        descriptor: physical_spec.descriptor,
                        retained: Some(retained.into()),
                        retained_uses,
                        invocation_workspace: InvocationWorkspaceRuntimeV2::default(),
                    },
                );
            }
        }
        Ok(())
    }

    /// Diarization: Sortformer's invocation state.
    pub(super) async fn allocate_diarization_state(
        &self,
        variant: ModelVariant,
        model_instance_id: ModelInstanceId,
        bundle_draft: &LoadedModelBundleDraft,
        state_publications: &mut HashMap<CapabilityKind, LoadedStatePublication>,
    ) -> Result<()> {
        if variant.family() == crate::catalog::ModelFamily::SortformerDiarization {
            let model = self
                .model_registry
                .get_diarization(variant)
                .await
                .ok_or_else(|| {
                    Error::ModelLoadError(format!(
                        "loaded Sortformer model {variant} is missing from the registry"
                    ))
                })?;
            let contracts =
                bundle_draft.execution_contracts(CapabilityKind::Diarization)?;
            let stage_graphs = contracts
                .iter()
                .map(|contract| contract.stages.as_ref())
                .collect::<Vec<_>>();
            let physical_spec = model.physical_state_spec(&stage_graphs)?;
            let publication = self
                .load_invocation_workspace_publication(
                    model_instance_id,
                    &contracts,
                    physical_spec.descriptor,
                    &physical_spec.invocation,
                    None,
                    HashMap::new(),
                )
                .await?;
            state_publications.insert(CapabilityKind::Diarization, publication);
        }
        Ok(())
    }

    /// LFM2.5-Audio: ASR and TTS state.
    pub(super) async fn allocate_lfm25_audio_state(
        &self,
        variant: ModelVariant,
        model_instance_id: ModelInstanceId,
        backend: BackendKind,
        state_fit_reserve: StateFitReserve,
        bundle_draft: &LoadedModelBundleDraft,
        state_publications: &mut HashMap<CapabilityKind, LoadedStatePublication>,
    ) -> Result<()> {
        if variant.family() == crate::catalog::ModelFamily::Lfm25Audio {
            if !managed_kv_backend_compiled(backend) {
                return Err(Error::ModelLoadError(format!(
                    "loaded model {variant} requires physical LFM2.5 Audio invocation state, but the {backend:?} build has no direct paged-attention runtime"
                )));
            }
            let model = self
                .model_registry
                .get_loading_lfm25_audio_lease(variant)
                .await
                .ok_or_else(|| {
                    Error::ModelLoadError(format!(
                        "loaded LFM2.5 Audio model {variant} is missing from the registry"
                    ))
                })?;
            for capability in [
                CapabilityKind::Asr,
                CapabilityKind::Tts,
                CapabilityKind::AudioChat,
                CapabilityKind::SpeechToSpeech,
            ] {
                let remaining_automatic_state_groups = match capability {
                    CapabilityKind::Asr => 5,
                    CapabilityKind::Tts => 3,
                    CapabilityKind::AudioChat => 2,
                    CapabilityKind::SpeechToSpeech => 1,
                    _ => unreachable!("LFM2.5 Audio capability list is closed above"),
                };
                let contracts = bundle_draft.execution_contracts(capability)?;
                let stage_graphs = contracts
                    .iter()
                    .map(|contract| contract.stages.as_ref())
                    .collect::<Vec<_>>();
                if capability == CapabilityKind::Asr {
                    let physical_spec = model.retained_asr_state_spec(&stage_graphs)?;
                    let retained_max_tokens = physical_spec.retained_max_tokens;
                    let retained = self
                        .core_engine
                        .load_managed_model_state_with_portable_copies(
                            model_instance_id,
                            &physical_spec.retained,
                            Some(retained_max_tokens),
                            2,
                            self.asr_state_rows(),
                            asr_minimum_tokens(variant),
                        )
                        .await?;
                    self.model_registry.publish_effective_context(
                        variant,
                        retained.logical_token_reach(),
                    )?;
                    self.model_registry.publish_kv_storage(variant, &retained);
                    crate::runtime::rollout::validate_managed_state_plan_eligibility(
                        variant,
                        CapabilityKind::Asr,
                        retained.state_plan_v2(),
                    )?;
                    let retained_uses = contracts
                        .iter()
                        .map(|contract| {
                            let graph = stage_graph_fingerprint(&contract.stages)?;
                            let retained_use = match contract.execution_profile.mode {
                                crate::engine::ExecutionMode::Sequence => {
                                    RetainedStateUseV2::ExternalPaged
                                }
                                crate::engine::ExecutionMode::Atomic => {
                                    RetainedStateUseV2::Inactive
                                }
                                _ => {
                                    return Err(Error::ModelLoadError(
                                        "LFM2.5 Audio ASR graph has an incompatible retained-state profile"
                                            .into(),
                                    ));
                                }
                            };
                            Ok((graph, retained_use))
                        })
                        .collect::<Result<HashMap<_, _>>>()?;
                    let main_invocation = physical_spec.main_invocation.as_ref().ok_or_else(|| {
                        Error::ModelLoadError(
                            "LFM2.5 Audio ASR retained topology is missing long-form main invocation state"
                                .into(),
                        )
                    })?;
                    let publication = self
                        .load_invocation_workspace_publication_with_remaining_groups(
                            model_instance_id,
                            &contracts,
                            physical_spec.descriptor,
                            main_invocation,
                            Some(retained.into()),
                            retained_uses,
                            LaterStateGroups::sharing(remaining_automatic_state_groups),
                        )
                        .await?;
                    state_publications.insert(capability, publication);
                    continue;
                }
                if capability == CapabilityKind::Tts {
                    let physical_spec = model.retained_tts_state_spec(&stage_graphs)?;
                    let retained_max_tokens = self
                        .model_registry
                        .effective_context(variant)
                        .map_or(physical_spec.retained_max_tokens, |tokens| {
                            physical_spec.retained_max_tokens.min(tokens)
                        });
                    // The ASR fit above already bound this model's
                    // retained runtime for the same contract; this call
                    // returns it without fitting again.
                    let retained = self
                        .core_engine
                        .load_managed_model_state(
                            model_instance_id,
                            &physical_spec.retained,
                            Some(retained_max_tokens),
                            0,
                            state_fit_reserve,
                            0,
                        )
                        .await?;
                    self.model_registry.publish_effective_context(
                        variant,
                        retained.logical_token_reach(),
                    )?;
                    self.model_registry.publish_kv_storage(variant, &retained);
                    crate::runtime::rollout::validate_managed_state_plan_eligibility(
                        variant,
                        CapabilityKind::Tts,
                        retained.state_plan_v2(),
                    )?;
                    let retained_uses = contracts
                        .iter()
                        .map(|contract| {
                            Ok((
                                stage_graph_fingerprint(&contract.stages)?,
                                RetainedStateUseV2::ExternalPaged,
                            ))
                        })
                        .collect::<Result<HashMap<_, _>>>()?;
                    let depthformer = physical_spec.depthformer_invocation.as_ref().ok_or_else(
                        || {
                            Error::ModelLoadError(
                                "LFM2.5 Audio TTS topology is missing Depthformer invocation state"
                                    .into(),
                            )
                        },
                    )?;
                    let publication = self
                        .load_invocation_workspace_publication_with_remaining_groups(
                            model_instance_id,
                            &contracts,
                            physical_spec.descriptor,
                            depthformer,
                            Some(retained.into()),
                            retained_uses,
                            LaterStateGroups::sharing(remaining_automatic_state_groups),
                        )
                        .await?;
                    state_publications.insert(capability, publication);
                    continue;
                }
                let mode = if capability == CapabilityKind::Asr {
                    crate::models::architectures::lfm25_audio::physical::Lfm25AudioStateMode::MainOnly
                } else {
                    crate::models::architectures::lfm25_audio::physical::Lfm25AudioStateMode::MainAndDepthformer
                };
                let physical_spec = model.physical_state_spec(mode, &stage_graphs)?;
                let publication = self
                    .load_invocation_workspace_publication_with_remaining_groups(
                        model_instance_id,
                        &contracts,
                        physical_spec.descriptor,
                        &physical_spec.invocation,
                        None,
                        HashMap::new(),
                        LaterStateGroups::sharing(remaining_automatic_state_groups),
                    )
                    .await?;
                state_publications.insert(capability, publication);
            }
        }
        Ok(())
    }

    /// Qwen3-TTS: retained talker state and the code-predictor workspace.
    pub(super) async fn allocate_qwen_tts_state(
        &self,
        variant: ModelVariant,
        model_instance_id: ModelInstanceId,
        backend: BackendKind,
        state_fit_reserve: StateFitReserve,
        bundle_draft: &LoadedModelBundleDraft,
        state_publications: &mut HashMap<CapabilityKind, LoadedStatePublication>,
    ) -> Result<()> {
        if variant.family() == crate::catalog::ModelFamily::Qwen3Tts {
            let model = self
                .model_registry
                .get_qwen_tts(variant)
                .await
                .ok_or_else(|| {
                    Error::ModelLoadError(format!(
                        "loaded Qwen3 TTS model {variant} is missing from the registry"
                    ))
                })?;
            // Plan both capabilities first: the retained fit and each
            // predictor fit must leave room for the predictor workspaces
            // loaded after them.
            let mut planned = Vec::with_capacity(2);
            for capability in [CapabilityKind::Tts, CapabilityKind::StreamingTts] {
                if self.adapter_registry.require(capability, variant).is_err() {
                    continue;
                }
                let contracts = bundle_draft.execution_contracts(capability)?;
                let stage_graphs = contracts
                    .iter()
                    .map(|contract| contract.stages.as_ref())
                    .collect::<Vec<_>>();
                let physical_spec = model.physical_state_spec(&stage_graphs)?;
                // At its context floor, if it has a context axis.
                let floor = physical_spec
                    .descriptor
                    .capacity_axis_bounds(StateCapacityAxis::DecoderContext)
                    .map(|(minimum, _)| minimum);
                let predictor_bytes = invocation_state_bytes(
                    &physical_spec.descriptor,
                    &physical_spec.predictor_contract,
                    &contracts,
                    floor,
                )?;
                planned.push((capability, contracts, physical_spec, predictor_bytes));
            }
            let mut later_predictor_bytes = planned
                .iter()
                .try_fold(0_u64, |total, (.., bytes)| total.checked_add(*bytes))
                .ok_or_else(|| {
                    Error::ModelLoadError("Qwen3-TTS state byte plan overflow".into())
                })?;
            let mut remaining_groups = u64::try_from(planned.len()).unwrap_or(u64::MAX);
            for (capability, contracts, physical_spec, predictor_bytes) in planned {
                if !managed_kv_backend_compiled(backend) {
                    return Err(Error::ModelLoadError(format!(
                        "loaded model {variant} requires physical TTS state, but the {backend:?} build has no direct paged-attention runtime"
                    )));
                }
                let retained_max_tokens = self
                    .model_registry
                    .effective_context(variant)
                    .map_or(physical_spec.retained_max_tokens, |tokens| {
                        physical_spec.retained_max_tokens.min(tokens)
                    });
                // Both capabilities bind one retained arena: the first call
                // fits it, leaving both predictor workspaces free; the second
                // gets the bound runtime back.
                let retained = self
                    .core_engine
                    .load_managed_model_state(
                        model_instance_id,
                        &physical_spec.retained,
                        Some(retained_max_tokens),
                        tts_state_minimum_tokens(variant),
                        state_fit_reserve,
                        later_predictor_bytes,
                    )
                    .await?;
                // This capability's predictor is fitted next; only the ones
                // after it stay held back.
                later_predictor_bytes = later_predictor_bytes.saturating_sub(predictor_bytes);
                // Each request reserves the encoder memory it creates
                // (prefill workspace), so the rows are not also pending.
                self.core_engine
                    .price_tensor_rows_with_requests(model_instance_id)
                    .await?;
                self.model_registry.publish_effective_context(
                    variant,
                    retained.logical_token_reach(),
                )?;
                self.model_registry.publish_kv_storage(variant, &retained);
                crate::runtime::rollout::validate_managed_state_plan_eligibility(
                    variant,
                    capability,
                    retained.state_plan_v2(),
                )?;
                let retained_uses = contracts
                    .iter()
                    .map(|contract| {
                        let graph = stage_graph_fingerprint(&contract.stages)?;
                        let retained_use = match contract.execution_profile.cache_mode {
                            CacheMode::ExternalPaged
                                if contract.execution_profile.cache_namespace.is_some()
                                    && contract.execution_profile.kv_dtype != "none" =>
                            {
                                RetainedStateUseV2::ExternalPaged
                            }
                            CacheMode::None
                                if contract.execution_profile.cache_namespace.is_none()
                                    && contract.execution_profile.kv_dtype == "none" =>
                            {
                                RetainedStateUseV2::Inactive
                            }
                            _ => {
                                return Err(Error::ModelLoadError(
                                    "Qwen3 TTS retained-state graph has an incompatible cache profile"
                                        .to_string(),
                                ));
                            }
                        };
                        Ok((graph, retained_use))
                    })
                    .collect::<Result<HashMap<_, _>>>()?;
                let publication = self
                    .load_invocation_workspace_publication_with_remaining_groups(
                        model_instance_id,
                        &contracts,
                        physical_spec.descriptor,
                        &physical_spec.predictor_contract,
                        Some(retained.into()),
                        retained_uses,
                        LaterStateGroups {
                            remaining: remaining_groups,
                            minimum_bytes: later_predictor_bytes,
                        },
                    )
                    .await?;
                remaining_groups = remaining_groups.saturating_sub(1);
                state_publications.insert(capability, publication);
            }
        }
        Ok(())
    }

    /// VibeVoice TTS: retained and invocation state.
    pub(super) async fn allocate_vibevoice_tts_state(
        &self,
        variant: ModelVariant,
        model_instance_id: ModelInstanceId,
        backend: BackendKind,
        state_fit_reserve: StateFitReserve,
        bundle_draft: &LoadedModelBundleDraft,
        state_publications: &mut HashMap<CapabilityKind, LoadedStatePublication>,
    ) -> Result<()> {
        if variant.family() == crate::catalog::ModelFamily::VibeVoiceTts {
            if !managed_kv_backend_compiled(backend) {
                return Err(Error::ModelLoadError(format!(
                    "loaded model {variant} requires physical TTS invocation state, but the {backend:?} build has no direct paged-attention runtime"
                )));
            }
            let model = self
                .model_registry
                .get_loading_vibevoice_tts(variant)
                .await
                .ok_or_else(|| {
                    Error::ModelLoadError(format!(
                        "loaded VibeVoice TTS model {variant} is missing from the registry"
                    ))
                })?;
            let contracts = bundle_draft.execution_contracts(CapabilityKind::Tts)?;
            let stage_graphs = contracts
                .iter()
                .map(|contract| contract.stages.as_ref())
                .collect::<Vec<_>>();
            let physical_spec = model.physical_state_spec(&stage_graphs)?;
            let retained_contract = physical_spec.retained.as_ref().ok_or_else(|| {
                Error::ModelLoadError(
                    "VibeVoice TTS normal graph did not publish retained state".into(),
                )
            })?;
            let retained_max_tokens = physical_spec.retained_max_tokens;
            let retained = self
                .core_engine
                .load_managed_model_state(
                    model_instance_id,
                    retained_contract,
                    retained_max_tokens,
                    tts_state_minimum_tokens(variant),
                    state_fit_reserve,
                    0,
                )
                .await?;
            self.model_registry.publish_effective_context(
                variant,
                retained.logical_token_reach(),
            )?;
            self.model_registry.publish_kv_storage(variant, &retained);
            crate::runtime::rollout::validate_managed_state_plan_eligibility(
                variant,
                CapabilityKind::Tts,
                retained.state_plan_v2(),
            )?;
            let retained_uses = contracts
                .iter()
                .map(|contract| {
                    let graph = stage_graph_fingerprint(&contract.stages)?;
                    let retained_use = match contract.execution_profile.cache_mode {
                        CacheMode::ExternalPaged => RetainedStateUseV2::ExternalPaged,
                        CacheMode::None => RetainedStateUseV2::Inactive,
                    };
                    Ok((graph, retained_use))
                })
                .collect::<Result<HashMap<_, _>>>()?;
            let publication = self
                .load_invocation_workspace_publication(
                    model_instance_id,
                    &contracts,
                    physical_spec.descriptor,
                    &physical_spec.invocation,
                    Some(retained.into()),
                    retained_uses,
                )
                .await?;
            state_publications.insert(CapabilityKind::Tts, publication);
        }
        Ok(())
    }

    /// Fish S2 TTS: retained and invocation state.
    pub(super) async fn allocate_fish_s2_tts_state(
        &self,
        variant: ModelVariant,
        model_instance_id: ModelInstanceId,
        backend: BackendKind,
        bundle_draft: &LoadedModelBundleDraft,
        state_publications: &mut HashMap<CapabilityKind, LoadedStatePublication>,
    ) -> Result<()> {
        if variant.family() == crate::catalog::ModelFamily::FishS2Tts {
            if !managed_kv_backend_compiled(backend) {
                return Err(Error::ModelLoadError(format!(
                    "loaded model {variant} requires physical TTS invocation state, but the {backend:?} build has no direct paged-attention runtime"
                )));
            }
            let model = self
                .model_registry
                .get_loading_fish_s2_tts(variant)
                .await
                .ok_or_else(|| {
                    Error::ModelLoadError(format!(
                        "loaded Fish S2 TTS model {variant} is missing from the registry"
                    ))
                })?;
            let headroom = match self.coordinator.resource_authority().planning_headroom_bytes(backend)? {
                ResourceAmount::Known(bytes) => Some(bytes),
                _ => None,
            };
            let native_batching = match crate::env::value("IZWI_FISH_NATIVE_BATCHING") {
                Some(value) => izwi_env::parse_bool("IZWI_FISH_NATIVE_BATCHING", &value)
                    .map_err(|error| Error::ConfigError(error.to_string()))?,
                None => false,
            };
            // Automatic staged rows are sized for tensor-width launches. Fish
            // fits its own autoregressive width (no tensor arena backs its
            // transactions), so it reads an automatic limit as automatic.
            let fish_limits = if self.core_engine.config().automatic_row_limits.staged_transactions {
                std::borrow::Cow::Owned(crate::config::EngineConfig { max_staged_transactions: 0, ..self.config.clone() })
            } else {
                std::borrow::Cow::Borrowed(&self.config)
            };
            let capacity = crate::runtime::fish_capacity::ResolvedFishServingCapacity::resolve(
                model.config(), &fish_limits, headroom, native_batching,
                model.state_element_bytes(),
            )?;
            tracing::info!(?capacity, ?backend, "Sealing Fish serving capacity; device throughput qualification is separate");
            bundle_draft.seal_fish_capacity(capacity.clone())?;
            state_publications.extend(self.load_fish_s2_state_publications(
                model_instance_id, bundle_draft,
                self.config.max_sequence_length.explicit_tokens().unwrap_or(model.config().max_seq_len).min(model.config().max_seq_len),
                &capacity,
                |graphs| model.physical_state_spec(graphs),
            ).await?);
        }
        Ok(())
    }

    /// Voxtral TTS: retained and invocation state.
    pub(super) async fn allocate_voxtral_tts_state(
        &self,
        variant: ModelVariant,
        model_instance_id: ModelInstanceId,
        backend: BackendKind,
        state_fit_reserve: StateFitReserve,
        bundle_draft: &LoadedModelBundleDraft,
        state_publications: &mut HashMap<CapabilityKind, LoadedStatePublication>,
    ) -> Result<()> {
        if variant.family() == crate::catalog::ModelFamily::VoxtralTts {
            if !managed_kv_backend_compiled(backend) {
                return Err(Error::ModelLoadError(format!(
                    "loaded model {variant} requires physical TTS invocation state, but the {backend:?} build has no direct paged-attention runtime"
                )));
            }
            let model = self
                .model_registry
                .get_loading_voxtral_tts(variant)
                .await
                .ok_or_else(|| {
                    Error::ModelLoadError(format!(
                        "loaded Voxtral TTS model {variant} is missing from the registry"
                    ))
                })?;
            let contracts = bundle_draft.execution_contracts(CapabilityKind::Tts)?;
            let stage_graphs = contracts
                .iter()
                .map(|contract| contract.stages.as_ref())
                .collect::<Vec<_>>();
            let physical_spec = model.physical_state_spec(&stage_graphs)?;
            // The engine route runs the LM on retained paged K/V (one
            // context per request, shared by prefill and decode).
            let retained = self
                .core_engine
                .load_managed_model_state(
                    model_instance_id,
                    &physical_spec.retained,
                    Some(physical_spec.retained_max_tokens),
                    tts_state_minimum_tokens(variant),
                    state_fit_reserve,
                    0,
                )
                .await?;
            self.model_registry.publish_effective_context(
                variant,
                retained.logical_token_reach(),
            )?;
            self.model_registry.publish_kv_storage(variant, &retained);
            crate::runtime::rollout::validate_managed_state_plan_eligibility(
                variant,
                CapabilityKind::Tts,
                retained.state_plan_v2(),
            )?;
            let retained_uses = contracts
                .iter()
                .map(|contract| {
                    Ok((
                        stage_graph_fingerprint(&contract.stages)?,
                        RetainedStateUseV2::ExternalPaged,
                    ))
                })
                .collect::<Result<HashMap<_, _>>>()?;
            // Stages keep only scratch; no stage owns K/V of its own.
            let publication = self.load_scratch_only_workspace_publication(
                &contracts,
                physical_spec.descriptor,
                Some(retained.into()),
                retained_uses,
            )?;
            state_publications.insert(CapabilityKind::Tts, publication);
        }
        Ok(())
    }

    /// Chatterbox TTS: T3's two retained guidance branches, with per-stage
    /// scratch for the rest.
    pub(super) async fn allocate_chatterbox_tts_state(
        &self,
        variant: ModelVariant,
        model_instance_id: ModelInstanceId,
        backend: BackendKind,
        state_fit_reserve: StateFitReserve,
        bundle_draft: &LoadedModelBundleDraft,
        state_publications: &mut HashMap<CapabilityKind, LoadedStatePublication>,
    ) -> Result<()> {
        if variant.family() != crate::catalog::ModelFamily::ChatterboxTts {
            return Ok(());
        }
        if !managed_kv_backend_compiled(backend) {
            return Err(Error::ModelLoadError(format!(
                "loaded model {variant} requires physical TTS state, but the {backend:?} build has no direct paged-attention runtime"
            )));
        }
        let model = self
            .model_registry
            .get_loading_chatterbox_tts(variant)
            .await
            .ok_or_else(|| {
                Error::ModelLoadError(format!(
                    "loaded Chatterbox model {variant} is missing from the registry"
                ))
            })?;
        let contracts = bundle_draft.execution_contracts(CapabilityKind::Tts)?;
        let stage_graphs = contracts
            .iter()
            .map(|contract| contract.stages.as_ref())
            .collect::<Vec<_>>();
        let physical_spec = model.physical_state_spec(&stage_graphs)?;
        let retained = self
            .core_engine
            .load_managed_model_state(
                model_instance_id,
                &physical_spec.retained,
                Some(physical_spec.retained_max_tokens),
                tts_state_minimum_tokens(variant),
                state_fit_reserve,
                0,
            )
            .await?;
        self.model_registry
            .publish_effective_context(variant, retained.logical_token_reach())?;
        self.model_registry.publish_kv_storage(variant, &retained);
        crate::runtime::rollout::validate_managed_state_plan_eligibility(
            variant,
            CapabilityKind::Tts,
            retained.state_plan_v2(),
        )?;
        let retained_uses = contracts
            .iter()
            .map(|contract| {
                Ok((
                    stage_graph_fingerprint(&contract.stages)?,
                    RetainedStateUseV2::ExternalPaged,
                ))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        let publication = self.load_scratch_only_workspace_publication(
            &contracts,
            physical_spec.descriptor,
            Some(retained.into()),
            retained_uses,
        )?;
        state_publications.insert(CapabilityKind::Tts, publication);
        Ok(())
    }

    /// Kokoro: the stateless static TTS publication.
    pub(super) async fn allocate_kokoro_tts_state(
        &self,
        variant: ModelVariant,
    ) -> Result<()> {
        if variant.family() == crate::catalog::ModelFamily::KokoroTts {
            let model = self
                .model_registry
                .get_loading_kokoro(variant)
                .await
                .ok_or_else(|| {
                    Error::ModelLoadError(format!(
                        "loaded Kokoro TTS model {variant} is missing from the registry"
                    ))
                })?;
            let effective_context = kokoro_effective_context_tokens(
                model.config().context_length(),
                self.config.portable_context_ceiling(),
            )?;
            self.model_registry
                .publish_effective_context(variant, effective_context)?;
        }
        Ok(())
    }
}
