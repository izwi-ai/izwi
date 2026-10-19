//! Invocation workspace publication: planning and validating the physical
//! invocation state a loaded capability publishes.

use super::*;

#[derive(Debug, Clone)]
pub(super) struct InvocationAllocationV2 {
    pub(super) adapter_instance: AdapterInstanceId,
    pub(super) key: InvocationWorkspaceKeyV2,
    pub(super) domain: InvocationWorkspaceDomain,
    pub(super) slot_count: u32,
}

pub(super) fn invalid_invocation_publication(message: impl Into<String>) -> Error {
    Error::ModelLoadError(message.into())
}

pub(super) fn validate_physical_publication_backing(
    descriptor: &CapabilityStateDescriptorV2,
    executions: &[LoadedExecutionContract],
    retained: Option<&RetainedStateRuntimeV2>,
    retained_uses: &HashMap<[u8; 32], RetainedStateUseV2>,
) -> Result<()> {
    let execution_graphs = executions
        .iter()
        .map(|execution| stage_graph_fingerprint(&execution.stages))
        .collect::<Result<HashSet<_>>>()?;
    match (&descriptor.retained, retained) {
        (RetainedStateCapability::Stateless, None) if retained_uses.is_empty() => Ok(()),
        (RetainedStateCapability::Managed { contract }, Some(retained))
            if contract.fingerprint()? == retained.state_plan_v2().contract_fingerprint
                && retained_uses.keys().copied().collect::<HashSet<_>>() == execution_graphs =>
        {
            Ok(())
        }
        (RetainedStateCapability::Stateless, None) => Err(invalid_invocation_publication(
            "invocation-only publication cannot declare retained graph mappings",
        )),
        (RetainedStateCapability::Stateless, Some(_)) => Err(invalid_invocation_publication(
            "invocation-only publication unexpectedly owns retained backing",
        )),
        (RetainedStateCapability::Managed { .. }, None) => Err(invalid_invocation_publication(
            "retained invocation publication is missing physical retained backing",
        )),
        (RetainedStateCapability::Managed { .. }, Some(_)) => Err(invalid_invocation_publication(
            "retained invocation publication has mismatched backing or graph mappings",
        )),
    }
}

/// Resolve one capability-authored invocation descriptor into exact physical
/// allocations. This is deliberately model-neutral: graph, stage, domain, and
/// concurrency identities all come from the sealed execution/state contracts.
pub(super) fn plan_invocation_allocations(
    descriptor: &CapabilityStateDescriptorV2,
    invocation_contract: &InferenceStateContract,
    executions: &[LoadedExecutionContract],
) -> Result<Vec<InvocationAllocationV2>> {
    if executions.is_empty() {
        return Err(invalid_invocation_publication(
            "physical invocation publication has no execution stage graphs",
        ));
    }
    invocation_contract.validate()?;

    let mut contract_domains = HashMap::new();
    for domain in &invocation_contract.domains {
        if domain.scope() != StateScope::Invocation {
            return Err(invalid_invocation_publication(
                "physical invocation contract contains retained state",
            ));
        }
        if contract_domains.insert(domain.id(), domain).is_some() {
            return Err(invalid_invocation_publication(
                "physical invocation contract repeats a state domain",
            ));
        }
    }

    let mut executions_by_graph = HashMap::new();
    for execution in executions {
        let graph = stage_graph_fingerprint(&execution.stages)?;
        match executions_by_graph.entry(graph) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(execution);
            }
            std::collections::hash_map::Entry::Occupied(entry) => {
                let current = entry.get();
                if current.adapter_instance_id != execution.adapter_instance_id
                    || current.stages != execution.stages
                {
                    return Err(invalid_invocation_publication(
                        "one invocation stage graph maps to multiple loaded adapters",
                    ));
                }
            }
        }
    }

    let InvocationWorkspaceSet::Bounded { profiles } = &descriptor.invocation else {
        return Err(invalid_invocation_publication(
            "physical invocation publication has no bounded workspace profiles",
        ));
    };
    let execution_graphs = executions_by_graph.keys().copied().collect::<HashSet<_>>();
    let profile_graphs = profiles
        .iter()
        .map(|profile| profile.stage_graph_fingerprint)
        .collect::<HashSet<_>>();
    if profile_graphs.len() != profiles.len() || profile_graphs != execution_graphs {
        return Err(invalid_invocation_publication(
            "physical invocation profiles do not map exactly to the loaded stage graphs",
        ));
    }

    let mut mapped_domains: HashMap<StateDomainId, &StateDomainSpec> = HashMap::new();
    let mut physical_keys = HashSet::new();
    let mut allocations = Vec::new();
    for profile in profiles {
        let execution = executions_by_graph
            .get(&profile.stage_graph_fingerprint)
            .copied()
            .ok_or_else(|| {
                invalid_invocation_publication(
                    "physical invocation profile lost its loaded stage graph",
                )
            })?;
        descriptor.validate_against_stages(&execution.stages)?;
        for workspace in &profile.stages {
            let stage = execution
                .stages
                .iter()
                .find(|candidate| candidate.id == workspace.stage)
                .ok_or_else(|| {
                    invalid_invocation_publication(
                        "physical invocation profile lost its execution stage",
                    )
                })?;
            let slot_count = workspace.slot_count(stage.max_batch_size)?;
            for domain in &workspace.domains {
                let InvocationWorkspaceDomain::State {
                    state, capacity, ..
                } = domain
                else {
                    // Scratch is accounted by the stage workspace formula and
                    // has no persistent typed backing to allocate here.
                    continue;
                };
                let capacity_matches = matches!(state, StateDomainSpec::PagedAttention(_))
                    && capacity.paged_max_tokens().is_some()
                    || matches!(
                        (state, capacity),
                        (
                            StateDomainSpec::StaticAttention(_)
                                | StateDomainSpec::Tensor(_)
                                | StateDomainSpec::Append(_)
                                | StateDomainSpec::Ring(_)
                                | StateDomainSpec::StaticTensor(_),
                            InvocationStateCapacity::SemanticBounded
                        )
                    );
                if !capacity_matches {
                    return Err(invalid_invocation_publication(
                        "physical invocation descriptor uses a capacity incompatible with its state domain",
                    ));
                }
                match mapped_domains.entry(state.id()) {
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(state);
                    }
                    std::collections::hash_map::Entry::Occupied(entry) if *entry.get() == state => {
                    }
                    std::collections::hash_map::Entry::Occupied(_) => {
                        return Err(invalid_invocation_publication(
                            "one invocation domain has inconsistent physical definitions",
                        ));
                    }
                }
                let key = InvocationWorkspaceKeyV2 {
                    stage_graph: profile.stage_graph_fingerprint,
                    stage: workspace.stage,
                    domain: state.id(),
                };
                if !physical_keys.insert(key) {
                    return Err(invalid_invocation_publication(
                        "physical invocation publication repeats a graph/stage/domain mapping",
                    ));
                }
                allocations.push(InvocationAllocationV2 {
                    adapter_instance: execution.adapter_instance_id,
                    key,
                    domain: domain.clone(),
                    slot_count,
                });
            }
        }
    }

    if mapped_domains.len() != contract_domains.len()
        || mapped_domains.iter().any(|(id, state)| {
            contract_domains
                .get(id)
                .is_none_or(|contract_state| *contract_state != *state)
        })
    {
        return Err(invalid_invocation_publication(
            "physical invocation descriptor and contract have missing or extra domain mappings",
        ));
    }
    if allocations.is_empty() {
        return Err(invalid_invocation_publication(
            "physical invocation publication has no typed state domains",
        ));
    }
    Ok(allocations)
}

pub(super) fn validate_scratch_only_invocation_publication(
    descriptor: &CapabilityStateDescriptorV2,
    executions: &[LoadedExecutionContract],
) -> Result<()> {
    if executions.is_empty() {
        return Err(invalid_invocation_publication(
            "scratch-only invocation publication has no execution stage graphs",
        ));
    }
    let execution_graphs = executions
        .iter()
        .map(|execution| stage_graph_fingerprint(&execution.stages))
        .collect::<Result<HashSet<_>>>()?;
    let profile_graphs = match &descriptor.invocation {
        InvocationWorkspaceSet::None {
            stage_graph_fingerprints,
        } => stage_graph_fingerprints
            .iter()
            .copied()
            .collect::<HashSet<_>>(),
        InvocationWorkspaceSet::Bounded { profiles } => {
            if profiles.iter().any(|profile| {
                profile.stages.iter().any(|stage| {
                    stage
                        .domains
                        .iter()
                        .any(|domain| matches!(domain, InvocationWorkspaceDomain::State { .. }))
                })
            }) {
                return Err(invalid_invocation_publication(
                    "scratch-only invocation publication contains typed state",
                ));
            }
            profiles
                .iter()
                .map(|profile| profile.stage_graph_fingerprint)
                .collect::<HashSet<_>>()
        }
    };
    if profile_graphs != execution_graphs {
        return Err(invalid_invocation_publication(
            "scratch-only invocation profiles do not map exactly to the loaded stage graphs",
        ));
    }
    for execution in executions {
        descriptor.validate_against_stages(&execution.stages)?;
    }
    Ok(())
}

impl ModelLifecycleController {
    /// Seal each advertised Fish route against its own adapter identity. Slow
    /// retained state is model-scoped and reused; invocation backing is per adapter.
    pub(super) async fn load_fish_s2_state_publications(
        &self,
        model_instance_id: ModelInstanceId,
        bundle_draft: &LoadedModelBundleDraft,
        max_sequence_tokens: usize,
        capacity: &crate::runtime::fish_capacity::ResolvedFishServingCapacity,
        physical_state_spec: impl Fn(&[&[StageDescriptor]]) -> Result<FishS2PhysicalStateSpec>,
    ) -> Result<HashMap<CapabilityKind, LoadedStatePublication>> {
        let variant = ModelVariant::FishAudioS2Pro;
        let mut publications = HashMap::new();
        let mut retained_runtime: Option<Arc<crate::engine::ManagedKvModelRuntime>> = None;
        for capability in [CapabilityKind::Tts, CapabilityKind::StreamingTts] {
            let contracts = bundle_draft.execution_contracts(capability)?;
            let stage_graphs = contracts
                .iter()
                .map(|contract| contract.stages.as_ref())
                .collect::<Vec<_>>();
            let physical_spec = physical_state_spec(&stage_graphs)?;
            let retained_contract = physical_spec.retained.as_ref().ok_or_else(|| {
                Error::ModelLoadError(
                    "Fish S2 TTS normal graph did not publish retained state".into(),
                )
            })?;
            // Fit and allocate slow state once. Refitting after its allocation
            // would incorrectly price another copy against remaining headroom.
            let retained = match &retained_runtime {
                Some(retained) => {
                    if retained.state_plan_v2().contract_fingerprint
                        != retained_contract.fingerprint()?
                    {
                        return Err(Error::ModelLoadError(
                            "Fish capabilities disagree on retained state".into(),
                        ));
                    }
                    retained.clone()
                }
                None => {
                    let retained = self
                        .core_engine
                        .load_managed_model_state_with_row_limits(
                            model_instance_id,
                            retained_contract,
                            Some(max_sequence_tokens),
                            capacity.active_rows,
                            capacity.staged_rows,
                        )
                        .await?;
                    retained_runtime = Some(retained.clone());
                    retained
                }
            };
            self.model_registry
                .publish_effective_context(variant, retained.logical_token_reach())?;
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
            publications.insert(capability, publication);
        }
        Ok(publications)
    }

    pub(super) fn load_scratch_only_workspace_publication(
        &self,
        executions: &[LoadedExecutionContract],
        descriptor: CapabilityStateDescriptorV2,
        retained: Option<RetainedStateRuntimeV2>,
        retained_uses: HashMap<[u8; 32], RetainedStateUseV2>,
    ) -> Result<LoadedStatePublication> {
        validate_physical_publication_backing(
            &descriptor,
            executions,
            retained.as_ref(),
            &retained_uses,
        )?;
        validate_scratch_only_invocation_publication(&descriptor, executions)?;
        Ok(LoadedStatePublication::PhysicalV2 {
            descriptor,
            retained,
            retained_uses,
            invocation_workspace: InvocationWorkspaceRuntimeV2::default(),
        })
    }

    pub(super) fn fit_invocation_decoder_context(
        &self,
        variant: ModelVariant,
        descriptor: &mut CapabilityStateDescriptorV2,
        invocation_contract: &InferenceStateContract,
        executions: &[LoadedExecutionContract],
        remaining_automatic_state_groups: u64,
    ) -> Result<()> {
        let backend = self.backend_router.context().backend_kind;
        if backend == BackendKind::Cuda {
            return Ok(());
        }
        let axis_bounds = descriptor.capacity_axis_bounds(StateCapacityAxis::DecoderContext);
        // ASR state leaves live room for the requests it serves (ADM-7), and
        // keeps the one reserve the retained fit keeps too (ADM-11): the
        // configured floor or the measured untracked growth, not a catalog
        // difference.
        let authority = self.coordinator.resource_authority();
        let asr = asr_memory::is_asr_variant(variant);
        let headroom = authority.planning_headroom_bytes(backend)?;
        let (live_headroom, safety_reserve_bytes) = if asr {
            (
                authority.state_fit_headroom_bytes(backend)?,
                authority.state_safety_reserve_bytes(
                    backend,
                    self.config.portable_context_reserve_bytes,
                ),
            )
        } else {
            (
                headroom,
                portable_context_reserve_bytes(variant, self.config.portable_context_reserve_bytes),
            )
        };
        // A capacity shortfall of ASR state is a 503, not a load defect.
        let does_not_fit = |message: String| {
            if asr {
                Error::Overloaded(message)
            } else {
                Error::ModelLoadError(message)
            }
        };
        let ResourceAmount::Known(headroom_bytes) = headroom else {
            let Some((_, maximum)) = axis_bounds else {
                return Ok(());
            };
            let fallback = self
                .config
                .max_sequence_length
                .explicit_tokens()
                .map_or(maximum.min(4_096), |tokens| maximum.min(tokens as u64));
            descriptor.resolve_capacity_axis(StateCapacityAxis::DecoderContext, fallback)?;
            self.model_registry
                .publish_effective_context(variant, fallback)?;
            return Ok(());
        };
        let (budget, growth_budget) =
            state_fit_budgets(headroom_bytes, live_headroom, safety_reserve_bytes);
        let automatic_budget =
            automatic_state_group_budget(growth_budget, remaining_automatic_state_groups);
        let required = |tokens: Option<u64>| -> Result<u64> {
            let mut candidate = descriptor.clone();
            if let Some(tokens) = tokens {
                candidate.resolve_capacity_axis(StateCapacityAxis::DecoderContext, tokens)?;
            }
            plan_invocation_allocations(&candidate, invocation_contract, executions)?
                .into_iter()
                .try_fold(0_u64, |total, allocation| {
                    total
                        .checked_add(
                            allocation
                                .domain
                                .maximum_bytes()?
                                .checked_mul(u64::from(allocation.slot_count))
                                .ok_or_else(|| {
                                    Error::ModelLoadError(
                                        "invocation context byte plan overflow".into(),
                                    )
                                })?,
                        )
                        .ok_or_else(|| {
                            Error::ModelLoadError("invocation context byte plan overflow".into())
                        })
                })
        };
        let Some((minimum, maximum)) = axis_bounds else {
            let bytes = required(None)?;
            if bytes > budget {
                tracing::warn!(
                    "portable fixed invocation state does not fit: state_bytes={bytes}, planning_headroom={headroom_bytes}, safety_reserve={safety_reserve_bytes}"
                );
                return Err(does_not_fit(state_shortfall_reason(
                    format_args!("not enough memory for {variant}'s working state"),
                    bytes,
                    headroom_bytes,
                    safety_reserve_bytes,
                )));
            }
            return Ok(());
        };
        // The invocation context ceiling is the model-authored maximum; the
        // memory-budgeted binary search below decides how much of it fits.
        let intent = portable_invocation_context_intent(
            self.config.max_sequence_length,
            self.model_registry.effective_context(variant),
            maximum,
        );
        let selected = if let PortableInvocationContextIntent::Explicit { selected } = intent {
            let bytes = required(Some(selected))?;
            if bytes > budget {
                tracing::warn!(
                    "explicit portable context does not fit: context={selected}, state_bytes={bytes}, planning_headroom={headroom_bytes}, safety_reserve={safety_reserve_bytes}"
                );
                return Err(does_not_fit(state_shortfall_reason(
                    format_args!("not enough memory for {variant} at a {selected}-token context"),
                    bytes,
                    headroom_bytes,
                    safety_reserve_bytes,
                )));
            }
            selected
        } else {
            let PortableInvocationContextIntent::Automatic { ceiling } = intent else {
                unreachable!("portable invocation context intent was matched above")
            };
            // An ASR invocation holds at least one chunk per sequence, unless
            // the model's own context is smaller (P2.5).
            let minimum = if asr {
                minimum.max(asr_minimum_tokens(variant).min(maximum))
            } else {
                minimum
            };
            if ceiling < minimum {
                return Err(does_not_fit(format!(
                    "published automatic context is below the invocation minimum: context={ceiling}, minimum={minimum}"
                )));
            }
            let minimum_bytes = required(Some(minimum))?;
            if minimum_bytes > budget {
                tracing::warn!(
                    "portable invocation context minimum does not fit: context={minimum}, state_bytes={minimum_bytes}, planning_headroom={headroom_bytes}, safety_reserve={safety_reserve_bytes}"
                );
                return Err(does_not_fit(state_shortfall_reason(
                    format_args!(
                        "not enough memory for {variant}'s minimum {minimum}-token context"
                    ),
                    minimum_bytes,
                    headroom_bytes,
                    safety_reserve_bytes,
                )));
            }
            let (mut low, mut high) = (minimum, ceiling);
            while low < high {
                let middle = low + (high - low).div_ceil(2);
                if required(Some(middle))? <= automatic_budget {
                    low = middle;
                } else {
                    high = middle - 1;
                }
            }
            low
        };
        tracing::info!(
            model = %variant,
            selected_tokens = selected,
            state_bytes = required(Some(selected))?,
            planning_headroom = headroom_bytes,
            safety_reserve = safety_reserve_bytes,
            "Fitted invocation decoder context"
        );
        descriptor.resolve_capacity_axis(StateCapacityAxis::DecoderContext, selected)?;
        let published = self
            .model_registry
            .effective_context(variant)
            .map_or(selected, |current| selected.min(current as u64));
        self.model_registry
            .publish_effective_context(variant, published)?;
        Ok(())
    }

    pub(super) async fn load_invocation_workspace_publication(
        &self,
        model_instance_id: ModelInstanceId,
        executions: &[LoadedExecutionContract],
        descriptor: CapabilityStateDescriptorV2,
        invocation_contract: &InferenceStateContract,
        retained: Option<RetainedStateRuntimeV2>,
        retained_uses: HashMap<[u8; 32], RetainedStateUseV2>,
    ) -> Result<LoadedStatePublication> {
        self.load_invocation_workspace_publication_with_remaining_groups(
            model_instance_id,
            executions,
            descriptor,
            invocation_contract,
            retained,
            retained_uses,
            1,
        )
        .await
    }

    pub(super) async fn load_invocation_workspace_publication_with_remaining_groups(
        &self,
        model_instance_id: ModelInstanceId,
        executions: &[LoadedExecutionContract],
        mut descriptor: CapabilityStateDescriptorV2,
        invocation_contract: &InferenceStateContract,
        retained: Option<RetainedStateRuntimeV2>,
        retained_uses: HashMap<[u8; 32], RetainedStateUseV2>,
        remaining_automatic_state_groups: u64,
    ) -> Result<LoadedStatePublication> {
        let variant = self
            .resident_variant_for_instance(model_instance_id)
            .ok_or_else(|| {
                Error::ModelLoadError(
                    "invocation state lost its authoritative model generation".into(),
                )
            })?;
        self.fit_invocation_decoder_context(
            variant,
            &mut descriptor,
            invocation_contract,
            executions,
            remaining_automatic_state_groups,
        )?;
        validate_physical_publication_backing(
            &descriptor,
            executions,
            retained.as_ref(),
            &retained_uses,
        )?;
        let allocations =
            plan_invocation_allocations(&descriptor, invocation_contract, executions)?;
        let mut bindings = Vec::with_capacity(allocations.len());
        for allocation in allocations {
            let backing = self
                .core_engine
                .resolve_and_load_invocation_workspace(
                    model_instance_id,
                    allocation.adapter_instance,
                    allocation.key.stage_graph,
                    allocation.key.stage,
                    invocation_contract,
                    &allocation.domain,
                    allocation.slot_count,
                )
                .await?;
            bindings.push(InvocationWorkspaceBindingV2 {
                key: allocation.key,
                backing,
            });
        }
        Ok(LoadedStatePublication::PhysicalV2 {
            descriptor,
            retained,
            retained_uses,
            invocation_workspace: InvocationWorkspaceRuntimeV2::new(bindings)?,
        })
    }
}
