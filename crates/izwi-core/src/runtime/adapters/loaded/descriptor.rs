//! Loaded capability descriptors: execution contracts and the state
//! publication each loaded capability seals.

use super::*;

pub(super) fn loaded_execution_contracts(
    execution: &dyn LoadedExecutionAdapter,
) -> Result<Vec<LoadedExecutionContract>> {
    let metadata = execution.metadata();
    let mut requirements = vec![
        StreamingRequirements::NONE,
        StreamingRequirements::transport_only(),
    ];
    if metadata.streaming_mode != StreamingMode::None {
        requirements.push(StreamingRequirements {
            transport_output: false,
            model_native: true,
            asr_long_form: false,
        });
        requirements.push(StreamingRequirements::native(true));
    }
    // Every ASR family enumerates its long-form variant: adapters that
    // implement the atomic graph produce it here, and adapters whose contract
    // ignores the flag re-seal an identical graph (consistent identities).
    if metadata.capability == CapabilityKind::Asr {
        let long_form = requirements
            .iter()
            .copied()
            .map(|requirements| requirements.with_asr_long_form(true))
            .collect::<Vec<_>>();
        requirements.extend(long_form);
    }
    // VibeVoice and Fish S2 TTS retain their public/direct atomic routes as
    // separately authenticated invocation graphs while normal requests use
    // retained sequence graphs. The internal long-form bit is only a graph-
    // enumeration discriminator here; request routing never labels TTS as ASR.
    if metadata.capability == CapabilityKind::Tts
        && matches!(
            metadata.model_variant.family(),
            crate::catalog::ModelFamily::VibeVoiceTts
        )
    {
        let atomic = requirements
            .iter()
            .copied()
            .map(|requirements| requirements.with_asr_long_form(true))
            .collect::<Vec<_>>();
        requirements.extend(atomic);
    }
    let contracts = requirements
        .into_iter()
        .map(|requirements| {
            let contract = execution.contract(requirements)?;
            contract.validate_physical_launch_policy()?;
            Ok(contract)
        })
        .collect::<Result<Vec<_>>>()?;
    let launch_policy = contracts
        .first()
        .map(|contract| {
            contract
                .execution_profile
                .effective_physical_launch_policy()
        })
        .ok_or_else(|| Error::ModelLoadError("loaded adapter produced no contracts".into()))?;
    if contracts.iter().any(|contract| {
        contract
            .execution_profile
            .effective_physical_launch_policy()
            != launch_policy
    }) {
        return Err(Error::ModelLoadError(
            "one loaded adapter instance produced inconsistent launch policies".into(),
        ));
    }
    Ok(contracts)
}

impl LoadedCapabilityDescriptor {
    pub(super) fn new(
        execution: Arc<dyn LoadedExecutionAdapter>,
        state: Option<LoadedStatePublication>,
        backend_kind: BackendKind,
    ) -> Result<Self> {
        let contracts = loaded_execution_contracts(execution.as_ref())?;
        for contract in &contracts {
            if contract.execution_profile.backend != backend_kind {
                return Err(Error::ModelLoadError(
                    "state ABI v2 execution contract does not match the authoritative loaded backend"
                        .to_string(),
                ));
            }
        }
        let state = match state {
            Some(state) => state,
            None if execution.metadata().state_requirement
                == InferenceStateRequirement::Stateless
                && contracts.iter().all(|contract| {
                    contract.execution_profile.cache_mode == CacheMode::None
                        && contract
                            .stages
                            .iter()
                            .all(|stage| stage.max_workspace_bytes == 0)
                }) =>
            {
                let stage_graphs = contracts
                    .iter()
                    .map(|contract| contract.stages.as_ref())
                    .collect::<Vec<_>>();
                LoadedStatePublication::V2(CapabilityStateDescriptorV2::stateless_for_stage_graphs(
                    &stage_graphs,
                )?)
            }
            None => {
                return Err(Error::ModelLoadError(format!(
                    "loaded model {} capability {:?} requires an explicit load-sealed ABI-v2 state publication",
                    execution.metadata().model_variant,
                    execution.metadata().capability,
                )));
            }
        };
        let mut v2_runtimes = HashMap::new();
        let state = match state {
            LoadedStatePublication::V2(descriptor) => {
                if !descriptor.is_stateless() {
                    return Err(Error::ModelLoadError(
                        "managed state ABI v2 publication requires physical backing".to_string(),
                    ));
                }
                if execution.metadata().state_requirement.requires_retained() {
                    return Err(Error::ModelLoadError(
                        "capability requiring retained inference state cannot publish a stateless runtime"
                            .to_string(),
                    ));
                }
                for contract in &contracts {
                    if contract.execution_profile.cache_mode != CacheMode::None
                        || contract.execution_profile.cache_namespace.is_some()
                        || contract.execution_profile.kv_dtype != "none"
                    {
                        return Err(Error::ModelLoadError(
                        "stateless state ABI v2 contradicts execution that declares retained cache state"
                            .to_string(),
                        ));
                    }
                    let has_invocation =
                        !descriptor.has_zero_invocation_workspace_for(&contract.stages)?;
                    if has_invocation {
                        return Err(Error::ModelLoadError(
                            "state ABI v2 invocation workspace requires load-sealed physical backing"
                                .to_string(),
                        ));
                    }
                    if execution.metadata().state_requirement.requires_invocation() {
                        return Err(Error::ModelLoadError(
                            "capability requiring invocation state cannot publish a zero-workspace runtime"
                                .to_string(),
                        ));
                    }
                    let binding = contract.adapter_binding()?;
                    let stateless = StatelessCapabilityRuntimeV2::seal(
                        backend_kind,
                        &binding,
                        descriptor.clone(),
                    )?;
                    let graph = stateless.stage_graph_fingerprint;
                    let runtime = Arc::new(CapabilityStateRuntimeV2::stateless(stateless));
                    match v2_runtimes.entry(graph) {
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            entry.insert(runtime);
                        }
                        std::collections::hash_map::Entry::Occupied(entry) => {
                            if entry.get().as_ref() != runtime.as_ref() {
                                return Err(Error::ModelLoadError(
                                "one state ABI v2 stage graph resolved to inconsistent runtime identities"
                                    .to_string(),
                            ));
                            }
                        }
                    }
                }
                LoadedStatePublication::V2(descriptor)
            }
            LoadedStatePublication::ManagedV2 { contract, physical } => {
                if !execution.metadata().state_requirement.requires_retained() {
                    return Err(Error::ModelLoadError(
                        "capability declared without retained state published a retained physical runtime"
                            .to_string(),
                    ));
                }
                if execution.metadata().state_requirement.requires_invocation() {
                    return Err(Error::ModelLoadError(
                        "capability requiring invocation state cannot publish a retained-only physical runtime"
                            .to_string(),
                    ));
                }
                let stage_graphs = contracts
                    .iter()
                    .map(|contract| contract.stages.as_ref())
                    .collect::<Vec<_>>();
                let descriptor =
                    CapabilityStateDescriptorV2::managed_for_stage_graphs(contract, &stage_graphs)?;
                for contract in &contracts {
                    let retained_state_use = match contract.execution_profile.cache_mode {
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
                                "managed state ABI v2 requires each graph to declare either external paged state or no retained state"
                                    .to_string(),
                            ));
                        }
                    };
                    let binding = contract.adapter_binding()?;
                    let managed = ManagedCapabilityRuntimeV2::seal(
                        backend_kind,
                        &binding,
                        descriptor.clone(),
                        physical.clone(),
                        retained_state_use,
                    )?;
                    let graph = managed.stage_graph_fingerprint;
                    let runtime = Arc::new(CapabilityStateRuntimeV2::managed(managed));
                    match v2_runtimes.entry(graph) {
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            entry.insert(runtime);
                        }
                        std::collections::hash_map::Entry::Occupied(entry) => {
                            if entry.get().as_ref() != runtime.as_ref() {
                                return Err(Error::ModelLoadError(
                                    "one managed state ABI v2 graph resolved inconsistent runtime identities"
                                        .to_string(),
                                ));
                            }
                        }
                    }
                }
                LoadedStatePublication::V2(descriptor)
            }
            LoadedStatePublication::PhysicalV2 {
                descriptor,
                retained,
                retained_uses,
                invocation_workspace,
            } => {
                if !retained_backing_matches(
                    execution.metadata().state_requirement.requires_retained(),
                    !matches!(descriptor.retained, RetainedStateCapability::Stateless),
                    retained.is_some(),
                ) {
                    return Err(Error::ModelLoadError(
                        "physical retained backing does not match the capability lifetime declaration"
                            .to_string(),
                    ));
                }
                let expected_graphs = contracts
                    .iter()
                    .map(|contract| stage_graph_fingerprint(&contract.stages))
                    .collect::<Result<HashSet<_>>>()?;
                let declared_graphs = retained_uses.keys().copied().collect::<HashSet<_>>();
                if retained.is_some() {
                    if declared_graphs != expected_graphs {
                        return Err(Error::ModelLoadError(
                            "physical retained-state use must be declared for every exact stage graph"
                                .to_string(),
                        ));
                    }
                } else if !declared_graphs.is_empty() {
                    return Err(Error::ModelLoadError(
                        "invocation-only physical state cannot declare retained-state use"
                            .to_string(),
                    ));
                }
                let capability_has_invocation =
                    contracts
                        .iter()
                        .try_fold(false, |has_invocation, contract| {
                            Ok::<_, Error>(
                                has_invocation
                                    || !descriptor
                                        .has_zero_invocation_workspace_for(&contract.stages)?,
                            )
                        })?;
                if execution.metadata().state_requirement.requires_invocation()
                    != capability_has_invocation
                {
                    return Err(Error::ModelLoadError(
                        "physical invocation workspace does not match the capability lifetime declaration"
                            .to_string(),
                    ));
                }
                for contract in &contracts {
                    let binding = contract.adapter_binding()?;
                    let (graph, runtime) = if let Some(retained) = retained.as_ref() {
                        let graph = stage_graph_fingerprint(&contract.stages)?;
                        let retained_state_use =
                            retained_uses.get(&graph).copied().ok_or_else(|| {
                                Error::ModelLoadError(
                                    "physical retained-state use is missing an exact stage graph"
                                        .to_string(),
                                )
                            })?;
                        validate_retained_state_use(
                            retained,
                            retained_state_use,
                            &contract.execution_profile,
                        )?;
                        let managed = ManagedCapabilityRuntimeV2::seal_with_invocation_workspace(
                            backend_kind,
                            &binding,
                            descriptor.clone(),
                            retained.clone(),
                            retained_state_use,
                            invocation_workspace.clone(),
                        )?;
                        (
                            managed.stage_graph_fingerprint,
                            Arc::new(CapabilityStateRuntimeV2::managed(managed)),
                        )
                    } else {
                        if contract.execution_profile.cache_mode != CacheMode::None
                            || contract.execution_profile.cache_namespace.is_some()
                            || contract.execution_profile.kv_dtype != "none"
                        {
                            return Err(Error::ModelLoadError(
                                "invocation-only state ABI v2 graph declared retained cache state"
                                    .to_string(),
                            ));
                        }
                        let invocation =
                            InvocationCapabilityRuntimeV2::seal_with_invocation_workspace(
                                backend_kind,
                                &binding,
                                descriptor.clone(),
                                invocation_workspace.clone(),
                            )?;
                        (
                            invocation.stage_graph_fingerprint,
                            Arc::new(CapabilityStateRuntimeV2::invocation(invocation)),
                        )
                    };
                    match v2_runtimes.entry(graph) {
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            entry.insert(runtime);
                        }
                        std::collections::hash_map::Entry::Occupied(entry) => {
                            if entry.get().as_ref() != runtime.as_ref() {
                                return Err(Error::ModelLoadError(
                                    "one physical state ABI v2 graph resolved inconsistent runtime identities"
                                        .to_string(),
                                ));
                            }
                        }
                    }
                }
                LoadedStatePublication::V2(descriptor)
            }
        };
        Ok(Self {
            execution,
            state,
            v2_runtimes,
        })
    }

    pub(super) fn contract(&self, streaming: StreamingRequirements) -> Result<LoadedExecutionContract> {
        let contract = self.execution.contract(streaming)?;
        contract.validate_physical_launch_policy()?;
        Ok(contract)
    }

    pub(super) fn binding(&self, streaming: StreamingRequirements) -> Result<LoadedCapabilityBinding> {
        let contract = self.contract(streaming)?;
        self.state.validate(&contract.stages)?;
        let execution = contract.adapter_binding()?;
        let state = match &self.state {
            LoadedStatePublication::V2(_) => {
                let graph = stage_graph_fingerprint(&contract.stages)?;
                let runtime = self.v2_runtimes.get(&graph).ok_or_else(|| {
                    Error::InferenceError(
                        "selected execution graph has no load-sealed state ABI v2 runtime"
                            .to_string(),
                    )
                })?;
                runtime.validate_against(contract.execution_profile.backend, &execution)?;
                runtime.clone()
            }
            LoadedStatePublication::ManagedV2 { .. }
            | LoadedStatePublication::PhysicalV2 { .. } => {
                return Err(Error::InferenceError(
                    "managed state publication was not load-sealed".to_string(),
                ));
            }
        };
        Ok(LoadedCapabilityBinding { execution, state })
    }
}

pub(super) fn validate_retained_state_use(
    retained: &RetainedStateRuntimeV2,
    retained_state_use: RetainedStateUseV2,
    profile: &ExecutionProfile,
) -> Result<()> {
    let cacheless = profile.cache_mode == CacheMode::None
        && profile.cache_namespace.is_none()
        && profile.kv_dtype == "none";
    let external_paged = profile.cache_mode == CacheMode::ExternalPaged
        && profile.cache_namespace.is_some()
        && profile.kv_dtype != "none";
    let valid = if retained.is_tensor_only() {
        cacheless
            && matches!(
                retained_state_use,
                RetainedStateUseV2::ExternalTensor | RetainedStateUseV2::Inactive
            )
    } else {
        matches!(
            retained_state_use,
            RetainedStateUseV2::ExternalPaged | RetainedStateUseV2::ExternalPagedStatic
                if external_paged
        ) || matches!(retained_state_use, RetainedStateUseV2::Inactive if cacheless)
    };
    if !valid {
        return Err(Error::ModelLoadError(
            format!(
                "retained-state use {retained_state_use:?} does not match its physical backing (tensor_only={}) and exact execution profile (cache_mode={:?}, namespace={}, kv_dtype={})",
                retained.is_tensor_only(),
                profile.cache_mode,
                profile.cache_namespace.is_some(),
                profile.kv_dtype,
            ),
        ));
    }
    Ok(())
}

/// Whether a physical publication's retained backing matches what it may
/// hold. The capability declares the most it may retain; the model's
/// descriptor declares what this load authors (and the publication checks
/// that against the backing). A load may author less than its capability
/// allows: Granite Speech without resumable prefill offers no retained graph,
/// so its load authors a stateless route and allocates nothing (GRN-8). A
/// load may never author retained state its capability does not declare.
fn retained_backing_matches(
    capability_retains: bool,
    load_authors_retained: bool,
    has_backing: bool,
) -> bool {
    if load_authors_retained && !capability_retains {
        return false;
    }
    has_backing == load_authors_retained
}

#[cfg(test)]
mod retained_backing_tests {
    use super::retained_backing_matches;

    #[test]
    fn a_load_may_author_less_retained_state_than_its_capability_allows() {
        // Retaining capability, retained route authored and backed.
        assert!(retained_backing_matches(true, true, true));
        // Retaining capability, stateless route (Granite without resumable
        // prefill): nothing is allocated, and that matches.
        assert!(retained_backing_matches(true, false, false));
        // Stateless capability, stateless route.
        assert!(retained_backing_matches(false, false, false));
    }

    #[test]
    fn backing_must_follow_the_authored_route_and_never_exceed_the_capability() {
        assert!(!retained_backing_matches(true, true, false));
        assert!(!retained_backing_matches(true, false, true));
        assert!(!retained_backing_matches(false, false, true));
        assert!(!retained_backing_matches(false, true, true));
        assert!(!retained_backing_matches(false, true, false));
    }
}
