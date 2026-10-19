//! Binding requests and model state to the managed cache, and the capacity
//! claims that keep a request's pages reserved.

use super::*;

/// Move the tensor arena's committed backing between the model lease's
/// pending and materialized claims. The tensor envelope is authorized at load
/// but allocated lazily: without this, every committed row stays pending
/// while also shrinking live free memory, so guarded admission charges it
/// twice. Growth is recorded after the state exists (commits publish tensors
/// the model already allocated). A shrink restores the pending claim, so the
/// caller must drop the [`RetiredTensorState`] only after this returns.
///
/// Every caller has already applied its arena transition, so a failure here
/// is logged rather than returned: a stale record keeps the previous claim.
pub(super) fn reconcile_tensor_materialization(state: &mut ManagedKvModelState) {
    if state.tensor_rows_request_priced {
        return;
    }
    let Some(arena) = state.runtime.tensor_state() else {
        return;
    };
    let result = (|| -> Result<Option<u64>> {
        let bytes = arena
            .committed_backing_bytes()?
            .min(arena.capacity().authorized_bytes());
        if bytes == state.tensor_materialized_bytes {
            return Ok(None);
        }
        if let Some(lease) = state.resource_lease.as_ref() {
            let materialized = state
                .materialized_resources
                .checked_add(tensor_state_resources(state.runtime.plan.backend, bytes)?)?;
            if bytes > state.tensor_materialized_bytes {
                lease.record_materialized_usage(materialized)?;
            } else {
                lease.prepare_materialized_release(materialized)?;
            }
        }
        Ok(Some(bytes))
    })();
    match result {
        Ok(Some(bytes)) => state.tensor_materialized_bytes = bytes,
        Ok(None) => {}
        Err(error) => tracing::error!(
            error = %error,
            recorded_bytes = state.tensor_materialized_bytes,
            "managed tensor-state materialization could not be recorded on the model lease"
        ),
    }
}

/// Release one tensor sequence, restoring its pending claim before its state
/// can be freed.
pub(super) fn release_tensor_sequence(
    state: &mut ManagedKvModelState,
    arena: &TensorStateArena,
    sequence: PhysicalStateSequenceId,
) -> Result<()> {
    let retired = arena.release(sequence)?;
    reconcile_tensor_materialization(state);
    drop(retired);
    Ok(())
}

pub(super) fn reserve_managed_arena(
    authority: &Arc<ResourceAuthority>,
    model_instance: ModelInstanceId,
    backend: BackendKind,
    resources: ResourceVector,
) -> Result<ResourceLease> {
    let owner = ReservationOwner::new(
        ReservationClass::Model,
        format!("managed-kv:{}:{backend:?}", model_instance.get()),
    );
    authority.reserve(owner, resources)
}

pub(super) fn ensure_session_tables(state: &mut ManagedKvModelState, session: &SessionKey) -> Result<()> {
    if !state.registered_sessions.insert(session.clone()) {
        return Ok(());
    }
    let generation = ManagedSessionGeneration::INITIAL;
    state
        .session_generations
        .insert(session.clone(), generation);
    let mut registered = Vec::new();
    for group in &state.runtime.plan.groups {
        let coordinator = state
            .coordinators
            .get_mut(&group.arena)
            .expect("resolved arena has a coordinator");
        if let Err(error) = coordinator.register_table(session.clone(), group.domain) {
            for (arena, domain) in registered {
                let _ = state
                    .coordinators
                    .get_mut(&arena)
                    .expect("registered arena exists")
                    .release_table(session, domain);
            }
            state.registered_sessions.remove(session);
            state.session_generations.remove(session);
            return Err(coordinator_error(error));
        }
        registered.push((group.arena, group.domain));
    }
    Ok(())
}

pub(super) fn validate_reservation_session_generation(
    state: &ManagedKvModelState,
    reservation: &ManagedCacheReservation,
) -> Result<()> {
    let current = state
        .session_generations
        .get(&reservation.session)
        .copied()
        .ok_or_else(|| {
            Error::InferenceError(
                "managed KV reservation has no registered session generation".into(),
            )
        })?;
    if reservation.session_generation != current {
        return Err(Error::InferenceError(format!(
            "managed KV reservation session generation {} is stale; current generation is {}",
            reservation.session_generation.get(),
            current.get()
        )));
    }
    Ok(())
}

/// Claim the prompt and the current step, plus up to `quantum_tokens` of
/// output headroom, never more than the request's prompt-plus-output budget.
/// Only the current step is required: if other claims leave no room for the
/// headroom the claim grows just enough for this step, and if not even that
/// fits the row waits for capacity (`Backpressure`). A claim never shrinks.
pub(super) fn ensure_chunked_capacity_claim(
    state: &mut ManagedKvModelState,
    session: &SessionKey,
    request: &EngineCoreRequest,
    work: &WorkUnit,
    quantum_tokens: u32,
) -> Result<bool> {
    let WorkUnit::SequenceStep { input, phase, .. } = work else {
        return ensure_capacity_claim(state, session, request);
    };
    let prompt = u64::try_from(request.num_prompt_tokens())
        .map_err(|_| Error::Overloaded("chunked token demand exceeds u64".into()))?;
    let budget = prompt
        .checked_add(u64::try_from(request.params.max_tokens.max(1)).unwrap_or(u64::MAX))
        .ok_or_else(|| Error::Overloaded("chunked token demand overflowed".into()))?;
    let mut demand_by_arena = HashMap::<KvArenaId, (u64, u64)>::new();
    for group in &state.runtime.plan.groups {
        let projected = request.project_paged_state_input(group.domain, *phase, *input)?;
        let needed = prompt.max(
            u64::try_from(projected.end)
                .map_err(|_| Error::Overloaded("chunked token demand exceeds u64".into()))?,
        );
        let desired = budget
            .max(needed)
            .min(needed.saturating_add(u64::from(quantum_tokens)));
        // A sliding window retains at most the window, the span being written
        // and one page of alignment, exactly as the incremental claim prices.
        let retained = |tokens: u64| -> Result<u64> {
            Ok(sliding_window_for_domain(&state.contract, group.domain)?
                .map(|window| {
                    tokens.min(
                        u64::from(window)
                            .saturating_add(projected.len() as u64)
                            .saturating_add(u64::from(group.page_tokens - 1)),
                    )
                })
                .unwrap_or(tokens))
        };
        let page_tokens = u64::from(group.page_tokens);
        let entry = demand_by_arena.entry(group.arena).or_default();
        entry.0 = entry
            .0
            .checked_add(retained(needed)?.div_ceil(page_tokens))
            .ok_or_else(|| Error::Overloaded("chunked arena page demand overflow".into()))?;
        entry.1 = entry
            .1
            .checked_add(retained(desired)?.div_ceil(page_tokens))
            .ok_or_else(|| Error::Overloaded("chunked arena page demand overflow".into()))?;
    }
    let mut claims = Vec::with_capacity(demand_by_arena.len());
    for (arena, (needed_pages, desired_pages)) in demand_by_arena {
        let group = state
            .runtime
            .plan
            .groups
            .iter()
            .find(|group| group.arena == arena)
            .ok_or_else(|| Error::InferenceError("managed capacity arena disappeared".into()))?;
        let capacity = u64::from(group.capacity_pages);
        if needed_pages > capacity {
            let token_capacity = capacity.saturating_mul(u64::from(group.page_tokens));
            return Err(Error::Overloaded(format!(
                "request {} needs {needed_pages} pages for its next step, but managed state arena {} can retain at most {token_capacity} tokens ({capacity} pages of {} tokens)",
                request.id,
                group.id.get(),
                group.page_tokens
            )));
        }
        let previous = state
            .capacity_claims
            .get(session)
            .and_then(|claims| claims.iter().find(|(claimed, _)| *claimed == arena))
            .map(|(_, pages)| u64::from(*pages))
            .unwrap_or(0);
        let others = state
            .capacity_claims
            .iter()
            .filter(|(owner, _)| *owner != session)
            .flat_map(|(_, claims)| claims)
            .filter(|(claimed, _)| *claimed == arena)
            .map(|(_, pages)| u64::from(*pages))
            .sum::<u64>();
        let available = capacity.saturating_sub(others);
        let required = previous.max(needed_pages);
        if required > available {
            return Err(Error::Backpressure(format!(
                "managed KV chunked admission is waiting: request {} needs {required} pages, {others} are claimed by other requests, capacity is {capacity}",
                request.id
            )));
        }
        let pages = required.max(desired_pages.min(available));
        claims.push((
            arena,
            u32::try_from(pages)
                .map_err(|_| Error::Overloaded("chunked page claim exceeds u32".into()))?,
        ));
    }
    let newly_installed = !state.capacity_claims.contains_key(session);
    state.capacity_claims.insert(session.clone(), claims);
    Ok(newly_installed)
}

pub(super) fn ensure_capacity_claim(
    state: &mut ManagedKvModelState,
    session: &SessionKey,
    request: &EngineCoreRequest,
) -> Result<bool> {
    if state.capacity_claims.contains_key(session) {
        return Ok(false);
    }
    let requested_tokens = request
        .num_prompt_tokens()
        .checked_add(request.params.max_tokens.max(1))
        .ok_or_else(|| Error::Overloaded("request token capacity demand overflowed".into()))?;
    let requested_tokens = u64::try_from(requested_tokens)
        .map_err(|_| Error::Overloaded("request token capacity demand exceeds u64".into()))?;

    let mut requested_by_arena = HashMap::<KvArenaId, u64>::new();
    for group in &state.runtime.plan.groups {
        let window_tokens = sliding_window_for_domain(&state.contract, group.domain)?
            .map(u64::from)
            .unwrap_or(requested_tokens);
        let retained_tokens = requested_tokens.min(window_tokens);
        let required_pages = retained_tokens.div_ceil(u64::from(group.page_tokens));
        let requested = requested_by_arena.entry(group.arena).or_default();
        *requested = requested.saturating_add(required_pages);
    }

    for (arena, requested_pages) in &requested_by_arena {
        let capacity_pages = state
            .runtime
            .plan
            .groups
            .iter()
            .find(|group| group.arena == *arena)
            .map(|group| u64::from(group.capacity_pages))
            .ok_or_else(|| Error::InferenceError("managed capacity arena disappeared".into()))?;
        if *requested_pages > capacity_pages {
            let group = state
                .runtime
                .plan
                .groups
                .iter()
                .find(|group| group.arena == *arena)
                .expect("capacity arena group exists");
            let token_capacity = capacity_pages.saturating_mul(u64::from(group.page_tokens));
            return Err(Error::Overloaded(format!(
                "request {} needs {requested_tokens} prompt-plus-output tokens, but managed state arena {} can retain at most {token_capacity} tokens ({} pages of {} tokens)",
                request.id,
                group.id.get(),
                capacity_pages,
                group.page_tokens
            )));
        }
        let claimed_pages = state
            .capacity_claims
            .values()
            .flat_map(|claims| claims.iter())
            .filter_map(|(claimed_arena, pages)| {
                (*claimed_arena == *arena).then_some(u64::from(*pages))
            })
            .sum::<u64>();
        if claimed_pages.saturating_add(*requested_pages) > capacity_pages {
            return Err(Error::Backpressure(format!(
                "managed KV full-request admission is waiting: request {} needs {} pages, {} are already claimed, and arena capacity is {} pages",
                request.id, requested_pages, claimed_pages, capacity_pages
            )));
        }
    }

    let claims = requested_by_arena
        .into_iter()
        .map(|(arena, pages)| {
            u32::try_from(pages)
                .map(|pages| (arena, pages))
                .map_err(|_| Error::Overloaded("managed KV page claim exceeds u32".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    state.capacity_claims.insert(session.clone(), claims);
    Ok(true)
}

/// Claim the known prompt and the exact next scheduled progress (including
/// adapter-projected speculative spans), never the potential complete answer.
/// Validate all arenas before publishing any changes to the session claim.
pub(super) fn ensure_incremental_capacity_claim(
    state: &mut ManagedKvModelState,
    session: &SessionKey,
    request: &EngineCoreRequest,
    work: &WorkUnit,
) -> Result<bool> {
    let WorkUnit::SequenceStep { input, phase, .. } = work else {
        return ensure_capacity_claim(state, session, request);
    };
    let mut requested_by_arena = HashMap::<KvArenaId, u64>::new();
    for group in &state.runtime.plan.groups {
        let projected = request.project_paged_state_input(group.domain, *phase, *input)?;
        let target = u64::try_from(request.num_prompt_tokens().max(projected.end))
            .map_err(|_| Error::Overloaded("incremental token demand exceeds u64".into()))?;
        // Window movement can retain the old window while preparing the next
        // chunk; reserve its append and page alignment as well.
        let retained = sliding_window_for_domain(&state.contract, group.domain)?
            .map(|window| {
                target.min(
                    u64::from(window)
                        .saturating_add(projected.len() as u64)
                        .saturating_add(u64::from(group.page_tokens - 1)),
                )
            })
            .unwrap_or(target);
        let required_pages = retained.div_ceil(u64::from(group.page_tokens));
        let requested = requested_by_arena.entry(group.arena).or_default();
        *requested = requested
            .checked_add(required_pages)
            .ok_or_else(|| Error::Overloaded("incremental arena page demand overflow".into()))?;
    }
    let mut claims = Vec::with_capacity(requested_by_arena.len());
    for (arena, requested_pages) in requested_by_arena {
        let group = state
            .runtime
            .plan
            .groups
            .iter()
            .find(|group| group.arena == arena)
            .ok_or_else(|| Error::InferenceError("managed capacity arena disappeared".into()))?;
        let previous = state
            .capacity_claims
            .get(session)
            .and_then(|claims| claims.iter().find(|(arena, _)| *arena == group.arena))
            .map(|(_, pages)| u64::from(*pages))
            .unwrap_or(0);
        let pages = requested_pages.max(previous);
        let capacity = u64::from(group.capacity_pages);
        if pages > capacity {
            return Err(Error::Overloaded(format!(
                "request {} next managed step needs {pages} pages, arena capacity is {capacity}",
                request.id
            )));
        }
        let others = state
            .capacity_claims
            .iter()
            .filter(|(owner, _)| *owner != session)
            .flat_map(|(_, claims)| claims)
            .filter(|(arena, _)| *arena == group.arena)
            .map(|(_, pages)| u64::from(*pages))
            .sum::<u64>();
        if others.saturating_add(pages) > capacity {
            return Err(Error::Backpressure(format!(
                "managed KV incremental admission is waiting: request {} needs {pages} pages, {others} are claimed by other requests, capacity is {capacity}",
                request.id
            )));
        }
        claims.push((
            group.arena,
            u32::try_from(pages)
                .map_err(|_| Error::Overloaded("incremental page demand exceeds u32".into()))?,
        ));
    }
    let newly_installed = !state.capacity_claims.contains_key(session);
    state.capacity_claims.insert(session.clone(), claims);
    Ok(newly_installed)
}

pub(super) fn full_context_sequence_capacity(plan: &ResolvedKvPlan, reach: u64) -> u64 {
    if reach == 0 {
        return 0;
    }
    plan.groups
        .iter()
        .map(|group| {
            let pages = reach.div_ceil(u64::from(group.page_tokens));
            u64::from(group.capacity_pages) / pages.max(1)
        })
        .min()
        .unwrap_or(0)
}

impl ManagedKvCacheManager {
    pub(crate) fn capacity_claim_sessions(
        &self,
        model_instance: ModelInstanceId,
    ) -> Vec<SessionKey> {
        self.models
            .get(&model_instance)
            .map(|state| state.capacity_claims.keys().cloned().collect())
            .unwrap_or_default()
    }

    #[cfg(test)]
    pub(crate) fn bind_request(
        &mut self,
        model_instance: ModelInstanceId,
        backend: BackendKind,
        capacity_pages: usize,
        page_tokens_hint: usize,
        capability: &InferenceStateCapability,
    ) -> Result<Option<Arc<ManagedKvModelRuntime>>> {
        let Some(contract) = capability.managed_contract() else {
            return Ok(None);
        };
        let capacity = ManagedStateCapacityRequest {
            total_paged_pages: u32::try_from(capacity_pages).map_err(|_| {
                Error::InvalidInput("managed KV page capacity exceeds u32".to_string())
            })?,
            logical_token_reach: None,
            retained_sequence_rows: u32::try_from(capacity_pages).map_err(|_| {
                Error::InvalidInput("managed KV sequence capacity exceeds u32".to_string())
            })?,
            staged_transaction_rows: u32::try_from(capacity_pages).map_err(|_| {
                Error::InvalidInput("managed KV transaction capacity exceeds u32".to_string())
            })?,
        };
        self.bind_model_state_with_capacity(
            model_instance,
            backend,
            capacity,
            page_tokens_hint,
            contract,
        )
        .map(Some)
    }

    #[cfg(test)]
    pub(crate) fn bind_request_with_capacity(
        &mut self,
        model_instance: ModelInstanceId,
        backend: BackendKind,
        capacity: ManagedStateCapacityRequest,
        page_tokens_hint: usize,
        capability: &InferenceStateCapability,
    ) -> Result<Option<Arc<ManagedKvModelRuntime>>> {
        self.bind_request_with_capacity_policy(
            model_instance,
            backend,
            capacity,
            page_tokens_hint,
            capability,
            false,
        )
    }

    pub(crate) fn bind_request_with_capacity_policy(
        &mut self,
        model_instance: ModelInstanceId,
        backend: BackendKind,
        capacity: ManagedStateCapacityRequest,
        page_tokens_hint: usize,
        capability: &InferenceStateCapability,
        materialize_cuda_paged_at_load: bool,
    ) -> Result<Option<Arc<ManagedKvModelRuntime>>> {
        let Some(contract) = capability.managed_contract() else {
            return Ok(None);
        };
        self.bind_model_state_with_capacity_policy(
            model_instance,
            backend,
            capacity,
            page_tokens_hint,
            contract,
            materialize_cuda_paged_at_load,
        )
        .map(Some)
    }

    pub(crate) fn bind_model_state_with_capacity(
        &mut self,
        model_instance: ModelInstanceId,
        backend: BackendKind,
        capacity: ManagedStateCapacityRequest,
        page_tokens_hint: usize,
        contract: &InferenceStateContract,
    ) -> Result<Arc<ManagedKvModelRuntime>> {
        self.bind_model_state_with_capacity_policy(
            model_instance,
            backend,
            capacity,
            page_tokens_hint,
            contract,
            false,
        )
    }

    /// The runtime `model_instance` already bound for `contract`, if any:
    /// every capability of one model generation shares one retained arena.
    /// An incompatible contract for a bound instance is an error.
    pub(crate) fn bound_runtime(
        &self,
        model_instance: ModelInstanceId,
        backend: BackendKind,
        contract: &InferenceStateContract,
    ) -> Result<Option<Arc<ManagedKvModelRuntime>>> {
        let Some(state) = self.models.get(&model_instance) else {
            return Ok(None);
        };
        if &state.contract != contract
            || state.runtime.plan.backend != backend
            || state.runtime.state_plan_v2.contract_fingerprint != contract.fingerprint()?
        {
            return Err(Error::InvalidInput(
                "one loaded model instance published incompatible managed KV contracts".to_string(),
            ));
        }
        Ok(Some(state.runtime.clone()))
    }

    pub(super) fn bind_model_state_with_capacity_policy(
        &mut self,
        model_instance: ModelInstanceId,
        backend: BackendKind,
        capacity: ManagedStateCapacityRequest,
        page_tokens_hint: usize,
        contract: &InferenceStateContract,
        materialize_cuda_paged_at_load: bool,
    ) -> Result<Arc<ManagedKvModelRuntime>> {
        validate_sliding_contract(contract, backend)?;
        if let Some(runtime) = self.bound_runtime(model_instance, backend, contract)? {
            return Ok(runtime);
        }
        if backend != self.worker_backend {
            return Err(Error::InvalidInput(format!(
                "managed KV request targets {backend:?}, but its worker is bound to {:?}",
                self.worker_backend
            )));
        }
        let backend_runtime = self.backend_runtime.as_ref().ok_or_else(|| {
            Error::InvalidInput(
                self.backend_unavailable
                    .clone()
                    .unwrap_or_else(|| format!("managed KV is unavailable for {backend:?}")),
            )
        })?;
        let page_tokens_hint = u32::try_from(page_tokens_hint)
            .map_err(|_| Error::InvalidInput("managed KV page size exceeds u32".to_string()))?;
        let first_arena_generation = self.next_arena_generation;
        let state_plan_v2 = negotiate_state_plan(
            contract,
            &StateBackendPlanRequest {
                backend,
                device_ordinal: self.worker_device_ordinal,
                page_tokens_hint: Some(page_tokens_hint),
                storage_dtype_hint: None,
            },
        )?;
        let (allocation_plan, tensor_capacity) = plan_managed_state_capacity_with_policy(
            &state_plan_v2,
            model_instance,
            capacity,
            materialize_cuda_paged_at_load,
            self.prefix_snapshot_rows_for(contract),
        )?;
        let plan = ResolvedKvPlan::from_runtime_allocation(
            first_arena_generation,
            &state_plan_v2,
            &allocation_plan,
        )?;
        let tensor_state = tensor_capacity
            .map(|capacity| {
                TensorStateArena::new_with_contract(
                    Arc::new(state_plan_v2.clone()),
                    contract,
                    capacity,
                    self.worker_device.clone(),
                )
            })
            .transpose()?
            .map(Arc::new);

        let maximum_state_resources = allocation_plan.maximum_resources(&state_plan_v2)?;
        let resources = managed_state_resources(backend, maximum_state_resources)?;
        let mut materialized_resources = managed_state_resources(
            backend,
            allocation_plan.initial_state_resources(&state_plan_v2)?,
        )?;
        // A fitted CUDA context is a durable allocation promise, not a
        // one-time observation. Reserve the worst contiguous-replacement peak
        // before Ready publication so later request/workspace allocations
        // cannot consume memory required by lazy KV growth.
        //
        // DS4: the host pool's byte ceiling is part of the model's durable
        // envelope, so the lease authorizes it up front (same charge shape as
        // the materialization below) and the load-time charge cannot exceed
        // the reservation. The supervisor has already validated the budget
        // against the assignment's host/shared limit.
        let mut authorization = if backend == BackendKind::Cuda {
            cuda_managed_state_peak_authorization(resources, &plan.groups)?
        } else {
            resources
        };
        if let Some(policy) = self
            .host_offload_policy
            .filter(|_| self.max_prefix_cache_pages > 0)
        {
            let charge = match backend {
                BackendKind::Metal => ResourceVector {
                    unified_bytes: ResourceAmount::Known(policy.budget_bytes),
                    ..ResourceVector::zero()
                },
                _ => ResourceVector {
                    host_bytes: ResourceAmount::Known(policy.budget_bytes),
                    ..ResourceVector::zero()
                },
            };
            authorization = authorization.checked_add(charge)?;
        }
        let resource_lease = self
            .resource_authority
            .as_ref()
            .map(|authority| {
                reserve_managed_arena(authority, model_instance, backend, authorization)
            })
            .transpose()?;
        let mut arenas = HashMap::with_capacity(plan.groups.len());
        let mut coordinators = HashMap::with_capacity(plan.groups.len());
        let mut prefix_indexes = HashMap::with_capacity(plan.groups.len());
        let mut allocation_ledger = StateAllocationLedger::new(&allocation_plan);
        for group in &plan.groups {
            let config = arena_config(contract, group)?;
            let arena = backend_runtime.allocate_arena(config)?;
            if arena.backend_kind() != self.worker_backend {
                return Err(Error::InferenceError(format!(
                    "managed KV runtime allocated a {:?} arena for a {:?} worker",
                    arena.backend_kind(),
                    self.worker_backend
                )));
            }
            if arena.device_location() != self.worker_device_location {
                return Err(Error::InferenceError(format!(
                    "managed KV runtime allocated arena device {:?} for exact worker device {:?}",
                    arena.device_location(),
                    self.worker_device_location
                )));
            }
            let resident_pages = arena.resident_capacity_pages();
            if arenas.insert(group.arena, arena).is_some() {
                return Err(Error::InferenceError(
                    "resolved KV plan reused one arena identity".to_string(),
                ));
            }
            let requested_owned_bytes = group
                .bytes_per_page
                .checked_mul(u64::from(resident_pages))
                .ok_or_else(|| Error::Overloaded("managed KV byte total overflow".into()))?;
            allocation_ledger.reconcile_group_receipt(
                &allocation_plan,
                crate::kv::v2::StateGroupId::new(group.id.get()),
                group.domain,
                AllocationReceipt {
                    requested_owned_bytes,
                    committed_owned_bytes: requested_owned_bytes,
                    allocator_overhead_bytes: 0,
                    residency: ResidencyMeasurement::Unknown,
                },
            )?;
            self.telemetry.record_backing_allocation();
            coordinators.insert(
                group.arena,
                KvCacheCoordinator::new(group.arena, group.capacity_pages as usize),
            );
            prefix_indexes.insert(
                group.arena,
                CoordinatedPrefixIndex::with_telemetry(
                    prefix_index_pages(
                        self.max_prefix_cache_pages,
                        self.automatic_prefix_cache_pages,
                        backend,
                        materialize_cuda_paged_at_load,
                        group.capacity_pages,
                    ),
                    self.telemetry.clone(),
                ),
            );
        }
        allocation_ledger.ensure_ready(&allocation_plan)?;
        if let Some(lease) = resource_lease.as_ref() {
            // CUDA paged arenas materialize only their sealed initial growth
            // quantum. CPU/Metal remain fully backed; tensor capacity remains
            // demand allocated.
            lease.record_materialized_usage(materialized_resources)?;
        }
        // Tensor-state capacity is independently authorized and may materialize
        // lazily. Keep that sealed envelope in model accounting while paged
        // CUDA backing reports only its current resident extent.
        let non_paged_physical_bytes = tensor_state
            .as_ref()
            .map(|arena| arena.capacity().authorized_bytes())
            .unwrap_or(0);
        // DS4: build the host tier when offload is engaged and committed
        // prefix pages exist to demote. The pool's byte ceiling is charged to
        // the node memory ledger up front (DINV-05) — the budget is the
        // operator's declared ceiling for host-tier pages.
        let host_offload = match self
            .host_offload_policy
            .filter(|_| self.max_prefix_cache_pages > 0)
        {
            Some(policy) => {
                let mut offload = offload::ModelOffload {
                    arenas: HashMap::with_capacity(plan.groups.len()),
                };
                // The model lease is charged one budget, so the arenas share
                // it: giving each arena the whole budget let a multi-domain
                // model keep `arenas x budget` bytes resident on the host.
                let per_arena_budget = split_host_offload_budget(policy.budget_bytes, arenas.len());
                for arena in arenas.values() {
                    offload.arenas.insert(
                        arena.id(),
                        offload::ArenaOffload::new(
                            crate::backends::kv::arena_page_bytes(arena.config()),
                            per_arena_budget,
                        )?,
                    );
                }
                let charge = match backend {
                    BackendKind::Metal => ResourceVector {
                        unified_bytes: ResourceAmount::Known(policy.budget_bytes),
                        ..ResourceVector::zero()
                    },
                    _ => ResourceVector {
                        host_bytes: ResourceAmount::Known(policy.budget_bytes),
                        ..ResourceVector::zero()
                    },
                };
                materialized_resources = materialized_resources.checked_add(charge)?;
                if let Some(lease) = resource_lease.as_ref() {
                    lease.record_materialized_usage(materialized_resources)?;
                }
                Some(offload)
            }
            None => None,
        };
        // DS1.2b: derive the snapshot-sharing surface from the strict contract
        // gate. The declared snapshot interval must land on the resolved page
        // grid. The index is bounded by the snapshot budget the tensor arena
        // authorized beside its rows (zero rows without a prefix namespace).
        let tensor_snapshots = tensor_snapshot_prefix_policy(contract).and_then(|policy| {
            let group = plan
                .groups
                .iter()
                .find(|group| group.domain == policy.paged_domain)?;
            let capacity = tensor_capacity.as_ref()?;
            let interval_tokens = u32::try_from(policy.interval_steps).ok()?;
            if interval_tokens == 0
                || !policy
                    .interval_steps
                    .is_multiple_of(u64::from(group.page_tokens))
            {
                // Hybrid prefix reuse is unusable without snapshots, so say
                // why instead of silently serving every turn cold.
                tracing::warn!(
                    ?model_instance,
                    snapshot_interval_tokens = policy.interval_steps,
                    page_tokens = group.page_tokens,
                    "Recurrent-state snapshot interval is not a multiple of the KV page size; hybrid prefix reuse is disabled for this model"
                );
                return None;
            }
            Some(TensorSnapshotSharing {
                paged_arena: group.arena,
                interval_tokens,
                index: TensorStateSnapshotIndex::with_telemetry(
                    usize::try_from(capacity.snapshot_capacity()).unwrap_or(0),
                    capacity.snapshot_capacity_bytes(),
                    self.telemetry.clone(),
                ),
                policy,
            })
        });
        let runtime = Arc::new(ManagedKvModelRuntime {
            plan: Arc::new(plan),
            state_plan_v2: Arc::new(state_plan_v2),
            allocation_plan: Arc::new(allocation_plan),
            arenas,
            tensor_state,
            non_paged_physical_bytes,
            maximum_sequence_tokens: AtomicU64::new(0),
        });
        self.models.insert(
            model_instance,
            ManagedKvModelState {
                closing: false,
                contract: contract.clone(),
                runtime: runtime.clone(),
                coordinators,
                prefix_indexes,
                pending_prefixes: HashMap::new(),
                pending_promotions: HashMap::new(),
                exact_target_transactions: HashSet::new(),
                registered_sessions: HashSet::new(),
                session_generations: HashMap::new(),
                capacity_claims: HashMap::new(),
                incremental_claim_sessions: HashSet::new(),
                tensor_sequences: HashMap::new(),
                tensor_snapshots,
                host_offload,
                resource_lease,
                materialized_resources,
                tensor_materialized_bytes: 0,
                tensor_rows_request_priced: false,
                allocation_ledger,
            },
        );
        self.next_arena_generation = first_arena_generation
            .checked_add(u32::try_from(runtime.plan.groups.len()).map_err(|_| {
                Error::InvalidInput("managed KV arena count exceeds u32".to_string())
            })?)
            .ok_or_else(|| Error::InvalidInput("managed KV arena generation overflow".into()))?;
        Ok(runtime)
    }

    /// Stop charging a model's lazy tensor rows as pending live claims, for
    /// a model whose requests reserve the rows they create (Qwen3-TTS prices
    /// its encoder memory in the prefill workspace, then commits it). The
    /// envelope stays authorized (planning still holds it); recording it as
    /// materialized means a row is counted once, by its request, and the
    /// rows no request has created never refuse a request whose own need
    /// fits live memory. Idempotent.
    pub(crate) fn price_tensor_rows_with_requests(
        &mut self,
        model_instance: ModelInstanceId,
    ) -> Result<()> {
        let state = self.models.get_mut(&model_instance).ok_or_else(|| {
            Error::InvalidInput("managed tensor rows target an unbound model".into())
        })?;
        if state.tensor_rows_request_priced {
            return Ok(());
        }
        let Some(arena) = state.runtime.tensor_state() else {
            return Err(Error::InvalidInput(
                "request-priced tensor rows need a model with tensor state".into(),
            ));
        };
        let envelope = arena.capacity().authorized_bytes();
        if let Some(lease) = state.resource_lease.as_ref() {
            lease.record_materialized_usage(state.materialized_resources.checked_add(
                tensor_state_resources(state.runtime.plan.backend, envelope)?,
            )?)?;
        }
        state.tensor_materialized_bytes = envelope;
        state.tensor_rows_request_priced = true;
        Ok(())
    }

    /// Resolve an arena runtime that was allocated by model loading. Request
    /// admission must never create backing storage or expand model residency.
    #[cfg(test)]
    pub(crate) fn require_loaded_runtime(
        &self,
        model_instance: ModelInstanceId,
        backend: BackendKind,
        capability: &InferenceStateCapability,
    ) -> Result<Option<Arc<ManagedKvModelRuntime>>> {
        let Some(contract) = capability.managed_contract() else {
            return Ok(None);
        };
        let state = self.models.get(&model_instance).ok_or_else(|| {
            Error::InferenceError(
                "loaded adapter published managed KV without load-time physical allocation"
                    .to_string(),
            )
        })?;
        if state.closing {
            return Err(Error::InferenceError(
                "managed KV runtime is closing".to_string(),
            ));
        }
        if backend != self.worker_backend
            || state.runtime.plan.backend != backend
            || &state.contract != contract
        {
            return Err(Error::InferenceError(
                "load-time managed KV runtime does not match the request capability".to_string(),
            ));
        }
        Ok(Some(state.runtime.clone()))
    }
}
