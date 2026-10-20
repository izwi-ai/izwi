//! Managed state capacity planning: portable and CUDA-resident token reach,
//! row fitting and the resources a managed state needs.

use super::*;

/// `prefix_snapshot_rows` adds committed prefix snapshots to the lazily
/// materialized tensor envelope (see
/// [`ManagedKvCacheManager::prefix_snapshot_rows_for`]), so the allocation
/// plan, the lease authorization and the CUDA fit all price them.
pub(super) fn plan_managed_state_capacity_with_policy(
    state_plan: &ResolvedStatePlan,
    model_instance: ModelInstanceId,
    request: ManagedStateCapacityRequest,
    materialize_cuda_paged_at_load: bool,
    prefix_snapshot_rows: u32,
) -> Result<(StateRuntimeAllocationPlan, Option<TensorStateCapacity>)> {
    if request.total_paged_pages == 0
        || request.retained_sequence_rows == 0
        || request.staged_transaction_rows == 0
    {
        return Err(Error::InvalidInput(
            "managed state capacity requires non-zero page, sequence, and transaction limits"
                .into(),
        ));
    }
    if request.staged_transaction_rows > request.retained_sequence_rows {
        return Err(Error::InvalidInput(
            "managed state transaction rows cannot exceed retained sequence rows".into(),
        ));
    }
    if state_plan.paged_attention.is_empty() {
        return Err(Error::InvalidInput(
            "managed state capacity requires a paged-attention anchor".into(),
        ));
    }
    let page_sizes = state_plan
        .paged_attention
        .iter()
        .map(|group| group.page_tokens)
        .collect::<Vec<_>>();
    if usize::try_from(request.total_paged_pages).unwrap_or(usize::MAX) < page_sizes.len() {
        return Err(Error::Backpressure(
            "managed KV page budget cannot back every paged state group".into(),
        ));
    }
    let upper = u64::from(request.total_paged_pages)
        .checked_mul(u64::from(
            *page_sizes.iter().max().expect("non-empty page sizes"),
        ))
        .ok_or_else(|| Error::InvalidInput("managed KV token reach overflow".into()))?;
    let required_pages = |tokens: u64| -> Result<u64> {
        page_sizes.iter().try_fold(0_u64, |total, page_tokens| {
            let pages = tokens.div_ceil(u64::from(*page_tokens));
            total
                .checked_add(pages)
                .ok_or_else(|| Error::InvalidInput("managed KV page demand overflow".into()))
        })
    };
    let token_reach = match request.logical_token_reach {
        Some(tokens) if tokens > 0 => tokens,
        Some(_) => {
            return Err(Error::InvalidInput(
                "managed state logical token reach must be non-zero".into(),
            ));
        }
        None => {
            let (mut low, mut high) = (1_u64, upper);
            while low < high {
                let middle = low + (high - low).div_ceil(2);
                if required_pages(middle)? <= u64::from(request.total_paged_pages) {
                    low = middle;
                } else {
                    high = middle - 1;
                }
            }
            low
        }
    };
    let mut groups =
        Vec::with_capacity(state_plan.paged_attention.len() + state_plan.non_paged.len());
    let mut paged_sequence_capacity = u32::MAX;
    for group in &state_plan.paged_attention {
        let blocks = u32::try_from(token_reach.div_ceil(u64::from(group.page_tokens)))
            .map_err(|_| Error::InvalidInput("managed KV group capacity exceeds u32".into()))?;
        // One retained sequence needs at least one page in every paged group,
        // but a page is not a sequence slot. In particular, the thousands of
        // pages needed for one long-context request must never multiply the
        // per-sequence recurrent/convolution state authorization.
        paged_sequence_capacity = paged_sequence_capacity.min(blocks);
        let strategy = managed_paged_capacity_strategy(
            state_plan.backend,
            blocks,
            materialize_cuda_paged_at_load,
        );
        groups.push(GroupCapacityRequest {
            group: group.group,
            domain: group.domain,
            strategy,
        });
    }
    let sequence_capacity = request.retained_sequence_rows.min(paged_sequence_capacity);
    let transaction_capacity = request.staged_transaction_rows.min(sequence_capacity);
    let snapshot_capacity = if state_plan.non_paged.is_empty() {
        0
    } else {
        prefix_snapshot_rows
    };
    let lazy_blocks = sequence_capacity
        .checked_add(transaction_capacity)
        .and_then(|blocks| blocks.checked_add(snapshot_capacity))
        .ok_or_else(|| Error::InvalidInput("managed tensor state capacity overflow".into()))?;
    for domain in &state_plan.non_paged {
        groups.push(GroupCapacityRequest {
            group: domain.group(),
            domain: domain.domain(),
            strategy: CapacityStrategy::BoundedLazy {
                max_blocks: lazy_blocks,
            },
        });
    }
    groups.sort_unstable_by_key(|group| (group.group, group.domain));
    let workspace = WorkspaceContract {
        fixed_bytes: 0,
        dimensions: vec![],
        terms: vec![],
        placement: match state_plan.backend {
            BackendKind::Cpu => WorkspacePlacement::Host,
            BackendKind::Metal | BackendKind::Cuda => WorkspacePlacement::BackendLocal,
        },
        concurrency_slots: transaction_capacity,
    };
    let registry = StateBackendRegistry::new(state_plan.backend, state_plan.device_ordinal)?;
    let allocation_plan = StateRuntimeAllocationPlan::build_exact(
        state_plan,
        model_instance,
        groups,
        workspace,
        &registry,
    )?;
    let tensor_capacity = (!state_plan.non_paged.is_empty())
        .then(|| {
            TensorStateCapacity::for_plan(state_plan, sequence_capacity, transaction_capacity)?
                .with_snapshot_capacity(snapshot_capacity)
        })
        .transpose()?;
    if let Some(capacity) = tensor_capacity {
        let planned_authorization = capacity
            .per_sequence_bytes()
            .checked_mul(u64::from(lazy_blocks))
            .ok_or_else(|| Error::InvalidInput("tensor authorization overflow".into()))?;
        if capacity.authorized_bytes() != planned_authorization {
            return Err(Error::InferenceError(
                "tensor arena authorization diverges from the allocation plan".into(),
            ));
        }
    }
    Ok((allocation_plan, tensor_capacity))
}

pub(super) fn managed_paged_capacity_strategy(
    backend: BackendKind,
    blocks: u32,
    materialize_cuda_paged_at_load: bool,
) -> CapacityStrategy {
    if backend == BackendKind::Cuda && blocks > 64 && !materialize_cuda_paged_at_load {
        let growth = cuda_paged_growth_geometry(blocks);
        CapacityStrategy::AdmissionGrowable {
            initial_blocks: growth.initial_pages,
            growth_quantum: growth.growth_quantum_pages,
            max_blocks: blocks,
        }
    } else {
        CapacityStrategy::Fixed { blocks }
    }
}

/// Share of a KV arena, allocated in full at load by the CUDA resident fit,
/// that an automatic prefix page budget may hold: one page in this many.
pub(super) const AUTOMATIC_PREFIX_ARENA_SHARE: usize = 4;

/// Page budget of one arena's committed-prefix index.
///
/// `budget` is the prefix page budget the cache policy resolved: the
/// operator's value, or [`crate::config::DEFAULT_MAX_PREFIX_CACHE_PAGES`]
/// when `automatic`, and zero when prefix reuse is disabled. An operator's
/// value applies as set, and so does an automatic budget on CPU, on Metal and
/// on CUDA arenas that grow on demand. On a CUDA arena allocated in full at
/// load (`materialized_at_load`), an automatic budget rises to a quarter of
/// the arena. Those pages occupy device memory whether or not the index holds
/// them, and a request that needs pages held only by the index evicts them,
/// least recently used first (`evict_lru_excluding` in `prepare_inner`), so
/// the larger budget only keeps more chat history reusable: a quarter of
/// Qwen3.8's ~9,700-page arena on an H100 is ~2,400 pages, over 150k
/// tokens. A budget never exceeds its arena.
pub(super) fn prefix_index_pages(
    budget: usize,
    automatic: bool,
    backend: BackendKind,
    materialized_at_load: bool,
    arena_pages: u32,
) -> usize {
    let arena_pages = usize::try_from(arena_pages).unwrap_or(usize::MAX);
    let scales = automatic && backend == BackendKind::Cuda && materialized_at_load;
    let budget = if budget > 0 && scales {
        budget.max(arena_pages / AUTOMATIC_PREFIX_ARENA_SHARE)
    } else {
        budget
    };
    budget.min(arena_pages)
}

/// Maximum simultaneously live page rows for the CUDA arena's current
/// doubling/quantized contiguous replacement schedule.
pub(super) fn cuda_contiguous_replacement_peak_pages(maximum_pages: u32) -> Result<u64> {
    if maximum_pages == 0 {
        return Err(Error::InvalidInput(
            "CUDA KV replacement peak requires non-zero pages".into(),
        ));
    }
    let geometry = cuda_paged_growth_geometry(maximum_pages);
    let mut current = geometry.initial_pages;
    let mut peak = u64::from(current);
    while current < maximum_pages {
        let quantum = geometry.growth_quantum_pages;
        let rounded_target = current.saturating_add(quantum).min(maximum_pages);
        let amortized_addition = current
            .max(quantum)
            .div_ceil(quantum)
            .saturating_mul(quantum);
        let doubled = current
            .saturating_add(amortized_addition)
            .min(maximum_pages);
        let target = rounded_target.max(doubled).min(maximum_pages);
        peak = peak.max(u64::from(current) + u64::from(target));
        if target <= current {
            return Err(Error::InferenceError(
                "CUDA KV replacement growth made no progress".into(),
            ));
        }
        current = target;
    }
    Ok(peak)
}

pub(super) fn cuda_largest_contiguous_replacement_extra(
    paged: impl IntoIterator<Item = (u32, u64)>,
) -> Result<u64> {
    paged
        .into_iter()
        .try_fold(0_u64, |largest_extra, (maximum_pages, bytes_per_page)| {
            let peak_pages = cuda_contiguous_replacement_peak_pages(maximum_pages)?;
            let extra_pages = peak_pages
                .checked_sub(u64::from(maximum_pages))
                .ok_or_else(|| {
                    Error::InferenceError("CUDA KV replacement peak underflow".into())
                })?;
            let extra_bytes = bytes_per_page.checked_mul(extra_pages).ok_or_else(|| {
                Error::ModelLoadError("CUDA KV replacement byte plan overflow".into())
            })?;
            Ok(largest_extra.max(extra_bytes))
        })
}

pub(super) fn cuda_managed_state_peak_authorization(
    steady_resources: ResourceVector,
    groups: &[crate::kv::ResolvedKvGroup],
) -> Result<ResourceVector> {
    let ResourceAmount::Known(steady_device_bytes) = steady_resources.device_bytes else {
        return Err(Error::ModelLoadError(
            "CUDA managed state requires known steady device bytes".into(),
        ));
    };
    let replacement_extra =
        cuda_largest_contiguous_replacement_extra(groups.iter().filter_map(|group| {
            match group.capacity_strategy {
                CapacityStrategy::AdmissionGrowable { .. } => {
                    Some((group.capacity_pages, group.bytes_per_page))
                }
                CapacityStrategy::Fixed { .. }
                | CapacityStrategy::BoundedLazy { .. }
                | CapacityStrategy::Reserved { .. } => None,
            }
        }))?;
    Ok(ResourceVector {
        device_bytes: ResourceAmount::Known(
            steady_device_bytes
                .checked_add(replacement_extra)
                .ok_or_else(|| Error::ModelLoadError("CUDA state peak overflow".into()))?,
        ),
        ..steady_resources
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CudaContiguousPagedGeometry {
    pub(super) page_tokens: u32,
    pub(super) bytes_per_page: u64,
}

/// Choose retained rows and the logical reach for one resolved plan. Only
/// the plan's geometry is read, so CPU tests can drive the CUDA policy.
///
/// `prefix_snapshot_rows` committed prefix snapshots are fixed state: they
/// count against the automatic rows' state share (so automatic retained rows
/// shrink by about that many, and KV is unchanged), and against the whole
/// budget beside explicit rows.
#[allow(clippy::too_many_arguments)]
pub(super) fn fit_resident_rows_and_reach(
    state_plan: &ResolvedStatePlan,
    model_instance: ModelInstanceId,
    maximum_tokens_for_rows: impl Fn(u32) -> Result<u64>,
    headroom_bytes: u64,
    safety_reserve_bytes: u64,
    page_tokens_hint: u32,
    rows: CudaResidentRowRequest,
    prefix_snapshot_rows: u32,
) -> Result<CudaResidentFit> {
    let budget_bytes = headroom_bytes.saturating_sub(safety_reserve_bytes);
    let geometry = |retained_sequence_rows: u32| {
        cuda_resident_geometry(
            state_plan,
            model_instance,
            maximum_tokens_for_rows(retained_sequence_rows)?,
            page_tokens_hint,
            retained_sequence_rows,
            rows.staged_transaction_rows.min(retained_sequence_rows),
            prefix_snapshot_rows,
        )
    };
    let chosen = match rows.automatic_state_budget_fraction {
        None => {
            let explicit = geometry(rows.retained_sequence_rows)?;
            if explicit.minimum_bytes > budget_bytes {
                tracing::warn!(
                    "CUDA managed context cannot fit the model-authored minimum: {}",
                    explicit.describe(headroom_bytes, safety_reserve_bytes)
                );
                return Err(Error::ModelLoadError(format!(
                    "{}; {CUDA_EXPLICIT_ROWS_GUIDANCE}",
                    crate::engine::resources::capacity_shortfall_reason(
                        "CUDA managed context cannot fit the model-authored minimum",
                        explicit.minimum_bytes,
                        budget_bytes,
                    )
                )));
            }
            explicit
        }
        Some(fraction) => {
            let floor = geometry(
                rows.staged_transaction_rows
                    .min(rows.retained_sequence_rows),
            )?;
            if floor.minimum_bytes > budget_bytes {
                tracing::warn!(
                    "CUDA managed context cannot fit the model-authored minimum at the decode width: {}",
                    floor.describe(headroom_bytes, safety_reserve_bytes)
                );
                return Err(Error::ModelLoadError(format!(
                    "{}; {CUDA_AUTOMATIC_ROWS_GUIDANCE}",
                    crate::engine::resources::capacity_shortfall_reason(
                        "CUDA managed context cannot fit the model-authored minimum at the decode width",
                        floor.minimum_bytes,
                        budget_bytes,
                    )
                )));
            }
            let state_cap = (budget_bytes as f64 * fraction) as u64;
            let fits = |candidate: &CudaResidentGeometry| {
                candidate.resident_state_bytes <= state_cap
                    && candidate.minimum_bytes <= budget_bytes
            };
            // The floor is the decode width: it may exceed the share but
            // must stay usable. Above it, take the most rows that fit.
            let mut chosen = floor;
            if fits(&floor) {
                let (mut low, mut high) =
                    (floor.retained_sequence_rows, rows.retained_sequence_rows);
                while low < high {
                    let middle = low + (high - low).div_ceil(2);
                    let candidate = geometry(middle)?;
                    if fits(&candidate) {
                        low = middle;
                        chosen = candidate;
                    } else {
                        high = middle - 1;
                    }
                }
            }
            chosen
        }
    };
    let paged = state_plan
        .paged_attention
        .iter()
        .map(|paged| CudaContiguousPagedGeometry {
            page_tokens: paged.page_tokens,
            bytes_per_page: paged.bytes_per_page,
        })
        .collect::<Vec<_>>();
    Ok(CudaResidentFit {
        logical_token_reach: fit_cuda_resident_token_reach(
            chosen.maximum_tokens,
            page_tokens_hint,
            &paged,
            chosen.resident_state_bytes,
            budget_bytes,
        )?,
        retained_sequence_rows: chosen.retained_sequence_rows,
        staged_transaction_rows: chosen.staged_transaction_rows,
        prefix_snapshot_rows: chosen.prefix_snapshot_rows,
        per_sequence_state_bytes: chosen.per_sequence_state_bytes,
        resident_state_bytes: chosen.resident_state_bytes,
    })
}

/// Fixed state and minimum resident footprint of one candidate row count.
#[derive(Debug, Clone, Copy)]
pub(super) struct CudaResidentGeometry {
    pub(super) retained_sequence_rows: u32,
    pub(super) staged_transaction_rows: u32,
    pub(super) prefix_snapshot_rows: u32,
    pub(super) maximum_tokens: u64,
    pub(super) per_sequence_state_bytes: u64,
    pub(super) resident_state_bytes: u64,
    pub(super) minimum_tokens: u64,
    pub(super) minimum_bytes: u64,
}

impl CudaResidentGeometry {
    pub(super) fn describe(&self, headroom_bytes: u64, safety_reserve_bytes: u64) -> String {
        format!(
            "retained_sequences={}, staged_transactions={}, prefix_snapshots={}, per_sequence_state_bytes={}, resident_state_bytes={}, minimum_tokens={}, minimum_bytes={}, planning_headroom={headroom_bytes}, safety_reserve={safety_reserve_bytes}",
            self.retained_sequence_rows,
            self.staged_transaction_rows,
            self.prefix_snapshot_rows,
            self.per_sequence_state_bytes,
            self.resident_state_bytes,
            self.minimum_tokens,
            self.minimum_bytes,
        )
    }
}

/// Price one row count: the fixed (non-paged) state the allocation plan
/// authorizes for retained plus staged rows, and that state plus one page per
/// paged domain. Only plan geometry is read, so a CPU-negotiated plan of the
/// same contract prices the same bytes in tests.
pub(super) fn cuda_resident_geometry(
    state_plan: &ResolvedStatePlan,
    model_instance: ModelInstanceId,
    maximum_tokens: u64,
    page_tokens: u32,
    retained_sequence_rows: u32,
    staged_transaction_rows: u32,
    prefix_snapshot_rows: u32,
) -> Result<CudaResidentGeometry> {
    let (maximum_allocation, tensor_capacity) = plan_managed_state_capacity_with_policy(
        state_plan,
        model_instance,
        ManagedStateCapacityRequest {
            total_paged_pages: u32::MAX,
            logical_token_reach: Some(maximum_tokens),
            retained_sequence_rows,
            staged_transaction_rows,
        },
        false,
        prefix_snapshot_rows,
    )?;
    let paged = state_plan
        .paged_attention
        .iter()
        .map(|paged| CudaContiguousPagedGeometry {
            page_tokens: paged.page_tokens,
            bytes_per_page: paged.bytes_per_page,
        })
        .collect::<Vec<_>>();
    let steady_resources = managed_state_resources(
        state_plan.backend,
        maximum_allocation.maximum_resources(state_plan)?,
    )?;
    let steady_bytes = match state_plan.backend {
        BackendKind::Cuda => known_resource_bytes(steady_resources.device_bytes, "device"),
        BackendKind::Metal => known_resource_bytes(steady_resources.unified_bytes, "unified"),
        BackendKind::Cpu => known_resource_bytes(steady_resources.host_bytes, "host"),
    }?;
    let maximum_paged_bytes =
        state_plan
            .paged_attention
            .iter()
            .try_fold(0_u64, |total, paged| {
                let maximum_blocks = maximum_allocation
                    .group_capacity(paged.group, paged.domain)?
                    .strategy
                    .maximum_blocks();
                total
                    .checked_add(
                        paged
                            .bytes_per_page
                            .checked_mul(u64::from(maximum_blocks))
                            .ok_or_else(|| {
                                Error::ModelLoadError("CUDA KV byte plan overflow".into())
                            })?,
                    )
                    .ok_or_else(|| Error::ModelLoadError("CUDA KV byte plan overflow".into()))
            })?;
    let resident_state_bytes = steady_bytes
        .checked_sub(maximum_paged_bytes)
        .ok_or_else(|| Error::ModelLoadError("CUDA fixed state byte plan underflow".into()))?;
    let minimum_tokens = u64::from(page_tokens).min(maximum_tokens);
    Ok(CudaResidentGeometry {
        retained_sequence_rows,
        staged_transaction_rows,
        prefix_snapshot_rows: tensor_capacity.map_or(0, |capacity| capacity.snapshot_capacity()),
        maximum_tokens,
        per_sequence_state_bytes: tensor_capacity
            .map_or(0, |capacity| capacity.per_sequence_bytes()),
        resident_state_bytes,
        minimum_tokens,
        minimum_bytes: cuda_resident_required_bytes(minimum_tokens, &paged, resident_state_bytes)?,
    })
}

/// Price the worst admission-barrier replacement across independently grown
/// paged arenas. Every arena may already be at its final steady backing when
/// another arena grows, but only the arena currently being replaced retains an
/// additional old backing alongside its target allocation.
#[cfg(test)]
pub(super) fn cuda_contiguous_replacement_required_bytes(
    tokens: u64,
    paged: &[CudaContiguousPagedGeometry],
    fixed_state_bytes: u64,
) -> Result<u64> {
    if paged.is_empty() {
        return Err(Error::InvalidInput(
            "CUDA contiguous context fitting requires paged state".into(),
        ));
    }
    let (steady_paged_bytes, largest_replacement_extra) =
        paged
            .iter()
            .try_fold((0_u64, 0_u64), |(steady_total, largest_extra), geometry| {
                if geometry.page_tokens == 0 || geometry.bytes_per_page == 0 {
                    return Err(Error::InvalidInput(
                        "CUDA contiguous context fitting requires non-zero paged geometry".into(),
                    ));
                }
                let pages = u32::try_from(tokens.div_ceil(u64::from(geometry.page_tokens)))
                    .map_err(|_| Error::ModelLoadError("CUDA KV page demand exceeds u32".into()))?;
                let peak_pages = cuda_contiguous_replacement_peak_pages(pages)?;
                let steady_bytes = geometry
                    .bytes_per_page
                    .checked_mul(u64::from(pages))
                    .ok_or_else(|| Error::ModelLoadError("CUDA KV byte plan overflow".into()))?;
                let peak_bytes = geometry
                    .bytes_per_page
                    .checked_mul(peak_pages)
                    .ok_or_else(|| Error::ModelLoadError("CUDA KV byte plan overflow".into()))?;
                let replacement_extra = peak_bytes.checked_sub(steady_bytes).ok_or_else(|| {
                    Error::ModelLoadError("CUDA KV replacement peak underflow".into())
                })?;
                Ok((
                    steady_total.checked_add(steady_bytes).ok_or_else(|| {
                        Error::ModelLoadError("CUDA state peak byte plan overflow".into())
                    })?,
                    largest_extra.max(replacement_extra),
                ))
            })?;
    fixed_state_bytes
        .checked_add(steady_paged_bytes)
        .and_then(|bytes| bytes.checked_add(largest_replacement_extra))
        .ok_or_else(|| Error::ModelLoadError("CUDA state peak byte plan overflow".into()))
}

pub(super) fn cuda_resident_required_bytes(
    tokens: u64,
    paged: &[CudaContiguousPagedGeometry],
    fixed_state_bytes: u64,
) -> Result<u64> {
    if paged.is_empty() {
        return Err(Error::InvalidInput(
            "CUDA resident context fitting requires paged state".into(),
        ));
    }
    paged.iter().try_fold(fixed_state_bytes, |total, geometry| {
        if geometry.page_tokens == 0 || geometry.bytes_per_page == 0 {
            return Err(Error::InvalidInput(
                "CUDA resident context fitting requires non-zero paged geometry".into(),
            ));
        }
        let pages = tokens.div_ceil(u64::from(geometry.page_tokens));
        total
            .checked_add(
                geometry
                    .bytes_per_page
                    .checked_mul(pages)
                    .ok_or_else(|| Error::ModelLoadError("CUDA KV byte plan overflow".into()))?,
            )
            .ok_or_else(|| Error::ModelLoadError("CUDA state byte plan overflow".into()))
    })
}

pub(super) fn fit_cuda_resident_token_reach(
    maximum_tokens: u64,
    minimum_page_tokens: u32,
    paged: &[CudaContiguousPagedGeometry],
    fixed_state_bytes: u64,
    budget_bytes: u64,
) -> Result<u64> {
    let minimum_tokens = u64::from(minimum_page_tokens).min(maximum_tokens);
    if cuda_resident_required_bytes(maximum_tokens, paged, fixed_state_bytes)? <= budget_bytes {
        return Ok(maximum_tokens);
    }
    let (mut low, mut high) = (minimum_tokens, maximum_tokens);
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        if cuda_resident_required_bytes(middle, paged, fixed_state_bytes)? <= budget_bytes {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    Ok(low)
}

#[cfg(test)]
pub(super) fn fit_cuda_contiguous_token_reach(
    maximum_tokens: u64,
    minimum_page_tokens: u32,
    paged: &[CudaContiguousPagedGeometry],
    fixed_state_bytes: u64,
    budget_bytes: u64,
) -> Result<u64> {
    let minimum_tokens = u64::from(minimum_page_tokens).min(maximum_tokens);
    if cuda_contiguous_replacement_required_bytes(maximum_tokens, paged, fixed_state_bytes)?
        <= budget_bytes
    {
        return Ok(maximum_tokens);
    }
    let (mut low, mut high) = (minimum_tokens, maximum_tokens);
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        if cuda_contiguous_replacement_required_bytes(middle, paged, fixed_state_bytes)?
            <= budget_bytes
        {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    Ok(low)
}

pub(super) fn known_resource_bytes(amount: ResourceAmount, domain: &str) -> Result<u64> {
    match amount {
        ResourceAmount::Known(bytes) => Ok(bytes),
        ResourceAmount::Unknown => Err(Error::ModelLoadError(format!(
            "portable context planning has unknown {domain} capacity"
        ))),
    }
}

pub(super) fn managed_state_resources(
    backend: BackendKind,
    state: StateResourceVector,
) -> Result<ResourceVector> {
    let mut resources = ResourceVector::zero();
    let host = state
        .host_bytes
        .checked_add(state.pinned_bytes)
        .and_then(|bytes| bytes.checked_add(state.metadata_bytes))
        .ok_or_else(|| Error::Overloaded("managed state host byte total overflow".into()))?;
    match backend {
        BackendKind::Cpu => {
            let total = host
                .checked_add(state.device_bytes)
                .ok_or_else(|| Error::Overloaded("managed CPU state total overflow".into()))?;
            resources.host_bytes = ResourceAmount::Known(total);
        }
        BackendKind::Metal => {
            let total = host
                .checked_add(state.device_bytes)
                .ok_or_else(|| Error::Overloaded("managed Metal state total overflow".into()))?;
            resources.unified_bytes = ResourceAmount::Known(total);
        }
        BackendKind::Cuda => {
            resources.host_bytes = ResourceAmount::Known(host);
            resources.device_bytes = ResourceAmount::Known(state.device_bytes);
        }
    }
    Ok(resources)
}

/// Tensor-state bytes in the backend's lease domain (device on CUDA, unified
/// on Metal, host on CPU), as `managed_state_resources` prices the envelope.
pub(super) fn tensor_state_resources(backend: BackendKind, bytes: u64) -> Result<ResourceVector> {
    managed_state_resources(
        backend,
        StateResourceVector {
            device_bytes: bytes,
            ..StateResourceVector::default()
        },
    )
}

impl ManagedKvCacheManager {
    fn portable_state_plan(
        &self,
        contract: &InferenceStateContract,
        page_tokens_hint: u32,
    ) -> Result<ResolvedStatePlan> {
        negotiate_state_plan(
            contract,
            &StateBackendPlanRequest {
                backend: self.worker_backend,
                device_ordinal: self.worker_device_ordinal,
                page_tokens_hint: Some(page_tokens_hint),
                storage_dtype_hint: None,
            },
        )
    }

    /// What [`Self::fit_portable_logical_token_reach`] prices `state_plan`
    /// at `tokens` per sequence, `(retained, staged)` rows and `copies`.
    fn planned_portable_state_bytes(
        &self,
        state_plan: &ResolvedStatePlan,
        model_instance: ModelInstanceId,
        prefix_snapshot_rows: u32,
        tokens: u64,
        (retained_sequence_rows, staged_transaction_rows): (u32, u32),
        portable_state_copies: u32,
    ) -> Result<u64> {
        let (allocation, _) = plan_managed_state_capacity_with_policy(
            state_plan,
            model_instance,
            ManagedStateCapacityRequest {
                total_paged_pages: u32::MAX,
                logical_token_reach: Some(tokens),
                retained_sequence_rows,
                staged_transaction_rows,
            },
            false,
            prefix_snapshot_rows,
        )?;
        let resources = managed_state_resources(
            self.worker_backend,
            allocation.maximum_resources(state_plan)?,
        )?;
        let bytes = match self.worker_backend {
            BackendKind::Cpu => known_resource_bytes(resources.host_bytes, "host"),
            BackendKind::Metal => known_resource_bytes(resources.unified_bytes, "unified"),
            BackendKind::Cuda => {
                return Err(Error::InvalidInput(
                    "CUDA managed state is not priced by the portable fit".into(),
                ))
            }
        }?;
        bytes
            .checked_mul(u64::from(portable_state_copies))
            .ok_or_else(|| {
                Error::ModelLoadError("aggregate managed-state byte plan overflow".into())
            })
    }

    /// The bytes `contract`'s portable managed state takes at `tokens` per
    /// sequence, priced exactly as the portable fit prices it, so a pre-load
    /// floor and the fit after the load agree. `None` on CUDA, which does
    /// not fit portable state.
    pub(crate) fn portable_state_bytes(
        &self,
        contract: &InferenceStateContract,
        tokens: u64,
        page_tokens_hint: usize,
        rows: (u32, u32),
        portable_state_copies: u32,
    ) -> Result<Option<u64>> {
        if self.worker_backend == BackendKind::Cuda {
            return Ok(None);
        }
        let page_tokens_hint = u32::try_from(page_tokens_hint)
            .map_err(|_| Error::InvalidInput("managed KV page size exceeds u32".into()))?;
        let state_plan = self.portable_state_plan(contract, page_tokens_hint)?;
        // The page minimum, as the fit floors its minimum.
        let tokens = u64::from(page_tokens_hint).max(tokens);
        // The instance only labels the plan; no arena is bound for it.
        self.planned_portable_state_bytes(
            &state_plan,
            ModelInstanceId::new(u64::MAX),
            self.prefix_snapshot_rows_for(contract),
            tokens,
            rows,
            portable_state_copies,
        )
        .map(Some)
    }

    /// Resolve the largest portable logical reach whose exact managed-state
    /// plan fits the stable shared-memory envelope. CUDA deliberately bypasses
    /// this path and keeps admission-growable arenas.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fit_portable_logical_token_reach(
        &self,
        model_instance: ModelInstanceId,
        contract: &InferenceStateContract,
        maximum_tokens: u64,
        safety_reserve_bytes: u64,
        page_tokens_hint: usize,
        retained_sequence_rows: u32,
        staged_transaction_rows: u32,
        portable_state_copies: u32,
        headroom: PortableFitHeadroom,
    ) -> Result<u64> {
        if self.worker_backend == BackendKind::Cuda || maximum_tokens == 0 {
            return Ok(maximum_tokens);
        }
        if portable_state_copies == 0 {
            return Err(Error::InvalidInput(
                "portable managed-state copy count must be positive".into(),
            ));
        }
        let Some(authority) = self.resource_authority.as_ref() else {
            return Ok(maximum_tokens.min(4_096));
        };
        let planning = authority.planning_headroom_bytes(self.worker_backend)?;
        let (growth_headroom, safety_reserve_bytes) = match headroom {
            PortableFitHeadroom::Planning { .. } => (planning, safety_reserve_bytes),
            // Retained state keeps the one reserve its invocation fit keeps
            // too: the configured floor or, for an observed load, the
            // measured untracked growth (ADM-11).
            PortableFitHeadroom::Live { reserve, .. } => (
                authority.state_fit_headroom_bytes(self.worker_backend)?,
                match reserve {
                    StateFitReserve::MeasuredUntracked => authority
                        .state_safety_reserve_bytes(self.worker_backend, safety_reserve_bytes),
                    StateFitReserve::Configured => safety_reserve_bytes,
                },
            ),
        };
        let ResourceAmount::Known(headroom_bytes) = planning else {
            return Ok(maximum_tokens.min(4_096));
        };
        let later_minimum_bytes = match headroom {
            PortableFitHeadroom::Live {
                later_minimum_bytes,
                ..
            } => later_minimum_bytes,
            PortableFitHeadroom::Planning { .. } => 0,
        };
        // The model-authored minimum is admitted against planning headroom;
        // live memory caps only growth above it. Right after a large load the
        // live sample undercounts what the OS reclaims on demand (clean
        // file-backed pages of the checkpoint just read, compressible
        // pages), so a live floor refused even a one-page state. Both leave
        // the later groups' minimums free beside the reserve.
        let held_bytes = safety_reserve_bytes.saturating_add(later_minimum_bytes);
        let floor_budget_bytes = headroom_bytes.saturating_sub(held_bytes);
        let growth_budget_bytes = match growth_headroom {
            ResourceAmount::Known(bytes) => bytes.saturating_sub(held_bytes),
            ResourceAmount::Unknown => floor_budget_bytes,
        }
        .min(floor_budget_bytes);
        let page_tokens_hint = u32::try_from(page_tokens_hint)
            .map_err(|_| Error::InvalidInput("managed KV page size exceeds u32".into()))?;
        let state_plan = self.portable_state_plan(contract, page_tokens_hint)?;
        let prefix_snapshot_rows = self.prefix_snapshot_rows_for(contract);
        let required_bytes = |tokens: u64| -> Result<u64> {
            self.planned_portable_state_bytes(
                &state_plan,
                model_instance,
                prefix_snapshot_rows,
                tokens,
                (retained_sequence_rows, staged_transaction_rows),
                portable_state_copies,
            )
        };

        let floor_tokens = match headroom {
            PortableFitHeadroom::Live { minimum_tokens, .. }
            | PortableFitHeadroom::Planning { minimum_tokens } => minimum_tokens,
        };
        let minimum_tokens = u64::from(page_tokens_hint)
            .max(floor_tokens)
            .min(maximum_tokens);
        let minimum_bytes = required_bytes(minimum_tokens)?;
        if minimum_bytes > floor_budget_bytes {
            tracing::warn!(
                "portable context cannot fit the model-authored minimum: minimum_tokens={minimum_tokens}, state_bytes={minimum_bytes}, later_minimum_bytes={later_minimum_bytes}, planning_headroom={headroom_bytes}, safety_reserve={safety_reserve_bytes}, shortfall_bytes={}",
                minimum_bytes - floor_budget_bytes
            );
            let subject = if later_minimum_bytes > 0 {
                format!(
                    "not enough memory for this model's minimum {minimum_tokens}-token state and the working state loaded after it"
                )
            } else {
                format!("not enough memory for this model's minimum {minimum_tokens}-token state")
            };
            let message = crate::engine::resources::state_shortfall_reason(
                subject,
                minimum_bytes.saturating_add(later_minimum_bytes),
                headroom_bytes,
                safety_reserve_bytes,
            );
            // A model-authored floor (ASR, TTS, chat) that headroom cannot
            // hold is a capacity refusal (503), not a load defect.
            return Err(match headroom {
                PortableFitHeadroom::Planning { minimum_tokens: 0 } => {
                    Error::ModelLoadError(message)
                }
                _ => Error::Overloaded(message),
            });
        }
        let budget_bytes = growth_budget_bytes.max(minimum_bytes);
        let selected = if required_bytes(maximum_tokens)? <= budget_bytes {
            maximum_tokens
        } else {
            let (mut low, mut high) = (minimum_tokens, maximum_tokens);
            while low < high {
                let middle = low + (high - low).div_ceil(2);
                if required_bytes(middle)? <= budget_bytes {
                    low = middle;
                } else {
                    high = middle - 1;
                }
            }
            low
        };
        tracing::info!(
            model_instance = ?model_instance,
            selected_tokens = selected,
            minimum_tokens,
            maximum_tokens,
            state_bytes = required_bytes(selected)?,
            growth_budget_bytes = budget_bytes,
            retained_sequence_rows,
            staged_transaction_rows,
            portable_state_copies,
            "Fitted portable retained state"
        );
        Ok(selected)
    }

    /// Fit a CUDA logical reach whose complete paged backing is materialized
    /// before Ready publication. This turns the live-capacity observation into
    /// physical ownership instead of a revocable promise to grow later.
    ///
    /// Explicit rows are a hard contract. Automatic retained rows shrink to
    /// the largest count whose recurrent state fits the configured share of
    /// the budget, never below the staged (decode) width; the rest of the
    /// budget backs KV pages.
    pub(crate) fn fit_cuda_resident_logical_token_reach(
        &self,
        model_instance: ModelInstanceId,
        contract: &InferenceStateContract,
        maximum_tokens_for_rows: impl Fn(u32) -> Result<u64>,
        safety_reserve_bytes: u64,
        page_tokens_hint: usize,
        rows: CudaResidentRowRequest,
    ) -> Result<CudaResidentFit> {
        let requested = |logical_token_reach| CudaResidentFit {
            logical_token_reach,
            retained_sequence_rows: rows.retained_sequence_rows,
            staged_transaction_rows: rows.staged_transaction_rows,
            prefix_snapshot_rows: 0,
            per_sequence_state_bytes: 0,
            resident_state_bytes: 0,
        };
        let maximum_tokens = maximum_tokens_for_rows(rows.retained_sequence_rows)?;
        if self.worker_backend != BackendKind::Cuda || maximum_tokens == 0 {
            return Ok(requested(maximum_tokens));
        }
        let Some(authority) = self.resource_authority.as_ref() else {
            return Err(Error::ModelLoadError(
                "CUDA managed context fitting requires a resource authority".into(),
            ));
        };
        let ResourceAmount::Known(headroom_bytes) =
            authority.planning_headroom_bytes(BackendKind::Cuda)?
        else {
            return Err(Error::ModelLoadError(
                "CUDA managed context fitting requires known device capacity".into(),
            ));
        };
        let page_tokens_hint = u32::try_from(page_tokens_hint)
            .map_err(|_| Error::InvalidInput("managed KV page size exceeds u32".into()))?;
        let state_plan = negotiate_state_plan(
            contract,
            &StateBackendPlanRequest {
                backend: BackendKind::Cuda,
                device_ordinal: self.worker_device_ordinal,
                page_tokens_hint: Some(page_tokens_hint),
                storage_dtype_hint: None,
            },
        )?;
        fit_resident_rows_and_reach(
            &state_plan,
            model_instance,
            maximum_tokens_for_rows,
            headroom_bytes,
            safety_reserve_bytes,
            page_tokens_hint,
            rows,
            self.prefix_snapshot_rows_for(contract),
        )
    }

    /// Device memory the worker's resource authority reports in total.
    pub(crate) fn device_capacity(&self) -> ResourceAmount {
        self.resource_authority
            .as_ref()
            .map_or(ResourceAmount::Unknown, |authority| {
                authority.snapshot().physical.capacity.device_bytes
            })
    }
}
