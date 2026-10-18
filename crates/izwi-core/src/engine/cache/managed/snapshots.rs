//! Managed KV telemetry and runtime snapshots.

use super::*;

impl ManagedKvOperationSnapshot {
    pub(super) fn add_assign(&mut self, other: Self) {
        self.slot_write_dispatches = self
            .slot_write_dispatches
            .saturating_add(other.slot_write_dispatches);
        self.paged_decode_dispatches = self
            .paged_decode_dispatches
            .saturating_add(other.paged_decode_dispatches);
        self.paged_prefill_dispatches = self
            .paged_prefill_dispatches
            .saturating_add(other.paged_prefill_dispatches);
        self.page_zero_dispatches = self
            .page_zero_dispatches
            .saturating_add(other.page_zero_dispatches);
        self.page_copy_dispatches = self
            .page_copy_dispatches
            .saturating_add(other.page_copy_dispatches);
        self.backing_allocations = self
            .backing_allocations
            .saturating_add(other.backing_allocations);
        self.backing_allocations_observed_arenas = self
            .backing_allocations_observed_arenas
            .saturating_add(other.backing_allocations_observed_arenas);
        self.workspace_bytes = self.workspace_bytes.saturating_add(other.workspace_bytes);
        self.workspace_bytes_observed_arenas = self
            .workspace_bytes_observed_arenas
            .saturating_add(other.workspace_bytes_observed_arenas);
        self.workspace_budget_bytes = self
            .workspace_budget_bytes
            .saturating_add(other.workspace_budget_bytes);
        self.workspace_budget_bytes_observed_arenas = self
            .workspace_budget_bytes_observed_arenas
            .saturating_add(other.workspace_budget_bytes_observed_arenas);
        self.workspace_high_water_bytes = self
            .workspace_high_water_bytes
            .saturating_add(other.workspace_high_water_bytes);
        self.workspace_high_water_bytes_observed_arenas = self
            .workspace_high_water_bytes_observed_arenas
            .saturating_add(other.workspace_high_water_bytes_observed_arenas);
        self.workspace_allocations = self
            .workspace_allocations
            .saturating_add(other.workspace_allocations);
        self.workspace_allocations_observed_arenas = self
            .workspace_allocations_observed_arenas
            .saturating_add(other.workspace_allocations_observed_arenas);
        self.host_synchronizations = self
            .host_synchronizations
            .saturating_add(other.host_synchronizations);
        self.cpu_reference_attention_dispatches = self
            .cpu_reference_attention_dispatches
            .saturating_add(other.cpu_reference_attention_dispatches);
        self.portable_attention_dispatches = self
            .portable_attention_dispatches
            .saturating_add(other.portable_attention_dispatches);
        self.cuda_native_attention_dispatches = self
            .cuda_native_attention_dispatches
            .saturating_add(other.cuda_native_attention_dispatches);
        self.cuda_flash_attention_dispatches = self
            .cuda_flash_attention_dispatches
            .saturating_add(other.cuda_flash_attention_dispatches);
        self.cuda_split_decode_dispatches = self
            .cuda_split_decode_dispatches
            .saturating_add(other.cuda_split_decode_dispatches);
        self.metal_native_attention_dispatches = self
            .metal_native_attention_dispatches
            .saturating_add(other.metal_native_attention_dispatches);
        self.cuda_graph_warmups = self
            .cuda_graph_warmups
            .saturating_add(other.cuda_graph_warmups);
        self.cuda_graph_captures = self
            .cuda_graph_captures
            .saturating_add(other.cuda_graph_captures);
        self.cuda_graph_replays = self
            .cuda_graph_replays
            .saturating_add(other.cuda_graph_replays);
        self.cuda_graph_fallbacks = self
            .cuda_graph_fallbacks
            .saturating_add(other.cuda_graph_fallbacks);
        self.cuda_graph_backoff_hits = self
            .cuda_graph_backoff_hits
            .saturating_add(other.cuda_graph_backoff_hits);
        self.cuda_graph_evictions = self
            .cuda_graph_evictions
            .saturating_add(other.cuda_graph_evictions);
    }
}

pub(super) fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

pub(super) fn add_coordinator_stats(
    totals: &mut ManagedKvCoordinatorSnapshot,
    arena: &ManagedKvCoordinatorSnapshot,
) {
    totals.capacity_pages = totals.capacity_pages.saturating_add(arena.capacity_pages);
    totals.allocated_pages = totals.allocated_pages.saturating_add(arena.allocated_pages);
    totals.free_pages = totals.free_pages.saturating_add(arena.free_pages);
    totals.admission_claimed_pages = totals
        .admission_claimed_pages
        .saturating_add(arena.admission_claimed_pages);
    totals.admission_available_pages = totals
        .admission_available_pages
        .saturating_add(arena.admission_available_pages);
    totals.admission_claims = totals
        .admission_claims
        .saturating_add(arena.admission_claims);
    totals.table_refs = totals.table_refs.saturating_add(arena.table_refs);
    totals.prefix_refs = totals.prefix_refs.saturating_add(arena.prefix_refs);
    totals.execution_pins = totals.execution_pins.saturating_add(arena.execution_pins);
    totals.transfer_pins = totals.transfer_pins.saturating_add(arena.transfer_pins);
    totals.reservations = totals.reservations.saturating_add(arena.reservations);
    totals.active_transactions = totals
        .active_transactions
        .saturating_add(arena.active_transactions);
}

impl ManagedKvCacheManager {
    #[cfg(test)]
    pub(crate) fn telemetry_snapshot(&self) -> ManagedKvTelemetrySnapshot {
        self.telemetry.snapshot()
    }

    pub(crate) fn runtime_snapshot(&self) -> ManagedKvRuntimeSnapshot {
        let mut totals = ManagedKvRuntimeTotalsSnapshot {
            models: usize_to_u64(self.models.len()),
            ..ManagedKvRuntimeTotalsSnapshot::default()
        };
        let mut models = self
            .models
            .iter()
            .map(|(model_instance, state)| {
                totals.registered_sessions = totals
                    .registered_sessions
                    .saturating_add(usize_to_u64(state.registered_sessions.len()));
                totals.physical_bytes = totals
                    .physical_bytes
                    .saturating_add(state.runtime.physical_bytes());
                totals.resident_paged_bytes = totals
                    .resident_paged_bytes
                    .saturating_add(state.runtime.resident_paged_bytes());
                totals.authorized_tensor_bytes = totals
                    .authorized_tensor_bytes
                    .saturating_add(state.runtime.authorized_tensor_bytes());
                totals.materialized_tensor_bytes = totals
                    .materialized_tensor_bytes
                    .saturating_add(state.tensor_materialized_bytes);
                let mut arenas = state
                    .runtime
                    .plan
                    .groups
                    .iter()
                    .map(|group| {
                        let coordinator = state
                            .coordinators
                            .get(&group.arena)
                            .expect("resolved managed arena has a coordinator")
                            .stats();
                        let admission_claimed_pages = state
                            .capacity_claims
                            .values()
                            .flat_map(|claims| claims.iter())
                            .filter_map(|(arena, pages)| {
                                (*arena == group.arena).then_some(u64::from(*pages))
                            })
                            .sum::<u64>();
                        let admission_claims = state
                            .capacity_claims
                            .values()
                            .filter(|claims| claims.iter().any(|(arena, _)| *arena == group.arena))
                            .count();
                        let coordinator = ManagedKvCoordinatorSnapshot {
                            capacity_pages: usize_to_u64(coordinator.capacity_pages),
                            allocated_pages: usize_to_u64(coordinator.allocated_pages),
                            free_pages: usize_to_u64(coordinator.free_pages),
                            admission_claimed_pages,
                            admission_available_pages: usize_to_u64(coordinator.capacity_pages)
                                .saturating_sub(admission_claimed_pages),
                            admission_claims: usize_to_u64(admission_claims),
                            table_refs: usize_to_u64(coordinator.table_refs),
                            prefix_refs: usize_to_u64(coordinator.prefix_refs),
                            execution_pins: usize_to_u64(coordinator.execution_pins),
                            transfer_pins: usize_to_u64(coordinator.transfer_pins),
                            reservations: usize_to_u64(coordinator.reservations),
                            active_transactions: usize_to_u64(coordinator.active_transactions),
                        };
                        let arena = state
                            .runtime
                            .arenas
                            .get(&group.arena)
                            .expect("resolved managed arena has physical storage");
                        let operation_stats = arena.operation_stats();
                        let operations = ManagedKvOperationSnapshot {
                            slot_write_dispatches: operation_stats.slot_write_dispatches,
                            paged_prefill_dispatches: operation_stats.paged_prefill_dispatches,
                            paged_decode_dispatches: operation_stats.paged_decode_dispatches,
                            page_zero_dispatches: operation_stats.page_zero_dispatches,
                            page_copy_dispatches: operation_stats.page_copy_dispatches,
                            backing_allocations: operation_stats.backing_allocations.unwrap_or(0),
                            backing_allocations_observed_arenas: u64::from(
                                operation_stats.backing_allocations.is_some(),
                            ),
                            workspace_bytes: operation_stats.workspace_bytes.unwrap_or(0),
                            workspace_bytes_observed_arenas: u64::from(
                                operation_stats.workspace_bytes.is_some(),
                            ),
                            workspace_budget_bytes: operation_stats
                                .workspace_budget_bytes
                                .unwrap_or(0),
                            workspace_budget_bytes_observed_arenas: u64::from(
                                operation_stats.workspace_budget_bytes.is_some(),
                            ),
                            workspace_high_water_bytes: operation_stats
                                .workspace_high_water_bytes
                                .unwrap_or(0),
                            workspace_high_water_bytes_observed_arenas: u64::from(
                                operation_stats.workspace_high_water_bytes.is_some(),
                            ),
                            workspace_allocations: operation_stats
                                .workspace_allocations
                                .unwrap_or(0),
                            workspace_allocations_observed_arenas: u64::from(
                                operation_stats.workspace_allocations.is_some(),
                            ),
                            host_synchronizations: operation_stats.host_synchronizations,
                            cpu_reference_attention_dispatches: operation_stats
                                .cpu_reference_attention_dispatches,
                            portable_attention_dispatches: operation_stats
                                .portable_attention_dispatches,
                            cuda_native_attention_dispatches: operation_stats
                                .cuda_native_attention_dispatches,
                            cuda_flash_attention_dispatches: operation_stats
                                .cuda_flash_attention_dispatches,
                            cuda_split_decode_dispatches: operation_stats
                                .cuda_split_decode_dispatches,
                            metal_native_attention_dispatches: operation_stats
                                .metal_native_attention_dispatches,
                            cuda_graph_warmups: operation_stats.cuda_graph_warmups,
                            cuda_graph_captures: operation_stats.cuda_graph_captures,
                            cuda_graph_replays: operation_stats.cuda_graph_replays,
                            cuda_graph_fallbacks: operation_stats.cuda_graph_fallbacks,
                            cuda_graph_backoff_hits: operation_stats.cuda_graph_backoff_hits,
                            cuda_graph_evictions: operation_stats.cuda_graph_evictions,
                        };
                        add_coordinator_stats(&mut totals.coordinator, &coordinator);
                        totals.operations.add_assign(operations.clone());
                        totals.arenas = totals.arenas.saturating_add(1);
                        ManagedKvArenaRuntimeSnapshot {
                            generation: group.arena.generation,
                            group_id: group.id.get(),
                            domain_id: group.domain.get(),
                            device_ordinal: group.arena.device_ordinal,
                            page_tokens: group.page_tokens,
                            storage_dtype: format!("{:?}", group.storage.dtype())
                                .to_ascii_lowercase(),
                            token_capacity: u64::from(group.capacity_pages)
                                .saturating_mul(u64::from(group.page_tokens)),
                            bytes_per_page: group.bytes_per_page,
                            physical_bytes: arena.resident_bytes(),
                            prefix_capacity_pages: state
                                .prefix_indexes
                                .get(&group.arena)
                                .map_or(0, |index| usize_to_u64(index.capacity_pages())),
                            coordinator,
                            operations,
                        }
                    })
                    .collect::<Vec<_>>();
                arenas.sort_by_key(|arena| (arena.generation, arena.group_id));
                let tensor_capacity = state.runtime.tensor_state().map(|arena| arena.capacity());
                ManagedKvModelRuntimeSnapshot {
                    model_instance: *model_instance,
                    plan_fingerprint: state.runtime.plan.fingerprint().to_string(),
                    state_plan_v2_fingerprint: state
                        .runtime
                        .state_plan_v2
                        .fingerprint()
                        .to_string(),
                    backend: state.runtime.plan.backend,
                    device_ordinal: state.runtime.plan.device_ordinal,
                    resident_paged_bytes: state.runtime.resident_paged_bytes(),
                    authorized_tensor_bytes: state.runtime.authorized_tensor_bytes(),
                    materialized_tensor_bytes: state.tensor_materialized_bytes,
                    physical_bytes: state.runtime.physical_bytes(),
                    registered_sessions: usize_to_u64(state.registered_sessions.len()),
                    single_sequence_token_capacity: state.runtime.maximum_sequence_tokens(),
                    aggregate_token_capacity: state.runtime.logical_token_reach(),
                    full_context_sequence_capacity: full_context_sequence_capacity(
                        &state.runtime.plan,
                        state.runtime.maximum_sequence_tokens(),
                    ),
                    retained_sequence_capacity: tensor_capacity
                        .map(|capacity| capacity.sequence_capacity()),
                    staged_transaction_capacity: tensor_capacity
                        .map(|capacity| capacity.transaction_capacity()),
                    tensor_bytes_per_sequence: tensor_capacity
                        .map(|capacity| capacity.per_sequence_bytes()),
                    prefix_snapshot_capacity: tensor_capacity
                        .map(|capacity| capacity.snapshot_capacity()),
                    prefix_snapshot_budget_bytes: tensor_capacity
                        .map_or(0, |capacity| capacity.snapshot_capacity_bytes()),
                    prefix_snapshots: state
                        .tensor_snapshots
                        .as_ref()
                        .map_or(0, |sharing| usize_to_u64(sharing.index.len())),
                    prefix_snapshot_bytes: state
                        .tensor_snapshots
                        .as_ref()
                        .map_or(0, |sharing| sharing.index.total_bytes()),
                    incremental_claim_sessions: usize_to_u64(
                        state.incremental_claim_sessions.len(),
                    ),
                    arenas,
                }
            })
            .collect::<Vec<_>>();
        models.sort_by_key(|model| model.model_instance);
        let mut counters = self.telemetry.snapshot();
        counters.prefix_retained_pages = totals.coordinator.prefix_refs;
        counters.kv_host_pages = self.models.values().fold(0_u64, |total, state| {
            total
                + state
                    .host_offload
                    .as_ref()
                    .map(|offload| {
                        offload
                            .arenas
                            .values()
                            .map(|arena| arena.pool.resident_pages() as u64)
                            .sum::<u64>()
                    })
                    .unwrap_or(0)
        });
        ManagedKvRuntimeSnapshot {
            memory_accounting: "resident_paged_plus_authorized_tensor",
            totals,
            counters,
            models,
        }
    }

    #[cfg(test)]
    pub(crate) fn snapshot(
        &self,
        model_instance: ModelInstanceId,
        session: &SessionKey,
        domain: CacheDomainId,
    ) -> Option<KvSnapshot> {
        let state = self.models.get(&model_instance)?;
        state
            .coordinators
            .values()
            .find_map(|coordinator| coordinator.snapshot(session, domain).ok())
    }
}
