//! Preparing a step's managed state: reservations, admission and the prepare
//! transaction.

use super::*;

pub(super) fn reservation_for_group(
    group: KvGroupId,
    page_tokens: u32,
    snapshot: &KvSnapshot,
    target_tokens: u32,
    shared_prefix: &[crate::kv::CacheBlockRef],
) -> Result<KvGroupReservation> {
    let required_pages = target_tokens
        .checked_add(page_tokens - 1)
        .ok_or_else(|| Error::Overloaded("managed KV page count overflow".into()))?
        / page_tokens;
    let required_pages = usize::try_from(required_pages)
        .map_err(|_| Error::Overloaded("managed KV page count exceeds usize".into()))?;
    let existing = snapshot
        .groups
        .iter()
        .find(|table| table.group == group)
        .map(|table| table.blocks.as_slice())
        .unwrap_or_default();
    let mut blocks = Vec::with_capacity(required_pages);
    if snapshot.committed_tokens == 0 {
        blocks.extend(shared_prefix.iter().copied().map(KvBlockIntent::Shared));
    } else if !shared_prefix.is_empty() {
        return Err(Error::InferenceError(
            "managed KV prefix pages cannot replace a committed request table".into(),
        ));
    }
    for (index, block) in existing.iter().take(required_pages).copied().enumerate() {
        let is_partial_tail = !snapshot.committed_tokens.is_multiple_of(page_tokens)
            && index + 1 == existing.len()
            && target_tokens > snapshot.committed_tokens;
        blocks.push(if is_partial_tail {
            KvBlockIntent::Writable(block)
        } else {
            KvBlockIntent::Existing(block)
        });
    }
    blocks.extend(std::iter::repeat_n(
        KvBlockIntent::Fresh,
        required_pages.saturating_sub(blocks.len()),
    ));
    Ok(KvGroupReservation { group, blocks })
}

pub(super) fn abort_domains(
    state: &mut ManagedKvModelState,
    txn_id: PlanId,
    domains: &[ManagedCacheDomainReservation],
) {
    for domain in domains {
        let _ = state
            .coordinators
            .get_mut(&domain.arena)
            .expect("reservation arena has a coordinator")
            .abort(txn_id);
    }
}

pub(super) fn abort_reservation(state: &mut ManagedKvModelState, reservation: &ManagedCacheReservation) {
    state.pending_prefixes.remove(&reservation.txn_id);
    // DS4.3: an aborted transaction never publishes its restored pages, so
    // the host entries stay authoritative and keep their slots.
    state.pending_promotions.remove(&reservation.txn_id);
    state.exact_target_transactions.remove(&reservation.txn_id);
    abort_domains(state, reservation.txn_id, &reservation.domains);
    if reservation.clocked_state.is_some() {
        if let (Some(arena), Ok(transaction)) = (
            state.runtime.tensor_state(),
            PhysicalStateTransactionId::new(reservation.txn_id),
        ) {
            let _ = arena.abort(transaction);
        }
    }
}

impl ManagedKvCacheManager {
    #[cfg(test)]
    pub(crate) fn prepare(
        &mut self,
        runtime: &ManagedKvModelRuntime,
        txn_id: PlanId,
        session: &SessionKey,
        work: &WorkUnit,
        request: Option<&EngineCoreRequest>,
    ) -> Result<Option<ManagedCacheReservation>> {
        self.prepare_with_admission(
            runtime,
            txn_id,
            session,
            work,
            request,
            ManagedClaimPolicy::Full,
            true,
        )
    }

    /// Prepare one scheduled row. `prefix_attach` is false when the row's
    /// adapter cannot resume prefill above a non-zero cursor: a full-prefill
    /// row always runs its whole prompt from position 0, so attaching shared
    /// prefix pages would hand it a cache that does not start empty.
    pub(crate) fn prepare_row(
        &mut self,
        runtime: &ManagedKvModelRuntime,
        txn_id: PlanId,
        session: &SessionKey,
        work: &WorkUnit,
        request: Option<&EngineCoreRequest>,
        claim: ManagedClaimPolicy,
        prefix_attach: bool,
    ) -> Result<Option<ManagedCacheReservation>> {
        self.prepare_with_admission(
            runtime,
            txn_id,
            session,
            work,
            request,
            claim,
            prefix_attach,
        )
    }

    /// Incremental admission is only safe for execution adapters that can
    /// replay an already-streamed request after capacity preemption.
    #[cfg(test)]
    pub(crate) fn prepare_incremental(
        &mut self,
        runtime: &ManagedKvModelRuntime,
        txn_id: PlanId,
        session: &SessionKey,
        work: &WorkUnit,
        request: Option<&EngineCoreRequest>,
    ) -> Result<Option<ManagedCacheReservation>> {
        self.prepare_with_admission(
            runtime,
            txn_id,
            session,
            work,
            request,
            ManagedClaimPolicy::Incremental,
            true,
        )
    }

    pub(super) fn prepare_with_admission(
        &mut self,
        runtime: &ManagedKvModelRuntime,
        txn_id: PlanId,
        session: &SessionKey,
        work: &WorkUnit,
        request: Option<&EngineCoreRequest>,
        claim: ManagedClaimPolicy,
        prefix_attach: bool,
    ) -> Result<Option<ManagedCacheReservation>> {
        let previous_claim = self
            .models
            .get(&runtime.plan.model_instance)
            .and_then(|state| state.capacity_claims.get(session))
            .cloned();
        let result = self.prepare_inner(
            runtime,
            txn_id,
            session,
            work,
            request,
            claim,
            prefix_attach,
        );
        if result.is_ok()
            && claim != ManagedClaimPolicy::Full
            && matches!(work, WorkUnit::SequenceStep { .. })
        {
            if let Some(state) = self.models.get_mut(&runtime.plan.model_instance) {
                if state.capacity_claims.contains_key(session) {
                    state.incremental_claim_sessions.insert(session.clone());
                }
            }
        }
        if result.is_err() {
            if let Some(state) = self.models.get_mut(&runtime.plan.model_instance) {
                // Every arena belongs to the same logical transaction. Also
                // catch errors from projection/prefix setup between arenas.
                for coordinator in state.coordinators.values_mut() {
                    let _ = coordinator.abort(txn_id);
                }
                match previous_claim {
                    Some(claim) => {
                        state.capacity_claims.insert(session.clone(), claim);
                    }
                    None => {
                        state.capacity_claims.remove(session);
                    }
                }
            }
        }
        result
    }

    pub(super) fn prepare_inner(
        &mut self,
        runtime: &ManagedKvModelRuntime,
        txn_id: PlanId,
        session: &SessionKey,
        work: &WorkUnit,
        request: Option<&EngineCoreRequest>,
        claim: ManagedClaimPolicy,
        prefix_attach: bool,
    ) -> Result<Option<ManagedCacheReservation>> {
        let (
            sequence_input,
            sequence_phase,
            auxiliary_state,
            realtime_cache_append,
            allow_unchanged_prefix,
            exact_target_prefix,
        ) = match work {
            WorkUnit::SequenceStep {
                input,
                auxiliary_state,
                phase,
                ..
            } => (
                Some(*input),
                Some(*phase),
                auxiliary_state.as_deref(),
                None,
                *phase == crate::engine::SequencePhase::Decode,
                false,
            ),
            WorkUnit::RealtimePush {
                max_cache_append, ..
            }
            | WorkUnit::RealtimeFinish {
                max_cache_append, ..
            } => (None, None, None, Some(*max_cache_append), true, false),
            WorkUnit::RealtimePreparation {
                auxiliary_state, ..
            } => (None, None, auxiliary_state.as_deref(), Some(0), true, false),
            // Prompt prefill and completion advance no retained tensor state:
            // a family keeps any such state of those transactions in its own
            // checkpointed host state.
            WorkUnit::RealtimeCompletion { .. } => {
                (None, None, Some(&[][..]), Some(0), true, false)
            }
            WorkUnit::RealtimePromptPrefill { cache_append, .. } => {
                if *cache_append == 0 {
                    return Err(Error::InvalidInput(
                        "realtime prompt prefill requires a positive exact cache append".into(),
                    ));
                }
                (None, None, Some(&[][..]), Some(*cache_append), false, true)
            }
            WorkUnit::RealtimeDecodeContinuation {
                max_cache_append,
                auxiliary_state,
                ..
            } => {
                if *max_cache_append != 1 {
                    return Err(Error::InvalidInput(
                        "realtime decode continuation requires exactly one cache append".into(),
                    ));
                }
                (None, None, auxiliary_state.as_deref(), Some(1), false, true)
            }
            _ => return Ok(None),
        };
        if self
            .models
            .get(&runtime.plan.model_instance)
            .is_some_and(|state| state.closing)
        {
            return Err(Error::InferenceError(
                "managed KV runtime is closing".to_string(),
            ));
        }
        let namespace = managed_prefix_namespace(request, runtime, self.prefix_cache_salt)?;
        let selected_tensor_state = auxiliary_state
            .map(|spans| {
                spans
                    .iter()
                    .map(|span| {
                        let input = span.input();
                        Ok(TensorStateSelection {
                            group: span.group(),
                            clock: span.clock().clone(),
                            expected_cursor: u64::try_from(input.start).map_err(|_| {
                                Error::InvalidInput("clocked state cursor exceeds u64".to_string())
                            })?,
                            target_cursor: u64::try_from(input.end).map_err(|_| {
                                Error::InvalidInput("clocked state cursor exceeds u64".to_string())
                            })?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        if realtime_cache_append.is_some()
            && runtime.tensor_state().is_some()
            && selected_tensor_state.is_none()
        {
            return Err(Error::InferenceError(
                "realtime paged reservation cannot implicitly advance tensor state".into(),
            ));
        }
        if sequence_input.is_some_and(|input| input.is_empty()) && selected_tensor_state.is_none() {
            return Err(Error::InferenceError(
                "legacy decoder-coupled tensor state cannot advance without a paged span".into(),
            ));
        }
        let needs_tensor_transaction = runtime.tensor_state().is_some()
            && selected_tensor_state
                .as_ref()
                .is_none_or(|selections| !selections.is_empty());
        if runtime.tensor_state().is_none()
            && selected_tensor_state
                .as_ref()
                .is_some_and(|selections| !selections.is_empty())
        {
            return Err(Error::InferenceError(
                "clocked state spans selected a model without a tensor-state arena".into(),
            ));
        }
        let tensor_transaction = needs_tensor_transaction
            .then(|| PhysicalStateTransactionId::new(txn_id))
            .transpose()?;
        let needs_tensor_sequence = needs_tensor_transaction
            && self
                .models
                .get(&runtime.plan.model_instance)
                .is_some_and(|state| !state.tensor_sequences.contains_key(session));
        let tensor_sequence_candidate = if needs_tensor_sequence {
            let candidate = PhysicalStateSequenceId::new(self.next_tensor_sequence)?;
            self.next_tensor_sequence = self
                .next_tensor_sequence
                .checked_add(1)
                .ok_or_else(|| Error::InferenceError("tensor-state sequence id overflow".into()))?;
            Some(candidate)
        } else {
            None
        };
        let state = self
            .models
            .get_mut(&runtime.plan.model_instance)
            .ok_or_else(|| Error::InferenceError("managed KV model runtime is missing".into()))?;
        if state.runtime.plan.id != runtime.plan.id {
            return Err(Error::InferenceError(
                "request carries a stale managed KV runtime".to_string(),
            ));
        }
        // DS4: drain demotion before capacity claims so pages freed to the
        // host tier are visible to this very reservation.
        let host_offload_policy = self.host_offload_policy;
        Self::run_host_offload_tick(host_offload_policy, state, &self.telemetry);
        let host_page_ceiling = host_offload_policy
            .map(|policy| policy.max_promotion_pages)
            .unwrap_or(0);
        let installed_claim = if let Some(request) = request {
            match claim {
                ManagedClaimPolicy::Incremental if sequence_input.is_some() => {
                    ensure_incremental_capacity_claim(state, session, request, work)?
                }
                ManagedClaimPolicy::Chunked { quantum_tokens } if sequence_input.is_some() => {
                    ensure_chunked_capacity_claim(state, session, request, work, quantum_tokens)?
                }
                _ => ensure_capacity_claim(state, session, request)?,
            }
        } else {
            false
        };
        if let Err(error) = ensure_session_tables(state, session) {
            if installed_claim {
                state.capacity_claims.remove(session);
            }
            return Err(error);
        }
        let session_generation =
            state
                .session_generations
                .get(session)
                .copied()
                .ok_or_else(|| {
                    Error::InferenceError("registered managed session lost its generation".into())
                })?;

        let mut domains = Vec::with_capacity(runtime.plan.groups.len());
        let mut pending_prefixes = Vec::new();
        // DS4.3: host tails restored this transaction, committed at finalize.
        let mut pending_promotions = Vec::new();
        // DS1.2b: snapshot-sharing facts copied out of `state` so the paged
        // group loop can consult them while mutating coordinators. Clocked
        // (selected) transactions have no fork path and never share.
        let snapshot_sharing = if selected_tensor_state.is_none() {
            state.tensor_snapshots.as_ref().map(|sharing| {
                (
                    sharing.policy.clone(),
                    sharing.paged_arena,
                    sharing.interval_tokens,
                )
            })
        } else {
            None
        };
        let mut tensor_attach: Option<(u64, Arc<[(StateDomainId, Arc<[StateComponentValue]>)]>)> =
            None;
        for group in &runtime.plan.groups {
            let domain_sequence_input = match (sequence_input, sequence_phase, request) {
                (Some(input), Some(phase), Some(request)) => {
                    Some(request.project_paged_state_input(group.domain, phase, input)?)
                }
                (input, _, _) => input,
            };
            let coordinator = state
                .coordinators
                .get_mut(&group.arena)
                .expect("resolved arena has a coordinator");
            let snapshot = coordinator
                .snapshot(session, group.domain)
                .map_err(coordinator_error)?;
            let mut target_committed_tokens = match (domain_sequence_input, realtime_cache_append) {
                (Some(input), None) => u32::try_from(input.end).map_err(|_| {
                    Error::InvalidInput("managed KV token position exceeds u32".to_string())
                })?,
                // A realtime target bounds what the row may append. It
                // spans at least one page, which a paged view needs even
                // before a session's first token (its first preparation
                // appends nothing).
                (None, Some(max_cache_append)) => snapshot
                    .committed_tokens
                    .checked_add(u32::try_from(max_cache_append).map_err(|_| {
                        Error::InvalidInput(
                            "realtime managed KV append ceiling exceeds u32".to_string(),
                        )
                    })?)
                    .ok_or_else(|| {
                        Error::InvalidInput("realtime managed KV target overflow".to_string())
                    })?
                    .max(1),
                _ => unreachable!("managed work target was authenticated above"),
            };
            if target_committed_tokens < snapshot.committed_tokens {
                abort_domains(state, txn_id, &domains);
                return Err(Error::InferenceError(
                    "scheduled KV target regressed behind the committed cache table".to_string(),
                ));
            }
            if domain_sequence_input.is_some_and(|input| input.is_empty()) {
                if target_committed_tokens != snapshot.committed_tokens {
                    abort_domains(state, txn_id, &domains);
                    return Err(Error::InferenceError(
                        "empty decoder span disagrees with the committed paged cursor".into(),
                    ));
                }
                continue;
            }
            let prefix_eligible = prefix_attach
                && snapshot.committed_tokens == 0
                && domain_sequence_input.is_some_and(|input| input.start == 0)
                // Prefix reuse is bounded by this transaction's committed
                // target below, so the first scheduler-visible chunk does not
                // need to cover the entire logical prompt.
                && request.is_some_and(|request| {
                    domain_sequence_input
                        .is_some_and(|input| input.end <= request.prompt_tokens.len())
                })
                && target_committed_tokens > 1
                && prefix_enabled_for_domain(&state.contract, group.domain);
            // A subordinate retained-session generation is a semantic restart
            // whose model handoff requires an exact context-0 physical cache.
            // Published pages remain available to unrelated fresh sessions,
            // but this restarted session must rebuild its first generation
            // span instead of attaching a prefix and beginning above zero.
            // DS1.5: an admission probe may have scheduled this span to start
            // exactly at a verified shared-prefix cursor. A fresh table whose
            // span starts there attaches directly instead of recomputing the
            // resident prefix.
            let expected_attach_cursor = request
                .and_then(|req| req.managed_prefix_cursor())
                .filter(|cursor| {
                    *cursor > 0
                        && snapshot.committed_tokens == 0
                        && domain_sequence_input
                            .is_some_and(|input| input.start == *cursor as usize)
                });
            let mut prefix_match =
                if prefix_eligible && session_generation == ManagedSessionGeneration::INITIAL {
                    if let Some(namespace) = namespace.as_ref() {
                        let reusable_tokens =
                            usize::try_from(target_committed_tokens - 1).unwrap_or(usize::MAX);
                        let host_chain = state.host_offload.as_mut().and_then(|offload| {
                            offload
                                .arenas
                                .get_mut(&group.arena)
                                .map(|arena| &mut arena.chain)
                        });
                        let prefix_index = state
                            .prefix_indexes
                            .get_mut(&group.arena)
                            .expect("resolved arena has a prefix index");
                        managed_prefix_lookup(
                            prefix_index,
                            host_chain,
                            group.page_tokens,
                            namespace,
                            &request
                                .expect("prefix namespace requires a request")
                                .prompt_tokens[..reusable_tokens],
                            host_page_ceiling,
                        )?
                    } else {
                        self.telemetry.record_prefix_rejection();
                        Default::default()
                    }
                } else if let Some(cursor) = expected_attach_cursor {
                    if let Some(namespace) = namespace.as_ref() {
                        let reusable_tokens = cursor as usize;
                        let host_chain = state.host_offload.as_mut().and_then(|offload| {
                            offload
                                .arenas
                                .get_mut(&group.arena)
                                .map(|arena| &mut arena.chain)
                        });
                        let prefix_index = state
                            .prefix_indexes
                            .get_mut(&group.arena)
                            .expect("resolved arena has a prefix index");
                        managed_prefix_lookup(
                            prefix_index,
                            host_chain,
                            group.page_tokens,
                            namespace,
                            &request
                                .expect("prefix namespace requires a request")
                                .prompt_tokens[..reusable_tokens],
                            host_page_ceiling,
                        )?
                    } else {
                        Default::default()
                    }
                } else {
                    Default::default()
                };
            if let Some(cursor) = expected_attach_cursor {
                // The attach must reconcile the same snapshot boundary the
                // probe verified; a shorter realized cursor means the shared
                // state was evicted under us and the scheduled span is stale.
                if let Some((policy, paged_arena, _interval)) = &snapshot_sharing {
                    if *paged_arena == group.arena && policy.paged_domain == group.domain {
                        if let Some(sharing) = state.tensor_snapshots.as_mut() {
                            match reconcile_tensor_snapshot_attach(
                                sharing,
                                group.page_tokens,
                                &mut prefix_match,
                                needs_tensor_sequence,
                                &self.telemetry,
                            ) {
                                Ok(Some(attach)) => tensor_attach = Some(attach),
                                Ok(None) => {}
                                Err(error) => {
                                    abort_domains(state, txn_id, &domains);
                                    return Err(error);
                                }
                            }
                        }
                    }
                }
                if prefix_match.reused_tokens != cursor {
                    abort_domains(state, txn_id, &domains);
                    return Err(Error::Backpressure(format!(
                        "{MANAGED_PREFIX_CURSOR_LOST}: admission probed cursor {cursor} but the index now offers {}",
                        prefix_match.reused_tokens
                    )));
                }
            }
            // DS1.2b: tensor domains may only follow a paged attach when a
            // committed snapshot exists at the attach cursor. Walk the matched
            // chain back to the nearest snapshot-backed boundary, falling back
            // to zero reuse when none exists or the session cannot fork.
            let mut publication_start_tokens;
            if prefix_eligible {
                if let Some((policy, paged_arena, interval_tokens)) = &snapshot_sharing {
                    if *paged_arena == group.arena && policy.paged_domain == group.domain {
                        let pre_tensor_reused = prefix_match.reused_tokens;
                        if let Some(sharing) = state.tensor_snapshots.as_mut() {
                            match reconcile_tensor_snapshot_attach(
                                sharing,
                                group.page_tokens,
                                &mut prefix_match,
                                needs_tensor_sequence,
                                &self.telemetry,
                            ) {
                                Ok(Some(attach)) => tensor_attach = Some(attach),
                                Ok(None) => {}
                                Err(error) => {
                                    abort_domains(state, txn_id, &domains);
                                    return Err(error);
                                }
                            }
                        }
                        // Nothing attached: align the fresh commit cursor to the
                        // declared snapshot interval so the committed tensor
                        // state can be published at the same boundary the paged
                        // pages end on. The residual tokens are served by the
                        // next prefill chunk. An attached session keeps its
                        // exact target and simply commits the residual span.
                        //
                        // The reduction is only sound when the scheduled input
                        // span already ends at (or below) the boundary — the
                        // scheduler's first-chunk alignment guarantees that.
                        // Truncating a commit target below the span the
                        // executor is about to append would leave the model
                        // writing pages past its block table.
                        if tensor_attach.is_none() && prefix_match.reused_tokens == 0 {
                            let interval = u64::from(*interval_tokens);
                            let target = u64::from(target_committed_tokens);
                            let boundary = target - target % interval;
                            let span_fits_boundary = domain_sequence_input
                                .map(|input| {
                                    u64::try_from(input.end).is_ok_and(|end| end <= boundary)
                                })
                                .unwrap_or(false);
                            if span_fits_boundary && boundary > 0 && boundary < target {
                                target_committed_tokens =
                                    u32::try_from(boundary).map_err(|_| {
                                        Error::InvalidInput(
                                            "managed KV snapshot boundary exceeds u32".into(),
                                        )
                                    })?;
                            }
                        }
                        // Pages the lookup matched but the reconcile dropped are
                        // already published by another session under the same
                        // digests; republishing them from fresh blocks would
                        // bind one digest to two blocks. Publish only pages the
                        // lookup did not already see.
                        publication_start_tokens =
                            pre_tensor_reused.max(prefix_match.reused_tokens);
                    } else {
                        publication_start_tokens = prefix_match.reused_tokens;
                    }
                } else {
                    publication_start_tokens = prefix_match.reused_tokens;
                }
            } else {
                publication_start_tokens = prefix_match.reused_tokens;
            }
            // DS4: a host tail means pages from the device boundary onward
            // bind into the device index at this transaction's commit — the
            // restored ones and any recomputed ones alike — so publication
            // starts at the device-resident boundary instead of the execution
            // cursor (finalized after the restore below).
            let host_tail_boundary = prefix_match
                .host_tail
                .as_ref()
                .map(|tail| tail.device_end_tokens);
            let sliding_window = sliding_window_for_domain(&state.contract, group.domain)?;
            // The earliest query of this span sits at `start` and still attends
            // the `window` keys ending at itself, so the table must keep keys
            // from `start + 1 - window`. Keeping only `target - window` drops
            // the lookback of every query but the last in a multi-token span
            // (chunked prefill, speculative verification). A single-token step
            // is unchanged, and the next step advances the window again.
            let target_window_start = sliding_window
                .map(|window| {
                    let start = domain_sequence_input
                        .map(|input| u32::try_from(input.start).unwrap_or(u32::MAX))
                        .unwrap_or(snapshot.committed_tokens);
                    target_committed_tokens
                        .saturating_sub(window)
                        .min(start.saturating_add(1).saturating_sub(window))
                })
                .unwrap_or(0);
            let established_window_table = sliding_window.is_some()
                && snapshot.groups.iter().any(|table| table.group == group.id);
            let reserve_request = if established_window_table {
                None
            } else {
                let reservation = reservation_for_group(
                    group.id,
                    group.page_tokens,
                    &snapshot,
                    target_committed_tokens,
                    &prefix_match.blocks,
                )?;
                Some(KvReserveRequest {
                    txn_id,
                    expected: snapshot.clone(),
                    target_committed_tokens,
                    target_window_start: 0,
                    groups: vec![reservation],
                })
            };
            let reserve_once = |coordinator: &mut KvCacheCoordinator| {
                if established_window_table {
                    coordinator.reserve_window(KvWindowReserveRequest {
                        txn_id,
                        expected: snapshot.clone(),
                        target_committed_tokens,
                        target_window_start,
                        page_tokens: group.page_tokens,
                    })
                } else {
                    let request = reserve_request
                        .as_ref()
                        .expect("non-window reservation exists");
                    match coordinator.reserve(request.clone()) {
                        Err(KvCoordinatorError::WriteConflict) => {
                            let mut copy_on_write = request.clone();
                            for intent in &mut copy_on_write.groups[0].blocks {
                                if let KvBlockIntent::Writable(source) = *intent {
                                    *intent = KvBlockIntent::CopyOnWrite(source);
                                }
                            }
                            coordinator.reserve(copy_on_write)
                        }
                        result => result,
                    }
                }
            };
            let mut reserved = reserve_once(coordinator);
            while matches!(reserved, Err(KvCoordinatorError::Capacity)) {
                let protected = prefix_match.blocks.iter().copied().collect::<HashSet<_>>();
                let evicted = state
                    .prefix_indexes
                    .get_mut(&group.arena)
                    .expect("resolved arena has a prefix index")
                    .evict_lru_excluding(coordinator, &protected)
                    .map_err(prefix_error)?;
                if evicted.is_empty() {
                    break;
                }
                reserved = reserve_once(coordinator);
            }
            if let Err(error) = reserved {
                abort_domains(state, txn_id, &domains);
                if matches!(error, KvCoordinatorError::Capacity) {
                    return Err(Error::Backpressure(
                        "managed KV arena has no reservable pages".to_string(),
                    ));
                }
                return Err(coordinator_error(error));
            }
            let prepared = match coordinator.prepare(txn_id) {
                Ok(prepared) => prepared,
                Err(error) => {
                    let _ = coordinator.abort(txn_id);
                    abort_domains(state, txn_id, &domains);
                    return Err(coordinator_error(error));
                }
            };
            let required_resident_pages = prepared
                .provisional_groups
                .iter()
                .flat_map(|group| group.blocks.iter())
                .map(|block| block.index.saturating_add(1))
                .max()
                .unwrap_or(0);
            let arena = runtime
                .arena(group.arena)
                .expect("resolved arena allocated");
            let growth_result = (|| -> Result<()> {
                if let Some(growth) = arena.plan_resident_growth(required_resident_pages)? {
                    let added_bytes = group
                        .bytes_per_page
                        .checked_mul(u64::from(growth.added_pages()))
                        .ok_or_else(|| {
                            Error::Overloaded("managed KV growth byte total overflow".into())
                        })?;
                    let final_materialized =
                        state.materialized_resources.checked_add(ResourceVector {
                            device_bytes: ResourceAmount::Known(added_bytes),
                            ..ResourceVector::zero()
                        })?;
                    // Replacing a contiguous Candle tensor keeps the old arena
                    // alive while allocating the complete target arena. Reserve
                    // that admission-only peak before touching device memory,
                    // then shrink the lease to the final resident footprint.
                    let target_bytes = group
                        .bytes_per_page
                        .checked_mul(u64::from(growth.target_pages))
                        .ok_or_else(|| {
                            Error::Overloaded("managed KV growth peak byte total overflow".into())
                        })?;
                    let allocation_peak =
                        state.materialized_resources.checked_add(ResourceVector {
                            device_bytes: ResourceAmount::Known(target_bytes),
                            ..ResourceVector::zero()
                        })?;
                    if let Some(lease) = state.resource_lease.as_ref() {
                        if !allocation_peak.fits_within(lease.resources()) {
                            return Err(Error::InferenceError(
                                "managed CUDA growth exceeds its load-time replacement reservation"
                                    .into(),
                            ));
                        }
                    }
                    on_device_thread(arena.as_ref(), || arena.grow_resident_pages(growth))?;
                    state.allocation_ledger.reconcile_group_receipt(
                        runtime.allocation_plan(),
                        crate::kv::v2::StateGroupId::new(group.id.get()),
                        group.domain,
                        AllocationReceipt {
                            requested_owned_bytes: added_bytes,
                            committed_owned_bytes: added_bytes,
                            allocator_overhead_bytes: 0,
                            residency: ResidencyMeasurement::Unknown,
                        },
                    )?;
                    if let Some(lease) = state.resource_lease.as_ref() {
                        lease.record_materialized_usage(final_materialized.checked_add(
                            tensor_state_resources(
                                state.runtime.plan.backend,
                                state.tensor_materialized_bytes,
                            )?,
                        )?)?;
                    }
                    state.materialized_resources = final_materialized;
                    self.telemetry.record_backing_allocation();
                }
                Ok(())
            })();
            if let Err(error) = growth_result {
                let _ = coordinator.abort(txn_id);
                abort_domains(state, txn_id, &domains);
                return Err(error);
            }
            let fresh = &prepared.fresh_blocks;
            if !fresh.is_empty() {
                let arena = runtime
                    .arena(group.arena)
                    .expect("resolved arena allocated");
                if let Err(error) = on_device_thread(arena.as_ref(), || {
                    arena.zero_pages(fresh).and_then(|fence| fence.wait())
                }) {
                    let _ = coordinator.abort(txn_id);
                    abort_domains(state, txn_id, &domains);
                    return Err(error);
                }
                self.telemetry.record_zero(fresh.len());
            }
            if !prepared.page_copies.is_empty() {
                let arena = runtime
                    .arena(group.arena)
                    .expect("resolved arena allocated");
                if let Err(error) = on_device_thread(arena.as_ref(), || {
                    arena
                        .copy_pages(&prepared.page_copies)
                        .and_then(|fence| fence.wait())
                }) {
                    let _ = coordinator.abort(txn_id);
                    abort_domains(state, txn_id, &domains);
                    return Err(error);
                }
                self.telemetry.record_copy(prepared.page_copies.len());
                if !prefix_match.blocks.is_empty() {
                    self.telemetry
                        .record_prefix_copy_on_write(prepared.page_copies.len());
                }
            }
            // DS4.3: restore the host-resident tail of the matched prefix into
            // the fresh pages this transaction reserved for their span. A page
            // that fails to restore truncates the promotion (the failing entry
            // and its host descendants are purged); with an admission cursor
            // the shortened match degrades to the cursor-lost re-plan, without
            // one the executor simply recomputes the truncated span.
            if let Some(tail) = prefix_match.host_tail.take() {
                let promoted = {
                    let arena = runtime
                        .arena(group.arena)
                        .expect("resolved arena allocated");
                    match state
                        .host_offload
                        .as_mut()
                        .and_then(|offload| offload.arenas.get_mut(&group.arena))
                    {
                        Some(arena_offload) => {
                            let table = prepared
                                .provisional_groups
                                .iter()
                                .find(|table| table.group == group.id);
                            promote_host_tail(
                                arena.as_ref(),
                                arena_offload,
                                group.arena,
                                group.page_tokens,
                                table,
                                tail,
                            )
                        }
                        // The host tier vanished mid-prepare: nothing was
                        // restored, so the match falls back to its device head.
                        None => PromotedTail {
                            restored_tokens: tail.device_end_tokens,
                            record: None,
                        },
                    }
                };
                if promoted.restored_tokens < prefix_match.reused_tokens {
                    if expected_attach_cursor
                        .is_some_and(|cursor| promoted.restored_tokens != cursor)
                    {
                        let _ = coordinator.abort(txn_id);
                        abort_domains(state, txn_id, &domains);
                        return Err(Error::Backpressure(format!(
                            "{MANAGED_PREFIX_CURSOR_LOST}: promotion restored cursor {} but admission probed {}",
                            promoted.restored_tokens, prefix_match.reused_tokens
                        )));
                    }
                    prefix_match.reused_tokens = promoted.restored_tokens;
                }
                if let Some(record) = promoted.record {
                    pending_promotions.push(record);
                }
            }
            let execution_start_tokens = snapshot.committed_tokens.max(prefix_match.reused_tokens);
            publication_start_tokens = match host_tail_boundary {
                Some(device_end_tokens) => publication_start_tokens.min(device_end_tokens),
                None => publication_start_tokens.max(execution_start_tokens),
            };
            domains.push(ManagedCacheDomainReservation {
                arena: group.arena,
                domain: group.domain,
                expected_version: prepared.expected.version,
                expected_committed_tokens: prepared.expected.committed_tokens,
                execution_start_tokens,
                target_committed_tokens: prepared.target_committed_tokens,
                target_window_start: prepared.target_window_start,
                first_page_offset: prepared.target_window_start % group.page_tokens,
                provisional_groups: prepared.provisional_groups,
                writable_blocks: prepared.writable_blocks,
            });
            // Phase 5a: a request declaring its stable-history boundary also
            // publishes the pages of every later prefill span inside its
            // prompt, so a conversation's chain keeps growing turn by turn
            // instead of stopping at the first chunk.
            let declared_boundaries = request
                .map(|request| {
                    [
                        request.managed_preamble_boundary(),
                        request.managed_snapshot_boundary(),
                    ]
                })
                .unwrap_or_default();
            let declared_boundary = declared_boundaries.iter().any(Option::is_some);
            let boundary_publication = declared_boundary
                && session_generation == ManagedSessionGeneration::INITIAL
                && target_committed_tokens > 1
                && request.is_some_and(|request| {
                    domain_sequence_input
                        .is_some_and(|input| input.end <= request.prompt_tokens.len())
                })
                && prefix_enabled_for_domain(&state.contract, group.domain);
            if prefix_eligible || expected_attach_cursor.is_some() || boundary_publication {
                let Some(namespace) = namespace.as_ref() else {
                    continue;
                };
                let publications = prefix_publications(
                    namespace,
                    &request
                        .expect("prefix namespace requires a request")
                        .prompt_tokens,
                    group.page_tokens,
                    publication_start_tokens,
                    target_committed_tokens,
                    domains
                        .last()
                        .expect("domain reservation was just appended"),
                    group.id,
                )?;
                if !publications.is_empty() {
                    // Without a declared boundary keep the DS1.2b rule (any
                    // aligned commit of an eligible span). With them, only a
                    // span ending on a boundary's interval floor (first turn
                    // or history) snapshots: each snapshot is the full
                    // hybrid state.
                    let snapshot = match (declared_boundary, &snapshot_sharing) {
                        (false, _) => true,
                        (true, Some((_, _, interval))) if *interval > 0 => {
                            declared_boundaries.iter().flatten().any(|boundary| {
                                target_committed_tokens == boundary - boundary % *interval
                            })
                        }
                        (true, _) => false,
                    };
                    pending_prefixes.push(PendingPrefixCommit {
                        arena: group.arena,
                        page_tokens: group.page_tokens,
                        publications,
                        snapshot,
                    });
                }
            }
        }
        if !pending_prefixes.is_empty()
            && state
                .pending_prefixes
                .insert(txn_id, pending_prefixes)
                .is_some()
        {
            abort_domains(state, txn_id, &domains);
            return Err(Error::InferenceError(
                "managed KV transaction duplicated pending prefix publication".into(),
            ));
        }
        if !pending_promotions.is_empty() {
            state.pending_promotions.insert(txn_id, pending_promotions);
        }
        let clocked_state = if needs_tensor_transaction {
            let arena = runtime
                .tensor_state()
                .expect("tensor transaction requires an arena");
            let transaction = tensor_transaction.expect("tensor arena has a transaction");
            let (sequence, newly_registered) =
                if let Some(sequence) = state.tensor_sequences.get(session).copied() {
                    (sequence, false)
                } else {
                    let sequence = tensor_sequence_candidate.expect("tensor arena has a candidate");
                    if let Err(error) = arena.register(sequence) {
                        abort_domains(state, txn_id, &domains);
                        state.pending_prefixes.remove(&txn_id);
                        state.pending_promotions.remove(&txn_id);
                        return Err(error);
                    }
                    state.tensor_sequences.insert(session.clone(), sequence);
                    (sequence, true)
                };
            // DS1.2b: attach-by-fork. Seed the freshly registered sequence with
            // the committed snapshot (read -> begin -> stage -> commit at the
            // fork cursor, the DS1.1 prototype recipe) so the executor's
            // tensor updates stage on top of the attach cursor exactly as the
            // paged window does.
            if let Some((fork_cursor, snapshot_domains)) = tensor_attach.take() {
                if !newly_registered {
                    abort_domains(state, txn_id, &domains);
                    state.pending_prefixes.remove(&txn_id);
                    state.pending_promotions.remove(&txn_id);
                    return Err(Error::InferenceError(
                        "managed tensor snapshot attach requires a fresh sequence".into(),
                    ));
                }
                let fork_result = (|| -> Result<()> {
                    let fork_transaction = self.next_tensor_fork_transaction;
                    self.next_tensor_fork_transaction =
                        fork_transaction.checked_add(1).ok_or_else(|| {
                            Error::InferenceError(
                                "tensor fork transaction space is exhausted".into(),
                            )
                        })?;
                    let seed = PhysicalStateTransactionId::new(u64::MAX - fork_transaction)?;
                    let staged = (|| -> Result<()> {
                        arena.begin(seed, sequence)?;
                        for (domain, components) in snapshot_domains.iter() {
                            arena.stage_replace(
                                seed,
                                *domain,
                                0,
                                fork_cursor,
                                components.iter().cloned().collect(),
                            )?;
                        }
                        // A fresh sequence has no committed state to retire.
                        arena.commit(seed, fork_cursor).map(drop)
                    })();
                    if staged.is_err() {
                        let _ = arena.abort(seed);
                    }
                    staged
                })();
                if let Err(error) = fork_result {
                    abort_domains(state, txn_id, &domains);
                    state.pending_prefixes.remove(&txn_id);
                    state.pending_promotions.remove(&txn_id);
                    state.tensor_sequences.remove(session);
                    release_tensor_sequence(state, arena, sequence).map_err(|release_error| {
                        Error::InferenceError(format!(
                            "tensor snapshot fork failed ({error}); newly registered sequence rollback also failed: {release_error}"
                        ))
                    })?;
                    return Err(error);
                }
                // The fork commits snapshot tensors that already exist.
                reconcile_tensor_materialization(state);
                self.telemetry.record_tensor_snapshot_attach();
            }
            let managed_reservation = if let Some(selections) = selected_tensor_state.as_ref() {
                ManagedClockedStateReservation::selected(
                    runtime.plan().model_instance,
                    sequence.get(),
                    selections.clone().into(),
                )
            } else {
                ManagedClockedStateReservation::legacy(
                    runtime.plan().model_instance,
                    sequence.get(),
                )
            };
            let managed_reservation = match managed_reservation {
                Ok(reservation) => reservation,
                Err(error) => {
                    abort_domains(state, txn_id, &domains);
                    state.pending_prefixes.remove(&txn_id);
                    state.pending_promotions.remove(&txn_id);
                    if newly_registered {
                        state.tensor_sequences.remove(session);
                        release_tensor_sequence(state, arena, sequence)?;
                    }
                    return Err(error);
                }
            };
            let begin = if let Some(selections) = selected_tensor_state.as_ref() {
                arena.begin_selected(transaction, sequence, selections)
            } else {
                arena.begin(transaction, sequence)
            };
            if let Err(error) = begin {
                abort_domains(state, txn_id, &domains);
                state.pending_prefixes.remove(&txn_id);
                state.pending_promotions.remove(&txn_id);
                if newly_registered {
                    state.tensor_sequences.remove(session);
                    release_tensor_sequence(state, arena, sequence).map_err(|release_error| {
                        Error::InferenceError(format!(
                            "tensor transaction admission failed ({error}); newly registered sequence rollback also failed: {release_error}"
                        ))
                    })?;
                }
                return Err(error);
            }
            Some(managed_reservation)
        } else {
            None
        };
        if domains.is_empty() && clocked_state.is_none() {
            return Ok(None);
        }
        if exact_target_prefix && !state.exact_target_transactions.insert(txn_id) {
            abort_reservation(
                state,
                &ManagedCacheReservation {
                    txn_id,
                    session: session.clone(),
                    session_generation,
                    domains: domains.clone(),
                    clocked_state: clocked_state.clone(),
                    allow_unchanged_prefix,
                },
            );
            return Err(Error::InferenceError(
                "managed exact-target transaction identity was reused".into(),
            ));
        }
        Ok(Some(ManagedCacheReservation {
            txn_id,
            session: session.clone(),
            session_generation,
            domains,
            clocked_state,
            allow_unchanged_prefix,
        }))
    }
}
