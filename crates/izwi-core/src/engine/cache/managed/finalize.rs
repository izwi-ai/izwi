//! Finalizing prepared transactions, releasing sessions and unloading models.

use super::*;

impl ManagedKvCacheManager {
    pub(crate) fn finalize(
        &mut self,
        reservation: &ManagedCacheReservation,
        receipt: Option<&ManagedCacheReceipt>,
        commit: bool,
    ) -> Result<()> {
        let model_instance = if let Some(domain) = reservation.domains.first() {
            domain.arena.model_instance
        } else if let Some(clocked) = reservation.clocked_state.as_ref() {
            clocked.model_instance()
        } else {
            return Err(Error::InferenceError(
                "managed reservation contains no paged or clocked state".into(),
            ));
        };
        let state = self
            .models
            .get_mut(&model_instance)
            .ok_or_else(|| Error::InferenceError("managed KV model state is missing".into()))?;
        if !commit {
            abort_reservation(state, reservation);
            self.telemetry.record_abort();
            return Ok(());
        }
        if let Err(error) = validate_reservation_session_generation(state, reservation) {
            abort_reservation(state, reservation);
            return Err(error);
        }
        if let Some(clocked) = reservation.clocked_state.as_ref() {
            let sequence = PhysicalStateSequenceId::new(clocked.sequence())?;
            if clocked.model_instance() != model_instance
                || state.tensor_sequences.get(&reservation.session) != Some(&sequence)
            {
                abort_reservation(state, reservation);
                return Err(Error::InferenceError(
                    "clocked-state reservation crossed its model/session sequence fence".into(),
                ));
            }
        }
        let receipt = match receipt {
            Some(receipt) => receipt,
            None => {
                abort_reservation(state, reservation);
                return Err(Error::InferenceError(
                    "committing managed KV row omitted its write receipt".into(),
                ));
            }
        };
        if &receipt.reservation != reservation {
            abort_reservation(state, reservation);
            return Err(Error::InferenceError(
                "managed KV receipt crossed a row transaction fence".to_string(),
            ));
        }
        if let Err(error) = receipt.validate() {
            abort_reservation(state, reservation);
            return Err(error);
        }

        let accepted_prefix = receipt.accepted_prefix();
        if state
            .exact_target_transactions
            .contains(&reservation.txn_id)
            && reservation.domains.iter().any(|domain| {
                accepted_prefix.unwrap_or(domain.target_committed_tokens)
                    != domain.target_committed_tokens
            })
        {
            abort_reservation(state, reservation);
            return Err(Error::InferenceError(
                "managed KV exact-target reservation rejected a partial prefix".into(),
            ));
        }
        // DS4.3: host tails restored at prepare, reclaimed at this commit.
        let promotions = state
            .pending_promotions
            .get(&reservation.txn_id)
            .cloned()
            .unwrap_or_default();
        let mut resolved_domains = Vec::with_capacity(reservation.domains.len());
        for domain in &reservation.domains {
            let Some(written) = receipt
                .domains
                .iter()
                .find(|receipt| receipt.arena == domain.arena && receipt.domain == domain.domain)
            else {
                abort_reservation(state, reservation);
                return Err(Error::InferenceError(
                    "managed KV receipt omitted a cache domain".into(),
                ));
            };
            let group = state
                .runtime
                .plan
                .groups
                .iter()
                .find(|group| group.arena == domain.arena && group.domain == domain.domain)
                .ok_or_else(|| {
                    Error::InferenceError(
                        "managed KV reservation lost its authoritative page geometry".into(),
                    )
                });
            let group = match group {
                Ok(group) => group,
                Err(error) => {
                    abort_reservation(state, reservation);
                    return Err(error);
                }
            };
            let committed_tokens = accepted_prefix.unwrap_or(domain.target_committed_tokens);
            if committed_tokens < domain.execution_start_tokens
                || committed_tokens > domain.target_committed_tokens
            {
                abort_reservation(state, reservation);
                return Err(Error::InferenceError(
                    "managed KV accepted prefix is outside one domain reservation".into(),
                ));
            }
            let target_window_start = if accepted_prefix.is_some() {
                match sliding_window_for_domain(&state.contract, domain.domain) {
                    Ok(Some(window)) => committed_tokens
                        .saturating_sub(window)
                        .min(domain.expected_committed_tokens),
                    Ok(None) => 0,
                    Err(error) => {
                        abort_reservation(state, reservation);
                        return Err(error);
                    }
                }
            } else {
                domain.target_window_start
            };
            // DS4.3: pages restored from the host tier were filled by the
            // promotion copy rather than the executor's backend writes; the
            // committed write receipt covers both. Restored pages the accepted
            // cursor truncates away are left out — their host entries stay
            // authoritative and only the surviving span is reclaimed.
            let mut written_blocks = written.written_blocks.clone();
            if let Some(promotion) = promotions
                .iter()
                .find(|promotion| promotion.arena == domain.arena)
            {
                for (index, block) in promotion.blocks.iter().enumerate() {
                    let page_end = u64::from(promotion.device_end_tokens)
                        + u64::from(promotion.page_tokens).saturating_mul(index as u64 + 1);
                    if page_end <= u64::from(committed_tokens) && !written_blocks.contains(block) {
                        written_blocks.push(*block);
                    }
                }
            }
            resolved_domains.push((
                domain,
                written_blocks,
                committed_tokens,
                target_window_start,
                group.page_tokens,
            ));
        }

        // Mark every live transaction written. This changes no table/index
        // ownership and is rolled back by abort if any later validation fails.
        for (domain, written_blocks, committed_tokens, target_window_start, page_tokens) in
            &resolved_domains
        {
            let receipt = KvWriteReceipt {
                txn_id: reservation.txn_id,
                committed_tokens: *committed_tokens,
                written_blocks: written_blocks.clone(),
            };
            let coordinator = state
                .coordinators
                .get_mut(&domain.arena)
                .expect("reservation arena has a coordinator");
            let completed = if accepted_prefix.is_some() {
                coordinator.complete_write_prefix(receipt, *target_window_start, *page_tokens)
            } else {
                coordinator.complete_write(receipt)
            };
            if let Err(error) = completed {
                abort_reservation(state, reservation);
                return Err(coordinator_error(error));
            }
        }
        let mut pending = state
            .pending_prefixes
            .get(&reservation.txn_id)
            .cloned()
            .unwrap_or_default();
        let mut staged = Vec::<(
            KvArenaId,
            KvCoordinatorCommitPlan,
            Option<StagedPrefixCommit>,
        )>::with_capacity(reservation.domains.len());
        // DS1.2b: snapshot boundary (chain digest, cursor end) captured from
        // the accepted paged publications of the sharing arena.
        let mut snapshot_boundary: Option<([u8; 32], u64)> = None;
        // DS1.2b: committed cursor of a legacy tensor transaction, set after
        // `arena.commit` succeeds.
        let mut tensor_commit_cursor: Option<u64> = None;
        for domain in &reservation.domains {
            let prefix = if let Some(index) = pending
                .iter()
                .position(|publication| publication.arena == domain.arena)
            {
                let mut publication = pending.swap_remove(index);
                let committed_tokens = accepted_prefix.unwrap_or(domain.target_committed_tokens);
                publication.publications.retain(|candidate| {
                    candidate
                        .key
                        .start_position
                        .checked_add(candidate.key.tokens.len() as u64)
                        .is_some_and(|end| end <= u64::from(committed_tokens))
                });
                if publication.publications.is_empty() {
                    None
                } else {
                    // DS1.2b: the snapshot boundary is the accepted page whose
                    // end equals the committed cursor; capture it for the
                    // tensor publication after the state commits.
                    if let Some(sharing) = state.tensor_snapshots.as_ref() {
                        if sharing.paged_arena == domain.arena && publication.snapshot {
                            let last = publication.publications.last().expect("non-empty");
                            if let Some(end) = last
                                .key
                                .start_position
                                .checked_add(last.key.tokens.len() as u64)
                            {
                                snapshot_boundary = Some((last.key.digest(), end));
                            }
                        }
                    }
                    let staged_prefix = state
                        .prefix_indexes
                        .get(&domain.arena)
                        .expect("reservation arena has a prefix index")
                        .stage_transaction(publication.page_tokens, &publication.publications);
                    Some(match staged_prefix {
                        Ok(staged) => staged,
                        Err(error) => {
                            abort_reservation(state, reservation);
                            return Err(prefix_error(error));
                        }
                    })
                }
            } else {
                None
            };
            let coordinator = state
                .coordinators
                .get_mut(&domain.arena)
                .expect("reservation arena has a coordinator");
            let commit = coordinator.stage_commit_with_prefix_updates(
                reservation.txn_id,
                prefix
                    .as_ref()
                    .map(StagedPrefixCommit::retained)
                    .unwrap_or(&[]),
                prefix
                    .as_ref()
                    .map(StagedPrefixCommit::released)
                    .unwrap_or(&[]),
            );
            match commit {
                Ok(commit) => staged.push((domain.arena, commit, prefix)),
                Err(error) => {
                    abort_reservation(state, reservation);
                    return Err(coordinator_error(error));
                }
            }
        }
        if !pending.is_empty() {
            abort_reservation(state, reservation);
            return Err(Error::InferenceError(
                "managed KV transaction contains a prefix publication for an unknown domain".into(),
            ));
        }
        if let Some(clocked_state) = reservation.clocked_state.as_ref() {
            // Every failure below must abort the reservation; returning early
            // with `?` would leave the coordinator transaction `Written` with
            // its pins and reservations held, which later blocks unload.
            let committed = (|| {
                let arena = state.runtime.tensor_state().ok_or_else(|| {
                    Error::InferenceError("tensor-state reservation lost its physical arena".into())
                })?;
                let transaction = PhysicalStateTransactionId::new(reservation.txn_id)?;
                if clocked_state.selections().is_some() {
                    let completion = receipt
                        .clocked_state()
                        .expect("validated selected receipt has a completion")
                        .completion();
                    return arena.commit_selected(transaction, completion);
                }
                let target_cursor = reservation
                    .domains
                    .first()
                    .map(|domain| {
                        u64::from(accepted_prefix.unwrap_or(domain.target_committed_tokens))
                    })
                    .ok_or_else(|| {
                        Error::InferenceError(
                            "legacy tensor transaction requires a paged cursor".into(),
                        )
                    })?;
                if reservation.domains.iter().any(|domain| {
                    u64::from(accepted_prefix.unwrap_or(domain.target_committed_tokens))
                        != target_cursor
                }) {
                    return Err(Error::InferenceError(
                        "one managed state transaction resolved divergent domain cursors".into(),
                    ));
                }
                arena.commit(transaction, target_cursor)
            })();
            let retired = match committed {
                Ok(retired) => retired,
                Err(error) => {
                    abort_reservation(state, reservation);
                    return Err(error);
                }
            };
            reconcile_tensor_materialization(state);
            drop(retired);
            if clocked_state.selections().is_none() {
                // DS1.2b: a legacy transaction committed every shared tensor
                // domain at one cursor; record it for the snapshot publication
                // below. Clocked completions publish only selected groups and
                // never share.
                tensor_commit_cursor = reservation.domains.first().map(|domain| {
                    u64::from(accepted_prefix.unwrap_or(domain.target_committed_tokens))
                });
            }
        }
        // Every fallible operation has succeeded. Applying these plans cannot
        // fail, and the engine state lock prevents an interleaving mutation.
        for (arena, commit, prefix) in staged {
            state
                .coordinators
                .get_mut(&arena)
                .expect("staged arena has a coordinator")
                .apply_staged_commit(commit);
            if let Some(prefix) = prefix {
                state
                    .prefix_indexes
                    .get_mut(&arena)
                    .expect("staged arena has a prefix index")
                    .apply_staged(prefix);
            }
        }
        // DS4.3: the transaction committed, so its restored pages are
        // device-resident and published into the prefix index; reclaim the
        // host entries they replace. An entry already evicted from the host
        // tier is simply gone.
        for promotion in state
            .pending_promotions
            .remove(&reservation.txn_id)
            .unwrap_or_default()
        {
            if let Some(arena_offload) = state
                .host_offload
                .as_mut()
                .and_then(|offload| offload.arenas.get_mut(&promotion.arena))
            {
                for digest in &promotion.digests {
                    if let Some(slot) = arena_offload.chain.remove(digest) {
                        let _ = arena_offload.pool.release_slot(slot);
                    }
                }
            }
            self.telemetry
                .record_promotion(promotion.digests.len(), promotion.latency_ns);
        }
        state.pending_prefixes.remove(&reservation.txn_id);
        state.exact_target_transactions.remove(&reservation.txn_id);
        // DS1.2b: publish the committed tensor snapshot when the commit cursor
        // is the aligned boundary of the accepted page chain. This runs after
        // every fallible operation, so a skipped publication only loses reuse
        // and never leaves committed state without its paged chain.
        if let Some((digest, boundary)) = snapshot_boundary {
            if tensor_commit_cursor.is_some_and(|cursor| cursor == boundary) {
                let published = publish_tensor_snapshot(state, reservation, digest, boundary);
                match published {
                    Ok(true) => self.telemetry.record_tensor_snapshot_publish(),
                    Ok(false) => {}
                    Err(error) => tracing::warn!(
                        error = %error,
                        "committed tensor snapshot publication skipped"
                    ),
                }
            }
        }
        self.telemetry.record_commit();
        Ok(())
    }

    pub(crate) fn release_session(&mut self, session: &SessionKey) -> Result<()> {
        for state in self.models.values_mut() {
            if !state.registered_sessions.contains(session) {
                continue;
            }
            for group in &state.runtime.plan.groups {
                state
                    .coordinators
                    .get(&group.arena)
                    .expect("resolved arena has a coordinator")
                    .validate_table_release(session, group.domain)
                    .map_err(coordinator_error)?;
            }
            if let Some(sequence) = state.tensor_sequences.get(session).copied() {
                state
                    .runtime
                    .tensor_state()
                    .ok_or_else(|| {
                        Error::InferenceError(
                            "registered tensor-state sequence lost its arena".into(),
                        )
                    })?
                    .validate_release(sequence)?;
            }
            for group in &state.runtime.plan.groups {
                state
                    .coordinators
                    .get_mut(&group.arena)
                    .expect("resolved arena has a coordinator")
                    .release_table(session, group.domain)
                    .map_err(coordinator_error)?;
            }
            if let Some(sequence) = state.tensor_sequences.remove(session) {
                let arena = state.runtime.tensor_state().cloned().ok_or_else(|| {
                    Error::InferenceError("registered tensor-state sequence lost its arena".into())
                })?;
                release_tensor_sequence(state, &arena, sequence)?;
            }
            state.registered_sessions.remove(session);
            state.session_generations.remove(session);
            state.capacity_claims.remove(session);
            state.incremental_claim_sessions.remove(session);
        }
        Ok(())
    }

    /// Replace every paged table for one exact retained session by an empty
    /// table in a new subordinate generation. The request/session identity and
    /// its capacity claim remain owned throughout the reset, so another row
    /// cannot consume the admitted logical capacity between attempts.
    pub(crate) fn reset_session_generation(
        &mut self,
        runtime: &ManagedKvModelRuntime,
        session: &SessionKey,
        expected: ManagedSessionGeneration,
    ) -> Result<ManagedSessionGeneration> {
        let state = self
            .models
            .get_mut(&runtime.plan.model_instance)
            .ok_or_else(|| Error::InferenceError("managed KV model state is missing".into()))?;
        if state.closing || state.runtime.plan.id != runtime.plan.id {
            return Err(Error::InferenceError(
                "managed KV reset carries a closing or stale model runtime".into(),
            ));
        }
        if runtime.tensor_state().is_some() {
            return Err(Error::InferenceError(
                "managed KV session reset requires a paged-only runtime".into(),
            ));
        }
        if !state.registered_sessions.contains(session) {
            return Err(Error::InferenceError(
                "managed KV reset requires a registered session".into(),
            ));
        }
        let current = state
            .session_generations
            .get(session)
            .copied()
            .ok_or_else(|| {
                Error::InferenceError("registered managed session lost its generation".into())
            })?;
        if current != expected {
            return Err(Error::InferenceError(format!(
                "managed KV reset expected session generation {}, found {}",
                expected.get(),
                current.get()
            )));
        }
        let next = current.next()?;
        let mut staged = Vec::<(KvArenaId, KvCoordinatorTableResetPlan)>::with_capacity(
            state.runtime.plan.groups.len(),
        );
        for group in &state.runtime.plan.groups {
            let coordinator = state
                .coordinators
                .get(&group.arena)
                .expect("resolved arena has a coordinator");
            let snapshot = coordinator
                .snapshot(session, group.domain)
                .map_err(coordinator_error)?;
            let next_version = snapshot.version.checked_add(1).ok_or_else(|| {
                Error::InferenceError("managed KV table version overflow during reset".into())
            })?;
            staged.push((
                group.arena,
                coordinator
                    .stage_table_reset(session, group.domain, next_version)
                    .map_err(coordinator_error)?,
            ));
        }

        // Every fallible ownership/version check completed above. EngineCore
        // serializes manager mutation, so applying the staged resets cannot
        // race another reservation between domains.
        for (arena, reset) in staged {
            state
                .coordinators
                .get_mut(&arena)
                .expect("staged reset arena remains registered")
                .apply_staged_table_reset(reset);
        }
        state.session_generations.insert(session.clone(), next);
        Ok(next)
    }

    /// Drain and retire every arena belonging to one exact loaded-model
    /// generation. The model-scoped physical lease is retained until no
    /// session, row transaction, device fence, or external runtime handle can
    /// still reference the backing storage.
    pub(crate) fn prepare_unload_model(&mut self, model_instance: ModelInstanceId) -> Result<bool> {
        self.prepare_unload_model_with_runtime_owners(model_instance, 1)
    }

    pub(crate) fn prepare_unload_model_with_runtime_owners(
        &mut self,
        model_instance: ModelInstanceId,
        expected_runtime_owners: usize,
    ) -> Result<bool> {
        if expected_runtime_owners == 0 {
            return Err(Error::InvalidInput(
                "managed KV unload expected zero runtime owners".into(),
            ));
        }
        let Some(state) = self.models.get_mut(&model_instance) else {
            return Ok(false);
        };
        state.closing = true;
        if !state.registered_sessions.is_empty() {
            return Err(Error::InferenceError(format!(
                "managed KV model {} still has registered sessions",
                model_instance.get()
            )));
        }
        if !state.capacity_claims.is_empty() {
            return Err(Error::InferenceError(format!(
                "managed KV model {} still has logical capacity claims",
                model_instance.get()
            )));
        }
        if Arc::strong_count(&state.runtime) != expected_runtime_owners {
            return Err(Error::InferenceError(format!(
                "managed KV model {} has {} runtime owners, expected {expected_runtime_owners}",
                model_instance.get(),
                Arc::strong_count(&state.runtime),
            )));
        }
        // Physical caches clone the arena handle, not the runtime. A clone
        // that outlived its step would keep device backing alive after the
        // lease is released, so the runtime must be the arena's only owner.
        if let Some((arena, owners)) = state
            .runtime
            .arenas
            .iter()
            .map(|(id, arena)| (id, Arc::strong_count(arena)))
            .find(|(_, owners)| *owners != 1)
        {
            return Err(Error::InferenceError(format!(
                "managed KV model {} arena generation {} has {owners} owners, expected 1",
                model_instance.get(),
                arena.generation,
            )));
        }
        for group in &state.runtime.plan.groups {
            loop {
                let evicted = state
                    .prefix_indexes
                    .get_mut(&group.arena)
                    .expect("resolved arena has a prefix index")
                    .evict_lru(
                        state
                            .coordinators
                            .get_mut(&group.arena)
                            .expect("resolved arena has a coordinator"),
                    )
                    .map_err(prefix_error)?;
                if evicted.is_empty() {
                    break;
                }
            }
        }
        for coordinator in state.coordinators.values() {
            let stats = coordinator.stats();
            if stats.allocated_pages != 0
                || stats.table_refs != 0
                || stats.prefix_refs != 0
                || stats.execution_pins != 0
                || stats.transfer_pins != 0
                || stats.reservations != 0
                || stats.active_transactions != 0
            {
                return Err(Error::InferenceError(format!(
                    "managed KV model {} still has live page ownership or transactions",
                    model_instance.get()
                )));
            }
        }
        for arena in state.runtime.arenas.values() {
            arena.drain()?;
        }
        // Committed prefix snapshots are the last tensor state the model
        // holds (no session is registered): free them with their claim.
        clear_tensor_snapshots(state)?;
        if let Some(lease) = state.resource_lease.as_ref() {
            lease.prepare_materialized_release(ResourceVector::zero())?;
        }
        Ok(true)
    }

    pub(crate) fn commit_prepared_unload_model(&mut self, model_instance: ModelInstanceId) -> bool {
        self.models.remove(&model_instance).is_some()
    }

    pub(crate) fn unload_model(&mut self, model_instance: ModelInstanceId) -> Result<bool> {
        if !self.prepare_unload_model(model_instance)? {
            return Ok(false);
        }
        Ok(self.commit_prepared_unload_model(model_instance))
    }
}
