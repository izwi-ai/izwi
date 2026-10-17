//! Managed prefix reuse: namespaces, lookup and publication, host offload
//! and tensor snapshots.

use super::*;

pub(super) fn managed_prefix_namespace(
    request: Option<&EngineCoreRequest>,
    runtime: &ManagedKvModelRuntime,
    cache_salt: Option<[u8; 32]>,
) -> Result<Option<KvPrefixNamespace>> {
    let (Some(request), Some(cache_salt)) = (request, cache_salt) else {
        return Ok(None);
    };
    let Some(binding) = request.execution_adapter_binding() else {
        return Ok(None);
    };
    if request.task_type != crate::engine::TaskType::Chat
        || request.model_instance_id() != Some(runtime.plan.model_instance)
        || binding.model_instance_id != runtime.plan.model_instance
    {
        return Ok(None);
    }

    // The current managed producer is text-only Qwen3. Each digest is derived
    // from exact lifecycle/adapter facts already sealed onto this request. The
    // model-generation fence deliberately prevents reuse across reloads until
    // artifact revisions and tokenizer ABIs become first-class lifecycle data.
    let model_revision = digest_parts(
        b"izwi.kv.loaded-model-generation.v1\0",
        &[&runtime.plan.model_instance.get().to_le_bytes()],
    );
    let adapter_abi = digest_parts(
        b"izwi.kv.loaded-adapter-abi.v1\0",
        &[
            &binding.adapter_instance_id.get().to_le_bytes(),
            &binding.adapter_abi_revision.get().to_le_bytes(),
            binding.capability_id.as_bytes(),
        ],
    );
    let tokenizer_or_input_encoding = digest_parts(
        b"izwi.kv.loaded-input-encoding.v1\0",
        &[
            &runtime.plan.model_instance.get().to_le_bytes(),
            &binding.adapter_instance_id.get().to_le_bytes(),
            binding.capability_id.as_bytes(),
        ],
    );
    let position_semantics = digest_parts(
        b"izwi.kv.position-semantics.v1\0",
        &[&runtime.plan.contract_fingerprint],
    );
    // One process serves many tenants behind a gateway. Without the tenant in
    // the salt, a tenant could time another tenant's prompts through prefix
    // hits, so a trusted tenant key scopes every page to its own namespace.
    let cache_salt = match request.tenant_key {
        Some(tenant_key) => digest_parts(
            b"izwi.kv.tenant-prefix-salt.v1\0",
            &[&cache_salt, &tenant_key],
        ),
        None => cache_salt,
    };
    Ok(Some(KvPrefixNamespace {
        model_instance: runtime.plan.model_instance,
        model_revision,
        adapter_abi,
        tokenizer_or_input_encoding,
        position_semantics,
        plan: runtime.plan.fingerprint(),
        multimodal_artifact: None,
        cache_salt,
    }))
}

/// Evenly divide one model's host-offload budget between its arenas so the
/// pools together never exceed the single budget charged to the model lease.
pub(super) fn split_host_offload_budget(budget_bytes: u64, arenas: usize) -> u64 {
    budget_bytes / u64::try_from(arenas.max(1)).unwrap_or(u64::MAX)
}

pub(super) fn digest_parts(domain: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for part in parts {
        hasher.update((part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    hasher.finalize().into()
}

pub(super) fn prefix_enabled_for_domain(
    contract: &InferenceStateContract,
    domain_id: crate::kv::CacheDomainId,
) -> bool {
    // Only paged-attention domains route through the token-page index below.
    // For a hybrid contract the paged attach is sound only when every other
    // domain can fork its committed state at the same cursor, so the strict
    // sharing gate must accept the whole contract; only paged-only contracts
    // keep the plain paged behavior. A ring (LFM2 ShortConv), append or
    // static domain has no snapshot fork path, so attaching pages beside it
    // would continue from a reset ring at cursor N.
    let paged_shareable = contract.domains.iter().any(|domain| {
        if domain.id() != domain_id {
            return false;
        }
        matches!(domain, StateDomainSpec::PagedAttention(spec) if matches!(
            spec.header.prefix,
            PrefixPolicy::CommittedPages { .. }
        ))
    });
    if !paged_shareable {
        return false;
    }
    let stateful_domains = contract
        .domains
        .iter()
        .filter(|domain| !matches!(domain, StateDomainSpec::PagedAttention(_)))
        .count();
    stateful_domains == 0 || tensor_snapshot_prefix_policy(contract).is_some()
}

pub(super) fn sliding_window_for_domain(
    contract: &InferenceStateContract,
    domain_id: crate::kv::CacheDomainId,
) -> Result<Option<u32>> {
    let Some(domain) = contract
        .domains
        .iter()
        .find(|domain| domain.id() == domain_id)
    else {
        return Err(Error::InferenceError(
            "managed KV plan references a missing semantic domain".into(),
        ));
    };
    let StateDomainSpec::PagedAttention(spec) = domain else {
        return Ok(None);
    };
    let first = spec.layers.first().ok_or_else(|| {
        Error::InvalidInput("managed paged-attention domain has no layers".into())
    })?;
    if spec
        .layers
        .iter()
        .any(|layer| layer.pattern != first.pattern)
    {
        // A mixed local/global decoder (for example Gemma 3) shares one
        // physical page table across layers. Keep the complete logical table;
        // local layers lower their own bounded view at attention dispatch.
        return Ok(None);
    }
    Ok(match first.pattern {
        AttentionPattern::Full => None,
        AttentionPattern::SlidingWindow { window_tokens } => Some(window_tokens),
    })
}

pub(super) fn validate_sliding_contract(
    contract: &InferenceStateContract,
    backend: BackendKind,
) -> Result<()> {
    for domain in &contract.domains {
        if sliding_window_for_domain(contract, domain.id())?.is_some() {
            // Every managed paged-attention backend carries the authoritative
            // first-page offset in its decode and prefill metadata. Keep this
            // match exhaustive so a future backend must make that contract an
            // explicit admission decision.
            match backend {
                BackendKind::Cpu | BackendKind::Metal | BackendKind::Cuda => {}
            }
        }
    }
    Ok(())
}

/// DS4: the longest committed prefix for one arena, continuing past the
/// device index into the host chain when the arena carries a host tail. The
/// digest chain stays complete across the tier boundary, so snapshot
/// reconciliation can walk a match back through host-resident pages.
pub(super) fn managed_prefix_lookup(
    prefix_index: &mut CoordinatedPrefixIndex,
    host_chain: Option<&mut offload::HostChainIndex>,
    page_tokens: u32,
    namespace: &KvPrefixNamespace,
    tokens: &[u32],
    host_page_ceiling: usize,
) -> Result<KvPrefixMatch> {
    let telemetry = Arc::clone(prefix_index.telemetry());
    let matched = match host_chain {
        Some(chain) => offload::lookup_longest_with_host(
            prefix_index,
            chain,
            namespace,
            tokens,
            page_tokens,
            host_page_ceiling,
        )
        .map_err(prefix_error)?,
        None => prefix_index
            .lookup_longest(namespace, tokens, page_tokens)
            .map_err(prefix_error)?,
    };
    // Counted here, at preparation, and not in the index: admission probes
    // look up the same prompt first and would count every request twice.
    telemetry.record_prefix_lookup(usize_to_u64(tokens.len()), u64::from(matched.reused_tokens));
    Ok(matched)
}

/// DS4.3: outcome of restoring a matched host tail during prepare. The
/// restored boundary may sit below the matched end when a page failed to
/// copy; `record` carries the successfully restored span to the transaction's
/// commit.
pub(super) struct PromotedTail {
    pub(super) restored_tokens: u32,
    pub(super) record: Option<PendingPrefixPromotion>,
}

/// Run one core-side KV page operation (zero, copy, growth, offload capture
/// or restore) on its arena's device thread instead of the tokio worker that
/// prepares the step.
pub(super) fn on_device_thread<T: Send>(
    arena: &dyn KvArena,
    operation: impl FnOnce() -> Result<T> + Send,
) -> Result<T> {
    DeviceThread::for_location(arena.device_location())?.run_sync(operation)?
}

/// Restores a matched host tail into the fresh device pages the current
/// transaction reserved for its span. The provisional table supplies one
/// block per tail page; a shorter table truncates the promotion exactly like
/// a failed copy does.
pub(super) fn promote_host_tail(
    arena: &dyn KvArena,
    offload: &mut offload::ArenaOffload,
    arena_id: KvArenaId,
    page_tokens: u32,
    table: Option<&super::super::coordinator::GroupBlockTable>,
    tail: KvHostTailMatch,
) -> PromotedTail {
    let page = usize::try_from(page_tokens).unwrap_or(usize::MAX).max(1);
    let first_page = usize::try_from(tail.device_end_tokens).unwrap_or(usize::MAX) / page;
    let mut blocks = Vec::with_capacity(tail.digests.len());
    for index in first_page..first_page + tail.digests.len() {
        match table.and_then(|table| table.blocks.get(index)).copied() {
            Some(block) => blocks.push(block),
            None => break,
        }
    }
    // A failure to reach the device thread restores nothing: the match
    // falls back to its device-resident head.
    let copy = on_device_thread(arena, || {
        Ok(offload::promote_tail(
            arena,
            offload,
            &blocks,
            &tail.digests[..blocks.len()],
            &tail.slots[..blocks.len()],
        ))
    })
    .unwrap_or(offload::PromotionCopy {
        restored_pages: 0,
        latency_ns: 0,
    });
    let restored_tokens = tail.device_end_tokens.saturating_add(
        u32::try_from(copy.restored_pages.saturating_mul(page)).unwrap_or(u32::MAX),
    );
    let record = (copy.restored_pages > 0).then(|| PendingPrefixPromotion {
        arena: arena_id,
        page_tokens,
        device_end_tokens: tail.device_end_tokens,
        blocks: blocks[..copy.restored_pages].to_vec(),
        digests: tail.digests[..copy.restored_pages].to_vec(),
        latency_ns: copy.latency_ns,
    });
    PromotedTail {
        restored_tokens,
        record,
    }
}

pub(super) fn prefix_publications(
    namespace: &KvPrefixNamespace,
    tokens: &[u32],
    page_tokens: u32,
    execution_start_tokens: u32,
    target_tokens: u32,
    reservation: &ManagedCacheDomainReservation,
    group: KvGroupId,
) -> Result<Vec<KvPrefixPublication>> {
    let page_tokens_usize = usize::try_from(page_tokens)
        .map_err(|_| Error::InvalidInput("managed prefix page size exceeds usize".into()))?;
    let target = usize::try_from(target_tokens)
        .map_err(|_| Error::InvalidInput("managed prefix target exceeds usize".into()))?;
    if target > tokens.len() || page_tokens_usize == 0 {
        return Ok(Vec::new());
    }
    let table = reservation
        .provisional_groups
        .iter()
        .find(|table| table.group == group)
        .ok_or_else(|| Error::InferenceError("managed prefix group table is missing".into()))?;
    let first_new_page = usize::try_from(execution_start_tokens / page_tokens)
        .map_err(|_| Error::InvalidInput("managed prefix start exceeds usize".into()))?;
    let complete_pages = target / page_tokens_usize;
    let mut previous = None;
    let mut publications = Vec::new();
    for page_index in 0..complete_pages {
        let start = page_index * page_tokens_usize;
        let key = KvPrefixPageKey::new(
            namespace,
            previous,
            start as u64,
            tokens[start..start + page_tokens_usize].to_vec(),
        )
        .map_err(prefix_error)?;
        previous = Some(key.digest());
        if page_index < first_new_page {
            continue;
        }
        let block = table.blocks.get(page_index).copied().ok_or_else(|| {
            Error::InferenceError("managed prefix publication exceeds its block table".into())
        })?;
        publications.push(KvPrefixPublication { key, block });
    }
    Ok(publications)
}

/// DS1.2b: reconcile a paged prefix match with the committed tensor snapshot
/// index. The fork cursor must equal the paged attach cursor, so the matched
/// chain is walked back to the nearest boundary holding a committed snapshot;
/// a chain with no snapshot anywhere falls back to zero reuse rather than
/// attaching attention pages the tensor domains cannot follow.
#[allow(clippy::type_complexity)]
pub(super) fn reconcile_tensor_snapshot_attach(
    sharing: &mut TensorSnapshotSharing,
    page_tokens: u32,
    prefix_match: &mut KvPrefixMatch,
    can_fork: bool,
    telemetry: &ManagedKvTelemetry,
) -> Result<Option<(u64, Arc<[(StateDomainId, Arc<[StateComponentValue]>)]>)>> {
    if prefix_match.reused_tokens == 0 {
        return Ok(None);
    }
    if !can_fork {
        // The session already owns a live tensor sequence, so a fork would
        // cross a non-zero cursor. Only zero reuse is sound.
        telemetry.record_tensor_snapshot_truncation();
        prefix_match.blocks.clear();
        prefix_match.page_digests.clear();
        prefix_match.host_tail = None;
        prefix_match.reused_tokens = 0;
        return Ok(None);
    }
    while !prefix_match.page_digests.is_empty() {
        let boundary = u64::from(prefix_match.reused_tokens);
        let digest = prefix_match
            .page_digests
            .last()
            .copied()
            .expect("non-empty page digest chain");
        if let Some(domains) = sharing.index.lookup(digest, boundary)? {
            return Ok(Some((boundary, domains)));
        }
        prefix_match.pop_last_page(page_tokens);
        telemetry.record_tensor_snapshot_truncation();
    }
    Ok(None)
}

/// DS1.2b: build and index the committed snapshot for one aligned legacy
/// commit. `Ok(false)` means nothing new is held: the boundary was already
/// published identically, or the snapshot budget cannot hold it.
///
/// Every indexed snapshot is pinned in the tensor arena's committed backing,
/// so its bytes stay recorded as materialized on the model lease after the
/// publisher's sequence replaces or releases that state. Evicted snapshots
/// are unpinned and the lease reconciled (restoring their pending claim)
/// before their tensors drop.
pub(super) fn publish_tensor_snapshot(
    state: &mut ManagedKvModelState,
    reservation: &ManagedCacheReservation,
    digest: [u8; 32],
    boundary: u64,
) -> Result<bool> {
    let Some(sharing) = state.tensor_snapshots.as_mut() else {
        return Ok(false);
    };
    let interval = u64::from(sharing.interval_tokens);
    if boundary == 0 || !boundary.is_multiple_of(interval) {
        return Ok(false);
    }
    let Some(clocked) = reservation.clocked_state.as_ref() else {
        return Ok(false);
    };
    let sequence = PhysicalStateSequenceId::new(clocked.sequence())?;
    let arena = state.runtime.tensor_state().cloned().ok_or_else(|| {
        Error::InferenceError("tensor snapshot publication lost its physical arena".into())
    })?;
    let mut entry = Vec::with_capacity(sharing.policy.tensor_domains.len());
    for domain in sharing.policy.tensor_domains.iter() {
        let snapshot = arena.read(sequence, *domain)?.ok_or_else(|| {
            Error::InferenceError(
                "committed tensor snapshot publication found no domain state".into(),
            )
        })?;
        if snapshot.cursor != boundary {
            return Err(Error::InferenceError(
                "committed tensor snapshot cursor disagrees with its paged boundary".into(),
            ));
        }
        entry.push((*domain, snapshot.components));
    }
    let domains: Arc<[(StateDomainId, Arc<[StateComponentValue]>)]> = entry.into();
    let byte_size = tensor_snapshot_backing_bytes(&domains);
    let publication = sharing
        .index
        .publish(digest, boundary, domains.clone(), byte_size)?;
    if publication.inserted {
        for (_, components) in domains.iter() {
            arena.pin_snapshot(components)?;
        }
    }
    for evicted in &publication.evicted {
        for (_, components) in evicted.domains.iter() {
            arena.unpin_snapshot(components)?;
        }
    }
    if !publication.evicted.is_empty() {
        reconcile_tensor_materialization(state);
    }
    drop(publication.evicted);
    Ok(publication.inserted)
}

/// Drop every indexed snapshot (model unload): unpin them, reconcile the
/// lease, then free them.
pub(super) fn clear_tensor_snapshots(state: &mut ManagedKvModelState) -> Result<()> {
    let (Some(sharing), Some(arena)) = (
        state.tensor_snapshots.as_mut(),
        state.runtime.tensor_state().cloned(),
    ) else {
        return Ok(());
    };
    let cleared = sharing.index.clear();
    for entry in &cleared {
        for (_, components) in entry.domains.iter() {
            arena.unpin_snapshot(components)?;
        }
    }
    reconcile_tensor_materialization(state);
    drop(cleared);
    Ok(())
}

pub(super) fn prefix_error(error: impl fmt::Display) -> Error {
    Error::InferenceError(format!(
        "managed KV prefix index rejected operation: {error}"
    ))
}

impl ManagedKvCacheManager {
    #[cfg(test)]
    pub(crate) fn with_prefix_cache_salt(
        resource_authority: Option<Arc<ResourceAuthority>>,
        salt: Option<[u8; 32]>,
    ) -> Self {
        Self::with_prefix_cache_policy(resource_authority, salt, usize::MAX)
    }

    pub(crate) fn with_prefix_cache_policy(
        resource_authority: Option<Arc<ResourceAuthority>>,
        salt: Option<[u8; 32]>,
        max_prefix_cache_pages: usize,
    ) -> Self {
        let mut manager = Self::new(resource_authority);
        manager.prefix_cache_salt = salt;
        manager.max_prefix_cache_pages = max_prefix_cache_pages;
        manager
    }

    pub(crate) fn for_worker_with_prefix_cache_policy(
        resource_authority: Option<Arc<ResourceAuthority>>,
        salt: Option<[u8; 32]>,
        max_prefix_cache_pages: usize,
        backend: BackendKind,
        device: Device,
    ) -> Self {
        let mut manager = Self::for_worker(resource_authority, backend, device);
        manager.prefix_cache_salt = salt;
        manager.max_prefix_cache_pages = max_prefix_cache_pages;
        manager
    }

    /// Enables DS4 hierarchical offload with the resolved policy. A `None`
    /// policy (no host budget) keeps offload fully dormant.
    pub(crate) fn set_host_offload_policy(&mut self, policy: Option<HostOffloadPolicy>) {
        self.host_offload_policy = policy;
    }

    /// Committed prefix snapshots each snapshot-sharing model may hold (its
    /// snapshot budget, in rows of hybrid state). Takes effect for models
    /// loaded afterwards.
    pub(crate) fn set_prefix_snapshot_rows(&mut self, rows: u32) {
        self.prefix_snapshot_rows = rows;
    }

    /// Whether the prefix page budget was left unset (see
    /// [`prefix_index_pages`]). Takes effect for models loaded afterwards.
    pub(crate) fn set_automatic_prefix_cache_pages(&mut self, automatic: bool) {
        self.automatic_prefix_cache_pages = automatic;
    }

    /// The snapshot rows a model with `contract` is fitted and reserved for:
    /// the configured budget when its hybrid contract shares committed
    /// snapshots and prefix caching is engaged (a namespace and a page
    /// budget), else zero. A snapshot outlives its publisher's committed
    /// state, so these bytes are authorized on the model lease at load,
    /// recorded as materialized while the index pins them, and released when
    /// it evicts them; they never grow memory outside the reservation.
    pub(crate) fn prefix_snapshot_rows_for(&self, contract: &InferenceStateContract) -> u32 {
        if self.prefix_cache_salt.is_some()
            && self.max_prefix_cache_pages > 0
            && tensor_snapshot_prefix_policy(contract).is_some()
        {
            self.prefix_snapshot_rows
        } else {
            0
        }
    }

    /// DS1.5: read-only admission probe for the managed prefix cursor.
    ///
    /// Returns the largest committed boundary the first prefill span may
    /// start above: complete shared pages plus, for snapshot-sharing
    /// contracts, a committed tensor snapshot at that exact boundary. The
    /// scheduler plans the first span from this cursor; prepare re-verifies
    /// the same lookup transactionally before any attach is staged, so a
    /// boundary that disappears between admission and execution degrades to
    /// the cursor-lost retry instead of a stale attach.
    pub(crate) fn probe_managed_prefix(
        &mut self,
        runtime: &ManagedKvModelRuntime,
        request: &EngineCoreRequest,
    ) -> Result<Option<u32>> {
        let namespace = managed_prefix_namespace(Some(request), runtime, self.prefix_cache_salt)?;
        let Some(namespace) = namespace else {
            return Ok(None);
        };
        let Some(state) = self.models.get_mut(&runtime.plan.model_instance) else {
            return Ok(None);
        };
        if state.closing {
            return Ok(None);
        }
        // The cursor must leave at least one private prompt token; a fully
        // cached prompt has no executable first span.
        let prompt_tokens = request.prompt_tokens.as_slice();
        if prompt_tokens.len() < 2 {
            return Ok(None);
        }
        let reusable = prompt_tokens.len() - 1;
        let max_host_pages = self
            .host_offload_policy
            .map(|policy| policy.max_promotion_pages)
            .unwrap_or(0);
        for group in runtime.plan.groups.iter() {
            if !prefix_enabled_for_domain(&state.contract, group.domain) {
                continue;
            }
            let chain = state.host_offload.as_mut().and_then(|offload| {
                offload
                    .arenas
                    .get_mut(&group.arena)
                    .map(|arena| &mut arena.chain)
            });
            let prefix_index = state.prefix_indexes.get_mut(&group.arena).ok_or_else(|| {
                Error::InferenceError(
                    "resolved arena has no prefix index for the prefix probe".into(),
                )
            })?;
            let mut matched = match chain {
                Some(chain) => offload::lookup_longest_with_host(
                    &mut *prefix_index,
                    chain,
                    &namespace,
                    &prompt_tokens[..reusable],
                    group.page_tokens,
                    max_host_pages,
                )
                .map_err(prefix_error)?,
                None => prefix_index
                    .lookup_longest(&namespace, &prompt_tokens[..reusable], group.page_tokens)
                    .map_err(prefix_error)?,
            };
            if matched.reused_tokens == 0 {
                continue;
            }
            // Snapshot-sharing contracts may only start above a boundary that
            // also carries a committed tensor snapshot; walk the matched chain
            // back the same way prepare's reconcile does, without touching
            // attach/truncation telemetry (no fork happens here).
            if let Some(sharing) = state.tensor_snapshots.as_mut() {
                if sharing.paged_arena == group.arena && sharing.policy.paged_domain == group.domain
                {
                    while !matched.page_digests.is_empty() {
                        let boundary = u64::from(matched.reused_tokens);
                        let digest = matched
                            .page_digests
                            .last()
                            .copied()
                            .expect("matched chain holds complete pages");
                        if sharing.index.lookup(digest, boundary)?.is_some() {
                            return Ok(Some(matched.reused_tokens));
                        }
                        matched.pop_last_page(group.page_tokens);
                    }
                    continue;
                }
            }
            return Ok(Some(matched.reused_tokens));
        }
        Ok(None)
    }

    /// SCH-F7: [`Self::probe_managed_prefix`] for one exact session at
    /// schedule time. It offers a cursor only where prepare would attach it:
    /// an initial session generation (a reset generation must rebuild from
    /// context 0), no committed table tokens, and no live tensor sequence that
    /// a snapshot fork would have to cross. Anything else would lose the
    /// cursor at prepare and replan the request.
    pub(crate) fn probe_managed_prefix_for_session(
        &mut self,
        runtime: &ManagedKvModelRuntime,
        request: &EngineCoreRequest,
        session: &SessionKey,
    ) -> Result<Option<u32>> {
        if let Some(state) = self.models.get(&runtime.plan.model_instance) {
            let restarted = state
                .session_generations
                .get(session)
                .is_some_and(|generation| *generation != ManagedSessionGeneration::INITIAL);
            let committed = state.runtime.plan.groups.iter().any(|group| {
                state
                    .coordinators
                    .get(&group.arena)
                    .and_then(|coordinator| coordinator.snapshot(session, group.domain).ok())
                    .is_some_and(|snapshot| snapshot.committed_tokens > 0)
            });
            if restarted || committed || state.tensor_sequences.contains_key(session) {
                return Ok(None);
            }
        }
        self.probe_managed_prefix(runtime, request)
    }

    /// DS4: drain demotion work for one model — when a device arena's
    /// pressure crosses the high watermark, move LRU committed prefix
    /// subtrees to the host pool until the low watermark, the bounded step
    /// budget, the victim supply, or the host budget runs out. Bounded and
    /// synchronous: steps run inside the manager's own critical section.
    pub(super) fn run_host_offload_tick(
        policy: Option<HostOffloadPolicy>,
        state: &mut ManagedKvModelState,
        telemetry: &ManagedKvTelemetry,
    ) {
        fn pressure(coordinator: &KvCacheCoordinator) -> Option<f32> {
            let stats = coordinator.stats();
            (stats.capacity_pages > 0)
                .then(|| stats.allocated_pages as f32 / stats.capacity_pages as f32)
        }
        let Some(policy) = policy else { return };
        let Some(offload) = state.host_offload.as_mut() else {
            return;
        };
        let arena_ids: Vec<KvArenaId> = offload.arenas.keys().copied().collect();
        for arena_id in arena_ids {
            let Some(coordinator) = state.coordinators.get_mut(&arena_id) else {
                continue;
            };
            if pressure(coordinator).is_none_or(|current| current < policy.high_watermark) {
                continue;
            }
            let Some(arena) = state.runtime.arena(arena_id) else {
                continue;
            };
            let Some(index) = state.prefix_indexes.get_mut(&arena_id) else {
                continue;
            };
            let Some(arena_offload) = offload.arenas.get_mut(&arena_id) else {
                continue;
            };
            // A failure to reach the device thread demotes nothing.
            let _ = on_device_thread(arena.as_ref(), || {
                let mut steps = policy.max_in_flight_pages;
                while steps > 0 {
                    if pressure(coordinator)
                        .is_none_or(|current| current <= policy.low_watermark)
                    {
                        break;
                    }
                    match offload::demote_step(
                        coordinator,
                        index,
                        arena.as_ref(),
                        arena_offload,
                        telemetry,
                    ) {
                        Ok(DemoteOutcome::Demoted { .. }) => steps -= 1,
                        Ok(_) => break,
                        Err(_) => break,
                    }
                }
                Ok(())
            });
        }
    }
}
