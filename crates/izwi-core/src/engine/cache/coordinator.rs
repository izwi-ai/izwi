//! Transactional control plane for backend-owned physical KV pages.
//!
//! This module deliberately contains no tensors or device pointers. It owns
//! generation-safe identities, committed request tables, reservations, and the
//! reference/pin counts that determine when an arena slot may be reused.

use std::collections::{HashMap, HashSet, VecDeque};

use thiserror::Error;

use crate::backends::kv::KvPageCopy;
use crate::engine::execution::{PlanId, SessionKey};
use crate::kv::{CacheBlockRef, CacheDomainId, KvArenaId, KvGroupId};

use super::window::{plan_window_step, KvWindowError};

/// Committed block table for one compatible physical group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupBlockTable {
    pub group: KvGroupId,
    pub blocks: Vec<CacheBlockRef>,
}

/// Immutable view of one committed request/domain table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvSnapshot {
    pub arena: KvArenaId,
    pub session: SessionKey,
    pub domain: CacheDomainId,
    pub version: u64,
    pub committed_tokens: u32,
    pub window_start: u32,
    pub groups: Vec<GroupBlockTable>,
}

/// Desired source/ownership of one page in a provisional table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvBlockIntent {
    /// Retain a committed immutable page from the expected snapshot.
    Existing(CacheBlockRef),
    /// Exclusively reserve an existing private tail for in-place writes.
    Writable(CacheBlockRef),
    /// Acquire a committed immutable page held by the prefix index.
    Shared(CacheBlockRef),
    /// Allocate a new transaction-private page.
    Fresh,
    /// Allocate a private destination and copy an immutable committed source
    /// into it before any model writes are allowed.
    CopyOnWrite(CacheBlockRef),
}

/// Provisional table description for one physical group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvGroupReservation {
    pub group: KvGroupId,
    pub blocks: Vec<KvBlockIntent>,
}

/// Atomic reservation request based on an exact committed snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvReserveRequest {
    pub txn_id: PlanId,
    pub expected: KvSnapshot,
    pub target_committed_tokens: u32,
    pub target_window_start: u32,
    pub groups: Vec<KvGroupReservation>,
}

/// Model-neutral append plus logical-window rotation request.
///
/// Unlike `KvReserveRequest`, callers do not supply a replacement block table.
/// The coordinator derives it from the exact snapshot and page geometry, which
/// prevents early dereference or accidental retention of hidden leading pages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvWindowReserveRequest {
    pub txn_id: PlanId,
    pub expected: KvSnapshot,
    pub target_committed_tokens: u32,
    pub target_window_start: u32,
    pub page_tokens: u32,
}

/// Immutable execution metadata produced by `prepare`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvPreparedReservation {
    pub txn_id: PlanId,
    pub expected: KvSnapshot,
    pub provisional_groups: Vec<GroupBlockTable>,
    pub writable_blocks: Vec<CacheBlockRef>,
    pub page_copies: Vec<KvPageCopy>,
    pub target_committed_tokens: u32,
    pub target_window_start: u32,
}

/// Backend acknowledgement that all required K/V writes are complete.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvWriteReceipt {
    pub txn_id: PlanId,
    pub committed_tokens: u32,
    pub written_blocks: Vec<CacheBlockRef>,
}

/// Observable transaction state for diagnostics and tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvTransactionState {
    Reserved,
    Prepared,
    Written,
}

/// Terminal transaction disposition retained to reject duplicate reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvTerminalState {
    Committed,
    Aborted,
}

/// Number of exact terminal dispositions retained for diagnostics.
///
/// Duplicate/stale transaction rejection does not depend on this window: the
/// coordinator also keeps a constant-size high-water mark. The window only
/// preserves the committed-versus-aborted detail for recently completed work.
const TERMINAL_TRANSACTION_HISTORY_LIMIT: usize = 4_096;

#[derive(Debug)]
struct TerminalTransactionHistory {
    states: HashMap<PlanId, KvTerminalState>,
    completion_order: VecDeque<PlanId>,
    high_watermark: Option<PlanId>,
    limit: usize,
}

impl TerminalTransactionHistory {
    fn new(limit: usize) -> Self {
        assert!(limit > 0, "terminal history must retain at least one entry");
        Self {
            // Avoid eagerly reserving the full diagnostic window for every
            // model arena; these grow only as that arena completes work.
            states: HashMap::new(),
            completion_order: VecDeque::new(),
            high_watermark: None,
            limit,
        }
    }

    fn contains(&self, txn_id: PlanId) -> bool {
        // Scheduler PlanIds are globally monotonic on first admission. Active
        // transactions may finish out of order, so exact recent states remain
        // separate, but an ID at or below a completed high-water mark can only
        // be a delayed report, a stale reservation, or a reused exhausted ID.
        self.high_watermark.is_some_and(|high| txn_id <= high)
    }

    fn state(&self, txn_id: PlanId) -> Option<KvTerminalState> {
        self.states.get(&txn_id).copied()
    }

    fn record(&mut self, txn_id: PlanId, state: KvTerminalState) {
        self.high_watermark = Some(self.high_watermark.map_or(txn_id, |high| high.max(txn_id)));
        if self.states.insert(txn_id, state).is_none() {
            self.completion_order.push_back(txn_id);
        }
        while self.states.len() > self.limit {
            let expired = self
                .completion_order
                .pop_front()
                .expect("terminal history order matches retained states");
            self.states.remove(&expired);
        }
    }
}

/// Exact coordinator counters. Managed arena bytes are intentionally absent:
/// the resource authority accounts for physical backing once at arena scope.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvCoordinatorStats {
    pub capacity_pages: usize,
    pub allocated_pages: usize,
    pub free_pages: usize,
    pub table_refs: usize,
    pub prefix_refs: usize,
    pub execution_pins: usize,
    pub transfer_pins: usize,
    pub reservations: usize,
    pub active_transactions: usize,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum KvCoordinatorError {
    #[error("request/domain table is not registered")]
    MissingTable,
    #[error("request/domain table is already registered")]
    DuplicateTable,
    #[error("transaction {0} is active or has already reached a terminal state")]
    DuplicateTransaction(PlanId),
    #[error("transaction {0} is not active")]
    MissingTransaction(PlanId),
    #[error("transaction {txn_id} is in {actual:?}, expected {expected:?}")]
    InvalidTransactionState {
        txn_id: PlanId,
        expected: KvTransactionState,
        actual: KvTransactionState,
    },
    #[error("the expected request table snapshot is stale")]
    VersionConflict,
    #[error("cache block belongs to another arena generation")]
    WrongArena,
    #[error("cache block handle is stale or no longer allocated")]
    StaleBlock,
    #[error("cache block belongs to another physical group")]
    WrongGroup,
    #[error("cache block is duplicated in a provisional table")]
    DuplicateBlock,
    #[error("physical group is duplicated in a provisional table")]
    DuplicateGroup,
    #[error("existing block is not present in the expected group table")]
    BlockNotInSnapshot,
    #[error("shared block is not retained by the committed prefix index")]
    UnpublishedPrefix,
    #[error("cache block already has an exclusive writer")]
    WriteConflict,
    #[error("cache arena has insufficient free pages")]
    Capacity,
    #[error("committed token or window positions are invalid")]
    InvalidTokenRange,
    #[error("a sliding-window reservation requires an established physical group table")]
    MissingWindowGroups,
    #[error(transparent)]
    Window(#[from] KvWindowError),
    #[error("write receipt does not cover the transaction's exact writable set")]
    InvalidWriteReceipt,
    #[error("cache block has no matching reference or pin to release")]
    ReferenceUnderflow,
    #[error("request/domain table still has an active transaction")]
    ActiveTransaction,
    #[error("coordinator invariant failed: {0}")]
    Invariant(String),
}

pub type KvCoordinatorResult<T> = std::result::Result<T, KvCoordinatorError>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TableKey {
    session: SessionKey,
    domain: CacheDomainId,
}

impl TableKey {
    fn new(session: SessionKey, domain: CacheDomainId) -> Self {
        Self { session, domain }
    }
}

#[derive(Debug, Clone)]
struct BlockSlot {
    generation: u32,
    group: Option<KvGroupId>,
    allocated: bool,
    retired: bool,
    table_refs: usize,
    prefix_refs: usize,
    execution_pins: usize,
    transfer_pins: usize,
    reservations: usize,
    writer: Option<PlanId>,
}

impl Default for BlockSlot {
    fn default() -> Self {
        Self {
            generation: 0,
            group: None,
            allocated: false,
            retired: false,
            table_refs: 0,
            prefix_refs: 0,
            execution_pins: 0,
            transfer_pins: 0,
            reservations: 0,
            writer: None,
        }
    }
}

impl BlockSlot {
    fn ownership_count(&self) -> usize {
        self.table_refs
            + self.prefix_refs
            + self.execution_pins
            + self.transfer_pins
            + self.reservations
    }
}

#[derive(Debug, Clone, Copy)]
struct TransactionHold {
    block: CacheBlockRef,
    writer: bool,
}

#[derive(Debug, Clone)]
struct Transaction {
    id: PlanId,
    key: TableKey,
    expected: KvSnapshot,
    provisional_groups: Vec<GroupBlockTable>,
    writable_blocks: Vec<CacheBlockRef>,
    page_copies: Vec<KvPageCopy>,
    holds: Vec<TransactionHold>,
    target_committed_tokens: u32,
    target_window_start: u32,
    state: KvTransactionState,
}

/// Fully validated metadata delta. Construction performs every fallible CAS,
/// ownership, reference-count, and version check; applying it under the same
/// engine state lock is infallible.
pub(crate) struct KvCoordinatorCommitPlan {
    txn: Transaction,
    retain_prefix: Vec<CacheBlockRef>,
    release_prefix: Vec<CacheBlockRef>,
    old_blocks: HashSet<CacheBlockRef>,
    new_blocks: HashSet<CacheBlockRef>,
    next_version: u64,
}

/// Transactional metadata coordinator for one physical arena generation.
pub struct KvCacheCoordinator {
    arena: KvArenaId,
    slots: Vec<BlockSlot>,
    free: VecDeque<u32>,
    tables: HashMap<TableKey, KvSnapshot>,
    transactions: HashMap<PlanId, Transaction>,
    terminal_transactions: TerminalTransactionHistory,
}

impl KvCacheCoordinator {
    pub fn new(arena: KvArenaId, capacity_pages: usize) -> Self {
        let slots = vec![BlockSlot::default(); capacity_pages];
        let free = (0..capacity_pages as u32).collect();
        Self {
            arena,
            slots,
            free,
            tables: HashMap::new(),
            transactions: HashMap::new(),
            terminal_transactions: TerminalTransactionHistory::new(
                TERMINAL_TRANSACTION_HISTORY_LIMIT,
            ),
        }
    }

    pub fn arena(&self) -> KvArenaId {
        self.arena
    }

    pub fn register_table(
        &mut self,
        session: SessionKey,
        domain: CacheDomainId,
    ) -> KvCoordinatorResult<KvSnapshot> {
        let key = TableKey::new(session.clone(), domain);
        if self.tables.contains_key(&key) {
            return Err(KvCoordinatorError::DuplicateTable);
        }
        let snapshot = KvSnapshot {
            arena: self.arena,
            session,
            domain,
            version: 0,
            committed_tokens: 0,
            window_start: 0,
            groups: Vec::new(),
        };
        self.tables.insert(key, snapshot.clone());
        Ok(snapshot)
    }

    pub fn snapshot(
        &self,
        session: &SessionKey,
        domain: CacheDomainId,
    ) -> KvCoordinatorResult<KvSnapshot> {
        self.tables
            .get(&TableKey::new(session.clone(), domain))
            .cloned()
            .ok_or(KvCoordinatorError::MissingTable)
    }

    pub fn transaction_state(&self, txn_id: PlanId) -> Option<KvTransactionState> {
        self.transactions.get(&txn_id).map(|txn| txn.state)
    }

    /// Returns the exact disposition when `txn_id` remains in the bounded
    /// diagnostic window. `None` does not prove that the transaction was never
    /// completed; stale-ID rejection is retained independently.
    pub fn terminal_state(&self, txn_id: PlanId) -> Option<KvTerminalState> {
        self.terminal_transactions.state(txn_id)
    }

    /// Reserve an exact versioned append/window transition.
    ///
    /// The old table remains intact until commit, so abort is lossless. Pages
    /// before the retained suffix are dereferenced only by commit. Extending a
    /// partial tail uses in-place mutation only when it is exclusively owned;
    /// otherwise a transaction-private copy is reserved.
    pub fn reserve_window(&mut self, request: KvWindowReserveRequest) -> KvCoordinatorResult<()> {
        if request.expected.arena != self.arena {
            return Err(KvCoordinatorError::WrongArena);
        }
        let key = TableKey::new(request.expected.session.clone(), request.expected.domain);
        let current = self
            .tables
            .get(&key)
            .ok_or(KvCoordinatorError::MissingTable)?;
        if current != &request.expected {
            return Err(KvCoordinatorError::VersionConflict);
        }
        if request.expected.groups.is_empty() {
            return Err(KvCoordinatorError::MissingWindowGroups);
        }
        let page_plan = plan_window_step(
            request.expected.window_start,
            request.expected.committed_tokens,
            request.target_window_start,
            request.target_committed_tokens,
            request.page_tokens,
        )?;
        let expected_pages = page_plan.released_pages + page_plan.retained_pages;
        let mut groups = Vec::with_capacity(request.expected.groups.len());
        for expected_group in &request.expected.groups {
            if expected_group.blocks.len() != expected_pages {
                return Err(KvCoordinatorError::Window(KvWindowError::InvalidTable {
                    expected: expected_pages,
                    actual: expected_group.blocks.len(),
                }));
            }
            let retained = &expected_group.blocks[page_plan.released_pages..];
            let mut blocks = Vec::with_capacity(
                page_plan
                    .retained_pages
                    .saturating_add(page_plan.fresh_pages),
            );
            for (index, block) in retained.iter().copied().enumerate() {
                let intent = if page_plan.writable_retained_page == Some(index) {
                    self.validate_block(block, Some(expected_group.group))?;
                    let slot = &self.slots[block.index as usize];
                    if slot.writer.is_none()
                        && slot.table_refs == 1
                        && slot.prefix_refs == 0
                        && slot.execution_pins == 0
                        && slot.transfer_pins == 0
                        && slot.reservations == 0
                    {
                        KvBlockIntent::Writable(block)
                    } else {
                        KvBlockIntent::CopyOnWrite(block)
                    }
                } else {
                    KvBlockIntent::Existing(block)
                };
                blocks.push(intent);
            }
            blocks.extend(std::iter::repeat(KvBlockIntent::Fresh).take(page_plan.fresh_pages));
            groups.push(KvGroupReservation {
                group: expected_group.group,
                blocks,
            });
        }

        self.reserve(KvReserveRequest {
            txn_id: request.txn_id,
            expected: request.expected,
            target_committed_tokens: request.target_committed_tokens,
            target_window_start: request.target_window_start,
            groups,
        })
    }

    /// Atomically reserve shared/fresh pages and exclusive writable tails.
    pub fn reserve(&mut self, request: KvReserveRequest) -> KvCoordinatorResult<()> {
        if self.transactions.contains_key(&request.txn_id)
            || self.terminal_transactions.contains(request.txn_id)
        {
            return Err(KvCoordinatorError::DuplicateTransaction(request.txn_id));
        }
        if request.expected.arena != self.arena {
            return Err(KvCoordinatorError::WrongArena);
        }
        if request.target_committed_tokens < request.expected.committed_tokens
            || request.target_window_start < request.expected.window_start
            || request.target_window_start > request.target_committed_tokens
        {
            return Err(KvCoordinatorError::InvalidTokenRange);
        }

        let key = TableKey::new(request.expected.session.clone(), request.expected.domain);
        let current = self
            .tables
            .get(&key)
            .ok_or(KvCoordinatorError::MissingTable)?;
        if current != &request.expected {
            return Err(KvCoordinatorError::VersionConflict);
        }

        let mut groups_seen = HashSet::new();
        let mut blocks_seen = HashSet::new();
        let mut fresh_count = 0usize;
        let mut has_writable = false;
        let mut has_provisional_blocks = false;

        for group_request in &request.groups {
            if !groups_seen.insert(group_request.group) {
                return Err(KvCoordinatorError::DuplicateGroup);
            }
            let expected_group = request
                .expected
                .groups
                .iter()
                .find(|group| group.group == group_request.group);
            for intent in &group_request.blocks {
                has_provisional_blocks = true;
                match *intent {
                    KvBlockIntent::Fresh => {
                        fresh_count += 1;
                        has_writable = true;
                    }
                    KvBlockIntent::CopyOnWrite(block) => {
                        self.validate_block(block, Some(group_request.group))?;
                        if !blocks_seen.insert(block) {
                            return Err(KvCoordinatorError::DuplicateBlock);
                        }
                        if !expected_group.is_some_and(|group| group.blocks.contains(&block)) {
                            return Err(KvCoordinatorError::BlockNotInSnapshot);
                        }
                        let slot = &self.slots[block.index as usize];
                        if slot.writer.is_some() {
                            return Err(KvCoordinatorError::WriteConflict);
                        }
                        fresh_count += 1;
                        has_writable = true;
                    }
                    KvBlockIntent::Existing(block)
                    | KvBlockIntent::Writable(block)
                    | KvBlockIntent::Shared(block) => {
                        self.validate_block(block, Some(group_request.group))?;
                        if !blocks_seen.insert(block) {
                            return Err(KvCoordinatorError::DuplicateBlock);
                        }
                        match *intent {
                            KvBlockIntent::Existing(_) | KvBlockIntent::Writable(_) => {
                                if !expected_group
                                    .is_some_and(|group| group.blocks.contains(&block))
                                {
                                    return Err(KvCoordinatorError::BlockNotInSnapshot);
                                }
                            }
                            KvBlockIntent::Shared(_) => {
                                let slot = &self.slots[block.index as usize];
                                if slot.prefix_refs == 0 {
                                    return Err(KvCoordinatorError::UnpublishedPrefix);
                                }
                            }
                            KvBlockIntent::Fresh | KvBlockIntent::CopyOnWrite(_) => unreachable!(),
                        }
                        if matches!(intent, KvBlockIntent::Writable(_)) {
                            let slot = &self.slots[block.index as usize];
                            if slot.writer.is_some()
                                || slot.table_refs != 1
                                || slot.prefix_refs != 0
                                || slot.execution_pins != 0
                                || slot.transfer_pins != 0
                                || slot.reservations != 0
                            {
                                return Err(KvCoordinatorError::WriteConflict);
                            }
                            has_writable = true;
                        }
                    }
                }
            }
        }
        let reusable_pages = self
            .free
            .iter()
            .filter(|index| self.slots[**index as usize].generation < u32::MAX)
            .count();
        if fresh_count > reusable_pages {
            return Err(KvCoordinatorError::Capacity);
        }
        if request.target_committed_tokens > request.expected.committed_tokens
            && (request.target_window_start < request.target_committed_tokens
                || has_provisional_blocks)
            && !has_writable
        {
            return Err(KvCoordinatorError::InvalidTokenRange);
        }

        let mut provisional_groups = Vec::with_capacity(request.groups.len());
        let mut writable_blocks = Vec::new();
        let mut page_copies = Vec::new();
        let mut holds = Vec::new();

        for group_request in request.groups {
            let mut blocks = Vec::with_capacity(group_request.blocks.len());
            for intent in group_request.blocks {
                match intent {
                    KvBlockIntent::Existing(block) => blocks.push(block),
                    KvBlockIntent::Writable(block) => {
                        let slot = &mut self.slots[block.index as usize];
                        slot.reservations += 1;
                        slot.writer = Some(request.txn_id);
                        holds.push(TransactionHold {
                            block,
                            writer: true,
                        });
                        writable_blocks.push(block);
                        blocks.push(block);
                    }
                    KvBlockIntent::Shared(block) => {
                        self.slots[block.index as usize].reservations += 1;
                        holds.push(TransactionHold {
                            block,
                            writer: false,
                        });
                        blocks.push(block);
                    }
                    KvBlockIntent::Fresh => {
                        let block = self.allocate_block(group_request.group, request.txn_id);
                        holds.push(TransactionHold {
                            block,
                            writer: true,
                        });
                        writable_blocks.push(block);
                        blocks.push(block);
                    }
                    KvBlockIntent::CopyOnWrite(source) => {
                        let destination = self.allocate_block(group_request.group, request.txn_id);
                        holds.push(TransactionHold {
                            block: destination,
                            writer: true,
                        });
                        writable_blocks.push(destination);
                        page_copies.push(KvPageCopy {
                            source,
                            destination,
                        });
                        blocks.push(destination);
                    }
                }
            }
            provisional_groups.push(GroupBlockTable {
                group: group_request.group,
                blocks,
            });
        }

        self.transactions.insert(
            request.txn_id,
            Transaction {
                id: request.txn_id,
                key,
                expected: request.expected,
                provisional_groups,
                writable_blocks,
                page_copies,
                holds,
                target_committed_tokens: request.target_committed_tokens,
                target_window_start: request.target_window_start,
                state: KvTransactionState::Reserved,
            },
        );
        Ok(())
    }

    /// Pin all source/destination pages and return immutable execution metadata.
    pub fn prepare(&mut self, txn_id: PlanId) -> KvCoordinatorResult<KvPreparedReservation> {
        let txn = self
            .transactions
            .get(&txn_id)
            .ok_or(KvCoordinatorError::MissingTransaction(txn_id))?;
        if txn.state != KvTransactionState::Reserved {
            return Err(KvCoordinatorError::InvalidTransactionState {
                txn_id,
                expected: KvTransactionState::Reserved,
                actual: txn.state,
            });
        }
        let pages = transaction_execution_blocks(txn);
        let writable: HashSet<_> = txn.writable_blocks.iter().copied().collect();
        for block in &pages {
            self.validate_block(*block, None)?;
            if self.slots[block.index as usize]
                .writer
                .is_some_and(|writer| writer != txn_id || !writable.contains(block))
            {
                return Err(KvCoordinatorError::WriteConflict);
            }
        }
        for block in pages {
            self.slots[block.index as usize].execution_pins += 1;
        }
        let txn = self
            .transactions
            .get_mut(&txn_id)
            .expect("transaction exists");
        txn.state = KvTransactionState::Prepared;
        Ok(KvPreparedReservation {
            txn_id,
            expected: txn.expected.clone(),
            provisional_groups: txn.provisional_groups.clone(),
            writable_blocks: txn.writable_blocks.clone(),
            page_copies: txn.page_copies.clone(),
            target_committed_tokens: txn.target_committed_tokens,
            target_window_start: txn.target_window_start,
        })
    }

    /// Validate an exact backend write receipt and make the transaction committable.
    pub fn complete_write(&mut self, receipt: KvWriteReceipt) -> KvCoordinatorResult<()> {
        let txn = self
            .transactions
            .get_mut(&receipt.txn_id)
            .ok_or(KvCoordinatorError::MissingTransaction(receipt.txn_id))?;
        if txn.state != KvTransactionState::Prepared {
            return Err(KvCoordinatorError::InvalidTransactionState {
                txn_id: receipt.txn_id,
                expected: KvTransactionState::Prepared,
                actual: txn.state,
            });
        }
        let expected: HashSet<_> = txn.writable_blocks.iter().copied().collect();
        let written: HashSet<_> = receipt.written_blocks.iter().copied().collect();
        if receipt.committed_tokens != txn.target_committed_tokens
            || written.len() != receipt.written_blocks.len()
            || written != expected
        {
            return Err(KvCoordinatorError::InvalidWriteReceipt);
        }
        txn.state = KvTransactionState::Written;
        Ok(())
    }

    /// CAS the expected table version and atomically publish optional full pages.
    pub fn commit(
        &mut self,
        txn_id: PlanId,
        publish_prefix: &[CacheBlockRef],
    ) -> KvCoordinatorResult<KvSnapshot> {
        self.commit_with_prefix_updates(txn_id, publish_prefix, &[])
    }

    /// CAS the table and apply prefix-index ownership changes as one metadata
    /// transaction. Callers stage their index mutation first and make it
    /// visible only after this method succeeds.
    pub fn commit_with_prefix_updates(
        &mut self,
        txn_id: PlanId,
        retain_prefix: &[CacheBlockRef],
        release_prefix: &[CacheBlockRef],
    ) -> KvCoordinatorResult<KvSnapshot> {
        let plan =
            match self.stage_commit_with_prefix_updates(txn_id, retain_prefix, release_prefix) {
                Err(KvCoordinatorError::VersionConflict) => {
                    self.abort_internal(txn_id);
                    return Err(KvCoordinatorError::VersionConflict);
                }
                Err(error) => return Err(error),
                Ok(plan) => plan,
            };
        Ok(self.apply_staged_commit(plan))
    }

    /// Validate a written transaction without publishing any table or
    /// reference-count mutation.
    pub(crate) fn stage_commit_with_prefix_updates(
        &self,
        txn_id: PlanId,
        retain_prefix: &[CacheBlockRef],
        release_prefix: &[CacheBlockRef],
    ) -> KvCoordinatorResult<KvCoordinatorCommitPlan> {
        let txn = self
            .transactions
            .get(&txn_id)
            .cloned()
            .ok_or(KvCoordinatorError::MissingTransaction(txn_id))?;
        if txn.state != KvTransactionState::Written {
            return Err(KvCoordinatorError::InvalidTransactionState {
                txn_id,
                expected: KvTransactionState::Written,
                actual: txn.state,
            });
        }
        let current = self
            .tables
            .get(&txn.key)
            .ok_or(KvCoordinatorError::MissingTable)?;
        if current != &txn.expected {
            return Err(KvCoordinatorError::VersionConflict);
        }

        let writable: HashSet<_> = txn.writable_blocks.iter().copied().collect();
        let mut published = HashSet::new();
        for block in retain_prefix {
            self.validate_block(*block, None)?;
            if !writable.contains(block) || !published.insert(*block) {
                return Err(KvCoordinatorError::InvalidWriteReceipt);
            }
        }
        let released = unique_blocks(release_prefix)?;
        for block in &released {
            self.validate_block(*block, None)?;
            let retained_here = usize::from(published.contains(block));
            if self.slots[block.index as usize].prefix_refs + retained_here == 0 {
                return Err(KvCoordinatorError::ReferenceUnderflow);
            }
        }

        let old_blocks: HashSet<_> = unique_table_blocks(&txn.expected.groups)
            .into_iter()
            .collect();
        let new_blocks: HashSet<_> = unique_table_blocks(&txn.provisional_groups)
            .into_iter()
            .collect();
        for block in &new_blocks {
            self.validate_block(*block, None)?;
        }
        for block in old_blocks.difference(&new_blocks) {
            if self.slots[block.index as usize].table_refs == 0 {
                return Err(KvCoordinatorError::ReferenceUnderflow);
            }
        }
        for block in new_blocks.difference(&old_blocks) {
            self.slots[block.index as usize]
                .table_refs
                .checked_add(1)
                .ok_or_else(|| {
                    KvCoordinatorError::Invariant("table reference overflow".to_string())
                })?;
        }
        for block in retain_prefix {
            self.slots[block.index as usize]
                .prefix_refs
                .checked_add(1)
                .ok_or_else(|| {
                    KvCoordinatorError::Invariant("prefix reference overflow".to_string())
                })?;
        }
        let next_version = txn.expected.version.checked_add(1).ok_or_else(|| {
            KvCoordinatorError::Invariant("request table version overflow".to_string())
        })?;

        Ok(KvCoordinatorCommitPlan {
            txn,
            retain_prefix: retain_prefix.to_vec(),
            release_prefix: released,
            old_blocks,
            new_blocks,
            next_version,
        })
    }

    /// Apply a plan produced by [`Self::stage_commit_with_prefix_updates`].
    /// The engine holds the coordinator state lock between staging and apply.
    pub(crate) fn apply_staged_commit(&mut self, plan: KvCoordinatorCommitPlan) -> KvSnapshot {
        let KvCoordinatorCommitPlan {
            txn,
            retain_prefix,
            release_prefix,
            old_blocks,
            new_blocks,
            next_version,
        } = plan;
        // Add new ownership before releasing reservations or removed table refs.
        for block in new_blocks.difference(&old_blocks) {
            self.slots[block.index as usize].table_refs += 1;
        }
        for block in &retain_prefix {
            self.slots[block.index as usize].prefix_refs += 1;
        }
        for block in &release_prefix {
            self.slots[block.index as usize].prefix_refs -= 1;
        }

        self.release_transaction_pins(&txn);
        self.release_transaction_holds(&txn);
        for block in old_blocks.difference(&new_blocks) {
            let slot = &mut self.slots[block.index as usize];
            slot.table_refs -= 1;
            self.recycle_if_unowned(block.index);
        }
        for block in release_prefix {
            self.recycle_if_unowned(block.index);
        }

        let txn_id = txn.id;
        let committed = KvSnapshot {
            arena: self.arena,
            session: txn.expected.session,
            domain: txn.expected.domain,
            version: next_version,
            committed_tokens: txn.target_committed_tokens,
            window_start: txn.target_window_start,
            groups: txn.provisional_groups,
        };
        self.tables.insert(txn.key, committed.clone());
        self.transactions.remove(&txn_id);
        self.terminal_transactions
            .record(txn_id, KvTerminalState::Committed);
        committed
    }

    /// Idempotently abort a reservation and release all pins/private ownership.
    pub fn abort(&mut self, txn_id: PlanId) -> KvCoordinatorResult<bool> {
        if self.transactions.contains_key(&txn_id) {
            self.abort_internal(txn_id);
            return Ok(true);
        }
        if self.terminal_transactions.contains(txn_id) {
            return Ok(false);
        }
        Err(KvCoordinatorError::MissingTransaction(txn_id))
    }

    /// Preflight table release without mutating page ownership.
    pub fn validate_table_release(
        &self,
        session: &SessionKey,
        domain: CacheDomainId,
    ) -> KvCoordinatorResult<()> {
        let key = TableKey::new(session.clone(), domain);
        if self.transactions.values().any(|txn| txn.key == key) {
            return Err(KvCoordinatorError::ActiveTransaction);
        }
        let snapshot = self
            .tables
            .get(&key)
            .ok_or(KvCoordinatorError::MissingTable)?;
        for block in unique_table_blocks(&snapshot.groups) {
            self.validate_block(block, None)?;
            if self.slots[block.index as usize].table_refs == 0 {
                return Err(KvCoordinatorError::ReferenceUnderflow);
            }
        }
        Ok(())
    }

    /// Release a completed request table. Active reservations must abort first.
    pub fn release_table(
        &mut self,
        session: &SessionKey,
        domain: CacheDomainId,
    ) -> KvCoordinatorResult<()> {
        self.validate_table_release(session, domain)?;
        let key = TableKey::new(session.clone(), domain);
        let snapshot = self
            .tables
            .get(&key)
            .cloned()
            .ok_or(KvCoordinatorError::MissingTable)?;
        let blocks = unique_table_blocks(&snapshot.groups);
        self.tables.remove(&key);
        for block in blocks {
            self.validate_block(block, None)?;
            let slot = &mut self.slots[block.index as usize];
            slot.table_refs -= 1;
            self.recycle_if_unowned(block.index);
        }
        Ok(())
    }

    /// Add a durable prefix-index reference to a committed page.
    pub fn retain_prefix(&mut self, block: CacheBlockRef) -> KvCoordinatorResult<()> {
        self.validate_block(block, None)?;
        let slot = &mut self.slots[block.index as usize];
        if slot.table_refs == 0 && slot.prefix_refs == 0 {
            return Err(KvCoordinatorError::UnpublishedPrefix);
        }
        if slot.prefix_refs == 0 && (slot.writer.is_some() || slot.reservations != 0) {
            return Err(KvCoordinatorError::WriteConflict);
        }
        slot.prefix_refs += 1;
        Ok(())
    }

    pub fn release_prefix(&mut self, block: CacheBlockRef) -> KvCoordinatorResult<()> {
        self.release_prefixes(&[block])
    }

    /// Atomically release a set of durable prefix-index references.
    pub fn release_prefixes(&mut self, blocks: &[CacheBlockRef]) -> KvCoordinatorResult<()> {
        let blocks = unique_blocks(blocks)?;
        for block in &blocks {
            self.validate_block(*block, None)?;
            if self.slots[block.index as usize].prefix_refs == 0 {
                return Err(KvCoordinatorError::ReferenceUnderflow);
            }
        }
        for block in blocks {
            self.slots[block.index as usize].prefix_refs -= 1;
            self.recycle_if_unowned(block.index);
        }
        Ok(())
    }

    /// Pin pages across an acknowledged host/device transfer.
    pub fn pin_transfer(&mut self, blocks: &[CacheBlockRef]) -> KvCoordinatorResult<()> {
        let unique = unique_blocks(blocks)?;
        for block in &unique {
            self.validate_block(*block, None)?;
        }
        for block in unique {
            self.slots[block.index as usize].transfer_pins += 1;
        }
        Ok(())
    }

    pub fn unpin_transfer(&mut self, blocks: &[CacheBlockRef]) -> KvCoordinatorResult<()> {
        let unique = unique_blocks(blocks)?;
        for block in &unique {
            self.validate_block(*block, None)?;
            if self.slots[block.index as usize].transfer_pins == 0 {
                return Err(KvCoordinatorError::ReferenceUnderflow);
            }
        }
        for block in unique {
            self.slots[block.index as usize].transfer_pins -= 1;
            self.recycle_if_unowned(block.index);
        }
        Ok(())
    }

    pub fn stats(&self) -> KvCoordinatorStats {
        let allocated_pages = self.slots.iter().filter(|slot| slot.allocated).count();
        KvCoordinatorStats {
            capacity_pages: self.slots.len(),
            allocated_pages,
            free_pages: self.free.len(),
            table_refs: self.slots.iter().map(|slot| slot.table_refs).sum(),
            prefix_refs: self.slots.iter().map(|slot| slot.prefix_refs).sum(),
            execution_pins: self.slots.iter().map(|slot| slot.execution_pins).sum(),
            transfer_pins: self.slots.iter().map(|slot| slot.transfer_pins).sum(),
            reservations: self.slots.iter().map(|slot| slot.reservations).sum(),
            active_transactions: self.transactions.len(),
        }
    }

    /// Expensive internal consistency check intended for tests/shadow diagnostics.
    pub fn check_invariants(&self) -> KvCoordinatorResult<()> {
        let free: HashSet<_> = self.free.iter().copied().collect();
        if free.len() != self.free.len() {
            return Err(KvCoordinatorError::Invariant(
                "free list contains duplicate slots".to_string(),
            ));
        }
        let mut table_refs = vec![0usize; self.slots.len()];
        for snapshot in self.tables.values() {
            if snapshot.arena != self.arena {
                return Err(KvCoordinatorError::Invariant(
                    "table belongs to another arena".to_string(),
                ));
            }
            for block in unique_table_blocks(&snapshot.groups) {
                self.validate_block(block, None)?;
                table_refs[block.index as usize] += 1;
            }
        }
        let mut reservations = vec![0usize; self.slots.len()];
        let mut execution_pins = vec![0usize; self.slots.len()];
        let mut writers = vec![None; self.slots.len()];
        for txn in self.transactions.values() {
            for hold in &txn.holds {
                self.validate_block(hold.block, None)?;
                reservations[hold.block.index as usize] += 1;
                if hold.writer {
                    let writer = &mut writers[hold.block.index as usize];
                    if writer.replace(txn.id).is_some() {
                        return Err(KvCoordinatorError::Invariant(
                            "slot has multiple transaction writers".to_string(),
                        ));
                    }
                }
            }
            if matches!(
                txn.state,
                KvTransactionState::Prepared | KvTransactionState::Written
            ) {
                for block in transaction_execution_blocks(txn) {
                    execution_pins[block.index as usize] += 1;
                }
            }
        }
        for (index, slot) in self.slots.iter().enumerate() {
            let on_free_list = free.contains(&(index as u32));
            let allocation_state_valid = if slot.retired {
                !slot.allocated && !on_free_list
            } else if slot.allocated {
                !on_free_list
            } else {
                on_free_list
            };
            if !allocation_state_valid {
                return Err(KvCoordinatorError::Invariant(format!(
                    "slot {index} allocation/free-list disagreement"
                )));
            }
            if slot.table_refs != table_refs[index]
                || slot.reservations != reservations[index]
                || slot.execution_pins != execution_pins[index]
                || slot.writer != writers[index]
            {
                return Err(KvCoordinatorError::Invariant(format!(
                    "slot {index} reference accounting drift"
                )));
            }
            if !slot.allocated && slot.ownership_count() != 0 {
                return Err(KvCoordinatorError::Invariant(format!(
                    "free slot {index} retains ownership"
                )));
            }
            if slot.allocated && slot.ownership_count() == 0 {
                return Err(KvCoordinatorError::Invariant(format!(
                    "allocated slot {index} has no owner"
                )));
            }
        }
        Ok(())
    }

    fn allocate_block(&mut self, group: KvGroupId, txn_id: PlanId) -> CacheBlockRef {
        let index = self.free.pop_back().expect("capacity prevalidated");
        let slot = &mut self.slots[index as usize];
        slot.generation = slot
            .generation
            .checked_add(1)
            .expect("exhausted slots are excluded from capacity");
        slot.group = Some(group);
        slot.allocated = true;
        slot.retired = false;
        slot.reservations = 1;
        slot.writer = Some(txn_id);
        CacheBlockRef {
            arena: self.arena,
            group,
            index,
            slot_generation: slot.generation,
        }
    }

    fn validate_block(
        &self,
        block: CacheBlockRef,
        expected_group: Option<KvGroupId>,
    ) -> KvCoordinatorResult<()> {
        if block.arena != self.arena {
            return Err(KvCoordinatorError::WrongArena);
        }
        let Some(slot) = self.slots.get(block.index as usize) else {
            return Err(KvCoordinatorError::StaleBlock);
        };
        if !slot.allocated || slot.generation != block.slot_generation {
            return Err(KvCoordinatorError::StaleBlock);
        }
        if slot.group != Some(block.group) {
            return Err(KvCoordinatorError::WrongGroup);
        }
        if expected_group.is_some_and(|group| group != block.group) {
            return Err(KvCoordinatorError::WrongGroup);
        }
        Ok(())
    }

    fn abort_internal(&mut self, txn_id: PlanId) {
        let txn = self
            .transactions
            .remove(&txn_id)
            .expect("active transaction prevalidated");
        self.release_transaction_pins(&txn);
        self.release_transaction_holds(&txn);
        self.terminal_transactions
            .record(txn_id, KvTerminalState::Aborted);
    }

    fn release_transaction_pins(&mut self, txn: &Transaction) {
        if !matches!(
            txn.state,
            KvTransactionState::Prepared | KvTransactionState::Written
        ) {
            return;
        }
        for block in transaction_execution_blocks(txn) {
            let slot = &mut self.slots[block.index as usize];
            slot.execution_pins = slot
                .execution_pins
                .checked_sub(1)
                .expect("prepared transaction owns an execution pin");
            self.recycle_if_unowned(block.index);
        }
    }

    fn release_transaction_holds(&mut self, txn: &Transaction) {
        for hold in &txn.holds {
            let slot = &mut self.slots[hold.block.index as usize];
            slot.reservations = slot
                .reservations
                .checked_sub(1)
                .expect("active transaction owns a reservation");
            if hold.writer && slot.writer == Some(txn.id) {
                slot.writer = None;
            }
            self.recycle_if_unowned(hold.block.index);
        }
    }

    fn recycle_if_unowned(&mut self, index: u32) {
        let slot = &mut self.slots[index as usize];
        if slot.allocated && slot.ownership_count() == 0 {
            slot.allocated = false;
            slot.group = None;
            slot.writer = None;
            if slot.generation == u32::MAX {
                slot.retired = true;
            } else {
                self.free.push_back(index);
            }
        }
    }
}

fn unique_table_blocks(groups: &[GroupBlockTable]) -> Vec<CacheBlockRef> {
    let mut seen = HashSet::new();
    groups
        .iter()
        .flat_map(|group| group.blocks.iter().copied())
        .filter(|block| seen.insert(*block))
        .collect()
}

fn transaction_execution_blocks(txn: &Transaction) -> Vec<CacheBlockRef> {
    let mut blocks = unique_table_blocks(&txn.provisional_groups);
    let mut seen = blocks.iter().copied().collect::<HashSet<_>>();
    blocks.extend(
        txn.page_copies
            .iter()
            .map(|copy| copy.source)
            .filter(|block| seen.insert(*block)),
    );
    blocks
}

fn unique_blocks(blocks: &[CacheBlockRef]) -> KvCoordinatorResult<Vec<CacheBlockRef>> {
    let unique: HashSet<_> = blocks.iter().copied().collect();
    if unique.len() != blocks.len() {
        return Err(KvCoordinatorError::DuplicateBlock);
    }
    Ok(blocks.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, IndexOp, Tensor};

    use crate::backends::kv::{CpuKvArena, KvArena, KvArenaConfig, KvLayerConfig, KvWriteArgs};
    use crate::backends::BackendKind;
    use crate::engine::cache::window::advance_window;
    use crate::engine::ModelInstanceId;
    use crate::kv::{KvLayerBinding, KvSlotRef};

    fn arena(generation: u32) -> KvArenaId {
        KvArenaId {
            model_instance: ModelInstanceId::new(7),
            backend: BackendKind::Cpu,
            device_ordinal: None,
            generation,
        }
    }

    fn session(name: &str, epoch: u64) -> SessionKey {
        SessionKey::new(name.to_string(), epoch)
    }

    fn reserve_fresh(
        coordinator: &mut KvCacheCoordinator,
        txn_id: PlanId,
        snapshot: KvSnapshot,
        pages: usize,
        tokens: u32,
    ) {
        coordinator
            .reserve(KvReserveRequest {
                txn_id,
                expected: snapshot,
                target_committed_tokens: tokens,
                target_window_start: 0,
                groups: vec![KvGroupReservation {
                    group: KvGroupId::new(0),
                    blocks: vec![KvBlockIntent::Fresh; pages],
                }],
            })
            .unwrap();
    }

    fn prepare_and_complete(
        coordinator: &mut KvCacheCoordinator,
        txn_id: PlanId,
    ) -> KvPreparedReservation {
        let prepared = coordinator.prepare(txn_id).unwrap();
        coordinator
            .complete_write(KvWriteReceipt {
                txn_id,
                committed_tokens: prepared.target_committed_tokens,
                written_blocks: prepared.writable_blocks.clone(),
            })
            .unwrap();
        prepared
    }

    #[test]
    fn reserve_prepare_commit_and_release_conserve_pages() {
        let mut coordinator = KvCacheCoordinator::new(arena(1), 4);
        let key = session("a", 1);
        let initial = coordinator
            .register_table(key.clone(), CacheDomainId::new(0))
            .unwrap();
        reserve_fresh(&mut coordinator, 10, initial, 2, 17);
        assert_eq!(coordinator.stats().reservations, 2);
        let prepared = prepare_and_complete(&mut coordinator, 10);
        let committed = coordinator
            .commit(10, &[prepared.writable_blocks[0]])
            .unwrap();

        assert_eq!(committed.version, 1);
        assert_eq!(committed.committed_tokens, 17);
        assert_eq!(coordinator.stats().table_refs, 2);
        assert_eq!(coordinator.stats().prefix_refs, 1);
        assert_eq!(coordinator.stats().reservations, 0);
        coordinator.check_invariants().unwrap();

        coordinator
            .release_table(&key, CacheDomainId::new(0))
            .unwrap();
        assert_eq!(coordinator.stats().allocated_pages, 1);
        coordinator
            .release_prefix(prepared.writable_blocks[0])
            .unwrap();
        assert_eq!(coordinator.stats().allocated_pages, 0);
        assert_eq!(coordinator.stats().free_pages, 4);
        coordinator.check_invariants().unwrap();
    }

    #[test]
    fn abort_is_idempotent_at_every_precommit_phase() {
        for phase in 0..3 {
            let mut coordinator = KvCacheCoordinator::new(arena(1), 2);
            let initial = coordinator
                .register_table(session("abort", phase), CacheDomainId::new(0))
                .unwrap();
            reserve_fresh(&mut coordinator, 20 + phase, initial, 1, 1);
            if phase >= 1 {
                let prepared = coordinator.prepare(20 + phase).unwrap();
                if phase == 2 {
                    coordinator
                        .complete_write(KvWriteReceipt {
                            txn_id: 20 + phase,
                            committed_tokens: 1,
                            written_blocks: prepared.writable_blocks,
                        })
                        .unwrap();
                }
            }
            assert!(coordinator.abort(20 + phase).unwrap());
            assert!(!coordinator.abort(20 + phase).unwrap());
            assert_eq!(
                coordinator.terminal_state(20 + phase),
                Some(KvTerminalState::Aborted)
            );
            assert_eq!(coordinator.stats().allocated_pages, 0);
            coordinator.check_invariants().unwrap();
        }
    }

    #[test]
    fn terminal_history_is_bounded_without_allowing_stale_transaction_reuse() {
        let mut coordinator = KvCacheCoordinator::new(arena(1), 1);
        coordinator.terminal_transactions = TerminalTransactionHistory::new(2);
        let initial = coordinator
            .register_table(session("terminal-history", 1), CacheDomainId::new(0))
            .unwrap();

        for txn_id in 10..=12 {
            reserve_fresh(&mut coordinator, txn_id, initial.clone(), 1, 1);
            assert!(coordinator.abort(txn_id).unwrap());
        }

        assert_eq!(coordinator.terminal_transactions.states.len(), 2);
        assert_eq!(coordinator.terminal_state(10), None);
        assert_eq!(
            coordinator.terminal_state(11),
            Some(KvTerminalState::Aborted)
        );
        assert_eq!(
            coordinator.terminal_state(12),
            Some(KvTerminalState::Aborted)
        );

        // Exact diagnostics may expire, but the high-water mark makes delayed
        // aborts idempotent and permanently rejects a reused scheduler plan ID.
        assert!(!coordinator.abort(10).unwrap());
        assert_eq!(
            coordinator.reserve(KvReserveRequest {
                txn_id: 10,
                expected: initial,
                target_committed_tokens: 1,
                target_window_start: 0,
                groups: vec![KvGroupReservation {
                    group: KvGroupId::new(0),
                    blocks: vec![KvBlockIntent::Fresh],
                }],
            }),
            Err(KvCoordinatorError::DuplicateTransaction(10))
        );
    }

    #[test]
    fn terminal_high_watermark_tolerates_out_of_order_completion() {
        let mut coordinator = KvCacheCoordinator::new(arena(1), 2);
        let initial = coordinator
            .register_table(session("out-of-order-terminal", 1), CacheDomainId::new(0))
            .unwrap();
        reserve_fresh(&mut coordinator, 20, initial.clone(), 1, 1);
        reserve_fresh(&mut coordinator, 21, initial, 1, 1);

        assert!(coordinator.abort(21).unwrap());
        assert!(coordinator.abort(20).unwrap());
        assert!(!coordinator.abort(19).unwrap());
        assert_eq!(
            coordinator.abort(22),
            Err(KvCoordinatorError::MissingTransaction(22))
        );
        let current = coordinator
            .snapshot(&session("out-of-order-terminal", 1), CacheDomainId::new(0))
            .unwrap();
        reserve_fresh(&mut coordinator, 22, current, 1, 1);
        assert!(coordinator.abort(22).unwrap());
        coordinator.check_invariants().unwrap();
    }

    #[test]
    fn stale_snapshot_loses_without_mutating_committed_table() {
        let mut coordinator = KvCacheCoordinator::new(arena(1), 4);
        let key = session("conflict", 1);
        let initial = coordinator
            .register_table(key.clone(), CacheDomainId::new(0))
            .unwrap();
        reserve_fresh(&mut coordinator, 31, initial.clone(), 1, 1);
        reserve_fresh(&mut coordinator, 32, initial, 1, 1);
        prepare_and_complete(&mut coordinator, 31);
        prepare_and_complete(&mut coordinator, 32);
        let winner = coordinator.commit(31, &[]).unwrap();
        assert_eq!(
            coordinator.commit(32, &[]),
            Err(KvCoordinatorError::VersionConflict)
        );
        assert_eq!(
            coordinator.snapshot(&key, CacheDomainId::new(0)).unwrap(),
            winner
        );
        assert_eq!(
            coordinator.terminal_state(32),
            Some(KvTerminalState::Aborted)
        );
        assert_eq!(coordinator.stats().allocated_pages, 1);
        coordinator.check_invariants().unwrap();
    }

    #[test]
    fn stale_generation_cannot_free_pin_share_or_enter_a_table() {
        let mut coordinator = KvCacheCoordinator::new(arena(1), 1);
        let first_key = session("first", 1);
        let first = coordinator
            .register_table(first_key.clone(), CacheDomainId::new(0))
            .unwrap();
        reserve_fresh(&mut coordinator, 40, first, 1, 1);
        let old_block = prepare_and_complete(&mut coordinator, 40).writable_blocks[0];
        coordinator.commit(40, &[]).unwrap();
        coordinator
            .release_table(&first_key, CacheDomainId::new(0))
            .unwrap();

        let second_key = session("second", 1);
        let second = coordinator
            .register_table(second_key, CacheDomainId::new(0))
            .unwrap();
        reserve_fresh(&mut coordinator, 41, second, 1, 1);
        let new_block = coordinator.prepare(41).unwrap().writable_blocks[0];
        assert_eq!(old_block.index, new_block.index);
        assert_ne!(old_block.slot_generation, new_block.slot_generation);
        assert_eq!(
            coordinator.pin_transfer(&[old_block]),
            Err(KvCoordinatorError::StaleBlock)
        );
        assert_eq!(
            coordinator.release_prefix(old_block),
            Err(KvCoordinatorError::StaleBlock)
        );
        coordinator.abort(41).unwrap();
        coordinator.check_invariants().unwrap();
    }

    #[test]
    fn transfer_pin_defers_reuse_until_acknowledged() {
        let mut coordinator = KvCacheCoordinator::new(arena(1), 1);
        let key = session("transfer", 1);
        let initial = coordinator
            .register_table(key.clone(), CacheDomainId::new(0))
            .unwrap();
        reserve_fresh(&mut coordinator, 50, initial, 1, 1);
        let block = prepare_and_complete(&mut coordinator, 50).writable_blocks[0];
        coordinator.commit(50, &[]).unwrap();
        coordinator.pin_transfer(&[block]).unwrap();
        coordinator
            .release_table(&key, CacheDomainId::new(0))
            .unwrap();
        assert_eq!(coordinator.stats().allocated_pages, 1);
        assert_eq!(coordinator.stats().free_pages, 0);
        coordinator.unpin_transfer(&[block]).unwrap();
        assert_eq!(coordinator.stats().allocated_pages, 0);
        coordinator.check_invariants().unwrap();
    }

    #[test]
    fn published_prefix_can_be_shared_but_private_page_cannot() {
        let mut coordinator = KvCacheCoordinator::new(arena(1), 3);
        let owner = session("owner", 1);
        let owner_initial = coordinator
            .register_table(owner, CacheDomainId::new(0))
            .unwrap();
        reserve_fresh(&mut coordinator, 60, owner_initial, 2, 2);
        let prepared = prepare_and_complete(&mut coordinator, 60);
        let published = prepared.writable_blocks[0];
        let private = prepared.writable_blocks[1];
        coordinator.commit(60, &[published]).unwrap();

        let borrower = session("borrower", 1);
        let borrower_initial = coordinator
            .register_table(borrower, CacheDomainId::new(0))
            .unwrap();
        let request = |block| KvReserveRequest {
            txn_id: 61,
            expected: borrower_initial.clone(),
            target_committed_tokens: 0,
            target_window_start: 0,
            groups: vec![KvGroupReservation {
                group: KvGroupId::new(0),
                blocks: vec![KvBlockIntent::Shared(block)],
            }],
        };
        assert_eq!(
            coordinator.reserve(request(private)),
            Err(KvCoordinatorError::UnpublishedPrefix)
        );
        coordinator.reserve(request(published)).unwrap();
        coordinator.abort(61).unwrap();
        coordinator.check_invariants().unwrap();
    }

    #[test]
    fn writable_tail_is_exclusive_before_physical_write() {
        let mut coordinator = KvCacheCoordinator::new(arena(1), 2);
        let key = session("tail", 1);
        let initial = coordinator
            .register_table(key, CacheDomainId::new(0))
            .unwrap();
        reserve_fresh(&mut coordinator, 70, initial, 1, 1);
        let tail = prepare_and_complete(&mut coordinator, 70).writable_blocks[0];
        let committed = coordinator.commit(70, &[]).unwrap();

        let append = |txn_id| KvReserveRequest {
            txn_id,
            expected: committed.clone(),
            target_committed_tokens: 2,
            target_window_start: 0,
            groups: vec![KvGroupReservation {
                group: KvGroupId::new(0),
                blocks: vec![KvBlockIntent::Writable(tail)],
            }],
        };
        coordinator.reserve(append(71)).unwrap();
        assert_eq!(
            coordinator.reserve(append(72)),
            Err(KvCoordinatorError::WriteConflict)
        );
        coordinator.abort(71).unwrap();
        coordinator.reserve(append(72)).unwrap();
        coordinator.abort(72).unwrap();
        coordinator.check_invariants().unwrap();
    }

    #[test]
    fn malformed_receipt_cannot_advance_table() {
        let mut coordinator = KvCacheCoordinator::new(arena(1), 2);
        let key = session("receipt", 1);
        let initial = coordinator
            .register_table(key.clone(), CacheDomainId::new(0))
            .unwrap();
        reserve_fresh(&mut coordinator, 80, initial.clone(), 1, 1);
        let prepared = coordinator.prepare(80).unwrap();
        assert_eq!(
            coordinator.complete_write(KvWriteReceipt {
                txn_id: 80,
                committed_tokens: 1,
                written_blocks: Vec::new(),
            }),
            Err(KvCoordinatorError::InvalidWriteReceipt)
        );
        assert_eq!(
            coordinator.snapshot(&key, CacheDomainId::new(0)).unwrap(),
            initial
        );
        assert_eq!(
            coordinator.transaction_state(80),
            Some(KvTransactionState::Prepared)
        );
        assert_eq!(prepared.writable_blocks.len(), 1);
        coordinator.abort(80).unwrap();
        coordinator.check_invariants().unwrap();
    }

    #[test]
    fn arena_generation_fences_all_handles_and_snapshots() {
        let mut old = KvCacheCoordinator::new(arena(1), 1);
        let old_snapshot = old
            .register_table(session("reload", 1), CacheDomainId::new(0))
            .unwrap();
        reserve_fresh(&mut old, 90, old_snapshot, 1, 1);
        let old_block = old.prepare(90).unwrap().writable_blocks[0];

        let mut replacement = KvCacheCoordinator::new(arena(2), 1);
        replacement
            .register_table(session("reload", 2), CacheDomainId::new(0))
            .unwrap();
        assert_eq!(
            replacement.pin_transfer(&[old_block]),
            Err(KvCoordinatorError::WrongArena)
        );
        old.abort(90).unwrap();
    }

    #[test]
    fn physical_tail_cow_pins_the_source_and_copies_before_commit() -> crate::Result<()> {
        const LAYER: KvLayerBinding = KvLayerBinding {
            model_layer: 0,
            physical_layer: 0,
        };
        let arena_id = arena(1);
        let physical = CpuKvArena::new(KvArenaConfig {
            id: arena_id,
            group: KvGroupId::new(0),
            page_tokens: 2,
            capacity_pages: 2,
            dtype: DType::F32,
            layers: vec![KvLayerConfig {
                binding: LAYER,
                num_kv_heads: 1,
                key_head_dim: 2,
                value_head_dim: 2,
            }],
        })?;
        let mut coordinator = KvCacheCoordinator::new(arena_id, 2);
        let session = session("cow", 1);
        let initial = coordinator
            .register_table(session.clone(), CacheDomainId::new(0))
            .unwrap();
        reserve_fresh(&mut coordinator, 100, initial, 1, 1);
        let source = prepare_and_complete(&mut coordinator, 100).writable_blocks[0];
        let committed = coordinator.commit(100, &[]).unwrap();

        let slots = physical.lower_slots(&[
            KvSlotRef {
                block: source,
                offset: 0,
            },
            KvSlotRef {
                block: source,
                offset: 1,
            },
        ])?;
        let keys = Tensor::from_vec(vec![1f32, 2., 3., 4.], (2, 1, 2), &Device::Cpu)?;
        let values = Tensor::from_vec(vec![5f32, 6., 7., 8.], (2, 1, 2), &Device::Cpu)?;
        physical
            .write_slots(
                LAYER,
                KvWriteArgs {
                    keys: &keys,
                    values: &values,
                    slots: slots.as_ref(),
                },
            )?
            .wait()?;

        coordinator
            .reserve(KvReserveRequest {
                txn_id: 101,
                expected: committed,
                target_committed_tokens: 2,
                target_window_start: 0,
                groups: vec![KvGroupReservation {
                    group: KvGroupId::new(0),
                    blocks: vec![KvBlockIntent::CopyOnWrite(source)],
                }],
            })
            .unwrap();
        let prepared = coordinator.prepare(101).unwrap();
        assert_eq!(prepared.page_copies.len(), 1);
        assert_eq!(prepared.page_copies[0].source, source);
        assert_eq!(coordinator.stats().execution_pins, 2);
        physical.copy_pages(&prepared.page_copies)?.wait()?;
        let destination = prepared.page_copies[0].destination;
        let (stored_keys, stored_values) = physical.layer_tensors(LAYER)?;
        assert_eq!(
            stored_keys
                .i(destination.index as usize)?
                .to_vec3::<f32>()?,
            vec![vec![vec![1., 2.]], vec![vec![3., 4.]]]
        );
        assert_eq!(
            stored_values
                .i(destination.index as usize)?
                .to_vec3::<f32>()?,
            vec![vec![vec![5., 6.]], vec![vec![7., 8.]]]
        );
        coordinator
            .complete_write(KvWriteReceipt {
                txn_id: 101,
                committed_tokens: 2,
                written_blocks: prepared.writable_blocks,
            })
            .unwrap();
        let forked = coordinator.commit(101, &[]).unwrap();
        assert_eq!(forked.groups[0].blocks, vec![destination]);
        assert_eq!(coordinator.stats().execution_pins, 0);
        assert_eq!(
            coordinator.pin_transfer(&[source]),
            Err(KvCoordinatorError::StaleBlock)
        );
        coordinator.check_invariants().unwrap();
        Ok(())
    }

    #[test]
    fn sliding_window_commit_releases_only_wholly_hidden_pages() {
        let mut coordinator = KvCacheCoordinator::new(arena(1), 4);
        let session = session("window", 1);
        let initial = coordinator
            .register_table(session.clone(), CacheDomainId::new(0))
            .unwrap();
        reserve_fresh(&mut coordinator, 110, initial, 3, 10);
        let initial_pages = prepare_and_complete(&mut coordinator, 110).writable_blocks;
        let committed = coordinator.commit(110, &[]).unwrap();
        let advanced = advance_window(&initial_pages, 0, 5, 10, 4).unwrap();
        coordinator
            .reserve_window(KvWindowReserveRequest {
                txn_id: 111,
                expected: committed,
                target_committed_tokens: 10,
                target_window_start: advanced.window_start,
                page_tokens: 4,
            })
            .unwrap();
        let prepared = coordinator.prepare(111).unwrap();
        assert!(prepared.writable_blocks.is_empty());
        coordinator
            .complete_write(KvWriteReceipt {
                txn_id: 111,
                committed_tokens: 10,
                written_blocks: Vec::new(),
            })
            .unwrap();
        let trimmed = coordinator.commit(111, &[]).unwrap();
        assert_eq!(trimmed.window_start, 5);
        assert_eq!(trimmed.groups[0].blocks, advanced.visible_blocks);
        assert_eq!(coordinator.stats().allocated_pages, 2);
        assert_eq!(coordinator.stats().table_refs, 2);
        assert_eq!(
            coordinator.pin_transfer(&[advanced.released_blocks[0]]),
            Err(KvCoordinatorError::StaleBlock)
        );
        coordinator.check_invariants().unwrap();
    }

    #[test]
    fn window_abort_preserves_the_versioned_table_and_all_old_pages() {
        let mut coordinator = KvCacheCoordinator::new(arena(1), 5);
        let session = session("window-abort", 1);
        let initial = coordinator
            .register_table(session.clone(), CacheDomainId::new(0))
            .unwrap();
        reserve_fresh(&mut coordinator, 120, initial, 3, 10);
        prepare_and_complete(&mut coordinator, 120);
        let committed = coordinator.commit(120, &[]).unwrap();
        let committed_pages = committed.groups[0].blocks.clone();

        coordinator
            .reserve_window(KvWindowReserveRequest {
                txn_id: 121,
                expected: committed.clone(),
                target_committed_tokens: 13,
                target_window_start: 5,
                page_tokens: 4,
            })
            .unwrap();
        let prepared = coordinator.prepare(121).unwrap();
        assert_eq!(prepared.provisional_groups[0].blocks.len(), 3);
        assert_eq!(prepared.writable_blocks.len(), 2);
        assert_eq!(coordinator.stats().allocated_pages, 4);
        coordinator
            .complete_write(KvWriteReceipt {
                txn_id: 121,
                committed_tokens: 13,
                written_blocks: prepared.writable_blocks,
            })
            .unwrap();
        assert!(coordinator.abort(121).unwrap());

        assert_eq!(
            coordinator
                .snapshot(&session, CacheDomainId::new(0))
                .unwrap(),
            committed
        );
        assert_eq!(coordinator.stats().allocated_pages, 3);
        coordinator.pin_transfer(&committed_pages).unwrap();
        coordinator.unpin_transfer(&committed_pages).unwrap();
        coordinator.check_invariants().unwrap();
    }

    #[test]
    fn stale_window_snapshot_cannot_rotate_a_new_generation() {
        let mut coordinator = KvCacheCoordinator::new(arena(1), 4);
        let session = session("window-version", 1);
        let initial = coordinator
            .register_table(session, CacheDomainId::new(0))
            .unwrap();
        reserve_fresh(&mut coordinator, 130, initial, 2, 8);
        prepare_and_complete(&mut coordinator, 130);
        let stale = coordinator.commit(130, &[]).unwrap();

        coordinator
            .reserve_window(KvWindowReserveRequest {
                txn_id: 131,
                expected: stale.clone(),
                target_committed_tokens: 9,
                target_window_start: 4,
                page_tokens: 4,
            })
            .unwrap();
        prepare_and_complete(&mut coordinator, 131);
        coordinator.commit(131, &[]).unwrap();

        assert_eq!(
            coordinator.reserve_window(KvWindowReserveRequest {
                txn_id: 132,
                expected: stale,
                target_committed_tokens: 10,
                target_window_start: 5,
                page_tokens: 4,
            }),
            Err(KvCoordinatorError::VersionConflict)
        );
        assert_eq!(coordinator.stats().active_transactions, 0);
        coordinator.check_invariants().unwrap();
    }

    #[test]
    fn randomized_window_rotation_plateaus_physical_pages_and_survives_aborts() {
        const PAGE_TOKENS: u32 = 16;
        const WINDOW_TOKENS: u32 = 64;
        const COMMITTED_PAGE_BOUND: usize = 5;
        const TRANSACTION_PAGE_BOUND: usize = 6;

        let mut coordinator = KvCacheCoordinator::new(arena(1), TRANSACTION_PAGE_BOUND);
        let session = session("window-random", 1);
        let initial = coordinator
            .register_table(session.clone(), CacheDomainId::new(0))
            .unwrap();
        reserve_fresh(&mut coordinator, 140, initial, 1, 1);
        prepare_and_complete(&mut coordinator, 140);
        let mut committed = coordinator.commit(140, &[]).unwrap();
        let mut txn_id = 141;
        let mut random = 0x9e37_79b9_u32;

        while committed.committed_tokens < 10_000 {
            random = random.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let step = 1 + random % 7;
            let target = committed.committed_tokens.saturating_add(step).min(10_000);
            let window_start = target.saturating_sub(WINDOW_TOKENS);
            let before = committed.clone();

            coordinator
                .reserve_window(KvWindowReserveRequest {
                    txn_id,
                    expected: before.clone(),
                    target_committed_tokens: target,
                    target_window_start: window_start,
                    page_tokens: PAGE_TOKENS,
                })
                .unwrap();
            assert!(coordinator.stats().allocated_pages <= TRANSACTION_PAGE_BOUND);
            let prepared = coordinator.prepare(txn_id).unwrap();

            if random & 0x1f == 0 {
                if random & 0x20 != 0 {
                    coordinator
                        .complete_write(KvWriteReceipt {
                            txn_id,
                            committed_tokens: target,
                            written_blocks: prepared.writable_blocks,
                        })
                        .unwrap();
                }
                assert!(coordinator.abort(txn_id).unwrap());
                assert_eq!(
                    coordinator
                        .snapshot(&session, CacheDomainId::new(0))
                        .unwrap(),
                    before
                );
                assert_eq!(
                    coordinator.stats().allocated_pages,
                    before.groups[0].blocks.len()
                );
                txn_id += 1;
                continue;
            }

            coordinator
                .complete_write(KvWriteReceipt {
                    txn_id,
                    committed_tokens: target,
                    written_blocks: prepared.writable_blocks,
                })
                .unwrap();
            committed = coordinator.commit(txn_id, &[]).unwrap();
            let expected_pages = crate::engine::cache::window::pages_for_logical_range(
                window_start,
                target,
                PAGE_TOKENS,
            )
            .unwrap();
            assert_eq!(committed.groups[0].blocks.len(), expected_pages);
            assert!(coordinator.stats().allocated_pages <= COMMITTED_PAGE_BOUND);
            assert_eq!(coordinator.stats().allocated_pages, expected_pages);
            coordinator.check_invariants().unwrap();
            txn_id += 1;
        }
    }
}
