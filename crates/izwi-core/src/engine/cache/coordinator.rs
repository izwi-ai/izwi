//! Transactional control plane for backend-owned physical KV pages.
//!
//! This module deliberately contains no tensors or device pointers. It owns
//! generation-safe identities, committed request tables, reservations, and the
//! reference/pin counts that determine when an arena slot may be reused.

use std::collections::{HashMap, HashSet, VecDeque};

use thiserror::Error;

use crate::engine::execution::{PlanId, SessionKey};
use crate::kv::{CacheBlockRef, CacheDomainId, KvArenaId, KvGroupId};

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

/// Immutable execution metadata produced by `prepare`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvPreparedReservation {
    pub txn_id: PlanId,
    pub expected: KvSnapshot,
    pub provisional_groups: Vec<GroupBlockTable>,
    pub writable_blocks: Vec<CacheBlockRef>,
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
    holds: Vec<TransactionHold>,
    target_committed_tokens: u32,
    target_window_start: u32,
    state: KvTransactionState,
}

/// Transactional metadata coordinator for one physical arena generation.
pub struct KvCacheCoordinator {
    arena: KvArenaId,
    slots: Vec<BlockSlot>,
    free: VecDeque<u32>,
    tables: HashMap<TableKey, KvSnapshot>,
    transactions: HashMap<PlanId, Transaction>,
    terminal_transactions: HashMap<PlanId, KvTerminalState>,
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
            terminal_transactions: HashMap::new(),
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

    pub fn terminal_state(&self, txn_id: PlanId) -> Option<KvTerminalState> {
        self.terminal_transactions.get(&txn_id).copied()
    }

    /// Atomically reserve shared/fresh pages and exclusive writable tails.
    pub fn reserve(&mut self, request: KvReserveRequest) -> KvCoordinatorResult<()> {
        if self.transactions.contains_key(&request.txn_id)
            || self.terminal_transactions.contains_key(&request.txn_id)
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
                match *intent {
                    KvBlockIntent::Fresh => {
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
                            KvBlockIntent::Fresh => unreachable!(),
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
        if request.target_committed_tokens > request.expected.committed_tokens && !has_writable {
            return Err(KvCoordinatorError::InvalidTokenRange);
        }

        let mut provisional_groups = Vec::with_capacity(request.groups.len());
        let mut writable_blocks = Vec::new();
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
        let pages = unique_table_blocks(&txn.provisional_groups);
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
            self.abort_internal(txn_id);
            return Err(KvCoordinatorError::VersionConflict);
        }

        let writable: HashSet<_> = txn.writable_blocks.iter().copied().collect();
        let mut published = HashSet::new();
        for block in publish_prefix {
            self.validate_block(*block, None)?;
            if !writable.contains(block) || !published.insert(*block) {
                return Err(KvCoordinatorError::InvalidWriteReceipt);
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
        let next_version = txn.expected.version.checked_add(1).ok_or_else(|| {
            KvCoordinatorError::Invariant("request table version overflow".to_string())
        })?;

        // Add new ownership before releasing reservations or removed table refs.
        for block in new_blocks.difference(&old_blocks) {
            self.slots[block.index as usize].table_refs += 1;
        }
        for block in publish_prefix {
            self.slots[block.index as usize].prefix_refs += 1;
        }

        self.release_transaction_pins(&txn);
        self.release_transaction_holds(&txn);
        for block in old_blocks.difference(&new_blocks) {
            let slot = &mut self.slots[block.index as usize];
            slot.table_refs -= 1;
            self.recycle_if_unowned(block.index);
        }

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
            .insert(txn_id, KvTerminalState::Committed);
        Ok(committed)
    }

    /// Idempotently abort a reservation and release all pins/private ownership.
    pub fn abort(&mut self, txn_id: PlanId) -> KvCoordinatorResult<bool> {
        if self.transactions.contains_key(&txn_id) {
            self.abort_internal(txn_id);
            return Ok(true);
        }
        if self.terminal_transactions.contains_key(&txn_id) {
            return Ok(false);
        }
        Err(KvCoordinatorError::MissingTransaction(txn_id))
    }

    /// Release a completed request table. Active reservations must abort first.
    pub fn release_table(
        &mut self,
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
            .cloned()
            .ok_or(KvCoordinatorError::MissingTable)?;
        let blocks = unique_table_blocks(&snapshot.groups);
        for block in &blocks {
            self.validate_block(*block, None)?;
            if self.slots[block.index as usize].table_refs == 0 {
                return Err(KvCoordinatorError::ReferenceUnderflow);
            }
        }
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
        self.validate_block(block, None)?;
        let slot = &mut self.slots[block.index as usize];
        if slot.prefix_refs == 0 {
            return Err(KvCoordinatorError::ReferenceUnderflow);
        }
        slot.prefix_refs -= 1;
        self.recycle_if_unowned(block.index);
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
                for block in unique_table_blocks(&txn.provisional_groups) {
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
            .insert(txn_id, KvTerminalState::Aborted);
    }

    fn release_transaction_pins(&mut self, txn: &Transaction) {
        if !matches!(
            txn.state,
            KvTransactionState::Prepared | KvTransactionState::Written
        ) {
            return;
        }
        for block in unique_table_blocks(&txn.provisional_groups) {
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
    use crate::backends::BackendKind;
    use crate::engine::ModelInstanceId;

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
}
