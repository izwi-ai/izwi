use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Tensor};

use crate::error::{Error, Result};
use crate::kv::v2::{
    ResolvedNonPagedDomainPlan, ResolvedStatePlan, ResolvedTensorComponent, StateComponentId,
    StateDType, StateDomainId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PhysicalStateSequenceId(u64);

impl PhysicalStateSequenceId {
    pub(crate) fn new(value: u64) -> Result<Self> {
        (value != 0)
            .then_some(Self(value))
            .ok_or_else(|| invalid("physical state sequence id must be non-zero"))
    }

    pub(crate) const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PhysicalStateTransactionId(u64);

impl PhysicalStateTransactionId {
    pub(crate) fn new(value: u64) -> Result<Self> {
        (value != 0)
            .then_some(Self(value))
            .ok_or_else(|| invalid("physical state transaction id must be non-zero"))
    }

    pub(crate) const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone)]
pub(crate) struct StateComponentValue {
    pub(crate) component: StateComponentId,
    /// `None` is a first-class committed absence for optional or not-yet
    /// initialized state. It avoids fake sentinel allocations.
    pub(crate) tensor: Option<Tensor>,
}

#[derive(Debug, Clone)]
pub(crate) struct StateDomainSnapshot {
    pub(crate) cursor: u64,
    pub(crate) components: Arc<[StateComponentValue]>,
}

#[derive(Clone, Default)]
struct SequenceState {
    domains: HashMap<StateDomainId, StateDomainSnapshot>,
}

struct StagedTransaction {
    sequence: PhysicalStateSequenceId,
    state: SequenceState,
    touched: HashSet<StateDomainId>,
}

#[derive(Default)]
struct ArenaState {
    closed: bool,
    sequences: HashMap<PhysicalStateSequenceId, SequenceState>,
    transactions: HashMap<PhysicalStateTransactionId, StagedTransaction>,
}

/// Immutable admission and byte-authorization envelope for one tensor arena.
///
/// Committed state and transaction-private staged replacements can coexist.
/// Their capacities are therefore accounted independently instead of assuming
/// one generation of backing per retained sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TensorStateCapacity {
    per_sequence_bytes: u64,
    sequence_capacity: u32,
    transaction_capacity: u32,
    committed_capacity_bytes: u64,
    staging_capacity_bytes: u64,
    authorized_bytes: u64,
}

impl TensorStateCapacity {
    pub(crate) fn for_plan(
        plan: &ResolvedStatePlan,
        sequence_capacity: u32,
        transaction_capacity: u32,
    ) -> Result<Self> {
        if sequence_capacity == 0
            || transaction_capacity == 0
            || transaction_capacity > sequence_capacity
        {
            return Err(invalid(
                "tensor state capacity requires non-zero transaction capacity not exceeding sequence capacity",
            ));
        }
        let per_sequence_bytes = plan.non_paged.iter().try_fold(0_u64, |total, domain| {
            total
                .checked_add(domain.maximum_bytes())
                .ok_or_else(|| invalid("tensor state per-sequence byte bound overflow"))
        })?;
        if per_sequence_bytes == 0 {
            return Err(invalid(
                "tensor state capacity requires resolved non-paged state bytes",
            ));
        }
        let (committed_capacity_bytes, staging_capacity_bytes, authorized_bytes) =
            capacity_byte_totals(per_sequence_bytes, sequence_capacity, transaction_capacity)?;
        Ok(Self {
            per_sequence_bytes,
            sequence_capacity,
            transaction_capacity,
            committed_capacity_bytes,
            staging_capacity_bytes,
            authorized_bytes,
        })
    }

    pub(crate) const fn per_sequence_bytes(self) -> u64 {
        self.per_sequence_bytes
    }

    pub(crate) const fn sequence_capacity(self) -> u32 {
        self.sequence_capacity
    }

    pub(crate) const fn transaction_capacity(self) -> u32 {
        self.transaction_capacity
    }

    pub(crate) const fn committed_capacity_bytes(self) -> u64 {
        self.committed_capacity_bytes
    }

    pub(crate) const fn staging_capacity_bytes(self) -> u64 {
        self.staging_capacity_bytes
    }

    pub(crate) const fn authorized_bytes(self) -> u64 {
        self.authorized_bytes
    }
}

fn capacity_byte_totals(
    per_sequence_bytes: u64,
    sequence_capacity: u32,
    transaction_capacity: u32,
) -> Result<(u64, u64, u64)> {
    let committed = per_sequence_bytes
        .checked_mul(u64::from(sequence_capacity))
        .ok_or_else(|| invalid("tensor state committed byte capacity overflow"))?;
    let staging = per_sequence_bytes
        .checked_mul(u64::from(transaction_capacity))
        .ok_or_else(|| invalid("tensor state staging byte capacity overflow"))?;
    let authorized = committed
        .checked_add(staging)
        .ok_or_else(|| invalid("tensor state authorized byte capacity overflow"))?;
    Ok((committed, staging, authorized))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TensorStateOccupancy {
    pub(crate) active_sequences: u32,
    pub(crate) active_transactions: u32,
}

/// Model-neutral transactional ownership for retained tensor, append, and
/// ring state. Tensors remain on the selected Candle device; a transition is
/// invisible until every domain in its consistency closure is staged and the
/// transaction is committed under one lock.
pub(crate) struct TensorStateArena {
    plan: Arc<ResolvedStatePlan>,
    capacity: TensorStateCapacity,
    device: Device,
    state: Mutex<ArenaState>,
}

impl std::fmt::Debug for TensorStateArena {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TensorStateArena")
            .field("plan", &self.plan.id)
            .field("capacity", &self.capacity)
            .field("device", &self.device.location())
            .finish_non_exhaustive()
    }
}

impl TensorStateArena {
    pub(crate) fn new(
        plan: Arc<ResolvedStatePlan>,
        capacity: TensorStateCapacity,
        device: Device,
    ) -> Result<Self> {
        if plan.non_paged.is_empty() {
            return Err(invalid(
                "tensor state arena requires at least one resolved non-paged domain",
            ));
        }
        if plan.backend != crate::backends::backend_kind_for_device(&device) {
            return Err(invalid(
                "tensor state arena device does not match its resolved backend",
            ));
        }
        if plan
            .non_paged
            .iter()
            .any(|domain| matches!(domain, ResolvedNonPagedDomainPlan::StaticAttention(_)))
        {
            return Err(invalid(
                "static attention requires its direct backend arena, not the tensor-state arena",
            ));
        }
        let expected_capacity = TensorStateCapacity::for_plan(
            &plan,
            capacity.sequence_capacity,
            capacity.transaction_capacity,
        )?;
        if capacity != expected_capacity {
            return Err(invalid(
                "tensor state capacity does not match the resolved state plan",
            ));
        }
        Ok(Self {
            plan,
            capacity,
            device,
            state: Mutex::new(ArenaState::default()),
        })
    }

    pub(crate) fn plan(&self) -> &ResolvedStatePlan {
        &self.plan
    }

    pub(crate) const fn capacity(&self) -> TensorStateCapacity {
        self.capacity
    }

    pub(crate) fn occupancy(&self) -> Result<TensorStateOccupancy> {
        let state = self.lock()?;
        Ok(TensorStateOccupancy {
            active_sequences: u32::try_from(state.sequences.len())
                .map_err(|_| invalid("tensor state sequence occupancy exceeds u32"))?,
            active_transactions: u32::try_from(state.transactions.len())
                .map_err(|_| invalid("tensor state transaction occupancy exceeds u32"))?,
        })
    }

    pub(crate) fn register(&self, sequence: PhysicalStateSequenceId) -> Result<()> {
        let mut state = self.lock()?;
        if state.closed {
            return Err(invalid("physical state arena is closed"));
        }
        if state.sequences.contains_key(&sequence) {
            return Err(invalid("physical state sequence is already registered"));
        }
        if state.sequences.len() >= self.capacity.sequence_capacity as usize {
            return Err(Error::Backpressure(
                "tensor state sequence capacity is exhausted".into(),
            ));
        }
        state.sequences.insert(sequence, SequenceState::default());
        Ok(())
    }

    pub(crate) fn begin(
        &self,
        transaction: PhysicalStateTransactionId,
        sequence: PhysicalStateSequenceId,
    ) -> Result<()> {
        let mut state = self.lock()?;
        if state.closed {
            return Err(invalid("physical state arena is closed"));
        }
        if state.transactions.contains_key(&transaction) {
            return Err(invalid("physical state transaction id is already active"));
        }
        if state.transactions.len() >= self.capacity.transaction_capacity as usize {
            return Err(Error::Backpressure(
                "tensor state transaction capacity is exhausted".into(),
            ));
        }
        if state
            .transactions
            .values()
            .any(|active| active.sequence == sequence)
        {
            return Err(invalid(
                "physical state sequence already has an active transaction",
            ));
        }
        let live = state
            .sequences
            .get(&sequence)
            .cloned()
            .ok_or_else(|| invalid("physical state sequence is not registered"))?;
        state.transactions.insert(
            transaction,
            StagedTransaction {
                sequence,
                state: live,
                touched: HashSet::new(),
            },
        );
        Ok(())
    }

    pub(crate) fn stage_replace(
        &self,
        transaction: PhysicalStateTransactionId,
        domain: StateDomainId,
        expected_cursor: u64,
        target_cursor: u64,
        components: Vec<StateComponentValue>,
    ) -> Result<()> {
        if target_cursor < expected_cursor {
            return Err(invalid("physical state cursor cannot move backwards"));
        }
        let resolved = self
            .plan
            .non_paged
            .iter()
            .find(|candidate| candidate.domain() == domain)
            .ok_or_else(|| invalid("physical state transaction references an unknown domain"))?;
        self.validate_components(resolved, &components)?;

        let mut state = self.lock()?;
        let staged = state
            .transactions
            .get_mut(&transaction)
            .ok_or_else(|| invalid("physical state transaction is not active"))?;
        let current = staged
            .state
            .domains
            .get(&domain)
            .map(|snapshot| snapshot.cursor)
            .unwrap_or(0);
        if current != expected_cursor {
            return Err(invalid("physical state transaction cursor is stale"));
        }
        staged.state.domains.insert(
            domain,
            StateDomainSnapshot {
                cursor: target_cursor,
                components: components.into(),
            },
        );
        staged.touched.insert(domain);
        Ok(())
    }

    pub(crate) fn read(
        &self,
        sequence: PhysicalStateSequenceId,
        domain: StateDomainId,
    ) -> Result<Option<StateDomainSnapshot>> {
        Ok(self
            .lock()?
            .sequences
            .get(&sequence)
            .ok_or_else(|| invalid("physical state sequence is not registered"))?
            .domains
            .get(&domain)
            .cloned())
    }

    pub(crate) fn read_transaction_base(
        &self,
        transaction: PhysicalStateTransactionId,
        domain: StateDomainId,
    ) -> Result<Option<StateDomainSnapshot>> {
        Ok(self
            .lock()?
            .transactions
            .get(&transaction)
            .ok_or_else(|| invalid("physical state transaction is not active"))?
            .state
            .domains
            .get(&domain)
            .cloned())
    }

    pub(crate) fn commit(
        &self,
        transaction: PhysicalStateTransactionId,
        expected_cursor: u64,
    ) -> Result<()> {
        let required_domains = self
            .plan
            .non_paged
            .iter()
            .map(ResolvedNonPagedDomainPlan::domain)
            .collect::<Vec<_>>();
        let mut state = self.lock()?;
        let staged = state
            .transactions
            .get(&transaction)
            .ok_or_else(|| invalid("physical state transaction is not active"))?;
        if required_domains.iter().any(|domain| {
            !staged.touched.contains(domain)
                || staged
                    .state
                    .domains
                    .get(domain)
                    .map_or(true, |snapshot| snapshot.cursor != expected_cursor)
        }) {
            return Err(invalid(
                "physical state transaction did not stage every domain at the target cursor",
            ));
        }
        let staged = state
            .transactions
            .remove(&transaction)
            .expect("validated physical state transaction remains active");
        let live = state
            .sequences
            .get_mut(&staged.sequence)
            .ok_or_else(|| invalid("physical state sequence was released during a transaction"))?;
        *live = staged.state;
        Ok(())
    }

    pub(crate) fn abort(&self, transaction: PhysicalStateTransactionId) -> Result<()> {
        self.lock()?
            .transactions
            .remove(&transaction)
            .map(|_| ())
            .ok_or_else(|| invalid("physical state transaction is not active"))
    }

    pub(crate) fn release(&self, sequence: PhysicalStateSequenceId) -> Result<()> {
        let mut state = self.lock()?;
        validate_sequence_release(&state, sequence)?;
        state
            .sequences
            .remove(&sequence)
            .expect("validated physical state sequence remains registered");
        Ok(())
    }

    pub(crate) fn validate_release(&self, sequence: PhysicalStateSequenceId) -> Result<()> {
        let state = self.lock()?;
        validate_sequence_release(&state, sequence)
    }

    /// Prevent new sequences/transactions and prove all existing users have
    /// released their state. A failed drain remains closed so unload can retry
    /// after the active owners finish without admitting new work.
    pub(crate) fn close_and_validate_drained(&self) -> Result<()> {
        let mut state = self.lock()?;
        state.closed = true;
        if !state.transactions.is_empty() || !state.sequences.is_empty() {
            return Err(invalid(
                "physical state arena still has active sequences or transactions",
            ));
        }
        Ok(())
    }

    fn validate_components(
        &self,
        domain: &ResolvedNonPagedDomainPlan,
        values: &[StateComponentValue],
    ) -> Result<()> {
        let (components, multiplier) = match domain {
            ResolvedNonPagedDomainPlan::StaticTensor(plan) => (plan.components.as_slice(), 1),
            ResolvedNonPagedDomainPlan::Tensor(plan) => (plan.components.as_slice(), 1),
            ResolvedNonPagedDomainPlan::Append(plan) => {
                let per_step = plan
                    .components_per_step
                    .iter()
                    .map(|component| component.maximum_bytes)
                    .sum::<u64>();
                let multiplier = plan.maximum_bytes.checked_div(per_step).ok_or_else(|| {
                    invalid("append state resolved an invalid component byte bound")
                })?;
                (plan.components_per_step.as_slice(), multiplier)
            }
            ResolvedNonPagedDomainPlan::Ring(plan) => {
                let per_step = plan
                    .components_per_step
                    .iter()
                    .map(|component| component.maximum_bytes)
                    .sum::<u64>();
                let multiplier = plan.maximum_bytes.checked_div(per_step).ok_or_else(|| {
                    invalid("ring state resolved an invalid component byte bound")
                })?;
                (plan.components_per_step.as_slice(), multiplier)
            }
            ResolvedNonPagedDomainPlan::StaticAttention(_) => {
                return Err(invalid(
                    "static attention cannot be staged through tensor replacement",
                ));
            }
        };
        if values.len() != components.len() {
            return Err(invalid(
                "physical state update must cover every component exactly once",
            ));
        }
        for (index, (value, component)) in values.iter().zip(components).enumerate() {
            if value.component != component.component
                || (index > 0 && values[index - 1].component >= value.component)
            {
                return Err(invalid(
                    "physical state components must use canonical resolved identities",
                ));
            }
            if let Some(tensor) = value.tensor.as_ref() {
                validate_tensor(tensor, component, multiplier, &self.device)?;
            }
        }
        Ok(())
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, ArenaState>> {
        self.state
            .lock()
            .map_err(|_| Error::InferenceError("tensor state arena lock poisoned".into()))
    }
}

fn validate_tensor(
    tensor: &Tensor,
    component: &ResolvedTensorComponent,
    multiplier: u64,
    device: &Device,
) -> Result<()> {
    if tensor.device().location() != device.location()
        || tensor.dtype() != candle_dtype(component.storage.dtype())?
    {
        return Err(invalid(
            "physical state tensor device or dtype does not match its resolved component",
        ));
    }
    let bytes = u64::try_from(
        tensor
            .elem_count()
            .checked_mul(tensor.dtype().size_in_bytes())
            .ok_or_else(|| invalid("physical state tensor byte count overflow"))?,
    )
    .map_err(|_| invalid("physical state tensor byte count exceeds u64"))?;
    let maximum = component
        .maximum_bytes
        .checked_mul(multiplier)
        .ok_or_else(|| invalid("physical state component capacity overflow"))?;
    if bytes == 0 || bytes > maximum {
        return Err(invalid(
            "physical state tensor exceeds its resolved component capacity",
        ));
    }
    Ok(())
}

fn candle_dtype(dtype: StateDType) -> Result<DType> {
    match dtype {
        StateDType::F32 => Ok(DType::F32),
        StateDType::F16 => Ok(DType::F16),
        StateDType::Bf16 => Ok(DType::BF16),
        StateDType::I8 | StateDType::Q4 => Err(invalid(
            "quantized tensor state requires an explicit packing ABI",
        )),
    }
}

fn validate_sequence_release(state: &ArenaState, sequence: PhysicalStateSequenceId) -> Result<()> {
    if state
        .transactions
        .values()
        .any(|transaction| transaction.sequence == sequence)
    {
        return Err(invalid("physical state sequence has an active transaction"));
    }
    if !state.sequences.contains_key(&sequence) {
        return Err(invalid("physical state sequence is not registered"));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::state::{negotiate_state_plan, StateBackendPlanRequest};
    use crate::backends::BackendKind;
    use crate::kv::v2::{
        BoundedShape, CheckpointPolicy, InferenceStateContract, PlacementPolicy, PrefixPolicy,
        ShapeAxis, ShapeDimension, ShapeExtent, StateClock, StateDomainHeader, StateDomainSpec,
        StateGroupId, StateGroupSpec, StateScope, TensorComponentSpec, TensorRole,
        TensorStateDomainSpec, CURRENT_INFERENCE_STATE_ABI,
    };

    fn contract() -> InferenceStateContract {
        InferenceStateContract {
            abi: CURRENT_INFERENCE_STATE_ABI,
            domains: vec![StateDomainSpec::Tensor(TensorStateDomainSpec {
                header: StateDomainHeader {
                    id: StateDomainId::new(1),
                    scope: StateScope::Retained,
                    clock: StateClock::DecoderTokens,
                    placement: PlacementPolicy::BackendLocal,
                    prefix: PrefixPolicy::Disabled,
                    checkpoint: CheckpointPolicy::Transactional,
                },
                components: vec![TensorComponentSpec {
                    id: StateComponentId::new(1),
                    role: TensorRole::RecurrentHidden,
                    shape: BoundedShape {
                        dimensions: vec![ShapeDimension {
                            axis: ShapeAxis::Hidden,
                            extent: ShapeExtent::RuntimeBounded { min: 1, max: 8 },
                        }],
                    },
                    accepted_dtypes: vec![StateDType::F32],
                }],
            })],
            groups: vec![StateGroupSpec {
                id: StateGroupId::new(1),
                domains: vec![StateDomainId::new(1)],
                prefix_shareable: false,
            }],
        }
    }

    fn arena() -> TensorStateArena {
        let plan = negotiate_state_plan(
            &contract(),
            &StateBackendPlanRequest {
                backend: BackendKind::Cpu,
                device_ordinal: None,
                page_tokens_hint: None,
                storage_dtype_hint: None,
            },
        )
        .unwrap();
        let capacity = TensorStateCapacity::for_plan(&plan, 2, 2).unwrap();
        TensorStateArena::new(Arc::new(plan), capacity, Device::Cpu).unwrap()
    }

    fn value(values: &[f32]) -> StateComponentValue {
        StateComponentValue {
            component: StateComponentId::new(1),
            tensor: Some(Tensor::from_slice(values, values.len(), &Device::Cpu).unwrap()),
        }
    }

    fn absent_value() -> StateComponentValue {
        StateComponentValue {
            component: StateComponentId::new(1),
            tensor: None,
        }
    }

    #[test]
    fn replacements_are_invisible_until_atomic_commit() {
        let arena = arena();
        let sequence = PhysicalStateSequenceId::new(1).unwrap();
        let first = PhysicalStateTransactionId::new(1).unwrap();
        arena.register(sequence).unwrap();
        arena.begin(first, sequence).unwrap();
        arena
            .stage_replace(first, StateDomainId::new(1), 0, 1, vec![value(&[1.0, 2.0])])
            .unwrap();
        assert!(arena
            .read(sequence, StateDomainId::new(1))
            .unwrap()
            .is_none());
        arena.commit(first, 1).unwrap();
        let committed = arena
            .read(sequence, StateDomainId::new(1))
            .unwrap()
            .unwrap();
        assert_eq!(committed.cursor, 1);
        assert_eq!(
            committed.components[0]
                .tensor
                .as_ref()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![1.0, 2.0]
        );
    }

    #[test]
    fn explicit_absence_is_transactional_without_sentinel_storage() {
        let arena = arena();
        let sequence = PhysicalStateSequenceId::new(1).unwrap();
        let transaction = PhysicalStateTransactionId::new(1).unwrap();
        arena.register(sequence).unwrap();
        arena.begin(transaction, sequence).unwrap();
        arena
            .stage_replace(
                transaction,
                StateDomainId::new(1),
                0,
                1,
                vec![absent_value()],
            )
            .unwrap();
        arena.commit(transaction, 1).unwrap();

        let snapshot = arena
            .read(sequence, StateDomainId::new(1))
            .unwrap()
            .unwrap();
        assert_eq!(snapshot.cursor, 1);
        assert!(snapshot.components[0].tensor.is_none());
        arena.release(sequence).unwrap();
    }

    #[test]
    fn abort_preserves_live_state_and_capacity_is_enforced() {
        let arena = arena();
        let sequence = PhysicalStateSequenceId::new(1).unwrap();
        arena.register(sequence).unwrap();
        let committed = PhysicalStateTransactionId::new(1).unwrap();
        arena.begin(committed, sequence).unwrap();
        arena
            .stage_replace(committed, StateDomainId::new(1), 0, 1, vec![value(&[3.0])])
            .unwrap();
        arena.commit(committed, 1).unwrap();

        let aborted = PhysicalStateTransactionId::new(2).unwrap();
        arena.begin(aborted, sequence).unwrap();
        assert!(arena
            .stage_replace(aborted, StateDomainId::new(1), 1, 2, vec![value(&[0.0; 9])],)
            .is_err());
        arena.abort(aborted).unwrap();
        assert_eq!(
            arena
                .read(sequence, StateDomainId::new(1))
                .unwrap()
                .unwrap()
                .components[0]
                .tensor
                .as_ref()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![3.0]
        );
    }

    #[test]
    fn commit_rejects_an_unstaged_domain_and_remains_abortable() {
        let arena = arena();
        let sequence = PhysicalStateSequenceId::new(1).unwrap();
        let transaction = PhysicalStateTransactionId::new(1).unwrap();
        arena.register(sequence).unwrap();
        arena.begin(transaction, sequence).unwrap();
        assert!(arena.commit(transaction, 1).is_err());
        arena.abort(transaction).unwrap();
        assert!(arena
            .read(sequence, StateDomainId::new(1))
            .unwrap()
            .is_none());
    }

    #[test]
    fn sequence_release_preflight_preserves_an_active_transaction() {
        let arena = arena();
        let sequence = PhysicalStateSequenceId::new(1).unwrap();
        let transaction = PhysicalStateTransactionId::new(1).unwrap();
        arena.register(sequence).unwrap();
        arena.begin(transaction, sequence).unwrap();
        assert!(arena.validate_release(sequence).is_err());
        assert!(arena.release(sequence).is_err());
        arena.abort(transaction).unwrap();
        arena.validate_release(sequence).unwrap();
        arena.release(sequence).unwrap();
    }

    #[test]
    fn one_sequence_cannot_fork_concurrent_transactions() {
        let arena = arena();
        let sequence = PhysicalStateSequenceId::new(1).unwrap();
        arena.register(sequence).unwrap();
        arena
            .begin(PhysicalStateTransactionId::new(1).unwrap(), sequence)
            .unwrap();
        assert!(arena
            .begin(PhysicalStateTransactionId::new(2).unwrap(), sequence)
            .is_err());
        arena
            .abort(PhysicalStateTransactionId::new(1).unwrap())
            .unwrap();
        arena
            .begin(PhysicalStateTransactionId::new(2).unwrap(), sequence)
            .unwrap();
    }

    #[test]
    fn capacity_accounts_for_committed_and_staged_generations() {
        let arena = arena();
        let capacity = arena.capacity();
        assert_eq!(capacity.per_sequence_bytes(), 8 * 4);
        assert_eq!(capacity.sequence_capacity(), 2);
        assert_eq!(capacity.transaction_capacity(), 2);
        assert_eq!(capacity.committed_capacity_bytes(), 2 * 8 * 4);
        assert_eq!(capacity.staging_capacity_bytes(), 2 * 8 * 4);
        assert_eq!(capacity.authorized_bytes(), 4 * 8 * 4);

        let plan = arena.plan();
        assert!(TensorStateCapacity::for_plan(plan, 0, 1).is_err());
        assert!(TensorStateCapacity::for_plan(plan, 1, 0).is_err());
        assert!(TensorStateCapacity::for_plan(plan, 1, 2).is_err());
        assert!(capacity_byte_totals(u64::MAX, 2, 1).is_err());
        assert!(capacity_byte_totals(u64::MAX / 2 + 1, 1, 1).is_err());
    }

    #[test]
    fn sequence_and_transaction_admission_are_bounded_and_reusable() {
        let arena = arena();
        let first = PhysicalStateSequenceId::new(1).unwrap();
        let second = PhysicalStateSequenceId::new(2).unwrap();
        let third = PhysicalStateSequenceId::new(3).unwrap();
        arena.register(first).unwrap();
        arena.register(second).unwrap();
        assert!(matches!(arena.register(third), Err(Error::Backpressure(_))));

        let first_txn = PhysicalStateTransactionId::new(1).unwrap();
        let second_txn = PhysicalStateTransactionId::new(2).unwrap();
        let third_txn = PhysicalStateTransactionId::new(3).unwrap();
        arena.begin(first_txn, first).unwrap();
        arena.begin(second_txn, second).unwrap();
        assert!(matches!(
            arena.begin(third_txn, first),
            Err(Error::Backpressure(_))
        ));

        arena.abort(first_txn).unwrap();
        arena.begin(third_txn, first).unwrap();
        arena.abort(third_txn).unwrap();
        arena.abort(second_txn).unwrap();
        arena.release(first).unwrap();
        arena.register(third).unwrap();
        assert_eq!(
            arena.occupancy().unwrap(),
            TensorStateOccupancy {
                active_sequences: 2,
                active_transactions: 0,
            }
        );
    }

    #[test]
    fn failed_drain_closes_admission_and_can_be_retried_after_release() {
        let arena = arena();
        let sequence = PhysicalStateSequenceId::new(1).unwrap();
        arena.register(sequence).unwrap();
        let transaction = PhysicalStateTransactionId::new(1).unwrap();
        arena.begin(transaction, sequence).unwrap();

        assert!(arena.close_and_validate_drained().is_err());
        assert!(arena
            .register(PhysicalStateSequenceId::new(2).unwrap())
            .is_err());
        assert!(arena
            .begin(PhysicalStateTransactionId::new(2).unwrap(), sequence)
            .is_err());

        arena.abort(transaction).unwrap();
        arena.release(sequence).unwrap();
        arena.close_and_validate_drained().unwrap();
    }
}
