use std::collections::HashMap;
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PhysicalStateTransactionId(u64);

impl PhysicalStateTransactionId {
    pub(crate) fn new(value: u64) -> Result<Self> {
        (value != 0)
            .then_some(Self(value))
            .ok_or_else(|| invalid("physical state transaction id must be non-zero"))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct StateComponentValue {
    pub(crate) component: StateComponentId,
    pub(crate) tensor: Tensor,
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
}

#[derive(Default)]
struct ArenaState {
    sequences: HashMap<PhysicalStateSequenceId, SequenceState>,
    transactions: HashMap<PhysicalStateTransactionId, StagedTransaction>,
}

/// Model-neutral transactional ownership for retained tensor, append, and
/// ring state. Tensors remain on the selected Candle device; a transition is
/// invisible until every domain in its consistency closure is staged and the
/// transaction is committed under one lock.
pub(crate) struct TensorStateArena {
    plan: Arc<ResolvedStatePlan>,
    device: Device,
    state: Mutex<ArenaState>,
}

impl std::fmt::Debug for TensorStateArena {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TensorStateArena")
            .field("plan", &self.plan.id)
            .field("device", &self.device.location())
            .finish_non_exhaustive()
    }
}

impl TensorStateArena {
    pub(crate) fn new(plan: Arc<ResolvedStatePlan>, device: Device) -> Result<Self> {
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
        Ok(Self {
            plan,
            device,
            state: Mutex::new(ArenaState::default()),
        })
    }

    pub(crate) fn plan(&self) -> &ResolvedStatePlan {
        &self.plan
    }

    pub(crate) fn register(&self, sequence: PhysicalStateSequenceId) -> Result<()> {
        let mut state = self.lock()?;
        if state.sequences.contains_key(&sequence) {
            return Err(invalid("physical state sequence is already registered"));
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
        if state.transactions.contains_key(&transaction) {
            return Err(invalid("physical state transaction id is already active"));
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

    pub(crate) fn commit(&self, transaction: PhysicalStateTransactionId) -> Result<()> {
        let mut state = self.lock()?;
        let staged = state
            .transactions
            .remove(&transaction)
            .ok_or_else(|| invalid("physical state transaction is not active"))?;
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
        if state
            .transactions
            .values()
            .any(|transaction| transaction.sequence == sequence)
        {
            return Err(invalid("physical state sequence has an active transaction"));
        }
        state
            .sequences
            .remove(&sequence)
            .map(|_| ())
            .ok_or_else(|| invalid("physical state sequence is not registered"))
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
            validate_tensor(value, component, multiplier, &self.device)?;
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
    value: &StateComponentValue,
    component: &ResolvedTensorComponent,
    multiplier: u64,
    device: &Device,
) -> Result<()> {
    if value.tensor.device().location() != device.location()
        || value.tensor.dtype() != candle_dtype(component.storage.dtype())?
    {
        return Err(invalid(
            "physical state tensor device or dtype does not match its resolved component",
        ));
    }
    let bytes = u64::try_from(
        value
            .tensor
            .elem_count()
            .checked_mul(value.tensor.dtype().size_in_bytes())
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
        TensorStateArena::new(Arc::new(plan), Device::Cpu).unwrap()
    }

    fn value(values: &[f32]) -> StateComponentValue {
        StateComponentValue {
            component: StateComponentId::new(1),
            tensor: Tensor::from_slice(values, values.len(), &Device::Cpu).unwrap(),
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
        arena.commit(first).unwrap();
        let committed = arena
            .read(sequence, StateDomainId::new(1))
            .unwrap()
            .unwrap();
        assert_eq!(committed.cursor, 1);
        assert_eq!(
            committed.components[0].tensor.to_vec1::<f32>().unwrap(),
            vec![1.0, 2.0]
        );
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
        arena.commit(committed).unwrap();

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
                .to_vec1::<f32>()
                .unwrap(),
            vec![3.0]
        );
    }
}
