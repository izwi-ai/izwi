//! Load-owned retained static-attention sidecars.
//!
//! A fixed arena pool is allocated at model load. An uncommitted install owns
//! a private staging slot; commit atomically promotes that slot to the
//! sequence's immutable read view. Static initialization is one-shot, so the
//! number of committed plus staging slots can never exceed sequence capacity.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use candle_core::{Device, Tensor};
use serde::Serialize;

use crate::backends::state::{
    InvocationStaticAttentionArena, StaticAttentionLayerValue, StaticAttentionMetadata,
    StaticAttentionRaggedRow,
};
use crate::backends::BackendKind;
use crate::engine::ModelInstanceId;
use crate::error::{Error, Result};
use crate::kv::v2::{
    DomainStepIntent, InferenceStateContract, ResolvedStatePlan, StateDomainId, StatePlanId,
    StateUpdateKind,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub(crate) struct RetainedStaticAttentionRuntimeIdV2 {
    pub(crate) model_instance: ModelInstanceId,
    pub(crate) allocation_generation: u32,
    pub(crate) state_plan: StatePlanId,
    pub(crate) domain: StateDomainId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct RetainedStaticAttentionSequenceId {
    runtime: RetainedStaticAttentionRuntimeIdV2,
    nonce: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct RetainedStaticAttentionTransactionId {
    runtime: RetainedStaticAttentionRuntimeIdV2,
    nonce: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RetainedStaticAttentionBatchRow {
    pub(crate) sequence: RetainedStaticAttentionSequenceId,
    pub(crate) query_start: u32,
    pub(crate) query_len: u32,
}

#[derive(Debug)]
struct SequenceState {
    committed_slot: Option<usize>,
    transaction_nonce: Option<u64>,
}

#[derive(Debug)]
struct TransactionState {
    sequence_nonce: u64,
    staging_slot: usize,
}

#[derive(Debug)]
struct RuntimeState {
    closed: bool,
    sequences: HashMap<u64, SequenceState>,
    transactions: HashMap<u64, TransactionState>,
    free_slots: Vec<usize>,
}

pub(crate) struct RetainedStaticAttentionRuntimeV2 {
    id: RetainedStaticAttentionRuntimeIdV2,
    plan: Arc<ResolvedStatePlan>,
    backend: BackendKind,
    per_sequence_bytes: u64,
    maximum_bytes: u64,
    arenas: Vec<Mutex<InvocationStaticAttentionArena>>,
    state: Mutex<RuntimeState>,
    next_sequence: AtomicU64,
    next_transaction: AtomicU64,
}

impl std::fmt::Debug for RetainedStaticAttentionRuntimeV2 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RetainedStaticAttentionRuntimeV2")
            .field("id", &self.id)
            .field("sequence_capacity", &self.sequence_capacity())
            .field("per_sequence_bytes", &self.per_sequence_bytes)
            .field("maximum_bytes", &self.maximum_bytes)
            .finish_non_exhaustive()
    }
}

impl RetainedStaticAttentionRuntimeV2 {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        model_instance: ModelInstanceId,
        allocation_generation: u32,
        contract: &InferenceStateContract,
        plan: Arc<ResolvedStatePlan>,
        domain: StateDomainId,
        sequence_capacity: u32,
        device: Device,
    ) -> Result<Self> {
        if model_instance.get() == 0 || allocation_generation == 0 || sequence_capacity == 0 {
            return Err(invalid(
                "retained static attention requires non-zero identities and capacity",
            ));
        }
        let capacity = usize::try_from(sequence_capacity)
            .map_err(|_| invalid("retained static-attention capacity exceeds usize"))?;
        let mut arenas = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            arenas.push(Mutex::new(InvocationStaticAttentionArena::new_retained(
                contract,
                plan.clone(),
                domain,
                device.clone(),
            )?));
        }
        let per_sequence_bytes = arenas
            .first()
            .ok_or_else(|| invalid("retained static-attention runtime has no arenas"))?
            .lock()
            .map_err(|_| invalid("retained static-attention arena is poisoned"))?
            .maximum_bytes();
        if per_sequence_bytes == 0 {
            return Err(invalid("retained static-attention arena has zero bytes"));
        }
        for arena in &arenas {
            let arena = arena
                .lock()
                .map_err(|_| invalid("retained static-attention arena is poisoned"))?;
            if arena.plan() != plan.as_ref()
                || arena.domain() != domain
                || arena.maximum_bytes() != per_sequence_bytes
            {
                return Err(invalid(
                    "retained static-attention arenas disagree on physical identity",
                ));
            }
        }
        let maximum_bytes = per_sequence_bytes
            .checked_mul(u64::from(sequence_capacity))
            .ok_or_else(|| invalid("retained static-attention byte capacity overflow"))?;
        Ok(Self {
            id: RetainedStaticAttentionRuntimeIdV2 {
                model_instance,
                allocation_generation,
                state_plan: plan.id,
                domain,
            },
            backend: plan.backend,
            plan,
            per_sequence_bytes,
            maximum_bytes,
            arenas,
            state: Mutex::new(RuntimeState {
                closed: false,
                sequences: HashMap::with_capacity(capacity),
                transactions: HashMap::with_capacity(capacity),
                free_slots: (0..capacity).rev().collect(),
            }),
            next_sequence: AtomicU64::new(1),
            next_transaction: AtomicU64::new(1),
        })
    }

    pub(crate) const fn id(&self) -> RetainedStaticAttentionRuntimeIdV2 {
        self.id
    }

    pub(crate) fn state_plan_v2(&self) -> &ResolvedStatePlan {
        &self.plan
    }

    pub(crate) fn sequence_capacity(&self) -> u32 {
        u32::try_from(self.arenas.len()).unwrap_or(u32::MAX)
    }

    pub(crate) const fn per_sequence_bytes(&self) -> u64 {
        self.per_sequence_bytes
    }

    pub(crate) const fn maximum_bytes(&self) -> u64 {
        self.maximum_bytes
    }

    pub(crate) fn register_sequence(&self) -> Result<RetainedStaticAttentionSequenceId> {
        let nonce = next_identity(&self.next_sequence, "sequence")?;
        let mut state = self.lock_state()?;
        if state.closed {
            return Err(invalid("retained static-attention runtime is closed"));
        }
        if state.sequences.len() >= self.arenas.len() {
            return Err(Error::Overloaded(
                "retained static-attention sequence capacity is full".into(),
            ));
        }
        state.sequences.insert(
            nonce,
            SequenceState {
                committed_slot: None,
                transaction_nonce: None,
            },
        );
        Ok(RetainedStaticAttentionSequenceId {
            runtime: self.id,
            nonce,
        })
    }

    pub(crate) fn begin_install(
        &self,
        sequence: RetainedStaticAttentionSequenceId,
        source_identity: [u8; 32],
        memory_tokens: u64,
    ) -> Result<RetainedStaticAttentionTransactionId> {
        self.authenticate_sequence(sequence)?;
        let nonce = next_identity(&self.next_transaction, "transaction")?;
        let mut state = self.lock_state()?;
        if state.closed {
            return Err(invalid("retained static-attention runtime is closed"));
        }
        let sequence_state = self.sequence_state(&state, sequence)?;
        if sequence_state.committed_slot.is_some() {
            return Err(invalid(
                "retained static-attention memory is immutable after initialization",
            ));
        }
        if sequence_state.transaction_nonce.is_some() {
            return Err(invalid(
                "retained static-attention sequence already has a transaction",
            ));
        }
        let slot = state.free_slots.pop().ok_or_else(|| {
            Error::Overloaded("retained static-attention staging capacity is full".into())
        })?;
        let intent = DomainStepIntent {
            domain: self.id.domain,
            expected_cursor: 0,
            target_cursor: memory_tokens,
            update: StateUpdateKind::StaticInitialize {
                source_identity,
                components: Vec::new(),
            },
        };
        if let Err(error) = self.lock_arena(slot)?.begin_install(&intent) {
            state.free_slots.push(slot);
            return Err(error);
        }
        state
            .sequences
            .get_mut(&sequence.nonce)
            .expect("authenticated sequence exists under the state lock")
            .transaction_nonce = Some(nonce);
        state.transactions.insert(
            nonce,
            TransactionState {
                sequence_nonce: sequence.nonce,
                staging_slot: slot,
            },
        );
        Ok(RetainedStaticAttentionTransactionId {
            runtime: self.id,
            nonce,
        })
    }

    pub(crate) fn stage_layer(
        &self,
        transaction: RetainedStaticAttentionTransactionId,
        layer: StaticAttentionLayerValue,
    ) -> Result<()> {
        // Keep the authenticated identity map locked until the physical write
        // completes. Abort cannot free and reassign this slot between lookup
        // and write, which closes the transaction-slot ABA window.
        let state = self.lock_state()?;
        let slot = self.transaction_slot_in_state(&state, transaction)?;
        self.lock_arena(slot)?.install_layer(layer)
    }

    pub(crate) fn commit_install(
        &self,
        transaction: RetainedStaticAttentionTransactionId,
    ) -> Result<()> {
        let mut state = self.lock_state()?;
        let slot = self.transaction_slot_in_state(&state, transaction)?;
        self.lock_arena(slot)?.commit_install()?;
        let transaction_state = state
            .transactions
            .remove(&transaction.nonce)
            .expect("authenticated transaction exists under the state lock");
        let sequence = state
            .sequences
            .get_mut(&transaction_state.sequence_nonce)
            .expect("transaction sequence exists under the state lock");
        sequence.committed_slot = Some(slot);
        sequence.transaction_nonce = None;
        Ok(())
    }

    pub(crate) fn abort_install(
        &self,
        transaction: RetainedStaticAttentionTransactionId,
    ) -> Result<()> {
        let mut state = self.lock_state()?;
        let slot = self.transaction_slot_in_state(&state, transaction)?;
        self.lock_arena(slot)?.reset_for_reuse()?;
        let transaction_state = state
            .transactions
            .remove(&transaction.nonce)
            .expect("authenticated transaction exists under the state lock");
        state
            .sequences
            .get_mut(&transaction_state.sequence_nonce)
            .expect("transaction sequence exists under the state lock")
            .transaction_nonce = None;
        state.free_slots.push(slot);
        Ok(())
    }

    pub(crate) fn install(
        &self,
        sequence: RetainedStaticAttentionSequenceId,
        source_identity: [u8; 32],
        layers: Vec<StaticAttentionLayerValue>,
    ) -> Result<()> {
        let memory_tokens = layers
            .first()
            .ok_or_else(|| invalid("retained static-attention install has no layers"))?
            .keys
            .dim(0)? as u64;
        let transaction = self.begin_install(sequence, source_identity, memory_tokens)?;
        for layer in layers {
            if let Err(error) = self.stage_layer(transaction, layer) {
                let _ = self.abort_install(transaction);
                return Err(error);
            }
        }
        if let Err(error) = self.commit_install(transaction) {
            let _ = self.abort_install(transaction);
            return Err(error);
        }
        Ok(())
    }

    pub(crate) fn read(
        &self,
        sequence: RetainedStaticAttentionSequenceId,
    ) -> Result<Option<StaticAttentionMetadata>> {
        let state = self.lock_state()?;
        let Some(slot) = self.sequence_state(&state, sequence)?.committed_slot else {
            return Ok(None);
        };
        self.lock_arena(slot)?.metadata()
    }

    pub(crate) fn attend(
        &self,
        sequence: RetainedStaticAttentionSequenceId,
        model_layer: u32,
        queries: &Tensor,
        rows: &[StaticAttentionRaggedRow],
        softmax_scale: f32,
    ) -> Result<Tensor> {
        let state = self.lock_state()?;
        let slot = self.committed_slot(&state, sequence)?;
        self.lock_arena(slot)?
            .attend(model_layer, queries, rows, softmax_scale)
    }

    /// One authenticated physical envelope for rows backed by distinct static
    /// memories. CPU uses a portable per-row inner loop after validating every
    /// sequence and query partition. Accelerators remain unavailable until an
    /// exact multi-memory provider is published.
    pub(crate) fn attend_batch_distinct(
        &self,
        model_layer: u32,
        queries: &Tensor,
        rows: &[RetainedStaticAttentionBatchRow],
        softmax_scale: f32,
    ) -> Result<Tensor> {
        if rows.is_empty() || queries.rank() != 3 {
            return Err(invalid(
                "retained static-attention batch requires rank-three queries and rows",
            ));
        }
        let total_queries = u32::try_from(queries.dim(0)?)
            .map_err(|_| invalid("retained static-attention query count exceeds u32"))?;
        let mut cursor = 0_u32;
        let mut identities = HashSet::with_capacity(rows.len());
        for row in rows {
            self.authenticate_sequence(row.sequence)?;
            if row.query_len == 0 || row.query_start != cursor || !identities.insert(row.sequence) {
                return Err(invalid("retained static-attention rows must be a canonical distinct-sequence partition"));
            }
            cursor = cursor
                .checked_add(row.query_len)
                .ok_or_else(|| invalid("retained static-attention query partition overflow"))?;
        }
        if cursor != total_queries {
            return Err(invalid(
                "retained static-attention rows do not cover the query tensor",
            ));
        }
        if self.backend != BackendKind::Cpu {
            return Err(invalid(
                "retained distinct-memory static-attention batching has no exact accelerator provider",
            ));
        }
        let state = self.lock_state()?;
        let mut outputs = Vec::with_capacity(rows.len());
        for row in rows {
            let slot = self.committed_slot(&state, row.sequence)?;
            let query = queries.narrow(
                0,
                usize::try_from(row.query_start)
                    .map_err(|_| invalid("query start exceeds usize"))?,
                usize::try_from(row.query_len)
                    .map_err(|_| invalid("query length exceeds usize"))?,
            )?;
            outputs.push(self.lock_arena(slot)?.attend(
                model_layer,
                &query,
                &[StaticAttentionRaggedRow {
                    query_start: 0,
                    query_len: row.query_len,
                }],
                softmax_scale,
            )?);
        }
        Tensor::cat(&outputs, 0).map_err(Error::from)
    }

    pub(crate) fn release_sequence(
        &self,
        sequence: RetainedStaticAttentionSequenceId,
    ) -> Result<()> {
        self.authenticate_sequence(sequence)?;
        let mut state = self.lock_state()?;
        let sequence_state = self.sequence_state(&state, sequence)?;
        if sequence_state.transaction_nonce.is_some() {
            return Err(invalid(
                "retained static-attention sequence has an active transaction",
            ));
        }
        if let Some(slot) = sequence_state.committed_slot {
            self.lock_arena(slot)?.reset_for_reuse()?;
            state.free_slots.push(slot);
        }
        state.sequences.remove(&sequence.nonce);
        Ok(())
    }

    pub(crate) fn close_and_validate_drained(&self) -> Result<()> {
        let mut state = self.lock_state()?;
        state.closed = true;
        if !state.transactions.is_empty() || !state.sequences.is_empty() {
            return Err(invalid(
                "retained static-attention runtime still has active owners",
            ));
        }
        Ok(())
    }

    fn authenticate_sequence(&self, sequence: RetainedStaticAttentionSequenceId) -> Result<()> {
        if sequence.runtime != self.id || sequence.nonce == 0 {
            return Err(invalid(
                "retained static-attention sequence belongs to another runtime",
            ));
        }
        Ok(())
    }

    fn authenticate_transaction(
        &self,
        transaction: RetainedStaticAttentionTransactionId,
    ) -> Result<()> {
        if transaction.runtime != self.id || transaction.nonce == 0 {
            return Err(invalid(
                "retained static-attention transaction belongs to another runtime",
            ));
        }
        Ok(())
    }

    fn sequence_state<'a>(
        &self,
        state: &'a RuntimeState,
        sequence: RetainedStaticAttentionSequenceId,
    ) -> Result<&'a SequenceState> {
        self.authenticate_sequence(sequence)?;
        state
            .sequences
            .get(&sequence.nonce)
            .ok_or_else(|| invalid("retained static-attention sequence is not active"))
    }

    fn committed_slot(
        &self,
        state: &RuntimeState,
        sequence: RetainedStaticAttentionSequenceId,
    ) -> Result<usize> {
        self.sequence_state(state, sequence)?
            .committed_slot
            .ok_or_else(|| invalid("retained static-attention sequence is not initialized"))
    }

    fn transaction_slot_in_state(
        &self,
        state: &RuntimeState,
        transaction: RetainedStaticAttentionTransactionId,
    ) -> Result<usize> {
        self.authenticate_transaction(transaction)?;
        let transaction_state = state
            .transactions
            .get(&transaction.nonce)
            .ok_or_else(|| invalid("retained static-attention transaction is not active"))?;
        if !state
            .sequences
            .contains_key(&transaction_state.sequence_nonce)
        {
            return Err(invalid(
                "retained static-attention transaction lost its sequence",
            ));
        }
        Ok(transaction_state.staging_slot)
    }

    fn lock_state(&self) -> Result<std::sync::MutexGuard<'_, RuntimeState>> {
        self.state
            .lock()
            .map_err(|_| invalid("retained static-attention runtime state is poisoned"))
    }

    fn lock_arena(
        &self,
        slot: usize,
    ) -> Result<std::sync::MutexGuard<'_, InvocationStaticAttentionArena>> {
        self.arenas
            .get(slot)
            .ok_or_else(|| invalid("retained static-attention slot is out of range"))?
            .lock()
            .map_err(|_| invalid("retained static-attention arena is poisoned"))
    }
}

fn next_identity(counter: &AtomicU64, label: &str) -> Result<u64> {
    counter
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
            value.checked_add(1)
        })
        .map_err(|_| {
            invalid(format!(
                "retained static-attention {label} identity exhausted"
            ))
        })
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::state::{negotiate_state_plan, StateBackendPlanRequest};
    use crate::kv::v2::{
        CheckpointPolicy, KeyEncoding, PlacementPolicy, PrefixPolicy, StateClock, StateDType,
        StateDomainHeader, StateDomainSpec, StateGroupId, StateGroupSpec,
        StaticAttentionDomainSpec, StaticAttentionLayerSpec, CURRENT_INFERENCE_STATE_ABI,
    };
    use std::sync::mpsc;
    use std::sync::TryLockError;
    use std::thread;

    fn contract(max_memory_tokens: u64) -> InferenceStateContract {
        InferenceStateContract {
            abi: CURRENT_INFERENCE_STATE_ABI,
            domains: vec![StateDomainSpec::StaticAttention(
                StaticAttentionDomainSpec {
                    header: StateDomainHeader {
                        id: StateDomainId::new(1),
                        scope: crate::kv::v2::StateScope::Retained,
                        clock: StateClock::EncoderTokens,
                        placement: PlacementPolicy::BackendLocal,
                        prefix: PrefixPolicy::Disabled,
                        checkpoint: CheckpointPolicy::None,
                    },
                    layers: vec![StaticAttentionLayerSpec {
                        model_layer: 0,
                        query_heads: 4,
                        kv_heads: 2,
                        key_head_dim: 2,
                        value_head_dim: 2,
                        key_encoding: KeyEncoding::Raw,
                    }],
                    max_memory_tokens,
                    accepted_dtypes: vec![StateDType::F32],
                },
            )],
            groups: vec![StateGroupSpec {
                id: StateGroupId::new(1),
                domains: vec![StateDomainId::new(1)],
                prefix_shareable: false,
            }],
        }
    }

    fn build_runtime(model: u64, capacity: u32) -> RetainedStaticAttentionRuntimeV2 {
        let contract = contract(4);
        let plan = Arc::new(
            negotiate_state_plan(
                &contract,
                &StateBackendPlanRequest {
                    backend: BackendKind::Cpu,
                    device_ordinal: None,
                    page_tokens_hint: None,
                    storage_dtype_hint: Some(StateDType::F32),
                },
            )
            .unwrap(),
        );
        RetainedStaticAttentionRuntimeV2::new(
            ModelInstanceId::new(model),
            1,
            &contract,
            plan,
            StateDomainId::new(1),
            capacity,
            Device::Cpu,
        )
        .unwrap()
    }

    fn layer(memory_tokens: usize, offset: f32) -> StaticAttentionLayerValue {
        let elements = memory_tokens * 2 * 2;
        StaticAttentionLayerValue {
            model_layer: 0,
            keys: Tensor::from_vec(
                (0..elements)
                    .map(|index| offset + (index + 1) as f32 / elements as f32)
                    .collect::<Vec<_>>(),
                (memory_tokens, 2, 2),
                &Device::Cpu,
            )
            .unwrap(),
            values: Tensor::from_vec(
                (0..elements)
                    .map(|index| offset + (index + 1) as f32)
                    .collect::<Vec<_>>(),
                (memory_tokens, 2, 2),
                &Device::Cpu,
            )
            .unwrap(),
        }
    }

    #[test]
    fn two_sequences_attend_through_one_distinct_memory_batch_envelope() {
        let runtime = build_runtime(1, 2);
        assert_eq!(runtime.maximum_bytes(), runtime.per_sequence_bytes() * 2);
        let first = runtime.register_sequence().unwrap();
        let second = runtime.register_sequence().unwrap();
        runtime
            .install(first, [1; 32], vec![layer(2, 0.0)])
            .unwrap();
        runtime
            .install(second, [2; 32], vec![layer(2, 100.0)])
            .unwrap();
        let queries = Tensor::from_vec(
            [1.0_f32, 0.0, 0.5, 0.5, 0.0, 1.0, -0.5, 0.5].repeat(2),
            (2, 4, 2),
            &Device::Cpu,
        )
        .unwrap();
        let output = runtime
            .attend_batch_distinct(
                0,
                &queries,
                &[
                    RetainedStaticAttentionBatchRow {
                        sequence: first,
                        query_start: 0,
                        query_len: 1,
                    },
                    RetainedStaticAttentionBatchRow {
                        sequence: second,
                        query_start: 1,
                        query_len: 1,
                    },
                ],
                1.0,
            )
            .unwrap();
        assert_eq!(output.dims(), &[2, 4, 2]);
        assert_ne!(
            output.narrow(0, 0, 1).unwrap().to_vec3::<f32>().unwrap(),
            output.narrow(0, 1, 1).unwrap().to_vec3::<f32>().unwrap()
        );
    }

    #[test]
    fn aborted_install_is_private_and_does_not_disturb_live_peer() {
        let runtime = build_runtime(2, 2);
        let live = runtime.register_sequence().unwrap();
        let failed = runtime.register_sequence().unwrap();
        runtime.install(live, [3; 32], vec![layer(2, 0.0)]).unwrap();
        let transaction = runtime.begin_install(failed, [4; 32], 2).unwrap();
        runtime.stage_layer(transaction, layer(2, 50.0)).unwrap();
        assert_eq!(runtime.read(failed).unwrap(), None);
        assert_eq!(
            runtime.read(live).unwrap().unwrap().source_identity,
            [3; 32]
        );
        runtime.abort_install(transaction).unwrap();
        assert_eq!(runtime.read(failed).unwrap(), None);
        runtime
            .install(failed, [5; 32], vec![layer(2, 75.0)])
            .unwrap();
        assert_eq!(
            runtime.read(failed).unwrap().unwrap().source_identity,
            [5; 32]
        );
    }

    #[test]
    fn mixed_release_reuses_capacity_and_foreign_identities_fail_closed() {
        let runtime = build_runtime(3, 2);
        let foreign = build_runtime(4, 1);
        let first = runtime.register_sequence().unwrap();
        let second = runtime.register_sequence().unwrap();
        runtime
            .install(first, [1; 32], vec![layer(1, 0.0)])
            .unwrap();
        runtime
            .install(second, [2; 32], vec![layer(1, 1.0)])
            .unwrap();
        assert!(foreign.read(first).is_err());
        runtime.release_sequence(first).unwrap();
        assert!(runtime.read(first).is_err());
        assert!(runtime.read(second).unwrap().is_some());
        let replacement = runtime.register_sequence().unwrap();
        runtime
            .install(replacement, [9; 32], vec![layer(1, 9.0)])
            .unwrap();
        let transaction = foreign
            .begin_install(foreign.register_sequence().unwrap(), [7; 32], 1)
            .unwrap();
        assert!(runtime.stage_layer(transaction, layer(1, 0.0)).is_err());
        foreign.abort_install(transaction).unwrap();
    }

    #[test]
    fn drain_closes_new_registration_until_all_sequences_release() {
        let runtime = build_runtime(5, 1);
        let sequence = runtime.register_sequence().unwrap();
        assert!(runtime.close_and_validate_drained().is_err());
        assert!(runtime.register_sequence().is_err());
        assert!(runtime.begin_install(sequence, [1; 32], 1).is_err());
        runtime.release_sequence(sequence).unwrap();
        runtime.close_and_validate_drained().unwrap();
    }

    #[test]
    fn begin_waiting_at_close_linearization_cannot_start_a_transaction() {
        let runtime = Arc::new(build_runtime(7, 1));
        let sequence = runtime.register_sequence().unwrap();
        let mut state = runtime.lock_state().unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let begin_runtime = runtime.clone();
        let begin = thread::spawn(move || {
            started_tx.send(()).unwrap();
            begin_runtime.begin_install(sequence, [3; 32], 1)
        });
        started_rx.recv().unwrap();
        state.closed = true;
        drop(state);
        assert!(begin.join().unwrap().is_err());
        runtime.release_sequence(sequence).unwrap();
        runtime.close_and_validate_drained().unwrap();
    }

    #[test]
    fn staged_write_serializes_abort_and_stale_transaction_cannot_hit_reused_slot() {
        let runtime = Arc::new(build_runtime(6, 1));
        let sequence = runtime.register_sequence().unwrap();
        let stale = runtime.begin_install(sequence, [1; 32], 1).unwrap();
        let slot = {
            let state = runtime.lock_state().unwrap();
            runtime.transaction_slot_in_state(&state, stale).unwrap()
        };
        let arena = runtime.lock_arena(slot).unwrap();
        let (stage_done_tx, stage_done_rx) = mpsc::channel();
        let stage_runtime = runtime.clone();
        let stage = thread::spawn(move || {
            let result = stage_runtime.stage_layer(stale, layer(1, 0.0));
            stage_done_tx.send(()).unwrap();
            result
        });
        loop {
            match runtime.state.try_lock() {
                Err(TryLockError::WouldBlock) => break,
                Err(TryLockError::Poisoned(_)) => panic!("runtime state poisoned"),
                Ok(guard) => {
                    drop(guard);
                    thread::yield_now();
                }
            }
        }
        let abort_runtime = runtime.clone();
        let abort = thread::spawn(move || abort_runtime.abort_install(stale));
        assert!(stage_done_rx.try_recv().is_err());
        drop(arena);
        stage.join().unwrap().unwrap();
        abort.join().unwrap().unwrap();

        let replacement = runtime.begin_install(sequence, [2; 32], 1).unwrap();
        assert!(runtime.stage_layer(stale, layer(1, 50.0)).is_err());
        runtime.stage_layer(replacement, layer(1, 100.0)).unwrap();
        runtime.commit_install(replacement).unwrap();
        assert_eq!(
            runtime.read(sequence).unwrap().unwrap().source_identity,
            [2; 32]
        );
    }
}
