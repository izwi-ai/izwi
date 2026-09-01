use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use crate::backends::BackendKind;
use crate::engine::ModelInstanceId;
use crate::error::{Error, Result};

use super::contract::{StateDomainId, StateGroupId};
use super::resolved::{ResolvedStatePlan, StatePlanId};

/// Runtime arena identity. Unlike a resolved plan, an arena is tied to one
/// exact loaded model and one allocation generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PhysicalArenaId {
    pub(crate) model_instance: ModelInstanceId,
    pub(crate) plan: StatePlanId,
    pub(crate) backend: BackendKind,
    pub(crate) device_ordinal: Option<u32>,
    pub(crate) generation: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PhysicalBlockRef {
    pub(crate) arena: PhysicalArenaId,
    pub(crate) group: StateGroupId,
    pub(crate) domain: StateDomainId,
    pub(crate) index: u32,
    pub(crate) slot_generation: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PhysicalSlotRef {
    pub(crate) block: PhysicalBlockRef,
    pub(crate) offset: u32,
}

/// Generation-pinned allocation lease acquired atomically by the manager.
/// The lease must retain its arena/block pins until every clone is dropped
/// after backend completion, closing the validation-to-dispatch TOCTOU gap.
pub(crate) trait PhysicalArenaLease: fmt::Debug + Send + Sync {
    fn contains_block(&self, block: PhysicalBlockRef) -> bool;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PreparedPagedAttentionRow {
    pub(crate) blocks: Vec<PhysicalBlockRef>,
    pub(crate) first_page_offset: u32,
    /// Canonical range in the flattened query/input token tensor.
    pub(crate) input_start: u32,
    pub(crate) query_len: u32,
    /// Visible context after every query token in this row is written.
    pub(crate) context_len: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PreparedPagedWrite {
    pub(crate) row: u32,
    pub(crate) slot: PhysicalSlotRef,
    pub(crate) source_token: u32,
}

/// Immutable, capability-bearing control metadata lowered once per execution
/// quantum and reused across every layer in the resolved domain.
#[derive(Debug, Clone)]
pub(crate) struct PreparedPagedAttentionBatch {
    pub(crate) lease: Arc<dyn PhysicalArenaLease>,
    pub(crate) plan: StatePlanId,
    pub(crate) arena: PhysicalArenaId,
    pub(crate) group: StateGroupId,
    pub(crate) domain: StateDomainId,
    pub(crate) page_tokens: u32,
    pub(crate) input_tokens: u32,
    pub(crate) rows: Vec<PreparedPagedAttentionRow>,
    pub(crate) writes: Vec<PreparedPagedWrite>,
}

impl PreparedPagedAttentionBatch {
    pub(crate) fn validate_against(&self, plan: &ResolvedStatePlan) -> Result<()> {
        if self.plan != plan.id || self.arena.plan != plan.id {
            return Err(invalid("prepared batch belongs to a different state plan"));
        }
        if self.arena.backend != plan.backend
            || self.arena.device_ordinal != plan.device_ordinal
            || self.arena.generation == 0
        {
            return Err(invalid(
                "prepared batch has an incompatible or stale arena identity",
            ));
        }
        let group = plan
            .paged_attention
            .iter()
            .find(|candidate| candidate.group == self.group && candidate.domain == self.domain)
            .ok_or_else(|| invalid("prepared batch references an unresolved state group"))?;
        if self.page_tokens != group.page_tokens {
            return Err(invalid("prepared batch page size does not match its plan"));
        }
        if self.rows.is_empty() {
            return Err(invalid("prepared paged-attention batch has no rows"));
        }

        let mut next_input = 0_u32;
        for row in &self.rows {
            if row.input_start != next_input || row.query_len == 0 {
                return Err(invalid(
                    "prepared rows must contain non-empty contiguous input ranges",
                ));
            }
            next_input = next_input
                .checked_add(row.query_len)
                .ok_or_else(|| invalid("prepared input range overflow"))?;
            if row.first_page_offset >= self.page_tokens {
                return Err(invalid("prepared row first-page offset is out of range"));
            }
            if row.context_len > 0 && row.blocks.is_empty() {
                return Err(invalid("non-empty prepared row has no block table"));
            }
            if row.query_len > row.context_len {
                return Err(invalid("prepared row query exceeds its final context"));
            }
            let available = u64::try_from(row.blocks.len())
                .ok()
                .and_then(|blocks| blocks.checked_mul(u64::from(self.page_tokens)))
                .and_then(|tokens| tokens.checked_sub(u64::from(row.first_page_offset)))
                .ok_or_else(|| invalid("prepared row token capacity overflow"))?;
            if u64::from(row.context_len) > available {
                return Err(invalid("prepared row context exceeds its block table"));
            }
            let mut row_blocks = HashSet::with_capacity(row.blocks.len());
            for block in &row.blocks {
                self.validate_block(*block)?;
                if !row_blocks.insert(block.index) {
                    return Err(invalid(
                        "prepared row aliases one physical block at multiple logical pages",
                    ));
                }
            }
        }
        if next_input != self.input_tokens {
            return Err(invalid(
                "prepared row input ranges do not cover the flattened input tensor",
            ));
        }

        let mut write_slots = HashSet::with_capacity(self.writes.len());
        let mut written_inputs = HashSet::with_capacity(self.writes.len());
        for write in &self.writes {
            self.validate_block(write.slot.block)?;
            if write.slot.offset >= self.page_tokens {
                return Err(invalid("prepared write offset is out of range"));
            }
            if !write_slots.insert(write.slot) {
                return Err(invalid("prepared batch writes one physical slot twice"));
            }
            let row = self
                .rows
                .get(write.row as usize)
                .ok_or_else(|| invalid("prepared write references an unknown row"))?;
            let input_end = row
                .input_start
                .checked_add(row.query_len)
                .ok_or_else(|| invalid("prepared input range overflow"))?;
            if write.source_token < row.input_start || write.source_token >= input_end {
                return Err(invalid(
                    "prepared write source token is outside its row input range",
                ));
            }
            if !written_inputs.insert((write.row, write.source_token)) {
                return Err(invalid("prepared input token is written more than once"));
            }

            let query_offset = write.source_token - row.input_start;
            let context_position = row
                .context_len
                .checked_sub(row.query_len)
                .and_then(|position| position.checked_add(query_offset))
                .ok_or_else(|| invalid("prepared write position overflow"))?;
            let physical_position = row
                .first_page_offset
                .checked_add(context_position)
                .ok_or_else(|| invalid("prepared write position overflow"))?;
            let block_index = physical_position / self.page_tokens;
            let expected_block = row
                .blocks
                .get(block_index as usize)
                .ok_or_else(|| invalid("prepared write has no corresponding row block"))?;
            if write.slot.block != *expected_block
                || write.slot.offset != physical_position % self.page_tokens
            {
                return Err(invalid(
                    "prepared write does not match its row's canonical physical slot",
                ));
            }
        }
        if written_inputs.len() != self.input_tokens as usize {
            return Err(invalid(
                "prepared writes do not cover every flattened input token exactly once",
            ));
        }
        Ok(())
    }

    fn validate_block(&self, block: PhysicalBlockRef) -> Result<()> {
        if block.arena != self.arena
            || block.group != self.group
            || block.domain != self.domain
            || block.slot_generation == 0
        {
            return Err(invalid(
                "prepared batch mixes arenas, generations, groups, or domains",
            ));
        }
        if !self.lease.contains_block(block) {
            return Err(invalid(
                "prepared block is not pinned by the batch's arena lease",
            ));
        }
        Ok(())
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::v2::resolved::test_plan;
    use crate::kv::v2::test_contract;

    fn arena(plan: StatePlanId, generation: u32) -> PhysicalArenaId {
        PhysicalArenaId {
            model_instance: ModelInstanceId::new(7),
            plan,
            backend: BackendKind::Cpu,
            device_ordinal: None,
            generation,
        }
    }

    fn block(arena: PhysicalArenaId, index: u32) -> PhysicalBlockRef {
        PhysicalBlockRef {
            arena,
            group: StateGroupId::new(1),
            domain: StateDomainId::new(1),
            index,
            slot_generation: 1,
        }
    }

    #[derive(Debug)]
    struct TestLease {
        arena: PhysicalArenaId,
        capacity: u32,
        slot_generation: u32,
    }

    impl PhysicalArenaLease for TestLease {
        fn contains_block(&self, block: PhysicalBlockRef) -> bool {
            block.arena == self.arena
                && block.group == StateGroupId::new(1)
                && block.domain == StateDomainId::new(1)
                && block.index < self.capacity
                && block.slot_generation == self.slot_generation
        }
    }

    #[test]
    fn prepared_batch_accepts_one_generation_safe_arena() {
        let contract = test_contract();
        let plan = test_plan(&contract);
        let arena = arena(plan.id, 1);
        let lease = Arc::new(TestLease {
            arena,
            capacity: 2,
            slot_generation: 1,
        });
        let batch = PreparedPagedAttentionBatch {
            lease,
            plan: plan.id,
            arena,
            group: StateGroupId::new(1),
            domain: StateDomainId::new(1),
            page_tokens: 16,
            input_tokens: 1,
            rows: vec![PreparedPagedAttentionRow {
                blocks: vec![block(arena, 0), block(arena, 1)],
                first_page_offset: 3,
                input_start: 0,
                query_len: 1,
                context_len: 20,
            }],
            writes: vec![PreparedPagedWrite {
                row: 0,
                slot: PhysicalSlotRef {
                    block: block(arena, 1),
                    offset: 6,
                },
                source_token: 0,
            }],
        };
        batch.validate_against(&plan).unwrap();
    }

    #[test]
    fn prepared_batch_rejects_mixed_arena_generations() {
        let contract = test_contract();
        let plan = test_plan(&contract);
        let current = arena(plan.id, 2);
        let stale = arena(plan.id, 1);
        let lease = Arc::new(TestLease {
            arena: current,
            capacity: 2,
            slot_generation: 1,
        });
        let batch = PreparedPagedAttentionBatch {
            lease,
            plan: plan.id,
            arena: current,
            group: StateGroupId::new(1),
            domain: StateDomainId::new(1),
            page_tokens: 16,
            input_tokens: 1,
            rows: vec![PreparedPagedAttentionRow {
                blocks: vec![block(current, 0), block(stale, 1)],
                first_page_offset: 0,
                input_start: 0,
                query_len: 1,
                context_len: 17,
            }],
            writes: vec![PreparedPagedWrite {
                row: 0,
                slot: PhysicalSlotRef {
                    block: block(stale, 1),
                    offset: 0,
                },
                source_token: 0,
            }],
        };
        assert!(batch.validate_against(&plan).is_err());
    }

    #[test]
    fn prepared_batch_rejects_stale_slots_and_out_of_capacity_blocks() {
        let contract = test_contract();
        let plan = test_plan(&contract);
        let arena = arena(plan.id, 1);
        let stale_lease = Arc::new(TestLease {
            arena,
            capacity: 2,
            slot_generation: 2,
        });
        let mut batch = PreparedPagedAttentionBatch {
            lease: stale_lease,
            plan: plan.id,
            arena,
            group: StateGroupId::new(1),
            domain: StateDomainId::new(1),
            page_tokens: 16,
            input_tokens: 1,
            rows: vec![PreparedPagedAttentionRow {
                blocks: vec![block(arena, 0), block(arena, 1)],
                first_page_offset: 0,
                input_start: 0,
                query_len: 1,
                context_len: 17,
            }],
            writes: vec![PreparedPagedWrite {
                row: 0,
                slot: PhysicalSlotRef {
                    block: block(arena, 1),
                    offset: 0,
                },
                source_token: 0,
            }],
        };
        assert!(batch.validate_against(&plan).is_err());

        batch.lease = Arc::new(TestLease {
            arena,
            capacity: 1,
            slot_generation: 1,
        });
        assert!(batch.validate_against(&plan).is_err());
    }

    #[test]
    fn prepared_batch_rejects_intra_row_page_aliases() {
        let contract = test_contract();
        let plan = test_plan(&contract);
        let arena = arena(plan.id, 1);
        let batch = PreparedPagedAttentionBatch {
            lease: Arc::new(TestLease {
                arena,
                capacity: 2,
                slot_generation: 1,
            }),
            plan: plan.id,
            arena,
            group: StateGroupId::new(1),
            domain: StateDomainId::new(1),
            page_tokens: 16,
            input_tokens: 1,
            rows: vec![PreparedPagedAttentionRow {
                blocks: vec![block(arena, 0), block(arena, 0)],
                first_page_offset: 0,
                input_start: 0,
                query_len: 1,
                context_len: 17,
            }],
            writes: vec![PreparedPagedWrite {
                row: 0,
                slot: PhysicalSlotRef {
                    block: block(arena, 0),
                    offset: 0,
                },
                source_token: 0,
            }],
        };
        assert!(batch.validate_against(&plan).is_err());
    }
}
