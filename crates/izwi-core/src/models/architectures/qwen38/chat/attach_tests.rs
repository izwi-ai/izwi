//! DS1.5: a fresh session that attaches a published prefix cursor must be
//! numerically identical to a fresh run computing the same prompt, and the
//! attach-span guards must fail closed. The attach is simulated with the real
//! primitives the manager uses: shared physical pages below the fork cursor,
//! plus a forked tensor-state sequence seeded from the publisher's committed
//! snapshot.

use super::*;
use crate::backends::kv::{CpuKvArena, KvArenaConfig, KvLayerConfig};
use crate::backends::state::{
    negotiate_state_plan, PhysicalStateSequenceId, PhysicalStateTransactionId,
    StateBackendPlanRequest, StateComponentValue, TensorStateArena, TensorStateCapacity,
};
use crate::engine::ModelInstanceId;
use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
use crate::models::architectures::qwen38::cache::{
    CONVOLUTION_STATE_DOMAIN, RECURRENT_STATE_DOMAIN,
};
use candle_core::{DType, Device};
use std::sync::Arc;

/// One shared paged arena with independently addressable block views, so a
/// "publisher" and an "attacher" can share prefix pages without clobbering
/// each other's private suffix pages.
struct PagedFixture {
    arena: Arc<CpuKvArena>,
    id: KvArenaId,
    group: KvGroupId,
    binding: KvLayerBinding,
}

impl PagedFixture {
    fn new(model_layer: u32) -> Self {
        let id = KvArenaId {
            model_instance: ModelInstanceId::new(91),
            backend: BackendKind::Cpu,
            device_ordinal: None,
            generation: 1,
        };
        let group = KvGroupId::new(model_layer);
        let binding = KvLayerBinding {
            model_layer,
            physical_layer: 0,
        };
        let arena = Arc::new(
            CpuKvArena::new(KvArenaConfig {
                id,
                group,
                page_tokens: 4,
                capacity_pages: 12,
                growth: None,
                dtype: DType::F32,
                layers: vec![KvLayerConfig {
                    binding,
                    num_kv_heads: 1,
                    key_head_dim: 2,
                    value_head_dim: 2,
                }],
            })
            .unwrap(),
        );
        Self {
            arena,
            id,
            group,
            binding,
        }
    }

    fn view(&self, blocks: &[u32], context_len: usize) -> PhysicalPagedKvCache {
        PhysicalPagedKvCache::new(
            self.arena.clone(),
            vec![self.binding],
            blocks
                .iter()
                .map(|&index| CacheBlockRef {
                    arena: self.id,
                    group: self.group,
                    index,
                    slot_generation: 1,
                })
                .collect(),
            context_len,
        )
        .unwrap()
    }
}

fn greedy_config() -> ChatGenerationConfig {
    ChatGenerationConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        seed: 42,
        ..Default::default()
    }
}

fn prepared_prompt() -> Qwen38PreparedPrompt {
    Qwen38PreparedPrompt {
        prompt_ids: (0..8).collect(),
        prompt_positions: (0..8).map(|position| [position; 3]).collect(),
        next_text_position: 8,
    }
}

fn tensor_arena(model: &Qwen38ChatModel) -> TensorStateArena {
    let contract = model
        .managed_composite_cache_contract(DType::F32, 4)
        .unwrap();
    let plan = negotiate_state_plan(
        &contract,
        &StateBackendPlanRequest {
            backend: BackendKind::Cpu,
            device_ordinal: None,
            page_tokens_hint: None,
            storage_dtype_hint: None,
        },
    )
    .unwrap();
    let capacity = TensorStateCapacity::for_plan(&plan, 8, 8).unwrap();
    TensorStateArena::new(Arc::new(plan), capacity, Device::Cpu).unwrap()
}

/// Publish side: prefill `[0, fork_cursor)` into the publisher's cache and
/// stage the hybrid tensor state at the same cursor, mirroring the aligned
/// snapshot commit the manager performs before publishing.
fn publish_prefix(
    model: &Qwen38ChatModel,
    prepared: &Qwen38PreparedPrompt,
    target: &PagedFixture,
    mtp: &PagedFixture,
    arena: &TensorStateArena,
    fork_cursor: usize,
) {
    let config = greedy_config();
    let mut state = model
        .begin_chunked_prefill_state_physical(
            &[],
            4,
            &config,
            Some(prepared),
            target.view(&[0, 1], 0),
            Some(mtp.view(&[0, 1], 0)),
        )
        .unwrap();
    assert!(!model
        .continue_chunked_prefill_physical(
            &mut state,
            &[],
            &config,
            Some(prepared),
            0,
            fork_cursor,
            prepared.prompt_ids.len(),
        )
        .unwrap());
    assert_eq!(state.physical_kv.context_len(), fork_cursor);
    let sequence = PhysicalStateSequenceId::new(1).unwrap();
    arena.register(sequence).unwrap();
    let transaction = PhysicalStateTransactionId::new(1).unwrap();
    arena.begin(transaction, sequence).unwrap();
    state.stage_tensor_state(arena, transaction.get()).unwrap();
    arena.commit(transaction, fork_cursor as u64).unwrap();
}

/// Fork the publisher's committed tensor snapshot into a fresh sequence at the
/// attach cursor — the manager's DS1.2b attach-by-fork recipe.
fn fork_tensor_state(
    arena: &TensorStateArena,
    fork_cursor: usize,
    attached_sequence: u64,
) -> PhysicalStateSequenceId {
    let publisher = PhysicalStateSequenceId::new(1).unwrap();
    let attached = PhysicalStateSequenceId::new(attached_sequence).unwrap();
    arena.register(attached).unwrap();
    let transaction = PhysicalStateTransactionId::new(2).unwrap();
    arena.begin(transaction, attached).unwrap();
    for domain in [RECURRENT_STATE_DOMAIN, CONVOLUTION_STATE_DOMAIN] {
        let snapshot = arena
            .read(publisher, domain)
            .unwrap()
            .unwrap_or_else(|| panic!("publisher has committed state for domain {domain:?}"));
        assert_eq!(snapshot.cursor, fork_cursor as u64);
        let components: Vec<StateComponentValue> = snapshot
            .components
            .iter()
            .map(|component| StateComponentValue {
                component: component.component,
                tensor: component.tensor.clone(),
            })
            .collect();
        arena
            .stage_replace(transaction, domain, 0, fork_cursor as u64, components)
            .unwrap();
    }
    arena.commit(transaction, fork_cursor as u64).unwrap();
    attached
}

/// Attach side: begin chunked prefill on a reservation whose physical cursor
/// already holds the shared prefix pages and the forked tensor state.
fn attached_state(
    model: &Qwen38ChatModel,
    prepared: &Qwen38PreparedPrompt,
    target: &PagedFixture,
    mtp: &PagedFixture,
    arena: &TensorStateArena,
    fork_cursor: usize,
    attached_sequence: u64,
) -> ChatDecodeState {
    let config = greedy_config();
    let mut state = model
        .begin_chunked_prefill_state_physical(
            &[],
            4,
            &config,
            Some(prepared),
            // Block 0 is the publisher's shared prefix page; blocks 4 and 5
            // are the attacher's private suffix pages.
            target.view(&[0, 5, 6], fork_cursor),
            Some(mtp.view(&[7, 8, 9], 0)),
        )
        .unwrap();
    let sequence = fork_tensor_state(arena, fork_cursor, attached_sequence);
    state.bind_tensor_sequence(sequence.get()).unwrap();
    state.restore_tensor_state(arena).unwrap();
    state
}

fn decode_to_completion(model: &Qwen38ChatModel, state: &mut ChatDecodeState) -> Vec<u32> {
    let mut generated = Vec::new();
    while !state.finished && generated.len() < 4 {
        model.decode_quantum(state, 1).unwrap();
        generated = state.generated_ids.clone();
    }
    assert!(state.finished, "decode reached the generation budget");
    generated
}

#[test]
fn attached_prefill_matches_the_fresh_run_and_fails_closed_on_bad_spans() {
    let model = crate::models::architectures::qwen38::chat::recovery_tests::model_fixture(true);
    let prepared = prepared_prompt();
    let total = prepared.prompt_ids.len();
    let fork_cursor = 4;
    // The hybrid fixture's full-attention layer is model layer 1 (layer 0 is
    // linear attention) and the MTP head binds model layer 2 (= block_count).
    let target = PagedFixture::new(1);
    let mtp = PagedFixture::new(2);
    let arena = tensor_arena(&model);

    // Publisher: pages [0, 4) plus the tensor snapshot become shareable.
    publish_prefix(&model, &prepared, &target, &mtp, &arena, fork_cursor);

    // Fresh reference: the whole prompt computed from zero.
    let mut reference = model
        .begin_chunked_prefill_state_physical(
            &[],
            4,
            &greedy_config(),
            Some(&prepared),
            target.view(&[2, 3, 4], 0),
            Some(mtp.view(&[2, 3, 4, 5, 6], 0)),
        )
        .unwrap();
    assert!(model
        .continue_chunked_prefill_physical(
            &mut reference,
            &[],
            &greedy_config(),
            Some(&prepared),
            0,
            total,
            total,
        )
        .unwrap());
    assert_eq!(reference.physical_kv.context_len(), total);
    let reference_ids = decode_to_completion(&model, &mut reference);
    assert!(!reference_ids.is_empty());

    // Attached session: pages [0, 4) and the tensor snapshot are already
    // resident. The caller clips the first span to the state's cursor, so
    // the model feeds only the private residual [4, 8).
    let mut attached = attached_state(&model, &prepared, &target, &mtp, &arena, fork_cursor, 2);
    assert!(model
        .continue_chunked_prefill_physical(
            &mut attached,
            &[],
            &greedy_config(),
            Some(&prepared),
            fork_cursor,
            total,
            total,
        )
        .unwrap());
    assert_eq!(attached.physical_kv.context_len(), total);
    assert_eq!(
        reference_ids,
        decode_to_completion(&model, &mut attached),
        "attached session must generate the fresh run's tokens"
    );

    // Chunked variant: the first span carries only part of the residual.
    let mut chunked = attached_state(&model, &prepared, &target, &mtp, &arena, fork_cursor, 3);
    assert!(!model
        .continue_chunked_prefill_physical(
            &mut chunked,
            &[],
            &greedy_config(),
            Some(&prepared),
            fork_cursor,
            6,
            total,
        )
        .unwrap());
    assert_eq!(chunked.physical_kv.context_len(), 6);
    assert!(model
        .continue_chunked_prefill_physical(
            &mut chunked,
            &[],
            &greedy_config(),
            Some(&prepared),
            6,
            total,
            total,
        )
        .unwrap());
    assert_eq!(reference_ids, decode_to_completion(&model, &mut chunked));

    // Fail closed: an unclipped span that ignores the attached cursor, an
    // empty residual span, and stale replays are rejections, never silent
    // appends.
    let mut guarded = attached_state(&model, &prepared, &target, &mtp, &arena, fork_cursor, 4);
    assert!(model
        .continue_chunked_prefill_physical(
            &mut guarded,
            &[],
            &greedy_config(),
            Some(&prepared),
            0,
            total,
            total,
        )
        .is_err());
    assert!(model
        .continue_chunked_prefill_physical(
            &mut guarded,
            &[],
            &greedy_config(),
            Some(&prepared),
            fork_cursor,
            fork_cursor,
            total,
        )
        .is_err());
    // A partial first span is valid (and incomplete); afterwards the logical
    // cursor advanced, so replaying it is a rejection.
    assert!(!model
        .continue_chunked_prefill_physical(
            &mut guarded,
            &[],
            &greedy_config(),
            Some(&prepared),
            fork_cursor,
            6,
            total,
        )
        .unwrap());
    assert_eq!(guarded.physical_kv.context_len(), 6);
    assert!(model
        .continue_chunked_prefill_physical(
            &mut guarded,
            &[],
            &greedy_config(),
            Some(&prepared),
            fork_cursor,
            6,
            total,
        )
        .is_err());
    assert!(model
        .continue_chunked_prefill_physical(
            &mut guarded,
            &[],
            &greedy_config(),
            Some(&prepared),
            fork_cursor,
            total,
            total,
        )
        .is_err());

    // A fresh (non-attached) sequence restores nothing and stays usable.
    let mut fresh = model
        .begin_chunked_prefill_state_physical(
            &[],
            4,
            &greedy_config(),
            Some(&prepared),
            target.view(&[7, 8, 9], 0),
            Some(mtp.view(&[2, 3, 4, 5, 6], 0)),
        )
        .unwrap();
    let empty_sequence = PhysicalStateSequenceId::new(5).unwrap();
    arena.register(empty_sequence).unwrap();
    fresh.bind_tensor_sequence(empty_sequence.get()).unwrap();
    fresh.restore_tensor_state(&arena).unwrap();
    assert!(model
        .continue_chunked_prefill_physical(
            &mut fresh,
            &[],
            &greedy_config(),
            Some(&prepared),
            0,
            total,
            total,
        )
        .unwrap());
    assert_eq!(
        reference_ids,
        decode_to_completion(&model, &mut fresh),
        "an empty-sequence restore must not perturb a fresh session"
    );
}
