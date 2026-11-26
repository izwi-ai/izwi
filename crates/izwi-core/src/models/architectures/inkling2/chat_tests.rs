//! The served chat model on engine-owned state, against the reference
//! fixture and the trunk's own dense state.

use candle_core::{DType, Device};

use super::chat::{ChatDecodeState, Inkling2ChatModel};
use super::checkpoint::Inkling2Checkpoint;
use super::config::tests::{fixture, tiny};
use super::prompt::END_SAMPLING;
use super::reply::tests::test_tokenizer_sized;
use super::tests::{golden, max_error, tiny_trunk};
use crate::backends::DeviceProfile;
use crate::model::ModelVariant;
use crate::models::shared::attention::physical::{
    test_physical_caches_for_model_layers_on, PhysicalPagedKvCache,
};
use crate::models::shared::chat::ChatGenerationConfig;

fn tiny_model() -> Inkling2ChatModel {
    let mut config = tiny();
    // ASCII byte tokens, then the special tokens.
    let tokenizer = test_tokenizer_sized(128);
    config.eos_token_id = tokenizer.token_to_id(END_SAMPLING).unwrap();
    let checkpoint = Inkling2Checkpoint::open(&fixture("inkling2_golden"), &tiny()).unwrap();
    Inkling2ChatModel::from_parts(
        ModelVariant::Inkling2Mini,
        DeviceProfile::cpu(),
        config,
        tokenizer,
        &checkpoint,
    )
    .unwrap()
}

fn caches(model: &Inkling2ChatModel, rows: usize) -> Vec<PhysicalPagedKvCache> {
    let config = model.config();
    let layers = (0..config.application_count() as u32).collect::<Vec<_>>();
    test_physical_caches_for_model_layers_on(
        &Device::Cpu,
        DType::F32,
        &layers,
        config.full.kv_heads,
        config.full.head_dim,
        96,
        rows,
    )
}

fn values(tensor: &candle_core::Tensor) -> Vec<f32> {
    tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

/// Prefill in spans on the paged cache gives the reference's logits for the
/// last prompt token.
#[test]
fn managed_prefill_matches_the_reference_logits() {
    let golden = golden();
    let model = tiny_model();
    let vocab = model.config().unpadded_vocab;
    let config = ChatGenerationConfig::default();
    for spans in [vec![golden.tokens.len()], vec![14, 5, 1, 20]] {
        let prompt = &golden.tokens[..spans.iter().sum::<usize>()];
        let mut state = model
            .begin_resumable_prefill_managed(prompt, 4, &config, caches(&model, 1).remove(0))
            .unwrap();
        let mut at = 0;
        for span in &spans {
            let complete = model
                .continue_resumable_prefill(&mut state, prompt, at, at + span)
                .unwrap();
            at += span;
            assert_eq!(complete, at == prompt.len());
            assert_eq!(state.prefill_progress(), at);
        }
        let logits = values(state_logits(&state));
        let row = prompt.len() - 1;
        let error = max_error(&logits, &golden.logits[row * vocab..(row + 1) * vocab]);
        assert!(error <= 2e-3, "spans {spans:?}: logit error {error}");
    }
}

fn state_logits(state: &ChatDecodeState) -> &candle_core::Tensor {
    state
        .unconsumed_logits_for_test()
        .expect("prefill left logits")
}

/// Greedy tokens from the reference trunk on its own dense state.
fn reference_greedy(prompt: &[u32], steps: usize) -> Vec<u32> {
    let trunk = tiny_trunk(&Device::Cpu);
    let mut state = trunk.new_state().unwrap();
    let mut hidden = trunk.forward(prompt, &mut state, None).unwrap();
    let mut out = Vec::new();
    for _ in 0..steps {
        let last = hidden.narrow(0, hidden.dim(0).unwrap() - 1, 1).unwrap();
        let logits = values(&trunk.logits(&last).unwrap());
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        out.push(next);
        hidden = trunk.forward(&[next], &mut state, None).unwrap();
    }
    out
}

fn generate(model: &Inkling2ChatModel, state: &mut ChatDecodeState, steps: usize) -> Vec<u32> {
    for _ in 0..steps {
        if model.decode_step(state).unwrap().finished {
            break;
        }
    }
    state.generated_ids_for_test().to_vec()
}

/// Greedy decode on the paged cache picks the tokens the dense reference
/// picks.
#[test]
fn managed_greedy_decode_matches_the_dense_reference() {
    let golden = golden();
    let model = tiny_model().without_turn_end_for_test();
    let prompt = &golden.tokens[..golden.prompt_tokens];
    let steps = 24;
    let want = reference_greedy(prompt, steps);
    let config = ChatGenerationConfig::default();
    let mut state = model
        .start_decode_managed(prompt, steps, &config, caches(&model, 1).remove(0))
        .unwrap();
    let got = generate(&model, &mut state, steps);
    assert_eq!(got, want);
    // Not one token repeated: the state really advances.
    assert!(got.iter().collect::<std::collections::BTreeSet<_>>().len() > 3);
}

/// A token that ends the turn finishes the reply and is not generated
/// text; the reference's tokens come before it.
#[test]
fn a_turn_ending_token_finishes_the_reply() {
    let golden = golden();
    let model = tiny_model();
    let prompt = &golden.tokens[..golden.prompt_tokens];
    let want = reference_greedy(prompt, 24);
    let ends = (128..141u32).collect::<Vec<_>>();
    let first_end = want.iter().position(|token| ends.contains(token));
    let config = ChatGenerationConfig::default();
    let mut state = model
        .start_decode_managed(prompt, 24, &config, caches(&model, 1).remove(0))
        .unwrap();
    let got = generate(&model, &mut state, 24);
    assert!(state.finished_for_test());
    // Message tokens that do not end the turn stay in the journal.
    assert_eq!(got, want[..got.len()]);
    if let Some(first_end) = first_end {
        assert!(got.len() <= first_end);
    }
    // A finished state stays finished.
    assert!(model.decode_step(&mut state).unwrap().finished);
}

/// A quantum that is rolled back leaves no trace: the same step again gives
/// the same token and text.
#[test]
fn a_rolled_back_quantum_repeats_exactly() {
    let golden = golden();
    let model = tiny_model().without_turn_end_for_test();
    let prompt = &golden.tokens[..golden.prompt_tokens];
    let config = ChatGenerationConfig::default();
    let mut state = model
        .start_decode_managed(prompt, 16, &config, caches(&model, 1).remove(0))
        .unwrap();
    generate(&model, &mut state, 3);
    let before = state.generated_ids_for_test().to_vec();
    let view = state.cache_view_for_test();
    let checkpoint = state.begin_managed_quantum(view).unwrap();
    let first = generate(&model, &mut state, 2);
    assert_eq!(first.len(), before.len() + 2);
    state.rollback_managed_quantum(checkpoint);
    assert_eq!(state.generated_ids_for_test(), before);
    let again = generate(&model, &mut state, 2);
    assert_eq!(again, first);
}

/// A session released for capacity and replayed into a fresh reservation
/// continues with the tokens of the uninterrupted run.
#[test]
fn a_replayed_session_resumes_exactly() {
    let golden = golden();
    let model = tiny_model().without_turn_end_for_test();
    let prompt = &golden.tokens[..golden.prompt_tokens];
    let config = ChatGenerationConfig::default();
    let steps = 12;
    let mut whole = model
        .start_decode_managed(prompt, steps, &config, caches(&model, 1).remove(0))
        .unwrap();
    let want = generate(&model, &mut whole, steps);
    assert_eq!(want.len(), steps);

    let mut state = model
        .start_decode_managed(prompt, steps, &config, caches(&model, 1).remove(0))
        .unwrap();
    generate(&model, &mut state, 3);
    let saved = state.replay_checkpoint().unwrap();
    assert_eq!(saved.replay_tokens(), prompt.len() + 2);
    drop(state);
    let mut resumed = model
        .begin_replay_managed(&saved, caches(&model, 1).remove(0))
        .unwrap();
    assert!(model.decode_step(&mut resumed).is_err());
    // Replay in two spans.
    let end = saved.replay_tokens();
    assert!(!model.continue_replay(&mut resumed, prompt, 0, 9).unwrap());
    assert!(model.continue_replay(&mut resumed, prompt, 9, end).unwrap());
    let got = generate(&model, &mut resumed, steps);
    assert_eq!(got, want);
}

/// Rows of a batch do not affect each other.
#[test]
fn batch_rows_decode_as_they_do_alone() {
    let golden = golden();
    let model = tiny_model().without_turn_end_for_test();
    let config = ChatGenerationConfig::default();
    let prompts = [&golden.tokens[..14], &golden.tokens[3..20]];
    let alone = prompts
        .iter()
        .map(|prompt| {
            let mut state = model
                .start_decode_managed(prompt, 8, &config, caches(&model, 1).remove(0))
                .unwrap();
            generate(&model, &mut state, 5)
        })
        .collect::<Vec<_>>();
    let mut pool = caches(&model, 2);
    let mut states = prompts
        .iter()
        .map(|prompt| {
            let mut state = model
                .start_decode_managed(prompt, 8, &config, pool.remove(0))
                .unwrap();
            // The bootstrap token comes from the prefill logits.
            model.decode_step(&mut state).unwrap();
            state
        })
        .collect::<Vec<_>>();
    for _ in 0..4 {
        let mut rows = states
            .iter_mut()
            .filter(|state| !state.finished_for_test())
            .collect::<Vec<_>>();
        if rows.is_empty() {
            break;
        }
        model.decode_step_batch(&mut rows).unwrap();
    }
    for (state, alone) in states.iter().zip(&alone) {
        assert_eq!(state.generated_ids_for_test(), alone);
    }
}
