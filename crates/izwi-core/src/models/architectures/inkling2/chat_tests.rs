//! The served chat model on engine-owned state, against the reference
//! fixture and the trunk's own dense state.

use candle_core::{DType, Device};

use super::chat::{
    fitted_image_size, ChatDecodeState, ExpertBackendRequest, Inkling2ChatModel,
    Inkling2ImageInput, Inkling2PreparedPrompt, MAX_IMAGE_TOKENS,
};
use super::checkpoint::Inkling2Checkpoint;
use super::config::tests::{fixture, tiny};
use super::prompt::{END_SAMPLING, IMAGE_SLOT};
use super::reply::tests::test_tokenizer_sized;
use super::tests::{golden, max_error, tiny_trunk};
use super::vision::image_tokens;
use super::vision_tests::image_golden;
use crate::backends::DeviceProfile;
use crate::model::ModelVariant;
use crate::models::shared::attention::physical::{
    test_physical_caches_for_model_layers_on, PhysicalPagedKvCache,
};
use crate::models::shared::chat::{
    ChatGenerationConfig, ChatMediaInput, ChatMediaKind, ChatMessage, ChatRole,
    CHAT_IMAGE_PLACEHOLDER,
};

fn tiny_model() -> Inkling2ChatModel {
    tiny_model_on(DeviceProfile::cpu())
}

fn tiny_model_on(profile: DeviceProfile) -> Inkling2ChatModel {
    tiny_model_with(profile, ExpertBackendRequest::Auto)
}

fn tiny_model_with(profile: DeviceProfile, experts: ExpertBackendRequest) -> Inkling2ChatModel {
    let mut config = tiny();
    // ASCII byte tokens, then the special tokens.
    let tokenizer = test_tokenizer_sized(128);
    config.eos_token_id = tokenizer.token_to_id(END_SAMPLING).unwrap();
    config.image_token_id = tokenizer.token_to_id(IMAGE_SLOT).unwrap();
    let checkpoint = Inkling2Checkpoint::open(&fixture("inkling2_golden"), &tiny()).unwrap();
    Inkling2ChatModel::from_parts(
        ModelVariant::Inkling2Mini,
        profile,
        config,
        tokenizer,
        &checkpoint,
        experts,
    )
    .unwrap()
}

fn caches(model: &Inkling2ChatModel, rows: usize) -> Vec<PhysicalPagedKvCache> {
    caches_on(model, &Device::Cpu, DType::F32, rows)
}

fn caches_on(
    model: &Inkling2ChatModel,
    device: &Device,
    dtype: DType,
    rows: usize,
) -> Vec<PhysicalPagedKvCache> {
    caches_sized(model, device, dtype, rows, 96)
}

/// One cache of `tokens` capacity, for prompts the byte-level test tokenizer
/// makes long.
fn roomy_cache(model: &Inkling2ChatModel) -> PhysicalPagedKvCache {
    caches_sized(model, &Device::Cpu, DType::F32, 1, 1024).remove(0)
}

fn caches_sized(
    model: &Inkling2ChatModel,
    device: &Device,
    dtype: DType,
    rows: usize,
    tokens: usize,
) -> Vec<PhysicalPagedKvCache> {
    let config = model.config();
    let layers = (0..config.application_count() as u32).collect::<Vec<_>>();
    test_physical_caches_for_model_layers_on(
        device,
        dtype,
        &layers,
        config.full.kv_heads,
        config.full.head_dim,
        tokens,
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
    let checkpoint = state.begin_managed_quantum(view, None).unwrap();
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

/// A conversation with the fixture's two images, sealed with their features.
fn prepared_with_images(model: &Inkling2ChatModel) -> Inkling2PreparedPrompt {
    let images = image_golden()
        .images
        .iter()
        .map(|image| Inkling2ImageInput {
            height: image.shape[0],
            width: image.shape[1],
            channels: image.shape[2],
            pixels: image.bytes.clone(),
        })
        .collect::<Vec<_>>();
    let messages = [ChatMessage {
        role: ChatRole::User,
        content: format!("Compare {CHAT_IMAGE_PLACEHOLDER} with {CHAT_IMAGE_PLACEHOLDER}."),
    }];
    let config = ChatGenerationConfig::default();
    let prepared = model
        .prepare_prompt_with_images(&messages, &config, &images)
        .unwrap();
    // One image, or text that spells the placeholder a third time, does not
    // pair off.
    assert!(model
        .prepare_prompt_with_images(&messages, &config, &images[..1])
        .is_err());
    // A text prompt cannot carry media.
    let mut with_media = config.clone();
    with_media.request.media_inputs.push(ChatMediaInput {
        kind: ChatMediaKind::Image,
        source: "data:image/png;base64,".into(),
    });
    assert!(model.prompt_token_ids(&messages, &with_media).is_err());
    prepared
}

/// The dense reference for a prompt with images: the trunk on its own state,
/// with the image features in the slots' place.
fn reference_logits_with_images(
    model: &Inkling2ChatModel,
    prepared: &Inkling2PreparedPrompt,
) -> Vec<f32> {
    let golden = image_golden();
    let features = golden
        .images
        .iter()
        .flat_map(|image| image.features.iter().copied())
        .collect::<Vec<_>>();
    let hidden = model.config().hidden;
    let features =
        candle_core::Tensor::from_vec(features, (prepared.image_tokens(), hidden), &Device::Cpu)
            .unwrap();
    let trunk = tiny_trunk(&Device::Cpu);
    let ids = prepared.prompt_ids();
    let embeddings = trunk
        .embed_with_images(ids, model.config().image_token_id, &features, 0)
        .unwrap();
    let mut state = trunk.new_state().unwrap();
    let hidden = trunk
        .forward_embeddings(&embeddings, &mut state, None)
        .unwrap();
    let last = hidden.narrow(0, ids.len() - 1, 1).unwrap();
    values(&trunk.logits(&last).unwrap())
}

/// A prompt with images prefills to the reference's logits however its
/// spans cut the image slots, and its slots count the images' cells.
#[test]
fn a_prompt_with_images_prefills_to_the_reference_logits() {
    let model = tiny_model();
    let prepared = prepared_with_images(&model);
    let slot = model.config().image_token_id;
    let ids = prepared.prompt_ids();
    let slots = ids.iter().filter(|id| **id == slot).count();
    assert_eq!(slots, 6 + 1);
    assert_eq!(prepared.image_tokens(), slots);
    let want = reference_logits_with_images(&model, &prepared);
    let first_slot = ids.iter().position(|id| *id == slot).unwrap();
    let config = ChatGenerationConfig::default();
    // One span; then spans that end inside the first image and between the
    // two.
    for cuts in [vec![], vec![first_slot + 2, first_slot + 6, ids.len() - 2]] {
        let mut state = model
            .begin_prepared_prefill_managed(&prepared, 4, &config, caches(&model, 1).remove(0))
            .unwrap();
        let mut at = 0;
        for end in cuts.into_iter().chain([ids.len()]) {
            let complete = model
                .continue_resumable_prefill(&mut state, ids, at, end)
                .unwrap();
            assert_eq!(complete, end == ids.len());
            at = end;
        }
        let error = max_error(&values(state_logits(&state)), &want);
        // The reference takes the fixture's features, the model its own
        // vision tower's: rounding apart, as the logits then are.
        assert!(error <= 2e-3, "logit error with images {error}");
    }
    // The same tokens without the features are another prompt.
    let plain = model
        .start_decode_managed(ids, 4, &config, caches(&model, 1).remove(0))
        .unwrap();
    assert!(max_error(&values(state_logits(&plain)), &want) > 1e-2);
}

/// A session with images that is released and replayed continues with the
/// tokens of the uninterrupted run: the replay has the image features.
#[test]
fn a_replayed_session_with_images_resumes_exactly() {
    let model = tiny_model().without_turn_end_for_test();
    let prepared = prepared_with_images(&model);
    let ids = prepared.prompt_ids();
    let config = ChatGenerationConfig::default();
    let steps = 8;
    let mut whole = model
        .start_prepared_decode_managed(&prepared, steps, &config, caches(&model, 1).remove(0))
        .unwrap();
    let want = generate(&model, &mut whole, steps);
    assert_eq!(want.len(), steps);

    let mut state = model
        .start_prepared_decode_managed(&prepared, steps, &config, caches(&model, 1).remove(0))
        .unwrap();
    generate(&model, &mut state, 3);
    let saved = state.replay_checkpoint().unwrap();
    drop(state);
    let mut resumed = model
        .begin_replay_managed(&saved, caches(&model, 1).remove(0))
        .unwrap();
    let end = saved.replay_tokens();
    // The first span ends inside an image, the second runs into the
    // generated tokens.
    let slot = model.config().image_token_id;
    let cut = ids.iter().position(|id| *id == slot).unwrap() + 3;
    assert!(!model.continue_replay(&mut resumed, ids, 0, cut).unwrap());
    assert!(model.continue_replay(&mut resumed, ids, cut, end).unwrap());
    assert_eq!(generate(&model, &mut resumed, steps), want);
}

#[test]
fn a_large_image_is_fitted_to_the_token_limit() {
    // Within the limit: untouched, whatever its shape.
    assert_eq!(
        fitted_image_size(1600, 1600, 40, MAX_IMAGE_TOKENS),
        (1600, 1600)
    );
    assert_eq!(fitted_image_size(7, 10, 4, 6), (7, 10));
    // A long strip within the limit is untouched too.
    assert_eq!(
        fitted_image_size(8192, 100, 40, MAX_IMAGE_TOKENS),
        (8192, 100)
    );
    for (height, width) in [(4000, 3000), (8192, 2000), (1601, 1600)] {
        let (h, w) = fitted_image_size(height, width, 40, MAX_IMAGE_TOKENS);
        let tokens = image_tokens(h, w, 40);
        assert!(tokens <= MAX_IMAGE_TOKENS, "{height}x{width} -> {h}x{w}");
        // It is not shrunk further than it must be.
        assert!(
            tokens * 10 >= MAX_IMAGE_TOKENS * 8,
            "{height}x{width} -> {tokens}"
        );
        // The shape is kept.
        let (want, got) = (height as f64 / width as f64, h as f64 / w as f64);
        assert!(
            (got / want - 1.0).abs() < 0.05,
            "{height}x{width} -> {h}x{w}"
        );
    }
}

/// Decodes until the reply ends and returns its text.
fn reply_text(model: &Inkling2ChatModel, state: &mut ChatDecodeState, limit: usize) -> String {
    for _ in 0..limit {
        let step = model.decode_step(state).unwrap();
        if step.finished {
            return step.text;
        }
    }
    panic!("the reply did not end in {limit} steps");
}

fn weather_tools() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({"type": "function", "function": {"name": "set_mode", "parameters": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["eco", "boost"]},
                "level": {"type": "integer", "minimum": 0, "maximum": 9}
            },
            "required": ["mode", "level"],
            "additionalProperties": false
        }}}),
    ]
}

/// A forced call on random weights: the sampler can only write arguments the
/// tool's schema allows, then close the message and end the turn, and the
/// reply reads as one call to that tool.
#[test]
fn a_forced_call_is_held_to_the_tools_schema() {
    use crate::chat_tools::{parse_answer, ToolCallSyntax, ToolChoice, ToolDialect, ToolPlan};
    let model = tiny_model();
    let plan = ToolPlan::new(
        ToolDialect::HermesJson,
        weather_tools(),
        ToolChoice::Named("set_mode".into()),
        false,
    )
    .unwrap();
    let messages = [ChatMessage {
        role: ChatRole::User,
        content: "Go fast.".into(),
    }];
    for seed in 1..=6u64 {
        let mut config = ChatGenerationConfig {
            temperature: 1.0,
            top_p: 1.0,
            seed,
            ..ChatGenerationConfig::default()
        };
        config.request.tools = weather_tools();
        config.request.response_prefix = plan.response_prefix().unwrap();
        config.request.tool_constraint = plan
            .constraint_in(false, ToolCallSyntax::InklingMessage)
            .map(Box::new);
        assert!(config.request.tool_constraint.is_some());
        let prompt = model.prompt_token_ids(&messages, &config).unwrap();
        let mut state = model
            .start_decode_managed(&prompt, 200, &config, roomy_cache(&model))
            .unwrap();
        let text = reply_text(&model, &mut state, 200);
        let (content, calls) = parse_answer(plan.parser(), &text);
        assert_eq!(content, None, "seed {seed}: {text:?}");
        assert_eq!(calls.len(), 1, "seed {seed}: {text:?}");
        assert_eq!(calls[0].function.name, "set_mode");
        let arguments: serde_json::Value =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert!(
            matches!(arguments["mode"].as_str(), Some("eco" | "boost")),
            "seed {seed}: {text:?}"
        );
        assert!(
            arguments["level"]
                .as_i64()
                .is_some_and(|level| (0..=9).contains(&level)),
            "seed {seed}: {text:?}"
        );
        assert_eq!(arguments.as_object().unwrap().len(), 2, "seed {seed}");
    }
}

/// `response_format: json_object` on random weights: the reply is one JSON
/// object and nothing else, with no thinking before it.
#[test]
fn a_json_reply_is_one_object() {
    let model = tiny_model();
    let messages = [ChatMessage {
        role: ChatRole::User,
        content: "Two colours, as JSON.".into(),
    }];
    let mut complete = 0;
    for seed in 1..=6u64 {
        let config = ChatGenerationConfig {
            temperature: 1.0,
            top_p: 1.0,
            seed,
            constrain_json_object: true,
            ..ChatGenerationConfig::default()
        };
        let prompt = model.prompt_token_ids(&messages, &config).unwrap();
        let mut state = model
            .start_decode_managed(&prompt, 400, &config, roomy_cache(&model))
            .unwrap();
        let text = reply_text(&model, &mut state, 400);
        assert!(text.starts_with('{'), "seed {seed}: {text:?}");
        match serde_json::from_str::<serde_json::Value>(&text) {
            Ok(value) => {
                assert!(value.is_object(), "seed {seed}: {text:?}");
                complete += 1;
            }
            // Random weights can write an object longer than the budget;
            // what there is of it is still JSON, cut short.
            Err(error) => {
                assert!(error.is_eof(), "seed {seed}: {text:?}: {error}");
                assert_eq!(state.generated_ids_for_test().len(), 400);
            }
        }
    }
    assert!(complete >= 2, "{complete} of 6 replies closed their object");
}

/// A prompt that attaches another's committed prefix (its pages, and the
/// convolution ring's snapshot at that cursor) prefills and decodes exactly
/// as a fresh run of the whole prompt does.
#[test]
fn an_attached_prefix_continues_exactly_as_a_fresh_run() {
    let model = tiny_model().without_turn_end_for_test();
    let rows = caches(&model, 3);
    attached_prefix_matches_a_fresh_run(model, rows);
}

/// The same on the Metal device with F16 keys and values, as served.
#[cfg(feature = "metal")]
#[test]
fn an_attached_prefix_continues_exactly_as_a_fresh_run_on_metal() {
    use crate::backends::{BackendPreference, DeviceSelector};
    let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
        "an_attached_prefix_continues_exactly_as_a_fresh_run_on_metal",
    ) else {
        return;
    };
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    if !matches!(profile.device, Device::Metal(_)) {
        eprintln!("skipping: no Metal device");
        return;
    }
    let model = tiny_model_on(profile).without_turn_end_for_test();
    let rows = caches_on(&model, &device, DType::F16, 3);
    attached_prefix_matches_a_fresh_run(model, rows);
}

fn attached_prefix_matches_a_fresh_run(
    mut model: Inkling2ChatModel,
    rows: Vec<PhysicalPagedKvCache>,
) {
    let golden = golden();
    model.set_prefix_caching(true);
    let prompt = &golden.tokens[..38];
    // Two whole pages of the test arena.
    let boundary = 32;
    let config = ChatGenerationConfig::default();
    let mut rows = rows.into_iter();
    let begin = |cache| {
        model
            .begin_resumable_prefill_managed(prompt, 8, &config, cache)
            .unwrap()
    };

    let mut fresh = begin(rows.next().unwrap());
    model
        .continue_resumable_prefill(&mut fresh, prompt, 0, boundary)
        .unwrap();
    model
        .continue_resumable_prefill(&mut fresh, prompt, boundary, prompt.len())
        .unwrap();

    let mut publisher = begin(rows.next().unwrap());
    model
        .continue_resumable_prefill(&mut publisher, prompt, 0, boundary)
        .unwrap();
    let ring = publisher.conv_for_test();
    assert_eq!(ring.cursor(), boundary as u64);

    // The publisher's two prefix pages, then the attacher's own.
    let shared = publisher.cache_view_for_test();
    let own = rows.next().unwrap();
    let blocks = shared.blocks[..2]
        .iter()
        .chain(&own.blocks[2..])
        .copied()
        .collect::<Vec<_>>();
    let attach = |boundary| {
        PhysicalPagedKvCache::new(
            own.arena().clone(),
            (0..own.arena().config().layers.len() as u32)
                .map(|layer| crate::kv::KvLayerBinding {
                    model_layer: layer,
                    physical_layer: layer,
                })
                .collect(),
            blocks.clone(),
            boundary,
        )
        .unwrap()
    };
    let mut attached = begin(attach(boundary));
    assert_eq!(attached.prefill_progress(), boundary);
    attached.restore_conv_for_test(ring);
    model
        .continue_resumable_prefill(&mut attached, prompt, boundary, prompt.len())
        .unwrap();
    assert_eq!(
        values(state_logits(&attached)),
        values(state_logits(&fresh))
    );
    assert_eq!(
        generate(&model, &mut attached, 8),
        generate(&model, &mut fresh, 8)
    );

    // The whole prompt as a prefix leaves nothing to prefill; and without
    // prefix caching a cache that starts above zero is refused rather than
    // continued from an empty ring.
    assert!(model
        .begin_resumable_prefill_managed(&prompt[..boundary], 8, &config, attach(boundary))
        .is_err());
    model.set_prefix_caching(false);
    assert!(model
        .begin_resumable_prefill_managed(prompt, 8, &config, attach(boundary))
        .is_err());
}

/// The history boundary is where the prompt's last message starts, so a
/// forced call or JSON opening is never part of what the next turn attaches.
#[test]
fn the_history_boundary_is_the_start_of_the_last_message() {
    let model = tiny_model();
    let user = |text: &str| ChatMessage {
        role: ChatRole::User,
        content: text.into(),
    };
    let config = ChatGenerationConfig::default();
    let first = model.prompt_token_ids(&[user("Hello")], &config).unwrap();
    let boundary = model.history_boundary(&first).unwrap();
    // The effort line, then the user message: the boundary is between them.
    assert!(boundary > 0 && boundary < first.len());
    // The next turn's prompt repeats everything up to it.
    let assistant = ChatMessage {
        role: ChatRole::Assistant,
        content: "Hi.".into(),
    };
    let second = model
        .prompt_token_ids(&[user("Hello"), assistant, user("Again")], &config)
        .unwrap();
    assert_eq!(second[..first.len()], first[..]);
    assert!(model.history_boundary(&second).unwrap() > first.len());
    // A JSON reply opens a model message; the boundary is before it.
    let json = ChatGenerationConfig {
        constrain_json_object: true,
        ..ChatGenerationConfig::default()
    };
    let opened = model.prompt_token_ids(&[user("Hello")], &json).unwrap();
    assert_eq!(model.history_boundary(&opened), Some(opened.len() - 2));
}

/// The two caches of the split layout: the full-attention applications' and
/// the sliding-window ones', each in an arena of its own.
fn split_caches(model: &Inkling2ChatModel) -> (PhysicalPagedKvCache, PhysicalPagedKvCache) {
    let config = model.config();
    let (windowed, full): (Vec<u32>, Vec<u32>) =
        (0..config.application_count() as u32).partition(|application| {
            config
                .heads(config.applications()[*application as usize].layer)
                .window
                .is_some()
        });
    let cache = |layers: &[u32]| {
        test_physical_caches_for_model_layers_on(
            &Device::Cpu,
            DType::F32,
            layers,
            config.full.kv_heads,
            config.full.head_dim,
            96,
            1,
        )
        .remove(0)
    };
    (cache(&full), cache(&windowed))
}

/// With the sliding-window applications in a cache of their own, the model
/// computes what it does with one cache; and it goes on doing so on a view of
/// that cache that has let go of the pages no window reaches, as the engine
/// hands it over once they are released.
#[test]
fn the_windowed_cache_can_release_pages_no_query_reaches() {
    use super::state::Inkling2Caches;
    let golden = golden();
    let model = tiny_model().without_turn_end_for_test();
    let window = model.config().sliding.window.unwrap();
    let prompt = &golden.tokens[..40];
    let config = ChatGenerationConfig::default();
    let steps = 12;
    let mut whole = model
        .start_decode_managed(prompt, steps, &config, caches(&model, 1).remove(0))
        .unwrap();
    let want_logits = values(state_logits(&whole));
    let want = generate(&model, &mut whole, steps);

    let (full, windowed) = split_caches(&model);
    let mut state = model
        .start_decode_managed(
            prompt,
            steps,
            &config,
            Inkling2Caches::new(full, Some(windowed)).unwrap(),
        )
        .unwrap();
    assert_eq!(values(state_logits(&state)), want_logits);
    // Two steps in, hand the state the next quantum's reservation: the main
    // cache as it is, and the windowed one without the pages before the
    // oldest token the next query can see.
    assert_eq!(generate(&model, &mut state, 2), want[..2]);
    let main = state.cache_view_for_test();
    let held = state.windowed_cache_for_test().unwrap();
    let context = held.context_len();
    let page = held.arena().config().page_tokens as usize;
    let window_start = context + 1 - window;
    let first_page = window_start / page;
    assert!(first_page >= 2, "the test must drop pages");
    let view = PhysicalPagedKvCache::new_windowed(
        held.arena().clone(),
        held.arena()
            .config()
            .layers
            .iter()
            .map(|layer| layer.binding)
            .collect(),
        held.blocks[first_page..].to_vec(),
        window_start,
        context,
    )
    .unwrap();
    state.begin_managed_quantum(main, Some(view)).unwrap();
    assert_eq!(generate(&model, &mut state, steps), want);
    // A reservation in the other layout is not this session's.
    let other = caches(&model, 1).remove(0);
    assert!(state.begin_managed_quantum(other, None).is_err());
}

/// The fused attention kernel inside the served model gives the composed
/// form's logits and tokens: on one cache with prefill cut into spans, and on
/// the split layout after the windowed cache has released pages.
#[test]
fn fused_attention_serves_what_composed_attention_does() {
    use super::state::Inkling2Caches;
    let golden = golden();
    let composed = tiny_model().without_turn_end_for_test();
    // The composed form is the CPU's default.
    assert!(composed.fused_attention().is_err());
    let fused = tiny_model()
        .without_turn_end_for_test()
        .with_fused_attention_for_test();
    let prompt = &golden.tokens[..40];
    let config = ChatGenerationConfig::default();
    let steps = 12;
    let mut reference = composed
        .start_decode_managed(prompt, steps, &config, caches(&composed, 1).remove(0))
        .unwrap();
    let want_logits = values(state_logits(&reference));
    let want = generate(&composed, &mut reference, steps);

    // Prefill in spans that end inside and beyond the window.
    let mut state = fused
        .begin_resumable_prefill_managed(prompt, steps, &config, caches(&fused, 1).remove(0))
        .unwrap();
    let mut at = 0;
    for end in [3, 4, 21, prompt.len()] {
        fused
            .continue_resumable_prefill(&mut state, prompt, at, end)
            .unwrap();
        at = end;
    }
    let error = max_error(&values(state_logits(&state)), &want_logits);
    assert!(error <= 2e-4, "fused prefill logits differ by {error}");
    assert_eq!(generate(&fused, &mut state, steps), want);

    // The split layout, then a windowed view without its early pages.
    let (full, windowed) = split_caches(&fused);
    let mut state = fused
        .start_decode_managed(
            prompt,
            steps,
            &config,
            Inkling2Caches::new(full, Some(windowed)).unwrap(),
        )
        .unwrap();
    assert_eq!(generate(&fused, &mut state, 2), want[..2]);
    let main = state.cache_view_for_test();
    let held = state.windowed_cache_for_test().unwrap();
    let window = fused.config().sliding.window.unwrap();
    let page = held.arena().config().page_tokens as usize;
    let context = held.context_len();
    let window_start = context + 1 - window;
    let view = PhysicalPagedKvCache::new_windowed(
        held.arena().clone(),
        held.arena()
            .config()
            .layers
            .iter()
            .map(|layer| layer.binding)
            .collect(),
        held.blocks[window_start / page..].to_vec(),
        window_start,
        context,
    )
    .unwrap();
    state.begin_managed_quantum(main, Some(view)).unwrap();
    assert_eq!(generate(&fused, &mut state, steps), want);
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

/// On Metal. With keys and values stored in F32 the served path tracks the
/// reference and greedy decode picks the CPU model's tokens. With F16 keys
/// and values, as served, the logits stay close; the random fixture routes
/// experts on near-ties, so its greedy tokens are not compared.
#[cfg(feature = "metal")]
#[test]
fn metal_prefill_and_decode_track_the_reference() {
    use crate::backends::{BackendPreference, DeviceSelector};
    let Some((device, _lease)) =
        crate::backends::metal_test_device_or_skip("metal_prefill_and_decode_track_the_reference")
    else {
        return;
    };
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    if !matches!(profile.device, Device::Metal(_)) {
        eprintln!("skipping: no Metal device");
        return;
    }
    let golden = golden();
    let model = tiny_model_on(profile).without_turn_end_for_test();
    // Attention runs in the fused kernel on Metal once it passes its check.
    if crate::env::value(super::chat::FUSED_ATTENTION_ENV).is_none() {
        assert!(
            model.fused_attention().is_ok(),
            "{:?}",
            model.fused_attention()
        );
    }
    let vocab = model.config().unpadded_vocab;
    let config = ChatGenerationConfig::default();
    let prompt = &golden.tokens[..golden.prompt_tokens];
    let row = prompt.len() - 1;
    let want_logits = &golden.logits[row * vocab..(row + 1) * vocab];
    for (dtype, tolerance) in [(DType::F32, 5e-2f32), (DType::F16, 0.3)] {
        let mut state = model
            .begin_resumable_prefill_managed(
                prompt,
                8,
                &config,
                caches_on(&model, &device, dtype, 1).remove(0),
            )
            .unwrap();
        model
            .continue_resumable_prefill(&mut state, prompt, 0, 9)
            .unwrap();
        model
            .continue_resumable_prefill(&mut state, prompt, 9, prompt.len())
            .unwrap();
        let error = max_error(&values(state_logits(&state)), want_logits);
        eprintln!("Metal {dtype:?}-key prefill logit error {error}");
        assert!(error <= tolerance, "{dtype:?} keys: logit error {error}");
        let got = generate(&model, &mut state, 8);
        assert_eq!(got.len(), 8);
        if dtype == DType::F32 {
            assert_eq!(got, reference_greedy(prompt, 8));
        }
    }
}

/// The device expert path (the kernels' CPU reference here) inside the
/// served model: it passes the load-time check, and the model then tracks
/// the reference logits and picks the host path's tokens.
#[test]
fn device_experts_serve_the_same_reply() {
    let golden = golden();
    let host = tiny_model().without_turn_end_for_test();
    assert!(!host.expert_placement().on_device);
    let device = tiny_model_with(DeviceProfile::cpu(), ExpertBackendRequest::Device)
        .without_turn_end_for_test();
    let placement = device.expert_placement();
    assert!(placement.on_device, "{}", placement.reason);
    assert!(placement.check_error.unwrap() <= 1e-4);
    assert_eq!(device.runtime_diagnostics()["experts"]["backend"], "device");

    let vocab = device.config().unpadded_vocab;
    let config = ChatGenerationConfig::default();
    let prompt = &golden.tokens[..golden.prompt_tokens];
    let mut state = device
        .start_decode_managed(prompt, 12, &config, caches(&device, 1).remove(0))
        .unwrap();
    let row = prompt.len() - 1;
    let error = max_error(
        &values(state_logits(&state)),
        &golden.logits[row * vocab..(row + 1) * vocab],
    );
    assert!(error <= 2e-3, "logit error {error}");
    let got = generate(&device, &mut state, 12);
    let mut state = host
        .start_decode_managed(prompt, 12, &config, caches(&host, 1).remove(0))
        .unwrap();
    assert_eq!(got, generate(&host, &mut state, 12));
}

/// The experts on the Metal device, inside the served model: the load-time
/// check passes and prefill tracks the reference logits as closely as the
/// host expert path does there.
#[cfg(feature = "metal")]
#[test]
fn metal_device_experts_pass_the_load_check_and_track_the_reference() {
    use crate::backends::{BackendPreference, DeviceSelector};
    let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
        "metal_device_experts_pass_the_load_check_and_track_the_reference",
    ) else {
        return;
    };
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    if !matches!(profile.device, Device::Metal(_)) {
        eprintln!("skipping: no Metal device");
        return;
    }
    // The default on Metal is the host path.
    assert!(!tiny_model_on(profile.clone()).expert_placement().on_device);
    let model = tiny_model_with(profile, ExpertBackendRequest::Device).without_turn_end_for_test();
    let placement = model.expert_placement();
    assert!(placement.on_device, "{}", placement.reason);
    assert!(placement.check_error.unwrap() <= 1e-4);

    let golden = golden();
    let vocab = model.config().unpadded_vocab;
    let config = ChatGenerationConfig::default();
    let prompt = &golden.tokens[..golden.prompt_tokens];
    let mut state = model
        .start_decode_managed(
            prompt,
            8,
            &config,
            caches_on(&model, &device, DType::F32, 1).remove(0),
        )
        .unwrap();
    let row = prompt.len() - 1;
    let error = max_error(
        &values(state_logits(&state)),
        &golden.logits[row * vocab..(row + 1) * vocab],
    );
    assert!(error <= 5e-2, "logit error {error}");
    assert_eq!(generate(&model, &mut state, 8).len(), 8);
}

#[test]
fn the_expert_backend_request_parses() {
    use crate::backends::BackendKind;
    let parse = ExpertBackendRequest::parse;
    assert_eq!(parse(None), ExpertBackendRequest::Auto);
    assert_eq!(parse(Some("auto")), ExpertBackendRequest::Auto);
    assert_eq!(parse(Some(" Host ")), ExpertBackendRequest::Host);
    assert_eq!(parse(Some("DEVICE")), ExpertBackendRequest::Device);
    assert_eq!(parse(Some("something else")), ExpertBackendRequest::Auto);
    assert!(ExpertBackendRequest::Auto.wants_device(BackendKind::Cuda));
    assert!(!ExpertBackendRequest::Auto.wants_device(BackendKind::Metal));
    assert!(!ExpertBackendRequest::Auto.wants_device(BackendKind::Cpu));
    assert!(!ExpertBackendRequest::Host.wants_device(BackendKind::Cuda));
    assert!(ExpertBackendRequest::Device.wants_device(BackendKind::Cpu));
}
