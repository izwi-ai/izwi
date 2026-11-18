//! Model tests on the tiny synthetic checkpoint (`synthetic.rs`): the
//! loader, the prompt contract, the decode loop and its streaming.

use candle_core::DType;

use crate::backends::DeviceProfile;
use crate::catalog::ModelVariant;
use crate::models::shared::attention::physical::{test_physical_caches_on, PhysicalPagedKvCache};

use super::model::RumbleTextStream;
use super::prompt::{build_rumble_prompt, RumblePromptOptions, RumbleTranscriptStyle};
use super::synthetic::{synthetic_token_id, SyntheticRumbleCheckpoint, SYNTHETIC_VOCAB};
use super::transcript::split_rumble_output;
use super::{RumbleAsrModel, RumbleGenerationOptions, RumbleRequestOptions};

fn load(checkpoint: &SyntheticRumbleCheckpoint) -> RumbleAsrModel {
    RumbleAsrModel::load(
        checkpoint.path(),
        ModelVariant::Rumble2Flash,
        DeviceProfile::cpu(),
    )
    .expect("synthetic Rumble loads")
}

fn cache(model: &RumbleAsrModel) -> PhysicalPagedKvCache {
    let text = &model.config().text_config;
    test_physical_caches_on(
        &model.device().device,
        text.num_hidden_layers,
        text.num_key_value_heads,
        model.config().decoder_head_dim(),
        model.sequence_tokens(),
        1,
    )
    .pop()
    .expect("one cache")
}

fn tone(samples: usize) -> Vec<f32> {
    (0..samples)
        .map(|index| ((index as f32 * 0.031).sin() + (index as f32 * 0.0071).cos()) * 0.2)
        .collect()
}

fn options(max_new_tokens: usize) -> RumbleGenerationOptions {
    RumbleGenerationOptions {
        max_new_tokens,
        ..RumbleGenerationOptions::default()
    }
}

#[test]
fn the_native_layout_loads_and_reports_itself() {
    let checkpoint = SyntheticRumbleCheckpoint::write();
    let model = load(&checkpoint);
    assert_eq!(model.text_dtype(), DType::F32);
    assert_eq!(model.audio_dtype(), DType::F32);
    assert_eq!(model.sequence_tokens(), 512);
    assert_eq!(model.max_audio_seconds_hint(), Some(30.0));
    assert_eq!(model.audio_token_rate(), 13.0);
    let summary = model.diagnostics_summary();
    assert_eq!(summary["family"], "rumble_asr");
    assert_eq!(summary["dtype"], "F32");
    assert_eq!(summary["vocab_size"], SYNTHETIC_VOCAB);
    assert_eq!(summary["decoder_layers"], 2);
    assert_eq!(summary["tied_head"], true);
    let specials = model.tokenizer().specials();
    assert_eq!(specials.asr_text, synthetic_token_id("<asr_text>"));
    assert_eq!(
        specials.stop_ids(),
        vec![
            synthetic_token_id("<|im_end|>"),
            synthetic_token_id("<|endoftext|>")
        ]
    );
}

/// The census check runs before any weight is mapped, and names the tensor.
#[test]
fn a_checkpoint_in_another_layout_is_refused_by_name() {
    let renamed = SyntheticRumbleCheckpoint::write_with(|tensors| {
        let weight = tensors
            .remove("model.multi_modal_projector.linear_1.weight")
            .unwrap();
        tensors.insert("model.audio_tower.proj1.weight".into(), weight);
    });
    let error = match RumbleAsrModel::load(
        renamed.path(),
        ModelVariant::Rumble2Flash,
        DeviceProfile::cpu(),
    ) {
        Ok(_) => panic!("a re-laid-out checkpoint loaded"),
        Err(error) => error.to_string(),
    };
    assert!(
        error.contains("multi_modal_projector.linear_1.weight"),
        "{error}"
    );

    let extra = SyntheticRumbleCheckpoint::write_with(|tensors| {
        let table = tensors["model.language_model.embed_tokens.weight"].clone();
        tensors.insert("lm_head.weight".into(), table);
    });
    let error = match RumbleAsrModel::load(
        extra.path(),
        ModelVariant::Rumble2Flash,
        DeviceProfile::cpu(),
    ) {
        Ok(_) => panic!("a checkpoint with an unread head loaded"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("lm_head.weight"), "{error}");
}

#[test]
fn the_prompt_follows_the_published_template() {
    let checkpoint = SyntheticRumbleCheckpoint::write();
    let model = load(&checkpoint);
    let tokenizer = model.tokenizer();
    let id = synthetic_token_id;
    let text = |text: &str| tokenizer.encode(text).unwrap();

    let prompt = build_rumble_prompt(tokenizer, 5, &RumblePromptOptions::default()).unwrap();
    let mut expected = vec![id("<|im_start|>")];
    expected.extend(text("system\n"));
    expected.extend([id("<digits>"), id("<nofiller>"), id("<verbatim>")]);
    expected.push(id("<|im_end|>"));
    expected.extend(text("\n"));
    expected.push(id("<|im_start|>"));
    expected.extend(text("user\n"));
    expected.push(id("<|audio_start|>"));
    let audio_start = expected.len();
    expected.extend([id("<|audio_pad|>"); 5]);
    expected.push(id("<|audio_end|>"));
    expected.push(id("<|im_end|>"));
    expected.extend(text("\n"));
    expected.push(id("<|im_start|>"));
    expected.extend(text("assistant\n"));
    assert_eq!(prompt.ids, expected);
    assert_eq!(prompt.audio_start, audio_start);
    assert_eq!(prompt.audio_tokens, 5);
    assert_eq!(prompt.forced_language, None);
    // The plain-text pieces are byte-level: one token per character.
    assert_eq!(text("system\n").len(), 7);

    // A forced language prefills the assistant turn; context text follows
    // the style tokens; each style flag flips one token.
    let prompt = build_rumble_prompt(
        tokenizer,
        3,
        &RumblePromptOptions {
            language: Some("nl".into()),
            context: Some("  Izwi  ".into()),
            style: RumbleTranscriptStyle {
                digits: false,
                fillers: true,
                verbatim: false,
            },
        },
    )
    .unwrap();
    assert_eq!(prompt.forced_language.as_deref(), Some("Dutch"));
    assert_eq!(prompt.context_tokens, 4);
    let style_at = 1 + text("system\n").len();
    assert_eq!(
        prompt.ids[style_at..style_at + 3],
        [id("<nodigits>"), id("<filler>"), id("<noverbatim>")]
    );
    assert_eq!(prompt.ids[style_at + 3..style_at + 7], text("Izwi")[..]);
    let mut tail = text("language Dutch");
    tail.push(id("<asr_text>"));
    assert!(prompt.ids.ends_with(&tail));

    assert!(build_rumble_prompt(tokenizer, 0, &RumblePromptOptions::default()).is_err());
    let long = "word ".repeat(200);
    let error = build_rumble_prompt(
        tokenizer,
        3,
        &RumblePromptOptions {
            context: Some(long),
            ..RumblePromptOptions::default()
        },
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("context prompt"), "{error}");
}

#[test]
fn output_splits_on_the_marker_token_not_on_text() {
    let checkpoint = SyntheticRumbleCheckpoint::write();
    let model = load(&checkpoint);
    let tokenizer = model.tokenizer();
    let asr_text = synthetic_token_id("<asr_text>");
    let non_speech = synthetic_token_id("<non_speech>");

    let mut ids = tokenizer.encode("language English").unwrap();
    ids.push(asr_text);
    ids.extend(tokenizer.encode("language is a tool").unwrap());
    let output = split_rumble_output(tokenizer, &ids, None).unwrap();
    assert_eq!(output.language.as_deref(), Some("English"));
    assert_eq!(output.text, "language is a tool");

    // Forced: everything generated is transcript, even text that reads like
    // a header.
    let ids = tokenizer.encode("language English").unwrap();
    let output = split_rumble_output(tokenizer, &ids, Some("Dutch")).unwrap();
    assert_eq!(output.language.as_deref(), Some("Dutch"));
    assert_eq!(output.text, "language English");

    // Audio without speech: no language, and the marker is not text.
    let mut ids = tokenizer.encode("language None").unwrap();
    ids.push(asr_text);
    ids.push(non_speech);
    let output = split_rumble_output(tokenizer, &ids, None).unwrap();
    assert_eq!(output.language, None);
    assert_eq!(output.text, "");

    // A header cut off before the marker is not returned as a transcript.
    let ids = tokenizer.encode("language English").unwrap();
    let output = split_rumble_output(tokenizer, &ids, None).unwrap();
    assert_eq!(output.language.as_deref(), Some("English"));
    assert_eq!(output.text, "");
}

#[test]
fn the_stream_emits_only_transcript_text() {
    let checkpoint = SyntheticRumbleCheckpoint::write();
    let model = load(&checkpoint);
    let tokenizer = model.tokenizer();
    let mut ids = tokenizer.encode("language English").unwrap();
    ids.push(synthetic_token_id("<asr_text>"));
    ids.extend(tokenizer.encode("hello ").unwrap());
    ids.push(synthetic_token_id("<non_speech>"));
    ids.extend(tokenizer.encode("there").unwrap());

    let mut stream = RumbleTextStream::new(tokenizer, false);
    let mut streamed = String::new();
    for id in &ids {
        streamed.push_str(&stream.push(*id).unwrap());
    }
    streamed.push_str(&stream.finish().unwrap());
    assert_eq!(streamed, "hello there");
    assert_eq!(stream.text(), "hello there");

    // With a forced language the header is in the prompt, so text streams
    // from the first id.
    let mut stream = RumbleTextStream::new(tokenizer, true);
    let mut streamed = String::new();
    for id in tokenizer.encode("hi").unwrap() {
        streamed.push_str(&stream.push(id).unwrap());
    }
    streamed.push_str(&stream.finish().unwrap());
    assert_eq!(streamed, "hi");
}

#[test]
fn an_invocation_decodes_deterministically_within_its_budget() {
    let checkpoint = SyntheticRumbleCheckpoint::write();
    let model = load(&checkpoint);
    let audio = tone(24_000);
    let request = RumbleRequestOptions {
        language: Some("English".into()),
        ..RumbleRequestOptions::default()
    };
    let run = |max_new_tokens: usize| {
        let mut cache = cache(&model);
        let mut streamed = String::new();
        let mut on_delta = |delta: &str| streamed.push_str(delta);
        let transcription = model
            .transcribe_physical(
                &audio,
                16_000,
                &request,
                options(max_new_tokens),
                &mut cache,
                Some(&mut on_delta),
            )
            .unwrap();
        (transcription, streamed, cache.context_len())
    };
    let (first, streamed, context) = run(12);
    let diagnostics = first.diagnostics.clone().unwrap();
    // 1.5 s: 150 mel frames, 13 tokens for the full chunk and 7 for the half.
    assert_eq!(diagnostics["mel_frames"], 150);
    assert_eq!(diagnostics["audio_tokens"], 20);
    assert_eq!(diagnostics["forced_language"], "English");
    let prompt_tokens = diagnostics["prompt_tokens"].as_u64().unwrap() as usize;
    let generated = diagnostics["generated_tokens"].as_u64().unwrap() as usize;
    assert!(generated <= 12, "{generated}");
    assert!(generated >= 1);
    // The cache holds the prompt and every generated token that was fed
    // back: all of them but the last when the budget ended the decode.
    let fed_back = if diagnostics["stop_reason"] == "max_tokens" {
        generated - 1
    } else {
        generated
    };
    assert_eq!(context, prompt_tokens + fed_back);
    assert_eq!(first.language.as_deref(), Some("English"));
    // Streamed deltas are the transcript (the final text is trimmed).
    assert_eq!(streamed.trim(), first.text);

    let (second, _, _) = run(12);
    assert_eq!(second.text, first.text);
    assert_eq!(
        second.diagnostics.unwrap()["generated_tokens"],
        diagnostics["generated_tokens"]
    );

    // A smaller budget is a prefix of the same greedy decode.
    let (short, _, _) = run(3);
    let short_generated = short.diagnostics.unwrap()["generated_tokens"]
        .as_u64()
        .unwrap() as usize;
    assert_eq!(short_generated, generated.min(3));
}

#[test]
fn the_budget_is_narrowed_to_the_invocation_context_and_misuse_is_refused() {
    let checkpoint = SyntheticRumbleCheckpoint::write();
    let model = load(&checkpoint);
    let audio = tone(16_000);
    let request = RumbleRequestOptions::default();

    // A budget far past the 512-token context is narrowed, not an error.
    let mut cache_a = cache(&model);
    let transcription = model
        .transcribe_physical(
            &audio,
            16_000,
            &request,
            options(100_000),
            &mut cache_a,
            None,
        )
        .unwrap();
    let diagnostics = transcription.diagnostics.unwrap();
    let prompt_tokens = diagnostics["prompt_tokens"].as_u64().unwrap();
    assert_eq!(
        diagnostics["max_new_tokens"].as_u64().unwrap(),
        512 - prompt_tokens
    );

    // A used cache is refused: an invocation starts at context zero.
    let error = model
        .transcribe_physical(&audio, 16_000, &request, options(4), &mut cache_a, None)
        .unwrap_err()
        .to_string();
    assert!(error.contains("context zero"), "{error}");

    // Audio past the 30 s window is refused before the tower runs.
    let mut cache_b = cache(&model);
    let error = model
        .transcribe_physical(
            &tone(16_000 * 31),
            16_000,
            &request,
            options(4),
            &mut cache_b,
            None,
        )
        .unwrap_err()
        .to_string();
    assert!(error.contains("up to 30s"), "{error}");
    assert_eq!(cache_b.context_len(), 0);
}

/// Different audio reaches the decoder: the transcript's tokens depend on
/// it (a tower whose output never reached the prompt would decode alike).
#[test]
fn the_audio_conditions_the_decode() {
    let checkpoint = SyntheticRumbleCheckpoint::write();
    let model = load(&checkpoint);
    let features_a = model.prepare_features(&tone(16_000), 16_000).unwrap();
    let silence = vec![0.0_f32; 16_000];
    let features_b = model.prepare_features(&silence, 16_000).unwrap();
    let a = model.audio_embeddings(&features_a).unwrap().embeddings;
    let b = model.audio_embeddings(&features_b).unwrap().embeddings;
    assert_eq!(a.dims(), &[1, 13, 16]);
    let difference = (a - b)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(difference > 1e-3, "{difference}");

    let prompt = model
        .build_prompt(13, &RumblePromptOptions::default())
        .unwrap();
    let embeddings = model
        .prompt_embeddings(
            &prompt,
            &model.audio_embeddings(&features_a).unwrap().embeddings,
        )
        .unwrap();
    assert_eq!(embeddings.dims(), &[1, prompt.ids.len(), 16]);
    // The placeholder rows are the audio rows, not the pad token's.
    let spliced = embeddings.narrow(1, prompt.audio_start, 13).unwrap();
    let audio = model.audio_embeddings(&features_a).unwrap().embeddings;
    let same = (spliced - audio)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert_eq!(same, 0.0);
}
