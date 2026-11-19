//! Real-checkpoint tests for Rumble-2-Flash.
//!
//! All are ignored: they need the published checkpoint, read from
//! `IZWI_RUMBLE_MODEL_DIR` or else the default models directory.
//!
//! * `the_published_tokenizer_and_prompt_match_the_checkpoint` needs only
//!   the checkpoint: the special ids against the published added-token
//!   table, and the default prompt against those ids.
//! * `rumble_transcribes_the_sample_clip_*` needs only the checkpoint and
//!   the repository's `data/fox.wav`: a loose end-to-end check.
//! * `the_retained_route_matches_the_atomic_route_*` needs the same: on the
//!   sample clip, with and without a forced language and style, a retained
//!   row (prefilled in spans, decoded alone or in a batch with the others)
//!   must emit the atomic route's text, language and token count.
//! * `rumble_matches_the_f32_reference_*` compares against transformers
//!   (`scripts/fixtures/generate_rumble_reference.py`). It also needs the
//!   fixture at `tests/fixtures/rumble/reference.json` and the 16 kHz audio
//!   the script wrote, in `IZWI_RUMBLE_AUDIO_DIR`. Per case it checks that
//!   izwi's own request options build the reference prompt ids, that the
//!   log-mel features and projected audio embeddings match at the fixture's
//!   sample points, and that the greedy tokens equal the reference's up to
//!   the first step where the reference itself is a near tie.
//!
//! Run: `IZWI_RUMBLE_AUDIO_DIR=… cargo test -p izwi-core --release
//! --features metal --lib rumble::reference_tests -- --ignored --nocapture`

use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::backends::DeviceProfile;
use crate::catalog::ModelVariant;
use crate::engine::AsrTranscriptStyle;
use crate::models::shared::attention::physical::{
    test_physical_caches_with_dtype_on, PhysicalPagedKvCache,
};
use crate::runtime::audio_io::decode_audio_bytes;

use super::prompt::{RumblePromptOptions, RumbleTranscriptStyle};
use super::{RumbleAsrModel, RumbleGenerationOptions, RumbleRequestOptions};

const MAX_NEW_TOKENS: usize = 256;

struct Tolerance {
    /// Reference steps whose top-2 margin is below this are near ties,
    /// where the argmax may differ and the transcript is izwi's own after.
    near_tie: f32,
    /// Max |izwi - reference| at the sample points, as a fraction of the
    /// reference RMS.
    features: f32,
    projected: f32,
}

#[derive(serde::Deserialize)]
struct Fixture {
    audio_sha256: BTreeMap<String, String>,
    cases: BTreeMap<String, FixtureCase>,
}

#[derive(serde::Deserialize)]
struct FixtureCase {
    audio: String,
    language: Option<String>,
    prompt: Option<String>,
    digits: Option<bool>,
    filler: Option<bool>,
    verbatim: Option<bool>,
    prompt_ids: Vec<u32>,
    mel_frames: usize,
    audio_tokens: usize,
    features: FixtureSample,
    projected: FixtureSample,
    generated_ids: Vec<u32>,
    steps: Vec<FixtureStep>,
    text: String,
}

#[derive(serde::Deserialize)]
struct FixtureSample {
    shape: Vec<usize>,
    indices: Vec<usize>,
    values: Vec<f32>,
    rms: f32,
}

#[derive(serde::Deserialize)]
struct FixtureStep {
    token: u32,
    margin: f32,
}

fn model_dir() -> PathBuf {
    std::env::var_os("IZWI_RUMBLE_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("izwi")
                .join("models")
                .join(ModelVariant::Rumble2Flash.dir_name())
        })
}

fn load(profile: DeviceProfile) -> RumbleAsrModel {
    RumbleAsrModel::load(&model_dir(), ModelVariant::Rumble2Flash, profile)
        .expect("the published Rumble-2-Flash checkpoint loads")
}

fn cache(model: &RumbleAsrModel) -> PhysicalPagedKvCache {
    caches(model, 1).pop().expect("one cache")
}

/// `rows` caches on one pool, as rows that decode in one batch hold.
fn caches(model: &RumbleAsrModel, rows: usize) -> Vec<PhysicalPagedKvCache> {
    let text = &model.config().text_config;
    test_physical_caches_with_dtype_on(
        &model.device().device,
        model.text_dtype(),
        text.num_hidden_layers,
        text.num_key_value_heads,
        model.config().decoder_head_dim(),
        model.sequence_tokens(),
        rows,
    )
}

fn sampled_error(actual: &[f32], sample: &FixtureSample, label: &str) -> f32 {
    let mut worst = 0.0_f32;
    for (index, expected) in sample.indices.iter().zip(&sample.values) {
        let value = *actual
            .get(*index)
            .unwrap_or_else(|| panic!("{label}: sample index {index} is out of range"));
        let error = (value - expected).abs();
        // A NaN must fail, not read as zero error.
        worst = if error.is_finite() {
            worst.max(error)
        } else {
            f32::INFINITY
        };
    }
    worst / sample.rms.max(f32::MIN_POSITIVE)
}

/// The special ids and the default prompt, against the published tables.
#[test]
#[ignore = "needs the Rumble-2-Flash checkpoint"]
fn the_published_tokenizer_and_prompt_match_the_checkpoint() {
    let model = load(DeviceProfile::cpu());
    let tokenizer = model.tokenizer();
    let specials = tokenizer.specials();
    let published: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/rumble/added_tokens.json"
    ))
    .unwrap();
    let id = |content: &str| -> u32 {
        published["added_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .find(|token| token["content"] == content)
            .unwrap_or_else(|| panic!("{content} is not a published added token"))["id"]
            .as_u64()
            .unwrap() as u32
    };
    assert_eq!(specials.im_start, id("<|im_start|>"));
    assert_eq!(specials.im_end, id("<|im_end|>"));
    assert_eq!(specials.endoftext, id("<|endoftext|>"));
    assert_eq!(specials.audio_start, id("<|audio_start|>"));
    assert_eq!(specials.audio_end, id("<|audio_end|>"));
    assert_eq!(specials.audio_pad, id("<|audio_pad|>"));
    assert_eq!(specials.asr_text, id("<asr_text>"));
    assert_eq!(specials.non_speech, Some(id("<non_speech>")));
    assert_eq!(specials.digits, id("<digits>"));
    assert_eq!(specials.no_digits, id("<nodigits>"));
    assert_eq!(specials.filler, id("<filler>"));
    assert_eq!(specials.no_filler, id("<nofiller>"));
    assert_eq!(specials.verbatim, id("<verbatim>"));
    assert_eq!(specials.no_verbatim, id("<noverbatim>"));
    assert_eq!(tokenizer.vocab_size(), 99_630);

    // Plain text never encodes to a control token, and the header text
    // round-trips.
    for text in [
        "system\n",
        "user\n",
        "assistant\n",
        "\n",
        "language English",
    ] {
        let ids = tokenizer.encode(text).unwrap();
        assert!(!ids.is_empty(), "{text:?}");
        assert!(
            ids.iter().all(|id| !specials.is_control(*id)),
            "{text:?} encodes to a control token: {ids:?}"
        );
        assert_eq!(tokenizer.decode(&ids).unwrap(), text, "{text:?}");
    }

    let prompt = model
        .build_prompt(13, &RumblePromptOptions::default())
        .unwrap();
    let style_at = 1 + tokenizer.encode("system\n").unwrap().len();
    assert_eq!(prompt.ids[0], specials.im_start);
    assert_eq!(
        prompt.ids[style_at..style_at + 4],
        [
            specials.digits,
            specials.no_filler,
            specials.verbatim,
            specials.im_end
        ]
    );
    assert_eq!(
        prompt.ids[prompt.audio_start..prompt.audio_start + 13],
        [specials.audio_pad; 13]
    );
    assert_eq!(prompt.ids[prompt.audio_start - 1], specials.audio_start);
    assert_eq!(prompt.ids[prompt.audio_start + 13], specials.audio_end);
    let summary = model.diagnostics_summary();
    assert_eq!(summary["decoder_layers"], 14);
    assert_eq!(summary["audio_layers"], 18);
    eprintln!("{summary}");
}

fn transcribe_sample(profile: DeviceProfile) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/fox.wav");
    let (samples, rate) = decode_audio_bytes(&std::fs::read(path).unwrap()).unwrap();
    let model = load(profile);
    for language in [Some("English"), None] {
        let mut cache = cache(&model);
        let mut streamed = String::new();
        let mut on_delta = |delta: &str| streamed.push_str(delta);
        let transcription = model
            .transcribe_physical(
                &samples,
                rate,
                &RumbleRequestOptions {
                    language: language.map(str::to_string),
                    ..RumbleRequestOptions::default()
                },
                RumbleGenerationOptions::default(),
                &mut cache,
                Some(&mut on_delta),
            )
            .unwrap();
        eprintln!(
            "language {language:?} -> {:?}: {:?}",
            transcription.language, transcription.text
        );
        let lowered = transcription.text.to_lowercase();
        assert!(lowered.contains("quick brown fox"), "{lowered}");
        assert_eq!(transcription.language.as_deref(), Some("English"));
        assert_eq!(streamed.trim(), transcription.text);
        assert_eq!(
            transcription.diagnostics.unwrap()["stop_reason"],
            "stop_token"
        );
    }
}

#[test]
#[ignore = "needs the Rumble-2-Flash checkpoint"]
fn rumble_transcribes_the_sample_clip_on_cpu() {
    transcribe_sample(DeviceProfile::cpu());
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the Rumble-2-Flash checkpoint"]
fn rumble_transcribes_the_sample_clip_on_metal() {
    use crate::backends::{BackendPreference, DeviceSelector};
    let _lease =
        crate::backends::metal_test_device_or_skip("rumble_transcribes_the_sample_clip_on_metal");
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    if !matches!(profile.device, candle_core::Device::Metal(_)) {
        eprintln!("skipping: no Metal device");
        return;
    }
    transcribe_sample(profile);
}

/// Each request below runs atomically, then as a retained row decoded
/// alone, then as one row of a batch of all of them.
fn compare_retained_with_atomic(profile: DeviceProfile) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/fox.wav");
    let (samples, rate) = decode_audio_bytes(&std::fs::read(path).unwrap()).unwrap();
    let model = load(profile);
    let requests = [
        RumbleRequestOptions::default(),
        RumbleRequestOptions {
            language: Some("English".into()),
            ..RumbleRequestOptions::default()
        },
        RumbleRequestOptions {
            language: Some("English".into()),
            style: RumbleTranscriptStyle {
                digits: false,
                fillers: true,
                verbatim: false,
            },
            ..RumbleRequestOptions::default()
        },
        // A shorter clip, so the batch's rows finish at different steps.
        RumbleRequestOptions {
            prompt: Some("fox, dog".into()),
            ..RumbleRequestOptions::default()
        },
    ];
    let audio = |index: usize| -> &[f32] {
        if index == 3 {
            &samples[..samples.len() / 2]
        } else {
            &samples
        }
    };
    let begin = |index: usize, cache: PhysicalPagedKvCache| {
        let artifact = model
            .prepare_prompt_artifact(audio(index), rate, &requests[index])
            .unwrap();
        let mut state = model
            .begin_resumable_prefill_managed(artifact, RumbleGenerationOptions::default(), cache)
            .unwrap();
        // Prefill in spans, as the scheduler resumes it.
        let total = state.prefill_token_count();
        let mut start = 0;
        while start < total {
            let end = (start + 48).min(total);
            model
                .continue_resumable_prefill(&mut state, start, end)
                .unwrap();
            let _ = state.take_managed_write_completions();
            start = end;
        }
        state
    };
    let mut atomic = Vec::new();
    for (index, request) in requests.iter().enumerate() {
        let mut cache = cache(&model);
        let transcription = model
            .transcribe_physical(
                audio(index),
                rate,
                request,
                RumbleGenerationOptions::default(),
                &mut cache,
                None,
            )
            .unwrap();
        let generated = transcription.diagnostics.as_ref().unwrap()["generated_tokens"]
            .as_u64()
            .unwrap() as usize;
        eprintln!(
            "atomic {index}: {:?} {:?} ({generated} tokens)",
            transcription.language, transcription.text
        );
        assert!(!transcription.text.is_empty(), "request {index}");
        atomic.push((transcription.text, transcription.language, generated));
    }

    let mut alone_ids = Vec::new();
    for (index, (text, language, generated)) in atomic.iter().enumerate() {
        let mut state = begin(index, cache(&model));
        let decoded = loop {
            let step = model.decode_step(&mut state).unwrap();
            let _ = state.take_managed_write_completions();
            if step.finished {
                break step.text;
            }
        };
        assert_eq!(&decoded, text, "alone, request {index}");
        assert_eq!(state.language(), language.as_deref(), "request {index}");
        assert_eq!(state.generated_ids().len(), *generated, "request {index}");
        alone_ids.push(state.generated_ids().to_vec());
    }

    let mut states = caches(&model, requests.len())
        .into_iter()
        .enumerate()
        .map(|(index, cache)| begin(index, cache))
        .collect::<Vec<_>>();
    let mut texts = vec![None; states.len()];
    let mut widths = Vec::new();
    loop {
        let mut live = states
            .iter_mut()
            .enumerate()
            .filter(|(_, state)| !state.finished())
            .collect::<Vec<_>>();
        if live.is_empty() {
            break;
        }
        widths.push(live.len());
        let indices = live.iter().map(|(index, _)| *index).collect::<Vec<_>>();
        let mut rows = live
            .iter_mut()
            .map(|(_, state)| &mut **state)
            .collect::<Vec<_>>();
        let steps = model.decode_step_batch(&mut rows).unwrap();
        for (index, step) in indices.into_iter().zip(steps) {
            if step.finished {
                texts[index] = Some(step.text);
            }
        }
        for state in states.iter_mut() {
            let _ = state.take_managed_write_completions();
        }
    }
    eprintln!("batch widths: {widths:?}");
    assert_eq!(widths[0], requests.len());
    assert!(widths.iter().any(|width| *width < requests.len()));
    for (index, (text, language, _)) in atomic.iter().enumerate() {
        assert_eq!(
            texts[index].as_deref(),
            Some(text.as_str()),
            "batched {index}"
        );
        assert_eq!(
            states[index].language(),
            language.as_deref(),
            "batched {index}"
        );
        assert_eq!(
            states[index].generated_ids(),
            alone_ids[index],
            "batched {index}"
        );
    }
}

#[test]
#[ignore = "needs the Rumble-2-Flash checkpoint"]
fn the_retained_route_matches_the_atomic_route_on_cpu() {
    compare_retained_with_atomic(DeviceProfile::cpu());
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the Rumble-2-Flash checkpoint"]
fn the_retained_route_matches_the_atomic_route_on_metal() {
    use crate::backends::{BackendPreference, DeviceSelector};
    let _lease = crate::backends::metal_test_device_or_skip(
        "the_retained_route_matches_the_atomic_route_on_metal",
    );
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    if !matches!(profile.device, candle_core::Device::Metal(_)) {
        eprintln!("skipping: no Metal device");
        return;
    }
    compare_retained_with_atomic(profile);
}

fn fixture() -> Fixture {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rumble/reference.json");
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "{} is missing ({error}); generate it with scripts/fixtures/generate_rumble_reference.py",
            path.display()
        )
    });
    serde_json::from_str(&raw).unwrap()
}

fn run_reference_leg(profile: DeviceProfile, tolerance: Tolerance) {
    let fixture = fixture();
    let audio_dir = PathBuf::from(
        std::env::var_os("IZWI_RUMBLE_AUDIO_DIR")
            .expect("IZWI_RUMBLE_AUDIO_DIR names the audio the generator wrote"),
    );
    let model = load(profile);
    let mut near_ties = 0;
    for (name, case) in &fixture.cases {
        let bytes = std::fs::read(audio_dir.join(&case.audio)).unwrap();
        assert!(
            fixture.audio_sha256.contains_key(&case.audio),
            "{name}: audio is not in the fixture"
        );
        let (samples, rate) = decode_audio_bytes(&bytes).unwrap();
        assert_eq!(rate, 16_000, "{name}");
        let request = RumbleRequestOptions {
            language: case.language.clone(),
            prompt: case.prompt.clone(),
            style: RumbleTranscriptStyle::from_request(&AsrTranscriptStyle {
                digits: case.digits,
                filler: case.filler,
                verbatim: case.verbatim,
            }),
        };

        let features = model.prepare_features(&samples, rate).unwrap();
        assert_eq!(features.mel_frames, case.mel_frames, "{name}: mel frames");
        assert_eq!(features.audio_tokens, case.audio_tokens, "{name}: tokens");
        assert_eq!(
            case.features.shape,
            vec![features.mel_frames, features.n_mels],
            "{name}"
        );
        let feature_error = sampled_error(&features.flat, &case.features, name);
        assert!(
            feature_error <= tolerance.features,
            "{name}: log-mel features differ by {feature_error:.3e} of the RMS"
        );

        let prompt = model
            .build_prompt(features.audio_tokens, &request.prompt_options())
            .unwrap();
        assert_eq!(prompt.ids, case.prompt_ids, "{name}: prompt ids");

        let audio = model.audio_embeddings(&features).unwrap().embeddings;
        assert_eq!(
            audio.dims()[1..],
            case.projected.shape[..],
            "{name}: projected shape"
        );
        let projected = audio
            .to_dtype(candle_core::DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let projected_error = sampled_error(&projected, &case.projected, name);
        assert!(
            projected_error <= tolerance.projected,
            "{name}: projected audio differs by {projected_error:.3e} of the RMS"
        );

        let mut cache = cache(&model);
        let transcription = model
            .transcribe_physical(
                &samples,
                rate,
                &request,
                RumbleGenerationOptions {
                    max_new_tokens: MAX_NEW_TOKENS,
                    ..RumbleGenerationOptions::default()
                },
                &mut cache,
                None,
            )
            .unwrap();
        // The first near tie in the reference's own decode: before it the
        // greedy paths must agree, so the transcripts do when there is none.
        let first_near_tie = case
            .steps
            .iter()
            .position(|step| step.margin < tolerance.near_tie);
        match first_near_tie {
            None if case.steps.len() == case.generated_ids.len() => {
                assert_eq!(transcription.text, case.text, "{name}: transcript");
            }
            _ => near_ties += 1,
        }
        assert!(
            case.steps
                .iter()
                .zip(&case.generated_ids)
                .all(|(step, id)| step.token == *id),
            "{name}: fixture steps are not its greedy tokens"
        );
        eprintln!(
            "{name}: features {feature_error:.2e}, projected {projected_error:.2e}, {:?}",
            transcription.text
        );
    }
    eprintln!(
        "{} cases, {near_ties} with a near tie or a decode past the teacher-forced steps",
        fixture.cases.len()
    );
}

#[test]
#[ignore = "needs the Rumble-2-Flash checkpoint, the reference fixture and IZWI_RUMBLE_AUDIO_DIR"]
fn rumble_matches_the_f32_reference_on_cpu() {
    run_reference_leg(
        DeviceProfile::cpu(),
        Tolerance {
            near_tie: 0.05,
            features: 1e-3,
            projected: 1e-2,
        },
    );
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the Rumble-2-Flash checkpoint, the reference fixture and IZWI_RUMBLE_AUDIO_DIR"]
fn rumble_matches_the_f32_reference_on_metal() {
    use crate::backends::{BackendPreference, DeviceSelector};
    let _lease =
        crate::backends::metal_test_device_or_skip("rumble_matches_the_f32_reference_on_metal");
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    if !matches!(profile.device, candle_core::Device::Metal(_)) {
        eprintln!("skipping: no Metal device");
        return;
    }
    run_reference_leg(
        profile,
        Tolerance {
            near_tie: 0.25,
            features: 1e-3,
            projected: 2e-2,
        },
    );
}
