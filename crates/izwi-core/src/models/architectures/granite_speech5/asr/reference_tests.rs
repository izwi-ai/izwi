//! Real-checkpoint parity for Granite Speech 5.0 2B against IBM's model code
//! (transformers, F32 CPU) in `tests/fixtures/granite_speech5/reference.json`
//! (`scripts/fixtures/generate_granite_speech5_reference.py`).
//!
//! Ignored: they need the checkpoint and the 16 kHz audio the script wrote.
//! The model comes from `IZWI_GRANITE_SPEECH5_MODEL_DIR`, else the default
//! models directory; the audio from `IZWI_GRANITE_SPEECH5_AUDIO_DIR`. Each
//! backend leg checks, per case:
//! * izwi's own request options render the reference prompt ids, audio
//!   placeholder expanded;
//! * the stacked log-mel features and the projected audio embeddings match
//!   at the fixture's sample points;
//! * teacher-forced decode tracks the reference top-8 logits, and the argmax
//!   equals the reference token wherever the reference is not a near tie;
//! * the free-running transcript equals the reference's.
//!
//! Run: `IZWI_GRANITE_SPEECH5_AUDIO_DIR=… cargo test -p izwi-core --release
//! --features metal --lib granite_speech5::asr::reference_tests -- --ignored --nocapture`
use super::*;
use crate::backends::DeviceProfile;
use crate::runtime::audio_io::decode_audio_bytes;
use candle_core::Tensor;
use std::collections::BTreeMap;

/// Reference steps whose top-2 margin is below this are near ties.
const NEAR_TIE: f32 = 0.25;
const MAX_NEW_TOKENS: usize = 200;

struct Tolerance {
    /// Max |izwi - reference| at the sample points, as a fraction of the
    /// reference RMS.
    features: f32,
    projected: f32,
    /// Max |izwi - reference| over each step's reference top-8 logits.
    logit: f32,
}

#[derive(serde::Deserialize)]
struct Fixture {
    audio: BTreeMap<String, String>,
    audio_sha256: BTreeMap<String, String>,
    cases: BTreeMap<String, FixtureCase>,
}

#[derive(serde::Deserialize)]
struct FixtureCase {
    audio: String,
    prompt_ids: Vec<u32>,
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
    next: u32,
    margin: f32,
    top: Vec<(u32, f32)>,
}

fn fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/granite_speech5/reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn model_dir() -> PathBuf {
    std::env::var_os("IZWI_GRANITE_SPEECH5_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("izwi")
                .join("models")
                .join("Granite-Speech-5.0-2B")
        })
}

/// The request izwi sends for each fixture case.
fn request(case: &str) -> GraniteSpeech5RequestOptions {
    let mut request = GraniteSpeech5RequestOptions::default();
    match case {
        "test_punctuated" | "multilingual_punctuated" => {}
        "test_raw" => {
            request.prompt = Some(GRANITE_SPEECH5_RAW_ASR_INSTRUCTION.to_string());
        }
        "test_keywords" => {
            request.keywords = vec!["Agentic AI".into(), "sovereign data networks".into()];
        }
        "multilingual_language_id" => request.language = Some("fr".into()),
        "multilingual_to_english" => request.translate_to = Some("English".into()),
        "test_to_german" => request.translate_to = Some("German".into()),
        other => panic!("unknown fixture case {other}"),
    }
    request
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    hex::encode(sha2::Sha256::digest(bytes))
}

/// The case's audio, checked to be the bytes the reference read.
fn audio(fixture: &Fixture, key: &str) -> Option<(Vec<f32>, u32)> {
    let dir = PathBuf::from(std::env::var_os("IZWI_GRANITE_SPEECH5_AUDIO_DIR")?);
    let bytes = std::fs::read(dir.join(&fixture.audio[key])).unwrap();
    assert_eq!(
        sha256_hex(&bytes),
        fixture.audio_sha256[key],
        "{key} audio differs from the audio the reference read"
    );
    let (samples, rate) = decode_audio_bytes(&bytes).unwrap();
    assert_eq!(rate, 16_000);
    Some((samples, rate))
}

/// Max |izwi - reference| at the sample points over the reference RMS.
fn sample_error(name: &str, actual: &[f32], shape: &[usize], sample: &FixtureSample) -> f32 {
    assert_eq!(shape, sample.shape.as_slice(), "{name} shape");
    let error = sample
        .indices
        .iter()
        .zip(&sample.values)
        .map(|(index, expected)| (actual[*index] - expected).abs())
        .fold(0f32, f32::max);
    error / sample.rms.max(f32::EPSILON)
}

fn top_logit_error(logits: &Tensor, step: &FixtureStep) -> (u32, f32) {
    let logits = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index as u32)
        .unwrap();
    let error = step
        .top
        .iter()
        .map(|(id, expected)| (logits[*id as usize] - expected).abs())
        .fold(0f32, f32::max);
    (argmax, error)
}

fn scratch_cache(model: &GraniteSpeech5AsrModel, tokens: usize) -> PhysicalPagedKvCache {
    let text = &model.config.text_config;
    PhysicalPagedKvCache::scratch_for_probe(
        &model.device.device,
        model.runtime.kv_dtype(),
        text.num_hidden_layers,
        text.num_key_value_heads,
        text.hidden_size / text.num_attention_heads,
        tokens,
    )
    .unwrap()
}

fn run_case(
    model: &GraniteSpeech5AsrModel,
    name: &str,
    case: &FixtureCase,
    samples: &[f32],
    tolerance: &Tolerance,
) {
    let request = request(name);
    let prompt = model
        .build_prompt(&request.prompt_options().unwrap())
        .unwrap();
    let features = model.preprocessor.prepare(samples, 16_000).unwrap();
    let flat_features = features.input_features.concat();
    let features_error = sample_error(
        "features",
        &flat_features,
        &[1, features.encoder_frames, features.encoder_dim],
        &case.features,
    );

    let (audio_embeds, _) = model.runtime.audio_embeddings(&features).unwrap();
    let projected = audio_embeds
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let projected_error = sample_error(
        "projected",
        &projected,
        audio_embeds.dims(),
        &case.projected,
    );
    let special = model.prompt_tokenizer.special_tokens();
    let prompt_ids = GraniteSpeech5Runtime::expanded_input_ids(
        &prompt,
        audio_embeds.dim(1).unwrap(),
        special.audio_token_id,
    )
    .unwrap();
    assert_eq!(prompt_ids, case.prompt_ids, "{name}: prompt ids");

    // Teacher-forced decode over the reference's own tokens.
    let total = prompt_ids.len() + case.steps.len() + 1;
    let mut cache = scratch_cache(model, total);
    let lm = model.runtime.language_model();
    let rope = lm.rope_cache(total).unwrap();
    let mut logits = model
        .runtime
        .prefill_logits(&prompt, special, &audio_embeds, &mut cache)
        .unwrap();
    let mut logit_error = 0f32;
    for (index, (step, token)) in case.steps.iter().zip(&case.generated_ids).enumerate() {
        let (argmax, error) = top_logit_error(&logits, step);
        logit_error = logit_error.max(error);
        if step.margin >= NEAR_TIE {
            assert_eq!(
                argmax, step.next,
                "{name}: step {index} argmax (reference margin {:.3})",
                step.margin
            );
        }
        if *token == special.eos_token_id {
            break;
        }
        let input = Tensor::new(&[[*token]], &model.device.device).unwrap();
        let raw = lm
            .forward_token(&input, prompt_ids.len() + index, &mut cache, &rope)
            .unwrap();
        logits = lm.scale_logits(&raw).unwrap();
    }

    // Free-running transcript, as the ASR route runs it.
    let mut cache = scratch_cache(model, prompt_ids.len() + MAX_NEW_TOKENS + 1);
    let transcript = model
        .transcribe_physical(
            samples,
            16_000,
            &request,
            GraniteSpeech5GenerationOptions {
                max_new_tokens: MAX_NEW_TOKENS,
                ..GraniteSpeech5GenerationOptions::default()
            },
            &mut cache,
            None,
        )
        .unwrap();

    eprintln!(
        "{name}: features {features_error:.2e}, projected {projected_error:.2e}, \
         top-8 logits {logit_error:.4}\n  izwi: {:?}",
        transcript.text
    );
    assert!(
        features_error <= tolerance.features,
        "{name}: features error {features_error:.2e}"
    );
    assert!(
        projected_error <= tolerance.projected,
        "{name}: projected error {projected_error:.2e}"
    );
    assert!(
        logit_error <= tolerance.logit,
        "{name}: top-8 logit error {logit_error:.3}"
    );
    assert_eq!(transcript.text, case.text.trim(), "{name}: transcript");
}

fn run_leg(profile: DeviceProfile, tolerance: Tolerance) {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: no checkpoint in {}", dir.display());
        return;
    }
    let fixture = fixture();
    let model = GraniteSpeech5AsrModel::load(&dir, profile).unwrap();
    for (name, case) in &fixture.cases {
        let Some((samples, _)) = audio(&fixture, &case.audio) else {
            eprintln!("skipping: IZWI_GRANITE_SPEECH5_AUDIO_DIR is not set");
            return;
        };
        run_case(&model, name, case, &samples, &tolerance);
    }
}

#[test]
#[ignore = "needs the Granite Speech 5.0 checkpoint and IZWI_GRANITE_SPEECH5_AUDIO_DIR"]
fn granite_speech5_matches_the_f32_reference_on_cpu() {
    // Measured on the 7 cases: features 5e-6, projected 9e-6 (of the RMS),
    // top-8 logits below 5e-4.
    run_leg(
        DeviceProfile::cpu(),
        Tolerance {
            features: 1e-4,
            projected: 1e-4,
            logit: 0.01,
        },
    );
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the Granite Speech 5.0 checkpoint and IZWI_GRANITE_SPEECH5_AUDIO_DIR"]
fn granite_speech5_matches_the_f32_reference_on_metal() {
    use crate::backends::{BackendPreference, DeviceSelector};
    use candle_core::Device;
    let _lease = crate::backends::metal_test_device_or_skip(
        "granite_speech5_matches_the_f32_reference_on_metal",
    );
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    if !matches!(profile.device, Device::Metal(_)) {
        eprintln!("skipping: no Metal device");
        return;
    }
    // Selective F16 decoder and weight-only F16 encoder. Measured on the 7
    // cases: features 5e-6, projected 7e-6 (of the RMS), top-8 logits 0.038.
    run_leg(
        profile,
        Tolerance {
            features: 1e-4,
            projected: 1e-4,
            logit: 0.15,
        },
    );
}
