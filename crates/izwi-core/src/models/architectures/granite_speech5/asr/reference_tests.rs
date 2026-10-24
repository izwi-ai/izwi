//! Real-checkpoint parity for Granite Speech 5.0 2B against IBM's model code
//! (transformers, F32 CPU) in `tests/fixtures/granite_speech5/reference.json`
//! (`scripts/fixtures/generate_granite_speech5_reference.py`).
//!
//! Ignored: they need the checkpoint and the 16 kHz audio the script wrote.
//! The safetensors release comes from `IZWI_GRANITE_SPEECH5_MODEL_DIR`, the
//! GGUF release from `IZWI_GRANITE_SPEECH5_GGUF_MODEL_DIR`, else the default
//! models directory; the audio from `IZWI_GRANITE_SPEECH5_AUDIO_DIR`. Each
//! backend leg checks, per case:
//! * izwi's own request options render the reference prompt ids, audio
//!   placeholder expanded;
//! * the stacked log-mel features and the projected audio embeddings match
//!   at the fixture's sample points;
//! * teacher-forced decode tracks the reference top-8 logits, and the argmax
//!   equals the reference token wherever the reference is not a near tie;
//! * the free-running transcript equals the reference's (the Q8_0 GGUF:
//!   within a word error rate), unless izwi's argmax left the reference's
//!   tokens at one of those near ties: the transcript is its own from there.
//!
//! A further test compares every tensor the GGUF loader builds with the
//! safetensors release's.
//!
//! Run: `IZWI_GRANITE_SPEECH5_AUDIO_DIR=… cargo test -p izwi-core --release
//! --features metal --lib granite_speech5::asr::reference_tests -- --ignored --nocapture`
use super::*;
use crate::backends::DeviceProfile;
use crate::runtime::audio_io::decode_audio_bytes;
use candle_core::Tensor;
use std::collections::BTreeMap;

const MAX_NEW_TOKENS: usize = 200;

struct Tolerance {
    /// Reference steps whose top-2 margin is below this are near ties, where
    /// the argmax may differ.
    near_tie: f32,
    /// Word error rate of the free-running transcript against the
    /// reference's.
    word_error_rate: f32,
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

fn model_dir(variant: ModelVariant) -> PathBuf {
    let env = if variant.is_granite_speech5_gguf() {
        "IZWI_GRANITE_SPEECH5_GGUF_MODEL_DIR"
    } else {
        "IZWI_GRANITE_SPEECH5_MODEL_DIR"
    };
    std::env::var_os(env).map(PathBuf::from).unwrap_or_else(|| {
        dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("izwi")
            .join("models")
            .join(variant.dir_name())
    })
}

/// Word-level edit distance over lowercase words without punctuation, per
/// reference word.
fn word_error_rate(hypothesis: &str, reference: &str) -> f32 {
    let words = |text: &str| -> Vec<String> {
        text.split_whitespace()
            .map(|word| {
                word.chars()
                    .filter(|ch| ch.is_alphanumeric() || *ch == '\'')
                    .collect::<String>()
                    .to_lowercase()
            })
            .filter(|word| !word.is_empty())
            .collect()
    };
    let (hypothesis, reference) = (words(hypothesis), words(reference));
    let mut row = (0..=hypothesis.len()).collect::<Vec<_>>();
    for (i, expected) in reference.iter().enumerate() {
        let mut previous = row[0];
        row[0] = i + 1;
        for (j, word) in hypothesis.iter().enumerate() {
            let substitution = previous + usize::from(word != expected);
            previous = row[j + 1];
            row[j + 1] = substitution.min(row[j] + 1).min(row[j + 1] + 1);
        }
    }
    row[hypothesis.len()] as f32 / reference.len().max(1) as f32
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
    // The first step whose argmax is not the reference's token. Free-running
    // decode has the reference's context up to there, so it takes the same
    // branch.
    let mut departure = None;
    for (index, (step, token)) in case.steps.iter().zip(&case.generated_ids).enumerate() {
        let (argmax, error) = top_logit_error(&logits, step);
        logit_error = logit_error.max(error);
        if argmax != step.next && departure.is_none() {
            departure = Some((index, step.margin));
        }
        if step.margin >= tolerance.near_tie {
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

    let word_error_rate = word_error_rate(&transcript.text, &case.text);
    let departed = departure
        .map(|(index, margin)| format!(", left the reference at step {index} (margin {margin:.3})"))
        .unwrap_or_default();
    eprintln!(
        "{name}: features {features_error:.2e}, projected {projected_error:.2e}, \
         top-8 logits {logit_error:.4}, WER {word_error_rate:.3}{departed}\n  izwi: {:?}",
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
    match departure {
        // A near tie went the other way; the asserts above allow nothing else.
        Some(_) => {}
        None if tolerance.word_error_rate == 0.0 => {
            assert_eq!(transcript.text, case.text.trim(), "{name}: transcript");
        }
        None => assert!(
            word_error_rate <= tolerance.word_error_rate,
            "{name}: WER {word_error_rate:.3}: {:?}",
            transcript.text
        ),
    }
}

fn run_leg(profile: DeviceProfile, variant: ModelVariant, tolerance: Tolerance) {
    let dir = model_dir(variant);
    if ensure_granite_speech5_artifacts(&dir, variant).is_err() {
        eprintln!("skipping: no {variant} checkpoint in {}", dir.display());
        return;
    }
    let fixture = fixture();
    let model = GraniteSpeech5AsrModel::load(&dir, variant, profile).unwrap();
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
        ModelVariant::GraniteSpeech502B,
        Tolerance {
            near_tie: 0.25,
            word_error_rate: 0.0,
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
        ModelVariant::GraniteSpeech502B,
        Tolerance {
            near_tie: 0.25,
            word_error_rate: 0.0,
            features: 1e-4,
            projected: 1e-4,
            logit: 0.15,
        },
    );
}

#[test]
#[ignore = "needs the Granite Speech 5.0 GGUF checkpoint and IZWI_GRANITE_SPEECH5_AUDIO_DIR"]
fn granite_speech5_gguf_tracks_the_f32_reference_on_cpu() {
    run_leg(
        DeviceProfile::cpu(),
        ModelVariant::GraniteSpeech502BGguf,
        GGUF_TOLERANCE,
    );
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the Granite Speech 5.0 GGUF checkpoint and IZWI_GRANITE_SPEECH5_AUDIO_DIR"]
fn granite_speech5_gguf_tracks_the_f32_reference_on_metal() {
    use crate::backends::{BackendPreference, DeviceSelector};
    use candle_core::Device;
    let _lease = crate::backends::metal_test_device_or_skip(
        "granite_speech5_gguf_tracks_the_f32_reference_on_metal",
    );
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    if !matches!(profile.device, Device::Metal(_)) {
        eprintln!("skipping: no Metal device");
        return;
    }
    run_leg(profile, ModelVariant::GraniteSpeech502BGguf, GGUF_TOLERANCE);
}

/// The Q8_0 language model against the F32 reference. The mmproj's F16
/// encoder and projector are the BF16 release's weights exactly. Measured on
/// the 7 cases: projected 9e-6 (of the RMS); top-8 logits 0.66 on the CPU,
/// whose Q8_0 matmul also rounds activations to 8 bits, 0.37 on Metal. Each
/// leg took the other branch of one reference tie below 0.08.
const GGUF_TOLERANCE: Tolerance = Tolerance {
    near_tie: 1.0,
    word_error_rate: 0.1,
    features: 1e-4,
    projected: 1e-4,
    logit: 1.0,
};

/// Relative L2 distance of `actual` from `expected`.
fn relative_l2(actual: &Tensor, expected: &Tensor) -> f32 {
    let actual = actual.to_dtype(DType::F32).unwrap();
    let expected = expected.to_dtype(DType::F32).unwrap();
    assert_eq!(actual.dims(), expected.dims());
    let norm = |t: &Tensor| {
        t.sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .sqrt()
    };
    norm(&(&actual - &expected).unwrap()) / norm(&expected).max(f32::MIN_POSITIVE)
}

/// Every tensor the GGUF loader builds equals the safetensors release's: the
/// mmproj's encoder and projector exactly up to F16 rounding (BatchNorm
/// against the folded running statistics), and the Q8_0 language model
/// within quantization error, its Q and K rows back in half-split order.
#[test]
#[ignore = "needs both Granite Speech 5.0 checkpoints"]
fn granite_speech5_gguf_tensors_match_the_safetensors_release() {
    use super::gguf::{joined_qtensor, mmproj_var_builder, open_gguf, QuantizedPart};
    use candle_core::safetensors::MmapedSafetensors;
    let safetensors_dir = model_dir(ModelVariant::GraniteSpeech502B);
    let gguf_dir = model_dir(ModelVariant::GraniteSpeech502BGguf);
    let (
        Ok(GraniteSpeech5Weights::Safetensors(weights)),
        Ok(GraniteSpeech5Weights::Gguf {
            language_model,
            mmproj,
        }),
    ) = (
        ensure_granite_speech5_artifacts(&safetensors_dir, ModelVariant::GraniteSpeech502B),
        ensure_granite_speech5_artifacts(&gguf_dir, ModelVariant::GraniteSpeech502BGguf),
    )
    else {
        eprintln!("skipping: needs both checkpoints");
        return;
    };
    let cpu = candle_core::Device::Cpu;
    let reference = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[&weights], DType::F32, &cpu).unwrap()
    };
    let config = GraniteSpeech5Config::load(&safetensors_dir).unwrap();

    // Encoder and projector.
    let mmproj = mmproj_var_builder(open_gguf(&mmproj).unwrap(), DType::F32, &cpu).unwrap();
    let mapped = unsafe { MmapedSafetensors::new(&weights).unwrap() };
    let mut names = mapped
        .tensors()
        .into_iter()
        .map(|(name, _)| name)
        .filter(|name| name.starts_with("encoder.") || name.starts_with("projector."))
        .filter(|name| !name.contains(".batch_norm."))
        .collect::<Vec<_>>();
    names.sort();
    let mut checked = 0;
    let mut worst = (String::new(), 0f32);
    for name in &names {
        let expected = reference.get_unchecked_dtype(name, DType::F32).unwrap();
        let actual = mmproj.get_unchecked_dtype(name, DType::F32).unwrap();
        let error = relative_l2(&actual, &expected);
        if error > worst.1 {
            worst = (name.clone(), error);
        }
        checked += 1;
    }
    eprintln!(
        "mmproj: {checked} tensors, worst relative L2 {:.2e} ({})",
        worst.1, worst.0
    );
    assert!(worst.1 < 1e-3, "{worst:?}");
    for layer in 0..config.encoder_config.num_layers {
        let bn = |part: &str| {
            reference
                .get_unchecked_dtype(
                    &format!("encoder.layers.{layer}.conv.batch_norm.{part}"),
                    DType::F32,
                )
                .unwrap()
        };
        let scale = (bn("weight") / (bn("running_var") + 1e-5).unwrap().sqrt().unwrap()).unwrap();
        let shift = (bn("bias") - (bn("running_mean") * &scale).unwrap()).unwrap();
        let folded = |part: &str| {
            mmproj
                .get_unchecked_dtype(
                    &format!("encoder.layers.{layer}.conv.batch_norm.folded_{part}"),
                    DType::F32,
                )
                .unwrap()
        };
        assert!(
            relative_l2(&folded("scale"), &scale) < 1e-3,
            "layer {layer} scale"
        );
        assert!(
            relative_l2(&folded("shift"), &shift) < 1e-3,
            "layer {layer} shift"
        );
    }

    // Language model.
    let gguf = open_gguf(&language_model).unwrap();
    let text = &config.text_config;
    let dequantized = |parts: &[QuantizedPart<'_>]| {
        joined_qtensor(&gguf, parts, &cpu)
            .unwrap()
            .dequantize(&cpu)
            .unwrap()
    };
    let mut worst = (String::new(), 0f32);
    let mut compare = |gguf_name: &str, rope_heads: Option<usize>, hf_name: &str| {
        let actual = dequantized(&[QuantizedPart {
            name: gguf_name,
            rope_heads,
        }]);
        let expected = reference
            .get_unchecked_dtype(&format!("language_model.{hf_name}"), DType::F32)
            .unwrap();
        let error = relative_l2(&actual, &expected);
        if error > worst.1 {
            worst = (gguf_name.to_string(), error);
        }
    };
    for layer in 0..text.num_hidden_layers {
        let gguf_name = |part: &str| format!("blk.{layer}.{part}.weight");
        let hf = |part: &str| format!("model.layers.{layer}.{part}.weight");
        compare(
            &gguf_name("attn_q"),
            Some(text.num_attention_heads),
            &hf("self_attn.q_proj"),
        );
        compare(
            &gguf_name("attn_k"),
            Some(text.num_key_value_heads),
            &hf("self_attn.k_proj"),
        );
        compare(&gguf_name("attn_v"), None, &hf("self_attn.v_proj"));
        compare(&gguf_name("attn_output"), None, &hf("self_attn.o_proj"));
        compare(&gguf_name("ffn_gate"), None, &hf("mlp.gate_proj"));
        compare(&gguf_name("ffn_up"), None, &hf("mlp.up_proj"));
        compare(&gguf_name("ffn_down"), None, &hf("mlp.down_proj"));
    }
    compare("output.weight", None, "lm_head.weight");
    compare("token_embd.weight", None, "model.embed_tokens.weight");
    eprintln!(
        "language model: worst relative L2 {:.2e} ({})",
        worst.1, worst.0
    );
    // Q8_0 rounds each weight to 1/127 of its block's largest; a Q or K
    // left in llama.cpp's row order would be off by order 1.
    assert!(worst.1 < 1e-2, "{worst:?}");
    // The q_proj of layer 0 left in llama.cpp's order is far off.
    let interleaved = dequantized(&[QuantizedPart {
        name: "blk.0.attn_q.weight",
        rope_heads: None,
    }]);
    let q_proj = reference
        .get_unchecked_dtype(
            "language_model.model.layers.0.self_attn.q_proj.weight",
            DType::F32,
        )
        .unwrap();
    assert!(relative_l2(&interleaved, &q_proj) > 0.5);
}
