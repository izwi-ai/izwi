//! Voice cloning against upstream: `tests/fixtures/chatterbox/clone_reference.json`
//! from `scripts/fixtures/generate_chatterbox_clone_reference.py`, on that
//! script's synthetic, bit-reproducible reference. The DSP stages run on
//! every test pass; the three encoders need the checkpoint:
//!
//! Run: `IZWI_CHATTERBOX_MODEL_DIR=… cargo test -p izwi-core --release --lib
//! chatterbox::clone_reference_tests -- --ignored --nocapture`

use std::path::PathBuf;

use candle_core::{DType, Device};

use super::config::{S3GEN_WEIGHTS_FILE, VOICE_ENCODER_WEIGHTS_FILE};
use super::fixture_support::{reference_signal, sample_error, Sample};
use super::reference::ChatterboxReferenceEncoders;
use super::reference_dsp::{
    kaldi_fbank, log_mel_s3tokenizer, mel_s3gen_reference, mel_voice_encoder, sinc_resample,
    trim_silence,
};
use super::weights::{host_f32, mmap_safetensors};

#[derive(serde::Deserialize)]
struct CloneFixture {
    signal: Sample,
    wav16: Sample,
    s3gen: S3GenClone,
    t3: T3Clone,
}

#[derive(serde::Deserialize)]
struct S3GenClone {
    mel: Sample,
    tokens: Vec<u32>,
    fbank: Sample,
    xvector: Vec<f32>,
}

#[derive(serde::Deserialize)]
struct T3Clone {
    tokenizer_mel: Sample,
    tokens: Vec<u32>,
    trim: [usize; 2],
    ve_mel: Sample,
    ve: Vec<f32>,
}

fn fixture() -> CloneFixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/chatterbox/clone_reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// Max |a − b| over the largest |b|, and the cosine similarity.
fn vector_error(actual: &[f32], expected: &[f32]) -> (f32, f32) {
    assert_eq!(actual.len(), expected.len());
    let scale = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
    let max = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let dot: f32 = actual.iter().zip(expected).map(|(a, b)| a * b).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    (max / scale, dot / (norm(actual) * norm(expected)))
}

#[test]
fn clone_frontends_match_upstream_dsp() {
    let fixture = fixture();
    let wav24 = reference_signal();
    let error = sample_error("signal", &wav24, &[wav24.len()], &fixture.signal);
    assert_eq!(error, 0.0, "the synthetic reference is regenerated exactly");

    let wav16 = sinc_resample(&wav24, 24_000, 16_000);
    let error = sample_error("wav16", &wav16, &[wav16.len()], &fixture.wav16);
    eprintln!("wav16 {error:.2e}");
    assert!(error < 1e-5, "16 kHz resample {error}");

    let (mel, frames) = mel_s3gen_reference(&wav24);
    let error = sample_error("s3gen mel", &mel, &[frames, 80], &fixture.s3gen.mel);
    eprintln!("s3gen mel {error:.2e}");
    assert!(error < 1e-4, "S3Gen reference mel {error}");

    let (fbank, frames) = kaldi_fbank(&wav16);
    let error = sample_error("fbank", &fbank, &[frames, 80], &fixture.s3gen.fbank);
    eprintln!("fbank {error:.2e}");
    assert!(error < 1e-3, "Kaldi fbank {error}");

    let trimmed = trim_silence(&wav16, 20.0);
    let start = (trimmed.as_ptr() as usize - wav16.as_ptr() as usize) / std::mem::size_of::<f32>();
    assert_eq!([start, start + trimmed.len()], fixture.t3.trim, "trim span");

    let (mel, frames) = mel_voice_encoder(trimmed);
    let error = sample_error("ve mel", &mel, &[frames, 40], &fixture.t3.ve_mel);
    eprintln!("ve mel {error:.2e}");
    assert!(error < 1e-4, "voice-encoder mel {error}");
}

#[test]
#[ignore = "needs the Chatterbox checkpoint (IZWI_CHATTERBOX_MODEL_DIR)"]
fn clone_encoders_match_upstream_on_cpu() {
    run_clone_encoders(Device::Cpu, 1e-4, 1e-3);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the Chatterbox checkpoint (IZWI_CHATTERBOX_MODEL_DIR)"]
fn clone_encoders_match_upstream_on_metal() {
    use crate::backends::{BackendPreference, DeviceSelector};
    let _lease =
        crate::backends::metal_test_device_or_skip("clone_encoders_match_upstream_on_metal");
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    if !matches!(profile.device, Device::Metal(_)) {
        eprintln!("skipping: no Metal device");
        return;
    }
    run_clone_encoders(profile.device, 1e-3, 5e-3);
}

/// The encoders on `device`: the tokenizer's log-mel within `mel_tolerance`
/// of upstream's RMS, at most one token off in either prompt, and both
/// speaker vectors within `vector_tolerance` of their largest component.
fn run_clone_encoders(device: Device, mel_tolerance: f32, vector_tolerance: f32) {
    let fixture = fixture();
    let dir = std::env::var_os("IZWI_CHATTERBOX_MODEL_DIR")
        .map(PathBuf::from)
        .expect("IZWI_CHATTERBOX_MODEL_DIR");
    let s3gen = mmap_safetensors(&dir.join(S3GEN_WEIGHTS_FILE), DType::F32, &device).unwrap();
    let ve = mmap_safetensors(&dir.join(VOICE_ENCODER_WEIGHTS_FILE), DType::F32, &device).unwrap();
    let filters = host_f32(&s3gen.get((128, 201), "tokenizer._mel_filters").unwrap()).unwrap();
    let encoders = ChatterboxReferenceEncoders::load(ve, s3gen).unwrap();

    let wav24 = reference_signal();
    let wav16 = sinc_resample(&wav24, 24_000, 16_000);
    let t3_input = &wav16[..wav16.len().min(6 * 16_000)];

    let (mel, frames) = log_mel_s3tokenizer(t3_input, &filters);
    let error = sample_error(
        "tokenizer mel",
        &mel,
        &[128, frames],
        &fixture.t3.tokenizer_mel,
    );
    eprintln!("tokenizer mel {error:.2e}");
    assert!(error < mel_tolerance, "S3 tokenizer log-mel {error}");

    let token_mismatches = |actual: &[u32], expected: &[u32]| {
        assert_eq!(actual.len(), expected.len(), "token count");
        actual.iter().zip(expected).filter(|(a, b)| a != b).count()
    };
    let t3_tokens = encoders
        .tokenizer()
        .tokenize(t3_input, Some(150), &device)
        .unwrap();
    let mismatched = token_mismatches(&t3_tokens, &fixture.t3.tokens);
    eprintln!(
        "T3 prompt tokens: {mismatched} of {} differ",
        t3_tokens.len()
    );
    assert!(
        mismatched <= 1,
        "T3 prompt tokens differ in {mismatched} places"
    );

    let mut s3gen_tokens = encoders
        .tokenizer()
        .tokenize(&wav16, None, &device)
        .unwrap();
    s3gen_tokens.truncate(fixture.s3gen.tokens.len());
    let mismatched = token_mismatches(&s3gen_tokens, &fixture.s3gen.tokens);
    eprintln!(
        "S3Gen prompt tokens: {mismatched} of {} differ",
        s3gen_tokens.len()
    );
    assert!(
        mismatched <= 1,
        "S3Gen prompt tokens differ in {mismatched} places"
    );

    let xvector = encoders.campplus().embed(&wav16, &device).unwrap();
    let (error, cosine) = vector_error(&xvector, &fixture.s3gen.xvector);
    eprintln!("x-vector {error:.2e}, cosine {cosine:.7}");
    assert!(
        error < vector_tolerance && cosine > 0.9999,
        "CAMPPlus x-vector"
    );

    let ve_embedding = encoders.voice_encoder().embed(&wav16, &device).unwrap();
    let (error, cosine) = vector_error(&ve_embedding, &fixture.t3.ve);
    eprintln!("voice encoder {error:.2e}, cosine {cosine:.7}");
    assert!(
        error < vector_tolerance && cosine > 0.9999,
        "voice-encoder embedding"
    );

    // The whole preparation at the reference's own rate.
    let voice = encoders.prepare_voice(&wav24, 24_000).unwrap();
    assert_eq!(voice.t3.prompt_speech_tokens, t3_tokens);
    assert_eq!(voice.s3gen.prompt_tokens, s3gen_tokens);
    assert_eq!(voice.s3gen.prompt_mel_frames, 2 * s3gen_tokens.len());
}
