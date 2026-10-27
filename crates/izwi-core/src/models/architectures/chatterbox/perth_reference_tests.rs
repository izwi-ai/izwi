//! PerTh against upstream: `tests/fixtures/chatterbox/perth_reference.json`
//! from `scripts/fixtures/generate_chatterbox_perth_reference.py`, on the
//! synthetic reference signal taken as 32 kHz audio. The STFT and the frame
//! mask run on every test pass; the encoder needs the checkpoint in
//! `IZWI_CHATTERBOX_MODEL_DIR`:
//!
//! Run: `cargo test -p izwi-core --release --lib chatterbox::perth_reference_tests
//! -- --ignored --nocapture`

use std::path::PathBuf;

use candle_core::Device;

use super::fixture_support::{reference_signal, sample_error, Sample};
use super::perth::{
    frame_mask, stft_magphase, PerthDetector, PerthWatermarker, PERTH_CHECKPOINT_FILE,
};

#[derive(serde::Deserialize)]
struct PerthFixture {
    sample_rate: u32,
    signal: Sample,
    magspec: Sample,
    mask: Vec<u8>,
    residual: Sample,
    watermarked: Sample,
    detector: DetectorScores,
}

#[derive(serde::Deserialize)]
struct DetectorScores {
    original: f32,
    watermarked: f32,
}

fn checkpoint() -> PathBuf {
    std::env::var_os("IZWI_CHATTERBOX_MODEL_DIR")
        .map(PathBuf::from)
        .expect("IZWI_CHATTERBOX_MODEL_DIR")
        .join(PERTH_CHECKPOINT_FILE)
}

fn fixture() -> PerthFixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/chatterbox/perth_reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn perth_stft_and_mask_match_upstream() {
    let fixture = fixture();
    let signal = reference_signal();
    assert_eq!(
        sample_error("signal", &signal, &[signal.len()], &fixture.signal),
        0.0
    );
    let (magspec, _, frames) = stft_magphase(&signal).unwrap();
    let error = sample_error("magspec", &magspec, &[1025, frames], &fixture.magspec);
    eprintln!("magspec {error:.2e}");
    // torch transforms in F32 and izwi in F64: bins near the −180 dB floor in
    // silent frames differ by a few hundredths of a dB, which the watermark
    // never sees (the watermarked signal matches within 2e-6).
    assert!(error < 1e-3, "normalised magnitude {error}");
    let mask = frame_mask(&magspec, frames)
        .into_iter()
        .map(u8::from)
        .collect::<Vec<_>>();
    assert_eq!(mask, fixture.mask, "frame mask");
}

#[test]
#[ignore = "needs perth_net_250000.pth.tar in IZWI_CHATTERBOX_MODEL_DIR"]
fn perth_watermark_matches_upstream_on_cpu() {
    let fixture = fixture();
    let watermarker = PerthWatermarker::load(&checkpoint(), &Device::Cpu).unwrap();
    let signal = reference_signal();

    let (magspec, _, frames) = stft_magphase(&signal).unwrap();
    let mask = frame_mask(&magspec, frames);
    let mut residual = watermarker.residual_for_tests(&magspec, frames).unwrap();
    for (index, value) in residual.iter_mut().enumerate() {
        if !mask[index % frames] {
            *value = 0.0;
        }
    }
    let error = sample_error("residual", &residual, &[128, frames], &fixture.residual);
    eprintln!("residual {error:.2e}");
    assert!(error < 1e-4, "encoder residual {error}");

    let watermarked = watermarker.apply(&signal, fixture.sample_rate).unwrap();
    let error = sample_error(
        "watermarked",
        &watermarked,
        &[watermarked.len()],
        &fixture.watermarked,
    );
    eprintln!("watermarked {error:.2e}");
    assert!(error < 1e-4, "watermarked signal {error}");
}

/// The detector port scores the reference signal and izwi's watermarked
/// signal as upstream's detector did, and a stream watermarked piece by
/// piece is detected like the whole signal.
#[test]
#[ignore = "needs perth_net_250000.pth.tar in IZWI_CHATTERBOX_MODEL_DIR"]
fn perth_detector_and_stream_match_upstream_on_cpu() {
    let fixture = fixture();
    let watermarker = PerthWatermarker::load(&checkpoint(), &Device::Cpu).unwrap();
    let detector = PerthDetector::load(&checkpoint(), &Device::Cpu).unwrap();
    let signal = reference_signal();
    let rate = fixture.sample_rate;
    let whole = watermarker.apply(&signal, rate).unwrap();
    let (original, marked) = (
        detector.detect(&signal, rate).unwrap(),
        detector.detect(&whole, rate).unwrap(),
    );
    eprintln!(
        "detector original {original:.4} (upstream {:.4}), watermarked {marked:.4} (upstream {:.4})",
        fixture.detector.original, fixture.detector.watermarked
    );
    assert!((original - fixture.detector.original).abs() < 0.02);
    assert!((marked - fixture.detector.watermarked).abs() < 0.02);

    // Pieces of 0.3 s, as a stream delivers them.
    let mut stream = watermarker.stream(rate);
    let mut streamed = Vec::new();
    for piece in signal.chunks((rate as usize * 3) / 10) {
        streamed.extend(watermarker.push(&mut stream, piece, false).unwrap());
    }
    streamed.extend(watermarker.push(&mut stream, &[], true).unwrap());
    let score = detector.detect(&streamed, rate).unwrap();
    let shared = streamed.len().min(whole.len());
    let diff = streamed[..shared]
        .iter()
        .zip(&whole[..shared])
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!(
        "stream: {} of {} samples, score {score:.4}, max |stream − whole| {diff:.2e}",
        streamed.len(),
        whole.len()
    );
    assert!(streamed.len().abs_diff(whole.len()) <= rate as usize / 100);
    assert!(
        (score - marked).abs() < 0.05,
        "streamed watermark score {score} against {marked} whole"
    );
}
