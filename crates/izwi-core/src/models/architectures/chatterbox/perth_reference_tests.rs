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
use super::perth::{frame_mask, stft_magphase, PerthWatermarker, PERTH_CHECKPOINT_FILE};

#[derive(serde::Deserialize)]
struct PerthFixture {
    sample_rate: u32,
    signal: Sample,
    magspec: Sample,
    mask: Vec<u8>,
    residual: Sample,
    watermarked: Sample,
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
    let dir = std::env::var_os("IZWI_CHATTERBOX_MODEL_DIR")
        .map(PathBuf::from)
        .expect("IZWI_CHATTERBOX_MODEL_DIR");
    let watermarker =
        PerthWatermarker::load(&dir.join(PERTH_CHECKPOINT_FILE), &Device::Cpu).unwrap();
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
