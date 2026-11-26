//! Parity of the composed trunk with the reference fixture
//! (`tests/fixtures/inkling2_golden`, written by
//! `scripts/fixtures/generate_inkling2_golden.py`).

use std::collections::BTreeMap;

use candle_core::{Device, Tensor};
use safetensors::SafeTensors;

use super::checkpoint::Inkling2Checkpoint;
use super::config::tests::{fixture, tiny};
use super::trunk::{Inkling2Trace, Inkling2Trunk};

pub(crate) struct Golden {
    pub tokens: Vec<u32>,
    pub prompt_tokens: usize,
    /// `[tokens, unpadded_vocab]`, row major.
    pub logits: Vec<f32>,
    pub trace: BTreeMap<String, Vec<f32>>,
}

pub(crate) fn golden() -> Golden {
    let bytes = std::fs::read(fixture("inkling2_golden").join("golden.safetensors")).unwrap();
    let file = SafeTensors::deserialize(&bytes).unwrap();
    let f32s = |name: &str| {
        file.tensor(name)
            .unwrap()
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|quad| f32::from_le_bytes(*quad))
            .collect::<Vec<_>>()
    };
    let i64s = |name: &str| {
        file.tensor(name)
            .unwrap()
            .data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|bytes| i64::from_le_bytes(*bytes))
            .collect::<Vec<_>>()
    };
    Golden {
        tokens: i64s("tokens")
            .into_iter()
            .map(|token| token as u32)
            .collect(),
        prompt_tokens: i64s("prompt_tokens")[0] as usize,
        logits: f32s("logits"),
        trace: file
            .names()
            .into_iter()
            .filter_map(|name| {
                name.strip_prefix("trace.")
                    .map(|key| (key.to_string(), f32s(name)))
            })
            .collect(),
    }
}

pub(crate) fn tiny_trunk(device: &Device) -> Inkling2Trunk {
    let config = tiny();
    let checkpoint = Inkling2Checkpoint::open(&fixture("inkling2_golden"), &config).unwrap();
    Inkling2Trunk::load(&checkpoint, &config, device).unwrap()
}

fn values(tensor: &Tensor) -> Vec<f32> {
    tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

/// Largest absolute difference; a non-finite value on either side is
/// infinite error.
pub(crate) fn max_error(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    got.iter().zip(want).fold(0.0f32, |worst, (got, want)| {
        let error = (got - want).abs();
        if error.is_finite() {
            worst.max(error)
        } else {
            f32::INFINITY
        }
    })
}

#[test]
fn max_error_does_not_hide_a_nan() {
    assert_eq!(max_error(&[1.0, f32::NAN], &[1.0, 2.0]), f32::INFINITY);
    assert_eq!(max_error(&[1.0, 2.5], &[1.0, 2.0]), 0.5);
}

#[test]
fn composed_trunk_matches_the_reference_logits() {
    trunk_matches_the_reference(&Device::Cpu, 2e-4, 2e-3);
}

/// The same composed trunk on the Metal device (F32 activations; Candle's
/// Metal kernels use fast math, so the tolerance is wider).
#[cfg(feature = "metal")]
#[test]
fn composed_trunk_matches_the_reference_logits_on_metal() {
    let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
        "composed_trunk_matches_the_reference_logits_on_metal",
    ) else {
        return;
    };
    trunk_matches_the_reference(&device, 5e-3, 5e-2);
}

fn trunk_matches_the_reference(device: &Device, activation_tolerance: f32, logit_tolerance: f32) {
    let golden = golden();
    let trunk = tiny_trunk(device);
    let mut state = trunk.new_state().unwrap();
    let mut trace = Inkling2Trace::default();
    let hidden = trunk
        .forward(&golden.tokens, &mut state, Some(&mut trace))
        .unwrap();
    assert_eq!(state.position(), golden.tokens.len());

    // Every recorded activation, in order, so a mismatch names the first
    // place the trunks diverge.
    assert_eq!(trace.entries.len(), golden.trace.len());
    for (name, tensor) in &trace.entries {
        let want = golden
            .trace
            .get(name)
            .unwrap_or_else(|| panic!("no reference for {name}"));
        let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
        let error = max_error(&values(tensor), want);
        assert!(
            error <= activation_tolerance * scale,
            "{name}: error {error} at scale {scale}"
        );
    }
    let logits = values(&trunk.logits(&hidden).unwrap());
    let error = max_error(&logits, &golden.logits);
    eprintln!("{device:?}: logit error {error}");
    assert!(error <= logit_tolerance, "logit error {error}");
    // The fixture is not degenerate: logits spread over several units.
    let spread = golden.logits.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(spread > 5.0);
}

/// Prefill in uneven spans, then one token at a time: the same logits as one
/// span, and the windowed layers hold a bounded number of keys.
#[test]
fn incremental_decode_matches_one_span() {
    let golden = golden();
    let trunk = tiny_trunk(&Device::Cpu);
    let vocab = trunk.config().unpadded_vocab;
    let window = trunk.config().sliding.window.unwrap();
    let mut state = trunk.new_state().unwrap();
    let mut at = 0usize;
    let mut spans = vec![golden.prompt_tokens, 5, 3];
    spans.extend(std::iter::repeat_n(
        1,
        golden.tokens.len() - golden.prompt_tokens - 8,
    ));
    for span in spans {
        let hidden = trunk
            .forward(&golden.tokens[at..at + span], &mut state, None)
            .unwrap();
        let logits = values(&trunk.logits(&hidden).unwrap());
        let error = max_error(&logits, &golden.logits[at * vocab..(at + span) * vocab]);
        assert!(
            error <= 2e-3,
            "tokens {at}..{}: logit error {error}",
            at + span
        );
        at += span;
        let kinds = trunk.config().applications();
        for (held, application) in state.held_keys().iter().zip(&kinds) {
            match trunk.config().heads(application.layer).window {
                Some(_) => assert!(*held < window + span, "held {held}"),
                None => assert_eq!(*held, at),
            }
        }
    }
    assert_eq!(at, golden.tokens.len());
    // The sequence is longer than the window, so trimming really happened.
    assert!(golden.tokens.len() > 2 * window);
}

/// A cloned state continues exactly as the original.
#[test]
fn a_state_snapshot_restores_exactly() {
    let golden = golden();
    let trunk = tiny_trunk(&Device::Cpu);
    let mut state = trunk.new_state().unwrap();
    trunk
        .forward(&golden.tokens[..20], &mut state, None)
        .unwrap();
    let snapshot = state.clone();
    let first = values(
        &trunk
            .forward(&golden.tokens[20..23], &mut state, None)
            .unwrap(),
    );
    // Advance the original further, then resume the snapshot.
    trunk
        .forward(&golden.tokens[23..30], &mut state, None)
        .unwrap();
    let mut resumed = snapshot;
    let again = values(
        &trunk
            .forward(&golden.tokens[20..23], &mut resumed, None)
            .unwrap(),
    );
    assert_eq!(
        first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        again.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    // And it differs from a run without the prefix, so the state matters.
    let mut fresh = trunk.new_state().unwrap();
    let cold = values(
        &trunk
            .forward(&golden.tokens[20..23], &mut fresh, None)
            .unwrap(),
    );
    assert!(max_error(&cold, &first) > 1e-2);
}
