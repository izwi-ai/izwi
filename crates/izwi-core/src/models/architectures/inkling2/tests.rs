//! Parity of the composed trunk with the reference fixture
//! (`tests/fixtures/inkling2_golden`, written by
//! `scripts/fixtures/generate_inkling2_golden.py`).

use std::collections::BTreeMap;

use candle_core::{DType, Device, Tensor};
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

/// The tiny trunk with its projection weights held in `dtype`.
pub(crate) fn tiny_trunk_in(device: &Device, dtype: DType) -> Inkling2Trunk {
    let config = tiny();
    let checkpoint = Inkling2Checkpoint::open(&fixture("inkling2_golden"), &config).unwrap();
    Inkling2Trunk::load_with_dtype(&checkpoint, &config, device, dtype).unwrap()
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

/// The same composed trunk on the Metal device as a load holds it there: F16
/// projection weights under F32 activations, accumulated in F32. Candle's
/// Metal kernels use fast math, so the tolerance is wider than the host's.
#[cfg(feature = "metal")]
#[test]
fn composed_trunk_matches_the_reference_logits_on_metal() {
    let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
        "composed_trunk_matches_the_reference_logits_on_metal",
    ) else {
        return;
    };
    let trunk = tiny_trunk(&device);
    if crate::env::value(super::precision::DENSE_DTYPE_ENV).is_none() {
        assert_eq!(trunk.projection_dtype(), DType::F32);
    }
    assert_eq!(trunk.widened_projections(), 0);
    reference_errors(&trunk, METAL_F16_TOLERANCES.0, METAL_F16_TOLERANCES.1);
}

/// Activation (relative to each activation's scale) and logit tolerances on
/// Metal, the same with the projections in F32 and in F16.
#[cfg(feature = "metal")]
const METAL_F16_TOLERANCES: (f32, f32) = (5e-3, 5e-2);

/// The projection dtypes on Metal against the reference and against one
/// another. F32 is the path before 16-bit weights and F16 is what Metal
/// loads: its weights are the same values and it accumulates in F32, so it
/// stays as close to the reference as F32 does. BF16 is what CUDA loads on
/// request, run here because its code path is the same on any device: it rounds the
/// activations of every projection to BF16 as the reference does, and on
/// this random fixture, whose experts route on near-ties, that moves the
/// logits by whole units. It is measured and only held to be finite.
#[cfg(feature = "metal")]
#[test]
fn projection_dtypes_agree_on_metal() {
    let Some((device, _lease)) =
        crate::backends::metal_test_device_or_skip("projection_dtypes_agree_on_metal")
    else {
        return;
    };
    let golden = golden();
    let mut logits = Vec::new();
    for (dtype, (activation, logit)) in [
        (DType::F32, METAL_F16_TOLERANCES),
        (DType::F16, METAL_F16_TOLERANCES),
        (DType::BF16, (f32::MAX, f32::MAX)),
    ] {
        let trunk = tiny_trunk_in(&device, dtype);
        assert_eq!(trunk.projection_dtype(), dtype);
        assert_eq!(trunk.widened_projections(), 0);
        let (activation_error, logit_error) = reference_errors(&trunk, activation, logit);
        eprintln!(
            "Metal {dtype:?} projections: activation error {activation_error:.3e} of scale, logit error {logit_error:.3e}"
        );
        // One span, then the same tokens a row at a time: the decode step's
        // kernel and the span's product agree.
        let mut state = trunk.new_state().unwrap();
        let hidden = trunk.forward(&golden.tokens, &mut state, None).unwrap();
        let span = values(&trunk.logits(&hidden).unwrap());
        let mut state = trunk.new_state().unwrap();
        let mut stepped = Vec::new();
        for token in &golden.tokens {
            let hidden = trunk.forward(&[*token], &mut state, None).unwrap();
            stepped.extend(values(&trunk.logits(&hidden).unwrap()));
        }
        let step_error = max_error(&stepped, &span);
        eprintln!(
            "Metal {dtype:?} projections: one row at a time against one span {step_error:.3e}"
        );
        if dtype != DType::BF16 {
            assert!(step_error <= METAL_F16_TOLERANCES.1, "{step_error}");
        }
        logits.push(span);
    }
    let f16_error = max_error(&logits[1], &logits[0]);
    let bf16_error = max_error(&logits[2], &logits[0]);
    eprintln!("Metal logits against F32 projections: F16 {f16_error:.3e}, BF16 {bf16_error:.3e}");
    assert!(f16_error <= 5e-3, "F16 {f16_error}");
    assert!(bf16_error.is_finite());
    // F16 weights do not change which token the model picks anywhere in the
    // fixture.
    let vocab = tiny().unpadded_vocab;
    let picks = |logits: &[f32]| {
        logits
            .chunks(vocab)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(token, _)| token)
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(picks(&logits[1]), picks(&logits[0]));
}

fn trunk_matches_the_reference(device: &Device, activation_tolerance: f32, logit_tolerance: f32) {
    reference_errors(&tiny_trunk(device), activation_tolerance, logit_tolerance);
}

/// Runs the fixture through `trunk`, holds every recorded activation and the
/// logits to the tolerances, and returns the worst activation error
/// (relative to its activation's scale) and the logit error.
fn reference_errors(
    trunk: &Inkling2Trunk,
    activation_tolerance: f32,
    logit_tolerance: f32,
) -> (f32, f32) {
    let golden = golden();
    let mut state = trunk.new_state().unwrap();
    let mut trace = Inkling2Trace::default();
    let hidden = trunk
        .forward(&golden.tokens, &mut state, Some(&mut trace))
        .unwrap();
    assert_eq!(state.position(), golden.tokens.len());

    // Every recorded activation, in order, so a mismatch names the first
    // place the trunks diverge.
    assert_eq!(trace.entries.len(), golden.trace.len());
    let mut worst = 0f32;
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
        worst = worst.max(error / scale);
    }
    let logits = values(&trunk.logits(&hidden).unwrap());
    let error = max_error(&logits, &golden.logits);
    eprintln!("{:?}: logit error {error}", trunk.device());
    assert!(error <= logit_tolerance, "logit error {error}");
    // The fixture is not degenerate: logits spread over several units.
    let spread = golden.logits.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(spread > 5.0);
    (worst, error)
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

/// A recorded pass gives the convolution state after any prefix of its
/// span, as a pass over that prefix alone leaves it, and the taps it asks
/// for: the input of an application, and the final output for the id past
/// the last application.
#[test]
fn a_recorded_pass_gives_the_state_after_any_prefix_and_its_taps() {
    use super::trunk::{Inkling2DenseContext, Inkling2PassRecord};
    let golden = golden();
    let device = Device::Cpu;
    let trunk = tiny_trunk(&device);
    let config = tiny();
    let tokens = &golden.tokens[..9];
    let embeddings = trunk.embed(tokens).unwrap();
    let applications = config.application_count();

    let mut conv = trunk.new_conv_state().unwrap();
    let mut context = Inkling2DenseContext::new(applications);
    let mut record = Inkling2PassRecord::with_taps(vec![3, applications]);
    record.keep_conv_inputs = true;
    let hidden = trunk
        .forward_span_recorded(&embeddings, 0, &mut conv, &mut context, None, &mut record)
        .unwrap();
    let taps = record.taps.clone().unwrap();
    assert_eq!(taps.dims(), [9, 2, config.hidden]);
    // The last tap is the pass's output.
    let last = taps.narrow(1, 1, 1).unwrap().squeeze(1).unwrap();
    assert_eq!(values(&last), values(&hidden));

    for consumed in [1, 4, 9] {
        let mut alone = trunk.new_conv_state().unwrap();
        let mut context = Inkling2DenseContext::new(applications);
        trunk
            .forward_span(
                &embeddings.narrow(0, 0, consumed).unwrap(),
                0,
                &mut alone,
                &mut context,
                None,
                None,
            )
            .unwrap();
        let after = record.conv_after(consumed, config.conv_kernel).unwrap();
        for (got, want) in after.histories.iter().zip(&alone.histories) {
            // The same histories up to the summation order of a product
            // over more rows.
            for (got, want) in got.tensors().iter().zip(want.tensors()) {
                let want = values(want);
                let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
                let error = max_error(&values(got), &want);
                assert!(
                    error <= 2e-4 * scale,
                    "after {consumed} tokens: {error} of {scale}"
                );
            }
        }
    }
    // The span's own end state is the state after all of it.
    let end = record.conv_after(9, config.conv_kernel).unwrap();
    for (got, want) in end.histories.iter().zip(&conv.histories) {
        assert_eq!(values(&got.key), values(&want.key));
    }
}
