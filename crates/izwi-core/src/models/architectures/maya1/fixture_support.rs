//! Test helpers shared by the Maya1 fixture tests: the fixture files and the
//! strided sample points `scripts/fixtures/generate_maya1_reference.py`
//! records.

use std::path::PathBuf;

use serde::de::DeserializeOwned;

#[derive(serde::Deserialize)]
pub(super) struct Sample {
    pub(super) shape: Vec<usize>,
    pub(super) indices: Vec<usize>,
    pub(super) values: Vec<f32>,
    pub(super) rms: f32,
}

/// A fixture file of `tests/fixtures/maya1`.
pub(super) fn fixture<T: DeserializeOwned>(name: &str) -> T {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/maya1")
        .join(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

/// Max |izwi − upstream| at the sample points over the upstream RMS; a
/// non-finite difference counts as infinite.
pub(super) fn sample_error(name: &str, actual: &[f32], shape: &[usize], sample: &Sample) -> f32 {
    assert_eq!(shape, sample.shape.as_slice(), "{name} shape");
    let error = sample
        .indices
        .iter()
        .zip(&sample.values)
        .map(|(&index, &expected)| {
            let diff = (actual[index] - expected).abs();
            if diff.is_finite() {
                diff
            } else {
                f32::INFINITY
            }
        })
        .fold(0f32, |worst, diff| if diff > worst { diff } else { worst });
    error / sample.rms.max(f32::EPSILON)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_error_treats_nan_as_infinite() {
        let sample = Sample {
            shape: vec![2],
            indices: vec![0, 1],
            values: vec![1.0, 1.0],
            rms: 1.0,
        };
        assert_eq!(sample_error("x", &[1.0, 1.0], &[2], &sample), 0.0);
        assert_eq!(
            sample_error("x", &[f32::NAN, 1.0], &[2], &sample),
            f32::INFINITY
        );
    }
}
