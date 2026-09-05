//! Fish Speech sampling policy. Keep this separate from chat-model sampling:
//! upstream filters cumulative probabilities before temperature, and excludes
//! the token that crosses top-p (except that rank zero always survives).

use candle_core::{DType, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::error::{Error, Result};
use crate::models::shared::sampling::bounded_device_sampling_candidates;
use crate::models::shared::telemetry;

#[derive(Debug, Clone)]
pub struct FishS2Sampler {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    rng: StdRng,
}

impl FishS2Sampler {
    pub fn new(temperature: f32, top_p: f32, seed: u64) -> Self {
        Self::with_top_k(temperature, top_p, 30, seed)
    }

    pub fn with_top_k(temperature: f32, top_p: f32, top_k: usize, seed: u64) -> Self {
        Self {
            temperature,
            top_p,
            top_k,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub(crate) fn sample(&mut self, distribution: &FishS2SamplingDistribution) -> Result<u32> {
        self.sample_with_policy(distribution, self.temperature, self.top_p)
    }

    pub(crate) fn sample_with_policy(
        &mut self,
        distribution: &FishS2SamplingDistribution,
        temperature: f32,
        top_p: f32,
    ) -> Result<u32> {
        validate_policy(temperature, top_p)?;
        let probabilities = distribution.probabilities(temperature, top_p, self.top_k)?;
        if probabilities.len() == 1 {
            // Greedy mode has no RNG side effects. Stochastic singleton draws
            // still consume a value, keeping replay stable across support sizes.
            if temperature > 0.0 {
                let _ = self.rng.r#gen::<f32>();
            }
            return Ok(probabilities[0].0);
        }
        let mut draw = self.rng.r#gen::<f32>();
        for (token, probability) in &probabilities {
            if draw < *probability {
                return Ok(*token);
            }
            draw -= probability;
        }
        // Only floating-point summation residue can reach this branch.
        Ok(probabilities.last().unwrap().0)
    }
}

pub(crate) fn validate_policy(temperature: f32, top_p: f32) -> Result<()> {
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(Error::InvalidInput(
            "Fish S2 temperature must be finite and non-negative".into(),
        ));
    }
    if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
        return Err(Error::InvalidInput(
            "Fish S2 top_p must be finite and in (0, 1]".into(),
        ));
    }
    Ok(())
}

pub(crate) struct FishS2SamplingDistribution {
    /// Descending raw logits, with global vocabulary indices preserved.
    values: Vec<(u32, f32)>,
    /// Probabilities of these candidates under the entire allowed raw row.
    raw_probabilities: Vec<f32>,
}

impl FishS2SamplingDistribution {
    pub(crate) fn from_logits(
        row: &Tensor,
        allowed_mask: Option<&[bool]>,
        top_k: usize,
        required_top_p: f32,
        label: &str,
    ) -> Result<Self> {
        let vocab = row.dims1()?;
        if vocab == 0 {
            return Err(Error::InferenceError(format!(
                "Fish S2 {label} sampler received empty logits"
            )));
        }
        if vocab > u32::MAX as usize {
            return Err(Error::InferenceError(format!(
                "Fish S2 {label} vocabulary does not fit token IDs"
            )));
        }
        if allowed_mask.is_some_and(|mask| mask.len() != vocab) {
            return Err(Error::InferenceError(format!(
                "Fish S2 {label} mask length does not match logits length {vocab}"
            )));
        }
        let allowed_count = allowed_mask
            .map(|mask| mask.iter().filter(|allowed| **allowed).count())
            .unwrap_or(vocab);
        if allowed_count == 0 {
            return Err(Error::InferenceError(format!(
                "Fish S2 {label} sampler has no allowed tokens"
            )));
        }

        let row_f32 = row.to_dtype(DType::F32)?;
        if row.device().is_cuda() || row.device().is_metal() {
            // Diagnose the raw head before any semantic mask or the shared
            // candidate helper can hide non-finite values. Read one scalar in
            // the healthy path; only failure reads the row for precise counts.
            let finite = row_f32
                .eq(&row_f32)?
                .broadcast_mul(&row_f32.abs()?.le(f32::MAX as f64)?)?
                .to_dtype(DType::U32)?
                .sum_all()?;
            telemetry::record_host_read(DType::U32, 1);
            if finite.to_scalar::<u32>()? as usize != vocab {
                let values = read_row(&row_f32)?;
                return Err(non_finite_error(&values, row, label));
            }

            // top_k=0 requests the existing Candle selection plus the full-row
            // log-sum-exp. Applying top-k inside that helper would normalize
            // over a truncated row and change Fish's top-p support.
            if let Some(candidates) = bounded_device_sampling_candidates(
                &row_f32,
                vocab,
                0,
                1.0,
                &[],
                1.0,
                0.0,
                allowed_mask,
            )? {
                if let Some(logsumexp) = candidates.logsumexp {
                    let raw_probabilities = candidates
                        .values
                        .iter()
                        .map(|value| (*value - logsumexp).exp())
                        .collect::<Vec<_>>();
                    let cumulative = raw_probabilities.iter().sum::<f32>();
                    let covers_top_k = top_k > 0 && candidates.values.len() >= top_k;
                    let covers_all = candidates.values.len() >= allowed_count;
                    // A strict excess is needed: a token at exactly top-p is
                    // retained, so equality alone cannot prove completeness.
                    let covers_top_p = cumulative > required_top_p;
                    if (covers_top_k || covers_all || covers_top_p)
                        && raw_probabilities.iter().all(|value| value.is_finite())
                    {
                        return Ok(Self {
                            values: candidates
                                .indices
                                .into_iter()
                                .zip(candidates.values)
                                .collect(),
                            raw_probabilities,
                        });
                    }
                }
            }
        }

        let values = read_row(&row_f32)?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(non_finite_error(&values, row, label));
        }
        Self::from_values(&values, allowed_mask)
    }

    fn from_values(values: &[f32], allowed_mask: Option<&[bool]>) -> Result<Self> {
        let mut values = values
            .iter()
            .enumerate()
            .filter(|(index, _)| allowed_mask.is_none_or(|mask| mask[*index]))
            .map(|(index, value)| (index as u32, *value))
            .collect::<Vec<_>>();
        values.sort_by(|left, right| right.1.total_cmp(&left.1));
        let max = values
            .first()
            .ok_or_else(|| Error::InferenceError("Fish S2 sampler has no allowed tokens".into()))?
            .1;
        let mut raw_probabilities = values
            .iter()
            .map(|(_, value)| (*value - max).exp())
            .collect::<Vec<_>>();
        let total = raw_probabilities.iter().sum::<f32>();
        for probability in &mut raw_probabilities {
            *probability /= total;
        }
        Ok(Self {
            values,
            raw_probabilities,
        })
    }

    fn probabilities(&self, temperature: f32, top_p: f32, top_k: usize) -> Result<Vec<(u32, f32)>> {
        validate_policy(temperature, top_p)?;
        if temperature == 0.0 {
            return Ok(vec![(self.values[0].0, 1.0)]);
        }
        let limit = if top_k == 0 { usize::MAX } else { top_k };
        let max = self.values[0].1;
        let temperature = temperature.max(1e-5);
        let mut cumulative = 0.0f32;
        let mut kept = Vec::new();
        for (rank, ((token, value), raw_probability)) in
            self.values.iter().zip(&self.raw_probabilities).enumerate()
        {
            cumulative += raw_probability;
            if rank > 0 && (rank >= limit || cumulative > top_p) {
                break;
            }
            // Subtract before dividing to avoid overflow for large finite raw
            // logits and small temperatures while preserving softmax ratios.
            kept.push((*token, ((*value - max) / temperature).exp()));
        }
        let total = kept.iter().map(|(_, value)| value).sum::<f32>();
        if !total.is_finite() || total <= 0.0 {
            return Err(Error::InferenceError(
                "Fish S2 filtered sampling probabilities are invalid".into(),
            ));
        }
        for (_, probability) in &mut kept {
            *probability /= total;
        }
        Ok(kept)
    }
}

fn read_row(row: &Tensor) -> Result<Vec<f32>> {
    telemetry::record_host_read(DType::F32, row.elem_count());
    row.to_vec1::<f32>().map_err(Error::from)
}

fn non_finite_error(values: &[f32], row: &Tensor, label: &str) -> Error {
    let nan = values.iter().filter(|value| value.is_nan()).count();
    let positive_infinity = values
        .iter()
        .filter(|value| **value == f32::INFINITY)
        .count();
    let negative_infinity = values
        .iter()
        .filter(|value| **value == f32::NEG_INFINITY)
        .count();
    let backend = if row.device().is_cuda() {
        "cuda"
    } else if row.device().is_metal() {
        "metal"
    } else {
        "cpu"
    };
    Error::InferenceError(format!(
        "Fish S2 {label} raw logits are non-finite before masking: backend={backend}, dtype={:?}, shape={:?}, nan={nan}, positive_infinity={positive_infinity}, negative_infinity={negative_infinity}",
        row.dtype(), row.dims()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn distribution(probabilities: &[f32]) -> FishS2SamplingDistribution {
        let logits = probabilities
            .iter()
            .map(|value| value.ln())
            .collect::<Vec<_>>();
        FishS2SamplingDistribution::from_values(&logits, None).unwrap()
    }

    #[cfg(any(feature = "metal", feature = "cuda"))]
    fn check_device_distribution(device: &candle_core::Device) {
        let values: Vec<f32> = (0..4097)
            .map(|index| (index as f32 * 0.371).sin() * 3.0)
            .collect();
        let cpu = Tensor::new(values.as_slice(), &candle_core::Device::Cpu).unwrap();
        let accelerator = Tensor::new(values.as_slice(), device).unwrap();
        let allowed = vec![true; values.len()];
        let expected =
            FishS2SamplingDistribution::from_logits(&cpu, Some(&allowed), 30, 0.9, "oracle")
                .unwrap()
                .probabilities(0.8, 0.8, 30)
                .unwrap();
        let actual = FishS2SamplingDistribution::from_logits(
            &accelerator,
            Some(&allowed),
            30,
            0.9,
            "device",
        )
        .unwrap()
        .probabilities(0.8, 0.8, 30)
        .unwrap();
        assert_eq!(actual.len(), expected.len());
        for ((token, probability), (expected_token, expected_probability)) in
            actual.iter().zip(expected)
        {
            assert_eq!(*token, expected_token);
            assert!((probability - expected_probability).abs() < 2e-5);
        }
        let bad = Tensor::new(&[f32::NAN, 1.0f32], device).unwrap();
        assert!(FishS2SamplingDistribution::from_logits(
            &bad,
            Some(&[false, true]),
            30,
            1.0,
            "raw"
        )
        .is_err());
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires an available Metal device; never falls back to CPU"]
    fn metal_sampling_matches_cpu_policy_and_rejects_raw_nan() {
        check_device_distribution(&candle_core::Device::new_metal(0).expect("Metal device"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires an available CUDA device; never falls back to CPU"]
    fn cuda_sampling_matches_cpu_policy_and_rejects_raw_nan() {
        check_device_distribution(&candle_core::Device::new_cuda(0).expect("CUDA device"));
    }

    #[test]
    fn upstream_nucleus_excludes_crossing_token_before_temperature() {
        let values = distribution(&[0.45, 0.30, 0.15, 0.10]);
        let probabilities = values.probabilities(0.5, 0.8, 30).unwrap();
        assert_eq!(
            probabilities.iter().map(|item| item.0).collect::<Vec<_>>(),
            [0, 1]
        );
        // Frozen probability ratio from upstream logits_to_probs: .45^2/.30^2.
        assert!((probabilities[0].1 - 0.6923077).abs() < 1e-6);
        assert!((probabilities[1].1 - 0.3076923).abs() < 1e-6);
        let sharp = distribution(&[0.7, 0.2, 0.1]);
        assert_eq!(sharp.probabilities(0.8, 0.8, 30).unwrap(), [(0, 1.0)]);
    }

    #[test]
    fn upstream_keeps_exact_boundary_and_top_one_intersects_top_k() {
        let values = FishS2SamplingDistribution::from_values(&[0.0; 4], None).unwrap();
        assert_eq!(
            values.probabilities(0.8, 0.5, 30).unwrap(),
            [(0, 0.5), (1, 0.5)]
        );
        assert_eq!(values.probabilities(0.8, 0.1, 30).unwrap(), [(0, 1.0)]);
        assert_eq!(
            values.probabilities(0.8, 1.0, 2).unwrap(),
            [(0, 0.5), (1, 0.5)]
        );
        assert_eq!(values.probabilities(0.8, 1.0, 0).unwrap().len(), 4);
    }

    #[test]
    fn raw_logits_fail_even_outside_mask_and_in_greedy_mode() {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let row = Tensor::new(&[invalid, 2.0, 1.0], &Device::Cpu).unwrap();
            let error = FishS2SamplingDistribution::from_logits(
                &row,
                Some(&[false, true, true]),
                30,
                0.8,
                "semantic",
            )
            .err()
            .unwrap()
            .to_string();
            assert!(
                error.contains("raw logits are non-finite before masking"),
                "{error}"
            );
            assert!(error.contains("backend=cpu, dtype=F32"), "{error}");
        }
    }

    #[test]
    fn seeded_sampler_clone_replays_and_zero_temperature_is_greedy() {
        let values = FishS2SamplingDistribution::from_values(&[0.0, -0.1, -0.2], None).unwrap();
        let mut sampler = FishS2Sampler::with_top_k(0.8, 1.0, 0, 92);
        let mut checkpoint = sampler.clone();
        for _ in 0..30 {
            assert_eq!(
                sampler.sample(&values).unwrap(),
                checkpoint.sample(&values).unwrap()
            );
        }
        let mut greedy = FishS2Sampler::new(0.0, 0.8, 92);
        for _ in 0..10 {
            assert_eq!(greedy.sample(&values).unwrap(), 0);
        }
    }

    #[test]
    fn rejects_nonfinite_and_out_of_range_sampling_controls() {
        for temperature in [-1.0, f32::NAN, f32::INFINITY] {
            assert!(validate_policy(temperature, 0.8).is_err());
        }
        for top_p in [0.0, -0.1, 1.1, f32::NAN, f32::INFINITY] {
            assert!(validate_policy(0.8, top_p).is_err());
        }
    }

    #[cfg(any(feature = "metal", feature = "cuda"))]
    fn assert_accelerator_matches_host(device: Device) {
        let values = (0..512)
            .map(|index| (index as f32 * 0.17).sin() * 4.0)
            .collect::<Vec<_>>();
        let row = Tensor::new(values.as_slice(), &device).unwrap();
        let mask = (0..512).map(|index| index % 7 != 0).collect::<Vec<_>>();
        let accelerator =
            FishS2SamplingDistribution::from_logits(&row, Some(&mask), 30, 0.9, "test").unwrap();
        let host = FishS2SamplingDistribution::from_values(&values, Some(&mask)).unwrap();
        for top_p in [0.8, 0.9] {
            let actual = accelerator.probabilities(0.8, top_p, 30).unwrap();
            let expected = host.probabilities(0.8, top_p, 30).unwrap();
            assert_eq!(
                actual.iter().map(|item| item.0).collect::<Vec<_>>(),
                expected.iter().map(|item| item.0).collect::<Vec<_>>()
            );
            for (left, right) in actual.iter().zip(expected) {
                assert!((left.1 - right.1).abs() < 1e-5);
            }
        }
        let invalid = Tensor::new(&[f32::NAN, 1.0, 2.0], &device).unwrap();
        assert!(FishS2SamplingDistribution::from_logits(
            &invalid,
            Some(&[false, true, true]),
            30,
            0.8,
            "test"
        )
        .is_err());
    }

    #[test]
    #[cfg(feature = "metal")]
    #[ignore = "requires an available Metal device; never falls back to CPU"]
    fn fish_s2_sampling_metal_matches_host() {
        assert_accelerator_matches_host(Device::new_metal(0).expect("Metal device required"));
    }

    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "requires an available CUDA device; never falls back to CPU"]
    fn fish_s2_sampling_cuda_matches_host() {
        assert_accelerator_matches_host(Device::new_cuda(0).expect("CUDA device required"));
    }
}
