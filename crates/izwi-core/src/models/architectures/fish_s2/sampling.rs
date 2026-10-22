//! Fish Speech sampling policy. Keep this separate from chat-model sampling:
//! upstream filters cumulative probabilities before temperature, and excludes
//! the token that crosses top-p (except that rank zero always survives).

use candle_core::{DType, IndexOp, Tensor};
// ChaCha12 is what StdRng runs, but it stays Clone (samplers snapshot their
// RNG) and its seeded stream is stable across releases.
use rand_chacha::ChaCha12Rng;
use rand::{RngExt, SeedableRng};

use crate::error::{Error, Result};
use crate::models::shared::sampling::{
    bounded_device_sampling_candidates, device_top_k, gumbel_noise, DEVICE_SORT_CHUNK,
};
use crate::models::shared::telemetry;

#[derive(Debug, Clone)]
pub struct FishS2Sampler {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    rng: ChaCha12Rng,
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
            rng: ChaCha12Rng::seed_from_u64(seed),
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
                let _ = self.rng.random::<f32>();
            }
            return Ok(probabilities[0].0);
        }
        let mut draw = self.rng.random::<f32>();
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

/// Per-frame inputs for device-resident sampling of `tokens` codes from one
/// sampler: Gumbel noise rows drawn from the sampler's RNG (none when
/// greedy) and the upper-triangular ones matrix the top-p cumulative sum
/// multiplies by. Both are uploaded once per frame.
pub(crate) struct FishS2ResidentDraws {
    noise: Option<Tensor>,
    upper: Tensor,
    top_k: usize,
}

impl FishS2Sampler {
    /// `None` when this policy needs the host path: top-k 0 (whole-row
    /// top-p support) or above the device sort bound.
    pub(crate) fn resident_draws(
        &mut self,
        tokens: usize,
        device: &candle_core::Device,
    ) -> Result<Option<FishS2ResidentDraws>> {
        validate_policy(self.temperature, self.top_p)?;
        if self.top_k == 0 || self.top_k > DEVICE_SORT_CHUNK {
            return Ok(None);
        }
        let noise = if self.temperature > 0.0 {
            let rng = &mut self.rng;
            let mut draw = || rng.random::<f32>();
            let values = gumbel_noise(tokens * self.top_k, &mut draw);
            Some(Tensor::from_vec(values, (tokens, self.top_k), device)?)
        } else {
            None
        };
        Ok(Some(FishS2ResidentDraws {
            noise,
            upper: Tensor::triu2(self.top_k, DType::F32, device)?,
            top_k: self.top_k,
        }))
    }
}

/// Fish's policy on the logits' device (upstream `logits_to_probs`): top-p
/// over the raw softmax of the whole row, dropping candidates whose
/// cumulative probability exceeds `top_p` except rank 0, then top-k, then
/// temperature. Returns the tempered logits of the `top_k` highest
/// candidates, with removed ones pushed to -1e30, and their token ids.
pub(crate) fn resident_filtered_candidates(
    row: &Tensor,
    temperature: f32,
    top_p: f32,
    upper: &Tensor,
    top_k: usize,
) -> Result<Option<(Tensor, Tensor)>> {
    let row = row.to_dtype(DType::F32)?;
    let Some((values, indices)) = device_top_k(&row, top_k)? else {
        return Ok(None);
    };
    let k = values.dim(0)?;
    let max = values.narrow(0, 0, 1)?;
    let log_sum_exp = row
        .broadcast_sub(&max)?
        .exp()?
        .sum_keepdim(0)?
        .log()?
        .broadcast_add(&max)?;
    let raw = values.broadcast_sub(&log_sum_exp)?.exp()?;
    let cumulative = raw
        .unsqueeze(0)?
        .matmul(&upper.narrow(0, 0, k)?.narrow(1, 0, k)?)?
        .squeeze(0)?;
    // Rank 0 has nothing before it; every later rank has a positive prefix.
    let keep = cumulative
        .le(top_p as f64)?
        .maximum(&(&cumulative - &raw)?.le(0.0)?)?
        .to_dtype(DType::F32)?;
    let tempered = (values.broadcast_sub(&max)? / f64::from(temperature.max(1e-5)))?;
    let masked = (tempered + keep.affine(1e30, -1e30)?)?;
    Ok(Some((masked, indices)))
}

/// One device-resident Fish draw for `row`: the token as a `[1]` U32 tensor
/// and the row's count of non-finite logits as a `[1]` U32 tensor, both left
/// on the device for the caller to read back once per frame. `noise_row` is
/// this token's Gumbel noise (`None` when greedy).
pub(crate) fn sample_resident(
    row: &Tensor,
    temperature: f32,
    top_p: f32,
    draws: &FishS2ResidentDraws,
    token: usize,
) -> Result<Option<(Tensor, Tensor)>> {
    let vocab = row.dims1()?;
    if vocab == 0 {
        return Err(Error::InferenceError(
            "Fish S2 fast sampler received empty logits".into(),
        ));
    }
    let row_f32 = row.to_dtype(DType::F32)?;
    let non_finite = row_f32
        .ne(&row_f32)?
        .maximum(&row_f32.abs()?.gt(f32::MAX as f64)?)?
        .to_dtype(DType::U32)?
        .sum_keepdim(0)?;
    let Some((masked, indices)) =
        resident_filtered_candidates(&row_f32, temperature, top_p, &draws.upper, draws.top_k)?
    else {
        return Ok(None);
    };
    let indices = indices.to_dtype(DType::U32)?;
    let token = match &draws.noise {
        // Greedy: the highest logit, rank 0 of the sorted candidates.
        None => indices.narrow(0, 0, 1)?,
        Some(noise) => {
            let k = masked.dim(0)?;
            let pick = masked
                .broadcast_add(&noise.i(token)?.narrow(0, 0, k)?)?
                .argmax_keepdim(0)?
                .to_dtype(DType::U32)?;
            indices.index_select(&pick, 0)?
        }
    };
    Ok(Some((token, non_finite)))
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

    /// FSH-4: the device filter keeps exactly the host policy's support
    /// (top-p over the whole row's raw softmax, rank 0 always, then top-k)
    /// with the same tempered probabilities.
    #[test]
    fn resident_filter_matches_the_host_policy() {
        use crate::kernels::numeric_check::xorshift_values;

        let device = Device::Cpu;
        let values = xorshift_values(64, 0xF15F, 3.0);
        let row = Tensor::from_vec(values.clone(), 64, &device).unwrap();
        for (temperature, top_p, top_k) in [
            (0.7f32, 0.9f32, 30usize),
            (1.3, 0.5, 8),
            (0.8, 1.0, 12),
            (0.5, 0.01, 30),
        ] {
            let host = FishS2SamplingDistribution::from_values(&values, None)
                .unwrap()
                .probabilities(temperature, top_p, top_k)
                .unwrap();
            let upper = Tensor::triu2(top_k, DType::F32, &device).unwrap();
            let (masked, indices) =
                resident_filtered_candidates(&row, temperature, top_p, &upper, top_k)
                    .unwrap()
                    .unwrap();
            let probabilities = candle_nn::ops::softmax_last_dim(&masked)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let indices = indices.to_vec1::<u32>().unwrap();
            let device_support = indices
                .iter()
                .zip(&probabilities)
                .filter(|(_, p)| **p > 0.0)
                .map(|(token, p)| (*token, *p))
                .collect::<Vec<_>>();
            assert_eq!(
                device_support.len(),
                host.len(),
                "policy {temperature}/{top_p}/{top_k}"
            );
            for ((token, p), (host_token, host_p)) in device_support.iter().zip(&host) {
                assert_eq!(token, host_token);
                assert!((p - host_p).abs() <= 1e-5, "{token}: {p} vs {host_p}");
            }
        }
    }

    /// FSH-4: the production device path (Metal) keeps the CPU filter's
    /// support and probabilities, and greedy draws agree.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_resident_filter_matches_cpu() {
        use crate::kernels::numeric_check::xorshift_values;

        let Some((metal, _lease)) =
            crate::backends::metal_test_device_or_skip("metal_resident_filter_matches_cpu")
        else {
            return;
        };
        let values = xorshift_values(4096, 0x5EED, 4.0);
        let run = |device: &Device| {
            let row = Tensor::from_vec(values.clone(), 4096, device).unwrap();
            let upper = Tensor::triu2(30, DType::F32, device).unwrap();
            let (masked, indices) = resident_filtered_candidates(&row, 0.7, 0.9, &upper, 30)
                .unwrap()
                .unwrap();
            let probabilities = candle_nn::ops::softmax_last_dim(&masked)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let mut greedy = FishS2Sampler::with_top_k(0.0, 0.9, 30, 1);
            let draws = greedy.resident_draws(1, device).unwrap().unwrap();
            let (code, non_finite) = sample_resident(&row, 0.0, 0.9, &draws, 0).unwrap().unwrap();
            (
                indices.to_vec1::<u32>().unwrap(),
                probabilities,
                code.to_vec1::<u32>().unwrap(),
                non_finite.to_vec1::<u32>().unwrap(),
            )
        };
        let (cpu_indices, cpu_p, cpu_code, _) = run(&Device::Cpu);
        let (metal_indices, metal_p, metal_code, metal_bad) = run(&metal);
        assert_eq!(metal_indices, cpu_indices);
        assert_eq!(metal_code, cpu_code);
        assert_eq!(metal_bad, vec![0]);
        for (m, c) in metal_p.iter().zip(&cpu_p) {
            assert!((m - c).abs() <= 1e-5, "{m} vs {c}");
        }
    }

    /// FSH-4: Gumbel-max over the filtered candidates draws tokens at the
    /// host policy's probabilities, and greedy picks the top logit.
    #[test]
    fn resident_draws_follow_the_host_distribution() {
        let device = Device::Cpu;
        let values = vec![2.0f32, 1.5, 1.0, 0.2, -1.0, -3.0];
        let row = Tensor::from_vec(values.clone(), 6, &device).unwrap();
        let host = FishS2SamplingDistribution::from_values(&values, None)
            .unwrap()
            .probabilities(0.8, 0.95, 4)
            .unwrap();
        let mut sampler = FishS2Sampler::with_top_k(0.8, 0.95, 4, 99);
        let draws_per_batch = 500;
        let mut counts = [0usize; 6];
        for _ in 0..8 {
            let draws = sampler
                .resident_draws(draws_per_batch, &device)
                .unwrap()
                .unwrap();
            for token in 0..draws_per_batch {
                let (code, non_finite) = sample_resident(&row, 0.8, 0.95, &draws, token)
                    .unwrap()
                    .unwrap();
                assert_eq!(non_finite.to_vec1::<u32>().unwrap(), vec![0]);
                counts[code.to_vec1::<u32>().unwrap()[0] as usize] += 1;
            }
        }
        let total = (8 * draws_per_batch) as f32;
        for (token, p) in &host {
            let observed = counts[*token as usize] as f32 / total;
            assert!((observed - p).abs() < 0.03, "{token}: {observed} vs {p}");
        }
        let outside = (0..6u32)
            .filter(|token| host.iter().all(|(kept, _)| kept != token))
            .map(|token| counts[token as usize])
            .sum::<usize>();
        assert_eq!(outside, 0, "only the host support can be drawn");

        let mut greedy = FishS2Sampler::with_top_k(0.0, 0.9, 4, 1);
        let draws = greedy.resident_draws(1, &device).unwrap().unwrap();
        let (code, _) = sample_resident(&row, 0.0, 0.9, &draws, 0).unwrap().unwrap();
        assert_eq!(code.to_vec1::<u32>().unwrap(), vec![0]);
        assert!(FishS2Sampler::with_top_k(0.7, 0.9, 0, 1)
            .resident_draws(1, &device)
            .unwrap()
            .is_none());
    }

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
        let (device, _lease) = crate::backends::metal_test_device_or_skip(
            "metal_sampling_matches_cpu_policy_and_rejects_raw_nan",
        )
        .expect("Metal device");
        check_device_distribution(&device);
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
        let (device, _lease) =
            crate::backends::metal_test_device_or_skip("fish_s2_sampling_metal_matches_host")
                .expect("Metal device required");
        assert_accelerator_matches_host(device);
    }

    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "requires an available CUDA device; never falls back to CPU"]
    fn fish_s2_sampling_cuda_matches_host() {
        assert_accelerator_matches_host(Device::new_cuda(0).expect("CUDA device required"));
    }
}
