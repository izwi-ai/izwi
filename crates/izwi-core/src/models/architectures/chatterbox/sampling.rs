//! T3's speech-token sampler, in upstream's order (`t3.py`, `inference`):
//! classifier-free guidance, repetition penalty, temperature, min-p, top-p,
//! then a draw from the request's own stream.
//!
//! Every processor matches its transformers counterpart
//! (`RepetitionPenaltyLogitsProcessor`, `MinPLogitsWarper`,
//! `TopPLogitsWarper`, one token always kept). Greedy decoding is not
//! offered: upstream never decodes greedily, and argmax collapses Chatterbox
//! into silence tokens.

use super::noise::SplitMix64;
use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct T3SamplingParams {
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub min_p: f32,
    pub top_p: f32,
    /// Guidance weight; 0 runs the conditional sequence alone.
    pub cfg_weight: f32,
}

impl T3SamplingParams {
    pub fn validate(&self) -> Result<()> {
        let check = |ok: bool, message: &str| {
            if ok {
                Ok(())
            } else {
                Err(Error::InvalidInput(message.to_string()))
            }
        };
        check(
            self.temperature.is_finite() && self.temperature > 0.0,
            "Chatterbox temperature must be positive",
        )?;
        check(
            self.repetition_penalty.is_finite() && self.repetition_penalty > 0.0,
            "Chatterbox repetition penalty must be positive",
        )?;
        check(
            (0.0..=1.0).contains(&self.min_p),
            "Chatterbox min_p must be within [0, 1]",
        )?;
        check(
            self.top_p > 0.0 && self.top_p <= 1.0,
            "Chatterbox top_p must be within (0, 1]",
        )?;
        check(
            self.cfg_weight.is_finite() && self.cfg_weight >= 0.0,
            "Chatterbox cfg_weight must be non-negative",
        )
    }

    pub fn guided(&self) -> bool {
        self.cfg_weight > 0.0
    }
}

/// Guided, penalised and filtered logits for one step, before sampling.
/// `history` is every speech token the sequence holds so far, the start
/// token included.
pub fn process_logits(
    conditional: &[f32],
    unconditional: Option<&[f32]>,
    history: &[u32],
    params: &T3SamplingParams,
) -> Result<Vec<f32>> {
    let mut logits = match unconditional {
        Some(unconditional) => {
            if unconditional.len() != conditional.len() {
                return Err(Error::InferenceError(
                    "Chatterbox guidance rows differ in width".into(),
                ));
            }
            conditional
                .iter()
                .zip(unconditional)
                .map(|(&c, &u)| c + params.cfg_weight * (c - u))
                .collect::<Vec<_>>()
        }
        None => conditional.to_vec(),
    };
    if logits.iter().any(|value| value.is_nan()) {
        return Err(Error::InferenceError(
            "Chatterbox T3 produced NaN logits".into(),
        ));
    }

    let mut seen = vec![false; logits.len()];
    for &token in history {
        let Some(slot) = seen.get_mut(token as usize) else {
            continue;
        };
        if std::mem::replace(slot, true) {
            continue;
        }
        let score = &mut logits[token as usize];
        *score = if *score < 0.0 {
            *score * params.repetition_penalty
        } else {
            *score / params.repetition_penalty
        };
    }

    if params.temperature != 1.0 {
        for score in &mut logits {
            *score /= params.temperature;
        }
    }

    let top = argmax(&logits);
    if params.min_p > 0.0 {
        let probs = softmax(&logits);
        let threshold = f64::from(params.min_p) * probs[top];
        for (index, score) in logits.iter_mut().enumerate() {
            if index != top && probs[index] < threshold {
                *score = f32::NEG_INFINITY;
            }
        }
    }

    if params.top_p < 1.0 {
        let probs = softmax(&logits);
        let mut ascending = (0..logits.len()).collect::<Vec<_>>();
        ascending.sort_by(|&a, &b| logits[a].total_cmp(&logits[b]).then(b.cmp(&a)));
        let floor = 1.0 - f64::from(params.top_p);
        let mut cumulative = 0.0f64;
        for &index in &ascending[..ascending.len() - 1] {
            cumulative += probs[index];
            if cumulative <= floor {
                logits[index] = f32::NEG_INFINITY;
            } else {
                break;
            }
        }
    }
    Ok(logits)
}

/// Draw a token from processed logits with one uniform of `rng`.
pub fn sample_processed(logits: &[f32], rng: &mut SplitMix64) -> Result<u32> {
    let probs = softmax(logits);
    let total = probs.iter().sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err(Error::InferenceError(
            "Chatterbox T3 sampling distribution is empty".into(),
        ));
    }
    let target = rng.next_open01() * total;
    let mut cumulative = 0.0f64;
    let mut last_positive = None;
    for (index, &p) in probs.iter().enumerate() {
        if p <= 0.0 {
            continue;
        }
        cumulative += p;
        last_positive = Some(index);
        if target < cumulative {
            return Ok(index as u32);
        }
    }
    last_positive
        .map(|index| index as u32)
        .ok_or_else(|| Error::InferenceError("Chatterbox T3 sampled nothing".into()))
}

pub fn sample_next(
    conditional: &[f32],
    unconditional: Option<&[f32]>,
    history: &[u32],
    params: &T3SamplingParams,
    rng: &mut SplitMix64,
) -> Result<u32> {
    let logits = process_logits(conditional, unconditional, history, params)?;
    sample_processed(&logits, rng)
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best
}

/// Softmax in F64; `-inf` entries get probability zero.
fn softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![0.0; logits.len()];
    }
    let exps = logits
        .iter()
        .map(|&value| {
            if value == f32::NEG_INFINITY {
                0.0
            } else {
                (f64::from(value) - f64::from(max)).exp()
            }
        })
        .collect::<Vec<_>>();
    let sum = exps.iter().sum::<f64>();
    exps.into_iter().map(|value| value / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> T3SamplingParams {
        T3SamplingParams {
            temperature: 0.8,
            repetition_penalty: 1.2,
            min_p: 0.05,
            top_p: 1.0,
            cfg_weight: 0.5,
        }
    }

    #[test]
    fn guidance_penalty_and_temperature_follow_transformers() {
        let processed = process_logits(
            &[2.0, -1.0, 0.5, 1.0],
            Some(&[1.0, -2.0, 0.5, 2.0]),
            &[1, 1, 3],
            &T3SamplingParams {
                min_p: 0.0,
                ..params()
            },
        )
        .unwrap();
        // cond + 0.5 (cond - uncond) = [2.5, -0.5, 0.5, 0.5]; tokens 1 and 3
        // penalised once each (negative ×1.2, positive ÷1.2); then ÷0.8.
        let expected = [2.5 / 0.8, -0.5 * 1.2 / 0.8, 0.5 / 0.8, 0.5 / 1.2 / 0.8];
        for (actual, expected) in processed.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{processed:?}");
        }
    }

    #[test]
    fn min_p_drops_tokens_below_its_share_of_the_top_probability() {
        let processed = process_logits(
            &[5.0, 0.0, 4.0, -10.0],
            None,
            &[],
            &T3SamplingParams {
                temperature: 1.0,
                repetition_penalty: 1.0,
                min_p: 0.05,
                top_p: 1.0,
                cfg_weight: 0.0,
            },
        )
        .unwrap();
        assert_eq!(processed[0], 5.0);
        assert_eq!(processed[2], 4.0);
        assert_eq!(processed[1], f32::NEG_INFINITY);
        assert_eq!(processed[3], f32::NEG_INFINITY);
    }

    #[test]
    fn top_p_keeps_the_smallest_head_covering_the_mass() {
        let processed = process_logits(
            &[3.0, 2.0, 1.0, 0.0],
            None,
            &[],
            &T3SamplingParams {
                temperature: 1.0,
                repetition_penalty: 1.0,
                min_p: 0.0,
                top_p: 0.7,
                cfg_weight: 0.0,
            },
        )
        .unwrap();
        // Probabilities ≈ [0.644, 0.237, 0.087, 0.032]: the top token alone
        // falls short of 0.7, so the second stays and the tail goes.
        assert!(processed[0].is_finite() && processed[1].is_finite());
        assert_eq!(processed[2], f32::NEG_INFINITY);
        assert_eq!(processed[3], f32::NEG_INFINITY);
    }

    #[test]
    fn sampling_is_repeatable_per_seed_and_respects_filters() {
        let logits = [1.0, f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY];
        let draw = |seed| {
            let mut rng = SplitMix64::new(seed);
            (0..64)
                .map(|_| sample_processed(&logits, &mut rng).unwrap())
                .collect::<Vec<_>>()
        };
        assert_eq!(draw(3), draw(3));
        assert!(draw(3).iter().all(|&token| token == 0 || token == 2));
        assert!(draw(3).contains(&0) && draw(3).contains(&2));
    }

    #[test]
    fn invalid_parameters_and_nan_logits_are_refused() {
        assert!(T3SamplingParams {
            temperature: 0.0,
            ..params()
        }
        .validate()
        .is_err());
        assert!(T3SamplingParams {
            top_p: 0.0,
            ..params()
        }
        .validate()
        .is_err());
        assert!(params().validate().is_ok());
        assert!(process_logits(&[f32::NAN, 1.0], None, &[], &params()).is_err());
    }
}
