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
use crate::kernels::penalized_topk::{CandidatePenalty, CandidateRow};

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

/// Candidates a T3 row returns on the device path: min-p (0.05 by default)
/// keeps the scores within ln(1/min_p) of the best, rarely more than this.
pub(crate) const DEVICE_CANDIDATES: usize = 256;

/// The device's share of a draw: guidance between logits rows `rows` (the
/// conditional and, when guided, the unconditional one) and the repetition
/// penalty over `history` (distinct tokens, ascending), exactly as
/// [`process_logits`] writes them before the temperature. `None` when min-p
/// is off (below 1e-6): top-p alone or plain sampling reads every token.
pub(crate) fn device_candidates<'a>(
    params: &T3SamplingParams,
    rows: (usize, Option<usize>),
    history: &'a [u32],
) -> Option<CandidateRow<'a>> {
    if !params.min_p.ge(&1e-6) {
        return None;
    }
    Some(CandidateRow {
        guidance: rows.1.map(|uncond| (uncond, params.cfg_weight)),
        penalty: CandidatePenalty::Repetition(params.repetition_penalty),
        penalized: history,
        ..CandidateRow::plain(rows.0)
    })
}

/// The tokens a draw can pick, with their processed logits in token order,
/// from a row's device candidates (`values` its guided and penalised logits,
/// largest first, at tokens `indices`, of a `vocab`-token row): exactly the
/// finite logits [`process_logits`] leaves, or `None` when the candidates
/// cannot decide it. Min-p compares each probability with min_p times the
/// best one; both share the softmax total, which only the whole row gives,
/// so a token is decided by its `exp(s - max)` against min_p with a 1e-12
/// relative margin (a total's rounding moves the comparison by about 1e-16),
/// and the last candidate must fall below it, so that every token past the
/// candidates goes too.
pub(crate) fn candidate_survivors(
    values: &[f32],
    indices: &[u32],
    params: &T3SamplingParams,
    vocab: usize,
) -> Option<Vec<(u32, f32)>> {
    if values.is_empty()
        || values.len() != indices.len()
        || values.len() >= vocab
        || !params.min_p.ge(&1e-6)
    {
        return None;
    }
    let scores = values
        .iter()
        .map(|&value| {
            if params.temperature != 1.0 {
                value / params.temperature
            } else {
                value
            }
        })
        .collect::<Vec<_>>();
    if scores.iter().any(|score| !score.is_finite()) {
        return None;
    }
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // The first token at the maximum, as `argmax` picks it.
    let top = indices
        .iter()
        .zip(&scores)
        .filter(|&(_, &score)| score == max)
        .map(|(&index, _)| index)
        .min()?;
    let min_p = f64::from(params.min_p);
    let (below, above) = (min_p * (1.0 - 1e-12), min_p * (1.0 + 1e-12));
    let weight = |score: f32| (f64::from(score) - f64::from(max)).exp();
    if !weight(*scores.last()?).lt(&below) {
        return None;
    }
    let mut kept = Vec::new();
    for (&index, &score) in indices.iter().zip(&scores) {
        if index != top {
            let w = weight(score);
            if w < below {
                continue;
            }
            if w <= above {
                return None;
            }
        }
        kept.push((index, score));
    }
    kept.sort_by_key(|&(index, _)| index);
    if params.top_p < 1.0 {
        let probs = sparse_softmax(&kept);
        let mut ascending = (0..kept.len()).collect::<Vec<_>>();
        ascending.sort_by(|&a, &b| {
            kept[a]
                .1
                .total_cmp(&kept[b].1)
                .then(kept[b].0.cmp(&kept[a].0))
        });
        let floor = 1.0 - f64::from(params.top_p);
        let mut cumulative = 0.0f64;
        for &slot in &ascending[..ascending.len() - 1] {
            cumulative += probs[slot];
            if cumulative <= floor {
                kept[slot].1 = f32::NEG_INFINITY;
            } else {
                break;
            }
        }
        kept.retain(|&(_, score)| score != f32::NEG_INFINITY);
    }
    Some(kept)
}

/// [`sample_processed`] over [`candidate_survivors`]: the same token and the
/// same uniform consumed.
pub(crate) fn sample_survivors(survivors: &[(u32, f32)], rng: &mut SplitMix64) -> Result<u32> {
    let probs = sparse_softmax(survivors);
    let total = probs.iter().sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err(Error::InferenceError(
            "Chatterbox T3 sampling distribution is empty".into(),
        ));
    }
    let target = rng.next_open01() * total;
    let mut cumulative = 0.0f64;
    let mut last_positive = None;
    for (&(index, _), &p) in survivors.iter().zip(&probs) {
        if p <= 0.0 {
            continue;
        }
        cumulative += p;
        last_positive = Some(index);
        if target < cumulative {
            return Ok(index);
        }
    }
    last_positive.ok_or_else(|| Error::InferenceError("Chatterbox T3 sampled nothing".into()))
}

/// [`softmax`] of a row whose other entries are `-inf`, over `entries` in
/// token order.
fn sparse_softmax(entries: &[(u32, f32)]) -> Vec<f64> {
    let max = entries
        .iter()
        .map(|&(_, value)| value)
        .filter(|value| value.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![0.0; entries.len()];
    }
    let exps = entries
        .iter()
        .map(|&(_, value)| {
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

    /// A draw from device candidates (packed by the kernel's host reference)
    /// is the host sampler's token, with its stream left in the same state,
    /// over random guided and unguided logits, histories, penalties,
    /// temperatures, min-p and top-p.
    #[test]
    fn device_candidates_draw_the_host_samplers_token() {
        use crate::kernels::penalized_topk::{candidates_packed, unpack};
        use candle_core::{Device, Tensor};
        const VOCAB: usize = 8_194;
        let mut source = SplitMix64::new(29);
        let (mut drawn, mut whole) = (0, 0);
        for case in 0..300usize {
            let scale = [1.0f32, 2.5, 5.0][case % 3];
            let mut row = || {
                let mut values = source
                    .standard_normals(VOCAB)
                    .into_iter()
                    .map(|v| v * scale)
                    .collect::<Vec<_>>();
                if case % 4 == 0 {
                    for value in &mut values {
                        *value = (*value * 2.0).round() / 2.0;
                    }
                }
                values
            };
            let conditional = row();
            let unconditional = row();
            let mut history = (0..(case * 7) % 300)
                .map(|_| (source.next_u64() % VOCAB as u64) as u32)
                .collect::<Vec<_>>();
            history.push(VOCAB as u32 + 3);
            let params = T3SamplingParams {
                temperature: [0.8, 1.0, 0.3, 1.5][case % 4],
                repetition_penalty: [1.2, 1.0, 2.0, 0.9][(case / 2) % 4],
                min_p: [0.05, 0.2, 0.001, 0.5][(case / 3) % 4],
                top_p: [1.0, 0.95, 0.6][(case / 5) % 3],
                cfg_weight: [0.5, 0.0, 1.5][(case / 7) % 3],
            };
            let guided = params.guided();
            let seed = 1 + case as u64;
            let mut host_rng = SplitMix64::new(seed);
            let expected = sample_next(
                &conditional,
                guided.then_some(unconditional.as_slice()),
                &history,
                &params,
                &mut host_rng,
            )
            .unwrap();

            let mut distinct = history
                .iter()
                .copied()
                .filter(|&token| (token as usize) < VOCAB)
                .collect::<Vec<_>>();
            distinct.sort_unstable();
            distinct.dedup();
            let candidates =
                device_candidates(&params, (0, guided.then_some(1)), &distinct).unwrap();
            let logits = Tensor::from_vec(
                [conditional, unconditional].concat(),
                (2, VOCAB),
                &Device::Cpu,
            )
            .unwrap();
            let packed = candidates_packed(&logits, VOCAB, &[candidates], DEVICE_CANDIDATES)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<u32>()
                .unwrap();
            let candidates = unpack(&packed, DEVICE_CANDIDATES).remove(0);
            assert_eq!(candidates.non_finite, 0);
            match candidate_survivors(&candidates.values, &candidates.indices, &params, VOCAB) {
                Some(survivors) => {
                    let mut rng = SplitMix64::new(seed);
                    let token = sample_survivors(&survivors, &mut rng).unwrap();
                    assert_eq!(token, expected, "case {case}: {params:?}");
                    assert_eq!(rng.next_u64(), host_rng.next_u64(), "case {case}");
                    drawn += 1;
                }
                None => whole += 1,
            }
        }
        assert!(drawn >= 200, "{drawn} drawn from candidates, {whole} whole");
    }

    #[test]
    fn min_p_off_reads_whole_rows_and_unprovable_cuts_ask_for_them() {
        let off = T3SamplingParams {
            min_p: 0.0,
            ..params()
        };
        assert!(device_candidates(&off, (0, None), &[]).is_none());
        // A flat row: hundreds of tokens survive min-p, past the candidates.
        let values = vec![1.0f32; 64];
        let indices = (0..64).collect::<Vec<u32>>();
        assert!(candidate_survivors(&values, &indices, &params(), 8_194).is_none());
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
