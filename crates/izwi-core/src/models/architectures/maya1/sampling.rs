//! Maya1's next-token sampler, in the order Transformers applies the card's
//! settings: repetition penalty, the minimum-length stop suppression, then
//! temperature, top-k and top-p, then a draw from the request's own seeded
//! stream.
//!
//! It works on the head rows only ([`super::config::HEAD_ROWS`]: the stop
//! token, seven control ids and the audio codes). Upstream samples the whole
//! vocabulary; text tokens after the speech start would be garbage it
//! filters out afterwards, and its first streaming script masked them
//! exactly like this (`OnlyAudioAfterSOS`). The control ids after the stop
//! token are never sampled.

use super::config::{HEAD_ROWS, MAYA1_DEFAULTS};
use crate::error::{Error, Result};
use crate::models::shared::noise::SplitMix64;

/// Head row of the stop token; rows 1..=7 are control ids.
const STOP_ROW: usize = 0;
const CONTROL_ROWS: std::ops::RangeInclusive<usize> = 1..=7;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Maya1SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    /// 0 disables top-k.
    pub top_k: usize,
    pub repetition_penalty: f32,
    /// Tokens sampled before the stop token may be.
    pub min_new_tokens: usize,
}

impl Default for Maya1SamplingParams {
    fn default() -> Self {
        let d = MAYA1_DEFAULTS;
        Self {
            temperature: d.temperature,
            top_p: d.top_p,
            top_k: d.top_k,
            repetition_penalty: d.repetition_penalty,
            min_new_tokens: d.min_new_tokens,
        }
    }
}

impl Maya1SamplingParams {
    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite() || !(0.01..=2.0).contains(&self.temperature) {
            return Err(Error::InvalidInput(
                "Maya1 temperature must be within [0.01, 2]; it does not decode greedily".into(),
            ));
        }
        if !self.top_p.is_finite() || !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err(Error::InvalidInput(
                "Maya1 top_p must be within (0, 1]".into(),
            ));
        }
        if !self.repetition_penalty.is_finite() || !(1.0..=2.0).contains(&self.repetition_penalty) {
            return Err(Error::InvalidInput(
                "Maya1 repetition_penalty must be within [1, 2]".into(),
            ));
        }
        Ok(())
    }
}

/// How often a request has sampled each head row, for the repetition
/// penalty (its prompt holds no head token, so only sampled tokens count).
/// Counts rather than flags so a rolled-back step can be forgotten exactly.
#[derive(Debug, Clone)]
pub struct HeadHistory {
    counts: Vec<u32>,
}

impl Default for HeadHistory {
    fn default() -> Self {
        Self {
            counts: vec![0; HEAD_ROWS],
        }
    }
}

impl HeadHistory {
    pub fn record(&mut self, row: usize) {
        if let Some(count) = self.counts.get_mut(row) {
            *count += 1;
        }
    }

    /// Undo one [`Self::record`] of `row`.
    pub fn forget(&mut self, row: usize) {
        if let Some(count) = self.counts.get_mut(row) {
            *count = count.saturating_sub(1);
        }
    }

    pub fn contains(&self, row: usize) -> bool {
        self.counts.get(row).is_some_and(|&count| count > 0)
    }
}

/// Logits of the head rows after every processing step but the draw; a
/// removed row is `-inf`. `generated` is how many tokens the request has
/// sampled so far.
pub fn process_logits(
    logits: &[f32],
    history: &HeadHistory,
    generated: usize,
    params: &Maya1SamplingParams,
) -> Result<Vec<f32>> {
    if logits.len() != HEAD_ROWS {
        return Err(Error::InferenceError(format!(
            "Maya1 sampler got {} logits, expected {HEAD_ROWS}",
            logits.len()
        )));
    }
    if logits.iter().any(|value| value.is_nan()) {
        return Err(Error::InferenceError("Maya1 logits are NaN".into()));
    }
    let mut scores = logits.to_vec();
    let penalty = params.repetition_penalty;
    if penalty != 1.0 {
        for (row, score) in scores.iter_mut().enumerate() {
            if history.contains(row) {
                *score = if *score < 0.0 {
                    *score * penalty
                } else {
                    *score / penalty
                };
            }
        }
    }
    if generated < params.min_new_tokens {
        scores[STOP_ROW] = f32::NEG_INFINITY;
    }
    for row in CONTROL_ROWS {
        scores[row] = f32::NEG_INFINITY;
    }
    for score in &mut scores {
        *score /= params.temperature;
    }
    if params.top_k > 0 && params.top_k < scores.len() {
        let mut sorted = scores.clone();
        let (_, kth, _) = sorted.select_nth_unstable_by(params.top_k - 1, |a, b| b.total_cmp(a));
        let kth = *kth;
        for score in &mut scores {
            if *score < kth {
                *score = f32::NEG_INFINITY;
            }
        }
    }
    if params.top_p < 1.0 {
        apply_top_p(&mut scores, params.top_p);
    }
    if scores.iter().all(|score| *score == f32::NEG_INFINITY) {
        return Err(Error::InferenceError(
            "Maya1 sampler removed every token".into(),
        ));
    }
    Ok(scores)
}

/// Transformers' `TopPLogitsWarper`: sort ascending, and remove the rows
/// whose cumulative probability is at most `1 - top_p`, always keeping the
/// most likely one.
fn apply_top_p(scores: &mut [f32], top_p: f32) {
    let mut live = scores
        .iter()
        .enumerate()
        .filter(|(_, score)| score.is_finite())
        .map(|(row, &score)| (row, score))
        .collect::<Vec<_>>();
    live.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    let max = live.last().map_or(0.0, |entry| entry.1);
    let weights = live
        .iter()
        .map(|&(_, score)| f64::from(score - max).exp())
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f64>();
    let threshold = 1.0 - f64::from(top_p);
    let mut cumulative = 0.0;
    for (index, &(row, _)) in live.iter().enumerate() {
        cumulative += weights[index] / total;
        if cumulative <= threshold && index + 1 < live.len() {
            scores[row] = f32::NEG_INFINITY;
        }
    }
}

/// Sample the next head row from `logits` (see [`process_logits`]) with a
/// uniform from `rng`, walking the survivors from most to least likely.
pub fn sample_next(
    logits: &[f32],
    history: &HeadHistory,
    generated: usize,
    params: &Maya1SamplingParams,
    rng: &mut SplitMix64,
) -> Result<usize> {
    let scores = process_logits(logits, history, generated, params)?;
    let mut live = scores
        .iter()
        .enumerate()
        .filter(|(_, score)| score.is_finite())
        .map(|(row, &score)| (row, score))
        .collect::<Vec<_>>();
    live.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let max = live[0].1;
    let weights = live
        .iter()
        .map(|&(_, score)| f64::from(score - max).exp())
        .collect::<Vec<_>>();
    let target = rng.next_open01() * weights.iter().sum::<f64>();
    let mut cumulative = 0.0;
    for (index, &(row, _)) in live.iter().enumerate() {
        cumulative += weights[index];
        if target < cumulative {
            return Ok(row);
        }
    }
    Ok(live.last().expect("at least one survivor").0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits_with(values: &[(usize, f32)]) -> Vec<f32> {
        let mut logits = vec![-20.0; HEAD_ROWS];
        for &(row, value) in values {
            logits[row] = value;
        }
        logits
    }

    #[test]
    fn stop_is_suppressed_until_the_minimum_length() {
        let logits = logits_with(&[(STOP_ROW, 10.0), (100, 1.0)]);
        let params = Maya1SamplingParams::default();
        let early = process_logits(&logits, &HeadHistory::default(), 27, &params).unwrap();
        assert_eq!(early[STOP_ROW], f32::NEG_INFINITY);
        let later = process_logits(&logits, &HeadHistory::default(), 28, &params).unwrap();
        assert!(later[STOP_ROW].is_finite());
    }

    #[test]
    fn control_rows_are_never_live() {
        let logits = logits_with(&[(3, 50.0), (200, 1.0)]);
        let scores = process_logits(
            &logits,
            &HeadHistory::default(),
            100,
            &Maya1SamplingParams::default(),
        )
        .unwrap();
        assert!(CONTROL_ROWS
            .clone()
            .all(|row| scores[row] == f32::NEG_INFINITY));
        assert!(scores[200].is_finite());
    }

    #[test]
    fn repetition_penalty_divides_positive_and_multiplies_negative_logits() {
        let logits = logits_with(&[(10, 2.2), (11, -1.0)]);
        let mut history = HeadHistory::default();
        history.record(10);
        history.record(11);
        history.record(12);
        history.forget(12);
        assert!(history.contains(10) && !history.contains(12));
        let params = Maya1SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            ..Maya1SamplingParams::default()
        };
        let scores = process_logits(&logits, &history, 100, &params).unwrap();
        assert!((scores[10] - 2.0).abs() < 1e-6);
        assert!((scores[11] + 1.1).abs() < 1e-6);
    }

    #[test]
    fn top_k_keeps_ties_and_top_p_keeps_the_smallest_nucleus() {
        let params = Maya1SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 2,
            ..Maya1SamplingParams::default()
        };
        let logits = logits_with(&[(20, 3.0), (21, 2.0), (22, 2.0), (23, 1.0)]);
        let scores = process_logits(&logits, &HeadHistory::default(), 100, &params).unwrap();
        let live = (0..HEAD_ROWS)
            .filter(|&row| scores[row].is_finite())
            .collect::<Vec<_>>();
        assert_eq!(live, vec![20, 21, 22]);

        // p(20) ≈ 0.88 alone covers 0.85: the rest is cut.
        let params = Maya1SamplingParams {
            top_p: 0.85,
            top_k: 0,
            ..params
        };
        let logits = logits_with(&[(30, 4.0), (31, 2.0)]);
        let scores = process_logits(&logits, &HeadHistory::default(), 100, &params).unwrap();
        assert!(scores[30].is_finite());
        assert_eq!(scores[31], f32::NEG_INFINITY);
    }

    #[test]
    fn draws_are_seeded_and_follow_the_distribution() {
        let logits = logits_with(&[(40, 1.0), (41, 1.0)]);
        let params = Maya1SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            ..Maya1SamplingParams::default()
        };
        let draw = |seed| {
            let mut rng = SplitMix64::new(seed);
            (0..2_000)
                .map(|_| {
                    sample_next(&logits, &HeadHistory::default(), 100, &params, &mut rng).unwrap()
                })
                .collect::<Vec<_>>()
        };
        let first = draw(5);
        assert_eq!(first, draw(5));
        let share = first.iter().filter(|&&row| row == 40).count() as f64 / 2_000.0;
        assert!((share - 0.5).abs() < 0.05, "{share}");
    }
}
