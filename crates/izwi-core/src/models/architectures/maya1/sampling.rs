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
use crate::kernels::penalized_topk::{CandidatePenalty, CandidateRow, MAX_CANDIDATES};
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
    /// The rows with a non-zero count, ascending.
    distinct: Vec<u32>,
}

impl Default for HeadHistory {
    fn default() -> Self {
        Self {
            counts: vec![0; HEAD_ROWS],
            distinct: Vec::new(),
        }
    }
}

impl HeadHistory {
    pub fn record(&mut self, row: usize) {
        if let Some(count) = self.counts.get_mut(row) {
            *count += 1;
            if *count == 1 {
                if let Err(at) = self.distinct.binary_search(&(row as u32)) {
                    self.distinct.insert(at, row as u32);
                }
            }
        }
    }

    /// Undo one [`Self::record`] of `row`.
    pub fn forget(&mut self, row: usize) {
        if let Some(count) = self.counts.get_mut(row) {
            if *count == 1 {
                if let Ok(at) = self.distinct.binary_search(&(row as u32)) {
                    self.distinct.remove(at);
                }
            }
            *count = count.saturating_sub(1);
        }
    }

    pub fn contains(&self, row: usize) -> bool {
        self.counts.get(row).is_some_and(|&count| count > 0)
    }

    /// Every row [`Self::contains`], ascending.
    pub fn distinct(&self) -> &[u32] {
        &self.distinct
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
    Ok(draw(&scores, rng))
}

/// The draw of [`sample_next`] from processed `scores` (`-inf` removed;
/// ties go to the lower position).
fn draw(scores: &[f32], rng: &mut SplitMix64) -> usize {
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
            return row;
        }
    }
    live.last().expect("at least one survivor").0
}

/// The device's share of a request's next draw: the head logits with the
/// repetition penalty applied to its history and the stop and control rows
/// banned, exactly as [`process_logits`] writes them before the
/// temperature, and how many of the largest to return. `None` when top-k is
/// off or wider than the candidates a row returns: that row reads its whole
/// logits.
pub(crate) fn device_candidates<'a>(
    history: &'a HeadHistory,
    generated: usize,
    params: &Maya1SamplingParams,
) -> Option<(CandidateRow<'a>, usize)> {
    let count = candidate_count(params)?;
    let penalty = if params.repetition_penalty != 1.0 {
        CandidatePenalty::Repetition(params.repetition_penalty)
    } else {
        CandidatePenalty::None
    };
    let first = if generated < params.min_new_tokens {
        STOP_ROW
    } else {
        *CONTROL_ROWS.start()
    };
    Some((
        CandidateRow {
            penalty,
            penalized: history.distinct(),
            banned: first..*CONTROL_ROWS.end() + 1,
            ..CandidateRow::plain(0)
        },
        count,
    ))
}

/// Candidates a row returns for `params`: more than its top-k (so the last
/// one can prove the cut), at least 64.
fn candidate_count(params: &Maya1SamplingParams) -> Option<usize> {
    if params.top_k == 0 || params.top_k >= HEAD_ROWS {
        return None;
    }
    let count = (params.top_k + 1).next_power_of_two().max(64);
    (count <= MAX_CANDIDATES).then_some(count)
}

/// The rows top-k keeps, from a row's device candidates (`values` its
/// penalised and banned logits, largest first, at head rows `indices`),
/// with their tempered scores in row order: exactly the finite rows
/// [`process_logits`] leaves before top-p, or `None` when the candidates
/// cannot show that nothing past them survives (the last candidate ties
/// the k-th score) and the row must be sampled from its whole logits.
pub(crate) fn candidate_survivors(
    values: &[f32],
    indices: &[u32],
    params: &Maya1SamplingParams,
) -> Option<Vec<(u32, f32)>> {
    let k = params.top_k;
    if k == 0 || k >= values.len() || values.len() != indices.len() || values.len() >= HEAD_ROWS {
        return None;
    }
    let scores = values
        .iter()
        .map(|&value| value / params.temperature)
        .collect::<Vec<_>>();
    let mut sorted = scores.clone();
    let (_, kth, _) = sorted.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
    let kth = *kth;
    // Every row past the candidates scores at most the last one.
    let last = *scores.last()?;
    if !last.lt(&kth) {
        return None;
    }
    let mut kept = indices
        .iter()
        .zip(scores)
        .filter(|&(_, score)| !score.lt(&kth))
        .map(|(&row, score)| (row, score))
        .collect::<Vec<_>>();
    kept.sort_by_key(|&(row, _)| row);
    Some(kept)
}

/// [`sample_next`]'s top-p and draw over [`candidate_survivors`]: the same
/// head row and the same uniform consumed.
pub(crate) fn sample_survivors(
    survivors: &[(u32, f32)],
    params: &Maya1SamplingParams,
    rng: &mut SplitMix64,
) -> Result<usize> {
    let mut scores = survivors
        .iter()
        .map(|&(_, score)| score)
        .collect::<Vec<_>>();
    if params.top_p < 1.0 {
        apply_top_p(&mut scores, params.top_p);
    }
    if scores.iter().all(|score| *score == f32::NEG_INFINITY) {
        return Err(Error::InferenceError(
            "Maya1 sampler removed every token".into(),
        ));
    }
    Ok(survivors[draw(&scores, rng)].0 as usize)
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

    /// The device path's draw: candidates packed by the kernel's host
    /// reference, survivors, then top-p and the draw. `None` where the
    /// candidates cannot prove the cut.
    fn candidate_draw(
        logits: &[f32],
        history: &HeadHistory,
        generated: usize,
        params: &Maya1SamplingParams,
        rng: &mut SplitMix64,
    ) -> Option<Result<usize>> {
        use crate::kernels::penalized_topk::{candidates_packed, unpack};
        let (row, count) = device_candidates(history, generated, params)?;
        let tensor = candle_core::Tensor::from_vec(
            logits.to_vec(),
            (1, HEAD_ROWS),
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let packed = candidates_packed(&tensor, HEAD_ROWS, &[row], count)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<u32>()
            .unwrap();
        let candidates = unpack(&packed, count).remove(0);
        assert_eq!(candidates.non_finite, 0);
        let survivors = candidate_survivors(&candidates.values, &candidates.indices, params)?;
        Some(sample_survivors(&survivors, params, rng))
    }

    #[test]
    fn device_candidates_draw_the_host_samplers_token() {
        let mut source = SplitMix64::new(91);
        let (mut drawn, mut whole) = (0, 0);
        for case in 0..300u64 {
            let scale = [0.5f32, 2.0, 6.0][case as usize % 3];
            let mut logits = source
                .standard_normals(HEAD_ROWS)
                .into_iter()
                .map(|value| value * scale)
                .collect::<Vec<_>>();
            logits[STOP_ROW] = 1.0 + (case % 5) as f32;
            // Coarse values in some rows: ties everywhere, also at the cut.
            if case % 4 == 0 {
                for value in &mut logits {
                    *value = (*value * 2.0).round() / 2.0;
                }
            }
            if case % 7 == 0 {
                logits[11] = -0.0;
                logits[12] = 0.0;
            }
            let mut history = HeadHistory::default();
            for _ in 0..(case as usize * 13) % 600 {
                history.record(source.next_u64() as usize % HEAD_ROWS);
            }
            let params = Maya1SamplingParams {
                temperature: [0.4, 0.01, 1.3, 2.0][case as usize % 4],
                top_p: [0.9, 1.0, 0.3, 0.999][(case / 4) as usize % 4],
                top_k: [50, 1, 7, 63, 64, 200][(case / 3) as usize % 6],
                repetition_penalty: [1.1, 1.0, 2.0][(case / 2) as usize % 3],
                min_new_tokens: 28,
            };
            let generated = [0, 27, 28, 400][(case / 5) as usize % 4];
            let seed = 1 + case;
            let mut host_rng = SplitMix64::new(seed);
            let expected =
                sample_next(&logits, &history, generated, &params, &mut host_rng).unwrap();
            let mut rng = SplitMix64::new(seed);
            match candidate_draw(&logits, &history, generated, &params, &mut rng) {
                Some(token) => {
                    assert_eq!(token.unwrap(), expected, "case {case}: {params:?}");
                    assert_eq!(rng.next_u64(), host_rng.next_u64(), "case {case}: stream");
                    drawn += 1;
                }
                None => whole += 1,
            }
        }
        // Only rows tied at the cut read their whole logits.
        assert!(drawn >= 250, "{drawn} drawn from candidates, {whole} whole");
    }

    #[test]
    fn candidates_tied_at_the_cut_ask_for_the_whole_row() {
        // 64 equal leaders: the k-th score ties the last candidate, so a
        // tie past the candidates could survive top-k.
        let mut logits = vec![-5.0; HEAD_ROWS];
        for row in 100..200 {
            logits[row] = 3.0;
        }
        let params = Maya1SamplingParams::default();
        let mut rng = SplitMix64::new(3);
        assert!(candidate_draw(&logits, &HeadHistory::default(), 100, &params, &mut rng).is_none());
        // Top-k off: no candidates at all.
        let params = Maya1SamplingParams { top_k: 0, ..params };
        assert!(device_candidates(&HeadHistory::default(), 0, &params).is_none());
    }

    #[test]
    fn history_lists_its_distinct_rows_through_rollbacks() {
        let mut history = HeadHistory::default();
        for row in [9, 3, 9, 200, 3] {
            history.record(row);
        }
        assert_eq!(history.distinct(), &[3, 9, 200]);
        history.forget(9);
        assert_eq!(history.distinct(), &[3, 9, 200]);
        history.forget(9);
        history.forget(200);
        assert_eq!(history.distinct(), &[3]);
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
