//! Host-side sampling with transformers' semantics (`Lfm3AudioForConditional
//! Generation._sample`): a repetition penalty over the unique tokens of a
//! history, suppressed ids, then temperature, top-k (ties kept) and top-p,
//! and a multinomial draw; greedy without a temperature.

use crate::models::shared::noise::SplitMix64;

/// transformers' `RepetitionPenaltyLogitsProcessor`: each distinct token of
/// `history` has its score divided by `penalty` when positive, multiplied
/// when negative.
pub(crate) fn apply_repetition_penalty(
    scores: &mut [f32],
    history: impl IntoIterator<Item = u32>,
    penalty: f32,
) {
    if penalty == 1.0 {
        return;
    }
    let mut seen = std::collections::HashSet::new();
    penalize_distinct(
        scores,
        history.into_iter().filter(|token| seen.insert(*token)),
        penalty,
    );
}

/// [`apply_repetition_penalty`] for a history whose tokens are already
/// distinct.
pub(crate) fn penalize_distinct(
    scores: &mut [f32],
    distinct: impl IntoIterator<Item = u32>,
    penalty: f32,
) {
    if penalty == 1.0 {
        return;
    }
    for token in distinct {
        if let Some(score) = scores.get_mut(token as usize) {
            *score = if *score < 0.0 {
                *score * penalty
            } else {
                *score / penalty
            };
        }
    }
}

/// Draw one id from `scores` (already penalized and suppressed): argmax when
/// `temperature` is zero, else temperature, top-k (`0` keeps all), top-p and
/// a multinomial draw from `rng`.
pub(crate) fn sample(
    scores: &mut [f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    rng: &mut SplitMix64,
) -> u32 {
    if temperature <= 0.0 {
        return argmax(scores);
    }
    for score in scores.iter_mut() {
        *score /= temperature;
    }
    if top_k > 0 && top_k < scores.len() {
        let mut sorted = scores.to_vec();
        let kth = *sorted
            .select_nth_unstable_by(top_k - 1, |a, b| b.total_cmp(a))
            .1;
        for score in scores.iter_mut() {
            if *score < kth {
                *score = f32::NEG_INFINITY;
            }
        }
    }
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs = scores
        .iter()
        .map(|score| ((*score - max) as f64).exp())
        .collect::<Vec<_>>();
    if top_p < 1.0 {
        let mut order = (0..probs.len()).collect::<Vec<_>>();
        order.sort_by(|a, b| probs[*b].total_cmp(&probs[*a]));
        let total = probs.iter().sum::<f64>();
        let mut cumulative = 0.0;
        let mut keep = vec![false; probs.len()];
        for (rank, index) in order.into_iter().enumerate() {
            // transformers keeps the smallest set whose mass reaches top_p,
            // and always the most likely token.
            if rank > 0 && cumulative / total > top_p as f64 {
                break;
            }
            cumulative += probs[index];
            keep[index] = true;
        }
        for (prob, keep) in probs.iter_mut().zip(keep) {
            if !keep {
                *prob = 0.0;
            }
        }
    }
    let total = probs.iter().sum::<f64>();
    let mut target = rng.next_open01() * total;
    for (index, prob) in probs.iter().enumerate() {
        if *prob <= 0.0 {
            continue;
        }
        if target < *prob {
            return index as u32;
        }
        target -= prob;
    }
    argmax(scores)
}

pub(crate) fn argmax(scores: &[f32]) -> u32 {
    let mut best = 0usize;
    for (index, score) in scores.iter().enumerate() {
        if *score > scores[best] {
            best = index;
        }
    }
    best as u32
}

/// A seeded random permutation of `0..len` (Fisher-Yates), the MaskGIT
/// reveal order of one frame.
pub(crate) fn permutation(len: usize, rng: &mut SplitMix64) -> Vec<usize> {
    let mut order = (0..len).collect::<Vec<_>>();
    for index in (1..len).rev() {
        let pick = (rng.next_u64() % (index as u64 + 1)) as usize;
        order.swap(index, pick);
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repetition_penalty_counts_each_token_once() {
        let mut scores = vec![2.0, -2.0, 1.0];
        apply_repetition_penalty(&mut scores, [0, 0, 1], 2.0);
        assert_eq!(scores, vec![1.0, -4.0, 1.0]);
    }

    #[test]
    fn penalizing_distinct_tokens_matches_the_deduplicating_penalty() {
        let mut deduplicated = vec![2.0, -2.0, 0.5, 3.0];
        let mut distinct = deduplicated.clone();
        apply_repetition_penalty(&mut deduplicated, [1, 1, 3, 9, 3], 1.5);
        penalize_distinct(&mut distinct, [3, 1, 9], 1.5);
        assert_eq!(deduplicated, distinct);
    }

    #[test]
    fn top_k_keeps_only_the_best_ids_and_greedy_is_argmax() {
        let mut rng = SplitMix64::new(7);
        for _ in 0..200 {
            let mut scores = vec![0.1, 3.0, 2.9, -1.0, 2.8];
            let picked = sample(&mut scores, 1.0, 2, 1.0, &mut rng);
            assert!(picked == 1 || picked == 2, "{picked}");
        }
        assert_eq!(sample(&mut [0.1, 3.0, 2.9], 0.0, 0, 1.0, &mut rng), 1);
    }

    #[test]
    fn permutations_are_seeded_and_complete() {
        let order = permutation(12, &mut SplitMix64::new(3));
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, (0..12).collect::<Vec<_>>());
        assert_eq!(order, permutation(12, &mut SplitMix64::new(3)));
        assert_ne!(order, permutation(12, &mut SplitMix64::new(4)));
    }
}
