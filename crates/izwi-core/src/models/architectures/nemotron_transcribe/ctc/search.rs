//! NeMo's anchor-guided banded max-sum CTC search (`_banded_ctc_search`,
//! `_ctc_anchor_band_starts`, `_ctc_banded_forward`,
//! `_ctc_banded_backtrace`), with the same F32 arithmetic and tie order.
//!
//! Per frame the DP scores a band of `width` target states starting at
//! `starts[t]`. Under 1000 searched frames the band holds every state;
//! otherwise its half-width is `ceil(0.1·states + 100)`. The band centre
//! interpolates from the first to the last state through anchors: token
//! trigrams that occur exactly once among the greedy onsets and once in
//! the target, kept as the longest chain increasing in time and state.
//! Ties prefer stay, then step, then skip; a skip only lands on a
//! non-blank state whose label differs from the one two states back.
//!
//! Upstream keeps a `frames × width` byte matrix of backpointers. Here
//! memory stays bounded: the forward pass keeps one score row every
//! `segment ≈ √frames` frames, and the backtrace recomputes each segment's
//! backpointers from its checkpoint, newest segment first. Recomputing runs
//! the same arithmetic, so the path is identical, at about twice the
//! forward time: `O(frames × width)` time, `O(√frames × width)` memory.
//!
//! `np.interp` is reproduced with the multiply-add fused, as NumPy's
//! arm64 build computes it (clang contracts `slope*(x - x0) + y0`); an x86
//! NumPy build rounds twice, which can move a band edge by one state where
//! the centre is within an ulp of a half-integer.

use std::collections::HashMap;

/// Searched-frame count from which the band narrows.
const BANDED_FROM_FRAMES: usize = 1000;
/// Fewest frames per backtrace checkpoint segment.
const MIN_SEGMENT: usize = 64;

/// The result of one stream's search.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SearchOutcome {
    /// Target state per searched frame; `None` when no finite path exists.
    pub path: Option<Vec<usize>>,
    /// The path score (`-inf` when the stream cannot fit the frames).
    pub score: f32,
    /// First band state per searched frame; empty when the search did not
    /// run (too few frames).
    pub starts: Vec<usize>,
}

impl SearchOutcome {
    fn failed(score: f32) -> Self {
        Self {
            path: None,
            score,
            starts: Vec::new(),
        }
    }
}

/// `half` of `_banded_ctc_search` for `frames` searched frames.
pub(crate) fn half_width(frames: usize, states: usize) -> usize {
    if frames < BANDED_FROM_FRAMES {
        states
    } else {
        states.min((0.10 * states as f64 + 100.0).ceil() as usize)
    }
}

/// `numpy.argmax` of a row: the first maximum, or the first NaN.
pub(crate) fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (index, &value) in row.iter().enumerate() {
        if value.is_nan() {
            return index;
        }
        if value > row[best] {
            best = index;
        }
    }
    best
}

/// `_banded_ctc_search` over `positions` (indices into the timeline of
/// `logp`, row-major `timeline × columns`). `labels` are the blank-expanded
/// target columns (blank = column 0); `prior[frame]` is added to non-blank
/// states; `greedy` is the timeline's greedy column per frame.
pub(crate) fn banded_ctc_search(
    logp: &[f32],
    columns: usize,
    labels: &[u32],
    positions: &[usize],
    prior: Option<&[f32]>,
    greedy: &[u32],
) -> SearchOutcome {
    let frames = positions.len();
    let states = labels.len();
    let target: Vec<u32> = labels.iter().skip(1).step_by(2).copied().collect();
    let repeats = target.windows(2).filter(|pair| pair[0] == pair[1]).count();
    if frames == 0 || states < 3 || frames < target.len() + repeats {
        return SearchOutcome::failed(f32::NEG_INFINITY);
    }
    let half = half_width(frames, states);
    let width = states.min(2 * half + 1);
    let searched_greedy: Vec<u32> = positions.iter().map(|&frame| greedy[frame]).collect();
    let starts = anchor_band_starts(&searched_greedy, &target, states, half, width);
    let dp = BandedDp::new(logp, columns, labels, positions, prior, &starts, width);
    let (path, score) = dp.search(states);
    SearchOutcome {
        path,
        score,
        starts,
    }
}

/// `_ctc_anchor_band_starts`: the first band state per searched frame.
pub(crate) fn anchor_band_starts(
    greedy: &[u32],
    target: &[u32],
    states: usize,
    half: usize,
    width: usize,
) -> Vec<usize> {
    let frames = greedy.len();
    if frames == 0 {
        return Vec::new();
    }
    // Greedy token onsets: non-blank frames that differ from the previous.
    let times: Vec<usize> = (0..frames)
        .filter(|&t| greedy[t] != 0 && (t == 0 || greedy[t] != greedy[t - 1]))
        .collect();
    let observed: Vec<u32> = times.iter().map(|&t| greedy[t]).collect();
    let left = trigram_positions(&observed);
    let right = trigram_positions(target);
    let mut pairs: Vec<(usize, usize)> = left
        .iter()
        .filter_map(|(key, &(first, count))| {
            let &(target_first, target_count) = right.get(key)?;
            (count == 1 && target_count == 1)
                .then(|| (times[first + 1], 2 * (target_first + 1) + 1))
        })
        .collect();
    pairs.sort_unstable();

    // Longest strictly increasing chain of states (`bisect_left`).
    let mut tails: Vec<usize> = Vec::new();
    let mut tail_ids: Vec<usize> = Vec::new();
    let mut parent: Vec<Option<usize>> = Vec::with_capacity(pairs.len());
    for (index, &(_, state)) in pairs.iter().enumerate() {
        let slot = tails.partition_point(|&tail| tail < state);
        parent.push(slot.checked_sub(1).map(|previous| tail_ids[previous]));
        if slot == tails.len() {
            tails.push(state);
            tail_ids.push(index);
        } else {
            tails[slot] = state;
            tail_ids[slot] = index;
        }
    }
    let mut chosen = Vec::new();
    let mut cursor = tail_ids.last().copied();
    while let Some(index) = cursor {
        chosen.push(pairs[index]);
        cursor = parent[index];
    }
    chosen.reverse();

    let mut xs = vec![0.0f64];
    let mut ys = vec![0.0f64];
    for &(time, state) in &chosen {
        if 0 < time && time < frames - 1 {
            xs.push(time as f64);
            ys.push(state as f64);
        }
    }
    xs.push((frames - 1) as f64);
    ys.push((states - 1) as f64);

    let ceiling = (states - width) as i64;
    let half = half as i64;
    let mut starts = Vec::with_capacity(frames);
    let mut running_min = i64::MAX;
    let mut segment = 0;
    for t in 0..frames {
        let x = t as f64;
        while segment + 1 < xs.len() && xs[segment + 1] <= x {
            segment += 1;
        }
        let center = interp_at(x, segment, &xs, &ys);
        let start = (center.round_ties_even() as i64 - half).clamp(0, ceiling);
        // A band advances at most two states per frame ...
        let t = t as i64;
        running_min = running_min.min(start - 2 * t);
        let start = running_min + 2 * t;
        // ... and must still reach the last states by the last frame.
        let floor = (ceiling - 2 * (frames as i64 - 1 - t)).max(0);
        starts.push(start.max(floor) as usize);
    }
    starts
}

/// `np.interp(x, xs, ys)` given `segment`, the last index with
/// `xs[segment] <= x`, with NumPy's arm64 rounding (one fused multiply-add).
fn interp_at(x: f64, segment: usize, xs: &[f64], ys: &[f64]) -> f64 {
    if segment + 1 >= xs.len() || xs[segment] == x {
        return ys[segment];
    }
    let slope = (ys[segment + 1] - ys[segment]) / (xs[segment + 1] - xs[segment]);
    slope.mul_add(x - xs[segment], ys[segment])
}

/// `_ctc_trigram_positions`, reduced to what the anchors read: the first
/// start of each trigram and how often it occurs.
fn trigram_positions(tokens: &[u32]) -> HashMap<[u32; 3], (usize, usize)> {
    let mut table: HashMap<[u32; 3], (usize, usize)> = HashMap::new();
    for (index, window) in tokens.windows(3).enumerate() {
        let entry = table
            .entry([window[0], window[1], window[2]])
            .or_insert((index, 0));
        entry.1 += 1;
    }
    table
}

/// The banded DP of one stream.
struct BandedDp<'a> {
    logp: &'a [f32],
    columns: usize,
    labels: &'a [u32],
    positions: &'a [usize],
    prior: Option<&'a [f32]>,
    starts: &'a [usize],
    width: usize,
    can_skip: Vec<bool>,
}

impl<'a> BandedDp<'a> {
    fn new(
        logp: &'a [f32],
        columns: usize,
        labels: &'a [u32],
        positions: &'a [usize],
        prior: Option<&'a [f32]>,
        starts: &'a [usize],
        width: usize,
    ) -> Self {
        let can_skip = (0..labels.len())
            .map(|state| state >= 2 && labels[state] != 0 && labels[state] != labels[state - 2])
            .collect();
        Self {
            logp,
            columns,
            labels,
            positions,
            prior,
            starts,
            width,
            can_skip,
        }
    }

    /// Emission score of `label` at timeline frame `frame`.
    fn emission(&self, frame: usize, label: u32, score: f32) -> f32 {
        let value = score + self.logp[frame * self.columns + label as usize];
        match self.prior {
            Some(prior) if label != 0 => value + prior[frame],
            _ => value,
        }
    }

    /// Scores of the first searched frame: the first two states.
    fn first_row(&self) -> Vec<f32> {
        let mut row = vec![f32::NEG_INFINITY; self.width];
        let frame = self.positions[0];
        for (state, slot) in row.iter_mut().enumerate().take(2) {
            *slot = self.emission(frame, self.labels[state], 0.0);
        }
        row
    }

    /// One DP frame: `current` (and its backpointers) from `previous`.
    fn advance(
        &self,
        t: usize,
        previous: &[f32],
        current: &mut [f32],
        mut moves: Option<&mut [u8]>,
    ) {
        let start = self.starts[t];
        let delta = start - self.starts[t - 1];
        let frame = self.positions[t];
        let at = |k: Option<usize>| match k {
            Some(k) if k < self.width => previous[k],
            _ => f32::NEG_INFINITY,
        };
        for j in 0..self.width {
            let k = j + delta;
            let mut best = at(Some(k));
            let mut step = 0u8;
            let one = at(k.checked_sub(1));
            if one > best {
                best = one;
                step = 1;
            }
            let two = at(k.checked_sub(2));
            if two > best && self.can_skip[start + j] {
                best = two;
                step = 2;
            }
            if let Some(moves) = moves.as_deref_mut() {
                moves[j] = step;
            }
            current[j] = self.emission(frame, self.labels[start + j], best);
        }
    }

    /// Forward pass with checkpoints, then the checkpointed backtrace: the
    /// path (`None` when no finite path ends the band) and its score.
    fn search(&self, states: usize) -> (Option<Vec<usize>>, f32) {
        let frames = self.positions.len();
        let segment = MIN_SEGMENT.max((frames as f64).sqrt().ceil() as usize);
        let mut checkpoints: Vec<Vec<f32>> = Vec::with_capacity(frames / segment + 1);
        let mut previous = self.first_row();
        let mut current = vec![f32::NEG_INFINITY; self.width];
        checkpoints.push(previous.clone());
        for t in 1..frames {
            self.advance(t, &previous, &mut current, None);
            std::mem::swap(&mut previous, &mut current);
            if t % segment == 0 {
                checkpoints.push(previous.clone());
            }
        }

        // The path ends in the last or the second-to-last state.
        let last_start = self.starts[frames - 1];
        let mut best: Option<(usize, f32)> = None;
        for end in [states - 1, states - 2] {
            let Some(local) = end.checked_sub(last_start).filter(|&l| l < self.width) else {
                continue;
            };
            let score = previous[local];
            // `argmax`: the first maximum, or the first NaN.
            let better = match best {
                None => true,
                Some((_, held)) => !held.is_nan() && (score.is_nan() || score > held),
            };
            if better {
                best = Some((end, score));
            }
        }
        let Some((mut state, score)) = best else {
            return (None, f32::NEG_INFINITY);
        };
        if !score.is_finite() {
            return (None, score);
        }

        let mut path = vec![0usize; frames];
        let mut moves = vec![0u8; segment * self.width];
        let mut row = vec![f32::NEG_INFINITY; self.width];
        let mut next = vec![f32::NEG_INFINITY; self.width];
        for (index, checkpoint) in checkpoints.iter().enumerate().rev() {
            let first = index * segment;
            let last = (first + segment).min(frames - 1);
            if last <= first {
                continue;
            }
            row.copy_from_slice(checkpoint);
            for t in first + 1..=last {
                let offset = (t - first - 1) * self.width;
                self.advance(
                    t,
                    &row,
                    &mut next,
                    Some(&mut moves[offset..offset + self.width]),
                );
                std::mem::swap(&mut row, &mut next);
            }
            for t in (first + 1..=last).rev() {
                path[t] = state;
                let Some(local) = state
                    .checked_sub(self.starts[t])
                    .filter(|&l| l < self.width)
                else {
                    return (None, score);
                };
                let step = moves[(t - first - 1) * self.width + local] as usize;
                let Some(previous_state) = state.checked_sub(step) else {
                    return (None, score);
                };
                state = previous_state;
            }
        }
        path[0] = state;
        (Some(path), score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's dense formulation (`_ctc_banded_forward_numpy` and
    /// `_ctc_banded_backtrace` with a full backpointer matrix), for
    /// checking the checkpointed backtrace.
    fn dense_reference(
        logp: &[f32],
        columns: usize,
        labels: &[u32],
        positions: &[usize],
        prior: Option<&[f32]>,
        starts: &[usize],
        width: usize,
    ) -> SearchOutcome {
        let dp = BandedDp::new(logp, columns, labels, positions, prior, starts, width);
        let frames = positions.len();
        let mut moves = vec![0u8; frames * width];
        let mut previous = dp.first_row();
        let mut current = vec![0.0; width];
        for t in 1..frames {
            dp.advance(
                t,
                &previous,
                &mut current,
                Some(&mut moves[t * width..(t + 1) * width]),
            );
            std::mem::swap(&mut previous, &mut current);
        }
        let states = labels.len();
        let mut best = None;
        for end in [states - 1, states - 2] {
            if let Some(local) = end.checked_sub(starts[frames - 1]).filter(|&l| l < width) {
                if best.is_none_or(|(_, held): (usize, f32)| previous[local] > held) {
                    best = Some((end, previous[local]));
                }
            }
        }
        let (mut state, score) = best.unwrap();
        if !score.is_finite() {
            return SearchOutcome::failed(score);
        }
        let mut path = vec![0; frames];
        for t in (0..frames).rev() {
            path[t] = state;
            if t > 0 {
                state -= moves[t * width + state - starts[t]] as usize;
            }
        }
        SearchOutcome {
            path: Some(path),
            score,
            starts: starts.to_vec(),
        }
    }

    /// Deterministic noise in `[0, 1)`.
    fn noise(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32) / (1u64 << 24) as f32
    }

    /// Log-probs over `columns` with `tokens` planted at `onsets`.
    fn planted(
        frames: usize,
        columns: usize,
        tokens: &[u32],
        onsets: &[usize],
        seed: u64,
    ) -> Vec<f32> {
        let mut seed = seed;
        let mut logp = vec![0.0f32; frames * columns];
        for t in 0..frames {
            for c in 0..columns {
                logp[t * columns + c] = -8.0 - 4.0 * noise(&mut seed);
            }
            logp[t * columns] = -0.1 * noise(&mut seed);
        }
        for (&token, &t) in tokens.iter().zip(onsets) {
            logp[t * columns + token as usize] = -0.05 * noise(&mut seed);
            logp[t * columns] = -6.0;
        }
        logp
    }

    fn expand(tokens: &[u32]) -> Vec<u32> {
        let mut labels = vec![0];
        for &token in tokens {
            labels.extend([token, 0]);
        }
        labels
    }

    #[test]
    fn short_streams_follow_planted_onsets_through_the_full_band() {
        let tokens = [1, 2, 2, 3, 1];
        let onsets = [3, 7, 9, 15, 30];
        let frames = 40;
        let logp = planted(frames, 4, &tokens, &onsets, 7);
        let greedy: Vec<u32> = (0..frames)
            .map(|t| argmax(&logp[t * 4..t * 4 + 4]) as u32)
            .collect();
        let labels = expand(&tokens);
        let positions: Vec<usize> = (0..frames).collect();
        let outcome = banded_ctc_search(&logp, 4, &labels, &positions, None, &greedy);
        let path = outcome.path.expect("a path");
        for (index, &onset) in onsets.iter().enumerate() {
            assert_eq!(path[onset], 2 * index + 1, "token {index}");
        }
        // The repeated token needs a blank between its two frames.
        assert_eq!(path[8], 4);
        assert!(outcome.score.is_finite());
    }

    #[test]
    fn too_few_frames_fail_like_upstream() {
        // Two equal tokens need three frames.
        let labels = expand(&[1, 1]);
        let logp = vec![0.0f32; 2 * 2];
        let outcome = banded_ctc_search(&logp, 2, &labels, &[0, 1], None, &[1, 1]);
        assert_eq!(outcome, SearchOutcome::failed(f32::NEG_INFINITY));
        let outcome = banded_ctc_search(&logp, 2, &labels, &[], None, &[1, 1]);
        assert!(outcome.path.is_none());
    }

    #[test]
    fn ties_prefer_stay_then_step_then_skip() {
        // All emissions equal: every path ties, so the backtrace keeps
        // staying as long as it can and the path is the latest possible.
        let labels = expand(&[1, 2]);
        let frames = 5;
        let logp = vec![-1.0f32; frames * 3];
        let positions: Vec<usize> = (0..frames).collect();
        let greedy = vec![0; frames];
        let outcome = banded_ctc_search(&logp, 3, &labels, &positions, None, &greedy);
        // Final states 4 (blank) and 3 (token 2) tie: argmax keeps the
        // first, the last state. Walking back, stay wins every tie it can,
        // so the tokens are emitted as early as possible.
        assert_eq!(outcome.path.unwrap(), vec![1, 3, 4, 4, 4]);
    }

    #[test]
    fn checkpointed_backtrace_equals_the_dense_one() {
        // Long enough for a narrow band and many checkpoint segments.
        let mut seed = 11u64;
        let columns = 9;
        let tokens: Vec<u32> = (0..180)
            .map(|_| 1 + (noise(&mut seed) * 8.0) as u32)
            .collect();
        let frames = 1400;
        let onsets: Vec<usize> = (0..tokens.len()).map(|i| 5 + i * 7 + (i % 3)).collect();
        let logp = planted(frames, columns, &tokens, &onsets, 3);
        let greedy: Vec<u32> = (0..frames)
            .map(|t| argmax(&logp[t * columns..(t + 1) * columns]) as u32)
            .collect();
        let prior: Vec<f32> = (0..frames).map(|_| -noise(&mut seed)).collect();
        let positions: Vec<usize> = (0..frames).filter(|t| t % 11 != 0).collect();
        let labels = expand(&tokens);
        let states = labels.len();
        let half = half_width(positions.len(), states);
        let width = states.min(2 * half + 1);
        assert!(width < states, "the band must be narrower than the target");
        let searched: Vec<u32> = positions.iter().map(|&t| greedy[t]).collect();
        let target: Vec<u32> = labels.iter().skip(1).step_by(2).copied().collect();
        let starts = anchor_band_starts(&searched, &target, states, half, width);
        for pair in starts.windows(2) {
            assert!(pair[1] >= pair[0] && pair[1] - pair[0] <= 2);
        }
        assert_eq!(starts[0], 0);
        assert_eq!(*starts.last().unwrap(), states - width);
        let dense = dense_reference(
            &logp,
            columns,
            &labels,
            &positions,
            Some(&prior),
            &starts,
            width,
        );
        let outcome = banded_ctc_search(&logp, columns, &labels, &positions, Some(&prior), &greedy);
        assert!(dense.path.is_some());
        assert_eq!(outcome, dense);
    }

    #[test]
    fn interpolation_fuses_the_multiply_add() {
        // slope = 1/3 rounds down; (1/3)·3 + 0 fused is exactly 1, unfused 1.
        let xs = [0.0, 3.0, 9.0];
        let ys = [0.0, 1.0, 5.0];
        assert_eq!(interp_at(0.0, 0, &xs, &ys), 0.0);
        assert_eq!(interp_at(3.0, 1, &xs, &ys), 1.0);
        assert_eq!(interp_at(9.0, 2, &xs, &ys), 5.0);
        let slope = 4.0f64 / 6.0;
        assert_eq!(interp_at(4.0, 1, &xs, &ys), slope.mul_add(1.0, 1.0));
    }

    #[test]
    fn argmax_matches_numpy() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[1.0, f32::NAN, 5.0]), 1);
        assert_eq!(argmax(&[f32::NEG_INFINITY, f32::NEG_INFINITY]), 0);
    }
}
