//! Mapping t-SOT speaker tags to diarizer columns
//! (`_one_pass_speaker_mapping`) and the linear assignment it solves
//! (SciPy's `linear_sum_assignment`, ported from `rectangular_lsap.cpp`:
//! Crouse's shortest augmenting path, with SciPy's tie-breaking, so equal
//! costs resolve to the same columns).
//!
//! The similarity is computed in F64 as upstream, but its dot products
//! and norms are plain sequential sums where NumPy calls BLAS and pairwise
//! summation; the two can differ in the last bit, which only matters when
//! two assignments tie to within that bit.

/// Speaker column per stream, as `_one_pass_speaker_mapping` returns it:
/// `None` for a stream left unassigned (more streams than columns).
///
/// * `greedy`: greedy column per CTC frame (compact columns, blank 0);
/// * `columns`: number of compact columns;
/// * `expected`: each stream's target token columns (`labels[1::2]`);
/// * `speaker`: row-major `frames × speakers` probabilities on the CTC grid.
///
/// `None` when the assignment has no solution (SciPy raises).
pub(crate) fn speaker_mapping(
    greedy: &[u32],
    columns: usize,
    expected: &[Vec<u32>],
    speaker: &[f32],
    speakers: usize,
    threshold: f32,
) -> Option<Vec<Option<usize>>> {
    if expected.is_empty() {
        return Some(Vec::new());
    }
    if speakers == 0 {
        return Some(vec![None; expected.len()]);
    }
    let mut observed = vec![vec![0u64; columns]; speakers];
    for (t, &token) in greedy.iter().enumerate() {
        let onset = token != 0 && (t == 0 || greedy[t - 1] != token);
        if !onset {
            continue;
        }
        for (column, counts) in observed.iter_mut().enumerate() {
            if speaker[t * speakers + column] >= threshold {
                counts[token as usize] += 1;
            }
        }
    }
    let expected: Vec<Vec<u64>> = expected
        .iter()
        .map(|tokens| {
            let mut counts = vec![0u64; columns];
            for &token in tokens {
                counts[token as usize] += 1;
            }
            counts
        })
        .collect();
    let idf: Vec<f64> = (0..columns)
        .map(|c| {
            let present = observed.iter().filter(|counts| counts[c] > 0).count();
            (speakers as f64 / (1 + present) as f64).ln_1p()
        })
        .collect();
    let weigh = |counts: &Vec<u64>| -> Vec<f64> {
        counts
            .iter()
            .zip(&idf)
            .map(|(&count, &weight)| (count as f64 * weight).sqrt())
            .collect()
    };
    let a: Vec<Vec<f64>> = expected.iter().map(weigh).collect();
    let b: Vec<Vec<f64>> = observed.iter().map(weigh).collect();
    let norm = |v: &Vec<f64>| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let total = |counts: &Vec<u64>| counts.iter().sum::<u64>() as f64;

    let mut cost = Vec::with_capacity(a.len() * b.len());
    for (row, counts) in a.iter().zip(&expected) {
        for (column, observed_counts) in b.iter().zip(&observed) {
            let dot: f64 = row.iter().zip(column).map(|(x, y)| x * y).sum();
            let mut similarity = dot / 1e-8f64.max(norm(row) * norm(column));
            // Penalize grossly incompatible amounts of speech.
            similarity -= 0.1
                * ((total(counts) + 1.0) / (total(observed_counts) + 1.0))
                    .ln()
                    .abs();
            cost.push(-similarity);
        }
    }
    let pairs = linear_sum_assignment(&cost, a.len(), b.len())?;
    let mut mapping = vec![None; a.len()];
    for (row, column) in pairs {
        mapping[row] = Some(column);
    }
    Some(mapping)
}

/// SciPy's `linear_sum_assignment(cost)` (minimisation) on a row-major
/// `rows × cols` matrix: the `(row, column)` pairs in row order, or `None`
/// for a NaN or `-inf` entry or an infeasible matrix.
pub(crate) fn linear_sum_assignment(
    cost: &[f64],
    rows: usize,
    cols: usize,
) -> Option<Vec<(usize, usize)>> {
    if rows == 0 || cols == 0 {
        return Some(Vec::new());
    }
    let transpose = cols < rows;
    let (nr, nc) = if transpose {
        (cols, rows)
    } else {
        (rows, cols)
    };
    let matrix: Vec<f64> = if transpose {
        let mut transposed = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                transposed[j * rows + i] = cost[i * cols + j];
            }
        }
        transposed
    } else {
        cost.to_vec()
    };
    if matrix.iter().any(|&c| c.is_nan() || c == f64::NEG_INFINITY) {
        return None;
    }

    let mut u = vec![0.0f64; nr];
    let mut v = vec![0.0f64; nc];
    let mut shortest = vec![0.0f64; nc];
    let mut path = vec![usize::MAX; nc];
    let mut col4row = vec![usize::MAX; nr];
    let mut row4col = vec![usize::MAX; nc];
    let mut sr = vec![false; nr];
    let mut sc = vec![false; nc];
    let mut remaining = vec![0usize; nc];

    for current_row in 0..nr {
        // Shortest augmenting path from `current_row`.
        let mut min_value = 0.0f64;
        let mut num_remaining = nc;
        for (it, slot) in remaining.iter_mut().enumerate() {
            // Reverse order, so a constant matrix yields the identity.
            *slot = nc - it - 1;
        }
        sr.fill(false);
        sc.fill(false);
        shortest.fill(f64::INFINITY);
        let mut i = current_row;
        let sink = loop {
            let mut index = usize::MAX;
            let mut lowest = f64::INFINITY;
            sr[i] = true;
            for it in 0..num_remaining {
                let j = remaining[it];
                let r = min_value + matrix[i * nc + j] - u[i] - v[j];
                if r < shortest[j] {
                    path[j] = i;
                    shortest[j] = r;
                }
                if shortest[j] < lowest || (shortest[j] == lowest && row4col[j] == usize::MAX) {
                    lowest = shortest[j];
                    index = it;
                }
            }
            min_value = lowest;
            if min_value == f64::INFINITY || index == usize::MAX {
                return None;
            }
            let j = remaining[index];
            sc[j] = true;
            num_remaining -= 1;
            remaining[index] = remaining[num_remaining];
            if row4col[j] == usize::MAX {
                break j;
            }
            i = row4col[j];
        };

        // Update the dual variables.
        u[current_row] += min_value;
        for row in 0..nr {
            if sr[row] && row != current_row {
                u[row] += min_value - shortest[col4row[row]];
            }
        }
        for column in 0..nc {
            if sc[column] {
                v[column] -= min_value - shortest[column];
            }
        }

        // Augment the previous solution.
        let mut j = sink;
        loop {
            let row = path[j];
            row4col[j] = row;
            std::mem::swap(&mut col4row[row], &mut j);
            if row == current_row {
                break;
            }
        }
    }

    if transpose {
        // Rows of the transposed problem are the original columns; report
        // them ordered by their original row.
        let mut order: Vec<usize> = (0..nr).collect();
        order.sort_by_key(|&column| col4row[column]);
        Some(
            order
                .into_iter()
                .map(|column| (col4row[column], column))
                .collect(),
        )
    } else {
        Some((0..nr).map(|row| (row, col4row[row])).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every assignment of `rows ≤ cols`, for the optimum.
    fn brute_force(cost: &[f64], rows: usize, cols: usize) -> f64 {
        fn go(cost: &[f64], rows: usize, cols: usize, row: usize, used: &mut Vec<bool>) -> f64 {
            if row == rows {
                return 0.0;
            }
            let mut best = f64::INFINITY;
            for c in 0..cols {
                if !used[c] {
                    used[c] = true;
                    best = best.min(cost[row * cols + c] + go(cost, rows, cols, row + 1, used));
                    used[c] = false;
                }
            }
            best
        }
        go(cost, rows, cols, 0, &mut vec![false; cols])
    }

    #[test]
    fn assignment_is_optimal_and_breaks_ties_like_scipy() {
        // SciPy: a constant matrix gives the identity.
        assert_eq!(
            linear_sum_assignment(&[1.0; 9], 3, 3).unwrap(),
            vec![(0, 0), (1, 1), (2, 2)]
        );
        // SciPy: [[4,1,3],[2,0,5],[3,2,2]] -> rows [0,1,2], cols [1,0,2].
        let cost = [4.0, 1.0, 3.0, 2.0, 0.0, 5.0, 3.0, 2.0, 2.0];
        assert_eq!(
            linear_sum_assignment(&cost, 3, 3).unwrap(),
            vec![(0, 1), (1, 0), (2, 2)]
        );
        // Tall matrices are solved transposed and reported by row.
        let tall = [1.0, 5.0, 2.0, 0.5, 9.0, 9.0];
        let pairs = linear_sum_assignment(&tall, 3, 2).unwrap();
        assert_eq!(pairs, vec![(0, 0), (1, 1)]);
        // NaN or -inf entries are rejected.
        assert!(linear_sum_assignment(&[f64::NAN, 0.0], 1, 2).is_none());
        assert!(linear_sum_assignment(&[f64::NEG_INFINITY, 0.0], 1, 2).is_none());
        // Random wide matrices: optimal.
        let mut seed = 5u64;
        for trial in 0..200 {
            let rows = 1 + trial % 4;
            let cols = rows + (trial / 4) % 3;
            let cost: Vec<f64> = (0..rows * cols)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((seed >> 59) as f64) - 8.0
                })
                .collect();
            let pairs = linear_sum_assignment(&cost, rows, cols).unwrap();
            let total: f64 = pairs.iter().map(|&(r, c)| cost[r * cols + c]).sum();
            assert_eq!(total, brute_force(&cost, rows, cols));
            let mut used: Vec<usize> = pairs.iter().map(|&(_, c)| c).collect();
            used.sort_unstable();
            used.dedup();
            assert_eq!(used.len(), rows);
        }
    }

    #[test]
    fn mapping_follows_token_distributions() {
        // Columns: blank 0, tokens 1..=4. Stream A says 1 2, stream B 3 4.
        // Diarizer column 1 is active while 1 2 are emitted, column 0
        // while 3 4 are; column 2 never.
        let greedy = [0, 1, 0, 2, 2, 0, 3, 0, 4, 0];
        let mut speaker = vec![0.0f32; 10 * 3];
        for t in 0..5 {
            speaker[t * 3 + 1] = 0.9;
        }
        for t in 5..10 {
            speaker[t * 3] = 0.8;
        }
        let expected = vec![vec![1, 2], vec![3, 4]];
        let mapping = speaker_mapping(&greedy, 5, &expected, &speaker, 3, 0.5).unwrap();
        assert_eq!(mapping, vec![Some(1), Some(0)]);
        // More streams than columns: the extra stream is unmapped.
        let single: Vec<f32> = (0..10).map(|t| speaker[t * 3 + 1]).collect();
        let mapping = speaker_mapping(&greedy, 5, &expected, &single, 1, 0.5).unwrap();
        assert_eq!(mapping, vec![Some(0), None]);
        // SciPy on all-zero matrices, wide and tall.
        assert_eq!(
            linear_sum_assignment(&[0.0; 8], 2, 4).unwrap(),
            vec![(0, 0), (1, 1)]
        );
        assert_eq!(
            linear_sum_assignment(&[0.0; 8], 4, 2).unwrap(),
            vec![(0, 0), (1, 1)]
        );
        // No speakers: nothing maps.
        assert_eq!(
            speaker_mapping(&greedy, 5, &expected, &[], 0, 0.5).unwrap(),
            vec![None, None]
        );
        assert!(speaker_mapping(&greedy, 5, &[], &speaker, 3, 0.5)
            .unwrap()
            .is_empty());
    }
}
