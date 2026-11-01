//! The sampler (§3.6): an own copy of the fish-speech sampler Drama 3 ships
//! with (`fish_omni/tower.py:47-73`), with repetition-aware sampling (RAS)
//! of the main token (`tower.py:438-470`).
//!
//! One draw over a row of candidate logits:
//!
//! ```text
//! v      = logits / max(temperature, 1e-5)                   # F32
//! order  = v sorted descending (ties keep candidate order)
//! p      = softmax(v) in rank order
//! keep r = r == 0 || (r < top_k && cumsum(p)[r] <= top_p)   # the token crossing top_p goes too
//! q      = softmax(v) over the kept ranks
//! pick   = argmax_r q[r] / E[order[r]]                       # E ~ Exp(1) per candidate
//! ```
//!
//! Top-p reads the *tempered* distribution (Fish S2's sampler filters the
//! raw softmax before temperature; this one does not). The exponential race
//! draws from `q` exactly. Its noise is indexed by candidate, not by rank, so
//! a recorded rollout replays bit for bit whatever the sort does, and a
//! device path handed the same noise picks the same token.
//!
//! **Main token.** The candidates are the compact head's (`head.rs`): eos,
//! then the semantic band. Every frame draws twice, at the request's policy
//! and at the fixed RAS policy (1.0, top-k 30, top-p 0.95), so the noise
//! stream never depends on the outcome. The second draw replaces the first
//! when the first is a *semantic* token among the last 16 main ids, whose
//! window starts with the prompt's rows. Eos is never redrawn.
//!
//! **Codebook heads.** One draw each at the heads' policy, no RAS. A head is
//! as wide as the semantic codebook, so it may emit a code the residual
//! codebooks do not hold; the tower feeds it back unclamped and only the
//! codec clamps it.
//!
//! **Defaults (D7).** Temperature 1.0, top-p 0.9, the main token uncapped,
//! the heads capped at top-k 30. A request's `top_k` caps both. There is no
//! repetition penalty.
//!
//! **First frame.** §7.5 excludes eos from both main draws until the
//! generation has a frame (the zero-frame rule Fish S2 carries). Upstream has
//! no such mask, so the caller passes it per draw and a parity replay turns
//! it off.
//!
//! **State.** A row's sampler owns its noise stream (ChaCha12 seeded from
//! the request) and its RAS window. Both fit in a
//! [`FishDrama3SamplerCheckpoint`] of a few hundred bytes, which the quantum
//! takes and restores (§7.2). A draw consumes one exponential per candidate,
//! masked or not, so the stream stays aligned across policies.
//!
//! Device-resident sampling (`metal::sampling`) slots in behind the same
//! split: the noise comes from [`FishDrama3RowSampler::main_noise`] and
//! [`FishDrama3RowSampler::exponentials`] in the order the host draws take
//! it, the selection is [`race`]'s rule over the device's logits, and RAS
//! is [`FishDrama3RowSampler::finish_main`] on the host.

use std::collections::VecDeque;
#[cfg(test)]
use std::sync::Arc;

// ChaCha12 is what StdRng runs, but it stays Clone (checkpoints snapshot it)
// and its seeded stream is stable across releases.
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;

use super::config::FishDrama3TokenIds;
use crate::error::{Error, Result};

/// Main ids the RAS window holds.
pub const RAS_WINDOW: usize = 16;
/// The temperature floor: a lower one (0 included) is raised to it.
pub const MIN_TEMPERATURE: f32 = 1e-5;
pub const DEFAULT_TEMPERATURE: f32 = 1.0;
pub const DEFAULT_TOP_P: f32 = 0.9;
/// The heads' cap when the request names no `top_k`.
pub const DEFAULT_HEAD_TOP_K: usize = 30;

/// The policy of one draw.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FishDrama3DrawPolicy {
    /// Raised to [`MIN_TEMPERATURE`]; 0 is therefore near-greedy.
    pub temperature: f32,
    /// In `(0, 1]`.
    pub top_p: f32,
    /// At least 1; `None` keeps every rank.
    pub top_k: Option<usize>,
}

impl FishDrama3DrawPolicy {
    /// The RAS redraw (`RAS_TEMPERATURE`, `RAS_TOP_K`, `RAS_TOP_P`).
    pub const RAS: Self = Self {
        temperature: 1.0,
        top_p: 0.95,
        top_k: Some(30),
    };

    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(Error::InvalidInput(format!(
                "Fish Audio Drama 3 temperature must be finite and non-negative; got {}",
                self.temperature
            )));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(Error::InvalidInput(format!(
                "Fish Audio Drama 3 top_p must be in (0, 1]; got {}",
                self.top_p
            )));
        }
        // Rank 0 always stays, so a cap of 0 would silently mean greedy.
        if self.top_k == Some(0) {
            return Err(Error::InvalidInput(
                "Fish Audio Drama 3 top_k must be at least 1".into(),
            ));
        }
        Ok(())
    }
}

/// A request's sampling: the main token's policy, the heads' and the seed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FishDrama3SamplingParams {
    pub main: FishDrama3DrawPolicy,
    pub heads: FishDrama3DrawPolicy,
    pub seed: u64,
}

impl FishDrama3SamplingParams {
    /// D7 with the request's values over it. `top_k`, when given, caps the
    /// main token and the heads alike (fish-speech and vllm); without it
    /// the main token is uncapped and the heads take
    /// [`DEFAULT_HEAD_TOP_K`].
    pub fn from_request(
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<usize>,
        seed: u64,
    ) -> Result<Self> {
        let policy = |cap| FishDrama3DrawPolicy {
            temperature: temperature.unwrap_or(DEFAULT_TEMPERATURE),
            top_p: top_p.unwrap_or(DEFAULT_TOP_P),
            top_k: cap,
        };
        let params = Self {
            main: policy(top_k),
            heads: policy(top_k.or(Some(DEFAULT_HEAD_TOP_K))),
            seed,
        };
        params.validate()?;
        Ok(params)
    }

    pub fn validate(&self) -> Result<()> {
        self.main.validate()?;
        self.heads.validate()
    }
}

/// One frame's main token and how RAS chose it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FishDrama3MainDraw {
    /// Eos or a semantic id.
    pub token: u32,
    /// The draw at the request's policy.
    pub first: u32,
    /// The draw at the RAS policy, always made.
    pub retry: u32,
    /// Whether the retry replaced the first draw.
    pub redrawn: bool,
}

/// Where a row's exponentials come from.
#[derive(Clone, PartialEq)]
enum Noise {
    Seeded(Box<ChaCha12Rng>),
    /// A recorded stream, consumed in order (parity replays).
    #[cfg(test)]
    Recorded {
        values: Arc<[f32]>,
        cursor: usize,
    },
}

/// `-ln(u)` for `u` uniform on `(0, 1)` from 53 bits: never 0 or infinite.
fn exponential(bits: u64) -> f32 {
    let unit = ((bits >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64);
    -unit.ln() as f32
}

/// A rollback point of one row's sampler: its noise stream and RAS window.
#[derive(Clone, PartialEq)]
pub struct FishDrama3SamplerCheckpoint {
    noise: Noise,
    window: VecDeque<u32>,
}

impl std::fmt::Debug for FishDrama3SamplerCheckpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FishDrama3SamplerCheckpoint")
            .field("window", &self.window)
            .finish_non_exhaustive()
    }
}

/// One request's sampler.
#[derive(Clone)]
pub struct FishDrama3RowSampler {
    params: FishDrama3SamplingParams,
    ids: FishDrama3TokenIds,
    noise: Noise,
    /// The last [`RAS_WINDOW`] main ids, oldest first.
    window: VecDeque<u32>,
}

impl std::fmt::Debug for FishDrama3RowSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FishDrama3RowSampler")
            .field("params", &self.params)
            .field("window", &self.window)
            .finish_non_exhaustive()
    }
}

fn non_finite_error(logits: &[f32], label: &str) -> Error {
    let nan = logits.iter().filter(|v| v.is_nan()).count();
    let positive = logits.iter().filter(|v| **v == f32::INFINITY).count();
    let negative = logits.iter().filter(|v| **v == f32::NEG_INFINITY).count();
    Error::InferenceError(format!(
        "Fish Audio Drama 3 {label} logits are non-finite: nan={nan}, positive_infinity={positive}, negative_infinity={negative}"
    ))
}

/// The ranks one draw keeps over candidates `first..logits.len()`: their
/// candidate indices in rank order, and `exp(v - max)` of each (the race's
/// unnormalised weights). The logits must be finite.
fn kept(
    logits: &[f32],
    policy: &FishDrama3DrawPolicy,
    first: usize,
) -> Result<(Vec<usize>, Vec<f32>)> {
    if first >= logits.len() {
        return Err(Error::InferenceError(format!(
            "a Fish Audio Drama 3 draw has no candidate past {first} of {}",
            logits.len()
        )));
    }
    let temperature = policy.temperature.max(MIN_TEMPERATURE);
    let scaled = logits.iter().map(|l| l / temperature).collect::<Vec<_>>();
    let mut order = (first..logits.len()).collect::<Vec<_>>();
    // `sort_by` is stable: ties keep candidate order.
    order.sort_by(|&a, &b| scaled[b].total_cmp(&scaled[a]));
    let max = scaled[order[0]];
    let weights = order
        .iter()
        .map(|&i| (scaled[i] - max).exp())
        .collect::<Vec<_>>();
    let inverse = (1.0 / weights.iter().map(|w| f64::from(*w)).sum::<f64>()) as f32;
    let limit = policy.top_k.unwrap_or(usize::MAX);
    // Inclusive cumulative probabilities only grow, so the kept ranks are a
    // prefix: the first rank past the cap or over top_p ends it.
    let mut cumulative = 0f64;
    let mut count = 0;
    for (rank, weight) in weights.iter().enumerate() {
        cumulative += f64::from(weight * inverse);
        if rank > 0 && (rank >= limit || cumulative as f32 > policy.top_p) {
            break;
        }
        count += 1;
    }
    order.truncate(count);
    let mut weights = weights;
    weights.truncate(count);
    Ok((order, weights))
}

/// One draw over candidates `first..logits.len()` (`first` 1 drops eos from
/// the main row): the winning candidate's index. `noise` holds one
/// exponential per candidate of the whole row, in candidate order. The
/// logits must be finite.
pub fn race(
    logits: &[f32],
    policy: &FishDrama3DrawPolicy,
    first: usize,
    noise: &[f32],
) -> Result<usize> {
    if noise.len() != logits.len() {
        return Err(Error::InferenceError(format!(
            "{} exponentials for {} Fish Audio Drama 3 candidates",
            noise.len(),
            logits.len()
        )));
    }
    let (order, weights) = kept(logits, policy, first)?;
    let inverse = (1.0 / weights.iter().map(|w| f64::from(*w)).sum::<f64>()) as f32;
    let mut best = (order[0], f32::NEG_INFINITY);
    for (&candidate, weight) in order.iter().zip(&weights) {
        // `q / E` as upstream divides; the first maximum in rank order wins.
        let score = weight * inverse / noise[candidate];
        if score > best.1 {
            best = (candidate, score);
        }
    }
    Ok(best.0)
}

impl FishDrama3RowSampler {
    /// A request's sampler, its noise seeded from `params.seed` and its RAS
    /// window holding the last [`RAS_WINDOW`] of the prompt's main ids.
    pub fn new(
        params: FishDrama3SamplingParams,
        ids: FishDrama3TokenIds,
        prompt_main_ids: &[u32],
    ) -> Result<Self> {
        params.validate()?;
        let noise = Noise::Seeded(Box::new(ChaCha12Rng::seed_from_u64(params.seed)));
        Ok(Self::with_noise(params, ids, prompt_main_ids, noise))
    }

    fn with_noise(
        params: FishDrama3SamplingParams,
        ids: FishDrama3TokenIds,
        prompt_main_ids: &[u32],
        noise: Noise,
    ) -> Self {
        let tail = prompt_main_ids.len().saturating_sub(RAS_WINDOW);
        Self {
            params,
            ids,
            noise,
            window: prompt_main_ids[tail..].iter().copied().collect(),
        }
    }

    /// [`Self::new`] drawing its exponentials from `noise`, in order: each
    /// draw takes one per candidate of its row, in candidate order (main
    /// rows eos first). Errors once the stream runs out.
    #[cfg(test)]
    pub(crate) fn with_recorded_noise(
        params: FishDrama3SamplingParams,
        ids: FishDrama3TokenIds,
        prompt_main_ids: &[u32],
        noise: Vec<f32>,
    ) -> Result<Self> {
        params.validate()?;
        Ok(Self::with_noise(
            params,
            ids,
            prompt_main_ids,
            Noise::Recorded {
                values: noise.into(),
                cursor: 0,
            },
        ))
    }

    pub fn params(&self) -> &FishDrama3SamplingParams {
        &self.params
    }

    /// The RAS window, oldest first.
    pub fn window(&self) -> Vec<u32> {
        self.window.iter().copied().collect()
    }

    /// The next `n` exponentials of the stream, one per candidate.
    pub fn exponentials(&mut self, n: usize) -> Result<Vec<f32>> {
        match &mut self.noise {
            Noise::Seeded(rng) => Ok((0..n).map(|_| exponential(rng.next_u64())).collect()),
            #[cfg(test)]
            Noise::Recorded { values, cursor } => {
                let end = *cursor + n;
                let drawn = values.get(*cursor..end).ok_or_else(|| {
                    Error::InferenceError(format!(
                        "the recorded noise holds {} exponentials; a draw needs {end}",
                        values.len()
                    ))
                })?;
                *cursor = end;
                Ok(drawn.to_vec())
            }
        }
    }

    pub fn checkpoint(&self) -> FishDrama3SamplerCheckpoint {
        FishDrama3SamplerCheckpoint {
            noise: self.noise.clone(),
            window: self.window.clone(),
        }
    }

    pub fn restore(&mut self, checkpoint: FishDrama3SamplerCheckpoint) {
        self.noise = checkpoint.noise;
        self.window = checkpoint.window;
    }

    fn main_token(&self, index: usize) -> u32 {
        if index == 0 {
            self.ids.eos
        } else {
            self.ids.semantic_start + (index - 1) as u32
        }
    }

    /// The main token of the next frame from the compact head's logits
    /// (eos, then the semantic band), and the token joins the RAS window.
    /// `allow_eos` false (the first frame, §7.5) leaves eos out of both
    /// draws. Non-finite logits are an error, eos included.
    pub fn sample_main(&mut self, logits: &[f32], allow_eos: bool) -> Result<FishDrama3MainDraw> {
        let candidates = 1 + self.ids.semantic_codes();
        if logits.len() != candidates {
            return Err(Error::InferenceError(format!(
                "Fish Audio Drama 3 main logits hold {} candidates; the compact head has {candidates}",
                logits.len()
            )));
        }
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(non_finite_error(logits, "main"));
        }
        let first_candidate = usize::from(!allow_eos);
        let noise = self.exponentials(candidates)?;
        let first = race(logits, &self.params.main, first_candidate, &noise)?;
        let noise = self.exponentials(candidates)?;
        let retry = race(logits, &FishDrama3DrawPolicy::RAS, first_candidate, &noise)?;
        Ok(self.finish_main(first, retry))
    }

    /// The exponentials of one frame's two main draws, in the order
    /// [`Self::sample_main`] takes them (the request policy's, then RAS's),
    /// for a device race over the same candidates.
    pub fn main_noise(&mut self) -> Result<(Vec<f32>, Vec<f32>)> {
        let candidates = 1 + self.ids.semantic_codes();
        Ok((
            self.exponentials(candidates)?,
            self.exponentials(candidates)?,
        ))
    }

    /// RAS over a frame's two draws (candidate indices of the compact head,
    /// the request policy's and the retry's), and the token joins the
    /// window: the end of [`Self::sample_main`], which a device race shares.
    pub fn finish_main(&mut self, first: usize, retry: usize) -> FishDrama3MainDraw {
        let (first, retry) = (self.main_token(first), self.main_token(retry));
        // Only semantic repeats: the window starts with prompt rows, which
        // hold the user turn's eos, and a stop is never redrawn.
        let redrawn = self.ids.is_semantic(first) && self.window.contains(&first);
        let token = if redrawn { retry } else { first };
        if self.window.len() == RAS_WINDOW {
            self.window.pop_front();
        }
        self.window.push_back(token);
        FishDrama3MainDraw {
            token,
            first,
            retry,
            redrawn,
        }
    }

    /// A codebook head's code from its logits (one per code). Non-finite
    /// logits are an error.
    pub fn sample_code(&mut self, logits: &[f32]) -> Result<u32> {
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(non_finite_error(logits, "codebook head"));
        }
        let noise = self.exponentials(logits.len())?;
        Ok(race(logits, &self.params.heads, 0, &noise)? as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::tests::published;
    use super::*;
    use crate::kernels::numeric_check::xorshift_values;

    fn ids() -> FishDrama3TokenIds {
        published().ids
    }

    fn policy(temperature: f32, top_p: f32, top_k: Option<usize>) -> FishDrama3DrawPolicy {
        FishDrama3DrawPolicy {
            temperature,
            top_p,
            top_k,
        }
    }

    /// The plain F64 sampler: the same rule written from `tower.py:47-73`
    /// with every value in F64.
    fn reference(
        logits: &[f32],
        policy: &FishDrama3DrawPolicy,
        first: usize,
        noise: &[f32],
    ) -> usize {
        let temperature = f64::from(policy.temperature.max(MIN_TEMPERATURE));
        let v = logits
            .iter()
            .map(|l| f64::from(*l) / temperature)
            .collect::<Vec<_>>();
        let mut order = (first..logits.len()).collect::<Vec<_>>();
        order.sort_by(|&a, &b| v[b].total_cmp(&v[a]));
        let max = v[order[0]];
        let e = order
            .iter()
            .map(|&i| (v[i] - max).exp())
            .collect::<Vec<_>>();
        let total = e.iter().sum::<f64>();
        let mut cumulative = 0.0;
        let kept = e
            .iter()
            .enumerate()
            .map(|(rank, w)| {
                cumulative += w / total;
                rank == 0
                    || (rank < policy.top_k.unwrap_or(usize::MAX)
                        && cumulative <= f64::from(policy.top_p))
            })
            .collect::<Vec<_>>();
        let kept_total = e
            .iter()
            .zip(&kept)
            .filter(|(_, k)| **k)
            .map(|(w, _)| w)
            .sum::<f64>();
        let mut best = (order[0], f64::NEG_INFINITY);
        for ((&candidate, w), k) in order.iter().zip(&e).zip(&kept) {
            let q = if *k { w / kept_total } else { 0.0 };
            let score = q / f64::from(noise[candidate]);
            if score > best.1 {
                best = (candidate, score);
            }
        }
        best.0
    }

    /// Logits whose softmax at temperature 1 is `probabilities`.
    fn logits_of(probabilities: &[f64]) -> Vec<f32> {
        probabilities.iter().map(|p| p.ln() as f32).collect()
    }

    /// Noise that makes `favoured` win any race it stays in: a tiny
    /// exponential for it, ordinary ones elsewhere.
    fn favouring(n: usize, favoured: usize) -> Vec<f32> {
        (0..n)
            .map(|i| if i == favoured { 1e-30 } else { 1.0 })
            .collect()
    }

    /// The candidates a policy can ever pick: each one favoured in turn.
    fn reachable(logits: &[f32], policy: &FishDrama3DrawPolicy, first: usize) -> Vec<usize> {
        (0..logits.len())
            .filter(|&i| race(logits, policy, first, &favouring(logits.len(), i)).unwrap() == i)
            .collect()
    }

    /// Top-p reads the tempered distribution, includes the token reaching
    /// exactly top-p, drops the token crossing it, and always keeps rank 0.
    #[test]
    fn the_token_crossing_top_p_is_dropped_and_rank_0_always_stays() {
        // Candidate order is not rank order.
        let logits = logits_of(&[0.15, 0.45, 0.10, 0.30]);
        // Cumulative 0.45, 0.75, 0.90, 1.00: the third rank crosses 0.8.
        // (0.90 itself lands a hair over F32 0.9, so upstream drops it too.)
        assert_eq!(reachable(&logits, &policy(1.0, 0.8, None), 0), [1, 3]);
        assert_eq!(reachable(&logits, &policy(1.0, 0.91, None), 0), [0, 1, 3]);
        assert_eq!(reachable(&logits, &policy(1.0, 1.0, None), 0), [0, 1, 2, 3]);
        // Rank 0 alone crosses a small top-p and still stays.
        assert_eq!(reachable(&logits, &policy(1.0, 0.01, None), 0), [1]);
        // A token reaching exactly top-p stays: four equal logits, top-p 0.5.
        let flat = vec![0.0f32; 4];
        assert_eq!(reachable(&flat, &policy(1.0, 0.5, None), 0), [0, 1]);
        // Tempered: at T 0.5 the probabilities go as p², so the cumulative
        // at rank 1 is (.2025 + .09) / .325 = 0.90 and top-p 0.8 keeps rank
        // 0 only; at T 2 they go as sqrt(p), the cumulative runs .349, .634,
        // .836, and top-p 0.85 keeps three ranks where 0.8 keeps two.
        assert_eq!(reachable(&logits, &policy(0.5, 0.8, None), 0), [1]);
        assert_eq!(reachable(&logits, &policy(2.0, 0.85, None), 0), [0, 1, 3]);
        assert_eq!(reachable(&logits, &policy(2.0, 0.8, None), 0), [1, 3]);
        // The kept weights renormalise: the race picks in proportion.
        let (order, weights) = kept(&logits, &policy(1.0, 0.8, None), 0).unwrap();
        assert_eq!(order, [1, 3]);
        assert!((weights[1] / weights[0] - 0.30 / 0.45).abs() < 1e-6);
    }

    #[test]
    fn top_k_caps_the_ranks_and_the_temperature_has_a_floor() {
        let logits = xorshift_values(64, 3, 4.0);
        let mut ranked = (0..64).collect::<Vec<usize>>();
        ranked.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        for k in [1usize, 2, 5, 30] {
            let mut top = ranked[..k].to_vec();
            top.sort_unstable();
            assert_eq!(reachable(&logits, &policy(1.0, 1.0, Some(k)), 0), top);
        }
        // The cap and top-p together: whichever ends the prefix first.
        let both = reachable(&logits, &policy(1.0, 0.3, Some(30)), 0);
        assert!(both.len() < 30 && !both.is_empty());
        // Temperature 0 and anything under 1e-5 are 1e-5: greedy here,
        // whatever the noise.
        for temperature in [0.0, 1e-9, MIN_TEMPERATURE] {
            assert_eq!(
                reachable(&logits, &policy(temperature, 0.9, None), 0),
                [ranked[0]],
                "{temperature}"
            );
        }
        let noise = (0..64).map(|i| 0.01 + i as f32 * 0.1).collect::<Vec<_>>();
        let floor = race(&logits, &policy(0.0, 1.0, None), 0, &noise).unwrap();
        let tiny = race(&logits, &policy(MIN_TEMPERATURE, 1.0, None), 0, &noise).unwrap();
        assert_eq!(floor, tiny);
        // `first` drops the leading candidates entirely.
        let top = ranked[0];
        assert!(!reachable(&logits, &policy(1.0, 1.0, None), top + 1).contains(&top));
    }

    /// The F32 sampler and the plain F64 one pick the same candidate under
    /// the same noise, over many rows and policies (the default, the RAS
    /// policy, low and high temperatures, tight and loose caps).
    #[test]
    fn the_sampler_agrees_with_the_f64_reference_under_identical_noise() {
        let policies = [
            policy(1.0, 0.9, None),
            FishDrama3DrawPolicy::RAS,
            policy(1.0, 0.9, Some(30)),
            policy(0.7, 0.7, Some(30)),
            policy(0.3, 1.0, None),
            policy(1.6, 0.5, Some(4)),
            policy(0.0, 0.9, None),
        ];
        let mut sampler = FishDrama3RowSampler::new(
            FishDrama3SamplingParams::from_request(None, None, None, 41).unwrap(),
            ids(),
            &[],
        )
        .unwrap();
        let mut picks = std::collections::BTreeSet::new();
        for case in 0..300u64 {
            let n = [7usize, 64, 4097][case as usize % 3];
            let logits = xorshift_values(n, 1000 + case, 2.0 + case as f32 % 5.0);
            let noise = sampler.exponentials(n).unwrap();
            for (p, policy) in policies.iter().enumerate() {
                for first in [0, 1] {
                    let ours = race(&logits, policy, first, &noise).unwrap();
                    assert_eq!(
                        ours,
                        reference(&logits, policy, first, &noise),
                        "case {case} policy {p} first {first}"
                    );
                    picks.insert(ours);
                }
            }
        }
        // The races were not all decided by rank 0.
        assert!(picks.len() > 100);
    }

    /// Draws land on candidates in the proportions of the kept, renormalised
    /// tempered distribution.
    #[test]
    fn draws_follow_the_kept_distribution() {
        let logits = vec![2.0f32, 1.5, 1.0, 0.2, -1.0, -3.0];
        let draw = policy(0.8, 0.95, Some(4));
        let (order, weights) = kept(&logits, &draw, 0).unwrap();
        let total = weights.iter().sum::<f32>();
        let mut sampler = FishDrama3RowSampler::new(
            FishDrama3SamplingParams::from_request(None, None, None, 99).unwrap(),
            ids(),
            &[],
        )
        .unwrap();
        let draws = 8000;
        let mut counts = [0usize; 6];
        for _ in 0..draws {
            let noise = sampler.exponentials(6).unwrap();
            counts[race(&logits, &draw, 0, &noise).unwrap()] += 1;
        }
        for (candidate, weight) in order.iter().zip(&weights) {
            let observed = counts[*candidate] as f32 / draws as f32;
            let expected = weight / total;
            assert!(
                (observed - expected).abs() < 0.02,
                "{candidate}: {observed} vs {expected}"
            );
        }
        let outside = (0..6)
            .filter(|c| !order.contains(c))
            .map(|c| counts[c])
            .sum::<usize>();
        assert_eq!(outside, 0);
    }

    #[test]
    fn the_request_top_k_caps_both_and_defaults_split() {
        let d = FishDrama3SamplingParams::from_request(None, None, None, 7).unwrap();
        assert_eq!(d.main, policy(1.0, 0.9, None));
        assert_eq!(d.heads, policy(1.0, 0.9, Some(30)));
        assert_eq!(d.seed, 7);
        let capped =
            FishDrama3SamplingParams::from_request(Some(0.7), Some(0.7), Some(12), 7).unwrap();
        assert_eq!(capped.main, policy(0.7, 0.7, Some(12)));
        assert_eq!(capped.heads, capped.main);
        // Validation.
        for (temperature, top_p, top_k) in [
            (Some(-0.1), None, None),
            (Some(f32::NAN), None, None),
            (Some(f32::INFINITY), None, None),
            (None, Some(0.0), None),
            (None, Some(-0.5), None),
            (None, Some(1.01), None),
            (None, Some(f32::NAN), None),
            (None, None, Some(0)),
        ] {
            assert!(
                FishDrama3SamplingParams::from_request(temperature, top_p, top_k, 1).is_err(),
                "{temperature:?} {top_p:?} {top_k:?}"
            );
        }
        assert!(FishDrama3SamplingParams::from_request(Some(0.0), Some(1.0), Some(1), 1).is_ok());
        let mut bad = d;
        bad.heads.top_k = Some(0);
        assert!(FishDrama3RowSampler::new(bad, ids(), &[]).is_err());
    }

    /// Main logits over eos and the band: `favoured` (a candidate index)
    /// far ahead, the rest flat.
    fn main_logits(favoured: usize) -> Vec<f32> {
        let mut logits = vec![0.0f32; 1 + ids().semantic_codes()];
        logits[favoured] = 50.0;
        logits
    }

    fn sampler(prompt: &[u32], seed: u64) -> FishDrama3RowSampler {
        FishDrama3RowSampler::new(
            FishDrama3SamplingParams::from_request(None, None, None, seed).unwrap(),
            ids(),
            prompt,
        )
        .unwrap()
    }

    /// The window holds the prompt's last 16 main ids; only a semantic
    /// first draw found there is redrawn, eos never is, and every drawn
    /// token joins the window.
    #[test]
    fn ras_redraws_semantic_repeats_in_the_window_over_prompt_rows() {
        let ids = ids();
        let s = |code: u32| ids.semantic(code).unwrap();
        // A reference frame 17 rows from the end (outside), one 16 rows from
        // the end (inside), the user turn's eos, then text.
        let mut prompt = vec![248_045, s(7), s(9), 8678, 198];
        prompt.extend([248_046, 198, 248_045, 846, 198, 248_083, 21_251]);
        prompt.extend([248_046, 198, 248_045, 74_455, 198, ids.voice]);
        assert_eq!(prompt.len(), 18);
        let mut row = sampler(&prompt, 1);
        assert_eq!(row.window(), prompt[2..].to_vec());

        // The first draw is `<|semantic:9|>`, in the window: redrawn.
        let draw = row.sample_main(&main_logits(10), true).unwrap();
        assert_eq!((draw.first, draw.redrawn), (s(9), true));
        assert_eq!(draw.token, draw.retry);
        // RAS at top-k 30 over a row where one logit leads by 50 also lands
        // on it; the decision is what matters here.
        assert_eq!(draw.retry, s(9));
        assert_eq!(*row.window().last().unwrap(), s(9));
        assert_eq!(row.window().len(), RAS_WINDOW);

        // `<|semantic:7|>` fell out of the window (17 rows back): kept.
        let mut row = sampler(&prompt, 1);
        let draw = row.sample_main(&main_logits(8), true).unwrap();
        assert_eq!((draw.token, draw.redrawn), (s(7), false));

        // Eos sits in the window (the user turn's) and is never redrawn.
        let mut row = sampler(&prompt, 1);
        let draw = row.sample_main(&main_logits(0), true).unwrap();
        assert_eq!(
            (draw.first, draw.token, draw.redrawn),
            (ids.eos, ids.eos, false)
        );

        // Over a rollout the rule holds draw by draw against the window as
        // it stood: one favourite code over an otherwise flat band.
        let mut row = sampler(&[s(3)], 5);
        let mut logits = vec![0.0f32; 1 + ids.semantic_codes()];
        logits[4] = 8.0;
        let mut redrawn = 0;
        for _ in 0..64 {
            let window = row.window();
            let draw = row.sample_main(&logits, true).unwrap();
            assert_eq!(
                draw.redrawn,
                ids.is_semantic(draw.first) && window.contains(&draw.first)
            );
            let expected = if draw.redrawn { draw.retry } else { draw.first };
            assert_eq!(draw.token, expected);
            redrawn += usize::from(draw.redrawn);
        }
        assert!(redrawn > 0);
    }

    /// The first frame leaves eos out of both draws; later frames may stop.
    #[test]
    fn the_first_frame_masks_eos() {
        let ids = ids();
        let mut row = sampler(&[ids.voice], 3);
        let logits = main_logits(0);
        for _ in 0..16 {
            let draw = row.sample_main(&logits, false).unwrap();
            assert!(ids.is_semantic(draw.token));
            assert!(ids.is_semantic(draw.first) && ids.is_semantic(draw.retry));
        }
        assert_eq!(row.sample_main(&logits, true).unwrap().token, ids.eos);
        // The mask moves no noise: the stream after a masked draw is where
        // it is after an unmasked one.
        let mut masked = sampler(&[ids.voice], 4);
        let mut open = sampler(&[ids.voice], 4);
        masked.sample_main(&main_logits(5), false).unwrap();
        open.sample_main(&main_logits(5), true).unwrap();
        assert_eq!(
            masked.exponentials(8).unwrap(),
            open.exponentials(8).unwrap()
        );
    }

    /// Recorded noise replays exactly: each main draw takes two rows of
    /// exponentials (first, then RAS) and each head one, in candidate order,
    /// and the tokens and RAS decisions are what the F64 reference picks
    /// under that noise.
    #[test]
    fn recorded_noise_replays_tokens_and_ras_decisions() {
        let ids = ids();
        let main = 1 + ids.semantic_codes();
        let codes = 4096;
        let params = FishDrama3SamplingParams::from_request(None, None, None, 0).unwrap();
        let frames = 6;
        let noise = xorshift_values(frames * (2 * main + 3 * codes), 9, 1.0)
            .into_iter()
            .map(|u| -((u.abs() * 0.999 + 0.0005).ln()))
            .collect::<Vec<_>>();
        let prompt = [ids.voice];
        let replay = || {
            let mut row =
                FishDrama3RowSampler::with_recorded_noise(params, ids, &prompt, noise.clone())
                    .unwrap();
            let mut out = Vec::new();
            for frame in 0..frames {
                // A peaked row whose favourite repeats, so RAS fires.
                let mut logits = xorshift_values(main, 50 + frame as u64, 1.0);
                logits[1 + (frame % 2) * 3] += 8.0;
                let draw = row.sample_main(&logits, frame > 0).unwrap();
                let heads = (0..3)
                    .map(|h| {
                        row.sample_code(&xorshift_values(codes, 70 + h + frame as u64, 3.0))
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                out.push((draw, heads));
            }
            assert!(row.exponentials(1).is_err(), "the stream is used up");
            out
        };
        let once = replay();
        assert_eq!(once, replay());
        // The same picks from the reference over the same slices.
        let recorded: &[f32] = &noise;
        let mut cursor = 0;
        let mut take = |n: usize| {
            cursor += n;
            &recorded[cursor - n..cursor]
        };
        let mut window = vec![ids.voice];
        let token = |index: usize| {
            if index == 0 {
                ids.eos
            } else {
                ids.semantic_start + index as u32 - 1
            }
        };
        let mut redraws = 0;
        for (frame, (draw, heads)) in once.iter().enumerate() {
            let mut logits = xorshift_values(main, 50 + frame as u64, 1.0);
            logits[1 + (frame % 2) * 3] += 8.0;
            let first_candidate = usize::from(frame == 0);
            let first = token(reference(
                &logits,
                &params.main,
                first_candidate,
                take(main),
            ));
            let retry = token(reference(
                &logits,
                &FishDrama3DrawPolicy::RAS,
                first_candidate,
                take(main),
            ));
            let redrawn = ids.is_semantic(first) && window.contains(&first);
            redraws += usize::from(redrawn);
            assert_eq!(
                *draw,
                FishDrama3MainDraw {
                    token: if redrawn { retry } else { first },
                    first,
                    retry,
                    redrawn,
                },
                "frame {frame}"
            );
            window.push(draw.token);
            for (h, code) in heads.iter().enumerate() {
                let logits = xorshift_values(codes, 70 + h as u64 + frame as u64, 3.0);
                assert_eq!(
                    *code as usize,
                    reference(&logits, &params.heads, 0, take(codes))
                );
            }
        }
        assert!(redraws > 0, "the rollout exercises RAS");
    }

    /// A checkpoint restores the noise stream and the window bit for bit;
    /// one seed is one stream, and another seed another.
    #[test]
    fn a_checkpoint_restores_the_stream_and_window_exactly() {
        let ids = ids();
        let logits = xorshift_values(1 + ids.semantic_codes(), 4, 2.0);
        let heads = xorshift_values(4096, 5, 2.0);
        let mut row = sampler(&[ids.voice], 77);
        row.sample_main(&logits, false).unwrap();
        let checkpoint = row.checkpoint();
        let run = |row: &mut FishDrama3RowSampler| {
            let mut out = Vec::new();
            for _ in 0..5 {
                out.push(row.sample_main(&logits, true).unwrap().token);
                out.push(row.sample_code(&heads).unwrap());
            }
            (out, row.exponentials(16).unwrap(), row.window())
        };
        let first = run(&mut row);
        let after = row.checkpoint();
        row.restore(checkpoint.clone());
        assert_eq!(row.checkpoint(), checkpoint);
        let again = run(&mut row);
        assert_eq!(first.0, again.0);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&first.1), bits(&again.1));
        assert_eq!(first.2, again.2);
        assert_eq!(row.checkpoint(), after);
        // Determinism from the seed alone, and seeds differ.
        let mut a = sampler(&[], 77);
        let mut b = sampler(&[], 77);
        let mut c = sampler(&[], 78);
        let (na, nb, nc) = (
            a.exponentials(64).unwrap(),
            b.exponentials(64).unwrap(),
            c.exponentials(64).unwrap(),
        );
        assert_eq!(bits(&na), bits(&nb));
        assert_ne!(bits(&na), bits(&nc));
        assert!(na.iter().all(|e| e.is_finite() && *e > 0.0));
        let mean = na.iter().map(|e| f64::from(*e)).sum::<f64>() / 64.0;
        assert!((0.5..2.0).contains(&mean), "{mean}");
    }

    /// Non-finite logits are refused anywhere in the row, the masked eos
    /// included, and a refused draw moves nothing.
    #[test]
    fn non_finite_logits_are_refused() {
        let ids = ids();
        let mut row = sampler(&[ids.voice], 8);
        let before = row.checkpoint();
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut logits = main_logits(5);
            logits[0] = bad;
            let error = row.sample_main(&logits, false).unwrap_err().to_string();
            assert!(error.contains("main logits are non-finite"), "{error}");
            let mut heads = vec![0.0f32; 4096];
            heads[4095] = bad;
            let error = row.sample_code(&heads).unwrap_err().to_string();
            assert!(
                error.contains("codebook head logits are non-finite"),
                "{error}"
            );
        }
        assert_eq!(row.checkpoint(), before);
        // The wrong width, and the race's own guards.
        assert!(row.sample_main(&[0.0; 4], true).is_err());
        assert!(race(&[0.0, 1.0], &policy(1.0, 0.9, None), 0, &[1.0]).is_err());
        assert!(race(&[0.0, 1.0], &policy(1.0, 0.9, None), 2, &[1.0, 1.0]).is_err());
    }
}
