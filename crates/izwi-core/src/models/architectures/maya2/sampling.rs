//! Maya2's samplers, as upstream's engine runs them (`maya3/engine.py:652-694`,
//! `serve/fast_depth_kernels.py`).
//!
//! - **cb0** (host, per row): logits / T; the loop guard's end-of-audio
//!   bias; a presence penalty for each distinct code of the last frames
//!   (end of audio exempt); top-k over all 2,049 classes keeping ties at the
//!   k-th value; optional top-p; softmax. The controlled stop reads that
//!   distribution, then end of audio is removed and a code drawn from the
//!   rest.
//! - **Depth** (device, all rows at once): logits / T, top-k keeping ties,
//!   Gumbel-max with noise from each row's own host stream, so a frame's 31
//!   levels need no read-back until the frame is done.
//!
//! Every draw comes from the request's own SplitMix64 streams, never from
//! the batch: a row samples the same whoever shares its step (upstream's
//! output changes with batch composition).

use candle_core::{DType, Device, Tensor, D};

use super::config::{
    Maya2SamplingDefaults, AEOS_CLASS, CB0_CLASSES, CODEBOOK_SIZE, MAYA2_DEFAULTS,
    PRESENCE_PENALTY_RANGE, STOP_HAZARD, STOP_TAU, TEMPERATURE_RANGE,
};
use super::guards::Maya2StopReason;
use super::runtime::Maya2Stream;
use crate::error::{Error, Result};
use crate::kernels::counter_noise;
use crate::kernels::penalized_topk::{CandidatePenalty, CandidateRow, MAX_CANDIDATES};
use crate::models::shared::noise::{mix, stream_seed, SplitMix64};

/// A request's cb0 sampler settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Maya2SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    /// Subtracted from the logit of each code sampled in the last
    /// `presence_window` frames.
    pub presence_penalty: f32,
}

impl Default for Maya2SamplingParams {
    fn default() -> Self {
        let Maya2SamplingDefaults {
            temperature,
            top_k,
            top_p,
            presence_penalty,
            ..
        } = MAYA2_DEFAULTS;
        Self {
            temperature,
            top_k,
            top_p,
            presence_penalty,
        }
    }
}

impl Maya2SamplingParams {
    pub fn validate(&self) -> Result<()> {
        let in_range =
            |value: f32, (lo, hi): (f32, f32)| value.is_finite() && (lo..=hi).contains(&value);
        if !in_range(self.temperature, TEMPERATURE_RANGE) {
            return Err(Error::InvalidInput(format!(
                "Maya2 temperature must be in {}..={}",
                TEMPERATURE_RANGE.0, TEMPERATURE_RANGE.1
            )));
        }
        if self.top_k == 0 || self.top_k > CB0_CLASSES {
            return Err(Error::InvalidInput(format!(
                "Maya2 top_k must be in 1..={CB0_CLASSES}"
            )));
        }
        if !(self.top_p.is_finite() && self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err(Error::InvalidInput("Maya2 top_p must be in (0, 1]".into()));
        }
        if !in_range(self.presence_penalty, PRESENCE_PENALTY_RANGE) {
            return Err(Error::InvalidInput(format!(
                "Maya2 repetition (presence) penalty must be in {}..={}",
                PRESENCE_PENALTY_RANGE.0, PRESENCE_PENALTY_RANGE.1
            )));
        }
        Ok(())
    }
}

/// Per-piece state of the cb0 sampler and the controlled stop.
#[derive(Debug, Clone, PartialEq)]
pub struct Cb0State {
    /// cb0 codes of the last frames, oldest first.
    recent: Vec<u32>,
    /// 1 − Π(1 − p(end of audio)) since the stop became allowed.
    hazard: f32,
}

impl Default for Cb0State {
    fn default() -> Self {
        Self {
            recent: Vec::with_capacity(MAYA2_DEFAULTS.presence_window),
            hazard: 0.0,
        }
    }
}

/// One cb0 draw.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cb0Draw {
    /// The frame's codebook-0 code (never end of audio).
    pub code: u32,
    /// Set when the controlled stop fires: this frame is not emitted.
    pub stop: Option<Maya2StopReason>,
    pub p_aeos: f32,
}

/// Draw cb0 from `logits` (2,049 raw classes) after `frames` frames, the
/// stop allowed from `minimum` frames on.
pub fn sample_cb0(
    logits: &[f32],
    params: &Maya2SamplingParams,
    aeos_bias: f32,
    frames: usize,
    minimum: usize,
    state: &mut Cb0State,
    rng: &mut SplitMix64,
) -> Result<Cb0Draw> {
    if logits.len() != CB0_CLASSES {
        return Err(Error::InferenceError(format!(
            "Maya2 cb0 head gave {} logits, expected {CB0_CLASSES}",
            logits.len()
        )));
    }
    if let Some(bad) = logits.iter().find(|value| !value.is_finite()) {
        return Err(Error::InferenceError(format!(
            "Maya2 cb0 logits are not finite ({bad})"
        )));
    }
    let inverse = 1.0 / params.temperature;
    let mut scaled = logits.iter().map(|&l| l * inverse).collect::<Vec<_>>();
    scaled[AEOS_CLASS] += aeos_bias;
    if params.presence_penalty > 0.0 {
        let mut seen = [false; CODEBOOK_SIZE];
        for &code in &state.recent {
            seen[code as usize] = true;
        }
        for (logit, seen) in scaled.iter_mut().zip(seen) {
            if seen {
                *logit -= params.presence_penalty;
            }
        }
    }
    // Top-k with ties at the k-th value kept.
    let mut sorted = scaled.clone();
    sorted.sort_unstable_by(|a, b| b.total_cmp(a));
    let kth = sorted[params.top_k.min(CB0_CLASSES) - 1];
    let mut kept = scaled
        .iter()
        .map(|&l| if l >= kth { l } else { f32::NEG_INFINITY })
        .collect::<Vec<_>>();
    let mut probs = softmax(&kept);
    if params.top_p < 1.0 {
        // Drop a class once the mass ranked above it exceeds top_p.
        let mut order = (0..CB0_CLASSES).collect::<Vec<_>>();
        order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
        let mut before = 0.0f32;
        for &class in &order {
            if before > params.top_p {
                kept[class] = f32::NEG_INFINITY;
            }
            before += probs[class];
        }
        probs = softmax(&kept);
    }
    let p_aeos = probs[AEOS_CLASS];
    let argmax = probs
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, &p)| {
            if p > best.1 {
                (i, p)
            } else {
                best
            }
        })
        .0;
    let stop = controlled_stop(state, p_aeos, argmax == AEOS_CLASS, frames, minimum);
    // Draw from the distribution without end of audio. When end of audio
    // held all the mass, take the best other class.
    let mass = probs[..AEOS_CLASS].iter().sum::<f32>();
    let code = if mass > 0.0 {
        let target = rng.next_open01() as f32 * mass;
        let mut cumulative = 0.0f32;
        let mut pick = None;
        for (class, &p) in probs[..AEOS_CLASS].iter().enumerate() {
            if p <= 0.0 {
                continue;
            }
            cumulative += p;
            pick = Some(class);
            if cumulative >= target {
                break;
            }
        }
        pick.unwrap_or(0)
    } else {
        scaled[..AEOS_CLASS]
            .iter()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |best, (i, &l)| {
                if l > best.1 {
                    (i, l)
                } else {
                    best
                }
            })
            .0
    } as u32;
    state.remember(code);
    Ok(Cb0Draw { code, stop, p_aeos })
}

/// The controlled stop after `frames` frames (allowed from `minimum`),
/// accumulating the hazard.
fn controlled_stop(
    state: &mut Cb0State,
    p_aeos: f32,
    aeos_argmax: bool,
    frames: usize,
    minimum: usize,
) -> Option<Maya2StopReason> {
    if frames < minimum {
        return None;
    }
    state.hazard = 1.0 - (1.0 - state.hazard) * (1.0 - p_aeos);
    if aeos_argmax {
        Some(Maya2StopReason::AeosArgmax)
    } else if p_aeos >= STOP_TAU {
        Some(Maya2StopReason::AeosTau)
    } else if state.hazard >= STOP_HAZARD {
        Some(Maya2StopReason::AeosHazard)
    } else {
        None
    }
}

impl Cb0State {
    fn remember(&mut self, code: u32) {
        if self.recent.len() == MAYA2_DEFAULTS.presence_window {
            self.recent.remove(0);
        }
        self.recent.push(code);
    }

    /// The distinct codes the presence penalty reads, ascending.
    pub fn recent_codes(&self) -> Vec<u32> {
        let mut codes = self.recent.clone();
        codes.sort_unstable();
        codes.dedup();
        codes
    }
}

/// The device's share of a cb0 draw: the logits scaled by `1 / T`, the
/// end-of-audio bias and the presence penalty over `recent` (the state's
/// [`Cb0State::recent_codes`]), exactly as [`sample_cb0`] writes them before
/// top-k, and how many of the largest to return. `None` when top-k is wider
/// than the candidates a row returns: that row reads its whole logits.
pub(crate) fn cb0_device_candidates<'a>(
    params: &Maya2SamplingParams,
    aeos_bias: f32,
    recent: &'a [u32],
) -> Option<(CandidateRow<'a>, usize)> {
    let count = (params.top_k + 1).next_power_of_two().max(64);
    if params.top_k == 0 || count > MAX_CANDIDATES || count >= CB0_CLASSES {
        return None;
    }
    let penalty = if params.presence_penalty > 0.0 {
        CandidatePenalty::Presence(params.presence_penalty)
    } else {
        CandidatePenalty::None
    };
    Some((
        CandidateRow {
            scale: 1.0 / params.temperature,
            bias: Some((AEOS_CLASS, aeos_bias)),
            penalty,
            penalized: recent,
            ..CandidateRow::plain(0)
        },
        count,
    ))
}

/// [`sample_cb0`] from a row's device candidates (`values` its scaled,
/// biased and penalised logits, largest first, at classes `indices`): the
/// same draw, stop and stream use, over the kept classes only. `None`, with
/// nothing consumed, when the candidates cannot show that no class past
/// them survives top-k (the last ties the k-th), or when end of audio holds
/// all the mass (the fallback then reads every other class).
pub fn sample_cb0_candidates(
    values: &[f32],
    indices: &[u32],
    params: &Maya2SamplingParams,
    frames: usize,
    minimum: usize,
    state: &mut Cb0State,
    rng: &mut SplitMix64,
) -> Option<Cb0Draw> {
    let k = params.top_k.min(CB0_CLASSES);
    if k == 0 || k >= values.len() || values.len() != indices.len() {
        return None;
    }
    let mut sorted = values.to_vec();
    let (_, kth, _) = sorted.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
    let kth = *kth;
    if !values.last()?.lt(&kth) {
        return None;
    }
    // The kept classes in class order: every other class is -inf.
    let mut kept = indices
        .iter()
        .zip(values)
        .filter(|&(_, &l)| l >= kth)
        .map(|(&class, &l)| (class as usize, l))
        .collect::<Vec<_>>();
    kept.sort_by_key(|&(class, _)| class);
    let sparse_softmax = |kept: &[(usize, f32)]| {
        let max = kept
            .iter()
            .map(|&(_, l)| l)
            .fold(f32::NEG_INFINITY, f32::max);
        let exp = kept
            .iter()
            .map(|&(_, l)| if l.is_finite() { (l - max).exp() } else { 0.0 })
            .collect::<Vec<_>>();
        let sum = exp.iter().sum::<f32>();
        exp.into_iter().map(|e| e / sum).collect::<Vec<_>>()
    };
    let mut probs = sparse_softmax(&kept);
    if params.top_p < 1.0 {
        let mut order = (0..kept.len()).collect::<Vec<_>>();
        order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
        let mut before = 0.0f32;
        for &slot in &order {
            if before > params.top_p {
                kept[slot].1 = f32::NEG_INFINITY;
            }
            before += probs[slot];
        }
        probs = sparse_softmax(&kept);
    }
    let p_aeos = kept
        .iter()
        .position(|&(class, _)| class == AEOS_CLASS)
        .map_or(0.0, |slot| probs[slot]);
    let argmax = kept
        .iter()
        .zip(&probs)
        .fold((0, f32::NEG_INFINITY), |best, (&(class, _), &p)| {
            if p > best.1 {
                (class, p)
            } else {
                best
            }
        })
        .0;
    let codes = kept
        .iter()
        .zip(&probs)
        .filter(|(&(class, _), _)| class < AEOS_CLASS)
        .map(|(&(class, _), &p)| (class, p))
        .collect::<Vec<_>>();
    let mass = codes.iter().map(|&(_, p)| p).sum::<f32>();
    if !mass.gt(&0.0) {
        return None;
    }
    let stop = controlled_stop(state, p_aeos, argmax == AEOS_CLASS, frames, minimum);
    let target = rng.next_open01() as f32 * mass;
    let mut cumulative = 0.0f32;
    let mut pick = None;
    for &(class, p) in &codes {
        if p <= 0.0 {
            continue;
        }
        cumulative += p;
        pick = Some(class);
        if cumulative >= target {
            break;
        }
    }
    let code = pick.unwrap_or(0) as u32;
    state.remember(code);
    Some(Cb0Draw { code, stop, p_aeos })
}

/// Whether the depth decoder's noise is generated on `device` from each
/// row's seed and frame ([`depth_counter_gumbel`]): on CUDA, unless
/// `IZWI_MAYA2_DEVICE_NOISE=false`. Elsewhere each row's host stream draws it
/// and it is uploaded ([`depth_gumbel`]).
pub(crate) fn device_noise(device: &Device) -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    device.is_cuda()
        && *ENABLED.get_or_init(|| crate::env::flag_or("IZWI_MAYA2_DEVICE_NOISE", true))
}

/// Whether cb0 draws read device candidates ([`sample_cb0_candidates`]): on
/// CUDA, unless `IZWI_MAYA2_DEVICE_SAMPLING=false` or the candidate kernel's
/// one-time check failed.
pub(crate) fn device_cb0_sampling(device: &Device) -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    device.is_cuda()
        && *ENABLED.get_or_init(|| crate::env::flag_or("IZWI_MAYA2_DEVICE_SAMPLING", true))
        && crate::kernels::penalized_topk::device_path::ready(device)
}

/// The depth noise key of a request's frame `frame`: its depth stream mixed
/// with the frame index, so a frame's noise is the same however it is
/// batched or retried.
pub fn depth_noise_key(seed: u64, frame: usize) -> u64 {
    mix(stream_seed(seed, Maya2Stream::Depth.into()) ^ mix(frame as u64 + 1))
}

/// [`depth_gumbel`]'s `[rows, levels, codes]` from counter noise generated
/// on `device` (`kernels::counter_noise`), row `r` keyed `keys[r]`
/// ([`depth_noise_key`]): one 8-byte key a row goes up instead of the
/// noise.
pub fn depth_counter_gumbel(
    keys: &[u64],
    levels: usize,
    codes: usize,
    device: &Device,
) -> Result<Tensor> {
    Ok(
        counter_noise::gumbel(keys, levels * codes, device)?.reshape((
            keys.len(),
            levels,
            codes,
        ))?,
    )
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp = logits
        .iter()
        .map(|&l| if l.is_finite() { (l - max).exp() } else { 0.0 })
        .collect::<Vec<_>>();
    let sum = exp.iter().sum::<f32>();
    exp.into_iter().map(|e| e / sum).collect()
}

/// Gumbel noise `[rows, levels, codes]` on `device` for one frame: each
/// row's uniforms come from its own stream, so a row's draws never depend on
/// who shares the step.
pub fn depth_gumbel(
    rngs: &mut [&mut SplitMix64],
    levels: usize,
    codes: usize,
    device: &Device,
) -> Result<Tensor> {
    let per_row = levels * codes;
    let mut uniforms = Vec::with_capacity(rngs.len() * per_row);
    for rng in rngs.iter_mut() {
        uniforms.extend(rng.uniforms(per_row));
    }
    let u = Tensor::from_vec(uniforms, (rngs.len(), levels, codes), device)?
        // An F32 uniform can round to 1; keep −ln(−ln u) finite.
        .clamp(1e-12f32, 1.0f32 - f32::EPSILON / 2.0)?;
    Ok(u.log()?.neg()?.log()?.neg()?)
}

/// One depth level for every row: logits `[rows, 2048]` scaled by
/// `1 / temperature`, top-k keeping ties, Gumbel-max with `gumbel`
/// `[rows, 2048]`. Codes `[rows]` (U32), left on the device.
pub fn sample_depth_level(
    logits: &Tensor,
    temperature: f32,
    top_k: usize,
    gumbel: &Tensor,
) -> Result<Tensor> {
    let scaled = (logits.to_dtype(DType::F32)? * (1.0 / f64::from(temperature)))?;
    let kth = kth_largest(&scaled, top_k)?;
    let keep = scaled.broadcast_ge(&kth)?;
    let floor = Tensor::full(f32::NEG_INFINITY, scaled.shape(), scaled.device())?;
    let score = keep.where_cond(&(scaled + gumbel)?, &floor)?;
    Ok(score.argmax(D::Minus1)?.to_dtype(DType::U32)?)
}

/// The k-th largest value of each row of `x` `[rows, cols]`, `[rows, 1]`.
/// Rows wider than the device sort bound are sorted in halves first (the
/// k-th largest of a row is among the k largest of whichever half holds it).
fn kth_largest(x: &Tensor, k: usize) -> Result<Tensor> {
    const SORT_CHUNK: usize = 1024;
    let (rows, cols) = x.dims2()?;
    let k = k.clamp(1, cols);
    let candidates = if cols > SORT_CHUNK && cols.is_multiple_of(SORT_CHUNK) && k <= SORT_CHUNK {
        let chunks = cols / SORT_CHUNK;
        let (sorted, _) = x
            .reshape((rows * chunks, SORT_CHUNK))?
            .sort_last_dim(false)?;
        sorted.narrow(1, 0, k)?.reshape((rows, chunks * k))?
    } else {
        x.clone()
    };
    let (sorted, _) = candidates.contiguous()?.sort_last_dim(false)?;
    Ok(sorted.narrow(1, k - 1, 1)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits_with(values: &[(usize, f32)]) -> Vec<f32> {
        let mut logits = vec![-30.0f32; CB0_CLASSES];
        for &(class, value) in values {
            logits[class] = value;
        }
        logits
    }

    #[test]
    fn controlled_stop_waits_for_the_minimum() {
        let logits = logits_with(&[(AEOS_CLASS, 5.0), (3, 1.0)]);
        let params = Maya2SamplingParams::default();
        let mut rng = SplitMix64::new(1);
        let mut state = Cb0State::default();
        let early = sample_cb0(&logits, &params, 0.0, 2, 4, &mut state, &mut rng).unwrap();
        assert_eq!(early.stop, None);
        assert_eq!(early.code, 3, "end of audio is never drawn");
        let late = sample_cb0(&logits, &params, 0.0, 4, 4, &mut state, &mut rng).unwrap();
        assert_eq!(late.stop, Some(Maya2StopReason::AeosArgmax));
    }

    #[test]
    fn tau_and_hazard_rules_fire_in_order() {
        // No presence penalty: it would shift mass to end of audio as codes
        // repeat and fire the hazard early.
        let params = Maya2SamplingParams {
            temperature: 1.0,
            presence_penalty: 0.0,
            ..Default::default()
        };
        let mut rng = SplitMix64::new(2);
        // p(AEOS) ≈ 0.4 without being the argmax: the tau rule.
        let logits = logits_with(&[(AEOS_CLASS, 0.0), (5, 0.4)]);
        let mut state = Cb0State::default();
        let draw = sample_cb0(&logits, &params, 0.0, 10, 4, &mut state, &mut rng).unwrap();
        assert!(
            draw.p_aeos > STOP_TAU && draw.p_aeos < 0.5,
            "{}",
            draw.p_aeos
        );
        assert_eq!(draw.stop, Some(Maya2StopReason::AeosTau));
        // p(AEOS) = 0.25 per step accumulates hazard to ≥ 0.6 on step four.
        let logits = logits_with(&[(AEOS_CLASS, 0.0), (5, 0.0), (6, 0.0), (7, 0.0)]);
        let mut state = Cb0State::default();
        let reasons = (0..4)
            .map(|_| {
                sample_cb0(&logits, &params, 0.0, 10, 4, &mut state, &mut rng)
                    .unwrap()
                    .stop
            })
            .collect::<Vec<_>>();
        assert_eq!(reasons[..3], [None, None, None]);
        assert_eq!(reasons[3], Some(Maya2StopReason::AeosHazard));
    }

    #[test]
    fn presence_penalty_and_bias_shift_the_distribution() {
        let params = Maya2SamplingParams {
            temperature: 1.0,
            presence_penalty: PRESENCE_PENALTY_RANGE.1,
            ..Default::default()
        };
        // p(code 1) is e/(e + 1) ≈ 0.73 unpenalised, ≈ 0.018 penalised.
        let logits = logits_with(&[(1, 3.0), (2, 2.0)]);
        let mut rng = SplitMix64::new(3);
        let ones = |history: &[u32], rng: &mut SplitMix64| {
            (0..200)
                .filter(|_| {
                    let mut state = Cb0State::default();
                    state.recent.extend_from_slice(history);
                    sample_cb0(&logits, &params, 0.0, 0, 4, &mut state, rng)
                        .unwrap()
                        .code
                        == 1
                })
                .count()
        };
        assert!(ones(&[], &mut rng) > 100);
        assert!(ones(&[1], &mut rng) < 20);
        // The loop boost raises end of audio to the argmax.
        let logits = logits_with(&[(1, 3.0), (2, 2.0), (AEOS_CLASS, 0.0)]);
        let draw = sample_cb0(
            &logits,
            &params,
            20.0,
            10,
            4,
            &mut Cb0State::default(),
            &mut rng,
        )
        .unwrap();
        assert_eq!(draw.stop, Some(Maya2StopReason::AeosArgmax));
    }

    #[test]
    fn top_k_keeps_ties() {
        let params = Maya2SamplingParams {
            temperature: 1.0,
            top_k: 1,
            presence_penalty: 0.0,
            ..Default::default()
        };
        let logits = logits_with(&[(8, 2.0), (9, 2.0)]);
        let mut seen = [false; 2];
        let mut rng = SplitMix64::new(4);
        for _ in 0..64 {
            let draw = sample_cb0(
                &logits,
                &params,
                0.0,
                0,
                4,
                &mut Cb0State::default(),
                &mut rng,
            )
            .unwrap();
            seen[(draw.code - 8) as usize] = true;
        }
        assert_eq!(seen, [true, true]);
    }

    #[test]
    fn depth_levels_sample_inside_the_top_k() {
        let device = Device::Cpu;
        let rows = 3;
        let mut values = vec![0f32; rows * CODEBOOK_SIZE];
        for row in 0..rows {
            for (code, value) in values[row * CODEBOOK_SIZE..(row + 1) * CODEBOOK_SIZE]
                .iter_mut()
                .enumerate()
            {
                *value = ((code * 37 + row * 11) % 997) as f32 / 100.0;
            }
        }
        let logits = Tensor::from_vec(values.clone(), (rows, CODEBOOK_SIZE), &device).unwrap();
        let mut a = SplitMix64::new(5);
        let mut b = SplitMix64::new(6);
        let mut c = SplitMix64::new(7);
        let noise = depth_gumbel(&mut [&mut a, &mut b, &mut c], 2, CODEBOOK_SIZE, &device).unwrap();
        let codes = sample_depth_level(
            &logits,
            0.7,
            4,
            &noise.narrow(1, 0, 1).unwrap().squeeze(1).unwrap(),
        )
        .unwrap()
        .to_vec1::<u32>()
        .unwrap();
        for (row, &code) in codes.iter().enumerate() {
            let row_values = &values[row * CODEBOOK_SIZE..(row + 1) * CODEBOOK_SIZE];
            let mut sorted = row_values.to_vec();
            sorted.sort_by(|x, y| y.total_cmp(x));
            assert!(
                row_values[code as usize] >= sorted[3],
                "row {row} drew outside the top 4"
            );
        }
        // A row's draw does not depend on the rows beside it.
        let mut a2 = SplitMix64::new(5);
        let alone = depth_gumbel(&mut [&mut a2], 2, CODEBOOK_SIZE, &device).unwrap();
        let solo = sample_depth_level(
            &logits.narrow(0, 0, 1).unwrap(),
            0.7,
            4,
            &alone.narrow(1, 0, 1).unwrap().squeeze(1).unwrap(),
        )
        .unwrap()
        .to_vec1::<u32>()
        .unwrap();
        assert_eq!(solo[0], codes[0]);
    }

    /// A cb0 draw from device candidates (packed by the kernel's host
    /// reference) is the host sampler's: code, stop, p(AEOS), hazard,
    /// presence history and stream, over random logits, temperatures,
    /// top-k/top-p, presence penalties, histories and loop biases.
    #[test]
    fn cb0_candidates_draw_the_host_samplers_code() {
        use crate::kernels::penalized_topk::{candidates_packed, unpack};
        let mut source = SplitMix64::new(17);
        let (mut drawn, mut whole) = (0, 0);
        for case in 0..400usize {
            let scale = [1.0f32, 3.0, 8.0][case % 3];
            let mut logits = source
                .standard_normals(CB0_CLASSES)
                .into_iter()
                .map(|v| v * scale)
                .collect::<Vec<_>>();
            if case % 4 == 0 {
                for value in &mut logits {
                    *value = (*value * 2.0).round() / 2.0;
                }
            }
            logits[AEOS_CLASS] = [-4.0, 0.0, 3.0, 9.0][(case / 3) % 4];
            let params = Maya2SamplingParams {
                temperature: [0.5, 1.0, 1.3][(case / 2) % 3],
                top_k: [64, 1, 20, 100, 127][(case / 5) % 5],
                top_p: [1.0, 0.9, 0.5, 0.05][(case / 7) % 4],
                presence_penalty: [0.0, 0.7, PRESENCE_PENALTY_RANGE.1][case % 3],
            };
            let mut state = Cb0State::default();
            for _ in 0..(case % 13) {
                state.remember((source.next_u64() % 64) as u32);
            }
            state.hazard = [0.0, 0.3, 0.55][case % 3];
            let bias = [0.0, 2.5][case % 2];
            let frames = [0, 10][(case / 11) % 2];
            let seed = 1 + case as u64;

            let mut host_state = state.clone();
            let mut host_rng = SplitMix64::new(seed);
            let expected = sample_cb0(
                &logits,
                &params,
                bias,
                frames,
                4,
                &mut host_state,
                &mut host_rng,
            )
            .unwrap();

            let recent = state.recent_codes();
            let (row, count) = cb0_device_candidates(&params, bias, &recent).unwrap();
            let tensor = Tensor::from_vec(logits.clone(), (1, CB0_CLASSES), &Device::Cpu).unwrap();
            let packed = candidates_packed(&tensor, CB0_CLASSES, &[row], count)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<u32>()
                .unwrap();
            let candidates = unpack(&packed, count).remove(0);
            let mut rng = SplitMix64::new(seed);
            let before = (state.clone(), rng.clone());
            match sample_cb0_candidates(
                &candidates.values,
                &candidates.indices,
                &params,
                frames,
                4,
                &mut state,
                &mut rng,
            ) {
                Some(draw) => {
                    assert_eq!(draw, expected, "case {case}: {params:?}");
                    assert_eq!(state, host_state, "case {case}");
                    assert_eq!(rng.next_u64(), host_rng.next_u64(), "case {case}");
                    drawn += 1;
                }
                None => {
                    // Nothing consumed: the whole row samples as before.
                    assert_eq!(
                        (state, rng.next_u64()),
                        (before.0, before.1.clone().next_u64())
                    );
                    whole += 1;
                }
            }
        }
        assert!(drawn >= 330, "{drawn} from candidates, {whole} whole");
    }

    #[test]
    fn depth_noise_keys_follow_the_seed_and_frame() {
        assert_eq!(depth_noise_key(5, 3), depth_noise_key(5, 3));
        assert_ne!(depth_noise_key(5, 3), depth_noise_key(5, 4));
        assert_ne!(depth_noise_key(5, 3), depth_noise_key(6, 3));
        let noise = depth_counter_gumbel(&[depth_noise_key(5, 3)], 2, 8, &Device::Cpu).unwrap();
        assert_eq!(noise.dims(), [1, 2, 8]);
    }

    #[test]
    fn params_validate_their_ranges() {
        assert!(Maya2SamplingParams::default().validate().is_ok());
        let bad = Maya2SamplingParams {
            temperature: 3.0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());
        let bad = Maya2SamplingParams {
            top_k: 0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());
    }
}
