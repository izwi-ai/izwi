//! Device-resident sampling on Metal and CUDA (§6): Fish S2's resident
//! pattern (`fish_s2/sampling.rs`, `fish_s2/fast.rs`) as the family's own
//! copy, over the family's sampler (`sampling.rs`). Every step is a Candle
//! op or the shared bounded top-k (`models::shared::sampling::device_top_k`),
//! so one implementation serves both devices.
//!
//! The host draws every exponential a frame needs from the row's own
//! stream, in the order the host sampler takes them (the main draw's, the
//! RAS retry's, then one per codebook head), and uploads them. The device
//! runs each race over its own logits:
//!
//! ```text
//! ranks   = top min(cap, 512) of the row, sorted on the device
//! p       = exp(v / T - max) / sum over every candidate     # the tempered softmax
//! keep r  = r == 0 || (r < cap && cumsum(p)[r] <= top_p)    # a prefix, as the host's
//! pick    = argmax_r keep_r * q_r / E[candidate_r]           # the exponential race
//! ```
//!
//! A codebook head's pick indexes the next tower layer's injection on the
//! device, so a frame reads back twice: readback 1 is each row's main and
//! RAS picks (RAS itself, a window lookup, runs on the host), readback 2
//! the 9 head codes. A main draw whose kept prefix runs past the device's
//! 512 ranks (an uncapped flat distribution) re-runs on the host from the
//! same exponentials, which costs one more readback of that row's logits;
//! a request whose head cap exceeds 512 samples its heads on the host.
//! Picks agree with the host sampler's except where rounding splits a
//! near-tie (the device sums and exponentiates in F32 in another order).

use std::sync::atomic::{AtomicU64, Ordering};

use candle_core::{DType, Tensor};

use super::super::sampling::{FishDrama3DrawPolicy, MIN_TEMPERATURE};
use crate::error::{Error, Result};
use crate::models::shared::sampling::device_top_k;

/// The most ranks a device draw sorts (the shared bounded top-k converges
/// for at most half its 1,024-wide sort chunk).
pub(crate) const DEVICE_RANKS: usize = 512;

static RESIDENT_FRAMES: AtomicU64 = AtomicU64::new(0);
static READBACKS: AtomicU64 = AtomicU64::new(0);
static HOST_FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// Process-wide count of frames whose main token and 9 codes were sampled
/// on the device.
pub fn fish_drama3_resident_frames() -> u64 {
    RESIDENT_FRAMES.load(Ordering::Relaxed)
}

/// Process-wide count of the resident sampler's synchronizing readbacks
/// (two per step of frames, plus fallbacks).
pub fn fish_drama3_resident_readbacks() -> u64 {
    READBACKS.load(Ordering::Relaxed)
}

/// Process-wide count of draws that fell back to the host sampler (a main
/// draw past [`DEVICE_RANKS`], or a head cap above it).
pub fn fish_drama3_resident_host_fallbacks() -> u64 {
    HOST_FALLBACKS.load(Ordering::Relaxed)
}

pub(crate) fn count_frames(frames: usize) {
    RESIDENT_FRAMES.fetch_add(frames as u64, Ordering::Relaxed);
}

pub(crate) fn count_fallback() {
    HOST_FALLBACKS.fetch_add(1, Ordering::Relaxed);
}

/// Reads `tensor` (U32) back: one synchronizing readback.
pub(crate) fn read_back(tensor: &Tensor) -> Result<Vec<u32>> {
    READBACKS.fetch_add(1, Ordering::Relaxed);
    Ok(tensor.flatten_all()?.to_vec1::<u32>()?)
}

/// Whether a draw under `policy` can run on the device for any row.
pub(crate) fn device_cap(policy: &FishDrama3DrawPolicy) -> Option<usize> {
    policy.top_k.filter(|cap| *cap <= DEVICE_RANKS)
}

/// One row's ranks on the device, shared by the draws over the same logits
/// (a positive temperature keeps the order).
pub(crate) struct Ranked {
    /// The row's logits with candidates before `first` at `-inf`, `[V]` F32.
    row: Tensor,
    /// The top ranks' logits, descending, `[k]` F32.
    values: Tensor,
    /// Their candidate indices, `[k]` U32.
    indices: Tensor,
    /// The row's non-finite logits, `[1]` U32.
    non_finite: Tensor,
    /// Candidates a draw may pick: `V - first`.
    candidates: usize,
}

/// The top `ranks` candidates of `logits` (`[V]` F32 on the device) from
/// `first` on (`first` 1 drops eos from a main draw).
pub(crate) fn rank(logits: &Tensor, first: usize, ranks: usize) -> Result<Ranked> {
    let width = logits.dims1()?;
    if first >= width || ranks == 0 || ranks > DEVICE_RANKS {
        return Err(Error::InferenceError(format!(
            "a device draw of {ranks} ranks over candidates {first}..{width}"
        )));
    }
    let logits = logits.to_dtype(DType::F32)?;
    // NaN fails `x == x`; an infinity passes it and exceeds F32::MAX.
    let non_finite = (logits.ne(&logits)?.to_dtype(DType::U32)?
        + logits
            .abs()?
            .gt(f64::from(f32::MAX))?
            .to_dtype(DType::U32)?)?
    .sum_keepdim(0)?;
    let row = if first == 0 {
        logits.contiguous()?
    } else {
        Tensor::cat(
            &[
                &Tensor::full(f32::NEG_INFINITY, first, logits.device())?,
                &logits.narrow(0, first, width - first)?,
            ],
            0,
        )?
    };
    let candidates = width - first;
    let (values, indices) = device_top_k(&row, ranks.min(candidates))?.ok_or_else(|| {
        Error::InferenceError(format!("the device cannot rank {ranks} candidates"))
    })?;
    Ok(Ranked {
        row,
        values,
        indices: indices.to_dtype(DType::U32)?,
        non_finite,
        candidates,
    })
}

/// One device race.
pub(crate) struct DeviceDraw {
    /// The winning candidate, `[1]` U32.
    pub(crate) pick: Tensor,
    /// The row's non-finite logits, `[1]` U32.
    pub(crate) non_finite: Tensor,
    /// 1 when the kept prefix may run past the ranked candidates (the draw
    /// must re-run on the host), `[1]` U32.
    pub(crate) overflow: Tensor,
}

/// [`super::super::sampling::race`]'s rule over `ranked` under `policy`,
/// with `noise` `[V]` F32: one exponential per candidate of the whole row.
pub(crate) fn race(
    ranked: &Ranked,
    policy: &FishDrama3DrawPolicy,
    noise: &Tensor,
) -> Result<DeviceDraw> {
    let device = ranked.values.device();
    let k = ranked.values.dim(0)?;
    let cap = policy.top_k.unwrap_or(usize::MAX);
    let temperature = Tensor::from_slice(&[policy.temperature.max(MIN_TEMPERATURE)], 1, device)?;
    super::count_uploads(1);
    let tempered = ranked.values.broadcast_div(&temperature)?;
    let max = tempered.narrow(0, 0, 1)?;
    let weights = tempered.broadcast_sub(&max)?.exp()?;
    let total = ranked
        .row
        .broadcast_div(&temperature)?
        .broadcast_sub(&max)?
        .exp()?
        .sum_keepdim(0)?;
    let probabilities = weights.broadcast_div(&total)?;
    let upper = Tensor::triu2(k, DType::F32, device)?;
    let cumulative = probabilities.unsqueeze(0)?.matmul(&upper)?.squeeze(0)?;
    let rank = Tensor::arange(0u32, k as u32, device)?.to_dtype(DType::F32)?;
    let flag = |mask: Tensor| mask.to_dtype(DType::F32);
    let within =
        (flag(rank.lt(cap.min(k) as f64)?)? * flag(cumulative.le(f64::from(policy.top_p))?)?)?;
    let keep = flag(rank.eq(0.0)?)?.maximum(&within)?;
    let kept = (&weights * &keep)?;
    let score = kept.broadcast_div(&kept.sum_keepdim(0)?)?.div(
        &noise
            .to_dtype(DType::F32)?
            .index_select(&ranked.indices, 0)?,
    )?;
    let pick = ranked
        .indices
        .index_select(&score.argmax_keepdim(0)?.to_dtype(DType::U32)?, 0)?;
    // The host keeps going past the ranks only when they all fit the cap
    // and top_p.
    let overflow = if k < ranked.candidates && cap > k {
        cumulative
            .narrow(0, k - 1, 1)?
            .le(f64::from(policy.top_p))?
            .to_dtype(DType::U32)?
    } else {
        Tensor::zeros(1, DType::U32, device)?
    };
    Ok(DeviceDraw {
        pick,
        non_finite: ranked.non_finite.clone(),
        overflow,
    })
}

#[cfg(all(test, any(feature = "metal", feature = "cuda")))]
mod tests {
    use super::super::super::sampling::race as host_race;
    use super::*;
    use crate::kernels::numeric_check::xorshift_values;

    fn exponentials(n: usize, seed: u64) -> Vec<f32> {
        xorshift_values(n, seed, 0.5)
            .into_iter()
            .map(|u| -((u + 0.5).clamp(1e-6, 1.0 - 1e-6)).ln())
            .collect()
    }

    /// The device race picks what the host race picks under the same
    /// exponentials, over the policies the family runs (capped, uncapped,
    /// near-greedy, eos masked), and flags an uncapped flat row whose kept
    /// prefix runs past the device's ranks.
    #[cfg(feature = "metal")]
    #[test]
    fn the_device_race_agrees_with_the_host_race() {
        let Some((device, _lease)) =
            crate::backends::metal_test_device_or_skip("the_device_race_agrees_with_the_host_race")
        else {
            return;
        };
        race_agrees_with_the_host(&device);
    }

    /// [`the_device_race_agrees_with_the_host_race`] on CUDA.
    #[cfg(feature = "cuda")]
    #[test]
    fn the_cuda_race_agrees_with_the_host_race() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        race_agrees_with_the_host(&device);
    }

    fn race_agrees_with_the_host(device: &candle_core::Device) {
        let device = device.clone();
        let policies = [
            FishDrama3DrawPolicy {
                temperature: 1.0,
                top_p: 0.9,
                top_k: None,
            },
            FishDrama3DrawPolicy::RAS,
            FishDrama3DrawPolicy {
                temperature: 0.7,
                top_p: 0.7,
                top_k: Some(30),
            },
            FishDrama3DrawPolicy {
                temperature: 0.0,
                top_p: 1.0,
                top_k: Some(5),
            },
        ];
        let width = 4097;
        let mut draws = 0;
        for seed in 0..12u64 {
            // Peaked logits, as a trained head's: a few large values.
            let logits = xorshift_values(width, 100 + seed, 4.0)
                .into_iter()
                .enumerate()
                .map(|(i, v)| {
                    if i % 97 == (seed as usize) {
                        v + 9.0
                    } else {
                        v
                    }
                })
                .collect::<Vec<_>>();
            let on_device = Tensor::from_slice(&logits, width, &device).unwrap();
            for first in [0usize, 1] {
                let ranked = rank(&on_device, first, DEVICE_RANKS).unwrap();
                for (p, policy) in policies.iter().enumerate() {
                    let noise = exponentials(width, seed * 7 + p as u64);
                    let draw = race(
                        &ranked,
                        policy,
                        &Tensor::from_slice(&noise, width, &device).unwrap(),
                    )
                    .unwrap();
                    let got = draw.pick.to_vec1::<u32>().unwrap()[0] as usize;
                    let expected = host_race(&logits, policy, first, &noise).unwrap();
                    assert_eq!(got, expected, "seed {seed}, first {first}, policy {p}");
                    assert_eq!(draw.non_finite.to_vec1::<u32>().unwrap(), [0]);
                    assert_eq!(draw.overflow.to_vec1::<u32>().unwrap(), [0]);
                    draws += 1;
                }
            }
        }
        assert_eq!(draws, 96);
        // A flat row under an uncapped top-p of 0.9 keeps ~3,700 ranks.
        let flat = Tensor::from_slice(&vec![0.5f32; width], width, &device).unwrap();
        let ranked = rank(&flat, 1, DEVICE_RANKS).unwrap();
        let noise = Tensor::from_slice(&exponentials(width, 3), width, &device).unwrap();
        let draw = race(&ranked, &policies[0], &noise).unwrap();
        assert_eq!(draw.overflow.to_vec1::<u32>().unwrap(), [1]);
        // A non-finite logit is counted.
        let mut bad = vec![0.5f32; width];
        bad[7] = f32::NAN;
        bad[9] = f32::INFINITY;
        let ranked = rank(&Tensor::from_slice(&bad, width, &device).unwrap(), 0, 30).unwrap();
        assert_eq!(ranked.non_finite.to_vec1::<u32>().unwrap(), [2]);
    }
}
