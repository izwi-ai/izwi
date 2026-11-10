//! The top candidates of logits rows after a host sampler's own per-token
//! arithmetic, so the host reads back a few hundred bytes a row instead of
//! the row and finishes the draw on them exactly as it would on the whole
//! row.
//!
//! For output row `r`, over the first `vocab` columns:
//!
//! ```text
//! m_i = c_i                         c = logits[cond row]
//! m_i = c_i + w * (c_i - u_i)       guidance, u = logits[uncond row]
//! m_i = m_i * scale
//! m_i = m_i + bias                  at one index
//! g_i = m_i < 0 ? m_i * p : m_i / p repetition penalty, at listed ids
//! g_i = m_i - p                     presence penalty, at listed ids
//! g_i = -inf                        in the banned range
//! ```
//!
//! each step rounded once in F32, as the hosts' samplers write them, then
//! the `count` largest `g` by total order (ties to the lowest index), sorted,
//! with their indices and how many raw inputs or `m` were not finite. On
//! CUDA one block per row computes it (IEEE intrinsics that nvcc never
//! contracts into an FMA); elsewhere the host does. Because `g` is exact, a
//! sampler that only needs the values at or above some threshold can prove
//! from the last candidate that nothing below it was cut.
use std::ops::Range;

use candle_core::{DType, Device, Result, Tensor};

/// The most candidates one row returns.
pub const MAX_CANDIDATES: usize = 256;
const PARAMS: usize = 12;
const NONE: u32 = u32::MAX;

/// What a row's listed ids get.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CandidatePenalty {
    None,
    /// `m < 0 ? m * p : m / p` (transformers' repetition penalty).
    Repetition(f32),
    /// `m - p`.
    Presence(f32),
}

/// One output row's inputs.
#[derive(Debug, Clone)]
pub struct CandidateRow<'a> {
    /// The logits row read as `c`.
    pub logits_row: usize,
    /// `(uncond row, w)` for classifier-free guidance.
    pub guidance: Option<(usize, f32)>,
    pub scale: f32,
    /// `(index, bias)` added after the scale.
    pub bias: Option<(usize, f32)>,
    pub penalty: CandidatePenalty,
    /// The ids the penalty applies to, each at most once.
    pub penalized: &'a [u32],
    /// Ids whose `g` is `-inf`.
    pub banned: Range<usize>,
}

impl CandidateRow<'_> {
    /// `c` alone: no guidance, scale 1, nothing penalized or banned.
    pub fn plain(logits_row: usize) -> CandidateRow<'static> {
        CandidateRow {
            logits_row,
            guidance: None,
            scale: 1.0,
            bias: None,
            penalty: CandidatePenalty::None,
            penalized: &[],
            banned: 0..0,
        }
    }

    fn params(&self, ids: Range<usize>) -> [u32; PARAMS] {
        let (mode, penalty) = match self.penalty {
            CandidatePenalty::None => (0, 0.0f32),
            CandidatePenalty::Repetition(p) => (1, p),
            CandidatePenalty::Presence(p) => (2, p),
        };
        [
            self.logits_row as u32,
            self.guidance.map_or(NONE, |(row, _)| row as u32),
            self.guidance.map_or(0.0f32, |(_, w)| w).to_bits(),
            self.scale.to_bits(),
            self.bias.map_or(NONE, |(index, _)| index as u32),
            self.bias.map_or(0.0f32, |(_, b)| b).to_bits(),
            mode,
            penalty.to_bits(),
            self.banned.start as u32,
            self.banned.end.max(self.banned.start) as u32,
            ids.start as u32,
            ids.end as u32,
        ]
    }

    /// `m_i` and whether a raw input or `m_i` is not finite.
    fn base(&self, c: &[f32], u: Option<&[f32]>, i: usize) -> (f32, bool) {
        let mut m = c[i];
        let mut bad = !m.is_finite();
        if let (Some(u), Some((_, w))) = (u, self.guidance) {
            bad |= !u[i].is_finite();
            m = c[i] + w * (c[i] - u[i]);
        }
        m *= self.scale;
        if self.bias.is_some_and(|(index, _)| index == i) {
            m += self.bias.map_or(0.0, |(_, b)| b);
        }
        (m, bad || !m.is_finite())
    }

    fn penalize(&self, m: f32) -> f32 {
        match self.penalty {
            CandidatePenalty::None => m,
            CandidatePenalty::Repetition(p) => {
                if m < 0.0 {
                    m * p
                } else {
                    m / p
                }
            }
            CandidatePenalty::Presence(p) => m - p,
        }
    }
}

/// One row's candidates: the largest `g` first (ties by index), `-inf` and
/// [`u32::MAX`] past the row's width, and the non-finite count.
#[derive(Debug, Clone, PartialEq)]
pub struct RowCandidates {
    pub values: Vec<f32>,
    pub indices: Vec<u32>,
    pub non_finite: u32,
}

/// `g` of one row on the host (the reference), from its `c` and `u` rows.
pub fn host_scores(row: &CandidateRow<'_>, c: &[f32], u: Option<&[f32]>) -> (Vec<f32>, u32) {
    let vocab = c.len();
    let mut non_finite = 0u32;
    let banned = |i: usize| row.banned.contains(&i);
    let mut g = (0..vocab)
        .map(|i| {
            let (m, bad) = row.base(c, u, i);
            non_finite += u32::from(bad);
            if banned(i) {
                f32::NEG_INFINITY
            } else {
                m
            }
        })
        .collect::<Vec<_>>();
    if row.penalty != CandidatePenalty::None {
        for &id in row.penalized {
            let i = id as usize;
            if i < vocab && !banned(i) {
                g[i] = row.penalize(row.base(c, u, i).0);
            }
        }
    }
    (g, non_finite)
}

/// The total-order key the kernel ranks by (larger first).
fn key(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// The kernel's packed output for one row of `g`.
fn host_pack(g: &[f32], non_finite: u32, count: usize, out: &mut Vec<u32>) {
    let mut order = (0..g.len()).collect::<Vec<_>>();
    let take = count.min(g.len());
    let rank = |a: &usize, b: &usize| key(g[*b]).cmp(&key(g[*a])).then(a.cmp(b));
    if take < order.len() {
        order.select_nth_unstable_by(take, rank);
        order.truncate(take);
    }
    order.sort_by(rank);
    out.extend((0..count).map(|r| {
        order
            .get(r)
            .map_or(f32::NEG_INFINITY.to_bits(), |&i| g[i].to_bits())
    }));
    out.extend((0..count).map(|r| order.get(r).map_or(NONE, |&i| i as u32)));
    out.push(non_finite);
}

/// The candidates of `rows` over the first `vocab` columns of `logits`
/// (`[logits rows, width]`, F32), as `[rows, 2 count + 1]` U32 on the
/// logits' device (value bits, indices, the non-finite count), for one
/// readback with whatever else the step reads.
pub fn candidates_packed(
    logits: &Tensor,
    vocab: usize,
    rows: &[CandidateRow<'_>],
    count: usize,
) -> Result<Tensor> {
    let (logit_rows, width) = logits.dims2()?;
    let in_range = |row: usize| row < logit_rows;
    if rows.is_empty()
        || logits.dtype() != DType::F32
        || vocab == 0
        || vocab > width
        || vocab >= NONE as usize
        || count == 0
        || count > MAX_CANDIDATES
        || rows.iter().any(|row| {
            !in_range(row.logits_row)
                || row.guidance.is_some_and(|(u, _)| !in_range(u))
                || row.bias.is_some_and(|(index, _)| index >= vocab)
        })
    {
        candle_core::bail!(
            "invalid candidate operands: logits {:?} {:?}, vocab {vocab}, {} rows, count {count}",
            logits.dims(),
            logits.dtype(),
            rows.len()
        )
    }
    let logits = logits.contiguous()?;
    match logits.device() {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => cuda_impl::run(&logits, vocab, rows, count),
        Device::Cpu => {
            let values = logits.flatten_all()?.to_vec1::<f32>()?;
            let row_of = |row: usize| &values[row * width..row * width + vocab];
            let mut out = Vec::with_capacity(rows.len() * (2 * count + 1));
            for row in rows {
                let (g, bad) = host_scores(
                    row,
                    row_of(row.logits_row),
                    row.guidance.map(|(u, _)| row_of(u)),
                );
                host_pack(&g, bad, count, &mut out);
            }
            Tensor::from_vec(out, (rows.len(), 2 * count + 1), &Device::Cpu)
        }
        other => candle_core::bail!("penalized top-k has no {:?} form", other.location()),
    }
}

/// The host values of a [`candidates_packed`] result.
pub fn unpack(packed: &[u32], count: usize) -> Vec<RowCandidates> {
    packed
        .chunks_exact(2 * count + 1)
        .map(|row| RowCandidates {
            values: row[..count]
                .iter()
                .map(|&bits| f32::from_bits(bits))
                .collect(),
            indices: row[count..2 * count].to_vec(),
            non_finite: row[2 * count],
        })
        .collect()
}

/// Whether the device path serves `device`: CUDA, once its one-time bitwise
/// check against the host reference has passed (resolved on the first
/// device that asks). Counters for the families' diagnostics.
pub mod device_path {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::sync::OnceLock;

    use candle_core::{Device, Tensor};

    use super::{candidates_packed, CandidatePenalty, CandidateRow};

    static RESOLVED: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    static ROWS: AtomicU64 = AtomicU64::new(0);
    static WHOLE_ROWS: AtomicU64 = AtomicU64::new(0);

    pub fn ready(device: &Device) -> bool {
        if !device.is_cuda() {
            return false;
        }
        RESOLVED
            .get_or_init(|| {
                self_check(device).map_err(|error| {
                    tracing::warn!(
                        %error,
                        "device candidate self-check failed; TTS samplers read whole logits rows"
                    );
                    error.to_string()
                })
            })
            .is_ok()
    }

    /// Rows sampled from device candidates, and rows that read their whole
    /// logits instead (non-finite input, or candidates that could not prove
    /// they hold the sampler's survivors).
    pub fn record(rows: usize, whole_rows: usize) {
        ROWS.fetch_add(rows as u64, Relaxed);
        WHOLE_ROWS.fetch_add(whole_rows as u64, Relaxed);
    }

    pub fn summary() -> serde_json::Value {
        let (backend, reason) = match RESOLVED.get() {
            None => ("unresolved", None),
            Some(Ok(())) => ("device", None),
            Some(Err(reason)) => ("host", Some(reason.as_str())),
        };
        serde_json::json!({
            "backend": backend,
            "reason": reason,
            "candidate_rows": ROWS.load(Relaxed),
            "whole_rows": WHOLE_ROWS.load(Relaxed),
        })
    }

    /// Rows over every operand form (guidance, scale, bias, both penalties,
    /// bans, non-finite inputs, ties, signed zeros, a width not a multiple
    /// of the block, more candidates than columns), packed on the device and
    /// on the host: every bit must agree.
    pub(super) fn self_check(device: &Device) -> crate::Result<()> {
        const WIDTH: usize = 3_001;
        const VOCAB: usize = 2_999;
        let mut values = crate::kernels::numeric_check::xorshift_values(4 * WIDTH, 17, 9.0);
        for (index, value) in values.iter_mut().enumerate() {
            match index % 97 {
                0 => *value = 1.5,
                1 => *value = -0.0,
                2 => *value = 0.0,
                _ => {}
            }
        }
        values[WIDTH + 13] = f32::NAN;
        values[2 * WIDTH + 5] = f32::INFINITY;
        let host = Tensor::from_vec(values, (4, WIDTH), &Device::Cpu)?;
        let ids = (0..VOCAB as u32).step_by(7).collect::<Vec<_>>();
        let rows = [
            CandidateRow {
                penalty: CandidatePenalty::Repetition(1.1),
                penalized: &ids,
                banned: 0..8,
                ..CandidateRow::plain(0)
            },
            CandidateRow {
                guidance: Some((2, 0.5)),
                penalty: CandidatePenalty::Repetition(0.8),
                penalized: &ids[3..],
                ..CandidateRow::plain(1)
            },
            CandidateRow {
                scale: 1.0 / 0.7,
                bias: Some((VOCAB - 1, 3.25)),
                penalty: CandidatePenalty::Presence(1.5),
                penalized: &ids[..40],
                banned: 10..11,
                ..CandidateRow::plain(3)
            },
            CandidateRow::plain(2),
        ];
        let on_device = host.to_device(device)?;
        for count in [64, 200] {
            let expected = candidates_packed(&host, VOCAB, &rows, count)?
                .flatten_all()?
                .to_vec1::<u32>()?;
            let got = candidates_packed(&on_device, VOCAB, &rows, count)?
                .flatten_all()?
                .to_vec1::<u32>()?;
            if got != expected {
                let at = got.iter().zip(&expected).position(|(a, b)| a != b);
                return Err(crate::Error::InferenceError(format!(
                    "device candidates differ from the host at word {at:?} (count {count})"
                )));
            }
        }
        let narrow = candidates_packed(&on_device.narrow(1, 0, 100)?, 100, &rows[3..], 128)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        let expected = candidates_packed(&host.narrow(1, 0, 100)?, 100, &rows[3..], 128)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        if narrow != expected {
            return Err(crate::Error::InferenceError(
                "device candidates of a narrow row differ from the host".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::{CudaStorage, WrapErr};
    use candle_core::op::BackpropOp;
    use candle_core::{Result, Shape, Storage, Tensor};

    use super::CandidateRow;

    const MODULE: &str = "izwi_penalized_topk";

    pub(super) fn run(
        logits: &Tensor,
        vocab: usize,
        rows: &[CandidateRow<'_>],
        count: usize,
    ) -> Result<Tensor> {
        let device = logits.device().as_cuda_device()?;
        let (_, width) = logits.dims2()?;
        let mut words = Vec::with_capacity(rows.len() * super::PARAMS);
        let mut ids = Vec::new();
        for row in rows {
            let start = ids.len();
            if row.penalty != super::CandidatePenalty::None {
                ids.extend_from_slice(row.penalized);
            }
            words.extend(row.params(start..ids.len()));
        }
        // One upload: the parameters, then the ids (never empty).
        let params_len = words.len();
        words.extend(ids);
        words.push(0);
        let operands = device.clone_htod(&words)?;
        let (storage, _) = logits.storage_and_layout();
        let Storage::Cuda(cuda) = &*storage else {
            candle_core::bail!("penalized top-k needs a CUDA tensor")
        };
        let Some((start, end)) = logits.layout().contiguous_offsets() else {
            candle_core::bail!("penalized top-k needs contiguous logits")
        };
        // SAFETY: the kernel writes every scratch entry of its row before
        // reading it, and every output word.
        let scratch = unsafe { device.alloc::<f32>(rows.len() * vocab)? };
        let stride = 2 * count + 1;
        let out = unsafe { device.alloc::<u32>(rows.len() * stride)? };
        let function = device.get_or_load_custom_func(
            "izwi_penalized_topk",
            MODULE,
            super::super::cuda::cuda_ptx_penalized_topk(),
        )?;
        let view = cuda.as_cuda_slice::<f32>()?.slice(start..end);
        let params = operands.slice(..params_len);
        let id_view = operands.slice(params_len..);
        let mut builder = function.builder();
        builder.arg(&view);
        candle_core::builder_arg!(builder, width as i32, vocab as i32);
        builder.arg(&params);
        builder.arg(&id_view);
        builder.arg(&scratch);
        builder.arg(&out);
        candle_core::builder_arg!(builder, count as i32);
        let config = LaunchConfig {
            grid_dim: (rows.len() as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: argument order and types match the kernel; every row
        // index was checked against the logits.
        unsafe { builder.launch(config) }.w()?;
        Ok(Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(out, device.clone())),
            Shape::from((rows.len(), stride)),
            BackpropOp::none(),
            false,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_scores_follow_each_operand_in_order() {
        let c = [2.0f32, -1.0, 0.5, 1.0, -0.0, 3.0];
        let u = [1.0f32, -2.0, 0.5, 2.0, 0.0, 3.0];
        let row = CandidateRow {
            guidance: Some((1, 0.5)),
            scale: 2.0,
            bias: Some((2, 0.25)),
            penalty: CandidatePenalty::Repetition(1.25),
            penalized: &[1, 3, 4],
            banned: 5..6,
            ..CandidateRow::plain(0)
        };
        let (g, bad) = host_scores(&row, &c, Some(&u));
        assert_eq!(bad, 0);
        // m = (c + 0.5 (c - u)) * 2: [5, -1, 1 (+0.25), 1, -0, 6].
        assert_eq!(g[0], 5.0);
        assert_eq!(g[1], -1.25);
        assert_eq!(g[2], 1.25);
        assert_eq!(g[3], 1.0 / 1.25);
        assert_eq!(g[4].to_bits(), (-0.0f32).to_bits(), "-0 is divided");
        assert_eq!(g[5], f32::NEG_INFINITY);
        let presence = CandidateRow {
            penalty: CandidatePenalty::Presence(0.5),
            penalized: &[0],
            ..CandidateRow::plain(0)
        };
        assert_eq!(host_scores(&presence, &c, None).0[0], 1.5);
        let (_, bad) = host_scores(&CandidateRow::plain(0), &[f32::NAN, 1.0], None);
        assert_eq!(bad, 1);
    }

    #[test]
    fn cpu_candidates_are_the_largest_by_total_order_ties_to_the_lowest_index() {
        let values = [1.0f32, 3.0, 3.0, -0.0, 0.0, 2.0, 3.0];
        let logits = Tensor::from_vec(values.to_vec(), (1, 7), &Device::Cpu).unwrap();
        let packed = candidates_packed(&logits, 7, &[CandidateRow::plain(0)], 4)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<u32>()
            .unwrap();
        let rows = unpack(&packed, 4);
        assert_eq!(rows[0].indices, vec![1, 2, 6, 5]);
        assert_eq!(rows[0].values, vec![3.0, 3.0, 3.0, 2.0]);
        // +0 ranks above -0; past the width the row pads.
        let packed = candidates_packed(&logits, 7, &[CandidateRow::plain(0)], 9)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<u32>()
            .unwrap();
        let row = &unpack(&packed, 9)[0];
        assert_eq!(&row.indices[4..], &[0, 4, 3, NONE, NONE]);
        assert_eq!(row.values[8], f32::NEG_INFINITY);
    }

    #[test]
    fn the_self_check_fixture_is_consistent_on_the_host() {
        // The fixture the device is checked against must itself be valid.
        device_path::self_check(&Device::Cpu).unwrap();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_candidates_match_the_host_bit_for_bit() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        device_path::self_check(&device).unwrap();
    }
}
