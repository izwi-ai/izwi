//! Load-time device timings of the Qwen3.6 decode kernels at batch widths of
//! 1 to 32 rows, so a deploy shows where a decode step's device time goes
//! without a profiler:
//!
//! - the first DeltaNet layer's input and output projections, the first
//!   attention layer's q/k/v and o projections, and the LM head, as each
//!   production call runs them, with the weight bandwidth they reach;
//! - the first sparse layer's routed experts at 1 to 128 routed tokens,
//!   through both the per-(token, slot) GEMVs and the expert-major grouped
//!   kernels, whichever production would pick;
//! - one DeltaNet decode row (conv step and recurrence), which multi-row
//!   decode runs once per row and layer;
//!
//! each in device milliseconds per call, with the per-step device time those
//! imply for the whole model. Attention over the KV cache, norms, residual
//! adds and sampling are not measured.
//!
//! CUDA only, once after the load-time checks, within an 800 ms wall-clock
//! budget after which the remaining measurements are reported as skipped.
//! `IZWI_QWEN36_LOAD_TIMINGS=off` skips them all. A failure is reported and
//! never fails the load.

use std::time::{Duration, Instant};

use super::*;
use crate::kernels::cuda::timing::device_ms_per_call;

/// Switch for the load-time kernel timings (`off`, `legacy`, `0`, `false`).
pub(crate) const LOAD_TIMINGS_ENV: &str = "IZWI_QWEN36_LOAD_TIMINGS";
/// Wall-clock budget; measurements after it are reported as skipped.
const BUDGET: Duration = Duration::from_millis(800);
/// Decode widths of the projection and DeltaNet measurements.
const PROJECTION_ROWS: [usize; 6] = [1, 4, 5, 8, 16, 32];
/// Routed token counts of the MoE measurements.
const MOE_TOKENS: [usize; 6] = [1, 4, 8, 32, 64, 128];
/// Timed calls per measurement, after one warm-up call.
const ITERATIONS: usize = 5;

/// The smallest measured routed-token count from which the decode kernels
/// beat the per-pair GEMVs at it and at every larger measured count (`wins`:
/// ascending `(tokens, decode faster)`); `None` when they lose at the largest
/// or nothing was measured.
fn decode_min_tokens(wins: &[(usize, bool)]) -> Option<usize> {
    let first = wins
        .iter()
        .rposition(|(_, faster)| !faster)
        .map_or(0, |last_loss| last_loss + 1);
    wins.get(first).map(|(tokens, _)| *tokens)
}

/// Deterministic unstructured inputs in [-1, 1).
fn xorshift_values(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

impl Qwen36Projection {
    /// Bytes a block-FP8 projection streams per call (0 for other storage).
    fn compact_bytes(&self) -> u64 {
        match self {
            Self::CompactFp8 { weights, scales } => {
                (weights.elem_count() * weights.dtype().size_in_bytes()
                    + scales.elem_count() * scales.dtype().size_in_bytes()) as u64
            }
            Self::Quantized(_) => 0,
        }
    }
}

impl Qwen36TextModel {
    /// The load-time kernel timings for the diagnostics; see the module docs.
    pub(crate) fn kernel_timings(&self) -> serde_json::Value {
        if legacy_requested(LOAD_TIMINGS_ENV) {
            return serde_json::json!({ "status": "off", "switch": LOAD_TIMINGS_ENV });
        }
        if !self.device.is_cuda() {
            return serde_json::json!({ "status": "cuda_only" });
        }
        let started = Instant::now();
        match self.measure_kernel_timings(started, BUDGET) {
            Ok(mut timings) => {
                timings["status"] = "measured".into();
                timings["wall_ms"] = (started.elapsed().as_secs_f64() * 1e3).into();
                timings["switch"] = LOAD_TIMINGS_ENV.into();
                timings
            }
            Err(error) => serde_json::json!({ "status": "failed", "error": error.to_string() }),
        }
    }

    /// Device milliseconds per call on CUDA; host wall time elsewhere, which
    /// only the CPU tests of these measurements use.
    fn timed(&self, work: &mut dyn FnMut() -> Result<()>) -> Result<f64> {
        if !self.device.is_cuda() {
            work()?;
            let started = Instant::now();
            for _ in 0..ITERATIONS {
                work()?;
            }
            return Ok(started.elapsed().as_secs_f64() * 1e3 / ITERATIONS as f64);
        }
        device_ms_per_call(&self.device, 1, ITERATIONS, || {
            work().map_err(|error| candle_core::Error::Msg(error.to_string()))
        })?
        .ok_or_else(|| Error::InferenceError("CUDA timing ran off CUDA".into()))
    }

    fn timing_input(&self, rows: usize, cols: usize, seed: u64, dtype: DType) -> Result<Tensor> {
        Ok(Tensor::from_vec(
            xorshift_values(rows * cols, seed),
            (rows, cols),
            &self.device,
        )?
        .to_dtype(dtype)?)
    }

    pub(super) fn measure_kernel_timings(
        &self,
        started: Instant,
        budget: Duration,
    ) -> Result<serde_json::Value> {
        let linear = self.layers.iter().find_map(|layer| match &layer.mixer {
            Qwen36Mixer::Linear(linear) => Some(linear),
            Qwen36Mixer::Full(_) => None,
        });
        let full = self.layers.iter().find_map(|layer| match &layer.mixer {
            Qwen36Mixer::Full(attention) => Some(attention),
            Qwen36Mixer::Linear(_) => None,
        });
        let sparse = self.layers.iter().find_map(|layer| match &layer.ffn {
            Qwen36FeedForward::Sparse(moe) => moe.fused_experts(),
            Qwen36FeedForward::Dense(_) => None,
        });
        let (Some(linear), Some(full)) = (linear, full) else {
            return Err(Error::InferenceError(
                "kernel timings need a DeltaNet and an attention layer".into(),
            ));
        };
        let linear_layers = self
            .layers
            .iter()
            .filter(|layer| matches!(layer.mixer, Qwen36Mixer::Linear(_)))
            .count();
        let full_layers = self.layers.len() - linear_layers;
        let sparse_layers = self
            .layers
            .iter()
            .filter(|layer| matches!(layer.ffn, Qwen36FeedForward::Sparse(_)))
            .count();
        let hidden = self.hidden_size();
        let value_width = linear.num_v_heads * linear.head_v_dim;
        let attended = full.num_heads * full.head_dim;
        let kinds: [(&str, &Qwen36Projection, usize, usize); 5] = [
            (
                "linear_attn.in_proj",
                linear.packed_in.as_ref().unwrap_or(&linear.qkv_proj),
                hidden,
                linear_layers,
            ),
            (
                "linear_attn.out_proj",
                &linear.out_proj,
                value_width,
                linear_layers,
            ),
            (
                "self_attn.qkv",
                full.packed_qkv.as_ref().unwrap_or(&full.q_proj),
                hidden,
                full_layers,
            ),
            ("self_attn.o_proj", &full.o_proj, attended, full_layers),
            ("lm_head", &self.output, hidden, 1),
        ];
        let mut widths = serde_json::Map::new();
        let mut gdn_row_ms = None;
        for rows in PROJECTION_ROWS {
            if started.elapsed() > budget {
                widths.insert(
                    rows.to_string(),
                    serde_json::json!({ "skipped": "time budget" }),
                );
                continue;
            }
            let mut entry = serde_json::Map::new();
            let mut step_ms = 0.0;
            let mut step_bytes = 0u64;
            for (index, (kind, projection, cols, layers)) in kinds.iter().enumerate() {
                let x = self.timing_input(
                    rows,
                    *cols,
                    7 + index as u64,
                    probe_dtype(projection, &self.device),
                )?;
                let ms = self.timed(&mut || projection.forward(&x).map(drop))?;
                let bytes = projection.compact_bytes();
                step_ms += ms * *layers as f64;
                step_bytes += bytes * *layers as u64;
                entry.insert(
                    (*kind).to_string(),
                    serde_json::json!({
                        "ms": ms,
                        "gb_per_s": bytes as f64 / (ms.max(f64::MIN_POSITIVE) * 1e6),
                    }),
                );
            }
            if gdn_row_ms.is_none() {
                gdn_row_ms = self.deltanet_row_ms(linear)?;
            }
            let gdn_ms = gdn_row_ms.map(|ms| ms * rows as f64 * linear_layers as f64);
            entry.insert(
                "projection_step_ms".into(),
                serde_json::Value::from(step_ms),
            );
            entry.insert(
                "projection_gb_per_s".into(),
                (step_bytes as f64 / (step_ms.max(f64::MIN_POSITIVE) * 1e6)).into(),
            );
            entry.insert("deltanet_rows_step_ms".into(), gdn_ms.into());
            widths.insert(rows.to_string(), serde_json::Value::Object(entry));
        }
        let moe = match sparse {
            Some(experts) => self.measure_moe(experts, hidden, sparse_layers, started, budget)?,
            None => serde_json::json!({ "status": "no fused sparse layer" }),
        };
        Ok(serde_json::json!({
            "layers": {
                "deltanet": linear_layers,
                "attention": full_layers,
                "sparse": sparse_layers,
            },
            "deltanet_row_ms": gdn_row_ms,
            "projections_by_rows": widths,
            "moe_by_tokens": moe,
        }))
    }

    /// One DeltaNet decode row (conv step + recurrence) on the fused path,
    /// from zero state; `None` when the layer runs the legacy path.
    fn deltanet_row_ms(&self, linear: &Qwen36LinearAttention) -> Result<Option<f64>> {
        let Some(spec) = linear.fused_decode.as_ref() else {
            return Ok(None);
        };
        let dtype = norm_dtype(&self.device);
        let value_width = linear.num_v_heads * linear.head_v_dim;
        let mixed = self.timing_input(1, linear.conv_dim, 31, dtype)?;
        let z = self.timing_input(1, value_width, 37, dtype)?;
        let beta = self.timing_input(1, linear.num_v_heads, 41, dtype)?;
        let alpha = self.timing_input(1, linear.num_v_heads, 43, dtype)?;
        let slot = Tensor::zeros(linear.conv_dim, DType::F32, &self.device)?;
        let state = Tensor::zeros(
            (1, linear.num_v_heads, linear.head_k_dim, linear.head_v_dim),
            DType::F32,
            &self.device,
        )?;
        let ms = self.timed(&mut || {
            let (conv, _) = gdn::conv_decode(&mixed, &linear.conv_kernel, [&slot, &slot, &slot])?;
            gdn::recurrent_decode(
                &conv,
                &z,
                &beta,
                &alpha,
                &linear.dt_bias,
                &linear.a,
                &linear.norm.weight,
                &state,
                spec,
            )?;
            Ok(())
        })?;
        Ok(Some(ms))
    }

    /// The first sparse layer's routed experts through both kernel forms.
    fn measure_moe(
        &self,
        experts: &crate::models::architectures::qwen36moe::fused_moe::Qwen36MoeFusedExperts,
        hidden: usize,
        sparse_layers: usize,
        started: Instant,
        budget: Duration,
    ) -> Result<serde_json::Value> {
        let mut by_tokens = serde_json::Map::new();
        let mut crossover = None;
        // (tokens, decode faster than the per-pair GEMVs) up to the decode
        // kernels' size.
        let mut decode_wins = Vec::new();
        for tokens in MOE_TOKENS {
            if started.elapsed() > budget {
                by_tokens.insert(
                    tokens.to_string(),
                    serde_json::json!({ "skipped": "time budget" }),
                );
                continue;
            }
            let flat = self.timing_input(tokens, hidden, 53 + tokens as u64, DType::BF16)?;
            let routing = experts.route(&flat)?;
            let per_pair =
                self.timed(&mut || experts.combine_per_pair(&flat, &routing).map(drop))?;
            let grouped = if experts.grouped() {
                Some(self.timed(&mut || experts.combine_grouped(&flat, &routing).map(drop))?)
            } else {
                None
            };
            if crossover.is_none() && grouped.is_some_and(|grouped| grouped < per_pair) {
                crossover = Some(tokens);
            }
            let decode = if experts.decode_ready()
                && tokens <= crate::kernels::cuda::moe::DECODE_MAX_TOKENS
            {
                let ms = self.timed(&mut || experts.combine_decode(&flat, &routing).map(drop))?;
                decode_wins.push((tokens, ms < per_pair));
                Some(ms)
            } else {
                None
            };
            by_tokens.insert(
                tokens.to_string(),
                serde_json::json!({
                    "per_pair_ms": per_pair,
                    "grouped_ms": grouped,
                    "decode_ms": decode,
                    "per_pair_step_ms": per_pair * sparse_layers as f64,
                    "grouped_step_ms": grouped.map(|ms| ms * sparse_layers as f64),
                    "decode_step_ms": decode.map(|ms| ms * sparse_layers as f64),
                }),
            );
        }
        let decode_min_tokens = decode_min_tokens(&decode_wins);
        // Where the grouped kernels overtook the per-pair GEMVs; past the
        // largest measurement when they never did. Prefill spans are larger.
        let measured_all = by_tokens
            .values()
            .all(|entry| entry.get("skipped").is_none());
        let crossover = match crossover {
            Some(tokens) => Some(tokens),
            None if experts.grouped() && measured_all => MOE_TOKENS.last().map(|tokens| tokens * 2),
            None => None,
        };
        Ok(serde_json::json!({
            "by_tokens": by_tokens,
            "grouped_crossover_tokens": crossover,
            "decode_min_tokens": decode_min_tokens,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::decode_min_tokens;

    #[test]
    fn decode_kernels_serve_from_where_they_win_at_every_larger_count() {
        assert_eq!(decode_min_tokens(&[]), None);
        assert_eq!(
            decode_min_tokens(&[(1, true), (4, true), (8, true), (32, true)]),
            Some(1)
        );
        assert_eq!(
            decode_min_tokens(&[(1, false), (4, true), (8, true), (32, true)]),
            Some(4)
        );
        // A win below a loss does not count: only the wins up to the largest.
        assert_eq!(
            decode_min_tokens(&[(1, true), (4, false), (8, true), (32, true)]),
            Some(8)
        );
        assert_eq!(
            decode_min_tokens(&[(1, true), (4, true), (8, true), (32, false)]),
            None
        );
    }
}
