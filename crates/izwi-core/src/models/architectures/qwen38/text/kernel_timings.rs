//! Load-time device timings of the Qwen3.8 hot kernels, so a deploy shows
//! where decode and prefill time goes without a profiler:
//!
//! - every distinct trunk block-FP8 projection kind and the LM head at one
//!   row, as production runs them (the decode GEMV), as achieved weight
//!   bandwidth (GB/s), with the per-token GEMV time those rates imply for
//!   the whole trunk;
//! - the prefill projections at 256 rows through the dequantized GEMM (each
//!   kind that resolved it), the compact `mm` kernel against the GEMM on one
//!   kind at 256 rows and at the 16-row switch-over, as TFLOP/s, and the
//!   projection time per 256-token chunk those rates imply;
//! - the DeltaNet prefill recurrence for one 256-token chunk on the path
//!   the layers resolved.
//!
//! CUDA only, once after the load-time checks: about 100-200 ms (estimate),
//! with a 400 ms wall-clock budget after which the remaining measurements
//! are skipped. `IZWI_QWEN38_LOAD_TIMINGS=off` skips them all. Intervals
//! come from raw driver events (`timing::device_ms_per_call`), so a
//! measurement neither consumes nor reports another component's latched
//! CUDA error; a failure is reported and never fails the load.

use std::time::{Duration, Instant};

use super::super::fast_path::legacy_requested;
use super::*;
use crate::kernels::cuda::timing::device_ms_per_call;

/// Switch for the load-time kernel timings (`off`, `legacy`, `0`, `false`).
pub(crate) const LOAD_TIMINGS_ENV: &str = "IZWI_QWEN38_LOAD_TIMINGS";
/// Wall-clock budget; measurements after it are reported as skipped.
const BUDGET: Duration = Duration::from_millis(400);
/// Rows of the prefill measurements (one 256-token chunk).
const PREFILL_ROWS: usize = 256;
/// Device time one decode-shape measurement aims for.
const GEMV_TARGET_MS: f64 = 5.0;

impl Qwen38TextModel {
    /// The load-time kernel timings for the diagnostics; see the module docs.
    pub(in crate::models::architectures::qwen38) fn kernel_timings(&self) -> serde_json::Value {
        if legacy_requested(LOAD_TIMINGS_ENV) {
            return serde_json::json!({ "status": "off", "switch": LOAD_TIMINGS_ENV });
        }
        if !self.device.is_cuda() {
            return serde_json::json!({ "status": "cuda_only" });
        }
        let started = Instant::now();
        match self.measure_kernel_timings(started) {
            Ok(mut timings) => {
                timings["status"] = "measured".into();
                timings["wall_ms"] = (started.elapsed().as_secs_f64() * 1e3).into();
                timings["switch"] = LOAD_TIMINGS_ENV.into();
                timings
            }
            Err(error) => serde_json::json!({ "status": "failed", "error": error.to_string() }),
        }
    }

    /// `(kind, projection, layers of that kind)`: one native block-FP8
    /// projection per distinct trunk kind (from the first DeltaNet layer,
    /// its MLP and the first attention layer) and the LM head.
    fn timing_projections(&self) -> Vec<(String, &Qwen38Projection, usize)> {
        fn group<'a>(
            kinds: &mut Vec<(String, &'a Qwen38Projection, usize)>,
            name: &str,
            group: &'a Qwen38ProjectionGroup,
            count: usize,
        ) {
            match group {
                Qwen38ProjectionGroup::Packed { projection, .. } => {
                    kinds.push((format!("{name} (packed)"), projection, count))
                }
                Qwen38ProjectionGroup::Separate(parts) => {
                    if let Some(first) = parts.first() {
                        kinds.push((format!("{name} (first part)"), first, count))
                    }
                }
            }
        }
        let linear = self.layers.iter().find_map(|layer| match &layer.mixer {
            Qwen38Mixer::Linear(linear) => Some((layer, linear)),
            Qwen38Mixer::Full(_) => None,
        });
        let linear_layers = self
            .layers
            .iter()
            .filter(|layer| matches!(layer.mixer, Qwen38Mixer::Linear(_)))
            .count();
        let full_layers = self.layers.len() - linear_layers;
        let mut kinds = Vec::new();
        if let Some((layer, linear)) = linear {
            group(
                &mut kinds,
                "linear_attn.in_proj_qkvz",
                &linear.qkv_z_proj,
                linear_layers,
            );
            kinds.push((
                "linear_attn.out_proj".into(),
                &linear.out_proj,
                linear_layers,
            ));
            group(
                &mut kinds,
                "mlp.gate_up",
                &layer.mlp.gate_up,
                self.layers.len(),
            );
            kinds.push(("mlp.down_proj".into(), &layer.mlp.down, self.layers.len()));
        }
        if let Some(attention) = self.full_attention_layers().next() {
            group(
                &mut kinds,
                "self_attn.qkv",
                &attention.qkv_proj,
                full_layers,
            );
            kinds.push(("self_attn.o_proj".into(), &attention.o_proj, full_layers));
        }
        kinds.push(("lm_head".into(), &self.output, 1));
        kinds.retain(|(_, projection, _)| matches!(projection, Qwen38Projection::NativeFp8 { .. }));
        kinds
    }

    /// Bytes one decode token reads from the resident block-FP8 projections
    /// (every trunk projection and the LM head, E4M3 bytes plus F32 scales).
    fn native_fp8_bytes_per_token(&self) -> u64 {
        let bytes = |projection: &Qwen38Projection| match projection {
            Qwen38Projection::NativeFp8 { raw, .. } => {
                (raw.weights.elem_count() + raw.scales.elem_count() * 4) as u64
            }
            Qwen38Projection::Quantized(_) | Qwen38Projection::Dense(_) => 0,
        };
        let group = |group: &Qwen38ProjectionGroup| match group {
            Qwen38ProjectionGroup::Packed { projection, .. } => bytes(projection),
            Qwen38ProjectionGroup::Separate(parts) => parts.iter().map(bytes).sum(),
        };
        let trunk: u64 = self
            .layers
            .iter()
            .map(|layer| {
                group(&layer.mlp.gate_up)
                    + bytes(&layer.mlp.down)
                    + match &layer.mixer {
                        Qwen38Mixer::Linear(linear) => {
                            group(&linear.qkv_z_proj) + bytes(&linear.out_proj)
                        }
                        Qwen38Mixer::Full(attention) => {
                            group(&attention.qkv_proj) + bytes(&attention.o_proj)
                        }
                    }
            })
            .sum();
        trunk + bytes(&self.output)
    }

    fn measure_kernel_timings(&self, started: Instant) -> Result<serde_json::Value> {
        let dtype = self.activation_dtype();
        let device = &self.device;
        let input = |rows: usize, k: usize, seed: u64| -> Result<Tensor> {
            Ok(Tensor::from_vec(
                probe_activations(rows, k, dtype, seed),
                (1, rows, k),
                &Device::Cpu,
            )?
            .to_dtype(dtype)?
            .to_device(device)?)
        };
        let timed = |warmup: usize,
                     iterations: usize,
                     work: &mut dyn FnMut() -> Result<Tensor>|
         -> Result<f64> {
            device_ms_per_call(device, warmup, iterations, || {
                work()
                    .map(drop)
                    .map_err(|error| candle_core::Error::Msg(error.to_string()))
            })?
            .ok_or_else(|| Error::InferenceError("CUDA timing ran off CUDA".into()))
        };
        let over_budget = || started.elapsed() > BUDGET;
        let skipped = |kind: &str| serde_json::json!({ "kind": kind, "skipped": "time budget" });
        let kinds = self.timing_projections();

        let mut decode = Vec::new();
        let (mut measured_bytes, mut measured_ms) = (0f64, 0f64);
        for (kind, projection, _) in &kinds {
            let Qwen38Projection::NativeFp8 { raw, .. } = projection else {
                continue;
            };
            if over_budget() {
                decode.push(skipped(kind));
                continue;
            }
            let [n, k] = raw.shape;
            let bytes = (raw.weights.elem_count() + raw.scales.elem_count() * 4) as f64;
            let iterations = (GEMV_TARGET_MS * 1e-3 * 2.0e12 / bytes)
                .ceil()
                .clamp(3.0, 64.0);
            let x = input(1, k, 7)?;
            let ms = timed(1, iterations as usize, &mut || Ok(projection.forward(&x)?))?;
            measured_bytes += bytes;
            measured_ms += ms;
            decode.push(serde_json::json!({
                "kind": kind, "n": n, "k": k, "ms": ms, "gb_per_s": bytes / (ms * 1e6),
            }));
        }
        let per_token = self.native_fp8_bytes_per_token();
        let weighted = (measured_ms > 0.0).then(|| measured_bytes / (measured_ms * 1e6));

        let mut prefill = Vec::new();
        let mut chunk_ms = 0f64;
        for (kind, projection, count) in &kinds {
            let Qwen38Projection::NativeFp8 {
                raw, prefill_gemm, ..
            } = projection
            else {
                continue;
            };
            if kind == "lm_head" {
                continue;
            }
            if !prefill_gemm {
                prefill.push(serde_json::json!({ "kind": kind, "gemm": "not resolved" }));
                continue;
            }
            if over_budget() {
                prefill.push(skipped(kind));
                continue;
            }
            let [n, k] = raw.shape;
            let x = input(PREFILL_ROWS, k, 11)?;
            let ms = timed(1, 3, &mut || Ok(projection.forward(&x)?))?;
            chunk_ms += ms * *count as f64;
            prefill.push(serde_json::json!({
                "kind": kind, "n": n, "k": k, "layers": count, "gemm_ms": ms,
                "gemm_tflop_per_s": flops(PREFILL_ROWS, n, k) / (ms * 1e9),
            }));
        }
        // The compact kernel against the GEMM on the smallest output
        // projection: at 256 rows, and at the 16-row switch-over.
        let compare = kinds
            .iter()
            .filter(|(kind, ..)| kind.ends_with("out_proj") || kind.ends_with("o_proj"))
            .filter_map(|(kind, projection, _)| match projection {
                Qwen38Projection::NativeFp8 { raw, .. } => Some((kind, *projection, raw.shape)),
                _ => None,
            })
            .min_by_key(|(_, _, [n, k])| n * k);
        let compact_vs_gemm = match compare {
            Some(_) if over_budget() => serde_json::json!({ "skipped": "time budget" }),
            Some((kind, projection, [n, k])) => {
                let compact = projection.with_prefill_gemm(false).expect("native FP8");
                let gemm = projection.with_prefill_gemm(true).expect("native FP8");
                let (wide, narrow) = (input(PREFILL_ROWS, k, 13)?, input(16, k, 17)?);
                // `mm` is the slow path being measured: one timed 256-row
                // call, no warm-up of its own (the 16-row call loads it).
                let mm_narrow = timed(1, 3, &mut || Ok(compact.forward(&narrow)?))?;
                let mm_wide = timed(0, 1, &mut || Ok(compact.forward(&wide)?))?;
                let gemm_wide = timed(1, 3, &mut || Ok(gemm.forward(&wide)?))?;
                let gemm_narrow = timed(1, 3, &mut || Ok(gemm.forward(&narrow)?))?;
                serde_json::json!({
                    "kind": kind, "n": n, "k": k,
                    "mm_ms_256_rows": mm_wide,
                    "mm_tflop_per_s_256_rows": flops(PREFILL_ROWS, n, k) / (mm_wide * 1e9),
                    "gemm_ms_256_rows": gemm_wide,
                    "mm_ms_16_rows": mm_narrow,
                    "gemm_ms_16_rows": gemm_narrow,
                    "gemm_faster_at_16_rows": gemm_narrow < mm_narrow,
                })
            }
            None => serde_json::json!({ "skipped": "no native block-FP8 output projection" }),
        };

        let deltanet = match self.layers.iter().find_map(|layer| match &layer.mixer {
            Qwen38Mixer::Linear(linear) => Some(linear),
            Qwen38Mixer::Full(_) => None,
        }) {
            Some(_) if over_budget() => serde_json::json!({ "skipped": "time budget" }),
            Some(linear) => {
                let linear_layers = self
                    .layers
                    .iter()
                    .filter(|layer| matches!(layer.mixer, Qwen38Mixer::Linear(_)))
                    .count();
                let ms = linear.time_prefill_chunk(PREFILL_ROWS, &timed)?;
                serde_json::json!({
                    "path": if linear.gdn_prefill { "grouped" } else { "legacy" },
                    "tokens": PREFILL_ROWS,
                    "ms_per_layer": ms,
                    "linear_layers": linear_layers,
                    "ms_per_chunk_all_layers": ms * linear_layers as f64,
                })
            }
            None => serde_json::json!({ "skipped": "no DeltaNet layer" }),
        };

        Ok(serde_json::json!({
            "decode_fp8": {
                "rows": 1,
                "kinds": decode,
                "weighted_gb_per_s": weighted,
                "fp8_bytes_per_token": per_token,
                "gemv_ms_per_token_at_weighted_rate": weighted.map(|rate| per_token as f64 / (rate * 1e6)),
            },
            "prefill_projections": {
                "rows": PREFILL_ROWS,
                "kinds": prefill,
                "gemm_ms_per_chunk": chunk_ms,
                "compact_vs_gemm": compact_vs_gemm,
            },
            "deltanet_prefill": deltanet,
        }))
    }
}

/// Multiply-add FLOPs of an `[rows, k] x [k, n]` product.
fn flops(rows: usize, n: usize, k: usize) -> f64 {
    2.0 * rows as f64 * n as f64 * k as f64
}

impl Qwen38LinearAttention {
    /// Device milliseconds of one `tokens`-token prefill recurrence at this
    /// layer's geometry on the path it resolved (synthetic conv output,
    /// gates and state).
    fn time_prefill_chunk(
        &self,
        tokens: usize,
        timed: &dyn Fn(usize, usize, &mut dyn FnMut() -> Result<Tensor>) -> Result<f64>,
    ) -> Result<f64> {
        let heads = self.num_v_heads;
        let conv_dim = 2 * self.num_k_heads * self.head_k_dim + heads * self.head_v_dim;
        let device = self.conv_kernel.device();
        let tensor = |values: Vec<f32>, shape: &[usize]| -> Result<Tensor> {
            Ok(Tensor::from_vec(values, shape, &Device::Cpu)?.to_device(device)?)
        };
        let mixed_qkv = tensor(
            probe_activations(tokens, conv_dim, DType::F32, 41),
            &[1, tokens, conv_dim],
        )?;
        let g = tensor(
            probe_activations(tokens, heads, DType::F32, 42)
                .iter()
                .map(|v| -(0.02 + 0.5 * v.abs()))
                .collect(),
            &[1, tokens, heads],
        )?;
        let beta = tensor(
            probe_activations(tokens, heads, DType::F32, 43)
                .iter()
                .map(|v| 0.5 + 0.45 * v.clamp(-1.0, 1.0))
                .collect(),
            &[1, tokens, heads],
        )?;
        let state = Tensor::zeros(
            (1, heads, self.head_k_dim, self.head_v_dim),
            DType::F32,
            device,
        )?;
        if self.gdn_prefill {
            timed(1, 3, &mut || {
                Ok(qwen38_deltanet_sequence(
                    &mixed_qkv,
                    &g,
                    &beta,
                    &state,
                    self.num_k_heads,
                    heads,
                    self.head_k_dim,
                    self.head_v_dim,
                )?
                .0)
            })
        } else {
            // Today's path is the slow one: a single timed call after one
            // warm-up.
            timed(1, 1, &mut || {
                let (query, key, value) = self.normalized_qkv(&mixed_qkv, tokens)?;
                Ok(self
                    .legacy_sequence_recurrence(query, key, value, &g, &beta, state.clone())?
                    .0)
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_timings_flop_count() {
        assert_eq!(flops(256, 5120, 6144), 2.0 * 256.0 * 5120.0 * 6144.0);
    }
}
