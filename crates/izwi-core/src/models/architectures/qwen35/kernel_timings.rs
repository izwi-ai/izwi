//! Load-time device timings of the Qwen3.5 decode kernels at batch widths
//! of 1 to 32 rows, so a deploy shows where a decode step's device time goes
//! without a profiler:
//!
//! - the first DeltaNet layer's four input projections through the shared
//!   quantization, its batched recurrence (conv step, recurrence and gated
//!   norm in two launches) and its out-projection;
//! - the first layer's MLP (gate and up through the shared quantization, the
//!   fused SiLU product, down) and one RMSNorm;
//! - the first attention layer's q/k/v projections and its o-projection;
//! - the LM head;
//!
//! each in device milliseconds per call, with the per-step device time those
//! imply for the whole model (each kind times its layers) and the weight
//! bandwidth the projections reach. Attention over the KV cache, the
//! residual adds and sampling are not measured.
//!
//! CUDA only, once after the load-time checks, within a 600 ms wall-clock
//! budget after which the remaining widths are reported as skipped.
//! `IZWI_QWEN35_LOAD_TIMINGS=off` skips them all. A failure is reported and
//! never fails the load.
use std::time::{Duration, Instant};

use candle_core::{Module, Tensor};

use super::{Qwen35Mixer, Qwen35TextModel};
use crate::error::{Error, Result};
use crate::kernels::cuda::timing::device_ms_per_call;
use crate::models::architectures::qwen35::fast_path::{legacy_requested, xorshift_values};
use crate::models::architectures::qwen35::linear::{forward_shared, Qwen35Linear};

/// Switch for the load-time kernel timings (`off`, `legacy`, `0`, `false`).
pub(crate) const LOAD_TIMINGS_ENV: &str = "IZWI_QWEN35_LOAD_TIMINGS";

/// Wall-clock budget; widths after it are reported as skipped.
const BUDGET: Duration = Duration::from_millis(600);

/// The batch widths measured: the graphed widths' range.
const WIDTHS: [usize; 5] = [1, 4, 8, 16, 32];

/// Timed calls per measurement, after one warm-up call.
const ITERATIONS: usize = 5;

/// Bytes a projection's weight holds (0 for a dense one).
fn weight_bytes(linear: &Qwen35Linear) -> u64 {
    match linear.matmul() {
        candle_core::quantized::QMatMul::QTensor(weight) => weight.storage_size_in_bytes() as u64,
        _ => 0,
    }
}

impl Qwen35TextModel {
    /// The load-time kernel timings for the diagnostics; see the module docs.
    pub(super) fn measure_load_timings(&self) -> serde_json::Value {
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

    /// Every projection's weight bytes: what one decode step reads from the
    /// projections, whatever its width.
    fn projection_bytes_per_step(&self) -> u64 {
        let mut bytes = weight_bytes(&self.output);
        for layer in &self.layers {
            bytes += [&layer.ffn.gate, &layer.ffn.up, &layer.ffn.down]
                .into_iter()
                .map(weight_bytes)
                .sum::<u64>();
            bytes += match &layer.mixer {
                Qwen35Mixer::Linear(mixer) => [
                    &mixer.qkv_proj,
                    &mixer.gate_proj,
                    &mixer.beta_proj,
                    &mixer.alpha_proj,
                    &mixer.out_proj,
                ]
                .into_iter()
                .map(weight_bytes)
                .sum::<u64>(),
                Qwen35Mixer::Full(attention) => [
                    &attention.q_proj,
                    &attention.k_proj,
                    &attention.v_proj,
                    &attention.o_proj,
                ]
                .into_iter()
                .map(weight_bytes)
                .sum::<u64>(),
            };
        }
        bytes
    }

    fn measure_kernel_timings(&self, started: Instant) -> Result<serde_json::Value> {
        let device = &self.device;
        let timed = |work: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
            device_ms_per_call(device, 1, ITERATIONS, || {
                work().map_err(|error| candle_core::Error::Msg(error.to_string()))
            })?
            .ok_or_else(|| Error::InferenceError("CUDA timing ran off CUDA".into()))
        };
        let input = |rows: usize, cols: usize, seed: u64| -> Result<Tensor> {
            Ok(Tensor::from_vec(
                xorshift_values(rows * cols, seed, 1.0),
                (rows, 1, cols),
                device,
            )?)
        };
        let layers = self.layers.len();
        let linear = self.layers.iter().find_map(|layer| match &layer.mixer {
            Qwen35Mixer::Linear(mixer) => Some((layer, mixer)),
            Qwen35Mixer::Full(_) => None,
        });
        let full = self.layers.iter().find_map(|layer| match &layer.mixer {
            Qwen35Mixer::Full(attention) => Some(attention),
            Qwen35Mixer::Linear(_) => None,
        });
        let linear_layers = self
            .layers
            .iter()
            .filter(|layer| matches!(layer.mixer, Qwen35Mixer::Linear(_)))
            .count();
        let full_layers = layers - linear_layers;
        let (Some((first, mixer)), Some(attention)) = (linear, full) else {
            return Err(Error::InferenceError(
                "kernel timings need a DeltaNet and an attention layer".into(),
            ));
        };
        let counts = [
            ("deltanet_in_proj", linear_layers),
            ("deltanet_recurrence", linear_layers),
            ("deltanet_out_proj", linear_layers),
            ("mlp", layers),
            ("rms_norm", 2 * layers + 1),
            ("attention_qkv_proj", full_layers),
            ("attention_o_proj", full_layers),
            ("lm_head", 1),
        ];
        let projection_kinds = [
            "deltanet_in_proj",
            "deltanet_out_proj",
            "mlp",
            "attention_qkv_proj",
            "attention_o_proj",
            "lm_head",
        ];
        let bytes = self.projection_bytes_per_step();
        let hidden = self.hidden_size();
        let value_width = mixer.num_v_heads * mixer.head_v_dim;
        let mut widths = serde_json::Map::new();
        for rows in WIDTHS {
            if started.elapsed() > BUDGET {
                widths.insert(
                    rows.to_string(),
                    serde_json::json!({ "skipped": "time budget" }),
                );
                continue;
            }
            let x = input(rows, hidden, 7 + rows as u64)?;
            let mut ms = serde_json::Map::new();
            let mut record = |kind: &str, value: f64| {
                ms.insert(kind.to_string(), value.into());
            };
            let projections = [
                &mixer.qkv_proj,
                &mixer.gate_proj,
                &mixer.beta_proj,
                &mixer.alpha_proj,
            ];
            record(
                "deltanet_in_proj",
                timed(&mut || forward_shared(&x, projections).map(drop))?,
            );
            #[cfg(feature = "cuda")]
            if let Some(spec) = mixer.fused_decode.as_ref() {
                let [mixed_qkv, z, beta_raw, alpha] = forward_shared(&x, projections)?;
                let conv_dim = mixer.conv_dim;
                let history = Tensor::zeros(conv_dim, candle_core::DType::F32, device)?;
                let state = Tensor::zeros(
                    (1, mixer.num_v_heads, mixer.head_k_dim, mixer.head_v_dim),
                    candle_core::DType::F32,
                    device,
                )?;
                let mut outputs = Vec::with_capacity(rows);
                let mut entries =
                    Vec::with_capacity(rows * crate::kernels::cuda::gdn::TABLE_ENTRIES);
                for _ in 0..rows {
                    let history_out =
                        Tensor::zeros((3, conv_dim), candle_core::DType::F32, device)?;
                    let state_out = state.zeros_like()?;
                    entries.extend(crate::kernels::cuda::gdn::row_entries(
                        [&history, &history, &history],
                        &history_out,
                        &state,
                        &state_out,
                        spec,
                    )?);
                    outputs.push((history_out, state_out));
                }
                let mut table =
                    crate::kernels::cuda::segment_graph::DeviceTable::new(device, entries.len())?;
                table.write(0, &entries)?;
                let (x, z) = (
                    mixed_qkv.reshape((rows, conv_dim))?,
                    z.reshape((rows, value_width))?,
                );
                let (beta_raw, alpha) = (
                    beta_raw.reshape((rows, mixer.num_v_heads))?,
                    alpha.reshape((rows, mixer.num_v_heads))?,
                );
                record(
                    "deltanet_recurrence",
                    timed(&mut || {
                        crate::kernels::cuda::gdn::decode_rows(
                            &x,
                            &mixer.conv_kernel,
                            &z,
                            &beta_raw,
                            &alpha,
                            &mixer.dt_bias,
                            &mixer.a,
                            &mixer.norm.weight,
                            &table,
                            rows,
                            spec,
                        )
                        .map(drop)
                        .map_err(Error::from)
                    })?,
                );
                drop(outputs);
            }
            let y = input(rows, value_width, 11)?;
            record(
                "deltanet_out_proj",
                timed(&mut || mixer.out_proj.forward(&y).map(drop))?,
            );
            record("mlp", timed(&mut || first.ffn.forward(&x).map(drop))?);
            record(
                "rms_norm",
                timed(&mut || first.attn_norm.forward(&x).map(drop).map_err(Error::from))?,
            );
            record(
                "attention_qkv_proj",
                timed(&mut || {
                    forward_shared(
                        &x,
                        [&attention.q_proj, &attention.k_proj, &attention.v_proj],
                    )
                    .map(drop)
                })?,
            );
            let attended = input(rows, attention.num_heads * attention.head_dim, 13)?;
            record(
                "attention_o_proj",
                timed(&mut || attention.o_proj.forward(&attended).map(drop))?,
            );
            record("lm_head", timed(&mut || self.output.forward(&x).map(drop))?);
            let per_step = |kinds: &mut dyn Iterator<Item = &str>| -> f64 {
                kinds
                    .map(|kind| {
                        let count = counts
                            .iter()
                            .find(|(name, _)| *name == kind)
                            .map_or(0, |(_, count)| *count);
                        ms.get(kind).and_then(|v| v.as_f64()).unwrap_or(0.0) * count as f64
                    })
                    .sum()
            };
            let step_ms = per_step(&mut counts.iter().map(|(kind, _)| *kind));
            let projection_ms = per_step(&mut projection_kinds.iter().copied());
            let mut entry = serde_json::Map::new();
            entry.insert("ms_per_call".into(), serde_json::Value::Object(ms.clone()));
            entry.insert("estimated_step_ms".into(), step_ms.into());
            entry.insert("projection_step_ms".into(), projection_ms.into());
            entry.insert(
                "projection_gb_per_s".into(),
                (bytes as f64 / (projection_ms.max(f64::MIN_POSITIVE) * 1e6)).into(),
            );
            entry.insert(
                "estimated_tokens_per_s".into(),
                (rows as f64 * 1e3 / step_ms.max(f64::MIN_POSITIVE)).into(),
            );
            widths.insert(rows.to_string(), serde_json::Value::Object(entry));
        }
        Ok(serde_json::json!({
            "layers_per_step": counts
                .iter()
                .map(|(kind, count)| ((*kind).to_string(), serde_json::Value::from(*count)))
                .collect::<serde_json::Map<_, _>>(),
            "projection_bytes_per_step": bytes,
            "widths": widths,
        }))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_measured_widths_span_the_graphed_widths() {
        assert_eq!(super::WIDTHS.first(), Some(&1));
        assert_eq!(
            super::WIDTHS.last(),
            Some(&super::super::decode_graph::MAX_GRAPH_ROWS)
        );
    }
}
