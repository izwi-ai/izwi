//! Maya2's backbone: a Qwen3-1.7B decoder (GQA 16/8, head 128, per-head
//! q/k RMSNorm, RoPE θ 10⁶), family-local on izwi's shared paged-attention
//! primitives (the Maya1 trunk's shape).
//!
//! It reads embeddings, not token ids: text positions, summed audio-frame
//! codes and the speaker anchor are composed by the caller. One
//! [`Maya2Backbone::forward`] call carries rows from any requests (prompt
//! chunks and one-frame decode steps alike): the dense layers run once over
//! every position and attention once per KV arena through the shared mixed
//! paged step. Each row's last position comes back normed, `[rows, hidden]`;
//! the codebook-0 head is applied by the caller. The 194,548-row text head
//! is never computed.

use std::sync::Arc;

use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{RmsNorm, VarBuilder};

use super::config::Maya2BackboneConfig;
use crate::error::{Error, Result};
use crate::models::shared::attention::mixed::{
    MixedRowLayout, MixedRowSpan, PreparedPhysicalMixedStep,
};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::rope::{rope_frequencies, RopeScaling};

struct Layer {
    input_norm: RmsNorm,
    /// q, k and v stacked: [q_out + 2 kv_out, hidden].
    qkv: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    o_proj: Tensor,
    post_norm: RmsNorm,
    /// gate and up stacked: [2 intermediate, hidden].
    gate_up: Tensor,
    down: Tensor,
}

pub struct Maya2Backbone {
    cfg: Maya2BackboneConfig,
    dtype: DType,
    device: Device,
    layers: Vec<Layer>,
    norm: RmsNorm,
    /// RoPE cos/sin, [positions, head_dim / 2], built on the host in F64.
    rope_cos: Tensor,
    rope_sin: Tensor,
    max_positions: usize,
}

/// The most decode rows one forward takes, and the span up to which a
/// projection keeps each row's sums apart (a longer span is one request's
/// prefill, whose shape is its own).
pub(crate) const MAX_DECODE_GROUP: usize = 32;

/// `x` `[rows, in]` · `weight`ᵀ, each row rounded the same however many
/// requests share the call, so a request sounds the same under load as
/// alone. Candle picks a matmul's kernel and tiles from its row count
/// (`gemv` for one row, GEMM tiles by size; the CPU's matmul as well), and
/// each rounds a row's sums differently. On the CPU and Metal each row of a
/// decode-sized call takes the single-row kernel a lone request gets; on
/// CUDA the call is padded to [`MAX_DECODE_GROUP`] rows, one shape for every
/// batch.
pub(super) fn rows_apart_matmul(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let rows = x.dim(0)?;
    let weight = weight.t()?;
    if rows > MAX_DECODE_GROUP {
        return Ok(x.matmul(&weight)?);
    }
    if x.device().is_cuda() {
        let padded = x.pad_with_zeros(0, 0, MAX_DECODE_GROUP - rows)?;
        return Ok(padded.matmul(&weight)?.narrow(0, 0, rows)?);
    }
    if rows == 1 {
        return Ok(x.matmul(&weight)?);
    }
    let products = (0..rows)
        .map(|row| Ok(x.narrow(0, row, 1)?.matmul(&weight)?))
        .collect::<Result<Vec<_>>>()?;
    Ok(Tensor::cat(&products, 0)?)
}

/// One row of a forward: `embeds` `[positions, hidden]` appended to `cache`.
pub struct Maya2BackboneRow<'a> {
    pub embeds: Tensor,
    pub cache: &'a mut PhysicalPagedKvCache,
}

impl Maya2Backbone {
    /// Load the decoder layers from `vb` (the `backbone.lm.model` prefix) in
    /// its dtype; only `cfg.num_hidden_layers` layers are read.
    pub fn load(cfg: Maya2BackboneConfig, vb: VarBuilder, max_positions: usize) -> Result<Self> {
        let dtype = vb.dtype();
        let device = vb.device().clone();
        let layers = (0..cfg.num_hidden_layers)
            .map(|index| load_layer(&cfg, &vb.pp(format!("layers.{index}"))))
            .collect::<Result<Vec<_>>>()?;
        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let freqs = rope_frequencies(cfg.head_dim, cfg.rope_theta, &RopeScaling::Default)?;
        let (cos, sin) = freqs.cos_sin_table(0, max_positions);
        let pairs = cfg.head_dim / 2;
        let table = |values: Vec<f32>| -> Result<Tensor> {
            Ok(
                Tensor::from_vec(values, (max_positions, pairs), &Device::Cpu)?
                    .to_dtype(dtype)?
                    .to_device(&device)?,
            )
        };
        Ok(Self {
            rope_cos: table(cos)?,
            rope_sin: table(sin)?,
            cfg,
            dtype,
            device,
            layers,
            norm,
            max_positions,
        })
    }

    pub fn config(&self) -> &Maya2BackboneConfig {
        &self.cfg
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn max_positions(&self) -> usize {
        self.max_positions
    }

    /// Requests' rows as the engine batches them, each row's last hidden
    /// state in row order: the bits each row gets alone. A prefill row runs
    /// by itself (its span alone sets its matmul shapes); single-position
    /// decode rows share forwards in groups of at most [`MAX_DECODE_GROUP`],
    /// whose projections keep rows apart ([`rows_apart_matmul`]).
    pub fn forward_batch(&self, rows: &mut [Maya2BackboneRow<'_>]) -> Result<Tensor> {
        let decode = rows
            .iter()
            .all(|row| row.embeds.dim(0).is_ok_and(|n| n == 1));
        let group = if decode { MAX_DECODE_GROUP } else { 1 };
        let mut hidden = rows
            .chunks_mut(group)
            .map(|chunk| self.forward(chunk))
            .collect::<Result<Vec<_>>>()?;
        Ok(if hidden.len() == 1 {
            hidden.pop().expect("one group")
        } else {
            Tensor::cat(&hidden, 0)?
        })
    }

    /// Run `rows` through the decoder; each row's last hidden state after
    /// the final norm, `[rows, hidden]` in the model dtype, in row order.
    pub fn forward(&self, rows: &mut [Maya2BackboneRow<'_>]) -> Result<Tensor> {
        if rows.is_empty() {
            return Err(Error::InvalidInput(
                "Maya2 backbone forward has no rows".into(),
            ));
        }
        let cfg = &self.cfg;
        let mut spans = Vec::with_capacity(rows.len());
        for row in rows.iter() {
            let (count, hidden) = row.embeds.dims2()?;
            let start = row.cache.context_len();
            let end = start.saturating_add(count);
            if count == 0
                || hidden != cfg.hidden_size
                || end > self.max_positions
                || end > row.cache.capacity_tokens()
            {
                return Err(Error::InvalidInput(format!(
                    "Maya2 row of {count} positions at {start} exceeds its capacity ({} positions, cache {})",
                    self.max_positions,
                    row.cache.capacity_tokens()
                )));
            }
            row.cache.validate_model(
                cfg.num_hidden_layers,
                cfg.num_key_value_heads,
                cfg.head_dim,
            )?;
            spans.push(MixedRowSpan {
                start_pos: start,
                q_len: count,
            });
        }

        // Rows grouped by arena, decode rows first within each group; the
        // flattened position order is group after group.
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for index in 0..rows.len() {
            let arena = rows[index].cache.arena();
            match groups
                .iter_mut()
                .find(|group| Arc::ptr_eq(rows[group[0]].cache.arena(), arena))
            {
                Some(group) => group.push(index),
                None => groups.push(vec![index]),
            }
        }
        for group in &mut groups {
            group.sort_by_key(|&index| spans[index].q_len != 1);
        }
        let order = groups.iter().flatten().copied().collect::<Vec<_>>();

        let mut steps = Vec::with_capacity(groups.len());
        let mut group_tokens = Vec::with_capacity(groups.len());
        for group in &groups {
            let caches = group.iter().map(|&i| &*rows[i].cache).collect::<Vec<_>>();
            let decode_rows = group.iter().filter(|&&i| spans[i].q_len == 1).count();
            let layout =
                MixedRowLayout::new(group.iter().map(|&i| spans[i]).collect(), decode_rows)?;
            group_tokens.push(layout.total_tokens());
            steps.push(PreparedPhysicalMixedStep::new(&caches, layout)?);
        }

        let result = (|| -> Result<Tensor> {
            let embeds = order
                .iter()
                .map(|&i| rows[i].embeds.to_dtype(self.dtype))
                .collect::<candle_core::Result<Vec<_>>>()?;
            let mut hidden = if embeds.len() == 1 {
                embeds[0].clone()
            } else {
                Tensor::cat(&embeds, 0)?
            };
            let positions = order
                .iter()
                .flat_map(|&i| {
                    let span = spans[i];
                    (span.start_pos..span.start_pos + span.q_len).map(|p| p as u32)
                })
                .collect::<Vec<_>>();
            let positions = Tensor::from_vec(positions.clone(), positions.len(), &self.device)?;
            let cos = self.rope_cos.index_select(&positions, 0)?;
            let sin = self.rope_sin.index_select(&positions, 0)?;
            let caches = groups
                .iter()
                .map(|group| group.iter().map(|&i| &*rows[i].cache).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            for (index, layer) in self.layers.iter().enumerate() {
                hidden = layer.forward(
                    &hidden,
                    &cos,
                    &sin,
                    cfg,
                    index,
                    &mut steps,
                    &caches,
                    &group_tokens,
                )?;
            }
            // Last position of each row, back in row order.
            let mut offsets = vec![0usize; rows.len()];
            let mut offset = 0usize;
            for &i in &order {
                offset += spans[i].q_len;
                offsets[i] = offset - 1;
            }
            let last = Tensor::from_vec(
                offsets.iter().map(|&o| o as u32).collect::<Vec<_>>(),
                offsets.len(),
                &self.device,
            )?;
            Ok(self.norm.forward(&hidden.index_select(&last, 0)?)?)
        })();

        let outcome = result
            .as_ref()
            .map(|_| ())
            .map_err(|error| error.to_string());
        let mut slots = rows
            .iter_mut()
            .map(|row| Some(&mut *row.cache))
            .collect::<Vec<_>>();
        let mut finish_error = None;
        for (group, step) in groups.iter().zip(steps) {
            let mut caches = group
                .iter()
                .map(|&i| slots[i].take().expect("each row belongs to one group"))
                .collect::<Vec<_>>();
            let status = outcome.clone().map_err(Error::InferenceError);
            if let Err(error) = step.finish(&mut caches, status) {
                finish_error.get_or_insert(error);
            }
        }
        let hidden = result?;
        if let Some(error) = finish_error {
            return Err(error);
        }
        Ok(hidden)
    }
}

fn load_layer(cfg: &Maya2BackboneConfig, vb: &VarBuilder) -> Result<Layer> {
    let hidden = cfg.hidden_size;
    let attn = vb.pp("self_attn");
    let mlp = vb.pp("mlp");
    let weight = |vb: VarBuilder, rows: usize, cols: usize| -> Result<Tensor> {
        Ok(vb.get((rows, cols), "weight")?)
    };
    let qkv = Tensor::cat(
        &[
            weight(attn.pp("q_proj"), cfg.q_out(), hidden)?,
            weight(attn.pp("k_proj"), cfg.kv_out(), hidden)?,
            weight(attn.pp("v_proj"), cfg.kv_out(), hidden)?,
        ],
        0,
    )?;
    let gate_up = Tensor::cat(
        &[
            weight(mlp.pp("gate_proj"), cfg.intermediate_size, hidden)?,
            weight(mlp.pp("up_proj"), cfg.intermediate_size, hidden)?,
        ],
        0,
    )?;
    Ok(Layer {
        input_norm: candle_nn::rms_norm(hidden, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
        qkv,
        q_norm: attn.pp("q_norm").get(cfg.head_dim, "weight")?,
        k_norm: attn.pp("k_norm").get(cfg.head_dim, "weight")?,
        o_proj: weight(attn.pp("o_proj"), hidden, cfg.q_out())?,
        post_norm: candle_nn::rms_norm(
            hidden,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?,
        gate_up,
        down: weight(mlp.pp("down_proj"), hidden, cfg.intermediate_size)?,
    })
}

impl Layer {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cfg: &Maya2BackboneConfig,
        layer: usize,
        steps: &mut [PreparedPhysicalMixedStep],
        caches: &[Vec<&PhysicalPagedKvCache>],
        group_tokens: &[usize],
    ) -> Result<Tensor> {
        let tokens = x.dim(0)?;
        let (q_out, kv_out, head_dim) = (cfg.q_out(), cfg.kv_out(), cfg.head_dim);
        let eps = cfg.rms_norm_eps as f32;
        let qkv = rows_apart_matmul(&self.input_norm.forward(x)?, &self.qkv)?;
        let q = qkv
            .narrow(1, 0, q_out)?
            .reshape((tokens, cfg.num_attention_heads, head_dim))?;
        let k =
            qkv.narrow(1, q_out, kv_out)?
                .reshape((tokens, cfg.num_key_value_heads, head_dim))?;
        let v = qkv
            .narrow(1, q_out + kv_out, kv_out)?
            .reshape((tokens, cfg.num_key_value_heads, head_dim))?
            .contiguous()?;
        let q = candle_nn::ops::rms_norm(&q.contiguous()?, &self.q_norm, eps)?;
        let k = candle_nn::ops::rms_norm(&k.contiguous()?, &self.k_norm, eps)?;
        let q = rotate_half_split(&q, cos, sin)?;
        let k = rotate_half_split(&k, cos, sin)?;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut outputs = Vec::with_capacity(steps.len());
        let mut offset = 0usize;
        for ((step, group), &count) in steps.iter_mut().zip(caches).zip(group_tokens) {
            let slice = |t: &Tensor| t.narrow(0, offset, count)?.contiguous();
            outputs.push(step.write_and_attend(
                group,
                layer,
                &slice(&q)?,
                &slice(&k)?,
                &slice(&v)?,
                scale,
                None,
                None,
            )?);
            offset += count;
        }
        let attn = if outputs.len() == 1 {
            outputs.pop().expect("one group")
        } else {
            Tensor::cat(&outputs, 0)?
        };
        let attn = rows_apart_matmul(&attn.reshape((tokens, q_out))?, &self.o_proj)?;
        let x = x.add(&attn)?;
        let gate_up = rows_apart_matmul(&self.post_norm.forward(&x)?, &self.gate_up)?;
        let gate = gate_up.narrow(1, 0, cfg.intermediate_size)?;
        let up = gate_up.narrow(1, cfg.intermediate_size, cfg.intermediate_size)?;
        let mlp = rows_apart_matmul(&candle_nn::ops::silu(&gate)?.mul(&up)?, &self.down)?;
        Ok(x.add(&mlp)?)
    }
}

/// Qwen's half-split rotation of `x` [tokens, heads, dim] by per-position
/// `cos`/`sin` [tokens, dim / 2].
fn rotate_half_split(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let cos = cos.unsqueeze(1)?;
    let sin = sin.unsqueeze(1)?;
    let first = x1.broadcast_mul(&cos)?.sub(&x2.broadcast_mul(&sin)?)?;
    let second = x1.broadcast_mul(&sin)?.add(&x2.broadcast_mul(&cos)?)?;
    Ok(Tensor::cat(&[first, second], D::Minus1)?.contiguous()?)
}

#[cfg(test)]
mod tests {
    use candle_nn::VarMap;

    use super::*;
    use crate::models::shared::noise::mix;

    /// Maya 2's layout at the paged-attention kernels' head width, still
    /// toy-sized: GQA 2/1, two layers.
    fn tiny_config() -> Maya2BackboneConfig {
        Maya2BackboneConfig {
            hidden_size: 256,
            intermediate_size: 512,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            vocab_size: 64,
        }
    }

    fn tiny_backbone(device: &Device, dtype: DType) -> Maya2Backbone {
        let varmap = VarMap::new();
        let load = || {
            Maya2Backbone::load(
                tiny_config(),
                VarBuilder::from_varmap(&varmap, dtype, device),
                64,
            )
            .unwrap()
        };
        load();
        let mut seed = 5u64;
        for (name, var) in varmap.data().lock().unwrap().iter() {
            let shape = var.shape().clone();
            let values = (0..shape.elem_count())
                .map(|_| {
                    seed = mix(seed.wrapping_add(1));
                    let unit = (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                    if name.ends_with("norm.weight") {
                        1.0 + unit * 0.2
                    } else {
                        unit * 0.2
                    }
                })
                .collect::<Vec<_>>();
            let value = Tensor::from_vec(values, shape, device).unwrap();
            var.set(&value.to_dtype(dtype).unwrap()).unwrap();
        }
        load()
    }

    fn embeds(positions: usize, salt: u64, device: &Device, dtype: DType) -> Tensor {
        let values = (0..positions * 256)
            .map(|i| ((mix(i as u64 ^ salt) >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0)
            .collect::<Vec<_>>();
        Tensor::from_vec(values, (positions, 256), device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
    }

    fn values(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    /// How far row `row`'s hidden states move beside other requests, as the
    /// engine batches them (its pages after or before theirs in the arena):
    /// its prefill beside another prefill, and its decode step beside two
    /// other decode rows, each against the row alone.
    fn batch_variance(device: &Device, dtype: DType, row: usize) -> f32 {
        let backbone = tiny_backbone(device, dtype);
        let cfg = tiny_config();
        // Caches in one arena, as the engine allocates them, so batched rows
        // share each attention call.
        let caches = |rows: usize| {
            crate::models::shared::attention::physical::test_physical_caches_with_dtype_on(
                device,
                dtype,
                cfg.num_hidden_layers,
                cfg.num_key_value_heads,
                cfg.head_dim,
                64,
                rows,
            )
        };
        let prompts = [(7, 1u64), (12, 2), (3, 3)];
        let step = |row: u64| embeds(1, 100 + row, device, dtype);
        // Row 0 alone.
        let mut alone = caches(1).pop().unwrap();
        let alone_prefill = backbone
            .forward_batch(&mut [Maya2BackboneRow {
                embeds: embeds(prompts[row].0, prompts[row].1, device, dtype),
                cache: &mut alone,
            }])
            .unwrap();
        let alone_decode = backbone
            .forward_batch(&mut [Maya2BackboneRow {
                embeds: step(row as u64),
                cache: &mut alone,
            }])
            .unwrap();
        // Row 0 among others.
        let mut caches = caches(3);
        let mut rows = caches
            .iter_mut()
            .zip(prompts)
            .map(|(cache, (positions, salt))| Maya2BackboneRow {
                embeds: embeds(positions, salt, device, dtype),
                cache,
            })
            .collect::<Vec<_>>();
        let batched_prefill = backbone.forward_batch(&mut rows).unwrap();
        drop(rows);
        let mut rows = caches
            .iter_mut()
            .enumerate()
            .map(|(row, cache)| Maya2BackboneRow {
                embeds: step(row as u64),
                cache,
            })
            .collect::<Vec<_>>();
        let batched_decode = backbone.forward_batch(&mut rows).unwrap();
        let diff = |a: &Tensor, b: &Tensor| {
            values(a)
                .iter()
                .zip(values(&b.narrow(0, row, 1).unwrap()))
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max)
        };
        diff(&alone_prefill, &batched_prefill).max(diff(&alone_decode, &batched_decode))
    }

    #[test]
    fn a_rows_hidden_states_do_not_depend_on_its_batch() {
        for row in 0..3 {
            let diff = batch_variance(&Device::Cpu, DType::F32, row);
            assert_eq!(diff, 0.0, "row {row} moves by {diff} beside other rows");
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn a_rows_hidden_states_do_not_depend_on_its_batch_on_metal() {
        use crate::backends::{BackendPreference, DeviceSelector};
        let _lease = crate::backends::metal_test_device_or_skip(
            "a_rows_hidden_states_do_not_depend_on_its_batch_on_metal",
        );
        let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
        if !matches!(profile.device, Device::Metal(_)) {
            return;
        }
        for row in 0..3 {
            let diff = batch_variance(&profile.device, DType::F16, row);
            assert_eq!(diff, 0.0, "row {row} moves by {diff} beside other rows");
        }
    }
}
