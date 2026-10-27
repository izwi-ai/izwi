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
        let qkv = self.input_norm.forward(x)?.matmul(&self.qkv.t()?)?;
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
        let attn = attn.reshape((tokens, q_out))?.matmul(&self.o_proj.t()?)?;
        let x = x.add(&attn)?;
        let gate_up = self.post_norm.forward(&x)?.matmul(&self.gate_up.t()?)?;
        let gate = gate_up.narrow(1, 0, cfg.intermediate_size)?;
        let up = gate_up.narrow(1, cfg.intermediate_size, cfg.intermediate_size)?;
        let mlp = candle_nn::ops::silu(&gate)?
            .mul(&up)?
            .matmul(&self.down.t()?)?;
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
