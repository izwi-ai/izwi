//! Maya1's language model: Llama 3.2-3B's decoder (`LlamaForCausalLM`,
//! GQA 24/8, head 128, Llama3-scaled RoPE, tied embeddings), family-local on
//! izwi's shared paged-attention primitives.
//!
//! One [`Maya1Trunk::forward`] call carries rows from any requests: the
//! dense layers run once over every token and attention runs once per KV
//! arena through the shared mixed paged step. Each row's logits are computed
//! for its last position and only for the head rows a decode step can
//! sample ([`HEAD_FIRST_TOKEN`].. + [`HEAD_ROWS`]), a slice of the tied
//! embedding table: the full head is 15% of the weights read per token.

use std::sync::Arc;

use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{RmsNorm, VarBuilder};

use super::config::{Maya1LmConfig, HEAD_FIRST_TOKEN, HEAD_ROWS};
use crate::error::{Error, Result};
use crate::models::shared::attention::mixed::{
    MixedRowLayout, MixedRowSpan, PreparedPhysicalMixedStep,
};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::rope::rope_frequencies;

/// A projection `[out, in]`: dense in the model dtype, or quantized.
pub(crate) enum Projection {
    Dense(Tensor),
    Quantized(QMatMul),
}

impl Projection {
    /// `x` [tokens, in] → [tokens, out].
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(match self {
            Self::Dense(weight) => x.matmul(&weight.t()?)?,
            Self::Quantized(weight) => weight.forward(x)?,
        })
    }
}

/// The token table: dense rows to gather, or a quantized table gathered by
/// dequantizing the rows a step needs.
pub(crate) enum TokenTable {
    Dense(Tensor),
    Quantized(crate::models::shared::weights::token_table::QuantizedTokenTable),
}

impl TokenTable {
    fn embed(&self, ids: &Tensor, dtype: DType) -> Result<Tensor> {
        Ok(match self {
            Self::Dense(table) => table.index_select(ids, 0)?,
            Self::Quantized(table) => table.embed(ids)?.to_dtype(dtype)?,
        })
    }
}

pub(crate) struct Maya1Layer {
    pub(crate) input_norm: RmsNorm,
    /// q, k and v stacked: [q_out + 2 kv_out, hidden].
    pub(crate) qkv: Projection,
    pub(crate) o_proj: Projection,
    pub(crate) post_norm: RmsNorm,
    /// gate and up stacked: [2 intermediate, hidden].
    pub(crate) gate_up: Projection,
    pub(crate) down: Projection,
}

pub struct Maya1Trunk {
    cfg: Maya1LmConfig,
    dtype: DType,
    device: Device,
    embed: TokenTable,
    head: Projection,
    layers: Vec<Maya1Layer>,
    norm: RmsNorm,
    /// RoPE cos/sin, [positions, head_dim / 2], built on the host in F64.
    rope_cos: Tensor,
    rope_sin: Tensor,
    max_positions: usize,
}

/// One row of a forward: `tokens` appended to `cache`.
pub struct Maya1ForwardRow<'a> {
    pub tokens: &'a [u32],
    pub cache: &'a mut PhysicalPagedKvCache,
}

impl Maya1Trunk {
    /// Load the decoder from a Transformers-layout `vb` in its dtype; only
    /// `cfg.num_hidden_layers` layers are read (fixtures truncate them).
    pub fn load(cfg: Maya1LmConfig, vb: VarBuilder, max_positions: usize) -> Result<Self> {
        let dtype = vb.dtype();
        let device = vb.device().clone();
        let hidden = cfg.hidden_size;
        let model = vb.pp("model");
        let table = model.get((cfg.vocab_size, hidden), "embed_tokens.weight")?;
        let head = Projection::Dense(
            table
                .narrow(0, HEAD_FIRST_TOKEN as usize, HEAD_ROWS)?
                .contiguous()?,
        );
        let layers = (0..cfg.num_hidden_layers)
            .map(|index| load_dense_layer(&cfg, &model.pp(format!("layers.{index}"))))
            .collect::<Result<Vec<_>>>()?;
        let norm = candle_nn::rms_norm(hidden, cfg.rms_norm_eps, model.pp("norm"))?;
        Self::assemble(
            cfg,
            dtype,
            device,
            TokenTable::Dense(table),
            head,
            layers,
            norm,
            max_positions,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn assemble(
        cfg: Maya1LmConfig,
        dtype: DType,
        device: Device,
        embed: TokenTable,
        head: Projection,
        layers: Vec<Maya1Layer>,
        norm: RmsNorm,
        max_positions: usize,
    ) -> Result<Self> {
        let freqs = rope_frequencies(cfg.head_dim, cfg.rope_theta, &cfg.rope_scaling)?;
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
            embed,
            head,
            layers,
            norm,
            max_positions,
        })
    }

    pub fn config(&self) -> &Maya1LmConfig {
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

    /// Run `rows` through the decoder and return each row's head logits at
    /// its last position, `[rows, HEAD_ROWS]` in row order (F32).
    pub fn forward(&self, rows: &mut [Maya1ForwardRow<'_>]) -> Result<Tensor> {
        if rows.is_empty() {
            return Err(Error::InvalidInput("Maya1 forward has no rows".into()));
        }
        let cfg = &self.cfg;
        let mut spans = Vec::with_capacity(rows.len());
        for row in rows.iter() {
            let count = row.tokens.len();
            let start = row.cache.context_len();
            let end = start.saturating_add(count);
            if count == 0 || end > self.max_positions || end > row.cache.capacity_tokens() {
                return Err(Error::InvalidInput(format!(
                    "Maya1 row of {count} tokens at {start} exceeds its capacity ({} positions, cache {})",
                    self.max_positions,
                    row.cache.capacity_tokens()
                )));
            }
            if let Some(&id) = row.tokens.iter().find(|&&id| id as usize >= cfg.vocab_size) {
                return Err(Error::InvalidInput(format!(
                    "Maya1 token {id} is outside the {}-token vocabulary",
                    cfg.vocab_size
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
        // flattened token order is group after group.
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
            let ids = order
                .iter()
                .flat_map(|&i| rows[i].tokens.iter().copied())
                .collect::<Vec<_>>();
            let ids = Tensor::from_vec(ids.clone(), ids.len(), &self.device)?;
            let mut hidden = self.embed.embed(&ids, self.dtype)?;
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
            // Last token of each row, back in row order.
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
            let tails = hidden.index_select(&last, 0)?;
            let logits = self.head.forward(&self.norm.forward(&tails)?)?;
            Ok(logits.to_dtype(DType::F32)?)
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
        let logits = result?;
        if let Some(error) = finish_error {
            return Err(error);
        }
        Ok(logits)
    }
}

fn load_dense_layer(cfg: &Maya1LmConfig, vb: &VarBuilder) -> Result<Maya1Layer> {
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
    Ok(Maya1Layer {
        input_norm: candle_nn::rms_norm(hidden, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
        qkv: Projection::Dense(qkv),
        o_proj: Projection::Dense(weight(attn.pp("o_proj"), hidden, cfg.q_out())?),
        post_norm: candle_nn::rms_norm(
            hidden,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?,
        gate_up: Projection::Dense(gate_up),
        down: Projection::Dense(weight(mlp.pp("down_proj"), hidden, cfg.intermediate_size)?),
    })
}

impl Maya1Layer {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cfg: &Maya1LmConfig,
        layer: usize,
        steps: &mut [PreparedPhysicalMixedStep],
        caches: &[Vec<&PhysicalPagedKvCache>],
        group_tokens: &[usize],
    ) -> Result<Tensor> {
        let tokens = x.dim(0)?;
        let (q_out, kv_out, head_dim) = (cfg.q_out(), cfg.kv_out(), cfg.head_dim);
        let qkv = self.qkv.forward(&self.input_norm.forward(x)?)?;
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
        let attn = self.o_proj.forward(&attn.reshape((tokens, q_out))?)?;
        let x = x.add(&attn)?;
        let gate_up = self.gate_up.forward(&self.post_norm.forward(&x)?)?;
        let gate = gate_up.narrow(1, 0, cfg.intermediate_size)?;
        let up = gate_up.narrow(1, cfg.intermediate_size, cfg.intermediate_size)?;
        let mlp = self.down.forward(&candle_nn::ops::silu(&gate)?.mul(&up)?)?;
        Ok(x.add(&mlp)?)
    }
}

/// Llama's half-split rotation of `x` [tokens, heads, dim] by per-token
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
pub(crate) mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Tensor, D};
    use candle_nn::VarBuilder;

    use super::*;
    use crate::models::shared::noise::SplitMix64;
    use crate::models::shared::rope::{Llama3RopeScaling, RopeScaling};

    /// Maya1's layout at toy width: GQA 4/2, head 16, two layers; the
    /// vocabulary must still reach the audio codes.
    pub(crate) fn tiny_config() -> Maya1LmConfig {
        Maya1LmConfig {
            hidden_size: 64,
            intermediate_size: 96,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            rope_scaling: RopeScaling::Llama3(Llama3RopeScaling {
                factor: 32.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: 8_192,
            }),
            vocab_size: 156_960,
        }
    }

    fn random(rng: &mut SplitMix64, shape: &[usize], scale: f32) -> Tensor {
        let count = shape.iter().product();
        let values = rng
            .standard_normals(count)
            .into_iter()
            .map(|v| v * scale)
            .collect::<Vec<_>>();
        Tensor::from_vec(values, shape, &Device::Cpu).unwrap()
    }

    pub(crate) fn tiny_weights(cfg: &Maya1LmConfig) -> HashMap<String, Tensor> {
        let mut rng = SplitMix64::new(11);
        let h = cfg.hidden_size;
        let mut weights = HashMap::new();
        weights.insert(
            "model.embed_tokens.weight".into(),
            random(&mut rng, &[cfg.vocab_size, h], 0.5),
        );
        weights.insert(
            "model.norm.weight".into(),
            random(&mut rng, &[h], 0.1).affine(1.0, 1.0).unwrap(),
        );
        for layer in 0..cfg.num_hidden_layers {
            let mut put = |name: &str, shape: &[usize], scale: f32| {
                weights.insert(
                    format!("model.layers.{layer}.{name}"),
                    random(&mut rng, shape, scale),
                );
            };
            put("self_attn.q_proj.weight", &[cfg.q_out(), h], 0.15);
            put("self_attn.k_proj.weight", &[cfg.kv_out(), h], 0.15);
            put("self_attn.v_proj.weight", &[cfg.kv_out(), h], 0.15);
            put("self_attn.o_proj.weight", &[h, cfg.q_out()], 0.15);
            put("mlp.gate_proj.weight", &[cfg.intermediate_size, h], 0.15);
            put("mlp.up_proj.weight", &[cfg.intermediate_size, h], 0.15);
            put("mlp.down_proj.weight", &[h, cfg.intermediate_size], 0.15);
            for norm in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
                let gain = random(&mut rng, &[h], 0.1).affine(1.0, 1.0).unwrap();
                weights.insert(format!("model.layers.{layer}.{norm}"), gain);
            }
        }
        weights
    }

    fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Tensor {
        let variance = x.sqr().unwrap().mean_keepdim(D::Minus1).unwrap();
        x.broadcast_div(&variance.affine(1.0, eps).unwrap().sqrt().unwrap())
            .unwrap()
            .broadcast_mul(weight)
            .unwrap()
    }

    /// A plain Llama forward over the whole sequence (explicit causal
    /// attention, K/V heads repeated): head-row logits at every position.
    pub(crate) fn dense_reference(
        cfg: &Maya1LmConfig,
        w: &HashMap<String, Tensor>,
        tokens: &[u32],
    ) -> Vec<Vec<f32>> {
        let n = tokens.len();
        let (hq, hkv, d) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        let ids = Tensor::from_slice(tokens, n, &Device::Cpu).unwrap();
        let mut x = w["model.embed_tokens.weight"]
            .index_select(&ids, 0)
            .unwrap();
        let freqs = rope_frequencies(d, cfg.rope_theta, &cfg.rope_scaling).unwrap();
        let (cos, sin) = freqs.cos_sin_table(0, n);
        let cos = Tensor::from_vec(cos, (n, d / 2), &Device::Cpu).unwrap();
        let sin = Tensor::from_vec(sin, (n, d / 2), &Device::Cpu).unwrap();
        let mask = Tensor::from_vec(
            (0..n * n)
                .map(|i| {
                    if i % n > i / n {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>(),
            (n, n),
            &Device::Cpu,
        )
        .unwrap();
        for layer in 0..cfg.num_hidden_layers {
            let get = |name: &str| &w[&format!("model.layers.{layer}.{name}")];
            let normed = rms_norm(&x, get("input_layernorm.weight"), cfg.rms_norm_eps);
            let proj = |name: &str, input: &Tensor| input.matmul(&get(name).t().unwrap()).unwrap();
            let q = proj("self_attn.q_proj.weight", &normed)
                .reshape((n, hq, d))
                .unwrap();
            let k = proj("self_attn.k_proj.weight", &normed)
                .reshape((n, hkv, d))
                .unwrap();
            let v = proj("self_attn.v_proj.weight", &normed)
                .reshape((n, hkv, d))
                .unwrap();
            let q = rotate_half_split(&q, &cos, &sin).unwrap();
            let k = rotate_half_split(&k, &cos, &sin).unwrap();
            let mut heads = Vec::new();
            for head in 0..hq {
                let kv = head / (hq / hkv);
                let qh = q.narrow(1, head, 1).unwrap().squeeze(1).unwrap();
                let kh = k.narrow(1, kv, 1).unwrap().squeeze(1).unwrap();
                let vh = v.narrow(1, kv, 1).unwrap().squeeze(1).unwrap();
                let scores = (qh.matmul(&kh.t().unwrap()).unwrap() / (d as f64).sqrt())
                    .unwrap()
                    .add(&mask)
                    .unwrap();
                let weights = candle_nn::ops::softmax_last_dim(&scores).unwrap();
                heads.push(weights.matmul(&vh).unwrap());
            }
            let attn = Tensor::cat(&heads, 1).unwrap();
            x = x.add(&proj("self_attn.o_proj.weight", &attn)).unwrap();
            let normed = rms_norm(&x, get("post_attention_layernorm.weight"), cfg.rms_norm_eps);
            let gate = proj("mlp.gate_proj.weight", &normed);
            let up = proj("mlp.up_proj.weight", &normed);
            let mlp = candle_nn::ops::silu(&gate).unwrap().mul(&up).unwrap();
            x = x.add(&proj("mlp.down_proj.weight", &mlp)).unwrap();
        }
        let x = rms_norm(&x, &w["model.norm.weight"], cfg.rms_norm_eps);
        let head = w["model.embed_tokens.weight"]
            .narrow(0, HEAD_FIRST_TOKEN as usize, HEAD_ROWS)
            .unwrap();
        x.matmul(&head.t().unwrap())
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()
    }

    pub(crate) fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                let diff = (x - y).abs();
                if diff.is_finite() {
                    diff
                } else {
                    f32::INFINITY
                }
            })
            .fold(0.0, |worst, diff| if diff > worst { diff } else { worst })
    }

    #[test]
    fn paged_trunk_matches_a_dense_llama_on_prefill_decode_and_mixed_rows() {
        let cfg = tiny_config();
        let weights = tiny_weights(&cfg);
        let vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &Device::Cpu);
        let trunk = Maya1Trunk::load(cfg.clone(), vb, 64).unwrap();
        let cache = || {
            PhysicalPagedKvCache::scratch_for_probe(
                &Device::Cpu,
                DType::F32,
                cfg.num_hidden_layers,
                cfg.num_key_value_heads,
                cfg.head_dim,
                64,
            )
            .unwrap()
        };
        // A prompt of text-range ids, then audio codes.
        let a = [
            128_259u32, 128_000, 40, 7_000, 52_011, 128_009, 128_260, 128_261, 128_257,
        ];
        let codes = [
            128_266u32, 132_400, 136_500, 140_600, 144_700, 148_800, 152_900,
        ];
        let b = [128_259u32, 128_000, 99, 1_234, 128_257];
        let full_a = a.iter().chain(&codes).copied().collect::<Vec<_>>();
        let reference_a = dense_reference(&cfg, &weights, &full_a);
        let reference_b = dense_reference(&cfg, &weights, &b);

        let (mut cache_a, mut cache_b) = (cache(), cache());
        let prefill = trunk
            .forward(&mut [Maya1ForwardRow {
                tokens: &a,
                cache: &mut cache_a,
            }])
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert!(max_diff(&prefill[0], &reference_a[a.len() - 1]) < 1e-4);
        // Row a decodes one code while row b prefills, in one call.
        for (step, &code) in codes.iter().enumerate() {
            let logits = if step == 0 {
                trunk
                    .forward(&mut [
                        Maya1ForwardRow {
                            tokens: &[code],
                            cache: &mut cache_a,
                        },
                        Maya1ForwardRow {
                            tokens: &b,
                            cache: &mut cache_b,
                        },
                    ])
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap()
            } else {
                trunk
                    .forward(&mut [Maya1ForwardRow {
                        tokens: &[code],
                        cache: &mut cache_a,
                    }])
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap()
            };
            let position = a.len() + step;
            let diff = max_diff(&logits[0], &reference_a[position]);
            assert!(diff < 1e-4, "decode step {step}: {diff}");
            if step == 0 {
                assert!(max_diff(&logits[1], &reference_b[b.len() - 1]) < 1e-4);
            }
        }
        assert_eq!(cache_a.context_len(), full_a.len());
    }
}
