//! T3: Chatterbox's text-to-speech-token model (`t3/t3.py`).
//!
//! A Llama-520M backbone fed its own embeddings: 34 conditioning positions
//! (speaker projection, 32 perceiver queries over the reference's speech
//! tokens, an exaggeration projection), then the text with learned text
//! positions, then speech tokens with learned speech positions. RoPE (Llama3
//! scaled) runs inside the backbone on top of the learned positions.
//!
//! Classifier-free guidance runs a second, unconditional sequence whose text
//! embeddings are zero before the learned positions are added. Each sequence
//! owns its own paged KV, and the two live in different arenas (one state
//! group each). One [`T3Model::forward`] call carries any mix of rows from
//! any requests: the dense layers run once over every token, and attention
//! runs once per arena through the shared mixed paged step.

use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{Embedding, LayerNorm, Linear, Module, RmsNorm, VarBuilder};

use super::conds::T3VoiceConditioning;
use super::config::T3Config;
use crate::error::{Error, Result};
use crate::models::shared::attention::mixed::{
    MixedRowLayout, MixedRowSpan, PreparedPhysicalMixedStep,
};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::rope::rope_frequencies;

pub struct T3Model {
    cfg: T3Config,
    dtype: DType,
    device: Device,
    text_emb: Embedding,
    speech_emb: Embedding,
    text_pos: Embedding,
    speech_pos: Embedding,
    speaker_proj: Linear,
    /// `emotion_adv_fc.weight[:, 0]` as a row, [1, hidden].
    exaggeration_row: Tensor,
    perceiver: Perceiver,
    layers: Vec<T3Layer>,
    norm: RmsNorm,
    speech_head: Linear,
    /// RoPE cos/sin, [positions, head_dim / 2], built on the host in F64.
    rope_cos: Tensor,
    rope_sin: Tensor,
    max_positions: usize,
}

/// A voice's conditioning, embedded once and reused by every request with
/// that voice: the speaker row and the perceiver output, [33, hidden].
#[derive(Debug, Clone)]
pub struct T3PreparedVoice {
    rows: Tensor,
}

impl T3PreparedVoice {
    pub fn device_bytes(&self) -> usize {
        self.rows.elem_count() * self.rows.dtype().size_in_bytes()
    }
}

/// One row of a T3 forward: `embeds` [tokens, hidden] appended to `cache`.
pub struct T3ForwardRow<'a> {
    pub embeds: Tensor,
    pub cache: &'a mut PhysicalPagedKvCache,
}

struct T3Layer {
    input_norm: RmsNorm,
    qkv: Linear,
    o_proj: Linear,
    post_norm: RmsNorm,
    gate_up: Linear,
    down: Linear,
}

struct Perceiver {
    query: Tensor,
    norm: LayerNorm,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    proj_out: Linear,
    heads: usize,
}

impl T3Model {
    pub fn load(cfg: T3Config, vb: VarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let device = vb.device().clone();
        let hidden = cfg.hidden_size;
        let embedding = |rows: usize, name: &str| -> Result<Embedding> {
            Ok(candle_nn::embedding(rows, hidden, vb.pp(name))?)
        };
        let text_emb = embedding(cfg.text_vocab_size, "text_emb")?;
        let speech_emb = embedding(cfg.speech_vocab_size, "speech_emb")?;
        let text_pos = embedding(cfg.text_positions, "text_pos_emb.emb")?;
        let speech_pos = embedding(cfg.speech_positions, "speech_pos_emb.emb")?;

        let cond = vb.pp("cond_enc");
        let speaker_proj = candle_nn::linear(cfg.speaker_embed_size, hidden, cond.pp("spkr_enc"))?;
        let exaggeration_row = cond
            .get((hidden, 1), "emotion_adv_fc.weight")?
            .reshape((1, hidden))?;
        let perceiver = Perceiver::load(&cfg, cond.pp("perceiver"))?;

        let tfmr = vb.pp("tfmr");
        let layers = (0..cfg.num_hidden_layers)
            .map(|index| T3Layer::load(&cfg, tfmr.pp(format!("layers.{index}"))))
            .collect::<Result<Vec<_>>>()?;
        let norm = candle_nn::rms_norm(hidden, cfg.rms_norm_eps, tfmr.pp("norm"))?;
        let speech_head =
            candle_nn::linear_no_bias(hidden, cfg.speech_vocab_size, vb.pp("speech_head"))?;

        let max_positions = cfg.conditioning_len() + cfg.text_positions + cfg.speech_positions;
        let freqs = rope_frequencies(cfg.head_dim, cfg.rope_theta, &cfg.rope_scaling())?;
        let (cos, sin) = freqs.cos_sin_table(0, max_positions);
        let pairs = cfg.head_dim / 2;
        let rope_cos =
            Tensor::from_vec(cos, (max_positions, pairs), &Device::Cpu)?.to_dtype(dtype)?;
        let rope_sin =
            Tensor::from_vec(sin, (max_positions, pairs), &Device::Cpu)?.to_dtype(dtype)?;

        Ok(Self {
            text_emb,
            speech_emb,
            text_pos,
            speech_pos,
            speaker_proj,
            exaggeration_row,
            perceiver,
            layers,
            norm,
            speech_head,
            rope_cos: rope_cos.to_device(&device)?,
            rope_sin: rope_sin.to_device(&device)?,
            max_positions,
            cfg,
            dtype,
            device,
        })
    }

    pub fn config(&self) -> &T3Config {
        &self.cfg
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Positions one sequence can reach: conditioning, the learned text
    /// table and the learned speech table.
    pub fn max_positions(&self) -> usize {
        self.max_positions
    }

    fn ids(&self, ids: &[u32]) -> Result<Tensor> {
        Ok(Tensor::from_slice(ids, ids.len(), &self.device)?)
    }

    fn positions(&self, start: usize, len: usize) -> Result<Tensor> {
        Ok(Tensor::arange(
            start as u32,
            (start + len) as u32,
            &self.device,
        )?)
    }

    /// Embed a voice: the speaker row, then the perceiver over the
    /// reference speech tokens with their learned speech positions.
    pub fn prepare_voice(&self, voice: &T3VoiceConditioning) -> Result<T3PreparedVoice> {
        let tokens = &voice.prompt_speech_tokens;
        if tokens.is_empty() || tokens.len() > self.cfg.speech_cond_prompt_len {
            return Err(Error::InvalidInput(format!(
                "Chatterbox T3 needs 1..={} reference speech tokens, got {}",
                self.cfg.speech_cond_prompt_len,
                tokens.len()
            )));
        }
        let speaker = Tensor::from_slice(
            &voice.speaker_embedding,
            (1, self.cfg.speaker_embed_size),
            &self.device,
        )?
        .to_dtype(self.dtype)?;
        let speaker = self.speaker_proj.forward(&speaker)?;
        let prompt = self
            .speech_emb
            .forward(&self.ids(tokens)?)?
            .add(&self.speech_pos.forward(&self.positions(0, tokens.len())?)?)?;
        let perceived = self.perceiver.forward(&prompt)?;
        Ok(T3PreparedVoice {
            rows: Tensor::cat(&[speaker, perceived], 0)?.contiguous()?,
        })
    }

    /// The 34 conditioning positions for a voice at `exaggeration`.
    pub fn conditioning(&self, voice: &T3PreparedVoice, exaggeration: f32) -> Result<Tensor> {
        let exaggeration = self.exaggeration_row.affine(f64::from(exaggeration), 0.0)?;
        Ok(Tensor::cat(&[voice.rows.clone(), exaggeration], 0)?)
    }

    /// Prefill embeddings of the conditional and (with guidance) the
    /// unconditional sequence: conditioning, framed text with learned text
    /// positions, and the speech start token twice (upstream's prompt
    /// carries one in its speech tokens and appends another), both at speech
    /// position 0.
    pub fn prefill_embeds(
        &self,
        conditioning: &Tensor,
        text_ids: &[u32],
        guided: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        if text_ids.len() < 2 || text_ids.len() > self.cfg.text_positions {
            return Err(Error::InvalidInput(format!(
                "Chatterbox text sequence has {} tokens; it must hold start and stop and fit {} positions",
                text_ids.len(),
                self.cfg.text_positions
            )));
        }
        if let Some(&id) = text_ids
            .iter()
            .find(|&&id| id as usize >= self.cfg.text_vocab_size)
        {
            return Err(Error::InvalidInput(format!(
                "Chatterbox text token {id} is outside the {}-token vocabulary",
                self.cfg.text_vocab_size
            )));
        }
        let text_positions = self.text_pos.forward(&self.positions(0, text_ids.len())?)?;
        let start = self.speech_step_embed(self.cfg.start_speech_token, 0)?;
        let row = |text: Tensor| -> Result<Tensor> {
            Ok(Tensor::cat(
                &[conditioning.clone(), text, start.clone(), start.clone()],
                0,
            )?)
        };
        let text = self
            .text_emb
            .forward(&self.ids(text_ids)?)?
            .add(&text_positions)?;
        let conditional = row(text)?;
        let unconditional = guided.then(|| row(text_positions)).transpose()?;
        Ok((conditional, unconditional))
    }

    /// The embedding of `token` at speech position `position` ([1, hidden]).
    pub fn speech_step_embed(&self, token: u32, position: usize) -> Result<Tensor> {
        if position >= self.cfg.speech_positions {
            return Err(Error::InvalidInput(format!(
                "Chatterbox speech position {position} exceeds the {}-row table",
                self.cfg.speech_positions
            )));
        }
        Ok(self
            .speech_emb
            .forward(&self.ids(&[token])?)?
            .add(&self.speech_pos.forward(&self.positions(position, 1)?)?)?)
    }

    /// Run `rows` through the backbone and return each row's speech logits
    /// at its last position, `[rows, speech_vocab]` in row order (F32).
    pub fn forward(&self, rows: &mut [T3ForwardRow<'_>]) -> Result<Tensor> {
        if rows.is_empty() {
            return Err(Error::InvalidInput(
                "Chatterbox T3 forward has no rows".into(),
            ));
        }
        let hidden_size = self.cfg.hidden_size;
        let mut spans = Vec::with_capacity(rows.len());
        for row in rows.iter() {
            let (count, hidden) = row.embeds.dims2()?;
            let start = row.cache.context_len();
            let end = start.saturating_add(count);
            if count == 0
                || hidden != hidden_size
                || end > self.max_positions
                || end > row.cache.capacity_tokens()
            {
                return Err(Error::InvalidInput(format!(
                    "Chatterbox T3 row of {count} tokens at {start} exceeds its shape or capacity ({} positions, cache {})",
                    self.max_positions,
                    row.cache.capacity_tokens()
                )));
            }
            row.cache.validate_model(
                self.cfg.num_hidden_layers,
                self.cfg.num_attention_heads,
                self.cfg.head_dim,
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
        {
            for group in &groups {
                let caches = group.iter().map(|&i| &*rows[i].cache).collect::<Vec<_>>();
                let decode_rows = group.iter().filter(|&&i| spans[i].q_len == 1).count();
                let layout =
                    MixedRowLayout::new(group.iter().map(|&i| spans[i]).collect(), decode_rows)?;
                group_tokens.push(layout.total_tokens());
                steps.push(PreparedPhysicalMixedStep::new(&caches, layout)?);
            }
        }

        let result = (|| -> Result<Tensor> {
            let embeds = order
                .iter()
                .map(|&i| rows[i].embeds.to_dtype(self.dtype))
                .collect::<candle_core::Result<Vec<_>>>()?;
            let mut hidden = Tensor::cat(&embeds, 0)?;
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
                    &self.cfg,
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
            let logits = self.speech_head.forward(&self.norm.forward(&tails)?)?;
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

impl T3Layer {
    fn load(cfg: &T3Config, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let attn = vb.pp("self_attn");
        let mlp = vb.pp("mlp");
        let weight = |vb: &VarBuilder, rows: usize, cols: usize| -> Result<Tensor> {
            Ok(vb.get((rows, cols), "weight")?)
        };
        let qkv = Tensor::cat(
            &[
                weight(&attn.pp("q_proj"), hidden, hidden)?,
                weight(&attn.pp("k_proj"), hidden, hidden)?,
                weight(&attn.pp("v_proj"), hidden, hidden)?,
            ],
            0,
        )?;
        let gate_up = Tensor::cat(
            &[
                weight(&mlp.pp("gate_proj"), cfg.intermediate_size, hidden)?,
                weight(&mlp.pp("up_proj"), cfg.intermediate_size, hidden)?,
            ],
            0,
        )?;
        Ok(Self {
            input_norm: candle_nn::rms_norm(hidden, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            qkv: Linear::new(qkv, None),
            o_proj: candle_nn::linear_no_bias(hidden, hidden, attn.pp("o_proj"))?,
            post_norm: candle_nn::rms_norm(
                hidden,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            gate_up: Linear::new(gate_up, None),
            down: candle_nn::linear_no_bias(cfg.intermediate_size, hidden, mlp.pp("down_proj"))?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cfg: &T3Config,
        layer: usize,
        steps: &mut [PreparedPhysicalMixedStep],
        caches: &[Vec<&PhysicalPagedKvCache>],
        group_tokens: &[usize],
    ) -> Result<Tensor> {
        let tokens = x.dim(0)?;
        let (heads, head_dim, hidden) = (cfg.num_attention_heads, cfg.head_dim, cfg.hidden_size);
        let qkv = self.qkv.forward(&self.input_norm.forward(x)?)?;
        let q = qkv
            .narrow(1, 0, hidden)?
            .reshape((tokens, heads, head_dim))?;
        let k = qkv
            .narrow(1, hidden, hidden)?
            .reshape((tokens, heads, head_dim))?;
        let v = qkv
            .narrow(1, 2 * hidden, hidden)?
            .reshape((tokens, heads, head_dim))?
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
        let attn = self.o_proj.forward(&attn.reshape((tokens, hidden))?)?;
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

impl Perceiver {
    fn load(cfg: &T3Config, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let attn = vb.pp("attn");
        Ok(Self {
            query: vb
                .get((1, cfg.perceiver_queries, hidden), "pre_attention_query")?
                .i(0)?,
            norm: candle_nn::layer_norm(hidden, 1e-5, attn.pp("norm"))?,
            to_q: candle_nn::linear(hidden, hidden, attn.pp("to_q"))?,
            to_k: candle_nn::linear(hidden, hidden, attn.pp("to_k"))?,
            to_v: candle_nn::linear(hidden, hidden, attn.pp("to_v"))?,
            proj_out: candle_nn::linear(hidden, hidden, attn.pp("proj_out"))?,
            heads: cfg.perceiver_heads,
        })
    }

    /// `AttentionBlock2` applied twice with the same weights: the learned
    /// queries attend to the prompt, then to themselves.
    fn forward(&self, prompt: &Tensor) -> Result<Tensor> {
        let cross = self.block(&self.query, prompt)?;
        self.block(&cross, &cross)
    }

    fn block(&self, x1: &Tensor, x2: &Tensor) -> Result<Tensor> {
        let (queries, hidden) = x1.dims2()?;
        let keys = x2.dim(0)?;
        let head_dim = hidden / self.heads;
        let x1n = self.norm.forward(x1)?;
        let x2n = self.norm.forward(x2)?;
        let split = |t: Tensor, len: usize| -> candle_core::Result<Tensor> {
            t.reshape((len, self.heads, head_dim))?
                .transpose(0, 1)?
                .contiguous()
        };
        let q = split(self.to_q.forward(&x1n)?, queries)?;
        let k = split(self.to_k.forward(&x2n)?, keys)?;
        let v = split(self.to_v.forward(&x2n)?, keys)?;
        let scores = q
            .matmul(&k.t()?)?
            .affine(1.0 / (head_dim as f64).sqrt(), 0.0)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let attended = weights
            .matmul(&v)?
            .transpose(0, 1)?
            .reshape((queries, hidden))?;
        Ok(x1.add(&self.proj_out.forward(&attended)?)?)
    }
}
