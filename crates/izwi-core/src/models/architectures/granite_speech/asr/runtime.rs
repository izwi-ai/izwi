//! Native Candle runtime for Granite Speech ASR.

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{
    batch_norm, conv1d, conv1d_no_bias, embedding, layer_norm, linear, linear_no_bias, ops,
    BatchNorm, Conv1d, Conv1dConfig, Embedding, LayerNorm, Linear, Module, ModuleT, RmsNorm,
    VarBuilder,
};

use crate::backends::DeviceProfile;
use crate::error::{Error, Result};
use crate::models::architectures::qwen3::core::{causal_mask, repeat_kv, Qwen3Cache};

use super::config::{GraniteSpeechConfig, GraniteSpeechEncoderConfig, GraniteTextConfig};
use super::preprocessor::GraniteSpeechAudioFeatures;
use super::prompt::{GraniteSpeechPrompt, GraniteSpeechSpecialTokens};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraniteSpeechGenerationStats {
    pub prompt_tokens: usize,
    pub audio_tokens: usize,
    pub generated_tokens: usize,
    pub stop_reason: String,
    pub stop_token: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraniteSpeechGeneration {
    pub token_ids: Vec<u32>,
    pub text: String,
    pub stats: GraniteSpeechGenerationStats,
}

pub struct GraniteSpeechRuntime {
    device: Device,
    dtype: DType,
    encoder: GraniteSpeechEncoder,
    projector: GraniteSpeechProjector,
    text_model: GraniteLanguageModel,
}

impl GraniteSpeechRuntime {
    pub fn load(
        shard_paths: &[PathBuf],
        config: &GraniteSpeechConfig,
        device: &DeviceProfile,
        dtype: DType,
    ) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(shard_paths, dtype, &device.device).map_err(|err| {
                Error::ModelLoadError(format!(
                    "Failed to mmap Granite Speech safetensors: {err}"
                ))
            })?
        };
        let encoder = GraniteSpeechEncoder::load(&config.encoder_config, vb.pp("encoder"))?;
        let projector = GraniteSpeechProjector::load(config, vb.pp("projector"))?;
        let text_model = GraniteLanguageModel::load(&config.text_config, vb.pp("language_model"))?;
        Ok(Self {
            device: device.device.clone(),
            dtype,
            encoder,
            projector,
            text_model,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn audio_embeddings(&self, features: &GraniteSpeechAudioFeatures) -> Result<Tensor> {
        let frames = features.encoder_frames;
        let dim = features.encoder_dim;
        if frames == 0 || dim == 0 {
            return Err(Error::InvalidInput(
                "Granite Speech audio produced no encoder features".to_string(),
            ));
        }
        let flat = features
            .input_features
            .iter()
            .flat_map(|frame| frame.iter().copied())
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(flat, (1, frames, dim), &self.device)?
            .to_dtype(self.dtype)?;
        let encoded = self.encoder.forward(&input)?;
        self.projector.forward(&encoded)
    }

    pub fn generate(
        &self,
        prompt: &GraniteSpeechPrompt,
        special_tokens: &GraniteSpeechSpecialTokens,
        audio_embeds: &Tensor,
        max_new_tokens: usize,
        extra_stop_token_ids: &[u32],
        stop_sequences: &[String],
        decode: &mut dyn FnMut(&[u32]) -> Result<String>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<GraniteSpeechGeneration> {
        let audio_tokens = audio_embeds.dim(1)?;
        let input_ids = expand_audio_tokens(
            &prompt.input_ids,
            &prompt.audio_token_positions,
            audio_tokens,
            special_tokens.audio_token_id,
        )?;
        let audio_start = prompt
            .audio_token_positions
            .first()
            .copied()
            .ok_or_else(|| Error::InvalidInput("Granite prompt has no audio token".to_string()))?;

        let mut cache = Qwen3Cache::new(self.text_model.num_layers());
        let mut logits = self.text_model.forward_prompt_with_audio(
            &input_ids,
            audio_start,
            audio_tokens,
            audio_embeds,
            &mut cache,
        )?;
        let mut generated = Vec::new();
        let mut rendered = String::new();
        let mut stop_reason = "max_tokens".to_string();
        let mut stop_token = None;
        let stop_tokens = stop_token_set(special_tokens, extra_stop_token_ids);

        for step in 0..max_new_tokens.max(1) {
            let token = argmax_last_logits(&logits)?;
            if stop_tokens.contains(&token) {
                stop_reason = "stop_token".to_string();
                stop_token = Some(token);
                break;
            }

            generated.push(token);
            let mut next_text = decode(&generated)?;
            let stopped_on_sequence = truncate_at_stop_sequence(&mut next_text, stop_sequences);
            if stopped_on_sequence {
                stop_reason = "stop_sequence".to_string();
            }
            if next_text.len() > rendered.len() {
                let delta = &next_text[rendered.len()..];
                on_delta(delta);
            }
            rendered = next_text;
            if stopped_on_sequence {
                break;
            }

            let token_tensor = Tensor::from_vec(vec![token], (1, 1), &self.device)?;
            logits = self.text_model.forward(
                &token_tensor,
                input_ids.len() + step,
                Some(&mut cache),
            )?;
        }

        Ok(GraniteSpeechGeneration {
            text: rendered,
            token_ids: generated.clone(),
            stats: GraniteSpeechGenerationStats {
                prompt_tokens: input_ids.len(),
                audio_tokens,
                generated_tokens: generated.len(),
                stop_reason,
                stop_token,
            },
        })
    }
}

fn expand_audio_tokens(
    input_ids: &[u32],
    audio_token_positions: &[usize],
    audio_tokens: usize,
    audio_token_id: u32,
) -> Result<Vec<u32>> {
    if audio_token_positions.len() != 1 {
        return Err(Error::InvalidInput(format!(
            "Granite Speech expects exactly one audio placeholder, got {}",
            audio_token_positions.len()
        )));
    }
    if audio_tokens == 0 {
        return Err(Error::InvalidInput(
            "Granite Speech projected zero audio tokens".to_string(),
        ));
    }
    let pos = audio_token_positions[0];
    if pos >= input_ids.len() {
        return Err(Error::InvalidInput(
            "Granite Speech audio placeholder is out of prompt bounds".to_string(),
        ));
    }
    let mut expanded = Vec::with_capacity(input_ids.len() + audio_tokens.saturating_sub(1));
    expanded.extend_from_slice(&input_ids[..pos]);
    expanded.extend(std::iter::repeat_n(audio_token_id, audio_tokens));
    expanded.extend_from_slice(&input_ids[pos + 1..]);
    Ok(expanded)
}

fn stop_token_set(special_tokens: &GraniteSpeechSpecialTokens, extra: &[u32]) -> Vec<u32> {
    let mut tokens = vec![special_tokens.eos_token_id, special_tokens.pad_token_id];
    tokens.extend_from_slice(extra);
    tokens.sort_unstable();
    tokens.dedup();
    tokens
}

fn truncate_at_stop_sequence(text: &mut String, stop_sequences: &[String]) -> bool {
    for stop in stop_sequences
        .iter()
        .map(|value| value.as_str())
        .filter(|value| !value.is_empty())
    {
        if let Some(stop_at) = text.find(stop) {
            text.truncate(stop_at);
            return true;
        }
    }
    false
}

fn argmax_last_logits(logits: &Tensor) -> Result<u32> {
    let last = match logits.dims() {
        [1, vocab] => logits.reshape((*vocab,))?,
        [1, 1, vocab] => logits.reshape((*vocab,))?,
        dims => {
            return Err(Error::InferenceError(format!(
                "Granite Speech logits expected [1,vocab] or [1,1,vocab], got {dims:?}"
            )));
        }
    };
    let values = last.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let (idx, _) = values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .ok_or_else(|| Error::InferenceError("Granite Speech empty logits".to_string()))?;
    Ok(idx as u32)
}

struct GraniteSpeechEncoder {
    input_linear: Linear,
    layers: Vec<GraniteConformerBlock>,
    out: Linear,
    out_mid: Linear,
    cat_hidden_layers: Vec<usize>,
}

impl GraniteSpeechEncoder {
    fn load(config: &GraniteSpeechEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let input_linear = linear(config.input_dim, config.hidden_dim, vb.pp("input_linear"))?;
        let mut layers = Vec::with_capacity(config.num_layers);
        for idx in 0..config.num_layers {
            layers.push(GraniteConformerBlock::load(
                config,
                vb.pp(format!("layers.{idx}")),
            )?);
        }
        let out = linear(config.hidden_dim, config.output_dim, vb.pp("out"))?;
        let out_mid = linear(config.output_dim, config.hidden_dim, vb.pp("out_mid"))?;
        Ok(Self {
            input_linear,
            layers,
            out,
            out_mid,
            cat_hidden_layers: config.cat_hidden_layers.clone(),
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = self.input_linear.forward(input)?;
        let mut exported = Vec::with_capacity(self.cat_hidden_layers.len() + 1);
        if self.cat_hidden_layers.contains(&0) {
            exported.push(x.clone());
        }
        let midpoint = self.layers.len() / 2;
        for (idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            let layer_idx = idx + 1;
            if self.cat_hidden_layers.contains(&layer_idx) {
                exported.push(x.clone());
            }
            if layer_idx == midpoint {
                let mid = self.out.forward(&x)?;
                let mid = ops::softmax_last_dim(&mid)?;
                x = x.broadcast_add(&self.out_mid.forward(&mid)?)?;
            }
        }
        if exported.is_empty() {
            Ok(x)
        } else {
            exported.push(x);
            let refs = exported.iter().collect::<Vec<_>>();
            Tensor::cat(&refs, 2).map_err(Error::from)
        }
    }
}

struct GraniteConformerBlock {
    ff1: GraniteConformerFeedForward,
    attn: GraniteConformerAttention,
    conv: GraniteConformerConv,
    ff2: GraniteConformerFeedForward,
    post_norm: LayerNorm,
}

impl GraniteConformerBlock {
    fn load(config: &GraniteSpeechEncoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ff1: GraniteConformerFeedForward::load(config, vb.pp("ff1"))?,
            attn: GraniteConformerAttention::load(config, vb.pp("attn"))?,
            conv: GraniteConformerConv::load(config, vb.pp("conv"))?,
            ff2: GraniteConformerFeedForward::load(config, vb.pp("ff2"))?,
            post_norm: layer_norm(config.hidden_dim, 1e-5, vb.pp("post_norm"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let ff1 = self.ff1.forward(x)?;
        let mut out = x.broadcast_add(&(ff1 * 0.5)?)?;
        let attn = self.attn.forward(&out)?;
        out = out.broadcast_add(&attn)?;
        let conv = self.conv.forward(&out)?;
        out = out.broadcast_add(&conv)?;
        let ff2 = self.ff2.forward(&out)?;
        out = out.broadcast_add(&(ff2 * 0.5)?)?;
        self.post_norm.forward(&out).map_err(Error::from)
    }
}

struct GraniteConformerFeedForward {
    pre_norm: LayerNorm,
    up_proj: Linear,
    down_proj: Linear,
}

impl GraniteConformerFeedForward {
    fn load(config: &GraniteSpeechEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let hidden = config.hidden_dim;
        let intermediate = hidden * config.feedforward_mult;
        Ok(Self {
            pre_norm: layer_norm(hidden, 1e-5, vb.pp("pre_norm"))?,
            up_proj: linear(hidden, intermediate, vb.pp("up_proj"))?,
            down_proj: linear(intermediate, hidden, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.pre_norm.forward(x)?;
        let x = self.up_proj.forward(&x)?.silu()?;
        self.down_proj.forward(&x).map_err(Error::from)
    }
}

struct GraniteConformerAttention {
    pre_norm: LayerNorm,
    to_q: Linear,
    to_kv: Linear,
    to_out: Linear,
    rel_pos_emb: Embedding,
    attention_dists: Tensor,
    context_size: usize,
    num_heads: usize,
    dim_head: usize,
}

impl GraniteConformerAttention {
    fn load(config: &GraniteSpeechEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let inner = config.dim_head * config.num_heads;
        let attention_dists =
            encoder_attention_dists(config.context_size, config.max_pos_emb, vb.device())?;
        Ok(Self {
            pre_norm: layer_norm(config.hidden_dim, 1e-5, vb.pp("pre_norm"))?,
            to_q: linear_no_bias(config.hidden_dim, inner, vb.pp("to_q"))?,
            to_kv: linear_no_bias(config.hidden_dim, inner * 2, vb.pp("to_kv"))?,
            to_out: linear(inner, config.hidden_dim, vb.pp("to_out"))?,
            rel_pos_emb: embedding(
                2 * config.max_pos_emb + 1,
                config.dim_head,
                vb.pp("rel_pos_emb"),
            )?,
            attention_dists,
            context_size: config.context_size,
            num_heads: config.num_heads,
            dim_head: config.dim_head,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.pre_norm.forward(x)?;
        let (batch, seq_len, _) = x.dims3()?;
        let nblocks = seq_len.saturating_add(self.context_size - 1) / self.context_size;
        let padded_len = nblocks * self.context_size;
        let pad = padded_len.saturating_sub(seq_len);
        let padded = if pad > 0 {
            x.pad_with_zeros(1, 0, pad)?
        } else {
            x
        };

        let q = self.to_q.forward(&padded)?;
        let kv = self.to_kv.forward(&padded)?;
        let k = kv.narrow(2, 0, self.num_heads * self.dim_head)?;
        let v = kv.narrow(2, self.num_heads * self.dim_head, self.num_heads * self.dim_head)?;
        let q = q
            .reshape((batch, nblocks, self.context_size, self.num_heads, self.dim_head))?
            .transpose(2, 3)?;
        let k = k
            .reshape((batch, nblocks, self.context_size, self.num_heads, self.dim_head))?
            .transpose(2, 3)?;
        let v = v
            .reshape((batch, nblocks, self.context_size, self.num_heads, self.dim_head))?
            .transpose(2, 3)?;

        let pos_bias = self.relative_position_bias(&q)?;
        let scale = 1.0 / (self.dim_head as f64).sqrt();
        let mut attn = q.matmul(&k.t()?)?;
        attn = (attn * scale)?.broadcast_add(&pos_bias)?;
        if pad > 0 {
            let mask = encoder_padding_mask(
                self.context_size,
                seq_len % self.context_size,
                nblocks,
                attn.device(),
                attn.dtype(),
            )?;
            attn = attn.broadcast_add(&mask)?;
        }
        let attn = ops::softmax(&attn.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(q.dtype())?;
        let out = attn.matmul(&v)?;
        let out = out
            .transpose(2, 3)?
            .reshape((batch, padded_len, self.num_heads * self.dim_head))?
            .narrow(1, 0, seq_len)?;
        self.to_out.forward(&out).map_err(Error::from)
    }

    fn relative_position_bias(&self, q: &Tensor) -> Result<Tensor> {
        let out_dtype = q.dtype();
        let rel = self
            .rel_pos_emb
            .forward(&self.attention_dists)?
            .to_dtype(out_dtype)?;
        let (batch, blocks, heads, context, dim) = q.dims5()?;
        let q = q
            .reshape((batch * blocks * heads, context, dim))?
            .to_dtype(DType::F32)?;
        let rel = rel.to_dtype(DType::F32)?;
        let mut rows = Vec::with_capacity(context);
        for query_idx in 0..context {
            let q_row = q.narrow(1, query_idx, 1)?;
            let rel_row = rel.narrow(0, query_idx, 1)?.squeeze(0)?;
            rows.push(relative_position_score_row(&q_row, &rel_row)?);
        }
        let refs = rows.iter().collect::<Vec<_>>();
        let out = Tensor::cat(&refs, 1)?
            .reshape((batch, blocks, heads, context, context))?;
        (out / (dim as f64).sqrt())?.to_dtype(out_dtype).map_err(Error::from)
    }
}

fn relative_position_score_row(q_row: &Tensor, rel_row: &Tensor) -> Result<Tensor> {
    let q_row = q_row.squeeze(1)?;
    q_row.matmul(&rel_row.t()?)?.unsqueeze(1).map_err(Error::from)
}

fn encoder_attention_dists(context: usize, max_pos_emb: usize, device: &Device) -> Result<Tensor> {
    let mut values = Vec::with_capacity(context * context);
    for i in 0..context {
        for j in 0..context {
            let dist = i as isize - j as isize;
            let clamped = dist.clamp(-(context as isize), context as isize);
            values.push((clamped + max_pos_emb as isize) as u32);
        }
    }
    Tensor::from_vec(values, (context, context), device).map_err(Error::from)
}

fn encoder_padding_mask(
    context: usize,
    remainder: usize,
    nblocks: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let valid = remainder.max(1);
    let mut data = vec![0f32; context * context];
    for q in 0..context {
        for k in 0..context {
            if q >= valid || k >= valid {
                data[q * context + k] = -1e4;
            }
        }
    }
    Tensor::from_vec(data, (1, nblocks, 1, context, context), device)?
        .to_dtype(dtype)
        .map_err(Error::from)
}

struct GraniteConformerConv {
    norm: LayerNorm,
    up_conv: Conv1d,
    depth_conv: Conv1d,
    batch_norm: BatchNorm,
    down_conv: Conv1d,
}

impl GraniteConformerConv {
    fn load(config: &GraniteSpeechEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let hidden = config.hidden_dim;
        let inner = hidden * config.conv_expansion_factor;
        let mut depth_cfg = Conv1dConfig::default();
        depth_cfg.groups = inner;
        Ok(Self {
            norm: layer_norm(hidden, 1e-5, vb.pp("norm"))?,
            up_conv: conv1d(hidden, inner * 2, 1, Conv1dConfig::default(), vb.pp("up_conv"))?,
            depth_conv: conv1d_no_bias(
                inner,
                inner,
                config.conv_kernel_size,
                depth_cfg,
                vb.pp("depth_conv.conv"),
            )?,
            batch_norm: batch_norm(inner, 1e-5, vb.pp("batch_norm"))?,
            down_conv: conv1d(inner, hidden, 1, Conv1dConfig::default(), vb.pp("down_conv"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.norm.forward(x)?;
        let x = x.transpose(1, 2)?.contiguous()?;
        let x = self.up_conv.forward(&x)?;
        let x = glu_channels(&x)?;
        let pad = self.depth_conv_weight_width() / 2;
        let right = pad.saturating_sub((self.depth_conv_weight_width() + 1) % 2);
        let x = x.pad_with_zeros(2, pad, right)?;
        let x = self.depth_conv.forward(&x)?;
        let x = self.batch_norm.forward_t(&x, false)?.silu()?;
        self.down_conv
            .forward(&x)?
            .transpose(1, 2)?
            .contiguous()
            .map_err(Error::from)
    }

    fn depth_conv_weight_width(&self) -> usize {
        self.depth_conv.weight().dims().get(2).copied().unwrap_or(1)
    }
}

fn glu_channels(x: &Tensor) -> Result<Tensor> {
    let channels = x.dim(1)?;
    let half = channels / 2;
    let gate = x.narrow(1, 0, half)?;
    let value = x.narrow(1, half, half)?;
    gate.broadcast_mul(&ops::sigmoid(&value)?)
        .map_err(Error::from)
}

struct GraniteSpeechProjector {
    query: Tensor,
    qformer: GraniteQFormer,
    linear: Linear,
    window_size: usize,
    num_queries: usize,
}

impl GraniteSpeechProjector {
    fn load(config: &GraniteSpeechConfig, vb: VarBuilder) -> Result<Self> {
        let window_size = config.window_size.max(1);
        let downsample = config.downsample_rate.max(1);
        let num_queries = window_size / downsample;
        let query = vb.get(
            (1, num_queries, config.projector_config.hidden_size),
            "query",
        )?;
        Ok(Self {
            query,
            qformer: GraniteQFormer::load(config, vb.pp("qformer"))?,
            linear: linear(
                config.projector_config.hidden_size,
                config.text_config.hidden_size,
                vb.pp("linear"),
            )?,
            window_size,
            num_queries,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, dim) = hidden.dims3()?;
        let nblocks = seq_len.saturating_add(self.window_size - 1) / self.window_size;
        let pad = nblocks * self.window_size - seq_len;
        let hidden = if pad > 0 {
            hidden.pad_with_zeros(1, 0, pad)?
        } else {
            hidden.clone()
        };
        let windows = hidden.reshape((batch * nblocks, self.window_size, dim))?;
        let query = self
            .query
            .broadcast_as((batch * nblocks, self.num_queries, self.query.dim(2)?))?;
        let output = self.qformer.forward(&query, &windows)?;
        let output = output.reshape((batch, nblocks * self.num_queries, output.dim(2)?))?;
        self.linear.forward(&output).map_err(Error::from)
    }
}

struct GraniteQFormer {
    layernorm: LayerNorm,
    layers: Vec<GraniteQFormerLayer>,
}

impl GraniteQFormer {
    fn load(config: &GraniteSpeechConfig, vb: VarBuilder) -> Result<Self> {
        let hidden = config.projector_config.hidden_size;
        let mut layers = Vec::with_capacity(config.projector_config.num_hidden_layers);
        for idx in 0..config.projector_config.num_hidden_layers {
            layers.push(GraniteQFormerLayer::load(
                config,
                idx,
                vb.pp(format!("encoder.layer.{idx}")),
            )?);
        }
        Ok(Self {
            layernorm: layer_norm(hidden, config.projector_config.layer_norm_eps, vb.pp("layernorm"))?,
            layers,
        })
    }

    fn forward(&self, query: &Tensor, encoder_hidden: &Tensor) -> Result<Tensor> {
        let mut x = self.layernorm.forward(query)?;
        for layer in &self.layers {
            x = layer.forward(&x, encoder_hidden)?;
        }
        Ok(x)
    }
}

struct GraniteQFormerLayer {
    attention: GraniteQFormerAttention,
    crossattention: Option<GraniteQFormerAttention>,
    intermediate_query: Linear,
    output_query_dense: Linear,
    output_query_norm: LayerNorm,
}

impl GraniteQFormerLayer {
    fn load(config: &GraniteSpeechConfig, idx: usize, vb: VarBuilder) -> Result<Self> {
        let hidden = config.projector_config.hidden_size;
        let intermediate = config.projector_config.intermediate_size;
        let has_cross = idx % config.projector_config.cross_attention_frequency.max(1) == 0;
        Ok(Self {
            attention: GraniteQFormerAttention::load(config, false, vb.pp("attention"))?,
            crossattention: if has_cross {
                Some(GraniteQFormerAttention::load(
                    config,
                    true,
                    vb.pp("crossattention"),
                )?)
            } else {
                None
            },
            intermediate_query: linear(
                hidden,
                intermediate,
                vb.pp("intermediate_query.dense"),
            )?,
            output_query_dense: linear(
                intermediate,
                hidden,
                vb.pp("output_query.dense"),
            )?,
            output_query_norm: layer_norm(
                hidden,
                config.projector_config.layer_norm_eps,
                vb.pp("output_query.LayerNorm"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor, encoder_hidden: &Tensor) -> Result<Tensor> {
        let mut out = self.attention.forward(x, None)?;
        if let Some(crossattention) = &self.crossattention {
            out = crossattention.forward(&out, Some(encoder_hidden))?;
        }
        let ff = self.intermediate_query.forward(&out)?.gelu_erf()?;
        let ff = self.output_query_dense.forward(&ff)?;
        self.output_query_norm
            .forward(&ff.broadcast_add(&out)?)
            .map_err(Error::from)
    }
}

struct GraniteQFormerAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output_dense: Linear,
    output_norm: LayerNorm,
    num_heads: usize,
    head_dim: usize,
}

impl GraniteQFormerAttention {
    fn load(config: &GraniteSpeechConfig, cross: bool, vb: VarBuilder) -> Result<Self> {
        let hidden = config.projector_config.hidden_size;
        let kv_hidden = if cross {
            config.projector_config.encoder_hidden_size
        } else {
            hidden
        };
        let all_heads = config.projector_config.num_attention_heads
            * (hidden / config.projector_config.num_attention_heads);
        Ok(Self {
            query: linear(hidden, all_heads, vb.pp("attention.query"))?,
            key: linear(kv_hidden, all_heads, vb.pp("attention.key"))?,
            value: linear(kv_hidden, all_heads, vb.pp("attention.value"))?,
            output_dense: linear(all_heads, hidden, vb.pp("output.dense"))?,
            output_norm: layer_norm(
                hidden,
                config.projector_config.layer_norm_eps,
                vb.pp("output.LayerNorm"),
            )?,
            num_heads: config.projector_config.num_attention_heads,
            head_dim: hidden / config.projector_config.num_attention_heads,
        })
    }

    fn forward(&self, x: &Tensor, encoder_hidden: Option<&Tensor>) -> Result<Tensor> {
        let kv_input = encoder_hidden.unwrap_or(x);
        let q = self.transpose_for_scores(&self.query.forward(x)?)?;
        let k = self.transpose_for_scores(&self.key.forward(kv_input)?)?;
        let v = self.transpose_for_scores(&self.value.forward(kv_input)?)?;
        let mut scores = q.matmul(&k.t()?)?;
        scores = (scores / (self.head_dim as f64).sqrt())?;
        let probs = ops::softmax(&scores.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(q.dtype())?;
        let context = probs
            .matmul(&v)?
            .transpose(1, 2)?
            .flatten_from(D::Minus2)?;
        let out = self.output_dense.forward(&context)?;
        self.output_norm
            .forward(&out.broadcast_add(x)?)
            .map_err(Error::from)
    }

    fn transpose_for_scores(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        x.reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)
            .map_err(Error::from)
    }
}

struct GraniteLanguageModel {
    embed_tokens: Embedding,
    layers: Vec<GraniteDecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    cfg: GraniteTextConfig,
    device: Device,
}

impl GraniteLanguageModel {
    fn load(config: &GraniteTextConfig, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("model.embed_tokens"),
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for idx in 0..config.num_hidden_layers {
            layers.push(GraniteDecoderLayer::load(
                config,
                vb.pp(format!("model.layers.{idx}")),
            )?);
        }
        let norm = candle_nn::rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("model.norm"),
        )?;
        let lm_head = if config.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
        };
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            cfg: config.clone(),
            device: vb.device().clone(),
        })
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn embeddings(&self, input_ids: &[u32]) -> Result<Tensor> {
        let ids = input_ids.iter().map(|id| *id as i64).collect::<Vec<_>>();
        let ids = Tensor::from_vec(ids, (1, input_ids.len()), &self.device)?;
        self.embed_tokens.forward(&ids).map_err(Error::from)
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        start_pos: usize,
        cache: Option<&mut Qwen3Cache>,
    ) -> Result<Tensor> {
        let embeds = self.embed_tokens.forward(input_ids)?;
        self.forward_with_embeds(&embeds, start_pos, cache)
    }

    fn forward_prompt_with_audio(
        &self,
        input_ids: &[u32],
        audio_start: usize,
        audio_len: usize,
        audio_embeds: &Tensor,
        cache: &mut Qwen3Cache,
    ) -> Result<Tensor> {
        let mut llm_ids = input_ids.to_vec();
        for idx in audio_start..audio_start + audio_len {
            if idx < llm_ids.len() {
                llm_ids[idx] = 0;
            }
        }
        let embeds = self.embeddings(&llm_ids)?;
        let seq_len = embeds.dim(1)?;
        let hidden = embeds.dim(2)?;
        let before = if audio_start > 0 {
            embeds.narrow(1, 0, audio_start)?
        } else {
            Tensor::zeros((1, 0, hidden), embeds.dtype(), embeds.device())?
        };
        let after_start = audio_start + audio_len;
        let after = if after_start < seq_len {
            embeds.narrow(1, after_start, seq_len - after_start)?
        } else {
            Tensor::zeros((1, 0, hidden), embeds.dtype(), embeds.device())?
        };
        let audio = audio_embeds.to_dtype(embeds.dtype())?;
        let merged = Tensor::cat(&[before, audio, after], 1)?;
        self.forward_with_embeds(&merged, 0, Some(cache))
    }

    fn forward_with_embeds(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        mut cache: Option<&mut Qwen3Cache>,
    ) -> Result<Tensor> {
        let mut x = (embeds * self.cfg.embedding_multiplier as f64)?;
        for (idx, layer) in self.layers.iter().enumerate() {
            let cache_ref = cache.as_deref_mut();
            x = layer.forward(&x, start_pos, cache_ref, idx)?;
        }
        let hidden = self.norm.forward(&x)?;
        let logits = self.lm_head.forward(&hidden)?;
        (logits / self.cfg.logits_scaling as f64)
            .and_then(|tensor| tensor.to_dtype(DType::F32))
            .map_err(Error::from)
    }
}

struct GraniteDecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: GraniteTextAttention,
    post_attention_layernorm: RmsNorm,
    mlp: GraniteTextMlp,
    residual_multiplier: f32,
}

impl GraniteDecoderLayer {
    fn load(config: &GraniteTextConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            input_layernorm: candle_nn::rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            self_attn: GraniteTextAttention::load(config, vb.pp("self_attn"))?,
            post_attention_layernorm: candle_nn::rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: GraniteTextMlp::load(config, vb.pp("mlp"))?,
            residual_multiplier: config.residual_multiplier,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        cache: Option<&mut Qwen3Cache>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = x;
        let normed = self.input_layernorm.forward(x)?;
        let attn = self.self_attn.forward(&normed, start_pos, cache, layer_idx)?;
        let x = residual.broadcast_add(&(attn * self.residual_multiplier as f64)?)?;
        let residual = &x;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp = self.mlp.forward(&normed)?;
        residual
            .broadcast_add(&(mlp * self.residual_multiplier as f64)?)
            .map_err(Error::from)
    }
}

struct GraniteTextAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f64,
    attention_multiplier: f32,
}

impl GraniteTextAttention {
    fn load(config: &GraniteTextConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        Ok(Self {
            q_proj: linear_no_bias(
                config.hidden_size,
                config.num_attention_heads * head_dim,
                vb.pp("q_proj"),
            )?,
            k_proj: linear_no_bias(
                config.hidden_size,
                config.num_key_value_heads * head_dim,
                vb.pp("k_proj"),
            )?,
            v_proj: linear_no_bias(
                config.hidden_size,
                config.num_key_value_heads * head_dim,
                vb.pp("v_proj"),
            )?,
            o_proj: linear_no_bias(
                config.num_attention_heads * head_dim,
                config.hidden_size,
                vb.pp("o_proj"),
            )?,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
            rope_theta: config.rope_theta,
            attention_multiplier: config.attention_multiplier,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        cache: Option<&mut Qwen3Cache>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        let mut q = self
            .q_proj
            .forward(x)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?;
        let mut k = self
            .k_proj
            .forward(x)?
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;
        let (cos, sin) = build_rope_cache(
            seq_len,
            self.head_dim,
            start_pos,
            self.rope_theta,
            x.device(),
            q.dtype(),
        )?;
        q = apply_rotary_emb(&q, &cos, &sin)?;
        k = apply_rotary_emb(&k, &cos, &sin)?;
        let (k, v, total_len) = if let Some(cache) = cache {
            cache.append(layer_idx, k.clone(), v.clone())?;
            let (cached_k, cached_v) = cache.materialize(layer_idx)?;
            let total_len = cached_k.dim(1)?;
            (cached_k, cached_v, total_len)
        } else {
            let total_len = k.dim(1)?;
            (k, v, total_len)
        };

        let out = if seq_len == 1 {
            dense_decode_attention_scaled(
                &q,
                &k,
                &v,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.attention_multiplier,
            )?
        } else {
            text_attention(
                &q,
                &k,
                &v,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                start_pos,
                total_len,
                self.attention_multiplier,
            )?
        };
        let out = out.reshape((batch, seq_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out).map_err(Error::from)
    }
}

fn build_rope_cache(
    seq_len: usize,
    head_dim: usize,
    start_pos: usize,
    rope_theta: f64,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let half_dim = head_dim / 2;
    let mut cos = Vec::with_capacity(seq_len * half_dim);
    let mut sin = Vec::with_capacity(seq_len * half_dim);
    for pos in start_pos..start_pos + seq_len {
        for idx in 0..half_dim {
            let power = (2.0 * idx as f64) / head_dim as f64;
            let inv = 1.0 / rope_theta.powf(power);
            let angle = pos as f64 * inv;
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    Ok((
        Tensor::from_vec(cos, (seq_len, half_dim), device)?.to_dtype(dtype)?,
        Tensor::from_vec(sin, (seq_len, half_dim), device)?.to_dtype(dtype)?,
    ))
}

fn apply_rotary_emb(x: &Tensor, cos_half: &Tensor, sin_half: &Tensor) -> Result<Tensor> {
    let half_dim = x.dim(3)? / 2;
    let x1 = x.narrow(3, 0, half_dim)?;
    let x2 = x.narrow(3, half_dim, half_dim)?;
    let cos = cos_half.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin_half.unsqueeze(0)?.unsqueeze(2)?;
    let first = x1
        .broadcast_mul(&cos)?
        .broadcast_sub(&x2.broadcast_mul(&sin)?)?;
    let second = x1
        .broadcast_mul(&sin)?
        .broadcast_add(&x2.broadcast_mul(&cos)?)?;
    Tensor::cat(&[first, second], 3).map_err(Error::from)
}

fn text_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    start_pos: usize,
    total_len: usize,
    attention_multiplier: f32,
) -> Result<Tensor> {
    let q_heads = q.transpose(1, 2)?.contiguous()?;
    let k = repeat_kv(k, num_heads, num_kv_heads)?;
    let v = repeat_kv(v, num_heads, num_kv_heads)?;
    let k_heads = k.transpose(1, 2)?.contiguous()?;
    let v_heads = v.transpose(1, 2)?.contiguous()?;
    let mut attn = q_heads.matmul(&k_heads.t()?)?;
    attn = (attn * attention_multiplier as f64)?;
    let mask = causal_mask(q.dim(1)?, total_len, start_pos, q.device(), attn.dtype())?;
    attn = attn.broadcast_add(&mask)?;
    let attn = ops::softmax(&attn.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(q.dtype())?;
    attn.matmul(&v_heads)?
        .transpose(1, 2)
        .map_err(Error::from)
        .and_then(|out| out.reshape((q.dim(0)?, q.dim(1)?, num_heads, head_dim)).map_err(Error::from))
}

fn dense_decode_attention_scaled(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attention_multiplier: f32,
) -> Result<Tensor> {
    let k = repeat_kv(k, num_heads, num_kv_heads)?;
    let v = repeat_kv(v, num_heads, num_kv_heads)?;
    let q_heads = q.transpose(1, 2)?.contiguous()?;
    let k_heads = k.transpose(1, 2)?.contiguous()?;
    let v_heads = v.transpose(1, 2)?.contiguous()?;
    let mut attn = q_heads.matmul(&k_heads.t()?)?;
    attn = (attn * attention_multiplier as f64)?;
    let attn = ops::softmax(&attn.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(q.dtype())?;
    let out = attn.matmul(&v_heads)?;
    out.transpose(1, 2)?
        .reshape((q.dim(0)?, q.dim(1)?, num_heads, head_dim))
        .map_err(Error::from)
}

struct GraniteTextMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl GraniteTextMlp {
    fn load(config: &GraniteTextConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(config.hidden_size, config.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(config.hidden_size, config.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(config.intermediate_size, config.hidden_size, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        self.down_proj
            .forward(&gate.broadcast_mul(&up)?)
            .map_err(Error::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_single_audio_placeholder_to_projected_token_count() {
        let input = vec![10, 20, 30];
        let expanded = expand_audio_tokens(&input, &[1], 3, 99).unwrap();
        assert_eq!(expanded, vec![10, 99, 99, 99, 30]);
    }

    #[test]
    fn rejects_multiple_audio_placeholders_for_single_audio_input() {
        let err = expand_audio_tokens(&[1, 2], &[0, 1], 3, 99).unwrap_err();
        assert!(format!("{err}").contains("exactly one audio placeholder"));
    }

    #[test]
    fn relative_position_score_row_handles_head_batched_query_rows() {
        let device = Device::Cpu;
        let q_row = Tensor::zeros((8, 1, 128), DType::F32, &device).unwrap();
        let rel_row = Tensor::zeros((200, 128), DType::F32, &device).unwrap();

        let scores = relative_position_score_row(&q_row, &rel_row).unwrap();

        assert_eq!(scores.dims(), &[8, 1, 200]);
    }

    #[test]
    fn stop_token_set_deduplicates_eos_pad_and_extra_ids() {
        let tokens = GraniteSpeechSpecialTokens {
            bos_token_id: 1,
            eos_token_id: 2,
            pad_token_id: 2,
            audio_token_id: 3,
            audio_token: "<|audio|>".to_string(),
        };
        assert_eq!(stop_token_set(&tokens, &[4, 2, 4]), vec![2, 4]);
    }

    #[test]
    fn stop_sequence_truncates_generated_text() {
        let mut text = "hello<stop>ignored".to_string();
        assert!(truncate_at_stop_sequence(&mut text, &["<stop>".to_string()]));
        assert_eq!(text, "hello");
    }
}
