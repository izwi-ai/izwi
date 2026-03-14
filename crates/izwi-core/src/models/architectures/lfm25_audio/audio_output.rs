use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, Module};
use candle_transformers::models::with_tracing::QMatMul;
use candle_transformers::quantized_nn::RmsNorm;

use crate::error::{Error, Result};
use crate::models::architectures::qwen3::core::repeat_kv;
use crate::models::shared::weights::gguf::GgufLoader;

use super::config::Lfm25AudioDecoderConfig;
use super::sampling::{sample_from_logits, Lfm25SamplingConfig, SimpleRng};

const DEPTHFORMER_HEADS: usize = 32;
const DEPTHFORMER_KV_HEADS: usize = 8;
const DEPTHFORMER_NORM_EPS: f64 = 1e-5;
const DEPTHFORMER_ROPE_BASE: f32 = 10_000.0;

pub struct Lfm25AudioHead {
    frame_embedding: Embedding,
    codebook_offsets: Vec<u32>,
    depth_linear: QLinear,
    depthformer: Depthformer,
    depth_embeddings: Vec<CodebookEmbeddingHead>,
    codebooks: usize,
    depthformer_dim: usize,
    audio_end_token_id: u32,
}

impl Lfm25AudioHead {
    pub fn load(
        loader: &GgufLoader,
        cfg: &Lfm25AudioDecoderConfig,
        hidden_size: usize,
        device: &Device,
    ) -> Result<Self> {
        let frame_embedding = load_embedding_any(
            loader,
            device,
            &[
                "audio_embedding.embedding.weight".to_string(),
                "audio_head.audio_embedding.embedding.weight".to_string(),
            ],
        )?;
        let depth_linear = QLinear::load(
            loader,
            device,
            &["depth_linear.weight".to_string()],
            &["depth_linear.bias".to_string()],
        )?;
        let depthformer = Depthformer::load(loader, cfg, device)?;

        let mut depth_embeddings = Vec::with_capacity(cfg.codebooks);
        for idx in 0..cfg.codebooks {
            depth_embeddings.push(CodebookEmbeddingHead::load(loader, device, idx)?);
        }

        let codebook_offsets = (0..cfg.codebooks)
            .map(|idx| {
                u32::try_from(idx * cfg.audio_vocab_size).map_err(|_| {
                    Error::ModelLoadError(format!(
                        "Depthformer codebook offset out of range for codebook {idx}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        if hidden_size == 0 || cfg.depthformer_dim == 0 {
            return Err(Error::ModelLoadError(
                "Invalid LFM2.5 Audio decoder dimensions".to_string(),
            ));
        }

        Ok(Self {
            frame_embedding,
            codebook_offsets,
            depth_linear,
            depthformer,
            depth_embeddings,
            codebooks: cfg.codebooks,
            depthformer_dim: cfg.depthformer_dim,
            audio_end_token_id: cfg.audio_end_token_id,
        })
    }

    pub fn audio_end_token_id(&self) -> u32 {
        self.audio_end_token_id
    }

    pub fn embed_audio_frame(&self, frame_tokens: &[u32], device: &Device) -> Result<Tensor> {
        if frame_tokens.len() != self.codebooks {
            return Err(Error::InvalidInput(format!(
                "Expected {} audio codebooks, received {}",
                self.codebooks,
                frame_tokens.len()
            )));
        }

        let mut offset_tokens = Vec::with_capacity(frame_tokens.len());
        for (idx, token) in frame_tokens.iter().copied().enumerate() {
            offset_tokens.push(token.saturating_add(self.codebook_offsets[idx]));
        }

        let token_ids = Tensor::from_vec(offset_tokens, (1, self.codebooks), device)?;
        let embeds = self.frame_embedding.forward(&token_ids)?;
        let embeds = embeds.sum(1)?;
        embeds.unsqueeze(1).map_err(Error::from)
    }

    pub fn sample_audio_frame(
        &self,
        hidden: &Tensor,
        config: &Lfm25SamplingConfig,
        rng: &mut SimpleRng,
    ) -> Result<Vec<u32>> {
        let hidden = ensure_rank3(hidden)?;
        let depth_input = self.depth_linear.forward(&hidden)?;
        let depth_input = depth_input
            .reshape((1, 1, self.codebooks, self.depthformer_dim))?
            .squeeze(0)?
            .squeeze(0)?; // [C, D]

        let mut next_embed = Tensor::zeros(
            self.depthformer_dim,
            hidden.dtype(),
            hidden.device(),
        )?;
        let mut caches = self.depthformer.empty_cache();
        let mut tokens = Vec::with_capacity(self.codebooks);

        for codebook_idx in 0..self.codebooks {
            let cur = depth_input.i(codebook_idx)?.broadcast_add(&next_embed)?;
            let cur = cur.unsqueeze(0)?.unsqueeze(0)?;
            let out = self.depthformer.forward_cached(&cur, &mut caches)?;
            let token = self.depth_embeddings[codebook_idx].sample(&out, config, rng)?;
            next_embed = self.depth_embeddings[codebook_idx].embed(token, hidden.device())?;
            tokens.push(token);
        }

        Ok(tokens)
    }
}

struct Depthformer {
    layers: Vec<DepthformerBlock>,
}

impl Depthformer {
    fn load(loader: &GgufLoader, cfg: &Lfm25AudioDecoderConfig, device: &Device) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.depthformer_layers);
        for idx in 0..cfg.depthformer_layers {
            layers.push(DepthformerBlock::load(
                loader,
                device,
                idx,
                cfg.depthformer_dim,
            )?);
        }
        Ok(Self { layers })
    }

    fn empty_cache(&self) -> Vec<Option<(Tensor, Tensor)>> {
        vec![None; self.layers.len()]
    }

    fn forward_cached(
        &self,
        x: &Tensor,
        caches: &mut [Option<(Tensor, Tensor)>],
    ) -> Result<Tensor> {
        let mut hidden = x.clone();
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            hidden = layer.forward_cached(&hidden, cache)?;
        }
        Ok(hidden)
    }
}

struct DepthformerBlock {
    attn_norm: RmsNorm,
    attn: DepthAttention,
    ffn_norm: RmsNorm,
    mlp: DepthMlp,
}

impl DepthformerBlock {
    fn load(loader: &GgufLoader, device: &Device, idx: usize, dim: usize) -> Result<Self> {
        let attn_norm = load_rms_norm_any(
            loader,
            device,
            &[
                format!("audio_head.depthformer.blocks.{idx}.attn_norm.weight"),
                format!("depthformer.blocks.{idx}.attn_norm.weight"),
            ],
        )?;
        let ffn_norm = load_rms_norm_any(
            loader,
            device,
            &[
                format!("audio_head.depthformer.blocks.{idx}.ffn_norm.weight"),
                format!("depthformer.blocks.{idx}.ffn_norm.weight"),
            ],
        )?;
        let attn = DepthAttention::load(loader, device, idx, dim)?;
        let mlp = DepthMlp::load(loader, device, idx)?;
        Ok(Self {
            attn_norm,
            attn,
            ffn_norm,
            mlp,
        })
    }

    fn forward_cached(
        &self,
        x: &Tensor,
        cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let hidden = self.attn.forward(&self.attn_norm.forward(x)?, cache)?;
        let hidden = hidden.broadcast_add(x)?;
        let ffn = self.mlp.forward(&self.ffn_norm.forward(&hidden)?)?;
        hidden.broadcast_add(&ffn).map_err(Error::from)
    }
}

struct DepthAttention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    cos: Tensor,
    sin: Tensor,
}

impl DepthAttention {
    fn load(loader: &GgufLoader, device: &Device, idx: usize, dim: usize) -> Result<Self> {
        let q_proj = QLinear::load(
            loader,
            device,
            &[format!(
                "audio_head.depthformer.blocks.{idx}.attn.q_proj.weight"
            )],
            &[],
        )?;
        let k_proj = QLinear::load(
            loader,
            device,
            &[format!(
                "audio_head.depthformer.blocks.{idx}.attn.k_proj.weight"
            )],
            &[],
        )?;
        let v_proj = QLinear::load(
            loader,
            device,
            &[format!(
                "audio_head.depthformer.blocks.{idx}.attn.v_proj.weight"
            )],
            &[],
        )?;
        let o_proj = QLinear::load(
            loader,
            device,
            &[format!(
                "audio_head.depthformer.blocks.{idx}.attn.o_proj.weight"
            )],
            &[],
        )?;
        let q_norm = load_rms_norm_any(
            loader,
            device,
            &[format!(
                "audio_head.depthformer.blocks.{idx}.attn.q_norm.weight"
            )],
        )?;
        let k_norm = load_rms_norm_any(
            loader,
            device,
            &[format!(
                "audio_head.depthformer.blocks.{idx}.attn.k_norm.weight"
            )],
        )?;
        let (cos, sin) = precompute_freqs(
            dim / DEPTHFORMER_HEADS,
            DEPTHFORMER_ROPE_BASE,
            32,
            device,
        )?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            cos,
            sin,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (batch, seq_len, hidden_size) = hidden.dims3()?;

        let q = self
            .q_proj
            .forward(hidden)?
            .reshape((batch, seq_len, DEPTHFORMER_HEADS, hidden_size / DEPTHFORMER_HEADS))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(hidden)?
            .reshape((batch, seq_len, DEPTHFORMER_KV_HEADS, hidden_size / DEPTHFORMER_HEADS))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(hidden)?
            .reshape((batch, seq_len, DEPTHFORMER_KV_HEADS, hidden_size / DEPTHFORMER_HEADS))?
            .transpose(1, 2)?
            .contiguous()?;

        let index_pos = cache
            .as_ref()
            .map(|(keys, _values)| keys.dim(2).unwrap_or(0))
            .unwrap_or(0);
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let q = apply_rotary_emb(&q, &self.cos, &self.sin, index_pos)?;
        let k = apply_rotary_emb(&k, &self.cos, &self.sin, index_pos)?;

        let (all_k, all_v) = if let Some((old_k, old_v)) = cache.as_ref() {
            (
                Tensor::cat(&[old_k, &k], 2)?,
                Tensor::cat(&[old_v, &v], 2)?,
            )
        } else {
            (k, v)
        };
        *cache = Some((all_k.clone(), all_v.clone()));

        let key = repeat_kv(&all_k, DEPTHFORMER_HEADS, DEPTHFORMER_KV_HEADS)?;
        let value = repeat_kv(&all_v, DEPTHFORMER_HEADS, DEPTHFORMER_KV_HEADS)?;
        let scores = (q.matmul(&key.transpose(2, 3)?.contiguous()?)?
            / ((hidden_size / DEPTHFORMER_HEADS) as f64).sqrt())?;
        let scores = causal_mask(scores, index_pos)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = probs.matmul(&value)?;
        let out = out
            .transpose(1, 2)?
            .reshape((batch, seq_len, hidden_size))?;
        self.o_proj.forward(&out)
    }
}

struct DepthMlp {
    w1: QLinear,
    w2: QLinear,
    w3: QLinear,
}

impl DepthMlp {
    fn load(loader: &GgufLoader, device: &Device, idx: usize) -> Result<Self> {
        Ok(Self {
            w1: QLinear::load(
                loader,
                device,
                &[format!("audio_head.depthformer.blocks.{idx}.ffn.w1.weight")],
                &[],
            )?,
            w2: QLinear::load(
                loader,
                device,
                &[format!("audio_head.depthformer.blocks.{idx}.ffn.w2.weight")],
                &[],
            )?,
            w3: QLinear::load(
                loader,
                device,
                &[format!("audio_head.depthformer.blocks.{idx}.ffn.w3.weight")],
                &[],
            )?,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.w1.forward(hidden)?)?;
        let up = self.w3.forward(hidden)?;
        self.w2.forward(&(&gate * &up)?)
    }
}

struct CodebookEmbeddingHead {
    embedding: Embedding,
    norm: RmsNorm,
    to_logits: QLinear,
}

impl CodebookEmbeddingHead {
    fn load(loader: &GgufLoader, device: &Device, idx: usize) -> Result<Self> {
        let embedding = load_embedding_any(
            loader,
            device,
            &[format!("depth_embeddings.{idx}.embedding.weight")],
        )?;
        let norm = load_rms_norm_any(
            loader,
            device,
            &[format!("depth_embeddings.{idx}.embedding_norm.weight")],
        )?;
        let to_logits = QLinear::load(
            loader,
            device,
            &[format!("depth_embeddings.{idx}.to_logits.weight")],
            &[],
        )?;
        Ok(Self {
            embedding,
            norm,
            to_logits,
        })
    }

    fn embed(&self, token: u32, device: &Device) -> Result<Tensor> {
        let token = Tensor::from_vec(vec![token], (1,), device)?;
        let embed = self.embedding.forward(&token)?;
        embed.squeeze(0).map_err(Error::from)
    }

    fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let hidden = ensure_rank3(hidden)?;
        let hidden = self.norm.forward(&hidden)?;
        let logits = self.to_logits.forward(&hidden)?;
        logits.squeeze(0)?.squeeze(0).map_err(Error::from)
    }

    fn sample(
        &self,
        hidden: &Tensor,
        config: &Lfm25SamplingConfig,
        rng: &mut SimpleRng,
    ) -> Result<u32> {
        let logits = self.logits(hidden)?;
        sample_from_logits(&logits, logits.dim(0)?, config, rng)
    }
}

#[derive(Debug)]
struct QLinear {
    weight: QMatMul,
    bias: Option<Tensor>,
}

impl QLinear {
    fn load(
        loader: &GgufLoader,
        device: &Device,
        weight_names: &[String],
        bias_names: &[String],
    ) -> Result<Self> {
        let weight = Arc::new(load_qtensor_any(loader, device, weight_names)?);
        let weight = QMatMul::from_weights(weight).map_err(Error::from)?;
        let bias = load_optional_tensor_any(loader, device, bias_names, DType::F32)?;
        Ok(Self { weight, bias })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let out = self.weight.forward(hidden)?;
        if let Some(bias) = &self.bias {
            out.broadcast_add(bias).map_err(Error::from)
        } else {
            Ok(out)
        }
    }
}

fn load_embedding_any(loader: &GgufLoader, device: &Device, names: &[String]) -> Result<Embedding> {
    let weight = load_qtensor_any(loader, device, names)?
        .dequantize(device)
        .map_err(Error::from)?;
    let (_, dim) = weight.dims2()?;
    Ok(Embedding::new(weight, dim))
}

fn load_rms_norm_any(loader: &GgufLoader, device: &Device, names: &[String]) -> Result<RmsNorm> {
    RmsNorm::from_qtensor(load_qtensor_any(loader, device, names)?, DEPTHFORMER_NORM_EPS)
        .map_err(Error::from)
}

fn load_qtensor_any(
    loader: &GgufLoader,
    device: &Device,
    names: &[String],
) -> Result<candle_core::quantized::QTensor> {
    for name in names {
        if loader.has_tensor(name) {
            return loader.load_qtensor(name, device);
        }
    }
    Err(Error::ModelLoadError(format!(
        "Missing GGUF tensor; tried {}",
        names.join(" | ")
    )))
}

fn load_optional_tensor_any(
    loader: &GgufLoader,
    device: &Device,
    names: &[String],
    dtype: DType,
) -> Result<Option<Tensor>> {
    for name in names {
        if loader.has_tensor(name) {
            return loader.load_tensor(name, dtype, device).map(Some);
        }
    }
    Ok(None)
}

fn precompute_freqs(
    head_dim: usize,
    freq_base: f32,
    context_length: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|idx| 1.0f32 / freq_base.powf(idx as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let angles = Tensor::arange(0u32, context_length as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((context_length, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    Ok((angles.cos()?, angles.sin()?))
}

fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor, index_pos: usize) -> Result<Tensor> {
    let (_, _, seq_len, _) = x.dims4()?;
    let cos = cos.narrow(0, index_pos, seq_len)?;
    let sin = sin.narrow(0, index_pos, seq_len)?;
    candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin).map_err(Error::from)
}

fn causal_mask(scores: Tensor, index_pos: usize) -> Result<Tensor> {
    let (_batch, _heads, query_len, key_len) = scores.dims4()?;
    if query_len == 1 {
        return Ok(scores);
    }

    let mask: Vec<u8> = (0..query_len)
        .flat_map(|i| {
            let global_i = index_pos + i;
            (0..key_len).map(move |j| u8::from(j > global_i))
        })
        .collect();
    let mask = Tensor::from_slice(&mask, (1, 1, query_len, key_len), scores.device())?;
    let neg_inf = Tensor::new(f32::NEG_INFINITY, scores.device())?;
    let masked = mask.where_cond(&neg_inf.broadcast_as(scores.shape().dims())?, &scores)?;
    Ok(masked)
}

fn ensure_rank3(hidden: &Tensor) -> Result<Tensor> {
    match hidden.rank() {
        3 => Ok(hidden.clone()),
        2 => hidden.unsqueeze(0).map_err(Error::from),
        1 => hidden.unsqueeze(0)?.unsqueeze(0).map_err(Error::from),
        rank => Err(Error::InferenceError(format!(
            "Expected 1D/2D/3D hidden state, got rank {rank}"
        ))),
    }
}
