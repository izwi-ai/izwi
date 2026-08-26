use std::sync::Arc;
use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, Module};
use candle_transformers::models::with_tracing::QMatMul;

use crate::backends::kv::{
    submit_ordered_after_write, KvSlotMap, KvWriteArgs, KvWriteCompletionCollector,
    PagedKvDecodeArgs,
};
use crate::error::{Error, Result};
use crate::kernels::{
    try_fused_qk_rms_norm, try_fused_rms_norm, try_fused_rope_pair_bshd,
    try_fused_silu_mul_with_status,
};
use crate::kv::KvDecodeBatchMetadata;
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PreparedPhysicalPagedStep};
use crate::models::shared::telemetry::record_rope_kernel;
use crate::models::shared::weights::gguf::GgufLoader;

use super::config::Lfm25AudioDecoderConfig;
use super::sampling::{
    greedy_token_tensor_from_logits, sample_from_logits, Lfm25SamplingConfig, SimpleRng,
};

use super::config::{
    LFM25_DEPTHFORMER_KV_HEADS as DEPTHFORMER_KV_HEADS,
    LFM25_DEPTHFORMER_QUERY_HEADS as DEPTHFORMER_HEADS,
};
const DEPTHFORMER_NORM_EPS: f64 = 1e-5;
const DEPTHFORMER_ROPE_BASE: f32 = 10_000.0;

pub struct Lfm25AudioHead {
    frame_embedding: Embedding,
    codebook_offsets: Vec<u32>,
    codebook_offsets_tensor: Tensor,
    depth_linear: QLinear,
    depthformer: Depthformer,
    depth_embeddings: Vec<CodebookEmbeddingHead>,
    codebooks: usize,
    depthformer_dim: usize,
    audio_end_token_id: u32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Lfm25AudioHeadProfile {
    pub depth_linear_ms: f64,
    pub depth_reshape_ms: f64,
    pub cache_setup_ms: f64,
    pub codebook_input_ms: f64,
    pub depthformer_ms: f64,
    pub sample_ms: f64,
    pub embed_ms: f64,
    pub materialize_ms: f64,
    pub materialize_pack_ms: f64,
    pub materialize_readback_ms: f64,
    pub codebook_steps: u64,
}

pub struct Lfm25SampledAudioFrame {
    samples: Vec<CodebookSample>,
    embedding: Tensor,
}

impl Lfm25SampledAudioFrame {
    pub fn embedding(&self) -> &Tensor {
        &self.embedding
    }

    pub(crate) fn tokens(&self) -> Result<Vec<u32>> {
        self.samples.iter().map(CodebookSample::to_token).collect()
    }
}

#[derive(Debug)]
pub struct Lfm25StackedAudioFrameTokens {
    pub tokens: Tensor,
    pub materialize_ms: f64,
    pub pack_ms: f64,
}

#[derive(Debug)]
pub struct Lfm25AudioFirstTokens {
    pub tokens: Vec<u32>,
    pub materialize_ms: f64,
    pub pack_ms: f64,
    pub readback_ms: f64,
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
        let codebook_offsets_tensor =
            Tensor::from_vec(codebook_offsets.clone(), (cfg.codebooks,), device)?;

        if hidden_size == 0 || cfg.depthformer_dim == 0 {
            return Err(Error::ModelLoadError(
                "Invalid LFM2.5 Audio decoder dimensions".to_string(),
            ));
        }

        Ok(Self {
            frame_embedding,
            codebook_offsets,
            codebook_offsets_tensor,
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
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Vec<u32>> {
        self.sample_audio_frame_with_profile(hidden, config, rng, cache)
            .map(|(samples, _profile)| samples)
    }

    pub fn sample_audio_frame_with_profile(
        &self,
        hidden: &Tensor,
        config: &Lfm25SamplingConfig,
        rng: &mut SimpleRng,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<(Vec<u32>, Lfm25AudioHeadProfile)> {
        let (frame, mut profile) =
            self.sample_audio_frame_embedded_with_profile(hidden, config, rng, cache)?;
        let materialize_started = Instant::now();
        let materialized = materialize_codebook_samples_with_profile(&frame.samples)?;
        profile.materialize_ms = elapsed_ms(materialize_started);
        profile.materialize_pack_ms = materialized.pack_ms;
        profile.materialize_readback_ms = materialized.readback_ms;
        Ok((materialized.samples, profile))
    }

    pub fn sample_audio_frame_embedded_with_profile(
        &self,
        hidden: &Tensor,
        config: &Lfm25SamplingConfig,
        rng: &mut SimpleRng,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<(Lfm25SampledAudioFrame, Lfm25AudioHeadProfile)> {
        let mut profile = Lfm25AudioHeadProfile::default();
        let hidden = ensure_rank3(hidden)?;
        let depth_linear_started = Instant::now();
        let depth_input = self.depth_linear.forward(&hidden)?;
        profile.depth_linear_ms = elapsed_ms(depth_linear_started);

        let depth_reshape_started = Instant::now();
        let depth_input = depth_input
            .reshape((1, 1, self.codebooks, self.depthformer_dim))?
            .squeeze(0)?
            .squeeze(0)?; // [C, D]
        profile.depth_reshape_ms = elapsed_ms(depth_reshape_started);

        let cache_setup_started = Instant::now();
        let mut next_embed = Tensor::zeros(self.depthformer_dim, hidden.dtype(), hidden.device())?;
        cache.reset_invocation()?;
        let mut samples = Vec::with_capacity(self.codebooks);
        profile.cache_setup_ms = elapsed_ms(cache_setup_started);

        for codebook_idx in 0..self.codebooks {
            let codebook_input_started = Instant::now();
            let cur = depth_input.i(codebook_idx)?.broadcast_add(&next_embed)?;
            let cur = cur.unsqueeze(0)?.unsqueeze(0)?;
            profile.codebook_input_ms += elapsed_ms(codebook_input_started);

            let depthformer_started = Instant::now();
            let out = self.depthformer.forward_physical(&cur, cache)?;
            profile.depthformer_ms += elapsed_ms(depthformer_started);

            let sample_started = Instant::now();
            let sample = self.depth_embeddings[codebook_idx].sample(&out, config, rng)?;
            profile.sample_ms += elapsed_ms(sample_started);

            let embed_started = Instant::now();
            next_embed =
                self.depth_embeddings[codebook_idx].embed_sample(&sample, hidden.device())?;
            profile.embed_ms += elapsed_ms(embed_started);
            samples.push(sample);
            profile.codebook_steps = profile.codebook_steps.saturating_add(1);
        }

        let embedding = self.embed_sampled_audio_frame(&samples, hidden.device())?;
        Ok((Lfm25SampledAudioFrame { samples, embedding }, profile))
    }

    pub(crate) fn sample_audio_frame_retained(
        &self,
        hidden: &Tensor,
        config: &Lfm25SamplingConfig,
        rng: &mut SimpleRng,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Lfm25SampledAudioFrame> {
        self.sample_audio_frame_retained_batch(
            hidden,
            std::slice::from_ref(config),
            &mut [rng],
            &mut [cache],
        )?
        .pop()
        .ok_or_else(|| Error::InferenceError("LFM2.5 Audio scalar frame is absent".into()))
    }

    pub(crate) fn sample_audio_frame_retained_batch(
        &self,
        hidden: &Tensor,
        configs: &[Lfm25SamplingConfig],
        rngs: &mut [&mut SimpleRng],
        caches: &mut [&mut PhysicalPagedKvCache],
    ) -> Result<Vec<Lfm25SampledAudioFrame>> {
        let (batch, seq_len, _) = hidden.dims3()?;
        if batch == 0
            || seq_len != 1
            || configs.len() != batch
            || rngs.len() != batch
            || caches.len() != batch
        {
            return Err(Error::InvalidInput(
                "LFM2.5 Audio retained frame rows do not match".into(),
            ));
        }
        if batch == 1 {
            let depth_input = self
                .depth_linear
                .forward(hidden)?
                .reshape((1, 1, self.codebooks, self.depthformer_dim))?
                .squeeze(0)?
                .squeeze(0)?;
            let mut next_embed =
                Tensor::zeros(self.depthformer_dim, hidden.dtype(), hidden.device())?;
            let mut samples = Vec::with_capacity(self.codebooks);
            for codebook_idx in 0..self.codebooks {
                let cur = depth_input
                    .i(codebook_idx)?
                    .broadcast_add(&next_embed)?
                    .unsqueeze(0)?
                    .unsqueeze(0)?;
                let out = self.depthformer.forward_physical(&cur, caches[0])?;
                let sample =
                    self.depth_embeddings[codebook_idx].sample(&out, &configs[0], rngs[0])?;
                next_embed =
                    self.depth_embeddings[codebook_idx].embed_sample(&sample, hidden.device())?;
                samples.push(sample);
            }
            let embedding = self.embed_sampled_audio_frame(&samples, hidden.device())?;
            return Ok(vec![Lfm25SampledAudioFrame { samples, embedding }]);
        }

        let depth_input = self.depth_linear.forward(hidden)?.reshape((
            batch,
            self.codebooks,
            self.depthformer_dim,
        ))?;
        let mut next_embed = Tensor::zeros(
            (batch, self.depthformer_dim),
            hidden.dtype(),
            hidden.device(),
        )?;
        let mut rows = (0..batch)
            .map(|_| Vec::with_capacity(self.codebooks))
            .collect::<Vec<Vec<CodebookSample>>>();
        for codebook_idx in 0..self.codebooks {
            let cur = depth_input
                .narrow(1, codebook_idx, 1)?
                .squeeze(1)?
                .broadcast_add(&next_embed)?
                .unsqueeze(1)?;
            let out = self.depthformer.forward_physical_batch(&cur, caches)?;
            let mut embeds = Vec::with_capacity(batch);
            for row in 0..batch {
                let row_out = out.narrow(0, row, 1)?;
                let sample = self.depth_embeddings[codebook_idx].sample(
                    &row_out,
                    &configs[row],
                    rngs[row],
                )?;
                embeds.push(
                    self.depth_embeddings[codebook_idx].embed_sample(&sample, hidden.device())?,
                );
                rows[row].push(sample);
            }
            next_embed = Tensor::stack(&embeds.iter().collect::<Vec<_>>(), 0)?;
        }
        rows.into_iter()
            .map(|samples| {
                let embedding = self.embed_sampled_audio_frame(&samples, hidden.device())?;
                Ok(Lfm25SampledAudioFrame { samples, embedding })
            })
            .collect()
    }

    pub fn first_tokens_with_profile(
        &self,
        frames: &[Lfm25SampledAudioFrame],
    ) -> Result<Lfm25AudioFirstTokens> {
        let materialize_started = Instant::now();
        if frames.is_empty() {
            return Ok(Lfm25AudioFirstTokens {
                tokens: Vec::new(),
                materialize_ms: elapsed_ms(materialize_started),
                pack_ms: 0.0,
                readback_ms: 0.0,
            });
        }

        let device_tokens = frames
            .iter()
            .map(|frame| frame.samples.first())
            .collect::<Option<Vec<_>>>();
        if let Some(samples) = device_tokens {
            if samples
                .iter()
                .all(|sample| matches!(sample, CodebookSample::DeviceGreedy(_)))
            {
                let tensors = samples
                    .iter()
                    .filter_map(|sample| match sample {
                        CodebookSample::DeviceGreedy(token) => Some(token),
                        CodebookSample::Host(_) => None,
                    })
                    .collect::<Vec<_>>();
                let pack_started = Instant::now();
                let packed = Tensor::cat(&tensors, 0)?;
                let pack_ms = elapsed_ms(pack_started);
                let readback_started = Instant::now();
                let tokens = packed.to_vec1::<u32>().map_err(Error::from)?;
                let readback_ms = elapsed_ms(readback_started);
                return Ok(Lfm25AudioFirstTokens {
                    tokens,
                    materialize_ms: elapsed_ms(materialize_started),
                    pack_ms,
                    readback_ms,
                });
            }
        }

        let readback_started = Instant::now();
        let tokens = frames
            .iter()
            .map(|frame| {
                frame
                    .samples
                    .first()
                    .ok_or_else(|| {
                        Error::InferenceError("Empty LFM2.5 Audio sampled frame".to_string())
                    })?
                    .to_token()
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Lfm25AudioFirstTokens {
            tokens,
            materialize_ms: elapsed_ms(materialize_started),
            pack_ms: 0.0,
            readback_ms: elapsed_ms(readback_started),
        })
    }

    pub fn stack_frame_tokens_with_profile(
        &self,
        frames: &[Lfm25SampledAudioFrame],
        device: &Device,
    ) -> Result<Lfm25StackedAudioFrameTokens> {
        let materialize_started = Instant::now();
        if frames.is_empty() {
            return Ok(Lfm25StackedAudioFrameTokens {
                tokens: Tensor::zeros((0, self.codebooks), DType::U32, device)?,
                materialize_ms: elapsed_ms(materialize_started),
                pack_ms: 0.0,
            });
        }

        if frames
            .iter()
            .flat_map(|frame| frame.samples.iter())
            .all(|sample| matches!(sample, CodebookSample::DeviceGreedy(_)))
        {
            let tensors = frames
                .iter()
                .flat_map(|frame| frame.samples.iter())
                .filter_map(|sample| match sample {
                    CodebookSample::DeviceGreedy(token) => Some(token),
                    CodebookSample::Host(_) => None,
                })
                .collect::<Vec<_>>();
            let pack_started = Instant::now();
            let tokens = Tensor::cat(&tensors, 0)?
                .reshape((frames.len(), self.codebooks))?
                .contiguous()?;
            let pack_ms = elapsed_ms(pack_started);
            return Ok(Lfm25StackedAudioFrameTokens {
                tokens,
                materialize_ms: elapsed_ms(materialize_started),
                pack_ms,
            });
        }

        let pack_started = Instant::now();
        let mut flat = Vec::with_capacity(frames.len() * self.codebooks);
        for frame in frames {
            for sample in &frame.samples {
                flat.push(sample.to_token()?);
            }
        }
        let tokens = Tensor::from_vec(flat, (frames.len(), self.codebooks), device)?;
        let pack_ms = elapsed_ms(pack_started);
        Ok(Lfm25StackedAudioFrameTokens {
            tokens,
            materialize_ms: elapsed_ms(materialize_started),
            pack_ms,
        })
    }

    fn embed_sampled_audio_frame(
        &self,
        samples: &[CodebookSample],
        device: &Device,
    ) -> Result<Tensor> {
        if samples.len() != self.codebooks {
            return Err(Error::InvalidInput(format!(
                "Expected {} audio codebooks, received {}",
                self.codebooks,
                samples.len()
            )));
        }

        if samples
            .iter()
            .all(|sample| matches!(sample, CodebookSample::DeviceGreedy(_)))
        {
            let tensors = samples
                .iter()
                .filter_map(|sample| match sample {
                    CodebookSample::DeviceGreedy(token) => Some(token),
                    CodebookSample::Host(_) => None,
                })
                .collect::<Vec<_>>();
            let token_ids = Tensor::cat(&tensors, 0)?
                .broadcast_add(&self.codebook_offsets_tensor)?
                .reshape((1, self.codebooks))?
                .contiguous()?;
            let embeds = self.frame_embedding.forward(&token_ids)?;
            let embeds = embeds.sum(1)?;
            return embeds.unsqueeze(1).map_err(Error::from);
        }

        let tokens = samples
            .iter()
            .map(CodebookSample::to_token)
            .collect::<Result<Vec<_>>>()?;
        self.embed_audio_frame(&tokens, device)
    }
}

struct Depthformer {
    layers: Vec<DepthformerBlock>,
    dim: usize,
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
        Ok(Self {
            layers,
            dim: cfg.depthformer_dim,
        })
    }

    fn forward_physical(&self, x: &Tensor, cache: &mut PhysicalPagedKvCache) -> Result<Tensor> {
        let (batch, seq_len, hidden) = x.dims3()?;
        if batch != 1
            || hidden != self.dim
            || seq_len != 1
            || cache.context_len() >= cache.capacity_tokens()
        {
            return Err(Error::InvalidInput(
                "Depthformer input does not match its physical invocation state".into(),
            ));
        }
        let head_dim = self.dim / DEPTHFORMER_HEADS;
        cache.validate_model(self.layers.len(), DEPTHFORMER_KV_HEADS, head_dim)?;
        let index_pos = cache.context_len();
        let mut prepared = cache.prepare_append(index_pos, seq_len)?;
        let mut hidden = x.clone();
        for (physical_layer, layer) in self.layers.iter().enumerate() {
            hidden =
                layer.forward_physical(&hidden, index_pos, physical_layer, cache, &mut prepared)?;
        }
        cache.commit_prepared(prepared)?;
        Ok(hidden)
    }

    fn forward_physical_batch(
        &self,
        x: &Tensor,
        caches: &mut [&mut PhysicalPagedKvCache],
    ) -> Result<Tensor> {
        let (batch, seq_len, hidden) = x.dims3()?;
        if batch <= 1 || seq_len != 1 || hidden != self.dim || caches.len() != batch {
            return Err(Error::InvalidInput(
                "Depthformer batch input does not match its physical rows".into(),
            ));
        }
        let head_dim = self.dim / DEPTHFORMER_HEADS;
        for cache in caches.iter() {
            cache.validate_model(self.layers.len(), DEPTHFORMER_KV_HEADS, head_dim)?;
            if cache.context_len() >= cache.capacity_tokens() {
                return Err(Error::InvalidInput(
                    "Depthformer batch row exceeded its invocation capacity".into(),
                ));
            }
        }
        let first = &*caches[0];
        if caches
            .iter()
            .any(|cache| !Arc::ptr_eq(cache.arena(), first.arena()))
        {
            return Err(Error::InvalidInput(
                "Depthformer batch rows do not share one physical arena".into(),
            ));
        }
        let positions = caches
            .iter()
            .map(|cache| cache.context_len())
            .collect::<Vec<_>>();
        let slots = caches
            .iter()
            .enumerate()
            .map(|(row, cache)| {
                cache
                    .slots_for_append(positions[row], 1)
                    .map(|slots| slots[0])
            })
            .collect::<Result<Vec<_>>>()?;
        let lowered = first.arena().lower_slots(&slots)?;
        let mut completions =
            KvWriteCompletionCollector::new(first.arena().config(), lowered.logical_slots())?;
        let execution = (|| -> Result<Tensor> {
            let mut hidden = x.clone();
            for (physical_layer, layer) in self.layers.iter().enumerate() {
                let cache_refs = caches.iter().map(|cache| &**cache).collect::<Vec<_>>();
                hidden = layer.forward_physical_batch(
                    &hidden,
                    &positions,
                    physical_layer,
                    &cache_refs,
                    lowered.as_ref(),
                    &mut completions,
                )?;
            }
            Ok(hidden)
        })();
        let hidden = match execution {
            Ok(hidden) => hidden,
            Err(error) => {
                return match completions.drain() {
                    Ok(()) => Err(error),
                    Err(drain) => Err(Error::InferenceError(format!(
                        "Depthformer batch failed: {error}; fence drain failed: {drain}"
                    ))),
                }
            }
        };
        let completion = Arc::new(completions.seal()?);
        for (row, cache) in caches.iter_mut().enumerate() {
            cache.commit_shared_completion(positions[row], 1, completion.clone())?;
        }
        Ok(hidden)
    }
}

struct DepthformerBlock {
    attn_norm: DepthRmsNorm,
    attn: DepthAttention,
    ffn_norm: DepthRmsNorm,
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
                format!("depthformer.layers.{idx}.operator_norm.weight"),
            ],
        )?;
        let ffn_norm = load_rms_norm_any(
            loader,
            device,
            &[
                format!("audio_head.depthformer.blocks.{idx}.ffn_norm.weight"),
                format!("depthformer.blocks.{idx}.ffn_norm.weight"),
                format!("depthformer.layers.{idx}.ffn_norm.weight"),
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

    fn forward_physical(
        &self,
        x: &Tensor,
        index_pos: usize,
        physical_layer: usize,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
    ) -> Result<Tensor> {
        let hidden = self.attn.forward_physical(
            &self.attn_norm.forward(x)?,
            index_pos,
            physical_layer,
            cache,
            prepared,
        )?;
        let hidden = hidden.broadcast_add(x)?;
        let ffn = self.mlp.forward(&self.ffn_norm.forward(&hidden)?)?;
        hidden.broadcast_add(&ffn).map_err(Error::from)
    }

    fn forward_physical_batch(
        &self,
        x: &Tensor,
        positions: &[usize],
        physical_layer: usize,
        caches: &[&PhysicalPagedKvCache],
        slots: &dyn KvSlotMap,
        completions: &mut KvWriteCompletionCollector,
    ) -> Result<Tensor> {
        let hidden = self.attn.forward_physical_batch(
            &self.attn_norm.forward(x)?,
            positions,
            physical_layer,
            caches,
            slots,
            completions,
        )?;
        let hidden = hidden.broadcast_add(x)?;
        let ffn = self.mlp.forward(&self.ffn_norm.forward(&hidden)?)?;
        hidden.broadcast_add(&ffn).map_err(Error::from)
    }
}

struct DepthAttention {
    qkv: DepthQkvProjection,
    o_proj: QLinear,
    q_norm: DepthRmsNorm,
    k_norm: DepthRmsNorm,
    qk_norm_weight: Tensor,
    cos: Tensor,
    sin: Tensor,
    cos_sin: Tensor,
}

#[derive(Debug, Clone)]
struct DepthRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl DepthRmsNorm {
    fn from_qtensor(weight: candle_core::quantized::QTensor, eps: f64) -> Result<Self> {
        let weight = weight.dequantize(&weight.device()).map_err(Error::from)?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if input.device().is_metal() {
            let input = if input.is_contiguous() {
                input.clone()
            } else {
                input.contiguous()?
            };
            if let Some(output) = try_fused_rms_norm(&input, &self.weight, self.eps) {
                return Ok(output);
            }
        }
        candle_nn::ops::rms_norm(input, &self.weight, self.eps as f32).map_err(Error::from)
    }

    fn eps(&self) -> f64 {
        self.eps
    }

    fn weight(&self) -> &Tensor {
        &self.weight
    }
}

enum DepthQkvProjection {
    Packed(QLinear),
    Separate {
        q_proj: QLinear,
        k_proj: QLinear,
        v_proj: QLinear,
    },
}

impl DepthAttention {
    fn load(loader: &GgufLoader, device: &Device, idx: usize, dim: usize) -> Result<Self> {
        let packed_qkv_names = vec![
            format!("audio_head.depthformer.blocks.{idx}.attn.qkv_proj.weight"),
            format!("depthformer.layers.{idx}.operator.qkv_proj.weight"),
        ];
        let qkv = if packed_qkv_names.iter().any(|name| loader.has_tensor(name)) {
            DepthQkvProjection::Packed(QLinear::load(loader, device, &packed_qkv_names, &[])?)
        } else {
            DepthQkvProjection::Separate {
                q_proj: QLinear::load(
                    loader,
                    device,
                    &[format!(
                        "audio_head.depthformer.blocks.{idx}.attn.q_proj.weight"
                    )],
                    &[],
                )?,
                k_proj: QLinear::load(
                    loader,
                    device,
                    &[format!(
                        "audio_head.depthformer.blocks.{idx}.attn.k_proj.weight"
                    )],
                    &[],
                )?,
                v_proj: QLinear::load(
                    loader,
                    device,
                    &[format!(
                        "audio_head.depthformer.blocks.{idx}.attn.v_proj.weight"
                    )],
                    &[],
                )?,
            }
        };
        let o_proj = QLinear::load(
            loader,
            device,
            &[
                format!("audio_head.depthformer.blocks.{idx}.attn.o_proj.weight"),
                format!("depthformer.layers.{idx}.operator.out_proj.weight"),
            ],
            &[],
        )?;
        let q_norm = load_rms_norm_any(
            loader,
            device,
            &[
                format!("audio_head.depthformer.blocks.{idx}.attn.q_norm.weight"),
                format!("depthformer.layers.{idx}.operator.attention.q_layernorm.weight"),
            ],
        )?;
        let k_norm = load_rms_norm_any(
            loader,
            device,
            &[
                format!("audio_head.depthformer.blocks.{idx}.attn.k_norm.weight"),
                format!("depthformer.layers.{idx}.operator.attention.k_layernorm.weight"),
            ],
        )?;
        let qk_norm_weight = Tensor::cat(&[q_norm.weight(), k_norm.weight()], 0)?.contiguous()?;
        let (cos, sin) =
            precompute_freqs(dim / DEPTHFORMER_HEADS, DEPTHFORMER_ROPE_BASE, 32, device)?;
        let cos_sin = Tensor::cat(&[&cos, &sin], 1)?.contiguous()?;
        Ok(Self {
            qkv,
            o_proj,
            q_norm,
            k_norm,
            qk_norm_weight,
            cos,
            sin,
            cos_sin,
        })
    }

    fn forward_physical(
        &self,
        hidden: &Tensor,
        index_pos: usize,
        physical_layer: usize,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
    ) -> Result<Tensor> {
        let (batch, seq_len, hidden_size) = hidden.dims3()?;
        let head_dim = hidden_size / DEPTHFORMER_HEADS;
        let kv_hidden = DEPTHFORMER_KV_HEADS * head_dim;

        let (q_hidden, k_hidden, v_hidden) = match &self.qkv {
            DepthQkvProjection::Packed(qkv_proj) => {
                let qkv = qkv_proj.forward(hidden)?;
                let total = qkv.dim(2)?;
                let expected = hidden_size + (2 * kv_hidden);
                if total != expected {
                    return Err(Error::ModelLoadError(format!(
                        "Packed depthformer qkv projection has unexpected width: got {total}, expected {expected}"
                    )));
                }
                let q = qkv.i((.., .., ..hidden_size))?.contiguous()?;
                let k = qkv
                    .i((.., .., hidden_size..hidden_size + kv_hidden))?
                    .contiguous()?;
                let v = qkv.i((.., .., hidden_size + kv_hidden..))?.contiguous()?;
                (q, k, v)
            }
            DepthQkvProjection::Separate {
                q_proj,
                k_proj,
                v_proj,
            } => (
                q_proj.forward(hidden)?,
                k_proj.forward(hidden)?,
                v_proj.forward(hidden)?,
            ),
        };

        // reshape [b, s, hidden] -> [b, s, heads, head_dim]
        let q = q_hidden
            .reshape((batch, seq_len, DEPTHFORMER_HEADS, head_dim))?
            .contiguous()?;
        let k = k_hidden
            .reshape((batch, seq_len, DEPTHFORMER_KV_HEADS, head_dim))?
            .contiguous()?;
        let v = v_hidden
            .reshape((batch, seq_len, DEPTHFORMER_KV_HEADS, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // norm on last dim (head_dim), then RoPE
        let (q, k) = if seq_len == 1 && (q.device().is_metal() || q.device().is_cuda()) {
            if let Some((q, k)) =
                try_fused_qk_rms_norm(&q, &k, &self.qk_norm_weight, self.q_norm.eps())
            {
                (q, k)
            } else {
                (self.q_norm.forward(&q)?, self.k_norm.forward(&k)?)
            }
        } else {
            (self.q_norm.forward(&q)?, self.k_norm.forward(&k)?)
        };

        let (q, k) = if let Some((q, k)) =
            try_apply_rotary_emb_pair_bshd(&q, &k, &self.cos_sin, index_pos)?
        {
            (
                q.transpose(1, 2)?.contiguous()?,
                k.transpose(1, 2)?.contiguous()?,
            )
        } else {
            let q = q.transpose(1, 2)?.contiguous()?;
            let k = k.transpose(1, 2)?.contiguous()?;
            (
                apply_rotary_emb(&q, &self.cos, &self.sin, index_pos)?,
                apply_rotary_emb(&k, &self.cos, &self.sin, index_pos)?,
            )
        };

        let queries = q.squeeze(0)?.transpose(0, 1)?.contiguous()?;
        let keys = k.squeeze(0)?.transpose(0, 1)?.contiguous()?;
        let values = v.squeeze(0)?.transpose(0, 1)?.contiguous()?;
        let out = cache.write_and_attend(
            physical_layer,
            prepared,
            &queries,
            &keys,
            &values,
            (1.0f64 / (head_dim as f64).sqrt()) as f32,
        )?;
        let out = out.reshape((batch, seq_len, hidden_size))?;
        self.o_proj.forward(&out)
    }

    fn forward_physical_batch(
        &self,
        hidden: &Tensor,
        positions: &[usize],
        physical_layer: usize,
        caches: &[&PhysicalPagedKvCache],
        slots: &dyn KvSlotMap,
        completions: &mut KvWriteCompletionCollector,
    ) -> Result<Tensor> {
        let (batch, seq_len, hidden_size) = hidden.dims3()?;
        if batch <= 1 || seq_len != 1 || positions.len() != batch || caches.len() != batch {
            return Err(Error::InvalidInput(
                "Depthformer attention batch dimensions do not match".into(),
            ));
        }
        let first = caches[0];
        if slots.arena_id() != first.arena().id()
            || slots.len() != batch
            || caches
                .iter()
                .any(|cache| !Arc::ptr_eq(cache.arena(), first.arena()))
        {
            return Err(Error::InvalidInput(
                "Depthformer attention rows do not share physical authority".into(),
            ));
        }
        let head_dim = hidden_size / DEPTHFORMER_HEADS;
        let kv_hidden = DEPTHFORMER_KV_HEADS * head_dim;
        let (q_hidden, k_hidden, v_hidden) = match &self.qkv {
            DepthQkvProjection::Packed(qkv_proj) => {
                let qkv = qkv_proj.forward(hidden)?;
                let expected = hidden_size + 2 * kv_hidden;
                if qkv.dim(2)? != expected {
                    return Err(Error::ModelLoadError(
                        "Packed depthformer batch qkv width is invalid".into(),
                    ));
                }
                (
                    qkv.i((.., .., ..hidden_size))?.contiguous()?,
                    qkv.i((.., .., hidden_size..hidden_size + kv_hidden))?
                        .contiguous()?,
                    qkv.i((.., .., hidden_size + kv_hidden..))?.contiguous()?,
                )
            }
            DepthQkvProjection::Separate {
                q_proj,
                k_proj,
                v_proj,
            } => (
                q_proj.forward(hidden)?,
                k_proj.forward(hidden)?,
                v_proj.forward(hidden)?,
            ),
        };
        let q = q_hidden
            .reshape((batch, 1, DEPTHFORMER_HEADS, head_dim))?
            .contiguous()?;
        let k = k_hidden
            .reshape((batch, 1, DEPTHFORMER_KV_HEADS, head_dim))?
            .contiguous()?;
        let values = v_hidden
            .reshape((batch, DEPTHFORMER_KV_HEADS, head_dim))?
            .contiguous()?;
        let (q, k) = if q.device().is_metal() || q.device().is_cuda() {
            match try_fused_qk_rms_norm(&q, &k, &self.qk_norm_weight, self.q_norm.eps()) {
                Some(pair) => pair,
                None => (self.q_norm.forward(&q)?, self.k_norm.forward(&k)?),
            }
        } else {
            (self.q_norm.forward(&q)?, self.k_norm.forward(&k)?)
        };
        let mut queries = Vec::with_capacity(batch);
        let mut keys = Vec::with_capacity(batch);
        for row in 0..batch {
            let q = q.narrow(0, row, 1)?;
            let k = k.narrow(0, row, 1)?;
            let (q, k) = if let Some((q, k)) =
                try_apply_rotary_emb_pair_bshd(&q, &k, &self.cos_sin, positions[row])?
            {
                (q.transpose(1, 2)?, k.transpose(1, 2)?)
            } else {
                let q = q.transpose(1, 2)?.contiguous()?;
                let k = k.transpose(1, 2)?.contiguous()?;
                (
                    apply_rotary_emb(&q, &self.cos, &self.sin, positions[row])?,
                    apply_rotary_emb(&k, &self.cos, &self.sin, positions[row])?,
                )
            };
            queries.push(q.reshape((DEPTHFORMER_HEADS, head_dim))?);
            keys.push(k.reshape((DEPTHFORMER_KV_HEADS, head_dim))?);
        }
        let queries = Tensor::stack(&queries.iter().collect::<Vec<_>>(), 0)?.contiguous()?;
        let keys = Tensor::stack(&keys.iter().collect::<Vec<_>>(), 0)?.contiguous()?;
        let metadata = KvDecodeBatchMetadata {
            sequences: caches
                .iter()
                .enumerate()
                .map(|(row, cache)| cache.sequence_table(positions[row] + 1))
                .collect::<Result<Vec<_>>>()?,
        };
        let binding = first.layer_binding(physical_layer)?;
        let completion = first.arena().write_slots(
            binding,
            KvWriteArgs {
                keys: &keys,
                values: &values,
                slots,
            },
        )?;
        let (out, completion) = submit_ordered_after_write(completion, || {
            first.arena().paged_decode(
                binding,
                PagedKvDecodeArgs {
                    queries: &queries,
                    batch: &metadata,
                    softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                    softcap: None,
                },
            )
        })?;
        completions.collect(completion)?;
        self.o_proj.forward(&out.reshape((batch, 1, hidden_size))?)
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
                &[
                    format!("audio_head.depthformer.blocks.{idx}.ffn.w1.weight"),
                    format!("depthformer.layers.{idx}.feed_forward.w1.weight"),
                ],
                &[],
            )?,
            w2: QLinear::load(
                loader,
                device,
                &[
                    format!("audio_head.depthformer.blocks.{idx}.ffn.w2.weight"),
                    format!("depthformer.layers.{idx}.feed_forward.w2.weight"),
                ],
                &[],
            )?,
            w3: QLinear::load(
                loader,
                device,
                &[
                    format!("audio_head.depthformer.blocks.{idx}.ffn.w3.weight"),
                    format!("depthformer.layers.{idx}.feed_forward.w3.weight"),
                ],
                &[],
            )?,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let gate = self.w1.forward(hidden)?;
        let up = self.w3.forward(hidden)?;
        let hidden = if let Some(fused) = try_fused_silu_mul_with_status(&gate, &up) {
            fused.tensor
        } else {
            let gate = candle_nn::ops::silu(&gate)?;
            gate.broadcast_mul(&up)?
        };
        self.w2.forward(&hidden)
    }
}

struct CodebookEmbeddingHead {
    embedding: Embedding,
    norm: DepthRmsNorm,
    to_logits: QLinear,
}

enum CodebookSample {
    Host(u32),
    DeviceGreedy(Tensor),
}

impl CodebookSample {
    fn to_token(&self) -> Result<u32> {
        match self {
            CodebookSample::Host(token) => Ok(*token),
            CodebookSample::DeviceGreedy(token) => {
                token.squeeze(0)?.to_scalar::<u32>().map_err(Error::from)
            }
        }
    }
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
        self.embed_token_tensor(&token)
    }

    fn embed_sample(&self, sample: &CodebookSample, device: &Device) -> Result<Tensor> {
        match sample {
            CodebookSample::Host(token) => self.embed(*token, device),
            CodebookSample::DeviceGreedy(token) => self.embed_token_tensor(token),
        }
    }

    fn embed_token_tensor(&self, token: &Tensor) -> Result<Tensor> {
        let embed = self.embedding.forward(token)?;
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
    ) -> Result<CodebookSample> {
        let logits = self.logits(hidden)?;
        let vocab_limit = logits.dim(0)?;
        if config.temperature <= 1e-5 {
            if let Some(token) = greedy_token_tensor_from_logits(&logits, vocab_limit)? {
                return Ok(CodebookSample::DeviceGreedy(token));
            }
        }
        sample_from_logits(&logits, vocab_limit, config, rng).map(CodebookSample::Host)
    }
}

struct MaterializedCodebookSamples {
    samples: Vec<u32>,
    pack_ms: f64,
    readback_ms: f64,
}

fn materialize_codebook_samples_with_profile(
    samples: &[CodebookSample],
) -> Result<MaterializedCodebookSamples> {
    if samples
        .iter()
        .all(|sample| matches!(sample, CodebookSample::DeviceGreedy(_)))
    {
        let tensors = samples
            .iter()
            .filter_map(|sample| match sample {
                CodebookSample::DeviceGreedy(token) => Some(token),
                CodebookSample::Host(_) => None,
            })
            .collect::<Vec<_>>();
        if !tensors.is_empty() {
            let pack_started = Instant::now();
            let packed = Tensor::cat(&tensors, 0)?;
            let pack_ms = elapsed_ms(pack_started);
            let readback_started = Instant::now();
            let samples = packed.to_vec1::<u32>().map_err(Error::from)?;
            let readback_ms = elapsed_ms(readback_started);
            return Ok(MaterializedCodebookSamples {
                samples,
                pack_ms,
                readback_ms,
            });
        }
    }

    let readback_started = Instant::now();
    let samples = samples
        .iter()
        .map(|sample| match sample {
            CodebookSample::Host(token) => Ok(*token),
            CodebookSample::DeviceGreedy(token) => {
                token.squeeze(0)?.to_scalar::<u32>().map_err(Error::from)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(MaterializedCodebookSamples {
        samples,
        pack_ms: 0.0,
        readback_ms: elapsed_ms(readback_started),
    })
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

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn load_embedding_any(loader: &GgufLoader, device: &Device, names: &[String]) -> Result<Embedding> {
    let weight = load_qtensor_any(loader, device, names)?
        .dequantize(device)
        .map_err(Error::from)?;
    let (_, dim) = weight.dims2()?;
    Ok(Embedding::new(weight, dim))
}

fn load_rms_norm_any(
    loader: &GgufLoader,
    device: &Device,
    names: &[String],
) -> Result<DepthRmsNorm> {
    DepthRmsNorm::from_qtensor(
        load_qtensor_any(loader, device, names)?,
        DEPTHFORMER_NORM_EPS,
    )
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
    record_rope_kernel();
    candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin).map_err(Error::from)
}

fn try_apply_rotary_emb_pair_bshd(
    q: &Tensor,
    k: &Tensor,
    cos_sin: &Tensor,
    index_pos: usize,
) -> Result<Option<(Tensor, Tensor)>> {
    let (_, seq_len, _, _) = q.dims4()?;
    let packed = cos_sin.narrow(0, index_pos, seq_len)?.contiguous()?;
    if let Some((q, k)) = try_fused_rope_pair_bshd(&q.contiguous()?, &k.contiguous()?, &packed) {
        record_rope_kernel();
        record_rope_kernel();
        return Ok(Some((q, k)));
    }
    Ok(None)
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
