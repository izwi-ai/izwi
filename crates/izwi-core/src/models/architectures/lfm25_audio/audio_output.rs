use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::Module;
use candle_transformers::models::with_tracing::QMatMul;

use crate::backends::kv::{
    submit_ordered_after_write, KvSlotMap, KvWriteArgs, KvWriteCompletionCollector,
    PagedKvDecodeArgs,
};
use crate::error::{Error, Result};
use crate::kernels::{try_fused_qk_rms_norm, try_fused_rms_norm, try_fused_silu_mul_with_status};
use crate::kv::KvDecodeBatchMetadata;
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PreparedPhysicalPagedStep};
use crate::models::shared::sampling::device_resident_top_k_sample;
use crate::models::shared::telemetry::record_rope_kernel;
use crate::models::shared::weights::gguf::GgufLoader;
use crate::models::shared::weights::token_table::QuantizedTokenTable;

use super::config::Lfm25AudioDecoderConfig;
use super::sampling::{
    greedy_token_tensor_from_logits, sample_from_logits, Lfm25SamplingConfig, SimpleRng,
};

use super::config::{
    LFM25_DEPTHFORMER_KV_HEADS as DEPTHFORMER_KV_HEADS,
    LFM25_DEPTHFORMER_QUERY_HEADS as DEPTHFORMER_HEADS,
};
const DEPTHFORMER_NORM_EPS: f64 = 1e-5;
/// The reference Depthformer's `MHA(theta=1_000_000.0)` (liquid-audio
/// transformer.py; llama.cpp audio-decoder uses base 1e6 too).
const DEPTHFORMER_ROPE_BASE: f64 = 1_000_000.0;

pub struct Lfm25AudioHead {
    frame_embedding: QuantizedTokenTable,
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
        Ok(materialize_codebook_samples_with_profile(self.samples.iter())?.samples)
    }

    /// Materialize all rows together, preserving each frame's codebook order.
    /// Greedy CUDA frames cross the device/host boundary only once per batch.
    pub(crate) fn tokens_batch(frames: &[Self]) -> Result<Vec<Vec<u32>>> {
        let materialized = materialize_codebook_samples_with_profile(
            frames.iter().flat_map(|frame| frame.samples.iter()),
        )?;
        let mut offset = 0;
        Ok(frames
            .iter()
            .map(|frame| {
                let end = offset + frame.samples.len();
                let tokens = materialized.samples[offset..end].to_vec();
                offset = end;
                tokens
            })
            .collect())
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
        let frame_embedding = load_token_table_any(
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
            depth_embeddings.push(CodebookEmbeddingHead::load(
                loader,
                device,
                idx,
                cfg.audio_end_token_id,
            )?);
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
        let embeds = self.frame_embedding.embed(&token_ids)?;
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
        let materialized = materialize_codebook_samples_with_profile(frame.samples.iter())?;
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

        Ok((self.finish_frame(samples, hidden.device())?, profile))
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
            return Ok(vec![self.finish_frame(samples, hidden.device())?]);
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
            .map(|samples| self.finish_frame(samples, hidden.device()))
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
                .all(|sample| matches!(sample, CodebookSample::Device(_)))
            {
                let tensors = samples
                    .iter()
                    .filter_map(|sample| match sample {
                        CodebookSample::Device(token) => Some(token),
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
            .all(|sample| matches!(sample, CodebookSample::Device(_)))
        {
            let tensors = frames
                .iter()
                .flat_map(|frame| frame.samples.iter())
                .filter_map(|sample| match sample {
                    CodebookSample::Device(token) => Some(token),
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

    /// Close one sampled frame as the references do: when codebook 0 is
    /// end-of-audio, every codebook becomes end-of-audio (`next_token[:] =
    /// 2048` in liquid-audio, the same in llama.cpp), and the frame embedding
    /// fed back to the backbone is built from those codes. Device-resident
    /// codes stay on device; the override is a `where` on the stacked codes.
    fn finish_frame(
        &self,
        samples: Vec<CodebookSample>,
        device: &Device,
    ) -> Result<Lfm25SampledAudioFrame> {
        if samples.len() != self.codebooks {
            return Err(Error::InvalidInput(format!(
                "Expected {} audio codebooks, received {}",
                self.codebooks,
                samples.len()
            )));
        }
        if samples
            .iter()
            .all(|sample| matches!(sample, CodebookSample::Host(_)))
        {
            let mut tokens = samples
                .iter()
                .map(CodebookSample::to_token)
                .collect::<Result<Vec<_>>>()?;
            end_of_audio_frame_host(&mut tokens, self.audio_end_token_id);
            let embedding = self.embed_audio_frame(&tokens, device)?;
            return Ok(Lfm25SampledAudioFrame {
                samples: tokens.into_iter().map(CodebookSample::Host).collect(),
                embedding,
            });
        }

        let tokens = samples
            .iter()
            .map(|sample| match sample {
                CodebookSample::Host(token) => {
                    Tensor::from_vec(vec![*token], (1,), device).map_err(Error::from)
                }
                CodebookSample::Device(token) => Ok(token.clone()),
            })
            .collect::<Result<Vec<_>>>()?;
        let tokens = end_of_audio_frame_device(&Tensor::cat(&tokens, 0)?, self.audio_end_token_id)?;
        let token_ids = tokens
            .broadcast_add(&self.codebook_offsets_tensor)?
            .reshape((1, self.codebooks))?
            .contiguous()?;
        let embedding = self
            .frame_embedding
            .embed(&token_ids)?
            .sum(1)?
            .unsqueeze(1)?;
        let samples = (0..self.codebooks)
            .map(|codebook| tokens.narrow(0, codebook, 1).map(CodebookSample::Device))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Lfm25SampledAudioFrame { samples, embedding })
    }
}

/// When codebook 0 is end-of-audio, every codebook becomes end-of-audio.
fn end_of_audio_frame_host(tokens: &mut [u32], end: u32) {
    if tokens.first() == Some(&end) {
        tokens.fill(end);
    }
}

/// [`end_of_audio_frame_host`] on a `[codebooks]` U32 tensor, without a
/// readback.
fn end_of_audio_frame_device(tokens: &Tensor, end: u32) -> Result<Tensor> {
    let codebooks = tokens.dim(0)?;
    let all_end = Tensor::full(end, codebooks, tokens.device())?;
    let is_end = tokens
        .narrow(0, 0, 1)?
        .to_dtype(DType::F32)?
        .eq(f64::from(end))?
        .broadcast_as(codebooks)?;
    is_end
        .where_cond(&all_end, tokens)?
        .contiguous()
        .map_err(Error::from)
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
        Ok(Self {
            qkv,
            o_proj,
            q_norm,
            k_norm,
            qk_norm_weight,
            cos,
            sin,
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

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let (q, k) = (
            apply_rotary_emb(&q, &self.cos, &self.sin, index_pos)?,
            apply_rotary_emb(&k, &self.cos, &self.sin, index_pos)?,
        );

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
            let q = q.transpose(1, 2)?.contiguous()?;
            let k = k.transpose(1, 2)?.contiguous()?;
            let (q, k) = (
                apply_rotary_emb(&q, &self.cos, &self.sin, positions[row])?,
                apply_rotary_emb(&k, &self.cos, &self.sin, positions[row])?,
            );
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
    embedding: QuantizedTokenTable,
    norm: DepthRmsNorm,
    to_logits: QLinear,
    /// Codes this head may emit: all 2049 for codebook 0, below the
    /// end-of-audio code for codebooks 1-7. EOA is only meaningful in
    /// codebook 0 (training sets every codebook to it at once), and a stray
    /// 2048 in codebooks 1-7 is an index the detokenizer cannot decode.
    code_limit: Option<usize>,
}

/// One sampled code: read back already, or still a `[1]` U32 tensor on the
/// accelerator (argmax, or a device-resident Gumbel-max draw).
enum CodebookSample {
    Host(u32),
    Device(Tensor),
}

/// Audio codes drawn on device without a readback (diagnostics).
static DEVICE_SAMPLED_AUDIO_CODES: AtomicU64 = AtomicU64::new(0);

pub(crate) fn lfm25_device_sampled_audio_codes() -> u64 {
    DEVICE_SAMPLED_AUDIO_CODES.load(Ordering::Relaxed)
}

impl CodebookSample {
    fn to_token(&self) -> Result<u32> {
        match self {
            CodebookSample::Host(token) => Ok(*token),
            CodebookSample::Device(token) => {
                token.squeeze(0)?.to_scalar::<u32>().map_err(Error::from)
            }
        }
    }
}

impl CodebookEmbeddingHead {
    fn load(
        loader: &GgufLoader,
        device: &Device,
        idx: usize,
        audio_end_token_id: u32,
    ) -> Result<Self> {
        let embedding = load_token_table_any(
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
            code_limit: (idx > 0).then_some(audio_end_token_id as usize),
        })
    }

    fn embed(&self, token: u32, device: &Device) -> Result<Tensor> {
        let token = Tensor::from_vec(vec![token], (1,), device)?;
        self.embed_token_tensor(&token)
    }

    fn embed_sample(&self, sample: &CodebookSample, device: &Device) -> Result<Tensor> {
        match sample {
            CodebookSample::Host(token) => self.embed(*token, device),
            CodebookSample::Device(token) => self.embed_token_tensor(token),
        }
    }

    fn embed_token_tensor(&self, token: &Tensor) -> Result<Tensor> {
        let embed = self.embedding.embed(token)?;
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
        let vocab = logits.dim(0)?;
        let vocab_limit = self.code_limit.map_or(vocab, |limit| limit.min(vocab));
        if config.temperature <= 1e-5 {
            if let Some(token) = greedy_token_tensor_from_logits(&logits, vocab_limit)? {
                return Ok(CodebookSample::Device(token));
            }
        } else if config.top_p >= 1.0 {
            // Keep the draw on device so the frame needs one readback, not
            // one per codebook (LFT-9).
            if let Some(token) = device_resident_top_k_sample(
                &logits,
                vocab_limit,
                config.top_k,
                config.temperature,
                &mut || rng.next_f32(),
            )? {
                DEVICE_SAMPLED_AUDIO_CODES.fetch_add(1, Ordering::Relaxed);
                return Ok(CodebookSample::Device(token));
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

fn materialize_codebook_samples_with_profile<'a>(
    samples: impl Iterator<Item = &'a CodebookSample> + Clone,
) -> Result<MaterializedCodebookSamples> {
    if samples
        .clone()
        .all(|sample| matches!(sample, CodebookSample::Device(_)))
    {
        let tensors = samples
            .clone()
            .filter_map(|sample| match sample {
                CodebookSample::Device(token) => Some(token),
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
        .clone()
        .map(|sample| match sample {
            CodebookSample::Host(token) => Ok(*token),
            CodebookSample::Device(token) => {
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

/// An embedding table kept in its stored (quantized) form: rows are gathered
/// and dequantized on demand, exactly the dequantized table's rows.
fn load_token_table_any(
    loader: &GgufLoader,
    device: &Device,
    names: &[String],
) -> Result<QuantizedTokenTable> {
    QuantizedTokenTable::new(load_qtensor_any(loader, device, names)?)
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

/// RoPE cos/sin `[positions, head_dim / 2]`, built on the host in f64 (device
/// fast-math cos/sin is not exact) and uploaded once.
fn precompute_freqs(
    head_dim: usize,
    freq_base: f64,
    context_length: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(context_length * half);
    let mut sin = Vec::with_capacity(context_length * half);
    for position in 0..context_length {
        for pair in 0..half {
            let inv_freq = 1.0 / freq_base.powf((2 * pair) as f64 / head_dim as f64);
            let angle = position as f64 * inv_freq;
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    Ok((
        Tensor::from_vec(cos, (context_length, half), device)?,
        Tensor::from_vec(sin, (context_length, half), device)?,
    ))
}

/// The reference's complex-multiply RoPE on `[batch, heads, seq, head_dim]`:
/// dimensions `(2i, 2i + 1)` form one pair (`view_as_complex` over
/// "(D two)"; llama.cpp `LLAMA_ROPE_TYPE_NORM`), not `(i, i + head_dim / 2)`.
fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor, index_pos: usize) -> Result<Tensor> {
    let (_, _, seq_len, _) = x.dims4()?;
    let cos = cos.narrow(0, index_pos, seq_len)?;
    let sin = sin.narrow(0, index_pos, seq_len)?;
    record_rope_kernel();
    candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin).map_err(Error::from)
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

#[cfg(test)]
mod tests {
    use super::{CodebookSample, Lfm25SampledAudioFrame};
    use candle_core::{DType, Device, Tensor};

    /// LFT-4: an end-of-audio frame is fed back as all end-of-audio codes,
    /// on host and on device; EOA outside codebook 0 changes nothing.
    #[test]
    fn end_of_audio_frame_feeds_back_all_end_codes() {
        let end = 2048u32;
        let mut eoa = vec![end, 5, 6, 7, 1, 2, 3, 4];
        super::end_of_audio_frame_host(&mut eoa, end);
        assert_eq!(eoa, vec![end; 8]);
        let mut speech = vec![9, end, 6, 7, 1, 2, 3, 4];
        super::end_of_audio_frame_host(&mut speech, end);
        assert_eq!(speech, vec![9, end, 6, 7, 1, 2, 3, 4]);

        let device = Device::Cpu;
        let run = |tokens: Vec<u32>| {
            let tensor = Tensor::from_vec(tokens, 8, &device).unwrap();
            super::end_of_audio_frame_device(&tensor, end)
                .unwrap()
                .to_vec1::<u32>()
                .unwrap()
        };
        assert_eq!(run(vec![end, 5, 6, 7, 1, 2, 3, 4]), vec![end; 8]);
        assert_eq!(
            run(vec![9, end, 6, 7, 1, 2, 3, 4]),
            vec![9, end, 6, 7, 1, 2, 3, 4]
        );
    }

    /// The device end-of-audio override runs on Metal without a readback.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_end_of_audio_frame_override_matches_host() {
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "metal_end_of_audio_frame_override_matches_host",
        ) else {
            return;
        };
        for tokens in [
            vec![2048u32, 5, 6, 7, 1, 2, 3, 4],
            vec![9, 2048, 6, 7, 1, 2, 3, 4],
        ] {
            let mut expected = tokens.clone();
            super::end_of_audio_frame_host(&mut expected, 2048);
            let tensor = Tensor::from_vec(tokens, 8, &device).unwrap();
            let actual = super::end_of_audio_frame_device(&tensor, 2048)
                .unwrap()
                .to_vec1::<u32>()
                .unwrap();
            assert_eq!(actual, expected);
        }
    }

    /// A codebook head over 3 codes (EOA = 2) whose logits always favour EOA.
    fn eoa_favouring_head(idx: usize) -> super::CodebookEmbeddingHead {
        eoa_favouring_head_on(idx, &Device::Cpu)
    }

    fn eoa_favouring_head_on(idx: usize, device: &Device) -> super::CodebookEmbeddingHead {
        use crate::models::shared::weights::token_table::QuantizedTokenTable;
        use candle_core::quantized::{GgmlDType, QTensor};
        use candle_transformers::models::with_tracing::QMatMul;

        let quantize = |values: Vec<f32>, shape: &[usize]| {
            QTensor::quantize(
                &Tensor::from_vec(values, shape, device).unwrap(),
                GgmlDType::F32,
            )
            .unwrap()
        };
        // Rows are output codes: code 2 reads every hidden unit with weight 5.
        let to_logits = quantize(
            vec![0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0],
            &[3, 4],
        );
        super::CodebookEmbeddingHead {
            embedding: QuantizedTokenTable::new(quantize(vec![0.0; 12], &[3, 4])).unwrap(),
            norm: super::DepthRmsNorm::from_qtensor(quantize(vec![1.0; 4], &[4]), 1e-5).unwrap(),
            to_logits: super::QLinear {
                weight: QMatMul::from_weights(std::sync::Arc::new(to_logits)).unwrap(),
                bias: None,
            },
            code_limit: (idx > 0).then_some(2),
        }
    }

    /// LFT-11: codebooks 1-7 never emit the end-of-audio code (which the
    /// detokenizer cannot decode), greedy or sampled; codebook 0 still can.
    #[test]
    fn codebooks_after_the_first_never_emit_end_of_audio() {
        use super::super::sampling::{Lfm25SamplingConfig, SimpleRng};

        let hidden = Tensor::ones((1, 1, 4), DType::F32, &Device::Cpu).unwrap();
        let greedy = Lfm25SamplingConfig::default();
        let sampled = Lfm25SamplingConfig {
            temperature: 50.0,
            top_k: 0,
            top_p: 1.0,
        };
        let mut rng = SimpleRng::new(17);
        let first = eoa_favouring_head(0);
        assert_eq!(
            first
                .sample(&hidden, &greedy, &mut rng)
                .unwrap()
                .to_token()
                .unwrap(),
            2
        );
        let later = eoa_favouring_head(3);
        assert_ne!(
            later
                .sample(&hidden, &greedy, &mut rng)
                .unwrap()
                .to_token()
                .unwrap(),
            2
        );
        for _ in 0..64 {
            let code = later
                .sample(&hidden, &sampled, &mut rng)
                .unwrap()
                .to_token()
                .unwrap();
            assert!(code < 2, "codebook 3 emitted end-of-audio");
        }
    }

    /// LFT-9: on Metal a sampled audio code stays on device (no per-codebook
    /// readback) and still respects the codebook's code limit.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_sampled_audio_codes_stay_on_device() {
        use super::super::sampling::{Lfm25SamplingConfig, SimpleRng};

        let Some((device, _lease)) =
            crate::backends::metal_test_device_or_skip("metal_sampled_audio_codes_stay_on_device")
        else {
            return;
        };
        let head = eoa_favouring_head_on(3, &device);
        let hidden = Tensor::ones((1, 1, 4), DType::F32, &device).unwrap();
        let config = Lfm25SamplingConfig {
            temperature: 50.0,
            top_k: 2,
            top_p: 1.0,
        };
        let mut rng = SimpleRng::new(23);
        let before = super::lfm25_device_sampled_audio_codes();
        for _ in 0..16 {
            let sample = head.sample(&hidden, &config, &mut rng).unwrap();
            assert!(matches!(sample, CodebookSample::Device(_)));
            assert!(sample.to_token().unwrap() < 2);
        }
        assert!(super::lfm25_device_sampled_audio_codes() >= before + 16);
    }

    /// LFT-3: Depthformer RoPE matches the reference `apply_rotary_emb`
    /// (complex multiply over adjacent pairs, theta 1e6) at codebook
    /// positions 4..7 for the published head_dim 32.
    #[test]
    fn depthformer_rope_is_interleaved_with_base_1e6() {
        use crate::kernels::numeric_check::{host_f32, max_abs_diff, xorshift_tensor};

        let (heads, seq, head_dim, start) = (2usize, 4usize, 32usize, 4usize);
        let x = xorshift_tensor(&[1, heads, seq, head_dim], 3, 1.0, &Device::Cpu).unwrap();
        let (cos, sin) =
            super::precompute_freqs(head_dim, super::DEPTHFORMER_ROPE_BASE, 32, &Device::Cpu)
                .unwrap();
        let out = host_f32(&super::apply_rotary_emb(&x, &cos, &sin, start).unwrap()).unwrap();
        let input = host_f32(&x).unwrap();
        let mut reference = vec![0f32; input.len()];
        for row in 0..heads * seq {
            let position = (start + row % seq) as f64;
            for pair in 0..head_dim / 2 {
                let angle = position / 1e6f64.powf((2 * pair) as f64 / head_dim as f64);
                let base = row * head_dim + 2 * pair;
                let (a, b) = (f64::from(input[base]), f64::from(input[base + 1]));
                reference[base] = (a * angle.cos() - b * angle.sin()) as f32;
                reference[base + 1] = (a * angle.sin() + b * angle.cos()) as f32;
            }
        }
        let error = max_abs_diff(&out, &reference);
        assert!(error < 1e-5, "Depthformer RoPE max abs error {error}");
    }

    /// The interleaved RoPE runs on Metal (Candle's `rope_i` kernel) and
    /// agrees with the CPU result.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_depthformer_rope_matches_cpu() {
        use crate::kernels::numeric_check::{host_f32, max_abs_diff, xorshift_tensor};

        let Some((device, _lease)) =
            crate::backends::metal_test_device_or_skip("metal_depthformer_rope_matches_cpu")
        else {
            return;
        };
        let x = xorshift_tensor(&[1, 2, 3, 32], 5, 1.0, &Device::Cpu).unwrap();
        let run = |device: &Device| {
            let (cos, sin) =
                super::precompute_freqs(32, super::DEPTHFORMER_ROPE_BASE, 32, device).unwrap();
            let x = x.to_device(device).unwrap();
            host_f32(&super::apply_rotary_emb(&x, &cos, &sin, 5).unwrap()).unwrap()
        };
        let error = max_abs_diff(&run(&device), &run(&Device::Cpu));
        assert!(error < 1e-5, "Metal vs CPU RoPE max abs error {error}");
    }

    fn frame(tokens: &[u32], host_index: Option<usize>) -> Lfm25SampledAudioFrame {
        frame_on_device(tokens, host_index, &Device::Cpu)
    }

    fn frame_on_device(
        tokens: &[u32],
        host_index: Option<usize>,
        device: &Device,
    ) -> Lfm25SampledAudioFrame {
        Lfm25SampledAudioFrame {
            samples: tokens
                .iter()
                .enumerate()
                .map(|(index, &token)| {
                    if host_index == Some(index) {
                        CodebookSample::Host(token)
                    } else {
                        CodebookSample::Device(Tensor::new(&[token], device).unwrap())
                    }
                })
                .collect(),
            embedding: Tensor::zeros((1, 1, 2), DType::F32, device).unwrap(),
        }
    }

    #[test]
    fn lfm25_packed_frame_tokens_preserve_rows_codebooks_and_end_tokens() {
        let expected = vec![
            vec![2048, 1, 2, 3, 4, 5, 6, 7],
            vec![10, 11, 12, 13, 14, 15, 16, 17],
            vec![20, 21, 22, 23, 24, 25, 26, 27],
        ];
        let frames = expected
            .iter()
            .map(|ids| frame(ids, None))
            .collect::<Vec<_>>();
        assert_eq!(
            Lfm25SampledAudioFrame::tokens_batch(&frames).unwrap(),
            expected
        );
        assert_eq!(frames[0].tokens().unwrap(), expected[0]);
    }

    #[test]
    fn lfm25_packed_frame_tokens_handle_mixed_sampling_and_empty_rows() {
        let expected = vec![vec![], vec![2048], vec![7, 3, 9]];
        for host_index in [None, Some(0), Some(1)] {
            let frames = expected
                .iter()
                .map(|ids| frame(ids, host_index))
                .collect::<Vec<_>>();
            assert_eq!(
                Lfm25SampledAudioFrame::tokens_batch(&frames).unwrap(),
                expected
            );
        }
        assert!(Lfm25SampledAudioFrame::tokens_batch(&[])
            .unwrap()
            .is_empty());
        assert!(frame(&[], None).tokens().unwrap().is_empty());
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires CUDA device; run explicitly to validate packed frame readback"]
    fn lfm25_cuda_packed_frame_tokens_preserve_greedy_and_mixed_rows() {
        let device = Device::new_cuda(0).expect("CUDA device required");
        let expected = vec![
            vec![2048, 1, 2, 3, 4, 5, 6, 7],
            vec![9, 8, 7, 6, 5, 4, 3, 2],
        ];
        for host_index in [None, Some(3)] {
            let frames = expected
                .iter()
                .map(|ids| frame_on_device(ids, host_index, &device))
                .collect::<Vec<_>>();
            assert_eq!(
                Lfm25SampledAudioFrame::tokens_batch(&frames).unwrap(),
                expected
            );
            assert_eq!(frames[0].tokens().unwrap(), expected[0]);
        }
    }
}
