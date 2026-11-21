//! Slow semantic transformer for Audio8 DualAR generation.

use candle_core::{DType, Tensor};
use candle_nn::{ops, Embedding, Linear, Module, RmsNorm, VarBuilder};

use crate::error::{Error, Result};
use crate::kernels::cuda::stretch_graphs::StretchGraphs;
use crate::models::architectures::audio8::config::Audio8Config;
use crate::models::architectures::audio8::rotary::Audio8RotaryCache;
use crate::models::architectures::audio8::tokenizer::Audio8ConditioningPrompt;
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PreparedPhysicalPagedStep};

#[derive(Debug, Clone, PartialEq)]
pub struct Audio8SlowConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub codebook_size: usize,
    pub num_codebooks: usize,
    pub semantic_start_token_id: u32,
    pub semantic_end_token_id: u32,
    /// Whether the fused QKV projection carries a bias.
    pub attention_qkv_bias: bool,
}

#[derive(Debug, Clone)]
pub struct Audio8SlowOutput {
    pub logits: Tensor,
    pub hidden_states: Tensor,
}

pub struct Audio8SlowTransformer {
    cfg: Audio8SlowConfig,
    embeddings: Embedding,
    codebook_embeddings: Embedding,
    layers: Vec<Audio8SlowLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    semantic_head: Option<(Linear, u32)>,
    /// On CUDA, the decode step's stretches as graphs (see
    /// [`Self::forward_embeds_batch`]), or why there are none.
    graphs: std::result::Result<StretchGraphs, String>,
}

/// Kill switch for the slow transformer's CUDA decode-step graphs (a
/// boolean; on when unset, read at load). The global
/// `IZWI_CUDA_DECODE_GRAPHS=off` turns them off too.
pub(crate) const AUDIO8_SLOW_GRAPHS_ENV: &str = "IZWI_AUDIO8_SLOW_GRAPHS";

/// Device bytes the slow decode graphs may hold on CUDA: one graph per
/// stretch (layers + 1) at [`AUDIO8_DECODE_ROWS`] rows, each retaining its
/// stable inputs (the residual and attention rows, the rotary rows) and
/// outputs (the residual and the next layer's q/k/v), about 1 MiB a stretch
/// for the published 36-layer, 2560-wide trunk, beside a working set of
/// 32-row MLP activations. Priced resident; the graphs allocate on the
/// first decode steps.
pub(crate) const AUDIO8_SLOW_GRAPH_DEVICE_BYTES: u64 = 64 * 1024 * 1024;

/// [`AUDIO8_SLOW_GRAPH_DEVICE_BYTES`] on `backend` with the graphs on; zero
/// elsewhere.
pub(crate) fn audio8_slow_graph_device_bytes(backend: crate::backends::BackendKind) -> u64 {
    if backend == crate::backends::BackendKind::Cuda
        && crate::env::flag_or(AUDIO8_SLOW_GRAPHS_ENV, true)
        && !crate::kernels::cuda::stretch_graphs::decode_graphs_switched_off()
    {
        AUDIO8_SLOW_GRAPH_DEVICE_BYTES
    } else {
        0
    }
}

struct Audio8SlowLayer {
    input_layernorm: RmsNorm,
    self_attn: Audio8PackedAttention,
    post_attention_layernorm: RmsNorm,
    mlp: Audio8Mlp,
}

struct Audio8PackedAttention {
    qkv_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary: Audio8RotaryCache,
}

struct Audio8Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Audio8SlowConfig {
    pub fn from_config(config: &Audio8Config) -> Result<Self> {
        let text = &config.text_config;
        let head_dim = text
            .head_dim
            .unwrap_or_else(|| text.hidden_size / text.num_attention_heads);
        let intermediate_size = text.intermediate_size.ok_or_else(|| {
            Error::ConfigError("Audio8 text_config missing intermediate_size".to_string())
        })?;
        Ok(Self {
            vocab_size: text.vocab_size,
            hidden_size: text.hidden_size,
            intermediate_size,
            num_hidden_layers: text.num_hidden_layers,
            num_attention_heads: text.num_attention_heads,
            num_key_value_heads: text.num_key_value_heads,
            head_dim,
            max_seq_len: text.max_seq_len,
            rope_theta: text.rope_theta.unwrap_or(1_000_000.0),
            rms_norm_eps: text.rms_norm_eps.unwrap_or(1e-6),
            codebook_size: config.codebook_size,
            num_codebooks: config.num_codebooks,
            semantic_start_token_id: config.semantic_start_token_id,
            semantic_end_token_id: config.semantic_end_token_id,
            attention_qkv_bias: text.attention_qkv_bias,
        })
    }

    fn semantic_contains(&self, token_id: u32) -> bool {
        token_id >= self.semantic_start_token_id && token_id <= self.semantic_end_token_id
    }

    fn q_size(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    fn kv_size(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
}

impl Audio8SlowTransformer {
    pub fn load(cfg: Audio8SlowConfig, vb: VarBuilder) -> Result<Self> {
        let rotary = Audio8RotaryCache::new(
            cfg.max_seq_len,
            cfg.head_dim,
            cfg.rope_theta,
            DType::BF16,
            vb.device(),
        )?;
        let embeddings =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let codebook_embeddings = candle_nn::embedding(
            cfg.codebook_size * cfg.num_codebooks,
            cfg.hidden_size,
            vb.pp("codebook_embeddings"),
        )?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            layers.push(Audio8SlowLayer::load(
                &cfg,
                &rotary,
                vb.pp(format!("layers.{idx}")),
            )?);
        }
        let norm = load_rms_norm_alias(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            &vb,
            &["norm", "model.norm"],
        )?;
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            Linear::new(embeddings.embeddings().clone(), None)
        };
        // Every CUDA decode step pads to one width (`decode_rows`): one set
        // of graphs, no per-row state to copy.
        let graphs = StretchGraphs::resolve(
            "Audio8 slow transformer",
            AUDIO8_SLOW_GRAPHS_ENV,
            vb.device(),
            cfg.num_hidden_layers + 1,
        )
        .map(|graphs| graphs.with_widths(|rows| rows == AUDIO8_DECODE_ROWS));
        Ok(Self {
            cfg,
            embeddings,
            codebook_embeddings,
            layers,
            norm,
            lm_head,
            semantic_head: None,
            graphs,
        })
    }

    /// The decode-step graphs' state and counters, for the diagnostics.
    pub(crate) fn graphs_diagnostics(&self) -> serde_json::Value {
        let mut diagnostics = match &self.graphs {
            Ok(graphs) => graphs.diagnostics(),
            Err(reason) => serde_json::json!({ "off": reason }),
        };
        diagnostics["rows"] = AUDIO8_DECODE_ROWS.into();
        diagnostics
    }

    /// Run decode steps eagerly (a test's reference for the graphs).
    #[cfg(test)]
    pub(crate) fn turn_off_graphs(&mut self) {
        self.graphs = Err("turned off by a test".into());
    }

    /// Gather the only output rows sampling reads: EOS followed by the semantic
    /// vocabulary. Input embeddings remain the complete checkpoint matrix.
    pub(crate) fn configure_semantic_head(&mut self, eos: u32) -> Result<()> {
        if eos as usize >= self.cfg.vocab_size
            || (self.cfg.semantic_start_token_id..=self.cfg.semantic_end_token_id).contains(&eos)
        {
            return Err(Error::ModelLoadError(
                "Audio8 EOS must be outside its semantic vocabulary".into(),
            ));
        }
        let weight = self.lm_head.weight();
        let eos_weight = weight.narrow(0, eos as usize, 1)?;
        let semantic_weight = weight.narrow(
            0,
            self.cfg.semantic_start_token_id as usize,
            (self.cfg.semantic_end_token_id - self.cfg.semantic_start_token_id + 1) as usize,
        )?;
        self.semantic_head = Some((
            Linear::new(
                crate::models::shared::memory::exact::exact_concat_rows(&[
                    &eos_weight,
                    &semantic_weight,
                ])?,
                None,
            ),
            eos,
        ));
        Ok(())
    }

    pub(crate) fn eos_logit_index(&self, eos: u32) -> u32 {
        if self.semantic_head.is_some() {
            0
        } else {
            eos
        }
    }

    pub(crate) fn token_id_from_logit(&self, index: u32) -> Result<u32> {
        if let Some((_, eos)) = &self.semantic_head {
            if index == 0 {
                return Ok(*eos);
            }
            let token = self
                .cfg
                .semantic_start_token_id
                .checked_add(index - 1)
                .filter(|id| *id <= self.cfg.semantic_end_token_id)
                .ok_or_else(|| {
                    Error::InferenceError("Audio8 compact head index is out of range".into())
                })?;
            Ok(token)
        } else if (index as usize) < self.cfg.vocab_size {
            Ok(index)
        } else {
            Err(Error::InferenceError(
                "Audio8 head index is out of range".into(),
            ))
        }
    }

    fn project_logits(&self, input: &Tensor) -> Result<Tensor> {
        self.semantic_head
            .as_ref()
            .map(|(head, _)| head)
            .unwrap_or(&self.lm_head)
            .forward(input)
            .map_err(Error::from)
    }

    pub fn config(&self) -> &Audio8SlowConfig {
        &self.cfg
    }

    /// The device the transformer's weights live on.
    pub(crate) fn device(&self) -> &candle_core::Device {
        self.lm_head.weight().device()
    }

    /// The RoPE tables every slow layer shares.
    fn rotary(&self) -> Result<&Audio8RotaryCache> {
        self.layers
            .first()
            .map(|layer| &layer.self_attn.rotary)
            .ok_or_else(|| Error::InvalidInput("Audio8 slow transformer has no layers".into()))
    }

    /// Persistent RoPE table bytes, shared by every slow layer.
    pub fn rotary_cache_bytes(&self) -> u64 {
        self.layers
            .first()
            .map(|layer| layer.self_attn.rotary.storage_bytes())
            .unwrap_or(0)
    }

    pub fn embed_prompt(&self, prompt: &Audio8ConditioningPrompt) -> Result<Tensor> {
        if prompt.values.len() != self.cfg.num_codebooks + 1 {
            return Err(Error::InvalidInput(format!(
                "Audio8 prompt has {} rows, expected {}",
                prompt.values.len(),
                self.cfg.num_codebooks + 1
            )));
        }
        if prompt.prompt_length == 0 || prompt.values[0].len() != prompt.prompt_length {
            return Err(Error::InvalidInput(
                "Audio8 prompt has invalid prompt_length".to_string(),
            ));
        }
        for row in &prompt.values {
            if row.len() != prompt.prompt_length {
                return Err(Error::InvalidInput(
                    "Audio8 prompt rows must all have the same length".to_string(),
                ));
            }
        }

        let device = self.embeddings.embeddings().device();
        let row0 = Tensor::from_vec(prompt.values[0].clone(), (1, prompt.prompt_length), device)?;
        let mut x = self.embeddings.forward(&row0)?;

        let mut vq_sum: Option<Tensor> = None;
        for codebook_idx in 0..self.cfg.num_codebooks {
            let offset = u32::try_from(
                codebook_idx
                    .checked_mul(self.cfg.codebook_size)
                    .ok_or_else(|| {
                        Error::ConfigError(
                            "Audio8 codebook embedding offset overflowed".to_string(),
                        )
                    })?,
            )
            .map_err(|_| {
                Error::ConfigError("Audio8 codebook embedding offset exceeds u32".to_string())
            })?;
            let ids = prompt.values[codebook_idx + 1]
                .iter()
                .map(|code| {
                    code.checked_add(offset).ok_or_else(|| {
                        Error::ConfigError("Audio8 codebook id overflowed".to_string())
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let ids = Tensor::from_vec(ids, (1, prompt.prompt_length), device)?;
            let emb = self.codebook_embeddings.forward(&ids)?;
            vq_sum = Some(match vq_sum {
                Some(sum) => sum.broadcast_add(&emb)?,
                None => emb,
            });
        }

        if let Some(vq_sum) = vq_sum {
            let mask = prompt.values[0]
                .iter()
                .map(|token| {
                    if self.cfg.semantic_contains(*token) {
                        1.0f32
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>();
            let mask = Tensor::from_vec(mask, (1, prompt.prompt_length, 1), device)?
                .to_dtype(x.dtype())?;
            x = x.broadcast_add(&vq_sum.broadcast_mul(&mask)?)?;
        }

        Ok(x)
    }

    pub fn forward_embeds(
        &self,
        x: &Tensor,
        start_pos: usize,
        cache: &mut PhysicalPagedKvCache,
        return_all: bool,
    ) -> Result<Audio8SlowOutput> {
        let (batch_size, sequence_len, hidden_size) = x.dims3()?;
        if batch_size != 1 || sequence_len == 0 || hidden_size != self.cfg.hidden_size {
            return Err(Error::InvalidInput(format!(
                "Audio8 slow physical paging expects [1,sequence,{}], got {:?}",
                self.cfg.hidden_size,
                x.dims()
            )));
        }
        let end_pos = start_pos.checked_add(sequence_len).ok_or_else(|| {
            Error::InvalidInput("Audio8 slow physical context length overflow".into())
        })?;
        if end_pos > self.cfg.max_seq_len || end_pos > cache.capacity_tokens() {
            return Err(Error::InvalidInput(format!(
                "Audio8 slow physical append ends at {end_pos}, beyond model/cache capacity {}/{}",
                self.cfg.max_seq_len,
                cache.capacity_tokens()
            )));
        }
        cache.validate_model(
            self.cfg.num_hidden_layers,
            self.cfg.num_key_value_heads,
            self.cfg.head_dim,
        )?;
        let mut prepared = cache.prepare_append(start_pos, sequence_len)?;
        let mut hidden = x.clone();
        for (idx, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, start_pos, cache, &mut prepared, idx)?;
        }

        let hidden_for_fast = self.norm.forward(&hidden)?;
        let logits_input = if return_all {
            hidden_for_fast.clone()
        } else {
            let seq_len = hidden_for_fast.dim(1)?;
            hidden_for_fast.narrow(1, seq_len - 1, 1)?
        };
        let logits = self.project_logits(&logits_input)?;
        let hidden_states = if return_all {
            hidden_for_fast
        } else {
            let seq_len = hidden_for_fast.dim(1)?;
            hidden_for_fast.narrow(1, seq_len - 1, 1)?
        };
        let output = Audio8SlowOutput {
            logits,
            hidden_states,
        };
        cache.commit_prepared(prepared)?;
        Ok(output)
    }

    /// Packed ragged prefill or one-token decode. Dense work is shared across
    /// all tokens; physical attention retains each row's own causal context.
    pub(crate) fn forward_embeds_batch(
        &self,
        inputs: &[Tensor],
        caches: &mut [&mut PhysicalPagedKvCache],
    ) -> Result<Vec<Audio8SlowOutput>> {
        if inputs.is_empty() || inputs.len() != caches.len() {
            return Err(Error::InvalidInput(
                "Audio8 slow batch rows do not match".into(),
            ));
        }
        let mut counts = Vec::with_capacity(inputs.len());
        let starts = caches
            .iter()
            .map(|cache| cache.context_len())
            .collect::<Vec<_>>();
        for ((x, cache), &start) in inputs.iter().zip(caches.iter()).zip(&starts) {
            let (batch, count, hidden) = x.dims3()?;
            if batch != 1
                || count == 0
                || hidden != self.cfg.hidden_size
                || start
                    .checked_add(count)
                    .is_none_or(|end| end > self.cfg.max_seq_len || end > cache.capacity_tokens())
            {
                return Err(Error::InvalidInput(
                    "Audio8 slow packed row exceeds its shape/context bound".into(),
                ));
            }
            cache.validate_model(
                self.cfg.num_hidden_layers,
                self.cfg.num_key_value_heads,
                self.cfg.head_dim,
            )?;
            counts.push(count);
        }
        let mut batch =
            super::batch::Audio8PhysicalBatch::new(caches, &counts, self.cfg.num_hidden_layers)?;
        let rows = counts.len();
        // A decode step on CUDA runs its dense work at a fixed row count, so
        // cuBLAS picks one kernel whatever the step's width and a request
        // decodes the same alone and beside others (see `decode_rows`).
        let padded = decode_rows(inputs[0].device(), &counts);
        let execution = (|| {
            let real = counts.iter().sum::<usize>();
            let mut hidden = Tensor::cat(inputs, 1)?;
            if let Some(padded) = padded {
                hidden = hidden.pad_with_zeros(1, 0, padded - real)?;
            }
            let width = hidden.dim(1)?;
            // Every packed token's rotary tables at its own position (spare
            // rows at position 0), once for every layer: one rotation per
            // projection, not one per row.
            let positions = starts
                .iter()
                .zip(&counts)
                .flat_map(|(&start, &count)| start..start + count)
                .collect::<Vec<_>>();
            let (cos, sin) = self.rotary()?.position_tables(&positions, width)?;
            // The layers run as `layers + 1` stretches around the paged
            // attention: the first layer's norm, projections and rotation;
            // then each layer's residual branches with the next layer's
            // projections; then the last layer's residual branches and the
            // final norm. On CUDA a decode step replays each stretch as a
            // graph at the padded width, captured and checked bit for bit
            // against the eager stretch first; elsewhere (and for prefill)
            // the stretches run as they are, in the same order. A graph
            // replays the eager stretch's kernels at the same shapes, so
            // the bits are the eager step's; attention reads the real rows
            // only.
            let mut run = match (&self.graphs, padded) {
                (Ok(graphs), Some(padded)) => graphs.begin(padded),
                _ => None,
            };
            let mut stretch = |index: usize,
                               inputs: Vec<Tensor>,
                               body: &dyn Fn(&[Tensor]) -> Result<Vec<Tensor>>|
             -> Result<Vec<Tensor>> {
                match run.as_mut() {
                    Some(run) => run.run(index, &inputs, body),
                    None => body(&inputs),
                }
            };
            let first = |inputs: &[Tensor]| -> Result<Vec<Tensor>> {
                let (q, k, v) = self.layers[0].pre_attention(&inputs[0], &inputs[1], &inputs[2])?;
                Ok(vec![q, k, v])
            };
            let mut qkv = stretch(0, vec![hidden.clone(), cos.clone(), sin.clone()], &first)?;
            let last = self.layers.len() - 1;
            let mut tail = Vec::new();
            for (index, layer) in self.layers.iter().enumerate() {
                let attn = layer.self_attn.attend_batch(
                    &qkv[0], &qkv[1], &qkv[2], real, caches[0], &mut batch, index,
                )?;
                if index == last {
                    // A padded step is a decode step: its tails are its
                    // first rows, and the logits of all the padded rows are
                    // one projection of one shape (each row's logits are
                    // its own: a matmul keeps rows apart).
                    let body = |inputs: &[Tensor]| -> Result<Vec<Tensor>> {
                        let normed = self
                            .norm
                            .forward(&layer.post_attention(&inputs[0], &inputs[1])?)?;
                        if padded.is_none() {
                            return Ok(vec![normed]);
                        }
                        let logits = self.project_logits(&normed.reshape((
                            width,
                            1,
                            self.cfg.hidden_size,
                        ))?)?;
                        Ok(vec![normed, logits])
                    };
                    tail = stretch(index + 1, vec![hidden, attn], &body)?;
                    break;
                }
                let next = &self.layers[index + 1];
                let body = |inputs: &[Tensor]| -> Result<Vec<Tensor>> {
                    let x = layer.post_attention(&inputs[0], &inputs[1])?;
                    let (q, k, v) = next.pre_attention(&x, &inputs[2], &inputs[3])?;
                    Ok(vec![x, q, k, v])
                };
                let mut outputs = stretch(
                    index + 1,
                    vec![hidden, attn, cos.clone(), sin.clone()],
                    &body,
                )?;
                hidden = outputs.remove(0);
                qkv = outputs;
            }
            let mut tail = tail.into_iter();
            let hidden = tail
                .next()
                .ok_or_else(|| Error::InferenceError("Audio8 slow step returned nothing".into()))?;
            if let Some(logits) = tail.next() {
                return (0..rows)
                    .map(|row| {
                        Ok(Audio8SlowOutput {
                            hidden_states: hidden.narrow(1, row, 1)?.copy()?,
                            logits: logits.narrow(0, row, 1)?.copy()?,
                        })
                    })
                    .collect::<Result<Vec<_>>>();
            }
            let mut offset = 0;
            let tails = counts
                .iter()
                .map(|&count| {
                    offset += count;
                    hidden.narrow(1, offset - 1, 1)
                })
                .collect::<candle_core::Result<Vec<_>>>()?;
            let packed_tails = Tensor::cat(&tails, 0)?;
            let logits = self.project_logits(&packed_tails)?;
            (0..counts.len())
                .map(|row| {
                    Ok(Audio8SlowOutput {
                        hidden_states: packed_tails.narrow(0, row, 1)?.copy()?,
                        logits: logits.narrow(0, row, 1)?.copy()?,
                    })
                })
                .collect::<Result<Vec<_>>>()
        })();
        batch.finish(caches, execution)
    }

    /// Share embedding lookups without padding independent prompts.
    pub(crate) fn embed_prompt_batch(
        &self,
        prompts: &[Audio8ConditioningPrompt],
    ) -> Result<Vec<Tensor>> {
        if prompts.is_empty() {
            return Err(Error::InvalidInput(
                "Audio8 embedding batch is empty".into(),
            ));
        }
        let mut combined = Audio8ConditioningPrompt {
            values: vec![Vec::new(); self.cfg.num_codebooks + 1],
            vq_mask: Vec::new(),
            prompt_length: 0,
        };
        for prompt in prompts {
            if prompt.values.len() != combined.values.len()
                || prompt.prompt_length == 0
                || prompt
                    .values
                    .iter()
                    .any(|row| row.len() != prompt.prompt_length)
            {
                return Err(Error::InvalidInput(
                    "Audio8 embedding batch contains malformed prompt".into(),
                ));
            }
            combined.prompt_length = combined
                .prompt_length
                .checked_add(prompt.prompt_length)
                .ok_or_else(|| {
                    Error::InvalidInput("Audio8 embedding batch size overflow".into())
                })?;
            for (dst, src) in combined.values.iter_mut().zip(&prompt.values) {
                dst.extend_from_slice(src);
            }
            combined.vq_mask.extend_from_slice(&prompt.vq_mask);
        }
        let packed = self.embed_prompt(&combined)?;
        let mut offset = 0;
        prompts
            .iter()
            .map(|prompt| {
                let row = packed.narrow(1, offset, prompt.prompt_length)?;
                offset += prompt.prompt_length;
                Ok(row)
            })
            .collect()
    }

    pub fn forward_prompt(
        &self,
        prompt: &Audio8ConditioningPrompt,
        cache: &mut PhysicalPagedKvCache,
        return_all: bool,
    ) -> Result<Audio8SlowOutput> {
        let x = self.embed_prompt(prompt)?;
        self.forward_embeds(&x, 0, cache, return_all)
    }

    pub fn semantic_allowed_mask(&self, im_end_token_id: u32) -> Result<Vec<bool>> {
        if let Some((head, eos)) = &self.semantic_head {
            if *eos != im_end_token_id {
                return Err(Error::ModelLoadError(
                    "Audio8 compact EOS changed after load".into(),
                ));
            }
            return Ok(vec![true; head.weight().dim(0)?]);
        }
        if self.cfg.semantic_end_token_id as usize >= self.cfg.vocab_size
            || im_end_token_id as usize >= self.cfg.vocab_size
        {
            return Err(Error::ModelLoadError(
                "Audio8 semantic tokens exceed vocabulary".into(),
            ));
        }
        let mut mask = vec![false; self.cfg.vocab_size];
        mask[self.cfg.semantic_start_token_id as usize..=self.cfg.semantic_end_token_id as usize]
            .fill(true);
        mask[im_end_token_id as usize] = true;
        Ok(mask)
    }
}

impl Audio8SlowLayer {
    /// The attention input of `hidden` `[1, tokens, hidden]`: each token's
    /// rotated query and key and its value (see
    /// [`Audio8PackedAttention::project_batch`]).
    fn pre_attention(
        &self,
        hidden: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        self.self_attn
            .project_batch(&self.input_layernorm.forward(hidden)?, cos, sin)
    }

    /// The residual branches after attention: `attn` `[1, tokens, q_size]`
    /// through the output projection onto `hidden`, then the MLP.
    fn post_attention(&self, hidden: &Tensor, attn: &Tensor) -> Result<Tensor> {
        let hidden = hidden.broadcast_add(&self.self_attn.o_proj.forward(attn)?)?;
        let ff = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&hidden)?)?;
        hidden.broadcast_add(&ff).map_err(Error::from)
    }

    fn load(cfg: &Audio8SlowConfig, rotary: &Audio8RotaryCache, vb: VarBuilder) -> Result<Self> {
        let input_layernorm = load_rms_norm_alias(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            &vb,
            &["input_layernorm", "attention_norm"],
        )?;
        let self_attn = Audio8PackedAttention::load(cfg, rotary, vb.pp("self_attn"))?;
        let post_attention_layernorm = load_rms_norm_alias(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            &vb,
            &["post_attention_layernorm", "ffn_norm"],
        )?;
        let mlp = Audio8Mlp::load(cfg, vb.pp("mlp"))?;
        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let attn = self
            .self_attn
            .forward(&normed, start_pos, cache, prepared, layer_idx)?;
        let x = x.broadcast_add(&attn)?;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp = self.mlp.forward(&normed)?;
        x.broadcast_add(&mlp).map_err(Error::from)
    }
}

impl Audio8PackedAttention {
    fn load(cfg: &Audio8SlowConfig, rotary: &Audio8RotaryCache, vb: VarBuilder) -> Result<Self> {
        let total = cfg.q_size() + 2 * cfg.kv_size();
        // The published slow transformer has a biased fused QKV projection
        // and no Q/K normalization. A checkpoint that carries Q/K norm
        // weights is another model: refuse it instead of ignoring them.
        if vb.contains_tensor("q_norm.weight") || vb.contains_tensor("k_norm.weight") {
            return Err(Error::ModelLoadError(
                "Audio8 slow attention does not use Q/K normalization weights".into(),
            ));
        }
        let qkv_proj = if cfg.attention_qkv_bias {
            candle_nn::linear(cfg.hidden_size, total, vb.pp("qkv_proj"))?
        } else {
            candle_nn::linear_no_bias(cfg.hidden_size, total, vb.pp("qkv_proj"))?
        };
        Ok(Self {
            qkv_proj,
            o_proj: candle_nn::linear_no_bias(cfg.q_size(), cfg.hidden_size, vb.pp("o_proj"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rotary: rotary.clone(),
        })
    }

    /// Every token of `x` `[1, tokens, hidden]` (spare rows included):
    /// its query and key rotated at its own position (`cos`/`sin`, one row
    /// a token) and its value, each `[tokens, heads, head_dim]`. Rows stay
    /// apart, so a token's bits are its own whatever rows share the call.
    fn project_batch(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let tokens = x.dim(1)?;
        let qsize = self.num_heads * self.head_dim;
        let kvsize = self.num_kv_heads * self.head_dim;
        let qkv = self.qkv_proj.forward(x)?;
        let q = qkv
            .narrow(2, 0, qsize)?
            .reshape((1, tokens, self.num_heads, self.head_dim))?;
        let k =
            qkv.narrow(2, qsize, kvsize)?
                .reshape((1, tokens, self.num_kv_heads, self.head_dim))?;
        let v = qkv
            .narrow(2, qsize + kvsize, kvsize)?
            .reshape((tokens, self.num_kv_heads, self.head_dim))?
            .contiguous()?;
        let q = self.rotary.apply_tables(&q, cos, sin)?.squeeze(0)?;
        let k = self.rotary.apply_tables(&k, cos, sin)?.squeeze(0)?;
        Ok((q, k, v))
    }

    /// Paged attention of the first `total` tokens of `q`/`k`/`v` (from
    /// [`Self::project_batch`]), whose rows after them are spare rows that
    /// attention skips: `[1, tokens, q_size]` with zero spare rows.
    #[allow(clippy::too_many_arguments)]
    fn attend_batch(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        total: usize,
        cache: &PhysicalPagedKvCache,
        batch: &mut super::batch::Audio8PhysicalBatch,
        layer: usize,
    ) -> Result<Tensor> {
        let width = q.dim(0)?;
        let real = |t: &Tensor| -> Result<Tensor> {
            let t = if width > total {
                t.narrow(0, 0, total)?
            } else {
                t.clone()
            };
            t.contiguous().map_err(Error::from)
        };
        let out = batch.attend(
            cache,
            layer,
            &real(q)?,
            &real(k)?,
            &real(v)?,
            1.0 / (self.head_dim as f32).sqrt(),
        )?;
        let out = out.reshape((1, total, self.num_heads * self.head_dim))?;
        if width > total {
            Ok(out.pad_with_zeros(1, 0, width - total)?)
        } else {
            Ok(out)
        }
    }

    fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let bsz = x.dim(0)?;
        let seq_len = x.dim(1)?;
        if bsz != 1 {
            return Err(Error::InvalidInput(
                "Audio8 slow physical paged attention expects one sequence".into(),
            ));
        }
        let q_size = self.num_heads * self.head_dim;
        let kv_size = self.num_kv_heads * self.head_dim;
        let qkv = self.qkv_proj.forward(x)?;
        let q = qkv
            .narrow(2, 0, q_size)?
            .reshape((bsz, seq_len, self.num_heads, self.head_dim))?;
        let k = qkv.narrow(2, q_size, kv_size)?.reshape((
            bsz,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ))?;
        let v = qkv.narrow(2, q_size + kv_size, kv_size)?.reshape((
            bsz,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ))?;

        let q = self.rotary.apply(&q, start_pos)?;
        let k = self.rotary.apply(&k, start_pos)?;
        let q = q.squeeze(0)?;
        let k = k.squeeze(0)?;
        let v = v.squeeze(0)?;
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let out = cache.write_and_attend(layer_idx, prepared, &q, &k, &v, scale)?;
        let out = out.reshape((bsz, seq_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out).map_err(Error::from)
    }
}

impl Audio8Mlp {
    fn load(cfg: &Audio8SlowConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("gate_proj"),
            )?,
            up_proj: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("up_proj"),
            )?,
            down_proj: candle_nn::linear_no_bias(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("down_proj"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let hidden = ops::silu(&gate)?.broadcast_mul(&up)?;
        self.down_proj.forward(&hidden).map_err(Error::from)
    }
}

fn load_rms_norm_alias(dim: usize, eps: f64, vb: &VarBuilder, aliases: &[&str]) -> Result<RmsNorm> {
    for alias in aliases {
        if vb.contains_tensor(&format!("{alias}.weight")) {
            return candle_nn::rms_norm(dim, eps, vb.pp(*alias)).map_err(Error::from);
        }
    }
    candle_nn::rms_norm(dim, eps, vb.pp(aliases[0])).map_err(Error::from)
}

#[cfg(test)]
pub(super) fn tiny_for_batch_tests() -> Audio8SlowTransformer {
    tests::tiny_model(&candle_core::Device::Cpu)
}

/// Rows a decode step's dense work runs at on CUDA.
pub(crate) const AUDIO8_DECODE_ROWS: usize = 32;

/// The fixed row count a step's dense work runs at, or `None` to run the
/// rows as they are. On CUDA a decode step (every row one token) of at most
/// [`AUDIO8_DECODE_ROWS`] rows pads to that count: cuBLAS chooses its kernel,
/// tiles and split by the row count, so a row's bits depended on how many
/// requests shared its step. Norms, residuals and activations are row-wise
/// and attention runs per row, so the padding changes no real row. Prefill
/// spans (one request alone, on its grid) and the CPU and Metal run as they
/// are.
fn decode_rows(device: &candle_core::Device, counts: &[usize]) -> Option<usize> {
    (device.is_cuda()
        && counts.len() <= AUDIO8_DECODE_ROWS
        && counts.iter().all(|&count| count == 1))
    .then_some(AUDIO8_DECODE_ROWS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Shape};
    use std::collections::HashMap;

    fn tiny_cfg() -> Audio8SlowConfig {
        Audio8SlowConfig {
            vocab_size: 32,
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            max_seq_len: 16,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            codebook_size: 8,
            num_codebooks: 2,
            semantic_start_token_id: 20,
            semantic_end_token_id: 27,
            attention_qkv_bias: true,
        }
    }

    fn tensor(device: &Device, shape: impl Into<Shape>, value: f32) -> Tensor {
        Tensor::full(value, shape, device).unwrap()
    }

    pub(super) fn tiny_model(device: &Device) -> Audio8SlowTransformer {
        let cfg = tiny_cfg();
        let mut tensors = HashMap::new();
        tensors.insert(
            "embed_tokens.weight".to_string(),
            tensor(device, (cfg.vocab_size, cfg.hidden_size), 0.01),
        );
        tensors.insert(
            "codebook_embeddings.weight".to_string(),
            tensor(
                device,
                (cfg.codebook_size * cfg.num_codebooks, cfg.hidden_size),
                0.02,
            ),
        );
        tensors.insert(
            "norm.weight".to_string(),
            tensor(device, (cfg.hidden_size,), 1.0),
        );
        tensors.insert(
            "layers.0.input_layernorm.weight".to_string(),
            tensor(device, (cfg.hidden_size,), 1.0),
        );
        tensors.insert(
            "layers.0.post_attention_layernorm.weight".to_string(),
            tensor(device, (cfg.hidden_size,), 1.0),
        );
        tensors.insert(
            "layers.0.self_attn.qkv_proj.weight".to_string(),
            tensor(
                device,
                (cfg.q_size() + 2 * cfg.kv_size(), cfg.hidden_size),
                0.01,
            ),
        );
        tensors.insert(
            "layers.0.self_attn.qkv_proj.bias".to_string(),
            tensor(device, (cfg.q_size() + 2 * cfg.kv_size(),), 0.005),
        );
        tensors.insert(
            "layers.0.self_attn.o_proj.weight".to_string(),
            tensor(device, (cfg.hidden_size, cfg.q_size()), 0.01),
        );
        tensors.insert(
            "layers.0.mlp.gate_proj.weight".to_string(),
            tensor(device, (cfg.intermediate_size, cfg.hidden_size), 0.01),
        );
        tensors.insert(
            "layers.0.mlp.up_proj.weight".to_string(),
            tensor(device, (cfg.intermediate_size, cfg.hidden_size), 0.01),
        );
        tensors.insert(
            "layers.0.mlp.down_proj.weight".to_string(),
            tensor(device, (cfg.hidden_size, cfg.intermediate_size), 0.01),
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
        Audio8SlowTransformer::load(cfg, vb).unwrap()
    }

    fn check_compact_head(device: &Device) {
        let mut model = tiny_model(device);
        let cfg = model.cfg.clone();
        let weights: Vec<f32> = (0..cfg.vocab_size * cfg.hidden_size)
            .map(|i| ((i as f32 + 0.3) * 0.17).sin())
            .collect();
        model.lm_head = Linear::new(
            Tensor::from_vec(weights, (cfg.vocab_size, cfg.hidden_size), device).unwrap(),
            None,
        );
        let input = Tensor::from_vec(
            vec![0.2f32, -0.7, 1.3, 0.4, -1.2, 0.3, 0.7, 1.1],
            (1, 2, 4),
            device,
        )
        .unwrap();
        let full = model.project_logits(&input).unwrap();
        let expected = Tensor::cat(
            &[
                &full.narrow(2, 1, 1).unwrap(),
                &full.narrow(2, 20, 8).unwrap(),
            ],
            2,
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        model.configure_semantic_head(1).unwrap();
        let compact = model.project_logits(&input).unwrap();
        assert_eq!(compact.dims(), &[1, 2, 9]);
        let actual = compact.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 2e-5, "{actual} != {expected}");
        }
        assert_eq!(model.token_id_from_logit(0).unwrap(), 1);
        assert_eq!(model.token_id_from_logit(1).unwrap(), 20);
        assert_eq!(model.token_id_from_logit(8).unwrap(), 27);
        assert!(model.token_id_from_logit(9).is_err());
        assert_eq!(model.semantic_allowed_mask(1).unwrap(), vec![true; 9]);
        assert_eq!(
            model.embeddings.embeddings().dims(),
            &[cfg.vocab_size, cfg.hidden_size]
        );
    }

    #[test]
    fn compact_head_matches_gathered_full_logits_and_preserves_token_ids() {
        check_compact_head(&Device::Cpu);
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires an available Metal device; never falls back to CPU"]
    fn metal_compact_head_matches_full_projection() {
        let (device, _lease) = crate::backends::metal_test_device_or_skip(
            "metal_compact_head_matches_full_projection",
        )
        .expect("Metal device");
        check_compact_head(&device);
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires an available CUDA device; never falls back to CPU"]
    fn cuda_compact_head_matches_full_projection() {
        check_compact_head(&Device::new_cuda(0).expect("CUDA device"));
    }

    #[test]
    fn embeds_prompt_with_codebooks_only_on_semantic_tokens() {
        let device = Device::Cpu;
        let model = tiny_model(&device);
        let prompt = Audio8ConditioningPrompt {
            values: vec![vec![1, 20, 21], vec![0, 3, 4], vec![0, 5, 6]],
            vq_mask: vec![false, true, true],
            prompt_length: 3,
        };
        let embeds = model.embed_prompt(&prompt).unwrap();
        assert_eq!(embeds.dims(), &[1, 3, 4]);
        let values = embeds.to_vec3::<f32>().unwrap();
        assert!(values[0][1][0] > values[0][0][0]);
    }

    #[test]
    fn packed_batch_matches_scalar_ragged_prefill_and_permuted_decode_above_five() {
        let model = tiny_model(&Device::Cpu);
        let cfg = model.config();
        let caches = || {
            super::super::physical::test_physical_caches(
                900,
                cfg.num_hidden_layers,
                cfg.num_key_value_heads,
                cfg.head_dim,
                cfg.max_seq_len,
                7,
            )
        };
        let mut batched = caches();
        let mut scalar = caches();
        let inputs = (0..7)
            .map(|row| {
                Tensor::from_vec(
                    (0..(row % 3 + 1) * cfg.hidden_size)
                        .map(|i| ((i * 3 + row * 7) as f32 * 0.2).sin())
                        .collect::<Vec<_>>(),
                    (1, row % 3 + 1, cfg.hidden_size),
                    &Device::Cpu,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let expected = inputs
            .iter()
            .zip(&mut scalar)
            .map(|(input, cache)| model.forward_embeds(input, 0, cache, false).unwrap())
            .collect::<Vec<_>>();
        let actual = model
            .forward_embeds_batch(&inputs, &mut batched.iter_mut().collect::<Vec<_>>())
            .unwrap();
        for (a, b) in actual.iter().zip(expected) {
            let a = a.logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let b = b.logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert!(a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-5));
        }
        for cache in &mut batched {
            assert_eq!(cache.take_completed_writes().len(), 1);
        }
        batched.reverse();
        scalar.reverse();
        let inputs = (0..7)
            .map(|row| {
                Tensor::from_vec(
                    vec![0.1 + row as f32, -0.4, 0.7, 0.2],
                    (1, 1, 4),
                    &Device::Cpu,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let expected = inputs
            .iter()
            .zip(&mut scalar)
            .map(|(input, cache)| {
                model
                    .forward_embeds(input, cache.context_len(), cache, false)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let actual = model
            .forward_embeds_batch(&inputs, &mut batched.iter_mut().collect::<Vec<_>>())
            .unwrap();
        for ((a, b), (batch, reference)) in
            actual.iter().zip(expected).zip(batched.iter().zip(&scalar))
        {
            assert_eq!(batch.context_len(), reference.context_len());
            let a = a
                .hidden_states
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let b = b
                .hidden_states
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert!(a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-5));
        }
    }

    /// On CUDA, decode steps through the stretch graphs (warm, capture,
    /// verify, then replay) give the eager steps' bits, and every row
    /// decodes the same alone as beside others (the dense work runs at the
    /// fixed 32 rows either way).
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_graphed_decode_steps_match_eager_and_alone_bit_for_bit() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        let graphed = tiny_model(&device);
        assert!(graphed.graphs.is_ok(), "{}", graphed.graphs_diagnostics());
        let mut eager = tiny_model(&device);
        eager.turn_off_graphs();
        let cfg = graphed.config().clone();
        let bits = |t: &Tensor| {
            t.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        };
        let input = |row: usize, step: usize, len: usize| {
            Tensor::from_vec(
                (0..len * cfg.hidden_size)
                    .map(|i| ((i * 5 + row * 11 + step * 3) as f32 * 0.17).sin())
                    .collect::<Vec<_>>(),
                (1, len, cfg.hidden_size),
                &device,
            )
            .unwrap()
        };
        // A model's rows decoded together or each alone, after prompts of
        // different lengths: every step's outputs as bits, row by row.
        let run = |model: &Audio8SlowTransformer, together: bool| {
            let rows = 3;
            let mut caches =
                crate::models::shared::attention::physical::test_physical_caches_with_dtype_on(
                    &device,
                    DType::F32,
                    cfg.num_hidden_layers,
                    cfg.num_key_value_heads,
                    cfg.head_dim,
                    cfg.max_seq_len,
                    rows,
                );
            for (row, cache) in caches.iter_mut().enumerate() {
                model
                    .forward_embeds_batch(&[input(row, 0, row + 1)], &mut [cache])
                    .unwrap();
            }
            let mut out = vec![Vec::new(); rows];
            for step in 1..8 {
                let inputs = (0..rows).map(|row| input(row, step, 1)).collect::<Vec<_>>();
                let outputs = if together {
                    model
                        .forward_embeds_batch(&inputs, &mut caches.iter_mut().collect::<Vec<_>>())
                        .unwrap()
                } else {
                    inputs
                        .iter()
                        .zip(caches.iter_mut())
                        .map(|(input, cache)| {
                            model
                                .forward_embeds_batch(std::slice::from_ref(input), &mut [cache])
                                .unwrap()
                                .remove(0)
                        })
                        .collect()
                };
                for (row, output) in outputs.iter().enumerate() {
                    out[row].push((bits(&output.hidden_states), bits(&output.logits)));
                }
            }
            out
        };
        let together = run(&graphed, true);
        let diagnostics = graphed.graphs_diagnostics();
        assert!(diagnostics["off"].is_null(), "{diagnostics}");
        assert!(
            diagnostics["replays"].as_u64().unwrap_or(0) > 0,
            "{diagnostics}"
        );
        assert_eq!(together, run(&eager, true));
        assert_eq!(together, run(&graphed, false));
        assert_eq!(together, run(&eager, false));
    }

    #[test]
    fn slow_transformer_forward_returns_logits_and_hidden_tail() {
        let device = Device::Cpu;
        let model = tiny_model(&device);
        let prompt = Audio8ConditioningPrompt {
            values: vec![vec![1, 20, 21], vec![0, 3, 4], vec![0, 5, 6]],
            vq_mask: vec![false, true, true],
            prompt_length: 3,
        };
        let cfg = model.config();
        let mut cache = super::super::physical::test_physical_cache(
            201,
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.max_seq_len,
        );
        let output = model.forward_prompt(&prompt, &mut cache, false).unwrap();
        assert_eq!(output.logits.dims(), &[1, 1, 32]);
        assert_eq!(output.hidden_states.dims(), &[1, 1, 4]);
        assert_eq!(cache.context_len(), 3);
    }
}
