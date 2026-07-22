//! Code Predictor for multi-codebook RVQ token generation.
//!
//! The code predictor generates the residual codebook tokens after the talker
//! has produced the first (semantic) codebook. It uses a smaller transformer
//! for efficient multi-token prediction.

use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{ops, Embedding, Linear, Module, RmsNorm, VarBuilder};

use crate::error::{Error, Result};
use crate::models::architectures::qwen3::tts::config::CodePredictorConfig;
use crate::models::architectures::qwen3::tts::rope::{
    build_rope_inv_freq, build_rope_window_full, qwen_rotate_half,
};
pub use crate::models::shared::attention::physical::PhysicalPagedKvCache as CodePredictorPhysicalCache;
use crate::models::shared::attention::physical::PreparedPhysicalPagedStep;
use crate::models::shared::weights::mlx;

/// The predictor starts each semantic frame from talker hidden state followed
/// by the selected semantic embedding.
pub const CODE_PREDICTOR_PHYSICAL_PREFILL_TOKENS: usize = 2;

/// Exact physical context occupied by one predictor frame.
pub const fn code_predictor_physical_context_tokens(acoustic_groups: usize) -> usize {
    CODE_PREDICTOR_PHYSICAL_PREFILL_TOKENS.saturating_add(acoustic_groups.saturating_sub(1))
}

/// Code Predictor model
pub struct CodePredictor {
    codec_embeddings: Vec<Embedding>,
    small_to_mtp_projection: Option<Linear>,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_heads: Vec<Linear>,
    device: Device,
    cfg: CodePredictorConfig,
    num_code_groups: usize,
}

impl CodePredictor {
    /// Load the code predictor from VarBuilder
    pub fn load(cfg: CodePredictorConfig, vb: VarBuilder, num_code_groups: usize) -> Result<Self> {
        // Use text_hidden_size for codec embeddings if specified, otherwise hidden_size
        let codec_embed_dim = cfg.text_hidden_size.unwrap_or(cfg.hidden_size);

        // Load codec embeddings (one per codebook, but weights only have 15)
        // The model has embeddings 0-14 (15 total), not 16
        let num_codec_embeddings = num_code_groups.min(15);
        let mut codec_embeddings = Vec::with_capacity(num_codec_embeddings);
        for idx in 0..num_codec_embeddings {
            let embed = mlx::load_embedding(
                cfg.vocab_size,
                codec_embed_dim,
                vb.pp(format!("model.codec_embedding.{idx}")),
            )?;
            codec_embeddings.push(embed);
        }

        let small_to_mtp_projection = if codec_embed_dim != cfg.hidden_size {
            Some(mlx::load_linear(
                codec_embed_dim,
                cfg.hidden_size,
                vb.pp("small_to_mtp_projection"),
            )?)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            let layer = Layer::load(&cfg, vb.pp(format!("model.layers.{idx}")))?;
            layers.push(layer);
        }

        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;

        // Load output heads (one per code group, but weights only have 15)
        let num_lm_heads = num_code_groups.min(15);
        let mut lm_heads = Vec::with_capacity(num_lm_heads);
        for idx in 0..num_lm_heads {
            let head = mlx::load_linear_no_bias(
                cfg.hidden_size,
                cfg.vocab_size,
                vb.pp(format!("lm_head.{idx}")),
            )?;
            lm_heads.push(head);
        }

        Ok(Self {
            codec_embeddings,
            small_to_mtp_projection,
            layers,
            norm,
            lm_heads,
            device: vb.device().clone(),
            cfg,
            num_code_groups,
        })
    }

    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get number of code groups
    pub fn num_code_groups(&self) -> usize {
        self.num_code_groups
    }

    /// Number of acoustic code groups predicted after the semantic codebook.
    pub fn num_acoustic_groups(&self) -> usize {
        self.codec_embeddings.len()
    }

    /// Exact physical KV capacity required by one predictor invocation.
    ///
    /// The two-token prefill produces the first acoustic code. Each remaining
    /// acoustic group appends one dependent token, so a standard 15-group
    /// predictor ends at cursor 16.
    pub fn physical_context_tokens_per_frame(&self) -> usize {
        code_predictor_physical_context_tokens(self.lm_heads.len())
    }

    /// Validate the fresh invocation workspace required for one semantic frame.
    pub fn validate_physical_workspace(&self, cache: &CodePredictorPhysicalCache) -> Result<()> {
        if self.lm_heads.len() != self.codec_embeddings.len() {
            return Err(Error::InferenceError(format!(
                "Qwen3-TTS physical predictor has {} heads for {} acoustic embeddings",
                self.lm_heads.len(),
                self.codec_embeddings.len()
            )));
        }
        if cache.context_len() != 0 {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS physical predictor requires a fresh cursor-0 workspace, got {}",
                cache.context_len()
            )));
        }
        cache.validate_model(
            self.cfg.num_hidden_layers,
            self.cfg.num_key_value_heads,
            self.cfg.head_dim(),
        )?;
        let required_tokens = self.physical_context_tokens_per_frame();
        if cache.capacity_tokens() < required_tokens {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS physical predictor workspace holds {} tokens, requires {required_tokens}",
                cache.capacity_tokens()
            )));
        }
        Ok(())
    }

    /// Forward a predictor input against scheduler-owned invocation pages.
    ///
    /// The supplied start position must be the workspace's exact cursor.
    pub fn forward_physical(
        &self,
        first_codebook: &Tensor,
        start_pos: usize,
        cache: &mut CodePredictorPhysicalCache,
    ) -> Result<Vec<Tensor>> {
        let mut x = self.codec_embeddings[0].forward(first_codebook)?;
        if let Some(proj) = &self.small_to_mtp_projection {
            x = proj.forward(&x)?;
        }

        let (x, prepared) = self.forward_physical_hidden_uncommitted(&x, start_pos, cache)?;
        let mut outputs = Vec::with_capacity(self.num_code_groups);
        for head in &self.lm_heads {
            outputs.push(head.forward(&x)?);
        }
        cache.commit_prepared(prepared)?;
        Ok(outputs)
    }

    /// Generate one frame's acoustic groups using a fresh physical workspace.
    ///
    /// The workspace is invocation-local: callers must provide cursor 0 for
    /// every semantic frame and discard it on error. Successful generation
    /// advances the cursor from 0 to [`Self::physical_context_tokens_per_frame`].
    pub fn generate_acoustic_codes_physical(
        &self,
        talker_hidden: &Tensor,
        semantic_embed: &Tensor,
        cache: &mut CodePredictorPhysicalCache,
    ) -> Result<Vec<u32>> {
        self.validate_physical_workspace(cache)?;
        let required_tokens = self.physical_context_tokens_per_frame();

        let (talker_batch, talker_tokens, talker_dim) = talker_hidden.dims3()?;
        let (semantic_batch, semantic_tokens, semantic_dim) = semantic_embed.dims3()?;
        if talker_batch != 1
            || talker_tokens != 1
            || semantic_batch != 1
            || semantic_tokens != 1
            || talker_dim != semantic_dim
        {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS physical predictor expects matching [1,1,hidden] inputs, got {:?} and {:?}",
                talker_hidden.dims(),
                semantic_embed.dims()
            )));
        }

        let input = Tensor::cat(&[talker_hidden, semantic_embed], 1)?;
        let mut hidden = if let Some(proj) = &self.small_to_mtp_projection {
            proj.forward(&input)?
        } else {
            input
        };
        let prefill_tokens = hidden.dim(1)?;
        if prefill_tokens != CODE_PREDICTOR_PHYSICAL_PREFILL_TOKENS {
            return Err(Error::InferenceError(format!(
                "Qwen3-TTS physical predictor formed {prefill_tokens} prefill tokens, expected {CODE_PREDICTOR_PHYSICAL_PREFILL_TOKENS}"
            )));
        }
        let (next_hidden, prefill) = self.forward_physical_hidden_uncommitted(&hidden, 0, cache)?;
        hidden = next_hidden;

        let last_hidden = hidden.i((.., prefill_tokens - 1..prefill_tokens, ..))?;
        let num_acoustic = self.lm_heads.len();
        if num_acoustic == 0 {
            cache.commit_prepared(prefill)?;
            return Ok(Vec::new());
        }

        let first_logits = self.lm_heads[0].forward(&last_hidden)?;
        let mut prev_code = argmax_token(&first_logits.i((0, 0))?)?;
        cache.commit_prepared(prefill)?;

        let mut all_codes = Vec::with_capacity(num_acoustic);
        all_codes.push(prev_code);
        for group_idx in 1..num_acoustic {
            let mut step_hidden = self
                .codec_embedding_row(group_idx - 1, prev_code)?
                .unsqueeze(0)?;
            if let Some(proj) = &self.small_to_mtp_projection {
                step_hidden = proj.forward(&step_hidden)?;
            }

            let step_start = cache.context_len();
            let (next_hidden, step_prepared) =
                self.forward_physical_hidden_uncommitted(&step_hidden, step_start, cache)?;
            step_hidden = next_hidden;
            let logits = self.lm_heads[group_idx].forward(&step_hidden)?;
            prev_code = argmax_token(&logits.i((0, 0))?)?;
            cache.commit_prepared(step_prepared)?;
            all_codes.push(prev_code);
        }

        if cache.context_len() != required_tokens {
            return Err(Error::InferenceError(format!(
                "Qwen3-TTS physical predictor ended at cursor {}, expected {required_tokens}",
                cache.context_len()
            )));
        }
        Ok(all_codes)
    }

    fn forward_physical_hidden_uncommitted(
        &self,
        x: &Tensor,
        start_pos: usize,
        cache: &CodePredictorPhysicalCache,
    ) -> Result<(Tensor, PreparedPhysicalPagedStep)> {
        let (batch_size, sequence_len, hidden_size) = x.dims3()?;
        if batch_size != 1 || sequence_len == 0 || hidden_size != self.cfg.hidden_size {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS physical predictor expects [1,sequence,{}], got {:?}",
                self.cfg.hidden_size,
                x.dims()
            )));
        }
        if start_pos != cache.context_len() {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS physical predictor starts at {start_pos}, expected invocation cursor {}",
                cache.context_len()
            )));
        }
        cache.validate_model(
            self.cfg.num_hidden_layers,
            self.cfg.num_key_value_heads,
            self.cfg.head_dim(),
        )?;
        let mut prepared = cache.prepare_append(start_pos, sequence_len)?;

        let mut hidden = x.clone();
        for (idx, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward_physical(&hidden, start_pos, cache, &mut prepared, idx)?;
        }
        let hidden = self.norm.forward(&hidden)?;
        Ok((hidden, prepared))
    }

    /// Sum acoustic embeddings for the 15 generated acoustic codes.
    /// Returned tensor shape is [1, 1, codec_embed_dim].
    pub fn get_acoustic_embeddings_sum(&self, acoustic_codes: &[u32]) -> Result<Tensor> {
        if acoustic_codes.len() != self.codec_embeddings.len() {
            return Err(Error::InvalidInput(format!(
                "Expected {} acoustic codes, got {}",
                self.codec_embeddings.len(),
                acoustic_codes.len()
            )));
        }

        let mut sum = self
            .codec_embedding_row(0, acoustic_codes[0])?
            .unsqueeze(0)?;

        for (group_idx, code) in acoustic_codes.iter().enumerate().skip(1) {
            let embed = self.codec_embedding_row(group_idx, *code)?.unsqueeze(0)?;
            sum = sum.broadcast_add(&embed)?;
        }

        Ok(sum)
    }

    fn codec_embedding_row(&self, group_idx: usize, code: u32) -> Result<Tensor> {
        if self.device.is_cuda() {
            self.codec_embeddings[group_idx]
                .embeddings()
                .i(code as usize)?
                .unsqueeze(0)
                .map_err(Error::from)
        } else {
            let code_tensor = Tensor::from_vec(vec![code], (1,), &self.device)?;
            self.codec_embeddings[group_idx]
                .forward(&code_tensor)
                .map_err(Error::from)
        }
    }
}

/// Transformer layer for code predictor
struct Layer {
    input_layernorm: RmsNorm,
    self_attn: Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

impl Layer {
    fn load(cfg: &CodePredictorConfig, vb: VarBuilder) -> Result<Self> {
        let input_layernorm =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let self_attn = Attention::load(cfg, vb.pp("self_attn"))?;
        let post_attention_layernorm = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let mlp = Mlp::load(cfg, vb.pp("mlp"))?;

        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        })
    }

    fn forward_physical(
        &self,
        x: &Tensor,
        start_pos: usize,
        cache: &CodePredictorPhysicalCache,
        prepared: &mut PreparedPhysicalPagedStep,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self
            .self_attn
            .forward_physical(&normed, start_pos, cache, prepared, layer_idx)?;
        let x = x.broadcast_add(&attn_out)?;

        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        x.broadcast_add(&mlp_out).map_err(Error::from)
    }
}

/// Multi-head attention for code predictor
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_inv_freq: Vec<f32>,
}

impl Attention {
    fn load(cfg: &CodePredictorConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.head_dim();

        let q_proj = mlx::load_linear_no_bias(
            cfg.hidden_size,
            cfg.num_attention_heads * head_dim,
            vb.pp("q_proj"),
        )?;
        let k_proj = mlx::load_linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            vb.pp("k_proj"),
        )?;
        let v_proj = mlx::load_linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            vb.pp("v_proj"),
        )?;
        let o_proj = mlx::load_linear_no_bias(
            cfg.num_attention_heads * head_dim,
            cfg.hidden_size,
            vb.pp("o_proj"),
        )?;
        let q_norm = candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            rope_inv_freq: build_rope_inv_freq(head_dim, cfg.rope_theta),
        })
    }

    fn apply_qk_norm(
        &self,
        x: Tensor,
        heads: usize,
        seq_len: usize,
        norm: &RmsNorm,
    ) -> Result<Tensor> {
        let bsz = x.dim(0)?;
        let reshaped = x.reshape((bsz * seq_len * heads, self.head_dim))?;
        let normed = norm.forward(&reshaped)?;
        normed
            .reshape((bsz, seq_len, heads, self.head_dim))
            .map_err(Error::from)
    }

    fn apply_rope(&self, x: Tensor, start_pos: usize) -> Result<Tensor> {
        let seq_len = x.dim(1)?;
        let half_dim = self.head_dim / 2;

        let (cos, sin) = build_rope_window_full(
            seq_len,
            start_pos,
            &self.rope_inv_freq,
            x.device(),
            x.dtype(),
        )?;

        let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

        let rotated = qwen_rotate_half(&x, half_dim)?;

        let out = x.broadcast_mul(&cos)?;
        out.broadcast_add(&rotated.broadcast_mul(&sin)?)
            .map_err(Error::from)
    }

    /// Direct grouped-query attention over scheduler-owned predictor pages.
    fn forward_physical(
        &self,
        x: &Tensor,
        start_pos: usize,
        cache: &CodePredictorPhysicalCache,
        prepared: &mut PreparedPhysicalPagedStep,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (bsz, seq_len, _) = x.dims3()?;
        if bsz != 1 || seq_len == 0 {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS physical predictor attention expects [1,sequence,hidden], got {:?}",
                x.dims()
            )));
        }

        let mut q =
            self.q_proj
                .forward(x)?
                .reshape((bsz, seq_len, self.num_heads, self.head_dim))?;
        let mut k =
            self.k_proj
                .forward(x)?
                .reshape((bsz, seq_len, self.num_kv_heads, self.head_dim))?;
        let v =
            self.v_proj
                .forward(x)?
                .reshape((bsz, seq_len, self.num_kv_heads, self.head_dim))?;

        q = self.apply_qk_norm(q, self.num_heads, seq_len, &self.q_norm)?;
        k = self.apply_qk_norm(k, self.num_kv_heads, seq_len, &self.k_norm)?;
        q = self.apply_rope(q, start_pos)?;
        k = self.apply_rope(k, start_pos)?;

        let q = q
            .reshape((seq_len, self.num_heads, self.head_dim))?
            .contiguous()?;
        let k = k
            .reshape((seq_len, self.num_kv_heads, self.head_dim))?
            .contiguous()?;
        let v = v
            .reshape((seq_len, self.num_kv_heads, self.head_dim))?
            .contiguous()?;
        let out = cache.write_and_attend(
            layer_idx,
            prepared,
            &q,
            &k,
            &v,
            1.0 / (self.head_dim as f32).sqrt(),
        )?;
        let out = out.reshape((bsz, seq_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out).map_err(Error::from)
    }
}

/// SwiGLU MLP
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn load(cfg: &CodePredictorConfig, vb: VarBuilder) -> Result<Self> {
        let gate_proj =
            mlx::load_linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?;
        let up_proj =
            mlx::load_linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?;
        let down_proj =
            mlx::load_linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let act = ops::silu(&gate)?;
        let hidden = act.broadcast_mul(&up)?;
        self.down_proj.forward(&hidden).map_err(Error::from)
    }
}

fn argmax_token(logits: &Tensor) -> Result<u32> {
    if !logits.device().is_cuda() {
        return argmax_token_reference(logits);
    }

    let idx = logits.argmax(D::Minus1)?;
    let idx = if idx.rank() == 0 {
        idx
    } else {
        idx.squeeze(0)?
    };
    idx.to_dtype(DType::U32)?
        .to_scalar::<u32>()
        .map_err(Error::from)
}

fn argmax_token_reference(logits: &Tensor) -> Result<u32> {
    let logits = logits.to_dtype(DType::F32)?;
    let logits = match logits.rank() {
        1 => logits,
        2 => {
            let (rows, _cols) = logits.dims2()?;
            if rows != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected Qwen3-TTS predictor logits shape: {:?}",
                    logits.shape().dims()
                )));
            }
            logits.i(0)?
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected Qwen3-TTS predictor logits rank: {rank}"
            )))
        }
    };
    let values = logits.to_vec1::<f32>()?;
    let mut max_idx = 0usize;
    let mut max_val = f32::NEG_INFINITY;
    for (idx, &val) in values.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = idx;
        }
    }
    Ok(max_idx as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_cuda_predictor_argmax_uses_reference_ordering() {
        let logits = Tensor::new(vec![0.0f32, 4.0, 4.0, 3.0], &Device::Cpu).unwrap();

        assert_eq!(argmax_token(&logits).unwrap(), 1);
    }

    #[test]
    fn predictor_argmax_reference_accepts_single_row_logits() {
        let logits = Tensor::new(&[[0.0f32, 1.0, 7.0, 3.0]], &Device::Cpu).unwrap();

        assert_eq!(argmax_token(&logits).unwrap(), 2);
    }

    #[test]
    fn physical_predictor_cursor_matches_prefill_and_dependent_groups() {
        assert_eq!(code_predictor_physical_context_tokens(0), 2);
        assert_eq!(code_predictor_physical_context_tokens(1), 2);
        assert_eq!(code_predictor_physical_context_tokens(15), 16);
    }
}
