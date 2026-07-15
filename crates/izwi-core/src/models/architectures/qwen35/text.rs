use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{ops, rotary_emb, Embedding};
use candle_transformers::models::with_tracing::QMatMul;
use candle_transformers::quantized_nn::RmsNorm;

use crate::error::{Error, Result};
use crate::kernels::{
    try_fused_gated_delta_recurrent, try_fused_gated_rms_norm, try_fused_l2_norm,
    try_fused_silu_mul, try_qwen35_causal_conv_sequence, try_tiled_deltanet_recurrence,
    use_block_fusion_for_device,
};
use crate::models::architectures::qwen3::core::repeat_kv;
use crate::models::shared::attention::flash::{
    try_fused_self_attention, try_fused_self_attention_preserve_dtype,
};
use crate::models::shared::attention::paged::{
    append_to_pages, default_kv_page_size, default_kv_quantization, materialize_pages,
    paged_decode_attention, KvCacheQuantization, KvPage,
};
use crate::models::shared::memory::accounting::{compact_tensor_storage, TensorStorageAccounting};
use crate::models::shared::telemetry::{
    record_decode_attention_path, record_prefill_sequence_span, record_rope_kernel,
    record_rope_manual, DecodeAttentionPath,
};
use crate::models::shared::weights::gguf::GgufLoader;

use super::chat::Qwen35TextConfig;

pub struct Qwen35TextModel {
    device: Device,
    token_embeddings: Embedding,
    layers: Vec<Qwen35Layer>,
    linear_triplet_starts: Vec<bool>,
    output_norm: RmsNorm,
    output: QMatMul,
    block_fusion_decode_enabled: bool,
    block_fusion_prefill_enabled: bool,
    block_fusion_prefill_min_tokens: usize,
    finite_diagnostics_enabled: bool,
}

pub struct Qwen35TextRuntimeState {
    layers: Vec<Qwen35LayerRuntimeState>,
}

impl Qwen35TextRuntimeState {
    /// Backing allocations retained by the per-request text runtime state.
    ///
    /// This intentionally excludes model-global caches (notably full-attention
    /// RoPE windows), so callers requiring a complete scheduler claim must keep
    /// Qwen3.5 fail-closed until those caches are independently bounded.
    pub fn allocated_session_bytes(&self) -> Option<u64> {
        let mut accounting = TensorStorageAccounting::default();
        self.account_storage(&mut accounting)?;
        Some(accounting.bytes())
    }

    pub(crate) fn account_storage(&self, accounting: &mut TensorStorageAccounting) -> Option<()> {
        for layer in &self.layers {
            match layer {
                Qwen35LayerRuntimeState::Linear {
                    conv_state,
                    recurrent_state,
                } => {
                    if let Some(conv_state) = conv_state {
                        for slot in &conv_state.slots {
                            accounting.add_tensor(slot)?;
                        }
                    }
                    if let Some(recurrent_state) = recurrent_state {
                        accounting.add_tensor(recurrent_state)?;
                    }
                }
                Qwen35LayerRuntimeState::Full {
                    k_pages,
                    v_pages,
                    dense_k_cache_h,
                    dense_v_cache_h,
                    ..
                } => {
                    for page in k_pages.iter().chain(v_pages.iter()) {
                        page.account_storage(accounting)?;
                    }
                    if let Some(cache) = dense_k_cache_h {
                        accounting.add_tensor(cache)?;
                    }
                    if let Some(cache) = dense_v_cache_h {
                        accounting.add_tensor(cache)?;
                    }
                }
            }
        }
        Some(())
    }
}

struct ConvRingState {
    slots: Vec<Tensor>,
    next_idx: usize,
}

impl ConvRingState {
    /// Move logical history slots into one exact-size backing allocation.
    ///
    /// Sequence-prefill slots are views into the entire projected token span.
    /// Keeping those views in runtime state would retain the full projection
    /// instead of the fixed `kernel_size - 1` history required by the conv.
    fn compact_owned(&mut self) -> Result<()> {
        if self.slots.is_empty() {
            return Ok(());
        }

        let slot_refs: Vec<&Tensor> = self.slots.iter().collect();
        let packed = compact_tensor_storage(&Tensor::cat(&slot_refs, 1)?)?;
        let mut slots = Vec::with_capacity(self.slots.len());
        for idx in 0..self.slots.len() {
            slots.push(packed.narrow(1, idx, 1)?);
        }
        self.slots = slots;
        Ok(())
    }
}

enum Qwen35LayerRuntimeState {
    Linear {
        conv_state: Option<ConvRingState>,
        recurrent_state: Option<Tensor>,
    },
    Full {
        k_pages: Vec<KvPage>,
        v_pages: Vec<KvPage>,
        dense_k_cache_h: Option<Tensor>,
        dense_v_cache_h: Option<Tensor>,
        dense_kv_tokens: usize,
    },
}

fn full_attention_state_has_cached_prefix(state: &Qwen35LayerRuntimeState) -> Result<bool> {
    match state {
        Qwen35LayerRuntimeState::Full {
            k_pages,
            v_pages,
            dense_k_cache_h,
            dense_v_cache_h,
            dense_kv_tokens,
        } => Ok(!k_pages.is_empty()
            || !v_pages.is_empty()
            || dense_k_cache_h.is_some()
            || dense_v_cache_h.is_some()
            || *dense_kv_tokens > 0),
        _ => Err(Error::InferenceError(
            "Qwen3.5 layer runtime state does not match full-attention layer".to_string(),
        )),
    }
}

struct Qwen35Layer {
    attn_norm: RmsNorm,
    mixer: Qwen35Mixer,
    post_attention_norm: RmsNorm,
    mlp: Qwen35Mlp,
}

enum Qwen35Mixer {
    Linear(Qwen35LinearAttention),
    Full(Qwen35FullAttention),
}

struct Qwen35Mlp {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
}

struct Qwen35FullAttention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    rope_theta: f64,
    mrope_sections: Vec<usize>,
    kv_page_size: usize,
    kv_quantization: KvCacheQuantization,
    dense_decode_attention_enabled: bool,
    dense_decode_max_pages: usize,
    dense_decode_max_tokens: usize,
    rope_kernel_enabled: bool,
    rope_inv_freqs: Vec<f32>,
}

struct Qwen35LinearAttention {
    qkv_proj: QMatMul,
    gate_proj: QMatMul,
    beta_proj: QMatMul,
    alpha_proj: QMatMul,
    dt_bias: Tensor,
    a: Tensor,
    conv_kernel: Tensor,
    conv_kernel_slices: Vec<Tensor>,
    norm: Qwen35GatedRmsNorm,
    out_proj: QMatMul,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    conv_dim: usize,
    kernel_size: usize,
    tiled_recurrence_enabled: bool,
    tiled_recurrence_tile_size_override: Option<usize>,
}

struct Qwen35GatedRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl Qwen35TextModel {
    pub fn load(loader: &GgufLoader, cfg: &Qwen35TextConfig, device: &Device) -> Result<Self> {
        if cfg.attention_key_length != cfg.attention_value_length {
            return Err(Error::ModelLoadError(format!(
                "Qwen3.5 full attention currently requires key/value head dims to match, found {} and {}",
                cfg.attention_key_length, cfg.attention_value_length
            )));
        }
        if cfg.ssm_time_step_rank == 0 || cfg.ssm_inner_size % cfg.ssm_time_step_rank != 0 {
            return Err(Error::ModelLoadError(format!(
                "Invalid Qwen3.5 linear attention dims: inner_size={}, time_step_rank={}",
                cfg.ssm_inner_size, cfg.ssm_time_step_rank
            )));
        }

        let embedding_weights = loader
            .load_qtensor("token_embd.weight", device)?
            .dequantize(device)
            .map_err(Error::from)?;
        let (vocab_size, hidden_size) = embedding_weights.dims2()?;
        if hidden_size != cfg.embedding_length {
            return Err(Error::ModelLoadError(format!(
                "Qwen3.5 token embedding width mismatch: GGUF has {hidden_size}, metadata says {}",
                cfg.embedding_length
            )));
        }
        let _ = vocab_size;

        let token_embeddings = Embedding::new(embedding_weights, hidden_size);
        let output_norm = load_rms_norm(loader, device, "output_norm.weight", cfg)?;
        let output = if loader.has_tensor("output.weight") {
            load_qmatmul(loader, device, "output.weight")?
        } else {
            load_qmatmul(loader, device, "token_embd.weight")?
        };
        let block_fusion_decode_enabled = qwen35_block_fusion_decode_enabled(device);
        let (block_fusion_prefill_enabled, block_fusion_prefill_min_tokens) =
            qwen35_block_fusion_prefill_policy(device);
        let finite_diagnostics_enabled = qwen35_env_bool("IZWI_QWEN35_FINITE_DIAGNOSTICS", false);

        let mut layers = Vec::with_capacity(cfg.block_count);
        for layer_idx in 0..cfg.block_count {
            let prefix = format!("blk.{layer_idx}");
            let attn_norm =
                load_rms_norm(loader, device, &format!("{prefix}.attn_norm.weight"), cfg)?;
            let post_attention_norm = load_rms_norm(
                loader,
                device,
                &format!("{prefix}.post_attention_norm.weight"),
                cfg,
            )?;
            let mlp = Qwen35Mlp::load(loader, device, &prefix)?;
            let mixer = if is_full_attention_layer(layer_idx, cfg.full_attention_interval) {
                Qwen35Mixer::Full(Qwen35FullAttention::load(loader, device, &prefix, cfg)?)
            } else {
                Qwen35Mixer::Linear(Qwen35LinearAttention::load(loader, device, &prefix, cfg)?)
            };

            layers.push(Qwen35Layer {
                attn_norm,
                mixer,
                post_attention_norm,
                mlp,
            });
        }
        let linear_triplet_starts = (0..layers.len())
            .map(|layer_idx| {
                layer_idx + 2 < layers.len()
                    && layers[layer_idx].is_linear()
                    && layers[layer_idx + 1].is_linear()
                    && layers[layer_idx + 2].is_linear()
            })
            .collect();

        Ok(Self {
            device: device.clone(),
            token_embeddings,
            layers,
            linear_triplet_starts,
            output_norm,
            output,
            block_fusion_decode_enabled,
            block_fusion_prefill_enabled,
            block_fusion_prefill_min_tokens,
            finite_diagnostics_enabled,
        })
    }

    pub fn new_state(&self) -> Qwen35TextRuntimeState {
        Qwen35TextRuntimeState {
            layers: self.layers.iter().map(Qwen35Layer::new_state).collect(),
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.token_embeddings.hidden_size()
    }

    pub fn forward_token_id_at(
        &self,
        token_id: u32,
        position_ids: [usize; 3],
        state: &mut Qwen35TextRuntimeState,
    ) -> Result<Tensor> {
        let input = Tensor::from_vec(vec![token_id], (1, 1), &self.device)?;
        let hidden = self.token_embeddings.forward(&input)?;
        self.forward_input_embedding_at(&hidden, position_ids, state)
    }

    pub fn forward_token_id_hidden_at(
        &self,
        token_id: u32,
        position_ids: [usize; 3],
        state: &mut Qwen35TextRuntimeState,
    ) -> Result<Tensor> {
        let input = Tensor::from_vec(vec![token_id], (1, 1), &self.device)?;
        let hidden = self.token_embeddings.forward(&input)?;
        self.forward_input_embedding_hidden_at(&hidden, position_ids, state)
    }

    pub fn forward_input_embedding_at(
        &self,
        input_embedding: &Tensor,
        position_ids: [usize; 3],
        state: &mut Qwen35TextRuntimeState,
    ) -> Result<Tensor> {
        let hidden =
            self.forward_input_embedding_hidden_at(input_embedding, position_ids, state)?;
        self.forward_hidden_to_logits(&hidden)
    }

    pub fn forward_input_embedding_hidden_at(
        &self,
        input_embedding: &Tensor,
        position_ids: [usize; 3],
        state: &mut Qwen35TextRuntimeState,
    ) -> Result<Tensor> {
        self.validate_runtime_state(state)?;
        let mut hidden = input_embedding.clone();
        let mut layer_idx = 0usize;
        while layer_idx < self.layers.len() {
            if self.block_fusion_decode_enabled && self.linear_triplet_starts[layer_idx] {
                for offset in 0..3 {
                    let idx = layer_idx + offset;
                    let layer_state = &mut state.layers[idx];
                    hidden = self.layers[idx].forward(&hidden, layer_state, position_ids)?;
                    validate_qwen35_finite_tensor(
                        &hidden,
                        idx,
                        self.layers[idx].decode_diagnostic_path(),
                        self.finite_diagnostics_enabled,
                    )?;
                }
                layer_idx += 3;
            } else {
                let layer_state = &mut state.layers[layer_idx];
                hidden = self.layers[layer_idx].forward(&hidden, layer_state, position_ids)?;
                validate_qwen35_finite_tensor(
                    &hidden,
                    layer_idx,
                    self.layers[layer_idx].decode_diagnostic_path(),
                    self.finite_diagnostics_enabled,
                )?;
                layer_idx += 1;
            }
        }
        Ok(hidden)
    }

    pub fn prefill_token_ids(
        &self,
        token_ids: &[u32],
        position_ids: &[[usize; 3]],
        state: &mut Qwen35TextRuntimeState,
        compute_logits: bool,
    ) -> Result<Option<Tensor>> {
        if token_ids.is_empty() {
            return Ok(None);
        }
        if token_ids.len() != position_ids.len() {
            return Err(Error::InvalidInput(format!(
                "Qwen3.5 prefill span mismatch: {} token ids for {} position ids",
                token_ids.len(),
                position_ids.len()
            )));
        }
        self.validate_runtime_state(state)?;
        record_prefill_sequence_span(token_ids.len());

        // Pre-initialize all lazy layer states before the hot loop to avoid
        // allocation costs inside the per-token iteration. This moves all
        // Tensor::zeros calls out of the N×L inner loop.
        for (layer, layer_state) in self.layers.iter().zip(state.layers.iter_mut()) {
            layer.ensure_state_initialized(layer_state, &self.device)?;
        }

        // Batch embedding lookup: single forward pass for all tokens.
        let input = Tensor::from_vec(token_ids.to_vec(), (1, token_ids.len()), &self.device)?;
        let mut hidden = self.token_embeddings.forward(&input)?;
        let mut layer_idx = 0usize;
        let can_fuse_prefill = self.block_fusion_prefill_enabled
            && token_ids.len() >= self.block_fusion_prefill_min_tokens;
        while layer_idx < self.layers.len() {
            if can_fuse_prefill && self.linear_triplet_starts[layer_idx] {
                for offset in 0..3 {
                    let idx = layer_idx + offset;
                    let layer_state = &mut state.layers[idx];
                    hidden =
                        self.layers[idx].forward_sequence(&hidden, layer_state, position_ids)?;
                    validate_qwen35_finite_tensor(
                        &hidden,
                        idx,
                        self.layers[idx].prefill_diagnostic_path(),
                        self.finite_diagnostics_enabled,
                    )?;
                }
                layer_idx += 3;
            } else {
                let layer_state = &mut state.layers[layer_idx];
                hidden =
                    self.layers[layer_idx].forward_sequence(&hidden, layer_state, position_ids)?;
                validate_qwen35_finite_tensor(
                    &hidden,
                    layer_idx,
                    self.layers[layer_idx].prefill_diagnostic_path(),
                    self.finite_diagnostics_enabled,
                )?;
                layer_idx += 1;
            }
        }

        if !compute_logits {
            return Ok(None);
        }

        let token_count = token_ids.len();
        let last_hidden = hidden.narrow(1, token_count - 1, 1).map_err(Error::from)?;
        self.forward_hidden_to_logits(&last_hidden).map(Some)
    }

    pub fn forward_hidden_to_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let hidden = self.output_norm.forward(hidden)?;
        validate_qwen35_finite_tensor(
            &hidden,
            self.layers.len(),
            "output.norm",
            self.finite_diagnostics_enabled,
        )?;
        let logits = self.output.forward(&hidden)?;
        let logits = logits.i((0, 0))?;
        validate_qwen35_finite_tensor(
            &logits,
            self.layers.len(),
            "output.logits",
            self.finite_diagnostics_enabled,
        )?;
        Ok(logits)
    }

    fn validate_runtime_state(&self, state: &Qwen35TextRuntimeState) -> Result<()> {
        if state.layers.len() != self.layers.len() {
            return Err(Error::InferenceError(format!(
                "Qwen3.5 runtime state layer mismatch: state has {}, model has {}",
                state.layers.len(),
                self.layers.len()
            )));
        }
        Ok(())
    }
}

impl Qwen35Layer {
    fn is_linear(&self) -> bool {
        matches!(self.mixer, Qwen35Mixer::Linear(_))
    }

    fn decode_diagnostic_path(&self) -> &'static str {
        match self.mixer {
            Qwen35Mixer::Linear(_) => "decode.linear_layer_output",
            Qwen35Mixer::Full(_) => "decode.full_attention_layer_output",
        }
    }

    fn prefill_diagnostic_path(&self) -> &'static str {
        match self.mixer {
            Qwen35Mixer::Linear(_) => "prefill.linear_layer_output",
            Qwen35Mixer::Full(_) => "prefill.full_attention_layer_output",
        }
    }

    fn new_state(&self) -> Qwen35LayerRuntimeState {
        match self.mixer {
            Qwen35Mixer::Linear(_) => Qwen35LayerRuntimeState::Linear {
                conv_state: None,
                recurrent_state: None,
            },
            Qwen35Mixer::Full(_) => Qwen35LayerRuntimeState::Full {
                k_pages: Vec::new(),
                v_pages: Vec::new(),
                dense_k_cache_h: None,
                dense_v_cache_h: None,
                dense_kv_tokens: 0,
            },
        }
    }

    /// Pre-initialize lazy state tensors so the first-use allocation cost
    /// does not happen inside the per-token hot loop during prefill.
    fn ensure_state_initialized(
        &self,
        state: &mut Qwen35LayerRuntimeState,
        device: &Device,
    ) -> Result<()> {
        match (&self.mixer, state) {
            (
                Qwen35Mixer::Linear(mixer),
                Qwen35LayerRuntimeState::Linear {
                    conv_state,
                    recurrent_state,
                },
            ) => {
                if conv_state.is_none() && mixer.kernel_size > 1 {
                    // Persistent runtime state cannot outlive a released scratch-pool
                    // lease. Keep the fixed history in one exact-size allocation.
                    let history_len = mixer.kernel_size - 1;
                    let zero =
                        owned_zero_tensor(&[mixer.conv_dim, history_len], DType::F32, device)?;
                    let mut slots = Vec::with_capacity(history_len);
                    for idx in 0..history_len {
                        slots.push(zero.narrow(1, idx, 1)?);
                    }
                    *conv_state = Some(ConvRingState { slots, next_idx: 0 });
                }
                if recurrent_state.is_none() {
                    *recurrent_state = Some(owned_zero_tensor(
                        &[1, mixer.num_v_heads, mixer.head_k_dim, mixer.head_v_dim],
                        DType::F32,
                        device,
                    )?);
                }
            }
            _ => {} // Full attention layers don't need pre-init
        }
        Ok(())
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        state: &mut Qwen35LayerRuntimeState,
        position_ids: [usize; 3],
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let hidden_states = self.attn_norm.forward(hidden_states)?;
        let hidden_states = match &self.mixer {
            Qwen35Mixer::Linear(mixer) => mixer.forward(&hidden_states, state)?,
            Qwen35Mixer::Full(mixer) => mixer.forward(&hidden_states, state, position_ids)?,
        };
        let hidden_states = (&residual + &hidden_states)?;

        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_norm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        (&residual + &hidden_states).map_err(Error::from)
    }

    fn forward_sequence(
        &self,
        hidden_states: &Tensor,
        state: &mut Qwen35LayerRuntimeState,
        position_ids: &[[usize; 3]],
    ) -> Result<Tensor> {
        let seq_len = hidden_states.dim(1)?;
        if seq_len == 1 {
            let position_id = *position_ids.first().ok_or_else(|| {
                Error::InvalidInput(
                    "Qwen3.5 forward_sequence expected at least one position id".to_string(),
                )
            })?;
            return self.forward(hidden_states, state, position_id);
        }
        if seq_len != position_ids.len() {
            return Err(Error::InvalidInput(format!(
                "Qwen3.5 layer sequence mismatch: seq_len={}, position_ids={}",
                seq_len,
                position_ids.len()
            )));
        }

        let residual = hidden_states.clone();
        let hidden_states = self.attn_norm.forward(hidden_states)?;
        let hidden_states = match &self.mixer {
            Qwen35Mixer::Linear(mixer) => mixer.forward_sequence(&hidden_states, state)?,
            Qwen35Mixer::Full(mixer) => {
                mixer.forward_sequence(&hidden_states, state, position_ids)?
            }
        };
        let hidden_states = (&residual + &hidden_states)?;

        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_norm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        (&residual + &hidden_states).map_err(Error::from)
    }
}

impl Qwen35Mlp {
    fn load(loader: &GgufLoader, device: &Device, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate: load_qmatmul(loader, device, &format!("{prefix}.ffn_gate.weight"))?,
            up: load_qmatmul(loader, device, &format!("{prefix}.ffn_up.weight"))?,
            down: load_qmatmul(loader, device, &format!("{prefix}.ffn_down.weight"))?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Use fused SiLU-gate-up if available (reduces memory bandwidth)
        let gate_proj_out = self.gate.forward(hidden_states)?;
        let up_proj_out = self.up.forward(hidden_states)?;

        let hidden = if let Some(fused) = try_fused_silu_mul(&gate_proj_out, &up_proj_out) {
            fused
        } else {
            let gate = ops::silu(&gate_proj_out)?;
            (&gate * &up_proj_out)?
        };

        self.down.forward(&hidden).map_err(Error::from)
    }
}

impl Qwen35FullAttention {
    fn load(
        loader: &GgufLoader,
        device: &Device,
        prefix: &str,
        cfg: &Qwen35TextConfig,
    ) -> Result<Self> {
        let (dense_decode_attention_enabled, dense_decode_max_pages) =
            qwen35_dense_decode_policy(device);
        let kv_page_size = default_kv_page_size();
        let kv_quantization = default_kv_quantization();
        let dense_decode_max_tokens = kv_page_size.saturating_mul(dense_decode_max_pages.max(1));
        Ok(Self {
            q_proj: load_qmatmul(loader, device, &format!("{prefix}.attn_q.weight"))?,
            k_proj: load_qmatmul(loader, device, &format!("{prefix}.attn_k.weight"))?,
            v_proj: load_qmatmul(loader, device, &format!("{prefix}.attn_v.weight"))?,
            o_proj: load_qmatmul(loader, device, &format!("{prefix}.attn_output.weight"))?,
            q_norm: load_rms_norm(loader, device, &format!("{prefix}.attn_q_norm.weight"), cfg)?,
            k_norm: load_rms_norm(loader, device, &format!("{prefix}.attn_k_norm.weight"), cfg)?,
            num_heads: cfg.attention_head_count,
            num_kv_heads: cfg.attention_head_count_kv,
            head_dim: cfg.attention_key_length,
            rope_dim: cfg.rope_dimension_count.min(cfg.attention_key_length),
            rope_theta: cfg.rope_freq_base,
            mrope_sections: cfg
                .rope_dimension_sections
                .iter()
                .copied()
                .filter(|section| *section > 0)
                .take(3)
                .collect(),
            kv_page_size,
            kv_quantization,
            dense_decode_attention_enabled,
            dense_decode_max_pages,
            dense_decode_max_tokens,
            rope_kernel_enabled: qwen35_rope_kernel_enabled(device),
            rope_inv_freqs: build_rope_inv_freqs(
                cfg.rope_dimension_count.min(cfg.attention_key_length),
                cfg.rope_freq_base,
            )?,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        state: &mut Qwen35LayerRuntimeState,
        position_ids: [usize; 3],
    ) -> Result<Tensor> {
        let (k_pages, v_pages, dense_k_cache_h, dense_v_cache_h, dense_kv_tokens) = match state {
            Qwen35LayerRuntimeState::Full {
                k_pages,
                v_pages,
                dense_k_cache_h,
                dense_v_cache_h,
                dense_kv_tokens,
            } => (
                k_pages,
                v_pages,
                dense_k_cache_h,
                dense_v_cache_h,
                dense_kv_tokens,
            ),
            _ => {
                return Err(Error::InferenceError(
                    "Qwen3.5 layer runtime state does not match full-attention layer".to_string(),
                ))
            }
        };

        let q_proj = self.q_proj.forward(hidden_states)?.reshape((
            1,
            1,
            self.num_heads,
            self.head_dim * 2,
        ))?;
        let query_states = q_proj.narrow(3, 0, self.head_dim)?;
        let gate = q_proj.narrow(3, self.head_dim, self.head_dim)?.reshape((
            1,
            1,
            self.num_heads * self.head_dim,
        ))?;
        let key_states = self.k_proj.forward(hidden_states)?.reshape((
            1,
            1,
            self.num_kv_heads,
            self.head_dim,
        ))?;
        let value_states = self.v_proj.forward(hidden_states)?.reshape((
            1,
            1,
            self.num_kv_heads,
            self.head_dim,
        ))?;

        let query_states = self.q_norm.forward(&query_states.contiguous()?)?;
        let key_states = self.k_norm.forward(&key_states.contiguous()?)?;
        let (query_states, key_states) =
            self.apply_rope(&query_states, &key_states, position_ids)?;

        // Keep KV in dense head-major cache while dense decode remains active.
        // We lazily migrate to paged KV once dense threshold is exceeded.
        if self.dense_decode_attention_enabled && k_pages.is_empty() && v_pages.is_empty() {
            append_dense_kv_cache_h(dense_k_cache_h, &key_states.transpose(1, 2)?.contiguous()?)?;
            append_dense_kv_cache_h(
                dense_v_cache_h,
                &value_states.transpose(1, 2)?.contiguous()?,
            )?;
            *dense_kv_tokens = dense_kv_tokens.saturating_add(1);
            maybe_materialize_dense_kv_pages(
                self.kv_page_size,
                self.kv_quantization,
                self.dense_decode_max_tokens,
                k_pages,
                v_pages,
                dense_kv_tokens,
                dense_k_cache_h,
                dense_v_cache_h,
            )?;
        } else {
            // Any dense tensor associated with paged storage becomes stale as
            // soon as a page is appended. Force a fresh materialization below.
            *dense_k_cache_h = None;
            *dense_v_cache_h = None;
            *dense_kv_tokens = 0;
            append_to_pages(
                self.kv_page_size,
                k_pages,
                &key_states,
                self.kv_quantization,
            )?;
            append_to_pages(
                self.kv_page_size,
                v_pages,
                &value_states,
                self.kv_quantization,
            )?;
        }

        let use_paged_decode = query_states.dim(1)? == 1
            && !k_pages.is_empty()
            && !v_pages.is_empty()
            && !qwen35_use_dense_decode_attention(
                self.dense_decode_attention_enabled,
                self.dense_decode_max_pages,
                k_pages.len(),
            );
        let is_decode_step = query_states.dim(1)? == 1;
        let attn_output = if use_paged_decode {
            // Once decode switches to paged attention, we do not need dense caches anymore.
            *dense_k_cache_h = None;
            *dense_v_cache_h = None;
            *dense_kv_tokens = 0;
            paged_decode_attention(
                &query_states,
                k_pages,
                v_pages,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
            )?
            .reshape((1, 1, self.num_heads * self.head_dim))?
        } else {
            let key_states_h = if let Some(cached) = dense_k_cache_h.as_ref() {
                cached.clone()
            } else {
                let materialized = materialize_pages(k_pages)?;
                let materialized_h = materialized.transpose(1, 2)?.contiguous()?;
                if self.dense_decode_attention_enabled {
                    *dense_k_cache_h = Some(materialized_h.clone());
                }
                materialized_h
            };
            let value_states_h = if let Some(cached) = dense_v_cache_h.as_ref() {
                cached.clone()
            } else {
                let materialized = materialize_pages(v_pages)?;
                let materialized_h = materialized.transpose(1, 2)?.contiguous()?;
                if self.dense_decode_attention_enabled {
                    *dense_v_cache_h = Some(materialized_h.clone());
                }
                materialized_h
            };
            if is_decode_step {
                record_decode_attention_path(DecodeAttentionPath::Dense);
            }

            let query_states = query_states.transpose(1, 2)?.contiguous()?;
            let attn_output = if let Some(out) = try_fused_self_attention(
                &query_states,
                &key_states_h,
                &value_states_h,
                None,
                self.head_dim,
                true,
            )? {
                out
            } else {
                unfused_causal_attention(
                    &query_states,
                    &key_states_h,
                    &value_states_h,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                )?
            };
            attn_output
                .transpose(1, 2)?
                .reshape((1, 1, self.num_heads * self.head_dim))?
        };
        let attn_output = (&attn_output * &ops::sigmoid(&gate)?)?;
        self.o_proj.forward(&attn_output).map_err(Error::from)
    }

    fn forward_sequence(
        &self,
        hidden_states: &Tensor,
        state: &mut Qwen35LayerRuntimeState,
        position_ids: &[[usize; 3]],
    ) -> Result<Tensor> {
        let seq_len = hidden_states.dim(1)?;
        if seq_len == 1 {
            let position_id = *position_ids.first().ok_or_else(|| {
                Error::InvalidInput(
                    "Qwen3.5 full-attention sequence expected at least one position id".to_string(),
                )
            })?;
            return self.forward(hidden_states, state, position_id);
        }
        if seq_len != position_ids.len() {
            return Err(Error::InvalidInput(format!(
                "Qwen3.5 full-attention sequence mismatch: seq_len={}, position_ids={}",
                seq_len,
                position_ids.len()
            )));
        }

        let state_has_cached_prefix = full_attention_state_has_cached_prefix(state)?;
        let (k_pages, v_pages, dense_k_cache_h, dense_v_cache_h, dense_kv_tokens) = match state {
            Qwen35LayerRuntimeState::Full {
                k_pages,
                v_pages,
                dense_k_cache_h,
                dense_v_cache_h,
                dense_kv_tokens,
            } => (
                k_pages,
                v_pages,
                dense_k_cache_h,
                dense_v_cache_h,
                dense_kv_tokens,
            ),
            _ => {
                return Err(Error::InferenceError(
                    "Qwen3.5 layer runtime state does not match full-attention layer".to_string(),
                ))
            }
        };

        let q_proj = self.q_proj.forward(hidden_states)?.reshape((
            1,
            seq_len,
            self.num_heads,
            self.head_dim * 2,
        ))?;
        let query_states = q_proj.narrow(3, 0, self.head_dim)?;
        let gate = q_proj.narrow(3, self.head_dim, self.head_dim)?.reshape((
            1,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;
        let key_states = self.k_proj.forward(hidden_states)?.reshape((
            1,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ))?;
        let value_states_kv = self.v_proj.forward(hidden_states)?.reshape((
            1,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ))?;

        let query_states = self.q_norm.forward(&query_states.contiguous()?)?;
        let key_states = self.k_norm.forward(&key_states.contiguous()?)?;
        let (query_states, key_states_kv) =
            self.apply_rope_sequence(&query_states, &key_states, position_ids)?;

        let query_states = query_states.transpose(1, 2)?.contiguous()?;
        let key_states_h = key_states_kv.transpose(1, 2)?.contiguous()?;
        let value_states_h = value_states_kv.transpose(1, 2)?.contiguous()?;

        // Cached chunks have rectangular Q/K lengths. Materialize the prefix
        // once, concatenate the current K/V, and apply a bottom-right-aligned
        // causal mask so query i sees the entire prefix plus current tokens
        // through i. The supplied mRoPE positions above remain authoritative.
        let cached_key_states_h = cached_kv_head_major(k_pages, dense_k_cache_h)?;
        let cached_value_states_h = cached_kv_head_major(v_pages, dense_v_cache_h)?;
        if cached_key_states_h.is_some() != cached_value_states_h.is_some() {
            return Err(Error::InferenceError(
                "Qwen3.5 full-attention key/value prefix caches are inconsistent".to_string(),
            ));
        }
        if k_pages.is_empty() && v_pages.is_empty() && *dense_kv_tokens > 0 {
            let cached_tokens = cached_key_states_h
                .as_ref()
                .map(|cache| cache.dim(2))
                .transpose()?
                .unwrap_or(0);
            let cached_value_tokens = cached_value_states_h
                .as_ref()
                .map(|cache| cache.dim(2))
                .transpose()?
                .unwrap_or(0);
            if cached_tokens != *dense_kv_tokens || cached_value_tokens != *dense_kv_tokens {
                return Err(Error::InferenceError(format!(
                    "Qwen3.5 dense KV prefix mismatch: tracked={}, k={}, v={}",
                    *dense_kv_tokens, cached_tokens, cached_value_tokens
                )));
            }
        }
        if state_has_cached_prefix
            && cached_key_states_h.is_none()
            && cached_value_states_h.is_none()
        {
            return Err(Error::InferenceError(format!(
                "Qwen3.5 full-attention state tracks a cached prefix ({} dense tokens) without KV storage",
                *dense_kv_tokens
            )));
        }

        let has_cached_prefix = state_has_cached_prefix;
        let attention_key_states_h = if let Some(cached) = cached_key_states_h.as_ref() {
            Tensor::cat(&[cached, &key_states_h], 2)?
        } else {
            key_states_h.clone()
        };
        let attention_value_states_h = if let Some(cached) = cached_value_states_h.as_ref() {
            Tensor::cat(&[cached, &value_states_h], 2)?
        } else {
            value_states_h.clone()
        };

        // Candle's Metal vector-SDPA path (q_len <= 8) does not consume the
        // causal flag. Keep short Metal prefills on the reference path; a
        // single-token decode remains correct because every cached key is past.
        let fused_prefill_has_exact_causality = !query_states.device().is_metal() || seq_len > 8;
        let attn_output = if !has_cached_prefix && fused_prefill_has_exact_causality {
            if let Some(out) = try_fused_self_attention_preserve_dtype(
                &query_states,
                &attention_key_states_h,
                &attention_value_states_h,
                None,
                self.head_dim,
                true,
            )? {
                out
            } else {
                unfused_causal_attention(
                    &query_states,
                    &attention_key_states_h,
                    &attention_value_states_h,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                )?
            }
        } else {
            // Until fused kernels explicitly guarantee rectangular, offset-aware
            // causal semantics, use the F32 reference path for cached chunks.
            unfused_causal_attention(
                &query_states,
                &attention_key_states_h,
                &attention_value_states_h,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
            )?
        };
        let attn_output =
            attn_output
                .transpose(1, 2)?
                .reshape((1, seq_len, self.num_heads * self.head_dim))?;
        let attn_output = (&attn_output * &ops::sigmoid(&gate)?)?;
        let output = self.o_proj.forward(&attn_output)?;

        // Persist newly computed KV after the sequence attention pass.
        if self.dense_decode_attention_enabled && k_pages.is_empty() && v_pages.is_empty() {
            append_dense_kv_cache_h(dense_k_cache_h, &key_states_h)?;
            append_dense_kv_cache_h(dense_v_cache_h, &value_states_h)?;
            *dense_kv_tokens = dense_kv_tokens.saturating_add(seq_len);
            maybe_materialize_dense_kv_pages(
                self.kv_page_size,
                self.kv_quantization,
                self.dense_decode_max_tokens,
                k_pages,
                v_pages,
                dense_kv_tokens,
                dense_k_cache_h,
                dense_v_cache_h,
            )?;
        } else {
            *dense_k_cache_h = None;
            *dense_v_cache_h = None;
            *dense_kv_tokens = 0;
            append_to_pages(
                self.kv_page_size,
                k_pages,
                &key_states_kv,
                self.kv_quantization,
            )?;
            append_to_pages(
                self.kv_page_size,
                v_pages,
                &value_states_kv,
                self.kv_quantization,
            )?;
        }

        Ok(output)
    }

    fn apply_rope(
        &self,
        query_states: &Tensor,
        key_states: &Tensor,
        position_ids: [usize; 3],
    ) -> Result<(Tensor, Tensor)> {
        if self.rope_dim == 0 {
            return Ok((query_states.clone(), key_states.clone()));
        }
        let (cos, sin) = self.mrope(position_ids, query_states.device(), query_states.dtype())?;

        let query_rot = query_states.narrow(3, 0, self.rope_dim)?.contiguous()?;
        let key_rot = key_states.narrow(3, 0, self.rope_dim)?.contiguous()?;
        let (query_rot, key_rot) = if self.should_try_rope_kernel(query_states.dtype()) {
            match try_apply_rope_thd(&query_rot, &key_rot, &cos, &sin)? {
                Some((query_rot, key_rot)) => {
                    record_rope_kernel();
                    (query_rot, key_rot)
                }
                None => {
                    record_rope_manual();
                    (
                        apply_rotary_emb(&query_rot, &cos, &sin)?,
                        apply_rotary_emb(&key_rot, &cos, &sin)?,
                    )
                }
            }
        } else {
            record_rope_manual();
            (
                apply_rotary_emb(&query_rot, &cos, &sin)?,
                apply_rotary_emb(&key_rot, &cos, &sin)?,
            )
        };

        if self.rope_dim == self.head_dim {
            return Ok((query_rot, key_rot));
        }

        let query_pass = query_states.narrow(3, self.rope_dim, self.head_dim - self.rope_dim)?;
        let key_pass = key_states.narrow(3, self.rope_dim, self.head_dim - self.rope_dim)?;
        Ok((
            Tensor::cat(&[&query_rot, &query_pass], 3)?,
            Tensor::cat(&[&key_rot, &key_pass], 3)?,
        ))
    }

    fn apply_rope_sequence(
        &self,
        query_states: &Tensor,
        key_states: &Tensor,
        position_ids: &[[usize; 3]],
    ) -> Result<(Tensor, Tensor)> {
        let seq_len = query_states.dim(1)?;
        if seq_len != position_ids.len() {
            return Err(Error::InvalidInput(format!(
                "Qwen3.5 rotary sequence mismatch: seq_len={}, position_ids={}",
                seq_len,
                position_ids.len()
            )));
        }
        if self.rope_dim == 0 {
            return Ok((query_states.clone(), key_states.clone()));
        }

        let mut cos_tokens = Vec::with_capacity(seq_len);
        let mut sin_tokens = Vec::with_capacity(seq_len);
        for &position_id in position_ids {
            let (cos, sin) =
                self.mrope(position_id, query_states.device(), query_states.dtype())?;
            cos_tokens.push(cos);
            sin_tokens.push(sin);
        }
        let cos_refs: Vec<&Tensor> = cos_tokens.iter().collect();
        let sin_refs: Vec<&Tensor> = sin_tokens.iter().collect();
        let cos = Tensor::cat(&cos_refs, 1)?.contiguous()?;
        let sin = Tensor::cat(&sin_refs, 1)?.contiguous()?;

        let query_rot = query_states.narrow(3, 0, self.rope_dim)?.contiguous()?;
        let key_rot = key_states.narrow(3, 0, self.rope_dim)?.contiguous()?;
        let (query_rot, key_rot) = if self.should_try_rope_kernel(query_states.dtype()) {
            match try_apply_rope_thd(&query_rot, &key_rot, &cos, &sin)? {
                Some((query_rot, key_rot)) => {
                    for _ in 0..seq_len {
                        record_rope_kernel();
                    }
                    (query_rot, key_rot)
                }
                None => {
                    for _ in 0..seq_len {
                        record_rope_manual();
                    }
                    (
                        apply_rotary_emb(&query_rot, &cos, &sin)?,
                        apply_rotary_emb(&key_rot, &cos, &sin)?,
                    )
                }
            }
        } else {
            for _ in 0..seq_len {
                record_rope_manual();
            }
            (
                apply_rotary_emb(&query_rot, &cos, &sin)?,
                apply_rotary_emb(&key_rot, &cos, &sin)?,
            )
        };

        if self.rope_dim == self.head_dim {
            return Ok((query_rot, key_rot));
        }

        let query_pass = query_states.narrow(3, self.rope_dim, self.head_dim - self.rope_dim)?;
        let key_pass = key_states.narrow(3, self.rope_dim, self.head_dim - self.rope_dim)?;
        Ok((
            Tensor::cat(&[&query_rot, &query_pass], 3)?,
            Tensor::cat(&[&key_rot, &key_pass], 3)?,
        ))
    }

    fn mrope(
        &self,
        position_ids: [usize; 3],
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        build_mrope(
            self.rope_dim,
            position_ids,
            &self.mrope_sections,
            &self.rope_inv_freqs,
            device,
            dtype,
        )
    }

    fn should_try_rope_kernel(&self, dtype: DType) -> bool {
        if !self.rope_kernel_enabled {
            return false;
        }
        if self.rope_dim == 0 || self.rope_dim % 2 != 0 {
            return false;
        }
        matches!(dtype, DType::F16 | DType::BF16 | DType::F32)
    }
}

/// Correctness reference for Qwen3.5 causal attention. Q/K/V use head-major
/// `[batch, heads, tokens, head_dim]` layout. When `query_len < key_len`, query
/// positions are bottom-right aligned with the cached prefix.
fn unfused_causal_attention(
    query_states: &Tensor,
    key_states_h: &Tensor,
    value_states_h: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let (query_batch, query_heads, query_len, query_head_dim) = query_states.dims4()?;
    let (key_batch, key_heads, key_len, key_head_dim) = key_states_h.dims4()?;
    let (value_batch, value_heads, value_len, value_head_dim) = value_states_h.dims4()?;
    if query_len == 0 || key_len == 0 || query_len > key_len {
        return Err(Error::InvalidInput(format!(
            "Invalid Qwen3.5 causal attention lengths: query={query_len}, key={key_len}"
        )));
    }
    if query_batch != key_batch
        || key_batch != value_batch
        || query_heads != num_heads
        || key_heads != num_kv_heads
        || value_heads != num_kv_heads
        || key_len != value_len
        || query_head_dim != head_dim
        || key_head_dim != head_dim
        || value_head_dim != head_dim
    {
        return Err(Error::InvalidInput(format!(
            "Invalid Qwen3.5 attention shapes: q={:?}, k={:?}, v={:?}, heads={num_heads}/{num_kv_heads}, head_dim={head_dim}",
            query_states.dims(),
            key_states_h.dims(),
            value_states_h.dims(),
        )));
    }

    // `repeat_kv` consumes sequence-major KV and returns sequence-major GQA
    // expansion. Convert back to head-major for the score/value matmuls.
    let key_states = key_states_h.transpose(1, 2)?.contiguous()?;
    let value_states = value_states_h.transpose(1, 2)?.contiguous()?;
    let key_states = repeat_kv(&key_states, num_heads, num_kv_heads)?;
    let value_states = repeat_kv(&value_states, num_heads, num_kv_heads)?;
    let key_states = key_states.transpose(1, 2)?.contiguous()?;
    let value_states = value_states.transpose(1, 2)?.contiguous()?;

    let output_dtype = query_states.dtype();
    let key_states_t = key_states.transpose(2, 3)?.contiguous()?;
    let scores = query_states.matmul(&key_states_t)?;
    let mut scores = (scores / (head_dim as f64).sqrt())?.to_dtype(DType::F32)?;
    if query_len > 1 {
        let query_start = key_len - query_len;
        let mask = qwen35_causal_attention_mask(
            query_len,
            key_len,
            query_start,
            scores.device(),
            DType::F32,
        )?;
        scores = scores.broadcast_add(&mask)?;
    }
    let probabilities = ops::softmax_last_dim(&scores)?;
    let values_f32 = value_states.to_dtype(DType::F32)?;
    let output = probabilities.contiguous()?.matmul(&values_f32)?;
    if output.dtype() == output_dtype {
        Ok(output)
    } else {
        output.to_dtype(output_dtype).map_err(Error::from)
    }
}

fn cached_kv_head_major(
    pages: &[KvPage],
    dense_cache_h: &Option<Tensor>,
) -> Result<Option<Tensor>> {
    // Pages are authoritative whenever present. A dense tensor can be a
    // transient materialization of an earlier page set and therefore stale
    // after a later append.
    if !pages.is_empty() {
        return Ok(Some(
            materialize_pages(pages)?.transpose(1, 2)?.contiguous()?,
        ));
    }
    Ok(dense_cache_h.clone())
}

fn qwen35_causal_attention_mask(
    query_len: usize,
    key_len: usize,
    query_start: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    if query_len == 0
        || key_len == 0
        || query_start > key_len
        || query_start.saturating_add(query_len) > key_len
    {
        return Err(Error::InvalidInput(format!(
            "Invalid Qwen3.5 causal mask: query={query_len}, key={key_len}, start={query_start}"
        )));
    }

    let mut values = vec![0f32; query_len * key_len];
    for query_idx in 0..query_len {
        let last_visible_key = query_start + query_idx;
        for key_idx in (last_visible_key + 1)..key_len {
            values[query_idx * key_len + key_idx] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(values, (1, 1, query_len, key_len), device)?
        .to_dtype(dtype)
        .map_err(Error::from)
}

impl Qwen35LinearAttention {
    fn load(
        loader: &GgufLoader,
        device: &Device,
        prefix: &str,
        cfg: &Qwen35TextConfig,
    ) -> Result<Self> {
        let num_k_heads = cfg.ssm_group_count;
        let num_v_heads = cfg.ssm_time_step_rank;
        let head_k_dim = cfg.ssm_state_size;
        let head_v_dim = cfg.ssm_inner_size / cfg.ssm_time_step_rank;
        let conv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

        let dt_bias_name = if loader.has_tensor(&format!("{prefix}.ssm_dt.bias")) {
            format!("{prefix}.ssm_dt.bias")
        } else {
            format!("{prefix}.ssm_dt")
        };
        let dt_bias = load_vector(loader, device, &dt_bias_name, num_v_heads)?.reshape((
            1,
            1,
            num_v_heads,
        ))?;
        let a = load_vector(loader, device, &format!("{prefix}.ssm_a"), num_v_heads)?.reshape((
            1,
            1,
            num_v_heads,
        ))?;
        let conv_kernel = normalize_conv_kernel(
            load_dense(
                loader,
                device,
                &format!("{prefix}.ssm_conv1d.weight"),
                Some(DType::F32),
            )?,
            conv_dim,
            cfg.ssm_conv_kernel,
        )?;
        let conv_kernel_slices = pre_slice_conv_kernel(&conv_kernel, cfg.ssm_conv_kernel)?;
        let norm = Qwen35GatedRmsNorm {
            weight: load_vector(
                loader,
                device,
                &format!("{prefix}.ssm_norm.weight"),
                head_v_dim,
            )?,
            eps: cfg.attention_layer_norm_rms_epsilon,
        };

        Ok(Self {
            qkv_proj: load_qmatmul(loader, device, &format!("{prefix}.attn_qkv.weight"))?,
            gate_proj: load_qmatmul(loader, device, &format!("{prefix}.attn_gate.weight"))?,
            beta_proj: load_qmatmul(loader, device, &format!("{prefix}.ssm_beta.weight"))?,
            alpha_proj: load_qmatmul(loader, device, &format!("{prefix}.ssm_alpha.weight"))?,
            dt_bias,
            a,
            conv_kernel,
            conv_kernel_slices,
            norm,
            out_proj: load_qmatmul(loader, device, &format!("{prefix}.ssm_out.weight"))?,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_dim,
            kernel_size: cfg.ssm_conv_kernel,
            tiled_recurrence_enabled: qwen35_tiled_recurrence_enabled(),
            tiled_recurrence_tile_size_override: qwen35_tiled_recurrence_tile_size_override(),
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        state: &mut Qwen35LayerRuntimeState,
    ) -> Result<Tensor> {
        let (conv_state, recurrent_state) = match state {
            Qwen35LayerRuntimeState::Linear {
                conv_state,
                recurrent_state,
            } => (conv_state, recurrent_state),
            _ => {
                return Err(Error::InferenceError(
                    "Qwen3.5 layer runtime state does not match linear-attention layer".to_string(),
                ))
            }
        };

        let mixed_qkv = self.qkv_proj.forward(hidden_states)?;
        let z = self.gate_proj.forward(hidden_states)?;
        let beta = ops::sigmoid(&self.beta_proj.forward(hidden_states)?)?;
        let alpha = self.alpha_proj.forward(hidden_states)?;
        let g = softplus(&alpha.broadcast_add(&self.dt_bias)?)?.broadcast_mul(&self.a)?;

        let mixed_qkv = self.depthwise_conv_step(&mixed_qkv, conv_state)?;

        let key_width = self.num_k_heads * self.head_k_dim;
        let value_width = self.num_v_heads * self.head_v_dim;
        let query =
            mixed_qkv
                .narrow(2, 0, key_width)?
                .reshape((1, self.num_k_heads, self.head_k_dim))?;
        let key = mixed_qkv.narrow(2, key_width, key_width)?.reshape((
            1,
            self.num_k_heads,
            self.head_k_dim,
        ))?;
        let value = mixed_qkv.narrow(2, key_width * 2, value_width)?.reshape((
            1,
            self.num_v_heads,
            self.head_v_dim,
        ))?;

        let mut query = l2norm(&query, 1e-6)?;
        let mut key = l2norm(&key, 1e-6)?;
        if self.num_v_heads != self.num_k_heads {
            if self.num_k_heads == 0 || !self.num_v_heads.is_multiple_of(self.num_k_heads) {
                return Err(Error::InferenceError(format!(
                    "Invalid linear-attention head layout: num_v_heads={}, num_k_heads={}",
                    self.num_v_heads, self.num_k_heads
                )));
            }
            let repeats = self.num_v_heads / self.num_k_heads;
            query = repeat_head_states(&query, repeats)?;
            key = repeat_head_states(&key, repeats)?;
        }

        let current_state = if let Some(state) = recurrent_state.take() {
            state
        } else {
            Tensor::zeros(
                (1, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                value.dtype(),
                value.device(),
            )?
        };

        let beta = beta.reshape((1, self.num_v_heads))?;
        let g = g.reshape((1, self.num_v_heads))?;
        let (output, next_state) =
            recurrent_gated_delta(&query, &key, &value, &g, &beta, current_state)?;
        *recurrent_state = Some(compact_tensor_storage(&next_state)?);

        let output = output.reshape((self.num_v_heads, self.head_v_dim))?;
        let z = z.reshape((self.num_v_heads, self.head_v_dim))?;
        let output = self.norm.forward(&output, &z)?;
        let output = output.reshape((1, 1, self.num_v_heads * self.head_v_dim))?;
        self.out_proj.forward(&output).map_err(Error::from)
    }

    fn forward_sequence(
        &self,
        hidden_states: &Tensor,
        state: &mut Qwen35LayerRuntimeState,
    ) -> Result<Tensor> {
        let seq_len = hidden_states.dim(1)?;
        if seq_len == 1 {
            return self.forward(hidden_states, state);
        }

        let (conv_state, recurrent_state) = match state {
            Qwen35LayerRuntimeState::Linear {
                conv_state,
                recurrent_state,
            } => (conv_state, recurrent_state),
            _ => {
                return Err(Error::InferenceError(
                    "Qwen3.5 layer runtime state does not match linear-attention layer".to_string(),
                ))
            }
        };

        let mixed_qkv = self.qkv_proj.forward(hidden_states)?;
        let z = self.gate_proj.forward(hidden_states)?;
        let beta = ops::sigmoid(&self.beta_proj.forward(hidden_states)?)?;
        let alpha = self.alpha_proj.forward(hidden_states)?;
        let g = softplus(&alpha.broadcast_add(&self.dt_bias)?)?.broadcast_mul(&self.a)?;

        let mixed_qkv = self.depthwise_conv_sequence(&mixed_qkv, conv_state)?;

        let key_width = self.num_k_heads * self.head_k_dim;
        let value_width = self.num_v_heads * self.head_v_dim;
        let query = mixed_qkv.narrow(2, 0, key_width)?.reshape((
            1,
            seq_len,
            self.num_k_heads,
            self.head_k_dim,
        ))?;
        let key = mixed_qkv.narrow(2, key_width, key_width)?.reshape((
            1,
            seq_len,
            self.num_k_heads,
            self.head_k_dim,
        ))?;
        let value = mixed_qkv.narrow(2, key_width * 2, value_width)?.reshape((
            1,
            seq_len,
            self.num_v_heads,
            self.head_v_dim,
        ))?;

        let query = l2norm(&query, 1e-6)?;
        let key = l2norm(&key, 1e-6)?;
        if self.num_k_heads == 0 || !self.num_v_heads.is_multiple_of(self.num_k_heads) {
            return Err(Error::InferenceError(format!(
                "Invalid linear-attention head layout: num_v_heads={}, num_k_heads={}",
                self.num_v_heads, self.num_k_heads
            )));
        }

        let current_state = if let Some(state) = recurrent_state.take() {
            state
        } else {
            Tensor::zeros(
                (1, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                value.dtype(),
                value.device(),
            )?
        };

        let beta = beta.reshape((1, seq_len, self.num_v_heads))?;
        let g = g.reshape((1, seq_len, self.num_v_heads))?;
        let tile_size =
            qwen35_tiled_recurrence_tile_size(seq_len, self.tiled_recurrence_tile_size_override);
        let fused_sequence = if self.tiled_recurrence_enabled {
            try_tiled_deltanet_recurrence(
                &query,
                &key,
                &value,
                &g,
                &beta,
                &current_state,
                tile_size,
            )
        } else {
            None
        };
        let (output, next_state) = if let Some(fused_sequence) = fused_sequence {
            fused_sequence
        } else {
            // CUDA's equal-head kernel and the portable Candle reference consume
            // tiled Q/K heads. The Metal sequence op above consumes the compact
            // converted-GGUF 16K layout directly for both 16V and 32V models.
            let (query, key) = if self.num_v_heads == self.num_k_heads {
                (query, key)
            } else {
                let repeats = self.num_v_heads / self.num_k_heads;
                (
                    repeat_head_states_seq(&query, repeats)?,
                    repeat_head_states_seq(&key, repeats)?,
                )
            };
            if self.tiled_recurrence_enabled {
                if let Some(fused_sequence) = try_tiled_deltanet_recurrence(
                    &query,
                    &key,
                    &value,
                    &g,
                    &beta,
                    &current_state,
                    tile_size,
                ) {
                    fused_sequence
                } else {
                    recurrent_gated_delta_sequence(&query, &key, &value, &g, &beta, current_state)?
                }
            } else {
                recurrent_gated_delta_sequence(&query, &key, &value, &g, &beta, current_state)?
            }
        };
        *recurrent_state = Some(compact_tensor_storage(&next_state)?);

        let output = output.reshape((seq_len * self.num_v_heads, self.head_v_dim))?;
        let z = z.reshape((seq_len * self.num_v_heads, self.head_v_dim))?;
        let output = self.norm.forward(&output, &z)?;
        let output = output.reshape((1, seq_len, self.num_v_heads * self.head_v_dim))?;
        self.out_proj.forward(&output).map_err(Error::from)
    }

    fn depthwise_conv_sequence(
        &self,
        mixed_qkv: &Tensor,
        conv_state: &mut Option<ConvRingState>,
    ) -> Result<Tensor> {
        let seq_len = mixed_qkv.dim(1)?;
        if seq_len == 1 {
            return self.depthwise_conv_step(mixed_qkv, conv_state);
        }

        if self.kernel_size > 1 {
            let history_len = self.kernel_size - 1;
            let buffer = conv_state.as_mut().ok_or_else(|| {
                Error::InferenceError("conv_state not initialized but kernel_size > 1".to_string())
            })?;
            if buffer.slots.len() != history_len || buffer.next_idx >= history_len {
                return Err(Error::InferenceError(format!(
                    "Invalid Qwen3.5 convolution history: slots={}, next_idx={}, expected_slots={history_len}",
                    buffer.slots.len(),
                    buffer.next_idx
                )));
            }
            let logical_slots: Vec<&Tensor> = (0..history_len)
                .map(|idx| &buffer.slots[(buffer.next_idx + idx) % history_len])
                .collect();
            let history = Tensor::cat(&logical_slots, 1)?;
            if let Some((output, final_history)) =
                try_qwen35_causal_conv_sequence(mixed_qkv, &self.conv_kernel, &history)
            {
                let final_history = compact_tensor_storage(&final_history)?;
                buffer.slots = (0..history_len)
                    .map(|idx| final_history.narrow(1, idx, 1))
                    .collect::<candle_core::Result<Vec<_>>>()?;
                buffer.next_idx = 0;
                return Ok(output);
            }
        }

        let mut outputs = Vec::with_capacity(seq_len);
        for idx in 0..seq_len {
            let token = mixed_qkv.narrow(1, idx, 1)?;
            outputs.push(self.depthwise_conv_step(&token, conv_state)?);
        }
        if let Some(conv_state) = conv_state.as_mut() {
            conv_state.compact_owned()?;
        }
        let output_refs: Vec<&Tensor> = outputs.iter().collect();
        Tensor::cat(&output_refs, 1).map_err(Error::from)
    }

    fn depthwise_conv_step(
        &self,
        mixed_qkv: &Tensor,
        conv_state: &mut Option<ConvRingState>,
    ) -> Result<Tensor> {
        let current = mixed_qkv.i((0, 0))?;
        let current = if current.dtype() != self.conv_kernel.dtype() {
            current.to_dtype(self.conv_kernel.dtype())?
        } else {
            current
        };
        let current = current.reshape((self.conv_dim, 1))?;

        let convolved = if self.kernel_size <= 1 {
            (&current * &self.conv_kernel)?.sum(D::Minus1)?
        } else {
            let buffer = if let Some(state) = conv_state.as_mut() {
                state
            } else {
                return Err(Error::InferenceError(
                    "conv_state not initialized but kernel_size > 1".to_string(),
                ));
            };

            // Compute convolution as sum of elementwise products
            // self.conv_kernel shape: (conv_dim, kernel_size)
            // ring buffer contains kernel_size - 1 past states of shape (conv_dim, 1)

            // Start with current * conv_kernel[:, kernel_size - 1]
            let k_slice = &self.conv_kernel_slices[self.kernel_size - 1];
            let mut convolved = (&current * k_slice)?;

            // Add previous tokens * their respective kernel weights.
            // Read history in oldest -> newest order from the circular ring.
            let history_len = self.kernel_size - 1;
            for i in 0..(self.kernel_size - 1) {
                let ring_idx = (buffer.next_idx + i) % history_len;
                let prev_token = &buffer.slots[ring_idx];
                let k_slice = &self.conv_kernel_slices[i];
                convolved = (&convolved + &(prev_token * k_slice)?)?;
            }

            // Update the ring buffer in O(1): overwrite oldest and advance cursor.
            buffer.slots[buffer.next_idx] = current;
            buffer.next_idx = (buffer.next_idx + 1) % history_len;

            convolved.squeeze(1)?
        };

        let convolved = ops::silu(&convolved)?;
        convolved
            .reshape((1, 1, self.conv_dim))
            .map_err(Error::from)
    }
}

impl Qwen35GatedRmsNorm {
    fn forward(&self, hidden_states: &Tensor, gate: &Tensor) -> Result<Tensor> {
        if hidden_states.dtype() == DType::F32 {
            if let Some(result) =
                try_fused_gated_rms_norm(hidden_states, gate, &self.weight, self.eps)
            {
                return Ok(result);
            }
        }

        let normalized = candle_nn::ops::rms_norm(hidden_states, &self.weight, self.eps as f32)?;
        (&normalized * &ops::silu(gate)?).map_err(Error::from)
    }
}

fn is_full_attention_layer(layer_idx: usize, full_attention_interval: usize) -> bool {
    full_attention_interval > 0 && (layer_idx + 1).is_multiple_of(full_attention_interval)
}

fn load_qmatmul(loader: &GgufLoader, device: &Device, name: &str) -> Result<QMatMul> {
    let weights = Arc::new(loader.load_qtensor(name, device)?);
    QMatMul::from_weights(weights).map_err(Error::from)
}

fn load_rms_norm(
    loader: &GgufLoader,
    device: &Device,
    name: &str,
    cfg: &Qwen35TextConfig,
) -> Result<RmsNorm> {
    RmsNorm::from_qtensor(
        loader.load_qtensor(name, device)?,
        cfg.attention_layer_norm_rms_epsilon,
    )
    .map_err(Error::from)
}

fn load_dense(
    loader: &GgufLoader,
    device: &Device,
    name: &str,
    dtype: Option<DType>,
) -> Result<Tensor> {
    let mut tensor = loader
        .load_qtensor(name, device)?
        .dequantize(device)
        .map_err(Error::from)?;
    if let Some(dtype) = dtype {
        if tensor.dtype() != dtype {
            tensor = tensor.to_dtype(dtype)?;
        }
    }
    Ok(tensor)
}

fn load_vector(
    loader: &GgufLoader,
    device: &Device,
    name: &str,
    expected_len: usize,
) -> Result<Tensor> {
    let tensor = load_dense(loader, device, name, Some(DType::F32))?;
    let actual_len = tensor.elem_count();
    if actual_len != expected_len {
        return Err(Error::ModelLoadError(format!(
            "Unexpected tensor size for {name}: expected {expected_len} elements, found {actual_len}"
        )));
    }
    tensor.reshape((expected_len,)).map_err(Error::from)
}

fn normalize_conv_kernel(
    tensor: Tensor,
    expected_channels: usize,
    expected_kernel: usize,
) -> Result<Tensor> {
    match tensor.rank() {
        2 => {
            let (d0, d1) = tensor.dims2()?;
            if d0 == expected_channels && d1 == expected_kernel {
                Ok(tensor)
            } else if d0 == expected_kernel && d1 == expected_channels {
                tensor.transpose(0, 1)?.contiguous().map_err(Error::from)
            } else {
                Err(Error::ModelLoadError(format!(
                    "Unexpected Qwen3.5 conv kernel shape: ({d0}, {d1}) for expected ({expected_channels}, {expected_kernel})"
                )))
            }
        }
        3 => {
            let dims = tensor.dims();
            if dims == [expected_channels, 1, expected_kernel] {
                tensor.squeeze(1).map_err(Error::from)
            } else if dims == [expected_kernel, 1, expected_channels] {
                tensor
                    .squeeze(1)?
                    .transpose(0, 1)?
                    .contiguous()
                    .map_err(Error::from)
            } else {
                Err(Error::ModelLoadError(format!(
                    "Unexpected rank-3 Qwen3.5 conv kernel shape: {:?}",
                    dims
                )))
            }
        }
        rank => Err(Error::ModelLoadError(format!(
            "Unexpected Qwen3.5 conv kernel rank {rank}"
        ))),
    }
}

fn pre_slice_conv_kernel(conv_kernel: &Tensor, kernel_size: usize) -> Result<Vec<Tensor>> {
    let mut slices = Vec::with_capacity(kernel_size);
    for idx in 0..kernel_size {
        slices.push(conv_kernel.narrow(1, idx, 1)?);
    }
    Ok(slices)
}

fn build_mrope(
    rope_dim: usize,
    position_ids: [usize; 3],
    mrope_sections: &[usize],
    inv_freqs: &[f32],
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let half_dim = rope_dim / 2;
    if inv_freqs.len() != half_dim {
        return Err(Error::InferenceError(format!(
            "Invalid Qwen3.5 rotary dimension {rope_dim}"
        )));
    }

    let mut temporal = vec![0f32; half_dim];
    let mut height = vec![0f32; half_dim];
    let mut width = vec![0f32; half_dim];
    for (idx, inv_freq) in inv_freqs.iter().enumerate() {
        temporal[idx] = position_ids[0] as f32 * inv_freq;
        height[idx] = position_ids[1] as f32 * inv_freq;
        width[idx] = position_ids[2] as f32 * inv_freq;
    }

    let mut interleaved = temporal.clone();
    if position_ids[0] != position_ids[1] || position_ids[0] != position_ids[2] {
        if mrope_sections.iter().sum::<usize>() != half_dim || mrope_sections.len() < 3 {
            return Err(Error::InferenceError(format!(
                "Invalid Qwen3.5 multimodal RoPE sections {:?} for rotary dim {}",
                mrope_sections, rope_dim
            )));
        }

        for (offset, source, section_len) in [
            (1usize, &height, mrope_sections[1]),
            (2usize, &width, mrope_sections[2]),
        ] {
            let stop = section_len * 3;
            for idx in (offset..stop.min(half_dim)).step_by(3) {
                interleaved[idx] = source[idx];
            }
        }
    }

    let emb = Tensor::from_vec(interleaved, (1, 1, half_dim), device)?.to_dtype(dtype)?;
    Ok((emb.cos()?, emb.sin()?))
}

fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half_dim = x.dim(3)? / 2;
    let x1 = x.narrow(3, 0, half_dim)?;
    let x2 = x.narrow(3, half_dim, half_dim)?;
    let cos = cos.unsqueeze(2)?;
    let sin = sin.unsqueeze(2)?;
    let out_first = x1
        .broadcast_mul(&cos)?
        .broadcast_sub(&x2.broadcast_mul(&sin)?)?;
    let out_second = x1
        .broadcast_mul(&sin)?
        .broadcast_add(&x2.broadcast_mul(&cos)?)?;
    Tensor::cat(&[&out_first, &out_second], 3).map_err(Error::from)
}

fn try_apply_rope_thd(
    query_rot: &Tensor,
    key_rot: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<Option<(Tensor, Tensor)>> {
    let kernel_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let query_rot = rotary_emb::rope_thd(query_rot, cos, sin)?;
        let key_rot = rotary_emb::rope_thd(key_rot, cos, sin)?;
        candle_core::Result::<(Tensor, Tensor)>::Ok((query_rot, key_rot))
    }));

    match kernel_result {
        Ok(Ok((query_rot, key_rot))) => Ok(Some((query_rot, key_rot))),
        Ok(Err(_)) | Err(_) => Ok(None),
    }
}

fn build_rope_inv_freqs(rope_dim: usize, rope_theta: f64) -> Result<Vec<f32>> {
    let half_dim = rope_dim / 2;
    let inv_freqs: Vec<f32> = (0..rope_dim)
        .step_by(2)
        .map(|idx| (1.0f64 / rope_theta.powf(idx as f64 / rope_dim as f64)) as f32)
        .collect();
    if inv_freqs.len() != half_dim {
        return Err(Error::InferenceError(format!(
            "Invalid Qwen3.5 rotary dimension {rope_dim}"
        )));
    }
    Ok(inv_freqs)
}

/// Allocate persistent runtime state with independent backing storage.
///
/// Candle tensor views can outlive a scratch-pool checkout, so state that
/// survives a forward call must never escape after its pool lease is released.
fn owned_zero_tensor(shape: &[usize], dtype: DType, device: &Device) -> Result<Tensor> {
    compact_tensor_storage(&Tensor::zeros(shape.to_vec(), dtype, device)?).map_err(Error::from)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NonFiniteCounts {
    nan: usize,
    positive_infinity: usize,
    negative_infinity: usize,
}

impl NonFiniteCounts {
    fn total(self) -> usize {
        self.nan
            .saturating_add(self.positive_infinity)
            .saturating_add(self.negative_infinity)
    }
}

/// Collect diagnostics without exposing model inputs, outputs, or token data.
/// This host readback is intentionally guarded by an opt-in environment flag
/// at inference call sites because checking every layer synchronizes the GPU.
fn non_finite_counts(tensor: &Tensor) -> Result<NonFiniteCounts> {
    let values = tensor
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let mut counts = NonFiniteCounts {
        nan: 0,
        positive_infinity: 0,
        negative_infinity: 0,
    };
    for value in values {
        if value.is_nan() {
            counts.nan = counts.nan.saturating_add(1);
        } else if value == f32::INFINITY {
            counts.positive_infinity = counts.positive_infinity.saturating_add(1);
        } else if value == f32::NEG_INFINITY {
            counts.negative_infinity = counts.negative_infinity.saturating_add(1);
        }
    }
    Ok(counts)
}

fn validate_qwen35_finite_tensor(
    tensor: &Tensor,
    layer_idx: usize,
    path: &str,
    enabled: bool,
) -> Result<()> {
    if !enabled {
        return Ok(());
    }

    let counts = non_finite_counts(tensor)?;
    if counts.total() == 0 {
        return Ok(());
    }

    Err(Error::InferenceError(format!(
        "Qwen3.5 first non-finite tensor at {path}, layer {layer_idx}: \
         {} of {} values ({} NaN, {} +Inf, {} -Inf), shape {:?}, dtype {:?}",
        counts.total(),
        tensor.elem_count(),
        counts.nan,
        counts.positive_infinity,
        counts.negative_infinity,
        tensor.dims(),
        tensor.dtype(),
    )))
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    // Evaluate DeltaNet discretization in F32 with the stable identity
    // max(x, 0) + log(1 + exp(-abs(x))). The direct log(exp(x) + 1)
    // formulation overflows for valid large positive activations.
    let x = x.to_dtype(DType::F32)?;
    let positive = x.relu()?;
    let correction = (x.abs()?.neg()?.exp()? + 1.0)?.log()?;
    (&positive + &correction).map_err(Error::from)
}

fn l2norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    // Try fused Metal kernel first for F32 tensors
    if x.dtype() == DType::F32 {
        if let Some(result) = try_fused_l2_norm(x, eps) {
            return Ok(result);
        }
    }

    // Fallback to standard implementation
    x.broadcast_div(&(x.sqr()?.sum_keepdim(D::Minus1)? + eps)?.sqrt()?)
        .map_err(Error::from)
}

fn repeat_head_states(x: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats <= 1 {
        return Ok(x.clone());
    }
    let (batch, heads, dim) = x.dims3()?;
    // Match llama.cpp's tiled repeat layout for Qwen3.5 linear attention:
    // [h0, h1, ...] -> [h0, h1, ..., h0, h1, ...].
    let expanded = x.unsqueeze(1)?.broadcast_as((batch, repeats, heads, dim))?;
    expanded
        .reshape((batch, repeats * heads, dim))
        .map_err(Error::from)
}

fn repeat_head_states_seq(x: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats <= 1 {
        return Ok(x.clone());
    }
    let (batch, seq, heads, dim) = x.dims4()?;
    let expanded = x
        .unsqueeze(2)?
        .broadcast_as((batch, seq, repeats, heads, dim))?;
    expanded
        .reshape((batch, seq, repeats * heads, dim))
        .map_err(Error::from)
}

fn qwen35_env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

fn qwen35_tiled_recurrence_enabled() -> bool {
    qwen35_env_bool("IZWI_QWEN35_TILED_RECURRENCE", true)
}

fn qwen35_rope_kernel_enabled(device: &Device) -> bool {
    device.is_metal() && qwen35_env_bool("IZWI_QWEN35_ROPE_KERNEL", true)
}

fn qwen35_tiled_recurrence_tile_size_override() -> Option<usize> {
    if let Ok(raw) = std::env::var("IZWI_QWEN35_TILED_RECURRENCE_TILE_SIZE") {
        if let Ok(parsed) = raw.trim().parse::<usize>() {
            return Some(parsed.max(1));
        }
    }
    None
}

fn qwen35_tiled_recurrence_tile_size(seq_len: usize, override_size: Option<usize>) -> usize {
    if let Some(override_size) = override_size {
        return override_size.min(seq_len.max(1));
    }

    if seq_len >= 256 {
        64
    } else if seq_len >= 64 {
        32
    } else if seq_len >= 16 {
        16
    } else {
        seq_len.max(1)
    }
}

fn qwen35_block_fusion_decode_enabled(device: &Device) -> bool {
    device.is_metal()
        && use_block_fusion_for_device(device)
        && qwen35_env_bool("IZWI_QWEN35_BLOCK_FUSION_DECODE", true)
}

fn qwen35_block_fusion_prefill_policy(device: &Device) -> (bool, usize) {
    if !device.is_metal() || !use_block_fusion_for_device(device) {
        return (false, 8);
    }
    let enabled = qwen35_env_bool("IZWI_QWEN35_BLOCK_FUSION_PREFILL", true);
    let min_tokens = std::env::var("IZWI_QWEN35_BLOCK_FUSION_PREFILL_MIN_TOKENS")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(8);
    (enabled, min_tokens)
}

fn qwen35_use_dense_decode_attention(
    dense_decode_attention_enabled: bool,
    dense_decode_max_pages: usize,
    page_count: usize,
) -> bool {
    if !dense_decode_attention_enabled {
        return false;
    }
    page_count <= dense_decode_max_pages.max(1)
}

fn qwen35_dense_decode_policy(device: &Device) -> (bool, usize) {
    let enabled = qwen35_use_dense_decode_attention_feature(device);
    let max_pages = qwen35_dense_decode_max_pages();
    (enabled, max_pages)
}

fn qwen35_use_dense_decode_attention_feature(device: &Device) -> bool {
    device.is_metal() && qwen35_env_bool("IZWI_QWEN35_DENSE_DECODE_ATTENTION", true)
}

fn append_dense_kv_cache_h(cache: &mut Option<Tensor>, append: &Tensor) -> Result<()> {
    if append.dim(2)? == 0 {
        return Ok(());
    }
    let append = append.contiguous()?;
    let next = match cache {
        Some(existing) => {
            let existing_ref: &Tensor = &*existing;
            Tensor::cat(&[existing_ref, &append], 2)?
        }
        None => append.clone(),
    };
    *cache = Some(compact_tensor_storage(&next)?);
    Ok(())
}

fn maybe_materialize_dense_kv_pages(
    page_size: usize,
    quantization: KvCacheQuantization,
    dense_decode_max_tokens: usize,
    k_pages: &mut Vec<KvPage>,
    v_pages: &mut Vec<KvPage>,
    dense_kv_tokens: &mut usize,
    dense_k_cache_h: &mut Option<Tensor>,
    dense_v_cache_h: &mut Option<Tensor>,
) -> Result<()> {
    if !k_pages.is_empty() || !v_pages.is_empty() {
        return Ok(());
    }

    let Some(k_cache_h) = dense_k_cache_h.as_ref() else {
        return Ok(());
    };
    let Some(v_cache_h) = dense_v_cache_h.as_ref() else {
        return Ok(());
    };
    if *dense_kv_tokens == 0 {
        return Ok(());
    }
    if *dense_kv_tokens <= dense_decode_max_tokens {
        return Ok(());
    }
    if k_cache_h.dim(2)? != *dense_kv_tokens || v_cache_h.dim(2)? != *dense_kv_tokens {
        return Err(Error::InferenceError(format!(
            "Qwen3.5 dense KV cache token mismatch: tracked={}, k={}, v={}",
            *dense_kv_tokens,
            k_cache_h.dim(2)?,
            v_cache_h.dim(2)?
        )));
    }

    let k_cache_h = dense_k_cache_h.take().ok_or_else(|| {
        Error::InferenceError("Qwen3.5 missing dense key cache during page migration".to_string())
    })?;
    let v_cache_h = dense_v_cache_h.take().ok_or_else(|| {
        Error::InferenceError("Qwen3.5 missing dense value cache during page migration".to_string())
    })?;
    let k_dense = k_cache_h.transpose(1, 2)?.contiguous()?;
    let v_dense = v_cache_h.transpose(1, 2)?.contiguous()?;
    append_to_pages(page_size, k_pages, &k_dense, quantization)?;
    append_to_pages(page_size, v_pages, &v_dense, quantization)?;
    *dense_kv_tokens = 0;
    Ok(())
}

fn qwen35_page_count_for_tokens(seq_len: usize, page_size: usize) -> Result<usize> {
    if page_size == 0 {
        return Err(Error::InvalidInput(
            "Qwen3.5 KV page size must be greater than zero".to_string(),
        ));
    }
    Ok(seq_len.div_ceil(page_size))
}

fn qwen35_dense_decode_max_pages() -> usize {
    if let Ok(raw) = std::env::var("IZWI_QWEN35_DENSE_DECODE_MAX_PAGES") {
        if let Ok(parsed) = raw.trim().parse::<usize>() {
            return parsed.max(1);
        }
    }

    // Adaptive default:
    // - dense decode stays active longer on compressed KV modes where memory pressure is lower.
    // - callers can still force explicit behavior with IZWI_QWEN35_DENSE_DECODE_MAX_PAGES.
    let base_pages = 12usize;
    let adaptive_pages = match default_kv_quantization() {
        KvCacheQuantization::None => base_pages,
        KvCacheQuantization::Int8 => base_pages.saturating_mul(2),
        KvCacheQuantization::Q4_0 => base_pages.saturating_mul(3),
    };
    let cap = std::env::var("IZWI_QWEN35_DENSE_DECODE_MAX_PAGES_CAP")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(64)
        .max(1);
    adaptive_pages.min(cap).max(1)
}

fn recurrent_gated_delta_sequence(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: Tensor,
) -> Result<(Tensor, Tensor)> {
    let seq_len = query.dim(1)?;
    let mut outputs = Vec::with_capacity(seq_len);
    let mut state = state;

    for idx in 0..seq_len {
        let q_t = query.narrow(1, idx, 1)?.squeeze(1)?;
        let k_t = key.narrow(1, idx, 1)?.squeeze(1)?;
        let v_t = value.narrow(1, idx, 1)?.squeeze(1)?;
        let g_t = g.narrow(1, idx, 1)?.squeeze(1)?;
        let beta_t = beta.narrow(1, idx, 1)?.squeeze(1)?;

        let (output_t, next_state) = recurrent_gated_delta(&q_t, &k_t, &v_t, &g_t, &beta_t, state)?;
        outputs.push(output_t.unsqueeze(1)?);
        state = next_state;
    }

    let output_refs: Vec<&Tensor> = outputs.iter().collect();
    let output = Tensor::cat(&output_refs, 1)?;
    Ok((output, state))
}

fn recurrent_gated_delta(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: Tensor,
) -> Result<(Tensor, Tensor)> {
    // Try fused Metal kernel first (for F32 on Metal devices)
    if query.dtype() == DType::F32 {
        if let Some(result) = try_fused_gated_delta_recurrent(query, key, value, g, beta, &state) {
            return Ok(result);
        }
    }

    // Optimized implementation using matmul for batched reductions.
    // Shapes: query/key (1, H, Dk), value (1, H, Dv), state (1, H, Dk, Dv)
    // g (1, H), beta (1, H)
    let dim = query.dim(D::Minus1)?;
    let scale = 1.0 / (dim as f64).sqrt();
    let query = (query * scale)?;
    let g = g.exp()?.reshape((1, g.dim(1)?, 1, 1))?;
    let beta = beta.reshape((1, beta.dim(1)?, 1))?;

    // Gate the state: state = state * exp(g)
    let state = state.broadcast_mul(&g)?;

    // kv_mem = sum(state * key[..., None], dim=2) = matmul(key[:, :, None, :], state).squeeze(2)
    // key: (1, H, Dk) -> (1, H, 1, Dk)  matmul  state: (1, H, Dk, Dv) -> (1, H, 1, Dv) -> squeeze -> (1, H, Dv)
    let kv_mem = key.unsqueeze(2)?.matmul(&state)?.squeeze(2)?;

    // delta = (value - kv_mem) * beta
    let delta = (value - &kv_mem)?.broadcast_mul(&beta)?;

    // state += key[:, :, :, None] * delta[:, :, None, :]  (outer product)
    // = matmul(key.unsqueeze(3), delta.unsqueeze(2)) + state
    let state = (&state + &key.unsqueeze(3)?.matmul(&delta.unsqueeze(2)?)?)?;

    // output = sum(state * query[..., None], dim=2) = matmul(query[:, :, None, :], state).squeeze(2)
    let output = query.unsqueeze(2)?.matmul(&state)?.squeeze(2)?;
    Ok((output, state))
}

#[cfg(test)]
mod tests {
    use super::{
        append_dense_kv_cache_h, apply_rotary_emb, build_mrope,
        full_attention_state_has_cached_prefix, non_finite_counts, owned_zero_tensor,
        qwen35_dense_decode_max_pages, qwen35_page_count_for_tokens, repeat_head_states,
        repeat_head_states_seq, softplus, unfused_causal_attention, ConvRingState, KvPage,
        Qwen35LayerRuntimeState, Qwen35TextRuntimeState,
    };
    use candle_core::{DType, Device, IndexOp, Tensor};
    use candle_nn::rotary_emb;
    use std::collections::HashSet;
    use std::sync::{Arc, Barrier};

    fn tensor_storage_address(tensor: &Tensor) -> usize {
        let (storage, _) = tensor.storage_and_layout();
        std::ptr::from_ref(&*storage) as usize
    }

    #[test]
    fn owned_recurrent_states_do_not_alias_across_concurrent_sessions() {
        const SESSION_COUNT: usize = 8;
        let barrier = Arc::new(Barrier::new(SESSION_COUNT));
        let handles = (0..SESSION_COUNT)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    owned_zero_tensor(&[1, 2, 4, 4], DType::F32, &Device::Cpu)
                        .expect("persistent recurrent state should allocate")
                })
            })
            .collect::<Vec<_>>();
        let states = handles
            .into_iter()
            .map(|handle| handle.join().expect("allocation thread should finish"))
            .collect::<Vec<_>>();

        let storage_addresses = states
            .iter()
            .map(tensor_storage_address)
            .collect::<HashSet<_>>();
        assert_eq!(storage_addresses.len(), SESSION_COUNT);
        assert!(states.iter().all(|state| state.dims() == [1, 2, 4, 4]));
        assert!(states.iter().all(|state| {
            state
                .flatten_all()
                .and_then(|state| state.to_vec1::<f32>())
                .is_ok_and(|values| values.iter().all(|value| *value == 0.0))
        }));
    }

    #[cfg(feature = "metal")]
    #[test]
    fn owned_recurrent_states_do_not_alias_on_metal() {
        let Ok(Ok(device)) = std::panic::catch_unwind(|| Device::new_metal(0)) else {
            return;
        };
        let first = owned_zero_tensor(&[1, 2, 4, 4], DType::F32, &device)
            .expect("first Metal recurrent state should allocate");
        let second = owned_zero_tensor(&[1, 2, 4, 4], DType::F32, &device)
            .expect("second Metal recurrent state should allocate");

        assert_ne!(
            tensor_storage_address(&first),
            tensor_storage_address(&second)
        );
        assert_eq!(
            first.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0.0; 32]
        );
        assert_eq!(
            second.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0.0; 32]
        );
    }

    #[test]
    fn repeat_head_states_uses_tiled_order() {
        let x = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 2, 2), &Device::Cpu)
            .expect("tensor should build");
        let repeated = repeat_head_states(&x, 2).expect("repeat should succeed");
        let values = repeated.to_vec3::<f32>().expect("values");

        assert_eq!(
            values,
            vec![vec![
                vec![1.0, 2.0],
                vec![3.0, 4.0],
                vec![1.0, 2.0],
                vec![3.0, 4.0]
            ]]
        );
    }

    #[test]
    fn repeat_head_states_seq_uses_tiled_order() {
        let x = Tensor::from_vec(
            vec![
                // seq 0
                1f32, 2.0, 3.0, 4.0, // seq 1
                5.0, 6.0, 7.0, 8.0,
            ],
            (1, 2, 2, 2),
            &Device::Cpu,
        )
        .expect("tensor should build");

        let repeated = repeat_head_states_seq(&x, 2).expect("repeat should succeed");
        let values = repeated
            .reshape((1, 2, 8))
            .expect("reshape")
            .to_vec3::<f32>()
            .expect("values");

        assert_eq!(
            values,
            vec![vec![
                vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
                vec![5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0]
            ]]
        );
    }

    #[test]
    fn compact_conv_ring_drops_full_prefill_backing() {
        let backing = Tensor::zeros((1, 40, 32), DType::F32, &Device::Cpu).unwrap();
        let mut slots = Vec::new();
        for token_idx in 37..40 {
            slots.push(backing.i((0, token_idx)).unwrap().reshape((32, 1)).unwrap());
        }
        let mut state = Qwen35TextRuntimeState {
            layers: vec![Qwen35LayerRuntimeState::Linear {
                conv_state: Some(ConvRingState { slots, next_idx: 0 }),
                recurrent_state: None,
            }],
        };

        let retained_prefill_bytes = state.allocated_session_bytes().unwrap();
        assert!(retained_prefill_bytes >= 40 * 32 * 4);

        let Qwen35LayerRuntimeState::Linear { conv_state, .. } = &mut state.layers[0] else {
            unreachable!("test state is linear")
        };
        conv_state
            .as_mut()
            .unwrap()
            .compact_owned()
            .expect("compaction should succeed");

        assert_eq!(state.allocated_session_bytes(), Some(3 * 32 * 4));
    }

    #[test]
    fn persistent_zero_states_have_independent_storage() {
        let first = owned_zero_tensor(&[1, 2, 3, 4], DType::F32, &Device::Cpu).unwrap();
        let second = owned_zero_tensor(&[1, 2, 3, 4], DType::F32, &Device::Cpu).unwrap();
        let state = Qwen35TextRuntimeState {
            layers: vec![
                Qwen35LayerRuntimeState::Linear {
                    conv_state: None,
                    recurrent_state: Some(first),
                },
                Qwen35LayerRuntimeState::Linear {
                    conv_state: None,
                    recurrent_state: Some(second),
                },
            ],
        };

        assert_eq!(state.allocated_session_bytes(), Some(2 * 2 * 3 * 4 * 4));
    }

    #[test]
    fn unfused_causal_prefill_and_cached_chunk_match_incremental_steps() {
        let device = Device::Cpu;
        let query = Tensor::from_vec(
            vec![
                0.2f32, 0.1, 0.3, -0.2, 0.4, 0.5, -0.1, 0.6, // head 0
                -0.3, 0.2, 0.7, 0.1, 0.2, -0.4, 0.5, 0.3, // head 1
            ],
            (1, 2, 4, 2),
            &device,
        )
        .unwrap();
        let key = Tensor::from_vec(
            vec![0.1f32, 0.3, 0.4, -0.2, 0.6, 0.5, -0.1, 0.7],
            (1, 1, 4, 2),
            &device,
        )
        .unwrap();
        let value = Tensor::from_vec(
            vec![1.0f32, 10.0, 2.0, 20.0, 4.0, 40.0, 8.0, 80.0],
            (1, 1, 4, 2),
            &device,
        )
        .unwrap();

        let prefill =
            unfused_causal_attention(&query, &key, &value, 2, 1, 2).expect("causal prefill");
        let rectangular =
            unfused_causal_attention(&query.narrow(2, 2, 2).unwrap(), &key, &value, 2, 1, 2)
                .expect("rectangular prefix attention");
        let expected_rectangular = prefill.narrow(2, 2, 2).unwrap();
        let rectangular_values = rectangular.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected_values = expected_rectangular
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (actual, expected) in rectangular_values.iter().zip(expected_values.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }

        let mut steps = Vec::new();
        for token_idx in 0..4 {
            let q = query.narrow(2, token_idx, 1).unwrap();
            let k = key.narrow(2, 0, token_idx + 1).unwrap();
            let v = value.narrow(2, 0, token_idx + 1).unwrap();
            steps.push(
                unfused_causal_attention(&q, &k, &v, 2, 1, 2).expect("incremental attention"),
            );
        }
        let step_refs: Vec<&Tensor> = steps.iter().collect();
        let incremental = Tensor::cat(&step_refs, 2).unwrap();
        let cached_chunk =
            unfused_causal_attention(&query.narrow(2, 2, 2).unwrap(), &key, &value, 2, 1, 2)
                .expect("rectangular cached chunk");

        let prefill_values = prefill.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let incremental_values = incremental.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (prefill, incremental) in prefill_values.iter().zip(incremental_values.iter()) {
            assert!((prefill - incremental).abs() < 1e-5);
        }

        let expected_chunk = prefill.narrow(2, 2, 2).unwrap();
        let expected_values = expected_chunk
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let cached_values = cached_chunk
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (expected, cached) in expected_values.iter().zip(cached_values.iter()) {
            assert!((expected - cached).abs() < 1e-5);
        }
    }

    #[test]
    fn dense_paged_and_tracked_full_attention_state_count_as_prefixes() {
        let dense = Tensor::zeros((1, 1, 2, 2), DType::F32, &Device::Cpu).unwrap();
        let dense_state = Qwen35LayerRuntimeState::Full {
            k_pages: Vec::new(),
            v_pages: Vec::new(),
            dense_k_cache_h: Some(dense),
            dense_v_cache_h: None,
            dense_kv_tokens: 0,
        };
        assert!(full_attention_state_has_cached_prefix(&dense_state).unwrap());

        let page = KvPage::Dense(Tensor::zeros((1, 2, 1, 2), DType::F32, &Device::Cpu).unwrap());
        let paged_state = Qwen35LayerRuntimeState::Full {
            k_pages: vec![page],
            v_pages: Vec::new(),
            dense_k_cache_h: None,
            dense_v_cache_h: None,
            dense_kv_tokens: 0,
        };
        assert!(full_attention_state_has_cached_prefix(&paged_state).unwrap());

        let tracked_state = Qwen35LayerRuntimeState::Full {
            k_pages: Vec::new(),
            v_pages: Vec::new(),
            dense_k_cache_h: None,
            dense_v_cache_h: None,
            dense_kv_tokens: 2,
        };
        assert!(full_attention_state_has_cached_prefix(&tracked_state).unwrap());

        let empty = Qwen35LayerRuntimeState::Full {
            k_pages: Vec::new(),
            v_pages: Vec::new(),
            dense_k_cache_h: None,
            dense_v_cache_h: None,
            dense_kv_tokens: 0,
        };
        assert!(!full_attention_state_has_cached_prefix(&empty).unwrap());
    }

    #[test]
    fn softplus_is_f32_and_stable_for_large_magnitudes() {
        let input = Tensor::from_vec(vec![-1000f32, 0.0, 1000.0], 3, &Device::Cpu).unwrap();
        let output = softplus(&input).expect("softplus");
        assert_eq!(output.dtype(), DType::F32);
        let values = output.to_vec1::<f32>().unwrap();
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values[0].abs() < 1e-6);
        assert!((values[1] - std::f32::consts::LN_2).abs() < 1e-6);
        assert!((values[2] - 1000.0).abs() < 1e-4);
    }

    #[test]
    fn non_finite_diagnostics_count_without_exposing_values() {
        let tensor = Tensor::from_vec(
            vec![0.0f32, f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
            4,
            &Device::Cpu,
        )
        .unwrap();
        let counts = non_finite_counts(&tensor).unwrap();
        assert_eq!(counts.nan, 1);
        assert_eq!(counts.positive_infinity, 1);
        assert_eq!(counts.negative_infinity, 1);
        assert_eq!(counts.total(), 3);
    }

    #[test]
    fn dense_decode_max_pages_adapts_to_kv_quantization() {
        let _guard = crate::env_test_lock().lock().expect("env lock");

        std::env::remove_var("IZWI_QWEN35_DENSE_DECODE_MAX_PAGES");
        std::env::remove_var("IZWI_QWEN35_DENSE_DECODE_MAX_PAGES_CAP");

        std::env::remove_var("IZWI_KV_CACHE_DTYPE");
        assert_eq!(qwen35_dense_decode_max_pages(), 12);

        std::env::set_var("IZWI_KV_CACHE_DTYPE", "int8");
        assert_eq!(qwen35_dense_decode_max_pages(), 24);

        std::env::set_var("IZWI_KV_CACHE_DTYPE", "q4_0");
        assert_eq!(qwen35_dense_decode_max_pages(), 36);

        std::env::set_var("IZWI_QWEN35_DENSE_DECODE_MAX_PAGES_CAP", "20");
        assert_eq!(qwen35_dense_decode_max_pages(), 20);

        std::env::set_var("IZWI_QWEN35_DENSE_DECODE_MAX_PAGES", "7");
        assert_eq!(qwen35_dense_decode_max_pages(), 7);

        std::env::remove_var("IZWI_QWEN35_DENSE_DECODE_MAX_PAGES");
        std::env::remove_var("IZWI_QWEN35_DENSE_DECODE_MAX_PAGES_CAP");
        std::env::remove_var("IZWI_KV_CACHE_DTYPE");
    }

    #[test]
    fn build_mrope_uses_half_dim_layout_and_sections() {
        let (cos, sin) = build_mrope(
            12,
            [3, 5, 7],
            &[2, 2, 2],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            &Device::Cpu,
            DType::F32,
        )
        .expect("mrope should build");

        assert_eq!(cos.dims(), &[1, 1, 6]);
        assert_eq!(sin.dims(), &[1, 1, 6]);

        let cos_vals = cos.to_vec3::<f32>().expect("cos values");
        let sin_vals = sin.to_vec3::<f32>().expect("sin values");
        let expected = [3.0f32, 5.0, 7.0, 3.0, 5.0, 7.0];
        for (idx, expected_theta) in expected.iter().enumerate() {
            assert!((cos_vals[0][0][idx] - expected_theta.cos()).abs() < 1e-5);
            assert!((sin_vals[0][0][idx] - expected_theta.sin()).abs() < 1e-5);
        }
    }

    #[test]
    fn rotary_emb_manual_matches_rope_thd() {
        let x = Tensor::from_vec(
            (0..(1 * 3 * 2 * 8))
                .map(|v| v as f32 / 10.0)
                .collect::<Vec<_>>(),
            (1, 3, 2, 8),
            &Device::Cpu,
        )
        .expect("x");
        let theta = [
            0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
        ];
        let cos = Tensor::from_vec(
            theta.iter().map(|v| v.cos()).collect::<Vec<_>>(),
            (1, 3, 4),
            &Device::Cpu,
        )
        .expect("cos");
        let sin = Tensor::from_vec(
            theta.iter().map(|v| v.sin()).collect::<Vec<_>>(),
            (1, 3, 4),
            &Device::Cpu,
        )
        .expect("sin");

        let manual = apply_rotary_emb(&x, &cos, &sin).expect("manual");
        let kernel = rotary_emb::rope_thd(&x, &cos, &sin).expect("kernel");

        let manual_vals = manual
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("manual vals");
        let kernel_vals = kernel
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("kernel vals");
        assert_eq!(manual_vals.len(), kernel_vals.len());
        for (manual, kernel) in manual_vals.iter().zip(kernel_vals.iter()) {
            assert!((manual - kernel).abs() < 1e-5);
        }
    }

    #[test]
    fn dense_kv_cache_head_major_appends_sequence_along_token_axis() {
        let device = Device::Cpu;
        let mut cache: Option<Tensor> = None;
        let first = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 2, 1, 2), &device).unwrap();
        let second = Tensor::from_vec(vec![5f32, 6.0, 7.0, 8.0], (1, 2, 1, 2), &device).unwrap();

        append_dense_kv_cache_h(&mut cache, &first).expect("append first");
        append_dense_kv_cache_h(&mut cache, &second).expect("append second");

        let cache = cache.expect("cache should exist");
        assert_eq!(cache.dims(), &[1, 2, 2, 2]);
        let flat = cache.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(flat, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn qwen35_page_count_uses_ceil_division() {
        assert_eq!(qwen35_page_count_for_tokens(0, 64).unwrap(), 0);
        assert_eq!(qwen35_page_count_for_tokens(1, 64).unwrap(), 1);
        assert_eq!(qwen35_page_count_for_tokens(64, 64).unwrap(), 1);
        assert_eq!(qwen35_page_count_for_tokens(65, 64).unwrap(), 2);
        assert!(qwen35_page_count_for_tokens(10, 0).is_err());
    }
}
