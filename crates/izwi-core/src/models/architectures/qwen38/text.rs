use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{ops, rotary_emb, Embedding, Linear};
use safetensors::Dtype as SafeDType;

use super::cache::{CONVOLUTION_STATE_DOMAIN, RECURRENT_STATE_DOMAIN};
use super::chat::Qwen38TextConfig;
use super::native::{
    IndexedSafetensors, ProjectionMaterialization, Qwen38LayerType, Qwen38NativeConfig,
};
use super::telemetry::{
    record_cuda_attention_dtype_casts, record_cuda_head_expansion_materialization,
    record_cuda_kernel, record_cuda_projection, record_cuda_rope,
    record_cuda_state_initial_allocation, CudaKernelPath, CudaProjectionPath,
};
use crate::backends::state::{
    PhysicalStateSequenceId, PhysicalStateTransactionId, StateComponentValue, TensorStateArena,
};
use crate::error::{Error, Result};
use crate::kernels::{
    try_fused_gated_delta_recurrent, try_fused_gated_rms_norm, try_fused_l2_norm,
    try_fused_silu_mul, try_qwen38_causal_conv_decode, try_qwen38_causal_conv_sequence,
    try_tiled_deltanet_recurrence,
};
use crate::kv::v2::{StateComponentId, StateDomainId};
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PreparedPhysicalPagedStep};
use crate::models::shared::memory::accounting::{
    deep_copy_tensor_storage, TensorStorageAccounting,
};
use crate::models::shared::telemetry::{
    record_prefill_sequence_span, record_rope_kernel, record_rope_manual,
};

pub struct Qwen38TextModel {
    device: Device,
    projection_representation: Qwen38ProjectionRepresentation,
    token_embeddings: Embedding,
    layers: Vec<Qwen38Layer>,
    output_norm: Qwen38RmsNorm,
    output: Qwen38Projection,
    finite_diagnostics_enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen38ProjectionRepresentation {
    ExpandedF32,
    ExpandedF16,
    ExpandedBf16,
    PackedQ8WithDenseBf16,
}

impl Qwen38ProjectionRepresentation {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ExpandedF32 => "expanded_f32",
            Self::ExpandedF16 => "expanded_f16",
            Self::ExpandedBf16 => "expanded_bf16",
            Self::PackedQ8WithDenseBf16 => "q8_0_requantized_projections_with_dense_bf16",
        }
    }
}

#[derive(Clone)]
pub struct Qwen38TextRuntimeState {
    layers: Vec<Qwen38LayerRuntimeState>,
}

impl Qwen38TextRuntimeState {
    /// Backing allocations retained by the per-request text runtime state.
    ///
    /// This intentionally excludes model-global caches (notably full-attention
    /// RoPE windows), so callers requiring a complete scheduler claim must keep
    /// Qwen3.8 fail-closed until those caches are independently bounded.
    pub fn allocated_session_bytes(&self) -> Option<u64> {
        let mut accounting = TensorStorageAccounting::default();
        self.account_storage(&mut accounting)?;
        Some(accounting.bytes())
    }

    pub(crate) fn account_storage(&self, accounting: &mut TensorStorageAccounting) -> Option<()> {
        for layer in &self.layers {
            match layer {
                Qwen38LayerRuntimeState::Linear {
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
                Qwen38LayerRuntimeState::Full => {}
            }
        }
        Some(())
    }

    pub(crate) fn restore_tensor_domains(
        &mut self,
        arena: &TensorStateArena,
        sequence: PhysicalStateSequenceId,
    ) -> Result<()> {
        let recurrent = arena.read(sequence, recurrent_domain_v2())?;
        let convolution = arena.read(sequence, convolution_domain_v2())?;
        if recurrent.is_none() && convolution.is_none() {
            return Ok(());
        }
        let recurrent = recurrent.ok_or_else(|| {
            Error::InferenceError("Qwen3.8 recurrent state is missing its convolution peer".into())
        })?;
        let convolution = convolution.ok_or_else(|| {
            Error::InferenceError("Qwen3.8 convolution state is missing its recurrent peer".into())
        })?;
        let mut recurrent_components = recurrent.components.iter();
        let mut convolution_components = convolution.components.iter();
        for layer in &mut self.layers {
            let Qwen38LayerRuntimeState::Linear {
                conv_state,
                recurrent_state,
            } = layer
            else {
                continue;
            };
            let recurrent = recurrent_components.next().ok_or_else(|| {
                Error::InferenceError("Qwen3.8 recurrent component coverage is incomplete".into())
            })?;
            let convolution = convolution_components.next().ok_or_else(|| {
                Error::InferenceError("Qwen3.8 convolution component coverage is incomplete".into())
            })?;
            let recurrent_tensor = recurrent.tensor.as_ref().ok_or_else(|| {
                Error::InferenceError("Qwen3.8 recurrent component is absent".into())
            })?;
            let convolution_tensor = convolution.tensor.as_ref().ok_or_else(|| {
                Error::InferenceError("Qwen3.8 convolution component is absent".into())
            })?;
            *recurrent_state = Some(recurrent_tensor.clone());
            let history_len = convolution_tensor.dim(0)?;
            let slots = (0..history_len)
                .map(|index| convolution_tensor.i(index).map_err(Error::from))
                .collect::<Result<Vec<_>>>()?;
            *conv_state = Some(ConvRingState { slots, next_idx: 0 });
        }
        if recurrent_components.next().is_some() || convolution_components.next().is_some() {
            return Err(Error::InferenceError(
                "Qwen3.8 tensor state has components for unknown layers".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn stage_tensor_domains(
        &mut self,
        arena: &TensorStateArena,
        transaction: PhysicalStateTransactionId,
        target_cursor: u64,
    ) -> Result<()> {
        let recurrent_cursor = arena
            .read_transaction_base(transaction, recurrent_domain_v2())?
            .map(|snapshot| snapshot.cursor)
            .unwrap_or(0);
        let convolution_cursor = arena
            .read_transaction_base(transaction, convolution_domain_v2())?
            .map(|snapshot| snapshot.cursor)
            .unwrap_or(0);
        let mut recurrent = Vec::new();
        let mut convolution = Vec::new();
        for layer in &self.layers {
            let Qwen38LayerRuntimeState::Linear {
                conv_state,
                recurrent_state,
            } = layer
            else {
                continue;
            };
            let recurrent_tensor = recurrent_state.as_ref().ok_or_else(|| {
                Error::InferenceError("Qwen3.8 recurrent state was not initialized".into())
            })?;
            let ring = conv_state.as_ref().ok_or_else(|| {
                Error::InferenceError("Qwen3.8 convolution state was not initialized".into())
            })?;
            if ring.slots.is_empty() || ring.next_idx >= ring.slots.len() {
                return Err(Error::InferenceError(
                    "Qwen3.8 convolution ring is invalid at the physical boundary".into(),
                ));
            }
            let ordered = (0..ring.slots.len())
                .map(|offset| &ring.slots[(ring.next_idx + offset) % ring.slots.len()])
                .collect::<Vec<_>>();
            let ring_tensor = Tensor::stack(&ordered, 0)?;
            let component = u32::try_from(recurrent.len() + 1)
                .map_err(|_| Error::InvalidInput("Qwen3.8 state component overflow".into()))?;
            recurrent.push(StateComponentValue {
                component: StateComponentId::new(component),
                tensor: Some(recurrent_tensor.clone()),
            });
            convolution.push(StateComponentValue {
                component: StateComponentId::new(component),
                tensor: Some(ring_tensor),
            });
        }
        arena.stage_replace(
            transaction,
            recurrent_domain_v2(),
            recurrent_cursor,
            target_cursor,
            recurrent,
        )?;
        arena.stage_replace(
            transaction,
            convolution_domain_v2(),
            convolution_cursor,
            target_cursor,
            convolution,
        )?;
        // The arena now owns the only retained handles. Keep the decode state
        // as control metadata between quanta so engine abort cannot expose a
        // partially drained model state.
        for layer in &mut self.layers {
            if let Qwen38LayerRuntimeState::Linear {
                conv_state,
                recurrent_state,
            } = layer
            {
                *conv_state = None;
                *recurrent_state = None;
            }
        }
        Ok(())
    }
}

fn recurrent_domain_v2() -> StateDomainId {
    RECURRENT_STATE_DOMAIN
}

fn convolution_domain_v2() -> StateDomainId {
    CONVOLUTION_STATE_DOMAIN
}

#[derive(Clone)]
struct ConvRingState {
    slots: Vec<Tensor>,
    next_idx: usize,
}

impl ConvRingState {
    /// Move every logical history slot into independent fixed-history storage.
    ///
    /// Sequence-prefill slots are views into the entire projected token span.
    /// Keeping those views in runtime state would retain the full projection
    /// instead of the fixed `kernel_size - 1` history required by the conv.
    fn compact_owned(&mut self) -> Result<()> {
        if self.slots.is_empty() {
            return Ok(());
        }

        self.slots = self
            .slots
            .iter()
            .map(deep_copy_tensor_storage)
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(())
    }

    /// Retain the current one-token projection as the newest decode slot.
    ///
    /// Decode projects exactly one token, so this view cannot retain a
    /// sequence-sized backing tensor. Sequence prefill detaches its final fixed
    /// history once at the sequence boundary instead.
    fn push_decode(&mut self, current: &Tensor) -> Result<()> {
        if self.slots.is_empty() || self.next_idx >= self.slots.len() {
            return Err(Error::InferenceError(format!(
                "Invalid Qwen3.8 convolution ring: slots={}, next_idx={}",
                self.slots.len(),
                self.next_idx
            )));
        }
        self.slots[self.next_idx] = current.clone();
        self.next_idx = (self.next_idx + 1) % self.slots.len();
        Ok(())
    }
}

#[derive(Clone)]
enum Qwen38LayerRuntimeState {
    Linear {
        conv_state: Option<ConvRingState>,
        recurrent_state: Option<Tensor>,
    },
    Full,
}

struct Qwen38Layer {
    attn_norm: Qwen38RmsNorm,
    mixer: Qwen38Mixer,
    post_attention_norm: Qwen38RmsNorm,
    mlp: Qwen38Mlp,
}

enum Qwen38Mixer {
    Linear(Qwen38LinearAttention),
    Full(Qwen38FullAttention),
}

struct Qwen38Mlp {
    gate_up: Qwen38ProjectionGroup,
    down: Qwen38Projection,
}

enum Qwen38Projection {
    Quantized(QMatMul),
    Dense(Linear),
}

enum Qwen38ProjectionGroup {
    Separate(Vec<Qwen38Projection>),
    Packed {
        projection: Qwen38Projection,
        widths: Vec<usize>,
    },
}

struct Qwen38RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl Module for Qwen38RmsNorm {
    fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        candle_nn::ops::rms_norm(input, &self.weight, self.eps as f32)
    }
}

impl Module for Qwen38Projection {
    fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        if input.device().is_cuda() {
            record_cuda_projection(match self {
                Self::Quantized(_) => CudaProjectionPath::Q8,
                Self::Dense(_) => CudaProjectionPath::Dense,
            });
        }
        match self {
            Self::Quantized(projection) => projection.forward(input),
            Self::Dense(projection) => projection.forward(input),
        }
    }
}

impl Qwen38ProjectionGroup {
    fn forward(&self, input: &Tensor) -> Result<Vec<Tensor>> {
        match self {
            Self::Separate(projections) => projections
                .iter()
                .map(|projection| projection.forward(input).map_err(Error::from))
                .collect(),
            Self::Packed { projection, widths } => {
                let output = projection.forward(input)?;
                split_projection_output(&output, widths)
            }
        }
    }
}

struct Qwen38FullAttention {
    qkv_proj: Qwen38ProjectionGroup,
    o_proj: Qwen38Projection,
    q_norm: Qwen38RmsNorm,
    k_norm: Qwen38RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    rope_theta: f64,
    mrope_sections: Vec<usize>,
    rope_kernel_enabled: bool,
    rope_inv_freqs: Vec<f32>,
}

struct Qwen38LinearAttention {
    qkv_z_proj: Qwen38ProjectionGroup,
    alpha_beta_proj: Qwen38ProjectionGroup,
    dt_bias: Tensor,
    a: Tensor,
    conv_kernel: Tensor,
    conv_kernel_slices: Vec<Tensor>,
    norm: Qwen38GatedRmsNorm,
    out_proj: Qwen38Projection,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    conv_dim: usize,
    kernel_size: usize,
    tiled_recurrence_enabled: bool,
    tiled_recurrence_tile_size_override: Option<usize>,
}

struct Qwen38GatedRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl Qwen38TextModel {
    pub fn load_native(
        tensors: &IndexedSafetensors,
        native: &Qwen38NativeConfig,
        device: &Device,
    ) -> Result<Self> {
        let cfg = &native.text;
        let target = native_projection_target(device);
        let projection_representation = native_projection_representation(device, target);
        let block = native.block_fp8.block_shape;
        let embedding_weights = tensors.materialize_dense_tensor(
            "model.language_model.embed_tokens.weight",
            &[native.vocab_size, cfg.embedding_length],
            target,
            device,
        )?;
        let token_embeddings = Embedding::new(embedding_weights, cfg.embedding_length);
        let output_norm = load_native_zero_centered_norm(
            tensors,
            "model.language_model.norm.weight",
            cfg.embedding_length,
            cfg.attention_layer_norm_rms_epsilon,
            target,
            device,
        )?;
        let output = load_native_projection(
            tensors,
            "lm_head.weight",
            [native.vocab_size, cfg.embedding_length],
            block,
            target,
            device,
        )?;

        let mut layers = Vec::with_capacity(cfg.block_count);
        for (layer_idx, layer_type) in native.layer_types.iter().copied().enumerate() {
            let prefix = format!("model.language_model.layers.{layer_idx}");
            let attn_norm = load_native_zero_centered_norm(
                tensors,
                &format!("{prefix}.input_layernorm.weight"),
                cfg.embedding_length,
                cfg.attention_layer_norm_rms_epsilon,
                target,
                device,
            )?;
            let post_attention_norm = load_native_zero_centered_norm(
                tensors,
                &format!("{prefix}.post_attention_layernorm.weight"),
                cfg.embedding_length,
                cfg.attention_layer_norm_rms_epsilon,
                target,
                device,
            )?;
            let mlp = Qwen38Mlp::load_native(tensors, device, &prefix, cfg, block, target)?;
            let mixer = match layer_type {
                Qwen38LayerType::FullAttention => Qwen38Mixer::Full(
                    Qwen38FullAttention::load_native(tensors, device, &prefix, cfg, block, target)?,
                ),
                Qwen38LayerType::LinearAttention => {
                    Qwen38Mixer::Linear(Qwen38LinearAttention::load_native(
                        tensors, device, &prefix, cfg, block, target,
                    )?)
                }
            };
            layers.push(Qwen38Layer {
                attn_norm,
                mixer,
                post_attention_norm,
                mlp,
            });
        }
        Ok(Self {
            device: device.clone(),
            projection_representation,
            token_embeddings,
            layers,
            output_norm,
            output,
            finite_diagnostics_enabled: qwen38_env_bool("IZWI_QWEN38_FINITE_DIAGNOSTICS", false),
        })
    }

    pub fn new_state(&self) -> Qwen38TextRuntimeState {
        Qwen38TextRuntimeState {
            layers: self.layers.iter().map(Qwen38Layer::new_state).collect(),
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.token_embeddings.hidden_size()
    }

    pub fn projection_representation(&self) -> Qwen38ProjectionRepresentation {
        self.projection_representation
    }

    pub(crate) fn forward_token_id_at_physical(
        &self,
        token_id: u32,
        position_ids: [usize; 3],
        state: &mut Qwen38TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Tensor> {
        let input = Tensor::from_vec(vec![token_id], (1, 1), &self.device)?;
        let hidden = self.token_embeddings.forward(&input)?;
        let hidden = self.forward_hidden_physical(&hidden, &[position_ids], state, cache)?;
        self.forward_hidden_to_logits(&hidden)
    }

    pub(crate) fn prefill_token_ids_physical(
        &self,
        token_ids: &[u32],
        position_ids: &[[usize; 3]],
        state: &mut Qwen38TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
        compute_logits: bool,
    ) -> Result<Option<Tensor>> {
        if token_ids.is_empty() {
            return Ok(None);
        }
        if token_ids.len() != position_ids.len() {
            return Err(Error::InvalidInput(format!(
                "Qwen3.8 physical prefill span mismatch: {} token ids for {} position ids",
                token_ids.len(),
                position_ids.len()
            )));
        }
        record_prefill_sequence_span(token_ids.len());
        let input = Tensor::from_vec(token_ids.to_vec(), (1, token_ids.len()), &self.device)?;
        let hidden = self.token_embeddings.forward(&input)?;
        let hidden = self.forward_hidden_physical(&hidden, position_ids, state, cache)?;
        if !compute_logits {
            return Ok(None);
        }
        let last = hidden.narrow(1, token_ids.len() - 1, 1)?;
        self.forward_hidden_to_logits(&last).map(Some)
    }

    pub(crate) fn forward_input_embedding_at_physical(
        &self,
        input_embedding: &Tensor,
        position_ids: [usize; 3],
        state: &mut Qwen38TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Tensor> {
        let hidden =
            self.forward_hidden_physical(input_embedding, &[position_ids], state, cache)?;
        self.forward_hidden_to_logits(&hidden)
    }

    fn forward_hidden_physical(
        &self,
        input: &Tensor,
        position_ids: &[[usize; 3]],
        state: &mut Qwen38TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Tensor> {
        self.validate_runtime_state(state)?;
        let (_, sequence_len, hidden_size) = input.dims3()?;
        if sequence_len == 0
            || sequence_len != position_ids.len()
            || hidden_size != self.hidden_size()
        {
            return Err(Error::InvalidInput(
                "Qwen3.8 physical hidden span does not match its positions or model width".into(),
            ));
        }
        let sparse_layers = self
            .layers
            .iter()
            .enumerate()
            .filter_map(|(index, layer)| {
                matches!(layer.mixer, Qwen38Mixer::Full(_)).then_some(index as u32)
            })
            .collect::<Vec<_>>();
        let first_full = self.layers.iter().find_map(|layer| match &layer.mixer {
            Qwen38Mixer::Full(attention) => Some(attention),
            Qwen38Mixer::Linear(_) => None,
        });
        let first_full = first_full.ok_or_else(|| {
            Error::InferenceError("Qwen3.8 model has no full-attention layer".into())
        })?;
        cache.validate_sparse_model(
            &sparse_layers,
            first_full.num_kv_heads,
            first_full.head_dim,
            first_full.head_dim,
        )?;
        let start_pos = cache.context_len();
        let mut prepared = cache.prepare_append(start_pos, sequence_len)?;
        for (layer, layer_state) in self.layers.iter().zip(state.layers.iter_mut()) {
            layer.ensure_state_initialized(layer_state, &self.device)?;
        }
        let mut hidden = input.clone();
        let mut physical_layer = 0usize;
        for (layer_index, (layer, layer_state)) in
            self.layers.iter().zip(state.layers.iter_mut()).enumerate()
        {
            hidden = layer.forward_physical(
                &hidden,
                layer_state,
                position_ids,
                cache,
                &mut prepared,
                &mut physical_layer,
            )?;
            validate_qwen38_finite_tensor(
                &hidden,
                layer_index,
                if sequence_len == 1 {
                    layer.decode_diagnostic_path()
                } else {
                    layer.prefill_diagnostic_path()
                },
                self.finite_diagnostics_enabled,
            )?;
        }
        if physical_layer != sparse_layers.len() {
            return Err(Error::InferenceError(
                "Qwen3.8 physical attention did not cover every sparse layer".into(),
            ));
        }
        cache.commit_prepared(prepared)?;
        Ok(hidden)
    }

    pub fn forward_hidden_to_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let hidden = self.output_norm.forward(hidden)?;
        validate_qwen38_finite_tensor(
            &hidden,
            self.layers.len(),
            "output.norm",
            self.finite_diagnostics_enabled,
        )?;
        let logits = self.output.forward(&hidden)?;
        let logits = logits.i((0, 0))?;
        validate_qwen38_finite_tensor(
            &logits,
            self.layers.len(),
            "output.logits",
            self.finite_diagnostics_enabled,
        )?;
        Ok(logits)
    }

    fn validate_runtime_state(&self, state: &Qwen38TextRuntimeState) -> Result<()> {
        if state.layers.len() != self.layers.len() {
            return Err(Error::InferenceError(format!(
                "Qwen3.8 runtime state layer mismatch: state has {}, model has {}",
                state.layers.len(),
                self.layers.len()
            )));
        }
        Ok(())
    }
}

impl Qwen38Layer {
    fn decode_diagnostic_path(&self) -> &'static str {
        match self.mixer {
            Qwen38Mixer::Linear(_) => "decode.linear_layer_output",
            Qwen38Mixer::Full(_) => "decode.full_attention_layer_output",
        }
    }

    fn prefill_diagnostic_path(&self) -> &'static str {
        match self.mixer {
            Qwen38Mixer::Linear(_) => "prefill.linear_layer_output",
            Qwen38Mixer::Full(_) => "prefill.full_attention_layer_output",
        }
    }

    fn new_state(&self) -> Qwen38LayerRuntimeState {
        match self.mixer {
            Qwen38Mixer::Linear(_) => Qwen38LayerRuntimeState::Linear {
                conv_state: None,
                recurrent_state: None,
            },
            Qwen38Mixer::Full(_) => Qwen38LayerRuntimeState::Full,
        }
    }

    /// Pre-initialize lazy state tensors so the first-use allocation cost
    /// does not happen inside the per-token hot loop during prefill.
    fn ensure_state_initialized(
        &self,
        state: &mut Qwen38LayerRuntimeState,
        device: &Device,
    ) -> Result<()> {
        if let (
            Qwen38Mixer::Linear(mixer),
            Qwen38LayerRuntimeState::Linear {
                conv_state,
                recurrent_state,
            },
        ) = (&self.mixer, state)
        {
            if conv_state.is_none() && mixer.kernel_size > 1 {
                // Persistent runtime state cannot outlive a released scratch-pool
                // lease. Independent exact-size slots allow O(1) replacement
                // without retaining dead regions of a shared backing buffer.
                let history_len = mixer.kernel_size - 1;
                let mut slots = Vec::with_capacity(history_len);
                for _ in 0..history_len {
                    slots.push(owned_zero_tensor(&[mixer.conv_dim, 1], DType::F32, device)?);
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
            // Full attention layers don't need pre-initialization.
        }
        Ok(())
    }

    fn forward_physical(
        &self,
        hidden_states: &Tensor,
        state: &mut Qwen38LayerRuntimeState,
        position_ids: &[[usize; 3]],
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        physical_layer: &mut usize,
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let normalized = self.attn_norm.forward(hidden_states)?;
        let mixed = match &self.mixer {
            Qwen38Mixer::Linear(mixer) => {
                if normalized.dim(1)? == 1 {
                    mixer.forward(&normalized, state)?
                } else {
                    mixer.forward_sequence(&normalized, state)?
                }
            }
            Qwen38Mixer::Full(mixer) => {
                let output = mixer.forward_physical(
                    &normalized,
                    position_ids,
                    cache,
                    prepared,
                    *physical_layer,
                )?;
                *physical_layer = physical_layer.checked_add(1).ok_or_else(|| {
                    Error::InvalidInput("Qwen3.8 physical layer ordinal overflow".into())
                })?;
                output
            }
        };
        let hidden_states = (&residual + &mixed)?;
        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_norm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        (&residual + &hidden_states).map_err(Error::from)
    }
}

impl Qwen38Mlp {
    fn load_native(
        tensors: &IndexedSafetensors,
        device: &Device,
        prefix: &str,
        cfg: &Qwen38TextConfig,
        block: [usize; 2],
        target: ProjectionMaterialization,
    ) -> Result<Self> {
        let gate_name = format!("{prefix}.mlp.gate_proj.weight");
        let up_name = format!("{prefix}.mlp.up_proj.weight");
        Ok(Self {
            gate_up: load_native_projection_group(
                tensors,
                &[
                    (&gate_name, [cfg.feed_forward_length, cfg.embedding_length]),
                    (&up_name, [cfg.feed_forward_length, cfg.embedding_length]),
                ],
                block,
                target,
                device,
            )?,
            down: load_native_projection(
                tensors,
                &format!("{prefix}.mlp.down_proj.weight"),
                [cfg.embedding_length, cfg.feed_forward_length],
                block,
                target,
                device,
            )?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Use fused SiLU-gate-up if available (reduces memory bandwidth)
        let mut gate_up = self.gate_up.forward(hidden_states)?.into_iter();
        let gate_proj_out = gate_up.next().ok_or_else(|| {
            Error::InferenceError("Qwen3.8 gate/up projection omitted gate output".into())
        })?;
        let up_proj_out = gate_up.next().ok_or_else(|| {
            Error::InferenceError("Qwen3.8 gate/up projection omitted up output".into())
        })?;

        let fused = try_fused_silu_mul(&gate_proj_out, &up_proj_out);
        if gate_proj_out.device().is_cuda() {
            record_cuda_kernel(CudaKernelPath::SiluMul, fused.is_some());
        }
        let hidden = if let Some(fused) = fused {
            fused
        } else {
            let gate = ops::silu(&gate_proj_out)?;
            (&gate * &up_proj_out)?
        };

        self.down.forward(&hidden).map_err(Error::from)
    }
}

impl Qwen38FullAttention {
    fn load_native(
        tensors: &IndexedSafetensors,
        device: &Device,
        prefix: &str,
        cfg: &Qwen38TextConfig,
        block: [usize; 2],
        target: ProjectionMaterialization,
    ) -> Result<Self> {
        let q_width = cfg
            .attention_head_count
            .checked_mul(cfg.attention_key_length)
            .and_then(|width| width.checked_mul(2))
            .ok_or_else(|| Error::ModelLoadError("Qwen3.8 Q projection width overflow".into()))?;
        let kv_width = cfg
            .attention_head_count_kv
            .checked_mul(cfg.attention_key_length)
            .ok_or_else(|| Error::ModelLoadError("Qwen3.8 KV projection width overflow".into()))?;
        let q_name = format!("{prefix}.self_attn.q_proj.weight");
        let k_name = format!("{prefix}.self_attn.k_proj.weight");
        let v_name = format!("{prefix}.self_attn.v_proj.weight");
        Ok(Self {
            qkv_proj: load_native_projection_group(
                tensors,
                &[
                    (&q_name, [q_width, cfg.embedding_length]),
                    (&k_name, [kv_width, cfg.embedding_length]),
                    (&v_name, [kv_width, cfg.embedding_length]),
                ],
                block,
                target,
                device,
            )?,
            o_proj: load_native_projection(
                tensors,
                &format!("{prefix}.self_attn.o_proj.weight"),
                [
                    cfg.embedding_length,
                    cfg.attention_head_count * cfg.attention_value_length,
                ],
                block,
                target,
                device,
            )?,
            q_norm: load_native_zero_centered_norm(
                tensors,
                &format!("{prefix}.self_attn.q_norm.weight"),
                cfg.attention_key_length,
                cfg.attention_layer_norm_rms_epsilon,
                target,
                device,
            )?,
            k_norm: load_native_zero_centered_norm(
                tensors,
                &format!("{prefix}.self_attn.k_norm.weight"),
                cfg.attention_key_length,
                cfg.attention_layer_norm_rms_epsilon,
                target,
                device,
            )?,
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
            rope_kernel_enabled: qwen38_rope_kernel_enabled(device),
            rope_inv_freqs: build_rope_inv_freqs(
                cfg.rope_dimension_count.min(cfg.attention_key_length),
                cfg.rope_freq_base,
            )?,
        })
    }

    fn forward_physical(
        &self,
        hidden_states: &Tensor,
        position_ids: &[[usize; 3]],
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        physical_layer: usize,
    ) -> Result<Tensor> {
        let seq_len = hidden_states.dim(1)?;
        if seq_len == 0 || seq_len != position_ids.len() {
            return Err(Error::InvalidInput(format!(
                "Qwen3.8 physical attention received {} tokens and {} positions",
                seq_len,
                position_ids.len()
            )));
        }
        let mut qkv = self.qkv_proj.forward(hidden_states)?.into_iter();
        let q_proj = qkv
            .next()
            .ok_or_else(|| Error::InferenceError("Qwen3.8 QKV projection omitted Q".into()))?
            .reshape((1, seq_len, self.num_heads, self.head_dim * 2))?;
        let query_states = q_proj.narrow(3, 0, self.head_dim)?;
        let gate = q_proj.narrow(3, self.head_dim, self.head_dim)?.reshape((
            1,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;
        let key_states = qkv
            .next()
            .ok_or_else(|| Error::InferenceError("Qwen3.8 QKV projection omitted K".into()))?
            .reshape((1, seq_len, self.num_kv_heads, self.head_dim))?;
        let value_states = qkv
            .next()
            .ok_or_else(|| Error::InferenceError("Qwen3.8 QKV projection omitted V".into()))?
            .reshape((1, seq_len, self.num_kv_heads, self.head_dim))?;
        let query_states = self.q_norm.forward(&query_states.contiguous()?)?;
        let key_states = self.k_norm.forward(&key_states.contiguous()?)?;
        let (query_states, key_states) = if seq_len == 1 {
            self.apply_rope(&query_states, &key_states, position_ids[0])?
        } else {
            self.apply_rope_sequence(&query_states, &key_states, position_ids)?
        };
        let queries = query_states
            .reshape((seq_len, self.num_heads, self.head_dim))?
            .contiguous()?;
        let keys = key_states
            .reshape((seq_len, self.num_kv_heads, self.head_dim))?
            .contiguous()?;
        let values = value_states
            .reshape((seq_len, self.num_kv_heads, self.head_dim))?
            .contiguous()?;
        let storage_dtype = cache.arena().config().dtype;
        let output_dtype = queries.dtype();
        if queries.device().is_cuda() && storage_dtype != output_dtype {
            // Q, K, V enter storage dtype and the attention result returns to
            // the activation dtype.
            record_cuda_attention_dtype_casts(4);
        }
        let queries = queries.to_dtype(storage_dtype)?;
        let keys = keys.to_dtype(storage_dtype)?;
        let values = values.to_dtype(storage_dtype)?;
        let output = cache.write_and_attend(
            physical_layer,
            prepared,
            &queries,
            &keys,
            &values,
            1.0 / (self.head_dim as f32).sqrt(),
        )?;
        let output =
            output
                .to_dtype(output_dtype)?
                .reshape((1, seq_len, self.num_heads * self.head_dim))?;
        let output = (&output * &ops::sigmoid(&gate)?)?;
        self.o_proj.forward(&output).map_err(Error::from)
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
                    if query_states.device().is_cuda() {
                        record_cuda_rope(true);
                    }
                    (query_rot, key_rot)
                }
                None => {
                    record_rope_manual();
                    if query_states.device().is_cuda() {
                        record_cuda_rope(false);
                    }
                    (
                        apply_rotary_emb(&query_rot, &cos, &sin)?,
                        apply_rotary_emb(&key_rot, &cos, &sin)?,
                    )
                }
            }
        } else {
            record_rope_manual();
            if query_states.device().is_cuda() {
                record_cuda_rope(false);
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

    fn apply_rope_sequence(
        &self,
        query_states: &Tensor,
        key_states: &Tensor,
        position_ids: &[[usize; 3]],
    ) -> Result<(Tensor, Tensor)> {
        let seq_len = query_states.dim(1)?;
        if seq_len != position_ids.len() {
            return Err(Error::InvalidInput(format!(
                "Qwen3.8 rotary sequence mismatch: seq_len={}, position_ids={}",
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
                        if query_states.device().is_cuda() {
                            record_cuda_rope(true);
                        }
                    }
                    (query_rot, key_rot)
                }
                None => {
                    for _ in 0..seq_len {
                        record_rope_manual();
                        if query_states.device().is_cuda() {
                            record_cuda_rope(false);
                        }
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
                if query_states.device().is_cuda() {
                    record_cuda_rope(false);
                }
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
        if self.rope_dim == 0 || !self.rope_dim.is_multiple_of(2) {
            return false;
        }
        matches!(dtype, DType::F16 | DType::BF16 | DType::F32)
    }
}

impl Qwen38LinearAttention {
    fn load_native(
        tensors: &IndexedSafetensors,
        device: &Device,
        prefix: &str,
        cfg: &Qwen38TextConfig,
        block: [usize; 2],
        target: ProjectionMaterialization,
    ) -> Result<Self> {
        let num_k_heads = cfg.ssm_group_count;
        let num_v_heads = cfg.ssm_time_step_rank;
        let head_k_dim = cfg.ssm_state_size;
        let head_v_dim = cfg.ssm_inner_size / cfg.ssm_time_step_rank;
        let conv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
        let linear = format!("{prefix}.linear_attn");
        let dt_bias = tensors
            .materialize_dense_tensor(
                &format!("{linear}.dt_bias"),
                &[num_v_heads],
                ProjectionMaterialization::F32,
                device,
            )?
            .reshape((1, 1, num_v_heads))?;
        let a_log = tensors.materialize_dense_tensor(
            &format!("{linear}.A_log"),
            &[num_v_heads],
            ProjectionMaterialization::F32,
            device,
        )?;
        let a = a_log.exp()?.neg()?.reshape((1, 1, num_v_heads))?;
        let conv_kernel = normalize_conv_kernel(
            tensors.materialize_dense_tensor(
                &format!("{linear}.conv1d.weight"),
                &[conv_dim, 1, cfg.ssm_conv_kernel],
                ProjectionMaterialization::F32,
                device,
            )?,
            conv_dim,
            cfg.ssm_conv_kernel,
        )?;
        let conv_kernel_slices = pre_slice_conv_kernel(&conv_kernel, cfg.ssm_conv_kernel)?;
        let norm = Qwen38GatedRmsNorm {
            weight: tensors.materialize_dense_tensor(
                &format!("{linear}.norm.weight"),
                &[head_v_dim],
                ProjectionMaterialization::F32,
                device,
            )?,
            eps: cfg.attention_layer_norm_rms_epsilon,
        };
        let qkv_name = format!("{linear}.in_proj_qkv.weight");
        let z_name = format!("{linear}.in_proj_z.weight");
        let alpha_name = format!("{linear}.in_proj_a.weight");
        let beta_name = format!("{linear}.in_proj_b.weight");
        Ok(Self {
            qkv_z_proj: load_native_projection_group(
                tensors,
                &[
                    (&qkv_name, [conv_dim, cfg.embedding_length]),
                    (&z_name, [cfg.ssm_inner_size, cfg.embedding_length]),
                ],
                block,
                target,
                device,
            )?,
            alpha_beta_proj: load_native_projection_group(
                tensors,
                &[
                    (&alpha_name, [num_v_heads, cfg.embedding_length]),
                    (&beta_name, [num_v_heads, cfg.embedding_length]),
                ],
                block,
                target,
                device,
            )?,
            dt_bias,
            a,
            conv_kernel,
            conv_kernel_slices,
            norm,
            out_proj: load_native_projection(
                tensors,
                &format!("{linear}.out_proj.weight"),
                [cfg.embedding_length, cfg.ssm_inner_size],
                block,
                target,
                device,
            )?,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_dim,
            kernel_size: cfg.ssm_conv_kernel,
            tiled_recurrence_enabled: qwen38_tiled_recurrence_enabled(),
            tiled_recurrence_tile_size_override: qwen38_tiled_recurrence_tile_size_override(),
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        state: &mut Qwen38LayerRuntimeState,
    ) -> Result<Tensor> {
        let (conv_state, recurrent_state) = match state {
            Qwen38LayerRuntimeState::Linear {
                conv_state,
                recurrent_state,
            } => (conv_state, recurrent_state),
            _ => {
                return Err(Error::InferenceError(
                    "Qwen3.8 layer runtime state does not match linear-attention layer".to_string(),
                ))
            }
        };

        let residual_dtype = hidden_states.dtype();
        let mut qkv_z = self.qkv_z_proj.forward(hidden_states)?.into_iter();
        let mixed_qkv = required_group_output(&mut qkv_z, "DeltaNet QKV")?.to_dtype(DType::F32)?;
        let z = required_group_output(&mut qkv_z, "DeltaNet Z")?.to_dtype(DType::F32)?;
        let mut alpha_beta = self.alpha_beta_proj.forward(hidden_states)?.into_iter();
        let alpha =
            required_group_output(&mut alpha_beta, "DeltaNet alpha")?.to_dtype(DType::F32)?;
        let beta = ops::sigmoid(
            &required_group_output(&mut alpha_beta, "DeltaNet beta")?.to_dtype(DType::F32)?,
        )?;
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
            if query.device().is_cuda() {
                // Repeat-interleave materializes one expanded Q and K tensor.
                record_cuda_head_expansion_materialization();
                record_cuda_head_expansion_materialization();
            }
            query = repeat_interleave_head_states(&query, repeats)?;
            key = repeat_interleave_head_states(&key, repeats)?;
        }

        let current_state = if let Some(state) = recurrent_state.take() {
            state
        } else {
            if value.device().is_cuda() {
                record_cuda_state_initial_allocation();
            }
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
        // The recurrent decode output owns a fresh Candle-managed allocation;
        // retaining it avoids a full state-sized copy on every layer/token.
        *recurrent_state = Some(next_state);

        let output = output.reshape((self.num_v_heads, self.head_v_dim))?;
        let z = z.reshape((self.num_v_heads, self.head_v_dim))?;
        let output = self.norm.forward(&output, &z)?;
        let output = output
            .reshape((1, 1, self.num_v_heads * self.head_v_dim))?
            .to_dtype(residual_dtype)?;
        self.out_proj.forward(&output).map_err(Error::from)
    }

    fn forward_sequence(
        &self,
        hidden_states: &Tensor,
        state: &mut Qwen38LayerRuntimeState,
    ) -> Result<Tensor> {
        let seq_len = hidden_states.dim(1)?;
        if seq_len == 1 {
            return self.forward(hidden_states, state);
        }

        let (conv_state, recurrent_state) = match state {
            Qwen38LayerRuntimeState::Linear {
                conv_state,
                recurrent_state,
            } => (conv_state, recurrent_state),
            _ => {
                return Err(Error::InferenceError(
                    "Qwen3.8 layer runtime state does not match linear-attention layer".to_string(),
                ))
            }
        };

        let residual_dtype = hidden_states.dtype();
        let mut qkv_z = self.qkv_z_proj.forward(hidden_states)?.into_iter();
        let mixed_qkv = required_group_output(&mut qkv_z, "DeltaNet QKV")?.to_dtype(DType::F32)?;
        let z = required_group_output(&mut qkv_z, "DeltaNet Z")?.to_dtype(DType::F32)?;
        let mut alpha_beta = self.alpha_beta_proj.forward(hidden_states)?.into_iter();
        let alpha =
            required_group_output(&mut alpha_beta, "DeltaNet alpha")?.to_dtype(DType::F32)?;
        let beta = ops::sigmoid(
            &required_group_output(&mut alpha_beta, "DeltaNet beta")?.to_dtype(DType::F32)?,
        )?;
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
            if value.device().is_cuda() {
                record_cuda_state_initial_allocation();
            }
            Tensor::zeros(
                (1, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                value.dtype(),
                value.device(),
            )?
        };

        let beta = beta.reshape((1, seq_len, self.num_v_heads))?;
        let g = g.reshape((1, seq_len, self.num_v_heads))?;
        let tile_size =
            qwen38_tiled_recurrence_tile_size(seq_len, self.tiled_recurrence_tile_size_override);
        let fused_sequence = if self.tiled_recurrence_enabled {
            let fused = try_tiled_deltanet_recurrence(
                &query,
                &key,
                &value,
                &g,
                &beta,
                &current_state,
                tile_size,
            );
            if query.device().is_cuda() {
                record_cuda_kernel(CudaKernelPath::DeltaNetPrefill, fused.is_some());
            }
            fused
        } else {
            None
        };
        let (output, next_state) = if let Some(fused_sequence) = fused_sequence {
            fused_sequence
        } else {
            // The portable recurrence consumes one Q/K head per value head.
            // Native Qwen3.8 stores 16 Q/K heads and 48 value heads, so expand
            // with Transformers' repeat-interleave ordering before fallback.
            let (query, key) = if self.num_v_heads == self.num_k_heads {
                (query, key)
            } else {
                let repeats = self.num_v_heads / self.num_k_heads;
                if query.device().is_cuda() {
                    record_cuda_head_expansion_materialization();
                    record_cuda_head_expansion_materialization();
                }
                (
                    repeat_interleave_head_states_seq(&query, repeats)?,
                    repeat_interleave_head_states_seq(&key, repeats)?,
                )
            };
            if self.tiled_recurrence_enabled {
                let fused_sequence = try_tiled_deltanet_recurrence(
                    &query,
                    &key,
                    &value,
                    &g,
                    &beta,
                    &current_state,
                    tile_size,
                );
                if query.device().is_cuda() {
                    record_cuda_kernel(CudaKernelPath::DeltaNetPrefill, fused_sequence.is_some());
                }
                if let Some(fused_sequence) = fused_sequence {
                    fused_sequence
                } else {
                    recurrent_gated_delta_sequence(&query, &key, &value, &g, &beta, current_state)?
                }
            } else {
                recurrent_gated_delta_sequence(&query, &key, &value, &g, &beta, current_state)?
            }
        };
        // Sequence kernels may return the final state as a view into their
        // packed output. Detach only this fixed-size escape value, using
        // Candle-managed storage rather than an application private buffer.
        *recurrent_state = Some(deep_copy_tensor_storage(&next_state)?);

        let output = output.reshape((seq_len * self.num_v_heads, self.head_v_dim))?;
        let z = z.reshape((seq_len * self.num_v_heads, self.head_v_dim))?;
        let output = self.norm.forward(&output, &z)?;
        let output = output
            .reshape((1, seq_len, self.num_v_heads * self.head_v_dim))?
            .to_dtype(residual_dtype)?;
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
                    "Invalid Qwen3.8 convolution history: slots={}, next_idx={}, expected_slots={history_len}",
                    buffer.slots.len(),
                    buffer.next_idx
                )));
            }
            let logical_slots: Vec<&Tensor> = (0..history_len)
                .map(|idx| &buffer.slots[(buffer.next_idx + idx) % history_len])
                .collect();
            let history = Tensor::cat(&logical_slots, 1)?;
            let fused = try_qwen38_causal_conv_sequence(mixed_qkv, &self.conv_kernel, &history);
            if mixed_qkv.device().is_cuda() {
                record_cuda_kernel(CudaKernelPath::CausalConvPrefill, fused.is_some());
            }
            if let Some((output, final_history)) = fused {
                let final_history = deep_copy_tensor_storage(&final_history)?;
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
        if self.kernel_size == 4
            && mixed_qkv.device().is_cuda()
            && qwen38_env_bool("IZWI_QWEN38_CAUSAL_CONV_DECODE", false)
        {
            let buffer = conv_state.as_mut().ok_or_else(|| {
                Error::InferenceError(
                    "conv_state not initialized but Qwen3.8 decode convolution needs history"
                        .to_string(),
                )
            })?;
            if buffer.slots.len() != 3 || buffer.next_idx >= 3 {
                return Err(Error::InferenceError(format!(
                    "Invalid Qwen3.8 decode convolution history: slots={}, next_idx={}",
                    buffer.slots.len(),
                    buffer.next_idx
                )));
            }
            let logical_slots: Vec<&Tensor> = (0..3)
                .map(|idx| &buffer.slots[(buffer.next_idx + idx) % 3])
                .collect();
            let history = Tensor::cat(&logical_slots, 1)?;
            let fused = try_qwen38_causal_conv_decode(mixed_qkv, &self.conv_kernel, &history);
            record_cuda_kernel(CudaKernelPath::CausalConvDecode, fused.is_some());
            if let Some((output, next_history)) = fused {
                buffer.slots = (0..3)
                    .map(|idx| next_history.narrow(1, idx, 1))
                    .collect::<candle_core::Result<Vec<_>>>()?;
                buffer.next_idx = 0;
                return Ok(output);
            }
        }

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
            buffer.push_decode(&current)?;

            convolved.squeeze(1)?
        };

        let convolved = ops::silu(&convolved)?;
        convolved
            .reshape((1, 1, self.conv_dim))
            .map_err(Error::from)
    }
}

impl Qwen38GatedRmsNorm {
    fn forward(&self, hidden_states: &Tensor, gate: &Tensor) -> Result<Tensor> {
        if hidden_states.dtype() == DType::F32 {
            let fused = try_fused_gated_rms_norm(hidden_states, gate, &self.weight, self.eps);
            if hidden_states.device().is_cuda() {
                record_cuda_kernel(CudaKernelPath::GatedRmsNorm, fused.is_some());
            }
            if let Some(result) = fused {
                return Ok(result);
            }
        }

        let normalized = candle_nn::ops::rms_norm(hidden_states, &self.weight, self.eps as f32)?;
        (&normalized * &ops::silu(gate)?).map_err(Error::from)
    }
}

fn native_projection_target(device: &Device) -> ProjectionMaterialization {
    if device.is_cpu() {
        ProjectionMaterialization::F32
    } else if device.is_metal() {
        ProjectionMaterialization::F16
    } else {
        ProjectionMaterialization::BF16
    }
}

fn native_projection_representation(
    device: &Device,
    target: ProjectionMaterialization,
) -> Qwen38ProjectionRepresentation {
    if device.is_cuda() {
        return Qwen38ProjectionRepresentation::PackedQ8WithDenseBf16;
    }
    match target {
        ProjectionMaterialization::F32 => Qwen38ProjectionRepresentation::ExpandedF32,
        ProjectionMaterialization::F16 => Qwen38ProjectionRepresentation::ExpandedF16,
        ProjectionMaterialization::BF16 => Qwen38ProjectionRepresentation::ExpandedBf16,
    }
}

fn load_native_projection(
    tensors: &IndexedSafetensors,
    name: &str,
    shape: [usize; 2],
    block: [usize; 2],
    target: ProjectionMaterialization,
    device: &Device,
) -> Result<Qwen38Projection> {
    if device.is_cuda() && tensors.tensor_info(name)?.dtype == SafeDType::F8_E4M3 {
        return tensors
            .materialize_q8_projection(name, shape, block, device)
            .map(Qwen38Projection::Quantized);
    }
    tensors
        .materialize_projection(name, shape, block, target, device)
        .map(|weight| Qwen38Projection::Dense(Linear::new(weight, None)))
}

fn load_native_projection_group(
    tensors: &IndexedSafetensors,
    projections: &[(&str, [usize; 2])],
    block: [usize; 2],
    target: ProjectionMaterialization,
    device: &Device,
) -> Result<Qwen38ProjectionGroup> {
    let load_separate = || {
        projections
            .iter()
            .map(|(name, shape)| {
                load_native_projection(tensors, name, *shape, block, target, device)
            })
            .collect::<Result<Vec<_>>>()
            .map(Qwen38ProjectionGroup::Separate)
    };
    if !qwen38_projection_packing_enabled(device)
        || !projection_group_geometry_compatible(projections)
    {
        return load_separate();
    }

    let all_fp8 = projections.iter().all(|(name, _)| {
        tensors
            .tensor_info(name)
            .is_ok_and(|info| info.dtype == SafeDType::F8_E4M3)
    });
    let any_fp8 = projections.iter().any(|(name, _)| {
        tensors
            .tensor_info(name)
            .is_ok_and(|info| info.dtype == SafeDType::F8_E4M3)
    });
    // Do not merge different persistent representations. The separate path
    // remains the safe fallback for unusual checkpoint variants.
    if any_fp8 && !all_fp8 {
        return load_separate();
    }

    let projection = if device.is_cuda() && all_fp8 {
        tensors
            .materialize_q8_projection_group(projections, block, device)
            .map(Qwen38Projection::Quantized)?
    } else {
        tensors
            .materialize_projection_group(projections, block, target, device)
            .map(|weight| Qwen38Projection::Dense(Linear::new(weight, None)))?
    };
    Ok(Qwen38ProjectionGroup::Packed {
        projection,
        widths: projections.iter().map(|(_, shape)| shape[0]).collect(),
    })
}

fn projection_group_geometry_compatible(projections: &[(&str, [usize; 2])]) -> bool {
    let Some((_, first)) = projections.first() else {
        return false;
    };
    projections
        .iter()
        .all(|(_, shape)| shape[0] > 0 && shape[1] == first[1])
}

fn split_projection_output(output: &Tensor, widths: &[usize]) -> Result<Vec<Tensor>> {
    let output_axis = output.rank().checked_sub(1).ok_or_else(|| {
        Error::InferenceError("Qwen3.8 projection output cannot be scalar".into())
    })?;
    let output_width = output.dim(output_axis)?;
    let expected_width = widths.iter().try_fold(0usize, |sum, width| {
        sum.checked_add(*width)
            .ok_or_else(|| Error::InferenceError("Qwen3.8 projection split width overflow".into()))
    })?;
    if output_width != expected_width {
        return Err(Error::InferenceError(format!(
            "Qwen3.8 packed projection produced width {output_width}, expected {expected_width}"
        )));
    }
    let mut offset = 0usize;
    widths
        .iter()
        .map(|width| {
            let split = output
                .narrow(output_axis, offset, *width)
                .map_err(Error::from)?;
            offset += *width;
            Ok(split)
        })
        .collect()
}

fn required_group_output(outputs: &mut std::vec::IntoIter<Tensor>, label: &str) -> Result<Tensor> {
    outputs.next().ok_or_else(|| {
        Error::InferenceError(format!("Qwen3.8 packed projection omitted {label} output"))
    })
}

fn load_native_zero_centered_norm(
    tensors: &IndexedSafetensors,
    name: &str,
    length: usize,
    eps: f64,
    target: ProjectionMaterialization,
    device: &Device,
) -> Result<Qwen38RmsNorm> {
    let weight = tensors.materialize_dense_tensor(name, &[length], target, device)?;
    let weight = (weight + 1.0)?;
    Ok(Qwen38RmsNorm { weight, eps })
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
                    "Unexpected Qwen3.8 conv kernel shape: ({d0}, {d1}) for expected ({expected_channels}, {expected_kernel})"
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
                    "Unexpected rank-3 Qwen3.8 conv kernel shape: {:?}",
                    dims
                )))
            }
        }
        rank => Err(Error::ModelLoadError(format!(
            "Unexpected Qwen3.8 conv kernel rank {rank}"
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
            "Invalid Qwen3.8 rotary dimension {rope_dim}"
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
                "Invalid Qwen3.8 multimodal RoPE sections {:?} for rotary dim {}",
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
            "Invalid Qwen3.8 rotary dimension {rope_dim}"
        )));
    }
    Ok(inv_freqs)
}

/// Allocate persistent runtime state with independent backing storage.
///
/// Candle tensor views can outlive a scratch-pool checkout, so state that
/// survives a forward call must never escape after its pool lease is released.
fn owned_zero_tensor(shape: &[usize], dtype: DType, device: &Device) -> Result<Tensor> {
    Tensor::zeros(shape.to_vec(), dtype, device).map_err(Error::from)
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

fn validate_qwen38_finite_tensor(
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
        "Qwen3.8 first non-finite tensor at {path}, layer {layer_idx}: \
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
        let fused = try_fused_l2_norm(x, eps);
        if x.device().is_cuda() {
            record_cuda_kernel(CudaKernelPath::L2Norm, fused.is_some());
        }
        if let Some(result) = fused {
            return Ok(result);
        }
    }

    // Fallback to standard implementation
    x.broadcast_div(&(x.sqr()?.sum_keepdim(D::Minus1)? + eps)?.sqrt()?)
        .map_err(Error::from)
}

fn repeat_interleave_head_states(x: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats <= 1 {
        return Ok(x.clone());
    }
    let (batch, heads, dim) = x.dims3()?;
    x.unsqueeze(2)?
        .broadcast_as((batch, heads, repeats, dim))?
        .reshape((batch, heads * repeats, dim))
        .map_err(Error::from)
}

fn repeat_interleave_head_states_seq(x: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats <= 1 {
        return Ok(x.clone());
    }
    let (batch, seq, heads, dim) = x.dims4()?;
    x.unsqueeze(3)?
        .broadcast_as((batch, seq, heads, repeats, dim))?
        .reshape((batch, seq, heads * repeats, dim))
        .map_err(Error::from)
}

fn qwen38_env_bool(name: &str, default: bool) -> bool {
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

fn qwen38_projection_packing_enabled(device: &Device) -> bool {
    qwen38_projection_packing_policy(
        device.is_cuda(),
        qwen38_env_bool("IZWI_QWEN38_PACKED_PROJECTIONS", false),
    )
}

fn qwen38_projection_packing_policy(is_cuda: bool, requested: bool) -> bool {
    is_cuda && requested
}

fn qwen38_tiled_recurrence_enabled() -> bool {
    qwen38_env_bool("IZWI_QWEN38_TILED_RECURRENCE", true)
}

fn qwen38_rope_kernel_enabled(device: &Device) -> bool {
    let override_enabled = std::env::var("IZWI_QWEN38_ROPE_KERNEL")
        .ok()
        .and_then(|raw| match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        });
    qwen38_rope_kernel_policy(device.is_metal(), device.is_cuda(), override_enabled)
}

fn qwen38_rope_kernel_policy(
    is_metal: bool,
    is_cuda: bool,
    override_enabled: Option<bool>,
) -> bool {
    if is_metal {
        return override_enabled.unwrap_or(true);
    }
    if is_cuda {
        return override_enabled.unwrap_or(true);
    }
    false
}

fn qwen38_tiled_recurrence_tile_size_override() -> Option<usize> {
    if let Ok(raw) = std::env::var("IZWI_QWEN38_TILED_RECURRENCE_TILE_SIZE") {
        if let Ok(parsed) = raw.trim().parse::<usize>() {
            return Some(parsed.max(1));
        }
    }
    None
}

fn qwen38_tiled_recurrence_tile_size(seq_len: usize, override_size: Option<usize>) -> usize {
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
        let fused = try_fused_gated_delta_recurrent(query, key, value, g, beta, &state);
        if query.device().is_cuda() {
            record_cuda_kernel(CudaKernelPath::DeltaNetDecode, fused.is_some());
        }
        if let Some(result) = fused {
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
        apply_rotary_emb, build_mrope, convolution_domain_v2, non_finite_counts, owned_zero_tensor,
        projection_group_geometry_compatible, qwen38_projection_packing_policy,
        qwen38_rope_kernel_policy, recurrent_domain_v2, repeat_interleave_head_states,
        repeat_interleave_head_states_seq, softplus, ConvRingState, Linear,
        Qwen38LayerRuntimeState, Qwen38Projection, Qwen38ProjectionGroup, Qwen38TextRuntimeState,
    };
    use crate::models::architectures::qwen38::cache::{
        CONVOLUTION_STATE_DOMAIN, RECURRENT_STATE_DOMAIN,
    };
    use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
    use candle_core::{DType, Device, IndexOp, Tensor};
    use candle_nn::rotary_emb;
    use std::collections::HashSet;
    use std::sync::{Arc, Barrier};

    fn tensor_storage_address(tensor: &Tensor) -> usize {
        let (storage, _) = tensor.storage_and_layout();
        std::ptr::from_ref(&*storage) as usize
    }

    #[test]
    fn retained_state_access_uses_the_contracts_canonical_domain_ids() {
        assert_eq!(recurrent_domain_v2(), RECURRENT_STATE_DOMAIN);
        assert_eq!(convolution_domain_v2(), CONVOLUTION_STATE_DOMAIN);
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
    fn native_head_repeat_uses_transformers_repeat_interleave_order() {
        let x = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 2, 2), &Device::Cpu)
            .expect("tensor should build");
        let repeated = repeat_interleave_head_states(&x, 2).expect("repeat should succeed");
        assert_eq!(
            repeated.to_vec3::<f32>().expect("values"),
            vec![vec![
                vec![1.0, 2.0],
                vec![1.0, 2.0],
                vec![3.0, 4.0],
                vec![3.0, 4.0]
            ]]
        );

        let sequence = x.unsqueeze(1).expect("sequence");
        let repeated =
            repeat_interleave_head_states_seq(&sequence, 2).expect("repeat should succeed");
        assert_eq!(
            repeated
                .reshape((1, 1, 8))
                .unwrap()
                .to_vec3::<f32>()
                .unwrap(),
            vec![vec![vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]]]
        );
    }

    #[test]
    fn dense_projection_supports_batched_sequence_inputs() {
        use candle_nn::Module;

        let projection = Qwen38Projection::Dense(Linear::new(
            Tensor::from_vec(vec![1f32, 0.0, 0.0, 1.0, 1.0, 1.0], (3, 2), &Device::Cpu).unwrap(),
            None,
        ));
        let input = Tensor::from_vec(vec![2f32, 3.0, 4.0, 5.0], (1, 2, 2), &Device::Cpu).unwrap();
        let output = projection.forward(&input).unwrap();
        assert_eq!(output.dims3().unwrap(), (1, 2, 3));
        assert_eq!(
            output.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![2.0, 3.0, 5.0, 4.0, 5.0, 9.0]
        );
    }

    #[test]
    fn packed_projection_group_matches_separate_outputs_exactly() {
        let first =
            Tensor::from_vec(vec![1f32, 0.0, 0.0, 1.0, 1.0, 1.0], (3, 2), &Device::Cpu).unwrap();
        let second = Tensor::from_vec(vec![2f32, -1.0, -2.0, 1.0], (2, 2), &Device::Cpu).unwrap();
        let packed = Tensor::cat(&[&first, &second], 0).unwrap();
        let separate = Qwen38ProjectionGroup::Separate(vec![
            Qwen38Projection::Dense(Linear::new(first, None)),
            Qwen38Projection::Dense(Linear::new(second, None)),
        ]);
        let packed = Qwen38ProjectionGroup::Packed {
            projection: Qwen38Projection::Dense(Linear::new(packed, None)),
            widths: vec![3, 2],
        };
        let input = Tensor::from_vec(vec![2f32, 3.0, 4.0, 5.0], (1, 2, 2), &Device::Cpu).unwrap();

        let separate = separate.forward(&input).unwrap();
        let packed = packed.forward(&input).unwrap();
        assert_eq!(packed.len(), separate.len());
        for (packed, separate) in packed.iter().zip(&separate) {
            assert_eq!(packed.dims(), separate.dims());
            assert_eq!(
                packed.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                separate.flatten_all().unwrap().to_vec1::<f32>().unwrap()
            );
        }
    }

    #[test]
    fn projection_packing_is_cuda_only_and_disabled_without_opt_in() {
        assert!(!qwen38_projection_packing_policy(false, false));
        assert!(!qwen38_projection_packing_policy(false, true));
        assert!(!qwen38_projection_packing_policy(true, false));
        assert!(qwen38_projection_packing_policy(true, true));
    }

    #[test]
    fn projection_group_rejects_incompatible_inner_dimensions() {
        assert!(projection_group_geometry_compatible(&[
            ("q", [3, 8]),
            ("k", [2, 8]),
            ("v", [2, 8]),
        ]));
        assert!(!projection_group_geometry_compatible(&[
            ("q", [3, 8]),
            ("k", [2, 4]),
        ]));
        assert!(!projection_group_geometry_compatible(&[]));
    }

    #[test]
    fn q8_projection_supports_model_rank_batched_sequence_inputs() {
        use candle_nn::Module;

        let mut weights = vec![1.0_f32; 32];
        weights.extend(std::iter::repeat_n(2.0, 32));
        weights.extend(std::iter::repeat_n(-1.0, 32));
        let weight = Tensor::from_vec(weights, (3, 32), &Device::Cpu).unwrap();
        let quantized = QTensor::quantize(&weight, GgmlDType::Q8_0).unwrap();
        let projection = Qwen38Projection::Quantized(QMatMul::QTensor(Arc::new(quantized)));
        let input = Tensor::ones((2, 3, 32), DType::F32, &Device::Cpu).unwrap();
        let output = projection.forward(&input).unwrap();
        assert_eq!(output.dims3().unwrap(), (2, 3, 3));
        for batch in output.to_vec3::<f32>().unwrap() {
            for token in batch {
                assert!((token[0] - 32.0).abs() < 0.5, "{} != 32", token[0]);
                assert!((token[1] - 64.0).abs() < 0.5, "{} != 64", token[1]);
                assert!((token[2] + 32.0).abs() < 0.5, "{} != -32", token[2]);
            }
        }
    }

    #[test]
    fn native_projection_inner_dimensions_satisfy_q8_block_geometry() {
        // Every projection K dimension is one of the model width, MLP width,
        // or attention/DeltaNet output width. The loader also validates each
        // concrete tensor before quantizing it.
        for inner_dim in [5_120_usize, 17_408, 6_144] {
            assert!(inner_dim.is_multiple_of(GgmlDType::Q8_0.block_size()));
        }
    }

    #[test]
    fn compact_conv_ring_drops_full_prefill_backing() {
        let backing = Tensor::zeros((1, 40, 32), DType::F32, &Device::Cpu).unwrap();
        let mut slots = Vec::new();
        for token_idx in 37..40 {
            slots.push(backing.i((0, token_idx)).unwrap().reshape((32, 1)).unwrap());
        }
        let mut state = Qwen38TextRuntimeState {
            layers: vec![Qwen38LayerRuntimeState::Linear {
                conv_state: Some(ConvRingState { slots, next_idx: 0 }),
                recurrent_state: None,
            }],
        };

        let retained_prefill_bytes = state.allocated_session_bytes().unwrap();
        assert!(retained_prefill_bytes >= 40 * 32 * 4);

        let Qwen38LayerRuntimeState::Linear { conv_state, .. } = &mut state.layers[0] else {
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
    fn conv_ring_decode_push_reuses_one_token_projection_backing() {
        let slots = (0..3)
            .map(|_| Tensor::zeros((32, 1), DType::F32, &Device::Cpu).unwrap())
            .collect();
        let mut ring = ConvRingState { slots, next_idx: 0 };
        let projection = Tensor::zeros((1, 1, 32), DType::F32, &Device::Cpu).unwrap();
        let current = projection.i((0, 0)).unwrap().reshape((32, 1)).unwrap();
        let projection_storage = tensor_storage_address(&current);

        ring.push_decode(&current)
            .expect("ring push should succeed");
        assert_eq!(tensor_storage_address(&ring.slots[0]), projection_storage);
        drop(current);
        drop(projection);

        let state = Qwen38TextRuntimeState {
            layers: vec![Qwen38LayerRuntimeState::Linear {
                conv_state: Some(ring),
                recurrent_state: None,
            }],
        };
        assert_eq!(state.allocated_session_bytes(), Some(3 * 32 * 4));
    }

    #[test]
    fn persistent_zero_states_have_independent_storage() {
        let first = owned_zero_tensor(&[1, 2, 3, 4], DType::F32, &Device::Cpu).unwrap();
        let second = owned_zero_tensor(&[1, 2, 3, 4], DType::F32, &Device::Cpu).unwrap();
        let state = Qwen38TextRuntimeState {
            layers: vec![
                Qwen38LayerRuntimeState::Linear {
                    conv_state: None,
                    recurrent_state: Some(first),
                },
                Qwen38LayerRuntimeState::Linear {
                    conv_state: None,
                    recurrent_state: Some(second),
                },
            ],
        };

        assert_eq!(state.allocated_session_bytes(), Some(2 * 2 * 3 * 4 * 4));
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
            (0..(3 * 2 * 8))
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
    fn qwen38_cuda_rope_kernel_defaults_on_with_explicit_rollback() {
        assert!(qwen38_rope_kernel_policy(true, false, None));
        assert!(qwen38_rope_kernel_policy(false, true, None));
        assert!(qwen38_rope_kernel_policy(false, true, Some(true)));
        assert!(!qwen38_rope_kernel_policy(true, false, Some(false)));
        assert!(!qwen38_rope_kernel_policy(false, false, Some(true)));
    }

    fn synthetic_decode_state(
        device: &Device,
        layer_count: usize,
        num_v_heads: usize,
    ) -> Qwen38TextRuntimeState {
        let mut layers = Vec::with_capacity(layer_count);
        for layer_idx in 0..layer_count {
            if (layer_idx + 1).is_multiple_of(4) {
                layers.push(Qwen38LayerRuntimeState::Full);
            } else {
                let conv_width = num_v_heads * 2;
                layers.push(Qwen38LayerRuntimeState::Linear {
                    conv_state: Some(ConvRingState {
                        slots: (0..3)
                            .map(|_| Tensor::zeros((conv_width, 1), DType::F32, device).unwrap())
                            .collect(),
                        next_idx: 0,
                    }),
                    recurrent_state: Some(
                        Tensor::zeros((1, num_v_heads, 2, 2), DType::F32, device).unwrap(),
                    ),
                });
            }
        }
        Qwen38TextRuntimeState { layers }
    }

    fn advance_synthetic_decode_state(
        state: &mut Qwen38TextRuntimeState,
        device: &Device,
        num_v_heads: usize,
    ) {
        for layer in &mut state.layers {
            match layer {
                Qwen38LayerRuntimeState::Linear {
                    conv_state: Some(conv_state),
                    recurrent_state,
                } => {
                    let conv_width = num_v_heads * 2;
                    let projection = Tensor::zeros((1, 1, conv_width), DType::F32, device).unwrap();
                    let current = projection
                        .i((0, 0))
                        .unwrap()
                        .reshape((conv_width, 1))
                        .unwrap();
                    conv_state.push_decode(&current).unwrap();
                    *recurrent_state =
                        Some(Tensor::zeros((1, num_v_heads, 2, 2), DType::F32, device).unwrap());
                }
                Qwen38LayerRuntimeState::Full => {}
                _ => panic!("synthetic state must initialize every linear cache"),
            }
        }
    }

    #[test]
    fn persistent_decode_storage_plateaus_for_small_and_large_topologies() {
        for (layer_count, num_v_heads) in [(24, 16), (32, 32)] {
            let mut state = synthetic_decode_state(&Device::Cpu, layer_count, num_v_heads);
            let baseline = state.allocated_session_bytes().unwrap();

            for _ in 0..96 {
                advance_synthetic_decode_state(&mut state, &Device::Cpu, num_v_heads);
                assert_eq!(
                    state.allocated_session_bytes(),
                    Some(baseline),
                    "{layer_count}-layer/{num_v_heads}V state retained growing backing storage"
                );
            }
        }
    }
}
