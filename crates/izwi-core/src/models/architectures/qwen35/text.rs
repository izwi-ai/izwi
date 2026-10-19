use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{ops, rotary_emb};
use candle_core::quantized::QMatMul;
use candle_transformers::quantized_nn::RmsNorm;

use crate::backends::kv::{
    submit_ordered_after_write, KvSlotMap, KvWriteArgs, KvWriteCompletionCollector,
    PagedKvDecodeArgs,
};
use crate::backends::state::{
    PhysicalStateSequenceId, PhysicalStateTransactionId, StateComponentValue, TensorStateArena,
};
use crate::error::{Error, Result};
use crate::kernels::cuda::gdn::{self, GdnDecodeSpec};
use crate::kernels::cuda::rope;
use crate::kernels::{
    try_fused_gated_delta_recurrent, try_fused_gated_rms_norm, try_fused_l2_norm,
    try_fused_silu_mul, try_qwen35_causal_conv_sequence, try_tiled_deltanet_recurrence,
};
use crate::kv::v2::{StateComponentId, StateDomainId};
use crate::kv::KvDecodeBatchMetadata;
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PreparedPhysicalPagedStep};
use crate::models::shared::memory::accounting::{
    deep_copy_tensor_storage, TensorStorageAccounting,
};
use crate::models::shared::telemetry::{
    record_prefill_sequence_span, record_rope_kernel, record_rope_manual,
};
use crate::models::shared::weights::gguf::GgufLoader;
use crate::models::shared::weights::token_table::QuantizedTokenTable;

use super::cache::{CONVOLUTION_STATE_DOMAIN, RECURRENT_STATE_DOMAIN};
use super::chat::Qwen35TextConfig;
use super::fast_path::{
    compare_values, legacy_requested, summarize, xorshift_values, Qwen35CudaSwitches,
    Qwen35FastPath,
};
use super::linear::{forward_shared, Qwen35Linear};
use super::telemetry::Qwen35Telemetry;

/// Environment switch that forces the Candle op chain for DeltaNet decode.
const FUSED_DECODE_ENV: &str = "IZWI_QWEN35_FUSED_DECODE";

#[path = "decode_graph.rs"]
mod decode_graph;

pub struct Qwen35TextModel {
    device: Device,
    /// The vocabulary table, kept quantized: embedding rows are gathered
    /// from it, and a tied output head shares it.
    token_table: QuantizedTokenTable,
    layers: Vec<Qwen35Layer>,
    output_norm: RmsNorm,
    output: Qwen35Linear,
    finite_diagnostics_enabled: bool,
    telemetry: Arc<Qwen35Telemetry>,
    /// The checkpoint's context length (the graph's block-table capacity).
    max_context_tokens: usize,
    /// Whole-step CUDA graph decode, when resolved (opt-in).
    decode_graph: Option<decode_graph::Qwen35DecodeGraph>,
    decode_graph_path: Qwen35FastPath,
}

/// M-RoPE cos/sin for the token positions of one forward, shaped
/// `(rows, tokens, rope_dim / 2)` and shared by every full-attention layer.
struct Qwen35RopeTable {
    cos: Tensor,
    sin: Tensor,
}

#[derive(Clone)]
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
                Qwen35LayerRuntimeState::Full => {}
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
            Error::InferenceError("Qwen3.5 recurrent state is missing its convolution peer".into())
        })?;
        let convolution = convolution.ok_or_else(|| {
            Error::InferenceError("Qwen3.5 convolution state is missing its recurrent peer".into())
        })?;
        let mut recurrent_components = recurrent.components.iter();
        let mut convolution_components = convolution.components.iter();
        for layer in &mut self.layers {
            let Qwen35LayerRuntimeState::Linear {
                conv_state,
                recurrent_state,
            } = layer
            else {
                continue;
            };
            let recurrent = recurrent_components.next().ok_or_else(|| {
                Error::InferenceError("Qwen3.5 recurrent component coverage is incomplete".into())
            })?;
            let convolution = convolution_components.next().ok_or_else(|| {
                Error::InferenceError("Qwen3.5 convolution component coverage is incomplete".into())
            })?;
            let recurrent_tensor = recurrent.tensor.as_ref().ok_or_else(|| {
                Error::InferenceError("Qwen3.5 recurrent component is absent".into())
            })?;
            let convolution_tensor = convolution.tensor.as_ref().ok_or_else(|| {
                Error::InferenceError("Qwen3.5 convolution component is absent".into())
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
                "Qwen3.5 tensor state has components for unknown layers".into(),
            ));
        }
        Ok(())
    }

    /// Stage the recurrent and convolution state into `transaction` and
    /// return the number of convolution rings stacked for it.
    pub(crate) fn stage_tensor_domains(
        &mut self,
        arena: &TensorStateArena,
        transaction: PhysicalStateTransactionId,
        target_cursor: u64,
    ) -> Result<usize> {
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
            let Qwen35LayerRuntimeState::Linear {
                conv_state,
                recurrent_state,
            } = layer
            else {
                continue;
            };
            let recurrent_tensor = recurrent_state.as_ref().ok_or_else(|| {
                Error::InferenceError("Qwen3.5 recurrent state was not initialized".into())
            })?;
            let ring = conv_state.as_ref().ok_or_else(|| {
                Error::InferenceError("Qwen3.5 convolution state was not initialized".into())
            })?;
            if ring.slots.is_empty() || ring.next_idx >= ring.slots.len() {
                return Err(Error::InferenceError(
                    "Qwen3.5 convolution ring is invalid at the physical boundary".into(),
                ));
            }
            let ordered = (0..ring.slots.len())
                .map(|offset| &ring.slots[(ring.next_idx + offset) % ring.slots.len()])
                .collect::<Vec<_>>();
            let ring_tensor = Tensor::stack(&ordered, 0)?;
            let component = u32::try_from(recurrent.len() + 1)
                .map_err(|_| Error::InvalidInput("Qwen3.5 state component overflow".into()))?;
            recurrent.push(StateComponentValue {
                component: StateComponentId::new(component),
                tensor: Some(recurrent_tensor.clone()),
            });
            convolution.push(StateComponentValue {
                component: StateComponentId::new(component),
                tensor: Some(ring_tensor),
            });
        }
        let conv_ring_stacks = convolution.len();
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
            if let Qwen35LayerRuntimeState::Linear {
                conv_state,
                recurrent_state,
            } = layer
            {
                *conv_state = None;
                *recurrent_state = None;
            }
        }
        Ok(conv_ring_stacks)
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
    /// The history slots oldest first.
    fn ordered_slots(&self) -> Vec<Tensor> {
        (0..self.slots.len())
            .map(|offset| self.slots[(self.next_idx + offset) % self.slots.len()].clone())
            .collect()
    }

    /// The history oldest first as one `[slots, ...]` tensor (self-checks).
    fn history(&self) -> Result<Tensor> {
        Tensor::stack(&self.ordered_slots(), 0).map_err(Error::from)
    }

    /// A ring over a `[slots, conv_dim]` history whose first row is the
    /// oldest, as the fused conv decode returns it. Slots are `(conv_dim, 1)`
    /// views into that one tensor.
    fn from_history(history: &Tensor, conv_dim: usize) -> Result<Self> {
        let slots = (0..history.dim(0)?)
            .map(|index| history.i(index)?.reshape((conv_dim, 1)))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self { slots, next_idx: 0 })
    }

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
                "Invalid Qwen3.5 convolution ring: slots={}, next_idx={}",
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
enum Qwen35LayerRuntimeState {
    Linear {
        conv_state: Option<ConvRingState>,
        recurrent_state: Option<Tensor>,
    },
    Full,
}

struct Qwen35Layer {
    attn_norm: RmsNorm,
    mixer: Qwen35Mixer,
    post_attention_norm: RmsNorm,
    ffn: Qwen35Mlp,
}

enum Qwen35Mixer {
    Linear(Qwen35LinearAttention),
    Full(Qwen35FullAttention),
}

struct Qwen35Mlp {
    gate: Qwen35Linear,
    up: Qwen35Linear,
    down: Qwen35Linear,
}

struct Qwen35FullAttention {
    q_proj: Qwen35Linear,
    k_proj: Qwen35Linear,
    v_proj: Qwen35Linear,
    o_proj: Qwen35Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    rope_theta: f64,
    mrope_sections: Vec<usize>,
    rope_kernel_enabled: bool,
    rope_inv_freqs: Vec<f32>,
    /// F32 head-norm gains and epsilon for the fused graph attention
    /// (Candle's `RmsNorm` keeps its gain private).
    q_norm_gain: Tensor,
    k_norm_gain: Tensor,
    qk_norm_eps: f64,
    /// Graph attention inputs, resolved with the decode graph.
    graph_qk: Option<decode_graph::GraphQk>,
}

struct Qwen35LinearAttention {
    qkv_proj: Qwen35Linear,
    gate_proj: Qwen35Linear,
    beta_proj: Qwen35Linear,
    alpha_proj: Qwen35Linear,
    dt_bias: Tensor,
    a: Tensor,
    conv_kernel: Tensor,
    conv_kernel_slices: Vec<Tensor>,
    norm: Qwen35GatedRmsNorm,
    out_proj: Qwen35Linear,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    conv_dim: usize,
    kernel_size: usize,
    tiled_recurrence_enabled: bool,
    tiled_recurrence_tile_size_override: Option<usize>,
    /// Fused single-token decode layout, set only after a load-time
    /// self-check passes (see [`Qwen35LinearAttention::resolve_fused_decode`]).
    fused_decode: Option<GdnDecodeSpec>,
    fused_decode_path: Qwen35FastPath,
}

struct Qwen35GatedRmsNorm {
    weight: Tensor,
    eps: f64,
}

/// Checkpoint-format seam for the Qwen3.5 hybrid trunk.
///
/// Names handed to the source are the logical GGUF-style names
/// (`token_embd.weight`, `blk.{i}.attn_q.weight`, ...); implementations
/// translate to their own checkpoint layout internally.
pub(crate) trait Qwen35WeightSource {
    fn has(&self, name: &str) -> bool;

    fn qmatmul(&self, name: &str, device: &Device) -> Result<QMatMul>;

    fn rms_norm(&self, name: &str, eps: f64, device: &Device) -> Result<RmsNorm>;

    /// Dense tensor, coerced to `dtype` when requested (always F32 when
    /// `Some(F32)`).
    fn dense(&self, name: &str, dtype: Option<DType>, device: &Device) -> Result<Tensor>;

    /// Token table `[vocab, hidden]`, in its stored quantized form.
    fn token_table(&self, device: &Device) -> Result<QuantizedTokenTable>;
}

/// GGUF-backed source for the dense Qwen3.5 family.
pub(crate) struct GgufSource<'a> {
    loader: &'a GgufLoader,
}

impl<'a> GgufSource<'a> {
    pub(crate) fn new(loader: &'a GgufLoader) -> Self {
        Self { loader }
    }
}

impl Qwen35WeightSource for GgufSource<'_> {
    fn has(&self, name: &str) -> bool {
        self.loader.has_tensor(name)
    }

    fn qmatmul(&self, name: &str, device: &Device) -> Result<QMatMul> {
        load_qmatmul(self.loader, device, name)
    }

    fn rms_norm(&self, name: &str, eps: f64, device: &Device) -> Result<RmsNorm> {
        load_rms_norm(self.loader, device, name, eps)
    }

    fn dense(&self, name: &str, dtype: Option<DType>, device: &Device) -> Result<Tensor> {
        load_dense(self.loader, device, name, dtype)
    }

    fn token_table(&self, device: &Device) -> Result<QuantizedTokenTable> {
        QuantizedTokenTable::new(self.loader.load_qtensor("token_embd.weight", device)?)
    }
}

impl Qwen35TextModel {
    pub fn load(loader: &GgufLoader, cfg: &Qwen35TextConfig, device: &Device) -> Result<Self> {
        Self::load_with_source(&GgufSource::new(loader), cfg, device)
    }

    pub(crate) fn load_with_source(
        source: &dyn Qwen35WeightSource,
        cfg: &Qwen35TextConfig,
        device: &Device,
    ) -> Result<Self> {
        if cfg.attention_key_length != cfg.attention_value_length {
            return Err(Error::ModelLoadError(format!(
                "Qwen3.5 full attention currently requires key/value head dims to match, found {} and {}",
                cfg.attention_key_length, cfg.attention_value_length
            )));
        }
        if cfg.ssm_time_step_rank == 0 || !cfg.ssm_inner_size.is_multiple_of(cfg.ssm_time_step_rank)
        {
            return Err(Error::ModelLoadError(format!(
                "Invalid Qwen3.5 linear attention dims: inner_size={}, time_step_rank={}",
                cfg.ssm_inner_size, cfg.ssm_time_step_rank
            )));
        }

        let token_table = source.token_table(device)?;
        let hidden_size = token_table.hidden_size();
        if hidden_size != cfg.embedding_length {
            return Err(Error::ModelLoadError(format!(
                "Qwen3.5 token embedding width mismatch: checkpoint has {hidden_size}, metadata says {}",
                cfg.embedding_length
            )));
        }

        let output_norm = source.rms_norm("output_norm.weight", cfg.attention_layer_norm_rms_epsilon, device)?;
        // A tied head (no `output.weight`, as in the 0.8B/2B/4B checkpoints)
        // shares the token table instead of reading it a second time.
        let output = Qwen35Linear::new(if source.has("output.weight") {
            source.qmatmul("output.weight", device)?
        } else {
            token_table.tied_head()?
        });
        let finite_diagnostics_enabled =
            crate::env::flag_or("IZWI_QWEN35_FINITE_DIAGNOSTICS", false);

        let mut layers = Vec::with_capacity(cfg.block_count);
        for layer_idx in 0..cfg.block_count {
            let prefix = format!("blk.{layer_idx}");
            let attn_norm = source.rms_norm(
                &format!("{prefix}.attn_norm.weight"),
                cfg.attention_layer_norm_rms_epsilon,
                device,
            )?;
            let post_attention_norm = source.rms_norm(
                &format!("{prefix}.post_attention_norm.weight"),
                cfg.attention_layer_norm_rms_epsilon,
                device,
            )?;
            let ffn = Qwen35Mlp::load_via(source, device, &prefix)?;
            let mixer = if is_full_attention_layer(layer_idx, cfg.full_attention_interval) {
                Qwen35Mixer::Full(Qwen35FullAttention::load_via(source, device, &prefix, cfg)?)
            } else {
                Qwen35Mixer::Linear(Qwen35LinearAttention::load_via(source, device, &prefix, cfg)?)
            };

            layers.push(Qwen35Layer {
                attn_norm,
                mixer,
                post_attention_norm,
                ffn,
            });
        }
        Ok(Self {
            device: device.clone(),
            token_table,
            layers,
            output_norm,
            output,
            finite_diagnostics_enabled,
            telemetry: Arc::default(),
            max_context_tokens: cfg.context_length,
            decode_graph: None,
            decode_graph_path: Qwen35FastPath::legacy("unresolved"),
        })
    }

    pub(crate) fn telemetry(&self) -> &Arc<Qwen35Telemetry> {
        &self.telemetry
    }

    /// Resolve every fused fast path after load. `allow_cpu` lets tests run
    /// the portable CPU references; production keeps CPU on the Candle chain.
    pub(crate) fn resolve_fast_paths(&mut self, switches: Qwen35CudaSwitches, allow_cpu: bool) {
        for linear in self.linears_mut() {
            linear.resolve_kquant(switches.kquant_off);
        }
        let hidden = self.hidden_size();
        for layer in &mut self.layers {
            if let Qwen35Mixer::Linear(mixer) = &mut layer.mixer {
                mixer.resolve_fused_decode(
                    hidden,
                    &self.device,
                    switches.fused_decode_off,
                    allow_cpu,
                );
            }
        }
        self.resolve_decode_graph(switches.graphs_off, allow_cpu);
    }

    /// Resolve whole-step CUDA graph decode after the fused paths it needs.
    /// `allow_eager` lets tests run the graph body eagerly off CUDA.
    pub(crate) fn resolve_decode_graph(
        &mut self,
        graphs_off: Option<&'static str>,
        allow_eager: bool,
    ) {
        self.decode_graph = None;
        if self.device.is_cuda() {
            let device = self.device.clone();
            for layer in &mut self.layers {
                if let Qwen35Mixer::Full(attention) = &mut layer.mixer {
                    attention.graph_qk = None;
                    if rope::supported(&device, DType::F16, attention.head_dim, attention.rope_dim)
                        && attention.rope_inv_freqs.len() == attention.rope_dim / 2
                    {
                        match decode_graph::GraphQk::new(attention, &device) {
                            Ok(qk) => attention.graph_qk = Some(qk),
                            Err(error) => tracing::warn!(
                                %error,
                                "Qwen3.5 graph attention inputs failed to load"
                            ),
                        }
                    }
                }
            }
        }
        let (graph, path) = decode_graph::Qwen35DecodeGraph::resolve(
            self,
            graphs_off,
            self.max_context_tokens,
            allow_eager,
        );
        self.decode_graph = graph;
        self.decode_graph_path = path;
    }

    /// Graph decode resolution and runtime state, for the admin diagnostics.
    pub(crate) fn decode_graph_summary(&self) -> serde_json::Value {
        match &self.decode_graph {
            Some(graph) => graph.summary(),
            None => summarize([&self.decode_graph_path]),
        }
    }

    /// Whether every projection runs through the K-quant MMVQ.
    fn projections_all_kquant(&self) -> bool {
        let mut all = true;
        for layer in &self.layers {
            let linears: Vec<&Qwen35Linear> = match &layer.mixer {
                Qwen35Mixer::Linear(mixer) => vec![
                    &mixer.qkv_proj,
                    &mixer.gate_proj,
                    &mixer.beta_proj,
                    &mixer.alpha_proj,
                    &mixer.out_proj,
                ],
                Qwen35Mixer::Full(attention) => vec![
                    &attention.q_proj,
                    &attention.k_proj,
                    &attention.v_proj,
                    &attention.o_proj,
                ],
            };
            all &= linears
                .into_iter()
                .chain([&layer.ffn.gate, &layer.ffn.up, &layer.ffn.down])
                .all(|linear| *linear.kquant_path() == Qwen35FastPath::Fused);
        }
        all && *self.output.kquant_path() == Qwen35FastPath::Fused
    }

    fn first_attention(&self) -> Result<&Qwen35FullAttention> {
        self.layers
            .iter()
            .find_map(|layer| match &layer.mixer {
                Qwen35Mixer::Full(attention) => Some(attention),
                Qwen35Mixer::Linear(_) => None,
            })
            .ok_or_else(|| {
                Error::InferenceError("Qwen3.5 model has no full-attention layer".into())
            })
    }

    /// Every projection: the layers' in order, then the LM head.
    fn linears_mut(&mut self) -> Vec<&mut Qwen35Linear> {
        let mut linears = Vec::new();
        for layer in &mut self.layers {
            match &mut layer.mixer {
                Qwen35Mixer::Linear(mixer) => linears.extend([
                    &mut mixer.qkv_proj,
                    &mut mixer.gate_proj,
                    &mut mixer.beta_proj,
                    &mut mixer.alpha_proj,
                    &mut mixer.out_proj,
                ]),
                Qwen35Mixer::Full(attention) => linears.extend([
                    &mut attention.q_proj,
                    &mut attention.k_proj,
                    &mut attention.v_proj,
                    &mut attention.o_proj,
                ]),
            }
            linears.extend([&mut layer.ffn.gate, &mut layer.ffn.up, &mut layer.ffn.down]);
        }
        linears.push(&mut self.output);
        linears
    }

    /// K-quant MMVQ resolution over every projection, for the admin
    /// diagnostics.
    pub(crate) fn projection_summary(&self) -> serde_json::Value {
        let mut paths = Vec::new();
        for layer in &self.layers {
            match &layer.mixer {
                Qwen35Mixer::Linear(mixer) => paths.extend([
                    mixer.qkv_proj.kquant_path(),
                    mixer.gate_proj.kquant_path(),
                    mixer.beta_proj.kquant_path(),
                    mixer.alpha_proj.kquant_path(),
                    mixer.out_proj.kquant_path(),
                ]),
                Qwen35Mixer::Full(attention) => paths.extend([
                    attention.q_proj.kquant_path(),
                    attention.k_proj.kquant_path(),
                    attention.v_proj.kquant_path(),
                    attention.o_proj.kquant_path(),
                ]),
            }
            paths.extend([
                layer.ffn.gate.kquant_path(),
                layer.ffn.up.kquant_path(),
                layer.ffn.down.kquant_path(),
            ]);
        }
        paths.push(self.output.kquant_path());
        summarize(paths)
    }

    /// Fused DeltaNet decode resolution, for the admin diagnostics.
    pub(crate) fn gdn_decode_summary(&self) -> serde_json::Value {
        summarize(self.layers.iter().filter_map(|layer| match &layer.mixer {
            Qwen35Mixer::Linear(mixer) => Some(&mixer.fused_decode_path),
            Qwen35Mixer::Full(_) => None,
        }))
    }

    /// The RoPE table for `positions`, or `None` when attention has no
    /// rotary dimensions. Built once per forward on the host (one upload).
    fn rope_table(
        &self,
        attention: &Qwen35FullAttention,
        positions: &[[usize; 3]],
        shape: (usize, usize),
        dtype: DType,
    ) -> Result<Option<Qwen35RopeTable>> {
        let table = attention.rope_table(positions, shape, &self.device, dtype)?;
        if table.is_some() {
            self.telemetry.record_rope_host_upload();
        }
        Ok(table)
    }

    pub fn new_state(&self) -> Qwen35TextRuntimeState {
        Qwen35TextRuntimeState {
            layers: self.layers.iter().map(Qwen35Layer::new_state).collect(),
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.token_table.hidden_size()
    }

    pub(crate) fn forward_token_id_at_physical(
        &self,
        token_id: u32,
        position_ids: [usize; 3],
        state: &mut Qwen35TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Tensor> {
        if self.decode_graph.is_some() {
            // Graph decode serves the one-row batch path.
            return self
                .forward_token_ids_batch_at_physical(
                    &[token_id],
                    &[position_ids],
                    &mut [state],
                    &mut [cache],
                )?
                .i((0, 0))
                .map_err(Error::from);
        }
        let input = Tensor::from_vec(vec![token_id], (1, 1), &self.device)?;
        let hidden = self.token_table.embed(&input)?;
        self.forward_hidden_to_logits_physical(&hidden, position_ids, state, cache)
    }

    /// One token's layers and LM head. The KV writes are sealed after the LM
    /// head is enqueued, so the device keeps working while the host waits on
    /// the write fences (the batched decode already orders it this way).
    fn forward_hidden_to_logits_physical(
        &self,
        hidden: &Tensor,
        position_ids: [usize; 3],
        state: &mut Qwen35TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Tensor> {
        let (hidden, prepared) =
            self.forward_hidden_physical_prepared(hidden, &[position_ids], state, cache)?;
        match self.forward_hidden_to_logits(&hidden) {
            Ok(logits) => {
                cache.commit_prepared(prepared)?;
                Ok(logits)
            }
            // Committing zero tokens drains every write and exposes none.
            Err(error) => match cache.commit_prepared_prefix(prepared, 0) {
                Ok(()) => Err(error),
                Err(drain) => Err(Error::InferenceError(format!(
                    "Qwen3.5 LM head failed: {error}; KV write drain also failed: {drain}"
                ))),
            },
        }
    }

    pub(crate) fn forward_token_ids_batch_at_physical(
        &self,
        token_ids: &[u32],
        position_ids: &[[usize; 3]],
        states: &mut [&mut Qwen35TextRuntimeState],
        caches: &mut [&mut PhysicalPagedKvCache],
    ) -> Result<Tensor> {
        let batch_size = token_ids.len();
        if batch_size == 0
            || position_ids.len() != batch_size
            || states.len() != batch_size
            || caches.len() != batch_size
        {
            return Err(Error::InvalidInput(
                "Qwen3.5 hybrid decode batch rows do not match".into(),
            ));
        }
        for state in states.iter() {
            self.validate_runtime_state(state)?;
        }
        let sparse_layers = self
            .layers
            .iter()
            .enumerate()
            .filter_map(|(index, layer)| {
                matches!(layer.mixer, Qwen35Mixer::Full(_)).then_some(index as u32)
            })
            .collect::<Vec<_>>();
        let first_full = self
            .layers
            .iter()
            .find_map(|layer| match &layer.mixer {
                Qwen35Mixer::Full(attention) => Some(attention),
                Qwen35Mixer::Linear(_) => None,
            })
            .ok_or_else(|| {
                Error::InferenceError("Qwen3.5 model has no full-attention layer".into())
            })?;
        let start_positions = caches
            .iter()
            .map(|cache| cache.context_len())
            .collect::<Vec<_>>();
        for cache in caches.iter() {
            cache.validate_sparse_model(
                &sparse_layers,
                first_full.num_kv_heads,
                first_full.head_dim,
                first_full.head_dim,
            )?;
        }
        let first = &*caches[0];
        let slots = caches
            .iter()
            .enumerate()
            .map(|(row, cache)| {
                cache
                    .slots_for_append(start_positions[row], 1)
                    .map(|slots| slots[0])
            })
            .collect::<Result<Vec<_>>>()?;
        let lowered = first.arena().lower_slots(&slots)?;
        let metadata = KvDecodeBatchMetadata {
            sequences: caches
                .iter()
                .enumerate()
                .map(|(row, cache)| cache.sequence_table(start_positions[row] + 1))
                .collect::<Result<Vec<_>>>()?,
        };
        let mut completions =
            KvWriteCompletionCollector::new(first.arena().config(), lowered.logical_slots())?;
        let execution = (|| -> Result<Tensor> {
            if let (1, Some(graph)) = (batch_size, &self.decode_graph) {
                let cache = &*caches[0];
                // The graph's warm check: the standard decode on a copy of
                // the row state, its writes collected and sealed apart.
                let mut reference_state = states[0].clone();
                let mut reference = || -> Result<Tensor> {
                    let mut collector = KvWriteCompletionCollector::new(
                        first.arena().config(),
                        lowered.logical_slots(),
                    )?;
                    let logits = self.standard_decode_logits(
                        token_ids,
                        position_ids,
                        &mut [&mut reference_state],
                        &[cache],
                        first_full,
                        lowered.as_ref(),
                        &metadata,
                        &mut collector,
                        sparse_layers.len(),
                    );
                    let sealed = collector.seal();
                    let logits = logits?;
                    sealed?;
                    Ok(logits)
                };
                if let Some(logits) = graph.forward(
                    self,
                    token_ids[0],
                    position_ids[0],
                    slots[0],
                    states[0],
                    cache,
                    lowered.as_ref(),
                    &metadata,
                    &mut completions,
                    &mut reference,
                )? {
                    return Ok(logits);
                }
            }
            let cache_refs = caches.iter().map(|cache| &**cache).collect::<Vec<_>>();
            self.standard_decode_logits(
                token_ids,
                position_ids,
                states,
                &cache_refs,
                first_full,
                lowered.as_ref(),
                &metadata,
                &mut completions,
                sparse_layers.len(),
            )
        })();
        let logits = match execution {
            Ok(logits) => logits,
            Err(error) => {
                return match completions.drain() {
                    Ok(()) => Err(error),
                    Err(drain) => Err(Error::InferenceError(format!(
                    "Qwen3.5 target batch failed: {error}; write-fence drain also failed: {drain}"
                ))),
                }
            }
        };
        let completion = Arc::new(completions.seal()?);
        for (row, cache) in caches.iter_mut().enumerate() {
            cache.commit_shared_completion(start_positions[row], 1, completion.clone())?;
        }
        Ok(logits)
    }

    /// The standard decode for `token_ids` (one per row): embedding, every
    /// layer through the arena's attention, the final norm and the LM head.
    /// Returns logits `[batch, 1, vocab]`.
    #[allow(clippy::too_many_arguments)]
    fn standard_decode_logits(
        &self,
        token_ids: &[u32],
        position_ids: &[[usize; 3]],
        states: &mut [&mut Qwen35TextRuntimeState],
        caches: &[&PhysicalPagedKvCache],
        first_full: &Qwen35FullAttention,
        slots: &dyn KvSlotMap,
        metadata: &KvDecodeBatchMetadata,
        completions: &mut KvWriteCompletionCollector,
        sparse_layers: usize,
    ) -> Result<Tensor> {
        let batch_size = token_ids.len();
        let input = Tensor::from_slice(token_ids, (batch_size, 1), &self.device)?;
        let mut hidden = self.token_table.embed(&input)?;
        let rope = self.rope_table(first_full, position_ids, (batch_size, 1), hidden.dtype())?;
        let mut physical_layer = 0usize;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let mut layer_states = states
                .iter_mut()
                .map(|state| &mut state.layers[layer_index])
                .collect::<Vec<_>>();
            for state in layer_states.iter_mut() {
                layer.ensure_state_initialized(state, &self.device)?;
            }
            hidden = layer.forward_physical_decode_batch(
                &hidden,
                &mut layer_states,
                position_ids,
                rope.as_ref(),
                caches,
                slots,
                metadata,
                completions,
                &mut physical_layer,
            )?;
            validate_qwen35_finite_tensor(
                &hidden,
                layer_index,
                layer.decode_diagnostic_path(),
                self.finite_diagnostics_enabled,
            )?;
        }
        if physical_layer != sparse_layers {
            return Err(Error::InferenceError(
                "Qwen3.5 batched attention did not cover every sparse layer".into(),
            ));
        }
        self.project_hidden_span(&hidden)
    }

    pub(crate) fn prefill_token_ids_physical(
        &self,
        token_ids: &[u32],
        position_ids: &[[usize; 3]],
        state: &mut Qwen35TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
        compute_logits: bool,
    ) -> Result<Option<Tensor>> {
        if token_ids.is_empty() {
            return Ok(None);
        }
        if token_ids.len() != position_ids.len() {
            return Err(Error::InvalidInput(format!(
                "Qwen3.5 physical prefill span mismatch: {} token ids for {} position ids",
                token_ids.len(),
                position_ids.len()
            )));
        }
        record_prefill_sequence_span(token_ids.len());
        let input = Tensor::from_vec(token_ids.to_vec(), (1, token_ids.len()), &self.device)?;
        let hidden = self.token_table.embed(&input)?;
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
        state: &mut Qwen35TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Tensor> {
        self.forward_hidden_to_logits_physical(input_embedding, position_ids, state, cache)
    }

    pub(crate) fn prefill_input_embeddings_physical(
        &self,
        input_embeddings: &Tensor,
        position_ids: &[[usize; 3]],
        state: &mut Qwen35TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
        compute_logits: bool,
    ) -> Result<Option<Tensor>> {
        let (_, sequence_len, _) = input_embeddings.dims3()?;
        if sequence_len == 0 || sequence_len != position_ids.len() {
            return Err(Error::InvalidInput(
                "Qwen3.5 embedding prefill span does not match its positions".into(),
            ));
        }
        record_prefill_sequence_span(sequence_len);
        let hidden = self.forward_hidden_physical(input_embeddings, position_ids, state, cache)?;
        if !compute_logits {
            return Ok(None);
        }
        let last = hidden.narrow(1, sequence_len - 1, 1)?;
        self.forward_hidden_to_logits(&last).map(Some)
    }

    fn forward_hidden_physical(
        &self,
        input: &Tensor,
        position_ids: &[[usize; 3]],
        state: &mut Qwen35TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Tensor> {
        let (hidden, prepared) =
            self.forward_hidden_physical_prepared(input, position_ids, state, cache)?;
        cache.commit_prepared(prepared)?;
        Ok(hidden)
    }

    /// The layers of one span, with its KV writes enqueued but not yet
    /// sealed. The caller must commit or abort the returned step.
    fn forward_hidden_physical_prepared(
        &self,
        input: &Tensor,
        position_ids: &[[usize; 3]],
        state: &mut Qwen35TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<(Tensor, PreparedPhysicalPagedStep)> {
        self.validate_runtime_state(state)?;
        let (_, sequence_len, hidden_size) = input.dims3()?;
        if sequence_len == 0
            || sequence_len != position_ids.len()
            || hidden_size != self.hidden_size()
        {
            return Err(Error::InvalidInput(
                "Qwen3.5 physical hidden span does not match its positions or model width".into(),
            ));
        }
        let sparse_layers = self
            .layers
            .iter()
            .enumerate()
            .filter_map(|(index, layer)| {
                matches!(layer.mixer, Qwen35Mixer::Full(_)).then_some(index as u32)
            })
            .collect::<Vec<_>>();
        let first_full = self.layers.iter().find_map(|layer| match &layer.mixer {
            Qwen35Mixer::Full(attention) => Some(attention),
            Qwen35Mixer::Linear(_) => None,
        });
        let first_full = first_full.ok_or_else(|| {
            Error::InferenceError("Qwen3.5 model has no full-attention layer".into())
        })?;
        cache.validate_sparse_model(
            &sparse_layers,
            first_full.num_kv_heads,
            first_full.head_dim,
            first_full.head_dim,
        )?;
        let rope = self.rope_table(first_full, position_ids, (1, sequence_len), input.dtype())?;
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
                rope.as_ref(),
                cache,
                &mut prepared,
                &mut physical_layer,
            )?;
            validate_qwen35_finite_tensor(
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
                "Qwen3.5 physical attention did not cover every sparse layer".into(),
            ));
        }
        Ok((hidden, prepared))
    }

    pub fn forward_hidden_to_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        self.project_hidden_span(hidden)?
            .i((0, 0))
            .map_err(Error::from)
    }

    fn project_hidden_span(&self, hidden: &Tensor) -> Result<Tensor> {
        let hidden = self.output_norm.forward(hidden)?;
        validate_qwen35_finite_tensor(
            &hidden,
            self.layers.len(),
            "output.norm",
            self.finite_diagnostics_enabled,
        )?;
        let logits = self.output.forward(&hidden)?;
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
            Qwen35Mixer::Full(_) => Qwen35LayerRuntimeState::Full,
        }
    }

    /// Pre-initialize lazy state tensors so the first-use allocation cost
    /// does not happen inside the per-token hot loop during prefill.
    fn ensure_state_initialized(
        &self,
        state: &mut Qwen35LayerRuntimeState,
        device: &Device,
    ) -> Result<()> {
        if let (
            Qwen35Mixer::Linear(mixer),
            Qwen35LayerRuntimeState::Linear {
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
        state: &mut Qwen35LayerRuntimeState,
        position_ids: &[[usize; 3]],
        rope: Option<&Qwen35RopeTable>,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        physical_layer: &mut usize,
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let normalized = self.attn_norm.forward(hidden_states)?;
        let mixed = match &self.mixer {
            Qwen35Mixer::Linear(mixer) => {
                if normalized.dim(1)? == 1 {
                    mixer.forward(&normalized, state)?
                } else {
                    mixer.forward_sequence(&normalized, state)?
                }
            }
            Qwen35Mixer::Full(mixer) => {
                let output = mixer.forward_physical(
                    &normalized,
                    position_ids,
                    rope,
                    cache,
                    prepared,
                    *physical_layer,
                )?;
                *physical_layer = physical_layer.checked_add(1).ok_or_else(|| {
                    Error::InvalidInput("Qwen3.5 physical layer ordinal overflow".into())
                })?;
                output
            }
        };
        let hidden_states = (&residual + &mixed)?;
        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_norm.forward(&hidden_states)?;
        let hidden_states = self.ffn.forward(&hidden_states)?;
        (&residual + &hidden_states).map_err(Error::from)
    }

    fn forward_physical_decode_batch(
        &self,
        hidden_states: &Tensor,
        states: &mut [&mut Qwen35LayerRuntimeState],
        position_ids: &[[usize; 3]],
        rope: Option<&Qwen35RopeTable>,
        caches: &[&PhysicalPagedKvCache],
        slots: &dyn KvSlotMap,
        metadata: &KvDecodeBatchMetadata,
        completions: &mut KvWriteCompletionCollector,
        physical_layer: &mut usize,
    ) -> Result<Tensor> {
        let batch_size = hidden_states.dim(0)?;
        if hidden_states.dim(1)? != 1
            || batch_size == 0
            || states.len() != batch_size
            || position_ids.len() != batch_size
            || caches.len() != batch_size
        {
            return Err(Error::InvalidInput(
                "Qwen3.5 layer decode batch dimensions do not match".into(),
            ));
        }
        let residual = hidden_states.clone();
        let normalized = self.attn_norm.forward(hidden_states)?;
        let mixed = match &self.mixer {
            Qwen35Mixer::Linear(mixer) => mixer.forward_decode_batch(&normalized, states)?,
            Qwen35Mixer::Full(mixer) => {
                let output = mixer.forward_physical_decode_batch(
                    &normalized,
                    position_ids,
                    rope,
                    caches,
                    slots,
                    metadata,
                    completions,
                    *physical_layer,
                )?;
                *physical_layer = physical_layer.checked_add(1).ok_or_else(|| {
                    Error::InvalidInput("Qwen3.5 physical layer ordinal overflow".into())
                })?;
                output
            }
        };
        let hidden_states = (&residual + &mixed)?;
        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_norm.forward(&hidden_states)?;
        let hidden_states = self.ffn.forward(&hidden_states)?;
        (&residual + &hidden_states).map_err(Error::from)
    }
}

impl Qwen35Mlp {
    fn load_via(
        source: &dyn Qwen35WeightSource,
        device: &Device,
        prefix: &str,
    ) -> Result<Self> {
        Ok(Self {
            gate: Qwen35Linear::new(source.qmatmul(&format!("{prefix}.ffn_gate.weight"), device)?),
            up: Qwen35Linear::new(source.qmatmul(&format!("{prefix}.ffn_up.weight"), device)?),
            down: Qwen35Linear::new(source.qmatmul(&format!("{prefix}.ffn_down.weight"), device)?),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Use fused SiLU-gate-up if available (reduces memory bandwidth)
        let [gate_proj_out, up_proj_out] = forward_shared(hidden_states, [&self.gate, &self.up])?;

        let hidden = if let Some(fused) = try_fused_silu_mul(&gate_proj_out, &up_proj_out) {
            fused
        } else {
            let gate = ops::silu(&gate_proj_out)?;
            (&gate * &up_proj_out)?
        };

        self.down.forward(&hidden)
    }
}

impl Qwen35FullAttention {
    fn load_via(
        source: &dyn Qwen35WeightSource,
        device: &Device,
        prefix: &str,
        cfg: &Qwen35TextConfig,
    ) -> Result<Self> {
        Ok(Self {
            q_proj: Qwen35Linear::new(source.qmatmul(&format!("{prefix}.attn_q.weight"), device)?),
            k_proj: Qwen35Linear::new(source.qmatmul(&format!("{prefix}.attn_k.weight"), device)?),
            v_proj: Qwen35Linear::new(source.qmatmul(&format!("{prefix}.attn_v.weight"), device)?),
            o_proj: Qwen35Linear::new(
                source.qmatmul(&format!("{prefix}.attn_output.weight"), device)?,
            ),
            q_norm: source.rms_norm(
                &format!("{prefix}.attn_q_norm.weight"),
                cfg.attention_layer_norm_rms_epsilon,
                device,
            )?,
            k_norm: source.rms_norm(
                &format!("{prefix}.attn_k_norm.weight"),
                cfg.attention_layer_norm_rms_epsilon,
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
            rope_kernel_enabled: qwen35_rope_kernel_enabled(device),
            rope_inv_freqs: build_rope_inv_freqs(
                cfg.rope_dimension_count.min(cfg.attention_key_length),
                cfg.rope_freq_base,
            )?,
            q_norm_gain: source
                .dense(
                    &format!("{prefix}.attn_q_norm.weight"),
                    Some(DType::F32),
                    device,
                )?
                .flatten_all()?,
            k_norm_gain: source
                .dense(
                    &format!("{prefix}.attn_k_norm.weight"),
                    Some(DType::F32),
                    device,
                )?
                .flatten_all()?,
            qk_norm_eps: cfg.attention_layer_norm_rms_epsilon,
            graph_qk: None,
        })
    }

    fn forward_physical(
        &self,
        hidden_states: &Tensor,
        position_ids: &[[usize; 3]],
        rope: Option<&Qwen35RopeTable>,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        physical_layer: usize,
    ) -> Result<Tensor> {
        let seq_len = hidden_states.dim(1)?;
        if seq_len == 0 || seq_len != position_ids.len() {
            return Err(Error::InvalidInput(format!(
                "Qwen3.5 physical attention received {} tokens and {} positions",
                seq_len,
                position_ids.len()
            )));
        }
        let [q_out, k_out, v_out] =
            forward_shared(hidden_states, [&self.q_proj, &self.k_proj, &self.v_proj])?;
        let q_proj = q_out.reshape((1, seq_len, self.num_heads, self.head_dim * 2))?;
        let query_states = q_proj.narrow(3, 0, self.head_dim)?;
        let gate = q_proj.narrow(3, self.head_dim, self.head_dim)?.reshape((
            1,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;
        let key_states = k_out.reshape((1, seq_len, self.num_kv_heads, self.head_dim))?;
        let value_states = v_out.reshape((1, seq_len, self.num_kv_heads, self.head_dim))?;
        let query_states = self.q_norm.forward(&query_states.contiguous()?)?;
        let key_states = self.k_norm.forward(&key_states.contiguous()?)?;
        let (query_states, key_states) = self.apply_rope(
            &query_states,
            &key_states,
            rope.map(|table| (&table.cos, &table.sin)),
        )?;
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
        self.o_proj.forward(&output)
    }

    fn forward_physical_decode_batch(
        &self,
        hidden_states: &Tensor,
        position_ids: &[[usize; 3]],
        rope: Option<&Qwen35RopeTable>,
        caches: &[&PhysicalPagedKvCache],
        slots: &dyn KvSlotMap,
        metadata: &KvDecodeBatchMetadata,
        completions: &mut KvWriteCompletionCollector,
        physical_layer: usize,
    ) -> Result<Tensor> {
        let batch_size = hidden_states.dim(0)?;
        if hidden_states.dim(1)? != 1
            || batch_size == 0
            || position_ids.len() != batch_size
            || caches.len() != batch_size
        {
            return Err(Error::InvalidInput(
                "Qwen3.5 full-attention decode batch dimensions do not match".into(),
            ));
        }
        let first = caches[0];
        if caches.iter().any(|cache| {
            !Arc::ptr_eq(cache.arena(), first.arena())
                || cache.layer_binding(physical_layer).ok()
                    != first.layer_binding(physical_layer).ok()
        }) {
            return Err(Error::InvalidInput(
                "Qwen3.5 decode rows must share one arena and sparse layer binding".into(),
            ));
        }
        if slots.arena_id() != first.arena().id() || slots.len() != batch_size {
            return Err(Error::InvalidInput(
                "Qwen3.5 decode received an incompatible prepared slot map".into(),
            ));
        }
        let [q_out, k_out, v_out] =
            forward_shared(hidden_states, [&self.q_proj, &self.k_proj, &self.v_proj])?;
        let attended = self.decode_attend(
            &q_out,
            &k_out,
            &v_out,
            rope,
            caches,
            slots,
            metadata,
            completions,
            physical_layer,
        )?;
        self.decode_output(&attended, &q_out)
    }

    /// The arena's attention for decode rows, from the q/k/v projection
    /// outputs (`q_out` in the gated `[heads, 2 * head_dim]` layout): q/k
    /// norms, RoPE, the KV write and the paged decode. Returns
    /// `[batch, 1, heads * head_dim]` before the output gate.
    #[allow(clippy::too_many_arguments)]
    fn decode_attend(
        &self,
        q_out: &Tensor,
        k_out: &Tensor,
        v_out: &Tensor,
        rope: Option<&Qwen35RopeTable>,
        caches: &[&PhysicalPagedKvCache],
        slots: &dyn KvSlotMap,
        metadata: &KvDecodeBatchMetadata,
        completions: &mut KvWriteCompletionCollector,
        physical_layer: usize,
    ) -> Result<Tensor> {
        let batch_size = q_out.dim(0)?;
        let first = caches[0];
        let q_proj = q_out.reshape((batch_size, 1, self.num_heads, self.head_dim * 2))?;
        let query_states = q_proj.narrow(3, 0, self.head_dim)?;
        let key_states = k_out.reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?;
        let value_states = v_out.reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?;
        let query_states = self.q_norm.forward(&query_states.contiguous()?)?;
        let key_states = self.k_norm.forward(&key_states.contiguous()?)?;
        let mut queries = Vec::with_capacity(batch_size);
        let mut keys = Vec::with_capacity(batch_size);
        let mut values = Vec::with_capacity(batch_size);
        for row in 0..batch_size {
            let q_row = query_states.i(row)?.unsqueeze(0)?;
            let k_row = key_states.i(row)?.unsqueeze(0)?;
            let row_rope = rope
                .map(|table| -> Result<(Tensor, Tensor)> {
                    Ok((
                        table.cos.narrow(0, row, 1)?.contiguous()?,
                        table.sin.narrow(0, row, 1)?.contiguous()?,
                    ))
                })
                .transpose()?;
            let (q_row, k_row) = self.apply_rope(
                &q_row,
                &k_row,
                row_rope.as_ref().map(|(cos, sin)| (cos, sin)),
            )?;
            queries.push(q_row.reshape((self.num_heads, self.head_dim))?);
            keys.push(k_row.reshape((self.num_kv_heads, self.head_dim))?);
            values.push(
                value_states
                    .i(row)?
                    .reshape((self.num_kv_heads, self.head_dim))?,
            );
        }
        let query_refs = queries.iter().collect::<Vec<_>>();
        let key_refs = keys.iter().collect::<Vec<_>>();
        let value_refs = values.iter().collect::<Vec<_>>();
        let queries = Tensor::stack(&query_refs, 0)?.contiguous()?;
        let keys = Tensor::stack(&key_refs, 0)?.contiguous()?;
        let values = Tensor::stack(&value_refs, 0)?.contiguous()?;
        let storage_dtype = first.arena().config().dtype;
        let output_dtype = queries.dtype();
        let queries = queries.to_dtype(storage_dtype)?;
        let keys = keys.to_dtype(storage_dtype)?;
        let values = values.to_dtype(storage_dtype)?;
        let binding = first.layer_binding(physical_layer)?;
        let completion = first.arena().write_slots(
            binding,
            KvWriteArgs {
                keys: &keys,
                values: &values,
                slots,
            },
        )?;
        let (output, completion) = submit_ordered_after_write(completion, || {
            first.arena().paged_decode(
                binding,
                PagedKvDecodeArgs {
                    queries: &queries,
                    batch: metadata,
                    softmax_scale: 1.0 / (self.head_dim as f32).sqrt(),
                    softcap: None,
                },
            )
        })?;
        completions.collect(completion)?;
        output
            .to_dtype(output_dtype)?
            .reshape((batch_size, 1, self.num_heads * self.head_dim))
            .map_err(Error::from)
    }

    /// Gate the attention output with the second half of each gated q head,
    /// then project it back to the hidden size.
    fn decode_output(&self, attended: &Tensor, q_out: &Tensor) -> Result<Tensor> {
        let batch_size = attended.dim(0)?;
        let gate = q_out
            .reshape((batch_size, 1, self.num_heads, self.head_dim * 2))?
            .narrow(3, self.head_dim, self.head_dim)?
            .reshape((batch_size, 1, self.num_heads * self.head_dim))?;
        let output = (attended * &ops::sigmoid(&gate)?)?;
        self.o_proj.forward(&output)
    }

    /// Rotate the leading `rope_dim` channels of `query_states` and
    /// `key_states` (`[batch, tokens, heads, head_dim]`) with a prebuilt
    /// `(batch, tokens, rope_dim / 2)` cos/sin table. `None` means no rotary
    /// dimensions.
    fn apply_rope(
        &self,
        query_states: &Tensor,
        key_states: &Tensor,
        rope: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor)> {
        let Some((cos, sin)) = rope.filter(|_| self.rope_dim != 0) else {
            return Ok((query_states.clone(), key_states.clone()));
        };
        let tokens = query_states.dim(1)?;
        if cos.dims() != [query_states.dim(0)?, tokens, self.rope_dim / 2] {
            return Err(Error::InvalidInput(format!(
                "Qwen3.5 rotary table {:?} does not match queries {:?}",
                cos.dims(),
                query_states.dims()
            )));
        }

        let query_rot = query_states.narrow(3, 0, self.rope_dim)?.contiguous()?;
        let key_rot = key_states.narrow(3, 0, self.rope_dim)?.contiguous()?;
        let (query_rot, key_rot) = if self.should_try_rope_kernel(query_states.dtype()) {
            match try_apply_rope_thd(&query_rot, &key_rot, cos, sin)? {
                Some((query_rot, key_rot)) => {
                    for _ in 0..tokens {
                        record_rope_kernel();
                    }
                    (query_rot, key_rot)
                }
                None => {
                    for _ in 0..tokens {
                        record_rope_manual();
                    }
                    (
                        apply_rotary_emb(&query_rot, cos, sin)?,
                        apply_rotary_emb(&key_rot, cos, sin)?,
                    )
                }
            }
        } else {
            for _ in 0..tokens {
                record_rope_manual();
            }
            (
                apply_rotary_emb(&query_rot, cos, sin)?,
                apply_rotary_emb(&key_rot, cos, sin)?,
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

    /// Host-built cos/sin table for `positions`, row-major over `shape`
    /// (`rows * tokens == positions.len()`); `None` without rotary dims.
    fn rope_table(
        &self,
        positions: &[[usize; 3]],
        shape: (usize, usize),
        device: &Device,
        dtype: DType,
    ) -> Result<Option<Qwen35RopeTable>> {
        if self.rope_dim == 0 {
            return Ok(None);
        }
        let (cos, sin) = build_mrope_table(
            self.rope_dim,
            positions,
            shape,
            &self.mrope_sections,
            &self.rope_inv_freqs,
            device,
            dtype,
        )?;
        Ok(Some(Qwen35RopeTable { cos, sin }))
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

impl Qwen35LinearAttention {
    fn load_via(
        source: &dyn Qwen35WeightSource,
        device: &Device,
        prefix: &str,
        cfg: &Qwen35TextConfig,
    ) -> Result<Self> {
        let num_k_heads = cfg.ssm_group_count;
        let num_v_heads = cfg.ssm_time_step_rank;
        let head_k_dim = cfg.ssm_state_size;
        let head_v_dim = cfg.ssm_inner_size / cfg.ssm_time_step_rank;
        let conv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

        let dt_bias_name = if source.has(&format!("{prefix}.ssm_dt.bias")) {
            format!("{prefix}.ssm_dt.bias")
        } else {
            format!("{prefix}.ssm_dt")
        };
        let dt_bias = source
            .dense(&dt_bias_name, Some(DType::F32), device)?
            .reshape((num_v_heads,))?
            .reshape((1, 1, num_v_heads))?;
        if dt_bias.elem_count() != num_v_heads {
            return Err(Error::ModelLoadError(format!(
                "Unexpected tensor size for {dt_bias_name}: expected {num_v_heads} elements, found {}",
                dt_bias.elem_count()
            )));
        }
        let a_name = format!("{prefix}.ssm_a");
        let a = source
            .dense(&a_name, Some(DType::F32), device)?
            .reshape((num_v_heads,))?
            .reshape((1, 1, num_v_heads))?;
        if a.elem_count() != num_v_heads {
            return Err(Error::ModelLoadError(format!(
                "Unexpected tensor size for {a_name}: expected {num_v_heads} elements, found {}",
                a.elem_count()
            )));
        }
        let conv_kernel = normalize_conv_kernel(
            source.dense(
                &format!("{prefix}.ssm_conv1d.weight"),
                Some(DType::F32),
                device,
            )?,
            conv_dim,
            cfg.ssm_conv_kernel,
        )?;
        let conv_kernel_slices = pre_slice_conv_kernel(&conv_kernel, cfg.ssm_conv_kernel)?;
        let norm_weight_name = format!("{prefix}.ssm_norm.weight");
        let norm_weight = source
            .dense(&norm_weight_name, Some(DType::F32), device)?
            .reshape((head_v_dim,))?;
        if norm_weight.elem_count() != head_v_dim {
            return Err(Error::ModelLoadError(format!(
                "Unexpected tensor size for {norm_weight_name}: expected {head_v_dim} elements, found {}",
                norm_weight.elem_count()
            )));
        }
        let norm = Qwen35GatedRmsNorm {
            weight: norm_weight,
            eps: cfg.attention_layer_norm_rms_epsilon,
        };

        Ok(Self {
            qkv_proj: Qwen35Linear::new(
                source.qmatmul(&format!("{prefix}.attn_qkv.weight"), device)?,
            ),
            gate_proj: Qwen35Linear::new(
                source.qmatmul(&format!("{prefix}.attn_gate.weight"), device)?,
            ),
            beta_proj: Qwen35Linear::new(
                source.qmatmul(&format!("{prefix}.ssm_beta.weight"), device)?,
            ),
            alpha_proj: Qwen35Linear::new(
                source.qmatmul(&format!("{prefix}.ssm_alpha.weight"), device)?,
            ),
            dt_bias,
            a,
            conv_kernel,
            conv_kernel_slices,
            norm,
            out_proj: Qwen35Linear::new(
                source.qmatmul(&format!("{prefix}.ssm_out.weight"), device)?,
            ),
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_dim,
            kernel_size: cfg.ssm_conv_kernel,
            tiled_recurrence_enabled: qwen35_tiled_recurrence_enabled(),
            tiled_recurrence_tile_size_override: qwen35_tiled_recurrence_tile_size_override(),
            fused_decode: None,
            fused_decode_path: Qwen35FastPath::legacy("unresolved"),
        })
    }

    /// Resolve the fused single-token DeltaNet decode (`kernels/cuda/gdn.rs`,
    /// which also dispatches the Metal kernels) for this layer. Production
    /// enables it on CUDA and Metal (`allow_cpu` lets tests run the portable
    /// reference) for 128-dim heads and a 4-tap conv, after a self-check
    /// against the Candle op chain on this layer's own weights. GGUF stores
    /// value heads in llama.cpp's tiled order.
    fn resolve_fused_decode(
        &mut self,
        hidden: usize,
        device: &Device,
        switch_off: Option<&'static str>,
        allow_cpu: bool,
    ) {
        self.fused_decode = None;
        self.fused_decode_path = if legacy_requested(FUSED_DECODE_ENV) {
            Qwen35FastPath::legacy(format!("{FUSED_DECODE_ENV}=legacy"))
        } else if let Some(reason) = switch_off {
            Qwen35FastPath::legacy(reason)
        } else if !(device.is_cuda() || device.is_metal() || (allow_cpu && device.is_cpu())) {
            Qwen35FastPath::legacy("fused DeltaNet decode runs on CUDA and Metal only")
        } else if !gdn::supported(device, self.head_k_dim, self.head_v_dim, self.kernel_size)
            || self.num_k_heads == 0
            || !self.num_v_heads.is_multiple_of(self.num_k_heads)
        {
            Qwen35FastPath::legacy(
                "fused DeltaNet decode needs SM80+, 128-dim heads, a 4-tap conv and value heads divisible by key heads",
            )
        } else {
            let spec = GdnDecodeSpec {
                key_heads: self.num_k_heads,
                value_heads: self.num_v_heads,
                grouped: false,
                norm_eps: self.norm.eps as f32,
            };
            match self.fused_decode_self_check(&spec, hidden, device) {
                Ok(()) => {
                    self.fused_decode = Some(spec);
                    Qwen35FastPath::Fused
                }
                Err(error) => {
                    tracing::warn!(
                        %error,
                        "Qwen3.5 fused DeltaNet decode self-check failed; using the Candle op chain"
                    );
                    Qwen35FastPath::legacy(format!("self-check failed: {error}"))
                }
            }
        };
    }

    /// One synthetic decode step through both paths from identical states,
    /// with unstructured inputs. Outputs, recurrent states and conv histories
    /// must match.
    fn fused_decode_self_check(
        &self,
        spec: &GdnDecodeSpec,
        hidden: usize,
        device: &Device,
    ) -> Result<()> {
        let host = |tensor: &Tensor| -> Result<Vec<f32>> {
            Ok(tensor
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?)
        };
        let x = Tensor::from_vec(xorshift_values(hidden, 1, 1.5), (1, 1, hidden), device)?;
        let slot = |seed: u64| -> Result<Tensor> {
            Ok(Tensor::from_vec(
                xorshift_values(self.conv_dim, seed, 1.0),
                (self.conv_dim, 1),
                device,
            )?)
        };
        let state_len = self.num_v_heads * self.head_k_dim * self.head_v_dim;
        let initial = Qwen35LayerRuntimeState::Linear {
            conv_state: Some(ConvRingState {
                slots: vec![slot(2)?, slot(3)?, slot(4)?],
                next_idx: 1,
            }),
            recurrent_state: Some(Tensor::from_vec(
                xorshift_values(state_len, 5, 0.3),
                (1, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                device,
            )?),
        };
        let mut legacy_state = initial.clone();
        let mut fused_state = initial;
        let expected = self.forward_legacy(&x, &mut legacy_state)?;
        let actual = self.forward_fused(spec, &x, &mut fused_state)?;
        compare_values(
            "DeltaNet decode output",
            &host(&actual)?,
            &host(&expected)?,
            1e-3,
            1e-2,
        )?;
        let (
            Qwen35LayerRuntimeState::Linear {
                conv_state: Some(legacy_ring),
                recurrent_state: Some(legacy_recurrent),
            },
            Qwen35LayerRuntimeState::Linear {
                conv_state: Some(fused_ring),
                recurrent_state: Some(fused_recurrent),
            },
        ) = (&legacy_state, &fused_state)
        else {
            return Err(Error::InferenceError(
                "DeltaNet decode self-check lost its layer state".into(),
            ));
        };
        compare_values(
            "DeltaNet recurrent state",
            &host(fused_recurrent)?,
            &host(legacy_recurrent)?,
            1e-3,
            1e-2,
        )?;
        // The rings may differ physically (fresh slots vs cursor); their
        // logical oldest-first histories must be identical.
        compare_values(
            "DeltaNet conv history",
            &host(&fused_ring.history()?)?,
            &host(&legacy_ring.history()?)?,
            1e-6,
            1e-6,
        )?;
        Ok(())
    }

    /// Whether a single-token decode can take the fused kernels with this
    /// layer state (3-slot F32 ring, F32 or absent recurrent state).
    fn fused_decode_accepts(hidden_states: &Tensor, state: &Qwen35LayerRuntimeState) -> bool {
        hidden_states.dtype() == DType::F32
            && matches!(
                state,
                Qwen35LayerRuntimeState::Linear {
                    conv_state: Some(ring),
                    recurrent_state,
                } if ring.slots.len() == gdn::CONV_TAPS - 1
                    && ring.next_idx < ring.slots.len()
                    && ring.slots.iter().all(|slot| slot.dtype() == DType::F32)
                    && recurrent_state.as_ref().is_none_or(|s| s.dtype() == DType::F32)
            )
    }

    /// One token's conv, recurrence and gated norm through the fused kernels,
    /// from one row of each projection (`beta_raw` before its sigmoid,
    /// `alpha` before softplus). Advances the ring and replaces the recurrent
    /// state only after both kernels succeed; returns the out-projection
    /// input `[value_heads * head_v_dim]`.
    fn fused_decode_row(
        &self,
        spec: &GdnDecodeSpec,
        mixed_qkv: &Tensor,
        z: &Tensor,
        beta_raw: &Tensor,
        alpha: &Tensor,
        state: &mut Qwen35LayerRuntimeState,
    ) -> Result<Tensor> {
        let Qwen35LayerRuntimeState::Linear {
            conv_state: Some(ring),
            recurrent_state,
        } = state
        else {
            return Err(Error::InferenceError(
                "Qwen3.5 fused DeltaNet decode needs an initialized linear-attention state".into(),
            ));
        };
        let history = ring.ordered_slots();
        let (conv, next_history) = gdn::conv_decode(
            mixed_qkv,
            &self.conv_kernel,
            [&history[0], &history[1], &history[2]],
        )?;
        let zeros;
        let previous = match recurrent_state.as_ref() {
            Some(previous) => previous,
            None => {
                zeros = Tensor::zeros(
                    (1, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                    DType::F32,
                    conv.device(),
                )?;
                &zeros
            }
        };
        let (y, next) = gdn::recurrent_decode(
            &conv,
            z,
            beta_raw,
            alpha,
            &self.dt_bias,
            &self.a,
            &self.norm.weight,
            previous,
            spec,
        )?;
        *ring = ConvRingState::from_history(&next_history, self.conv_dim)?;
        *recurrent_state = Some(next);
        Ok(y)
    }

    fn forward_fused(
        &self,
        spec: &GdnDecodeSpec,
        hidden_states: &Tensor,
        state: &mut Qwen35LayerRuntimeState,
    ) -> Result<Tensor> {
        let [mixed_qkv, z, beta_raw, alpha] = forward_shared(
            hidden_states,
            [
                &self.qkv_proj,
                &self.gate_proj,
                &self.beta_proj,
                &self.alpha_proj,
            ],
        )?;
        let y = self.fused_decode_row(spec, &mixed_qkv, &z, &beta_raw, &alpha, state)?;
        self.out_proj
            .forward(&y.reshape((1, 1, self.num_v_heads * self.head_v_dim))?)
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        state: &mut Qwen35LayerRuntimeState,
    ) -> Result<Tensor> {
        match &self.fused_decode {
            Some(spec) if Self::fused_decode_accepts(hidden_states, state) => {
                self.forward_fused(spec, hidden_states, state)
            }
            _ => self.forward_legacy(hidden_states, state),
        }
    }

    fn forward_legacy(
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

        let [mixed_qkv, z, beta_raw, alpha] = forward_shared(
            hidden_states,
            [
                &self.qkv_proj,
                &self.gate_proj,
                &self.beta_proj,
                &self.alpha_proj,
            ],
        )?;
        let beta = ops::sigmoid(&beta_raw)?;
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
        // The recurrent decode output owns a fresh Candle-managed allocation;
        // retaining it avoids a full state-sized copy on every layer/token.
        *recurrent_state = Some(next_state);

        let output = output.reshape((self.num_v_heads, self.head_v_dim))?;
        let z = z.reshape((self.num_v_heads, self.head_v_dim))?;
        let output = self.norm.forward(&output, &z)?;
        let output = output.reshape((1, 1, self.num_v_heads * self.head_v_dim))?;
        self.out_proj.forward(&output)
    }

    fn forward_decode_batch(
        &self,
        hidden_states: &Tensor,
        states: &mut [&mut Qwen35LayerRuntimeState],
    ) -> Result<Tensor> {
        let batch_size = hidden_states.dim(0)?;
        if hidden_states.dim(1)? != 1 || batch_size == 0 || states.len() != batch_size {
            return Err(Error::InvalidInput(
                "Qwen3.5 linear-attention decode batch dimensions do not match".into(),
            ));
        }
        if let Some(spec) = self.fused_decode.as_ref().filter(|_| {
            states
                .iter()
                .all(|state| Self::fused_decode_accepts(hidden_states, state))
        }) {
            return self.forward_decode_batch_fused(spec, hidden_states, states);
        }
        let [mixed_qkv, z, beta_raw, alpha] = forward_shared(
            hidden_states,
            [
                &self.qkv_proj,
                &self.gate_proj,
                &self.beta_proj,
                &self.alpha_proj,
            ],
        )?;
        let beta = ops::sigmoid(&beta_raw)?;
        if self.num_k_heads == 0 || !self.num_v_heads.is_multiple_of(self.num_k_heads) {
            return Err(Error::InferenceError(format!(
                "Invalid linear-attention head layout: num_v_heads={}, num_k_heads={}",
                self.num_v_heads, self.num_k_heads
            )));
        }
        let key_width = self.num_k_heads * self.head_k_dim;
        let value_width = self.num_v_heads * self.head_v_dim;
        let mut output_rows = Vec::with_capacity(batch_size);
        let mut gate_rows = Vec::with_capacity(batch_size);
        for row in 0..batch_size {
            let (conv_state, recurrent_state) = match &mut *states[row] {
                Qwen35LayerRuntimeState::Linear {
                    conv_state,
                    recurrent_state,
                } => (conv_state, recurrent_state),
                _ => {
                    return Err(Error::InferenceError(
                        "Qwen3.5 layer runtime state does not match linear-attention layer".into(),
                    ))
                }
            };
            let mixed_qkv = mixed_qkv.i(row)?.unsqueeze(0)?;
            let z = z.i(row)?.unsqueeze(0)?;
            let alpha = alpha.i(row)?.unsqueeze(0)?;
            let beta = beta.i(row)?.unsqueeze(0)?;
            let g = softplus(&alpha.broadcast_add(&self.dt_bias)?)?.broadcast_mul(&self.a)?;
            let mixed_qkv = self.depthwise_conv_step(&mixed_qkv, conv_state)?;
            let query = mixed_qkv.narrow(2, 0, key_width)?.reshape((
                1,
                self.num_k_heads,
                self.head_k_dim,
            ))?;
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
            *recurrent_state = Some(next_state);
            output_rows.push(output.reshape((self.num_v_heads, self.head_v_dim))?);
            gate_rows.push(z.reshape((self.num_v_heads, self.head_v_dim))?);
        }
        let output_refs = output_rows.iter().collect::<Vec<_>>();
        let gate_refs = gate_rows.iter().collect::<Vec<_>>();
        let output = Tensor::stack(&output_refs, 0)?;
        let gate = Tensor::stack(&gate_refs, 0)?;
        let output = self.norm.forward(&output, &gate)?.reshape((
            batch_size,
            1,
            self.num_v_heads * self.head_v_dim,
        ))?;
        self.out_proj.forward(&output)
    }

    /// Batched decode through the fused kernels: the projections run once
    /// for every row, then each row's conv and recurrence run on its own
    /// state.
    fn forward_decode_batch_fused(
        &self,
        spec: &GdnDecodeSpec,
        hidden_states: &Tensor,
        states: &mut [&mut Qwen35LayerRuntimeState],
    ) -> Result<Tensor> {
        let batch_size = hidden_states.dim(0)?;
        let [mixed_qkv, z, beta_raw, alpha] = forward_shared(
            hidden_states,
            [
                &self.qkv_proj,
                &self.gate_proj,
                &self.beta_proj,
                &self.alpha_proj,
            ],
        )?;
        let mut rows = Vec::with_capacity(batch_size);
        for (row, state) in states.iter_mut().enumerate() {
            rows.push(self.fused_decode_row(
                spec,
                &mixed_qkv.i(row)?,
                &z.i(row)?,
                &beta_raw.i(row)?,
                &alpha.i(row)?,
                state,
            )?);
        }
        let row_refs = rows.iter().collect::<Vec<_>>();
        let output = Tensor::stack(&row_refs, 0)?.reshape((
            batch_size,
            1,
            self.num_v_heads * self.head_v_dim,
        ))?;
        self.out_proj.forward(&output)
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

        let [mixed_qkv, z, beta_raw, alpha] = forward_shared(
            hidden_states,
            [
                &self.qkv_proj,
                &self.gate_proj,
                &self.beta_proj,
                &self.alpha_proj,
            ],
        )?;
        let beta = ops::sigmoid(&beta_raw)?;
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
        // Sequence kernels may return the final state as a view into their
        // packed output. Detach only this fixed-size escape value, using
        // Candle-managed storage rather than an application private buffer.
        *recurrent_state = Some(deep_copy_tensor_storage(&next_state)?);

        let output = output.reshape((seq_len * self.num_v_heads, self.head_v_dim))?;
        let z = z.reshape((seq_len * self.num_v_heads, self.head_v_dim))?;
        let output = self.norm.forward(&output, &z)?;
        let output = output.reshape((1, seq_len, self.num_v_heads * self.head_v_dim))?;
        self.out_proj.forward(&output)
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
    QMatMul::from_arc(weights).map_err(Error::from)
}

fn load_rms_norm(
    loader: &GgufLoader,
    device: &Device,
    name: &str,
    eps: f64,
) -> Result<RmsNorm> {
    RmsNorm::from_qtensor(loader.load_qtensor(name, device)?, eps).map_err(Error::from)
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

/// Interleaved M-RoPE angles for one position. Each angle is the F32
/// product `position * inv_freq`, as in HF (`inv_freq` and positions in
/// fp32), so the rounding matches the reference implementation.
fn mrope_angles(
    rope_dim: usize,
    position_ids: [usize; 3],
    mrope_sections: &[usize],
    inv_freqs: &[f32],
) -> Result<Vec<f32>> {
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
    Ok(interleaved)
}

/// Cos/sin tables `(rows, tokens, rope_dim / 2)` for row-major `positions`.
///
/// Built on the host and uploaded in one copy: every full-attention layer
/// of a forward shares one table, instead of each layer (and, in prefill,
/// each token) building its own on the device. Cos/sin of the F32 angle are
/// taken in F64, because Candle compiles its Metal kernels with fast math
/// (errors near 1e-2 at long positions).
fn build_mrope_table(
    rope_dim: usize,
    positions: &[[usize; 3]],
    (rows, tokens): (usize, usize),
    mrope_sections: &[usize],
    inv_freqs: &[f32],
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let half_dim = rope_dim / 2;
    if rows.checked_mul(tokens) != Some(positions.len()) || positions.is_empty() {
        return Err(Error::InvalidInput(format!(
            "Qwen3.5 rotary table shape ({rows}, {tokens}) does not cover {} positions",
            positions.len()
        )));
    }
    let table_len = positions.len() * half_dim;
    let mut values = vec![0f32; table_len * 2];
    let (cos_values, sin_values) = values.split_at_mut(table_len);
    for (index, position_ids) in positions.iter().enumerate() {
        let angles = mrope_angles(rope_dim, *position_ids, mrope_sections, inv_freqs)?;
        let row = index * half_dim..(index + 1) * half_dim;
        for ((cos, sin), angle) in cos_values[row.clone()]
            .iter_mut()
            .zip(&mut sin_values[row])
            .zip(angles)
        {
            let (angle_sin, angle_cos) = f64::from(angle).sin_cos();
            *cos = angle_cos as f32;
            *sin = angle_sin as f32;
        }
    }
    let table = Tensor::from_vec(values, (2, rows, tokens, half_dim), device)?.to_dtype(dtype)?;
    Ok((table.get(0)?, table.get(1)?))
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
        failure => {
            crate::models::shared::telemetry::report_kernel_fallback("qwen35.rope", &failure);
            Ok(None)
        }
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

fn qwen35_tiled_recurrence_enabled() -> bool {
    crate::env::flag_or("IZWI_QWEN35_TILED_RECURRENCE", true)
}

fn qwen35_rope_kernel_enabled(device: &Device) -> bool {
    qwen35_rope_kernel_policy(
        device.is_metal(),
        device.is_cuda(),
        crate::env::flag("IZWI_QWEN35_ROPE_KERNEL"),
    )
}

fn qwen35_rope_kernel_policy(
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
        apply_rotary_emb, build_mrope_table, convolution_domain_v2, non_finite_counts,
        owned_zero_tensor, qwen35_rope_kernel_policy, recurrent_domain_v2, repeat_head_states,
        repeat_head_states_seq, softplus, ConvRingState, Qwen35GatedRmsNorm,
        Qwen35LayerRuntimeState, Qwen35LinearAttention, Qwen35TextRuntimeState,
    };
    use crate::models::architectures::qwen35::cache::{
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

    /// A DeltaNet layer at the fused kernels' geometry (128-dim heads, a
    /// 4-tap conv) with unstructured weights, small enough for the CPU.
    fn synthetic_fused_layer(
        device: &Device,
        key_heads: usize,
        value_heads: usize,
        hidden: usize,
    ) -> Qwen35LinearAttention {
        use super::xorshift_values;
        let d = super::gdn::HEAD_DIM;
        let conv_dim = (2 * key_heads + value_heads) * d;
        let tensor =
            |values: Vec<f32>, shape: &[usize]| Tensor::from_vec(values, shape, device).unwrap();
        let dense = |out: usize, input: usize, seed: u64, scale: f32| {
            super::Qwen35Linear::new(QMatMul::Tensor(tensor(
                xorshift_values(out * input, seed, scale),
                &[out, input],
            )))
        };
        let conv_kernel = tensor(xorshift_values(conv_dim * 4, 11, 0.4), &[conv_dim, 4]);
        let conv_kernel_slices = super::pre_slice_conv_kernel(&conv_kernel, 4).unwrap();
        let a = xorshift_values(value_heads, 12, 1.0)
            .into_iter()
            .map(|value| -0.05 - 0.4 * value.abs())
            .collect();
        let norm = xorshift_values(d, 13, 0.3)
            .into_iter()
            .map(|value| 1.0 + value)
            .collect();
        Qwen35LinearAttention {
            qkv_proj: dense(conv_dim, hidden, 1, 0.15),
            gate_proj: dense(value_heads * d, hidden, 2, 0.15),
            beta_proj: dense(value_heads, hidden, 3, 0.2),
            alpha_proj: dense(value_heads, hidden, 4, 0.2),
            dt_bias: tensor(xorshift_values(value_heads, 5, 0.5), &[1, 1, value_heads]),
            a: tensor(a, &[1, 1, value_heads]),
            conv_kernel,
            conv_kernel_slices,
            norm: Qwen35GatedRmsNorm {
                weight: tensor(norm, &[d]),
                eps: 1e-6,
            },
            out_proj: dense(hidden, value_heads * d, 6, 0.05),
            num_k_heads: key_heads,
            num_v_heads: value_heads,
            head_k_dim: d,
            head_v_dim: d,
            conv_dim,
            kernel_size: 4,
            tiled_recurrence_enabled: false,
            tiled_recurrence_tile_size_override: None,
            fused_decode: None,
            fused_decode_path: super::Qwen35FastPath::legacy("unresolved"),
        }
    }

    fn fused_layer_initial_state(
        layer: &Qwen35LinearAttention,
        seed: u64,
    ) -> Qwen35LayerRuntimeState {
        use super::xorshift_values;
        let device = layer.conv_kernel.device();
        let slot = |offset: u64| {
            Tensor::from_vec(
                xorshift_values(layer.conv_dim, seed + offset, 1.0),
                (layer.conv_dim, 1),
                device,
            )
            .unwrap()
        };
        Qwen35LayerRuntimeState::Linear {
            conv_state: Some(ConvRingState {
                slots: vec![slot(1), slot(2), slot(3)],
                next_idx: (seed % 3) as usize,
            }),
            recurrent_state: Some(
                Tensor::from_vec(
                    xorshift_values(layer.num_v_heads * 128 * 128, seed + 4, 0.2),
                    (1, layer.num_v_heads, 128, 128),
                    device,
                )
                .unwrap(),
            ),
        }
    }

    fn values(tensor: &Tensor) -> Vec<f32> {
        tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    fn linear_state_parts(state: &Qwen35LayerRuntimeState) -> (Vec<f32>, Vec<f32>) {
        let Qwen35LayerRuntimeState::Linear {
            conv_state: Some(ring),
            recurrent_state: Some(recurrent),
        } = state
        else {
            panic!("linear state is incomplete")
        };
        (values(&ring.history().unwrap()), values(recurrent))
    }

    /// Decode several tokens through the fused path and the Candle chain from
    /// one state, solo and batched, and compare outputs and states.
    fn assert_fused_decode_matches_legacy(layer: &Qwen35LinearAttention, hidden: usize) {
        use super::{compare_values, xorshift_values};
        let device = layer.conv_kernel.device();
        let mut fused = [
            fused_layer_initial_state(layer, 20),
            fused_layer_initial_state(layer, 30),
        ];
        let mut legacy = fused.clone();
        for step in 0..3u64 {
            let input = Tensor::from_vec(
                xorshift_values(2 * hidden, 40 + step, 1.5),
                (2, 1, hidden),
                device,
            )
            .unwrap();
            let expected = (0..2)
                .map(|row| {
                    layer
                        .forward_legacy(
                            &input.i(row).unwrap().unsqueeze(0).unwrap(),
                            &mut legacy[row],
                        )
                        .unwrap()
                })
                .collect::<Vec<_>>();
            let actual = if step == 1 {
                let mut refs = fused.iter_mut().collect::<Vec<_>>();
                layer.forward_decode_batch(&input, &mut refs).unwrap()
            } else {
                let rows = (0..2)
                    .map(|row| {
                        layer
                            .forward(
                                &input.i(row).unwrap().unsqueeze(0).unwrap(),
                                &mut fused[row],
                            )
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                Tensor::cat(&rows.iter().collect::<Vec<_>>(), 0).unwrap()
            };
            let expected = Tensor::cat(&expected.iter().collect::<Vec<_>>(), 0).unwrap();
            compare_values("output", &values(&actual), &values(&expected), 1e-4, 1e-3)
                .unwrap_or_else(|error| panic!("step {step}: {error}"));
            for (fused, legacy) in fused.iter().zip(&legacy) {
                let (fused_ring, fused_recurrent) = linear_state_parts(fused);
                let (legacy_ring, legacy_recurrent) = linear_state_parts(legacy);
                // The batched projection may round differently from the
                // single-row one in the last bit.
                compare_values("conv history", &fused_ring, &legacy_ring, 1e-6, 1e-5)
                    .unwrap_or_else(|error| panic!("step {step}: {error}"));
                compare_values("state", &fused_recurrent, &legacy_recurrent, 1e-4, 1e-3)
                    .unwrap_or_else(|error| panic!("step {step}: {error}"));
            }
        }
    }

    #[test]
    fn fused_deltanet_decode_matches_the_candle_chain_for_both_head_layouts() {
        let _guard = crate::env_test_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::remove_var(super::FUSED_DECODE_ENV);
        let device = &Device::Cpu;
        // 0.8B/2B pair every value head with its own key head; 4B/9B tile 2x.
        for (key_heads, value_heads) in [(2, 2), (2, 4)] {
            let mut layer = synthetic_fused_layer(device, key_heads, value_heads, 64);
            layer.resolve_fused_decode(64, device, None, true);
            assert_eq!(
                layer.fused_decode_path,
                super::Qwen35FastPath::Fused,
                "{key_heads}/{value_heads}"
            );
            assert_eq!(
                layer.fused_decode.map(|spec| spec.grouped),
                Some(false),
                "GGUF value heads are tiled"
            );
            assert_fused_decode_matches_legacy(&layer, 64);
        }
    }

    #[test]
    fn fused_deltanet_self_check_rejects_the_wrong_value_head_order() {
        let device = &Device::Cpu;
        let layer = synthetic_fused_layer(device, 2, 4, 64);
        let grouped = super::GdnDecodeSpec {
            key_heads: 2,
            value_heads: 4,
            grouped: true,
            norm_eps: 1e-6,
        };
        let error = layer
            .fused_decode_self_check(&grouped, 64, device)
            .expect_err("grouped pairing must not pass for tiled GGUF heads");
        assert!(error.to_string().contains("diverges"), "{error}");
    }

    #[test]
    fn fused_deltanet_decode_honours_its_switches() {
        let _guard = crate::env_test_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let device = &Device::Cpu;
        let mut layer = synthetic_fused_layer(device, 2, 2, 64);
        std::env::set_var(super::FUSED_DECODE_ENV, "legacy");
        layer.resolve_fused_decode(64, device, None, true);
        std::env::remove_var(super::FUSED_DECODE_ENV);
        assert_eq!(
            layer.fused_decode_path,
            super::Qwen35FastPath::legacy("IZWI_QWEN35_FUSED_DECODE=legacy")
        );
        assert!(layer.fused_decode.is_none());

        layer.resolve_fused_decode(64, device, Some("cuda.fused_decode=off"), true);
        assert_eq!(
            layer.fused_decode_path,
            super::Qwen35FastPath::legacy("cuda.fused_decode=off")
        );

        // Production keeps the CPU on the Candle chain.
        layer.resolve_fused_decode(64, device, None, false);
        assert!(layer.fused_decode.is_none());
    }

    #[cfg(feature = "metal")]
    #[test]
    fn fused_deltanet_decode_matches_the_candle_chain_on_metal() {
        let _guard = crate::env_test_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::remove_var(super::FUSED_DECODE_ENV);
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "fused_deltanet_decode_matches_the_candle_chain_on_metal",
        ) else {
            return;
        };
        for (key_heads, value_heads) in [(2, 2), (2, 4)] {
            let mut layer = synthetic_fused_layer(&device, key_heads, value_heads, 64);
            // Production resolution: no CPU allowance needed on Metal.
            layer.resolve_fused_decode(64, &device, None, false);
            assert_eq!(
                layer.fused_decode_path,
                super::Qwen35FastPath::Fused,
                "{key_heads}/{value_heads}"
            );
            assert_fused_decode_matches_legacy(&layer, 64);
        }
    }

    #[test]
    fn batched_linear_decode_matches_independent_scalar_rows_with_ring_history() {
        let device = &Device::Cpu;
        let dense = |out: usize, input: usize, scale: f32| {
            let weights = Tensor::from_vec(
                (0..out * input)
                    .map(|index| scale * ((index % input) + 1) as f32)
                    .collect::<Vec<_>>(),
                (out, input),
                device,
            )
            .unwrap();
            let weights = QTensor::quantize(&weights, GgmlDType::F32).unwrap();
            super::Qwen35Linear::new(QMatMul::from_arc(Arc::new(weights)).unwrap())
        };
        let conv_kernel = Tensor::from_vec(
            (0..24)
                .map(|index| 0.01 * ((index % 4) + 1) as f32)
                .collect::<Vec<_>>(),
            (6, 4),
            device,
        )
        .unwrap();
        let conv_kernel_slices = super::pre_slice_conv_kernel(&conv_kernel, 4).unwrap();
        let mixer = Qwen35LinearAttention {
            qkv_proj: dense(6, 4, 0.01),
            gate_proj: dense(2, 4, 0.02),
            beta_proj: dense(1, 4, -0.02),
            alpha_proj: dense(1, 4, 0.03),
            dt_bias: Tensor::zeros((1, 1, 1), DType::F32, device).unwrap(),
            a: Tensor::full(-0.5f32, (1, 1, 1), device).unwrap(),
            conv_kernel,
            conv_kernel_slices,
            norm: Qwen35GatedRmsNorm {
                weight: Tensor::ones(2, DType::F32, device).unwrap(),
                eps: 1e-6,
            },
            out_proj: dense(4, 2, 0.04),
            num_k_heads: 1,
            num_v_heads: 1,
            head_k_dim: 2,
            head_v_dim: 2,
            conv_dim: 6,
            kernel_size: 4,
            tiled_recurrence_enabled: false,
            tiled_recurrence_tile_size_override: None,
            fused_decode: None,
            fused_decode_path: super::Qwen35FastPath::legacy("test"),
        };
        let initial_state = |row: usize| Qwen35LayerRuntimeState::Linear {
            conv_state: Some(ConvRingState {
                slots: (0..3)
                    .map(|slot| {
                        Tensor::full((row * 3 + slot + 1) as f32 * 0.01, (6, 1), device).unwrap()
                    })
                    .collect(),
                next_idx: row,
            }),
            recurrent_state: None,
        };
        let initial = [initial_state(0), initial_state(1)];
        let input = Tensor::from_vec(
            vec![0.2f32, -0.1, 0.3, 0.4, -0.3, 0.5, 0.1, 0.2],
            (2, 1, 4),
            device,
        )
        .unwrap();
        let mut scalar_states = initial.clone();
        let scalar_rows = (0..2)
            .map(|row| {
                mixer
                    .forward(
                        &input.i(row).unwrap().unsqueeze(0).unwrap(),
                        &mut scalar_states[row],
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let scalar_refs = scalar_rows.iter().collect::<Vec<_>>();
        let scalar = Tensor::cat(&scalar_refs, 0).unwrap();
        let mut batch_states = initial;
        let mut batch_refs = batch_states.iter_mut().collect::<Vec<_>>();
        let batched = mixer.forward_decode_batch(&input, &mut batch_refs).unwrap();
        let scalar_values = scalar.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let batch_values = batched.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (scalar, batch) in scalar_values.iter().zip(&batch_values) {
            assert!((scalar - batch).abs() < 1e-5, "{scalar} != {batch}");
        }
        for (scalar, batch) in scalar_states.iter().zip(&batch_states) {
            let Qwen35LayerRuntimeState::Linear {
                conv_state: Some(scalar_conv),
                recurrent_state: Some(scalar_recurrent),
            } = scalar
            else {
                panic!("scalar row omitted hybrid state")
            };
            let Qwen35LayerRuntimeState::Linear {
                conv_state: Some(batch_conv),
                recurrent_state: Some(batch_recurrent),
            } = batch
            else {
                panic!("batch row omitted hybrid state")
            };
            assert_eq!(scalar_conv.next_idx, batch_conv.next_idx);
            assert_eq!(scalar_conv.slots.len(), batch_conv.slots.len());
            for (scalar, batch) in scalar_conv.slots.iter().zip(&batch_conv.slots) {
                let scalar = scalar.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let batch = batch.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                for (scalar, batch) in scalar.iter().zip(&batch) {
                    assert!((scalar - batch).abs() < 1e-5, "{scalar} != {batch}");
                }
            }
            let scalar = scalar_recurrent
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let batch = batch_recurrent
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            for (scalar, batch) in scalar.iter().zip(&batch) {
                assert!((scalar - batch).abs() < 1e-5, "{scalar} != {batch}");
            }
        }
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
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "owned_recurrent_states_do_not_alias_on_metal",
        ) else {
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

        let state = Qwen35TextRuntimeState {
            layers: vec![Qwen35LayerRuntimeState::Linear {
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
        let (cos, sin) = build_mrope_table(
            12,
            &[[3, 5, 7]],
            (1, 1),
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

    /// The table holds cos/sin of each F32 angle `position * inv_freq`
    /// (the HF rounding), evaluated in F64, in row-major position order.
    #[test]
    fn rope_table_rows_follow_positions_and_round_like_hf() {
        let inv_freqs = super::build_rope_inv_freqs(64, 10_000_000.0).expect("inv freqs");
        let positions = [[0usize; 3], [7; 3], [4_096; 3], [262_143; 3]];
        for shape in [(4, 1), (1, 4)] {
            let (cos, sin) = build_mrope_table(
                64,
                &positions,
                shape,
                &[11, 11, 10],
                &inv_freqs,
                &Device::Cpu,
                DType::F32,
            )
            .expect("table");
            assert_eq!(cos.dims(), &[shape.0, shape.1, 32]);
            assert_eq!(sin.dims(), &[shape.0, shape.1, 32]);
            let cos = cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let sin = sin.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for (row, position) in positions.iter().enumerate() {
                for (index, inv_freq) in inv_freqs.iter().enumerate() {
                    let angle = f64::from(position[0] as f32 * inv_freq);
                    assert_eq!(cos[row * 32 + index], angle.cos() as f32, "{row}/{index}");
                    assert_eq!(sin[row * 32 + index], angle.sin() as f32, "{row}/{index}");
                }
            }
        }
    }

    /// At short positions the host table agrees with the device-built F32
    /// cos/sin it replaces to within float rounding.
    #[test]
    fn rope_table_matches_the_device_built_f32_values() {
        let inv_freqs = super::build_rope_inv_freqs(64, 10_000_000.0).expect("inv freqs");
        let positions = (0..48).map(|position| [position; 3]).collect::<Vec<_>>();
        let (cos, sin) = build_mrope_table(
            64,
            &positions,
            (1, positions.len()),
            &[11, 11, 10],
            &inv_freqs,
            &Device::Cpu,
            DType::F32,
        )
        .expect("table");
        let angles = positions
            .iter()
            .flat_map(|position| inv_freqs.iter().map(move |inv| position[0] as f32 * inv))
            .collect::<Vec<_>>();
        let device_angles =
            Tensor::from_vec(angles, (1, positions.len(), 32), &Device::Cpu).unwrap();
        for (table, reference) in [
            (cos, device_angles.cos().unwrap()),
            (sin, device_angles.sin().unwrap()),
        ] {
            let table = table.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let reference = reference.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for (value, expected) in table.iter().zip(&reference) {
                assert!((value - expected).abs() <= 2e-7, "{value} vs {expected}");
            }
        }
    }

    #[test]
    fn rope_table_rejects_a_shape_that_does_not_cover_the_positions() {
        let inv_freqs = super::build_rope_inv_freqs(64, 10_000_000.0).expect("inv freqs");
        for (positions, shape) in [(vec![[1usize; 3]; 3], (2, 1)), (Vec::new(), (0, 1))] {
            assert!(build_mrope_table(
                64,
                &positions,
                shape,
                &[11, 11, 10],
                &inv_freqs,
                &Device::Cpu,
                DType::F32,
            )
            .is_err());
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
    fn qwen35_cuda_rope_kernel_defaults_on_with_explicit_rollback() {
        assert!(qwen35_rope_kernel_policy(true, false, None));
        assert!(qwen35_rope_kernel_policy(false, true, None));
        assert!(qwen35_rope_kernel_policy(false, true, Some(true)));
        assert!(!qwen35_rope_kernel_policy(true, false, Some(false)));
        assert!(!qwen35_rope_kernel_policy(false, false, Some(true)));
    }

    fn synthetic_decode_state(
        device: &Device,
        layer_count: usize,
        num_v_heads: usize,
    ) -> Qwen35TextRuntimeState {
        let mut layers = Vec::with_capacity(layer_count);
        for layer_idx in 0..layer_count {
            if (layer_idx + 1).is_multiple_of(4) {
                layers.push(Qwen35LayerRuntimeState::Full);
            } else {
                let conv_width = num_v_heads * 2;
                layers.push(Qwen35LayerRuntimeState::Linear {
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
        Qwen35TextRuntimeState { layers }
    }

    fn advance_synthetic_decode_state(
        state: &mut Qwen35TextRuntimeState,
        device: &Device,
        num_v_heads: usize,
    ) {
        for layer in &mut state.layers {
            match layer {
                Qwen35LayerRuntimeState::Linear {
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
                Qwen35LayerRuntimeState::Full => {}
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
