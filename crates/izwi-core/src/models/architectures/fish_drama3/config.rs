//! Config parsing for Fish Audio Drama 3 (`FishOmniModel`, `fish_omni`).
//!
//! `config.json` carries three parts:
//! * `text_config`: the Qwen3.5-MoE hybrid trunk (Gated DeltaNet and gated
//!   full attention, every layer a routed MoE with a gated shared expert);
//! * `audio_decoder_config`: the codebook-MTP tower (4 shared layers, then
//!   one layer and one head per residual codebook);
//! * `quantization_config`: `bsgemm-moe`, block-FP8 routed experts with F32
//!   128×128 scales.
//!
//! The top level adds the token ids and the prompt template version.
//!
//! Every switch the family implements one way fails closed under both
//! policies: a checkpoint that changed a RoPE layout, a norm, the routing,
//! the codebook placement or the quantization would otherwise load and
//! produce plausible noise. [`FishDrama3ConfigPolicy::Published`] also pins
//! every dimension to the published checkpoint;
//! [`FishDrama3ConfigPolicy::Synthetic`] lets test checkpoints shrink the
//! dimensions while keeping every structural relation. The census
//! (`checkpoint.rs`) then pins the tensors themselves.
//!
//! Three fields do not mean what their names say:
//! * `audio_decoder_config.attention_qk_norm: false` is ignored. The tower
//!   always applies its zero-centred q/k norm: training ran it, but the
//!   exporter never writes the field, so the released config carries the
//!   default. The census holds the `q_norm` / `k_norm` tensors of every
//!   tower layer.
//! * `audio_decoder_config.max_seq_len` (11) is the frame width (one main
//!   token and 10 codes), not an attention window: every tower layer attends
//!   over the whole sequence, text rows included.
//! * `audio_decoder_config.tie_word_embeddings: true` is ignored as well:
//!   the codebook heads are their own tensors.

use std::ops::RangeInclusive;
use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

use crate::error::{Error, Result};

pub const CONFIG_FILE: &str = "config.json";
pub const ARCHITECTURE: &str = "FishOmniModel";
pub const MODEL_TYPE: &str = "fish_omni";
pub const TEXT_MODEL_TYPE: &str = "fish_omni_text";
pub const TOWER_MODEL_TYPE: &str = "fish_qwen3_audio_decoder";
pub const TOWER_DECODER_TYPE: &str = "codebook_mtp_shared";
pub const FISH_SCHEMA_VERSION: u64 = 1;
pub const SPEECH_TEMPLATE_VERSION: &str = "reference-bracket-v1";
pub const QUANT_METHOD: &str = "bsgemm-moe";
/// The FP8 block edge: one F32 scale per 128×128 block of a routed expert.
pub const FP8_BLOCK: usize = 128;
/// `<|speaker:0|>` .. `<|speaker:15|>`.
pub const SPEAKER_TAGS: usize = 16;

/// The published checkpoint's values (`fishaudio/drama-3@78e5ab79`).
mod published {
    pub const LAYERS: usize = 40;
    /// Read from `layer_types` (3 linear : 1 full), never from an interval.
    pub const FULL_ATTENTION_LAYERS: [usize; 10] = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39];
    pub const HIDDEN: usize = 2048;
    pub const VOCAB: usize = 252_288;
    pub const MAX_POSITIONS: usize = 262_144;
    pub const HEADS: usize = 16;
    pub const KV_HEADS: usize = 2;
    pub const HEAD_DIM: usize = 256;
    pub const ROTARY_DIMS: usize = 64;
    pub const ROPE_THETA: f64 = 10_000_000.0;
    pub const LINEAR_KEY_HEADS: usize = 16;
    pub const LINEAR_KEY_DIM: usize = 128;
    pub const LINEAR_VALUE_HEADS: usize = 32;
    pub const LINEAR_VALUE_DIM: usize = 128;
    pub const CONV_KERNEL: usize = 4;
    pub const EXPERTS: usize = 256;
    pub const TOP_K: usize = 8;
    pub const MOE_INTERMEDIATE: usize = 512;
    pub const SHARED_INTERMEDIATE: usize = 512;
    pub const NORM_EPS: f64 = 1e-6;
    pub const CODEBOOKS: usize = 10;
    pub const CODEBOOK_VOCAB: usize = 4096;
    pub const TOWER_SHARED_LAYERS: usize = 4;
    pub const TOWER_INTERMEDIATE: usize = 6144;
    pub const PAD: u32 = 248_044;
    pub const EOS: u32 = 248_046;
    pub const AUDIO_PAD: u32 = 248_076;
    pub const VOICE: u32 = 248_081;
    pub const SPEAKER_FIRST: u32 = 248_083;
    pub const SEMANTIC_START: u32 = 248_099;
    pub const SEMANTIC_END: u32 = 252_194;
}

/// The policy a lifecycle load of `model_dir` holds its `config.json` to:
/// the published one, except for a test build's synthetic checkpoint marked
/// for the lifecycle (`synthetic::LIFECYCLE_MARKER`; this crate's tests, or
/// another crate's through the `test-support` feature). A release build has
/// no way to load anything but the pinned geometry.
pub(crate) fn lifecycle_policy(model_dir: &Path) -> FishDrama3ConfigPolicy {
    #[cfg(any(test, feature = "test-support"))]
    if super::synthetic::marked_for_lifecycle(model_dir) {
        return FishDrama3ConfigPolicy::Synthetic;
    }
    let _ = model_dir;
    FishDrama3ConfigPolicy::Published
}

/// How strictly `config.json` is held.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FishDrama3ConfigPolicy {
    /// Every value pinned to the published checkpoint.
    Published,
    /// Test checkpoints: dimensions are free (the hidden size stays a
    /// multiple of the FP8 block), every switch and structural relation is
    /// enforced.
    Synthetic,
}

/// What one trunk layer mixes; every layer then runs the MoE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FishDrama3LayerKind {
    /// Gated DeltaNet.
    LinearAttention,
    /// Gated full attention with partial RoPE.
    FullAttention,
}

/// How partial RoPE pairs the rotated dims.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FishDrama3RopeLayout {
    /// NeoX: dim `i` pairs with `i + rotary_dims / 2` (the trunk,
    /// `rotary_interleaved: false`).
    HalfSplit,
    /// Dim `2i` pairs with `2i + 1` (the tower, `rotary_interleaved: true`).
    InterleavedPairs,
}

/// Gated GQA attention with partial RoPE and zero-centred q/k norms.
///
/// The trunk's `q_proj` packs each head as `[query | gate]` (per-head
/// interleaved); the tower's `wqkv` is contiguous `[Q | Gate | K | V]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FishDrama3AttentionConfig {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Rotated dims per head: the first `rotary_dims` of `head_dim`.
    pub rotary_dims: usize,
    pub rope_theta: f64,
    pub rope_layout: FishDrama3RopeLayout,
}

impl FishDrama3AttentionConfig {
    /// Query (and gate) width.
    pub fn q_width(&self) -> usize {
        self.heads * self.head_dim
    }

    /// Key (and value) width.
    pub fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }
}

/// Gated DeltaNet geometry. V head `j` reads K head
/// `j / (value_heads / key_heads)` (grouped, HF layout).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FishDrama3LinearAttentionConfig {
    pub key_heads: usize,
    pub key_head_dim: usize,
    pub value_heads: usize,
    pub value_head_dim: usize,
    pub conv_kernel: usize,
}

impl FishDrama3LinearAttentionConfig {
    pub fn key_width(&self) -> usize {
        self.key_heads * self.key_head_dim
    }

    pub fn value_width(&self) -> usize {
        self.value_heads * self.value_head_dim
    }

    /// `in_proj_qkv` rows and the short conv's channels: `q | k | v`.
    pub fn conv_channels(&self) -> usize {
        2 * self.key_width() + self.value_width()
    }
}

/// Softmax top-k routing (renormalised) over FP8 experts, plus a dense
/// shared expert behind a sigmoid gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FishDrama3MoeConfig {
    pub experts: usize,
    pub top_k: usize,
    pub intermediate: usize,
    pub shared_intermediate: usize,
}

/// The Qwen3.5-MoE hybrid trunk.
#[derive(Debug, Clone, PartialEq)]
pub struct FishDrama3TrunkConfig {
    pub hidden: usize,
    pub vocab: usize,
    pub layers: Vec<FishDrama3LayerKind>,
    pub attention: FishDrama3AttentionConfig,
    pub linear: FishDrama3LinearAttentionConfig,
    pub moe: FishDrama3MoeConfig,
    /// Every trunk norm; `(1 + w)` gains except the DeltaNet output norm.
    pub rms_norm_eps: f64,
    pub max_positions: usize,
}

impl FishDrama3TrunkConfig {
    /// Layer indices of `kind`, in order.
    pub fn layers_of(&self, kind: FishDrama3LayerKind) -> Vec<usize> {
        self.layers
            .iter()
            .enumerate()
            .filter_map(|(index, layer)| (*layer == kind).then_some(index))
            .collect()
    }

    pub fn full_attention_layers(&self) -> Vec<usize> {
        self.layers_of(FishDrama3LayerKind::FullAttention)
    }

    pub fn linear_attention_layers(&self) -> Vec<usize> {
        self.layers_of(FishDrama3LayerKind::LinearAttention)
    }
}

/// The codebook-MTP tower: `shared_layers` shared layers, then one layer
/// and one head per residual codebook (`codebooks - 1`), all at the trunk's
/// width.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FishDrama3TowerConfig {
    pub dim: usize,
    /// Codebooks per frame: the semantic one and the residual ones.
    pub codebooks: usize,
    /// Codes per codebook head (the semantic codebook's size).
    pub codebook_vocab: usize,
    pub shared_layers: usize,
    pub attention: FishDrama3AttentionConfig,
    pub intermediate: usize,
    pub norm_eps: f64,
}

impl FishDrama3TowerConfig {
    /// Per-codebook layers and heads.
    pub fn residual_layers(&self) -> usize {
        self.codebooks - 1
    }

    /// One row of the `[main, c0..c9]` prompt layout (`max_seq_len`).
    pub fn frame_width(&self) -> usize {
        self.codebooks + 1
    }

    /// `wqkv` rows: `[Q | Gate | K | V]`.
    pub fn wqkv_rows(&self) -> usize {
        2 * self.attention.q_width() + 2 * self.attention.kv_width()
    }
}

/// The token ids the port uses.
///
/// `config.json` carries pad, eos, audio pad and the semantic band. The
/// speaker tags and `<|voice|>` are tokenizer-only: they sit in the published
/// vocabulary as the 16 ids right below the band and two below those, and
/// the ids here follow that layout (the tokenizer checks the strings).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FishDrama3TokenIds {
    /// `<|endoftext|>` (not S2's `<|pad|>`).
    pub pad: u32,
    /// `<|im_end|>`: the only stop token.
    pub eos: u32,
    pub audio_pad: u32,
    /// `<|voice|>`.
    pub voice: u32,
    /// `<|speaker:0|>`.
    pub speaker_first: u32,
    /// `<|semantic:0|>`.
    pub semantic_start: u32,
    /// `<|semantic:N-1|>`.
    pub semantic_end: u32,
}

impl FishDrama3TokenIds {
    pub fn speakers(&self) -> RangeInclusive<u32> {
        self.speaker_first..=self.speaker_first + SPEAKER_TAGS as u32 - 1
    }

    /// `<|speaker:index|>`.
    pub fn speaker(&self, index: usize) -> Option<u32> {
        (index < SPEAKER_TAGS).then(|| self.speaker_first + index as u32)
    }

    pub fn semantic_codes(&self) -> usize {
        (self.semantic_end - self.semantic_start) as usize + 1
    }

    pub fn is_semantic(&self, id: u32) -> bool {
        (self.semantic_start..=self.semantic_end).contains(&id)
    }

    /// `<|semantic:code|>`.
    pub fn semantic(&self, code: u32) -> Option<u32> {
        let id = self.semantic_start.checked_add(code)?;
        self.is_semantic(id).then_some(id)
    }

    /// The code of a semantic token.
    pub fn semantic_code(&self, id: u32) -> Option<u32> {
        self.is_semantic(id).then(|| id - self.semantic_start)
    }

    /// The main head's candidate rows: eos first, then the semantic band.
    /// Eos sits below the band, so the list is also ascending, as the
    /// checkpoint's row-slice reads require. No engine ever samples another
    /// row, so only these rows of `lm_head` are loaded.
    pub fn compact_head_rows(&self) -> Vec<usize> {
        std::iter::once(self.eos as usize)
            .chain(self.semantic_start as usize..=self.semantic_end as usize)
            .collect()
    }
}

/// Everything the family reads from `config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct FishDrama3Config {
    pub policy: FishDrama3ConfigPolicy,
    pub trunk: FishDrama3TrunkConfig,
    pub tower: FishDrama3TowerConfig,
    /// The FP8 block edge ([`FP8_BLOCK`] under both policies).
    pub fp8_block: usize,
    pub ids: FishDrama3TokenIds,
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    architectures: Vec<String>,
    model_type: String,
    fish_schema_version: u64,
    speech_template_version: String,
    dtype: String,
    #[serde(default)]
    audio_encoder_config: Option<Value>,
    eos_token_id: u32,
    pad_token_id: u32,
    audio_pad_token_id: u32,
    semantic_start_token_id: u32,
    semantic_end_token_id: u32,
    text_config: RawText,
    audio_decoder_config: RawTower,
    quantization_config: RawQuantization,
}

#[derive(Debug, Deserialize)]
struct RawText {
    model_type: String,
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    layer_types: Vec<String>,
    mlp_layer_types: Vec<String>,
    hidden_act: String,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    attention_bias: bool,
    attn_output_gate: bool,
    use_qk_norm: bool,
    rotary_interleaved: bool,
    layernorm_zero_centered_gamma: bool,
    rope_parameters: RawRope,
    rms_norm_eps: f64,
    max_position_embeddings: usize,
    linear_conv_kernel_dim: usize,
    linear_key_head_dim: usize,
    linear_value_head_dim: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    mamba_ssm_dtype: String,
    num_experts: usize,
    num_experts_per_tok: usize,
    moe_intermediate_size: usize,
    shared_expert_intermediate_size: usize,
    shared_expert_gate: bool,
    /// Upstream defaults to true when absent.
    #[serde(default)]
    norm_topk_prob: Option<bool>,
    tie_word_embeddings: bool,
    /// There is no BOS: null.
    #[serde(default)]
    bos_token_id: Option<u32>,
    eos_token_id: u32,
    pad_token_id: u32,
}

#[derive(Debug, Deserialize)]
struct RawRope {
    rope_type: String,
    rope_theta: f64,
    partial_rotary_factor: f64,
    #[serde(default)]
    mrope_section: Option<Value>,
    #[serde(default)]
    mrope_interleaved: Option<Value>,
}

/// Not read: `attention_qk_norm` and `tie_word_embeddings` (see the module
/// doc), the MoE fields (`use_moe` is false), `audio_hidden_dim` and the
/// training-only fields.
#[derive(Debug, Deserialize)]
struct RawTower {
    model_type: String,
    decoder_type: String,
    dim: usize,
    text_dim: usize,
    n_layer: usize,
    shared_enrichment_layers: usize,
    n_head: usize,
    n_local_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    norm_eps: f64,
    rope_base: f64,
    rotary_percent: f64,
    rotary_interleaved: bool,
    attention_output_gate: bool,
    attention_qkv_bias: bool,
    attention_o_bias: bool,
    layernorm_zero_centered_gamma: bool,
    use_moe: bool,
    mtp_use_swa: bool,
    mtp_spec: RawMtpSpec,
    #[serde(default)]
    window_size: Option<Value>,
    max_seq_len: usize,
    num_codebooks: usize,
    vocab_size: usize,
    vq_backbone_codebooks: usize,
    vq_feature_placement: String,
    vq_feature_scope: String,
}

#[derive(Debug, Deserialize)]
struct RawMtpSpec {
    attention_variant: String,
}

#[derive(Debug, Deserialize)]
struct RawQuantization {
    quant_method: String,
    weight_block_size: Vec<usize>,
    activation_scheme: String,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::ModelLoadError(format!("Fish Audio Drama 3 config: {}", message.into()))
}

/// A switch the family implements one way (both policies).
fn require<T: PartialEq + std::fmt::Debug>(field: &str, found: T, expected: T) -> Result<()> {
    if found == expected {
        Ok(())
    } else {
        Err(invalid(format!(
            "{field} is {found:?}; this family implements {expected:?}"
        )))
    }
}

/// A structural relation (both policies).
fn ensure(holds: bool, message: impl FnOnce() -> String) -> Result<()> {
    if holds {
        Ok(())
    } else {
        Err(invalid(message()))
    }
}

/// A published value (the `Published` policy only).
fn pin<T: PartialEq + std::fmt::Debug>(
    policy: FishDrama3ConfigPolicy,
    field: &str,
    found: T,
    expected: T,
) -> Result<()> {
    if policy == FishDrama3ConfigPolicy::Synthetic || found == expected {
        Ok(())
    } else {
        Err(invalid(format!(
            "{field} is {found:?}; the published checkpoint has {expected:?}"
        )))
    }
}

/// `head_dim * fraction`, required whole, even and within the head.
fn rotary_dims(field: &str, head_dim: usize, fraction: f64) -> Result<usize> {
    let exact = head_dim as f64 * fraction;
    let dims = exact.round() as usize;
    ensure(
        fraction.is_finite()
            && (exact - dims as f64).abs() < 1e-9
            && dims > 0
            && dims.is_multiple_of(2)
            && dims <= head_dim,
        || format!("{field} = {fraction} does not rotate a whole, even part of {head_dim} dims"),
    )?;
    Ok(dims)
}

fn positive_finite(field: &str, value: f64) -> Result<()> {
    ensure(value.is_finite() && value > 0.0, || {
        format!("{field} = {value} is not a positive finite number")
    })
}

fn attention_heads(field: &str, heads: usize, kv_heads: usize, head_dim: usize) -> Result<()> {
    ensure(
        heads > 0 && kv_heads > 0 && head_dim > 0 && heads.is_multiple_of(kv_heads),
        || {
            format!(
                "{field}: {heads} query heads do not group over {kv_heads} KV heads of {head_dim}"
            )
        },
    )
}

impl FishDrama3Config {
    /// Reads `config.json` from `model_dir` under the published policy.
    pub fn load(model_dir: &Path) -> Result<Self> {
        Self::load_with_policy(model_dir, FishDrama3ConfigPolicy::Published)
    }

    pub fn load_with_policy(model_dir: &Path, policy: FishDrama3ConfigPolicy) -> Result<Self> {
        let path = model_dir.join(CONFIG_FILE);
        let text = std::fs::read_to_string(&path)
            .map_err(|err| invalid(format!("cannot read {}: {err}", path.display())))?;
        Self::from_json(&text, policy)
    }

    pub fn from_json(text: &str, policy: FishDrama3ConfigPolicy) -> Result<Self> {
        let raw: RawConfig =
            serde_json::from_str(text).map_err(|err| invalid(format!("{CONFIG_FILE}: {err}")))?;
        Self::from_raw(raw, policy)
    }

    fn from_raw(raw: RawConfig, policy: FishDrama3ConfigPolicy) -> Result<Self> {
        require(
            "architectures",
            raw.architectures.as_slice(),
            &[ARCHITECTURE.to_string()][..],
        )?;
        require("model_type", raw.model_type.as_str(), MODEL_TYPE)?;
        require(
            "fish_schema_version",
            raw.fish_schema_version,
            FISH_SCHEMA_VERSION,
        )?;
        require(
            "speech_template_version",
            raw.speech_template_version.as_str(),
            SPEECH_TEMPLATE_VERSION,
        )?;
        require("dtype", raw.dtype.as_str(), "bfloat16")?;
        require(
            "audio_encoder_config",
            raw.audio_encoder_config.as_ref(),
            None,
        )?;

        let q = &raw.quantization_config;
        require(
            "quantization_config.quant_method",
            q.quant_method.as_str(),
            QUANT_METHOD,
        )?;
        require(
            "quantization_config.weight_block_size",
            q.weight_block_size.as_slice(),
            &[FP8_BLOCK, FP8_BLOCK][..],
        )?;
        // W8A16 everywhere: the dynamic activation scales are never used,
        // but another scheme would mean another checkpoint contract.
        require(
            "quantization_config.activation_scheme",
            q.activation_scheme.as_str(),
            "dynamic",
        )?;

        let trunk = trunk_config(&raw.text_config, policy)?;
        let tower = tower_config(&raw.audio_decoder_config, &trunk, policy)?;
        let ids = token_ids(&raw, &trunk, &tower, policy)?;
        Ok(Self {
            policy,
            trunk,
            tower,
            fp8_block: FP8_BLOCK,
            ids,
        })
    }

    /// An FP8 weight's scale grid, `[ceil(rows / block), ceil(cols / block)]`.
    pub fn fp8_scale_shape(&self, rows: usize, cols: usize) -> [usize; 2] {
        [rows.div_ceil(self.fp8_block), cols.div_ceil(self.fp8_block)]
    }
}

fn trunk_config(raw: &RawText, policy: FishDrama3ConfigPolicy) -> Result<FishDrama3TrunkConfig> {
    let t = "text_config";
    let rope = &raw.rope_parameters;
    for (field, found, expected) in [
        ("model_type", raw.model_type.as_str(), TEXT_MODEL_TYPE),
        ("hidden_act", raw.hidden_act.as_str(), "silu"),
        ("mamba_ssm_dtype", raw.mamba_ssm_dtype.as_str(), "float32"),
        (
            "rope_parameters.rope_type",
            rope.rope_type.as_str(),
            "default",
        ),
    ] {
        require(&format!("{t}.{field}"), found, expected)?;
    }
    for (field, found, expected) in [
        ("attention_bias", raw.attention_bias, false),
        ("attn_output_gate", raw.attn_output_gate, true),
        ("use_qk_norm", raw.use_qk_norm, true),
        // NeoX half-split RoPE.
        ("rotary_interleaved", raw.rotary_interleaved, false),
        (
            "layernorm_zero_centered_gamma",
            raw.layernorm_zero_centered_gamma,
            true,
        ),
        ("shared_expert_gate", raw.shared_expert_gate, true),
        ("norm_topk_prob", raw.norm_topk_prob.unwrap_or(true), true),
        ("tie_word_embeddings", raw.tie_word_embeddings, false),
    ] {
        require(&format!("{t}.{field}"), found, expected)?;
    }
    require(&format!("{t}.bos_token_id"), raw.bos_token_id, None)?;
    // Plain 1-D positions: an M-RoPE section would mean another position
    // contract.
    for (field, found) in [
        ("mrope_section", rope.mrope_section.as_ref()),
        ("mrope_interleaved", rope.mrope_interleaved.as_ref()),
    ] {
        require(&format!("{t}.rope_parameters.{field}"), found, None)?;
    }

    ensure(
        raw.layer_types.len() == raw.num_hidden_layers && raw.num_hidden_layers > 0,
        || {
            format!(
                "text_config.layer_types has {} entries for {} layers",
                raw.layer_types.len(),
                raw.num_hidden_layers
            )
        },
    )?;
    let layers = raw
        .layer_types
        .iter()
        .map(|kind| match kind.as_str() {
            "linear_attention" => Ok(FishDrama3LayerKind::LinearAttention),
            "full_attention" => Ok(FishDrama3LayerKind::FullAttention),
            other => Err(invalid(format!(
                "text_config.layer_types names {other:?}; this family implements \
                 linear_attention and full_attention"
            ))),
        })
        .collect::<Result<Vec<_>>>()?;
    ensure(
        raw.mlp_layer_types.len() == layers.len()
            && raw.mlp_layer_types.iter().all(|kind| kind == "moe"),
        || "text_config.mlp_layer_types must name moe for every layer".to_string(),
    )?;

    attention_heads(
        "text_config",
        raw.num_attention_heads,
        raw.num_key_value_heads,
        raw.head_dim,
    )?;
    let attention = FishDrama3AttentionConfig {
        heads: raw.num_attention_heads,
        kv_heads: raw.num_key_value_heads,
        head_dim: raw.head_dim,
        rotary_dims: rotary_dims(
            "text_config.rope_parameters.partial_rotary_factor",
            raw.head_dim,
            rope.partial_rotary_factor,
        )?,
        rope_theta: rope.rope_theta,
        rope_layout: FishDrama3RopeLayout::HalfSplit,
    };
    positive_finite("text_config.rope_parameters.rope_theta", rope.rope_theta)?;
    positive_finite("text_config.rms_norm_eps", raw.rms_norm_eps)?;

    let linear = FishDrama3LinearAttentionConfig {
        key_heads: raw.linear_num_key_heads,
        key_head_dim: raw.linear_key_head_dim,
        value_heads: raw.linear_num_value_heads,
        value_head_dim: raw.linear_value_head_dim,
        conv_kernel: raw.linear_conv_kernel_dim,
    };
    ensure(
        linear.key_heads > 0
            && linear.key_head_dim > 0
            && linear.value_head_dim > 0
            && linear.value_heads.is_multiple_of(linear.key_heads)
            && linear.value_heads > 0
            && linear.conv_kernel >= 2,
        || format!("text_config has an unsupported Gated DeltaNet geometry {linear:?}"),
    )?;

    let moe = FishDrama3MoeConfig {
        experts: raw.num_experts,
        top_k: raw.num_experts_per_tok,
        intermediate: raw.moe_intermediate_size,
        shared_intermediate: raw.shared_expert_intermediate_size,
    };
    ensure(
        moe.experts > 0
            && moe.top_k > 0
            && moe.top_k <= moe.experts
            && moe.intermediate > 0
            && moe.shared_intermediate > 0,
        || format!("text_config has an invalid MoE geometry {moe:?}"),
    )?;
    ensure(
        raw.hidden_size > 0 && raw.hidden_size.is_multiple_of(FP8_BLOCK),
        || {
            format!(
                "text_config.hidden_size {} is not a positive multiple of the FP8 block {FP8_BLOCK}",
                raw.hidden_size
            )
        },
    )?;
    ensure(raw.max_position_embeddings > 0, || {
        "text_config.max_position_embeddings is 0".to_string()
    })?;

    let trunk = FishDrama3TrunkConfig {
        hidden: raw.hidden_size,
        vocab: raw.vocab_size,
        layers,
        attention,
        linear,
        moe,
        rms_norm_eps: raw.rms_norm_eps,
        max_positions: raw.max_position_embeddings,
    };
    for kind in [
        FishDrama3LayerKind::LinearAttention,
        FishDrama3LayerKind::FullAttention,
    ] {
        ensure(!trunk.layers_of(kind).is_empty(), || {
            format!("text_config.layer_types has no {kind:?} layer")
        })?;
    }

    use published as p;
    let full = trunk.full_attention_layers();
    pin(
        policy,
        "text_config.num_hidden_layers",
        trunk.layers.len(),
        p::LAYERS,
    )?;
    pin(
        policy,
        "text_config.layer_types (full attention at)",
        full.as_slice(),
        &p::FULL_ATTENTION_LAYERS[..],
    )?;
    for (field, found, expected) in [
        ("text_config.hidden_size", trunk.hidden, p::HIDDEN),
        ("text_config.vocab_size", trunk.vocab, p::VOCAB),
        (
            "text_config.max_position_embeddings",
            trunk.max_positions,
            p::MAX_POSITIONS,
        ),
        ("text_config.num_attention_heads", attention.heads, p::HEADS),
        (
            "text_config.num_key_value_heads",
            attention.kv_heads,
            p::KV_HEADS,
        ),
        ("text_config.head_dim", attention.head_dim, p::HEAD_DIM),
        (
            "text_config rotary dims",
            attention.rotary_dims,
            p::ROTARY_DIMS,
        ),
        (
            "text_config.linear_num_key_heads",
            linear.key_heads,
            p::LINEAR_KEY_HEADS,
        ),
        (
            "text_config.linear_key_head_dim",
            linear.key_head_dim,
            p::LINEAR_KEY_DIM,
        ),
        (
            "text_config.linear_num_value_heads",
            linear.value_heads,
            p::LINEAR_VALUE_HEADS,
        ),
        (
            "text_config.linear_value_head_dim",
            linear.value_head_dim,
            p::LINEAR_VALUE_DIM,
        ),
        (
            "text_config.linear_conv_kernel_dim",
            linear.conv_kernel,
            p::CONV_KERNEL,
        ),
        ("text_config.num_experts", moe.experts, p::EXPERTS),
        ("text_config.num_experts_per_tok", moe.top_k, p::TOP_K),
        (
            "text_config.moe_intermediate_size",
            moe.intermediate,
            p::MOE_INTERMEDIATE,
        ),
        (
            "text_config.shared_expert_intermediate_size",
            moe.shared_intermediate,
            p::SHARED_INTERMEDIATE,
        ),
    ] {
        pin(policy, field, found, expected)?;
    }
    pin(
        policy,
        "text_config.rope_parameters.rope_theta",
        attention.rope_theta,
        p::ROPE_THETA,
    )?;
    pin(
        policy,
        "text_config.rms_norm_eps",
        trunk.rms_norm_eps,
        p::NORM_EPS,
    )?;
    Ok(trunk)
}

fn tower_config(
    raw: &RawTower,
    trunk: &FishDrama3TrunkConfig,
    policy: FishDrama3ConfigPolicy,
) -> Result<FishDrama3TowerConfig> {
    let f = "audio_decoder_config";
    for (field, found, expected) in [
        ("model_type", raw.model_type.as_str(), TOWER_MODEL_TYPE),
        (
            "decoder_type",
            raw.decoder_type.as_str(),
            TOWER_DECODER_TYPE,
        ),
        (
            "mtp_spec.attention_variant",
            raw.mtp_spec.attention_variant.as_str(),
            "full_attention",
        ),
        // The trunk reads codebook 1 of each frame; the tower injects the rest.
        (
            "vq_feature_placement",
            raw.vq_feature_placement.as_str(),
            "backbone_partial",
        ),
        ("vq_feature_scope", raw.vq_feature_scope.as_str(), "all_vq"),
    ] {
        require(&format!("{f}.{field}"), found, expected)?;
    }
    for (field, found, expected) in [
        // Interleaved-pair RoPE.
        ("rotary_interleaved", raw.rotary_interleaved, true),
        ("attention_output_gate", raw.attention_output_gate, true),
        ("attention_qkv_bias", raw.attention_qkv_bias, false),
        ("attention_o_bias", raw.attention_o_bias, false),
        (
            "layernorm_zero_centered_gamma",
            raw.layernorm_zero_centered_gamma,
            true,
        ),
        ("use_moe", raw.use_moe, false),
        ("mtp_use_swa", raw.mtp_use_swa, false),
    ] {
        require(&format!("{f}.{field}"), found, expected)?;
    }
    require(&format!("{f}.window_size"), raw.window_size.as_ref(), None)?;
    require(
        &format!("{f}.vq_backbone_codebooks"),
        raw.vq_backbone_codebooks,
        1,
    )?;

    ensure(raw.dim == trunk.hidden && raw.text_dim == raw.dim, || {
        format!(
            "{f}.dim {} / text_dim {} must equal the trunk's hidden size {}",
            raw.dim, raw.text_dim, trunk.hidden
        )
    })?;
    ensure(
        raw.n_layer == raw.shared_enrichment_layers && raw.n_layer >= 1,
        || {
            format!(
                "{f}.n_layer {} must equal shared_enrichment_layers {} (at least one)",
                raw.n_layer, raw.shared_enrichment_layers
            )
        },
    )?;
    ensure(raw.num_codebooks >= 2, || {
        format!(
            "{f}.num_codebooks {} leaves no residual codebook",
            raw.num_codebooks
        )
    })?;
    // `max_seq_len` is the frame width, not an attention window.
    ensure(raw.max_seq_len == raw.num_codebooks + 1, || {
        format!(
            "{f}.max_seq_len {} is not the frame width num_codebooks + 1 = {}",
            raw.max_seq_len,
            raw.num_codebooks + 1
        )
    })?;
    ensure(raw.intermediate_size > 0 && raw.vocab_size > 0, || {
        format!("{f} has an empty feed-forward or codebook")
    })?;
    attention_heads(f, raw.n_head, raw.n_local_heads, raw.head_dim)?;
    positive_finite(&format!("{f}.rope_base"), raw.rope_base)?;
    positive_finite(&format!("{f}.norm_eps"), raw.norm_eps)?;
    let attention = FishDrama3AttentionConfig {
        heads: raw.n_head,
        kv_heads: raw.n_local_heads,
        head_dim: raw.head_dim,
        rotary_dims: rotary_dims(
            &format!("{f}.rotary_percent"),
            raw.head_dim,
            raw.rotary_percent,
        )?,
        rope_theta: raw.rope_base,
        rope_layout: FishDrama3RopeLayout::InterleavedPairs,
    };
    let tower = FishDrama3TowerConfig {
        dim: raw.dim,
        codebooks: raw.num_codebooks,
        codebook_vocab: raw.vocab_size,
        shared_layers: raw.shared_enrichment_layers,
        attention,
        intermediate: raw.intermediate_size,
        norm_eps: raw.norm_eps,
    };

    use published as p;
    for (field, found, expected) in [
        ("num_codebooks", tower.codebooks, p::CODEBOOKS),
        ("vocab_size", tower.codebook_vocab, p::CODEBOOK_VOCAB),
        (
            "shared_enrichment_layers",
            tower.shared_layers,
            p::TOWER_SHARED_LAYERS,
        ),
        ("n_head", attention.heads, p::HEADS),
        ("n_local_heads", attention.kv_heads, p::KV_HEADS),
        ("head_dim", attention.head_dim, p::HEAD_DIM),
        ("rotary dims", attention.rotary_dims, p::ROTARY_DIMS),
        (
            "intermediate_size",
            tower.intermediate,
            p::TOWER_INTERMEDIATE,
        ),
    ] {
        pin(policy, &format!("{f}.{field}"), found, expected)?;
    }
    pin(
        policy,
        &format!("{f}.rope_base"),
        attention.rope_theta,
        p::ROPE_THETA,
    )?;
    pin(
        policy,
        &format!("{f}.norm_eps"),
        tower.norm_eps,
        p::NORM_EPS,
    )?;
    Ok(tower)
}

fn token_ids(
    raw: &RawConfig,
    trunk: &FishDrama3TrunkConfig,
    tower: &FishDrama3TowerConfig,
    policy: FishDrama3ConfigPolicy,
) -> Result<FishDrama3TokenIds> {
    require(
        "text_config.eos_token_id",
        raw.text_config.eos_token_id,
        raw.eos_token_id,
    )?;
    require(
        "text_config.pad_token_id",
        raw.text_config.pad_token_id,
        raw.pad_token_id,
    )?;
    let (start, end) = (raw.semantic_start_token_id, raw.semantic_end_token_id);
    ensure(start <= end && (end as usize) < trunk.vocab, || {
        format!(
            "the semantic band {start}..={end} does not lie inside the {}-token vocabulary",
            trunk.vocab
        )
    })?;
    ensure((end - start) as usize + 1 == tower.codebook_vocab, || {
        format!(
            "the semantic band holds {} tokens; the tower predicts {} codes",
            end - start + 1,
            tower.codebook_vocab
        )
    })?;
    // The speaker tags and `<|voice|>` sit below the band.
    let speaker_first = start.checked_sub(SPEAKER_TAGS as u32);
    let voice = start.checked_sub(SPEAKER_TAGS as u32 + 2);
    let (Some(speaker_first), Some(voice)) = (speaker_first, voice) else {
        return Err(invalid(format!(
            "semantic_start_token_id {start} leaves no room for the speaker tags"
        )));
    };
    let ids = FishDrama3TokenIds {
        pad: raw.pad_token_id,
        eos: raw.eos_token_id,
        audio_pad: raw.audio_pad_token_id,
        voice,
        speaker_first,
        semantic_start: start,
        semantic_end: end,
    };
    // Eos below the band keeps the compact head's candidate order ascending.
    ensure(
        ids.eos < voice && ids.pad < voice && ids.eos != ids.pad,
        || {
            format!(
                "eos {} and pad {} must be distinct ids below <|voice|> {voice}",
                ids.eos, ids.pad
            )
        },
    )?;
    ensure(
        ids.audio_pad < voice && ids.audio_pad != ids.eos && ids.audio_pad != ids.pad,
        || {
            format!(
                "audio_pad_token_id {} collides with another id",
                ids.audio_pad
            )
        },
    )?;

    use published as p;
    for (field, found, expected) in [
        ("pad_token_id", ids.pad, p::PAD),
        ("eos_token_id", ids.eos, p::EOS),
        ("audio_pad_token_id", ids.audio_pad, p::AUDIO_PAD),
        (
            "semantic_start_token_id",
            ids.semantic_start,
            p::SEMANTIC_START,
        ),
        ("semantic_end_token_id", ids.semantic_end, p::SEMANTIC_END),
        ("<|voice|>", ids.voice, p::VOICE),
        ("<|speaker:0|>", ids.speaker_first, p::SPEAKER_FIRST),
    ] {
        pin(policy, field, found, expected)?;
    }
    Ok(ids)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) const PUBLISHED_CONFIG: &str =
        include_str!("../../../../tests/fixtures/fish_drama3/config.json");

    pub(crate) fn published() -> FishDrama3Config {
        FishDrama3Config::from_json(PUBLISHED_CONFIG, FishDrama3ConfigPolicy::Published).unwrap()
    }

    /// A small config with every switch as published: hidden 256 (two FP8
    /// blocks), 4 trunk layers (3 linear, 1 full), 4 experts, a 64-token
    /// vocabulary with an 8-code semantic band at 40..=47, and a tower of
    /// 3 codebooks (1 shared layer, 2 residual layers).
    pub(crate) fn synthetic_json() -> Value {
        let mut c: Value = serde_json::from_str(PUBLISHED_CONFIG).unwrap();
        let t = &mut c["text_config"];
        t["hidden_size"] = 256.into();
        t["num_hidden_layers"] = 4.into();
        t["layer_types"] = serde_json::json!([
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention"
        ]);
        t["mlp_layer_types"] = serde_json::json!(["moe", "moe", "moe", "moe"]);
        t["num_attention_heads"] = 2.into();
        t["num_key_value_heads"] = 1.into();
        t["head_dim"] = 32.into();
        t["linear_num_key_heads"] = 2.into();
        t["linear_key_head_dim"] = 16.into();
        t["linear_num_value_heads"] = 4.into();
        t["linear_value_head_dim"] = 16.into();
        t["num_experts"] = 4.into();
        t["num_experts_per_tok"] = 2.into();
        t["moe_intermediate_size"] = 64.into();
        t["shared_expert_intermediate_size"] = 32.into();
        t["vocab_size"] = 64.into();
        t["max_position_embeddings"] = 512.into();
        t["eos_token_id"] = 4.into();
        t["pad_token_id"] = 2.into();
        let a = &mut c["audio_decoder_config"];
        a["dim"] = 256.into();
        a["text_dim"] = 256.into();
        a["n_layer"] = 1.into();
        a["shared_enrichment_layers"] = 1.into();
        a["n_head"] = 2.into();
        a["n_local_heads"] = 1.into();
        a["head_dim"] = 32.into();
        a["intermediate_size"] = 96.into();
        a["num_codebooks"] = 3.into();
        a["max_seq_len"] = 4.into();
        a["vocab_size"] = 8.into();
        c["eos_token_id"] = 4.into();
        c["pad_token_id"] = 2.into();
        c["audio_pad_token_id"] = 3.into();
        c["semantic_start_token_id"] = 40.into();
        c["semantic_end_token_id"] = 47.into();
        c
    }

    pub(crate) fn synthetic() -> FishDrama3Config {
        FishDrama3Config::from_json(
            &synthetic_json().to_string(),
            FishDrama3ConfigPolicy::Synthetic,
        )
        .unwrap()
    }

    #[test]
    fn the_published_config_parses_to_the_published_geometry() {
        let c = published();
        let t = &c.trunk;
        assert_eq!((t.hidden, t.vocab, t.layers.len()), (2048, 252_288, 40));
        assert_eq!(
            t.full_attention_layers(),
            vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
        );
        assert_eq!(t.linear_attention_layers().len(), 30);
        assert_eq!(
            t.attention,
            FishDrama3AttentionConfig {
                heads: 16,
                kv_heads: 2,
                head_dim: 256,
                rotary_dims: 64,
                rope_theta: 1e7,
                rope_layout: FishDrama3RopeLayout::HalfSplit,
            }
        );
        assert_eq!(
            (
                t.linear.key_width(),
                t.linear.value_width(),
                t.linear.conv_channels()
            ),
            (2048, 4096, 8192)
        );
        assert_eq!(t.linear.conv_kernel, 4);
        assert_eq!(
            t.moe,
            FishDrama3MoeConfig {
                experts: 256,
                top_k: 8,
                intermediate: 512,
                shared_intermediate: 512,
            }
        );
        assert_eq!(t.rms_norm_eps, 1e-6);
        assert_eq!(t.max_positions, 262_144);

        let w = &c.tower;
        assert_eq!(
            (w.dim, w.codebooks, w.codebook_vocab, w.shared_layers),
            (2048, 10, 4096, 4)
        );
        assert_eq!(
            (w.residual_layers(), w.frame_width(), w.wqkv_rows()),
            (9, 11, 9216)
        );
        assert_eq!(
            w.attention.rope_layout,
            FishDrama3RopeLayout::InterleavedPairs
        );
        assert_eq!(
            (
                w.attention.heads,
                w.attention.kv_heads,
                w.attention.head_dim
            ),
            (16, 2, 256)
        );
        assert_eq!((w.attention.rotary_dims, w.attention.rope_theta), (64, 1e7));
        assert_eq!((w.intermediate, w.norm_eps), (6144, 1e-6));

        assert_eq!(c.fp8_block, 128);
        assert_eq!(c.fp8_scale_shape(512, 2048), [4, 16]);
        assert_eq!(c.fp8_scale_shape(2048, 512), [16, 4]);

        let ids = c.ids;
        assert_eq!(
            (ids.pad, ids.eos, ids.audio_pad),
            (248_044, 248_046, 248_076)
        );
        assert_eq!(ids.voice, 248_081);
        assert_eq!(ids.speakers(), 248_083..=248_098);
        assert_eq!(ids.speaker(15), Some(248_098));
        assert_eq!(ids.speaker(16), None);
        assert_eq!((ids.semantic_start, ids.semantic_end), (248_099, 252_194));
        assert_eq!(ids.semantic_codes(), 4096);
        assert_eq!(ids.semantic(0), Some(248_099));
        assert_eq!(ids.semantic(4095), Some(252_194));
        assert_eq!(ids.semantic(4096), None);
        assert_eq!(ids.semantic_code(252_194), Some(4095));
        assert_eq!(ids.semantic_code(248_046), None);
        let rows = ids.compact_head_rows();
        assert_eq!(rows.len(), 4097);
        assert_eq!((rows[0], rows[1], rows[4096]), (248_046, 248_099, 252_194));
        assert!(rows.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn the_synthetic_config_keeps_the_structure_and_fails_the_published_pins() {
        let c = synthetic();
        assert_eq!(c.trunk.full_attention_layers(), vec![3]);
        assert_eq!(c.trunk.attention.rotary_dims, 8);
        assert_eq!(c.tower.residual_layers(), 2);
        assert_eq!(c.ids.voice, 22);
        assert_eq!(c.ids.speakers(), 24..=39);
        assert_eq!(
            c.ids.compact_head_rows(),
            vec![4, 40, 41, 42, 43, 44, 45, 46, 47]
        );
        let err = FishDrama3Config::from_json(
            &synthetic_json().to_string(),
            FishDrama3ConfigPolicy::Published,
        )
        .unwrap_err();
        assert!(err.to_string().contains("published checkpoint"), "{err}");
        // The published config is also a valid synthetic one.
        assert!(
            FishDrama3Config::from_json(PUBLISHED_CONFIG, FishDrama3ConfigPolicy::Synthetic)
                .is_ok()
        );
    }

    fn mutated(
        policy: FishDrama3ConfigPolicy,
        edit: impl Fn(&mut Value),
    ) -> Result<FishDrama3Config> {
        let mut config: Value = serde_json::from_str(PUBLISHED_CONFIG).unwrap();
        edit(&mut config);
        FishDrama3Config::from_json(&config.to_string(), policy)
    }

    #[test]
    fn changed_conventions_fail_closed() {
        use serde_json::json;
        use FishDrama3ConfigPolicy::{Published, Synthetic};
        type Edit = fn(&mut Value);
        // Each fails under both policies: switches and structural relations.
        let switches: &[(&str, Edit)] = &[
            ("architectures", |c| {
                c["architectures"] = json!(["Qwen3_5MoeForConditionalGeneration"])
            }),
            ("model_type", |c| c["model_type"] = "fish_qwen3_omni".into()),
            ("fish_schema_version", |c| {
                c["fish_schema_version"] = 2.into()
            }),
            ("speech_template_version", |c| {
                c["speech_template_version"] = "reference-bracket-v2".into()
            }),
            ("audio_encoder_config", |c| {
                c["audio_encoder_config"] = json!({})
            }),
            ("quant_method", |c| {
                c["quantization_config"]["quant_method"] = "fp8".into()
            }),
            ("weight_block_size", |c| {
                c["quantization_config"]["weight_block_size"] = json!([64, 64])
            }),
            ("text rotary_interleaved", |c| {
                c["text_config"]["rotary_interleaved"] = true.into()
            }),
            ("layer kind", |c| {
                c["text_config"]["layer_types"][5] = "sliding_attention".into()
            }),
            ("mlp_layer_types", |c| {
                c["text_config"]["mlp_layer_types"][0] = "dense".into()
            }),
            ("mrope_section", |c| {
                c["text_config"]["rope_parameters"]["mrope_section"] = json!([11, 11, 10])
            }),
            ("tie_word_embeddings", |c| {
                c["text_config"]["tie_word_embeddings"] = true.into()
            }),
            ("mamba_ssm_dtype", |c| {
                c["text_config"]["mamba_ssm_dtype"] = "bfloat16".into()
            }),
            ("use_qk_norm missing", |c| {
                c["text_config"]
                    .as_object_mut()
                    .unwrap()
                    .remove("use_qk_norm");
            }),
            ("hidden not a block multiple", |c| {
                c["text_config"]["hidden_size"] = 2000.into();
                c["audio_decoder_config"]["dim"] = 2000.into();
                c["audio_decoder_config"]["text_dim"] = 2000.into();
            }),
            ("bos_token_id", |c| {
                c["text_config"]["bos_token_id"] = 248_045.into()
            }),
            ("tower max_seq_len as a window", |c| {
                c["audio_decoder_config"]["max_seq_len"] = 4096.into()
            }),
            ("tower rotary_interleaved", |c| {
                c["audio_decoder_config"]["rotary_interleaved"] = false.into()
            }),
            ("tower n_layer", |c| {
                c["audio_decoder_config"]["n_layer"] = 5.into()
            }),
            ("vq_feature_placement", |c| {
                c["audio_decoder_config"]["vq_feature_placement"] = "mtp_shared_pre".into()
            }),
            ("vq_backbone_codebooks", |c| {
                c["audio_decoder_config"]["vq_backbone_codebooks"] = 2.into()
            }),
            ("tower use_moe", |c| {
                c["audio_decoder_config"]["use_moe"] = true.into()
            }),
            ("tower dim", |c| {
                c["audio_decoder_config"]["dim"] = 1024.into()
            }),
            ("semantic band vs tower vocab", |c| {
                c["semantic_end_token_id"] = 252_195.into()
            }),
            ("eos inside the band", |c| {
                c["eos_token_id"] = 248_100.into();
                c["text_config"]["eos_token_id"] = 248_100.into();
            }),
            ("text eos disagrees", |c| {
                c["text_config"]["eos_token_id"] = 248_045.into()
            }),
        ];
        assert!(switches.len() >= 10);
        for &(name, edit) in switches {
            for policy in [Published, Synthetic] {
                assert!(mutated(policy, edit).is_err(), "{name} under {policy:?}");
            }
        }
        // Published values: only the published policy pins them.
        let pins: &[(&str, Edit)] = &[
            ("full attention moved", |c| {
                c["text_config"]["layer_types"][3] = "linear_attention".into();
                c["text_config"]["layer_types"][2] = "full_attention".into();
            }),
            ("num_experts", |c| {
                c["text_config"]["num_experts"] = 128.into()
            }),
            ("vocab_size", |c| {
                c["text_config"]["vocab_size"] = 252_416.into()
            }),
            ("rope_theta", |c| {
                c["text_config"]["rope_parameters"]["rope_theta"] = 1e6.into()
            }),
            ("rms_norm_eps", |c| {
                c["text_config"]["rms_norm_eps"] = 1e-5.into()
            }),
            ("tower intermediate", |c| {
                c["audio_decoder_config"]["intermediate_size"] = 4096.into()
            }),
            // S2's `<|pad|>`.
            ("pad_token_id", |c| {
                c["pad_token_id"] = 248_077.into();
                c["text_config"]["pad_token_id"] = 248_077.into();
            }),
            ("tower rotary percent", |c| {
                c["audio_decoder_config"]["rotary_percent"] = 0.5.into()
            }),
        ];
        for &(name, edit) in pins {
            let err = mutated(Published, edit).unwrap_err();
            assert!(
                err.to_string().contains("published checkpoint"),
                "{name}: {err}"
            );
            assert!(mutated(Synthetic, edit).is_ok(), "{name} under Synthetic");
        }
        // The two config traps load either way.
        assert!(mutated(Published, |c| {
            c["audio_decoder_config"]["attention_qk_norm"] = true.into()
        })
        .is_ok());
        assert!(mutated(Published, |c| {
            c["audio_decoder_config"]
                .as_object_mut()
                .unwrap()
                .remove("attention_qk_norm");
        })
        .is_ok());
        assert!(mutated(Published, |_| {}).is_ok());
    }
}
