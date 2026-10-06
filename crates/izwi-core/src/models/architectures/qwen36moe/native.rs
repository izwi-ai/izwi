//! Native Hugging Face checkpoint ingestion for the qwen3_5_moe architecture
//! (published checkpoint: Qwen3.6-35B-A3B-FP8).
//!
//! The published checkpoint is an indexed Safetensors bundle whose matrix
//! weights use 128x128 block-scaled `F8_E4M3` (companion `weight_scale_inv`
//! tensors) while embeddings, norms, router/gate weights, DeltaNet in_proj /
//! conv tensors, and the MTP head stay dense. This module owns the
//! Qwen3.5-MoE configuration contract, the tensor-name plan, and checkpoint
//! validation; shard reading, block-FP8 decoding, and projection
//! materialization reuse the narrow primitives exposed by
//! `qwen38::native::IndexedSafetensors`.
//!
//! Tensor naming follows the upstream composite layout: language tensors live
//! under `model.` (plain) or `model.language_model.` (composite), the vision
//! tower under `model.visual.`, and the multi-token-prediction head under
//! `mtp.`. Both language layouts are accepted and canonicalized; vision and
//! MTP tensors are accounted but not loaded (text-only scope; the MTP
//! manifest is recorded for the later speculative-decoding phase).
//!
//! Per-layer tensor names and shapes are derived from the Qwen3-Next lineage
//! naming. The first open of a real downloaded checkpoint is expected to run
//! with fail-closed validation: any naming drift surfaces as a loud error
//! that names the unexpected or missing tensor instead of a silent skip.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use safetensors::Dtype as SafeDType;

use crate::backends::{BackendKind, DeviceProfile};
use crate::error::{Error, Result};
use crate::models::architectures::qwen38::native::{
    BlockFp8Config, IndexedSafetensors, ProjectionMaterialization, RawBlockFp8Projection,
};

/// Opt a process into accepting synthetic (non-published) Qwen3.5-MoE
/// checkpoint geometry. Production loads fail closed on anything but the
/// published 35B-A3B shape; benchmark and CI fixtures opt in explicitly with
/// this variable. Structural invariants (layer-type pattern, mrope coverage,
/// SSM width products, FP8 quantization routing) stay enforced under both
/// policies.
pub const ENV_ALLOW_SYNTHETIC_QWEN36_MOE_GEOMETRY: &str =
    "IZWI_ALLOW_SYNTHETIC_QWEN36_MOE_GEOMETRY";

const CONFIG_FILE: &str = "config.json";

/// Published Qwen3.6-35B-A3B-FP8 geometry (identical to the original
/// Qwen3.5-35B-A3B-FP8 checkpoint). Synthetic fixtures relax every
/// value; structural invariants stay enforced.
const PINNED_BLOCK_COUNT: usize = 40;
const PINNED_FULL_ATTENTION_INTERVAL: usize = 4;
const PINNED_HIDDEN_SIZE: usize = 2_048;
const PINNED_VOCAB_SIZE: usize = 248_320;
const PINNED_CONTEXT_TOKENS: usize = 262_144;
const PINNED_ATTENTION_HEADS: usize = 16;
const PINNED_ATTENTION_KV_HEADS: usize = 2;
const PINNED_HEAD_DIM: usize = 256;
const PINNED_ROPE_DIMENSION_COUNT: usize = 64;
const PINNED_ROPE_THETA: f64 = 10_000_000.0;
const PINNED_MROPE_SECTIONS: [usize; 3] = [11, 11, 10];
const PINNED_PARTIAL_ROTARY_FACTOR: f64 = 0.25;
const PINNED_SSM_TIME_STEP_RANK: usize = 32;
const PINNED_SSM_GROUP_COUNT: usize = 16;
const PINNED_SSM_STATE_SIZE: usize = 128;
const PINNED_SSM_VALUE_HEAD_DIM: usize = 128;
const PINNED_MOE_EXPERTS: usize = 256;
const PINNED_MOE_EXPERTS_PER_TOK: usize = 8;
const PINNED_MOE_INTERMEDIATE: usize = 512;
const PINNED_SHARED_EXPERT_INTERMEDIATE: usize = 512;
const PINNED_RMS_NORM_EPS: f64 = 1e-6;
const PINNED_ARCHITECTURE: &str = "Qwen3_5MoeForConditionalGeneration";

/// JSON body of the pinned 35B `config.json`, built from the constants above.
/// Shared by geometry validation and the admission memory plan so both see
/// one source of truth for the published geometry.
pub(crate) fn pinned_config_json() -> serde_json::Value {
    let mut layer_types = Vec::with_capacity(PINNED_BLOCK_COUNT);
    for layer in 0..PINNED_BLOCK_COUNT {
        if (layer + 1) % PINNED_FULL_ATTENTION_INTERVAL == 0 {
            layer_types.push(serde_json::json!("full_attention"));
        } else {
            layer_types.push(serde_json::json!("linear_attention"));
        }
    }
    serde_json::json!({
        "architectures": [PINNED_ARCHITECTURE],
        "model_type": "qwen3_5_moe",
        "text_config": {
            "num_hidden_layers": PINNED_BLOCK_COUNT,
            "full_attention_interval": PINNED_FULL_ATTENTION_INTERVAL,
            "hidden_size": PINNED_HIDDEN_SIZE,
            "vocab_size": PINNED_VOCAB_SIZE,
            "max_position_embeddings": PINNED_CONTEXT_TOKENS,
            "num_attention_heads": PINNED_ATTENTION_HEADS,
            "num_key_value_heads": PINNED_ATTENTION_KV_HEADS,
            "head_dim": PINNED_HEAD_DIM,
            "num_experts": PINNED_MOE_EXPERTS,
            "num_experts_per_tok": PINNED_MOE_EXPERTS_PER_TOK,
            "moe_intermediate_size": PINNED_MOE_INTERMEDIATE,
            "shared_expert_intermediate_size": PINNED_SHARED_EXPERT_INTERMEDIATE,
            "linear_num_value_heads": PINNED_SSM_TIME_STEP_RANK,
            "linear_num_key_heads": PINNED_SSM_GROUP_COUNT,
            "linear_key_head_dim": PINNED_SSM_STATE_SIZE,
            "linear_value_head_dim": PINNED_SSM_VALUE_HEAD_DIM,
            "linear_conv_kernel_dim": 4,
            "rms_norm_eps": PINNED_RMS_NORM_EPS,
            "mamba_ssm_dtype": "float32",
            "layer_types": layer_types,
            "rope_parameters": {
                "rope_type": "mrope",
                "mrope_interleaved": true,
                "mrope_section": PINNED_MROPE_SECTIONS,
                "rope_theta": PINNED_ROPE_THETA,
                "partial_rotary_factor": PINNED_PARTIAL_ROTARY_FACTOR
            }
        },
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["lm_head", "model.embed_tokens"]
        }
    })
}

/// The pinned 35B config, constructible without a checkpoint on disk.
pub(crate) fn pinned_native_config() -> Qwen36MoeNativeConfig {
    Qwen36MoeNativeConfig::from_json_with_policy(
        &serde_json::to_vec(&pinned_config_json()).expect("pinned config serializes"),
        Qwen36MoeGeometryPolicy::Pinned35B,
    )
    .expect("pinned config constants pass validation")
}

/// Element inventory of the published checkpoint's persistent representation,
/// derived from the same tensor plan the loader validates against: block-FP8
/// projection elements, dense elements, and the total tensor count (weights
/// plus scale companions) that drives instantiation slack at MoE scale.
pub(crate) struct PinnedRepresentationInventory {
    pub fp8_elements: u64,
    pub dense_elements: u64,
    pub tensor_count: u64,
    /// Host bytes of the F32-decoded `weight_scale_inv` companions (each
    /// block-FP8 tensor of `[rows, cols]` carries
    /// `[ceil(rows/128), ceil(cols/128)]` F32 scales).
    pub fp8_scale_bytes: u64,
    /// Elements of block-FP8 tensors whose geometry the CUDA fp8 projection
    /// kernel cannot execute (`n % 64 != 0 || k % 128 != 0`); under the
    /// native-FP8 residency these fall back to packed Q8_0.
    pub fp8_incompatible_elements: u64,
}

pub(crate) fn pinned_representation_inventory() -> PinnedRepresentationInventory {
    use ExpectedTensorKind::{BlockFp8, BlockFp8Scale, Dense, OptionalDense};
    let plan = expected_text_tensor_plan(&pinned_native_config())
        .expect("pinned config produces the validated tensor plan");
    let mut fp8_elements = 0u64;
    let mut dense_elements = 0u64;
    let mut fp8_scale_bytes = 0u64;
    let mut fp8_incompatible_elements = 0u64;
    for expected in plan.values() {
        let count = expected
            .shape
            .iter()
            .try_fold(1u64, |acc, &dim| {
                acc.checked_mul(u64::try_from(dim).unwrap_or(u64::MAX))
            })
            .unwrap_or(u64::MAX);
        match expected.kind {
            BlockFp8 => {
                fp8_elements = fp8_elements.saturating_add(count);
                let (rows, cols) = (expected.shape[0], expected.shape[1]);
                let scale_entries =
                    rows.div_ceil(128).saturating_mul(cols.div_ceil(128)) as u64;
                fp8_scale_bytes = fp8_scale_bytes.saturating_add(scale_entries.saturating_mul(4));
                if rows % 64 != 0 || cols % 128 != 0 {
                    fp8_incompatible_elements = fp8_incompatible_elements.saturating_add(count);
                }
            }
            Dense | OptionalDense => dense_elements = dense_elements.saturating_add(count),
            // Scale companions are consumed during dequantization and never
            // materialize into the persistent representation; they still count
            // toward the checkpoint's tensor count.
            BlockFp8Scale => {}
        }
    }
    PinnedRepresentationInventory {
        fp8_elements,
        dense_elements,
        tensor_count: plan.len() as u64,
        fp8_scale_bytes,
        fp8_incompatible_elements,
    }
}

/// Geometry-validation policy for a checkpoint open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen36MoeGeometryPolicy {
    /// Every configuration field is pinned to the published 35B-A3B checkpoint.
    Pinned35B,
    /// Benchmark/CI fixture geometry: values are free, structure is checked.
    Synthetic,
}

pub fn synthetic_geometry_enabled() -> bool {
    std::env::var(ENV_ALLOW_SYNTHETIC_QWEN36_MOE_GEOMETRY)
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE"))
}

impl Qwen36MoeGeometryPolicy {
    pub(crate) fn from_env() -> Self {
        if synthetic_geometry_enabled() {
            Self::Synthetic
        } else {
            Self::Pinned35B
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen36MoeLayerType {
    LinearAttention,
    FullAttention,
}

impl Qwen36MoeLayerType {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "linear_attention" => Some(Self::LinearAttention),
            "full_attention" => Some(Self::FullAttention),
            _ => None,
        }
    }
}

/// Validated Qwen3.5-MoE text configuration.
#[derive(Debug, Clone)]
pub struct Qwen36MoeNativeConfig {
    pub text: Qwen36MoeTextConfig,
    pub layer_types: Vec<Qwen36MoeLayerType>,
    pub block_fp8: BlockFp8Config,
}

/// Text-tower geometry the loader and execution module agree on.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen36MoeTextConfig {
    pub block_count: usize,
    pub full_attention_interval: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub context_tokens: usize,
    pub attention_head_count: usize,
    pub attention_head_count_kv: usize,
    pub attention_key_length: usize,
    pub attention_value_length: usize,
    pub rope_theta: f64,
    pub rope_dimension_count: usize,
    pub mrope_sections: [usize; 3],
    pub partial_rotary_factor: f64,
    pub ssm_time_step_rank: usize,
    pub ssm_group_count: usize,
    pub ssm_state_size: usize,
    pub ssm_value_head_dim: usize,
    pub ssm_conv_kernel: usize,
    pub moe_num_experts: usize,
    pub moe_num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub has_shared_expert_gate: bool,
    pub rms_norm_eps: f64,
    pub mamba_ssm_dtype_f32: bool,
    pub tie_word_embeddings: bool,
}

impl Qwen36MoeTextConfig {
    pub fn is_full_attention_layer(&self, layer: usize) -> bool {
        (layer + 1).is_multiple_of(self.full_attention_interval)
    }

    /// Widths of the fused DeltaNet input projections.
    pub fn ssm_qk_width(&self) -> usize {
        self.ssm_group_count * self.ssm_state_size
    }

    pub fn ssm_v_width(&self) -> usize {
        self.ssm_time_step_rank * self.ssm_value_head_dim
    }

    pub fn ssm_conv_channels(&self) -> usize {
        2 * self.ssm_qk_width() + self.ssm_v_width()
    }

    pub fn attention_query_width(&self) -> usize {
        self.attention_head_count * self.attention_key_length
    }

    pub fn attention_kv_width(&self) -> usize {
        self.attention_head_count_kv * self.attention_key_length
    }
}

fn config_error(field: &str, detail: impl std::fmt::Display) -> Error {
    Error::ModelLoadError(format!("config.json `{field}`: {detail}"))
}

fn req_f64(value: &serde_json::Value, field: &str) -> Result<f64> {
    value
        .as_f64()
        .ok_or_else(|| config_error(field, "expected a number"))
}

fn req_usize(value: &serde_json::Value, field: &str) -> Result<usize> {
    let raw = req_f64(value, field)?;
    if raw < 0.0 || raw.fract() != 0.0 {
        return Err(config_error(field, "expected a non-negative integer"));
    }
    usize::try_from(raw as u128).map_err(|_| config_error(field, "value out of range"))
}

fn req_str<'a>(value: &'a serde_json::Value, field: &str) -> Result<&'a str> {
    value
        .as_str()
        .ok_or_else(|| config_error(field, "expected a string"))
}

fn req_usize_array(value: &serde_json::Value, field: &str, len: usize) -> Result<Vec<usize>> {
    let items = value
        .as_array()
        .ok_or_else(|| config_error(field, "expected an array"))?;
    if items.len() != len {
        return Err(config_error(field, format!("expected {len} entries")));
    }
    items.iter().map(|item| req_usize(item, field)).collect()
}

fn require_positive(field: &str, value: usize) -> Result<()> {
    if value == 0 {
        return Err(config_error(field, "must be positive"));
    }
    Ok(())
}

fn require_eq_usize(
    field: &str,
    actual: usize,
    expected: usize,
    policy: Qwen36MoeGeometryPolicy,
) -> Result<()> {
    if policy == Qwen36MoeGeometryPolicy::Pinned35B && actual != expected {
        return Err(config_error(
            field,
            format!("pinned checkpoint expects {expected}, found {actual}"),
        ));
    }
    Ok(())
}

fn require_eq_f64(
    field: &str,
    actual: f64,
    expected: f64,
    policy: Qwen36MoeGeometryPolicy,
) -> Result<()> {
    if policy == Qwen36MoeGeometryPolicy::Pinned35B && (actual - expected).abs() > f64::EPSILON {
        return Err(config_error(
            field,
            format!("pinned checkpoint expects {expected}, found {actual}"),
        ));
    }
    Ok(())
}

impl Qwen36MoeNativeConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        Self::load_with_policy(model_dir, Qwen36MoeGeometryPolicy::from_env())
    }

    pub fn load_with_policy(model_dir: &Path, policy: Qwen36MoeGeometryPolicy) -> Result<Self> {
        let raw = std::fs::read(model_dir.join(CONFIG_FILE)).map_err(|err| {
            Error::ModelLoadError(format!(
                "Failed to read Qwen3.5/3.6-MoE config {}: {err}",
                model_dir.join(CONFIG_FILE).display()
            ))
        })?;
        Self::from_json_with_policy(&raw, policy)
    }

    /// Pinned-policy parse. Test-only callers validate pinned-geometry
    /// rejection; production loads go through `load_with_policy`, and
    /// deriving the policy from the process environment here would make
    /// pinned-geometry tests racy against env-mutating tests.
    pub fn from_json(raw: &[u8]) -> Result<Self> {
        Self::from_json_with_policy(raw, Qwen36MoeGeometryPolicy::Pinned35B)
    }

    pub fn from_json_with_policy(raw: &[u8], policy: Qwen36MoeGeometryPolicy) -> Result<Self> {
        let root: serde_json::Value = serde_json::from_slice(raw)
            .map_err(|err| config_error("root", format!("invalid JSON: {err}")))?;

        if let Some(architectures) = root.get("architectures").and_then(|v| v.as_array()) {
            let primary = architectures
                .first()
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            if primary != PINNED_ARCHITECTURE {
                return Err(config_error(
                    "architectures[0]",
                    format!("expected {PINNED_ARCHITECTURE}, found {primary}"),
                ));
            }
        } else {
            return Err(config_error("architectures", "missing"));
        }

        let text_root = root
            .get("text_config")
            .ok_or_else(|| config_error("text_config", "missing"))?;
        let rope_root = text_root.get("rope_parameters").unwrap_or(&root);

        let block_count = req_usize(
            text_root
                .get("num_hidden_layers")
                .ok_or_else(|| config_error("text_config.num_hidden_layers", "missing"))?,
            "text_config.num_hidden_layers",
        )?;
        let interval = req_usize(
            text_root
                .get("full_attention_interval")
                .ok_or_else(|| config_error("text_config.full_attention_interval", "missing"))?,
            "text_config.full_attention_interval",
        )?;
        let hidden_size = req_usize(
            text_root
                .get("hidden_size")
                .ok_or_else(|| config_error("text_config.hidden_size", "missing"))?,
            "text_config.hidden_size",
        )?;
        let head_dim = req_usize(
            text_root
                .get("head_dim")
                .ok_or_else(|| config_error("text_config.head_dim", "missing"))?,
            "text_config.head_dim",
        )?;

        let layer_types = text_root
            .get("layer_types")
            .and_then(|v| v.as_array())
            .ok_or_else(|| config_error("text_config.layer_types", "missing"))?;
        if layer_types.len() != block_count {
            return Err(config_error(
                "text_config.layer_types",
                format!("expected {block_count} entries for {block_count} layers"),
            ));
        }
        let layer_types = layer_types
            .iter()
            .map(|value| {
                Qwen36MoeLayerType::parse(req_str(value, "text_config.layer_types[*]")?).ok_or_else(
                    || {
                        config_error(
                            "text_config.layer_types[*]",
                            format!("expected linear_attention or full_attention, found {value}"),
                        )
                    },
                )
            })
            .collect::<Result<Vec<_>>>()?;
        for (layer, kind) in layer_types.iter().enumerate() {
            let expected_full = (layer + 1) % interval == 0;
            let actual_full = *kind == Qwen36MoeLayerType::FullAttention;
            if expected_full != actual_full {
                return Err(config_error(
                    "text_config.layer_types",
                    format!(
                        "layer {layer} disagrees with full_attention_interval {interval}: pattern expects {}, config declares {kind:?}",
                        if expected_full { "full attention" } else { "linear attention" }
                    ),
                ));
            }
        }

        let rope_theta = req_f64(
            rope_root
                .get("rope_theta")
                .ok_or_else(|| config_error("rope_parameters.rope_theta", "missing"))?,
            "rope_parameters.rope_theta",
        )?;
        let partial_rotary_factor = req_f64(
            rope_root
                .get("partial_rotary_factor")
                .ok_or_else(|| config_error("rope_parameters.partial_rotary_factor", "missing"))?,
            "rope_parameters.partial_rotary_factor",
        )?;
        if !(0.0 < partial_rotary_factor && partial_rotary_factor <= 1.0) {
            return Err(config_error(
                "rope_parameters.partial_rotary_factor",
                "must be in (0, 1]",
            ));
        }
        let mrope_sections_raw = req_usize_array(
            rope_root
                .get("mrope_section")
                .ok_or_else(|| config_error("rope_parameters.mrope_section", "missing"))?,
            "rope_parameters.mrope_section",
            3,
        )?;
        let mrope_sections = [
            mrope_sections_raw[0],
            mrope_sections_raw[1],
            mrope_sections_raw[2],
        ];
        let mrope_interleaved = rope_root
            .get("mrope_interleaved")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if !mrope_interleaved {
            return Err(config_error(
                "rope_parameters.mrope_interleaved",
                "Qwen3.5 uses interleaved mrope; nested (non-interleaved) sections are not supported",
            ));
        }

        let rope_dimension_count = match rope_root.get("rope_dimension_count") {
            Some(value) => req_usize(value, "rope_parameters.rope_dimension_count")?,
            None => {
                let derived = (head_dim as f64 * partial_rotary_factor).round();
                if (derived - head_dim as f64 * partial_rotary_factor).abs() > f64::EPSILON {
                    return Err(config_error(
                        "rope_parameters.partial_rotary_factor",
                        "partial rotary factor must produce an integer rope dimension",
                    ));
                }
                derived as usize
            }
        };
        let covered_pairs: usize = mrope_sections.iter().sum();
        if covered_pairs * 2 != rope_dimension_count {
            return Err(config_error(
                "rope_parameters.mrope_section",
                format!(
                    "sections cover {} rotary pairs but rope dimension is {rope_dimension_count}",
                    covered_pairs
                ),
            ));
        }

        let time_step_rank = req_usize(
            text_root
                .get("linear_num_value_heads")
                .ok_or_else(|| config_error("text_config.linear_num_value_heads", "missing"))?,
            "text_config.linear_num_value_heads",
        )?;
        let group_count = req_usize(
            text_root
                .get("linear_num_key_heads")
                .ok_or_else(|| config_error("text_config.linear_num_key_heads", "missing"))?,
            "text_config.linear_num_key_heads",
        )?;
        let key_head_dim = req_usize(
            text_root
                .get("linear_key_head_dim")
                .ok_or_else(|| config_error("text_config.linear_key_head_dim", "missing"))?,
            "text_config.linear_key_head_dim",
        )?;
        let value_head_dim = req_usize(
            text_root
                .get("linear_value_head_dim")
                .ok_or_else(|| config_error("text_config.linear_value_head_dim", "missing"))?,
            "text_config.linear_value_head_dim",
        )?;
        let conv_kernel = req_usize(
            text_root
                .get("linear_conv_kernel_dim")
                .ok_or_else(|| config_error("text_config.linear_conv_kernel_dim", "missing"))?,
            "text_config.linear_conv_kernel_dim",
        )?;
        let num_experts = req_usize(
            text_root
                .get("num_experts")
                .ok_or_else(|| config_error("text_config.num_experts", "missing"))?,
            "text_config.num_experts",
        )?;
        let experts_per_tok = req_usize(
            text_root
                .get("num_experts_per_tok")
                .ok_or_else(|| config_error("text_config.num_experts_per_tok", "missing"))?,
            "text_config.num_experts_per_tok",
        )?;
        let moe_intermediate = req_usize(
            text_root
                .get("moe_intermediate_size")
                .ok_or_else(|| config_error("text_config.moe_intermediate_size", "missing"))?,
            "text_config.moe_intermediate_size",
        )?;
        let shared_expert_intermediate = req_usize(
            text_root
                .get("shared_expert_intermediate_size")
                .ok_or_else(|| {
                    config_error("text_config.shared_expert_intermediate_size", "missing")
                })?,
            "text_config.shared_expert_intermediate_size",
        )?;

        for (field, value) in [
            ("text_config.num_hidden_layers", block_count),
            ("text_config.full_attention_interval", interval),
            ("text_config.hidden_size", hidden_size),
            ("text_config.head_dim", head_dim),
            ("text_config.linear_num_value_heads", time_step_rank),
            ("text_config.linear_num_key_heads", group_count),
            ("text_config.linear_key_head_dim", key_head_dim),
            ("text_config.linear_value_head_dim", value_head_dim),
            ("text_config.linear_conv_kernel_dim", conv_kernel),
            ("text_config.num_experts", num_experts),
            ("text_config.num_experts_per_tok", experts_per_tok),
            ("text_config.moe_intermediate_size", moe_intermediate),
            (
                "text_config.shared_expert_intermediate_size",
                shared_expert_intermediate,
            ),
        ] {
            require_positive(field, value)?;
        }
        if experts_per_tok > num_experts {
            return Err(config_error(
                "text_config.num_experts_per_tok",
                "cannot exceed num_experts",
            ));
        }
        if group_count * key_head_dim != hidden_size {
            return Err(config_error(
                "text_config.linear_num_key_heads",
                format!(
                    "linear key width {} must equal hidden_size {hidden_size}",
                    group_count * key_head_dim
                ),
            ));
        }
        if time_step_rank * value_head_dim != 2 * hidden_size {
            return Err(config_error(
                "text_config.linear_num_value_heads",
                format!(
                    "linear value width {} must equal 2 * hidden_size {}",
                    time_step_rank * value_head_dim,
                    2 * hidden_size
                ),
            ));
        }

        let rms_norm_eps = req_f64(
            text_root
                .get("rms_norm_eps")
                .ok_or_else(|| config_error("text_config.rms_norm_eps", "missing"))?,
            "text_config.rms_norm_eps",
        )?;
        let mamba_dtype = text_root
            .get("mamba_ssm_dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("float32");
        if mamba_dtype != "float32" {
            return Err(config_error(
                "text_config.mamba_ssm_dtype",
                format!("expected float32 DeltaNet state, found {mamba_dtype}"),
            ));
        }
        let tie_word_embeddings = root
            .get("tie_word_embeddings")
            .or_else(|| text_root.get("tie_word_embeddings"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let quantization = root
            .get("quantization_config")
            .ok_or_else(|| config_error("quantization_config", "missing"))?;
        if req_str(
            quantization
                .get("quant_method")
                .ok_or_else(|| config_error("quantization_config.quant_method", "missing"))?,
            "quantization_config.quant_method",
        )? != "fp8"
        {
            return Err(config_error(
                "quantization_config.quant_method",
                "expected fp8",
            ));
        }
        if req_str(
            quantization
                .get("fmt")
                .ok_or_else(|| config_error("quantization_config.fmt", "missing"))?,
            "quantization_config.fmt",
        )? != "e4m3"
        {
            return Err(config_error("quantization_config.fmt", "expected e4m3"));
        }
        if req_str(
            quantization
                .get("activation_scheme")
                .ok_or_else(|| config_error("quantization_config.activation_scheme", "missing"))?,
            "quantization_config.activation_scheme",
        )? != "dynamic"
        {
            return Err(config_error(
                "quantization_config.activation_scheme",
                "expected dynamic",
            ));
        }
        let block_shape_raw = req_usize_array(
            quantization
                .get("weight_block_size")
                .ok_or_else(|| config_error("quantization_config.weight_block_size", "missing"))?,
            "quantization_config.weight_block_size",
            2,
        )?;
        let block_fp8 = BlockFp8Config {
            block_shape: [block_shape_raw[0], block_shape_raw[1]],
        };

        // Pinned-value checks run after structural validation so synthetic
        // fixtures only relax values, never structure.
        require_eq_usize(
            "text_config.num_hidden_layers",
            block_count,
            PINNED_BLOCK_COUNT,
            policy,
        )?;
        require_eq_usize(
            "text_config.full_attention_interval",
            interval,
            PINNED_FULL_ATTENTION_INTERVAL,
            policy,
        )?;
        require_eq_usize(
            "text_config.hidden_size",
            hidden_size,
            PINNED_HIDDEN_SIZE,
            policy,
        )?;
        require_eq_usize(
            "text_config.num_attention_heads",
            req_usize(
                text_root
                    .get("num_attention_heads")
                    .ok_or_else(|| config_error("text_config.num_attention_heads", "missing"))?,
                "text_config.num_attention_heads",
            )?,
            PINNED_ATTENTION_HEADS,
            policy,
        )?;
        require_eq_usize(
            "text_config.num_key_value_heads",
            req_usize(
                text_root
                    .get("num_key_value_heads")
                    .ok_or_else(|| config_error("text_config.num_key_value_heads", "missing"))?,
                "text_config.num_key_value_heads",
            )?,
            PINNED_ATTENTION_KV_HEADS,
            policy,
        )?;
        require_eq_usize("text_config.head_dim", head_dim, PINNED_HEAD_DIM, policy)?;
        require_eq_usize(
            "text_config.vocab_size",
            req_usize(
                text_root
                    .get("vocab_size")
                    .ok_or_else(|| config_error("text_config.vocab_size", "missing"))?,
                "text_config.vocab_size",
            )?,
            PINNED_VOCAB_SIZE,
            policy,
        )?;
        require_eq_usize(
            "text_config.max_position_embeddings",
            req_usize(
                text_root.get("max_position_embeddings").ok_or_else(|| {
                    config_error("text_config.max_position_embeddings", "missing")
                })?,
                "text_config.max_position_embeddings",
            )?,
            PINNED_CONTEXT_TOKENS,
            policy,
        )?;
        require_eq_f64(
            "rope_parameters.rope_theta",
            rope_theta,
            PINNED_ROPE_THETA,
            policy,
        )?;
        require_eq_f64(
            "rope_parameters.partial_rotary_factor",
            partial_rotary_factor,
            PINNED_PARTIAL_ROTARY_FACTOR,
            policy,
        )?;
        require_eq_usize(
            "rope_parameters.rope_dimension_count",
            rope_dimension_count,
            PINNED_ROPE_DIMENSION_COUNT,
            policy,
        )?;
        require_eq_usize(
            "text_config.linear_num_value_heads",
            time_step_rank,
            PINNED_SSM_TIME_STEP_RANK,
            policy,
        )?;
        require_eq_usize(
            "text_config.linear_num_key_heads",
            group_count,
            PINNED_SSM_GROUP_COUNT,
            policy,
        )?;
        require_eq_usize(
            "text_config.linear_key_head_dim",
            key_head_dim,
            PINNED_SSM_STATE_SIZE,
            policy,
        )?;
        require_eq_usize(
            "text_config.linear_value_head_dim",
            value_head_dim,
            PINNED_SSM_VALUE_HEAD_DIM,
            policy,
        )?;
        require_eq_usize(
            "text_config.num_experts",
            num_experts,
            PINNED_MOE_EXPERTS,
            policy,
        )?;
        require_eq_usize(
            "text_config.num_experts_per_tok",
            experts_per_tok,
            PINNED_MOE_EXPERTS_PER_TOK,
            policy,
        )?;
        require_eq_usize(
            "text_config.moe_intermediate_size",
            moe_intermediate,
            PINNED_MOE_INTERMEDIATE,
            policy,
        )?;
        require_eq_usize(
            "text_config.shared_expert_intermediate_size",
            shared_expert_intermediate,
            PINNED_SHARED_EXPERT_INTERMEDIATE,
            policy,
        )?;
        require_eq_f64(
            "text_config.rms_norm_eps",
            rms_norm_eps,
            PINNED_RMS_NORM_EPS,
            policy,
        )?;

        let text = Qwen36MoeTextConfig {
            block_count,
            full_attention_interval: interval,
            hidden_size,
            vocab_size: req_usize(
                text_root
                    .get("vocab_size")
                    .ok_or_else(|| config_error("text_config.vocab_size", "missing"))?,
                "text_config.vocab_size",
            )?,
            context_tokens: req_usize(
                text_root.get("max_position_embeddings").ok_or_else(|| {
                    config_error("text_config.max_position_embeddings", "missing")
                })?,
                "text_config.max_position_embeddings",
            )?,
            attention_head_count: req_usize(
                text_root
                    .get("num_attention_heads")
                    .ok_or_else(|| config_error("text_config.num_attention_heads", "missing"))?,
                "text_config.num_attention_heads",
            )?,
            attention_head_count_kv: req_usize(
                text_root
                    .get("num_key_value_heads")
                    .ok_or_else(|| config_error("text_config.num_key_value_heads", "missing"))?,
                "text_config.num_key_value_heads",
            )?,
            attention_key_length: head_dim,
            attention_value_length: head_dim,
            rope_theta,
            rope_dimension_count,
            mrope_sections,
            partial_rotary_factor,
            ssm_time_step_rank: time_step_rank,
            ssm_group_count: group_count,
            ssm_state_size: key_head_dim,
            ssm_value_head_dim: value_head_dim,
            ssm_conv_kernel: conv_kernel,
            moe_num_experts: num_experts,
            moe_num_experts_per_tok: experts_per_tok,
            moe_intermediate_size: moe_intermediate,
            shared_expert_intermediate_size: shared_expert_intermediate,
            has_shared_expert_gate: true,
            rms_norm_eps,
            mamba_ssm_dtype_f32: true,
            tie_word_embeddings,
        };

        Ok(Self {
            text,
            layer_types,
            block_fp8,
        })
    }
}

/// Canonical tensor scope after language-layout normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen36MoeTensorScope {
    /// Language-model tensor, canonicalized to the `model.` / `lm_head.` form.
    Text,
    /// Vision-tower tensor; accounted, not loaded (text-only scope).
    Vision,
    /// Multi-token-prediction tensor; recorded for the speculative phase.
    Mtp,
    /// Anything else — rejected as checkpoint drift.
    Unknown,
}

/// Normalize an index tensor name to its canonical text form.
///
/// Returns `None` for scopes that are not loaded (vision, MTP).
pub fn canonical_text_tensor_name(name: &str) -> Option<(Qwen36MoeTensorScope, Option<String>)> {
    if name == "lm_head.weight" {
        return Some((Qwen36MoeTensorScope::Text, Some(name.to_string())));
    }
    if name.starts_with("mtp.") {
        return Some((Qwen36MoeTensorScope::Mtp, None));
    }
    let stripped = name.strip_prefix("model.")?;
    if stripped.starts_with("visual.") {
        return Some((Qwen36MoeTensorScope::Vision, None));
    }
    if stripped.starts_with("mtp.") {
        return Some((Qwen36MoeTensorScope::Mtp, None));
    }
    if let Some(composite) = stripped.strip_prefix("language_model.") {
        if composite.starts_with("visual.") {
            return Some((Qwen36MoeTensorScope::Vision, None));
        }
        if composite.starts_with("mtp.") {
            return Some((Qwen36MoeTensorScope::Mtp, None));
        }
        return Some((
            Qwen36MoeTensorScope::Text,
            Some(format!("model.{composite}")),
        ));
    }
    Some((Qwen36MoeTensorScope::Text, Some(name.to_string())))
}

/// One tensor of the qwen3.6 MTP draft manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen36MoeMtpTensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub kind: ExpectedTensorKind,
}

impl Qwen36MoeMtpTensorSpec {
    /// The safetensors dtype the manifest kind requires.
    pub fn kind_to_safe_dtype(&self) -> SafeDType {
        match self.kind {
            ExpectedTensorKind::BlockFp8 => SafeDType::F8_E4M3,
            ExpectedTensorKind::BlockFp8Scale
            | ExpectedTensorKind::Dense
            | ExpectedTensorKind::OptionalDense => SafeDType::BF16,
        }
    }
}

/// Validated inventory of the checkpoint's `mtp.*` tensors, recorded when the
/// MTP load policy is enabled. Loading the draft head over these tensors is
/// the next MTP phase; this struct is the handoff between validation and
/// construction.
#[derive(Debug, Clone, Default)]
pub struct Qwen36MoeMtpInventory {
    pub tensors: BTreeMap<String, (Vec<usize>, SafeDType, u64)>,
}

/// Whether the load validates and records the MTP draft manifest. MTP stays
/// opt-in: the default load skips `mtp.*` exactly as before.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen36MoeMtpLoadPolicy {
    Disabled,
    Enabled,
}

/// The qwen3.6 MTP draft topology mirrors the published qwen3_5_moe MTP
/// layer: ONE recurrent decoder layer sharing the target's token embeddings
/// and LM head, with a dense FFN sized by the MoE intermediate width and the
/// target's gated full-attention geometry.
///
/// NOTE: the exact published manifest (names, shapes, dtypes) must be
/// verified against the real Qwen3.6-35B-A3B-FP8 checkpoint at the hardware
/// handoff — the loader fails closed on any drift, and MTP is disabled by
/// default, so an unverified hypothesis can never silently misload.
pub const QWEN36_MOE_MTP_LAYERS: usize = 1;
pub const QWEN36_MOE_MTP_TENSOR_COUNT: usize = 22;

pub fn mtp_tensor_plan(
    text: &Qwen36MoeTextConfig,
    block_shape: [usize; 2],
) -> Result<Vec<Qwen36MoeMtpTensorSpec>> {
    let hidden = text.hidden_size;
    let intermediate = text.moe_intermediate_size;
    let head_dim = text.attention_key_length;
    let checked = |label: &str, value: Option<usize>| {
        value.ok_or_else(|| config_error(label, "dimension product overflow"))
    };
    let query_width = checked(
        "MTP query projection width",
        text.attention_head_count.checked_mul(head_dim),
    )?;
    let gated_query_width = checked("MTP gated query width", query_width.checked_mul(2))?;
    let kv_width = checked(
        "MTP key/value projection width",
        text.attention_head_count_kv.checked_mul(head_dim),
    )?;
    let fused_input = checked("MTP fused input width", hidden.checked_mul(2))?;

    fn push_dense(specs: &mut Vec<Qwen36MoeMtpTensorSpec>, name: String, shape: Vec<usize>) {
        specs.push(Qwen36MoeMtpTensorSpec {
            name,
            shape,
            kind: ExpectedTensorKind::Dense,
        });
    }
    fn push_projection(
        specs: &mut Vec<Qwen36MoeMtpTensorSpec>,
        name: String,
        shape: [usize; 2],
        block_shape: [usize; 2],
    ) -> Result<()> {
        let [rows, cols] = shape;
        specs.push(Qwen36MoeMtpTensorSpec {
            name: name.clone(),
            shape: vec![rows, cols],
            kind: ExpectedTensorKind::BlockFp8,
        });
        specs.push(Qwen36MoeMtpTensorSpec {
            name: format!("{name}_scale_inv"),
            shape: vec![rows.div_ceil(block_shape[0]), cols.div_ceil(block_shape[1])],
            kind: ExpectedTensorKind::BlockFp8Scale,
        });
        Ok(())
    }

    let mut specs = Vec::with_capacity(QWEN36_MOE_MTP_TENSOR_COUNT);
    push_dense(&mut specs, "mtp.fc.weight".into(), vec![hidden, fused_input]);
    for layer in 0..QWEN36_MOE_MTP_LAYERS {
        let prefix = format!("mtp.layers.{layer}");
        push_dense(&mut specs, format!("{prefix}.input_layernorm.weight"), vec![hidden]);
        push_dense(
            &mut specs,
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![hidden],
        );
        push_projection(
            &mut specs,
            format!("{prefix}.mlp.gate_proj.weight"),
            [intermediate, hidden],
            block_shape,
        )?;
        push_projection(
            &mut specs,
            format!("{prefix}.mlp.up_proj.weight"),
            [intermediate, hidden],
            block_shape,
        )?;
        push_projection(
            &mut specs,
            format!("{prefix}.mlp.down_proj.weight"),
            [hidden, intermediate],
            block_shape,
        )?;
        push_dense(&mut specs, format!("{prefix}.self_attn.q_norm.weight"), vec![head_dim]);
        push_dense(&mut specs, format!("{prefix}.self_attn.k_norm.weight"), vec![head_dim]);
        for (projection_name, shape) in [
            ("q_proj", [gated_query_width, hidden]),
            ("k_proj", [kv_width, hidden]),
            ("v_proj", [kv_width, hidden]),
            ("o_proj", [hidden, query_width]),
        ] {
            push_projection(
                &mut specs,
                format!("{prefix}.self_attn.{projection_name}.weight"),
                shape,
                block_shape,
            )?;
        }
    }
    push_dense(&mut specs, "mtp.norm.weight".into(), vec![hidden]);
    push_dense(&mut specs, "mtp.pre_fc_norm_embedding.weight".into(), vec![hidden]);
    push_dense(&mut specs, "mtp.pre_fc_norm_hidden.weight".into(), vec![hidden]);

    if specs.len() != QWEN36_MOE_MTP_TENSOR_COUNT {
        return Err(Error::ModelLoadError(format!(
            "Qwen3.5/3.6-MoE MTP manifest resolved to {} tensors, expected {QWEN36_MOE_MTP_TENSOR_COUNT}",
            specs.len()
        )));
    }
    Ok(specs)
}

/// The expected checkpoint contract for one canonical text tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedTensor {
    pub shape: Vec<usize>,
    pub kind: ExpectedTensorKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpectedTensorKind {
    /// Dense weight; any of BF16/F16/F32 storage is accepted.
    Dense,
    /// Block-scaled FP8 projection with a `weight_scale_inv` companion.
    BlockFp8,
    /// BF16 inverse-scale companion of a block-FP8 projection.
    BlockFp8Scale,
    /// Dense weight that published checkpoints may omit.
    OptionalDense,
}

fn insert_fp8_projection(
    plan: &mut BTreeMap<String, ExpectedTensor>,
    name: String,
    rows: usize,
    cols: usize,
    block_shape: [usize; 2],
) {
    plan.insert(
        name.clone(),
        ExpectedTensor {
            shape: vec![rows, cols],
            kind: ExpectedTensorKind::BlockFp8,
        },
    );
    // Scale naming follows the same convention as the materializers
    // (`scale_name_for_weight`): the `.weight` suffix is replaced.
    let scale_name = match name.strip_suffix(".weight") {
        Some(stem) => format!("{stem}.weight_scale_inv"),
        None => format!("{name}.weight_scale_inv"),
    };
    plan.insert(
        scale_name,
        ExpectedTensor {
            shape: vec![rows.div_ceil(block_shape[0]), cols.div_ceil(block_shape[1])],
            kind: ExpectedTensorKind::BlockFp8Scale,
        },
    );
}

/// Build the required canonical text tensor plan for a validated config.
pub fn expected_text_tensor_plan(
    config: &Qwen36MoeNativeConfig,
) -> Result<BTreeMap<String, ExpectedTensor>> {
    let text = &config.text;
    let hidden = text.hidden_size;
    let block_shape = config.block_fp8.block_shape;
    let mut plan = BTreeMap::new();

    let insert = |plan: &mut BTreeMap<String, ExpectedTensor>,
                  name: String,
                  shape: Vec<usize>,
                  kind: ExpectedTensorKind| {
        plan.insert(name, ExpectedTensor { shape, kind });
    };

    insert(
        &mut plan,
        "model.embed_tokens.weight".into(),
        vec![text.vocab_size, hidden],
        ExpectedTensorKind::Dense,
    );
    if !text.tie_word_embeddings {
        insert(
            &mut plan,
            "lm_head.weight".into(),
            vec![text.vocab_size, hidden],
            ExpectedTensorKind::Dense,
        );
    }
    insert(
        &mut plan,
        "model.norm.weight".into(),
        vec![hidden],
        ExpectedTensorKind::Dense,
    );

    for layer in 0..text.block_count {
        let prefix = format!("model.layers.{layer}");
        insert(
            &mut plan,
            format!("{prefix}.input_layernorm.weight"),
            vec![hidden],
            ExpectedTensorKind::Dense,
        );
        insert(
            &mut plan,
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![hidden],
            ExpectedTensorKind::Dense,
        );

        // Sparse MoE feed-forward block (identical on every layer).
        insert(
            &mut plan,
            format!("{prefix}.mlp.gate.weight"),
            vec![text.moe_num_experts, hidden],
            ExpectedTensorKind::Dense,
        );
        for expert in 0..text.moe_num_experts {
            for (suffix, shape) in expert_projection_shapes(text) {
                insert_fp8_projection(
                    &mut plan,
                    format!("{prefix}.mlp.experts.{expert}.{suffix}"),
                    shape[0],
                    shape[1],
                    block_shape,
                );
            }
        }
        for (suffix, shape) in expert_projection_shapes(text) {
            insert_fp8_projection(
                &mut plan,
                format!("{prefix}.mlp.shared_expert.{suffix}"),
                shape[0],
                shape[1],
                block_shape,
            );
        }
        insert(
            &mut plan,
            format!("{prefix}.mlp.shared_expert_gate.weight"),
            vec![1, hidden],
            ExpectedTensorKind::OptionalDense,
        );

        if text.is_full_attention_layer(layer) {
            // The gated full attention fuses the per-head sigmoid output
            // gate into q_proj: the published checkpoint's q_proj carries
            // `num_heads * head_dim * 2` rows (transformers qwen3_5_moe
            // `Qwen3_5MoeAttention` chunks the q_proj output into query
            // and gate halves; llama.cpp encodes the same fusion, which is
            // what the shared trunk consumes).
            insert_fp8_projection(
                &mut plan,
                format!("{prefix}.self_attn.q_proj.weight"),
                text.attention_query_width() * 2,
                hidden,
                block_shape,
            );
            insert_fp8_projection(
                &mut plan,
                format!("{prefix}.self_attn.k_proj.weight"),
                text.attention_kv_width(),
                hidden,
                block_shape,
            );
            insert_fp8_projection(
                &mut plan,
                format!("{prefix}.self_attn.v_proj.weight"),
                text.attention_kv_width(),
                hidden,
                block_shape,
            );
            insert_fp8_projection(
                &mut plan,
                format!("{prefix}.self_attn.o_proj.weight"),
                hidden,
                text.attention_query_width(),
                block_shape,
            );
            insert(
                &mut plan,
                format!("{prefix}.self_attn.q_norm.weight"),
                vec![text.attention_key_length],
                ExpectedTensorKind::Dense,
            );
            insert(
                &mut plan,
                format!("{prefix}.self_attn.k_norm.weight"),
                vec![text.attention_key_length],
                ExpectedTensorKind::Dense,
            );
        } else {
            // The published FP8 checkpoints quantize the two wide DeltaNet
            // input projections as 128x128 block FP8; the per-head tensors
            // (in_proj_a/b, A_log, dt_bias, conv1d, norm) stay dense alongside
            // the block-FP8 out_proj.
            insert_fp8_projection(
                &mut plan,
                format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                text.ssm_conv_channels(),
                hidden,
                block_shape,
            );
            insert_fp8_projection(
                &mut plan,
                format!("{prefix}.linear_attn.in_proj_z.weight"),
                text.ssm_v_width(),
                hidden,
                block_shape,
            );
            insert(
                &mut plan,
                format!("{prefix}.linear_attn.in_proj_b.weight"),
                vec![text.ssm_time_step_rank, hidden],
                ExpectedTensorKind::Dense,
            );
            insert(
                &mut plan,
                format!("{prefix}.linear_attn.in_proj_a.weight"),
                vec![text.ssm_time_step_rank, hidden],
                ExpectedTensorKind::Dense,
            );
            insert(
                &mut plan,
                format!("{prefix}.linear_attn.A_log"),
                vec![text.ssm_time_step_rank],
                ExpectedTensorKind::Dense,
            );
            insert(
                &mut plan,
                format!("{prefix}.linear_attn.dt_bias"),
                vec![text.ssm_time_step_rank],
                ExpectedTensorKind::Dense,
            );
            insert(
                &mut plan,
                format!("{prefix}.linear_attn.conv1d.weight"),
                vec![text.ssm_conv_channels(), 1, text.ssm_conv_kernel],
                ExpectedTensorKind::Dense,
            );
            // The gated RMS norm applies per value head: its weight is
            // `linear_value_head_dim` wide (the trunk reshapes V into
            // heads × head_dim before norming), not the full value width.
            insert(
                &mut plan,
                format!("{prefix}.linear_attn.norm.weight"),
                vec![text.ssm_value_head_dim],
                ExpectedTensorKind::Dense,
            );
            insert_fp8_projection(
                &mut plan,
                format!("{prefix}.linear_attn.out_proj.weight"),
                hidden,
                text.ssm_v_width(),
                block_shape,
            );
        }
    }

    Ok(plan)
}

fn expert_projection_shapes(text: &Qwen36MoeTextConfig) -> [(&'static str, Vec<usize>); 3] {
    [
        (
            "gate_proj.weight",
            vec![text.moe_intermediate_size, text.hidden_size],
        ),
        (
            "up_proj.weight",
            vec![text.moe_intermediate_size, text.hidden_size],
        ),
        (
            "down_proj.weight",
            vec![text.hidden_size, text.moe_intermediate_size],
        ),
    ]
}

/// Counting summary for tensors that are accounted but not loaded.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SkippedScopeInventory {
    pub vision_tensors: usize,
    pub mtp_tensors: usize,
    pub mtp_payload_bytes: u64,
}

/// Opened and validated Qwen3.5-MoE native checkpoint.
pub struct Qwen36MoeNativeCheckpoint {
    pub config: Qwen36MoeNativeConfig,
    pub tensors: IndexedSafetensors,
    /// Canonical text tensor name -> raw index name (layout normalization).
    text_tensor_names: BTreeMap<String, String>,
    pub skipped: SkippedScopeInventory,
    /// Present only when the load policy enabled MTP; the validated draft
    /// manifest the head-construction phase consumes.
    pub mtp: Option<Qwen36MoeMtpInventory>,
    /// The MTP manifest derived from the trunk geometry — always present so
    /// head construction can resolve tensor kinds even before validation.
    pub mtp_plan: Vec<Qwen36MoeMtpTensorSpec>,
    /// Canonical MTP tensor name -> raw index name (identity today, kept as a
    /// map so raw-name resolution is uniform across scopes).
    mtp_tensor_names: BTreeMap<String, String>,
}

impl Qwen36MoeNativeCheckpoint {
    pub fn open(model_dir: &Path) -> Result<Self> {
        Self::open_with_policy(model_dir, Qwen36MoeGeometryPolicy::from_env())
    }

    pub fn open_with_policies(
        model_dir: &Path,
        geometry: Qwen36MoeGeometryPolicy,
        mtp: Qwen36MoeMtpLoadPolicy,
    ) -> Result<Self> {
        let config = Qwen36MoeNativeConfig::load_with_policy(model_dir, geometry)?;
        let tensors = IndexedSafetensors::open(model_dir)?;
        Self::validate_with_policies(config, tensors, mtp)
    }

    /// Open with an explicit geometry policy. Production callers use
    /// `open`; synthetic-fixture tests pass `Synthetic` directly so
    /// parallel tests never mutate the process environment.
    pub fn open_with_policy(model_dir: &Path, policy: Qwen36MoeGeometryPolicy) -> Result<Self> {
        Self::open_with_policies(model_dir, policy, Qwen36MoeMtpLoadPolicy::Disabled)
    }

    pub fn validate(config: Qwen36MoeNativeConfig, tensors: IndexedSafetensors) -> Result<Self> {
        Self::validate_with_policies(config, tensors, Qwen36MoeMtpLoadPolicy::Disabled)
    }

    pub fn validate_with_policies(
        config: Qwen36MoeNativeConfig,
        tensors: IndexedSafetensors,
        mtp_policy: Qwen36MoeMtpLoadPolicy,
    ) -> Result<Self> {
        let plan = expected_text_tensor_plan(&config)?;
        let mtp_manifest = mtp_tensor_plan(&config.text, config.block_fp8.block_shape)?;
        let mtp_plan = match mtp_policy {
            Qwen36MoeMtpLoadPolicy::Disabled => None,
            Qwen36MoeMtpLoadPolicy::Enabled => Some(mtp_manifest.clone()),
        };
        let mut text_tensor_names = BTreeMap::new();
        let mut skipped = SkippedScopeInventory::default();
        let mut mtp_payload_bytes = 0u64;
        let mut mtp_tensor_names: BTreeMap<String, String> = BTreeMap::new();
        let mut mtp_tensors: BTreeMap<String, (Vec<usize>, SafeDType, u64)> = BTreeMap::new();

        for raw_name in tensors.tensor_names() {
            let (scope, canonical) = canonical_text_tensor_name(raw_name)
                .ok_or_else(|| {
                    Error::ModelLoadError(format!(
                        "Qwen3.5/3.6-MoE checkpoint tensor `{raw_name}` falls outside the known model scopes"
                    ))
                })?;
            match scope {
                Qwen36MoeTensorScope::Text => {
                    let canonical = canonical.expect("text scope always canonicalizes");
                    if let Some(existing) =
                        text_tensor_names.insert(canonical.clone(), raw_name.to_string())
                    {
                        return Err(Error::ModelLoadError(format!(
                            "Qwen3.5/3.6-MoE checkpoint declares both `{existing}` and `{raw_name}` for canonical text tensor `{canonical}`"
                        )));
                    }
                }
                Qwen36MoeTensorScope::Vision => {
                    skipped.vision_tensors += 1;
                }
                Qwen36MoeTensorScope::Mtp => {
                    skipped.mtp_tensors += 1;
                    mtp_payload_bytes += tensors
                        .tensor_info(raw_name)
                        .map(|info| info.storage_bytes as u64)
                        .unwrap_or(0);
                    // MTP canonical names are the raw index names (the scope
                    // canonicalizer returns None for them).
                    mtp_tensor_names.insert(raw_name.to_string(), raw_name.to_string());
                    if mtp_plan.is_some() {
                        let info = tensors.tensor_info(raw_name)?;
                        mtp_tensors.insert(
                            raw_name.to_string(),
                            (info.shape.clone(), info.dtype, info.storage_bytes as u64),
                        );
                    }
                }
                Qwen36MoeTensorScope::Unknown => {
                    return Err(Error::ModelLoadError(format!(
                        "Qwen3.5/3.6-MoE checkpoint tensor `{raw_name}` has an unknown scope"
                    )));
                }
            }
        }

        let mut missing = BTreeSet::new();
        for (name, expected) in &plan {
            let Some(raw_name) = text_tensor_names.get(name) else {
                if expected.kind != ExpectedTensorKind::OptionalDense {
                    missing.insert(name.clone());
                }
                continue;
            };
            let info = tensors.tensor_info(raw_name)?;
            let shape_ok = info.shape == expected.shape;
            let dtype_ok = match expected.kind {
                ExpectedTensorKind::Dense | ExpectedTensorKind::OptionalDense => matches!(
                    info.dtype,
                    SafeDType::BF16 | SafeDType::F16 | SafeDType::F32
                ),
                ExpectedTensorKind::BlockFp8 => info.dtype == SafeDType::F8_E4M3,
                ExpectedTensorKind::BlockFp8Scale => info.dtype == SafeDType::BF16,
            };
            if !shape_ok || !dtype_ok {
                return Err(Error::ModelLoadError(format!(
                    "Qwen3.5/3.6-MoE checkpoint tensor `{name}` contract drift: expected {:?} {:?}, found {:?} {:?}",
                    expected.kind, expected.shape, info.dtype, info.shape
                )));
            }
        }
        if !missing.is_empty() {
            let names: Vec<&str> = missing.iter().map(|name| name.as_str()).take(8).collect();
            return Err(Error::ModelLoadError(format!(
                "Qwen3.5/3.6-MoE checkpoint is missing {} required text tensors, including {names:?}",
                missing.len()
            )));
        }

        let mut unexpected: Vec<&str> = text_tensor_names
            .keys()
            .filter(|name| !plan.contains_key(*name))
            .map(|name| name.as_str())
            .collect();
        unexpected.sort_unstable();
        if !unexpected.is_empty() {
            let shown: Vec<&str> = unexpected.iter().copied().take(8).collect();
            return Err(Error::ModelLoadError(format!(
                "Qwen3.5/3.6-MoE checkpoint declares {} text tensors outside the validated plan, including {shown:?}; update the qwen36moe tensor plan before loading",
                unexpected.len()
            )));
        }

        skipped.mtp_payload_bytes = mtp_payload_bytes;

        // Fail-closed MTP manifest validation: every planned tensor must be
        // present with the exact shape and dtype, and no unplanned `mtp.*`
        // tensor may exist. A drift is a hard load error naming the delta —
        // the manifest is a handoff-verified hypothesis about the published
        // checkpoint, so any mismatch must stop the load, not guess.
        let mtp = if let Some(mtp_plan) = mtp_plan {
            for spec in &mtp_plan {
                let Some((shape, dtype, _)) = mtp_tensors.get(&spec.name) else {
                    return Err(Error::ModelLoadError(format!(
                        "Qwen3.5/3.6-MoE checkpoint is missing MTP tensor `{}` expected {:?}",
                        spec.name, spec.shape
                    )));
                };
                let dtype_ok = match spec.kind {
                    ExpectedTensorKind::Dense => {
                        matches!(dtype, SafeDType::BF16 | SafeDType::F16 | SafeDType::F32)
                    }
                    ExpectedTensorKind::BlockFp8 => *dtype == SafeDType::F8_E4M3,
                    ExpectedTensorKind::BlockFp8Scale => *dtype == SafeDType::BF16,
                    ExpectedTensorKind::OptionalDense => {
                        matches!(dtype, SafeDType::BF16 | SafeDType::F16 | SafeDType::F32)
                    }
                };
                if *shape != spec.shape || !dtype_ok {
                    return Err(Error::ModelLoadError(format!(
                        "Qwen3.5/3.6-MoE MTP tensor `{}` contract drift: expected {:?} {:?}, found {dtype:?} {shape:?}",
                        spec.name, spec.kind, spec.shape
                    )));
                }
            }
            if mtp_tensors.len() != mtp_plan.len() {
                let extra: Vec<&str> = mtp_tensors
                    .keys()
                    .filter(|name| !mtp_plan.iter().any(|spec| &spec.name == *name))
                    .map(|name| name.as_str())
                    .take(8)
                    .collect();
                return Err(Error::ModelLoadError(format!(
                    "Qwen3.5/3.6-MoE checkpoint declares {} unplanned MTP tensors, including {extra:?}; update the qwen36moe MTP manifest before enabling MTP",
                    mtp_tensors.len() - mtp_plan.len()
                )));
            }
            Some(Qwen36MoeMtpInventory {
                tensors: mtp_tensors,
            })
        } else {
            None
        };

        Ok(Self {
            config,
            tensors,
            text_tensor_names,
            skipped,
            mtp,
            mtp_plan: mtp_manifest,
            mtp_tensor_names,
        })
    }

    /// Raw index name for a canonical tensor name. MTP tensors are
    /// identity-named in the index (they never enter the text plan), so a
    /// `mtp.` canonical resolves to itself after an existence check.
    pub fn raw_tensor_name(&self, canonical: &str) -> Result<&str> {
        if let Some(raw) = self.mtp_tensor_names.get(canonical) {
            return Ok(raw.as_str());
        }
        self.text_tensor_names
            .get(canonical)
            .map(|s| s.as_str())
            .ok_or_else(|| {
                Error::ModelLoadError(format!(
                    "Qwen3.5/3.6-MoE checkpoint has no text tensor `{canonical}`"
                ))
            })
    }

    pub fn text_tensor_count(&self) -> usize {
        self.text_tensor_names.len()
    }

    /// Persistent projection residency policy for a backend.
    ///
    /// CPU packs requantized Q8_0 projections (the F32 expanded envelope for
    /// a 35B checkpoint is impractical and the Q8_0 path keeps Candle CPU
    /// QMatMul residency), Metal expands to F16 (Apple GPUs have no FP8
    /// path), CUDA keeps the checkpoint's raw block-FP8 bytes resident and
    /// decodes per GEMM inside the fp8 projection kernel, falling back per
    /// tensor to packed Q8_0 where the kernel contract cannot execute.
    pub fn projection_residency_policy(device: &DeviceProfile) -> Qwen36MoeProjectionResidency {
        Self::projection_residency_policy_with_performance(
            BackendKind::from(device.kind),
            &crate::performance::CudaPerformanceConfig::default(),
        )
    }

    /// Performance-config-aware residency: the qwen3.6 CUDA deployment
    /// defaults to raw block-FP8 residency (the pinned census is fully
    /// kernel-compatible), with `projection_backend = "q8"` — or the master
    /// CUDA switch — as the escape hatch back to expanded BF16 weights.
    /// This deliberately diverges from qwen3.8, whose Auto retains packed Q8.
    pub fn projection_residency_policy_with_performance(
        backend: BackendKind,
        performance: &crate::performance::CudaPerformanceConfig,
    ) -> Qwen36MoeProjectionResidency {
        match backend {
            BackendKind::Cpu => Qwen36MoeProjectionResidency::PackedQ8_0,
            BackendKind::Metal => Qwen36MoeProjectionResidency::ExpandedF16,
            BackendKind::Cuda
                if performance.enabled()
                    && performance.projection_backend
                        != crate::performance::CudaProjectionBackend::Q8 =>
            {
                Qwen36MoeProjectionResidency::NativeFp8WithQ8Fallback
            }
            BackendKind::Cuda => Qwen36MoeProjectionResidency::ExpandedBf16,
        }
    }

    /// Materialize one block-FP8 (or dense) projection in its persistent form.
    pub fn materialize_projection(
        &self,
        canonical_name: &str,
        expected_shape: [usize; 2],
        device: &candle_core::Device,
        residency: Qwen36MoeProjectionResidency,
    ) -> Result<Qwen36MoeProjection> {
        let raw_name = self.raw_tensor_name(canonical_name)?;
        let block_shape = self.config.block_fp8.block_shape;
        match residency {
            Qwen36MoeProjectionResidency::PackedQ8_0 => Ok(Qwen36MoeProjection::Packed(
                self.tensors.materialize_q8_projection(
                    raw_name,
                    expected_shape,
                    block_shape,
                    device,
                )?,
            )),
            Qwen36MoeProjectionResidency::ExpandedF16 => Ok(Qwen36MoeProjection::Dense(
                self.tensors.materialize_projection(
                    raw_name,
                    expected_shape,
                    block_shape,
                    ProjectionMaterialization::F16,
                    device,
                )?,
            )),
            Qwen36MoeProjectionResidency::ExpandedBf16 => Ok(Qwen36MoeProjection::Dense(
                self.tensors.materialize_projection(
                    raw_name,
                    expected_shape,
                    block_shape,
                    ProjectionMaterialization::BF16,
                    device,
                )?,
            )),
            Qwen36MoeProjectionResidency::ExpandedF32 => Ok(Qwen36MoeProjection::Dense(
                self.tensors.materialize_projection(
                    raw_name,
                    expected_shape,
                    block_shape,
                    ProjectionMaterialization::F32,
                    device,
                )?,
            )),
            Qwen36MoeProjectionResidency::NativeFp8WithQ8Fallback => {
                if Self::compact_block_fp8_supported(device, expected_shape, block_shape) {
                    Ok(Qwen36MoeProjection::CompactFp8(
                        self.materialize_compact_projection(raw_name, expected_shape, device)?,
                    ))
                } else {
                    Ok(Qwen36MoeProjection::Packed(
                        self.tensors.materialize_q8_projection(
                            raw_name,
                            expected_shape,
                            block_shape,
                            device,
                        )?,
                    ))
                }
            }
        }
    }

    /// Kernel contract of `kernels::cuda::fp8::block_fp8_projection`: 128x128
    /// block scales with weights `[n, k]` where `n % 64 == 0` and `k % 128 == 0`,
    /// executed on a CUDA device of compute capability >= 8. Outside the
    /// contract the native-FP8 residency falls back to packed Q8_0 per tensor.
    fn compact_block_fp8_supported(
        device: &candle_core::Device,
        expected_shape: [usize; 2],
        block_shape: [usize; 2],
    ) -> bool {
        block_shape == [128, 128]
            && crate::kernels::cuda::fp8::provider_supported(
                device,
                candle_core::DType::BF16,
                expected_shape[0],
                expected_shape[1],
            )
    }

    fn materialize_compact_projection(
        &self,
        raw_name: &str,
        expected_shape: [usize; 2],
        device: &candle_core::Device,
    ) -> Result<RawBlockFp8Projection> {
        self.tensors.materialize_block_fp8_raw(
            raw_name,
            expected_shape,
            self.config.block_fp8.block_shape,
            device,
        )
    }

    /// Materialize a dense (non-block-FP8) tensor such as norms, DeltaNet
    /// in_proj/conv tensors, router weights, or embeddings. Dense math
    /// tensors always expand; CPU keeps F32, Metal F16, CUDA BF16.
    pub fn materialize_dense(
        &self,
        canonical_name: &str,
        expected_shape: &[usize],
        device: &candle_core::Device,
        target: ProjectionMaterialization,
    ) -> Result<candle_core::Tensor> {
        let raw_name = self.raw_tensor_name(canonical_name)?;
        self.tensors
            .materialize_dense_tensor(raw_name, expected_shape, target, device)
    }

    /// Load a dense tensor as raw host F32 values.
    pub fn load_dense_f32(
        &self,
        canonical_name: &str,
        expected_shape: &[usize],
    ) -> Result<Vec<f32>> {
        let raw_name = self.raw_tensor_name(canonical_name)?;
        let tensor = self.tensors.materialize_dense_tensor(
            raw_name,
            expected_shape,
            ProjectionMaterialization::F32,
            &candle_core::Device::Cpu,
        )?;
        tensor
            .flatten_all()
            .and_then(|flat| flat.to_vec1::<f32>())
            .map_err(|err| {
                Error::ModelLoadError(format!(
                    "Failed to read dense tensor `{canonical_name}` as F32: {err}"
                ))
            })
    }
}

/// Persistent residency form of a block-FP8 projection.
#[derive(Clone)]
pub enum Qwen36MoeProjection {
    Dense(candle_core::Tensor),
    Packed(candle_core::quantized::QMatMul),
    /// Raw checkpoint residency: E4M3FN weight bytes plus F32 block scales
    /// stay resident and decode per GEMM inside the CUDA fp8 projection
    /// kernel. No expanded persistent weight tensor is created.
    CompactFp8(RawBlockFp8Projection),
}

/// Backend residency selection for Qwen3.5-MoE projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen36MoeProjectionResidency {
    PackedQ8_0,
    ExpandedF16,
    ExpandedBf16,
    ExpandedF32,
    /// Raw block-FP8 residency with a per-tensor packed-Q8_0 fallback for
    /// tensors whose geometry the fp8 projection kernel cannot execute.
    NativeFp8WithQ8Fallback,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn pinned_text_config_value() -> serde_json::Value {
        pinned_config_json()
    }

    fn pinned_config() -> Qwen36MoeNativeConfig {
        Qwen36MoeNativeConfig::from_json_with_policy(
            &serde_json::to_vec(&pinned_text_config_value()).unwrap(),
            Qwen36MoeGeometryPolicy::Pinned35B,
        )
        .unwrap()
    }

    fn synthetic_config() -> Qwen36MoeNativeConfig {
        let mut value = pinned_text_config_value();
        value["text_config"]["num_hidden_layers"] = json!(8);
        value["text_config"]["vocab_size"] = json!(64);
        value["text_config"]["max_position_embeddings"] = json!(512);
        value["text_config"]["num_experts"] = json!(2);
        value["text_config"]["num_experts_per_tok"] = json!(1);
        let mut layer_types = Vec::with_capacity(8);
        for layer in 0..8 {
            if (layer + 1) % 4 == 0 {
                layer_types.push(json!("full_attention"));
            } else {
                layer_types.push(json!("linear_attention"));
            }
        }
        value["text_config"]["layer_types"] = json!(layer_types);
        Qwen36MoeNativeConfig::from_json_with_policy(
            &serde_json::to_vec(&value).unwrap(),
            Qwen36MoeGeometryPolicy::Synthetic,
        )
        .unwrap()
    }

    /// Forward-capable variant of [`tiny_config`]: the published contract
    /// pins linear key width to hidden (2 heads × 16 = 32) and value width
    /// to 2 × hidden (4 heads × 16 = 64); the shared trunk's GDN
    /// V-head-repeat path requires value heads to be a multiple of key
    /// heads, and CPU Q8_0 requant needs every packed inner dimension
    /// divisible by 32 (hidden 32, query width 2×16=32, MoE width 32).
    fn forward_config() -> Qwen36MoeNativeConfig {
        Qwen36MoeNativeConfig::from_json_with_policy(
            r#"{
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "text_config": {
                    "num_hidden_layers": 4,
                    "full_attention_interval": 4,
                    "hidden_size": 32,
                    "vocab_size": 32,
                    "max_position_embeddings": 64,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 16,
                    "num_experts": 2,
                    "num_experts_per_tok": 1,
                    "moe_intermediate_size": 32,
                    "shared_expert_intermediate_size": 32,
                    "linear_num_value_heads": 4,
                    "linear_num_key_heads": 2,
                    "linear_key_head_dim": 16,
                    "linear_value_head_dim": 16,
                    "linear_conv_kernel_dim": 2,
                    "rms_norm_eps": 1e-6,
                    "mamba_ssm_dtype": "float32",
                    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                    "rope_parameters": {
                        "rope_type": "mrope",
                        "mrope_interleaved": true,
                        "mrope_section": [2, 2, 0],
                        "rope_theta": 1000000.0,
                        "partial_rotary_factor": 0.5
                    }
                },
                "quantization_config": {
                    "quant_method": "fp8",
                    "fmt": "e4m3",
                    "activation_scheme": "dynamic",
                    "weight_block_size": [4, 4]
                }
            }"#
            .as_bytes(),
            Qwen36MoeGeometryPolicy::Synthetic,
        )
        .unwrap()
    }

    #[test]
    fn native_checkpoint_builds_hybrid_trunk_and_forwards_finitely() {
        use crate::backends::kv::{CpuKvArena, KvArenaConfig, KvLayerConfig};
        use crate::engine::ModelInstanceId;
        use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
        use crate::models::architectures::qwen36moe::native_model::load_text_model_native;
        use crate::models::shared::attention::physical::PhysicalPagedKvCache;
        use std::sync::Arc;

        let config = forward_config();
        let dir = TestDir::new("trunk-forward");
        write_tiny_checkpoint(&config, dir.0.as_path());
        let checkpoint = Qwen36MoeNativeCheckpoint::open_with_policy(
            dir.0.as_path(),
            Qwen36MoeGeometryPolicy::Synthetic,
        )
        .unwrap();

        let device_profile = DeviceProfile::cpu();
        let (text_config, model, _mtp_head) = load_text_model_native(
            &checkpoint,
            &device_profile,
            &candle_core::Device::Cpu,
            &crate::performance::CudaPerformanceConfig::default(),
            false,
        )
        .unwrap();
        assert_eq!(text_config.block_count, 4);
        let moe = text_config
            .moe_ffn
            .expect("sparse geometry from native config");
        assert_eq!(moe.num_experts, 2);
        assert_eq!(moe.shared_expert_intermediate_size, 32);
        // One counter array per sparse layer.
        assert_eq!(model.expert_activation_counters().len(), 4);

        // Single full-attention layer (model layer 3 → physical 0).
        let id = KvArenaId {
            model_instance: ModelInstanceId::new(4243),
            backend: BackendKind::Cpu,
            device_ordinal: None,
            generation: 1,
        };
        let group = KvGroupId::new(1);
        let arena = Arc::new(
            CpuKvArena::new(KvArenaConfig {
                id,
                group,
                page_tokens: 8,
                capacity_pages: 8,
                growth: None,
                dtype: candle_core::DType::F32,
                layers: vec![KvLayerConfig {
                    binding: KvLayerBinding {
                        model_layer: 3,
                        physical_layer: 0,
                    },
                    num_kv_heads: 1,
                    key_head_dim: 16,
                    value_head_dim: 16,
                }],
            })
            .unwrap(),
        );
        let blocks = (0..8)
            .map(|index| CacheBlockRef {
                arena: id,
                group,
                index,
                slot_generation: 1,
            })
            .collect();
        let mut cache = PhysicalPagedKvCache::new(
            arena,
            vec![KvLayerBinding {
                model_layer: 3,
                physical_layer: 0,
            }],
            blocks,
            0,
        )
        .unwrap();
        let mut state = model.new_state();

        // Prefill three tokens, then decode one: every hybrid domain
        // (paged KV, recurrent F32 state, conv ring) plus the sparse-expert
        // FFNs execute, and the logits stay finite.
        let logits = model
            .prefill_token_ids_physical(
                &[1, 2, 3],
                &[[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                &mut state,
                &mut cache,
                true,
            )
            .unwrap()
            .expect("prefill logits");
        let values = logits
            .to_dtype(candle_core::DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(values.len(), config.text.vocab_size);
        assert!(
            values.iter().all(|v| v.is_finite()),
            "prefill logits must be finite"
        );

        let logits = model
            .forward_token_id_at_physical(4, [3, 3, 3], &mut state, &mut cache)
            .unwrap();
        let values = logits
            .to_dtype(candle_core::DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn residency_policy_follows_the_projection_backend_knob() {
        use crate::performance::{CudaPerformanceConfig, CudaProjectionBackend, OptimizationMode};
        let policy = Qwen36MoeNativeCheckpoint::projection_residency_policy_with_performance;
        // Portable backends are performance-independent.
        assert_eq!(
            policy(BackendKind::Cpu, &CudaPerformanceConfig::default()),
            Qwen36MoeProjectionResidency::PackedQ8_0
        );
        assert_eq!(
            policy(BackendKind::Metal, &CudaPerformanceConfig::default()),
            Qwen36MoeProjectionResidency::ExpandedF16
        );
        // CUDA default: raw block-FP8 residency (the deployment posture).
        assert_eq!(
            policy(BackendKind::Cuda, &CudaPerformanceConfig::default()),
            Qwen36MoeProjectionResidency::NativeFp8WithQ8Fallback
        );
        // Explicit Q8 (or the master CUDA switch) escapes to expanded BF16.
        let q8 = CudaPerformanceConfig {
            projection_backend: CudaProjectionBackend::Q8,
            ..CudaPerformanceConfig::default()
        };
        assert_eq!(
            policy(BackendKind::Cuda, &q8),
            Qwen36MoeProjectionResidency::ExpandedBf16
        );
        let master_off = CudaPerformanceConfig {
            mode: OptimizationMode::Off,
            ..CudaPerformanceConfig::default()
        };
        assert_eq!(
            policy(BackendKind::Cuda, &master_off),
            Qwen36MoeProjectionResidency::ExpandedBf16
        );
    }

    #[test]
    fn native_dense_source_honors_the_trunks_requested_dtype() {
        use crate::models::architectures::qwen35::text::Qwen35WeightSource;
        use crate::models::architectures::qwen36moe::native_model::Qwen36MoeNativeSource;

        let config = forward_config();
        let dir = TestDir::new("dense-dtype");
        write_tiny_checkpoint(&config, dir.0.as_path());
        let checkpoint = Qwen36MoeNativeCheckpoint::open_with_policy(
            dir.0.as_path(),
            Qwen36MoeGeometryPolicy::Synthetic,
        )
        .unwrap();

        // The CUDA plan is the case that matters: dense_target is BF16, but
        // the trunk requests F32 for the DeltaNet math tensors so they match
        // the F32 state arena. The source must honor the request instead of
        // silently returning the per-backend target dtype.
        let cuda_plan_on_cpu = DeviceProfile {
            device: candle_core::Device::Cpu,
            kind: crate::backends::DeviceKind::Cuda,
            capabilities: Default::default(),
            memory_pool: None,
        };
        let source = Qwen36MoeNativeSource::new(&checkpoint, &cuda_plan_on_cpu);
        let device = candle_core::Device::Cpu;

        let dt_bias = source
            .dense("blk.0.ssm_dt.bias", Some(candle_core::DType::F32), &device)
            .unwrap();
        assert_eq!(dt_bias.dtype(), candle_core::DType::F32);
        let conv_kernel = source
            .dense(
                "blk.0.ssm_conv1d.weight",
                Some(candle_core::DType::F32),
                &device,
            )
            .unwrap();
        assert_eq!(conv_kernel.dtype(), candle_core::DType::F32);
        // A_log is transformed to F32 at materialization regardless.
        let a = source
            .dense("blk.0.ssm_a", Some(candle_core::DType::F32), &device)
            .unwrap();
        assert_eq!(a.dtype(), candle_core::DType::F32);

        // Without a request the per-backend target still applies.
        let unconstrained = source
            .dense("blk.0.ssm_dt.bias", None, &device)
            .unwrap();
        assert_eq!(unconstrained.dtype(), candle_core::DType::BF16);
    }

    /// Run the shared trunk's prefill + decode under a backend dtype plan and
    /// assert the trunk's activation dtype survives the head with finite
    /// logits. Used by the hardware-gated plan tests: the CPU device cannot
    /// execute non-F32 plans faithfully (no BF16 gemm, unreliable F16 gemm),
    /// so a mixed-dtype graph must be validated on the backend that runs it.
    fn forward_under_dtype_plan(
        label: &str,
        model: &crate::models::architectures::qwen35::text::Qwen35TextModel,
        mut cache: crate::models::shared::attention::physical::PhysicalPagedKvCache,
        activation_dtype: candle_core::DType,
    ) {
        let mut state = model.new_state();
        let logits = model
            .prefill_token_ids_physical(
                &[1, 2, 3],
                &[[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                &mut state,
                &mut cache,
                true,
            )
            .unwrap()
            .expect("prefill logits");
        assert_eq!(
            logits.dtype(),
            activation_dtype,
            "{label}: trunk activation dtype must survive the head"
        );
        let values = logits
            .to_dtype(candle_core::DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            values.iter().all(|v| v.is_finite()),
            "{label}: prefill logits must be finite"
        );

        let logits = model
            .forward_token_id_at_physical(4, [3, 3, 3], &mut state, &mut cache)
            .unwrap();
        assert_eq!(logits.dtype(), activation_dtype, "{label}: decode dtype");
        let values = logits
            .to_dtype(candle_core::DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            values.iter().all(|v| v.is_finite()),
            "{label}: decode logits must be finite"
        );
    }

    #[cfg(feature = "metal")]
    #[test]
    fn native_trunk_forwards_under_the_metal_dtype_plan_on_metal() {
        use crate::backends::kv::{CandleAcceleratorKvArena, KvArenaConfig, KvLayerConfig};
        use crate::engine::ModelInstanceId;
        use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
        use crate::models::architectures::qwen35::text::Qwen35TextModel;
        use crate::models::architectures::qwen36moe::native_model::{
            qwen35_text_config_from_native, Qwen36MoeNativeSource,
        };
        use crate::models::shared::attention::physical::PhysicalPagedKvCache;
        use candle_core::{DType, DeviceLocation};
        use std::sync::Arc;

        let Some(device) = crate::backends::metal_device_if_available(0) else {
            eprintln!("metal device unavailable; metal plan leg not run");
            return;
        };
        let device_profile = DeviceProfile {
            device: device.clone(),
            kind: crate::backends::DeviceKind::Metal,
            capabilities: Default::default(),
            memory_pool: None,
        };

        let config = forward_config();
        let dir = TestDir::new("plan-metal-hw");
        write_tiny_checkpoint(&config, dir.0.as_path());
        let checkpoint = Qwen36MoeNativeCheckpoint::open_with_policy(
            dir.0.as_path(),
            Qwen36MoeGeometryPolicy::Synthetic,
        )
        .unwrap();

        // The production Metal plan through the real constructor: ExpandedF16
        // projections, F16 dense weights and norm weights, the F32 DeltaNet
        // island and state arena, and an F32 KV arena. Before the F32 island
        // this graph mixed F16 activations with F32 state in the conv and
        // recurrence.
        let source = Qwen36MoeNativeSource::new(&checkpoint, &device_profile);
        let text_config = qwen35_text_config_from_native(&checkpoint.config.text);
        let model = Qwen35TextModel::load_with_source(&source, &text_config, &device).unwrap();

        let DeviceLocation::Metal { gpu_id } = device.location() else {
            panic!("metal test device reported a non-metal location");
        };
        let gpu_id = gpu_id as u64;
        let id = KvArenaId {
            model_instance: ModelInstanceId::new(4246),
            backend: BackendKind::Metal,
            device_ordinal: Some((gpu_id ^ (gpu_id >> 32)) as u32),
            generation: 1,
        };
        let group = KvGroupId::new(1);
        let arena = Arc::new(
            CandleAcceleratorKvArena::new_mutation_only(
                KvArenaConfig {
                    id,
                    group,
                    page_tokens: 8,
                    capacity_pages: 8,
                    growth: None,
                    dtype: DType::F32,
                    layers: vec![KvLayerConfig {
                        binding: KvLayerBinding {
                            model_layer: 3,
                            physical_layer: 0,
                        },
                        num_kv_heads: 1,
                        key_head_dim: 16,
                        value_head_dim: 16,
                    }],
                },
                device.clone(),
            )
            .unwrap(),
        );
        let blocks = (0..8)
            .map(|index| CacheBlockRef {
                arena: id,
                group,
                index,
                slot_generation: 1,
            })
            .collect();
        let cache = PhysicalPagedKvCache::new(
            arena,
            vec![KvLayerBinding {
                model_layer: 3,
                physical_layer: 0,
            }],
            blocks,
            0,
        )
        .unwrap();
        forward_under_dtype_plan("plan-metal", &model, cache, DType::F16);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn native_trunk_dtype_plan_forwards_on_cuda_hardware() {
        use crate::backends::kv::{CandleAcceleratorKvArena, KvArenaConfig, KvLayerConfig};
        use crate::engine::ModelInstanceId;
        use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
        use crate::models::architectures::qwen35::text::Qwen35TextModel;
        use crate::models::architectures::qwen36moe::native_model::{
            qwen35_text_config_from_native, Qwen36MoeNativeSource,
        };
        use crate::models::shared::attention::physical::PhysicalPagedKvCache;
        use candle_core::{DType, DeviceLocation};
        use std::sync::Arc;

        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };

        let config = forward_config();
        let dir = TestDir::new("plan-cuda-hw");
        write_tiny_checkpoint(&config, dir.0.as_path());
        let checkpoint = Qwen36MoeNativeCheckpoint::open_with_policy(
            dir.0.as_path(),
            Qwen36MoeGeometryPolicy::Synthetic,
        )
        .unwrap();

        // The tiny fixture's projections are not all fp8-kernel compatible,
        // so this exercises the expanded-BF16 CUDA plan: real BF16 CUDA
        // matmuls through every trunk block, the F32 DeltaNet island, and an
        // F16 accelerator KV arena. The raw compact-FP8 residency has its own
        // kernel tests; the full-checkpoint FP8 E2E stays a hardware handoff.
        let source = Qwen36MoeNativeSource::for_plan_tests(
            &checkpoint,
            Qwen36MoeProjectionResidency::ExpandedBf16,
            ProjectionMaterialization::BF16,
        );
        let text_config = qwen35_text_config_from_native(&checkpoint.config.text);
        let model = Qwen35TextModel::load_with_source(&source, &text_config, &device).unwrap();

        let DeviceLocation::Cuda { gpu_id } = device.location() else {
            panic!("CUDA test device reported a non-CUDA location");
        };
        let id = KvArenaId {
            model_instance: ModelInstanceId::new(4245),
            backend: BackendKind::Cuda,
            device_ordinal: u32::try_from(gpu_id).ok(),
            generation: 1,
        };
        let group = KvGroupId::new(1);
        let arena = Arc::new(
            CandleAcceleratorKvArena::new_mutation_only(
                KvArenaConfig {
                    id,
                    group,
                    page_tokens: 8,
                    capacity_pages: 8,
                    growth: None,
                    dtype: DType::F16,
                    layers: vec![KvLayerConfig {
                        binding: KvLayerBinding {
                            model_layer: 3,
                            physical_layer: 0,
                        },
                        num_kv_heads: 1,
                        key_head_dim: 16,
                        value_head_dim: 16,
                    }],
                },
                device.clone(),
            )
            .unwrap(),
        );
        let blocks = (0..8)
            .map(|index| CacheBlockRef {
                arena: id,
                group,
                index,
                slot_generation: 1,
            })
            .collect();
        let cache = PhysicalPagedKvCache::new(
            arena,
            vec![KvLayerBinding {
                model_layer: 3,
                physical_layer: 0,
            }],
            blocks,
            0,
        )
        .unwrap();
        forward_under_dtype_plan("plan-cuda", &model, cache, DType::BF16);
    }

    #[test]
    fn parses_the_pinned_qwen35_moe_config() {
        let config = pinned_config();
        assert_eq!(config.text.block_count, 40);
        assert_eq!(config.text.full_attention_interval, 4);
        assert_eq!(config.text.hidden_size, 2_048);
        assert_eq!(config.text.vocab_size, 248_320);
        assert_eq!(config.text.attention_query_width(), 4_096);
        assert_eq!(config.text.attention_kv_width(), 512);
        assert_eq!(config.text.ssm_qk_width(), 2_048);
        assert_eq!(config.text.ssm_v_width(), 4_096);
        assert_eq!(config.text.ssm_conv_channels(), 8_192);
        assert_eq!(config.text.moe_num_experts, 256);
        assert_eq!(config.text.moe_num_experts_per_tok, 8);
        assert_eq!(config.text.mrope_sections, [11, 11, 10]);
        assert_eq!(config.block_fp8.block_shape, [128, 128]);
        assert_eq!(
            config
                .layer_types
                .iter()
                .filter(|kind| **kind == Qwen36MoeLayerType::FullAttention)
                .count(),
            10
        );
        assert!(!config.text.is_full_attention_layer(0));
        assert!(config.text.is_full_attention_layer(3));
    }

    #[test]
    fn rejects_layer_pattern_drift_with_field_specific_error() {
        let mut value = pinned_text_config_value();
        value["text_config"]["layer_types"][0] = json!("full_attention");
        let error = Qwen36MoeNativeConfig::from_json_with_policy(
            &serde_json::to_vec(&value).unwrap(),
            Qwen36MoeGeometryPolicy::Synthetic,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("text_config.layer_types"), "{error}");
        assert!(error.contains("full_attention_interval"), "{error}");
    }

    #[test]
    fn rejects_pinned_geometry_drift() {
        let mut value = pinned_text_config_value();
        value["text_config"]["num_experts"] = json!(128);
        let error = Qwen36MoeNativeConfig::from_json(&serde_json::to_vec(&value).unwrap())
            .unwrap_err()
            .to_string();
        assert!(error.contains("text_config.num_experts"), "{error}");
        assert!(error.contains("expects 256"), "{error}");
    }

    #[test]
    fn rejects_non_interleaved_mrope() {
        let mut value = pinned_text_config_value();
        value["text_config"]["rope_parameters"]["mrope_interleaved"] = json!(false);
        let error = Qwen36MoeNativeConfig::from_json_with_policy(
            &serde_json::to_vec(&value).unwrap(),
            Qwen36MoeGeometryPolicy::Synthetic,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("mrope_interleaved"), "{error}");
    }

    #[test]
    fn rejects_quantization_contract_drift() {
        let mut value = pinned_text_config_value();
        value["quantization_config"]["activation_scheme"] = json!("static");
        let error = Qwen36MoeNativeConfig::from_json(&serde_json::to_vec(&value).unwrap())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("quantization_config.activation_scheme"),
            "{error}"
        );
    }

    #[test]
    fn rejects_architecture_drift() {
        let mut value = pinned_text_config_value();
        value["architectures"] = json!(["Qwen3MoeForCausalLM"]);
        let error = Qwen36MoeNativeConfig::from_json(&serde_json::to_vec(&value).unwrap())
            .unwrap_err()
            .to_string();
        assert!(error.contains("architectures[0]"), "{error}");
    }

    #[test]
    fn canonicalizes_both_language_layouts_and_skips_auxiliary_scopes() {
        let (scope, canonical) =
            canonical_text_tensor_name("model.layers.0.input_layernorm.weight")
                .expect("plain layout");
        assert_eq!(scope, Qwen36MoeTensorScope::Text);
        assert_eq!(
            canonical.as_deref(),
            Some("model.layers.0.input_layernorm.weight")
        );

        let (scope, canonical) =
            canonical_text_tensor_name("model.language_model.layers.0.mlp.gate.weight")
                .expect("composite layout");
        assert_eq!(scope, Qwen36MoeTensorScope::Text);
        assert_eq!(canonical.as_deref(), Some("model.layers.0.mlp.gate.weight"));

        let (scope, canonical) = canonical_text_tensor_name("lm_head.weight").expect("lm head");
        assert_eq!(scope, Qwen36MoeTensorScope::Text);
        assert_eq!(canonical.as_deref(), Some("lm_head.weight"));

        let (scope, canonical) =
            canonical_text_tensor_name("model.visual.blocks.0.attn.qkv.weight").expect("vision");
        assert_eq!(scope, Qwen36MoeTensorScope::Vision);
        assert!(canonical.is_none());

        let (scope, canonical) =
            canonical_text_tensor_name("mtp.layers.0.self_attn.q_proj.weight").expect("mtp");
        assert_eq!(scope, Qwen36MoeTensorScope::Mtp);
        assert!(canonical.is_none());
    }

    #[test]
    fn plan_covers_every_layer_role_for_the_pinned_geometry() {
        let config = pinned_config();
        let plan = expected_text_tensor_plan(&config).unwrap();

        // 3 shared (embed + lm_head + final norm) + per layer: 2 norms +
        // router + 256 experts x 3 + shared expert x 3 + optional gate +
        // attention-role tensors (6 full, 9 linear) + one BF16 scale
        // companion per block-FP8 projection (775 full, 774 linear).
        let shared = 3usize;
        let per_layer_common = 2 + 1 + 256 * 3 + 3 + 1;
        let full_count = 10usize;
        let linear_count = 30usize;
        let full_scales = 256 * 3 + 3 + 4;
        let linear_scales = 256 * 3 + 3 + 3;
        let expected_count = shared
            + full_count * (per_layer_common + 6 + full_scales)
            + linear_count * (per_layer_common + 9 + linear_scales);
        assert_eq!(plan.len(), expected_count);
        assert_eq!(plan.len(), 62_303);

        let expert0 = plan
            .get("model.layers.0.mlp.experts.0.gate_proj.weight")
            .expect("expert tensor in plan");
        assert_eq!(expert0.kind, ExpectedTensorKind::BlockFp8);
        assert_eq!(expert0.shape, vec![512, 2_048]);

        let expert0_scale = plan
            .get("model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv")
            .expect("expert scale in plan");
        assert_eq!(expert0_scale.kind, ExpectedTensorKind::BlockFp8Scale);
        assert_eq!(expert0_scale.shape, vec![4, 16]);

        let expert0 = plan
            .get("model.layers.0.mlp.experts.0.gate_proj.weight")
            .expect("expert tensor in plan");
        assert_eq!(expert0.kind, ExpectedTensorKind::BlockFp8);
        assert_eq!(expert0.shape, vec![512, 2_048]);

        let gate = plan
            .get("model.layers.0.mlp.shared_expert_gate.weight")
            .expect("shared expert gate in plan");
        assert_eq!(gate.kind, ExpectedTensorKind::OptionalDense);

        let linear_in_proj = plan
            .get("model.layers.0.linear_attn.in_proj_qkv.weight")
            .expect("linear in_proj in plan");
        assert_eq!(linear_in_proj.kind, ExpectedTensorKind::BlockFp8);
        assert_eq!(linear_in_proj.shape, vec![8_192, 2_048]);

        let linear_in_proj_scale = plan
            .get("model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv")
            .expect("linear in_proj scale in plan");
        assert_eq!(linear_in_proj_scale.kind, ExpectedTensorKind::BlockFp8Scale);
        assert_eq!(linear_in_proj_scale.shape, vec![64, 16]);

        let linear_in_proj_z = plan
            .get("model.layers.0.linear_attn.in_proj_z.weight")
            .expect("linear in_proj_z in plan");
        assert_eq!(linear_in_proj_z.kind, ExpectedTensorKind::BlockFp8);
        assert_eq!(linear_in_proj_z.shape, vec![4_096, 2_048]);

        let linear_beta = plan
            .get("model.layers.0.linear_attn.in_proj_b.weight")
            .expect("linear beta projection in plan");
        assert_eq!(linear_beta.kind, ExpectedTensorKind::Dense);
        assert_eq!(linear_beta.shape, vec![32, 2_048]);

        let out_proj = plan
            .get("model.layers.0.linear_attn.out_proj.weight")
            .expect("linear out_proj in plan");
        assert_eq!(out_proj.kind, ExpectedTensorKind::BlockFp8);
        assert_eq!(out_proj.shape, vec![2_048, 4_096]);

        let q_proj = plan
            .get("model.layers.3.self_attn.q_proj.weight")
            .expect("full-attention layer in plan");
        assert_eq!(q_proj.kind, ExpectedTensorKind::BlockFp8);
        // Fused query + sigmoid gate: 16 heads × 256 dim × 2 halves.
        assert_eq!(q_proj.shape, vec![8_192, 2_048]);
    }

    #[test]
    fn plan_matches_the_published_checkpoint_census() {
        let config = pinned_config();
        let plan = expected_text_tensor_plan(&config).unwrap();

        // Fold the per-index plan into name-pattern rows: every digit run in
        // a canonical name (layer, expert, even the `1` in conv1d) becomes
        // `{}`. All entries sharing a pattern must agree on kind and shape.
        fn pattern(name: &str) -> String {
            let mut out = String::with_capacity(name.len());
            let mut chars = name.chars().peekable();
            while let Some(c) = chars.next() {
                if c.is_ascii_digit() {
                    while chars.peek().is_some_and(|next| next.is_ascii_digit()) {
                        chars.next();
                    }
                    out.push_str("{}");
                } else {
                    out.push(c);
                }
            }
            out
        }

        let mut rows: BTreeMap<String, (ExpectedTensorKind, Vec<usize>, usize)> = BTreeMap::new();
        for (name, expected) in &plan {
            let entry = rows
                .entry(pattern(name))
                .or_insert_with(|| (expected.kind, expected.shape.clone(), 0));
            assert_eq!(entry.0, expected.kind, "pattern kind drift at {name}");
            assert_eq!(entry.1, expected.shape, "pattern shape drift at {name}");
            entry.2 += 1;
        }

        // Frozen census of the published Qwen/Qwen3.6-35B-A3B-FP8 safetensors
        // headers (fetched 2026-10-04): 62,303 text-scope tensors, every
        // F8_E4M3 weight carrying a BF16 `weight_scale_inv` companion at
        // [ceil(rows/128), ceil(cols/128)]. The 3.5 checkpoint matches every
        // row except linear_attn.A_log and linear_attn.norm.weight (F32
        // there; both accepted by Dense). Fixtures derived from the plan
        // cannot catch plan-vs-published drift, so the observed census is
        // pinned here as executable contract.
        let published: &[(&str, ExpectedTensorKind, &[usize], usize)] = &[
            (
                "lm_head.weight",
                ExpectedTensorKind::Dense,
                &[248_320, 2_048],
                1,
            ),
            (
                "model.embed_tokens.weight",
                ExpectedTensorKind::Dense,
                &[248_320, 2_048],
                1,
            ),
            ("model.norm.weight", ExpectedTensorKind::Dense, &[2_048], 1),
            (
                "model.layers.{}.input_layernorm.weight",
                ExpectedTensorKind::Dense,
                &[2_048],
                40,
            ),
            (
                "model.layers.{}.post_attention_layernorm.weight",
                ExpectedTensorKind::Dense,
                &[2_048],
                40,
            ),
            (
                "model.layers.{}.mlp.gate.weight",
                ExpectedTensorKind::Dense,
                &[256, 2_048],
                40,
            ),
            (
                "model.layers.{}.mlp.shared_expert_gate.weight",
                ExpectedTensorKind::OptionalDense,
                &[1, 2_048],
                40,
            ),
            (
                "model.layers.{}.mlp.shared_expert.gate_proj.weight",
                ExpectedTensorKind::BlockFp8,
                &[512, 2_048],
                40,
            ),
            (
                "model.layers.{}.mlp.shared_expert.gate_proj.weight_scale_inv",
                ExpectedTensorKind::BlockFp8Scale,
                &[4, 16],
                40,
            ),
            (
                "model.layers.{}.mlp.shared_expert.up_proj.weight",
                ExpectedTensorKind::BlockFp8,
                &[512, 2_048],
                40,
            ),
            (
                "model.layers.{}.mlp.shared_expert.up_proj.weight_scale_inv",
                ExpectedTensorKind::BlockFp8Scale,
                &[4, 16],
                40,
            ),
            (
                "model.layers.{}.mlp.shared_expert.down_proj.weight",
                ExpectedTensorKind::BlockFp8,
                &[2_048, 512],
                40,
            ),
            (
                "model.layers.{}.mlp.shared_expert.down_proj.weight_scale_inv",
                ExpectedTensorKind::BlockFp8Scale,
                &[16, 4],
                40,
            ),
            (
                "model.layers.{}.mlp.experts.{}.down_proj.weight",
                ExpectedTensorKind::BlockFp8,
                &[2_048, 512],
                10_240,
            ),
            (
                "model.layers.{}.mlp.experts.{}.down_proj.weight_scale_inv",
                ExpectedTensorKind::BlockFp8Scale,
                &[16, 4],
                10_240,
            ),
            (
                "model.layers.{}.mlp.experts.{}.gate_proj.weight",
                ExpectedTensorKind::BlockFp8,
                &[512, 2_048],
                10_240,
            ),
            (
                "model.layers.{}.mlp.experts.{}.gate_proj.weight_scale_inv",
                ExpectedTensorKind::BlockFp8Scale,
                &[4, 16],
                10_240,
            ),
            (
                "model.layers.{}.mlp.experts.{}.up_proj.weight",
                ExpectedTensorKind::BlockFp8,
                &[512, 2_048],
                10_240,
            ),
            (
                "model.layers.{}.mlp.experts.{}.up_proj.weight_scale_inv",
                ExpectedTensorKind::BlockFp8Scale,
                &[4, 16],
                10_240,
            ),
            (
                "model.layers.{}.self_attn.k_norm.weight",
                ExpectedTensorKind::Dense,
                &[256],
                10,
            ),
            (
                "model.layers.{}.self_attn.q_norm.weight",
                ExpectedTensorKind::Dense,
                &[256],
                10,
            ),
            (
                "model.layers.{}.self_attn.k_proj.weight",
                ExpectedTensorKind::BlockFp8,
                &[512, 2_048],
                10,
            ),
            (
                "model.layers.{}.self_attn.k_proj.weight_scale_inv",
                ExpectedTensorKind::BlockFp8Scale,
                &[4, 16],
                10,
            ),
            (
                "model.layers.{}.self_attn.q_proj.weight",
                ExpectedTensorKind::BlockFp8,
                &[8_192, 2_048],
                10,
            ),
            (
                "model.layers.{}.self_attn.q_proj.weight_scale_inv",
                ExpectedTensorKind::BlockFp8Scale,
                &[64, 16],
                10,
            ),
            (
                "model.layers.{}.self_attn.v_proj.weight",
                ExpectedTensorKind::BlockFp8,
                &[512, 2_048],
                10,
            ),
            (
                "model.layers.{}.self_attn.v_proj.weight_scale_inv",
                ExpectedTensorKind::BlockFp8Scale,
                &[4, 16],
                10,
            ),
            (
                "model.layers.{}.self_attn.o_proj.weight",
                ExpectedTensorKind::BlockFp8,
                &[2_048, 4_096],
                10,
            ),
            (
                "model.layers.{}.self_attn.o_proj.weight_scale_inv",
                ExpectedTensorKind::BlockFp8Scale,
                &[16, 32],
                10,
            ),
            (
                "model.layers.{}.linear_attn.A_log",
                ExpectedTensorKind::Dense,
                &[32],
                30,
            ),
            (
                "model.layers.{}.linear_attn.conv{}d.weight",
                ExpectedTensorKind::Dense,
                &[8_192, 1, 4],
                30,
            ),
            (
                "model.layers.{}.linear_attn.dt_bias",
                ExpectedTensorKind::Dense,
                &[32],
                30,
            ),
            (
                "model.layers.{}.linear_attn.in_proj_a.weight",
                ExpectedTensorKind::Dense,
                &[32, 2_048],
                30,
            ),
            (
                "model.layers.{}.linear_attn.in_proj_b.weight",
                ExpectedTensorKind::Dense,
                &[32, 2_048],
                30,
            ),
            (
                "model.layers.{}.linear_attn.in_proj_qkv.weight",
                ExpectedTensorKind::BlockFp8,
                &[8_192, 2_048],
                30,
            ),
            (
                "model.layers.{}.linear_attn.in_proj_qkv.weight_scale_inv",
                ExpectedTensorKind::BlockFp8Scale,
                &[64, 16],
                30,
            ),
            (
                "model.layers.{}.linear_attn.in_proj_z.weight",
                ExpectedTensorKind::BlockFp8,
                &[4_096, 2_048],
                30,
            ),
            (
                "model.layers.{}.linear_attn.in_proj_z.weight_scale_inv",
                ExpectedTensorKind::BlockFp8Scale,
                &[32, 16],
                30,
            ),
            (
                "model.layers.{}.linear_attn.norm.weight",
                ExpectedTensorKind::Dense,
                &[128],
                30,
            ),
            (
                "model.layers.{}.linear_attn.out_proj.weight",
                ExpectedTensorKind::BlockFp8,
                &[2_048, 4_096],
                30,
            ),
            (
                "model.layers.{}.linear_attn.out_proj.weight_scale_inv",
                ExpectedTensorKind::BlockFp8Scale,
                &[16, 32],
                30,
            ),
        ];
        assert_eq!(rows.len(), published.len(), "pattern-set size");
        for (name, kind, shape, count) in published {
            let row = rows
                .get(*name)
                .unwrap_or_else(|| panic!("plan has no row for published pattern {name}"));
            assert_eq!(row.0, *kind, "{name}");
            assert_eq!(row.1, *shape, "{name}");
            assert_eq!(row.2, *count, "{name}");
        }
    }

    #[test]
    fn projection_residency_policy_matches_backend_envelopes() {
        let cpu = DeviceProfile::cpu();
        assert_eq!(
            Qwen36MoeNativeCheckpoint::projection_residency_policy(&cpu),
            Qwen36MoeProjectionResidency::PackedQ8_0
        );
    }

    // ---- Disk-based checkpoint validation (synthetic tiny geometry) ----

    struct TestDir(std::path::PathBuf);

    impl TestDir {
        fn path(&self) -> &std::path::Path {
            self.0.as_path()
        }
    }

    impl TestDir {
        fn new(label: &str) -> Self {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "izwi-qwen36moe-native-{label}-{}-{nonce}",
                std::process::id()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn write_index(dir: &Path, map: serde_json::Value) {
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({ "weight_map": map })).unwrap(),
        )
        .unwrap();
    }

    fn write_safetensors(path: &Path, tensors: &[(&str, SafeDType, Vec<usize>, &[u8])]) {
        use std::collections::BTreeMap as StdBTreeMap;
        let views = tensors
            .iter()
            .map(|(name, dtype, shape, data)| {
                (
                    (*name).to_string(),
                    safetensors::tensor::TensorView::new(*dtype, shape.clone(), data).unwrap(),
                )
            })
            .collect::<StdBTreeMap<_, _>>();
        safetensors::serialize_to_file(&views, &None, path).unwrap();
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| half::bf16::from_f32(*value).to_bits().to_le_bytes())
            .collect()
    }

    /// Fully coherent tiny geometry: hidden 32, 3 DeltaNet + 1 full-attention
    /// layer, 2 experts of intermediate 8 plus a shared expert, 2 rotary
    /// pairs, FP8 block shape [4, 4].
    fn tiny_config() -> Qwen36MoeNativeConfig {
        let mut value = pinned_text_config_value();
        value["text_config"]["num_hidden_layers"] = json!(4);
        value["text_config"]["hidden_size"] = json!(32);
        value["text_config"]["vocab_size"] = json!(32);
        value["text_config"]["max_position_embeddings"] = json!(64);
        value["text_config"]["num_attention_heads"] = json!(2);
        value["text_config"]["num_key_value_heads"] = json!(1);
        value["text_config"]["head_dim"] = json!(8);
        value["text_config"]["num_experts"] = json!(2);
        value["text_config"]["num_experts_per_tok"] = json!(1);
        value["text_config"]["moe_intermediate_size"] = json!(8);
        value["text_config"]["shared_expert_intermediate_size"] = json!(8);
        value["text_config"]["linear_num_value_heads"] = json!(2);
        value["text_config"]["linear_num_key_heads"] = json!(4);
        value["text_config"]["linear_key_head_dim"] = json!(8);
        value["text_config"]["linear_value_head_dim"] = json!(32);
        value["text_config"]["linear_conv_kernel_dim"] = json!(2);
        value["text_config"]["layer_types"] = json!([
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention"
        ]);
        value["text_config"]["rope_parameters"]["mrope_section"] = json!([1, 1, 0]);
        value["text_config"]["rope_parameters"]["partial_rotary_factor"] = json!(0.5);
        value["quantization_config"]["weight_block_size"] = json!([4, 4]);
        Qwen36MoeNativeConfig::from_json_with_policy(
            &serde_json::to_vec(&value).unwrap(),
            Qwen36MoeGeometryPolicy::Synthetic,
        )
        .unwrap()
    }

    type RawTensor = (String, SafeDType, Vec<usize>, Vec<u8>);

    fn bf16_ones(count: usize) -> Vec<u8> {
        bf16_bytes(&vec![1.0; count])
    }

    /// Uniform BF16 fill at a magnitude that keeps the all-constant fixture
    /// inside F16/BF16 range: unit-magnitude constants amplify multiplicatively
    /// through the 32-wide MoE cascade to ~1e5, which saturates half-precision
    /// plans into +Inf even though the graph itself is coherent.
    fn bf16_uniform(value: f32, count: usize) -> Vec<u8> {
        bf16_bytes(&vec![value; count])
    }

    fn push_dense(tensors: &mut Vec<RawTensor>, name: String, shape: Vec<usize>) {
        let count: usize = shape.iter().product();
        tensors.push((name, SafeDType::BF16, shape, bf16_uniform(0.05, count)));
    }

    fn push_fp8_proj(
        tensors: &mut Vec<RawTensor>,
        config: &Qwen36MoeNativeConfig,
        name: String,
        rows: usize,
        cols: usize,
    ) {
        let block_shape = config.block_fp8.block_shape;
        tensors.push((
            name.clone(),
            SafeDType::F8_E4M3,
            vec![rows, cols],
            vec![0x38u8; rows * cols],
        ));
        let scale_rows = rows.div_ceil(block_shape[0]);
        let scale_cols = cols.div_ceil(block_shape[1]);
        tensors.push((
            format!(
                "{}.weight_scale_inv",
                name.strip_suffix(".weight").unwrap_or(&name)
            ),
            SafeDType::BF16,
            vec![scale_rows, scale_cols],
            bf16_uniform(0.05, scale_rows * scale_cols),
        ));
    }

    fn tiny_checkpoint_tensors(config: &Qwen36MoeNativeConfig) -> Vec<RawTensor> {
        let text = &config.text;
        let hidden = text.hidden_size;
        let mut tensors: Vec<RawTensor> = Vec::new();

        push_dense(
            &mut tensors,
            "model.language_model.embed_tokens.weight".into(),
            vec![32, hidden],
        );
        push_dense(&mut tensors, "lm_head.weight".into(), vec![32, hidden]);
        push_dense(
            &mut tensors,
            "model.language_model.norm.weight".into(),
            vec![hidden],
        );

        for layer in 0..text.block_count {
            let prefix = format!("model.language_model.layers.{layer}");
            push_dense(
                &mut tensors,
                format!("{prefix}.input_layernorm.weight"),
                vec![hidden],
            );
            push_dense(
                &mut tensors,
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![hidden],
            );
            push_dense(
                &mut tensors,
                format!("{prefix}.mlp.gate.weight"),
                vec![text.moe_num_experts, hidden],
            );
            for expert in 0..text.moe_num_experts {
                push_fp8_proj(
                    &mut tensors,
                    config,
                    format!("{prefix}.mlp.experts.{expert}.gate_proj.weight"),
                    text.moe_intermediate_size,
                    hidden,
                );
                push_fp8_proj(
                    &mut tensors,
                    config,
                    format!("{prefix}.mlp.experts.{expert}.up_proj.weight"),
                    text.moe_intermediate_size,
                    hidden,
                );
                push_fp8_proj(
                    &mut tensors,
                    config,
                    format!("{prefix}.mlp.experts.{expert}.down_proj.weight"),
                    hidden,
                    text.moe_intermediate_size,
                );
            }
            push_fp8_proj(
                &mut tensors,
                config,
                format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
                text.moe_intermediate_size,
                hidden,
            );
            push_fp8_proj(
                &mut tensors,
                config,
                format!("{prefix}.mlp.shared_expert.up_proj.weight"),
                text.moe_intermediate_size,
                hidden,
            );
            push_fp8_proj(
                &mut tensors,
                config,
                format!("{prefix}.mlp.shared_expert.down_proj.weight"),
                hidden,
                text.moe_intermediate_size,
            );
            push_dense(
                &mut tensors,
                format!("{prefix}.mlp.shared_expert_gate.weight"),
                vec![1, hidden],
            );
            if text.is_full_attention_layer(layer) {
                push_fp8_proj(
                    &mut tensors,
                    config,
                    format!("{prefix}.self_attn.q_proj.weight"),
                    text.attention_query_width() * 2,
                    hidden,
                );
                push_fp8_proj(
                    &mut tensors,
                    config,
                    format!("{prefix}.self_attn.k_proj.weight"),
                    text.attention_kv_width(),
                    hidden,
                );
                push_fp8_proj(
                    &mut tensors,
                    config,
                    format!("{prefix}.self_attn.v_proj.weight"),
                    text.attention_kv_width(),
                    hidden,
                );
                push_fp8_proj(
                    &mut tensors,
                    config,
                    format!("{prefix}.self_attn.o_proj.weight"),
                    hidden,
                    text.attention_query_width(),
                );
                push_dense(
                    &mut tensors,
                    format!("{prefix}.self_attn.q_norm.weight"),
                    vec![text.attention_key_length],
                );
                push_dense(
                    &mut tensors,
                    format!("{prefix}.self_attn.k_norm.weight"),
                    vec![text.attention_key_length],
                );
            } else {
                push_fp8_proj(
                    &mut tensors,
                    config,
                    format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                    text.ssm_conv_channels(),
                    hidden,
                );
                push_fp8_proj(
                    &mut tensors,
                    config,
                    format!("{prefix}.linear_attn.in_proj_z.weight"),
                    text.ssm_v_width(),
                    hidden,
                );
                push_dense(
                    &mut tensors,
                    format!("{prefix}.linear_attn.in_proj_b.weight"),
                    vec![text.ssm_time_step_rank, hidden],
                );
                push_dense(
                    &mut tensors,
                    format!("{prefix}.linear_attn.in_proj_a.weight"),
                    vec![text.ssm_time_step_rank, hidden],
                );
                push_dense(
                    &mut tensors,
                    format!("{prefix}.linear_attn.A_log"),
                    vec![text.ssm_time_step_rank],
                );
                push_dense(
                    &mut tensors,
                    format!("{prefix}.linear_attn.dt_bias"),
                    vec![text.ssm_time_step_rank],
                );
                push_dense(
                    &mut tensors,
                    format!("{prefix}.linear_attn.conv1d.weight"),
                    vec![text.ssm_conv_channels(), 1, text.ssm_conv_kernel],
                );
                push_dense(
                    &mut tensors,
                    format!("{prefix}.linear_attn.norm.weight"),
                    vec![text.ssm_value_head_dim],
                );
                push_fp8_proj(
                    &mut tensors,
                    config,
                    format!("{prefix}.linear_attn.out_proj.weight"),
                    hidden,
                    text.ssm_v_width(),
                );
            }
        }
        tensors.push((
            "model.visual.blocks.0.attn.qkv.weight".into(),
            SafeDType::BF16,
            vec![4, 4],
            bf16_bytes(&[1.0; 16]),
        ));
        // The MTP draft manifest, generated from the same plan the loader
        // validates against — a drift in either side fails the tests.
        for spec in mtp_tensor_plan(&config.text, config.block_fp8.block_shape).unwrap() {
            let count: usize = spec.shape.iter().product();
            let bytes = match spec.kind {
                ExpectedTensorKind::Dense | ExpectedTensorKind::BlockFp8Scale => {
                    bf16_uniform(0.05, count)
                }
                ExpectedTensorKind::BlockFp8 => vec![0x38u8; count],
                ExpectedTensorKind::OptionalDense => bf16_uniform(0.05, count),
            };
            tensors.push((
                spec.name.clone(),
                spec.kind_to_safe_dtype(),
                spec.shape.clone(),
                bytes,
            ));
        }
        tensors
    }

    fn write_tiny_checkpoint(config: &Qwen36MoeNativeConfig, dir: &Path) {
        write_tiny_checkpoint_tensors(config, dir, tiny_checkpoint_tensors(config));
    }

    fn write_tiny_checkpoint_tensors(
        config: &Qwen36MoeNativeConfig,
        dir: &Path,
        tensors: Vec<RawTensor>,
    ) {
        std::fs::write(
            dir.join(CONFIG_FILE),
            json!({
                "architectures": [PINNED_ARCHITECTURE],
                "text_config": {
                    "num_hidden_layers": config.text.block_count,
                    "full_attention_interval": config.text.full_attention_interval,
                    "hidden_size": config.text.hidden_size,
                    "vocab_size": config.text.vocab_size,
                    "max_position_embeddings": config.text.context_tokens,
                    "num_attention_heads": config.text.attention_head_count,
                    "num_key_value_heads": config.text.attention_head_count_kv,
                    "head_dim": config.text.attention_key_length,
                    "num_experts": config.text.moe_num_experts,
                    "num_experts_per_tok": config.text.moe_num_experts_per_tok,
                    "moe_intermediate_size": config.text.moe_intermediate_size,
                    "shared_expert_intermediate_size": config.text.shared_expert_intermediate_size,
                    "linear_num_value_heads": config.text.ssm_time_step_rank,
                    "linear_num_key_heads": config.text.ssm_group_count,
                    "linear_key_head_dim": config.text.ssm_state_size,
                    "linear_value_head_dim": config.text.ssm_value_head_dim,
                    "linear_conv_kernel_dim": config.text.ssm_conv_kernel,
                    "rms_norm_eps": config.text.rms_norm_eps,
                    "mamba_ssm_dtype": "float32",
                    "layer_types": config
                        .layer_types
                        .iter()
                        .map(|kind| match kind {
                            Qwen36MoeLayerType::LinearAttention => "linear_attention",
                            Qwen36MoeLayerType::FullAttention => "full_attention",
                        })
                        .collect::<Vec<_>>(),
                    "rope_parameters": {
                        "rope_type": "mrope",
                        "mrope_interleaved": true,
                        "mrope_section": config.text.mrope_sections,
                        "rope_theta": config.text.rope_theta,
                        "partial_rotary_factor": config.text.partial_rotary_factor
                    }
                },
                "quantization_config": {
                    "quant_method": "fp8",
                    "fmt": "e4m3",
                    "activation_scheme": "dynamic",
                    "weight_block_size": config.block_fp8.block_shape
                }
            })
            .to_string(),
        )
        .unwrap();

        let mut weight_map = serde_json::Map::new();
        let tensor_refs: Vec<(&str, SafeDType, Vec<usize>, &[u8])> = tensors
            .iter()
            .map(|(name, dtype, shape, data)| {
                (name.as_str(), *dtype, shape.clone(), data.as_slice())
            })
            .collect();
        for (name, _, _, _) in &tensor_refs {
            weight_map.insert((*name).to_string(), json!("layers.safetensors"));
        }
        write_safetensors(
            &dir.join("layers.safetensors"),
            &tensor_refs
                .iter()
                .map(|(name, dtype, shape, data)| (*name, *dtype, shape.clone(), *data))
                .collect::<Vec<_>>(),
        );
        write_index(dir, serde_json::Value::Object(weight_map));
    }

    #[test]
    fn opens_a_synthetic_checkpoint_and_skips_auxiliary_scopes() {
        let config = tiny_config();
        let dir = TestDir::new("open-ok");
        write_tiny_checkpoint(&config, dir.0.as_path());

        let checkpoint = Qwen36MoeNativeCheckpoint::open_with_policy(
            dir.0.as_path(),
            Qwen36MoeGeometryPolicy::Synthetic,
        )
        .unwrap();
        let text = &checkpoint.config.text;
        assert_eq!(text.block_count, 4);
        assert_eq!(text.moe_num_experts, 2);
        assert_eq!(checkpoint.skipped.vision_tensors, 1);
        // The full MTP draft manifest rides every checkpoint; the default
        // load skips it until the MTP policy enables validation.
        assert_eq!(checkpoint.skipped.mtp_tensors, 22);
        assert!(checkpoint.skipped.mtp_payload_bytes > 0);
        assert!(checkpoint.mtp.is_none());
        // Composite layout is canonicalized: `raw_tensor_name` resolves the
        // raw index name for a canonical plan name.
        let raw = checkpoint
            .raw_tensor_name("model.layers.0.linear_attn.in_proj_qkv.weight")
            .unwrap();
        assert_eq!(
            raw,
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
        );
        let raw_lm = checkpoint.raw_tensor_name("lm_head.weight").unwrap();
        assert_eq!(raw_lm, "lm_head.weight");
    }

    #[test]
    fn native_fp8_residency_falls_back_to_packed_q8_off_cuda() {
        let config = tiny_config();
        let dir = TestDir::new("native-fp8-fallback");
        write_tiny_checkpoint(&config, dir.0.as_path());
        let checkpoint = Qwen36MoeNativeCheckpoint::open_with_policy(
            dir.0.as_path(),
            Qwen36MoeGeometryPolicy::Synthetic,
        )
        .unwrap();

        let projection = checkpoint
            .materialize_projection(
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                [8, 32],
                &candle_core::Device::Cpu,
                Qwen36MoeProjectionResidency::NativeFp8WithQ8Fallback,
            )
            .unwrap();
        assert!(matches!(projection, Qwen36MoeProjection::Packed(_)));
    }

    #[test]
    fn compact_fp8_residency_keeps_raw_bytes_and_decodes_block_scales() {
        use crate::models::architectures::qwen35::text::Qwen35Projection;

        let mut config = tiny_config();
        // The fp8 projection kernel contract pins 128x128 block scales; the
        // tiny geometry still produces a [1, 1] scale grid, and the CPU
        // kernel path executes any geometry so the decode stays testable
        // without CUDA hardware.
        config.block_fp8.block_shape = [128, 128];
        let dir = TestDir::new("compact-fp8-raw");
        let mut tensors = tiny_checkpoint_tensors(&config);
        let scale_name = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv";
        let scale = tensors
            .iter_mut()
            .find(|(name, ..)| name == scale_name)
            .unwrap();
        assert_eq!(scale.2, vec![1, 1]);
        scale.3 = bf16_bytes(&[2.0]);
        write_tiny_checkpoint_tensors(&config, dir.0.as_path(), tensors);

        let checkpoint = Qwen36MoeNativeCheckpoint::open_with_policy(
            dir.0.as_path(),
            Qwen36MoeGeometryPolicy::Synthetic,
        )
        .unwrap();
        let canonical = "model.layers.0.mlp.experts.0.gate_proj.weight";
        let expected_shape = [8, 32];
        let raw_name = checkpoint.raw_tensor_name(canonical).unwrap();
        let compact = checkpoint
            .materialize_compact_projection(raw_name, expected_shape, &candle_core::Device::Cpu)
            .unwrap();
        assert_eq!(compact.weights.dims(), &[8, 32]);
        assert_eq!(compact.weights.dtype(), candle_core::DType::U8);
        assert_eq!(compact.scales.dims(), &[1, 1]);
        assert_eq!(
            compact
                .scales
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            [2.0]
        );

        // One-hot activation at column 5: every weight decodes to 1.0, so the
        // projection output is the block scale itself.
        let mut input = vec![0f32; 32];
        input[5] = 1.0;
        let x = candle_core::Tensor::from_vec(input, (1, 32), &candle_core::Device::Cpu).unwrap();
        let y = Qwen35Projection::CompactFp8 {
            weights: compact.weights,
            scales: compact.scales,
        }
        .forward(&x)
        .unwrap();
        assert_eq!(y.dims(), &[1, 8]);
        assert_eq!(y.to_vec2::<f32>().unwrap(), [vec![2.0; 8]]);
    }

    /// Minimal HF tokenizer pair for the native fixture: every byte of the
    /// rendered ChatML prompt (letters, newline, specials) resolves to a
    /// token, and the model vocab (32) equals the tokenizer vocab so greedy
    /// argmax stays in range.
    fn write_fixture_tokenizer(dir: &std::path::Path) {
        let mut vocab = serde_json::Map::new();
        for (index, token) in [
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
            "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "\u{010A}", "0", "<|im_start|>",
            "<|im_end|>", "<|image_pad|>", "<|video_pad|>",
        ]
        .into_iter()
            .enumerate()
        {
            vocab.insert(token.to_string(), serde_json::json!(index as u32));
        }
        let tokenizer = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 28, "content": "<|im_start|>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": false, "special": true},
                {"id": 29, "content": "<|im_end|>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": false, "special": true},
                {"id": 30, "content": "<|image_pad|>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": false, "special": true},
                {"id": 31, "content": "<|video_pad|>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": null,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false,
                              "trim_offsets": false, "use_regex": false},
            "post_processor": null,
            "decoder": {"type": "ByteLevel", "add_prefix_space": true,
                        "trim_offsets": true, "use_regex": true},
            "model": {
                "type": "BPE", "dropout": null, "unk_token": null,
                "continuing_subword_prefix": null, "end_of_word_suffix": null,
                "fuse_unk": false, "byte_fallback": false,
                "vocab": vocab,
                "merges": []
            }
        });
        std::fs::write(
            dir.join("tokenizer.json"),
            serde_json::to_string(&tokenizer).unwrap(),
        )
        .unwrap();
        let tokenizer_config = serde_json::json!({
            "eos_token": "<|im_end|>",
            "chat_template": "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n"
        });
        std::fs::write(
            dir.join("tokenizer_config.json"),
            serde_json::to_string(&tokenizer_config).unwrap(),
        )
        .unwrap();
    }

    /// Per-row physical cache for the native fixture: one full-attention
    /// layer (model_layer 3, the interval-4 layout) over its own arena.
    fn native_physical_cache(
        model: &crate::models::architectures::qwen36moe::chat::Qwen36MoeChatModel,
    ) -> crate::models::shared::attention::physical::PhysicalPagedKvCache {
        use crate::backends::kv::{CpuKvArena, KvArenaConfig, KvLayerConfig};
        use crate::models::shared::attention::physical::PhysicalPagedKvCache;
        use crate::engine::ModelInstanceId;
        use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
        use std::sync::Arc;
        let kv_heads = model.text_config().attention_head_count_kv;
        let head_dim = model.text_config().attention_key_length;
        let id = KvArenaId {
            model_instance: ModelInstanceId::new(4248),
            backend: BackendKind::Cpu,
            device_ordinal: None,
            generation: 1,
        };
        let group = KvGroupId::new(1);
        let arena = Arc::new(
            CpuKvArena::new(KvArenaConfig {
                id,
                group,
                page_tokens: 8,
                capacity_pages: 16,
                growth: None,
                dtype: candle_core::DType::F32,
                layers: vec![KvLayerConfig {
                    binding: KvLayerBinding {
                        model_layer: 3,
                        physical_layer: 0,
                    },
                    num_kv_heads: kv_heads as u32,
                    key_head_dim: head_dim as u32,
                    value_head_dim: head_dim as u32,
                }],
            })
            .unwrap(),
        );
        let blocks = (0..16)
            .map(|index| CacheBlockRef {
                arena: id,
                group,
                index: index as u32,
                slot_generation: 1,
            })
            .collect();
        PhysicalPagedKvCache::new(
            arena,
            vec![KvLayerBinding {
                model_layer: 3,
                physical_layer: 0,
            }],
            blocks,
            0,
        )
        .unwrap()
    }

    #[test]
    fn mtp_quantum_matches_scalar_greedy_decode_token_for_token() {
        use crate::model::ModelVariant;
        use crate::models::shared::chat::{ChatGenerationConfig, ChatMessage, ChatRole};

        // The chat-level loader derives its geometry policy from the process
        // environment; the tiny native fixture needs the synthetic hatch.
        // The env lock keeps this exclusive of tests that clear the
        // process environment, and the variable is removed on the way out.
        let _env_guard = crate::env_test_lock().lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("IZWI_ALLOW_SYNTHETIC_QWEN36_MOE_GEOMETRY", "1");

        let write_native_fixture = |tag: &str| {
            let config = forward_config();
            let dir = TestDir::new(tag);
            write_tiny_checkpoint(&config, dir.path());
            write_fixture_tokenizer(dir.path());
            (config, dir)
        };
        let greedy_config = || ChatGenerationConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            ..ChatGenerationConfig::default()
        };
        let load = |tag: &str, mtp: bool| {
            let (config, dir) = write_native_fixture(tag);
            let device = DeviceProfile::cpu();
            // PerformanceConfig keeps private resolution state, so the knobs
            // are assigned after the default rather than via struct update.
            #[allow(clippy::field_reassign_with_default)]
            let performance = {
                let mut performance = crate::performance::PerformanceConfig::default();
                performance.cuda.mtp = if mtp {
                    crate::performance::OptimizationMode::Auto
                } else {
                    crate::performance::OptimizationMode::Off
                };
                performance.cuda.mtp_draft_tokens = if mtp { 2 } else { 1 };
                performance
            };
            let model = crate::models::architectures::qwen36moe::chat::Qwen36MoeChatModel::load_with_performance(
                dir.0.as_path(),
                ModelVariant::Qwen36Moe35BA3BFp8,
                device,
                &performance,
                false,
            )
            .unwrap();
            assert_eq!(model.preferred_decode_tokens(), if mtp { 3 } else { 1 });
            (config, dir, model)
        };

        let messages = vec![ChatMessage {
            role: ChatRole::User,
            content: "abcd".to_string(),
        }];
        let config = greedy_config();

        // Scalar reference: MTP off, plain decode steps.
        let (_config, dir_off, scalar_model) = load("mtp-parity-off", false);
        let cache = native_physical_cache(&scalar_model);
        let mut scalar_state = scalar_model
            .start_decode_state_physical(&messages, 24, &config, None, cache)
            .unwrap();
        let mut scalar_deltas = Vec::new();
        for _ in 0..3 {
            let step = scalar_model
                .decode_quantum(&mut scalar_state, 4)
                .unwrap();
            scalar_deltas.push(step.delta);
            assert!(!step.finished);
        }
        drop(scalar_state);
        std::fs::remove_dir_all(dir_off.path()).ok();

        // MTP: same prompt, speculative rounds — must be token-identical.
        let (_config, dir_on, mtp_model) = load("mtp-parity-on", true);
        let cache = native_physical_cache(&mtp_model);
        let mut mtp_state = mtp_model
            .start_decode_state_physical(&messages, 24, &config, None, cache)
            .unwrap();
        let mut mtp_deltas = Vec::new();
        for _ in 0..3 {
            let step = mtp_model.decode_quantum(&mut mtp_state, 4).unwrap();
            mtp_deltas.push(step.delta);
            assert!(!step.finished);
        }
        drop(mtp_state);
        std::fs::remove_dir_all(dir_on.path()).ok();

        assert_eq!(
            mtp_deltas, scalar_deltas,
            "speculative decode must match scalar greedy decode exactly"
        );
        std::env::remove_var("IZWI_ALLOW_SYNTHETIC_QWEN36_MOE_GEOMETRY");
    }

    #[test]
    fn mtp_head_loads_from_the_validated_manifest_and_drafts() {
        use crate::backends::kv::{CpuKvArena, KvArenaConfig, KvLayerConfig};
        use crate::engine::ModelInstanceId;
        use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
        use crate::models::architectures::qwen36moe::native_model::load_text_model_native;
        use crate::models::shared::attention::physical::PhysicalPagedKvCache;
        use candle_core::{DType, Device, Tensor};
        use std::sync::Arc;

        let config = forward_config();
        let dir = TestDir::new("mtp-head");
        write_tiny_checkpoint(&config, dir.0.as_path());
        let checkpoint = Qwen36MoeNativeCheckpoint::open_with_policies(
            dir.0.as_path(),
            Qwen36MoeGeometryPolicy::Synthetic,
            Qwen36MoeMtpLoadPolicy::Enabled,
        )
        .unwrap();
        let device_profile = DeviceProfile::cpu();
        let (text_config, model, mtp_head) = load_text_model_native(
            &checkpoint,
            &device_profile,
            &Device::Cpu,
            &crate::performance::CudaPerformanceConfig::default(),
            true,
        )
        .unwrap();
        let head = mtp_head.expect("MTP head constructed when the policy enables it");

        // The MTP layer binds one paged KV layer at model_layer = block_count.
        let id = KvArenaId {
            model_instance: ModelInstanceId::new(4247),
            backend: BackendKind::Cpu,
            device_ordinal: None,
            generation: 1,
        };
        let group = KvGroupId::new(1);
        let arena = Arc::new(
            CpuKvArena::new(KvArenaConfig {
                id,
                group,
                page_tokens: 8,
                capacity_pages: 8,
                growth: None,
                dtype: DType::F32,
                layers: vec![KvLayerConfig {
                    binding: KvLayerBinding {
                        model_layer: text_config.block_count as u32,
                        physical_layer: 0,
                    },
                    num_kv_heads: text_config.attention_head_count_kv as u32,
                    key_head_dim: text_config.attention_key_length as u32,
                    value_head_dim: text_config.attention_value_length as u32,
                }],
            })
            .unwrap(),
        );
        let blocks = (0..8)
            .map(|index| CacheBlockRef {
                arena: id,
                group,
                index: index as u32,
                slot_generation: 1,
            })
            .collect();
        let mut mtp_cache = PhysicalPagedKvCache::new(
            arena,
            vec![KvLayerBinding {
                model_layer: text_config.block_count as u32,
                physical_layer: 0,
            }],
            blocks,
            0,
        )
        .unwrap();

        // Greedy draft of depth 2 from a zero seed: one continuation pair is
        // committed to the MTP cache, both tokens stay in vocabulary.
        let seed = Tensor::zeros((1, 1, text_config.embedding_length), DType::F32, &Device::Cpu)
            .unwrap();
        let tokens = head
            .draft_greedy(
                &model,
                &seed,
                2,
                &[[3, 3, 3]],
                config.text.vocab_size,
                &mut mtp_cache,
            )
            .unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(
            tokens
                .iter()
                .all(|token| (*token as usize) < config.text.vocab_size)
        );
        assert_eq!(mtp_cache.context_len(), 1, "depth 2 commits one pair");
    }

    #[test]
    fn mtp_policy_validates_the_draft_manifest_fail_closed() {
        let config = forward_config();
        let dir = TestDir::new("mtp-ok");
        write_tiny_checkpoint(&config, dir.0.as_path());

        // Disabled (default): manifest skipped, no inventory.
        let checkpoint = Qwen36MoeNativeCheckpoint::open_with_policies(
            dir.0.as_path(),
            Qwen36MoeGeometryPolicy::Synthetic,
            Qwen36MoeMtpLoadPolicy::Disabled,
        )
        .unwrap();
        assert!(checkpoint.mtp.is_none());

        // Enabled: the fixture carries the exact plan, so validation passes
        // and the inventory records every draft tensor.
        let checkpoint = Qwen36MoeNativeCheckpoint::open_with_policies(
            dir.0.as_path(),
            Qwen36MoeGeometryPolicy::Synthetic,
            Qwen36MoeMtpLoadPolicy::Enabled,
        )
        .unwrap();
        let inventory = checkpoint.mtp.as_ref().expect("MTP inventory recorded");
        assert_eq!(inventory.tensors.len(), 22);
        assert!(inventory.tensors.contains_key("mtp.fc.weight"));
        assert!(
            inventory
                .tensors
                .contains_key("mtp.layers.0.self_attn.q_proj.weight")
        );
        assert!(inventory.tensors.contains_key("mtp.norm.weight"));

        // Missing tensor: fail closed naming it.
        let missing = |dropped: &str| {
            let dir = TestDir::new("mtp-missing");
            let tensors: Vec<RawTensor> = tiny_checkpoint_tensors(&config)
                .into_iter()
                .filter(|(name, ..)| name != dropped)
                .collect();
            write_tiny_checkpoint_tensors(&config, dir.0.as_path(), tensors);
            Qwen36MoeNativeCheckpoint::open_with_policies(
                dir.0.as_path(),
                Qwen36MoeGeometryPolicy::Synthetic,
                Qwen36MoeMtpLoadPolicy::Enabled,
            )
            .err()
            .expect("missing MTP tensor must fail closed")
            .to_string()
        };
        let error = missing("mtp.fc.weight");
        assert!(error.contains("missing MTP tensor"), "{error}");
        assert!(error.contains("mtp.fc.weight"), "{error}");

        // Extra unplanned tensor: fail closed.
        let dir = TestDir::new("mtp-extra");
        let mut tensors = tiny_checkpoint_tensors(&config);
        tensors.push((
            "mtp.mystery.weight".into(),
            SafeDType::BF16,
            vec![4],
            bf16_uniform(0.05, 4),
        ));
        write_tiny_checkpoint_tensors(&config, dir.0.as_path(), tensors);
        let error = Qwen36MoeNativeCheckpoint::open_with_policies(
            dir.0.as_path(),
            Qwen36MoeGeometryPolicy::Synthetic,
            Qwen36MoeMtpLoadPolicy::Enabled,
        )
        .err()
        .expect("unplanned MTP tensor must fail closed")
        .to_string();
        assert!(error.contains("unplanned MTP tensors"), "{error}");
        assert!(error.contains("mtp.mystery.weight"), "{error}");

        // Shape drift: fail closed.
        let dir = TestDir::new("mtp-shape");
        let mut tensors = tiny_checkpoint_tensors(&config);
        let drifted = tensors
            .iter_mut()
            .find(|(name, ..)| name == "mtp.norm.weight")
            .unwrap();
        drifted.2 = vec![drifted.2[0] + 1];
        drifted.3 = bf16_uniform(0.05, drifted.2[0]);
        write_tiny_checkpoint_tensors(&config, dir.0.as_path(), tensors);
        let error = Qwen36MoeNativeCheckpoint::open_with_policies(
            dir.0.as_path(),
            Qwen36MoeGeometryPolicy::Synthetic,
            Qwen36MoeMtpLoadPolicy::Enabled,
        )
        .err()
        .expect("MTP shape drift must fail closed")
        .to_string();
        assert!(error.contains("contract drift"), "{error}");
        assert!(error.contains("mtp.norm.weight"), "{error}");
    }

    #[test]
    fn rejects_unexpected_and_missing_text_tensors_by_name() {
        let config = tiny_config();
        let dir = TestDir::new("open-extra");
        write_tiny_checkpoint(&config, dir.0.as_path());
        // Rewrite the checkpoint with one extra unexpected text tensor.
        let mut with_extra = tiny_checkpoint_tensors(&config);
        with_extra.push((
            "model.language_model.layers.0.mlp.mystery.weight".into(),
            SafeDType::BF16,
            vec![4],
            bf16_bytes(&[1.0, 2.0, 3.0, 4.0]),
        ));
        let refs: Vec<(&str, SafeDType, Vec<usize>, &[u8])> = with_extra
            .iter()
            .map(|(name, dtype, shape, data)| {
                (name.as_str(), *dtype, shape.clone(), data.as_slice())
            })
            .collect();
        write_safetensors(&dir.0.join("extra.safetensors"), &refs);
        let mut weight_map = serde_json::Map::new();
        for (name, ..) in &refs {
            weight_map.insert((*name).to_string(), json!("extra.safetensors"));
        }
        write_index(dir.0.as_path(), serde_json::Value::Object(weight_map));

        let error = Qwen36MoeNativeCheckpoint::open_with_policy(
            dir.0.as_path(),
            Qwen36MoeGeometryPolicy::Synthetic,
        )
        .err()
        .expect("unexpected tensor must fail closed")
        .to_string();
        assert!(error.contains("outside the validated plan"), "{error}");
        assert!(error.contains("mystery"), "{error}");

        // Missing required tensor (drop the full-attention o_proj) fails by
        // naming the gap.
        let dir_missing = TestDir::new("open-missing");
        write_tiny_checkpoint(&config, dir_missing.0.as_path());
        let tensors = tiny_checkpoint_tensors(&config);
        let pruned: Vec<(&str, SafeDType, Vec<usize>, &[u8])> = tensors
            .iter()
            .filter(|(name, ..)| {
                !name.ends_with("layers.3.self_attn.o_proj.weight")
                    && !name.ends_with("layers.3.self_attn.o_proj.weight_scale_inv")
            })
            .map(|(name, dtype, shape, data)| {
                (name.as_str(), *dtype, shape.clone(), data.as_slice())
            })
            .collect();
        let mut missing_map = serde_json::Map::new();
        for (name, ..) in &pruned {
            missing_map.insert((*name).to_string(), json!("layers.safetensors"));
        }
        write_safetensors(&dir_missing.0.join("layers.safetensors"), &pruned);
        write_index(
            dir_missing.0.as_path(),
            serde_json::Value::Object(missing_map),
        );
        let error = Qwen36MoeNativeCheckpoint::open_with_policy(
            dir_missing.0.as_path(),
            Qwen36MoeGeometryPolicy::Synthetic,
        )
        .err()
        .expect("missing tensor must fail closed")
        .to_string();
        assert!(error.contains("missing 2 required text tensors"), "{error}");
        assert!(error.contains("o_proj"), "{error}");
    }
}
