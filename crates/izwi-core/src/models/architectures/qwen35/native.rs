//! Native Hugging Face checkpoint foundations for Qwen3.5.
//!
//! The published Qwen3.8 checkpoint is an indexed Safetensors bundle whose
//! matrix weights use 128x128 block-scaled `F8_E4M3`. Candle can deserialize
//! the storage dtype, but casting the tensor directly does not apply the
//! companion `weight_scale_inv`. This module deliberately keeps index lookup,
//! shard lifetime, validation, and projection materialization explicit.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::{self, File};
use std::path::{Component, Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use half::{bf16, f16};
use memmap2::MmapOptions;
use safetensors::{tensor::TensorView, Dtype as SafeDType, SafeTensors};
use serde::Deserialize;

use crate::error::{Error, Result};

use super::chat::Qwen35TextConfig;

const CONFIG_FILE: &str = "config.json";
const INDEX_FILE: &str = "model.safetensors.index.json";
const SCALE_SUFFIX: &str = ".weight_scale_inv";
const WEIGHT_SUFFIX: &str = ".weight";

/// The exact model revision against which the first native implementation is
/// designed and validated.
pub const QWEN38_27B_FP8_REVISION: &str = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35LayerType {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockFp8Config {
    pub block_shape: [usize; 2],
}

/// Validated native configuration plus the existing runtime geometry.
#[derive(Debug, Clone)]
pub struct Qwen35NativeConfig {
    pub text: Qwen35TextConfig,
    pub layer_types: Vec<Qwen35LayerType>,
    pub vocab_size: usize,
    pub attn_output_gate: bool,
    pub partial_rotary_factor: f64,
    pub mrope_interleaved: bool,
    pub tie_word_embeddings: bool,
    pub block_fp8: BlockFp8Config,
}

impl Qwen35NativeConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join(CONFIG_FILE);
        let raw = fs::read(&path).map_err(|err| {
            Error::ModelLoadError(format!(
                "Failed to read native Qwen3.5 config {}: {err}",
                path.display()
            ))
        })?;
        Self::from_json(&raw)
    }

    pub fn from_json(raw: &[u8]) -> Result<Self> {
        let config: HfConfig = serde_json::from_slice(raw).map_err(|err| {
            Error::ModelLoadError(format!("Invalid native Qwen3.5 config.json: {err}"))
        })?;
        validate_hf_config(config)
    }
}

#[derive(Debug, Deserialize)]
struct HfConfig {
    architectures: Vec<String>,
    language_model_only: bool,
    model_type: String,
    text_config: HfTextConfig,
    tie_word_embeddings: bool,
    quantization_config: HfQuantizationConfig,
}

#[derive(Debug, Deserialize)]
struct HfTextConfig {
    attention_bias: bool,
    attention_dropout: f64,
    attn_output_gate: bool,
    bos_token_id: u32,
    dtype: String,
    eos_token_id: u32,
    full_attention_interval: usize,
    head_dim: usize,
    hidden_act: String,
    hidden_size: usize,
    intermediate_size: usize,
    layer_types: Vec<String>,
    linear_conv_kernel_dim: usize,
    linear_key_head_dim: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    linear_value_head_dim: usize,
    mamba_ssm_dtype: String,
    max_position_embeddings: usize,
    model_type: String,
    mtp_num_hidden_layers: usize,
    mtp_use_dedicated_embeddings: bool,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    output_gate_type: String,
    partial_rotary_factor: f64,
    rms_norm_eps: f64,
    rope_parameters: HfRopeParameters,
    tie_word_embeddings: bool,
    use_cache: bool,
    vocab_size: usize,
}

#[derive(Debug, Deserialize)]
struct HfRopeParameters {
    mrope_interleaved: bool,
    mrope_section: Vec<usize>,
    partial_rotary_factor: f64,
    rope_theta: f64,
    rope_type: String,
}

#[derive(Debug, Deserialize)]
struct HfQuantizationConfig {
    activation_scheme: String,
    fmt: String,
    quant_method: String,
    weight_block_size: Vec<usize>,
}

fn validate_hf_config(config: HfConfig) -> Result<Qwen35NativeConfig> {
    require_config_eq(
        "architectures",
        config.architectures,
        vec!["Qwen3_5ForConditionalGeneration".to_string()],
    )?;
    require_config_eq("model_type", config.model_type.as_str(), "qwen3_5")?;
    require_config_eq("language_model_only", config.language_model_only, false)?;
    require_config_eq("tie_word_embeddings", config.tie_word_embeddings, false)?;

    let text = config.text_config;
    require_config_eq(
        "text_config.model_type",
        text.model_type.as_str(),
        "qwen3_5_text",
    )?;
    require_config_eq("text_config.attention_bias", text.attention_bias, false)?;
    require_config_float("text_config.attention_dropout", text.attention_dropout, 0.0)?;
    require_config_eq("text_config.attn_output_gate", text.attn_output_gate, true)?;
    require_config_eq("text_config.bos_token_id", text.bos_token_id, 248_044)?;
    require_config_eq("text_config.eos_token_id", text.eos_token_id, 248_044)?;
    require_config_eq("text_config.dtype", text.dtype.as_str(), "bfloat16")?;
    require_config_eq(
        "text_config.full_attention_interval",
        text.full_attention_interval,
        4,
    )?;
    require_config_eq("text_config.head_dim", text.head_dim, 256)?;
    require_config_eq("text_config.hidden_act", text.hidden_act.as_str(), "silu")?;
    require_config_eq("text_config.hidden_size", text.hidden_size, 5_120)?;
    require_config_eq(
        "text_config.intermediate_size",
        text.intermediate_size,
        17_408,
    )?;
    require_config_eq(
        "text_config.linear_conv_kernel_dim",
        text.linear_conv_kernel_dim,
        4,
    )?;
    require_config_eq(
        "text_config.linear_key_head_dim",
        text.linear_key_head_dim,
        128,
    )?;
    require_config_eq(
        "text_config.linear_num_key_heads",
        text.linear_num_key_heads,
        16,
    )?;
    require_config_eq(
        "text_config.linear_num_value_heads",
        text.linear_num_value_heads,
        48,
    )?;
    require_config_eq(
        "text_config.linear_value_head_dim",
        text.linear_value_head_dim,
        128,
    )?;
    require_config_eq(
        "text_config.mamba_ssm_dtype",
        text.mamba_ssm_dtype.as_str(),
        "float32",
    )?;
    require_config_eq(
        "text_config.max_position_embeddings",
        text.max_position_embeddings,
        262_144,
    )?;
    require_config_eq(
        "text_config.mtp_num_hidden_layers",
        text.mtp_num_hidden_layers,
        1,
    )?;
    require_config_eq(
        "text_config.mtp_use_dedicated_embeddings",
        text.mtp_use_dedicated_embeddings,
        false,
    )?;
    require_config_eq(
        "text_config.num_attention_heads",
        text.num_attention_heads,
        24,
    )?;
    require_config_eq("text_config.num_hidden_layers", text.num_hidden_layers, 64)?;
    require_config_eq(
        "text_config.num_key_value_heads",
        text.num_key_value_heads,
        4,
    )?;
    require_config_eq(
        "text_config.output_gate_type",
        text.output_gate_type.as_str(),
        "swish",
    )?;
    require_config_float(
        "text_config.partial_rotary_factor",
        text.partial_rotary_factor,
        0.25,
    )?;
    require_config_float("text_config.rms_norm_eps", text.rms_norm_eps, 1e-6)?;
    require_config_eq(
        "text_config.tie_word_embeddings",
        text.tie_word_embeddings,
        false,
    )?;
    require_config_eq("text_config.use_cache", text.use_cache, true)?;
    require_config_eq("text_config.vocab_size", text.vocab_size, 248_320)?;

    let expected_layers = (0..text.num_hidden_layers)
        .map(|index| {
            if (index + 1).is_multiple_of(text.full_attention_interval) {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect::<Vec<_>>();
    let actual_layers = text
        .layer_types
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    require_config_eq(
        "text_config.layer_types",
        actual_layers.as_slice(),
        expected_layers.as_slice(),
    )?;
    let layer_types = actual_layers
        .into_iter()
        .map(|kind| match kind {
            "linear_attention" => Qwen35LayerType::LinearAttention,
            "full_attention" => Qwen35LayerType::FullAttention,
            _ => unreachable!("layer types were checked above"),
        })
        .collect::<Vec<_>>();

    let rope = text.rope_parameters;
    require_config_eq(
        "text_config.rope_parameters.rope_type",
        rope.rope_type.as_str(),
        "default",
    )?;
    require_config_eq(
        "text_config.rope_parameters.mrope_interleaved",
        rope.mrope_interleaved,
        true,
    )?;
    require_config_eq(
        "text_config.rope_parameters.mrope_section",
        rope.mrope_section.as_slice(),
        [11, 11, 10].as_slice(),
    )?;
    require_config_float(
        "text_config.rope_parameters.partial_rotary_factor",
        rope.partial_rotary_factor,
        text.partial_rotary_factor,
    )?;
    require_config_float(
        "text_config.rope_parameters.rope_theta",
        rope.rope_theta,
        10_000_000.0,
    )?;
    let rope_dimension_count = (text.head_dim as f64 * text.partial_rotary_factor) as usize;
    let section_dimensions = rope.mrope_section.iter().sum::<usize>() * 2;
    if section_dimensions != rope_dimension_count {
        return Err(config_error(
            "text_config.rope_parameters.mrope_section",
            format!(
                "covers {section_dimensions} rotary dimensions, expected {rope_dimension_count}"
            ),
        ));
    }

    let quant = config.quantization_config;
    require_config_eq(
        "quantization_config.quant_method",
        quant.quant_method.as_str(),
        "fp8",
    )?;
    require_config_eq("quantization_config.fmt", quant.fmt.as_str(), "e4m3")?;
    require_config_eq(
        "quantization_config.activation_scheme",
        quant.activation_scheme.as_str(),
        "dynamic",
    )?;
    require_config_eq(
        "quantization_config.weight_block_size",
        quant.weight_block_size.as_slice(),
        [128, 128].as_slice(),
    )?;

    let ssm_inner_size = text
        .linear_num_value_heads
        .checked_mul(text.linear_value_head_dim)
        .ok_or_else(|| config_error("text_config.linear_num_value_heads", "dimension overflow"))?;
    let runtime = Qwen35TextConfig {
        architecture: "qwen3_5".to_string(),
        block_count: text.num_hidden_layers,
        context_length: text.max_position_embeddings,
        embedding_length: text.hidden_size,
        feed_forward_length: text.intermediate_size,
        attention_head_count: text.num_attention_heads,
        attention_head_count_kv: text.num_key_value_heads,
        attention_key_length: text.head_dim,
        attention_value_length: text.head_dim,
        rope_dimension_sections: rope.mrope_section,
        rope_dimension_count,
        rope_freq_base: rope.rope_theta,
        attention_layer_norm_rms_epsilon: text.rms_norm_eps,
        ssm_conv_kernel: text.linear_conv_kernel_dim,
        ssm_state_size: text.linear_key_head_dim,
        ssm_group_count: text.linear_num_key_heads,
        ssm_time_step_rank: text.linear_num_value_heads,
        ssm_inner_size,
        full_attention_interval: text.full_attention_interval,
    };

    Ok(Qwen35NativeConfig {
        text: runtime,
        layer_types,
        vocab_size: text.vocab_size,
        attn_output_gate: text.attn_output_gate,
        partial_rotary_factor: text.partial_rotary_factor,
        mrope_interleaved: rope.mrope_interleaved,
        tie_word_embeddings: text.tie_word_embeddings,
        block_fp8: BlockFp8Config {
            block_shape: [quant.weight_block_size[0], quant.weight_block_size[1]],
        },
    })
}

fn require_config_eq<T>(field: &str, actual: T, expected: T) -> Result<()>
where
    T: PartialEq + std::fmt::Debug,
{
    if actual != expected {
        return Err(config_error(
            field,
            format!("found {actual:?}, expected {expected:?}"),
        ));
    }
    Ok(())
}

fn require_config_float(field: &str, actual: f64, expected: f64) -> Result<()> {
    if !actual.is_finite() || (actual - expected).abs() > f64::EPSILON * expected.abs().max(1.0) {
        return Err(config_error(
            field,
            format!("found {actual:?}, expected {expected:?}"),
        ));
    }
    Ok(())
}

fn config_error(field: &str, detail: impl std::fmt::Display) -> Error {
    Error::ModelLoadError(format!(
        "Unsupported native Qwen3.5 config field `{field}`: {detail}"
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeTensorScope {
    Text,
    LmHead,
    Vision,
    Mtp,
    Unknown,
}

pub fn native_tensor_scope(name: &str) -> NativeTensorScope {
    if name.starts_with("model.language_model.") {
        NativeTensorScope::Text
    } else if name == "lm_head.weight" {
        NativeTensorScope::LmHead
    } else if name.starts_with("model.visual.") {
        NativeTensorScope::Vision
    } else if name.starts_with("mtp.") {
        NativeTensorScope::Mtp
    } else {
        NativeTensorScope::Unknown
    }
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndexDocument {
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeTensorInfo {
    pub dtype: SafeDType,
    pub shape: Vec<usize>,
    pub storage_bytes: usize,
    pub shard: PathBuf,
}

/// Tensor-to-shard index for an HF checkpoint.
///
/// No shard mappings are retained. Each tensor operation maps only its indexed
/// shard and releases that mapping before returning.
#[derive(Debug, Clone)]
pub struct IndexedSafetensors {
    model_dir: PathBuf,
    weight_map: BTreeMap<String, String>,
}

impl IndexedSafetensors {
    pub fn open(model_dir: &Path) -> Result<Self> {
        let model_dir = model_dir.canonicalize().map_err(|err| {
            Error::ModelLoadError(format!(
                "Failed to resolve native checkpoint directory {}: {err}",
                model_dir.display()
            ))
        })?;
        let index_path = model_dir.join(INDEX_FILE);
        let raw = fs::read(&index_path).map_err(|err| {
            Error::ModelLoadError(format!(
                "Failed to read Safetensors index {}: {err}",
                index_path.display()
            ))
        })?;
        let index: SafetensorsIndexDocument = serde_json::from_slice(&raw).map_err(|err| {
            Error::ModelLoadError(format!(
                "Invalid Safetensors index {}: {err}",
                index_path.display()
            ))
        })?;
        if index.weight_map.is_empty() {
            return Err(Error::ModelLoadError(format!(
                "Safetensors index {} has an empty weight_map",
                index_path.display()
            )));
        }

        let mut validated_shards = BTreeSet::new();
        for (tensor, shard) in &index.weight_map {
            if tensor.trim().is_empty() {
                return Err(Error::ModelLoadError(format!(
                    "Safetensors index {} contains an empty tensor name",
                    index_path.display()
                )));
            }
            validate_relative_shard_name(shard)?;
            validated_shards.insert(shard.as_str());
        }
        for shard in validated_shards {
            validate_shard_file(&model_dir, shard)?;
        }

        Ok(Self {
            model_dir,
            weight_map: index.weight_map,
        })
    }

    pub fn tensor_count(&self) -> usize {
        self.weight_map.len()
    }

    pub fn contains_tensor(&self, name: &str) -> bool {
        self.weight_map.contains_key(name)
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.weight_map.keys().map(String::as_str)
    }

    pub fn shard_names(&self) -> Vec<&str> {
        self.weight_map
            .values()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn tensor_info(&self, name: &str) -> Result<NativeTensorInfo> {
        let shard = self.shard_path_for_tensor(name)?;
        self.with_tensor_view(name, None, None, |view| {
            Ok(NativeTensorInfo {
                dtype: view.dtype(),
                shape: view.shape().to_vec(),
                storage_bytes: view.data().len(),
                shard,
            })
        })
    }

    pub fn with_tensor_view<T, F>(
        &self,
        name: &str,
        expected_dtype: Option<SafeDType>,
        expected_shape: Option<&[usize]>,
        consume: F,
    ) -> Result<T>
    where
        F: FnOnce(TensorView<'_>) -> Result<T>,
    {
        let shard_path = self.shard_path_for_tensor(name)?;
        let file = File::open(&shard_path).map_err(|err| {
            Error::ModelLoadError(format!(
                "Failed to open Safetensors shard {} for tensor `{name}`: {err}",
                shard_path.display()
            ))
        })?;
        // SAFETY: the mapping is immutable, the file is not mutated through
        // this process, and neither the view nor mapping escapes this callback.
        let mapping = unsafe { MmapOptions::new().map(&file) }.map_err(|err| {
            Error::ModelLoadError(format!(
                "Failed to map Safetensors shard {} for tensor `{name}`: {err}",
                shard_path.display()
            ))
        })?;
        let tensors = SafeTensors::deserialize(&mapping).map_err(|err| {
            Error::ModelLoadError(format!(
                "Failed to parse Safetensors shard {} for tensor `{name}`: {err}",
                shard_path.display()
            ))
        })?;
        let view = tensors.tensor(name).map_err(|err| {
            Error::ModelLoadError(format!(
                "Safetensors index maps tensor `{name}` to {}, but the shard does not contain it: {err}",
                shard_path.display()
            ))
        })?;
        if let Some(dtype) = expected_dtype {
            if view.dtype() != dtype {
                return Err(Error::ModelLoadError(format!(
                    "Native tensor `{name}` dtype mismatch: found {:?}, expected {:?}",
                    view.dtype(),
                    dtype
                )));
            }
        }
        if let Some(shape) = expected_shape {
            if view.shape() != shape {
                return Err(Error::ModelLoadError(format!(
                    "Native tensor `{name}` shape mismatch: found {:?}, expected {:?}",
                    view.shape(),
                    shape
                )));
            }
        }
        consume(view)
    }

    pub fn load_block_fp8_f32(
        &self,
        weight_name: &str,
        expected_shape: [usize; 2],
        block_shape: [usize; 2],
    ) -> Result<Vec<f32>> {
        let scale_name = scale_name_for_weight(weight_name)?;
        let scale_shape = block_scale_shape(expected_shape, block_shape)?;
        let scales = self.with_tensor_view(
            &scale_name,
            Some(SafeDType::BF16),
            Some(&scale_shape),
            |view| decode_bf16_le(view.data(), &scale_name),
        )?;
        self.with_tensor_view(
            weight_name,
            Some(SafeDType::F8_E4M3),
            Some(&expected_shape),
            |view| {
                dequantize_e4m3fn_blockwise_f32(
                    view.data(),
                    expected_shape,
                    &scales,
                    scale_shape,
                    block_shape,
                )
                .map_err(|err| contextualize_weight_error(err, weight_name))
            },
        )
    }

    /// Materialize a projection into its persistent execution dtype.
    ///
    /// The raw shard mapping and scale buffer are gone when this function
    /// returns, so callers retain only the expanded projection.
    pub fn materialize_projection(
        &self,
        weight_name: &str,
        expected_shape: [usize; 2],
        block_shape: [usize; 2],
        target: ProjectionMaterialization,
        device: &Device,
    ) -> Result<Tensor> {
        let info = self.tensor_info(weight_name)?;
        let scale_name = scale_name_for_weight(weight_name)?;
        let values = match info.dtype {
            SafeDType::F8_E4M3 => {
                self.load_block_fp8_f32(weight_name, expected_shape, block_shape)?
            }
            SafeDType::BF16 | SafeDType::F16 | SafeDType::F32 => {
                if self.contains_tensor(&scale_name) {
                    return Err(Error::ModelLoadError(format!(
                        "Native dense tensor `{weight_name}` has an unexpected scale tensor `{scale_name}`"
                    )));
                }
                self.with_tensor_view(
                    weight_name,
                    Some(info.dtype),
                    Some(&expected_shape),
                    |view| decode_dense_f32(view, weight_name),
                )?
            }
            dtype => {
                return Err(Error::ModelLoadError(format!(
                    "Native projection `{weight_name}` uses unsupported dtype {dtype:?}"
                )));
            }
        };
        materialize_f32(values, &expected_shape, target, device).map_err(|err| {
            Error::ModelLoadError(format!(
                "Failed to materialize native projection `{weight_name}` as {target:?}: {err}"
            ))
        })
    }

    /// Materialize an ordinary BF16/F16/F32 tensor of any rank.
    ///
    /// This covers embeddings, normalization vectors, convolution kernels,
    /// DeltaNet parameters, and other native tensors which do not use the
    /// block-FP8 projection representation. A scale companion is rejected so
    /// callers cannot accidentally bypass FP8 dequantization.
    pub fn materialize_dense_tensor(
        &self,
        name: &str,
        expected_shape: &[usize],
        target: ProjectionMaterialization,
        device: &Device,
    ) -> Result<Tensor> {
        let info = self.tensor_info(name)?;
        if info.shape != expected_shape {
            return Err(Error::ModelLoadError(format!(
                "Native tensor `{name}` shape mismatch: found {:?}, expected {expected_shape:?}",
                info.shape
            )));
        }
        if name.ends_with(WEIGHT_SUFFIX) {
            let scale_name = scale_name_for_weight(name)?;
            if self.contains_tensor(&scale_name) {
                return Err(Error::ModelLoadError(format!(
                    "Native tensor `{name}` has block-FP8 scale companion `{scale_name}`; use materialize_projection"
                )));
            }
        }
        let values =
            self.with_tensor_view(name, Some(info.dtype), Some(expected_shape), |view| {
                decode_dense_f32(view, name)
            })?;
        materialize_f32(values, expected_shape, target, device).map_err(|err| {
            Error::ModelLoadError(format!(
                "Failed to materialize native tensor `{name}` as {target:?}: {err}"
            ))
        })
    }

    pub fn validate_required_text_tensor_names(&self, config: &Qwen35NativeConfig) -> Result<()> {
        let required = required_text_tensor_names(config);
        for name in &required {
            if !self.contains_tensor(name) {
                return Err(Error::ModelLoadError(format!(
                    "Native Qwen3.5 text checkpoint is missing required tensor `{name}`"
                )));
            }
        }
        for name in self.tensor_names() {
            if matches!(
                native_tensor_scope(name),
                NativeTensorScope::Text | NativeTensorScope::LmHead
            ) && name.ends_with(SCALE_SUFFIX)
                && !required.contains(name)
            {
                return Err(Error::ModelLoadError(format!(
                    "Native Qwen3.5 text checkpoint has unexpected scale tensor `{name}`"
                )));
            }
        }
        Ok(())
    }

    fn shard_path_for_tensor(&self, name: &str) -> Result<PathBuf> {
        let shard = self.weight_map.get(name).ok_or_else(|| {
            Error::ModelLoadError(format!("Tensor `{name}` is absent from {INDEX_FILE}"))
        })?;
        validate_shard_file(&self.model_dir, shard)
    }
}

/// Validated config and index for the native checkpoint. Vision and MTP names
/// remain indexed but are intentionally not required by the text-only slice.
#[derive(Debug, Clone)]
pub struct Qwen35NativeCheckpoint {
    pub config: Qwen35NativeConfig,
    pub tensors: IndexedSafetensors,
}

impl Qwen35NativeCheckpoint {
    pub fn open(model_dir: &Path) -> Result<Self> {
        let config = Qwen35NativeConfig::load(model_dir)?;
        let tensors = IndexedSafetensors::open(model_dir)?;
        tensors.validate_required_text_tensor_names(&config)?;
        Ok(Self { config, tensors })
    }
}

fn validate_relative_shard_name(name: &str) -> Result<()> {
    let path = Path::new(name);
    let valid = !name.trim().is_empty()
        && !path.is_absolute()
        && path.extension().and_then(|ext| ext.to_str()) == Some("safetensors")
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)));
    if !valid {
        return Err(Error::ModelLoadError(format!(
            "Safetensors index contains unsafe shard path `{name}`"
        )));
    }
    Ok(())
}

fn validate_shard_file(model_dir: &Path, name: &str) -> Result<PathBuf> {
    validate_relative_shard_name(name)?;
    let path = model_dir.join(name);
    let metadata = fs::symlink_metadata(&path).map_err(|err| {
        Error::ModelLoadError(format!(
            "Safetensors index references missing shard {}: {err}",
            path.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(Error::ModelLoadError(format!(
            "Safetensors shard {} must be a regular non-symlink file",
            path.display()
        )));
    }
    let canonical = path.canonicalize().map_err(|err| {
        Error::ModelLoadError(format!(
            "Failed to resolve Safetensors shard {}: {err}",
            path.display()
        ))
    })?;
    if !canonical.starts_with(model_dir) {
        return Err(Error::ModelLoadError(format!(
            "Safetensors shard {} resolves outside the checkpoint directory",
            path.display()
        )));
    }
    Ok(canonical)
}

fn required_text_tensor_names(config: &Qwen35NativeConfig) -> HashSet<String> {
    let mut names = HashSet::new();
    names.insert("model.language_model.embed_tokens.weight".to_string());
    names.insert("model.language_model.norm.weight".to_string());
    names.insert("lm_head.weight".to_string());
    for (index, layer_type) in config.layer_types.iter().enumerate() {
        let prefix = format!("model.language_model.layers.{index}");
        names.insert(format!("{prefix}.input_layernorm.weight"));
        names.insert(format!("{prefix}.post_attention_layernorm.weight"));
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            insert_projection_pair(&mut names, format!("{prefix}.mlp.{projection}.weight"));
        }
        match layer_type {
            Qwen35LayerType::LinearAttention => {
                for suffix in [
                    "linear_attn.A_log",
                    "linear_attn.dt_bias",
                    "linear_attn.conv1d.weight",
                    "linear_attn.in_proj_a.weight",
                    "linear_attn.in_proj_b.weight",
                    "linear_attn.norm.weight",
                ] {
                    names.insert(format!("{prefix}.{suffix}"));
                }
                for projection in ["in_proj_qkv", "in_proj_z", "out_proj"] {
                    insert_projection_pair(
                        &mut names,
                        format!("{prefix}.linear_attn.{projection}.weight"),
                    );
                }
            }
            Qwen35LayerType::FullAttention => {
                names.insert(format!("{prefix}.self_attn.q_norm.weight"));
                names.insert(format!("{prefix}.self_attn.k_norm.weight"));
                for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                    insert_projection_pair(
                        &mut names,
                        format!("{prefix}.self_attn.{projection}.weight"),
                    );
                }
            }
        }
    }
    names
}

fn insert_projection_pair(names: &mut HashSet<String>, weight: String) {
    let scale = scale_name_for_weight(&weight).expect("known projection weight name");
    names.insert(weight);
    names.insert(scale);
}

fn scale_name_for_weight(weight_name: &str) -> Result<String> {
    let stem = weight_name.strip_suffix(WEIGHT_SUFFIX).ok_or_else(|| {
        Error::ModelLoadError(format!(
            "Block-FP8 projection name `{weight_name}` does not end in `{WEIGHT_SUFFIX}`"
        ))
    })?;
    Ok(format!("{stem}{SCALE_SUFFIX}"))
}

fn contextualize_weight_error(error: Error, weight_name: &str) -> Error {
    Error::ModelLoadError(format!(
        "Failed to dequantize native tensor `{weight_name}`: {error}"
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionMaterialization {
    F32,
    F16,
    BF16,
}

impl ProjectionMaterialization {
    pub fn dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::BF16 => DType::BF16,
        }
    }
}

fn materialize_f32(
    values: Vec<f32>,
    shape: &[usize],
    target: ProjectionMaterialization,
    device: &Device,
) -> candle_core::Result<Tensor> {
    match target {
        ProjectionMaterialization::F32 => Tensor::from_vec(values, shape.to_vec(), device),
        ProjectionMaterialization::F16 => {
            let values = values.into_iter().map(f16::from_f32).collect::<Vec<_>>();
            Tensor::from_vec(values, shape.to_vec(), device)
        }
        ProjectionMaterialization::BF16 => {
            let values = values.into_iter().map(bf16::from_f32).collect::<Vec<_>>();
            Tensor::from_vec(values, shape.to_vec(), device)
        }
    }
}

fn decode_dense_f32(view: TensorView<'_>, name: &str) -> Result<Vec<f32>> {
    let values = match view.dtype() {
        SafeDType::BF16 => decode_bf16_le(view.data(), name)?,
        SafeDType::F16 => decode_f16_le(view.data(), name)?,
        SafeDType::F32 => decode_f32_le(view.data(), name)?,
        dtype => {
            return Err(Error::ModelLoadError(format!(
                "Native dense tensor `{name}` uses unsupported dtype {dtype:?}"
            )));
        }
    };
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(Error::ModelLoadError(format!(
            "Native dense tensor `{name}` contains non-finite value {value:?} at element {index}"
        )));
    }
    Ok(values)
}

fn decode_bf16_le(data: &[u8], name: &str) -> Result<Vec<f32>> {
    decode_u16_float_le(data, name, |bits| bf16::from_bits(bits).to_f32())
}

fn decode_f16_le(data: &[u8], name: &str) -> Result<Vec<f32>> {
    decode_u16_float_le(data, name, |bits| f16::from_bits(bits).to_f32())
}

fn decode_u16_float_le(data: &[u8], name: &str, decode: impl Fn(u16) -> f32) -> Result<Vec<f32>> {
    if !data.len().is_multiple_of(2) {
        return Err(Error::ModelLoadError(format!(
            "Native tensor `{name}` has an odd byte length {} for a 16-bit dtype",
            data.len()
        )));
    }
    Ok(data
        .chunks_exact(2)
        .map(|bytes| decode(u16::from_le_bytes([bytes[0], bytes[1]])))
        .collect())
}

fn decode_f32_le(data: &[u8], name: &str) -> Result<Vec<f32>> {
    if !data.len().is_multiple_of(4) {
        return Err(Error::ModelLoadError(format!(
            "Native tensor `{name}` byte length {} is not divisible by four",
            data.len()
        )));
    }
    Ok(data
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect())
}

fn block_scale_shape(weight_shape: [usize; 2], block_shape: [usize; 2]) -> Result<[usize; 2]> {
    if weight_shape.contains(&0) {
        return Err(Error::ModelLoadError(format!(
            "Block-FP8 weight has invalid zero dimension {weight_shape:?}"
        )));
    }
    if block_shape.contains(&0) {
        return Err(Error::ModelLoadError(format!(
            "Block-FP8 block shape has invalid zero dimension {block_shape:?}"
        )));
    }
    Ok([
        weight_shape[0].div_ceil(block_shape[0]),
        weight_shape[1].div_ceil(block_shape[1]),
    ])
}

/// Portable reference dequantization for row-major E4M3FN matrices.
///
/// Scales are row-major `[ceil(rows / block_rows), ceil(cols / block_cols)]`.
/// The implementation indexes scales per tile and never expands them to the
/// matrix shape.
pub fn dequantize_e4m3fn_blockwise_f32(
    weight: &[u8],
    weight_shape: [usize; 2],
    scales: &[f32],
    scale_shape: [usize; 2],
    block_shape: [usize; 2],
) -> Result<Vec<f32>> {
    let expected_scale_shape = block_scale_shape(weight_shape, block_shape)?;
    if scale_shape != expected_scale_shape {
        return Err(Error::ModelLoadError(format!(
            "Block-FP8 scale shape mismatch: found {scale_shape:?}, expected {expected_scale_shape:?} for weight {weight_shape:?} and blocks {block_shape:?}"
        )));
    }
    let weight_len = weight_shape[0]
        .checked_mul(weight_shape[1])
        .ok_or_else(|| Error::ModelLoadError("Block-FP8 weight element count overflow".into()))?;
    if weight.len() != weight_len {
        return Err(Error::ModelLoadError(format!(
            "Block-FP8 weight byte length mismatch: found {}, expected {weight_len}",
            weight.len()
        )));
    }
    let scale_len = scale_shape[0]
        .checked_mul(scale_shape[1])
        .ok_or_else(|| Error::ModelLoadError("Block-FP8 scale element count overflow".into()))?;
    if scales.len() != scale_len {
        return Err(Error::ModelLoadError(format!(
            "Block-FP8 scale length mismatch: found {}, expected {scale_len}",
            scales.len()
        )));
    }
    for (index, scale) in scales.iter().copied().enumerate() {
        if !scale.is_finite() || scale < 0.0 {
            return Err(Error::ModelLoadError(format!(
                "Block-FP8 inverse scale at element {index} is invalid: {scale:?}"
            )));
        }
    }

    let rows = weight_shape[0];
    let cols = weight_shape[1];
    let mut output = Vec::with_capacity(weight_len);
    for row in 0..rows {
        let scale_row = row / block_shape[0];
        for col in 0..cols {
            let scale_col = col / block_shape[1];
            let scale = scales[scale_row * scale_shape[1] + scale_col];
            let bits = weight[row * cols + col];
            let value = decode_e4m3fn(bits);
            if !value.is_finite() {
                return Err(Error::ModelLoadError(format!(
                    "Block-FP8 weight contains non-finite E4M3FN value 0x{bits:02x} at [{row}, {col}]"
                )));
            }
            let dequantized = value * scale;
            if !dequantized.is_finite() {
                return Err(Error::ModelLoadError(format!(
                    "Block-FP8 dequantization overflow at [{row}, {col}]: {value:?} * {scale:?}"
                )));
            }
            output.push(dequantized);
        }
    }
    Ok(output)
}

/// Decode NVIDIA/PyTorch `float8_e4m3fn` storage exactly into F32.
fn decode_e4m3fn(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 3) & 0x0f;
    let mantissa = bits & 0x07;
    if exponent == 0 {
        // Subnormal step is 2^(1-bias)/8 = 2^-9 for bias 7.
        sign * f32::from(mantissa) * (1.0 / 512.0)
    } else if exponent == 0x0f && mantissa == 0x07 {
        f32::NAN
    } else {
        sign * (1.0 + f32::from(mantissa) / 8.0) * 2.0_f32.powi(i32::from(exponent) - 7)
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use safetensors::tensor::TensorView;
    use serde_json::json;

    use super::*;

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(label: &str) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "izwi-qwen35-native-{label}-{}-{nonce}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn valid_config_json() -> Vec<u8> {
        let layer_types = (0..64)
            .map(|index| {
                if (index + 1) % 4 == 0 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect::<Vec<_>>();
        serde_json::to_vec(&json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "language_model_only": false,
            "model_type": "qwen3_5",
            "tie_word_embeddings": false,
            "text_config": {
                "attention_bias": false,
                "attention_dropout": 0.0,
                "attn_output_gate": true,
                "bos_token_id": 248044,
                "dtype": "bfloat16",
                "eos_token_id": 248044,
                "full_attention_interval": 4,
                "head_dim": 256,
                "hidden_act": "silu",
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "layer_types": layer_types,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "linear_value_head_dim": 128,
                "mamba_ssm_dtype": "float32",
                "max_position_embeddings": 262144,
                "model_type": "qwen3_5_text",
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": false,
                "num_attention_heads": 24,
                "num_hidden_layers": 64,
                "num_key_value_heads": 4,
                "output_gate_type": "swish",
                "partial_rotary_factor": 0.25,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {
                    "mrope_interleaved": true,
                    "mrope_section": [11, 11, 10],
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 10000000,
                    "rope_type": "default"
                },
                "tie_word_embeddings": false,
                "use_cache": true,
                "vocab_size": 248320
            },
            "quantization_config": {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "weight_block_size": [128, 128]
            }
        }))
        .unwrap()
    }

    fn write_index(dir: &Path, map: serde_json::Value) {
        fs::write(
            dir.join(INDEX_FILE),
            serde_json::to_vec(&json!({ "weight_map": map })).unwrap(),
        )
        .unwrap();
    }

    fn write_safetensors(path: &Path, tensors: &[(&str, SafeDType, Vec<usize>, &[u8])]) {
        let views = tensors
            .iter()
            .map(|(name, dtype, shape, data)| {
                (
                    (*name).to_string(),
                    TensorView::new(*dtype, shape.clone(), data).unwrap(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        safetensors::serialize_to_file(&views, &None, path).unwrap();
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| bf16::from_f32(*value).to_bits().to_le_bytes())
            .collect()
    }

    #[test]
    fn parses_and_maps_the_frozen_native_config() {
        let config = Qwen35NativeConfig::from_json(&valid_config_json()).unwrap();
        assert_eq!(config.text.block_count, 64);
        assert_eq!(config.text.embedding_length, 5_120);
        assert_eq!(config.text.ssm_group_count, 16);
        assert_eq!(config.text.ssm_time_step_rank, 48);
        assert_eq!(config.text.ssm_inner_size, 6_144);
        assert_eq!(config.text.rope_dimension_count, 64);
        assert_eq!(config.block_fp8.block_shape, [128, 128]);
        assert_eq!(
            config
                .layer_types
                .iter()
                .filter(|kind| **kind == Qwen35LayerType::FullAttention)
                .count(),
            16
        );
    }

    #[test]
    fn rejects_changed_layer_pattern_with_field_specific_error() {
        let mut value: serde_json::Value = serde_json::from_slice(&valid_config_json()).unwrap();
        value["text_config"]["layer_types"][0] = json!("full_attention");
        let error = Qwen35NativeConfig::from_json(&serde_json::to_vec(&value).unwrap())
            .unwrap_err()
            .to_string();
        assert!(error.contains("text_config.layer_types"), "{error}");
    }

    #[test]
    fn rejects_changed_quantization_contract() {
        let mut value: serde_json::Value = serde_json::from_slice(&valid_config_json()).unwrap();
        value["quantization_config"]["activation_scheme"] = json!("static");
        let error = Qwen35NativeConfig::from_json(&serde_json::to_vec(&value).unwrap())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("quantization_config.activation_scheme"),
            "{error}"
        );
    }

    #[test]
    fn dequantizes_multiple_128_blocks_without_expanding_scales() {
        let shape = [129, 257];
        let block = [128, 128];
        let scale_shape = [2, 3];
        let weight = vec![0x38; shape[0] * shape[1]]; // +1.0
        let scales = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let output =
            dequantize_e4m3fn_blockwise_f32(&weight, shape, &scales, scale_shape, block).unwrap();
        let at = |row: usize, col: usize| output[row * shape[1] + col];
        assert_eq!(at(0, 0), 1.0);
        assert_eq!(at(0, 128), 2.0);
        assert_eq!(at(0, 256), 3.0);
        assert_eq!(at(128, 0), 4.0);
        assert_eq!(at(128, 128), 5.0);
        assert_eq!(at(128, 256), 6.0);
    }

    #[test]
    fn decodes_e4m3fn_zero_subnormal_sign_and_max_finite() {
        let bits = [0x00, 0x80, 0x01, 0x81, 0x38, 0xb8, 0x7e, 0xfe];
        let output =
            dequantize_e4m3fn_blockwise_f32(&bits, [1, bits.len()], &[1.0], [1, 1], [128, 128])
                .unwrap();
        assert_eq!(output[0].to_bits(), 0.0_f32.to_bits());
        assert_eq!(output[1].to_bits(), (-0.0_f32).to_bits());
        assert_eq!(output[2], 1.0 / 512.0);
        assert_eq!(output[3], -1.0 / 512.0);
        assert_eq!(&output[4..], &[1.0, -1.0, 448.0, -448.0]);
    }

    #[test]
    fn rejects_e4m3fn_nan_and_nonfinite_scales() {
        let nan_error =
            dequantize_e4m3fn_blockwise_f32(&[0x7f], [1, 1], &[1.0], [1, 1], [128, 128])
                .unwrap_err()
                .to_string();
        assert!(nan_error.contains("non-finite E4M3FN"), "{nan_error}");

        let scale_error =
            dequantize_e4m3fn_blockwise_f32(&[0x38], [1, 1], &[f32::INFINITY], [1, 1], [128, 128])
                .unwrap_err()
                .to_string();
        assert!(scale_error.contains("inverse scale"), "{scale_error}");
    }

    #[test]
    fn rejects_wrong_scale_orientation_and_lengths() {
        let shape_error =
            dequantize_e4m3fn_blockwise_f32(&[0x38; 6], [2, 3], &[1.0, 2.0], [2, 1], [2, 2])
                .unwrap_err()
                .to_string();
        assert!(
            shape_error.contains("scale shape mismatch"),
            "{shape_error}"
        );

        let length_error =
            dequantize_e4m3fn_blockwise_f32(&[0x38; 6], [2, 3], &[1.0], [1, 2], [2, 2])
                .unwrap_err()
                .to_string();
        assert!(
            length_error.contains("scale length mismatch"),
            "{length_error}"
        );
    }

    #[test]
    fn indexed_fixture_dequantizes_bf16_scales_and_materializes_f32() {
        let dir = TestDir::new("fixture");
        let weight_name = "model.language_model.layers.0.mlp.gate_proj.weight";
        let scale_name = "model.language_model.layers.0.mlp.gate_proj.weight_scale_inv";
        let weights = [0x38, 0x40, 0xb8, 0xc0]; // 1, 2, -1, -2
        let scales = bf16_bytes(&[0.5]);
        write_safetensors(
            &dir.path().join("layers-0.safetensors"),
            &[
                (weight_name, SafeDType::F8_E4M3, vec![2, 2], &weights),
                (scale_name, SafeDType::BF16, vec![1, 1], &scales),
            ],
        );
        write_index(
            dir.path(),
            json!({
                "model.language_model.layers.0.mlp.gate_proj.weight": "layers-0.safetensors",
                "model.language_model.layers.0.mlp.gate_proj.weight_scale_inv": "layers-0.safetensors"
            }),
        );

        let source = IndexedSafetensors::open(dir.path()).unwrap();
        let tensor = source
            .materialize_projection(
                weight_name,
                [2, 2],
                [128, 128],
                ProjectionMaterialization::F32,
                &Device::Cpu,
            )
            .unwrap();
        assert_eq!(
            tensor.to_vec2::<f32>().unwrap(),
            vec![vec![0.5, 1.0], vec![-0.5, -1.0]]
        );
    }

    #[test]
    fn indexed_loader_opens_only_the_requested_shard() {
        let dir = TestDir::new("one-shard");
        let good = [0.0_f32.to_le_bytes(), 1.0_f32.to_le_bytes()].concat();
        write_safetensors(
            &dir.path().join("outside.safetensors"),
            &[("lm_head.weight", SafeDType::F32, vec![1, 2], &good)],
        );
        fs::write(dir.path().join("layers-0.safetensors"), b"not safetensors").unwrap();
        write_index(
            dir.path(),
            json!({
                "lm_head.weight": "outside.safetensors",
                "model.language_model.layers.0.mlp.gate_proj.weight": "layers-0.safetensors"
            }),
        );

        let source = IndexedSafetensors::open(dir.path()).unwrap();
        let tensor = source
            .materialize_projection(
                "lm_head.weight",
                [1, 2],
                [128, 128],
                ProjectionMaterialization::F32,
                &Device::Cpu,
            )
            .unwrap();
        assert_eq!(tensor.to_vec2::<f32>().unwrap(), vec![vec![0.0, 1.0]]);
    }

    #[test]
    fn materializes_dense_vectors_and_higher_rank_tensors() {
        let dir = TestDir::new("dense-tensors");
        let vector = bf16_bytes(&[1.0, -2.0, 3.5]);
        let kernel = bf16_bytes(&[1.0, 2.0, 3.0, 4.0]);
        write_safetensors(
            &dir.path().join("layers-0.safetensors"),
            &[
                (
                    "model.language_model.layers.0.linear_attn.A_log",
                    SafeDType::BF16,
                    vec![3],
                    &vector,
                ),
                (
                    "model.language_model.layers.0.linear_attn.conv1d.weight",
                    SafeDType::BF16,
                    vec![1, 1, 4],
                    &kernel,
                ),
            ],
        );
        write_index(
            dir.path(),
            json!({
                "model.language_model.layers.0.linear_attn.A_log": "layers-0.safetensors",
                "model.language_model.layers.0.linear_attn.conv1d.weight": "layers-0.safetensors"
            }),
        );

        let source = IndexedSafetensors::open(dir.path()).unwrap();
        let vector = source
            .materialize_dense_tensor(
                "model.language_model.layers.0.linear_attn.A_log",
                &[3],
                ProjectionMaterialization::F32,
                &Device::Cpu,
            )
            .unwrap();
        assert_eq!(vector.to_vec1::<f32>().unwrap(), vec![1.0, -2.0, 3.5]);
        let kernel = source
            .materialize_dense_tensor(
                "model.language_model.layers.0.linear_attn.conv1d.weight",
                &[1, 1, 4],
                ProjectionMaterialization::F32,
                &Device::Cpu,
            )
            .unwrap();
        assert_eq!(
            kernel.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn indexed_loader_rejects_unsafe_and_missing_shards() {
        let unsafe_dir = TestDir::new("unsafe-index");
        write_index(unsafe_dir.path(), json!({ "x": "../escape.safetensors" }));
        let error = IndexedSafetensors::open(unsafe_dir.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("unsafe shard path"), "{error}");

        let missing_dir = TestDir::new("missing-index");
        write_index(missing_dir.path(), json!({ "x": "missing.safetensors" }));
        let error = IndexedSafetensors::open(missing_dir.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("missing shard"), "{error}");
    }

    #[test]
    fn indexed_loader_rejects_missing_scale_wrong_dtype_and_shape() {
        let dir = TestDir::new("bad-scale");
        let weight_name = "model.language_model.layers.0.mlp.gate_proj.weight";
        let scale_name = "model.language_model.layers.0.mlp.gate_proj.weight_scale_inv";
        let weights = [0x38; 4];
        let wrong_scale = [1.0_f32.to_le_bytes()].concat();
        write_safetensors(
            &dir.path().join("layers-0.safetensors"),
            &[
                (weight_name, SafeDType::F8_E4M3, vec![2, 2], &weights),
                (scale_name, SafeDType::F32, vec![1, 1], &wrong_scale),
            ],
        );
        write_index(
            dir.path(),
            json!({
                "model.language_model.layers.0.mlp.gate_proj.weight": "layers-0.safetensors",
                "model.language_model.layers.0.mlp.gate_proj.weight_scale_inv": "layers-0.safetensors"
            }),
        );
        let source = IndexedSafetensors::open(dir.path()).unwrap();
        let error = source
            .load_block_fp8_f32(weight_name, [2, 2], [128, 128])
            .unwrap_err()
            .to_string();
        assert!(error.contains("dtype mismatch"), "{error}");

        let missing_dir = TestDir::new("missing-scale");
        write_safetensors(
            &missing_dir.path().join("layers-0.safetensors"),
            &[(weight_name, SafeDType::F8_E4M3, vec![2, 2], &weights)],
        );
        write_index(
            missing_dir.path(),
            json!({
                "model.language_model.layers.0.mlp.gate_proj.weight": "layers-0.safetensors"
            }),
        );
        let source = IndexedSafetensors::open(missing_dir.path()).unwrap();
        let error = source
            .load_block_fp8_f32(weight_name, [2, 2], [128, 128])
            .unwrap_err()
            .to_string();
        assert!(error.contains("weight_scale_inv"), "{error}");
    }

    #[test]
    fn tensor_scope_keeps_vision_and_mtp_explicitly_separate() {
        assert_eq!(
            native_tensor_scope("model.language_model.layers.0.input_layernorm.weight"),
            NativeTensorScope::Text
        );
        assert_eq!(
            native_tensor_scope("lm_head.weight"),
            NativeTensorScope::LmHead
        );
        assert_eq!(
            native_tensor_scope("model.visual.blocks.0.attn.qkv.weight"),
            NativeTensorScope::Vision
        );
        assert_eq!(native_tensor_scope("mtp.fc.weight"), NativeTensorScope::Mtp);
    }
}
