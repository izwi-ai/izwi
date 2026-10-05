//! Native block-FP8 checkpoint → Qwen3.5 hybrid trunk construction.
//!
//! Bridges [`Qwen35MoeNativeCheckpoint`] (config contract, tensor plan, and
//! per-backend FP8 materialization from Phase 1) into the shared
//! `Qwen35TextModel` trunk: the GGUF-style logical tensor names the trunk
//! requests are translated to the canonical HF layout
//! (`model.layers.{i}...`), and every projection materializes in the
//! backend's persistent residency (CPU packed Q8_0, Metal expanded F16,
//! CUDA expanded BF16).

use std::sync::Arc;

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Tensor};
use candle_transformers::quantized_nn::RmsNorm;

use crate::backends::{BackendKind, DeviceProfile};
use crate::error::{Error, Result};
use crate::models::architectures::qwen35::text::{
    Qwen35MoeFfnGeometry, Qwen35Projection, Qwen35TextModel, Qwen35WeightSource,
};
use crate::models::architectures::qwen38::native::ProjectionMaterialization;
use crate::models::architectures::qwen35moe::native::{
    ExpectedTensorKind, Qwen35MoeNativeCheckpoint, Qwen35MoeProjection,
    Qwen35MoeProjectionResidency,
};
use crate::models::architectures::qwen35moe::sparse::{
    Qwen35MoeExpertWeights, Qwen35MoeLinear, Qwen35MoeSharedExpertWeights, Qwen35MoeSparseMlp,
};

/// Map a logical GGUF-style trunk tensor name to its canonical checkpoint
/// name. `None` when the logical name has no native-counterpart mapping.
fn canonical_name(logical: &str) -> Option<String> {
    let mut parts = logical.split('.');
    let first = parts.next()?;
    if first == "token_embd" {
        return Some("model.embed_tokens.weight".into());
    }
    if first == "output" {
        return Some("lm_head.weight".into());
    }
    if first == "output_norm" {
        return Some("model.norm.weight".into());
    }
    if first != "blk" {
        return None;
    }
    let layer: usize = parts.next()?.parse().ok()?;
    let rest: Vec<&str> = parts.collect();
    let suffix = rest.join(".");
    let canonical_suffix = match suffix.as_str() {
        "attn_norm.weight" => "input_layernorm.weight".to_string(),
        "post_attention_norm.weight" => "post_attention_layernorm.weight".to_string(),
        "attn_q.weight" => "self_attn.q_proj.weight".to_string(),
        "attn_k.weight" => "self_attn.k_proj.weight".to_string(),
        "attn_v.weight" => "self_attn.v_proj.weight".to_string(),
        "attn_output.weight" => "self_attn.o_proj.weight".to_string(),
        "attn_q_norm.weight" => "self_attn.q_norm.weight".to_string(),
        "attn_k_norm.weight" => "self_attn.k_norm.weight".to_string(),
        "attn_qkv.weight" => "linear_attn.in_proj_qkv.weight".to_string(),
        "attn_gate.weight" => "linear_attn.in_proj_z.weight".to_string(),
        "ssm_beta.weight" => "linear_attn.in_proj_b.weight".to_string(),
        "ssm_alpha.weight" => "linear_attn.in_proj_a.weight".to_string(),
        "ssm_dt.bias" | "ssm_dt" => "linear_attn.dt_bias".to_string(),
        "ssm_a" => "linear_attn.A_log".to_string(),
        "ssm_conv1d.weight" => "linear_attn.conv1d.weight".to_string(),
        "ssm_norm.weight" => "linear_attn.norm.weight".to_string(),
        "ssm_out.weight" => "linear_attn.out_proj.weight".to_string(),
        other => return Some(format!("model.layers.{layer}.{other}")),
    };
    Some(format!("model.layers.{layer}.{canonical_suffix}"))
}

/// Native-checkpoint-backed [`Qwen35WeightSource`].
pub(crate) struct Qwen35MoeNativeSource<'a> {
    checkpoint: &'a Qwen35MoeNativeCheckpoint,
    residency: Qwen35MoeProjectionResidency,
    dense_target: ProjectionMaterialization,
}

impl<'a> Qwen35MoeNativeSource<'a> {
    pub(crate) fn new(
        checkpoint: &'a Qwen35MoeNativeCheckpoint,
        device_profile: &DeviceProfile,
    ) -> Self {
        let residency =
            Qwen35MoeNativeCheckpoint::projection_residency_policy(device_profile);
        let dense_target = match BackendKind::from(device_profile.kind) {
            BackendKind::Cpu => ProjectionMaterialization::F32,
            BackendKind::Metal => ProjectionMaterialization::F16,
            BackendKind::Cuda => ProjectionMaterialization::BF16,
        };
        Self {
            checkpoint,
            residency,
            dense_target,
        }
    }

    fn resolve(&self, logical: &str) -> Result<(String, Vec<usize>, ExpectedTensorKind)> {
        let canonical = canonical_name(logical).ok_or_else(|| {
            Error::ModelLoadError(format!(
                "qwen35moe native checkpoint has no mapping for trunk tensor `{logical}`"
            ))
        })?;
        let raw = self.checkpoint.raw_tensor_name(&canonical)?;
        let info = self.checkpoint.tensors.tensor_info(raw)?;
        Ok((canonical, info.shape.clone(), tensor_kind(&info.dtype)))
    }

    fn materialize_dense_weight(
        &self,
        logical: &str,
        device: &Device,
    ) -> Result<Tensor> {
        let (canonical, shape, _kind) = self.resolve(logical)?;
        let tensor = self.checkpoint.materialize_dense(
            &canonical,
            &shape,
            device,
            self.dense_target,
        )?;
        if canonical.ends_with(".A_log") {
            // The published checkpoint stores the DeltaNet decay as
            // `A_log`; the trunk's recurrence consumes `a` directly in
            // `g = softplus(alpha + dt_bias) * a`, i.e. `a = -exp(A_log)`
            // (the same form the llama.cpp conversion emits as `ssm_a`).
            let values = tensor
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let transformed: Vec<f32> = values.iter().map(|v| -v.exp()).collect();
            return Tensor::from_vec(transformed, tensor.dims(), &device.clone())
                .map_err(Error::from);
        }
        Ok(tensor)
    }

    fn wrap_dense_projection(
        tensor: Tensor,
        residency: Qwen35MoeProjectionResidency,
    ) -> Result<Qwen35Projection> {
        let ggml_dtype = match residency {
            Qwen35MoeProjectionResidency::PackedQ8_0 => GgmlDType::F32,
            Qwen35MoeProjectionResidency::ExpandedF16 => GgmlDType::F16,
            Qwen35MoeProjectionResidency::ExpandedBf16 | Qwen35MoeProjectionResidency::NativeFp8WithQ8Fallback => {
                GgmlDType::BF16
            }
            Qwen35MoeProjectionResidency::ExpandedF32 => GgmlDType::F32,
        };
        let quantized = QTensor::quantize(&tensor, ggml_dtype).map_err(Error::from)?;
        Ok(Qwen35Projection::Quantized(
            QMatMul::from_arc(Arc::new(quantized)).map_err(Error::from)?,
        ))
    }
}

fn tensor_kind(dtype: &safetensors::Dtype) -> ExpectedTensorKind {
    match dtype {
        safetensors::Dtype::F8_E4M3 => ExpectedTensorKind::BlockFp8,
        safetensors::Dtype::BF16 => ExpectedTensorKind::BlockFp8Scale,
        _ => ExpectedTensorKind::Dense,
    }
}

impl Qwen35WeightSource for Qwen35MoeNativeSource<'_> {
    fn has(&self, name: &str) -> bool {
        canonical_name(name)
            .and_then(|canonical| self.checkpoint.raw_tensor_name(&canonical).ok())
            .is_some()
    }

    fn projection(&self, name: &str, device: &Device) -> Result<Qwen35Projection> {
        let (canonical, shape, kind) = self.resolve(name)?;
        let expected: [usize; 2] = match shape.as_slice() {
            [rows, cols] => [*rows, *cols],
            other => {
                return Err(Error::ModelLoadError(format!(
                    "qwen35moe native projection `{canonical}` has rank-{} shape {other:?}; expected a 2-D projection",
                    other.len()
                )))
            }
        };
        // FP8-excluded projections (lm_head, and anything dense in the
        // published contract) ride the dense materialization path; only
        // F8_E4M3 storage goes through the block-FP8 decode/requant.
        if kind != ExpectedTensorKind::BlockFp8 {
            let tensor = self.checkpoint.materialize_dense(
                &canonical,
                &shape,
                device,
                self.dense_target,
            )?;
            return Self::wrap_dense_projection(tensor, self.residency);
        }
        match self.checkpoint.materialize_projection(
            &canonical,
            expected,
            device,
            self.residency,
        )? {
            Qwen35MoeProjection::Packed(qmatmul) => Ok(Qwen35Projection::Quantized(qmatmul)),
            Qwen35MoeProjection::Dense(tensor) => {
                Self::wrap_dense_projection(tensor, self.residency)
            }
            Qwen35MoeProjection::CompactFp8(raw) => Ok(Qwen35Projection::CompactFp8 {
                weights: raw.weights,
                scales: raw.scales,
            }),
        }
    }

    fn rms_norm(&self, name: &str, eps: f64, device: &Device) -> Result<RmsNorm> {
        let tensor = self.materialize_dense_weight(name, device)?;
        let quantized = QTensor::quantize(&tensor, GgmlDType::F32).map_err(Error::from)?;
        RmsNorm::from_qtensor(quantized, eps).map_err(Error::from)
    }

    fn dense(&self, name: &str, _dtype: Option<DType>, device: &Device) -> Result<Tensor> {
        // Native dense math tensors (norms, DeltaNet in-proj/conv, dt_bias,
        // A_log) always materialize through the per-backend dense target.
        self.materialize_dense_weight(name, device)
    }

    fn moe_ffn(
        &self,
        layer: usize,
        geometry: &Qwen35MoeFfnGeometry,
        device: &Device,
    ) -> Result<Qwen35MoeSparseMlp> {
        let prefix = format!("model.layers.{layer}.mlp");

        // Router: `mlp.gate.weight` is FP8-excluded (BF16 dense) in the
        // published quantization contract, so it rides the dense path.
        let router_canonical = format!("{prefix}.gate.weight");
        let router_raw = self.checkpoint.raw_tensor_name(&router_canonical)?;
        let router_shape = self
            .checkpoint
            .tensors
            .tensor_info(router_raw)?
            .shape
            .clone();
        let router = Qwen35MoeLinear::from_dense(self.checkpoint.materialize_dense(
            &router_canonical,
            &router_shape,
            device,
            ProjectionMaterialization::F32,
        )?);

        let mut experts = Vec::with_capacity(geometry.num_experts);
        for expert in 0..geometry.num_experts {
            experts.push(Qwen35MoeExpertWeights {
                gate: self.materialize_expert_projection(
                    &format!("{prefix}.experts.{expert}.gate_proj.weight"),
                    device,
                )?,
                up: self.materialize_expert_projection(
                    &format!("{prefix}.experts.{expert}.up_proj.weight"),
                    device,
                )?,
                down: self.materialize_expert_projection(
                    &format!("{prefix}.experts.{expert}.down_proj.weight"),
                    device,
                )?,
            });
        }

        let shared = Qwen35MoeSharedExpertWeights {
            gate: self.materialize_expert_projection(&format!("{prefix}.shared_expert.gate_proj.weight"), device)?,
            up: self.materialize_expert_projection(&format!("{prefix}.shared_expert.up_proj.weight"), device)?,
            down: self.materialize_expert_projection(&format!("{prefix}.shared_expert.down_proj.weight"), device)?,
            output_gate: self
                .checkpoint
                .raw_tensor_name(&format!("{prefix}.shared_expert_gate.weight"))
                .is_ok()
                .then(|| -> Result<Qwen35MoeLinear> {
                    let gate_canonical = format!("{prefix}.shared_expert_gate.weight");
                    let raw = self.checkpoint.raw_tensor_name(&gate_canonical)?;
                    let shape = self.checkpoint.tensors.tensor_info(raw)?.shape.clone();
                    Ok(Qwen35MoeLinear::from_dense(
                        self.checkpoint.materialize_dense(
                            &gate_canonical,
                            &shape,
                            device,
                            ProjectionMaterialization::F32,
                        )?,
                    ))
                })
                .transpose()?,
        };

        Qwen35MoeSparseMlp::from_weights(router, experts, shared, geometry)
    }

    fn token_embeddings(&self, device: &Device) -> Result<Tensor> {
        let (canonical, shape, _kind) = self.resolve("token_embd.weight")?;
        self.checkpoint
            .materialize_dense(&canonical, &shape, device, self.dense_target)
    }
}

impl Qwen35MoeNativeSource<'_> {
    fn materialize_expert_projection(
        &self,
        canonical: &str,
        device: &Device,
    ) -> Result<Qwen35MoeLinear> {
        let raw = self.checkpoint.raw_tensor_name(canonical)?;
        let shape = self.checkpoint.tensors.tensor_info(raw)?.shape.clone();
        let expected: [usize; 2] = match shape.as_slice() {
            [rows, cols] => [*rows, *cols],
            other => {
                return Err(Error::ModelLoadError(format!(
                    "qwen35moe native expert projection `{canonical}` has shape {other:?}; expected a 2-D projection"
                )))
            }
        };
        match self
            .checkpoint
            .materialize_projection(canonical, expected, device, self.residency)?
        {
            Qwen35MoeProjection::Packed(qmatmul) => Ok(Qwen35MoeLinear::Quantized(qmatmul)),
            Qwen35MoeProjection::Dense(tensor) => Ok(Qwen35MoeLinear::from_dense(tensor)),
            Qwen35MoeProjection::CompactFp8(raw) => Ok(Qwen35MoeLinear::CompactFp8 {
                weights: raw.weights,
                scales: raw.scales,
            }),
        }
    }
}

/// Map the validated native text geometry onto the shared trunk config.
pub(crate) fn qwen35_text_config_from_native(
    native: &crate::models::architectures::qwen35moe::native::Qwen35MoeTextConfig,
) -> crate::models::architectures::qwen35::chat::Qwen35TextConfig {
    let inner_size = native.ssm_time_step_rank * native.ssm_value_head_dim;
    crate::models::architectures::qwen35::chat::Qwen35TextConfig {
        architecture: "qwen35moe".to_string(),
        block_count: native.block_count,
        context_length: native.context_tokens,
        embedding_length: native.hidden_size,
        feed_forward_length: native.moe_intermediate_size,
        attention_head_count: native.attention_head_count,
        attention_head_count_kv: native.attention_head_count_kv,
        attention_key_length: native.attention_key_length,
        attention_value_length: native.attention_value_length,
        rope_dimension_sections: native.mrope_sections.to_vec(),
        rope_dimension_count: native.rope_dimension_count,
        rope_freq_base: native.rope_theta,
        attention_layer_norm_rms_epsilon: native.rms_norm_eps,
        ssm_conv_kernel: native.ssm_conv_kernel,
        ssm_state_size: native.ssm_state_size,
        ssm_group_count: native.ssm_group_count,
        ssm_time_step_rank: native.ssm_time_step_rank,
        ssm_inner_size: inner_size,
        full_attention_interval: native.full_attention_interval,
        moe_ffn: Some(Qwen35MoeFfnGeometry {
            num_experts: native.moe_num_experts,
            num_experts_per_tok: native.moe_num_experts_per_tok,
            expert_intermediate_size: native.moe_intermediate_size,
            shared_expert_intermediate_size: native.shared_expert_intermediate_size,
        }),
    }
}

/// Build the shared hybrid trunk from an opened native FP8 checkpoint.
pub(crate) fn load_text_model_native(
    checkpoint: &Qwen35MoeNativeCheckpoint,
    device_profile: &DeviceProfile,
    device: &Device,
) -> Result<(crate::models::architectures::qwen35::chat::Qwen35TextConfig, Qwen35TextModel)> {
    let text_config = qwen35_text_config_from_native(&checkpoint.config.text);
    let source = Qwen35MoeNativeSource::new(checkpoint, device_profile);
    let model = Qwen35TextModel::load_with_source(&source, &text_config, device)?;
    Ok((text_config, model))
}
