//! Native block-FP8 checkpoint → Qwen3.5 hybrid trunk construction.
//!
//! Bridges [`Qwen36MoeNativeCheckpoint`] (config contract, tensor plan, and
//! per-backend FP8 materialization from Phase 1) into the shared
//! `Qwen35TextModel` trunk: the GGUF-style logical tensor names the trunk
//! requests are translated to the canonical HF layout
//! (`model.layers.{i}...`), and every projection materializes in the
//! backend's persistent residency (CPU packed Q8_0, Metal expanded F16,
//! CUDA raw block-FP8 with per-tensor Q8_0 fallback).

use candle_core::quantized::QMatMul;

use candle_core::{DType, Device, Tensor};

use crate::backends::{BackendKind, DeviceProfile};
use crate::error::{Error, Result};
use crate::models::architectures::qwen35::mtp::Qwen35MtpHead;
use crate::models::architectures::qwen35::text::{
    Qwen35MoeFfnGeometry, Qwen35Projection, Qwen35RmsNorm, Qwen35TextModel, Qwen35WeightSource,
};
use crate::models::architectures::qwen38::native::ProjectionMaterialization;
use crate::models::architectures::qwen36moe::native::{
    ExpectedTensorKind, Qwen36MoeNativeCheckpoint, Qwen36MoeProjection,
    Qwen36MoeProjectionResidency,
};
use crate::models::architectures::qwen36moe::sparse::{
    Qwen36MoeExpertWeights, Qwen36MoeLinear, Qwen36MoeSharedExpertWeights, Qwen36MoeSparseMlp,
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
    // The MTP draft head reuses the trunk's own loaders through a virtual
    // block prefix: `mtpblk.{layer}.{logical}` resolves to the checkpoint's
    // `mtp.layers.{layer}.*` (or layer-free `mtp.*`) names. Same suffix
    // conventions as the text mapping so Qwen35FullAttention/Qwen35Mlp load
    // the draft layer exactly like a trunk layer.
    if first == "mtpblk" {
        let layer: usize = parts.next()?.parse().ok()?;
        let suffix = parts.collect::<Vec<_>>().join(".");
        let canonical_suffix = match suffix.as_str() {
            "attn_q.weight" => "self_attn.q_proj.weight".to_string(),
            "attn_k.weight" => "self_attn.k_proj.weight".to_string(),
            "attn_v.weight" => "self_attn.v_proj.weight".to_string(),
            "attn_output.weight" => "self_attn.o_proj.weight".to_string(),
            "attn_q_norm.weight" => "self_attn.q_norm.weight".to_string(),
            "attn_k_norm.weight" => "self_attn.k_norm.weight".to_string(),
            "attn_norm.weight" => "input_layernorm.weight".to_string(),
            "post_attention_norm.weight" => "post_attention_layernorm.weight".to_string(),
            "ffn_gate.weight" => "mlp.gate_proj.weight".to_string(),
            "ffn_up.weight" => "mlp.up_proj.weight".to_string(),
            "ffn_down.weight" => "mlp.down_proj.weight".to_string(),
            "mtp_fc.weight" => "fc.weight".to_string(),
            "mtp_norm.weight" => "norm.weight".to_string(),
            "mtp_pre_fc_norm_embedding.weight" => "pre_fc_norm_embedding.weight".to_string(),
            "mtp_pre_fc_norm_hidden.weight" => "pre_fc_norm_hidden.weight".to_string(),
            other => return Some(format!("mtp.layers.{layer}.{other}")),
        };
        // Layer-free tensors (fc, norms) live directly under `mtp.`.
        if matches!(
            suffix.as_str(),
            "mtp_fc.weight" | "mtp_norm.weight" | "mtp_pre_fc_norm_embedding.weight" | "mtp_pre_fc_norm_hidden.weight"
        ) {
            return Some(format!("mtp.{canonical_suffix}"));
        }
        return Some(format!("mtp.layers.{layer}.{canonical_suffix}"));
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
pub(crate) struct Qwen36MoeNativeSource<'a> {
    checkpoint: &'a Qwen36MoeNativeCheckpoint,
    residency: Qwen36MoeProjectionResidency,
    dense_target: ProjectionMaterialization,
}

impl<'a> Qwen36MoeNativeSource<'a> {
    /// Test seam: build the source under an explicit residency/dense plan so
    /// a backend's dtype plan (CUDA, Metal) can be executed end-to-end on the
    /// CPU device, where candle's dtype checks are identical to — and louder
    /// than — the accelerator backends'. The production constructor derives
    /// both fields from the device profile instead.
    #[cfg(test)]
    pub(crate) fn for_plan_tests(
        checkpoint: &'a Qwen36MoeNativeCheckpoint,
        residency: Qwen36MoeProjectionResidency,
        dense_target: ProjectionMaterialization,
    ) -> Self {
        Self {
            checkpoint,
            residency,
            dense_target,
        }
    }

    pub(crate) fn new(
        checkpoint: &'a Qwen36MoeNativeCheckpoint,
        device_profile: &DeviceProfile,
    ) -> Self {
        Self::new_with_performance(
            checkpoint,
            device_profile,
            &crate::performance::CudaPerformanceConfig::default(),
        )
    }

    pub(crate) fn new_with_performance(
        checkpoint: &'a Qwen36MoeNativeCheckpoint,
        device_profile: &DeviceProfile,
        performance: &crate::performance::CudaPerformanceConfig,
    ) -> Self {
        let residency = Qwen36MoeNativeCheckpoint::projection_residency_policy_with_performance(
            BackendKind::from(device_profile.kind),
            performance,
        );
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
                "qwen36moe native checkpoint has no mapping for trunk tensor `{logical}`"
            ))
        })?;
        if canonical.starts_with("mtp.") {
            // MTP tensors resolve identity-named against the raw index; the
            // manifest supplies the expected tensor kind.
            let info = self.checkpoint.tensors.tensor_info(&canonical)?;
            let kind = self
                .checkpoint
                .mtp_plan
                .iter()
                .find(|spec| spec.name == canonical)
                .map(|spec| spec.kind)
                .unwrap_or(ExpectedTensorKind::Dense);
            return Ok((canonical, info.shape.clone(), kind));
        }
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

    fn wrap_dense_projection(tensor: Tensor) -> Result<Qwen35Projection> {
        // The materialized tensor already carries the plan's dtype (F32 CPU,
        // F16 Metal, BF16 CUDA). A quantized-nn-style QTensor round-trip
        // would silently upcast it: candle's QTensor::dequantize is F32-only
        // for every GGML dtype, so the BF16/F16 residencies turned into F32
        // weights and broke the non-F32 activation graphs at the first
        // matmul. Keep the tensor directly; QMatMul's Tensor branch matmuls
        // in the weight's dtype, which matches the plan's activations.
        Ok(Qwen35Projection::Quantized(QMatMul::Tensor(tensor)))
    }
}

fn tensor_kind(dtype: &safetensors::Dtype) -> ExpectedTensorKind {
    match dtype {
        safetensors::Dtype::F8_E4M3 => ExpectedTensorKind::BlockFp8,
        safetensors::Dtype::BF16 => ExpectedTensorKind::BlockFp8Scale,
        _ => ExpectedTensorKind::Dense,
    }
}

impl Qwen35WeightSource for Qwen36MoeNativeSource<'_> {
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
                    "qwen36moe native projection `{canonical}` has rank-{} shape {other:?}; expected a 2-D projection",
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
            return Self::wrap_dense_projection(tensor);
        }
        match self.checkpoint.materialize_projection(
            &canonical,
            expected,
            device,
            self.residency,
        )? {
            Qwen36MoeProjection::Packed(qmatmul) => Ok(Qwen35Projection::Quantized(qmatmul)),
            Qwen36MoeProjection::Dense(tensor) => Self::wrap_dense_projection(tensor),
            Qwen36MoeProjection::CompactFp8(raw) => Ok(Qwen35Projection::CompactFp8 {
                weights: raw.weights,
                scales: raw.scales,
            }),
        }
    }

    fn rms_norm(&self, name: &str, eps: f64, device: &Device) -> Result<Qwen35RmsNorm> {
        // Candle's rmsnorm op requires x and weight in the same dtype on
        // every backend, and the trunk's activations carry the dense
        // target's dtype (BF16 CUDA, F16 Metal, F32 CPU). quantized_nn's
        // RmsNorm always dequantizes its weight to F32, so a BF16/F16
        // activation plan would die at the first norm — on CUDA through
        // Map2's "dtype mismatch in binary op". Keep the materialized
        // activation-dtype tensor instead; the checkpoint stores these
        // weights in that same dtype, so nothing is requantized.
        Ok(Qwen35RmsNorm::new(
            self.materialize_dense_weight(name, device)?,
            eps,
        ))
    }

    fn dense(&self, name: &str, dtype: Option<DType>, device: &Device) -> Result<Tensor> {
        // Native dense math tensors (DeltaNet dt_bias/conv/A_log, ssm norm)
        // materialize through the per-backend dense target, then honor the
        // trunk's requested dtype: the DeltaNet math is pinned to F32 so it
        // matches the F32 state arena under every residency plan (the
        // per-backend targets otherwise hand back BF16 on CUDA and F16 on
        // Metal).
        let tensor = self.materialize_dense_weight(name, device)?;
        match dtype {
            Some(target) if tensor.dtype() != target => {
                tensor.to_dtype(target).map_err(Error::from)
            }
            _ => Ok(tensor),
        }
    }

    fn moe_ffn(
        &self,
        layer: usize,
        geometry: &Qwen35MoeFfnGeometry,
        device: &Device,
    ) -> Result<Qwen36MoeSparseMlp> {
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
        let router = Qwen36MoeLinear::from_dense(self.checkpoint.materialize_dense(
            &router_canonical,
            &router_shape,
            device,
            ProjectionMaterialization::F32,
        )?);

        let mut experts = Vec::with_capacity(geometry.num_experts);
        for expert in 0..geometry.num_experts {
            experts.push(Qwen36MoeExpertWeights {
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

        let shared = Qwen36MoeSharedExpertWeights {
            gate: self.materialize_expert_projection(&format!("{prefix}.shared_expert.gate_proj.weight"), device)?,
            up: self.materialize_expert_projection(&format!("{prefix}.shared_expert.up_proj.weight"), device)?,
            down: self.materialize_expert_projection(&format!("{prefix}.shared_expert.down_proj.weight"), device)?,
            output_gate: self
                .checkpoint
                .raw_tensor_name(&format!("{prefix}.shared_expert_gate.weight"))
                .is_ok()
                .then(|| -> Result<Qwen36MoeLinear> {
                    let gate_canonical = format!("{prefix}.shared_expert_gate.weight");
                    let raw = self.checkpoint.raw_tensor_name(&gate_canonical)?;
                    let shape = self.checkpoint.tensors.tensor_info(raw)?.shape.clone();
                    Ok(Qwen36MoeLinear::from_dense(
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

        Qwen36MoeSparseMlp::from_weights(router, experts, shared, geometry)
    }

    fn token_embeddings(&self, device: &Device) -> Result<Tensor> {
        let (canonical, shape, _kind) = self.resolve("token_embd.weight")?;
        self.checkpoint
            .materialize_dense(&canonical, &shape, device, self.dense_target)
    }
}

impl Qwen36MoeNativeSource<'_> {
    fn materialize_expert_projection(
        &self,
        canonical: &str,
        device: &Device,
    ) -> Result<Qwen36MoeLinear> {
        let raw = self.checkpoint.raw_tensor_name(canonical)?;
        let shape = self.checkpoint.tensors.tensor_info(raw)?.shape.clone();
        let expected: [usize; 2] = match shape.as_slice() {
            [rows, cols] => [*rows, *cols],
            other => {
                return Err(Error::ModelLoadError(format!(
                    "qwen36moe native expert projection `{canonical}` has shape {other:?}; expected a 2-D projection"
                )))
            }
        };
        match self
            .checkpoint
            .materialize_projection(canonical, expected, device, self.residency)?
        {
            Qwen36MoeProjection::Packed(qmatmul) => Ok(Qwen36MoeLinear::Quantized(qmatmul)),
            Qwen36MoeProjection::Dense(tensor) => Ok(Qwen36MoeLinear::from_dense(tensor)),
            Qwen36MoeProjection::CompactFp8(raw) => Ok(Qwen36MoeLinear::CompactFp8 {
                weights: raw.weights,
                scales: raw.scales,
            }),
        }
    }
}

/// Map the validated native text geometry onto the shared trunk config.
pub(crate) fn qwen35_text_config_from_native(
    native: &crate::models::architectures::qwen36moe::native::Qwen36MoeTextConfig,
) -> crate::models::architectures::qwen35::chat::Qwen35TextConfig {
    let inner_size = native.ssm_time_step_rank * native.ssm_value_head_dim;
    crate::models::architectures::qwen35::chat::Qwen35TextConfig {
        architecture: "qwen36moe".to_string(),
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
    checkpoint: &Qwen36MoeNativeCheckpoint,
    device_profile: &DeviceProfile,
    device: &Device,
    performance: &crate::performance::CudaPerformanceConfig,
    mtp_enabled: bool,
) -> Result<(
    crate::models::architectures::qwen35::chat::Qwen35TextConfig,
    Qwen35TextModel,
    Option<Qwen35MtpHead>,
)> {
    let text_config = qwen35_text_config_from_native(&checkpoint.config.text);
    let source = Qwen36MoeNativeSource::new_with_performance(checkpoint, device_profile, performance);
    let model = Qwen35TextModel::load_with_source(&source, &text_config, device)?;
    let mtp_head = if mtp_enabled {
        checkpoint
            .mtp
            .as_ref()
            .ok_or_else(|| {
                Error::ModelLoadError(
                    "Qwen3.5/3.6-MoE MTP enabled but the draft manifest was not validated".into(),
                )
            })?;
        Some(Qwen35MtpHead::load_via(
            &source,
            &text_config,
            device,
            performance.mtp_draft_tokens,
        )?)
    } else {
        None
    };
    Ok((text_config, model, mtp_head))
}
