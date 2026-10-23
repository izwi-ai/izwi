//! Device precision and kernel policy for Granite Speech 5.0.
//!
//! These are Granite Speech 4.1's measured defaults, fixed rather than read
//! from the `IZWI_GRANITE_*` switches, which belong to 4.1. The one escape
//! hatch is the global `IZWI_METAL_DTYPE=f32`, which pins every Metal
//! projection to F32.

use candle_core::{DType, Device};

use crate::backends::{DTypeSelection, DeviceProfile, MetalDTypeOverride};
use crate::catalog::ModelFamily;
use crate::error::Result;

/// Decoder positions the LM range probe runs.
pub(super) const LM_PROBE_TOKENS: usize = 16;
/// Metal syncs on every token read, so its decode loop reads the generated
/// tokens back once per interval: a decode runs at most this many steps
/// past its stop token.
pub(super) const METAL_STOP_CHECK_INTERVAL: usize = 8;

/// Which language-model parts run in F16 on F32 weights (Metal's selective
/// F16 policy); all false runs the LM in its base dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LmExecutionPlan {
    pub(super) f16_lm_head: bool,
    pub(super) f16_qkv: bool,
    pub(super) f16_attention_core: bool,
    pub(super) f16_mlp: bool,
    pub(super) f16_attention_output: bool,
}

impl LmExecutionPlan {
    /// Every projection, the head and the attention core in the base dtype:
    /// the range probe's fallback.
    pub(super) const BASE_DTYPE: Self = Self {
        f16_lm_head: false,
        f16_qkv: false,
        f16_attention_core: false,
        f16_mlp: false,
        f16_attention_output: false,
    };

    pub(super) fn any_f16(self) -> bool {
        self != Self::BASE_DTYPE
    }

    /// Selective F16 on Metal over F32 weights, unless `IZWI_METAL_DTYPE`
    /// pins F32.
    pub(super) fn for_device(device: &Device, dtype: DType) -> Self {
        Self::for_policy(device.is_metal(), dtype, MetalDTypeOverride::from_env())
    }

    fn for_policy(is_metal: bool, dtype: DType, global: MetalDTypeOverride) -> Self {
        let selective = is_metal && dtype == DType::F32 && global != MetalDTypeOverride::F32;
        Self {
            f16_lm_head: selective,
            f16_qkv: selective,
            f16_attention_core: selective,
            f16_mlp: selective,
            f16_attention_output: selective,
        }
    }
}

/// What the LM runs below F32, for its load-time range probe: F16 when the
/// selective F16 parts sit on F32 weights, else the base dtype.
pub(super) fn lm_precision(
    dtype: DType,
    plan: LmExecutionPlan,
    device: &DeviceProfile,
) -> DTypeSelection {
    if dtype == DType::F32 && plan.any_f16() {
        DTypeSelection {
            dtype: DType::F16,
            reason: format!(
                "Metal selective F16 LM projections, head and attention core on F32 weights ({})",
                device.metal_bf16_checkpoint_lm_tier().reason
            )
            .into(),
        }
    } else {
        DTypeSelection {
            dtype,
            reason: format!("{dtype:?} LM weights and activations").into(),
        }
    }
}

/// Q/K/V and gate/up are each joined into one projection on every backend.
pub(super) const PROJECTION_FUSION: bool = true;

/// o_proj and down_proj are pre-scaled by the residual multiplier on Metal.
pub(super) fn residual_prescale(device: &Device) -> bool {
    device.is_metal()
}

/// Metal's fused RMSNorm kernel.
pub(super) fn fused_rms_norm(device: &Device) -> bool {
    device.is_metal()
}

/// Bias-free LM projections keep a cached transposed weight on Metal.
pub(super) fn cached_linear_t(device: &Device) -> bool {
    device.is_metal()
}

/// The RoPE table is built in the attention core's dtype on Metal.
pub(super) fn rope_cache_in_attention_dtype(device: &Device) -> bool {
    device.is_metal()
}

/// RoPE kernels by backend. All of them read the host-built cos/sin table.
pub(super) fn rope_pair_bshd_kernel(device: &Device, dtype: DType, head_dim: usize) -> bool {
    device.is_metal() && even_head(head_dim) && matches!(dtype, DType::F16 | DType::F32)
}

pub(super) fn rope_bhtd_kernel(device: &Device, dtype: DType, head_dim: usize) -> bool {
    device.is_metal() && even_head(head_dim) && half_or_full(dtype)
}

pub(super) fn rope_thd_kernel(device: &Device, dtype: DType, head_dim: usize) -> bool {
    device.is_cuda() && even_head(head_dim) && half_or_full(dtype)
}

fn even_head(head_dim: usize) -> bool {
    head_dim > 0 && head_dim.is_multiple_of(2)
}

fn half_or_full(dtype: DType) -> bool {
    matches!(dtype, DType::F16 | DType::BF16 | DType::F32)
}

/// The Q-Former's fused attention runs on CUDA only.
pub(super) fn qformer_fused_attention(device: &Device) -> bool {
    device.is_cuda()
}

/// Metal checks for the stop token once per interval instead of per token.
pub(super) fn chunked_stop_check(device: &Device) -> bool {
    device.is_metal()
}

/// The load-time representation a load of `dtype` weights on `device`
/// resolves. Load admission prices the checkpoint in this form
/// (`runtime/lifecycle/asr_memory.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GraniteSpeech5LoadRepresentation {
    /// A tied LM head gets its own F16 copy of the embedding (5.0's head is
    /// untied, so this loads the head itself in F16).
    pub(crate) f16_lm_head: bool,
    pub(crate) f16_qkv: bool,
    pub(crate) f16_mlp: bool,
    pub(crate) f16_attention_output: bool,
    pub(crate) qkv_fusion: bool,
    pub(crate) gate_up_fusion: bool,
    pub(crate) residual_prescale: bool,
}

pub(crate) fn granite_speech5_load_representation(
    device: &Device,
    dtype: DType,
) -> GraniteSpeech5LoadRepresentation {
    let plan = LmExecutionPlan::for_device(device, dtype);
    GraniteSpeech5LoadRepresentation {
        f16_lm_head: plan.f16_lm_head,
        f16_qkv: plan.f16_qkv,
        f16_mlp: plan.f16_mlp,
        f16_attention_output: plan.f16_attention_output,
        qkv_fusion: PROJECTION_FUSION,
        gate_up_fusion: PROJECTION_FUSION,
        residual_prescale: residual_prescale(device),
    }
}

/// The base weight dtype on `device`: the backend's family policy for the
/// checkpoint's config hint (BF16 on CUDA, F32 on CPU and Metal). Load
/// admission prices the checkpoint at this dtype.
pub(crate) fn select_granite_speech5_dtype(
    device: &DeviceProfile,
    config_hint: Option<&str>,
) -> Result<DType> {
    if device.kind.is_metal()
        && MetalDTypeOverride::from_env() == MetalDTypeOverride::Auto
        && device.metal_bf16_checkpoint_lm_tier().dtype == DType::BF16
    {
        // Apple9+'s BF16 tier, held until Metal paged attention takes BF16.
        return Ok(DType::BF16);
    }
    match config_hint {
        Some(raw) => device.select_model_dtype_checked(
            ModelFamily::GraniteSpeech5Asr,
            Some(raw),
            "Granite Speech 5.0 ASR",
        ),
        None => Ok(device.select_model_dtype(ModelFamily::GraniteSpeech5Asr, None)),
    }
}

/// Whether the conformer encoder's linear and pointwise-convolution weights
/// load as weight-only F16 with F32 activations: on Metal by default, off
/// under `IZWI_METAL_DTYPE=f32`.
pub(crate) fn resolve_granite_speech5_encoder_precision(device: &DeviceProfile) -> DTypeSelection {
    device.resolve_metal_weight_only_f16(None)
}

/// Whether checkpoint tensor `name` is one of the encoder weights that load
/// as weight-only F16 (their biases stay in the encoder dtype).
pub(crate) fn granite_speech5_tensor_is_f16_encoder_projection(name: &str) -> bool {
    let Some(encoder) = name.strip_prefix("encoder.") else {
        return false;
    };
    if matches!(
        encoder,
        "input_linear.weight" | "out.weight" | "out_mid.weight"
    ) {
        return true;
    }
    let Some((index, tail)) = encoder
        .strip_prefix("layers.")
        .and_then(|layer| layer.split_once('.'))
    else {
        return false;
    };
    index.parse::<usize>().is_ok()
        && matches!(
            tail,
            "ff1.up_proj.weight"
                | "ff1.down_proj.weight"
                | "ff2.up_proj.weight"
                | "ff2.down_proj.weight"
                | "attn.to_q.weight"
                | "attn.to_kv.weight"
                | "attn.to_out.weight"
                | "conv.up_conv.weight"
                | "conv.down_conv.weight"
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selective_f16_is_metal_f32_only_and_honours_the_f32_escape() {
        let metal = LmExecutionPlan::for_policy(true, DType::F32, MetalDTypeOverride::Auto);
        assert!(metal.any_f16());
        assert!(metal.f16_attention_core && metal.f16_lm_head && metal.f16_mlp);
        assert_eq!(
            LmExecutionPlan::for_policy(true, DType::F32, MetalDTypeOverride::F32),
            LmExecutionPlan::BASE_DTYPE
        );
        assert_eq!(
            LmExecutionPlan::for_policy(true, DType::BF16, MetalDTypeOverride::Auto),
            LmExecutionPlan::BASE_DTYPE
        );
        assert_eq!(
            LmExecutionPlan::for_policy(false, DType::F32, MetalDTypeOverride::Auto),
            LmExecutionPlan::BASE_DTYPE
        );
    }

    #[test]
    fn cpu_policies_use_the_portable_paths() {
        let cpu = Device::Cpu;
        assert!(!residual_prescale(&cpu));
        assert!(!fused_rms_norm(&cpu));
        assert!(!cached_linear_t(&cpu));
        assert!(!rope_pair_bshd_kernel(&cpu, DType::F32, 128));
        assert!(!rope_bhtd_kernel(&cpu, DType::F32, 128));
        assert!(!rope_thd_kernel(&cpu, DType::F32, 128));
        assert!(!qformer_fused_attention(&cpu));
        assert!(!chunked_stop_check(&cpu));
        let representation = granite_speech5_load_representation(&cpu, DType::F32);
        assert!(representation.qkv_fusion && representation.gate_up_fusion);
        assert!(!representation.residual_prescale && !representation.f16_qkv);
    }

    #[test]
    fn f16_encoder_projection_names_cover_linear_and_pointwise_weights_only() {
        for name in [
            "encoder.input_linear.weight",
            "encoder.out.weight",
            "encoder.out_mid.weight",
            "encoder.layers.15.attn.to_kv.weight",
            "encoder.layers.0.conv.up_conv.weight",
        ] {
            assert!(
                granite_speech5_tensor_is_f16_encoder_projection(name),
                "{name}"
            );
        }
        for name in [
            "encoder.input_linear.bias",
            "encoder.layers.0.attn.rel_pos_emb.weight",
            "encoder.layers.0.conv.depth_conv.conv.weight",
            "encoder.layers.x.attn.to_q.weight",
            "projector.linear.weight",
        ] {
            assert!(
                !granite_speech5_tensor_is_f16_encoder_projection(name),
                "{name}"
            );
        }
    }
}
