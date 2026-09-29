//! GGUF fixture-path configuration for the qwen35moe family.
//!
//! The official Qwen3.5-35B-A3B-FP8 checkpoint is native block-FP8
//! safetensors (see [`super::native`]); GGUF support for this family exists
//! only as the tiny synthetic fixture path that lets CI exercise the hybrid
//! trunk + sparse-expert mechanics without the 35B download (the DS10
//! fixture pattern). Fixture checkpoints declare the same geometry keys as
//! the dense `qwen35` arch, renamed to `qwen35moe`, plus the qwen3-MoE-style
//! expert keys.

use crate::error::{Error, Result};
use crate::models::architectures::qwen35::chat::{
    required_f64, required_usize, required_usize_array, Qwen35TextConfig,
};
use crate::models::architectures::qwen35::text::Qwen35MoeFfnGeometry;
use crate::models::shared::weights::gguf::GgufLoader;

/// Filename the qwen35moe loader treats as the synthetic-fixture checkpoint.
/// The published variant never ships as GGUF, so this name is unambiguous.
pub const QWEN35_MOE_FIXTURE_GGUF_FILENAME: &str = "qwen35moe-fixture.gguf";

pub(crate) fn parse_fixture_gguf_config(loader: &GgufLoader) -> Result<Qwen35TextConfig> {
    let architecture = loader
        .get_metadata_string("general.architecture")
        .unwrap_or_else(|| "qwen35moe".to_string());
    if architecture != "qwen35moe" {
        return Err(Error::ModelLoadError(format!(
            "Expected general.architecture=qwen35moe for the qwen35moe fixture, found {architecture}"
        )));
    }

    let expert_count = required_usize(loader, "qwen35moe.expert_count")?;
    let expert_used_count = required_usize(loader, "qwen35moe.expert_used_count")?;
    let expert_feed_forward_length =
        required_usize(loader, "qwen35moe.expert_feed_forward_length")?;
    let shared_feed_forward_length =
        required_usize(loader, "qwen35moe.expert_shared_feed_forward_length")?;
    if expert_used_count == 0 || expert_used_count > expert_count {
        return Err(Error::ModelLoadError(format!(
            "qwen35moe.expert_used_count {expert_used_count} must be within 1..={expert_count}"
        )));
    }

    Ok(Qwen35TextConfig {
        architecture,
        block_count: required_usize(loader, "qwen35moe.block_count")?,
        context_length: required_usize(loader, "qwen35moe.context_length")?,
        embedding_length: required_usize(loader, "qwen35moe.embedding_length")?,
        // The dense feed-forward width is meaningless for a fully sparse
        // checkpoint; the routed expert width drives workspace estimates.
        feed_forward_length: expert_feed_forward_length,
        attention_head_count: required_usize(loader, "qwen35moe.attention.head_count")?,
        attention_head_count_kv: required_usize(loader, "qwen35moe.attention.head_count_kv")?,
        attention_key_length: required_usize(loader, "qwen35moe.attention.key_length")?,
        attention_value_length: required_usize(loader, "qwen35moe.attention.value_length")?,
        rope_dimension_sections: required_usize_array(
            loader,
            "qwen35moe.rope.dimension_sections",
        )?,
        rope_dimension_count: required_usize(loader, "qwen35moe.rope.dimension_count")?,
        rope_freq_base: required_f64(loader, "qwen35moe.rope.freq_base")?,
        attention_layer_norm_rms_epsilon: required_f64(
            loader,
            "qwen35moe.attention.layer_norm_rms_epsilon",
        )?,
        ssm_conv_kernel: required_usize(loader, "qwen35moe.ssm.conv_kernel")?,
        ssm_state_size: required_usize(loader, "qwen35moe.ssm.state_size")?,
        ssm_group_count: required_usize(loader, "qwen35moe.ssm.group_count")?,
        ssm_time_step_rank: required_usize(loader, "qwen35moe.ssm.time_step_rank")?,
        ssm_inner_size: required_usize(loader, "qwen35moe.ssm.inner_size")?,
        full_attention_interval: required_usize(loader, "qwen35moe.full_attention_interval")?,
        moe_ffn: Some(Qwen35MoeFfnGeometry {
            num_experts: expert_count,
            num_experts_per_tok: expert_used_count,
            expert_intermediate_size: expert_feed_forward_length,
            shared_expert_intermediate_size: shared_feed_forward_length,
        }),
    })
}
