//! Native Candle runtime for Granite Speech ASR.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{
    batch_norm, conv1d, conv1d_no_bias, embedding, layer_norm, linear, linear_no_bias, ops,
    rotary_emb, BatchNorm, Conv1d, Conv1dConfig, Embedding, LayerNorm, Linear, Module, ModuleT,
    RmsNorm, VarBuilder,
};

use crate::backends::DeviceProfile;
use crate::error::{Error, Result};
use crate::kernels::{
    try_fused_decode_gqa_attention, try_fused_rms_norm, try_fused_rope_pair_bshd,
    try_fused_silu_mul_with_status,
};
use crate::models::architectures::qwen3::core::{causal_mask, repeat_kv, Qwen3Cache};
use crate::models::shared::attention::flash::{
    try_fused_self_attention, try_fused_self_attention_scaled,
};
use crate::models::shared::attention::paged::default_kv_page_size;
use crate::models::shared::telemetry::{
    record_decode_attention_path, record_rope_kernel, record_rope_manual, DecodeAttentionPath,
};

use super::config::{GraniteSpeechConfig, GraniteSpeechEncoderConfig, GraniteTextConfig};
use super::preprocessor::GraniteSpeechAudioFeatures;
use super::prompt::{GraniteSpeechPrompt, GraniteSpeechSpecialTokens};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraniteSpeechGenerationStats {
    pub prompt_tokens: usize,
    pub audio_tokens: usize,
    pub generated_tokens: usize,
    pub max_new_tokens: usize,
    pub stop_reason: String,
    pub stop_token: Option<u32>,
    pub dense_decode_cache_enabled: bool,
    pub dense_head_decode_enabled: bool,
    pub qkv_projection_fused: bool,
    pub gate_up_projection_fused: bool,
    pub rope_cache_precomputed: bool,
    pub cuda_device_argmax: bool,
    pub residual_branches_prescaled: bool,
    pub f16_lm_head: bool,
    pub f16_qkv: bool,
    pub f16_attention_core: bool,
    pub f16_mlp: bool,
    pub f16_attention_output: bool,
    pub dense_decode_preallocated: bool,
    pub dense_decode_initial_capacity: usize,
    pub deferred_stop_check: bool,
    pub chunked_stop_check: bool,
    pub stop_check_interval: usize,
    pub dense_decode_max_tokens: usize,
    pub timings: GraniteSpeechGenerationTimings,
    pub decode_profile: Option<GraniteSpeechDecodeProfile>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraniteSpeechGenerationTimings {
    pub prefill: Duration,
    pub decode: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraniteSpeechDecodeProfile {
    pub timing_kind: &'static str,
    pub steps: usize,
    pub layer_count: usize,
    pub step_total_samples: Vec<Duration>,
    pub totals: GraniteSpeechDecodeLoopProfile,
    pub forward: GraniteSpeechForwardProfile,
    pub layers: Vec<GraniteSpeechLayerDecodeProfile>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GraniteSpeechDecodeLoopProfile {
    pub argmax: Duration,
    pub scalar_read: Duration,
    pub stop_check: Duration,
    pub model_forward: Duration,
    pub text_decode: Duration,
    pub delta_emit: Duration,
    pub step_total: Duration,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GraniteSpeechForwardProfile {
    pub token_embedding: Duration,
    pub rope_build: Duration,
    pub layers_total: Duration,
    pub final_norm: Duration,
    pub lm_head: Duration,
    pub lm_head_f16_calls: usize,
    pub lm_head_f32_calls: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GraniteSpeechLayerDecodeProfile {
    pub total: Duration,
    pub input_norm: Duration,
    pub attention: GraniteSpeechAttentionDecodeProfile,
    pub post_attention_norm: Duration,
    pub mlp: GraniteSpeechMlpDecodeProfile,
    pub residual: Duration,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GraniteSpeechAttentionDecodeProfile {
    pub qkv: Duration,
    pub rope: Duration,
    pub cache: Duration,
    pub kernel: Duration,
    pub output: Duration,
    pub dense_head_calls: usize,
    pub dense_head_fused: usize,
    pub dense_head_fallback: usize,
    pub materialized_decode_calls: usize,
    pub prefill_attention_calls: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GraniteSpeechMlpDecodeProfile {
    pub gate_up: Duration,
    pub activation: Duration,
    pub down: Duration,
    pub fused_silu_mul_attempts: usize,
    pub fused_silu_mul_custom: usize,
    pub fused_silu_mul_fallback: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraniteSpeechAudioEmbeddingStats {
    pub upload: Duration,
    pub encoder: Duration,
    pub projector: Duration,
    pub encoder_frames: usize,
    pub encoder_dim: usize,
    pub conformer_context_size: usize,
    pub conformer_blocks: usize,
    pub conformer_pad_frames: usize,
    pub conformer_layers: usize,
    pub qformer_windows: usize,
    pub qformer_window_size: usize,
    pub qformer_queries_per_window: usize,
    pub qformer_layers: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraniteSpeechGeneration {
    pub token_ids: Vec<u32>,
    pub text: String,
    pub stats: GraniteSpeechGenerationStats,
}

pub struct GraniteSpeechRuntime {
    device: Device,
    dtype: DType,
    encoder: GraniteSpeechEncoder,
    projector: GraniteSpeechProjector,
    text_model: GraniteLanguageModel,
}

struct GraniteSpeechDecodeProfiler {
    profile: GraniteSpeechDecodeProfile,
}

impl GraniteSpeechDecodeProfiler {
    fn from_env(layer_count: usize) -> Option<Self> {
        granite_decode_profile_enabled().then(|| Self {
            profile: GraniteSpeechDecodeProfile {
                timing_kind: "host_wall_clock_no_device_sync",
                steps: 0,
                layer_count,
                step_total_samples: Vec::new(),
                totals: GraniteSpeechDecodeLoopProfile::default(),
                forward: GraniteSpeechForwardProfile::default(),
                layers: vec![GraniteSpeechLayerDecodeProfile::default(); layer_count],
            },
        })
    }

    fn add_step(&mut self, step: GraniteSpeechDecodeLoopProfile) {
        self.profile.steps = self.profile.steps.saturating_add(1);
        self.profile.step_total_samples.push(step.step_total);
        self.profile.totals.argmax += step.argmax;
        self.profile.totals.scalar_read += step.scalar_read;
        self.profile.totals.stop_check += step.stop_check;
        self.profile.totals.model_forward += step.model_forward;
        self.profile.totals.text_decode += step.text_decode;
        self.profile.totals.delta_emit += step.delta_emit;
        self.profile.totals.step_total += step.step_total;
    }

    fn add_final_text_decode(&mut self, duration: Duration) {
        self.profile.totals.text_decode += duration;
    }

    fn add_layer(&mut self, layer_idx: usize, layer: GraniteSpeechLayerDecodeProfile) {
        self.profile.forward.layers_total += layer.total;
        if let Some(total) = self.profile.layers.get_mut(layer_idx) {
            total.total += layer.total;
            total.input_norm += layer.input_norm;
            total.attention.qkv += layer.attention.qkv;
            total.attention.rope += layer.attention.rope;
            total.attention.cache += layer.attention.cache;
            total.attention.kernel += layer.attention.kernel;
            total.attention.output += layer.attention.output;
            total.attention.dense_head_calls += layer.attention.dense_head_calls;
            total.attention.dense_head_fused += layer.attention.dense_head_fused;
            total.attention.dense_head_fallback += layer.attention.dense_head_fallback;
            total.attention.materialized_decode_calls += layer.attention.materialized_decode_calls;
            total.attention.prefill_attention_calls += layer.attention.prefill_attention_calls;
            total.post_attention_norm += layer.post_attention_norm;
            total.mlp.gate_up += layer.mlp.gate_up;
            total.mlp.activation += layer.mlp.activation;
            total.mlp.down += layer.mlp.down;
            total.mlp.fused_silu_mul_attempts += layer.mlp.fused_silu_mul_attempts;
            total.mlp.fused_silu_mul_custom += layer.mlp.fused_silu_mul_custom;
            total.mlp.fused_silu_mul_fallback += layer.mlp.fused_silu_mul_fallback;
            total.residual += layer.residual;
        }
    }

    fn finish(self) -> GraniteSpeechDecodeProfile {
        self.profile
    }
}

fn granite_decode_profile_enabled() -> bool {
    std::env::var("IZWI_GRANITE_DECODE_PROFILE")
        .ok()
        .or_else(|| std::env::var("IZWI_GRANITE_DECODE_PROFILING").ok())
        .and_then(|raw| parse_env_bool(&raw))
        .unwrap_or(false)
}

fn granite_projection_fusion_enabled(device: &Device) -> bool {
    let override_enabled = std::env::var("IZWI_GRANITE_PROJECTION_FUSION")
        .ok()
        .and_then(|raw| parse_env_bool(&raw));
    granite_projection_fusion_policy(device.is_metal(), device.is_cuda(), override_enabled)
}

fn granite_projection_fusion_policy(
    is_metal: bool,
    is_cuda: bool,
    override_enabled: Option<bool>,
) -> bool {
    override_enabled.unwrap_or(is_metal || is_cuda)
}

fn granite_gate_up_projection_fusion_enabled(device: &Device) -> bool {
    let override_enabled = std::env::var("IZWI_GRANITE_GATE_UP_PROJECTION_FUSION")
        .ok()
        .or_else(|| std::env::var("IZWI_GRANITE_PROJECTION_FUSION").ok())
        .and_then(|raw| parse_env_bool(&raw));
    granite_gate_up_projection_fusion_policy(device.is_metal(), device.is_cuda(), override_enabled)
}

fn granite_gate_up_projection_fusion_policy(
    is_metal: bool,
    is_cuda: bool,
    override_enabled: Option<bool>,
) -> bool {
    override_enabled.unwrap_or(is_metal || is_cuda)
}

fn granite_rope_kernel_enabled(device: &Device, dtype: DType, head_dim: usize) -> bool {
    let override_enabled = std::env::var("IZWI_GRANITE_ROPE_KERNEL")
        .ok()
        .and_then(|raw| parse_env_bool(&raw));
    if !granite_rope_kernel_policy(device.is_metal(), device.is_cuda(), override_enabled) {
        return false;
    }
    if head_dim == 0 || head_dim % 2 != 0 {
        return false;
    }
    matches!(dtype, DType::F16 | DType::BF16 | DType::F32)
}

fn granite_rope_bhtd_kernel_enabled(device: &Device, dtype: DType, head_dim: usize) -> bool {
    let override_enabled = std::env::var("IZWI_GRANITE_ROPE_BHTD_KERNEL")
        .ok()
        .and_then(|raw| parse_env_bool(&raw));
    if !granite_rope_bhtd_kernel_policy(device.is_metal(), override_enabled) {
        return false;
    }
    if head_dim == 0 || head_dim % 2 != 0 {
        return false;
    }
    matches!(dtype, DType::F16 | DType::BF16 | DType::F32)
}

fn granite_rope_pair_bshd_kernel_enabled(device: &Device, dtype: DType, head_dim: usize) -> bool {
    let override_enabled = std::env::var("IZWI_GRANITE_ROPE_PAIR_BSHD_KERNEL")
        .ok()
        .and_then(|raw| parse_env_bool(&raw));
    if !granite_rope_pair_bshd_kernel_policy(device.is_metal(), override_enabled) {
        return false;
    }
    if head_dim == 0 || head_dim % 2 != 0 {
        return false;
    }
    matches!(dtype, DType::F16 | DType::F32)
}

fn granite_rope_kernel_policy(
    _is_metal: bool,
    is_cuda: bool,
    override_enabled: Option<bool>,
) -> bool {
    override_enabled.unwrap_or(is_cuda)
}

fn granite_rope_bhtd_kernel_policy(is_metal: bool, override_enabled: Option<bool>) -> bool {
    override_enabled.unwrap_or(is_metal)
}

fn granite_rope_pair_bshd_kernel_policy(is_metal: bool, override_enabled: Option<bool>) -> bool {
    override_enabled.unwrap_or(is_metal)
}

fn granite_native_greedy_logits_enabled() -> bool {
    std::env::var("IZWI_GRANITE_NATIVE_GREEDY_LOGITS")
        .ok()
        .and_then(|raw| parse_env_bool(&raw))
        .unwrap_or(true)
}

fn granite_mlp_try_fused_silu_mul(device: &Device) -> bool {
    let override_enabled = std::env::var("IZWI_GRANITE_FUSED_SILU_MUL")
        .ok()
        .and_then(|raw| parse_env_bool(&raw));
    granite_mlp_fused_silu_mul_policy(device.is_metal(), override_enabled)
}

fn granite_mlp_fused_silu_mul_policy(_is_metal: bool, override_enabled: Option<bool>) -> bool {
    override_enabled.unwrap_or(true)
}

fn granite_try_fused_rms_norm(device: &Device) -> bool {
    let override_enabled = std::env::var("IZWI_GRANITE_FUSED_RMS_NORM")
        .ok()
        .and_then(|raw| parse_env_bool(&raw));
    granite_fused_rms_norm_policy(device.is_metal(), override_enabled)
}

fn granite_fused_rms_norm_policy(is_metal: bool, override_enabled: Option<bool>) -> bool {
    override_enabled.unwrap_or(is_metal)
}

fn granite_residual_prescale_enabled(device: &Device) -> bool {
    let override_enabled = std::env::var("IZWI_GRANITE_RESIDUAL_PRESCALE")
        .ok()
        .and_then(|raw| parse_env_bool(&raw));
    granite_residual_prescale_policy(device.is_metal(), override_enabled)
}

fn granite_residual_prescale_policy(is_metal: bool, override_enabled: Option<bool>) -> bool {
    override_enabled.unwrap_or(is_metal)
}

fn granite_dense_decode_preallocate_enabled(device: &Device) -> bool {
    let override_enabled = std::env::var("IZWI_GRANITE_DENSE_CACHE_PREALLOCATE")
        .ok()
        .and_then(|raw| parse_env_bool(&raw));
    granite_dense_decode_preallocate_policy(device.is_metal(), override_enabled)
}

fn granite_dense_decode_preallocate_policy(is_metal: bool, override_enabled: Option<bool>) -> bool {
    override_enabled.unwrap_or(is_metal)
}

fn granite_dense_decode_initial_capacity(
    device: &Device,
    prompt_tokens: usize,
    max_new_tokens: usize,
) -> usize {
    if !granite_dense_decode_preallocate_enabled(device) {
        return 0;
    }
    prompt_tokens.saturating_add(max_new_tokens.max(1))
}

fn granite_deferred_stop_check_enabled(
    device: &Device,
    decode_each_step: bool,
    max_new_tokens: usize,
) -> bool {
    if decode_each_step {
        return false;
    }
    let override_enabled = std::env::var("IZWI_GRANITE_DEFER_STOP_CHECK")
        .ok()
        .and_then(|raw| parse_env_bool(&raw));
    let default_max_tokens = std::env::var("IZWI_GRANITE_DEFER_STOP_MAX_TOKENS")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(128);
    granite_deferred_stop_check_policy(
        device.is_metal(),
        override_enabled,
        max_new_tokens.max(1),
        default_max_tokens,
    )
}

fn granite_deferred_stop_check_policy(
    is_metal: bool,
    override_enabled: Option<bool>,
    max_new_tokens: usize,
    default_max_tokens: usize,
) -> bool {
    override_enabled.unwrap_or(is_metal && max_new_tokens <= default_max_tokens)
}

fn granite_chunked_stop_check_enabled(
    device: &Device,
    decode_each_step: bool,
    max_new_tokens: usize,
) -> bool {
    if decode_each_step {
        return false;
    }
    let override_enabled = std::env::var("IZWI_GRANITE_CHUNKED_STOP_CHECK")
        .ok()
        .and_then(|raw| parse_env_bool(&raw));
    let default_max_tokens = std::env::var("IZWI_GRANITE_CHUNKED_STOP_MAX_TOKENS")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(128);
    granite_chunked_stop_check_policy(
        device.is_metal(),
        override_enabled,
        max_new_tokens.max(1),
        default_max_tokens,
    )
}

fn granite_chunked_stop_check_policy(
    is_metal: bool,
    override_enabled: Option<bool>,
    max_new_tokens: usize,
    default_max_tokens: usize,
) -> bool {
    override_enabled.unwrap_or(is_metal && max_new_tokens <= default_max_tokens)
}

fn granite_chunked_stop_check_interval(max_new_tokens: usize) -> usize {
    let interval = std::env::var("IZWI_GRANITE_CHUNKED_STOP_INTERVAL")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(8);
    interval.clamp(1, max_new_tokens.max(1))
}

fn granite_f16_lm_head_enabled(device: &Device, dtype: DType) -> bool {
    let override_enabled = std::env::var("IZWI_GRANITE_F16_LM_HEAD")
        .ok()
        .and_then(|raw| parse_env_bool(&raw));
    granite_f16_lm_head_policy(device.is_metal(), dtype, override_enabled)
}

fn granite_f16_lm_head_policy(
    is_metal: bool,
    dtype: DType,
    override_enabled: Option<bool>,
) -> bool {
    override_enabled.unwrap_or(is_metal && dtype == DType::F32)
}

fn granite_rope_cache_attention_dtype_enabled(device: &Device) -> bool {
    let override_enabled = std::env::var("IZWI_GRANITE_ROPE_CACHE_ATTENTION_DTYPE")
        .ok()
        .and_then(|raw| parse_env_bool(&raw));
    granite_rope_cache_attention_dtype_policy(device.is_metal(), override_enabled)
}

fn granite_rope_cache_attention_dtype_policy(
    is_metal: bool,
    override_enabled: Option<bool>,
) -> bool {
    override_enabled.unwrap_or(is_metal)
}

fn granite_rope_cache_dtype(
    model_dtype: DType,
    f16_attention_core: bool,
    use_attention_dtype: bool,
) -> DType {
    if use_attention_dtype && f16_attention_core {
        DType::F16
    } else {
        model_dtype
    }
}

fn granite_cached_linear_t_enabled(device: &Device) -> bool {
    let override_enabled = std::env::var("IZWI_GRANITE_CACHED_LINEAR_T")
        .ok()
        .and_then(|raw| parse_env_bool(&raw));
    granite_cached_linear_t_policy(device.is_metal(), override_enabled)
}

fn granite_cached_linear_t_policy(is_metal: bool, override_enabled: Option<bool>) -> bool {
    override_enabled.unwrap_or(is_metal)
}

fn granite_f16_qkv_enabled(device: &Device, dtype: DType) -> bool {
    let override_enabled = std::env::var("IZWI_GRANITE_F16_QKV")
        .ok()
        .and_then(|raw| parse_env_bool(&raw));
    granite_f16_qkv_policy(device.is_metal(), dtype, override_enabled)
}

fn granite_f16_qkv_policy(is_metal: bool, dtype: DType, override_enabled: Option<bool>) -> bool {
    override_enabled.unwrap_or(is_metal && dtype == DType::F32)
}

fn granite_f16_attention_core_enabled(device: &Device, dtype: DType) -> bool {
    let override_enabled = std::env::var("IZWI_GRANITE_F16_ATTENTION_CORE")
        .ok()
        .or_else(|| std::env::var("IZWI_GRANITE_F16_ATTN_CORE").ok())
        .and_then(|raw| parse_env_bool(&raw));
    granite_f16_attention_core_policy(device.is_metal(), dtype, override_enabled)
}

fn granite_f16_attention_core_policy(
    is_metal: bool,
    dtype: DType,
    override_enabled: Option<bool>,
) -> bool {
    override_enabled.unwrap_or(is_metal && dtype == DType::F32)
}

fn granite_f16_mlp_enabled(device: &Device, dtype: DType) -> bool {
    let override_enabled = std::env::var("IZWI_GRANITE_F16_MLP")
        .ok()
        .and_then(|raw| parse_env_bool(&raw));
    granite_f16_mlp_policy(device.is_metal(), dtype, override_enabled)
}

fn granite_f16_mlp_policy(is_metal: bool, dtype: DType, override_enabled: Option<bool>) -> bool {
    override_enabled.unwrap_or(is_metal && dtype == DType::F32)
}

fn granite_f16_attention_output_enabled(device: &Device, dtype: DType) -> bool {
    let override_enabled = std::env::var("IZWI_GRANITE_F16_ATTENTION_OUTPUT")
        .ok()
        .or_else(|| std::env::var("IZWI_GRANITE_F16_O_PROJ").ok())
        .and_then(|raw| parse_env_bool(&raw));
    granite_f16_attention_output_policy(device.is_metal(), dtype, override_enabled)
}

fn granite_f16_attention_output_policy(
    is_metal: bool,
    dtype: DType,
    override_enabled: Option<bool>,
) -> bool {
    override_enabled.unwrap_or(is_metal && dtype == DType::F32)
}

fn parse_env_bool(raw: &str) -> Option<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn profile_start(enabled: bool) -> Option<Instant> {
    enabled.then(Instant::now)
}

fn profile_elapsed(start: Option<Instant>) -> Duration {
    start.map(|started| started.elapsed()).unwrap_or_default()
}

impl GraniteSpeechRuntime {
    pub fn load(
        shard_paths: &[PathBuf],
        config: &GraniteSpeechConfig,
        device: &DeviceProfile,
        dtype: DType,
    ) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(shard_paths, dtype, &device.device).map_err(
                |err| {
                    Error::ModelLoadError(format!(
                        "Failed to mmap Granite Speech safetensors: {err}"
                    ))
                },
            )?
        };
        let encoder = GraniteSpeechEncoder::load(&config.encoder_config, vb.pp("encoder"))?;
        let projector = GraniteSpeechProjector::load(config, vb.pp("projector"))?;
        let text_model = GraniteLanguageModel::load(&config.text_config, vb.pp("language_model"))?;
        Ok(Self {
            device: device.device.clone(),
            dtype,
            encoder,
            projector,
            text_model,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn audio_embeddings(&self, features: &GraniteSpeechAudioFeatures) -> Result<Tensor> {
        self.audio_embeddings_with_stats(features)
            .map(|(embeddings, _stats)| embeddings)
    }

    pub fn audio_embeddings_with_stats(
        &self,
        features: &GraniteSpeechAudioFeatures,
    ) -> Result<(Tensor, GraniteSpeechAudioEmbeddingStats)> {
        let frames = features.encoder_frames;
        let dim = features.encoder_dim;
        if frames == 0 || dim == 0 {
            return Err(Error::InvalidInput(
                "Granite Speech audio produced no encoder features".to_string(),
            ));
        }
        let flat = features
            .input_features
            .iter()
            .flat_map(|frame| frame.iter().copied())
            .collect::<Vec<_>>();
        let upload_start = Instant::now();
        let input = Tensor::from_vec(flat, (1, frames, dim), &self.device)?.to_dtype(self.dtype)?;
        let upload = upload_start.elapsed();
        let encoder_start = Instant::now();
        let encoded = self.encoder.forward(&input)?;
        let encoder = encoder_start.elapsed();
        let projector_start = Instant::now();
        let embeddings = self.projector.forward(&encoded)?;
        let projector = projector_start.elapsed();
        let stats = GraniteSpeechAudioEmbeddingStats {
            upload,
            encoder,
            projector,
            encoder_frames: frames,
            encoder_dim: dim,
            conformer_context_size: self.encoder.context_size(),
            conformer_blocks: self.encoder.block_count_for_frames(frames),
            conformer_pad_frames: self.encoder.pad_frames_for_frames(frames),
            conformer_layers: self.encoder.layer_count(),
            qformer_windows: self.projector.window_count_for_frames(encoded.dim(1)?),
            qformer_window_size: self.projector.window_size(),
            qformer_queries_per_window: self.projector.num_queries(),
            qformer_layers: self.projector.layer_count(),
        };
        Ok((embeddings, stats))
    }

    pub fn generate(
        &self,
        prompt: &GraniteSpeechPrompt,
        special_tokens: &GraniteSpeechSpecialTokens,
        audio_embeds: &Tensor,
        max_new_tokens: usize,
        extra_stop_token_ids: &[u32],
        stop_sequences: &[String],
        decode: &mut dyn FnMut(&[u32]) -> Result<String>,
        on_delta: &mut dyn FnMut(&str),
        emit_deltas: bool,
    ) -> Result<GraniteSpeechGeneration> {
        let audio_tokens = audio_embeds.dim(1)?;
        let input_ids = expand_audio_tokens(
            &prompt.input_ids,
            &prompt.audio_token_positions,
            audio_tokens,
            special_tokens.audio_token_id,
        )?;
        let audio_start = prompt
            .audio_token_positions
            .first()
            .copied()
            .ok_or_else(|| Error::InvalidInput("Granite prompt has no audio token".to_string()))?;
        let max_steps = max_new_tokens.max(1);
        let dense_decode_initial_capacity =
            granite_dense_decode_initial_capacity(&self.device, input_ids.len(), max_steps);

        let mut cache = Qwen3Cache::with_page_size_and_dense_decode_initial_capacity(
            self.text_model.num_layers(),
            default_kv_page_size(),
            &self.device,
            dense_decode_initial_capacity,
        );
        let dense_decode_max_tokens = cache.dense_decode_max_tokens();
        let dense_decode_cache_enabled = dense_decode_max_tokens > 0;
        let dense_head_decode_enabled =
            dense_decode_cache_enabled && granite_dense_head_decode_allowed(&self.device);
        let cuda_device_argmax = self.device.is_cuda();
        let mut profiler = GraniteSpeechDecodeProfiler::from_env(self.text_model.num_layers());
        let profiling = profiler.is_some();
        let prefill_start = Instant::now();
        let rope_cache = self
            .text_model
            .precompute_rope_cache(input_ids.len().saturating_add(max_steps))?;
        let mut logits = self.text_model.forward_prompt_with_audio_with_rope(
            &input_ids,
            audio_start,
            audio_tokens,
            audio_embeds,
            &mut cache,
            rope_cache.as_ref(),
        )?;
        let prefill = prefill_start.elapsed();
        let mut generated = Vec::new();
        let mut rendered = String::new();
        let mut stop_reason = "max_tokens".to_string();
        let mut stop_token = None;
        let stop_tokens = stop_token_set(special_tokens, extra_stop_token_ids);
        let decode_each_step = emit_deltas || !stop_sequences.is_empty();
        let deferred_stop_check =
            granite_deferred_stop_check_enabled(&self.device, decode_each_step, max_steps);
        let chunked_stop_check = !deferred_stop_check
            && granite_chunked_stop_check_enabled(&self.device, decode_each_step, max_steps);
        let stop_check_interval = if chunked_stop_check {
            granite_chunked_stop_check_interval(max_steps)
        } else {
            1
        };
        let mut deferred_token_tensors =
            Vec::with_capacity(if deferred_stop_check { max_steps } else { 0 });
        let mut chunk_token_tensors = Vec::with_capacity(if chunked_stop_check {
            stop_check_interval
        } else {
            0
        });
        let decode_start = Instant::now();

        for step in 0..max_steps {
            let step_start = profile_start(profiling);
            let mut step_profile = GraniteSpeechDecodeLoopProfile::default();
            let argmax_start = profile_start(profiling);
            let token_tensor = argmax_last_token_tensor(&logits)?;
            step_profile.argmax = profile_elapsed(argmax_start);
            if deferred_stop_check {
                deferred_token_tensors.push(token_tensor.clone());
                if step + 1 < max_steps {
                    let forward_start = profile_start(profiling);
                    logits = self.text_model.forward_profiled(
                        &token_tensor,
                        input_ids.len() + step,
                        Some(&mut cache),
                        rope_cache.as_ref(),
                        profiler.as_mut(),
                    )?;
                    step_profile.model_forward += profile_elapsed(forward_start);
                }
                step_profile.step_total = profile_elapsed(step_start);
                if let Some(profiler) = profiler.as_mut() {
                    profiler.add_step(step_profile);
                }
                continue;
            }
            if chunked_stop_check {
                chunk_token_tensors.push(token_tensor.clone());
                if step + 1 < max_steps {
                    let forward_start = profile_start(profiling);
                    logits = self.text_model.forward_profiled(
                        &token_tensor,
                        input_ids.len() + step,
                        Some(&mut cache),
                        rope_cache.as_ref(),
                        profiler.as_mut(),
                    )?;
                    step_profile.model_forward += profile_elapsed(forward_start);
                }

                let should_check_stop =
                    chunk_token_tensors.len() >= stop_check_interval || step + 1 == max_steps;
                if should_check_stop {
                    let scalar_start = profile_start(profiling);
                    let tokens = collect_deferred_token_tensors(&chunk_token_tensors)?;
                    step_profile.scalar_read = profile_elapsed(scalar_start);
                    chunk_token_tensors.clear();

                    let stop_start = profile_start(profiling);
                    let (tokens, first_stop_token) =
                        truncate_tokens_at_first_stop(tokens, &stop_tokens);
                    generated.extend(tokens);
                    if let Some(token) = first_stop_token {
                        stop_reason = "stop_token".to_string();
                        stop_token = Some(token);
                        step_profile.stop_check = profile_elapsed(stop_start);
                        step_profile.step_total = profile_elapsed(step_start);
                        if let Some(profiler) = profiler.as_mut() {
                            profiler.add_step(step_profile);
                        }
                        break;
                    }
                    step_profile.stop_check = profile_elapsed(stop_start);
                }

                step_profile.step_total = profile_elapsed(step_start);
                if let Some(profiler) = profiler.as_mut() {
                    profiler.add_step(step_profile);
                }
                continue;
            }

            let next_logits = if !decode_each_step && step + 1 < max_steps {
                let forward_start = profile_start(profiling);
                let next_logits = self.text_model.forward_profiled(
                    &token_tensor,
                    input_ids.len() + step,
                    Some(&mut cache),
                    rope_cache.as_ref(),
                    profiler.as_mut(),
                )?;
                step_profile.model_forward += profile_elapsed(forward_start);
                Some(next_logits)
            } else {
                None
            };
            let scalar_start = profile_start(profiling);
            let token = token_tensor.reshape(())?.to_scalar::<u32>()?;
            step_profile.scalar_read = profile_elapsed(scalar_start);
            let stop_start = profile_start(profiling);
            if stop_tokens.binary_search(&token).is_ok() {
                stop_reason = "stop_token".to_string();
                stop_token = Some(token);
                step_profile.stop_check = profile_elapsed(stop_start);
                step_profile.step_total = profile_elapsed(step_start);
                if let Some(profiler) = profiler.as_mut() {
                    profiler.add_step(step_profile);
                }
                break;
            }
            step_profile.stop_check = profile_elapsed(stop_start);

            generated.push(token);
            if decode_each_step {
                let text_decode_start = profile_start(profiling);
                let mut next_text = decode(&generated)?;
                step_profile.text_decode += profile_elapsed(text_decode_start);
                let stopped_on_sequence = truncate_at_stop_sequence(&mut next_text, stop_sequences);
                if stopped_on_sequence {
                    stop_reason = "stop_sequence".to_string();
                }
                if emit_deltas && next_text.len() > rendered.len() {
                    let delta = &next_text[rendered.len()..];
                    let delta_start = profile_start(profiling);
                    on_delta(delta);
                    step_profile.delta_emit += profile_elapsed(delta_start);
                }
                rendered = next_text;
                if stopped_on_sequence {
                    step_profile.step_total = profile_elapsed(step_start);
                    if let Some(profiler) = profiler.as_mut() {
                        profiler.add_step(step_profile);
                    }
                    break;
                }
            }

            logits = if let Some(next_logits) = next_logits {
                next_logits
            } else {
                let token_tensor = Tensor::from_vec(vec![token], (1, 1), &self.device)?;
                let forward_start = profile_start(profiling);
                let next_logits = self.text_model.forward_profiled(
                    &token_tensor,
                    input_ids.len() + step,
                    Some(&mut cache),
                    rope_cache.as_ref(),
                    profiler.as_mut(),
                )?;
                step_profile.model_forward += profile_elapsed(forward_start);
                next_logits
            };
            step_profile.step_total = profile_elapsed(step_start);
            if let Some(profiler) = profiler.as_mut() {
                profiler.add_step(step_profile);
            };
        }
        if deferred_stop_check && !deferred_token_tensors.is_empty() {
            let tokens = collect_deferred_token_tensors(&deferred_token_tensors)?;
            let (tokens, first_stop_token) = truncate_tokens_at_first_stop(tokens, &stop_tokens);
            generated = tokens;
            if let Some(token) = first_stop_token {
                stop_reason = "stop_token".to_string();
                stop_token = Some(token);
            }
        }
        if !decode_each_step && !generated.is_empty() {
            let text_decode_start = profile_start(profiling);
            rendered = decode(&generated)?;
            if let Some(profiler) = profiler.as_mut() {
                profiler.add_final_text_decode(profile_elapsed(text_decode_start));
            }
        }
        let decode = decode_start.elapsed();
        let decode_profile = profiler.map(GraniteSpeechDecodeProfiler::finish);

        Ok(GraniteSpeechGeneration {
            text: rendered,
            token_ids: generated.clone(),
            stats: GraniteSpeechGenerationStats {
                prompt_tokens: input_ids.len(),
                audio_tokens,
                generated_tokens: generated.len(),
                max_new_tokens,
                stop_reason,
                stop_token,
                dense_decode_cache_enabled,
                dense_head_decode_enabled,
                qkv_projection_fused: self.text_model.qkv_projection_fused(),
                gate_up_projection_fused: self.text_model.gate_up_projection_fused(),
                rope_cache_precomputed: rope_cache.is_some(),
                cuda_device_argmax,
                residual_branches_prescaled: self.text_model.residual_branches_prescaled(),
                f16_lm_head: self.text_model.f16_lm_head(),
                f16_qkv: self.text_model.f16_qkv(),
                f16_attention_core: self.text_model.f16_attention_core(),
                f16_mlp: self.text_model.f16_mlp(),
                f16_attention_output: self.text_model.f16_attention_output(),
                dense_decode_preallocated: dense_decode_initial_capacity > 0,
                dense_decode_initial_capacity,
                deferred_stop_check,
                chunked_stop_check,
                stop_check_interval,
                dense_decode_max_tokens,
                timings: GraniteSpeechGenerationTimings { prefill, decode },
                decode_profile,
            },
        })
    }
}

fn expand_audio_tokens(
    input_ids: &[u32],
    audio_token_positions: &[usize],
    audio_tokens: usize,
    audio_token_id: u32,
) -> Result<Vec<u32>> {
    if audio_token_positions.len() != 1 {
        return Err(Error::InvalidInput(format!(
            "Granite Speech expects exactly one audio placeholder, got {}",
            audio_token_positions.len()
        )));
    }
    if audio_tokens == 0 {
        return Err(Error::InvalidInput(
            "Granite Speech projected zero audio tokens".to_string(),
        ));
    }
    let pos = audio_token_positions[0];
    if pos >= input_ids.len() {
        return Err(Error::InvalidInput(
            "Granite Speech audio placeholder is out of prompt bounds".to_string(),
        ));
    }
    let mut expanded = Vec::with_capacity(input_ids.len() + audio_tokens.saturating_sub(1));
    expanded.extend_from_slice(&input_ids[..pos]);
    expanded.extend(std::iter::repeat_n(audio_token_id, audio_tokens));
    expanded.extend_from_slice(&input_ids[pos + 1..]);
    Ok(expanded)
}

fn stop_token_set(special_tokens: &GraniteSpeechSpecialTokens, extra: &[u32]) -> Vec<u32> {
    let mut tokens = vec![special_tokens.eos_token_id, special_tokens.pad_token_id];
    tokens.extend_from_slice(extra);
    tokens.sort_unstable();
    tokens.dedup();
    tokens
}

fn collect_deferred_token_tensors(token_tensors: &[Tensor]) -> Result<Vec<u32>> {
    if token_tensors.is_empty() {
        return Ok(Vec::new());
    }
    let tokens = Tensor::cat(token_tensors, 1)?;
    match tokens.dims() {
        [1, _] => tokens.squeeze(0)?.to_vec1::<u32>().map_err(Error::from),
        dims => Err(Error::InferenceError(format!(
            "Granite deferred token collection expected [1,tokens], got {dims:?}"
        ))),
    }
}

fn truncate_tokens_at_first_stop(tokens: Vec<u32>, stop_tokens: &[u32]) -> (Vec<u32>, Option<u32>) {
    let Some(stop_at) = tokens
        .iter()
        .position(|token| stop_tokens.binary_search(token).is_ok())
    else {
        return (tokens, None);
    };
    let stop_token = tokens[stop_at];
    let mut tokens = tokens;
    tokens.truncate(stop_at);
    (tokens, Some(stop_token))
}

fn truncate_at_stop_sequence(text: &mut String, stop_sequences: &[String]) -> bool {
    for stop in stop_sequences
        .iter()
        .map(|value| value.as_str())
        .filter(|value| !value.is_empty())
    {
        if let Some(stop_at) = text.find(stop) {
            text.truncate(stop_at);
            return true;
        }
    }
    false
}

fn argmax_last_logits(logits: &Tensor) -> Result<u32> {
    argmax_last_token_tensor(logits)?
        .reshape(())?
        .to_scalar::<u32>()
        .map_err(Error::from)
}

fn argmax_last_token_tensor(logits: &Tensor) -> Result<Tensor> {
    let last = match logits.dims() {
        [1, vocab] => logits.reshape((*vocab,))?,
        [1, 1, vocab] => logits.reshape((*vocab,))?,
        [1, seq, vocab] if *seq > 0 => logits.narrow(1, seq - 1, 1)?.reshape((*vocab,))?,
        [1, 0, _] => {
            return Err(Error::InferenceError(
                "Granite Speech logits sequence is empty".to_string(),
            ));
        }
        dims => {
            return Err(Error::InferenceError(format!(
                "Granite Speech logits expected [1,vocab] or [1,seq,vocab], got {dims:?}"
            )));
        }
    };
    let idx = last.argmax(D::Minus1)?;
    let idx = if idx.rank() == 0 {
        idx
    } else {
        idx.squeeze(0)?
    };
    idx.to_dtype(DType::U32)?
        .reshape((1, 1))
        .map_err(Error::from)
}

struct GraniteSpeechEncoder {
    input_linear: Linear,
    layers: Vec<GraniteConformerBlock>,
    out: Linear,
    out_mid: Linear,
    cat_hidden_layers: Vec<usize>,
}

impl GraniteSpeechEncoder {
    fn load(config: &GraniteSpeechEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let input_linear = linear(config.input_dim, config.hidden_dim, vb.pp("input_linear"))?;
        let mut layers = Vec::with_capacity(config.num_layers);
        for idx in 0..config.num_layers {
            layers.push(GraniteConformerBlock::load(
                config,
                vb.pp(format!("layers.{idx}")),
            )?);
        }
        let out = linear(config.hidden_dim, config.output_dim, vb.pp("out"))?;
        let out_mid = linear(config.output_dim, config.hidden_dim, vb.pp("out_mid"))?;
        Ok(Self {
            input_linear,
            layers,
            out,
            out_mid,
            cat_hidden_layers: config.cat_hidden_layers.clone(),
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = self.input_linear.forward(input)?;
        let mut exported = Vec::with_capacity(self.cat_hidden_layers.len() + 1);
        if self.cat_hidden_layers.contains(&0) {
            exported.push(x.clone());
        }
        let midpoint = self.layers.len() / 2;
        for (idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            let layer_idx = idx + 1;
            if self.cat_hidden_layers.contains(&layer_idx) {
                exported.push(x.clone());
            }
            if layer_idx == midpoint {
                let mid = self.out.forward(&x)?;
                let mid = ops::softmax_last_dim(&mid)?;
                x = x.broadcast_add(&self.out_mid.forward(&mid)?)?;
            }
        }
        if exported.is_empty() {
            Ok(x)
        } else {
            exported.push(x);
            let refs = exported.iter().collect::<Vec<_>>();
            Tensor::cat(&refs, 2).map_err(Error::from)
        }
    }

    fn context_size(&self) -> usize {
        self.layers
            .first()
            .map(|layer| layer.context_size())
            .unwrap_or(0)
    }

    fn block_count_for_frames(&self, frames: usize) -> usize {
        let context = self.context_size().max(1);
        frames.saturating_add(context - 1) / context
    }

    fn pad_frames_for_frames(&self, frames: usize) -> usize {
        self.block_count_for_frames(frames)
            .saturating_mul(self.context_size().max(1))
            .saturating_sub(frames)
    }

    fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

struct GraniteConformerBlock {
    ff1: GraniteConformerFeedForward,
    attn: GraniteConformerAttention,
    conv: GraniteConformerConv,
    ff2: GraniteConformerFeedForward,
    post_norm: LayerNorm,
}

impl GraniteConformerBlock {
    fn load(config: &GraniteSpeechEncoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ff1: GraniteConformerFeedForward::load(config, vb.pp("ff1"))?,
            attn: GraniteConformerAttention::load(config, vb.pp("attn"))?,
            conv: GraniteConformerConv::load(config, vb.pp("conv"))?,
            ff2: GraniteConformerFeedForward::load(config, vb.pp("ff2"))?,
            post_norm: layer_norm(config.hidden_dim, 1e-5, vb.pp("post_norm"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let ff1 = self.ff1.forward(x)?;
        let mut out = x.broadcast_add(&(ff1 * 0.5)?)?;
        let attn = self.attn.forward(&out)?;
        out = out.broadcast_add(&attn)?;
        let conv = self.conv.forward(&out)?;
        out = out.broadcast_add(&conv)?;
        let ff2 = self.ff2.forward(&out)?;
        out = out.broadcast_add(&(ff2 * 0.5)?)?;
        self.post_norm.forward(&out).map_err(Error::from)
    }

    fn context_size(&self) -> usize {
        self.attn.context_size
    }
}

struct GraniteConformerFeedForward {
    pre_norm: LayerNorm,
    up_proj: Linear,
    down_proj: Linear,
}

impl GraniteConformerFeedForward {
    fn load(config: &GraniteSpeechEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let hidden = config.hidden_dim;
        let intermediate = hidden * config.feedforward_mult;
        Ok(Self {
            pre_norm: layer_norm(hidden, 1e-5, vb.pp("pre_norm"))?,
            up_proj: linear(hidden, intermediate, vb.pp("up_proj"))?,
            down_proj: linear(intermediate, hidden, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.pre_norm.forward(x)?;
        let x = self.up_proj.forward(&x)?.silu()?;
        self.down_proj.forward(&x).map_err(Error::from)
    }
}

struct GraniteConformerAttention {
    pre_norm: LayerNorm,
    to_q: Linear,
    to_kv: Linear,
    to_out: Linear,
    rel_pos_emb: Embedding,
    attention_dists: Tensor,
    context_size: usize,
    num_heads: usize,
    dim_head: usize,
}

impl GraniteConformerAttention {
    fn load(config: &GraniteSpeechEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let inner = config.dim_head * config.num_heads;
        let attention_dists =
            encoder_attention_dists(config.context_size, config.max_pos_emb, vb.device())?;
        Ok(Self {
            pre_norm: layer_norm(config.hidden_dim, 1e-5, vb.pp("pre_norm"))?,
            to_q: linear_no_bias(config.hidden_dim, inner, vb.pp("to_q"))?,
            to_kv: linear_no_bias(config.hidden_dim, inner * 2, vb.pp("to_kv"))?,
            to_out: linear(inner, config.hidden_dim, vb.pp("to_out"))?,
            rel_pos_emb: embedding(
                2 * config.max_pos_emb + 1,
                config.dim_head,
                vb.pp("rel_pos_emb"),
            )?,
            attention_dists,
            context_size: config.context_size,
            num_heads: config.num_heads,
            dim_head: config.dim_head,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.pre_norm.forward(x)?;
        let (batch, seq_len, _) = x.dims3()?;
        let nblocks = seq_len.saturating_add(self.context_size - 1) / self.context_size;
        let padded_len = nblocks * self.context_size;
        let pad = padded_len.saturating_sub(seq_len);
        let padded = if pad > 0 {
            x.pad_with_zeros(1, 0, pad)?
        } else {
            x
        };

        let q = self.to_q.forward(&padded)?;
        let kv = self.to_kv.forward(&padded)?;
        let k = kv.narrow(2, 0, self.num_heads * self.dim_head)?;
        let v = kv.narrow(
            2,
            self.num_heads * self.dim_head,
            self.num_heads * self.dim_head,
        )?;
        let q = encoder_attention_heads(
            &q,
            batch,
            nblocks,
            self.context_size,
            self.num_heads,
            self.dim_head,
        )?;
        let k = encoder_attention_heads(
            &k,
            batch,
            nblocks,
            self.context_size,
            self.num_heads,
            self.dim_head,
        )?;
        let v = encoder_attention_heads(
            &v,
            batch,
            nblocks,
            self.context_size,
            self.num_heads,
            self.dim_head,
        )?;

        let pos_bias = self.relative_position_bias(&q)?;
        let scale = 1.0 / (self.dim_head as f64).sqrt();
        let mut attn = encoder_attention_scores(&q, &k)?;
        attn = (attn * scale)?.broadcast_add(&pos_bias)?;
        if pad > 0 {
            let mask = encoder_padding_mask(
                self.context_size,
                seq_len % self.context_size,
                nblocks,
                attn.device(),
                attn.dtype(),
            )?;
            attn = attn.broadcast_add(&mask)?;
        }
        let attn = ops::softmax(&attn.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(q.dtype())?;
        let out = encoder_attention_context(&attn, &v)?;
        let out = out
            .transpose(2, 3)?
            .reshape((batch, padded_len, self.num_heads * self.dim_head))?
            .narrow(1, 0, seq_len)?;
        self.to_out.forward(&out).map_err(Error::from)
    }

    fn relative_position_bias(&self, q: &Tensor) -> Result<Tensor> {
        let out_dtype = q.dtype();
        let rel = self
            .rel_pos_emb
            .forward(&self.attention_dists)?
            .to_dtype(out_dtype)?;
        let (batch, blocks, heads, context, dim) = q.dims5()?;
        let q = q
            .reshape((batch * blocks * heads, context, dim))?
            .to_dtype(DType::F32)?;
        let rel = rel.to_dtype(DType::F32)?;
        let mut rows = Vec::with_capacity(context);
        for query_idx in 0..context {
            let q_row = q.narrow(1, query_idx, 1)?;
            let rel_row = rel.narrow(0, query_idx, 1)?.squeeze(0)?;
            rows.push(relative_position_score_row(&q_row, &rel_row)?);
        }
        let refs = rows.iter().collect::<Vec<_>>();
        let out = Tensor::cat(&refs, 1)?.reshape((batch, blocks, heads, context, context))?;
        (out / (dim as f64).sqrt())?
            .to_dtype(out_dtype)
            .map_err(Error::from)
    }
}

fn encoder_attention_heads(
    x: &Tensor,
    batch: usize,
    nblocks: usize,
    context: usize,
    num_heads: usize,
    dim_head: usize,
) -> Result<Tensor> {
    x.reshape((batch, nblocks, context, num_heads, dim_head))?
        .transpose(2, 3)?
        .contiguous()
        .map_err(Error::from)
}

fn encoder_attention_scores(q: &Tensor, k: &Tensor) -> Result<Tensor> {
    if q.device().is_cpu() {
        let (batch, blocks, heads, context, dim_head) = q.dims5()?;
        let flat = batch * blocks * heads;
        let q = q.reshape((flat, context, dim_head))?;
        let k = k.reshape((flat, context, dim_head))?;
        return q
            .matmul(&k.t()?)?
            .reshape((batch, blocks, heads, context, context))
            .map_err(Error::from);
    }
    q.matmul(&k.t()?).map_err(Error::from)
}

fn encoder_attention_context(attn: &Tensor, v: &Tensor) -> Result<Tensor> {
    if attn.device().is_cpu() {
        let (batch, blocks, heads, context, _) = attn.dims5()?;
        let dim_head = v.dim(D::Minus1)?;
        let flat = batch * blocks * heads;
        let attn = attn.reshape((flat, context, context))?;
        let v = v.reshape((flat, context, dim_head))?;
        return attn
            .matmul(&v)?
            .reshape((batch, blocks, heads, context, dim_head))
            .map_err(Error::from);
    }
    attn.matmul(v).map_err(Error::from)
}

fn relative_position_score_row(q_row: &Tensor, rel_row: &Tensor) -> Result<Tensor> {
    let q_row = q_row.squeeze(1)?.contiguous()?;
    let rel_row = rel_row.t()?.contiguous()?;
    q_row.matmul(&rel_row)?.unsqueeze(1).map_err(Error::from)
}

fn encoder_attention_dists(context: usize, max_pos_emb: usize, device: &Device) -> Result<Tensor> {
    let mut values = Vec::with_capacity(context * context);
    for i in 0..context {
        for j in 0..context {
            let dist = i as isize - j as isize;
            let clamped = dist.clamp(-(context as isize), context as isize);
            values.push((clamped + max_pos_emb as isize) as u32);
        }
    }
    Tensor::from_vec(values, (context, context), device).map_err(Error::from)
}

fn encoder_padding_mask(
    context: usize,
    remainder: usize,
    nblocks: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let valid = remainder.max(1);
    let mut data = vec![0f32; nblocks * context * context];
    if nblocks > 0 {
        let final_block = (nblocks - 1) * context * context;
        for q in 0..context {
            for k in 0..context {
                if q >= valid || k >= valid {
                    data[final_block + q * context + k] = -1e4;
                }
            }
        }
    }
    Tensor::from_vec(data, (1, nblocks, 1, context, context), device)?
        .to_dtype(dtype)
        .map_err(Error::from)
}

struct GraniteConformerConv {
    norm: LayerNorm,
    up_conv: Conv1d,
    depth_conv: Conv1d,
    batch_norm: BatchNorm,
    down_conv: Conv1d,
}

impl GraniteConformerConv {
    fn load(config: &GraniteSpeechEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let hidden = config.hidden_dim;
        let inner = hidden * config.conv_expansion_factor;
        let mut depth_cfg = Conv1dConfig::default();
        depth_cfg.groups = inner;
        Ok(Self {
            norm: layer_norm(hidden, 1e-5, vb.pp("norm"))?,
            up_conv: conv1d(
                hidden,
                inner * 2,
                1,
                Conv1dConfig::default(),
                vb.pp("up_conv"),
            )?,
            depth_conv: conv1d_no_bias(
                inner,
                inner,
                config.conv_kernel_size,
                depth_cfg,
                vb.pp("depth_conv.conv"),
            )?,
            batch_norm: batch_norm(inner, 1e-5, vb.pp("batch_norm"))?,
            down_conv: conv1d(
                inner,
                hidden,
                1,
                Conv1dConfig::default(),
                vb.pp("down_conv"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.norm.forward(x)?;
        let x = x.transpose(1, 2)?.contiguous()?;
        let x = self.up_conv.forward(&x)?;
        let x = glu_channels(&x)?;
        let pad = self.depth_conv_weight_width() / 2;
        let right = pad.saturating_sub((self.depth_conv_weight_width() + 1) % 2);
        let x = x.pad_with_zeros(2, pad, right)?;
        let x = self.depth_conv.forward(&x)?;
        let x = self.batch_norm.forward_t(&x, false)?.silu()?;
        self.down_conv
            .forward(&x)?
            .transpose(1, 2)?
            .contiguous()
            .map_err(Error::from)
    }

    fn depth_conv_weight_width(&self) -> usize {
        self.depth_conv.weight().dims().get(2).copied().unwrap_or(1)
    }
}

fn glu_channels(x: &Tensor) -> Result<Tensor> {
    let channels = x.dim(1)?;
    let half = channels / 2;
    let gate = x.narrow(1, 0, half)?;
    let value = x.narrow(1, half, half)?;
    gate.broadcast_mul(&ops::sigmoid(&value)?)
        .map_err(Error::from)
}

struct GraniteSpeechProjector {
    query: Tensor,
    qformer: GraniteQFormer,
    linear: Linear,
    window_size: usize,
    num_queries: usize,
}

impl GraniteSpeechProjector {
    fn load(config: &GraniteSpeechConfig, vb: VarBuilder) -> Result<Self> {
        let window_size = config.window_size.max(1);
        let downsample = config.downsample_rate.max(1);
        let num_queries = window_size / downsample;
        let query = vb.get(
            (1, num_queries, config.projector_config.hidden_size),
            "query",
        )?;
        Ok(Self {
            query,
            qformer: GraniteQFormer::load(config, vb.pp("qformer"))?,
            linear: linear(
                config.projector_config.hidden_size,
                config.text_config.hidden_size,
                vb.pp("linear"),
            )?,
            window_size,
            num_queries,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, dim) = hidden.dims3()?;
        let nblocks = seq_len.saturating_add(self.window_size - 1) / self.window_size;
        let pad = nblocks * self.window_size - seq_len;
        let hidden = if pad > 0 {
            hidden.pad_with_zeros(1, 0, pad)?
        } else {
            hidden.clone()
        };
        let windows = hidden.reshape((batch * nblocks, self.window_size, dim))?;
        let query =
            self.query
                .broadcast_as((batch * nblocks, self.num_queries, self.query.dim(2)?))?;
        let output = self.qformer.forward(&query, &windows)?;
        let output = output.reshape((batch, nblocks * self.num_queries, output.dim(2)?))?;
        self.linear.forward(&output).map_err(Error::from)
    }

    fn window_count_for_frames(&self, frames: usize) -> usize {
        frames.saturating_add(self.window_size - 1) / self.window_size
    }

    fn window_size(&self) -> usize {
        self.window_size
    }

    fn num_queries(&self) -> usize {
        self.num_queries
    }

    fn layer_count(&self) -> usize {
        self.qformer.layer_count()
    }
}

struct GraniteQFormer {
    layernorm: LayerNorm,
    layers: Vec<GraniteQFormerLayer>,
}

impl GraniteQFormer {
    fn load(config: &GraniteSpeechConfig, vb: VarBuilder) -> Result<Self> {
        let hidden = config.projector_config.hidden_size;
        let mut layers = Vec::with_capacity(config.projector_config.num_hidden_layers);
        for idx in 0..config.projector_config.num_hidden_layers {
            layers.push(GraniteQFormerLayer::load(
                config,
                idx,
                vb.pp(format!("encoder.layer.{idx}")),
            )?);
        }
        Ok(Self {
            layernorm: layer_norm(
                hidden,
                config.projector_config.layer_norm_eps,
                vb.pp("layernorm"),
            )?,
            layers,
        })
    }

    fn forward(&self, query: &Tensor, encoder_hidden: &Tensor) -> Result<Tensor> {
        let mut x = self.layernorm.forward(query)?;
        for layer in &self.layers {
            x = layer.forward(&x, encoder_hidden)?;
        }
        Ok(x)
    }

    fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

struct GraniteQFormerLayer {
    attention: GraniteQFormerAttention,
    crossattention: Option<GraniteQFormerAttention>,
    intermediate_query: Linear,
    output_query_dense: Linear,
    output_query_norm: LayerNorm,
}

impl GraniteQFormerLayer {
    fn load(config: &GraniteSpeechConfig, idx: usize, vb: VarBuilder) -> Result<Self> {
        let hidden = config.projector_config.hidden_size;
        let intermediate = config.projector_config.intermediate_size;
        let has_cross = idx % config.projector_config.cross_attention_frequency.max(1) == 0;
        Ok(Self {
            attention: GraniteQFormerAttention::load(config, false, vb.pp("attention"))?,
            crossattention: if has_cross {
                Some(GraniteQFormerAttention::load(
                    config,
                    true,
                    vb.pp("crossattention"),
                )?)
            } else {
                None
            },
            intermediate_query: linear(hidden, intermediate, vb.pp("intermediate_query.dense"))?,
            output_query_dense: linear(intermediate, hidden, vb.pp("output_query.dense"))?,
            output_query_norm: layer_norm(
                hidden,
                config.projector_config.layer_norm_eps,
                vb.pp("output_query.LayerNorm"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor, encoder_hidden: &Tensor) -> Result<Tensor> {
        let mut out = self.attention.forward(x, None)?;
        if let Some(crossattention) = &self.crossattention {
            out = crossattention.forward(&out, Some(encoder_hidden))?;
        }
        let ff = self.intermediate_query.forward(&out)?.gelu_erf()?;
        let ff = self.output_query_dense.forward(&ff)?;
        self.output_query_norm
            .forward(&ff.broadcast_add(&out)?)
            .map_err(Error::from)
    }
}

struct GraniteQFormerAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output_dense: Linear,
    output_norm: LayerNorm,
    num_heads: usize,
    head_dim: usize,
}

impl GraniteQFormerAttention {
    fn load(config: &GraniteSpeechConfig, cross: bool, vb: VarBuilder) -> Result<Self> {
        let hidden = config.projector_config.hidden_size;
        let kv_hidden = if cross {
            config.projector_config.encoder_hidden_size
        } else {
            hidden
        };
        let all_heads = config.projector_config.num_attention_heads
            * (hidden / config.projector_config.num_attention_heads);
        Ok(Self {
            query: linear(hidden, all_heads, vb.pp("attention.query"))?,
            key: linear(kv_hidden, all_heads, vb.pp("attention.key"))?,
            value: linear(kv_hidden, all_heads, vb.pp("attention.value"))?,
            output_dense: linear(all_heads, hidden, vb.pp("output.dense"))?,
            output_norm: layer_norm(
                hidden,
                config.projector_config.layer_norm_eps,
                vb.pp("output.LayerNorm"),
            )?,
            num_heads: config.projector_config.num_attention_heads,
            head_dim: hidden / config.projector_config.num_attention_heads,
        })
    }

    fn forward(&self, x: &Tensor, encoder_hidden: Option<&Tensor>) -> Result<Tensor> {
        let kv_input = encoder_hidden.unwrap_or(x);
        let q = self.transpose_for_scores(&self.query.forward(x)?)?;
        let k = self.transpose_for_scores(&self.key.forward(kv_input)?)?;
        let v = self.transpose_for_scores(&self.value.forward(kv_input)?)?;
        if granite_qformer_fused_attention_allowed(q.device()) {
            if let Some(context) = try_fused_self_attention(&q, &k, &v, None, self.head_dim, false)?
            {
                let context = context.transpose(1, 2)?.flatten_from(D::Minus2)?;
                let out = self.output_dense.forward(&context)?;
                return self
                    .output_norm
                    .forward(&out.broadcast_add(x)?)
                    .map_err(Error::from);
            }
        }
        let mut scores = q.matmul(&k.t()?)?;
        scores = (scores / (self.head_dim as f64).sqrt())?;
        let probs = ops::softmax(&scores.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(q.dtype())?;
        let context = probs.matmul(&v)?.transpose(1, 2)?.flatten_from(D::Minus2)?;
        let out = self.output_dense.forward(&context)?;
        self.output_norm
            .forward(&out.broadcast_add(x)?)
            .map_err(Error::from)
    }

    fn transpose_for_scores(&self, x: &Tensor) -> Result<Tensor> {
        qformer_attention_heads(x, self.num_heads, self.head_dim)
    }
}

fn qformer_attention_heads(x: &Tensor, num_heads: usize, head_dim: usize) -> Result<Tensor> {
    let (batch, seq_len, _) = x.dims3()?;
    x.reshape((batch, seq_len, num_heads, head_dim))?
        .transpose(1, 2)?
        .contiguous()
        .map_err(Error::from)
}

#[derive(Clone, Debug)]
struct GraniteLinearNoBias {
    linear: Linear,
    weight_t: Option<Tensor>,
}

impl GraniteLinearNoBias {
    fn new(linear: Linear) -> Result<Self> {
        let weight_t = if granite_cached_linear_t_enabled(linear.weight().device()) {
            Some(linear.weight().t()?)
        } else {
            None
        };
        Ok(Self { linear, weight_t })
    }

    fn from_weight(weight: Tensor) -> Result<Self> {
        Self::new(Linear::new(weight, None))
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(weight_t) = self.weight_t.as_ref() {
            return cached_linear_no_bias_forward(x, weight_t)
                .or_else(|_| self.linear.forward(x).map_err(Error::from));
        }
        self.linear.forward(x).map_err(Error::from)
    }

    fn weight(&self) -> &Tensor {
        self.linear.weight()
    }
}

fn cached_linear_no_bias_forward(x: &Tensor, weight_t: &Tensor) -> Result<Tensor> {
    match *x.dims() {
        [b1, b2, m, k] if x.is_contiguous() => x
            .reshape((b1 * b2 * m, k))?
            .matmul(weight_t)?
            .reshape((b1, b2, m, ()))
            .map_err(Error::from),
        [bsize, m, k] if x.is_contiguous() => x
            .reshape((bsize * m, k))?
            .matmul(weight_t)?
            .reshape((bsize, m, ()))
            .map_err(Error::from),
        _ => x.matmul(weight_t).map_err(Error::from),
    }
}

struct GraniteLanguageModel {
    embed_tokens: Embedding,
    layers: Vec<GraniteDecoderLayer>,
    norm: RmsNorm,
    lm_head: GraniteLinearNoBias,
    lm_head_f16: Option<GraniteLinearNoBias>,
    cfg: GraniteTextConfig,
    device: Device,
    residual_branches_prescaled: bool,
}

impl GraniteLanguageModel {
    fn load(config: &GraniteTextConfig, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("model.embed_tokens"),
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let residual_branches_prescaled = granite_residual_prescale_enabled(vb.device());
        for idx in 0..config.num_hidden_layers {
            layers.push(GraniteDecoderLayer::load(
                config,
                vb.pp(format!("model.layers.{idx}")),
                residual_branches_prescaled,
            )?);
        }
        let norm =
            candle_nn::rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("model.norm"))?;
        let lm_head = if config.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
        };
        let lm_head = GraniteLinearNoBias::new(lm_head)?;
        let lm_head_f16 =
            if granite_f16_lm_head_enabled(vb.device(), embed_tokens.embeddings().dtype()) {
                Some(GraniteLinearNoBias::from_weight(
                    lm_head.weight().to_dtype(DType::F16)?,
                )?)
            } else {
                None
            };
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            lm_head_f16,
            cfg: config.clone(),
            device: vb.device().clone(),
            residual_branches_prescaled,
        })
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn residual_branches_prescaled(&self) -> bool {
        self.residual_branches_prescaled
    }

    fn f16_lm_head(&self) -> bool {
        self.lm_head_f16.is_some()
    }

    fn f16_qkv(&self) -> bool {
        self.layers
            .first()
            .map(|layer| layer.f16_qkv())
            .unwrap_or(false)
    }

    fn f16_attention_core(&self) -> bool {
        self.layers
            .first()
            .map(|layer| layer.f16_attention_core())
            .unwrap_or(false)
    }

    fn f16_mlp(&self) -> bool {
        self.layers
            .first()
            .map(|layer| layer.f16_mlp())
            .unwrap_or(false)
    }

    fn f16_attention_output(&self) -> bool {
        self.layers
            .first()
            .map(|layer| layer.f16_attention_output())
            .unwrap_or(false)
    }

    fn qkv_projection_fused(&self) -> bool {
        self.layers
            .first()
            .map(|layer| layer.qkv_projection_fused())
            .unwrap_or(false)
    }

    fn gate_up_projection_fused(&self) -> bool {
        self.layers
            .first()
            .map(|layer| layer.gate_up_projection_fused())
            .unwrap_or(false)
    }

    fn precompute_rope_cache(&self, total_len: usize) -> Result<Option<GraniteRopeCache>> {
        let Some(layer) = self.layers.first() else {
            return Ok(None);
        };
        let dtype = granite_rope_cache_dtype(
            self.embed_tokens.embeddings().dtype(),
            layer.self_attn.f16_attention_core,
            granite_rope_cache_attention_dtype_enabled(&self.device),
        );
        GraniteRopeCache::build(
            total_len,
            layer.self_attn.head_dim,
            layer.self_attn.rope_theta,
            &self.device,
            dtype,
        )
        .map(Some)
    }

    fn embeddings(&self, input_ids: &[u32]) -> Result<Tensor> {
        let ids = input_ids.iter().map(|id| *id as i64).collect::<Vec<_>>();
        let ids = Tensor::from_vec(ids, (1, input_ids.len()), &self.device)?;
        self.embed_tokens.forward(&ids).map_err(Error::from)
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        start_pos: usize,
        cache: Option<&mut Qwen3Cache>,
    ) -> Result<Tensor> {
        self.forward_profiled(input_ids, start_pos, cache, None, None)
    }

    fn forward_profiled(
        &self,
        input_ids: &Tensor,
        start_pos: usize,
        cache: Option<&mut Qwen3Cache>,
        rope_cache: Option<&GraniteRopeCache>,
        mut profile: Option<&mut GraniteSpeechDecodeProfiler>,
    ) -> Result<Tensor> {
        let profiling = profile.is_some();
        let embed_start = profile_start(profiling);
        let embeds = self.embed_tokens.forward(input_ids)?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.profile.forward.token_embedding += profile_elapsed(embed_start);
        }
        self.forward_with_embeds_profiled(&embeds, start_pos, cache, rope_cache, profile)
    }

    fn forward_prompt_with_audio(
        &self,
        input_ids: &[u32],
        audio_start: usize,
        audio_len: usize,
        audio_embeds: &Tensor,
        cache: &mut Qwen3Cache,
    ) -> Result<Tensor> {
        self.forward_prompt_with_audio_with_rope(
            input_ids,
            audio_start,
            audio_len,
            audio_embeds,
            cache,
            None,
        )
    }

    fn forward_prompt_with_audio_with_rope(
        &self,
        input_ids: &[u32],
        audio_start: usize,
        audio_len: usize,
        audio_embeds: &Tensor,
        cache: &mut Qwen3Cache,
        rope_cache: Option<&GraniteRopeCache>,
    ) -> Result<Tensor> {
        let mut llm_ids = input_ids.to_vec();
        for idx in audio_start..audio_start + audio_len {
            if idx < llm_ids.len() {
                llm_ids[idx] = 0;
            }
        }
        let embeds = self.embeddings(&llm_ids)?;
        let seq_len = embeds.dim(1)?;
        let hidden = embeds.dim(2)?;
        let before = if audio_start > 0 {
            embeds.narrow(1, 0, audio_start)?
        } else {
            Tensor::zeros((1, 0, hidden), embeds.dtype(), embeds.device())?
        };
        let after_start = audio_start + audio_len;
        let after = if after_start < seq_len {
            embeds.narrow(1, after_start, seq_len - after_start)?
        } else {
            Tensor::zeros((1, 0, hidden), embeds.dtype(), embeds.device())?
        };
        let audio = audio_embeds.to_dtype(embeds.dtype())?;
        let merged = Tensor::cat(&[before, audio, after], 1)?;
        self.forward_with_embeds_profiled(&merged, 0, Some(cache), rope_cache, None)
    }

    fn forward_with_embeds(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        cache: Option<&mut Qwen3Cache>,
    ) -> Result<Tensor> {
        self.forward_with_embeds_profiled(embeds, start_pos, cache, None, None)
    }

    fn forward_with_embeds_profiled(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        mut cache: Option<&mut Qwen3Cache>,
        rope_cache: Option<&GraniteRopeCache>,
        mut profile: Option<&mut GraniteSpeechDecodeProfiler>,
    ) -> Result<Tensor> {
        let profiling = profile.is_some();
        let mut x = (embeds * self.cfg.embedding_multiplier as f64)?;
        let rope_start = profile_start(profiling);
        let rope = self
            .layers
            .first()
            .map(|layer| {
                rope_cache
                    .map(|cache| cache.slice(start_pos, x.dim(1)?))
                    .unwrap_or_else(|| {
                        build_rope_cache(
                            x.dim(1)?,
                            layer.self_attn.head_dim,
                            start_pos,
                            layer.self_attn.rope_theta,
                            x.device(),
                            x.dtype(),
                        )
                        .and_then(|(cos, sin)| GraniteRopeSlice::new(cos, sin))
                    })
            })
            .transpose()?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.profile.forward.rope_build += profile_elapsed(rope_start);
        }
        for (idx, layer) in self.layers.iter().enumerate() {
            let cache_ref = cache.as_deref_mut();
            if let Some(profile) = profile.as_deref_mut() {
                let layer_start = profile_start(profiling);
                let mut layer_profile = GraniteSpeechLayerDecodeProfile::default();
                x = layer.forward(
                    &x,
                    start_pos,
                    cache_ref,
                    idx,
                    rope.as_ref(),
                    Some(&mut layer_profile),
                )?;
                layer_profile.total = profile_elapsed(layer_start);
                profile.add_layer(idx, layer_profile);
            } else {
                x = layer.forward(&x, start_pos, cache_ref, idx, rope.as_ref(), None)?;
            }
        }
        let final_norm_start = profile_start(profiling);
        let hidden = granite_rms_norm_forward(&self.norm, &x)?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.profile.forward.final_norm += profile_elapsed(final_norm_start);
        }
        let hidden = last_hidden_for_logits(&hidden)?;
        let lm_head_start = profile_start(profiling);
        let use_f16_lm_head = self.lm_head_f16.is_some();
        if let Some(profile) = profile.as_deref_mut() {
            if use_f16_lm_head {
                profile.profile.forward.lm_head_f16_calls += 1;
            } else {
                profile.profile.forward.lm_head_f32_calls += 1;
            }
        }
        let logits = if let Some(lm_head_f16) = self.lm_head_f16.as_ref() {
            lm_head_f16.forward(&hidden.to_dtype(DType::F16)?)?
        } else {
            self.lm_head.forward(&hidden)?
        };
        if let Some(profile) = profile.as_deref_mut() {
            profile.profile.forward.lm_head += profile_elapsed(lm_head_start);
        }
        if granite_native_greedy_logits_enabled() {
            Ok(logits)
        } else {
            (logits / self.cfg.logits_scaling as f64)
                .and_then(|tensor| tensor.to_dtype(DType::F32))
                .map_err(Error::from)
        }
    }
}

struct GraniteDecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: GraniteTextAttention,
    post_attention_layernorm: RmsNorm,
    mlp: GraniteTextMlp,
    residual_multiplier: f32,
    residual_branches_prescaled: bool,
}

impl GraniteDecoderLayer {
    fn load(
        config: &GraniteTextConfig,
        vb: VarBuilder,
        residual_branches_prescaled: bool,
    ) -> Result<Self> {
        Ok(Self {
            input_layernorm: candle_nn::rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            self_attn: GraniteTextAttention::load(
                config,
                vb.pp("self_attn"),
                residual_branches_prescaled,
            )?,
            post_attention_layernorm: candle_nn::rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: GraniteTextMlp::load(config, vb.pp("mlp"), residual_branches_prescaled)?,
            residual_multiplier: config.residual_multiplier,
            residual_branches_prescaled,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        cache: Option<&mut Qwen3Cache>,
        layer_idx: usize,
        rope: Option<&GraniteRopeSlice>,
        mut profile: Option<&mut GraniteSpeechLayerDecodeProfile>,
    ) -> Result<Tensor> {
        let profiling = profile.is_some();
        let residual = x;
        let norm_start = profile_start(profiling);
        let normed = granite_rms_norm_forward(&self.input_layernorm, x)?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.input_norm += profile_elapsed(norm_start);
        }
        let attn = if let Some(profile) = profile.as_deref_mut() {
            self.self_attn.forward(
                &normed,
                start_pos,
                cache,
                layer_idx,
                rope,
                Some(&mut profile.attention),
            )?
        } else {
            self.self_attn
                .forward(&normed, start_pos, cache, layer_idx, rope, None)?
        };
        let residual_start = profile_start(profiling);
        let attn = self.scale_residual_branch(attn)?;
        let x = residual.broadcast_add(&attn)?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.residual += profile_elapsed(residual_start);
        }
        let residual = &x;
        let post_norm_start = profile_start(profiling);
        let normed = granite_rms_norm_forward(&self.post_attention_layernorm, &x)?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.post_attention_norm += profile_elapsed(post_norm_start);
        }
        let mlp = if let Some(profile) = profile.as_deref_mut() {
            self.mlp.forward(&normed, Some(&mut profile.mlp))?
        } else {
            self.mlp.forward(&normed, None)?
        };
        let residual_start = profile_start(profiling);
        let mlp = self.scale_residual_branch(mlp)?;
        let out = residual.broadcast_add(&mlp)?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.residual += profile_elapsed(residual_start);
        }
        Ok(out)
    }

    fn scale_residual_branch(&self, branch: Tensor) -> Result<Tensor> {
        if self.residual_branches_prescaled {
            Ok(branch)
        } else {
            (branch * self.residual_multiplier as f64).map_err(Error::from)
        }
    }

    fn qkv_projection_fused(&self) -> bool {
        self.self_attn.qkv_projection_fused()
    }

    fn gate_up_projection_fused(&self) -> bool {
        self.mlp.gate_up_projection_fused()
    }

    fn f16_qkv(&self) -> bool {
        self.self_attn.f16_qkv()
    }

    fn f16_attention_core(&self) -> bool {
        self.self_attn.f16_attention_core()
    }

    fn f16_mlp(&self) -> bool {
        self.mlp.f16_mlp()
    }

    fn f16_attention_output(&self) -> bool {
        self.self_attn.f16_attention_output()
    }
}

struct GraniteTextFusedQkvProjection {
    fused: GraniteLinearNoBias,
    q_out: usize,
    k_out: usize,
    v_out: usize,
}

impl GraniteTextFusedQkvProjection {
    fn new(q_proj: &Linear, k_proj: &Linear, v_proj: &Linear) -> Result<Self> {
        let q_weight = q_proj.weight();
        let k_weight = k_proj.weight();
        let v_weight = v_proj.weight();
        let q_out = q_weight.dim(0)?;
        let k_out = k_weight.dim(0)?;
        let v_out = v_weight.dim(0)?;
        let in_dim = q_weight.dim(1)?;
        if k_weight.dim(1)? != in_dim || v_weight.dim(1)? != in_dim {
            return Err(Error::InferenceError(
                "Granite fused QKV projection input dimensions do not match".to_string(),
            ));
        }
        let weight = Tensor::cat(&[q_weight, k_weight, v_weight], 0)?;
        Ok(Self {
            fused: GraniteLinearNoBias::from_weight(weight)?,
            q_out,
            k_out,
            v_out,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let qkv = self.fused.forward(x)?;
        let last_dim = qkv.rank().saturating_sub(1);
        let q = qkv.narrow(last_dim, 0, self.q_out)?;
        let k = qkv.narrow(last_dim, self.q_out, self.k_out)?;
        let v = qkv.narrow(last_dim, self.q_out + self.k_out, self.v_out)?;
        Ok((q, k, v))
    }

    fn to_dtype(&self, dtype: DType) -> Result<Self> {
        Ok(Self {
            fused: GraniteLinearNoBias::from_weight(self.fused.weight().to_dtype(dtype)?)?,
            q_out: self.q_out,
            k_out: self.k_out,
            v_out: self.v_out,
        })
    }
}

enum GraniteTextQkvProjectionParts {
    Fused(GraniteTextFusedQkvProjection),
    Separate {
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
    },
}

impl GraniteTextQkvProjectionParts {
    fn load(config: &GraniteTextConfig, head_dim: usize, vb: VarBuilder) -> Result<Self> {
        let q_proj = linear_no_bias(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_no_bias(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_no_bias(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            vb.pp("v_proj"),
        )?;
        if granite_projection_fusion_enabled(vb.device()) {
            Ok(Self::Fused(GraniteTextFusedQkvProjection::new(
                &q_proj, &k_proj, &v_proj,
            )?))
        } else {
            Ok(Self::Separate {
                q_proj,
                k_proj,
                v_proj,
            })
        }
    }

    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        match self {
            Self::Fused(fused) => fused.forward(x),
            Self::Separate {
                q_proj,
                k_proj,
                v_proj,
            } => Ok((q_proj.forward(x)?, k_proj.forward(x)?, v_proj.forward(x)?)),
        }
    }

    fn is_fused(&self) -> bool {
        matches!(self, Self::Fused(_))
    }

    fn to_dtype(&self, dtype: DType) -> Result<Self> {
        match self {
            Self::Fused(fused) => fused.to_dtype(dtype).map(Self::Fused),
            Self::Separate {
                q_proj,
                k_proj,
                v_proj,
            } => Ok(Self::Separate {
                q_proj: Linear::new(q_proj.weight().to_dtype(dtype)?, None),
                k_proj: Linear::new(k_proj.weight().to_dtype(dtype)?, None),
                v_proj: Linear::new(v_proj.weight().to_dtype(dtype)?, None),
            }),
        }
    }
}

struct GraniteTextQkvProjection {
    projection: GraniteTextQkvProjectionParts,
    f16_projection: Option<GraniteTextQkvProjectionParts>,
}

impl GraniteTextQkvProjection {
    fn load(config: &GraniteTextConfig, head_dim: usize, vb: VarBuilder) -> Result<Self> {
        let projection = GraniteTextQkvProjectionParts::load(config, head_dim, vb.clone())?;
        let f16_projection = if granite_f16_qkv_enabled(vb.device(), projection.weight_dtype()?) {
            Some(projection.to_dtype(DType::F16)?)
        } else {
            None
        };
        Ok(Self {
            projection,
            f16_projection,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        if let Some(f16_projection) = self.f16_projection.as_ref() {
            return self.forward_to_dtype_with_projection(x, x.dtype(), f16_projection);
        }
        self.projection.forward(x)
    }

    fn forward_to_dtype(
        &self,
        x: &Tensor,
        output_dtype: DType,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        if let Some(f16_projection) = self.f16_projection.as_ref() {
            return self.forward_to_dtype_with_projection(x, output_dtype, f16_projection);
        }
        let qkv = self.projection.forward(x)?;
        cast_qkv_dtype(qkv, output_dtype)
    }

    fn forward_to_dtype_with_projection(
        &self,
        x: &Tensor,
        output_dtype: DType,
        projection: &GraniteTextQkvProjectionParts,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (q, k, v) = projection.forward(&x.to_dtype(DType::F16)?)?;
        cast_qkv_dtype((q, k, v), output_dtype)
    }

    fn is_fused(&self) -> bool {
        self.projection.is_fused()
    }

    fn f16_qkv(&self) -> bool {
        self.f16_projection.is_some()
    }
}

impl GraniteTextQkvProjectionParts {
    fn weight_dtype(&self) -> Result<DType> {
        match self {
            Self::Fused(fused) => Ok(fused.fused.weight().dtype()),
            Self::Separate { q_proj, .. } => Ok(q_proj.weight().dtype()),
        }
    }
}

fn cast_qkv_dtype(
    (q, k, v): (Tensor, Tensor, Tensor),
    dtype: DType,
) -> Result<(Tensor, Tensor, Tensor)> {
    if q.dtype() == dtype && k.dtype() == dtype && v.dtype() == dtype {
        return Ok((q, k, v));
    }
    Ok((q.to_dtype(dtype)?, k.to_dtype(dtype)?, v.to_dtype(dtype)?))
}

fn granite_rms_norm_forward(norm: &RmsNorm, x: &Tensor) -> Result<Tensor> {
    if granite_try_fused_rms_norm(x.device()) {
        if let Some(out) = try_fused_rms_norm(x, norm.weight(), norm.eps()) {
            return Ok(out);
        }
    }
    norm.forward(x).map_err(Error::from)
}

struct GraniteTextAttention {
    qkv_proj: GraniteTextQkvProjection,
    o_proj: GraniteLinearNoBias,
    o_proj_f16: Option<GraniteLinearNoBias>,
    f16_attention_core: bool,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f64,
    attention_multiplier: f32,
}

impl GraniteTextAttention {
    fn load(config: &GraniteTextConfig, vb: VarBuilder, prescale_residual: bool) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let o_proj = linear_no_bias(
            config.num_attention_heads * head_dim,
            config.hidden_size,
            vb.pp("o_proj"),
        )?;
        let o_proj =
            maybe_prescale_residual_linear(o_proj, config.residual_multiplier, prescale_residual)?;
        let o_proj = GraniteLinearNoBias::new(o_proj)?;
        let qkv_proj = GraniteTextQkvProjection::load(config, head_dim, vb.clone())?;
        let o_proj_f16 =
            if granite_f16_attention_output_enabled(vb.device(), o_proj.weight().dtype()) {
                Some(GraniteLinearNoBias::from_weight(
                    o_proj.weight().to_dtype(DType::F16)?,
                )?)
            } else {
                None
            };
        let f16_attention_core = qkv_proj.f16_qkv()
            && granite_f16_attention_core_enabled(vb.device(), o_proj.weight().dtype());
        Ok(Self {
            qkv_proj,
            o_proj,
            o_proj_f16,
            f16_attention_core,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
            rope_theta: config.rope_theta,
            attention_multiplier: config.attention_multiplier,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        cache: Option<&mut Qwen3Cache>,
        layer_idx: usize,
        rope: Option<&GraniteRopeSlice>,
        mut profile: Option<&mut GraniteSpeechAttentionDecodeProfile>,
    ) -> Result<Tensor> {
        let profiling = profile.is_some();
        let (batch, seq_len, _) = x.dims3()?;
        let qkv_start = profile_start(profiling);
        let qkv_dtype = if self.f16_attention_core {
            DType::F16
        } else {
            x.dtype()
        };
        let (q, k, v) = self.qkv_proj.forward_to_dtype(x, qkv_dtype)?;
        let mut q = q.reshape((batch, seq_len, self.num_heads, self.head_dim))?;
        let mut k = k.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.qkv += profile_elapsed(qkv_start);
        }
        let rope_start = profile_start(profiling);
        let rope_slice = match rope {
            Some(slice) => slice.clone(),
            None => {
                let (cos, sin) = build_rope_cache(
                    seq_len,
                    self.head_dim,
                    start_pos,
                    self.rope_theta,
                    x.device(),
                    q.dtype(),
                )?;
                GraniteRopeSlice::new(cos, sin)?
            }
        };
        let (cos, sin, packed) = if rope_slice.cos.dtype() == q.dtype()
            && rope_slice.sin.dtype() == q.dtype()
            && rope_slice.packed.dtype() == q.dtype()
        {
            (rope_slice.cos, rope_slice.sin, rope_slice.packed)
        } else {
            (
                rope_slice.cos.to_dtype(q.dtype())?,
                rope_slice.sin.to_dtype(q.dtype())?,
                rope_slice.packed.to_dtype(q.dtype())?,
            )
        };
        let direct_rope =
            if granite_rope_pair_bshd_kernel_enabled(q.device(), q.dtype(), self.head_dim) {
                try_fused_rope_pair_bshd(&q, &k, &packed)
            } else {
                None
            };
        if let Some((q_out, k_out)) = direct_rope {
            record_rope_kernel();
            record_rope_kernel();
            q = q_out;
            k = k_out;
        } else if granite_rope_bhtd_kernel_enabled(q.device(), q.dtype(), self.head_dim) {
            if let Some((q_out, k_out)) = try_apply_granite_rope_pair_bhtd(&q, &k, &cos, &sin)? {
                record_rope_kernel();
                record_rope_kernel();
                q = q_out;
                k = k_out;
            } else {
                record_rope_manual();
                record_rope_manual();
                q = apply_rotary_emb(&q, &cos, &sin)?;
                k = apply_rotary_emb(&k, &cos, &sin)?;
            }
        } else if granite_rope_kernel_enabled(q.device(), q.dtype(), self.head_dim) {
            let cos_kernel = cos.unsqueeze(0)?.contiguous()?;
            let sin_kernel = sin.unsqueeze(0)?.contiguous()?;
            if let Some((q_out, k_out)) =
                try_apply_granite_rope_pair_thd(&q, &k, &cos_kernel, &sin_kernel)?
            {
                record_rope_kernel();
                record_rope_kernel();
                q = q_out;
                k = k_out;
            } else {
                record_rope_manual();
                record_rope_manual();
                q = apply_rotary_emb(&q, &cos, &sin)?;
                k = apply_rotary_emb(&k, &cos, &sin)?;
            }
        } else {
            q = apply_rotary_emb(&q, &cos, &sin)?;
            k = apply_rotary_emb(&k, &cos, &sin)?;
        }
        if let Some(profile) = profile.as_deref_mut() {
            profile.rope += profile_elapsed(rope_start);
        }
        let cache_start = profile_start(profiling);
        let (k, v, total_len) = if let Some(cache) = cache {
            cache.append(layer_idx, k.clone(), v.clone())?;
            if granite_dense_head_decode_allowed(q.device()) && seq_len == 1 && start_pos > 0 {
                if let Some((k_heads, v_heads)) = cache.dense_heads(layer_idx)? {
                    if let Some(profile) = profile.as_deref_mut() {
                        profile.cache += profile_elapsed(cache_start);
                    }
                    record_decode_attention_path(DecodeAttentionPath::Dense);
                    let kernel_start = profile_start(profiling);
                    let (out, dense_kernel) = dense_decode_attention_heads_scaled(
                        &q,
                        &k_heads,
                        &v_heads,
                        self.num_heads,
                        self.num_kv_heads,
                        self.head_dim,
                        self.attention_multiplier,
                    )?;
                    if let Some(profile) = profile.as_deref_mut() {
                        profile.kernel += profile_elapsed(kernel_start);
                        profile.dense_head_calls += 1;
                        match dense_kernel {
                            GraniteDenseDecodeAttentionKernel::Fused => {
                                profile.dense_head_fused += 1;
                            }
                            GraniteDenseDecodeAttentionKernel::Fallback => {
                                profile.dense_head_fallback += 1;
                            }
                        }
                    }
                    let out = out.reshape((batch, seq_len, self.num_heads * self.head_dim))?;
                    let output_start = profile_start(profiling);
                    let out = self.output_projection(&out, x.dtype())?;
                    if let Some(profile) = profile.as_deref_mut() {
                        profile.output += profile_elapsed(output_start);
                    }
                    return Ok(out);
                }
            }
            let (cached_k, cached_v) = cache.materialize(layer_idx)?;
            let total_len = cached_k.dim(1)?;
            (cached_k, cached_v, total_len)
        } else {
            let total_len = k.dim(1)?;
            (k, v, total_len)
        };
        if let Some(profile) = profile.as_deref_mut() {
            profile.cache += profile_elapsed(cache_start);
        }

        let kernel_start = profile_start(profiling);
        if let Some(profile) = profile.as_deref_mut() {
            if seq_len == 1 {
                profile.materialized_decode_calls += 1;
            } else {
                profile.prefill_attention_calls += 1;
            }
        }
        let out = if seq_len == 1 {
            dense_decode_attention_scaled(
                &q,
                &k,
                &v,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.attention_multiplier,
            )?
        } else {
            text_attention(
                &q,
                &k,
                &v,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                start_pos,
                total_len,
                self.attention_multiplier,
            )?
        };
        if let Some(profile) = profile.as_deref_mut() {
            profile.kernel += profile_elapsed(kernel_start);
        }
        let out = out.reshape((batch, seq_len, self.num_heads * self.head_dim))?;
        let output_start = profile_start(profiling);
        let out = self.output_projection(&out, x.dtype())?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.output += profile_elapsed(output_start);
        }
        Ok(out)
    }

    fn qkv_projection_fused(&self) -> bool {
        self.qkv_proj.is_fused()
    }

    fn f16_qkv(&self) -> bool {
        self.qkv_proj.f16_qkv()
    }

    fn f16_attention_core(&self) -> bool {
        self.f16_attention_core
    }

    fn f16_attention_output(&self) -> bool {
        self.o_proj_f16.is_some()
    }

    fn output_projection(&self, x: &Tensor, output_dtype: DType) -> Result<Tensor> {
        let out = if let Some(o_proj_f16) = self.o_proj_f16.as_ref() {
            o_proj_f16.forward(&x.to_dtype(DType::F16)?)?
        } else if x.dtype() == self.o_proj.weight().dtype() {
            self.o_proj.forward(x)?
        } else {
            self.o_proj
                .forward(&x.to_dtype(self.o_proj.weight().dtype())?)?
        };
        if out.dtype() == output_dtype {
            Ok(out)
        } else {
            out.to_dtype(output_dtype).map_err(Error::from)
        }
    }
}

fn granite_dense_head_decode_allowed(device: &Device) -> bool {
    granite_dense_head_decode_policy(device.is_metal(), device.is_cuda())
}

fn granite_dense_head_decode_policy(is_metal: bool, is_cuda: bool) -> bool {
    is_metal || is_cuda
}

fn granite_decode_gqa_kernel_enabled(device: &Device, dtype: DType, total_len: usize) -> bool {
    let override_enabled = std::env::var("IZWI_GRANITE_DECODE_GQA_KERNEL")
        .ok()
        .and_then(|raw| parse_env_bool(&raw));
    if !granite_decode_gqa_kernel_policy(device.is_metal(), override_enabled) {
        return false;
    }
    total_len > 0 && total_len <= 2048 && matches!(dtype, DType::F16 | DType::F32)
}

fn granite_decode_gqa_kernel_policy(is_metal: bool, override_enabled: Option<bool>) -> bool {
    override_enabled.unwrap_or(is_metal)
}

fn granite_qformer_fused_attention_allowed(device: &Device) -> bool {
    device.is_cuda()
}

#[derive(Debug, Clone)]
struct GraniteRopeCache {
    cos: Tensor,
    sin: Tensor,
    packed: Tensor,
}

#[derive(Debug, Clone)]
struct GraniteRopeSlice {
    cos: Tensor,
    sin: Tensor,
    packed: Tensor,
}

impl GraniteRopeSlice {
    fn new(cos: Tensor, sin: Tensor) -> Result<Self> {
        let packed = Tensor::cat(&[&cos, &sin], 1)?;
        Ok(Self { cos, sin, packed })
    }
}

impl GraniteRopeCache {
    fn build(
        total_len: usize,
        head_dim: usize,
        rope_theta: f64,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let (cos, sin) = build_rope_cache(total_len, head_dim, 0, rope_theta, device, dtype)?;
        let packed = Tensor::cat(&[&cos, &sin], 1)?;
        Ok(Self { cos, sin, packed })
    }

    fn slice(&self, start_pos: usize, seq_len: usize) -> Result<GraniteRopeSlice> {
        if start_pos.saturating_add(seq_len) > self.cos.dim(0)? {
            return Err(Error::InferenceError(format!(
                "Granite RoPE cache too short for start_pos={start_pos}, seq_len={seq_len}"
            )));
        }
        Ok(GraniteRopeSlice {
            cos: self.cos.narrow(0, start_pos, seq_len)?,
            sin: self.sin.narrow(0, start_pos, seq_len)?,
            packed: self.packed.narrow(0, start_pos, seq_len)?,
        })
    }
}

fn build_rope_cache(
    seq_len: usize,
    head_dim: usize,
    start_pos: usize,
    rope_theta: f64,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let half_dim = head_dim / 2;
    let mut cos = Vec::with_capacity(seq_len * half_dim);
    let mut sin = Vec::with_capacity(seq_len * half_dim);
    for pos in start_pos..start_pos + seq_len {
        for idx in 0..half_dim {
            let power = (2.0 * idx as f64) / head_dim as f64;
            let inv = 1.0 / rope_theta.powf(power);
            let angle = pos as f64 * inv;
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    Ok((
        Tensor::from_vec(cos, (seq_len, half_dim), device)?.to_dtype(dtype)?,
        Tensor::from_vec(sin, (seq_len, half_dim), device)?.to_dtype(dtype)?,
    ))
}

fn apply_rotary_emb(x: &Tensor, cos_half: &Tensor, sin_half: &Tensor) -> Result<Tensor> {
    let half_dim = x.dim(3)? / 2;
    let x1 = x.narrow(3, 0, half_dim)?;
    let x2 = x.narrow(3, half_dim, half_dim)?;
    let cos = cos_half.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin_half.unsqueeze(0)?.unsqueeze(2)?;
    let first = x1
        .broadcast_mul(&cos)?
        .broadcast_sub(&x2.broadcast_mul(&sin)?)?;
    let second = x1
        .broadcast_mul(&sin)?
        .broadcast_add(&x2.broadcast_mul(&cos)?)?;
    Tensor::cat(&[first, second], 3).map_err(Error::from)
}

fn try_apply_granite_rope_pair_bhtd(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<Option<(Tensor, Tensor)>> {
    let kernel_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let q_heads = q.transpose(1, 2)?.contiguous()?;
        let k_heads = k.transpose(1, 2)?.contiguous()?;
        let q_out = rotary_emb::rope(&q_heads, cos, sin)?
            .transpose(1, 2)?
            .contiguous()?;
        let k_out = rotary_emb::rope(&k_heads, cos, sin)?
            .transpose(1, 2)?
            .contiguous()?;
        candle_core::Result::<(Tensor, Tensor)>::Ok((q_out, k_out))
    }));
    match kernel_result {
        Ok(Ok((q_out, k_out))) => Ok(Some((q_out, k_out))),
        Ok(Err(_)) | Err(_) => Ok(None),
    }
}

fn try_apply_granite_rope_pair_thd(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<Option<(Tensor, Tensor)>> {
    let kernel_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let q_out = rotary_emb::rope_thd(&q.contiguous()?, cos, sin)?;
        let k_out = rotary_emb::rope_thd(&k.contiguous()?, cos, sin)?;
        candle_core::Result::<(Tensor, Tensor)>::Ok((q_out, k_out))
    }));
    match kernel_result {
        Ok(Ok((q_out, k_out))) => Ok(Some((q_out, k_out))),
        Ok(Err(_)) | Err(_) => Ok(None),
    }
}

fn last_hidden_for_logits(hidden: &Tensor) -> Result<Tensor> {
    match hidden.dims() {
        [batch, seq, dim] if *seq > 0 => hidden
            .narrow(1, seq - 1, 1)?
            .reshape((*batch, 1, *dim))
            .map_err(Error::from),
        [_, 0, _] => Err(Error::InferenceError(
            "Granite Speech hidden sequence is empty".to_string(),
        )),
        dims => Err(Error::InferenceError(format!(
            "Granite Speech hidden states expected [batch,seq,dim], got {dims:?}"
        ))),
    }
}

fn text_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    start_pos: usize,
    total_len: usize,
    attention_multiplier: f32,
) -> Result<Tensor> {
    let q_heads = q.transpose(1, 2)?.contiguous()?;
    let k = repeat_kv(k, num_heads, num_kv_heads)?;
    let v = repeat_kv(v, num_heads, num_kv_heads)?;
    let k_heads = k.transpose(1, 2)?.contiguous()?;
    let v_heads = v.transpose(1, 2)?.contiguous()?;
    let mut attn = q_heads.matmul(&k_heads.t()?)?;
    attn = (attn * attention_multiplier as f64)?;
    let mask = causal_mask(q.dim(1)?, total_len, start_pos, q.device(), attn.dtype())?;
    attn = attn.broadcast_add(&mask)?;
    let attn = ops::softmax(&attn.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(q.dtype())?;
    attn.matmul(&v_heads)?
        .transpose(1, 2)
        .map_err(Error::from)
        .and_then(|out| {
            out.reshape((q.dim(0)?, q.dim(1)?, num_heads, head_dim))
                .map_err(Error::from)
        })
}

fn dense_decode_attention_scaled(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attention_multiplier: f32,
) -> Result<Tensor> {
    let k = repeat_kv(k, num_heads, num_kv_heads)?;
    let v = repeat_kv(v, num_heads, num_kv_heads)?;
    let q_heads = q.transpose(1, 2)?.contiguous()?;
    let k_heads = k.transpose(1, 2)?.contiguous()?;
    let v_heads = v.transpose(1, 2)?.contiguous()?;
    let mut attn = q_heads.matmul(&k_heads.t()?)?;
    attn = (attn * attention_multiplier as f64)?;
    let attn = ops::softmax(&attn.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(q.dtype())?;
    let out = attn.matmul(&v_heads)?;
    out.transpose(1, 2)?
        .reshape((q.dim(0)?, q.dim(1)?, num_heads, head_dim))
        .map_err(Error::from)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GraniteDenseDecodeAttentionKernel {
    Fused,
    Fallback,
}

fn dense_decode_attention_heads_scaled(
    q: &Tensor,
    k_heads: &Tensor,
    v_heads: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attention_multiplier: f32,
) -> Result<(Tensor, GraniteDenseDecodeAttentionKernel)> {
    let q_heads = q.transpose(1, 2)?.contiguous()?;
    if granite_decode_gqa_kernel_enabled(q.device(), q_heads.dtype(), k_heads.dim(2)?) {
        if let Some(out) = try_fused_decode_gqa_attention(
            &q_heads,
            k_heads,
            v_heads,
            num_heads,
            num_kv_heads,
            head_dim,
            attention_multiplier,
        ) {
            let out = out
                .transpose(1, 2)?
                .reshape((q.dim(0)?, q.dim(1)?, num_heads, head_dim))
                .map_err(Error::from)?;
            return Ok((out, GraniteDenseDecodeAttentionKernel::Fused));
        }
    }
    if let Some(out) = try_fused_self_attention_scaled(
        &q_heads,
        k_heads,
        v_heads,
        None,
        head_dim,
        false,
        attention_multiplier,
    )? {
        let out = out
            .transpose(1, 2)?
            .reshape((q.dim(0)?, q.dim(1)?, num_heads, head_dim))
            .map_err(Error::from)?;
        return Ok((out, GraniteDenseDecodeAttentionKernel::Fused));
    }
    let k_heads = if num_heads == num_kv_heads {
        k_heads.clone()
    } else {
        let k = k_heads.transpose(1, 2)?.contiguous()?;
        repeat_kv(&k, num_heads, num_kv_heads)?
            .transpose(1, 2)?
            .contiguous()?
    };
    let v_heads = if num_heads == num_kv_heads {
        v_heads.clone()
    } else {
        let v = v_heads.transpose(1, 2)?.contiguous()?;
        repeat_kv(&v, num_heads, num_kv_heads)?
            .transpose(1, 2)?
            .contiguous()?
    };
    let mut attn = q_heads.matmul(&k_heads.t()?)?;
    attn = (attn * attention_multiplier as f64)?;
    let attn = ops::softmax(&attn.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(q.dtype())?;
    let out = attn.matmul(&v_heads)?;
    let out = out
        .transpose(1, 2)?
        .reshape((q.dim(0)?, q.dim(1)?, num_heads, head_dim))
        .map_err(Error::from)?;
    Ok((out, GraniteDenseDecodeAttentionKernel::Fallback))
}

struct GraniteTextFusedGateUpProjection {
    fused: GraniteLinearNoBias,
    intermediate_size: usize,
}

impl GraniteTextFusedGateUpProjection {
    fn new(gate_proj: &Linear, up_proj: &Linear) -> Result<Self> {
        let gate_weight = gate_proj.weight();
        let up_weight = up_proj.weight();
        let intermediate_size = gate_weight.dim(0)?;
        if up_weight.dim(0)? != intermediate_size || up_weight.dim(1)? != gate_weight.dim(1)? {
            return Err(Error::InferenceError(
                "Granite fused MLP projection dimensions do not match".to_string(),
            ));
        }
        let weight = Tensor::cat(&[gate_weight, up_weight], 0)?;
        Ok(Self {
            fused: GraniteLinearNoBias::from_weight(weight)?,
            intermediate_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let gate_up = self.fused.forward(x)?;
        let last_dim = gate_up.rank().saturating_sub(1);
        let gate = gate_up.narrow(last_dim, 0, self.intermediate_size)?;
        let up = gate_up.narrow(last_dim, self.intermediate_size, self.intermediate_size)?;
        Ok((gate, up))
    }

    fn to_dtype(&self, dtype: DType) -> Result<Self> {
        Ok(Self {
            fused: GraniteLinearNoBias::from_weight(self.fused.weight().to_dtype(dtype)?)?,
            intermediate_size: self.intermediate_size,
        })
    }
}

enum GraniteTextGateUpProjection {
    Fused(GraniteTextFusedGateUpProjection),
    Separate { gate_proj: Linear, up_proj: Linear },
}

impl GraniteTextGateUpProjection {
    fn load(config: &GraniteTextConfig, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("gate_proj"),
        )?;
        let up_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("up_proj"),
        )?;
        if granite_gate_up_projection_fusion_enabled(vb.device()) {
            Ok(Self::Fused(GraniteTextFusedGateUpProjection::new(
                &gate_proj, &up_proj,
            )?))
        } else {
            Ok(Self::Separate { gate_proj, up_proj })
        }
    }

    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Fused(fused) => fused.forward(x),
            Self::Separate { gate_proj, up_proj } => {
                Ok((gate_proj.forward(x)?, up_proj.forward(x)?))
            }
        }
    }

    fn is_fused(&self) -> bool {
        matches!(self, Self::Fused(_))
    }

    fn to_dtype(&self, dtype: DType) -> Result<Self> {
        match self {
            Self::Fused(fused) => fused.to_dtype(dtype).map(Self::Fused),
            Self::Separate { gate_proj, up_proj } => Ok(Self::Separate {
                gate_proj: Linear::new(gate_proj.weight().to_dtype(dtype)?, None),
                up_proj: Linear::new(up_proj.weight().to_dtype(dtype)?, None),
            }),
        }
    }
}

struct GraniteTextMlp {
    gate_up_proj: GraniteTextGateUpProjection,
    down_proj: GraniteLinearNoBias,
    f16_mlp: Option<GraniteTextF16Mlp>,
}

struct GraniteTextF16Mlp {
    gate_up_proj: GraniteTextGateUpProjection,
    down_proj: GraniteLinearNoBias,
}

impl GraniteTextF16Mlp {
    fn new(
        gate_up_proj: &GraniteTextGateUpProjection,
        down_proj: &GraniteLinearNoBias,
    ) -> Result<Self> {
        Ok(Self {
            gate_up_proj: gate_up_proj.to_dtype(DType::F16)?,
            down_proj: GraniteLinearNoBias::from_weight(down_proj.weight().to_dtype(DType::F16)?)?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        profile: Option<&mut GraniteSpeechMlpDecodeProfile>,
    ) -> Result<Tensor> {
        let output_dtype = x.dtype();
        let x = x.to_dtype(DType::F16)?;
        let out =
            GraniteTextMlp::forward_with_parts(&self.gate_up_proj, &self.down_proj, &x, profile)?;
        if out.dtype() == output_dtype {
            Ok(out)
        } else {
            out.to_dtype(output_dtype).map_err(Error::from)
        }
    }
}

impl GraniteTextMlp {
    fn load(config: &GraniteTextConfig, vb: VarBuilder, prescale_residual: bool) -> Result<Self> {
        let down_proj = linear_no_bias(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("down_proj"),
        )?;
        let gate_up_proj = GraniteTextGateUpProjection::load(config, vb.clone())?;
        let down_proj = maybe_prescale_residual_linear(
            down_proj,
            config.residual_multiplier,
            prescale_residual,
        )?;
        let down_proj = GraniteLinearNoBias::new(down_proj)?;
        let f16_mlp = if granite_f16_mlp_enabled(vb.device(), down_proj.weight().dtype()) {
            Some(GraniteTextF16Mlp::new(&gate_up_proj, &down_proj)?)
        } else {
            None
        };
        Ok(Self {
            gate_up_proj,
            down_proj,
            f16_mlp,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        profile: Option<&mut GraniteSpeechMlpDecodeProfile>,
    ) -> Result<Tensor> {
        if let Some(f16_mlp) = self.f16_mlp.as_ref() {
            return f16_mlp.forward(x, profile);
        }
        Self::forward_with_parts(&self.gate_up_proj, &self.down_proj, x, profile)
    }

    fn forward_with_parts(
        gate_up_proj: &GraniteTextGateUpProjection,
        down_proj: &GraniteLinearNoBias,
        x: &Tensor,
        mut profile: Option<&mut GraniteSpeechMlpDecodeProfile>,
    ) -> Result<Tensor> {
        let profiling = profile.is_some();
        let gate_up_start = profile_start(profiling);
        let (gate, up) = gate_up_proj.forward(x)?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.gate_up += profile_elapsed(gate_up_start);
        }
        let activation_start = profile_start(profiling);
        let try_fused_silu_mul = granite_mlp_try_fused_silu_mul(gate.device());
        if let Some(profile) = profile.as_deref_mut() {
            if try_fused_silu_mul {
                profile.fused_silu_mul_attempts += 1;
            }
        }
        let hidden = if try_fused_silu_mul {
            if let Some(fused) = try_fused_silu_mul_with_status(&gate, &up) {
                if let Some(profile) = profile.as_deref_mut() {
                    if fused.used_custom_kernel {
                        profile.fused_silu_mul_custom += 1;
                    } else {
                        profile.fused_silu_mul_fallback += 1;
                    }
                }
                fused.tensor
            } else {
                if let Some(profile) = profile.as_deref_mut() {
                    profile.fused_silu_mul_fallback += 1;
                }
                let gate = gate.silu()?;
                gate.broadcast_mul(&up)?
            }
        } else {
            let gate = gate.silu()?;
            gate.broadcast_mul(&up)?
        };
        if let Some(profile) = profile.as_deref_mut() {
            profile.activation += profile_elapsed(activation_start);
        }
        let down_start = profile_start(profiling);
        let out = down_proj.forward(&hidden)?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.down += profile_elapsed(down_start);
        }
        Ok(out)
    }

    fn gate_up_projection_fused(&self) -> bool {
        self.gate_up_proj.is_fused()
    }

    fn f16_mlp(&self) -> bool {
        self.f16_mlp.is_some()
    }
}

fn maybe_prescale_residual_linear(
    linear: Linear,
    residual_multiplier: f32,
    enabled: bool,
) -> Result<Linear> {
    if !enabled || residual_multiplier == 1.0 {
        return Ok(linear);
    }
    if linear.bias().is_some() {
        return Err(Error::InferenceError(
            "Granite residual pre-scaling only supports bias-free projections".to_string(),
        ));
    }
    let weight = (linear.weight() * residual_multiplier as f64)?;
    Ok(Linear::new(weight, None))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_tensor_close(lhs: &Tensor, rhs: &Tensor) {
        assert_eq!(lhs.dims(), rhs.dims());
        let lhs = lhs
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let rhs = rhs
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (idx, (lhs, rhs)) in lhs.iter().zip(rhs.iter()).enumerate() {
            assert!(
                (lhs - rhs).abs() <= 1e-5,
                "tensor mismatch at {idx}: {lhs} != {rhs}"
            );
        }
    }

    #[test]
    fn fused_granite_qkv_projection_matches_separate_linears() {
        let device = Device::Cpu;
        let q_proj = Linear::new(
            Tensor::from_vec(
                vec![
                    0.1f32, 0.2, 0.3, -0.4, 0.5, 0.6, 0.7, -0.8, 0.9, 1.0, -1.1, 1.2,
                ],
                (4, 3),
                &device,
            )
            .unwrap(),
            None,
        );
        let k_proj = Linear::new(
            Tensor::from_vec(vec![0.3f32, -0.2, 0.1, 0.6, 0.4, -0.5], (2, 3), &device).unwrap(),
            None,
        );
        let v_proj = Linear::new(
            Tensor::from_vec(vec![-0.7f32, 0.8, 0.9, 0.2, -0.1, 0.4], (2, 3), &device).unwrap(),
            None,
        );
        let fused = GraniteTextFusedQkvProjection::new(&q_proj, &k_proj, &v_proj).unwrap();
        let input = Tensor::from_vec(
            vec![1.0f32, -2.0, 0.5, 0.25, 1.5, -0.75],
            (1, 2, 3),
            &device,
        )
        .unwrap();

        let (q, k, v) = fused.forward(&input).unwrap();

        assert_tensor_close(&q, &q_proj.forward(&input).unwrap());
        assert_tensor_close(&k, &k_proj.forward(&input).unwrap());
        assert_tensor_close(&v, &v_proj.forward(&input).unwrap());
    }

    #[test]
    fn fused_granite_gate_up_projection_matches_separate_linears() {
        let device = Device::Cpu;
        let gate_proj = Linear::new(
            Tensor::from_vec(
                vec![0.1f32, -0.2, 0.3, 0.4, 0.5, -0.6, -0.7, 0.8, 0.9],
                (3, 3),
                &device,
            )
            .unwrap(),
            None,
        );
        let up_proj = Linear::new(
            Tensor::from_vec(
                vec![-0.3f32, 0.2, 0.1, 0.6, -0.4, 0.5, 0.7, 0.9, -0.8],
                (3, 3),
                &device,
            )
            .unwrap(),
            None,
        );
        let fused = GraniteTextFusedGateUpProjection::new(&gate_proj, &up_proj).unwrap();
        let input = Tensor::from_vec(
            vec![1.0f32, -2.0, 0.5, 0.25, 1.5, -0.75],
            (1, 2, 3),
            &device,
        )
        .unwrap();

        let (gate, up) = fused.forward(&input).unwrap();

        assert_tensor_close(&gate, &gate_proj.forward(&input).unwrap());
        assert_tensor_close(&up, &up_proj.forward(&input).unwrap());
    }

    #[test]
    fn granite_rope_cache_slice_matches_direct_build() {
        let device = Device::Cpu;
        let cache = GraniteRopeCache::build(16, 4, 10000.0, &device, DType::F32).unwrap();
        let cached = cache.slice(5, 3).unwrap();
        let (direct_cos, direct_sin) =
            build_rope_cache(3, 4, 5, 10000.0, &device, DType::F32).unwrap();
        let direct_packed = Tensor::cat(&[&direct_cos, &direct_sin], 1).unwrap();

        assert_tensor_close(&cached.cos, &direct_cos);
        assert_tensor_close(&cached.sin, &direct_sin);
        assert_tensor_close(&cached.packed, &direct_packed);
    }

    #[test]
    fn granite_projection_fusion_defaults_to_accelerated_backends() {
        assert!(granite_projection_fusion_policy(true, false, None));
        assert!(granite_projection_fusion_policy(false, true, None));
        assert!(granite_projection_fusion_policy(true, true, None));
        assert!(!granite_projection_fusion_policy(false, false, None));
        assert!(granite_projection_fusion_policy(false, false, Some(true)));
        assert!(!granite_projection_fusion_policy(true, true, Some(false)));
    }

    #[test]
    fn granite_gate_up_projection_fusion_defaults_to_accelerated_backends() {
        assert!(granite_gate_up_projection_fusion_policy(true, false, None));
        assert!(granite_gate_up_projection_fusion_policy(false, true, None));
        assert!(!granite_gate_up_projection_fusion_policy(
            false, false, None
        ));
        assert!(granite_gate_up_projection_fusion_policy(
            false,
            false,
            Some(true)
        ));
        assert!(!granite_gate_up_projection_fusion_policy(
            true,
            false,
            Some(false)
        ));
    }

    #[test]
    fn granite_mlp_fused_silu_mul_defaults_on_with_override() {
        assert!(granite_mlp_fused_silu_mul_policy(true, None));
        assert!(granite_mlp_fused_silu_mul_policy(false, None));
        assert!(granite_mlp_fused_silu_mul_policy(true, Some(true)));
        assert!(!granite_mlp_fused_silu_mul_policy(true, Some(false)));
    }

    #[test]
    fn granite_fused_rms_norm_defaults_to_metal_only() {
        assert!(granite_fused_rms_norm_policy(true, None));
        assert!(!granite_fused_rms_norm_policy(false, None));
        assert!(granite_fused_rms_norm_policy(false, Some(true)));
        assert!(granite_fused_rms_norm_policy(true, Some(true)));
        assert!(!granite_fused_rms_norm_policy(true, Some(false)));
    }

    #[test]
    fn granite_residual_prescale_policy_defaults_to_metal_only() {
        assert!(granite_residual_prescale_policy(true, None));
        assert!(!granite_residual_prescale_policy(false, None));
        assert!(granite_residual_prescale_policy(false, Some(true)));
        assert!(granite_residual_prescale_policy(true, Some(true)));
        assert!(!granite_residual_prescale_policy(true, Some(false)));
    }

    #[test]
    fn granite_dense_decode_preallocate_defaults_to_metal_only() {
        assert!(granite_dense_decode_preallocate_policy(true, None));
        assert!(!granite_dense_decode_preallocate_policy(false, None));
        assert!(granite_dense_decode_preallocate_policy(false, Some(true)));
        assert!(!granite_dense_decode_preallocate_policy(true, Some(false)));
    }

    #[test]
    fn residual_prescaled_linear_matches_post_projection_scale() {
        let device = Device::Cpu;
        let weight = Tensor::from_vec(
            vec![
                0.5f32, -1.0, 0.25, //
                1.5, 0.75, -0.5,
            ],
            (2, 3),
            &device,
        )
        .unwrap();
        let input = Tensor::from_vec(vec![2.0f32, -3.0, 4.0], (1, 1, 3), &device).unwrap();
        let linear = Linear::new(weight, None);
        let residual_multiplier = 0.22f32;

        let explicit = (linear.forward(&input).unwrap() * residual_multiplier as f64).unwrap();
        let prescaled = maybe_prescale_residual_linear(linear, residual_multiplier, true).unwrap();
        let folded = prescaled.forward(&input).unwrap();

        assert_tensor_close(&explicit, &folded);
    }

    #[test]
    fn granite_rope_kernel_defaults_to_cuda_only() {
        assert!(!granite_rope_kernel_policy(true, false, None));
        assert!(granite_rope_kernel_policy(false, true, None));
        assert!(!granite_rope_kernel_policy(false, false, None));
        assert!(granite_rope_kernel_policy(false, false, Some(true)));
        assert!(!granite_rope_kernel_policy(true, true, Some(false)));
    }

    #[test]
    fn granite_rope_bhtd_kernel_defaults_to_metal_only() {
        assert!(granite_rope_bhtd_kernel_policy(true, None));
        assert!(!granite_rope_bhtd_kernel_policy(false, None));
        assert!(granite_rope_bhtd_kernel_policy(false, Some(true)));
        assert!(!granite_rope_bhtd_kernel_policy(true, Some(false)));
    }

    #[test]
    fn granite_rope_pair_bshd_kernel_defaults_to_metal_only() {
        assert!(granite_rope_pair_bshd_kernel_policy(true, None));
        assert!(!granite_rope_pair_bshd_kernel_policy(false, None));
        assert!(granite_rope_pair_bshd_kernel_policy(false, Some(true)));
        assert!(!granite_rope_pair_bshd_kernel_policy(true, Some(false)));
    }

    #[test]
    fn granite_manual_rotary_matches_rope_thd() {
        let device = Device::Cpu;
        let seq_len = 4usize;
        let head_dim = 8usize;
        let x = Tensor::from_vec(
            (0..(seq_len * 2 * head_dim))
                .map(|v| (v as f32) * 0.01)
                .collect::<Vec<_>>(),
            (1, seq_len, 2, head_dim),
            &device,
        )
        .expect("x");
        let (cos, sin) =
            build_rope_cache(seq_len, head_dim, 0, 10000.0, &device, DType::F32).expect("cache");

        let manual = apply_rotary_emb(&x, &cos, &sin).expect("manual");
        let (kernel, _) = try_apply_granite_rope_pair_thd(
            &x,
            &x,
            &cos.unsqueeze(0).expect("cos").contiguous().expect("cos"),
            &sin.unsqueeze(0).expect("sin").contiguous().expect("sin"),
        )
        .expect("kernel result")
        .expect("kernel output");

        assert_tensor_close(&manual, &kernel);
    }

    #[test]
    fn granite_manual_rotary_matches_rope_bhtd() {
        let device = Device::Cpu;
        let seq_len = 4usize;
        let head_dim = 8usize;
        let q_heads = 2usize;
        let k_heads = 1usize;
        let q = Tensor::from_vec(
            (0..(seq_len * q_heads * head_dim))
                .map(|v| (v as f32) * 0.01)
                .collect::<Vec<_>>(),
            (1, seq_len, q_heads, head_dim),
            &device,
        )
        .expect("q");
        let k = Tensor::from_vec(
            (0..(seq_len * k_heads * head_dim))
                .map(|v| (v as f32) * -0.02)
                .collect::<Vec<_>>(),
            (1, seq_len, k_heads, head_dim),
            &device,
        )
        .expect("k");
        let (cos, sin) =
            build_rope_cache(seq_len, head_dim, 0, 10000.0, &device, DType::F32).expect("cache");

        let q_manual = apply_rotary_emb(&q, &cos, &sin).expect("manual q");
        let k_manual = apply_rotary_emb(&k, &cos, &sin).expect("manual k");
        let (q_kernel, k_kernel) = try_apply_granite_rope_pair_bhtd(&q, &k, &cos, &sin)
            .expect("kernel result")
            .expect("kernel output");

        assert_tensor_close(&q_manual, &q_kernel);
        assert_tensor_close(&k_manual, &k_kernel);
    }

    #[test]
    fn expands_single_audio_placeholder_to_projected_token_count() {
        let input = vec![10, 20, 30];
        let expanded = expand_audio_tokens(&input, &[1], 3, 99).unwrap();
        assert_eq!(expanded, vec![10, 99, 99, 99, 30]);
    }

    #[test]
    fn rejects_multiple_audio_placeholders_for_single_audio_input() {
        let err = expand_audio_tokens(&[1, 2], &[0, 1], 3, 99).unwrap_err();
        assert!(format!("{err}").contains("exactly one audio placeholder"));
    }

    #[test]
    fn relative_position_score_row_handles_head_batched_query_rows() {
        let device = Device::Cpu;
        let q = Tensor::zeros((8, 200, 128), DType::F32, &device).unwrap();
        let q_row = q.narrow(1, 17, 1).unwrap();
        let rel_row = Tensor::zeros((200, 128), DType::F32, &device).unwrap();

        assert!(!q_row.squeeze(1).unwrap().is_contiguous());
        let scores = relative_position_score_row(&q_row, &rel_row).unwrap();

        assert_eq!(scores.dims(), &[8, 1, 200]);
    }

    #[test]
    fn encoder_attention_heads_compacts_transposed_context_layout() {
        let device = Device::Cpu;
        let projected = Tensor::zeros((1, 200, 1024), DType::F32, &device).unwrap();
        let strided = projected
            .reshape((1, 1, 200, 8, 128))
            .unwrap()
            .transpose(2, 3)
            .unwrap();

        assert_eq!(strided.dims(), &[1, 1, 8, 200, 128]);
        assert!(!strided.is_contiguous());

        let heads = encoder_attention_heads(&projected, 1, 1, 200, 8, 128).unwrap();

        assert_eq!(heads.dims(), &[1, 1, 8, 200, 128]);
        assert!(heads.is_contiguous());
        assert_eq!(heads.stride(), &[204800, 204800, 25600, 128, 1]);
        assert_eq!(
            heads.t().unwrap().stride(),
            &[204800, 204800, 25600, 1, 128]
        );
    }

    #[test]
    fn encoder_attention_scores_handle_cpu_five_dim_heads() {
        let device = Device::Cpu;
        let q = Tensor::zeros((1, 2, 3, 4, 5), DType::F32, &device).unwrap();
        let k = Tensor::zeros((1, 2, 3, 4, 5), DType::F32, &device).unwrap();

        let scores = encoder_attention_scores(&q, &k).unwrap();

        assert_eq!(scores.dims(), &[1, 2, 3, 4, 4]);
    }

    #[test]
    fn encoder_attention_context_handles_cpu_five_dim_heads() {
        let device = Device::Cpu;
        let attn = Tensor::zeros((1, 2, 3, 4, 4), DType::F32, &device).unwrap();
        let v = Tensor::zeros((1, 2, 3, 4, 5), DType::F32, &device).unwrap();

        let context = encoder_attention_context(&attn, &v).unwrap();

        assert_eq!(context.dims(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn encoder_padding_mask_allocates_each_block_and_masks_final_block() {
        let device = Device::Cpu;
        let mask = encoder_padding_mask(4, 2, 2, &device, DType::F32).unwrap();
        let values = mask.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(mask.dims(), &[1, 2, 1, 4, 4]);
        assert_eq!(values.len(), 32);
        assert!(values[..16].iter().all(|value| *value == 0.0));
        assert_eq!(values[16], 0.0);
        assert_eq!(values[17], 0.0);
        assert_eq!(values[18], -1e4);
        assert_eq!(values[20], 0.0);
        assert_eq!(values[24], -1e4);
    }

    #[test]
    fn qformer_attention_heads_compacts_transposed_sequence_layout() {
        let device = Device::Cpu;
        let projected = Tensor::zeros((2, 15, 1024), DType::F32, &device).unwrap();
        let strided = projected
            .reshape((2, 15, 16, 64))
            .unwrap()
            .transpose(1, 2)
            .unwrap();

        assert_eq!(strided.dims(), &[2, 16, 15, 64]);
        assert!(!strided.is_contiguous());

        let heads = qformer_attention_heads(&projected, 16, 64).unwrap();

        assert_eq!(heads.dims(), &[2, 16, 15, 64]);
        assert!(heads.is_contiguous());
        assert_eq!(heads.stride(), &[15360, 960, 64, 1]);
        assert_eq!(heads.t().unwrap().stride(), &[15360, 960, 1, 64]);
    }

    #[test]
    fn dense_head_decode_matches_materialized_scaled_attention() {
        let device = Device::Cpu;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 2;
        let attention_multiplier = 0.37;
        let q = Tensor::from_vec(
            vec![0.2f32, -0.4, 0.1, 0.8, -0.3, 0.7, 0.5, -0.6],
            (1, 1, num_heads, head_dim),
            &device,
        )
        .unwrap();
        let k = Tensor::from_vec(
            vec![
                0.1f32, 0.2, -0.3, 0.4, 0.5, -0.2, //
                -0.6, 0.7, 0.2, -0.1, 0.3, 0.9,
            ],
            (1, 3, num_kv_heads, head_dim),
            &device,
        )
        .unwrap();
        let v = Tensor::from_vec(
            vec![
                -0.2f32, 0.4, 0.6, -0.1, 0.8, 0.3, //
                0.5, -0.7, -0.4, 0.2, 0.1, 0.9,
            ],
            (1, 3, num_kv_heads, head_dim),
            &device,
        )
        .unwrap();
        let k_heads = k.transpose(1, 2).unwrap().contiguous().unwrap();
        let v_heads = v.transpose(1, 2).unwrap().contiguous().unwrap();

        let materialized = dense_decode_attention_scaled(
            &q,
            &k,
            &v,
            num_heads,
            num_kv_heads,
            head_dim,
            attention_multiplier,
        )
        .unwrap();
        let (dense_heads, _path) = dense_decode_attention_heads_scaled(
            &q,
            &k_heads,
            &v_heads,
            num_heads,
            num_kv_heads,
            head_dim,
            attention_multiplier,
        )
        .unwrap();
        let lhs = materialized
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let rhs = dense_heads.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let max_diff = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(max_diff < 1e-6, "max diff {max_diff}");
    }

    #[test]
    fn dense_head_decode_policy_skips_cpu() {
        assert!(!granite_dense_head_decode_allowed(&Device::Cpu));
    }

    #[test]
    fn dense_head_decode_policy_enables_accelerated_backends() {
        assert!(!granite_dense_head_decode_policy(false, false));
        assert!(granite_dense_head_decode_policy(true, false));
        assert!(granite_dense_head_decode_policy(false, true));
        assert!(granite_dense_head_decode_policy(true, true));
    }

    #[test]
    fn granite_decode_gqa_kernel_defaults_to_metal_only() {
        assert!(granite_decode_gqa_kernel_policy(true, None));
        assert!(!granite_decode_gqa_kernel_policy(false, None));
        assert!(granite_decode_gqa_kernel_policy(false, Some(true)));
        assert!(!granite_decode_gqa_kernel_policy(true, Some(false)));
    }

    #[test]
    fn qformer_fused_attention_policy_skips_cpu() {
        assert!(!granite_qformer_fused_attention_allowed(&Device::Cpu));
    }

    #[test]
    fn qformer_attention_cpu_fallback_preserves_shape() {
        let device = Device::Cpu;
        let hidden = 8;
        let heads = 2;
        let linear = || {
            Linear::new(
                Tensor::zeros((hidden, hidden), DType::F32, &device).unwrap(),
                Some(Tensor::zeros(hidden, DType::F32, &device).unwrap()),
            )
        };
        let attention = GraniteQFormerAttention {
            query: linear(),
            key: linear(),
            value: linear(),
            output_dense: linear(),
            output_norm: LayerNorm::new(
                Tensor::ones(hidden, DType::F32, &device).unwrap(),
                Tensor::zeros(hidden, DType::F32, &device).unwrap(),
                1e-12,
            ),
            num_heads: heads,
            head_dim: hidden / heads,
        };
        let x = Tensor::zeros((3, 5, hidden), DType::F32, &device).unwrap();
        let out = attention.forward(&x, None).unwrap();

        assert_eq!(out.dims(), &[3, 5, hidden]);
    }

    #[test]
    fn argmax_last_logits_uses_last_prefill_position() {
        let logits = Tensor::from_vec(
            vec![
                0.0f32, 100.0, 0.0, 0.0, //
                0.0, 0.0, 90.0, 0.0, //
                0.0, 0.0, 0.0, 7.0,
            ],
            (1, 3, 4),
            &Device::Cpu,
        )
        .unwrap();

        assert_eq!(argmax_last_logits(&logits).unwrap(), 3);
    }

    #[test]
    fn argmax_last_token_tensor_matches_scalar_argmax() {
        let logits = Tensor::from_vec(
            vec![
                0.0f32, 100.0, 0.0, 0.0, //
                0.0, 0.0, 90.0, 0.0, //
                0.0, 0.0, 0.0, 7.0,
            ],
            (1, 3, 4),
            &Device::Cpu,
        )
        .unwrap();

        let token = argmax_last_token_tensor(&logits).unwrap();

        assert_eq!(token.dims(), &[1, 1]);
        assert_eq!(token.reshape(()).unwrap().to_scalar::<u32>().unwrap(), 3);
    }

    #[test]
    fn argmax_last_token_is_unchanged_by_positive_logit_scaling() {
        let logits = Tensor::from_vec(
            vec![
                -1.0f32, 0.5, 0.0, 0.25, //
                0.0, -4.0, 8.0, 2.0,
            ],
            (1, 2, 4),
            &Device::Cpu,
        )
        .unwrap();
        let scaled = (logits.clone() / 8.0)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();

        assert_eq!(
            argmax_last_logits(&logits).unwrap(),
            argmax_last_logits(&scaled).unwrap()
        );
    }

    #[test]
    fn argmax_last_logits_rejects_empty_prefill_sequence() {
        let logits = Tensor::zeros((1, 0, 4), DType::F32, &Device::Cpu).unwrap();
        let err = argmax_last_logits(&logits).unwrap_err();

        assert!(format!("{err}").contains("logits sequence is empty"));
    }

    #[test]
    fn f16_lm_head_policy_defaults_to_metal_f32_only() {
        assert!(granite_f16_lm_head_policy(true, DType::F32, None));
        assert!(!granite_f16_lm_head_policy(true, DType::F16, None));
        assert!(!granite_f16_lm_head_policy(false, DType::F32, None));
        assert!(granite_f16_lm_head_policy(false, DType::F32, Some(true)));
        assert!(!granite_f16_lm_head_policy(true, DType::F32, Some(false)));
    }

    #[test]
    fn rope_cache_dtype_tracks_attention_core_dtype() {
        assert!(granite_rope_cache_attention_dtype_policy(true, None));
        assert!(!granite_rope_cache_attention_dtype_policy(false, None));
        assert!(granite_rope_cache_attention_dtype_policy(
            false,
            Some(true)
        ));
        assert!(!granite_rope_cache_attention_dtype_policy(
            true,
            Some(false)
        ));
        assert_eq!(
            granite_rope_cache_dtype(DType::F32, false, true),
            DType::F32
        );
        assert_eq!(
            granite_rope_cache_dtype(DType::F16, false, true),
            DType::F16
        );
        assert_eq!(
            granite_rope_cache_dtype(DType::F32, true, true),
            DType::F16
        );
        assert_eq!(
            granite_rope_cache_dtype(DType::F32, true, false),
            DType::F32
        );
    }

    #[test]
    fn cached_linear_t_policy_defaults_to_metal_only() {
        assert!(granite_cached_linear_t_policy(true, None));
        assert!(!granite_cached_linear_t_policy(false, None));
        assert!(granite_cached_linear_t_policy(false, Some(true)));
        assert!(!granite_cached_linear_t_policy(true, Some(false)));
    }

    #[test]
    fn cached_linear_no_bias_forward_matches_linear() {
        let device = Device::Cpu;
        let weight = Tensor::from_vec(
            vec![
                0.5f32, -1.0, 0.25, //
                1.5, 0.75, -0.5,
            ],
            (2, 3),
            &device,
        )
        .unwrap();
        let input = Tensor::from_vec(vec![2.0f32, -3.0, 4.0], (1, 1, 3), &device).unwrap();
        let linear = Linear::new(weight.clone(), None);
        let cached = GraniteLinearNoBias {
            linear: linear.clone(),
            weight_t: Some(weight.t().unwrap()),
        };

        assert_tensor_close(
            &cached.forward(&input).unwrap(),
            &linear.forward(&input).unwrap(),
        );
    }

    #[test]
    fn f16_mlp_policy_defaults_to_metal_f32_only() {
        assert!(granite_f16_mlp_policy(true, DType::F32, None));
        assert!(!granite_f16_mlp_policy(true, DType::F16, None));
        assert!(!granite_f16_mlp_policy(false, DType::F32, None));
        assert!(granite_f16_mlp_policy(false, DType::F32, Some(true)));
        assert!(!granite_f16_mlp_policy(true, DType::F32, Some(false)));
    }

    #[test]
    fn f16_qkv_policy_defaults_to_metal_f32_only() {
        assert!(granite_f16_qkv_policy(true, DType::F32, None));
        assert!(!granite_f16_qkv_policy(true, DType::F16, None));
        assert!(!granite_f16_qkv_policy(false, DType::F32, None));
        assert!(granite_f16_qkv_policy(false, DType::F32, Some(true)));
        assert!(!granite_f16_qkv_policy(true, DType::F32, Some(false)));
    }

    #[test]
    fn f16_attention_core_policy_defaults_to_metal_f32_only() {
        assert!(granite_f16_attention_core_policy(true, DType::F32, None));
        assert!(!granite_f16_attention_core_policy(true, DType::F16, None));
        assert!(!granite_f16_attention_core_policy(false, DType::F32, None));
        assert!(granite_f16_attention_core_policy(
            false,
            DType::F32,
            Some(true)
        ));
        assert!(!granite_f16_attention_core_policy(
            true,
            DType::F32,
            Some(false)
        ));
    }

    #[test]
    fn f16_attention_output_policy_defaults_to_metal_f32_only() {
        assert!(granite_f16_attention_output_policy(true, DType::F32, None));
        assert!(!granite_f16_attention_output_policy(true, DType::F16, None));
        assert!(!granite_f16_attention_output_policy(
            false,
            DType::F32,
            None
        ));
        assert!(granite_f16_attention_output_policy(
            false,
            DType::F32,
            Some(true)
        ));
        assert!(!granite_f16_attention_output_policy(
            true,
            DType::F32,
            Some(false)
        ));
    }

    #[test]
    fn stop_token_set_deduplicates_eos_pad_and_extra_ids() {
        let tokens = GraniteSpeechSpecialTokens {
            bos_token_id: 1,
            eos_token_id: 2,
            pad_token_id: 2,
            audio_token_id: 3,
            audio_token: "<|audio|>".to_string(),
        };
        assert_eq!(stop_token_set(&tokens, &[4, 2, 4]), vec![2, 4]);
    }

    #[test]
    fn deferred_stop_check_policy_defaults_to_short_metal_decodes() {
        assert!(granite_deferred_stop_check_policy(true, None, 76, 128));
        assert!(!granite_deferred_stop_check_policy(true, None, 129, 128));
        assert!(!granite_deferred_stop_check_policy(false, None, 76, 128));
        assert!(granite_deferred_stop_check_policy(
            false,
            Some(true),
            4096,
            128
        ));
        assert!(!granite_deferred_stop_check_policy(
            true,
            Some(false),
            76,
            128
        ));
    }

    #[test]
    fn chunked_stop_check_policy_defaults_to_short_metal_decodes() {
        assert!(granite_chunked_stop_check_policy(true, None, 76, 128));
        assert!(granite_chunked_stop_check_policy(true, None, 128, 128));
        assert!(!granite_chunked_stop_check_policy(true, None, 129, 128));
        assert!(!granite_chunked_stop_check_policy(false, None, 76, 128));
        assert!(granite_chunked_stop_check_policy(
            false,
            Some(true),
            4096,
            128
        ));
        assert!(!granite_chunked_stop_check_policy(
            true,
            Some(false),
            76,
            128
        ));
    }

    #[test]
    fn deferred_stop_tokens_match_incremental_stop_semantics() {
        let (tokens, stop) = truncate_tokens_at_first_stop(vec![10, 11, 2, 12], &[2, 4]);

        assert_eq!(tokens, vec![10, 11]);
        assert_eq!(stop, Some(2));
    }

    #[test]
    fn deferred_stop_tokens_preserve_max_token_outputs_without_stop() {
        let (tokens, stop) = truncate_tokens_at_first_stop(vec![10, 11, 12], &[2, 4]);

        assert_eq!(tokens, vec![10, 11, 12]);
        assert_eq!(stop, None);
    }

    #[test]
    fn collect_deferred_token_tensors_returns_single_host_read_row() {
        let device = Device::Cpu;
        let first = Tensor::from_vec(vec![7u32], (1, 1), &device).unwrap();
        let second = Tensor::from_vec(vec![8u32], (1, 1), &device).unwrap();

        assert_eq!(
            collect_deferred_token_tensors(&[first, second]).unwrap(),
            vec![7, 8]
        );
    }

    #[test]
    fn stop_sequence_truncates_generated_text() {
        let mut text = "hello<stop>ignored".to_string();
        assert!(truncate_at_stop_sequence(
            &mut text,
            &["<stop>".to_string()]
        ));
        assert_eq!(text, "hello");
    }
}
