mod nemo;
#[cfg(test)]
mod parity_tests;
mod physical;

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::ops;
use candle_nn::{
    batch_norm, layer_norm, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, LayerNorm, Linear, Module,
    VarBuilder,
};
use izwi_vad::{speech_mask_for_frames_f32, VadRegionConfig};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::backends::state::{InvocationTensorComponentValue, InvocationTensorUpdateV2};
use crate::backends::{BackendKind, DeviceProfile};
use crate::engine::{InvocationTensorLease, StageDescriptor};
use crate::error::{Error, Result};
use crate::kernels::depthwise_conv::{
    batch_norm_affine, depthwise_conv1d, depthwise_conv2d, fold_channel_affine,
    DepthwiseConv1dSpec, DepthwiseConv2dSpec,
};
use crate::kv::v2::{
    ComponentShapeInstantiation, DomainStepIntent, ShapeAxis, ShapeDimensionValue,
    StateComponentId, StateUpdateKind,
};
use crate::model::ModelVariant;
use crate::models::shared::weights::mlx;
use crate::runtime::{DiarizationConfig, DiarizationResult, DiarizationSegment};

use nemo::SortformerArtifacts;
pub(crate) use nemo::{
    ensure_sortformer_artifacts, sortformer_checkpoint_paths, sortformer_extraction_is_current,
};
pub(crate) use physical::SortformerPhysicalStateSpec;

const TARGET_SAMPLE_RATE: u32 = 16_000;
const DEFAULT_MIN_SPEECH_MS: f32 = 240.0;
const DEFAULT_MIN_SILENCE_MS: f32 = 200.0;
const PREEMPH: f32 = 0.97;
const LOG_GUARD: f32 = 5.960_464_5e-8;
const NORMALIZE_EPS: f32 = 1e-5;
const TS_VAD_FRAME_LENGTH_SECS: f32 = 0.01;
const TS_VAD_UNIT_FRAME_COUNT: usize = 8;
const PRODUCTION_FEATURE_BINS: usize = 128;
const PRODUCTION_N_FFT: usize = 512;
const PRODUCTION_HOP_LENGTH: usize = 160;
const PRODUCTION_CONV_CHANNELS: usize = 256;
const PRODUCTION_CONFORMER_LAYERS: usize = 17;
const PRODUCTION_CONFORMER_D_MODEL: usize = 512;
const PRODUCTION_CONFORMER_FF_DIM: usize = 2048;
const PRODUCTION_CONFORMER_HEADS: usize = 8;
const PRODUCTION_TRANSFORMER_LAYERS: usize = 18;
const PRODUCTION_TRANSFORMER_D_MODEL: usize = 192;
const PRODUCTION_TRANSFORMER_INNER_DIM: usize = 768;
const PRODUCTION_TRANSFORMER_HEADS: usize = 8;
const PRODUCTION_MAX_CHUNK_LEN: usize = 340;
const PRODUCTION_MAX_CHUNK_LEFT_CONTEXT: usize = 1;
const PRODUCTION_MAX_CHUNK_RIGHT_CONTEXT: usize = 40;
const PRODUCTION_MAX_SPKCACHE_LEN: usize = 188;
const PRODUCTION_MAX_FIFO_LEN: usize = 188;
// Nemotron-3-Diarization: NeMo `TransformerEncoder` with `feature_stacking`
// subsampling and RoPE self-attention, plus a subpixel upsampler that
// expands the encoded 80 ms stream to the 10 ms output rate.
const NEMOTRON3_ROPE_LAYERS: usize = 31;
const NEMOTRON3_ROPE_D_MODEL: usize = 512;
const NEMOTRON3_ROPE_FF_DIM: usize = 2048;
const NEMOTRON3_ROPE_HEADS: usize = 8;
const NEMOTRON3_ROPE_MAX_POSITIONS: usize = 5_000;
const NEMOTRON3_MAX_CHUNK_LEN: usize = 340;
const NEMOTRON3_MAX_CHUNK_RIGHT_CONTEXT: usize = 40;
const NEMOTRON3_MAX_SPKCACHE_LEN: usize = 264;
const NEMOTRON3_MAX_FIFO_LEN: usize = 264;
const NEMOTRON3_HEAD_D_MODEL: usize = 192;
// The formulas below explicitly count the largest live tensors in each
// model stage. A 1.5x factor covers what they cannot see: allocator retention
// and kernel scratch. Measured on the real checkpoints (CPU, release, one
// diarize call that fills the cache and runs full chunks; see
// `sortformer_workspace_measured_peak_per_profile`), the process footprint
// grew 3.8x the live malloc bytes on v2.1 (allocator retention), and the
// estimate is 1.1-1.25x that footprint for every v2.1 profile.
const SORTFORMER_WORKSPACE_SAFETY_PERCENT: u128 = 150;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SortformerWorkspaceEstimate {
    pub host_bytes: u64,
    pub accelerator_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortformerWorkspaceEvent {
    Materialized {
        workspace: SortformerWorkspaceEstimate,
    },
    Releasing {
        workspace: SortformerWorkspaceEstimate,
    },
}

struct SortformerWorkspaceGuard<'a, F>
where
    F: FnMut(SortformerWorkspaceEvent) -> Result<()>,
{
    observer: &'a mut F,
    workspace: SortformerWorkspaceEstimate,
    active: bool,
}

impl<'a, F> SortformerWorkspaceGuard<'a, F>
where
    F: FnMut(SortformerWorkspaceEvent) -> Result<()>,
{
    fn new(observer: &'a mut F, workspace: SortformerWorkspaceEstimate) -> Result<Self> {
        observer(SortformerWorkspaceEvent::Materialized { workspace })?;
        Ok(Self {
            observer,
            workspace,
            active: true,
        })
    }

    fn release(mut self) -> Result<()> {
        let result = (self.observer)(SortformerWorkspaceEvent::Releasing {
            workspace: self.workspace,
        });
        if result.is_ok() {
            self.active = false;
        }
        result
    }
}

impl<F> Drop for SortformerWorkspaceGuard<'_, F>
where
    F: FnMut(SortformerWorkspaceEvent) -> Result<()>,
{
    fn drop(&mut self) {
        if self.active {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                (self.observer)(SortformerWorkspaceEvent::Releasing {
                    workspace: self.workspace,
                })
            }));
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
struct SortformerModelConfig {
    sample_rate: Option<u32>,
    max_num_of_spks: Option<usize>,
    streaming_mode: Option<bool>,
    /// NeMo `SortformerEncLabelModel`: probability rows at 10 ms instead of
    /// the encoded rate (absent: false).
    high_resolution: Option<bool>,
    /// Output frame period in 10 ms units (absent: the encoder's
    /// subsampling factor).
    output_subsampling_factor: Option<usize>,
    async_streaming: Option<bool>,
    preprocessor: Option<SortformerPreprocessorConfig>,
    encoder: Option<SortformerEncoderConfig>,
    transformer_encoder: Option<SortformerTransformerConfig>,
    sortformer_modules: Option<SortformerModulesConfig>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct SortformerPreprocessorConfig {
    sample_rate: Option<u32>,
    window_size: Option<f32>,
    window_stride: Option<f32>,
    features: Option<usize>,
    n_fft: Option<usize>,
    normalize: Option<String>,
}

/// The graph-defining fields of the `encoder` section: NeMo's
/// `ConformerEncoder` (v2.1) or `TransformerEncoder` (Nemotron-3). An absent
/// field takes that class's default, which is what NeMo builds.
#[derive(Debug, Clone, Default, serde::Deserialize)]
struct SortformerEncoderConfig {
    #[serde(rename = "_target_")]
    target: Option<String>,
    n_heads: Option<usize>,
    xscaling: Option<bool>,
    subsampling: Option<String>,
    subsampling_factor: Option<usize>,
    self_attention_model: Option<String>,
    causal_downsampling: Option<bool>,
    att_context_size: Option<serde_yaml::Value>,
    conv_norm_type: Option<String>,
    conv_context_size: Option<serde_yaml::Value>,
    untie_biases: Option<bool>,
    use_bias: Option<bool>,
    global_tokens: Option<usize>,
    reduction: Option<serde_yaml::Value>,
    qkv_bias: Option<bool>,
    qk_norm: Option<bool>,
    pre_block_norm: Option<bool>,
    attn_mode: Option<String>,
    rope_base: Option<f64>,
    rotary_fraction: Option<f64>,
    causal_tail_len: Option<usize>,
}

/// v2.1's `transformer_encoder` section (NeMo
/// `transformer_encoders.TransformerEncoder`).
#[derive(Debug, Clone, Default, serde::Deserialize)]
struct SortformerTransformerConfig {
    #[serde(rename = "_target_")]
    target: Option<String>,
    num_attention_heads: Option<usize>,
    hidden_act: Option<String>,
    pre_ln: Option<bool>,
    mask_future: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SortformerEncoderKind {
    Conformer,
    FeatureStackingRope,
}

/// The graph a checkpoint yaml defines, checked against the one izwi runs
/// (SRT-8, NDZ-4): a field outside the served set fails the load instead
/// of loading same-shaped weights into a different graph.
#[derive(Debug, Clone, Copy, PartialEq)]
enum SortformerGraphSpec {
    /// v2.1: `dw_striding` subsampling, a relative-position Conformer and a
    /// post-LN ReLU transformer at the encoded rate.
    Conformer {
        heads: usize,
        xscaling: bool,
        transformer_heads: usize,
    },
    /// Nemotron-3: feature stacking, a pre-norm RoPE transformer with full
    /// attention, and the subpixel upsampler to 10 ms rows.
    FeatureStackingRope {
        heads: usize,
        stacking_factor: usize,
        rope_base: f32,
    },
}

impl SortformerGraphSpec {
    fn encoder_kind(self) -> SortformerEncoderKind {
        match self {
            Self::Conformer { .. } => SortformerEncoderKind::Conformer,
            Self::FeatureStackingRope { .. } => SortformerEncoderKind::FeatureStackingRope,
        }
    }
}

/// Discriminates the acoustic graph from the encoder's `_target_`; anything
/// that is not one of the two served encoders fails closed.
fn resolve_encoder_kind(cfg: &SortformerEncoderConfig) -> Result<SortformerEncoderKind> {
    let target = cfg
        .target
        .as_deref()
        .and_then(|value| value.rsplit('.').next())
        .unwrap_or("");
    match target {
        "ConformerEncoder" => Ok(SortformerEncoderKind::Conformer),
        "TransformerEncoder" => Ok(SortformerEncoderKind::FeatureStackingRope),
        _ => Err(Error::ModelLoadError(format!(
            "unsupported Sortformer encoder target {:?}",
            cfg.target
        ))),
    }
}

/// Fails the load unless the yaml's `field` holds the `served` value.
fn require_config<T: PartialEq + std::fmt::Debug>(field: &str, value: T, served: T) -> Result<()> {
    if value == served {
        return Ok(());
    }
    Err(Error::ModelLoadError(format!(
        "unsupported Sortformer config {field} = {value:?}; izwi serves only {served:?}"
    )))
}

/// The Conformer conv module folds an inference BatchNorm into its depthwise
/// kernel at load; any other normalization fails closed.
fn validate_conv_norm_type(cfg: Option<&SortformerEncoderConfig>) -> Result<()> {
    match cfg.and_then(|cfg| cfg.conv_norm_type.as_deref()) {
        None | Some("batch_norm") => Ok(()),
        Some(other) => Err(Error::ModelLoadError(format!(
            "unsupported Sortformer conv_norm_type {other:?}; only batch_norm is served"
        ))),
    }
}

/// Full-context attention: NeMo's default `att_context_size` (absent or
/// null) or `[-1, -1]`. A limited or multi-lookahead context fails closed.
fn validate_full_attention_context(value: Option<&serde_yaml::Value>) -> Result<()> {
    let full = match value {
        None | Some(serde_yaml::Value::Null) => true,
        Some(serde_yaml::Value::Sequence(items)) => {
            items.len() == 2 && items.iter().all(|item| item.as_i64() == Some(-1))
        }
        Some(_) => false,
    };
    if full {
        return Ok(());
    }
    Err(Error::ModelLoadError(format!(
        "unsupported Sortformer config encoder.att_context_size = {value:?}; izwi serves only [-1, -1]"
    )))
}

/// Reads and validates every graph-defining field of a Sortformer checkpoint
/// yaml. Shape-defining values (layers, widths) come from the tensors and
/// the production topology pins; these are the flags the tensors cannot
/// show. Head counts and the RoPE base are taken from the yaml.
fn resolve_graph_spec(config: &SortformerModelConfig) -> Result<SortformerGraphSpec> {
    let encoder = config.encoder.as_ref().ok_or_else(|| {
        Error::ModelLoadError("Sortformer config has no encoder section".to_string())
    })?;
    let modules = config.sortformer_modules.clone().unwrap_or_default();
    require_config(
        "async_streaming",
        config.async_streaming.unwrap_or(false),
        false,
    )?;
    let high_resolution = config.high_resolution.unwrap_or(false);
    let learnable_silence = modules.use_learnable_sil_emb.unwrap_or(false);
    match resolve_encoder_kind(encoder)? {
        SortformerEncoderKind::Conformer => {
            // NeMo `ConformerEncoder` defaults: striding x4, rel_pos, 4 heads.
            require_config(
                "encoder.subsampling",
                encoder.subsampling.as_deref().unwrap_or("striding"),
                "dw_striding",
            )?;
            let factor = encoder.subsampling_factor.unwrap_or(4);
            require_config(
                "encoder.subsampling_factor",
                factor,
                TS_VAD_UNIT_FRAME_COUNT,
            )?;
            require_config(
                "encoder.causal_downsampling",
                encoder.causal_downsampling.unwrap_or(false),
                false,
            )?;
            require_config(
                "encoder.self_attention_model",
                encoder.self_attention_model.as_deref().unwrap_or("rel_pos"),
                "rel_pos",
            )?;
            validate_full_attention_context(encoder.att_context_size.as_ref())?;
            validate_conv_norm_type(Some(encoder))?;
            require_config(
                "encoder.conv_context_size",
                encoder.conv_context_size.as_ref(),
                None,
            )?;
            require_config(
                "encoder.untie_biases",
                encoder.untie_biases.unwrap_or(true),
                true,
            )?;
            require_config("encoder.use_bias", encoder.use_bias.unwrap_or(true), true)?;
            require_config(
                "encoder.global_tokens",
                encoder.global_tokens.unwrap_or(0),
                0,
            )?;
            require_config("encoder.reduction", encoder.reduction.as_ref(), None)?;
            require_config("high_resolution", high_resolution, false)?;
            require_config(
                "output_subsampling_factor",
                config.output_subsampling_factor.unwrap_or(factor),
                factor,
            )?;
            require_config(
                "sortformer_modules.use_learnable_sil_emb",
                learnable_silence,
                false,
            )?;
            let transformer = config.transformer_encoder.as_ref().ok_or_else(|| {
                Error::ModelLoadError(
                    "Sortformer Conformer config has no transformer_encoder section".to_string(),
                )
            })?;
            require_config(
                "transformer_encoder._target_",
                transformer
                    .target
                    .as_deref()
                    .and_then(|value| value.rsplit('.').next()),
                Some("TransformerEncoder"),
            )?;
            // NeMo `TransformerEncoder` defaults: ReLU, post-LN, one head.
            require_config(
                "transformer_encoder.hidden_act",
                transformer.hidden_act.as_deref().unwrap_or("relu"),
                "relu",
            )?;
            require_config(
                "transformer_encoder.pre_ln",
                transformer.pre_ln.unwrap_or(false),
                false,
            )?;
            require_config(
                "transformer_encoder.mask_future",
                transformer.mask_future.unwrap_or(false),
                false,
            )?;
            Ok(SortformerGraphSpec::Conformer {
                heads: encoder.n_heads.unwrap_or(4),
                xscaling: encoder.xscaling.unwrap_or(true),
                transformer_heads: transformer.num_attention_heads.unwrap_or(1),
            })
        }
        SortformerEncoderKind::FeatureStackingRope => {
            // NeMo `TransformerEncoder` defaults: feature stacking x4,
            // rel_pos, no xscaling, pre-block norm, full attention.
            require_config(
                "encoder.subsampling",
                encoder.subsampling.as_deref().unwrap_or("feature_stacking"),
                "feature_stacking",
            )?;
            let factor = encoder.subsampling_factor.unwrap_or(4);
            require_config(
                "encoder.self_attention_model",
                encoder.self_attention_model.as_deref().unwrap_or("rel_pos"),
                "rope",
            )?;
            require_config("encoder.xscaling", encoder.xscaling.unwrap_or(false), false)?;
            require_config("encoder.qkv_bias", encoder.qkv_bias.unwrap_or(false), false)?;
            require_config("encoder.qk_norm", encoder.qk_norm.unwrap_or(false), false)?;
            require_config(
                "encoder.pre_block_norm",
                encoder.pre_block_norm.unwrap_or(true),
                true,
            )?;
            require_config(
                "encoder.attn_mode",
                encoder.attn_mode.as_deref().unwrap_or("full"),
                "full",
            )?;
            require_config(
                "encoder.rotary_fraction",
                encoder.rotary_fraction.unwrap_or(1.0),
                1.0,
            )?;
            require_config(
                "encoder.causal_tail_len",
                encoder.causal_tail_len.unwrap_or(0),
                0,
            )?;
            let rope_base = encoder.rope_base.unwrap_or(10_000.0);
            if !(rope_base.is_finite() && rope_base > 1.0) {
                return Err(Error::ModelLoadError(format!(
                    "unsupported Sortformer config encoder.rope_base = {rope_base}"
                )));
            }
            require_config("high_resolution", high_resolution, true)?;
            require_config(
                "output_subsampling_factor",
                config.output_subsampling_factor.unwrap_or(factor),
                1,
            )?;
            require_config(
                "sortformer_modules.use_learnable_sil_emb",
                learnable_silence,
                true,
            )?;
            if config.transformer_encoder.is_some() {
                return Err(Error::ModelLoadError(
                    "unsupported Sortformer config: a transformer_encoder beside the RoPE encoder"
                        .to_string(),
                ));
            }
            Ok(SortformerGraphSpec::FeatureStackingRope {
                heads: encoder.n_heads.unwrap_or(8),
                stacking_factor: factor,
                rope_base: rope_base as f32,
            })
        }
    }
}

#[derive(Debug, Clone, Default, serde::Deserialize)]
struct SortformerModulesConfig {
    num_spks: Option<usize>,
    use_learnable_sil_emb: Option<bool>,
    fc_d_model: Option<usize>,
    subsampling_factor: Option<usize>,
    spkcache_len: Option<usize>,
    fifo_len: Option<usize>,
    chunk_len: Option<usize>,
    spkcache_update_period: Option<usize>,
    chunk_left_context: Option<usize>,
    chunk_right_context: Option<usize>,
    spkcache_sil_frames_per_spk: Option<usize>,
    pred_score_threshold: Option<f32>,
    scores_boost_latest: Option<f32>,
    sil_threshold: Option<f32>,
    strong_boost_rate: Option<f32>,
    weak_boost_rate: Option<f32>,
    min_pos_scores_rate: Option<f32>,
}

/// Streaming geometry a loaded checkpoint runs, chosen once per process
/// with `IZWI_SORTFORMER_STREAMING_PROFILE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SortformerStreamingProfile {
    /// The checkpoint yaml's `sortformer_modules` values (NeMo's
    /// `from_pretrained` default; Nemotron-3's training geometry).
    Model,
    /// The model card's low-latency profile.
    LowLatency,
    /// The model card's very-high-latency (offline) profile: the
    /// configuration NVIDIA benchmarks, and the default for file
    /// diarization, which never streams output to the client.
    VeryHighLatency,
}

impl SortformerStreamingProfile {
    const DEFAULT: Self = Self::VeryHighLatency;

    fn name(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::LowLatency => "low_latency",
            Self::VeryHighLatency => "very_high_latency",
        }
    }

    /// The model card geometry this profile sets for `encoder_kind`, in
    /// 80 ms frames; `None` keeps the checkpoint yaml's values.
    fn card_geometry(self, encoder_kind: SortformerEncoderKind) -> Option<SortformerCardGeometry> {
        let geometry = |chunk_len, chunk_right_context, fifo_len, update_period, spkcache_len| {
            Some(SortformerCardGeometry {
                chunk_len,
                chunk_right_context,
                fifo_len,
                spkcache_update_period: update_period,
                spkcache_len,
            })
        };
        match (encoder_kind, self) {
            (_, Self::Model) => None,
            // diar_streaming_sortformer_4spk-v2.1 card: 6/7/188/144/188 and
            // 340/40/40/300/188 (chunk / right context / FIFO / update /
            // cache); the left context stays the yaml's 1.
            (SortformerEncoderKind::Conformer, Self::LowLatency) => geometry(6, 7, 188, 144, 188),
            (SortformerEncoderKind::Conformer, Self::VeryHighLatency) => {
                geometry(340, 40, 40, 300, 188)
            }
            // Nemotron-3-Diarization card and HF config.json: low 9/4/264/
            // 222/264 and offline 340/40/40/300/264, left context 0.
            (SortformerEncoderKind::FeatureStackingRope, Self::LowLatency) => {
                geometry(9, 4, 264, 222, 264)
            }
            (SortformerEncoderKind::FeatureStackingRope, Self::VeryHighLatency) => {
                geometry(340, 40, 40, 300, 264)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SortformerCardGeometry {
    chunk_len: usize,
    chunk_right_context: usize,
    fifo_len: usize,
    spkcache_update_period: usize,
    spkcache_len: usize,
}

#[derive(Debug, Clone, Copy)]
struct SortformerStreamingConfig {
    fc_d_model: usize,
    num_speakers: usize,
    subsampling_factor: usize,
    /// Probability rows produced per encoded 80 ms frame: 1 for v2.1 (its
    /// head runs at the encoded rate and post-processing repeats), 8 for
    /// Nemotron-3 (its subpixel upsampler emits 10 ms rows directly).
    output_frames_per_encoded_frame: usize,
    spkcache_len: usize,
    fifo_len: usize,
    chunk_len: usize,
    spkcache_update_period: usize,
    chunk_left_context: usize,
    chunk_right_context: usize,
    spkcache_sil_frames_per_spk: usize,
    pred_score_threshold: f32,
    scores_boost_latest: f32,
    sil_threshold: f32,
    strong_boost_rate: f32,
    weak_boost_rate: f32,
    min_pos_scores_rate: f32,
}

impl SortformerStreamingConfig {
    fn validate(self) -> Result<Self> {
        let min_spkcache_len = (1 + self.spkcache_sil_frames_per_spk) * self.num_speakers;
        if self.num_speakers == 0
            || self.subsampling_factor == 0
            || self.fc_d_model == 0
            || self.chunk_len == 0
            || self.spkcache_update_period == 0
            || self.output_frames_per_encoded_frame == 0
            || !self.subsampling_factor.is_multiple_of(self.output_frames_per_encoded_frame)
        {
            return Err(Error::ModelLoadError(
                "Sortformer streaming config contains zero-valued required fields".to_string(),
            ));
        }
        if !self.spkcache_len.is_multiple_of(self.num_speakers) {
            return Err(Error::ModelLoadError(format!(
                "Sortformer spkcache_len {} is not divisible by {} speaker channels",
                self.spkcache_len, self.num_speakers
            )));
        }
        if self.spkcache_len < min_spkcache_len {
            return Err(Error::ModelLoadError(format!(
                "Sortformer spkcache_len {} is smaller than the required minimum {}",
                self.spkcache_len, min_spkcache_len
            )));
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SortformerEncoderTopology {
    Conformer {
        conv_channels: usize,
        layers: usize,
        d_model: usize,
        ff_dim: usize,
        heads: usize,
    },
    FeatureStackingRope {
        layers: usize,
        d_model: usize,
        ff_dim: usize,
        heads: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SortformerFrameExpanderTopology {
    /// v2.1: the sortformer transformer layers operating at the encoded rate.
    SortformerTransformer {
        layers: usize,
        d_model: usize,
        inner_dim: usize,
        heads: usize,
    },
    /// Nemotron-3: the learned subpixel upsampler expanding to 10 ms.
    SubpixelUpsampler {
        d_model: usize,
        upsample_factor: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SortformerWorkspaceTopology {
    feature_bins: usize,
    n_fft: usize,
    hop_length: usize,
    encoder: SortformerEncoderTopology,
    expander: SortformerFrameExpanderTopology,
}

impl SortformerWorkspaceTopology {
    const fn v21_production() -> Self {
        Self {
            feature_bins: PRODUCTION_FEATURE_BINS,
            n_fft: PRODUCTION_N_FFT,
            hop_length: PRODUCTION_HOP_LENGTH,
            encoder: SortformerEncoderTopology::Conformer {
                conv_channels: PRODUCTION_CONV_CHANNELS,
                layers: PRODUCTION_CONFORMER_LAYERS,
                d_model: PRODUCTION_CONFORMER_D_MODEL,
                ff_dim: PRODUCTION_CONFORMER_FF_DIM,
                heads: PRODUCTION_CONFORMER_HEADS,
            },
            expander: SortformerFrameExpanderTopology::SortformerTransformer {
                layers: PRODUCTION_TRANSFORMER_LAYERS,
                d_model: PRODUCTION_TRANSFORMER_D_MODEL,
                inner_dim: PRODUCTION_TRANSFORMER_INNER_DIM,
                heads: PRODUCTION_TRANSFORMER_HEADS,
            },
        }
    }

    const fn nemotron3_production() -> Self {
        Self {
            feature_bins: PRODUCTION_FEATURE_BINS,
            n_fft: PRODUCTION_N_FFT,
            hop_length: PRODUCTION_HOP_LENGTH,
            encoder: SortformerEncoderTopology::FeatureStackingRope {
                layers: NEMOTRON3_ROPE_LAYERS,
                d_model: NEMOTRON3_ROPE_D_MODEL,
                ff_dim: NEMOTRON3_ROPE_FF_DIM,
                heads: NEMOTRON3_ROPE_HEADS,
            },
            expander: SortformerFrameExpanderTopology::SubpixelUpsampler {
                d_model: NEMOTRON3_HEAD_D_MODEL,
                upsample_factor: TS_VAD_UNIT_FRAME_COUNT,
            },
        }
    }

    fn encoder_d_model(&self) -> usize {
        match self.encoder {
            SortformerEncoderTopology::Conformer { d_model, .. } => d_model,
            SortformerEncoderTopology::FeatureStackingRope { d_model, .. } => d_model,
        }
    }

    fn validate_production(self, cfg: SortformerStreamingConfig) -> Result<()> {
        let profile = PRODUCTION_PROFILES
            .iter()
            .find(|profile| profile.topology == self)
            .ok_or_else(|| {
                Error::ModelLoadError(format!(
                    "unsupported Sortformer workspace topology {self:?}"
                ))
            })?;
        if cfg.fc_d_model != profile.topology.encoder_d_model()
            || cfg.subsampling_factor != TS_VAD_UNIT_FRAME_COUNT
            || cfg.chunk_len > profile.max_chunk_len
            || cfg.chunk_left_context > profile.max_chunk_left_context
            || cfg.chunk_right_context > profile.max_chunk_right_context
            || cfg.spkcache_len > profile.max_spkcache_len
            || cfg.fifo_len > profile.max_fifo_len
        {
            return Err(Error::ModelLoadError(format!(
                "unsupported Sortformer streaming workspace configuration: {cfg:?}"
            )));
        }
        Ok(())
    }
}

/// Immutable pre-admission envelope for one served Sortformer checkpoint
/// class: the pinned tensor topology plus the largest streaming profile its
/// workspace estimate may use.
struct SortformerProductionProfile {
    topology: SortformerWorkspaceTopology,
    max_chunk_len: usize,
    max_chunk_left_context: usize,
    max_chunk_right_context: usize,
    max_spkcache_len: usize,
    max_fifo_len: usize,
}

const V21_PRODUCTION_PROFILE: SortformerProductionProfile = SortformerProductionProfile {
    topology: SortformerWorkspaceTopology::v21_production(),
    max_chunk_len: PRODUCTION_MAX_CHUNK_LEN,
    max_chunk_left_context: PRODUCTION_MAX_CHUNK_LEFT_CONTEXT,
    max_chunk_right_context: PRODUCTION_MAX_CHUNK_RIGHT_CONTEXT,
    max_spkcache_len: PRODUCTION_MAX_SPKCACHE_LEN,
    max_fifo_len: PRODUCTION_MAX_FIFO_LEN,
};

const NEMOTRON3_PRODUCTION_PROFILE: SortformerProductionProfile = SortformerProductionProfile {
    topology: SortformerWorkspaceTopology::nemotron3_production(),
    max_chunk_len: NEMOTRON3_MAX_CHUNK_LEN,
    max_chunk_left_context: 0,
    max_chunk_right_context: NEMOTRON3_MAX_CHUNK_RIGHT_CONTEXT,
    max_spkcache_len: NEMOTRON3_MAX_SPKCACHE_LEN,
    max_fifo_len: NEMOTRON3_MAX_FIFO_LEN,
};

const PRODUCTION_PROFILES: [SortformerProductionProfile; 2] =
    [V21_PRODUCTION_PROFILE, NEMOTRON3_PRODUCTION_PROFILE];

fn production_workspace_streaming_config() -> SortformerStreamingConfig {
    SortformerStreamingConfig {
        fc_d_model: PRODUCTION_CONFORMER_D_MODEL,
        num_speakers: 4,
        subsampling_factor: TS_VAD_UNIT_FRAME_COUNT,
        output_frames_per_encoded_frame: 1,
        spkcache_len: PRODUCTION_MAX_SPKCACHE_LEN,
        fifo_len: PRODUCTION_MAX_FIFO_LEN,
        chunk_len: PRODUCTION_MAX_CHUNK_LEN,
        spkcache_update_period: 300,
        chunk_left_context: PRODUCTION_MAX_CHUNK_LEFT_CONTEXT,
        chunk_right_context: PRODUCTION_MAX_CHUNK_RIGHT_CONTEXT,
        spkcache_sil_frames_per_spk: 3,
        pred_score_threshold: 0.25,
        scores_boost_latest: 0.05,
        sil_threshold: 0.2,
        strong_boost_rate: 0.75,
        weak_boost_rate: 1.5,
        min_pos_scores_rate: 0.5,
    }
}

/// Largest streaming envelope the Nemotron-3 checkpoint may claim: the card
/// offline chunk and right context with the card low-latency FIFO.
fn nemotron3_workspace_streaming_config() -> SortformerStreamingConfig {
    SortformerStreamingConfig {
        chunk_len: NEMOTRON3_MAX_CHUNK_LEN,
        chunk_right_context: NEMOTRON3_MAX_CHUNK_RIGHT_CONTEXT,
        fifo_len: NEMOTRON3_MAX_FIFO_LEN,
        spkcache_update_period: 300,
        ..nemotron3_checkpoint_streaming_config()
    }
}

/// The Nemotron-3 checkpoint yaml's trained sortformer_modules values.
fn nemotron3_checkpoint_streaming_config() -> SortformerStreamingConfig {
    SortformerStreamingConfig {
        fc_d_model: NEMOTRON3_ROPE_D_MODEL,
        num_speakers: 8,
        subsampling_factor: TS_VAD_UNIT_FRAME_COUNT,
        output_frames_per_encoded_frame: TS_VAD_UNIT_FRAME_COUNT,
        spkcache_len: NEMOTRON3_MAX_SPKCACHE_LEN,
        fifo_len: 0,
        chunk_len: 264,
        spkcache_update_period: 264,
        chunk_left_context: 0,
        chunk_right_context: 0,
        spkcache_sil_frames_per_spk: 1,
        pred_score_threshold: 0.25,
        scores_boost_latest: 0.05,
        sil_threshold: 0.2,
        strong_boost_rate: 0.75,
        weak_boost_rate: 1.5,
        min_pos_scores_rate: 0.5,
    }
}

/// Immutable refusal ceiling for the supported production checkpoints. The
/// estimate uses each checkpoint's largest supported streaming profile and
/// takes the larger of the two. It is never reserved: a diarize stage
/// reserves the loaded model's own profile-shaped estimate and refuses one
/// above this ceiling.
pub fn production_workspace_authorization(
    target_sample_count: usize,
    separate_device_memory: bool,
) -> Result<SortformerWorkspaceEstimate> {
    let v21 = workspace_estimate_for(
        SortformerWorkspaceTopology::v21_production(),
        production_workspace_streaming_config(),
        target_sample_count,
        separate_device_memory,
    )?;
    let nemotron3 = workspace_estimate_for(
        SortformerWorkspaceTopology::nemotron3_production(),
        nemotron3_workspace_streaming_config(),
        target_sample_count,
        separate_device_memory,
    )?;
    Ok(SortformerWorkspaceEstimate {
        host_bytes: v21.host_bytes.max(nemotron3.host_bytes),
        accelerator_bytes: v21.accelerator_bytes.max(nemotron3.accelerator_bytes),
    })
}

/// The workspace `variant` estimates once loaded under the default
/// streaming profile, so tests can price a request without the checkpoint.
/// v2.1's `model_config.yaml` pins chunk 188, left and right context 1,
/// FIFO 0 and update period 188 before the profile applies.
#[cfg(test)]
pub(crate) fn default_profile_workspace_estimate(
    variant: ModelVariant,
    target_sample_count: usize,
    separate_device_memory: bool,
) -> Result<SortformerWorkspaceEstimate> {
    let (topology, checkpoint, encoder_kind) = match variant {
        ModelVariant::DiarStreamingSortformer4SpkV21 => (
            SortformerWorkspaceTopology::v21_production(),
            SortformerStreamingConfig {
                chunk_len: 188,
                chunk_left_context: 1,
                chunk_right_context: 1,
                fifo_len: 0,
                spkcache_update_period: 188,
                ..production_workspace_streaming_config()
            },
            SortformerEncoderKind::Conformer,
        ),
        ModelVariant::Nemotron3Diarization => (
            SortformerWorkspaceTopology::nemotron3_production(),
            nemotron3_checkpoint_streaming_config(),
            SortformerEncoderKind::FeatureStackingRope,
        ),
        other => {
            return Err(Error::InvalidInput(format!(
                "{other} is not a Sortformer diarizer"
            )))
        }
    };
    let cfg = apply_streaming_profile(
        checkpoint,
        encoder_kind,
        SortformerStreamingProfile::DEFAULT,
    );
    workspace_estimate_for(topology, cfg, target_sample_count, separate_device_memory)
}

fn workspace_estimate_for(
    topology: SortformerWorkspaceTopology,
    cfg: SortformerStreamingConfig,
    target_sample_count: usize,
    separate_device_memory: bool,
) -> Result<SortformerWorkspaceEstimate> {
    topology.validate_production(cfg)?;
    let feature_frames = target_sample_count / topology.hop_length;
    let chunk_feature_cap = cfg
        .chunk_len
        .checked_add(cfg.chunk_left_context)
        .and_then(|frames| frames.checked_add(cfg.chunk_right_context))
        .and_then(|frames| frames.checked_mul(cfg.subsampling_factor))
        .ok_or_else(|| Error::Overloaded("Sortformer chunk shape overflowed".to_string()))?;
    let chunk_feature_frames = feature_frames.min(chunk_feature_cap);
    let chunk_encoded_frames = match topology.encoder {
        SortformerEncoderTopology::Conformer { .. } => subsampled_len_3x(chunk_feature_frames),
        SortformerEncoderTopology::FeatureStackingRope { .. } => {
            chunk_feature_frames.div_ceil(cfg.subsampling_factor)
        }
    };
    let composite_frames = chunk_encoded_frames
        .checked_add(cfg.spkcache_len)
        .and_then(|frames| frames.checked_add(cfg.fifo_len))
        .ok_or_else(|| Error::Overloaded("Sortformer composite shape overflowed".to_string()))?;

    let u = |value: usize| value as u128;
    let f32_bytes = u(DType::F32.size_in_bytes());
    let feature_elements = u(chunk_feature_frames) * u(topology.feature_bins);
    let feature_bytes = feature_elements * f32_bytes;

    // Direct chunk preprocessing retains one feature matrix plus a single FFT
    // and spectrum row. Duration-shaped probability rows and the cache/row
    // copies used by the streaming state are host allocations.
    let fft_scratch_bytes = (u(topology.n_fft) * 2 + u(topology.n_fft / 2 + 1)) * f32_bytes;
    let streaming_row_bytes =
        u(composite_frames) * u(topology.encoder_d_model()) * f32_bytes * 32;
    let output_bytes = u(feature_frames) * u(cfg.num_speakers) * f32_bytes;
    let staging_bytes = if separate_device_memory {
        feature_bytes
    } else {
        0
    };
    let host_bytes = (fft_scratch_bytes + streaming_row_bytes + output_bytes + staging_bytes) * 2;

    // The count below covers the largest live activation set of one encoder
    // layer plus the frame expander and classifier head. Layers execute
    // sequentially; checkpoint weights are covered by the model residency
    // lease rather than this job workspace.
    let graph_elements = match (topology.encoder, topology.expander) {
        (
            SortformerEncoderTopology::Conformer {
                conv_channels,
                d_model: conformer_d_model,
                ff_dim: conformer_ff_dim,
                heads: conformer_heads,
                ..
            },
            SortformerFrameExpanderTopology::SortformerTransformer {
                d_model: transformer_d_model,
                inner_dim: transformer_inner_dim,
                heads: transformer_heads,
                ..
            },
        ) => {
            // Conv subsampling's first output is the largest
            // duration/frequency activation of the encoder stack. The
            // dw_striding chain holds at most three of it at once: the conv0
            // matmul result and its NCHW copy, then that copy and its ReLU
            // output; the depthwise stage after it holds the ReLU output plus
            // two quarter-size tensors (per-group outputs and their `cat`).
            // Live malloc bytes peaked at 2.02x conv0's output (+96.2 MiB at
            // 1,520 feature frames), against the 8x counted before.
            let conv_t = u(chunk_feature_frames.div_ceil(2));
            let conv_f = u(topology.feature_bins.div_ceil(2));
            let conv_elements = conv_t * conv_f * u(conv_channels);
            let conformer_linear = u(composite_frames) * u(conformer_d_model);
            let conformer_ff = u(composite_frames) * u(conformer_ff_dim);
            let conformer_scores = u(conformer_heads)
                * u(composite_frames)
                * u(composite_frames.saturating_mul(2).saturating_sub(1));
            let conformer_elements =
                conformer_linear * 24 + conformer_ff * 4 + conformer_scores * 6;
            let transformer_linear = u(composite_frames) * u(transformer_d_model);
            let transformer_ff = u(composite_frames) * u(transformer_inner_dim);
            let transformer_scores =
                u(transformer_heads) * u(composite_frames) * u(composite_frames);
            let transformer_elements =
                transformer_linear * 20 + transformer_ff * 4 + transformer_scores * 4;
            conv_elements * 3 + conformer_elements + transformer_elements + conformer_linear * 8
        }
        (
            SortformerEncoderTopology::FeatureStackingRope {
                d_model: rope_d_model,
                ff_dim: rope_ff_dim,
                heads: rope_heads,
                ..
            },
            SortformerFrameExpanderTopology::SubpixelUpsampler {
                d_model: head_d_model,
                upsample_factor,
            },
        ) => {
            let rope_linear = u(composite_frames) * u(rope_d_model);
            let rope_scores = u(rope_heads) * u(composite_frames) * u(composite_frames);
            let rope_ff = u(composite_frames) * u(rope_ff_dim);
            // Live set per layer: norm1, fused QKV, rope'd q/k, scores,
            // softmax, context, out proj, residual, norm2, FFN in/out.
            let rope_elements = rope_linear * 13 + rope_scores * 2 + rope_ff * 3;
            let expanded_frames = u(composite_frames) * u(upsample_factor);
            let expanded_linear = expanded_frames * u(head_d_model);
            let head_elements = u(composite_frames) * u(head_d_model) * 2
                + u(composite_frames) * u(head_d_model) * u(upsample_factor) * 4
                + expanded_linear * 5
                + expanded_frames * u(cfg.num_speakers) * 2;
            rope_elements + head_elements
        }
        _ => {
            return Err(Error::ModelLoadError(format!(
                "mismatched Sortformer encoder/expander topology {topology:?}"
            )))
        }
    };
    let accelerator_bytes =
        ((feature_elements + graph_elements) * f32_bytes * SORTFORMER_WORKSPACE_SAFETY_PERCENT)
            .div_ceil(100);

    let to_u64 = |bytes: u128, domain: &str| {
        u64::try_from(bytes).map_err(|_| {
            Error::Overloaded(format!("Sortformer {domain} workspace estimate overflowed"))
        })
    };
    Ok(SortformerWorkspaceEstimate {
        host_bytes: to_u64(host_bytes, "host")?,
        accelerator_bytes: to_u64(accelerator_bytes, "accelerator")?,
    })
}

#[derive(Debug, Clone)]
struct SortformerStreamingChunkPlan {
    feature_start: usize,
    feature_end: usize,
    left_offset: usize,
    right_offset: usize,
}

/// `(rows, D)` embeddings kept on the model device. Empty is `None`, so no
/// zero-sized tensor reaches a device kernel. Every operation is an exact
/// copy (`narrow`, `cat`, `index_select`), so the values match the host rows
/// they replace bit for bit.
#[derive(Debug, Clone, Default)]
struct EmbeddingRows(Option<Tensor>);

impl EmbeddingRows {
    fn new(tensor: Tensor) -> Result<Self> {
        let (rows, _) = tensor.dims2()?;
        Ok(Self((rows > 0).then_some(tensor)))
    }

    fn len(&self) -> usize {
        self.0.as_ref().map_or(0, |tensor| tensor.dims()[0])
    }

    fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    fn tensor(&self) -> Option<&Tensor> {
        self.0.as_ref()
    }

    fn narrow(&self, start: usize, len: usize) -> Result<Self> {
        if len == 0 {
            return Ok(Self(None));
        }
        let tensor = self.0.as_ref().ok_or_else(|| {
            Error::InferenceError("Sortformer embedding rows out of range".into())
        })?;
        Ok(Self(Some(tensor.narrow(0, start, len)?)))
    }

    fn cat(parts: &[&Self]) -> Result<Self> {
        let tensors = parts
            .iter()
            .filter_map(|part| part.0.as_ref())
            .collect::<Vec<_>>();
        Ok(Self(match tensors.as_slice() {
            [] => None,
            [only] => Some((*only).clone()),
            many => Some(Tensor::cat(many, 0)?),
        }))
    }

    fn index_select(&self, indices: &[u32]) -> Result<Self> {
        if indices.is_empty() {
            return Ok(Self(None));
        }
        let tensor = self.0.as_ref().ok_or_else(|| {
            Error::InferenceError("Sortformer embedding rows out of range".into())
        })?;
        let indices = Tensor::from_slice(indices, indices.len(), tensor.device())?;
        Ok(Self(Some(tensor.index_select(&indices, 0)?)))
    }
}

#[derive(Debug, Clone)]
struct SortformerStreamingState {
    spkcache: EmbeddingRows,
    spkcache_preds: Option<Vec<Vec<f32>>>,
    fifo: EmbeddingRows,
    fifo_preds: Vec<Vec<f32>>,
    mean_sil_emb: Vec<f32>,
    n_sil_frames: usize,
    /// Device-to-host reads the state update made (silent FIFO rows for the
    /// running silence mean).
    host_reads: u64,
}

impl SortformerStreamingState {
    fn new(emb_dim: usize) -> Self {
        Self {
            spkcache: EmbeddingRows::default(),
            spkcache_preds: None,
            fifo: EmbeddingRows::default(),
            fifo_preds: Vec::new(),
            mean_sil_emb: vec![0.0; emb_dim],
            n_sil_frames: 0,
            host_reads: 0,
        }
    }
}

/// Per-model counters for the streaming loop (SRT-7), reported under
/// `streaming_path` in `family_diagnostics`.
#[derive(Debug, Default)]
struct SortformerStreamingCounters {
    chunks: AtomicU64,
    /// Bounded steps the streaming passes ran in (PIPE-8): one per pass when
    /// it runs whole, one per runtime stage when it is staged.
    steps: AtomicU64,
    host_reads: AtomicU64,
    physical_state_commits: AtomicU64,
}

/// A streaming pass that runs in bounded steps (PIPE-8): its chunk plan, the
/// streaming state, and the probabilities so far. A step may run as its own
/// runtime stage, so a long file does not hold an execution permit for its
/// whole pass.
#[derive(Debug)]
pub(crate) struct SortformerStream {
    cfg: SortformerStreamingConfig,
    sample_count: usize,
    feature_frames: usize,
    plans: Vec<SortformerStreamingChunkPlan>,
    next_plan: usize,
    chunks_per_step: usize,
    /// `None` only while a step runs.
    state: Option<SortformerStreamingState>,
    track_silence_mean: bool,
    total_preds: Vec<Vec<f32>>,
    chunks: u64,
    host_reads: u64,
}

impl SortformerStream {
    /// Every chunk has run; the stream can finish.
    pub(crate) fn is_finished(&self) -> bool {
        self.next_plan >= self.plans.len()
    }

    /// Steps left before the stream can finish.
    pub(crate) fn steps_remaining(&self) -> usize {
        (self.plans.len() - self.next_plan).div_ceil(self.chunks_per_step)
    }
}

fn commit_sortformer_streaming_state(
    lease: &mut InvocationTensorLease,
    cfg: SortformerStreamingConfig,
    state: &SortformerStreamingState,
) -> Result<()> {
    if state.spkcache.len() > cfg.spkcache_len
        || state.fifo.len() > cfg.fifo_len
        || state
            .spkcache_preds
            .as_ref()
            .is_some_and(|preds| preds.len() != state.spkcache.len())
        || state.fifo_preds.len() != state.fifo.len()
        || state.mean_sil_emb.len() != cfg.fc_d_model
    {
        return Err(Error::InferenceError(
            "Sortformer streaming state exceeds its physical geometry".into(),
        ));
    }
    let expected_cursor = lease.arena()?.absolute_cursor();
    let target_cursor = expected_cursor
        .checked_add(1)
        .ok_or_else(|| Error::InferenceError("Sortformer state cursor overflow".into()))?;
    // Stage the state tensors on the arena's own device. The Sortformer model
    // runs on CPU for every non-CUDA backend, so the model device is NOT a
    // valid staging device on Metal (and may differ from the worker device on
    // multi-device CUDA): the arena backing is what validates the commit.
    let device = {
        let arena = lease.arena()?;
        arena.device().clone()
    };
    let speaker_embeddings =
        padded_embedding_tensor(&state.spkcache, cfg.spkcache_len, cfg.fc_d_model, &device)?;
    let speaker_predictions = padded_prediction_tensor(
        state.spkcache_preds.as_deref().unwrap_or_default(),
        cfg.spkcache_len,
        cfg.num_speakers,
        &device,
    )?;
    let silence_mean = Tensor::from_vec(state.mean_sil_emb.clone(), cfg.fc_d_model, &device)?;
    let control = Tensor::from_vec(
        vec![
            state.spkcache.len() as f32,
            state.fifo.len() as f32,
            f32::from(state.spkcache_preds.is_some()),
            state.n_sil_frames as f32,
        ],
        4,
        &device,
    )?;
    let mut components = vec![
        InvocationTensorComponentValue {
            component: StateComponentId::new(1),
            tensor: speaker_embeddings,
        },
        InvocationTensorComponentValue {
            component: StateComponentId::new(2),
            tensor: speaker_predictions,
        },
        InvocationTensorComponentValue {
            component: StateComponentId::new(5),
            tensor: silence_mean,
        },
        InvocationTensorComponentValue {
            component: StateComponentId::new(6),
            tensor: control,
        },
    ];
    let mut declared = vec![
        ComponentShapeInstantiation {
            component: StateComponentId::new(1),
            dimensions: vec![
                ShapeDimensionValue {
                    axis: ShapeAxis::Frames,
                    units: cfg.spkcache_len as u64,
                },
                ShapeDimensionValue {
                    axis: ShapeAxis::Hidden,
                    units: cfg.fc_d_model as u64,
                },
            ],
        },
        ComponentShapeInstantiation {
            component: StateComponentId::new(2),
            dimensions: vec![
                ShapeDimensionValue {
                    axis: ShapeAxis::Frames,
                    units: cfg.spkcache_len as u64,
                },
                ShapeDimensionValue {
                    axis: ShapeAxis::Custom("speakers".into()),
                    units: cfg.num_speakers as u64,
                },
            ],
        },
        ComponentShapeInstantiation {
            component: StateComponentId::new(5),
            dimensions: vec![ShapeDimensionValue {
                axis: ShapeAxis::Hidden,
                units: cfg.fc_d_model as u64,
            }],
        },
        ComponentShapeInstantiation {
            component: StateComponentId::new(6),
            dimensions: vec![ShapeDimensionValue {
                axis: ShapeAxis::Custom("control".into()),
                units: 4,
            }],
        },
    ];
    if cfg.fifo_len > 0 {
        let fifo_embeddings =
            padded_embedding_tensor(&state.fifo, cfg.fifo_len, cfg.fc_d_model, &device)?;
        let fifo_predictions =
            padded_prediction_tensor(&state.fifo_preds, cfg.fifo_len, cfg.num_speakers, &device)?;
        components.insert(
            2,
            InvocationTensorComponentValue {
                component: StateComponentId::new(3),
                tensor: fifo_embeddings,
            },
        );
        components.insert(
            3,
            InvocationTensorComponentValue {
                component: StateComponentId::new(4),
                tensor: fifo_predictions,
            },
        );
        declared.insert(
            2,
            ComponentShapeInstantiation {
                component: StateComponentId::new(3),
                dimensions: vec![
                    ShapeDimensionValue {
                        axis: ShapeAxis::Frames,
                        units: cfg.fifo_len as u64,
                    },
                    ShapeDimensionValue {
                        axis: ShapeAxis::Hidden,
                        units: cfg.fc_d_model as u64,
                    },
                ],
            },
        );
        declared.insert(
            3,
            ComponentShapeInstantiation {
                component: StateComponentId::new(4),
                dimensions: vec![
                    ShapeDimensionValue {
                        axis: ShapeAxis::Frames,
                        units: cfg.fifo_len as u64,
                    },
                    ShapeDimensionValue {
                        axis: ShapeAxis::Custom("speakers".into()),
                        units: cfg.num_speakers as u64,
                    },
                ],
            },
        );
    }
    lease.apply_intent(
        &DomainStepIntent {
            domain: physical::SORTFORMER_STREAMING_STATE_DOMAIN,
            expected_cursor,
            target_cursor,
            update: StateUpdateKind::TensorReplace {
                components: declared,
            },
        },
        InvocationTensorUpdateV2::TensorReplace { components },
    )
}

fn padded_embedding_tensor(
    rows: &EmbeddingRows,
    capacity: usize,
    hidden: usize,
    device: &Device,
) -> Result<Tensor> {
    let Some(tensor) = rows.tensor() else {
        return Tensor::zeros((capacity, hidden), DType::F32, device).map_err(Error::from);
    };
    let (len, width) = tensor.dims2()?;
    if width != hidden || len > capacity {
        return Err(Error::InferenceError(format!(
            "Sortformer embedding rows [{len}, {width}] do not fit [{capacity}, {hidden}]"
        )));
    }
    tensor
        .to_device(device)?
        .pad_with_zeros(0, 0, capacity - len)?
        .contiguous()
        .map_err(Error::from)
}

fn padded_prediction_tensor(
    rows: &[Vec<f32>],
    capacity: usize,
    num_speakers: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut flat = vec![0.0_f32; capacity.saturating_mul(num_speakers)];
    for (index, row) in rows.iter().enumerate() {
        if row.len() != num_speakers {
            return Err(Error::InferenceError(format!(
                "Sortformer prediction row has width {}; expected {num_speakers}",
                row.len()
            )));
        }
        flat[index * num_speakers..(index + 1) * num_speakers].copy_from_slice(row);
    }
    Tensor::from_vec(flat, (capacity, num_speakers), device).map_err(Error::from)
}

#[derive(Debug, Clone)]
struct SortformerCacheCandidate {
    flat_index: usize,
    frame_index: Option<usize>,
    score: f32,
}

const SORTFORMER_SCORE_BOOST_DELTA: f32 = std::f32::consts::LN_2;

/// Speaker-channel count pinned by each served Sortformer checkpoint.
/// The checkpoint YAML must agree or the load fails closed.
fn expected_speaker_count(variant: ModelVariant) -> Result<usize> {
    variant.diarization_speaker_channels().ok_or_else(|| {
        Error::ModelLoadError(format!(
            "Unsupported Sortformer diarization variant: {}",
            variant.dir_name()
        ))
    })
}

pub struct SortformerDiarizerModel {
    variant: ModelVariant,
    _artifacts: SortformerArtifacts,
    _checkpoint_tensor_count: usize,
    model: SortformerInferenceModel,
}

impl SortformerDiarizerModel {
    pub(crate) fn physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<SortformerPhysicalStateSpec> {
        let cfg = self.model.streaming.ok_or_else(|| {
            Error::ModelLoadError(
                "Sortformer physical state requires the bounded streaming profile".into(),
            )
        })?;
        physical::sortformer_physical_state_spec(cfg, stage_graphs)
    }

    pub fn load(
        model_dir: &Path,
        variant: ModelVariant,
        device_profile: DeviceProfile,
    ) -> Result<Self> {
        if !variant.is_diarization() {
            return Err(Error::InvalidInput(format!(
                "Variant {} is not a Sortformer diarization model",
                variant.dir_name()
            )));
        }

        let artifacts = ensure_sortformer_artifacts(model_dir, variant)?;
        let tensor_info =
            candle_core::pickle::read_pth_tensor_info(&artifacts.checkpoint_path, false, None)
                .map_err(|e| {
                    Error::ModelLoadError(format!(
                        "Failed to inspect Sortformer checkpoint {}: {}",
                        artifacts.checkpoint_path.display(),
                        e
                    ))
                })?;

        let config: SortformerModelConfig = serde_yaml::from_str(
            &std::fs::read_to_string(&artifacts.model_config_path).map_err(|e| {
                Error::ModelLoadError(format!(
                    "Failed reading Sortformer config {}: {}",
                    artifacts.model_config_path.display(),
                    e
                ))
            })?,
        )
        .map_err(|e| {
            Error::ModelLoadError(format!(
                "Failed parsing Sortformer config {}: {}",
                artifacts.model_config_path.display(),
                e
            ))
        })?;

        let sample_rate = config.sample_rate.unwrap_or(TARGET_SAMPLE_RATE);
        if sample_rate != TARGET_SAMPLE_RATE {
            return Err(Error::ModelLoadError(format!(
                "Unsupported Sortformer sample rate {sample_rate}; expected {TARGET_SAMPLE_RATE}"
            )));
        }

        let expected_spks = expected_speaker_count(variant)?;
        let num_spks = config.max_num_of_spks.unwrap_or(expected_spks);
        let module_spks = config
            .sortformer_modules
            .as_ref()
            .and_then(|modules| modules.num_spks)
            .unwrap_or(num_spks);
        if num_spks != expected_spks || module_spks != expected_spks {
            return Err(Error::ModelLoadError(format!(
                "Unsupported Sortformer speaker count {num_spks} (sortformer_modules.num_spks {module_spks}); {} expects {expected_spks}",
                variant.dir_name()
            )));
        }
        let graph = resolve_graph_spec(&config)?;

        let device = sortformer_model_device(&device_profile);
        let vb =
            VarBuilder::from_pth(&artifacts.checkpoint_path, DType::F32, &device).map_err(|e| {
                Error::ModelLoadError(format!(
                    "Failed to load Sortformer checkpoint {}: {}",
                    artifacts.checkpoint_path.display(),
                    e
                ))
            })?;

        let preprocessor_cfg =
            config
                .preprocessor
                .clone()
                .unwrap_or(SortformerPreprocessorConfig {
                    sample_rate: Some(TARGET_SAMPLE_RATE),
                    window_size: Some(0.025),
                    window_stride: Some(0.01),
                    features: Some(128),
                    n_fft: Some(512),
                    normalize: Some("NA".to_string()),
                });

        let modules_cfg = config.sortformer_modules.clone().unwrap_or_default();
        let streaming_mode = config.streaming_mode.unwrap_or(false);
        let model = SortformerInferenceModel::load(
            &vb,
            preprocessor_cfg,
            resolve_streaming_profile(),
            num_spks,
            streaming_mode,
            graph,
            modules_cfg.clone(),
            device.clone(),
            device_profile.kind.is_cuda(),
        )?;

        let model = Self {
            variant,
            _artifacts: artifacts,
            _checkpoint_tensor_count: tensor_info.len(),
            model,
        };
        tracing::info!(
            "{} streams with the {} profile: {}",
            variant.dir_name(),
            model.model.streaming_profile.name(),
            model.family_diagnostics()
        );
        Ok(model)
    }

    /// The active streaming profile and geometry, for loaded-model
    /// diagnostics.
    pub fn family_diagnostics(&self) -> serde_json::Value {
        let streaming = self.model.streaming.map(|cfg| {
            serde_json::json!({
                "chunk_len": cfg.chunk_len,
                "chunk_left_context": cfg.chunk_left_context,
                "chunk_right_context": cfg.chunk_right_context,
                "fifo_len": cfg.fifo_len,
                "spkcache_update_period": cfg.spkcache_update_period,
                "spkcache_len": cfg.spkcache_len,
            })
        });
        serde_json::json!({
            "streaming_profile": self.model.streaming_profile.name(),
            "streaming": streaming,
            "kernels": self.model.encoder.kernel_diagnostics(),
            "streaming_path": self.model.streaming_path_diagnostics(),
        })
    }

    pub fn diarize(
        &self,
        audio: &[f32],
        sample_rate: u32,
        config: &DiarizationConfig,
    ) -> Result<DiarizationResult> {
        self.diarize_with_workspace_observer(audio, sample_rate, config, |_| Ok(()))
    }

    /// Complete peak job workspace for the loaded streaming topology.
    /// `target_sample_count` describes 16 kHz audio after runtime decoding.
    pub fn workspace_estimate(
        &self,
        target_sample_count: usize,
    ) -> Result<SortformerWorkspaceEstimate> {
        self.model.workspace_estimate(target_sample_count)
    }

    pub fn diarize_with_workspace_observer<F>(
        &self,
        audio: &[f32],
        sample_rate: u32,
        config: &DiarizationConfig,
        observer: F,
    ) -> Result<DiarizationResult>
    where
        F: FnMut(SortformerWorkspaceEvent) -> Result<()>,
    {
        self.diarize_with_workspace_observer_impl(audio, sample_rate, config, observer)
    }

    fn diarize_with_workspace_observer_impl<F>(
        &self,
        audio: &[f32],
        sample_rate: u32,
        config: &DiarizationConfig,
        mut observer: F,
    ) -> Result<DiarizationResult>
    where
        F: FnMut(SortformerWorkspaceEvent) -> Result<()>,
    {
        if audio.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }
        if sample_rate == 0 {
            return Err(Error::InvalidInput("Invalid sample rate: 0".to_string()));
        }
        self.validate_speaker_limits(config)?;

        let resampled_samples;
        let samples = if sample_rate == TARGET_SAMPLE_RATE {
            audio
        } else {
            resampled_samples =
                crate::audio::resample_mono_high_quality(audio, sample_rate, TARGET_SAMPLE_RATE)?;
            &resampled_samples
        };

        let duration_secs = samples.len() as f32 / TARGET_SAMPLE_RATE as f32;
        if samples.is_empty() {
            return Ok(DiarizationResult {
                segments: Vec::new(),
                duration_secs,
                speaker_count: 0,
            });
        }
        let workspace_estimate = self.model.workspace_estimate(samples.len())?;
        let workspace = SortformerWorkspaceGuard::new(&mut observer, workspace_estimate)?;

        let (speaker_probs, frame_stride_samples) =
            self.model.infer_speaker_probabilities(samples)?;
        let result =
            self.segments_from_probabilities(samples, speaker_probs, frame_stride_samples, config);
        workspace.release()?;
        Ok(result)
    }

    fn validate_speaker_limits(&self, config: &DiarizationConfig) -> Result<()> {
        crate::catalog::validate_diarization_speaker_limits(
            config.min_speakers,
            config.max_speakers,
            self.model.num_speakers,
        )
        .map_err(Error::InvalidInput)
    }

    /// Start a diarization of 16 kHz `audio` that runs in steps of about
    /// `step_audio_secs` each (PIPE-8). Run [`Self::advance_stream`] until the
    /// stream is finished, then [`Self::finish_stream_physical`]; the result
    /// is the one [`Self::diarize`] gives, bit for bit.
    pub(crate) fn begin_stream(
        &self,
        audio: &[f32],
        config: &DiarizationConfig,
        step_audio_secs: f32,
    ) -> Result<SortformerStream> {
        if audio.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }
        self.validate_speaker_limits(config)?;
        let cfg = self.model.streaming.ok_or_else(|| {
            Error::InferenceError("staged Sortformer requires the bounded streaming path".into())
        })?;
        let feature_frames = self.model.preprocessor.feature_frame_count(audio.len());
        Ok(self.model.begin_stream(
            audio.len(),
            feature_frames,
            cfg,
            chunks_per_stream_step(cfg, self.model.preprocessor.hop_length, step_audio_secs),
        ))
    }

    /// Run the stream's next step under the per-stage workspace observer.
    pub(crate) fn advance_stream<F>(
        &self,
        stream: &mut SortformerStream,
        audio: &[f32],
        mut observer: F,
    ) -> Result<()>
    where
        F: FnMut(SortformerWorkspaceEvent) -> Result<()>,
    {
        let workspace_estimate = self.model.workspace_estimate(audio.len())?;
        let workspace = SortformerWorkspaceGuard::new(&mut observer, workspace_estimate)?;
        self.model.advance_stream(stream, audio)?;
        workspace.release()
    }

    /// Commit a finished stream's final cache to the invocation state and
    /// turn its probabilities into speaker segments.
    pub(crate) fn finish_stream_physical<F>(
        &self,
        stream: SortformerStream,
        audio: &[f32],
        config: &DiarizationConfig,
        state: &mut InvocationTensorLease,
        mut observer: F,
    ) -> Result<DiarizationResult>
    where
        F: FnMut(SortformerWorkspaceEvent) -> Result<()>,
    {
        if state.domain() != physical::SORTFORMER_STREAMING_STATE_DOMAIN {
            return Err(Error::InferenceError(
                "Sortformer received a foreign physical streaming domain".into(),
            ));
        }
        if stream.sample_count != audio.len() {
            return Err(Error::InferenceError(format!(
                "Sortformer stream over {} samples finished with {}",
                stream.sample_count,
                audio.len()
            )));
        }
        let workspace_estimate = self.model.workspace_estimate(audio.len())?;
        let workspace = SortformerWorkspaceGuard::new(&mut observer, workspace_estimate)?;
        let speaker_probs = if stream.feature_frames == 0 {
            Vec::new()
        } else {
            self.model.finish_stream(stream, Some(state))?
        };
        let result = self.segments_from_probabilities(
            audio,
            speaker_probs,
            self.model.probs_frame_stride_samples()?,
            config,
        );
        workspace.release()?;
        Ok(result)
    }

    /// Speaker segments from the probability rows of 16 kHz `samples`.
    fn segments_from_probabilities(
        &self,
        samples: &[f32],
        speaker_probs: Vec<Vec<f32>>,
        frame_stride_samples: usize,
        config: &DiarizationConfig,
    ) -> DiarizationResult {
        let duration_secs = samples.len() as f32 / TARGET_SAMPLE_RATE as f32;
        // Probability rows expand to the 10 ms timeline either because the
        // head runs at the encoded rate (v2.1, repeated in post-processing)
        // or because the upsampler already emitted 10 ms rows (Nemotron-3).
        let frame_repeat = (frame_stride_samples / self.model.preprocessor.hop_length).max(1);
        if speaker_probs.is_empty() {
            return DiarizationResult {
                segments: Vec::new(),
                duration_secs,
                speaker_count: 0,
            };
        }

        let explicit_min_speech_ms = config
            .min_speech_duration_ms
            .filter(|value| value.is_finite())
            .map(|value| {
                value.clamp(
                    frame_stride_samples as f32 * 1000.0 / TARGET_SAMPLE_RATE as f32,
                    5000.0,
                )
            });
        let explicit_min_silence_ms = config
            .min_silence_duration_ms
            .filter(|value| value.is_finite())
            .map(|value| {
                value.clamp(
                    frame_stride_samples as f32 * 1000.0 / TARGET_SAMPLE_RATE as f32,
                    5000.0,
                )
            });
        // D3: NeMo has no external VAD gate; Sortformer is itself the speech
        // detector. The opt-in gate keeps whole probabilities and clips the
        // post-processed segments to padded VAD speech regions instead.
        let vad_regions = if resolve_vad_gate(config) {
            let min_speech_ms = explicit_min_speech_ms
                .unwrap_or(DEFAULT_MIN_SPEECH_MS)
                .clamp(
                    frame_stride_samples as f32 * 1000.0 / TARGET_SAMPLE_RATE as f32,
                    5000.0,
                );
            let min_silence_ms = explicit_min_silence_ms
                .unwrap_or(DEFAULT_MIN_SILENCE_MS)
                .clamp(
                    frame_stride_samples as f32 * 1000.0 / TARGET_SAMPLE_RATE as f32,
                    5000.0,
                );
            let vad_mask = sortformer_vad_frame_mask(
                samples,
                speaker_probs.len(),
                frame_stride_samples,
                min_speech_ms,
                min_silence_ms,
            );
            Some(padded_vad_regions(&vad_mask, frame_stride_samples))
        } else {
            None
        };
        let gated_probs = speaker_probs;

        let num_speakers = self.model.num_speakers;

        // Limits above the channel count were refused up front.
        let max_speakers = config.max_speakers.unwrap_or(num_speakers).max(1);
        let min_speakers = config.min_speakers.unwrap_or(1).clamp(1, max_speakers);
        let limit_speaker_channels = should_limit_speaker_channels(config);

        let postprocessing_params =
            resolve_postprocessing_params(config, explicit_min_speech_ms, explicit_min_silence_ms);

        let mut raw_segments = Vec::<RawSegment>::new();
        let mut speaker_stats = Vec::<SpeakerActivityStats>::new();
        for speaker_idx in 0..num_speakers {
            let mut speaker_segments = predlist_to_timestamps(
                &gated_probs,
                speaker_idx,
                &postprocessing_params,
                frame_repeat,
            );
            if let Some(regions) = &vad_regions {
                speaker_segments = gate_segments(&speaker_segments, regions);
            }
            if speaker_segments.is_empty() {
                if limit_speaker_channels {
                    speaker_stats.push(SpeakerActivityStats {
                        speaker_idx,
                        total_duration_secs: 0.0,
                        peak_probability: 0.0,
                        segment_count: 0,
                    });
                }
                continue;
            }

            let peak_probability = gated_probs
                .iter()
                .map(|row| row[speaker_idx])
                .fold(0.0f32, f32::max);
            let total_duration_secs = speaker_segments
                .iter()
                .map(|(start_secs, end_secs)| (end_secs - start_secs).max(0.0))
                .sum::<f32>();

            for (start_secs, end_secs) in speaker_segments {
                let start_secs = start_secs.clamp(0.0, duration_secs);
                let end_secs = end_secs.clamp(0.0, duration_secs);
                if end_secs <= start_secs {
                    continue;
                }
                let confidence = average_speaker_probability_for_range(
                    &gated_probs,
                    speaker_idx,
                    start_secs,
                    end_secs,
                    frame_stride_samples,
                );
                raw_segments.push(RawSegment {
                    speaker_idx,
                    start_secs,
                    end_secs,
                    confidence,
                });
            }

            if limit_speaker_channels {
                speaker_stats.push(SpeakerActivityStats {
                    speaker_idx,
                    total_duration_secs,
                    peak_probability,
                    segment_count: raw_segments
                        .iter()
                        .filter(|segment| segment.speaker_idx == speaker_idx)
                        .count(),
                });
            }
        }

        if raw_segments.is_empty() {
            return DiarizationResult {
                segments: Vec::new(),
                duration_secs,
                speaker_count: 0,
            };
        }

        if limit_speaker_channels {
            let selected_speakers =
                select_speaker_channels(&speaker_stats, min_speakers, max_speakers, num_speakers);
            reassign_dropped_speaker_segments(
                &mut raw_segments,
                &selected_speakers,
                &gated_probs,
                frame_stride_samples,
            );
        }

        raw_segments.sort_by(|a, b| {
            a.start_secs
                .total_cmp(&b.start_secs)
                .then(a.speaker_idx.cmp(&b.speaker_idx))
        });

        let mut speaker_first_start = BTreeMap::<usize, f32>::new();
        for segment in &raw_segments {
            speaker_first_start
                .entry(segment.speaker_idx)
                .and_modify(|cur| {
                    if segment.start_secs < *cur {
                        *cur = segment.start_secs;
                    }
                })
                .or_insert(segment.start_secs);
        }

        let mut ordered = speaker_first_start.into_iter().collect::<Vec<_>>();
        ordered.sort_by(|a, b| a.1.total_cmp(&b.1));
        let speaker_remap = ordered
            .iter()
            .enumerate()
            .map(|(i, (speaker_idx, _))| (*speaker_idx, i))
            .collect::<HashMap<_, _>>();

        let speaker_labels = (0..ordered.len())
            .map(|idx| format!("SPEAKER_{idx:02}"))
            .collect::<Vec<_>>();

        let mut segments = raw_segments
            .into_iter()
            .map(|segment| {
                let remapped = speaker_remap
                    .get(&segment.speaker_idx)
                    .copied()
                    .unwrap_or(0);
                DiarizationSegment {
                    speaker: speaker_labels
                        .get(remapped)
                        .cloned()
                        .unwrap_or_else(|| format!("SPEAKER_{remapped:02}")),
                    start_secs: segment.start_secs,
                    end_secs: segment.end_secs,
                    confidence: segment.confidence,
                }
            })
            .collect::<Vec<_>>();

        merge_adjacent_segments(&mut segments, 0.0);
        segments.sort_by(|a, b| {
            a.start_secs
                .total_cmp(&b.start_secs)
                .then(a.speaker.cmp(&b.speaker))
        });

        let speaker_count = segments
            .iter()
            .map(|segment| segment.speaker.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len();

        DiarizationResult {
            segments,
            duration_secs,
            speaker_count,
        }
    }

    pub fn variant(&self) -> ModelVariant {
        self.variant
    }
}

fn sortformer_model_device(device_profile: &DeviceProfile) -> Device {
    if sortformer_uses_selected_model_device(BackendKind::from(device_profile.kind)) {
        device_profile.device.clone()
    } else {
        Device::Cpu
    }
}

/// Whether the weights live on the `backend`'s device: only CUDA; Metal and
/// CPU loads keep them in host memory (on Metal, outside what the Metal
/// device observer sees). Load admission prices the same placement.
pub(crate) fn sortformer_uses_selected_model_device(backend: BackendKind) -> bool {
    backend == BackendKind::Cuda
}

#[derive(Debug, Clone)]
struct RawSegment {
    speaker_idx: usize,
    start_secs: f32,
    end_secs: f32,
    confidence: Option<f32>,
}

#[derive(Debug, Clone, Copy)]
struct SpeakerActivityStats {
    speaker_idx: usize,
    total_duration_secs: f32,
    peak_probability: f32,
    segment_count: usize,
}

#[derive(Debug, Clone, Copy)]
struct PostProcessingParams {
    onset: f32,
    offset: f32,
    pad_onset: f32,
    pad_offset: f32,
    min_duration_on: f32,
    min_duration_off: f32,
    filter_speech_first: bool,
}

struct SortformerInferenceModel {
    device: Device,
    separate_device_memory: bool,
    preprocessor: SortformerPreprocessor,
    encoder: SortformerAcousticEncoder,
    encoder_proj: Linear,
    expander: SortformerFrameExpander,
    head: SortformerSpeakerHead,
    num_speakers: usize,
    streaming: Option<SortformerStreamingConfig>,
    streaming_profile: SortformerStreamingProfile,
    /// Nemotron-3's learned silence embedding used for cache silence slots;
    /// v2.1 keeps None and tracks a running silence mean instead.
    silence_embedding: Option<Vec<f32>>,
    streaming_counters: SortformerStreamingCounters,
}

impl SortformerInferenceModel {
    fn load(
        vb: &VarBuilder,
        preprocessor_cfg: SortformerPreprocessorConfig,
        profile: SortformerStreamingProfile,
        num_spks: usize,
        streaming_mode: bool,
        graph: SortformerGraphSpec,
        modules_cfg: SortformerModulesConfig,
        device: Device,
        separate_device_memory: bool,
    ) -> Result<Self> {
        if !streaming_mode {
            return Err(Error::ModelLoadError(
                "offline Sortformer is not supported by the bounded production runtime".to_string(),
            ));
        }
        let feature_bins = preprocessor_cfg.features.unwrap_or(128);
        let preprocessor = SortformerPreprocessor::load(vb, preprocessor_cfg)?;
        let encoder_kind = graph.encoder_kind();
        let encoder = match graph {
            SortformerGraphSpec::Conformer {
                heads, xscaling, ..
            } => {
                let encoder = SortformerConformerEncoder::load(vb.pp("encoder"), xscaling)?;
                if let Some(layer) = encoder
                    .layers
                    .iter()
                    .find(|layer| layer.self_attn.num_heads != heads)
                {
                    return Err(Error::ModelLoadError(format!(
                        "Sortformer config encoder.n_heads = {heads} does not match the checkpoint's {} relative-position heads",
                        layer.self_attn.num_heads
                    )));
                }
                SortformerAcousticEncoder::Conformer(encoder)
            }
            SortformerGraphSpec::FeatureStackingRope {
                heads,
                stacking_factor,
                rope_base,
            } => SortformerAcousticEncoder::FeatureStackingRope(SortformerRopeEncoder::load(
                vb.pp("encoder"),
                feature_bins,
                heads,
                stacking_factor,
                rope_base,
            )?),
        };

        let encoder_proj_w = vb
            .pp("sortformer_modules.encoder_proj")
            .get_unchecked_dtype("weight", DType::F32)?;
        let (proj_out, proj_in) = encoder_proj_w.dims2()?;
        let encoder_proj =
            mlx::load_linear(proj_in, proj_out, vb.pp("sortformer_modules.encoder_proj"))?;

        let expander = match graph {
            SortformerGraphSpec::Conformer {
                transformer_heads, ..
            } => {
                SortformerFrameExpander::SortformerTransformer(SortformerTransformerEncoder::load(
                    vb.pp("transformer_encoder"),
                    transformer_heads,
                )?)
            }
            SortformerGraphSpec::FeatureStackingRope { .. } => {
                SortformerFrameExpander::SubpixelUpsampler(SortformerSubpixelUpsampler::load(
                    vb.pp("sortformer_modules"),
                )?)
            }
        };
        let head = SortformerSpeakerHead::load(vb.pp("sortformer_modules"), num_spks)?;
        let silence_embedding = match encoder_kind {
            SortformerEncoderKind::Conformer => None,
            SortformerEncoderKind::FeatureStackingRope => {
                let silence = vb
                    .pp("sortformer_modules")
                    .get_unchecked_dtype("learnable_sil_emb", DType::F32)?
                    .to_vec1::<f32>()?;
                if silence.len() != proj_in {
                    return Err(Error::ModelLoadError(format!(
                        "Sortformer learnable silence embedding width {} does not match the encoder width {proj_in}",
                        silence.len()
                    )));
                }
                Some(silence)
            }
        };
        let streaming = resolve_streaming_config(
            profile,
            &modules_cfg,
            encoder.d_model(),
            num_spks,
            encoder_kind,
        )?;

        let model = Self {
            device,
            separate_device_memory,
            preprocessor,
            encoder,
            encoder_proj,
            expander,
            head,
            num_speakers: num_spks,
            streaming: Some(streaming),
            streaming_profile: profile,
            silence_embedding,
            streaming_counters: SortformerStreamingCounters::default(),
        };
        model.validate_production_topology(proj_in, proj_out)?;
        Ok(model)
    }

    fn workspace_topology(&self) -> Result<SortformerWorkspaceTopology> {
        let encoder = match &self.encoder {
            SortformerAcousticEncoder::Conformer(encoder) => {
                let conformer_layer = encoder.layers.first().ok_or_else(|| {
                    Error::ModelLoadError("Sortformer Conformer encoder has no layers".to_string())
                })?;
                if encoder.layers.iter().any(|layer| {
                    layer.d_model != conformer_layer.d_model
                        || layer.ff_dim != conformer_layer.ff_dim
                        || layer.self_attn.num_heads != conformer_layer.self_attn.num_heads
                        || layer.self_attn.head_dim != conformer_layer.self_attn.head_dim
                }) {
                    return Err(Error::ModelLoadError(
                        "non-uniform Sortformer Conformer layers are not supported by the production workspace envelope"
                            .to_string(),
                    ));
                }
                SortformerEncoderTopology::Conformer {
                    conv_channels: encoder.pre_encode.out_channels,
                    layers: encoder.layers.len(),
                    d_model: encoder.d_model,
                    ff_dim: conformer_layer.ff_dim,
                    heads: conformer_layer.self_attn.num_heads,
                }
            }
            SortformerAcousticEncoder::FeatureStackingRope(encoder) => {
                SortformerEncoderTopology::FeatureStackingRope {
                    layers: encoder.layers.len(),
                    d_model: encoder.d_model,
                    ff_dim: encoder.ff_dim,
                    heads: encoder.num_heads,
                }
            }
        };
        let expander = match &self.expander {
            SortformerFrameExpander::SortformerTransformer(transformer) => {
                let transformer_layer = transformer.layers.first().ok_or_else(|| {
                    Error::ModelLoadError(
                        "Sortformer transformer encoder has no layers".to_string(),
                    )
                })?;
                if transformer.layers.iter().any(|layer| {
                    layer.d_model != transformer_layer.d_model
                        || layer.inner_size != transformer_layer.inner_size
                        || layer.num_heads != transformer_layer.num_heads
                        || layer.head_dim != transformer_layer.head_dim
                }) {
                    return Err(Error::ModelLoadError(
                        "non-uniform Sortformer transformer layers are not supported by the production workspace envelope"
                            .to_string(),
                    ));
                }
                SortformerFrameExpanderTopology::SortformerTransformer {
                    layers: transformer.layers.len(),
                    d_model: transformer_layer.d_model,
                    inner_dim: transformer_layer.inner_size,
                    heads: transformer_layer.num_heads,
                }
            }
            SortformerFrameExpander::SubpixelUpsampler(upsampler) => {
                SortformerFrameExpanderTopology::SubpixelUpsampler {
                    d_model: upsampler.d_model,
                    upsample_factor: upsampler.upsample_factor,
                }
            }
        };
        Ok(SortformerWorkspaceTopology {
            feature_bins: self.preprocessor.n_mels,
            n_fft: self.preprocessor.n_fft,
            hop_length: self.preprocessor.hop_length,
            encoder,
            expander,
        })
    }

    fn validate_production_topology(
        &self,
        projection_in: usize,
        projection_out: usize,
    ) -> Result<()> {
        if self.preprocessor.sample_rate != TARGET_SAMPLE_RATE as usize
            || self.preprocessor.normalize != SortformerFeatureNormalize::None
        {
            return Err(Error::ModelLoadError(format!(
                "unsupported Sortformer preprocessing/projection topology: sample_rate={}, normalize={:?}, projection=[{}, {}]",
                self.preprocessor.sample_rate,
                self.preprocessor.normalize,
                projection_out,
                projection_in
            )));
        }
        match (&self.encoder, &self.expander) {
            (
                SortformerAcousticEncoder::Conformer(_),
                SortformerFrameExpander::SortformerTransformer(_),
            ) => {
                if projection_in != PRODUCTION_CONFORMER_D_MODEL
                    || projection_out != PRODUCTION_TRANSFORMER_D_MODEL
                    || self.head.hidden_dim != PRODUCTION_TRANSFORMER_D_MODEL
                {
                    return Err(Error::ModelLoadError(format!(
                        "unsupported Sortformer projection topology: projection=[{projection_in}, {projection_out}], head={}",
                        self.head.hidden_dim
                    )));
                }
            }
            (
                SortformerAcousticEncoder::FeatureStackingRope(encoder),
                SortformerFrameExpander::SubpixelUpsampler(upsampler),
            ) => {
                if projection_in != encoder.d_model
                    || projection_out != NEMOTRON3_HEAD_D_MODEL
                    || self.head.hidden_dim != NEMOTRON3_HEAD_D_MODEL
                    || upsampler.d_model != NEMOTRON3_HEAD_D_MODEL
                    || upsampler.upsample_factor != TS_VAD_UNIT_FRAME_COUNT
                {
                    return Err(Error::ModelLoadError(format!(
                        "unsupported Sortformer rope projection topology: projection=[{projection_in}, {projection_out}], head={}",
                        self.head.hidden_dim
                    )));
                }
            }
            _ => {
                return Err(Error::ModelLoadError(
                    "mismatched Sortformer encoder/expander graph".to_string(),
                ))
            }
        }
        let streaming = self.streaming.ok_or_else(|| {
            Error::ModelLoadError(
                "offline Sortformer is not supported by the bounded production runtime".to_string(),
            )
        })?;
        self.workspace_topology()?.validate_production(streaming)
    }

    /// Sample stride between output probability rows: v2.1 emits one row per
    /// encoded 80 ms frame, Nemotron-3 one row per 10 ms mel frame.
    fn probs_frame_stride_samples(&self) -> Result<usize> {
        let cfg = self.streaming.ok_or_else(|| {
            Error::InferenceError(
                "offline Sortformer reached the bounded production path".to_string(),
            )
        })?;
        Ok(self.preprocessor.hop_length * cfg.subsampling_factor
            / cfg.output_frames_per_encoded_frame)
    }

    fn output_frames_per_encoded_frame(&self) -> Result<usize> {
        let cfg = self.streaming.ok_or_else(|| {
            Error::InferenceError(
                "offline Sortformer reached the bounded production path".to_string(),
            )
        })?;
        Ok(cfg.output_frames_per_encoded_frame)
    }

    fn workspace_estimate(
        &self,
        target_sample_count: usize,
    ) -> Result<SortformerWorkspaceEstimate> {
        let streaming = self.streaming.ok_or_else(|| {
            Error::ModelLoadError(
                "offline Sortformer is not supported by the bounded production runtime".to_string(),
            )
        })?;
        workspace_estimate_for(
            self.workspace_topology()?,
            streaming,
            target_sample_count,
            self.separate_device_memory,
        )
    }

    fn infer_speaker_probabilities(&self, samples: &[f32]) -> Result<(Vec<Vec<f32>>, usize)> {
        let feature_frames = self.preprocessor.feature_frame_count(samples.len());
        if feature_frames == 0 {
            return Ok((Vec::new(), self.probs_frame_stride_samples()?));
        }
        let streaming_cfg = self.streaming.ok_or_else(|| {
            Error::InferenceError(
                "offline Sortformer reached the bounded production path".to_string(),
            )
        })?;
        let out = self.infer_speaker_probabilities_streaming(
            samples,
            feature_frames,
            streaming_cfg,
            None,
        )?;

        Ok((out, self.probs_frame_stride_samples()?))
    }

    fn infer_speaker_probabilities_offline(
        &self,
        features: &Tensor,
        feature_frames: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let (encoded, encoded_len) = self.encoder.forward(features, feature_frames)?;
        if encoded_len == 0 {
            return Ok(Vec::new());
        }
        let probs = self.forward_probabilities(&encoded, encoded_len)?;
        tensor_to_probability_rows(&probs, encoded_len, self.num_speakers)
    }

    fn infer_speaker_probabilities_streaming(
        &self,
        samples: &[f32],
        feature_frames: usize,
        cfg: SortformerStreamingConfig,
        physical_state: Option<&mut InvocationTensorLease>,
    ) -> Result<Vec<Vec<f32>>> {
        let mut stream = self.begin_stream(samples.len(), feature_frames, cfg, usize::MAX);
        self.advance_stream(&mut stream, samples)?;
        self.finish_stream(stream, physical_state)
    }

    /// Start a streaming pass over `sample_count` samples whose steps each
    /// run at most `chunks_per_step` chunks.
    fn begin_stream(
        &self,
        sample_count: usize,
        feature_frames: usize,
        cfg: SortformerStreamingConfig,
        chunks_per_step: usize,
    ) -> SortformerStream {
        let mut state = SortformerStreamingState::new(cfg.fc_d_model);
        if let Some(silence) = &self.silence_embedding {
            state.mean_sil_emb = silence.clone();
        }
        SortformerStream {
            cfg,
            sample_count,
            feature_frames,
            plans: plan_streaming_feature_chunks(feature_frames, cfg),
            next_plan: 0,
            chunks_per_step: chunks_per_step.max(1),
            state: Some(state),
            track_silence_mean: self.silence_embedding.is_none(),
            total_preds: Vec::new(),
            chunks: 0,
            host_reads: 0,
        }
    }

    /// Run the stream's next step: up to `chunks_per_step` chunks. The state
    /// carries over unchanged between steps, so any split of the chunks into
    /// steps gives the probabilities of one uninterrupted pass, bit for bit.
    fn advance_stream(&self, stream: &mut SortformerStream, samples: &[f32]) -> Result<()> {
        if samples.len() != stream.sample_count {
            return Err(Error::InferenceError(format!(
                "Sortformer stream over {} samples resumed with {}",
                stream.sample_count,
                samples.len()
            )));
        }
        let cfg = stream.cfg;
        let mut state = stream
            .state
            .take()
            .ok_or_else(|| Error::InferenceError("Sortformer stream state was lost".into()))?;
        let end = stream
            .next_plan
            .saturating_add(stream.chunks_per_step)
            .min(stream.plans.len());
        // The cache, FIFO and chunk embeddings stay on the model device; the
        // only per-chunk host reads are the probabilities (once) and, for the
        // running silence mean, the silent rows popped from the FIFO.
        for plan in &stream.plans[stream.next_plan..end] {
            let chunk = self
                .preprocessor
                .compute_feature_range(samples, plan.feature_start, plan.feature_end)?
                .to_device(&self.device)?
                .transpose(1, 2)?
                .contiguous()?;
            let (chunk_pre_encoded, chunk_pre_encoded_len) =
                self.encoder.pre_encode(&chunk, chunk.dim(1)?)?;
            if chunk_pre_encoded_len == 0 {
                continue;
            }

            let chunk_rows =
                EmbeddingRows::new(chunk_pre_encoded.i((0, ..chunk_pre_encoded_len, ..))?)?;
            let composite = EmbeddingRows::cat(&[&state.spkcache, &state.fifo, &chunk_rows])?;
            let Some(composite) = composite.tensor() else {
                continue;
            };
            stream.chunks += 1;

            let composite_len = composite.dim(0)?;
            let (encoded, encoded_len) = self
                .encoder
                .forward_pre_encoded(&composite.unsqueeze(0)?, composite_len)?;
            let probs = self.forward_probabilities(&encoded, encoded_len)?;
            let output_rows = encoded_len * cfg.output_frames_per_encoded_frame;
            let prob_rows = tensor_to_probability_rows(&probs, output_rows, self.num_speakers)?;
            stream.host_reads += 1;
            let pooled;
            let pred_rows = if cfg.output_frames_per_encoded_frame == 1 {
                &prob_rows
            } else {
                // A trailing encoded frame over a zero-padded mel group keeps
                // its pooled probability, as in NeMo (which masks by the
                // encoder length, ceil(T / 8)) and HF (NDZ-6).
                pooled = pool_probability_rows(
                    &prob_rows,
                    encoded_len,
                    cfg.output_frames_per_encoded_frame,
                    self.num_speakers,
                )?;
                &pooled
            };
            let (updated_state, _chunk_preds, output_range) = update_streaming_state(
                state,
                &chunk_rows,
                pred_rows,
                pre_encoded_left_offset(plan.left_offset, cfg.subsampling_factor),
                pre_encoded_right_offset(plan.right_offset, cfg.subsampling_factor),
                stream.track_silence_mean,
                cfg,
            )?;
            state = updated_state;
            stream
                .total_preds
                .extend_from_slice(&prob_rows[output_range]);
        }
        stream.state = Some(state);
        stream.next_plan = end;
        self.streaming_counters
            .steps
            .fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// The probabilities of a stream that has run every chunk. With
    /// `physical_state` the final cache is committed to the invocation state.
    fn finish_stream(
        &self,
        mut stream: SortformerStream,
        physical_state: Option<&mut InvocationTensorLease>,
    ) -> Result<Vec<Vec<f32>>> {
        if !stream.is_finished() {
            return Err(Error::InferenceError(format!(
                "Sortformer stream finished after {} of {} chunks",
                stream.next_plan,
                stream.plans.len()
            )));
        }
        let cfg = stream.cfg;
        let state = stream
            .state
            .take()
            .ok_or_else(|| Error::InferenceError("Sortformer stream state was lost".into()))?;
        let host_reads = stream.host_reads + state.host_reads;
        // Nothing reads the invocation state back while the stream runs, so
        // it is committed once, with the final cache, instead of being staged
        // on the arena device after every chunk.
        if let Some(lease) = physical_state {
            if lease.arena()?.absolute_cursor() != 0 {
                return Err(Error::InferenceError(
                    "Sortformer invocation state was not reset before inference".into(),
                ));
            }
            commit_sortformer_streaming_state(lease, cfg, &state)?;
            self.streaming_counters
                .physical_state_commits
                .fetch_add(1, Ordering::Relaxed);
        }
        self.streaming_counters
            .chunks
            .fetch_add(stream.chunks, Ordering::Relaxed);
        self.streaming_counters
            .host_reads
            .fetch_add(host_reads, Ordering::Relaxed);
        let mut total_preds = stream.total_preds;
        if cfg.output_frames_per_encoded_frame > 1 {
            // The upsampler pads the final stacked group; trim the trailing
            // rows back to the mel frame count.
            total_preds.truncate(stream.feature_frames);
        }

        Ok(total_preds)
    }

    fn streaming_path_diagnostics(&self) -> serde_json::Value {
        let counters = &self.streaming_counters;
        serde_json::json!({
            "chunks": counters.chunks.load(Ordering::Relaxed),
            "steps": counters.steps.load(Ordering::Relaxed),
            "host_reads": counters.host_reads.load(Ordering::Relaxed),
            "physical_state_commits": counters.physical_state_commits.load(Ordering::Relaxed),
        })
    }

    fn forward_probabilities(&self, encoded: &Tensor, encoded_len: usize) -> Result<Tensor> {
        let mut x = encoded.i((.., ..encoded_len, ..))?;
        x = x.apply(&self.encoder_proj)?;
        x = self.expander.forward(&x)?;
        let probs = self.head.forward(&x)?;
        let (_, output_rows, speaker_dim) = probs.dims3()?;
        if speaker_dim != self.num_speakers
            || output_rows != encoded_len * self.output_frames_per_encoded_frame()?
        {
            return Err(Error::InferenceError(format!(
                "Unexpected Sortformer probability shape [{output_rows}, {speaker_dim}]; expected [{}, {}]",
                encoded_len * self.output_frames_per_encoded_frame()?,
                self.num_speakers
            )));
        }
        Ok(probs)
    }
}

fn resolve_streaming_config(
    profile: SortformerStreamingProfile,
    modules_cfg: &SortformerModulesConfig,
    encoder_d_model: usize,
    num_spks: usize,
    encoder_kind: SortformerEncoderKind,
) -> Result<SortformerStreamingConfig> {
    let cfg = SortformerStreamingConfig {
        fc_d_model: modules_cfg.fc_d_model.unwrap_or(encoder_d_model),
        num_speakers: num_spks,
        subsampling_factor: modules_cfg
            .subsampling_factor
            .unwrap_or(TS_VAD_UNIT_FRAME_COUNT),
        output_frames_per_encoded_frame: match encoder_kind {
            SortformerEncoderKind::Conformer => 1,
            SortformerEncoderKind::FeatureStackingRope => TS_VAD_UNIT_FRAME_COUNT,
        },
        spkcache_len: modules_cfg.spkcache_len.unwrap_or(188),
        fifo_len: modules_cfg.fifo_len.unwrap_or(0),
        chunk_len: modules_cfg.chunk_len.unwrap_or(188),
        spkcache_update_period: modules_cfg.spkcache_update_period.unwrap_or(188),
        chunk_left_context: modules_cfg.chunk_left_context.unwrap_or(1),
        chunk_right_context: modules_cfg.chunk_right_context.unwrap_or(1),
        spkcache_sil_frames_per_spk: modules_cfg.spkcache_sil_frames_per_spk.unwrap_or(3),
        pred_score_threshold: modules_cfg
            .pred_score_threshold
            .unwrap_or(0.25)
            .clamp(1e-4, 0.99),
        scores_boost_latest: modules_cfg.scores_boost_latest.unwrap_or(0.05).max(0.0),
        sil_threshold: modules_cfg.sil_threshold.unwrap_or(0.2).max(0.0),
        strong_boost_rate: modules_cfg.strong_boost_rate.unwrap_or(0.75).max(0.0),
        weak_boost_rate: modules_cfg.weak_boost_rate.unwrap_or(1.5).max(0.0),
        min_pos_scores_rate: modules_cfg
            .min_pos_scores_rate
            .unwrap_or(0.5)
            .clamp(0.0, 1.0),
    };

    apply_streaming_profile(cfg, encoder_kind, profile).validate()
}

fn apply_streaming_profile(
    mut cfg: SortformerStreamingConfig,
    encoder_kind: SortformerEncoderKind,
    profile: SortformerStreamingProfile,
) -> SortformerStreamingConfig {
    if let Some(geometry) = profile.card_geometry(encoder_kind) {
        cfg.chunk_len = geometry.chunk_len;
        cfg.chunk_right_context = geometry.chunk_right_context;
        cfg.fifo_len = geometry.fifo_len;
        cfg.spkcache_update_period = geometry.spkcache_update_period;
        cfg.spkcache_len = geometry.spkcache_len;
    }
    cfg
}

/// Reads `IZWI_SORTFORMER_STREAMING_PROFILE`. Every profile is published
/// for both checkpoints; an unknown value logs and keeps the default rather
/// than failing the load.
fn resolve_streaming_profile() -> SortformerStreamingProfile {
    let Ok(value) = std::env::var("IZWI_SORTFORMER_STREAMING_PROFILE") else {
        return SortformerStreamingProfile::DEFAULT;
    };
    match value.trim().to_ascii_lowercase().as_str() {
        "" => SortformerStreamingProfile::DEFAULT,
        "model" => SortformerStreamingProfile::Model,
        "low_latency" | "low-latency" | "low" => SortformerStreamingProfile::LowLatency,
        "very_high_latency" | "very-high-latency" | "high_latency" | "high-latency" | "high"
        | "offline" => SortformerStreamingProfile::VeryHighLatency,
        other => {
            tracing::warn!(
                "unknown IZWI_SORTFORMER_STREAMING_PROFILE {other:?}; using the {} profile",
                SortformerStreamingProfile::DEFAULT.name()
            );
            SortformerStreamingProfile::DEFAULT
        }
    }
}

/// Whole chunks in a step of about `step_audio_secs` of 16 kHz audio, at
/// least one.
fn chunks_per_stream_step(
    cfg: SortformerStreamingConfig,
    hop_length: usize,
    step_audio_secs: f32,
) -> usize {
    let chunk_samples = cfg.chunk_len * cfg.subsampling_factor * hop_length;
    let step_samples = (step_audio_secs.max(0.0) * TARGET_SAMPLE_RATE as f32) as usize;
    (step_samples / chunk_samples.max(1)).max(1)
}

fn plan_streaming_feature_chunks(
    feature_frames: usize,
    cfg: SortformerStreamingConfig,
) -> Vec<SortformerStreamingChunkPlan> {
    if feature_frames == 0 {
        return Vec::new();
    }

    let context_frames = cfg.subsampling_factor;
    let chunk_width = cfg.chunk_len * context_frames;
    let left_context = cfg.chunk_left_context * context_frames;
    let right_context = cfg.chunk_right_context * context_frames;

    let mut plans = Vec::new();
    let mut start = 0usize;
    while start < feature_frames {
        let left_offset = left_context.min(start);
        let end = (start + chunk_width).min(feature_frames);
        let right_offset = right_context.min(feature_frames.saturating_sub(end));
        plans.push(SortformerStreamingChunkPlan {
            feature_start: start - left_offset,
            feature_end: end + right_offset,
            left_offset,
            right_offset,
        });
        start = end;
    }
    plans
}

fn pre_encoded_left_offset(left_offset: usize, subsampling_factor: usize) -> usize {
    ((left_offset as f32) / (subsampling_factor as f32)).round() as usize
}

fn pre_encoded_right_offset(right_offset: usize, subsampling_factor: usize) -> usize {
    ((right_offset as f32) / (subsampling_factor as f32)).ceil() as usize
}

fn tensor_to_probability_rows(
    tensor: &Tensor,
    row_count: usize,
    num_speakers: usize,
) -> Result<Vec<Vec<f32>>> {
    if row_count == 0 {
        return Ok(Vec::new());
    }

    let view = tensor.i((0, ..row_count, ..))?;
    let (_, speaker_dim) = view.dims2()?;
    if speaker_dim != num_speakers {
        return Err(Error::InferenceError(format!(
            "Unexpected Sortformer probability tensor width {}; expected {}",
            speaker_dim, num_speakers
        )));
    }
    crate::models::shared::telemetry::record_host_read(DType::F32, row_count * num_speakers);
    let values = view.flatten_all()?.to_vec1::<f32>()?;
    Ok(values
        .chunks(num_speakers)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>())
}

/// Pools the upsampled probability rows back to the encoded frame rate for
/// streaming cache management, matching the reference `_pool_probs`
/// (sigmoid has already been applied by the speaker head). The rows are
/// already on the host; the arithmetic is Candle's CPU `mean`: an in-order
/// sum from zero, then a multiply by `1 / factor`.
fn pool_probability_rows(
    rows: &[Vec<f32>],
    encoded_len: usize,
    upsample_factor: usize,
    num_speakers: usize,
) -> Result<Vec<Vec<f32>>> {
    if rows.len() != encoded_len * upsample_factor
        || rows.iter().any(|row| row.len() != num_speakers)
    {
        return Err(Error::InferenceError(format!(
            "unexpected Sortformer upsampled probability rows ({}); expected [{}, {num_speakers}]",
            rows.len(),
            encoded_len * upsample_factor
        )));
    }
    let scale = (1.0 / upsample_factor as f64) as f32;
    Ok(rows
        .chunks_exact(upsample_factor.max(1))
        .map(|group| {
            let mut sum = vec![0.0f32; num_speakers];
            for row in group {
                for (acc, value) in sum.iter_mut().zip(row) {
                    *acc += value;
                }
            }
            sum.into_iter().map(|value| value * scale).collect()
        })
        .collect())
}

fn update_streaming_state(
    mut state: SortformerStreamingState,
    chunk_rows: &EmbeddingRows,
    preds: &[Vec<f32>],
    lc: usize,
    rc: usize,
    track_silence_mean: bool,
    cfg: SortformerStreamingConfig,
) -> Result<(SortformerStreamingState, Vec<Vec<f32>>, std::ops::Range<usize>)> {
    let spkcache_len = state.spkcache.len();
    let fifo_len = state.fifo.len();
    if preds.len() < spkcache_len + fifo_len + chunk_rows.len() {
        return Err(Error::InferenceError(format!(
            "Streaming Sortformer prediction rows {} do not cover spkcache ({spkcache_len}) + fifo ({fifo_len}) + chunk ({})",
            preds.len(),
            chunk_rows.len()
        )));
    }

    state.fifo_preds = preds[spkcache_len..spkcache_len + fifo_len].to_vec();

    let chunk_valid_len = chunk_rows.len().saturating_sub(lc + rc);
    let chunk_start = lc.min(chunk_rows.len());
    let chunk_end = (chunk_start + chunk_valid_len).min(chunk_rows.len());
    let chunk_payload = chunk_rows.narrow(chunk_start, chunk_end - chunk_start)?;
    let chunk_preds =
        preds[spkcache_len + fifo_len + chunk_start..spkcache_len + fifo_len + chunk_end].to_vec();
    let output_start = (spkcache_len + fifo_len + chunk_start)
        * cfg.output_frames_per_encoded_frame;
    let output_end =
        (spkcache_len + fifo_len + chunk_end) * cfg.output_frames_per_encoded_frame;

    state.fifo = EmbeddingRows::cat(&[&state.fifo, &chunk_payload])?;
    state.fifo_preds.extend(chunk_preds.clone());

    if fifo_len + chunk_payload.len() > cfg.fifo_len {
        // NeMo: max(update_period, chunk - fifo_len + fifo) in signed
        // arithmetic. The branch guarantees chunk + fifo > fifo_len, so
        // subtracting from the sum never saturates.
        let overflow = (chunk_payload.len() + fifo_len) - cfg.fifo_len;
        let pop_out_len = cfg
            .spkcache_update_period
            .max(overflow)
            .min(fifo_len + chunk_payload.len());
        let pop_out_embs = state.fifo.narrow(0, pop_out_len)?;
        let pop_out_preds = state.fifo_preds[..pop_out_len].to_vec();

        if track_silence_mean {
            update_silence_profile(&mut state, &pop_out_embs, &pop_out_preds, cfg.sil_threshold)?;
        }
        state.fifo = state
            .fifo
            .narrow(pop_out_len, state.fifo.len() - pop_out_len)?;
        state.fifo_preds.drain(..pop_out_len);

        let prev_spkcache_len = state.spkcache.len();
        state.spkcache = EmbeddingRows::cat(&[&state.spkcache, &pop_out_embs])?;
        if let Some(spkcache_preds) = state.spkcache_preds.as_mut() {
            spkcache_preds.extend(pop_out_preds);
        } else if state.spkcache.len() > cfg.spkcache_len {
            let mut seeded_preds = preds[..prev_spkcache_len].to_vec();
            seeded_preds.extend(pop_out_preds);
            state.spkcache_preds = Some(seeded_preds);
        }

        if state.spkcache.len() > cfg.spkcache_len {
            let spkcache_preds = state.spkcache_preds.as_ref().ok_or_else(|| {
                Error::InferenceError(
                    "Sortformer speaker cache predictions were not initialized".to_string(),
                )
            })?;
            let (compressed_cache, compressed_preds) =
                compress_spkcache(&state.spkcache, spkcache_preds, &state.mean_sil_emb, cfg)?;
            state.spkcache = compressed_cache;
            state.spkcache_preds = Some(compressed_preds);
        }
    }

    Ok((state, chunk_preds, output_start..output_end))
}

fn update_silence_profile(
    state: &mut SortformerStreamingState,
    emb_seq: &EmbeddingRows,
    preds: &[Vec<f32>],
    sil_threshold: f32,
) -> Result<()> {
    if emb_seq.len() != preds.len() {
        return Err(Error::InferenceError(format!(
            "Sortformer silence update has {} embeddings for {} prediction rows",
            emb_seq.len(),
            preds.len()
        )));
    }
    let silent = preds
        .iter()
        .enumerate()
        .filter(|(_, pred)| pred.iter().copied().sum::<f32>() < sil_threshold)
        .map(|(index, _)| index as u32)
        .collect::<Vec<_>>();
    let Some(silent_rows) = emb_seq.index_select(&silent)?.0 else {
        return Ok(());
    };
    // Only the silent rows come to the host, in frame order, so the running
    // mean keeps its sequential f32 arithmetic.
    let (rows, width) = silent_rows.dims2()?;
    crate::models::shared::telemetry::record_host_read(DType::F32, rows * width);
    state.host_reads += 1;
    for emb in silent_rows.to_vec2::<f32>()? {
        let total = state.n_sil_frames as f32;
        for (idx, value) in emb.iter().copied().enumerate() {
            state.mean_sil_emb[idx] = if state.n_sil_frames == 0 {
                value
            } else {
                (state.mean_sil_emb[idx] * total + value) / (total + 1.0)
            };
        }
        state.n_sil_frames += 1;
    }
    Ok(())
}

fn compress_spkcache(
    emb_seq: &EmbeddingRows,
    preds: &[Vec<f32>],
    mean_sil_emb: &[f32],
    cfg: SortformerStreamingConfig,
) -> Result<(EmbeddingRows, Vec<Vec<f32>>)> {
    if emb_seq.len() != preds.len() {
        return Err(Error::InferenceError(format!(
            "Sortformer speaker cache compression length mismatch: {} embeddings vs {} prediction rows",
            emb_seq.len(),
            preds.len()
        )));
    }

    let spkcache_len_per_spk =
        cfg.spkcache_len / cfg.num_speakers - cfg.spkcache_sil_frames_per_spk;
    let strong_boost_per_spk =
        ((spkcache_len_per_spk as f32) * cfg.strong_boost_rate).floor() as usize;
    let weak_boost_per_spk = ((spkcache_len_per_spk as f32) * cfg.weak_boost_rate).floor() as usize;
    let min_pos_scores_per_spk =
        ((spkcache_len_per_spk as f32) * cfg.min_pos_scores_rate).floor() as usize;

    let mut scores = get_log_pred_scores(preds, cfg.num_speakers, cfg.pred_score_threshold);
    disable_low_scores(preds, &mut scores, min_pos_scores_per_spk, cfg.num_speakers);

    if cfg.scores_boost_latest > 0.0 && emb_seq.len() > cfg.spkcache_len {
        for row in scores.iter_mut().skip(cfg.spkcache_len) {
            for score in row.iter_mut().filter(|score| score.is_finite()) {
                *score += cfg.scores_boost_latest;
            }
        }
    }

    boost_topk_scores(&mut scores, strong_boost_per_spk, 2.0, cfg.num_speakers);
    boost_topk_scores(&mut scores, weak_boost_per_spk, 1.0, cfg.num_speakers);

    let speaker_frame_span = emb_seq.len() + cfg.spkcache_sil_frames_per_spk;
    let mut candidates = Vec::with_capacity(speaker_frame_span * cfg.num_speakers);
    for speaker_idx in 0..cfg.num_speakers {
        let base = speaker_idx * speaker_frame_span;
        for (frame_idx, frame_scores) in scores.iter().enumerate() {
            candidates.push(SortformerCacheCandidate {
                flat_index: base + frame_idx,
                frame_index: Some(frame_idx),
                score: frame_scores[speaker_idx],
            });
        }
        for silence_idx in 0..cfg.spkcache_sil_frames_per_spk {
            candidates.push(SortformerCacheCandidate {
                flat_index: base + emb_seq.len() + silence_idx,
                frame_index: None,
                score: f32::INFINITY,
            });
        }
    }

    candidates.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then(a.flat_index.cmp(&b.flat_index))
    });
    let mut selected = candidates
        .into_iter()
        .take(cfg.spkcache_len)
        .collect::<Vec<_>>();
    // NeMo `_get_topk_indices` maps every -inf pick to `max_index` before
    // sorting, so disabled slots follow all kept frames and silence pads.
    // The order feeds position-dependent attention over the cache.
    selected.sort_by_key(|a| (a.score == f32::NEG_INFINITY, a.flat_index));

    // Gather on the device: row `emb_seq.len()` of the source is the
    // silence embedding.
    let silence_row = emb_seq.len() as u32;
    let mut rows = Vec::with_capacity(cfg.spkcache_len);
    let mut spkcache_preds = Vec::with_capacity(cfg.spkcache_len);
    for candidate in selected {
        match candidate.frame_index {
            Some(frame_idx) if candidate.score.is_finite() => {
                rows.push(frame_idx as u32);
                spkcache_preds.push(preds[frame_idx].clone());
            }
            _ => {
                rows.push(silence_row);
                spkcache_preds.push(vec![0.0; cfg.num_speakers]);
            }
        }
    }
    let source = if rows.contains(&silence_row) {
        let device = emb_seq
            .tensor()
            .map_or(Device::Cpu, |tensor| tensor.device().clone());
        let silence = EmbeddingRows::new(Tensor::from_slice(
            mean_sil_emb,
            (1, mean_sil_emb.len()),
            &device,
        )?)?;
        EmbeddingRows::cat(&[emb_seq, &silence])?
    } else {
        emb_seq.clone()
    };

    Ok((source.index_select(&rows)?, spkcache_preds))
}

fn get_log_pred_scores(
    preds: &[Vec<f32>],
    num_speakers: usize,
    pred_score_threshold: f32,
) -> Vec<Vec<f32>> {
    preds
        .iter()
        .map(|frame| {
            let log_one_minus = (0..num_speakers)
                .map(|speaker_idx| (1.0 - frame[speaker_idx]).clamp(pred_score_threshold, 1.0).ln())
                .collect::<Vec<f32>>();
            let log_one_minus_sum = log_one_minus.iter().copied().sum::<f32>();
            (0..num_speakers)
                .map(|speaker_idx| {
                    let log_prob = frame[speaker_idx].clamp(pred_score_threshold, 1.0).ln();
                    log_prob - log_one_minus[speaker_idx] + log_one_minus_sum - 0.5f32.ln()
                })
                .collect::<Vec<f32>>()
        })
        .collect()
}

fn disable_low_scores(
    preds: &[Vec<f32>],
    scores: &mut [Vec<f32>],
    min_pos_scores_per_spk: usize,
    num_speakers: usize,
) {
    let mut positive_counts = vec![0usize; num_speakers];
    for (pred_row, score_row) in preds.iter().zip(scores.iter_mut()) {
        for speaker_idx in 0..num_speakers {
            if pred_row[speaker_idx] <= 0.5 {
                score_row[speaker_idx] = f32::NEG_INFINITY;
            } else if score_row[speaker_idx] > 0.0 {
                positive_counts[speaker_idx] += 1;
            }
        }
    }

    for (pred_row, score_row) in preds.iter().zip(scores.iter_mut()) {
        for speaker_idx in 0..num_speakers {
            if pred_row[speaker_idx] > 0.5
                && score_row[speaker_idx].is_finite()
                && score_row[speaker_idx] <= 0.0
                && positive_counts[speaker_idx] >= min_pos_scores_per_spk
            {
                score_row[speaker_idx] = f32::NEG_INFINITY;
            }
        }
    }
}

fn boost_topk_scores(
    scores: &mut [Vec<f32>],
    n_boost_per_spk: usize,
    scale_factor: f32,
    num_speakers: usize,
) {
    if n_boost_per_spk == 0 {
        return;
    }

    for speaker_idx in 0..num_speakers {
        let mut ranked = scores
            .iter()
            .enumerate()
            .filter_map(|(frame_idx, frame_scores)| {
                frame_scores[speaker_idx]
                    .is_finite()
                    .then_some((frame_scores[speaker_idx], frame_idx))
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        for (_, frame_idx) in ranked.into_iter().take(n_boost_per_spk) {
            scores[frame_idx][speaker_idx] += scale_factor * SORTFORMER_SCORE_BOOST_DELTA;
        }
    }
}

struct SortformerPreprocessor {
    sample_rate: usize,
    n_fft: usize,
    win_length: usize,
    hop_length: usize,
    _window: Vec<f32>,
    padded_window: Vec<f32>,
    fb: Vec<f32>,
    n_mels: usize,
    n_freqs: usize,
    normalize: SortformerFeatureNormalize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SortformerFeatureNormalize {
    None,
    PerFeature,
    AllFeatures,
}

impl SortformerPreprocessor {
    fn load(vb: &VarBuilder, cfg: SortformerPreprocessorConfig) -> Result<Self> {
        let sample_rate = cfg.sample_rate.unwrap_or(TARGET_SAMPLE_RATE) as usize;
        let n_fft = cfg.n_fft.unwrap_or(512);
        let win_length =
            ((cfg.window_size.unwrap_or(0.025) * sample_rate as f32).round() as usize).max(1);
        let hop_length =
            ((cfg.window_stride.unwrap_or(0.01) * sample_rate as f32).round() as usize).max(1);
        let n_mels = cfg.features.unwrap_or(128);
        let normalize = match cfg
            .normalize
            .as_deref()
            .map(|value| value.trim().to_ascii_lowercase())
        {
            Some(value) if value == "per_feature" => SortformerFeatureNormalize::PerFeature,
            Some(value) if value == "all_features" => SortformerFeatureNormalize::AllFeatures,
            _ => SortformerFeatureNormalize::None,
        };

        let preproc_vb = vb.pp("preprocessor.featurizer");
        let window = match preproc_vb.get_unchecked_dtype("window", DType::F32) {
            Ok(window_tensor) => window_tensor.to_vec1::<f32>()?,
            Err(_) => hann_window(win_length),
        };

        let (fb, loaded_mels, loaded_freqs) = match preproc_vb.get_unchecked_dtype("fb", DType::F32)
        {
            Ok(fb_tensor) => {
                let (_, mels, freqs) = fb_tensor.dims3()?;
                let fb = fb_tensor.squeeze(0)?.flatten_all()?.to_vec1::<f32>()?;
                (fb, mels, freqs)
            }
            Err(_) => {
                let generated =
                    mel_filterbank(sample_rate, n_fft, n_mels, 0.0, sample_rate as f32 / 2.0);
                (generated, n_mels, n_fft / 2 + 1)
            }
        };

        let n_freqs = n_fft / 2 + 1;
        if loaded_freqs != n_freqs {
            return Err(Error::ModelLoadError(format!(
                "Unexpected Sortformer filterbank bins: expected {}, got {}",
                n_freqs, loaded_freqs
            )));
        }

        let mut padded_window = vec![0.0f32; n_fft];
        let src_len = window.len().min(n_fft);
        let offset = (n_fft - src_len) / 2;
        padded_window[offset..offset + src_len].copy_from_slice(&window[..src_len]);

        Ok(Self {
            sample_rate,
            n_fft,
            win_length,
            hop_length,
            _window: window,
            padded_window,
            fb,
            n_mels: loaded_mels,
            n_freqs,
            normalize,
        })
    }

    fn feature_frame_count(&self, sample_count: usize) -> usize {
        sample_count / self.hop_length
    }

    /// Compute only the feature frames needed by one streaming inference
    /// window. This is algebraically identical to slicing `compute_features`,
    /// but never materializes duration-shaped sample, padding, spectrum, or
    /// mel buffers.
    fn compute_feature_range(
        &self,
        audio: &[f32],
        feature_start: usize,
        feature_end: usize,
    ) -> Result<Tensor> {
        let feature_frames = self.feature_frame_count(audio.len());
        if feature_start > feature_end || feature_end > feature_frames {
            return Err(Error::InferenceError(format!(
                "Sortformer feature range {feature_start}..{feature_end} exceeds {feature_frames} frames"
            )));
        }
        if self.normalize != SortformerFeatureNormalize::None {
            return Err(Error::InferenceError(
                "normalized Sortformer features reached the bounded streaming path".to_string(),
            ));
        }

        let range_frames = feature_end - feature_start;
        if range_frames == 0 {
            return Tensor::zeros((1, self.n_mels, 0), DType::F32, &Device::Cpu)
                .map_err(Error::from);
        }
        let mel_elements = self
            .n_mels
            .checked_mul(range_frames)
            .ok_or_else(|| Error::Overloaded("Sortformer feature shape overflowed".to_string()))?;
        let mut mel = vec![0.0f32; mel_elements];
        let center_pad = self.n_fft / 2;
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.n_fft);
        let mut buffer = vec![Complex::<f32>::new(0.0, 0.0); self.n_fft];
        let mut spectrum = vec![0.0f32; self.n_freqs];

        for (local_frame, frame_idx) in (feature_start..feature_end).enumerate() {
            let padded_start = frame_idx.checked_mul(self.hop_length).ok_or_else(|| {
                Error::Overloaded("Sortformer feature frame offset overflowed".to_string())
            })?;
            for (window_idx, value) in buffer.iter_mut().enumerate() {
                let padded_index = padded_start.checked_add(window_idx).ok_or_else(|| {
                    Error::Overloaded("Sortformer FFT window offset overflowed".to_string())
                })?;
                let sample = if padded_index < center_pad {
                    0.0
                } else {
                    let sample_idx = padded_index - center_pad;
                    if sample_idx >= audio.len() {
                        0.0
                    } else if sample_idx == 0 {
                        audio[0]
                    } else {
                        audio[sample_idx] - PREEMPH * audio[sample_idx - 1]
                    }
                };
                value.re = sample * self.padded_window[window_idx];
                value.im = 0.0;
            }
            fft.process(&mut buffer);
            for (bin, power) in spectrum.iter_mut().enumerate() {
                let magnitude =
                    (buffer[bin].re * buffer[bin].re + buffer[bin].im * buffer[bin].im).sqrt();
                *power = magnitude * magnitude;
            }
            for mel_idx in 0..self.n_mels {
                let fb_row = &self.fb[mel_idx * self.n_freqs..(mel_idx + 1) * self.n_freqs];
                let mut acc = 0.0f32;
                for bin in 0..self.n_freqs {
                    acc += spectrum[bin] * fb_row[bin];
                }
                mel[mel_idx * range_frames + local_frame] = (acc + LOG_GUARD).ln();
            }
        }

        Tensor::from_vec(mel, (1, self.n_mels, range_frames), &Device::Cpu).map_err(Error::from)
    }

    fn compute_features(&self, audio: &[f32]) -> Result<(Tensor, usize)> {
        if audio.is_empty() {
            return Ok((
                Tensor::zeros((1, self.n_mels, 1), DType::F32, &Device::Cpu)?,
                0,
            ));
        }

        let mut x = audio.to_vec();
        preemphasis(&mut x, PREEMPH);

        let center_pad = self.n_fft / 2;
        let mut padded = Vec::with_capacity(x.len() + center_pad * 2);
        padded.extend(std::iter::repeat_n(0.0, center_pad));
        padded.extend_from_slice(&x);
        padded.extend(std::iter::repeat_n(0.0, center_pad));

        let frame_count = if padded.len() >= self.n_fft {
            (padded.len() - self.n_fft) / self.hop_length + 1
        } else {
            1
        };

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.n_fft);

        let mut spectrum = vec![0f32; frame_count * self.n_freqs];
        let mut buffer = vec![Complex::<f32>::new(0.0, 0.0); self.n_fft];
        for frame_idx in 0..frame_count {
            let start = frame_idx * self.hop_length;
            let slice = &padded[start..start + self.n_fft];
            for i in 0..self.n_fft {
                buffer[i].re = slice[i] * self.padded_window[i];
                buffer[i].im = 0.0;
            }
            fft.process(&mut buffer);
            for k in 0..self.n_freqs {
                let mag = (buffer[k].re * buffer[k].re + buffer[k].im * buffer[k].im).sqrt();
                spectrum[frame_idx * self.n_freqs + k] = mag * mag;
            }
        }

        let mut mel = vec![0f32; self.n_mels * frame_count];
        for m in 0..self.n_mels {
            for t in 0..frame_count {
                let mut acc = 0f32;
                let spec_row = &spectrum[t * self.n_freqs..(t + 1) * self.n_freqs];
                let fb_row = &self.fb[m * self.n_freqs..(m + 1) * self.n_freqs];
                for f in 0..self.n_freqs {
                    acc += spec_row[f] * fb_row[f];
                }
                mel[m * frame_count + t] = (acc + LOG_GUARD).ln();
            }
        }

        let valid_frames = audio.len() / self.hop_length;
        let normalized_frames = valid_frames.min(frame_count);
        match self.normalize {
            SortformerFeatureNormalize::None => {}
            SortformerFeatureNormalize::PerFeature => {
                normalize_per_feature(&mut mel, self.n_mels, frame_count, normalized_frames)
            }
            SortformerFeatureNormalize::AllFeatures => {
                normalize_all_features(&mut mel, self.n_mels, frame_count, normalized_frames)
            }
        }

        if valid_frames < frame_count {
            for m in 0..self.n_mels {
                for t in valid_frames..frame_count {
                    mel[m * frame_count + t] = 0.0;
                }
            }
        }

        let features = Tensor::from_vec(mel, (1, self.n_mels, frame_count), &Device::Cpu)?;
        Ok((features, valid_frames.min(frame_count)))
    }
}

/// Per-model engagement counters for the channel-independent depthwise
/// convolutions (SRT-5), reported under `kernels` in `family_diagnostics`.
#[derive(Debug, Default)]
struct SortformerPathCounters {
    conformer_depthwise_conv1d: AtomicU64,
    subsampling_depthwise_conv2d: AtomicU64,
    rel_pos_tables_built: AtomicU64,
}

impl SortformerPathCounters {
    fn conformer_depthwise_conv1d(&self) -> u64 {
        self.conformer_depthwise_conv1d.load(Ordering::Relaxed)
    }

    fn subsampling_depthwise_conv2d(&self) -> u64 {
        self.subsampling_depthwise_conv2d.load(Ordering::Relaxed)
    }

    fn rel_pos_tables_built(&self) -> u64 {
        self.rel_pos_tables_built.load(Ordering::Relaxed)
    }
}

struct SortformerConformerEncoder {
    pre_encode: ConvSubsamplingDw,
    layers: Vec<ConformerLayer>,
    d_model: usize,
    input_scale: f64,
    frame_stride_samples: usize,
    counters: Arc<SortformerPathCounters>,
    /// The relative-position table of the last composite length: the
    /// steady-state chunks all share one length.
    rel_pos_cache: std::sync::Mutex<Option<(usize, Tensor)>>,
}

impl SortformerConformerEncoder {
    fn load(vb: VarBuilder, xscaling: bool) -> Result<Self> {
        let counters = Arc::new(SortformerPathCounters::default());
        let pre_encode = ConvSubsamplingDw::load(vb.pp("pre_encode"), &counters)?;

        let mut layers = Vec::new();
        let mut idx = 0usize;
        loop {
            let layer_vb = vb.pp(format!("layers.{idx}"));
            if !layer_vb.contains_tensor("norm_out.weight") {
                break;
            }
            layers.push(ConformerLayer::load(layer_vb, &counters)?);
            idx += 1;
        }
        if layers.is_empty() {
            return Err(Error::ModelLoadError(
                "Sortformer Conformer encoder has no layers".to_string(),
            ));
        }

        let d_model = layers[0].d_model();
        Ok(Self {
            pre_encode,
            layers,
            d_model,
            input_scale: if xscaling {
                (d_model as f64).sqrt()
            } else {
                1.0
            },
            frame_stride_samples: 160 * 8,
            counters,
            rel_pos_cache: std::sync::Mutex::new(None),
        })
    }

    fn forward(&self, features: &Tensor, feature_frames: usize) -> Result<(Tensor, usize)> {
        let features_t = features.transpose(1, 2)?;
        let (x, encoded_len) = self.pre_encode(&features_t, feature_frames)?;
        self.forward_pre_encoded(&x, encoded_len)
    }

    fn pre_encode(&self, features_t: &Tensor, feature_frames: usize) -> Result<(Tensor, usize)> {
        self.pre_encode.forward(features_t, feature_frames)
    }

    fn forward_pre_encoded(
        &self,
        pre_encoded: &Tensor,
        encoded_len: usize,
    ) -> Result<(Tensor, usize)> {
        let mut x = if self.input_scale != 1.0 {
            pre_encoded.affine(self.input_scale, 0.0)?
        } else {
            pre_encoded.clone()
        };
        let pos_len = x.dim(1)?;
        let pos_emb = self.rel_pos_embedding(pos_len, x.device())?;
        for layer in &self.layers {
            x = layer.forward(&x, &pos_emb)?;
        }
        Ok((x, encoded_len))
    }

    fn rel_pos_embedding(&self, len: usize, device: &Device) -> Result<Tensor> {
        let mut cache = self
            .rel_pos_cache
            .lock()
            .map_err(|_| Error::InferenceError("Sortformer position cache poisoned".into()))?;
        if let Some((cached_len, table)) = cache.as_ref() {
            if *cached_len == len && table.device().same_device(device) {
                return Ok(table.clone());
            }
        }
        let table = build_rel_positional_embedding(len, self.d_model, device)?;
        self.counters
            .rel_pos_tables_built
            .fetch_add(1, Ordering::Relaxed);
        *cache = Some((len, table.clone()));
        Ok(table)
    }

    fn frame_stride_samples(&self) -> usize {
        self.frame_stride_samples
    }

    fn d_model(&self) -> usize {
        self.d_model
    }
}

struct ConvSubsamplingDw {
    conv0: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv5: Conv2d,
    conv6: Conv2d,
    out: Linear,
    out_channels: usize,
    counters: Arc<SortformerPathCounters>,
}

impl ConvSubsamplingDw {
    fn load(vb: VarBuilder, counters: &Arc<SortformerPathCounters>) -> Result<Self> {
        let conv0_w = vb.pp("conv.0").get_unchecked_dtype("weight", DType::F32)?;
        let (out_channels, _, _, _) = conv0_w.dims4()?;

        let stride_cfg = Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let point_cfg = Conv2dConfig {
            stride: 1,
            padding: 0,
            ..Default::default()
        };

        let conv0 = mlx::load_conv2d(1, out_channels, 3, stride_cfg, vb.pp("conv.0"))?;

        let mut dw_stride_cfg = stride_cfg;
        dw_stride_cfg.groups = out_channels;
        let conv2 = mlx::load_conv2d(1, out_channels, 3, dw_stride_cfg, vb.pp("conv.2"))?;
        let conv3 = mlx::load_conv2d(out_channels, out_channels, 1, point_cfg, vb.pp("conv.3"))?;
        let conv5 = mlx::load_conv2d(1, out_channels, 3, dw_stride_cfg, vb.pp("conv.5"))?;
        let conv6 = mlx::load_conv2d(out_channels, out_channels, 1, point_cfg, vb.pp("conv.6"))?;

        let out_w = vb.pp("out").get_unchecked_dtype("weight", DType::F32)?;
        let (out_dim, in_dim) = out_w.dims2()?;
        let out = mlx::load_linear(in_dim, out_dim, vb.pp("out"))?;

        Ok(Self {
            conv0,
            conv2,
            conv3,
            conv5,
            conv6,
            out,
            out_channels,
            counters: Arc::clone(counters),
        })
    }

    /// conv.2 / conv.5: the stride-2 depthwise convs, run by the shared
    /// strided-tap op (about 22 ops) instead of Candle's grouped conv2d, which
    /// runs one convolution per channel (SRT-5).
    fn depthwise(&self, conv: &Conv2d, x: &Tensor) -> Result<Tensor> {
        let cfg = conv.config();
        let spec = DepthwiseConv2dSpec {
            padding: cfg.padding,
            stride: cfg.stride,
        };
        let y = depthwise_conv2d(x, conv.weight(), conv.bias(), spec)?;
        self.counters
            .subsampling_depthwise_conv2d
            .fetch_add(1, Ordering::Relaxed);
        Ok(y)
    }

    fn forward(&self, features_t: &Tensor, feature_frames: usize) -> Result<(Tensor, usize)> {
        let mut x = features_t.unsqueeze(1)?; // [B,1,T,F]

        x = self.conv0.forward(&x)?;
        x = x.relu()?;

        x = self.depthwise(&self.conv2, &x)?;
        x = self.conv3.forward(&x)?;
        x = x.relu()?;

        x = self.depthwise(&self.conv5, &x)?;
        x = self.conv6.forward(&x)?;
        x = x.relu()?;

        let (b, c, t, f) = x.dims4()?;
        let x = x
            .transpose(1, 2)?
            .reshape((b, t, c * f))?
            .apply(&self.out)?;
        let encoded_len = subsampled_len_3x(feature_frames).min(t);
        Ok((x, encoded_len))
    }
}

fn subsampled_len_3x(mut len: usize) -> usize {
    for _ in 0..3 {
        len = len.div_ceil(2);
    }
    len
}

struct ConformerLayer {
    norm_ff1: LayerNorm,
    ff1: FeedForward,
    norm_self_att: LayerNorm,
    self_attn: RelPosSelfAttention,
    norm_conv: LayerNorm,
    conv: ConformerConv,
    norm_ff2: LayerNorm,
    ff2: FeedForward,
    norm_out: LayerNorm,
    d_model: usize,
    ff_dim: usize,
}

impl ConformerLayer {
    fn load(vb: VarBuilder, counters: &Arc<SortformerPathCounters>) -> Result<Self> {
        let d_model = vb
            .pp("norm_out")
            .get_unchecked_dtype("weight", DType::F32)?
            .dim(0)?;

        let ff_dim = vb
            .pp("feed_forward1.linear1")
            .get_unchecked_dtype("weight", DType::F32)?
            .dims2()?
            .0;

        let norm_ff1 = layer_norm(d_model, 1e-5, vb.pp("norm_feed_forward1"))?;
        let ff1 = FeedForward::load(vb.pp("feed_forward1"), d_model, ff_dim)?;

        let norm_self_att = layer_norm(d_model, 1e-5, vb.pp("norm_self_att"))?;
        let self_attn = RelPosSelfAttention::load(vb.pp("self_attn"), d_model)?;

        let norm_conv = layer_norm(d_model, 1e-5, vb.pp("norm_conv"))?;
        let conv = ConformerConv::load(vb.pp("conv"), d_model, counters)?;

        let norm_ff2 = layer_norm(d_model, 1e-5, vb.pp("norm_feed_forward2"))?;
        let ff2 = FeedForward::load(vb.pp("feed_forward2"), d_model, ff_dim)?;

        let norm_out = layer_norm(d_model, 1e-5, vb.pp("norm_out"))?;

        Ok(Self {
            norm_ff1,
            ff1,
            norm_self_att,
            self_attn,
            norm_conv,
            conv,
            norm_ff2,
            ff2,
            norm_out,
            d_model,
            ff_dim,
        })
    }

    fn d_model(&self) -> usize {
        self.d_model
    }

    fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let mut residual = x.clone();

        let ff1 = self.ff1.forward(&self.norm_ff1.forward(&residual)?)?;
        residual = residual.broadcast_add(&ff1.affine(0.5, 0.0)?)?;

        let attn = self
            .self_attn
            .forward(&self.norm_self_att.forward(&residual)?, pos_emb)?;
        residual = residual.broadcast_add(&attn)?;

        let conv = self.conv.forward(&self.norm_conv.forward(&residual)?)?;
        residual = residual.broadcast_add(&conv)?;

        let ff2 = self.ff2.forward(&self.norm_ff2.forward(&residual)?)?;
        residual = residual.broadcast_add(&ff2.affine(0.5, 0.0)?)?;

        self.norm_out
            .forward(&residual)
            .map_err(|e| Error::InferenceError(e.to_string()))
    }
}

struct FeedForward {
    linear1: Linear,
    linear2: Linear,
}

impl FeedForward {
    fn load(vb: VarBuilder, d_model: usize, ff_dim: usize) -> Result<Self> {
        let linear1 = mlx::load_linear(d_model, ff_dim, vb.pp("linear1"))?;
        let linear2 = mlx::load_linear(ff_dim, d_model, vb.pp("linear2"))?;
        Ok(Self { linear1, linear2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = swish(&x)?;
        self.linear2
            .forward(&x)
            .map_err(|e| Error::InferenceError(e.to_string()))
    }
}

struct ConformerConv {
    pointwise_conv1: Conv1d,
    /// The depthwise kernel `(C, 1, K)` and bias with the inference
    /// BatchNorm folded in. The shared shift-and-accumulate op runs 2K + 2
    /// element-wise ops instead of Candle's one convolution per channel
    /// (SRT-5).
    depthwise_weight: Tensor,
    depthwise_bias: Tensor,
    depthwise_spec: DepthwiseConv1dSpec,
    pointwise_conv2: Conv1d,
    d_model: usize,
    counters: Arc<SortformerPathCounters>,
}

impl ConformerConv {
    fn load(
        vb: VarBuilder,
        d_model: usize,
        counters: &Arc<SortformerPathCounters>,
    ) -> Result<Self> {
        let kernel_size = vb
            .pp("depthwise_conv")
            .get_unchecked_dtype("weight", DType::F32)?
            .dims3()?
            .2;

        let pointwise_conv1 = mlx::load_conv1d(
            d_model,
            d_model * 2,
            1,
            Conv1dConfig::default(),
            vb.pp("pointwise_conv1"),
        )?;

        // Loaded as a grouped Conv1d for the layout checks; only its
        // weight and bias are used.
        let depthwise_conv = mlx::load_conv1d(
            d_model,
            d_model,
            kernel_size,
            Conv1dConfig {
                padding: (kernel_size - 1) / 2,
                groups: d_model,
                ..Default::default()
            },
            vb.pp("depthwise_conv"),
        )?;

        let batch_norm = batch_norm(d_model, 1e-5, vb.pp("batch_norm"))?;
        let (scale, shift) = batch_norm_affine(
            batch_norm.running_mean(),
            batch_norm.running_var(),
            batch_norm.eps(),
            batch_norm.weight_and_bias(),
        )?;
        let (depthwise_weight, depthwise_bias) = fold_channel_affine(
            depthwise_conv.weight(),
            depthwise_conv.bias(),
            &scale,
            &shift,
        )?;

        let pointwise_conv2 = mlx::load_conv1d(
            d_model,
            d_model,
            1,
            Conv1dConfig::default(),
            vb.pp("pointwise_conv2"),
        )?;

        Ok(Self {
            pointwise_conv1,
            depthwise_weight,
            depthwise_bias,
            depthwise_spec: DepthwiseConv1dSpec::symmetric((kernel_size - 1) / 2),
            pointwise_conv2,
            d_model,
            counters: Arc::clone(counters),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.transpose(1, 2)?;

        x = self.pointwise_conv1.forward(&x)?;
        let x_a = x.i((.., ..self.d_model, ..))?;
        let x_b = x.i((.., self.d_model.., ..))?;
        x = x_a.broadcast_mul(&ops::sigmoid(&x_b)?)?;

        x = depthwise_conv1d(
            &x,
            &self.depthwise_weight,
            Some(&self.depthwise_bias),
            self.depthwise_spec,
        )?;
        self.counters
            .conformer_depthwise_conv1d
            .fetch_add(1, Ordering::Relaxed);
        x = swish(&x)?;
        x = self.pointwise_conv2.forward(&x)?;

        x.transpose(1, 2).map_err(Error::from)
    }
}

struct RelPosSelfAttention {
    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_out: Linear,
    linear_pos: Linear,
    pos_bias_u: Tensor,
    pos_bias_v: Tensor,
    num_heads: usize,
    head_dim: usize,
    d_model: usize,
}

impl RelPosSelfAttention {
    fn load(vb: VarBuilder, d_model: usize) -> Result<Self> {
        let pos_bias_u = vb.get_unchecked_dtype("pos_bias_u", DType::F32)?;
        let (num_heads, head_dim) = pos_bias_u.dims2()?;
        let pos_bias_v = vb.get((num_heads, head_dim), "pos_bias_v")?;

        if num_heads * head_dim != d_model {
            return Err(Error::ModelLoadError(format!(
                "Sortformer attention head dims mismatch: heads={num_heads}, head_dim={head_dim}, d_model={d_model}"
            )));
        }

        let linear_q = mlx::load_linear(d_model, d_model, vb.pp("linear_q"))?;
        let linear_k = mlx::load_linear(d_model, d_model, vb.pp("linear_k"))?;
        let linear_v = mlx::load_linear(d_model, d_model, vb.pp("linear_v"))?;
        let linear_out = mlx::load_linear(d_model, d_model, vb.pp("linear_out"))?;
        let linear_pos = mlx::load_linear_no_bias(d_model, d_model, vb.pp("linear_pos"))?;

        Ok(Self {
            linear_q,
            linear_k,
            linear_v,
            linear_out,
            linear_pos,
            pos_bias_u,
            pos_bias_v,
            num_heads,
            head_dim,
            d_model,
        })
    }

    fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;

        let q = self
            .linear_q
            .forward(x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .linear_k
            .forward(x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .linear_v
            .forward(x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let p = self
            .linear_pos
            .forward(pos_emb)?
            .reshape((1, 2 * t - 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let pos_bias_u = self
            .pos_bias_u
            .reshape((1, self.num_heads, 1, self.head_dim))?;
        let pos_bias_v = self
            .pos_bias_v
            .reshape((1, self.num_heads, 1, self.head_dim))?;

        let q_u = q.broadcast_add(&pos_bias_u)?.contiguous()?;
        let q_v = q.broadcast_add(&pos_bias_v)?.contiguous()?;

        let k_t = k.transpose(2, 3)?.contiguous()?;
        let p_t = p.transpose(2, 3)?.contiguous()?;
        let matrix_ac = q_u.matmul(&k_t)?;
        let matrix_bd = rel_shift(&q_v.matmul(&p_t)?)?;
        let matrix_bd = matrix_bd.narrow(3, 0, t)?;

        let scores = matrix_ac
            .broadcast_add(&matrix_bd)?
            .affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?;
        let attn = ops::softmax(&scores, 3)?;

        let out = attn.contiguous()?.matmul(&v)?;
        let out = out.transpose(1, 2)?.reshape((b, t, self.d_model))?;

        self.linear_out
            .forward(&out)
            .map_err(|e| Error::InferenceError(e.to_string()))
    }
}

fn rel_shift(x: &Tensor) -> Result<Tensor> {
    let (b, h, qlen, pos_len) = x.dims4()?;
    let x = x.pad_with_zeros(3, 1, 0)?;
    let x = x.reshape((b, h, pos_len + 1, qlen))?;
    let x = x.narrow(2, 1, pos_len)?;
    x.reshape((b, h, qlen, pos_len)).map_err(Error::from)
}

struct SortformerTransformerEncoder {
    layers: Vec<SortformerTransformerLayer>,
}

impl SortformerTransformerEncoder {
    fn load(vb: VarBuilder, num_heads: usize) -> Result<Self> {
        let mut layers = Vec::new();
        let mut idx = 0usize;
        loop {
            let layer_vb = vb.pp(format!("layers.{idx}"));
            if !layer_vb.contains_tensor("layer_norm_1.weight") {
                break;
            }
            layers.push(SortformerTransformerLayer::load(layer_vb, num_heads)?);
            idx += 1;
        }
        if layers.is_empty() {
            return Err(Error::ModelLoadError(
                "Sortformer transformer encoder has no layers".to_string(),
            ));
        }
        Ok(Self { layers })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();
        for layer in &self.layers {
            out = layer.forward(&out)?;
        }
        Ok(out)
    }
}

struct SortformerTransformerLayer {
    norm1: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    out_proj: Linear,
    norm2: LayerNorm,
    dense_in: Linear,
    dense_out: Linear,
    d_model: usize,
    inner_size: usize,
    num_heads: usize,
    head_dim: usize,
}

impl SortformerTransformerLayer {
    fn load(vb: VarBuilder, num_heads: usize) -> Result<Self> {
        let d_model = vb
            .pp("layer_norm_1")
            .get_unchecked_dtype("weight", DType::F32)?
            .dim(0)?;

        let q_w = vb
            .pp("first_sub_layer.query_net")
            .get_unchecked_dtype("weight", DType::F32)?;
        let (_, q_in) = q_w.dims2()?;
        if q_in != d_model {
            return Err(Error::ModelLoadError(format!(
                "Sortformer transformer query input dim mismatch: expected {d_model}, got {q_in}"
            )));
        }

        let dense_in_w = vb
            .pp("second_sub_layer.dense_in")
            .get_unchecked_dtype("weight", DType::F32)?;
        let (inner_size, dense_in) = dense_in_w.dims2()?;
        if dense_in != d_model {
            return Err(Error::ModelLoadError(format!(
                "Sortformer transformer FFN input dim mismatch: expected {d_model}, got {dense_in}"
            )));
        }

        if num_heads == 0 || d_model % num_heads != 0 {
            return Err(Error::ModelLoadError(format!(
                "Sortformer transformer hidden size {d_model} is not divisible by {num_heads} heads"
            )));
        }
        let head_dim = d_model / num_heads;

        Ok(Self {
            norm1: layer_norm(d_model, 1e-5, vb.pp("layer_norm_1"))?,
            q: mlx::load_linear(d_model, d_model, vb.pp("first_sub_layer.query_net"))?,
            k: mlx::load_linear(d_model, d_model, vb.pp("first_sub_layer.key_net"))?,
            v: mlx::load_linear(d_model, d_model, vb.pp("first_sub_layer.value_net"))?,
            out_proj: mlx::load_linear(d_model, d_model, vb.pp("first_sub_layer.out_projection"))?,
            norm2: layer_norm(d_model, 1e-5, vb.pp("layer_norm_2"))?,
            dense_in: mlx::load_linear(d_model, inner_size, vb.pp("second_sub_layer.dense_in"))?,
            dense_out: mlx::load_linear(inner_size, d_model, vb.pp("second_sub_layer.dense_out"))?,
            d_model,
            inner_size,
            num_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let attn = self.self_attention(x)?;
        let h = self.norm1.forward(&x.broadcast_add(&attn)?)?;
        let ff = self
            .dense_out
            .forward(&self.dense_in.forward(&h)?.relu()?)?;
        self.norm2
            .forward(&h.broadcast_add(&ff)?)
            .map_err(Error::from)
    }

    fn self_attention(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;

        let q = self
            .q
            .forward(x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k
            .forward(x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v
            .forward(x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let k_t = k.transpose(2, 3)?.contiguous()?;
        let scores = q
            .matmul(&k_t)?
            .affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?;
        let attn = ops::softmax(&scores, 3)?;
        let ctx = attn.contiguous()?.matmul(&v)?;

        let ctx = ctx.transpose(1, 2)?.reshape((b, t, self.d_model))?;
        self.out_proj.forward(&ctx).map_err(Error::from)
    }
}

struct SortformerSpeakerHead {
    first_hidden_to_hidden: Linear,
    single_hidden_to_spks: Linear,
    hidden_dim: usize,
}

impl SortformerSpeakerHead {
    fn load(vb: VarBuilder, num_speakers: usize) -> Result<Self> {
        let first_w = vb
            .pp("first_hidden_to_hidden")
            .get_unchecked_dtype("weight", DType::F32)?;
        let (first_out, first_in) = first_w.dims2()?;
        if first_out != first_in {
            return Err(Error::ModelLoadError(format!(
                "Unexpected Sortformer hidden projection shape: [{first_out}, {first_in}]"
            )));
        }

        let second_w = vb
            .pp("single_hidden_to_spks")
            .get_unchecked_dtype("weight", DType::F32)?;
        let (spk_out, spk_in) = second_w.dims2()?;
        if spk_out != num_speakers {
            return Err(Error::ModelLoadError(format!(
                "Unexpected Sortformer speaker head output dim {spk_out}; expected {num_speakers}"
            )));
        }
        if spk_in != first_out {
            return Err(Error::ModelLoadError(format!(
                "Sortformer speaker head dim mismatch: hidden={first_out}, input={spk_in}"
            )));
        }

        let first_hidden_to_hidden =
            mlx::load_linear(first_in, first_out, vb.pp("first_hidden_to_hidden"))?;
        let single_hidden_to_spks =
            mlx::load_linear(spk_in, spk_out, vb.pp("single_hidden_to_spks"))?;
        Ok(Self {
            first_hidden_to_hidden,
            single_hidden_to_spks,
            hidden_dim: first_out,
        })
    }

    fn forward(&self, hidden_out: &Tensor) -> Result<Tensor> {
        let hidden_out = hidden_out.relu()?;
        let hidden_out = self.first_hidden_to_hidden.forward(&hidden_out)?;
        let hidden_out = hidden_out.relu()?;
        let spk_logits = self.single_hidden_to_spks.forward(&hidden_out)?;
        ops::sigmoid(&spk_logits).map_err(Error::from)
    }
}

/// Acoustic encoder front-end, discriminated by the checkpoint's encoder
/// section. Both variants reduce 10 ms mel frames onto an 80 ms encoded
/// stream and attend over the composite cache sequence.
enum SortformerAcousticEncoder {
    Conformer(SortformerConformerEncoder),
    FeatureStackingRope(SortformerRopeEncoder),
}

impl SortformerAcousticEncoder {
    fn pre_encode(&self, features_t: &Tensor, feature_frames: usize) -> Result<(Tensor, usize)> {
        match self {
            Self::Conformer(encoder) => encoder.pre_encode(features_t, feature_frames),
            Self::FeatureStackingRope(encoder) => encoder.pre_encode(features_t, feature_frames),
        }
    }

    fn forward_pre_encoded(
        &self,
        composite: &Tensor,
        encoded_len: usize,
    ) -> Result<(Tensor, usize)> {
        match self {
            Self::Conformer(encoder) => encoder.forward_pre_encoded(composite, encoded_len),
            Self::FeatureStackingRope(encoder) => {
                encoder.forward_pre_encoded(composite, encoded_len)
            }
        }
    }

    fn forward(&self, features: &Tensor, feature_frames: usize) -> Result<(Tensor, usize)> {
        match self {
            Self::Conformer(encoder) => encoder.forward(features, feature_frames),
            Self::FeatureStackingRope(encoder) => {
                let features_t = features.transpose(1, 2)?;
                let (embeds, embedded_len) =
                    encoder.pre_encode(&features_t, feature_frames)?;
                encoder.forward_pre_encoded(&embeds, embedded_len)
            }
        }
    }

    fn d_model(&self) -> usize {
        match self {
            Self::Conformer(encoder) => encoder.d_model(),
            Self::FeatureStackingRope(encoder) => encoder.d_model,
        }
    }

    /// Which convolution paths ran, so a deploy shows the fast paths engaged.
    fn kernel_diagnostics(&self) -> serde_json::Value {
        let shared_total = crate::kernels::depthwise_conv::depthwise_conv1d_calls();
        match self {
            Self::Conformer(encoder) => serde_json::json!({
                "conformer_depthwise_conv1d": "shift_accumulate_folded_batch_norm",
                "conformer_depthwise_conv1d_calls": encoder.counters.conformer_depthwise_conv1d(),
                "subsampling_depthwise_conv2d": "strided_taps",
                "subsampling_depthwise_conv2d_calls": encoder.counters.subsampling_depthwise_conv2d(),
                "shared_depthwise_conv1d_calls_total": shared_total,
                "shared_depthwise_conv2d_calls_total":
                    crate::kernels::depthwise_conv::depthwise_conv2d_calls(),
                "rel_pos_tables_built": encoder.counters.rel_pos_tables_built(),
            }),
            // Feature stacking and RoPE attention: no depthwise convolution.
            Self::FeatureStackingRope(_) => serde_json::json!({
                "shared_depthwise_conv1d_calls_total": shared_total,
            }),
        }
    }
}

/// Post-attention frame expander: v2.1's sortformer transformer runs at the
/// encoded rate, while Nemotron-3's learned upsampler expands to 10 ms.
enum SortformerFrameExpander {
    SortformerTransformer(SortformerTransformerEncoder),
    SubpixelUpsampler(SortformerSubpixelUpsampler),
}

impl SortformerFrameExpander {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::SortformerTransformer(transformer) => transformer.forward(x),
            Self::SubpixelUpsampler(upsampler) => upsampler.forward(x),
        }
    }
}

/// NeMo `TransformerEncoder` with `subsampling: feature_stacking` and
/// `self_attention_model: rope` (Nemotron-3-Diarization). `pre_encode`
/// stacks consecutive mel frames and projects them; the cached streaming
/// sequence holds those raw projected embeds, and `forward_pre_encoded`
/// applies the input norm, the RoPE attention layers and the final norm,
/// matching the reference speaker-cache semantics.
struct SortformerRopeEncoder {
    pre_encode_proj: Linear,
    embed_norm: LayerNorm,
    layers: Vec<SortformerRopeLayer>,
    final_norm: LayerNorm,
    d_model: usize,
    num_heads: usize,
    head_dim: usize,
    rope_inv_freq: Vec<f32>,
    stacking_factor: usize,
    feature_bins: usize,
    ff_dim: usize,
}

impl SortformerRopeEncoder {
    /// `num_heads`, `stacking_factor` and `rope_base` are the yaml's
    /// `n_heads`, `subsampling_factor` and `rope_base`.
    fn load(
        vb: VarBuilder,
        feature_bins: usize,
        num_heads: usize,
        stacking_factor: usize,
        rope_base: f32,
    ) -> Result<Self> {
        let proj_vb = vb.pp("pre_encode.proj");
        let proj_w = proj_vb.get_unchecked_dtype("weight", DType::F32)?;
        let (proj_out, proj_in) = proj_w.dims2()?;
        if feature_bins == 0 || proj_in != stacking_factor * feature_bins {
            return Err(Error::ModelLoadError(format!(
                "unsupported Sortformer rope stacking projection [{proj_out}, {proj_in}] for {stacking_factor} stacked frames of {feature_bins} mel bins"
            )));
        }
        let d_model = proj_out;
        if num_heads == 0 || d_model % num_heads != 0 || !(d_model / num_heads).is_multiple_of(2) {
            return Err(Error::ModelLoadError(format!(
                "Sortformer rope hidden size {d_model} does not split into even {num_heads} heads"
            )));
        }
        let head_dim = d_model / num_heads;

        let mut layers = Vec::new();
        let mut idx = 0usize;
        loop {
            let layer_vb = vb.pp(format!("layers.{idx}"));
            if !layer_vb.contains_tensor("norm1.weight") {
                break;
            }
            layers.push(SortformerRopeLayer::load(layer_vb, d_model, num_heads, head_dim)?);
            idx += 1;
        }
        if layers.is_empty() {
            return Err(Error::ModelLoadError(
                "Sortformer rope encoder has no layers".to_string(),
            ));
        }

        let ff_in_w = vb
            .pp("layers.0.ffn.net.0")
            .get_unchecked_dtype("weight", DType::F32)?;
        let (ff_dim, ff_in) = ff_in_w.dims2()?;
        if ff_in != d_model {
            return Err(Error::ModelLoadError(format!(
                "Sortformer rope FFN input dim mismatch: expected {d_model}, got {ff_in}"
            )));
        }

        let rope_inv_freq = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / rope_base.powf(i as f32 / head_dim as f32))
            .collect();

        Ok(Self {
            pre_encode_proj: mlx::load_linear_no_bias(proj_in, proj_out, proj_vb)?,
            embed_norm: layer_norm(d_model, 1e-5, vb.pp("embed_norm"))?,
            layers,
            final_norm: layer_norm(d_model, 1e-5, vb.pp("final_norm"))?,
            d_model,
            num_heads,
            head_dim,
            rope_inv_freq,
            stacking_factor,
            feature_bins,
            ff_dim,
        })
    }

    /// Groups consecutive mel frames into stacked feature vectors. Input is
    /// `[1, time, bins]`; the final incomplete group is zero-padded like the
    /// reference feature stacker, and each output row is
    /// `[frame g*S bins, frame g*S+1 bins, ...]` matching the reference
    /// frame-major stacking order.
    fn stack_features(&self, features_t: &Tensor, feature_frames: usize) -> Result<Tensor> {
        let groups = feature_frames.div_ceil(self.stacking_factor);
        let padded_frames = groups * self.stacking_factor;
        let valid = features_t.i((0, ..feature_frames, ..))?;
        let padded = if padded_frames > feature_frames {
            valid.pad_with_zeros(0, 0, padded_frames - feature_frames)?
        } else {
            valid
        };
        padded
            .reshape((1, groups, self.stacking_factor * self.feature_bins))
            .map_err(Error::from)
    }

    fn pre_encode(&self, features_t: &Tensor, feature_frames: usize) -> Result<(Tensor, usize)> {
        if feature_frames == 0 {
            let empty = Tensor::zeros((1, 0, self.d_model), DType::F32, features_t.device())?;
            return Ok((empty, 0));
        }
        let stacked = self.stack_features(features_t, feature_frames)?;
        let embeds = self.pre_encode_proj.forward(&stacked)?;
        let len = embeds.dim(1)?;
        Ok((embeds, len))
    }

    fn rope_cos_sin(&self, len: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        if len > NEMOTRON3_ROPE_MAX_POSITIONS {
            return Err(Error::InferenceError(format!(
                "Sortformer rope sequence length {len} exceeds the {NEMOTRON3_ROPE_MAX_POSITIONS} position ceiling"
            )));
        }
        let positions = Tensor::from_vec(
            (0..len).map(|position| position as f32).collect::<Vec<_>>(),
            len,
            device,
        )?;
        let inv_freq =
            Tensor::from_vec(self.rope_inv_freq.clone(), self.head_dim / 2, device)?;
        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[&freqs, &freqs], 1)?;
        Ok((emb.cos()?, emb.sin()?))
    }

    fn forward_pre_encoded(
        &self,
        composite: &Tensor,
        encoded_len: usize,
    ) -> Result<(Tensor, usize)> {
        if encoded_len == 0 {
            let empty = Tensor::zeros((1, 0, self.d_model), DType::F32, composite.device())?;
            return Ok((empty, 0));
        }
        let (cos, sin) = self.rope_cos_sin(encoded_len, composite.device())?;
        let mut out = self.embed_norm.forward(&composite.i((.., ..encoded_len, ..))?)?;
        for layer in &self.layers {
            out = layer.forward(&out, &cos, &sin)?;
        }
        let out = self.final_norm.forward(&out)?;
        Ok((out, encoded_len))
    }
}

struct SortformerRopeLayer {
    norm1: LayerNorm,
    norm2: LayerNorm,
    /// Fused `w_qkv` weight pre-transposed to `[d_model, 3 * d_model]`.
    w_qkv_t: Tensor,
    out_proj: Linear,
    ff_in: Linear,
    ff_out: Linear,
    num_heads: usize,
    head_dim: usize,
    d_model: usize,
}

impl SortformerRopeLayer {
    fn load(vb: VarBuilder, d_model: usize, num_heads: usize, head_dim: usize) -> Result<Self> {
        let w_qkv = vb
            .pp("attn.w_qkv")
            .get_unchecked_dtype("weight", DType::F32)?;
        let (fused_out, fused_in) = w_qkv.dims2()?;
        if fused_in != d_model || fused_out != 3 * d_model {
            return Err(Error::ModelLoadError(format!(
                "unexpected Sortformer rope fused QKV shape [{fused_out}, {fused_in}]; expected [{}, {d_model}]",
                3 * d_model
            )));
        }
        let w_qkv_t = w_qkv.transpose(0, 1)?.contiguous()?;

        let ff_in_w = vb
            .pp("ffn.net.0")
            .get_unchecked_dtype("weight", DType::F32)?;
        let (ff_inner, ff_in_dim) = ff_in_w.dims2()?;
        if ff_in_dim != d_model {
            return Err(Error::ModelLoadError(format!(
                "Sortformer rope FFN input dim mismatch: expected {d_model}, got {ff_in_dim}"
            )));
        }
        let ff_out_w = vb
            .pp("ffn.net.3")
            .get_unchecked_dtype("weight", DType::F32)?;
        let (ff_out_dim, ff_out_in) = ff_out_w.dims2()?;
        if ff_out_in != ff_inner || ff_out_dim != d_model {
            return Err(Error::ModelLoadError(format!(
                "Sortformer rope FFN output shape [{ff_out_dim}, {ff_out_in}] does not close [{d_model}, {ff_inner}]"
            )));
        }

        Ok(Self {
            norm1: layer_norm(d_model, 1e-5, vb.pp("norm1"))?,
            norm2: layer_norm(d_model, 1e-5, vb.pp("norm2"))?,
            w_qkv_t,
            out_proj: mlx::load_linear(d_model, d_model, vb.pp("attn.out_proj"))?,
            ff_in: mlx::load_linear(d_model, ff_inner, vb.pp("ffn.net.0"))?,
            ff_out: mlx::load_linear(ff_inner, d_model, vb.pp("ffn.net.3"))?,
            num_heads,
            head_dim,
            d_model,
        })
    }

    /// Pre-norm transformer block: `x + attn(norm1(x))` then
    /// `h + ffn(norm2(h))`, matching `pre_block_norm: true`.
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let attn_out = self.attention(&self.norm1.forward(x)?, cos, sin)?;
        let h = x.broadcast_add(&attn_out)?;
        let ff = self
            .ff_out
            .forward(&self.ff_in.forward(&self.norm2.forward(&h)?)?.gelu_erf()?)?;
        h.broadcast_add(&ff).map_err(Error::from)
    }

    fn attention(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let qkv = x.matmul(&self.w_qkv_t.unsqueeze(0)?)?;
        let split = |offset: usize| -> Result<Tensor> {
            qkv.i((.., .., offset..offset + self.d_model))?
                .reshape((b, t, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
                .map_err(Error::from)
        };
        let q = split(0)?;
        let k = split(self.d_model)?;
        let v = split(2 * self.d_model)?;

        let cos = cos.reshape((1, 1, t, self.head_dim))?;
        let sin = sin.reshape((1, 1, t, self.head_dim))?;
        let q = apply_rope(&q, &cos, &sin)?;
        let k = apply_rope(&k, &cos, &sin)?;

        let scores = q
            .matmul(&k.transpose(2, 3)?.contiguous()?)?
            .affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?;
        let attn = ops::softmax(&scores, 3)?;
        let ctx = attn.contiguous()?.matmul(&v)?;
        let ctx = ctx.transpose(1, 2)?.reshape((b, t, self.d_model))?;
        self.out_proj.forward(&ctx).map_err(Error::from)
    }
}

/// Full rotary embedding on q/k (NeoX half-split convention, positions
/// restarting at zero for every composite cache sequence, matching the
/// reference per-chunk forward).
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_, _, _, head_dim) = x.dims4()?;
    let half = head_dim / 2;
    let x1 = x.narrow(3, 0, half)?.contiguous()?;
    let x2 = x.narrow(3, half, half)?.contiguous()?;
    let rotated = Tensor::cat(&[&x2.neg()?, &x1], 3)?;
    x.broadcast_mul(cos)?
        .broadcast_add(&rotated.broadcast_mul(sin)?)
        .map_err(Error::from)
}

/// Nemotron-3 subpixel upsampler: a Conv1d(head, head * 8, k=3, pad=1)
/// whose channel output is split back across time, expanding the encoded
/// 80 ms sequence to the 10 ms output rate before the classifier head.
struct SortformerSubpixelUpsampler {
    conv: Conv1d,
    d_model: usize,
    upsample_factor: usize,
}

impl SortformerSubpixelUpsampler {
    fn load(vb: VarBuilder) -> Result<Self> {
        let conv_w = vb
            .pp("subpixel_upsample")
            .get_unchecked_dtype("weight", DType::F32)?;
        let (out_channels, in_channels, kernel) = conv_w.dims3()?;
        if kernel != 3
            || in_channels != NEMOTRON3_HEAD_D_MODEL
            || out_channels != NEMOTRON3_HEAD_D_MODEL * TS_VAD_UNIT_FRAME_COUNT
        {
            return Err(Error::ModelLoadError(format!(
                "unexpected Sortformer subpixel upsampler shape [{out_channels}, {in_channels}, {kernel}]"
            )));
        }
        let conv = mlx::load_conv1d(
            in_channels,
            out_channels,
            kernel,
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("subpixel_upsample"),
        )?;
        Ok(Self {
            conv,
            d_model: NEMOTRON3_HEAD_D_MODEL,
            upsample_factor: TS_VAD_UNIT_FRAME_COUNT,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        if d != self.d_model {
            return Err(Error::InferenceError(format!(
                "Sortformer upsampler expected hidden width {d}; got {}",
                self.d_model
            )));
        }
        let conv_out = self.conv.forward(&x.transpose(1, 2)?.contiguous()?)?;
        let (_, channels, _) = conv_out.dims3()?;
        if channels != d * self.upsample_factor {
            return Err(Error::InferenceError(format!(
                "unexpected Sortformer upsampler channel count {channels}; expected {}",
                d * self.upsample_factor
            )));
        }
        conv_out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t * self.upsample_factor, d))
            .map_err(Error::from)
    }
}

fn resolve_postprocessing_params(
    _config: &DiarizationConfig,
    min_duration_on_ms: Option<f32>,
    min_duration_off_ms: Option<f32>,
) -> PostProcessingParams {
    let env = PostProcessingEnv::get();
    let mut params = match env.preset.as_str() {
        "callhome" | "callhome_v2" => PostProcessingParams {
            onset: 0.641,
            offset: 0.561,
            pad_onset: 0.229,
            pad_offset: 0.079,
            min_duration_on: 0.511,
            min_duration_off: 0.296,
            filter_speech_first: true,
        },
        "dihard3" | "dihard3_v2" => PostProcessingParams {
            onset: 0.56,
            offset: 1.0,
            pad_onset: 0.063,
            pad_offset: 0.002,
            min_duration_on: 0.007,
            min_duration_off: 0.151,
            filter_speech_first: true,
        },
        "legacy" | "legacy_model" => PostProcessingParams {
            onset: 0.25,
            offset: 0.25,
            pad_onset: 0.0,
            pad_offset: 0.0,
            min_duration_on: min_duration_on_ms.unwrap_or(0.0).max(0.0) / 1000.0,
            min_duration_off: min_duration_off_ms.unwrap_or(0.0).max(0.0) / 1000.0,
            filter_speech_first: true,
        },
        _ => PostProcessingParams {
            onset: 0.5,
            offset: 0.5,
            pad_onset: 0.0,
            pad_offset: 0.0,
            min_duration_on: min_duration_on_ms.unwrap_or(0.0).max(0.0) / 1000.0,
            min_duration_off: min_duration_off_ms.unwrap_or(0.0).max(0.0) / 1000.0,
            filter_speech_first: true,
        },
    };

    if let Some(value) = env.onset {
        params.onset = value.clamp(0.0, 1.0);
    }
    if let Some(value) = env.offset {
        params.offset = value.clamp(0.0, 1.0);
    }
    if let Some(value) = env.pad_onset {
        params.pad_onset = value.max(0.0);
    }
    if let Some(value) = env.pad_offset {
        params.pad_offset = value.max(0.0);
    }
    if let Some(value) = env.min_duration_on {
        params.min_duration_on = value.max(0.0);
    }
    if let Some(value) = env.min_duration_off {
        params.min_duration_off = value.max(0.0);
    }
    if let Some(value) = env.filter_speech_first {
        params.filter_speech_first = value;
    }

    params
}

fn should_limit_speaker_channels(config: &DiarizationConfig) -> bool {
    config.min_speakers.is_some() || config.max_speakers.is_some()
}

/// The `IZWI_SORTFORMER_PP_*` overrides, read once rather than per request.
#[derive(Debug, Clone)]
struct PostProcessingEnv {
    preset: String,
    onset: Option<f32>,
    offset: Option<f32>,
    pad_onset: Option<f32>,
    pad_offset: Option<f32>,
    min_duration_on: Option<f32>,
    min_duration_off: Option<f32>,
    filter_speech_first: Option<bool>,
}

impl PostProcessingEnv {
    fn get() -> Self {
        static ENV: crate::env::ReadOnce<PostProcessingEnv> =
            crate::env::ReadOnce::new(PostProcessingEnv::read);
        ENV.get()
    }

    fn read() -> Self {
        Self {
            preset: std::env::var("IZWI_SORTFORMER_PP_PRESET")
                .unwrap_or_else(|_| "model".to_string())
                .to_ascii_lowercase(),
            onset: env_postprocessing_value("IZWI_SORTFORMER_PP_ONSET"),
            offset: env_postprocessing_value("IZWI_SORTFORMER_PP_OFFSET"),
            pad_onset: env_postprocessing_value("IZWI_SORTFORMER_PP_PAD_ONSET"),
            pad_offset: env_postprocessing_value("IZWI_SORTFORMER_PP_PAD_OFFSET"),
            min_duration_on: env_postprocessing_value("IZWI_SORTFORMER_PP_MIN_DURATION_ON"),
            min_duration_off: env_postprocessing_value("IZWI_SORTFORMER_PP_MIN_DURATION_OFF"),
            filter_speech_first: crate::env::flag("IZWI_SORTFORMER_PP_FILTER_SPEECH_FIRST"),
        }
    }
}

fn env_postprocessing_value(key: &str) -> Option<f32> {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<f32>().ok())
        .filter(|value| value.is_finite())
}

fn select_speaker_channels(
    stats: &[SpeakerActivityStats],
    min_speakers: usize,
    max_speakers: usize,
    channel_count: usize,
) -> Vec<usize> {
    let keep = max_speakers.clamp(min_speakers, channel_count);
    let mut ranked = stats.to_vec();
    ranked.sort_by(|a, b| {
        b.total_duration_secs
            .total_cmp(&a.total_duration_secs)
            .then(b.segment_count.cmp(&a.segment_count))
            .then(b.peak_probability.total_cmp(&a.peak_probability))
            .then(a.speaker_idx.cmp(&b.speaker_idx))
    });

    let active = ranked
        .iter()
        .filter(|stat| stat.segment_count > 0 && stat.total_duration_secs > 0.0)
        .map(|stat| stat.speaker_idx)
        .take(keep)
        .collect::<Vec<_>>();

    if active.len() >= min_speakers {
        return active;
    }

    ranked
        .into_iter()
        .take(keep)
        .map(|stat| stat.speaker_idx)
        .collect()
}

/// SRT-12: limiting the speaker count keeps the speech of the channels it
/// drops. Each of their segments goes to the selected channel with the
/// highest mean probability over it (ties to the lower channel), whose
/// probability becomes its confidence; overlapping segments of one speaker
/// merge afterwards.
fn reassign_dropped_speaker_segments(
    segments: &mut [RawSegment],
    selected: &[usize],
    probs: &[Vec<f32>],
    frame_stride_samples: usize,
) {
    for segment in segments
        .iter_mut()
        .filter(|segment| !selected.contains(&segment.speaker_idx))
    {
        let best = selected
            .iter()
            .map(|&speaker| {
                let probability = average_speaker_probability_for_range(
                    probs,
                    speaker,
                    segment.start_secs,
                    segment.end_secs,
                    frame_stride_samples,
                );
                (speaker, probability)
            })
            .max_by(|a, b| {
                a.1.unwrap_or(0.0)
                    .total_cmp(&b.1.unwrap_or(0.0))
                    .then(b.0.cmp(&a.0))
            });
        if let Some((speaker, probability)) = best {
            segment.speaker_idx = speaker;
            segment.confidence = probability;
        }
    }
}

/// NeMo's current post-processing of one speaker channel
/// (`predlist_to_timestamps`): binarize the probabilities at their own frame
/// period (`frame_repeat` x 10 ms: 80 ms for v2.1, 10 ms for Nemotron-3) with
/// `binarization_vectorized`, then apply `filtering`.
fn predlist_to_timestamps(
    probs: &[Vec<f32>],
    speaker_idx: usize,
    params: &PostProcessingParams,
    frame_repeat: usize,
) -> Vec<(f32, f32)> {
    let frame_secs = TS_VAD_FRAME_LENGTH_SECS * frame_repeat.max(1) as f32;
    let sequence = probs
        .iter()
        .map(|row| row[speaker_idx].clamp(0.0, 1.0))
        .collect::<Vec<_>>();
    filtering(&binarization(&sequence, frame_secs, params), params)
}

/// NeMo's `binarization_vectorized` (SRT-11), a hysteresis over frames of
/// `frame_secs`. With `onset >= offset` a frame above `onset` turns speech on,
/// one below `offset` turns it off, and frames between keep the state. With
/// `onset < offset` (the `dihard3` preset) a frame at or above `offset` sets
/// speech, one at or below `onset` clears it, and each frame strictly between
/// toggles it. A segment spans its first speech frame to the first frame
/// after it without speech, or to the end of the sequence (N frames, where
/// the legacy loop stopped at frame N - 1), widened by the pads; the start is
/// clamped at 0. Padded segments that touch or overlap merge.
fn binarization(
    sequence: &[f32],
    frame_secs: f32,
    params: &PostProcessingParams,
) -> Vec<(f32, f32)> {
    let mut segments = Vec::new();
    let mut push = |start: usize, end: usize| {
        let start_secs = (start as f32 * frame_secs - params.pad_onset).max(0.0);
        let end_secs = end as f32 * frame_secs + params.pad_offset;
        if end_secs > start_secs {
            segments.push((start_secs, end_secs));
        }
    };
    let mut speech_start = None::<usize>;
    for (idx, &value) in sequence.iter().enumerate() {
        let speech = speech_start.is_some();
        let next = if params.onset >= params.offset {
            if value > params.onset {
                true
            } else if value < params.offset {
                false
            } else {
                speech
            }
        } else if value >= params.offset {
            true
        } else if value <= params.onset {
            false
        } else if value > params.onset && value < params.offset {
            !speech
        } else {
            speech
        };
        match (speech_start, next) {
            (None, true) => speech_start = Some(idx),
            (Some(start), false) => {
                push(start, idx);
                speech_start = None;
            }
            _ => {}
        }
    }
    if let Some(start) = speech_start {
        push(start, sequence.len());
    }
    if params.pad_onset > 0.0 || params.pad_offset > 0.0 {
        merge_overlap_ranges(&segments)
    } else {
        segments
    }
}

fn filtering(segments: &[(f32, f32)], params: &PostProcessingParams) -> Vec<(f32, f32)> {
    if segments.is_empty() {
        return Vec::new();
    }

    let mut speech_segments = segments.to_vec();
    if params.filter_speech_first {
        if params.min_duration_on > 0.0 {
            speech_segments = filter_short_segments(&speech_segments, params.min_duration_on);
        }
        if params.min_duration_off > 0.0 && speech_segments.len() > 1 {
            let non_speech_segments = get_gap_segments(&speech_segments);
            let short_non_speech_segments = remove_ranges(
                &non_speech_segments,
                &filter_short_segments(&non_speech_segments, params.min_duration_off),
            );
            if !short_non_speech_segments.is_empty() {
                speech_segments.extend(short_non_speech_segments);
                speech_segments = merge_overlap_ranges(&speech_segments);
            }
        }
    } else {
        if params.min_duration_off > 0.0 && speech_segments.len() > 1 {
            let non_speech_segments = get_gap_segments(&speech_segments);
            let short_non_speech_segments = remove_ranges(
                &non_speech_segments,
                &filter_short_segments(&non_speech_segments, params.min_duration_off),
            );
            if !short_non_speech_segments.is_empty() {
                speech_segments.extend(short_non_speech_segments);
                speech_segments = merge_overlap_ranges(&speech_segments);
            }
        }
        if params.min_duration_on > 0.0 {
            speech_segments = filter_short_segments(&speech_segments, params.min_duration_on);
        }
    }
    speech_segments
}

fn remove_ranges(
    original_segments: &[(f32, f32)],
    to_be_removed_segments: &[(f32, f32)],
) -> Vec<(f32, f32)> {
    if original_segments.is_empty() || to_be_removed_segments.is_empty() {
        return original_segments.to_vec();
    }

    original_segments
        .iter()
        .copied()
        .filter(|segment| {
            !to_be_removed_segments.iter().any(|removed| {
                (segment.0 - removed.0).abs() <= f32::EPSILON
                    && (segment.1 - removed.1).abs() <= f32::EPSILON
            })
        })
        .collect()
}

fn filter_short_segments(segments: &[(f32, f32)], threshold: f32) -> Vec<(f32, f32)> {
    segments
        .iter()
        .copied()
        .filter(|(start, end)| (end - start) >= threshold)
        .collect()
}

fn get_gap_segments(segments: &[(f32, f32)]) -> Vec<(f32, f32)> {
    if segments.len() <= 1 {
        return Vec::new();
    }

    let sorted = sort_ranges(segments);
    sorted
        .windows(2)
        .filter_map(|window| {
            let (_, left_end) = window[0];
            let (right_start, _) = window[1];
            (right_start > left_end).then_some((left_end, right_start))
        })
        .collect()
}

fn merge_overlap_ranges(segments: &[(f32, f32)]) -> Vec<(f32, f32)> {
    if segments.len() <= 1 {
        return segments.to_vec();
    }

    let mut sorted = sort_ranges(segments);
    let mut merged = Vec::with_capacity(sorted.len());
    let mut current = sorted.remove(0);
    for segment in sorted {
        if current.1 >= segment.0 {
            current.1 = current.1.max(segment.1);
        } else {
            merged.push(current);
            current = segment;
        }
    }
    merged.push(current);
    merged
}

fn sort_ranges(segments: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let mut sorted = segments.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    sorted
}

fn average_speaker_probability_for_range(
    probs: &[Vec<f32>],
    speaker_idx: usize,
    start_secs: f32,
    end_secs: f32,
    frame_stride_samples: usize,
) -> Option<f32> {
    if probs.is_empty() || end_secs <= start_secs || frame_stride_samples == 0 {
        return None;
    }

    let frame_stride_secs = frame_stride_samples as f32 / TARGET_SAMPLE_RATE as f32;
    let start_frame = (start_secs / frame_stride_secs).floor().max(0.0) as usize;
    let end_frame = ((end_secs / frame_stride_secs).ceil().max(0.0) as usize).min(probs.len());
    if start_frame >= end_frame {
        return None;
    }

    let mut sum = 0.0f32;
    let mut count = 0usize;
    for row in probs.iter().take(end_frame).skip(start_frame) {
        sum += row[speaker_idx];
        count += 1;
    }

    (count > 0).then_some((sum / count as f32).clamp(0.0, 1.0))
}

fn sortformer_vad_frame_mask(
    samples: &[f32],
    frame_count: usize,
    frame_stride_samples: usize,
    min_speech_ms: f32,
    min_silence_ms: f32,
) -> Vec<bool> {
    if frame_count == 0 || frame_stride_samples == 0 {
        return Vec::new();
    }

    let vad_config = VadRegionConfig {
        min_speech_ms: ms_f32_to_u32(min_speech_ms),
        min_silence_ms: ms_f32_to_u32(min_silence_ms),
        speech_pad_ms: 0,
        merge_gap_ms: ms_f32_to_u32(min_silence_ms),
        ..VadRegionConfig::default()
    };
    speech_mask_for_frames_f32(
        samples,
        TARGET_SAMPLE_RATE,
        frame_count,
        frame_stride_samples,
        &vad_config,
    )
    .unwrap_or_else(|_| vec![true; frame_count])
}

/// The opt-in VAD gate: the request's `vad_gate`, else
/// `IZWI_SORTFORMER_VAD_GATE` (read once), else off.
fn resolve_vad_gate(config: &DiarizationConfig) -> bool {
    static ENV: crate::env::ReadOnce<Option<bool>> =
        crate::env::ReadOnce::new(|| crate::env::flag("IZWI_SORTFORMER_VAD_GATE"));
    config.vad_gate.or_else(|| ENV.get()).unwrap_or(false)
}

/// Speech regions of a frame mask in seconds, each widened by one encoded
/// Sortformer frame (80 ms) on both sides so the gate never clips an onset
/// or offset the model placed inside that frame.
fn padded_vad_regions(mask: &[bool], frame_stride_samples: usize) -> Vec<(f32, f32)> {
    let frame_secs = frame_stride_samples as f32 / TARGET_SAMPLE_RATE as f32;
    let encoded_frame_secs =
        (TS_VAD_UNIT_FRAME_COUNT * PRODUCTION_HOP_LENGTH) as f32 / TARGET_SAMPLE_RATE as f32;
    let pad_secs = encoded_frame_secs.max(frame_secs);
    let regions = collect_active_regions(mask)
        .into_iter()
        .map(|(start, end)| {
            (
                (start as f32 * frame_secs - pad_secs).max(0.0),
                (end + 1) as f32 * frame_secs + pad_secs,
            )
        })
        .collect::<Vec<_>>();
    merge_overlap_ranges(&regions)
}

/// Gates whole post-processed segments: one that meets no speech region is
/// dropped, and one that does is trimmed to the first and last regions it
/// meets. Pauses the VAD finds inside a turn never split it.
fn gate_segments(segments: &[(f32, f32)], regions: &[(f32, f32)]) -> Vec<(f32, f32)> {
    segments
        .iter()
        .filter_map(|&(start, end)| {
            let mut met = regions
                .iter()
                .filter(|&&(region_start, region_end)| region_start < end && region_end > start);
            let first = met.next()?;
            let last = met.next_back().unwrap_or(first);
            Some((start.max(first.0), end.min(last.1)))
        })
        .collect()
}

fn ms_f32_to_u32(ms: f32) -> u32 {
    ms.round().max(1.0) as u32
}

fn collect_active_regions(active: &[bool]) -> Vec<(usize, usize)> {
    let mut regions = Vec::new();
    let mut idx = 0usize;
    while idx < active.len() {
        if !active[idx] {
            idx += 1;
            continue;
        }
        let start = idx;
        while idx < active.len() && active[idx] {
            idx += 1;
        }
        let end = idx.saturating_sub(1);
        regions.push((start, end));
    }
    regions
}

fn merge_adjacent_segments(segments: &mut Vec<DiarizationSegment>, merge_gap_secs: f32) {
    if segments.len() <= 1 {
        return;
    }

    let mut by_speaker: BTreeMap<String, Vec<DiarizationSegment>> = BTreeMap::new();
    for segment in segments.drain(..) {
        by_speaker
            .entry(segment.speaker.clone())
            .or_default()
            .push(segment);
    }

    let mut merged_all = Vec::new();
    for (_, mut speaker_segments) in by_speaker {
        speaker_segments.sort_by(|a, b| a.start_secs.total_cmp(&b.start_secs));
        let mut iter = speaker_segments.into_iter();
        let Some(mut current) = iter.next() else {
            continue;
        };

        for segment in iter {
            let gap = (segment.start_secs - current.end_secs).max(0.0);
            if gap <= merge_gap_secs {
                current.end_secs = current.end_secs.max(segment.end_secs);
                current.confidence = match (current.confidence, segment.confidence) {
                    (Some(a), Some(b)) => Some((a + b) / 2.0),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    (None, None) => None,
                };
            } else {
                merged_all.push(current);
                current = segment;
            }
        }
        merged_all.push(current);
    }

    merged_all.sort_by(|a, b| {
        a.start_secs
            .total_cmp(&b.start_secs)
            .then(a.speaker.cmp(&b.speaker))
    });
    *segments = merged_all;
}

fn build_rel_positional_embedding(len: usize, d_model: usize, device: &Device) -> Result<Tensor> {
    if len == 0 {
        return Err(Error::InvalidInput(
            "Cannot build positional embedding for empty sequence".to_string(),
        ));
    }

    let pos_len = 2 * len - 1;
    let mut positions = Vec::with_capacity(pos_len);
    for p in (-(len as isize - 1))..=(len as isize - 1) {
        positions.push((-p) as f32);
    }

    let mut emb = vec![0f32; pos_len * d_model];
    let denom = (10_000f32).ln() / d_model as f32;

    for (pi, p) in positions.iter().enumerate() {
        for i in (0..d_model).step_by(2) {
            let div = (-denom * i as f32).exp();
            let angle = p * div;
            emb[pi * d_model + i] = angle.sin();
            if i + 1 < d_model {
                emb[pi * d_model + i + 1] = angle.cos();
            }
        }
    }

    Tensor::from_vec(emb, (1, pos_len, d_model), device).map_err(Error::from)
}

fn swish(x: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&ops::sigmoid(x)?).map_err(Error::from)
}

fn hann_window(win_length: usize) -> Vec<f32> {
    if win_length <= 1 {
        return vec![1.0; win_length.max(1)];
    }

    (0..win_length)
        .map(|i| {
            let x = (2.0 * std::f32::consts::PI * i as f32) / (win_length as f32 - 1.0);
            0.5 - 0.5 * x.cos()
        })
        .collect()
}

fn hz_to_mel_slaney(hz: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f32).ln() / 27.0;

    if hz < min_log_hz {
        hz / f_sp
    } else {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    }
}

fn mel_to_hz_slaney(mel: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f32).ln() / 27.0;

    if mel < min_log_mel {
        mel * f_sp
    } else {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    }
}

fn mel_filterbank(
    sample_rate: usize,
    n_fft: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let nyquist = sample_rate as f32 / 2.0;
    let mel_min = hz_to_mel_slaney(fmin.max(0.0));
    let mel_max = hz_to_mel_slaney(fmax.min(nyquist).max(fmin));

    let mel_points: Vec<f32> = (0..(n_mels + 2))
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.into_iter().map(mel_to_hz_slaney).collect();
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| nyquist * i as f32 / (n_freqs.saturating_sub(1).max(1)) as f32)
        .collect();

    let mut fb = vec![0f32; n_mels * n_freqs];
    for m in 0..n_mels {
        let left = hz_points[m];
        let center = hz_points[m + 1];
        let right = hz_points[m + 2];
        let lower_width = (center - left).max(1e-12);
        let upper_width = (right - center).max(1e-12);
        let enorm = if right > left {
            2.0 / (right - left)
        } else {
            0.0
        };

        for (k, &freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - left) / lower_width;
            let upper = (right - freq) / upper_width;
            fb[m * n_freqs + k] = lower.min(upper).max(0.0) * enorm;
        }
    }

    fb
}

fn preemphasis(x: &mut [f32], preemph: f32) {
    if x.len() < 2 {
        return;
    }

    let mut prev = x[0];
    for sample in x.iter_mut().skip(1) {
        let cur = *sample;
        *sample = cur - preemph * prev;
        prev = cur;
    }
}

fn normalize_per_feature(mel: &mut [f32], n_mels: usize, frames: usize, valid_frames: usize) {
    if valid_frames == 0 {
        return;
    }

    for m in 0..n_mels {
        let row = &mut mel[m * frames..(m + 1) * frames];

        let mean = row[..valid_frames].iter().copied().sum::<f32>() / valid_frames as f32;

        let var = if valid_frames > 1 {
            row[..valid_frames]
                .iter()
                .map(|v| {
                    let d = *v - mean;
                    d * d
                })
                .sum::<f32>()
                / (valid_frames as f32 - 1.0)
        } else {
            0.0
        };

        let std = var.sqrt() + NORMALIZE_EPS;
        for v in row[..valid_frames].iter_mut() {
            *v = (*v - mean) / std;
        }
    }
}

fn normalize_all_features(mel: &mut [f32], n_mels: usize, frames: usize, valid_frames: usize) {
    if valid_frames == 0 {
        return;
    }

    let total = n_mels * valid_frames;
    if total == 0 {
        return;
    }

    let mut sum = 0.0f32;
    for m in 0..n_mels {
        let row = &mel[m * frames..(m + 1) * frames];
        sum += row[..valid_frames].iter().copied().sum::<f32>();
    }
    let mean = sum / total as f32;

    let var = if total > 1 {
        let mut accum = 0.0f32;
        for m in 0..n_mels {
            let row = &mel[m * frames..(m + 1) * frames];
            for value in &row[..valid_frames] {
                let delta = *value - mean;
                accum += delta * delta;
            }
        }
        accum / (total as f32 - 1.0)
    } else {
        0.0
    };

    let std = var.sqrt() + NORMALIZE_EPS;
    for m in 0..n_mels {
        let row = &mut mel[m * frames..(m + 1) * frames];
        for value in &mut row[..valid_frames] {
            *value = (*value - mean) / std;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::BackendKind;
    use crate::runtime::audio_io::decode_audio_bytes;
    use std::path::PathBuf;
    use std::sync::Mutex;

    /// The crate-wide environment lock, so the profile tests here order
    /// against every other test that touches the process environment.
    fn env_lock() -> &'static Mutex<()> {
        crate::env_test_lock()
    }

    fn streaming_cfg_for_test() -> SortformerStreamingConfig {
        SortformerStreamingConfig {
            fc_d_model: 2,
            num_speakers: 4,
            subsampling_factor: 8,
            output_frames_per_encoded_frame: 1,
            spkcache_len: 4,
            fifo_len: 2,
            chunk_len: 2,
            spkcache_update_period: 2,
            chunk_left_context: 1,
            chunk_right_context: 1,
            spkcache_sil_frames_per_spk: 0,
            pred_score_threshold: 0.25,
            scores_boost_latest: 0.0,
            sil_threshold: 0.2,
            strong_boost_rate: 0.75,
            weak_boost_rate: 1.5,
            min_pos_scores_rate: 0.5,
        }
    }

    /// Builds a live invocation arena for the Sortformer streaming state on
    /// the requested backend and hands out one lease — the exact objects the
    /// served diarization path commits through.
    fn streaming_state_commit_lease(
        cfg: SortformerStreamingConfig,
        backend: BackendKind,
        device: Device,
    ) -> InvocationTensorLease {
        use crate::backends::state::{negotiate_state_plan, StateBackendPlanRequest};
        use crate::engine::{InvocationTensorPoolOwner, ModelInstanceId};
        use crate::kv::v2::{
            InvocationStateCapacity, InvocationWorkspaceDomain, PlacementPolicy, WorkspaceFormula,
        };

        // Mirror the arena's own device-identity derivation so the negotiated
        // plan matches the resident device on every backend (Metal's ordinal
        // is the registry-id fold, not 0).
        let device_ordinal = match device.location() {
            candle_core::DeviceLocation::Cpu => None,
            candle_core::DeviceLocation::Cuda { gpu_id } => Some(u32::try_from(gpu_id).unwrap()),
            candle_core::DeviceLocation::Metal { gpu_id } => {
                let id = gpu_id as u64;
                Some((id ^ (id >> 32)) as u32)
            }
        };

        let contract = physical::sortformer_invocation_contract(cfg).unwrap();
        let plan = std::sync::Arc::new(
            negotiate_state_plan(
                &contract,
                &StateBackendPlanRequest {
                    backend,
                    device_ordinal,
                    page_tokens_hint: None,
                    storage_dtype_hint: None,
                },
            )
            .unwrap(),
        );
        let state_domain = contract
            .domains
            .iter()
            .find(|domain| domain.id() == physical::SORTFORMER_STREAMING_STATE_DOMAIN)
            .unwrap()
            .clone();
        let fixed_bytes = plan
            .non_paged
            .iter()
            .find(|resolved| resolved.domain() == physical::SORTFORMER_STREAMING_STATE_DOMAIN)
            .unwrap()
            .maximum_bytes();
        let workspace_domain = InvocationWorkspaceDomain::State {
            state: state_domain,
            capacity: InvocationStateCapacity::SemanticBounded,
            placement: PlacementPolicy::BackendLocal,
            formula: WorkspaceFormula {
                fixed_bytes,
                dimensions: vec![],
                terms: vec![],
            },
        };
        let owner = InvocationTensorPoolOwner::new(
            &contract,
            plan,
            workspace_domain,
            device,
            ModelInstanceId::new(7),
            1,
            1,
        )
        .unwrap();
        owner.lease().unwrap()
    }

    /// Host rows as device-resident streaming embeddings.
    fn emb_rows(rows: &[Vec<f32>]) -> EmbeddingRows {
        let width = rows.first().map_or(0, Vec::len);
        let flat = rows.iter().flatten().copied().collect::<Vec<_>>();
        EmbeddingRows::new(Tensor::from_vec(flat, (rows.len(), width), &Device::Cpu).unwrap())
            .unwrap()
    }

    fn host_rows(rows: &EmbeddingRows) -> Vec<Vec<f32>> {
        rows.tensor()
            .map_or_else(Vec::new, |tensor| tensor.to_vec2::<f32>().unwrap())
    }

    fn synthetic_streaming_state(cfg: &SortformerStreamingConfig) -> SortformerStreamingState {
        SortformerStreamingState {
            spkcache: emb_rows(&vec![vec![0.5; cfg.fc_d_model]; 4.min(cfg.spkcache_len)]),
            spkcache_preds: Some(vec![vec![0.25; cfg.num_speakers]; 4.min(cfg.spkcache_len)]),
            fifo: EmbeddingRows::default(),
            fifo_preds: Vec::new(),
            mean_sil_emb: vec![0.1; cfg.fc_d_model],
            n_sil_frames: 12,
            host_reads: 0,
        }
    }

    #[test]
    fn sortformer_streaming_commit_stages_state_on_the_arena_device() {
        let cases = [
            ("v2.1", streaming_cfg_for_test()),
            (
                "nemotron",
                resolve_streaming_config(
                    SortformerStreamingProfile::Model,
                    &SortformerModulesConfig {
                        // The checkpoint publishes an 8-channel-divisible
                        // cache; the 188 default is 4-channel only.
                        spkcache_len: Some(192),
                        ..SortformerModulesConfig::default()
                    },
                    512,
                    8,
                    SortformerEncoderKind::FeatureStackingRope,
                )
                .unwrap(),
            ),
            ("workspace", production_workspace_streaming_config()),
        ];
        for (label, cfg) in cases {
            let mut lease = streaming_state_commit_lease(cfg, BackendKind::Cpu, Device::Cpu);
            let state = synthetic_streaming_state(&cfg);
            commit_sortformer_streaming_state(&mut lease, cfg, &state)
                .unwrap_or_else(|err| panic!("{label} first commit failed: {err}"));
            commit_sortformer_streaming_state(&mut lease, cfg, &state)
                .unwrap_or_else(|err| panic!("{label} second commit failed: {err}"));
            let arena = lease.arena().unwrap();
            assert_eq!(arena.absolute_cursor(), 2, "{label}");
            assert_eq!(arena.device().location(), Device::Cpu.location(), "{label}");
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn sortformer_streaming_commit_stages_state_on_the_metal_arena_device() {
        let _env = env_lock().lock().unwrap();
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "sortformer_streaming_commit_stages_state_on_the_metal_arena_device",
        ) else {
            // No Metal device on this host: the CPU test above still covers
            // the device-routing contract.
            return;
        };
        let cfg = resolve_streaming_config(
            SortformerStreamingProfile::Model,
            &SortformerModulesConfig {
                spkcache_len: Some(192),
                ..SortformerModulesConfig::default()
            },
            512,
            8,
            SortformerEncoderKind::FeatureStackingRope,
        )
        .unwrap();
        let mut lease = streaming_state_commit_lease(cfg, BackendKind::Metal, device.clone());
        let state = synthetic_streaming_state(&cfg);
        // Before the arena-device fix this commit failed with "invocation
        // tensor input has incompatible device, dtype, or rank": the state
        // tensors were staged on the model's CPU device while the arena
        // backing was Metal-resident.
        commit_sortformer_streaming_state(&mut lease, cfg, &state).unwrap();
        let arena = lease.arena().unwrap();
        assert_eq!(arena.device().location(), device.location());
        assert_eq!(arena.absolute_cursor(), 1);
    }

    #[cfg(feature = "cuda")]
    #[ignore = "requires CUDA hardware"]
    #[test]
    fn sortformer_streaming_commit_stages_state_on_the_cuda_arena_device() {
        let device = Device::new_cuda(0).unwrap();
        let cfg = resolve_streaming_config(
            SortformerStreamingProfile::Model,
            &SortformerModulesConfig {
                spkcache_len: Some(192),
                ..SortformerModulesConfig::default()
            },
            512,
            8,
            SortformerEncoderKind::FeatureStackingRope,
        )
        .unwrap();
        let mut lease = streaming_state_commit_lease(cfg, BackendKind::Cuda, device.clone());
        let state = synthetic_streaming_state(&cfg);
        commit_sortformer_streaming_state(&mut lease, cfg, &state).unwrap();
        let arena = lease.arena().unwrap();
        assert_eq!(arena.device().location(), device.location());
        assert_eq!(arena.absolute_cursor(), 1);
    }

    #[test]
    fn sortformer_only_uses_selected_model_device_for_cuda() {
        assert!(!sortformer_uses_selected_model_device(BackendKind::Cpu));
        assert!(!sortformer_uses_selected_model_device(BackendKind::Metal));
        assert!(sortformer_uses_selected_model_device(BackendKind::Cuda));
    }

    #[test]
    fn streaming_feature_ranges_match_full_preprocessing_slices() {
        let n_fft = 8;
        let n_mels = 3;
        let n_freqs = n_fft / 2 + 1;
        let padded_window = hann_window(n_fft);
        let preprocessor = SortformerPreprocessor {
            sample_rate: TARGET_SAMPLE_RATE as usize,
            n_fft,
            win_length: n_fft,
            hop_length: 4,
            _window: padded_window.clone(),
            padded_window,
            fb: (0..n_mels * n_freqs)
                .map(|idx| (idx + 1) as f32 / 17.0)
                .collect(),
            n_mels,
            n_freqs,
            normalize: SortformerFeatureNormalize::None,
        };
        let audio = (0..41)
            .map(|idx| ((idx as f32) * 0.31).sin() * 0.7)
            .collect::<Vec<_>>();
        let (full, valid_frames) = preprocessor.compute_features(&audio).unwrap();

        for (start, end) in [(0, 5), (3, valid_frames), (valid_frames - 1, valid_frames)] {
            let expected = full
                .i((0, .., start..end))
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            let chunk = preprocessor
                .compute_feature_range(&audio, start, end)
                .unwrap()
                .i(0)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            assert_eq!(chunk, expected, "feature range {start}..{end}");
        }
    }

    #[test]
    fn production_workspace_is_chunk_bounded_and_topology_checked() {
        let topology = SortformerWorkspaceTopology::v21_production();
        let cfg = production_workspace_streaming_config();
        topology.validate_production(cfg).unwrap();

        let chunk_sized_samples = (PRODUCTION_MAX_CHUNK_LEN
            + PRODUCTION_MAX_CHUNK_LEFT_CONTEXT
            + PRODUCTION_MAX_CHUNK_RIGHT_CONTEXT)
            * TS_VAD_UNIT_FRAME_COUNT
            * PRODUCTION_HOP_LENGTH;
        let chunk = workspace_estimate_for(topology, cfg, chunk_sized_samples, false).unwrap();
        let hour = workspace_estimate_for(topology, cfg, 60 * 60 * 16_000, false).unwrap();
        assert_eq!(hour.accelerator_bytes, chunk.accelerator_bytes);
        assert!(hour.host_bytes > chunk.host_bytes);
        assert!(hour.accelerator_bytes < 3 * 1024 * 1024 * 1024);

        let nemotron3 = SortformerWorkspaceTopology::nemotron3_production();
        let nemotron3_cfg = nemotron3_workspace_streaming_config();
        nemotron3.validate_production(nemotron3_cfg).unwrap();
        let nemotron3_chunk =
            workspace_estimate_for(nemotron3, nemotron3_cfg, chunk_sized_samples, false).unwrap();
        assert!(nemotron3_chunk.accelerator_bytes > 0);
        assert!(nemotron3_chunk.host_bytes > 0);

        let mut unsupported = topology;
        if let SortformerEncoderTopology::Conformer { conv_channels, .. } =
            &mut unsupported.encoder
        {
            *conv_channels += 1;
        }
        assert!(matches!(
            workspace_estimate_for(unsupported, cfg, chunk_sized_samples, false),
            Err(Error::ModelLoadError(_))
        ));

        let mut oversized_profile = cfg;
        oversized_profile.chunk_len = PRODUCTION_MAX_CHUNK_LEN + 1;
        assert!(matches!(
            workspace_estimate_for(topology, oversized_profile, chunk_sized_samples, false),
            Err(Error::ModelLoadError(_))
        ));

        // The v2.1 checkpoint must not claim the Nemotron-3 envelope.
        assert!(matches!(
            workspace_estimate_for(
                topology,
                nemotron3_cfg,
                chunk_sized_samples,
                false
            ),
            Err(Error::ModelLoadError(_))
        ));
    }

    #[test]
    fn sortformer_vad_frame_mask_returns_silence_for_silent_audio() {
        let samples = vec![0.0; TARGET_SAMPLE_RATE as usize];
        let mask = sortformer_vad_frame_mask(&samples, 100, 160, 240.0, 200.0);

        assert_eq!(mask.len(), 100);
        assert!(mask.iter().all(|active| !active));
    }

    /// D3: the VAD gate is opt-in. A request value wins over the
    /// environment, which wins over the off default.
    #[test]
    fn vad_gate_is_off_unless_requested() {
        let _env = env_lock().lock().unwrap();
        let key = "IZWI_SORTFORMER_VAD_GATE";
        let previous = std::env::var(key).ok();
        let requested = |vad_gate| DiarizationConfig {
            vad_gate,
            ..DiarizationConfig::default()
        };
        std::env::remove_var(key);
        let unset = resolve_vad_gate(&DiarizationConfig::default());
        let request_on = resolve_vad_gate(&requested(Some(true)));
        std::env::set_var(key, "1");
        let env_on = resolve_vad_gate(&DiarizationConfig::default());
        let request_off = resolve_vad_gate(&requested(Some(false)));
        match previous {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
        assert!(!unset);
        assert!(request_on);
        assert!(env_on);
        assert!(!request_off);
    }

    /// SRT-4: when on, the gate widens each VAD speech region by one
    /// encoded Sortformer frame (80 ms) and trims whole segments to the
    /// regions; it never touches probabilities.
    #[test]
    fn vad_gate_pads_regions_by_one_encoded_frame_and_clips_segments() {
        let mask = [false, false, true, true, false, false, false, true, false];
        // v2.1 rows are 80 ms: frames 2-3 and 7, each widened by one frame.
        let v21 = padded_vad_regions(&mask, 1280);
        let expected = [(0.08, 0.40), (0.48, 0.72)];
        assert_eq!(v21.len(), expected.len());
        for (actual, expected) in v21.iter().zip(expected) {
            assert!((actual.0 - expected.0).abs() < 1e-6, "{v21:?}");
            assert!((actual.1 - expected.1).abs() < 1e-6, "{v21:?}");
        }
        // Nemotron-3 rows are 10 ms: the pad is still 80 ms (8 rows).
        let nemotron = padded_vad_regions(&mask, 160);
        assert_eq!(nemotron.len(), 1);
        assert!((nemotron[0].0 - 0.0).abs() < 1e-6);
        assert!((nemotron[0].1 - 0.16).abs() < 1e-6);

        // A segment is trimmed to the regions it meets, not split at the
        // pause between them; one outside every region is dropped.
        let gated = gate_segments(&[(0.0, 0.6), (0.9, 1.2), (0.2, 0.3)], &v21);
        assert_eq!(gated.len(), 2, "{gated:?}");
        assert!((gated[0].0 - 0.08).abs() < 1e-6 && (gated[0].1 - 0.6).abs() < 1e-6);
        assert!((gated[1].0 - 0.2).abs() < 1e-6 && (gated[1].1 - 0.3).abs() < 1e-6);
    }

    #[test]
    fn select_speaker_channels_prefers_active_speakers_by_duration() {
        let stats = vec![
            SpeakerActivityStats {
                speaker_idx: 0,
                total_duration_secs: 8.0,
                peak_probability: 0.70,
                segment_count: 2,
            },
            SpeakerActivityStats {
                speaker_idx: 1,
                total_duration_secs: 2.0,
                peak_probability: 0.90,
                segment_count: 3,
            },
            SpeakerActivityStats {
                speaker_idx: 2,
                total_duration_secs: 5.0,
                peak_probability: 0.60,
                segment_count: 1,
            },
            SpeakerActivityStats {
                speaker_idx: 3,
                total_duration_secs: 0.0,
                peak_probability: 0.99,
                segment_count: 0,
            },
        ];

        let selected = select_speaker_channels(&stats, 1, 2, 4);
        assert_eq!(selected, vec![0, 2]);
    }

    #[test]
    fn select_speaker_channels_backfills_when_min_exceeds_active() {
        let stats = vec![
            SpeakerActivityStats {
                speaker_idx: 0,
                total_duration_secs: 0.0,
                peak_probability: 0.40,
                segment_count: 0,
            },
            SpeakerActivityStats {
                speaker_idx: 1,
                total_duration_secs: 3.0,
                peak_probability: 0.60,
                segment_count: 2,
            },
            SpeakerActivityStats {
                speaker_idx: 2,
                total_duration_secs: 0.0,
                peak_probability: 0.80,
                segment_count: 0,
            },
            SpeakerActivityStats {
                speaker_idx: 3,
                total_duration_secs: 0.0,
                peak_probability: 0.20,
                segment_count: 0,
            },
        ];

        let selected = select_speaker_channels(&stats, 2, 2, 4);
        assert_eq!(selected, vec![1, 2]);
    }

    #[test]
    fn select_speaker_channels_keeps_all_four_active_speakers() {
        let stats = vec![
            SpeakerActivityStats {
                speaker_idx: 0,
                total_duration_secs: 8.0,
                peak_probability: 0.81,
                segment_count: 3,
            },
            SpeakerActivityStats {
                speaker_idx: 1,
                total_duration_secs: 6.0,
                peak_probability: 0.75,
                segment_count: 3,
            },
            SpeakerActivityStats {
                speaker_idx: 2,
                total_duration_secs: 4.0,
                peak_probability: 0.72,
                segment_count: 2,
            },
            SpeakerActivityStats {
                speaker_idx: 3,
                total_duration_secs: 2.0,
                peak_probability: 0.68,
                segment_count: 2,
            },
        ];

        let selected = select_speaker_channels(&stats, 1, 4, 4);
        assert_eq!(selected, vec![0, 1, 2, 3]);
    }

    #[test]
    fn select_speaker_channels_keeps_six_active_of_eight_channels() {
        // Nemotron-3's head is fixed at 8 channels; six carry speech and two
        // stay near-silent. A max_speakers=8 request must keep all six
        // active channels, not clamp to v2.1's four.
        let mut stats = (0..6usize)
            .map(|speaker_idx| SpeakerActivityStats {
                speaker_idx,
                total_duration_secs: 8.0 - speaker_idx as f32,
                peak_probability: 0.90 - speaker_idx as f32 * 0.05,
                segment_count: 3,
            })
            .collect::<Vec<_>>();
        stats.push(SpeakerActivityStats {
            speaker_idx: 6,
            total_duration_secs: 0.0,
            peak_probability: 0.01,
            segment_count: 0,
        });
        stats.push(SpeakerActivityStats {
            speaker_idx: 7,
            total_duration_secs: 0.0,
            peak_probability: 0.02,
            segment_count: 0,
        });

        let selected = select_speaker_channels(&stats, 1, 8, 8);
        assert_eq!(selected.len(), 6);
        assert_eq!(selected, vec![0, 1, 2, 3, 4, 5]);
        // A v2.1-era 4-speaker request against the same 8-channel stats
        // keeps only the four most active channels.
        let clamped = select_speaker_channels(&stats, 1, 4, 8);
        assert_eq!(clamped, vec![0, 1, 2, 3]);
    }

    fn streaming_geometry(cfg: &SortformerStreamingConfig) -> [usize; 6] {
        [
            cfg.chunk_len,
            cfg.chunk_left_context,
            cfg.chunk_right_context,
            cfg.fifo_len,
            cfg.spkcache_update_period,
            cfg.spkcache_len,
        ]
    }

    fn with_profile_env<T>(value: Option<&str>, f: impl FnOnce() -> T) -> T {
        let _env = env_lock().lock().unwrap();
        let key = "IZWI_SORTFORMER_STREAMING_PROFILE";
        let previous = std::env::var(key).ok();
        match value {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
        let result = f();
        match previous {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
        result
    }

    /// D2: file diarization defaults to the card very-high-latency profile
    /// (SRT-3), and the knob still selects the yaml and low profiles. An
    /// unknown value keeps the default instead of failing the load.
    #[test]
    fn resolve_streaming_profile_defaults_to_card_very_high_latency() {
        for (value, expected) in [
            (None, SortformerStreamingProfile::VeryHighLatency),
            (Some(""), SortformerStreamingProfile::VeryHighLatency),
            (Some("high"), SortformerStreamingProfile::VeryHighLatency),
            (Some("offline"), SortformerStreamingProfile::VeryHighLatency),
            (Some("model"), SortformerStreamingProfile::Model),
            (Some("Low-Latency"), SortformerStreamingProfile::LowLatency),
            (Some("bogus"), SortformerStreamingProfile::VeryHighLatency),
        ] {
            assert_eq!(
                with_profile_env(value, resolve_streaming_profile),
                expected,
                "{value:?}"
            );
        }
    }

    #[test]
    fn resolve_streaming_config_applies_the_v21_card_profiles() {
        let resolve = |profile| {
            resolve_streaming_config(
                profile,
                &SortformerModulesConfig::default(),
                512,
                4,
                SortformerEncoderKind::Conformer,
            )
            .unwrap()
        };
        // chunk / left / right / FIFO / update period / cache
        assert_eq!(
            streaming_geometry(&resolve(SortformerStreamingProfile::DEFAULT)),
            [340, 1, 40, 40, 300, 188]
        );
        assert_eq!(
            streaming_geometry(&resolve(SortformerStreamingProfile::Model)),
            [188, 1, 1, 0, 188, 188]
        );
        assert_eq!(
            streaming_geometry(&resolve(SortformerStreamingProfile::LowLatency)),
            [6, 1, 7, 188, 144, 188]
        );
    }

    /// NDZ-1/NDZ-3: Nemotron-3 defaults to its card offline profile and
    /// accepts every published profile, including under the knob a v2.1
    /// operator sets.
    #[test]
    fn resolve_streaming_config_nemotron3_defaults_to_card_offline_profile() {
        let modules_cfg = SortformerModulesConfig {
            fc_d_model: Some(512),
            subsampling_factor: Some(8),
            spkcache_len: Some(264),
            fifo_len: Some(0),
            chunk_len: Some(264),
            spkcache_update_period: Some(264),
            chunk_left_context: Some(0),
            chunk_right_context: Some(0),
            spkcache_sil_frames_per_spk: Some(1),
            ..SortformerModulesConfig::default()
        };
        let resolve = |profile| {
            resolve_streaming_config(
                profile,
                &modules_cfg,
                512,
                8,
                SortformerEncoderKind::FeatureStackingRope,
            )
            .unwrap()
        };
        let default = resolve(SortformerStreamingProfile::DEFAULT);
        assert_eq!(default.fc_d_model, 512);
        assert_eq!(default.num_speakers, 8);
        assert_eq!(default.output_frames_per_encoded_frame, 8);
        assert_eq!(streaming_geometry(&default), [340, 0, 40, 40, 300, 264]);
        assert_eq!(
            streaming_geometry(&resolve(SortformerStreamingProfile::Model)),
            [264, 0, 0, 0, 264, 264]
        );
        assert_eq!(
            streaming_geometry(&resolve(SortformerStreamingProfile::LowLatency)),
            [9, 0, 4, 264, 222, 264]
        );
        let under_high_knob = with_profile_env(Some("high"), || {
            resolve_streaming_config(
                resolve_streaming_profile(),
                &modules_cfg,
                512,
                8,
                SortformerEncoderKind::FeatureStackingRope,
            )
        });
        assert_eq!(
            streaming_geometry(&under_high_knob.unwrap()),
            [340, 0, 40, 40, 300, 264]
        );
    }

    /// Every published profile of both checkpoints fits its production
    /// envelope and the request authorization, and its physical state
    /// carries the FIFO components exactly when the profile has a FIFO.
    #[test]
    fn every_published_profile_fits_the_production_authorization() {
        // The pipeline's one-hour limit at 16 kHz.
        let one_hour = 60 * 60 * TARGET_SAMPLE_RATE as usize;
        let authorization = production_workspace_authorization(one_hour, false).unwrap();
        let cases = [
            (
                SortformerWorkspaceTopology::v21_production(),
                SortformerStreamingConfig {
                    chunk_len: 188,
                    chunk_left_context: 1,
                    chunk_right_context: 1,
                    fifo_len: 0,
                    spkcache_update_period: 188,
                    ..production_workspace_streaming_config()
                },
                SortformerEncoderKind::Conformer,
            ),
            (
                SortformerWorkspaceTopology::nemotron3_production(),
                nemotron3_checkpoint_streaming_config(),
                SortformerEncoderKind::FeatureStackingRope,
            ),
        ];
        for (topology, checkpoint, kind) in cases {
            for profile in [
                SortformerStreamingProfile::Model,
                SortformerStreamingProfile::LowLatency,
                SortformerStreamingProfile::VeryHighLatency,
            ] {
                let cfg = apply_streaming_profile(checkpoint, kind, profile)
                    .validate()
                    .unwrap();
                topology.validate_production(cfg).unwrap();
                let estimate = workspace_estimate_for(topology, cfg, one_hour, false).unwrap();
                assert!(
                    estimate.host_bytes <= authorization.host_bytes,
                    "{kind:?} {profile:?}"
                );
                assert!(
                    estimate.accelerator_bytes <= authorization.accelerator_bytes,
                    "{kind:?} {profile:?}"
                );
                let contract = physical::sortformer_invocation_contract(cfg).unwrap();
                let crate::kv::v2::StateDomainSpec::Tensor(state) = &contract.domains[0] else {
                    panic!("Sortformer state must be a tensor domain");
                };
                let expected_components = if cfg.fifo_len > 0 { 6 } else { 4 };
                assert_eq!(
                    state.components.len(),
                    expected_components,
                    "{kind:?} {profile:?}"
                );
            }
        }
    }

    #[test]
    fn resolve_postprocessing_params_defaults_to_reference_binarization() {
        let params = resolve_postprocessing_params(&DiarizationConfig::default(), None, None);
        assert_eq!(params.onset, 0.5);
        assert_eq!(params.offset, 0.5);
        assert_eq!(params.pad_onset, 0.0);
        assert_eq!(params.pad_offset, 0.0);
        assert_eq!(params.min_duration_on, 0.0);
        assert_eq!(params.min_duration_off, 0.0);
        assert!(params.filter_speech_first);
    }

    #[test]
    fn should_limit_speaker_channels_only_when_requested() {
        assert!(!should_limit_speaker_channels(&DiarizationConfig::default()));
        assert!(should_limit_speaker_channels(&DiarizationConfig {
            max_speakers: Some(2),
            ..DiarizationConfig::default()
        }));
    }

    #[test]
    fn plan_streaming_feature_chunks_matches_nemo_style_context_windows() {
        let mut cfg = streaming_cfg_for_test();
        cfg.chunk_len = 6;
        cfg.fifo_len = 188;
        cfg.chunk_right_context = 7;

        let plans = plan_streaming_feature_chunks(120, cfg);
        assert_eq!(plans.len(), 3);
        assert_eq!(plans[0].feature_start, 0);
        assert_eq!(plans[0].feature_end, 104);
        assert_eq!(plans[0].left_offset, 0);
        assert_eq!(plans[0].right_offset, 56);

        assert_eq!(plans[1].feature_start, 40);
        assert_eq!(plans[1].feature_end, 120);
        assert_eq!(plans[1].left_offset, 8);
        assert_eq!(plans[1].right_offset, 24);

        assert_eq!(plans[2].feature_start, 88);
        assert_eq!(plans[2].feature_end, 120);
        assert_eq!(plans[2].left_offset, 8);
        assert_eq!(plans[2].right_offset, 0);
    }

    #[test]
    fn streaming_chunk_offsets_follow_nemo_round_and_ceil_rules() {
        assert_eq!(pre_encoded_left_offset(3, 8), 0);
        assert_eq!(pre_encoded_left_offset(4, 8), 1);
        assert_eq!(pre_encoded_left_offset(8, 8), 1);

        assert_eq!(pre_encoded_right_offset(1, 8), 1);
        assert_eq!(pre_encoded_right_offset(9, 8), 2);
        assert_eq!(pre_encoded_right_offset(56, 8), 7);
    }

    #[test]
    fn update_streaming_state_moves_oldest_fifo_frames_into_speaker_cache() {
        let cfg = streaming_cfg_for_test();
        let state = SortformerStreamingState::new(2);
        let first_chunk = vec![vec![1.0, 1.0], vec![2.0, 2.0]];
        let first_preds = vec![
            vec![0.9, 0.0, 0.0, 0.0],
            vec![0.8, 0.0, 0.0, 0.0],
        ];
        let (state, first_chunk_preds, _) = update_streaming_state(
            state,
            &emb_rows(&first_chunk),
            &first_preds,
            0,
            0,
            true,
            cfg,
        )
        .unwrap();
        assert_eq!(first_chunk_preds, first_preds);
        assert!(state.spkcache.is_empty());
        assert_eq!(host_rows(&state.fifo), first_chunk);

        let second_chunk = vec![vec![3.0, 3.0], vec![4.0, 4.0]];
        let second_preds = vec![
            vec![0.9, 0.0, 0.0, 0.0],
            vec![0.8, 0.0, 0.0, 0.0],
            vec![0.0, 0.9, 0.0, 0.0],
            vec![0.0, 0.8, 0.0, 0.0],
        ];
        let (state, second_chunk_preds, _) = update_streaming_state(
            state,
            &emb_rows(&second_chunk),
            &second_preds,
            0,
            0,
            true,
            cfg,
        )
        .unwrap();

        assert_eq!(
            host_rows(&state.spkcache),
            vec![vec![1.0, 1.0], vec![2.0, 2.0]]
        );
        assert_eq!(host_rows(&state.fifo), vec![vec![3.0, 3.0], vec![4.0, 4.0]]);
        assert!(state.spkcache_preds.is_none());
        assert_eq!(
            second_chunk_preds,
            vec![vec![0.0, 0.9, 0.0, 0.0], vec![0.0, 0.8, 0.0, 0.0]]
        );
    }

    /// SRT-2, NeMo `streaming_update`: with chunk 6 < FIFO 188 the pop is
    /// `max(update_period, chunk - fifo_len + fifo)` = max(144, 4) = 144,
    /// leaving 48 FIFO frames, not the 186 a saturating subtraction pops.
    #[test]
    fn update_streaming_state_pops_update_period_when_chunk_shorter_than_fifo() {
        let cfg = SortformerStreamingConfig {
            spkcache_len: 188,
            fifo_len: 188,
            chunk_len: 6,
            spkcache_update_period: 144,
            chunk_left_context: 0,
            chunk_right_context: 0,
            spkcache_sil_frames_per_spk: 3,
            ..streaming_cfg_for_test()
        };
        let mut state = SortformerStreamingState::new(2);
        state.fifo = emb_rows(&(0..186).map(|i| vec![i as f32, 0.0]).collect::<Vec<_>>());
        let chunk = (186..192).map(|i| vec![i as f32, 0.0]).collect::<Vec<_>>();
        let preds = vec![vec![0.9, 0.0, 0.0, 0.0]; 192];

        let (state, _, _) =
            update_streaming_state(state, &emb_rows(&chunk), &preds, 0, 0, true, cfg).unwrap();

        assert_eq!(state.spkcache.len(), 144);
        assert_eq!(state.fifo.len(), 48);
        let fifo = host_rows(&state.fifo);
        assert_eq!(fifo[0], vec![144.0, 0.0]);
        assert_eq!(fifo[47], vec![191.0, 0.0]);
    }

    /// The pop arithmetic against NeMo's signed formula on hand-computed
    /// cases: `min(max(period, chunk - fifo_len + fifo), fifo + chunk)`.
    #[test]
    fn update_streaming_state_pop_length_follows_nemo_signed_formula() {
        // (fifo_len, update_period, current fifo, chunk, expected pop)
        let cases = [
            (188, 144, 186, 6, 144),
            (188, 144, 183, 6, 144),
            (40, 300, 0, 340, 300),
            (40, 300, 40, 340, 340),
            (40, 300, 40, 120, 160),
            (0, 188, 0, 188, 188),
            (8, 12, 8, 20, 20),
            (8, 12, 4, 20, 16),
        ];
        for (fifo_len, period, current, chunk_len, expected) in cases {
            let cfg = SortformerStreamingConfig {
                spkcache_len: 4096,
                fifo_len,
                chunk_len,
                spkcache_update_period: period,
                ..streaming_cfg_for_test()
            };
            let mut state = SortformerStreamingState::new(2);
            state.fifo = emb_rows(&vec![vec![0.0, 0.0]; current]);
            let chunk = emb_rows(&vec![vec![1.0, 1.0]; chunk_len]);
            let preds = vec![vec![0.9, 0.0, 0.0, 0.0]; current + chunk_len];
            let (state, _, _) =
                update_streaming_state(state, &chunk, &preds, 0, 0, true, cfg).unwrap();
            assert_eq!(
                state.spkcache.len(),
                expected,
                "fifo_len {fifo_len}, period {period}, fifo {current}, chunk {chunk_len}"
            );
            assert_eq!(state.fifo.len(), current + chunk_len - expected);
        }
    }

    /// SRT-1, NeMo `_get_topk_indices`: every selected -inf pick becomes
    /// `max_index` before the sort, so the disabled slots land after all
    /// kept frames and silence pads instead of at their flat index.
    #[test]
    fn compress_spkcache_places_disabled_slots_at_the_end_like_nemo() {
        let cfg = SortformerStreamingConfig {
            spkcache_len: 16,
            spkcache_sil_frames_per_spk: 1,
            ..streaming_cfg_for_test()
        };
        let emb_seq = (0..20).map(|i| vec![i as f32, 0.0]).collect::<Vec<_>>();
        let mut preds = vec![vec![0.05; 4]; 20];
        for frame in 0..3 {
            preds[frame][0] = 0.9;
        }
        for frame in 5..7 {
            preds[frame][1] = 0.9;
        }
        let silence = vec![-1.0, -1.0];

        let (cache, cache_preds) =
            compress_spkcache(&emb_rows(&emb_seq), &preds, &silence, cfg).unwrap();
        let cache = host_rows(&cache);

        // Kept: speaker 0's frames 0-2 then its silence pad, speaker 1's
        // frames 5-6 then its pad, the pads of speakers 2 and 3, and the
        // seven disabled -inf picks last.
        let mut expected = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![2.0, 0.0],
            silence.clone(),
            vec![5.0, 0.0],
            vec![6.0, 0.0],
        ];
        expected.extend(std::iter::repeat_n(silence.clone(), 10));
        assert_eq!(cache, expected);
        for (row, pred) in cache.iter().zip(&cache_preds) {
            if *row == silence {
                assert_eq!(pred, &vec![0.0; 4]);
            } else {
                assert_eq!(pred, &preds[row[0] as usize]);
            }
        }
    }

    #[test]
    fn binarization_matches_nemo_threshold_transitions() {
        let params = PostProcessingParams {
            onset: 0.5,
            offset: 0.5,
            pad_onset: 0.0,
            pad_offset: 0.0,
            min_duration_on: 0.0,
            min_duration_off: 0.0,
            filter_speech_first: true,
        };

        let sequence = vec![0.1, 0.6, 0.7, 0.2, 0.1];
        let segments = binarization(&sequence, TS_VAD_FRAME_LENGTH_SECS, &params);

        assert_eq!(segments, vec![(0.01, 0.03)]);
    }

    fn binarization_params(
        onset: f32,
        offset: f32,
        pad_onset: f32,
        pad_offset: f32,
    ) -> PostProcessingParams {
        PostProcessingParams {
            onset,
            offset,
            pad_onset,
            pad_offset,
            min_duration_on: 0.0,
            min_duration_off: 0.0,
            filter_speech_first: true,
        }
    }

    fn assert_segments_near(actual: &[(f32, f32)], expected: &[(f32, f32)]) {
        assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
        for (a, e) in actual.iter().zip(expected) {
            assert!(
                (a.0 - e.0).abs() < 1e-5 && (a.1 - e.1).abs() < 1e-5,
                "{actual:?} vs {expected:?}"
            );
        }
    }

    /// SRT-11: NeMo's `binarization_vectorized` at the native 80 ms frame
    /// ends a segment still active at the last frame at N frames (0.48 s);
    /// the legacy loop over 10 ms repeats stopped at 8N - 1 (0.47 s).
    #[test]
    fn vectorized_binarization_ends_a_trailing_segment_at_the_sequence_end() {
        let params = binarization_params(0.5, 0.5, 0.0, 0.0);
        let sequence = [0.2, 0.7, 0.8, 0.3, 0.6, 0.9];
        assert_segments_near(
            &binarization(&sequence, 0.08, &params),
            &[(0.08, 0.24), (0.32, 0.48)],
        );
        // The same probabilities as v2.1 rows through the post-processing.
        let rows = sequence.iter().map(|p| vec![*p]).collect::<Vec<_>>();
        assert_segments_near(
            &predlist_to_timestamps(&rows, 0, &params, TS_VAD_UNIT_FRAME_COUNT),
            &[(0.08, 0.24), (0.32, 0.48)],
        );
    }

    /// SRT-11: with onset < offset (the dihard3 preset's 0.56 / 1.0) a frame
    /// between the thresholds toggles speech instead of holding it; a frame
    /// at or above offset sets it, one at or below onset clears it, and NaN
    /// neither sets, clears nor toggles.
    #[test]
    fn vectorized_binarization_toggles_between_onset_and_offset() {
        let params = binarization_params(0.3, 0.7, 0.0, 0.0);
        let frame = TS_VAD_FRAME_LENGTH_SECS;
        assert_segments_near(
            &binarization(&[0.5, 0.5, 0.5], frame, &params),
            &[(0.0, 0.01), (0.02, 0.03)],
        );
        assert_segments_near(
            &binarization(&[0.5, 0.8, 0.5, 0.5, 0.2, 0.5, 0.9], frame, &params),
            &[(0.0, 0.02), (0.03, 0.04), (0.05, 0.07)],
        );
        assert_segments_near(
            &binarization(&[0.8, f32::NAN, 0.1], frame, &params),
            &[(0.0, 0.02)],
        );
    }

    /// SRT-11: pads widen each segment (the start clamped at 0, the end not)
    /// and padded segments that touch or overlap merge.
    #[test]
    fn vectorized_binarization_pads_and_merges_like_nemo() {
        let params = binarization_params(0.5, 0.5, 0.1, 0.05);
        let sequence = [0.9, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.9];
        assert_segments_near(
            &binarization(&sequence, 0.08, &params),
            &[(0.0, 0.29), (0.46, 0.69)],
        );
    }

    /// SRT-12: `max_speakers` below the active count reassigns the dropped
    /// channel's speech to the kept channel most probable over it instead of
    /// deleting it, so the attributed speech does not shrink.
    #[test]
    fn limiting_speakers_reassigns_dropped_speech_to_the_most_probable_kept_speaker() {
        // 80 ms rows over 3.2 s: speaker 0 talks 0-1.6 s, speaker 1 1.6-2.4
        // s (where speaker 2 is the more probable of the kept two), speaker 2
        // 2.4-3.2 s and 0.8-1.2 s over speaker 0.
        let stride = TS_VAD_UNIT_FRAME_COUNT * PRODUCTION_HOP_LENGTH;
        let probs = (0..40)
            .map(|frame| match frame {
                0..=9 => vec![0.9, 0.1, 0.1],
                10..=14 => vec![0.9, 0.1, 0.8],
                15..=19 => vec![0.9, 0.1, 0.1],
                20..=29 => vec![0.3, 0.9, 0.6],
                _ => vec![0.1, 0.1, 0.9],
            })
            .collect::<Vec<_>>();
        let segment = |speaker_idx, start_secs, end_secs| RawSegment {
            speaker_idx,
            start_secs,
            end_secs,
            confidence: None,
        };
        let mut segments = vec![
            segment(0, 0.0, 1.6),
            segment(2, 0.8, 1.2),
            segment(1, 1.6, 2.4),
            segment(2, 2.4, 3.2),
        ];
        let speech = |segments: &[RawSegment]| {
            merge_overlap_ranges(
                &segments
                    .iter()
                    .map(|segment| (segment.start_secs, segment.end_secs))
                    .collect::<Vec<_>>(),
            )
        };
        let before = speech(&segments);
        let stats = (0..3)
            .map(|speaker_idx| SpeakerActivityStats {
                speaker_idx,
                total_duration_secs: segments
                    .iter()
                    .filter(|segment| segment.speaker_idx == speaker_idx)
                    .map(|segment| segment.end_secs - segment.start_secs)
                    .sum(),
                peak_probability: 0.9,
                segment_count: 1,
            })
            .collect::<Vec<_>>();
        let selected = select_speaker_channels(&stats, 1, 2, 3);
        assert_eq!(selected, vec![0, 2]);
        reassign_dropped_speaker_segments(&mut segments, &selected, &probs, stride);

        assert!(segments.iter().all(|segment| segment.speaker_idx != 1));
        let moved = &segments[2];
        assert_eq!(
            (moved.speaker_idx, moved.start_secs, moved.end_secs),
            (2, 1.6, 2.4)
        );
        // Its confidence is now speaker 2's probability over it.
        assert_eq!(
            moved.confidence,
            average_speaker_probability_for_range(&probs, 2, 1.6, 2.4, stride)
        );
        assert!(moved.confidence.unwrap() < 0.7);
        assert_eq!(speech(&segments), before);
    }

    #[test]
    fn filtering_respects_filter_speech_first_toggle() {
        let segments = vec![(0.0, 0.10), (0.14, 0.22)];
        let speech_first = PostProcessingParams {
            onset: 0.5,
            offset: 0.5,
            pad_onset: 0.0,
            pad_offset: 0.0,
            min_duration_on: 0.12,
            min_duration_off: 0.08,
            filter_speech_first: true,
        };
        let nonspeech_first = PostProcessingParams {
            filter_speech_first: false,
            ..speech_first
        };

        assert!(filtering(&segments, &speech_first).is_empty());
        assert_eq!(filtering(&segments, &nonspeech_first), vec![(0.0, 0.22)]);
    }

    #[test]
    fn merge_adjacent_segments_merges_per_speaker_with_overlap_present() {
        let mut segments = vec![
            DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.0,
                end_secs: 1.0,
                confidence: Some(0.8),
            },
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 0.8,
                end_secs: 1.4,
                confidence: Some(0.9),
            },
            DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start_secs: 1.05,
                end_secs: 2.0,
                confidence: Some(0.6),
            },
        ];

        merge_adjacent_segments(&mut segments, 0.1);

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].speaker, "SPEAKER_00");
        assert!((segments[0].start_secs - 0.0).abs() < 1e-6);
        assert!((segments[0].end_secs - 2.0).abs() < 1e-6);
        assert_eq!(segments[1].speaker, "SPEAKER_01");
    }

    #[test]
    #[ignore = "requires local Sortformer checkpoint"]
    fn nemotron3_local_checkpoint_diarizes_eight_channels_on_cpu() {
        let models_root = std::env::var("IZWI_MODELS_DIR")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                dirs::data_local_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join("izwi")
                    .join("models")
            });
        let model_dir = models_root.join(ModelVariant::Nemotron3Diarization.dir_name());
        if !model_dir.join("Nemotron-3-Diarization.nemo").exists() {
            eprintln!(
                "Skipping Nemotron-3 checkpoint test, model not found at {}",
                model_dir.display()
            );
            return;
        }

        let audio_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/diarization-2.mp3");
        let audio_bytes = std::fs::read(&audio_path).expect("sample audio should exist");
        let (samples, sample_rate) = decode_audio_bytes(&audio_bytes).expect("audio should decode");

        let model = {
            let _env = env_lock().lock().unwrap();
            SortformerDiarizerModel::load(
                &model_dir,
                ModelVariant::Nemotron3Diarization,
                DeviceProfile::cpu(),
            )
        }
        .expect("nemotron3 checkpoint should load");
        let config = DiarizationConfig {
            max_speakers: Some(8),
            ..DiarizationConfig::default()
        };
        let diarization = model
            .diarize(&samples, sample_rate, &config)
            .expect("nemotron3 diarization should run");

        assert!(
            !diarization.segments.is_empty(),
            "expected diarization segments from the real checkpoint"
        );
        let duration = samples.len() as f32 / sample_rate as f32;
        for segment in &diarization.segments {
            let speaker = parse_test_speaker_id(&segment.speaker);
            assert!(speaker < 8, "speaker channel out of range: {segment:?}");
            assert!(segment.start_secs < segment.end_secs);
            assert!(segment.end_secs <= duration + 0.1);
            // The upsampler emits rows at the 10 ms mel rate; segment
            // boundaries must land on that grid (within f32 slop).
            for boundary in [segment.start_secs, segment.end_secs] {
                let millis = (boundary * 1000.0) as f64;
                let nearest_grid = (millis / 10.0).round() * 10.0;
                assert!(
                    (millis - nearest_grid).abs() < 0.05,
                    "boundary {boundary} not on the 10 ms grid"
                );
            }
        }
        let distinct_speakers = diarization
            .segments
            .iter()
            .map(|segment| parse_test_speaker_id(&segment.speaker))
            .collect::<std::collections::BTreeSet<_>>();
        println!(
            "nemotron3 cpu diarization: {} segments, {} distinct channels over {duration:.1}s",
            diarization.segments.len(),
            distinct_speakers.len()
        );
    }

    fn parse_test_speaker_id(label: &str) -> usize {
        label
            .chars()
            .filter(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse::<usize>()
            .unwrap_or(0)
    }

    /// Builds a synthetic `SortformerRopeEncoder` checkpoint: 2 layers,
    /// d_model 64 (8 heads x 8), FFN 32, 16 mel bins stacked by 4.
    fn rope_encoder_fixture() -> Result<SortformerRopeEncoder> {
        rope_encoder_fixture_with_base(10_000.0)
    }

    fn rope_encoder_fixture_with_base(rope_base: f32) -> Result<SortformerRopeEncoder> {
        fn seed_values(len: usize, seed: f32) -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32 + 1.0) * seed) * 0.01)
                .collect()
        }
        fn insert(
            tensors: &mut std::collections::HashMap<String, Tensor>,
            name: String,
            shape: Vec<usize>,
            seed: f32,
        ) {
            tensors.insert(
                format!("encoder.{name}"),
                Tensor::from_vec(seed_values(shape.iter().product(), seed), shape, &Device::Cpu)
                    .unwrap(),
            );
        }
        let mut tensors = std::collections::HashMap::<String, Tensor>::new();
        insert(&mut tensors, "pre_encode.proj.weight".into(), vec![64, 64], 1.0);
        insert(&mut tensors, "embed_norm.weight".into(), vec![64], 1.0);
        insert(&mut tensors, "embed_norm.bias".into(), vec![64], 0.0);
        insert(&mut tensors, "final_norm.weight".into(), vec![64], 1.0);
        insert(&mut tensors, "final_norm.bias".into(), vec![64], 0.0);
        for layer in 0..2 {
            let prefix = format!("layers.{layer}.");
            insert(&mut tensors, format!("{prefix}norm1.weight"), vec![64], 1.0);
            insert(&mut tensors, format!("{prefix}norm1.bias"), vec![64], 0.0);
            insert(&mut tensors, format!("{prefix}norm2.weight"), vec![64], 1.0);
            insert(&mut tensors, format!("{prefix}norm2.bias"), vec![64], 0.0);
            insert(&mut tensors, format!("{prefix}attn.w_qkv.weight"), vec![192, 64], 2.0);
            insert(&mut tensors, format!("{prefix}attn.out_proj.weight"), vec![64, 64], 3.0);
            insert(&mut tensors, format!("{prefix}attn.out_proj.bias"), vec![64], 0.0);
            insert(&mut tensors, format!("{prefix}ffn.net.0.weight"), vec![32, 64], 4.0);
            insert(&mut tensors, format!("{prefix}ffn.net.0.bias"), vec![32], 0.0);
            insert(&mut tensors, format!("{prefix}ffn.net.3.weight"), vec![64, 32], 5.0);
            insert(&mut tensors, format!("{prefix}ffn.net.3.bias"), vec![64], 0.0);
        }
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        SortformerRopeEncoder::load(vb.pp("encoder"), 16, 8, 4, rope_base)
    }

    #[test]
    fn rope_encoder_stacks_frames_frame_major_and_pads_tail() {
        let encoder = rope_encoder_fixture().unwrap();
        // [1, time=6, bins=16], value = t * 100 + f.
        let features = Tensor::from_vec(
            (0..6usize)
                .flat_map(|t| (0..16usize).map(move |f| (t * 100 + f) as f32))
                .collect::<Vec<_>>(),
            (1, 6, 16),
            &Device::Cpu,
        )
        .unwrap();
        let stacked = encoder.stack_features(&features, 6).unwrap();
        assert_eq!(stacked.dims3().unwrap(), (1, 2, 64));
        let values = stacked.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Row 0: frames 0..4 stacked frame-major; row 1: frames 4, 5, then
        // two zero-padded frames.
        let row0 = (0..4usize)
            .flat_map(|t| (0..16usize).map(move |f| (t * 100 + f) as f32))
            .collect::<Vec<_>>();
        let mut row1 = (4..6usize)
            .flat_map(|t| (0..16usize).map(move |f| (t * 100 + f) as f32))
            .collect::<Vec<_>>();
        row1.extend(std::iter::repeat_n(0.0, 32));
        assert_eq!(values[..64], row0[..]);
        assert_eq!(values[64..], row1[..]);
    }

    #[test]
    fn rope_encoder_loads_synthetic_checkpoint_and_forwards() {
        let encoder = rope_encoder_fixture().unwrap();
        assert_eq!(encoder.layers.len(), 2);
        assert_eq!(encoder.d_model, 64);
        assert_eq!(encoder.stacking_factor, 4);

        // 35 mel frames -> 9 stacked groups (final one padded).
        let features = Tensor::from_vec(
            (0..35 * 16)
                .map(|i| (i as f32) * 0.01)
                .collect::<Vec<_>>(),
            (1, 35, 16),
            &Device::Cpu,
        )
        .unwrap();
        let (embeds, embedded_len) = encoder.pre_encode(&features, 35).unwrap();
        assert_eq!(embedded_len, 9);
        assert_eq!(embeds.dims3().unwrap(), (1, 9, 64));

        let (encoded, encoded_len) = encoder.forward_pre_encoded(&embeds, embedded_len).unwrap();
        assert_eq!(encoded_len, 9);
        assert_eq!(encoded.dims3().unwrap(), (1, 9, 64));
        let values = encoded.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(values.iter().all(|v| v.is_finite()));

        // The RoPE position ceiling fails closed.
        assert!(encoder
            .rope_cos_sin(NEMOTRON3_ROPE_MAX_POSITIONS + 1, &Device::Cpu)
            .is_err());
    }

    /// NeMo's `FeedForward` and HF's `hidden_act: gelu` use the exact erf
    /// GELU; candle's `gelu()` is the tanh approximation (NDZ-2).
    #[test]
    fn rope_layer_ffn_uses_exact_erf_gelu() {
        let d_model = 8;
        let identity = Tensor::eye(d_model, DType::F32, &Device::Cpu).unwrap();
        let zeros = |shape: &[usize]| Tensor::zeros(shape, DType::F32, &Device::Cpu).unwrap();
        let ones = |len: usize| Tensor::ones(len, DType::F32, &Device::Cpu).unwrap();
        let tensors = std::collections::HashMap::from([
            ("norm1.weight".to_string(), ones(d_model)),
            ("norm1.bias".to_string(), zeros(&[d_model])),
            ("norm2.weight".to_string(), ones(d_model)),
            ("norm2.bias".to_string(), zeros(&[d_model])),
            // Zero attention leaves h = x, so the block is x + ffn(norm2(x)).
            (
                "attn.w_qkv.weight".to_string(),
                zeros(&[3 * d_model, d_model]),
            ),
            (
                "attn.out_proj.weight".to_string(),
                zeros(&[d_model, d_model]),
            ),
            ("attn.out_proj.bias".to_string(), zeros(&[d_model])),
            ("ffn.net.0.weight".to_string(), identity.clone()),
            ("ffn.net.0.bias".to_string(), zeros(&[d_model])),
            ("ffn.net.3.weight".to_string(), identity),
            ("ffn.net.3.bias".to_string(), zeros(&[d_model])),
        ]);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        let layer = SortformerRopeLayer::load(vb, d_model, 2, 4).unwrap();
        let values = (0..3 * d_model)
            .map(|i| ((i * 7) % 11) as f32 - 5.0)
            .collect::<Vec<_>>();
        let x = Tensor::from_vec(values, (1, 3, d_model), &Device::Cpu).unwrap();
        let (cos, sin) = (
            Tensor::ones((3, 4), DType::F32, &Device::Cpu).unwrap(),
            zeros(&[3, 4]),
        );

        let actual = layer.forward(&x, &cos, &sin).unwrap();
        let normed = layer.norm2.forward(&x).unwrap();
        let expected = (&x + normed.gelu_erf().unwrap()).unwrap();
        let max_abs = (actual - expected)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            max_abs <= 1e-6,
            "FFN activation is not exact GELU: {max_abs:e}"
        );
    }

    #[test]
    fn subpixel_upsampler_expands_to_output_rate() {
        // Center-tap identity weights reproduce nearest-neighbour upsampling,
        // matching the reference initializer.
        let weight = Tensor::from_vec(
            (0..1536usize)
                .flat_map(move |c| {
                    (0..192usize)
                        .flat_map(move |i| (0..3usize).map(move |k| if c % 192 == i && k == 1 { 1.0 } else { 0.0 }))
                })
                .collect::<Vec<_>>(),
            (1536, 192, 3),
            &Device::Cpu,
        )
        .unwrap();
        let tensors = std::collections::HashMap::from([(
            "sortformer_modules.subpixel_upsample.weight".to_string(),
            weight,
        )]);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        let upsampler = SortformerSubpixelUpsampler::load(vb.pp("sortformer_modules")).unwrap();

        let x = Tensor::from_vec(
            (0..3 * 192).map(|i| (i as f32) * 0.5).collect::<Vec<_>>(),
            (1, 3, 192),
            &Device::Cpu,
        )
        .unwrap();
        let out = upsampler.forward(&x).unwrap();
        assert_eq!(out.dims3().unwrap(), (1, 24, 192));
        let x_values = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let out_values = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for g in 0..3usize {
            for r in 0..8usize {
                for d in 0..192usize {
                    assert_eq!(out_values[(g * 8 + r) * 192 + d], x_values[g * 192 + d]);
                }
            }
        }
    }

    #[test]
    fn pool_probability_rows_averages_within_encoded_frames() {
        let probs = vec![
            vec![0.1f32, 0.2],
            vec![0.3, 0.4],
            vec![0.5, 0.6],
            vec![0.7, 0.8],
        ];
        let pooled = pool_probability_rows(&probs, 2, 2, 2).unwrap();
        let expected = [vec![0.2, 0.3], vec![0.6, 0.7]];
        for (row, expected_row) in pooled.iter().zip(expected.iter()) {
            for (value, expected_value) in row.iter().zip(expected_row.iter()) {
                assert!((value - expected_value).abs() < 1e-6);
            }
        }

        assert!(pool_probability_rows(&probs, 2, 2, 4).is_err());
        assert!(pool_probability_rows(&probs, 3, 2, 2).is_err());
    }

    /// SRT-7: host pooling of the once-read probabilities is bit-identical
    /// to the device `mean` it replaces (Nemotron-3's 8x upsampled rows).
    #[test]
    fn host_pooling_matches_candle_mean_bit_for_bit() {
        let (encoded_len, factor, speakers) = (37, 8, 8);
        let values = random_values(encoded_len * factor * speakers, 0xB17E, 0.5, 0.5);
        let probs = Tensor::from_vec(
            values.clone(),
            (1, encoded_len * factor, speakers),
            &Device::Cpu,
        )
        .unwrap();
        let expected = probs
            .reshape((1, encoded_len, factor, speakers))
            .unwrap()
            .mean(2)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let rows = values
            .chunks(speakers)
            .map(<[f32]>::to_vec)
            .collect::<Vec<_>>();
        let pooled = pool_probability_rows(&rows, encoded_len, factor, speakers).unwrap();
        for (actual, expected) in pooled.iter().flatten().zip(expected.iter().flatten()) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn update_streaming_state_emits_upsampled_output_range() {
        let mut cfg = streaming_cfg_for_test();
        cfg.output_frames_per_encoded_frame = 8;

        let state = SortformerStreamingState::new(2);
        let chunk_rows = emb_rows(&[vec![1.0, 1.0], vec![2.0, 2.0]]);
        let preds = vec![vec![0.9, 0.0, 0.0, 0.0], vec![0.8, 0.0, 0.0, 0.0]];
        let (_, _, output_range) =
            update_streaming_state(state, &chunk_rows, &preds, 0, 0, false, cfg).unwrap();
        assert_eq!(output_range, 0..16);

        // Left/right context frames are attended but not emitted: their
        // upsampled rows sit outside the output range.
        let state = SortformerStreamingState::new(2);
        let chunk_rows = emb_rows(&[vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0]]);
        let preds = vec![
            vec![0.9, 0.0, 0.0, 0.0],
            vec![0.8, 0.0, 0.0, 0.0],
            vec![0.0, 0.9, 0.0, 0.0],
        ];
        let (_, _, output_range) =
            update_streaming_state(state, &chunk_rows, &preds, 1, 1, false, cfg).unwrap();
        assert_eq!(output_range, 8..16);
    }

    /// The workspace estimate bounds the measured per-chunk peak of each
    /// streaming profile, and over-reserves v2.1 by at most the 1.5x safety
    /// factor. Peaks are the largest process-footprint growth of one diarize
    /// call recorded by `sortformer_workspace_measured_peak_per_profile`
    /// (M1 Pro, CPU, release build, 2026-10-14), for the sample counts it
    /// uses: the speaker cache and FIFO filled, then three full chunks.
    #[test]
    fn sortformer_workspace_estimate_bounds_the_measured_peak() {
        let v21 = SortformerWorkspaceTopology::v21_production();
        for (profile, samples, measured) in [
            (
                SortformerStreamingProfile::Model,
                970_240_usize,
                384_631_072_u64,
            ),
            (
                SortformerStreamingProfile::VeryHighLatency,
                1_754_880,
                750_699_096,
            ),
            (SortformerStreamingProfile::LowLatency, 535_040, 234_455_304),
        ] {
            let cfg = resolve_streaming_config(
                profile,
                &SortformerModulesConfig::default(),
                PRODUCTION_CONFORMER_D_MODEL,
                4,
                SortformerEncoderKind::Conformer,
            )
            .unwrap();
            let estimate = workspace_estimate_for(v21, cfg, samples, false).unwrap();
            let total = estimate.host_bytes + estimate.accelerator_bytes;
            assert!(
                total >= measured,
                "v2.1 {profile:?}: {total} B < measured {measured} B"
            );
            assert!(
                total <= measured * 3 / 2,
                "v2.1 {profile:?}: {total} B is over 1.5x the measured {measured} B"
            );
        }

        // Nemotron-3's peak was 51 MiB: the estimate bounds it with room to
        // spare (its RoPE and upsampler counts are not calibrated here).
        let nemotron3 = workspace_estimate_for(
            SortformerWorkspaceTopology::nemotron3_production(),
            nemotron3_workspace_streaming_config(),
            1_351_680,
            false,
        )
        .unwrap();
        assert!(nemotron3.host_bytes + nemotron3.accelerator_bytes >= 53_624_904);
        // Its default card offline profile peaked at 108.6 MiB (115.5 s,
        // composite 684 frames; 2026-10-15).
        let card_offline = apply_streaming_profile(
            nemotron3_checkpoint_streaming_config(),
            SortformerEncoderKind::FeatureStackingRope,
            SortformerStreamingProfile::VeryHighLatency,
        );
        let estimate = workspace_estimate_for(
            SortformerWorkspaceTopology::nemotron3_production(),
            card_offline,
            1_848_320,
            false,
        )
        .unwrap();
        assert!(estimate.host_bytes + estimate.accelerator_bytes >= 113_901_664);
    }

    /// Live process memory for the measurement below: the physical
    /// footprint (what the OS and Metal working set charge) and the bytes
    /// malloc has handed out and not taken back. `purge` first returns
    /// malloc's cached free pages, so a baseline does not hide reuse.
    #[cfg(target_os = "macos")]
    fn process_memory_sample(purge: bool) -> (u64, u64) {
        #[repr(C)]
        struct MallocStatistics {
            blocks_in_use: u32,
            size_in_use: usize,
            max_size_in_use: usize,
            size_allocated: usize,
        }
        extern "C" {
            fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut std::ffi::c_void) -> i32;
            fn malloc_zone_statistics(zone: *mut std::ffi::c_void, stats: *mut MallocStatistics);
            fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize;
        }
        // rusage_info_v0: a 16-byte UUID, then u64 fields; ri_phys_footprint
        // is the eighth of them (index 9 counting the UUID as two words).
        let mut info = [0_u64; 32];
        let mut stats = MallocStatistics {
            blocks_in_use: 0,
            size_in_use: 0,
            max_size_in_use: 0,
            size_allocated: 0,
        };
        // SAFETY: both buffers are large enough for the structures libSystem
        // writes (rusage_info_v0 is 96 bytes) and outlive the calls. Pressure
        // relief only returns free pages to the OS.
        unsafe {
            if purge {
                malloc_zone_pressure_relief(std::ptr::null_mut(), 0);
            }
            assert_eq!(
                proc_pid_rusage(std::process::id() as i32, 0, info.as_mut_ptr().cast()),
                0
            );
            malloc_zone_statistics(std::ptr::null_mut(), &mut stats);
        }
        (info[9], stats.size_in_use as u64)
    }

    /// Measures one diarize call per streaming profile on the real
    /// checkpoints (CPU, which is also where Sortformer runs on the Metal
    /// backend) and prints the peak footprint and malloc deltas next to the
    /// loaded model's own workspace estimate. The audio fills the speaker
    /// cache and FIFO and then runs full chunks, so the composite is at its
    /// largest. Run with `--ignored --nocapture`; the figures calibrate
    /// `workspace_estimate_for` (see `MEASURED_*` in the bound test).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires local Sortformer checkpoints; measurement only"]
    fn sortformer_workspace_measured_peak_per_profile() {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        use std::sync::Arc;
        let models_root = std::env::var("IZWI_MODELS_DIR")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                dirs::data_local_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join("izwi")
                    .join("models")
            });
        let _env = env_lock().lock().unwrap();
        let key = "IZWI_SORTFORMER_STREAMING_PROFILE";
        let previous = std::env::var(key).ok();
        for (variant, profile) in [
            (ModelVariant::DiarStreamingSortformer4SpkV21, "model"),
            (ModelVariant::DiarStreamingSortformer4SpkV21, "high_latency"),
            (ModelVariant::DiarStreamingSortformer4SpkV21, "low_latency"),
            (ModelVariant::Nemotron3Diarization, "model"),
            (ModelVariant::Nemotron3Diarization, "high_latency"),
            (ModelVariant::Nemotron3Diarization, "low_latency"),
        ] {
            let model_dir = models_root.join(variant.dir_name());
            if !model_dir.exists() {
                eprintln!(
                    "skipping {variant}: no checkpoint at {}",
                    model_dir.display()
                );
                continue;
            }
            std::env::set_var(key, profile);
            let model = SortformerDiarizerModel::load(&model_dir, variant, DeviceProfile::cpu())
                .expect("checkpoint should load");
            let cfg = model.model.streaming.expect("streaming profile");
            let chunk_frames = (cfg.chunk_len + cfg.chunk_left_context + cfg.chunk_right_context)
                * cfg.subsampling_factor;
            let fill_frames = (cfg.spkcache_len + cfg.fifo_len) * cfg.subsampling_factor;
            let samples = (fill_frames + 3 * chunk_frames) * PRODUCTION_HOP_LENGTH;
            let mut state = 0x9e37_79b9_7f4a_7c15_u64;
            let audio: Vec<f32> = (0..samples)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    ((state >> 40) as f32 / (1_u64 << 24) as f32 - 0.5) * 0.2
                })
                .collect();
            let config = DiarizationConfig::default();
            model
                .diarize(
                    &audio[..TARGET_SAMPLE_RATE as usize / 2],
                    TARGET_SAMPLE_RATE,
                    &config,
                )
                .expect("warm-up");
            let estimate = model.workspace_estimate(samples).unwrap();
            let mut peaks = Vec::new();
            for _ in 0..3 {
                let (footprint_base, malloc_base) = process_memory_sample(true);
                let stop = Arc::new(AtomicBool::new(false));
                let footprint_peak = Arc::new(AtomicU64::new(footprint_base));
                let malloc_peak = Arc::new(AtomicU64::new(malloc_base));
                let sampler = {
                    let (stop, footprint_peak, malloc_peak) =
                        (stop.clone(), footprint_peak.clone(), malloc_peak.clone());
                    std::thread::spawn(move || {
                        while !stop.load(Ordering::Acquire) {
                            let (footprint, malloc) = process_memory_sample(false);
                            footprint_peak.fetch_max(footprint, Ordering::AcqRel);
                            malloc_peak.fetch_max(malloc, Ordering::AcqRel);
                            std::thread::sleep(std::time::Duration::from_micros(200));
                        }
                    })
                };
                model
                    .diarize(&audio, TARGET_SAMPLE_RATE, &config)
                    .expect("diarize");
                stop.store(true, Ordering::Release);
                sampler.join().unwrap();
                peaks.push((
                    footprint_peak.load(Ordering::Acquire) - footprint_base,
                    malloc_peak.load(Ordering::Acquire) - malloc_base,
                ));
            }
            let footprint = peaks.iter().map(|peak| peak.0).max().unwrap();
            let malloc = peaks.iter().map(|peak| peak.1).max().unwrap();
            let estimated = estimate.host_bytes + estimate.accelerator_bytes;
            let mib = |bytes: u64| bytes as f64 / (1 << 20) as f64;
            println!(
                "{variant} {profile}: {:.1} s audio, chunk {chunk_frames} feature frames, \
                 composite {} frames; footprint peak +{:.1} MiB, malloc peak +{:.1} MiB \
                 (runs {peaks:?}); estimate {:.1} MiB (host {:.1} + accelerator {:.1}), \
                 {:.2}x the larger peak",
                samples as f64 / TARGET_SAMPLE_RATE as f64,
                subsampled_len_3x(chunk_frames) + cfg.spkcache_len + cfg.fifo_len,
                mib(footprint),
                mib(malloc),
                mib(estimated),
                mib(estimate.host_bytes),
                mib(estimate.accelerator_bytes),
                estimated as f64 / footprint.max(malloc) as f64,
            );
        }
        match previous {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }

    /// Deterministic values in [-scale, scale) + offset.
    fn random_values(len: usize, seed: u64, scale: f32, offset: f32) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
                (unit * 2.0 - 1.0) * scale + offset
            })
            .collect()
    }

    fn random_tensor(shape: &[usize], seed: u64, scale: f32, offset: f32) -> Tensor {
        let values = random_values(shape.iter().product(), seed, scale, offset);
        Tensor::from_vec(values, shape, &Device::Cpu).unwrap()
    }

    /// NaN-safe: a NaN or infinite difference counts as infinitely far.
    fn tensor_max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        assert_eq!(a.dims(), b.dims());
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        a.iter()
            .zip(&b)
            .map(|(x, y)| {
                let d = (x - y).abs();
                if d.is_finite() {
                    d
                } else {
                    f32::INFINITY
                }
            })
            .fold(0.0, f32::max)
    }

    const SYNTHETIC_D_MODEL: usize = 16;
    const SYNTHETIC_CONV_CHANNELS: usize = 4;
    const SYNTHETIC_FEATURE_BINS: usize = 16;

    /// A synthetic v2.1-shaped Conformer encoder checkpoint: dw_striding
    /// subsampling (4 channels, 16 mel bins), `layers` Conformer layers of
    /// width 16 (2 heads), FFN 32, depthwise kernel 9 with BatchNorm.
    fn conformer_encoder_tensors(layers: usize) -> HashMap<String, Tensor> {
        let d = SYNTHETIC_D_MODEL;
        let c = SYNTHETIC_CONV_CHANNELS;
        let mut tensors = HashMap::new();
        let mut seed = 0x5EED_0001_u64;
        let mut insert = |name: String, shape: &[usize], scale: f32, offset: f32| {
            seed += 0x9E37_79B9;
            tensors.insert(
                format!("encoder.{name}"),
                random_tensor(shape, seed, scale, offset),
            );
        };
        for conv in ["0", "2", "5"] {
            insert(
                format!("pre_encode.conv.{conv}.weight"),
                &[c, 1, 3, 3],
                0.5,
                0.0,
            );
            insert(format!("pre_encode.conv.{conv}.bias"), &[c], 0.1, 0.0);
        }
        for conv in ["3", "6"] {
            insert(
                format!("pre_encode.conv.{conv}.weight"),
                &[c, c, 1, 1],
                0.5,
                0.0,
            );
            insert(format!("pre_encode.conv.{conv}.bias"), &[c], 0.1, 0.0);
        }
        let out_in = c * SYNTHETIC_FEATURE_BINS / 8;
        insert("pre_encode.out.weight".into(), &[d, out_in], 0.3, 0.0);
        insert("pre_encode.out.bias".into(), &[d], 0.1, 0.0);
        for layer in 0..layers {
            let l = format!("layers.{layer}.");
            for norm in [
                "norm_feed_forward1",
                "norm_self_att",
                "norm_conv",
                "norm_feed_forward2",
                "norm_out",
            ] {
                insert(format!("{l}{norm}.weight"), &[d], 0.1, 1.0);
                insert(format!("{l}{norm}.bias"), &[d], 0.1, 0.0);
            }
            for ff in ["feed_forward1", "feed_forward2"] {
                insert(format!("{l}{ff}.linear1.weight"), &[32, d], 0.25, 0.0);
                insert(format!("{l}{ff}.linear1.bias"), &[32], 0.1, 0.0);
                insert(format!("{l}{ff}.linear2.weight"), &[d, 32], 0.18, 0.0);
                insert(format!("{l}{ff}.linear2.bias"), &[d], 0.1, 0.0);
            }
            insert(format!("{l}self_attn.pos_bias_u"), &[2, d / 2], 0.1, 0.0);
            insert(format!("{l}self_attn.pos_bias_v"), &[2, d / 2], 0.1, 0.0);
            for linear in ["linear_q", "linear_k", "linear_v", "linear_out"] {
                insert(format!("{l}self_attn.{linear}.weight"), &[d, d], 0.25, 0.0);
                insert(format!("{l}self_attn.{linear}.bias"), &[d], 0.1, 0.0);
            }
            insert(
                format!("{l}self_attn.linear_pos.weight"),
                &[d, d],
                0.25,
                0.0,
            );
            insert(
                format!("{l}conv.pointwise_conv1.weight"),
                &[2 * d, d, 1],
                0.25,
                0.0,
            );
            insert(format!("{l}conv.pointwise_conv1.bias"), &[2 * d], 0.1, 0.0);
            insert(
                format!("{l}conv.depthwise_conv.weight"),
                &[d, 1, 9],
                0.3,
                0.0,
            );
            insert(format!("{l}conv.depthwise_conv.bias"), &[d], 0.1, 0.0);
            insert(format!("{l}conv.batch_norm.weight"), &[d], 0.2, 1.0);
            insert(format!("{l}conv.batch_norm.bias"), &[d], 0.1, 0.0);
            insert(format!("{l}conv.batch_norm.running_mean"), &[d], 0.2, 0.0);
            insert(format!("{l}conv.batch_norm.running_var"), &[d], 0.4, 0.8);
            insert(
                format!("{l}conv.pointwise_conv2.weight"),
                &[d, d, 1],
                0.25,
                0.0,
            );
            insert(format!("{l}conv.pointwise_conv2.bias"), &[d], 0.1, 0.0);
        }
        tensors
    }

    fn conformer_encoder_fixture(layers: usize) -> SortformerConformerEncoder {
        let vb =
            VarBuilder::from_tensors(conformer_encoder_tensors(layers), DType::F32, &Device::Cpu);
        SortformerConformerEncoder::load(vb.pp("encoder"), true).unwrap()
    }

    /// SRT-5: the Conformer conv module (shared shift-and-accumulate op with
    /// the folded BatchNorm) matches Candle's per-channel grouped conv1d
    /// followed by the unfolded BatchNorm.
    #[test]
    fn conformer_conv_folded_depthwise_matches_grouped_conv_and_batch_norm() {
        use candle_nn::ModuleT;

        let d = SYNTHETIC_D_MODEL;
        let tensors = conformer_encoder_tensors(1);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        let vb = vb.pp("encoder.layers.0.conv");
        let counters = Arc::new(SortformerPathCounters::default());
        let conv = ConformerConv::load(vb.clone(), d, &counters).unwrap();

        let grouped = Conv1d::new(
            vb.get((d, 1, 9), "depthwise_conv.weight").unwrap(),
            Some(vb.get(d, "depthwise_conv.bias").unwrap()),
            Conv1dConfig {
                padding: 4,
                groups: d,
                ..Default::default()
            },
        );
        let norm = batch_norm(d, 1e-5, vb.pp("batch_norm")).unwrap();
        for (batch, time, seed) in [(1, 37, 11), (2, 9, 12), (1, 1, 13)] {
            let x = random_tensor(&[batch, time, d], seed, 1.0, 0.0);
            let h = conv
                .pointwise_conv1
                .forward(&x.transpose(1, 2).unwrap())
                .unwrap();
            let h = h
                .i((.., ..d, ..))
                .unwrap()
                .broadcast_mul(&ops::sigmoid(&h.i((.., d.., ..)).unwrap()).unwrap())
                .unwrap();
            let h = grouped.forward(&h).unwrap();
            let h = norm.forward_t(&h, false).unwrap();
            let expected = conv
                .pointwise_conv2
                .forward(&swish(&h).unwrap())
                .unwrap()
                .transpose(1, 2)
                .unwrap();
            let actual = conv.forward(&x).unwrap();
            let diff = tensor_max_abs_diff(&expected, &actual);
            assert!(diff <= 1e-5, "batch {batch} time {time}: max |diff| {diff}");
        }
        assert_eq!(counters.conformer_depthwise_conv1d(), 3);
    }

    /// SRT-5 engagement: every Conformer layer's depthwise conv runs through
    /// the shared op, once per layer per streaming chunk.
    #[test]
    fn conformer_encoder_runs_each_depthwise_conv_through_the_shared_op() {
        let layers = 3;
        let encoder = conformer_encoder_fixture(layers);
        let shared_before = crate::kernels::depthwise_conv::depthwise_conv1d_calls();
        let chunks = 4;
        for chunk in 0..chunks {
            let mel_frames = 40 + 8 * chunk;
            let features_t = random_tensor(
                &[1, mel_frames, SYNTHETIC_FEATURE_BINS],
                100 + chunk as u64,
                1.0,
                0.0,
            );
            let (pre, len) = encoder.pre_encode(&features_t, mel_frames).unwrap();
            let (encoded, _) = encoder.forward_pre_encoded(&pre, len).unwrap();
            assert_eq!(encoded.dims3().unwrap(), (1, len, SYNTHETIC_D_MODEL));
            assert_eq!(
                encoder.counters.conformer_depthwise_conv1d(),
                ((chunk + 1) * layers) as u64
            );
            assert_eq!(
                encoder.counters.subsampling_depthwise_conv2d(),
                2 * (chunk + 1) as u64
            );
        }
        // Other tests may run concurrently; the shared counter only grows.
        assert!(
            crate::kernels::depthwise_conv::depthwise_conv1d_calls()
                >= shared_before + (chunks * layers) as u64
        );
        let diagnostics = SortformerAcousticEncoder::Conformer(encoder).kernel_diagnostics();
        assert_eq!(
            diagnostics["conformer_depthwise_conv1d_calls"],
            serde_json::json!(chunks * layers)
        );
        assert_eq!(
            diagnostics["subsampling_depthwise_conv2d_calls"],
            serde_json::json!(2 * chunks)
        );
    }

    /// SRT-5: the dw_striding subsampling (strided-tap depthwise conv2d)
    /// matches the same graph through Candle's per-channel grouped conv2d,
    /// on odd chunk lengths.
    #[test]
    fn subsampling_depthwise_conv2d_matches_grouped_conv2d() {
        let encoder = conformer_encoder_fixture(1);
        let sub = &encoder.pre_encode;
        for (mel_frames, seed) in [(40, 21), (37, 22), (9, 23), (1, 24)] {
            let features_t =
                random_tensor(&[1, mel_frames, SYNTHETIC_FEATURE_BINS], seed, 1.0, 0.0);
            let mut x = features_t.unsqueeze(1).unwrap();
            x = sub.conv0.forward(&x).unwrap().relu().unwrap();
            x = sub.conv2.forward(&x).unwrap();
            x = sub.conv3.forward(&x).unwrap().relu().unwrap();
            x = sub.conv5.forward(&x).unwrap();
            x = sub.conv6.forward(&x).unwrap().relu().unwrap();
            let (b, c, t, f) = x.dims4().unwrap();
            let expected = x
                .transpose(1, 2)
                .unwrap()
                .reshape((b, t, c * f))
                .unwrap()
                .apply(&sub.out)
                .unwrap();
            let (actual, len) = encoder.pre_encode(&features_t, mel_frames).unwrap();
            assert_eq!(len, subsampled_len_3x(mel_frames));
            let diff = tensor_max_abs_diff(&expected, &actual);
            assert!(diff <= 1e-5, "{mel_frames} frames: max |diff| {diff}");
        }
    }

    #[test]
    fn conformer_conv_norm_type_fails_closed_unless_batch_norm() {
        let cfg = |norm: Option<&str>| SortformerEncoderConfig {
            conv_norm_type: norm.map(str::to_string),
            ..SortformerEncoderConfig::default()
        };
        assert!(validate_conv_norm_type(None).is_ok());
        assert!(validate_conv_norm_type(Some(&cfg(None))).is_ok());
        assert!(validate_conv_norm_type(Some(&cfg(Some("batch_norm")))).is_ok());
        for other in ["layer_norm", "group_norm", "fused_batch_norm"] {
            assert!(validate_conv_norm_type(Some(&cfg(Some(other)))).is_err());
        }
    }

    const V21_MODEL_CONFIG: &str =
        include_str!("../../../../../tests/fixtures/sortformer_config/v21_model_config.yaml");
    const NEMOTRON3_MODEL_CONFIG: &str =
        include_str!("../../../../../tests/fixtures/sortformer_config/nemotron3_model_config.yaml");

    /// The yaml with the value at `path` replaced (`Some`) or removed
    /// (`None`), parsed as a Sortformer config.
    fn edited_config(
        yaml: &str,
        path: &[&str],
        value: Option<serde_yaml::Value>,
    ) -> SortformerModelConfig {
        let mut root: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        let (last, parents) = path.split_last().unwrap();
        let mut node = &mut root;
        for key in parents {
            node = node.get_mut(*key).unwrap();
        }
        let map = node.as_mapping_mut().unwrap();
        match value {
            Some(value) => {
                map.insert((*last).into(), value);
            }
            None => {
                map.remove(*last);
            }
        }
        serde_yaml::from_value(root).unwrap()
    }

    fn yaml_value(text: &str) -> serde_yaml::Value {
        serde_yaml::from_str(text).unwrap()
    }

    /// SRT-8 / NDZ-4: both published checkpoint yamls resolve to the graph
    /// izwi runs, with head counts and the RoPE base read from the yaml.
    #[test]
    fn published_sortformer_yamls_resolve_the_served_graphs() {
        let v21: SortformerModelConfig = serde_yaml::from_str(V21_MODEL_CONFIG).unwrap();
        assert_eq!(
            resolve_graph_spec(&v21).unwrap(),
            SortformerGraphSpec::Conformer {
                heads: 8,
                xscaling: true,
                transformer_heads: 8,
            }
        );
        let nemotron: SortformerModelConfig = serde_yaml::from_str(NEMOTRON3_MODEL_CONFIG).unwrap();
        assert_eq!(
            resolve_graph_spec(&nemotron).unwrap(),
            SortformerGraphSpec::FeatureStackingRope {
                heads: 8,
                stacking_factor: 8,
                rope_base: 10_000.0,
            }
        );
    }

    /// SRT-8 / NDZ-4: a graph-defining field the graph does not implement
    /// fails the load and names the field, whether it is set or left to a
    /// NeMo default izwi does not serve.
    #[test]
    fn unserved_sortformer_graph_fields_fail_closed() {
        let cases: &[(&str, &[&str], Option<&str>, &str)] = &[
            (
                V21_MODEL_CONFIG,
                &["async_streaming"],
                Some("true"),
                "async_streaming",
            ),
            (
                V21_MODEL_CONFIG,
                &["encoder", "_target_"],
                Some("x.SqueezeformerEncoder"),
                "encoder target",
            ),
            (
                V21_MODEL_CONFIG,
                &["encoder", "subsampling"],
                Some("striding"),
                "encoder.subsampling",
            ),
            (
                V21_MODEL_CONFIG,
                &["encoder", "subsampling"],
                None,
                "encoder.subsampling",
            ),
            (
                V21_MODEL_CONFIG,
                &["encoder", "subsampling_factor"],
                Some("4"),
                "encoder.subsampling_factor",
            ),
            (
                V21_MODEL_CONFIG,
                &["encoder", "causal_downsampling"],
                Some("true"),
                "encoder.causal_downsampling",
            ),
            (
                V21_MODEL_CONFIG,
                &["encoder", "self_attention_model"],
                Some("abs_pos"),
                "encoder.self_attention_model",
            ),
            (
                V21_MODEL_CONFIG,
                &["encoder", "att_context_size"],
                Some("[70, 1]"),
                "encoder.att_context_size",
            ),
            (
                V21_MODEL_CONFIG,
                &["encoder", "att_context_size"],
                Some("[[-1, -1], [70, 1]]"),
                "encoder.att_context_size",
            ),
            (
                V21_MODEL_CONFIG,
                &["encoder", "conv_norm_type"],
                Some("layer_norm"),
                "conv_norm_type",
            ),
            (
                V21_MODEL_CONFIG,
                &["encoder", "conv_context_size"],
                Some("causal"),
                "encoder.conv_context_size",
            ),
            (
                V21_MODEL_CONFIG,
                &["encoder", "untie_biases"],
                Some("false"),
                "encoder.untie_biases",
            ),
            (
                V21_MODEL_CONFIG,
                &["encoder", "use_bias"],
                Some("false"),
                "encoder.use_bias",
            ),
            (
                V21_MODEL_CONFIG,
                &["encoder", "global_tokens"],
                Some("4"),
                "encoder.global_tokens",
            ),
            (
                V21_MODEL_CONFIG,
                &["high_resolution"],
                Some("true"),
                "high_resolution",
            ),
            (
                V21_MODEL_CONFIG,
                &["output_subsampling_factor"],
                Some("1"),
                "output_subsampling_factor",
            ),
            (
                V21_MODEL_CONFIG,
                &["sortformer_modules", "use_learnable_sil_emb"],
                Some("true"),
                "use_learnable_sil_emb",
            ),
            (
                V21_MODEL_CONFIG,
                &["transformer_encoder"],
                None,
                "transformer_encoder",
            ),
            (
                V21_MODEL_CONFIG,
                &["transformer_encoder", "pre_ln"],
                Some("true"),
                "transformer_encoder.pre_ln",
            ),
            (
                V21_MODEL_CONFIG,
                &["transformer_encoder", "hidden_act"],
                Some("gelu"),
                "transformer_encoder.hidden_act",
            ),
            (
                V21_MODEL_CONFIG,
                &["transformer_encoder", "mask_future"],
                Some("true"),
                "transformer_encoder.mask_future",
            ),
            (
                NEMOTRON3_MODEL_CONFIG,
                &["encoder", "subsampling"],
                Some("dw_striding"),
                "encoder.subsampling",
            ),
            (
                NEMOTRON3_MODEL_CONFIG,
                &["encoder", "self_attention_model"],
                Some("rel_pos"),
                "encoder.self_attention_model",
            ),
            (
                NEMOTRON3_MODEL_CONFIG,
                &["encoder", "self_attention_model"],
                None,
                "encoder.self_attention_model",
            ),
            (
                NEMOTRON3_MODEL_CONFIG,
                &["encoder", "xscaling"],
                Some("true"),
                "encoder.xscaling",
            ),
            (
                NEMOTRON3_MODEL_CONFIG,
                &["encoder", "qkv_bias"],
                Some("true"),
                "encoder.qkv_bias",
            ),
            (
                NEMOTRON3_MODEL_CONFIG,
                &["encoder", "qk_norm"],
                Some("true"),
                "encoder.qk_norm",
            ),
            (
                NEMOTRON3_MODEL_CONFIG,
                &["encoder", "pre_block_norm"],
                Some("false"),
                "encoder.pre_block_norm",
            ),
            (
                NEMOTRON3_MODEL_CONFIG,
                &["encoder", "attn_mode"],
                Some("causal"),
                "encoder.attn_mode",
            ),
            (
                NEMOTRON3_MODEL_CONFIG,
                &["encoder", "rotary_fraction"],
                Some("0.5"),
                "encoder.rotary_fraction",
            ),
            (
                NEMOTRON3_MODEL_CONFIG,
                &["encoder", "rope_base"],
                Some("0.0"),
                "encoder.rope_base",
            ),
            (
                NEMOTRON3_MODEL_CONFIG,
                &["encoder", "causal_tail_len"],
                Some("16"),
                "encoder.causal_tail_len",
            ),
            (
                NEMOTRON3_MODEL_CONFIG,
                &["high_resolution"],
                Some("false"),
                "high_resolution",
            ),
            (
                NEMOTRON3_MODEL_CONFIG,
                &["output_subsampling_factor"],
                Some("8"),
                "output_subsampling_factor",
            ),
            (
                NEMOTRON3_MODEL_CONFIG,
                &["output_subsampling_factor"],
                None,
                "output_subsampling_factor",
            ),
            (
                NEMOTRON3_MODEL_CONFIG,
                &["sortformer_modules", "use_learnable_sil_emb"],
                Some("false"),
                "use_learnable_sil_emb",
            ),
            (
                NEMOTRON3_MODEL_CONFIG,
                &["transformer_encoder"],
                Some("{num_attention_heads: 8}"),
                "transformer_encoder",
            ),
        ];
        for (yaml, path, value, field) in cases {
            let config = edited_config(yaml, path, value.map(yaml_value));
            let err = resolve_graph_spec(&config)
                .expect_err(&format!("{path:?} = {value:?} must fail closed"))
                .to_string();
            assert!(err.contains(field), "{path:?} = {value:?}: {err}");
        }
    }

    /// NDZ-4: the head counts and the RoPE base come from the yaml (NeMo
    /// defaults when absent), and a head count off the production topology
    /// is refused by its pin rather than loaded with eight heads.
    #[test]
    fn sortformer_head_counts_and_rope_base_come_from_the_yaml() {
        let spec = |yaml, path: &[&str], value: Option<&str>| {
            resolve_graph_spec(&edited_config(yaml, path, value.map(yaml_value))).unwrap()
        };
        assert!(matches!(
            spec(V21_MODEL_CONFIG, &["encoder", "n_heads"], Some("4")),
            SortformerGraphSpec::Conformer { heads: 4, .. }
        ));
        assert!(matches!(
            spec(V21_MODEL_CONFIG, &["encoder", "n_heads"], None),
            SortformerGraphSpec::Conformer { heads: 4, .. }
        ));
        assert!(matches!(
            spec(
                V21_MODEL_CONFIG,
                &["transformer_encoder", "num_attention_heads"],
                Some("4")
            ),
            SortformerGraphSpec::Conformer {
                transformer_heads: 4,
                ..
            }
        ));
        assert!(matches!(
            spec(
                V21_MODEL_CONFIG,
                &["transformer_encoder", "num_attention_heads"],
                None
            ),
            SortformerGraphSpec::Conformer {
                transformer_heads: 1,
                ..
            }
        ));
        assert!(matches!(
            spec(NEMOTRON3_MODEL_CONFIG, &["encoder", "n_heads"], Some("16")),
            SortformerGraphSpec::FeatureStackingRope { heads: 16, .. }
        ));
        assert!(matches!(
            spec(
                NEMOTRON3_MODEL_CONFIG,
                &["encoder", "rope_base"],
                Some("500000.0")
            ),
            SortformerGraphSpec::FeatureStackingRope {
                rope_base: 500_000.0,
                ..
            }
        ));

        let cfg = production_workspace_streaming_config();
        let mut topology = SortformerWorkspaceTopology::v21_production();
        assert!(topology.validate_production(cfg).is_ok());
        topology.expander = SortformerFrameExpanderTopology::SortformerTransformer {
            layers: PRODUCTION_TRANSFORMER_LAYERS,
            d_model: PRODUCTION_TRANSFORMER_D_MODEL,
            inner_dim: PRODUCTION_TRANSFORMER_INNER_DIM,
            heads: 4,
        };
        assert!(topology.validate_production(cfg).is_err());

        // The RoPE frequencies follow the yaml's base.
        let encoder = rope_encoder_fixture_with_base(100.0).unwrap();
        let head_dim = encoder.head_dim as f32;
        assert!((encoder.rope_inv_freq[1] - 100f32.powf(-2.0 / head_dim)).abs() < 1e-7);
    }

    /// A synthetic v2.1-shaped streaming model: the synthetic Conformer
    /// encoder, a projection to 8, one 8-head transformer layer and the
    /// 4-speaker head, streaming 6-frame chunks through an 8-row cache and
    /// a 4-row FIFO so the cache compresses every few chunks. Every frame
    /// counts as silent, so the running silence mean is exercised.
    fn synthetic_streaming_model(layers: usize) -> SortformerInferenceModel {
        let mut tensors = conformer_encoder_tensors(layers);
        let (d, h, inner) = (SYNTHETIC_D_MODEL, 8, 16);
        let mut seed = 0xFEED_0001_u64;
        let mut insert = |name: &str, shape: &[usize], scale: f32, offset: f32| {
            seed += 0x9E37_79B9;
            tensors.insert(name.to_string(), random_tensor(shape, seed, scale, offset));
        };
        insert("sortformer_modules.encoder_proj.weight", &[h, d], 0.3, 0.0);
        insert("sortformer_modules.encoder_proj.bias", &[h], 0.1, 0.0);
        let l = "transformer_encoder.layers.0.";
        for norm in ["layer_norm_1", "layer_norm_2"] {
            insert(&format!("{l}{norm}.weight"), &[h], 0.1, 1.0);
            insert(&format!("{l}{norm}.bias"), &[h], 0.1, 0.0);
        }
        for linear in ["query_net", "key_net", "value_net", "out_projection"] {
            insert(
                &format!("{l}first_sub_layer.{linear}.weight"),
                &[h, h],
                0.3,
                0.0,
            );
            insert(&format!("{l}first_sub_layer.{linear}.bias"), &[h], 0.1, 0.0);
        }
        insert(
            &format!("{l}second_sub_layer.dense_in.weight"),
            &[inner, h],
            0.3,
            0.0,
        );
        insert(
            &format!("{l}second_sub_layer.dense_in.bias"),
            &[inner],
            0.1,
            0.0,
        );
        insert(
            &format!("{l}second_sub_layer.dense_out.weight"),
            &[h, inner],
            0.3,
            0.0,
        );
        insert(
            &format!("{l}second_sub_layer.dense_out.bias"),
            &[h],
            0.1,
            0.0,
        );
        insert(
            "sortformer_modules.first_hidden_to_hidden.weight",
            &[h, h],
            0.4,
            0.0,
        );
        insert(
            "sortformer_modules.first_hidden_to_hidden.bias",
            &[h],
            0.1,
            0.0,
        );
        insert(
            "sortformer_modules.single_hidden_to_spks.weight",
            &[4, h],
            1.5,
            0.0,
        );
        insert(
            "sortformer_modules.single_hidden_to_spks.bias",
            &[4],
            0.5,
            0.0,
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        let preprocessor = SortformerPreprocessor::load(
            &vb,
            SortformerPreprocessorConfig {
                sample_rate: Some(TARGET_SAMPLE_RATE),
                window_size: Some(0.025),
                window_stride: Some(0.01),
                features: Some(SYNTHETIC_FEATURE_BINS),
                n_fft: Some(512),
                normalize: Some("NA".to_string()),
            },
        )
        .unwrap();
        let encoder = SortformerConformerEncoder::load(vb.pp("encoder"), true).unwrap();
        let streaming = SortformerStreamingConfig {
            fc_d_model: d,
            spkcache_len: 8,
            fifo_len: 4,
            chunk_len: 6,
            spkcache_update_period: 3,
            chunk_left_context: 1,
            chunk_right_context: 2,
            spkcache_sil_frames_per_spk: 1,
            sil_threshold: 10.0,
            ..production_workspace_streaming_config()
        }
        .validate()
        .unwrap();
        SortformerInferenceModel {
            device: Device::Cpu,
            separate_device_memory: false,
            preprocessor,
            encoder: SortformerAcousticEncoder::Conformer(encoder),
            encoder_proj: mlx::load_linear(d, h, vb.pp("sortformer_modules.encoder_proj")).unwrap(),
            expander: SortformerFrameExpander::SortformerTransformer(
                SortformerTransformerEncoder::load(vb.pp("transformer_encoder"), 8).unwrap(),
            ),
            head: SortformerSpeakerHead::load(vb.pp("sortformer_modules"), 4).unwrap(),
            num_speakers: 4,
            streaming: Some(streaming),
            streaming_profile: SortformerStreamingProfile::Model,
            silence_embedding: None,
            streaming_counters: SortformerStreamingCounters::default(),
        }
    }

    /// SRT-7: the streaming loop keeps its state on the model device. Per
    /// chunk it reads the probabilities once (plus the silent FIFO rows for
    /// the running silence mean), reuses the relative-position table of a
    /// repeated composite length, and commits the physical invocation state
    /// once at the end instead of after every chunk. The physical and plain
    /// paths give bit-identical probabilities.
    #[test]
    fn streaming_state_stays_on_device_with_bounded_host_reads() {
        let model = synthetic_streaming_model(2);
        let cfg = model.streaming.unwrap();
        let samples = random_values(16_000 * 6, 0xA0D10, 0.3, 0.0);
        let feature_frames = model.preprocessor.feature_frame_count(samples.len());
        let planned = plan_streaming_feature_chunks(feature_frames, cfg).len() as u64;
        assert!(planned >= 8, "{planned} chunks");

        let plain = model
            .infer_speaker_probabilities_streaming(&samples, feature_frames, cfg, None)
            .unwrap();
        let path = model.streaming_path_diagnostics();
        let chunks = path["chunks"].as_u64().unwrap();
        let host_reads = path["host_reads"].as_u64().unwrap();
        assert_eq!(chunks, planned);
        assert!(
            host_reads > chunks && host_reads <= 2 * chunks,
            "{host_reads} host reads over {chunks} chunks"
        );
        assert_eq!(path["physical_state_commits"], serde_json::json!(0));
        let SortformerAcousticEncoder::Conformer(encoder) = &model.encoder else {
            unreachable!("synthetic model is a Conformer");
        };
        let tables = encoder.counters.rel_pos_tables_built();
        assert!(
            tables >= 1 && tables < chunks / 2,
            "{tables} tables, {chunks} chunks"
        );
        assert_eq!(encoder.counters.conformer_depthwise_conv1d(), 2 * chunks);

        let mut lease = streaming_state_commit_lease(cfg, BackendKind::Cpu, Device::Cpu);
        let physical = model
            .infer_speaker_probabilities_streaming(&samples, feature_frames, cfg, Some(&mut lease))
            .unwrap();
        assert_eq!(lease.arena().unwrap().absolute_cursor(), 1);
        assert_eq!(
            model.streaming_path_diagnostics()["physical_state_commits"],
            serde_json::json!(1)
        );
        assert_eq!(physical.len(), plain.len());
        for (a, b) in physical.iter().flatten().zip(plain.iter().flatten()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    /// PIPE-8: a stream run in bounded steps carries its state between
    /// steps unchanged, so every split gives the one-pass probabilities and
    /// the same committed state, bit for bit, and counts one step per stage.
    #[test]
    fn staged_stream_matches_one_pass_bit_for_bit() {
        let model = synthetic_streaming_model(2);
        let cfg = model.streaming.unwrap();
        let samples = random_values(16_000 * 6, 0x57A6E, 0.3, 0.0);
        let feature_frames = model.preprocessor.feature_frame_count(samples.len());
        let planned = plan_streaming_feature_chunks(feature_frames, cfg).len();
        assert!(planned >= 8, "{planned} chunks");
        let mut one_pass_lease = streaming_state_commit_lease(cfg, BackendKind::Cpu, Device::Cpu);
        let one_pass = model
            .infer_speaker_probabilities_streaming(
                &samples,
                feature_frames,
                cfg,
                Some(&mut one_pass_lease),
            )
            .unwrap();
        let committed = |lease: &InvocationTensorLease| {
            lease
                .read_snapshot()
                .unwrap()
                .components
                .into_iter()
                .map(|slice| {
                    let values = slice
                        .tensor
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap();
                    values.into_iter().map(f32::to_bits).collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };

        for chunks_per_step in [1, 2, 3, planned] {
            let steps_before = model.streaming_path_diagnostics()["steps"]
                .as_u64()
                .unwrap();
            let mut stream =
                model.begin_stream(samples.len(), feature_frames, cfg, chunks_per_step);
            let expected_steps = planned.div_ceil(chunks_per_step);
            assert_eq!(stream.steps_remaining(), expected_steps);
            while !stream.is_finished() {
                model.advance_stream(&mut stream, &samples).unwrap();
            }
            assert_eq!(
                model.streaming_path_diagnostics()["steps"]
                    .as_u64()
                    .unwrap()
                    - steps_before,
                expected_steps as u64
            );
            let mut lease = streaming_state_commit_lease(cfg, BackendKind::Cpu, Device::Cpu);
            let staged = model.finish_stream(stream, Some(&mut lease)).unwrap();
            assert_eq!(
                staged.len(),
                one_pass.len(),
                "{chunks_per_step} chunks/step"
            );
            for (a, b) in staged.iter().flatten().zip(one_pass.iter().flatten()) {
                assert_eq!(a.to_bits(), b.to_bits(), "{chunks_per_step} chunks/step");
            }
            assert_eq!(committed(&lease), committed(&one_pass_lease));
        }

        // A stream cannot finish early or resume over other audio.
        let mut stream = model.begin_stream(samples.len(), feature_frames, cfg, 1);
        assert!(model.advance_stream(&mut stream, &samples[1..]).is_err());
        model.advance_stream(&mut stream, &samples).unwrap();
        assert!(model.finish_stream(stream, None).is_err());
    }

    /// A step covers about 90 s of 16 kHz audio in whole chunks: three
    /// 27.2 s chunks of the v2.1 card profile, never zero.
    #[test]
    fn stream_steps_cover_bounded_audio() {
        let cfg = production_workspace_streaming_config();
        assert_eq!(chunks_per_stream_step(cfg, PRODUCTION_HOP_LENGTH, 90.0), 3);
        assert_eq!(chunks_per_stream_step(cfg, PRODUCTION_HOP_LENGTH, 1.0), 1);
        assert_eq!(chunks_per_stream_step(cfg, PRODUCTION_HOP_LENGTH, 0.0), 1);
    }
}
